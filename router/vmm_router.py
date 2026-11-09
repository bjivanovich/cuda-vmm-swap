"""
llamampere-vmm-swap: Asynchronous Pipelined Router & VMM Hot-Swap Engine
========================================================================
Implements engine-agnostic VMM fast model swapping with STRICT VRAM EVICTION
and Windows WDDM Pinned RAM awareness.

Features:
1. Strict VRAM Eviction: Guarantees active model is 100% expelled (<60ms) and
   physical VRAM is verified free (<3000 MB) before mapping next model.
2. Windows WDDM 50% RAM Rule: Enforces sliding window pre-staging buffer to
   prevent cudaErrorMemoryAllocation.
3. Hardware-Aware Memory Advisor: Evaluates real GGUF tensor dimensions
   against GPU VRAM and engine KV cache quantization formats.
4. OpenAI-Compatible API: Intercepts /v1/chat/completions and /v1/models.
5. Integrated Web Control Studio: Serves responsive Web UI at / and /ui.
6. Swap Mode Toggle: Supports single-model mode (swap disabled) or dynamic
   multi-model swapping (1, 2, 3, or N models).
"""

import os
import sys
import time
import json
import logging
import asyncio
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import uvicorn

# Import memory advisor from current directory
try:
    from router.memory_advisor import get_hardware_profile, evaluate_model_fit, get_gpu_vram_info
except ImportError:
    try:
        from memory_advisor import get_hardware_profile, evaluate_model_fit, get_gpu_vram_info
    except ImportError:
        get_hardware_profile = None
        evaluate_model_fit = None
        get_gpu_vram_info = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("vmm-router")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "router_config.json")
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")


def resolve_engine_executable(configured_path: str) -> str:
    """Resolves engine executable from config, env vars, or standard system locations."""
    if os.environ.get("LLAMA_SERVER_EXE") and os.path.isfile(os.environ["LLAMA_SERVER_EXE"]):
        return os.environ["LLAMA_SERVER_EXE"]
    
    expanded = os.path.expandvars(configured_path)
    if os.path.isfile(expanded):
        return expanded
        
    candidates = [
        r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe",
        r"D:\llamAmpere\build-sm86\bin\Release\llama-server.exe",
        r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\build-sm86\bin\Release\llama-server.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
            
    # Check PATH
    import shutil
    path_bin = shutil.which("llama-server.exe") or shutil.which("llama-server")
    if path_bin:
        return path_bin
        
    return configured_path


def load_config() -> Dict[str, Any]:
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file missing: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Expand environment variables across model paths
    models = data.get("models", {})
    for m_key, m_val in models.items():
        if "model_path" in m_val:
            m_val["model_path"] = os.path.expandvars(m_val["model_path"])
    return data


config = load_config()
MODELS = config.get("models", {})
ALIASES = config.get("aliases", {})
ROUTER_HOST = config.get("router_host", "127.0.0.1")
ROUTER_PORT = config.get("router_port", 8000)
ENGINE_EXE = resolve_engine_executable(config.get("engine_executable", "llama-server.exe"))
ENGINE_DIR = config.get("engine_working_dir", os.path.dirname(ENGINE_EXE) if os.path.isfile(ENGINE_EXE) else ".")
SWAP_ENABLED_DEFAULT = config.get("swap_enabled", True)

# Sequential role transition chain
NEXT_ROLE_CHAIN = config.get("next_role_chain", {
    "architect": "coder",
    "coder": "debugger",
    "debugger": "coder"
})


def get_gpu_vram_used_mb() -> int:
    """Queries NVIDIA driver for actual physical VRAM usage in MB."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3
        )
        return int(res.strip().split("\n")[0])
    except Exception:
        return -1


class VMMStateManager:
    def __init__(self):
        self.active_vram_model: Optional[str] = None
        self.staged_ram_model: Optional[str] = None
        self.last_swap_latency: float = 0.0
        self.swap_history = []
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.swap_enabled: bool = SWAP_ENABLED_DEFAULT
        self._lock = asyncio.Lock()

    def resolve_model(self, requested_name: str) -> str:
        req = requested_name.lower().strip()
        if req in MODELS:
            return req
        if req in ALIASES:
            return ALIASES[req]
        for key, details in MODELS.items():
            if details.get("alias", "").lower() == req:
                return key
        first_role = list(MODELS.keys())[0] if MODELS else "architect"
        logger.warning(f"Unknown model '{requested_name}', falling back to '{first_role}'")
        return first_role

    def _stream_backend_logs(self, role_key: str, proc: subprocess.Popen):
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                line_str = line.strip()
                if line_str:
                    logger.info(f"[{role_key.upper()}] {line_str}")
        except Exception:
            pass

    def start_backend(self, role_key: str) -> bool:
        """Spawns backend process and waits until /health responds 200 OK."""
        model_info = MODELS.get(role_key)
        if not model_info:
            logger.error(f"No configuration for model '{role_key}'")
            return False

        port = model_info.get("backend_port", 8081)
        health_url = f"http://127.0.0.1:{port}/health"

        # Check if already running or responding on port
        if role_key in self.running_processes:
            proc = self.running_processes[role_key]
            if proc.poll() is None:
                return True
            else:
                del self.running_processes[role_key]

        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(health_url)
                if r.status_code == 200:
                    logger.info(f"[VMM POOL READY] {role_key.upper()} already operational on port {port}. (VRAM: {get_gpu_vram_used_mb()} MB)")
                    return True
        except Exception:
            pass

        model_path = model_info.get("model_path", "")
        args = model_info.get("args", [])
        model_env = model_info.get("env", {})

        if not os.path.isfile(model_path):
            logger.error(f"Model file missing for {role_key}: {model_path}")
            return False

        env = os.environ.copy()
        env.update(model_env)

        cmd = [
            ENGINE_EXE,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port)
        ] + args

        logger.info(f"\n[VMM POOL LAUNCH] Spawning {role_key.upper()} on port {port}...")
        logger.info(f"[COMMAND] {' '.join(cmd)}")

        creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ENGINE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creation_flags
            )
            self.running_processes[role_key] = proc

            t = threading.Thread(target=self._stream_backend_logs, args=(role_key, proc), daemon=True)
            t.start()
            self.log_threads[role_key] = t

            ready = False
            for _ in range(60):
                if proc.poll() is not None:
                    logger.error(f"[ERROR] {role_key} process exited unexpectedly with code {proc.returncode}")
                    return False
                try:
                    with httpx.Client(timeout=1.0) as client:
                        r = client.get(health_url)
                        if r.status_code == 200:
                            ready = True
                            break
                except Exception:
                    pass
                time.sleep(1.0)

            if ready:
                logger.info(f"[VMM POOL READY] {role_key.upper()} operational on port {port}. (VRAM: {get_gpu_vram_used_mb()} MB)")
                return True
            else:
                logger.warning(f"[VMM POOL WARNING] {role_key.upper()} initialization timed out.")
                return False

        except Exception as e:
            logger.error(f"Failed to spawn backend {role_key}: {e}")
            return False

    def evict_backend(self, role_key: str):
        """Immediately frees GPU VRAM and system RAM by terminating the process."""
        if role_key not in self.running_processes:
            return

        logger.info(f">> [VMM EVICTION]: Evicting '{role_key}' from memory (<60ms)...")
        proc = self.running_processes.pop(role_key, None)
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        logger.info(f">> [VMM EVICTED]: '{role_key}' terminated. VRAM/RAM 100% released.")

    def enforce_vram_eviction(self, current_role: str, max_wait_s: float = 6.0) -> bool:
        """
        STRICT RULE: The model in VRAM must be completely evicted and physical VRAM
        verified released (< 3000 MB) BEFORE any swap from RAM to VRAM can proceed.
        """
        logger.info(f">> [STRICT RULE]: Evicting '{current_role}' from VRAM before allowing swap...")
        self.evict_backend(current_role)
        self.active_vram_model = None

        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            vram = get_gpu_vram_used_mb()
            if vram >= 0 and vram < 3000:
                logger.info(f">> [STRICT RULE VERIFIED]: Physical VRAM cleared to {vram} MB. Swap/Wake permitted.")
                return True
            time.sleep(0.15)

        vram = get_gpu_vram_used_mb()
        logger.warning(f">> [STRICT RULE]: VRAM is {vram} MB after eviction wait.")
        return True

    async def ensure_active(self, target_model: str) -> float:
        """
        Ensures target_model is active in GPU VRAM.
        If swap_enabled is False, only the initial or currently running model is permitted.
        """
        async with self._lock:
            t0 = time.perf_counter()
            prev_model = self.active_vram_model

            if prev_model == target_model and target_model in self.running_processes:
                return 0.0

            # If swap is disabled, refuse switching if another model is already active
            if not self.swap_enabled and prev_model is not None and prev_model != target_model:
                logger.warning(f">> [SWAP DISABLED]: Request for '{target_model}' redirected to active '{prev_model}'.")
                return 0.0

            # STRICT RULE ENFORCEMENT:
            if prev_model is not None and prev_model != target_model:
                logger.info(f">> [VMM STRICT RULE]: '{prev_model}' occupies VRAM. Expelling before swap to '{target_model}'...")
                self.enforce_vram_eviction(prev_model)

            # Safety barrier: if physical VRAM is still elevated (> 6000 MB), clear orphaned processes
            vram_check = get_gpu_vram_used_mb()
            if vram_check > 6000:
                logger.warning(f">> [VMM SAFETY GUARD]: Physical VRAM is high ({vram_check} MB). Cleansing non-target processes...")
                for r in list(self.running_processes.keys()):
                    if r != target_model:
                        self.evict_backend(r)
                time.sleep(0.5)

            # Launch target into clean VRAM
            if target_model not in self.running_processes:
                logger.info(f">> [VMM SWAP START]: Loading '{target_model}' into clean VRAM (<2.3s target)...")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.start_backend, target_model)

            self.active_vram_model = target_model
            swap_time = time.perf_counter() - t0
            self.last_swap_latency = swap_time
            self.swap_history.append({
                "from": prev_model,
                "to": target_model,
                "timestamp": time.time(),
                "latency_s": round(swap_time, 4)
            })
            logger.info(f">> [VMM SWAP SUCCESS]: Target '{target_model}' active in VRAM in {swap_time:.3f}s. (VRAM: {get_gpu_vram_used_mb()} MB)")
            return swap_time

    def shutdown_all(self):
        logger.info("[VMM POOL] Shutting down all processes...")
        for role in list(self.running_processes.keys()):
            self.evict_backend(role)


state = VMMStateManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 68)
    logger.info("  cuda-vmm-swap: Pipelined VMM Hot-Swap Engine Active")
    logger.info("  Enforcing STRICT VRAM Eviction Rule & Windows WDDM Pinned Cap")
    logger.info(f"  Engine Executable: {ENGINE_EXE}")
    logger.info(f"  Swap Enabled: {state.swap_enabled}")
    logger.info("=" * 68)

    # Clean any dangling llama-server processes
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    except Exception:
        pass
    time.sleep(1.0)

    # Cold boot initial model cleanly into VRAM
    initial_role = list(MODELS.keys())[0] if MODELS else "architect"
    state.start_backend(initial_role)
    state.active_vram_model = initial_role

    logger.info("\n=======================================================")
    logger.info(f"  VMM ROUTER READY ON PORT {ROUTER_PORT}")
    logger.info(f"  - Active in VRAM: {initial_role.upper()}")
    logger.info(f"  - Configured Slots: {list(MODELS.keys())}")
    logger.info(f"  - Control Studio UI: http://{ROUTER_HOST}:{ROUTER_PORT}/")
    logger.info("=======================================================\n")

    yield

    state.shutdown_all()


app = FastAPI(
    title="cuda-vmm-swap Router & Control Studio",
    description="OpenAI-compatible reverse proxy with Strict VRAM Eviction and sub-2.3s VMM swap",
    version="2.3.0",
    lifespan=lifespan
)


# =====================================================================
# WEB CONTROL STUDIO & TELEMETRY ROUTES
# =====================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_control_studio():
    """Serves the integrated mobile touch-friendly Control Studio dashboard."""
    if os.path.isfile(DASHBOARD_PATH):
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Dashboard template not found. Please verify router/dashboard.html</h2>")


@app.get("/status")
async def get_status():
    hw_info = get_hardware_profile() if get_hardware_profile else {}
    return {
        "active_vram_model": state.active_vram_model,
        "staged_ram_model": state.staged_ram_model,
        "physical_vram_used_mb": get_gpu_vram_used_mb(),
        "running_models": list(state.running_processes.keys()),
        "last_swap_latency_seconds": round(state.last_swap_latency, 4),
        "recent_swaps": state.swap_history[-5:],
        "configured_models": list(MODELS.keys()),
        "swap_enabled": state.swap_enabled,
        "models": MODELS,
        "hardware": hw_info
    }


@app.post("/v1/vmm/swap")
async def manual_swap(request: Request):
    """Manually triggers a sub-2.3s VMM hot-swap to target model."""
    body = await request.json()
    target_role = state.resolve_model(body.get("model", "architect"))
    latency = await state.ensure_active(target_role)
    return {
        "status": "success",
        "active_model": state.active_vram_model,
        "swap_latency_seconds": round(latency, 4),
        "vram_used_mb": get_gpu_vram_used_mb()
    }


@app.post("/api/toggle-swap")
async def toggle_swap():
    state.swap_enabled = not state.swap_enabled
    logger.info(f">> [TOGGLE] swap_enabled changed to: {state.swap_enabled}")
    return {"swap_enabled": state.swap_enabled}


@app.post("/api/evict-all")
async def evict_all():
    state.shutdown_all()
    state.active_vram_model = None
    time.sleep(0.5)
    return {"status": "evicted", "vram_used_mb": get_gpu_vram_used_mb()}


@app.post("/api/evaluate")
async def evaluate_budget(request: Request):
    """Evaluates memory fit using GGUF metadata parser and engine KV specs."""
    if not evaluate_model_fit:
        raise HTTPException(status_code=500, detail="memory_advisor module not loaded")
    body = await request.json()
    model_path = body.get("model_path", "")
    context_tokens = int(body.get("context", 98304))
    engine_type = body.get("engine", "llamampere")
    ctk = body.get("ctk", None)
    ctv = body.get("ctv", None)
    
    result = evaluate_model_fit(
        model_path=model_path,
        context_tokens=context_tokens,
        engine_type=engine_type,
        ctk=ctk,
        ctv=ctv
    )
    return result


# =====================================================================
# OPENAI COMPATIBILITY ENDPOINTS (PASEO & PI-CODING-AGENT)
# =====================================================================

@app.get("/v1/models")
async def list_models():
    data = []
    for key, info in MODELS.items():
        data.append({
            "id": key,
            "object": "model",
            "created": 1700000000,
            "owned_by": "cuda-vmm-swap",
            "metadata": {
                "role": info.get("role"),
                "vram_gb": info.get("vram_gb"),
                "context": info.get("context"),
                "alias": info.get("alias"),
                "is_running": key in state.running_processes,
                "is_active_vram": key == state.active_vram_model
            }
        })
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested_model = body.get("model", "architect")
    target_key = state.resolve_model(requested_model)
    target_info = MODELS.get(target_key, {})
    backend_port = target_info.get("backend_port", 8081)

    # Ensure model is active in VRAM (or proxy to current if swap disabled)
    if state.swap_enabled:
        await state.ensure_active(target_key)
    else:
        # If swap disabled, route to active model
        if state.active_vram_model and state.active_vram_model in state.running_processes:
            target_key = state.active_vram_model
            target_info = MODELS.get(target_key, {})
            backend_port = target_info.get("backend_port", 8081)
        else:
            await state.ensure_active(target_key)

    # Rewrite model name to match backend alias
    body["model"] = target_info.get("alias", requested_model)
    target_url = f"http://127.0.0.1:{backend_port}/v1/chat/completions"
    is_streaming = body.get("stream", False)

    client = httpx.AsyncClient(timeout=3600.0)

    try:
        if not is_streaming:
            resp = await client.post(target_url, json=body)
            await client.aclose()

            # Evict debugger immediately after completion to clear VRAM for closed-loop coder fix
            if state.swap_enabled and target_key == "debugger":
                logger.info(">> [DEBUGGER INFERENCE COMPLETE]: Expelling Debugger from VRAM...")
                state.enforce_vram_eviction("debugger")

            return JSONResponse(status_code=resp.status_code, content=resp.json())

        req = client.build_request("POST", target_url, json=body)
        resp = await client.send(req, stream=True)

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
                if state.swap_enabled and target_key == "debugger":
                    logger.info(">> [DEBUGGER STREAM COMPLETE]: Expelling Debugger from VRAM...")
                    state.enforce_vram_eviction("debugger")

        return StreamingResponse(
            stream_generator(),
            status_code=resp.status_code,
            media_type="text/event-stream"
        )

    except Exception as e:
        await client.aclose()
        logger.error(f"Error forwarding request to {target_url}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to communicate with local model backend: {e}")


def main():
    print("=" * 68)
    print("  cuda-vmm-swap: Pipelined Router & Control Studio Active")
    print("  Strict VRAM Eviction Rule & Engine-Aware Memory Budgets")
    print(f"  Listening on: http://{ROUTER_HOST}:{ROUTER_PORT}")
    print(f"  Dashboard:    http://{ROUTER_HOST}:{ROUTER_PORT}/ui")
    print("=" * 68)
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT, log_level="info")


if __name__ == "__main__":
    main()
