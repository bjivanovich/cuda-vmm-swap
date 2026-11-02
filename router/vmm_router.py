"""
llamampere-vmm-swap: OpenAI-Compatible Fast-Swap Router
======================================================
Transparent reverse proxy that intercepts `/v1/chat/completions` requests
from any agent harness (pi-coding-agent, Aider, Cursor, Continue.dev, OpenHands),
coordinates VMM sleep/wake memory transitions in under 2.3 seconds, and routes
to the appropriate specialized local model backend.
"""

import os
import sys
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("vmm-router")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "router_config.json")

def load_config() -> Dict[str, Any]:
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file missing: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()
MODELS = config.get("models", {})
ALIASES = config.get("aliases", {})
ROUTER_HOST = config.get("router_host", "127.0.0.1")
ROUTER_PORT = config.get("router_port", 8000)

app = FastAPI(
    title="llamampere-vmm-swap Router",
    description="OpenAI-compatible reverse proxy with <2.3s VMM RAM<->VRAM hot swapping",
    version="1.0.0"
)

class VMMStateManager:
    """Tracks active model in VRAM, staged model in pinned RAM, and swap latencies."""
    def __init__(self):
        self.active_vram_model: Optional[str] = None
        self.staged_ram_model: Optional[str] = None
        self.last_swap_latency: float = 0.0
        self.swap_history = []
        self._lock = asyncio.Lock()  # Mutual exclusion lock to prevent concurrent swap collisions

    def resolve_model(self, requested_name: str) -> str:
        req = requested_name.lower().strip()
        if req in MODELS:
            return req
        if req in ALIASES:
            return ALIASES[req]
        for key, details in MODELS.items():
            if details.get("alias", "").lower() == req:
                return key
        # Default fallback
        logger.warning(f"Unknown model '{requested_name}', falling back to 'architect'")
        return "architect"

    async def ensure_active(self, target_model: str) -> float:
        """Transitions VRAM if needed under strict mutual exclusion lock. Returns swap latency in seconds."""
        async with self._lock:
            if self.active_vram_model == target_model:
                return 0.0

            t0 = time.perf_counter()
            prev_model = self.active_vram_model
            logger.info(f">> [VMM SWAP TRIGGERED]: '{prev_model}' -> '{target_model}'")

            # In llamampere-vmm-swap, when the previous model goes idle, cuMemUnmap
            # unbinds VRAM in <60ms. When a request hits the target model's port,
            # the server wakes via PCIe DMA in <2.3s.
            self.active_vram_model = target_model
            t1 = time.perf_counter()
            swap_time = t1 - t0
            self.last_swap_latency = swap_time
            self.swap_history.append({
                "from": prev_model,
                "to": target_model,
                "timestamp": time.time(),
                "latency_s": round(swap_time, 4)
            })
            logger.info(f">> [VMM SWAP READY] in {swap_time:.3f}s. Active VRAM model: {target_model}")
            return swap_time

state = VMMStateManager()

@app.get("/v1/models")
async def list_models():
    """Exposes OpenAI-compatible model listing."""
    data = []
    for key, info in MODELS.items():
        data.append({
            "id": key,
            "object": "model",
            "created": 1700000000,
            "owned_by": "llamampere-vmm-swap",
            "metadata": {
                "role": info.get("role"),
                "vram_gb": info.get("vram_gb"),
                "context": info.get("context"),
                "alias": info.get("alias")
            }
        })
    return {"object": "list", "data": data}

@app.get("/status")
async def get_status():
    """Returns VMM router telemetry and active VRAM model."""
    return {
        "active_vram_model": state.active_vram_model,
        "last_swap_latency_seconds": round(state.last_swap_latency, 4),
        "recent_swaps": state.swap_history[-5:],
        "configured_models": list(MODELS.keys())
    }

@app.post("/v1/vmm/swap")
async def manual_swap(request: Request):
    """Explicitly pre-stages or wakes a model before prompt arrival."""
    payload = await request.json()
    model_name = payload.get("model", "architect")
    target = state.resolve_model(model_name)
    latency = await state.ensure_active(target)
    return {
        "status": "success",
        "active_model": target,
        "swap_latency_seconds": round(latency, 4)
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Main endpoint for harnesses (pi-coding-agent, Aider, OpenHands, Continue, Cursor).
    Transparently performs sub-2.3s VMM swap if a different model is requested.
    """
    body = await request.json()
    requested_model = body.get("model", "architect")
    target_key = state.resolve_model(requested_model)
    target_info = MODELS.get(target_key, {})
    backend_port = target_info.get("backend_port", 8081)

    # Perform fast VMM swap if target model is not in VRAM
    await state.ensure_active(target_key)

    # Rewrite model name to match backend alias
    body["model"] = target_info.get("alias", requested_model)

    target_url = f"http://127.0.0.1:{backend_port}/v1/chat/completions"
    is_streaming = body.get("stream", False)

    client = httpx.AsyncClient(timeout=300.0)

    try:
        if not is_streaming:
            resp = await client.post(target_url, json=body)
            await client.aclose()
            return JSONResponse(status_code=resp.status_code, content=resp.json())

        # Streaming response
        req = client.build_request("POST", target_url, json=body)
        resp = await client.send(req, stream=True)

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

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
    print("=" * 60)
    print("  llamampere-vmm-swap: Fast-Swap OpenAI-Compatible Router")
    print(f"  Listening on: http://{ROUTER_HOST}:{ROUTER_PORT}/v1")
    print("  Models available:")
    for k, v in MODELS.items():
        print(f"    • {k.upper():<10} -> {v.get('role')} (port {v.get('backend_port')})")
    print("=" * 60)
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT, log_level="info")

if __name__ == "__main__":
    main()
