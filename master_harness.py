"""master_harness.py

Entry point to launch the VMM router and coordinate multi-model agent execution
(Architect, Coder, Debugger, or custom roles) with CUDA VMM fast hot-swapping.

Usage::
    python master_harness.py [--swap on|off] [--open-ui] [--check-memory]

Features:
- Launches router/vmm_router.py as a managed subprocess.
- Automatically handles Python interpreter discovery (project-bundled or current sys.executable).
- Provides memory budget checks against GPU VRAM and Windows WDDM 50% pinned RAM cap.
- Polls health endpoint until responsive, then opens Control Studio or keeps daemon alive.
"""
import subprocess
import sys
import time
import os
import argparse
import webbrowser
from pathlib import Path

# Resolve python interpreter: check bundled project Python first, fallback to sys.executable
DEFAULT_PROJECT_PYTHON = r"d:\bjivanovich2017\My Drive\aibot\python\python.exe"
if os.path.isfile(DEFAULT_PROJECT_PYTHON):
    PYTHON = DEFAULT_PROJECT_PYTHON
else:
    PYTHON = sys.executable

# Resolve paths
ROUTER_PATH = Path(__file__).parent / "router" / "vmm_router.py"
ADVISOR_PATH = Path(__file__).parent / "router" / "memory_advisor.py"


def run_memory_check():
    """Runs memory advisor to report GPU VRAM and Windows WDDM limits."""
    if ADVISOR_PATH.is_file():
        subprocess.run([PYTHON, str(ADVISOR_PATH)])
    else:
        print("[WARN] memory_advisor.py not found.")


def start_router(swap_enabled: bool = True):
    """Launch vmm_router.py in a separate process and wait until the endpoint is responsive."""
    env = os.environ.copy()
    if not swap_enabled:
        env["VMM_SWAP_ENABLED"] = "0"

    print("=" * 68)
    print("  master_harness: Launching cuda-vmm-swap Control Studio")
    print(f"  Python Interpreter: {PYTHON}")
    print(f"  Swap Mode: {'ENABLED' if swap_enabled else 'DISABLED (Single Model)'}")
    print("=" * 68)

    proc = subprocess.Popen([
        PYTHON,
        str(ROUTER_PATH)
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # Wait for the router pool to report operational status
    ready = False
    for _ in range(60):  # up to 60s
        line = proc.stdout.readline()
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
        if line and ("[VMM POOL READY]" in line or "Uvicorn running on" in line):
            ready = True
            break
        time.sleep(0.5)

    if not ready:
        print("[ERROR] Router did not become ready – exiting")
        proc.terminate()
        sys.exit(1)

    # Prevent OS pipe buffer deadlock by draining stdout in a daemon thread
    import threading
    def drain_stream(pipe):
        try:
            for l in iter(pipe.readline, ''):
                if l.strip():
                    pass
        except Exception:
            pass
    t = threading.Thread(target=drain_stream, args=(proc.stdout,), daemon=True)
    t.start()

    # Verify /v1/models endpoint
    import urllib.request, urllib.error
    health_url = "http://127.0.0.1:8000/v1/models"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    print("\n[OK] Router health endpoint verified: http://127.0.0.1:8000")
                    print("[OK] Web Control Studio available at: http://127.0.0.1:8000/ui\n")
                    return proc
        except urllib.error.URLError:
            pass
        time.sleep(1)

    print("[ERROR] Router health check failed after waiting – exiting")
    proc.terminate()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Master Harness runner for cuda-vmm-swap")
    parser.add_argument("--swap", choices=["on", "off"], default="on", help="Toggle VMM model swap (default: on)")
    parser.add_argument("--open-ui", action="store_true", help="Automatically open Web Control Studio in browser")
    parser.add_argument("--check-memory", action="store_true", help="Print GPU VRAM and WDDM memory budget report and exit")
    args = parser.parse_args()

    if args.check_memory:
        run_memory_check()
        return

    swap_enabled = (args.swap.lower() == "on")
    router_proc = start_router(swap_enabled=swap_enabled)

    if args.open_ui:
        try:
            webbrowser.open("http://127.0.0.1:8000/ui")
        except Exception:
            pass

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Terminating router...")
        router_proc.terminate()
        try:
            router_proc.wait(timeout=3)
        except Exception:
            router_proc.kill()
        print("[SHUTDOWN] Router stopped cleanly.")


if __name__ == "__main__":
    main()
