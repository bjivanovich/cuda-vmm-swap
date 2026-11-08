"""master_harness.py

Entry point to launch the VMM router and coordinate the three model agents (Architect, Coder, Debugger)
for Kotlin app generation.

Usage::
    python master_harness.py

The script starts the router (vmm_router.py) as a background subprocess,
waits for it to become healthy, then keeps the process alive.
"""
import subprocess, sys, time, os
from pathlib import Path

# Path to the bundled Python interpreter (project‑specific)
PYTHON = r"d:\\bjivanovich2017\\My Drive\\aibot\\python\\python.exe"

# Resolve router script location relative to this file
ROUTER_PATH = Path(__file__).parent / "router" / "vmm_router.py"

def start_router():
    """Launch vmm_router.py in a separate process and wait until the OpenAI-compatible endpoint is responsive."""
    proc = subprocess.Popen([
        PYTHON,
        str(ROUTER_PATH)
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Wait for the router to print the pool ready line (model loading may still be in progress)
    ready = False
    for _ in range(60):  # up to 60 s
        line = proc.stdout.readline()
        if line and "[VMM POOL READY]" in line:
            ready = True
            print("Router pool ready – awaiting model health…")
            break
        time.sleep(1)
    if not ready:
        print("Router did not become ready – exiting")
        proc.terminate()
        sys.exit(1)
    # Prevent OS pipe buffer deadlock by draining stdout in a daemon thread
    import threading
    def drain_stream(pipe):
        try:
            for _ in iter(pipe.readline, ''):
                pass
        except Exception:
            pass
    t = threading.Thread(target=drain_stream, args=(proc.stdout,), daemon=True)
    t.start()

    # Additional check: poll the /v1/models endpoint until it returns 200 OK
    import urllib.request, urllib.error
    health_url = "http://127.0.0.1:8000/v1/models"
    for attempt in range(60):  # try for up to 60 s
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    print("Router health endpoint reachable – proceeding.")
                    return proc
        except urllib.error.URLError:
            pass
        time.sleep(1)
    print("Router health check failed after waiting – exiting")
    proc.terminate()
    sys.exit(1)

def main():
    router_proc = start_router()
    try:
        # Keep the script alive while the router runs; agents are managed via the router API.
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down router…")
        router_proc.terminate()

if __name__ == "__main__":
    main()
