#!/usr/bin/env python3
"""
llamampere-vmm-swap: Autonomous 3-Agent Pipeline Runner
======================================================
Executes a complete sequential engineering workflow across 3 distinct models:
  1. Architect (ATX-Swift-1.5 Q5_K_M): Systemic specification & modular design
  2. Coder (Gyro-Coder TQ2_0): Synthesis of production code & unit tests
  3. Debugger / Auditor (Cyber-ATX Q5_K_M): Forensic security audit & test validation

Demonstrates real-time VMM hot-swapping (<2.3s) on a single 24 GB GPU.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import urllib.request
import urllib.error

DEFAULT_ROUTER_URL = "http://127.0.0.1:8000/v1/chat/completions"

def call_agent(router_url: str, model_role: str, system_prompt: str, user_prompt: str) -> tuple[str, float]:
    """Sends prompt to the VMM router and measures total round-trip time."""
    payload = {
        "model": model_role,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2 if model_role == "debugger" else 0.6,
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        router_url,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            t1 = time.perf_counter()
            res_json = json.loads(resp.read().decode("utf-8"))
            content = res_json["choices"][0]["message"]["content"]
            return content, (t1 - t0)
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Failed connecting to VMM router at {router_url}: {e}")
        print("Ensure 'python router/vmm_router.py' is running!")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run autonomous 3-agent pipeline with VMM hot swaps")
    parser.add_argument("--task", type=str, default="Create a high-performance distributed telemetry engine in Python with shared memory ring buffers and async event loop.", help="Task prompt for the multi-agent pipeline")
    parser.add_argument("--router", type=str, default=DEFAULT_ROUTER_URL, help="VMM Router API URL")
    parser.add_argument("--outdir", type=str, default="./pipeline_output", help="Directory to save generated artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("  llamampere-vmm-swap: Autonomous 3-Agent Sequential Pipeline")
    print(f"  Target Task: {args.task}")
    print(f"  Router URL:  {args.router}")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: ARCHITECT (ATX-Swift-1.5 Q5_K_M)
    # -------------------------------------------------------------
    print("\n[STAGE 1/3] Invoking ARCHITECT (ATX-Swift-1.5 Q5_K_M)...")
    print(">> Role: Deep architectural breakdown, data structures, and contract design")
    sys_arch = "You are an elite principal systems architect. Design a modular, unconstrained, high-performance architecture for the user request. Specify all interfaces, protocols, and data structures in complete detail."
    
    plan, duration_arch = call_agent(args.router, "architect", sys_arch, args.task)
    plan_path = os.path.join(args.outdir, "01_architecture_plan.md")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan)
    print(f">> Plan completed in {duration_arch:.2f}s (saved to {plan_path})")

    # -------------------------------------------------------------
    # STAGE 2: CODER (Gyro-Coder TQ2_0)
    # -------------------------------------------------------------
    print("\n[STAGE 2/3] Transitioning to CODER (Gyro-Coder TQ2_0)...")
    print(">> Triggering VMM Hot-Swap (RAM -> VRAM DMA across PCIe 3.0 x16)...")
    sys_coder = "You are an expert software engineer. Implement the complete, production-ready Python solution based strictly on the architectural blueprint. Include runnable unit tests with unittest. Output pure Python code in a single executable block."
    prompt_coder = f"Architectural Blueprint:\n\n{plan}\n\nTask: Implement the complete source code and unit tests."

    code_res, duration_coder = call_agent(args.router, "coder", sys_coder, prompt_coder)
    
    # Extract code block if wrapped in markdown
    extracted_code = code_res
    if "```python" in code_res:
        extracted_code = code_res.split("```python")[1].split("```")[0].strip()
    elif "```" in code_res:
        extracted_code = code_res.split("```")[1].split("```")[0].strip()

    code_path = os.path.join(args.outdir, "02_implementation.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(extracted_code)
    print(f">> Code synthesis completed in {duration_coder:.2f}s (saved to {code_path})")

    # -------------------------------------------------------------
    # SANDBOX EXECUTION CHECK
    # -------------------------------------------------------------
    print("\n[SANDBOX] Running generated code in live Python interpreter...")
    proc = subprocess.run([sys.executable, code_path], capture_output=True, text=True, timeout=30)
    exit_code = proc.returncode
    stdout = proc.stdout
    stderr = proc.stderr
    print(f">> Execution Exit Code: {exit_code}")
    if exit_code == 0:
        print(">> Execution Status: SUCCESS (0 errors)")
    else:
        print(f">> Execution Status: FAILED (Exit Code {exit_code})")
        print(f"   stderr: {stderr[:300]}")

    # -------------------------------------------------------------
    # STAGE 3: DEBUGGER & AUDITOR (Cyber-ATX Q5_K_M)
    # -------------------------------------------------------------
    print("\n[STAGE 3/3] Transitioning to DEBUGGER / AUDITOR (Cyber-ATX Q5_K_M)...")
    print(">> Triggering VMM Hot-Swap (RAM -> VRAM DMA across PCIe 3.0 x16)...")
    sys_debug = "You are a cybersecurity auditor and senior systems debugger. Review the provided implementation and runtime output. Identify race conditions, security vulnerabilities, or bugs. Provide the final hardened, production-grade code."
    prompt_debug = f"Original Code:\n```python\n{extracted_code}\n```\n\nRuntime Execution Result:\nExit Code: {exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n\nAudit, fix, and provide hardened code."

    audit_res, duration_debug = call_agent(args.router, "debugger", sys_debug, prompt_debug)
    
    hardened_code = audit_res
    if "```python" in audit_res:
        hardened_code = audit_res.split("```python")[1].split("```")[0].strip()
    elif "```" in audit_res:
        hardened_code = audit_res.split("```")[1].split("```")[0].strip()

    audit_path = os.path.join(args.outdir, "03_hardened_audited.py")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(hardened_code)
    print(f">> Hardening & security audit completed in {duration_debug:.2f}s (saved to {audit_path})")

    # -------------------------------------------------------------
    # FINAL BENCHMARK SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AUTONOMOUS 3-AGENT PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Stage 1 (Architect - Swift Q5):     {duration_arch:.2f} s")
    print(f"  Stage 2 (Coder - Gyro-Coder TQ2):   {duration_coder:.2f} s")
    print(f"  Stage 3 (Debugger - Cyber Q5):      {duration_debug:.2f} s")
    print(f"  Total End-to-End Pipeline Time:     {(duration_arch + duration_coder + duration_debug):.2f} s")
    print(f"  Artifacts saved to:                 {os.path.abspath(args.outdir)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
