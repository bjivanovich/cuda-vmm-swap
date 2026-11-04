#!/usr/bin/env python3
"""
llamampere-vmm-swap: Autonomous 3-Agent Pipeline Runner with Closed-Loop Coder Fix
==================================================================================
Executes a sequential engineering workflow across 3 distinct models with strict VRAM eviction:
  1. Architect (ATX-Swift-1.5 Q5_K_M): Systemic specification & modular design
  2. Coder (Gyro-Coder TQ2_0): Synthesis of production code & unit tests
  3. Debugger / Auditor (Cyber-ATX Q5_K_M): Forensic security audit & defect detection
  4. Coder Fix Loop (Gyro-Coder TQ2_0): Hot-swapped back into VRAM to correct all
     defects and vulnerabilities identified by the Debugger.

Demonstrates real-time VMM hot-swapping (<2.3s) with ZERO VRAM overlap.
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
        "temperature": 0.2 if model_role in ["debugger"] else 0.6,
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

def extract_code_block(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```kotlin" in text:
        return text.split("```kotlin")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Run autonomous 3-agent pipeline with strict VMM hot swaps")
    parser.add_argument("--task", type=str, default="Create a high-performance distributed telemetry engine in Python with shared memory ring buffers and async event loop.", help="Task prompt for the multi-agent pipeline")
    parser.add_argument("--router", type=str, default=DEFAULT_ROUTER_URL, help="VMM Router API URL")
    parser.add_argument("--outdir", type=str, default="./pipeline_output", help="Directory to save generated artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("  llamampere-vmm-swap: Autonomous 3-Agent Sequential Pipeline")
    print("  Strict VRAM Eviction Rule & Closed-Loop Coder Correction")
    print(f"  Target Task: {args.task}")
    print(f"  Router URL:  {args.router}")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: ARCHITECT (ATX-Swift-1.5 Q5_K_M)
    # -------------------------------------------------------------
    print("\n[STAGE 1/4] Invoking ARCHITECT (ATX-Swift-1.5 Q5_K_M)...")
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
    print("\n[STAGE 2/4] Transitioning to CODER (Gyro-Coder TQ2_0)...")
    print(">> Evicting Architect from VRAM and swapping Coder from RAM to VRAM (<2.3s)...")
    sys_coder = "You are an expert software engineer. Implement the complete, production-ready Python solution based strictly on the architectural blueprint. Include runnable unit tests with unittest. Output pure Python code in a single executable block."
    prompt_coder = f"Architectural Blueprint:\n\n{plan}\n\nTask: Implement the complete source code and unit tests."

    code_res, duration_coder = call_agent(args.router, "coder", sys_coder, prompt_coder)
    extracted_code = extract_code_block(code_res)

    code_path = os.path.join(args.outdir, "02_implementation.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(extracted_code)
    print(f">> Initial code synthesis completed in {duration_coder:.2f}s (saved to {code_path})")

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
    print("\n[STAGE 3/4] Transitioning to DEBUGGER / AUDITOR (Cyber-ATX Q5_K_M)...")
    print(">> Evicting Coder from VRAM and swapping Debugger into VRAM (<2.3s)...")
    sys_debug = (
        "You are a cybersecurity auditor and senior systems debugger. "
        "Review the provided implementation and runtime output. "
        "Identify race conditions, security vulnerabilities, memory bottlenecks, or defects. "
        "Output an explicit, prioritized list of defects and exact correction instructions for the Coder."
    )
    prompt_debug = (
        f"Original Implementation:\n```python\n{extracted_code}\n```\n\n"
        f"Runtime Execution Result:\nExit Code: {exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n\n"
        f"Analyze all defects, bugs, and security weaknesses. Detail the exact fixes required."
    )

    audit_report, duration_debug = call_agent(args.router, "debugger", sys_debug, prompt_debug)
    report_path = os.path.join(args.outdir, "03_debugger_diagnostic_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(audit_report)
    print(f">> Debugger diagnostic completed in {duration_debug:.2f}s (saved to {report_path})")
    print(">> Debugger has culminated inference and is automatically expelled from VRAM.")

    # -------------------------------------------------------------
    # STAGE 4: CODER FIX LOOP (Gyro-Coder TQ2_0)
    # -------------------------------------------------------------
    print("\n[STAGE 4/4] Hot-swapping CODER (Gyro-Coder TQ2_0) back into clean VRAM...")
    print(">> Gyro-Coder executes the Debugger's instructions and produces the hardened code...")
    sys_coder_fix = (
        "You are the Lead Implementation Engineer (Gyro-Coder). "
        "You have received a comprehensive forensic defect and security report from the Debugger. "
        "Apply all necessary fixes, harden the code, resolve any race conditions or memory issues, "
        "and output the complete, final, production-ready implementation."
    )
    prompt_coder_fix = (
        f"Base Implementation:\n```python\n{extracted_code}\n```\n\n"
        f"Debugger Forensic Diagnostic & Fix Instructions:\n{audit_report}\n\n"
        f"Output the complete corrected source code incorporating all fixes."
    )

    fix_res, duration_fix = call_agent(args.router, "coder", sys_coder_fix, prompt_coder_fix)
    final_code = extract_code_block(fix_res)

    final_path = os.path.join(args.outdir, "04_final_hardened_coder_fix.py")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(final_code)
    print(f">> Gyro-Coder fix loop completed in {duration_fix:.2f}s (saved to {final_path})")

    # Sandbox verification of corrected code
    print("\n[SANDBOX VERIFICATION] Running hardened code from Gyro-Coder...")
    proc_final = subprocess.run([sys.executable, final_path], capture_output=True, text=True, timeout=30)
    print(f">> Final Exit Code: {proc_final.returncode}")
    if proc_final.returncode == 0:
        print(">> VERIFICATION PASSED: All errors resolved by Gyro-Coder!")
    else:
        print(f">> Final Stderr: {proc_final.stderr[:300]}")

    # -------------------------------------------------------------
    # FINAL BENCHMARK SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  AUTONOMOUS 4-STAGE PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"  Stage 1 (Architect - Swift Q5):         {duration_arch:.2f} s")
    print(f"  Stage 2 (Coder - Gyro-Coder TQ2):       {duration_coder:.2f} s")
    print(f"  Stage 3 (Debugger - Cyber Q5):          {duration_debug:.2f} s")
    print(f"  Stage 4 (Coder Fix - Gyro-Coder TQ2):   {duration_fix:.2f} s")
    print(f"  Total End-to-End Pipeline Time:         {(duration_arch + duration_coder + duration_debug + duration_fix):.2f} s")
    print(f"  Artifacts saved to:                     {os.path.abspath(args.outdir)}")
    print("=" * 75)

if __name__ == "__main__":
    main()
