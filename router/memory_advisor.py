"""
memory_advisor.py - Hardware Detection & Engine-Aware Memory Budget Calculator
==============================================================================
Part of the cuda-vmm-swap project.

Calculates exact physical VRAM and system RAM requirements for local LLM models
running under CUDA Virtual Memory Management (VMM).

Hardware & OS Invariants:
1. Windows WDDM Pinned RAM Cap: Windows restricts pinned host memory
   (cudaHostAlloc) to approximately 48-50% of physical system RAM.
2. Strict Sliding Window: At most 1 model resides in VRAM, at most 1 model is
   pinned in RAM staging buffer.
3. KV Cache Footprint: Exactly computed from GGUF architecture tensors
   (layers, kv_heads, head_dim) and engine-specific KV quantization types.

Supported Engines:
- llamAmpere / llamAmpere-Ultra (JakeATX): TurboQuant (turbo2..turbo5, tq5, tq6),
  SJ-KVaRN fused rotation codec, Q8_0, Q4_0, FP16.
- llama.cpp (Upstream): F16, Q8_0, Q5_0, Q5_1, Q4_0, Q4_1, IQ4_NL.
(Note: vLLM, Aphrodite, ExLlamaV2/TabbyAPI and hybrid offload engines like Strata
are excluded by design from native sub-2.3s VMM swap).
"""

import os
import sys
import ctypes
import subprocess
import json
from typing import Dict, Any, Optional, Tuple

try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False


# =====================================================================
# 1. HARDWARE DETECTION (GPU VRAM & SYSTEM RAM)
# =====================================================================

def get_system_ram_mb() -> int:
    """Returns total physical system RAM in megabytes."""
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys // (1024 * 1024))
    
    # Fallback to psutil or os
    try:
        import psutil
        return int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:
        return 32768  # Conservative 32 GB fallback


def get_gpu_vram_info() -> Dict[str, Any]:
    """Queries NVIDIA driver for total, used, and free physical VRAM in MB."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.check_output(cmd, text=True, timeout=3).strip()
        lines = res.split("\n")
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(",")]
            return {
                "gpu_name": parts[0],
                "total_mb": int(parts[1]),
                "used_mb": int(parts[2]),
                "free_mb": int(parts[3]),
                "detected": True
            }
    except Exception:
        pass
    
    return {
        "gpu_name": "NVIDIA CUDA Device (Fallback)",
        "total_mb": 24576,
        "used_mb": 0,
        "free_mb": 24576,
        "detected": False
    }


def get_hardware_profile() -> Dict[str, Any]:
    """Computes full memory budget and Windows WDDM pinned cap."""
    gpu = get_gpu_vram_info()
    ram_total_mb = get_system_ram_mb()
    
    # Windows WDDM driver caps cudaHostAlloc (pinned memory) at ~48-50% of physical RAM
    wddm_pinned_cap_mb = int(ram_total_mb * 0.48)
    
    return {
        "gpu": gpu,
        "system_ram_total_mb": ram_total_mb,
        "wddm_pinned_cap_mb": wddm_pinned_cap_mb,
        "overhead_cuda_mb": 1100  # Baseline CUDA runtime, context and activation buffers
    }


# =====================================================================
# 2. ENGINE & KV QUANTIZATION SPECIFICATIONS
# =====================================================================

# Bytes per element per KV type (factoring block headers and scaling)
KV_BYTES_PER_ELEMENT = {
    # Upstream llama.cpp types
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,  # 8 bits + block scale
    "q5_1": 0.70,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,  # 4 bits + block scale
    "iq4_nl": 0.5625,

    # JakeATX / llamAmpere custom & proprietary TurboQuant types
    # (TurboQuant algorithms originate from Google Research; SM86 fused implementation by JakeATX)
    "turbo5": 0.65,    # 5-bit key
    "turbo4": 0.53,    # 4-bit value / key
    "turbo3": 0.40,    # 3-bit MMA fused
    "turbo2": 0.28,    # 2-bit (value-only)
    "tq5_0": 0.65,     # 5-bit
    "tq6_0": 0.78,     # 6-bit
    "sjkvarn": 0.53    # SJ-KVaRN fused rotation compact state (typically ~4-bit effective)
}

ENGINE_CAPABILITIES = {
    "llamampere": {
        "name": "llamAmpere-Ultra (SM86 Native)",
        "supported_k_types": ["turbo5", "turbo4", "turbo3", "tq5_0", "tq6_0", "sjkvarn", "q8_0", "q4_0", "f16"],
        "supported_v_types": ["turbo4", "turbo3", "turbo2", "turbo5", "tq5_0", "tq6_0", "sjkvarn", "q8_0", "q4_0", "f16"],
        "supports_kv_unified": True,
        "supports_flash_attn_i8": True,
        "supports_vmm_native": True,
        "supports_modular_ple": True,
        "default_ctk": "turbo5",
        "default_ctv": "turbo4"
    },
    "llamacpp": {
        "name": "llama.cpp (Upstream with VMM patch)",
        "supported_k_types": ["q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f16", "f32"],
        "supported_v_types": ["q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f16", "f32"],
        "supports_kv_unified": False,
        "supports_flash_attn_i8": False,
        "supports_vmm_native": True,
        "default_ctk": "q8_0",
        "default_ctv": "q8_0"
    }
}


# =====================================================================
# 3. GGUF METADATA EXTRACTION
# =====================================================================

def inspect_gguf_model(model_path: str) -> Dict[str, Any]:
    """
    Reads GGUF binary header in milliseconds to extract exact attention
    tensor dimensions: layers, kv_heads, head_dim.
    """
    if not os.path.isfile(model_path):
        return {"error": f"File not found: {model_path}"}
    
    file_size_mb = int(os.path.getsize(model_path) // (1024 * 1024))
    
    if not GGUF_AVAILABLE:
        # Fallback estimation based on file size if gguf package not installed
        # Typical 27B GQA model defaults:
        return {
            "model_path": model_path,
            "file_size_mb": file_size_mb,
            "architecture": "generic_fallback",
            "layers": 65 if file_size_mb > 15000 else 32,
            "kv_heads": 4 if file_size_mb > 15000 else 8,
            "head_dim": 128,
            "is_gqa": True,
            "exact_header": False
        }
    
    try:
        reader = gguf.GGUFReader(model_path)
        arch_field = reader.get_field("general.architecture")
        arch = bytes(arch_field.parts[-1]).decode() if arch_field else "llama"
        
        # Read layers
        f_layer = reader.get_field(f"{arch}.block_count")
        layers = int(f_layer.parts[-1][0]) if f_layer else 32
        
        # Read KV heads (GQA check)
        f_kv_heads = reader.get_field(f"{arch}.attention.head_count_kv")
        f_heads = reader.get_field(f"{arch}.attention.head_count")
        heads = int(f_heads.parts[-1][0]) if f_heads else 32
        kv_heads = int(f_kv_heads.parts[-1][0]) if f_kv_heads else heads
        
        # Read embedding length or head dimension
        f_embd = reader.get_field(f"{arch}.embedding_length")
        f_key_len = reader.get_field(f"{arch}.attention.key_length")
        
        if f_key_len:
            head_dim = int(f_key_len.parts[-1][0])
        elif f_embd and heads:
            head_dim = int(f_embd.parts[-1][0]) // heads
        else:
            head_dim = 128
            
        return {
            "model_path": model_path,
            "file_size_mb": file_size_mb,
            "architecture": arch,
            "layers": layers,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "is_gqa": kv_heads < heads,
            "exact_header": True
        }
    except Exception as e:
        return {
            "model_path": model_path,
            "file_size_mb": file_size_mb,
            "architecture": "error_fallback",
            "layers": 65 if file_size_mb > 15000 else 32,
            "kv_heads": 4,
            "head_dim": 128,
            "is_gqa": True,
            "error": str(e),
            "exact_header": False
        }


# =====================================================================
# 4. MATHEMATICAL MEMORY EVALUATION
# =====================================================================

def calculate_kv_cache_mb(
    layers: int,
    kv_heads: int,
    head_dim: int,
    context_tokens: int,
    ctk: str = "turbo5",
    ctv: str = "turbo4"
) -> Tuple[float, float, float]:
    """
    Computes exact megabytes required for Key and Value cache buffers.
    Returns: (kv_total_mb, k_mb, v_mb)
    """
    bytes_k = KV_BYTES_PER_ELEMENT.get(ctk.lower(), 1.0)
    bytes_v = KV_BYTES_PER_ELEMENT.get(ctv.lower(), 1.0)
    
    # Number of elements per token for Key and Value
    elements_per_token_half = layers * kv_heads * head_dim
    
    k_bytes = elements_per_token_half * context_tokens * bytes_k
    v_bytes = elements_per_token_half * context_tokens * bytes_v
    
    k_mb = k_bytes / (1024.0 * 1024.0)
    v_mb = v_bytes / (1024.0 * 1024.0)
    
    return (k_mb + v_mb, k_mb, v_mb)


def evaluate_model_fit(
    model_path: str,
    context_tokens: int = 98304,
    engine_type: str = "llamampere",
    ctk: Optional[str] = None,
    ctv: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates whether a model, given context and KV quant, physically fits in
    GPU VRAM and respects the Windows WDDM 50% pinned RAM cap.
    """
    hw = get_hardware_profile()
    gpu_total_mb = hw["gpu"]["total_mb"]
    pinned_cap_mb = hw["wddm_pinned_cap_mb"]
    cuda_overhead_mb = hw["overhead_cuda_mb"]
    
    engine_meta = ENGINE_CAPABILITIES.get(engine_type.lower(), ENGINE_CAPABILITIES["llamampere"])
    
    # Resolve default quantization if not specified
    eff_ctk = ctk if ctk else engine_meta["default_ctk"]
    eff_ctv = ctv if ctv else engine_meta["default_ctv"]
    
    # Extract model tensor dimensions
    model_meta = inspect_gguf_model(model_path)
    weights_mb = model_meta.get("file_size_mb", 0)
    layers = model_meta.get("layers", 32)
    kv_heads = model_meta.get("kv_heads", 4)
    head_dim = model_meta.get("head_dim", 128)
    
    kv_total_mb, k_mb, v_mb = calculate_kv_cache_mb(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        context_tokens=context_tokens,
        ctk=eff_ctk,
        ctv=eff_ctv
    )
    
    total_vram_required_mb = weights_mb + kv_total_mb + cuda_overhead_mb
    vram_remaining_mb = gpu_total_mb - total_vram_required_mb
    fits_vram = vram_remaining_mb >= 0
    
    # Check Windows WDDM Pinned Memory Constraint for Background Pre-Staging
    fits_pinned_ram = weights_mb <= pinned_cap_mb
    pinned_remaining_mb = pinned_cap_mb - weights_mb
    
    # Maximum recommended context before OOM with this model and KV quant
    vram_for_kv_mb = max(0.0, gpu_total_mb - weights_mb - cuda_overhead_mb)
    bytes_per_token = (layers * kv_heads * head_dim * (KV_BYTES_PER_ELEMENT.get(eff_ctk, 1.0) + KV_BYTES_PER_ELEMENT.get(eff_ctv, 1.0)))
    max_safe_context = int((vram_for_kv_mb * 1024 * 1024) / bytes_per_token) if bytes_per_token > 0 else 0
    
    # Modular external PLE detection (--ple / -noPLE)
    uses_modular_ple = "-nople" in model_path.lower()
    ple_table_path = None
    ple_size_mb = 0
    if uses_modular_ple:
        model_dir = os.path.dirname(model_path)
        candidate = os.path.join(model_dir, "Qwen3.8-Flash-Next-PLE-Q8_0.gguf")
        if os.path.isfile(candidate):
            ple_table_path = candidate
            ple_size_mb = int(os.path.getsize(candidate) // (1024 * 1024))

    return {
        "hardware": hw,
        "engine": {
            "type": engine_type,
            "name": engine_meta["name"],
            "ctk": eff_ctk,
            "ctv": eff_ctv
        },
        "model": model_meta,
        "budget": {
            "weights_mb": weights_mb,
            "kv_cache_total_mb": round(kv_total_mb, 2),
            "kv_cache_k_mb": round(k_mb, 2),
            "kv_cache_v_mb": round(v_mb, 2),
            "cuda_overhead_mb": cuda_overhead_mb,
            "total_vram_required_mb": round(total_vram_required_mb, 2),
            "vram_total_mb": gpu_total_mb,
            "vram_remaining_mb": round(vram_remaining_mb, 2),
            "fits_vram": fits_vram,
            "max_safe_context_tokens": max_safe_context,
            "wddm_pinned_cap_mb": pinned_cap_mb,
            "fits_pinned_ram": fits_pinned_ram,
            "pinned_remaining_mb": pinned_remaining_mb,
            "modular_ple": {
                "active": uses_modular_ple,
                "table_path": ple_table_path,
                "table_size_mb": ple_size_mb,
                "vram_overhead_mb": 0,
                "pinned_ram_overhead_mb": 0,
                "storage_mode": "NVMe lazy kernel mmap (TENSOR_READ_LAZY)"
            }
        }
    }


def print_memory_report(eval_result: Dict[str, Any]):
    """Prints a clean ASCII breakdown of the memory budget."""
    hw = eval_result["hardware"]
    eng = eval_result["engine"]
    m = eval_result["model"]
    b = eval_result["budget"]
    
    print("=" * 72)
    print("  cuda-vmm-swap: Engine-Aware Memory Budget & Viability Report")
    print("=" * 72)
    print(f"  GPU Hardware:         {hw['gpu']['gpu_name']}")
    print(f"  Total GPU VRAM:       {hw['gpu']['total_mb']:,} MB ({hw['gpu']['total_mb']/1024:.2f} GB)")
    print(f"  System Physical RAM:  {hw['system_ram_total_mb']:,} MB ({hw['system_ram_total_mb']/1024:.2f} GB)")
    print(f"  Windows WDDM Cap:     {hw['wddm_pinned_cap_mb']:,} MB (Max pinned buffer for VMM hot-swap)")
    print("-" * 72)
    print(f"  Target Engine:        {eng['name']}")
    print(f"  Model Architecture:   {m.get('architecture', 'Unknown')} ({m.get('layers', 0)} layers, {m.get('kv_heads', 0)} KV heads, {m.get('head_dim', 0)} dim)")
    print(f"  KV Quantization:      Key: {eng['ctk']} | Value: {eng['ctv']}")
    if b.get("modular_ple", {}).get("active"):
        ple = b["modular_ple"]
        print(f"  Modular PLE Table:    {ple['table_size_mb']:,} MB ({ple['table_size_mb']/1024:.2f} GB) on NVMe [0 MB VRAM / 0 MB Pinned RAM]")
    print("-" * 72)
    print("  VRAM BREAKDOWN:")
    print(f"    - Model Weights:    {b['weights_mb']:,} MB ({b['weights_mb']/1024:.2f} GB)")
    print(f"    - KV Cache:         {b['kv_cache_total_mb']:,} MB ({b['kv_cache_total_mb']/1024:.2f} GB)")
    print(f"    - CUDA Overhead:    {b['cuda_overhead_mb']:,} MB ({b['cuda_overhead_mb']/1024:.2f} GB)")
    print(f"    - TOTAL REQUIRED:   {b['total_vram_required_mb']:,} MB ({b['total_vram_required_mb']/1024:.2f} GB)")
    print("-" * 72)
    
    if b["fits_vram"]:
        print(f"  [VERDICT VRAM]:       PASSED (Free VRAM remaining: {b['vram_remaining_mb']:,} MB)")
        print(f"  [MAX SAFE CONTEXT]:   ~{b['max_safe_context_tokens']:,} tokens with current quant")
    else:
        print(f"  [VERDICT VRAM]:       FAILED - OOM RISK (Exceeds VRAM by {abs(b['vram_remaining_mb']):,} MB)")
        print("  [RECOMMENDATION]:     Enable TurboQuant (turbo4/turbo5) or reduce context window.")
        
    if b["fits_pinned_ram"]:
        print(f"  [VERDICT WDDM RAM]:   PASSED (Fits in Windows 50% pinned RAM buffer, {b['pinned_remaining_mb']:,} MB headroom)")
    else:
        print(f"  [VERDICT WDDM RAM]:   FAILED - Exceeds Windows WDDM pinned cap by {abs(b['pinned_remaining_mb']):,} MB")
        print("                        Background RAM staging will cause cudaErrorMemoryAllocation.")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate exact VRAM and WDDM RAM budgets for cuda-vmm-swap")
    parser.add_argument("--model", type=str, default="", help="Path to GGUF model file")
    parser.add_argument("--context", type=int, default=98304, help="Context tokens (e.g. 32768, 65536, 98304, 163840)")
    parser.add_argument("--engine", type=str, default="llamampere", choices=["llamampere", "llamacpp"], help="Inference engine")
    parser.add_argument("--ctk", type=str, default="turbo5", help="KV Key quantization")
    parser.add_argument("--ctv", type=str, default="turbo4", help="KV Value quantization")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()
    
    if not args.model:
        # If no model provided, search D:\models or current dir
        search_dirs = [r"D:\models", "."]
        found = []
        for d in search_dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".gguf"):
                        found.append(os.path.join(d, f))
        if found:
            args.model = found[0]
            print(f"[INFO] Auto-selected detected model: {args.model}")
        else:
            hw = get_hardware_profile()
            print("No model specified. Hardware Summary:")
            print(json.dumps(hw, indent=2))
            sys.exit(0)
            
    res = evaluate_model_fit(
        model_path=args.model,
        context_tokens=args.context,
        engine_type=args.engine,
        ctk=args.ctk,
        ctv=args.ctv
    )
    
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_memory_report(res)
