# llamampere-vmm-swap

**Fast LLM model swapping between RAM and VRAM in under 2.3 seconds using CUDA Virtual Memory Management (VMM).**

Patch, architecture, and benchmarks for a custom `llama.cpp` fork (**llamAmpere-Ultra**) — verified on a single **NVIDIA GeForce RTX 3090 (24 GB VRAM)** running a sequential 3-agent pipeline: **Architect → Coder → Debugger/Auditor**, supporting models from 14.3 GB to 19.9 GB VRAM active footprint with context windows up to **160k - 256k tokens**.

> **Credit & Acknowledgement:** The core low-level VMM `sleep` / `wake` concept was pioneered by Reddit user
> **u/enndeeee** in [Infernix PR #43](https://github.com/infernix/llama.cpp/pull/43) (`SleepableBuffer`).
> This repository adapts and expands that approach for Windows 11 / WDDM driver limitations, multi-model agent switching, background pre-staging, and the `llamAmpere-Ultra` execution engine.

---

## The Problem: The 30-Second Multi-Agent Wall

In local sequential multi-agent workflows, each agent requires a specialized model (e.g. unconstrained systemic reasoning for architecture, clean syntax synthesis for code, or cybersecurity forensic fine-tuning for debugging). On a single 24 GB GPU, models cannot coexist in VRAM simultaneously.

Standard `llama.cpp` process destruction and reinitialization incurs catastrophic switching penalties:

| Operation | Time Lost | Cause |
|---|---|---|
| Process Teardown & Context Free | 5 – 8 s | Windows WDDM driver unbinding and memory release |
| Cold CUDA Driver Initialization | 2.5 – 5 s | `cuInit`, `cuBLAS`, and `cuBLASLt` handle creation |
| Model Weight Re-read (NVMe Disk) | 15 – 30 s | 1 MB chunked synchronous disk-to-GPU copies |
| CUDA Graph Compilation | 3 – 5 s | Rebuilding computation graph from scratch |
| **Total Inter-Agent Latency** | **28 – 60+ s** | **Unusable for interactive agent pipelines** |

---

## The Solution: CUDA Driver VMM + Background Pre-Staging

By utilizing the low-level **CUDA Virtual Memory Management (VMM) API (`cuMem*`)**:
1. **Invariable Virtual Addresses**: Each model reserves a unique, permanent virtual address space (`cuMemAddressReserve`) at process launch.
2. **Instant Eviction (`Sleep`)**: When an agent finishes, `cuMemUnmap` releases physical VRAM in **under 60 ms** without terminating the process or destroying the CUDA context.
3. **Sub-2.3s DMA Restoration (`Wake`)**: `cuMemMap` binds physical pages back to the address and streams weights directly from pinned host RAM (`cudaHostAlloc`) across PCIe Gen4 x16 at **8.44 - 8.50 GB/s**.
4. **Asynchronous Overlapped Pre-Staging**: While Model A generates tokens in VRAM (25–45s), a background OS thread stages and pins Model B in host RAM. When Model A sleeps, Model B is already primed in RAM, waking in **1.71s - 2.29s**.

```
[TIME]
t = 0s      [ARCHITECT in VRAM] (ATX-Swift-1.5 Q5_K_M) ──────────► cuMemUnmap (<60 ms)
            (Generates technical blueprint & architecture)               │
            [Background Thread]: Pins Gyro-Coder in RAM (19.9 GB)        │
                                                                         ▼
t = 25s     [CODER WAKE]: cuMemMap + PCIe DMA (2.29s) ◄──────────────────┘
            [CODER in VRAM] (Gyro-Coder TQ2_0) ──────────────────► cuMemUnmap (<60 ms)
            (Synthesizes clean code via 70B MoE + A8 PLE N-Gram)         │
            [Background Thread]: Pins Cyber-ATX in RAM (18.8 GB)         │
                                                                         ▼
t = 70s     [DEBUGGER WAKE]: cuMemMap + PCIe DMA (2.21s) ◄───────────────┘
            [DEBUGGER in VRAM] (Cyber-ATX Q5_K_M)
            (Audits vulnerabilities, verifies execution, runs live test suite)
```

---

## The Multi-Agent Triad (Architecture, Coding, Debugging)

The pipeline is purpose-built to autonomously design, implement, and audit software without corporate refusals or syntactic degradation.

```
┌────────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ 1. ARCHITECT                   │ 2. CODER                      │ 3. DEBUGGER / AUDITOR         │
├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ ATX-Swift-1.5 Q5_K_M           │ Gyro-Coder TQ2_0              │ Cyber-ATX Q5_K_M              │
│ (bjivanovich / HF)             │ (AgentionAI / HF)             │ (bjivanovich / HF)            │
│                                │                               │                               │
│ • Uncensored / abliterated     │ • 70B MoE (3.8B active)       │ • Cybersecurity fine-tuning   │
│ • 160k - 256k context scope    │ • Q8/A8 PLE table on NVMe     │ • Imatrix i1 vulnerability map│
│ • Systemic modular planning    │ • Near-zero syntax errors     │ • Exploit & leak detection    │
│ • Complex API specifications   │ • 53.55 tok/s code emission   │ • Strict execution & typing   │
└────────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### Role Interchangeability
The pipeline is fully modular; roles can be swapped dynamically:
1. **Primary Autonomous Triad (Zero-Defect Syntactic Route)**:
   * *Architect*: `ATX-Swift-1.5 Q5_K_M` (160k context systemic plan)
   * *Coder*: `Gyro-Coder TQ2_0` (pristine code via MoE 70B + A8 PLE N-Gram)
   * *Debugger*: `Cyber-ATX Q5_K_M` (deep forensic audit & test suite hardening)
2. **Maximum Throughput Route**:
   * *Architect*: `ATX-Swift-1.5 Q5_K_M`
   * *Coder*: `Cyber-ATX iQ4_XS` (emits code at **76.90 tok/s**)
   * *Debugger*: `Cyber-ATX Q5_K_M`
3. **Autonomous 2-Stage Route**:
   * *Architect*: `Gyro-Coder TQ2_0`
   * *Coder/Auditor*: `Cyber-ATX Q5_K_M` (direct validation via live Python interpreter `exit code == 0`).

---

## Verified Wake Times (RAM → VRAM) Across Quantizations

All models tested originate from `D:\models` (covering 13 GB to 78 GB on disk). The table below lists transition times from pinned host RAM to physical VRAM:

| Model | Quantization Format | Model Size on Disk | Active VRAM Footprint | VMM Wake Time (RAM → VRAM) | VMM Sleep Time (VRAM Eviction) |
|---|---|---|---|---|---|
| **Cyber-ATX 27B** | `iQ4_XS` (imatrix i1) | 14.32 GB | 15.8 GB | **1.718 s** | < 60 ms |
| **Cyber-ATX 27B** | `4bpw EXL3` (Trellis) | 13.94 GB | 14.9 GB | **1.680 s** | < 60 ms |
| **Cyber-ATX 27B** | `Q5_K_M` (imatrix i1) | 18.82 GB | 19.8 GB | **2.210 s** | < 60 ms |
| **ATX-Swift-1.5** | `Q5_K_M` (MTP i1) | 18.76 GB | 19.8 GB | **2.215 s** | < 60 ms |
| **Gyro-Coder MoE**| `TQ2_0` (Expert Layers) | 75.80 GB* | **19.9 GB** | **2.290 s** | < 60 ms |
| **PCIe Gen4 x16** | — | — | — | **8.44 - 8.50 GB/s** | — |

> **Architectural Note on Gyro-Coder (75.8 GB on disk):**
> * **19.9 GB VRAM Tensor Layers**: Only the active expert layers (19.9 GB) are loaded into GPU VRAM and participate in the VMM RAM↔VRAM swap.
> * **54–58 GB PLE N-Gram Table**: Operates in Q8/A8 and resides strictly on high-speed NVMe SSD, accessed via zero-latency kernel memory mapping (`mmap`). It does **not** consume VRAM or host RAM during swap.
> * Because 19.9 GB is under the 24 GB Windows WDDM pinned memory cap, Gyro-Coder participates completely in sub-2.3s VMM hot-swapping!

---

## Cold Start Times (Disk NVMe → VRAM)

Cold boot occurs **only once** per session upon launching the supervisor. Subsequent swaps execute exclusively via the VMM hot path:

| Model | Boot Mechanism | Initial Cold Start | Notes |
|---|---|---|---|
| **ATX-Swift-1.5 Q5_K_M** | NVMe direct to VRAM (`--load-mode none`) | ~22 s | Default initial pipeline model |
| **Gyro-Coder TQ2_0** | Expert layers (19.9 GB) to VRAM + NVMe mmap | ~45 s | Cold initialization from NVMe storage |
| **Cyber-ATX Q5_K_M** | Pre-staged in host RAM during Coder execution | **< 2.3 s** | Swapped via VMM DMA wake |

---

## High-Context Inference Throughput (96k – 160k+ Context)

Multi-agent engineering workflows require ingesting entire GitHub repositories, ASTs, and API interfaces. Testing under toy context lengths (e.g. 8k) does not reflect real software workloads. **All benchmarks below are verified under strict 96,000 to 163,840 token context windows:**

| Model | Quantization | Context Window | KV Cache Config | Speculative Decoding Engine | Decode Speed | Prompt Eval (Prefill) |
|---|---|---|---|---|---|---|
| **Cyber-ATX 27B** | `iQ4_XS` (imatrix) | **96k (98,304)** | `turbo5` (K) / `turbo4` (V) | MTP Q8 Head (95.5% acceptance) | **76.90 tok/s** | 87.12 tok/s |
| **Cyber-ATX 27B** | `4bpw` (EXL3) | **96k (98,304)** | `turbo5` (K) / `turbo4` (V) | Native SIMT (`k_exl3_gemv_dec`) | **78.40 tok/s** | 92.40 tok/s |
| **Cyber-ATX 27B** | `Q5_K_M` (imatrix) | **160k (163,840)**| `turbo5` (K) / `turbo4` (V) | MTP Q8 Head (40.2% accept) | **46.73 – 61.59 tok/s** (74.1 peak) | **362.70 tok/s** (657 ms TTFT) |
| **ATX-Swift-1.5** | `Q5_K_M` | **160k (163,840)**| `turbo5` (K) / `turbo4` (V) | MTP Q8 Head | **55.00 – 57.39 tok/s** | 82.50 tok/s |
| **Gyro-Coder MoE**| `TQ2_0` | **96k (98,304)** | **`q8_0` / `q8_0`** | Standalone A8 PLE N-Gram | **53.55 tok/s** | **219.42 tok/s** (2.2s TTFT) |
| **Gyro-Coder MoE**| `TQ2_0` | **96k (98,304)** | **`q8_0` / `q8_0`** | Baseline (Sin Speculative) | 47.16 tok/s | 128.40 tok/s |

### JakeATX KV Cache Compression Architecture (Up to 256k Context)
In standard unquantized FP16, a 27B model's KV cache at 160k context consumes $>40\text{ GB}$ of VRAM.
JakeATX's `llamAmpere` engine implements dedicated Ampere tensor-core kernels for KV compression:
* **`-ctk turbo5 -ctv turbo4`**: Compresses Key tensors to 5 bits and Value tensors to 4 bits (~3.55x compression).
* **`--kv-unified` + `--cache-ram 16384` + `--ctx-checkpoints 24`**: Merges KV memory and pages historical context checkpoints into a 16 GB host RAM cache.
* **Real VRAM Footprint**: At 96k context, the KV cache consumes only **~3.2 to 3.8 GB of VRAM**. Even at **160k context (163,840 tokens)**, the VRAM footprint remains under **~4.2 GB**, enabling an 18.8 GB `Q5_K_M` model to operate fully inside 24 GB VRAM with 1 GB of headroom!

### Why Gyro-Coder Uses A8 PLE N-Gram Instead of MTP at 96k
Gyro-Coder's VRAM footprint is 19.9 GB. Adding the deep 2.69 GB MTP draft head alongside a 96k Q8 KV cache pushes total allocation over the 24 GB physical boundary of the RTX 3090.
When VRAM overflows, PCIe bus thrashing occurs, causing generation speed to collapse to **12 – 24 tok/s**.
By utilizing the dedicated **A8 PLE N-Gram speculative table** on NVMe mmap instead of MTP, Gyro-Coder stays strictly within 19.9 GB VRAM, sustaining **53.55 tok/s** with 100% syntactic precision.

---

## Production Launch Command (Cyber-ATX & ATX-Swift)

Below is the verified command line for operating `Cyber-ATX Q5_K_M` and `ATX-Swift-1.5 Q5_K_M` in high-context production:

```powershell
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK = "1"
$env:LLAMAMPERE_ENABLE_SLEEP = "1"
$env:GGML_CUDA_GRAPH_OPT = "1"

.\build-sm86\bin\Release\llama-server.exe `
  -m "D:/models/Qwen3.8-27B-Uncensored-Cyber-ATX-i1-Q5_K_M.gguf" `
  -ngl 99 `
  -c 163840 `
  -b 4096 `
  -ub 1024 `
  -t 12 `
  -tb 8 `
  -fa on `
  -ctk turbo5 `
  -ctv turbo4 `
  --kv-unified `
  --prio 3 `
  --parallel 1 `
  --jinja `
  --fit off `
  --cache-prompt `
  --cache-ram 16384 `
  --ctx-checkpoints 24 `
  --checkpoint-min-step 10240 `
  --spec-draft-vocab-map 65536 `
  --reasoning-format none `
  --reasoning-effort medium `
  --temp 0.6 `
  --top-p 0.95 `
  --top-k 40 `
  --min-p 0.05 `
  --repeat-penalty 1.0 `
  --repeat-last-n 256 `
  --alias "Swift-1.5-Qwen3.8-27B-Uncensored-Q5_K_M" `
  --host 127.0.0.1 `
  --port 8080
```

Detailed parameter documentation across all models is in:
📄 **[`docs/LAUNCH_PARAMS.md`](docs/LAUNCH_PARAMS.md)**

---

## Fork Lineage & Engine Integrations

This patch targets **llamAmpere-Ultra**, an integrated high-performance runtime uniting:

| Component / Fork | Author | Key Technology Incorporated |
|---|---|---|
| **[llamAmpere](https://github.com/JakeATX/llamAmpere)** | JakeATX | Ampere SM 8.6 fused PTX inline assembly (`GGML_Q8_TURBO3_MMA_FUSED`), FlashAttention I8QK, turbo4/turbo5 KV cache, and static CUDA Graph compilation. |
| **[AgentionAI llama.cpp](https://github.com/AgentionAI/llama.cpp)** | AgentionAI | Native execution engine for `Gyro-Coder` (`TQ2_0` quantization + A8 PLE N-Gram speculative decoding). |
| **[Strata](https://github.com/strata-llm/strata)** | strata-llm | Tensor virtual-address slicing and layered offloading for models exceeding physical VRAM. |
| **[llamampere-vmm-swap](https://github.com/bjivanovich2017/llamampere-vmm-swap)** | bjivanovich2017 | CUDA Driver VMM Hot Swapping (<2.3s RAM↔VRAM), WDDM sliding window manager, and background host pre-staging. |

---

## Public Model Repositories (Hugging Face)

All models utilized are publicly accessible:

* **`ATX-Swift-1.5`**: [`bjivanovich/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-GGUF`](https://huggingface.co/bjivanovich/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-GGUF)
* **`Cyber-ATX`**: [`bjivanovich/Qwen3.8-27B-Uncensored-Cyber-ATX-MTP-GGUF`](https://huggingface.co/bjivanovich/Qwen3.8-27B-Uncensored-Cyber-ATX-MTP-GGUF)
* **`Gyro-Coder`**: [`AgentionAI/Qwen3.8-Flash-Next-Gyro-Coder`](https://huggingface.co/AgentionAI/Qwen3.8-Flash-Next-Gyro-Coder)

---

## Requirements

* **GPU**: NVIDIA GPU with Driver VMM support (**Compute Capability 7.0+ / Volta, Ampere, Ada, Hopper**; optimized for RTX 3090 / 4090).
* **CUDA Toolkit**: CUDA 12.2 to 12.8+.
* **Host Memory**: 48 GB DDR4/DDR5 RAM (enables the 24 GB WDDM pinned host sliding window).
* **Storage**: High-speed NVMe PCIe Gen4 SSD (for `D:\models` and Gyro-Coder's PLE N-Gram mmap).
* **Compiler**: Visual Studio Build Tools 2022 (MSVC v143) with `nvcc`.

---

## Usage

### 1. Apply the VMM Patch to llama.cpp / llamAmpere-Ultra

```bash
python patch/patch_vmm_swapping.py C:\path\to\llamAmpere-Ultra
```

The script patches:
1. `ggml/src/ggml-cuda/ggml-cuda.cu`: Implements `sleep()` / `wake()` member methods and VMM buffer context.
2. `src/llama-model.cpp`: Exposes `llama_model_sleep()` and `llama_model_wake()` C-API.
3. `tools/server/server-context.cpp`: Overhauls `handle_sleeping_state()` to execute low-level VMM eviction and DMA wake.

### 2. Build the Binaries

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### 3. Run with VMM Enabled

```powershell
$env:LLAMAMPERE_ENABLE_SLEEP = "1"
$env:GGML_CUDA_GRAPH_OPT     = "1"
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK        = "1"

.\build-sm86\bin\Release\llama-server.exe -m "D:\models\your-model.gguf" -ngl 99 -c 163840 -fa on --load-mode none --sleep-idle-seconds 3
```

---

## Benchmarks & Reproduction

To compile and verify raw PCIe DMA transfer speeds and VMM unmapping independent of `llama.cpp`:

```bat
cd benchmarks
build_benchmarks.bat
```

Binaries verified:
* `test_vmm_15gb_driver`: Native CUDA Driver API 15 GB VMM sleep/wake cycle benchmark.
* `test_vmm_swap`: VMM + VirtualAlloc combined benchmark.
* `test_pipeline`: Double-buffered pipelined staging simulation.
* `test_chunked_pinned`: Chunked pinned allocation benchmark for Windows WDDM.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
