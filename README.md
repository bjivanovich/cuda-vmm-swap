# llamampere-vmm-swap

**Fast LLM model swapping between RAM and VRAM in under 2.3 seconds using CUDA Virtual Memory Management (VMM).**

Patch and benchmarks for a custom `llama.cpp` fork — verified on a single RTX 3090 (24 GB VRAM)
running a sequential 3-agent pipeline: Architect → Coder → Debugger, with models ranging
from 14.3 GB to 19.9 GB.

> **Credit:** The core VMM `sleep` / `wake` pattern was pioneered by Reddit user
> **u/enndeeee** in [Infernix PR #43](https://github.com/infernix/llama.cpp/pull/43)
> (`SleepableBuffer`). This repo adapts and extends that approach for Windows 11 / WDDM,
> multi-model pipelines, and the llamAmpere-Ultra fork.

---

## The Problem

Standard `llama.cpp` model swapping destroys and recreates the CUDA context on every
agent transition. On a single RTX 3090 with 24 GB VRAM:

| Bottleneck | Time |
|---|---|
| Process teardown + CUDA context destruction | 5 – 8 s |
| Cold CUDA context reinit (`cuInit`, cuBLAS, cuBLASLt) | 2.5 – 5 s |
| Tensor re-upload from disk (NVMe, 1 MB batches) | 15 – 30 s |
| **Total per swap (before this patch)** | **28 – 60+ s** |

---

## The Solution: VRAM ↔ RAM in Under 2.3 Seconds

Using the CUDA Driver VMM API (`cuMem*`), each model holds its own **permanent virtual
address range**. Physical VRAM is mapped and unmapped behind that address without ever
touching the CUDA context or the computation graph:

| Operation | Time | Mechanism |
|---|---|---|
| Sleep (evict 19 GB from VRAM) | **< 60 ms** | `cuMemUnmap` — releases physical pages, VA stays |
| Wake (restore from pinned RAM) | **1.7 – 2.3 s** | `cuMemMap` + async PCIe DMA at 8.44 GB/s |
| CUDA context and graphs | **preserved** | Zero reconstruction, zero cold init |

---

## Verified Wake Times (RAM → VRAM)

These are the models that swap in and out of VRAM in the pipeline. The Architect
(ATX-Swift-1.5) loads once at cold start directly to VRAM and does not participate
in hot swaps.

| Model | Quantization | VRAM Footprint | Pinnable in RAM | Wake Time (RAM→VRAM) | Inference Speed |
|---|---|---|---|---|---|
| Cyber-ATX 27B | iQ4_XS (imatrix) | 15.8 GB | 14.3 GB | **1.71 s** | 76.90 tok/s @ 96k ctx |
| Cyber-ATX 27B | Q5_K_M (imatrix) | 19.8 GB | 18.8 GB | **2.21 s** | 57.4 – 61.6 tok/s |
| Gyro-Coder MoE | TQ2_0 (expert layers) | **19.9 GB** | **19.9 GB** | **~2.3 s** | 49.8 – 53.5 tok/s |
| PCIe Gen4 x16 | — | — | — | 8.44 GB/s sustained | — |

> **How Gyro-Coder fits:** The model has two distinct components:
> - **Expert / tensor layers (TQ2_0):** 19.9 GB — loaded into VRAM. Fits comfortably in the RTX 3090's 24 GB.
> - **N-gram / PLE table (Q8/A8):** ~54–58 GB — stays on NVMe and is accessed via `mmap` at near-zero latency during inference. Never loaded into RAM or VRAM.
>
> Only the 19.9 GB expert layers participate in VMM hot-swap. They can be fully pre-staged
> in pinned host RAM (19.9 GB < 24 GB WDDM cap), making Gyro-Coder a full participant
> in the sliding window pipeline.

**Additional quantization benchmarks are pending.** To contribute results from your
hardware, see the [benchmarks](benchmarks/) folder.

---

## Cold Start Times (for reference)

| Model | Load Method | Cold Start Time | Notes |
|---|---|---|---|
| ATX-Swift-1.5 Q5_K_M | NVMe → VRAM (`--load-mode none`) | ~22 s | Loads once per session |
| Gyro-Coder TQ2_0 | Expert layers (19.9 GB) NVMe → VRAM; N-gram/PLE mmap | ~45 s | N-gram table (~54–58 GB) stays on disk |
| Cyber-ATX iQ4_XS | Pre-staged in pinned RAM → VMM wake | < 2 s | After initial staging |

Cold start happens once per session. All subsequent swaps use the VMM hot path.

---

## Inference Speed by Context and Parameters

tok/s varies significantly by context length, KV-cache quantization, and speculative
decoding settings. Representative measurements on RTX 3090:

| Model | Quant | Context | KV-Cache | Speculative | tok/s |
|---|---|---|---|---|---|
| Cyber-ATX | iQ4_XS | 96k | turbo5 / turbo4 | MTP (95% accept) | **76.90** |
| Cyber-ATX | iQ4_XS | 8k | turbo5 / turbo4 | MTP | **61.59** |
| Cyber-ATX | Q5_K_M | 96k | turbo5 / turbo4 | MTP | 57.4 |
| ATX-Swift-1.5 | Q5_K_M | 128k | turbo5 / turbo4 | MTP | 55 – 60 |
| Gyro-Coder | TQ2_0 | 96k | default | A8 PLE N-gram | **53.55** |
| Gyro-Coder | TQ2_0 | 64k | default | A8 PLE N-gram | 49.79 |

> These numbers use model-specific flags. See [docs/LAUNCH_PARAMS.md](docs/LAUNCH_PARAMS.md)
> for per-model recommended command lines.

---

## Pipeline Architecture

```
Agent roles (current validated configuration):

  [ARCHITECT]              [CODER]                          [DEBUGGER / AUDITOR]
  ATX-Swift-1.5 Q5_K_M    Gyro-Coder TQ2_0                 Cyber-ATX Q5_K_M
  (bjivanovich / HF)       (AgentionAI / HF)                (bjivanovich / HF)
  18.8 GB VRAM             19.9 GB VRAM + NVMe PLE mmap     18.8 GB VRAM
  Loads once at startup    Hot-swaps from NVMe / pinned RAM  Hot-swaps from pinned RAM
```

### Execution Flow

```
[VRAM]  ATX-Swift (Architect) ─────── generates spec ──► cuMemUnmap (<60ms)
                                                                  |
[RAM]   Cyber-ATX iQ4 pre-staged ─────────────────────────────► cuMemMap + DMA (1.71s)
[VRAM]                                 Cyber-ATX (Coder) ─────── writes code ──► cuMemUnmap (<60ms)
                                                                                         |
[RAM]   Cyber-ATX Q5_K_M staged in background ──────────────────────────────────────► cuMemMap + DMA (2.21s)
[VRAM]                                                 Cyber-ATX Q5_K_M (Debugger) ── audits & hardens
```

### Windows WDDM Pinned Memory Constraint

On Windows 11 with 48 GB DDR4, `cudaHostAlloc` (pinned host memory) is capped at
~50% of physical RAM (~24 GB). Two models cannot be pinned simultaneously if their
combined size exceeds this limit.

**Solution: Strict Sliding Window — max 1 model in VRAM + 1 model in pinned RAM.**

The pinned buffer of the model that just woke into VRAM is freed immediately
(`cudaFreeHost`). The next model is staged in the background while the current
agent runs inference. See [docs/SLIDING_WINDOW.md](docs/SLIDING_WINDOW.md).

---

## Fork Lineage

This patch targets **llamAmpere-Ultra**, a custom fork combining:

| Fork | Author | Contribution |
|---|---|---|
| [llamAmpere](https://github.com/JakeATX/llamAmpere) | JakeATX | Ampere-specific kernel optimizations (MMA fused, Q8 turbo, CUDA graph opt) for RTX 3090 |
| [Strata](https://github.com/strata-llm/strata) | strata-llm | VRAM offload / layer-split for models exceeding VRAM (not the primary use case here, but included for larger model support) |
| [AgentionAI llama.cpp](https://github.com/AgentionAI/llama.cpp) | AgentionAI | Engine support for Gyro-Coder (TQ2_0 + A8 PLE N-gram speculative table) |

The VMM sleep/wake patch (`patch_vmm_swapping.py`) applies on top of this merged fork.
It should also apply cleanly to upstream llama.cpp with minor adjustments.

---

## Models

All models are public on Hugging Face:

| Model | HF Repo | Quantization | Notes |
|---|---|---|---|
| ATX-Swift-1.5 | [bjivanovich/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-GGUF](https://huggingface.co/bjivanovich/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-GGUF) | Q5_K_M | Architect role. MTP head Q8_0. Abliterated. |
| Cyber-ATX | [bjivanovich/Qwen3.8-27B-Uncensored-Cyber-ATX-MTP-GGUF](https://huggingface.co/bjivanovich/Qwen3.8-27B-Uncensored-Cyber-ATX-MTP-GGUF) | iQ4_XS / Q5_K_M | Coder / Debugger. imatrix i1. Abliterated. |
| Gyro-Coder | [AgentionAI/Qwen3.8-Flash-Next-Gyro-Coder](https://huggingface.co/AgentionAI/Qwen3.8-Flash-Next-Gyro-Coder) | TQ2_0 + A8 PLE | MoE 70B (3.8B active). Requires AgentionAI fork engine. |

---

## Requirements

- NVIDIA GPU with VMM support (compute capability **7.0+ / Volta or later**; RTX 20xx / 30xx / 40xx, A100, H100)
- CUDA Toolkit 11.2+
- llamAmpere-Ultra (or upstream llama.cpp, any 2025+ build)
- Windows 10/11 (WDDM) or Linux
- Visual Studio Build Tools 2022 + nvcc (for Windows benchmarks)
- 48 GB+ system RAM recommended for the 3-model sliding window

---

## Usage

### Step 1: Apply the patch to your llama.cpp source

```bash
# Edit BASE_DIR at the top of the script, or pass as argument:
python patch/patch_vmm_swapping.py C:\path\to\llamAmpere-Ultra
```

The patch modifies three files:
- `ggml/src/ggml-cuda/ggml-cuda.cu` — VMM buffer context, `sleep()` / `wake()` methods, VMM-backed `alloc_buffer`
- `src/llama-model.cpp` — `llama_model_sleep()` / `llama_model_wake()` public API
- `tools/server/server-context.cpp` — integrates fast VMM path into `handle_sleeping_state()`

### Step 2: Build with CUDA

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### Step 3: Enable VMM sleep at runtime

```powershell
# Windows — set before launching each server instance
$env:LLAMAMPERE_ENABLE_SLEEP = "1"
$env:GGML_CUDA_GRAPH_OPT     = "1"
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK        = "1"

.\bin\Release\llama-server.exe `
    -m "D:\models\your-model.gguf" `
    -ngl 99 -c 8192 `
    -ctk turbo5 -ctv turbo4 `
    -fa on --load-mode none `
    --sleep-idle-seconds 3
```

See [docs/LAUNCH_PARAMS.md](docs/LAUNCH_PARAMS.md) for per-model recommended parameters.

---

## Benchmarks

Reproduce the raw transfer benchmarks (no llama.cpp dependency required):

```bat
cd benchmarks
build_benchmarks.bat
```

Tests included:
- `test_vmm_15gb_driver` — Full VMM sleep/wake cycle, native CUDA Driver API
- `test_vmm_swap` — VMM + VirtualAlloc combined
- `test_pipeline` — Double-buffered pipelined 15 GB staging
- `test_chunked_pinned` — Chunked `cudaHostAlloc` (WDDM cap workaround)
- `test_host_mirror` — `VirtualAlloc` + `cudaHostRegister` speed baseline

---

## Hardware Tested

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB GDDR6X) |
| System RAM | 48 GB DDR4 |
| Storage | NVMe PCIe Gen4 (models on D:\models, ~4.3 GB/s sequential) |
| PCIe | Gen4 x16 |
| OS | Windows 11 / WDDM |
| CUDA | 12.8 |
| Driver | 572.x |

---

## Known Limitations

- **Gyro-Coder N-gram / PLE table is NOT in VRAM or RAM.** The ~54–58 GB N-gram table lives on NVMe and is accessed via `mmap`. Only the 19.9 GB expert layers reside in VRAM and participate in VMM hot-swap. Cold start (~45 s) loads those expert layers from NVMe once per session.
- **Windows WDDM pinned RAM cap**: ~50% of physical RAM. With 48 GB, max pinnable at once is ~24 GB. All three models' VRAM footprints (14.3–19.9 GB each) are individually under this cap, enabling the sliding window pattern.
- **VMM requires Volta+ (cc 7.0+)**. Pascal (GTX 10xx, cc 6.x) is not supported.
- **The patch targets a specific llamAmpere-Ultra snapshot**. Upstream llama.cpp API changes may require minor adjustments to the patch script.
- **Linux**: The WDDM pinned memory limit does not apply. The sliding window is still good practice but the `cudaErrorMemoryAllocation` failure mode is Windows-specific.
- **Gyro-Coder requires the AgentionAI fork engine** for TQ2_0 + A8 PLE N-gram support. It will not run on upstream llama.cpp.

---

## Acknowledgements

- **u/enndeeee** (Reddit / r/LocalLLaMA) — Authored the original `SleepableBuffer` approach in [Infernix PR #43](https://github.com/infernix/llama.cpp/pull/43), which introduced `cuMemUnmap` / `cuMemMap` as a drop-in replacement for context teardown. This project's VMM pattern is a direct adaptation of that work to Windows and multi-model pipelines.
- **JakeATX** — llamAmpere fork with Ampere-specific kernel optimizations.
- **AgentionAI** — Gyro-Coder model and TQ2_0 / A8 PLE engine support.

---

## License

MIT — use freely, attribution appreciated.

---

## Related

- [Reddit thread: Fast Model Swapping RAM<->VRAM (<2s)](https://www.reddit.com/r/LocalLLaMA/comments/1xk6qbw/fast_model_swapping_ram_vram_2s_on_a_single_24gb/)
- [Infernix SleepableBuffer PR #43](https://github.com/infernix/llama.cpp/pull/43) — original inspiration
- [CUDA VMM docs: cuMemAddressReserve](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- [llamAmpere (JakeATX)](https://github.com/JakeATX/llamAmpere)
- [AgentionAI / Gyro-Coder](https://huggingface.co/AgentionAI/Qwen3.8-Flash-Next-Gyro-Coder)
- [llama.cpp upstream](https://github.com/ggerganov/llama.cpp)
