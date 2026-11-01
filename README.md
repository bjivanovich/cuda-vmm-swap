# llamampere-vmm-swap

**Fast LLM model swapping between RAM and VRAM in under 2 seconds using CUDA Virtual Memory Management (VMM).**

Patch and benchmarks for `llama.cpp` — verified on a single RTX 3090 (24 GB VRAM) with a sequential 3-agent pipeline: Architect → Coder → Debugger.

---

## Results

| Model | Size | Wake Time (RAM→VRAM) | Sleep Time (VRAM free) |
|---|---|---|---|
| Cyber-ATX 27B iQ4_XS | 14.3 GB | **1.718 s** | < 60 ms |
| ATX-Swift / Cyber-ATX Q5_K_M | 18.8 GB | **2.21 s** | < 60 ms |
| PCIe Gen4 x16 bandwidth | — | **8.5 GB/s** sustained | — |

**Before this patch:** model reloading from disk took 28+ seconds per swap.  
**After this patch:** sub-2-second hot-swapping between 3 models on a single 24 GB GPU.

---

## How It Works

### Core: CUDA Driver VMM (`cuMem*`)

Standard `cudaMalloc` allocations cannot be partially evicted without destroying the CUDA context and the process. The solution uses the low-level CUDA Driver Virtual Memory Management API:

1. **At startup:** Each model reserves its own virtual address range via `cuMemAddressReserve`. No physical VRAM is consumed yet.
2. **Wake (load into VRAM):** `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` → then async DMA from pinned host RAM via `cudaMemcpyAsync`. Physical VRAM is mapped to the pre-reserved virtual address.
3. **Sleep (evict from VRAM):** `cuMemUnmap` — releases the physical VRAM backing while keeping the virtual address range intact. Takes **< 60 ms** for any model size. The CUDA context remains live.

Because the computation graph virtual address remains constant across sleep/wake cycles, **no graph reconstruction is needed during the swap.**

### Windows WDDM Pinned Memory Limit: Sliding Window

On Windows with WDDM drivers, `cudaHostAlloc` (pinned host memory) is capped at approximately 50% of total system RAM. With 48 GB DDR4, the cap is ~24 GB — not enough to hold two large models (14.3 GB + 18.8 GB = 33.1 GB) simultaneously.

**Solution: Strict Sliding Window (max 2 models hot: 1 in VRAM + 1 pinned RAM)**

```
Phase 1:  Architect in VRAM | Coder pinned in RAM (14.3 GB)
Swap 1:   cuMemUnmap Architect (<60ms) | cuMemMap Coder, DMA 14.3 GB → VRAM (1.71s)
          -> Free Coder pinned RAM immediately (0 GB pinned)
          -> Background thread: pin Debugger (18.8 GB) into RAM while Coder runs
Swap 2:   cuMemUnmap Coder (<60ms) | cuMemMap Debugger, DMA 18.8 GB → VRAM (2.21s)
```

---

## Requirements

- NVIDIA GPU with VMM support (Pascal / Turing / Ampere / Ada, compute capability 7.0+)
- CUDA Toolkit 11.2+
- llama.cpp (any recent build from 2025+)
- Windows 10/11 (WDDM) or Linux
- Visual Studio Build Tools 2022 (for Windows benchmarks)
- nvcc

---

## Usage

### Step 1: Apply the patch to llama.cpp

```bash
# Point BASE_DIR to your llama.cpp source directory (edit the variable at the top of the script)
python patch/patch_vmm_swapping.py
```

The patch modifies three files in your llama.cpp source:
- `ggml/src/ggml-cuda/ggml-cuda.cu` — adds VMM buffer context, `sleep()` / `wake()` methods
- `src/llama-model.cpp` — exposes `llama_model_sleep()` / `llama_model_wake()` API
- `tools/server/server-context.cpp` — integrates fast VMM sleep/wake into the server sleeping state handler

### Step 2: Build llama.cpp with CUDA (standard process)

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### Step 3: Enable VMM sleep at runtime

```bash
set LLAMAMPERE_ENABLE_SLEEP=1
./llama-server [your usual args]
```

The server will now use `cuMemUnmap` / `cuMemMap` for sleeping/waking instead of full model teardown and reload.

---

## Benchmarks

To reproduce the raw transfer benchmarks (no llama.cpp dependency):

```bash
cd benchmarks
build_benchmarks.bat
```

This compiles and runs:
- `test_vmm_15gb_driver` — Full VMM sleep/wake cycle on 15 GB using native CUDA Driver API
- `test_vmm_swap` — VMM + VirtualAlloc combined test
- `test_pipeline` — Double-buffered pipelined 15 GB staging transfer
- `test_chunked_pinned` — Chunked `cudaHostAlloc` test (WDDM cap workaround)
- `test_host_mirror` — `VirtualAlloc` + `cudaHostRegister` speed test

---

## Hardware Tested

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB GDDR6X) |
| System RAM | 48 GB DDR4 |
| PCIe | Gen4 x16 |
| OS | Windows 11 / WDDM |
| CUDA | 12.x |

---

## Pipeline Models Tested

| Role | Model | Size | Inference Speed |
|---|---|---|---|
| Architect | ATX-Swift-1.5 Q5_K_M | 18.8 GB | — |
| Coder | Gyro-Coder TQ2_0 + A8 PLE mmap | 19.9 GB VRAM + 54.4 GB NVMe | 53.55 tok/s |
| Debugger | Cyber-ATX Q5_K_M | 18.8 GB | 76.90 tok/s @ 96k ctx |

---

## License

MIT — use freely, attribution appreciated.

---

## Related

- [Reddit thread: Fast Model Swapping RAM<->VRAM (<2s) on a Single 24GB GPU](https://www.reddit.com/r/LocalLLaMA/comments/1xk6qbw/fast_model_swapping_ram_vram_2s_on_a_single_24gb/)
- [CUDA VMM docs: cuMemAddressReserve](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
