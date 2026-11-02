# Engine Parameters & Execution Flags Guide

This document details all launch parameters, environment variables, engine-specific implementations, and hardware limits used across the **llamAmpere-Ultra** multi-agent pipeline.

---

## 1. Environment Variables (Ampere & Engine Optimizations)

These environment variables configure low-level kernel execution, CUDA graph compilation, and VMM memory allocation:

| Environment Variable | Recommended Value | Engine / Fork | Hardware Scope | Technical Explanation |
| :--- | :---: | :--- | :--- | :--- |
| `LLAMAMPERE_ENABLE_SLEEP` | `1` | **llamampere-vmm-swap** | Compute 7.0+ (Ampere / Ada) | Enables low-level CUDA Driver VMM allocation (`cuMemAddressReserve` + `cuMemMap`). Required for <60ms eviction and sub-2.3s DMA hot swapping. |
| `GGML_CUDA_GRAPH_OPT` | `1` | **llamAmpere** (JakeATX) | NVIDIA SM 8.6+ (RTX 3090) | Enables static CUDA Graph capture and reuse for token decode rounds. Eliminates kernel launch overhead (reusing up to 16 CUDA graphs per session). |
| `GGML_Q8_TURBO3_MMA_FUSED`| `1` | **llamAmpere** (JakeATX) | NVIDIA Ampere (SM 8.6) | Employs fused PTX inline assembly to route Q8 / Q4 dequantization and matrix multiply directly into Ampere **Tensor Cores** (MMA), bypassing standard SIMT arithmetic. |
| `GGML_CUDA_FA_I8QK` | `1` | **llamAmpere** (JakeATX) | NVIDIA Ampere (SM 8.6) | Enables 8-bit quantized FlashAttention keys and queries. Dramatically reduces memory bandwidth pressure during attention computation across 96k - 196k context. |
| `LLAMA_SPEC_CHAIN` | `1` | **llamAmpere** / Upstream | All CUDA | Enables multi-step speculative draft chains for Multi-Token Prediction (MTP). |
| `LLAMA_SPEC_CHAIN_SAMPLED`| `1` | **llamAmpere** | All CUDA | Applies temperature-scaled sampling to the speculative draft tokens rather than greedy selection, preventing draft repetition loops in long codeblocks. |
| `LLAMA_SPEC_ADAPT_COST` | `1` | **llamAmpere** | All CUDA | Dynamically adjusts speculative draft length according to model verification cost and draft acceptance rate. |

---

## 2. CLI Flags: Detailed Parameter Reference

Below are the command-line flags passed to `llama-server.exe`:

### Memory, Context & Paged Attention
* **`-ngl 99`**
  * *Purpose*: Offload all 99 layers to GPU VRAM. Ensures zero compute falls back to CPU.
* **`-c 98304` (or `-c 131072`)**
  * *Purpose*: Sets the context window to 96k (98,304 tokens) or 128k. Essential for digesting complete multi-file project repositories, ASTs, and logs in a single prompt.
* **`-fa on`**
  * *Purpose*: Enables FlashAttention. Obligatory for context lengths >32k to avoid quadratic $O(N^2)$ VRAM explosion.
* **`--load-mode none`**
  * *Purpose*: **Critical Windows WDDM optimization**. Bypasses persistent OS `mmap` of the GGUF file in system RAM. The engine reads weights sequentially and frees temporary buffers, leaving system RAM available for the pinned VMM mirror (`cudaHostAlloc`). Prevents pagefile thrashing.
* **`-b 2048 -ub 512`**
  * *Purpose*: Sets logical batch size (`-b`) and physical micro-batch size (`-ub`). Balances prompt processing speed with peak activation memory footprint.

### KV Cache Quantization
* **`-ctk turbo5 -ctv turbo4`** (or `--spec-draft-type-k q8_0 --spec-draft-type-v q8_0`)
  * *Purpose*: Compresses the Key and Value cache tensors.
  * *Impact*:
    * For Dense 27B models (`Cyber-ATX`, `ATX-Swift`), using `turbo5` (Keys) and `turbo4` (Values) reduces 96k context memory from >28 GB (in FP16) to **~6.5 - 8.2 GB**, fitting comfortably inside the RTX 3090 alongside the 15-18 GB model weights.
    * For `Gyro-Coder TQ2_0`, the KV cache is set strictly to **`Q8_0`** (`--spec-draft-type-k q8_0 --spec-draft-type-v q8_0`). Never leave KV cache unquantized ("default"/FP16), as it would instantly trigger an Out-Of-Memory error at 96k.

### Speculative Decoding & Multi-Token Prediction (MTP)
* **`--spec-draft-vocab-map <path>`**
  * *Purpose*: Maps the model vocabulary directly to the MTP draft prediction head (e.g. `atx_65536.txt`). Prevents token mismatch errors and ensures high acceptance (>85-95%).
* **`--reasoning-format none`**
  * *Purpose*: Suppresses verbose natural language reasoning tokens in Qwen 3.8 models when generating pure code, increasing direct code throughput and MTP acceptance.

### VMM Sleep Idle Trigger
* **`--sleep-idle-seconds 3`** (or `4`)
  * *Purpose*: Automatically triggers the `cuMemUnmap` sleep transition after 3-4 seconds of inactivity following request completion, freeing up to ~20 GB of physical VRAM for the next pipeline model.

---

## 3. Recommended Production Commands Per Model

### A. Architect: `ATX-Swift-1.5 Q5_K_M` (bjivanovich)
* **Goal**: Uncensored systemic reasoning, extensive architectural design, 128k context support.
* **Engine**: `llamAmpere`

```powershell
$env:LLAMAMPERE_ENABLE_SLEEP="1"
$env:GGML_CUDA_GRAPH_OPT="1"
$env:GGML_Q8_TURBO3_MMA_FUSED="1"
$env:GGML_CUDA_FA_I8QK="1"

.\llama-server.exe `
  -m "D:\models\ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-i1-Q5_K_M.gguf" `
  -ngl 99 `
  -c 131072 `
  -b 2048 -ub 512 `
  -fa on `
  -ctk turbo5 -ctv turbo4 `
  --load-mode none `
  --spec-draft-vocab-map "D:\llamAmpere\docs\mtp-vocab\atx_65536.txt" `
  --temp 0.6 --top-p 0.95 --top-k 40 --min-p 0.05 `
  --sleep-idle-seconds 3 `
  --host 127.0.0.1 --port 8089
```

---

### B. Coder: `Gyro-Coder TQ2_0` (AgentionAI)
* **Goal**: High-fidelity programming idioms with near-zero syntax errors via 70B MoE knowledge and A8 PLE N-Gram table.
* **Engine**: `AgentionAI llama.cpp`
* **Important**: MTP draft head is **disabled** for Gyro-Coder at 96k context. Combining 19.9 GB of VRAM weights + 2.69 GB MTP + 96k Q8 KV cache exceeds the 24 GB hardware limit, degrading performance to 12-24 tok/s. The A8 PLE N-Gram table on NVMe mmap handles speculative acceleration instead, sustaining **53.55 tok/s**.

```powershell
$env:LLAMAMPERE_ENABLE_SLEEP="1"
$env:GGML_CUDA_GRAPH_OPT="1"
$env:GGML_Q8_TURBO3_MMA_FUSED="1"

.\llama-server.exe `
  -m "D:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf" `
  -ngl 99 `
  -c 98304 `
  -b 2048 -ub 512 `
  -fa on `
  --spec-draft-type-k q8_0 `
  --spec-draft-type-v q8_0 `
  --moe-profile python `
  --load-mode none `
  --temp 0.6 --top-p 0.95 --top-k 40 --min-p 0.05 `
  --sleep-idle-seconds 3 `
  --host 127.0.0.1 --port 8089
```

---

### C. Coder (Alternative High-Speed): `Cyber-ATX iQ4_XS` (bjivanovich)
* **Goal**: Blazing fast code throughput (**76.90 tok/s**) at 96k context. Leaves ~9.6 GB of VRAM free.
* **Engine**: `llamAmpere`

```powershell
$env:LLAMAMPERE_ENABLE_SLEEP="1"
$env:GGML_CUDA_GRAPH_OPT="1"
$env:GGML_Q8_TURBO3_MMA_FUSED="1"
$env:GGML_CUDA_FA_I8QK="1"

.\llama-server.exe `
  -m "D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-IQ4_XS-M.gguf" `
  -ngl 99 `
  -c 98304 `
  -b 4096 -ub 1024 `
  -fa on `
  -ctk turbo5 -ctv turbo4 `
  --load-mode none `
  --spec-draft-vocab-map "D:\llamAmpere\docs\mtp-vocab\atx_65536.txt" `
  --reasoning-format none `
  --temp 0.6 --top-p 0.95 --top-k 40 --min-p 0.05 `
  --sleep-idle-seconds 3 `
  --host 127.0.0.1 --port 8089
```

---

### D. Debugger & Security Auditor: `Cyber-ATX Q5_K_M` (bjivanovich)
* **Goal**: Deep vulnerability auditing, memory safety, race condition prevention, and strict execution error fixing.
* **Engine**: `llamAmpere`

```powershell
$env:LLAMAMPERE_ENABLE_SLEEP="1"
$env:GGML_CUDA_GRAPH_OPT="1"
$env:GGML_Q8_TURBO3_MMA_FUSED="1"
$env:GGML_CUDA_FA_I8QK="1"

.\llama-server.exe `
  -m "D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-Q5_K_M.gguf" `
  -ngl 99 `
  -c 98304 `
  -b 2048 -ub 512 `
  -fa on `
  -ctk turbo5 -ctv turbo4 `
  --load-mode none `
  --spec-draft-vocab-map "D:\llamAmpere\docs\mtp-vocab\atx_65536.txt" `
  --temp 0.2 --top-p 0.95 --top-k 20 --min-p 0.05 `
  --sleep-idle-seconds 3 `
  --host 127.0.0.1 --port 8089
```

---

## 4. Hardware & Software Requirements

* **GPU Architecture**: NVIDIA Ampere SM 8.6 (e.g. GeForce RTX 3090, RTX A6000) or Ada Lovelace SM 8.9.
  * *Note*: The PTX fused kernels (`GGML_Q8_TURBO3_MMA_FUSED`) are tailored for Ampere Tensor Cores.
* **CUDA Version**: CUDA Toolkit 12.2 to 12.8+.
* **Host OS**: Windows 11 (requires `--load-mode none` to operate within WDDM 50% pinned host memory cap) or Linux (x86_64).
