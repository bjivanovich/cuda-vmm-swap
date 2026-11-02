# Engine Parameters & Execution Flags Guide

This document details the production launch parameters, environment variables, engine-specific implementations, and memory architectures utilized across the **llamAmpere-Ultra** multi-agent pipeline.

---

## 1. Environment Variables (Ampere SM 8.6 & Engine Optimizations)

These environment variables configure low-level kernel execution, CUDA graph compilation, and VMM memory allocation:

| Environment Variable | Recommended Value | Engine / Fork | Hardware Scope | Technical Explanation |
| :--- | :---: | :--- | :--- | :--- |
| `LLAMAMPERE_ENABLE_SLEEP` | `1` | **llamampere-vmm-swap** | Compute 7.0+ (Ampere / Ada) | Enables low-level CUDA Driver VMM allocation (`cuMemAddressReserve` + `cuMemMap`). Required for <60ms eviction and sub-2.3s DMA hot swapping. |
| `GGML_Q8_TURBO3_MMA_FUSED`| `1` | **llamAmpere** (JakeATX) | NVIDIA Ampere (SM 8.6) | Employs fused PTX inline assembly to route Q8 / Q4 dequantization and matrix multiply directly into Ampere **Tensor Cores** (MMA), bypassing standard SIMT arithmetic. |
| `GGML_CUDA_FA_I8QK` | `1` | **llamAmpere** (JakeATX) | NVIDIA Ampere (SM 8.6) | Enables 8-bit quantized FlashAttention keys and queries. Dramatically reduces memory bandwidth pressure during attention computation across 96k - 256k context. |
| `GGML_CUDA_GRAPH_OPT` | `1` | **llamAmpere** (JakeATX) | NVIDIA SM 8.6+ (RTX 3090) | Enables static CUDA Graph capture and reuse for token decode rounds. Eliminates kernel launch overhead (reusing up to 16 CUDA graphs per session). |
| `LLAMA_SPEC_CHAIN` | `1` | **llamAmpere** / Upstream | All CUDA | Enables multi-step speculative draft chains for Multi-Token Prediction (MTP). |
| `LLAMA_SPEC_CHAIN_SAMPLED`| `1` | **llamAmpere** | All CUDA | Applies temperature-scaled sampling to the speculative draft tokens rather than greedy selection, preventing draft repetition loops in long codeblocks. |
| `LLAMA_SPEC_ADAPT_COST` | `1` | **llamAmpere** | All CUDA | Dynamically adjusts speculative draft length according to model verification cost and draft acceptance rate. |

---

## 2. CLI Flags: Detailed Parameter Reference

Below is the verified breakdown of parameters passed to `llama-server.exe`:

### High-Context Allocation (160k - 256k) & Memory Paging
* **`-c 163840` (or up to `-c 262144`)**
  * *Purpose*: Allocates a massive 160k (163,840 tokens) or 256k context window. Essential for ingesting full GitHub codebases, dependencies, and complex architectures in a single prompt.
* **`-fa on`**
  * *Purpose*: Enables FlashAttention. Obligatory for contexts $>32\text{k}$ to prevent $O(N^2)$ VRAM explosion.
* **`--kv-unified`**
  * *Purpose*: Merges the KV allocations into a single unified contiguous memory pool, eliminating memory fragmentation across lengthy generation passes.
* **`--load-mode none`**
  * *Purpose*: **Critical Windows WDDM optimization**. Bypasses persistent OS `mmap` of the GGUF file in system RAM. The engine reads weights sequentially and frees temporary buffers, leaving system RAM available for the pinned VMM mirror (`cudaHostAlloc`) and the tiered prompt cache. Prevents Windows pagefile thrashing.
* **`-b 4096 -ub 1024`**
  * *Purpose*: Sets logical batch size (`-b 4096`) and physical micro-batch size (`-ub 1024`). Delivers maximum prompt prefill throughput (up to 362 tok/s) without exceeding activation buffers.
* **`-t 12 -tb 8`**
  * *Purpose*: 12 CPU worker threads for prompt processing and AST parsing, with 8 dedicated batch evaluation threads.
* **`--fit off`**
  * *Purpose*: Disables automatic context window truncation. Forces the engine to respect the exact requested context size.

### JakeATX KV Cache Compression (`turbo5` / `turbo4` & 4bpw)
In standard FP16, a 27B model's KV cache at 160k context would consume over **40 GB of VRAM**, making single-GPU execution physically impossible.
JakeATX's `llamAmpere` engine implements high-compression KV cache kernels:
* **`-ctk turbo5 -ctv turbo4`** (Key = 5-bit, Value = 4-bit)
  * *Impact*: Compresses the KV cache by **~3.55x**.
  * Combined with `--kv-unified` and tiered checkpointing, **resident VRAM consumption at 96k is only ~3.2 - 3.8 GB**, and even at **160k context (163,840 tokens)**, the VRAM footprint stays under **~4.2 GB**!
  * This allows the 18.8 GB `Cyber-ATX Q5_K_M` to run at full 160k context inside the 24 GB RTX 3090 ($18.8\text{ GB weights} + 4.2\text{ GB KV} = 23.0\text{ GB total VRAM}$).

### Tiered Prompt Caching & Checkpointing
* **`--cache-prompt`**
  * *Purpose*: Retains prompt evaluation states across conversational rounds so repeated instructions, system prompts, and project files do not need to be re-evaluated.
* **`--cache-ram 16384`**
  * *Purpose*: Allocates a dedicated 16 GB buffer in host RAM for prompt and KV checkpoints.
* **`--ctx-checkpoints 24 --checkpoint-min-step 10240`**
  * *Purpose*: Creates 24 snapshot checkpoints spaced every 10,240 tokens. When context expands beyond active working memory, inactive historical context is stored in host RAM rather than wasting GPU VRAM, allowing instant context recovery.

### Speculative Decoding & Multi-Token Prediction (MTP)
* **`--spec-draft-vocab-map 65536` (or `auto`)**
  * *Purpose*: Maps the vocabulary dynamically to the high-precision Q8_0 MTP draft head. Guarantees 85% to 95.5% speculative token acceptance in structured code.
* **`--spec-draft-n-min-adaptive 3`**
  * *Purpose*: (JakeATX / llamAmpere) Sets the minimum adaptive speculation window to 3 tokens per forward pass. Prevents the drafter from under-speculating during continuous sequences of code tokens.
* **`--spec-draft-n-max 5`**
  * *Purpose*: (JakeATX / llamAmpere) Caps the maximum speculative draft depth to 5 tokens. Optimizes the balance between draft computation and base-model verification passes, preventing rollback penalties if draft prediction diverges.
* **`--spec-draft-p-min 0`**
  * *Purpose*: (JakeATX / llamAmpere) Disables early probability cutoffs during draft exploration, permitting full candidate speculation depth.
* **`--reasoning-format none --reasoning-effort medium`**
  * *Purpose*: Suppresses verbose natural language reasoning markers in Qwen 3.8 models, channeling compute directly into code generation.

### Sampling & Token Controls
* **`--temp 0.6 --top-p 0.95 --top-k 40 --min-p 0.05`**
  * *Purpose*: The verified sweet spot for code generation. Eliminates syntax hallucinations while retaining algorithmic creativity.
* **`--repeat-penalty 1.0 --repeat-last-n 256`**
  * *Purpose*: Clean code generation without penalizing necessary repeated identifiers, loops, or standard variable declarations.

---

## 3. Production Launch Configurations

### A. Architect: `ATX-Swift-1.5 Q5_K_M` (bjivanovich)
* **Role**: Uncensored systemic reasoning, deep modular architecture, and API design.
* **Engine**: `llamAmpere` (build-sm86)

```powershell
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK = "1"
$env:LLAMAMPERE_ENABLE_SLEEP = "1"
$env:GGML_CUDA_GRAPH_OPT = "1"

.\build-sm86\bin\Release\llama-server.exe `
  -m "D:/models/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-i1-Q5_K_M.gguf" `
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
  --spec-draft-n-min-adaptive 3 `
  --spec-draft-n-max 5 `
  --spec-draft-p-min 0 `
  --reasoning-format none `
  --reasoning-effort medium `
  --temp 0.6 `
  --top-p 0.95 `
  --top-k 40 `
  --min-p 0.05 `
  --repeat-penalty 1.0 `
  --repeat-last-n 256 `
  --alias "ATX-Swift-1.5-Qwen3.8-27B-Uncensored-Q5_K_M" `
  --host 127.0.0.1 `
  --port 8080
```

---

### B. Coder: `Gyro-Coder TQ2_0` (AgentionAI)
* **Role**: Synthesis of clean, idiomatic code with near-zero syntax errors via 70B MoE knowledge and A8 PLE N-Gram table.
* **Engine**: `AgentionAI llama.cpp`
* **VRAM**: 19.9 GB (expert layers). N-Gram table (~54-58 GB) accessed on NVMe via `mmap`.
* **KV Cache**: `q8_0` for both Key and Value.

```powershell
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_GRAPH_OPT = "1"
$env:LLAMAMPERE_ENABLE_SLEEP = "1"

.\build-sm86\bin\Release\llama-server.exe `
  -m "D:/models/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf" `
  -ngl 99 `
  -c 98304 `
  -ctk q8_0 `
  -ctv q8_0 `
  -b 2048 `
  -ub 512 `
  -t 8 `
  -tb 8 `
  -fa on `
  --ctx-checkpoints 0 `
  --parallel 1 `
  --prio 3 `
  --moe-profile python `
  --temp 0.6 `
  --top-p 0.95 `
  --top-k 40 `
  --min-p 0.05 `
  --repeat-penalty 1.05 `
  --repeat-last-n 256 `
  --alias "Qwen3.8-Flash-Next-Gyro-Coder-Baseline" `
  --host 127.0.0.1 `
  --port 8085
```

---

### C. Debugger & Security Auditor: `Cyber-ATX Q5_K_M` (bjivanovich)
* **Role**: High-precision debugging, cybersecurity analysis, memory leak detection, race condition resolution, and unit test enforcement.
* **Engine**: `llamAmpere` (build-sm86)
* **Calibration**: Imatrix `i1` over cybersecurity, network protocol, and vulnerability corpora.

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
  --spec-draft-n-min-adaptive 3 `
  --spec-draft-n-max 5 `
  --spec-draft-p-min 0 `
  --reasoning-format none `
  --reasoning-effort medium `
  --temp 0.2 `
  --top-p 0.90 `
  --top-k 20 `
  --min-p 0.05 `
  --repeat-penalty 1.08 `
  --repeat-last-n 256 `
  --alias "Cyber-ATX-27B-Uncensored-Q5_K_M" `
  --host 127.0.0.1 `
  --port 8080
```
