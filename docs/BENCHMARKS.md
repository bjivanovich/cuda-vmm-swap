# Benchmark Results

Hardware: NVIDIA GeForce RTX 3090 24GB | 48GB DDR4 | PCIe 3.0 x16 (~8.5 GB/s effective DMA) | Windows 11 WDDM | CUDA 12.x

## VMM Sleep/Wake Cycle (Full Model Size)

*Note: Transition time reflects the full hardware + driver cycle: host pinned memory allocation/staging, virtual address mapping (`cuMemMap`), PCIe DMA transfer, and CUDA stream synchronization.*

| Operation | Model Size | Time | Notes |
|---|---|---|---|
| Sleep (cuMemUnmap) | 14.3 GB | < 60 ms | Physical VRAM fully released |
| Sleep (cuMemUnmap) | 18.8 GB | < 60 ms | Physical VRAM fully released |
| Wake (cuMemMap + DMA) | 14.3 GB | **1.718 s** | PCIe 3.0 x16, pinned RAM source |
| Wake (cuMemMap + DMA) | 18.8 GB | **2.21 s** | PCIe 3.0 x16, pinned RAM source |

## Raw PCIe Transfer Rates

| Test | Transfer | Time | Effective Bandwidth |
|---|---|---|---|
| test_vmm_15gb_driver | 15 GB HtoD (pinned) | ~1.76 s | **~8.5 GB/s** (~68% PCIe 3.0 x16 effective) |
| test_vmm_swap | 15 GB HtoD (VirtualAlloc) | ~1.95 s | ~7.7 GB/s |
| test_pipeline | 15 GB double-buffered staging | ~2.1 s | ~7.1 GB/s |
| test_chunked_pinned | 15 GB chunked cudaHostAlloc | ~1.81 s | ~8.3 GB/s |

## Inference Speed (with VMM active)

| Model | Quant | Context | Speed |
|---|---|---|---|
| Cyber-ATX 27B | iQ4_XS | 96k | **76.90 tok/s** |
| ATX-Swift-1.5 27B | Q5_K_M + MTP | 96k | **52.53 tok/s** |
| Gyro-Coder (MoE) | TQ2_0 + A8 PLE mmap | 96k | **44.03 - 53.55 tok/s** |
| Cyber-ATX 27B | Q5_K_M + MTP + i1 | 96k | **41.66 tok/s** |

## Live End-to-End Multi-Agent Deep Telemetry (96k Context)

Captured from a live run of the autonomous 3-agent software engineering pipeline on a single NVIDIA RTX 3090 (24GB VRAM):

| Metric | Stage 1: Architect | Stage 2: Coder | Stage 3: Debugger & Auditor |
|---|---|---|---|
| **Model** | `ATX-Swift-1.5` (27B Dense) | `Gyro-Coder` (MoE 70B/3.8B) | `Cyber-ATX` (27B Dense) |
| **Quantization** | Q5_K_M + MTP | TQ2_0 + A8 PLE mmap | Q5_K_M + MTP + i1 |
| **Context Window** | 98,304 tokens (96k) | 98,304 tokens (96k) | 98,304 tokens (96k) |
| **KV Cache Type** | turbo5 / turbo4 (Unified) | Q8_0 / Q8_0 | turbo5 / turbo4 (Unified) |
| **Peak VRAM** | 22,089 MB (~21.57 GB) | 22,783 MB (~22.25 GB) | 22,149 MB (~21.63 GB) |
| **Physical VRAM Release** | < 60 ms (unmapped to 540 MB) | < 60 ms (unmapped to 540 MB) | < 60 ms (unmapped to 540 MB) |
| **VMM Swap / DMA Load** | 1.71s - 2.21s (Warm Pinned) | 1.95s - 2.29s (Warm Pinned) | 1.71s - 2.21s (Warm Pinned) |
| **Prompt Tokens Ingested** | 118 tokens | 2,186 tokens (System + Blueprint) | 728 tokens (Plan + Trace + Code) |
| **Prompt Prefill Time** | 520.10 ms (0.52 s) | 7,273.24 ms (7.27 s) | 970.49 ms (0.97 s) |
| **Prompt Ingestion Speed** | **226.88 tok/s** | **300.55 tok/s** | **750.14 tok/s** |
| **Completion Tokens Generated** | 2,048 tokens | 3,500 tokens | 3,500 tokens |
| **Decode / Generation Time** | 38,965.36 ms (38.97 s) | 79,465.59 ms (79.47 s) | 83,982.09 ms (83.98 s) |
| **Decode Speed** | **52.53 tok/s** | **44.03 tok/s** | **41.66 tok/s** |
| **Stage Output** | System Architecture Plan | Complete Source & Tests | Hardened Production System |
| **Sandbox Execution** | N/A | Exit Code 0 (0.119 s) | Exit Code 0 (Hardened Verified) |

### Disambiguation: Physical Swap vs Prefill vs Decode

To evaluate end-to-end performance accurately, the timing components are separated:
1. **Physical VMM Swap Time (`< 2.3s`)**: The physical time required by `cuMemMap` + PCIe Gen4 x16 DMA to transfer the active model layers from host pinned memory into device VRAM. This is independent of context size or prompt length.
2. **Prefill / Prompt Processing Time**: The time required to evaluate the prompt tokens. Ingesting a concise prompt (118 tokens) takes ~520 ms, while ingesting a large multi-page architecture blueprint (2,186 tokens) takes ~7.2 seconds. On SM86 with MMA-fused FlashAttention, prefill throughput reaches **up to 750 tok/s**.
3. **Decode / Token Generation Time**: The autoregressive generation phase. On Ampere SM86, dense 27B Q5_K_M models generate sustained tokens at **42-53 tok/s** with MTP, while 27B iQ4_XS reaches **76.90 tok/s**.

## Baseline Comparison

| Method | Transition Time | Notes |
|---|---|---|
| Disk reload (NVMe) | 28-60+ s | Standard llama.cpp sleep/wake |
| VMM swap (this patch) | 1.71 - 2.21 s | From pinned RAM via PCIe DMA |
| **Speedup** | **~16-30x** | |

## WDDM Pinned Memory Cap (Windows)

On Windows 11 with 48 GB DDR4, the WDDM driver caps total pinned host memory
(cudaHostAlloc) to approximately 24 GB (~50% of physical RAM).

This means only ONE large model can be pre-staged in pinned RAM at a time when models
exceed 12 GB each. The Sliding Window invariant (see SLIDING_WINDOW.md) is required
to operate within this constraint.
