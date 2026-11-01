# Benchmark Results

Hardware: RTX 3090 24GB | 48GB DDR4 | PCIe Gen4 x16 | Windows 11 WDDM | CUDA 12.x

## VMM Sleep/Wake Cycle (Full Model Size)

| Operation | Model Size | Time | Notes |
|---|---|---|---|
| Sleep (cuMemUnmap) | 14.3 GB | < 60 ms | Physical VRAM fully released |
| Sleep (cuMemUnmap) | 18.8 GB | < 60 ms | Physical VRAM fully released |
| Wake (cuMemMap + DMA) | 14.3 GB | **1.718 s** | PCIe Gen4 x16, pinned RAM source |
| Wake (cuMemMap + DMA) | 18.8 GB | **2.21 s** | PCIe Gen4 x16, pinned RAM source |

## Raw PCIe Transfer Rates

| Test | Transfer | Time | Bandwidth |
|---|---|---|---|
| test_vmm_15gb_driver | 15 GB HtoD (pinned) | ~1.76 s | **~8.5 GB/s** |
| test_vmm_swap | 15 GB HtoD (VirtualAlloc) | ~1.95 s | ~7.7 GB/s |
| test_pipeline | 15 GB double-buffered staging | ~2.1 s | ~7.1 GB/s |
| test_chunked_pinned | 15 GB chunked cudaHostAlloc | ~1.81 s | ~8.3 GB/s |

## Inference Speed (with VMM active)

| Model | Quant | Context | Speed |
|---|---|---|---|
| Cyber-ATX 27B | iQ4_XS | 96k | **76.90 tok/s** |
| Gyro-Coder (MoE) | TQ2_0 + A8 PLE mmap | 96k | **53.55 tok/s** |

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
