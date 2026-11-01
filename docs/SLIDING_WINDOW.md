# Sliding Window Invariant

## Problem: Windows WDDM Pinned Memory Cap

On Windows with WDDM drivers, `cudaHostAlloc` (pinned host memory used as the RAM-side
staging buffer for VMM wake operations) is capped at approximately 50% of total physical
system RAM.

With 48 GB DDR4, the hard cap is approximately **24 GB**.

Our 3-model pipeline requires:
- Architect: 18.8 GB
- Coder: 14.3 GB (iQ4_XS) or 19.9 GB (TQ2_0)
- Debugger: 18.8 GB

Total if all three were pinned simultaneously: 52-58 GB — more than double the 24 GB cap.
Attempting to pin all three causes `cudaErrorMemoryAllocation` (error code 2).

## Solution: Strict Sliding Window

**Invariant: At most 2 models are hot at any time. Exactly 1 in VRAM, exactly 1 in pinned RAM.**

The third model is either not yet pinned (waiting in pageable RAM or NVMe mmap) or
its pinned allocation has been freed after it successfully woke into VRAM.

## Full Pipeline Execution

```
Initial state:
  Architect  -> VRAM (18.8 GB mapped)
  Coder      -> pinned RAM (14.3 GB cudaHostAlloc)
  Debugger   -> pageable RAM or NVMe mmap (NOT pinned)

Step 1: Architect generates task spec
  Architect in VRAM -> runs inference

Swap 1: Architect -> sleep, Coder -> wake
  cuMemUnmap Architect          [< 60 ms, VRAM freed]
  cuMemCreate + cuMemMap Coder  [immediate]
  cudaMemcpyAsync Coder RAM->VRAM [1.71s at 8.5 GB/s]
  cudaFreeHost(coder_host_mirror) [pinned RAM drops to 0 GB]

Background task (while Coder runs inference, ~25-45s):
  cudaHostAlloc(debugger_host_mirror, 18.8 GB) [pinned, 18.8 GB < 24 GB cap OK]
  Load Debugger weights into debugger_host_mirror from pageable RAM or NVMe

Step 2: Coder generates code
  Coder in VRAM -> runs inference
  Debugger being staged in background

Swap 2: Coder -> sleep, Debugger -> wake
  cuMemUnmap Coder              [< 60 ms, VRAM freed]
  cuMemCreate + cuMemMap Debugger [immediate]
  cudaMemcpyAsync Debugger RAM->VRAM [2.21s at 8.5 GB/s]
  cudaFreeHost(debugger_host_mirror) [pinned RAM drops to 0 GB]

Background task (while Debugger runs, ~20-40s):
  Reload Architect into pinned RAM (next cycle)

Step 3: Debugger reviews and fixes code
  Debugger in VRAM -> runs inference
```

## Pinned RAM Usage Timeline

```
Time    | VRAM          | Pinned RAM           | Total Pinned
--------|---------------|----------------------|-------------
t=0     | Architect     | Coder (14.3 GB)      | 14.3 GB
t=Swap1 | Coder         | 0 GB (freed)         | 0 GB
t=bg1   | Coder         | Debugger (18.8 GB)   | 18.8 GB
t=Swap2 | Debugger      | 0 GB (freed)         | 0 GB
t=bg2   | Debugger      | Architect (18.8 GB)  | 18.8 GB
```

At no point does pinned RAM exceed 18.8 GB, well within the 24 GB WDDM cap.

## Linux Note

On Linux, the pinned memory cap is much higher (often 80%+ of RAM or effectively
unlimited depending on kernel and driver configuration). The Sliding Window is still
a good practice to avoid excessive RAM pressure, but the `cudaErrorMemoryAllocation`
failure mode is Windows-specific.
