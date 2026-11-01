# Architecture: CUDA VMM Sleep/Wake

## Problem

Standard `cudaMalloc` allocations are tied to the CUDA context. To free physical VRAM
from one model, the common approach is to call `llama_free()`, which destroys the model,
the CUDA context, and the computation graph. Reloading a model from disk then takes
28-60+ seconds depending on storage speed and model size.

This is unacceptable for sequential multi-agent pipelines where models must alternate
in and out of VRAM frequently.

## Solution: CUDA Virtual Memory Management (cuMem* Driver API)

CUDA 10.2+ exposes the Virtual Memory Management (VMM) API in the driver. It separates
the **virtual address space** from the **physical VRAM backing**.

### Key insight

The computation graph in llama.cpp holds pointers to specific virtual addresses. If those
virtual addresses remain constant across evictions, the graph never needs to be rebuilt.
Only the physical mapping behind the address changes.

### Allocation lifecycle

```
Startup (once per model):
  cuMemAddressReserve(vaddr, model_size)  <- Reserve virtual address, NO physical VRAM used
  cuMemCreate(handle, model_size)          <- Allocate physical VRAM
  cuMemMap(vaddr, model_size, handle)      <- Map physical VRAM to virtual address
  cuMemSetAccess(...)                      <- Mark as read/write
  cudaMemcpy weights -> vaddr             <- Load weights from disk/RAM

Sleep (evict from VRAM, <60ms):
  cudaMemcpy vaddr -> host_mirror         <- Copy current weights to pinned RAM (if dirty)
  cuMemUnmap(vaddr, model_size)           <- Release PHYSICAL VRAM backing
  # vaddr still exists, graph still valid, CUDA context still alive
  # Physical VRAM is now FREE and can be used by another model

Wake (restore to VRAM, ~2s for 15-18GB on PCIe Gen4):
  cuMemCreate(handle, model_size)          <- Allocate fresh physical VRAM
  cuMemMap(vaddr, model_size, handle)      <- Map to the SAME virtual address as before
  cuMemSetAccess(...)
  cudaMemcpyAsync(vaddr, host_mirror, ...) <- Stream weights back from pinned RAM via PCIe DMA
  cudaStreamSynchronize(...)
  # Model is live again, graph unchanged, zero reconstruction cost
```

## Files Modified in llama.cpp

### `ggml/src/ggml-cuda/ggml-cuda.cu`

**Changes:**
- `ggml_backend_cuda_buffer_context` struct: added `size`, `aligned_size`, `host_mirror`,
  `is_vmm`, `is_asleep` fields and `sleep()` / `wake()` method declarations.
- `ggml_backend_cuda_buffer_context::sleep()`: copies weights to `host_mirror` (pinned RAM),
  then calls `cuMemUnmap` to release physical VRAM.
- `ggml_backend_cuda_buffer_context::wake()`: re-allocates physical VRAM with `cuMemCreate`,
  remaps to the original virtual address, streams weights back via async DMA.
- `ggml_backend_cuda_buffer_type_alloc_buffer`: when `LLAMAMPERE_ENABLE_SLEEP=1` and the
  device supports VMM, uses `cuMemAddressReserve` + `cuMemMap` instead of `cudaMalloc`.
- Destructor updated to handle VMM buffers: unmaps and frees address range, frees
  host_mirror.

### `src/llama-model.cpp`

**Changes:**
- Added `llama_model_sleep(llama_model*)` function: iterates all CUDA backend buffers in
  the model's pimpl and calls `ggml_backend_cuda_buffer_sleep` on each.
- Added `llama_model_wake(llama_model*)` function: same pattern, calls
  `ggml_backend_cuda_buffer_wake`.

### `tools/server/server-context.cpp`

**Changes:**
- `handle_sleeping_state(bool new_state)`: replaces the `destroy()` / `load_model()` cycle
  with `llama_model_sleep()` / `llama_model_wake()` when available. Falls back to original
  behavior if VMM is not active.

## Environment Variable

| Variable | Value | Effect |
|---|---|---|
| `LLAMAMPERE_ENABLE_SLEEP` | `1` or `true` | Enable VMM allocations for all large (>32 MB) CUDA buffers |
| (not set / `0`) | — | Standard `cudaMalloc` behavior, no VMM, no sleep/wake |

## GPU Requirements

VMM support requires compute capability 7.0+ (Volta and later). Confirmed working on:
- RTX 3090 (Ampere, cc 8.6)
- Should work on: RTX 20xx / 30xx / 40xx, A100, H100, RTX 4000/6000 Ada
