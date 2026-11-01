#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CU_CHECK(res) do { \
    CUresult err = (res); \
    if (err != CUDA_SUCCESS) { \
        const char * err_str = nullptr; \
        cuGetErrorString(err, &err_str); \
        std::cerr << "CUDA DRIVER ERROR at line " << __LINE__ << ": " << (err_str ? err_str : "unknown") << "\n"; \
        return 1; \
    } \
} while (0)

#define CUDA_CHECK(res) do { \
    cudaError_t err = (res); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA RUNTIME ERROR at line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
        return 1; \
    } \
} while (0)

int main() {
    std::cout << "=== FULL 15 GB VMM SWAP BENCHMARK (NATIVE DRIVER API) ===\n";
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    const size_t test_bytes = 15ULL * 1024 * 1024 * 1024; // 15 GB
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t aligned_size = ((test_bytes + granularity - 1) / granularity) * granularity;

    std::cout << "1. Allocating 15 GB Pinned Host RAM...\n";
    void * host_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host_pinned, aligned_size, cudaHostAllocPortable));
    std::cout << "   Pinned Host RAM ready at " << host_pinned << "\n";

    std::cout << "2. Reserving and Mapping 15 GB VRAM via CUDA VMM...\n";
    CUdeviceptr vaddr = 0;
    CU_CHECK(cuMemAddressReserve(&vaddr, aligned_size, 0, 0, 0));

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));
    CU_CHECK(cuMemMap(vaddr, aligned_size, 0, handle, 0));
    CU_CHECK(cuMemRelease(handle));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(vaddr, aligned_size, &access, 1));
    std::cout << "   15 GB VRAM mapped at 0x" << std::hex << vaddr << std::dec << "\n";

    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    // --- SLEEP BENCHMARK ---
    std::cout << "\n>>> [BENCHMARK] Simulating SLEEP: Unmapping physical VRAM...\n";
    auto ts0 = std::chrono::high_resolution_clock::now();
    CU_CHECK(cuMemUnmap(vaddr, aligned_size));
    auto ts1 = std::chrono::high_resolution_clock::now();
    double sleep_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    std::cout << ">>> SLEEP TOOK: " << sleep_ms << " ms! Physical VRAM = 0 MB.\n";

    // --- WAKE BENCHMARK (RAM -> VRAM TRANSFER) ---
    std::cout << "\n>>> [BENCHMARK] Simulating WAKE: Remapping VRAM + Streaming 15 GB from RAM...\n";
    auto tw0 = std::chrono::high_resolution_clock::now();

    // Remap physical VRAM
    CU_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));
    CU_CHECK(cuMemMap(vaddr, aligned_size, 0, handle, 0));
    CU_CHECK(cuMemRelease(handle));
    CU_CHECK(cuMemSetAccess(vaddr, aligned_size, &access, 1));

    // Stream from Pinned RAM to VRAM using DMA
    CU_CHECK(cuMemcpyHtoDAsync(vaddr, host_pinned, aligned_size, stream));
    CU_CHECK(cuStreamSynchronize(stream));

    auto tw1 = std::chrono::high_resolution_clock::now();
    double wake_ms = std::chrono::duration<double, std::milli>(tw1 - tw0).count();
    double wake_sec = wake_ms / 1000.0;
    double gb = (aligned_size / (1024.0 * 1024.0 * 1024.0));
    double rate_gbps = gb / wake_sec;

    std::cout << "\n======================================================\n";
    std::cout << ">>> 15 GB WAKE RESULTS:\n";
    std::cout << ">>> Payload:       " << gb << " GB\n";
    std::cout << ">>> TOTAL TIME:    " << wake_sec << " SECONDS (" << wake_ms << " ms)!\n";
    std::cout << ">>> PCIe Speed:    " << rate_gbps << " GB/s\n";
    std::cout << "======================================================\n";

    // Cleanup
    CU_CHECK(cuMemUnmap(vaddr, aligned_size));
    CU_CHECK(cuMemAddressFree(vaddr, aligned_size));
    CUDA_CHECK(cudaFreeHost(host_pinned));
    CU_CHECK(cuStreamDestroy(stream));
    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
    return 0;
}
