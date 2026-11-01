#include <cuda.h>
#include <cuda_runtime.h>
#include <windows.h>
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
    std::cout << "=== 15 GB RAM -> VRAM SPEED TEST WITH VMM & VIRTUALALLOC ===\n";
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

    std::cout << "Allocating 15 GB Host RAM via VirtualAlloc...\n";
    void * host_ram = VirtualAlloc(NULL, aligned_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!host_ram) {
        std::cerr << "VirtualAlloc failed! GetLastError: " << GetLastError() << "\n";
        return 1;
    }
    std::cout << "Host RAM allocated successfully at " << host_ram << "\n";

    std::cout << "Reserving 15 GB VRAM virtual address...\n";
    CUdeviceptr vaddr = 0;
    CU_CHECK(cuMemAddressReserve(&vaddr, aligned_size, 0, 0, 0));

    std::cout << "Mapping physical VRAM...\n";
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));
    CU_CHECK(cuMemMap(vaddr, aligned_size, 0, handle, 0));
    CU_CHECK(cuMemRelease(handle));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(vaddr, aligned_size, &access, 1));
    std::cout << "VRAM mapped successfully.\n";

    // Measure transfer time from Host RAM to VRAM
    std::cout << "Measuring raw 15 GB Host RAM -> VRAM transfer...\n";
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync((void*)vaddr, host_ram, aligned_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double gb = (aligned_size / (1024.0 * 1024.0 * 1024.0));
    std::cout << ">>> 15 GB TRANSFER RESULT:\n";
    std::cout << ">>> Transferred: " << gb << " GB\n";
    std::cout << ">>> Time: " << sec << " SECONDS!\n";
    std::cout << ">>> Speed: " << (gb / sec) << " GB/s\n";

    // Test unmap (sleep)
    auto ts0 = std::chrono::high_resolution_clock::now();
    CU_CHECK(cuMemUnmap(vaddr, aligned_size));
    auto ts1 = std::chrono::high_resolution_clock::now();
    std::cout << ">>> SLEEP (cuMemUnmap): " << std::chrono::duration<double, std::milli>(ts1 - ts0).count() << " ms!\n";

    // Cleanup
    CU_CHECK(cuMemAddressFree(vaddr, aligned_size));
    VirtualFree(host_ram, 0, MEM_RELEASE);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
    std::cout << "=== FINISHED SUCCESSFULLY ===\n";
    return 0;
}
