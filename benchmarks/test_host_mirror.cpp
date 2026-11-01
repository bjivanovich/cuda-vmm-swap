#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

int main() {
    size_t size = 18853396480ULL; // 17.56 GiB
    std::cout << "Testing VirtualAlloc of " << size / (1024*1024) << " MB...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    void * host_mem = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!host_mem) {
        std::cerr << "VirtualAlloc failed! GetLastError=" << GetLastError() << "\n";
        return 1;
    }
    std::cout << "VirtualAlloc succeeded in " 
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms at " << host_mem << "\n";

    // Allocate 1 GB on GPU to test HtoD and DtoH transfer rates
    size_t gpu_chunk = 1024ULL * 1024 * 1024;
    void * dev_mem = nullptr;
    cudaError_t err = cudaMalloc(&dev_mem, gpu_chunk);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Testing DtoH transfer (pageable VirtualAlloc)...\n";
    auto t_d2h_0 = std::chrono::high_resolution_clock::now();
    err = cudaMemcpy(host_mem, dev_mem, gpu_chunk, cudaMemcpyDeviceToHost);
    auto t_d2h_1 = std::chrono::high_resolution_clock::now();
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy DtoH failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_1 - t_d2h_0).count();
    std::cout << "DtoH 1 GB: " << d2h_ms << " ms (" << (1.0 / (d2h_ms / 1000.0)) << " GB/s)\n";

    std::cout << "Testing HtoD transfer (pageable VirtualAlloc)...\n";
    auto t_h2d_0 = std::chrono::high_resolution_clock::now();
    err = cudaMemcpy(dev_mem, host_mem, gpu_chunk, cudaMemcpyHostToDevice);
    auto t_h2d_1 = std::chrono::high_resolution_clock::now();
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy HtoD failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_1 - t_h2d_0).count();
    std::cout << "HtoD 1 GB: " << h2d_ms << " ms (" << (1.0 / (h2d_ms / 1000.0)) << " GB/s)\n";

    std::cout << "Testing cudaHostRegister on VirtualAlloc memory...\n";
    auto t_reg_0 = std::chrono::high_resolution_clock::now();
    err = cudaHostRegister(host_mem, size, cudaHostRegisterDefault);
    auto t_reg_1 = std::chrono::high_resolution_clock::now();
    if (err != cudaSuccess) {
        std::cout << "cudaHostRegister failed on full 18GB: " << cudaGetErrorString(err) << " (code=" << (int)err << ")\n";
    } else {
        std::cout << "cudaHostRegister SUCCEEDED in " 
                  << std::chrono::duration<double, std::milli>(t_reg_1 - t_reg_0).count() << " ms!\n";
        // Test pinned speed
        auto t_pin_0 = std::chrono::high_resolution_clock::now();
        err = cudaMemcpy(dev_mem, host_mem, gpu_chunk, cudaMemcpyHostToDevice);
        auto t_pin_1 = std::chrono::high_resolution_clock::now();
        double pin_ms = std::chrono::duration<double, std::milli>(t_pin_1 - t_pin_0).count();
        std::cout << "HtoD 1 GB (pinned): " << pin_ms << " ms (" << (1.0 / (pin_ms / 1000.0)) << " GB/s)\n";
        cudaHostUnregister(host_mem);
    }

    cudaFree(dev_mem);
    VirtualFree(host_mem, 0, MEM_RELEASE);
    return 0;
}
