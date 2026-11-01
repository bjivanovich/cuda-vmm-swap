#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    size_t total_size = 18853396480ULL; // 17.56 GiB
    size_t chunk_size = 1024ULL * 1024 * 1024; // 1 GB chunks
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;

    std::cout << "Testing chunked cudaHostAlloc (" << num_chunks << " chunks of " << chunk_size / (1024*1024) << " MB)...\n";
    std::vector<void*> chunks;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    for (size_t i = 0; i < num_chunks; i++) {
        size_t this_chunk = (i == num_chunks - 1) ? (total_size - i * chunk_size) : chunk_size;
        void * ptr = nullptr;
        cudaError_t err = cudaHostAlloc(&ptr, this_chunk, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            std::cerr << "Failed chunk " << i << ": " << cudaGetErrorString(err) << " (code=" << (int)err << ")\n";
            ok = false;
            break;
        }
        chunks.push_back(ptr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ok) {
        double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "SUCCESS! Allocated " << chunks.size() << " pinned chunks in " << alloc_ms << " ms!\n";

        // Test GPU transfer speed for 15 GB
        size_t test_bytes = 15ULL * 1024 * 1024 * 1024;
        void * dev_mem = nullptr;
        cudaMalloc(&dev_mem, test_bytes);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::cout << "Streaming 15 GB from chunked pinned memory to GPU...\n";
        auto t_dma_0 = std::chrono::high_resolution_clock::now();
        size_t remaining = test_bytes;
        size_t dev_offset = 0;
        for (size_t i = 0; i < chunks.size() && remaining > 0; i++) {
            size_t c_size = (remaining > chunk_size) ? chunk_size : remaining;
            char * dev_dst = ((char*)dev_mem) + dev_offset;
            cudaMemcpyAsync(dev_dst, chunks[i], c_size, cudaMemcpyHostToDevice, stream);
            dev_offset += c_size;
            remaining -= c_size;
        }
        cudaStreamSynchronize(stream);
        auto t_dma_1 = std::chrono::high_resolution_clock::now();
        double dma_sec = std::chrono::duration<double>(t_dma_1 - t_dma_0).count();
        std::cout << "15 GB Chunked Pinned Transfer: " << dma_sec << " s (" << (15.0 / dma_sec) << " GB/s)!\n";

        cudaStreamDestroy(stream);
        cudaFree(dev_mem);
    }

    for (void * ptr : chunks) {
        cudaFreeHost(ptr);
    }
    return 0;
}
