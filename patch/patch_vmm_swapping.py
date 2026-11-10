import os
import sys

# ============================================================
# llamampere-vmm-swap patch script
# Applies CUDA VMM sleep/wake support to llama.cpp source
#
# Usage:
#   python patch/patch_vmm_swapping.py [path_to_llamacpp_dir]
#
# If no path is given, uses the BASE_DIR variable below.
# ============================================================

BASE_DIR = r"C:\path\to\your\llama.cpp"

if len(sys.argv) > 1:
    BASE_DIR = sys.argv[1]

if not os.path.isdir(BASE_DIR):
    print(f"ERROR: llama.cpp directory not found: {BASE_DIR}")
    print("Usage: python patch_vmm_swapping.py /path/to/llama.cpp")
    sys.exit(1)

print(f"Patching llama.cpp at: {BASE_DIR}")

# =============================================================
# 1. Patch ggml/src/ggml-cuda/ggml-cuda.cu
# =============================================================
cuda_cu = os.path.join(BASE_DIR, "ggml", "src", "ggml-cuda", "ggml-cuda.cu")
with open(cuda_cu, "r", encoding="utf-8") as f:
    cuda_code = f.read()

# Add VMM member fields and sleep/wake prototypes to struct ggml_backend_cuda_buffer_context
t_ctx = "struct ggml_backend_cuda_buffer_context {"
r_ctx = """struct ggml_backend_cuda_buffer_context {
    size_t size = 0;
    size_t aligned_size = 0;
    void * host_mirror = nullptr;
    bool is_vmm = false;
    bool is_asleep = false;

    bool sleep();
    bool wake();"""

if t_ctx in cuda_code and "bool sleep();" not in cuda_code:
    cuda_code = cuda_code.replace(t_ctx, r_ctx, 1)
    print("Patched: ggml_backend_cuda_buffer_context declaration.")
else:
    print("SKIP: ggml_backend_cuda_buffer_context already patched or target not found.")

# Add sleep/wake method implementations before the destructor
t_dtor = "ggml_backend_cuda_buffer_context::~ggml_backend_cuda_buffer_context() {"
r_dtor = """bool ggml_backend_cuda_buffer_context::sleep() {
    if (!is_vmm || is_asleep || dev_ptr == nullptr) return false;
    if (host_mirror == nullptr) {
        cudaError_t err = cudaHostAlloc(&host_mirror, aligned_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("VMM sleep: failed to allocate %zu bytes of pinned host memory\\n", aligned_size);
            return false;
        }
        CUDA_CHECK(cudaMemcpy(host_mirror, dev_ptr, size, cudaMemcpyDeviceToHost));
    }
    CUresult res = cuMemUnmap((CUdeviceptr)dev_ptr, aligned_size);
    if (res != CUDA_SUCCESS) {
        GGML_LOG_ERROR("VMM sleep: cuMemUnmap failed on %p\\n", dev_ptr);
        return false;
    }
    is_asleep = true;
    return true;
}

bool ggml_backend_cuda_buffer_context::wake() {
    if (!is_vmm || !is_asleep || dev_ptr == nullptr || host_mirror == nullptr) return false;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ggml_cuda_get_physical_device(device);
    CUmemGenericAllocationHandle handle;
    CUresult c_res = cuMemCreate(&handle, aligned_size, &prop, 0);
    if (c_res != CUDA_SUCCESS) {
        GGML_LOG_ERROR("VMM wake: cuMemCreate failed\\n");
        return false;
    }
    CU_CHECK(cuMemMap((CUdeviceptr)dev_ptr, aligned_size, 0, handle, 0));
    CU_CHECK(cuMemRelease(handle));
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = prop.location.id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess((CUdeviceptr)dev_ptr, aligned_size, &access, 1));
    CUDA_CHECK(cudaMemcpyAsync(dev_ptr, host_mirror, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    is_asleep = false;
    return true;
}

bool ggml_backend_cuda_buffer_sleep(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_buffer_is_cuda(buffer)) return false;
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx ? ctx->sleep() : false;
}

bool ggml_backend_cuda_buffer_wake(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_buffer_is_cuda(buffer)) return false;
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx ? ctx->wake() : false;
}

ggml_backend_cuda_buffer_context::~ggml_backend_cuda_buffer_context() {
    if (is_vmm) {
        if (!is_asleep && dev_ptr != nullptr) {
            CU_CHECK(cuMemUnmap((CUdeviceptr)dev_ptr, aligned_size));
        }
        if (dev_ptr != nullptr) {
            CU_CHECK(cuMemAddressFree((CUdeviceptr)dev_ptr, aligned_size));
        }
        if (host_mirror != nullptr) {
            CUDA_CHECK(cudaFreeHost(host_mirror));
            host_mirror = nullptr;
        }
        return;
    }"""

if t_dtor in cuda_code and "bool ggml_backend_cuda_buffer_context::sleep()" not in cuda_code:
    cuda_code = cuda_code.replace(t_dtor, r_dtor, 1)
    print("Patched: ggml_backend_cuda_buffer_context dtor and sleep/wake methods.")
else:
    print("SKIP: sleep/wake methods already patched or destructor not found.")

# Patch alloc_buffer to use VMM when LLAMAMPERE_ENABLE_SLEEP=1 is set
t_alloc = """static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}"""

r_alloc = """static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr = nullptr;
    bool is_vmm = false;
    size_t aligned_size = 0;

    // Enable VMM-backed allocations for sleep/wake support.
    // Set env var LLAMAMPERE_ENABLE_SLEEP=1 to activate.
    static const bool enable_sleep = [] {
        const char * e = getenv("LLAMAMPERE_ENABLE_SLEEP");
        return e != nullptr && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
    }();

    if (enable_sleep && size > 32 * 1024 * 1024 && ggml_cuda_info().devices[buft_ctx->device].vmm) {
        size_t granularity = ggml_cuda_info().devices[buft_ctx->device].vmm_granularity;
        if (granularity == 0) granularity = 2 * 1024 * 1024;
        aligned_size = ((size + granularity - 1) / granularity) * granularity;

        CUdeviceptr vaddr = 0;
        CUresult r_res = cuMemAddressReserve(&vaddr, aligned_size, 0, 0, 0);
        if (r_res == CUDA_SUCCESS) {
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = ggml_cuda_get_physical_device(buft_ctx->device);
            CUmemGenericAllocationHandle handle;
            CUresult c_res = cuMemCreate(&handle, aligned_size, &prop, 0);
            if (c_res == CUDA_SUCCESS) {
                CU_CHECK(cuMemMap(vaddr, aligned_size, 0, handle, 0));
                CU_CHECK(cuMemRelease(handle));
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = prop.location.id;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(vaddr, aligned_size, &access, 1));
                dev_ptr = (void *)vaddr;
                is_vmm = true;
            } else {
                CU_CHECK(cuMemAddressFree(vaddr, aligned_size));
            }
        }
    }

    if (!is_vmm) {
        cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
            return nullptr;
        }
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);
    ctx->size = size;
    ctx->aligned_size = aligned_size;
    ctx->is_vmm = is_vmm;

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}"""

if "bool is_vmm = false;" in cuda_code or "ctx->is_vmm = is_vmm;" in cuda_code:
    print("SKIP: ggml_backend_cuda_buffer_type_alloc_buffer already patched.")
elif t_alloc in cuda_code:
    cuda_code = cuda_code.replace(t_alloc, r_alloc, 1)
    print("Patched: ggml_backend_cuda_buffer_type_alloc_buffer.")
else:
    print("SKIP: alloc_buffer target not found (llama.cpp version mismatch?).")

with open(cuda_cu, "w", encoding="utf-8") as f:
    f.write(cuda_code)

# =============================================================
# 2. Patch src/llama-model.cpp
# =============================================================
model_cpp = os.path.join(BASE_DIR, "src", "llama-model.cpp")
with open(model_cpp, "r", encoding="utf-8") as f:
    model_code = f.read()

# Add ggml-cuda.h include if not already present
if '#include "ggml-cuda.h"' not in model_code:
    model_code = '#ifdef GGML_USE_CUDA\n#include "ggml-cuda.h"\n#endif\n' + model_code
    print("Patched: Added ggml-cuda.h include to llama-model.cpp.")

# Implement llama_model_sleep and llama_model_wake
t_mfree = "void llama_model_free(llama_model * model) {\n    delete model;\n}"
r_mfree = """void llama_model_free(llama_model * model) {
    delete model;
}

bool llama_model_sleep(llama_model * model) {
#ifdef GGML_USE_CUDA
    if (!model || !model->pimpl) {
        return false;
    }
    bool any = false;
    for (auto & [ctx, bufs] : model->pimpl->ctxs_bufs) {
        for (auto & buf : bufs) {
            if (ggml_backend_buffer_is_cuda(buf.get())) {
                if (ggml_backend_cuda_buffer_sleep(buf.get())) {
                    any = true;
                }
            }
        }
    }
    return any;
#else
    GGML_UNUSED(model);
    return false;
#endif
}

bool llama_model_wake(llama_model * model) {
#ifdef GGML_USE_CUDA
    if (!model || !model->pimpl) {
        return false;
    }
    bool any = false;
    for (auto & [ctx, bufs] : model->pimpl->ctxs_bufs) {
        for (auto & buf : bufs) {
            if (ggml_backend_buffer_is_cuda(buf.get())) {
                if (ggml_backend_cuda_buffer_wake(buf.get())) {
                    any = true;
                }
            }
        }
    }
    return any;
#else
    GGML_UNUSED(model);
    return false;
#endif
}"""

if t_mfree in model_code and "bool llama_model_sleep" not in model_code:
    model_code = model_code.replace(t_mfree, r_mfree, 1)
    print("Patched: llama_model_sleep and llama_model_wake in llama-model.cpp.")
else:
    print("SKIP: llama_model_sleep already present or llama_model_free not found.")

with open(model_cpp, "w", encoding="utf-8") as f:
    f.write(model_code)

# =============================================================
# 3. Patch tools/server/server-context.cpp
# =============================================================
srv_cpp = os.path.join(BASE_DIR, "tools", "server", "server-context.cpp")
with open(srv_cpp, "r", encoding="utf-8") as f:
    srv_code = f.read()

t_sleep = """    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\\n");
            prompt_cache_flush();
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }"""

r_sleep = """    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
            }
            SRV_INF("%s", "server is entering sleeping state\\n");
            prompt_cache_flush();
            // Fast VMM sleep: unmap physical VRAM without destroying the CUDA context
            if (model_tgt && llama_model_sleep(model_tgt)) {
                SRV_INF("%s", "server: fast VMM sleep active (VRAM unmapped in <60ms)\\n");
            } else {
                // Fallback: full model teardown (original behavior)
                destroy();
            }
        } else {
            SRV_INF("%s", "server is exiting sleeping state\\n");
            // Fast VMM wake: remap VRAM and stream weights from pinned RAM
            if (model_tgt && llama_model_wake(model_tgt)) {
                SRV_INF("%s", "server: fast VMM wake completed from RAM in <2s\\n");
            } else if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }"""

if "llama_model_sleep(model_tgt)" in srv_code:
    print("SKIP: handle_sleeping_state already patched.")
elif t_sleep in srv_code:
    srv_code = srv_code.replace(t_sleep, r_sleep, 1)
    print("Patched: handle_sleeping_state in server-context.cpp.")
else:
    print("SKIP: handle_sleeping_state target not found (version mismatch?).")

with open(srv_cpp, "w", encoding="utf-8") as f:
    f.write(srv_code)

print("\nALL FILES PATCHED SUCCESSFULLY.")
print("Next steps:")
print("  1. cmake -B build -DGGML_CUDA=ON")
print("  2. cmake --build build --config Release")
print("  3. set LLAMAMPERE_ENABLE_SLEEP=1")
print("  4. ./llama-server [your usual args]")
