// NVRTC-based JIT runtime for generated constraint kernels.
//
// Compiles CUDA C source at runtime and launches the fused constraint kernel.
// Compiled kernels are cached by their SEMANTIC HASH (a content hash of the program
// bytecode computed on the Rust side) — never by pointers or implicit scope, per the
// repository's cache-keying rule (pointer-keyed caches silently alias across
// alloc/free cycles and corrupt later proves).

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct JitCache {
    std::mutex mutex;
    // semantic_hash -> compiled function (module kept alive for process lifetime).
    std::unordered_map<uint64_t, CUfunction> functions;
};

JitCache &jit_cache() {
    static JitCache cache;
    return cache;
}

bool compile_kernel(const char *source, const char *kernel_name, CUfunction *out) {
    nvrtcProgram program;
    if (nvrtcCreateProgram(&program, source, "stwo_jit.cu", 0, nullptr, nullptr) !=
        NVRTC_SUCCESS) {
        return false;
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    char arch_flag[64];
    snprintf(arch_flag, sizeof(arch_flag), "--gpu-architecture=compute_%d%d", props.major,
             props.minor);
    const char *options[] = {arch_flag, "--std=c++14"};

    nvrtcResult compile_result = nvrtcCompileProgram(program, 2, options);
    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(program, &log[0]);
        fprintf(stderr, "stwo JIT: NVRTC compilation failed:\n%s\n", log.c_str());
        nvrtcDestroyProgram(&program);
        return false;
    }

    size_t ptx_size = 0;
    nvrtcGetPTXSize(program, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(program, ptx.data());
    nvrtcDestroyProgram(&program);

    // Ensure the runtime API's primary context is current for the driver API.
    cudaFree(0);
    CUmodule module;
    if (cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr) != CUDA_SUCCESS) {
        fprintf(stderr, "stwo JIT: cuModuleLoadDataEx failed\n");
        return false;
    }
    if (cuModuleGetFunction(out, module, kernel_name) != CUDA_SUCCESS) {
        fprintf(stderr, "stwo JIT: kernel %s not found in module\n", kernel_name);
        return false;
    }
    return true;
}

}  // namespace

// Compiles (cached by semantic_hash) and launches the fused constraint kernel.
// Returns true on success; false means the caller must use the CPU lane.
extern "C" bool stwo_cuda_jit_eval_fused(
    const char *source,
    const char *kernel_name,
    uint64_t semantic_hash,
    const uint32_t *trace_values,
    const uint32_t *interaction_offsets,
    const uint32_t *base_params,
    const uint32_t *ext_params,
    const uint32_t *random_coeff_powers,
    const uint32_t *denom_inv,
    uint32_t *coord_0,
    uint32_t *coord_1,
    uint32_t *coord_2,
    uint32_t *coord_3,
    uint32_t row_count,
    uint32_t log_n_rows
) {
    CUfunction function = nullptr;
    {
        JitCache &cache = jit_cache();
        std::lock_guard<std::mutex> guard(cache.mutex);
        auto it = cache.functions.find(semantic_hash);
        if (it != cache.functions.end()) {
            function = it->second;
        } else {
            if (!compile_kernel(source, kernel_name, &function)) {
                return false;
            }
            cache.functions.emplace(semantic_hash, function);
        }
    }

    void *args[] = {
        (void *)&trace_values, (void *)&interaction_offsets, (void *)&base_params,
        (void *)&ext_params,   (void *)&random_coeff_powers, (void *)&denom_inv,
        (void *)&coord_0,      (void *)&coord_1,             (void *)&coord_2,
        (void *)&coord_3,      (void *)&row_count,           (void *)&log_n_rows,
    };
    const unsigned block = 128;  // must match the generated kernel's __launch_bounds__
    const unsigned grid = (row_count + block - 1) / block;
    if (cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr) !=
        CUDA_SUCCESS) {
        fprintf(stderr, "stwo JIT: cuLaunchKernel failed for %s\n", kernel_name);
        return false;
    }
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "stwo JIT: kernel %s failed: %s\n", kernel_name,
                cudaGetErrorString(sync));
        return false;
    }
    return true;
}
