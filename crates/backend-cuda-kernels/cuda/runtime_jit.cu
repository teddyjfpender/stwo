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
#include <sys/stat.h>
#include <cstdlib>
#include <chrono>

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

// Filesystem PTX cache: JIT-compiled kernels persist across processes, keyed by the
// program's CONTENT semantic hash + the GPU architecture (the same content-keying
// rule as the in-memory cache — never pointers, never implicit scope). Directory:
// $STWO_JIT_CACHE_DIR, else $HOME/.cache/stwo-jit. Disable with STWO_JIT_CACHE_DIR
// set to the empty string.
static std::string ptx_cache_path(uint64_t semantic_hash, int major, int minor) {
    const char *dir = getenv("STWO_JIT_CACHE_DIR");
    std::string base;
    if (dir != nullptr) {
        if (dir[0] == '\0') return std::string();
        base = dir;
    } else {
        const char *home = getenv("HOME");
        if (home == nullptr) return std::string();
        base = std::string(home) + "/.cache/stwo-jit";
    }
    char name[128];
    snprintf(name, sizeof(name), "/sm%d%d_%016llx.ptx", major, minor,
             (unsigned long long)semantic_hash);
    return base + name;
}

bool jit_log_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("STWO_JIT_LOG");
        enabled = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

bool compile_kernel(const char *source, const char *kernel_name, uint64_t semantic_hash,
                    CUfunction *out) {
    auto t_start = std::chrono::steady_clock::now();
    bool from_disk = false;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::vector<char> ptx;
    std::string cache_file = ptx_cache_path(semantic_hash, props.major, props.minor);
    if (!cache_file.empty()) {
        if (FILE *f = fopen(cache_file.c_str(), "rb")) {
            fseek(f, 0, SEEK_END);
            long len = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (len > 0) {
                ptx.resize((size_t)len);
                if (fread(ptx.data(), 1, (size_t)len, f) != (size_t)len) ptx.clear();
            }
            fclose(f);
            from_disk = !ptx.empty();
        }
    }

    if (ptx.empty()) {
        nvrtcProgram program;
        if (nvrtcCreateProgram(&program, source, "stwo_jit.cu", 0, nullptr, nullptr) !=
            NVRTC_SUCCESS) {
            return false;
        }
        char arch_flag[64];
        snprintf(arch_flag, sizeof(arch_flag), "--gpu-architecture=compute_%d%d",
                 props.major, props.minor);
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
        ptx.resize(ptx_size);
        nvrtcGetPTX(program, ptx.data());
        nvrtcDestroyProgram(&program);

        if (!cache_file.empty()) {
            // mkdir -p the cache dir (two levels at most), best-effort.
            std::string dir = cache_file.substr(0, cache_file.find_last_of('/'));
            std::string parent = dir.substr(0, dir.find_last_of('/'));
            mkdir(parent.c_str(), 0755);
            mkdir(dir.c_str(), 0755);
            std::string tmp = cache_file + ".tmp";
            if (FILE *f = fopen(tmp.c_str(), "wb")) {
                fwrite(ptx.data(), 1, ptx.size(), f);
                fclose(f);
                rename(tmp.c_str(), cache_file.c_str());
            }
        }
    }

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
    if (jit_log_enabled()) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t_start)
                      .count();
        fprintf(stderr, "stwo JIT: %s ready in %lld ms (%s)\n", kernel_name, (long long)ms,
                from_disk ? "disk PTX cache" : "NVRTC compile");
    }
    return true;
}

}  // namespace

// Compile (if not cached) and cache the fused kernel WITHOUT launching it. Used by
// the parallel pre-compile pass that warms the kernel cache before composition.
// NVRTC compilation is thread-safe and the disk cache write is atomic, so callers
// run this concurrently across components; the actual compile happens OUTSIDE the
// cache mutex (only the map check and insert are locked), so concurrent calls for
// DIFFERENT kernels parallelize. A rare double-compile of the SAME kernel is benign:
// the PTX is identical, the disk write is temp+rename atomic, and the second insert
// is a no-op (its module leaks, bounded by the kernel count — a one-time warmup cost).
extern "C" bool stwo_cuda_jit_compile(
    const char *source,
    const char *kernel_name,
    uint64_t cache_key
) {
    {
        JitCache &cache = jit_cache();
        std::lock_guard<std::mutex> guard(cache.mutex);
        if (cache.functions.find(cache_key) != cache.functions.end()) {
            return true;
        }
    }
    CUfunction function = nullptr;
    if (!compile_kernel(source, kernel_name, cache_key, &function)) {
        return false;
    }
    {
        JitCache &cache = jit_cache();
        std::lock_guard<std::mutex> guard(cache.mutex);
        cache.functions.emplace(cache_key, function);
    }
    return true;
}

// Compiles (cached by cache_key = semantic hash mixed with the Rust emitter's
// CODEGEN_VERSION) and launches the fused constraint kernel. Returns true once the
// kernel is enqueued on the legacy default stream; false means nothing was launched
// and the caller must use the CPU lane. No device synchronization happens here: the
// kernel's outputs are only ever consumed by later same-stream work or by
// synchronous D2H copies, both of which the legacy stream orders after this launch.
extern "C" bool stwo_cuda_jit_eval_fused(
    const char *source,
    const char *kernel_name,
    uint64_t cache_key,
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
        auto it = cache.functions.find(cache_key);
        if (it != cache.functions.end()) {
            function = it->second;
        } else {
            if (!compile_kernel(source, kernel_name, cache_key, &function)) {
                return false;
            }
            cache.functions.emplace(cache_key, function);
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
    // A launch failure happens before any device write, so returning false here still
    // permits the CPU fallback. After a successful enqueue the kernel updates the
    // accumulator in place; an asynchronous execution fault becomes a sticky context
    // error and aborts the prove at the next checked CUDA call — it must NOT fall
    // back (the accumulator state would be indeterminate), and it cannot, since we
    // already returned true.
    if (cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr) !=
        CUDA_SUCCESS) {
        fprintf(stderr, "stwo JIT: cuLaunchKernel failed for %s\n", kernel_name);
        return false;
    }
    return true;
}
