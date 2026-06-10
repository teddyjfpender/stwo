#ifndef CUDA_MEM_POOL_H
#define CUDA_MEM_POOL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Stream-ordered allocator (see docs/gpu-architecture-analysis.md, item 1).
//
// Every allocation, zero-fill, and free is enqueued on the LEGACY DEFAULT STREAM
// with no host synchronization:
//
//   - cudaMallocFromPoolAsync(stream 0): the returned pointer is usable
//     immediately by later default-stream work — stream ordering guarantees the
//     memory is backed before any consumer runs.
//   - cudaMemsetAsync(stream 0): ordered before any later default-stream reader.
//   - cudaFreeAsync(stream 0): ordered after all prior default-stream users of
//     the buffer, so no use-after-free is possible for default-stream work.
//
// The previous implementation created, synchronized, and destroyed a PRIVATE
// stream around every single alloc/free (4-6k stream lifecycle events plus full
// host stalls per prove). With a never-release pool (threshold = UINT64_MAX, the
// NitrooZK setup) the steady-state cost of an allocation is now a single stream-
// ordered pool op.
//
// Host code must never dereference these pointers directly; all host reads in
// this crate go through synchronous cudaMemcpy, which orders after the default
// stream and blocks until complete (the fence).
// ---------------------------------------------------------------------------

// Initialize the CUDA memory pool (idempotent; sets the never-release threshold).
extern "C" cudaError_t cuda_mem_pool_init();

// Destroy the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_destroy();

// Cached default-mem-pool handle for the current device. Resolved once per
// process (the backend is single-device); nullptr when stream-ordered allocation
// is unavailable, in which case callers fall back to plain cudaMalloc/cudaFree.
cudaMemPool_t stwo_default_mem_pool();

template<typename T>
T* cuda_allocator_allocate_for_proving(size_t count) {
    T* ptr = nullptr;
    size_t size = sizeof(T) * count;

    cudaMemPool_t pool = stwo_default_mem_pool();
    if (pool != nullptr) {
        cudaError_t err = cudaMallocFromPoolAsync((void**)&ptr, size, pool, 0);
        if (err == cudaSuccess) {
            return ptr;
        }
        printf("Failed to allocate %zu bytes from pool: %s\n", size, cudaGetErrorString(err));
    }

    cudaError_t fallback_err = cudaMalloc((void**)&ptr, size);
    if (fallback_err != cudaSuccess) {
        printf("Failed to fallback cudaMalloc(%zu): %s\n", size, cudaGetErrorString(fallback_err));
        return nullptr;
    }
    return ptr;
}

template<typename T>
T* cuda_allocator_allocate_zeroes_for_proving(size_t count) {
    T* ptr = cuda_allocator_allocate_for_proving<T>(count);
    if (ptr != nullptr && count > 0) {
        cudaError_t err = cudaMemsetAsync(ptr, 0, sizeof(T) * count, 0);
        if (err != cudaSuccess) {
            printf("Failed to zero %zu bytes: %s\n", sizeof(T) * count, cudaGetErrorString(err));
            cudaFreeAsync(ptr, 0);
            return nullptr;
        }
    }
    return ptr;
}

template<typename T>
void cuda_allocator_free_for_proving(T* ptr) {
    if (ptr == nullptr) {
        return;
    }
    // cudaFreeAsync handles both pool allocations and plain cudaMalloc memory
    // (CUDA >= 11.2); ordering on stream 0 guarantees all prior default-stream
    // users of the buffer have finished before the memory is recycled.
    cudaError_t err = cudaFreeAsync(const_cast<void*>(reinterpret_cast<const void*>(ptr)), 0);
    if (err != cudaSuccess) {
        printf("Failed stream-ordered free, falling back: %s\n", cudaGetErrorString(err));
        cudaFree(const_cast<void*>(reinterpret_cast<const void*>(ptr)));
    }
}

// Legacy aliases (same stream-ordered behavior).
template<typename T>
T* cuda_mem_pool_allocate(size_t count) {
    return cuda_allocator_allocate_for_proving<T>(count);
}

template<typename T>
T* cuda_mem_pool_allocate_zeroes(size_t count) {
    return cuda_allocator_allocate_zeroes_for_proving<T>(count);
}

template<typename T>
void cuda_mem_pool_free(T* ptr) {
    cuda_allocator_free_for_proving(ptr);
}

// C-style wrappers for specific types
extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count);
extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count);
extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr);

#endif // CUDA_MEM_POOL_H
