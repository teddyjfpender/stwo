#ifndef CUDA_MEM_POOL_H
#define CUDA_MEM_POOL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

struct CudaAllocatorContext {
    cudaStream_t stream;
    cudaMemPool_t mem_pool;
    cudaError_t pool_init_status;
    bool use_mem_pool;
    bool owns_stream;
};

// Initialize the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_init();

// Destroy the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_destroy();

cudaError_t cuda_allocator_context_init(CudaAllocatorContext* context, cudaStream_t stream);
cudaError_t cuda_allocator_context_create_private_stream(CudaAllocatorContext* context);
cudaError_t cuda_allocator_context_synchronize(const CudaAllocatorContext* context);
cudaError_t cuda_allocator_context_release(CudaAllocatorContext* context);

template<typename T>
T* cuda_allocator_allocate_in_context(const CudaAllocatorContext& context, size_t count) {
    T* ptr = nullptr;
    size_t size = sizeof(T) * count;

    if (context.use_mem_pool && context.mem_pool != nullptr) {
        cudaError_t err =
            cudaMallocFromPoolAsync((void**)&ptr, size, context.mem_pool, context.stream);
        if (err == cudaSuccess) {
            return ptr;
        }

        printf("Failed to allocate %zu bytes from context pool: %s\n", size, cudaGetErrorString(err));
    }

    cudaError_t fallback_err = cudaMalloc((void**)&ptr, size);
    if (fallback_err != cudaSuccess) {
        printf("Failed to fallback cudaMalloc(%zu): %s\n", size, cudaGetErrorString(fallback_err));
        return nullptr;
    }

    return ptr;
}

template<typename T>
T* cuda_allocator_allocate_zeroes_in_context(const CudaAllocatorContext& context, size_t count) {
    T* ptr = cuda_allocator_allocate_in_context<T>(context, count);
    if (ptr != nullptr) {
        cudaError_t err = cudaMemsetAsync(ptr, 0, sizeof(T) * count, context.stream);
        if (err != cudaSuccess) {
            printf("Failed to zero %zu bytes in context: %s\n", sizeof(T) * count, cudaGetErrorString(err));
            cudaFree(ptr);
            return nullptr;
        }
    }
    return ptr;
}

template<typename T>
void cuda_allocator_free_in_context(const CudaAllocatorContext& context, T* ptr) {
    if (ptr == nullptr) {
        return;
    }

    if (!context.use_mem_pool || context.mem_pool == nullptr) {
        cudaFree(const_cast<void*>(reinterpret_cast<const void*>(ptr)));
        return;
    }

    cudaError_t err = cudaFreeAsync(const_cast<void*>(reinterpret_cast<const void*>(ptr)), context.stream);
    if (err != cudaSuccess) {
        printf("Failed to free pointer from context pool: %s\n", cudaGetErrorString(err));
        cudaFree(const_cast<void*>(reinterpret_cast<const void*>(ptr)));
    }
}

inline CudaAllocatorContext cuda_legacy_allocator_context() {
    CudaAllocatorContext context = {};
    cuda_allocator_context_init(&context, 0);
    return context;
}

template<typename T>
T* cuda_allocator_allocate_for_proving(size_t count) {
    CudaAllocatorContext context = {};
    cudaError_t context_err = cuda_allocator_context_create_private_stream(&context);
    if (context_err != cudaSuccess) {
        printf(
            "Failed to create proving allocator context: %s\n",
            cudaGetErrorString(context_err)
        );
        return nullptr;
    }

    T* ptr = cuda_allocator_allocate_in_context<T>(context, count);
    if (ptr != nullptr) {
        cudaError_t sync_err = cuda_allocator_context_synchronize(&context);
        if (sync_err != cudaSuccess) {
            printf(
                "Failed to synchronize proving allocator context: %s\n",
                cudaGetErrorString(sync_err)
            );
            cuda_allocator_free_in_context(context, ptr);
            cuda_allocator_context_synchronize(&context);
            ptr = nullptr;
        }
    }
    cuda_allocator_context_release(&context);
    return ptr;
}

template<typename T>
T* cuda_allocator_allocate_zeroes_for_proving(size_t count) {
    CudaAllocatorContext context = {};
    cudaError_t context_err = cuda_allocator_context_create_private_stream(&context);
    if (context_err != cudaSuccess) {
        printf(
            "Failed to create proving allocator zeroing context: %s\n",
            cudaGetErrorString(context_err)
        );
        return nullptr;
    }

    T* ptr = cuda_allocator_allocate_zeroes_in_context<T>(context, count);
    if (ptr != nullptr) {
        cudaError_t sync_err = cuda_allocator_context_synchronize(&context);
        if (sync_err != cudaSuccess) {
            printf(
                "Failed to synchronize proving allocator zeroing context: %s\n",
                cudaGetErrorString(sync_err)
            );
            cuda_allocator_free_in_context(context, ptr);
            cuda_allocator_context_synchronize(&context);
            ptr = nullptr;
        }
    }
    cuda_allocator_context_release(&context);
    return ptr;
}

template<typename T>
void cuda_allocator_free_for_proving(T* ptr) {
    if (ptr == nullptr) {
        return;
    }

    CudaAllocatorContext context = {};
    cudaError_t context_err = cuda_allocator_context_create_private_stream(&context);
    if (context_err != cudaSuccess) {
        printf(
            "Failed to create proving allocator free context: %s\n",
            cudaGetErrorString(context_err)
        );
        cudaFree(const_cast<void*>(reinterpret_cast<const void*>(ptr)));
        return;
    }

    cuda_allocator_free_in_context(context, ptr);
    if (context.use_mem_pool) {
        cuda_allocator_context_synchronize(&context);
    }
    cuda_allocator_context_release(&context);
}

// Allocate memory from the pool (with safe fallback to cudaMalloc when pool is unavailable)
template<typename T>
T* cuda_mem_pool_allocate(size_t count) {
    CudaAllocatorContext context = cuda_legacy_allocator_context();
    T* ptr = cuda_allocator_allocate_in_context<T>(context, count);
    if (ptr != nullptr) {
        cuda_allocator_context_synchronize(&context);
    }
    return ptr;
}

// Allocate zeroed memory from the pool
template<typename T>
T* cuda_mem_pool_allocate_zeroes(size_t count) {
    CudaAllocatorContext context = cuda_legacy_allocator_context();
    T* ptr = cuda_allocator_allocate_zeroes_in_context<T>(context, count);
    if (ptr != nullptr) {
        cuda_allocator_context_synchronize(&context);
    }
    return ptr;
}

// Free memory back to the pool (fallback aware)
template<typename T>
void cuda_mem_pool_free(T* ptr) {
    if (ptr != nullptr) {
        CudaAllocatorContext context = cuda_legacy_allocator_context();
        cuda_allocator_free_in_context(context, ptr);
        if (context.use_mem_pool) {
            cuda_allocator_context_synchronize(&context);
        }
    }
}

// C-style wrappers for specific types
extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count);
extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count);
extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr);

#endif // CUDA_MEM_POOL_H
