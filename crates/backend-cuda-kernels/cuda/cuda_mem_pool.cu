#include "cuda_mem_pool.cuh"
#include <cuda_runtime.h>

extern "C" cudaError_t cuda_mem_pool_init() {
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        return err;
    }
    cudaMemPool_t default_pool = nullptr;
    return cudaDeviceGetDefaultMemPool(&default_pool, device_id);
}

extern "C" cudaError_t cuda_mem_pool_destroy() {
    return cudaSuccess;
}

cudaError_t cuda_allocator_context_init(CudaAllocatorContext* context, cudaStream_t stream) {
    if (context == nullptr) {
        return cudaErrorInvalidValue;
    }

    context->stream = stream;
    context->mem_pool = nullptr;
    context->pool_init_status = cudaSuccess;
    context->use_mem_pool = false;
    context->owns_stream = false;

    int device_id;
    cudaError_t device_err = cudaGetDevice(&device_id);
    if (device_err == cudaSuccess) {
        cudaMemPool_t default_pool = nullptr;
        context->pool_init_status = cudaDeviceGetDefaultMemPool(&default_pool, device_id);
        if (context->pool_init_status == cudaSuccess && default_pool != nullptr) {
            context->mem_pool = default_pool;
            context->use_mem_pool = true;
            return cudaSuccess;
        }
    } else {
        context->pool_init_status = device_err;
    }

    printf(
        "Falling back to direct cudaMalloc/cudaFree for stream %p: %s\n",
        static_cast<void*>(stream),
        cudaGetErrorString(context->pool_init_status)
    );

    return cudaSuccess;
}

cudaError_t cuda_allocator_context_create_private_stream(CudaAllocatorContext* context) {
    if (context == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        return err;
    }

    err = cuda_allocator_context_init(context, stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }

    context->owns_stream = true;
    return cudaSuccess;
}

cudaError_t cuda_allocator_context_synchronize(const CudaAllocatorContext* context) {
    if (context == nullptr) {
        return cudaErrorInvalidValue;
    }

    return cudaStreamSynchronize(context->stream);
}

cudaError_t cuda_allocator_context_release(CudaAllocatorContext* context) {
    if (context == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;
    if (context->owns_stream && context->stream != nullptr) {
        err = cudaStreamDestroy(context->stream);
    }

    context->stream = nullptr;
    context->mem_pool = nullptr;
    context->pool_init_status = cudaSuccess;
    context->use_mem_pool = false;
    context->owns_stream = false;

    return err;
}

extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count) {
    return cuda_mem_pool_allocate<uint32_t>(count);
}

extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count) {
    return cuda_mem_pool_allocate_zeroes<uint32_t>(count);
}

extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr) {
    cuda_mem_pool_free(ptr);
}
