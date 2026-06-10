#include "utils.cuh"
#include "cuda_mem_pool.cuh"

#include <cstdio>

// Must match the definition in utils.cuh
#define USE_CUDA_MEM_POOL 1


__host__ int log_2(int value) {
    return __builtin_ctz(value);
}

namespace {

constexpr uint32_t MAX_MULTI_LAYER_BATCH_GET_LAYERS = 24;

CudaAllocatorContext create_private_utils_allocator_context() {
    CudaAllocatorContext context = {};
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_create_private_stream(&context));
    return context;
}

void release_private_utils_allocator_context(CudaAllocatorContext* context) {
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_release(context));
}

const uint32_t* const* upload_device_pointer_vec_with_private_context(
    const uint32_t* const* host_ptr,
    uint32_t size
) {
    if (size == 0) {
        return nullptr;
    }

    CudaAllocatorContext context = create_private_utils_allocator_context();
    const uint32_t** device_ptr =
        cuda_allocator_allocate_in_context<const uint32_t*>(context, size);
    if (device_ptr == nullptr) {
        printf("Failed to allocate device pointer vector upload buffer\n");
        release_private_utils_allocator_context(&context);
        return nullptr;
    }

    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        const_cast<uint32_t**>(device_ptr),
        host_ptr,
        size * sizeof(uint32_t*),
        cudaMemcpyHostToDevice,
        context.stream
    ));
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
    release_private_utils_allocator_context(&context);
    return device_ptr;
}

void release_uploaded_pointer_vec_with_private_context(
    const uint32_t* const* device_ptr
) {
    if (device_ptr == nullptr) {
        return;
    }

    CudaAllocatorContext context = create_private_utils_allocator_context();
    cuda_allocator_free_in_context(context, device_ptr);
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
    release_private_utils_allocator_context(&context);
}

}  // namespace

void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr, host_ptr, size);
}

uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr = cuda_proving_alloc_zeroes_u32_words(size);
    cuda_mem_copy_host_to_device(host_ptr, device_ptr, size);
    return device_ptr;
}

void copy_uint32_t_vec_from_device_to_device(uint32_t *from, uint32_t *dst, int size) {
    cuda_mem_copy_device_to_device<uint32_t>(from, dst, size);
}

void copy_uint32_t_vec_from_device_to_device_offset(uint32_t *from, uint32_t *dst, int size, int offset) {
    cuda_mem_copy_device_to_device<uint32_t>(from, dst + offset, size);
}

uint32_t* cuda_malloc_uint32_t(int size) {
    return cuda_proving_alloc_zeroes_u32_words(size);
}

Blake2sHash* cuda_malloc_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_proving_malloc<Blake2sHash>(size);
    // cudaMemset(device_ptr, 0x00, sizeof(Blake2sHash) * size);
    return device_ptr;
}

__global__ void print_array(uint32_t *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        printf("%d, ", array[idx]);
    }
}

uint32_t* cuda_alloc_zeroes_uint32_t(int size) {
    return cuda_proving_alloc_zeroes_u32_words(size);
}

void cuda_set_uint32_t(uint32_t *device_ptr, size_t index, uint32_t value) {
    cuda_mem_copy_host_to_device<uint32_t>(&value, device_ptr + index, 1);
}

void cuda_increase_at(uint32_t *device_ptr, uint32_t address) {
    uint32_t value;
    cudaMemcpy(&value, device_ptr + address, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    value += 1;
    cudaMemcpy(device_ptr + address, &value, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

uint32_t cuda_get_uint32_t(uint32_t *device_ptr, size_t index) {
    uint32_t value = 0x0;
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr + index, &value, 1);
    return value;
}

qm31 cuda_get_secure_field(qm31 *device_ptr, size_t index) {
    qm31 value = {};
    cuda_mem_copy_device_to_host<qm31>(device_ptr + index, &value, 1);
    return value;
}

Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc_blake_2s_hash(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
}

Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size) {
    Blake2sHash* device_ptr = cuda_proving_clone_to_device<Blake2sHash>(host_ptr, size);
    return device_ptr;
}

void cuda_get_blake_2s_hash(Blake2sHash *device_ptr, Blake2sHash *host_ptr, size_t index) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr + index, host_ptr, 1);
}

void cuda_set_blake_2s_hash(Blake2sHash *device_ptr, size_t index, const Blake2sHash *host_ptr) {
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        device_ptr + index,
        host_ptr,
        sizeof(Blake2sHash),
        cudaMemcpyHostToDevice
    ));
}

// Kernel: Batch get Blake2s hashes from device memory by indices
__global__ void batch_get_blake2s_kernel(
    const Blake2sHash* src,
    Blake2sHash* dst,
    const uint32_t* indices,
    uint32_t n_indices
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_indices) {
        uint32_t src_idx = indices[idx];
        // Copy 32 bytes (Blake2sHash is 32 bytes)
        // Using uint4 for efficient 16-byte aligned memory access
        const uint4* src_ptr = reinterpret_cast<const uint4*>(&src[src_idx]);
        uint4* dst_ptr = reinterpret_cast<uint4*>(&dst[idx]);

        // Copy in two uint4 chunks (2 * 16 bytes = 32 bytes)
        dst_ptr[0] = src_ptr[0];
        dst_ptr[1] = src_ptr[1];
    }
}

// Host function: Batch get Blake2s hashes
void cuda_batch_get_blake_2s_hash(
    Blake2sHash *device_ptr,
    Blake2sHash *host_ptr,
    uint32_t *indices,
    uint32_t n_indices
) {
    if (n_indices == 0) {
        return;
    }

    CudaAllocatorContext context = create_private_utils_allocator_context();

    // 1. Allocate GPU memory for indices array from the explicit allocator context.
    uint32_t* d_indices = cuda_allocator_allocate_in_context<uint32_t>(context, n_indices);
    if (!d_indices) {
        printf("Failed to allocate indices buffer in batch_get\n");
        release_private_utils_allocator_context(&context);
        return;
    }

    // 2. Allocate GPU memory for result array from the same explicit context.
    Blake2sHash* d_result = cuda_allocator_allocate_in_context<Blake2sHash>(context, n_indices);
    if (!d_result) {
        printf("Failed to allocate result buffer in batch_get\n");
        cuda_allocator_free_in_context(context, d_indices);
        ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
        release_private_utils_allocator_context(&context);
        return;
    }

    // 3. Copy indices to GPU asynchronously on the private stream.
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        d_indices,
        indices,
        n_indices * sizeof(uint32_t),
        cudaMemcpyHostToDevice,
        context.stream
    ));

    // 4. Launch kernel to gather hashes in parallel
    const int block_size = 256;
    const int num_blocks = (n_indices + block_size - 1) / block_size;
    batch_get_blake2s_kernel<<<num_blocks, block_size, 0, context.stream>>>(
        device_ptr, d_result, d_indices, n_indices
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // 5. Copy result back to CPU asynchronously on the same explicit stream.
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        host_ptr,
        d_result,
        n_indices * sizeof(Blake2sHash),
        cudaMemcpyDeviceToHost,
        context.stream
    ));

    // 6. Synchronize the explicit stream to ensure all operations complete.
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));

    // 7. Free temporary GPU memory back through the same explicit context.
    cuda_allocator_free_in_context(context, d_indices);
    cuda_allocator_free_in_context(context, d_result);
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
    release_private_utils_allocator_context(&context);
}

// Multi-layer batch get kernel
__global__ void multi_layer_batch_get_kernel(
    const Blake2sHash* const* layer_ptrs,  // Array of layer device pointers
    Blake2sHash* dst,
    const LayerIndexPair* pairs,
    uint32_t n_pairs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_pairs) {
        LayerIndexPair pair = pairs[idx];
        const Blake2sHash* src_layer = layer_ptrs[pair.layer_idx];

        // Copy 32 bytes using uint4 for efficient aligned access
        const uint4* src_ptr = reinterpret_cast<const uint4*>(&src_layer[pair.hash_idx]);
        uint4* dst_ptr = reinterpret_cast<uint4*>(&dst[idx]);

        // 2 * uint4 = 2 * 16 bytes = 32 bytes = 1 Blake2sHash
        dst_ptr[0] = src_ptr[0];
        dst_ptr[1] = src_ptr[1];
    }
}

// Multi-layer batch get host function
void cuda_multi_layer_batch_get_blake_2s_hash(
    const Blake2sHash **layer_device_ptrs,
    Blake2sHash *host_ptr,
    const LayerIndexPair *pairs,
    uint32_t n_pairs
) {
    if (n_pairs == 0) {
        return;
    }

    CudaAllocatorContext context = create_private_utils_allocator_context();

    // 1. Allocate GPU memory for layer pointers array.
    const Blake2sHash** d_layer_ptrs =
        cuda_allocator_allocate_in_context<const Blake2sHash*>(context, MAX_MULTI_LAYER_BATCH_GET_LAYERS);
    if (!d_layer_ptrs) {
        printf("Failed to allocate layer pointers in multi_layer_batch_get\n");
        release_private_utils_allocator_context(&context);
        return;
    }

    // 2. Allocate GPU memory for pairs array.
    LayerIndexPair* d_pairs = cuda_allocator_allocate_in_context<LayerIndexPair>(context, n_pairs);
    if (!d_pairs) {
        printf("Failed to allocate pairs buffer in multi_layer_batch_get\n");
        cuda_allocator_free_in_context(context, d_layer_ptrs);
        ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
        release_private_utils_allocator_context(&context);
        return;
    }

    // 3. Allocate GPU memory for result array.
    Blake2sHash* d_result = cuda_allocator_allocate_in_context<Blake2sHash>(context, n_pairs);
    if (!d_result) {
        printf("Failed to allocate result buffer in multi_layer_batch_get\n");
        cuda_allocator_free_in_context(context, d_layer_ptrs);
        cuda_allocator_free_in_context(context, d_pairs);
        ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
        release_private_utils_allocator_context(&context);
        return;
    }

    // 4. Copy layer pointers and pairs to GPU asynchronously on the explicit stream.
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        (void*)d_layer_ptrs,
        layer_device_ptrs,
        MAX_MULTI_LAYER_BATCH_GET_LAYERS * sizeof(Blake2sHash*),
        cudaMemcpyHostToDevice,
        context.stream
    ));
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        d_pairs,
        pairs,
        n_pairs * sizeof(LayerIndexPair),
        cudaMemcpyHostToDevice,
        context.stream
    ));

    // 5. Launch kernel to gather hashes in parallel from multiple layers
    const int block_size = 256;
    const int num_blocks = (n_pairs + block_size - 1) / block_size;
    multi_layer_batch_get_kernel<<<num_blocks, block_size, 0, context.stream>>>(
        d_layer_ptrs, d_result, d_pairs, n_pairs
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // 6. Copy result back to CPU asynchronously on the same explicit stream.
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        host_ptr,
        d_result,
        n_pairs * sizeof(Blake2sHash),
        cudaMemcpyDeviceToHost,
        context.stream
    ));

    // 7. Synchronize the explicit stream.
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));

    // 8. Free temporary GPU memory through the same explicit context.
    cuda_allocator_free_in_context(context, d_layer_ptrs);
    cuda_allocator_free_in_context(context, d_pairs);
    cuda_allocator_free_in_context(context, d_result);
    ASSERT_CUDA_SUCCESS(cuda_allocator_context_synchronize(&context));
    release_private_utils_allocator_context(&context);
}

void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr, host_ptr, size);
}

void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size) {
    cuda_mem_copy_device_to_device<Blake2sHash>(from, dst, size);
}

const uint32_t* const* copy_device_pointer_vec_from_host_to_device(
    const uint32_t* const* host_ptr,
    uint32_t size
) {
    return upload_device_pointer_vec_with_private_context(host_ptr, size);
}

void cuda_release_uploaded_pointer_vec(const uint32_t* const* device_ptr) {
    release_uploaded_pointer_vec_with_private_context(device_ptr);
}

// void** copy_device_pointer_vec_from_host_to_device(const void** ptrs, size_t n) {
//     void** d_ptrs;
//     cudaMalloc(&d_ptrs, n * sizeof(void*));
//     cudaMemcpy(d_ptrs, ptrs, n * sizeof(void*), cudaMemcpyHostToDevice);
//     return d_ptrs;
// }

void cuda_free_memory(void *device_ptr) {
    cuda_proving_free(static_cast<uint8_t*>(device_ptr));
}

// Stub implementations for backward compatibility
// These will be removed once all code is migrated to use CUDA memory pool directly
extern "C" uint32_t* pool_allocate_cuda(size_t size) {
    return cuda_mem_pool_allocate_uint32(size);
}

extern "C" void pool_deallocate_cuda(uint32_t* ptr, size_t size) {
    (void)size; // Unused parameter
    cuda_mem_pool_free_uint32(ptr);
}

extern "C" uint32_t* pool_allocate_zeroes_cuda(size_t size) {
    return cuda_mem_pool_allocate_zeroes_uint32(size);
}

// Test function to compute offset_bit_reversed_circle_domain_index on GPU
// This is used to verify CUDA matches Rust implementation
__global__ void test_offset_indices_kernel(
    unsigned int* result,
    unsigned int domain_log_size,
    unsigned int eval_log_size,
    int offset,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = offset_bit_reversed_circle_domain_index(i, domain_log_size, eval_log_size, offset);
    }
}

extern "C" void test_offset_bit_reversed_indices(
    unsigned int* result_host,
    unsigned int domain_log_size,
    unsigned int eval_log_size,
    int offset,
    unsigned int n
) {
    unsigned int* result_device = cuda_proving_malloc<unsigned int>(n);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    test_offset_indices_kernel<<<num_blocks, block_size>>>(
        result_device, domain_log_size, eval_log_size, offset, n
    );

    cudaDeviceSynchronize();
    cuda_mem_copy_device_to_host(result_device, result_host, n);
    cuda_proving_free(result_device);
}

// Get CUDA memory info (free and total memory in bytes)
extern "C" void cuda_get_memory_info(size_t* free_mem, size_t* total_mem) {
    cudaError_t err = cudaMemGetInfo(free_mem, total_mem);
    if (err != cudaSuccess) {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        *free_mem = 0;
        *total_mem = 0;
    }
}
