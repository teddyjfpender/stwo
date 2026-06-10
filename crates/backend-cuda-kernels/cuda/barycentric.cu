#include "barycentric.cuh"

#include "batch_inverse.cuh"
#include "utils.cuh"

namespace {

constexpr uint32_t BARYCENTRIC_BLOCK_DIM = 256;

__global__ void barycentric_scale_weights_kernel(
    const qm31 *point_vanishing_inverses,
    uint32_t size,
    qm31 even_scale,
    qm31 odd_scale,
    qm31 *result_weights
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    qm31 scale = (index & 1U) == 0 ? even_scale : odd_scale;
    result_weights[index] = mul(point_vanishing_inverses[index], scale);
}

__global__ void barycentric_eval_partial_kernel(
    const m31 *eval_values,
    const qm31 *weights,
    uint32_t size,
    qm31 *partial_sums
) {
    extern __shared__ qm31 shared[];

    qm31 thread_sum = {{0, 0}, {0, 0}};
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (index < size) {
        thread_sum = add(thread_sum, mul(eval_values[index], weights[index]));
        index += stride;
    }

    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = add(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

__global__ void reduce_qm31_partial_sums_kernel(
    const qm31 *input,
    uint32_t size,
    qm31 *output
) {
    extern __shared__ qm31 shared[];

    qm31 thread_sum = {{0, 0}, {0, 0}};
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (index < size) {
        thread_sum = add(thread_sum, input[index]);
        index += stride;
    }

    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = add(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared[0];
    }
}

} // namespace

extern "C"
void barycentric_weights_from_point_vanishings(
    const qm31 *point_vanishings,
    uint32_t size,
    qm31 even_scale,
    qm31 odd_scale,
    qm31 *result_weights
) {
    qm31 *inverses = cuda_proving_malloc<qm31>(size);
    batch_inverse_secure_field(const_cast<qm31 *>(point_vanishings), inverses, size);

    uint32_t num_blocks = (size + BARYCENTRIC_BLOCK_DIM - 1) / BARYCENTRIC_BLOCK_DIM;
    barycentric_scale_weights_kernel<<<num_blocks, BARYCENTRIC_BLOCK_DIM>>>(
        inverses,
        size,
        even_scale,
        odd_scale,
        result_weights
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_proving_free(inverses);
}

extern "C"
qm31 barycentric_eval_base_field(
    const m31 *eval_values,
    const qm31 *weights,
    uint32_t size
) {
    uint32_t num_blocks = (size + BARYCENTRIC_BLOCK_DIM - 1) / BARYCENTRIC_BLOCK_DIM;
    qm31 *partials = cuda_proving_malloc<qm31>(num_blocks);
    barycentric_eval_partial_kernel<<<num_blocks, BARYCENTRIC_BLOCK_DIM, sizeof(qm31) * BARYCENTRIC_BLOCK_DIM>>>(
        eval_values,
        weights,
        size,
        partials
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    uint32_t current_size = num_blocks;
    qm31 *current = partials;
    while (current_size > 1) {
        uint32_t next_num_blocks = (current_size + BARYCENTRIC_BLOCK_DIM - 1) / BARYCENTRIC_BLOCK_DIM;
        qm31 *next = cuda_proving_malloc<qm31>(next_num_blocks);
        reduce_qm31_partial_sums_kernel<<<next_num_blocks, BARYCENTRIC_BLOCK_DIM, sizeof(qm31) * BARYCENTRIC_BLOCK_DIM>>>(
            current,
            current_size,
            next
        );
        ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
        cuda_proving_free(current);
        current = next;
        current_size = next_num_blocks;
    }

    qm31 result = {{0, 0}, {0, 0}};
    cuda_mem_copy_device_to_host(current, &result, 1);

    cuda_proving_free(current);

    return result;
}
