#include <cstdio>
#include <vector>
#include <array>
#include "fields.cuh"
#include "timer.cuh"
#include "utils.cuh"
#include "cuda_mem_pool.cuh"
#include "prefix_sum.cuh"
#include "bit_reverse.cuh"
#include <cub/cub.cuh>

__global__ void circle_domain_order_to_coset_order_kernel(const m31* in, m31* out, int n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2) {
        out[2 * i] = in[i];
        out[2 * i + 1] = in[n - 1 - i];
    }
}

__global__ void coset_order_to_circle_domain_order_kernel(const m31* d_in, m31* d_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int half_len = n / 2;
        if (tid < half_len) {
            d_out[tid] = d_in[tid << 1];
        } else {
            int i = tid - half_len;
            d_out[tid] = d_in[n - 1 - (i << 1)];
        }
    }
}


__global__ void bit_reverse_generic_m31_on(m31 *array, int size, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rev_idx = bit_reverse(idx, bits);
    if (rev_idx > idx && idx < size) {
        m31 temp = array[idx];
        array[idx] = array[rev_idx];
        array[rev_idx] = temp;
    }
}

// Stream-parameterized scan body: every launch, CUB call, and copy runs on
// `stream`. Caller owns the P3 bridge discipline (see cuda_mem_pool.cuh) and
// must pre-allocate/free the scratch on the legacy stream around the bridges.
static void inclusive_prefix_sum_on(
    cudaStream_t stream,
    m31 *device_bit_rev_circle_domain_evals,
    m31 *eval_tmp,
    void *d_temp_storage,
    size_t temp_storage_bytes,
    unsigned len
) {
    {
        int bits = log_2((int)len);
        int block_size = 1024;
        int num_blocks = ((int)len + block_size - 1) / block_size;
        bit_reverse_generic_m31_on<<<num_blocks, block_size, 0, stream>>>(
            device_bit_rev_circle_domain_evals, (int)len, bits);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }

    int total_threads = len / 2;
    int block_dim = total_threads < THREAD_COUNT_MAX ? total_threads : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (total_threads + block_dim - 1) / block_dim;
    circle_domain_order_to_coset_order_kernel<<<num_blocks, block_dim, 0, stream>>>(
        device_bit_rev_circle_domain_evals, eval_tmp, len);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    ASSERT_CUDA_SUCCESS(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        (M31 *)eval_tmp, (M31 *)device_bit_rev_circle_domain_evals, len, stream));

    total_threads = len;
    block_dim = total_threads < THREAD_COUNT_MAX ? total_threads : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (total_threads + block_dim - 1) / block_dim;
    coset_order_to_circle_domain_order_kernel<<<num_blocks, block_dim, 0, stream>>>(
        device_bit_rev_circle_domain_evals, eval_tmp, len);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    {
        int bits = log_2((int)len);
        int block_size = 1024;
        int nb = ((int)len + block_size - 1) / block_size;
        bit_reverse_generic_m31_on<<<nb, block_size, 0, stream>>>(eval_tmp, (int)len, bits);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }

    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
        device_bit_rev_circle_domain_evals, eval_tmp, sizeof(m31) * len,
        cudaMemcpyDeviceToDevice, stream));
}

static size_t scan_temp_bytes(unsigned len) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    ASSERT_CUDA_SUCCESS(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, (M31 *)nullptr, (M31 *)nullptr, len));
    return temp_storage_bytes;
}

void inclusive_prefix_sum(m31 *device_bit_rev_circle_domain_evals, unsigned len) {
    m31 *eval_tmp = reinterpret_cast<m31 *>(cuda_proving_alloc_zeroes_u32_words(len));
    size_t temp_storage_bytes = scan_temp_bytes(len);
    void *d_temp_storage = cuda_proving_malloc<uint8_t>((unsigned)temp_storage_bytes);
    // The legacy stream is its own bridge: run the body on stream 0 directly.
    inclusive_prefix_sum_on(
        (cudaStream_t)0, device_bit_rev_circle_domain_evals, eval_tmp, d_temp_storage,
        temp_storage_bytes, len);
    cuda_proving_free(eval_tmp);
    cuda_proving_free(d_temp_storage);
}

// Four independent scans (the per-coordinate logup cumsum) on four pool streams.
// Scratch is allocated and freed on the legacy stream OUTSIDE the bridges, per the
// P3 contract; each stream waits on the legacy frontier (the shift kernel's
// writes) and the legacy stream waits on all four before any consumer runs.
extern "C" void inclusive_prefix_sum_x4(
    uint32_t *c0, uint32_t *c1, uint32_t *c2, uint32_t *c3, unsigned len
) {
    m31 *cols[4] = {c0, c1, c2, c3};
    m31 *tmps[4];
    void *temps[4];
    size_t temp_storage_bytes = scan_temp_bytes(len);
    for (int i = 0; i < 4; ++i) {
        tmps[i] = reinterpret_cast<m31 *>(cuda_proving_alloc_zeroes_u32_words(len));
        temps[i] = cuda_proving_malloc<uint8_t>((unsigned)temp_storage_bytes);
    }
    for (int i = 0; i < 4; ++i) {
        stwo_stream_wait_legacy(i);
        inclusive_prefix_sum_on(stwo_pool_stream(i), cols[i], tmps[i], temps[i],
                                temp_storage_bytes, len);
        stwo_legacy_wait_stream(i);
    }
    for (int i = 0; i < 4; ++i) {
        cuda_proving_free(tmps[i]);
        cuda_proving_free(temps[i]);
    }
}

