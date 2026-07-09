#include "rfft.cuh"
#include "poly_utils.cuh"
#include "utils.cuh"

// CUDA caps grid.y and grid.z at 65535 (maxGridSize[1]/[2]). Every batched NTT
// launcher maps the column (batch) axis onto grid.y or grid.z, so a same-log_size
// column group larger than this overflows the launch configuration. The column
// (batch) set is tiled into chunks of at most this many columns.
static constexpr unsigned MAX_NTT_BATCH_COLUMNS = 65535;

// log_stride = 4,3,2,1,0
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void shfl_xor_bf(m31* vals, const unsigned log_stride,
                                    const unsigned lane_id) {
  const unsigned mask = 1 << log_stride;
  const unsigned num_pair_per_thread = 1 << (LOG_VALS_PER_THREAD - 1);
  __syncwarp();
#pragma unroll
  for (unsigned i = 0; i < num_pair_per_thread; i++) {
    m31* ptr = lane_id & mask ? vals + 2 * i : vals + 2 * i + 1;
    *ptr = __shfl_xor_sync(0xffffffff, *ptr, mask);
  }
}

__global__ void rfft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_circle_twiddle(inverse_twiddles_tree, idx);

        m31 temp = mul(val1, twiddle);

        values[2 * idx] = add(val0, temp);
        values[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void rfft_line_part(m31 *values, m31 *inverse_twiddles_tree, int values_size, int inverse_twiddles_size,
                               int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        m31 temp = mul(val1, twiddle);

        values[idx0] = add(val0, temp);
        values[idx1] = sub(val0, temp);
    }
}

void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size) {
    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;

    int log_values_size = log_2(values_size);
    int layer_domain_size = 1;
    int layer_domain_offset = (values_size >> 1) - 2;
    int i = log_values_size - 1;
    while (i > 0) {
        rfft_line_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size, layer_domain_size,
                                                  layer_domain_offset, i);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    rfft_circle_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


// row_block_offset: index of this launch's first row-block along the (tiled)
// grid.y axis; idx is identical to a single launch with blockIdx.y + offset.
__global__ void batch_rfft_circle_part(m31 **values, m31 *inverse_twiddles_tree, int number_of_columns, int number_of_rows, int row_block_offset) {
    int idx = (blockIdx.y + row_block_offset) * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (idx < (number_of_rows >> 1) && column_index < number_of_columns) {
        m31 *column = values[column_index];

        m31 val0 = column[2 * idx];
        m31 val1 = column[2 * idx + 1];
        m31 twiddle = get_circle_twiddle(inverse_twiddles_tree, idx);

        m31 temp = mul(val1, twiddle);

        column[2 * idx] = add(val0, temp);
        column[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void batch_rfft_line_part(
        m31 **values, m31 *inverse_twiddles_tree, int number_of_columns, int number_of_rows, int layer_domain_offset, int layer,
        int row_block_offset
) {
    int idx = (blockIdx.y + row_block_offset) * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (idx < (number_of_rows >> 1) && column_index < number_of_columns) {
        m31 *column = values[column_index];

        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = column[idx0];
        m31 val1 = column[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        m31 temp = mul(val1, twiddle);

        column[idx0] = add(val0, temp);
        column[idx1] = sub(val0, temp);
    }
}

void evaluate_columns(const int *eval_domain_sizes, m31 **values, m31 *twiddles_tree, int twiddles_size, int number_of_columns, const int *column_sizes) {
    // TODO: Handle case where columns are of different sizes.
    int number_of_rows = column_sizes[0];
    int eval_domain_size = eval_domain_sizes[0];

    m31 **device_values = cuda_proving_clone_to_device<m31*>(values, number_of_columns);

    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];

    int block_size = 1024;
    int number_of_blocks = ((number_of_rows >> 1) + block_size - 1) / block_size;

    // The row-block axis is grid.y (CUDA cap 65535); columns ride grid.x
    // (cap 2^31-1, never workload-limited here). Tile the row-block axis:
    // each chunk covers disjoint idx values via row_block_offset, so the
    // union of the tiled launches touches exactly the same (column, idx)
    // pairs as one big launch would.
    constexpr int MAX_Y_BLOCKS = 65535;

    int log_number_of_rows = log_2(number_of_rows);
    int layer_domain_size = 1;
    int layer_domain_offset = (number_of_rows >> 1) - 2;
    int i = log_number_of_rows - 1;

    while (i > 0) {
        for (int y_base = 0; y_base < number_of_blocks; y_base += MAX_Y_BLOCKS) {
            dim3 grid_dimensions(number_of_columns, min(number_of_blocks - y_base, MAX_Y_BLOCKS));
            batch_rfft_line_part<<<grid_dimensions, block_size>>>(
                    device_values, twiddles_tree, number_of_columns, number_of_rows, layer_domain_offset, i, y_base
            );
            ASSERT_CUDA_SUCCESS(cudaGetLastError());
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    for (int y_base = 0; y_base < number_of_blocks; y_base += MAX_Y_BLOCKS) {
        dim3 grid_dimensions(number_of_columns, min(number_of_blocks - y_base, MAX_Y_BLOCKS));
        batch_rfft_circle_part<<<grid_dimensions, block_size>>>(device_values, twiddles_tree, number_of_columns, number_of_rows, y_base);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_proving_free(device_values);
}



template <unsigned LOG_VALS_PER_THREAD>
__global__ void n2b_nofinal_block_batch(m31** input, m31** output,
                                  const unsigned log_n, const unsigned num_poly,
                                  unsigned min_stage, unsigned max_stage, m31 *g_twiddles) {

    const unsigned min_stride = 1 << (log_n - max_stage);
    const unsigned middle_stage = min_stage + LOG_VALS_PER_THREAD;
    const unsigned num_stage = 2 * LOG_VALS_PER_THREAD;

    const unsigned ntt_idx = blockIdx.z;
    const unsigned block_index_x = blockIdx.x;
    const unsigned block_index_y = blockIdx.y;
    const unsigned warp_idx_in_block = threadIdx.y;
    const unsigned thread_idx_in_warp = threadIdx.x;

    const m31* input_ntt_start = input[ntt_idx];
    const unsigned block_start =
        (block_index_x << LOG_WARP) +
        (block_index_y << (log_n - max_stage + num_stage));

    __shared__ m31 smem[32 << (2 * LOG_VALS_PER_THREAD)]; // 32 * (2^(2 or 3))
    m31 vals[1 << LOG_VALS_PER_THREAD];

    unsigned offset = warp_idx_in_block * min_stride + thread_idx_in_warp;
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
            const unsigned address =
                block_start + i * (min_stride << LOG_VALS_PER_THREAD) + offset;
            vals[i] = input_ntt_start[address];
    }

    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned i = 1; i < min_stage; i++) {
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
#pragma unroll
  for (unsigned stage = min_stage; stage < middle_stage; stage++) {
    const unsigned log_inner_stride_size = LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
        const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
        const unsigned inner_pair_idx = gid >> log_inner_stride_size;
        const unsigned inner_left_idx = inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
        const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);
        const unsigned outer_pair_idx = (block_start + offset) >> (1 + log_n - stage);

        const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
        const m31 temp = vals[inner_left_idx];
        vals[inner_left_idx] = add(temp, twiddle);
        vals[inner_right_idx] = sub(temp, twiddle);

    }
    layer_domain_size <<= 1;
    layer_domain_offset -= layer_domain_size;
  }

#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++)
    smem[thread_idx_in_warp + (i << (LOG_WARP + LOG_VALS_PER_THREAD)) +
         (warp_idx_in_block << LOG_WARP)] = vals[i];
  __syncthreads();
  offset = warp_idx_in_block * (min_stride << LOG_VALS_PER_THREAD) +
           thread_idx_in_warp;
#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
    vals[i] = smem[thread_idx_in_warp + (i << LOG_WARP) +
                    (warp_idx_in_block << (LOG_WARP + LOG_VALS_PER_THREAD))];
  }

#pragma unroll
  for (unsigned stage = middle_stage; stage <= max_stage; stage++) {
    const unsigned log_inner_stride_size = LOG_VALS_PER_THREAD - 1 - (stage - middle_stage);
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
        const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
        const unsigned inner_pair_idx = gid >> log_inner_stride_size;
        const unsigned inner_left_idx = inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
        const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);
        const unsigned outer_pair_idx = (block_start + offset) >> (1 + log_n - stage);
        const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
        const m31 temp = vals[inner_left_idx];
        vals[inner_left_idx] = add(temp, twiddle);
        vals[inner_right_idx] = sub(temp, twiddle);
    }
    layer_domain_size <<= 1;
    layer_domain_offset -= layer_domain_size;
  }

  m31* output_ntt_start = output[ntt_idx];
#pragma unroll
  for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
    const unsigned address = block_start + i * min_stride + offset;
    output_ntt_start[address] = vals[i];
  }
}

static cudaError_t ntt_n2b_nofinal_6_stage_batch_on(
    m31** input, m31** output, unsigned log_n, unsigned num_poly,
    unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size,
    unsigned eval_domain_size, cudaStream_t stream
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 3;
    dim3 block_dim{32, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (log_n - end_stage);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / 32;
    grid_dim.y = (1 << log_n) / (min_stride << num_stage);

    n2b_nofinal_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        input, output, log_n, num_poly, start_stage, end_stage, g_twiddles);
    return cudaGetLastError();
}

EXTERN void ntt_n2b_nofinal_6_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly, unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_nofinal_6_stage_batch_on(
        input, output, log_n, num_poly, start_stage, g_twiddles, twiddles_size,
        eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

static cudaError_t ntt_n2b_nofinal_8_stage_batch_on(
    m31** input, m31** output, unsigned log_n, unsigned num_poly,
    unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size,
    unsigned eval_domain_size, cudaStream_t stream
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 4;
    if (log_n - start_stage + 1 < 2 * log_val_per_thread + 5) {
        return cudaErrorInvalidValue;
    }
    dim3 block_dim{32, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (log_n - end_stage);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / 32;
    grid_dim.y = (1 << log_n) / (min_stride << num_stage);

    if ((grid_dim.y * grid_dim.x * block_dim.x * block_dim.y
         << log_val_per_thread) != (1u << log_n)) {
        return cudaErrorInvalidConfiguration;
    }
    n2b_nofinal_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        input, output, log_n, num_poly, start_stage, end_stage, g_twiddles);
    return cudaGetLastError();
}

EXTERN void ntt_n2b_nofinal_8_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly, unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_nofinal_8_stage_batch_on(
        input, output, log_n, num_poly, start_stage, g_twiddles, twiddles_size,
        eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

template <unsigned LOG_VALS_PER_THREAD>
__global__ void n2b_final_warp_batch(m31** input, m31** output,
                               const unsigned log_n, const unsigned num_poly,
                               unsigned min_stage, m31 *g_twiddles) {
    const unsigned ntt_idx = blockIdx.y;
    const unsigned thread_idx_in_warp = threadIdx.x;
    const unsigned warps_idx_in_ntt = blockDim.y * blockIdx.x + threadIdx.y;
    const unsigned log_vals_per_warp = LOG_VALS_PER_THREAD + LOG_WARP;
    const m31* input_ntt_start = input[ntt_idx];
    unsigned warp_start =
        (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp;

    m31 vals[1 << LOG_VALS_PER_THREAD];
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
        vals[i] = input_ntt_start[warp_start + (i << LOG_WARP)];
    }

    // int log_values_size = log_n;
    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned i = 1; i < min_stage; i++) {
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    unsigned stage = min_stage;
    #pragma unroll
    for (; stage < min_stage + LOG_VALS_PER_THREAD; stage++) {
        const unsigned log_inner_stride_size =
            LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
    #pragma unroll
        for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
            const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
            const unsigned inner_pair_idx = gid >> log_inner_stride_size;
            const unsigned inner_left_idx =
                inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
            const unsigned inner_right_idx =
                inner_left_idx + (1 << log_inner_stride_size);
            const unsigned outer_pair_idx = warp_start >> (1 + log_n - stage);
            const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
            const m31 temp = vals[inner_left_idx];
            vals[inner_left_idx] = add(temp, twiddle);
            vals[inner_right_idx] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    #pragma unroll
    for (; stage <= log_n; stage++) {
        const unsigned log_stride = log_n - stage;
        shfl_xor_bf<LOG_VALS_PER_THREAD>(vals, log_stride, thread_idx_in_warp);
    #pragma unroll
        for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
            const unsigned inner_pair_idx =
                (thread_idx_in_warp >> log_stride) + (i << (LOG_WARP - log_stride));
            const unsigned outer_pair_idx = warps_idx_in_ntt
                                            << (log_vals_per_warp - 1 - log_stride);
            m31 twiddle = m31(1);
            if (stage == log_n) {
                twiddle = mul(get_circle_twiddle(g_twiddles, inner_pair_idx + outer_pair_idx), vals[2 * i + 1]);
            } else {
                twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[2 * i + 1]);
            }

            const m31 temp = vals[2 * i];
            vals[2 * i] = add(temp, twiddle);
            vals[2 * i + 1] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    m31* output_ntt_start = output[ntt_idx];
    warp_start = (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp * 2;
    uint64_t* src = reinterpret_cast<uint64_t*>(vals);
    uint64_t* dst = reinterpret_cast<uint64_t*>(output_ntt_start + warp_start);


    #pragma unroll
    for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
        dst[(i << LOG_WARP)] = src[i];
    }
}

static cudaError_t ntt_n2b_final_7_stage_batch_on(
    m31** input, m31** output, unsigned log_n, unsigned num_poly,
    unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size,
    unsigned eval_domain_size, cudaStream_t stream
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 2;
    if (log_n + 1 - (log_val_per_thread + LOG_WARP) != start_stage) {
        return cudaErrorInvalidValue;
    }
    dim3 block_dim = dim3{32, 1, 1};
    dim3 grid_dim = dim3{1, 1, 1};
    const unsigned num_warp = 1 << (log_n - LOG_WARP - log_val_per_thread);
    block_dim.y = min(num_warp, 4);
    grid_dim.y = num_poly;
    grid_dim.x = num_warp / block_dim.y;

    n2b_final_warp_batch<log_val_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        input, output, log_n, num_poly, start_stage, g_twiddles);
    return cudaGetLastError();
}

EXTERN void ntt_n2b_final_7_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned num_poly,
                              unsigned start_stage,
                              m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_final_7_stage_batch_on(
        input, output, log_n, num_poly, start_stage, g_twiddles, twiddles_size,
        eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

static cudaError_t ntt_n2b_final_8_stage_batch_on(
    m31** input, m31** output, unsigned log_n, unsigned num_poly,
    unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size,
    unsigned eval_domain_size, cudaStream_t stream
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 3;
    if (log_n + 1 - (log_val_per_thread + LOG_WARP) != start_stage) {
        return cudaErrorInvalidValue;
    }
    dim3 block_dim = dim3{32, 1, 1};
    dim3 grid_dim = dim3{1, 1, 1};
    const unsigned num_warp = 1 << (log_n - LOG_WARP - log_val_per_thread);
    block_dim.y = min(num_warp, 4);
    grid_dim.y = num_poly;
    grid_dim.x = num_warp / block_dim.y;

    n2b_final_warp_batch<log_val_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        input, output, log_n, num_poly, start_stage, g_twiddles);
    return cudaGetLastError();
}

EXTERN void ntt_n2b_final_8_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned num_poly,
                              unsigned start_stage,
                              m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_final_8_stage_batch_on(
        input, output, log_n, num_poly, start_stage, g_twiddles, twiddles_size,
        eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


template <unsigned LOG_WARP_PER_BLOCK>
__global__ void n2b_final_block_warp_batch(
    m31** input, m31** output, const unsigned log_n,
    const unsigned num_poly, unsigned min_stage, m31 *g_twiddles) {

    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    const unsigned ntt_idx = blockIdx.z;
    const unsigned warp_idx_in_block = threadIdx.y;
    const unsigned thread_idx_in_warp = threadIdx.x;

    const m31* input_ntt_start = input[ntt_idx];
    const unsigned block_index_y = blockIdx.x;
    const unsigned block_start = (block_index_y << (LOG_WARP + LOG_VALS_PER_THREAD + LOG_WARP_PER_BLOCK));

    __shared__ m31 smem[32 << (LOG_WARP_PER_BLOCK + LOG_VALS_PER_THREAD)];
    m31 vals[1 << LOG_VALS_PER_THREAD];

    unsigned offset = (warp_idx_in_block << LOG_WARP) + thread_idx_in_warp;
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
        const unsigned address = block_start + (i << (LOG_WARP + LOG_WARP_PER_BLOCK)) + offset;
        vals[i] = input_ntt_start[address];
    }

    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned stage = 1; stage < min_stage; stage++) {
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    unsigned stage = min_stage;
    #pragma unroll
    for (; stage < min_stage + LOG_WARP_PER_BLOCK; stage++) {
        const unsigned log_inner_stride_size = LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
    #pragma unroll
        for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
            const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
            const unsigned inner_pair_idx = gid >> log_inner_stride_size;
            const unsigned inner_left_idx =
                inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
            const unsigned inner_right_idx =
                inner_left_idx + (1 << log_inner_stride_size);
            const unsigned outer_pair_idx =
                (block_start + offset) >> (1 + log_n - stage);
            const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
            const m31 temp = vals[inner_left_idx];
            vals[inner_left_idx] = add(temp, twiddle);
            vals[inner_right_idx] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    #pragma unroll
    for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
        smem[thread_idx_in_warp + (i << (LOG_WARP + LOG_WARP_PER_BLOCK)) +
            (warp_idx_in_block << LOG_WARP)] = vals[i];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
        vals[i] = smem[thread_idx_in_warp + (i << LOG_WARP) +
                    (warp_idx_in_block << (LOG_WARP + LOG_VALS_PER_THREAD))];
    }
    offset = (warp_idx_in_block << (LOG_WARP + LOG_VALS_PER_THREAD)) +
            thread_idx_in_warp;

    const unsigned warps_idx_in_ntt = blockDim.y * block_index_y + threadIdx.y;
    const unsigned log_vals_per_warp = LOG_VALS_PER_THREAD + LOG_WARP;
    unsigned warp_start =
        (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp;
    const unsigned new_min_stage = min_stage + LOG_WARP_PER_BLOCK;
    stage = new_min_stage;
    #pragma unroll
    for (; stage < new_min_stage + LOG_VALS_PER_THREAD; stage++) {
        const unsigned log_inner_stride_size =
            LOG_VALS_PER_THREAD - 1 - (stage - new_min_stage);
    #pragma unroll
        for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
            const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
            const unsigned inner_pair_idx = gid >> log_inner_stride_size;
            const unsigned inner_left_idx =
                inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
            const unsigned inner_right_idx =
                inner_left_idx + (1 << log_inner_stride_size);
            const unsigned outer_pair_idx = warp_start >> (1 + log_n - stage);
            const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
            const m31 temp = vals[inner_left_idx];
            vals[inner_left_idx] = add(temp, twiddle);
            vals[inner_right_idx] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    #pragma unroll
    for (; stage <= log_n; stage++) {
        const unsigned log_stride = log_n - stage;
        shfl_xor_bf<LOG_VALS_PER_THREAD>(vals, log_stride, thread_idx_in_warp);
    #pragma unroll
        for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
            const unsigned inner_pair_idx =
                (thread_idx_in_warp >> log_stride) + (i << (LOG_WARP - log_stride));
            const unsigned outer_pair_idx = warps_idx_in_ntt
                                            << (log_vals_per_warp - 1 - log_stride);
            m31 twiddle = m31(1);
            if (stage == log_n) {
                twiddle = mul(get_circle_twiddle(g_twiddles, inner_pair_idx + outer_pair_idx), vals[2 * i + 1]);
            } else {
                twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[2 * i + 1]);
            }

            const m31 temp = vals[2 * i];
            vals[2 * i] = add(temp, twiddle);
            vals[2 * i + 1] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    m31* output_ntt_start = output[ntt_idx];
    warp_start = (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp * 2;
    uint64_t* src = reinterpret_cast<uint64_t*>(vals);
    uint64_t* dst = reinterpret_cast<uint64_t*>(output_ntt_start + warp_start);

    #pragma unroll
    for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++)
        dst[i << LOG_WARP] = src[i];
}

static cudaError_t ntt_n2b_final_10_stage_batch_on(
    m31** input, m31** output, unsigned log_n, unsigned num_poly,
    unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size,
    unsigned eval_domain_size, cudaStream_t stream
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 2;
    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    if (log_n + 1 - start_stage !=
        LOG_VALS_PER_THREAD + LOG_WARP + log_warp_per_block) {
        return cudaErrorInvalidValue;
    }
    dim3 block_dim = {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - LOG_WARP - LOG_VALS_PER_THREAD - log_warp_per_block);

    n2b_final_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim, 0, stream>>>(
        input, output, log_n, num_poly, start_stage, g_twiddles);
    return cudaGetLastError();
}

EXTERN void ntt_n2b_final_10_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly, unsigned start_stage,
                               m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_final_10_stage_batch_on(
        input, output, log_n, num_poly, start_stage, g_twiddles, twiddles_size,
        eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

static cudaError_t ntt_n2b_final_11_stage_batch_on(
    m31** input, m31** output, unsigned log_n, unsigned num_poly,
    unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size,
    unsigned eval_domain_size, cudaStream_t stream
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 3;
    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    if (log_n + 1 - start_stage !=
        LOG_VALS_PER_THREAD + LOG_WARP + log_warp_per_block) {
        return cudaErrorInvalidValue;
    }
    dim3 block_dim = {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - LOG_WARP - LOG_VALS_PER_THREAD - log_warp_per_block);
    n2b_final_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim, 0, stream>>>(
        input, output,  log_n, num_poly, start_stage, g_twiddles);
    return cudaGetLastError();
}

EXTERN void ntt_n2b_final_11_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly, unsigned start_stage,
                               m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_final_11_stage_batch_on(
        input, output, log_n, num_poly, start_stage, g_twiddles, twiddles_size,
        eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


__global__ void ntt_n2b_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned stage, m31 *layer_twiddles) {
    const unsigned ntt_index = blockIdx.y;
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = 1 << (log_n - stage);
    const unsigned group_idx = gid & (stride - 1);
    const unsigned pair_idx = gid >> (log_n - stage);

    const m31* input_start = input[ntt_index];
    const unsigned left_index = group_idx + pair_idx * 2 * stride;
    const unsigned right_index = left_index + stride;

    m31 left = input_start[left_index];
    m31 right = input_start[right_index];

    m31 twiddle = m31(1);
    if (stage == log_n) {
        twiddle = get_circle_twiddle(layer_twiddles, pair_idx);
    } else {
        twiddle = layer_twiddles[pair_idx];
    }


    m31 twiddle_x = mul(twiddle, right);
    const m31 temp = left;
    m31 left_r = add(temp, twiddle_x);
    m31 right_r = sub(temp, twiddle_x);

    m31* output_start = output[ntt_index];

    output_start[left_index] = left_r;
    output_start[right_index] = right_r;
}

static cudaError_t ntt_n2b_native_device_batch_on(
    m31** device_values, unsigned log_n, unsigned num_poly,
    unsigned start_stage, unsigned end_stage, m31 *g_twiddles,
    unsigned twiddles_size, unsigned eval_domain_size, cudaStream_t stream
) {
    if (start_stage < 1 || end_stage > log_n) {
        return cudaErrorInvalidValue;
    }
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    dim3 block_dim{};
    block_dim.x = log_n <= 9 ? 1 << (log_n - 1) : 256;
    dim3 grid_dim{};
    grid_dim.y = num_poly;
    grid_dim.x = log_n <= 9 ? 1 : 1 << (log_n - 9);

    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned stage = 1; stage < log_n; stage++) {

        if (stage >= start_stage) {
            ntt_n2b_stage_batch<<<grid_dim, block_dim, 0, stream>>>(
                device_values, device_values, log_n, stage,
                &g_twiddles[layer_domain_offset]);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
        }

        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    if (end_stage == log_n) {
        ntt_n2b_stage_batch<<<grid_dim, block_dim, 0, stream>>>(
            device_values, device_values, log_n, log_n, g_twiddles);
        return cudaGetLastError();
    }
    return cudaSuccess;
}

EXTERN void ntt_n2b_native_batch(m31** value,
                           unsigned log_n, unsigned num_poly,
                           unsigned start_stage, unsigned end_stage,
                           m31 *g_twiddles,
                           unsigned twiddles_size, unsigned eval_domain_size) {
    m31 **device_values = cuda_proving_clone_to_device<m31*>(value, num_poly);
    ASSERT_CUDA_SUCCESS(ntt_n2b_native_device_batch_on(
        device_values, log_n, num_poly, start_stage, end_stage, g_twiddles,
        twiddles_size, eval_domain_size, 0));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_proving_free(device_values);
}

static cudaError_t ntt_n2b_columns_dispatch_on(
    m31** device_values,
    unsigned log_n,
    unsigned num_poly,
    uint32_t* g_twiddles,
    unsigned twiddles_size,
    unsigned eval_domain_size,
    cudaStream_t stream,
    bool legacy_debug_sync
) {
    auto finish_stage = [legacy_debug_sync](cudaError_t err) -> cudaError_t {
        if (err != cudaSuccess) {
            return err;
        }
        if (legacy_debug_sync) {
            stwo_maybe_debug_sync();
            return cudaGetLastError();
        }
        return cudaSuccess;
    };
    auto nofinal = [&](unsigned n_stages, unsigned start_stage) -> cudaError_t {
        cudaError_t err;
        switch (n_stages) {
            case 6:
                err = ntt_n2b_nofinal_6_stage_batch_on(
                    device_values, device_values, log_n, num_poly, start_stage,
                    g_twiddles, twiddles_size, eval_domain_size, stream);
                break;
            case 8:
                err = ntt_n2b_nofinal_8_stage_batch_on(
                    device_values, device_values, log_n, num_poly, start_stage,
                    g_twiddles, twiddles_size, eval_domain_size, stream);
                break;
            default:
                return cudaErrorInvalidConfiguration;
        }
        return finish_stage(err);
    };
    auto final = [&](unsigned n_stages, unsigned start_stage) -> cudaError_t {
        cudaError_t err;
        switch (n_stages) {
            case 7:
                err = ntt_n2b_final_7_stage_batch_on(
                    device_values, device_values, log_n, num_poly, start_stage,
                    g_twiddles, twiddles_size, eval_domain_size, stream);
                break;
            case 8:
                err = ntt_n2b_final_8_stage_batch_on(
                    device_values, device_values, log_n, num_poly, start_stage,
                    g_twiddles, twiddles_size, eval_domain_size, stream);
                break;
            case 10:
                err = ntt_n2b_final_10_stage_batch_on(
                    device_values, device_values, log_n, num_poly, start_stage,
                    g_twiddles, twiddles_size, eval_domain_size, stream);
                break;
            case 11:
                err = ntt_n2b_final_11_stage_batch_on(
                    device_values, device_values, log_n, num_poly, start_stage,
                    g_twiddles, twiddles_size, eval_domain_size, stream);
                break;
            default:
                return cudaErrorInvalidConfiguration;
        }
        return finish_stage(err);
    };

    if (log_n < 13) {
        return finish_stage(ntt_n2b_native_device_batch_on(
            device_values, log_n, num_poly, 1, log_n, g_twiddles,
            twiddles_size, eval_domain_size, stream));
    } else if (log_n >= 13 && log_n <= 19) {
        const auto& config = LAUNCH_N2B_CONFIG_13_19[log_n - 13];
        const uint32_t start_stage1 = 1 + config[0];
        cudaError_t err = nofinal(config[0], 1);
        return err == cudaSuccess ? final(config[1], start_stage1) : err;
    } else if (log_n >= 20 && log_n <= 27) {
        const auto& config = LAUNCH_N2B_CONFIG_20_27[log_n - 20];
        const uint32_t start_stage1 = 1 + config[0];
        const uint32_t start_stage2 = start_stage1 + config[1];
        cudaError_t err = nofinal(config[0], 1);
        if (err == cudaSuccess) err = nofinal(config[1], start_stage1);
        return err == cudaSuccess ? final(config[2], start_stage2) : err;
    } else if (log_n >= 28 && log_n <= 30) {
        const auto& config = LAUNCH_N2B_CONFIG_28_30[log_n - 28];
        const uint32_t start_stage1 = 1 + config[0];
        const uint32_t start_stage2 = start_stage1 + config[1];
        const uint32_t start_stage3 = start_stage2 + config[2];
        cudaError_t err = nofinal(config[0], 1);
        if (err == cudaSuccess) err = nofinal(config[1], start_stage1);
        if (err == cudaSuccess) err = nofinal(config[2], start_stage2);
        return err == cudaSuccess ? final(config[3], start_stage3) : err;
    }
    return cudaErrorInvalidValue;
}

static void ntt_n2b_columns_dispatch(
    m31** device_values, unsigned log_n, unsigned num_poly,
    uint32_t* g_twiddles, unsigned twiddles_size, unsigned eval_domain_size
) {
    ASSERT_CUDA_SUCCESS(ntt_n2b_columns_dispatch_on(
        device_values, log_n, num_poly, g_twiddles, twiddles_size,
        eval_domain_size, 0, true));
}

// Tile the column (batch) axis into chunks of at most MAX_NTT_BATCH_COLUMNS so the
// per-launcher grid.y/grid.z (= num_poly) never exceeds CUDA's 65535 limit. Columns
// are transformed independently -- num_poly is never used for indexing inside any
// kernel; only blockIdx.{y,z} selects input[ntt_idx]/output[ntt_idx] -- so processing
// a contiguous sub-range of columns is bit-for-bit identical to one big launch. Each
// chunk is cloned from the host pointer array (values_columns + base) exactly as the
// original single call did, so memory layout and math are unchanged. Any group with
// num_poly <= 65535 (every workload before the 14M-step PIEs) runs a single iteration
// with base == 0 and behaves exactly as before.
EXTERN void ntt_n2b_columns(
    uint32_t** values_columns,
    unsigned log_n,
    unsigned num_poly,
    uint32_t* g_twiddles,
    unsigned twiddles_size,
    unsigned eval_domain_size
) {
    for (unsigned base = 0; base < num_poly; base += MAX_NTT_BATCH_COLUMNS) {
        const unsigned chunk = min(num_poly - base, MAX_NTT_BATCH_COLUMNS);
        m31 **device_values =
            cuda_proving_clone_to_device<m31*>(values_columns + base, chunk);
        // Surface any sticky error from an earlier async launch at this call site
        // instead of letting it masquerade as a failure of the launches below.
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
        ntt_n2b_columns_dispatch(device_values, log_n, chunk, g_twiddles,
                                 twiddles_size, eval_domain_size);
        cuda_proving_free(device_values);
    }
}

// Allocation-free, explicit-stream N2B transform for graph capture. `device_values`
// is already a DEVICE-resident array of device-column pointers owned by the caller.
// No upload, allocation, free, default-stream launch, or host synchronization occurs.
extern "C" int stwo_ntt_n2b_columns_on(
    uint32_t **device_values,
    unsigned log_n,
    unsigned num_poly,
    uint32_t *g_twiddles,
    unsigned twiddles_size,
    unsigned eval_domain_size,
    void *stream
) {
    if (device_values == nullptr || g_twiddles == nullptr || stream == nullptr ||
        log_n == 0 || log_n > 30 || num_poly == 0 || eval_domain_size == 0 ||
        eval_domain_size > twiddles_size) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    for (unsigned base = 0; base < num_poly; base += MAX_NTT_BATCH_COLUMNS) {
        const unsigned chunk = min(num_poly - base, MAX_NTT_BATCH_COLUMNS);
        cudaError_t err = ntt_n2b_columns_dispatch_on(
            reinterpret_cast<m31 **>(device_values + base), log_n, chunk,
            g_twiddles, twiddles_size, eval_domain_size, cuda_stream, false);
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
}

__global__ void stage_lde_columns(
    const uint32_t *const *coefficient_values,
    const uint32_t *coefficient_sizes,
    uint32_t **device_values,
    unsigned eval_domain_size
) {
    const unsigned column = blockIdx.y;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    // The safe Rust preparation layer validates this bound before uploading the
    // descriptor. Clamp defensively here so a raw FFI caller still cannot make
    // the staging kernel read beyond the NTT half-domain.
    const unsigned coefficient_count = min(coefficient_sizes[column], eval_domain_size);
    if (index < coefficient_count) {
        device_values[column][index] = coefficient_values[column][index];
    } else if (index < 2 * eval_domain_size) {
        device_values[column][index] = 0;
    }
}

// Allocation-free, explicit-stream LDE for graph capture. Both pointer tables
// and every pointed-to buffer are caller-owned device memory. Staging and N2B
// run on the supplied stream, with no host upload, allocation, free, or sync.
extern "C" int stwo_lde_n2b_columns_on(
    const uint32_t *const *coefficient_values,
    const uint32_t *coefficient_sizes,
    uint32_t **device_values,
    unsigned log_n,
    unsigned num_poly,
    uint32_t *g_twiddles,
    unsigned twiddles_size,
    unsigned eval_domain_size,
    void *stream
) {
    if (coefficient_values == nullptr || coefficient_sizes == nullptr ||
        device_values == nullptr ||
        g_twiddles == nullptr || stream == nullptr || log_n == 0 ||
        log_n > 30 || num_poly == 0 ||
        eval_domain_size != (1u << (log_n - 1)) ||
        eval_domain_size > twiddles_size) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr unsigned block_size = 256;
    const unsigned output_size = 2 * eval_domain_size;
    const unsigned grid_x = (output_size + block_size - 1) / block_size;
    for (unsigned base = 0; base < num_poly; base += MAX_NTT_BATCH_COLUMNS) {
        const unsigned chunk = min(num_poly - base, MAX_NTT_BATCH_COLUMNS);
        stage_lde_columns<<<dim3(grid_x, chunk), block_size, 0, cuda_stream>>>(
            coefficient_values + base, coefficient_sizes + base,
            device_values + base, eval_domain_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        err = ntt_n2b_columns_dispatch_on(
            reinterpret_cast<m31 **>(device_values + base), log_n, chunk,
            g_twiddles, twiddles_size, eval_domain_size, cuda_stream, false);
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
}
