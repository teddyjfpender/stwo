// LDE-write + leaf-absorb fusion UNDER RETENTION (PLAN_10MHZ Step 3.1).
//
// The producer-fused NttHash lane (rfft.cu: n2b_final_warp_hash16_batch /
// n2b_final_block_warp_hash16_batch) is disqualified when a commitment group
// must RETAIN its evaluations (retain_evaluations = true, the resident SN2
// opening plan): it absorbs the final NTT tile into the per-row blake2s leaf
// states straight from registers/shared memory and never materializes the
// completed LDE. Retained groups therefore fall back today to a full LDE
// write followed by a LeafUpdate pass that re-reads every evaluation word.
//
// This lane adds the missing variant: the SAME final-stage kernels with ONE
// added global store. After the final circle butterfly of each column, every
// thread writes its finished values back into the (in-place) evaluation
// buffer using exactly the uint64 packing and index mapping the unfused
// n2b_final_warp_batch / n2b_final_block_warp_batch kernels use. Net traffic
// for the whole group: one read of coefficients (staging), the in-place N2B
// passes, ONE write of evaluations, ZERO re-read for hashing.
//
// BYTE-IDENTITY ARGUMENT (gated by prepared_commit_native.rs, both modes in
// one invocation):
// * Evaluations: the butterfly arithmetic preceding the store is
//   instruction-for-instruction the arithmetic of rfft.cu's
//   APPLY_CIRCLE=true final kernels (same stage order, same twiddle
//   indexing, same canonical M31 primitives from fields.cu), and the store
//   uses the identical `dst[i << LOG_WARP] = src[i]` uint64 mapping, so the
//   evaluation buffer holds exactly the bytes stwo_lde_n2b_columns_on
//   writes. A store changes no computed value.
// * Digests: the absorb step is unchanged from the already-gated hash16
//   lane. All 16 columns of a row are staged through SHARED MEMORY into one
//   64-byte message block in CANONICAL COMMITTED COLUMN ORDER
//   (messages[row * 16 + column], columns walked 0..15 in the committed
//   order of the group's device pointer table) and compressed with the same
//   running byte count `4 * (cols_done + 16)` and lastblock flag that
//   stream_leaf_update_in_gpu / stream_leaf_finalize_in_gpu would use. The
//   NTT tile order never leaks into the absorb order: the shared-memory
//   transpose restores row-major canonical order before compression.
// * Ordering/races: within one column iteration, every thread's global
//   reads happen before the column's first __syncwarp (warp variant) or
//   __syncthreads (block variant), and the in-place writeback happens after
//   the last one, so stores cannot race loads; row ranges are disjoint
//   across warps and blocks, and column c+1 reads a different buffer.
//
// Eligibility is enforced by the Rust plan (commit_graph.rs RetainedNttHash,
// opt-in via STWO_CUDA_NTT_LEAF_FUSED=1, default OFF): exactly 16 columns,
// one batch, log_n == lifting_log_size >= 13, retain_evaluations = true.
// These are the SAME shape constraints as the NttHash lane; under them the
// group is a full-lifting same-log tile, so the leaf row index IS the
// evaluation index (lifted_column_index is the identity) and the absorb
// order coincides with the canonical committed order by construction.
//
// The domain-progressive twin reuses the identical transform and transpose
// under a different sink contract. Rust admits only a globally 16-aligned
// canonical block at one evaluation log >= 13. Its predecessor prefix is
// therefore empty or a whole number of BLAKE2s blocks. The kernel compresses
// the old pending block when non-empty, then installs the register-resident
// final NTT tile as the new lazy pending block. A later progressive absorb or
// finalize observes exactly the state produced by the separate absorb kernel.
// Retained columns keep the identical final uint64 stores; dead unretained
// columns omit only that final materialization. The message transpose and
// canonical absorb are independent of this write mask.

#include "ntt_leaf_fused.cuh"
#include "blake2s.cuh"
#include "poly_utils.cuh"
#include "rfft.cuh"
#include "utils.cuh"

namespace {

static_assert(offsetof(ProgressiveBlake2sState, h) == 0,
              "progressive chaining value must prefix the state");
static_assert(sizeof(Blake2sHash) == sizeof(uint32_t) * 8,
              "progressive chaining value must match Blake2sHash");

// Duplicate of rfft.cu's file-local shfl_xor_bf (the butterfly operand
// exchange for the in-register final stages), under a unique name so the two
// translation units cannot collide at device link. Kept in LOCKSTEP with
// rfft.cu; the byte-identity gate catches any divergence.
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void shfl_xor_bf_fused(m31 *vals, const unsigned log_stride,
                                          const unsigned lane_id) {
  const unsigned mask = 1 << log_stride;
  const unsigned num_pair_per_thread = 1 << (LOG_VALS_PER_THREAD - 1);
  __syncwarp();
#pragma unroll
  for (unsigned i = 0; i < num_pair_per_thread; i++) {
    m31 *ptr = lane_id & mask ? vals + 2 * i : vals + 2 * i + 1;
    *ptr = __shfl_xor_sync(0xffffffff, *ptr, mask);
  }
}

// Both consumers see the same register-resident canonical row message. The
// streaming leaf consumes it immediately; the progressive leaf keeps the
// newest full block pending, exactly matching Blake2sHasher's lazy update
// rule. At an aligned boundary, a non-empty progressive state necessarily has
// one previous full block pending, so compress that block before replacing it.
template <bool PROGRESSIVE>
DEVICE_FORCEINLINE void consume_fused_leaf_message(
    void *raw_states,
    unsigned row,
    const uint32_t message[16],
    uint32_t cols_done,
    uint32_t is_final
) {
    if constexpr (PROGRESSIVE) {
        ProgressiveBlake2sState *state =
            &reinterpret_cast<ProgressiveBlake2sState *>(raw_states)[row];
        if (cols_done != 0) {
            stwo_blake2s_compress_leaf_block_device(
                reinterpret_cast<Blake2sHash *>(state), state->pending,
                4u * cols_done, 0u);
        }
        #pragma unroll
        for (unsigned word = 0; word < 16; ++word) {
            state->pending[word] = message[word];
        }
    } else {
        const uint32_t total_bytes = 4u * (cols_done + 16u);
        const uint32_t lastblock = is_final != 0 ? 0xffffffffu : 0u;
        stwo_blake2s_compress_leaf_block_device(
            &reinterpret_cast<Blake2sHash *>(raw_states)[row], message,
            total_bytes, lastblock);
    }
}

template <bool PROGRESSIVE>
DEVICE_FORCEINLINE bool writes_completed_evaluation(
    uint32_t retained_write_mask, unsigned column
) {
    if constexpr (PROGRESSIVE) {
        return (retained_write_mask & (1u << column)) != 0;
    }
    return true;
}

// n2b_final_warp_hash16_batch (rfft.cu) + the evaluation writeback. Rows
// [global_warp_start, global_warp_start + VALUES_PER_WARP) are owned by one
// warp for both the store and the absorb.
template <unsigned LOG_VALS_PER_THREAD, bool PROGRESSIVE>
__global__ void n2b_final_warp_hash16_write_batch(
    m31 **values,   // in: prefinal state; out: completed evaluations
    const unsigned log_n,
    unsigned min_stage,
    m31 *g_twiddles,
    uint32_t cols_done,
    uint32_t is_final,
    uint32_t retained_write_mask,
    void *states
) {
    extern __shared__ uint32_t messages[];
    const unsigned lane = threadIdx.x;
    const unsigned local_warp = threadIdx.y;
    const unsigned global_warp = blockDim.y * blockIdx.x + local_warp;
    const unsigned log_values_per_warp = LOG_VALS_PER_THREAD + LOG_WARP;
    const unsigned global_warp_start = global_warp << log_values_per_warp;
    const unsigned local_warp_start = local_warp << log_values_per_warp;

    for (unsigned column = 0; column < 16; ++column) {
        m31 *column_values = values[column];
        unsigned warp_start = global_warp_start + lane;
        m31 vals[1 << LOG_VALS_PER_THREAD];
        #pragma unroll
        for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); ++i) {
            vals[i] = column_values[warp_start + (i << LOG_WARP)];
        }

        unsigned layer_domain_size = 1;
        unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;
        for (unsigned i = 1; i < min_stage; ++i) {
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }
        unsigned stage = min_stage;
        #pragma unroll
        for (; stage < min_stage + LOG_VALS_PER_THREAD; ++stage) {
            const unsigned log_inner_stride =
                LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
            #pragma unroll
            for (unsigned gid = 0; gid < (1 << (LOG_VALS_PER_THREAD - 1)); ++gid) {
                const unsigned inner_group = gid & ((1 << log_inner_stride) - 1);
                const unsigned inner_pair = gid >> log_inner_stride;
                const unsigned left_index =
                    inner_group + (inner_pair << (log_inner_stride + 1));
                const unsigned right_index = left_index + (1 << log_inner_stride);
                const unsigned outer_pair = warp_start >> (1 + log_n - stage);
                const m31 product = mul(
                    g_twiddles[layer_domain_offset + inner_pair + outer_pair],
                    vals[right_index]);
                const m31 left = vals[left_index];
                vals[left_index] = add(left, product);
                vals[right_index] = sub(left, product);
            }
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }
        #pragma unroll
        for (; stage <= log_n; ++stage) {
            const unsigned log_stride = log_n - stage;
            shfl_xor_bf_fused<LOG_VALS_PER_THREAD>(vals, log_stride, lane);
            #pragma unroll
            for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
                const unsigned inner_pair =
                    (lane >> log_stride) + (i << (LOG_WARP - log_stride));
                const unsigned outer_pair = global_warp
                    << (log_values_per_warp - 1 - log_stride);
                const m31 product = stage == log_n
                    ? mul(get_circle_twiddle(g_twiddles, inner_pair + outer_pair),
                          vals[2 * i + 1])
                    : mul(g_twiddles[layer_domain_offset + inner_pair + outer_pair],
                          vals[2 * i + 1]);
                const m31 left = vals[2 * i];
                vals[2 * i] = add(left, product);
                vals[2 * i + 1] = sub(left, product);
            }
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }

        // Evaluation writeback — the identical uint64 store
        // n2b_final_warp_batch performs, so the retained buffer holds
        // exactly the unfused lane's bytes. Safe in place: all loads of
        // this column happened before the shfl __syncwarp barriers above.
        if (writes_completed_evaluation<PROGRESSIVE>(retained_write_mask, column)) {
            uint64_t *src = reinterpret_cast<uint64_t *>(vals);
            uint64_t *dst = reinterpret_cast<uint64_t *>(
                column_values + global_warp_start + 2 * lane);
            #pragma unroll
            for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
                dst[i << LOG_WARP] = src[i];
            }
        }

        #pragma unroll
        for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
            const unsigned row = local_warp_start + 2 * lane + (i << 6);
            messages[(row + 0) * 16 + column] = vals[2 * i];
            messages[(row + 1) * 16 + column] = vals[2 * i + 1];
        }
    }
    __syncthreads();

    #pragma unroll
    for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
        #pragma unroll
        for (unsigned side = 0; side < 2; ++side) {
            const unsigned local_row = local_warp_start + 2 * lane + (i << 6) + side;
            const unsigned global_row = global_warp_start + 2 * lane + (i << 6) + side;
            uint32_t message[16];
            #pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
                message[k] = messages[local_row * 16 + k];
            }
            consume_fused_leaf_message<PROGRESSIVE>(
                states, global_row, message, cols_done, is_final);
        }
    }
}

// n2b_final_block_warp_hash16_batch (rfft.cu) + the evaluation writeback.
template <unsigned LOG_WARP_PER_BLOCK, bool PROGRESSIVE>
__global__ void n2b_final_block_warp_hash16_write_batch(
    m31 **values,   // in: prefinal state; out: completed evaluations
    const unsigned log_n,
    unsigned min_stage,
    m31 *g_twiddles,
    uint32_t cols_done,
    uint32_t is_final,
    uint32_t retained_write_mask,
    void *states
) {
    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    constexpr unsigned VALUES_PER_WARP = 1 << (LOG_WARP + LOG_VALS_PER_THREAD);
    constexpr unsigned VALUES_PER_BLOCK = 32 << (LOG_WARP_PER_BLOCK + LOG_VALS_PER_THREAD);
    extern __shared__ uint32_t shared_words[];
    m31 *smem = reinterpret_cast<m31 *>(shared_words);
    uint32_t *messages = shared_words + VALUES_PER_BLOCK;

    const unsigned local_warp = threadIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned block_start = blockIdx.x
        << (LOG_WARP + LOG_VALS_PER_THREAD + LOG_WARP_PER_BLOCK);
    const unsigned global_warp = blockDim.y * blockIdx.x + local_warp;
    const unsigned global_warp_start = global_warp << (LOG_WARP + LOG_VALS_PER_THREAD);
    const unsigned local_warp_start = local_warp * VALUES_PER_WARP;

    for (unsigned column = 0; column < 16; ++column) {
        m31 *column_values = values[column];
        m31 vals[1 << LOG_VALS_PER_THREAD];
        unsigned offset = (local_warp << LOG_WARP) + lane;
        #pragma unroll
        for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); ++i) {
            vals[i] = column_values[
                block_start + (i << (LOG_WARP + LOG_WARP_PER_BLOCK)) + offset];
        }

        unsigned layer_domain_size = 1;
        unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;
        for (unsigned stage = 1; stage < min_stage; ++stage) {
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }
        unsigned stage = min_stage;
        #pragma unroll
        for (; stage < min_stage + LOG_WARP_PER_BLOCK; ++stage) {
            const unsigned log_inner_stride =
                LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
            #pragma unroll
            for (unsigned gid = 0; gid < (1 << (LOG_VALS_PER_THREAD - 1)); ++gid) {
                const unsigned inner_group = gid & ((1 << log_inner_stride) - 1);
                const unsigned inner_pair = gid >> log_inner_stride;
                const unsigned left_index =
                    inner_group + (inner_pair << (log_inner_stride + 1));
                const unsigned right_index = left_index + (1 << log_inner_stride);
                const unsigned outer_pair = (block_start + offset) >> (1 + log_n - stage);
                const m31 product = mul(
                    g_twiddles[layer_domain_offset + inner_pair + outer_pair],
                    vals[right_index]);
                const m31 left = vals[left_index];
                vals[left_index] = add(left, product);
                vals[right_index] = sub(left, product);
            }
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }

        #pragma unroll
        for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); ++i) {
            smem[lane + (i << (LOG_WARP + LOG_WARP_PER_BLOCK))
                + (local_warp << LOG_WARP)] = vals[i];
        }
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); ++i) {
            vals[i] = smem[lane + (i << LOG_WARP)
                + (local_warp << (LOG_WARP + LOG_VALS_PER_THREAD))];
        }
        offset = (local_warp << (LOG_WARP + LOG_VALS_PER_THREAD)) + lane;

        const unsigned new_min_stage = min_stage + LOG_WARP_PER_BLOCK;
        stage = new_min_stage;
        #pragma unroll
        for (; stage < new_min_stage + LOG_VALS_PER_THREAD; ++stage) {
            const unsigned log_inner_stride =
                LOG_VALS_PER_THREAD - 1 - (stage - new_min_stage);
            #pragma unroll
            for (unsigned gid = 0; gid < (1 << (LOG_VALS_PER_THREAD - 1)); ++gid) {
                const unsigned inner_group = gid & ((1 << log_inner_stride) - 1);
                const unsigned inner_pair = gid >> log_inner_stride;
                const unsigned left_index =
                    inner_group + (inner_pair << (log_inner_stride + 1));
                const unsigned right_index = left_index + (1 << log_inner_stride);
                const unsigned outer_pair =
                    (global_warp_start + lane) >> (1 + log_n - stage);
                const m31 product = mul(
                    g_twiddles[layer_domain_offset + inner_pair + outer_pair],
                    vals[right_index]);
                const m31 left = vals[left_index];
                vals[left_index] = add(left, product);
                vals[right_index] = sub(left, product);
            }
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }
        #pragma unroll
        for (; stage <= log_n; ++stage) {
            const unsigned log_stride = log_n - stage;
            shfl_xor_bf_fused<LOG_VALS_PER_THREAD>(vals, log_stride, lane);
            #pragma unroll
            for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
                const unsigned inner_pair =
                    (lane >> log_stride) + (i << (LOG_WARP - log_stride));
                const unsigned outer_pair = global_warp
                    << (LOG_WARP + LOG_VALS_PER_THREAD - 1 - log_stride);
                const m31 product = stage == log_n
                    ? mul(get_circle_twiddle(g_twiddles, inner_pair + outer_pair),
                          vals[2 * i + 1])
                    : mul(g_twiddles[layer_domain_offset + inner_pair + outer_pair],
                          vals[2 * i + 1]);
                const m31 left = vals[2 * i];
                vals[2 * i] = add(left, product);
                vals[2 * i + 1] = sub(left, product);
            }
            layer_domain_size <<= 1;
            layer_domain_offset -= layer_domain_size;
        }

        // Evaluation writeback — the identical uint64 store
        // n2b_final_block_warp_batch performs. Safe in place: every warp's
        // loads of this column precede the smem __syncthreads above, and the
        // written rows are disjoint per warp/block.
        if (writes_completed_evaluation<PROGRESSIVE>(retained_write_mask, column)) {
            uint64_t *src = reinterpret_cast<uint64_t *>(vals);
            uint64_t *dst = reinterpret_cast<uint64_t *>(
                column_values + global_warp_start + 2 * lane);
            #pragma unroll
            for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
                dst[i << LOG_WARP] = src[i];
            }
        }

        #pragma unroll
        for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
            const unsigned row = local_warp_start + 2 * lane + (i << 6);
            messages[(row + 0) * 16 + column] = vals[2 * i];
            messages[(row + 1) * 16 + column] = vals[2 * i + 1];
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i = 0; i < (1 << (LOG_VALS_PER_THREAD - 1)); ++i) {
        #pragma unroll
        for (unsigned side = 0; side < 2; ++side) {
            const unsigned local_row = local_warp_start + 2 * lane + (i << 6) + side;
            const unsigned global_row = global_warp_start + 2 * lane + (i << 6) + side;
            uint32_t message[16];
            #pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
                message[k] = messages[local_row * 16 + k];
            }
            consume_fused_leaf_message<PROGRESSIVE>(
                states, global_row, message, cols_done, is_final);
        }
    }
}

// Final-stage count for `log_n` — the last LAUNCH_N2B_CONFIG entry, i.e. the
// same row rfft.cu's dispatchers read. 0 marks an unsupported log size.
unsigned leaf_fused_final_stages(unsigned log_n) {
    if (log_n >= 13 && log_n <= 19) {
        return (unsigned)LAUNCH_N2B_CONFIG_13_19[log_n - 13][1];
    }
    if (log_n >= 20 && log_n <= 27) {
        return (unsigned)LAUNCH_N2B_CONFIG_20_27[log_n - 20][2];
    }
    if (log_n >= 28 && log_n <= 30) {
        return (unsigned)LAUNCH_N2B_CONFIG_28_30[log_n - 28][3];
    }
    return 0;
}

template <unsigned LOG_VALS_PER_THREAD, bool PROGRESSIVE>
cudaError_t leaf_fused_final_warp_on(
    m31 **values,
    unsigned log_n,
    unsigned start_stage,
    m31 *twiddles,
    unsigned twiddle_words,
    unsigned eval_domain_size,
    uint32_t cols_done,
    uint32_t is_final,
    uint32_t retained_write_mask,
    void *states,
    cudaStream_t stream
) {
    if (log_n + 1 - (LOG_VALS_PER_THREAD + LOG_WARP) != start_stage) {
        return cudaErrorInvalidValue;
    }
    twiddles += twiddle_words - eval_domain_size;
    const unsigned num_warps = 1 << (log_n - LOG_WARP - LOG_VALS_PER_THREAD);
    dim3 block_dim{32, min(num_warps, 4u), 1};
    dim3 grid_dim{num_warps / block_dim.y, 1, 1};
    const size_t shared_bytes = size_t(block_dim.y)
        * (1u << (LOG_WARP + LOG_VALS_PER_THREAD)) * 16u * sizeof(uint32_t);
    n2b_final_warp_hash16_write_batch<LOG_VALS_PER_THREAD, PROGRESSIVE>
        <<<grid_dim, block_dim, shared_bytes, stream>>>(
            values, log_n, start_stage, twiddles, cols_done, is_final,
            retained_write_mask, states);
    return cudaGetLastError();
}

template <unsigned LOG_WARP_PER_BLOCK, bool PROGRESSIVE>
cudaError_t leaf_fused_final_block_on(
    m31 **values,
    unsigned log_n,
    unsigned start_stage,
    m31 *twiddles,
    unsigned twiddle_words,
    unsigned eval_domain_size,
    uint32_t cols_done,
    uint32_t is_final,
    uint32_t retained_write_mask,
    void *states,
    cudaStream_t stream
) {
    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    if (log_n + 1 - start_stage !=
        LOG_VALS_PER_THREAD + LOG_WARP + LOG_WARP_PER_BLOCK) {
        return cudaErrorInvalidValue;
    }
    twiddles += twiddle_words - eval_domain_size;
    dim3 block_dim{32, 1u << LOG_WARP_PER_BLOCK, 1};
    dim3 grid_dim{1u << (log_n - LOG_WARP - LOG_VALS_PER_THREAD
        - LOG_WARP_PER_BLOCK), 1, 1};
    constexpr size_t values_per_block =
        32u << (LOG_WARP_PER_BLOCK + LOG_VALS_PER_THREAD);
    constexpr size_t shared_bytes = values_per_block * 17u * sizeof(uint32_t);
    n2b_final_block_warp_hash16_write_batch<LOG_WARP_PER_BLOCK, PROGRESSIVE>
        <<<grid_dim, block_dim, shared_bytes, stream>>>(
            values, log_n, start_stage, twiddles, cols_done, is_final,
            retained_write_mask, states);
    return cudaGetLastError();
}

template <bool PROGRESSIVE>
cudaError_t configure_leaf_fused(unsigned log_n) {
    switch (leaf_fused_final_stages(log_n)) {
        case 7:
            return cudaFuncSetAttribute(
                n2b_final_warp_hash16_write_batch<2, PROGRESSIVE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024);
        case 8:
            return cudaFuncSetAttribute(
                n2b_final_warp_hash16_write_batch<3, PROGRESSIVE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        case 10:
            return cudaFuncSetAttribute(
                n2b_final_block_warp_hash16_write_batch<2, PROGRESSIVE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 68 * 1024);
        case 11:
            return cudaFuncSetAttribute(
                n2b_final_block_warp_hash16_write_batch<3, PROGRESSIVE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 136 * 1024);
        default:
            return cudaErrorInvalidValue;
    }
}

template <bool PROGRESSIVE>
int launch_leaf_fused(
    const uint32_t *const *coefficient_values,
    const uint32_t *coefficient_sizes,
    uint32_t **device_values,
    unsigned log_n,
    uint32_t *twiddles,
    unsigned twiddle_words,
    unsigned eval_domain_size,
    uint32_t cols_done,
    uint32_t is_final,
    uint32_t retained_write_mask,
    void *states,
    void *stream
) {
    int err = stwo_lde_n2b_prefinal16_on(
        coefficient_values, coefficient_sizes, device_values, log_n, twiddles,
        twiddle_words, eval_domain_size, stream);
    if (err != (int)cudaSuccess) {
        return err;
    }

    const unsigned final_stages = leaf_fused_final_stages(log_n);
    const unsigned final_start = log_n + 1 - final_stages;
    m31 **values = reinterpret_cast<m31 **>(device_values);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (final_stages) {
        case 7:
            return leaf_fused_final_warp_on<2, PROGRESSIVE>(
                values, log_n, final_start, twiddles, twiddle_words,
                eval_domain_size, cols_done, is_final, retained_write_mask,
                states, cuda_stream);
        case 8:
            return leaf_fused_final_warp_on<3, PROGRESSIVE>(
                values, log_n, final_start, twiddles, twiddle_words,
                eval_domain_size, cols_done, is_final, retained_write_mask,
                states, cuda_stream);
        case 10:
            return leaf_fused_final_block_on<2, PROGRESSIVE>(
                values, log_n, final_start, twiddles, twiddle_words,
                eval_domain_size, cols_done, is_final, retained_write_mask,
                states, cuda_stream);
        case 11:
            return leaf_fused_final_block_on<3, PROGRESSIVE>(
                values, log_n, final_start, twiddles, twiddle_words,
                eval_domain_size, cols_done, is_final, retained_write_mask,
                states, cuda_stream);
        default:
            return cudaErrorInvalidConfiguration;
    }
}

} // namespace

// Same dynamic shared-memory ceilings as rfft.cu's
// configure_n2b_hash16_kernel, applied to the write+hash twins.
extern "C" int stwo_ntt_leaf_fused_configure(unsigned log_n) {
    return configure_leaf_fused<false>(log_n);
}

extern "C" int stwo_ntt_leaf_fused_on(
    const uint32_t *const *coefficient_values,
    const uint32_t *coefficient_sizes,
    uint32_t **device_values,
    unsigned log_n,
    uint32_t *twiddles,
    unsigned twiddle_words,
    unsigned eval_domain_size,
    uint32_t cols_done,
    uint32_t is_final,
    Blake2sHash *states,
    void *stream
) {
    // Same admission contract as stwo_lde_n2b_hash16_on.
    if (coefficient_values == nullptr || coefficient_sizes == nullptr ||
        device_values == nullptr || log_n < 13 || log_n > 30 ||
        twiddles == nullptr || eval_domain_size != (1u << (log_n - 1)) ||
        eval_domain_size > twiddle_words || (cols_done % 16) != 0 ||
        is_final > 1 || states == nullptr || stream == nullptr) {
        return cudaErrorInvalidValue;
    }

    return launch_leaf_fused<false>(
        coefficient_values, coefficient_sizes, device_values, log_n, twiddles,
        twiddle_words, eval_domain_size, cols_done, is_final, 0xffffu, states,
        stream);
}

extern "C" int stwo_ntt_progressive_leaf_fused_configure(unsigned log_n) {
    return configure_leaf_fused<true>(log_n);
}

extern "C" int stwo_ntt_progressive_leaf_fused_on(
    const uint32_t *const *coefficient_values,
    const uint32_t *coefficient_sizes,
    uint32_t **device_values,
    unsigned log_n,
    uint32_t *twiddles,
    unsigned twiddle_words,
    unsigned eval_domain_size,
    uint32_t cols_done,
    uint32_t retained_write_mask,
    ProgressiveBlake2sState *states,
    void *stream
) {
    if (coefficient_values == nullptr || coefficient_sizes == nullptr ||
        device_values == nullptr || log_n < 13 || log_n > 30 ||
        twiddles == nullptr || eval_domain_size != (1u << (log_n - 1)) ||
        eval_domain_size > twiddle_words || (cols_done % 16) != 0 ||
        (retained_write_mask & ~0xffffu) != 0 || states == nullptr ||
        stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    return launch_leaf_fused<true>(
        coefficient_values, coefficient_sizes, device_values, log_n, twiddles,
        twiddle_words, eval_domain_size, cols_done, 0u, retained_write_mask,
        states, stream);
}
