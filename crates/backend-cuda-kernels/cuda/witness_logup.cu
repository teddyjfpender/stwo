// Generic device logup-input kernels for witness-on-GPU component ports.
//
// Every generated interaction writer in stwo-cairo follows one of two shapes,
// per packed row:
//   pair:   d_i = combine([REL_ID_i const, staged tuple columns_i...]),
//           numerator = sign * (d0*m1 + d1*m0), denominator = d0*d1
//   single: d = combine([REL_ID const, staged tuple columns...]),
//           numerator = (sign * m, 0, 0, 0), denominator = d
// with combine(v) = sum_k alphas[k]*v_k - z and multiplicity m one of:
//   - the constant 1                        (mult_col == nullptr, enabler == ~0)
//   - a trace/staged column                 (mult_col != nullptr)
//   - the Enabler pattern: 1 iff row < off  (mult_col == nullptr, enabler == off)
// The staged tuple columns are device buffers written by the component's base
// kernel (the device analogue of host `lookup_data`). Outputs feed
// finalize_device_raw_logup unchanged.

#include "batch_inverse.cuh"
#include "fields.cuh"
#include "utils.cuh"

namespace {

constexpr uint32_t WL_BLOCK = 256;
constexpr uint32_t WL_NO_ENABLER = 0xFFFFFFFFu;

DEVICE_FORCEINLINE qm31 wl_mul_m31(qm31 x, m31 s) {
    return qm31{cm31{mul(x.a.a, s), mul(x.a.b, s)}, cm31{mul(x.b.a, s), mul(x.b.b, s)}};
}

DEVICE_FORCEINLINE qm31 wl_combine(
    uint32_t rel_id,
    const uint32_t *const *cols,
    uint32_t n_cols,
    uint32_t row,
    const qm31 *alphas,
    qm31 z
) {
    qm31 acc = wl_mul_m31(alphas[0], rel_id);
    for (uint32_t k = 0; k < n_cols; ++k) {
        acc = add(acc, wl_mul_m31(alphas[1 + k], cols[k][row]));
    }
    return sub(acc, z);
}

DEVICE_FORCEINLINE m31 wl_mult(
    const uint32_t *mult_col,
    uint32_t enabler_offset,
    uint32_t row
) {
    if (mult_col != nullptr) {
        return mult_col[row];
    }
    if (enabler_offset != WL_NO_ENABLER) {
        return row < enabler_offset ? 1u : 0u;
    }
    return 1u;
}

__global__ void tuple_pair_logup_kernel(
    uint32_t rel_id0, const uint32_t *const *cols0, uint32_t n0,
    uint32_t rel_id1, const uint32_t *const *cols1, uint32_t n1,
    const uint32_t *mult0_col, uint32_t enabler0,
    const uint32_t *mult1_col, uint32_t enabler1,
    uint32_t negate,
    uint32_t column_length,
    const qm31 *alphas,
    qm31 z,
    qm31 *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    qm31 d0 = wl_combine(rel_id0, cols0, n0, row, alphas, z);
    qm31 d1 = wl_combine(rel_id1, cols1, n1, row, alphas, z);
    m31 m0 = wl_mult(mult0_col, enabler0, row);
    m31 m1 = wl_mult(mult1_col, enabler1, row);
    qm31 numerator = add(wl_mul_m31(d0, m1), wl_mul_m31(d1, m0));
    if (negate) {
        numerator = neg(numerator);
    }
    denoms[row] = mul(d0, d1);
    num0[row] = numerator.a.a;
    num1[row] = numerator.a.b;
    num2[row] = numerator.b.a;
    num3[row] = numerator.b.b;
}

__global__ void tuple_single_logup_kernel(
    uint32_t rel_id, const uint32_t *const *cols, uint32_t n,
    const uint32_t *mult_col, uint32_t enabler, uint32_t negate,
    uint32_t column_length,
    const qm31 *alphas,
    qm31 z,
    qm31 *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    denoms[row] = wl_combine(rel_id, cols, n, row, alphas, z);
    m31 m = wl_mult(mult_col, enabler, row);
    num0[row] = negate ? neg(m) : m;
    num1[row] = 0;
    num2[row] = 0;
    num3[row] = 0;
}

// Counts width-W tuples (column-tuple groups) into relation-indexed count
// tables through a dense input->row LUT keyed by base-B digits:
//   key = sum_j v_j * B^(W-1-j)  with B = 1 << log_base
// (the rc tables' canonical packing; rc_9_9: W=2, log_base=9). Padding rows
// included, like every host feeding loop.
__global__ void tuple_count_kernel(
    const uint32_t *const *cols,   // n_tuples * width pointers, tuple-major
    uint32_t n_tuples,
    uint32_t width,
    const uint32_t *slot_bits,     // width entries: key = fold(key << bits_j | v_j)
    uint32_t n_relations,
    uint32_t column_length,
    const uint32_t *input_to_row_lut,
    uint32_t table_size,
    uint32_t *counts                // n_relations * table_size
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    for (uint32_t t = 0; t < n_tuples; ++t) {
        uint32_t key = 0;
        for (uint32_t j = 0; j < width; ++j) {
            key = (key << slot_bits[j]) | cols[t * width + j][row];
        }
        uint32_t rc_row = input_to_row_lut[key];
        atomicAdd(&counts[(t % n_relations) * (size_t)table_size + rc_row], 1u);
    }
}

}  // namespace

extern "C" void tuple_pair_logup(
    uint32_t rel_id0, const uint32_t *const *cols0, uint32_t n0,
    uint32_t rel_id1, const uint32_t *const *cols1, uint32_t n1,
    const uint32_t *mult0_col, uint32_t enabler0,
    const uint32_t *mult1_col, uint32_t enabler1,
    uint32_t negate,
    uint32_t column_length,
    const uint32_t *alphas,
    qm31 z,
    uint32_t *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t blocks = (column_length + WL_BLOCK - 1) / WL_BLOCK;
    tuple_pair_logup_kernel<<<blocks, WL_BLOCK>>>(
        rel_id0, cols0, n0, rel_id1, cols1, n1, mult0_col, enabler0, mult1_col, enabler1,
        negate, column_length, reinterpret_cast<const qm31 *>(alphas), z,
        reinterpret_cast<qm31 *>(denoms), num0, num1, num2, num3);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void tuple_single_logup(
    uint32_t rel_id, const uint32_t *const *cols, uint32_t n,
    const uint32_t *mult_col, uint32_t enabler, uint32_t negate,
    uint32_t column_length,
    const uint32_t *alphas,
    qm31 z,
    uint32_t *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t blocks = (column_length + WL_BLOCK - 1) / WL_BLOCK;
    tuple_single_logup_kernel<<<blocks, WL_BLOCK>>>(
        rel_id, cols, n, mult_col, enabler, negate, column_length,
        reinterpret_cast<const qm31 *>(alphas), z,
        reinterpret_cast<qm31 *>(denoms), num0, num1, num2, num3);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void tuple_count(
    const uint32_t *const *cols,
    uint32_t n_tuples,
    uint32_t width,
    const uint32_t *slot_bits,
    uint32_t n_relations,
    uint32_t column_length,
    const uint32_t *input_to_row_lut,
    uint32_t table_size,
    uint32_t *counts
) {
    uint32_t blocks = (column_length + WL_BLOCK - 1) / WL_BLOCK;
    tuple_count_kernel<<<blocks, WL_BLOCK>>>(
        cols, n_tuples, width, slot_bits, n_relations, column_length, input_to_row_lut,
        table_size, counts);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// verify_instruction base trace (W3 opcode-cohort beachhead): the generated
// SIMD writer's decode math, formula-for-formula, over 9 uploaded input
// columns (pc, off0..2, felt5_high, felt6, opcode_ext, instruction_id, mult;
// the instruction id is resolved host-side at the dedup'd table size). Writes
// the 17 trace columns plus the 3 staged combination columns the interaction
// tuples need (enc1 = off0_mid + off1_low*128, enc3 = off1_high + off2_low*32,
// enc5b = off2_high + felt5_high; sums stay < P, no reduction needed).
// ---------------------------------------------------------------------------
__global__ void verify_instruction_trace_kernel(
    const uint32_t *pc, const uint32_t *off0, const uint32_t *off1, const uint32_t *off2,
    const uint32_t *felt5_high, const uint32_t *felt6, const uint32_t *opcode_ext,
    const uint32_t *instruction_id, const uint32_t *mult,
    uint32_t column_length,
    uint32_t *const *trace,    // 17 trace columns
    uint32_t *enc1, uint32_t *enc3, uint32_t *enc5b
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v0 = off0[row], v1 = off1[row], v2 = off2[row];
    uint32_t v5h = felt5_high[row], v6 = felt6[row], vext = opcode_ext[row];
    trace[0][row] = v_pc;
    trace[1][row] = v0;
    trace[2][row] = v1;
    trace[3][row] = v2;
    trace[4][row] = v5h;
    trace[5][row] = v6;
    trace[6][row] = vext;
    uint32_t off0_low = v0 & 511u;
    uint32_t off0_mid = v0 >> 9;
    uint32_t off1_low = v1 & 3u;
    uint32_t off1_mid = (v1 >> 2) & 511u;
    uint32_t off1_high = v1 >> 11;
    uint32_t off2_low = v2 & 15u;
    uint32_t off2_mid = (v2 >> 4) & 511u;
    uint32_t off2_high = v2 >> 13;
    trace[7][row] = off0_low;
    trace[8][row] = off0_mid;
    trace[9][row] = off1_low;
    trace[10][row] = off1_mid;
    trace[11][row] = off1_high;
    trace[12][row] = off2_low;
    trace[13][row] = off2_mid;
    trace[14][row] = off2_high;
    trace[15][row] = instruction_id[row];
    trace[16][row] = mult[row];
    // M31 additions, exactly like the host's PackedM31 ops (enc1/enc3 operands
    // are bit-bounded well below P; felt5_high is only bounded by P, so the
    // modular add is required there — and harmless on the others).
    enc1[row] = add(off0_mid, mul(off1_low, 128u));
    enc3[row] = add(off1_high, mul(off2_low, 32u));
    enc5b[row] = add(off2_high, v5h);
}

extern "C" void verify_instruction_trace(
    const uint32_t *pc, const uint32_t *off0, const uint32_t *off1, const uint32_t *off2,
    const uint32_t *felt5_high, const uint32_t *felt6, const uint32_t *opcode_ext,
    const uint32_t *instruction_id, const uint32_t *mult,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *enc1, uint32_t *enc3, uint32_t *enc5b
) {
    uint32_t blocks = (column_length + WL_BLOCK - 1) / WL_BLOCK;
    verify_instruction_trace_kernel<<<blocks, WL_BLOCK>>>(
        pc, off0, off1, off2, felt5_high, felt6, opcode_ext, instruction_id, mult,
        column_length, const_cast<uint32_t *const *>(trace), enc1, enc3, enc5b);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
