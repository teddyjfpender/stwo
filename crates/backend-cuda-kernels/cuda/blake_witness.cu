// Device witness generation for the Cairo blake_g component (witness-on-GPU W3
// phase 2 — the BLAKE g-function family, the most GPU-natural port: 32-bit
// modular arithmetic + xor-rotations).
//
// Ports the generated SIMD writer's math formula-for-formula (see stwo-cairo
// `witness/components/blake_g.rs`). The blake_g g-function mixes 6 input words
// through 4 triple-sums (mod 2^32) and 4 xor-rotations (r16/r12/r8/r7), emitting
// 53 trace columns plus the lookup operands feeding 5 verify_bitwise_xor families
// and the blake_g relation.
//
// Semantics matched exactly against the host types (crates/common prover_types):
//   PackedUInt32 add  -> wrapping u32 add               (Blake2s mod-2^32 add)
//   .low()            -> x & 0xFFFF                      (low 16 bits, u16)
//   .high()           -> x >> 16                         (high 16 bits, u16)
//   from_limbs([l,h]) -> l + (h << 16)                   (wrapping)
//   split part[0]     -> col - ms_bits * 2^k             (canonical, in [0, 2^k))
//   xor               -> u16 bitwise xor
// All split operands are provably non-negative and < 2^16, so plain u32 integer
// arithmetic yields the canonical M31 representative — byte-identical to the
// host's PackedM31 field ops. The decisive gates live in stwo-cairo: a device-vs
// -host writer differential (STWO_CUDA_WITNESS_VERIFY) and the Cairo e2e proof
// byte-equality.

#include "fields.cuh"
#include "utils.cuh"

namespace {

constexpr uint32_t BG_BLOCK = 256;

// Number of committed trace columns (cairo-air blake_g N_TRACE_COLUMNS) and the
// auxiliary operand columns the interaction/count feed needs but the trace does
// not carry (the "split low part" values and the rot7/rot8 limbs).
constexpr int BG_N_TRACE = 53;
constexpr int BG_N_AUX = 20;
constexpr int BG_N_COLS = BG_N_TRACE + BG_N_AUX;  // 73

DEVICE_FORCEINLINE uint32_t lo16(uint32_t x) { return x & 0xFFFFu; }
DEVICE_FORCEINLINE uint32_t hi16(uint32_t x) { return x >> 16; }

// One thread per output row. Writes all 73 columns (0..52 trace, 53..72 aux).
// `inputs` is column_length * 6 raw u32 words, row-major (6 blake_g input words
// per row; padding rows already carry the host's replicated first input).
__global__ void blake_g_write_trace_kernel(
    const uint32_t *inputs,
    uint32_t n_rows,        // real (non-padding) rows; enabler = row < n_rows
    uint32_t column_length,
    uint32_t *const *cols   // BG_N_COLS device pointers, column_length each
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    const uint32_t *in = inputs + (size_t)row * 6;
    uint32_t in0 = in[0], in1 = in[1], in2 = in[2], in3 = in[3], in4 = in[4], in5 = in[5];

    uint32_t c[BG_N_COLS];

    // Input limbs (cols 0..11).
    c[0] = lo16(in0);  c[1] = hi16(in0);
    c[2] = lo16(in1);  c[3] = hi16(in1);
    c[4] = lo16(in2);  c[5] = hi16(in2);
    c[6] = lo16(in3);  c[7] = hi16(in3);
    c[8] = lo16(in4);  c[9] = hi16(in4);
    c[10] = lo16(in5); c[11] = hi16(in5);

    // Triple Sum 32: ts0 = in0 + in1 + in4.
    uint32_t ts0 = in0 + in1 + in4;
    c[12] = lo16(ts0); c[13] = hi16(ts0);

    // Xor Rot 32 R 16.
    c[14] = lo16(ts0) >> 8;                 // ms_8_bits(ts0.low)
    uint32_t s5_0 = c[12] - c[14] * 256;
    c[15] = hi16(ts0) >> 8;                 // ms_8_bits(ts0.high)
    uint32_t s7_0 = c[13] - c[15] * 256;
    c[16] = lo16(in3) >> 8;                 // ms_8_bits(in3.low)
    uint32_t s9_0 = c[6] - c[16] * 256;
    c[17] = hi16(in3) >> 8;                 // ms_8_bits(in3.high)
    uint32_t s11_0 = c[7] - c[17] * 256;
    c[18] = s5_0 ^ s9_0;
    c[19] = c[14] ^ c[16];
    c[20] = s7_0 ^ s11_0;
    c[21] = c[15] ^ c[17];
    uint32_t xr16_low = c[20] + c[21] * 256;
    uint32_t xr16_high = c[18] + c[19] * 256;
    uint32_t xr16 = xr16_low + (xr16_high << 16);

    // Triple Sum 32: ts22 = in2 + xr16.
    uint32_t ts22 = in2 + xr16;
    c[22] = lo16(ts22); c[23] = hi16(ts22);

    // Xor Rot 32 R 12.
    c[24] = lo16(in1) >> 12;                // ms_4_bits(in1.low)
    uint32_t s27_0 = c[2] - c[24] * 4096;
    c[25] = hi16(in1) >> 12;                // ms_4_bits(in1.high)
    uint32_t s29_0 = c[3] - c[25] * 4096;
    c[26] = lo16(ts22) >> 12;               // ms_4_bits(ts22.low)
    uint32_t s31_0 = c[22] - c[26] * 4096;
    c[27] = hi16(ts22) >> 12;               // ms_4_bits(ts22.high)
    uint32_t s33_0 = c[23] - c[27] * 4096;
    c[28] = s27_0 ^ s31_0;
    c[29] = c[24] ^ c[26];
    c[30] = s29_0 ^ s33_0;
    c[31] = c[25] ^ c[27];
    uint32_t xr12_low = c[29] + c[30] * 16;
    uint32_t xr12_high = c[31] + c[28] * 16;
    uint32_t xr12 = xr12_low + (xr12_high << 16);

    // Triple Sum 32: ts44 = ts0 + xr12 + in5.
    uint32_t ts44 = ts0 + xr12 + in5;
    c[32] = lo16(ts44); c[33] = hi16(ts44);

    // Xor Rot 32 R 8.
    c[34] = lo16(ts44) >> 8;
    uint32_t s49_0 = c[32] - c[34] * 256;
    c[35] = hi16(ts44) >> 8;
    uint32_t s51_0 = c[33] - c[35] * 256;
    c[36] = lo16(xr16) >> 8;
    uint32_t s53_0 = xr16_low - c[36] * 256;
    c[37] = hi16(xr16) >> 8;
    uint32_t s55_0 = xr16_high - c[37] * 256;
    c[38] = s49_0 ^ s53_0;
    c[39] = c[34] ^ c[36];
    c[40] = s51_0 ^ s55_0;
    c[41] = c[35] ^ c[37];
    uint32_t xr8_low = c[39] + c[40] * 256;
    uint32_t xr8_high = c[41] + c[38] * 256;
    uint32_t xr8 = xr8_low + (xr8_high << 16);

    // Triple Sum 32: ts66 = ts22 + xr8.
    uint32_t ts66 = ts22 + xr8;
    c[42] = lo16(ts66); c[43] = hi16(ts66);

    // Xor Rot 32 R 7.
    c[44] = lo16(xr12) >> 7;
    uint32_t s71_0 = xr12_low - c[44] * 128;
    c[45] = hi16(xr12) >> 7;
    uint32_t s73_0 = xr12_high - c[45] * 128;
    c[46] = lo16(ts66) >> 7;
    uint32_t s75_0 = c[42] - c[46] * 128;
    c[47] = hi16(ts66) >> 7;
    uint32_t s77_0 = c[43] - c[47] * 128;
    c[48] = s71_0 ^ s75_0;
    c[49] = c[44] ^ c[46];
    c[50] = s73_0 ^ s77_0;
    c[51] = c[45] ^ c[47];
    uint32_t xr7_low = c[49] + c[50] * 512;
    uint32_t xr7_high = c[51] + c[48] * 512;

    // Enabler (col 52).
    c[52] = (row < n_rows) ? 1u : 0u;

    // Aux operand columns.
    c[53] = s5_0;  c[54] = s7_0;  c[55] = s9_0;  c[56] = s11_0;
    c[57] = s27_0; c[58] = s29_0; c[59] = s31_0; c[60] = s33_0;
    c[61] = s49_0; c[62] = s51_0; c[63] = s53_0; c[64] = s55_0;
    c[65] = s71_0; c[66] = s73_0; c[67] = s75_0; c[68] = s77_0;
    c[69] = xr7_low; c[70] = xr7_high; c[71] = xr8_low; c[72] = xr8_high;

    for (int j = 0; j < BG_N_COLS; ++j) {
        cols[j][row] = c[j];
    }
}

// Generic xor multiplicity count feed. For each of `n_pairs` (a, b) column pairs,
// atomicAdd 1 into counts[rel_idx[k] * table_size + lut[(a << shift) | b]] over
// every row (padding included, exactly like the host par_iter over full columns).
__global__ void blake_g_xor_count_kernel(
    const uint32_t *const *a_cols,
    const uint32_t *const *b_cols,
    const uint32_t *rel_idx,
    uint32_t n_pairs,
    uint32_t column_length,
    uint32_t shift,
    const uint32_t *lut,
    uint32_t table_size,
    uint32_t *counts
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    for (uint32_t k = 0; k < n_pairs; ++k) {
        uint32_t a = a_cols[k][row];
        uint32_t b = b_cols[k][row];
        uint32_t rc_row = lut[(a << shift) | b];
        atomicAdd(&counts[rel_idx[k] * (size_t)table_size + rc_row], 1u);
    }
}

// xor_12 uses the EXPANDED table: the multiplicity column is chosen by the
// EXPAND_BITS MSBs of a and b, the row by the LIMB_BITS LSBs — a closed form
// (no LUT), replicating the host `add_input`.
//   column_index = (ah << expand_bits) + bh,  row_index = (al << limb_bits) + bl.
__global__ void blake_g_xor12_count_kernel(
    const uint32_t *const *a_cols,
    const uint32_t *const *b_cols,
    uint32_t n_pairs,
    uint32_t column_length,
    uint32_t limb_bits,
    uint32_t expand_bits,
    uint32_t table_size,
    uint32_t *counts
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    uint32_t limb_mask = (1u << limb_bits) - 1u;
    for (uint32_t k = 0; k < n_pairs; ++k) {
        uint32_t a = a_cols[k][row];
        uint32_t b = b_cols[k][row];
        uint32_t ci = ((a >> limb_bits) << expand_bits) + (b >> limb_bits);
        uint32_t ri = ((a & limb_mask) << limb_bits) + (b & limb_mask);
        atomicAdd(&counts[ci * (size_t)table_size + ri], 1u);
    }
}

DEVICE_FORCEINLINE qm31 qm31_mul_m31(qm31 x, m31 s) {
    return qm31{cm31{mul(x.a.a, s), mul(x.a.b, s)}, cm31{mul(x.b.a, s), mul(x.b.b, s)}};
}

// combine([rel, a, b, xor]) = alpha[0]*rel + alpha[1]*a + alpha[2]*b + alpha[3]*xor - z.
DEVICE_FORCEINLINE qm31 combine4(
    const qm31 *alpha, qm31 z, uint32_t rel, uint32_t a, uint32_t b, uint32_t x) {
    qm31 acc = qm31_mul_m31(alpha[0], rel);
    acc = add(acc, qm31_mul_m31(alpha[1], a));
    acc = add(acc, qm31_mul_m31(alpha[2], b));
    acc = add(acc, qm31_mul_m31(alpha[3], x));
    return sub(acc, z);
}

// One pair-batched xor logup column: both lookups have multiplicity 1, so
//   numerator = d0 + d1, denominator = d0 * d1
// with d_i = combine([rel_i, a_i, b_i, xor_i]).
__global__ void blake_g_pair_logup_kernel(
    const uint32_t *a0, const uint32_t *b0, const uint32_t *x0,
    const uint32_t *a1, const uint32_t *b1, const uint32_t *x1,
    uint32_t rel0, uint32_t rel1,
    uint32_t column_length,
    const qm31 *alpha,  // 4 entries
    qm31 z,
    qm31 *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    qm31 d0 = combine4(alpha, z, rel0, a0[row], b0[row], x0[row]);
    qm31 d1 = combine4(alpha, z, rel1, a1[row], b1[row], x1[row]);
    qm31 num = add(d0, d1);
    denoms[row] = mul(d0, d1);
    num0[row] = num.a.a;
    num1[row] = num.a.b;
    num2[row] = num.b.a;
    num3[row] = num.b.b;
}

// The final blake_g-relation logup column:
//   denom = combine([rel, v_0 .. v_19]) over 21 alpha powers
//   numerator = (-enabler, 0, 0, 0)
__global__ void blake_g_final_logup_kernel(
    const uint32_t *const *val_cols,  // 20 device pointers (the non-const values)
    const uint32_t *enabler,
    uint32_t rel,
    uint32_t column_length,
    const qm31 *alpha,  // 21 entries
    qm31 z,
    qm31 *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    qm31 acc = qm31_mul_m31(alpha[0], rel);
    for (int k = 0; k < 20; ++k) {
        acc = add(acc, qm31_mul_m31(alpha[1 + k], val_cols[k][row]));
    }
    denoms[row] = sub(acc, z);
    num0[row] = neg(enabler[row]);
    num1[row] = 0;
    num2[row] = 0;
    num3[row] = 0;
}

}  // namespace

extern "C" void blake_g_write_trace(
    const uint32_t *inputs,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *cols  // 73 device pointers (device-resident table)
) {
    uint32_t blocks = (column_length + BG_BLOCK - 1) / BG_BLOCK;
    blake_g_write_trace_kernel<<<blocks, BG_BLOCK>>>(
        inputs, n_rows, column_length, const_cast<uint32_t *const *>(cols));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void blake_g_xor_count(
    const uint32_t *const *a_cols,
    const uint32_t *const *b_cols,
    const uint32_t *rel_idx,
    uint32_t n_pairs,
    uint32_t column_length,
    uint32_t shift,
    const uint32_t *lut,
    uint32_t table_size,
    uint32_t *counts
) {
    uint32_t blocks = (column_length + BG_BLOCK - 1) / BG_BLOCK;
    blake_g_xor_count_kernel<<<blocks, BG_BLOCK>>>(
        a_cols, b_cols, rel_idx, n_pairs, column_length, shift, lut, table_size, counts);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void blake_g_xor12_count(
    const uint32_t *const *a_cols,
    const uint32_t *const *b_cols,
    uint32_t n_pairs,
    uint32_t column_length,
    uint32_t limb_bits,
    uint32_t expand_bits,
    uint32_t table_size,
    uint32_t *counts
) {
    uint32_t blocks = (column_length + BG_BLOCK - 1) / BG_BLOCK;
    blake_g_xor12_count_kernel<<<blocks, BG_BLOCK>>>(
        a_cols, b_cols, n_pairs, column_length, limb_bits, expand_bits, table_size, counts);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void blake_g_pair_logup(
    const uint32_t *a0, const uint32_t *b0, const uint32_t *x0,
    const uint32_t *a1, const uint32_t *b1, const uint32_t *x1,
    uint32_t rel0, uint32_t rel1,
    uint32_t column_length,
    const uint32_t *alpha,  // 4 qm31s, element-major
    qm31 z,
    uint32_t *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t blocks = (column_length + BG_BLOCK - 1) / BG_BLOCK;
    blake_g_pair_logup_kernel<<<blocks, BG_BLOCK>>>(
        a0, b0, x0, a1, b1, x1, rel0, rel1, column_length,
        reinterpret_cast<const qm31 *>(alpha), z,
        reinterpret_cast<qm31 *>(denoms), num0, num1, num2, num3);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void blake_g_final_logup(
    const uint32_t *const *val_cols,
    const uint32_t *enabler,
    uint32_t rel,
    uint32_t column_length,
    const uint32_t *alpha,  // 21 qm31s, element-major
    qm31 z,
    uint32_t *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t blocks = (column_length + BG_BLOCK - 1) / BG_BLOCK;
    blake_g_final_logup_kernel<<<blocks, BG_BLOCK>>>(
        val_cols, enabler, rel, column_length,
        reinterpret_cast<const qm31 *>(alpha), z,
        reinterpret_cast<qm31 *>(denoms), num0, num1, num2, num3);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// Device edge (B3): build blake_g's ROW-MAJOR 6-word input buffer directly
// from blake_round's word-major sub buffer (8 instances x 6 raw u32 words at
// word_base) — the device-to-device replacement for the host input feed.
// Instance-major stacking; padding rows replicate the first packed row's
// lanes (the host resize rule) — identical layout to the host-built upload.
__global__ void blake_g_inputs_from_sub_kernel(
    const uint32_t *producer_sub,
    uint32_t producer_rows,
    uint32_t word_base,
    uint32_t n_instances,
    uint32_t consumer_rows,
    uint32_t *out_row_major
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= consumer_rows) {
        return;
    }
    uint32_t real_rows = n_instances * producer_rows;
    uint32_t src = row < real_rows ? row : (row & 15u);
    uint32_t j = src / producer_rows;
    uint32_t r = src % producer_rows;
    for (uint32_t w = 0; w < 6; ++w) {
        out_row_major[(size_t)row * 6 + w] =
            producer_sub[(size_t)(word_base + j * 6 + w) * producer_rows + r];
    }
}

extern "C" int stwo_blake_g_inputs_from_sub(
    const uint32_t *producer_sub_dev,
    uint32_t producer_rows,
    uint32_t word_base,
    uint32_t n_instances,
    uint32_t consumer_rows,
    uint32_t *out_row_major_dev
) {
    if (consumer_rows == 0) {
        return 0;
    }
    if ((size_t)n_instances * producer_rows > consumer_rows) {
        fprintf(stderr, "stwo_blake_g_inputs_from_sub: consumer_rows too small\n");
        return 1;
    }
    const uint32_t block = 256;
    uint32_t grid = (consumer_rows + block - 1) / block;
    blake_g_inputs_from_sub_kernel<<<grid, block>>>(
        producer_sub_dev, producer_rows, word_base, n_instances, consumer_rows,
        out_row_major_dev);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "stwo_blake_g_inputs_from_sub: launch failed\n");
        return 1;
    }
    return 0;
}
