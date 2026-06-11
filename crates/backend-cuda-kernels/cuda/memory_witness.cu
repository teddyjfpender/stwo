// Device witness generation for the Cairo memory tables (witness-on-GPU P1).
//
// Ports of the generated SIMD writers' math, formula-for-formula (see
// stwo-cairo `witness/components/memory_id_to_big.rs` and the spec in
// gpu_benchmarks/WITNESS_ON_GPU.md):
//
// - 252-bit limb split: walk the 8 (big) / 4 (small) little-endian u32 words,
//   emitting 28 (resp. 8) limbs of 9 bits each — identical to
//   `split_f252` in stwo-cairo-common (LSB-first bit-buffer walk).
// - rc_9_9 multiplicity feed: per row, limb pairs (2j, 2j+1) count into the
//   relation-indexed (j % 8) multiplicity tables through the rc table's
//   input->row LUT (a table-layout mapping, uploaded once — NOT closed-form).
//   Padding rows count their (0, 0) pairs exactly like the host loop does.
// - Logup denominators: combine(values) = sum_i alpha_powers[i] * values[i] - z
//   (the constraint-framework LookupElements formula) over
//   [RELATION_ID, offset + row, limb_0..limb_27]; numerators are (-mult, 0, 0, 0).
//
// All counts are order-independent (field/integer adds), so atomics preserve
// byte-equality by construction. The decisive gates live in stwo-cairo: a
// device-vs-host writer differential and the Cairo e2e proof byte-equality.

#include "batch_inverse.cuh"
#include "fields.cuh"
#include "utils.cuh"

namespace {

constexpr uint32_t MW_BLOCK = 256;
constexpr uint32_t FELT252_BITS_PER_WORD = 9;
constexpr uint32_t LIMB_MASK = (1u << FELT252_BITS_PER_WORD) - 1;

// LSB-first split of `n_words` 32-bit words into `n_limbs` 9-bit limbs —
// line-for-line the generic `split` in stwo-cairo-common/prover_types/felt.rs.
template <int N_WORDS, int N_LIMBS>
DEVICE_FORCEINLINE void split_le_9bit(const uint32_t *words, uint32_t *limbs) {
    uint32_t n_bits_in_word = 32;
    uint32_t word_i = 0;
    uint32_t word = words[0];
    for (int e = 0; e < N_LIMBS; ++e) {
        if (n_bits_in_word > FELT252_BITS_PER_WORD) {
            limbs[e] = word & LIMB_MASK;
            word >>= FELT252_BITS_PER_WORD;
            n_bits_in_word -= FELT252_BITS_PER_WORD;
            continue;
        }
        limbs[e] = word;
        word_i += 1;
        word = word_i < N_WORDS ? words[word_i] : 0;
        if (n_bits_in_word < FELT252_BITS_PER_WORD) {
            limbs[e] |= (word << n_bits_in_word) & LIMB_MASK;
            word >>= FELT252_BITS_PER_WORD - n_bits_in_word;
        }
        n_bits_in_word += 32 - FELT252_BITS_PER_WORD;
    }
}

// One thread per output row. Rows past `n_values` are padding: all-zero limbs
// (matching the host's zero-extension of the value table).
template <int N_WORDS, int N_LIMBS>
__global__ void memory_limb_split_kernel(
    const uint32_t *values,  // n_values * N_WORDS words, row-major
    uint32_t n_values,
    uint32_t column_length,
    uint32_t *const *limb_cols  // N_LIMBS device pointers, column_length each
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    uint32_t words[N_WORDS] = {0};
    if (row < n_values) {
        for (int w = 0; w < N_WORDS; ++w) {
            words[w] = values[(size_t)row * N_WORDS + w];
        }
    }
    uint32_t limbs[N_LIMBS];
    split_le_9bit<N_WORDS, N_LIMBS>(words, limbs);
    for (int j = 0; j < N_LIMBS; ++j) {
        limb_cols[j][row] = limbs[j];
    }
}

// rc_9_9 feed: per row, limb pairs (2j, 2j+1) for j in [0, n_pairs) count into
// counts[(j % 8) * rc_table_size + lut[v0 * 512 + v1]]. Includes padding rows,
// exactly like the host's par_iter over the full columns.
__global__ void rc99_count_kernel(
    const uint32_t *const *limb_cols,
    uint32_t n_pairs,
    uint32_t column_length,
    const uint32_t *input_to_row_lut,  // 2^18 entries: (v0 << 9 | v1) -> rc row
    uint32_t rc_table_size,
    uint32_t *counts  // 8 * rc_table_size
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    for (uint32_t j = 0; j < n_pairs; ++j) {
        uint32_t v0 = limb_cols[2 * j][row];
        uint32_t v1 = limb_cols[2 * j + 1][row];
        uint32_t rc_row = input_to_row_lut[(v0 << FELT252_BITS_PER_WORD) | v1];
        atomicAdd(&counts[(j % 8) * (size_t)rc_table_size + rc_row], 1u);
    }
}

DEVICE_FORCEINLINE qm31 qm31_mul_m31(qm31 x, m31 s) {
    return qm31{cm31{mul(x.a.a, s), mul(x.a.b, s)}, cm31{mul(x.b.a, s), mul(x.b.b, s)}};
}

// Final memory-relation column for one segment:
//   id[row]   = (id_offset + row) | id_tag     (plain u32; host guarantees < P)
//   denom     = alpha[0]*REL_ID + alpha[1]*id + sum_k alpha[2+k]*limb_k - z
//   numerator = (-mult[row], 0, 0, 0)
// alphas/z are channel-drawn QM31 params; limbs/mults are M31 device columns.
// Output denominators are element-major qm31 (the finalize lane's layout).
__global__ void memory_logup_inputs_kernel(
    const uint32_t *const *limb_cols,
    uint32_t n_limbs,
    const uint32_t *mults,
    uint32_t relation_id,
    uint32_t id_offset,
    uint32_t id_tag,
    uint32_t column_length,
    const qm31 *alpha_powers,  // n_limbs + 2 entries
    qm31 z,
    qm31 *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    uint32_t id = (id_offset + row) | id_tag;
    qm31 acc = qm31_mul_m31(alpha_powers[0], relation_id);
    acc = add(acc, qm31_mul_m31(alpha_powers[1], id));
    for (uint32_t k = 0; k < n_limbs; ++k) {
        acc = add(acc, qm31_mul_m31(alpha_powers[2 + k], limb_cols[k][row]));
    }
    denoms[row] = sub(acc, z);
    num0[row] = neg(mults[row]);
    num1[row] = 0;
    num2[row] = 0;
    num3[row] = 0;
}

// One pair-batched rc_9_9 logup column (the host writes 7 of these per big
// segment, 2 per small table):
//   d0 = alpha[0]*rel_id0 + alpha[1]*limb_a + alpha[2]*limb_b - z
//   d1 = alpha[0]*rel_id1 + alpha[1]*limb_c + alpha[2]*limb_d - z
//   numerator = d0 + d1   (both lookups have multiplicity 1)
//   denominator = d0 * d1
__global__ void memory_rc_pair_logup_kernel(
    const uint32_t *limb_a, const uint32_t *limb_b,
    const uint32_t *limb_c, const uint32_t *limb_d,
    uint32_t rel_id0,
    uint32_t rel_id1,
    uint32_t column_length,
    const qm31 *alpha_powers,  // 3 entries
    qm31 z,
    qm31 *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    qm31 d0 = qm31_mul_m31(alpha_powers[0], rel_id0);
    d0 = add(d0, qm31_mul_m31(alpha_powers[1], limb_a[row]));
    d0 = add(d0, qm31_mul_m31(alpha_powers[2], limb_b[row]));
    d0 = sub(d0, z);
    qm31 d1 = qm31_mul_m31(alpha_powers[0], rel_id1);
    d1 = add(d1, qm31_mul_m31(alpha_powers[1], limb_c[row]));
    d1 = add(d1, qm31_mul_m31(alpha_powers[2], limb_d[row]));
    d1 = sub(d1, z);
    qm31 numerator = add(d0, d1);
    denoms[row] = mul(d0, d1);
    num0[row] = numerator.a.a;
    num1[row] = numerator.a.b;
    num2[row] = numerator.b.a;
    num3[row] = numerator.b.b;
}

}  // namespace

extern "C" void memory_limb_split_big(
    const uint32_t *values,
    uint32_t n_values,
    uint32_t column_length,
    const uint32_t *const *limb_cols  // 28 device pointers (device-resident table)
) {
    uint32_t blocks = (column_length + MW_BLOCK - 1) / MW_BLOCK;
    memory_limb_split_kernel<8, 28><<<blocks, MW_BLOCK>>>(
        values, n_values, column_length, const_cast<uint32_t *const *>(limb_cols));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void memory_limb_split_small(
    const uint32_t *values,  // 4 words per value (u128 LE)
    uint32_t n_values,
    uint32_t column_length,
    const uint32_t *const *limb_cols  // 8 device pointers
) {
    uint32_t blocks = (column_length + MW_BLOCK - 1) / MW_BLOCK;
    memory_limb_split_kernel<4, 8><<<blocks, MW_BLOCK>>>(
        values, n_values, column_length, const_cast<uint32_t *const *>(limb_cols));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void memory_rc99_count(
    const uint32_t *const *limb_cols,
    uint32_t n_pairs,
    uint32_t column_length,
    const uint32_t *input_to_row_lut,
    uint32_t rc_table_size,
    uint32_t *counts
) {
    uint32_t blocks = (column_length + MW_BLOCK - 1) / MW_BLOCK;
    rc99_count_kernel<<<blocks, MW_BLOCK>>>(
        limb_cols, n_pairs, column_length, input_to_row_lut, rc_table_size, counts);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void memory_logup_inputs(
    const uint32_t *const *limb_cols,
    uint32_t n_limbs,
    const uint32_t *mults,
    uint32_t relation_id,
    uint32_t id_offset,
    uint32_t id_tag,
    uint32_t column_length,
    const uint32_t *alpha_powers,  // (n_limbs + 2) qm31s, element-major
    qm31 z,
    uint32_t *denoms,  // column_length qm31s, element-major
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t blocks = (column_length + MW_BLOCK - 1) / MW_BLOCK;
    memory_logup_inputs_kernel<<<blocks, MW_BLOCK>>>(
        limb_cols, n_limbs, mults, relation_id, id_offset, id_tag, column_length,
        reinterpret_cast<const qm31 *>(alpha_powers), z,
        reinterpret_cast<qm31 *>(denoms), num0, num1, num2, num3);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void memory_rc_pair_logup(
    const uint32_t *limb_a, const uint32_t *limb_b,
    const uint32_t *limb_c, const uint32_t *limb_d,
    uint32_t rel_id0,
    uint32_t rel_id1,
    uint32_t column_length,
    const uint32_t *alpha_powers,  // 3 qm31s, element-major
    qm31 z,
    uint32_t *denoms,
    uint32_t *num0, uint32_t *num1, uint32_t *num2, uint32_t *num3
) {
    uint32_t blocks = (column_length + MW_BLOCK - 1) / MW_BLOCK;
    memory_rc_pair_logup_kernel<<<blocks, MW_BLOCK>>>(
        limb_a, limb_b, limb_c, limb_d, rel_id0, rel_id1, column_length,
        reinterpret_cast<const qm31 *>(alpha_powers), z,
        reinterpret_cast<qm31 *>(denoms), num0, num1, num2, num3);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
