typedef unsigned long long u64;

#define STWO_M31_P 2147483647u

__device__ __forceinline__ unsigned stwo_m31_add(unsigned lhs, unsigned rhs) {
    unsigned sum = lhs + rhs;
    return sum >= STWO_M31_P ? sum - STWO_M31_P : sum;
}

__device__ __forceinline__ unsigned stwo_m31_sub(unsigned lhs, unsigned rhs) {
    return lhs >= rhs ? lhs - rhs : lhs + STWO_M31_P - rhs;
}

__device__ __forceinline__ unsigned stwo_m31_neg(unsigned value) {
    unsigned negated = STWO_M31_P - value;
    return negated == STWO_M31_P ? 0u : negated;
}

__device__ __forceinline__ unsigned stwo_m31_mul(unsigned lhs, unsigned rhs) {
    u64 product = (u64)lhs * (u64)rhs;
    u64 reduced = (((((product >> 31) + product + 1u) >> 31) + product) & (u64)STWO_M31_P);
    return (unsigned)reduced;
}

__device__ __forceinline__ unsigned stwo_m31_square(unsigned value) {
    return stwo_m31_mul(value, value);
}

__device__ __forceinline__ unsigned stwo_m31_pow2k(unsigned squarings, unsigned value) {
    unsigned result = value;
    for (unsigned i = 0; i < squarings; ++i) { result = stwo_m31_square(result); }
    return result;
}

__device__ __forceinline__ unsigned stwo_m31_inv(unsigned value) {
    unsigned t0 = stwo_m31_mul(stwo_m31_pow2k(2u, value), value);
    unsigned t1 = stwo_m31_mul(stwo_m31_pow2k(1u, t0), t0);
    unsigned t2 = stwo_m31_mul(stwo_m31_pow2k(3u, t1), t0);
    unsigned t3 = stwo_m31_mul(stwo_m31_pow2k(1u, t2), t0);
    unsigned t4 = stwo_m31_mul(stwo_m31_pow2k(8u, t3), t3);
    unsigned t5 = stwo_m31_mul(stwo_m31_pow2k(8u, t4), t3);
    return stwo_m31_mul(stwo_m31_pow2k(7u, t5), t2);
}

struct StwoCudaQm31 { unsigned a, b, c, d; };

__device__ __forceinline__ StwoCudaQm31 stwo_qm31_add(StwoCudaQm31 l, StwoCudaQm31 r) {
    return StwoCudaQm31{stwo_m31_add(l.a, r.a), stwo_m31_add(l.b, r.b),
                        stwo_m31_add(l.c, r.c), stwo_m31_add(l.d, r.d)};
}

__device__ __forceinline__ StwoCudaQm31 stwo_qm31_sub(StwoCudaQm31 l, StwoCudaQm31 r) {
    return StwoCudaQm31{stwo_m31_sub(l.a, r.a), stwo_m31_sub(l.b, r.b),
                        stwo_m31_sub(l.c, r.c), stwo_m31_sub(l.d, r.d)};
}

__device__ __forceinline__ StwoCudaQm31 stwo_qm31_mul_base(StwoCudaQm31 v, unsigned s) {
    return StwoCudaQm31{stwo_m31_mul(v.a, s), stwo_m31_mul(v.b, s),
                        stwo_m31_mul(v.c, s), stwo_m31_mul(v.d, s)};
}

__device__ __forceinline__ StwoCudaQm31 stwo_qm31_mul(StwoCudaQm31 l, StwoCudaQm31 r) {
    unsigned a0 = l.a, a1 = l.b, a2 = l.c, a3 = l.d;
    unsigned b0 = r.a, b1 = r.b, b2 = r.c, b3 = r.d;
    unsigned x0 = stwo_m31_sub(stwo_m31_mul(a0, b0), stwo_m31_mul(a1, b1));
    unsigned x1 = stwo_m31_add(stwo_m31_mul(a0, b1), stwo_m31_mul(a1, b0));
    unsigned y0 = stwo_m31_sub(stwo_m31_mul(a2, b2), stwo_m31_mul(a3, b3));
    unsigned y1 = stwo_m31_add(stwo_m31_mul(a2, b3), stwo_m31_mul(a3, b2));
    unsigned c0 = stwo_m31_sub(stwo_m31_mul(a0, b2), stwo_m31_mul(a1, b3));
    unsigned c1 = stwo_m31_add(stwo_m31_mul(a0, b3), stwo_m31_mul(a1, b2));
    unsigned c2 = stwo_m31_sub(stwo_m31_mul(a2, b0), stwo_m31_mul(a3, b1));
    unsigned c3 = stwo_m31_add(stwo_m31_mul(a2, b1), stwo_m31_mul(a3, b0));
    unsigned ry0 = stwo_m31_sub(stwo_m31_mul(2u, y0), y1);
    unsigned ry1 = stwo_m31_add(y0, stwo_m31_mul(2u, y1));
    return StwoCudaQm31{stwo_m31_add(x0, ry0), stwo_m31_add(x1, ry1),
                        stwo_m31_add(c0, c2), stwo_m31_add(c1, c3)};
}

__device__ __forceinline__ StwoCudaQm31 stwo_load_qm31(const unsigned *values, unsigned index) {
    unsigned base = index * 4u;
    return StwoCudaQm31{values[base], values[base + 1u], values[base + 2u], values[base + 3u]};
}

__device__ __forceinline__ unsigned stwo_bit_reverse(unsigned index, unsigned bits) {
    return __brev(index) >> (32u - bits);
}

__device__ __forceinline__ unsigned stwo_offset_bit_reversed_circle_domain_index(
    unsigned i, unsigned domain_log_size, unsigned eval_log_size, int offset
) {
    unsigned prev = stwo_bit_reverse(i, eval_log_size);
    unsigned half_size = 1u << (eval_log_size - 1u);
    int step = offset * (int)(1u << (eval_log_size - domain_log_size - 1u));
    if (prev < half_size) {
        int p = ((int)prev + step) % (int)half_size;
        if (p < 0) p += (int)half_size;
        prev = (unsigned)p;
    } else {
        int p = (int)prev - step;
        p = p % (int)half_size;
        if (p < 0) p += (int)half_size;
        prev = (unsigned)p + half_size;
    }
    return stwo_bit_reverse(prev, eval_log_size);
}

__device__ __forceinline__ unsigned stwo_trace_value(
    const unsigned *const *trace_cols, const unsigned *interaction_offsets, unsigned row_count,
    unsigned interaction, unsigned column, unsigned row_index, int offset
) {
    unsigned target_row;
    if (offset == 0) {
        target_row = row_index;
    } else {
        unsigned eval_log_size = 0u;
        unsigned tmp = row_count;
        while (tmp > 1u) { tmp >>= 1u; eval_log_size++; }
        unsigned domain_log_size = eval_log_size - 1u;
        target_row = stwo_offset_bit_reversed_circle_domain_index(
            row_index, domain_log_size, eval_log_size, offset);
    }
    unsigned global_column = interaction_offsets[interaction] + column;
    return trace_cols[global_column][target_row];
}

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_6a86368213950a80(
    const unsigned *const *trace_cols,
    const unsigned *interaction_offsets,
    const unsigned *base_params,
    const unsigned *ext_params,
    const unsigned *random_coeff_powers,
    const unsigned *denom_inv,
    unsigned *coord_0,
    unsigned *coord_1,
    unsigned *coord_2,
    unsigned *coord_3,
    unsigned row_count,
    unsigned log_n_rows,
    unsigned rc_base
) {
    unsigned row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= row_count) { return; }

    // Base instructions.
    unsigned b0 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 0u, row_index, 0);
    unsigned b1 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 1u, row_index, 0);
    unsigned b2 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 2u, row_index, 0);
    unsigned b3 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 8u, row_index, 0);
    unsigned b4 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 10u, row_index, 0);
    unsigned b5 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 72u, row_index, 0);
    unsigned b6 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 73u, row_index, 0);
    unsigned b7 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 74u, row_index, 0);
    unsigned b8 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 75u, row_index, 0);
    unsigned b9 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 76u, row_index, 0);
    unsigned b10 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 77u, row_index, 0);
    unsigned b11 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 78u, row_index, 0);
    unsigned b12 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 79u, row_index, 0);
    unsigned b13 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 80u, row_index, 0);
    unsigned b14 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 81u, row_index, 0);
    unsigned b15 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 82u, row_index, 0);
    unsigned b16 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 83u, row_index, 0);
    unsigned b17 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 84u, row_index, 0);
    unsigned b18 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 85u, row_index, 0);
    unsigned b19 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 86u, row_index, 0);
    unsigned b20 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 87u, row_index, 0);
    unsigned b21 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 88u, row_index, 0);
    unsigned b22 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 89u, row_index, 0);
    unsigned b23 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 90u, row_index, 0);
    unsigned b24 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 91u, row_index, 0);
    unsigned b25 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 92u, row_index, 0);
    unsigned b26 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 93u, row_index, 0);
    unsigned b27 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 94u, row_index, 0);
    unsigned b28 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 95u, row_index, 0);
    unsigned b29 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 96u, row_index, 0);
    unsigned b30 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 97u, row_index, 0);
    unsigned b31 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 98u, row_index, 0);
    unsigned b32 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 99u, row_index, 0);
    unsigned b33 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 100u, row_index, 0);
    unsigned b34 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 101u, row_index, 0);
    unsigned b35 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 102u, row_index, 0);
    unsigned b36 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 103u, row_index, 0);
    unsigned b37 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 104u, row_index, 0);
    unsigned b38 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 105u, row_index, 0);
    unsigned b39 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 106u, row_index, 0);
    unsigned b40 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 107u, row_index, 0);
    unsigned b41 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 108u, row_index, 0);
    unsigned b42 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 109u, row_index, 0);
    unsigned b43 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 110u, row_index, 0);
    unsigned b44 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 111u, row_index, 0);
    unsigned b45 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 112u, row_index, 0);
    unsigned b46 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 113u, row_index, 0);
    unsigned b47 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 114u, row_index, 0);
    unsigned b48 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 115u, row_index, 0);
    unsigned b49 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 116u, row_index, 0);
    unsigned b50 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 117u, row_index, 0);
    unsigned b51 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 118u, row_index, 0);
    unsigned b52 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 119u, row_index, 0);
    unsigned b53 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 120u, row_index, 0);
    unsigned b54 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 121u, row_index, 0);
    unsigned b55 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 122u, row_index, 0);
    unsigned b56 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 123u, row_index, 0);
    unsigned b57 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 124u, row_index, 0);
    unsigned b58 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 125u, row_index, 0);
    unsigned b59 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 126u, row_index, 0);
    unsigned b60 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 127u, row_index, 0);
    unsigned b61 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 128u, row_index, 0);
    unsigned b62 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 129u, row_index, 0);
    unsigned b63 = 0u;
    unsigned b64 = 524288u;
    unsigned b65 = stwo_m31_add(b34, b64);
    b64 = 524288u;
    b34 = stwo_m31_add(b35, b64);
    b64 = 524288u;
    b35 = stwo_m31_add(b36, b64);
    b64 = 524288u;
    b36 = stwo_m31_add(b37, b64);
    b64 = 524288u;
    b37 = stwo_m31_add(b38, b64);
    b64 = 524288u;
    b38 = stwo_m31_add(b39, b64);
    b64 = 524288u;
    b39 = stwo_m31_add(b40, b64);
    b64 = 524288u;
    b40 = stwo_m31_add(b41, b64);
    b64 = 524288u;
    b41 = stwo_m31_add(b42, b64);
    b64 = 524288u;
    b42 = stwo_m31_add(b43, b64);
    b64 = 524288u;
    b43 = stwo_m31_add(b44, b64);
    b64 = 524288u;
    b44 = stwo_m31_add(b45, b64);
    b64 = 524288u;
    b45 = stwo_m31_add(b46, b64);
    b64 = 524288u;
    b46 = stwo_m31_add(b47, b64);
    b64 = 524288u;
    b47 = stwo_m31_add(b48, b64);
    b64 = 524288u;
    b48 = stwo_m31_add(b49, b64);
    b64 = 524288u;
    b49 = stwo_m31_add(b50, b64);
    b64 = 524288u;
    b50 = stwo_m31_add(b51, b64);
    b64 = 524288u;
    b51 = stwo_m31_add(b52, b64);
    b64 = 524288u;
    b52 = stwo_m31_add(b53, b64);
    b64 = 524288u;
    b53 = stwo_m31_add(b54, b64);
    b64 = 524288u;
    b54 = stwo_m31_add(b55, b64);
    b64 = 524288u;
    b55 = stwo_m31_add(b56, b64);
    b64 = 524288u;
    b56 = stwo_m31_add(b57, b64);
    b64 = 524288u;
    b57 = stwo_m31_add(b58, b64);
    b64 = 524288u;
    b58 = stwo_m31_add(b59, b64);
    b64 = 524288u;
    b59 = stwo_m31_add(b60, b64);
    b64 = 524288u;
    b60 = stwo_m31_add(b61, b64);
    b64 = 1u;
    b61 = stwo_m31_add(b0, b64);
    b64 = stwo_m31_add(b61, b3);
    b61 = stwo_m31_add(b1, b4);
    b4 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 8u, row_index, 0);
    b3 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 9u, row_index, 0);
    unsigned b66 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 10u, row_index, 0);
    unsigned b67 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 11u, row_index, 0);
    unsigned b68 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 12u, row_index, 0);
    unsigned b69 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 13u, row_index, 0);
    unsigned b70 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 14u, row_index, 0);
    unsigned b71 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 15u, row_index, 0);
    unsigned b72 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 16u, row_index, 0);
    unsigned b73 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 17u, row_index, 0);
    unsigned b74 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 18u, row_index, 0);
    unsigned b75 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 19u, row_index, 0);
    unsigned b76 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 20u, row_index, 0);
    unsigned b77 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 21u, row_index, 0);
    unsigned b78 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 22u, row_index, 0);
    unsigned b79 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 23u, row_index, 0);
    unsigned b80 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 24u, row_index, 0);
    unsigned b81 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 25u, row_index, 0);
    unsigned b82 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 26u, row_index, 0);
    unsigned b83 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 27u, row_index, 0);
    unsigned b84 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 28u, row_index, 0);
    unsigned b85 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 29u, row_index, 0);
    unsigned b86 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 30u, row_index, 0);
    unsigned b87 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 31u, row_index, 0);
    unsigned b88 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 32u, row_index, 0);
    unsigned b89 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 33u, row_index, 0);
    unsigned b90 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 34u, row_index, 0);
    unsigned b91 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 35u, row_index, 0);
    unsigned b92 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 36u, row_index, 0);
    unsigned b93 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 37u, row_index, 0);
    unsigned b94 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 38u, row_index, 0);
    unsigned b95 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 39u, row_index, 0);
    unsigned b96 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 40u, row_index, 0);
    unsigned b97 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 41u, row_index, 0);
    unsigned b98 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 42u, row_index, 0);
    unsigned b99 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 43u, row_index, 0);
    unsigned b100 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 44u, row_index, 0);
    unsigned b101 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 45u, row_index, 0);
    unsigned b102 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 46u, row_index, 0);
    unsigned b103 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 47u, row_index, 0);
    unsigned b104 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 48u, row_index, 0);
    unsigned b105 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 49u, row_index, 0);
    unsigned b106 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 50u, row_index, 0);
    unsigned b107 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 51u, row_index, 0);
    unsigned b108 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 52u, row_index, 0);
    unsigned b109 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 53u, row_index, 0);
    unsigned b110 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 54u, row_index, 0);
    unsigned b111 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 55u, row_index, 0);
    unsigned b112 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 56u, row_index, 0);
    unsigned b113 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 57u, row_index, 0);
    unsigned b114 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 58u, row_index, 0);
    unsigned b115 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 59u, row_index, 0);
    unsigned b116 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 60u, row_index, 0);
    unsigned b117 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 61u, row_index, 0);
    unsigned b118 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 62u, row_index, 0);
    unsigned b119 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 63u, row_index, 0);
    unsigned b120 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 64u, row_index, 0);
    unsigned b121 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 65u, row_index, 0);
    unsigned b122 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 66u, row_index, 0);
    unsigned b123 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 67u, row_index, 0);
    unsigned b124 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 68u, row_index, 0);
    unsigned b125 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 69u, row_index, 0);
    unsigned b126 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 70u, row_index, 0);
    unsigned b127 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 71u, row_index, 0);
    unsigned b128 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 72u, row_index, -1);
    unsigned b129 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 72u, row_index, 0);
    unsigned b130 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 73u, row_index, -1);
    unsigned b131 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 73u, row_index, 0);
    unsigned b132 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 74u, row_index, -1);
    unsigned b133 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 74u, row_index, 0);
    unsigned b134 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 75u, row_index, -1);
    unsigned b135 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 75u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e1 = StwoCudaQm31{ b5, b63, b63, b63 };
    StwoCudaQm31 e2 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 9u);
    e0 = stwo_qm31_add(e1, e2);
    e1 = stwo_load_qm31(ext_params, 2u);
    e2 = StwoCudaQm31{ b6, b63, b63, b63 };
    StwoCudaQm31 e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 3u);
    e0 = StwoCudaQm31{ b7, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 4u);
    e2 = StwoCudaQm31{ b8, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 5u);
    e0 = StwoCudaQm31{ b9, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 6u);
    e2 = StwoCudaQm31{ b10, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 10u);
    e0 = StwoCudaQm31{ b11, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 11u);
    e2 = StwoCudaQm31{ b12, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 12u);
    e0 = StwoCudaQm31{ b13, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 13u);
    e2 = StwoCudaQm31{ b14, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 14u);
    e0 = StwoCudaQm31{ b15, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 15u);
    e2 = StwoCudaQm31{ b16, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 16u);
    e0 = StwoCudaQm31{ b17, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 17u);
    e2 = StwoCudaQm31{ b18, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 18u);
    e0 = StwoCudaQm31{ b19, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 19u);
    e2 = StwoCudaQm31{ b20, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 20u);
    e0 = StwoCudaQm31{ b21, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 21u);
    e2 = StwoCudaQm31{ b22, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 22u);
    e0 = StwoCudaQm31{ b23, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 23u);
    e2 = StwoCudaQm31{ b24, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 24u);
    e0 = StwoCudaQm31{ b25, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 25u);
    e2 = StwoCudaQm31{ b26, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 26u);
    e0 = StwoCudaQm31{ b27, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 27u);
    e2 = StwoCudaQm31{ b28, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 28u);
    e0 = StwoCudaQm31{ b29, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 29u);
    e2 = StwoCudaQm31{ b30, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 30u);
    e0 = StwoCudaQm31{ b31, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 31u);
    e2 = StwoCudaQm31{ b32, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 32u);
    e0 = StwoCudaQm31{ b33, b63, b63, b63 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 7u);
    e2 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b65, b63, b63, b63 };
    e3 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 33u);
    e1 = stwo_qm31_add(e0, e3);
    e0 = stwo_load_qm31(ext_params, 7u);
    e3 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b34, b63, b63, b63 };
    StwoCudaQm31 e4 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 34u);
    e0 = stwo_qm31_add(e1, e4);
    e1 = stwo_load_qm31(ext_params, 7u);
    e4 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b35, b63, b63, b63 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 35u);
    e1 = stwo_qm31_add(e0, e5);
    e0 = stwo_load_qm31(ext_params, 7u);
    e5 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b36, b63, b63, b63 };
    StwoCudaQm31 e6 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 36u);
    e0 = stwo_qm31_add(e1, e6);
    e1 = stwo_load_qm31(ext_params, 7u);
    e6 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b37, b63, b63, b63 };
    StwoCudaQm31 e7 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 37u);
    e1 = stwo_qm31_add(e0, e7);
    e0 = stwo_load_qm31(ext_params, 7u);
    e7 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b38, b63, b63, b63 };
    StwoCudaQm31 e8 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 38u);
    e0 = stwo_qm31_add(e1, e8);
    e1 = stwo_load_qm31(ext_params, 7u);
    e8 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b39, b63, b63, b63 };
    StwoCudaQm31 e9 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 39u);
    e1 = stwo_qm31_add(e0, e9);
    e0 = stwo_load_qm31(ext_params, 7u);
    e9 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b40, b63, b63, b63 };
    StwoCudaQm31 e10 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 40u);
    e0 = stwo_qm31_add(e1, e10);
    e1 = stwo_load_qm31(ext_params, 7u);
    e10 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b41, b63, b63, b63 };
    StwoCudaQm31 e11 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 33u);
    e1 = stwo_qm31_add(e0, e11);
    e0 = stwo_load_qm31(ext_params, 7u);
    e11 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b42, b63, b63, b63 };
    StwoCudaQm31 e12 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 34u);
    e0 = stwo_qm31_add(e1, e12);
    e1 = stwo_load_qm31(ext_params, 7u);
    e12 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b43, b63, b63, b63 };
    StwoCudaQm31 e13 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 35u);
    e1 = stwo_qm31_add(e0, e13);
    e0 = stwo_load_qm31(ext_params, 7u);
    e13 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b44, b63, b63, b63 };
    StwoCudaQm31 e14 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 36u);
    e0 = stwo_qm31_add(e1, e14);
    e1 = stwo_load_qm31(ext_params, 7u);
    e14 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b45, b63, b63, b63 };
    StwoCudaQm31 e15 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 37u);
    e1 = stwo_qm31_add(e0, e15);
    e0 = stwo_load_qm31(ext_params, 7u);
    e15 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b46, b63, b63, b63 };
    StwoCudaQm31 e16 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 38u);
    e0 = stwo_qm31_add(e1, e16);
    e1 = stwo_load_qm31(ext_params, 7u);
    e16 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b47, b63, b63, b63 };
    StwoCudaQm31 e17 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 39u);
    e1 = stwo_qm31_add(e0, e17);
    e0 = stwo_load_qm31(ext_params, 7u);
    e17 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b48, b63, b63, b63 };
    StwoCudaQm31 e18 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 40u);
    e0 = stwo_qm31_add(e1, e18);
    e1 = stwo_load_qm31(ext_params, 7u);
    e18 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b49, b63, b63, b63 };
    StwoCudaQm31 e19 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 33u);
    e1 = stwo_qm31_add(e0, e19);
    e0 = stwo_load_qm31(ext_params, 7u);
    e19 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b50, b63, b63, b63 };
    StwoCudaQm31 e20 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 34u);
    e0 = stwo_qm31_add(e1, e20);
    e1 = stwo_load_qm31(ext_params, 7u);
    e20 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b51, b63, b63, b63 };
    StwoCudaQm31 e21 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 35u);
    e1 = stwo_qm31_add(e0, e21);
    e0 = stwo_load_qm31(ext_params, 7u);
    e21 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b52, b63, b63, b63 };
    StwoCudaQm31 e22 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 36u);
    e0 = stwo_qm31_add(e1, e22);
    e1 = stwo_load_qm31(ext_params, 7u);
    e22 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b53, b63, b63, b63 };
    StwoCudaQm31 e23 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 37u);
    e1 = stwo_qm31_add(e0, e23);
    e0 = stwo_load_qm31(ext_params, 7u);
    e23 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b54, b63, b63, b63 };
    StwoCudaQm31 e24 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 38u);
    e0 = stwo_qm31_add(e1, e24);
    e1 = stwo_load_qm31(ext_params, 7u);
    e24 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b55, b63, b63, b63 };
    StwoCudaQm31 e25 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 39u);
    e1 = stwo_qm31_add(e0, e25);
    e0 = stwo_load_qm31(ext_params, 7u);
    e25 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b56, b63, b63, b63 };
    StwoCudaQm31 e26 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 40u);
    e0 = stwo_qm31_add(e1, e26);
    e1 = stwo_load_qm31(ext_params, 7u);
    e26 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b57, b63, b63, b63 };
    StwoCudaQm31 e27 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 33u);
    e1 = stwo_qm31_add(e0, e27);
    e0 = stwo_load_qm31(ext_params, 7u);
    e27 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b58, b63, b63, b63 };
    StwoCudaQm31 e28 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 34u);
    e0 = stwo_qm31_add(e1, e28);
    e1 = stwo_load_qm31(ext_params, 7u);
    e28 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b59, b63, b63, b63 };
    StwoCudaQm31 e29 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 35u);
    e1 = stwo_qm31_add(e0, e29);
    e0 = stwo_load_qm31(ext_params, 7u);
    e29 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e1 = StwoCudaQm31{ b60, b63, b63, b63 };
    StwoCudaQm31 e30 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 36u);
    e0 = stwo_qm31_add(e1, e30);
    e1 = stwo_load_qm31(ext_params, 7u);
    e30 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 0u);
    e0 = StwoCudaQm31{ b0, b63, b63, b63 };
    StwoCudaQm31 e31 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 41u);
    e1 = stwo_qm31_add(e0, e31);
    e0 = stwo_load_qm31(ext_params, 2u);
    e31 = StwoCudaQm31{ b1, b63, b63, b63 };
    StwoCudaQm31 e32 = stwo_qm31_mul(e0, e31);
    e31 = stwo_qm31_add(e1, e32);
    e32 = stwo_load_qm31(ext_params, 3u);
    e1 = StwoCudaQm31{ b2, b63, b63, b63 };
    e0 = stwo_qm31_mul(e32, e1);
    e1 = stwo_qm31_add(e31, e0);
    e0 = stwo_load_qm31(ext_params, 7u);
    e31 = stwo_qm31_sub(e1, e0);
    e0 = StwoCudaQm31{ b62, b63, b63, b63 };
    e1 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e0);
    e0 = stwo_load_qm31(ext_params, 0u);
    e32 = StwoCudaQm31{ b64, b63, b63, b63 };
    StwoCudaQm31 e33 = stwo_qm31_mul(e0, e32);
    e32 = stwo_load_qm31(ext_params, 41u);
    e0 = stwo_qm31_add(e32, e33);
    e32 = stwo_load_qm31(ext_params, 2u);
    e33 = StwoCudaQm31{ b61, b63, b63, b63 };
    StwoCudaQm31 e34 = stwo_qm31_mul(e32, e33);
    e33 = stwo_qm31_add(e0, e34);
    e34 = stwo_load_qm31(ext_params, 3u);
    e0 = StwoCudaQm31{ b2, b63, b63, b63 };
    e32 = stwo_qm31_mul(e34, e0);
    e0 = stwo_qm31_add(e33, e32);
    e32 = stwo_load_qm31(ext_params, 7u);
    e33 = stwo_qm31_sub(e0, e32);
    e32 = stwo_load_qm31(ext_params, 42u);
    e0 = stwo_qm31_mul(e3, e32);
    e32 = stwo_load_qm31(ext_params, 42u);
    e34 = stwo_qm31_mul(e2, e32);
    e32 = stwo_qm31_add(e0, e34);
    e34 = stwo_qm31_mul(e2, e3);
    e3 = stwo_load_qm31(ext_params, 42u);
    e2 = stwo_qm31_mul(e5, e3);
    e3 = stwo_load_qm31(ext_params, 42u);
    e0 = stwo_qm31_mul(e4, e3);
    e3 = stwo_qm31_add(e2, e0);
    e0 = stwo_qm31_mul(e4, e5);
    e5 = stwo_load_qm31(ext_params, 42u);
    e4 = stwo_qm31_mul(e7, e5);
    e5 = stwo_load_qm31(ext_params, 42u);
    e2 = stwo_qm31_mul(e6, e5);
    e5 = stwo_qm31_add(e4, e2);
    e2 = stwo_qm31_mul(e6, e7);
    e7 = stwo_load_qm31(ext_params, 42u);
    e6 = stwo_qm31_mul(e9, e7);
    e7 = stwo_load_qm31(ext_params, 42u);
    e4 = stwo_qm31_mul(e8, e7);
    e7 = stwo_qm31_add(e6, e4);
    e4 = stwo_qm31_mul(e8, e9);
    e9 = stwo_load_qm31(ext_params, 42u);
    e8 = stwo_qm31_mul(e11, e9);
    e9 = stwo_load_qm31(ext_params, 42u);
    e6 = stwo_qm31_mul(e10, e9);
    e9 = stwo_qm31_add(e8, e6);
    e6 = stwo_qm31_mul(e10, e11);
    e11 = stwo_load_qm31(ext_params, 42u);
    e10 = stwo_qm31_mul(e13, e11);
    e11 = stwo_load_qm31(ext_params, 42u);
    e8 = stwo_qm31_mul(e12, e11);
    e11 = stwo_qm31_add(e10, e8);
    e8 = stwo_qm31_mul(e12, e13);
    e13 = stwo_load_qm31(ext_params, 42u);
    e12 = stwo_qm31_mul(e15, e13);
    e13 = stwo_load_qm31(ext_params, 42u);
    e10 = stwo_qm31_mul(e14, e13);
    e13 = stwo_qm31_add(e12, e10);
    e10 = stwo_qm31_mul(e14, e15);
    e15 = stwo_load_qm31(ext_params, 42u);
    e14 = stwo_qm31_mul(e17, e15);
    e15 = stwo_load_qm31(ext_params, 42u);
    e12 = stwo_qm31_mul(e16, e15);
    e15 = stwo_qm31_add(e14, e12);
    e12 = stwo_qm31_mul(e16, e17);
    e17 = stwo_load_qm31(ext_params, 42u);
    e16 = stwo_qm31_mul(e19, e17);
    e17 = stwo_load_qm31(ext_params, 42u);
    e14 = stwo_qm31_mul(e18, e17);
    e17 = stwo_qm31_add(e16, e14);
    e14 = stwo_qm31_mul(e18, e19);
    e19 = stwo_load_qm31(ext_params, 42u);
    e18 = stwo_qm31_mul(e21, e19);
    e19 = stwo_load_qm31(ext_params, 42u);
    e16 = stwo_qm31_mul(e20, e19);
    e19 = stwo_qm31_add(e18, e16);
    e16 = stwo_qm31_mul(e20, e21);
    e21 = stwo_load_qm31(ext_params, 42u);
    e20 = stwo_qm31_mul(e23, e21);
    e21 = stwo_load_qm31(ext_params, 42u);
    e18 = stwo_qm31_mul(e22, e21);
    e21 = stwo_qm31_add(e20, e18);
    e18 = stwo_qm31_mul(e22, e23);
    e23 = stwo_load_qm31(ext_params, 42u);
    e22 = stwo_qm31_mul(e25, e23);
    e23 = stwo_load_qm31(ext_params, 42u);
    e20 = stwo_qm31_mul(e24, e23);
    e23 = stwo_qm31_add(e22, e20);
    e20 = stwo_qm31_mul(e24, e25);
    e25 = stwo_load_qm31(ext_params, 42u);
    e24 = stwo_qm31_mul(e27, e25);
    e25 = stwo_load_qm31(ext_params, 42u);
    e22 = stwo_qm31_mul(e26, e25);
    e25 = stwo_qm31_add(e24, e22);
    e22 = stwo_qm31_mul(e26, e27);
    e27 = stwo_load_qm31(ext_params, 42u);
    e26 = stwo_qm31_mul(e29, e27);
    e27 = stwo_load_qm31(ext_params, 42u);
    e24 = stwo_qm31_mul(e28, e27);
    e27 = stwo_qm31_add(e26, e24);
    e24 = stwo_qm31_mul(e28, e29);
    e29 = stwo_load_qm31(ext_params, 42u);
    e28 = stwo_qm31_mul(e31, e29);
    e29 = StwoCudaQm31{ b62, b63, b63, b63 };
    e26 = stwo_qm31_mul(e30, e29);
    e29 = stwo_qm31_add(e28, e26);
    e26 = stwo_qm31_mul(e30, e31);
    e31 = StwoCudaQm31{ b4, b3, b66, b67 };
    e30 = StwoCudaQm31{ b68, b69, b70, b71 };
    e28 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e28, e34);
    e28 = stwo_qm31_sub(e31, e32);
    e31 = StwoCudaQm31{ b72, b73, b74, b75 };
    e32 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e32, e0);
    e32 = stwo_qm31_sub(e30, e3);
    e30 = StwoCudaQm31{ b76, b77, b78, b79 };
    e3 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e3, e2);
    e3 = stwo_qm31_sub(e31, e5);
    e31 = StwoCudaQm31{ b80, b81, b82, b83 };
    e5 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e5, e4);
    e5 = stwo_qm31_sub(e30, e7);
    e30 = StwoCudaQm31{ b84, b85, b86, b87 };
    e7 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e7, e6);
    e7 = stwo_qm31_sub(e31, e9);
    e31 = StwoCudaQm31{ b88, b89, b90, b91 };
    e9 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e9, e8);
    e9 = stwo_qm31_sub(e30, e11);
    e30 = StwoCudaQm31{ b92, b93, b94, b95 };
    e11 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e11, e10);
    e11 = stwo_qm31_sub(e31, e13);
    e31 = StwoCudaQm31{ b96, b97, b98, b99 };
    e13 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e13, e12);
    e13 = stwo_qm31_sub(e30, e15);
    e30 = StwoCudaQm31{ b100, b101, b102, b103 };
    e15 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e15, e14);
    e15 = stwo_qm31_sub(e31, e17);
    e31 = StwoCudaQm31{ b104, b105, b106, b107 };
    e17 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e17, e16);
    e17 = stwo_qm31_sub(e30, e19);
    e30 = StwoCudaQm31{ b108, b109, b110, b111 };
    e19 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e19, e18);
    e19 = stwo_qm31_sub(e31, e21);
    e31 = StwoCudaQm31{ b112, b113, b114, b115 };
    e21 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e21, e20);
    e21 = stwo_qm31_sub(e30, e23);
    e30 = StwoCudaQm31{ b116, b117, b118, b119 };
    e23 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e23, e22);
    e23 = stwo_qm31_sub(e31, e25);
    e31 = StwoCudaQm31{ b120, b121, b122, b123 };
    e25 = stwo_qm31_sub(e31, e30);
    e30 = stwo_qm31_mul(e25, e24);
    e25 = stwo_qm31_sub(e30, e27);
    e30 = StwoCudaQm31{ b124, b125, b126, b127 };
    e27 = stwo_qm31_sub(e30, e31);
    e31 = stwo_qm31_mul(e27, e26);
    e27 = stwo_qm31_sub(e31, e29);
    e31 = StwoCudaQm31{ b128, b130, b132, b134 };
    e29 = StwoCudaQm31{ b129, b131, b133, b135 };
    e26 = stwo_qm31_sub(e29, e31);
    e29 = stwo_qm31_sub(e26, e30);
    e26 = stwo_load_qm31(ext_params, 43u);
    e30 = stwo_qm31_add(e29, e26);
    e26 = stwo_qm31_mul(e30, e33);
    e30 = stwo_qm31_sub(e26, e1);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e28, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e32, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 2u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e5, stwo_load_qm31(random_coeff_powers, rc_base + 3u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e7, stwo_load_qm31(random_coeff_powers, rc_base + 4u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e9, stwo_load_qm31(random_coeff_powers, rc_base + 5u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e11, stwo_load_qm31(random_coeff_powers, rc_base + 6u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e13, stwo_load_qm31(random_coeff_powers, rc_base + 7u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e15, stwo_load_qm31(random_coeff_powers, rc_base + 8u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e17, stwo_load_qm31(random_coeff_powers, rc_base + 9u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e19, stwo_load_qm31(random_coeff_powers, rc_base + 10u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e21, stwo_load_qm31(random_coeff_powers, rc_base + 11u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e23, stwo_load_qm31(random_coeff_powers, rc_base + 12u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e25, stwo_load_qm31(random_coeff_powers, rc_base + 13u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e27, stwo_load_qm31(random_coeff_powers, rc_base + 14u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e30, stwo_load_qm31(random_coeff_powers, rc_base + 15u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
