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
    unsigned log_n_rows, unsigned interaction, unsigned column, unsigned row_index, int offset
) {
    unsigned target_row;
    if (offset == 0) {
        target_row = row_index;
    } else {
        unsigned eval_log_size = 0u;
        unsigned tmp = row_count;
        while (tmp > 1u) { tmp >>= 1u; eval_log_size++; }
        target_row = stwo_offset_bit_reversed_circle_domain_index(
            row_index, log_n_rows, eval_log_size, offset);
    }
    unsigned global_column = interaction_offsets[interaction] + column;
    return trace_cols[global_column][target_row];
}

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_b3298f3d64db6945(
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
    unsigned b0 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 0u, row_index, 0);
    unsigned b1 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 1u, row_index, 0);
    unsigned b2 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 2u, row_index, 0);
    unsigned b3 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 3u, row_index, 0);
    unsigned b4 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 4u, row_index, 0);
    unsigned b5 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 5u, row_index, 0);
    unsigned b6 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 6u, row_index, 0);
    unsigned b7 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 7u, row_index, 0);
    unsigned b8 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 8u, row_index, 0);
    unsigned b9 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 9u, row_index, 0);
    unsigned b10 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 10u, row_index, 0);
    unsigned b11 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 11u, row_index, 0);
    unsigned b12 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 12u, row_index, 0);
    unsigned b13 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 13u, row_index, 0);
    unsigned b14 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 14u, row_index, 0);
    unsigned b15 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 15u, row_index, 0);
    unsigned b16 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 16u, row_index, 0);
    unsigned b17 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 17u, row_index, 0);
    unsigned b18 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 18u, row_index, 0);
    unsigned b19 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 19u, row_index, 0);
    unsigned b20 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 20u, row_index, 0);
    unsigned b21 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 21u, row_index, 0);
    unsigned b22 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 22u, row_index, 0);
    unsigned b23 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 23u, row_index, 0);
    unsigned b24 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 24u, row_index, 0);
    unsigned b25 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 25u, row_index, 0);
    unsigned b26 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 26u, row_index, 0);
    unsigned b27 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 27u, row_index, 0);
    unsigned b28 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 28u, row_index, 0);
    unsigned b29 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 29u, row_index, 0);
    unsigned b30 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 30u, row_index, 0);
    unsigned b31 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 31u, row_index, 0);
    unsigned b32 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 32u, row_index, 0);
    unsigned b33 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 33u, row_index, 0);
    unsigned b34 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 34u, row_index, 0);
    unsigned b35 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 35u, row_index, 0);
    unsigned b36 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 36u, row_index, 0);
    unsigned b37 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 37u, row_index, 0);
    unsigned b38 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 38u, row_index, 0);
    unsigned b39 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 39u, row_index, 0);
    unsigned b40 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 40u, row_index, 0);
    unsigned b41 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 41u, row_index, 0);
    unsigned b42 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 104u, row_index, 0);
    unsigned b43 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 105u, row_index, 0);
    unsigned b44 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 106u, row_index, 0);
    unsigned b45 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 107u, row_index, 0);
    unsigned b46 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 108u, row_index, 0);
    unsigned b47 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 109u, row_index, 0);
    unsigned b48 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 110u, row_index, 0);
    unsigned b49 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 111u, row_index, 0);
    unsigned b50 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 112u, row_index, 0);
    unsigned b51 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 113u, row_index, 0);
    unsigned b52 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 125u, row_index, 0);
    unsigned b53 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 126u, row_index, 0);
    unsigned b54 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 127u, row_index, 0);
    unsigned b55 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 128u, row_index, 0);
    unsigned b56 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 129u, row_index, 0);
    unsigned b57 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 130u, row_index, 0);
    unsigned b58 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 131u, row_index, 0);
    unsigned b59 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 132u, row_index, 0);
    unsigned b60 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 133u, row_index, 0);
    unsigned b61 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 134u, row_index, 0);
    unsigned b62 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 136u, row_index, 0);
    unsigned b63 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 137u, row_index, 0);
    unsigned b64 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 138u, row_index, 0);
    unsigned b65 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 139u, row_index, 0);
    unsigned b66 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 140u, row_index, 0);
    unsigned b67 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 141u, row_index, 0);
    unsigned b68 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 142u, row_index, 0);
    unsigned b69 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 143u, row_index, 0);
    unsigned b70 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 144u, row_index, 0);
    unsigned b71 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 145u, row_index, 0);
    unsigned b72 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 157u, row_index, 0);
    unsigned b73 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 158u, row_index, 0);
    unsigned b74 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 159u, row_index, 0);
    unsigned b75 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 160u, row_index, 0);
    unsigned b76 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 161u, row_index, 0);
    unsigned b77 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 162u, row_index, 0);
    unsigned b78 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 163u, row_index, 0);
    unsigned b79 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 164u, row_index, 0);
    unsigned b80 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 165u, row_index, 0);
    unsigned b81 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 166u, row_index, 0);
    unsigned b82 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 168u, row_index, 0);
    unsigned b83 = 0u;
    unsigned b84 = 1u;
    unsigned b85 = stwo_m31_add(b1, b84);
    b84 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 28u, row_index, 0);
    unsigned b86 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 29u, row_index, 0);
    unsigned b87 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 30u, row_index, 0);
    unsigned b88 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 31u, row_index, 0);
    unsigned b89 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 32u, row_index, -1);
    unsigned b90 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 32u, row_index, 0);
    unsigned b91 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 33u, row_index, -1);
    unsigned b92 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 33u, row_index, 0);
    unsigned b93 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 34u, row_index, -1);
    unsigned b94 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 34u, row_index, 0);
    unsigned b95 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 35u, row_index, -1);
    unsigned b96 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 35u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = stwo_load_qm31(ext_params, 183u);
    StwoCudaQm31 e1 = StwoCudaQm31{ b0, b83, b83, b83 };
    StwoCudaQm31 e2 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 184u);
    e0 = stwo_qm31_add(e1, e2);
    e1 = stwo_load_qm31(ext_params, 185u);
    e2 = StwoCudaQm31{ b1, b83, b83, b83 };
    StwoCudaQm31 e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 186u);
    e0 = StwoCudaQm31{ b2, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 187u);
    e2 = StwoCudaQm31{ b3, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 188u);
    e0 = StwoCudaQm31{ b4, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 189u);
    e2 = StwoCudaQm31{ b5, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 190u);
    e0 = StwoCudaQm31{ b6, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 191u);
    e2 = StwoCudaQm31{ b7, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 192u);
    e0 = StwoCudaQm31{ b8, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 193u);
    e2 = StwoCudaQm31{ b9, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 194u);
    e0 = StwoCudaQm31{ b10, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 195u);
    e2 = StwoCudaQm31{ b11, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 196u);
    e0 = StwoCudaQm31{ b12, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 197u);
    e2 = StwoCudaQm31{ b13, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 198u);
    e0 = StwoCudaQm31{ b14, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 199u);
    e2 = StwoCudaQm31{ b15, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 200u);
    e0 = StwoCudaQm31{ b16, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 201u);
    e2 = StwoCudaQm31{ b17, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 202u);
    e0 = StwoCudaQm31{ b18, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 203u);
    e2 = StwoCudaQm31{ b19, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 204u);
    e0 = StwoCudaQm31{ b20, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 205u);
    e2 = StwoCudaQm31{ b21, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 206u);
    e0 = StwoCudaQm31{ b22, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 207u);
    e2 = StwoCudaQm31{ b23, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 208u);
    e0 = StwoCudaQm31{ b24, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 209u);
    e2 = StwoCudaQm31{ b25, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 210u);
    e0 = StwoCudaQm31{ b26, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 211u);
    e2 = StwoCudaQm31{ b27, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 212u);
    e0 = StwoCudaQm31{ b28, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 213u);
    e2 = StwoCudaQm31{ b29, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 214u);
    e0 = StwoCudaQm31{ b30, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 215u);
    e2 = StwoCudaQm31{ b31, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 216u);
    e0 = StwoCudaQm31{ b32, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 217u);
    e2 = StwoCudaQm31{ b33, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 218u);
    e0 = StwoCudaQm31{ b34, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 219u);
    e2 = StwoCudaQm31{ b35, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 220u);
    e0 = StwoCudaQm31{ b36, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 221u);
    e2 = StwoCudaQm31{ b37, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 222u);
    e0 = StwoCudaQm31{ b38, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 223u);
    e2 = StwoCudaQm31{ b39, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 224u);
    e0 = StwoCudaQm31{ b40, b83, b83, b83 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 225u);
    e2 = StwoCudaQm31{ b41, b83, b83, b83 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 226u);
    e0 = stwo_qm31_sub(e2, e3);
    e3 = StwoCudaQm31{ b82, b83, b83, b83 };
    e2 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e3);
    e3 = stwo_load_qm31(ext_params, 227u);
    e1 = StwoCudaQm31{ b0, b83, b83, b83 };
    StwoCudaQm31 e4 = stwo_qm31_mul(e3, e1);
    e1 = stwo_load_qm31(ext_params, 228u);
    e3 = stwo_qm31_add(e1, e4);
    e1 = stwo_load_qm31(ext_params, 229u);
    e4 = StwoCudaQm31{ b85, b83, b83, b83 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 230u);
    e3 = StwoCudaQm31{ b42, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 231u);
    e4 = StwoCudaQm31{ b43, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 232u);
    e3 = StwoCudaQm31{ b44, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 233u);
    e4 = StwoCudaQm31{ b45, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 234u);
    e3 = StwoCudaQm31{ b46, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 235u);
    e4 = StwoCudaQm31{ b47, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 236u);
    e3 = StwoCudaQm31{ b48, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 237u);
    e4 = StwoCudaQm31{ b49, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 238u);
    e3 = StwoCudaQm31{ b50, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 239u);
    e4 = StwoCudaQm31{ b51, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 240u);
    e3 = StwoCudaQm31{ b52, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 241u);
    e4 = StwoCudaQm31{ b53, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 242u);
    e3 = StwoCudaQm31{ b54, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 243u);
    e4 = StwoCudaQm31{ b55, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 244u);
    e3 = StwoCudaQm31{ b56, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 245u);
    e4 = StwoCudaQm31{ b57, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 246u);
    e3 = StwoCudaQm31{ b58, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 247u);
    e4 = StwoCudaQm31{ b59, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 248u);
    e3 = StwoCudaQm31{ b60, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 249u);
    e4 = StwoCudaQm31{ b61, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 250u);
    e3 = StwoCudaQm31{ b62, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 251u);
    e4 = StwoCudaQm31{ b63, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 252u);
    e3 = StwoCudaQm31{ b64, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 253u);
    e4 = StwoCudaQm31{ b65, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 254u);
    e3 = StwoCudaQm31{ b66, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 255u);
    e4 = StwoCudaQm31{ b67, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 256u);
    e3 = StwoCudaQm31{ b68, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 257u);
    e4 = StwoCudaQm31{ b69, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 258u);
    e3 = StwoCudaQm31{ b70, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 259u);
    e4 = StwoCudaQm31{ b71, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 260u);
    e3 = StwoCudaQm31{ b72, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 261u);
    e4 = StwoCudaQm31{ b73, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 262u);
    e3 = StwoCudaQm31{ b74, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 263u);
    e4 = StwoCudaQm31{ b75, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 264u);
    e3 = StwoCudaQm31{ b76, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 265u);
    e4 = StwoCudaQm31{ b77, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 266u);
    e3 = StwoCudaQm31{ b78, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 267u);
    e4 = StwoCudaQm31{ b79, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 268u);
    e3 = StwoCudaQm31{ b80, b83, b83, b83 };
    e1 = stwo_qm31_mul(e5, e3);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 269u);
    e4 = StwoCudaQm31{ b81, b83, b83, b83 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e3, e5);
    e5 = stwo_load_qm31(ext_params, 270u);
    e3 = stwo_qm31_sub(e4, e5);
    e5 = StwoCudaQm31{ b82, b83, b83, b83 };
    e4 = stwo_qm31_mul(e3, e5);
    e5 = stwo_qm31_mul(e0, e2);
    e2 = stwo_qm31_add(e4, e5);
    e5 = stwo_qm31_mul(e0, e3);
    e3 = StwoCudaQm31{ b84, b86, b87, b88 };
    e0 = StwoCudaQm31{ b89, b91, b93, b95 };
    e4 = StwoCudaQm31{ b90, b92, b94, b96 };
    e1 = stwo_qm31_sub(e4, e0);
    e4 = stwo_qm31_sub(e1, e3);
    e1 = stwo_load_qm31(ext_params, 287u);
    e3 = stwo_qm31_add(e4, e1);
    e1 = stwo_qm31_mul(e3, e5);
    e3 = stwo_qm31_sub(e1, e2);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
