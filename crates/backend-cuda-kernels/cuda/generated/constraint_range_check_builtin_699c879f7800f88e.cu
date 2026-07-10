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

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_af9e2263fe20c1af(
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
    unsigned b0 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 0u, 0u, row_index, 0);
    unsigned b1 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 0u, row_index, 0);
    unsigned b2 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 1u, row_index, 0);
    unsigned b3 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 2u, row_index, 0);
    unsigned b4 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 3u, row_index, 0);
    unsigned b5 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 4u, row_index, 0);
    unsigned b6 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 5u, row_index, 0);
    unsigned b7 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 6u, row_index, 0);
    unsigned b8 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 7u, row_index, 0);
    unsigned b9 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 8u, row_index, 0);
    unsigned b10 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 9u, row_index, 0);
    unsigned b11 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 10u, row_index, 0);
    unsigned b12 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 11u, row_index, 0);
    unsigned b13 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 12u, row_index, 0);
    unsigned b14 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 13u, row_index, 0);
    unsigned b15 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 14u, row_index, 0);
    unsigned b16 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 15u, row_index, 0);
    unsigned b17 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 16u, row_index, 0);
    unsigned b18 = 6431u;
    unsigned b19 = stwo_m31_add(b18, b0);
    b18 = 0u;
    b0 = 1u;
    unsigned b20 = stwo_m31_sub(b0, b17);
    b0 = stwo_m31_mul(b17, b20);
    b20 = 1u;
    unsigned b21 = stwo_m31_mul(b0, b20);
    b20 = 2u;
    b0 = stwo_m31_mul(b17, b20);
    b20 = stwo_m31_sub(b16, b0);
    b0 = 1u;
    b17 = stwo_m31_sub(b0, b20);
    b0 = stwo_m31_mul(b20, b17);
    b17 = 1u;
    b20 = stwo_m31_mul(b0, b17);
    b17 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 0u, row_index, -1);
    b0 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 0u, row_index, 0);
    unsigned b22 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 1u, row_index, -1);
    unsigned b23 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 1u, row_index, 0);
    unsigned b24 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 2u, row_index, -1);
    unsigned b25 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 2u, row_index, 0);
    unsigned b26 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 3u, row_index, -1);
    unsigned b27 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 3u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e1 = StwoCudaQm31{ b19, b18, b18, b18 };
    StwoCudaQm31 e2 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 1u);
    e0 = stwo_qm31_add(e1, e2);
    e1 = stwo_load_qm31(ext_params, 2u);
    e2 = StwoCudaQm31{ b1, b18, b18, b18 };
    StwoCudaQm31 e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 3u);
    e0 = stwo_qm31_sub(e2, e3);
    e3 = StwoCudaQm31{ b21, b18, b18, b18 };
    e2 = StwoCudaQm31{ b20, b18, b18, b18 };
    e1 = stwo_load_qm31(ext_params, 4u);
    StwoCudaQm31 e4 = StwoCudaQm31{ b1, b18, b18, b18 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_load_qm31(ext_params, 5u);
    e1 = stwo_qm31_add(e4, e5);
    e4 = stwo_load_qm31(ext_params, 6u);
    e5 = StwoCudaQm31{ b2, b18, b18, b18 };
    StwoCudaQm31 e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 7u);
    e1 = StwoCudaQm31{ b3, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 8u);
    e5 = StwoCudaQm31{ b4, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 9u);
    e1 = StwoCudaQm31{ b5, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 10u);
    e5 = StwoCudaQm31{ b6, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 11u);
    e1 = StwoCudaQm31{ b7, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 12u);
    e5 = StwoCudaQm31{ b8, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 13u);
    e1 = StwoCudaQm31{ b9, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 14u);
    e5 = StwoCudaQm31{ b10, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 15u);
    e1 = StwoCudaQm31{ b11, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 16u);
    e5 = StwoCudaQm31{ b12, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 17u);
    e1 = StwoCudaQm31{ b13, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 18u);
    e5 = StwoCudaQm31{ b14, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 19u);
    e1 = StwoCudaQm31{ b15, b18, b18, b18 };
    e4 = stwo_qm31_mul(e6, e1);
    e1 = stwo_qm31_add(e5, e4);
    e4 = stwo_load_qm31(ext_params, 20u);
    e5 = StwoCudaQm31{ b16, b18, b18, b18 };
    e6 = stwo_qm31_mul(e4, e5);
    e5 = stwo_qm31_add(e1, e6);
    e6 = stwo_load_qm31(ext_params, 21u);
    e1 = stwo_qm31_sub(e5, e6);
    e6 = stwo_load_qm31(ext_params, 22u);
    e5 = stwo_qm31_mul(e1, e6);
    e6 = stwo_load_qm31(ext_params, 23u);
    e4 = stwo_qm31_mul(e0, e6);
    e6 = stwo_qm31_add(e5, e4);
    e4 = stwo_qm31_mul(e0, e1);
    e1 = StwoCudaQm31{ b17, b22, b24, b26 };
    e0 = StwoCudaQm31{ b0, b23, b25, b27 };
    e5 = stwo_qm31_sub(e0, e1);
    e0 = stwo_load_qm31(ext_params, 24u);
    e1 = stwo_qm31_add(e5, e0);
    e0 = stwo_qm31_mul(e1, e4);
    e1 = stwo_qm31_sub(e0, e6);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e2, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e1, stwo_load_qm31(random_coeff_powers, rc_base + 2u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
