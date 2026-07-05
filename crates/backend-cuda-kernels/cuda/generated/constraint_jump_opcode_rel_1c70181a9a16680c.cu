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

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_80d573ea5cb70558(
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
    unsigned b3 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 3u, row_index, 0);
    unsigned b4 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 4u, row_index, 0);
    unsigned b5 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 5u, row_index, 0);
    unsigned b6 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 6u, row_index, 0);
    unsigned b7 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 7u, row_index, 0);
    unsigned b8 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 8u, row_index, 0);
    unsigned b9 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 9u, row_index, 0);
    unsigned b10 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 10u, row_index, 0);
    unsigned b11 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 11u, row_index, 0);
    unsigned b12 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 12u, row_index, 0);
    unsigned b13 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 13u, row_index, 0);
    unsigned b14 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 14u, row_index, 0);
    unsigned b15 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 15u, row_index, 0);
    unsigned b16 = 1u;
    unsigned b17 = stwo_m31_sub(b16, b4);
    b16 = stwo_m31_mul(b4, b17);
    b17 = 0u;
    unsigned b18 = 1u;
    unsigned b19 = stwo_m31_sub(b18, b5);
    b18 = stwo_m31_mul(b5, b19);
    b19 = 64u;
    unsigned b20 = stwo_m31_mul(b4, b19);
    b19 = 24u;
    unsigned b21 = stwo_m31_add(b19, b20);
    b19 = 1u;
    b20 = stwo_m31_sub(b19, b4);
    b19 = 128u;
    unsigned b22 = stwo_m31_mul(b20, b19);
    b19 = stwo_m31_add(b21, b22);
    b22 = 32u;
    b21 = stwo_m31_mul(b5, b22);
    b22 = 4u;
    b20 = stwo_m31_add(b22, b21);
    b22 = 32768u;
    b21 = stwo_m31_sub(b3, b22);
    b22 = 1u;
    unsigned b23 = stwo_m31_sub(b22, b4);
    b22 = stwo_m31_mul(b4, b2);
    b4 = stwo_m31_mul(b23, b1);
    b23 = stwo_m31_add(b22, b4);
    b4 = stwo_m31_sub(b6, b23);
    b23 = stwo_m31_add(b6, b21);
    b21 = 1u;
    b6 = stwo_m31_sub(b8, b21);
    b21 = stwo_m31_mul(b8, b6);
    b6 = 1u;
    b22 = stwo_m31_sub(b9, b6);
    b6 = stwo_m31_mul(b9, b22);
    b22 = 1u;
    unsigned b24 = stwo_m31_sub(b8, b22);
    b22 = stwo_m31_mul(b9, b24);
    b24 = 508u;
    unsigned b25 = stwo_m31_mul(b9, b24);
    b24 = 511u;
    unsigned b26 = stwo_m31_mul(b9, b24);
    b24 = 136u;
    unsigned b27 = stwo_m31_mul(b8, b24);
    b24 = stwo_m31_sub(b27, b9);
    b27 = 256u;
    unsigned b28 = stwo_m31_mul(b8, b27);
    b27 = 1u;
    unsigned b29 = stwo_m31_sub(b27, b14);
    b27 = stwo_m31_mul(b14, b29);
    b29 = 1u;
    unsigned b30 = stwo_m31_mul(b27, b29);
    b29 = 2u;
    b27 = stwo_m31_mul(b14, b29);
    b29 = stwo_m31_sub(b13, b27);
    b27 = 1u;
    b14 = stwo_m31_sub(b27, b29);
    b27 = stwo_m31_mul(b29, b14);
    b14 = 1u;
    b29 = stwo_m31_mul(b27, b14);
    b14 = stwo_m31_add(b13, b25);
    b25 = 512u;
    b27 = stwo_m31_mul(b11, b25);
    b25 = stwo_m31_add(b10, b27);
    b27 = 262144u;
    unsigned b31 = stwo_m31_mul(b12, b27);
    b27 = stwo_m31_add(b25, b31);
    b31 = 134217728u;
    b25 = stwo_m31_mul(b13, b31);
    b31 = stwo_m31_add(b27, b25);
    b25 = stwo_m31_sub(b31, b8);
    b31 = 536870912u;
    b8 = stwo_m31_mul(b31, b9);
    b31 = stwo_m31_sub(b25, b8);
    b8 = stwo_m31_mul(b15, b15);
    b25 = stwo_m31_sub(b8, b15);
    b8 = stwo_m31_add(b0, b31);
    b31 = stwo_m31_add(b1, b5);
    b5 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 0u, row_index, 0);
    b9 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 1u, row_index, 0);
    b27 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 2u, row_index, 0);
    b13 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 3u, row_index, 0);
    unsigned b32 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 4u, row_index, 0);
    unsigned b33 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 5u, row_index, 0);
    unsigned b34 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 6u, row_index, 0);
    unsigned b35 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 7u, row_index, 0);
    unsigned b36 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 8u, row_index, -1);
    unsigned b37 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 8u, row_index, 0);
    unsigned b38 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 9u, row_index, -1);
    unsigned b39 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 9u, row_index, 0);
    unsigned b40 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 10u, row_index, -1);
    unsigned b41 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 10u, row_index, 0);
    unsigned b42 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 11u, row_index, -1);
    unsigned b43 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 11u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = StwoCudaQm31{ b16, b17, b17, b17 };
    StwoCudaQm31 e1 = StwoCudaQm31{ b18, b17, b17, b17 };
    StwoCudaQm31 e2 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e3 = StwoCudaQm31{ b0, b17, b17, b17 };
    StwoCudaQm31 e4 = stwo_qm31_mul(e2, e3);
    e3 = stwo_load_qm31(ext_params, 1u);
    e2 = stwo_qm31_add(e3, e4);
    e3 = stwo_load_qm31(ext_params, 2u);
    e4 = stwo_qm31_add(e2, e3);
    e3 = stwo_load_qm31(ext_params, 3u);
    e2 = stwo_qm31_add(e4, e3);
    e3 = stwo_load_qm31(ext_params, 4u);
    e4 = StwoCudaQm31{ b3, b17, b17, b17 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e3, e4);
    e4 = stwo_qm31_add(e2, e5);
    e5 = stwo_load_qm31(ext_params, 5u);
    e2 = StwoCudaQm31{ b19, b17, b17, b17 };
    e3 = stwo_qm31_mul(e5, e2);
    e2 = stwo_qm31_add(e4, e3);
    e3 = stwo_load_qm31(ext_params, 6u);
    e4 = StwoCudaQm31{ b20, b17, b17, b17 };
    e5 = stwo_qm31_mul(e3, e4);
    e4 = stwo_qm31_add(e2, e5);
    e5 = stwo_load_qm31(ext_params, 7u);
    e2 = stwo_qm31_sub(e4, e5);
    e5 = StwoCudaQm31{ b4, b17, b17, b17 };
    e4 = stwo_load_qm31(ext_params, 0u);
    e3 = StwoCudaQm31{ b23, b17, b17, b17 };
    StwoCudaQm31 e6 = stwo_qm31_mul(e4, e3);
    e3 = stwo_load_qm31(ext_params, 8u);
    e4 = stwo_qm31_add(e3, e6);
    e3 = stwo_load_qm31(ext_params, 9u);
    e6 = StwoCudaQm31{ b7, b17, b17, b17 };
    StwoCudaQm31 e7 = stwo_qm31_mul(e3, e6);
    e6 = stwo_qm31_add(e4, e7);
    e7 = stwo_load_qm31(ext_params, 7u);
    e4 = stwo_qm31_sub(e6, e7);
    e7 = StwoCudaQm31{ b21, b17, b17, b17 };
    e6 = StwoCudaQm31{ b6, b17, b17, b17 };
    e3 = StwoCudaQm31{ b22, b17, b17, b17 };
    StwoCudaQm31 e8 = StwoCudaQm31{ b30, b17, b17, b17 };
    StwoCudaQm31 e9 = StwoCudaQm31{ b29, b17, b17, b17 };
    StwoCudaQm31 e10 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e11 = StwoCudaQm31{ b7, b17, b17, b17 };
    StwoCudaQm31 e12 = stwo_qm31_mul(e10, e11);
    e11 = stwo_load_qm31(ext_params, 10u);
    e10 = stwo_qm31_add(e11, e12);
    e11 = stwo_load_qm31(ext_params, 9u);
    e12 = StwoCudaQm31{ b10, b17, b17, b17 };
    StwoCudaQm31 e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 11u);
    e10 = StwoCudaQm31{ b11, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 4u);
    e12 = StwoCudaQm31{ b12, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 5u);
    e10 = StwoCudaQm31{ b14, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 6u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 12u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 13u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 14u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 15u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 16u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 17u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 18u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 19u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 20u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 21u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 22u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 23u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 24u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 25u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 26u);
    e10 = StwoCudaQm31{ b26, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 27u);
    e12 = StwoCudaQm31{ b26, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e10, e13);
    e13 = stwo_load_qm31(ext_params, 28u);
    e10 = StwoCudaQm31{ b24, b17, b17, b17 };
    e11 = stwo_qm31_mul(e13, e10);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 29u);
    e12 = stwo_qm31_add(e10, e11);
    e11 = stwo_load_qm31(ext_params, 29u);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 29u);
    e12 = stwo_qm31_add(e10, e11);
    e11 = stwo_load_qm31(ext_params, 29u);
    e10 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 29u);
    e12 = stwo_qm31_add(e10, e11);
    e11 = stwo_load_qm31(ext_params, 30u);
    e10 = StwoCudaQm31{ b28, b17, b17, b17 };
    e13 = stwo_qm31_mul(e11, e10);
    e10 = stwo_qm31_add(e12, e13);
    e13 = stwo_load_qm31(ext_params, 7u);
    e12 = stwo_qm31_sub(e10, e13);
    e13 = StwoCudaQm31{ b25, b17, b17, b17 };
    e10 = stwo_load_qm31(ext_params, 0u);
    e11 = StwoCudaQm31{ b0, b17, b17, b17 };
    StwoCudaQm31 e14 = stwo_qm31_mul(e10, e11);
    e11 = stwo_load_qm31(ext_params, 31u);
    e10 = stwo_qm31_add(e11, e14);
    e11 = stwo_load_qm31(ext_params, 9u);
    e14 = StwoCudaQm31{ b1, b17, b17, b17 };
    StwoCudaQm31 e15 = stwo_qm31_mul(e11, e14);
    e14 = stwo_qm31_add(e10, e15);
    e15 = stwo_load_qm31(ext_params, 11u);
    e10 = StwoCudaQm31{ b2, b17, b17, b17 };
    e11 = stwo_qm31_mul(e15, e10);
    e10 = stwo_qm31_add(e14, e11);
    e11 = stwo_load_qm31(ext_params, 7u);
    e14 = stwo_qm31_sub(e10, e11);
    e11 = StwoCudaQm31{ b15, b17, b17, b17 };
    e10 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e11);
    e11 = stwo_load_qm31(ext_params, 0u);
    e15 = StwoCudaQm31{ b8, b17, b17, b17 };
    StwoCudaQm31 e16 = stwo_qm31_mul(e11, e15);
    e15 = stwo_load_qm31(ext_params, 31u);
    e11 = stwo_qm31_add(e15, e16);
    e15 = stwo_load_qm31(ext_params, 9u);
    e16 = StwoCudaQm31{ b31, b17, b17, b17 };
    StwoCudaQm31 e17 = stwo_qm31_mul(e15, e16);
    e16 = stwo_qm31_add(e11, e17);
    e17 = stwo_load_qm31(ext_params, 11u);
    e11 = StwoCudaQm31{ b2, b17, b17, b17 };
    e15 = stwo_qm31_mul(e17, e11);
    e11 = stwo_qm31_add(e16, e15);
    e15 = stwo_load_qm31(ext_params, 7u);
    e16 = stwo_qm31_sub(e11, e15);
    e15 = stwo_load_qm31(ext_params, 32u);
    e11 = stwo_qm31_mul(e4, e15);
    e15 = stwo_load_qm31(ext_params, 32u);
    e17 = stwo_qm31_mul(e2, e15);
    e15 = stwo_qm31_add(e11, e17);
    e17 = stwo_qm31_mul(e2, e4);
    e4 = stwo_load_qm31(ext_params, 32u);
    e2 = stwo_qm31_mul(e14, e4);
    e4 = StwoCudaQm31{ b15, b17, b17, b17 };
    e11 = stwo_qm31_mul(e12, e4);
    e4 = stwo_qm31_add(e2, e11);
    e11 = stwo_qm31_mul(e12, e14);
    e14 = StwoCudaQm31{ b5, b9, b27, b13 };
    e12 = stwo_qm31_mul(e14, e17);
    e17 = stwo_qm31_sub(e12, e15);
    e12 = StwoCudaQm31{ b32, b33, b34, b35 };
    e15 = stwo_qm31_sub(e12, e14);
    e14 = stwo_qm31_mul(e15, e11);
    e15 = stwo_qm31_sub(e14, e4);
    e14 = StwoCudaQm31{ b36, b38, b40, b42 };
    e4 = StwoCudaQm31{ b37, b39, b41, b43 };
    e11 = stwo_qm31_sub(e4, e14);
    e4 = stwo_qm31_sub(e11, e12);
    e11 = stwo_load_qm31(ext_params, 33u);
    e12 = stwo_qm31_add(e4, e11);
    e11 = stwo_qm31_mul(e12, e16);
    e12 = stwo_qm31_sub(e11, e10);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e0, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e1, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e5, stwo_load_qm31(random_coeff_powers, rc_base + 2u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e7, stwo_load_qm31(random_coeff_powers, rc_base + 3u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e6, stwo_load_qm31(random_coeff_powers, rc_base + 4u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 5u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e8, stwo_load_qm31(random_coeff_powers, rc_base + 6u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e9, stwo_load_qm31(random_coeff_powers, rc_base + 7u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e13, stwo_load_qm31(random_coeff_powers, rc_base + 8u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e17, stwo_load_qm31(random_coeff_powers, rc_base + 9u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e15, stwo_load_qm31(random_coeff_powers, rc_base + 10u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e12, stwo_load_qm31(random_coeff_powers, rc_base + 11u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
