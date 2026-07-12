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

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_3aac0396ae78da3f(
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
    unsigned b25 = 1u;
    unsigned b26 = stwo_m31_sub(b25, b4);
    b25 = stwo_m31_mul(b4, b26);
    b26 = 0u;
    unsigned b27 = 64u;
    unsigned b28 = stwo_m31_mul(b4, b27);
    b27 = 1u;
    unsigned b29 = stwo_m31_sub(b27, b4);
    b27 = 128u;
    unsigned b30 = stwo_m31_mul(b29, b27);
    b27 = stwo_m31_add(b28, b30);
    b30 = 32768u;
    b28 = stwo_m31_sub(b3, b30);
    b30 = 1u;
    b29 = stwo_m31_sub(b30, b4);
    b30 = 1u;
    unsigned b31 = stwo_m31_sub(b30, b10);
    b30 = stwo_m31_mul(b10, b31);
    b31 = 1u;
    unsigned b32 = stwo_m31_mul(b30, b31);
    b31 = 2u;
    b30 = stwo_m31_mul(b10, b31);
    b31 = stwo_m31_sub(b9, b30);
    b30 = 1u;
    b10 = stwo_m31_sub(b30, b31);
    b30 = stwo_m31_mul(b31, b10);
    b10 = 1u;
    b31 = stwo_m31_mul(b30, b10);
    b10 = 512u;
    b30 = stwo_m31_mul(b7, b10);
    b10 = stwo_m31_add(b6, b30);
    b30 = 262144u;
    unsigned b33 = stwo_m31_mul(b8, b30);
    b30 = stwo_m31_add(b10, b33);
    b33 = 134217728u;
    b10 = stwo_m31_mul(b9, b33);
    b33 = stwo_m31_add(b30, b10);
    b10 = stwo_m31_sub(b33, b2);
    b33 = 1u;
    b30 = stwo_m31_add(b1, b33);
    b33 = 1u;
    unsigned b34 = stwo_m31_sub(b33, b16);
    b33 = stwo_m31_mul(b16, b34);
    b34 = 1u;
    unsigned b35 = stwo_m31_mul(b33, b34);
    b34 = 2u;
    b33 = stwo_m31_mul(b16, b34);
    b34 = stwo_m31_sub(b15, b33);
    b33 = 1u;
    b16 = stwo_m31_sub(b33, b34);
    b33 = stwo_m31_mul(b34, b16);
    b16 = 1u;
    b34 = stwo_m31_mul(b33, b16);
    b16 = 512u;
    b33 = stwo_m31_mul(b13, b16);
    b16 = stwo_m31_add(b12, b33);
    b33 = 262144u;
    unsigned b36 = stwo_m31_mul(b14, b33);
    b33 = stwo_m31_add(b16, b36);
    b36 = 134217728u;
    b16 = stwo_m31_mul(b15, b36);
    b36 = stwo_m31_add(b33, b16);
    b16 = 1u;
    b33 = stwo_m31_add(b0, b16);
    b16 = stwo_m31_sub(b36, b33);
    b33 = stwo_m31_mul(b4, b2);
    b4 = stwo_m31_mul(b29, b1);
    b29 = stwo_m31_add(b33, b4);
    b4 = stwo_m31_sub(b17, b29);
    b29 = stwo_m31_add(b17, b28);
    b28 = 1u;
    b17 = stwo_m31_sub(b28, b23);
    b28 = stwo_m31_mul(b23, b17);
    b17 = 1u;
    b33 = stwo_m31_mul(b28, b17);
    b17 = 2u;
    b28 = stwo_m31_mul(b23, b17);
    b17 = stwo_m31_sub(b22, b28);
    b28 = 1u;
    b23 = stwo_m31_sub(b28, b17);
    b28 = stwo_m31_mul(b17, b23);
    b23 = 1u;
    b17 = stwo_m31_mul(b28, b23);
    b23 = stwo_m31_mul(b24, b24);
    b28 = stwo_m31_sub(b23, b24);
    b23 = 512u;
    b36 = stwo_m31_mul(b20, b23);
    b23 = stwo_m31_add(b19, b36);
    b36 = 262144u;
    unsigned b37 = stwo_m31_mul(b21, b36);
    b36 = stwo_m31_add(b23, b37);
    b37 = 134217728u;
    b23 = stwo_m31_mul(b22, b37);
    b37 = stwo_m31_add(b36, b23);
    b23 = 2u;
    b36 = stwo_m31_add(b1, b23);
    b23 = 2u;
    unsigned b38 = stwo_m31_add(b1, b23);
    b23 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 0u, row_index, 0);
    unsigned b39 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 1u, row_index, 0);
    unsigned b40 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 2u, row_index, 0);
    unsigned b41 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 3u, row_index, 0);
    unsigned b42 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 4u, row_index, 0);
    unsigned b43 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 5u, row_index, 0);
    unsigned b44 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 6u, row_index, 0);
    unsigned b45 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 7u, row_index, 0);
    unsigned b46 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 8u, row_index, 0);
    unsigned b47 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 9u, row_index, 0);
    unsigned b48 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 10u, row_index, 0);
    unsigned b49 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 11u, row_index, 0);
    unsigned b50 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 12u, row_index, 0);
    unsigned b51 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 13u, row_index, 0);
    unsigned b52 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 14u, row_index, 0);
    unsigned b53 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 15u, row_index, 0);
    unsigned b54 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 16u, row_index, -1);
    unsigned b55 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 16u, row_index, 0);
    unsigned b56 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 17u, row_index, -1);
    unsigned b57 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 17u, row_index, 0);
    unsigned b58 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 18u, row_index, -1);
    unsigned b59 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 18u, row_index, 0);
    unsigned b60 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 19u, row_index, -1);
    unsigned b61 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 19u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = StwoCudaQm31{ b25, b26, b26, b26 };
    StwoCudaQm31 e1 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e2 = StwoCudaQm31{ b0, b26, b26, b26 };
    StwoCudaQm31 e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_load_qm31(ext_params, 1u);
    e1 = stwo_qm31_add(e2, e3);
    e2 = stwo_load_qm31(ext_params, 2u);
    e3 = stwo_qm31_add(e1, e2);
    e2 = stwo_load_qm31(ext_params, 3u);
    e1 = stwo_qm31_add(e3, e2);
    e2 = stwo_load_qm31(ext_params, 4u);
    e3 = StwoCudaQm31{ b3, b26, b26, b26 };
    StwoCudaQm31 e4 = stwo_qm31_mul(e2, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 5u);
    e1 = StwoCudaQm31{ b27, b26, b26, b26 };
    e2 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e2);
    e2 = stwo_load_qm31(ext_params, 6u);
    e3 = stwo_qm31_add(e1, e2);
    e2 = stwo_load_qm31(ext_params, 7u);
    e1 = stwo_qm31_sub(e3, e2);
    e2 = stwo_load_qm31(ext_params, 8u);
    e3 = StwoCudaQm31{ b1, b26, b26, b26 };
    e4 = stwo_qm31_mul(e2, e3);
    e3 = stwo_load_qm31(ext_params, 9u);
    e2 = stwo_qm31_add(e3, e4);
    e3 = stwo_load_qm31(ext_params, 10u);
    e4 = StwoCudaQm31{ b5, b26, b26, b26 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e3, e4);
    e4 = stwo_qm31_add(e2, e5);
    e5 = stwo_load_qm31(ext_params, 11u);
    e2 = stwo_qm31_sub(e4, e5);
    e5 = StwoCudaQm31{ b32, b26, b26, b26 };
    e4 = StwoCudaQm31{ b31, b26, b26, b26 };
    e3 = stwo_load_qm31(ext_params, 12u);
    StwoCudaQm31 e6 = StwoCudaQm31{ b5, b26, b26, b26 };
    StwoCudaQm31 e7 = stwo_qm31_mul(e3, e6);
    e6 = stwo_load_qm31(ext_params, 13u);
    e3 = stwo_qm31_add(e6, e7);
    e6 = stwo_load_qm31(ext_params, 14u);
    e7 = StwoCudaQm31{ b6, b26, b26, b26 };
    StwoCudaQm31 e8 = stwo_qm31_mul(e6, e7);
    e7 = stwo_qm31_add(e3, e8);
    e8 = stwo_load_qm31(ext_params, 15u);
    e3 = StwoCudaQm31{ b7, b26, b26, b26 };
    e6 = stwo_qm31_mul(e8, e3);
    e3 = stwo_qm31_add(e7, e6);
    e6 = stwo_load_qm31(ext_params, 16u);
    e7 = StwoCudaQm31{ b8, b26, b26, b26 };
    e8 = stwo_qm31_mul(e6, e7);
    e7 = stwo_qm31_add(e3, e8);
    e8 = stwo_load_qm31(ext_params, 17u);
    e3 = StwoCudaQm31{ b9, b26, b26, b26 };
    e6 = stwo_qm31_mul(e8, e3);
    e3 = stwo_qm31_add(e7, e6);
    e6 = stwo_load_qm31(ext_params, 18u);
    e7 = stwo_qm31_sub(e3, e6);
    e6 = StwoCudaQm31{ b10, b26, b26, b26 };
    e3 = stwo_load_qm31(ext_params, 19u);
    e8 = StwoCudaQm31{ b30, b26, b26, b26 };
    StwoCudaQm31 e9 = stwo_qm31_mul(e3, e8);
    e8 = stwo_load_qm31(ext_params, 20u);
    e3 = stwo_qm31_add(e8, e9);
    e8 = stwo_load_qm31(ext_params, 21u);
    e9 = StwoCudaQm31{ b11, b26, b26, b26 };
    StwoCudaQm31 e10 = stwo_qm31_mul(e8, e9);
    e9 = stwo_qm31_add(e3, e10);
    e10 = stwo_load_qm31(ext_params, 22u);
    e3 = stwo_qm31_sub(e9, e10);
    e10 = StwoCudaQm31{ b35, b26, b26, b26 };
    e9 = StwoCudaQm31{ b34, b26, b26, b26 };
    e8 = stwo_load_qm31(ext_params, 23u);
    StwoCudaQm31 e11 = StwoCudaQm31{ b11, b26, b26, b26 };
    StwoCudaQm31 e12 = stwo_qm31_mul(e8, e11);
    e11 = stwo_load_qm31(ext_params, 24u);
    e8 = stwo_qm31_add(e11, e12);
    e11 = stwo_load_qm31(ext_params, 25u);
    e12 = StwoCudaQm31{ b12, b26, b26, b26 };
    StwoCudaQm31 e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e8, e13);
    e13 = stwo_load_qm31(ext_params, 26u);
    e8 = StwoCudaQm31{ b13, b26, b26, b26 };
    e11 = stwo_qm31_mul(e13, e8);
    e8 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 27u);
    e12 = StwoCudaQm31{ b14, b26, b26, b26 };
    e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_qm31_add(e8, e13);
    e13 = stwo_load_qm31(ext_params, 28u);
    e8 = StwoCudaQm31{ b15, b26, b26, b26 };
    e11 = stwo_qm31_mul(e13, e8);
    e8 = stwo_qm31_add(e12, e11);
    e11 = stwo_load_qm31(ext_params, 29u);
    e12 = stwo_qm31_sub(e8, e11);
    e11 = StwoCudaQm31{ b16, b26, b26, b26 };
    e8 = StwoCudaQm31{ b4, b26, b26, b26 };
    e13 = stwo_load_qm31(ext_params, 30u);
    StwoCudaQm31 e14 = StwoCudaQm31{ b29, b26, b26, b26 };
    StwoCudaQm31 e15 = stwo_qm31_mul(e13, e14);
    e14 = stwo_load_qm31(ext_params, 31u);
    e13 = stwo_qm31_add(e14, e15);
    e14 = stwo_load_qm31(ext_params, 32u);
    e15 = StwoCudaQm31{ b18, b26, b26, b26 };
    StwoCudaQm31 e16 = stwo_qm31_mul(e14, e15);
    e15 = stwo_qm31_add(e13, e16);
    e16 = stwo_load_qm31(ext_params, 33u);
    e13 = stwo_qm31_sub(e15, e16);
    e16 = StwoCudaQm31{ b33, b26, b26, b26 };
    e15 = StwoCudaQm31{ b17, b26, b26, b26 };
    e14 = stwo_load_qm31(ext_params, 34u);
    StwoCudaQm31 e17 = StwoCudaQm31{ b18, b26, b26, b26 };
    StwoCudaQm31 e18 = stwo_qm31_mul(e14, e17);
    e17 = stwo_load_qm31(ext_params, 35u);
    e14 = stwo_qm31_add(e17, e18);
    e17 = stwo_load_qm31(ext_params, 36u);
    e18 = StwoCudaQm31{ b19, b26, b26, b26 };
    StwoCudaQm31 e19 = stwo_qm31_mul(e17, e18);
    e18 = stwo_qm31_add(e14, e19);
    e19 = stwo_load_qm31(ext_params, 37u);
    e14 = StwoCudaQm31{ b20, b26, b26, b26 };
    e17 = stwo_qm31_mul(e19, e14);
    e14 = stwo_qm31_add(e18, e17);
    e17 = stwo_load_qm31(ext_params, 38u);
    e18 = StwoCudaQm31{ b21, b26, b26, b26 };
    e19 = stwo_qm31_mul(e17, e18);
    e18 = stwo_qm31_add(e14, e19);
    e19 = stwo_load_qm31(ext_params, 39u);
    e14 = StwoCudaQm31{ b22, b26, b26, b26 };
    e17 = stwo_qm31_mul(e19, e14);
    e14 = stwo_qm31_add(e18, e17);
    e17 = stwo_load_qm31(ext_params, 40u);
    e18 = stwo_qm31_sub(e14, e17);
    e17 = StwoCudaQm31{ b28, b26, b26, b26 };
    e14 = stwo_load_qm31(ext_params, 41u);
    e19 = StwoCudaQm31{ b0, b26, b26, b26 };
    StwoCudaQm31 e20 = stwo_qm31_mul(e14, e19);
    e19 = stwo_load_qm31(ext_params, 42u);
    e14 = stwo_qm31_add(e19, e20);
    e19 = stwo_load_qm31(ext_params, 43u);
    e20 = StwoCudaQm31{ b1, b26, b26, b26 };
    StwoCudaQm31 e21 = stwo_qm31_mul(e19, e20);
    e20 = stwo_qm31_add(e14, e21);
    e21 = stwo_load_qm31(ext_params, 44u);
    e14 = StwoCudaQm31{ b2, b26, b26, b26 };
    e19 = stwo_qm31_mul(e21, e14);
    e14 = stwo_qm31_add(e20, e19);
    e19 = stwo_load_qm31(ext_params, 45u);
    e20 = stwo_qm31_sub(e14, e19);
    e19 = StwoCudaQm31{ b24, b26, b26, b26 };
    e14 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e19);
    e19 = stwo_load_qm31(ext_params, 46u);
    e21 = StwoCudaQm31{ b37, b26, b26, b26 };
    StwoCudaQm31 e22 = stwo_qm31_mul(e19, e21);
    e21 = stwo_load_qm31(ext_params, 47u);
    e19 = stwo_qm31_add(e21, e22);
    e21 = stwo_load_qm31(ext_params, 48u);
    e22 = StwoCudaQm31{ b36, b26, b26, b26 };
    StwoCudaQm31 e23 = stwo_qm31_mul(e21, e22);
    e22 = stwo_qm31_add(e19, e23);
    e23 = stwo_load_qm31(ext_params, 49u);
    e19 = StwoCudaQm31{ b38, b26, b26, b26 };
    e21 = stwo_qm31_mul(e23, e19);
    e19 = stwo_qm31_add(e22, e21);
    e21 = stwo_load_qm31(ext_params, 50u);
    e22 = stwo_qm31_sub(e19, e21);
    e21 = stwo_load_qm31(ext_params, 51u);
    e19 = stwo_qm31_mul(e2, e21);
    e21 = stwo_load_qm31(ext_params, 52u);
    e23 = stwo_qm31_mul(e1, e21);
    e21 = stwo_qm31_add(e19, e23);
    e23 = stwo_qm31_mul(e1, e2);
    e2 = stwo_load_qm31(ext_params, 53u);
    e1 = stwo_qm31_mul(e3, e2);
    e2 = stwo_load_qm31(ext_params, 54u);
    e19 = stwo_qm31_mul(e7, e2);
    e2 = stwo_qm31_add(e1, e19);
    e19 = stwo_qm31_mul(e7, e3);
    e3 = stwo_load_qm31(ext_params, 55u);
    e7 = stwo_qm31_mul(e13, e3);
    e3 = stwo_load_qm31(ext_params, 56u);
    e1 = stwo_qm31_mul(e12, e3);
    e3 = stwo_qm31_add(e7, e1);
    e1 = stwo_qm31_mul(e12, e13);
    e13 = stwo_load_qm31(ext_params, 57u);
    e12 = stwo_qm31_mul(e20, e13);
    e13 = StwoCudaQm31{ b24, b26, b26, b26 };
    e7 = stwo_qm31_mul(e18, e13);
    e13 = stwo_qm31_add(e12, e7);
    e7 = stwo_qm31_mul(e18, e20);
    e20 = StwoCudaQm31{ b23, b39, b40, b41 };
    e18 = stwo_qm31_mul(e20, e23);
    e23 = stwo_qm31_sub(e18, e21);
    e18 = StwoCudaQm31{ b42, b43, b44, b45 };
    e21 = stwo_qm31_sub(e18, e20);
    e20 = stwo_qm31_mul(e21, e19);
    e21 = stwo_qm31_sub(e20, e2);
    e20 = StwoCudaQm31{ b46, b47, b48, b49 };
    e2 = stwo_qm31_sub(e20, e18);
    e18 = stwo_qm31_mul(e2, e1);
    e2 = stwo_qm31_sub(e18, e3);
    e18 = StwoCudaQm31{ b50, b51, b52, b53 };
    e3 = stwo_qm31_sub(e18, e20);
    e20 = stwo_qm31_mul(e3, e7);
    e3 = stwo_qm31_sub(e20, e13);
    e20 = StwoCudaQm31{ b54, b56, b58, b60 };
    e13 = StwoCudaQm31{ b55, b57, b59, b61 };
    e7 = stwo_qm31_sub(e13, e20);
    e13 = stwo_qm31_sub(e7, e18);
    e7 = stwo_load_qm31(ext_params, 58u);
    e18 = stwo_qm31_add(e13, e7);
    e7 = stwo_qm31_mul(e18, e22);
    e18 = stwo_qm31_sub(e7, e14);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e0, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e5, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e4, stwo_load_qm31(random_coeff_powers, rc_base + 2u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e6, stwo_load_qm31(random_coeff_powers, rc_base + 3u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e10, stwo_load_qm31(random_coeff_powers, rc_base + 4u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e9, stwo_load_qm31(random_coeff_powers, rc_base + 5u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e11, stwo_load_qm31(random_coeff_powers, rc_base + 6u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e8, stwo_load_qm31(random_coeff_powers, rc_base + 7u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e16, stwo_load_qm31(random_coeff_powers, rc_base + 8u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e15, stwo_load_qm31(random_coeff_powers, rc_base + 9u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e17, stwo_load_qm31(random_coeff_powers, rc_base + 10u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e23, stwo_load_qm31(random_coeff_powers, rc_base + 11u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e21, stwo_load_qm31(random_coeff_powers, rc_base + 12u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e2, stwo_load_qm31(random_coeff_powers, rc_base + 13u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 14u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e18, stwo_load_qm31(random_coeff_powers, rc_base + 15u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
