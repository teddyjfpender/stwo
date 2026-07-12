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

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_bc74e68e54ebf0f3(
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
    unsigned b37 = 1u;
    unsigned b38 = stwo_m31_sub(b37, b6);
    b37 = stwo_m31_mul(b6, b38);
    b38 = 0u;
    unsigned b39 = 1u;
    unsigned b40 = stwo_m31_sub(b39, b7);
    b39 = stwo_m31_mul(b7, b40);
    b40 = 1u;
    unsigned b41 = stwo_m31_sub(b40, b8);
    b40 = stwo_m31_mul(b8, b41);
    b41 = 1u;
    unsigned b42 = stwo_m31_sub(b41, b9);
    b41 = stwo_m31_mul(b9, b42);
    b42 = 1u;
    unsigned b43 = stwo_m31_sub(b42, b8);
    b42 = stwo_m31_sub(b43, b9);
    b43 = 1u;
    unsigned b44 = stwo_m31_sub(b43, b42);
    b43 = stwo_m31_mul(b42, b44);
    b44 = 1u;
    unsigned b45 = stwo_m31_sub(b44, b10);
    b44 = stwo_m31_mul(b10, b45);
    b45 = 8u;
    unsigned b46 = stwo_m31_mul(b6, b45);
    b45 = 16u;
    unsigned b47 = stwo_m31_mul(b7, b45);
    b45 = stwo_m31_add(b46, b47);
    b47 = 32u;
    b46 = stwo_m31_mul(b8, b47);
    b47 = stwo_m31_add(b45, b46);
    b46 = 64u;
    b45 = stwo_m31_mul(b9, b46);
    b46 = stwo_m31_add(b47, b45);
    b45 = 128u;
    b47 = stwo_m31_mul(b42, b45);
    b45 = stwo_m31_add(b46, b47);
    b47 = 32u;
    b46 = stwo_m31_mul(b10, b47);
    b47 = 1u;
    unsigned b48 = stwo_m31_add(b47, b46);
    b47 = 256u;
    b46 = stwo_m31_add(b48, b47);
    b47 = 32768u;
    b48 = stwo_m31_sub(b3, b47);
    b47 = 32768u;
    unsigned b49 = stwo_m31_sub(b4, b47);
    b47 = 32768u;
    unsigned b50 = stwo_m31_sub(b5, b47);
    b47 = 1u;
    unsigned b51 = stwo_m31_sub(b47, b50);
    b47 = stwo_m31_mul(b8, b51);
    b51 = stwo_m31_mul(b6, b2);
    unsigned b52 = 1u;
    unsigned b53 = stwo_m31_sub(b52, b6);
    b52 = stwo_m31_mul(b53, b1);
    b53 = stwo_m31_add(b51, b52);
    b52 = stwo_m31_sub(b11, b53);
    b53 = stwo_m31_mul(b7, b2);
    b51 = 1u;
    b6 = stwo_m31_sub(b51, b7);
    b51 = stwo_m31_mul(b6, b1);
    b6 = stwo_m31_add(b53, b51);
    b51 = stwo_m31_sub(b12, b6);
    b6 = stwo_m31_mul(b8, b0);
    b53 = stwo_m31_mul(b9, b2);
    b9 = stwo_m31_add(b6, b53);
    b53 = stwo_m31_mul(b42, b1);
    b42 = stwo_m31_add(b9, b53);
    b53 = stwo_m31_sub(b13, b42);
    b42 = stwo_m31_add(b11, b48);
    b48 = stwo_m31_add(b12, b49);
    b49 = stwo_m31_add(b13, b50);
    b50 = 262144u;
    b13 = stwo_m31_mul(b33, b50);
    b50 = stwo_m31_mul(b24, b29);
    b12 = stwo_m31_sub(b50, b15);
    b50 = stwo_m31_mul(b24, b30);
    b11 = stwo_m31_mul(b25, b29);
    b9 = stwo_m31_add(b50, b11);
    b11 = stwo_m31_sub(b9, b16);
    b9 = 512u;
    b50 = stwo_m31_mul(b11, b9);
    b9 = stwo_m31_add(b12, b50);
    b50 = stwo_m31_sub(b13, b9);
    b9 = 262144u;
    b13 = stwo_m31_mul(b34, b9);
    b9 = stwo_m31_mul(b24, b31);
    b12 = stwo_m31_mul(b25, b30);
    b11 = stwo_m31_add(b9, b12);
    b12 = stwo_m31_mul(b26, b29);
    b9 = stwo_m31_add(b11, b12);
    b12 = stwo_m31_sub(b9, b17);
    b9 = stwo_m31_add(b33, b12);
    b12 = stwo_m31_mul(b24, b32);
    b11 = stwo_m31_mul(b25, b31);
    b6 = stwo_m31_add(b12, b11);
    b11 = stwo_m31_mul(b26, b30);
    b12 = stwo_m31_add(b6, b11);
    b11 = stwo_m31_mul(b27, b29);
    b6 = stwo_m31_add(b12, b11);
    b11 = stwo_m31_sub(b6, b18);
    b6 = 512u;
    b12 = stwo_m31_mul(b11, b6);
    b6 = stwo_m31_add(b9, b12);
    b12 = stwo_m31_sub(b13, b6);
    b6 = 262144u;
    b13 = stwo_m31_mul(b35, b6);
    b6 = stwo_m31_mul(b25, b32);
    b9 = stwo_m31_mul(b26, b31);
    b11 = stwo_m31_add(b6, b9);
    b9 = stwo_m31_mul(b27, b30);
    b6 = stwo_m31_add(b11, b9);
    b9 = stwo_m31_sub(b6, b19);
    b6 = stwo_m31_add(b34, b9);
    b9 = stwo_m31_mul(b26, b32);
    b11 = stwo_m31_mul(b27, b31);
    b7 = stwo_m31_add(b9, b11);
    b11 = stwo_m31_sub(b7, b20);
    b7 = 512u;
    b9 = stwo_m31_mul(b11, b7);
    b7 = stwo_m31_add(b6, b9);
    b9 = stwo_m31_sub(b13, b7);
    b7 = stwo_m31_mul(b27, b32);
    b13 = stwo_m31_add(b35, b7);
    b7 = 512u;
    b6 = stwo_m31_mul(b22, b7);
    b7 = stwo_m31_sub(b13, b6);
    b6 = stwo_m31_sub(b7, b21);
    b7 = stwo_m31_mul(b36, b36);
    b13 = stwo_m31_sub(b7, b36);
    b7 = 1u;
    b11 = stwo_m31_add(b0, b7);
    b7 = stwo_m31_add(b11, b8);
    b11 = stwo_m31_add(b1, b10);
    b10 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 0u, row_index, 0);
    b8 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 1u, row_index, 0);
    unsigned b54 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 2u, row_index, 0);
    unsigned b55 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 3u, row_index, 0);
    unsigned b56 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 4u, row_index, 0);
    unsigned b57 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 5u, row_index, 0);
    unsigned b58 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 6u, row_index, 0);
    unsigned b59 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 7u, row_index, 0);
    unsigned b60 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 8u, row_index, 0);
    unsigned b61 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 9u, row_index, 0);
    unsigned b62 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 10u, row_index, 0);
    unsigned b63 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 11u, row_index, 0);
    unsigned b64 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 12u, row_index, 0);
    unsigned b65 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 13u, row_index, 0);
    unsigned b66 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 14u, row_index, 0);
    unsigned b67 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 15u, row_index, 0);
    unsigned b68 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 16u, row_index, 0);
    unsigned b69 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 17u, row_index, 0);
    unsigned b70 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 18u, row_index, 0);
    unsigned b71 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 19u, row_index, 0);
    unsigned b72 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 20u, row_index, -1);
    unsigned b73 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 20u, row_index, 0);
    unsigned b74 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 21u, row_index, -1);
    unsigned b75 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 21u, row_index, 0);
    unsigned b76 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 22u, row_index, -1);
    unsigned b77 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 22u, row_index, 0);
    unsigned b78 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 23u, row_index, -1);
    unsigned b79 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 23u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = StwoCudaQm31{ b37, b38, b38, b38 };
    StwoCudaQm31 e1 = StwoCudaQm31{ b39, b38, b38, b38 };
    StwoCudaQm31 e2 = StwoCudaQm31{ b40, b38, b38, b38 };
    StwoCudaQm31 e3 = StwoCudaQm31{ b41, b38, b38, b38 };
    StwoCudaQm31 e4 = StwoCudaQm31{ b43, b38, b38, b38 };
    StwoCudaQm31 e5 = StwoCudaQm31{ b44, b38, b38, b38 };
    StwoCudaQm31 e6 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e7 = StwoCudaQm31{ b0, b38, b38, b38 };
    StwoCudaQm31 e8 = stwo_qm31_mul(e6, e7);
    e7 = stwo_load_qm31(ext_params, 1u);
    e6 = stwo_qm31_add(e7, e8);
    e7 = stwo_load_qm31(ext_params, 2u);
    e8 = StwoCudaQm31{ b3, b38, b38, b38 };
    StwoCudaQm31 e9 = stwo_qm31_mul(e7, e8);
    e8 = stwo_qm31_add(e6, e9);
    e9 = stwo_load_qm31(ext_params, 3u);
    e6 = StwoCudaQm31{ b4, b38, b38, b38 };
    e7 = stwo_qm31_mul(e9, e6);
    e6 = stwo_qm31_add(e8, e7);
    e7 = stwo_load_qm31(ext_params, 4u);
    e8 = StwoCudaQm31{ b5, b38, b38, b38 };
    e9 = stwo_qm31_mul(e7, e8);
    e8 = stwo_qm31_add(e6, e9);
    e9 = stwo_load_qm31(ext_params, 5u);
    e6 = StwoCudaQm31{ b45, b38, b38, b38 };
    e7 = stwo_qm31_mul(e9, e6);
    e6 = stwo_qm31_add(e8, e7);
    e7 = stwo_load_qm31(ext_params, 6u);
    e8 = StwoCudaQm31{ b46, b38, b38, b38 };
    e9 = stwo_qm31_mul(e7, e8);
    e8 = stwo_qm31_add(e6, e9);
    e9 = stwo_load_qm31(ext_params, 7u);
    e6 = stwo_qm31_sub(e8, e9);
    e9 = StwoCudaQm31{ b47, b38, b38, b38 };
    e8 = StwoCudaQm31{ b52, b38, b38, b38 };
    e7 = StwoCudaQm31{ b51, b38, b38, b38 };
    StwoCudaQm31 e10 = StwoCudaQm31{ b53, b38, b38, b38 };
    StwoCudaQm31 e11 = stwo_load_qm31(ext_params, 8u);
    StwoCudaQm31 e12 = StwoCudaQm31{ b42, b38, b38, b38 };
    StwoCudaQm31 e13 = stwo_qm31_mul(e11, e12);
    e12 = stwo_load_qm31(ext_params, 9u);
    e11 = stwo_qm31_add(e12, e13);
    e12 = stwo_load_qm31(ext_params, 10u);
    e13 = StwoCudaQm31{ b14, b38, b38, b38 };
    StwoCudaQm31 e14 = stwo_qm31_mul(e12, e13);
    e13 = stwo_qm31_add(e11, e14);
    e14 = stwo_load_qm31(ext_params, 11u);
    e11 = stwo_qm31_sub(e13, e14);
    e14 = stwo_load_qm31(ext_params, 12u);
    e13 = StwoCudaQm31{ b14, b38, b38, b38 };
    e12 = stwo_qm31_mul(e14, e13);
    e13 = stwo_load_qm31(ext_params, 13u);
    e14 = stwo_qm31_add(e13, e12);
    e13 = stwo_load_qm31(ext_params, 14u);
    e12 = StwoCudaQm31{ b15, b38, b38, b38 };
    StwoCudaQm31 e15 = stwo_qm31_mul(e13, e12);
    e12 = stwo_qm31_add(e14, e15);
    e15 = stwo_load_qm31(ext_params, 15u);
    e14 = StwoCudaQm31{ b16, b38, b38, b38 };
    e13 = stwo_qm31_mul(e15, e14);
    e14 = stwo_qm31_add(e12, e13);
    e13 = stwo_load_qm31(ext_params, 16u);
    e12 = StwoCudaQm31{ b17, b38, b38, b38 };
    e15 = stwo_qm31_mul(e13, e12);
    e12 = stwo_qm31_add(e14, e15);
    e15 = stwo_load_qm31(ext_params, 17u);
    e14 = StwoCudaQm31{ b18, b38, b38, b38 };
    e13 = stwo_qm31_mul(e15, e14);
    e14 = stwo_qm31_add(e12, e13);
    e13 = stwo_load_qm31(ext_params, 18u);
    e12 = StwoCudaQm31{ b19, b38, b38, b38 };
    e15 = stwo_qm31_mul(e13, e12);
    e12 = stwo_qm31_add(e14, e15);
    e15 = stwo_load_qm31(ext_params, 19u);
    e14 = StwoCudaQm31{ b20, b38, b38, b38 };
    e13 = stwo_qm31_mul(e15, e14);
    e14 = stwo_qm31_add(e12, e13);
    e13 = stwo_load_qm31(ext_params, 20u);
    e12 = StwoCudaQm31{ b21, b38, b38, b38 };
    e15 = stwo_qm31_mul(e13, e12);
    e12 = stwo_qm31_add(e14, e15);
    e15 = stwo_load_qm31(ext_params, 21u);
    e14 = StwoCudaQm31{ b22, b38, b38, b38 };
    e13 = stwo_qm31_mul(e15, e14);
    e14 = stwo_qm31_add(e12, e13);
    e13 = stwo_load_qm31(ext_params, 22u);
    e12 = stwo_qm31_sub(e14, e13);
    e13 = stwo_load_qm31(ext_params, 23u);
    e14 = StwoCudaQm31{ b48, b38, b38, b38 };
    e15 = stwo_qm31_mul(e13, e14);
    e14 = stwo_load_qm31(ext_params, 24u);
    e13 = stwo_qm31_add(e14, e15);
    e14 = stwo_load_qm31(ext_params, 25u);
    e15 = StwoCudaQm31{ b23, b38, b38, b38 };
    StwoCudaQm31 e16 = stwo_qm31_mul(e14, e15);
    e15 = stwo_qm31_add(e13, e16);
    e16 = stwo_load_qm31(ext_params, 26u);
    e13 = stwo_qm31_sub(e15, e16);
    e16 = stwo_load_qm31(ext_params, 27u);
    e15 = StwoCudaQm31{ b23, b38, b38, b38 };
    e14 = stwo_qm31_mul(e16, e15);
    e15 = stwo_load_qm31(ext_params, 28u);
    e16 = stwo_qm31_add(e15, e14);
    e15 = stwo_load_qm31(ext_params, 29u);
    e14 = StwoCudaQm31{ b24, b38, b38, b38 };
    StwoCudaQm31 e17 = stwo_qm31_mul(e15, e14);
    e14 = stwo_qm31_add(e16, e17);
    e17 = stwo_load_qm31(ext_params, 30u);
    e16 = StwoCudaQm31{ b25, b38, b38, b38 };
    e15 = stwo_qm31_mul(e17, e16);
    e16 = stwo_qm31_add(e14, e15);
    e15 = stwo_load_qm31(ext_params, 31u);
    e14 = StwoCudaQm31{ b26, b38, b38, b38 };
    e17 = stwo_qm31_mul(e15, e14);
    e14 = stwo_qm31_add(e16, e17);
    e17 = stwo_load_qm31(ext_params, 32u);
    e16 = StwoCudaQm31{ b27, b38, b38, b38 };
    e15 = stwo_qm31_mul(e17, e16);
    e16 = stwo_qm31_add(e14, e15);
    e15 = stwo_load_qm31(ext_params, 33u);
    e14 = stwo_qm31_sub(e16, e15);
    e15 = stwo_load_qm31(ext_params, 34u);
    e16 = StwoCudaQm31{ b49, b38, b38, b38 };
    e17 = stwo_qm31_mul(e15, e16);
    e16 = stwo_load_qm31(ext_params, 35u);
    e15 = stwo_qm31_add(e16, e17);
    e16 = stwo_load_qm31(ext_params, 36u);
    e17 = StwoCudaQm31{ b28, b38, b38, b38 };
    StwoCudaQm31 e18 = stwo_qm31_mul(e16, e17);
    e17 = stwo_qm31_add(e15, e18);
    e18 = stwo_load_qm31(ext_params, 37u);
    e15 = stwo_qm31_sub(e17, e18);
    e18 = stwo_load_qm31(ext_params, 38u);
    e17 = StwoCudaQm31{ b28, b38, b38, b38 };
    e16 = stwo_qm31_mul(e18, e17);
    e17 = stwo_load_qm31(ext_params, 39u);
    e18 = stwo_qm31_add(e17, e16);
    e17 = stwo_load_qm31(ext_params, 40u);
    e16 = StwoCudaQm31{ b29, b38, b38, b38 };
    StwoCudaQm31 e19 = stwo_qm31_mul(e17, e16);
    e16 = stwo_qm31_add(e18, e19);
    e19 = stwo_load_qm31(ext_params, 41u);
    e18 = StwoCudaQm31{ b30, b38, b38, b38 };
    e17 = stwo_qm31_mul(e19, e18);
    e18 = stwo_qm31_add(e16, e17);
    e17 = stwo_load_qm31(ext_params, 42u);
    e16 = StwoCudaQm31{ b31, b38, b38, b38 };
    e19 = stwo_qm31_mul(e17, e16);
    e16 = stwo_qm31_add(e18, e19);
    e19 = stwo_load_qm31(ext_params, 43u);
    e18 = StwoCudaQm31{ b32, b38, b38, b38 };
    e17 = stwo_qm31_mul(e19, e18);
    e18 = stwo_qm31_add(e16, e17);
    e17 = stwo_load_qm31(ext_params, 44u);
    e16 = stwo_qm31_sub(e18, e17);
    e17 = stwo_load_qm31(ext_params, 45u);
    e18 = StwoCudaQm31{ b33, b38, b38, b38 };
    e19 = stwo_qm31_mul(e17, e18);
    e18 = stwo_load_qm31(ext_params, 46u);
    e17 = stwo_qm31_add(e18, e19);
    e18 = stwo_load_qm31(ext_params, 47u);
    e19 = stwo_qm31_sub(e17, e18);
    e18 = StwoCudaQm31{ b50, b38, b38, b38 };
    e17 = stwo_load_qm31(ext_params, 48u);
    StwoCudaQm31 e20 = StwoCudaQm31{ b34, b38, b38, b38 };
    StwoCudaQm31 e21 = stwo_qm31_mul(e17, e20);
    e20 = stwo_load_qm31(ext_params, 49u);
    e17 = stwo_qm31_add(e20, e21);
    e20 = stwo_load_qm31(ext_params, 50u);
    e21 = stwo_qm31_sub(e17, e20);
    e20 = StwoCudaQm31{ b12, b38, b38, b38 };
    e17 = stwo_load_qm31(ext_params, 51u);
    StwoCudaQm31 e22 = StwoCudaQm31{ b35, b38, b38, b38 };
    StwoCudaQm31 e23 = stwo_qm31_mul(e17, e22);
    e22 = stwo_load_qm31(ext_params, 52u);
    e17 = stwo_qm31_add(e22, e23);
    e22 = stwo_load_qm31(ext_params, 53u);
    e23 = stwo_qm31_sub(e17, e22);
    e22 = StwoCudaQm31{ b9, b38, b38, b38 };
    e17 = StwoCudaQm31{ b6, b38, b38, b38 };
    StwoCudaQm31 e24 = StwoCudaQm31{ b13, b38, b38, b38 };
    StwoCudaQm31 e25 = stwo_load_qm31(ext_params, 54u);
    StwoCudaQm31 e26 = StwoCudaQm31{ b0, b38, b38, b38 };
    StwoCudaQm31 e27 = stwo_qm31_mul(e25, e26);
    e26 = stwo_load_qm31(ext_params, 55u);
    e25 = stwo_qm31_add(e26, e27);
    e26 = stwo_load_qm31(ext_params, 56u);
    e27 = StwoCudaQm31{ b1, b38, b38, b38 };
    StwoCudaQm31 e28 = stwo_qm31_mul(e26, e27);
    e27 = stwo_qm31_add(e25, e28);
    e28 = stwo_load_qm31(ext_params, 57u);
    e25 = StwoCudaQm31{ b2, b38, b38, b38 };
    e26 = stwo_qm31_mul(e28, e25);
    e25 = stwo_qm31_add(e27, e26);
    e26 = stwo_load_qm31(ext_params, 58u);
    e27 = stwo_qm31_sub(e25, e26);
    e26 = StwoCudaQm31{ b36, b38, b38, b38 };
    e25 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e26);
    e26 = stwo_load_qm31(ext_params, 59u);
    e28 = StwoCudaQm31{ b7, b38, b38, b38 };
    StwoCudaQm31 e29 = stwo_qm31_mul(e26, e28);
    e28 = stwo_load_qm31(ext_params, 60u);
    e26 = stwo_qm31_add(e28, e29);
    e28 = stwo_load_qm31(ext_params, 61u);
    e29 = StwoCudaQm31{ b11, b38, b38, b38 };
    StwoCudaQm31 e30 = stwo_qm31_mul(e28, e29);
    e29 = stwo_qm31_add(e26, e30);
    e30 = stwo_load_qm31(ext_params, 62u);
    e26 = StwoCudaQm31{ b2, b38, b38, b38 };
    e28 = stwo_qm31_mul(e30, e26);
    e26 = stwo_qm31_add(e29, e28);
    e28 = stwo_load_qm31(ext_params, 63u);
    e29 = stwo_qm31_sub(e26, e28);
    e28 = stwo_load_qm31(ext_params, 64u);
    e26 = stwo_qm31_mul(e11, e28);
    e28 = stwo_load_qm31(ext_params, 65u);
    e30 = stwo_qm31_mul(e6, e28);
    e28 = stwo_qm31_add(e26, e30);
    e30 = stwo_qm31_mul(e6, e11);
    e11 = stwo_load_qm31(ext_params, 66u);
    e6 = stwo_qm31_mul(e13, e11);
    e11 = stwo_load_qm31(ext_params, 67u);
    e26 = stwo_qm31_mul(e12, e11);
    e11 = stwo_qm31_add(e6, e26);
    e26 = stwo_qm31_mul(e12, e13);
    e13 = stwo_load_qm31(ext_params, 68u);
    e12 = stwo_qm31_mul(e15, e13);
    e13 = stwo_load_qm31(ext_params, 69u);
    e6 = stwo_qm31_mul(e14, e13);
    e13 = stwo_qm31_add(e12, e6);
    e6 = stwo_qm31_mul(e14, e15);
    e15 = stwo_load_qm31(ext_params, 70u);
    e14 = stwo_qm31_mul(e19, e15);
    e15 = stwo_load_qm31(ext_params, 71u);
    e12 = stwo_qm31_mul(e16, e15);
    e15 = stwo_qm31_add(e14, e12);
    e12 = stwo_qm31_mul(e16, e19);
    e19 = stwo_load_qm31(ext_params, 72u);
    e16 = stwo_qm31_mul(e23, e19);
    e19 = stwo_load_qm31(ext_params, 73u);
    e14 = stwo_qm31_mul(e21, e19);
    e19 = stwo_qm31_add(e16, e14);
    e14 = stwo_qm31_mul(e21, e23);
    e23 = StwoCudaQm31{ b36, b38, b38, b38 };
    e21 = stwo_qm31_mul(e29, e23);
    e23 = stwo_qm31_mul(e27, e25);
    e25 = stwo_qm31_add(e21, e23);
    e23 = stwo_qm31_mul(e27, e29);
    e29 = StwoCudaQm31{ b10, b8, b54, b55 };
    e27 = stwo_qm31_mul(e29, e30);
    e30 = stwo_qm31_sub(e27, e28);
    e27 = StwoCudaQm31{ b56, b57, b58, b59 };
    e28 = stwo_qm31_sub(e27, e29);
    e29 = stwo_qm31_mul(e28, e26);
    e28 = stwo_qm31_sub(e29, e11);
    e29 = StwoCudaQm31{ b60, b61, b62, b63 };
    e11 = stwo_qm31_sub(e29, e27);
    e27 = stwo_qm31_mul(e11, e6);
    e11 = stwo_qm31_sub(e27, e13);
    e27 = StwoCudaQm31{ b64, b65, b66, b67 };
    e13 = stwo_qm31_sub(e27, e29);
    e29 = stwo_qm31_mul(e13, e12);
    e13 = stwo_qm31_sub(e29, e15);
    e29 = StwoCudaQm31{ b68, b69, b70, b71 };
    e15 = stwo_qm31_sub(e29, e27);
    e27 = stwo_qm31_mul(e15, e14);
    e15 = stwo_qm31_sub(e27, e19);
    e27 = StwoCudaQm31{ b72, b74, b76, b78 };
    e19 = StwoCudaQm31{ b73, b75, b77, b79 };
    e14 = stwo_qm31_sub(e19, e27);
    e19 = stwo_qm31_sub(e14, e29);
    e14 = stwo_load_qm31(ext_params, 74u);
    e29 = stwo_qm31_add(e19, e14);
    e14 = stwo_qm31_mul(e29, e23);
    e29 = stwo_qm31_sub(e14, e25);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e0, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e1, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e2, stwo_load_qm31(random_coeff_powers, rc_base + 2u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 3u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e4, stwo_load_qm31(random_coeff_powers, rc_base + 4u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e5, stwo_load_qm31(random_coeff_powers, rc_base + 5u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e9, stwo_load_qm31(random_coeff_powers, rc_base + 6u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e8, stwo_load_qm31(random_coeff_powers, rc_base + 7u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e7, stwo_load_qm31(random_coeff_powers, rc_base + 8u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e10, stwo_load_qm31(random_coeff_powers, rc_base + 9u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e18, stwo_load_qm31(random_coeff_powers, rc_base + 10u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e20, stwo_load_qm31(random_coeff_powers, rc_base + 11u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e22, stwo_load_qm31(random_coeff_powers, rc_base + 12u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e17, stwo_load_qm31(random_coeff_powers, rc_base + 13u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e24, stwo_load_qm31(random_coeff_powers, rc_base + 14u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e30, stwo_load_qm31(random_coeff_powers, rc_base + 15u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e28, stwo_load_qm31(random_coeff_powers, rc_base + 16u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e11, stwo_load_qm31(random_coeff_powers, rc_base + 17u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e13, stwo_load_qm31(random_coeff_powers, rc_base + 18u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e15, stwo_load_qm31(random_coeff_powers, rc_base + 19u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e29, stwo_load_qm31(random_coeff_powers, rc_base + 20u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
