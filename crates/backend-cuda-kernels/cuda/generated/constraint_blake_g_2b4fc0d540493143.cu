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

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_a48e2af8f89f2334(
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
    unsigned b42 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 42u, row_index, 0);
    unsigned b43 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 43u, row_index, 0);
    unsigned b44 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 44u, row_index, 0);
    unsigned b45 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 45u, row_index, 0);
    unsigned b46 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 46u, row_index, 0);
    unsigned b47 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 47u, row_index, 0);
    unsigned b48 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 48u, row_index, 0);
    unsigned b49 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 49u, row_index, 0);
    unsigned b50 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 50u, row_index, 0);
    unsigned b51 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 51u, row_index, 0);
    unsigned b52 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 1u, 52u, row_index, 0);
    unsigned b53 = stwo_m31_add(b0, b2);
    unsigned b54 = stwo_m31_add(b53, b8);
    b53 = stwo_m31_sub(b54, b12);
    b54 = 32768u;
    unsigned b55 = stwo_m31_mul(b53, b54);
    b54 = 1u;
    b53 = stwo_m31_sub(b55, b54);
    b54 = stwo_m31_mul(b55, b53);
    b53 = 2u;
    unsigned b56 = stwo_m31_sub(b55, b53);
    b53 = stwo_m31_mul(b54, b56);
    b56 = 0u;
    b54 = stwo_m31_add(b1, b3);
    unsigned b57 = stwo_m31_add(b54, b9);
    b54 = stwo_m31_add(b57, b55);
    b57 = stwo_m31_sub(b54, b13);
    b54 = 32768u;
    b55 = stwo_m31_mul(b57, b54);
    b54 = 1u;
    b57 = stwo_m31_sub(b55, b54);
    b54 = stwo_m31_mul(b55, b57);
    b57 = 2u;
    unsigned b58 = stwo_m31_sub(b55, b57);
    b57 = stwo_m31_mul(b54, b58);
    b58 = 256u;
    b54 = stwo_m31_mul(b14, b58);
    b58 = stwo_m31_sub(b12, b54);
    b54 = 256u;
    b55 = stwo_m31_mul(b15, b54);
    b54 = stwo_m31_sub(b13, b55);
    b55 = 256u;
    unsigned b59 = stwo_m31_mul(b16, b55);
    b55 = stwo_m31_sub(b6, b59);
    b59 = 256u;
    unsigned b60 = stwo_m31_mul(b17, b59);
    b59 = stwo_m31_sub(b7, b60);
    b60 = 256u;
    unsigned b61 = stwo_m31_mul(b21, b60);
    b60 = stwo_m31_add(b20, b61);
    b61 = 256u;
    unsigned b62 = stwo_m31_mul(b19, b61);
    b61 = stwo_m31_add(b18, b62);
    b62 = stwo_m31_add(b4, b60);
    unsigned b63 = 0u;
    unsigned b64 = stwo_m31_add(b62, b63);
    b63 = stwo_m31_sub(b64, b22);
    b64 = 32768u;
    b62 = stwo_m31_mul(b63, b64);
    b64 = 1u;
    b63 = stwo_m31_sub(b62, b64);
    b64 = stwo_m31_mul(b62, b63);
    b63 = 2u;
    unsigned b65 = stwo_m31_sub(b62, b63);
    b63 = stwo_m31_mul(b64, b65);
    b65 = stwo_m31_add(b5, b61);
    b64 = 0u;
    unsigned b66 = stwo_m31_add(b65, b64);
    b64 = stwo_m31_add(b66, b62);
    b66 = stwo_m31_sub(b64, b23);
    b64 = 32768u;
    b62 = stwo_m31_mul(b66, b64);
    b64 = 1u;
    b66 = stwo_m31_sub(b62, b64);
    b64 = stwo_m31_mul(b62, b66);
    b66 = 2u;
    b65 = stwo_m31_sub(b62, b66);
    b66 = stwo_m31_mul(b64, b65);
    b65 = 4096u;
    b64 = stwo_m31_mul(b24, b65);
    b65 = stwo_m31_sub(b2, b64);
    b64 = 4096u;
    b62 = stwo_m31_mul(b25, b64);
    b64 = stwo_m31_sub(b3, b62);
    b62 = 4096u;
    unsigned b67 = stwo_m31_mul(b26, b62);
    b62 = stwo_m31_sub(b22, b67);
    b67 = 4096u;
    unsigned b68 = stwo_m31_mul(b27, b67);
    b67 = stwo_m31_sub(b23, b68);
    b68 = 16u;
    unsigned b69 = stwo_m31_mul(b30, b68);
    b68 = stwo_m31_add(b29, b69);
    b69 = 16u;
    unsigned b70 = stwo_m31_mul(b28, b69);
    b69 = stwo_m31_add(b31, b70);
    b70 = stwo_m31_add(b12, b68);
    b12 = stwo_m31_add(b70, b10);
    b70 = stwo_m31_sub(b12, b32);
    b12 = 32768u;
    unsigned b71 = stwo_m31_mul(b70, b12);
    b12 = 1u;
    b70 = stwo_m31_sub(b71, b12);
    b12 = stwo_m31_mul(b71, b70);
    b70 = 2u;
    unsigned b72 = stwo_m31_sub(b71, b70);
    b70 = stwo_m31_mul(b12, b72);
    b72 = stwo_m31_add(b13, b69);
    b13 = stwo_m31_add(b72, b11);
    b72 = stwo_m31_add(b13, b71);
    b13 = stwo_m31_sub(b72, b33);
    b72 = 32768u;
    b71 = stwo_m31_mul(b13, b72);
    b72 = 1u;
    b13 = stwo_m31_sub(b71, b72);
    b72 = stwo_m31_mul(b71, b13);
    b13 = 2u;
    b12 = stwo_m31_sub(b71, b13);
    b13 = stwo_m31_mul(b72, b12);
    b12 = 256u;
    b72 = stwo_m31_mul(b34, b12);
    b12 = stwo_m31_sub(b32, b72);
    b72 = 256u;
    b71 = stwo_m31_mul(b35, b72);
    b72 = stwo_m31_sub(b33, b71);
    b71 = 256u;
    unsigned b73 = stwo_m31_mul(b36, b71);
    b71 = stwo_m31_sub(b60, b73);
    b73 = 256u;
    b60 = stwo_m31_mul(b37, b73);
    b73 = stwo_m31_sub(b61, b60);
    b60 = 256u;
    b61 = stwo_m31_mul(b40, b60);
    b60 = stwo_m31_add(b39, b61);
    b61 = 256u;
    unsigned b74 = stwo_m31_mul(b38, b61);
    b61 = stwo_m31_add(b41, b74);
    b74 = stwo_m31_add(b22, b60);
    b22 = 0u;
    unsigned b75 = stwo_m31_add(b74, b22);
    b22 = stwo_m31_sub(b75, b42);
    b75 = 32768u;
    b74 = stwo_m31_mul(b22, b75);
    b75 = 1u;
    b22 = stwo_m31_sub(b74, b75);
    b75 = stwo_m31_mul(b74, b22);
    b22 = 2u;
    unsigned b76 = stwo_m31_sub(b74, b22);
    b22 = stwo_m31_mul(b75, b76);
    b76 = stwo_m31_add(b23, b61);
    b23 = 0u;
    b75 = stwo_m31_add(b76, b23);
    b23 = stwo_m31_add(b75, b74);
    b75 = stwo_m31_sub(b23, b43);
    b23 = 32768u;
    b74 = stwo_m31_mul(b75, b23);
    b23 = 1u;
    b75 = stwo_m31_sub(b74, b23);
    b23 = stwo_m31_mul(b74, b75);
    b75 = 2u;
    b76 = stwo_m31_sub(b74, b75);
    b75 = stwo_m31_mul(b23, b76);
    b76 = 128u;
    b23 = stwo_m31_mul(b44, b76);
    b76 = stwo_m31_sub(b68, b23);
    b23 = 128u;
    b68 = stwo_m31_mul(b45, b23);
    b23 = stwo_m31_sub(b69, b68);
    b68 = 128u;
    b69 = stwo_m31_mul(b46, b68);
    b68 = stwo_m31_sub(b42, b69);
    b69 = 128u;
    b74 = stwo_m31_mul(b47, b69);
    b69 = stwo_m31_sub(b43, b74);
    b74 = 512u;
    unsigned b77 = stwo_m31_mul(b50, b74);
    b74 = stwo_m31_add(b49, b77);
    b77 = 512u;
    unsigned b78 = stwo_m31_mul(b48, b77);
    b77 = stwo_m31_add(b51, b78);
    b78 = stwo_m31_mul(b52, b52);
    unsigned b79 = stwo_m31_sub(b78, b52);
    b78 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 0u, row_index, 0);
    unsigned b80 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 1u, row_index, 0);
    unsigned b81 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 2u, row_index, 0);
    unsigned b82 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 3u, row_index, 0);
    unsigned b83 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 4u, row_index, 0);
    unsigned b84 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 5u, row_index, 0);
    unsigned b85 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 6u, row_index, 0);
    unsigned b86 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 7u, row_index, 0);
    unsigned b87 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 8u, row_index, 0);
    unsigned b88 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 9u, row_index, 0);
    unsigned b89 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 10u, row_index, 0);
    unsigned b90 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 11u, row_index, 0);
    unsigned b91 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 12u, row_index, 0);
    unsigned b92 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 13u, row_index, 0);
    unsigned b93 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 14u, row_index, 0);
    unsigned b94 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 15u, row_index, 0);
    unsigned b95 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 16u, row_index, 0);
    unsigned b96 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 17u, row_index, 0);
    unsigned b97 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 18u, row_index, 0);
    unsigned b98 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 19u, row_index, 0);
    unsigned b99 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 20u, row_index, 0);
    unsigned b100 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 21u, row_index, 0);
    unsigned b101 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 22u, row_index, 0);
    unsigned b102 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 23u, row_index, 0);
    unsigned b103 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 24u, row_index, 0);
    unsigned b104 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 25u, row_index, 0);
    unsigned b105 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 26u, row_index, 0);
    unsigned b106 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 27u, row_index, 0);
    unsigned b107 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 28u, row_index, 0);
    unsigned b108 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 29u, row_index, 0);
    unsigned b109 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 30u, row_index, 0);
    unsigned b110 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 31u, row_index, 0);
    unsigned b111 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 32u, row_index, -1);
    unsigned b112 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 32u, row_index, 0);
    unsigned b113 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 33u, row_index, -1);
    unsigned b114 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 33u, row_index, 0);
    unsigned b115 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 34u, row_index, -1);
    unsigned b116 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 34u, row_index, 0);
    unsigned b117 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 35u, row_index, -1);
    unsigned b118 = stwo_trace_value(trace_cols, interaction_offsets, row_count, log_n_rows, 2u, 35u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = StwoCudaQm31{ b53, b56, b56, b56 };
    StwoCudaQm31 e1 = StwoCudaQm31{ b57, b56, b56, b56 };
    StwoCudaQm31 e2 = stwo_load_qm31(ext_params, 0u);
    StwoCudaQm31 e3 = StwoCudaQm31{ b58, b56, b56, b56 };
    StwoCudaQm31 e4 = stwo_qm31_mul(e2, e3);
    e3 = stwo_load_qm31(ext_params, 1u);
    e2 = stwo_qm31_add(e3, e4);
    e3 = stwo_load_qm31(ext_params, 2u);
    e4 = StwoCudaQm31{ b55, b56, b56, b56 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e3, e4);
    e4 = stwo_qm31_add(e2, e5);
    e5 = stwo_load_qm31(ext_params, 3u);
    e2 = StwoCudaQm31{ b18, b56, b56, b56 };
    e3 = stwo_qm31_mul(e5, e2);
    e2 = stwo_qm31_add(e4, e3);
    e3 = stwo_load_qm31(ext_params, 4u);
    e4 = stwo_qm31_sub(e2, e3);
    e3 = stwo_load_qm31(ext_params, 5u);
    e2 = StwoCudaQm31{ b14, b56, b56, b56 };
    e5 = stwo_qm31_mul(e3, e2);
    e2 = stwo_load_qm31(ext_params, 6u);
    e3 = stwo_qm31_add(e2, e5);
    e2 = stwo_load_qm31(ext_params, 7u);
    e5 = StwoCudaQm31{ b16, b56, b56, b56 };
    StwoCudaQm31 e6 = stwo_qm31_mul(e2, e5);
    e5 = stwo_qm31_add(e3, e6);
    e6 = stwo_load_qm31(ext_params, 8u);
    e3 = StwoCudaQm31{ b19, b56, b56, b56 };
    e2 = stwo_qm31_mul(e6, e3);
    e3 = stwo_qm31_add(e5, e2);
    e2 = stwo_load_qm31(ext_params, 9u);
    e5 = stwo_qm31_sub(e3, e2);
    e2 = stwo_load_qm31(ext_params, 10u);
    e3 = StwoCudaQm31{ b54, b56, b56, b56 };
    e6 = stwo_qm31_mul(e2, e3);
    e3 = stwo_load_qm31(ext_params, 11u);
    e2 = stwo_qm31_add(e3, e6);
    e3 = stwo_load_qm31(ext_params, 12u);
    e6 = StwoCudaQm31{ b59, b56, b56, b56 };
    StwoCudaQm31 e7 = stwo_qm31_mul(e3, e6);
    e6 = stwo_qm31_add(e2, e7);
    e7 = stwo_load_qm31(ext_params, 13u);
    e2 = StwoCudaQm31{ b20, b56, b56, b56 };
    e3 = stwo_qm31_mul(e7, e2);
    e2 = stwo_qm31_add(e6, e3);
    e3 = stwo_load_qm31(ext_params, 14u);
    e6 = stwo_qm31_sub(e2, e3);
    e3 = stwo_load_qm31(ext_params, 15u);
    e2 = StwoCudaQm31{ b15, b56, b56, b56 };
    e7 = stwo_qm31_mul(e3, e2);
    e2 = stwo_load_qm31(ext_params, 16u);
    e3 = stwo_qm31_add(e2, e7);
    e2 = stwo_load_qm31(ext_params, 17u);
    e7 = StwoCudaQm31{ b17, b56, b56, b56 };
    StwoCudaQm31 e8 = stwo_qm31_mul(e2, e7);
    e7 = stwo_qm31_add(e3, e8);
    e8 = stwo_load_qm31(ext_params, 18u);
    e3 = StwoCudaQm31{ b21, b56, b56, b56 };
    e2 = stwo_qm31_mul(e8, e3);
    e3 = stwo_qm31_add(e7, e2);
    e2 = stwo_load_qm31(ext_params, 19u);
    e7 = stwo_qm31_sub(e3, e2);
    e2 = StwoCudaQm31{ b63, b56, b56, b56 };
    e3 = StwoCudaQm31{ b66, b56, b56, b56 };
    e8 = stwo_load_qm31(ext_params, 20u);
    StwoCudaQm31 e9 = StwoCudaQm31{ b65, b56, b56, b56 };
    StwoCudaQm31 e10 = stwo_qm31_mul(e8, e9);
    e9 = stwo_load_qm31(ext_params, 21u);
    e8 = stwo_qm31_add(e9, e10);
    e9 = stwo_load_qm31(ext_params, 22u);
    e10 = StwoCudaQm31{ b62, b56, b56, b56 };
    StwoCudaQm31 e11 = stwo_qm31_mul(e9, e10);
    e10 = stwo_qm31_add(e8, e11);
    e11 = stwo_load_qm31(ext_params, 23u);
    e8 = StwoCudaQm31{ b28, b56, b56, b56 };
    e9 = stwo_qm31_mul(e11, e8);
    e8 = stwo_qm31_add(e10, e9);
    e9 = stwo_load_qm31(ext_params, 24u);
    e10 = stwo_qm31_sub(e8, e9);
    e9 = stwo_load_qm31(ext_params, 25u);
    e8 = StwoCudaQm31{ b24, b56, b56, b56 };
    e11 = stwo_qm31_mul(e9, e8);
    e8 = stwo_load_qm31(ext_params, 26u);
    e9 = stwo_qm31_add(e8, e11);
    e8 = stwo_load_qm31(ext_params, 27u);
    e11 = StwoCudaQm31{ b26, b56, b56, b56 };
    StwoCudaQm31 e12 = stwo_qm31_mul(e8, e11);
    e11 = stwo_qm31_add(e9, e12);
    e12 = stwo_load_qm31(ext_params, 28u);
    e9 = StwoCudaQm31{ b29, b56, b56, b56 };
    e8 = stwo_qm31_mul(e12, e9);
    e9 = stwo_qm31_add(e11, e8);
    e8 = stwo_load_qm31(ext_params, 29u);
    e11 = stwo_qm31_sub(e9, e8);
    e8 = stwo_load_qm31(ext_params, 30u);
    e9 = StwoCudaQm31{ b64, b56, b56, b56 };
    e12 = stwo_qm31_mul(e8, e9);
    e9 = stwo_load_qm31(ext_params, 31u);
    e8 = stwo_qm31_add(e9, e12);
    e9 = stwo_load_qm31(ext_params, 32u);
    e12 = StwoCudaQm31{ b67, b56, b56, b56 };
    StwoCudaQm31 e13 = stwo_qm31_mul(e9, e12);
    e12 = stwo_qm31_add(e8, e13);
    e13 = stwo_load_qm31(ext_params, 33u);
    e8 = StwoCudaQm31{ b30, b56, b56, b56 };
    e9 = stwo_qm31_mul(e13, e8);
    e8 = stwo_qm31_add(e12, e9);
    e9 = stwo_load_qm31(ext_params, 34u);
    e12 = stwo_qm31_sub(e8, e9);
    e9 = stwo_load_qm31(ext_params, 35u);
    e8 = StwoCudaQm31{ b25, b56, b56, b56 };
    e13 = stwo_qm31_mul(e9, e8);
    e8 = stwo_load_qm31(ext_params, 36u);
    e9 = stwo_qm31_add(e8, e13);
    e8 = stwo_load_qm31(ext_params, 37u);
    e13 = StwoCudaQm31{ b27, b56, b56, b56 };
    StwoCudaQm31 e14 = stwo_qm31_mul(e8, e13);
    e13 = stwo_qm31_add(e9, e14);
    e14 = stwo_load_qm31(ext_params, 38u);
    e9 = StwoCudaQm31{ b31, b56, b56, b56 };
    e8 = stwo_qm31_mul(e14, e9);
    e9 = stwo_qm31_add(e13, e8);
    e8 = stwo_load_qm31(ext_params, 39u);
    e13 = stwo_qm31_sub(e9, e8);
    e8 = StwoCudaQm31{ b70, b56, b56, b56 };
    e9 = StwoCudaQm31{ b13, b56, b56, b56 };
    e14 = stwo_load_qm31(ext_params, 40u);
    StwoCudaQm31 e15 = StwoCudaQm31{ b12, b56, b56, b56 };
    StwoCudaQm31 e16 = stwo_qm31_mul(e14, e15);
    e15 = stwo_load_qm31(ext_params, 41u);
    e14 = stwo_qm31_add(e15, e16);
    e15 = stwo_load_qm31(ext_params, 42u);
    e16 = StwoCudaQm31{ b71, b56, b56, b56 };
    StwoCudaQm31 e17 = stwo_qm31_mul(e15, e16);
    e16 = stwo_qm31_add(e14, e17);
    e17 = stwo_load_qm31(ext_params, 43u);
    e14 = StwoCudaQm31{ b38, b56, b56, b56 };
    e15 = stwo_qm31_mul(e17, e14);
    e14 = stwo_qm31_add(e16, e15);
    e15 = stwo_load_qm31(ext_params, 44u);
    e16 = stwo_qm31_sub(e14, e15);
    e15 = stwo_load_qm31(ext_params, 45u);
    e14 = StwoCudaQm31{ b34, b56, b56, b56 };
    e17 = stwo_qm31_mul(e15, e14);
    e14 = stwo_load_qm31(ext_params, 46u);
    e15 = stwo_qm31_add(e14, e17);
    e14 = stwo_load_qm31(ext_params, 47u);
    e17 = StwoCudaQm31{ b36, b56, b56, b56 };
    StwoCudaQm31 e18 = stwo_qm31_mul(e14, e17);
    e17 = stwo_qm31_add(e15, e18);
    e18 = stwo_load_qm31(ext_params, 48u);
    e15 = StwoCudaQm31{ b39, b56, b56, b56 };
    e14 = stwo_qm31_mul(e18, e15);
    e15 = stwo_qm31_add(e17, e14);
    e14 = stwo_load_qm31(ext_params, 49u);
    e17 = stwo_qm31_sub(e15, e14);
    e14 = stwo_load_qm31(ext_params, 50u);
    e15 = StwoCudaQm31{ b72, b56, b56, b56 };
    e18 = stwo_qm31_mul(e14, e15);
    e15 = stwo_load_qm31(ext_params, 51u);
    e14 = stwo_qm31_add(e15, e18);
    e15 = stwo_load_qm31(ext_params, 52u);
    e18 = StwoCudaQm31{ b73, b56, b56, b56 };
    StwoCudaQm31 e19 = stwo_qm31_mul(e15, e18);
    e18 = stwo_qm31_add(e14, e19);
    e19 = stwo_load_qm31(ext_params, 53u);
    e14 = StwoCudaQm31{ b40, b56, b56, b56 };
    e15 = stwo_qm31_mul(e19, e14);
    e14 = stwo_qm31_add(e18, e15);
    e15 = stwo_load_qm31(ext_params, 54u);
    e18 = stwo_qm31_sub(e14, e15);
    e15 = stwo_load_qm31(ext_params, 55u);
    e14 = StwoCudaQm31{ b35, b56, b56, b56 };
    e19 = stwo_qm31_mul(e15, e14);
    e14 = stwo_load_qm31(ext_params, 56u);
    e15 = stwo_qm31_add(e14, e19);
    e14 = stwo_load_qm31(ext_params, 57u);
    e19 = StwoCudaQm31{ b37, b56, b56, b56 };
    StwoCudaQm31 e20 = stwo_qm31_mul(e14, e19);
    e19 = stwo_qm31_add(e15, e20);
    e20 = stwo_load_qm31(ext_params, 58u);
    e15 = StwoCudaQm31{ b41, b56, b56, b56 };
    e14 = stwo_qm31_mul(e20, e15);
    e15 = stwo_qm31_add(e19, e14);
    e14 = stwo_load_qm31(ext_params, 59u);
    e19 = stwo_qm31_sub(e15, e14);
    e14 = StwoCudaQm31{ b22, b56, b56, b56 };
    e15 = StwoCudaQm31{ b75, b56, b56, b56 };
    e20 = stwo_load_qm31(ext_params, 60u);
    StwoCudaQm31 e21 = StwoCudaQm31{ b76, b56, b56, b56 };
    StwoCudaQm31 e22 = stwo_qm31_mul(e20, e21);
    e21 = stwo_load_qm31(ext_params, 61u);
    e20 = stwo_qm31_add(e21, e22);
    e21 = stwo_load_qm31(ext_params, 62u);
    e22 = StwoCudaQm31{ b68, b56, b56, b56 };
    StwoCudaQm31 e23 = stwo_qm31_mul(e21, e22);
    e22 = stwo_qm31_add(e20, e23);
    e23 = stwo_load_qm31(ext_params, 63u);
    e20 = StwoCudaQm31{ b48, b56, b56, b56 };
    e21 = stwo_qm31_mul(e23, e20);
    e20 = stwo_qm31_add(e22, e21);
    e21 = stwo_load_qm31(ext_params, 64u);
    e22 = stwo_qm31_sub(e20, e21);
    e21 = stwo_load_qm31(ext_params, 65u);
    e20 = StwoCudaQm31{ b44, b56, b56, b56 };
    e23 = stwo_qm31_mul(e21, e20);
    e20 = stwo_load_qm31(ext_params, 66u);
    e21 = stwo_qm31_add(e20, e23);
    e20 = stwo_load_qm31(ext_params, 67u);
    e23 = StwoCudaQm31{ b46, b56, b56, b56 };
    StwoCudaQm31 e24 = stwo_qm31_mul(e20, e23);
    e23 = stwo_qm31_add(e21, e24);
    e24 = stwo_load_qm31(ext_params, 68u);
    e21 = StwoCudaQm31{ b49, b56, b56, b56 };
    e20 = stwo_qm31_mul(e24, e21);
    e21 = stwo_qm31_add(e23, e20);
    e20 = stwo_load_qm31(ext_params, 69u);
    e23 = stwo_qm31_sub(e21, e20);
    e20 = stwo_load_qm31(ext_params, 70u);
    e21 = StwoCudaQm31{ b23, b56, b56, b56 };
    e24 = stwo_qm31_mul(e20, e21);
    e21 = stwo_load_qm31(ext_params, 71u);
    e20 = stwo_qm31_add(e21, e24);
    e21 = stwo_load_qm31(ext_params, 72u);
    e24 = StwoCudaQm31{ b69, b56, b56, b56 };
    StwoCudaQm31 e25 = stwo_qm31_mul(e21, e24);
    e24 = stwo_qm31_add(e20, e25);
    e25 = stwo_load_qm31(ext_params, 73u);
    e20 = StwoCudaQm31{ b50, b56, b56, b56 };
    e21 = stwo_qm31_mul(e25, e20);
    e20 = stwo_qm31_add(e24, e21);
    e21 = stwo_load_qm31(ext_params, 74u);
    e24 = stwo_qm31_sub(e20, e21);
    e21 = stwo_load_qm31(ext_params, 75u);
    e20 = StwoCudaQm31{ b45, b56, b56, b56 };
    e25 = stwo_qm31_mul(e21, e20);
    e20 = stwo_load_qm31(ext_params, 76u);
    e21 = stwo_qm31_add(e20, e25);
    e20 = stwo_load_qm31(ext_params, 77u);
    e25 = StwoCudaQm31{ b47, b56, b56, b56 };
    StwoCudaQm31 e26 = stwo_qm31_mul(e20, e25);
    e25 = stwo_qm31_add(e21, e26);
    e26 = stwo_load_qm31(ext_params, 78u);
    e21 = StwoCudaQm31{ b51, b56, b56, b56 };
    e20 = stwo_qm31_mul(e26, e21);
    e21 = stwo_qm31_add(e25, e20);
    e20 = stwo_load_qm31(ext_params, 79u);
    e25 = stwo_qm31_sub(e21, e20);
    e20 = StwoCudaQm31{ b79, b56, b56, b56 };
    e21 = StwoCudaQm31{ b52, b56, b56, b56 };
    e26 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e21);
    e21 = stwo_load_qm31(ext_params, 80u);
    StwoCudaQm31 e27 = StwoCudaQm31{ b0, b56, b56, b56 };
    StwoCudaQm31 e28 = stwo_qm31_mul(e21, e27);
    e27 = stwo_load_qm31(ext_params, 81u);
    e21 = stwo_qm31_add(e27, e28);
    e27 = stwo_load_qm31(ext_params, 82u);
    e28 = StwoCudaQm31{ b1, b56, b56, b56 };
    StwoCudaQm31 e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 83u);
    e21 = StwoCudaQm31{ b2, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 84u);
    e28 = StwoCudaQm31{ b3, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 85u);
    e21 = StwoCudaQm31{ b4, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 86u);
    e28 = StwoCudaQm31{ b5, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 87u);
    e21 = StwoCudaQm31{ b6, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 88u);
    e28 = StwoCudaQm31{ b7, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 89u);
    e21 = StwoCudaQm31{ b8, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 90u);
    e28 = StwoCudaQm31{ b9, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 91u);
    e21 = StwoCudaQm31{ b10, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 92u);
    e28 = StwoCudaQm31{ b11, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 93u);
    e21 = StwoCudaQm31{ b32, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 94u);
    e28 = StwoCudaQm31{ b33, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 95u);
    e21 = StwoCudaQm31{ b74, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 96u);
    e28 = StwoCudaQm31{ b77, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 97u);
    e21 = StwoCudaQm31{ b42, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 98u);
    e28 = StwoCudaQm31{ b43, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 99u);
    e21 = StwoCudaQm31{ b60, b56, b56, b56 };
    e27 = stwo_qm31_mul(e29, e21);
    e21 = stwo_qm31_add(e28, e27);
    e27 = stwo_load_qm31(ext_params, 100u);
    e28 = StwoCudaQm31{ b61, b56, b56, b56 };
    e29 = stwo_qm31_mul(e27, e28);
    e28 = stwo_qm31_add(e21, e29);
    e29 = stwo_load_qm31(ext_params, 101u);
    e21 = stwo_qm31_sub(e28, e29);
    e29 = stwo_load_qm31(ext_params, 102u);
    e28 = stwo_qm31_mul(e5, e29);
    e29 = stwo_load_qm31(ext_params, 103u);
    e27 = stwo_qm31_mul(e4, e29);
    e29 = stwo_qm31_add(e28, e27);
    e27 = stwo_qm31_mul(e4, e5);
    e5 = stwo_load_qm31(ext_params, 104u);
    e4 = stwo_qm31_mul(e7, e5);
    e5 = stwo_load_qm31(ext_params, 105u);
    e28 = stwo_qm31_mul(e6, e5);
    e5 = stwo_qm31_add(e4, e28);
    e28 = stwo_qm31_mul(e6, e7);
    e7 = stwo_load_qm31(ext_params, 106u);
    e6 = stwo_qm31_mul(e11, e7);
    e7 = stwo_load_qm31(ext_params, 107u);
    e4 = stwo_qm31_mul(e10, e7);
    e7 = stwo_qm31_add(e6, e4);
    e4 = stwo_qm31_mul(e10, e11);
    e11 = stwo_load_qm31(ext_params, 108u);
    e10 = stwo_qm31_mul(e13, e11);
    e11 = stwo_load_qm31(ext_params, 109u);
    e6 = stwo_qm31_mul(e12, e11);
    e11 = stwo_qm31_add(e10, e6);
    e6 = stwo_qm31_mul(e12, e13);
    e13 = stwo_load_qm31(ext_params, 110u);
    e12 = stwo_qm31_mul(e17, e13);
    e13 = stwo_load_qm31(ext_params, 111u);
    e10 = stwo_qm31_mul(e16, e13);
    e13 = stwo_qm31_add(e12, e10);
    e10 = stwo_qm31_mul(e16, e17);
    e17 = stwo_load_qm31(ext_params, 112u);
    e16 = stwo_qm31_mul(e19, e17);
    e17 = stwo_load_qm31(ext_params, 113u);
    e12 = stwo_qm31_mul(e18, e17);
    e17 = stwo_qm31_add(e16, e12);
    e12 = stwo_qm31_mul(e18, e19);
    e19 = stwo_load_qm31(ext_params, 114u);
    e18 = stwo_qm31_mul(e23, e19);
    e19 = stwo_load_qm31(ext_params, 115u);
    e16 = stwo_qm31_mul(e22, e19);
    e19 = stwo_qm31_add(e18, e16);
    e16 = stwo_qm31_mul(e22, e23);
    e23 = stwo_load_qm31(ext_params, 116u);
    e22 = stwo_qm31_mul(e25, e23);
    e23 = stwo_load_qm31(ext_params, 117u);
    e18 = stwo_qm31_mul(e24, e23);
    e23 = stwo_qm31_add(e22, e18);
    e18 = stwo_qm31_mul(e24, e25);
    e25 = StwoCudaQm31{ b78, b80, b81, b82 };
    e24 = stwo_qm31_mul(e25, e27);
    e27 = stwo_qm31_sub(e24, e29);
    e24 = StwoCudaQm31{ b83, b84, b85, b86 };
    e29 = stwo_qm31_sub(e24, e25);
    e25 = stwo_qm31_mul(e29, e28);
    e29 = stwo_qm31_sub(e25, e5);
    e25 = StwoCudaQm31{ b87, b88, b89, b90 };
    e5 = stwo_qm31_sub(e25, e24);
    e24 = stwo_qm31_mul(e5, e4);
    e5 = stwo_qm31_sub(e24, e7);
    e24 = StwoCudaQm31{ b91, b92, b93, b94 };
    e7 = stwo_qm31_sub(e24, e25);
    e25 = stwo_qm31_mul(e7, e6);
    e7 = stwo_qm31_sub(e25, e11);
    e25 = StwoCudaQm31{ b95, b96, b97, b98 };
    e11 = stwo_qm31_sub(e25, e24);
    e24 = stwo_qm31_mul(e11, e10);
    e11 = stwo_qm31_sub(e24, e13);
    e24 = StwoCudaQm31{ b99, b100, b101, b102 };
    e13 = stwo_qm31_sub(e24, e25);
    e25 = stwo_qm31_mul(e13, e12);
    e13 = stwo_qm31_sub(e25, e17);
    e25 = StwoCudaQm31{ b103, b104, b105, b106 };
    e17 = stwo_qm31_sub(e25, e24);
    e24 = stwo_qm31_mul(e17, e16);
    e17 = stwo_qm31_sub(e24, e19);
    e24 = StwoCudaQm31{ b107, b108, b109, b110 };
    e19 = stwo_qm31_sub(e24, e25);
    e25 = stwo_qm31_mul(e19, e18);
    e19 = stwo_qm31_sub(e25, e23);
    e25 = StwoCudaQm31{ b111, b113, b115, b117 };
    e23 = StwoCudaQm31{ b112, b114, b116, b118 };
    e18 = stwo_qm31_sub(e23, e25);
    e23 = stwo_qm31_sub(e18, e24);
    e18 = stwo_load_qm31(ext_params, 118u);
    e24 = stwo_qm31_add(e23, e18);
    e18 = stwo_qm31_mul(e24, e21);
    e24 = stwo_qm31_sub(e18, e26);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e0, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e1, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e2, stwo_load_qm31(random_coeff_powers, rc_base + 2u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e3, stwo_load_qm31(random_coeff_powers, rc_base + 3u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e8, stwo_load_qm31(random_coeff_powers, rc_base + 4u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e9, stwo_load_qm31(random_coeff_powers, rc_base + 5u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e14, stwo_load_qm31(random_coeff_powers, rc_base + 6u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e15, stwo_load_qm31(random_coeff_powers, rc_base + 7u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e20, stwo_load_qm31(random_coeff_powers, rc_base + 8u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e27, stwo_load_qm31(random_coeff_powers, rc_base + 9u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e29, stwo_load_qm31(random_coeff_powers, rc_base + 10u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e5, stwo_load_qm31(random_coeff_powers, rc_base + 11u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e7, stwo_load_qm31(random_coeff_powers, rc_base + 12u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e11, stwo_load_qm31(random_coeff_powers, rc_base + 13u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e13, stwo_load_qm31(random_coeff_powers, rc_base + 14u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e17, stwo_load_qm31(random_coeff_powers, rc_base + 15u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e19, stwo_load_qm31(random_coeff_powers, rc_base + 16u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e24, stwo_load_qm31(random_coeff_powers, rc_base + 17u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
