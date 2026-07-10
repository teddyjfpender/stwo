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

extern "C" __global__ void __launch_bounds__(128) stwo_jit_fused_119c78399329b29a(
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
    unsigned b6 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 257u, row_index, 0);
    unsigned b7 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 258u, row_index, 0);
    unsigned b8 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 259u, row_index, 0);
    unsigned b9 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 260u, row_index, 0);
    unsigned b10 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 261u, row_index, 0);
    unsigned b11 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 262u, row_index, 0);
    unsigned b12 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 263u, row_index, 0);
    unsigned b13 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 264u, row_index, 0);
    unsigned b14 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 265u, row_index, 0);
    unsigned b15 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 266u, row_index, 0);
    unsigned b16 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 267u, row_index, 0);
    unsigned b17 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 268u, row_index, 0);
    unsigned b18 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 269u, row_index, 0);
    unsigned b19 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 270u, row_index, 0);
    unsigned b20 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 271u, row_index, 0);
    unsigned b21 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 272u, row_index, 0);
    unsigned b22 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 273u, row_index, 0);
    unsigned b23 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 274u, row_index, 0);
    unsigned b24 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 275u, row_index, 0);
    unsigned b25 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 276u, row_index, 0);
    unsigned b26 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 277u, row_index, 0);
    unsigned b27 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 278u, row_index, 0);
    unsigned b28 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 279u, row_index, 0);
    unsigned b29 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 280u, row_index, 0);
    unsigned b30 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 281u, row_index, 0);
    unsigned b31 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 282u, row_index, 0);
    unsigned b32 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 283u, row_index, 0);
    unsigned b33 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 284u, row_index, 0);
    unsigned b34 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 285u, row_index, 0);
    unsigned b35 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 286u, row_index, 0);
    unsigned b36 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 287u, row_index, 0);
    unsigned b37 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 288u, row_index, 0);
    unsigned b38 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 289u, row_index, 0);
    unsigned b39 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 290u, row_index, 0);
    unsigned b40 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 291u, row_index, 0);
    unsigned b41 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 292u, row_index, 0);
    unsigned b42 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 293u, row_index, 0);
    unsigned b43 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 294u, row_index, 0);
    unsigned b44 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 295u, row_index, 0);
    unsigned b45 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 296u, row_index, 0);
    unsigned b46 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 297u, row_index, 0);
    unsigned b47 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 298u, row_index, 0);
    unsigned b48 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 299u, row_index, 0);
    unsigned b49 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 300u, row_index, 0);
    unsigned b50 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 301u, row_index, 0);
    unsigned b51 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 302u, row_index, 0);
    unsigned b52 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 303u, row_index, 0);
    unsigned b53 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 304u, row_index, 0);
    unsigned b54 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 305u, row_index, 0);
    unsigned b55 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 306u, row_index, 0);
    unsigned b56 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 307u, row_index, 0);
    unsigned b57 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 308u, row_index, 0);
    unsigned b58 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 309u, row_index, 0);
    unsigned b59 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 310u, row_index, 0);
    unsigned b60 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 311u, row_index, 0);
    unsigned b61 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 312u, row_index, 0);
    unsigned b62 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 313u, row_index, 0);
    unsigned b63 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 314u, row_index, 0);
    unsigned b64 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 315u, row_index, 0);
    unsigned b65 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 316u, row_index, 0);
    unsigned b66 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 317u, row_index, 0);
    unsigned b67 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 318u, row_index, 0);
    unsigned b68 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 319u, row_index, 0);
    unsigned b69 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 320u, row_index, 0);
    unsigned b70 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 321u, row_index, 0);
    unsigned b71 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 322u, row_index, 0);
    unsigned b72 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 323u, row_index, 0);
    unsigned b73 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 324u, row_index, 0);
    unsigned b74 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 325u, row_index, 0);
    unsigned b75 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 326u, row_index, 0);
    unsigned b76 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 327u, row_index, 0);
    unsigned b77 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 328u, row_index, 0);
    unsigned b78 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 329u, row_index, 0);
    unsigned b79 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 330u, row_index, 0);
    unsigned b80 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 331u, row_index, 0);
    unsigned b81 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 332u, row_index, 0);
    unsigned b82 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 333u, row_index, 0);
    unsigned b83 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 334u, row_index, 0);
    unsigned b84 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 335u, row_index, 0);
    unsigned b85 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 336u, row_index, 0);
    unsigned b86 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 337u, row_index, 0);
    unsigned b87 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 338u, row_index, 0);
    unsigned b88 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 339u, row_index, 0);
    unsigned b89 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 340u, row_index, 0);
    unsigned b90 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 1u, 341u, row_index, 0);
    unsigned b91 = 0u;
    unsigned b92 = stwo_m31_sub(b6, b36);
    b6 = 512u;
    unsigned b93 = stwo_m31_mul(b37, b6);
    b6 = stwo_m31_sub(b92, b93);
    b93 = 8192u;
    b92 = stwo_m31_mul(b6, b93);
    b93 = stwo_m31_sub(b7, b38);
    b7 = 512u;
    b6 = stwo_m31_mul(b39, b7);
    b7 = stwo_m31_sub(b93, b6);
    b6 = 8192u;
    b93 = stwo_m31_mul(b7, b6);
    b6 = stwo_m31_sub(b8, b40);
    b8 = 512u;
    b7 = stwo_m31_mul(b41, b8);
    b8 = stwo_m31_sub(b6, b7);
    b7 = 8192u;
    b6 = stwo_m31_mul(b8, b7);
    b7 = stwo_m31_sub(b9, b42);
    b9 = 512u;
    b8 = stwo_m31_mul(b43, b9);
    b9 = stwo_m31_sub(b7, b8);
    b8 = 8192u;
    b7 = stwo_m31_mul(b9, b8);
    b8 = stwo_m31_sub(b10, b44);
    b10 = 512u;
    b9 = stwo_m31_mul(b45, b10);
    b10 = stwo_m31_sub(b8, b9);
    b9 = 8192u;
    b8 = stwo_m31_mul(b10, b9);
    b9 = stwo_m31_sub(b11, b46);
    b11 = 512u;
    b10 = stwo_m31_mul(b47, b11);
    b11 = stwo_m31_sub(b9, b10);
    b10 = 8192u;
    b9 = stwo_m31_mul(b11, b10);
    b10 = stwo_m31_sub(b12, b48);
    b12 = 512u;
    b11 = stwo_m31_mul(b49, b12);
    b12 = stwo_m31_sub(b10, b11);
    b11 = 8192u;
    b10 = stwo_m31_mul(b12, b11);
    b11 = stwo_m31_sub(b13, b50);
    b13 = 512u;
    b12 = stwo_m31_mul(b51, b13);
    b13 = stwo_m31_sub(b11, b12);
    b12 = 8192u;
    b11 = stwo_m31_mul(b13, b12);
    b12 = stwo_m31_sub(b14, b52);
    b14 = 512u;
    b13 = stwo_m31_mul(b53, b14);
    b14 = stwo_m31_sub(b12, b13);
    b13 = 8192u;
    b12 = stwo_m31_mul(b14, b13);
    b13 = stwo_m31_sub(b16, b54);
    b16 = 512u;
    b14 = stwo_m31_mul(b55, b16);
    b16 = stwo_m31_sub(b13, b14);
    b14 = 8192u;
    b13 = stwo_m31_mul(b16, b14);
    b14 = stwo_m31_sub(b17, b56);
    b17 = 512u;
    b16 = stwo_m31_mul(b57, b17);
    b17 = stwo_m31_sub(b14, b16);
    b16 = 8192u;
    b14 = stwo_m31_mul(b17, b16);
    b16 = stwo_m31_sub(b18, b58);
    b18 = 512u;
    b17 = stwo_m31_mul(b59, b18);
    b18 = stwo_m31_sub(b16, b17);
    b17 = 8192u;
    b16 = stwo_m31_mul(b18, b17);
    b17 = stwo_m31_sub(b19, b60);
    b19 = 512u;
    b18 = stwo_m31_mul(b61, b19);
    b19 = stwo_m31_sub(b17, b18);
    b18 = 8192u;
    b17 = stwo_m31_mul(b19, b18);
    b18 = stwo_m31_sub(b20, b62);
    b20 = 512u;
    b19 = stwo_m31_mul(b63, b20);
    b20 = stwo_m31_sub(b18, b19);
    b19 = 8192u;
    b18 = stwo_m31_mul(b20, b19);
    b19 = stwo_m31_sub(b21, b64);
    b21 = 512u;
    b20 = stwo_m31_mul(b65, b21);
    b21 = stwo_m31_sub(b19, b20);
    b20 = 8192u;
    b19 = stwo_m31_mul(b21, b20);
    b20 = stwo_m31_sub(b22, b66);
    b22 = 512u;
    b21 = stwo_m31_mul(b67, b22);
    b22 = stwo_m31_sub(b20, b21);
    b21 = 8192u;
    b20 = stwo_m31_mul(b22, b21);
    b21 = stwo_m31_sub(b23, b68);
    b23 = 512u;
    b22 = stwo_m31_mul(b69, b23);
    b23 = stwo_m31_sub(b21, b22);
    b22 = 8192u;
    b21 = stwo_m31_mul(b23, b22);
    b22 = stwo_m31_sub(b24, b70);
    b24 = 512u;
    b23 = stwo_m31_mul(b71, b24);
    b24 = stwo_m31_sub(b22, b23);
    b23 = 8192u;
    b22 = stwo_m31_mul(b24, b23);
    b23 = stwo_m31_sub(b26, b72);
    b26 = 512u;
    b24 = stwo_m31_mul(b73, b26);
    b26 = stwo_m31_sub(b23, b24);
    b24 = 8192u;
    b23 = stwo_m31_mul(b26, b24);
    b24 = stwo_m31_sub(b27, b74);
    b27 = 512u;
    b26 = stwo_m31_mul(b75, b27);
    b27 = stwo_m31_sub(b24, b26);
    b26 = 8192u;
    b24 = stwo_m31_mul(b27, b26);
    b26 = stwo_m31_sub(b28, b76);
    b28 = 512u;
    b27 = stwo_m31_mul(b77, b28);
    b28 = stwo_m31_sub(b26, b27);
    b27 = 8192u;
    b26 = stwo_m31_mul(b28, b27);
    b27 = stwo_m31_sub(b29, b78);
    b29 = 512u;
    b28 = stwo_m31_mul(b79, b29);
    b29 = stwo_m31_sub(b27, b28);
    b28 = 8192u;
    b27 = stwo_m31_mul(b29, b28);
    b28 = stwo_m31_sub(b30, b80);
    b30 = 512u;
    b29 = stwo_m31_mul(b81, b30);
    b30 = stwo_m31_sub(b28, b29);
    b29 = 8192u;
    b28 = stwo_m31_mul(b30, b29);
    b29 = stwo_m31_sub(b31, b82);
    b31 = 512u;
    b30 = stwo_m31_mul(b83, b31);
    b31 = stwo_m31_sub(b29, b30);
    b30 = 8192u;
    b29 = stwo_m31_mul(b31, b30);
    b30 = stwo_m31_sub(b32, b84);
    b32 = 512u;
    b31 = stwo_m31_mul(b85, b32);
    b32 = stwo_m31_sub(b30, b31);
    b31 = 8192u;
    b30 = stwo_m31_mul(b32, b31);
    b31 = stwo_m31_sub(b33, b86);
    b33 = 512u;
    b32 = stwo_m31_mul(b87, b33);
    b33 = stwo_m31_sub(b31, b32);
    b32 = 8192u;
    b31 = stwo_m31_mul(b33, b32);
    b32 = stwo_m31_sub(b34, b88);
    b34 = 512u;
    b33 = stwo_m31_mul(b89, b34);
    b34 = stwo_m31_sub(b32, b33);
    b33 = 8192u;
    b32 = stwo_m31_mul(b34, b33);
    b33 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 44u, row_index, 0);
    b34 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 45u, row_index, 0);
    unsigned b94 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 46u, row_index, 0);
    unsigned b95 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 47u, row_index, 0);
    unsigned b96 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 48u, row_index, 0);
    unsigned b97 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 49u, row_index, 0);
    unsigned b98 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 50u, row_index, 0);
    unsigned b99 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 51u, row_index, 0);
    unsigned b100 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 52u, row_index, -1);
    unsigned b101 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 52u, row_index, 0);
    unsigned b102 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 53u, row_index, -1);
    unsigned b103 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 53u, row_index, 0);
    unsigned b104 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 54u, row_index, -1);
    unsigned b105 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 54u, row_index, 0);
    unsigned b106 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 55u, row_index, -1);
    unsigned b107 = stwo_trace_value(trace_cols, interaction_offsets, row_count, 2u, 55u, row_index, 0);

    // Ext instructions.
    StwoCudaQm31 e0 = stwo_load_qm31(ext_params, 447u);
    StwoCudaQm31 e1 = StwoCudaQm31{ b3, b91, b91, b91 };
    StwoCudaQm31 e2 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 448u);
    e0 = stwo_qm31_add(e1, e2);
    e1 = stwo_load_qm31(ext_params, 449u);
    e2 = StwoCudaQm31{ b36, b91, b91, b91 };
    StwoCudaQm31 e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 450u);
    e0 = StwoCudaQm31{ b37, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 451u);
    e2 = StwoCudaQm31{ b92, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 452u);
    e0 = StwoCudaQm31{ b38, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 453u);
    e2 = StwoCudaQm31{ b39, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 454u);
    e0 = StwoCudaQm31{ b93, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 455u);
    e2 = StwoCudaQm31{ b40, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 456u);
    e0 = StwoCudaQm31{ b41, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 457u);
    e2 = StwoCudaQm31{ b6, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 458u);
    e0 = StwoCudaQm31{ b42, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 459u);
    e2 = StwoCudaQm31{ b43, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 460u);
    e0 = StwoCudaQm31{ b7, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 461u);
    e2 = StwoCudaQm31{ b44, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 462u);
    e0 = StwoCudaQm31{ b45, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 463u);
    e2 = StwoCudaQm31{ b8, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 464u);
    e0 = StwoCudaQm31{ b46, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 465u);
    e2 = StwoCudaQm31{ b47, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 466u);
    e0 = StwoCudaQm31{ b9, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 467u);
    e2 = StwoCudaQm31{ b48, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 468u);
    e0 = StwoCudaQm31{ b49, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 469u);
    e2 = StwoCudaQm31{ b10, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 470u);
    e0 = StwoCudaQm31{ b50, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 471u);
    e2 = StwoCudaQm31{ b51, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 472u);
    e0 = StwoCudaQm31{ b11, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 473u);
    e2 = StwoCudaQm31{ b52, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 474u);
    e0 = StwoCudaQm31{ b53, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 475u);
    e2 = StwoCudaQm31{ b12, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e2);
    e2 = stwo_qm31_add(e0, e3);
    e3 = stwo_load_qm31(ext_params, 476u);
    e0 = StwoCudaQm31{ b15, b91, b91, b91 };
    e1 = stwo_qm31_mul(e3, e0);
    e0 = stwo_qm31_add(e2, e1);
    e1 = stwo_load_qm31(ext_params, 477u);
    e2 = stwo_qm31_sub(e0, e1);
    e1 = stwo_load_qm31(ext_params, 478u);
    e0 = StwoCudaQm31{ b4, b91, b91, b91 };
    e3 = stwo_qm31_mul(e1, e0);
    e0 = stwo_load_qm31(ext_params, 479u);
    e1 = stwo_qm31_add(e0, e3);
    e0 = stwo_load_qm31(ext_params, 480u);
    e3 = StwoCudaQm31{ b54, b91, b91, b91 };
    StwoCudaQm31 e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 481u);
    e1 = StwoCudaQm31{ b55, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 482u);
    e3 = StwoCudaQm31{ b13, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 483u);
    e1 = StwoCudaQm31{ b56, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 484u);
    e3 = StwoCudaQm31{ b57, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 485u);
    e1 = StwoCudaQm31{ b14, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 486u);
    e3 = StwoCudaQm31{ b58, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 487u);
    e1 = StwoCudaQm31{ b59, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 488u);
    e3 = StwoCudaQm31{ b16, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 489u);
    e1 = StwoCudaQm31{ b60, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 490u);
    e3 = StwoCudaQm31{ b61, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 491u);
    e1 = StwoCudaQm31{ b17, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 492u);
    e3 = StwoCudaQm31{ b62, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 493u);
    e1 = StwoCudaQm31{ b63, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 494u);
    e3 = StwoCudaQm31{ b18, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 495u);
    e1 = StwoCudaQm31{ b64, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 496u);
    e3 = StwoCudaQm31{ b65, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 497u);
    e1 = StwoCudaQm31{ b19, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 498u);
    e3 = StwoCudaQm31{ b66, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 499u);
    e1 = StwoCudaQm31{ b67, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 500u);
    e3 = StwoCudaQm31{ b20, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 501u);
    e1 = StwoCudaQm31{ b68, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 502u);
    e3 = StwoCudaQm31{ b69, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 503u);
    e1 = StwoCudaQm31{ b21, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 504u);
    e3 = StwoCudaQm31{ b70, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 505u);
    e1 = StwoCudaQm31{ b71, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 506u);
    e3 = StwoCudaQm31{ b22, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e3);
    e3 = stwo_qm31_add(e1, e4);
    e4 = stwo_load_qm31(ext_params, 507u);
    e1 = StwoCudaQm31{ b25, b91, b91, b91 };
    e0 = stwo_qm31_mul(e4, e1);
    e1 = stwo_qm31_add(e3, e0);
    e0 = stwo_load_qm31(ext_params, 508u);
    e3 = stwo_qm31_sub(e1, e0);
    e0 = stwo_load_qm31(ext_params, 509u);
    e1 = StwoCudaQm31{ b5, b91, b91, b91 };
    e4 = stwo_qm31_mul(e0, e1);
    e1 = stwo_load_qm31(ext_params, 510u);
    e0 = stwo_qm31_add(e1, e4);
    e1 = stwo_load_qm31(ext_params, 511u);
    e4 = StwoCudaQm31{ b72, b91, b91, b91 };
    StwoCudaQm31 e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 512u);
    e0 = StwoCudaQm31{ b73, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 513u);
    e4 = StwoCudaQm31{ b23, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 514u);
    e0 = StwoCudaQm31{ b74, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 515u);
    e4 = StwoCudaQm31{ b75, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 516u);
    e0 = StwoCudaQm31{ b24, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 517u);
    e4 = StwoCudaQm31{ b76, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 518u);
    e0 = StwoCudaQm31{ b77, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 519u);
    e4 = StwoCudaQm31{ b26, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 520u);
    e0 = StwoCudaQm31{ b78, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 521u);
    e4 = StwoCudaQm31{ b79, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 522u);
    e0 = StwoCudaQm31{ b27, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 523u);
    e4 = StwoCudaQm31{ b80, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 524u);
    e0 = StwoCudaQm31{ b81, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 525u);
    e4 = StwoCudaQm31{ b28, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 526u);
    e0 = StwoCudaQm31{ b82, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 527u);
    e4 = StwoCudaQm31{ b83, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 528u);
    e0 = StwoCudaQm31{ b29, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 529u);
    e4 = StwoCudaQm31{ b84, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 530u);
    e0 = StwoCudaQm31{ b85, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 531u);
    e4 = StwoCudaQm31{ b30, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 532u);
    e0 = StwoCudaQm31{ b86, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 533u);
    e4 = StwoCudaQm31{ b87, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 534u);
    e0 = StwoCudaQm31{ b31, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 535u);
    e4 = StwoCudaQm31{ b88, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 536u);
    e0 = StwoCudaQm31{ b89, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 537u);
    e4 = StwoCudaQm31{ b32, b91, b91, b91 };
    e5 = stwo_qm31_mul(e1, e4);
    e4 = stwo_qm31_add(e0, e5);
    e5 = stwo_load_qm31(ext_params, 538u);
    e0 = StwoCudaQm31{ b35, b91, b91, b91 };
    e1 = stwo_qm31_mul(e5, e0);
    e0 = stwo_qm31_add(e4, e1);
    e1 = stwo_load_qm31(ext_params, 539u);
    e4 = stwo_qm31_sub(e0, e1);
    e1 = StwoCudaQm31{ b90, b91, b91, b91 };
    e0 = stwo_qm31_sub(StwoCudaQm31{0u,0u,0u,0u}, e1);
    e1 = stwo_load_qm31(ext_params, 540u);
    e5 = StwoCudaQm31{ b0, b91, b91, b91 };
    StwoCudaQm31 e6 = stwo_qm31_mul(e1, e5);
    e5 = stwo_load_qm31(ext_params, 541u);
    e1 = stwo_qm31_add(e5, e6);
    e5 = stwo_load_qm31(ext_params, 542u);
    e6 = StwoCudaQm31{ b1, b91, b91, b91 };
    StwoCudaQm31 e7 = stwo_qm31_mul(e5, e6);
    e6 = stwo_qm31_add(e1, e7);
    e7 = stwo_load_qm31(ext_params, 543u);
    e1 = StwoCudaQm31{ b2, b91, b91, b91 };
    e5 = stwo_qm31_mul(e7, e1);
    e1 = stwo_qm31_add(e6, e5);
    e5 = stwo_load_qm31(ext_params, 544u);
    e6 = StwoCudaQm31{ b3, b91, b91, b91 };
    e7 = stwo_qm31_mul(e5, e6);
    e6 = stwo_qm31_add(e1, e7);
    e7 = stwo_load_qm31(ext_params, 545u);
    e1 = StwoCudaQm31{ b4, b91, b91, b91 };
    e5 = stwo_qm31_mul(e7, e1);
    e1 = stwo_qm31_add(e6, e5);
    e5 = stwo_load_qm31(ext_params, 546u);
    e6 = StwoCudaQm31{ b5, b91, b91, b91 };
    e7 = stwo_qm31_mul(e5, e6);
    e6 = stwo_qm31_add(e1, e7);
    e7 = stwo_load_qm31(ext_params, 547u);
    e1 = stwo_qm31_sub(e6, e7);
    e7 = stwo_load_qm31(ext_params, 572u);
    e6 = stwo_qm31_mul(e3, e7);
    e7 = stwo_load_qm31(ext_params, 573u);
    e5 = stwo_qm31_mul(e2, e7);
    e7 = stwo_qm31_add(e6, e5);
    e5 = stwo_qm31_mul(e2, e3);
    e3 = stwo_load_qm31(ext_params, 574u);
    e2 = stwo_qm31_mul(e1, e3);
    e3 = stwo_qm31_mul(e4, e0);
    e0 = stwo_qm31_add(e2, e3);
    e3 = stwo_qm31_mul(e4, e1);
    e1 = StwoCudaQm31{ b33, b34, b94, b95 };
    e4 = StwoCudaQm31{ b96, b97, b98, b99 };
    e2 = stwo_qm31_sub(e4, e1);
    e1 = stwo_qm31_mul(e2, e5);
    e2 = stwo_qm31_sub(e1, e7);
    e1 = StwoCudaQm31{ b100, b102, b104, b106 };
    e7 = StwoCudaQm31{ b101, b103, b105, b107 };
    e5 = stwo_qm31_sub(e7, e1);
    e7 = stwo_qm31_sub(e5, e4);
    e5 = stwo_load_qm31(ext_params, 575u);
    e4 = stwo_qm31_add(e7, e5);
    e5 = stwo_qm31_mul(e4, e3);
    e4 = stwo_qm31_sub(e5, e0);

    // Constraint accumulation. rc_base is the global index of this
    // kernel's first constraint within the component's random-coeff
    // powers (non-zero only for split kernels).
    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e2, stwo_load_qm31(random_coeff_powers, rc_base + 0u)));
    acc = stwo_qm31_add(acc, stwo_qm31_mul(e4, stwo_load_qm31(random_coeff_powers, rc_base + 1u)));

    // Fused denom_inv multiply + in-place accumulator update.
    unsigned denom_idx = row_index >> log_n_rows;
    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);
    coord_0[row_index] = stwo_m31_add(coord_0[row_index], result.a);
    coord_1[row_index] = stwo_m31_add(coord_1[row_index], result.b);
    coord_2[row_index] = stwo_m31_add(coord_2[row_index], result.c);
    coord_3[row_index] = stwo_m31_add(coord_3[row_index], result.d);
}
