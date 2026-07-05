// Computed-deduce device functions for the witness-JIT lane (ISA-V3 kinds 2/3).
//
// Transcribed 1:1 from `stwo-cairo`'s host reference
// (`witness/fast_deduction/pedersen.rs::PartialEcMul::<14>::deduce_output` and
// `PackedPedersenPointsTableWindowBits18::deduce_output`) onto the proven fp256
// primitives in `ec_ops.cuh`. The register ABI (flat `unsigned` banks) is the
// recorder's packing in `witness_eval/recording.rs` — change either side only
// with the other, and bump `WITNESS_CODEGEN_VERSION`.
//
// This header is compiled in TWO translation contexts:
//   1. NVRTC witness-JIT kernels: `codegen.rs` textually embeds the fp256 chain
//      (storage/ptx/carry-chain/config/dispatch/ec_ops, includes stripped) and
//      then this header, only when a program uses deduce kinds 2/3.
//   2. nvcc, via `stwo_wit_deduce_oracle.cu`: the precompiled truth-oracle
//      launcher that qualifies these functions against the host fast_deduction
//      over real rows, isolating fp256-math bugs from codegen bugs.
//
// Correctness perimeter: `ec_add_affine` is an INCOMPLETE affine add (no
// doubling/infinity branch). The pedersen shift-point table construction keeps
// all real (and, by the recording interp gates, all padding) additions on
// distinct-x points; the pod differential (`STWO_CUDA_WITNESS_VERIFY`) compares
// every row, so a violated premise fails loudly instead of shipping bytes.
#ifndef STWO_WIT_DEDUCE_CUH
#define STWO_WIT_DEDUCE_CUH

// Module-local pedersen table globals — DEFINITIONS, not externs. Device
// globals never cross CUmodule boundaries, so each module embedding this
// header carries its own copy; `runtime_jit.cu` fills them right after
// `cuModuleLoadDataEx` (and the oracle launcher fills its own at init).
// Layout: column c of row r = 9-bit M31 limb c of the point coordinate
// (x limbs in columns 0..28, y limbs in 28..56), rows flat per the host
// `PEDERSEN_TABLE_18` section layout [low0 | high0 | low2 | high2].
__device__ m31* g_stwo_wit_pedersen_cols[56];
__device__ unsigned g_stwo_wit_pedersen_n_rows;

// DeduceKind::PedersenPointsTableW18 (=3): in[1] = row index ->
// out[56] = [x limbs 0..28 | y limbs 28..56].
static __device__ __forceinline__ void stwo_wit_deduce_pedersen_points_w18(
    const unsigned* in, unsigned* out) {
    // n_rows is a power of two; the mask keeps a garbage index (poisoned or
    // adversarial recording) memory-safe. It is a no-op for every in-range row
    // — the differential gates own value correctness, this owns memory safety.
    unsigned row = in[0] & (g_stwo_wit_pedersen_n_rows - 1u);
    for (int c = 0; c < 56; ++c) {
        out[c] = g_stwo_wit_pedersen_cols[c][row];
    }
}

// DeduceKind::PartialEcMulW18 (=2): one W18 EC-mul round.
//   in[72]  = [chain, round, w0..w13, acc_x limbs 16..44, acc_y limbs 44..72]
//   out[72] = [chain, round+1, w1..w13,0, new_x limbs, new_y limbs]
// where table_row = round * 2^18 + w0 and (new_x, new_y) = acc + table[row].
static __device__ __forceinline__ void stwo_wit_deduce_partial_ec_mul_w18(
    const unsigned* in, unsigned* out) {
    AffinePointCuda acc, tbl, sum;
    felt252_from_m31_limbs(acc.x, in + 16);
    felt252_from_m31_limbs(acc.y, in + 44);

    // round < 28 and w0 < 2^18 on every recorded row (host-interp-gated), so
    // the product stays far below both 2^32 and the padded row count; the mask
    // is the same memory-safety clamp as above.
    unsigned row = in[1] * 262144u + in[2];
    row &= (g_stwo_wit_pedersen_n_rows - 1u);
    m31 limbs[28];
    for (int c = 0; c < 28; ++c) {
        limbs[c] = g_stwo_wit_pedersen_cols[c][row];
    }
    felt252_from_m31_limbs(tbl.x, limbs);
    for (int c = 0; c < 28; ++c) {
        limbs[c] = g_stwo_wit_pedersen_cols[28 + c][row];
    }
    felt252_from_m31_limbs(tbl.y, limbs);

    ec_add_affine(acc, tbl, sum);

    out[0] = in[0];      // chain passthrough
    out[1] = in[1] + 1u; // round+1; round < 28 << M31 P, so ++ IS the m31 add
    for (int i = 0; i < 13; ++i) {
        out[2 + i] = in[3 + i]; // windows shift left one
    }
    out[15] = 0u;
    felt252_to_m31_limbs(sum.x, out + 16);
    felt252_to_m31_limbs(sum.y, out + 44);
}

#endif // STWO_WIT_DEDUCE_CUH
