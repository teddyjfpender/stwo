//! CUDA source emitter for witness-JIT programs.
//!
//! Emits one `__global__` kernel, one thread per row: read the packed inputs, replay
//! the recorded instruction stream in registers, write committed columns, atomic-add
//! multiplicities, and store lookup words. The M31 field helpers are the *same*
//! formulas as the constraint lane's preamble (`super::super::jit::cuda_codegen`), which
//! are byte-equal-proven against the CPU reference by the Metal/CUDA conformance gate —
//! so reusing them keeps the witness kernel's field arithmetic on the same proven
//! footing. Integer ops map to native CUDA `unsigned` arithmetic.
//!
//! The kernel ABI is explicit C (pointer/scalar parameters only) — no Rust struct
//! layouts crossing the boundary, the same discipline that makes the constraint JIT
//! kernels portable across AIR/compiler revisions.

use super::isa::{DeduceKind, WitnessOp, WitnessProgram};

/// Bumped whenever the emitted source for a fixed program changes, mixed into the
/// cache key so new source can never collide with PTX an older build persisted for the
/// same bytecode (same rule as the constraint lane's `CODEGEN_VERSION`).
pub const WITNESS_CODEGEN_VERSION: u64 = 9;

/// Cache key: program semantic hash mixed (FNV-1a) with [`WITNESS_CODEGEN_VERSION`].
pub fn witness_jit_cache_key(semantic_hash: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in semantic_hash
        .to_le_bytes()
        .into_iter()
        .chain(WITNESS_CODEGEN_VERSION.to_le_bytes())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn witness_kernel_name(semantic_hash: u64) -> String {
    format!("stwo_jit_witness_{semantic_hash:016x}")
}

/// Compile a witness program into CUDA C source. Returns `None` if the program carries
/// an opcode this emitter does not handle (the caller then falls back to the host lane).
pub fn compile_witness_to_cuda_source(program: &WitnessProgram) -> Option<String> {
    let name = witness_kernel_name(program.semantic_hash());
    let mut src = String::with_capacity(8192);
    emit_preamble(&mut src);

    // ISA-V3 computed deduces: embed the needed __device__ functions (transcribed
    // 1:1 from the host fast_deduction routines; validated by the truth-oracle legs
    // + the component differential on hardware). Blake kinds embed small inline
    // functions; the fp256/EC kinds pull in the kernels crate's fp256 chain +
    // `stwo_wit_deduce.cuh` (see `emit_fp256_deduce_support`).
    let mut kinds_used: Vec<DeduceKind> = Vec::new();
    for inst in &program.insts {
        if WitnessOp::from_raw(inst.op) == Some(WitnessOp::DeduceCall) {
            let kind = DeduceKind::from_raw(inst.imm)?;
            if !kinds_used.contains(&kind) {
                kinds_used.push(kind);
            }
        }
    }
    let uses_fp256 = |k: &DeduceKind| {
        matches!(
            k,
            DeduceKind::PartialEcMulW18
                | DeduceKind::PedersenPointsTableW18
                | DeduceKind::FeltAdd
                | DeduceKind::FeltSub
                | DeduceKind::FeltMul
                | DeduceKind::FeltDiv
        )
    };
    if kinds_used.iter().any(uses_fp256) {
        emit_fp256_deduce_support(&mut src);
    }
    if kinds_used.contains(&DeduceKind::BlakeG) {
        // fast_deduction/blake.rs::PackedBlakeG::blake_g, verbatim on scalar u32
        // (rotate = u32::rotate_right).
        src.push_str(
            "static __device__ __forceinline__ unsigned stwo_wit_rotr(unsigned x, unsigned n) {\n\
             \x20   return (x >> n) | (x << (32u - n));\n\
             }\n\
             static __device__ __forceinline__ void stwo_wit_blake_g(\n\
             \x20   const unsigned *in, unsigned *out) {\n\
             \x20   unsigned a = in[0], b = in[1], c = in[2], d = in[3];\n\
             \x20   const unsigned m0 = in[4], m1 = in[5];\n\
             \x20   a = a + b + m0; d ^= a; d = stwo_wit_rotr(d, 16u);\n\
             \x20   c += d; b ^= c; b = stwo_wit_rotr(b, 12u);\n\
             \x20   a = a + b + m1; d ^= a; d = stwo_wit_rotr(d, 8u);\n\
             \x20   c += d; b ^= c; b = stwo_wit_rotr(b, 7u);\n\
             \x20   out[0] = a; out[1] = b; out[2] = c; out[3] = d;\n\
             }\n\n",
        );
    }
    if kinds_used.contains(&DeduceKind::BlakeRoundSigma) {
        // preprocessed_columns/blake.rs::BLAKE_SIGMA, verbatim.
        src.push_str(
            "static __device__ const unsigned STWO_WIT_BLAKE_SIGMA[10][16] = {\n\
             \x20   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},\n\
             \x20   {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},\n\
             \x20   {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},\n\
             \x20   {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},\n\
             \x20   {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},\n\
             \x20   {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},\n\
             \x20   {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},\n\
             \x20   {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},\n\
             \x20   {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},\n\
             \x20   {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0}\n\
             };\n\n",
        );
    }

    // Value-limb deduce with the encoded-id tag dispatch (semantics proven by the
    // exec_deduce_output differential over real PIE memories). Table pointer layout:
    // [0]=addr_to_id, [1..29]=big limb columns, [29..37]=small limb columns;
    // strides carry the table lengths [n_addrs, n_big, n_small] for clamping.
    // x^(P-2) by square-and-multiply over the fixed exponent 2^31 - 3
    // (binary: thirty-one bits, all ones except bit 1). Total function —
    // inverse(0) = 0 — matching the CPU `FieldExpOps::inverse` power semantics.
    src.push_str(
        "static __device__ __forceinline__ unsigned stwo_m31_inverse(unsigned a) {\n\
         \x20   unsigned result = a;                 // consumes exponent bit 30\n\
         \x20   for (int bit = 29; bit >= 0; --bit) {\n\
         \x20       result = stwo_m31_mul(result, result);\n\
         \x20       if (bit != 1) { result = stwo_m31_mul(result, a); }\n\
         \x20   }\n\
         \x20   return result;\n\
         }\n\n",
    );
    src.push_str(
        "static __device__ __forceinline__ unsigned stwo_wit_deduce_limb(\n\
         \x20   const unsigned *const *tb, const unsigned *ts, unsigned id, unsigned limb) {\n\
         \x20   unsigned tag = id >> 30u;\n\
         \x20   unsigned val = id & 0x3FFFFFFFu;\n\
         \x20   if (tag == 1u) { return val < ts[1] ? tb[1u + limb][val] : 0u; }\n\
         \x20   return (limb < 8u && val < ts[2]) ? tb[29u + limb][val] : 0u;\n\
         }\n\n",
    );

    src.push_str(&format!(
        "extern \"C\" __global__ void __launch_bounds__(256) {name}(\n\
         \x20   const unsigned *const *input_cols,   // [n_inputs][row]\n\
         \x20   const unsigned *const *table_bases,  // deduce_output LUTs, per table\n\
         \x20   const unsigned *table_strides,       // words per key, per table\n\
         \x20   unsigned *const *out_cols,           // [n_cols][row]\n\
         \x20   unsigned *const *mult_counts,        // atomic count tables, per mult table\n\
         \x20   unsigned *lookup_words,              // [k * row_count + row] (word-major)\n\
         \x20   unsigned *sub_words,                 // [k * row_count + row] (word-major)\n\
         \x20   unsigned row_count\n\
         ) {{\n\
         \x20   unsigned row = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (row >= row_count) {{ return; }}\n\n"
    ));

    emit_body(program, &mut src)?;

    src.push_str("}\n");
    Some(src)
}

fn emit_body(program: &WitnessProgram, src: &mut String) -> Option<()> {
    let mut declared = vec![false; program.n_regs as usize];
    let mut deduce_args: Vec<u32> = Vec::new();
    let mut deduce_seq = 0usize;

    for inst in &program.insts {
        let op = WitnessOp::from_raw(inst.op)?;
        let (a, b, imm) = (inst.a, inst.b, inst.imm);

        // Output opcodes write memory and produce no register.
        match op {
            WitnessOp::ColWrite => {
                src.push_str(&format!("    out_cols[{imm}u][row] = r{a};\n"));
                continue;
            }
            WitnessOp::MultPush => {
                // Order-independent atomic accumulation (byte-equal by construction —
                // field/count adds commute), exactly as the host AtomicMultiplicityColumn.
                src.push_str(&format!("    atomicAdd(&mult_counts[{imm}u][r{a}], 1u);\n"));
                continue;
            }
            WitnessOp::LookupWord => {
                // Word-major (`[k * row_count + row]`): adjacent threads write adjacent
                // addresses (coalesced), and the host copy repacks each 16-lane
                // PackedM31 from one contiguous 64B run. The prove accessors and the
                // selftest comparator index the same way — any change here must bump
                // WITNESS_CODEGEN_VERSION and update both.
                src.push_str(&format!(
                    "    lookup_words[{imm}u * row_count + row] = r{a};\n"
                ));
                continue;
            }
            WitnessOp::SubWord => {
                src.push_str(&format!(
                    "    sub_words[{imm}u * row_count + row] = r{a};\n"
                ));
                continue;
            }
            WitnessOp::DeduceArg => {
                deduce_args.push(a);
                continue;
            }
            WitnessOp::DeduceCall => {
                let kind = DeduceKind::from_raw(imm)?;
                let (n_args, n_outs) = kind.shape();
                if deduce_args.len() != n_args {
                    return None; // malformed program — recorder bug; fall back.
                }
                let args_list = deduce_args
                    .iter()
                    .map(|r| format!("r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let seq = deduce_seq;
                deduce_seq += 1;
                src.push_str(&format!(
                    "    const unsigned dargs{seq}[{n_args}] = {{ {args_list} }};\n\
                     \x20   unsigned douts{seq}[{n_outs}];\n"
                ));
                match kind {
                    DeduceKind::BlakeG => {
                        src.push_str(&format!("    stwo_wit_blake_g(dargs{seq}, douts{seq});\n"));
                    }
                    DeduceKind::BlakeRoundSigma => {
                        src.push_str(&format!(
                            "    for (int i = 0; i < 16; ++i) {{ douts{seq}[i] = \
                             STWO_WIT_BLAKE_SIGMA[dargs{seq}[0]][i]; }}\n"
                        ));
                    }
                    DeduceKind::PartialEcMulW18 => {
                        src.push_str(&format!(
                            "    stwo_wit_deduce_partial_ec_mul_w18(dargs{seq}, douts{seq});\n"
                        ));
                    }
                    DeduceKind::PedersenPointsTableW18 => {
                        src.push_str(&format!(
                            "    stwo_wit_deduce_pedersen_points_w18(dargs{seq}, douts{seq});\n"
                        ));
                    }
                    DeduceKind::FeltAdd => {
                        src.push_str(&format!(
                            "    stwo_wit_deduce_felt_add(dargs{seq}, douts{seq});\n"
                        ));
                    }
                    DeduceKind::FeltSub => {
                        src.push_str(&format!(
                            "    stwo_wit_deduce_felt_sub(dargs{seq}, douts{seq});\n"
                        ));
                    }
                    DeduceKind::FeltMul => {
                        src.push_str(&format!(
                            "    stwo_wit_deduce_felt_mul(dargs{seq}, douts{seq});\n"
                        ));
                    }
                    DeduceKind::FeltDiv => {
                        src.push_str(&format!(
                            "    stwo_wit_deduce_felt_div(dargs{seq}, douts{seq});\n"
                        ));
                    }
                }
                let base = inst.dst as usize;
                for i in 0..n_outs {
                    let reg = base + i;
                    let decl = if !declared[reg] {
                        declared[reg] = true;
                        "unsigned "
                    } else {
                        ""
                    };
                    src.push_str(&format!("    {decl}r{reg} = douts{seq}[{i}];\n"));
                }
                deduce_args.clear();
                continue;
            }
            _ => {}
        }

        let dst = inst.dst as usize;
        let decl = if !declared[dst] {
            declared[dst] = true;
            "unsigned "
        } else {
            ""
        };
        let expr = match op {
            WitnessOp::Input => format!("input_cols[{a}u][row]"),
            WitnessOp::Const => format!("{imm}u"),
            WitnessOp::M31Add => format!("stwo_m31_add(r{a}, r{b})"),
            WitnessOp::M31Sub => format!("stwo_m31_sub(r{a}, r{b})"),
            WitnessOp::M31Mul => format!("stwo_m31_mul(r{a}, r{b})"),
            WitnessOp::M31Neg => format!("stwo_m31_neg(r{a})"),
            WitnessOp::U16Add => format!("((r{a} + r{b}) & 0xFFFFu)"),
            WitnessOp::U16Shl => format!("((r{a} << {imm}u) & 0xFFFFu)"),
            WitnessOp::U16Shr => format!("((r{a} & 0xFFFFu) >> {imm}u)"),
            WitnessOp::U16And => format!("(r{a} & {imm}u)"),
            WitnessOp::U32Add => format!("(r{a} + r{b})"),
            WitnessOp::U32Sub => format!("(r{a} - r{b})"),
            WitnessOp::U32Mul => format!("(r{a} * r{b})"),
            WitnessOp::U32Shl => format!("(r{a} << {imm}u)"),
            WitnessOp::U32Shr => format!("(r{a} >> {imm}u)"),
            WitnessOp::U32And => format!("(r{a} & {imm}u)"),
            WitnessOp::U32Xor => format!("(r{a} ^ r{b})"),
            WitnessOp::AsM31 => format!("(r{a} % STWO_M31_P)"),
            WitnessOp::M31Inverse => format!("stwo_m31_inverse(r{a})"),
            WitnessOp::M31Eq => format!("(r{a} == r{b} ? 1u : 0u)"),
            WitnessOp::Trunc16 => format!("(r{a} & 0xFFFFu)"),
            WitnessOp::TableLimb => {
                // The REAL execution-table semantics (mirrors the proven
                // exec_deduce_output_kernel; see the table layout in
                // `exec_tables::witness_table_pointers`):
                //   table 0 (ADDR_TO_ID): raw encoded id at address — one flat column.
                //   table 1 (ID_TO_BIG): limb of the value behind an ENCODED id —
                //     tag==1 -> big table (28 limb columns), else small (8 columns,
                //     zero-extended). Out-of-range keys clamp to 0 (empty cells are
                //     never legitimately queried; clamping keeps a bad row from
                //     becoming an out-of-bounds device read).
                match b {
                    0 => format!("(r{a} < table_strides[0u] ? table_bases[0u][r{a}] : 0u)"),
                    1 => format!("stwo_wit_deduce_limb(table_bases, table_strides, r{a}, {imm}u)"),
                    t => {
                        let _ = t;
                        return None;
                    }
                }
            }
            WitnessOp::ColWrite
            | WitnessOp::MultPush
            | WitnessOp::LookupWord
            | WitnessOp::SubWord
            | WitnessOp::DeduceArg
            | WitnessOp::DeduceCall => unreachable!(),
        };
        src.push_str(&format!("    {decl}r{dst} = {expr};\n"));
    }
    Some(())
}

/// fp256/EC computed-deduce support: the kernels crate's proven fp256 chain
/// (storage → ptx carries → host math → carry chain → config → dispatch →
/// ec_ops) plus the `stwo_wit_deduce.cuh` shim, textually embedded because
/// NVRTC resolves no `#include`. The prelude supplies exactly what the stripped
/// includes provided (`<cstdint>` typedefs, the `utils.cuh` inline macros, the
/// `fields.cuh` m31 typedef, device printf for the chain's debug branches);
/// `size_t` is an NVRTC builtin. Only emitted when a program actually carries
/// EC deduces — the embed is ~70KB of source and only the pedersen-family
/// kernels should pay the NVRTC time for it.
///
/// The embedded `stwo_wit_deduce.cuh` defines module-scope table globals
/// (`g_stwo_wit_pedersen_cols` / `..._n_rows`) which `runtime_jit.cu` fills
/// right after module load — device globals do not cross CUmodule boundaries.
fn emit_fp256_deduce_support(src: &mut String) {
    const PRELUDE: &str = "\
// ---- fp256/EC embed prelude (self-contained TU: no headers resolved) ----
#define STWO_WIT_EMBED 1
namespace std {}
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned int m31;
#define UINT32_MAX 0xFFFFFFFFu
#if !defined(__align__)
#define __align__(n) alignas(n)
#endif
// NVRTC compiles device code ONLY and its JIT mode hard-errors on any function
// carrying __host__ (alone). HOST_* therefore lower to plain __device__ here:
// host-only chain functions become dead device functions and are discarded,
// while literal `__host__` sites in the chain sit behind !__CUDACC_RTC__ guards.
#define HOST_INLINE __device__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __device__ __forceinline__
extern \"C\" __device__ int printf(const char*, ...);

";
    const CHAIN: [&str; 8] = [
        include_str!("../../../../backend-cuda-kernels/cuda/fp256_storage.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/ptx.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/fp256_host_math.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/fp256_carry_chain.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/fp256_config.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/fp256_dispatch_st.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/ec_ops.cuh"),
        include_str!("../../../../backend-cuda-kernels/cuda/stwo_wit_deduce.cuh"),
    ];
    src.push_str(PRELUDE);
    for file in CHAIN {
        for line in file.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#include") || trimmed.starts_with("#pragma once") {
                continue;
            }
            src.push_str(line);
            src.push('\n');
        }
    }
    src.push('\n');
}

fn emit_preamble(src: &mut String) {
    // Field helpers copied verbatim from the constraint lane's CUDA preamble (proven
    // byte-equal to the CPU reference); only the three used by witness decode
    // (add/sub/mul/neg) are needed, but keeping the exact text keeps a single source of
    // truth for M31 arithmetic across both JIT lanes.
    src.push_str(
        "\
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

",
    );
}

#[cfg(test)]
mod tests {
    use super::super::recording::WitnessRecorder;
    use super::*;

    #[test]
    fn cache_key_depends_on_codegen_version() {
        // Distinct semantic hashes give distinct keys; the version is mixed in.
        assert_ne!(witness_jit_cache_key(1), witness_jit_cache_key(2));
        assert_ne!(witness_jit_cache_key(1), 1);
    }

    #[test]
    fn deduce_codegen_and_interp_roundtrip() {
        use super::super::interp::{interpret_row_with, DeduceHost};
        use super::super::isa::DeduceKind;

        // Record: g = blake_g(inputs 0..6); sigma = sigma(input 6); commit a few outs.
        let mut r = WitnessRecorder::new("deduce_probe");
        let ins: Vec<_> = (0..7).map(|i| r.input(i)).collect();
        let g = r.deduce(DeduceKind::BlakeG, &ins[..6]);
        let sg = r.deduce(DeduceKind::BlakeRoundSigma, &ins[6..7]);
        r.col_write(0, g[0]);
        r.col_write(1, g[3]);
        r.col_write(2, sg[15]);
        let prog = r.finish();

        // Codegen: embeds both device fns + the call/bank pattern.
        let src = compile_witness_to_cuda_source(&prog).expect("codegen succeeds");
        assert!(src.contains("stwo_wit_blake_g"), "src: {src}");
        assert!(src.contains("STWO_WIT_BLAKE_SIGMA"), "src: {src}");
        assert!(src.contains("dargs0[6]"), "src: {src}");
        assert!(src.contains("douts1[16]"), "src: {src}");

        // Interp with a reference host: blake2s g + sigma row, straight math.
        struct RefHost;
        impl DeduceHost for RefHost {
            fn deduce(&mut self, kind: u32, args: &[u32]) -> Vec<u32> {
                match kind {
                    0 => {
                        let (mut a, mut b, mut c, mut d, m0, m1) =
                            (args[0], args[1], args[2], args[3], args[4], args[5]);
                        a = a.wrapping_add(b).wrapping_add(m0);
                        d ^= a;
                        d = d.rotate_right(16);
                        c = c.wrapping_add(d);
                        b ^= c;
                        b = b.rotate_right(12);
                        a = a.wrapping_add(b).wrapping_add(m1);
                        d ^= a;
                        d = d.rotate_right(8);
                        c = c.wrapping_add(d);
                        b ^= c;
                        b = b.rotate_right(7);
                        vec![a, b, c, d]
                    }
                    1 => {
                        const SIGMA1: [u32; 16] =
                            [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3];
                        assert_eq!(args[0], 1);
                        SIGMA1.to_vec()
                    }
                    k => panic!("unexpected kind {k}"),
                }
            }
        }
        let inputs = [
            0xdead_beefu32,
            0x0123_4567,
            0x89ab_cdef,
            0x5555_aaaa,
            7,
            11,
            1,
        ];
        let out = interpret_row_with(&prog, &inputs, &|_t, _k, _l| 0u32, &mut RefHost);
        // Independent check of column 0 (a') for these inputs.
        let mut a = 0xdead_beefu32.wrapping_add(0x0123_4567).wrapping_add(7);
        let mut d = 0x5555_aaaa ^ a;
        d = d.rotate_right(16);
        let c = 0x89ab_cdefu32.wrapping_add(d);
        let mut b = 0x0123_4567u32 ^ c;
        b = b.rotate_right(12);
        a = a.wrapping_add(b).wrapping_add(11);
        assert_eq!(out.columns[0], a);
        assert_eq!(out.columns[2], 3); // SIGMA[1][15]
    }

    #[test]
    fn ec_deduce_codegen_embeds_fp256_support() {
        use super::super::isa::DeduceKind;

        // One W18 EC round chained into a points-table read — the aggregator shape.
        let mut r = WitnessRecorder::new("ec_deduce_probe");
        let ins: Vec<_> = (0..72).map(|i| r.input(i)).collect();
        let round = r.deduce(DeduceKind::PartialEcMulW18, &ins);
        let point = r.deduce(DeduceKind::PedersenPointsTableW18, &round[..1]);
        r.col_write(0, round[71]);
        r.col_write(1, point[55]);
        let prog = r.finish();

        let src = compile_witness_to_cuda_source(&prog).expect("EC deduce codegen succeeds");
        assert!(
            src.contains("stwo_wit_deduce_partial_ec_mul_w18(dargs0, douts0);"),
            "missing W18 call"
        );
        assert!(
            src.contains("stwo_wit_deduce_pedersen_points_w18(dargs1, douts1);"),
            "missing points call"
        );
        // The full fp256 chain + shim + per-module table globals are embedded…
        assert!(src.contains("ff_dispatch_st"));
        assert!(src.contains("ec_add_affine"));
        assert!(src.contains("g_stwo_wit_pedersen_cols"));
        // …and self-contained: NVRTC resolves no includes.
        assert!(!src.contains("#include"), "unstripped #include in embed");
        assert!(src.contains("douts0[72]"));
        assert!(src.contains("douts1[56]"));
    }

    #[test]
    fn felt_deduce_codegen_embeds_fp256_support() {
        use super::super::isa::DeduceKind;

        // A chained slope computation: div feeding mul feeding sub — the
        // partial_ec_mul shape.
        let mut r = WitnessRecorder::new("felt_deduce_probe");
        let ins: Vec<_> = (0..112).map(|i| r.input(i)).collect();
        let q = r.deduce(DeduceKind::FeltDiv, &ins[..56]);
        let mut margs = q.clone();
        margs.extend_from_slice(&ins[56..84]);
        let m = r.deduce(DeduceKind::FeltMul, &margs);
        let mut sargs = m.clone();
        sargs.extend_from_slice(&ins[84..112]);
        let d = r.deduce(DeduceKind::FeltSub, &sargs);
        let mut aargs = d.clone();
        aargs.extend_from_slice(&ins[..28]);
        let a = r.deduce(DeduceKind::FeltAdd, &aargs);
        r.col_write(0, a[27]);
        let prog = r.finish();

        let src = compile_witness_to_cuda_source(&prog).expect("felt codegen succeeds");
        assert!(src.contains("stwo_wit_deduce_felt_div(dargs0, douts0);"));
        assert!(src.contains("stwo_wit_deduce_felt_mul(dargs1, douts1);"));
        assert!(src.contains("stwo_wit_deduce_felt_sub(dargs2, douts2);"));
        assert!(src.contains("stwo_wit_deduce_felt_add(dargs3, douts3);"));
        assert!(src.contains("ff_dispatch_st"), "fp256 chain embedded");
        assert!(src.contains("douts0[28]"));
        assert!(!src.contains("#include"));
    }

    #[test]
    fn blake_only_programs_skip_fp256_embed() {
        use super::super::isa::DeduceKind;

        // Compile-size guard: the ~70KB fp256 chain must only be paid by kernels
        // that actually carry EC deduces.
        let mut r = WitnessRecorder::new("blake_only_probe");
        let ins: Vec<_> = (0..6).map(|i| r.input(i)).collect();
        let g = r.deduce(DeduceKind::BlakeG, &ins);
        r.col_write(0, g[0]);
        let prog = r.finish();

        let src = compile_witness_to_cuda_source(&prog).expect("codegen succeeds");
        assert!(src.contains("stwo_wit_blake_g"));
        assert!(!src.contains("ff_dispatch_st"));
        assert!(!src.contains("g_stwo_wit_pedersen_cols"));
    }

    #[test]
    fn emits_kernel_for_every_opcode_shape() {
        // A program that exercises inputs, all arithmetic families, a table read, and
        // every output kind must codegen to Some(source) mentioning each sink.
        let mut r = WitnessRecorder::new("codegen_probe");
        let pc = r.input(0);
        let id = r.table_limb(0, pc, 0);
        let lo = r.from_m31(id);
        let masked = r.u16_and(lo, 127);
        let shifted = r.u16_shl(masked, 9);
        let summed = r.u16_add(lo, shifted);
        let field = r.as_m31(summed);
        let c1 = r.constant(1);
        let dec = r.m31_sub(field, c1);
        let w = r.u32_xor(lo, masked);
        let _ = r.u32_add(w, w);
        r.col_write(0, dec);
        r.mult_push(3, id);
        r.lookup_word(0, pc);
        let prog = r.finish();

        let src = compile_witness_to_cuda_source(&prog).expect("codegen succeeds");
        assert!(src.contains(&witness_kernel_name(prog.semantic_hash())));
        assert!(src.contains("out_cols[0u][row]"));
        assert!(src.contains("atomicAdd(&mult_counts[3u]"));
        assert!(src.contains("lookup_words[0u * row_count + row]"));
        assert!(src.contains("table_bases[0u]"));
        assert!(src.contains("% STWO_M31_P"));
        assert!(src.contains("& 0xFFFFu"));
    }
}
