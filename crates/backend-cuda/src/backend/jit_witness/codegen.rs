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

use super::isa::{WitnessOp, WitnessProgram};

/// Bumped whenever the emitted source for a fixed program changes, mixed into the
/// cache key so new source can never collide with PTX an older build persisted for the
/// same bytecode (same rule as the constraint lane's `CODEGEN_VERSION`).
pub const WITNESS_CODEGEN_VERSION: u64 = 4;

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
            | WitnessOp::SubWord => unreachable!(),
        };
        src.push_str(&format!("    {decl}r{dst} = {expr};\n"));
    }
    Some(())
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
