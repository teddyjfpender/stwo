//! CUDA source emitter for V1 evaluation programs.
//!
//! Mechanical port of the Metal JIT shader compiler (`backend-metal/src/backend/jit/
//! shader.rs`, conformance-proven byte-equal): the same instruction unrolling and the
//! same field-arithmetic formulas, emitted as self-contained CUDA C for NVRTC. The
//! kernel ABI is explicit C (pointer/scalar parameters only) — generated kernels never
//! read Rust struct layouts, which is what made the precompiled NitrooZK kernel set
//! unportable across AIR/compiler revisions.

use super::program::{
    MetalEvaluationProgramBaseOpcodeV1 as BaseOp, MetalEvaluationProgramExtOpcodeV1 as ExtOp,
    OwnedMetalEvaluationProgramV1,
};

pub fn fused_kernel_name(semantic_hash: u64) -> String {
    format!("stwo_jit_fused_{semantic_hash:016x}")
}

/// Compile a V1 program into CUDA C source defining one fused `__global__` kernel:
/// constraint evaluation, random-coefficient accumulation, `denom_inv` multiply, and
/// coordinate stores, one thread per row.
pub fn compile_v1_to_cuda_source(program: &OwnedMetalEvaluationProgramV1) -> Option<String> {
    let header = program.header();
    let name = fused_kernel_name(header.semantic_hash);
    let mut src = String::with_capacity(16384);
    emit_preamble(&mut src);

    src.push_str(&format!(
        "extern \"C\" __global__ void __launch_bounds__(128) {name}(\n\
         \x20   const unsigned *const *trace_cols,\n\
         \x20   const unsigned *interaction_offsets,\n\
         \x20   const unsigned *base_params,\n\
         \x20   const unsigned *ext_params,\n\
         \x20   const unsigned *random_coeff_powers,\n\
         \x20   const unsigned *denom_inv,\n\
         \x20   unsigned *coord_0,\n\
         \x20   unsigned *coord_1,\n\
         \x20   unsigned *coord_2,\n\
         \x20   unsigned *coord_3,\n\
         \x20   unsigned row_count,\n\
         \x20   unsigned log_n_rows\n\
         ) {{\n\
         \x20   unsigned row_index = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (row_index >= row_count) {{ return; }}\n\n"
    ));

    emit_instruction_body(program, &mut src)?;

    src.push_str("    // Fused denom_inv multiply + coordinate stores.\n");
    src.push_str("    unsigned denom_idx = row_index >> log_n_rows;\n");
    src.push_str("    StwoCudaQm31 result = stwo_qm31_mul_base(acc, denom_inv[denom_idx]);\n");
    src.push_str("    coord_0[row_index] = result.a;\n");
    src.push_str("    coord_1[row_index] = result.b;\n");
    src.push_str("    coord_2[row_index] = result.c;\n");
    src.push_str("    coord_3[row_index] = result.d;\n");
    src.push_str("}\n");
    Some(src)
}

fn emit_instruction_body(program: &OwnedMetalEvaluationProgramV1, src: &mut String) -> Option<()> {
    let header = program.header();
    let mut base_declared = vec![false; header.max_base_regs as usize];
    let mut ext_declared = vec![false; header.max_ext_regs as usize];

    src.push_str("    // Base instructions.\n");
    for inst in program.base_insts() {
        let dst = inst.dst as usize;
        let decl = if !base_declared[dst] {
            base_declared[dst] = true;
            "unsigned "
        } else {
            ""
        };
        let dst_var = format!("b{dst}");
        match BaseOp::from_raw(inst.op)? {
            BaseOp::TraceCol => {
                let (interaction, column, offset) = (inst.interaction, inst.a, inst.imm);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_trace_value(trace_cols, \
                     interaction_offsets, row_count, {interaction}u, {column}u, row_index, \
                     {offset});\n"
                ));
            }
            // The recorder routes preprocessed columns through TraceCol interaction 0;
            // a program carrying this opcode came from another lowering path.
            BaseOp::PreprocessedCol => return None,
            BaseOp::Param => {
                let slot = inst.a;
                src.push_str(&format!("    {decl}{dst_var} = base_params[{slot}u];\n"));
            }
            BaseOp::Const => {
                let value = inst.a;
                src.push_str(&format!("    {decl}{dst_var} = {value}u;\n"));
            }
            BaseOp::Add => {
                let (a, b) = (inst.a, inst.b);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_m31_add(b{a}, b{b});\n"
                ));
            }
            BaseOp::Sub => {
                let (a, b) = (inst.a, inst.b);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_m31_sub(b{a}, b{b});\n"
                ));
            }
            BaseOp::Mul => {
                let (a, b) = (inst.a, inst.b);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_m31_mul(b{a}, b{b});\n"
                ));
            }
            BaseOp::Neg => {
                let a = inst.a;
                src.push_str(&format!("    {decl}{dst_var} = stwo_m31_neg(b{a});\n"));
            }
            BaseOp::Inv => {
                let a = inst.a;
                src.push_str(&format!("    {decl}{dst_var} = stwo_m31_inv(b{a});\n"));
            }
        }
    }
    src.push('\n');

    src.push_str("    // Ext instructions.\n");
    for inst in program.ext_insts() {
        let dst = inst.dst as usize;
        let decl = if !ext_declared[dst] {
            ext_declared[dst] = true;
            "StwoCudaQm31 "
        } else {
            ""
        };
        let dst_var = format!("e{dst}");
        match ExtOp::from_raw(inst.op)? {
            ExtOp::SecureCol => {
                let (a, b, c, d) = (inst.a, inst.b, inst.c, inst.d);
                src.push_str(&format!(
                    "    {decl}{dst_var} = StwoCudaQm31{{ b{a}, b{b}, b{c}, b{d} }};\n"
                ));
            }
            ExtOp::Param => {
                let slot = inst.a;
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_load_qm31(ext_params, {slot}u);\n"
                ));
            }
            ExtOp::Const => {
                let (a, b, c, d) = (inst.a, inst.b, inst.c, inst.d);
                src.push_str(&format!(
                    "    {decl}{dst_var} = StwoCudaQm31{{ {a}u, {b}u, {c}u, {d}u }};\n"
                ));
            }
            ExtOp::Add => {
                let (a, b) = (inst.a, inst.b);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_qm31_add(e{a}, e{b});\n"
                ));
            }
            ExtOp::Sub => {
                let (a, b) = (inst.a, inst.b);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_qm31_sub(e{a}, e{b});\n"
                ));
            }
            ExtOp::Mul => {
                let (a, b) = (inst.a, inst.b);
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_qm31_mul(e{a}, e{b});\n"
                ));
            }
            ExtOp::Neg => {
                let a = inst.a;
                src.push_str(&format!(
                    "    {decl}{dst_var} = stwo_qm31_sub(StwoCudaQm31{{0u,0u,0u,0u}}, e{a});\n"
                ));
            }
        }
    }
    src.push('\n');

    src.push_str("    // Constraint accumulation.\n");
    src.push_str("    StwoCudaQm31 acc = StwoCudaQm31{0u, 0u, 0u, 0u};\n");
    for (i, &root) in program.constraint_roots().iter().enumerate() {
        src.push_str(&format!(
            "    acc = stwo_qm31_add(acc, stwo_qm31_mul(e{root}, \
             stwo_load_qm31(random_coeff_powers, {i}u)));\n"
        ));
    }
    src.push('\n');
    Some(())
}

fn emit_preamble(src: &mut String) {
    // Same formulas as the Metal preamble (byte-equal with the CPU reference via the
    // Metal conformance gate); CUDA syntax.
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

",
    );
}
