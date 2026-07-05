//! AOT kernel-source emission surface (GPU_RESIDENT_PROVER_DESIGN.md §4, M3).
//!
//! The recording front-ends stay the source of truth; this module exposes them
//! to the `kernel_emit` tool so kernel SOURCES are generated at build time and
//! compiled offline (nvcc -O3, per-arch cubins embedded in the fatbin) instead
//! of at prove time (NVRTC + driver ptxas — the cold-start and sm_90-cliff
//! vehicle). The emitted source text is BYTE-IDENTICAL to what the JIT lane
//! would compile: same codegen, same cache keys — a prove-time cache-key lookup
//! that misses the AOT table simply falls back to NVRTC (the drift check).
//!
//! Constraint kernels are lowered UNCAPPED here (one fused kernel per
//! component): the 512/2048-instruction governor exists for load-time driver
//! ptxas, which the AOT path never runs. `force_relax` falsified -O0 at 2.6×
//! worse — AOT gives fused AND -O3, which no runtime option could.

use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::FrameworkEval;

use super::jit::{cuda_codegen, lower_for_aot as lower_framework_eval_to_v1_split};

/// One emitted kernel: `name`/`cache_key` are the launch-time lookup identity
/// (identical to the JIT lane's), `source` is the self-contained CUDA TU.
pub struct EmittedKernel {
    pub kernel_name: String,
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub source: String,
}

/// Emit the fused constraint kernel(s) for a component's evaluator. The lowering
/// hoists every statement constant into parameters, so the sources and cache
/// keys depend only on the AIR structure — any statement's evaluator emits the
/// same kernels. `max_kernel_instrs = usize::MAX` (the default) emits ONE fused
/// kernel unless a single constraint cone alone exceeds even that.
pub fn constraint_kernel_sources<F: FrameworkEval>(
    eval: &F,
    n_interactions: u32,
    claimed_sum: SecureField,
    log_size: u32,
    max_kernel_instrs: usize,
) -> Option<Vec<EmittedKernel>> {
    let (parts, _ext_params) = lower_framework_eval_to_v1_split(
        eval,
        n_interactions,
        0,
        0,
        claimed_sum,
        log_size,
        max_kernel_instrs,
    )
    .ok()?;
    parts
        .iter()
        .map(|part| {
            let semantic_hash = part.program.header().semantic_hash;
            let source = cuda_codegen::compile_v1_to_cuda_source(&part.program)?;
            Some(EmittedKernel {
                kernel_name: cuda_codegen::fused_kernel_name(semantic_hash),
                cache_key: cuda_codegen::jit_cache_key(semantic_hash),
                semantic_hash,
                source,
            })
        })
        .collect()
}

/// Emit a witness kernel from a recorded program (the lane's own codegen).
pub fn witness_kernel_source(
    program: &super::jit_witness::isa::WitnessProgram,
) -> Option<EmittedKernel> {
    let semantic_hash = program.semantic_hash();
    let source = super::jit_witness::codegen::compile_witness_to_cuda_source(program)?;
    Some(EmittedKernel {
        kernel_name: super::jit_witness::codegen::witness_kernel_name(semantic_hash),
        cache_key: super::jit_witness::codegen::witness_jit_cache_key(semantic_hash),
        semantic_hash,
        source,
    })
}
