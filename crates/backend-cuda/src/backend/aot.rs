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
//! Constraint kernels use the same instruction and compacted-live-lane split
//! policy as the runtime lowerer. The exact policy is sealed into the pack
//! identity, while per-key strict lookup proves complete source/shape coverage;
//! a matching policy tag alone never admits a partial or stale pack.

/// Stable identity of the AOT semantic-key/architecture set embedded in this
/// binary. Zero means no AOT pack is present and is never a valid graph key.
pub fn loaded_manifest_hash() -> u64 {
    let hash = stwo_backend_cuda_kernels::aot_pack::aot_pack_manifest_hash();
    if hash != 0 && loaded_constraint_max_live_u32_lanes() != constraint_split_max_live_u32_lanes()
    {
        return 0;
    }
    hash
}

/// Exact constraint split cap used by the loaded AOT pack. Zero means the
/// current binary has no pack and is invalid for resident composition planning.
pub fn loaded_constraint_max_instrs() -> usize {
    stwo_backend_cuda_kernels::aot_pack::aot_pack_constraint_max_instrs()
}

/// Exact compacted live-u32-lane cap used by the loaded AOT pack. Zero means no
/// pack is present. [`loaded_manifest_hash`] rejects a stale pack whose value does
/// not match the runtime lowerer's compiled policy.
pub fn loaded_constraint_max_live_u32_lanes() -> usize {
    stwo_backend_cuda_kernels::aot_pack::aot_pack_constraint_max_live_u32_lanes()
}

/// Runtime/AOT constraint splitter register-pressure policy identity.
pub const fn constraint_split_max_live_u32_lanes() -> usize {
    super::jit::CONSTRAINT_SPLIT_MAX_LIVE_U32_LANES
}

/// Cheap device-architecture admission check. Individual semantic lookups still
/// fail closed in strict GPU-native mode, so a partial pack cannot masquerade as
/// complete merely because it contains one kernel for the device.
pub fn supports_arch(sm_major: u32, sm_minor: u32) -> bool {
    stwo_backend_cuda_kernels::aot_pack::aot_pack_supports_arch(sm_major, sm_minor)
}

/// Cheap read-only admission check for an exact embedded kernel. This searches
/// the sealed binary's static AOT index and does not initialize CUDA.
pub fn contains_loaded_kernel(cache_key: u64, sm_major: u32, sm_minor: u32) -> bool {
    stwo_backend_cuda_kernels::aot_pack::aot_pack_contains(cache_key, sm_major, sm_minor)
}

pub use stwo_backend_cuda_kernels::raw::CudaJitAotStats as RuntimeStats;

/// Permanently select the fail-closed AOT-only lane for generated kernels.
///
/// This closes runtime admission and waits for every previously admitted compile,
/// cache publication, and launch enqueue to leave its native operation scope before
/// returning. It does not synchronize completion of arbitrary GPU work that was already
/// queued, so calling it during prover construction, before proof work begins, is a
/// precondition.
pub fn require_loaded_kernels() {
    unsafe { stwo_backend_cuda_kernels::raw::stwo_cuda_jit_set_require_aot(true) }
}

pub fn runtime_stats() -> RuntimeStats {
    let mut stats = RuntimeStats::default();
    unsafe { stwo_backend_cuda_kernels::raw::stwo_cuda_jit_get_aot_stats(&mut stats) };
    stats
}

pub fn reset_runtime_stats() {
    unsafe { stwo_backend_cuda_kernels::raw::stwo_cuda_jit_reset_aot_stats() }
}

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::FrameworkEval;

use super::jit::{
    cuda_codegen, lower_for_aot as lower_framework_eval_to_v1_split,
    lower_for_aot_with_live_cap as lower_framework_eval_to_v1_split_with_live_cap,
};

/// One emitted kernel: `name`/`cache_key` are the launch-time lookup identity
/// (identical to the JIT lane's), `source` is the self-contained CUDA TU.
pub struct EmittedKernel {
    pub kernel_name: String,
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub source: String,
}

/// One split part of a prepared constraint program. `rc_base` is the exact
/// global offset into that component's random-coefficient slice used by the
/// generated kernel ABI.
pub struct EmittedConstraintKernel {
    pub kernel: EmittedKernel,
    pub rc_base: u32,
}

/// Structural constraint program plus the evaluator constants hoisted into its
/// mutable parameter tables. The kernel identities are statement independent;
/// the parameter values are the setup oracle used by higher layers to bind each
/// stable slot to its device-side statement producer.
pub struct EmittedConstraintProgram {
    pub kernels: Vec<EmittedConstraintKernel>,
    pub base_param_values: Vec<BaseField>,
    pub ext_param_values: Vec<SecureField>,
}

/// Source-free identity and proof-varying parameter values for one already
/// installed constraint program. Warm executables use this path to prove that
/// the current evaluator still lowers to the installed kernel set without
/// formatting or allocating CUDA source again.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintProgramBindings {
    pub kernels: Vec<ConstraintKernelBinding>,
    pub base_param_values: Vec<BaseField>,
    pub ext_param_values: Vec<SecureField>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintKernelBinding {
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub rc_base: u32,
}

/// Record and lower only far enough to bind an installed AOT program. Unlike
/// [`constraint_program`], this deliberately performs no CUDA code generation.
pub fn constraint_program_bindings<F: FrameworkEval>(
    eval: &F,
    n_interactions: u32,
    claimed_sum: SecureField,
    log_size: u32,
    max_kernel_instrs: usize,
) -> Option<ConstraintProgramBindings> {
    let (parts, base_param_values, ext_param_values) = lower_framework_eval_to_v1_split(
        eval,
        n_interactions,
        0,
        0,
        claimed_sum,
        log_size,
        max_kernel_instrs,
    )
    .ok()?;
    let kernels = parts
        .iter()
        .map(|part| {
            let semantic_hash = part.program.header().semantic_hash;
            ConstraintKernelBinding {
                cache_key: cuda_codegen::jit_cache_key(semantic_hash),
                semantic_hash,
                rc_base: part.rc_base,
            }
        })
        .collect();
    Some(ConstraintProgramBindings {
        kernels,
        base_param_values,
        ext_param_values,
    })
}

/// Emit the complete prepared-program description for one concrete component.
/// This is the source of truth shared by offline AOT generation and the resident
/// composition planner; neither side may reconstruct split offsets or parameter
/// ordering from the manifest filename.
pub fn constraint_program<F: FrameworkEval>(
    eval: &F,
    n_interactions: u32,
    claimed_sum: SecureField,
    log_size: u32,
    max_kernel_instrs: usize,
) -> Option<EmittedConstraintProgram> {
    constraint_program_with_live_cap(
        eval,
        n_interactions,
        claimed_sum,
        log_size,
        max_kernel_instrs,
        constraint_split_max_live_u32_lanes(),
    )
}

/// Explicit-policy source emitter for offline ptxas/occupancy sweeps. Production
/// generation uses [`constraint_program`] and the compiled policy identity.
pub fn constraint_program_with_live_cap<F: FrameworkEval>(
    eval: &F,
    n_interactions: u32,
    claimed_sum: SecureField,
    log_size: u32,
    max_kernel_instrs: usize,
    max_live_u32_lanes: usize,
) -> Option<EmittedConstraintProgram> {
    let (parts, base_param_values, ext_param_values) =
        lower_framework_eval_to_v1_split_with_live_cap(
            eval,
            n_interactions,
            0,
            0,
            claimed_sum,
            log_size,
            max_kernel_instrs,
            max_live_u32_lanes,
        )
        .ok()?;
    let kernels = parts
        .iter()
        .map(|part| {
            let semantic_hash = part.program.header().semantic_hash;
            let source = cuda_codegen::compile_v1_to_cuda_source(&part.program)?;
            Some(EmittedConstraintKernel {
                kernel: EmittedKernel {
                    kernel_name: cuda_codegen::fused_kernel_name(semantic_hash),
                    cache_key: cuda_codegen::jit_cache_key(semantic_hash),
                    semantic_hash,
                    source,
                },
                rc_base: part.rc_base,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(EmittedConstraintProgram {
        kernels,
        base_param_values,
        ext_param_values,
    })
}

/// Emit the fused constraint kernel(s) for a component's evaluator. The lowering
/// hoists every statement constant into parameters, so structurally identical
/// evaluator recordings emit the same sources and cache keys across statement
/// values. `max_kernel_instrs = usize::MAX` (the default) emits ONE fused
/// kernel unless a single constraint cone alone exceeds even that.
pub fn constraint_kernel_sources<F: FrameworkEval>(
    eval: &F,
    n_interactions: u32,
    claimed_sum: SecureField,
    log_size: u32,
    max_kernel_instrs: usize,
) -> Option<Vec<EmittedKernel>> {
    constraint_kernel_sources_with_live_cap(
        eval,
        n_interactions,
        claimed_sum,
        log_size,
        max_kernel_instrs,
        constraint_split_max_live_u32_lanes(),
    )
}

/// Explicit-policy source-only wrapper for offline resource sweeps.
pub fn constraint_kernel_sources_with_live_cap<F: FrameworkEval>(
    eval: &F,
    n_interactions: u32,
    claimed_sum: SecureField,
    log_size: u32,
    max_kernel_instrs: usize,
    max_live_u32_lanes: usize,
) -> Option<Vec<EmittedKernel>> {
    Some(
        constraint_program_with_live_cap(
            eval,
            n_interactions,
            claimed_sum,
            log_size,
            max_kernel_instrs,
            max_live_u32_lanes,
        )?
        .kernels
        .into_iter()
        .map(|part| part.kernel)
        .collect(),
    )
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

/// Source-free identity of one kernel in a canonical two-phase witness plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessPhaseKernelBinding {
    pub ordinal: u32,
    pub kernel_name: String,
    pub cache_key: u64,
}

/// Exact runtime binding for a canonical two-phase witness plan.
///
/// This deliberately carries no CUDA source: the prepared phase runtime is
/// strict-AOT-only and must resolve both cache keys from the embedded pack
/// before either phase can be launched.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessPhaseProgramBindings {
    pub parent_semantic_hash: u64,
    pub plan_hash: u64,
    pub scratch_words_per_row: u32,
    pub phases: [WitnessPhaseKernelBinding; 2],
}

/// Bind a plan only when it is the exact canonical plan for this program and
/// cut. Forged or stale hashes, boundary sources, and moved-store schedules all
/// fail before an AOT lookup or device side effect.
pub fn witness_phase_program_bindings(
    program: &super::jit_witness::isa::WitnessProgram,
    plan: &super::jit_witness::codegen::phase_plan::WitnessPhasePlan,
) -> Option<WitnessPhaseProgramBindings> {
    let canonical = super::jit_witness::codegen::phase_plan::WitnessPhasePlan::at_cut(
        program,
        plan.cut_instruction,
    )
    .ok()?;
    if canonical != *plan {
        return None;
    }
    let phases = std::array::from_fn(|ordinal| {
        let ordinal = ordinal as u32;
        WitnessPhaseKernelBinding {
            ordinal,
            kernel_name: plan.phase_kernel_name(ordinal),
            cache_key: plan.phase_cache_key(ordinal),
        }
    });
    Some(WitnessPhaseProgramBindings {
        parent_semantic_hash: plan.parent_semantic_hash,
        plan_hash: plan.plan_hash,
        scratch_words_per_row: plan.scratch_words_per_row,
        phases,
    })
}

#[cfg(test)]
mod tests {
    use super::super::jit_witness::codegen::phase_plan::WitnessPhasePlan;
    use super::super::jit_witness::recording::WitnessRecorder;
    use super::*;

    #[derive(Default)]
    struct AdmissionModel {
        active: usize,
        admitted: bool,
        closed: bool,
        runtime_resolved: bool,
        setter_returned: bool,
        side_effect_after_commit: bool,
    }

    impl AdmissionModel {
        fn enter_runtime_operation(&mut self) {
            self.admitted = !self.closed;
            self.active += usize::from(self.admitted);
        }

        fn resolve_cached_runtime_function(&mut self) {
            self.runtime_resolved = !self.closed;
        }

        fn publish_or_enqueue(&mut self) {
            if self.runtime_resolved {
                self.side_effect_after_commit |= self.setter_returned;
            }
        }

        fn leave_runtime_operation(&mut self) {
            self.active -= usize::from(self.admitted);
            self.try_commit();
        }

        fn close_strict_admission(&mut self) {
            self.closed = true;
            self.try_commit();
        }

        fn try_commit(&mut self) {
            self.setter_returned |= self.closed && self.active == 0;
        }
    }

    #[test]
    fn strict_commit_cannot_be_crossed_by_runtime_publication_or_launch() {
        // Exhaust every linearization point for closure around a cached-runtime
        // operation: enter, resolve, publish/enqueue, leave. Closing before resolve
        // rejects the runtime origin; closing later waits for the operation guard.
        for close_before_step in 0..=4 {
            let mut model = AdmissionModel::default();
            for step in 0..4 {
                if close_before_step == step {
                    model.close_strict_admission();
                }
                match step {
                    0 => model.enter_runtime_operation(),
                    1 => model.resolve_cached_runtime_function(),
                    2 => model.publish_or_enqueue(),
                    3 => model.leave_runtime_operation(),
                    _ => unreachable!(),
                }
            }
            if close_before_step == 4 {
                model.close_strict_admission();
            }
            assert!(model.setter_returned);
            assert!(
                !model.side_effect_after_commit,
                "closure step {close_before_step}"
            );
        }
    }

    #[test]
    fn phase_bindings_are_canonical_source_free_identities() {
        let mut recorder = WitnessRecorder::new("aot_phase_bindings");
        let input = recorder.input(0);
        let constant = recorder.constant(7);
        let crossing = recorder.m31_add(input, constant);
        let output = recorder.m31_mul(crossing, input);
        recorder.col_write(0, output);
        let program = recorder.finish();
        let plan = WitnessPhasePlan::at_cut(&program, 3).unwrap();

        let bindings = witness_phase_program_bindings(&program, &plan).unwrap();
        assert_eq!(bindings.parent_semantic_hash, program.semantic_hash());
        assert_eq!(bindings.plan_hash, plan.plan_hash);
        assert_eq!(bindings.scratch_words_per_row, plan.scratch_words_per_row);
        for (ordinal, phase) in bindings.phases.iter().enumerate() {
            let ordinal = ordinal as u32;
            assert_eq!(phase.ordinal, ordinal);
            assert_eq!(phase.kernel_name, plan.phase_kernel_name(ordinal));
            assert_eq!(phase.cache_key, plan.phase_cache_key(ordinal));
        }

        let mut forged = plan.clone();
        forged.plan_hash ^= 1;
        assert!(witness_phase_program_bindings(&program, &forged).is_none());
    }
}
