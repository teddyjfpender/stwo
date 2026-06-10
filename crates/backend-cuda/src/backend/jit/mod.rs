//! JIT GPU constraint-evaluation lane for CUDA (NVRTC).
//!
//! Same architecture as the Metal JIT lane: the component's constraint tree is
//! recorded once to V1 bytecode (generic `EvalAtRow` recorder, logup included),
//! compiled to a fused CUDA kernel via NVRTC (cached by the bytecode's content
//! semantic hash), and evaluated in one dispatch. Kernels generated this way are
//! consistent with THIS build's AIR by construction and use an explicit C ABI —
//! the failure mode that disqualified the precompiled NitrooZK kernel set cannot
//! occur. Falls back to `false` (caller runs the CPU lane) on any failure.

mod cuda_codegen;
mod program;
mod recording;

use std::ffi::CString;

use program::lower_framework_eval_to_v1_with_logup;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval};

use crate::columns::{BaseFieldVec, SecureFieldVec};

/// Inputs prepared by the shared constraint-eval driver (single accumulator claim).
pub(crate) struct JitInputs<'a> {
    pub trace_ptrs: &'a [Vec<*const u32>],
    pub trace_column_lens: &'a [Vec<usize>],
    pub random_coeff_powers: &'a SecureFieldVec,
    pub denom_inv: &'a BaseFieldVec,
    pub accum_coords: [*mut u32; 4],
    pub n_rows: usize,
    pub trace_log_size: u32,
}

/// Record, compile (cached), and launch the fused kernel; the kernel adds
/// `sum_i rc[i]*constraint_i(row) * denom_inv[row >> trace_log]` directly into the
/// accumulator coordinates (`inputs.accum_coords`) in place — there is no separate
/// scratch buffer or accumulate pass. Returns `false` to request the CPU lane, in
/// which case the accumulator has NOT been touched.
pub(crate) fn try_jit_constraint_quotients<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    inputs: &JitInputs<'_>,
) -> bool {
    if std::env::var_os("STWO_CUDA_DISABLE_JIT").is_some() {
        return false;
    }
    try_jit_constraint_quotients_inner(component, inputs).is_some()
}

fn try_jit_constraint_quotients_inner<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    inputs: &JitInputs<'_>,
) -> Option<()> {
    // Lowering hoists every ext constant (lookup elements, cumsum shift) into
    // `ext_param_values`, so `program` — and its semantic hash — depends only on the
    // AIR structure. The hash therefore stays stable across statements and the
    // compiled kernel is reused from the in-memory or on-disk PTX cache.
    let (program, ext_param_values) = lower_framework_eval_to_v1_with_logup(
        component.evaluator(),
        inputs.trace_ptrs.len() as u32,
        0,
        0,
        component.claimed_sum(),
        component.evaluator().log_size(),
    )
    .ok()?;
    let source = cuda_codegen::compile_v1_to_cuda_source(&program)?;
    let kernel_name = cuda_codegen::fused_kernel_name(program.header().semantic_hash);

    let n_rows = inputs.n_rows;
    // Pointer-table trace ABI: the kernel indexes trace_cols[global_column][row], so
    // no flattening copies are needed (and no u32 length overflow at log >= 23 sizes;
    // each column pointer addresses its own buffer). In SubDomain mode columns are
    // longer than n_rows; the first n_rows bit-reversed entries are the evaluation
    // subdomain — the same prefix the CPU lane reads.
    let mut interaction_offsets = [0u32; 3];
    let mut all_column_ptrs: Vec<*const u32> = Vec::new();
    for (interaction, tree) in inputs.trace_ptrs.iter().enumerate() {
        interaction_offsets[interaction] = all_column_ptrs.len() as u32;
        for (column_idx, &src_ptr) in tree.iter().enumerate() {
            assert!(inputs.trace_column_lens[interaction][column_idx] >= n_rows);
            all_column_ptrs.push(src_ptr);
        }
    }
    let trace_table = crate::backend::UploadedDevicePointerVec::upload(&all_column_ptrs);
    let offsets_dev = BaseFieldVec::from_vec(
        interaction_offsets
            .iter()
            .map(|&v| stwo::core::fields::m31::BaseField::from_u32_unchecked(v))
            .collect(),
    );

    let source_c = CString::new(source).ok()?;
    let name_c = CString::new(kernel_name).ok()?;
    let empty = BaseFieldVec::new_zeroes(1);
    // Ext params = every constant the lowering hoisted out of the bytecode (lookup
    // elements, cumsum shift, structural constants), uploaded in slot order. A
    // one-element zero buffer keeps the ABI pointer valid for const-free programs.
    let ext_params = SecureFieldVec::from_vec(if ext_param_values.is_empty() {
        vec![num_traits::Zero::zero()]
    } else {
        ext_param_values
    });

    crate::columns::bindings::ensure_mem_pool_init();
    // The kernel accumulates in place into the accumulator coordinates; no scratch
    // columns and no separate accumulate dispatch (see cuda_codegen's fused store).
    let ok = unsafe {
        stwo_backend_cuda_kernels::raw::stwo_cuda_jit_eval_fused(
            source_c.as_ptr(),
            name_c.as_ptr(),
            cuda_codegen::jit_cache_key(program.header().semantic_hash),
            trace_table.as_ptr().cast(),
            offsets_dev.device_ptr,
            empty.device_ptr,
            ext_params.device_ptr,
            inputs.random_coeff_powers.device_ptr,
            inputs.denom_inv.device_ptr,
            inputs.accum_coords[0],
            inputs.accum_coords[1],
            inputs.accum_coords[2],
            inputs.accum_coords[3],
            n_rows as u32,
            inputs.trace_log_size,
        )
    };
    ok.then_some(())
}
