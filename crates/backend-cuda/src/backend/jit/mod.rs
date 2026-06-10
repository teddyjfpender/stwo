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

/// Record, compile (cached), and launch the fused kernel; the kernel writes
/// `sum_i rc[i]*constraint_i(row) * denom_inv[row >> trace_log]` into scratch and the
/// caller adds it into the accumulator. Returns false to request the CPU lane.
pub(crate) fn try_jit_constraint_quotients<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    inputs: &JitInputs<'_>,
) -> Option<[BaseFieldVec; 4]> {
    if std::env::var_os("STWO_CUDA_DISABLE_JIT").is_some() {
        return None;
    }

    let program = lower_framework_eval_to_v1_with_logup(
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
    // Flatten the trace into one column-major device buffer (n_rows per column; in
    // SubDomain mode columns are longer and the first n_rows bit-reversed entries are
    // exactly the evaluation subdomain — same prefix the CPU lane reads).
    let n_columns: usize = inputs.trace_ptrs.iter().map(|tree| tree.len()).sum();
    let flat = BaseFieldVec::new_uninitialized(n_columns * n_rows);
    let mut interaction_offsets = [0u32; 3];
    let mut next_column = 0usize;
    for (interaction, tree) in inputs.trace_ptrs.iter().enumerate() {
        interaction_offsets[interaction] = next_column as u32;
        for (column_idx, &src_ptr) in tree.iter().enumerate() {
            assert!(inputs.trace_column_lens[interaction][column_idx] >= n_rows);
            unsafe {
                stwo_backend_cuda_kernels::raw::copy_uint32_t_vec_from_device_to_device(
                    src_ptr,
                    flat.device_ptr.add(next_column * n_rows).cast_mut(),
                    n_rows as u32,
                );
            }
            next_column += 1;
        }
    }
    let offsets_dev = BaseFieldVec::from_vec(
        interaction_offsets
            .iter()
            .map(|&v| stwo::core::fields::m31::BaseField::from_u32_unchecked(v))
            .collect(),
    );

    let scratch: [BaseFieldVec; 4] = std::array::from_fn(|_| BaseFieldVec::new_zeroes(n_rows));
    let source_c = CString::new(source).ok()?;
    let name_c = CString::new(kernel_name).ok()?;
    let empty = BaseFieldVec::new_zeroes(1);

    crate::columns::bindings::ensure_mem_pool_init();
    let ok = unsafe {
        stwo_backend_cuda_kernels::raw::stwo_cuda_jit_eval_fused(
            source_c.as_ptr(),
            name_c.as_ptr(),
            program.header().semantic_hash,
            flat.device_ptr,
            offsets_dev.device_ptr,
            empty.device_ptr,
            empty.device_ptr,
            inputs.random_coeff_powers.device_ptr,
            inputs.denom_inv.device_ptr,
            scratch[0].device_ptr.cast_mut(),
            scratch[1].device_ptr.cast_mut(),
            scratch[2].device_ptr.cast_mut(),
            scratch[3].device_ptr.cast_mut(),
            n_rows as u32,
            inputs.trace_log_size,
        )
    };
    let _ = inputs.accum_coords;
    ok.then_some(scratch)
}
