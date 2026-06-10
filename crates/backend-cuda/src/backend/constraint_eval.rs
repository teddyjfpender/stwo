//! Per-component GPU constraint evaluation (NitrooZK lineage).
//!
//! ~100 precompiled per-component kernels (generated from stwo-cairo's AIR) are
//! dispatched by an FNV1a hash of the component's name; components without a kernel
//! fall back to the generic CPU lane *on the same accumulator claim* (claiming twice
//! would consume two random-coefficient ranges). The kernels read the component's
//! `FrameworkEval` struct through a raw pointer, so their generated field layout must
//! match this build's — the stwo-cairo e2e byte-equality gate is the arbiter for
//! every component.
//!
//! `STWO_CUDA_DISABLE_CONSTRAINT_KERNELS=1` forces the CPU lane;
//! `STWO_CUDA_CONSTRAINT_LOG=1` logs the lane taken per component.

use stwo::core::air::Component;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{DomainEvaluationAccumulator, Trace};
use stwo_constraint_framework::{
    accumulate_pointwise_cpu, constraint_quotient_inputs, ConstraintQuotientInputs,
    FrameworkComponent, FrameworkEval,
};

use super::CudaBackend;
use crate::columns::{BaseFieldVec, SecureFieldVec};

/// FNV1a-32 over the component name; must match the dispatch table in
/// `evaluate_constraints.cu`.
fn fnv1a_eval_id(name: &str) -> u32 {
    const FNV_OFFSET_BASIS: u32 = 0x811C_9DC5;
    const FNV_PRIME: u32 = 0x0100_0193;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in name.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The kernel-dispatch name for a component: the module segment preceding the type
/// name (e.g. `cairo_air::components::add_opcode::Eval` -> `add_opcode`), matching the
/// names stwo-cairo components carry in the NitrooZK dispatch table. Components whose
/// derived name has no kernel simply fall back to the CPU lane.
fn derived_eval_name<E>() -> &'static str {
    let full = core::any::type_name::<E>();
    let full = full.split('<').next().unwrap_or(full);
    let mut segments = full.rsplit("::");
    let _type_name = segments.next();
    segments.next().unwrap_or(full)
}

/// Evaluate a component's constraint quotients: per-component GPU kernel when one
/// exists, CPU pointwise evaluation otherwise — both on a single accumulator claim,
/// mirroring `evaluate_constraint_quotients_via_cpu` exactly.
pub fn evaluate_constraint_quotients<E: FrameworkEval + Sync>(
    component: &FrameworkComponent<E>,
    trace: &Trace<'_, CudaBackend>,
    evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
) {
    if component.n_constraints() == 0 {
        return;
    }

    let ConstraintQuotientInputs {
        eval_domain,
        trace_domain,
        trace,
        denom_inv,
    } = constraint_quotient_inputs(component, trace, evaluation_accumulator.evaluation_mode());

    let [mut accum] =
        evaluation_accumulator.columns([(eval_domain.log_size(), component.n_constraints())]);
    accum.random_coeff_powers.reverse();

    let eval_name = derived_eval_name::<E>();
    let log = std::env::var_os("STWO_CUDA_CONSTRAINT_LOG").is_some();
    // Lane control without rebuilds: DISABLE wins; otherwise an ALLOWLIST (comma-
    // separated component names) restricts the GPU lane to listed components — used to
    // bisect kernels whose generated constraints don't match this AIR revision.
    // OPT-IN: the ported kernel set was generated against NitrooZK's stwo v2.1.1 +
    // their stwo-cairo AIR rev. Differential verification against this stack showed
    // 100%-of-rows mismatches from row 0 on every component (the eval-struct
    // layout/AIR-rev skew fingerprint), so no kernel is qualified by default.
    // Qualification path for regenerated kernels: STWO_CUDA_CONSTRAINT_VERIFY=1 with
    // an ALLOWLIST; components reporting 0 mismatches may be promoted.
    let gpu_enabled = std::env::var_os("STWO_CUDA_DISABLE_CONSTRAINT_KERNELS").is_none()
        && match std::env::var("STWO_CUDA_CONSTRAINT_ALLOWLIST") {
            Ok(list) => list.split(',').any(|name| name.trim() == eval_name),
            Err(_) => std::env::var_os("STWO_CUDA_ENABLE_CONSTRAINT_KERNELS").is_some(),
        };

    // Common GPU marshaling, shared by the precompiled and JIT lanes.
    let gpu_denom_inv = BaseFieldVec::from_vec(denom_inv.clone());
    let random_coeff_powers = SecureFieldVec::from_vec(accum.random_coeff_powers.clone());
    let trace_ptrs: Vec<Vec<*const u32>> = (0..3)
        .map(|interaction| {
            trace
                .get(interaction)
                .map(|columns| {
                    columns
                        .iter()
                        .map(|column| column.values.device_ptr)
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    let trace_column_lens: Vec<Vec<usize>> = (0..3)
        .map(|interaction| {
            trace
                .get(interaction)
                .map(|columns| columns.iter().map(|column| column.values.len()).collect())
                .unwrap_or_default()
        })
        .collect();

    if gpu_enabled {
        // The dispatcher reads a 4-byte FNV1a id, then the raw eval struct (the
        // generated kernel code mirrors the Rust field layout of its component).
        let eval_id = fnv1a_eval_id(eval_name);
        let eval_bytes = unsafe {
            std::slice::from_raw_parts(
                component.evaluator() as *const E as *const u8,
                std::mem::size_of::<E>(),
            )
        };
        let mut eval_buffer = Vec::with_capacity(4 + eval_bytes.len());
        eval_buffer.extend_from_slice(&eval_id.to_ne_bytes());
        eval_buffer.extend_from_slice(eval_bytes);

        let trace_log_size = component.evaluator().log_size();
        let logup_counts =
            component.logup_counts().values().sum::<usize>() as u32 >> trace_log_size;
        let cumsum_shift =
            component.claimed_sum() / BaseField::from_u32_unchecked(1u32 << trace_log_size);

        let accum_prev_snapshot = if std::env::var_os("STWO_CUDA_CONSTRAINT_VERIFY").is_some() {
            SecureColumnByCoords {
                columns: accum.col.columns.each_ref().map(|column| column.to_cpu()),
            }
        } else {
            SecureColumnByCoords::zeros(0)
        };

        crate::columns::bindings::ensure_mem_pool_init();
        let handled = unsafe {
            stwo_backend_cuda_kernels::raw::evaluate_constraint_quotients_on_domain(
                accum.col.columns[0].device_ptr,
                accum.col.columns[1].device_ptr,
                accum.col.columns[2].device_ptr,
                accum.col.columns[3].device_ptr,
                trace_ptrs[0].as_ptr(),
                trace_ptrs[0].len() as u32,
                trace_ptrs[1].as_ptr(),
                trace_ptrs[1].len() as u32,
                trace_ptrs[2].as_ptr(),
                trace_ptrs[2].len() as u32,
                random_coeff_powers.device_ptr,
                gpu_denom_inv.device_ptr,
                trace_domain.log_size(),
                eval_domain.log_size(),
                component.n_constraints() as u32,
                logup_counts,
                eval_buffer.as_mut_ptr().cast(),
                {
                    let limbs = cumsum_shift.to_m31_array();
                    stwo_backend_cuda_kernels::raw::CudaSecureField {
                        a: limbs[0].0,
                        b: limbs[1].0,
                        c: limbs[2].0,
                        d: limbs[3].0,
                    }
                },
                true,  // should_accumulate: proving accumulates into the column
                false, // use_assert_evaluator: production
            )
        };
        if log {
            eprintln!(
                "stwo-backend-cuda constraint eval: component={eval_name} id={eval_id:#x} lane={}",
                if handled { "GPU" } else { "CPU-fallback" }
            );
        }
        if handled {
            // Differential-verify mode: recompute on CPU from the pre-GPU snapshot and
            // compare, reporting the mismatch pattern. The CPU result is kept so the
            // prove stays correct while kernels are being qualified.
            if std::env::var_os("STWO_CUDA_CONSTRAINT_VERIFY").is_some() {
                let gpu_result: Vec<Vec<BaseField>> = accum
                    .col
                    .columns
                    .iter()
                    .map(|column| column.to_cpu())
                    .collect();
                let trace_cols_cpu = trace.as_cols_ref().map_cols(|column| {
                    CircleEvaluation::new(column.domain, column.values.to_cpu())
                });
                let cpu_result = accumulate_pointwise_cpu(
                    component,
                    trace_cols_cpu.as_cols_ref(),
                    eval_domain.log_size(),
                    trace_domain.log_size(),
                    denom_inv.clone(),
                    &accum.random_coeff_powers,
                    &accum_prev_snapshot,
                );
                let mut mismatches = 0usize;
                let mut first: Option<(usize, usize)> = None;
                for (coord, gpu_column) in gpu_result.iter().enumerate() {
                    for (row, (gpu, cpu)) in gpu_column
                        .iter()
                        .zip(cpu_result.columns[coord].iter())
                        .enumerate()
                    {
                        if gpu != cpu {
                            mismatches += 1;
                            if first.is_none() {
                                first = Some((coord, row));
                            }
                        }
                    }
                }
                let total = 4 * gpu_result[0].len();
                eprintln!(
                    "VERIFY component={eval_name}: {mismatches}/{total} mismatched, first={first:?}"
                );
                if mismatches > 0 {
                    *accum.col = SecureColumnByCoords {
                        columns: cpu_result
                            .columns
                            .map(|values| values.into_iter().collect()),
                    };
                }
            }
            return;
        }
    } else if log {
        eprintln!("stwo-backend-cuda constraint eval: component={eval_name} lane=CPU (disabled)");
    }

    // JIT lane: kernels generated from THIS build's AIR via the recording evaluator
    // (NVRTC, content-hash cached) — consistent by construction, explicit C ABI.
    if let Some(scratch) = super::jit::try_jit_constraint_quotients(
        component,
        &super::jit::JitInputs {
            trace_ptrs: &trace_ptrs,
            trace_column_lens: &trace_column_lens,
            random_coeff_powers: &random_coeff_powers,
            denom_inv: &gpu_denom_inv,
            accum_coords: [
                accum.col.columns[0].device_ptr.cast_mut(),
                accum.col.columns[1].device_ptr.cast_mut(),
                accum.col.columns[2].device_ptr.cast_mut(),
                accum.col.columns[3].device_ptr.cast_mut(),
            ],
            n_rows: eval_domain.size(),
            trace_log_size: trace_domain.log_size(),
        },
    ) {
        if log {
            eprintln!("stwo-backend-cuda constraint eval: component={eval_name} lane=JIT");
        }
        let verify = std::env::var_os("STWO_CUDA_CONSTRAINT_VERIFY").is_some();
        let accum_prev_snapshot = if verify {
            Some(SecureColumnByCoords {
                columns: accum.col.columns.each_ref().map(|column| column.to_cpu()),
            })
        } else {
            None
        };
        // accum += scratch (exact field arithmetic; same value as the CPU lane's
        // accum_prev + row_res * denom_inv).
        let jit_column = SecureColumnByCoords::<CudaBackend> { columns: scratch };
        <CudaBackend as stwo::prover::AccumulationOps>::accumulate(accum.col, &jit_column);

        if let Some(snapshot) = accum_prev_snapshot {
            let gpu_result: Vec<Vec<BaseField>> = accum
                .col
                .columns
                .iter()
                .map(|column| column.to_cpu())
                .collect();
            let trace_cols_cpu = trace
                .as_cols_ref()
                .map_cols(|column| CircleEvaluation::new(column.domain, column.values.to_cpu()));
            let cpu_result = accumulate_pointwise_cpu(
                component,
                trace_cols_cpu.as_cols_ref(),
                eval_domain.log_size(),
                trace_domain.log_size(),
                denom_inv.clone(),
                &accum.random_coeff_powers,
                &snapshot,
            );
            let mut mismatches = 0usize;
            let mut first: Option<(usize, usize)> = None;
            for (coord, gpu_column) in gpu_result.iter().enumerate() {
                for (row, (gpu, cpu)) in gpu_column
                    .iter()
                    .zip(cpu_result.columns[coord].iter())
                    .enumerate()
                {
                    if gpu != cpu {
                        mismatches += 1;
                        if first.is_none() {
                            first = Some((coord, row));
                        }
                    }
                }
            }
            let total = 4 * gpu_result[0].len();
            eprintln!(
                "VERIFY-JIT component={eval_name}: {mismatches}/{total} mismatched, first={first:?}"
            );
            if mismatches > 0 {
                *accum.col = SecureColumnByCoords {
                    columns: cpu_result
                        .columns
                        .map(|values| values.into_iter().collect()),
                };
            }
        }
        return;
    }
    if log {
        eprintln!("stwo-backend-cuda constraint eval: component={eval_name} lane=CPU");
    }

    // CPU fallback on the SAME accumulator claim, mirroring
    // `evaluate_constraint_quotients_via_cpu`'s body.
    let trace_cols_cpu = trace
        .as_cols_ref()
        .map_cols(|column| CircleEvaluation::new(column.domain, column.values.to_cpu()));
    let accum_prev_cpu = SecureColumnByCoords {
        columns: accum.col.columns.each_ref().map(|column| column.to_cpu()),
    };
    let result = accumulate_pointwise_cpu(
        component,
        trace_cols_cpu.as_cols_ref(),
        eval_domain.log_size(),
        trace_domain.log_size(),
        denom_inv,
        &accum.random_coeff_powers,
        &accum_prev_cpu,
    );
    *accum.col = SecureColumnByCoords {
        columns: result.columns.map(|values| values.into_iter().collect()),
    };
}
