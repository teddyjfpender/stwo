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

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex, OnceLock};

use program::lower_framework_eval_to_v1_with_logup;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval};

use crate::columns::{BaseFieldVec, SecureFieldVec};

/// The structural codegen output for one component kernel: the CUDA source and
/// kernel name (as ready-to-pass `CString`s) plus the PTX-cache key. These depend
/// ONLY on the AIR structure (component type + log size + n_interactions), never on
/// the statement, so they are computed once and reused across every prove.
struct CachedCodegen {
    source_c: CString,
    name_c: CString,
    cache_key: u64,
}

/// Process-global codegen cache, keyed by (component type name, log size).
/// `lower_framework_eval_to_v1_with_logup` (symbolic recording) and
/// `compile_v1_to_cuda_source` (CUDA string emission) are single-threaded and, for
/// constraint-heavy components, expensive; caching their structural result here
/// makes the prelude skip them entirely on a hit and the eval lane skip the source
/// emission (it must still lower to obtain the per-prove ext params).
fn codegen_cache() -> &'static Mutex<HashMap<(&'static str, u32), Arc<CachedCodegen>>> {
    static CACHE: OnceLock<Mutex<HashMap<(&'static str, u32), Arc<CachedCodegen>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Cache lookup by structural key (no lowering).
fn cached_codegen(eval_name: &'static str, log_size: u32) -> Option<Arc<CachedCodegen>> {
    codegen_cache()
        .lock()
        .unwrap()
        .get(&(eval_name, log_size))
        .cloned()
}

/// Emit the CUDA source for an already-lowered program, build the `CString`s + cache
/// key, and store them in the codegen cache. Returns `None` if source emission fails.
fn build_and_cache_codegen(
    eval_name: &'static str,
    log_size: u32,
    program: &program::OwnedMetalEvaluationProgramV1,
) -> Option<Arc<CachedCodegen>> {
    let source = cuda_codegen::compile_v1_to_cuda_source(program)?;
    let kernel_name = cuda_codegen::fused_kernel_name(program.header().semantic_hash);
    let cached = Arc::new(CachedCodegen {
        source_c: CString::new(source).ok()?,
        name_c: CString::new(kernel_name).ok()?,
        cache_key: cuda_codegen::jit_cache_key(program.header().semantic_hash),
    });
    codegen_cache()
        .lock()
        .unwrap()
        .insert((eval_name, log_size), cached.clone());
    Some(cached)
}

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
    // Per-component opt-out: comma-separated component names routed to the CPU
    // constraint lane instead of JIT. Skips the few pathologically-large kernels
    // (e.g. partial_ec_mul_generic, whose EC-ladder constraint program takes NVRTC
    // many minutes to compile) when their instance count is tiny enough that CPU
    // eval is cheap — avoids the one-time compile on a cold PTX cache.
    if let Ok(skip) = std::env::var("STWO_CUDA_JIT_SKIP") {
        let name = super::constraint_eval::derived_eval_name::<E>();
        if skip.split(',').any(|s| s.trim() == name) {
            return false;
        }
    }
    try_jit_constraint_quotients_inner(component, inputs).is_some()
}

/// Compile (and disk/in-memory cache) this component's fused constraint kernel
/// WITHOUT evaluating — no trace, no launch. Called by the parallel pre-compile
/// pass to warm the cache before the sequential composition loop, so the per-row
/// cold-start NVRTC cost (minutes for the EC/Pedersen/Poseidon monster kernels)
/// is paid once, concurrently across components, instead of serially on the
/// critical path. Safe and idempotent: same content hash as the lazy eval path
/// (so it is a cache hit there), best-effort (failures just leave the lazy path
/// to compile), honors STWO_CUDA_DISABLE_JIT and STWO_CUDA_JIT_SKIP.
pub(crate) fn precompile_prepare<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
) -> Option<Box<dyn FnOnce() + Send>> {
    // Opt-in (default OFF). The prelude parallelizes the NVRTC *compile*, but the
    // per-component lowering/codegen it must run first is itself single-threaded
    // and, for constraint-heavy components (partial_ec_mul / pedersen / poseidon),
    // expensive — and the lazy eval lane re-does that codegen, so an always-on
    // prelude DOUBLES codegen on the warm path. Until the lowered program is cached
    // (the real fix), keep this opt-in so the default path is unregressed; the
    // pre-baked PTX seed is the cold-start mitigation meanwhile.
    if std::env::var_os("STWO_CUDA_PARALLEL_JIT_WARMUP").is_none() {
        return None;
    }
    if std::env::var_os("STWO_CUDA_DISABLE_JIT").is_some() {
        return None;
    }
    let eval_name = super::constraint_eval::derived_eval_name::<E>();
    if let Ok(skip) = std::env::var("STWO_CUDA_JIT_SKIP") {
        if skip.split(',').any(|s| s.trim() == eval_name) {
            return None;
        }
    }
    let log_size = component.evaluator().log_size();
    // Codegen-once: a hit means another component instance (or a prior prove) already
    // lowered + emitted this kernel's source. The prelude only needs to COMPILE, so on
    // a hit we skip lowering and codegen entirely — the expensive single-threaded step
    // the always-on prelude used to double. n_interactions = 3 matches the eval path's
    // `inputs.trace_ptrs.len()` (the (0..3) preprocessed/base/interaction trees), so the
    // semantic hash — and thus the PTX cache key — is identical to what the lazy lane
    // computes; the warmed kernel is reused there.
    let cached = match cached_codegen(eval_name, log_size) {
        Some(cached) => cached,
        None => {
            let (program, _ext_param_values) = lower_framework_eval_to_v1_with_logup(
                component.evaluator(),
                3,
                0,
                0,
                component.claimed_sum(),
                log_size,
            )
            .ok()?;
            build_and_cache_codegen(eval_name, log_size, &program)?
        }
    };
    // The compile closure captures only the `Arc<CachedCodegen>` (CStrings + key are
    // `Send + Sync`), so it runs safely on a parallel worker thread.
    Some(Box::new(move || {
        crate::columns::bindings::ensure_mem_pool_init();
        unsafe {
            stwo_backend_cuda_kernels::raw::stwo_cuda_jit_compile(
                cached.source_c.as_ptr(),
                cached.name_c.as_ptr(),
                cached.cache_key,
            );
        }
    }))
}

fn try_jit_constraint_quotients_inner<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    inputs: &JitInputs<'_>,
) -> Option<()> {
    let jit_log = std::env::var_os("STWO_JIT_LOG").is_some();
    let eval_name = super::constraint_eval::derived_eval_name::<E>();
    let log_size = component.evaluator().log_size();

    // Lowering hoists every ext constant (lookup elements, cumsum shift) into
    // `ext_param_values`, so `program` — and its semantic hash — depends only on the
    // AIR structure. The hash therefore stays stable across statements and the
    // compiled kernel is reused from the in-memory or on-disk PTX cache. The ext
    // constants are channel-drawn (per-prove) and const-folded during recording, so
    // we MUST re-lower every prove to recover `ext_param_values` — but the heavy
    // CUDA source emission below depends only on structure and is cached.
    let lower_t = std::time::Instant::now();
    let (program, ext_param_values) = lower_framework_eval_to_v1_with_logup(
        component.evaluator(),
        inputs.trace_ptrs.len() as u32,
        0,
        0,
        component.claimed_sum(),
        log_size,
    )
    .ok()?;
    let lower_ms = lower_t.elapsed().as_millis();

    // Codegen-once: reuse the cached CUDA source/name/key on a hit, skipping the
    // O(program-size) `compile_v1_to_cuda_source` string emission. The reuse is
    // guarded by the freshly-lowered program's semantic hash: we only adopt a cached
    // entry whose `cache_key` matches this prove's program, so a `(eval_name,
    // log_size)` key collision can never feed a mismatched kernel to the dispatch
    // (the semantic hash, not the name, is the source of truth). The FFI below always
    // receives `cached.cache_key`, which equals this program's real key in both arms.
    let real_key = cuda_codegen::jit_cache_key(program.header().semantic_hash);
    let gen_t = std::time::Instant::now();
    let (cached, source_hit) = match cached_codegen(eval_name, log_size) {
        Some(cached) if cached.cache_key == real_key => (cached, true),
        _ => (build_and_cache_codegen(eval_name, log_size, &program)?, false),
    };
    let gen_ms = gen_t.elapsed().as_millis();
    if jit_log {
        eprintln!(
            "[stwo-jit] {eval_name} log_size={log_size} lower={lower_ms}ms \
             source_gen={gen_ms}ms ({})",
            if source_hit { "cache-hit" } else { "emitted" },
        );
    }

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
            cached.source_c.as_ptr(),
            cached.name_c.as_ptr(),
            cached.cache_key,
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
