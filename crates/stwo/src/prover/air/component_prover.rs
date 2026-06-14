use dashmap::DashMap;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::air::{Component, Components};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleDomain;
use crate::core::ColumnVec;
use crate::prover::air::accumulation::{DomainEvaluationAccumulator, EvaluationMode};
use crate::prover::backend::{Backend, Col};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, SecureCirclePoly};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::CirclePoint;

/// Type alias for the weights hash map used in barycentric eval_at_point.
pub type WeightsHashMap<B> = DashMap<(u32, CirclePoint<SecureField>), Col<B, SecureField>>;

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );

    /// Optionally prepare a backend constraint-kernel compile for this component:
    /// do the (cheap, main-thread) lowering/codegen now and return a `Send`
    /// closure that performs the (expensive) compile when run. Returns `None`
    /// when there is nothing to precompile (default, and for backends without a
    /// JIT lane). The returned closures are run in parallel before the sequential
    /// constraint-evaluation loop, so the one-time per-AIR kernel compile is paid
    /// once and concurrently rather than serially on the critical path. The split
    /// (lower here, compile in the closure) keeps `!Sync` component state on the
    /// calling thread while only owned/`Send` data crosses to the worker threads.
    /// Best-effort: must never affect correctness — the eval path compiles lazily
    /// if this did nothing.
    fn precompile_prepare(&self) -> Option<Box<dyn FnOnce() + Send>> {
        None
    }
}

/// The set of polynomials that make up the trace.
pub struct Trace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: TreeVec<ColumnVec<&'a Poly<B>>>,
}

/// A struct for representing a polynomial corresponding to a trace column.
/// A polynomial is defined by it's evaluations on a circle domain of size at least it's degree,
/// and optionally its coefficients in the FFT basis.
pub struct Poly<B: Backend> {
    pub coeffs: Option<CircleCoefficients<B>>,
    pub evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
}

impl<B: Backend> Poly<B> {
    pub const fn new(
        coeffs: Option<CircleCoefficients<B>>,
        evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
    ) -> Self {
        Self { coeffs, evals }
    }

    pub fn eval_at_point(
        &self,
        point: CirclePoint<SecureField>,
        weights_hash_map: Option<&WeightsHashMap<B>>,
    ) -> SecureField {
        if let Some(coeffs) = &self.coeffs {
            coeffs.eval_at_point(point)
        } else {
            self.evals.barycentric_eval_at_point(
                &weights_hash_map
                    .unwrap()
                    .get(&(self.evals.domain.log_size(), point))
                    .expect("weights should exist for all sampled points"),
            )
        }
    }

    pub fn get_evaluation_on_domain(
        &self,
        domain: CircleDomain,
        twiddles: &TwiddleTree<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        if let Some(coeffs) = &self.coeffs {
            coeffs.evaluate_with_twiddles(domain, twiddles)
        } else {
            panic!("The polynomial's coefficients are not stored");
        }
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
    pub n_preprocessed_columns: usize,
}

impl<B: Backend> ComponentProvers<'_, B> {
    pub fn components(&self) -> Components<'_> {
        Components {
            components: self
                .components
                .iter()
                .map(|c| *c as &dyn Component)
                .collect_vec(),
            n_preprocessed_columns: self.n_preprocessed_columns,
        }
    }
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
        twiddles: &TwiddleTree<B>,
        log_blowup_factor: u32,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let components: Vec<&dyn Component> = self
            .components
            .iter()
            .map(|c| *c as &dyn Component)
            .collect();
        let evaluation_mode = EvaluationMode::infer(&components, log_blowup_factor);
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
            evaluation_mode,
        );
        // Parallel kernel warmup: backends with a JIT constraint lane compile each
        // component's fused kernel once (per AIR + arch, disk-cached) — a cold
        // compile is minutes per EC/hash component and was the dominant serial
        // cold-start cost. Lower on this thread (component state is `!Sync`), then
        // run the owned compile closures concurrently. No-op for non-JIT backends;
        // the eval loop below is unchanged and remains the source of truth.
        let precompile_jobs: Vec<Box<dyn FnOnce() + Send>> = self
            .components
            .iter()
            .filter_map(|component| component.precompile_prepare())
            .collect();
        if !precompile_jobs.is_empty() {
            crate::parallel_iter!(precompile_jobs).for_each(|job| job());
        }
        for component in &self.components {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        }
        accumulator.finalize(twiddles)
    }
}
