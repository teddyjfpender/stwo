use serde::{Deserialize, Serialize};
use stwo::core::channel::Blake2sChannelGeneric;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Backend, BackendForChannel};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}

// Byte-equality: grinding must reproduce the reference nonce, so it delegates to the
// SIMD backend on both channel variants.
impl<const IS_M31_OUTPUT: bool> GrindOps<Blake2sChannelGeneric<IS_M31_OUTPUT>> for CudaBackend {
    fn grind(channel: &Blake2sChannelGeneric<IS_M31_OUTPUT>, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

impl BackendForChannel<Blake2sMerkleChannel> for CudaBackend {}
impl BackendForChannel<stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel>
    for CudaBackend
{
}

impl stwo_constraint_framework::FrameworkBackend for CudaBackend {
    fn evaluate_constraint_quotients_on_domain<
        E: stwo_constraint_framework::FrameworkEval + Sync,
    >(
        component: &stwo_constraint_framework::FrameworkComponent<E>,
        trace: &stwo::prover::Trace<'_, Self>,
        evaluation_accumulator: &mut stwo::prover::DomainEvaluationAccumulator<Self>,
    ) {
        // v1: the generic CPU driver (columns round-trip through the host). A native
        // lane analogous to the Metal JIT shader can follow once it pays for itself.
        stwo_constraint_framework::evaluate_constraint_quotients_via_cpu(
            component,
            trace,
            evaluation_accumulator,
        );
    }
}

impl stwo::prover::backend::FromSimdColumns for CudaBackend {
    fn from_simd_base_column(
        column: stwo::prover::backend::Col<SimdBackend, stwo::core::fields::m31::BaseField>,
    ) -> stwo::prover::backend::Col<Self, stwo::core::fields::m31::BaseField> {
        use stwo::prover::backend::Column;
        crate::columns::BaseFieldVec::from_vec(column.to_cpu())
    }

    fn from_simd_evals(
        evals: Vec<
            stwo::prover::poly::circle::CircleEvaluation<
                SimdBackend,
                stwo::core::fields::m31::BaseField,
                stwo::prover::poly::BitReversedOrder,
            >,
        >,
    ) -> Vec<
        stwo::prover::poly::circle::CircleEvaluation<
            Self,
            stwo::core::fields::m31::BaseField,
            stwo::prover::poly::BitReversedOrder,
        >,
    > {
        evals
            .into_iter()
            .map(|eval| {
                stwo::prover::poly::circle::CircleEvaluation::new(
                    eval.domain,
                    Self::from_simd_base_column(eval.values),
                )
            })
            .collect()
    }
}
