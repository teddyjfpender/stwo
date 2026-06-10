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
impl GrindOps<Blake2sChannelGeneric<false>> for CudaBackend {
    /// GPU grind: chunked `atomicMin` search returning the LOWEST valid nonce, which
    /// matches the SIMD search order byte-exactly (same `H(POW_PREFIX, [0;12], digest,
    /// pow_bits)` preimage, same trailing-zero check on the first output word).
    fn grind(channel: &Blake2sChannelGeneric<false>, pow_bits: u32) -> u64 {
        use stwo::core::vcs::blake2_hash::Blake2sHasherGeneric;
        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");
        let digest = channel.digest();

        let mut hasher = Blake2sHasherGeneric::<false>::default();
        hasher.update(&Blake2sChannelGeneric::<false>::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&pow_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        let prefixed_words: Vec<u32> = prefixed_digest
            .0
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        crate::columns::bindings::ensure_mem_pool_init();
        unsafe { stwo_backend_cuda_kernels::raw::grind_blake2s(prefixed_words.as_ptr(), pow_bits) }
    }
}

impl GrindOps<Blake2sChannelGeneric<true>> for CudaBackend {
    /// The M31-output channel's PoW hash differs at finalize; the GPU kernel implements
    /// the non-M31 variant only, so this delegates (NitrooZK does the same).
    fn grind(channel: &Blake2sChannelGeneric<true>, pow_bits: u32) -> u64 {
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
        // Per-component GPU kernels (NitrooZK lineage) with CPU fallback on the same
        // accumulator claim; STWO_CUDA_DISABLE_CONSTRAINT_KERNELS forces the fallback.
        super::constraint_eval::evaluate_constraint_quotients(
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
