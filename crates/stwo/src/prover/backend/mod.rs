use std::fmt::Debug;

pub use cpu::CpuBackend;

use crate::core::channel::MerkleChannel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::proof_of_work::GrindOps;
use crate::prover::fri::FriOps;
use crate::prover::lookups::gkr_prover::GkrOps;
use crate::prover::poly::circle::PolyOps;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::{AccumulationOps, QuotientOps};

pub mod cpu;
pub mod simd;

pub trait Backend:
    Copy
    + Clone
    + Debug
    + ColumnOps<BaseField>
    + ColumnOps<SecureField>
    + PolyOps
    + QuotientOps
    + FriOps
    + AccumulationOps
    + GkrOps
{
}

pub trait BackendForChannel<MC: MerkleChannel>:
    Backend + MerkleOpsLifted<MC::H> + GrindOps<MC::C>
{
}

/// Conversion of columns produced by the [`simd::SimdBackend`] into this backend's column
/// representation.
///
/// Witness generation is typically written directly against the SIMD backend's column types —
/// the portable host representation. This trait is the transfer seam between such a witness
/// generator and the proving backend: the witness is generated on the host and handed to the
/// target backend at the commitment boundary. For device backends (GPU), the natural
/// implementation is the host-to-device upload; such backends should override
/// [`Self::from_simd_evals`] to batch or overlap transfers.
pub trait FromSimdColumns: Backend {
    /// Converts a SIMD base-field column into this backend's column type.
    fn from_simd_base_column(column: Col<simd::SimdBackend, BaseField>) -> Col<Self, BaseField>;

    /// Converts SIMD circle evaluations (e.g. a generated trace) into this backend's.
    fn from_simd_evals(
        evals: Vec<
            crate::prover::poly::circle::CircleEvaluation<
                simd::SimdBackend,
                BaseField,
                crate::prover::poly::BitReversedOrder,
            >,
        >,
    ) -> Vec<
        crate::prover::poly::circle::CircleEvaluation<
            Self,
            BaseField,
            crate::prover::poly::BitReversedOrder,
        >,
    > {
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;

        #[cfg(not(feature = "parallel"))]
        let iter = evals.into_iter();
        #[cfg(feature = "parallel")]
        let iter = evals.into_par_iter();

        iter.map(|eval| {
            crate::prover::poly::circle::CircleEvaluation::new(
                eval.domain,
                Self::from_simd_base_column(eval.values),
            )
        })
        .collect()
    }
}

impl FromSimdColumns for simd::SimdBackend {
    fn from_simd_base_column(column: Col<simd::SimdBackend, BaseField>) -> Col<Self, BaseField> {
        column
    }

    fn from_simd_evals(
        evals: Vec<
            crate::prover::poly::circle::CircleEvaluation<
                simd::SimdBackend,
                BaseField,
                crate::prover::poly::BitReversedOrder,
            >,
        >,
    ) -> Vec<
        crate::prover::poly::circle::CircleEvaluation<
            Self,
            BaseField,
            crate::prover::poly::BitReversedOrder,
        >,
    > {
        evals
    }
}

impl FromSimdColumns for CpuBackend {
    fn from_simd_base_column(column: Col<simd::SimdBackend, BaseField>) -> Col<Self, BaseField> {
        column.to_cpu()
    }
}

pub trait ColumnOps<T> {
    type Column: Column<T>;
    fn bit_reverse_column(column: &mut Self::Column);
}

pub type Col<B, T> = <B as ColumnOps<T>>::Column;

// TODO(alont): Consider removing the generic parameter and only support BaseField.
pub trait Column<T>: Clone + Debug + FromIterator<T> + Send + Sync {
    /// Creates a new column of zeros with the given length.
    fn zeros(len: usize) -> Self;
    /// Creates a new column of uninitialized values with the given length.
    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    unsafe fn uninitialized(len: usize) -> Self;
    /// Returns a cpu vector of the column.
    fn to_cpu(&self) -> Vec<T>;
    /// Returns the length of the column.
    fn len(&self) -> usize;
    /// Returns true if the column is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Retrieves the element at the given index.
    fn at(&self, index: usize) -> T;
    /// Retrieves the element at the given index without canonicalizing its representation.
    ///
    /// Backends whose stored representation may be unreduced (e.g. SIMD columns holding
    /// `[0, P]`) return the raw stored value; the default forwards to [`Self::at`]. Used where
    /// the exact stored bytes matter, such as recomputing Merkle leaf hashes.
    fn at_unreduced(&self, index: usize) -> T {
        self.at(index)
    }
    /// Sets the element at the given index.
    fn set(&mut self, index: usize, value: T);
    /// Splits the column into two halves.
    fn split_at_mid(self) -> (Self, Self);
    /// Shrinks the column's backing allocation to fit its length. Backends without a
    /// meaningful implementation may leave this as a no-op.
    fn shrink_to_fit(&mut self) {}
}
