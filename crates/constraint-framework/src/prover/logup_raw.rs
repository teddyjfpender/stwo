//! Deferred-finalize logup trace generation (the witness-on-GPU W1 lane).
//!
//! [`RawLogupTraceGenerator`] mirrors [`super::LogupTraceGenerator`]'s call shape
//! (`new_col` / `write_frac` / `finalize_col` / `into_raw`), but stores the raw
//! (numerator, denominator) columns instead of finalizing them. The finalize math —
//! batched denominator inversion, the fraction chain across columns, the claimed
//! sum, the cumsum shift, and the per-coordinate inclusive prefix sums — runs later,
//! on a backend of the caller's choice:
//!
//! - [`RawLogupTrace::finalize_on_simd`] replays the raws through the real
//!   [`super::LogupTraceGenerator`], so its output is **identical by construction** to the eager
//!   path (it *is* the eager path). This is the reference and the fallback.
//! - GPU backends consume [`RawLogupTrace`] directly (see `stwo-backend-cuda::finalize_raw_logup`):
//!   the raw pairs upload at 8 words/row — the same net PCIe as uploading the 4 words/row finalized
//!   column plus skipping the 4 words/row `from_simd_evals` transfer — and the interaction columns
//!   are born on device.
//!
//! Why the split is here and not lower: moving the per-row `combine()` write loops
//! to the GPU would require uploading the (wider) `lookup_data` inputs, which costs
//! more PCIe than it saves in host math. See `gpu_benchmarks/WITNESS_ON_GPU.md` in
//! the stwo-cairo fork for the full design.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::SecureColumn;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;

use super::logup::LogupTraceGenerator;

/// One logup column's raw inputs: per packed row, the numerator (a `SecureField` in
/// coordinate layout) and the denominator.
pub struct RawLogupColumn {
    pub numerator: SecureColumnByCoords<SimdBackend>,
    pub denominator: SecureColumn,
}

/// The raw (unfinalized) logup interaction trace of one component.
pub struct RawLogupTrace {
    pub log_size: u32,
    pub columns: Vec<RawLogupColumn>,
}

impl RawLogupTrace {
    /// Finalizes on the SIMD backend by replaying the raw pairs through the eager
    /// [`LogupTraceGenerator`] — byte-identical to having used it directly.
    pub fn finalize_on_simd(
        self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let mut gen = LogupTraceGenerator::new(self.log_size);
        for raw in self.columns {
            let mut col_gen = gen.new_col();
            for (vec_row, &denom) in raw.denominator.data.iter().enumerate() {
                let numerator = unsafe { raw.numerator.packed_at(vec_row) };
                col_gen.write_frac(vec_row, numerator, denom);
            }
            col_gen.finalize_col();
        }
        gen.finalize_last()
    }
}

/// Deferred-finalize counterpart of [`LogupTraceGenerator`]; see the module docs.
pub struct RawLogupTraceGenerator {
    log_size: u32,
    columns: Vec<RawLogupColumn>,
}

impl RawLogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        Self {
            log_size,
            columns: vec![],
        }
    }

    /// Mirrors [`LogupTraceGenerator::uninitialized`]; the raw generator allocates
    /// per column, so there is nothing to leave uninitialized here.
    ///
    /// # Safety
    /// Kept `unsafe` so the call sites stay drop-in compatible with the eager type.
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        Self::new(log_size)
    }

    /// Allocate a new lookup column.
    pub fn new_col(&mut self) -> RawLogupColGenerator<'_> {
        let size = 1 << self.log_size;
        RawLogupColGenerator {
            gen: self,
            numerator: unsafe { SecureColumnByCoords::<SimdBackend>::uninitialized(size) },
            denominator: unsafe { <SecureColumn as Column<SecureField>>::uninitialized(size) },
        }
    }

    pub fn into_raw(self) -> RawLogupTrace {
        RawLogupTrace {
            log_size: self.log_size,
            columns: self.columns,
        }
    }
}

/// Raw counterpart of [`super::LogupColGenerator`].
pub struct RawLogupColGenerator<'a> {
    gen: &'a mut RawLogupTraceGenerator,
    numerator: SecureColumnByCoords<SimdBackend>,
    denominator: SecureColumn,
}

impl RawLogupColGenerator<'_> {
    /// Write a fraction to the column at a packed row.
    pub fn write_frac(
        &mut self,
        vec_row: usize,
        numerator: PackedSecureField,
        denominator: PackedSecureField,
    ) {
        unsafe {
            self.numerator.set_packed(vec_row, numerator);
            *self.denominator.data.get_unchecked_mut(vec_row) = denominator;
        }
    }

    /// Store the raw column; no math happens until the backend finalize.
    pub fn finalize_col(self) {
        self.gen.columns.push(RawLogupColumn {
            numerator: self.numerator,
            denominator: self.denominator,
        });
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo::prover::backend::simd::m31::LOG_N_LANES;
    use stwo::prover::backend::simd::qm31::PackedSecureField;
    use stwo::prover::backend::Column;

    use super::RawLogupTraceGenerator;
    use crate::prover::logup::LogupTraceGenerator;
    use crate::qm31;

    /// The raw path replayed on SIMD must be byte-identical to the eager generator.
    #[test]
    fn raw_finalize_matches_eager() {
        const LOG_SIZE: u32 = 8;
        const N_COLS: usize = 3;
        let mut rng = SmallRng::seed_from_u64(0);
        let fracs: Vec<Vec<(PackedSecureField, PackedSecureField)>> = (0..N_COLS)
            .map(|_| {
                (0..1 << (LOG_SIZE - LOG_N_LANES))
                    .map(|_| {
                        let num = PackedSecureField::broadcast(qm31!(
                            rng.gen::<u32>() >> 1,
                            rng.gen::<u32>() >> 1,
                            rng.gen::<u32>() >> 1,
                            rng.gen::<u32>() >> 1
                        ));
                        // Nonzero denominator.
                        let den = PackedSecureField::broadcast(qm31!(
                            (rng.gen::<u32>() >> 1) | 1,
                            rng.gen::<u32>() >> 1,
                            rng.gen::<u32>() >> 1,
                            rng.gen::<u32>() >> 1
                        ));
                        (num, den)
                    })
                    .collect()
            })
            .collect();

        let mut eager = LogupTraceGenerator::new(LOG_SIZE);
        for col in &fracs {
            let mut col_gen = eager.new_col();
            for (vec_row, &(num, den)) in col.iter().enumerate() {
                col_gen.write_frac(vec_row, num, den);
            }
            col_gen.finalize_col();
        }
        let (eager_trace, eager_sum) = eager.finalize_last();

        let mut raw = RawLogupTraceGenerator::new(LOG_SIZE);
        for col in &fracs {
            let mut col_gen = raw.new_col();
            for (vec_row, &(num, den)) in col.iter().enumerate() {
                col_gen.write_frac(vec_row, num, den);
            }
            col_gen.finalize_col();
        }
        let (raw_trace, raw_sum) = raw.into_raw().finalize_on_simd();

        assert_eq!(eager_sum, raw_sum);
        assert_eq!(eager_trace.len(), raw_trace.len());
        for (e, r) in eager_trace.iter().zip(raw_trace.iter()) {
            assert_eq!(e.values.to_cpu(), r.values.to_cpu());
        }
    }
}
