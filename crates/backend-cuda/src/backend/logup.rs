//! Device-side logup interaction-trace finalize (the witness-on-GPU W1 lane).
//!
//! Consumes a [`RawLogupTrace`] (host-written (numerator, denominator) pairs; see
//! `stwo-constraint-framework::prover::logup_raw`) and performs the entire finalize
//! on the GPU: batched denominator inversion, the fraction chain across columns, the
//! claimed-sum reduction (16-byte fenced readback), the cumsum shift, and the four
//! per-coordinate inclusive prefix sums. The returned columns are **born on device**
//! — they feed the interaction-tree commit directly, with no `from_simd_evals`
//! transfer.
//!
//! PCIe accounting: the raw pairs upload at 8 words/row, exactly the 4 words/row a
//! host-finalized column costs via `from_simd_evals` plus the 4 words/row of
//! denominators that never needed to exist on device under the host path — net
//! traffic-neutral, while all finalize math leaves the host.
//!
//! Byte-equality: field adds are associative and commutative (any reduction or scan
//! order yields the same element — the SIMD reference's own parallel sum relies on
//! this), inverses are unique, and the per-column chain order is preserved. Gated by
//! `finalize_raw_logup_matches_simd` (hardware differential vs the SIMD reference)
//! and, after the stwo-cairo integration, the Cairo e2e proof byte-equality.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::RawLogupTrace;

use crate::backend::CudaBackend;
use crate::columns::base_field_vec::BaseFieldVec;
use crate::columns::bindings;

/// Finalizes a raw logup trace on the GPU. Returns the interaction-trace columns
/// (device-resident, in the same column order as the SIMD reference: 4 coordinate
/// columns per logup column) and the claimed sum.
pub fn finalize_raw_logup(
    raw: RawLogupTrace,
) -> (
    Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = raw.log_size;
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    crate::columns::bindings::ensure_mem_pool_init();

    // Chain the columns: value_k = num_k * inv(den_k) + value_{k-1}, computed in
    // place over the uploaded numerator coordinate columns.
    let mut finalized: Vec<[BaseFieldVec; 4]> = Vec::with_capacity(raw.columns.len());
    for raw_col in raw.columns {
        // Numerator coordinates are contiguous m31 columns in the SIMD layout —
        // upload as-is. The denominators are packed-lane and de-interleave on device.
        let coords: [BaseFieldVec; 4] = std::array::from_fn(|i| {
            let column = &raw_col.numerator.columns[i];
            let words: &[u32] =
                unsafe { std::slice::from_raw_parts(column.data.as_ptr().cast(), column.len()) };
            let device_ptr = unsafe {
                bindings::copy_uint32_t_vec_from_host_to_device(words.as_ptr(), size as u32)
            };
            BaseFieldVec::new(device_ptr, size)
        });
        let denom_words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                raw_col.denominator.data.as_ptr().cast(),
                // PackedSecureField = 4 coords x 16 lanes = 64 u32 words per packed element.
                raw_col.denominator.data.len() * 64,
            )
        };
        let denom_dev = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                denom_words.as_ptr(),
                denom_words.len() as u32,
            )
        };
        let denom = BaseFieldVec::new(denom_dev, denom_words.len());

        let prev = finalized.last();
        unsafe {
            bindings::logup_fraction_chain(
                coords[0].device_ptr,
                coords[1].device_ptr,
                coords[2].device_ptr,
                coords[3].device_ptr,
                denom.device_ptr,
                prev.map_or(std::ptr::null(), |p| p[0].device_ptr),
                prev.map_or(std::ptr::null(), |p| p[1].device_ptr),
                prev.map_or(std::ptr::null(), |p| p[2].device_ptr),
                prev.map_or(std::ptr::null(), |p| p[3].device_ptr),
                size as u32,
            );
        }
        finalized.push(coords);
    }

    // Last column: claimed sum, shift, and the four coordinate prefix sums.
    let last = finalized.last().expect("raw logup trace has no columns");
    let claimed_sum: SecureField = unsafe {
        bindings::logup_sum_secure_coords(
            last[0].device_ptr,
            last[1].device_ptr,
            last[2].device_ptr,
            last[3].device_ptr,
            size as u32,
        )
    }
    .into();
    let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << log_size);
    unsafe {
        bindings::logup_shift_secure_coords(
            last[0].device_ptr,
            last[1].device_ptr,
            last[2].device_ptr,
            last[3].device_ptr,
            cumsum_shift.into(),
            size as u32,
        );
        for coord in last.iter() {
            bindings::inclusive_prefix_sum(coord.device_ptr, size as u32);
        }
    }

    let trace = finalized
        .into_iter()
        .flat_map(|coords| coords.map(|col| CircleEvaluation::new(domain, col)))
        .collect();
    (trace, claimed_sum)
}

#[cfg(all(test, stwo_cuda_link))]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo::core::fields::qm31::SecureField;
    use stwo::prover::backend::simd::m31::LOG_N_LANES;
    use stwo::prover::backend::simd::qm31::PackedSecureField;
    use stwo::prover::backend::Column;
    use stwo_constraint_framework::RawLogupTraceGenerator;

    use super::finalize_raw_logup;

    fn random_raw(log_size: u32, n_cols: usize, seed: u64) -> RawLogupTraceGenerator {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut gen = RawLogupTraceGenerator::new(log_size);
        for _ in 0..n_cols {
            let mut col = gen.new_col();
            for vec_row in 0..1 << (log_size - LOG_N_LANES) {
                let num = PackedSecureField::broadcast(SecureField::from_u32_unchecked(
                    rng.gen::<u32>() >> 1,
                    rng.gen::<u32>() >> 1,
                    rng.gen::<u32>() >> 1,
                    rng.gen::<u32>() >> 1,
                ));
                let den = PackedSecureField::broadcast(SecureField::from_u32_unchecked(
                    (rng.gen::<u32>() >> 1) | 1,
                    rng.gen::<u32>() >> 1,
                    rng.gen::<u32>() >> 1,
                    rng.gen::<u32>() >> 1,
                ));
                col.write_frac(vec_row, num, den);
            }
            col.finalize_col();
        }
        gen
    }

    /// The device finalize must byte-match the SIMD reference, including the
    /// claimed sum and the prefix-summed last column.
    #[test]
    fn finalize_raw_logup_matches_simd() {
        for (log_size, n_cols, seed) in [(8u32, 3usize, 0u64), (12, 5, 1), (14, 1, 2)] {
            let simd = random_raw(log_size, n_cols, seed).into_raw();
            let cuda = random_raw(log_size, n_cols, seed).into_raw();

            let (simd_trace, simd_sum) = simd.finalize_on_simd();
            let (cuda_trace, cuda_sum) = finalize_raw_logup(cuda);

            assert_eq!(simd_sum, cuda_sum, "claimed sum log_size={log_size}");
            assert_eq!(simd_trace.len(), cuda_trace.len());
            for (i, (s, c)) in simd_trace.iter().zip(cuda_trace.iter()).enumerate() {
                assert_eq!(
                    s.values.to_cpu(),
                    c.values.to_cpu(),
                    "column {i} log_size={log_size}"
                );
            }
        }
    }
}
