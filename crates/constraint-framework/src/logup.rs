use core::array;
use core::ops::{Mul, Sub};

use num_traits::{One, Zero};
use std_shims::{vec, Vec};
use stwo::core::channel::Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::Fraction;

use super::EvalAtRow;

/// Evaluates constraints for batched logups.
/// These constraint enforce the sum of multiplicity_i / (z + sum_j alpha^j * x_j) = claimed_sum.
pub struct LogupAtRow<E: EvalAtRow> {
    /// The index of the interaction used for the cumulative sum columns.
    pub interaction: usize,
    /// The total sum of all the fractions divided by n_rows.
    pub cumsum_shift: SecureField,
    /// The evaluation of the last cumulative sum column.
    pub fracs: Vec<Fraction<E::EF, E::EF>>,
    pub is_finalized: bool,
    pub log_size: u32,
}

impl<E: EvalAtRow> Default for LogupAtRow<E> {
    fn default() -> Self {
        Self::dummy()
    }
}
impl<E: EvalAtRow> LogupAtRow<E> {
    pub fn new(interaction: usize, claimed_sum: SecureField, log_size: u32) -> Self {
        Self {
            interaction,
            cumsum_shift: claimed_sum / BaseField::from_u32_unchecked(1 << log_size),
            fracs: vec![],
            is_finalized: true,
            log_size,
        }
    }

    // TODO(alont): Remove this once unnecessary LogupAtRows are gone.
    pub fn dummy() -> Self {
        Self {
            interaction: 100,
            cumsum_shift: SecureField::one(),
            fracs: vec![],
            is_finalized: true,
            log_size: 10,
        }
    }
}

/// Ensures that the LogupAtRow is finalized.
/// LogupAtRow should be finalized exactly once.
impl<E: EvalAtRow> Drop for LogupAtRow<E> {
    fn drop(&mut self) {
        assert!(self.is_finalized, "LogupAtRow was not finalized");
    }
}

/// Interaction elements for the logup protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupElements<const N: usize> {
    pub z: SecureField,
    pub alpha: SecureField,
    pub alpha_powers: [SecureField; N],
}
impl<const N: usize> LookupElements<N> {
    pub fn draw(channel: &mut impl Channel) -> Self {
        let [z, alpha] = channel.draw_secure_felts(2).try_into().unwrap();
        let mut cur = SecureField::one();
        let alpha_powers = array::from_fn(|_| {
            let res = cur;
            cur *= alpha;
            res
        });
        Self {
            z,
            alpha,
            alpha_powers,
        }
    }
    /// Generic scalar fold: `Σ_j alpha_powers[j] · values[j] − z`.
    ///
    /// This is the unchanged, behaviorally-identical path used by the verifier and every
    /// non-SIMD caller. When the `prover` feature is enabled, `combine` instead routes through
    /// [`CombineDispatch`], whose specialized impl gives the SIMD packed types a
    /// delayed-reduction fast path that computes the *identical* field element; every other
    /// instantiation falls back to this exact fold via the trait's `default` impl.
    #[cfg(not(feature = "prover"))]
    pub fn combine<F: Clone, EF>(&self, values: &[F]) -> EF
    where
        EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
    {
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        values
            .iter()
            .zip(self.alpha_powers)
            .fold(EF::zero(), |acc, (value, power)| {
                acc + EF::from(power) * value.clone()
            })
            - EF::from(self.z)
    }

    /// See the `not(prover)` variant for the documentation.
    ///
    /// With the `prover` feature on, this dispatches (at monomorphization time) to a SIMD
    /// fast path for the packed field types and to the unchanged generic fold otherwise.
    #[cfg(feature = "prover")]
    pub fn combine<F: Clone, EF>(&self, values: &[F]) -> EF
    where
        EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
    {
        CombineDispatch::<F, EF>::combine_dispatch(self, values)
    }

    pub fn dummy() -> Self {
        let z = SecureField::from_u32_unchecked(1, 2, 3, 4);
        let alpha = SecureField::from_u32_unchecked(4, 3, 2, 1);
        let mut cur = SecureField::one();
        let alpha_powers = array::from_fn(|_| {
            let res = cur;
            cur *= alpha;
            res
        });
        Self {
            z,
            alpha,
            alpha_powers,
        }
    }
}

/// Compile-time dispatch backing [`LookupElements::combine`] when the `prover` feature is on.
///
/// The `default` impl is the generic scalar fold (identical to the historical `combine` body).
/// A specialized impl provides a SIMD delayed-reduction fast path for the packed field types.
/// The specialization is additive: it never affects any non-packed instantiation, so the
/// generic path stays byte-for-byte identical in behavior. (Without the `prover` feature this
/// trait does not exist and `combine` is the plain generic fold above — no specialization.)
#[cfg(feature = "prover")]
trait CombineDispatch<F: Clone, EF>
where
    EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
{
    fn combine_dispatch(&self, values: &[F]) -> EF;
}

#[cfg(feature = "prover")]
impl<const N: usize, F: Clone, EF> CombineDispatch<F, EF> for LookupElements<N>
where
    EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
{
    default fn combine_dispatch(&self, values: &[F]) -> EF {
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        values
            .iter()
            .zip(self.alpha_powers)
            .fold(EF::zero(), |acc, (value, power)| {
                acc + EF::from(power) * value.clone()
            })
            - EF::from(self.z)
    }
}

/// SIMD specialization of [`LookupElements::combine`] for the packed prover types.
///
/// `combine` computes `Σ_j alpha_powers[j] · values[j] − z`. With `EF = PackedQM31` and
/// `F = PackedM31`, the generic fold performs one full `PackedQM31 × PackedM31` multiply
/// (4 reduced `PackedM31` multiplies) plus a `PackedQM31` add per term. Because a
/// `QM31 × M31` product distributes coordinate-wise, this is exactly four independent M31
/// dot products `Σ_j alpha_powers[j].coord_c · values[j]`, one per QM31 coordinate.
///
/// We delegate those four dot products to [`dot_delayed_qm31_scalar`], which accumulates with
/// delayed modular reduction (a single reduction per coordinate at the end instead of one per
/// product). This yields the *identical* `PackedQM31` element as the generic fold — it is a
/// pure re-association of the exact integer arithmetic, with the same final `− z` subtraction.
/// The accumulation bound is documented on `dot_delayed_qm31_scalar`.
#[cfg(feature = "prover")]
impl<const N: usize>
    CombineDispatch<
        stwo::prover::backend::simd::m31::PackedM31,
        stwo::prover::backend::simd::qm31::PackedQM31,
    > for LookupElements<N>
{
    fn combine_dispatch(
        &self,
        values: &[stwo::prover::backend::simd::m31::PackedM31],
    ) -> stwo::prover::backend::simd::qm31::PackedQM31 {
        use stwo::prover::backend::simd::qm31::dot_delayed_qm31_scalar;

        // Preserve the generic path's precondition exactly.
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        // The length cap is asserted inside `dot_delayed_qm31_scalar`; `values.len()` here is
        // bounded by `alpha_powers.len() = N`, which is tiny in every real use.
        let combined = dot_delayed_qm31_scalar(&self.alpha_powers[..values.len()], values);
        combined - self.z
    }
}

/// SIMD specialization of [`LookupElements::combine`] for the `VeryPacked` prover types.
///
/// This is the composition-phase analogue of the `(PackedM31, PackedQM31)` impl above.
/// `SimdDomainEvaluator` (the constraint evaluator used by the composition phase — the
/// largest prove phase) has associated types `F = VeryPackedM31` and `EF = VeryPackedQM31`,
/// and every `RelationEntry`'s `values` slice is `&[Self::F] = &[VeryPackedM31]` (base-field
/// values, enforced by the `RelationEntry` type), so `combine` is monomorphized here with
/// `F = VeryPackedM31`, `EF = VeryPackedQM31`. Without this impl that instantiation falls
/// through to the generic per-element fold; the `(PackedM31, PackedQM31)` impl never fires in
/// composition.
///
/// `VeryPacked` is `N_VERY_PACKED_ELEMS` independent `Packed` sub-lanes and the generic fold
/// is element-wise, so this delegates to [`dot_delayed_very_packed_qm31_scalar`], which keeps
/// one per-sub-lane delayed-reduction accumulator and computes the *identical*
/// `VeryPackedQM31` element the generic fold would — a pure re-association of exact integer
/// arithmetic, with the same trailing `− z`. The bound is per `PackedM31` sub-lane and
/// unchanged from the `Packed` case; see the function's doc comment.
#[cfg(feature = "prover")]
impl<const N: usize>
    CombineDispatch<
        stwo::prover::backend::simd::very_packed_m31::VeryPackedM31,
        stwo::prover::backend::simd::very_packed_m31::VeryPackedQM31,
    > for LookupElements<N>
{
    fn combine_dispatch(
        &self,
        values: &[stwo::prover::backend::simd::very_packed_m31::VeryPackedM31],
    ) -> stwo::prover::backend::simd::very_packed_m31::VeryPackedQM31 {
        use stwo::prover::backend::simd::very_packed_m31::dot_delayed_very_packed_qm31_scalar;

        // Preserve the generic path's precondition exactly.
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        // The length cap is asserted inside `dot_delayed_very_packed_qm31_scalar`;
        // `values.len()` here is bounded by `alpha_powers.len() = N`, tiny in every real use.
        let combined =
            dot_delayed_very_packed_qm31_scalar(&self.alpha_powers[..values.len()], values);
        // `- self.z`: `VeryPackedQM31 - SecureField` broadcasts `z` across sub-lanes, the same
        // scalar subtraction the generic fold's trailing `- EF::from(self.z)` performs.
        combined - self.z
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::FieldExpOps;

    use super::LookupElements;

    #[test]
    fn test_lookup_elements_combine() {
        let mut channel = Blake2sChannel::default();
        let lookup_elements = LookupElements::<3>::draw(&mut channel);
        let values = [
            BaseField::from_u32_unchecked(123),
            BaseField::from_u32_unchecked(456),
            BaseField::from_u32_unchecked(789),
        ];

        assert_eq!(
            lookup_elements.combine::<BaseField, SecureField>(&values),
            BaseField::from_u32_unchecked(123)
                + BaseField::from_u32_unchecked(456) * lookup_elements.alpha
                + BaseField::from_u32_unchecked(789) * lookup_elements.alpha.pow(2)
                - lookup_elements.z
        );
    }

    /// The SIMD-specialized `combine` path (packed types) must produce, lane for lane, exactly
    /// what the generic scalar fold produces. Covers a spread of lengths (including the
    /// largest real `LookupElements` tuple, 96, plus one) and the unreduced `[0, P]` boundary.
    #[cfg(feature = "prover")]
    #[test]
    fn test_combine_packed_matches_scalar() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        use stwo::core::fields::m31::P;
        use stwo::prover::backend::simd::m31::{PackedBaseField, N_LANES};

        // Largest tuple: `N_ROUND_INPUT_FELTS = (16 + 16 + 16) * 2 = 96`, so test up to 97.
        const N: usize = 97;
        let mut channel = Blake2sChannel::default();
        let lookup_elements = LookupElements::<N>::draw(&mut channel);

        let mut rng = SmallRng::seed_from_u64(7);
        // Edge values exercising the unreduced `[0, P]` representation (incl. `P`).
        const EDGE: [u32; 6] = [0, 1, 2, (1 << 30) - 1, P - 1, P];

        for len in [0usize, 1, 2, 3, 4, 5, 50, 96, 97] {
            // Build `len` packed values; each lane gets a distinct value (some random, some
            // edge) so a lane-mapping bug would surface.
            let lane_values: Vec<[BaseField; N_LANES]> = (0..len)
                .map(|j| {
                    core::array::from_fn(|lane| {
                        if (lane + j).is_multiple_of(5) {
                            // Reduced edge values; `P` reduces to 0 via `from_u32_unchecked`'s
                            // caller — use raw reduce to mirror field semantics.
                            BaseField::from_u32_unchecked(EDGE[(lane + j) % EDGE.len()] % P)
                        } else {
                            BaseField::from_u32_unchecked(rng.gen::<u32>() % P)
                        }
                    })
                })
                .collect();

            let packed_values: Vec<PackedBaseField> = lane_values
                .iter()
                .map(|lanes| PackedBaseField::from_array(*lanes))
                .collect();

            let packed_res = lookup_elements
                .combine::<PackedBaseField, stwo::prover::backend::simd::qm31::PackedQM31>(
                    &packed_values,
                );
            let packed_lanes = packed_res.to_array();

            for lane in 0..N_LANES {
                let scalar_values: Vec<BaseField> =
                    lane_values.iter().map(|lanes| lanes[lane]).collect();
                let scalar_res = lookup_elements.combine::<BaseField, SecureField>(&scalar_values);
                assert_eq!(packed_lanes[lane], scalar_res, "len={len}, lane={lane}");
            }
        }
    }

    /// The SIMD-specialized `combine` path for the `VeryPacked` types (the composition-phase
    /// instantiation) must produce, *raw representative for raw representative*, exactly what
    /// the generic element-wise fold produces.
    ///
    /// We compare via raw `into_simd()` lanes, **not** `to_array()`: `to_array` reduces the
    /// boundary representative `P` down to `0`, which would mask a divergence where one path
    /// emits `P` and the other emits `0`. The reference is the generic `VeryPacked` fold
    /// (`Σ VeryPackedQM31::from(power) * value − z`, element-wise — exactly the `default`
    /// `combine_dispatch` body), computed directly here so it bypasses the specialization.
    ///
    /// Analytic note: both paths keep every coordinate sub-lane in `[0, P]`. The specialized
    /// path inherits this from `PackedQM31DelayedDot::finalize` (each coordinate reduced by the
    /// per-lane Mersenne reduction, which lands in `[0, P]`), per sub-lane; the generic path
    /// inherits it from the `PackedM31`/`PackedQM31` field ops. The boundary inputs below
    /// (all-`P` and per-sub-lane-rotated `P`) force the boundary representative through both.
    #[cfg(feature = "prover")]
    #[test]
    fn test_combine_very_packed_matches_generic_fold_raw() {
        use std::simd::u32x16;

        use num_traits::Zero;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        use stwo::core::fields::m31::P;
        use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
        use stwo::prover::backend::simd::very_packed_m31::{
            VeryPackedM31, VeryPackedQM31, N_VERY_PACKED_ELEMS,
        };

        // Largest tuple in `crates/examples` is `N_ROUND_INPUT_FELTS = 96`; test up to 97.
        const N: usize = 97;
        let mut channel = Blake2sChannel::default();
        let lookup_elements = LookupElements::<N>::draw(&mut channel);

        let mut rng = SmallRng::seed_from_u64(99);
        // Edge values exercising the unreduced `[0, P]` representation (incl. the boundary `P`).
        const EDGE: [u32; 6] = [0, 1, 2, (1 << 30) - 1, P - 1, P];

        // Build a `PackedM31` from raw lane values, allowing the boundary `P` that the public
        // `from_array` would reduce to `0`.
        let packed_raw = |lanes: [u32; N_LANES]| -> PackedM31 {
            unsafe { PackedM31::from_simd_unchecked(u32x16::from_array(lanes)) }
        };

        // The generic element-wise fold, computed without going through the specialization:
        // this is exactly the `default fn combine_dispatch` algorithm applied to `VeryPacked`.
        let generic_fold = |values: &[VeryPackedM31]| -> VeryPackedQM31 {
            let mut acc = VeryPackedQM31::zero();
            for (value, power) in values.iter().zip(lookup_elements.alpha_powers) {
                acc += VeryPackedQM31::from(power) * *value;
            }
            acc - lookup_elements.z
        };

        // Raw per-coordinate, per-sub-lane representatives of a `VeryPackedQM31` (no reduction).
        let raw_repr = |v: &VeryPackedQM31| -> Vec<[u32; N_LANES]> {
            let mut out = Vec::new();
            for sub_lane in 0..N_VERY_PACKED_ELEMS {
                for coord in v.0[sub_lane].into_packed_m31s() {
                    out.push(coord.into_simd().to_array());
                }
            }
            out
        };

        for len in [0usize, 1, 2, 3, 4, 5, 50, 96, 97] {
            // Per-term `VeryPackedM31`: each of the `N_LANES * N_VERY_PACKED_ELEMS` lanes gets a
            // distinct value (some edge incl. `P`, some random) so a lane- or sub-lane-mapping
            // bug surfaces. Built from raw lanes so `P` is preserved, not collapsed to `0`.
            let values: Vec<VeryPackedM31> = (0..len)
                .map(|j| {
                    VeryPackedM31::from(core::array::from_fn::<PackedM31, N_VERY_PACKED_ELEMS, _>(
                        |sub_lane| {
                            packed_raw(core::array::from_fn(|lane| {
                                let idx = lane + j + sub_lane * 7;
                                if idx.is_multiple_of(5) {
                                    EDGE[idx % EDGE.len()]
                                } else {
                                    rng.gen::<u32>() % P
                                }
                            }))
                        },
                    ))
                })
                .collect();

            // All-`P` boundary instance: every lane of every term is the boundary representative.
            let all_p: Vec<VeryPackedM31> = (0..len)
                .map(|_| VeryPackedM31::from([packed_raw([P; N_LANES]); N_VERY_PACKED_ELEMS]))
                .collect();

            for input in [&values, &all_p] {
                let specialized = lookup_elements.combine::<VeryPackedM31, VeryPackedQM31>(input);
                let reference = generic_fold(input);
                assert_eq!(
                    raw_repr(&specialized),
                    raw_repr(&reference),
                    "raw representative mismatch at len={len}"
                );
            }
        }
    }
}
