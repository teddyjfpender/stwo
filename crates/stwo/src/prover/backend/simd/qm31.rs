use std::array;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::simd::num::SimdUint;
use std::simd::Simd;

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};
use rand::distributions::{Distribution, Standard};

use super::cm31::PackedCM31;
use super::m31::{reduce_u64_lanes_to_m31, PackedM31, DOT_DELAYED_MAX_LEN, N_LANES};
use super::PACKED_QM31_BATCH_INVERSE_CHUNK_SIZE;
use crate::core::fields::m31::{M31, MODULUS_BITS};
use crate::core::fields::qm31::QM31;
use crate::core::fields::{batch_inverse_chunked, FieldExpOps};
use crate::core::utils;

pub type PackedSecureField = PackedQM31;

/// SIMD implementation of [`QM31`].
#[derive(Copy, Clone, Debug)]
pub struct PackedQM31(pub [PackedCM31; 2]);

unsafe impl Send for PackedQM31 {}
unsafe impl Sync for PackedQM31 {}

impl PackedQM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub const fn broadcast(value: QM31) -> Self {
        Self([
            PackedCM31::broadcast(value.0),
            PackedCM31::broadcast(value.1),
        ])
    }

    /// Returns all `a` values such that each vector element is represented as `a + bu`.
    pub const fn a(&self) -> PackedCM31 {
        self.0[0]
    }

    /// Returns all `b` values such that each vector element is represented as `a + bu`.
    pub const fn b(&self) -> PackedCM31 {
        self.0[1]
    }

    pub fn to_array(&self) -> [QM31; N_LANES] {
        let a = self.a().to_array();
        let b = self.b().to_array();
        array::from_fn(|i| QM31(a[i], b[i]))
    }

    pub fn from_array(values: [QM31; N_LANES]) -> Self {
        let a = values.map(|v| v.0);
        let b = values.map(|v| v.1);
        Self([PackedCM31::from_array(a), PackedCM31::from_array(b)])
    }

    /// Interleaves two vectors.
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let Self([a_evens, b_evens]) = self;
        let Self([a_odds, b_odds]) = other;
        let (a_lhs, a_rhs) = a_evens.interleave(a_odds);
        let (b_lhs, b_rhs) = b_evens.interleave(b_odds);
        (Self([a_lhs, b_lhs]), Self([a_rhs, b_rhs]))
    }

    /// Deinterleaves two vectors.
    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        let Self([a_lhs, b_lhs]) = self;
        let Self([a_rhs, b_rhs]) = other;
        let (a_evens, a_odds) = a_lhs.deinterleave(a_rhs);
        let (b_evens, b_odds) = b_lhs.deinterleave(b_rhs);
        (Self([a_evens, b_evens]), Self([a_odds, b_odds]))
    }

    /// Sums all the elements in the vector.
    pub fn pointwise_sum(self) -> QM31 {
        self.to_array().into_iter().sum()
    }

    /// Doubles each element in the vector.
    pub fn double(self) -> Self {
        let Self([a, b]) = self;
        Self([a.double(), b.double()])
    }

    /// Returns vectors `a, b, c, d` such that element `i` is represented as
    /// `QM31(a_i, b_i, c_i, d_i)`.
    pub const fn into_packed_m31s(self) -> [PackedM31; 4] {
        let Self([PackedCM31([a, b]), PackedCM31([c, d])]) = self;
        [a, b, c, d]
    }

    /// Creates an instance from vectors `a, b, c, d` such that element `i`
    /// is represented as `QM31(a_i, b_i, c_i, d_i)`.
    pub const fn from_packed_m31s([a, b, c, d]: [PackedM31; 4]) -> Self {
        Self([PackedCM31([a, b]), PackedCM31([c, d])])
    }
}

impl Add for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self.a() + rhs.a(), self.b() + rhs.b()])
    }
}

impl Sub for PackedQM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.a() - rhs.a(), self.b() - rhs.b()])
    }
}

impl Mul for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Compute using Karatsuba.
        //   (a + ub) * (c + ud) =
        //   (ac + (2+i)bd) + (ad + bc)u =
        //   ac + 2bd + ibd + (ad + bc)u.
        let ac = self.a() * rhs.a();
        let bd = self.b() * rhs.b();
        let bd_times_1_plus_i = PackedCM31([bd.a() - bd.b(), bd.a() + bd.b()]);
        // Computes ac + bd.
        let ac_p_bd = ac + bd;
        // Computes ad + bc.
        let ad_p_bc = (self.a() + self.b()) * (rhs.a() + rhs.b()) - ac_p_bd;
        // ac + 2bd + ibd =
        // ac + bd + bd + ibd
        let l = PackedCM31([
            ac_p_bd.a() + bd_times_1_plus_i.a(),
            ac_p_bd.b() + bd_times_1_plus_i.b(),
        ]);
        Self([l, ad_p_bc])
    }
}

impl Zero for PackedQM31 {
    fn zero() -> Self {
        Self([PackedCM31::zero(), PackedCM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

impl One for PackedQM31 {
    fn one() -> Self {
        Self([PackedCM31::one(), PackedCM31::zero()])
    }
}

impl AddAssign for PackedQM31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedQM31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl FieldExpOps for PackedQM31 {
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        // (a + bu)^-1 = (a - bu) / (a^2 - (2+i)b^2).
        let b2 = self.b().square();
        let ib2 = PackedCM31([-b2.b(), b2.a()]);
        let denom = self.a().square() - (b2 + b2 + ib2);
        let denom_inverse = denom.inverse();
        Self([self.a() * denom_inverse, -self.b() * denom_inverse])
    }

    fn batch_inverse(column: &[Self]) -> Vec<Self> {
        let mut result = unsafe { utils::uninit_vec(column.len()) };
        batch_inverse_chunked(column, &mut result, PACKED_QM31_BATCH_INVERSE_CHUNK_SIZE);
        result
    }
}

pub fn batch_inverse_packed_qm31(column: &[PackedQM31], dst: &mut [PackedQM31]) {
    assert!(column.len() <= dst.len());
    batch_inverse_chunked(column, dst, PACKED_QM31_BATCH_INVERSE_CHUNK_SIZE);
}

impl Add<PackedM31> for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: PackedM31) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Mul<PackedM31> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: PackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Mul<PackedCM31> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: PackedCM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Sub<PackedM31> for PackedQM31 {
    type Output = Self;

    fn sub(self, rhs: PackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a - rhs, b])
    }
}

impl Add<QM31> for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: QM31) -> Self::Output {
        self + PackedQM31::broadcast(rhs)
    }
}

impl Sub<QM31> for PackedQM31 {
    type Output = Self;

    fn sub(self, rhs: QM31) -> Self::Output {
        self - PackedQM31::broadcast(rhs)
    }
}

impl Mul<QM31> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: QM31) -> Self::Output {
        self * PackedQM31::broadcast(rhs)
    }
}

impl Mul<M31> for PackedQM31 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: M31) -> Self::Output {
        self * PackedM31::broadcast(rhs)
    }
}

impl Add<M31> for PackedQM31 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: M31) -> Self::Output {
        self + PackedM31::broadcast(rhs)
    }
}

impl SubAssign for PackedQM31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

unsafe impl Pod for PackedQM31 {}

unsafe impl Zeroable for PackedQM31 {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

impl Sum for PackedQM31 {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(Self::zero);
        iter.fold(first, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for PackedQM31 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.copied().sum()
    }
}

impl Neg for PackedQM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self([a, b]) = self;
        Self([-a, -b])
    }
}

impl Distribution<PackedQM31> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PackedQM31 {
        PackedQM31::from_array(rng.gen())
    }
}

impl From<PackedM31> for PackedQM31 {
    fn from(value: PackedM31) -> Self {
        PackedQM31::from_packed_m31s([
            value,
            PackedM31::zero(),
            PackedM31::zero(),
            PackedM31::zero(),
        ])
    }
}

impl From<QM31> for PackedQM31 {
    fn from(value: QM31) -> Self {
        PackedQM31::broadcast(value)
    }
}

/// Mask selecting the low 31 bits of a lane (i.e. `lo = (a·b) mod 2^31`).
const LOW_31_MASK_U64: u64 = (1 << 31) - 1;

/// Delayed-reduction accumulator for a `Σ coeff_j · value_j` dot product with `QM31` scalar
/// coefficients and [`PackedM31`] values, used by the R1 prover hot loops (the
/// `LookupElements::combine` SIMD path and the FRI quotient numerator loop).
///
/// A `QM31 × M31` product distributes coordinate-wise: the `c`-th `QM31` coordinate of
/// `coeff · value` is the `M31 × PackedM31` product `coeff.coord_c · value`. So a sum of such
/// products is four independent M31 dot products, one per coordinate. This struct holds the
/// four coordinate sums as unreduced per-lane `u64` accumulators and reduces them only at
/// [`Self::finalize`], rather than reducing each product as `PackedQM31 * PackedM31` would.
/// The result is the *identical* field element the naive reduced fold computes — a pure
/// re-association of exact integer arithmetic, no rounding or probabilistic argument.
///
/// ## Bound discipline (strategy 2: per-product partial reduction)
///
/// Each accumulated term is the Mersenne fold `hi + lo` of a single product. For
/// `coeff.coord_c ∈ [0, P)` (reduced `M31`) and `value ∈ [0, P]` (the `PackedM31`
/// representation invariant), the product is `< P² < 2^62`, so with `prod = hi·2^31 + lo`,
/// `hi, lo < 2^31` and `prod ≡ hi + lo (mod P)` with `hi + lo < 2^32`. After `k` accumulated
/// terms each lane is therefore `< k · 2^32`. [`Self::accumulate`] tracks `k` and
/// `debug_assert!`s it against [`DOT_DELAYED_MAX_LEN`] (= 2^30), keeping every lane `< 2^62`
/// — the range over which the final single-pass Mersenne reduction is exact. As with the
/// `PackedM31` ops, the per-coordinate output is in `[0, P]` (boundary-inclusive), matching
/// the naive path's representative.
pub(crate) struct PackedQM31DelayedDot {
    /// One unreduced lane accumulator per QM31 coordinate.
    coords: [Simd<u64, N_LANES>; 4],
    /// Number of products accumulated so far (the `k` in the bound `lane < k · 2^32`).
    n_terms: usize,
}

impl PackedQM31DelayedDot {
    #[inline]
    pub(crate) const fn new() -> Self {
        Self {
            coords: [Simd::from_array([0; N_LANES]); 4],
            n_terms: 0,
        }
    }

    /// Accumulates `coeff · value` (coordinate-wise) into the unreduced accumulators.
    #[inline]
    pub(crate) fn accumulate(&mut self, coeff_coords: &[M31; 4], value: PackedM31) {
        // Bound: each new term contributes `< 2^32` per lane; after this call the accumulator
        // has folded `n_terms + 1` products, so each lane is `< (n_terms + 1) · 2^32`. We cap
        // `n_terms` at DOT_DELAYED_MAX_LEN (= 2^30) so lanes stay `< 2^62`, where the final
        // Mersenne reduction is exact.
        self.n_terms += 1;
        debug_assert!(
            self.n_terms <= DOT_DELAYED_MAX_LEN,
            "delayed-reduction dot product exceeded DOT_DELAYED_MAX_LEN terms"
        );
        let value_u64 = value.into_simd().cast::<u64>();
        for (acc, coeff) in self.coords.iter_mut().zip(coeff_coords.iter()) {
            // coeff ∈ [0, P), value ∈ [0, P] ⇒ each lane product < P² < 2^62.
            let prod = Simd::<u64, N_LANES>::splat(coeff.0 as u64) * value_u64;
            // a·b = hi·2^31 + lo ≡ hi + lo (mod P), with hi + lo < 2^32.
            let lo = prod & Simd::splat(LOW_31_MASK_U64);
            let hi = prod >> Simd::splat(MODULUS_BITS as u64);
            *acc += hi + lo;
        }
    }

    /// Reduces the four coordinate accumulators once and assembles the result `PackedQM31`.
    #[inline]
    pub(crate) fn finalize(&self) -> PackedQM31 {
        PackedQM31::from_packed_m31s(self.coords.map(reduce_u64_lanes_to_m31))
    }
}

/// Dot product `Σ coeffs_j · values_j` where each `coeffs_j` is a scalar [`QM31`] constant
/// (broadcast across all lanes) and each `values_j` is a [`PackedM31`], with delayed modular
/// reduction. Each output coordinate is in the `PackedM31` `[0, P]` representation.
///
/// This is the SIMD specialization of the per-coordinate accumulation that
/// `LookupElements::combine` performs with `PackedQM31 * PackedM31` multiplies. See
/// [`PackedQM31DelayedDot`] for the coordinate decomposition and the accumulation bound. The
/// slices must have equal length, which must not exceed [`DOT_DELAYED_MAX_LEN`].
pub fn dot_delayed_qm31_scalar(coeffs: &[QM31], values: &[PackedM31]) -> PackedQM31 {
    assert_eq!(
        coeffs.len(),
        values.len(),
        "dot_delayed_qm31_scalar: length mismatch"
    );
    assert!(
        coeffs.len() <= DOT_DELAYED_MAX_LEN,
        "dot_delayed_qm31_scalar: length {} exceeds DOT_DELAYED_MAX_LEN",
        coeffs.len()
    );

    let mut dot = PackedQM31DelayedDot::new();
    for (coeff, val) in coeffs.iter().zip(values.iter()) {
        dot.accumulate(&coeff.to_m31_array(), *val);
    }
    dot.finalize()
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::prover::backend::simd::qm31::PackedQM31;

    #[test]
    fn addition_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedQM31::from_array(lhs);
        let packed_rhs = PackedQM31::from_array(rhs);

        let res = packed_lhs + packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn subtraction_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedQM31::from_array(lhs);
        let packed_rhs = PackedQM31::from_array(rhs);

        let res = packed_lhs - packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn multiplication_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedQM31::from_array(lhs);
        let packed_rhs = PackedQM31::from_array(rhs);

        let res = packed_lhs * packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn negation_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen();
        let packed_values = PackedQM31::from_array(values);

        let res = -packed_values;

        assert_eq!(res.to_array(), values.map(|v| -v));
    }

    // -- Delayed-reduction QM31-scalar dot product tests (R1) ----------------------------

    use std::simd::u32x16;

    use num_traits::Zero;

    use super::dot_delayed_qm31_scalar;
    use crate::core::fields::m31::P;
    use crate::core::fields::qm31::QM31;
    use crate::prover::backend::simd::m31::PackedM31;

    /// Builds a `PackedM31` from raw lane values, allowing the unreduced boundary value `P`.
    fn packed_from_raw(lanes: [u32; 16]) -> PackedM31 {
        unsafe { PackedM31::from_simd_unchecked(u32x16::from_array(lanes)) }
    }

    /// Reference: the naive reduced fold `Σ_j broadcast(coeff_j) * value_j` using the existing
    /// `PackedQM31 * PackedM31` multiply and `PackedQM31` add, reducing every product.
    fn naive_qm31_dot(coeffs: &[QM31], values: &[PackedM31]) -> PackedQM31 {
        coeffs
            .iter()
            .zip(values.iter())
            .fold(PackedQM31::zero(), |acc, (c, v)| {
                acc + PackedQM31::broadcast(*c) * *v
            })
    }

    #[test]
    fn dot_delayed_qm31_scalar_matches_naive_random() {
        let mut rng = SmallRng::seed_from_u64(21);
        for _ in 0..10_001 {
            let len = rng.gen_range(0..=130);
            let coeffs: Vec<QM31> = (0..len).map(|_| rng.gen()).collect();
            let values: Vec<PackedM31> = (0..len).map(|_| rng.gen()).collect();
            assert_eq!(
                dot_delayed_qm31_scalar(&coeffs, &values).to_array(),
                naive_qm31_dot(&coeffs, &values).to_array(),
                "len={len}"
            );
        }
    }

    /// Boundary `PackedM31` lanes (including the unreduced `P`) at the fold-boundary lengths
    /// 0..=5 and the largest real call-site length (96) + 1.
    #[test]
    fn dot_delayed_qm31_scalar_edge_lanes_and_lengths() {
        const EDGE: [u32; 6] = [0, 1, 2, (1 << 30) - 1, P - 1, P];
        let mut rng = SmallRng::seed_from_u64(22);
        let mixed =
            |seed: usize| -> [u32; 16] { array::from_fn(|lane| EDGE[(lane + seed) % EDGE.len()]) };

        for len in [0usize, 1, 2, 3, 4, 5, 97] {
            let coeffs: Vec<QM31> = (0..len).map(|_| rng.gen()).collect();
            let values: Vec<PackedM31> = (0..len).map(|j| packed_from_raw(mixed(j))).collect();
            assert_eq!(
                dot_delayed_qm31_scalar(&coeffs, &values).to_array(),
                naive_qm31_dot(&coeffs, &values).to_array(),
                "len={len}"
            );
        }
    }

    #[test]
    fn dot_delayed_qm31_scalar_empty_is_zero() {
        assert_eq!(
            dot_delayed_qm31_scalar(&[], &[]).to_array(),
            PackedQM31::zero().to_array()
        );
    }
}
