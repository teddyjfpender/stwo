use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use super::cm31::PackedCM31;
use super::m31::{PackedM31, DOT_DELAYED_MAX_LEN, N_LANES};
use super::qm31::{PackedQM31, PackedQM31DelayedDot};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::fields::{batch_inverse_in_place, FieldExpOps};

pub const LOG_N_VERY_PACKED_ELEMS: u32 = 1;
pub const N_VERY_PACKED_ELEMS: usize = 1 << LOG_N_VERY_PACKED_ELEMS;

#[derive(Clone, Debug, Copy)]
#[repr(transparent)]
pub struct Vectorized<A: Copy, const N: usize>(pub [A; N]);

impl<A: Copy, const N: usize> Vectorized<A, N> {
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> A,
    {
        Vectorized(std::array::from_fn(cb))
    }
}

impl<A: Copy, const N: usize> From<[A; N]> for Vectorized<A, N> {
    fn from(array: [A; N]) -> Self {
        Vectorized(array)
    }
}

unsafe impl<A: Copy, const N: usize> Zeroable for Vectorized<A, N> {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

unsafe impl<A: Pod, const N: usize> Pod for Vectorized<A, N> {}

pub type VeryPackedM31 = Vectorized<PackedM31, N_VERY_PACKED_ELEMS>;
pub type VeryPackedCM31 = Vectorized<PackedCM31, N_VERY_PACKED_ELEMS>;
pub type VeryPackedQM31 = Vectorized<PackedQM31, N_VERY_PACKED_ELEMS>;
pub type VeryPackedBaseField = VeryPackedM31;
pub type VeryPackedSecureField = VeryPackedQM31;

impl VeryPackedM31 {
    pub fn broadcast(value: M31) -> Self {
        Self::from_fn(|_| PackedM31::broadcast(value))
    }

    pub fn from_array(values: [M31; N_LANES * N_VERY_PACKED_ELEMS]) -> VeryPackedM31 {
        Self::from_fn(|i| {
            let start = i * N_LANES;
            let end = start + N_LANES;
            PackedM31::from_array(values[start..end].try_into().unwrap())
        })
    }

    pub fn to_array(&self) -> [M31; N_LANES * N_VERY_PACKED_ELEMS] {
        // Safety: We are transmuting &[A; N_VERY_PACKED_ELEMS] into &[i32; N_LANES *
        // N_VERY_PACKED_ELEMS] because we know that A contains [i32; N_LANES] and the
        // memory layout is contiguous.
        unsafe {
            std::slice::from_raw_parts(self.0.as_ptr() as *const M31, N_LANES * N_VERY_PACKED_ELEMS)
                .try_into()
                .unwrap()
        }
    }
}

impl VeryPackedCM31 {
    pub fn broadcast(value: CM31) -> Self {
        Self::from_fn(|_| PackedCM31::broadcast(value))
    }
}

impl VeryPackedQM31 {
    pub fn broadcast(value: QM31) -> Self {
        Self::from_fn(|_| PackedQM31::broadcast(value))
    }

    pub fn from_very_packed_m31s([a, b, c, d]: [VeryPackedM31; 4]) -> Self {
        Self::from_fn(|i| PackedQM31::from_packed_m31s([a.0[i], b.0[i], c.0[i], d.0[i]]))
    }

    pub fn into_very_packed_m31s(self) -> [VeryPackedM31; 4] {
        std::array::from_fn(|i| VeryPackedM31::from(self.0.map(|v| v.into_packed_m31s()[i])))
    }
}

/// Dot product `Σ coeffs_j · values_j` where each `coeffs_j` is a scalar [`QM31`] constant
/// (broadcast across all lanes) and each `values_j` is a [`VeryPackedM31`], with delayed
/// modular reduction. Each output coordinate is in the `PackedM31` `[0, P]` representation.
///
/// # Why this exists (R1 follow-up)
///
/// This is the `VeryPacked` analogue of [`super::qm31::dot_delayed_qm31_scalar`]. The
/// composition phase calls `LookupElements::combine` through `SimdDomainEvaluator`, whose
/// associated types are `VeryPackedM31` / `VeryPackedQM31`; the `(PackedM31, PackedQM31)`
/// specialization does not fire there. This function gives that path the same
/// delayed-reduction fast path.
///
/// # Shape: per-sub-lane reuse, not new math
///
/// `VeryPackedM31 = Vectorized<PackedM31, N_VERY_PACKED_ELEMS>` is simply `N_VERY_PACKED_ELEMS`
/// (= 2) independent `PackedM31` sub-lanes, and `VeryPackedQM31` is likewise
/// `N_VERY_PACKED_ELEMS` independent `PackedQM31` sub-lanes. The generic `VeryPacked` fold
/// computes the result sub-lane `i` purely from input sub-lane `i` (`Vectorized` ops are
/// element-wise — see the `Mul`/`Add` impls below). So we keep one
/// [`PackedQM31DelayedDot`] per sub-lane and route sub-lane `i` of every term into
/// accumulator `i`, then finalize each and reassemble. There is **no new arithmetic**: each
/// sub-lane is exactly the existing `PackedM31`-granularity delayed dot, so the soundness and
/// bound analysis of [`PackedQM31DelayedDot`] applies unchanged, per sub-lane.
///
/// # Bound
///
/// Unchanged from [`PackedQM31DelayedDot`]: the bound is per `PackedM31` sub-lane accumulator
/// and depends only on the number of terms `k` (`lane < k · 2^32`), not on the `VeryPacked`
/// width. `k = coeffs.len()`, capped by [`DOT_DELAYED_MAX_LEN`] and `debug_assert!`ed inside
/// each sub-lane accumulator's `accumulate`. The single reduction per coordinate per sub-lane
/// happens in `finalize`, exactly as in the `Packed` case, and yields the *identical* field
/// element the generic `VeryPacked` fold produces (a re-association of exact integer
/// arithmetic).
///
/// # Stack frame / rayon (R1 round-1 lesson, binding)
///
/// This is marked `#[inline(never)]`. Composition's `SimdDomainEvaluator` runs inside a rayon
/// `par_chunks_mut` closure (see `component_prover.rs`), and `combine` is called many times
/// per row from that closure. The accumulator array here is `N_VERY_PACKED_ELEMS`
/// `PackedQM31DelayedDot`s (4 coords × 2 sub-lanes × `u64x16` ≈ 1 KiB). Forcing this off the
/// inline path keeps that ~1 KiB frame from being inlined (and potentially replicated) into
/// the composition closure's frame, mirroring the `accumulate_numerators_chunk` fix.
///
/// The slices must have equal length, which must not exceed [`DOT_DELAYED_MAX_LEN`].
#[inline(never)]
pub fn dot_delayed_very_packed_qm31_scalar(
    coeffs: &[QM31],
    values: &[VeryPackedM31],
) -> VeryPackedQM31 {
    assert_eq!(
        coeffs.len(),
        values.len(),
        "dot_delayed_very_packed_qm31_scalar: length mismatch"
    );
    assert!(
        coeffs.len() <= DOT_DELAYED_MAX_LEN,
        "dot_delayed_very_packed_qm31_scalar: length {} exceeds DOT_DELAYED_MAX_LEN",
        coeffs.len()
    );

    // One delayed-reduction accumulator per `VeryPacked` sub-lane. Each is independent and
    // sees only its own sub-lane's `PackedM31` values, exactly as the generic element-wise
    // `Vectorized` fold would; this is pure per-sub-lane reuse of the `Packed` primitive.
    let mut dots = [(); N_VERY_PACKED_ELEMS].map(|()| PackedQM31DelayedDot::new());
    for (coeff, value) in coeffs.iter().zip(values.iter()) {
        let coeff_coords = coeff.to_m31_array();
        for (dot, sub_lane) in dots.iter_mut().zip(value.0.iter()) {
            dot.accumulate(&coeff_coords, *sub_lane);
        }
    }
    VeryPackedQM31::from_fn(|i| dots[i].finalize())
}

impl From<M31> for VeryPackedM31 {
    fn from(v: M31) -> Self {
        Self::broadcast(v)
    }
}

impl From<VeryPackedM31> for VeryPackedQM31 {
    fn from(value: VeryPackedM31) -> Self {
        VeryPackedQM31::from_very_packed_m31s([
            value,
            VeryPackedM31::zero(),
            VeryPackedM31::zero(),
            VeryPackedM31::zero(),
        ])
    }
}

impl From<QM31> for VeryPackedQM31 {
    fn from(value: QM31) -> Self {
        VeryPackedQM31::broadcast(value)
    }
}

trait Scalar {}
impl Scalar for M31 {}
impl Scalar for CM31 {}
impl Scalar for QM31 {}
impl Scalar for PackedM31 {}
impl Scalar for PackedCM31 {}
impl Scalar for PackedQM31 {}

impl<A: Add<B> + Copy, B: Copy, const N: usize> Add<Vectorized<B, N>> for Vectorized<A, N>
where
    <A as Add<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn add(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] + other.0[i])
    }
}

impl<A: Add<B> + Copy, B: Scalar + Copy, const N: usize> Add<B> for Vectorized<A, N>
where
    <A as Add<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn add(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] + other)
    }
}

impl<A: Sub<B> + Copy, B: Copy, const N: usize> Sub<Vectorized<B, N>> for Vectorized<A, N>
where
    <A as Sub<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn sub(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] - other.0[i])
    }
}

impl<A: Sub<B> + Copy, B: Scalar + Copy, const N: usize> Sub<B> for Vectorized<A, N>
where
    <A as Sub<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn sub(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] - other)
    }
}

impl<A: Mul<B> + Copy, B: Copy, const N: usize> Mul<Vectorized<B, N>> for Vectorized<A, N>
where
    <A as Mul<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn mul(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] * other.0[i])
    }
}

impl<A: Mul<B> + Copy, B: Scalar + Copy, const N: usize> Mul<B> for Vectorized<A, N>
where
    <A as Mul<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn mul(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] * other)
    }
}

impl<A: AddAssign<B> + Copy, B: Copy, const N: usize> AddAssign<Vectorized<B, N>>
    for Vectorized<A, N>
{
    fn add_assign(&mut self, other: Vectorized<B, N>) {
        for i in 0..N {
            self.0[i] += other.0[i];
        }
    }
}

impl<A: AddAssign<B> + Copy, B: Scalar + Copy, const N: usize> AddAssign<B> for Vectorized<A, N> {
    fn add_assign(&mut self, other: B) {
        for i in 0..N {
            self.0[i] += other;
        }
    }
}

impl<A: MulAssign<B> + Copy, B: Copy, const N: usize> MulAssign<Vectorized<B, N>>
    for Vectorized<A, N>
{
    fn mul_assign(&mut self, other: Vectorized<B, N>) {
        for i in 0..N {
            self.0[i] *= other.0[i];
        }
    }
}

impl<A: Neg + Copy, const N: usize> Neg for Vectorized<A, N>
where
    <A as Neg>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i].neg())
    }
}

impl<A: Zero + Copy, const N: usize> Zero for Vectorized<A, N> {
    fn zero() -> Self {
        Vectorized::from_fn(|_| A::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(A::is_zero)
    }
}

impl<A: One + Copy, const N: usize> One for Vectorized<A, N> {
    fn one() -> Self {
        Vectorized::from_fn(|_| A::one())
    }
}

impl<A: FieldExpOps + Zero + Copy, const N: usize> FieldExpOps for Vectorized<A, N> {
    fn inverse(&self) -> Self {
        let mut dst = [A::zero(); N];
        batch_inverse_in_place(&self.0, &mut dst);
        dst.into()
    }
}
