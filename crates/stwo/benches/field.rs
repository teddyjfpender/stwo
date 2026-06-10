use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_traits::{One, Zero};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SecureField, QM31};
use stwo::prover::backend::simd::m31::{dot_delayed, dot_delayed_scalar, PackedBaseField, N_LANES};
use stwo::prover::backend::simd::qm31::{dot_delayed_qm31_scalar, PackedQM31};
use stwo::prover::backend::simd::very_packed_m31::{
    dot_delayed_very_packed_qm31_scalar, VeryPackedM31, VeryPackedQM31, N_VERY_PACKED_ELEMS,
};

pub const N_ELEMENTS: usize = 1 << 16;
pub const N_STATE_ELEMENTS: usize = 8;

pub fn m31_operations_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<M31> = (0..N_ELEMENTS).map(|_| rng.gen()).collect();
    let mut state: [M31; N_STATE_ELEMENTS] = rng.gen();

    c.bench_function("M31 mul", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("M31 add", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem += *elem;
                    }
                }
            }
        })
    });
}

pub fn cm31_operations_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<CM31> = (0..N_ELEMENTS).map(|_| rng.gen()).collect();
    let mut state: [CM31; N_STATE_ELEMENTS] = rng.gen();

    c.bench_function("CM31 mul", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("CM31 add", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem += *elem;
                    }
                }
            }
        })
    });
}

pub fn qm31_operations_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<SecureField> = (0..N_ELEMENTS).map(|_| rng.gen()).collect();
    let mut state: [SecureField; N_STATE_ELEMENTS] = rng.gen();

    c.bench_function("SecureField mul", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("SecureField add", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem += *elem;
                    }
                }
            }
        })
    });
}

pub fn simd_m31_operations_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<PackedBaseField> = (0..N_ELEMENTS / N_LANES).map(|_| rng.gen()).collect();
    let mut states = vec![PackedBaseField::broadcast(BaseField::one()); N_STATE_ELEMENTS];

    c.bench_function("mul_simd", |b| {
        b.iter(|| {
            for elem in elements.iter() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        *state *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("add_simd", |b| {
        b.iter(|| {
            for elem in elements.iter() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        *state += *elem;
                    }
                }
            }
        })
    });

    c.bench_function("sub_simd", |b| {
        b.iter(|| {
            for elem in elements.iter() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        *state -= *elem;
                    }
                }
            }
        })
    });
}

/// Microbench for the R1 delayed-reduction dot products (naive reduced fold vs delayed), at
/// dot lengths 4 / 16 / 64. Tracks the kernel win independently of e2e noise.
pub fn dot_delayed_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    const LENGTHS: [usize; 3] = [4, 16, 64];

    // `dot_delayed`: both sides PackedM31.
    {
        let mut group = c.benchmark_group("dot_delayed");
        for &len in &LENGTHS {
            let a: Vec<PackedBaseField> = (0..len).map(|_| rng.gen()).collect();
            let b: Vec<PackedBaseField> = (0..len).map(|_| rng.gen()).collect();
            group.bench_with_input(BenchmarkId::new("naive", len), &len, |bn, _| {
                bn.iter(|| {
                    let mut acc = PackedBaseField::zero();
                    for (x, y) in a.iter().zip(b.iter()) {
                        acc += *x * *y;
                    }
                    acc
                })
            });
            group.bench_with_input(BenchmarkId::new("delayed", len), &len, |bn, _| {
                bn.iter(|| dot_delayed(&a, &b))
            });
        }
        group.finish();
    }

    // `dot_delayed_scalar`: scalar M31 coeffs against PackedM31 values.
    {
        let mut group = c.benchmark_group("dot_delayed_scalar");
        for &len in &LENGTHS {
            let coeffs: Vec<M31> = (0..len).map(|_| rng.gen()).collect();
            let values: Vec<PackedBaseField> = (0..len).map(|_| rng.gen()).collect();
            group.bench_with_input(BenchmarkId::new("naive", len), &len, |bn, _| {
                bn.iter(|| {
                    let mut acc = PackedBaseField::zero();
                    for (cf, v) in coeffs.iter().zip(values.iter()) {
                        acc += *v * *cf;
                    }
                    acc
                })
            });
            group.bench_with_input(BenchmarkId::new("delayed", len), &len, |bn, _| {
                bn.iter(|| dot_delayed_scalar(&coeffs, &values))
            });
        }
        group.finish();
    }

    // `dot_delayed_qm31_scalar`: scalar QM31 coeffs against PackedM31 values (the
    // `LookupElements::combine` / quotient-numerator hot loop shape).
    {
        let mut group = c.benchmark_group("dot_delayed_qm31_scalar");
        for &len in &LENGTHS {
            let coeffs: Vec<QM31> = (0..len).map(|_| rng.gen()).collect();
            let values: Vec<PackedBaseField> = (0..len).map(|_| rng.gen()).collect();
            group.bench_with_input(BenchmarkId::new("naive", len), &len, |bn, _| {
                bn.iter(|| {
                    let mut acc = PackedQM31::zero();
                    for (cf, v) in coeffs.iter().zip(values.iter()) {
                        acc += PackedQM31::broadcast(*cf) * *v;
                    }
                    acc
                })
            });
            group.bench_with_input(BenchmarkId::new("delayed", len), &len, |bn, _| {
                bn.iter(|| dot_delayed_qm31_scalar(&coeffs, &values))
            });
        }
        group.finish();
    }

    // `dot_delayed_very_packed_qm31_scalar`: scalar QM31 coeffs against VeryPackedM31 values
    // (the composition-phase `LookupElements::combine` hot loop shape, via
    // `SimdDomainEvaluator`'s `VeryPacked` associated types).
    {
        let mut group = c.benchmark_group("dot_delayed_very_packed_qm31_scalar");
        for &len in &LENGTHS {
            let coeffs: Vec<QM31> = (0..len).map(|_| rng.gen()).collect();
            let values: Vec<VeryPackedM31> = (0..len)
                .map(|_| {
                    VeryPackedM31::from(core::array::from_fn::<
                        PackedBaseField,
                        N_VERY_PACKED_ELEMS,
                        _,
                    >(|_| rng.gen()))
                })
                .collect();
            group.bench_with_input(BenchmarkId::new("naive", len), &len, |bn, _| {
                bn.iter(|| {
                    let mut acc = VeryPackedQM31::zero();
                    for (cf, v) in coeffs.iter().zip(values.iter()) {
                        acc += VeryPackedQM31::broadcast(*cf) * *v;
                    }
                    acc
                })
            });
            group.bench_with_input(BenchmarkId::new("delayed", len), &len, |bn, _| {
                bn.iter(|| dot_delayed_very_packed_qm31_scalar(&coeffs, &values))
            });
        }
        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = m31_operations_bench, cm31_operations_bench, qm31_operations_bench,
        simd_m31_operations_bench, dot_delayed_bench);
criterion_main!(benches);
