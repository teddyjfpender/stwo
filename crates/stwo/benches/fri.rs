use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::poly::line::LineDomain;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::CpuBackend;
use stwo::prover::fri::FriOps;
use stwo::prover::line::LineEvaluation;
use stwo::prover::poly::circle::{PolyOps, SecureEvaluation};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;

fn folding_benchmark(c: &mut Criterion) {
    const LOG_SIZE: u32 = 12;
    let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
    let evals = LineEvaluation::new(
        domain,
        SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                vec![BaseField::from_u32_unchecked(i as u32); 1 << LOG_SIZE]
            }),
        },
    );
    let alpha = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);
    let twiddles = CpuBackend::precompute_twiddles(domain.coset());
    c.bench_function("fold_line", |b| {
        b.iter(|| {
            black_box(CpuBackend::fold_line(
                black_box(&evals),
                black_box(&[alpha]),
                &twiddles,
            ));
        })
    });
}

/// Exercises the SIMD chunked fold path (`FOLD_CHUNK_SIZE`). The CPU `fold_line` bench above does
/// not touch that constant.
fn simd_folding_benchmark(c: &mut Criterion) {
    // Representative of the FRI fold input on a blake 2^18 proof (lifting_log_size ~ 18-20).
    const LOG_SIZE: u32 = 20;
    let alpha = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);

    // SIMD fold_line.
    let line_domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
    let line_evals = LineEvaluation::<SimdBackend>::new(
        line_domain,
        SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                (0..1 << LOG_SIZE)
                    .map(|_| BaseField::from_u32_unchecked(i as u32))
                    .collect()
            }),
        },
    );
    let line_twiddles = SimdBackend::precompute_twiddles(line_domain.coset());
    c.bench_function(&format!("simd_fold_line 2^{LOG_SIZE}"), |b| {
        b.iter(|| {
            black_box(SimdBackend::fold_line(
                black_box(&line_evals),
                black_box(&[alpha]),
                &line_twiddles,
            ));
        })
    });

    // SIMD fold_circle_into_line.
    let circle_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let circle_evals = SecureEvaluation::<SimdBackend, BitReversedOrder>::new(
        circle_domain,
        SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                (0..1 << LOG_SIZE)
                    .map(|_| BaseField::from_u32_unchecked(i as u32))
                    .collect()
            }),
        },
    );
    let circle_line_domain = LineDomain::new(circle_domain.half_coset);
    let circle_twiddles = SimdBackend::precompute_twiddles(circle_line_domain.coset());
    c.bench_function(&format!("simd_fold_circle_into_line 2^{LOG_SIZE}"), |b| {
        b.iter(|| {
            black_box(SimdBackend::fold_circle_into_line(
                black_box(&circle_evals),
                black_box(alpha),
                &circle_twiddles,
            ));
        })
    });
}

criterion_group!(benches, folding_benchmark, simd_folding_benchmark);
criterion_main!(benches);
