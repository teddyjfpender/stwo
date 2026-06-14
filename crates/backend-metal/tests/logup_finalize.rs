use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::Column;
use stwo_backend_metal::finalize_raw_logup;
use stwo_backend_metal_sys::metal::{metal_runtime_support, MetalRuntimeSupport};
use stwo_constraint_framework::RawLogupTraceGenerator;

fn require_metal() -> bool {
    if metal_runtime_support() == MetalRuntimeSupport::Available {
        return true;
    }
    eprintln!("skipping Metal logup finalize differential: Metal runtime unavailable");
    false
}

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

#[test]
fn finalize_raw_logup_matches_simd() {
    if !require_metal() {
        return;
    }

    for (log_size, n_cols, seed) in [(8u32, 3usize, 0u64), (12, 5, 1), (14, 1, 2)] {
        let simd = random_raw(log_size, n_cols, seed).into_raw();
        let metal = random_raw(log_size, n_cols, seed).into_raw();

        let (simd_trace, simd_sum) = simd.finalize_on_simd();
        let (metal_trace, metal_sum) = finalize_raw_logup(metal);

        assert_eq!(simd_sum, metal_sum, "claimed sum log_size={log_size}");
        assert_eq!(simd_trace.len(), metal_trace.len());
        for (i, (s, m)) in simd_trace.iter().zip(metal_trace.iter()).enumerate() {
            assert_eq!(
                s.values.to_cpu(),
                m.values.to_cpu(),
                "column {i} log_size={log_size}"
            );
        }
    }
}
