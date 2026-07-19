use stwo_backend_cuda_kernels::raw::{
    CudaQuotientNativeRunEntry, CudaQuotientNativeRunManifest, STWO_QUOTIENT_NATIVE_RUN_MAX_RUNS,
};

const TARGET_LOG: u32 = 23;
const TARGET_ROWS: u32 = 1 << TARGET_LOG;
const BLOCK_THREADS: u32 = 256;
const RUNS: [(u32, u32); 18] = [
    (4, 22),
    (6, 41),
    (7, 7),
    (8, 21),
    (10, 248),
    (11, 315),
    (12, 34),
    (13, 399),
    (14, 39),
    (15, 10),
    (16, 732),
    (17, 29),
    (18, 543),
    (19, 1_293),
    (20, 1_225),
    (21, 620),
    (22, 140),
    (23, 167),
];
const EXPECTED_OFFSETS: [u32; 17] = [
    0, 16, 80, 208, 464, 1_488, 3_536, 7_632, 15_824, 32_208, 64_976, 130_512, 261_584, 523_728,
    1_048_016, 2_096_592, 4_193_744,
];

fn source_row(row: u32, source_log: u32) -> u32 {
    let distance = TARGET_LOG - source_log;
    (row >> (distance + 1) << 1) + (row & 1)
}

fn sealed_manifest() -> CudaQuotientNativeRunManifest {
    let mut manifest = CudaQuotientNativeRunManifest {
        run_count: 17,
        direct_term_begin: 5_718,
        direct_term_end: 5_885,
        target_log_size: TARGET_LOG,
        ..Default::default()
    };
    let mut term_begin = 0;
    for (index, ((source_log, terms), offset)) in
        RUNS.iter().take(17).zip(EXPECTED_OFFSETS).enumerate()
    {
        manifest.runs[index] = CudaQuotientNativeRunEntry {
            term_begin,
            term_end: term_begin + terms,
            source_log_size: *source_log,
            scratch_offset_words: offset,
        };
        term_begin += terms;
    }
    manifest
}

#[test]
fn sealed_manifest_is_canonical_compact_and_zero_padded() {
    let manifest = sealed_manifest();
    assert_eq!(core::mem::size_of_val(&manifest), 400);
    assert_eq!(
        core::mem::size_of_val(&manifest) + 12 * core::mem::size_of::<*const u32>(),
        496
    );
    assert_eq!(manifest.runs[0].term_begin, 0);
    assert_eq!(
        manifest.runs[manifest.run_count as usize - 1].term_end,
        manifest.direct_term_begin
    );

    let mut expected_offset = 0;
    let mut previous_end = 0;
    let mut previous_log = 0;
    for (index, run) in manifest.runs[..manifest.run_count as usize]
        .iter()
        .enumerate()
    {
        assert!(run.term_begin < run.term_end);
        assert_eq!(run.scratch_offset_words, expected_offset);
        if index != 0 {
            assert_eq!(run.term_begin, previous_end);
            assert!(run.source_log_size > previous_log);
        }
        expected_offset += 1 << run.source_log_size;
        previous_end = run.term_end;
        previous_log = run.source_log_size;
    }
    assert_eq!(expected_offset, 8_388_048);
    assert_eq!(TARGET_ROWS - expected_offset, 560);
    assert!(manifest.runs[manifest.run_count as usize..]
        .iter()
        .all(|run| *run == CudaQuotientNativeRunEntry::default()));
    assert_eq!(manifest.runs.len(), STWO_QUOTIENT_NATIVE_RUN_MAX_RUNS);
}

#[test]
fn paired_expansion_owns_every_target_row_and_maps_inside_each_run() {
    let manifest = sealed_manifest();
    let half_rows = TARGET_ROWS / 2;
    let blocks = half_rows.div_ceil(BLOCK_THREADS);
    assert_eq!(blocks * BLOCK_THREADS, half_rows);

    for owner in [0, 1, BLOCK_THREADS - 1, half_rows / 2, half_rows - 1] {
        let rows = [owner, owner + half_rows];
        assert!(rows[0] < half_rows);
        assert!(rows[1] >= half_rows);
        assert!(rows[1] < TARGET_ROWS);
        for run in &manifest.runs[..manifest.run_count as usize] {
            for row in rows {
                let mapped = source_row(row, run.source_log_size);
                assert!(mapped < 1 << run.source_log_size);
                assert!(
                    run.scratch_offset_words + mapped
                        < run.scratch_offset_words + (1 << run.source_log_size)
                );
                assert!(run.scratch_offset_words + mapped < 8_388_048);
            }
        }
    }
}

#[test]
fn sn3_group0_work_reduction_model_is_exact() {
    let baseline_row_terms =
        u64::from(RUNS.iter().map(|run| run.1).sum::<u32>()) * u64::from(TARGET_ROWS);
    let native_precompute_row_terms = RUNS
        .iter()
        .take(17)
        .map(|&(source_log, terms)| u64::from(terms) * (1u64 << source_log))
        .sum::<u64>();
    let direct_tail_row_terms = u64::from(RUNS[17].1) * u64::from(TARGET_ROWS);
    let expansion_run_loads = 17 * u64::from(TARGET_ROWS);
    let candidate_products = native_precompute_row_terms + direct_tail_row_terms;
    let candidate_add_units = candidate_products + expansion_run_loads;

    assert_eq!(baseline_row_terms, 49_366_958_080);
    assert_eq!(native_precompute_row_terms, 4_049_247_264);
    assert_eq!(direct_tail_row_terms, 1_400_897_536);
    assert_eq!(expansion_run_loads, 142_606_336);
    assert_eq!(candidate_products, 5_450_144_800);
    assert_eq!(candidate_add_units, 5_592_751_136);
    assert_eq!(baseline_row_terms - candidate_products, 43_916_813_280);
    assert_eq!(baseline_row_terms - candidate_add_units, 43_774_206_944);
    assert!(baseline_row_terms * 1_000 > candidate_add_units * 8_826);
    assert!(baseline_row_terms * 1_000 < candidate_add_units * 8_827);
}
