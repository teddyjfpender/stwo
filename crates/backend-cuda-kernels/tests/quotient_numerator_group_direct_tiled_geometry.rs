const ROWS_PER_CTA: u32 = 512;
const THREADS: u32 = 256;
const STAGE_DISTANCE_MIN: u32 = 3;
const MAX_TERMS_PER_BATCH: u32 = 1024;

fn source_row(row: u32, group_log: u32, source_log: u32) -> u32 {
    let distance = group_log - source_log;
    (row >> (distance + 1) << 1) + (row & 1)
}

fn unique_rows(distance: u32) -> u32 {
    (ROWS_PER_CTA >> distance).max(2)
}

fn batch_capacity(tile_words: u32, distance: u32) -> u32 {
    (tile_words / unique_rows(distance)).min(MAX_TERMS_PER_BATCH)
}

fn should_stage(group_log: u32, source_log: u32) -> bool {
    group_log >= 9 && group_log - source_log >= STAGE_DISTANCE_MIN
}

#[test]
fn contiguous_two_row_owners_cover_every_row_without_a_tail() {
    for group_log in 9..=30 {
        let row_count = 1u64 << group_log;
        let blocks = row_count / u64::from(ROWS_PER_CTA);
        assert_eq!(blocks * u64::from(ROWS_PER_CTA), row_count);
        for block in [0, 1.min(blocks - 1), blocks / 2, blocks - 1] {
            let base = block * u64::from(ROWS_PER_CTA);
            let rows = (0..THREADS)
                .flat_map(|thread| [base + u64::from(thread), base + u64::from(thread + THREADS)])
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), ROWS_PER_CTA as usize);
            assert_eq!(rows.iter().copied().min(), Some(base));
            assert_eq!(
                rows.iter().copied().max(),
                Some(base + u64::from(ROWS_PER_CTA) - 1)
            );
            let mut sorted = rows;
            sorted.sort_unstable();
            assert!(sorted.into_iter().eq(base..base + u64::from(ROWS_PER_CTA)));
        }
    }
}

#[test]
fn staged_source_interval_is_exact_across_boundary_logs_and_last_tiles() {
    for group_log in 9..=30 {
        let blocks = (1u64 << group_log) / u64::from(ROWS_PER_CTA);
        for source_log in 0..group_log - 2 {
            let distance = group_log - source_log;
            assert!(should_stage(group_log, source_log));
            let unique = unique_rows(distance);
            let product_batch_terms = batch_capacity(1024, distance);
            assert!(product_batch_terms > 0);
            assert!(product_batch_terms * unique <= 1024);
            for block in [0, 1.min(blocks - 1), blocks / 2, blocks - 1] {
                let base = u32::try_from(block * u64::from(ROWS_PER_CTA)).unwrap();
                let source_base = source_row(base, group_log, source_log);
                for local in 0..ROWS_PER_CTA {
                    let mapped = source_row(base + local, group_log, source_log);
                    assert!(mapped < 1u32 << source_log.max(1));
                    assert!(mapped >= source_base);
                    assert!(mapped - source_base < unique);
                    assert_eq!(source_base + (mapped - source_base), mapped);
                }
            }
        }
        for source_log in group_log - 2..=group_log {
            assert!(!should_stage(group_log, source_log));
        }
    }
    assert!(!(0..=8).any(|group_log| should_stage(group_log, 0)));
}

#[test]
fn adaptive_sn3_batch_counts_and_logical_load_model_are_sealed() {
    let runs = [
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
        (19, 1293),
        (20, 1225),
        (21, 620),
        (22, 140),
        (23, 159),
    ];
    let batches = |tile_words| {
        runs.iter()
            .filter_map(|&(source_log, terms)| {
                let distance = 23 - source_log;
                (distance >= STAGE_DISTANCE_MIN).then(|| {
                    let capacity = batch_capacity(tile_words, distance);
                    (terms + capacity - 1) / capacity
                })
            })
            .sum::<u32>()
    };
    assert_eq!(batches(1024), 141);
    assert_eq!(batches(4096), 46);
    assert_eq!([batches(1024) * 2, batches(4096) * 2], [282, 92]);

    let candidate_loads_per_cta = runs
        .iter()
        .map(|&(source_log, terms)| {
            let distance = 23 - source_log;
            let rows = if distance >= STAGE_DISTANCE_MIN {
                unique_rows(distance)
            } else {
                ROWS_PER_CTA
            };
            terms * rows
        })
        .sum::<u32>();
    assert_eq!(candidate_loads_per_cta, 604_424);
    assert_eq!(5_877 * ROWS_PER_CTA, 3_009_024);
    assert_eq!(
        u64::from(candidate_loads_per_cta) * 16_384 * 4,
        39_611_531_264
    );

    // The product arm has 1,024 aligned uint4 entries (16 KiB). It therefore
    // has the raw 4-KiB arm's batching geometry, but performs each four-limb
    // scalar product once per unique (term, source row), not once per output.
    assert_eq!(batches(1024), 141);
    assert_eq!(batches(1024) * 2, 282);
    assert_eq!(5_877 * ROWS_PER_CTA * 4, 12_036_096);
    assert_eq!(candidate_loads_per_cta * 4, 2_417_696);
}
