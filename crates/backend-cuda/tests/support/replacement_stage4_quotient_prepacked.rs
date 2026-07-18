use std::collections::BTreeMap;

use stwo::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
use stwo::core::fields::qm31::SecureField;
use stwo_backend_cuda::{
    quotient_numerator_prepacked_plan_identity, quotient_numerator_prepacked_term_layout,
    quotient_numerator_staged_single_write_plan_with_overflow_capacities,
    quotient_numerator_workspace_requirements, DeviceArena, PreparedNumeratorSchedule,
    PreparedQuotientNumeratorError, PreparedQuotientNumeratorGraph, QuotientNumeratorColumn,
    QuotientNumeratorDestination, QuotientNumeratorPrepackedStatusCode,
    QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceRequirements,
    QuotientNumeratorWorkspaceSlots,
};

#[cfg(stwo_cuda_link)]
use super::replacement_stage4_common::{cuda_event_timings, PerformanceReceipt};
use super::replacement_stage4_common::{
    fill, hash_words, read_words, upload_words, FixtureReceipt,
};
use super::replacement_stage4_quotient as quotient;

const OVERFLOW_WORDS: usize = 128;
const STALE_OUTPUT_BYTE: u8 = 0x5a;

pub fn run() -> FixtureReceipt {
    let config = config(10);
    let points = points();
    let topology = quotient::topology(points);
    let requirements = quotient_numerator_workspace_requirements(config, &topology).unwrap();
    let plan = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
        config,
        &topology,
        &[OVERFLOW_WORDS],
    )
    .unwrap();
    let layout = quotient_numerator_prepacked_term_layout(&plan).unwrap();
    let slots = quotient::workspace_slots(&requirements);
    let source_words = [1 << 3, 1 << 5, 1 << 5, 1 << 3];
    let (baseline_arena, baseline_bytes) =
        quotient::make_arena(&requirements, &slots, source_words, OVERFLOW_WORDS);
    let (candidate_arena, candidate_bytes) =
        quotient::make_arena(&requirements, &slots, source_words, OVERFLOW_WORDS);
    let values = values();
    let eager_sources = quotient::source_set(0x1234_5678);
    let replay_sources = quotient::source_set(0x6a09_e667);
    let twiddles = quotient::required_forward_twiddle_words(config, &requirements);
    initialize(
        [&baseline_arena, &candidate_arena],
        points,
        &values,
        &twiddles,
        &eager_sources,
    );

    let baseline_destinations = quotient::destinations(&baseline_arena, &requirements);
    let candidate_destinations = quotient::destinations(&candidate_arena, &requirements);
    let baseline_columns = quotient::columns(&baseline_arena, &topology);
    let candidate_columns = quotient::columns(&candidate_arena, &topology);
    let baseline = prepare_staged(
        &baseline_arena,
        config,
        &slots,
        &baseline_columns,
        &baseline_destinations,
    );
    let candidate = prepare_prepacked(
        &candidate_arena,
        config,
        &slots,
        &candidate_columns,
        &candidate_destinations,
    );
    assert_eq!(
        candidate.schedule(),
        PreparedNumeratorSchedule::StagedPrepackedSingleWrite {
            packed_output_rows: plan.packed_output_rows(),
        }
    );
    let receipt = candidate.prepacked_receipt().unwrap();
    assert_eq!(
        receipt.plan_identity,
        quotient_numerator_prepacked_plan_identity(&plan).unwrap()
    );
    assert_eq!(receipt.source_count as usize, plan.sources().len());
    assert_eq!(receipt.used_words as usize, layout.used_words);
    assert_eq!(receipt.status_offset_words, layout.status_offset_words);
    assert!(layout.used_words <= requirements.term_point_words);

    let eager_alpha = SecureField::from_u32_unchecked(73, 79, 83, 89);
    upload_alpha([&baseline_arena, &candidate_arena], eager_alpha);
    baseline.launch().unwrap();
    candidate.launch().unwrap();
    let d2h_before = candidate_arena.context().telemetry().d2h_bytes;
    candidate.observe_prepacked_status().unwrap();
    assert_eq!(
        candidate_arena.context().telemetry().d2h_bytes - d2h_before,
        core::mem::size_of::<u32>() as u64
    );
    let eager_baseline = quotient::snapshot(
        &baseline_arena,
        &requirements,
        &baseline_destinations,
        &values,
        eager_alpha,
        &quotient::evaluations(config, &eager_sources),
    );
    let eager_candidate = quotient::snapshot(
        &candidate_arena,
        &requirements,
        &candidate_destinations,
        &values,
        eager_alpha,
        &quotient::evaluations(config, &eager_sources),
    );
    assert_eq!(eager_candidate, eager_baseline);

    let capture = baseline_arena.context().capture().unwrap();
    baseline.launch().unwrap();
    let baseline_graph = capture.finish().unwrap();
    let capture = candidate_arena.context().capture().unwrap();
    candidate.launch().unwrap();
    let candidate_graph = capture.finish().unwrap();

    let replay_alpha = SecureField::from_u32_unchecked(97, 101, 103, 107);
    for arena in [&baseline_arena, &candidate_arena] {
        quotient::upload_sources(arena, &replay_sources);
    }
    upload_alpha([&baseline_arena, &candidate_arena], replay_alpha);
    baseline_graph.launch(baseline_arena.context()).unwrap();
    candidate_graph.launch(candidate_arena.context()).unwrap();
    candidate.observe_prepacked_status().unwrap();
    let replay_baseline = quotient::snapshot(
        &baseline_arena,
        &requirements,
        &baseline_destinations,
        &values,
        replay_alpha,
        &quotient::evaluations(config, &replay_sources),
    );
    let replay_candidate = quotient::snapshot(
        &candidate_arena,
        &requirements,
        &candidate_destinations,
        &values,
        replay_alpha,
        &quotient::evaluations(config, &replay_sources),
    );
    assert_eq!(replay_candidate, replay_baseline);
    assert_ne!(eager_candidate, replay_candidate);

    fill_outputs(&candidate_arena, &candidate_destinations, STALE_OUTPUT_BYTE);
    let mut invalid_descriptors = plan.term_descriptors().to_vec();
    invalid_descriptors[0] = u32::MAX;
    upload_words(
        &candidate_arena,
        candidate_arena.bind(slots.batch_terms).unwrap(),
        &invalid_descriptors,
    );
    candidate_arena.context().sync().unwrap();
    candidate_graph.launch(candidate_arena.context()).unwrap();
    assert!(matches!(
        candidate.observe_prepacked_status(),
        Err(PreparedQuotientNumeratorError::PrepackedDeviceStatus(status))
            if status
                == QuotientNumeratorPrepackedStatusCode::PrepareSourceOutOfBounds.as_u32()
    ));
    assert_outputs_still_filled(&candidate_arena, &candidate_destinations, STALE_OUTPUT_BYTE);

    upload_words(
        &candidate_arena,
        candidate_arena.bind(slots.batch_terms).unwrap(),
        plan.term_descriptors(),
    );
    fill_outputs(&candidate_arena, &candidate_destinations, 0xa6);
    candidate_arena.context().sync().unwrap();
    candidate_graph.launch(candidate_arena.context()).unwrap();
    candidate.observe_prepacked_status().unwrap();
    let recovered = quotient::snapshot(
        &candidate_arena,
        &requirements,
        &candidate_destinations,
        &values,
        replay_alpha,
        &quotient::evaluations(config, &replay_sources),
    );
    assert_eq!(recovered, replay_baseline);
    quotient::assert_preserved(&baseline_arena, &replay_sources);
    quotient::assert_preserved(&candidate_arena, &replay_sources);

    let hashes = [
        ("eager_outputs".to_owned(), hash_words(&eager_candidate)),
        ("replay_outputs".to_owned(), hash_words(&replay_candidate)),
        ("reset_recovery_outputs".to_owned(), hash_words(&recovered)),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    let checks = [
        ("exact_plan_receipt", true),
        ("eager_staged_prepacked_identity", true),
        ("captured_replay_status_reset", true),
        ("invalid_descriptor_rejects_stale_output", true),
        ("replay_recovers_after_status_error", true),
        ("source_and_guard_preservation", true),
    ]
    .into_iter()
    .map(|(name, passed)| (name.to_owned(), passed))
    .collect();
    FixtureReceipt {
        name: "staged-prepacked-quotient-boundary",
        production_apis: vec![
            "PreparedQuotientNumeratorGraph::prepare_staged_prepacked_single_write_candidate",
            "PreparedQuotientNumeratorGraph::observe_prepacked_status",
        ],
        cases: 4,
        arena_bytes: baseline_bytes + candidate_bytes,
        checks,
        hashes,
    }
}

#[cfg(stwo_cuda_link)]
pub fn benchmark(lifting_log_size: u32, warmups: usize, iterations: usize) -> PerformanceReceipt {
    let config = config(lifting_log_size);
    let points = points();
    let topology = quotient::scaled_topology(points, lifting_log_size);
    let source_words = [
        1usize << (lifting_log_size - 3),
        1usize << (lifting_log_size - 1),
        1usize << (lifting_log_size - 2),
        1usize << (lifting_log_size - 3),
    ];
    let overflow_words = 1usize << lifting_log_size;
    let requirements = quotient_numerator_workspace_requirements(config, &topology).unwrap();
    let plan = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
        config,
        &topology,
        &[overflow_words],
    )
    .unwrap();
    let layout = quotient_numerator_prepacked_term_layout(&plan).unwrap();
    let slots = quotient::workspace_slots(&requirements);
    let (baseline_arena, baseline_bytes) =
        quotient::make_arena(&requirements, &slots, source_words, overflow_words);
    let (candidate_arena, candidate_bytes) =
        quotient::make_arena(&requirements, &slots, source_words, overflow_words);
    let values = values();
    let sources = quotient::source_set_for_words(0x1319_8a2e, source_words);
    let twiddles = quotient::required_forward_twiddle_words(config, &requirements);
    initialize(
        [&baseline_arena, &candidate_arena],
        points,
        &values,
        &twiddles,
        &sources,
    );
    upload_alpha(
        [&baseline_arena, &candidate_arena],
        SecureField::from_u32_unchecked(73, 79, 83, 89),
    );

    let baseline_destinations = quotient::destinations(&baseline_arena, &requirements);
    let candidate_destinations = quotient::destinations(&candidate_arena, &requirements);
    let baseline = prepare_staged(
        &baseline_arena,
        config,
        &slots,
        &quotient::columns(&baseline_arena, &topology),
        &baseline_destinations,
    );
    let candidate = prepare_prepacked(
        &candidate_arena,
        config,
        &slots,
        &quotient::columns(&candidate_arena, &topology),
        &candidate_destinations,
    );
    let capture = baseline_arena.context().capture().unwrap();
    baseline.launch().unwrap();
    let baseline_graph = capture.finish().unwrap();
    let capture = candidate_arena.context().capture().unwrap();
    candidate.launch().unwrap();
    let candidate_graph = capture.finish().unwrap();
    candidate_graph.launch(candidate_arena.context()).unwrap();
    candidate.observe_prepacked_status().unwrap();

    let baseline_timing = cuda_event_timings(baseline_arena.context(), warmups, iterations, || {
        baseline_graph.launch(baseline_arena.context())
    });
    let candidate_timing =
        cuda_event_timings(candidate_arena.context(), warmups, iterations, || {
            candidate_graph.launch(candidate_arena.context())
        });
    candidate.observe_prepacked_status().unwrap();
    assert_eq!(
        quotient::raw_snapshot(&baseline_arena, &requirements, &baseline_destinations),
        quotient::raw_snapshot(&candidate_arena, &requirements, &candidate_destinations)
    );
    quotient::assert_preserved(&baseline_arena, &sources);
    quotient::assert_preserved(&candidate_arena, &sources);

    PerformanceReceipt {
        name: format!("staged-prepacked-quotient-log{lifting_log_size}"),
        parameters: [
            ("lifting_log_size".to_owned(), u64::from(lifting_log_size)),
            ("groups".to_owned(), requirements.groups.len() as u64),
            ("terms".to_owned(), requirements.term_count as u64),
            ("packed_output_rows".to_owned(), plan.packed_output_rows()),
            ("prepacked_used_words".to_owned(), layout.used_words as u64),
        ]
        .into_iter()
        .collect(),
        arena_bytes: [
            ("staged".to_owned(), baseline_bytes),
            ("prepacked".to_owned(), candidate_bytes),
        ]
        .into_iter()
        .collect(),
        traffic_bytes: [(
            "prepacked_record_bytes".to_owned(),
            (layout.used_words * core::mem::size_of::<u32>()) as u64,
        )]
        .into_iter()
        .collect(),
        baseline_label: "staged-packed-single-write-cuda-graph".to_owned(),
        candidate_label: "staged-prepacked-single-write-cuda-graph".to_owned(),
        baseline: baseline_timing,
        candidate: candidate_timing,
        speedup: baseline_timing.median_ms / candidate_timing.median_ms,
    }
}

fn config(lifting_log_size: u32) -> QuotientNumeratorWorkspaceConfig {
    QuotientNumeratorWorkspaceConfig {
        lifting_log_size,
        log_blowup_factor: 2,
        max_lde_tile_words: 32 * (1usize << lifting_log_size),
    }
}

fn points() -> [CirclePoint<SecureField>; 4] {
    [
        SECURE_FIELD_CIRCLE_GEN.mul(3),
        SECURE_FIELD_CIRCLE_GEN.mul(5),
        SECURE_FIELD_CIRCLE_GEN.mul(7),
        SECURE_FIELD_CIRCLE_GEN.mul(11),
    ]
}

fn values() -> [SecureField; 5] {
    [
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
        SecureField::from_u32_unchecked(23, 29, 31, 37),
        SecureField::from_u32_unchecked(41, 43, 47, 53),
        SecureField::from_u32_unchecked(59, 61, 67, 71),
    ]
}

fn initialize(
    arenas: [&DeviceArena; 2],
    points: [CirclePoint<SecureField>; 4],
    values: &[SecureField; 5],
    twiddles: &[u32],
    sources: &[Vec<u32>; 4],
) {
    for arena in arenas {
        upload_words(
            arena,
            arena.bind(quotient::OODS_POINTS).unwrap(),
            &quotient::point_words(&[points[0], points[1], points[0], points[2], points[3]]),
        );
        upload_words(
            arena,
            arena.bind(quotient::OODS_VALUES).unwrap(),
            &quotient::secure_words(values),
        );
        upload_words(arena, arena.bind(quotient::TWIDDLES).unwrap(), twiddles);
        quotient::upload_sources(arena, sources);
        fill(arena, arena.bind(quotient::GUARD).unwrap(), 0xa5);
        arena.context().sync().unwrap();
    }
}

fn upload_alpha(arenas: [&DeviceArena; 2], alpha: SecureField) {
    for arena in arenas {
        upload_words(
            arena,
            arena.bind(quotient::ALPHA).unwrap(),
            &quotient::secure_words(&[alpha]),
        );
    }
}

fn prepare_staged<'a>(
    arena: &'a DeviceArena,
    config: QuotientNumeratorWorkspaceConfig,
    slots: &QuotientNumeratorWorkspaceSlots,
    columns: &[QuotientNumeratorColumn],
    destinations: &[QuotientNumeratorDestination],
) -> PreparedQuotientNumeratorGraph<'a> {
    PreparedQuotientNumeratorGraph::prepare_staged_packed_single_write(
        arena,
        config,
        columns,
        arena.bind(quotient::OODS_POINTS).unwrap(),
        arena.bind(quotient::OODS_VALUES).unwrap(),
        arena.bind(quotient::ALPHA).unwrap(),
        arena.bind(quotient::SAMPLE_POINTS).unwrap(),
        arena.bind(quotient::FIRST_TERMS).unwrap(),
        destinations,
        arena.bind(quotient::TWIDDLES).unwrap(),
        slots,
        &[arena.bind(quotient::OVERFLOW).unwrap()],
    )
    .unwrap()
}

fn prepare_prepacked<'a>(
    arena: &'a DeviceArena,
    config: QuotientNumeratorWorkspaceConfig,
    slots: &QuotientNumeratorWorkspaceSlots,
    columns: &[QuotientNumeratorColumn],
    destinations: &[QuotientNumeratorDestination],
) -> PreparedQuotientNumeratorGraph<'a> {
    PreparedQuotientNumeratorGraph::prepare_staged_prepacked_single_write_candidate(
        arena,
        config,
        columns,
        arena.bind(quotient::OODS_POINTS).unwrap(),
        arena.bind(quotient::OODS_VALUES).unwrap(),
        arena.bind(quotient::ALPHA).unwrap(),
        arena.bind(quotient::SAMPLE_POINTS).unwrap(),
        arena.bind(quotient::FIRST_TERMS).unwrap(),
        destinations,
        arena.bind(quotient::TWIDDLES).unwrap(),
        slots,
        &[arena.bind(quotient::OVERFLOW).unwrap()],
    )
    .unwrap()
}

fn fill_outputs(arena: &DeviceArena, destinations: &[QuotientNumeratorDestination], byte: u8) {
    for destination in destinations {
        for coordinate in destination.coordinates {
            fill(arena, coordinate, byte);
        }
    }
}

fn assert_outputs_still_filled(
    arena: &DeviceArena,
    destinations: &[QuotientNumeratorDestination],
    byte: u8,
) {
    let word = u32::from_ne_bytes([byte; 4]);
    for destination in destinations {
        for coordinate in destination.coordinates {
            assert_eq!(
                read_words(arena, coordinate, 1usize << destination.log_size),
                vec![word; 1usize << destination.log_size]
            );
        }
    }
}
