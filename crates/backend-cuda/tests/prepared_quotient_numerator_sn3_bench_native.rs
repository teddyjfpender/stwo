//! Ignored cheap-GPU benchmark for the sealed SN3 quotient-numerator shape.
//! gpu-lab-cohesion-review: topology construction, byte identity, graph replay timing, and the
//! single JSON result stay together so a benchmark cannot silently drift from its comparator.
//! Run receipts must place this JSON beside `nvidia-smi` identity/driver/clock output,
//! `nvcc --version`, `rustc -Vv`, and `git rev-parse HEAD`; those are environment facts, not test
//! semantics, so the existing pod wrapper owns them.
//! Identity-complete records additionally set `STWO_SN3_NUMERATOR_SOURCE_PROJECTION_SHA256` and
//! `STWO_SN3_NUMERATOR_CUDA_MODULE_SHA256`; a git HEAD alone is not an artifact identity.

#[path = "support/sn3_quotient_numerator_bench.rs"]
mod sn3_quotient_numerator_bench;

use sn3_quotient_numerator_bench::{
    artifact_identity, assert_affine_pattern_sanity, assert_canonical_output,
    capture_canonical_output, input_recipe_digest, json_samples, percentile, poison_outputs,
    replay_ms, source_pattern_seed, upload_affine_pattern, TWIDDLE_PATTERN_SEED,
};
use stwo::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
use stwo::core::fields::qm31::SecureField;
use stwo_backend_cuda::{
    quotient_numerator_hybrid_plan, quotient_numerator_workspace_requirements, ArenaLayout,
    ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena, PreparedQuotientNumeratorGraph,
    QuotientNumeratorColumn, QuotientNumeratorColumnSource, QuotientNumeratorColumnTopology,
    QuotientNumeratorDestination, QuotientNumeratorSourceKind, QuotientNumeratorWorkspaceConfig,
    QuotientNumeratorWorkspaceRequirements, QuotientNumeratorWorkspaceSlots, QuotientOodsSample,
};

const CONFIG: QuotientNumeratorWorkspaceConfig = QuotientNumeratorWorkspaceConfig {
    lifting_log_size: 24,
    log_blowup_factor: 1,
    max_lde_tile_words: 2 * (1 << 24),
};
const GROUP_LOGS: [u32; 19] = [
    23, 19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22,
];
const ELIGIBLE_LOGS: [u32; 18] = [
    19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22,
];
const COEFFICIENT_SOURCES: usize = 152;
const COEFFICIENT_BATCHES: usize = 74;
const TERMS: usize = 6_341;
const LEGACY_LOGICAL_OUTPUT_BYTES: u64 = 59_993_989_376;
const HYBRID_LOGICAL_OUTPUT_BYTES: u64 = 20_266_867_968;
const CHEAP_GPU_BUDGET_BYTES: u64 = 24 * 1024 * 1024 * 1024;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_ITERATIONS: usize = 20;
const EXPECTED_SN3_INPUT_RECIPE_BLAKE3: &str =
    "ba4d8f134e932066012cd2795a05c359ee6f4ef568daf665da4a6b94295ee93a";
const EAGER_LEGACY_POISON: u32 = 0xdead_beef;
const EAGER_HYBRID_POISON: u32 = 0xa5a5_5a5a;
const CAPTURE_LEGACY_POISON: u32 = 0x1357_9bdf;
const CAPTURE_HYBRID_POISON: u32 = 0x2468_ace0;
const POST_TIMING_LEGACY_POISON: u32 = 0x0bad_f00d;
const POST_TIMING_HYBRID_POISON: u32 = 0xc001_d00d;
const TIMED_LEGACY_POISON: u32 = 0x3141_5926;
const TIMED_HYBRID_POISON: u32 = 0x2718_2818;

const OODS_POINTS: ArenaSlotId = ArenaSlotId(100);
const OODS_VALUES: ArenaSlotId = ArenaSlotId(101);
const RANDOM_COEFFICIENT: ArenaSlotId = ArenaSlotId(102);
const SAMPLE_POINTS_OUTPUT: ArenaSlotId = ArenaSlotId(103);
const FIRST_TERMS_OUTPUT: ArenaSlotId = ArenaSlotId(104);
const TWIDDLES: ArenaSlotId = ArenaSlotId(105);
const SOURCE_BASE: u32 = 1_000;
const OUTPUT_BASE: u32 = 10_000;

#[test]
#[ignore = "requires a CUDA GPU with at least the reported combined arena bytes"]
fn sn3_hybrid_graph_host_wall_benchmark() {
    let (topology, points) = sn3_topology();
    let requirements = quotient_numerator_workspace_requirements(CONFIG, &topology).unwrap();
    let hybrid_plan = quotient_numerator_hybrid_plan(CONFIG, &topology).unwrap();
    assert_sn3_shape(&topology, &requirements, &hybrid_plan);
    let input_recipe_blake3 = input_recipe_digest(&topology, &requirements, &hybrid_plan, &points);

    // All exact topology assertions above intentionally precede the first CUDA allocation.
    let legacy_fixture = BenchmarkArena::new(&topology, &requirements);
    let hybrid_fixture = BenchmarkArena::new(&topology, &requirements);
    let combined_arena_bytes = legacy_fixture
        .allocation_bytes
        .checked_add(hybrid_fixture.allocation_bytes)
        .unwrap();
    assert!(legacy_fixture.allocation_bytes <= CHEAP_GPU_BUDGET_BYTES);
    assert!(hybrid_fixture.allocation_bytes <= CHEAP_GPU_BUDGET_BYTES);
    assert!(combined_arena_bytes <= CHEAP_GPU_BUDGET_BYTES);

    let slots = workspace_slots(&requirements);
    let legacy_columns = legacy_fixture.columns(&topology);
    let hybrid_columns = hybrid_fixture.columns(&topology);
    let legacy_destinations = legacy_fixture.destinations(&requirements);
    let hybrid_destinations = hybrid_fixture.destinations(&requirements);
    let legacy = prepare(
        &legacy_fixture,
        &legacy_columns,
        &legacy_destinations,
        &slots,
        false,
    );
    let hybrid = prepare(
        &hybrid_fixture,
        &hybrid_columns,
        &hybrid_destinations,
        &slots,
        true,
    );

    initialize(
        &legacy_fixture,
        &topology,
        &requirements,
        &points,
        EAGER_LEGACY_POISON,
    );
    initialize(
        &hybrid_fixture,
        &topology,
        &requirements,
        &points,
        EAGER_HYBRID_POISON,
    );
    legacy.launch().unwrap();
    hybrid.launch().unwrap();
    legacy_fixture.arena.context().sync().unwrap();
    hybrid_fixture.arena.context().sync().unwrap();
    let eager = capture_canonical_output(&legacy_fixture, &requirements);
    let eager_hybrid_blake3 =
        assert_canonical_output(&hybrid_fixture, &requirements, &eager, "eager hybrid");
    let validated_numerator_output_bytes = requirements
        .groups
        .iter()
        .map(|group| group.value_words as u64)
        .sum::<u64>()
        .checked_mul(16)
        .unwrap();
    let validated_auxiliary_output_bytes = (requirements.groups.len() as u64)
        .checked_mul(12 * 4)
        .unwrap();
    let validated_canonical_output_bytes = eager.len_bytes();
    assert_eq!(validated_numerator_output_bytes, 402_644_224);
    assert_eq!(validated_auxiliary_output_bytes, 912);
    assert_eq!(validated_canonical_output_bytes, 402_645_136);
    assert_eq!(
        validated_canonical_output_bytes,
        validated_numerator_output_bytes + validated_auxiliary_output_bytes
    );

    let capture = legacy_fixture.arena.context().capture().unwrap();
    legacy.launch().unwrap();
    let legacy_graph = capture.finish().unwrap();
    let capture = hybrid_fixture.arena.context().capture().unwrap();
    hybrid.launch().unwrap();
    let hybrid_graph = capture.finish().unwrap();

    poison_outputs(&legacy_fixture, &requirements, CAPTURE_LEGACY_POISON);
    poison_outputs(&hybrid_fixture, &requirements, CAPTURE_HYBRID_POISON);
    legacy_graph.launch(legacy_fixture.arena.context()).unwrap();
    legacy_fixture.arena.context().sync().unwrap();
    hybrid_graph.launch(hybrid_fixture.arena.context()).unwrap();
    hybrid_fixture.arena.context().sync().unwrap();
    let captured_legacy_blake3 =
        assert_canonical_output(&legacy_fixture, &requirements, &eager, "captured legacy");
    let captured_hybrid_blake3 =
        assert_canonical_output(&hybrid_fixture, &requirements, &eager, "captured hybrid");

    for round in 0..DEFAULT_WARMUPS {
        if round % 2 == 0 {
            replay_ms(&legacy_graph, legacy_fixture.arena.context());
            replay_ms(&hybrid_graph, hybrid_fixture.arena.context());
        } else {
            replay_ms(&hybrid_graph, hybrid_fixture.arena.context());
            replay_ms(&legacy_graph, legacy_fixture.arena.context());
        }
    }

    let iterations = std::env::var("STWO_SN3_NUMERATOR_BENCH_ITERS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("STWO_SN3_NUMERATOR_BENCH_ITERS must be an integer")
        })
        .unwrap_or(DEFAULT_ITERATIONS);
    assert!(
        iterations >= 5,
        "formal SN3 numerator A/B requires at least five iterations"
    );
    let mut legacy_ms = Vec::with_capacity(iterations);
    let mut hybrid_ms = Vec::with_capacity(iterations);

    // Sample zero is causally bound to work by poisoning immediately before the measured graph
    // replay and validating that exact replay in full before either graph runs again.
    poison_outputs(&legacy_fixture, &requirements, TIMED_LEGACY_POISON);
    poison_outputs(&hybrid_fixture, &requirements, TIMED_HYBRID_POISON);
    legacy_ms.push(replay_ms(&legacy_graph, legacy_fixture.arena.context()));
    hybrid_ms.push(replay_ms(&hybrid_graph, hybrid_fixture.arena.context()));
    let timed_legacy_blake3 = assert_canonical_output(
        &legacy_fixture,
        &requirements,
        &eager,
        "timed legacy sample 0",
    );
    let timed_hybrid_blake3 = assert_canonical_output(
        &hybrid_fixture,
        &requirements,
        &eager,
        "timed hybrid sample 0",
    );

    for iteration in 1..iterations {
        if iteration % 2 == 0 {
            legacy_ms.push(replay_ms(&legacy_graph, legacy_fixture.arena.context()));
            hybrid_ms.push(replay_ms(&hybrid_graph, hybrid_fixture.arena.context()));
        } else {
            hybrid_ms.push(replay_ms(&hybrid_graph, hybrid_fixture.arena.context()));
            legacy_ms.push(replay_ms(&legacy_graph, legacy_fixture.arena.context()));
        }
    }

    poison_outputs(&legacy_fixture, &requirements, POST_TIMING_LEGACY_POISON);
    poison_outputs(&hybrid_fixture, &requirements, POST_TIMING_HYBRID_POISON);
    legacy_graph.launch(legacy_fixture.arena.context()).unwrap();
    legacy_fixture.arena.context().sync().unwrap();
    hybrid_graph.launch(hybrid_fixture.arena.context()).unwrap();
    hybrid_fixture.arena.context().sync().unwrap();
    let post_timing_legacy_blake3 =
        assert_canonical_output(&legacy_fixture, &requirements, &eager, "post-timing legacy");
    let post_timing_hybrid_blake3 =
        assert_canonical_output(&hybrid_fixture, &requirements, &eager, "post-timing hybrid");
    let artifact_identity = artifact_identity();

    let legacy_p50 = percentile(&legacy_ms, 50);
    let legacy_p95 = percentile(&legacy_ms, 95);
    let hybrid_p50 = percentile(&hybrid_ms, 50);
    let hybrid_p95 = percentile(&hybrid_ms, 95);
    println!(
        concat!(
            "{{\"schema\":\"stwo.sn3_quotient_numerator_hybrid.host_wall.v3\"," ,
            "\"timing_scope\":\"per-replay host wall: graph launch plus stream synchronize; not CUDA events\"," ,
            "\"percentile_method\":\"nearest-rank\"," ,
            "\"result_class\":\"diagnostic same-lineage schedule A/B; not independent mathematical truth\"," ,
            "\"input_pattern\":\"input-recipe-sealed nonzero canonical-M31 affine row pattern; bounded chunked upload\"," ,
            "\"topology\":{{\"group_logs\":{:?},\"groups\":19,\"eligible_groups\":18," ,
            "\"legacy_groups\":1,\"coefficient_sources\":152,\"coefficient_batches\":74," ,
            "\"terms\":6341}},\"bytes\":{{\"legacy_logical_output\":{}," ,
            "\"hybrid_logical_output\":{},\"validated_numerator_output\":{}," ,
            "\"validated_auxiliary_output\":{},\"validated_canonical_output\":{}," ,
            "\"legacy_arena\":{},\"hybrid_arena\":{},\"combined_arenas\":{}}}," ,
            "\"identity\":{{\"output_digest_encoding\":\"framed u32 little-endian v1\"," ,
            "\"input_recipe_encoding\":\"typed topology, descriptors, and affine row recipes v2\"," ,
            "\"input_recipe_blake3\":\"{}\",\"eager_legacy_blake3\":\"{}\"," ,
            "\"eager_hybrid_blake3\":\"{}\",\"captured_legacy_blake3\":\"{}\"," ,
            "\"captured_hybrid_blake3\":\"{}\",\"timed_sample_index\":0," ,
            "\"timed_sample_causally_validated\":true,\"timed_legacy_blake3\":\"{}\"," ,
            "\"timed_hybrid_blake3\":\"{}\",\"post_timing_legacy_blake3\":\"{}\"," ,
            "\"post_timing_hybrid_blake3\":\"{}\",\"capture_revalidated\":true," ,
            "\"post_timing_revalidated\":true}}," ,
            "\"artifact_identity\":{{\"candidate_current_source_blake3\":\"{}\"," ,
            "\"test_binary_blake3\":\"{}\",\"source_projection_sha256\":{}," ,
            "\"cuda_module_sha256\":{},\"cuda_build_mode\":\"{}\"," ,
            "\"identity_complete\":{}}}," ,
            "\"comparator\":{{\"lineage\":\"same prepared implementation lineage; hybrid adds the single-write schedule and both reuse legacy kernels\"," ,
            "\"independent_truth\":false}}," ,
            "\"warmups\":{},\"iterations\":{},\"minimum_iterations\":5," ,
            "\"samples_ms\":{{\"legacy\":{},\"hybrid\":{}}}," ,
            "\"host_wall_ms\":{{\"legacy\":{{\"p50\":{:.6},\"p95\":{:.6}}}," ,
            "\"hybrid\":{{\"p50\":{:.6},\"p95\":{:.6}}}}}," ,
            "\"speedup\":{{\"p50\":{:.9},\"p95\":{:.9}}}}}"
        ),
        GROUP_LOGS,
        LEGACY_LOGICAL_OUTPUT_BYTES,
        HYBRID_LOGICAL_OUTPUT_BYTES,
        validated_numerator_output_bytes,
        validated_auxiliary_output_bytes,
        validated_canonical_output_bytes,
        legacy_fixture.allocation_bytes,
        hybrid_fixture.allocation_bytes,
        combined_arena_bytes,
        input_recipe_blake3,
        eager.digest(),
        eager_hybrid_blake3,
        captured_legacy_blake3,
        captured_hybrid_blake3,
        timed_legacy_blake3,
        timed_hybrid_blake3,
        post_timing_legacy_blake3,
        post_timing_hybrid_blake3,
        artifact_identity.candidate_source_blake3,
        artifact_identity.test_binary_blake3,
        artifact_identity.source_projection_json(),
        artifact_identity.cuda_module_json(),
        artifact_identity.cuda_build_mode,
        artifact_identity.is_complete(),
        DEFAULT_WARMUPS,
        iterations,
        json_samples(&legacy_ms),
        json_samples(&hybrid_ms),
        legacy_p50,
        legacy_p95,
        hybrid_p50,
        hybrid_p95,
        legacy_p50 / hybrid_p50,
        legacy_p95 / hybrid_p95,
    );
}

#[test]
fn sn3_input_recipe_is_deterministic_and_shape_exact() {
    assert_affine_pattern_sanity();
    let (topology, points) = sn3_topology();
    let requirements = quotient_numerator_workspace_requirements(CONFIG, &topology).unwrap();
    let hybrid = quotient_numerator_hybrid_plan(CONFIG, &topology).unwrap();
    assert_sn3_shape(&topology, &requirements, &hybrid);
    let digest = input_recipe_digest(&topology, &requirements, &hybrid, &points);
    assert_eq!(digest.to_hex().as_str(), EXPECTED_SN3_INPUT_RECIPE_BLAKE3);
    let mut changed_points = points.clone();
    changed_points.swap(0, 1);
    assert_ne!(
        input_recipe_digest(&topology, &requirements, &hybrid, &changed_points),
        digest
    );
}

fn sn3_topology() -> (
    Vec<QuotientNumeratorColumnTopology>,
    Vec<CirclePoint<SecureField>>,
) {
    let mut points = (0..GROUP_LOGS.len())
        .map(|index| SECURE_FIELD_CIRCLE_GEN.mul(2 * index as u128 + 3))
        .collect::<Vec<_>>();
    points.sort_by_key(|point| (point.x, point.y));
    points.dedup();
    assert_eq!(points.len(), GROUP_LOGS.len());

    let legacy_sample = QuotientOodsSample {
        input_index: 0,
        shape_point: points[0],
    };
    let mut topology = Vec::with_capacity(COEFFICIENT_SOURCES + ELIGIBLE_LOGS.len());
    topology.extend((0..114).map(|_| coefficient_topology(23, legacy_sample)));
    for (index, &log_size) in ELIGIBLE_LOGS
        .iter()
        .filter(|&&log_size| log_size != 23)
        .enumerate()
    {
        let count = 2 + usize::from(index < 4);
        topology.extend((0..count).map(|_| coefficient_topology(log_size, legacy_sample)));
    }
    assert_eq!(topology.len(), COEFFICIENT_SOURCES);

    let dense_terms = TERMS - COEFFICIENT_SOURCES - (ELIGIBLE_LOGS.len() - 1);
    for (group, &log_size) in ELIGIBLE_LOGS.iter().enumerate() {
        let sample = QuotientOodsSample {
            input_index: (group + 1) as u32,
            shape_point: points[group + 1],
        };
        topology.push(QuotientNumeratorColumnTopology {
            coefficient_log_size: log_size,
            source_kind: QuotientNumeratorSourceKind::Evaluation,
            samples: vec![sample; if group == 0 { dense_terms } else { 1 }],
        });
    }
    (topology, points)
}

fn coefficient_topology(
    coefficient_log_size: u32,
    sample: QuotientOodsSample,
) -> QuotientNumeratorColumnTopology {
    QuotientNumeratorColumnTopology {
        coefficient_log_size,
        source_kind: QuotientNumeratorSourceKind::Coefficients,
        samples: vec![sample],
    }
}

fn assert_sn3_shape(
    topology: &[QuotientNumeratorColumnTopology],
    requirements: &QuotientNumeratorWorkspaceRequirements,
    hybrid: &stwo_backend_cuda::QuotientNumeratorHybridPlan,
) {
    assert_eq!(topology.len(), COEFFICIENT_SOURCES + ELIGIBLE_LOGS.len());
    assert_eq!(requirements.term_count, TERMS);
    assert_eq!(requirements.input_sample_count, GROUP_LOGS.len());
    assert_eq!(requirements.groups.len(), GROUP_LOGS.len());
    assert_eq!(
        requirements
            .groups
            .iter()
            .map(|group| group.log_size)
            .collect::<Vec<_>>(),
        GROUP_LOGS
    );
    assert_eq!(
        requirements.groups[0].coefficient_source_count,
        COEFFICIENT_SOURCES
    );
    assert!(requirements.groups[1..]
        .iter()
        .all(|group| group.coefficient_source_count == 0));
    assert_eq!(requirements.batches.len(), COEFFICIENT_BATCHES);
    assert_eq!(
        requirements
            .batches
            .iter()
            .map(|batch| batch.coefficient_count)
            .sum::<usize>(),
        COEFFICIENT_SOURCES
    );
    assert!(requirements
        .batches
        .iter()
        .all(|batch| batch.coefficient_count != 0));

    let report = hybrid.report();
    assert_eq!(report.eligible_group_count, ELIGIBLE_LOGS.len());
    assert_eq!(report.legacy_group_count, 1);
    assert_eq!(report.legacy_batch_count, COEFFICIENT_BATCHES);
    assert_eq!(report.eligible_output_rows, 16_776_656);
    assert_eq!(report.legacy_output_rows, 8_388_608);
    assert_eq!(
        report.legacy_logical_output_bytes,
        LEGACY_LOGICAL_OUTPUT_BYTES
    );
    assert_eq!(
        report.hybrid_logical_output_bytes,
        HYBRID_LOGICAL_OUTPUT_BYTES
    );
    assert_eq!(hybrid.packed_terms().len(), 19_023);
    assert_eq!(hybrid.packed_group_offsets().len(), 168);
}

struct BenchmarkArena {
    arena: DeviceArena,
    source_ids: Vec<ArenaSlotId>,
    destination_ids: Vec<[ArenaSlotId; 4]>,
    allocation_bytes: u64,
}

impl BenchmarkArena {
    fn new(
        topology: &[QuotientNumeratorColumnTopology],
        requirements: &QuotientNumeratorWorkspaceRequirements,
    ) -> Self {
        let slots = workspace_slots(requirements);
        let mut specs = Vec::new();
        let mut cursor = 0usize;
        for requirement in requirements.arena_slot_requirements(&slots).unwrap() {
            push_spec(
                &mut specs,
                &mut cursor,
                requirement.id,
                requirement.len_words,
                requirement.alignment_words,
            );
        }
        for (id, words) in [
            (OODS_POINTS, requirements.input_sample_count * 8),
            (OODS_VALUES, requirements.input_sample_count * 4),
            (RANDOM_COEFFICIENT, 4),
            (SAMPLE_POINTS_OUTPUT, requirements.groups.len() * 8),
            (FIRST_TERMS_OUTPUT, requirements.groups.len() * 4),
            (TWIDDLES, requirements.forward_twiddle_words),
        ] {
            push_spec(&mut specs, &mut cursor, id, words, 8);
        }
        let source_ids = topology
            .iter()
            .enumerate()
            .map(|(index, column)| {
                let id = ArenaSlotId(SOURCE_BASE + index as u32);
                push_spec(&mut specs, &mut cursor, id, source_words(column), 8);
                id
            })
            .collect::<Vec<_>>();
        let destination_ids = requirements
            .groups
            .iter()
            .enumerate()
            .map(|(group, requirement)| {
                std::array::from_fn(|coordinate| {
                    let id = output_id(group, coordinate);
                    push_spec(&mut specs, &mut cursor, id, requirement.value_words, 8);
                    id
                })
            })
            .collect::<Vec<_>>();
        let allocation_bytes = (cursor as u64).checked_mul(4).unwrap();
        assert!(allocation_bytes <= CHEAP_GPU_BUDGET_BYTES);
        let layout = ArenaLayout::new(cursor, &specs).unwrap();
        let arena = DeviceArena::new(CudaExecContext::new().unwrap(), layout).unwrap();
        Self {
            arena,
            source_ids,
            destination_ids,
            allocation_bytes,
        }
    }

    fn columns(
        &self,
        topology: &[QuotientNumeratorColumnTopology],
    ) -> Vec<QuotientNumeratorColumn> {
        topology
            .iter()
            .zip(&self.source_ids)
            .map(|(topology, &id)| QuotientNumeratorColumn {
                coefficient_log_size: topology.coefficient_log_size,
                source: match topology.source_kind {
                    QuotientNumeratorSourceKind::Evaluation => {
                        QuotientNumeratorColumnSource::Evaluation(self.arena.bind(id).unwrap())
                    }
                    QuotientNumeratorSourceKind::Coefficients => {
                        QuotientNumeratorColumnSource::Coefficients(self.arena.bind(id).unwrap())
                    }
                },
                samples: topology.samples.clone(),
            })
            .collect()
    }

    fn destinations(
        &self,
        requirements: &QuotientNumeratorWorkspaceRequirements,
    ) -> Vec<QuotientNumeratorDestination> {
        requirements
            .groups
            .iter()
            .zip(&self.destination_ids)
            .map(|(group, ids)| QuotientNumeratorDestination {
                log_size: group.log_size,
                coordinates: ids.map(|id| self.arena.bind(id).unwrap()),
            })
            .collect()
    }
}

fn workspace_slots(
    requirements: &QuotientNumeratorWorkspaceRequirements,
) -> QuotientNumeratorWorkspaceSlots {
    QuotientNumeratorWorkspaceSlots {
        runtime_terms: ArenaSlotId(1),
        group_term_indices: ArenaSlotId(2),
        group_offsets: ArenaSlotId(3),
        line_coefficients: ArenaSlotId(4),
        term_points: ArenaSlotId(5),
        batch_terms: ArenaSlotId(6),
        batch_group_offsets: ArenaSlotId(7),
        batch_source_ptrs: ArenaSlotId(8),
        output_ptrs: ArenaSlotId(9),
        output_log_sizes: ArenaSlotId(10),
        coefficient_ptrs: (requirements.coefficient_pointer_words != 0).then_some(ArenaSlotId(11)),
        coefficient_sizes: (requirements.coefficient_size_words != 0).then_some(ArenaSlotId(12)),
        coefficient_output_ptrs: (requirements.coefficient_output_pointer_words != 0)
            .then_some(ArenaSlotId(13)),
        lde_tile: (requirements.lde_tile_words != 0).then_some(ArenaSlotId(14)),
    }
}

fn push_spec(
    specs: &mut Vec<ArenaSlotSpec>,
    cursor: &mut usize,
    id: ArenaSlotId,
    len_words: usize,
    alignment_words: usize,
) {
    assert!(len_words != 0);
    let padding = (alignment_words - *cursor % alignment_words) % alignment_words;
    *cursor = cursor.checked_add(padding).unwrap();
    specs.push(ArenaSlotSpec {
        id,
        offset_words: *cursor,
        len_words,
        alignment_words,
    });
    *cursor = cursor.checked_add(len_words).unwrap();
}

fn source_words(column: &QuotientNumeratorColumnTopology) -> usize {
    let log_size = column.coefficient_log_size
        + u32::from(column.source_kind == QuotientNumeratorSourceKind::Evaluation);
    1usize << log_size
}

#[allow(clippy::too_many_arguments)]
fn prepare<'a>(
    fixture: &'a BenchmarkArena,
    columns: &[QuotientNumeratorColumn],
    destinations: &[QuotientNumeratorDestination],
    slots: &QuotientNumeratorWorkspaceSlots,
    hybrid: bool,
) -> PreparedQuotientNumeratorGraph<'a> {
    let arguments = (
        fixture.arena.bind(OODS_POINTS).unwrap(),
        fixture.arena.bind(OODS_VALUES).unwrap(),
        fixture.arena.bind(RANDOM_COEFFICIENT).unwrap(),
        fixture.arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        fixture.arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        fixture.arena.bind(TWIDDLES).unwrap(),
    );
    if hybrid {
        PreparedQuotientNumeratorGraph::prepare_hybrid_candidate(
            &fixture.arena,
            CONFIG,
            columns,
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            destinations,
            arguments.5,
            slots,
        )
        .unwrap()
    } else {
        PreparedQuotientNumeratorGraph::prepare(
            &fixture.arena,
            CONFIG,
            columns,
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            destinations,
            arguments.5,
            slots,
        )
        .unwrap()
    }
}

fn initialize(
    fixture: &BenchmarkArena,
    topology: &[QuotientNumeratorColumnTopology],
    requirements: &QuotientNumeratorWorkspaceRequirements,
    points: &[CirclePoint<SecureField>],
    poison: u32,
) {
    let values = oods_values(points.len());
    upload(&fixture.arena, OODS_POINTS, &point_words(points));
    upload(&fixture.arena, OODS_VALUES, &secure_words(&values));
    upload(
        &fixture.arena,
        RANDOM_COEFFICIENT,
        &secure_words(&[random_coefficient()]),
    );
    upload_affine_pattern(
        &fixture.arena,
        TWIDDLES,
        requirements.forward_twiddle_words,
        TWIDDLE_PATTERN_SEED,
    );
    for (index, (&id, column)) in fixture.source_ids.iter().zip(topology).enumerate() {
        upload_affine_pattern(
            &fixture.arena,
            id,
            source_words(column),
            source_pattern_seed(index),
        );
    }
    poison_outputs(fixture, requirements, poison);
}

fn oods_values(count: usize) -> Vec<SecureField> {
    (0..count)
        .map(|index| {
            let value = index as u32 * 16 + 1;
            SecureField::from_u32_unchecked(value, value + 2, value + 4, value + 6)
        })
        .collect()
}

fn random_coefficient() -> SecureField {
    SecureField::from_u32_unchecked(307, 311, 313, 317)
}

fn output_id(group: usize, coordinate: usize) -> ArenaSlotId {
    ArenaSlotId(OUTPUT_BASE + (4 * group + coordinate) as u32)
}

fn upload(arena: &DeviceArena, slot: ArenaSlotId, words: &[u32]) {
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(
                arena.bind(slot).unwrap().as_void_ptr(),
                words.as_ptr().cast(),
                std::mem::size_of_val(words),
            )
            .unwrap();
    }
}

fn secure_words(values: &[SecureField]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|value| value.to_m31_array().map(|coordinate| coordinate.0))
        .collect()
}

fn point_words(values: &[CirclePoint<SecureField>]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|point| [point.x, point.y])
        .flat_map(|value| value.to_m31_array().map(|coordinate| coordinate.0))
        .collect()
}
