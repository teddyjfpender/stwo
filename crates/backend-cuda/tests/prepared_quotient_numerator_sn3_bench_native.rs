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
#[path = "support/sn3_quotient_topology_fixture.rs"]
mod sn3_quotient_topology_fixture;

use sn3_quotient_numerator_bench::{
    artifact_identity, assert_affine_pattern_sanity, assert_canonical_output,
    capture_canonical_output, input_recipe_digest, json_samples, percentile, poison_outputs,
    replay_ms, source_pattern_seed, upload_affine_pattern, TWIDDLE_PATTERN_SEED,
};
use sn3_quotient_topology_fixture::load_sn3_topology_fixture;
use stwo::core::circle::CirclePoint;
use stwo::core::fields::qm31::SecureField;
use stwo_backend_cuda::{
    gpu_memory_info, ArenaLayout, ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena,
    PreparedQuotientNumeratorGraph, QuotientNumeratorColumn, QuotientNumeratorColumnSource,
    QuotientNumeratorColumnTopology, QuotientNumeratorDestination, QuotientNumeratorSourceKind,
    QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceRequirements,
    QuotientNumeratorWorkspaceSlots,
};

const GROUP_LOGS: [u32; 19] = [
    23, 19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22,
];
const ELIGIBLE_LOGS: [u32; 18] = [
    19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22,
];
const EXPECTED_SN3_CONFIG: QuotientNumeratorWorkspaceConfig = QuotientNumeratorWorkspaceConfig {
    lifting_log_size: 24,
    log_blowup_factor: 1,
    max_lde_tile_words: 1 << 24,
};
const COEFFICIENT_COLUMNS: usize = 161;
const LEGACY_GROUP_COEFFICIENT_SOURCES: usize = 152;
const LEGACY_BATCHES: usize = 74;
const COEFFICIENT_BATCHES: usize = 71;
const TERMS: usize = 6_341;
const LEGACY_LOGICAL_OUTPUT_BYTES: u64 = 59_993_989_376;
const HYBRID_LOGICAL_OUTPUT_BYTES: u64 = 20_266_867_968;
const EXPECTED_SN3_SINGLE_WORKSPACE_ARENA_BYTES: u64 = 41_821_220_224;
const EXPECTED_SN3_DUAL_WORKSPACE_ARENA_BYTES: u64 = 41_889_121_376;
const EXPECTED_SN3_WORKSPACE_SPAN_BYTES: u64 = 67_901_168;
const EXPECTED_SN3_SECOND_WORKSPACE_ARENA_DELTA_BYTES: u64 = 67_901_152;
const MAX_BENCHMARK_ARENA_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_ITERATIONS: usize = 20;
const EXPECTED_SN3_TOPOLOGY_FIXTURE_BLAKE3: &str =
    "ea31e3ff054c8d12d32d5b84a3d712987b31bb1fd3fb044fb27758453b49fbda";
const EXPECTED_SN3_INPUT_RECIPE_BLAKE3: &str =
    "e4c2f871c2d05b81588a5407f06cb49c7ed76834d2e363d2214bd34e7defcf31";
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
const LEGACY_WORKSPACE_BASE: u32 = 1;
const HYBRID_WORKSPACE_BASE: u32 = 20;

#[test]
#[ignore = "requires CUDA and the reported 39.02 GiB dual-workspace arena"]
fn sn3_hybrid_graph_host_wall_benchmark() {
    let sn3 = load_sn3_topology_fixture(EXPECTED_SN3_TOPOLOGY_FIXTURE_BLAKE3);
    assert_sn3_shape(sn3.config, &sn3.topology, &sn3.requirements, &sn3.hybrid);
    let input_recipe_blake3 = input_recipe_digest(
        &sn3.topology,
        &sn3.requirements,
        &sn3.hybrid,
        &sn3.input_points,
    );

    // All exact topology assertions above intentionally precede the first CUDA allocation.
    let (device_free_before_arena, device_total_bytes) = gpu_memory_info();
    let fixture = BenchmarkArena::new(&sn3.topology, &sn3.requirements);
    assert!(fixture.allocation_bytes <= MAX_BENCHMARK_ARENA_BYTES);
    let arena_pool = fixture.arena.context().pool_memory().unwrap();
    let (device_free_after_arena, device_total_after_arena) = gpu_memory_info();
    assert_eq!(device_total_after_arena, device_total_bytes);
    assert!(arena_pool.used_bytes >= fixture.allocation_bytes as usize);
    assert!(arena_pool.reserved_bytes >= arena_pool.used_bytes);

    let columns = fixture.columns(&sn3.topology);
    let destinations = fixture.destinations(&sn3.requirements);
    // Both prepared graphs remain live for alternating replay. Their external inputs may share,
    // but setup mutates schedule descriptors, so each graph owns a complete workspace.
    let legacy = prepare(
        &fixture,
        &columns,
        &destinations,
        &fixture.legacy_slots,
        sn3.config,
        false,
    );
    let hybrid = prepare(
        &fixture,
        &columns,
        &destinations,
        &fixture.hybrid_slots,
        sn3.config,
        true,
    );

    initialize(
        &fixture,
        &sn3.topology,
        &sn3.requirements,
        &sn3.input_points,
        EAGER_LEGACY_POISON,
    );
    legacy.launch().unwrap();
    fixture.arena.context().sync().unwrap();
    let eager = capture_canonical_output(&fixture, &sn3.requirements);
    initialize(
        &fixture,
        &sn3.topology,
        &sn3.requirements,
        &sn3.input_points,
        EAGER_HYBRID_POISON,
    );
    hybrid.launch().unwrap();
    fixture.arena.context().sync().unwrap();
    let eager_hybrid_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "eager hybrid");
    let validated_numerator_output_bytes = sn3
        .requirements
        .groups
        .iter()
        .map(|group| group.value_words as u64)
        .sum::<u64>()
        .checked_mul(16)
        .unwrap();
    let validated_auxiliary_output_bytes = (sn3.requirements.groups.len() as u64)
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

    let capture = fixture.arena.context().capture().unwrap();
    legacy.launch().unwrap();
    let legacy_graph = capture.finish().unwrap();
    let capture = fixture.arena.context().capture().unwrap();
    hybrid.launch().unwrap();
    let hybrid_graph = capture.finish().unwrap();

    poison_outputs(&fixture, &sn3.requirements, CAPTURE_LEGACY_POISON);
    legacy_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let captured_legacy_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "captured legacy");
    poison_outputs(&fixture, &sn3.requirements, CAPTURE_HYBRID_POISON);
    hybrid_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let captured_hybrid_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "captured hybrid");

    for round in 0..DEFAULT_WARMUPS {
        if round % 2 == 0 {
            replay_ms(&legacy_graph, fixture.arena.context());
            replay_ms(&hybrid_graph, fixture.arena.context());
        } else {
            replay_ms(&hybrid_graph, fixture.arena.context());
            replay_ms(&legacy_graph, fixture.arena.context());
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
    poison_outputs(&fixture, &sn3.requirements, TIMED_LEGACY_POISON);
    legacy_ms.push(replay_ms(&legacy_graph, fixture.arena.context()));
    let timed_legacy_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "timed legacy sample 0");
    poison_outputs(&fixture, &sn3.requirements, TIMED_HYBRID_POISON);
    hybrid_ms.push(replay_ms(&hybrid_graph, fixture.arena.context()));
    let timed_hybrid_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "timed hybrid sample 0");

    for iteration in 1..iterations {
        if iteration % 2 == 0 {
            legacy_ms.push(replay_ms(&legacy_graph, fixture.arena.context()));
            hybrid_ms.push(replay_ms(&hybrid_graph, fixture.arena.context()));
        } else {
            hybrid_ms.push(replay_ms(&hybrid_graph, fixture.arena.context()));
            legacy_ms.push(replay_ms(&legacy_graph, fixture.arena.context()));
        }
    }

    poison_outputs(&fixture, &sn3.requirements, POST_TIMING_LEGACY_POISON);
    legacy_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let post_timing_legacy_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "post-timing legacy");
    poison_outputs(&fixture, &sn3.requirements, POST_TIMING_HYBRID_POISON);
    hybrid_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let post_timing_hybrid_blake3 =
        assert_canonical_output(&fixture, &sn3.requirements, &eager, "post-timing hybrid");
    let artifact_identity = artifact_identity();

    let legacy_p50 = percentile(&legacy_ms, 50);
    let legacy_p95 = percentile(&legacy_ms, 95);
    let hybrid_p50 = percentile(&hybrid_ms, 50);
    let hybrid_p95 = percentile(&hybrid_ms, 95);
    println!(
        concat!(
            "{{\"schema\":\"stwo.sn3_quotient_numerator_hybrid.host_wall.v5\"," ,
            "\"timing_scope\":\"per-replay host wall: graph launch plus stream synchronize; not CUDA events\"," ,
            "\"percentile_method\":\"nearest-rank\"," ,
            "\"result_class\":\"diagnostic same-lineage schedule A/B; not independent mathematical truth\"," ,
            "\"input_pattern\":\"input-recipe-sealed nonzero canonical-M31 affine row pattern; bounded chunked upload\"," ,
            "\"topology\":{{\"group_logs\":{:?},\"groups\":19,\"eligible_groups\":18," ,
            "\"legacy_groups\":1,\"coefficient_columns\":161,\"coefficient_sources\":152," ,
            "\"total_batches\":74,\"coefficient_batches\":71," ,
            "\"terms\":6341}},\"bytes\":{{\"legacy_logical_output\":{}," ,
            "\"hybrid_logical_output\":{},\"validated_numerator_output\":{}," ,
            "\"validated_auxiliary_output\":{},\"validated_canonical_output\":{}," ,
            "\"shared_data_dual_workspace_arena\":{}," ,
            "\"workspace_span_each\":{},\"second_workspace_arena_delta\":{}}}," ,
            "\"device_memory\":{{\"total\":{},\"free_before_arena\":{}," ,
            "\"free_after_arena\":{},\"isolated_pool_used_after_arena\":{}," ,
            "\"isolated_pool_reserved_after_arena\":{}}}," ,
            "\"identity\":{{\"output_digest_encoding\":\"framed u32 little-endian v1\"," ,
            "\"topology_fixture_blake3\":\"{}\"," ,
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
        fixture.allocation_bytes,
        EXPECTED_SN3_WORKSPACE_SPAN_BYTES,
        EXPECTED_SN3_SECOND_WORKSPACE_ARENA_DELTA_BYTES,
        device_total_bytes,
        device_free_before_arena,
        device_free_after_arena,
        arena_pool.used_bytes,
        arena_pool.reserved_bytes,
        sn3.digest,
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
    let sn3 = load_sn3_topology_fixture(EXPECTED_SN3_TOPOLOGY_FIXTURE_BLAKE3);
    assert_sn3_shape(sn3.config, &sn3.topology, &sn3.requirements, &sn3.hybrid);
    let digest = input_recipe_digest(
        &sn3.topology,
        &sn3.requirements,
        &sn3.hybrid,
        &sn3.input_points,
    );
    assert_eq!(digest.to_hex().as_str(), EXPECTED_SN3_INPUT_RECIPE_BLAKE3);
    let mut changed_points = sn3.input_points.clone();
    let distinct = changed_points
        .iter()
        .position(|point| *point != changed_points[0])
        .expect("exact SN3 fixture must contain distinct sample points");
    changed_points.swap(0, distinct);
    assert_ne!(
        input_recipe_digest(
            &sn3.topology,
            &sn3.requirements,
            &sn3.hybrid,
            &changed_points,
        ),
        digest
    );
    let plan = benchmark_arena_plan(&sn3.topology, &sn3.requirements);
    assert_eq!(
        plan.allocation_bytes,
        EXPECTED_SN3_DUAL_WORKSPACE_ARENA_BYTES
    );
    assert_eq!(
        plan.allocation_bytes - EXPECTED_SN3_SINGLE_WORKSPACE_ARENA_BYTES,
        EXPECTED_SN3_SECOND_WORKSPACE_ARENA_DELTA_BYTES
    );
    assert_schedule_workspaces_are_disjoint(
        &sn3.requirements,
        &plan.legacy_slots,
        &plan.hybrid_slots,
    );
}

fn assert_sn3_shape(
    config: QuotientNumeratorWorkspaceConfig,
    topology: &[QuotientNumeratorColumnTopology],
    requirements: &QuotientNumeratorWorkspaceRequirements,
    hybrid: &stwo_backend_cuda::QuotientNumeratorHybridPlan,
) {
    assert_eq!(config, EXPECTED_SN3_CONFIG);
    assert!(!topology.is_empty());
    assert_eq!(
        topology
            .iter()
            .filter(|column| column.source_kind == QuotientNumeratorSourceKind::Coefficients)
            .count(),
        COEFFICIENT_COLUMNS
    );
    assert_eq!(requirements.term_count, TERMS);
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
        LEGACY_GROUP_COEFFICIENT_SOURCES
    );
    assert!(requirements.groups[1..]
        .iter()
        .all(|group| group.coefficient_source_count == 0));
    assert_eq!(requirements.batches.len(), LEGACY_BATCHES);
    assert_eq!(
        requirements
            .batches
            .iter()
            .map(|batch| batch.coefficient_count)
            .sum::<usize>(),
        LEGACY_GROUP_COEFFICIENT_SOURCES
    );
    assert_eq!(
        requirements
            .batches
            .iter()
            .filter(|batch| batch.coefficient_count != 0)
            .count(),
        COEFFICIENT_BATCHES
    );

    let report = hybrid.report();
    assert_eq!(report.eligible_group_count, ELIGIBLE_LOGS.len());
    assert_eq!(report.legacy_group_count, 1);
    assert_eq!(report.legacy_batch_count, LEGACY_BATCHES);
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
    legacy_slots: QuotientNumeratorWorkspaceSlots,
    hybrid_slots: QuotientNumeratorWorkspaceSlots,
    source_ids: Vec<ArenaSlotId>,
    destination_ids: Vec<[ArenaSlotId; 4]>,
    allocation_bytes: u64,
}

struct BenchmarkArenaPlan {
    layout: ArenaLayout,
    legacy_slots: QuotientNumeratorWorkspaceSlots,
    hybrid_slots: QuotientNumeratorWorkspaceSlots,
    source_ids: Vec<ArenaSlotId>,
    destination_ids: Vec<[ArenaSlotId; 4]>,
    allocation_bytes: u64,
}

impl BenchmarkArena {
    fn new(
        topology: &[QuotientNumeratorColumnTopology],
        requirements: &QuotientNumeratorWorkspaceRequirements,
    ) -> Self {
        let plan = benchmark_arena_plan(topology, requirements);
        let arena = DeviceArena::new(CudaExecContext::new().unwrap(), plan.layout).unwrap();
        Self {
            arena,
            legacy_slots: plan.legacy_slots,
            hybrid_slots: plan.hybrid_slots,
            source_ids: plan.source_ids,
            destination_ids: plan.destination_ids,
            allocation_bytes: plan.allocation_bytes,
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

fn benchmark_arena_plan(
    topology: &[QuotientNumeratorColumnTopology],
    requirements: &QuotientNumeratorWorkspaceRequirements,
) -> BenchmarkArenaPlan {
    let legacy_slots = workspace_slots(requirements, LEGACY_WORKSPACE_BASE);
    let hybrid_slots = workspace_slots(requirements, HYBRID_WORKSPACE_BASE);
    let mut specs = Vec::new();
    let mut cursor = 0usize;
    for (slots, expected_bytes) in [
        (&legacy_slots, EXPECTED_SN3_WORKSPACE_SPAN_BYTES),
        (&hybrid_slots, EXPECTED_SN3_WORKSPACE_SPAN_BYTES),
    ] {
        let workspace_start = cursor;
        for requirement in requirements.arena_slot_requirements(slots).unwrap() {
            push_spec(
                &mut specs,
                &mut cursor,
                requirement.id,
                requirement.len_words,
                requirement.alignment_words,
            );
        }
        assert_eq!((cursor - workspace_start) as u64 * 4, expected_bytes);
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
    assert!(allocation_bytes <= MAX_BENCHMARK_ARENA_BYTES);
    BenchmarkArenaPlan {
        layout: ArenaLayout::new(cursor, &specs).unwrap(),
        legacy_slots,
        hybrid_slots,
        source_ids,
        destination_ids,
        allocation_bytes,
    }
}

fn workspace_slots(
    requirements: &QuotientNumeratorWorkspaceRequirements,
    base: u32,
) -> QuotientNumeratorWorkspaceSlots {
    let id = |offset| ArenaSlotId(base + offset);
    QuotientNumeratorWorkspaceSlots {
        runtime_terms: id(0),
        group_term_indices: id(1),
        group_offsets: id(2),
        line_coefficients: id(3),
        term_points: id(4),
        batch_terms: id(5),
        batch_group_offsets: id(6),
        batch_source_ptrs: id(7),
        output_ptrs: id(8),
        output_log_sizes: id(9),
        coefficient_ptrs: (requirements.coefficient_pointer_words != 0).then_some(id(10)),
        coefficient_sizes: (requirements.coefficient_size_words != 0).then_some(id(11)),
        coefficient_output_ptrs: (requirements.coefficient_output_pointer_words != 0)
            .then_some(id(12)),
        lde_tile: (requirements.lde_tile_words != 0).then_some(id(13)),
    }
}

fn assert_schedule_workspaces_are_disjoint(
    requirements: &QuotientNumeratorWorkspaceRequirements,
    legacy: &QuotientNumeratorWorkspaceSlots,
    hybrid: &QuotientNumeratorWorkspaceSlots,
) {
    let legacy = requirements.arena_slot_requirements(legacy).unwrap();
    let hybrid = requirements.arena_slot_requirements(hybrid).unwrap();
    assert!(legacy
        .iter()
        .all(|left| hybrid.iter().all(|right| left.id != right.id)));
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
    config: QuotientNumeratorWorkspaceConfig,
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
            config,
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
            config,
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
