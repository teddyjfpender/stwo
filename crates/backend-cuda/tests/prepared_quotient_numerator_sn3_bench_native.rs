//! Ignored cheap-GPU benchmark for the sealed SN3 quotient-numerator shape.
//! gpu-lab-cohesion-review: topology construction, byte identity, graph replay timing, and the
//! single JSON result stay together so a benchmark cannot silently drift from its comparator.
//! Run receipts must place this JSON beside `nvidia-smi` identity/driver/clock output,
//! `nvcc --version`, `rustc -Vv`, and `git rev-parse HEAD`; those are environment facts, not test
//! semantics, so the existing pod wrapper owns them.
//! Identity-complete records additionally set `STWO_SN3_BOUNDARY_SOURCE_PROJECTION_SHA256` and
//! `STWO_SN3_BOUNDARY_CUDA_MODULE_SHA256`; a git HEAD alone is not an artifact identity.

#[path = "support/sn3_quotient_numerator_bench.rs"]
mod sn3_quotient_numerator_bench;
#[path = "support/sn3_quotient_topology_fixture.rs"]
mod sn3_quotient_topology_fixture;

use blake3::{Hash, Hasher};
use sn3_quotient_numerator_bench::{
    artifact_identity, assert_affine_pattern_sanity, assert_canonical_output,
    capture_canonical_output, input_recipe_digest, json_samples, percentile, poison_outputs,
    source_pattern_seed, upload_affine_pattern, TWIDDLE_PATTERN_SEED,
};
use sn3_quotient_topology_fixture::load_sn3_topology_fixture;
use stwo::core::circle::CirclePoint;
use stwo::core::fields::qm31::SecureField;
use stwo_backend_cuda::{
    gpu_memory_info, quotient_numerator_staged_single_write_plan_with_overflow_capacities,
    quotient_numerator_workspace_requirements, quotient_workspace_requirements, ArenaLayout,
    ArenaRangeSpec, ArenaSlice, ArenaSlotId, ArenaSlotSpec, CudaExecContext, CudaGraphExec,
    DeviceArena, PreparedNumeratorSchedule, PreparedQuotientGraph, PreparedQuotientNumeratorGraph,
    QuotientNumeratorColumn, QuotientNumeratorColumnSource, QuotientNumeratorColumnTopology,
    QuotientNumeratorDestination, QuotientNumeratorSource, QuotientNumeratorSourceKind,
    QuotientNumeratorStagedSingleWritePlan, QuotientNumeratorStagingRole,
    QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceRequirements,
    QuotientNumeratorWorkspaceSlots, QuotientWorkspaceConfig, QuotientWorkspaceRequirements,
    QuotientWorkspaceSlots,
};

const GROUP_LOGS: [u32; 19] = [
    23, 19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22,
];
const ELIGIBLE_LOGS: [u32; 18] = [
    19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22,
];
const EXPECTED_SN3_TOPOLOGY_CONFIG: QuotientNumeratorWorkspaceConfig =
    QuotientNumeratorWorkspaceConfig {
        lifting_log_size: 24,
        log_blowup_factor: 1,
        max_lde_tile_words: 1 << 24,
    };
const SN3_QUOTIENT_CONFIG: QuotientWorkspaceConfig = QuotientWorkspaceConfig {
    lifting_log_size: 24,
    log_blowup_factor: 1,
};
const COEFFICIENT_COLUMNS: usize = 161;
const LEGACY_GROUP_COEFFICIENT_SOURCES: usize = 152;
const LEGACY_BATCHES: usize = 74;
const COEFFICIENT_BATCHES: usize = 71;
const TERMS: usize = 6_341;
const LEGACY_LOGICAL_OUTPUT_BYTES: u64 = 59_993_989_376;
const HYBRID_LOGICAL_OUTPUT_BYTES: u64 = 20_266_867_968;
const STAGED_PRIMARY_WORDS: usize = 536_870_912;
const STAGED_OVERFLOW_WORDS: usize = 452_984_832;
const RETAINED_COLUMN_COUNT: usize = LEGACY_GROUP_COEFFICIENT_SOURCES;
const RETAINED_IMAGE_WORDS: usize = 979_965_856;
const MUTATED_RETAINED_COLUMN: usize = 102;
const EXPECTED_CONTROL_GRAPH_KERNEL_NODES: u64 = 141;
const EXPECTED_RETAINED_GRAPH_KERNEL_NODES: u64 = 38;
const CONTROL_LIVE: u16 = 0b01;
const RETAINED_LIVE: u16 = 0b10;
const BOTH_LIVE: u16 = CONTROL_LIVE | RETAINED_LIVE;
const STAGED_DESCRIPTOR_WORKSPACE_BYTES: u64 = 787_904;
const QUOTIENT_INCREMENTAL_ARENA_WORDS: usize = 104_857_792;
const QUOTIENT_INCREMENTAL_ARENA_BYTES: u64 = 419_431_168;
const EXPECTED_SN3_ARENA_BYTES: u64 = 46_133_748_992;
const FRI_INPUT_OUTPUT_BYTES: u64 = 268_435_456;
const FRI_COPY_CHUNK_WORDS: usize = 1 << 20;
const MAX_BENCHMARK_ARENA_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_ITERATIONS: usize = 20;
const EXPECTED_SN3_TOPOLOGY_FIXTURE_BLAKE3: &str =
    "ea31e3ff054c8d12d32d5b84a3d712987b31bb1fd3fb044fb27758453b49fbda";
const EXPECTED_SN3_INPUT_RECIPE_BLAKE3: &str =
    "e4c2f871c2d05b81588a5407f06cb49c7ed76834d2e363d2214bd34e7defcf31";
const EXPECTED_SN3_BOUNDARY_INPUT_RECIPE_BLAKE3: &str =
    "a88aaa1f23b4c22201c9f9b1ed0aedc17bde8ababb50eae72a828a0eeab06ae9";
const INVERSE_TWIDDLE_PATTERN_SEED: u64 = 32_452_843;
const EAGER_LEGACY_POISON: u32 = 0xdead_beef;
const EAGER_HYBRID_POISON: u32 = 0xa5a5_5a5a;
const CAPTURE_LEGACY_POISON: u32 = 0x1357_9bdf;
const CAPTURE_HYBRID_POISON: u32 = 0x2468_ace0;
const POST_TIMING_LEGACY_POISON: u32 = 0x0bad_f00d;
const POST_TIMING_HYBRID_POISON: u32 = 0xc001_d00d;
const TIMED_LEGACY_POISON: u32 = 0x3141_5926;
const TIMED_HYBRID_POISON: u32 = 0x2718_2818;
const RETAINED_IMAGE_POISON: u32 = 0xfeed_5eed;

const OODS_POINTS: ArenaSlotId = ArenaSlotId(100);
const OODS_VALUES: ArenaSlotId = ArenaSlotId(101);
const RANDOM_COEFFICIENT: ArenaSlotId = ArenaSlotId(102);
const SAMPLE_POINTS_OUTPUT: ArenaSlotId = ArenaSlotId(103);
const FIRST_TERMS_OUTPUT: ArenaSlotId = ArenaSlotId(104);
const TWIDDLES: ArenaSlotId = ArenaSlotId(105);
const SHARED_STAGED_PRIMARY: ArenaSlotId = ArenaSlotId(106);
const SHARED_STAGED_OVERFLOW: ArenaSlotId = ArenaSlotId(107);
const QUOTIENT_WORKSPACE_BASE: u32 = 200;
const INVERSE_TWIDDLES: ArenaSlotId = ArenaSlotId(207);
const SOURCE_BASE: u32 = 1_000;
const OUTPUT_BASE: u32 = 10_000;
const PACKED_WORKSPACE_BASE: u32 = 1;
const DIRECT_WORKSPACE_BASE: u32 = 20;
const RETAINED_IMAGE_BASE: u32 = 20_000;

#[test]
#[ignore = "requires CUDA and the reported 42.97 GiB numerator-to-FRI-input arena"]
fn sn3_staged_group_direct_cuda_event_benchmark() {
    let sn3 = load_sn3_topology_fixture(EXPECTED_SN3_TOPOLOGY_FIXTURE_BLAKE3);
    assert_sn3_shape(sn3.config, &sn3.topology, &sn3.requirements, &sn3.hybrid);
    let input_recipe_blake3 = input_recipe_digest(
        &sn3.topology,
        &sn3.requirements,
        &sn3.hybrid,
        &sn3.input_points,
    );
    let quotient_requirements =
        quotient_workspace_requirements(SN3_QUOTIENT_CONFIG, &GROUP_LOGS).unwrap();
    assert_sn3_quotient_shape(&quotient_requirements);
    let boundary_input_recipe_blake3 =
        boundary_input_recipe_digest(input_recipe_blake3, &quotient_requirements);
    assert_eq!(
        boundary_input_recipe_blake3.to_hex().as_str(),
        EXPECTED_SN3_BOUNDARY_INPUT_RECIPE_BLAKE3
    );
    let staged_config = staged_config(sn3.config);
    let staged_requirements =
        quotient_numerator_workspace_requirements(staged_config, &sn3.topology).unwrap();
    let staged_plan = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
        staged_config,
        &sn3.topology,
        &[STAGED_OVERFLOW_WORDS],
    )
    .unwrap();
    assert_staged_sn3_shape(&staged_requirements, &staged_plan);
    let retained_topology = retained_topology(&sn3.topology, &staged_plan);
    let retained_config = staged_config;
    let retained_requirements =
        quotient_numerator_workspace_requirements(retained_config, &retained_topology).unwrap();
    let retained_plan = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
        retained_config,
        &retained_topology,
        &[],
    )
    .unwrap();
    assert_retained_sn3_shape(&retained_requirements, &retained_plan);

    // All exact topology assertions above intentionally precede the first CUDA allocation.
    let (device_free_before_arena, device_total_bytes) = gpu_memory_info();
    let fixture = BenchmarkArena::new(
        &sn3.topology,
        &staged_requirements,
        &staged_plan,
        &retained_requirements,
        &quotient_requirements,
    );
    assert!(fixture.allocation_bytes <= MAX_BENCHMARK_ARENA_BYTES);
    assert_eq!(fixture.allocation_bytes, EXPECTED_SN3_ARENA_BYTES);
    let retained_manifest_blake3 = fixture.retained_manifest_blake3;
    let arena_pool = fixture.arena.context().pool_memory().unwrap();
    let (device_free_after_arena, device_total_after_arena) = gpu_memory_info();
    assert_eq!(device_total_after_arena, device_total_bytes);
    assert!(arena_pool.used_bytes >= fixture.allocation_bytes as usize);
    assert!(arena_pool.reserved_bytes >= arena_pool.used_bytes);

    let columns = fixture.columns(&sn3.topology, &fixture.source_ids);
    let retained_columns = fixture.columns(&retained_topology, &fixture.retained_source_ids);
    let destinations = fixture.destinations(&staged_requirements);
    let control = prepare(
        &fixture,
        &columns,
        &destinations,
        &fixture.control_slots,
        staged_config,
        true,
        &[SHARED_STAGED_OVERFLOW],
    );
    assert_eq!(
        control.schedule(),
        PreparedNumeratorSchedule::StagedGroupDirect {
            output_rows: 25_165_264,
        }
    );
    let quotient_sources = control.quotient_sources();
    let quotient = PreparedQuotientGraph::prepare(
        &fixture.arena,
        SN3_QUOTIENT_CONFIG,
        &quotient_sources,
        fixture.arena.bind(TWIDDLES).unwrap(),
        fixture.arena.bind(INVERSE_TWIDDLES).unwrap(),
        &fixture.quotient_slots,
    )
    .unwrap();
    assert_eq!(
        quotient.output_evaluation().len_words(),
        quotient_requirements.output_value_words
    );

    initialize(
        &fixture,
        &sn3.topology,
        &staged_requirements,
        &quotient_requirements,
        &sn3.input_points,
        EAGER_LEGACY_POISON,
    );

    // The packed implementation is an independent eager oracle only. Its
    // descriptor workspace is then reused by the retained direct graph.
    let (eager, eager_fri_input) = {
        let oracle = prepare(
            &fixture,
            &columns,
            &destinations,
            &fixture.oracle_slots,
            staged_config,
            false,
            &[SHARED_STAGED_OVERFLOW],
        );
        assert_eq!(
            oracle.schedule(),
            PreparedNumeratorSchedule::StagedPackedSingleWrite {
                packed_output_rows: 25_165_264,
            }
        );
        poison_fri_input(&fixture, &quotient, EAGER_LEGACY_POISON);
        oracle.launch().unwrap();
        quotient.launch().unwrap();
        fixture.arena.context().sync().unwrap();
        (
            capture_canonical_output(&fixture, &staged_requirements),
            capture_fri_input(&fixture, &quotient),
        )
    };

    let retained = prepare(
        &fixture,
        &retained_columns,
        &destinations,
        &fixture.retained_slots,
        retained_config,
        true,
        &[],
    );
    assert_eq!(
        retained.schedule(),
        PreparedNumeratorSchedule::StagedGroupDirect {
            output_rows: 25_165_264,
        }
    );
    assert_quotient_sources_same(&retained.quotient_sources(), &control.quotient_sources());

    poison_outputs(&fixture, &staged_requirements, EAGER_LEGACY_POISON);
    poison_fri_input(&fixture, &quotient, EAGER_LEGACY_POISON);
    control.launch().unwrap();
    quotient.launch().unwrap();
    fixture.arena.context().sync().unwrap();
    let eager_control_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "eager coefficient-direct",
    );
    let eager_control_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "eager coefficient-direct",
    );

    poison_outputs(&fixture, &staged_requirements, EAGER_HYBRID_POISON);
    poison_fri_input(&fixture, &quotient, EAGER_HYBRID_POISON);
    retained.launch().unwrap();
    quotient.launch().unwrap();
    fixture.arena.context().sync().unwrap();
    let eager_retained_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "eager retained-direct",
    );
    let eager_retained_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "eager retained-direct",
    );

    let validated_numerator_output_bytes = staged_requirements
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
    assert_eq!(eager_fri_input.len_bytes(), FRI_INPUT_OUTPUT_BYTES);

    let capture = fixture.arena.context().capture().unwrap();
    control.launch().unwrap();
    quotient.launch().unwrap();
    let control_graph = capture.finish().unwrap();
    let capture = fixture.arena.context().capture().unwrap();
    retained.launch().unwrap();
    quotient.launch().unwrap();
    let retained_graph = capture.finish().unwrap();
    assert_eq!(
        control_graph.kernel_nodes(),
        EXPECTED_CONTROL_GRAPH_KERNEL_NODES
    );
    assert_eq!(
        retained_graph.kernel_nodes(),
        EXPECTED_RETAINED_GRAPH_KERNEL_NODES
    );
    assert_eq!(
        control_graph.kernel_nodes() - retained_graph.kernel_nodes(),
        103
    );
    let (device_free_after_graphs, device_total_after_graphs) = gpu_memory_info();
    assert_eq!(device_total_after_graphs, device_total_bytes);

    poison_outputs(&fixture, &staged_requirements, CAPTURE_LEGACY_POISON);
    poison_fri_input(&fixture, &quotient, CAPTURE_LEGACY_POISON);
    control_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let captured_control_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "captured coefficient-direct",
    );
    let captured_control_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "captured coefficient-direct",
    );
    poison_outputs(&fixture, &staged_requirements, CAPTURE_HYBRID_POISON);
    poison_fri_input(&fixture, &quotient, CAPTURE_HYBRID_POISON);
    retained_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let captured_retained_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "captured retained-direct",
    );
    let captured_retained_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "captured retained-direct",
    );

    // Two uninterrupted retained replays prove the image is persistent and
    // does not require a coefficient graph immediately before each launch.
    let mut persistence_blake3 = Vec::with_capacity(2);
    let mut persistence_fri_blake3 = Vec::with_capacity(2);
    for replay in 0..2 {
        poison_outputs(
            &fixture,
            &staged_requirements,
            CAPTURE_HYBRID_POISON ^ replay,
        );
        poison_fri_input(&fixture, &quotient, CAPTURE_HYBRID_POISON ^ replay);
        retained_graph.launch(fixture.arena.context()).unwrap();
        fixture.arena.context().sync().unwrap();
        persistence_blake3.push(assert_canonical_output(
            &fixture,
            &staged_requirements,
            &eager,
            "uninterrupted retained replay",
        ));
        persistence_fri_blake3.push(assert_fri_input(
            &fixture,
            &quotient,
            &eager_fri_input,
            "uninterrupted retained replay",
        ));
    }

    // Poison one sampled resident column. A retained replay must consume the
    // mutation, while the coefficient graph must rematerialize and restore it.
    poison_retained_image(&fixture, MUTATED_RETAINED_COLUMN, RETAINED_IMAGE_POISON);
    poison_outputs(&fixture, &staged_requirements, CAPTURE_HYBRID_POISON);
    poison_fri_input(&fixture, &quotient, CAPTURE_HYBRID_POISON);
    retained_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let mutated_output = capture_canonical_output(&fixture, &staged_requirements);
    let mutated_fri_input = capture_fri_input(&fixture, &quotient);
    assert_ne!(
        mutated_output.digest(),
        eager.digest(),
        "retained graph ignored the poisoned resident image"
    );
    assert_ne!(
        mutated_fri_input.digest, eager_fri_input.digest,
        "FRI boundary ignored the poisoned resident image"
    );

    control_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let restored_control_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "coefficient-direct mutation restoration",
    );
    let restored_control_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "coefficient-direct mutation restoration",
    );
    for replay in 0..2 {
        poison_outputs(
            &fixture,
            &staged_requirements,
            POST_TIMING_HYBRID_POISON ^ replay,
        );
        poison_fri_input(&fixture, &quotient, POST_TIMING_HYBRID_POISON ^ replay);
        retained_graph.launch(fixture.arena.context()).unwrap();
        fixture.arena.context().sync().unwrap();
        assert_canonical_output(
            &fixture,
            &staged_requirements,
            &eager,
            "restored retained persistence replay",
        );
        assert_fri_input(
            &fixture,
            &quotient,
            &eager_fri_input,
            "restored retained persistence replay",
        );
    }

    for round in 0..DEFAULT_WARMUPS {
        if round % 2 == 0 {
            replay_cuda_ms(&control_graph, fixture.arena.context());
            replay_cuda_ms(&retained_graph, fixture.arena.context());
        } else {
            replay_cuda_ms(&retained_graph, fixture.arena.context());
            replay_cuda_ms(&control_graph, fixture.arena.context());
        }
    }

    let iterations = std::env::var("STWO_SN3_BOUNDARY_BENCH_ITERS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("STWO_SN3_BOUNDARY_BENCH_ITERS must be an integer")
        })
        .unwrap_or(DEFAULT_ITERATIONS);
    assert!(
        iterations >= 6 && iterations % 2 == 0,
        "formal coefficient-direct/retained-direct A/B requires an even iteration count of at least six"
    );

    poison_outputs(&fixture, &staged_requirements, TIMED_LEGACY_POISON);
    poison_fri_input(&fixture, &quotient, TIMED_LEGACY_POISON);
    let causal_validation_control_ms = replay_cuda_ms(&control_graph, fixture.arena.context());
    let timed_control_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "causal validation coefficient-direct replay",
    );
    let timed_control_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "causal validation coefficient-direct replay",
    );
    poison_outputs(&fixture, &staged_requirements, TIMED_HYBRID_POISON);
    poison_fri_input(&fixture, &quotient, TIMED_HYBRID_POISON);
    let causal_validation_retained_ms = replay_cuda_ms(&retained_graph, fixture.arena.context());
    let timed_retained_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "causal validation retained-direct replay",
    );
    let timed_retained_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "causal validation retained-direct replay",
    );

    replay_cuda_ms(&control_graph, fixture.arena.context());
    replay_cuda_ms(&retained_graph, fixture.arena.context());

    let mut control_ms = Vec::with_capacity(iterations);
    let mut retained_ms = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        if iteration % 2 == 0 {
            control_ms.push(replay_cuda_ms(&control_graph, fixture.arena.context()));
            retained_ms.push(replay_cuda_ms(&retained_graph, fixture.arena.context()));
        } else {
            retained_ms.push(replay_cuda_ms(&retained_graph, fixture.arena.context()));
            control_ms.push(replay_cuda_ms(&control_graph, fixture.arena.context()));
        }
    }

    poison_outputs(&fixture, &staged_requirements, POST_TIMING_LEGACY_POISON);
    poison_fri_input(&fixture, &quotient, POST_TIMING_LEGACY_POISON);
    control_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let post_timing_control_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "post-timing coefficient-direct",
    );
    let post_timing_control_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "post-timing coefficient-direct",
    );
    poison_outputs(&fixture, &staged_requirements, POST_TIMING_HYBRID_POISON);
    poison_fri_input(&fixture, &quotient, POST_TIMING_HYBRID_POISON);
    retained_graph.launch(fixture.arena.context()).unwrap();
    fixture.arena.context().sync().unwrap();
    let post_timing_retained_blake3 = assert_canonical_output(
        &fixture,
        &staged_requirements,
        &eager,
        "post-timing retained-direct",
    );
    let post_timing_retained_fri_blake3 = assert_fri_input(
        &fixture,
        &quotient,
        &eager_fri_input,
        "post-timing retained-direct",
    );

    let artifact_identity = artifact_identity();
    let control_p50 = percentile(&control_ms, 50);
    let control_p95 = percentile(&control_ms, 95);
    let retained_p50 = percentile(&retained_ms, 50);
    let retained_p95 = percentile(&retained_ms, 95);
    let boundary_source_projection_sha256: serde_json::Value =
        serde_json::from_str(&artifact_identity.boundary_source_projection_json()).unwrap();
    let boundary_cuda_module_sha256: serde_json::Value =
        serde_json::from_str(&artifact_identity.boundary_cuda_module_json()).unwrap();
    let control_samples: serde_json::Value =
        serde_json::from_str(&json_samples(&control_ms)).unwrap();
    let retained_samples: serde_json::Value =
        serde_json::from_str(&json_samples(&retained_ms)).unwrap();
    println!(
        "{}",
        serde_json::json!({
            "schema": "stwo.sn3_numerator_to_fri_input_fixed_image.cuda_event.v1",
            "timing_scope": "per-replay CUDA-event device elapsed time for captured numerator then ordinary quotient-to-FRI-input graph",
            "result_class": "CURRENT TOPOLOGY / 0 NEW DELIVERY CREDIT; same-binary retained-direct versus coefficient-direct counterfactual; not end-to-end proof MHz",
            "topology": {
                "groups": 19,
                "terms": 6_341,
                "control_staged_batches": 19,
                "retained_staged_batches": 18,
                "converted_control_lde_aliases": RETAINED_COLUMN_COUNT,
                "retained_evaluation_log_sizes": retained_requirements.batches.iter().map(|batch| batch.evaluation_log_size).collect::<Vec<_>>(),
                "retained_coefficient_sources": 0,
                "output_rows": 25_165_264,
            },
            "bytes": {
                "validated_numerator_output": validated_numerator_output_bytes,
                "validated_auxiliary_output": validated_auxiliary_output_bytes,
                "validated_canonical_output": validated_canonical_output_bytes,
                "validated_fri_input_output": FRI_INPUT_OUTPUT_BYTES,
                "retained_image": RETAINED_IMAGE_WORDS as u64 * 4,
                "arena": fixture.allocation_bytes,
                "candidate_additional_physical": 0,
                "descriptor_workspace_each": STAGED_DESCRIPTOR_WORKSPACE_BYTES,
                "control_primary": STAGED_PRIMARY_WORDS as u64 * 4,
                "control_overflow": STAGED_OVERFLOW_WORDS as u64 * 4,
                "retained_staging_roles": 0,
                "incremental_quotient_workspace": QUOTIENT_INCREMENTAL_ARENA_BYTES,
            },
            "device_memory": {
                "total": device_total_bytes,
                "free_before_arena": device_free_before_arena,
                "free_after_arena": device_free_after_arena,
                "free_after_graph_instantiation": device_free_after_graphs,
                "isolated_pool_used_after_arena": arena_pool.used_bytes,
                "isolated_pool_reserved_after_arena": arena_pool.reserved_bytes,
            },
            "identity": {
                "topology_fixture_blake3": sn3.digest.to_string(),
                "numerator_input_recipe_blake3": input_recipe_blake3.to_string(),
                "boundary_input_recipe_blake3": boundary_input_recipe_blake3.to_string(),
                "retained_manifest_blake3": retained_manifest_blake3.to_string(),
                "oracle_blake3": eager.digest().to_string(),
                "eager_control_blake3": eager_control_blake3.to_string(),
                "eager_retained_blake3": eager_retained_blake3.to_string(),
                "captured_control_blake3": captured_control_blake3.to_string(),
                "captured_retained_blake3": captured_retained_blake3.to_string(),
                "persistence_blake3": persistence_blake3.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "mutation_numerator_blake3": mutated_output.digest().to_string(),
                "restored_control_blake3": restored_control_blake3.to_string(),
                "causal_control_blake3": timed_control_blake3.to_string(),
                "causal_retained_blake3": timed_retained_blake3.to_string(),
                "post_timing_control_blake3": post_timing_control_blake3.to_string(),
                "post_timing_retained_blake3": post_timing_retained_blake3.to_string(),
            },
            "fri_input_identity": {
                "oracle_blake3": eager_fri_input.digest.to_string(),
                "eager_control_blake3": eager_control_fri_blake3.to_string(),
                "eager_retained_blake3": eager_retained_fri_blake3.to_string(),
                "captured_control_blake3": captured_control_fri_blake3.to_string(),
                "captured_retained_blake3": captured_retained_fri_blake3.to_string(),
                "persistence_blake3": persistence_fri_blake3.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "mutation_blake3": mutated_fri_input.digest.to_string(),
                "restored_control_blake3": restored_control_fri_blake3.to_string(),
                "causal_control_blake3": timed_control_fri_blake3.to_string(),
                "causal_retained_blake3": timed_retained_fri_blake3.to_string(),
                "post_timing_control_blake3": post_timing_control_fri_blake3.to_string(),
                "post_timing_retained_blake3": post_timing_retained_fri_blake3.to_string(),
                "exact_word_comparison": true,
            },
            "correctness": {
                "candidate_only_consecutive_replays": 2,
                "mutation_consumed": true,
                "control_restored_image": true,
                "post_restore_candidate_only_replays": 2,
                "capture_revalidated": true,
                "post_timing_revalidated": true,
            },
            "capture_topology": {
                "control_kernel_nodes": control_graph.kernel_nodes(),
                "retained_kernel_nodes": retained_graph.kernel_nodes(),
                "removed_kernel_nodes": control_graph.kernel_nodes() - retained_graph.kernel_nodes(),
            },
            "artifact_identity": {
                "boundary_seal_blake3": artifact_identity.boundary_seal_blake3.to_string(),
                "boundary_rust_source_blake3": artifact_identity.boundary_rust_source_blake3.to_string(),
                "ordinary_cuda_source_blake3": artifact_identity.ordinary_cuda_source_blake3.to_string(),
                "test_binary_blake3": artifact_identity.test_binary_blake3.to_string(),
                "boundary_source_projection_sha256": boundary_source_projection_sha256,
                "boundary_cuda_module_sha256": boundary_cuda_module_sha256,
                "cuda_build_mode": artifact_identity.cuda_build_mode,
                "expected_cuda_module_build_identity": artifact_identity.expected_cuda_module_build_identity.to_string(),
                "linked_cuda_module_build_identity": artifact_identity.linked_cuda_module_build_identity.to_string(),
                "cuda_module_target_sms": artifact_identity.cuda_module_target_sms,
                "identity_complete": artifact_identity.is_complete(),
            },
            "comparator": {
                "control": "counterfactual coefficient-direct; materializes all 152 sampled coefficient LDEs every replay",
                "candidate": "current-policy analogue retained-direct; consumes all 152 resident evaluation aliases and materializes no coefficient sources",
                "delivery_credit": "zero; this measures work already retired by the corrected production FixedImage policy",
                "cold_retained_image_setup_in_timing": false,
                "numerator_comparator_source_blake3": artifact_identity.numerator_comparator_source_blake3.to_string(),
                "independent_truth": false,
            },
            "warmups_each": DEFAULT_WARMUPS,
            "equalization_replays_each": 1,
            "iterations_each": iterations,
            "causal_validation_cuda_event_ms": {
                "control": causal_validation_control_ms,
                "retained": causal_validation_retained_ms,
                "excluded_from_samples": true,
            },
            "samples_ms": {
                "control": control_samples,
                "retained": retained_samples,
            },
            "cuda_event_ms": {
                "control": {"p50": control_p50, "p95": control_p95},
                "retained": {"p50": retained_p50, "p95": retained_p95},
            },
            "speedup": {
                "p50": control_p50 / retained_p50,
                "p95": control_p95 / retained_p95,
                "current_topology_vs_counterfactual_saved_p50_ms": control_p50 - retained_p50,
                "new_delivery_credit_ms": 0.0,
            },
        })
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
    let quotient_requirements =
        quotient_workspace_requirements(SN3_QUOTIENT_CONFIG, &GROUP_LOGS).unwrap();
    assert_sn3_quotient_shape(&quotient_requirements);
    assert_eq!(
        boundary_input_recipe_digest(digest, &quotient_requirements)
            .to_hex()
            .as_str(),
        EXPECTED_SN3_BOUNDARY_INPUT_RECIPE_BLAKE3
    );
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
    let config = staged_config(sn3.config);
    let requirements = quotient_numerator_workspace_requirements(config, &sn3.topology).unwrap();
    let staged = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
        config,
        &sn3.topology,
        &[STAGED_OVERFLOW_WORDS],
    )
    .unwrap();
    assert_staged_sn3_shape(&requirements, &staged);
    let retained_topology = retained_topology(&sn3.topology, &staged);
    let retained_config = config;
    let retained_requirements =
        quotient_numerator_workspace_requirements(retained_config, &retained_topology).unwrap();
    let retained = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
        retained_config,
        &retained_topology,
        &[],
    )
    .unwrap();
    assert_retained_sn3_shape(&retained_requirements, &retained);
    let plan = benchmark_arena_plan(
        &sn3.topology,
        &requirements,
        &staged,
        &retained_requirements,
        &quotient_requirements,
    );
    assert_eq!(plan.allocation_bytes, EXPECTED_SN3_ARENA_BYTES);
    assert_descriptor_workspaces_are_disjoint(
        &requirements,
        &plan.oracle_slots,
        &plan.control_slots,
    );
    assert_reused_workspace_shape(&plan, &staged);
}

fn assert_sn3_shape(
    config: QuotientNumeratorWorkspaceConfig,
    topology: &[QuotientNumeratorColumnTopology],
    requirements: &QuotientNumeratorWorkspaceRequirements,
    hybrid: &stwo_backend_cuda::QuotientNumeratorHybridPlan,
) {
    assert_eq!(config, EXPECTED_SN3_TOPOLOGY_CONFIG);
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

fn assert_sn3_quotient_shape(requirements: &QuotientWorkspaceRequirements) {
    assert_eq!(requirements.config, SN3_QUOTIENT_CONFIG);
    assert_eq!(requirements.subdomain_log_size, 23);
    assert_eq!(requirements.sample_count, 19);
    assert_eq!(requirements.sample_point_words, 152);
    assert_eq!(requirements.first_linear_term_words, 76);
    assert_eq!(requirements.partial_log_size_words, 19);
    assert_eq!(requirements.partial_pointer_words, 152);
    assert_eq!(requirements.coordinate_pointer_words, 8);
    assert_eq!(requirements.coefficient_size_words, 4);
    assert_eq!(requirements.subdomain_value_words, 33_554_432);
    assert_eq!(requirements.output_value_words, 67_108_864);
    assert_eq!(requirements.forward_twiddle_words, 8_388_608);
    assert_eq!(requirements.inverse_twiddle_words, 4_194_304);
    assert_eq!(
        u64::try_from(requirements.output_value_words).unwrap() * 4,
        FRI_INPUT_OUTPUT_BYTES
    );
}

fn staged_config(
    topology_config: QuotientNumeratorWorkspaceConfig,
) -> QuotientNumeratorWorkspaceConfig {
    QuotientNumeratorWorkspaceConfig {
        max_lde_tile_words: 32 * (1usize << topology_config.lifting_log_size),
        ..topology_config
    }
}

fn retained_topology(
    control: &[QuotientNumeratorColumnTopology],
    staged: &QuotientNumeratorStagedSingleWritePlan,
) -> Vec<QuotientNumeratorColumnTopology> {
    let mut retained = control.to_vec();
    assert_eq!(staged.coefficient_ldes().len(), RETAINED_COLUMN_COUNT);
    for lde in staged.coefficient_ldes() {
        let column_index = lde.column();
        let column = &mut retained[column_index];
        assert_eq!(
            column.source_kind,
            QuotientNumeratorSourceKind::Coefficients
        );
        assert!(
            !column.samples.is_empty(),
            "retained FixedImage column {column_index} must be sampled"
        );
        assert_eq!(lde.evaluation_log_size(), column.coefficient_log_size + 1);
        column.source_kind = QuotientNumeratorSourceKind::Evaluation;
    }
    assert_eq!(
        retained
            .iter()
            .filter(|column| {
                column.source_kind == QuotientNumeratorSourceKind::Evaluation
                    && !column.samples.is_empty()
            })
            .count()
            - control
                .iter()
                .filter(|column| {
                    column.source_kind == QuotientNumeratorSourceKind::Evaluation
                        && !column.samples.is_empty()
                })
                .count(),
        RETAINED_COLUMN_COUNT,
    );
    assert!(retained.iter().all(|column| {
        column.source_kind == QuotientNumeratorSourceKind::Evaluation || column.samples.is_empty()
    }));
    retained
}

fn assert_staged_sn3_shape(
    requirements: &QuotientNumeratorWorkspaceRequirements,
    staged: &QuotientNumeratorStagedSingleWritePlan,
) {
    assert_eq!(requirements.config.max_lde_tile_words, STAGED_PRIMARY_WORDS);
    assert_eq!(requirements.batches.len(), 19);
    assert_eq!(requirements.term_count, TERMS);
    assert_eq!(
        requirements
            .groups
            .iter()
            .map(|group| group.log_size)
            .collect::<Vec<_>>(),
        GROUP_LOGS
    );
    assert_eq!(staged.requirements(), requirements);
    assert_eq!(staged.packed_output_rows(), 25_165_264);
    assert_eq!(staged.overflow_role_words(), [STAGED_OVERFLOW_WORDS]);
    let report = staged.report();
    assert_eq!(report.factor32_batch_count, 19);
    assert_eq!(report.factor32_staging_words, STAGED_PRIMARY_WORDS);
    assert_eq!(report.primary_staging_words, 526_981_024);
    assert_eq!(report.overflow_staging_words, STAGED_OVERFLOW_WORDS);
    assert_eq!(report.overflow_staging_role_count, 1);
}

fn assert_retained_sn3_shape(
    requirements: &QuotientNumeratorWorkspaceRequirements,
    staged: &QuotientNumeratorStagedSingleWritePlan,
) {
    assert_eq!(requirements.config.max_lde_tile_words, STAGED_PRIMARY_WORDS);
    assert_eq!(
        requirements.config.lifting_log_size,
        EXPECTED_SN3_TOPOLOGY_CONFIG.lifting_log_size
    );
    assert_eq!(
        requirements.config.log_blowup_factor,
        EXPECTED_SN3_TOPOLOGY_CONFIG.log_blowup_factor
    );
    assert_eq!(requirements.batches.len(), 18);
    assert_eq!(requirements.term_count, TERMS);
    assert_eq!(
        requirements
            .groups
            .iter()
            .map(|group| group.log_size)
            .collect::<Vec<_>>(),
        GROUP_LOGS
    );
    assert_eq!(requirements.groups[0].coefficient_source_count, 0);
    assert_eq!(
        requirements
            .batches
            .iter()
            .map(|batch| batch.coefficient_count)
            .sum::<usize>(),
        0
    );
    assert_eq!(requirements.lde_tile_words, 0);
    assert_eq!(requirements.forward_twiddle_words, 0);
    assert_eq!(staged.requirements(), requirements);
    assert_eq!(staged.packed_output_rows(), 25_165_264);
    assert!(staged.coefficient_ldes().is_empty());
    assert!(staged.overflow_role_words().is_empty());
    let report = staged.report();
    assert_eq!(report.factor32_batch_count, 18);
    assert_eq!(report.coefficient_source_count, 0);
    assert_eq!(report.total_staging_words, 0);
    assert_eq!(report.factor32_staging_words, 0);
    assert_eq!(report.primary_staging_words, 0);
    assert_eq!(report.overflow_staging_words, 0);
    assert_eq!(report.overflow_staging_role_count, 0);
}

struct BenchmarkArena {
    arena: DeviceArena,
    oracle_slots: QuotientNumeratorWorkspaceSlots,
    control_slots: QuotientNumeratorWorkspaceSlots,
    retained_slots: QuotientNumeratorWorkspaceSlots,
    quotient_slots: QuotientWorkspaceSlots,
    source_ids: Vec<ArenaSlotId>,
    retained_source_ids: Vec<ArenaSlotId>,
    destination_ids: Vec<[ArenaSlotId; 4]>,
    retained_manifest_blake3: Hash,
    allocation_bytes: u64,
}

struct BenchmarkArenaPlan {
    layout: ArenaLayout,
    oracle_slots: QuotientNumeratorWorkspaceSlots,
    control_slots: QuotientNumeratorWorkspaceSlots,
    retained_slots: QuotientNumeratorWorkspaceSlots,
    quotient_slots: QuotientWorkspaceSlots,
    source_ids: Vec<ArenaSlotId>,
    retained_source_ids: Vec<ArenaSlotId>,
    destination_ids: Vec<[ArenaSlotId; 4]>,
    retained_manifest_blake3: Hash,
    allocation_bytes: u64,
}

impl BenchmarkArena {
    fn new(
        topology: &[QuotientNumeratorColumnTopology],
        requirements: &QuotientNumeratorWorkspaceRequirements,
        staged: &QuotientNumeratorStagedSingleWritePlan,
        retained_requirements: &QuotientNumeratorWorkspaceRequirements,
        quotient_requirements: &QuotientWorkspaceRequirements,
    ) -> Self {
        let plan = benchmark_arena_plan(
            topology,
            requirements,
            staged,
            retained_requirements,
            quotient_requirements,
        );
        let arena = DeviceArena::new(CudaExecContext::new().unwrap(), plan.layout).unwrap();
        Self {
            arena,
            oracle_slots: plan.oracle_slots,
            control_slots: plan.control_slots,
            retained_slots: plan.retained_slots,
            quotient_slots: plan.quotient_slots,
            source_ids: plan.source_ids,
            retained_source_ids: plan.retained_source_ids,
            destination_ids: plan.destination_ids,
            retained_manifest_blake3: plan.retained_manifest_blake3,
            allocation_bytes: plan.allocation_bytes,
        }
    }

    fn columns(
        &self,
        topology: &[QuotientNumeratorColumnTopology],
        source_ids: &[ArenaSlotId],
    ) -> Vec<QuotientNumeratorColumn> {
        assert_eq!(topology.len(), source_ids.len());
        topology
            .iter()
            .zip(source_ids)
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
    staged: &QuotientNumeratorStagedSingleWritePlan,
    retained_requirements: &QuotientNumeratorWorkspaceRequirements,
    quotient_requirements: &QuotientWorkspaceRequirements,
) -> BenchmarkArenaPlan {
    let oracle_slots = workspace_slots(requirements, PACKED_WORKSPACE_BASE, SHARED_STAGED_PRIMARY);
    let control_slots = workspace_slots(requirements, DIRECT_WORKSPACE_BASE, SHARED_STAGED_PRIMARY);
    let retained_slots = workspace_slots(
        retained_requirements,
        PACKED_WORKSPACE_BASE,
        SHARED_STAGED_PRIMARY,
    );
    let quotient_slots = quotient_workspace_slots();
    let mut specs = Vec::new();
    let mut cursor = 0usize;
    for slots in [&oracle_slots, &control_slots] {
        let workspace_start = cursor;
        for requirement in requirements.arena_slot_requirements(slots).unwrap() {
            if requirement.id == SHARED_STAGED_PRIMARY {
                continue;
            }
            push_spec(
                &mut specs,
                &mut cursor,
                requirement.id,
                requirement.len_words,
                requirement.alignment_words,
            );
        }
        assert_eq!(
            (cursor - workspace_start) as u64 * 4,
            STAGED_DESCRIPTOR_WORKSPACE_BYTES
        );
    }
    push_spec(
        &mut specs,
        &mut cursor,
        SHARED_STAGED_PRIMARY,
        STAGED_PRIMARY_WORDS,
        8,
    );
    push_spec(
        &mut specs,
        &mut cursor,
        SHARED_STAGED_OVERFLOW,
        STAGED_OVERFLOW_WORDS,
        8,
    );
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
    let quotient_start = cursor;
    for requirement in quotient_requirements
        .arena_slot_requirements(&quotient_slots)
        .unwrap()
    {
        if requirement.id == SAMPLE_POINTS_OUTPUT || requirement.id == FIRST_TERMS_OUTPUT {
            continue;
        }
        push_spec(
            &mut specs,
            &mut cursor,
            requirement.id,
            requirement.len_words,
            requirement.alignment_words,
        );
    }
    push_spec(
        &mut specs,
        &mut cursor,
        INVERSE_TWIDDLES,
        quotient_requirements.inverse_twiddle_words,
        1,
    );
    assert_eq!(cursor - quotient_start, QUOTIENT_INCREMENTAL_ARENA_WORDS);
    let allocation_bytes = (cursor as u64).checked_mul(4).unwrap();
    assert!(allocation_bytes <= MAX_BENCHMARK_ARENA_BYTES);

    assert_workspace_capacity(retained_requirements, &retained_slots, &specs);
    let primary = slot_spec(&specs, SHARED_STAGED_PRIMARY);
    let overflow = slot_spec(&specs, SHARED_STAGED_OVERFLOW);
    let mut ranges = specs
        .iter()
        .copied()
        .map(|slot| ArenaRangeSpec {
            live_mask: if matches!(slot.id, SHARED_STAGED_PRIMARY | SHARED_STAGED_OVERFLOW) {
                CONTROL_LIVE
            } else {
                BOTH_LIVE
            },
            slot,
        })
        .collect::<Vec<_>>();

    let mut retained_source_ids = source_ids.clone();
    let mut retained_primary_columns = 0usize;
    let mut retained_overflow_columns = 0usize;
    let mut retained_image_words = 0usize;
    for lde in staged.coefficient_ldes() {
        let role_base = match lde.staging_role() {
            QuotientNumeratorStagingRole::Primary => {
                retained_primary_columns += 1;
                primary.offset_words
            }
            QuotientNumeratorStagingRole::Overflow(0) => {
                retained_overflow_columns += 1;
                overflow.offset_words
            }
            role => panic!("retained FixedImage column uses unexpected staging role {role:?}"),
        };
        retained_image_words = retained_image_words.checked_add(lde.len_words()).unwrap();
        let id = retained_image_id(lde.column());
        retained_source_ids[lde.column()] = id;
        ranges.push(ArenaRangeSpec {
            slot: ArenaSlotSpec {
                id,
                offset_words: role_base + lde.role_offset_words(),
                len_words: lde.len_words(),
                alignment_words: 8,
            },
            live_mask: RETAINED_LIVE,
        });
    }
    assert_eq!(retained_primary_columns, 125);
    assert_eq!(retained_overflow_columns, 27);
    assert_eq!(retained_image_words, RETAINED_IMAGE_WORDS);
    assert_eq!(
        staged
            .coefficient_ldes()
            .iter()
            .map(|lde| retained_source_ids[lde.column()])
            .collect::<Vec<_>>(),
        staged
            .coefficient_ldes()
            .iter()
            .map(|lde| retained_image_id(lde.column()))
            .collect::<Vec<_>>()
    );
    assert!(staged
        .coefficient_ldes()
        .iter()
        .any(|lde| lde.column() == MUTATED_RETAINED_COLUMN));
    let layout = unsafe {
        // SAFETY: control and retained graphs run on one explicit stream and
        // every transition is synchronized. CONTROL_LIVE owns coefficient
        // staging; RETAINED_LIVE owns disjoint aliases of those same bytes.
        ArenaLayout::new_reused(cursor, &ranges)
    }
    .unwrap();
    let retained_manifest_blake3 = retained_manifest_digest(&layout, staged);
    BenchmarkArenaPlan {
        layout,
        oracle_slots,
        control_slots,
        retained_slots,
        quotient_slots,
        source_ids,
        retained_source_ids,
        destination_ids,
        retained_manifest_blake3,
        allocation_bytes,
    }
}

fn workspace_slots(
    requirements: &QuotientNumeratorWorkspaceRequirements,
    base: u32,
    lde_tile: ArenaSlotId,
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
        lde_tile: (requirements.lde_tile_words != 0).then_some(lde_tile),
    }
}

fn quotient_workspace_slots() -> QuotientWorkspaceSlots {
    let id = |offset| ArenaSlotId(QUOTIENT_WORKSPACE_BASE + offset);
    QuotientWorkspaceSlots {
        sample_points: SAMPLE_POINTS_OUTPUT,
        first_linear_terms: FIRST_TERMS_OUTPUT,
        partial_log_sizes: id(0),
        partial_coordinate_ptrs: id(1),
        subdomain_coordinate_ptrs: id(2),
        output_coordinate_ptrs: id(3),
        coefficient_sizes: id(4),
        subdomain_values: id(5),
        output_values: id(6),
    }
}

fn assert_descriptor_workspaces_are_disjoint(
    requirements: &QuotientNumeratorWorkspaceRequirements,
    packed: &QuotientNumeratorWorkspaceSlots,
    direct: &QuotientNumeratorWorkspaceSlots,
) {
    assert_eq!(packed.lde_tile, Some(SHARED_STAGED_PRIMARY));
    assert_eq!(direct.lde_tile, Some(SHARED_STAGED_PRIMARY));
    let packed = requirements.arena_slot_requirements(packed).unwrap();
    let direct = requirements.arena_slot_requirements(direct).unwrap();
    assert!(packed
        .iter()
        .filter(|left| left.id != SHARED_STAGED_PRIMARY)
        .all(|left| direct
            .iter()
            .filter(|right| right.id != SHARED_STAGED_PRIMARY)
            .all(|right| left.id != right.id)));
}

fn assert_workspace_capacity(
    retained_requirements: &QuotientNumeratorWorkspaceRequirements,
    retained_slots: &QuotientNumeratorWorkspaceSlots,
    specs: &[ArenaSlotSpec],
) {
    for requirement in retained_requirements
        .arena_slot_requirements(retained_slots)
        .unwrap()
    {
        let allocated = slot_spec(specs, requirement.id);
        assert!(
            allocated.len_words >= requirement.len_words,
            "retained descriptor {:?} needs {} words but the reused oracle slot has {}",
            requirement.id,
            requirement.len_words,
            allocated.len_words
        );
        assert_eq!(
            allocated.offset_words % requirement.alignment_words,
            0,
            "retained descriptor {:?} alignment drift",
            requirement.id
        );
    }
}

fn assert_reused_workspace_shape(
    plan: &BenchmarkArenaPlan,
    staged: &QuotientNumeratorStagedSingleWritePlan,
) {
    assert_eq!(plan.oracle_slots.lde_tile, Some(SHARED_STAGED_PRIMARY));
    assert_eq!(plan.control_slots.lde_tile, Some(SHARED_STAGED_PRIMARY));
    assert_eq!(plan.retained_slots.lde_tile, None);
    assert_eq!(plan.retained_slots.coefficient_ptrs, None);
    assert_eq!(plan.retained_slots.coefficient_sizes, None);
    assert_eq!(plan.retained_slots.coefficient_output_ptrs, None);
    assert_eq!(
        [
            plan.retained_slots.runtime_terms,
            plan.retained_slots.group_term_indices,
            plan.retained_slots.group_offsets,
            plan.retained_slots.line_coefficients,
            plan.retained_slots.term_points,
            plan.retained_slots.batch_terms,
            plan.retained_slots.batch_group_offsets,
            plan.retained_slots.batch_source_ptrs,
            plan.retained_slots.output_ptrs,
            plan.retained_slots.output_log_sizes,
        ],
        [
            plan.oracle_slots.runtime_terms,
            plan.oracle_slots.group_term_indices,
            plan.oracle_slots.group_offsets,
            plan.oracle_slots.line_coefficients,
            plan.oracle_slots.term_points,
            plan.oracle_slots.batch_terms,
            plan.oracle_slots.batch_group_offsets,
            plan.oracle_slots.batch_source_ptrs,
            plan.oracle_slots.output_ptrs,
            plan.oracle_slots.output_log_sizes,
        ]
    );

    let primary = plan.layout.slot(SHARED_STAGED_PRIMARY).unwrap();
    let overflow = plan.layout.slot(SHARED_STAGED_OVERFLOW).unwrap();
    assert_eq!(primary.len_words, STAGED_PRIMARY_WORDS);
    assert_eq!(overflow.len_words, STAGED_OVERFLOW_WORDS);
    let mut words = 0usize;
    for lde in staged.coefficient_ldes() {
        let slot = plan.layout.slot(retained_image_id(lde.column())).unwrap();
        let role_base = match lde.staging_role() {
            QuotientNumeratorStagingRole::Primary => primary.offset_words,
            QuotientNumeratorStagingRole::Overflow(0) => overflow.offset_words,
            role => panic!("retained FixedImage column uses unexpected staging role {role:?}"),
        };
        assert_eq!(slot.offset_words, role_base + lde.role_offset_words());
        assert_eq!(slot.len_words, lde.len_words());
        words = words.checked_add(slot.len_words).unwrap();
    }
    assert_eq!(words, RETAINED_IMAGE_WORDS);
    assert_eq!(
        RETAINED_IMAGE_WORDS as u64 * 4,
        3_919_863_424,
        "all 152 sampled coefficient LDE aliases must retain the exact control image"
    );
}

fn slot_spec(specs: &[ArenaSlotSpec], id: ArenaSlotId) -> ArenaSlotSpec {
    specs
        .iter()
        .copied()
        .find(|spec| spec.id == id)
        .unwrap_or_else(|| panic!("missing arena slot {id:?}"))
}

fn retained_image_id(column: usize) -> ArenaSlotId {
    ArenaSlotId(
        RETAINED_IMAGE_BASE
            .checked_add(u32::try_from(column).unwrap())
            .unwrap(),
    )
}

fn retained_manifest_digest(
    layout: &ArenaLayout,
    staged: &QuotientNumeratorStagedSingleWritePlan,
) -> Hash {
    let mut hasher = Hasher::new();
    hasher.update(b"stwo.sn3-fixed-image-arena-manifest.v1\0");
    for lde in staged.coefficient_ldes() {
        let id = retained_image_id(lde.column());
        let slot = layout.slot(id).unwrap();
        for value in [
            u64::try_from(lde.column()).unwrap(),
            u64::from(lde.evaluation_log_size()),
            u64::from(id.0),
            u64::try_from(slot.offset_words).unwrap(),
            u64::try_from(slot.len_words).unwrap(),
            u64::try_from(slot.alignment_words).unwrap(),
        ] {
            hasher.update(&value.to_le_bytes());
        }
    }
    hasher.finalize()
}

fn assert_quotient_sources_same(
    retained: &[QuotientNumeratorSource],
    control: &[QuotientNumeratorSource],
) {
    assert_eq!(retained.len(), control.len());
    for (index, (retained, control)) in retained.iter().zip(control).enumerate() {
        assert_eq!(retained.log_size, control.log_size, "source {index} log");
        assert_eq!(
            retained.constants.sample_point, control.constants.sample_point,
            "source {index} sample point"
        );
        assert_eq!(
            retained.constants.first_linear_term_acc, control.constants.first_linear_term_acc,
            "source {index} first linear term"
        );
        for coordinate in 0..4 {
            assert_eq!(
                retained.coordinates[coordinate].id(),
                control.coordinates[coordinate].id(),
                "source {index} coordinate {coordinate} id"
            );
            assert_eq!(
                retained.coordinates[coordinate].len_words(),
                control.coordinates[coordinate].len_words(),
                "source {index} coordinate {coordinate} length"
            );
        }
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
    config: QuotientNumeratorWorkspaceConfig,
    group_direct: bool,
    overflow_role_ids: &[ArenaSlotId],
) -> PreparedQuotientNumeratorGraph<'a> {
    let arguments = (
        fixture.arena.bind(OODS_POINTS).unwrap(),
        fixture.arena.bind(OODS_VALUES).unwrap(),
        fixture.arena.bind(RANDOM_COEFFICIENT).unwrap(),
        fixture.arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        fixture.arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        fixture.arena.bind(TWIDDLES).unwrap(),
    );
    let overflow_roles = overflow_role_ids
        .iter()
        .map(|&id| fixture.arena.bind(id).unwrap())
        .collect::<Vec<_>>();
    if group_direct {
        PreparedQuotientNumeratorGraph::prepare_staged_group_direct(
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
            &overflow_roles,
        )
        .unwrap()
    } else {
        PreparedQuotientNumeratorGraph::prepare_staged_packed_single_write(
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
            &overflow_roles,
        )
        .unwrap()
    }
}

fn replay_cuda_ms(graph: &CudaGraphExec, context: &CudaExecContext) -> f64 {
    assert!(context.begin_timing().unwrap() >= 1);
    graph.launch(context).unwrap();
    context.mark_timing().unwrap();
    context.sync().unwrap();
    f64::from(context.elapsed_timing_ms(1).unwrap()[0])
}

fn boundary_input_recipe_digest(
    numerator_recipe: Hash,
    requirements: &QuotientWorkspaceRequirements,
) -> Hash {
    let mut hasher = Hasher::new();
    hasher.update(b"stwo.sn3-numerator-to-fri-input.input-recipe.v2\0");
    hasher.update(numerator_recipe.as_bytes());
    hasher.update(&u64::try_from(GROUP_LOGS.len()).unwrap().to_le_bytes());
    for log_size in GROUP_LOGS {
        hasher.update(&u64::from(log_size).to_le_bytes());
    }
    let pass = requirements.combine_pass_bytes;
    let fields = [
        u64::from(requirements.config.lifting_log_size),
        u64::from(requirements.config.log_blowup_factor),
        u64::from(requirements.subdomain_log_size),
        u64::try_from(requirements.sample_count).unwrap(),
        u64::try_from(requirements.sample_point_words).unwrap(),
        u64::try_from(requirements.first_linear_term_words).unwrap(),
        u64::try_from(requirements.partial_log_size_words).unwrap(),
        u64::try_from(requirements.partial_pointer_words).unwrap(),
        u64::try_from(requirements.coordinate_pointer_words).unwrap(),
        u64::try_from(requirements.coefficient_size_words).unwrap(),
        u64::try_from(requirements.subdomain_value_words).unwrap(),
        u64::try_from(requirements.output_value_words).unwrap(),
        u64::try_from(pass.rows).unwrap(),
        u64::try_from(pass.samples).unwrap(),
        u64::try_from(pass.denominator_inversions).unwrap(),
        u64::try_from(pass.eliminated_scratch_bytes).unwrap(),
        u64::try_from(pass.eliminated_logical_traffic_bytes).unwrap(),
        u64::try_from(pass.denominator_global_passes).unwrap(),
        u64::try_from(pass.output_write_bytes).unwrap(),
        u64::try_from(requirements.forward_twiddle_words).unwrap(),
        u64::try_from(requirements.inverse_twiddle_words).unwrap(),
        u64::from(requirements.half_coset_initial_index),
        u64::from(requirements.half_coset_step_size),
        INVERSE_TWIDDLE_PATTERN_SEED,
    ];
    hasher.update(&u64::try_from(fields.len()).unwrap().to_le_bytes());
    for value in fields {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize()
}

struct CanonicalFriInput {
    words: Vec<u32>,
    digest: Hash,
}

impl CanonicalFriInput {
    fn len_bytes(&self) -> u64 {
        u64::try_from(self.words.len()).unwrap() * 4
    }
}

fn capture_fri_input(
    fixture: &BenchmarkArena,
    quotient: &PreparedQuotientGraph<'_>,
) -> CanonicalFriInput {
    let source = quotient.output_evaluation();
    let mut words = Vec::with_capacity(source.len_words());
    let mut hasher = fri_input_hasher(source.len_words());
    for offset in (0..source.len_words()).step_by(FRI_COPY_CHUNK_WORDS) {
        let chunk = read_fri_input_chunk(&fixture.arena, source, offset);
        hash_fri_input_words(&mut hasher, &chunk);
        words.extend_from_slice(&chunk);
    }
    assert_eq!(words.len(), source.len_words());
    CanonicalFriInput {
        words,
        digest: hasher.finalize(),
    }
}

fn assert_fri_input(
    fixture: &BenchmarkArena,
    quotient: &PreparedQuotientGraph<'_>,
    expected: &CanonicalFriInput,
    label: &str,
) -> Hash {
    let source = quotient.output_evaluation();
    assert_eq!(
        source.len_words(),
        expected.words.len(),
        "{label}: FRI-input shape drift"
    );
    let mut hasher = fri_input_hasher(source.len_words());
    for offset in (0..source.len_words()).step_by(FRI_COPY_CHUNK_WORDS) {
        let actual = read_fri_input_chunk(&fixture.arena, source, offset);
        let expected_chunk = &expected.words[offset..offset + actual.len()];
        if let Some(relative_index) = actual
            .iter()
            .zip(expected_chunk)
            .position(|(actual, expected)| actual != expected)
        {
            let index = offset + relative_index;
            panic!(
                "{label}: FRI input differs at word {index}: expected {}, got {}",
                expected.words[index], actual[relative_index]
            );
        }
        hash_fri_input_words(&mut hasher, &actual);
    }
    let digest = hasher.finalize();
    assert_eq!(digest, expected.digest, "{label}: FRI-input digest drift");
    digest
}

fn poison_fri_input(fixture: &BenchmarkArena, quotient: &PreparedQuotientGraph<'_>, poison: u32) {
    let output = quotient.output_evaluation();
    unsafe {
        fixture
            .arena
            .context()
            .fill_u32_async(output.as_u32_ptr(), poison, output.len_words())
            .unwrap();
    }
    fixture.arena.context().sync().unwrap();
}

fn poison_retained_image(fixture: &BenchmarkArena, column: usize, poison: u32) {
    let image = fixture.arena.bind(retained_image_id(column)).unwrap();
    unsafe {
        fixture
            .arena
            .context()
            .fill_u32_async(image.as_u32_ptr(), poison, image.len_words())
            .unwrap();
    }
    fixture.arena.context().sync().unwrap();
}

fn read_fri_input_chunk(arena: &DeviceArena, source: ArenaSlice, offset: usize) -> Vec<u32> {
    let count = FRI_COPY_CHUNK_WORDS.min(source.len_words() - offset);
    let chunk = source.checked_subslice(offset, count).unwrap();
    let mut words = vec![0u32; count];
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                words.as_mut_ptr().cast(),
                chunk.as_void_ptr(),
                std::mem::size_of_val(&*words),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
    words
}

fn fri_input_hasher(word_count: usize) -> Hasher {
    let mut hasher = Hasher::new();
    hasher.update(b"stwo.sn3-quotient-fri-input.output.v1\0");
    hasher.update(&u64::try_from(word_count).unwrap().to_le_bytes());
    hasher
}

#[cfg(target_endian = "little")]
fn hash_fri_input_words(hasher: &mut Hasher, words: &[u32]) {
    // SAFETY: the byte view covers initialized `u32` objects and preserves
    // the required little-endian word framing on supported CUDA hosts.
    let bytes = unsafe {
        std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words))
    };
    hasher.update(bytes);
}

#[cfg(target_endian = "big")]
fn hash_fri_input_words(hasher: &mut Hasher, words: &[u32]) {
    for word in words {
        hasher.update(&word.to_le_bytes());
    }
}

fn initialize(
    fixture: &BenchmarkArena,
    topology: &[QuotientNumeratorColumnTopology],
    requirements: &QuotientNumeratorWorkspaceRequirements,
    quotient_requirements: &QuotientWorkspaceRequirements,
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
    upload_affine_pattern(
        &fixture.arena,
        INVERSE_TWIDDLES,
        quotient_requirements.inverse_twiddle_words,
        INVERSE_TWIDDLE_PATTERN_SEED,
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
