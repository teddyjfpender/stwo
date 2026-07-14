//! Ignored cheap-GPU benchmark for the sealed SN3 quotient-numerator shape.
//! gpu-lab-cohesion-review: topology construction, byte identity, graph replay timing, and the
//! single JSON result stay together so a benchmark cannot silently drift from its comparator.
//! Run receipts must place this JSON beside `nvidia-smi` identity/driver/clock output,
//! `nvcc --version`, `rustc -Vv`, and `git rev-parse HEAD`; those are environment facts, not test
//! semantics, so the existing pod wrapper owns them.

#![cfg(stwo_cuda_link)]

use std::time::Instant;

use stwo::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
use stwo::core::fields::qm31::SecureField;
use stwo_backend_cuda::{
    quotient_numerator_hybrid_plan, quotient_numerator_workspace_requirements, ArenaLayout,
    ArenaSlice, ArenaSlotId, ArenaSlotSpec, CudaExecContext, CudaGraphExec, DeviceArena,
    PreparedQuotientNumeratorGraph, QuotientNumeratorColumn, QuotientNumeratorColumnSource,
    QuotientNumeratorColumnTopology, QuotientNumeratorDestination, QuotientNumeratorSourceKind,
    QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceRequirements,
    QuotientNumeratorWorkspaceSlots, QuotientOodsSample,
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
const COPY_CHUNK_WORDS: usize = 1 << 20;

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
        0xdead_beef,
    );
    initialize(
        &hybrid_fixture,
        &topology,
        &requirements,
        &points,
        0xa5a5_5a5a,
    );
    legacy.launch().unwrap();
    hybrid.launch().unwrap();
    legacy_fixture.arena.context().sync().unwrap();
    hybrid_fixture.arena.context().sync().unwrap();
    let validated_output_bytes =
        assert_full_output_equality(&legacy_fixture, &hybrid_fixture, &requirements);

    let capture = legacy_fixture.arena.context().capture().unwrap();
    legacy.launch().unwrap();
    let legacy_graph = capture.finish().unwrap();
    let capture = hybrid_fixture.arena.context().capture().unwrap();
    hybrid.launch().unwrap();
    let hybrid_graph = capture.finish().unwrap();

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
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_ITERATIONS);
    let mut legacy_ms = Vec::with_capacity(iterations);
    let mut hybrid_ms = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        if iteration % 2 == 0 {
            legacy_ms.push(replay_ms(&legacy_graph, legacy_fixture.arena.context()));
            hybrid_ms.push(replay_ms(&hybrid_graph, hybrid_fixture.arena.context()));
        } else {
            hybrid_ms.push(replay_ms(&hybrid_graph, hybrid_fixture.arena.context()));
            legacy_ms.push(replay_ms(&legacy_graph, legacy_fixture.arena.context()));
        }
    }

    let legacy_p50 = percentile(&legacy_ms, 50);
    let legacy_p95 = percentile(&legacy_ms, 95);
    let hybrid_p50 = percentile(&hybrid_ms, 50);
    let hybrid_p95 = percentile(&hybrid_ms, 95);
    println!(
        concat!(
            "{{\"schema\":\"stwo.sn3_quotient_numerator_hybrid.host_wall.v1\"," ,
            "\"timing_scope\":\"per-replay host wall: graph launch plus stream synchronize; not CUDA events\"," ,
            "\"percentile_method\":\"nearest-rank\"," ,
            "\"topology\":{{\"group_logs\":{:?},\"groups\":19,\"eligible_groups\":18," ,
            "\"legacy_groups\":1,\"coefficient_sources\":152,\"coefficient_batches\":74," ,
            "\"terms\":6341}},\"bytes\":{{\"legacy_logical_output\":{}," ,
            "\"hybrid_logical_output\":{},\"validated_numerator_output\":{}," ,
            "\"legacy_arena\":{},\"hybrid_arena\":{},\"combined_arenas\":{}}}," ,
            "\"warmups\":{},\"iterations\":{},\"samples_ms\":{{\"legacy\":{},\"hybrid\":{}}}," ,
            "\"host_wall_ms\":{{\"legacy\":{{\"p50\":{:.6},\"p95\":{:.6}}}," ,
            "\"hybrid\":{{\"p50\":{:.6},\"p95\":{:.6}}}}}," ,
            "\"speedup\":{{\"p50\":{:.9},\"p95\":{:.9}}}}}"
        ),
        GROUP_LOGS,
        LEGACY_LOGICAL_OUTPUT_BYTES,
        HYBRID_LOGICAL_OUTPUT_BYTES,
        validated_output_bytes,
        legacy_fixture.allocation_bytes,
        hybrid_fixture.allocation_bytes,
        combined_arena_bytes,
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
    let values = (0..points.len())
        .map(|index| {
            let value = index as u32 * 16 + 1;
            SecureField::from_u32_unchecked(value, value + 2, value + 4, value + 6)
        })
        .collect::<Vec<_>>();
    upload(&fixture.arena, OODS_POINTS, &point_words(points));
    upload(&fixture.arena, OODS_VALUES, &secure_words(&values));
    upload(
        &fixture.arena,
        RANDOM_COEFFICIENT,
        &secure_words(&[SecureField::from_u32_unchecked(307, 311, 313, 317)]),
    );
    unsafe {
        fixture
            .arena
            .context()
            .fill_u32_async(
                fixture.arena.bind(TWIDDLES).unwrap().as_u32_ptr(),
                1,
                requirements.forward_twiddle_words,
            )
            .unwrap();
    }
    for (index, (&id, column)) in fixture.source_ids.iter().zip(topology).enumerate() {
        let value = ((index as u32 + 1) * 1_000_003) & 0x7fff_ffff;
        let slice = fixture.arena.bind(id).unwrap();
        unsafe {
            fixture
                .arena
                .context()
                .fill_u32_async(slice.as_u32_ptr(), value, source_words(column))
                .unwrap();
        }
    }
    for id in [SAMPLE_POINTS_OUTPUT, FIRST_TERMS_OUTPUT] {
        let slice = fixture.arena.bind(id).unwrap();
        unsafe {
            fixture
                .arena
                .context()
                .fill_u32_async(slice.as_u32_ptr(), poison, slice.len_words())
                .unwrap();
        }
    }
    for ids in &fixture.destination_ids {
        for &id in ids {
            let slice = fixture.arena.bind(id).unwrap();
            unsafe {
                fixture
                    .arena
                    .context()
                    .fill_u32_async(slice.as_u32_ptr(), poison, slice.len_words())
                    .unwrap();
            }
        }
    }
    fixture.arena.context().sync().unwrap();
}

fn assert_full_output_equality(
    legacy: &BenchmarkArena,
    hybrid: &BenchmarkArena,
    requirements: &QuotientNumeratorWorkspaceRequirements,
) -> u64 {
    assert_range_equal(
        &legacy.arena,
        legacy.arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        &hybrid.arena,
        hybrid.arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        requirements.groups.len() * 8,
        "sample points",
    );
    assert_range_equal(
        &legacy.arena,
        legacy.arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        &hybrid.arena,
        hybrid.arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        requirements.groups.len() * 4,
        "first terms",
    );
    let mut output_words = 0usize;
    for (group, requirement) in requirements.groups.iter().enumerate() {
        for coordinate in 0..4 {
            assert_range_equal(
                &legacy.arena,
                legacy
                    .arena
                    .bind(legacy.destination_ids[group][coordinate])
                    .unwrap(),
                &hybrid.arena,
                hybrid
                    .arena
                    .bind(hybrid.destination_ids[group][coordinate])
                    .unwrap(),
                requirement.value_words,
                &format!("group {group} coordinate {coordinate}"),
            );
            output_words = output_words.checked_add(requirement.value_words).unwrap();
        }
    }
    (output_words as u64).checked_mul(4).unwrap()
}

fn assert_range_equal(
    left_arena: &DeviceArena,
    left: ArenaSlice,
    right_arena: &DeviceArena,
    right: ArenaSlice,
    words: usize,
    label: &str,
) {
    for offset in (0..words).step_by(COPY_CHUNK_WORDS) {
        let count = COPY_CHUNK_WORDS.min(words - offset);
        let mut left_host = vec![0u32; count];
        let mut right_host = vec![0u32; count];
        unsafe {
            left_arena
                .context()
                .memcpy_d2h_async(
                    left_host.as_mut_ptr().cast(),
                    left.as_u32_ptr().add(offset).cast(),
                    count * 4,
                )
                .unwrap();
            right_arena
                .context()
                .memcpy_d2h_async(
                    right_host.as_mut_ptr().cast(),
                    right.as_u32_ptr().add(offset).cast(),
                    count * 4,
                )
                .unwrap();
        }
        left_arena.context().sync().unwrap();
        right_arena.context().sync().unwrap();
        assert_eq!(left_host, right_host, "{label} differs at word {offset}");
    }
}

fn replay_ms(graph: &CudaGraphExec, context: &CudaExecContext) -> f64 {
    let started = Instant::now();
    graph.launch(context).unwrap();
    context.sync().unwrap();
    started.elapsed().as_secs_f64() * 1_000.0
}

fn percentile(samples: &[f64], percentage: usize) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentage * sorted.len()).div_ceil(100).saturating_sub(1);
    sorted[rank]
}

fn json_samples(samples: &[f64]) -> String {
    format!(
        "[{}]",
        samples
            .iter()
            .map(|sample| format!("{sample:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    )
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
