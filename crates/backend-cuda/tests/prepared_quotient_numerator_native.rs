//! Native CUDA correctness and capture gate for prepared quotient numerators.

#![cfg(stwo_cuda_link)]

use stwo::core::constraints::complex_conjugate_line_coeffs;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::quotients::PointSample;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo_backend_cuda::{
    quotient_numerator_workspace_requirements, ArenaLayout, ArenaSlice, ArenaSlotId, ArenaSlotSpec,
    CudaExecContext, DeviceArena, PreparedQuotientNumeratorGraph, QuotientNumeratorColumn,
    QuotientNumeratorColumnSource, QuotientNumeratorColumnTopology, QuotientNumeratorDestination,
    QuotientNumeratorSourceKind, QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceSlots,
    QuotientOodsSample,
};

const OODS_POINTS: ArenaSlotId = ArenaSlotId(50_000);
const OODS_VALUES: ArenaSlotId = ArenaSlotId(50_001);
const RANDOM_COEFFICIENT: ArenaSlotId = ArenaSlotId(50_002);
const SAMPLE_POINTS_OUTPUT: ArenaSlotId = ArenaSlotId(50_003);
const FIRST_TERMS_OUTPUT: ArenaSlotId = ArenaSlotId(50_004);
const TWIDDLES: ArenaSlotId = ArenaSlotId(50_005);
const EVAL_A: ArenaSlotId = ArenaSlotId(50_006);
const EVAL_B: ArenaSlotId = ArenaSlotId(50_007);
const COEFF_A: ArenaSlotId = ArenaSlotId(50_008);
const COEFF_B: ArenaSlotId = ArenaSlotId(50_009);
const OUTPUT_BASE: u32 = 60_000;
const SLOT_WORDS: usize = 4096;

fn slots() -> QuotientNumeratorWorkspaceSlots {
    let mut next = 1u32;
    let mut id = || {
        let result = ArenaSlotId(next);
        next += 1;
        result
    };
    QuotientNumeratorWorkspaceSlots {
        runtime_terms: id(),
        group_term_indices: id(),
        group_offsets: id(),
        line_coefficients: id(),
        term_points: id(),
        batch_terms: id(),
        batch_group_offsets: id(),
        batch_source_ptrs: id(),
        output_ptrs: id(),
        output_log_sizes: id(),
        coefficient_ptrs: None,
        coefficient_sizes: None,
        coefficient_output_ptrs: None,
        lde_tile: None,
    }
}

fn slots_for_requirements(
    requirements: &stwo_backend_cuda::QuotientNumeratorWorkspaceRequirements,
) -> QuotientNumeratorWorkspaceSlots {
    let mut result = slots();
    result.coefficient_ptrs =
        (requirements.coefficient_pointer_words != 0).then_some(ArenaSlotId(100));
    result.coefficient_sizes =
        (requirements.coefficient_size_words != 0).then_some(ArenaSlotId(101));
    result.coefficient_output_ptrs =
        (requirements.coefficient_output_pointer_words != 0).then_some(ArenaSlotId(102));
    result.lde_tile = (requirements.lde_tile_words != 0).then_some(ArenaSlotId(103));
    result
}

#[test]
fn mixed_evaluation_and_coefficients_keep_the_same_numerator_semantics() {
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
    let config = QuotientNumeratorWorkspaceConfig {
        lifting_log_size: 6,
        log_blowup_factor: 2,
        max_lde_tile_words: 256,
    };
    let point = SECURE_FIELD_CIRCLE_GEN.mul(3);
    let coefficient_topology = vec![
        QuotientNumeratorColumnTopology {
            coefficient_log_size: 4,
            source_kind: QuotientNumeratorSourceKind::Coefficients,
            samples: vec![QuotientOodsSample {
                input_index: 0,
                shape_point: point,
            }],
        },
        QuotientNumeratorColumnTopology {
            coefficient_log_size: 3,
            source_kind: QuotientNumeratorSourceKind::Coefficients,
            samples: vec![QuotientOodsSample {
                input_index: 1,
                shape_point: point,
            }],
        },
    ];
    let mut mixed_topology = coefficient_topology.clone();
    mixed_topology[0].source_kind = QuotientNumeratorSourceKind::Evaluation;
    let coefficient_requirements =
        quotient_numerator_workspace_requirements(config, &coefficient_topology).unwrap();
    let mixed_requirements =
        quotient_numerator_workspace_requirements(config, &mixed_topology).unwrap();
    assert_eq!(coefficient_requirements.groups, mixed_requirements.groups);
    assert_eq!(
        coefficient_requirements
            .batches
            .iter()
            .map(|batch| batch.coefficient_count)
            .sum::<usize>(),
        2
    );
    assert_eq!(
        mixed_requirements
            .batches
            .iter()
            .map(|batch| batch.coefficient_count)
            .sum::<usize>(),
        1
    );
    assert!(mixed_requirements.lde_tile_words < coefficient_requirements.lde_tile_words);

    let coefficient_slots = slots_for_requirements(&coefficient_requirements);
    let mixed_slots = slots_for_requirements(&mixed_requirements);
    let coefficient_arena = arena(&coefficient_slots);
    let mixed_arena = arena(&mixed_slots);
    let full_domain = CanonicCoset::new(config.lifting_log_size).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(full_domain.half_coset);
    let twiddle_words = twiddles
        .twiddles
        .iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let coefficient_a = (0..1usize << 4)
        .map(|index| (104_729 + 7_919 * index as u32) % 0x7fff_ffff)
        .collect::<Vec<_>>();
    let coefficient_b = (0..1usize << 3)
        .map(|index| (130_363 + 5_003 * index as u32) % 0x7fff_ffff)
        .collect::<Vec<_>>();
    let canonical_lde = |words: &[u32], log_size: u32| {
        CircleCoefficients::<CpuBackend>::new(
            words
                .iter()
                .copied()
                .map(BaseField::from_u32_unchecked)
                .collect(),
        )
        .evaluate_with_twiddles(
            CanonicCoset::new(log_size + config.log_blowup_factor).circle_domain(),
            &twiddles,
        )
        .values
        .iter()
        .map(|value| value.0)
        .collect::<Vec<_>>()
    };
    let evaluation_a = canonical_lde(&coefficient_a, 4);
    let evaluation_b = canonical_lde(&coefficient_b, 3);
    let sampled_values = [
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
    ];
    for arena in [&coefficient_arena, &mixed_arena] {
        upload(arena, OODS_POINTS, &point_words(&[point, point]));
        upload(arena, OODS_VALUES, &secure_words(&sampled_values));
        upload(arena, TWIDDLES, &twiddle_words);
        upload(arena, COEFF_A, &coefficient_a);
        upload(arena, COEFF_B, &coefficient_b);
        upload(arena, EVAL_A, &evaluation_a);
        upload(arena, EVAL_B, &evaluation_b);
        arena.context().sync().unwrap();
    }
    let coefficient_columns = vec![
        QuotientNumeratorColumn {
            coefficient_log_size: 4,
            source: QuotientNumeratorColumnSource::Coefficients(
                coefficient_arena.bind(COEFF_A).unwrap(),
            ),
            samples: coefficient_topology[0].samples.clone(),
        },
        QuotientNumeratorColumn {
            coefficient_log_size: 3,
            source: QuotientNumeratorColumnSource::Coefficients(
                coefficient_arena.bind(COEFF_B).unwrap(),
            ),
            samples: coefficient_topology[1].samples.clone(),
        },
    ];
    let mixed_columns = vec![
        QuotientNumeratorColumn {
            coefficient_log_size: 4,
            source: QuotientNumeratorColumnSource::Evaluation(mixed_arena.bind(EVAL_A).unwrap()),
            samples: mixed_topology[0].samples.clone(),
        },
        QuotientNumeratorColumn {
            coefficient_log_size: 3,
            source: QuotientNumeratorColumnSource::Coefficients(mixed_arena.bind(COEFF_B).unwrap()),
            samples: mixed_topology[1].samples.clone(),
        },
    ];
    let destinations = |arena: &DeviceArena| {
        coefficient_requirements
            .groups
            .iter()
            .enumerate()
            .map(|(group, requirement)| QuotientNumeratorDestination {
                log_size: requirement.log_size,
                coordinates: std::array::from_fn(|coordinate| {
                    arena.bind(output_id(group, coordinate)).unwrap()
                }),
            })
            .collect::<Vec<_>>()
    };
    let coefficient_destinations = destinations(&coefficient_arena);
    let mixed_destinations = destinations(&mixed_arena);
    let coefficient_prepared = PreparedQuotientNumeratorGraph::prepare(
        &coefficient_arena,
        config,
        &coefficient_columns,
        coefficient_arena.bind(OODS_POINTS).unwrap(),
        coefficient_arena.bind(OODS_VALUES).unwrap(),
        coefficient_arena.bind(RANDOM_COEFFICIENT).unwrap(),
        coefficient_arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        coefficient_arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        &coefficient_destinations,
        coefficient_arena.bind(TWIDDLES).unwrap(),
        &coefficient_slots,
    )
    .unwrap();
    let mixed_prepared = PreparedQuotientNumeratorGraph::prepare(
        &mixed_arena,
        config,
        &mixed_columns,
        mixed_arena.bind(OODS_POINTS).unwrap(),
        mixed_arena.bind(OODS_VALUES).unwrap(),
        mixed_arena.bind(RANDOM_COEFFICIENT).unwrap(),
        mixed_arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        mixed_arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        &mixed_destinations,
        mixed_arena.bind(TWIDDLES).unwrap(),
        &mixed_slots,
    )
    .unwrap();
    let snapshot = |arena: &DeviceArena, destinations: &[QuotientNumeratorDestination]| {
        let mut result = vec![read_words(
            arena,
            arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
            8 * coefficient_requirements.groups.len(),
        )];
        result.push(read_words(
            arena,
            arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
            4 * coefficient_requirements.groups.len(),
        ));
        for destination in destinations {
            for coordinate in destination.coordinates {
                result.push(read_words(
                    arena,
                    coordinate,
                    1usize << destination.log_size,
                ));
            }
        }
        result
    };
    let immutable_before = read_words(
        &mixed_arena,
        mixed_arena.bind(EVAL_A).unwrap(),
        evaluation_a.len(),
    );
    let eager_alpha = SecureField::from_u32_unchecked(41, 43, 47, 53);
    upload(
        &coefficient_arena,
        RANDOM_COEFFICIENT,
        &secure_words(&[eager_alpha]),
    );
    upload(
        &mixed_arena,
        RANDOM_COEFFICIENT,
        &secure_words(&[eager_alpha]),
    );
    coefficient_prepared.launch().unwrap();
    mixed_prepared.launch().unwrap();
    coefficient_arena.context().sync().unwrap();
    mixed_arena.context().sync().unwrap();
    let eager_coefficient = snapshot(&coefficient_arena, &coefficient_destinations);
    let eager_mixed = snapshot(&mixed_arena, &mixed_destinations);
    assert_eq!(eager_coefficient, eager_mixed);

    let coefficient_capture = coefficient_arena.context().capture().unwrap();
    coefficient_prepared.launch().unwrap();
    let coefficient_graph = coefficient_capture.finish().unwrap();
    let mixed_capture = mixed_arena.context().capture().unwrap();
    mixed_prepared.launch().unwrap();
    let mixed_graph = mixed_capture.finish().unwrap();
    let replay_alpha = SecureField::from_u32_unchecked(59, 61, 67, 71);
    upload(
        &coefficient_arena,
        RANDOM_COEFFICIENT,
        &secure_words(&[replay_alpha]),
    );
    upload(
        &mixed_arena,
        RANDOM_COEFFICIENT,
        &secure_words(&[replay_alpha]),
    );
    coefficient_graph
        .launch(coefficient_arena.context())
        .unwrap();
    mixed_graph.launch(mixed_arena.context()).unwrap();
    coefficient_arena.context().sync().unwrap();
    mixed_arena.context().sync().unwrap();
    let replay_coefficient = snapshot(&coefficient_arena, &coefficient_destinations);
    let replay_mixed = snapshot(&mixed_arena, &mixed_destinations);
    assert_eq!(replay_coefficient, replay_mixed);
    assert_ne!(eager_coefficient, replay_coefficient);
    assert_eq!(
        read_words(
            &mixed_arena,
            mixed_arena.bind(EVAL_A).unwrap(),
            evaluation_a.len(),
        ),
        immutable_before
    );
}

fn output_id(group: usize, coordinate: usize) -> ArenaSlotId {
    ArenaSlotId(OUTPUT_BASE + (4 * group + coordinate) as u32)
}

fn arena(slots: &QuotientNumeratorWorkspaceSlots) -> DeviceArena {
    let mut ids = vec![
        slots.runtime_terms,
        slots.group_term_indices,
        slots.group_offsets,
        slots.line_coefficients,
        slots.term_points,
        slots.batch_terms,
        slots.batch_group_offsets,
        slots.batch_source_ptrs,
        slots.output_ptrs,
        slots.output_log_sizes,
        OODS_POINTS,
        OODS_VALUES,
        RANDOM_COEFFICIENT,
        SAMPLE_POINTS_OUTPUT,
        FIRST_TERMS_OUTPUT,
        TWIDDLES,
        EVAL_A,
        EVAL_B,
        COEFF_A,
        COEFF_B,
    ];
    ids.extend(
        [
            slots.coefficient_ptrs,
            slots.coefficient_sizes,
            slots.coefficient_output_ptrs,
            slots.lde_tile,
        ]
        .into_iter()
        .flatten(),
    );
    for group in 0..8 {
        for coordinate in 0..4 {
            ids.push(output_id(group, coordinate));
        }
    }
    let specs = ids
        .into_iter()
        .enumerate()
        .map(|(index, id)| ArenaSlotSpec {
            id,
            offset_words: index * SLOT_WORDS,
            // The kernels select twiddle subdomains relative to the logical
            // END, so pooled surplus here would test the wrong tower.
            len_words: if id == TWIDDLES { 1 << 5 } else { SLOT_WORDS },
            alignment_words: 8,
        })
        .collect::<Vec<_>>();
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(specs.len() * SLOT_WORDS, &specs).unwrap(),
    )
    .unwrap()
}

fn upload(arena: &DeviceArena, slot: ArenaSlotId, words: &[u32]) {
    let destination = arena.bind(slot).unwrap();
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(
                destination.as_void_ptr(),
                words.as_ptr().cast(),
                core::mem::size_of_val(words),
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

fn point_words(values: &[stwo::core::circle::CirclePoint<SecureField>]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|point| [point.x, point.y])
        .flat_map(|value| value.to_m31_array().map(|coordinate| coordinate.0))
        .collect()
}

fn read_words(arena: &DeviceArena, slice: ArenaSlice, words: usize) -> Vec<u32> {
    let mut host = vec![0u32; words];
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                host.as_mut_ptr().cast(),
                slice.as_void_ptr(),
                words * core::mem::size_of::<u32>(),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
    host
}

fn from_words(words: &[u32]) -> SecureField {
    SecureField::from_u32_unchecked(words[0], words[1], words[2], words[3])
}

fn expected_group(
    group_point: stwo::core::circle::CirclePoint<SecureField>,
    group_log: u32,
    alpha: SecureField,
    terms: &[(
        usize,
        u32,
        SecureField,
        stwo::core::circle::CirclePoint<SecureField>,
        &[u32],
    )],
) -> (SecureField, [Vec<u32>; 4]) {
    let matching = terms
        .iter()
        .filter(|(_, _, _, point, _)| *point == group_point)
        .collect::<Vec<_>>();
    let coefficients = matching
        .iter()
        .map(|(exponent, _, value, point, _)| {
            complex_conjugate_line_coeffs(
                &PointSample {
                    point: *point,
                    value: *value,
                },
                alpha.pow(*exponent as u128),
            )
        })
        .collect::<Vec<_>>();
    let first = coefficients.iter().map(|(a, ..)| *a).sum();
    let size = 1usize << group_log;
    let mut output: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0; size]);
    for row in 0..size {
        let mut numerator = SecureField::from(0u32);
        for ((_, source_log, _, _, source), (_, b, c)) in matching.iter().zip(&coefficients) {
            let ratio = group_log - *source_log;
            let source_row = (row >> (ratio + 1) << 1) + (row & 1);
            numerator += BaseField::from_u32_unchecked(source[source_row]) * *c - *b;
        }
        for (coordinate, value) in numerator.to_m31_array().into_iter().enumerate() {
            output[coordinate][row] = value.0;
        }
    }
    (first, output)
}

#[test]
fn eager_and_capture_replay_match_reference_grouping_and_lifting() {
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
    let config = QuotientNumeratorWorkspaceConfig {
        lifting_log_size: 6,
        log_blowup_factor: 2,
        max_lde_tile_words: 1,
    };
    let slots = slots();
    let arena = arena(&slots);
    let p0 = SECURE_FIELD_CIRCLE_GEN.mul(3);
    let p1 = SECURE_FIELD_CIRCLE_GEN.mul(7);
    let values = [
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
        SecureField::from_u32_unchecked(23, 29, 31, 37),
    ];
    let eval_a = (0..64)
        .map(|index| (17 * index + 3) as u32)
        .collect::<Vec<_>>();
    let eval_b = (0..32)
        .map(|index| (29 * index + 5) as u32)
        .collect::<Vec<_>>();
    upload(&arena, OODS_POINTS, &point_words(&[p0, p1, p1]));
    upload(&arena, OODS_VALUES, &secure_words(&values));
    upload(&arena, EVAL_A, &eval_a);
    upload(&arena, EVAL_B, &eval_b);

    let columns = vec![
        QuotientNumeratorColumn {
            coefficient_log_size: 4,
            source: QuotientNumeratorColumnSource::Evaluation(arena.bind(EVAL_A).unwrap()),
            samples: vec![
                QuotientOodsSample {
                    input_index: 0,
                    shape_point: p0,
                },
                QuotientOodsSample {
                    input_index: 1,
                    shape_point: p1,
                },
            ],
        },
        QuotientNumeratorColumn {
            coefficient_log_size: 3,
            source: QuotientNumeratorColumnSource::Evaluation(arena.bind(EVAL_B).unwrap()),
            samples: vec![QuotientOodsSample {
                input_index: 2,
                shape_point: p1,
            }],
        },
    ];
    let topology = columns
        .iter()
        .map(QuotientNumeratorColumnTopology::from)
        .collect::<Vec<_>>();
    let requirements = quotient_numerator_workspace_requirements(config, &topology).unwrap();
    let destinations = requirements
        .groups
        .iter()
        .enumerate()
        .map(|(group, requirement)| QuotientNumeratorDestination {
            log_size: requirement.log_size,
            coordinates: std::array::from_fn(|coordinate| {
                arena.bind(output_id(group, coordinate)).unwrap()
            }),
        })
        .collect::<Vec<_>>();
    let prepared = PreparedQuotientNumeratorGraph::prepare(
        &arena,
        config,
        &columns,
        arena.bind(OODS_POINTS).unwrap(),
        arena.bind(OODS_VALUES).unwrap(),
        arena.bind(RANDOM_COEFFICIENT).unwrap(),
        arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
        arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
        &destinations,
        arena.bind(TWIDDLES).unwrap(),
        &slots,
    )
    .unwrap();

    let check = |alpha: SecureField| {
        let point_words = read_words(
            &arena,
            arena.bind(SAMPLE_POINTS_OUTPUT).unwrap(),
            8 * requirements.groups.len(),
        );
        let first_words = read_words(
            &arena,
            arena.bind(FIRST_TERMS_OUTPUT).unwrap(),
            4 * requirements.groups.len(),
        );
        let period = CanonicCoset::new(config.lifting_log_size)
            .step()
            .repeated_double(6);
        let periodic_point = p1 + period.into_ef();
        let terms = [
            (0usize, 4u32, values[1], periodic_point, eval_a.as_slice()),
            (1, 4, values[0], p0, eval_a.as_slice()),
            (2, 4, values[1], p1, eval_a.as_slice()),
            (3, 3, values[2], p1, eval_b.as_slice()),
        ];
        for (group_index, group) in requirements.groups.iter().enumerate() {
            let point = &point_words[8 * group_index..8 * group_index + 8];
            assert_eq!(
                stwo::core::circle::CirclePoint {
                    x: from_words(&point[..4]),
                    y: from_words(&point[4..]),
                },
                group.shape_point
            );
            let (first, output) = expected_group(group.shape_point, group.log_size, alpha, &terms);
            assert_eq!(
                from_words(&first_words[4 * group_index..4 * group_index + 4]),
                first
            );
            for coordinate in 0..4 {
                assert_eq!(
                    read_words(
                        &arena,
                        destinations[group_index].coordinates[coordinate],
                        1usize << group.log_size,
                    ),
                    output[coordinate]
                );
            }
        }
    };

    let eager_alpha = SecureField::from_u32_unchecked(41, 43, 47, 53);
    upload(&arena, RANDOM_COEFFICIENT, &secure_words(&[eager_alpha]));
    arena.context().sync().unwrap();
    prepared.launch().unwrap();
    arena.context().sync().unwrap();
    check(eager_alpha);
    let capture = arena.context().capture().unwrap();
    prepared.launch().unwrap();
    let graph = capture.finish().unwrap();
    let replay_alpha = SecureField::from_u32_unchecked(59, 61, 67, 71);
    upload(&arena, RANDOM_COEFFICIENT, &secure_words(&[replay_alpha]));
    graph.launch(arena.context()).unwrap();
    arena.context().sync().unwrap();
    check(replay_alpha);
}
