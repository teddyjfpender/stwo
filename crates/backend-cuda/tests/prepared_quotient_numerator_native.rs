//! Native CUDA correctness and capture gate for prepared quotient numerators.

#![cfg(stwo_cuda_link)]

use stwo::core::constraints::complex_conjugate_line_coeffs;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::quotients::PointSample;
use stwo::core::poly::circle::CanonicCoset;
use stwo_backend_cuda::{
    quotient_numerator_workspace_requirements, ArenaLayout, ArenaSlice, ArenaSlotId, ArenaSlotSpec,
    CudaExecContext, DeviceArena, PreparedQuotientNumeratorGraph, QuotientNumeratorColumn,
    QuotientNumeratorColumnSource, QuotientNumeratorColumnTopology, QuotientNumeratorDestination,
    QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceSlots, QuotientOodsSample,
};

const OODS_POINTS: ArenaSlotId = ArenaSlotId(50_000);
const OODS_VALUES: ArenaSlotId = ArenaSlotId(50_001);
const RANDOM_COEFFICIENT: ArenaSlotId = ArenaSlotId(50_002);
const SAMPLE_POINTS_OUTPUT: ArenaSlotId = ArenaSlotId(50_003);
const FIRST_TERMS_OUTPUT: ArenaSlotId = ArenaSlotId(50_004);
const TWIDDLES: ArenaSlotId = ArenaSlotId(50_005);
const EVAL_A: ArenaSlotId = ArenaSlotId(50_006);
const EVAL_B: ArenaSlotId = ArenaSlotId(50_007);
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
    ];
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
            len_words: SLOT_WORDS,
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
