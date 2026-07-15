//! Native eager/capture differential for compact expand/absorb fusion.
//!
//! The materialized-only executor is compared with the ordinary compact graph
//! in independent arenas. Production terminal-fused shapes are rejected by the
//! binder and belong to the later composed executor gate.

use std::collections::BTreeMap;

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo_backend_cuda::{
    compact_domain_arena_slot_requirements, full_lifting_leaf_oracle,
    fused_compact_domain_arena_slot_requirements, progressive_leaf_oracle, ArenaLayout, ArenaSlice,
    ArenaSlotId, ArenaSlotSpec, CommitArenaSlotRequirement, CommitCoefficientColumn, CommitProgram,
    CommitWorkspaceConfig, CompactDomainProgram, CudaExecContext, DeviceArena,
    DirectCompactTerminalProgram, DirectRetainedB2nProgram, DomainCooperativeProgram,
    FusedCompactDomainProgram, MerkleFromLeavesSlots, PreparedCompactDomainCommitGraph,
    PreparedFusedCompactDomainCommitGraph, ProgressiveBatchSlots, ProgressiveCommitGeometry,
    ProgressiveCommitGroupGeometry, ProgressiveCommitWorkspaceRequirements,
    ProgressiveCommitWorkspaceSlots, ProgressiveLeafWorkspaceSlots, ProgressiveNttLeafFusionMode,
    TraceTreeRole,
};

const TWIDDLES: ArenaSlotId = ArenaSlotId(50_000);
const COEFFICIENT_BASE: u32 = 51_000;
const OUTPUT_BASE: u32 = 52_000;

#[derive(Clone)]
struct Case {
    name: &'static str,
    lifting_log_size: u32,
    groups: Vec<Vec<u32>>,
    transitions: usize,
}

fn programs(
    case: &Case,
) -> (
    CommitProgram,
    DomainCooperativeProgram,
    CompactDomainProgram,
    FusedCompactDomainProgram,
    DirectRetainedB2nProgram,
    DirectCompactTerminalProgram,
) {
    let base = CommitProgram::compile(
        CommitWorkspaceConfig {
            log_blowup_factor: 1,
            lifting_log_size: case.lifting_log_size,
            unretained_bottom_layers: 4,
            max_fused_tail_levels: 2,
        },
        ProgressiveCommitGeometry {
            lifting_log_size: case.lifting_log_size,
            log_blowup_factor: 1,
            groups: case
                .groups
                .iter()
                .map(|logs| ProgressiveCommitGroupGeometry {
                    coefficient_log_sizes: logs.clone(),
                    retain_evaluations: true,
                })
                .collect(),
        },
        ProgressiveNttLeafFusionMode::Fused16,
        true,
    )
    .unwrap();
    let domain = DomainCooperativeProgram::compile_mode_a(&base).unwrap();
    let compact = CompactDomainProgram::compile(&base, &domain).unwrap();
    let fused = FusedCompactDomainProgram::compile(&base, &domain, &compact).unwrap();
    let direct = DirectRetainedB2nProgram::compile(TraceTreeRole::Base, &base).unwrap();
    let terminal = DirectCompactTerminalProgram::compile(&compact, &direct).unwrap();
    assert_eq!(
        fused.receipt().transitions.len(),
        case.transitions,
        "{}",
        case.name
    );
    (base, domain, compact, fused, direct, terminal)
}

fn slots(requirements: &ProgressiveCommitWorkspaceRequirements) -> ProgressiveCommitWorkspaceSlots {
    let mut next = 1u32;
    let mut id = || {
        let result = ArenaSlotId(next);
        next += 1;
        result
    };
    let slab = id();
    ProgressiveCommitWorkspaceSlots {
        leaves: ProgressiveLeafWorkspaceSlots {
            lde_scratch: requirements.leaves.lde_scratch_words.map(|_| id()),
            state_ping: slab,
            state_pong: requirements.leaves.state_pong_words.map(|_| slab),
            leaf_hashes: slab,
            batches: requirements
                .leaves
                .batches
                .iter()
                .map(|_| ProgressiveBatchSlots {
                    coefficient_ptrs: id(),
                    coefficient_sizes: id(),
                    output_ptrs: id(),
                })
                .collect(),
        },
        merkle: MerkleFromLeavesSlots {
            leaves: slab,
            merkle_scratch: requirements.merkle.merkle_scratch_words.map(|_| slab),
            retained_layers: requirements
                .merkle
                .retained_layers
                .iter()
                .map(|_| id())
                .collect(),
            tail_level_ptrs: requirements.merkle.tail_pointer_words.map(|_| id()),
            tail_outputs: requirements
                .merkle
                .tail_outputs
                .iter()
                .map(|_| id())
                .collect(),
        },
    }
}

fn insert(
    requirements: &mut BTreeMap<ArenaSlotId, (usize, usize)>,
    requirement: CommitArenaSlotRequirement,
) {
    requirements
        .entry(requirement.id)
        .and_modify(|current| {
            current.0 = current.0.max(requirement.len_words);
            current.1 = current.1.max(requirement.alignment_words);
        })
        .or_insert((requirement.len_words, requirement.alignment_words));
}

fn arena(base: &CommitProgram, workspace: Vec<CommitArenaSlotRequirement>) -> DeviceArena {
    let mut requirements = BTreeMap::new();
    for requirement in workspace {
        insert(&mut requirements, requirement);
    }
    insert(
        &mut requirements,
        CommitArenaSlotRequirement {
            id: TWIDDLES,
            len_words: base.requirements().leaves.twiddle_words,
            alignment_words: 1,
        },
    );
    for column in &base.requirements().leaves.plan.columns {
        for (id, len_words) in [
            (
                ArenaSlotId(COEFFICIENT_BASE + column.canonical_index as u32),
                1usize << column.coefficient_log_size,
            ),
            (
                ArenaSlotId(OUTPUT_BASE + column.canonical_index as u32),
                1usize << column.evaluation_log_size,
            ),
        ] {
            insert(
                &mut requirements,
                CommitArenaSlotRequirement {
                    id,
                    len_words,
                    alignment_words: 1,
                },
            );
        }
    }
    let mut offset_words = 0usize;
    let specs = requirements
        .into_iter()
        .map(|(id, (len_words, alignment_words))| {
            offset_words = offset_words.next_multiple_of(alignment_words);
            let spec = ArenaSlotSpec {
                id,
                offset_words,
                len_words,
                alignment_words,
            };
            offset_words += len_words;
            spec
        })
        .collect::<Vec<_>>();
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset_words, &specs).unwrap(),
    )
    .unwrap()
}

fn upload(arena: &DeviceArena, destination: ArenaSlice, words: &[u32]) {
    assert_eq!(destination.len_words(), words.len());
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

fn read_hashes(arena: &DeviceArena, source: ArenaSlice) -> Vec<Blake2sHash> {
    let mut hashes = vec![Blake2sHash::default(); source.len_words() / 8];
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                hashes.as_mut_ptr().cast(),
                source.as_void_ptr().cast_const(),
                core::mem::size_of_val(hashes.as_slice()),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
    hashes
}

fn read_words(arena: &DeviceArena, source: ArenaSlice) -> Vec<u32> {
    let mut words = vec![0; source.len_words()];
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                words.as_mut_ptr().cast(),
                source.as_void_ptr().cast_const(),
                core::mem::size_of_val(words.as_slice()),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
    words
}

fn coefficients(base: &CommitProgram, seed: u64) -> Vec<Vec<u32>> {
    base.requirements()
        .leaves
        .plan
        .columns
        .iter()
        .map(|column| {
            (0..1usize << column.coefficient_log_size)
                .map(|row| {
                    (seed
                        .wrapping_add((column.canonical_index as u64 + 1) * 104_729)
                        .wrapping_add(row as u64 * 7_919)
                        % 0x7fff_ffff) as u32
                })
                .collect()
        })
        .collect()
}

fn evaluations(base: &CommitProgram, coefficients: &[Vec<u32>]) -> Vec<Vec<u32>> {
    base.requirements()
        .leaves
        .plan
        .columns
        .iter()
        .zip(coefficients)
        .map(|(column, words)| {
            let domain = CanonicCoset::new(column.evaluation_log_size).circle_domain();
            let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
            CircleCoefficients::<CpuBackend>::new(
                words
                    .iter()
                    .copied()
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            )
            .evaluate_with_twiddles(domain, &twiddles)
            .values
            .into_iter()
            .map(|value| value.0)
            .collect()
        })
        .collect()
}

fn seed_arena(
    arena: &DeviceArena,
    base: &CommitProgram,
    coefficient_words: &[Vec<u32>],
) -> (Vec<CommitCoefficientColumn>, Vec<Option<ArenaSlice>>) {
    let domain = CanonicCoset::new(base.identity().geometry.lifting_log_size).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset)
        .twiddles
        .iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    upload(arena, arena.bind(TWIDDLES).unwrap(), &twiddles);
    let columns = base
        .requirements()
        .leaves
        .plan
        .columns
        .iter()
        .zip(coefficient_words)
        .map(|(column, words)| {
            let source = arena
                .bind(ArenaSlotId(
                    COEFFICIENT_BASE + column.canonical_index as u32,
                ))
                .unwrap();
            upload(arena, source, words);
            CommitCoefficientColumn {
                coefficients: source,
                log_size: column.coefficient_log_size,
            }
        })
        .collect::<Vec<_>>();
    let retained = base
        .requirements()
        .leaves
        .plan
        .columns
        .iter()
        .map(|column| {
            Some(
                arena
                    .bind(ArenaSlotId(OUTPUT_BASE + column.canonical_index as u32))
                    .unwrap(),
            )
        })
        .collect();
    arena.context().sync().unwrap();
    (columns, retained)
}

fn assert_outputs(
    base: &CommitProgram,
    evaluations: &[Vec<u32>],
    arena: &DeviceArena,
    leaves: ArenaSlice,
    retained: &[Option<ArenaSlice>],
) -> Vec<Blake2sHash> {
    let actual = read_hashes(arena, leaves);
    assert_eq!(
        actual,
        progressive_leaf_oracle(&base.requirements().leaves.plan, evaluations).unwrap()
    );
    assert_eq!(
        actual,
        full_lifting_leaf_oracle(&base.requirements().leaves.plan, evaluations).unwrap()
    );
    for (expected, output) in evaluations.iter().zip(retained) {
        assert_eq!(&read_words(arena, output.unwrap()), expected);
    }
    actual
}

fn run(case: Case) {
    let (base, domain, compact, fused, direct, terminal) = programs(&case);
    let workspace_slots = slots(base.requirements());
    let baseline_arena = arena(
        &base,
        compact_domain_arena_slot_requirements(&compact, &base, &domain, &workspace_slots).unwrap(),
    );
    let candidate_arena = arena(
        &base,
        fused_compact_domain_arena_slot_requirements(
            &fused,
            &base,
            &domain,
            &compact,
            &workspace_slots,
        )
        .unwrap(),
    );
    let first_coefficients = coefficients(&base, 0x51ab_1eaf);
    let first_evaluations = evaluations(&base, &first_coefficients);
    let (baseline_columns, baseline_retained) =
        seed_arena(&baseline_arena, &base, &first_coefficients);
    let (candidate_columns, candidate_retained) =
        seed_arena(&candidate_arena, &base, &first_coefficients);
    let baseline: PreparedCompactDomainCommitGraph<'_> = compact
        .bind_prepared(
            &baseline_arena,
            &base,
            &domain,
            &workspace_slots,
            &baseline_columns,
            &baseline_retained,
            baseline_arena.bind(TWIDDLES).unwrap(),
        )
        .unwrap();
    let candidate: PreparedFusedCompactDomainCommitGraph<'_> = fused
        .bind_prepared_materialized_only(
            &candidate_arena,
            &base,
            &domain,
            &compact,
            &direct,
            &terminal,
            &workspace_slots,
            &candidate_columns,
            &candidate_retained,
            candidate_arena.bind(TWIDDLES).unwrap(),
        )
        .unwrap();

    baseline.launch().unwrap();
    candidate.launch().unwrap();
    let eager_baseline = assert_outputs(
        &base,
        &first_evaluations,
        &baseline_arena,
        baseline.leaf_hashes(),
        &baseline_retained,
    );
    let eager_candidate = assert_outputs(
        &base,
        &first_evaluations,
        &candidate_arena,
        candidate.leaf_hashes(),
        &candidate_retained,
    );
    assert_eq!(
        eager_candidate, eager_baseline,
        "{} eager leaves",
        case.name
    );
    assert_eq!(
        candidate.read_root_at_transcript_boundary().unwrap(),
        baseline.read_root_at_transcript_boundary().unwrap(),
        "{} eager root",
        case.name
    );

    let capture = baseline_arena.context().capture().unwrap();
    baseline.launch().unwrap();
    let baseline_graph = capture.finish().unwrap();
    let capture = candidate_arena.context().capture().unwrap();
    candidate.launch().unwrap();
    let candidate_graph = capture.finish().unwrap();
    let expected_saved = fused
        .receipt()
        .transitions
        .iter()
        .map(|transition| u64::from(transition.expansion_bands))
        .sum::<u64>();
    assert_eq!(
        baseline_graph.kernel_nodes() - candidate_graph.kernel_nodes(),
        expected_saved,
        "{} captured kernel savings",
        case.name
    );
    baseline_graph.launch(baseline_arena.context()).unwrap();
    candidate_graph.launch(candidate_arena.context()).unwrap();
    assert_eq!(
        assert_outputs(
            &base,
            &first_evaluations,
            &candidate_arena,
            candidate.leaf_hashes(),
            &candidate_retained,
        ),
        eager_candidate,
        "{} replay",
        case.name
    );

    let second_coefficients = coefficients(&base, 0xc001_cafe);
    for ((column, baseline), candidate) in base
        .requirements()
        .leaves
        .plan
        .columns
        .iter()
        .zip(&baseline_columns)
        .zip(&candidate_columns)
    {
        let words = &second_coefficients[column.canonical_index];
        upload(&baseline_arena, baseline.coefficients, words);
        upload(&candidate_arena, candidate.coefficients, words);
    }
    baseline_graph.launch(baseline_arena.context()).unwrap();
    candidate_graph.launch(candidate_arena.context()).unwrap();
    let second_evaluations = evaluations(&base, &second_coefficients);
    let mutated_baseline = assert_outputs(
        &base,
        &second_evaluations,
        &baseline_arena,
        baseline.leaf_hashes(),
        &baseline_retained,
    );
    let mutated_candidate = assert_outputs(
        &base,
        &second_evaluations,
        &candidate_arena,
        candidate.leaf_hashes(),
        &candidate_retained,
    );
    assert_eq!(
        mutated_candidate, mutated_baseline,
        "{} mutation",
        case.name
    );
    assert_ne!(
        mutated_candidate, eager_candidate,
        "{} stale replay",
        case.name
    );
    assert_eq!(
        candidate.read_root_at_transcript_boundary().unwrap(),
        baseline.read_root_at_transcript_boundary().unwrap(),
        "{} mutated root",
        case.name
    );
}

#[test]
#[cfg_attr(not(stwo_cuda_link), ignore = "requires native CUDA")]
fn fused_expand_absorb_matches_legacy_eager_capture_and_mutation() {
    assert!(stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT);
    for case in [
        Case {
            name: "zero-transition",
            lifting_log_size: 5,
            groups: vec![vec![4; 17]],
            transitions: 0,
        },
        Case {
            name: "tail16-odd",
            lifting_log_size: 5,
            groups: vec![vec![3; 16], vec![4; 1]],
            transitions: 1,
        },
        Case {
            name: "tail32-odd",
            lifting_log_size: 5,
            groups: vec![vec![3; 32], vec![4; 1]],
            transitions: 1,
        },
        Case {
            name: "tail33-multilog",
            lifting_log_size: 7,
            groups: vec![vec![3; 33], vec![6; 1]],
            transitions: 1,
        },
        Case {
            name: "even-two-transition",
            lifting_log_size: 7,
            groups: vec![vec![3; 17], vec![4; 16], vec![6; 1]],
            transitions: 2,
        },
    ] {
        run(case);
    }
}
