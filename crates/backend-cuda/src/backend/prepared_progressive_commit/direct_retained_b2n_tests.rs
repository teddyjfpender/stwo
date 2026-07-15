use super::*;
use crate::backend::progressive_commit::{
    ProgressiveCommitGeometry, ProgressiveCommitGroupGeometry,
};
use crate::backend::progressive_ntt_leaf_fusion::ProgressiveNttLeafFusionMode;

fn commit(blowup: u32) -> CommitProgram {
    CommitProgram::compile(
        CommitWorkspaceConfig {
            log_blowup_factor: blowup,
            lifting_log_size: 7,
            unretained_bottom_layers: 4,
            max_fused_tail_levels: 0,
        },
        ProgressiveCommitGeometry {
            lifting_log_size: 7,
            log_blowup_factor: blowup,
            groups: vec![ProgressiveCommitGroupGeometry {
                coefficient_log_sizes: vec![3, 3, 4],
                retain_evaluations: true,
            }],
        },
        ProgressiveNttLeafFusionMode::Separate,
        false,
    )
    .unwrap()
}

fn slots(batch_count: usize) -> ProgressiveCommitWorkspaceSlots {
    ProgressiveCommitWorkspaceSlots {
        leaves: ProgressiveLeafWorkspaceSlots {
            lde_scratch: None,
            state_ping: ArenaSlotId(100),
            state_pong: None,
            leaf_hashes: ArenaSlotId(101),
            batches: (0..batch_count)
                .map(|batch| {
                    let base = 10 + u32::try_from(batch).unwrap() * 3;
                    ProgressiveBatchSlots {
                        coefficient_ptrs: ArenaSlotId(base),
                        coefficient_sizes: ArenaSlotId(base + 1),
                        output_ptrs: ArenaSlotId(base + 2),
                    }
                })
                .collect(),
        },
        merkle: MerkleFromLeavesSlots {
            leaves: ArenaSlotId(101),
            merkle_scratch: None,
            retained_layers: vec![],
            tail_level_ptrs: None,
            tail_outputs: vec![],
        },
    }
}

#[test]
fn program_is_base_or_interaction_only_and_seals_exact_batches() {
    let admitted = commit(1);
    for role in [TraceTreeRole::Base, TraceTreeRole::Interaction] {
        let direct = DirectRetainedB2nProgram::compile(role, &admitted).unwrap();
        assert_eq!(direct.role(), role);
        assert_eq!(direct.batches().len(), 2);
        assert_eq!(direct.batches()[0].canonical_columns, [0, 1]);
        assert_eq!(direct.batches()[1].canonical_columns, [2]);
        assert!(direct
            .batches()
            .iter()
            .all(|batch| batch.retained_log_size == batch.source_log_size + 1));
    }
    for role in [TraceTreeRole::Preprocessed, TraceTreeRole::Composition] {
        assert_eq!(
            DirectRetainedB2nProgram::compile(role, &admitted),
            Err(DirectRetainedB2nError::UnsupportedRole(role))
        );
    }
    assert_eq!(
        DirectRetainedB2nProgram::compile(TraceTreeRole::Base, &commit(2)),
        Err(DirectRetainedB2nError::UnsupportedBlowup(2))
    );
}

#[test]
fn pointer_tables_exclusively_replace_the_existing_lde_tables() {
    let direct = DirectRetainedB2nProgram::compile(TraceTreeRole::Base, &commit(1)).unwrap();
    let slots = slots(direct.batches().len());
    let required = direct.arena_slot_requirements(&slots).unwrap();
    let expected = slots
        .leaves
        .batches
        .iter()
        .flat_map(|batch| [batch.coefficient_ptrs, batch.output_ptrs])
        .collect::<Vec<_>>();
    assert_eq!(
        required.iter().map(|entry| entry.id).collect::<Vec<_>>(),
        expected
    );

    let mut aliased = slots;
    aliased.leaves.batches[1].coefficient_ptrs = aliased.leaves.batches[0].output_ptrs;
    assert!(matches!(
        direct.arena_slot_requirements(&aliased),
        Err(DirectRetainedB2nError::InvalidAlias { .. })
    ));
}

#[test]
fn global_inverse_tree_extent_is_preserved_for_suffix_selection() {
    let direct = DirectRetainedB2nProgram::compile(TraceTreeRole::Base, &commit(1)).unwrap();
    let global_words = direct.twiddle_words * 4;
    let twiddles = ArenaSlice::dangling_at_for_test(9, 700, global_words);
    let admitted = admit_inverse_twiddles(&direct, twiddles, twiddles.context_token()).unwrap();
    assert_eq!(admitted, u32::try_from(global_words).unwrap());
    assert!(usize::try_from(admitted).unwrap() > direct.twiddle_words);

    let oversized =
        ArenaSlice::dangling_at_for_test(10, 800, usize::try_from(u32::MAX).unwrap() + 1);
    assert_eq!(
        admit_inverse_twiddles(&direct, oversized, oversized.context_token()),
        Err(DirectRetainedB2nError::SizeOverflow)
    );
}

#[test]
fn cpu_oracle_runs_real_b2n_and_duplicates_exact_words() {
    let direct = DirectRetainedB2nProgram::compile(TraceTreeRole::Base, &commit(1)).unwrap();
    let sources = direct
        .batches()
        .iter()
        .flat_map(|batch| {
            batch.canonical_columns.iter().map(|&canonical| {
                (0..1usize << batch.source_log_size)
                    .map(|row| ((canonical * 97 + row * 31 + 11) as u32) % P)
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    let oracle = direct.oracle(&sources).unwrap();
    for (column, retained) in oracle.retained_stage_two_inputs.iter().enumerate() {
        let half = retained.len() / 2;
        assert_ne!(retained[..half], sources[column]);
        assert_eq!(retained[..half], retained[half..]);
    }
}

#[test]
fn value_aliases_allow_only_the_same_owner_lower_prefix() {
    let direct = DirectRetainedB2nProgram::compile(TraceTreeRole::Base, &commit(1)).unwrap();
    let mut columns = vec![
        DirectRetainedB2nColumn {
            source_evaluations: ArenaSlice::dangling_at_for_test(1, 100, 8),
            retained_output: ArenaSlice::dangling_at_for_test(1, 100, 16),
        },
        DirectRetainedB2nColumn {
            source_evaluations: ArenaSlice::dangling_at_for_test(2, 200, 8),
            retained_output: ArenaSlice::dangling_at_for_test(3, 300, 16),
        },
        DirectRetainedB2nColumn {
            source_evaluations: ArenaSlice::dangling_at_for_test(4, 400, 16),
            retained_output: ArenaSlice::dangling_at_for_test(5, 500, 32),
        },
    ];
    let token = columns[0].source_evaluations.context_token();
    let logical = bind_logical_columns(&direct, &columns, token).unwrap();
    let twiddles = ArenaSlice::dangling_at_for_test(9, 700, 64);
    validate_value_aliases(&logical, twiddles).unwrap();
    assert!(exact_lower_prefix_alias(logical[0]));

    columns[0].source_evaluations = ArenaSlice::dangling_at_for_test(1, 101, 8);
    let logical = bind_logical_columns(&direct, &columns, token).unwrap();
    assert!(matches!(
        validate_value_aliases(&logical, twiddles),
        Err(DirectRetainedB2nError::InvalidAlias { .. })
    ));

    columns[0].source_evaluations = ArenaSlice::dangling_at_for_test(6, 100, 8);
    let logical = bind_logical_columns(&direct, &columns, token).unwrap();
    assert!(matches!(
        validate_value_aliases(&logical, twiddles),
        Err(DirectRetainedB2nError::InvalidAlias { .. })
    ));
}
