use std::collections::BTreeSet;

use super::super::{BG_FUSED_SEMANTIC_HASH, BG_N_AUX, BG_N_COLS, BG_N_TRACE};
use super::*;

const USES: [(usize, usize, usize, u32); 17] = [
    (0, 0, 4, 112_558_620),
    (4, 3, 4, 112_558_620),
    (8, 6, 4, 521_092_554),
    (12, 9, 4, 521_092_554),
    (16, 12, 4, 648_362_599),
    (20, 15, 4, 45_448_144),
    (24, 18, 4, 648_362_599),
    (28, 21, 4, 45_448_144),
    (32, 24, 4, 112_558_620),
    (36, 27, 4, 112_558_620),
    (40, 30, 4, 521_092_554),
    (44, 33, 4, 521_092_554),
    (48, 36, 4, 62_225_763),
    (52, 39, 4, 95_781_001),
    (56, 42, 4, 62_225_763),
    (60, 45, 4, 95_781_001),
    (64, 48, 21, BG_FINAL_RELATION),
];

use BlakeGRelationColumnSource::{Auxiliary as A, BaseTrace as B};

const EXPECTED_COLUMNS: [BlakeGRelationColumnSource; BG_N_PROJECTED_RELATION_COLUMNS] = [
    A(0),
    A(2),
    B(18),
    B(14),
    B(16),
    B(19),
    A(1),
    A(3),
    B(20),
    B(15),
    B(17),
    B(21),
    A(4),
    A(6),
    B(28),
    B(24),
    B(26),
    B(29),
    A(5),
    A(7),
    B(30),
    B(25),
    B(27),
    B(31),
    A(8),
    A(10),
    B(38),
    B(34),
    B(36),
    B(39),
    A(9),
    A(11),
    B(40),
    B(35),
    B(37),
    B(41),
    A(12),
    A(14),
    B(48),
    B(44),
    B(46),
    B(49),
    A(13),
    A(15),
    B(50),
    B(45),
    B(47),
    B(51),
    B(0),
    B(1),
    B(2),
    B(3),
    B(4),
    B(5),
    B(6),
    B(7),
    B(8),
    B(9),
    B(10),
    B(11),
    B(32),
    B(33),
    A(16),
    A(17),
    B(42),
    B(43),
    A(18),
    A(19),
];

fn legacy_words(columns: &[u32; BG_N_COLS]) -> [u32; BG_N_LOOKUP_WORDS] {
    let mut words = [0u32; BG_N_LOOKUP_WORDS];
    for tuple in 0..BG_TUPLE_RELATIONS.len() {
        words[4 * tuple] = BG_TUPLE_RELATIONS[tuple];
        for word in 0..3 {
            words[4 * tuple + 1 + word] =
                columns[BG_LOOKUP_TUPLE_COLUMNS[3 * tuple + word] as usize];
        }
    }
    words[64] = BG_FINAL_RELATION;
    for (word, &column) in BG_FINAL_COLUMNS.iter().enumerate() {
        words[65 + word] = columns[column as usize];
    }
    words[85] = 1;
    words[86] = columns[52];
    words
}

fn projected_columns(
    trace: &[u32; BG_N_TRACE],
    auxiliary: &[u32; BG_N_AUX],
) -> [u32; BG_N_PROJECTED_RELATION_COLUMNS] {
    BG_PROJECTED_RELATION_COLUMNS.map(|source| match source {
        BlakeGRelationColumnSource::BaseTrace(column) => trace[column as usize],
        BlakeGRelationColumnSource::Auxiliary(column) => auxiliary[column as usize],
    })
}

#[test]
fn projected_map_is_exhaustive_pinned_and_minimal() {
    assert_eq!(BG_PROJECTED_RELATION_COLUMNS, EXPECTED_COLUMNS);
    assert_eq!(BG_PROJECTED_RELATION_MAP_HASH, 0x4898_bf01_628f_45b7);
    let mut base_entries = 0;
    let mut auxiliary_entries = 0;
    let mut base_columns = BTreeSet::new();
    let mut auxiliary_columns = BTreeSet::new();
    for source in BG_PROJECTED_RELATION_COLUMNS {
        match source {
            BlakeGRelationColumnSource::BaseTrace(column) => {
                base_entries += 1;
                base_columns.insert(column);
            }
            BlakeGRelationColumnSource::Auxiliary(column) => {
                auxiliary_entries += 1;
                auxiliary_columns.insert(column);
            }
        }
    }
    assert_eq!((base_entries, auxiliary_entries), (48, 20));
    assert_eq!(base_columns.len(), 48);
    assert_eq!(auxiliary_columns, (0..20).collect());
    let unused = (0..BG_N_TRACE as u8)
        .filter(|column| !base_columns.contains(column))
        .collect::<Vec<_>>();
    assert_eq!(unused, BG_PROJECTED_UNUSED_TRACE_COLUMNS);
    assert_eq!(
        USES.len() + BG_N_PROJECTED_RELATION_COLUMNS + 2,
        BG_N_LOOKUP_WORDS
    );
    assert!(blake_g_projected_relation_identity_is_exact(
        "blake_g",
        BG_FUSED_SEMANTIC_HASH,
        BG_N_LOOKUP_WORDS,
        BG_N_PROJECTED_RELATION_COLUMNS,
        BG_PROJECTED_RELATION_MAP_HASH,
    ));
    for mutation in [
        (
            "blake_g_mutated",
            BG_FUSED_SEMANTIC_HASH,
            BG_N_LOOKUP_WORDS,
            BG_N_PROJECTED_RELATION_COLUMNS,
            BG_PROJECTED_RELATION_MAP_HASH,
        ),
        (
            "blake_g",
            BG_FUSED_SEMANTIC_HASH ^ 1,
            BG_N_LOOKUP_WORDS,
            BG_N_PROJECTED_RELATION_COLUMNS,
            BG_PROJECTED_RELATION_MAP_HASH,
        ),
        (
            "blake_g",
            BG_FUSED_SEMANTIC_HASH,
            BG_N_LOOKUP_WORDS - 1,
            BG_N_PROJECTED_RELATION_COLUMNS,
            BG_PROJECTED_RELATION_MAP_HASH,
        ),
        (
            "blake_g",
            BG_FUSED_SEMANTIC_HASH,
            BG_N_LOOKUP_WORDS,
            BG_N_PROJECTED_RELATION_COLUMNS - 1,
            BG_PROJECTED_RELATION_MAP_HASH,
        ),
        (
            "blake_g",
            BG_FUSED_SEMANTIC_HASH,
            BG_N_LOOKUP_WORDS,
            BG_N_PROJECTED_RELATION_COLUMNS,
            BG_PROJECTED_RELATION_MAP_HASH ^ 1,
        ),
    ] {
        assert!(!blake_g_projected_relation_identity_is_exact(
            mutation.0, mutation.1, mutation.2, mutation.3, mutation.4
        ));
    }
}

#[test]
fn projected_tuples_match_legacy_for_boundary_random_and_padding_rows() {
    let mut state = 0x9e37_79b9u32;
    for case in 0..1_024usize {
        let rows = 1usize << (case % 9 + 1);
        let n_real = [1, rows / 2, rows.saturating_sub(1), rows][case % 4].max(1);
        let row = [0, n_real.saturating_sub(1), n_real.min(rows - 1), rows - 1][case % 4];
        let mut columns = [0u32; BG_N_COLS];
        for value in &mut columns {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = state % 0x7fff_ffff;
        }
        columns[52] = u32::from(row < n_real);
        let trace = std::array::from_fn(|column| columns[column]);
        let auxiliary = std::array::from_fn(|column| columns[BG_N_TRACE + column]);
        let legacy = legacy_words(&columns);
        let projected = projected_columns(&trace, &auxiliary);
        for (use_index, &(legacy_offset, projected_offset, width, relation)) in
            USES.iter().enumerate()
        {
            assert_eq!(legacy[legacy_offset], relation);
            assert_eq!(
                projected[projected_offset..projected_offset + width - 1],
                legacy[legacy_offset + 1..legacy_offset + width],
                "case={case} use={use_index} row={row} n_real={n_real}",
            );
        }
        assert_eq!(legacy[85], 1, "One stays one on padding");
        assert_eq!(legacy[86], u32::from(row < n_real));
    }
}
