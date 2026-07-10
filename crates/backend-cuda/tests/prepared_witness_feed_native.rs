//! Native CUDA correctness/capture gate for resident witness count feeds.
//! A CPU/stub build compiles zero tests; GPU admission requires exactly one pass.

#![cfg(stwo_cuda_link)]

use core::ffi::c_void;

use stwo_backend_cuda::{
    witness_feed_clear_workspace_requirements, witness_feed_workspace_requirements, ArenaLayout,
    ArenaSlice, ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena,
    PreparedWitnessFeedClearGraph, PreparedWitnessFeedGraph, WitnessFeedArenaSlotRequirement,
    WitnessFeedClearWorkspaceSlots, WitnessFeedWorkspaceRequirements, WitnessFeedWorkspaceSlots,
    WITNESS_FEED_DESCRIPTOR_WORDS, WITNESS_FEED_NO_LUT,
};

const ROWS: usize = 32;
const SUB_WORDS: usize = 8;
const DESTINATION_WORDS: &[usize] = &[32, 8, 4, 1 << 8, 16 * (1 << 20)];

fn fold_descriptor(
    word_base: u32,
    bits: &[u32],
    relation: u32,
    table_size: u32,
    lut: u32,
    destination: u32,
) -> [u32; WITNESS_FEED_DESCRIPTOR_WORDS] {
    let mut entry = [0u32; WITNESS_FEED_DESCRIPTOR_WORDS];
    entry[0] = word_base;
    entry[1] = bits.len() as u32;
    entry[2..2 + bits.len()].copy_from_slice(bits);
    entry[7] = relation;
    entry[8] = table_size;
    entry[9] = lut;
    entry[10] = destination;
    entry
}

fn feed_inputs() -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut descriptors = Vec::new();
    descriptors.extend(fold_descriptor(0, &[2, 2], 0, 16, 0, 0));
    descriptors.extend(fold_descriptor(0, &[2, 2], 1, 16, 0, 0));
    let mut xor4 = fold_descriptor(0, &[4, 4, 4], 0, 1 << 8, 1, 3);
    xor4[11] = 2;
    descriptors.extend(xor4);
    let mut xor12 = fold_descriptor(4, &[12, 12, 12], 0, 1 << 20, WITNESS_FEED_NO_LUT, 4);
    xor12[11] = 3;
    descriptors.extend(xor12);
    let mut memory = [0u32; WITNESS_FEED_DESCRIPTOR_WORDS];
    memory[0] = 7;
    memory[1] = 1;
    memory[2] = 31;
    memory[8] = 8;
    memory[9] = WITNESS_FEED_NO_LUT;
    memory[10] = 1;
    memory[11] = 1;
    memory[12] = 4;
    memory[13] = 2;
    descriptors.extend(memory);
    (
        descriptors,
        vec![
            (0..16u32).rev().collect(),
            (0..256u32)
                .map(|value| value.rotate_left(3) & 255)
                .collect(),
        ],
    )
}

fn slots() -> WitnessFeedWorkspaceSlots {
    WitnessFeedWorkspaceSlots {
        descriptors: ArenaSlotId(2),
        lut_tables: vec![ArenaSlotId(3), ArenaSlotId(13)],
        lut_pointers: ArenaSlotId(4),
        multiplicity_destinations: vec![
            ArenaSlotId(5),
            ArenaSlotId(6),
            ArenaSlotId(7),
            ArenaSlotId(8),
            ArenaSlotId(12),
        ],
        multiplicity_pointers: ArenaSlotId(9),
    }
}

fn arena(
    requirements: &WitnessFeedWorkspaceRequirements,
    slots: &WitnessFeedWorkspaceSlots,
) -> DeviceArena {
    let mut requests = vec![WitnessFeedArenaSlotRequirement {
        id: ArenaSlotId(1),
        len_words: requirements.source_words,
        alignment_words: 1,
    }];
    requests.extend(requirements.arena_slot_requirements(slots).unwrap());
    let clear = witness_feed_clear_workspace_requirements(DESTINATION_WORDS).unwrap();
    requests.extend(clear.arena_slot_requirements(clear_slots()).unwrap());
    let mut offset = 0usize;
    let specs = requests
        .into_iter()
        .map(
            |WitnessFeedArenaSlotRequirement {
                 id,
                 len_words,
                 alignment_words,
             }| {
                offset = offset.next_multiple_of(alignment_words);
                let spec = ArenaSlotSpec {
                    id,
                    offset_words: offset,
                    len_words,
                    alignment_words,
                };
                offset += len_words;
                spec
            },
        )
        .collect::<Vec<_>>();
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset, &specs).unwrap(),
    )
    .unwrap()
}

fn clear_slots() -> WitnessFeedClearWorkspaceSlots {
    WitnessFeedClearWorkspaceSlots {
        destination_pointers: ArenaSlotId(10),
        destination_lengths: ArenaSlotId(11),
    }
}

fn upload(arena: &DeviceArena, destination: ArenaSlice, values: &[u32]) {
    assert_eq!(destination.len_words(), values.len());
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(
                destination.as_void_ptr(),
                values.as_ptr().cast::<c_void>(),
                core::mem::size_of_val(values),
            )
            .unwrap();
    }
}

fn read(arena: &DeviceArena, source: ArenaSlice) -> Vec<u32> {
    let mut result = vec![0u32; source.len_words()];
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                result.as_mut_ptr().cast::<c_void>(),
                source.as_void_ptr().cast_const(),
                source.len_bytes(),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
    result
}

fn source(seed: u32) -> Vec<u32> {
    let mut words = vec![0u32; ROWS * SUB_WORDS];
    for row in 0..ROWS {
        words[row] = (row as u32 + seed) & 3;
        words[ROWS + row] = ((row as u32 / 3) + 2 * seed) & 3;
        words[2 * ROWS + row] = words[row] ^ words[ROWS + row];
        let a12 = (17 * row as u32 + 13 * seed) & 0xfff;
        let b12 = (257 * row as u32 + 29 * seed) & 0xfff;
        words[4 * ROWS + row] = a12;
        words[5 * ROWS + row] = b12;
        words[6 * ROWS + row] = a12 ^ b12;
        words[7 * ROWS + row] = if row % 11 == 0 {
            (1 << 30) - 1
        } else if (row as u32 + seed) & 1 == 0 {
            (row as u32 + seed) & 3
        } else {
            (1 << 30) | ((row as u32 + 3 * seed) & 7)
        };
    }
    words
}

fn host_reference(source: &[u32], descriptors: &[u32], luts: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut counts = DESTINATION_WORDS
        .iter()
        .map(|&words| vec![0u32; words])
        .collect::<Vec<_>>();
    for entry in descriptors.chunks_exact(WITNESS_FEED_DESCRIPTOR_WORDS) {
        let word_base = entry[0] as usize;
        let width = entry[1] as usize;
        let table_size = entry[8] as usize;
        for row in 0..ROWS {
            if entry[11] == 1 {
                let value = source[word_base * ROWS + row];
                if value == (1 << 30) - 1 {
                    continue;
                }
                let tag = value >> 30;
                let index = (value & 0x3fff_ffff) as usize;
                if tag == 1 && index < table_size {
                    counts[entry[10] as usize][entry[7] as usize * table_size + index] += 1;
                } else if tag == 0 && index < entry[12] as usize {
                    let small_size = entry[12] as usize;
                    counts[entry[13] as usize][entry[7] as usize * small_size + index] += 1;
                }
                continue;
            }
            if entry[11] == 2 {
                let bits = entry[2];
                let mask = (1u32 << bits) - 1;
                let a = source[word_base * ROWS + row];
                let b = source[(word_base + 1) * ROWS + row];
                let c = source[(word_base + 2) * ROWS + row];
                if (a | b | c) <= mask && c == (a ^ b) {
                    let key = ((a << bits) | b) as usize;
                    let index = luts[entry[9] as usize][key] as usize;
                    counts[entry[10] as usize][entry[7] as usize * table_size + index] += 1;
                }
                continue;
            }
            if entry[11] == 3 {
                let a = source[word_base * ROWS + row];
                let b = source[(word_base + 1) * ROWS + row];
                let c = source[(word_base + 2) * ROWS + row];
                if (a | b | c) < (1 << 12) && c == (a ^ b) {
                    let column = ((a >> 10) << 2) | (b >> 10);
                    let table_row = ((a & 0x3ff) << 10) | (b & 0x3ff);
                    counts[entry[10] as usize]
                        [column as usize * table_size + table_row as usize] += 1;
                }
                continue;
            }
            let mut key = 0u32;
            for word in 0..width {
                key = (key << entry[2 + word]) | source[(word_base + word) * ROWS + row];
            }
            let keyed = i64::from(key) + i64::from(entry[12] as i32);
            if keyed < 0 || keyed as usize >= table_size {
                continue;
            }
            let key = keyed as usize;
            let index = if entry[9] == WITNESS_FEED_NO_LUT {
                key
            } else {
                luts[entry[9] as usize][key] as usize
            };
            if index < table_size {
                counts[entry[10] as usize][entry[7] as usize * table_size + index] += 1;
            }
        }
    }
    counts
}

fn snapshot(arena: &DeviceArena, prepared: &PreparedWitnessFeedGraph<'_>) -> Vec<Vec<u32>> {
    prepared
        .multiplicity_destinations()
        .iter()
        .copied()
        .map(|destination| read(arena, destination))
        .collect()
}

/// Eager, captured, and source-mutated replay counts match the descriptor-level
/// host reference, including canonical/expanded XOR and split memory feeds.
#[test]
fn prepared_witness_feed_eager_capture_and_mutated_replay_match_host() {
    let (descriptors, luts) = feed_inputs();
    let requirements = witness_feed_workspace_requirements(
        ROWS,
        SUB_WORDS,
        &descriptors,
        &luts,
        DESTINATION_WORDS,
    )
    .unwrap();
    let slots = slots();
    let arena = arena(&requirements, &slots);
    let source_slot = arena.bind(ArenaSlotId(1)).unwrap();
    let prepared = PreparedWitnessFeedGraph::prepare(
        &arena,
        source_slot,
        ROWS,
        SUB_WORDS,
        &descriptors,
        &luts,
        DESTINATION_WORDS,
        &slots,
    )
    .unwrap();
    let clear = PreparedWitnessFeedClearGraph::prepare(
        &arena,
        prepared.multiplicity_destinations(),
        clear_slots(),
    )
    .unwrap();

    let initial = source(0);
    upload(&arena, source_slot, &initial);
    arena.context().sync().unwrap();
    arena.context().reset_telemetry();
    clear.launch().unwrap();
    prepared.launch().unwrap();
    let eager_telemetry = arena.context().telemetry();
    assert_eq!(eager_telemetry.allocations, 0);
    assert_eq!(eager_telemetry.h2d_bytes, 0);
    assert_eq!(eager_telemetry.d2h_bytes, 0);
    assert_eq!(eager_telemetry.d2d_bytes, 0);
    assert_eq!(eager_telemetry.sync_calls, 0);
    assert_eq!(eager_telemetry.memset_bytes, 0);
    assert_eq!(
        snapshot(&arena, &prepared),
        host_reference(&initial, &descriptors, &luts)
    );

    let capture = arena.context().capture().unwrap();
    clear.launch().unwrap();
    prepared.launch().unwrap();
    let graph = capture.finish().unwrap();
    arena.context().reset_telemetry();
    graph.launch(arena.context()).unwrap();
    let replay_telemetry = arena.context().telemetry();
    assert_eq!(replay_telemetry.graph_launches, 1);
    assert_eq!(replay_telemetry.kernel_launches, 2);
    assert_eq!(replay_telemetry.allocations, 0);
    assert_eq!(replay_telemetry.h2d_bytes, 0);
    assert_eq!(replay_telemetry.d2h_bytes, 0);
    assert_eq!(replay_telemetry.d2d_bytes, 0);
    assert_eq!(replay_telemetry.sync_calls, 0);
    assert_eq!(
        snapshot(&arena, &prepared),
        host_reference(&initial, &descriptors, &luts)
    );

    let mutated = source(5);
    upload(&arena, source_slot, &mutated);
    graph.launch(arena.context()).unwrap();
    assert_eq!(
        snapshot(&arena, &prepared),
        host_reference(&mutated, &descriptors, &luts)
    );
}
