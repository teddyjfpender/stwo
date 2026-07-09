//! Native CUDA gate for the prepared FRI workspace. Keeping this as an
//! integration target avoids compiling the backend's legacy CUDA-only unit-test
//! modules when running the focused graph gate.

#![cfg(stwo_cuda_link)]

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasherGeneric};
use stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use stwo_backend_cuda::{
    fri_workspace_requirements, ArenaLayout, ArenaSlotId, ArenaSlotSpec, CudaExecContext,
    DeviceArena, FriArenaSlotRequirement, FriMerkleTreeSlots, FriWorkspaceConfig,
    FriWorkspaceRequirements, FriWorkspaceSlots, PreparedFriEvaluation, PreparedFriGraph,
};

const INPUT: ArenaSlotId = ArenaSlotId(50_000);
const TWIDDLES: ArenaSlotId = ArenaSlotId(50_001);
const SECURE_COORDINATES: usize = 4;
const M31_MODULUS: u32 = 0x7fff_ffff;

fn workspace_slots(requirements: &FriWorkspaceRequirements) -> FriWorkspaceSlots {
    let mut next = 1u32;
    let mut id = || {
        let result = ArenaSlotId(next);
        next += 1;
        result
    };
    FriWorkspaceSlots {
        evaluation_ping: id(),
        evaluation_pong: id(),
        input_coordinate_ptrs: id(),
        ping_coordinate_ptrs: id(),
        pong_coordinate_ptrs: id(),
        folding_challenges: requirements.rounds.iter().map(|_| id()).collect(),
        trees: requirements
            .trees
            .iter()
            .map(|tree| FriMerkleTreeSlots {
                layers_bottom_up: tree.layers_bottom_up.iter().map(|_| id()).collect(),
            })
            .collect(),
    }
}

fn read_evaluation(arena: &DeviceArena, evaluation: PreparedFriEvaluation) -> Vec<u32> {
    let coordinate_len = 1usize << evaluation.log_size;
    let mut host = vec![0u32; SECURE_COORDINATES * coordinate_len];
    for coordinate in 0..SECURE_COORDINATES {
        unsafe {
            arena
                .context()
                .memcpy_d2h_async(
                    host[coordinate * coordinate_len..].as_mut_ptr().cast(),
                    evaluation
                        .values
                        .as_u32_ptr()
                        .add(coordinate * evaluation.coordinate_stride)
                        .cast(),
                    coordinate_len * core::mem::size_of::<u32>(),
                )
                .unwrap();
        }
    }
    arena.context().sync().unwrap();
    host
}

fn reference_first_tree_root(input: &[u32], log_size: u32) -> Blake2sHash {
    let coordinate_stride = 1usize << log_size;
    let mut layer: Vec<Blake2sHash> = (0..coordinate_stride / 4)
        .map(|leaf| {
            let mut hasher = Blake2sHasherGeneric::<false>::default();
            let words: Vec<BaseField> = (0..4)
                .flat_map(|offset| {
                    (0..SECURE_COORDINATES).map(move |coordinate| {
                        BaseField::from_u32_unchecked(
                            input[coordinate * coordinate_stride + 4 * leaf + offset],
                        )
                    })
                })
                .collect();
            hasher.update_leaf(&words);
            <Blake2sHasherGeneric<false> as MerkleHasherLifted>::finalize(hasher)
        })
        .collect();
    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|children| {
                <Blake2sHasherGeneric<false> as MerkleHasherLifted>::hash_children((
                    children[0],
                    children[1],
                ))
            })
            .collect();
    }
    layer[0]
}

fn reference_fold_round(
    input: &[u32],
    input_log_size: u32,
    twiddles: &[u32],
    fold_step: u32,
    mut alpha: SecureField,
) -> Vec<u32> {
    let input_stride = 1usize << input_log_size;
    let mut values: Vec<SecureField> = (0..input_stride)
        .map(|row| {
            SecureField::from_m31_array(std::array::from_fn(|coordinate| {
                BaseField::from_u32_unchecked(input[coordinate * input_stride + row])
            }))
        })
        .collect();
    let twiddle_words = twiddles.len();
    for fold_index in 0..fold_step {
        let output_len = values.len() / 2;
        let twiddle_offset = if fold_index == 0 {
            twiddle_words - output_len
        } else {
            twiddle_words - values.len()
        };
        let next = (0..output_len)
            .map(|index| {
                let x_inverse = if fold_index == 0 {
                    let k = index >> 2;
                    match index & 3 {
                        0 => BaseField::from_u32_unchecked(twiddles[twiddle_offset + 2 * k + 1]),
                        1 => -BaseField::from_u32_unchecked(twiddles[twiddle_offset + 2 * k + 1]),
                        2 => -BaseField::from_u32_unchecked(twiddles[twiddle_offset + 2 * k]),
                        _ => BaseField::from_u32_unchecked(twiddles[twiddle_offset + 2 * k]),
                    }
                } else {
                    BaseField::from_u32_unchecked(twiddles[twiddle_offset + index])
                };
                let left = values[2 * index];
                let right = values[2 * index + 1];
                let f0 = left + right;
                let f1 = (left - right) * SecureField::from(x_inverse);
                f0 + alpha * f1
            })
            .collect();
        values = next;
        alpha *= alpha;
    }

    let mut coordinate_major = vec![0u32; SECURE_COORDINATES * values.len()];
    for (row, value) in values.into_iter().enumerate() {
        for (coordinate, felt) in value.to_m31_array().into_iter().enumerate() {
            coordinate_major[coordinate * (1usize << (input_log_size - fold_step)) + row] = felt.0;
        }
    }
    coordinate_major
}

/// Eager and captured execution use the exact same prepared methods and stable
/// addresses for both the packed first-layer tree and a multi-fold round.
#[test]
fn eager_and_capture_match() {
    let config = FriWorkspaceConfig {
        fri: FriConfig::new(1, 1, 16, 4),
        circle_log_size: 6,
        twiddle_log_size: 5,
    };
    let requirements = fri_workspace_requirements(config).unwrap();
    let slots = workspace_slots(&requirements);
    let mut requested = requirements.arena_slot_requirements(&slots).unwrap();
    requested.extend([
        FriArenaSlotRequirement {
            id: INPUT,
            len_words: SECURE_COORDINATES * (1usize << config.circle_log_size),
            alignment_words: 1,
        },
        FriArenaSlotRequirement {
            id: TWIDDLES,
            len_words: requirements.twiddle_words,
            alignment_words: 1,
        },
    ]);

    let mut offset = 0usize;
    let mut specs = Vec::with_capacity(requested.len());
    for requirement in requested {
        offset = offset.next_multiple_of(requirement.alignment_words);
        specs.push(ArenaSlotSpec {
            id: requirement.id,
            offset_words: offset,
            len_words: requirement.len_words,
            alignment_words: requirement.alignment_words,
        });
        offset += requirement.len_words;
    }
    let arena = DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset, &specs).unwrap(),
    )
    .unwrap();
    let input = arena.bind(INPUT).unwrap();
    let twiddles = arena.bind(TWIDDLES).unwrap();
    let host_input: Vec<u32> = (0..input.len_words())
        .map(|index| (index as u32 * 17 + 3) % M31_MODULUS)
        .collect();
    let host_twiddles: Vec<u32> = (0..twiddles.len_words())
        .map(|index| (index as u32 * 29 + 1) % M31_MODULUS)
        .collect();
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(
                input.as_void_ptr(),
                host_input.as_ptr().cast(),
                input.len_bytes(),
            )
            .unwrap();
        arena
            .context()
            .memcpy_h2d_async(
                twiddles.as_void_ptr(),
                host_twiddles.as_ptr().cast(),
                twiddles.len_bytes(),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();

    let prepared = PreparedFriGraph::prepare(&arena, config, input, twiddles, &slots).unwrap();

    prepared.launch_first_tree().unwrap();
    let eager_root = prepared.read_tree_root(0).unwrap();
    assert_eq!(
        eager_root,
        reference_first_tree_root(&host_input, config.circle_log_size)
    );
    let capture = arena.context().capture().unwrap();
    prepared.launch_first_tree().unwrap();
    let tree_graph = capture.finish().unwrap();
    tree_graph.launch(arena.context()).unwrap();
    let captured_root = prepared.read_tree_root(0).unwrap();
    assert_eq!(eager_root, captured_root);
    drop(tree_graph);

    let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
    prepared
        .upload_round_challenge_at_transcript_boundary(0, alpha)
        .unwrap();
    prepared.launch_round(0).unwrap();
    let eager_final = read_evaluation(&arena, prepared.final_evaluation());
    assert_eq!(
        eager_final,
        reference_fold_round(
            &host_input,
            config.circle_log_size,
            &host_twiddles,
            config.fri.fold_step,
            alpha,
        )
    );
    let capture = arena.context().capture().unwrap();
    prepared.launch_round(0).unwrap();
    let fold_graph = capture.finish().unwrap();
    fold_graph.launch(arena.context()).unwrap();
    let captured_final = read_evaluation(&arena, prepared.final_evaluation());
    assert_eq!(eager_final, captured_final);
}
