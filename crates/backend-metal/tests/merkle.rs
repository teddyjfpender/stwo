use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::prover::vcs_lifted::prover::MerkleProverLifted;
use stwo_backend_metal::{BaseFieldVec, MetalBackend};
use stwo_backend_metal_sys::metal::{metal_runtime_support, MetalRuntimeSupport};

fn require_metal() -> bool {
    if metal_runtime_support() == MetalRuntimeSupport::Available {
        return true;
    }
    eprintln!("skipping Metal Merkle differential: Metal runtime unavailable");
    false
}

#[test]
fn pruned_decommit_matches_full_tree() {
    if !require_metal() {
        return;
    }

    let max_log_size = 7u32;
    let lifting_log_size = max_log_size + 1;
    let host_columns = (2..=max_log_size)
        .map(|i| {
            (0..1u32 << i)
                .map(|j| BaseField::from_u32_unchecked(j * i + 11))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let metal_columns = host_columns
        .into_iter()
        .map(BaseFieldVec::from_vec)
        .collect::<Vec<_>>();
    let column_refs = metal_columns.iter().collect_vec();

    let full = MerkleProverLifted::<MetalBackend, Blake2sMerkleHasher>::commit(
        column_refs.clone(),
        lifting_log_size,
        0,
    );
    let pruned = MerkleProverLifted::<MetalBackend, Blake2sMerkleHasher>::commit_pruned(
        column_refs.clone(),
        lifting_log_size,
    );

    assert_eq!(full.root(), pruned.root());
    assert_eq!(full.log_size(), pruned.log_size());
    assert!(pruned.layers.len() < full.layers.len());

    let queries = vec![0, 5, 6, 7, 100, 201, 255];
    let (values_full, decommit_full) = full.decommit(&queries, column_refs.clone());
    let (values_pruned, decommit_pruned) = pruned.decommit(&queries, column_refs);

    assert_eq!(values_full, values_pruned);
    assert_eq!(
        decommit_full.decommitment.hash_witness,
        decommit_pruned.decommitment.hash_witness
    );
    assert_eq!(
        decommit_full.aux.all_node_values,
        decommit_pruned.aux.all_node_values
    );
}
