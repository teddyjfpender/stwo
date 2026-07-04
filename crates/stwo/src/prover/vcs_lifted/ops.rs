use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use crate::prover::backend::{Col, Column, ColumnOps};

/// Trait for performing Merkle operations on a commitment scheme.
pub trait MerkleOpsLifted<H: MerkleHasherLifted>:
    ColumnOps<BaseField> + ColumnOps<H::Hash> + PackLeavesOps + for<'de> Deserialize<'de> + Serialize
{
    /// Computes the leaves of the lifted Merkle commitment.
    fn build_leaves(columns: &[&Col<Self, BaseField>], lifting_log_size: u32)
        -> Col<Self, H::Hash>;

    /// Given a layer of hashes as input, computes a new layer by hashing pairs
    /// of adjacent elements of the input, as in a standard Merkle tree.
    fn build_next_layer(prev_layer: &Col<Self, H::Hash>) -> Col<Self, H::Hash>;

    /// Reads `pairs = [(layer_index_into `layers`, hash_index)]` in one batch.
    /// Semantics are EXACTLY `layers[l].at(i)` per pair, in order — the default
    /// does just that; device backends override with a single gather + one D2H
    /// instead of a synchronous 32-byte round-trip per node (the decommit walk
    /// issues thousands of these).
    fn batch_layer_reads(layers: &[&Col<Self, H::Hash>], pairs: &[(u32, u32)]) -> Vec<H::Hash> {
        pairs
            .iter()
            .map(|&(l, i)| layers[l as usize].at(i as usize))
            .collect()
    }
}

pub trait PackLeavesOps: ColumnOps<BaseField> {
    /// Given a column of QM31s (represented as 4 columns of M31s), reshapes it into 4 columns of
    /// QM31s (represented as 16 columns of M31s). Denoting the input column as [v₀, v₁, v₂, v₃,
    /// ...] where vᵢ ∈ QM31, the output is [[v₀, v₄, v₈, ...], [v₁, v₅, v₉, ...], [v₂, v₆, v₁₀,
    /// ...], [v₃, v₇, v₁₁, ...]].
    fn pack_leaves_input(
        values: &[&Col<Self, BaseField>; SECURE_EXTENSION_DEGREE],
    ) -> [Col<Self, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE];
}
