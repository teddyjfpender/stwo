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

    /// VRAM-diet leaf commit: build the lifted leaf layer by LDE'ing `coeffs` one
    /// column-group at a time (bounding peak memory to one group's evaluations +
    /// the per-leaf hash state, instead of ALL columns' evaluations at once).
    ///
    /// Returns `None` if the backend does not implement streaming — the caller
    /// then uses the bulk [`Self::build_leaves`] path. When `Some`, the leaf layer
    /// MUST equal what `build_leaves` produces for the same columns sorted
    /// ascending by size (the blowup is constant, so sorting `coeffs` by
    /// `log_size()` reproduces the bulk sort order). Byte-identical to
    /// `build_leaves(evaluate_polynomials(coeffs))`.
    ///
    /// The default is `None` (bulk path); device backends override.
    fn stream_commit_leaves(
        _coeffs: &[&crate::prover::poly::circle::CircleCoefficients<Self>],
        _log_blowup_factor: u32,
        _twiddles: &crate::prover::poly::twiddles::TwiddleTree<Self>,
        _lifting_log_size: u32,
    ) -> Option<Col<Self, H::Hash>>
    where
        Self: crate::prover::poly::circle::PolyOps,
    {
        None
    }

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

    /// Builds the top `n_levels` layers above `first` (parent of `first`
    /// first, root last). Semantics are EXACTLY `n_levels` chained
    /// [`Self::build_next_layer`] calls — the default does just that; device
    /// backends may override with one fused launch (the per-level launches at
    /// the top of the tree cost launch gaps, not hashing). A scheduling
    /// change only: layer contents are identical either way.
    fn build_top_layers(first: &Col<Self, H::Hash>, n_levels: u32) -> Vec<Col<Self, H::Hash>> {
        assert!(n_levels >= 1 && first.len() >> n_levels >= 1);
        let mut out = Vec::with_capacity(n_levels as usize);
        let mut current = Self::build_next_layer(first);
        for _ in 1..n_levels {
            let next = Self::build_next_layer(&current);
            out.push(current);
            current = next;
        }
        out.push(current);
        out
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
