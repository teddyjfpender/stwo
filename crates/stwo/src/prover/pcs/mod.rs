use hashbrown::HashMap;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tracing::{info, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::MaybeOwned;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::ExtendedMerkleDecommitmentLifted;
use crate::core::ColumnVec;
use crate::prover::air::component_prover::{Poly, Trace, WeightsHashMap};
use crate::prover::backend::{Backend, BackendForChannel, Col, Column};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::pcs::quotient_ops::compute_fri_quotients;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::prover::{GatheredColumns, MerkleProverLifted};

pub mod quotient_ops;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<MaybeOwned<'a, CommitmentTreeProver<B, MC>>>,
    pub config: PcsConfig,
    pub twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    /// See [`Self::set_low_memory`].
    pub low_memory: bool,
    /// Pre-allocated base field column pool for polynomial evaluation during commit.
    pub base_column_pool: MaybeOwned<'a, BaseColumnPool<B>>,
}
impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    /// Creates a new empty commitment scheme prover with the given configuration and twiddles. The
    /// commitment scheme does not store the polynomials coefficients by default.
    pub fn new(config: PcsConfig, twiddles: &'a TwiddleTree<B>) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            low_memory: false,
            base_column_pool: MaybeOwned::Owned(BaseColumnPool::new()),
        }
    }

    pub fn with_memory_pool(
        config: PcsConfig,
        twiddles: &'a TwiddleTree<B>,
        base_column_pool: &'a BaseColumnPool<B>,
    ) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            low_memory: false,
            base_column_pool: MaybeOwned::Borrowed(base_column_pool),
        }
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Enables low-memory mode.
    ///
    /// Once the committed column evaluations have served their last bulk consumer (the FRI
    /// quotients), each owned tree's columns are compacted to at most half their size by
    /// interpolating in place and dropping the (verified all-zero) upper coefficient half. At
    /// decommit time, the evaluations are regenerated transiently, column by column, via the
    /// bit-exact inverse of that interpolation, so the produced proof is identical.
    ///
    /// Trades one inverse FFT per column at compaction plus one FFT per column at
    /// decommitment for a significantly lower peak memory between the FRI phase and the end
    /// of proving.
    ///
    /// NOTE: when the original coefficients are not stored, regeneration runs a different
    /// (same-size) FFT schedule than the commit-time (subdomain-decomposed) evaluation. The
    /// two agree as field elements; bit-equality of the regenerated raw values additionally
    /// relies on FFT outputs being canonically represented (a field zero is stored as 0, not
    /// `P`), which holds for the current kernels and is exercised by the proof-equality tests.
    /// With `set_store_polynomials_coefficients`, regeneration replays the commit-time
    /// computation exactly and carries no such dependency.
    pub const fn set_low_memory(&mut self) {
        self.low_memory = true;
    }

    /// Evaluates the given polynomials, commits them into a Merkle tree, mixes the root into
    /// the channel, and appends the resulting tree to the scheme.
    fn commit(&mut self, polynomials: ColumnVec<CircleCoefficients<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let tree = CommitmentTreeProver::new(
            polynomials,
            self.config.fri_config.log_blowup_factor,
            self.twiddles,
            self.store_polynomials_coefficients,
            self.config.lifting_log_size,
            &self.base_column_pool,
        );
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(MaybeOwned::Owned(tree));
    }

    /// Appends an externally constructed [`CommitmentTreeProver`] to the scheme and mixes its
    /// Merkle root into the channel. Accepts both owned and borrowed trees.
    pub fn commit_tree(
        &mut self,
        tree: MaybeOwned<'a, CommitmentTreeProver<B, MC>>,
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(tree);
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder {
            tree_index: self.trees.len(),
            commitment_scheme: self,
            polys: Vec::default(),
        }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasherLifted>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    pub fn evaluations(
        &self,
    ) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().map(|poly| &poly.evals).collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        let polys = self.polynomials();
        Trace { polys }
    }

    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        let weights_dashmap = WeightsHashMap::<B>::new();

        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                let compute_weights = |(log_size, point): (u32, CirclePoint<SecureField>)| {
                    weights_dashmap.entry((log_size, point)).or_insert_with(|| {
                        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                            CanonicCoset::new(log_size),
                            point,
                        )
                    });
                };

                let log_size = poly.evals.domain.log_size();
                // For each sample point, compute the weights needed to evaluate the polynomial at
                // the folded sample point.
                // TODO(Leo): the computation `point.repeated_double(max_log_size - log_size)` is
                // likely repeated a bunch of times in a typical flat air. Consider moving it
                // outside the loop.
                #[cfg(not(feature = "parallel"))]
                points.iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });

                #[cfg(feature = "parallel")]
                points.par_iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });
            });

        weights_dashmap
    }

    pub fn prove_values(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        // Evaluate polynomials on open points.
        let span = span!(
            Level::INFO,
            "Evaluate columns out of domain",
            class = "EvaluateOutOfDomain"
        )
        .entered();

        let lifting_log_size = self.trees.last().unwrap().commitment.log_size();
        let weights_hash_map = if self.store_polynomials_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };

        // Lambda that evaluates a polynomial on a collection of circle points and returns a vector
        // of point samples.
        let eval_at_points = |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(
                        point.repeated_double(lifting_log_size - poly.evals.domain.log_size()),
                        weights_hash_map.as_ref(),
                    ),
                })
                .collect_vec()
        };

        #[cfg(not(feature = "parallel"))]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(eval_at_points);
        #[cfg(feature = "parallel")]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .par_map_cols(eval_at_points);

        span.exit();
        // The barycentric weights are only needed for the out-of-domain evaluations above.
        // Each entry is a full eval-domain-sized secure-field column, so dropping the map now
        // (instead of at the end of the function) significantly reduces peak memory during FRI.
        drop(weights_hash_map);
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        let columns = self.evaluations();
        print_column_size_histogram::<B, MC>(&columns);
        // Compute oods quotients for boundary constraints on the sampled points.
        let quotients = compute_fri_quotients(
            &columns,
            &samples,
            channel.draw_secure_felt(),
            lifting_log_size,
            self.twiddles,
            self.config.fri_config.log_blowup_factor,
        );

        // In low-memory mode, the full column evaluations have now served their last bulk
        // consumer (the FRI quotients above): compact each owned tree's columns. They are
        // regenerated transiently — and bit-exactly — at decommit time.
        let mut compact_trees: Vec<Option<CompactTreeColumns<B>>> = if self.low_memory {
            let _span = span!(Level::INFO, "Eval compaction", class = "EvalCompaction").entered();
            self.trees
                .0
                .iter_mut()
                .map(|tree| match tree {
                    MaybeOwned::Owned(tree) if !tree.polynomials.is_empty() => {
                        Some(compact_tree_columns(
                            std::mem::take(&mut tree.polynomials),
                            self.twiddles,
                            &self.base_column_pool,
                        ))
                    }
                    _ => None,
                })
                .collect()
        } else {
            self.trees.iter().map(|_| None).collect()
        };

        // Run FRI commitment phase on the oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        // Build the query position tree.
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].commitment.log_size(),
        );
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let commitments = self.roots();
        let (queried_values, decommitments, aux): (Vec<_>, Vec<_>, Vec<_>) = self
            .trees
            .as_ref()
            .zip_eq(query_positions_tree)
            .0
            .into_iter()
            .zip(compact_trees.drain(..))
            .map(|((tree, query_positions), compact)| match compact {
                Some(compact) => decommit_compact_tree(
                    tree,
                    compact,
                    query_positions,
                    self.twiddles,
                    &self.base_column_pool,
                ),
                None => tree.decommit(query_positions),
            })
            .map(|(v, x)| (v, x.decommitment, x.aux))
            .multiunzip();

        // Return evaluation buffers to the memory pool for reuse (owned trees only).
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                for poly in tree.polynomials.drain(..) {
                    let log_size = poly.evals.domain.log_size();
                    self.base_column_pool.give_back(log_size, poly.evals.values);
                }
            }
        }

        ExtendedCommitmentSchemeProof {
            proof: CommitmentSchemeProof {
                commitments,
                sampled_values,
                decommitments: TreeVec(decommitments),
                queried_values: TreeVec(queried_values),
                proof_of_work,
                fri_proof: fri_proof.proof,
                config: self.config,
            },
            aux: CommitmentSchemeProofAux {
                unsorted_query_locations,
                trace_decommitment: TreeVec(aux),
                fri: fri_proof.aux,
            },
        }
    }
}

/// Helper struct for aggregating polynomials and evaluations for a commitment tree.
pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CircleCoefficients<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns(columns, self.commitment_scheme.twiddles);
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        self.commitment_scheme.commit(self.polys, channel);
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: MerkleProverLifted<B, MC::H>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            base_column_pool,
        );
        span.exit();

        let _span = span!(Level::INFO, "Merkle").entered();
        let max_log_domain_size = polynomials
            .iter()
            .map(|poly| poly.evals.domain.log_size())
            .max()
            .unwrap_or_default();
        let lifting_log_size = lifting_log_size.unwrap_or(max_log_domain_size);
        // Pruned commit: the bottom tree layers are recomputed from the column evaluations at
        // decommit time instead of being held in memory for the whole proving pipeline.
        let tree = MerkleProverLifted::commit_pruned(
            polynomials
                .iter()
                .map(|poly: &Poly<B>| &poly.evals.values)
                .collect(),
            lifting_log_size,
        );

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    ///
    /// The rows the decommit reads (the queried rows plus the rows of unretained leaves
    /// whose hashes must be recomputed) are gathered up front with one batched
    /// [`Column::gather_unreduced`] per column, and the decommit runs over the sparse
    /// view — the same path the low-memory mode uses. Reading element-by-element during
    /// the walk costs queries x columns individual `at` calls, each of which is a full
    /// device readback on GPU backends; the values and output are identical either way.
    fn decommit(
        &self,
        queries: &[usize],
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<MC::H>,
    ) {
        let lifting_log_size = self.commitment.log_size();
        let leaf_indices = self.commitment.unretained_leaf_indices(queries);

        #[cfg(not(feature = "parallel"))]
        let iter = self.polynomials.iter();
        #[cfg(feature = "parallel")]
        let iter = self.polynomials.par_iter();

        let (log_sizes, rows): (Vec<u32>, Vec<HashMap<usize, BaseField>>) = iter
            .map(|poly| {
                let log_size = poly.evals.domain.log_size();
                let shift = lifting_log_size - log_size;
                // Deduplicated, in deterministic order (BTreeSet), mirroring the row
                // mapping in `decommit_inner`/`decommit_compact_tree`.
                let needed_rows: Vec<usize> = queries
                    .iter()
                    .chain(leaf_indices.iter())
                    .map(|pos| (pos >> (shift + 1) << 1) + (pos & 1))
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let values = poly.evals.values.gather_unreduced(&needed_rows);
                (
                    log_size,
                    needed_rows
                        .into_iter()
                        .zip(values)
                        .collect::<HashMap<_, _>>(),
                )
            })
            .unzip();

        self.commitment
            .decommit_gathered(queries, &GatheredColumns { log_sizes, rows })
    }
}

/// A compacted representation of a committed column in low-memory mode: enough information to
/// regenerate the committed evaluation bit-exactly at decommit time, at half (or less) of the
/// evaluation's memory.
enum CompactColumn<B: Backend> {
    /// The original coefficients the column was committed from (available when
    /// `store_polynomials_coefficients` is set). Regeneration replays the commit-time
    /// evaluation.
    Original(CircleCoefficients<B>),
    /// The lower half of the in-place interpolated coefficients; the upper half was verified
    /// to be all zeros and is reconstructed as such. Regeneration evaluates the rejoined
    /// polynomial on its own domain — the bit-exact inverse of the interpolation.
    Half(CircleCoefficients<B>),
    /// The full interpolated coefficients. Only used in the unexpected case that the upper
    /// coefficient half is not all zeros (i.e. the committed evaluation was not a low-degree
    /// extension); saves no memory but stays correct.
    Full(CircleCoefficients<B>),
}

/// The compacted columns of one commitment tree, in commit order, with their committed
/// evaluation domains.
struct CompactTreeColumns<B: Backend> {
    columns: Vec<(CompactColumn<B>, CircleDomain)>,
}

/// Compacts a tree's columns; see [`CommitmentSchemeProver::set_low_memory`].
fn compact_tree_columns<B: Backend>(
    polynomials: ColumnVec<Poly<B>>,
    twiddles: &TwiddleTree<B>,
    pool: &BaseColumnPool<B>,
) -> CompactTreeColumns<B> {
    #[cfg(not(feature = "parallel"))]
    let iter = polynomials.into_iter();
    #[cfg(feature = "parallel")]
    let iter = polynomials.into_par_iter();

    let columns = iter
        .map(|poly| {
            let domain = poly.evals.domain;
            if let Some(coeffs) = poly.coeffs {
                // The original coefficients are sufficient; recycle the evaluation buffer.
                pool.give_back(domain.log_size(), poly.evals.values);
                return (CompactColumn::Original(coeffs), domain);
            }
            // In-place interpolation: reuses the evaluation buffer.
            let coeffs = poly.evals.interpolate_with_twiddles(twiddles);
            let (mut left, right) = coeffs.split_at_mid();
            let upper_half_is_zero = right.coeffs.to_cpu().iter().all(|v| v.0 == 0);
            if upper_half_is_zero {
                // Actually release the upper half's memory.
                left.coeffs.shrink_to_fit();
                (CompactColumn::Half(left), domain)
            } else {
                (CompactColumn::Full(B::join_at_mid(left, right)), domain)
            }
        })
        .collect();

    CompactTreeColumns { columns }
}

/// Decommits a tree whose column evaluations were compacted: regenerates each column
/// transiently (bit-exactly), gathers only the rows the decommit reads, and decommits from the
/// gathered view.
fn decommit_compact_tree<B: BackendForChannel<MC>, MC: MerkleChannel>(
    tree: &CommitmentTreeProver<B, MC>,
    compact: CompactTreeColumns<B>,
    query_positions: &[usize],
    twiddles: &TwiddleTree<B>,
    pool: &BaseColumnPool<B>,
) -> (
    ColumnVec<Vec<BaseField>>,
    ExtendedMerkleDecommitmentLifted<MC::H>,
) {
    let lifting_log_size = tree.commitment.log_size();
    // The leaves whose hashes the decommit will recompute (unretained bottom tree layers).
    let leaf_indices = tree.commitment.unretained_leaf_indices(query_positions);

    #[cfg(not(feature = "parallel"))]
    let iter = compact.columns.into_iter();
    #[cfg(feature = "parallel")]
    let iter = compact.columns.into_par_iter();

    let (log_sizes, rows): (Vec<u32>, Vec<HashMap<usize, BaseField>>) = iter
        .map(|(column, domain)| {
            let log_size = domain.log_size();
            let shift = lifting_log_size - log_size;
            let buffer = pool.take_or_alloc(log_size);
            let evals = match column {
                CompactColumn::Original(coeffs) | CompactColumn::Full(coeffs) => {
                    B::evaluate_into(&coeffs, domain, twiddles, buffer)
                }
                CompactColumn::Half(left) => {
                    let zeros =
                        CircleCoefficients::new(Col::<B, BaseField>::zeros(left.coeffs.len()));
                    let joined = B::join_at_mid(left, zeros);
                    B::evaluate_into(&joined, domain, twiddles, buffer)
                }
            };
            let mut gathered = HashMap::new();
            for pos in query_positions.iter().chain(leaf_indices.iter()) {
                let row = (pos >> (shift + 1) << 1) + (pos & 1);
                // Gather raw stored representations: leaf-hash recomputation must reproduce
                // the exact committed bytes.
                gathered
                    .entry(row)
                    .or_insert_with(|| evals.values.at_unreduced(row));
            }
            pool.give_back(log_size, evals.values);
            (log_size, gathered)
        })
        .unzip();

    tree.commitment
        .decommit_gathered(query_positions, &GatheredColumns { log_sizes, rows })
}

fn print_column_size_histogram<B: BackendForChannel<MC>, MC: MerkleChannel>(
    columns_per_tree: &TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
) {
    let mut log_size_histogram = HashMap::new();
    for columns in columns_per_tree.iter() {
        for column in columns {
            *log_size_histogram
                .entry(column.domain.log_size())
                .or_insert(0) += 1;
        }
    }
    for (log_size, count) in log_size_histogram {
        info!("Log size {log_size}: {count}");
    }
}
