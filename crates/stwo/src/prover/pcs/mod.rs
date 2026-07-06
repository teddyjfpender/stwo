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
use crate::prover::air::component_prover::{oods_cache_cap, Poly, Trace, WeightsCache};
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
    /// See [`Self::set_stream_lde`].
    pub stream_lde: bool,
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
            stream_lde: false,
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
            stream_lde: false,
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

    /// Streamed-LDE mode (the VRAM diet): each owned tree's full-domain evaluations are
    /// released back to the pool IMMEDIATELY after its Merkle root is computed, instead
    /// of being retained until after the FRI quotients. Requires (and asserts)
    /// `store_polynomials_coefficients`: every later consumer runs from coefficients —
    /// composition via `EvaluationMode::ExtendToEvalDomain` (force it via
    /// `STWO_FORCE_EXTEND_EVAL_MODE=1`), OODS via coefficient `eval_at_point`, FRI
    /// quotients via `QuotientColumnSource::Coeffs` per-group regeneration, and
    /// decommit via the compact-tree machinery (which, with coefficients present, is a
    /// free `Original(coeffs)` wrapper). Peak pool drops by the committed-LDE
    /// retention term (measured 35.2GB → target ≤20GB on SN_PIE_2); cost is one extra
    /// NTT pass per consumer. Values are bit-identical: regeneration is the same
    /// `evaluate_with_twiddles` NTT that produced the committed evaluations, and every
    /// accumulation it feeds is an exact-field sum (associativity ⇒ grouping-invariant).
    pub fn set_stream_lde(&mut self) {
        assert!(
            self.store_polynomials_coefficients,
            "stream_lde requires store_polynomials_coefficients (later phases run from \
             coefficients)"
        );
        self.stream_lde = true;
    }

    /// Releases a just-committed owned tree's full-domain evaluation buffers back to the
    /// pool (streamed-LDE mode). Coefficients and evaluation DOMAIN metadata are
    /// retained; only the value buffers are dropped. Borrowed trees (e.g. the cached
    /// preprocessed tree) are left untouched — their owner decides their lifetime.
    fn release_committed_evals(&mut self) {
        let Some(MaybeOwned::Owned(tree)) = self.trees.0.last_mut() else {
            return;
        };
        for poly in tree.polynomials.iter_mut() {
            debug_assert!(
                poly.coeffs.is_some(),
                "stream_lde: committed poly must retain coefficients"
            );
            // DROP the buffers (freeing them to the backend allocator) rather than
            // giving them back to the BaseColumnPool: the column pool HOARDS returned
            // buffers (still allocated from the device pool's perspective), while the
            // streamed transients (per-component composition, per-group quotients,
            // decommit regeneration) allocate fresh device memory — measured as
            // 45.3GB used-high vs the 35.2GB baseline, i.e. double-booking. A real
            // free makes the memory reusable by every later transient.
            let values = std::mem::replace(&mut poly.evals.values, Col::<B, BaseField>::zeros(0));
            drop(values);
        }
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
        if self.stream_lde {
            self.release_committed_evals();
        }
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
        if self.stream_lde {
            // Owned externally built trees (e.g. a freshly generated preprocessed tree)
            // release their evaluations too; borrowed (cached) trees are untouched.
            self.release_committed_evals();
        }
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
    ) -> WeightsCache<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        // Bounded mode: skip the eager pre-build entirely and hand back an empty,
        // capacity-bounded LRU that computes weights lazily on demand during OODS
        // evaluation. This caps peak (device) memory at ~`cap` weight columns instead
        // of one per distinct (log_size, point) pair. Value-identical by construction
        // (see `oods_cache_cap`); this is the kill-switch-gated diet path.
        if let Some(cap) = oods_cache_cap() {
            return WeightsCache::new_bounded(cap);
        }

        // Default (unbounded): eager, parallel, deduped pre-build — unchanged.
        let WeightsCache::Unbounded(weights_dashmap) = WeightsCache::<B>::new_unbounded() else {
            unreachable!("new_unbounded constructs the Unbounded variant");
        };

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

        WeightsCache::Unbounded(weights_dashmap)
    }

    #[allow(unused_assignments)] // the final `pvt!` timer reset is intentionally unread
    pub fn prove_values(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        // Subscriber-independent bisection timers (STWO_PVT=1): the bench trace
        // subscriber only records spans with recognized `class` values, so new
        // spans are dropped. These eprintln deltas pin the streamed-LDE cost.
        let pvt_on = std::env::var("STWO_PVT").as_deref() == Ok("1");
        #[allow(unused_assignments)]
        let mut pvt_t = std::time::Instant::now();
        macro_rules! pvt {
            ($l:expr) => {
                if pvt_on {
                    eprintln!("PVT {} {:.3}", $l, pvt_t.elapsed().as_secs_f64());
                    pvt_t = std::time::Instant::now();
                }
            };
        }
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

        let samples: TreeVec<Vec<Vec<PointSample>>> = if let Some(cache) = &weights_hash_map {
            // Barycentric mode: group every (tree, column, point-slot) evaluation by
            // (log_size, folded point) — each group shares one weights column — and
            // evaluate each group's columns in ONE backend call
            // ([`PolyOps::barycentric_eval_columns_at_point`]). Values are identical
            // to the per-column path (exact field sums); device backends collapse a
            // launch+sync round trip per column into one launch pair per group.
            let polys = self.polynomials();
            let mut groups: std::collections::HashMap<
                (u32, CirclePoint<SecureField>),
                Vec<(usize, usize, usize)>,
            > = std::collections::HashMap::new();
            for (t, (tree_polys, tree_points)) in
                polys.0.iter().zip(sampled_points.0.iter()).enumerate()
            {
                for (c, (poly, points)) in tree_polys.iter().zip(tree_points.iter()).enumerate() {
                    let log_size = poly.evals.domain.log_size();
                    for (k, &point) in points.iter().enumerate() {
                        let folded = point.repeated_double(lifting_log_size - log_size);
                        groups
                            .entry((log_size, folded))
                            .or_default()
                            .push((t, c, k));
                    }
                }
            }
            let mut out: TreeVec<Vec<Vec<PointSample>>> = TreeVec(
                sampled_points
                    .0
                    .iter()
                    .map(|tree| {
                        tree.iter()
                            .map(|pts| {
                                pts.iter()
                                    .map(|&point| PointSample {
                                        point,
                                        value: SecureField::default(),
                                    })
                                    .collect_vec()
                            })
                            .collect_vec()
                    })
                    .collect_vec(),
            );
            for ((log_size, folded), entries) in groups {
                let evals_refs: Vec<&CircleEvaluation<B, BaseField, BitReversedOrder>> = entries
                    .iter()
                    .map(|&(t, c, _)| &polys.0[t][c].evals)
                    .collect_vec();
                let values = cache.with_weights(
                    (log_size, folded),
                    || {
                        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                            CanonicCoset::new(log_size),
                            folded,
                        )
                    },
                    |weights| B::barycentric_eval_columns_at_point(&evals_refs, weights),
                );
                for (&(t, c, k), value) in entries.iter().zip(values) {
                    out.0[t][c][k].value = value;
                }
            }
            out
        } else {
            // Coefficients mode: the per-column path (Horner on stored coefficients).
            #[cfg(not(feature = "parallel"))]
            {
                self.polynomials()
                    .zip_cols(&sampled_points)
                    .map_cols(eval_at_points)
            }
            #[cfg(feature = "parallel")]
            {
                self.polynomials()
                    .zip_cols(&sampled_points)
                    .par_map_cols(eval_at_points)
            }
        };

        span.exit();
        // The barycentric weights are only needed for the out-of-domain evaluations above.
        // Each entry is a full eval-domain-sized secure-field column, so dropping the map now
        // (instead of at the end of the function) significantly reduces peak memory during FRI.
        drop(weights_hash_map);
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());
        pvt!("oods");

        // Compute oods quotients for boundary constraints on the sampled points.
        // Streamed-LDE mode: the committed evaluations were released at commit time;
        // feed the quotient computation coefficient sources instead — each log-size
        // group is regenerated transiently inside (bit-exact, grouping-invariant
        // exact-field accumulation; see `QuotientColumnSource`).
        let quotients = if self.stream_lde {
            let sources: TreeVec<Vec<quotient_ops::QuotientColumnSource<'_, B>>> = TreeVec(
                self.trees
                    .as_ref()
                    .0
                    .iter()
                    .map(|tree| {
                        tree.polynomials
                            .iter()
                            .map(|poly| match &poly.coeffs {
                                Some(coeffs) => quotient_ops::QuotientColumnSource::Coeffs(
                                    coeffs,
                                    poly.evals.domain,
                                ),
                                // Borrowed/cached trees keep their evaluations resident.
                                None => quotient_ops::QuotientColumnSource::Eval(&poly.evals),
                            })
                            .collect()
                    })
                    .collect(),
            );
            quotient_ops::compute_fri_quotients_streamed(
                sources,
                &samples,
                channel.draw_secure_felt(),
                lifting_log_size,
                self.twiddles,
                self.config.fri_config.log_blowup_factor,
            )
        } else {
            let columns = self.evaluations();
            print_column_size_histogram::<B, MC>(&columns);
            compute_fri_quotients(
                &columns,
                &samples,
                channel.draw_secure_felt(),
                lifting_log_size,
                self.twiddles,
                self.config.fri_config.log_blowup_factor,
            )
        };

        // In low-memory mode, the full column evaluations have now served their last bulk
        // consumer (the FRI quotients above): compact each owned tree's columns. They are
        // regenerated transiently — and bit-exactly — at decommit time.
        // Streamed-LDE mode routes decommit through the same compact machinery: the
        // evaluations were released at commit, so the normal gather path would read
        // empty buffers; with coefficients present, compaction is a free
        // `Original(coeffs)` wrapper and `decommit_compact_tree` regenerates
        // bit-exactly per queried column (the give_back of the already-empty eval
        // buffers inside `compact_tree_columns` is a harmless no-op).
        let mut compact_trees: Vec<Option<CompactTreeColumns<B>>> = if self.low_memory
            || self.stream_lde
        {
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

        pvt!("quotients+compaction");
        // Run FRI commitment phase on the oods quotients.
        let span_fc = span!(Level::INFO, "FRI commit", class = "FriCommit").entered();
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);
        span_fc.exit();
        pvt!("fri_commit");

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let span_fd = span!(Level::INFO, "FRI decommit", class = "FriDecommit").entered();
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        span_fd.exit();
        pvt!("fri_decommit");
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
        let span_td = span!(Level::INFO, "Trees decommit", class = "TreesDecommit").entered();
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
        span_td.exit();
        pvt!("trees_decommit");

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
        // VRAM diet (STWO_CUDA_STREAM_LEAF_COMMIT): build the leaf layer by LDE'ing
        // the base columns one group at a time (never all evaluations resident),
        // then retain coefficients + release evaluations — the stream_lde state,
        // so later phases regenerate from coefficients. Requires
        // `store_polynomials_coefficients`; falls through to the bulk path if the
        // backend returns `None`. Byte-identical (the backend's streaming tests +
        // whole-proof gate cover it).
        if store_polynomials_coefficients
            && !polynomials.is_empty()
            && std::env::var("STWO_CUDA_STREAM_LEAF_COMMIT").as_deref() == Ok("1")
        {
            let max_log = polynomials.iter().map(|p| p.log_size()).max().unwrap();
            let lifting = lifting_log_size.unwrap_or(max_log + log_blowup_factor);
            // Leaf-hash order = columns sorted ascending by size (blowup constant, so
            // sorting coefficients by log_size reproduces the bulk `build_leaves` sort).
            // Sort REFERENCES — no coefficient clone (that would defeat the diet).
            let mut sorted: Vec<&CircleCoefficients<B>> = polynomials.iter().collect();
            sorted.sort_by_key(|c| c.log_size());
            if let Some(leaves) =
                B::stream_commit_leaves(&sorted, log_blowup_factor, twiddles, lifting)
            {
                let tree = MerkleProverLifted::commit_pruned_from_leaves(leaves, lifting);
                // Retain coefficients, release evaluations (domain kept, values empty).
                let polynomials = polynomials
                    .into_iter()
                    .map(|coeffs| {
                        let domain = CanonicCoset::new(coeffs.log_size() + log_blowup_factor)
                            .circle_domain();
                        // Evals released (empty, domain kept) — regenerated from the
                        // retained coefficients downstream (stream_lde state).
                        let evals = CircleEvaluation::new_released(domain);
                        Poly::new(Some(coeffs), evals)
                    })
                    .collect();
                return CommitmentTreeProver {
                    polynomials,
                    commitment: tree,
                };
            }
        }

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
                // (Streamed-LDE mode already released it at commit — len 0 — and an
                // empty buffer must not enter the pool's freelist.)
                if !poly.evals.values.is_empty() {
                    pool.give_back(domain.log_size(), poly.evals.values);
                }
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

    // Regenerate the released evaluations in ONE BATCHED pass (not per-column):
    // reconstruct each column's full-degree coefficients, then hand the whole set to
    // `evaluate_polynomials`, which groups by log size and runs one batched NTT per
    // group. The prior per-column `evaluate_into` issued hundreds of separate
    // single-column NTT launches — the dominant cost of the streamed-LDE decommit
    // (measured ~20s of the ~21s Prove STARKs span on SN_PIE_2). Batching is
    // byte-identical (same NTT, same values); it only changes launch structure.
    // Column order is preserved by `evaluate_polynomials` (wraps results in input
    // order), so the positional (column ↔ log_size ↔ gather) correspondence holds.
    let domains: Vec<CircleDomain> = compact.columns.iter().map(|(_, d)| *d).collect();
    let coeffs: ColumnVec<CircleCoefficients<B>> = compact
        .columns
        .into_iter()
        .map(|(column, _)| match column {
            CompactColumn::Original(coeffs) | CompactColumn::Full(coeffs) => coeffs,
            CompactColumn::Half(left) => {
                // Upper coefficient half was verified all-zero at compaction; restore it.
                let zeros = CircleCoefficients::new(Col::<B, BaseField>::zeros(left.coeffs.len()));
                B::join_at_mid(left, zeros)
            }
        })
        .collect();
    // Blowup 0: each column's coefficients already carry its full (LDE) log size, so
    // it evaluates back onto its committed domain — identical to the prior
    // `evaluate_into(coeffs, domain, ..)`.
    let polys = B::evaluate_polynomials(coeffs, 0, twiddles, false, pool);

    // Pair sequentially (rayon can't zip a parallel iterator with a plain Vec), then
    // parallelize over the tuples.
    let paired: Vec<(Poly<B>, CircleDomain)> = polys.into_iter().zip(domains).collect();
    #[cfg(not(feature = "parallel"))]
    let iter = paired.into_iter();
    #[cfg(feature = "parallel")]
    let iter = paired.into_par_iter();

    let (log_sizes, rows): (Vec<u32>, Vec<HashMap<usize, BaseField>>) = iter
        .map(|(poly, domain)| {
            let log_size = domain.log_size();
            let shift = lifting_log_size - log_size;
            let mut gathered = HashMap::new();
            for pos in query_positions.iter().chain(leaf_indices.iter()) {
                let row = (pos >> (shift + 1) << 1) + (pos & 1);
                // Gather raw stored representations: leaf-hash recomputation must reproduce
                // the exact committed bytes.
                gathered
                    .entry(row)
                    .or_insert_with(|| poly.evals.values.at_unreduced(row));
            }
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
