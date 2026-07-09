//! Allocation-free, explicit-stream CUDA commit island.
//!
//! [`CommitGraphPlan::launch`] is the only execution path: callers invoke it
//! directly for eager execution or between [`CudaExecContext::capture`] and
//! `CudaGraphCapture::finish` for graph execution. That keeps kernel order and
//! parameters identical across both modes.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::collections::BTreeSet;

use stwo_backend_cuda_kernels::raw::{self, Blake2sHash};

use super::exec_context::{check_cuda, ArenaSlice, ArenaSlotId, CudaExecContext, CudaRuntimeError};

const HASH_WORDS: usize = core::mem::size_of::<Blake2sHash>() / core::mem::size_of::<u32>();
const MAX_FUSED_TAIL_HASHES: u32 = 4096;

/// One same-size LDE batch. Both pointer tables are DEVICE tables in the same
/// canonical order. `coefficient_sizes` gives each source's exact length;
/// outputs have `2 * eval_domain_size` words. One explicit-stream kernel stages
/// coefficients plus a zero tail before the in-place N2B transform.
#[derive(Clone, Copy, Debug)]
pub struct CommitLdeBatch {
    pub coefficient_ptrs: ArenaSlice,
    /// DEVICE table containing the exact source length of every coefficient
    /// column. Sources may be shorter than `eval_domain_size` when blowup > 1;
    /// the staging kernel zero-fills the remainder before the N2B transform.
    pub coefficient_sizes: ArenaSlice,
    pub column_ptrs: ArenaSlice,
    pub column_count: u32,
    pub log_n: u32,
    pub twiddles: ArenaSlice,
    pub twiddles_size: u32,
    pub eval_domain_size: u32,
}

/// Canonically ordered leaf-column group. `column_ptrs` and
/// `column_log_sizes` are DEVICE tables in the exact leaf byte order. Batches
/// may partition this group by log size, but their total column count must equal
/// `column_count` before the leaf kernel consumes the canonical group table.
#[derive(Clone, Debug)]
pub struct CommitLeafGroup {
    pub first_column: u32,
    pub column_count: u32,
    pub column_ptrs: ArenaSlice,
    pub column_log_sizes: ArenaSlice,
    pub lde_batches: Vec<CommitLdeBatch>,
}

/// Fused top-of-tree tail. `level_ptrs` is a DEVICE pointer table containing
/// `level_outputs` in the same order; output `i` holds half as many hashes as
/// output `i-1` (or the tail input for `i=0`).
#[derive(Clone, Debug)]
pub struct CommitTailPlan {
    pub level_ptrs: ArenaSlice,
    pub level_outputs: Vec<ArenaSlice>,
}

/// Observable launch topology, used by local plan tests and benchmark traces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitLaunchKind {
    LeafInit {
        hashes: u32,
    },
    Lde {
        group: u32,
        batch: u32,
        columns: u32,
        log_n: u32,
    },
    LeafUpdate {
        group: u32,
        first_column: u32,
        columns: u32,
    },
    LeafFinalize {
        group: u32,
        first_column: u32,
        columns: u32,
    },
    InteriorLayer {
        level: u32,
        output_hashes: u32,
    },
    FusedTail {
        first_hashes: u32,
        levels: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommitGraphError {
    InvalidLiftingLogSize(u32),
    NoLeafGroups,
    EmptyLeafGroup(u32),
    NonCanonicalGroupStart {
        group: u32,
        expected: u32,
        actual: u32,
    },
    InvalidUpdateWidth {
        group: u32,
        columns: u32,
    },
    InvalidFinalWidth {
        group: u32,
        columns: u32,
    },
    EmptyLdeBatches(u32),
    LdeColumnCountMismatch {
        group: u32,
        expected: u32,
        actual: u32,
    },
    InvalidLdeGeometry {
        group: u32,
        batch: u32,
    },
    BufferTooSmall {
        role: &'static str,
        required_words: usize,
        actual_words: usize,
    },
    MisalignedPointerTable(&'static str),
    MisalignedHashBuffer(&'static str),
    AliasedArenaSlot(ArenaSlotId),
    InPlaceInteriorLayer(ArenaSlotId),
    PrematureArenaSlotReuse(ArenaSlotId),
    InvalidUnretainedBottomLayers {
        lifting_log_size: u32,
        unretained: u32,
    },
    ContextMismatch,
    TooManyColumns,
    InteriorPastRoot(u32),
    EmptyTail,
    TailTooWide(u32),
    TooManyTailLevels(usize),
    IncompleteTree(u32),
    SizeOverflow,
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for CommitGraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid CUDA commit graph: {self:?}")
    }
}

impl std::error::Error for CommitGraphError {}

impl From<CudaRuntimeError> for CommitGraphError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

#[derive(Clone, Copy, Debug)]
enum CommitLaunch {
    LeafInit {
        size: u32,
        state: ArenaSlice,
    },
    Lde {
        group: u32,
        batch: u32,
        params: CommitLdeBatch,
    },
    LeafUpdate {
        group: u32,
        first_column: u32,
        columns: u32,
        column_ptrs: ArenaSlice,
        column_log_sizes: ArenaSlice,
        lifting_log_size: u32,
        state: ArenaSlice,
    },
    LeafFinalize {
        group: u32,
        first_column: u32,
        columns: u32,
        column_ptrs: ArenaSlice,
        column_log_sizes: ArenaSlice,
        lifting_log_size: u32,
        state: ArenaSlice,
    },
    InteriorLayer {
        level: u32,
        input: ArenaSlice,
        output: ArenaSlice,
        output_hashes: u32,
    },
    FusedTail {
        input: ArenaSlice,
        first_hashes: u32,
        level_ptrs: ArenaSlice,
        levels: u32,
    },
}

impl CommitLaunch {
    fn kind(self) -> CommitLaunchKind {
        match self {
            Self::LeafInit { size, .. } => CommitLaunchKind::LeafInit { hashes: size },
            Self::Lde {
                group,
                batch,
                params,
            } => CommitLaunchKind::Lde {
                group,
                batch,
                columns: params.column_count,
                log_n: params.log_n,
            },
            Self::LeafUpdate {
                group,
                first_column,
                columns,
                ..
            } => CommitLaunchKind::LeafUpdate {
                group,
                first_column,
                columns,
            },
            Self::LeafFinalize {
                group,
                first_column,
                columns,
                ..
            } => CommitLaunchKind::LeafFinalize {
                group,
                first_column,
                columns,
            },
            Self::InteriorLayer {
                level,
                output_hashes,
                ..
            } => CommitLaunchKind::InteriorLayer {
                level,
                output_hashes,
            },
            Self::FusedTail {
                first_hashes,
                levels,
                ..
            } => CommitLaunchKind::FusedTail {
                first_hashes,
                levels,
            },
        }
    }
}

/// Validated, allocation-free launch plan for one lifted Blake2s commitment.
#[derive(Debug)]
pub struct CommitGraphPlan {
    context_token: NonNull<c_void>,
    launches: Vec<CommitLaunch>,
    root: ArenaSlice,
}

impl CommitGraphPlan {
    /// Build and fully validate the launch geometry before capture begins.
    ///
    /// Column pointer tables must already contain the canonical column order;
    /// this constructor never sorts or rewrites them. `interior_outputs` lists
    /// one output buffer per ordinary column-free Merkle level. If `tail` is
    /// present, its outputs finish the tree in one fused launch.
    pub fn new(
        lifting_log_size: u32,
        leaf_state: ArenaSlice,
        leaf_groups: Vec<CommitLeafGroup>,
        interior_outputs: Vec<ArenaSlice>,
        tail: Option<CommitTailPlan>,
    ) -> Result<Self, CommitGraphError> {
        Self::new_pruned(
            lifting_log_size,
            0,
            leaf_state,
            leaf_groups,
            interior_outputs,
            tail,
        )
    }

    /// Build a plan whose bottom `unretained_bottom_layers` (counting the leaf
    /// layer) are scratch. Those bottom outputs may reuse two arena slots in
    /// strict producer/consumer order; every retained and fused-tail output stays
    /// uniquely addressable for later decommitment.
    pub fn new_pruned(
        lifting_log_size: u32,
        unretained_bottom_layers: u32,
        leaf_state: ArenaSlice,
        leaf_groups: Vec<CommitLeafGroup>,
        interior_outputs: Vec<ArenaSlice>,
        tail: Option<CommitTailPlan>,
    ) -> Result<Self, CommitGraphError> {
        if lifting_log_size >= 31 {
            return Err(CommitGraphError::InvalidLiftingLogSize(lifting_log_size));
        }
        if unretained_bottom_layers > lifting_log_size {
            return Err(CommitGraphError::InvalidUnretainedBottomLayers {
                lifting_log_size,
                unretained: unretained_bottom_layers,
            });
        }
        if leaf_groups.is_empty() {
            return Err(CommitGraphError::NoLeafGroups);
        }
        let leaf_size = 1u32 << lifting_log_size;
        require_hash_capacity("leaf_state", leaf_state, leaf_size)?;
        let context_token = leaf_state.context_token();
        let mut hash_slots = BTreeSet::from([leaf_state.id()]);
        let mut retained_slots = BTreeSet::new();
        if unretained_bottom_layers == 0 {
            retained_slots.insert(leaf_state.id());
        }
        let mut input_slots = BTreeSet::new();

        let mut launches = Vec::new();
        launches.push(CommitLaunch::LeafInit {
            size: leaf_size,
            state: leaf_state,
        });

        let mut cols_done = 0u32;
        let last_group = leaf_groups.len() - 1;
        for (group_index, group) in leaf_groups.iter().enumerate() {
            let group_index = group_index as u32;
            require_same_context(context_token, group.column_ptrs)?;
            require_same_context(context_token, group.column_log_sizes)?;
            input_slots.insert(group.column_ptrs.id());
            input_slots.insert(group.column_log_sizes.id());
            if group.column_count == 0 {
                return Err(CommitGraphError::EmptyLeafGroup(group_index));
            }
            if group.first_column != cols_done {
                return Err(CommitGraphError::NonCanonicalGroupStart {
                    group: group_index,
                    expected: cols_done,
                    actual: group.first_column,
                });
            }
            require_pointer_table("leaf_column_ptrs", group.column_ptrs, group.column_count)?;
            require_words(
                "leaf_column_log_sizes",
                group.column_log_sizes,
                group.column_count as usize,
            )?;
            if group.lde_batches.is_empty() {
                return Err(CommitGraphError::EmptyLdeBatches(group_index));
            }

            let mut lde_columns = 0u32;
            for (batch_index, &batch) in group.lde_batches.iter().enumerate() {
                require_same_context(context_token, batch.coefficient_ptrs)?;
                require_same_context(context_token, batch.coefficient_sizes)?;
                require_same_context(context_token, batch.column_ptrs)?;
                require_same_context(context_token, batch.twiddles)?;
                validate_lde_batch(group_index, batch_index as u32, batch)?;
                input_slots.insert(batch.coefficient_ptrs.id());
                input_slots.insert(batch.coefficient_sizes.id());
                input_slots.insert(batch.column_ptrs.id());
                input_slots.insert(batch.twiddles.id());
                lde_columns = lde_columns
                    .checked_add(batch.column_count)
                    .ok_or(CommitGraphError::TooManyColumns)?;
                launches.push(CommitLaunch::Lde {
                    group: group_index,
                    batch: batch_index as u32,
                    params: batch,
                });
            }
            if lde_columns != group.column_count {
                return Err(CommitGraphError::LdeColumnCountMismatch {
                    group: group_index,
                    expected: group.column_count,
                    actual: lde_columns,
                });
            }

            let is_final = group_index as usize == last_group;
            if is_final {
                if group.column_count > 16 {
                    return Err(CommitGraphError::InvalidFinalWidth {
                        group: group_index,
                        columns: group.column_count,
                    });
                }
                launches.push(CommitLaunch::LeafFinalize {
                    group: group_index,
                    first_column: cols_done,
                    columns: group.column_count,
                    column_ptrs: group.column_ptrs,
                    column_log_sizes: group.column_log_sizes,
                    lifting_log_size,
                    state: leaf_state,
                });
            } else {
                if group.column_count % 16 != 0 {
                    return Err(CommitGraphError::InvalidUpdateWidth {
                        group: group_index,
                        columns: group.column_count,
                    });
                }
                launches.push(CommitLaunch::LeafUpdate {
                    group: group_index,
                    first_column: cols_done,
                    columns: group.column_count,
                    column_ptrs: group.column_ptrs,
                    column_log_sizes: group.column_log_sizes,
                    lifting_log_size,
                    state: leaf_state,
                });
            }
            cols_done = cols_done
                .checked_add(group.column_count)
                .ok_or(CommitGraphError::TooManyColumns)?;
        }

        let mut current = leaf_state;
        let mut current_hashes = leaf_size;
        for (level, &output) in interior_outputs.iter().enumerate() {
            require_same_context(context_token, output)?;
            if current_hashes < 2 {
                return Err(CommitGraphError::InteriorPastRoot(level as u32));
            }
            if output.id() == current.id() {
                return Err(CommitGraphError::InPlaceInteriorLayer(output.id()));
            }
            if retained_slots.contains(&output.id()) {
                return Err(CommitGraphError::PrematureArenaSlotReuse(output.id()));
            }
            hash_slots.insert(output.id());
            let output_hashes = current_hashes / 2;
            require_hash_capacity("interior_output", output, output_hashes)?;
            launches.push(CommitLaunch::InteriorLayer {
                level: level as u32,
                input: current,
                output,
                output_hashes,
            });
            current = output;
            current_hashes = output_hashes;
            // `level + 1` is this output's distance above the leaves. Outputs
            // beyond the pruned prefix must remain live through decommitment.
            if (level as u32 + 1) >= unretained_bottom_layers {
                retained_slots.insert(output.id());
            }
        }

        if let Some(tail) = tail {
            if tail.level_outputs.is_empty() {
                return Err(CommitGraphError::EmptyTail);
            }
            if current_hashes > MAX_FUSED_TAIL_HASHES {
                return Err(CommitGraphError::TailTooWide(current_hashes));
            }
            if tail.level_outputs.len() >= 32 {
                return Err(CommitGraphError::TooManyTailLevels(
                    tail.level_outputs.len(),
                ));
            }
            require_same_context(context_token, tail.level_ptrs)?;
            input_slots.insert(tail.level_ptrs.id());
            require_pointer_table(
                "tail_level_ptrs",
                tail.level_ptrs,
                tail.level_outputs.len() as u32,
            )?;
            let first_hashes = current_hashes;
            let tail_input = current;
            for &output in &tail.level_outputs {
                require_same_context(context_token, output)?;
                if current_hashes < 2 {
                    return Err(CommitGraphError::InteriorPastRoot(launches.len() as u32));
                }
                if output.id() == current.id() {
                    return Err(CommitGraphError::InPlaceInteriorLayer(output.id()));
                }
                if retained_slots.contains(&output.id()) {
                    return Err(CommitGraphError::PrematureArenaSlotReuse(output.id()));
                }
                hash_slots.insert(output.id());
                retained_slots.insert(output.id());
                current_hashes /= 2;
                require_hash_capacity("tail_output", output, current_hashes)?;
                current = output;
            }
            launches.push(CommitLaunch::FusedTail {
                input: tail_input,
                first_hashes,
                level_ptrs: tail.level_ptrs,
                levels: tail.level_outputs.len() as u32,
            });
        }

        if let Some(id) = hash_slots.intersection(&input_slots).next() {
            return Err(CommitGraphError::AliasedArenaSlot(*id));
        }

        if current_hashes != 1 {
            return Err(CommitGraphError::IncompleteTree(current_hashes));
        }
        Ok(Self {
            context_token,
            launches,
            root: current,
        })
    }

    /// Root hash slot produced by this plan.
    pub fn root(&self) -> ArenaSlice {
        self.root
    }

    /// Exact eager/capture launch topology, without allocating.
    pub fn launch_sequence(&self) -> impl ExactSizeIterator<Item = CommitLaunchKind> + '_ {
        self.launches.iter().copied().map(CommitLaunch::kind)
    }

    /// Enqueue the complete allocation-free commit sequence on `context`.
    /// Calling this inside capture and calling it eagerly execute identical code.
    pub fn launch(&self, context: &CudaExecContext) -> Result<(), CommitGraphError> {
        if context.identity_token() != self.context_token {
            return Err(CommitGraphError::ContextMismatch);
        }
        let stream = context.stream_raw().as_ptr();
        for &launch in &self.launches {
            let (operation, code) = unsafe {
                match launch {
                    CommitLaunch::LeafInit { size, state } => (
                        "commit_leaf_init",
                        raw::stwo_blake2s_leaf_init_on(size, state.as_u32_ptr().cast(), stream),
                    ),
                    CommitLaunch::Lde { params, .. } => (
                        "commit_lde_n2b",
                        raw::stwo_lde_n2b_columns_on(
                            params.coefficient_ptrs.as_u32_ptr().cast(),
                            params.coefficient_sizes.as_u32_ptr(),
                            params.column_ptrs.as_u32_ptr().cast(),
                            params.log_n,
                            params.column_count,
                            params.twiddles.as_u32_ptr(),
                            params.twiddles_size,
                            params.eval_domain_size,
                            stream,
                        ),
                    ),
                    CommitLaunch::LeafUpdate {
                        columns,
                        column_ptrs,
                        column_log_sizes,
                        lifting_log_size,
                        first_column,
                        state,
                        ..
                    } => (
                        "commit_leaf_update",
                        raw::stwo_blake2s_leaf_update_on(
                            1u32 << lifting_log_size,
                            columns,
                            column_ptrs.as_u32_ptr().cast(),
                            column_log_sizes.as_u32_ptr(),
                            lifting_log_size,
                            first_column,
                            state.as_u32_ptr().cast(),
                            stream,
                        ),
                    ),
                    CommitLaunch::LeafFinalize {
                        columns,
                        column_ptrs,
                        column_log_sizes,
                        lifting_log_size,
                        first_column,
                        state,
                        ..
                    } => (
                        "commit_leaf_finalize",
                        raw::stwo_blake2s_leaf_finalize_on(
                            1u32 << lifting_log_size,
                            columns,
                            column_ptrs.as_u32_ptr().cast(),
                            column_log_sizes.as_u32_ptr(),
                            lifting_log_size,
                            first_column,
                            state.as_u32_ptr().cast(),
                            stream,
                        ),
                    ),
                    CommitLaunch::InteriorLayer {
                        input,
                        output,
                        output_hashes,
                        ..
                    } => (
                        "commit_interior_layer",
                        raw::stwo_blake2s_layer_on(
                            input.as_u32_ptr().cast(),
                            output_hashes,
                            output.as_u32_ptr().cast(),
                            stream,
                        ),
                    ),
                    CommitLaunch::FusedTail {
                        input,
                        first_hashes,
                        level_ptrs,
                        levels,
                    } => (
                        "commit_fused_tail",
                        raw::stwo_blake2s_tail_on(
                            input.as_u32_ptr().cast(),
                            first_hashes,
                            level_ptrs.as_u32_ptr().cast(),
                            levels,
                            stream,
                        ),
                    ),
                }
            };
            check_cuda(operation, code)?;
        }
        Ok(())
    }
}

fn require_same_context(
    expected: NonNull<c_void>,
    slice: ArenaSlice,
) -> Result<(), CommitGraphError> {
    if slice.context_token() == expected {
        Ok(())
    } else {
        Err(CommitGraphError::ContextMismatch)
    }
}

fn require_words(
    role: &'static str,
    slice: ArenaSlice,
    required_words: usize,
) -> Result<(), CommitGraphError> {
    if slice.len_words() < required_words {
        Err(CommitGraphError::BufferTooSmall {
            role,
            required_words,
            actual_words: slice.len_words(),
        })
    } else {
        Ok(())
    }
}

fn require_hash_capacity(
    role: &'static str,
    slice: ArenaSlice,
    hashes: u32,
) -> Result<(), CommitGraphError> {
    if (slice.as_u32_ptr() as usize) % core::mem::align_of::<Blake2sHash>() != 0 {
        return Err(CommitGraphError::MisalignedHashBuffer(role));
    }
    let words = (hashes as usize)
        .checked_mul(HASH_WORDS)
        .ok_or(CommitGraphError::SizeOverflow)?;
    require_words(role, slice, words)
}

fn require_pointer_table(
    role: &'static str,
    slice: ArenaSlice,
    pointers: u32,
) -> Result<(), CommitGraphError> {
    if (slice.as_u32_ptr() as usize) % core::mem::align_of::<*mut u32>() != 0 {
        return Err(CommitGraphError::MisalignedPointerTable(role));
    }
    let bytes = (pointers as usize)
        .checked_mul(core::mem::size_of::<*mut u32>())
        .ok_or(CommitGraphError::SizeOverflow)?;
    let words = bytes
        .checked_add(core::mem::size_of::<u32>() - 1)
        .ok_or(CommitGraphError::SizeOverflow)?
        / core::mem::size_of::<u32>();
    require_words(role, slice, words)
}

fn validate_lde_batch(
    group: u32,
    batch: u32,
    params: CommitLdeBatch,
) -> Result<(), CommitGraphError> {
    if params.column_count == 0
        || !(4..=30).contains(&params.log_n)
        || params.eval_domain_size != (1u32 << (params.log_n - 1))
        || params.twiddles_size < params.eval_domain_size
    {
        return Err(CommitGraphError::InvalidLdeGeometry { group, batch });
    }
    require_pointer_table(
        "lde_coefficient_ptrs",
        params.coefficient_ptrs,
        params.column_count,
    )?;
    require_words(
        "lde_coefficient_sizes",
        params.coefficient_sizes,
        params.column_count as usize,
    )?;
    require_pointer_table("lde_column_ptrs", params.column_ptrs, params.column_count)?;
    require_words(
        "lde_twiddles",
        params.twiddles,
        params.twiddles_size as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(id: u32, words: usize) -> ArenaSlice {
        ArenaSlice::dangling_for_test(id, words)
    }

    fn batch(id: u32, columns: u32) -> CommitLdeBatch {
        CommitLdeBatch {
            coefficient_ptrs: slice(id, columns as usize * 2),
            coefficient_sizes: slice(id + 1, columns as usize),
            column_ptrs: slice(id + 2, columns as usize * 2),
            column_count: columns,
            log_n: 6,
            twiddles: slice(id + 3, 64),
            twiddles_size: 64,
            eval_domain_size: 32,
        }
    }

    fn groups() -> Vec<CommitLeafGroup> {
        vec![
            CommitLeafGroup {
                first_column: 0,
                column_count: 16,
                column_ptrs: slice(10, 32),
                column_log_sizes: slice(11, 16),
                lde_batches: vec![batch(20, 16)],
            },
            CommitLeafGroup {
                first_column: 16,
                column_count: 3,
                column_ptrs: slice(13, 6),
                column_log_sizes: slice(14, 3),
                lde_batches: vec![batch(30, 3)],
            },
        ]
    }

    #[test]
    fn plan_preserves_canonical_launch_order_and_reaches_root() {
        let plan = CommitGraphPlan::new(
            6,
            slice(1, 64 * HASH_WORDS),
            groups(),
            vec![slice(2, 32 * HASH_WORDS), slice(3, 16 * HASH_WORDS)],
            Some(CommitTailPlan {
                level_ptrs: slice(4, 8),
                level_outputs: vec![
                    slice(5, 8 * HASH_WORDS),
                    slice(6, 4 * HASH_WORDS),
                    slice(7, 2 * HASH_WORDS),
                    slice(8, HASH_WORDS),
                ],
            }),
        )
        .unwrap();

        assert_eq!(plan.root().id().0, 8);
        assert_eq!(
            plan.launch_sequence().collect::<Vec<_>>(),
            vec![
                CommitLaunchKind::LeafInit { hashes: 64 },
                CommitLaunchKind::Lde {
                    group: 0,
                    batch: 0,
                    columns: 16,
                    log_n: 6,
                },
                CommitLaunchKind::LeafUpdate {
                    group: 0,
                    first_column: 0,
                    columns: 16,
                },
                CommitLaunchKind::Lde {
                    group: 1,
                    batch: 0,
                    columns: 3,
                    log_n: 6,
                },
                CommitLaunchKind::LeafFinalize {
                    group: 1,
                    first_column: 16,
                    columns: 3,
                },
                CommitLaunchKind::InteriorLayer {
                    level: 0,
                    output_hashes: 32,
                },
                CommitLaunchKind::InteriorLayer {
                    level: 1,
                    output_hashes: 16,
                },
                CommitLaunchKind::FusedTail {
                    first_hashes: 16,
                    levels: 4,
                },
            ]
        );
    }

    #[test]
    fn plan_rejects_noncanonical_groups_and_incomplete_tree() {
        let mut bad_groups = groups();
        bad_groups[1].first_column = 17;
        assert!(matches!(
            CommitGraphPlan::new(6, slice(1, 64 * HASH_WORDS), bad_groups, vec![], None,),
            Err(CommitGraphError::NonCanonicalGroupStart { .. })
        ));

        assert_eq!(
            CommitGraphPlan::new(
                6,
                slice(1, 64 * HASH_WORDS),
                groups(),
                vec![slice(2, 32 * HASH_WORDS)],
                None,
            )
            .unwrap_err(),
            CommitGraphError::IncompleteTree(32)
        );
    }

    #[test]
    fn plan_rejects_lde_count_and_buffer_geometry_mismatches() {
        let mut bad_groups = groups();
        bad_groups[0].lde_batches[0].column_count = 15;
        assert!(matches!(
            CommitGraphPlan::new(6, slice(1, 64 * HASH_WORDS), bad_groups, vec![], None,),
            Err(CommitGraphError::LdeColumnCountMismatch { .. })
        ));

        let mut bad_groups = groups();
        bad_groups[0].column_log_sizes = slice(11, 15);
        assert!(matches!(
            CommitGraphPlan::new(6, slice(1, 64 * HASH_WORDS), bad_groups, vec![], None,),
            Err(CommitGraphError::BufferTooSmall {
                role: "leaf_column_log_sizes",
                ..
            })
        ));
    }

    #[test]
    fn pruned_plan_allows_ordered_ping_pong_but_not_premature_reuse() {
        let plan = CommitGraphPlan::new_pruned(
            6,
            3,
            slice(1, 64 * HASH_WORDS),
            groups(),
            vec![
                slice(2, 32 * HASH_WORDS),
                slice(1, 64 * HASH_WORDS),
                slice(3, 8 * HASH_WORDS),
                slice(4, 4 * HASH_WORDS),
                slice(5, 2 * HASH_WORDS),
                slice(6, HASH_WORDS),
            ],
            None,
        )
        .unwrap();
        assert_eq!(plan.root().id(), ArenaSlotId(6));

        assert_eq!(
            CommitGraphPlan::new_pruned(
                6,
                1,
                slice(1, 64 * HASH_WORDS),
                groups(),
                vec![
                    slice(2, 32 * HASH_WORDS),
                    slice(3, 16 * HASH_WORDS),
                    // Slot 2 is a retained log-5 layer and is still needed by decommit.
                    slice(2, 32 * HASH_WORDS),
                ],
                None,
            )
            .unwrap_err(),
            CommitGraphError::PrematureArenaSlotReuse(ArenaSlotId(2))
        );
    }
}
