//! Direct Base/Interaction interpolation into the retained LDE slab.
//!
//! This is the first executable edge of coefficient-image retirement. It
//! performs only B2N and the exact zero-extension first-layer duplication;
//! the remaining N2B stages are a separate, explicit successor. Failure never
//! selects the coefficient-backed LDE path.

use core::ffi::c_void;
use std::collections::BTreeSet;

use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;

use super::*;
use crate::backend::prepared_decommit::TraceTreeRole;

#[derive(Clone, Copy, Debug)]
pub struct DirectRetainedB2nColumn {
    /// Source-domain bit-reversed evaluations. Only the first `2^source_log`
    /// words are live when the physical arena slot has pooled surplus.
    pub source_evaluations: ArenaSlice,
    /// Canonical retained destination. Its logical extent is exactly twice the
    /// source extent; the terminal B2N writer fills both halves.
    pub retained_output: ArenaSlice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRetainedB2nBatchPlan {
    pub batch_index: u32,
    pub source_log_size: u32,
    pub retained_log_size: u32,
    pub canonical_columns: Vec<usize>,
    pub pointer_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRetainedB2nProgram {
    role: TraceTreeRole,
    commit_cache_key: u64,
    twiddle_words: usize,
    column_count: usize,
    batches: Vec<DirectRetainedB2nBatchPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRetainedB2nOracle {
    /// B2N coefficients duplicated as `[coefficients, coefficients]`, in
    /// canonical column order. These are the exact bytes consumed by N2B at
    /// stage two.
    pub retained_stage_two_inputs: Vec<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRetainedB2nLaunchKind {
    pub role: TraceTreeRole,
    pub batch_index: u32,
    pub source_log_size: u32,
    pub retained_log_size: u32,
    pub columns: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectRetainedB2nError {
    UnsupportedRole(TraceTreeRole),
    UnsupportedBlowup(u32),
    InvalidProgram,
    UnsupportedLogSize {
        canonical_column: usize,
        source_log_size: u32,
        retained_log_size: u32,
    },
    ColumnCount {
        expected: usize,
        actual: usize,
    },
    PointerSlotShape,
    ContextMismatch(ArenaSlotId),
    SourceTooSmall {
        canonical_column: usize,
        required_words: usize,
        actual_words: usize,
    },
    RetainedOutputTooSmall {
        canonical_column: usize,
        required_words: usize,
        actual_words: usize,
    },
    InvalidAlias {
        first: ArenaSlotId,
        second: ArenaSlotId,
    },
    TwiddlesTooSmall {
        required_words: usize,
        actual_words: usize,
    },
    InvalidOracleColumn {
        canonical_column: usize,
        required_words: usize,
        actual_words: usize,
    },
    NonCanonicalWord {
        canonical_column: usize,
        row: usize,
        word: u32,
    },
    SizeOverflow,
    Prepared(super::super::prepared_commit::PreparedCommitError),
    Cuda(super::super::exec_context::CudaRuntimeError),
}

impl core::fmt::Display for DirectRetainedB2nError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid direct-retained B2N binding: {self:?}")
    }
}

impl std::error::Error for DirectRetainedB2nError {}

impl From<super::super::prepared_commit::PreparedCommitError> for DirectRetainedB2nError {
    fn from(value: super::super::prepared_commit::PreparedCommitError) -> Self {
        Self::Prepared(value)
    }
}

impl From<super::super::exec_context::CudaRuntimeError> for DirectRetainedB2nError {
    fn from(value: super::super::exec_context::CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

#[derive(Clone, Copy)]
struct LogicalColumn {
    source: ArenaSlice,
    retained: ArenaSlice,
}

#[derive(Clone, Copy)]
struct PreparedBatch {
    input_pointers: ArenaSlice,
    output_pointers: ArenaSlice,
    batch_index: u32,
    source_log_size: u32,
    retained_log_size: u32,
    columns: u32,
}

pub struct PreparedDirectRetainedB2nGraph<'a> {
    arena: &'a DeviceArena,
    role: TraceTreeRole,
    commit_cache_key: u64,
    inverse_twiddles: ArenaSlice,
    batches: Vec<PreparedBatch>,
    exact_lower_prefix_aliases: usize,
}

impl DirectRetainedB2nProgram {
    /// Seal the exact Base/Interaction B2N batches from an admitted commitment
    /// program. The duplicate-first primitive currently requires blowup two.
    pub fn compile(
        role: TraceTreeRole,
        commit: &CommitProgram,
    ) -> Result<Self, DirectRetainedB2nError> {
        if !matches!(role, TraceTreeRole::Base | TraceTreeRole::Interaction) {
            return Err(DirectRetainedB2nError::UnsupportedRole(role));
        }
        let requirements = commit.requirements();
        let blowup = requirements.leaves.plan.geometry.log_blowup_factor;
        if blowup != 1 {
            return Err(DirectRetainedB2nError::UnsupportedBlowup(blowup));
        }
        let columns = &requirements.leaves.plan.columns;
        let mut seen = vec![false; columns.len()];
        let mut batches = Vec::with_capacity(requirements.leaves.plan.lde_batches.len());
        for (batch_index, batch) in requirements.leaves.plan.lde_batches.iter().enumerate() {
            let Some(&first) = batch.columns.first() else {
                return Err(DirectRetainedB2nError::InvalidProgram);
            };
            let source_log_size = columns
                .get(first)
                .ok_or(DirectRetainedB2nError::InvalidProgram)?
                .coefficient_log_size;
            for (&canonical, retained) in batch.columns.iter().zip(&batch.retained_columns) {
                let Some(column) = columns.get(canonical) else {
                    return Err(DirectRetainedB2nError::InvalidProgram);
                };
                if seen[canonical]
                    || !column.retained_evaluation
                    || retained.is_none()
                    || column.coefficient_log_size != source_log_size
                    || column.evaluation_log_size != batch.evaluation_log_size
                {
                    return Err(DirectRetainedB2nError::InvalidProgram);
                }
                if !(3..=29).contains(&source_log_size)
                    || column.evaluation_log_size != source_log_size + 1
                {
                    return Err(DirectRetainedB2nError::UnsupportedLogSize {
                        canonical_column: canonical,
                        source_log_size,
                        retained_log_size: column.evaluation_log_size,
                    });
                }
                seen[canonical] = true;
            }
            batches.push(DirectRetainedB2nBatchPlan {
                batch_index: u32::try_from(batch_index)
                    .map_err(|_| DirectRetainedB2nError::SizeOverflow)?,
                source_log_size,
                retained_log_size: batch.evaluation_log_size,
                canonical_columns: batch.columns.clone(),
                pointer_words: batch
                    .columns
                    .len()
                    .checked_mul(POINTER_WORDS)
                    .ok_or(DirectRetainedB2nError::SizeOverflow)?,
            });
        }
        if seen.iter().any(|covered| !covered) {
            return Err(DirectRetainedB2nError::InvalidProgram);
        }
        Ok(Self {
            role,
            commit_cache_key: commit.identity().cache_key,
            twiddle_words: requirements.leaves.twiddle_words,
            column_count: columns.len(),
            batches,
        })
    }

    pub fn role(&self) -> TraceTreeRole {
        self.role
    }

    pub fn commit_cache_key(&self) -> u64 {
        self.commit_cache_key
    }

    pub fn batches(&self) -> &[DirectRetainedB2nBatchPlan] {
        &self.batches
    }

    /// Pointer-table requirements reuse the commitment program's existing
    /// coefficient-pointer and output-pointer slots; no parallel table exists.
    /// This graph therefore replaces the ordinary LDE graph for a run. It must
    /// not coexist with a legacy graph bound to the same slots: preparing
    /// either graph overwrites those tables. A/B qualification needs disjoint
    /// slot sets.
    pub fn arena_slot_requirements(
        &self,
        slots: &ProgressiveCommitWorkspaceSlots,
    ) -> Result<Vec<CommitArenaSlotRequirement>, DirectRetainedB2nError> {
        if slots.leaves.batches.len() != self.batches.len() {
            return Err(DirectRetainedB2nError::PointerSlotShape);
        }
        let mut seen = BTreeSet::new();
        let mut output = Vec::with_capacity(2 * self.batches.len());
        for (batch, bound) in self.batches.iter().zip(&slots.leaves.batches) {
            for id in [bound.coefficient_ptrs, bound.output_ptrs] {
                if !seen.insert(id) {
                    return Err(DirectRetainedB2nError::InvalidAlias {
                        first: id,
                        second: id,
                    });
                }
                output.push(CommitArenaSlotRequirement {
                    id,
                    len_words: batch.pointer_words,
                    alignment_words: POINTER_WORDS,
                });
            }
        }
        Ok(output)
    }

    /// Independent host B2N plus first-layer oracle. Inputs are source-domain
    /// bit-reversed evaluations in canonical column order.
    pub fn oracle(
        &self,
        source_evaluations: &[Vec<u32>],
    ) -> Result<DirectRetainedB2nOracle, DirectRetainedB2nError> {
        if source_evaluations.len() != self.column_count {
            return Err(DirectRetainedB2nError::ColumnCount {
                expected: self.column_count,
                actual: source_evaluations.len(),
            });
        }
        let max_source_log = self
            .batches
            .iter()
            .map(|batch| batch.source_log_size)
            .max()
            .ok_or(DirectRetainedB2nError::InvalidProgram)?;
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(max_source_log).circle_domain().half_coset,
        );
        let mut output = vec![Vec::new(); self.column_count];
        for batch in &self.batches {
            let required_words = words(batch.source_log_size)?;
            for &canonical in &batch.canonical_columns {
                let values = &source_evaluations[canonical];
                if values.len() != required_words {
                    return Err(DirectRetainedB2nError::InvalidOracleColumn {
                        canonical_column: canonical,
                        required_words,
                        actual_words: values.len(),
                    });
                }
                for (row, &word) in values.iter().enumerate() {
                    if word >= P {
                        return Err(DirectRetainedB2nError::NonCanonicalWord {
                            canonical_column: canonical,
                            row,
                            word,
                        });
                    }
                }
                let coefficients =
                    CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                        CanonicCoset::new(batch.source_log_size).circle_domain(),
                        values
                            .iter()
                            .copied()
                            .map(BaseField::from_u32_unchecked)
                            .collect(),
                    )
                    .interpolate_with_twiddles(&twiddles)
                    .coeffs
                    .into_iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>();
                let mut retained = Vec::with_capacity(2 * coefficients.len());
                retained.extend_from_slice(&coefficients);
                retained.extend_from_slice(&coefficients);
                output[canonical] = retained;
            }
        }
        Ok(DirectRetainedB2nOracle {
            retained_stage_two_inputs: output,
        })
    }
}

impl<'a> PreparedDirectRetainedB2nGraph<'a> {
    /// Bind exact logical extents, upload stable pointer tables once, and seal
    /// the direct-only launch vector. No fallback implementation is stored.
    /// The supplied slots are exclusively owned for this graph's lifetime; see
    /// [`DirectRetainedB2nProgram::arena_slot_requirements`].
    pub fn prepare(
        arena: &'a DeviceArena,
        program: &DirectRetainedB2nProgram,
        slots: &ProgressiveCommitWorkspaceSlots,
        columns: &[DirectRetainedB2nColumn],
        inverse_twiddles: ArenaSlice,
    ) -> Result<Self, DirectRetainedB2nError> {
        if columns.len() != program.column_count {
            return Err(DirectRetainedB2nError::ColumnCount {
                expected: program.column_count,
                actual: columns.len(),
            });
        }
        let token = arena.context().identity_token();
        if inverse_twiddles.context_token() != token {
            return Err(DirectRetainedB2nError::ContextMismatch(
                inverse_twiddles.id(),
            ));
        }
        if inverse_twiddles.len_words() < program.twiddle_words {
            return Err(DirectRetainedB2nError::TwiddlesTooSmall {
                required_words: program.twiddle_words,
                actual_words: inverse_twiddles.len_words(),
            });
        }
        let inverse_twiddles = inverse_twiddles.truncated(program.twiddle_words);
        let logical = bind_logical_columns(program, columns, token)?;
        validate_value_aliases(&logical, inverse_twiddles)?;

        let requirements = program.arena_slot_requirements(slots)?;
        let mut prepared = Vec::with_capacity(program.batches.len());
        let mut pointer_tables = Vec::with_capacity(requirements.len());
        let mut uploads = Vec::with_capacity(requirements.len());
        for (batch, batch_slots) in program.batches.iter().zip(&slots.leaves.batches) {
            let input_pointers = bind_slot(
                arena,
                batch_slots.coefficient_ptrs,
                batch.pointer_words,
                POINTER_WORDS,
            )?;
            let output_pointers = bind_slot(
                arena,
                batch_slots.output_ptrs,
                batch.pointer_words,
                POINTER_WORDS,
            )?;
            pointer_tables.extend([input_pointers, output_pointers]);
            uploads.push((
                input_pointers,
                batch
                    .canonical_columns
                    .iter()
                    .map(|&canonical| logical[canonical].source.as_u32_ptr() as usize)
                    .collect::<Vec<_>>(),
            ));
            uploads.push((
                output_pointers,
                batch
                    .canonical_columns
                    .iter()
                    .map(|&canonical| logical[canonical].retained.as_u32_ptr() as usize)
                    .collect::<Vec<_>>(),
            ));
            prepared.push(PreparedBatch {
                input_pointers,
                output_pointers,
                batch_index: batch.batch_index,
                source_log_size: batch.source_log_size,
                retained_log_size: batch.retained_log_size,
                columns: u32::try_from(batch.canonical_columns.len())
                    .map_err(|_| DirectRetainedB2nError::SizeOverflow)?,
            });
        }
        validate_pointer_aliases(&pointer_tables, &logical, inverse_twiddles)?;
        for (destination, addresses) in &uploads {
            let bytes = addresses
                .len()
                .checked_mul(core::mem::size_of::<usize>())
                .ok_or(DirectRetainedB2nError::SizeOverflow)?;
            unsafe {
                arena.context().memcpy_h2d_async(
                    destination.as_void_ptr(),
                    addresses.as_ptr().cast::<c_void>(),
                    bytes,
                )?;
            }
        }
        arena.context().sync()?;

        Ok(Self {
            arena,
            role: program.role,
            commit_cache_key: program.commit_cache_key,
            inverse_twiddles,
            batches: prepared,
            exact_lower_prefix_aliases: logical
                .iter()
                .filter(|column| exact_lower_prefix_alias(**column))
                .count(),
        })
    }

    pub fn launch(&self) -> Result<(), DirectRetainedB2nError> {
        let twiddle_words = u32::try_from(self.inverse_twiddles.len_words())
            .map_err(|_| DirectRetainedB2nError::SizeOverflow)?;
        let stream = self.arena.context().stream_raw().as_ptr();
        for batch in &self.batches {
            let eval_domain_size = 1u32
                .checked_shl(batch.source_log_size - 1)
                .ok_or(DirectRetainedB2nError::SizeOverflow)?;
            let code = unsafe {
                stwo_backend_cuda_kernels::raw::stwo_ntt_b2n_columns_to_retained_on(
                    batch.input_pointers.as_u32_ptr().cast(),
                    batch.output_pointers.as_u32_ptr().cast(),
                    batch.source_log_size,
                    batch.columns,
                    self.inverse_twiddles.as_u32_ptr(),
                    twiddle_words,
                    eval_domain_size,
                    stream,
                )
            };
            check_cuda("direct_retained_b2n", code)?;
        }
        Ok(())
    }

    pub fn launch_sequence(
        &self,
    ) -> impl ExactSizeIterator<Item = DirectRetainedB2nLaunchKind> + '_ {
        self.batches
            .iter()
            .map(|batch| DirectRetainedB2nLaunchKind {
                role: self.role,
                batch_index: batch.batch_index,
                source_log_size: batch.source_log_size,
                retained_log_size: batch.retained_log_size,
                columns: batch.columns,
            })
    }

    pub fn commit_cache_key(&self) -> u64 {
        self.commit_cache_key
    }

    pub fn exact_lower_prefix_aliases(&self) -> usize {
        self.exact_lower_prefix_aliases
    }
}

fn bind_logical_columns(
    program: &DirectRetainedB2nProgram,
    columns: &[DirectRetainedB2nColumn],
    token: core::ptr::NonNull<c_void>,
) -> Result<Vec<LogicalColumn>, DirectRetainedB2nError> {
    let mut output = vec![None; columns.len()];
    for batch in &program.batches {
        let source_words = words(batch.source_log_size)?;
        let retained_words = words(batch.retained_log_size)?;
        for &canonical in &batch.canonical_columns {
            let column = columns[canonical];
            for slice in [column.source_evaluations, column.retained_output] {
                if slice.context_token() != token {
                    return Err(DirectRetainedB2nError::ContextMismatch(slice.id()));
                }
            }
            if column.source_evaluations.len_words() < source_words {
                return Err(DirectRetainedB2nError::SourceTooSmall {
                    canonical_column: canonical,
                    required_words: source_words,
                    actual_words: column.source_evaluations.len_words(),
                });
            }
            if column.retained_output.len_words() < retained_words {
                return Err(DirectRetainedB2nError::RetainedOutputTooSmall {
                    canonical_column: canonical,
                    required_words: retained_words,
                    actual_words: column.retained_output.len_words(),
                });
            }
            output[canonical] = Some(LogicalColumn {
                source: column.source_evaluations.truncated(source_words),
                retained: column.retained_output.truncated(retained_words),
            });
        }
    }
    output
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(DirectRetainedB2nError::InvalidProgram)
}

fn validate_value_aliases(
    columns: &[LogicalColumn],
    inverse_twiddles: ArenaSlice,
) -> Result<(), DirectRetainedB2nError> {
    let mut ranges = Vec::with_capacity(2 * columns.len());
    for (canonical, column) in columns.iter().enumerate() {
        ranges.push((
            canonical,
            false,
            column.source,
            address_range(column.source)?,
        ));
        ranges.push((
            canonical,
            true,
            column.retained,
            address_range(column.retained)?,
        ));
    }
    for (index, &(canonical, output, slice, range)) in ranges.iter().enumerate() {
        for &(other_canonical, other_output, other_slice, other_range) in &ranges[index + 1..] {
            let paired = canonical == other_canonical && output != other_output;
            let permitted = paired && exact_lower_prefix_alias(columns[canonical]);
            if (slice.id() == other_slice.id() || ranges_overlap(range, other_range)) && !permitted
            {
                return Err(DirectRetainedB2nError::InvalidAlias {
                    first: slice.id(),
                    second: other_slice.id(),
                });
            }
        }
        if slice.id() == inverse_twiddles.id()
            || ranges_overlap(range, address_range(inverse_twiddles)?)
        {
            return Err(DirectRetainedB2nError::InvalidAlias {
                first: slice.id(),
                second: inverse_twiddles.id(),
            });
        }
    }
    Ok(())
}

fn validate_pointer_aliases(
    pointer_tables: &[ArenaSlice],
    columns: &[LogicalColumn],
    inverse_twiddles: ArenaSlice,
) -> Result<(), DirectRetainedB2nError> {
    for (index, &table) in pointer_tables.iter().enumerate() {
        let range = address_range(table)?;
        for &other in &pointer_tables[index + 1..] {
            if table.id() == other.id() || ranges_overlap(range, address_range(other)?) {
                return Err(DirectRetainedB2nError::InvalidAlias {
                    first: table.id(),
                    second: other.id(),
                });
            }
        }
        for value in columns
            .iter()
            .flat_map(|column| [column.source, column.retained])
            .chain(core::iter::once(inverse_twiddles))
        {
            if table.id() == value.id() || ranges_overlap(range, address_range(value)?) {
                return Err(DirectRetainedB2nError::InvalidAlias {
                    first: table.id(),
                    second: value.id(),
                });
            }
        }
    }
    Ok(())
}

fn exact_lower_prefix_alias(column: LogicalColumn) -> bool {
    column.source.id() == column.retained.id()
        && column.source.as_u32_ptr() == column.retained.as_u32_ptr()
        && column.source.len_words().checked_mul(2) == Some(column.retained.len_words())
}

fn address_range(slice: ArenaSlice) -> Result<(usize, usize), DirectRetainedB2nError> {
    let start = slice.as_u32_ptr() as usize;
    let bytes = slice
        .len_words()
        .checked_mul(WORD_BYTES)
        .ok_or(DirectRetainedB2nError::SizeOverflow)?;
    Ok((
        start,
        start
            .checked_add(bytes)
            .ok_or(DirectRetainedB2nError::SizeOverflow)?,
    ))
}

const fn ranges_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

fn words(log_size: u32) -> Result<usize, DirectRetainedB2nError> {
    1usize
        .checked_shl(log_size)
        .ok_or(DirectRetainedB2nError::SizeOverflow)
}

#[cfg(test)]
#[path = "direct_retained_b2n_tests.rs"]
mod tests;
