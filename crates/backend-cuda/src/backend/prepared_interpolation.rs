//! Allocation-free, explicit-stream interpolation from resident evaluations.
//!
//! Relation kernels must retain their canonical evaluation columns for AIR
//! consumers while PCS stages consume a distinct coefficient representation.
//! Setup binds and uploads stable destination pointer tables once.  Every
//! [`PreparedInterpolationGraph::launch`] then copies evaluations to the
//! coefficient slots and inverse-transforms those copies on the proof-owned
//! stream, making the same launch sequence safe for eager execution and graph
//! capture.

use core::ffi::c_void;
use std::collections::BTreeSet;

use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const POINTER_WORDS: usize = core::mem::size_of::<*mut u32>().div_ceil(WORD_BYTES);

pub const INTERPOLATION_POINTER_ALIGNMENT_WORDS: usize =
    core::mem::align_of::<*mut u32>() / WORD_BYTES;

/// One evaluation/coefficient pair.  The two slices are required to be
/// different because both values remain live after interpolation.
#[derive(Clone, Copy, Debug)]
pub struct InterpolationColumn {
    pub evaluations: ArenaSlice,
    pub coefficients: ArenaSlice,
    pub log_size: u32,
}

/// One same-log inverse-NTT batch and its stable device pointer table.
#[derive(Clone, Debug)]
pub struct InterpolationBatch {
    pub columns: Vec<InterpolationColumn>,
    pub coefficient_pointers: ArenaSlotId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedInterpolationError {
    EmptyBatches,
    EmptyBatch(usize),
    InvalidLogSize {
        batch: usize,
        log_size: u32,
    },
    MixedLogSizes {
        batch: usize,
        column: usize,
        expected: u32,
        actual: u32,
    },
    ContextMismatch(ArenaSlotId),
    EvaluationAliasesCoefficient(ArenaSlotId),
    DuplicateEvaluation(ArenaSlotId),
    DuplicateCoefficient(ArenaSlotId),
    DuplicatePointerTable(ArenaSlotId),
    PointerTableAliasesValue(ArenaSlotId),
    PointerTableAliasesTwiddles(ArenaSlotId),
    TwiddlesAliasValue(ArenaSlotId),
    ColumnTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    PointerTableTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    MisalignedPointerTable(ArenaSlotId),
    TwiddlesTooSmall {
        required_words: usize,
        actual_words: usize,
    },
    SizeOverflow,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedInterpolationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid prepared CUDA interpolation: {self:?}")
    }
}

impl std::error::Error for PreparedInterpolationError {}

impl From<ArenaError> for PreparedInterpolationError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedInterpolationError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

#[derive(Clone, Debug)]
struct PreparedBatch {
    columns: Vec<InterpolationColumn>,
    coefficient_pointers: ArenaSlice,
    log_size: u32,
}

/// Prepared resident interpolation.  Setup may synchronize after uploading
/// immutable pointer descriptors; launch never allocates, synchronizes, or uses
/// the legacy/default CUDA stream.
pub struct PreparedInterpolationGraph<'a> {
    arena: &'a DeviceArena,
    inverse_twiddles: ArenaSlice,
    batches: Vec<PreparedBatch>,
}

impl<'a> PreparedInterpolationGraph<'a> {
    pub fn prepare(
        arena: &'a DeviceArena,
        batches: &[InterpolationBatch],
        inverse_twiddles: ArenaSlice,
    ) -> Result<Self, PreparedInterpolationError> {
        let pointer_tables = validate_batches(arena, batches, inverse_twiddles)?;

        let mut prepared = Vec::with_capacity(batches.len());
        let mut pointer_uploads = Vec::with_capacity(batches.len());
        for (batch, coefficient_pointers) in batches.iter().zip(pointer_tables) {
            let pointers: Vec<usize> = batch
                .columns
                .iter()
                .map(|column| column.coefficients.as_u32_ptr() as usize)
                .collect();
            let bytes = pointers
                .len()
                .checked_mul(core::mem::size_of::<usize>())
                .ok_or(PreparedInterpolationError::SizeOverflow)?;
            pointer_uploads.push((coefficient_pointers, pointers, bytes));
            prepared.push(PreparedBatch {
                columns: batch.columns.clone(),
                coefficient_pointers,
                log_size: batch.columns[0].log_size,
            });
        }
        for (destination, pointers, bytes) in &pointer_uploads {
            // SAFETY: pointer-table capacity/alignment were validated, and all
            // host vectors remain owned by `pointer_uploads` until the setup
            // stream synchronization below completes.
            unsafe {
                arena.context().memcpy_h2d_async(
                    destination.as_void_ptr(),
                    pointers.as_ptr().cast::<c_void>(),
                    *bytes,
                )?;
            }
        }
        arena.context().sync()?;

        Ok(Self {
            arena,
            inverse_twiddles,
            batches: prepared,
        })
    }

    /// Copy each canonical evaluation to its distinct coefficient slot and
    /// interpolate the copies in same-log batches on the arena stream.
    pub fn launch(&self) -> Result<(), PreparedInterpolationError> {
        let twiddle_words = u32::try_from(self.inverse_twiddles.len_words())
            .map_err(|_| PreparedInterpolationError::SizeOverflow)?;
        let stream = self.arena.context().stream_raw().as_ptr();

        for batch in &self.batches {
            let words = pow2(batch.log_size)?;
            let bytes = words
                .checked_mul(WORD_BYTES)
                .ok_or(PreparedInterpolationError::SizeOverflow)?;
            for column in &batch.columns {
                // SAFETY: validation proves distinct, same-context source and
                // destination ranges of at least `2^log_size` words.
                unsafe {
                    self.arena.context().memcpy_d2d_async(
                        column.coefficients.as_void_ptr(),
                        column.evaluations.as_void_ptr().cast_const(),
                        bytes,
                    )?;
                }
            }

            let column_count = u32::try_from(batch.columns.len())
                .map_err(|_| PreparedInterpolationError::SizeOverflow)?;
            let eval_domain_size = 1u32
                .checked_shl(batch.log_size - 1)
                .ok_or(PreparedInterpolationError::SizeOverflow)?;
            let code = unsafe {
                stwo_backend_cuda_kernels::raw::stwo_ntt_b2n_columns_on(
                    batch.coefficient_pointers.as_u32_ptr().cast::<*mut u32>(),
                    batch.log_size,
                    column_count,
                    self.inverse_twiddles.as_u32_ptr(),
                    twiddle_words,
                    eval_domain_size,
                    stream,
                )
            };
            check_cuda("prepared_interpolation", code)?;
        }
        Ok(())
    }

    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    pub fn column_count(&self) -> usize {
        self.batches.iter().map(|batch| batch.columns.len()).sum()
    }
}

fn validate_batches(
    arena: &DeviceArena,
    batches: &[InterpolationBatch],
    inverse_twiddles: ArenaSlice,
) -> Result<Vec<ArenaSlice>, PreparedInterpolationError> {
    if batches.is_empty() {
        return Err(PreparedInterpolationError::EmptyBatches);
    }
    let context = arena.context().identity_token();
    if inverse_twiddles.context_token() != context {
        return Err(PreparedInterpolationError::ContextMismatch(
            inverse_twiddles.id(),
        ));
    }

    let mut evaluations = BTreeSet::new();
    let mut coefficients = BTreeSet::new();
    let mut pointer_ids = BTreeSet::new();
    let mut max_twiddle_words = 0usize;
    let mut pointer_tables = Vec::with_capacity(batches.len());

    for (batch_index, batch) in batches.iter().enumerate() {
        let Some(first) = batch.columns.first() else {
            return Err(PreparedInterpolationError::EmptyBatch(batch_index));
        };
        if !(1..=30).contains(&first.log_size) {
            return Err(PreparedInterpolationError::InvalidLogSize {
                batch: batch_index,
                log_size: first.log_size,
            });
        }
        let required_words = pow2(first.log_size)?;
        max_twiddle_words = max_twiddle_words.max(required_words / 2);

        if !pointer_ids.insert(batch.coefficient_pointers) {
            return Err(PreparedInterpolationError::DuplicatePointerTable(
                batch.coefficient_pointers,
            ));
        }
        let pointer_words = batch
            .columns
            .len()
            .checked_mul(POINTER_WORDS)
            .ok_or(PreparedInterpolationError::SizeOverflow)?;
        let pointer_table = arena.bind(batch.coefficient_pointers)?;
        if pointer_table.len_words() < pointer_words {
            return Err(PreparedInterpolationError::PointerTableTooSmall {
                slot: batch.coefficient_pointers,
                required_words: pointer_words,
                actual_words: pointer_table.len_words(),
            });
        }
        if (pointer_table.as_u32_ptr() as usize)
            % (INTERPOLATION_POINTER_ALIGNMENT_WORDS * WORD_BYTES)
            != 0
        {
            return Err(PreparedInterpolationError::MisalignedPointerTable(
                batch.coefficient_pointers,
            ));
        }
        // Pooled slots may be larger than the pointer table; keep only the
        // logical extent so no consumer derives sizes from the surplus.
        pointer_tables.push(pointer_table.truncated(pointer_words));

        for (column_index, column) in batch.columns.iter().enumerate() {
            if column.log_size != first.log_size {
                return Err(PreparedInterpolationError::MixedLogSizes {
                    batch: batch_index,
                    column: column_index,
                    expected: first.log_size,
                    actual: column.log_size,
                });
            }
            for value in [column.evaluations, column.coefficients] {
                if value.context_token() != context {
                    return Err(PreparedInterpolationError::ContextMismatch(value.id()));
                }
                if value.len_words() < required_words {
                    return Err(PreparedInterpolationError::ColumnTooSmall {
                        slot: value.id(),
                        required_words,
                        actual_words: value.len_words(),
                    });
                }
            }
            insert_value_identities(&mut evaluations, &mut coefficients, *column)?;
        }
    }

    if let Some(id) = pointer_ids
        .iter()
        .find(|id| evaluations.contains(id) || coefficients.contains(id))
    {
        return Err(PreparedInterpolationError::PointerTableAliasesValue(*id));
    }
    if pointer_ids.contains(&inverse_twiddles.id()) {
        return Err(PreparedInterpolationError::PointerTableAliasesTwiddles(
            inverse_twiddles.id(),
        ));
    }
    if evaluations.contains(&inverse_twiddles.id()) || coefficients.contains(&inverse_twiddles.id())
    {
        return Err(PreparedInterpolationError::TwiddlesAliasValue(
            inverse_twiddles.id(),
        ));
    }
    if inverse_twiddles.len_words() < max_twiddle_words {
        return Err(PreparedInterpolationError::TwiddlesTooSmall {
            required_words: max_twiddle_words,
            actual_words: inverse_twiddles.len_words(),
        });
    }
    Ok(pointer_tables)
}

fn insert_value_identities(
    evaluations: &mut BTreeSet<ArenaSlotId>,
    coefficients: &mut BTreeSet<ArenaSlotId>,
    column: InterpolationColumn,
) -> Result<(), PreparedInterpolationError> {
    let evaluation = column.evaluations.id();
    let coefficient = column.coefficients.id();
    if evaluation == coefficient || coefficients.contains(&evaluation) {
        return Err(PreparedInterpolationError::EvaluationAliasesCoefficient(
            evaluation,
        ));
    }
    if evaluations.contains(&coefficient) {
        return Err(PreparedInterpolationError::EvaluationAliasesCoefficient(
            coefficient,
        ));
    }
    if !evaluations.insert(evaluation) {
        return Err(PreparedInterpolationError::DuplicateEvaluation(evaluation));
    }
    if !coefficients.insert(coefficient) {
        return Err(PreparedInterpolationError::DuplicateCoefficient(
            coefficient,
        ));
    }
    Ok(())
}

fn pow2(log_size: u32) -> Result<usize, PreparedInterpolationError> {
    1usize
        .checked_shl(log_size)
        .ok_or(PreparedInterpolationError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(evaluations: u32, coefficients: u32, log_size: u32) -> InterpolationColumn {
        InterpolationColumn {
            evaluations: ArenaSlice::dangling_for_test(evaluations, 1 << log_size),
            coefficients: ArenaSlice::dangling_for_test(coefficients, 1 << log_size),
            log_size,
        }
    }

    #[test]
    fn evaluation_and_coefficient_identity_sets_must_be_disjoint() {
        let mut evaluations = BTreeSet::new();
        let mut coefficients = BTreeSet::new();
        insert_value_identities(&mut evaluations, &mut coefficients, column(1, 2, 5)).unwrap();
        assert_eq!(
            insert_value_identities(&mut evaluations, &mut coefficients, column(3, 1, 5)),
            Err(PreparedInterpolationError::EvaluationAliasesCoefficient(
                ArenaSlotId(1)
            ))
        );
        assert_eq!(
            insert_value_identities(&mut BTreeSet::new(), &mut BTreeSet::new(), column(4, 4, 5),),
            Err(PreparedInterpolationError::EvaluationAliasesCoefficient(
                ArenaSlotId(4)
            ))
        );
    }

    #[test]
    fn batches_preserve_caller_column_order() {
        let batch = InterpolationBatch {
            columns: vec![column(7, 17, 6), column(4, 14, 6), column(9, 19, 6)],
            coefficient_pointers: ArenaSlotId(30),
        };
        assert_eq!(
            batch
                .columns
                .iter()
                .map(|column| (column.evaluations.id(), column.coefficients.id()))
                .collect::<Vec<_>>(),
            [
                (ArenaSlotId(7), ArenaSlotId(17)),
                (ArenaSlotId(4), ArenaSlotId(14)),
                (ArenaSlotId(9), ArenaSlotId(19)),
            ]
        );
    }
}
