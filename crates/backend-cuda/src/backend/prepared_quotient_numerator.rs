//! Prepared, arena-native FRI quotient numerator construction.
//!
//! Setup freezes the two distinct reference orders: Fiat-Shamir powers follow
//! tree/column/mask order, while accumulation follows stable evaluation-log
//! order. Replay consumes canonical device OODS points and values, derives the
//! periodicity terms, and writes the exact lifted partial numerators expected
//! by [`super::prepared_quotient::PreparedQuotientGraph`].

mod single_write;

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;

use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena,
};
use super::prepared_quotient::{QuotientNumeratorSource, QuotientSampleConstants};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const SECURE_WORDS: usize = 4;
const SECURE_POINT_WORDS: usize = 8;
const RUNTIME_TERM_WORDS: usize = 5;
const BATCH_TERM_WORDS: usize = 3;
const LINE_COEFFICIENT_WORDS: usize = 3 * SECURE_WORDS;
const POINTER_WORDS: usize = core::mem::size_of::<*mut u32>().div_ceil(WORD_BYTES);

pub const QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS: usize =
    core::mem::align_of::<*mut u32>() / WORD_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorWorkspaceConfig {
    pub lifting_log_size: u32,
    pub log_blowup_factor: u32,
    /// Maximum transient words used to materialize coefficient-form columns.
    pub max_lde_tile_words: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum QuotientNumeratorColumnSource {
    /// Full committed LDE in bit-reversed circle-domain order.
    Evaluation(ArenaSlice),
    /// Circle coefficients. Replay materializes the committed LDE into the tile.
    Coefficients(ArenaSlice),
}

impl QuotientNumeratorColumnSource {
    fn slice(self) -> ArenaSlice {
        match self {
            Self::Evaluation(slice) | Self::Coefficients(slice) => slice,
        }
    }

    fn is_coefficients(self) -> bool {
        matches!(self, Self::Coefficients(_))
    }

    fn kind(self) -> QuotientNumeratorSourceKind {
        if self.is_coefficients() {
            QuotientNumeratorSourceKind::Coefficients
        } else {
            QuotientNumeratorSourceKind::Evaluation
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotientNumeratorSourceKind {
    Evaluation,
    Coefficients,
}

/// One canonical OODS entry. `shape_point` is the fixed discovery point used
/// only to identify equal symbolic points and reproduce the planned stable
/// `(x, y)` group order; replay reads the actual point from `input_index`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientOodsSample {
    pub input_index: u32,
    pub shape_point: CirclePoint<SecureField>,
}

/// Columns must be supplied in the exact flattened PCS tree/column order.
#[derive(Clone, Debug)]
pub struct QuotientNumeratorColumn {
    pub coefficient_log_size: u32,
    pub source: QuotientNumeratorColumnSource,
    pub samples: Vec<QuotientOodsSample>,
}

/// Address-free topology used by the proof arena planner before device slots
/// exist. [`PreparedQuotientNumeratorGraph::prepare`] validates that the bound
/// sources have this exact shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorColumnTopology {
    pub coefficient_log_size: u32,
    pub source_kind: QuotientNumeratorSourceKind,
    pub samples: Vec<QuotientOodsSample>,
}

impl From<&QuotientNumeratorColumn> for QuotientNumeratorColumnTopology {
    fn from(column: &QuotientNumeratorColumn) -> Self {
        Self {
            coefficient_log_size: column.coefficient_log_size,
            source_kind: column.source.kind(),
            samples: column.samples.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuotientNumeratorDestination {
    pub log_size: u32,
    pub coordinates: [ArenaSlice; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorGroupRequirements {
    pub shape_point: CirclePoint<SecureField>,
    pub log_size: u32,
    pub value_words: usize,
    /// Distinct sampled coefficient columns contributing to this output.
    /// Zero means the group can consume retained evaluations exclusively.
    pub coefficient_source_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorBatchRequirements {
    pub evaluation_log_size: u32,
    pub source_count: usize,
    pub coefficient_count: usize,
    pub term_count: usize,
    pub lde_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorWorkspaceRequirements {
    pub config: QuotientNumeratorWorkspaceConfig,
    pub input_sample_count: usize,
    pub term_count: usize,
    pub groups: Vec<QuotientNumeratorGroupRequirements>,
    pub batches: Vec<QuotientNumeratorBatchRequirements>,
    pub runtime_term_words: usize,
    pub group_term_index_words: usize,
    pub group_offset_words: usize,
    pub line_coefficient_words: usize,
    pub term_point_words: usize,
    pub batch_term_words: usize,
    pub batch_group_offset_words: usize,
    pub batch_source_pointer_words: usize,
    pub coefficient_pointer_words: usize,
    pub coefficient_size_words: usize,
    pub coefficient_output_pointer_words: usize,
    pub output_pointer_words: usize,
    pub output_log_size_words: usize,
    pub lde_tile_words: usize,
    pub forward_twiddle_words: usize,
    pub max_output_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorWorkspaceSlots {
    pub runtime_terms: ArenaSlotId,
    pub group_term_indices: ArenaSlotId,
    pub group_offsets: ArenaSlotId,
    pub line_coefficients: ArenaSlotId,
    pub term_points: ArenaSlotId,
    pub batch_terms: ArenaSlotId,
    pub batch_group_offsets: ArenaSlotId,
    pub batch_source_ptrs: ArenaSlotId,
    pub output_ptrs: ArenaSlotId,
    pub output_log_sizes: ArenaSlotId,
    pub coefficient_ptrs: Option<ArenaSlotId>,
    pub coefficient_sizes: Option<ArenaSlotId>,
    pub coefficient_output_ptrs: Option<ArenaSlotId>,
    pub lde_tile: Option<ArenaSlotId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedQuotientNumeratorError {
    InvalidLiftingLogSize(u32),
    InvalidBlowup {
        lifting_log_size: u32,
        log_blowup_factor: u32,
    },
    EmptyTerms,
    TooManyTerms(usize),
    TooManyGroups(usize),
    ColumnLogTooLarge {
        column: usize,
        log_size: u32,
        maximum: u32,
    },
    TileTooSmall {
        column: usize,
        required_words: usize,
        available_words: usize,
    },
    DestinationCountMismatch {
        expected: usize,
        actual: usize,
    },
    DestinationLogMismatch {
        group: usize,
        expected: u32,
        actual: u32,
    },
    OptionalSlotShapeMismatch,
    DuplicateSlot(ArenaSlotId),
    ContextMismatch(ArenaSlotId),
    AliasedExternalSlot(ArenaSlotId),
    ExternalAliasesWorkspace(ArenaSlotId),
    SlotTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    MisalignedSlot {
        slot: ArenaSlotId,
        alignment_words: usize,
    },
    InputTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    SourceTooSmall {
        column: usize,
        required_words: usize,
        actual_words: usize,
    },
    ForwardTwiddlesTooSmall {
        required_words: usize,
        actual_words: usize,
    },
    SizeOverflow,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedQuotientNumeratorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "invalid prepared CUDA quotient numerator workspace: {self:?}"
        )
    }
}

impl std::error::Error for PreparedQuotientNumeratorError {}

impl From<ArenaError> for PreparedQuotientNumeratorError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedQuotientNumeratorError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

#[derive(Clone)]
struct PlannedTerm {
    sample_index: u32,
    exponent: u32,
    period: Option<CirclePoint<BaseField>>,
    shape_point: CirclePoint<SecureField>,
    column: usize,
    group: usize,
}

pub(super) struct PlannedBatch {
    pub(super) evaluation_log_size: u32,
    pub(super) columns: Vec<usize>,
    pub(super) coefficient_columns: Vec<usize>,
    pub(super) group_offsets: Vec<u32>,
    pub(super) terms: Vec<u32>,
    lde_words: usize,
}

pub(super) struct NumeratorPlan {
    pub(super) requirements: QuotientNumeratorWorkspaceRequirements,
    terms: Vec<PlannedTerm>,
    group_term_indices: Vec<u32>,
    pub(super) group_offsets: Vec<u32>,
    pub(super) batches: Vec<PlannedBatch>,
}

pub fn quotient_numerator_workspace_requirements(
    config: QuotientNumeratorWorkspaceConfig,
    columns: &[QuotientNumeratorColumnTopology],
) -> Result<QuotientNumeratorWorkspaceRequirements, PreparedQuotientNumeratorError> {
    Ok(build_plan(config, columns)?.requirements)
}

impl QuotientNumeratorWorkspaceRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: &QuotientNumeratorWorkspaceSlots,
    ) -> Result<Vec<QuotientNumeratorArenaSlotRequirement>, PreparedQuotientNumeratorError> {
        let mut requirements = vec![
            slot(slots.runtime_terms, self.runtime_term_words, 1),
            slot(slots.group_term_indices, self.group_term_index_words, 1),
            slot(slots.group_offsets, self.group_offset_words, 1),
            slot(slots.line_coefficients, self.line_coefficient_words, 1),
            slot(slots.term_points, self.term_point_words, 1),
            slot(slots.batch_terms, self.batch_term_words, 1),
            slot(slots.batch_group_offsets, self.batch_group_offset_words, 1),
            slot(
                slots.batch_source_ptrs,
                self.batch_source_pointer_words,
                QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
            ),
            slot(
                slots.output_ptrs,
                self.output_pointer_words,
                QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
            ),
            slot(slots.output_log_sizes, self.output_log_size_words, 1),
        ];
        let coefficient_slots = [
            slots.coefficient_ptrs,
            slots.coefficient_sizes,
            slots.coefficient_output_ptrs,
            slots.lde_tile,
        ];
        if self.coefficient_size_words == 0 {
            if coefficient_slots.iter().any(Option::is_some) {
                return Err(PreparedQuotientNumeratorError::OptionalSlotShapeMismatch);
            }
        } else {
            let [Some(coefficient_ptrs), Some(coefficient_sizes), Some(coefficient_output_ptrs), Some(lde_tile)] =
                coefficient_slots
            else {
                return Err(PreparedQuotientNumeratorError::OptionalSlotShapeMismatch);
            };
            requirements.extend([
                slot(
                    coefficient_ptrs,
                    self.coefficient_pointer_words,
                    QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
                ),
                slot(coefficient_sizes, self.coefficient_size_words, 1),
                slot(
                    coefficient_output_ptrs,
                    self.coefficient_output_pointer_words,
                    QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
                ),
                slot(lde_tile, self.lde_tile_words, 1),
            ]);
        }
        let mut seen = BTreeSet::new();
        for requirement in &requirements {
            if !seen.insert(requirement.id) {
                return Err(PreparedQuotientNumeratorError::DuplicateSlot(
                    requirement.id,
                ));
            }
        }
        Ok(requirements)
    }
}

fn slot(
    id: ArenaSlotId,
    len_words: usize,
    alignment_words: usize,
) -> QuotientNumeratorArenaSlotRequirement {
    QuotientNumeratorArenaSlotRequirement {
        id,
        len_words,
        alignment_words,
    }
}

pub(super) fn build_plan(
    config: QuotientNumeratorWorkspaceConfig,
    columns: &[QuotientNumeratorColumnTopology],
) -> Result<NumeratorPlan, PreparedQuotientNumeratorError> {
    if !(2..=30).contains(&config.lifting_log_size) {
        return Err(PreparedQuotientNumeratorError::InvalidLiftingLogSize(
            config.lifting_log_size,
        ));
    }
    if config.log_blowup_factor == 0 || config.log_blowup_factor >= config.lifting_log_size {
        return Err(PreparedQuotientNumeratorError::InvalidBlowup {
            lifting_log_size: config.lifting_log_size,
            log_blowup_factor: config.log_blowup_factor,
        });
    }
    let max_coefficient_log = config.lifting_log_size - config.log_blowup_factor;
    let lifting_step = CanonicCoset::new(config.lifting_log_size).step();
    let mut terms = Vec::new();
    let mut column_terms = vec![Vec::new(); columns.len()];
    let mut next_exponent = 0usize;
    let mut max_input_index = 0usize;

    for (column_index, column) in columns.iter().enumerate() {
        if !column.samples.is_empty() && column.coefficient_log_size > max_coefficient_log {
            return Err(PreparedQuotientNumeratorError::ColumnLogTooLarge {
                column: column_index,
                log_size: column.coefficient_log_size,
                maximum: max_coefficient_log,
            });
        }
        let evaluation_log_size = column
            .coefficient_log_size
            .checked_add(config.log_blowup_factor)
            .ok_or(PreparedQuotientNumeratorError::SizeOverflow)?;
        let mut append = |sample: QuotientOodsSample,
                          period: Option<CirclePoint<BaseField>>|
         -> Result<(), PreparedQuotientNumeratorError> {
            let exponent = u32::try_from(next_exponent).map_err(|_| {
                PreparedQuotientNumeratorError::TooManyTerms(next_exponent.saturating_add(1))
            })?;
            let shape_point = match period {
                Some(period) => sample.shape_point + period.into_ef(),
                None => sample.shape_point,
            };
            let term_index = terms.len();
            terms.push(PlannedTerm {
                sample_index: sample.input_index,
                exponent,
                period,
                shape_point,
                column: column_index,
                group: 0,
            });
            column_terms[column_index].push(term_index);
            next_exponent = next_exponent
                .checked_add(1)
                .ok_or(PreparedQuotientNumeratorError::SizeOverflow)?;
            max_input_index = max_input_index.max(sample.input_index as usize);
            Ok(())
        };

        if let [_, second] = column.samples.as_slice() {
            append(
                *second,
                Some(lifting_step.repeated_double(evaluation_log_size)),
            )?;
        }
        for &sample in &column.samples {
            append(sample, None)?;
        }
    }
    if terms.is_empty() {
        return Err(PreparedQuotientNumeratorError::EmptyTerms);
    }

    let mut point_logs = BTreeMap::<(SecureField, SecureField), u32>::new();
    for term in &terms {
        let log = columns[term.column].coefficient_log_size;
        point_logs
            .entry((term.shape_point.x, term.shape_point.y))
            .and_modify(|current| *current = (*current).max(log))
            .or_insert(log);
    }
    if point_logs.len() > u16::MAX as usize {
        return Err(PreparedQuotientNumeratorError::TooManyGroups(
            point_logs.len(),
        ));
    }
    let group_indices: BTreeMap<_, _> = point_logs
        .keys()
        .copied()
        .enumerate()
        .map(|(index, point)| (point, index))
        .collect();
    for term in &mut terms {
        term.group = group_indices[&(term.shape_point.x, term.shape_point.y)];
    }
    let mut coefficient_sources_by_group = vec![BTreeSet::new(); point_logs.len()];
    for term in &terms {
        if columns[term.column].source_kind == QuotientNumeratorSourceKind::Coefficients {
            coefficient_sources_by_group[term.group].insert(term.column);
        }
    }
    let groups = point_logs
        .iter()
        .enumerate()
        .map(|(group, (&(x, y), &log_size))| {
            Ok(QuotientNumeratorGroupRequirements {
                shape_point: CirclePoint { x, y },
                log_size,
                value_words: pow2(log_size)?,
                coefficient_source_count: coefficient_sources_by_group[group].len(),
            })
        })
        .collect::<Result<Vec<_>, PreparedQuotientNumeratorError>>()?;

    let mut grouped_terms: Vec<_> = (0..terms.len()).collect();
    grouped_terms.sort_by_key(|&term| terms[term].group);
    let group_term_indices = grouped_terms
        .iter()
        .map(|&term| {
            u32::try_from(term)
                .map_err(|_| PreparedQuotientNumeratorError::TooManyTerms(terms.len()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let group_offsets = offsets_by_group(&grouped_terms, groups.len(), |&term| terms[term].group)?;

    let mut sorted_columns: Vec<_> = columns
        .iter()
        .enumerate()
        .filter(|(index, _)| !column_terms[*index].is_empty())
        .map(|(index, column)| {
            (
                column.coefficient_log_size + config.log_blowup_factor,
                index,
            )
        })
        .collect();
    sorted_columns.sort_by_key(|&(log, index)| (log, index));

    let mut column_batches = Vec::<Vec<usize>>::new();
    let mut current = Vec::new();
    let mut current_log = None;
    let mut current_tile_words = 0usize;
    for (evaluation_log_size, column_index) in sorted_columns {
        let cost = if columns[column_index].source_kind == QuotientNumeratorSourceKind::Coefficients
        {
            pow2(evaluation_log_size)?
        } else {
            0
        };
        if cost > config.max_lde_tile_words {
            return Err(PreparedQuotientNumeratorError::TileTooSmall {
                column: column_index,
                required_words: cost,
                available_words: config.max_lde_tile_words,
            });
        }
        let log_changed = current_log.is_some_and(|log| log != evaluation_log_size);
        let tile_full = !current.is_empty()
            && current_tile_words
                .checked_add(cost)
                .ok_or(PreparedQuotientNumeratorError::SizeOverflow)?
                > config.max_lde_tile_words;
        if log_changed || tile_full {
            column_batches.push(core::mem::take(&mut current));
            current_tile_words = 0;
        }
        current_log = Some(evaluation_log_size);
        current_tile_words = current_tile_words
            .checked_add(cost)
            .ok_or(PreparedQuotientNumeratorError::SizeOverflow)?;
        current.push(column_index);
    }
    if !current.is_empty() {
        column_batches.push(current);
    }

    let mut batches = Vec::with_capacity(column_batches.len());
    for batch_columns in column_batches {
        let evaluation_log_size =
            columns[batch_columns[0]].coefficient_log_size + config.log_blowup_factor;
        let coefficient_columns: Vec<_> = batch_columns
            .iter()
            .copied()
            .filter(|&column| {
                columns[column].source_kind == QuotientNumeratorSourceKind::Coefficients
            })
            .collect();
        let eval_words = pow2(evaluation_log_size)?;
        let lde_words = eval_words
            .checked_mul(coefficient_columns.len())
            .ok_or(PreparedQuotientNumeratorError::SizeOverflow)?;
        let source_local: BTreeMap<_, _> = batch_columns
            .iter()
            .copied()
            .enumerate()
            .map(|(local, column)| (column, local))
            .collect();
        let mut batch_terms = batch_columns
            .iter()
            .flat_map(|column| column_terms[*column].iter().copied())
            .collect::<Vec<_>>();
        batch_terms.sort_by_key(|&term| terms[term].group);
        let batch_offsets =
            offsets_by_group(&batch_terms, groups.len(), |&term| terms[term].group)?;
        let mut descriptors = Vec::with_capacity(batch_terms.len() * BATCH_TERM_WORDS);
        for term in batch_terms {
            descriptors.extend([
                u32::try_from(source_local[&terms[term].column])
                    .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?,
                u32::try_from(term)
                    .map_err(|_| PreparedQuotientNumeratorError::TooManyTerms(terms.len()))?,
                columns[terms[term].column].coefficient_log_size,
            ]);
        }
        batches.push(PlannedBatch {
            evaluation_log_size,
            columns: batch_columns,
            coefficient_columns,
            group_offsets: batch_offsets,
            terms: descriptors,
            lde_words,
        });
    }

    let batch_requirements = batches
        .iter()
        .map(|batch| QuotientNumeratorBatchRequirements {
            evaluation_log_size: batch.evaluation_log_size,
            source_count: batch.columns.len(),
            coefficient_count: batch.coefficient_columns.len(),
            term_count: batch.terms.len() / BATCH_TERM_WORDS,
            lde_words: batch.lde_words,
        })
        .collect::<Vec<_>>();
    let coefficient_count = batches
        .iter()
        .map(|batch| batch.coefficient_columns.len())
        .sum::<usize>();
    let max_coefficient_eval_log = batches
        .iter()
        .filter(|batch| !batch.coefficient_columns.is_empty())
        .map(|batch| batch.evaluation_log_size)
        .max();
    let requirements = QuotientNumeratorWorkspaceRequirements {
        config,
        input_sample_count: max_input_index + 1,
        term_count: terms.len(),
        groups,
        batches: batch_requirements,
        runtime_term_words: checked_mul(terms.len(), RUNTIME_TERM_WORDS)?,
        group_term_index_words: terms.len(),
        group_offset_words: point_logs.len() + 1,
        line_coefficient_words: checked_mul(terms.len(), LINE_COEFFICIENT_WORDS)?,
        term_point_words: checked_mul(terms.len(), SECURE_POINT_WORDS)?,
        batch_term_words: checked_mul(terms.len(), BATCH_TERM_WORDS)?,
        batch_group_offset_words: checked_mul(batches.len(), point_logs.len() + 1)?,
        batch_source_pointer_words: checked_mul(
            batches.iter().map(|batch| batch.columns.len()).sum(),
            POINTER_WORDS,
        )?,
        coefficient_pointer_words: checked_mul(coefficient_count, POINTER_WORDS)?,
        coefficient_size_words: coefficient_count,
        coefficient_output_pointer_words: checked_mul(coefficient_count, POINTER_WORDS)?,
        output_pointer_words: checked_mul(point_logs.len() * 4, POINTER_WORDS)?,
        output_log_size_words: point_logs.len(),
        lde_tile_words: batches
            .iter()
            .map(|batch| batch.lde_words)
            .max()
            .unwrap_or(0),
        forward_twiddle_words: match max_coefficient_eval_log {
            Some(log) => pow2(log - 1)?,
            None => 0,
        },
        max_output_size: pow2(
            point_logs
                .values()
                .copied()
                .max()
                .ok_or(PreparedQuotientNumeratorError::EmptyTerms)?,
        )?,
    };
    Ok(NumeratorPlan {
        requirements,
        terms,
        group_term_indices,
        group_offsets,
        batches,
    })
}

fn offsets_by_group<T>(
    sorted: &[T],
    group_count: usize,
    group: impl Fn(&T) -> usize,
) -> Result<Vec<u32>, PreparedQuotientNumeratorError> {
    let mut offsets = Vec::with_capacity(group_count + 1);
    let mut cursor = 0usize;
    for target in 0..group_count {
        offsets.push(
            u32::try_from(cursor)
                .map_err(|_| PreparedQuotientNumeratorError::TooManyTerms(sorted.len()))?,
        );
        while cursor < sorted.len() && group(&sorted[cursor]) == target {
            cursor += 1;
        }
    }
    offsets.push(
        u32::try_from(cursor)
            .map_err(|_| PreparedQuotientNumeratorError::TooManyTerms(sorted.len()))?,
    );
    Ok(offsets)
}

struct PreparedBatch {
    evaluation_log_size: u32,
    source_ptr_offset: usize,
    coefficient_offset: usize,
    coefficient_count: usize,
    term_offset: usize,
    group_offset: usize,
}

#[derive(Clone, Copy)]
enum PreparedNumeratorSchedule {
    LegacyBatches,
    SingleWriteCandidate,
}

enum HostDescriptor {
    U32(Vec<u32>),
    Pointers(Vec<usize>),
}

impl HostDescriptor {
    fn bytes(&self) -> (*const c_void, usize) {
        match self {
            Self::U32(values) => (values.as_ptr().cast(), values.len() * WORD_BYTES),
            Self::Pointers(values) => (
                values.as_ptr().cast(),
                values.len() * core::mem::size_of::<usize>(),
            ),
        }
    }
}

struct PendingUpload {
    destination: ArenaSlice,
    descriptor: HostDescriptor,
}

/// Stable quotient numerator launch object. [`Self::launch`] performs no host
/// transfer, allocation, synchronization, or default-stream operation.
pub struct PreparedQuotientNumeratorGraph<'a> {
    arena: &'a DeviceArena,
    requirements: QuotientNumeratorWorkspaceRequirements,
    destinations: Vec<QuotientNumeratorDestination>,
    oods_sample_points: ArenaSlice,
    oods_sample_values: ArenaSlice,
    random_coefficient: ArenaSlice,
    sample_points_destination: ArenaSlice,
    first_linear_terms_destination: ArenaSlice,
    forward_twiddles: ArenaSlice,
    runtime_terms: ArenaSlice,
    group_term_indices: ArenaSlice,
    group_offsets: ArenaSlice,
    line_coefficients: ArenaSlice,
    term_points: ArenaSlice,
    batch_terms: ArenaSlice,
    batch_group_offsets: ArenaSlice,
    batch_source_ptrs: ArenaSlice,
    output_ptrs: ArenaSlice,
    output_log_sizes: ArenaSlice,
    coefficient_ptrs: Option<ArenaSlice>,
    coefficient_sizes: Option<ArenaSlice>,
    coefficient_output_ptrs: Option<ArenaSlice>,
    batches: Vec<PreparedBatch>,
    schedule: PreparedNumeratorSchedule,
}

impl<'a> PreparedQuotientNumeratorGraph<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        arena: &'a DeviceArena,
        config: QuotientNumeratorWorkspaceConfig,
        columns: &[QuotientNumeratorColumn],
        oods_sample_points: ArenaSlice,
        oods_sample_values: ArenaSlice,
        random_coefficient: ArenaSlice,
        sample_points_destination: ArenaSlice,
        first_linear_terms_destination: ArenaSlice,
        destinations: &[QuotientNumeratorDestination],
        forward_twiddles: ArenaSlice,
        slots: &QuotientNumeratorWorkspaceSlots,
    ) -> Result<Self, PreparedQuotientNumeratorError> {
        let topology = columns
            .iter()
            .map(QuotientNumeratorColumnTopology::from)
            .collect::<Vec<_>>();
        let plan = build_plan(config, &topology)?;
        let requirements = &plan.requirements;
        if destinations.len() != requirements.groups.len() {
            return Err(PreparedQuotientNumeratorError::DestinationCountMismatch {
                expected: requirements.groups.len(),
                actual: destinations.len(),
            });
        }
        for (group, (destination, requirement)) in
            destinations.iter().zip(&requirements.groups).enumerate()
        {
            if destination.log_size != requirement.log_size {
                return Err(PreparedQuotientNumeratorError::DestinationLogMismatch {
                    group,
                    expected: requirement.log_size,
                    actual: destination.log_size,
                });
            }
        }

        let slot_requirements = requirements.arena_slot_requirements(slots)?;
        let workspace_ids: BTreeSet<_> = slot_requirements
            .iter()
            .map(|requirement| requirement.id)
            .collect();
        let context_token = arena.context().identity_token();
        let mut external_ids = BTreeSet::new();
        let external = [
            oods_sample_points,
            oods_sample_values,
            random_coefficient,
            sample_points_destination,
            first_linear_terms_destination,
            forward_twiddles,
        ]
        .into_iter()
        .chain(columns.iter().map(|column| column.source.slice()))
        .chain(
            destinations
                .iter()
                .flat_map(|destination| destination.coordinates),
        );
        for slice in external {
            if slice.context_token() != context_token {
                return Err(PreparedQuotientNumeratorError::ContextMismatch(slice.id()));
            }
            if workspace_ids.contains(&slice.id()) {
                return Err(PreparedQuotientNumeratorError::ExternalAliasesWorkspace(
                    slice.id(),
                ));
            }
            if !external_ids.insert(slice.id()) {
                return Err(PreparedQuotientNumeratorError::AliasedExternalSlot(
                    slice.id(),
                ));
            }
        }

        require_words(
            oods_sample_points,
            checked_mul(requirements.input_sample_count, SECURE_POINT_WORDS)?,
        )?;
        require_words(
            oods_sample_values,
            checked_mul(requirements.input_sample_count, SECURE_WORDS)?,
        )?;
        require_words(random_coefficient, SECURE_WORDS)?;
        require_words(
            sample_points_destination,
            checked_mul(requirements.groups.len(), SECURE_POINT_WORDS)?,
        )?;
        require_words(
            first_linear_terms_destination,
            checked_mul(requirements.groups.len(), SECURE_WORDS)?,
        )?;
        if forward_twiddles.len_words() < requirements.forward_twiddle_words {
            return Err(PreparedQuotientNumeratorError::ForwardTwiddlesTooSmall {
                required_words: requirements.forward_twiddle_words,
                actual_words: forward_twiddles.len_words(),
            });
        }
        for (column_index, column) in columns.iter().enumerate() {
            let required = match column.source {
                QuotientNumeratorColumnSource::Evaluation(_) => {
                    pow2(column.coefficient_log_size + config.log_blowup_factor)?
                }
                QuotientNumeratorColumnSource::Coefficients(_) => {
                    pow2(column.coefficient_log_size)?
                }
            };
            if column.source.slice().len_words() < required {
                return Err(PreparedQuotientNumeratorError::SourceTooSmall {
                    column: column_index,
                    required_words: required,
                    actual_words: column.source.slice().len_words(),
                });
            }
        }
        for (destination, group) in destinations.iter().zip(&requirements.groups) {
            for coordinate in destination.coordinates {
                if coordinate.len_words() < group.value_words {
                    return Err(PreparedQuotientNumeratorError::InputTooSmall {
                        slot: coordinate.id(),
                        required_words: group.value_words,
                        actual_words: coordinate.len_words(),
                    });
                }
            }
        }

        let runtime_terms = bind_slot(
            arena,
            slots.runtime_terms,
            requirements.runtime_term_words,
            1,
        )?;
        let group_term_indices = bind_slot(
            arena,
            slots.group_term_indices,
            requirements.group_term_index_words,
            1,
        )?;
        let group_offsets = bind_slot(
            arena,
            slots.group_offsets,
            requirements.group_offset_words,
            1,
        )?;
        let line_coefficients = bind_slot(
            arena,
            slots.line_coefficients,
            requirements.line_coefficient_words,
            1,
        )?;
        let term_points = bind_slot(arena, slots.term_points, requirements.term_point_words, 1)?;
        let batch_terms = bind_slot(arena, slots.batch_terms, requirements.batch_term_words, 1)?;
        let batch_group_offsets = bind_slot(
            arena,
            slots.batch_group_offsets,
            requirements.batch_group_offset_words,
            1,
        )?;
        let batch_source_ptrs = bind_slot(
            arena,
            slots.batch_source_ptrs,
            requirements.batch_source_pointer_words,
            QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
        )?;
        let output_ptrs = bind_slot(
            arena,
            slots.output_ptrs,
            requirements.output_pointer_words,
            QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
        )?;
        let output_log_sizes = bind_slot(
            arena,
            slots.output_log_sizes,
            requirements.output_log_size_words,
            1,
        )?;
        let coefficient_ptrs = bind_optional(
            arena,
            slots.coefficient_ptrs,
            requirements.coefficient_pointer_words,
            QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
        )?;
        let coefficient_sizes = bind_optional(
            arena,
            slots.coefficient_sizes,
            requirements.coefficient_size_words,
            1,
        )?;
        let coefficient_output_ptrs = bind_optional(
            arena,
            slots.coefficient_output_ptrs,
            requirements.coefficient_output_pointer_words,
            QUOTIENT_NUMERATOR_POINTER_ALIGNMENT_WORDS,
        )?;
        let lde_tile = bind_optional(arena, slots.lde_tile, requirements.lde_tile_words, 1)?;

        let runtime_words = plan
            .terms
            .iter()
            .flat_map(|term| {
                let period = term.period.unwrap_or(CirclePoint {
                    x: BaseField::from(0),
                    y: BaseField::from(0),
                });
                [
                    term.sample_index,
                    term.exponent,
                    u32::from(term.period.is_some()),
                    period.x.0,
                    period.y.0,
                ]
            })
            .collect::<Vec<_>>();
        let batch_term_words = plan
            .batches
            .iter()
            .flat_map(|batch| batch.terms.iter().copied())
            .collect::<Vec<_>>();
        let batch_group_words = plan
            .batches
            .iter()
            .flat_map(|batch| batch.group_offsets.iter().copied())
            .collect::<Vec<_>>();
        let output_pointers = (0..4)
            .flat_map(|coordinate| {
                destinations.iter().map(move |destination| {
                    destination.coordinates[coordinate].as_u32_ptr() as usize
                })
            })
            .collect::<Vec<_>>();

        let mut source_pointers = Vec::new();
        let mut coefficient_pointers = Vec::new();
        let mut coefficient_size_values = Vec::new();
        let mut coefficient_output_pointers = Vec::new();
        let mut prepared_batches = Vec::with_capacity(plan.batches.len());
        let mut term_offset = 0usize;
        let mut group_offset = 0usize;
        for batch in &plan.batches {
            let source_ptr_offset = source_pointers.len();
            let coefficient_offset = coefficient_pointers.len();
            let eval_words = pow2(batch.evaluation_log_size)?;
            let coefficient_local: BTreeMap<_, _> = batch
                .coefficient_columns
                .iter()
                .copied()
                .enumerate()
                .map(|(local, column)| (column, local))
                .collect();
            for &column in &batch.columns {
                let pointer = match columns[column].source {
                    QuotientNumeratorColumnSource::Evaluation(slice) => slice.as_u32_ptr(),
                    QuotientNumeratorColumnSource::Coefficients(_) => unsafe {
                        lde_tile
                            .expect("coefficient slot shape validated")
                            .as_u32_ptr()
                            .add(coefficient_local[&column] * eval_words)
                    },
                };
                source_pointers.push(pointer as usize);
            }
            for (local, &column) in batch.coefficient_columns.iter().enumerate() {
                coefficient_pointers.push(columns[column].source.slice().as_u32_ptr() as usize);
                coefficient_size_values.push(
                    u32::try_from(pow2(columns[column].coefficient_log_size)?)
                        .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?,
                );
                coefficient_output_pointers.push(unsafe {
                    lde_tile
                        .expect("coefficient slot shape validated")
                        .as_u32_ptr()
                        .add(local * eval_words) as usize
                });
            }
            prepared_batches.push(PreparedBatch {
                evaluation_log_size: batch.evaluation_log_size,
                source_ptr_offset,
                coefficient_offset,
                coefficient_count: batch.coefficient_columns.len(),
                term_offset,
                group_offset,
            });
            term_offset += batch.terms.len() / BATCH_TERM_WORDS;
            group_offset += batch.group_offsets.len();
        }

        let mut uploads = vec![
            upload_u32(runtime_terms, runtime_words),
            upload_u32(group_term_indices, plan.group_term_indices),
            upload_u32(group_offsets, plan.group_offsets),
            upload_u32(batch_terms, batch_term_words),
            upload_u32(batch_group_offsets, batch_group_words),
            upload_ptrs(batch_source_ptrs, source_pointers),
            upload_ptrs(output_ptrs, output_pointers),
            upload_u32(
                output_log_sizes,
                requirements
                    .groups
                    .iter()
                    .map(|group| group.log_size)
                    .collect(),
            ),
        ];
        if let (Some(ptrs), Some(sizes), Some(outputs)) =
            (coefficient_ptrs, coefficient_sizes, coefficient_output_ptrs)
        {
            uploads.extend([
                upload_ptrs(ptrs, coefficient_pointers),
                upload_u32(sizes, coefficient_size_values),
                upload_ptrs(outputs, coefficient_output_pointers),
            ]);
        }
        upload_and_sync(arena, &uploads)?;

        Ok(Self {
            arena,
            requirements: plan.requirements,
            destinations: destinations.to_vec(),
            oods_sample_points,
            oods_sample_values,
            random_coefficient,
            sample_points_destination,
            first_linear_terms_destination,
            forward_twiddles,
            runtime_terms,
            group_term_indices,
            group_offsets,
            line_coefficients,
            term_points,
            batch_terms,
            batch_group_offsets,
            batch_source_ptrs,
            output_ptrs,
            output_log_sizes,
            coefficient_ptrs,
            coefficient_sizes,
            coefficient_output_ptrs,
            batches: prepared_batches,
            schedule: PreparedNumeratorSchedule::LegacyBatches,
        })
    }

    pub fn requirements(&self) -> &QuotientNumeratorWorkspaceRequirements {
        &self.requirements
    }

    pub fn destinations(&self) -> &[QuotientNumeratorDestination] {
        &self.destinations
    }

    /// Setup-only adapter for [`super::prepared_quotient::PreparedQuotientGraph::prepare`].
    pub fn quotient_sources(&self) -> Vec<QuotientNumeratorSource> {
        self.requirements
            .groups
            .iter()
            .zip(&self.destinations)
            .map(|(group, destination)| QuotientNumeratorSource {
                constants: QuotientSampleConstants {
                    sample_point: group.shape_point,
                    first_linear_term_acc: SecureField::from(0u32),
                },
                log_size: group.log_size,
                coordinates: destination.coordinates,
            })
            .collect()
    }

    pub fn launch(&self) -> Result<(), PreparedQuotientNumeratorError> {
        let group_count = u32::try_from(self.requirements.groups.len()).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyGroups(self.requirements.groups.len())
        })?;
        let term_count = u32::try_from(self.requirements.term_count).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyTerms(self.requirements.term_count)
        })?;
        let max_output_size = u32::try_from(self.requirements.max_output_size)
            .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?;
        let stream = self.arena.context().stream_raw().as_ptr();
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_prepare_quotient_numerator_terms_on(
                self.runtime_terms.as_u32_ptr(),
                term_count,
                self.oods_sample_points.as_u32_ptr(),
                self.oods_sample_values.as_u32_ptr().cast(),
                self.random_coefficient.as_u32_ptr().cast(),
                self.term_points.as_u32_ptr(),
                self.line_coefficients.as_u32_ptr().cast(),
                stream,
            )
        };
        check_cuda("prepared_quotient_numerator_terms", code)?;
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_finalize_quotient_numerator_groups_on(
                self.group_offsets.as_u32_ptr(),
                self.group_term_indices.as_u32_ptr(),
                group_count,
                self.term_points.as_u32_ptr(),
                self.line_coefficients.as_u32_ptr().cast(),
                self.sample_points_destination.as_u32_ptr(),
                self.first_linear_terms_destination.as_u32_ptr().cast(),
                stream,
            )
        };
        check_cuda("prepared_quotient_numerator_groups", code)?;

        let output_tables = |coordinate: usize| unsafe {
            self.output_ptrs
                .as_u32_ptr()
                .cast::<*mut u32>()
                .add(coordinate * self.requirements.groups.len())
        };
        if matches!(
            self.schedule,
            PreparedNumeratorSchedule::SingleWriteCandidate
        ) {
            self.launch_single_write_candidate(group_count, max_output_size, stream)?;
            return Ok(());
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_zero_quotient_numerator_outputs_on(
                self.output_log_sizes.as_u32_ptr(),
                group_count,
                max_output_size,
                output_tables(0),
                output_tables(1),
                output_tables(2),
                output_tables(3),
                stream,
            )
        };
        check_cuda("prepared_quotient_numerator_zero", code)?;

        let pointer_table = |slice: ArenaSlice, offset: usize| unsafe {
            slice.as_u32_ptr().cast::<*const u32>().add(offset)
        };
        for batch in &self.batches {
            if batch.coefficient_count != 0 {
                let coefficient_ptrs = self.coefficient_ptrs.expect("slot shape validated");
                let coefficient_sizes = self.coefficient_sizes.expect("slot shape validated");
                let coefficient_outputs =
                    self.coefficient_output_ptrs.expect("slot shape validated");
                let code = unsafe {
                    stwo_backend_cuda_kernels::raw::stwo_lde_n2b_columns_on(
                        pointer_table(coefficient_ptrs, batch.coefficient_offset),
                        coefficient_sizes.as_u32_ptr().add(batch.coefficient_offset),
                        coefficient_outputs
                            .as_u32_ptr()
                            .cast::<*mut u32>()
                            .add(batch.coefficient_offset),
                        batch.evaluation_log_size,
                        u32::try_from(batch.coefficient_count)
                            .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?,
                        self.forward_twiddles.as_u32_ptr(),
                        u32::try_from(self.forward_twiddles.len_words())
                            .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?,
                        1u32 << (batch.evaluation_log_size - 1),
                        stream,
                    )
                };
                check_cuda("prepared_quotient_numerator_lde", code)?;
            }
            let code = unsafe {
                stwo_backend_cuda_kernels::raw::stwo_accumulate_quotient_numerator_batch_on(
                    self.batch_group_offsets
                        .as_u32_ptr()
                        .add(batch.group_offset),
                    self.batch_terms
                        .as_u32_ptr()
                        .add(batch.term_offset * BATCH_TERM_WORDS),
                    group_count,
                    max_output_size,
                    pointer_table(self.batch_source_ptrs, batch.source_ptr_offset),
                    self.line_coefficients.as_u32_ptr().cast(),
                    self.output_log_sizes.as_u32_ptr(),
                    output_tables(0),
                    output_tables(1),
                    output_tables(2),
                    output_tables(3),
                    stream,
                )
            };
            check_cuda("prepared_quotient_numerator_accumulate", code)?;
        }
        Ok(())
    }
}

fn upload_u32(destination: ArenaSlice, values: Vec<u32>) -> PendingUpload {
    PendingUpload {
        destination,
        descriptor: HostDescriptor::U32(values),
    }
}

fn upload_ptrs(destination: ArenaSlice, values: Vec<usize>) -> PendingUpload {
    PendingUpload {
        destination,
        descriptor: HostDescriptor::Pointers(values),
    }
}

fn upload_and_sync(
    arena: &DeviceArena,
    uploads: &[PendingUpload],
) -> Result<(), PreparedQuotientNumeratorError> {
    for upload in uploads {
        let (source, bytes) = upload.descriptor.bytes();
        unsafe {
            arena
                .context()
                .memcpy_h2d_async(upload.destination.as_void_ptr(), source, bytes)?;
        }
    }
    arena.context().sync()?;
    Ok(())
}

fn bind_optional(
    arena: &DeviceArena,
    id: Option<ArenaSlotId>,
    required_words: usize,
    alignment_words: usize,
) -> Result<Option<ArenaSlice>, PreparedQuotientNumeratorError> {
    match (id, required_words) {
        (Some(id), words) if words != 0 => Ok(Some(bind_slot(arena, id, words, alignment_words)?)),
        (None, 0) => Ok(None),
        _ => Err(PreparedQuotientNumeratorError::OptionalSlotShapeMismatch),
    }
}

fn bind_slot(
    arena: &DeviceArena,
    id: ArenaSlotId,
    required_words: usize,
    alignment_words: usize,
) -> Result<ArenaSlice, PreparedQuotientNumeratorError> {
    let slice = arena.bind(id)?;
    if slice.len_words() < required_words {
        return Err(PreparedQuotientNumeratorError::SlotTooSmall {
            slot: id,
            required_words,
            actual_words: slice.len_words(),
        });
    }
    if (slice.as_u32_ptr() as usize) % (alignment_words * WORD_BYTES) != 0 {
        return Err(PreparedQuotientNumeratorError::MisalignedSlot {
            slot: id,
            alignment_words,
        });
    }
    // Pooled slots may be larger than any single logical buffer; expose only
    // the logical extent so no consumer derives sizes from the pooled surplus.
    Ok(slice.truncated(required_words))
}

fn require_words(
    slice: ArenaSlice,
    required_words: usize,
) -> Result<(), PreparedQuotientNumeratorError> {
    if slice.len_words() < required_words {
        return Err(PreparedQuotientNumeratorError::InputTooSmall {
            slot: slice.id(),
            required_words,
            actual_words: slice.len_words(),
        });
    }
    Ok(())
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, PreparedQuotientNumeratorError> {
    lhs.checked_mul(rhs)
        .ok_or(PreparedQuotientNumeratorError::SizeOverflow)
}

fn pow2(log_size: u32) -> Result<usize, PreparedQuotientNumeratorError> {
    1usize
        .checked_shl(log_size)
        .ok_or(PreparedQuotientNumeratorError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;

    use super::*;

    fn slice(id: u32, words: usize) -> ArenaSlice {
        ArenaSlice::dangling_for_test(id, words)
    }

    fn sample(index: u32, multiple: u128) -> QuotientOodsSample {
        QuotientOodsSample {
            input_index: index,
            shape_point: SECURE_FIELD_CIRCLE_GEN.mul(multiple),
        }
    }

    fn config() -> QuotientNumeratorWorkspaceConfig {
        QuotientNumeratorWorkspaceConfig {
            lifting_log_size: 8,
            log_blowup_factor: 2,
            max_lde_tile_words: 1 << 9,
        }
    }

    fn topology(columns: &[QuotientNumeratorColumn]) -> Vec<QuotientNumeratorColumnTopology> {
        columns
            .iter()
            .map(QuotientNumeratorColumnTopology::from)
            .collect()
    }

    #[test]
    fn plan_preserves_alpha_order_but_batches_by_stable_evaluation_log() {
        let columns = vec![
            QuotientNumeratorColumn {
                coefficient_log_size: 6,
                source: QuotientNumeratorColumnSource::Evaluation(slice(1, 256)),
                samples: vec![sample(0, 3), sample(1, 5)],
            },
            QuotientNumeratorColumn {
                coefficient_log_size: 4,
                source: QuotientNumeratorColumnSource::Coefficients(slice(2, 16)),
                samples: vec![sample(2, 3)],
            },
        ];
        let plan = build_plan(config(), &topology(&columns)).unwrap();
        assert_eq!(
            plan.terms
                .iter()
                .map(|term| term.exponent)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            plan.batches
                .iter()
                .map(|batch| batch.evaluation_log_size)
                .collect::<Vec<_>>(),
            vec![6, 8]
        );
        assert_eq!(plan.requirements.forward_twiddle_words, 32);
        assert_eq!(plan.requirements.input_sample_count, 3);
    }

    #[test]
    fn unsampled_column_may_exceed_lifting_but_sampled_column_may_not() {
        let mut columns = vec![
            QuotientNumeratorColumnTopology {
                coefficient_log_size: 25,
                source_kind: QuotientNumeratorSourceKind::Coefficients,
                samples: vec![],
            },
            QuotientNumeratorColumnTopology {
                coefficient_log_size: 6,
                source_kind: QuotientNumeratorSourceKind::Coefficients,
                samples: vec![sample(0, 3)],
            },
        ];
        let plan = build_plan(config(), &columns).unwrap();
        assert_eq!(plan.terms.len(), 1);
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].evaluation_log_size, 8);

        columns[0].samples.push(sample(1, 5));
        assert!(matches!(
            build_plan(config(), &columns),
            Err(PreparedQuotientNumeratorError::ColumnLogTooLarge {
                column: 0,
                log_size: 25,
                maximum: 6,
            })
        ));
    }

    #[test]
    fn periodicity_and_duplicate_points_group_and_lift_exactly() {
        let repeated = sample(2, 9);
        let columns = vec![
            QuotientNumeratorColumn {
                coefficient_log_size: 4,
                source: QuotientNumeratorColumnSource::Evaluation(slice(1, 64)),
                samples: vec![sample(0, 7), repeated],
            },
            QuotientNumeratorColumn {
                coefficient_log_size: 6,
                source: QuotientNumeratorColumnSource::Evaluation(slice(2, 256)),
                samples: vec![repeated],
            },
        ];
        let plan = build_plan(config(), &topology(&columns)).unwrap();
        let repeated_group = plan
            .requirements
            .groups
            .iter()
            .find(|group| group.shape_point == repeated.shape_point)
            .unwrap();
        assert_eq!(repeated_group.log_size, 6);
        assert_eq!(repeated_group.value_words, 64);
        assert_eq!(plan.terms.len(), 4);
        assert!(plan.terms[0].period.is_some());
        assert_eq!(plan.terms[1].shape_point, sample(0, 7).shape_point);
        assert_eq!(plan.terms[2].shape_point, repeated.shape_point);
    }

    #[test]
    fn groups_count_distinct_coefficient_sources_and_zero_means_evaluation_only() {
        let shared = sample(0, 3);
        let evaluation_only = sample(3, 5);
        let columns = vec![
            QuotientNumeratorColumnTopology {
                coefficient_log_size: 4,
                source_kind: QuotientNumeratorSourceKind::Evaluation,
                samples: vec![shared, evaluation_only],
            },
            QuotientNumeratorColumnTopology {
                coefficient_log_size: 4,
                source_kind: QuotientNumeratorSourceKind::Coefficients,
                // Two terms from one column still count as one source.
                samples: vec![sample(1, 3), sample(2, 3)],
            },
            QuotientNumeratorColumnTopology {
                coefficient_log_size: 4,
                source_kind: QuotientNumeratorSourceKind::Coefficients,
                samples: vec![sample(4, 3)],
            },
            QuotientNumeratorColumnTopology {
                coefficient_log_size: 25,
                source_kind: QuotientNumeratorSourceKind::Coefficients,
                samples: vec![],
            },
        ];
        let plan = build_plan(config(), &columns).unwrap();

        let shared_group = plan
            .requirements
            .groups
            .iter()
            .find(|group| group.shape_point == shared.shape_point)
            .unwrap();
        assert_eq!(shared_group.coefficient_source_count, 2);
        let evaluation_only_group = plan
            .requirements
            .groups
            .iter()
            .find(|group| group.shape_point == evaluation_only.shape_point)
            .unwrap();
        assert_eq!(evaluation_only_group.coefficient_source_count, 0);

        for (group, requirements) in plan.requirements.groups.iter().enumerate() {
            let evaluation_only =
                plan.terms
                    .iter()
                    .filter(|term| term.group == group)
                    .all(|term| {
                        columns[term.column].source_kind == QuotientNumeratorSourceKind::Evaluation
                    });
            assert_eq!(requirements.coefficient_source_count == 0, evaluation_only);
        }
    }

    #[test]
    fn coefficient_batches_respect_the_exact_tile_ceiling() {
        let columns = (0..3)
            .map(|index| QuotientNumeratorColumn {
                coefficient_log_size: 4,
                source: QuotientNumeratorColumnSource::Coefficients(slice(index + 1, 16)),
                samples: vec![sample(index, index as u128 + 2)],
            })
            .collect::<Vec<_>>();
        let mut config = config();
        config.max_lde_tile_words = 2 * 64;
        let requirements =
            quotient_numerator_workspace_requirements(config, &topology(&columns)).unwrap();
        assert_eq!(
            requirements
                .batches
                .iter()
                .map(|batch| batch.coefficient_count)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_eq!(requirements.lde_tile_words, 128);
    }

    #[test]
    fn coefficient_slots_are_present_iff_the_plan_materializes_coefficients() {
        let columns = vec![QuotientNumeratorColumn {
            coefficient_log_size: 4,
            source: QuotientNumeratorColumnSource::Evaluation(slice(1, 64)),
            samples: vec![sample(0, 2)],
        }];
        let requirements =
            quotient_numerator_workspace_requirements(config(), &topology(&columns)).unwrap();
        let mut next = 10;
        let mut id = || {
            let result = ArenaSlotId(next);
            next += 1;
            result
        };
        let slots = QuotientNumeratorWorkspaceSlots {
            runtime_terms: id(),
            group_term_indices: id(),
            group_offsets: id(),
            line_coefficients: id(),
            term_points: id(),
            batch_terms: id(),
            batch_group_offsets: id(),
            batch_source_ptrs: id(),
            output_ptrs: id(),
            output_log_sizes: id(),
            coefficient_ptrs: None,
            coefficient_sizes: None,
            coefficient_output_ptrs: None,
            lde_tile: None,
        };
        assert_eq!(
            requirements.arena_slot_requirements(&slots).unwrap().len(),
            10
        );
    }
}
