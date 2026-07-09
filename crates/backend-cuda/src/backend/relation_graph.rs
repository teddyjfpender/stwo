//! Prepared, arena-backed CommonLookupElements execution.
//!
//! Static relation descriptors and source-pointer tables are uploaded once in
//! [`PreparedRelationGraph::prepare`]. [`PreparedRelationGraph::launch`] is the
//! single eager/capture sequence: combine/pair, fraction chain, device reduction
//! and shift, then four caller-scratch prefix scans. It allocates, transfers, and
//! synchronizes nothing.

use core::ffi::c_void;
use std::collections::BTreeSet;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;

use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const POINTER_WORDS: usize = core::mem::size_of::<*const u32>().div_ceil(WORD_BYTES);
const DESCRIPTOR_WORDS: usize = 16;
const USE_WORDS: usize = 7;
const REDUCTION_BLOCK: usize = 256;
const SECURE_FIELD_WORDS: usize = 4;
const SCAN_TEMP_OVERHEAD_WORDS: usize = 1024;
const M31_MODULUS: u64 = 0x7fff_ffff;
const LARGE_MEMORY_VALUE_ID_BASE: u32 = 0x4000_0000;
const XOR12_ROWS: u32 = 1 << 20;
pub const RELATION_POINTER_ALIGNMENT_WORDS: usize =
    core::mem::align_of::<*const u32>() / WORD_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RelationTupleKind {
    LookupWords = 0,
    MemoryAddressChunk = 1,
    MemoryBigLimbs = 2,
    MemoryBigValue = 3,
    MemorySmallLimbs = 4,
    MemorySmallValue = 5,
    BitwiseXor12 = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RelationMultiplicityKind {
    One = 0,
    Enabler = 1,
    LookupWord = 2,
    MemoryAddressChunk = 3,
    MemoryBig = 4,
    MemorySmall = 5,
    BitwiseXor12 = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationUseDescriptor {
    pub tuple_kind: RelationTupleKind,
    pub tuple_arg: u32,
    pub tuple_words: u32,
    pub relation_id: u32,
    pub multiplicity_kind: RelationMultiplicityKind,
    pub multiplicity_arg: u32,
    pub negative: bool,
}

impl RelationUseDescriptor {
    fn to_words(self) -> [u32; USE_WORDS] {
        [
            self.tuple_kind as u32,
            self.tuple_arg,
            self.tuple_words,
            self.relation_id,
            self.multiplicity_kind as u32,
            self.multiplicity_arg,
            self.negative as u32,
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationColumnDescriptor {
    pub uses: Vec<RelationUseDescriptor>,
}

impl RelationColumnDescriptor {
    fn to_words(&self) -> Result<[u32; DESCRIPTOR_WORDS], RelationGraphError> {
        if !(1..=2).contains(&self.uses.len()) {
            return Err(RelationGraphError::InvalidColumnArity(self.uses.len()));
        }
        let mut output = [0u32; DESCRIPTOR_WORDS];
        output[0] = self.uses.len() as u32;
        for (index, relation_use) in self.uses.iter().enumerate() {
            let start = 1 + index * USE_WORDS;
            output[start..start + USE_WORDS].copy_from_slice(&relation_use.to_words());
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationSourceLayout {
    /// One contiguous word-major lookup buffer.
    LookupWords { words: u32 },
    /// `[id0, mult0, id1, mult1, ...]` column pointers.
    MemoryAddress { chunks: u32 },
    /// `[value limbs..., multiplicity]` column pointers.
    MemoryBig { value_words: u32 },
    /// `[value limbs..., multiplicity]` column pointers.
    MemorySmall { value_words: u32 },
    /// One multiplicity column pointer per expanded high-bit pair.
    BitwiseXor12 { multiplicity_columns: u32 },
}

impl RelationSourceLayout {
    pub fn pointer_count(self) -> Result<usize, RelationGraphError> {
        let count = match self {
            Self::LookupWords { .. } => 1,
            Self::MemoryAddress { chunks } => chunks
                .checked_mul(2)
                .ok_or(RelationGraphError::SizeOverflow)?,
            Self::MemoryBig { value_words } | Self::MemorySmall { value_words } => value_words
                .checked_add(1)
                .ok_or(RelationGraphError::SizeOverflow)?,
            Self::BitwiseXor12 {
                multiplicity_columns,
            } => multiplicity_columns,
        };
        usize::try_from(count).map_err(|_| RelationGraphError::SizeOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationRowExtent {
    Exact {
        n_real_rows: u32,
        padded_rows: u32,
        /// Added to derived memory ids. Zero for every non-big-memory source.
        source_offset_rows: u32,
    },
    Bounded {
        observed_rows: u32,
        max_rows: u32,
        padded_capacity: u32,
    },
}

impl RelationRowExtent {
    pub fn capacity_rows(self) -> u32 {
        match self {
            Self::Exact { padded_rows, .. } => padded_rows,
            Self::Bounded {
                padded_capacity, ..
            } => padded_capacity,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationBatchProgram {
    pub source_layout: RelationSourceLayout,
    pub columns: Vec<RelationColumnDescriptor>,
    /// Empty for an absent component. Memory-big templates may have many.
    pub instances: Vec<RelationRowExtent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationKernelProgram {
    pub relation_graph_hash: u64,
    pub template_use_count: usize,
    pub max_alpha_powers: u32,
    pub batches: Vec<RelationBatchProgram>,
}

impl RelationKernelProgram {
    pub fn validate(&self) -> Result<(), RelationGraphError> {
        if self.relation_graph_hash == 0 {
            return Err(RelationGraphError::InvalidGraphHash);
        }
        if self.max_alpha_powers == 0 {
            return Err(RelationGraphError::MissingAlphaPowers);
        }
        let mut use_count = 0usize;
        for (batch_index, batch) in self.batches.iter().enumerate() {
            if batch.columns.is_empty() {
                return Err(RelationGraphError::EmptyBatch(batch_index));
            }
            let _ = batch.source_layout.pointer_count()?;
            for column in &batch.columns {
                let _ = column.to_words()?;
                use_count = use_count
                    .checked_add(column.uses.len())
                    .ok_or(RelationGraphError::SizeOverflow)?;
                for relation_use in &column.uses {
                    validate_use(batch.source_layout, *relation_use)?;
                    if relation_use.tuple_words == 0
                        || relation_use.tuple_words > self.max_alpha_powers
                    {
                        return Err(RelationGraphError::TupleWidthOutOfBounds {
                            width: relation_use.tuple_words,
                            max: self.max_alpha_powers,
                        });
                    }
                    if relation_use.relation_id == 0 || relation_use.relation_id >= 0x7fff_ffff {
                        return Err(RelationGraphError::InvalidRelationId(
                            relation_use.relation_id,
                        ));
                    }
                }
            }
            for extent in &batch.instances {
                validate_extent(batch.source_layout, *extent)?;
            }
            if matches!(batch.source_layout, RelationSourceLayout::MemoryBig { .. }) {
                let mut expected_offset = 0u32;
                for extent in &batch.instances {
                    if let RelationRowExtent::Exact {
                        padded_rows,
                        source_offset_rows,
                        ..
                    } = *extent
                    {
                        if source_offset_rows != expected_offset {
                            return Err(RelationGraphError::InvalidRowExtent);
                        }
                        expected_offset = expected_offset
                            .checked_add(padded_rows)
                            .ok_or(RelationGraphError::SizeOverflow)?;
                    }
                }
            }
        }
        if use_count != self.template_use_count {
            return Err(RelationGraphError::UseCoverageMismatch {
                expected: self.template_use_count,
                actual: use_count,
            });
        }
        Ok(())
    }

    pub fn requirements(&self) -> Result<RelationGraphRequirements, RelationGraphError> {
        relation_graph_requirements(self)
    }

    fn descriptor_words(&self) -> Result<Vec<u32>, RelationGraphError> {
        let mut output = Vec::new();
        for batch in &self.batches {
            for column in &batch.columns {
                output.extend(column.to_words()?);
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationInstanceRequirement {
    pub batch_index: usize,
    pub instance_index: usize,
    pub row_capacity: u32,
    pub source_pointer_words: usize,
    pub output_words: usize,
    pub denominator_words: usize,
    pub claimed_sum_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationGraphRequirements {
    pub descriptor_words: usize,
    pub alpha_words: usize,
    pub z_words: usize,
    pub inverse_words: usize,
    pub reduction_words: usize,
    pub scan_eval_words: usize,
    /// Conservative CUB capacity; prepare queries the native exact byte count.
    pub scan_temp_words: usize,
    pub instances: Vec<RelationInstanceRequirement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationInstanceSlots {
    pub source_pointers: ArenaSlotId,
    /// Pair numerators are written here and fraction-chained in place into the
    /// final interaction coordinates.
    pub outputs: ArenaSlotId,
    pub denominators: ArenaSlotId,
    pub claimed_sum: ArenaSlotId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationGraphSlots {
    pub descriptors: ArenaSlotId,
    pub alphas: ArenaSlotId,
    pub z: ArenaSlotId,
    pub inverse_scratch: ArenaSlotId,
    pub reduction_a: ArenaSlotId,
    pub reduction_b: ArenaSlotId,
    pub scan_eval_scratch: ArenaSlotId,
    pub scan_temp_scratch: ArenaSlotId,
    /// Active instances in deterministic `(batch, instance)` order.
    pub instances: Vec<RelationInstanceSlots>,
}

impl RelationGraphRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: &RelationGraphSlots,
    ) -> Result<Vec<RelationArenaSlotRequirement>, RelationGraphError> {
        if slots.instances.len() != self.instances.len() {
            return Err(RelationGraphError::SlotShapeMismatch {
                expected: self.instances.len(),
                actual: slots.instances.len(),
            });
        }
        let mut output = vec![
            slot_requirement(slots.descriptors, self.descriptor_words, 1),
            slot_requirement(slots.alphas, self.alpha_words, 1),
            slot_requirement(slots.z, self.z_words, 1),
            slot_requirement(
                slots.inverse_scratch,
                self.inverse_words,
                SECURE_FIELD_WORDS,
            ),
            slot_requirement(slots.reduction_a, self.reduction_words, SECURE_FIELD_WORDS),
            slot_requirement(slots.reduction_b, self.reduction_words, SECURE_FIELD_WORDS),
            slot_requirement(slots.scan_eval_scratch, self.scan_eval_words, 1),
            slot_requirement(slots.scan_temp_scratch, self.scan_temp_words, 1),
        ];
        for (requirement, instance_slots) in self.instances.iter().zip(&slots.instances) {
            output.extend([
                slot_requirement(
                    instance_slots.source_pointers,
                    requirement.source_pointer_words,
                    RELATION_POINTER_ALIGNMENT_WORDS,
                ),
                slot_requirement(instance_slots.outputs, requirement.output_words, 1),
                slot_requirement(
                    instance_slots.denominators,
                    requirement.denominator_words,
                    SECURE_FIELD_WORDS,
                ),
                slot_requirement(
                    instance_slots.claimed_sum,
                    requirement.claimed_sum_words,
                    SECURE_FIELD_WORDS,
                ),
            ]);
        }
        let mut seen = BTreeSet::new();
        for requirement in &output {
            if !seen.insert(requirement.id) {
                return Err(RelationGraphError::DuplicateSlot(requirement.id));
            }
        }
        Ok(output)
    }
}

fn slot_requirement(
    id: ArenaSlotId,
    len_words: usize,
    alignment_words: usize,
) -> RelationArenaSlotRequirement {
    RelationArenaSlotRequirement {
        id,
        len_words: len_words.max(1),
        alignment_words,
    }
}

pub fn relation_graph_requirements(
    program: &RelationKernelProgram,
) -> Result<RelationGraphRequirements, RelationGraphError> {
    program.validate()?;
    let descriptor_words = program
        .batches
        .iter()
        .map(|batch| batch.columns.len())
        .sum::<usize>()
        .checked_mul(DESCRIPTOR_WORDS)
        .ok_or(RelationGraphError::SizeOverflow)?;
    let alpha_words = usize::try_from(program.max_alpha_powers)
        .map_err(|_| RelationGraphError::SizeOverflow)?
        .checked_mul(SECURE_FIELD_WORDS)
        .ok_or(RelationGraphError::SizeOverflow)?;
    let mut max_rows = 1usize;
    let mut instances = Vec::new();
    for (batch_index, batch) in program.batches.iter().enumerate() {
        for (instance_index, extent) in batch.instances.iter().enumerate() {
            let rows = usize::try_from(extent.capacity_rows())
                .map_err(|_| RelationGraphError::SizeOverflow)?;
            max_rows = max_rows.max(rows);
            let columns = batch.columns.len();
            let coordinate_words = columns
                .checked_mul(SECURE_FIELD_WORDS)
                .and_then(|value| value.checked_mul(rows))
                .ok_or(RelationGraphError::SizeOverflow)?;
            instances.push(RelationInstanceRequirement {
                batch_index,
                instance_index,
                row_capacity: extent.capacity_rows(),
                source_pointer_words: batch
                    .source_layout
                    .pointer_count()?
                    .checked_mul(POINTER_WORDS)
                    .ok_or(RelationGraphError::SizeOverflow)?,
                output_words: coordinate_words,
                denominator_words: coordinate_words,
                claimed_sum_words: SECURE_FIELD_WORDS,
            });
        }
    }
    let inverse_words = max_rows
        .checked_mul(SECURE_FIELD_WORDS)
        .ok_or(RelationGraphError::SizeOverflow)?;
    let reduction_words = max_rows
        .div_ceil(REDUCTION_BLOCK)
        .max(1)
        .checked_mul(SECURE_FIELD_WORDS)
        .ok_or(RelationGraphError::SizeOverflow)?;
    Ok(RelationGraphRequirements {
        descriptor_words,
        alpha_words,
        z_words: SECURE_FIELD_WORDS,
        inverse_words,
        reduction_words,
        scan_eval_words: max_rows,
        scan_temp_words: max_rows
            .checked_mul(2)
            .and_then(|words| words.checked_add(SCAN_TEMP_OVERHEAD_WORDS))
            .ok_or(RelationGraphError::SizeOverflow)?,
        instances,
    })
}

fn validate_extent(
    layout: RelationSourceLayout,
    extent: RelationRowExtent,
) -> Result<(), RelationGraphError> {
    match extent {
        RelationRowExtent::Exact {
            n_real_rows,
            padded_rows,
            source_offset_rows,
        } => {
            if n_real_rows == 0
                || n_real_rows > padded_rows
                || !padded_rows.is_power_of_two()
                || u64::from(padded_rows) >= M31_MODULUS
            {
                return Err(RelationGraphError::InvalidRowExtent);
            }
            let layout_valid = match layout {
                RelationSourceLayout::MemoryBig { .. } => source_offset_rows
                    .checked_add(padded_rows)
                    .is_some_and(|end| end <= LARGE_MEMORY_VALUE_ID_BASE),
                RelationSourceLayout::MemoryAddress { chunks } => {
                    source_offset_rows == 0
                        && u64::from(chunks) * u64::from(padded_rows) < M31_MODULUS
                }
                RelationSourceLayout::BitwiseXor12 { .. } => {
                    source_offset_rows == 0
                        && n_real_rows == XOR12_ROWS
                        && padded_rows == XOR12_ROWS
                }
                _ => source_offset_rows == 0,
            };
            if !layout_valid {
                return Err(RelationGraphError::InvalidRowExtent);
            }
        }
        RelationRowExtent::Bounded {
            observed_rows,
            max_rows,
            padded_capacity,
        } => {
            if max_rows == 0
                || observed_rows > max_rows
                || max_rows > padded_capacity
                || !padded_capacity.is_power_of_two()
                || u64::from(padded_capacity) >= M31_MODULUS
            {
                return Err(RelationGraphError::InvalidRowExtent);
            }
            let layout_valid = match layout {
                RelationSourceLayout::MemoryAddress { chunks } => {
                    u64::from(chunks) * u64::from(padded_capacity) < M31_MODULUS
                }
                RelationSourceLayout::BitwiseXor12 { .. } => padded_capacity == XOR12_ROWS,
                _ => true,
            };
            if !layout_valid {
                return Err(RelationGraphError::InvalidRowExtent);
            }
        }
    }
    Ok(())
}

fn validate_use(
    layout: RelationSourceLayout,
    relation_use: RelationUseDescriptor,
) -> Result<(), RelationGraphError> {
    let tuple_valid = match (layout, relation_use.tuple_kind) {
        (RelationSourceLayout::LookupWords { words }, RelationTupleKind::LookupWords) => {
            relation_use
                .tuple_arg
                .checked_add(relation_use.tuple_words)
                .is_some_and(|end| end <= words)
        }
        (RelationSourceLayout::MemoryAddress { chunks }, RelationTupleKind::MemoryAddressChunk) => {
            relation_use.tuple_arg < chunks && relation_use.tuple_words == 3
        }
        (RelationSourceLayout::MemoryBig { value_words }, RelationTupleKind::MemoryBigLimbs)
        | (
            RelationSourceLayout::MemorySmall { value_words },
            RelationTupleKind::MemorySmallLimbs,
        ) => {
            relation_use.tuple_words == 3
                && relation_use
                    .tuple_arg
                    .checked_add(2)
                    .is_some_and(|end| end <= value_words)
        }
        (RelationSourceLayout::MemoryBig { value_words }, RelationTupleKind::MemoryBigValue)
        | (
            RelationSourceLayout::MemorySmall { value_words },
            RelationTupleKind::MemorySmallValue,
        ) => {
            relation_use.tuple_arg == 0
                && value_words
                    .checked_add(2)
                    .is_some_and(|width| relation_use.tuple_words == width)
        }
        (
            RelationSourceLayout::BitwiseXor12 {
                multiplicity_columns,
            },
            RelationTupleKind::BitwiseXor12,
        ) => relation_use.tuple_arg < multiplicity_columns && relation_use.tuple_words == 4,
        _ => false,
    };
    let multiplicity_valid = match relation_use.multiplicity_kind {
        RelationMultiplicityKind::One | RelationMultiplicityKind::Enabler => {
            relation_use.multiplicity_arg == 0
        }
        RelationMultiplicityKind::LookupWord => matches!(
            layout,
            RelationSourceLayout::LookupWords { words }
                if relation_use.multiplicity_arg < words
        ),
        RelationMultiplicityKind::MemoryAddressChunk => matches!(
            layout,
            RelationSourceLayout::MemoryAddress { chunks }
                if relation_use.multiplicity_arg < chunks
        ),
        RelationMultiplicityKind::MemoryBig => matches!(
            layout,
            RelationSourceLayout::MemoryBig { value_words }
                if relation_use.multiplicity_arg == value_words
        ),
        RelationMultiplicityKind::MemorySmall => matches!(
            layout,
            RelationSourceLayout::MemorySmall { value_words }
                if relation_use.multiplicity_arg == value_words
        ),
        RelationMultiplicityKind::BitwiseXor12 => matches!(
            layout,
            RelationSourceLayout::BitwiseXor12 { multiplicity_columns }
                if relation_use.multiplicity_arg < multiplicity_columns
        ),
    };
    if !tuple_valid || !multiplicity_valid {
        return Err(RelationGraphError::SourceLayoutMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RelationGraphError {
    InvalidGraphHash,
    MissingAlphaPowers,
    EmptyBatch(usize),
    InvalidColumnArity(usize),
    InvalidRelationId(u32),
    TupleWidthOutOfBounds {
        width: u32,
        max: u32,
    },
    UseCoverageMismatch {
        expected: usize,
        actual: usize,
    },
    SourceLayoutMismatch,
    InvalidRowExtent,
    SizeOverflow,
    SlotShapeMismatch {
        expected: usize,
        actual: usize,
    },
    DuplicateSlot(ArenaSlotId),
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
    ContextMismatch(ArenaSlotId),
    SlotTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    MisalignedSlot(ArenaSlotId),
    SourceCountMismatch {
        expected: usize,
        actual: usize,
    },
    SourceTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    UnresolvedBoundedRows {
        batch: usize,
        instance: usize,
    },
    ChallengeWidthMismatch {
        expected: usize,
        actual: usize,
    },
    ScanScratchTooSmall {
        required_bytes: usize,
        actual_bytes: usize,
    },
}

impl core::fmt::Display for RelationGraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RelationGraphError {}

impl From<ArenaError> for RelationGraphError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for RelationGraphError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

pub struct RelationChallenges<'a> {
    pub alpha_powers: &'a [SecureField],
    pub z: SecureField,
}

#[derive(Clone, Debug)]
pub struct RelationInstanceSources {
    /// Layout-defined source columns. LookupWords uses one contiguous word-major
    /// buffer; every computed layout uses one pointer per source column.
    pub columns: Vec<ArenaSlice>,
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedRelationOutput {
    pub batch_index: usize,
    pub instance_index: usize,
    pub rows: u32,
    pub columns: u32,
    pub coordinates: ArenaSlice,
    pub claimed_sum: ArenaSlice,
}

struct PreparedInstance {
    output: PreparedRelationOutput,
    descriptor_ptr: *const u32,
    source_pointers: ArenaSlice,
    denominators: ArenaSlice,
    n_real_rows: u32,
    source_offset_rows: u32,
}

/// Descriptor-complete executable relation graph. The arena borrow makes every
/// captured pointer and every claimed-sum boundary buffer structurally live.
pub struct PreparedRelationGraph<'a> {
    arena: &'a DeviceArena,
    alphas: ArenaSlice,
    z: ArenaSlice,
    inverse_scratch: ArenaSlice,
    reduction_a: ArenaSlice,
    reduction_b: ArenaSlice,
    scan_eval_scratch: ArenaSlice,
    scan_temp_scratch: ArenaSlice,
    scan_temp_bytes: usize,
    instances: Vec<PreparedInstance>,
}

impl<'a> PreparedRelationGraph<'a> {
    /// Bind all caller-owned slots and source buffers, upload immutable
    /// descriptors/pointer tables plus the first challenge values, then perform
    /// the sole setup drain. No setup transfer is captured by `launch`.
    pub fn prepare(
        arena: &'a DeviceArena,
        program: &RelationKernelProgram,
        slots: &RelationGraphSlots,
        sources: &[RelationInstanceSources],
        challenges: RelationChallenges<'_>,
    ) -> Result<Self, RelationGraphError> {
        program.validate()?;
        let requirements = program.requirements()?;
        requirements.arena_slot_requirements(slots)?;
        if sources.len() != requirements.instances.len() {
            return Err(RelationGraphError::SourceCountMismatch {
                expected: requirements.instances.len(),
                actual: sources.len(),
            });
        }
        if challenges.alpha_powers.len() < program.max_alpha_powers as usize {
            return Err(RelationGraphError::ChallengeWidthMismatch {
                expected: program.max_alpha_powers as usize,
                actual: challenges.alpha_powers.len(),
            });
        }
        for requirement in &requirements.instances {
            if matches!(
                program.batches[requirement.batch_index].instances[requirement.instance_index],
                RelationRowExtent::Bounded { .. }
            ) {
                return Err(RelationGraphError::UnresolvedBoundedRows {
                    batch: requirement.batch_index,
                    instance: requirement.instance_index,
                });
            }
        }

        let descriptors = bind_slot(arena, slots.descriptors, requirements.descriptor_words, 1)?;
        let alphas = bind_slot(arena, slots.alphas, requirements.alpha_words, 1)?;
        let z = bind_slot(arena, slots.z, requirements.z_words, 1)?;
        let inverse_scratch = bind_slot(
            arena,
            slots.inverse_scratch,
            requirements.inverse_words,
            SECURE_FIELD_WORDS,
        )?;
        let reduction_a = bind_slot(
            arena,
            slots.reduction_a,
            requirements.reduction_words,
            SECURE_FIELD_WORDS,
        )?;
        let reduction_b = bind_slot(
            arena,
            slots.reduction_b,
            requirements.reduction_words,
            SECURE_FIELD_WORDS,
        )?;
        let scan_eval_scratch = bind_slot(
            arena,
            slots.scan_eval_scratch,
            requirements.scan_eval_words,
            1,
        )?;
        let scan_temp_scratch = bind_slot(
            arena,
            slots.scan_temp_scratch,
            requirements.scan_temp_words,
            1,
        )?;
        let exact_scan_temp_bytes = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_relation_scan_temp_bytes(
                u32::try_from(requirements.scan_eval_words)
                    .map_err(|_| RelationGraphError::SizeOverflow)?,
            )
        };
        if exact_scan_temp_bytes > scan_temp_scratch.len_bytes() {
            return Err(RelationGraphError::ScanScratchTooSmall {
                required_bytes: exact_scan_temp_bytes,
                actual_bytes: scan_temp_scratch.len_bytes(),
            });
        }

        let descriptor_words = program.descriptor_words()?;
        let alpha_words =
            secure_words(&challenges.alpha_powers[..program.max_alpha_powers as usize]);
        let z_words = secure_words(core::slice::from_ref(&challenges.z));
        let mut uploads = vec![
            PendingUpload::u32(descriptors, descriptor_words),
            PendingUpload::u32(alphas, alpha_words),
            PendingUpload::u32(z, z_words),
        ];

        let context_token = arena.context().identity_token();
        let mut prepared = Vec::with_capacity(requirements.instances.len());
        let mut descriptor_column_offset = Vec::with_capacity(program.batches.len());
        let mut column_offset = 0usize;
        for batch in &program.batches {
            descriptor_column_offset.push(column_offset);
            column_offset = column_offset
                .checked_add(batch.columns.len())
                .ok_or(RelationGraphError::SizeOverflow)?;
        }
        for ((requirement, instance_slots), source) in requirements
            .instances
            .iter()
            .zip(&slots.instances)
            .zip(sources)
        {
            let batch = &program.batches[requirement.batch_index];
            let RelationRowExtent::Exact {
                n_real_rows,
                padded_rows,
                source_offset_rows,
            } = batch.instances[requirement.instance_index]
            else {
                unreachable!("bounded extents rejected")
            };
            let expected_sources = batch.source_layout.pointer_count()?;
            if source.columns.len() != expected_sources {
                return Err(RelationGraphError::SourceCountMismatch {
                    expected: expected_sources,
                    actual: source.columns.len(),
                });
            }
            validate_sources(batch.source_layout, source, padded_rows, context_token)?;
            let source_pointers = bind_slot(
                arena,
                instance_slots.source_pointers,
                requirement.source_pointer_words,
                RELATION_POINTER_ALIGNMENT_WORDS,
            )?;
            let outputs = bind_slot(arena, instance_slots.outputs, requirement.output_words, 1)?;
            let denominators = bind_slot(
                arena,
                instance_slots.denominators,
                requirement.denominator_words,
                SECURE_FIELD_WORDS,
            )?;
            let claimed_sum = bind_slot(
                arena,
                instance_slots.claimed_sum,
                requirement.claimed_sum_words,
                SECURE_FIELD_WORDS,
            )?;
            uploads.push(PendingUpload::pointers(
                source_pointers,
                source
                    .columns
                    .iter()
                    .map(|column| column.as_u32_ptr() as usize)
                    .collect(),
            ));
            let descriptor_word_offset = descriptor_column_offset[requirement.batch_index]
                .checked_mul(DESCRIPTOR_WORDS)
                .ok_or(RelationGraphError::SizeOverflow)?;
            let descriptor_ptr = unsafe {
                descriptors
                    .as_u32_ptr()
                    .add(descriptor_word_offset)
                    .cast_const()
            };
            prepared.push(PreparedInstance {
                output: PreparedRelationOutput {
                    batch_index: requirement.batch_index,
                    instance_index: requirement.instance_index,
                    rows: padded_rows,
                    columns: u32::try_from(batch.columns.len())
                        .map_err(|_| RelationGraphError::SizeOverflow)?,
                    coordinates: outputs,
                    claimed_sum,
                },
                descriptor_ptr,
                source_pointers,
                denominators,
                n_real_rows,
                source_offset_rows,
            });
        }

        upload_and_sync(arena, &uploads)?;
        Ok(Self {
            arena,
            alphas,
            z,
            inverse_scratch,
            reduction_a,
            reduction_b,
            scan_eval_scratch,
            scan_temp_scratch,
            scan_temp_bytes: exact_scan_temp_bytes,
            instances: prepared,
        })
    }

    /// Allocation/copy/sync/default-stream-free sequence shared by eager mode and
    /// graph capture.
    pub fn launch(&self) -> Result<(), RelationGraphError> {
        let stream = self.arena.context().stream_raw().as_ptr();
        for instance in &self.instances {
            let rows = instance.output.rows;
            let columns = instance.output.columns;
            check_cuda("relation_pairs_on", unsafe {
                stwo_backend_cuda_kernels::raw::stwo_relation_pairs_on(
                    instance.source_pointers.as_u32_ptr().cast(),
                    u32::try_from(instance.source_pointers.len_words() / POINTER_WORDS)
                        .map_err(|_| RelationGraphError::SizeOverflow)?,
                    rows,
                    instance.n_real_rows,
                    instance.source_offset_rows,
                    instance.descriptor_ptr,
                    columns,
                    self.alphas.as_u32_ptr().cast_const(),
                    u32::try_from(self.alphas.len_words() / SECURE_FIELD_WORDS)
                        .map_err(|_| RelationGraphError::SizeOverflow)?,
                    self.z.as_u32_ptr().cast_const(),
                    instance.output.coordinates.as_u32_ptr(),
                    instance.denominators.as_u32_ptr(),
                    stream,
                )
            })?;
            check_cuda("relation_fraction_chain_on", unsafe {
                stwo_backend_cuda_kernels::raw::stwo_relation_fraction_chain_on(
                    instance.output.coordinates.as_u32_ptr(),
                    instance.denominators.as_u32_ptr().cast_const(),
                    self.inverse_scratch.as_u32_ptr(),
                    rows,
                    columns,
                    stream,
                )
            })?;
            let inverse_rows = M31::from_u32_unchecked(rows).inverse().0;
            check_cuda("relation_reduce_shift_on", unsafe {
                stwo_backend_cuda_kernels::raw::stwo_relation_reduce_shift_on(
                    instance.output.coordinates.as_u32_ptr(),
                    rows,
                    columns,
                    self.reduction_a.as_u32_ptr(),
                    self.reduction_b.as_u32_ptr(),
                    u32::try_from(self.reduction_a.len_words() / SECURE_FIELD_WORDS)
                        .map_err(|_| RelationGraphError::SizeOverflow)?,
                    instance.output.claimed_sum.as_u32_ptr(),
                    inverse_rows,
                    stream,
                )
            })?;
            check_cuda("relation_prefix_scan_on", unsafe {
                stwo_backend_cuda_kernels::raw::stwo_relation_prefix_scan_on(
                    instance.output.coordinates.as_u32_ptr(),
                    rows,
                    columns,
                    self.scan_eval_scratch.as_u32_ptr(),
                    self.scan_temp_scratch.as_void_ptr(),
                    self.scan_temp_bytes,
                    stream,
                )
            })?;
        }
        Ok(())
    }

    pub fn outputs(&self) -> impl ExactSizeIterator<Item = PreparedRelationOutput> + '_ {
        self.instances.iter().map(|instance| instance.output)
    }

    /// Update only transcript-derived challenge words between proof replays. This
    /// boundary is deliberately outside `launch` and must complete before capture
    /// or graph replay begins.
    pub fn upload_challenges_at_transcript_boundary(
        &self,
        challenges: RelationChallenges<'_>,
    ) -> Result<(), RelationGraphError> {
        let expected = self.alphas.len_words() / SECURE_FIELD_WORDS;
        if challenges.alpha_powers.len() < expected {
            return Err(RelationGraphError::ChallengeWidthMismatch {
                expected,
                actual: challenges.alpha_powers.len(),
            });
        }
        let uploads = [
            PendingUpload::u32(
                self.alphas,
                secure_words(&challenges.alpha_powers[..expected]),
            ),
            PendingUpload::u32(self.z, secure_words(core::slice::from_ref(&challenges.z))),
        ];
        upload_and_sync(self.arena, &uploads)
    }

    /// The only relation-side D2H: four words per active trace, immediately when
    /// the corresponding interaction claims must be mixed into the transcript.
    pub fn read_claimed_sums_at_transcript_boundary(
        &self,
    ) -> Result<Vec<SecureField>, RelationGraphError> {
        let mut words = vec![[0u32; SECURE_FIELD_WORDS]; self.instances.len()];
        for (instance, output) in self.instances.iter().zip(&mut words) {
            unsafe {
                self.arena.context().memcpy_d2h_async(
                    output.as_mut_ptr().cast(),
                    instance.output.claimed_sum.as_void_ptr().cast_const(),
                    SECURE_FIELD_WORDS * WORD_BYTES,
                )?;
            }
        }
        self.arena.context().sync()?;
        Ok(words
            .into_iter()
            .map(|coordinates| {
                SecureField::from_m31_array(coordinates.map(M31::from_u32_unchecked))
            })
            .collect())
    }
}

enum HostDescriptor {
    U32(Vec<u32>),
    Pointers(Vec<usize>),
}

struct PendingUpload {
    destination: ArenaSlice,
    descriptor: HostDescriptor,
}

impl PendingUpload {
    fn u32(destination: ArenaSlice, values: Vec<u32>) -> Self {
        Self {
            destination,
            descriptor: HostDescriptor::U32(values),
        }
    }

    fn pointers(destination: ArenaSlice, values: Vec<usize>) -> Self {
        Self {
            destination,
            descriptor: HostDescriptor::Pointers(values),
        }
    }

    fn bytes(&self) -> (*const c_void, usize) {
        match &self.descriptor {
            HostDescriptor::U32(values) => (
                values.as_ptr().cast(),
                values.len().saturating_mul(WORD_BYTES),
            ),
            HostDescriptor::Pointers(values) => (
                values.as_ptr().cast(),
                values.len().saturating_mul(core::mem::size_of::<usize>()),
            ),
        }
    }
}

fn secure_words(values: &[SecureField]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|value| value.to_m31_array().map(|coordinate| coordinate.0))
        .collect()
}

fn upload_and_sync(
    arena: &DeviceArena,
    uploads: &[PendingUpload],
) -> Result<(), RelationGraphError> {
    for upload in uploads {
        let (source, bytes) = upload.bytes();
        unsafe {
            arena
                .context()
                .memcpy_h2d_async(upload.destination.as_void_ptr(), source, bytes)?;
        }
    }
    arena.context().sync()?;
    Ok(())
}

fn validate_sources(
    layout: RelationSourceLayout,
    sources: &RelationInstanceSources,
    rows: u32,
    context_token: core::ptr::NonNull<c_void>,
) -> Result<(), RelationGraphError> {
    let rows = usize::try_from(rows).map_err(|_| RelationGraphError::SizeOverflow)?;
    for source in &sources.columns {
        if source.context_token() != context_token {
            return Err(RelationGraphError::ContextMismatch(source.id()));
        }
    }
    match layout {
        RelationSourceLayout::LookupWords { words } => {
            let required_words = usize::try_from(words)
                .map_err(|_| RelationGraphError::SizeOverflow)?
                .checked_mul(rows)
                .ok_or(RelationGraphError::SizeOverflow)?;
            let source = sources.columns[0];
            if source.len_words() < required_words {
                return Err(RelationGraphError::SourceTooSmall {
                    slot: source.id(),
                    required_words,
                    actual_words: source.len_words(),
                });
            }
        }
        _ => {
            for source in &sources.columns {
                if source.len_words() < rows {
                    return Err(RelationGraphError::SourceTooSmall {
                        slot: source.id(),
                        required_words: rows,
                        actual_words: source.len_words(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn bind_slot(
    arena: &DeviceArena,
    id: ArenaSlotId,
    required_words: usize,
    alignment_words: usize,
) -> Result<ArenaSlice, RelationGraphError> {
    let slice = arena.bind(id)?;
    if slice.len_words() < required_words.max(1) {
        return Err(RelationGraphError::SlotTooSmall {
            slot: id,
            required_words: required_words.max(1),
            actual_words: slice.len_words(),
        });
    }
    if (slice.as_u32_ptr() as usize) % (alignment_words * WORD_BYTES) != 0 {
        return Err(RelationGraphError::MisalignedSlot(id));
    }
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup_use(tuple_arg: u32, tuple_words: u32) -> RelationUseDescriptor {
        RelationUseDescriptor {
            tuple_kind: RelationTupleKind::LookupWords,
            tuple_arg,
            tuple_words,
            relation_id: 1,
            multiplicity_kind: RelationMultiplicityKind::One,
            multiplicity_arg: 0,
            negative: false,
        }
    }

    fn sample_program() -> RelationKernelProgram {
        RelationKernelProgram {
            relation_graph_hash: 1,
            template_use_count: 3,
            max_alpha_powers: 6,
            batches: vec![
                RelationBatchProgram {
                    source_layout: RelationSourceLayout::LookupWords { words: 8 },
                    columns: vec![RelationColumnDescriptor {
                        uses: vec![lookup_use(0, 3), lookup_use(3, 2)],
                    }],
                    instances: vec![RelationRowExtent::Exact {
                        n_real_rows: 6,
                        padded_rows: 8,
                        source_offset_rows: 0,
                    }],
                },
                RelationBatchProgram {
                    source_layout: RelationSourceLayout::MemoryBig { value_words: 4 },
                    columns: vec![RelationColumnDescriptor {
                        uses: vec![RelationUseDescriptor {
                            tuple_kind: RelationTupleKind::MemoryBigValue,
                            tuple_arg: 0,
                            tuple_words: 6,
                            relation_id: 2,
                            multiplicity_kind: RelationMultiplicityKind::MemoryBig,
                            multiplicity_arg: 4,
                            negative: true,
                        }],
                    }],
                    instances: vec![RelationRowExtent::Bounded {
                        observed_rows: 3,
                        max_rows: 7,
                        padded_capacity: 8,
                    }],
                },
            ],
        }
    }

    fn sample_slots() -> RelationGraphSlots {
        let mut next = 0u32;
        let mut id = || {
            let result = ArenaSlotId(next);
            next += 1;
            result
        };
        RelationGraphSlots {
            descriptors: id(),
            alphas: id(),
            z: id(),
            inverse_scratch: id(),
            reduction_a: id(),
            reduction_b: id(),
            scan_eval_scratch: id(),
            scan_temp_scratch: id(),
            instances: (0..2)
                .map(|_| RelationInstanceSlots {
                    source_pointers: id(),
                    outputs: id(),
                    denominators: id(),
                    claimed_sum: id(),
                })
                .collect(),
        }
    }

    #[test]
    fn requirements_cover_every_arena_buffer_without_cuda() {
        let requirements = relation_graph_requirements(&sample_program()).unwrap();
        assert_eq!(requirements.descriptor_words, 2 * DESCRIPTOR_WORDS);
        assert_eq!(requirements.alpha_words, 6 * SECURE_FIELD_WORDS);
        assert_eq!(requirements.z_words, SECURE_FIELD_WORDS);
        assert_eq!(requirements.inverse_words, 8 * SECURE_FIELD_WORDS);
        assert_eq!(requirements.reduction_words, SECURE_FIELD_WORDS);
        assert_eq!(requirements.scan_eval_words, 8);
        assert_eq!(requirements.scan_temp_words, 16 + SCAN_TEMP_OVERHEAD_WORDS);
        assert_eq!(requirements.instances.len(), 2);
        assert_eq!(
            requirements.instances[0].output_words,
            8 * SECURE_FIELD_WORDS
        );
        assert_eq!(
            requirements.instances[1].source_pointer_words,
            5 * POINTER_WORDS
        );

        let slots = sample_slots();
        let slot_requirements = requirements.arena_slot_requirements(&slots).unwrap();
        assert_eq!(slot_requirements.len(), 8 + 2 * 4);
        assert_eq!(
            slot_requirements[8].alignment_words,
            RELATION_POINTER_ALIGNMENT_WORDS
        );
    }

    #[test]
    fn graph_coverage_and_source_bounds_fail_closed() {
        let mut program = sample_program();
        program.template_use_count += 1;
        assert_eq!(
            program.validate(),
            Err(RelationGraphError::UseCoverageMismatch {
                expected: 4,
                actual: 3,
            })
        );

        assert_eq!(
            validate_use(
                RelationSourceLayout::LookupWords { words: 8 },
                lookup_use(7, 2),
            ),
            Err(RelationGraphError::SourceLayoutMismatch)
        );
        let big = RelationUseDescriptor {
            tuple_kind: RelationTupleKind::MemoryBigLimbs,
            tuple_arg: 3,
            tuple_words: 3,
            relation_id: 1,
            multiplicity_kind: RelationMultiplicityKind::One,
            multiplicity_arg: 0,
            negative: false,
        };
        assert_eq!(
            validate_use(RelationSourceLayout::MemoryBig { value_words: 4 }, big),
            Err(RelationGraphError::SourceLayoutMismatch)
        );
        assert_eq!(
            validate_use(
                RelationSourceLayout::BitwiseXor12 {
                    multiplicity_columns: 16,
                },
                RelationUseDescriptor {
                    tuple_kind: RelationTupleKind::BitwiseXor12,
                    tuple_arg: 16,
                    tuple_words: 4,
                    multiplicity_kind: RelationMultiplicityKind::BitwiseXor12,
                    multiplicity_arg: 16,
                    ..big
                },
            ),
            Err(RelationGraphError::SourceLayoutMismatch)
        );
    }

    #[test]
    fn duplicate_caller_slots_are_rejected() {
        let requirements = sample_program().requirements().unwrap();
        let mut slots = sample_slots();
        slots.instances[1].outputs = slots.instances[0].outputs;
        assert!(matches!(
            requirements.arena_slot_requirements(&slots),
            Err(RelationGraphError::DuplicateSlot(_))
        ));
    }
}
