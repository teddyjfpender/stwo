//! Prepared multiplicity feeds from a recorded witness's resident sub-input buffer.
//!
//! Setup validates the flat descriptor ABI, binds every arena address, and uploads
//! descriptors, LUTs, and pointer tables once. [`PreparedWitnessFeedGraph::launch`]
//! is one explicit-stream kernel enqueue with no allocation, copy, or synchronization.

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

use super::exec_context::{
    ArenaError, ArenaSlice, ArenaSlotId, CudaLaunchContext, CudaRuntimeError, DeviceArena,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const POINTER_WORDS: usize = core::mem::size_of::<*mut u32>().div_ceil(WORD_BYTES);

pub const WITNESS_FEED_DESCRIPTOR_WORDS: usize = 14;
pub const WITNESS_FEED_MAX_TUPLE_WORDS: usize = 5;
pub const WITNESS_FEED_NO_LUT: u32 = u32::MAX;
pub const WITNESS_FEED_POINTER_ALIGNMENT_WORDS: usize =
    core::mem::align_of::<*mut u32>() / WORD_BYTES;

/// One slot request for merging a feed graph into the proof-wide arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WitnessFeedArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

/// Pure, exact geometry for one prepared feed launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessFeedWorkspaceRequirements {
    pub row_count: usize,
    pub sub_words_per_row: usize,
    pub source_words: usize,
    pub descriptor_words: usize,
    pub descriptor_count: usize,
    pub lut_words: Vec<usize>,
    pub lut_pointer_words: usize,
    pub multiplicity_words: Vec<usize>,
    pub multiplicity_pointer_words: usize,
}

/// Arena identities used by one feed graph. Multiplicity slots are supplied by
/// the caller so several producer graphs can accumulate into the same targets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessFeedWorkspaceSlots {
    pub descriptors: ArenaSlotId,
    pub lut_tables: Vec<ArenaSlotId>,
    pub lut_pointers: ArenaSlotId,
    pub multiplicity_destinations: Vec<ArenaSlotId>,
    pub multiplicity_pointers: ArenaSlotId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessFeedClearWorkspaceRequirements {
    pub destination_words: Vec<usize>,
    pub destination_pointer_words: usize,
    pub destination_length_words: usize,
    pub max_destination_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WitnessFeedClearWorkspaceSlots {
    pub destination_pointers: ArenaSlotId,
    pub destination_lengths: ArenaSlotId,
}

impl WitnessFeedClearWorkspaceRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: WitnessFeedClearWorkspaceSlots,
    ) -> Result<[WitnessFeedArenaSlotRequirement; 2], PreparedWitnessFeedError> {
        if slots.destination_pointers == slots.destination_lengths {
            return Err(PreparedWitnessFeedError::DuplicateSlot(
                slots.destination_pointers,
            ));
        }
        Ok([
            pointers(slots.destination_pointers, self.destination_pointer_words),
            words(slots.destination_lengths, self.destination_length_words),
        ])
    }
}

impl WitnessFeedWorkspaceRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: &WitnessFeedWorkspaceSlots,
    ) -> Result<Vec<WitnessFeedArenaSlotRequirement>, PreparedWitnessFeedError> {
        check_count("lut_tables", self.lut_words.len(), slots.lut_tables.len())?;
        check_count(
            "multiplicity_destinations",
            self.multiplicity_words.len(),
            slots.multiplicity_destinations.len(),
        )?;
        let mut result = vec![
            words(slots.descriptors, self.descriptor_words),
            pointers(slots.lut_pointers, self.lut_pointer_words),
            pointers(slots.multiplicity_pointers, self.multiplicity_pointer_words),
        ];
        result.extend(
            slots
                .lut_tables
                .iter()
                .zip(&self.lut_words)
                .map(|(&id, &len_words)| words(id, len_words)),
        );
        result.extend(
            slots
                .multiplicity_destinations
                .iter()
                .zip(&self.multiplicity_words)
                .map(|(&id, &len_words)| words(id, len_words)),
        );
        ensure_distinct(result.iter().map(|entry| entry.id))?;
        Ok(result)
    }
}

fn words(id: ArenaSlotId, len_words: usize) -> WitnessFeedArenaSlotRequirement {
    WitnessFeedArenaSlotRequirement {
        id,
        len_words,
        alignment_words: 1,
    }
}

fn pointers(id: ArenaSlotId, len_words: usize) -> WitnessFeedArenaSlotRequirement {
    WitnessFeedArenaSlotRequirement {
        id,
        len_words,
        alignment_words: WITNESS_FEED_POINTER_ALIGNMENT_WORDS,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedWitnessFeedError {
    CudaUnavailable,
    ZeroRows,
    ZeroSubWords,
    RowCountOverflow,
    SizeOverflow,
    NoDescriptors,
    DescriptorStrideMismatch(usize),
    NoMultiplicityDestinations,
    EmptyLut(usize),
    EmptyMultiplicityDestination(usize),
    InvalidDescriptorKind {
        descriptor: usize,
        kind: u32,
    },
    InvalidTupleWidth {
        descriptor: usize,
        width: u32,
    },
    InvalidTupleBitWidth {
        descriptor: usize,
        word: usize,
        bits: u32,
    },
    NonzeroUnusedTupleBit {
        descriptor: usize,
        word: usize,
        bits: u32,
    },
    SourceRangeOverflow {
        descriptor: usize,
    },
    SourceRangeOutOfBounds {
        descriptor: usize,
        end_word: usize,
        sub_words_per_row: usize,
    },
    EmptyTable {
        descriptor: usize,
    },
    DestinationIndexOutOfRange {
        descriptor: usize,
        index: u32,
        count: usize,
    },
    DestinationTooSmall {
        descriptor: usize,
        index: usize,
        required_words: usize,
        actual_words: usize,
    },
    DestinationLengthNotMultiple {
        descriptor: usize,
        index: usize,
        destination_words: usize,
        table_size: usize,
    },
    DestinationTableSizeMismatch {
        descriptor: usize,
        index: usize,
        expected_table_size: usize,
        actual_table_size: usize,
    },
    UnusedDestination(usize),
    LutIndexOutOfRange {
        descriptor: usize,
        index: u32,
        count: usize,
    },
    LutSizeMismatch {
        descriptor: usize,
        index: usize,
        expected_words: usize,
        actual_words: usize,
    },
    LutValueOutOfRange {
        descriptor: usize,
        index: usize,
        offset: usize,
        value: u32,
        table_size: usize,
    },
    UnusedLut(usize),
    /// A dependent tuple such as `(a, b, a ^ b)` cannot be represented by the
    /// current fold ABI. Silently bounding its 24-bit fold to an xor8 LUT would
    /// discard valid rows.
    UnsupportedDependentTupleMapping {
        descriptor: usize,
        tuple_bits: u32,
        lut_words: usize,
    },
    InvalidMemoryDescriptor {
        descriptor: usize,
    },
    InvalidXorDescriptor {
        descriptor: usize,
        kind: u32,
    },
    AliasedMemoryDestinations {
        descriptor: usize,
        index: u32,
    },
    SlotShapeMismatch {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    DuplicateSlot(ArenaSlotId),
    SlotSizeMismatch {
        slot: ArenaSlotId,
        expected_words: usize,
        actual_words: usize,
    },
    SlotMisaligned(ArenaSlotId),
    ContextMismatch(ArenaSlotId),
    ConflictingDestination(ArenaSlotId),
    KernelLaunchFailed,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedWitnessFeedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "prepared witness feed rejected: {self:?}")
    }
}

impl std::error::Error for PreparedWitnessFeedError {}

impl From<ArenaError> for PreparedWitnessFeedError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedWitnessFeedError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

/// Validate the complete descriptor/LUT/destination ABI without requiring CUDA.
pub fn witness_feed_workspace_requirements(
    row_count: usize,
    sub_words_per_row: usize,
    descriptors: &[u32],
    luts: &[Vec<u32>],
    multiplicity_words: &[usize],
) -> Result<WitnessFeedWorkspaceRequirements, PreparedWitnessFeedError> {
    if row_count == 0 {
        return Err(PreparedWitnessFeedError::ZeroRows);
    }
    u32::try_from(row_count).map_err(|_| PreparedWitnessFeedError::RowCountOverflow)?;
    if sub_words_per_row == 0 {
        return Err(PreparedWitnessFeedError::ZeroSubWords);
    }
    let source_words = row_count
        .checked_mul(sub_words_per_row)
        .ok_or(PreparedWitnessFeedError::SizeOverflow)?;
    if descriptors.is_empty() {
        return Err(PreparedWitnessFeedError::NoDescriptors);
    }
    if !descriptors
        .len()
        .is_multiple_of(WITNESS_FEED_DESCRIPTOR_WORDS)
    {
        return Err(PreparedWitnessFeedError::DescriptorStrideMismatch(
            descriptors.len(),
        ));
    }
    u32::try_from(descriptors.len() / WITNESS_FEED_DESCRIPTOR_WORDS)
        .map_err(|_| PreparedWitnessFeedError::SizeOverflow)?;
    if multiplicity_words.is_empty() {
        return Err(PreparedWitnessFeedError::NoMultiplicityDestinations);
    }
    if let Some(index) = luts.iter().position(Vec::is_empty) {
        return Err(PreparedWitnessFeedError::EmptyLut(index));
    }
    if let Some(index) = multiplicity_words.iter().position(|&words| words == 0) {
        return Err(PreparedWitnessFeedError::EmptyMultiplicityDestination(
            index,
        ));
    }

    let mut used_luts = vec![false; luts.len()];
    let mut used_destinations = vec![false; multiplicity_words.len()];
    let mut destination_table_sizes = vec![None; multiplicity_words.len()];
    for (descriptor, entry) in descriptors
        .chunks_exact(WITNESS_FEED_DESCRIPTOR_WORDS)
        .enumerate()
    {
        validate_descriptor(
            descriptor,
            entry,
            sub_words_per_row,
            luts,
            multiplicity_words,
            &mut used_luts,
            &mut used_destinations,
            &mut destination_table_sizes,
        )?;
    }
    if let Some(index) = used_luts.iter().position(|&used| !used) {
        return Err(PreparedWitnessFeedError::UnusedLut(index));
    }
    if let Some(index) = used_destinations.iter().position(|&used| !used) {
        return Err(PreparedWitnessFeedError::UnusedDestination(index));
    }

    Ok(WitnessFeedWorkspaceRequirements {
        row_count,
        sub_words_per_row,
        source_words,
        descriptor_words: descriptors.len(),
        descriptor_count: descriptors.len() / WITNESS_FEED_DESCRIPTOR_WORDS,
        lut_words: luts.iter().map(Vec::len).collect(),
        lut_pointer_words: pointer_words(luts.len())?,
        multiplicity_words: multiplicity_words.to_vec(),
        multiplicity_pointer_words: pointer_words(multiplicity_words.len())?,
    })
}

pub fn witness_feed_clear_workspace_requirements(
    destination_words: &[usize],
) -> Result<WitnessFeedClearWorkspaceRequirements, PreparedWitnessFeedError> {
    if destination_words.is_empty() {
        return Err(PreparedWitnessFeedError::NoMultiplicityDestinations);
    }
    if let Some(index) = destination_words.iter().position(|&words| words == 0) {
        return Err(PreparedWitnessFeedError::EmptyMultiplicityDestination(
            index,
        ));
    }
    let max_destination_words = *destination_words.iter().max().unwrap();
    u32::try_from(destination_words.len()).map_err(|_| PreparedWitnessFeedError::SizeOverflow)?;
    if destination_words.len() > u16::MAX as usize {
        return Err(PreparedWitnessFeedError::SizeOverflow);
    }
    u32::try_from(max_destination_words).map_err(|_| PreparedWitnessFeedError::SizeOverflow)?;
    Ok(WitnessFeedClearWorkspaceRequirements {
        destination_words: destination_words.to_vec(),
        destination_pointer_words: pointer_words(destination_words.len())?,
        destination_length_words: destination_words.len(),
        max_destination_words,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_descriptor(
    descriptor: usize,
    entry: &[u32],
    sub_words_per_row: usize,
    luts: &[Vec<u32>],
    multiplicity_words: &[usize],
    used_luts: &mut [bool],
    used_destinations: &mut [bool],
    destination_table_sizes: &mut [Option<usize>],
) -> Result<(), PreparedWitnessFeedError> {
    let width = entry[1];
    if !(1..=WITNESS_FEED_MAX_TUPLE_WORDS as u32).contains(&width) {
        return Err(PreparedWitnessFeedError::InvalidTupleWidth { descriptor, width });
    }
    let word_base = entry[0] as usize;
    let end_word = word_base
        .checked_add(width as usize)
        .ok_or(PreparedWitnessFeedError::SourceRangeOverflow { descriptor })?;
    if end_word > sub_words_per_row {
        return Err(PreparedWitnessFeedError::SourceRangeOutOfBounds {
            descriptor,
            end_word,
            sub_words_per_row,
        });
    }
    let table_size = entry[8] as usize;
    if table_size == 0 {
        return Err(PreparedWitnessFeedError::EmptyTable { descriptor });
    }
    match entry[11] {
        0 => validate_fold_descriptor(
            descriptor,
            entry,
            width as usize,
            table_size,
            luts,
            multiplicity_words,
            used_luts,
            used_destinations,
            destination_table_sizes,
        ),
        1 => validate_memory_descriptor(
            descriptor,
            entry,
            width as usize,
            table_size,
            multiplicity_words,
            used_destinations,
            destination_table_sizes,
        ),
        2 => validate_xor_lut_descriptor(
            descriptor,
            entry,
            width as usize,
            table_size,
            luts,
            multiplicity_words,
            used_luts,
            used_destinations,
            destination_table_sizes,
        ),
        3 => validate_xor12_descriptor(
            descriptor,
            entry,
            width as usize,
            table_size,
            multiplicity_words,
            used_destinations,
            destination_table_sizes,
        ),
        kind => Err(PreparedWitnessFeedError::InvalidDescriptorKind { descriptor, kind }),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_xor_lut_descriptor(
    descriptor: usize,
    entry: &[u32],
    width: usize,
    table_size: usize,
    luts: &[Vec<u32>],
    multiplicity_words: &[usize],
    used_luts: &mut [bool],
    used_destinations: &mut [bool],
    destination_table_sizes: &mut [Option<usize>],
) -> Result<(), PreparedWitnessFeedError> {
    let bits = entry[2];
    if width != 3
        || !(1..16).contains(&bits)
        || entry[3] != bits
        || entry[4] != bits
        || entry[5..7].iter().any(|&value| value != 0)
        || entry[9] == WITNESS_FEED_NO_LUT
        || entry[12] != 0
        || entry[13] != 0
        || (1usize << (2 * bits)) != table_size
    {
        return Err(PreparedWitnessFeedError::InvalidXorDescriptor {
            descriptor,
            kind: 2,
        });
    }
    validate_destination(
        descriptor,
        entry[10],
        entry[7],
        table_size,
        multiplicity_words,
        used_destinations,
        destination_table_sizes,
    )?;
    let lut_index = entry[9] as usize;
    let Some(lut) = luts.get(lut_index) else {
        return Err(PreparedWitnessFeedError::LutIndexOutOfRange {
            descriptor,
            index: entry[9],
            count: luts.len(),
        });
    };
    if lut.len() != table_size {
        return Err(PreparedWitnessFeedError::LutSizeMismatch {
            descriptor,
            index: lut_index,
            expected_words: table_size,
            actual_words: lut.len(),
        });
    }
    if let Some((offset, &value)) = lut
        .iter()
        .enumerate()
        .find(|(_, value)| **value as usize >= table_size)
    {
        return Err(PreparedWitnessFeedError::LutValueOutOfRange {
            descriptor,
            index: lut_index,
            offset,
            value,
            table_size,
        });
    }
    used_luts[lut_index] = true;
    Ok(())
}

fn validate_xor12_descriptor(
    descriptor: usize,
    entry: &[u32],
    width: usize,
    table_size: usize,
    multiplicity_words: &[usize],
    used_destinations: &mut [bool],
    destination_table_sizes: &mut [Option<usize>],
) -> Result<(), PreparedWitnessFeedError> {
    if width != 3
        || entry[2..5] != [12, 12, 12]
        || entry[5..7].iter().any(|&value| value != 0)
        || entry[7] != 0
        || table_size != (1 << 20)
        || entry[9] != WITNESS_FEED_NO_LUT
        || entry[12] != 0
        || entry[13] != 0
    {
        return Err(PreparedWitnessFeedError::InvalidXorDescriptor {
            descriptor,
            kind: 3,
        });
    }
    // The derived high limbs address all sixteen columns, regardless of the
    // recorded relation_index (which is canonically zero for xor12).
    validate_destination(
        descriptor,
        entry[10],
        15,
        table_size,
        multiplicity_words,
        used_destinations,
        destination_table_sizes,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_fold_descriptor(
    descriptor: usize,
    entry: &[u32],
    width: usize,
    table_size: usize,
    luts: &[Vec<u32>],
    multiplicity_words: &[usize],
    used_luts: &mut [bool],
    used_destinations: &mut [bool],
    destination_table_sizes: &mut [Option<usize>],
) -> Result<(), PreparedWitnessFeedError> {
    let mut tuple_bits = 0u32;
    for word in 0..WITNESS_FEED_MAX_TUPLE_WORDS {
        let bits = entry[2 + word];
        if word < width {
            if !(1..32).contains(&bits) {
                return Err(PreparedWitnessFeedError::InvalidTupleBitWidth {
                    descriptor,
                    word,
                    bits,
                });
            }
            tuple_bits = tuple_bits
                .checked_add(bits)
                .ok_or(PreparedWitnessFeedError::SizeOverflow)?;
            if tuple_bits > 32 {
                return Err(PreparedWitnessFeedError::InvalidTupleBitWidth {
                    descriptor,
                    word,
                    bits,
                });
            }
        } else if bits != 0 {
            return Err(PreparedWitnessFeedError::NonzeroUnusedTupleBit {
                descriptor,
                word,
                bits,
            });
        }
    }

    validate_destination(
        descriptor,
        entry[10],
        entry[7],
        table_size,
        multiplicity_words,
        used_destinations,
        destination_table_sizes,
    )?;
    if entry[9] == WITNESS_FEED_NO_LUT {
        return Ok(());
    }
    let lut_index = entry[9] as usize;
    let Some(lut) = luts.get(lut_index) else {
        return Err(PreparedWitnessFeedError::LutIndexOutOfRange {
            descriptor,
            index: entry[9],
            count: luts.len(),
        });
    };
    if lut.len() != table_size {
        return Err(PreparedWitnessFeedError::LutSizeMismatch {
            descriptor,
            index: lut_index,
            expected_words: table_size,
            actual_words: lut.len(),
        });
    }
    // A LUT-addressed fold must cover its complete tuple domain. This rejects
    // triple-xor's `(a,b,a^b)` width instead of silently dropping most rows.
    if tuple_bits >= usize::BITS || (1usize << tuple_bits) > lut.len() {
        return Err(PreparedWitnessFeedError::UnsupportedDependentTupleMapping {
            descriptor,
            tuple_bits,
            lut_words: lut.len(),
        });
    }
    if let Some((offset, &value)) = lut
        .iter()
        .enumerate()
        .find(|(_, value)| **value as usize >= table_size)
    {
        return Err(PreparedWitnessFeedError::LutValueOutOfRange {
            descriptor,
            index: lut_index,
            offset,
            value,
            table_size,
        });
    }
    used_luts[lut_index] = true;
    Ok(())
}

fn validate_memory_descriptor(
    descriptor: usize,
    entry: &[u32],
    width: usize,
    big_table_size: usize,
    multiplicity_words: &[usize],
    used_destinations: &mut [bool],
    destination_table_sizes: &mut [Option<usize>],
) -> Result<(), PreparedWitnessFeedError> {
    if width != 1
        || entry[2] != 31
        || entry[3..7].iter().any(|&bits| bits != 0)
        || entry[9] != WITNESS_FEED_NO_LUT
        || entry[12] == 0
    {
        return Err(PreparedWitnessFeedError::InvalidMemoryDescriptor { descriptor });
    }
    if entry[10] == entry[13] {
        return Err(PreparedWitnessFeedError::AliasedMemoryDestinations {
            descriptor,
            index: entry[10],
        });
    }
    validate_destination(
        descriptor,
        entry[10],
        entry[7],
        big_table_size,
        multiplicity_words,
        used_destinations,
        destination_table_sizes,
    )?;
    validate_destination(
        descriptor,
        entry[13],
        entry[7],
        entry[12] as usize,
        multiplicity_words,
        used_destinations,
        destination_table_sizes,
    )
}

fn validate_destination(
    descriptor: usize,
    raw_index: u32,
    relation_index: u32,
    table_size: usize,
    multiplicity_words: &[usize],
    used_destinations: &mut [bool],
    destination_table_sizes: &mut [Option<usize>],
) -> Result<(), PreparedWitnessFeedError> {
    let index = raw_index as usize;
    let Some(&actual_words) = multiplicity_words.get(index) else {
        return Err(PreparedWitnessFeedError::DestinationIndexOutOfRange {
            descriptor,
            index: raw_index,
            count: multiplicity_words.len(),
        });
    };
    let required_words = (relation_index as usize + 1)
        .checked_mul(table_size)
        .ok_or(PreparedWitnessFeedError::SizeOverflow)?;
    if required_words > actual_words {
        return Err(PreparedWitnessFeedError::DestinationTooSmall {
            descriptor,
            index,
            required_words,
            actual_words,
        });
    }
    if actual_words % table_size != 0 {
        return Err(PreparedWitnessFeedError::DestinationLengthNotMultiple {
            descriptor,
            index,
            destination_words: actual_words,
            table_size,
        });
    }
    match destination_table_sizes[index] {
        Some(expected_table_size) if expected_table_size != table_size => {
            return Err(PreparedWitnessFeedError::DestinationTableSizeMismatch {
                descriptor,
                index,
                expected_table_size,
                actual_table_size: table_size,
            });
        }
        None => destination_table_sizes[index] = Some(table_size),
        _ => {}
    }
    used_destinations[index] = true;
    Ok(())
}

fn pointer_words(count: usize) -> Result<usize, PreparedWitnessFeedError> {
    count
        .max(1)
        .checked_mul(POINTER_WORDS)
        .ok_or(PreparedWitnessFeedError::SizeOverflow)
}

/// One allocation-free, capture-safe feed launch.
pub struct PreparedWitnessFeedGraph<'a> {
    arena: &'a DeviceArena,
    requirements: WitnessFeedWorkspaceRequirements,
    source: ArenaSlice,
    descriptors: ArenaSlice,
    lut_tables: Vec<ArenaSlice>,
    lut_pointers: ArenaSlice,
    multiplicity_destinations: Vec<ArenaSlice>,
    multiplicity_pointers: ArenaSlice,
}

/// One launch clears the complete union of arena-owned multiplicity slabs.
pub struct PreparedWitnessFeedClearGraph<'a> {
    arena: &'a DeviceArena,
    requirements: WitnessFeedClearWorkspaceRequirements,
    destinations: Vec<ArenaSlice>,
    destination_pointers: ArenaSlice,
    destination_lengths: ArenaSlice,
}

impl<'a> PreparedWitnessFeedClearGraph<'a> {
    pub fn prepare(
        arena: &'a DeviceArena,
        destinations: &[ArenaSlice],
        slots: WitnessFeedClearWorkspaceSlots,
    ) -> Result<Self, PreparedWitnessFeedError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(PreparedWitnessFeedError::CudaUnavailable);
        }
        let requirements = witness_feed_clear_workspace_requirements(
            &destinations
                .iter()
                .map(|destination| destination.len_words())
                .collect::<Vec<_>>(),
        )?;
        requirements.arena_slot_requirements(slots)?;
        let destination_pointers = bind_min(
            arena,
            slots.destination_pointers,
            requirements.destination_pointer_words,
            WITNESS_FEED_POINTER_ALIGNMENT_WORDS,
        )?;
        let destination_lengths = bind_min(
            arena,
            slots.destination_lengths,
            requirements.destination_length_words,
            1,
        )?;
        ensure_distinct(
            destinations
                .iter()
                .map(|destination| destination.id())
                .chain([destination_pointers.id(), destination_lengths.id()]),
        )?;
        for (&destination, &expected_words) in
            destinations.iter().zip(&requirements.destination_words)
        {
            bind_external_min(arena, destination, expected_words)?;
        }
        upload(arena, destination_pointers, &pointer_values(destinations))?;
        let lengths = requirements
            .destination_words
            .iter()
            .map(|&words| u32::try_from(words).map_err(|_| PreparedWitnessFeedError::SizeOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        upload(arena, destination_lengths, &lengths)?;
        arena.context().sync()?;
        Ok(Self {
            arena,
            requirements,
            destinations: destinations.to_vec(),
            destination_pointers,
            destination_lengths,
        })
    }

    pub fn launch(&self) -> Result<(), PreparedWitnessFeedError> {
        self.launch_on(self.arena.context().launch_context())
    }

    pub fn launch_on(&self, launch: CudaLaunchContext) -> Result<(), PreparedWitnessFeedError> {
        if launch.identity_token() != self.arena.context().identity_token() {
            return Err(CudaRuntimeError::ContextMismatch.into());
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_witness_feed_clear_on(
                self.destination_pointers.as_u32_ptr().cast(),
                self.destination_lengths.as_u32_ptr().cast_const(),
                self.requirements.destination_words.len() as u32,
                self.requirements.max_destination_words as u32,
                launch.stream_raw().as_ptr(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(PreparedWitnessFeedError::KernelLaunchFailed)
        }
    }

    pub fn destinations(&self) -> &[ArenaSlice] {
        &self.destinations
    }

    pub fn requirements(&self) -> &WitnessFeedClearWorkspaceRequirements {
        &self.requirements
    }
}

impl<'a> PreparedWitnessFeedGraph<'a> {
    /// Bind exact geometry, upload immutable launch data once, and drain setup
    /// before any graph capture begins.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        arena: &'a DeviceArena,
        source: ArenaSlice,
        row_count: usize,
        sub_words_per_row: usize,
        descriptors_host: &[u32],
        luts_host: &[Vec<u32>],
        multiplicity_words: &[usize],
        slots: &WitnessFeedWorkspaceSlots,
    ) -> Result<Self, PreparedWitnessFeedError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(PreparedWitnessFeedError::CudaUnavailable);
        }
        let requirements = witness_feed_workspace_requirements(
            row_count,
            sub_words_per_row,
            descriptors_host,
            luts_host,
            multiplicity_words,
        )?;
        requirements.arena_slot_requirements(slots)?;
        bind_external_min(arena, source, requirements.source_words)?;

        let descriptors = bind_min(arena, slots.descriptors, requirements.descriptor_words, 1)?;
        let lut_tables = bind_many_exact(arena, &slots.lut_tables, &requirements.lut_words, 1)?;
        let lut_pointers = bind_min(
            arena,
            slots.lut_pointers,
            requirements.lut_pointer_words,
            WITNESS_FEED_POINTER_ALIGNMENT_WORDS,
        )?;
        let multiplicity_destinations = bind_many_exact(
            arena,
            &slots.multiplicity_destinations,
            &requirements.multiplicity_words,
            1,
        )?;
        let multiplicity_pointers = bind_min(
            arena,
            slots.multiplicity_pointers,
            requirements.multiplicity_pointer_words,
            WITNESS_FEED_POINTER_ALIGNMENT_WORDS,
        )?;
        ensure_distinct(
            std::iter::once(source.id())
                .chain(std::iter::once(descriptors.id()))
                .chain(lut_tables.iter().map(|slice| slice.id()))
                .chain(std::iter::once(lut_pointers.id()))
                .chain(multiplicity_destinations.iter().map(|slice| slice.id()))
                .chain(std::iter::once(multiplicity_pointers.id())),
        )?;

        upload(arena, descriptors, descriptors_host)?;
        for (destination, lut) in lut_tables.iter().copied().zip(luts_host) {
            upload(arena, destination, lut)?;
        }
        let lut_pointer_values = if lut_tables.is_empty() {
            vec![0usize]
        } else {
            pointer_values(&lut_tables)
        };
        let multiplicity_pointer_values = pointer_values(&multiplicity_destinations);
        upload(arena, lut_pointers, &lut_pointer_values)?;
        upload(arena, multiplicity_pointers, &multiplicity_pointer_values)?;
        arena.context().sync()?;

        Ok(Self {
            arena,
            requirements,
            source,
            descriptors,
            lut_tables,
            lut_pointers,
            multiplicity_destinations,
            multiplicity_pointers,
        })
    }

    /// Enqueue only the existing feed-count kernel on the proof stream.
    pub fn launch(&self) -> Result<(), PreparedWitnessFeedError> {
        self.launch_on(self.arena.context().launch_context())
    }

    pub fn launch_on(&self, launch: CudaLaunchContext) -> Result<(), PreparedWitnessFeedError> {
        if launch.identity_token() != self.arena.context().identity_token() {
            return Err(CudaRuntimeError::ContextMismatch.into());
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_witness_feed_counts_on(
                self.source.as_u32_ptr().cast_const(),
                self.requirements.row_count as u32,
                self.descriptors.as_u32_ptr().cast_const(),
                self.requirements.descriptor_count as u32,
                self.lut_pointers.as_u32_ptr().cast(),
                self.multiplicity_pointers.as_u32_ptr().cast(),
                launch.stream_raw().as_ptr(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(PreparedWitnessFeedError::KernelLaunchFailed)
        }
    }

    pub fn requirements(&self) -> &WitnessFeedWorkspaceRequirements {
        &self.requirements
    }

    pub fn source(&self) -> ArenaSlice {
        self.source
    }

    pub fn multiplicity_destinations(&self) -> &[ArenaSlice] {
        &self.multiplicity_destinations
    }

    /// Immutable descriptor storage sealed into the captured kernel arguments.
    pub fn descriptor_slices(&self) -> Vec<ArenaSlice> {
        let mut result = Vec::with_capacity(self.lut_tables.len() + 3);
        result.push(self.descriptors);
        result.extend(self.lut_tables.iter().copied());
        result.extend([self.lut_pointers, self.multiplicity_pointers]);
        result
    }
}

/// Clear the union of all feed destinations exactly once per arena slot. Call
/// this once before every set of producer feed launches; duplicate slices from
/// graphs sharing a target are intentionally coalesced.
pub fn clear_witness_feed_destinations_once(
    arena: &DeviceArena,
    destinations: &[ArenaSlice],
) -> Result<u64, PreparedWitnessFeedError> {
    let mut unique = BTreeMap::<ArenaSlotId, ArenaSlice>::new();
    for &destination in destinations {
        bind_external_min(arena, destination, destination.len_words())?;
        if let Some(previous) = unique.insert(destination.id(), destination) {
            if previous.as_u32_ptr() != destination.as_u32_ptr()
                || previous.len_words() != destination.len_words()
            {
                return Err(PreparedWitnessFeedError::ConflictingDestination(
                    destination.id(),
                ));
            }
        }
    }
    let mut bytes = 0usize;
    for destination in unique.into_values() {
        bytes = bytes
            .checked_add(destination.len_bytes())
            .ok_or(PreparedWitnessFeedError::SizeOverflow)?;
        unsafe {
            arena
                .context()
                .memset_async(destination.as_void_ptr(), 0, destination.len_bytes())?;
        }
    }
    u64::try_from(bytes).map_err(|_| PreparedWitnessFeedError::SizeOverflow)
}

fn pointer_values(slices: &[ArenaSlice]) -> Vec<usize> {
    slices
        .iter()
        .map(|slice| slice.as_u32_ptr() as usize)
        .collect()
}

fn upload<T: Copy>(
    arena: &DeviceArena,
    destination: ArenaSlice,
    values: &[T],
) -> Result<(), PreparedWitnessFeedError> {
    let bytes = core::mem::size_of_val(values);
    if bytes > destination.len_bytes() {
        return Err(PreparedWitnessFeedError::SlotSizeMismatch {
            slot: destination.id(),
            expected_words: bytes.div_ceil(WORD_BYTES),
            actual_words: destination.len_words(),
        });
    }
    unsafe {
        arena.context().memcpy_h2d_async(
            destination.as_void_ptr(),
            values.as_ptr().cast::<c_void>(),
            bytes,
        )?;
    }
    Ok(())
}

fn bind_many_exact(
    arena: &DeviceArena,
    ids: &[ArenaSlotId],
    lengths: &[usize],
    alignment_words: usize,
) -> Result<Vec<ArenaSlice>, PreparedWitnessFeedError> {
    ids.iter()
        .zip(lengths)
        .map(|(&id, &words)| bind_min(arena, id, words, alignment_words))
        .collect()
}

fn bind_min(
    arena: &DeviceArena,
    id: ArenaSlotId,
    expected_words: usize,
    alignment_words: usize,
) -> Result<ArenaSlice, PreparedWitnessFeedError> {
    let slice = arena.bind(id)?;
    bind_external_min(arena, slice, expected_words)?;
    if (slice.as_u32_ptr() as usize) % (alignment_words * WORD_BYTES) != 0 {
        return Err(PreparedWitnessFeedError::SlotMisaligned(id));
    }
    Ok(slice)
}

// Pooled arena slots are sized to the largest disjoint-lifetime sharer, so a
// requirement is a lower bound: launches touch exactly the required words and
// never the pooled surplus (same contract as prepared_witness_input's
// bind_min; undersized, misaligned, or foreign-context slots still fail
// closed).
fn bind_external_min(
    arena: &DeviceArena,
    slice: ArenaSlice,
    expected_words: usize,
) -> Result<(), PreparedWitnessFeedError> {
    if slice.context_token() != arena.context().identity_token() {
        return Err(PreparedWitnessFeedError::ContextMismatch(slice.id()));
    }
    if slice.len_words() < expected_words {
        return Err(PreparedWitnessFeedError::SlotSizeMismatch {
            slot: slice.id(),
            expected_words,
            actual_words: slice.len_words(),
        });
    }
    Ok(())
}

fn check_count(
    role: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), PreparedWitnessFeedError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PreparedWitnessFeedError::SlotShapeMismatch {
            role,
            expected,
            actual,
        })
    }
}

fn ensure_distinct(
    ids: impl IntoIterator<Item = ArenaSlotId>,
) -> Result<(), PreparedWitnessFeedError> {
    let mut seen = BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(PreparedWitnessFeedError::DuplicateSlot(id));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fold_descriptor(
        word_base: u32,
        bits: &[u32],
        relation: u32,
        table_size: u32,
        lut: u32,
        destination: u32,
    ) -> [u32; WITNESS_FEED_DESCRIPTOR_WORDS] {
        let mut entry = [0u32; WITNESS_FEED_DESCRIPTOR_WORDS];
        entry[0] = word_base;
        entry[1] = bits.len() as u32;
        entry[2..2 + bits.len()].copy_from_slice(bits);
        entry[7] = relation;
        entry[8] = table_size;
        entry[9] = lut;
        entry[10] = destination;
        entry
    }

    fn memory_descriptor() -> [u32; WITNESS_FEED_DESCRIPTOR_WORDS] {
        let mut entry = [0u32; WITNESS_FEED_DESCRIPTOR_WORDS];
        entry[0] = 7;
        entry[1] = 1;
        entry[2] = 31;
        entry[8] = 8;
        entry[9] = WITNESS_FEED_NO_LUT;
        entry[10] = 2;
        entry[11] = 1;
        entry[12] = 4;
        entry[13] = 3;
        entry
    }

    fn valid_geometry() -> (Vec<u32>, Vec<Vec<u32>>, Vec<usize>) {
        let mut descriptors = Vec::new();
        descriptors.extend(fold_descriptor(0, &[2, 2], 1, 16, 0, 0));
        descriptors.extend(fold_descriptor(
            2,
            &[1, 1, 1, 1, 1],
            0,
            32,
            WITNESS_FEED_NO_LUT,
            1,
        ));
        descriptors.extend(memory_descriptor());
        (
            descriptors,
            vec![(0..16).rev().collect()],
            vec![32, 32, 8, 4],
        )
    }

    #[test]
    fn pure_requirements_cover_descriptors_luts_and_destinations() {
        let (descriptors, luts, destinations) = valid_geometry();
        let requirements =
            witness_feed_workspace_requirements(32, 8, &descriptors, &luts, &destinations).unwrap();
        assert_eq!(requirements.source_words, 256);
        assert_eq!(requirements.descriptor_count, 3);
        assert_eq!(requirements.descriptor_words, 42);
        assert_eq!(requirements.lut_words, [16]);
        assert_eq!(requirements.multiplicity_words, [32, 32, 8, 4]);
        assert_eq!(requirements.lut_pointer_words, POINTER_WORDS);
        assert_eq!(requirements.multiplicity_pointer_words, 4 * POINTER_WORDS);

        let slots = WitnessFeedWorkspaceSlots {
            descriptors: ArenaSlotId(1),
            lut_tables: vec![ArenaSlotId(2)],
            lut_pointers: ArenaSlotId(3),
            multiplicity_destinations: vec![
                ArenaSlotId(4),
                ArenaSlotId(5),
                ArenaSlotId(6),
                ArenaSlotId(7),
            ],
            multiplicity_pointers: ArenaSlotId(8),
        };
        assert_eq!(
            requirements.arena_slot_requirements(&slots).unwrap().len(),
            8
        );
    }

    #[test]
    fn xor_tuple_geometry_is_explicit_and_malformed_kinds_fail_closed() {
        let mut triple_xor = fold_descriptor(0, &[8, 8, 8], 0, 1 << 16, 0, 0);
        triple_xor[11] = 2;
        let requirements = witness_feed_workspace_requirements(
            32,
            3,
            &triple_xor,
            &[vec![0; 1 << 16]],
            &[1 << 16],
        )
        .unwrap();
        assert_eq!(requirements.descriptor_count, 1);

        triple_xor[11] = 9;
        assert!(matches!(
            witness_feed_workspace_requirements(
                32,
                3,
                &triple_xor,
                &[vec![0; 1 << 16]],
                &[1 << 16]
            ),
            Err(PreparedWitnessFeedError::InvalidDescriptorKind { .. })
        ));

        triple_xor[11] = 2;
        triple_xor[4] = 7;
        assert!(matches!(
            witness_feed_workspace_requirements(
                32,
                3,
                &triple_xor,
                &[vec![0; 1 << 16]],
                &[1 << 16]
            ),
            Err(PreparedWitnessFeedError::InvalidXorDescriptor { kind: 2, .. })
        ));

        let mut xor12 = fold_descriptor(0, &[12, 12, 12], 0, 1 << 20, WITNESS_FEED_NO_LUT, 0);
        xor12[11] = 3;
        assert!(
            witness_feed_workspace_requirements(32, 3, &xor12, &[], &[16 * (1 << 20)],).is_ok()
        );

        let (descriptors, luts, destinations) = valid_geometry();
        let requirements =
            witness_feed_workspace_requirements(32, 8, &descriptors, &luts, &destinations).unwrap();
        let duplicate = ArenaSlotId(1);
        let slots = WitnessFeedWorkspaceSlots {
            descriptors: duplicate,
            lut_tables: vec![duplicate],
            lut_pointers: ArenaSlotId(3),
            multiplicity_destinations: vec![
                ArenaSlotId(4),
                ArenaSlotId(5),
                ArenaSlotId(6),
                ArenaSlotId(7),
            ],
            multiplicity_pointers: ArenaSlotId(8),
        };
        assert_eq!(
            requirements.arena_slot_requirements(&slots).unwrap_err(),
            PreparedWitnessFeedError::DuplicateSlot(duplicate)
        );
    }

    #[test]
    fn batched_clear_plans_one_pointer_and_length_table() {
        let requirements = witness_feed_clear_workspace_requirements(&[16, 32, 8]).unwrap();
        assert_eq!(requirements.destination_words, [16, 32, 8]);
        assert_eq!(requirements.max_destination_words, 32);
        assert_eq!(requirements.destination_length_words, 3);
        let slots = WitnessFeedClearWorkspaceSlots {
            destination_pointers: ArenaSlotId(70),
            destination_lengths: ArenaSlotId(71),
        };
        let requests = requirements.arena_slot_requirements(slots).unwrap();
        assert_eq!(requests[0].len_words, 3 * POINTER_WORDS);
        assert_eq!(requests[1].len_words, 3);
    }
}
