//! Capture-safe native CUDA writer for Cairo's `ec_op_builtin`.
//!
//! The graph writes the component's committed base columns and word-major
//! lookup inputs, then writes all 252 `partial_ec_mul_generic` states directly
//! into that consumer's final padded input columns.  No host/sub-input staging
//! buffer exists on the launch path.

use std::collections::BTreeSet;

use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaLaunchContext, CudaRuntimeError,
    DeviceArena,
};
use super::prepared_execution_tables::PreparedExecutionTablesView;

pub const EC_OP_TRACE_COLUMNS: usize = 273;
pub const EC_OP_LOOKUP_WORDS_PER_ROW: usize = 488;
pub const EC_OP_PARTIAL_INPUT_COLUMNS: usize = 127;
pub const EC_OP_PARTIAL_REAL_ROUNDS: usize = 252;
pub const EC_OP_PARTIAL_PADDED_ROUNDS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EcOpWorkspaceRequirements {
    pub row_count: usize,
    pub trace_column_words: Vec<usize>,
    pub lookup_words: usize,
    pub partial_real_rows: usize,
    pub partial_row_count: usize,
    pub partial_input_column_words: Vec<usize>,
    pub segment_start_words: usize,
    pub address_count_words: usize,
    pub big_count_words: usize,
    pub small_count_words: usize,
    pub range_check_8_count_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcOpMultiplicityGeometry {
    pub address_count_words: usize,
    pub big_count_words: usize,
    pub small_count_words: usize,
    pub range_check_8_count_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcOpArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EcOpWorkspaceSlots {
    pub trace_columns: Vec<ArenaSlotId>,
    pub lookup_words: ArenaSlotId,
    pub partial_input_columns: Vec<ArenaSlotId>,
    pub segment_start: ArenaSlotId,
    pub address_counts: ArenaSlotId,
    pub big_counts: ArenaSlotId,
    pub small_counts: ArenaSlotId,
    pub range_check_8_counts: ArenaSlotId,
}

impl EcOpWorkspaceRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: &EcOpWorkspaceSlots,
    ) -> Result<Vec<EcOpArenaSlotRequirement>, PreparedEcOpError> {
        validate_slot_shape(self, slots)?;
        let mut result = Vec::with_capacity(
            self.trace_column_words.len() + self.partial_input_column_words.len() + 6,
        );
        result.extend(
            slots
                .trace_columns
                .iter()
                .zip(&self.trace_column_words)
                .map(|(&id, &len_words)| words(id, len_words)),
        );
        result.extend([
            words(slots.segment_start, self.segment_start_words),
            words(slots.address_counts, self.address_count_words),
            words(slots.big_counts, self.big_count_words),
            words(slots.small_counts, self.small_count_words),
            words(slots.range_check_8_counts, self.range_check_8_count_words),
        ]);
        result.push(words(slots.lookup_words, self.lookup_words));
        result.extend(
            slots
                .partial_input_columns
                .iter()
                .zip(&self.partial_input_column_words)
                .map(|(&id, &len_words)| words(id, len_words)),
        );
        let mut distinct = BTreeSet::new();
        for entry in &result {
            if !distinct.insert(entry.id) {
                return Err(PreparedEcOpError::DuplicateSlot(entry.id));
            }
        }
        Ok(result)
    }
}

fn words(id: ArenaSlotId, len_words: usize) -> EcOpArenaSlotRequirement {
    EcOpArenaSlotRequirement {
        id,
        len_words,
        alignment_words: 1,
    }
}

pub fn ec_op_workspace_requirements(
    row_count: usize,
    multiplicity: EcOpMultiplicityGeometry,
) -> Result<EcOpWorkspaceRequirements, PreparedEcOpError> {
    if row_count < 16 || !row_count.is_power_of_two() {
        return Err(PreparedEcOpError::InvalidRowCount(row_count));
    }
    u32::try_from(row_count).map_err(|_| PreparedEcOpError::SizeOverflow)?;
    let lookup_words = row_count
        .checked_mul(EC_OP_LOOKUP_WORDS_PER_ROW)
        .ok_or(PreparedEcOpError::SizeOverflow)?;
    let partial_real_rows = row_count
        .checked_mul(EC_OP_PARTIAL_REAL_ROUNDS)
        .ok_or(PreparedEcOpError::SizeOverflow)?;
    let partial_row_count = row_count
        .checked_mul(EC_OP_PARTIAL_PADDED_ROUNDS)
        .ok_or(PreparedEcOpError::SizeOverflow)?;
    u32::try_from(partial_row_count).map_err(|_| PreparedEcOpError::SizeOverflow)?;
    if multiplicity.address_count_words == 0
        || multiplicity.big_count_words == 0
        || multiplicity.small_count_words == 0
        || multiplicity.range_check_8_count_words < 256
    {
        return Err(PreparedEcOpError::InvalidMultiplicityGeometry);
    }
    for words in [
        multiplicity.address_count_words,
        multiplicity.big_count_words,
        multiplicity.small_count_words,
        multiplicity.range_check_8_count_words,
    ] {
        u32::try_from(words).map_err(|_| PreparedEcOpError::SizeOverflow)?;
    }
    Ok(EcOpWorkspaceRequirements {
        row_count,
        trace_column_words: vec![row_count; EC_OP_TRACE_COLUMNS],
        lookup_words,
        partial_real_rows,
        partial_row_count,
        partial_input_column_words: vec![partial_row_count; EC_OP_PARTIAL_INPUT_COLUMNS],
        segment_start_words: 1,
        address_count_words: multiplicity.address_count_words,
        big_count_words: multiplicity.big_count_words,
        small_count_words: multiplicity.small_count_words,
        range_check_8_count_words: multiplicity.range_check_8_count_words,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedEcOpError {
    CudaUnavailable,
    InvalidRowCount(usize),
    InvalidSegmentStart,
    InvalidMultiplicityGeometry,
    SegmentOutOfBounds {
        start: usize,
        end: usize,
        addresses: usize,
    },
    SizeOverflow,
    SlotShapeMismatch {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    SlotTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    DuplicateSlot(ArenaSlotId),
    ExecutionTableContextMismatch,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedEcOpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "prepared EC-op witness rejected: {self:?}")
    }
}

impl std::error::Error for PreparedEcOpError {}

impl From<ArenaError> for PreparedEcOpError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedEcOpError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedEcOpLaunchTelemetry {
    pub kernel_launches: u64,
    pub allocations: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub sync_calls: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedEcOpIngestTelemetry {
    pub h2d_bytes: u64,
    pub h2d_copies: u64,
    pub fill_calls: u64,
    pub sync_calls: u64,
}

impl PreparedEcOpLaunchTelemetry {
    const TWO_KERNELS: Self = Self {
        kernel_launches: 2,
        allocations: 0,
        h2d_bytes: 0,
        d2h_bytes: 0,
        sync_calls: 0,
    };
}

pub struct PreparedEcOpGraph<'a> {
    arena: &'a DeviceArena,
    execution_table_pointers: ArenaSlice,
    n_addresses: u32,
    n_big: u32,
    n_small: u32,
    segment_start: ArenaSlice,
    row_count: u32,
    trace_columns: Vec<ArenaSlice>,
    trace_pointers: Vec<*mut u32>,
    lookup_words: ArenaSlice,
    partial_row_count: u32,
    partial_input_columns: Vec<ArenaSlice>,
    partial_input_pointers: Vec<*mut u32>,
    address_counts: ArenaSlice,
    big_counts: ArenaSlice,
    small_counts: ArenaSlice,
    range_check_8_counts: ArenaSlice,
}

impl<'a> PreparedEcOpGraph<'a> {
    pub fn prepare(
        arena: &'a DeviceArena,
        execution_tables: PreparedExecutionTablesView<'a>,
        requirements: &EcOpWorkspaceRequirements,
        slots: &EcOpWorkspaceSlots,
    ) -> Result<Self, PreparedEcOpError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(PreparedEcOpError::CudaUnavailable);
        }
        if ec_op_workspace_requirements(
            requirements.row_count,
            EcOpMultiplicityGeometry {
                address_count_words: requirements.address_count_words,
                big_count_words: requirements.big_count_words,
                small_count_words: requirements.small_count_words,
                range_check_8_count_words: requirements.range_check_8_count_words,
            },
        )? != *requirements
        {
            return Err(PreparedEcOpError::InvalidRowCount(requirements.row_count));
        }
        requirements.arena_slot_requirements(slots)?;
        if !execution_tables.belongs_to(arena) {
            return Err(PreparedEcOpError::ExecutionTableContextMismatch);
        }
        let (n_addresses, n_big, n_small) = execution_tables.shape();
        if requirements.address_count_words < n_addresses.saturating_sub(1)
            || requirements.big_count_words < n_big
            || requirements.small_count_words < n_small
            || requirements.range_check_8_count_words < 256
        {
            return Err(PreparedEcOpError::InvalidMultiplicityGeometry);
        }

        let trace_columns = bind_many(arena, &slots.trace_columns, requirements.row_count)?;
        let lookup_words = bind_slot(arena, slots.lookup_words, requirements.lookup_words)?;
        let partial_input_columns = bind_many(
            arena,
            &slots.partial_input_columns,
            requirements.partial_row_count,
        )?;
        let segment_start = bind_slot(arena, slots.segment_start, 1)?;
        let address_counts = bind_slot(
            arena,
            slots.address_counts,
            requirements.address_count_words,
        )?;
        let big_counts = bind_slot(arena, slots.big_counts, requirements.big_count_words)?;
        let small_counts = bind_slot(arena, slots.small_counts, requirements.small_count_words)?;
        let range_check_8_counts = bind_slot(
            arena,
            slots.range_check_8_counts,
            requirements.range_check_8_count_words,
        )?;
        let trace_pointers = trace_columns
            .iter()
            .map(|column| column.as_u32_ptr())
            .collect();
        let partial_input_pointers = partial_input_columns
            .iter()
            .map(|column| column.as_u32_ptr())
            .collect();

        Ok(Self {
            arena,
            execution_table_pointers: execution_tables.table_pointers(),
            n_addresses: u32::try_from(n_addresses).map_err(|_| PreparedEcOpError::SizeOverflow)?,
            n_big: u32::try_from(n_big).map_err(|_| PreparedEcOpError::SizeOverflow)?,
            n_small: u32::try_from(n_small).map_err(|_| PreparedEcOpError::SizeOverflow)?,
            segment_start,
            row_count: u32::try_from(requirements.row_count)
                .map_err(|_| PreparedEcOpError::SizeOverflow)?,
            trace_columns,
            trace_pointers,
            lookup_words,
            partial_row_count: u32::try_from(requirements.partial_row_count)
                .map_err(|_| PreparedEcOpError::SizeOverflow)?,
            partial_input_columns,
            partial_input_pointers,
            address_counts,
            big_counts,
            small_counts,
            range_check_8_counts,
        })
    }

    /// Update the statement-varying builtin segment without changing any
    /// captured kernel parameter or arena address.
    pub fn ingest_segment_start(
        &self,
        segment_start: usize,
    ) -> Result<PreparedEcOpIngestTelemetry, PreparedEcOpError> {
        if segment_start == 0 {
            return Err(PreparedEcOpError::InvalidSegmentStart);
        }
        let end = segment_start
            .checked_add(
                (self.row_count as usize)
                    .checked_mul(7)
                    .ok_or(PreparedEcOpError::SizeOverflow)?,
            )
            .ok_or(PreparedEcOpError::SizeOverflow)?;
        if end > self.n_addresses as usize {
            return Err(PreparedEcOpError::SegmentOutOfBounds {
                start: segment_start,
                end,
                addresses: self.n_addresses as usize,
            });
        }
        let value = u32::try_from(segment_start).map_err(|_| PreparedEcOpError::SizeOverflow)?;
        unsafe {
            self.arena.context().fill_u32_async(
                self.segment_start.as_u32_ptr(),
                value,
                self.segment_start.len_words(),
            )?;
        }
        Ok(PreparedEcOpIngestTelemetry {
            h2d_bytes: 0,
            h2d_copies: 0,
            fill_calls: 1,
            sync_calls: 0,
        })
    }

    pub fn launch(&self) -> Result<PreparedEcOpLaunchTelemetry, PreparedEcOpError> {
        self.launch_on(self.arena.context().launch_context())
    }

    pub fn launch_on(
        &self,
        launch: CudaLaunchContext,
    ) -> Result<PreparedEcOpLaunchTelemetry, PreparedEcOpError> {
        if launch.identity_token() != self.arena.context().identity_token() {
            return Err(CudaRuntimeError::ContextMismatch.into());
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::ec_op_builtin_witness_on(
                self.execution_table_pointers.as_u32_ptr().cast(),
                self.n_addresses,
                self.n_big,
                self.n_small,
                self.segment_start.as_u32_ptr(),
                self.row_count,
                self.trace_pointers.as_ptr(),
                self.lookup_words.as_u32_ptr(),
                self.partial_input_pointers.as_ptr(),
                self.partial_row_count,
                self.address_counts.as_u32_ptr(),
                u32::try_from(self.address_counts.len_words())
                    .map_err(|_| PreparedEcOpError::SizeOverflow)?,
                self.big_counts.as_u32_ptr(),
                u32::try_from(self.big_counts.len_words())
                    .map_err(|_| PreparedEcOpError::SizeOverflow)?,
                self.small_counts.as_u32_ptr(),
                u32::try_from(self.small_counts.len_words())
                    .map_err(|_| PreparedEcOpError::SizeOverflow)?,
                self.range_check_8_counts.as_u32_ptr(),
                u32::try_from(self.range_check_8_counts.len_words())
                    .map_err(|_| PreparedEcOpError::SizeOverflow)?,
                launch.stream_raw().as_ptr(),
            )
        };
        check_cuda("ec_op_builtin_witness_on", code)?;
        Ok(PreparedEcOpLaunchTelemetry::TWO_KERNELS)
    }

    pub fn trace_columns(&self) -> &[ArenaSlice] {
        &self.trace_columns
    }

    pub fn lookup_words(&self) -> ArenaSlice {
        self.lookup_words
    }

    pub fn partial_input_columns(&self) -> &[ArenaSlice] {
        &self.partial_input_columns
    }

    pub fn segment_start_source(&self) -> ArenaSlice {
        self.segment_start
    }

    pub fn multiplicity_destinations(&self) -> [ArenaSlice; 4] {
        [
            self.address_counts,
            self.big_counts,
            self.small_counts,
            self.range_check_8_counts,
        ]
    }

    pub fn row_count(&self) -> usize {
        self.row_count as usize
    }

    pub fn partial_row_count(&self) -> usize {
        self.partial_row_count as usize
    }

    /// Stable scheduler weight: each row executes the full 252-step EC chain.
    pub fn estimated_work(&self) -> u64 {
        u64::from(self.row_count).saturating_mul(EC_OP_PARTIAL_REAL_ROUNDS as u64)
    }
}

fn bind_many(
    arena: &DeviceArena,
    ids: &[ArenaSlotId],
    required_words: usize,
) -> Result<Vec<ArenaSlice>, PreparedEcOpError> {
    ids.iter()
        .map(|&id| bind_slot(arena, id, required_words))
        .collect()
}

fn bind_slot(
    arena: &DeviceArena,
    id: ArenaSlotId,
    required_words: usize,
) -> Result<ArenaSlice, PreparedEcOpError> {
    let slice = arena.bind(id)?;
    if slice.len_words() < required_words {
        return Err(PreparedEcOpError::SlotTooSmall {
            slot: id,
            required_words,
            actual_words: slice.len_words(),
        });
    }
    Ok(slice)
}

fn validate_slot_shape(
    requirements: &EcOpWorkspaceRequirements,
    slots: &EcOpWorkspaceSlots,
) -> Result<(), PreparedEcOpError> {
    for (role, expected, actual) in [
        (
            "trace columns",
            requirements.trace_column_words.len(),
            slots.trace_columns.len(),
        ),
        (
            "partial input columns",
            requirements.partial_input_column_words.len(),
            slots.partial_input_columns.len(),
        ),
    ] {
        if actual != expected {
            return Err(PreparedEcOpError::SlotShapeMismatch {
                role,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots() -> EcOpWorkspaceSlots {
        let mut next = 1u32;
        let mut id = || {
            let result = ArenaSlotId(next);
            next += 1;
            result
        };
        EcOpWorkspaceSlots {
            trace_columns: (0..EC_OP_TRACE_COLUMNS).map(|_| id()).collect(),
            lookup_words: id(),
            partial_input_columns: (0..EC_OP_PARTIAL_INPUT_COLUMNS).map(|_| id()).collect(),
            segment_start: id(),
            address_counts: id(),
            big_counts: id(),
            small_counts: id(),
            range_check_8_counts: id(),
        }
    }

    fn multiplicity() -> EcOpMultiplicityGeometry {
        EcOpMultiplicityGeometry {
            address_count_words: 512,
            big_count_words: 128,
            small_count_words: 64,
            range_check_8_count_words: 256,
        }
    }

    #[test]
    fn pure_geometry_matches_generated_writer_and_round_order() {
        let requirements = ec_op_workspace_requirements(32, multiplicity()).unwrap();
        assert_eq!(requirements.trace_column_words, vec![32; 273]);
        assert_eq!(requirements.lookup_words, 32 * 488);
        assert_eq!(requirements.partial_real_rows, 32 * 252);
        assert_eq!(requirements.partial_row_count, 32 * 256);
        assert_eq!(requirements.partial_input_column_words, vec![32 * 256; 127]);
        assert_eq!(7 * 32 + 31, 255, "round-major consumer row formula");
        assert_eq!(
            requirements
                .arena_slot_requirements(&slots())
                .unwrap()
                .len(),
            273 + 1 + 127 + 5
        );
    }

    #[test]
    fn pure_geometry_fails_closed() {
        assert_eq!(
            ec_op_workspace_requirements(0, multiplicity()).unwrap_err(),
            PreparedEcOpError::InvalidRowCount(0)
        );
        assert_eq!(
            ec_op_workspace_requirements(24, multiplicity()).unwrap_err(),
            PreparedEcOpError::InvalidRowCount(24)
        );
        let requirements = ec_op_workspace_requirements(16, multiplicity()).unwrap();
        let mut bad = slots();
        bad.partial_input_columns.pop();
        assert_eq!(
            requirements.arena_slot_requirements(&bad).unwrap_err(),
            PreparedEcOpError::SlotShapeMismatch {
                role: "partial input columns",
                expected: 127,
                actual: 126,
            }
        );
        let mut duplicate = slots();
        duplicate.partial_input_columns[0] = duplicate.lookup_words;
        assert_eq!(
            requirements
                .arena_slot_requirements(&duplicate)
                .unwrap_err(),
            PreparedEcOpError::DuplicateSlot(duplicate.lookup_words)
        );
    }
}
