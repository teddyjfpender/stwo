//! Capture-safe materialization of the canonical Cairo memory base traces.
//!
//! The compact execution tables and Graph-A multiplicity slabs already live in
//! the proof arena. This graph only copies/slices them into the exact AIR base
//! columns; it allocates, transfers, and synchronizes nothing on launch.

use std::collections::BTreeSet;

use super::exec_context::{
    check_cuda, ArenaSlice, CudaLaunchContext, CudaRuntimeError, DeviceArena,
};
use super::prepared_execution_tables::{
    PreparedExecutionTablesGraph, EXECUTION_TABLE_BIG_LIMBS, EXECUTION_TABLE_SMALL_LIMBS,
};

pub const MEMORY_ADDRESS_BASE_COLUMNS: usize = 32;
pub const MEMORY_BIG_BASE_COLUMNS: usize = EXECUTION_TABLE_BIG_LIMBS + 1;
pub const MEMORY_SMALL_BASE_COLUMNS: usize = EXECUTION_TABLE_SMALL_LIMBS + 1;

#[derive(Clone, Copy, Debug)]
pub struct MemoryBaseTracePart<'a> {
    pub source_offset: usize,
    pub row_count: usize,
    pub outputs: &'a [ArenaSlice],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedMemoryBaseTraceError {
    CudaUnavailable,
    SizeOverflow,
    ShapeMismatch {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    SliceTooSmall {
        role: &'static str,
        required_words: usize,
        actual_words: usize,
    },
    ContextMismatch(&'static str),
    DuplicateSlice,
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedMemoryBaseTraceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "prepared memory base trace rejected: {self:?}")
    }
}

impl std::error::Error for PreparedMemoryBaseTraceError {}

impl From<CudaRuntimeError> for PreparedMemoryBaseTraceError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

struct PreparedValuePart {
    source_offset: u32,
    row_count: u32,
    output_pointers: Vec<*mut u32>,
}

pub struct PreparedMemoryBaseTraceGraph<'a> {
    arena: &'a DeviceArena,
    raw_addr_to_id: ArenaSlice,
    n_addrs: u32,
    address_counts: ArenaSlice,
    address_count_words: u32,
    address_rows: u32,
    address_output_pointers: Vec<*mut u32>,
    big_source_pointers: Vec<*const u32>,
    big_source_words: u32,
    big_counts: ArenaSlice,
    big_count_words: u32,
    big_parts: Vec<PreparedValuePart>,
    small_source_pointers: Vec<*const u32>,
    small_source_words: u32,
    small_counts: ArenaSlice,
    small_count_words: u32,
    small_part: PreparedValuePart,
}

impl<'a> PreparedMemoryBaseTraceGraph<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        arena: &'a DeviceArena,
        execution: &PreparedExecutionTablesGraph<'a>,
        address_counts: ArenaSlice,
        address_count_words: usize,
        address_rows: usize,
        address_outputs: &[ArenaSlice],
        big_counts: ArenaSlice,
        big_count_words: usize,
        big_parts: &[MemoryBaseTracePart<'_>],
        small_counts: ArenaSlice,
        small_count_words: usize,
        small_part: MemoryBaseTracePart<'_>,
    ) -> Result<Self, PreparedMemoryBaseTraceError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(PreparedMemoryBaseTraceError::CudaUnavailable);
        }
        check_shape(
            "address outputs",
            MEMORY_ADDRESS_BASE_COLUMNS,
            address_outputs.len(),
        )?;
        check_shape(
            "big sources",
            EXECUTION_TABLE_BIG_LIMBS,
            execution.big_limbs().len(),
        )?;
        check_shape(
            "small sources",
            EXECUTION_TABLE_SMALL_LIMBS,
            execution.small_limbs().len(),
        )?;
        if big_parts.is_empty() {
            return Err(PreparedMemoryBaseTraceError::ShapeMismatch {
                role: "big parts",
                expected: 1,
                actual: 0,
            });
        }
        for part in big_parts {
            check_shape("big outputs", MEMORY_BIG_BASE_COLUMNS, part.outputs.len())?;
        }
        check_shape(
            "small outputs",
            MEMORY_SMALL_BASE_COLUMNS,
            small_part.outputs.len(),
        )?;
        let expected_address_words = address_rows
            .checked_mul(16)
            .ok_or(PreparedMemoryBaseTraceError::SizeOverflow)?;
        if address_count_words != expected_address_words {
            return Err(PreparedMemoryBaseTraceError::ShapeMismatch {
                role: "address multiplicities",
                expected: expected_address_words,
                actual: address_count_words,
            });
        }
        let expected_big_words = big_parts.iter().try_fold(0usize, |end, part| {
            let next = part
                .source_offset
                .checked_add(part.row_count)
                .ok_or(PreparedMemoryBaseTraceError::SizeOverflow)?;
            Ok::<_, PreparedMemoryBaseTraceError>(end.max(next))
        })?;
        if big_count_words != expected_big_words {
            return Err(PreparedMemoryBaseTraceError::ShapeMismatch {
                role: "big multiplicities",
                expected: expected_big_words,
                actual: big_count_words,
            });
        }
        if small_part.source_offset != 0 || small_part.row_count != small_count_words {
            return Err(PreparedMemoryBaseTraceError::ShapeMismatch {
                role: "small multiplicities",
                expected: small_part.row_count,
                actual: small_count_words,
            });
        }

        let token = arena.context().identity_token();
        let mut all = Vec::new();
        for (role, slice, words) in [
            (
                "raw address table",
                execution.raw_addr_to_id(),
                execution.requirements().raw_addr_to_id_words,
            ),
            (
                "address multiplicities",
                address_counts,
                address_count_words,
            ),
            ("big multiplicities", big_counts, big_count_words),
            ("small multiplicities", small_counts, small_count_words),
        ] {
            validate_slice(role, slice, words, token)?;
            all.push(slice.id());
        }
        for (&source, words) in execution
            .big_limbs()
            .iter()
            .zip(std::iter::repeat(execution.requirements().big_column_words))
            .chain(execution.small_limbs().iter().zip(std::iter::repeat(
                execution.requirements().small_column_words,
            )))
        {
            validate_slice("memory limb source", source, words, token)?;
            all.push(source.id());
        }
        for (output, words) in address_outputs
            .iter()
            .copied()
            .zip(std::iter::repeat(address_rows))
            .chain(big_parts.iter().flat_map(|part| {
                part.outputs
                    .iter()
                    .copied()
                    .zip(std::iter::repeat(part.row_count))
            }))
            .chain(
                small_part
                    .outputs
                    .iter()
                    .copied()
                    .zip(std::iter::repeat(small_part.row_count)),
            )
        {
            validate_slice("memory base output", output, words, token)?;
            all.push(output.id());
        }
        let mut distinct = BTreeSet::new();
        if all.into_iter().any(|id| !distinct.insert(id)) {
            return Err(PreparedMemoryBaseTraceError::DuplicateSlice);
        }

        let prepared_part = |part: &MemoryBaseTracePart<'_>| {
            Ok::<_, PreparedMemoryBaseTraceError>(PreparedValuePart {
                source_offset: u32::try_from(part.source_offset)
                    .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
                row_count: u32::try_from(part.row_count)
                    .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
                output_pointers: part
                    .outputs
                    .iter()
                    .map(|slice| slice.as_u32_ptr())
                    .collect(),
            })
        };
        Ok(Self {
            arena,
            raw_addr_to_id: execution.raw_addr_to_id(),
            n_addrs: u32::try_from(execution.requirements().n_addrs)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            address_counts,
            address_count_words: u32::try_from(address_count_words)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            address_rows: u32::try_from(address_rows)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            address_output_pointers: address_outputs
                .iter()
                .map(|slice| slice.as_u32_ptr())
                .collect(),
            big_source_pointers: execution
                .big_limbs()
                .iter()
                .map(|slice| slice.as_u32_ptr().cast_const())
                .collect(),
            big_source_words: u32::try_from(execution.requirements().big_column_words)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            big_counts,
            big_count_words: u32::try_from(big_count_words)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            big_parts: big_parts
                .iter()
                .map(prepared_part)
                .collect::<Result<Vec<_>, _>>()?,
            small_source_pointers: execution
                .small_limbs()
                .iter()
                .map(|slice| slice.as_u32_ptr().cast_const())
                .collect(),
            small_source_words: u32::try_from(execution.requirements().small_column_words)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            small_counts,
            small_count_words: u32::try_from(small_count_words)
                .map_err(|_| PreparedMemoryBaseTraceError::SizeOverflow)?,
            small_part: prepared_part(&small_part)?,
        })
    }

    pub fn launch(&self) -> Result<(), PreparedMemoryBaseTraceError> {
        self.launch_on(self.arena.context().launch_context())
    }

    pub fn launch_on(&self, launch: CudaLaunchContext) -> Result<(), PreparedMemoryBaseTraceError> {
        if launch.identity_token() != self.arena.context().identity_token() {
            return Err(CudaRuntimeError::ContextMismatch.into());
        }
        let stream = launch.stream_raw().as_ptr();
        let address = unsafe {
            stwo_backend_cuda_kernels::raw::memory_address_base_trace_on(
                self.raw_addr_to_id.as_u32_ptr().cast_const(),
                self.n_addrs,
                self.address_counts.as_u32_ptr().cast_const(),
                self.address_count_words,
                self.address_rows,
                self.address_output_pointers.as_ptr(),
                stream,
            )
        };
        check_cuda("memory_address_base_trace_on", address)?;
        for part in &self.big_parts {
            self.launch_value_part(
                &self.big_source_pointers,
                EXECUTION_TABLE_BIG_LIMBS,
                self.big_source_words,
                self.big_counts,
                self.big_count_words,
                part,
                stream,
            )?;
        }
        self.launch_value_part(
            &self.small_source_pointers,
            EXECUTION_TABLE_SMALL_LIMBS,
            self.small_source_words,
            self.small_counts,
            self.small_count_words,
            &self.small_part,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_value_part(
        &self,
        sources: &[*const u32],
        n_limbs: usize,
        source_words: u32,
        counts: ArenaSlice,
        count_words: u32,
        part: &PreparedValuePart,
        stream: *mut core::ffi::c_void,
    ) -> Result<(), PreparedMemoryBaseTraceError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::memory_value_base_trace_on(
                sources.as_ptr(),
                n_limbs as u32,
                source_words,
                part.source_offset,
                counts.as_u32_ptr().cast_const(),
                count_words,
                part.row_count,
                part.output_pointers.as_ptr(),
                stream,
            )
        };
        check_cuda("memory_value_base_trace_on", code)?;
        Ok(())
    }

    pub fn kernel_launches(&self) -> usize {
        2 + self.big_parts.len()
    }
}

fn check_shape(
    role: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), PreparedMemoryBaseTraceError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PreparedMemoryBaseTraceError::ShapeMismatch {
            role,
            expected,
            actual,
        })
    }
}

fn validate_slice(
    role: &'static str,
    slice: ArenaSlice,
    required_words: usize,
    token: core::ptr::NonNull<core::ffi::c_void>,
) -> Result<(), PreparedMemoryBaseTraceError> {
    if slice.context_token() != token {
        return Err(PreparedMemoryBaseTraceError::ContextMismatch(role));
    }
    if slice.len_words() < required_words {
        return Err(PreparedMemoryBaseTraceError::SliceTooSmall {
            role,
            required_words,
            actual_words: slice.len_words(),
        });
    }
    Ok(())
}
