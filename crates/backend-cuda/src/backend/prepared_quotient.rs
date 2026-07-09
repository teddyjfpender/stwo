//! Prepared arena-native quotient combination and LDE into FRI input layout.
//!
//! The caller computes partial-numerator coordinate columns and supplies exact
//! OODS constants. Setup uploads stable pointer/log descriptors once;
//! [`PreparedQuotientGraph::launch`] combines on the quotient subdomain,
//! interpolates four M31 coordinates in place, and evaluates them directly into
//! the contiguous full-domain layout consumed by prepared FRI.

use core::ffi::c_void;
use std::collections::BTreeSet;

use stwo::core::circle::CirclePoint;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;

use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena,
};
use crate::columns::bindings::{CirclePointSecureField, CudaSecureField};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const SECURE_COORDINATES: usize = 4;
const COMPLEX_COORDINATES: usize = 2;
const POINTER_WORDS: usize = core::mem::size_of::<*mut u32>().div_ceil(WORD_BYTES);

pub const QUOTIENT_POINTER_ALIGNMENT_WORDS: usize = core::mem::align_of::<*mut u32>() / WORD_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientWorkspaceConfig {
    pub lifting_log_size: u32,
    pub log_blowup_factor: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotientWorkspaceRequirements {
    pub config: QuotientWorkspaceConfig,
    pub subdomain_log_size: u32,
    pub sample_count: usize,
    pub sample_point_words: usize,
    pub first_linear_term_words: usize,
    pub partial_log_size_words: usize,
    pub partial_pointer_words: usize,
    pub coordinate_pointer_words: usize,
    pub coefficient_size_words: usize,
    pub subdomain_value_words: usize,
    pub output_value_words: usize,
    pub denominator_words: usize,
    pub forward_twiddle_words: usize,
    pub inverse_twiddle_words: usize,
    pub half_coset_initial_index: u32,
    pub half_coset_step_size: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientWorkspaceSlots {
    pub sample_points: ArenaSlotId,
    pub first_linear_terms: ArenaSlotId,
    pub partial_log_sizes: ArenaSlotId,
    pub partial_coordinate_ptrs: ArenaSlotId,
    pub subdomain_coordinate_ptrs: ArenaSlotId,
    pub output_coordinate_ptrs: ArenaSlotId,
    pub coefficient_sizes: ArenaSlotId,
    pub subdomain_values: ArenaSlotId,
    pub output_values: ArenaSlotId,
    pub denominator_scratch: ArenaSlotId,
}

#[derive(Clone, Copy, Debug)]
pub struct QuotientSampleConstants {
    pub sample_point: CirclePoint<SecureField>,
    pub first_linear_term_acc: SecureField,
}

#[derive(Clone, Copy, Debug)]
pub struct QuotientNumeratorSource {
    pub constants: QuotientSampleConstants,
    pub log_size: u32,
    /// M31 coordinate columns in canonical QM31 order.
    pub coordinates: [ArenaSlice; SECURE_COORDINATES],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedQuotientError {
    InvalidLiftingLogSize(u32),
    InvalidBlowup {
        lifting_log_size: u32,
        log_blowup_factor: u32,
    },
    EmptySources,
    TooManySources(usize),
    PartialLogSizeTooLarge {
        source: usize,
        log_size: u32,
        subdomain_log_size: u32,
    },
    ConstantsCountMismatch {
        expected: usize,
        actual: usize,
    },
    DuplicateSlot(ArenaSlotId),
    ContextMismatch(ArenaSlotId),
    SourceAliasesWorkspace(ArenaSlotId),
    AliasedTwiddles(ArenaSlotId),
    AliasedSourceSlot(ArenaSlotId),
    SourceTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    SlotTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    MisalignedSlot {
        slot: ArenaSlotId,
        alignment_words: usize,
    },
    ForwardTwiddlesTooSmall {
        required_words: usize,
        actual_words: usize,
    },
    InverseTwiddlesTooSmall {
        required_words: usize,
        actual_words: usize,
    },
    SizeOverflow,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedQuotientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid prepared CUDA quotient workspace: {self:?}")
    }
}

impl std::error::Error for PreparedQuotientError {}

impl From<ArenaError> for PreparedQuotientError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedQuotientError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

pub fn quotient_workspace_requirements(
    config: QuotientWorkspaceConfig,
    partial_log_sizes: &[u32],
) -> Result<QuotientWorkspaceRequirements, PreparedQuotientError> {
    if !(2..=30).contains(&config.lifting_log_size) {
        return Err(PreparedQuotientError::InvalidLiftingLogSize(
            config.lifting_log_size,
        ));
    }
    if config.log_blowup_factor == 0 || config.log_blowup_factor >= config.lifting_log_size {
        return Err(PreparedQuotientError::InvalidBlowup {
            lifting_log_size: config.lifting_log_size,
            log_blowup_factor: config.log_blowup_factor,
        });
    }
    if partial_log_sizes.is_empty() {
        return Err(PreparedQuotientError::EmptySources);
    }
    let sample_count = partial_log_sizes.len();
    let _ = u32::try_from(sample_count)
        .map_err(|_| PreparedQuotientError::TooManySources(sample_count))?;
    let subdomain_log_size = config.lifting_log_size - config.log_blowup_factor;
    for (source, &log_size) in partial_log_sizes.iter().enumerate() {
        if log_size > subdomain_log_size {
            return Err(PreparedQuotientError::PartialLogSizeTooLarge {
                source,
                log_size,
                subdomain_log_size,
            });
        }
    }

    let full_domain = pow2(config.lifting_log_size)?;
    let subdomain = pow2(subdomain_log_size)?;
    let eval_domain = CanonicCoset::new(config.lifting_log_size).circle_domain();
    let (quotient_domain, _) = eval_domain.split(config.log_blowup_factor);
    Ok(QuotientWorkspaceRequirements {
        config,
        subdomain_log_size,
        sample_count,
        sample_point_words: typed_words::<CirclePointSecureField>(sample_count)?,
        first_linear_term_words: typed_words::<CudaSecureField>(sample_count)?,
        partial_log_size_words: sample_count,
        partial_pointer_words: sample_count
            .checked_mul(SECURE_COORDINATES)
            .and_then(|count| count.checked_mul(POINTER_WORDS))
            .ok_or(PreparedQuotientError::SizeOverflow)?,
        coordinate_pointer_words: SECURE_COORDINATES
            .checked_mul(POINTER_WORDS)
            .ok_or(PreparedQuotientError::SizeOverflow)?,
        coefficient_size_words: SECURE_COORDINATES,
        subdomain_value_words: subdomain
            .checked_mul(SECURE_COORDINATES)
            .ok_or(PreparedQuotientError::SizeOverflow)?,
        output_value_words: full_domain
            .checked_mul(SECURE_COORDINATES)
            .ok_or(PreparedQuotientError::SizeOverflow)?,
        denominator_words: subdomain
            .checked_mul(sample_count)
            .and_then(|count| count.checked_mul(COMPLEX_COORDINATES))
            .ok_or(PreparedQuotientError::SizeOverflow)?,
        forward_twiddle_words: pow2(config.lifting_log_size - 1)?,
        inverse_twiddle_words: pow2(subdomain_log_size - 1)?,
        half_coset_initial_index: quotient_domain.half_coset.initial_index.0 as u32,
        half_coset_step_size: quotient_domain.half_coset.step_size.0 as u32,
    })
}

impl QuotientWorkspaceRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: &QuotientWorkspaceSlots,
    ) -> Result<Vec<QuotientArenaSlotRequirement>, PreparedQuotientError> {
        let requirements = vec![
            slot(slots.sample_points, self.sample_point_words, 1),
            slot(slots.first_linear_terms, self.first_linear_term_words, 1),
            slot(slots.partial_log_sizes, self.partial_log_size_words, 1),
            slot(
                slots.partial_coordinate_ptrs,
                self.partial_pointer_words,
                QUOTIENT_POINTER_ALIGNMENT_WORDS,
            ),
            slot(
                slots.subdomain_coordinate_ptrs,
                self.coordinate_pointer_words,
                QUOTIENT_POINTER_ALIGNMENT_WORDS,
            ),
            slot(
                slots.output_coordinate_ptrs,
                self.coordinate_pointer_words,
                QUOTIENT_POINTER_ALIGNMENT_WORDS,
            ),
            slot(slots.coefficient_sizes, self.coefficient_size_words, 1),
            slot(slots.subdomain_values, self.subdomain_value_words, 1),
            slot(slots.output_values, self.output_value_words, 1),
            slot(
                slots.denominator_scratch,
                self.denominator_words,
                COMPLEX_COORDINATES,
            ),
        ];
        let mut seen = BTreeSet::new();
        for requirement in &requirements {
            if !seen.insert(requirement.id) {
                return Err(PreparedQuotientError::DuplicateSlot(requirement.id));
            }
        }
        Ok(requirements)
    }
}

fn slot(id: ArenaSlotId, len_words: usize, alignment_words: usize) -> QuotientArenaSlotRequirement {
    QuotientArenaSlotRequirement {
        id,
        len_words,
        alignment_words,
    }
}

enum HostDescriptor {
    U32(Vec<u32>),
    Pointers(Vec<usize>),
    Points(Vec<CirclePointSecureField>),
    Secure(Vec<CudaSecureField>),
}

impl HostDescriptor {
    fn bytes(&self) -> (*const c_void, usize) {
        match self {
            Self::U32(values) => typed_bytes(values),
            Self::Pointers(values) => typed_bytes(values),
            Self::Points(values) => typed_bytes(values),
            Self::Secure(values) => typed_bytes(values),
        }
    }
}

struct PendingUpload {
    destination: ArenaSlice,
    descriptor: HostDescriptor,
}

/// Stable quotient-to-FRI launch object. Launch never allocates, uploads, frees,
/// synchronizes, or touches the default stream.
pub struct PreparedQuotientGraph<'a> {
    arena: &'a DeviceArena,
    requirements: QuotientWorkspaceRequirements,
    sample_points: ArenaSlice,
    first_linear_terms: ArenaSlice,
    partial_log_sizes: ArenaSlice,
    partial_coordinate_ptrs: ArenaSlice,
    subdomain_coordinate_ptrs: ArenaSlice,
    output_coordinate_ptrs: ArenaSlice,
    coefficient_sizes: ArenaSlice,
    subdomain_values: ArenaSlice,
    output_values: ArenaSlice,
    denominator_scratch: ArenaSlice,
    forward_twiddles: ArenaSlice,
    inverse_twiddles: ArenaSlice,
}

impl<'a> PreparedQuotientGraph<'a> {
    pub fn prepare(
        arena: &'a DeviceArena,
        config: QuotientWorkspaceConfig,
        sources: &[QuotientNumeratorSource],
        forward_twiddles: ArenaSlice,
        inverse_subdomain_twiddles: ArenaSlice,
        slots: &QuotientWorkspaceSlots,
    ) -> Result<Self, PreparedQuotientError> {
        let logs: Vec<_> = sources.iter().map(|source| source.log_size).collect();
        let requirements = quotient_workspace_requirements(config, &logs)?;
        let slot_requirements = requirements.arena_slot_requirements(slots)?;
        let workspace_ids: BTreeSet<_> = slot_requirements.iter().map(|entry| entry.id).collect();
        let context_token = arena.context().identity_token();

        if forward_twiddles.id() == inverse_subdomain_twiddles.id() {
            return Err(PreparedQuotientError::AliasedTwiddles(
                forward_twiddles.id(),
            ));
        }
        let mut source_ids =
            BTreeSet::from([forward_twiddles.id(), inverse_subdomain_twiddles.id()]);
        for source in [forward_twiddles, inverse_subdomain_twiddles]
            .into_iter()
            .chain(sources.iter().flat_map(|source| source.coordinates))
        {
            if source.context_token() != context_token {
                return Err(PreparedQuotientError::ContextMismatch(source.id()));
            }
            if workspace_ids.contains(&source.id()) {
                return Err(PreparedQuotientError::SourceAliasesWorkspace(source.id()));
            }
            if source.id() != forward_twiddles.id()
                && source.id() != inverse_subdomain_twiddles.id()
                && !source_ids.insert(source.id())
            {
                return Err(PreparedQuotientError::AliasedSourceSlot(source.id()));
            }
        }
        if forward_twiddles.len_words() < requirements.forward_twiddle_words {
            return Err(PreparedQuotientError::ForwardTwiddlesTooSmall {
                required_words: requirements.forward_twiddle_words,
                actual_words: forward_twiddles.len_words(),
            });
        }
        if inverse_subdomain_twiddles.len_words() < requirements.inverse_twiddle_words {
            return Err(PreparedQuotientError::InverseTwiddlesTooSmall {
                required_words: requirements.inverse_twiddle_words,
                actual_words: inverse_subdomain_twiddles.len_words(),
            });
        }
        for source in sources {
            let required_words = pow2(source.log_size)?;
            for coordinate in source.coordinates {
                if coordinate.len_words() < required_words {
                    return Err(PreparedQuotientError::SourceTooSmall {
                        slot: coordinate.id(),
                        required_words,
                        actual_words: coordinate.len_words(),
                    });
                }
            }
        }

        let sample_points = bind_slot(
            arena,
            slots.sample_points,
            requirements.sample_point_words,
            1,
        )?;
        let first_linear_terms = bind_slot(
            arena,
            slots.first_linear_terms,
            requirements.first_linear_term_words,
            1,
        )?;
        let partial_log_sizes = bind_slot(
            arena,
            slots.partial_log_sizes,
            requirements.partial_log_size_words,
            1,
        )?;
        let partial_coordinate_ptrs = bind_slot(
            arena,
            slots.partial_coordinate_ptrs,
            requirements.partial_pointer_words,
            QUOTIENT_POINTER_ALIGNMENT_WORDS,
        )?;
        let subdomain_coordinate_ptrs = bind_slot(
            arena,
            slots.subdomain_coordinate_ptrs,
            requirements.coordinate_pointer_words,
            QUOTIENT_POINTER_ALIGNMENT_WORDS,
        )?;
        let output_coordinate_ptrs = bind_slot(
            arena,
            slots.output_coordinate_ptrs,
            requirements.coordinate_pointer_words,
            QUOTIENT_POINTER_ALIGNMENT_WORDS,
        )?;
        let coefficient_sizes = bind_slot(
            arena,
            slots.coefficient_sizes,
            requirements.coefficient_size_words,
            1,
        )?;
        let subdomain_values = bind_slot(
            arena,
            slots.subdomain_values,
            requirements.subdomain_value_words,
            1,
        )?;
        let output_values = bind_slot(
            arena,
            slots.output_values,
            requirements.output_value_words,
            1,
        )?;
        let denominator_scratch = bind_slot(
            arena,
            slots.denominator_scratch,
            requirements.denominator_words,
            COMPLEX_COORDINATES,
        )?;

        let subdomain_stride = pow2(requirements.subdomain_log_size)?;
        let output_stride = pow2(requirements.config.lifting_log_size)?;
        let partial_pointers = (0..SECURE_COORDINATES)
            .flat_map(|coordinate| {
                sources
                    .iter()
                    .map(move |source| source.coordinates[coordinate].as_u32_ptr() as usize)
            })
            .collect();
        let subdomain_pointers = coordinate_pointers(subdomain_values, subdomain_stride);
        let output_pointers = coordinate_pointers(output_values, output_stride);
        let constants: Vec<_> = sources.iter().map(|source| source.constants).collect();
        let uploads = vec![
            points_upload(sample_points, &constants),
            secure_upload(first_linear_terms, &constants),
            PendingUpload {
                destination: partial_log_sizes,
                descriptor: HostDescriptor::U32(logs),
            },
            PendingUpload {
                destination: partial_coordinate_ptrs,
                descriptor: HostDescriptor::Pointers(partial_pointers),
            },
            PendingUpload {
                destination: subdomain_coordinate_ptrs,
                descriptor: HostDescriptor::Pointers(subdomain_pointers),
            },
            PendingUpload {
                destination: output_coordinate_ptrs,
                descriptor: HostDescriptor::Pointers(output_pointers),
            },
            PendingUpload {
                destination: coefficient_sizes,
                descriptor: HostDescriptor::U32(vec![
                    u32::try_from(subdomain_stride).map_err(
                        |_| PreparedQuotientError::SizeOverflow
                    )?;
                    SECURE_COORDINATES
                ]),
            },
        ];
        upload_and_sync(arena, &uploads)?;

        Ok(Self {
            arena,
            requirements,
            sample_points,
            first_linear_terms,
            partial_log_sizes,
            partial_coordinate_ptrs,
            subdomain_coordinate_ptrs,
            output_coordinate_ptrs,
            coefficient_sizes,
            subdomain_values,
            output_values,
            denominator_scratch,
            forward_twiddles,
            inverse_twiddles: inverse_subdomain_twiddles,
        })
    }

    pub fn requirements(&self) -> &QuotientWorkspaceRequirements {
        &self.requirements
    }

    /// Update only transcript-derived constants between graph replays.
    pub fn upload_constants_at_transcript_boundary(
        &self,
        constants: &[QuotientSampleConstants],
    ) -> Result<(), PreparedQuotientError> {
        if constants.len() != self.requirements.sample_count {
            return Err(PreparedQuotientError::ConstantsCountMismatch {
                expected: self.requirements.sample_count,
                actual: constants.len(),
            });
        }
        upload_and_sync(
            self.arena,
            &[
                points_upload(self.sample_points, constants),
                secure_upload(self.first_linear_terms, constants),
            ],
        )
    }

    /// Combine, interpolate, and evaluate using only the arena's explicit stream.
    pub fn launch(&self) -> Result<(), PreparedQuotientError> {
        let sample_count = u32::try_from(self.requirements.sample_count)
            .map_err(|_| PreparedQuotientError::SizeOverflow)?;
        let subdomain_size = u32::try_from(pow2(self.requirements.subdomain_log_size)?)
            .map_err(|_| PreparedQuotientError::SizeOverflow)?;
        let partial_ptrs = self
            .partial_coordinate_ptrs
            .as_u32_ptr()
            .cast::<*const u32>();
        let subdomain_ptrs = self
            .subdomain_coordinate_ptrs
            .as_u32_ptr()
            .cast::<*mut u32>();
        let output_ptrs = self.output_coordinate_ptrs.as_u32_ptr().cast::<*mut u32>();
        let stream = self.arena.context().stream_raw().as_ptr();
        let partials = |coordinate: usize| unsafe {
            partial_ptrs.add(coordinate * self.requirements.sample_count)
        };
        let subdomain_coordinate = |coordinate: usize| unsafe {
            self.subdomain_values
                .as_u32_ptr()
                .add(coordinate * subdomain_size as usize)
        };

        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_combine_quotients_from_numerators_on(
                self.requirements.half_coset_initial_index,
                self.requirements.half_coset_step_size,
                subdomain_size,
                self.requirements.subdomain_log_size,
                self.sample_points.as_u32_ptr().cast_const(),
                sample_count,
                self.first_linear_terms.as_u32_ptr().cast(),
                self.partial_log_sizes.as_u32_ptr().cast_const(),
                partials(0),
                partials(1),
                partials(2),
                partials(3),
                subdomain_coordinate(0),
                subdomain_coordinate(1),
                subdomain_coordinate(2),
                subdomain_coordinate(3),
                self.denominator_scratch.as_u32_ptr(),
                u64::try_from(self.requirements.denominator_words / COMPLEX_COORDINATES)
                    .map_err(|_| PreparedQuotientError::SizeOverflow)?,
                stream,
            )
        };
        check_cuda("prepared_quotient_combine", code)?;

        let inverse_words = u32::try_from(self.inverse_twiddles.len_words())
            .map_err(|_| PreparedQuotientError::SizeOverflow)?;
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_ntt_b2n_columns_on(
                subdomain_ptrs,
                self.requirements.subdomain_log_size,
                SECURE_COORDINATES as u32,
                self.inverse_twiddles.as_u32_ptr(),
                inverse_words,
                1u32 << (self.requirements.subdomain_log_size - 1),
                stream,
            )
        };
        check_cuda("prepared_quotient_interpolate", code)?;

        let forward_words = u32::try_from(self.forward_twiddles.len_words())
            .map_err(|_| PreparedQuotientError::SizeOverflow)?;
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_lde_n2b_columns_on(
                subdomain_ptrs.cast::<*const u32>(),
                self.coefficient_sizes.as_u32_ptr().cast_const(),
                output_ptrs,
                self.requirements.config.lifting_log_size,
                SECURE_COORDINATES as u32,
                self.forward_twiddles.as_u32_ptr(),
                forward_words,
                1u32 << (self.requirements.config.lifting_log_size - 1),
                stream,
            )
        };
        check_cuda("prepared_quotient_evaluate", code)?;
        Ok(())
    }

    /// Contiguous `[coord0 | coord1 | coord2 | coord3]` full-domain evaluation.
    pub const fn output_evaluation(&self) -> ArenaSlice {
        self.output_values
    }
}

fn points_upload(destination: ArenaSlice, constants: &[QuotientSampleConstants]) -> PendingUpload {
    PendingUpload {
        destination,
        descriptor: HostDescriptor::Points(
            constants
                .iter()
                .map(|value| CirclePointSecureField::from(value.sample_point))
                .collect(),
        ),
    }
}

fn secure_upload(destination: ArenaSlice, constants: &[QuotientSampleConstants]) -> PendingUpload {
    PendingUpload {
        destination,
        descriptor: HostDescriptor::Secure(
            constants
                .iter()
                .map(|value| CudaSecureField::from(value.first_linear_term_acc))
                .collect(),
        ),
    }
}

fn coordinate_pointers(values: ArenaSlice, stride: usize) -> Vec<usize> {
    (0..SECURE_COORDINATES)
        .map(|coordinate| unsafe { values.as_u32_ptr().add(coordinate * stride) as usize })
        .collect()
}

fn typed_words<T>(count: usize) -> Result<usize, PreparedQuotientError> {
    core::mem::size_of::<T>()
        .checked_mul(count)
        .and_then(|bytes| bytes.checked_add(WORD_BYTES - 1))
        .map(|bytes| bytes / WORD_BYTES)
        .ok_or(PreparedQuotientError::SizeOverflow)
}

fn pow2(log_size: u32) -> Result<usize, PreparedQuotientError> {
    1usize
        .checked_shl(log_size)
        .ok_or(PreparedQuotientError::SizeOverflow)
}

fn typed_bytes<T>(values: &[T]) -> (*const c_void, usize) {
    (
        values.as_ptr().cast(),
        values.len().saturating_mul(core::mem::size_of::<T>()),
    )
}

fn bind_slot(
    arena: &DeviceArena,
    id: ArenaSlotId,
    required_words: usize,
    alignment_words: usize,
) -> Result<ArenaSlice, PreparedQuotientError> {
    let slice = arena.bind(id)?;
    if slice.len_words() < required_words {
        return Err(PreparedQuotientError::SlotTooSmall {
            slot: id,
            required_words,
            actual_words: slice.len_words(),
        });
    }
    if (slice.as_u32_ptr() as usize) % (alignment_words * WORD_BYTES) != 0 {
        return Err(PreparedQuotientError::MisalignedSlot {
            slot: id,
            alignment_words,
        });
    }
    Ok(slice)
}

fn upload_and_sync(
    arena: &DeviceArena,
    uploads: &[PendingUpload],
) -> Result<(), PreparedQuotientError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> QuotientWorkspaceConfig {
        QuotientWorkspaceConfig {
            lifting_log_size: 8,
            log_blowup_factor: 2,
        }
    }

    fn slots() -> QuotientWorkspaceSlots {
        let mut next = 1u32;
        let mut id = || {
            let value = ArenaSlotId(next);
            next += 1;
            value
        };
        QuotientWorkspaceSlots {
            sample_points: id(),
            first_linear_terms: id(),
            partial_log_sizes: id(),
            partial_coordinate_ptrs: id(),
            subdomain_coordinate_ptrs: id(),
            output_coordinate_ptrs: id(),
            coefficient_sizes: id(),
            subdomain_values: id(),
            output_values: id(),
            denominator_scratch: id(),
        }
    }

    #[test]
    fn requirements_match_exact_combine_interpolate_and_fri_layout() {
        let requirements = quotient_workspace_requirements(config(), &[6, 5]).unwrap();
        assert_eq!(requirements.subdomain_log_size, 6);
        assert_eq!(requirements.sample_point_words, 16);
        assert_eq!(requirements.first_linear_term_words, 8);
        assert_eq!(requirements.partial_pointer_words, 8 * POINTER_WORDS);
        assert_eq!(requirements.subdomain_value_words, 4 * 64);
        assert_eq!(requirements.output_value_words, 4 * 256);
        assert_eq!(requirements.denominator_words, 2 * 64 * 2);
        assert_eq!(requirements.forward_twiddle_words, 128);
        assert_eq!(requirements.inverse_twiddle_words, 32);
        assert_eq!(
            requirements
                .arena_slot_requirements(&slots())
                .unwrap()
                .len(),
            10
        );
    }

    #[test]
    fn requirements_reject_missing_or_out_of_domain_numerators() {
        assert_eq!(
            quotient_workspace_requirements(config(), &[]).unwrap_err(),
            PreparedQuotientError::EmptySources
        );
        assert_eq!(
            quotient_workspace_requirements(config(), &[7]).unwrap_err(),
            PreparedQuotientError::PartialLogSizeTooLarge {
                source: 0,
                log_size: 7,
                subdomain_log_size: 6
            }
        );
    }

    #[test]
    fn slot_plan_rejects_aliases() {
        let requirements = quotient_workspace_requirements(config(), &[6]).unwrap();
        let mut slots = slots();
        slots.output_values = slots.subdomain_values;
        assert!(matches!(
            requirements.arena_slot_requirements(&slots),
            Err(PreparedQuotientError::DuplicateSlot(_))
        ));
    }
}
