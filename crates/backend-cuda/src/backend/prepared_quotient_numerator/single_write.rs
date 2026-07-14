//! Disabled single-write schedule integration.
//!
//! The production constructor remains on legacy batches until the native CUDA
//! equality, sanitizer, and timing gates admit this schedule.

use core::ffi::c_void;

use super::*;
use crate::backend::quotient_numerator_single_write::{
    quotient_numerator_hybrid_plan, quotient_numerator_single_write_plan,
    QuotientNumeratorSingleWriteError,
};

impl<'a> PreparedQuotientNumeratorGraph<'a> {
    /// Experimental all-evaluation schedule. Setup reuses the validated legacy
    /// arena bindings and replaces only the flattened term descriptor payload.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_single_write_candidate(
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
    ) -> Result<Self, QuotientNumeratorSingleWriteError> {
        let topology = columns
            .iter()
            .map(QuotientNumeratorColumnTopology::from)
            .collect::<Vec<_>>();
        let candidate = quotient_numerator_single_write_plan(config, &topology)?;
        let mut prepared = Self::prepare(
            arena,
            config,
            columns,
            oods_sample_points,
            oods_sample_values,
            random_coefficient,
            sample_points_destination,
            first_linear_terms_destination,
            destinations,
            forward_twiddles,
            slots,
        )?;
        if candidate.requirements() != prepared.requirements() {
            return Err(QuotientNumeratorSingleWriteError::DescriptorInvariant(
                "candidate and prepared requirements differ",
            ));
        }
        upload_and_sync(
            arena,
            &[upload_u32(
                prepared.batch_terms,
                candidate.term_descriptors().to_vec(),
            )],
        )?;
        prepared.schedule = PreparedNumeratorSchedule::SingleWriteCandidate;
        Ok(prepared)
    }

    /// Experimental split schedule: evaluation-only groups are single-write;
    /// coefficient-backed groups retain the exact legacy batch chain.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_hybrid_candidate(
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
    ) -> Result<Self, QuotientNumeratorSingleWriteError> {
        let topology = columns
            .iter()
            .map(QuotientNumeratorColumnTopology::from)
            .collect::<Vec<_>>();
        let candidate = quotient_numerator_hybrid_plan(config, &topology)?;
        let mut prepared = Self::prepare(
            arena,
            config,
            columns,
            oods_sample_points,
            oods_sample_values,
            random_coefficient,
            sample_points_destination,
            first_linear_terms_destination,
            destinations,
            forward_twiddles,
            slots,
        )?;
        if candidate.requirements() != prepared.requirements() {
            return Err(QuotientNumeratorSingleWriteError::DescriptorInvariant(
                "hybrid and prepared requirements differ",
            ));
        }
        if candidate.batches().len() != prepared.batches.len() {
            return Err(QuotientNumeratorSingleWriteError::DescriptorInvariant(
                "hybrid and prepared batch counts differ",
            ));
        }
        let output_pointers = (0..4)
            .flat_map(|coordinate| {
                candidate.schedule_groups().iter().map(move |&group| {
                    destinations[group].coordinates[coordinate].as_u32_ptr() as usize
                })
            })
            .collect::<Vec<_>>();
        let output_log_sizes = candidate
            .schedule_groups()
            .iter()
            .map(|&group| prepared.requirements.groups[group].log_size)
            .collect::<Vec<_>>();
        upload_and_sync(
            arena,
            &[
                upload_u32(prepared.batch_terms, candidate.packed_terms().to_vec()),
                upload_u32(
                    prepared.batch_group_offsets,
                    candidate.packed_group_offsets().to_vec(),
                ),
                upload_ptrs(prepared.output_ptrs, output_pointers),
                upload_u32(prepared.output_log_sizes, output_log_sizes),
            ],
        )?;
        for (batch, placement) in prepared.batches.iter_mut().zip(candidate.batches()) {
            batch.term_offset = placement.term_offset;
            batch.group_offset = placement.group_offset;
        }
        let report = candidate.report();
        prepared.schedule = PreparedNumeratorSchedule::HybridCandidate {
            eligible_groups: report.eligible_group_count,
            legacy_groups: report.legacy_group_count,
        };
        Ok(prepared)
    }

    pub(super) fn launch_single_write_candidate(
        &self,
        group_offsets: ArenaSlice,
        group_count: u32,
        max_output_size: u32,
        stream: *mut c_void,
    ) -> Result<(), PreparedQuotientNumeratorError> {
        let output_table = |coordinate: usize| unsafe {
            self.output_ptrs
                .as_u32_ptr()
                .cast::<*mut u32>()
                .add(coordinate * self.requirements.groups.len())
        };
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_accumulate_quotient_numerator_single_write_on(
                group_offsets.as_u32_ptr(),
                self.batch_terms.as_u32_ptr(),
                group_count,
                max_output_size,
                self.batch_source_ptrs.as_u32_ptr().cast(),
                self.line_coefficients.as_u32_ptr().cast(),
                self.output_log_sizes.as_u32_ptr(),
                output_table(0),
                output_table(1),
                output_table(2),
                output_table(3),
                stream,
            )
        };
        check_cuda("prepared_quotient_numerator_single_write", code)?;
        Ok(())
    }
}
