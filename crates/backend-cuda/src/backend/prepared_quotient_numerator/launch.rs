//! Warm quotient-numerator execution; setup and descriptor compilation live in
//! the parent module and `single_write` sibling.

use super::*;

impl PreparedQuotientNumeratorGraph<'_> {
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
        match self.schedule {
            PreparedNumeratorSchedule::SingleWriteCandidate => {
                self.launch_single_write_candidate(
                    self.group_offsets,
                    group_count,
                    max_output_size,
                    stream,
                )?;
                return Ok(());
            }
            PreparedNumeratorSchedule::HybridCandidate { legacy_groups, .. } => {
                self.launch_single_write_candidate(
                    self.batch_group_offsets,
                    group_count,
                    max_output_size,
                    stream,
                )?;
                if legacy_groups == 0 {
                    return Ok(());
                }
            }
            PreparedNumeratorSchedule::StagedPackedSingleWrite { packed_output_rows } => {
                self.launch_all_staged_ldes(stream)?;
                self.launch_packed_single_write(packed_output_rows, stream)?;
                return Ok(());
            }
            PreparedNumeratorSchedule::StagedPrepackedSingleWrite { packed_output_rows } => {
                // `groups` is the last term_points reader. Repack that exact
                // owner immediately, then materialize staged sources before
                // the validated hot consumer.
                self.prepare_prepacked_terms(stream)?;
                self.launch_all_staged_ldes(stream)?;
                self.launch_prepacked_single_write(packed_output_rows, stream)?;
                return Ok(());
            }
            PreparedNumeratorSchedule::LegacyBatches => {}
        }
        if matches!(self.schedule, PreparedNumeratorSchedule::LegacyBatches) {
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
        }

        let (accumulation_group_count, output_offset) = match self.schedule {
            PreparedNumeratorSchedule::HybridCandidate {
                eligible_groups,
                legacy_groups,
            } => (
                u32::try_from(legacy_groups)
                    .map_err(|_| PreparedQuotientNumeratorError::TooManyGroups(legacy_groups))?,
                eligible_groups,
            ),
            _ => (group_count, 0),
        };

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
                    accumulation_group_count,
                    max_output_size,
                    pointer_table(self.batch_source_ptrs, batch.source_ptr_offset),
                    self.line_coefficients.as_u32_ptr().cast(),
                    self.output_log_sizes.as_u32_ptr().add(output_offset),
                    output_tables(0).add(output_offset),
                    output_tables(1).add(output_offset),
                    output_tables(2).add(output_offset),
                    output_tables(3).add(output_offset),
                    stream,
                )
            };
            check_cuda("prepared_quotient_numerator_accumulate", code)?;
        }
        Ok(())
    }

    fn launch_all_staged_ldes(
        &self,
        stream: *mut core::ffi::c_void,
    ) -> Result<(), PreparedQuotientNumeratorError> {
        let pointer_table = |slice: ArenaSlice, offset: usize| unsafe {
            slice.as_u32_ptr().cast::<*const u32>().add(offset)
        };
        for batch in &self.batches {
            if batch.coefficient_count == 0 {
                continue;
            }
            let coefficient_ptrs = self.coefficient_ptrs.expect("slot shape validated");
            let coefficient_sizes = self.coefficient_sizes.expect("slot shape validated");
            let coefficient_outputs = self.coefficient_output_ptrs.expect("slot shape validated");
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
            check_cuda("prepared_quotient_numerator_staged_lde", code)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn prepacked_schedule_reuses_term_points_only_after_finalize() {
        let source = include_str!("launch.rs");
        let finalized = source
            .find("check_cuda(\"prepared_quotient_numerator_groups\", code)")
            .unwrap();
        let branch_start = source
            .find("PreparedNumeratorSchedule::StagedPrepackedSingleWrite")
            .unwrap();
        let branch_end = source[branch_start..]
            .find("PreparedNumeratorSchedule::LegacyBatches")
            .map(|offset| branch_start + offset)
            .unwrap();
        let branch = &source[branch_start..branch_end];
        let prepare = branch.find("self.prepare_prepacked_terms(stream)").unwrap();
        let lde = branch.find("self.launch_all_staged_ldes(stream)").unwrap();
        let hot = branch
            .find("self.launch_prepacked_single_write(packed_output_rows, stream)")
            .unwrap();

        assert!(finalized < branch_start);
        assert!(prepare < lde);
        assert!(lde < hot);
    }
}
