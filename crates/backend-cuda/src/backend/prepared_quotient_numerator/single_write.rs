//! Disabled single-write schedule integration.
//!
//! The production constructor remains on legacy batches until the native CUDA
//! equality, sanitizer, and timing gates admit this schedule.

use core::ffi::c_void;

use super::*;
use crate::backend::quotient_numerator_prepacked_terms::{
    quotient_numerator_prepacked_plan_identity, quotient_numerator_prepacked_term_layout,
};
use crate::backend::quotient_numerator_single_write::{
    quotient_numerator_hybrid_plan, quotient_numerator_single_write_plan,
    QuotientNumeratorSingleWriteError,
};
use crate::backend::quotient_numerator_staged_single_write::{
    quotient_numerator_staged_single_write_plan_with_overflow_capacities,
    QuotientNumeratorStagedOperation, QuotientNumeratorStagedSingleWriteError,
    QuotientNumeratorStagedSource, QuotientNumeratorStagingRole,
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

    /// Test-only hot-loop replacement over the exact staged source manifest.
    ///
    /// The preparation kernel runs after group finalization, the last reader of
    /// `term_points`, and reuses only that dead extent. Production constructors
    /// do not select this schedule. Call [`Self::observe_prepacked_status`]
    /// after eager completion or every captured replay before accepting output.
    /// The quotient launch must therefore end its graph segment: embedding
    /// downstream commitments in the same graph would consume output before
    /// the host can observe the fail-closed status.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_staged_prepacked_single_write_candidate(
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
        overflow_roles: &[ArenaSlice],
    ) -> Result<Self, QuotientNumeratorStagedSingleWriteError> {
        let topology = columns
            .iter()
            .map(QuotientNumeratorColumnTopology::from)
            .collect::<Vec<_>>();
        let overflow_capacities = overflow_roles
            .iter()
            .map(|role| role.len_words())
            .collect::<Vec<_>>();
        let plan = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
            config,
            &topology,
            &overflow_capacities,
        )?;
        let layout = quotient_numerator_prepacked_term_layout(&plan)
            .map_err(PreparedQuotientNumeratorError::from)?;
        let plan_identity = quotient_numerator_prepacked_plan_identity(&plan)
            .map_err(PreparedQuotientNumeratorError::from)?;
        let source_count = u32::try_from(plan.sources().len())
            .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?;
        let used_words = u64::try_from(layout.used_words)
            .map_err(|_| PreparedQuotientNumeratorError::SizeOverflow)?;
        let packed_output_rows = plan.packed_output_rows();

        let mut prepared = Self::prepare_staged_packed_single_write(
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
            overflow_roles,
        )?;
        if prepared.schedule
            != (PreparedNumeratorSchedule::StagedPackedSingleWrite { packed_output_rows })
        {
            return Err(PreparedQuotientNumeratorError::PrepackedScheduleInvariant(
                "staged runtime and prepacked plan differ",
            )
            .into());
        }
        prepared.prepacked = Some(PreparedPrepackedBinding {
            receipt: PreparedPrepackedQuotientNumeratorReceipt {
                plan_identity,
                source_count,
                used_words,
                status_offset_words: layout.status_offset_words,
            },
        });
        prepared.schedule =
            PreparedNumeratorSchedule::StagedPrepackedSingleWrite { packed_output_rows };
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

    /// Replacement-v1 coefficient-inclusive schedule. Every coefficient LDE
    /// is materialized once into the primary factor-32 tile or one exact
    /// epoch-released overflow role, then one packed 1-D launch writes every
    /// quotient numerator exactly once. Overflow roles must be distinct and
    /// may not name any live workspace, source, destination, OODS, or twiddle
    /// slot.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_staged_packed_single_write(
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
        overflow_roles: &[ArenaSlice],
    ) -> Result<Self, QuotientNumeratorStagedSingleWriteError> {
        let topology = columns
            .iter()
            .map(QuotientNumeratorColumnTopology::from)
            .collect::<Vec<_>>();
        let canonical = build_plan(config, &topology)?;
        let overflow_capacities = overflow_roles
            .iter()
            .map(|role| role.len_words())
            .collect::<Vec<_>>();
        let candidate = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
            config,
            &topology,
            &overflow_capacities,
        )?;
        if candidate.requirements() != &canonical.requirements
            || candidate.group_offsets() != canonical.group_offsets.as_slice()
        {
            return Err(
                QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                    "staged and canonical numerator manifests differ",
                ),
            );
        }

        let expected_columns = canonical
            .batches
            .iter()
            .flat_map(|batch| batch.coefficient_columns.iter().copied())
            .collect::<Vec<_>>();
        if candidate
            .coefficient_ldes()
            .iter()
            .map(|lde| lde.column())
            .ne(expected_columns.iter().copied())
        {
            return Err(
                QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                    "staged LDE order differs from canonical batch order",
                ),
            );
        }
        let mut operations = candidate.operations().iter();
        let mut first_lde = 0usize;
        for batch in canonical
            .batches
            .iter()
            .filter(|batch| !batch.coefficient_columns.is_empty())
        {
            let Some(QuotientNumeratorStagedOperation::MaterializeLdes(launch)) = operations.next()
            else {
                return Err(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "staged LDE launch order differs from canonical batches",
                    ),
                );
            };
            if launch.evaluation_log_size() != batch.evaluation_log_size
                || launch.first_lde() != first_lde
                || launch.lde_count() != batch.coefficient_columns.len()
            {
                return Err(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "staged LDE launch geometry differs from canonical batch",
                    ),
                );
            }
            first_lde += launch.lde_count();
        }
        match operations.next() {
            Some(QuotientNumeratorStagedOperation::AccumulatePackedRows {
                group_count,
                term_count,
                packed_output_rows,
            }) if *group_count == canonical.requirements.groups.len()
                && *term_count == canonical.requirements.term_count
                && *packed_output_rows == candidate.packed_output_rows() => {}
            _ => {
                return Err(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "staged packed launch geometry differs from canonical manifest",
                    ),
                )
            }
        }
        if operations.next().is_some() || first_lde != candidate.coefficient_ldes().len() {
            return Err(
                QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                    "staged operation manifest has trailing or missing work",
                ),
            );
        }

        let expected_overflow_roles = candidate.overflow_role_words();
        if expected_overflow_roles.len() != overflow_roles.len() {
            return Err(
                QuotientNumeratorStagedSingleWriteError::OverflowBindingCountMismatch {
                    expected: expected_overflow_roles.len(),
                    actual: overflow_roles.len(),
                },
            );
        }
        for (required, role) in expected_overflow_roles.iter().zip(overflow_roles) {
            if *required > role.len_words() {
                return Err(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "staged overflow binding is smaller than its sealed used extent",
                    ),
                );
            }
        }

        let workspace_ids = canonical
            .requirements
            .arena_slot_requirements(slots)?
            .into_iter()
            .map(|requirement| requirement.id)
            .collect::<BTreeSet<_>>();
        let external_ids = [
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
        )
        .map(ArenaSlice::id)
        .collect::<BTreeSet<_>>();
        let context_token = arena.context().identity_token();
        for role in overflow_roles {
            if role.context_token() != context_token {
                return Err(PreparedQuotientNumeratorError::ContextMismatch(role.id()).into());
            }
        }
        validate_staged_overflow_role_ids(&workspace_ids, &external_ids, overflow_roles)?;

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
        if candidate.requirements() != prepared.requirements()
            || prepared.batches.len() != canonical.batches.len()
        {
            return Err(
                QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                    "staged and prepared numerator requirements differ",
                ),
            );
        }
        for (prepared_batch, canonical_batch) in prepared.batches.iter().zip(&canonical.batches) {
            if prepared_batch.evaluation_log_size != canonical_batch.evaluation_log_size
                || prepared_batch.coefficient_count != canonical_batch.coefficient_columns.len()
            {
                return Err(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "prepared coefficient batch differs from staged LDE manifest",
                    ),
                );
            }
        }

        if candidate.coefficient_ldes().is_empty() != prepared.lde_tile.is_none() {
            return Err(
                QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                    "staged coefficient manifest and primary LDE role presence differ",
                ),
            );
        }
        let primary = prepared.lde_tile;
        let role_slice = |role| -> Result<ArenaSlice, QuotientNumeratorStagedSingleWriteError> {
            match role {
                QuotientNumeratorStagingRole::Primary => primary.ok_or(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "staged coefficient manifest has no primary LDE role",
                    ),
                ),
                QuotientNumeratorStagingRole::Overflow(index) => {
                    overflow_roles.get(usize::from(index)).copied().ok_or(
                        QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                            "staged LDE names a missing overflow role",
                        ),
                    )
                }
            }
        };
        let coefficient_output_pointers = candidate
            .coefficient_ldes()
            .iter()
            .map(|lde| {
                let role = role_slice(lde.staging_role())?;
                if lde.role_end_words() > role.len_words() {
                    return Err(
                        QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                            "staged LDE crosses its physical role extent",
                        ),
                    );
                }
                Ok(unsafe { role.as_u32_ptr().add(lde.role_offset_words()) as usize })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_pointers = candidate
            .sources()
            .iter()
            .map(|source| match *source {
                QuotientNumeratorStagedSource::Evaluation { column, .. } => {
                    let QuotientNumeratorColumnSource::Evaluation(slice) = columns[column].source
                    else {
                        return Err(
                            QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                                "staged evaluation source is not a retained evaluation",
                            ),
                        );
                    };
                    Ok(slice.as_u32_ptr() as usize)
                }
                QuotientNumeratorStagedSource::StagedCoefficient(lde) => {
                    let role = role_slice(lde.staging_role())?;
                    Ok(unsafe { role.as_u32_ptr().add(lde.role_offset_words()) as usize })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut uploads = vec![
            upload_u32(prepared.batch_terms, candidate.term_descriptors().to_vec()),
            upload_u64(
                prepared.batch_group_offsets,
                candidate.packed_group_row_offsets().to_vec(),
            ),
            upload_ptrs(prepared.batch_source_ptrs, source_pointers),
        ];
        match (
            prepared.coefficient_output_ptrs,
            coefficient_output_pointers.is_empty(),
        ) {
            (Some(slot), false) => uploads.push(upload_ptrs(slot, coefficient_output_pointers)),
            (None, true) => {}
            _ => {
                return Err(
                    QuotientNumeratorStagedSingleWriteError::DescriptorInvariant(
                        "staged coefficient outputs and pointer-table presence differ",
                    ),
                )
            }
        }
        upload_and_sync(arena, &uploads)?;
        prepared.schedule = PreparedNumeratorSchedule::StagedPackedSingleWrite {
            packed_output_rows: candidate.packed_output_rows(),
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

    pub(super) fn launch_packed_single_write(
        &self,
        packed_output_rows: u64,
        stream: *mut c_void,
    ) -> Result<(), PreparedQuotientNumeratorError> {
        let group_count = u32::try_from(self.requirements.groups.len()).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyGroups(self.requirements.groups.len())
        })?;
        let output_table = |coordinate: usize| unsafe {
            self.output_ptrs
                .as_u32_ptr()
                .cast::<*mut u32>()
                .add(coordinate * self.requirements.groups.len())
        };
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_accumulate_quotient_numerator_packed_single_write_on(
                self.batch_group_offsets.as_u32_ptr().cast(),
                self.group_offsets.as_u32_ptr(),
                self.batch_terms.as_u32_ptr(),
                group_count,
                packed_output_rows,
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
        check_cuda("prepared_quotient_numerator_packed_single_write", code)?;
        Ok(())
    }

    pub(super) fn prepare_prepacked_terms(
        &self,
        stream: *mut c_void,
    ) -> Result<(), PreparedQuotientNumeratorError> {
        let binding =
            self.prepacked
                .ok_or(PreparedQuotientNumeratorError::PrepackedScheduleInvariant(
                    "prepacked schedule has no sealed binding",
                ))?;
        let group_count = u32::try_from(self.requirements.groups.len()).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyGroups(self.requirements.groups.len())
        })?;
        let term_count = u32::try_from(self.requirements.term_count).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyTerms(self.requirements.term_count)
        })?;
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_prepare_quotient_numerator_prepacked_terms_on(
                self.group_offsets.as_u32_ptr(),
                self.batch_terms.as_u32_ptr(),
                group_count,
                term_count,
                self.batch_source_ptrs.as_u32_ptr().cast(),
                binding.receipt.source_count,
                self.line_coefficients.as_u32_ptr().cast(),
                self.term_points.as_u32_ptr(),
                binding.receipt.used_words,
                stream,
            )
        };
        check_cuda("prepared_quotient_numerator_prepacked_terms", code)?;
        Ok(())
    }

    pub(super) fn launch_prepacked_single_write(
        &self,
        packed_output_rows: u64,
        stream: *mut c_void,
    ) -> Result<(), PreparedQuotientNumeratorError> {
        let binding =
            self.prepacked
                .ok_or(PreparedQuotientNumeratorError::PrepackedScheduleInvariant(
                    "prepacked schedule has no sealed binding",
                ))?;
        let group_count = u32::try_from(self.requirements.groups.len()).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyGroups(self.requirements.groups.len())
        })?;
        let term_count = u32::try_from(self.requirements.term_count).map_err(|_| {
            PreparedQuotientNumeratorError::TooManyTerms(self.requirements.term_count)
        })?;
        let output_table = |coordinate: usize| unsafe {
            self.output_ptrs
                .as_u32_ptr()
                .cast::<*mut u32>()
                .add(coordinate * self.requirements.groups.len())
        };
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::
                stwo_accumulate_quotient_numerator_prepacked_single_write_on(
                    self.batch_group_offsets.as_u32_ptr().cast(),
                    self.group_offsets.as_u32_ptr(),
                    group_count,
                    term_count,
                    packed_output_rows,
                    self.term_points.as_u32_ptr(),
                    binding.receipt.used_words,
                    self.output_log_sizes.as_u32_ptr(),
                    output_table(0),
                    output_table(1),
                    output_table(2),
                    output_table(3),
                    stream,
                )
        };
        check_cuda("prepared_quotient_numerator_prepacked_single_write", code)?;
        Ok(())
    }

    /// Status fence for the test-only prepacked schedule.
    ///
    /// This copies one word and synchronizes the proof stream. The captured
    /// schedule resets that word before every replay; callers must obtain `Ok`
    /// here at the quotient boundary before reading, committing, or otherwise
    /// consuming any candidate output.
    pub fn observe_prepacked_status(&self) -> Result<(), PreparedQuotientNumeratorError> {
        let binding =
            self.prepacked
                .ok_or(PreparedQuotientNumeratorError::PrepackedScheduleInvariant(
                    "status observation requires the prepacked schedule",
                ))?;
        let mut status = 0u32;
        let source = unsafe {
            self.term_points
                .as_u32_ptr()
                .add(binding.receipt.status_offset_words)
        };
        unsafe {
            self.arena.context().memcpy_d2h_async(
                (&mut status as *mut u32).cast(),
                source.cast(),
                core::mem::size_of::<u32>(),
            )?;
        }
        self.arena.context().sync()?;
        if status != 0 {
            return Err(PreparedQuotientNumeratorError::PrepackedDeviceStatus(
                status,
            ));
        }
        Ok(())
    }
}

fn validate_staged_overflow_role_ids(
    workspace_ids: &BTreeSet<ArenaSlotId>,
    external_ids: &BTreeSet<ArenaSlotId>,
    overflow_roles: &[ArenaSlice],
) -> Result<(), QuotientNumeratorStagedSingleWriteError> {
    let mut role_ids = BTreeSet::new();
    for role in overflow_roles {
        if workspace_ids.contains(&role.id()) {
            return Err(PreparedQuotientNumeratorError::ExternalAliasesWorkspace(role.id()).into());
        }
        if external_ids.contains(&role.id()) || !role_ids.insert(role.id()) {
            return Err(PreparedQuotientNumeratorError::AliasedExternalSlot(role.id()).into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod staged_binding_tests {
    use super::*;

    #[test]
    fn overflow_roles_reject_source_destination_and_twiddle_aliases() {
        let workspace_ids = BTreeSet::new();
        let source = ArenaSlice::dangling_for_test(11, 16);
        let destination = ArenaSlice::dangling_for_test(12, 16);
        let twiddles = ArenaSlice::dangling_for_test(13, 16);
        for external in [source, destination, twiddles] {
            let id = external.id();
            let external_ids = BTreeSet::from([id]);
            assert!(matches!(
                validate_staged_overflow_role_ids(&workspace_ids, &external_ids, &[external]),
                Err(QuotientNumeratorStagedSingleWriteError::Base(
                    PreparedQuotientNumeratorError::AliasedExternalSlot(actual)
                )) if actual == id
            ));
        }
    }

    #[test]
    fn overflow_roles_reject_workspace_and_role_aliases() {
        let workspace_id = ArenaSlotId(21);
        let workspace_ids = BTreeSet::from([workspace_id]);
        let workspace_role = ArenaSlice::dangling_for_test(21, 16);
        assert!(matches!(
            validate_staged_overflow_role_ids(&workspace_ids, &BTreeSet::new(), &[workspace_role]),
            Err(QuotientNumeratorStagedSingleWriteError::Base(
                PreparedQuotientNumeratorError::ExternalAliasesWorkspace(actual)
            )) if actual == workspace_id
        ));

        let duplicate = ArenaSlice::dangling_for_test(22, 16);
        assert!(matches!(
            validate_staged_overflow_role_ids(
                &BTreeSet::new(),
                &BTreeSet::new(),
                &[duplicate, duplicate],
            ),
            Err(QuotientNumeratorStagedSingleWriteError::Base(
                PreparedQuotientNumeratorError::AliasedExternalSlot(ArenaSlotId(22))
            ))
        ));
    }
}
