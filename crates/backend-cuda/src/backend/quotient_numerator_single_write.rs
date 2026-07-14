//! Candidate single-write quotient numerator schedule.
//!
//! This is deliberately not selected by the resident prover yet. It is safe
//! only when every sampled polynomial already has a retained evaluation: a
//! coefficient-backed source is materialized into a reused LDE tile, so its
//! pointer cannot survive a single launch spanning the old batches.

use super::prepared_quotient_numerator::{
    build_plan, PreparedQuotientNumeratorError, QuotientNumeratorColumnTopology,
    QuotientNumeratorWorkspaceConfig, QuotientNumeratorWorkspaceRequirements,
};

pub const QUOTIENT_NUMERATOR_SINGLE_WRITE_TERM_WORDS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotientNumeratorSingleWriteEligibility {
    Eligible,
    RequiresRetainedEvaluations {
        coefficient_columns: usize,
        coefficient_batches: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorSingleWriteReport {
    pub eligibility: QuotientNumeratorSingleWriteEligibility,
    pub group_count: usize,
    pub term_count: usize,
    pub source_count: usize,
    pub legacy_batch_count: usize,
    /// Zero plus one read/modify/write pass per legacy batch.
    pub legacy_output_passes: usize,
    pub candidate_output_passes: usize,
    pub output_rows: usize,
    /// Logical output loads and stores issued by the current kernels. This is
    /// not an HBM-counter measurement and excludes source/descriptor reads.
    pub legacy_logical_output_bytes: u64,
    pub candidate_logical_output_bytes: u64,
    pub legacy_descriptor_bytes: u64,
    pub candidate_descriptor_bytes: u64,
    /// Replay performs no descriptor construction or host transfer after setup.
    pub candidate_warm_host_preparation_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotientNumeratorSingleWritePlan {
    requirements: QuotientNumeratorWorkspaceRequirements,
    source_columns: Vec<usize>,
    group_offsets: Vec<u32>,
    term_descriptors: Vec<u32>,
    report: QuotientNumeratorSingleWriteReport,
}

impl QuotientNumeratorSingleWritePlan {
    pub fn requirements(&self) -> &QuotientNumeratorWorkspaceRequirements {
        &self.requirements
    }

    /// Indices into the caller's canonical column slice. Pointer-table order
    /// is stable legacy batch order: `(evaluation_log_size, column_index)`.
    pub fn source_columns(&self) -> &[usize] {
        &self.source_columns
    }

    pub fn group_offsets(&self) -> &[u32] {
        &self.group_offsets
    }

    /// Group-major `[source, term, source_log_size]` words. Terms inside each
    /// group retain the exact legacy batch order and stable in-batch order.
    pub fn term_descriptors(&self) -> &[u32] {
        &self.term_descriptors
    }

    pub fn report(&self) -> QuotientNumeratorSingleWriteReport {
        self.report
    }
}

#[derive(Debug)]
pub enum QuotientNumeratorSingleWriteError {
    Base(PreparedQuotientNumeratorError),
    RequiresRetainedEvaluations {
        coefficient_columns: usize,
        coefficient_batches: usize,
    },
    DescriptorInvariant(&'static str),
}

impl core::fmt::Display for QuotientNumeratorSingleWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Base(error) => error.fmt(f),
            Self::RequiresRetainedEvaluations {
                coefficient_columns,
                coefficient_batches,
            } => write!(
                f,
                "single-write quotient numerator requires retained evaluations; \
                 {coefficient_columns} coefficient columns remain in \
                 {coefficient_batches} batches"
            ),
            Self::DescriptorInvariant(message) => {
                write!(f, "invalid single-write quotient descriptor: {message}")
            }
        }
    }
}

impl std::error::Error for QuotientNumeratorSingleWriteError {}

impl From<PreparedQuotientNumeratorError> for QuotientNumeratorSingleWriteError {
    fn from(value: PreparedQuotientNumeratorError) -> Self {
        Self::Base(value)
    }
}

pub fn quotient_numerator_single_write_report(
    config: QuotientNumeratorWorkspaceConfig,
    columns: &[QuotientNumeratorColumnTopology],
) -> Result<QuotientNumeratorSingleWriteReport, QuotientNumeratorSingleWriteError> {
    let plan = build_plan(config, columns)?;
    report(&plan.requirements)
}

pub fn quotient_numerator_single_write_plan(
    config: QuotientNumeratorWorkspaceConfig,
    columns: &[QuotientNumeratorColumnTopology],
) -> Result<QuotientNumeratorSingleWritePlan, QuotientNumeratorSingleWriteError> {
    let legacy = build_plan(config, columns)?;
    let report = report(&legacy.requirements)?;
    if let QuotientNumeratorSingleWriteEligibility::RequiresRetainedEvaluations {
        coefficient_columns,
        coefficient_batches,
    } = report.eligibility
    {
        return Err(
            QuotientNumeratorSingleWriteError::RequiresRetainedEvaluations {
                coefficient_columns,
                coefficient_batches,
            },
        );
    }

    let mut source_columns = Vec::with_capacity(report.source_count);
    let mut batch_source_bases = Vec::with_capacity(legacy.batches.len());
    for batch in &legacy.batches {
        batch_source_bases.push(source_columns.len());
        source_columns.extend(batch.columns.iter().copied());
    }

    let mut group_offsets = Vec::with_capacity(report.group_count + 1);
    let mut term_descriptors = Vec::with_capacity(
        report
            .term_count
            .checked_mul(QUOTIENT_NUMERATOR_SINGLE_WRITE_TERM_WORDS)
            .ok_or(QuotientNumeratorSingleWriteError::DescriptorInvariant(
                "term word count overflowed",
            ))?,
    );
    for group in 0..report.group_count {
        group_offsets.push(descriptor_count(&term_descriptors)?);
        for (batch, &source_base) in legacy.batches.iter().zip(&batch_source_bases) {
            let begin = batch.group_offsets[group] as usize;
            let end = batch.group_offsets[group + 1] as usize;
            for descriptor in batch.terms[begin * 3..end * 3].chunks_exact(3) {
                let local_source = descriptor[0] as usize;
                if local_source >= batch.columns.len() {
                    return Err(QuotientNumeratorSingleWriteError::DescriptorInvariant(
                        "batch-local source is out of bounds",
                    ));
                }
                let source = source_base.checked_add(local_source).ok_or(
                    QuotientNumeratorSingleWriteError::DescriptorInvariant(
                        "global source index overflowed",
                    ),
                )?;
                term_descriptors.extend([
                    u32::try_from(source).map_err(|_| {
                        QuotientNumeratorSingleWriteError::DescriptorInvariant(
                            "global source index exceeds u32",
                        )
                    })?,
                    descriptor[1],
                    descriptor[2],
                ]);
            }
        }
    }
    group_offsets.push(descriptor_count(&term_descriptors)?);
    if term_descriptors.len() / 3 != report.term_count {
        return Err(QuotientNumeratorSingleWriteError::DescriptorInvariant(
            "flattening lost or duplicated terms",
        ));
    }

    Ok(QuotientNumeratorSingleWritePlan {
        requirements: legacy.requirements,
        source_columns,
        group_offsets,
        term_descriptors,
        report,
    })
}

fn descriptor_count(words: &[u32]) -> Result<u32, QuotientNumeratorSingleWriteError> {
    u32::try_from(words.len() / QUOTIENT_NUMERATOR_SINGLE_WRITE_TERM_WORDS).map_err(|_| {
        QuotientNumeratorSingleWriteError::DescriptorInvariant("term count exceeds u32")
    })
}

fn report(
    requirements: &QuotientNumeratorWorkspaceRequirements,
) -> Result<QuotientNumeratorSingleWriteReport, QuotientNumeratorSingleWriteError> {
    let coefficient_columns = requirements
        .batches
        .iter()
        .map(|batch| batch.coefficient_count)
        .sum::<usize>();
    let coefficient_batches = requirements
        .batches
        .iter()
        .filter(|batch| batch.coefficient_count != 0)
        .count();
    let eligibility = if coefficient_columns == 0 {
        QuotientNumeratorSingleWriteEligibility::Eligible
    } else {
        QuotientNumeratorSingleWriteEligibility::RequiresRetainedEvaluations {
            coefficient_columns,
            coefficient_batches,
        }
    };
    let batches = requirements.batches.len();
    let groups = requirements.groups.len();
    let sources = requirements
        .batches
        .iter()
        .map(|batch| batch.source_count)
        .sum::<usize>();
    let rows = requirements
        .groups
        .iter()
        .try_fold(0usize, |sum, group| sum.checked_add(group.value_words))
        .ok_or(QuotientNumeratorSingleWriteError::DescriptorInvariant(
            "output row count overflowed",
        ))?;
    let rows = u64::try_from(rows).map_err(|_| {
        QuotientNumeratorSingleWriteError::DescriptorInvariant("output rows exceed u64")
    })?;
    let batches_u64 = u64::try_from(batches).map_err(|_| {
        QuotientNumeratorSingleWriteError::DescriptorInvariant("batch count exceeds u64")
    })?;
    let legacy_output_bytes = rows.checked_mul(16 + 32 * batches_u64).ok_or(
        QuotientNumeratorSingleWriteError::DescriptorInvariant(
            "legacy output byte count overflowed",
        ),
    )?;
    let candidate_output_bytes =
        rows.checked_mul(16)
            .ok_or(QuotientNumeratorSingleWriteError::DescriptorInvariant(
                "candidate output byte count overflowed",
            ))?;
    let term_bytes = u64::try_from(requirements.term_count)
        .ok()
        .and_then(|terms| terms.checked_mul(12))
        .ok_or(QuotientNumeratorSingleWriteError::DescriptorInvariant(
            "term descriptor byte count overflowed",
        ))?;
    let pointer_bytes = u64::try_from(sources)
        .ok()
        .and_then(|count| count.checked_mul(core::mem::size_of::<usize>() as u64))
        .ok_or(QuotientNumeratorSingleWriteError::DescriptorInvariant(
            "source pointer byte count overflowed",
        ))?;
    let legacy_offset_bytes = u64::try_from(batches)
        .ok()
        .and_then(|count| count.checked_mul((groups as u64 + 1) * 4))
        .ok_or(QuotientNumeratorSingleWriteError::DescriptorInvariant(
            "legacy offset byte count overflowed",
        ))?;
    let candidate_offset_bytes = (groups as u64 + 1) * 4;

    Ok(QuotientNumeratorSingleWriteReport {
        eligibility,
        group_count: groups,
        term_count: requirements.term_count,
        source_count: sources,
        legacy_batch_count: batches,
        legacy_output_passes: batches + 1,
        candidate_output_passes: 1,
        output_rows: rows as usize,
        legacy_logical_output_bytes: legacy_output_bytes,
        candidate_logical_output_bytes: candidate_output_bytes,
        legacy_descriptor_bytes: term_bytes + pointer_bytes + legacy_offset_bytes,
        candidate_descriptor_bytes: term_bytes + pointer_bytes + candidate_offset_bytes,
        candidate_warm_host_preparation_bytes: 0,
    })
}

#[cfg(test)]
#[path = "quotient_numerator_single_write_tests.rs"]
mod tests;
