#ifndef QUOTIENT_NUMERATOR_SINGLE_WRITE_H
#define QUOTIENT_NUMERATOR_SINGLE_WRITE_H

#include "fields.cuh"

// Candidate-only quotient numerator entry. Descriptors are grouped by output
// group and contain [global_source, canonical_term, source_log_size]. Every
// sampled source must be a retained evaluation for the duration of the launch.
extern "C" int stwo_accumulate_quotient_numerator_single_write_on(
        const uint32_t *group_offsets,
        const uint32_t *term_descriptors,
        uint32_t group_count,
        uint32_t max_output_size,
        const uint32_t *const *source_evaluations,
        const qm31 *line_coefficients,
        const uint32_t *group_log_sizes,
        uint32_t *const *outputs_0,
        uint32_t *const *outputs_1,
        uint32_t *const *outputs_2,
        uint32_t *const *outputs_3,
        void *stream);

// Replacement-v1 coefficient-inclusive entry. `group_row_offsets` is a sealed
// u64 prefix sum with `group_count + 1` entries; its terminal entry must equal
// `packed_output_rows`. Exactly one 1-D thread is launched per useful row.
extern "C" int stwo_accumulate_quotient_numerator_packed_single_write_on(
        const uint64_t *group_row_offsets,
        const uint32_t *group_term_offsets,
        const uint32_t *term_descriptors,
        uint32_t group_count,
        uint64_t packed_output_rows,
        const uint32_t *const *source_evaluations,
        const qm31 *line_coefficients,
        const uint32_t *group_log_sizes,
        uint32_t *const *outputs_0,
        uint32_t *const *outputs_1,
        uint32_t *const *outputs_2,
        uint32_t *const *outputs_3,
        void *stream);

#endif // QUOTIENT_NUMERATOR_SINGLE_WRITE_H
