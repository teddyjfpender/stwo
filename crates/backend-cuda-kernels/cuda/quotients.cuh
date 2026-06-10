#ifndef QUOTIENTS_H
#define QUOTIENTS_H

#include "fields.cuh"
#include "point.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 1024;

extern "C"
void accumulate_quotients(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        m31 **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        uint32_t sample_column_indexes_size,
        qm31 *sample_column_values,
        uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3,
        uint32_t flattened_line_coeffs_size
);

extern "C"
void accumulate_partial_quotient_numerators(
        uint32_t domain_size,
        m31 **columns,
        uint32_t *sample_column_indexes,
        uint32_t sample_column_indexes_size,
        qm31 *line_coeffs_b,
        qm31 *line_coeffs_c,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3
);

extern "C"
void combine_quotients_from_numerators(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        uint32_t domain_log_size,
        secure_field_point *sample_points,
        uint32_t sample_size,
        qm31 *first_linear_term_accs,
        uint32_t *partial_numerator_log_sizes,
        m31 **partial_numerators_0,
        m31 **partial_numerators_1,
        m31 **partial_numerators_2,
        m31 **partial_numerators_3,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3
);

#endif // QUOTIENTS_H
