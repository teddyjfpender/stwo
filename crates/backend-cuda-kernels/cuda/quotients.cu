#include "quotients.cuh"
#include <cstdio>


typedef struct {
    secure_field_point point;
    uint32_t *columns;
    qm31 *values;
    uint32_t size;
    size_t offset;
} column_sample_batch;

HOST_DEVICE_FORCEINLINE point index_to_point(uint32_t index) {
    return point_pow(m31_circle_gen, (int)index);
}

DEVICE_FORCEINLINE point domain_at_index(uint32_t half_coset_initial_index, uint32_t half_coset_step_size, uint32_t index, uint32_t domain_size) {
    uint32_t half_coset_size = domain_size >> 1;

    if (index < half_coset_size) {
        int modulo_u31_mask = 0x7fffffff;
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) index;
        return index_to_point(global_index & modulo_u31_mask);
    } else {
        int modulo_u31_mask = 0x7fffffff;
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) (index - half_coset_size);
        return index_to_point((2147483648 - global_index) & modulo_u31_mask);
    }
}

void column_sample_batches_for(
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        const uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        column_sample_batch *result
) {
    unsigned int offset = 0;
    for (unsigned int index = 0; index < sample_size; index++) {
        result[index].point = sample_points[index];
        result[index].columns = &sample_column_indexes[offset];
        result[index].values = &sample_column_values[offset];
        result[index].size = sample_column_and_values_sizes[index];
        result[index].offset = offset;
        offset += sample_column_and_values_sizes[index];
    }
}

DEVICE_FORCEINLINE void complex_conjugate_line_coeffs(secure_field_point point, qm31 value, qm31 alpha, qm31* a_out, qm31* b_out, qm31* c_out) {
    qm31 a = sub(qm31{value.a, neg(value.b)}, value);
    qm31 c = sub(qm31{point.y.a, neg(point.y.b)}, point.y);
    qm31 b = sub(mul(value, c), mul(a, point.y));

    *a_out = mul(alpha, a);
    *b_out = mul(alpha, b);
    *c_out = mul(alpha, c);
}

__global__ void column_line_and_batch_random_coeffs(
    column_sample_batch *sample_batches,
    uint32_t sample_size,
    qm31 random_coefficient,
    qm31 *flattened_line_coeffs,
    uint32_t *line_coeffs_sizes,
    qm31 *batch_random_coeffs
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < sample_size) {
        // Calculate Batch Random Coeffs
        batch_random_coeffs[tid] = pow(random_coefficient, sample_batches[tid].size);

        // Calculate Column Line Coeffs
        line_coeffs_sizes[tid] = sample_batches[tid].size;
        size_t sample_batches_offset = sample_batches[tid].offset * 3;

        qm31 alpha = qm31{cm31{m31{1}, m31{0}}, cm31{m31{0}, m31{0}}};

        for(size_t j = 0; j < sample_batches[tid].size; ++j) {
            qm31 sampled_value = sample_batches[tid].values[j];
            alpha = mul(alpha, random_coefficient);
            secure_field_point point = sample_batches[tid].point;
            qm31 value = sampled_value;

            size_t sampled_offset = sample_batches_offset + (j * 3);
            complex_conjugate_line_coeffs(point, value, alpha, &flattened_line_coeffs[sampled_offset], &flattened_line_coeffs[sampled_offset + 1], &flattened_line_coeffs[sampled_offset + 2]);
        }
    }
}


DEVICE_FORCEINLINE void denominator_inverse(
        column_sample_batch *sample_batches,
        uint32_t sample_size,
        const point domain_point,
        cm31 *flat_denominators) {

    for (unsigned int i = 0; i < sample_size; i++) {
        cm31 prx = sample_batches[i].point.x.a;
        cm31 pry = sample_batches[i].point.y.a;
        cm31 pix = sample_batches[i].point.x.b;
        cm31 piy = sample_batches[i].point.y.b;

        cm31 first_substraction = {sub(prx.a, domain_point.x), prx.b};
        cm31 second_substraction = {sub(pry.a, domain_point.y), pry.b};
        cm31 result = sub(mul(first_substraction, piy),
                          mul(second_substraction, pix));
        flat_denominators[i] = inv(result);
    }
}

DEVICE_FORCEINLINE void denominator_inverse_from_sample_points(
        secure_field_point *sample_points,
        uint32_t sample_size,
        const point domain_point,
        cm31 *flat_denominators) {
    for (unsigned int i = 0; i < sample_size; i++) {
        cm31 prx = sample_points[i].x.a;
        cm31 pry = sample_points[i].y.a;
        cm31 pix = sample_points[i].x.b;
        cm31 piy = sample_points[i].y.b;

        cm31 first_substraction = {sub(prx.a, domain_point.x), prx.b};
        cm31 second_substraction = {sub(pry.a, domain_point.y), pry.b};
        cm31 result = sub(mul(first_substraction, piy),
                          mul(second_substraction, pix));
        flat_denominators[i] = inv(result);
    }
}

__global__ void accumulate_quotients_in_gpu(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        int domain_log_size,
        m31 **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        column_sample_batch *sample_batches,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3,
        qm31 *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        qm31 *batch_random_coeffs,
        cm31 *denominator_inverses
) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    denominator_inverses = &denominator_inverses[row * sample_size];

    if (row < domain_size) {
        uint32_t domain_index = bit_reverse(row, domain_log_size);
        point domain_point = domain_at_index(half_coset_initial_index, half_coset_step_size, domain_index, domain_size);

        denominator_inverse(
            sample_batches,
            sample_size,
            domain_point,
            denominator_inverses
        );

        int i = 0;

        qm31 row_accumulator = qm31{cm31{0, 0}, cm31{0, 0}};
        int line_coeffs_offset = 0;
        while (i < sample_size) {
            column_sample_batch sample_batch = sample_batches[i];
            qm31 *line_coeffs = &flattened_line_coeffs[line_coeffs_offset * 3];
            qm31 batch_coeff = batch_random_coeffs[i];
            int line_coeffs_size = line_coeffs_sizes[i];

            qm31 numerator = qm31{cm31{0, 0}, cm31{0, 0}};
            for(int j = 0; j < line_coeffs_size; j++) {
                qm31 a = line_coeffs[3 * j + 0];
                qm31 b = line_coeffs[3 * j + 1];
                qm31 c = line_coeffs[3 * j + 2];

                int column_index = sample_batch.columns[j];
                qm31 linear_term = add(mul_by_scalar(a, domain_point.y), b);
                qm31 value = mul_by_scalar(c, columns[column_index][row]);

                numerator = add(numerator, sub(value, linear_term));
            }

            row_accumulator = add(mul(row_accumulator, batch_coeff), mul(numerator, denominator_inverses[i]));
            line_coeffs_offset += line_coeffs_size;
            i++;
        }

        result_column_0[row] = row_accumulator.a.a;
        result_column_1[row] = row_accumulator.a.b;
        result_column_2[row] = row_accumulator.b.a;
        result_column_3[row] = row_accumulator.b.b;

    }
}
__global__ void dump_qm31_array(qm31 *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("(%d + %di) + (%d + %di)u, ", array[i].a.a, array[i].a.b, array[i].b.a, array[i].b.b);
    }
    printf("\n");
}

__global__ void accumulate_partial_quotient_numerators_in_gpu(
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
) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < domain_size) {
        qm31 numerator = qm31{cm31{0, 0}, cm31{0, 0}};
        for (uint32_t j = 0; j < sample_column_indexes_size; ++j) {
            uint32_t column_index = sample_column_indexes[j];
            qm31 value = mul_by_scalar(line_coeffs_c[j], columns[column_index][row]);
            numerator = add(numerator, sub(value, line_coeffs_b[j]));
        }

        result_column_0[row] = numerator.a.a;
        result_column_1[row] = numerator.a.b;
        result_column_2[row] = numerator.b.a;
        result_column_3[row] = numerator.b.b;
    }
}

__global__ void combine_quotients_from_numerators_in_gpu(
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
        uint32_t *result_column_3,
        cm31 *denominator_inverses
) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    denominator_inverses = &denominator_inverses[row * sample_size];

    if (row < domain_size) {
        uint32_t domain_index = bit_reverse(row, domain_log_size);
        point domain_point = domain_at_index(
                half_coset_initial_index,
                half_coset_step_size,
                domain_index,
                domain_size
        );

        denominator_inverse_from_sample_points(
                sample_points,
                sample_size,
                domain_point,
                denominator_inverses
        );

        qm31 quotient = qm31{cm31{0, 0}, cm31{0, 0}};
        for (uint32_t i = 0; i < sample_size; ++i) {
            uint32_t partial_log_size = partial_numerator_log_sizes[i];
            uint32_t log_ratio = domain_log_size - partial_log_size;
            uint32_t lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

            qm31 partial_numerator = qm31{
                    cm31{
                            partial_numerators_0[i][lifted_idx],
                            partial_numerators_1[i][lifted_idx]
                    },
                    cm31{
                            partial_numerators_2[i][lifted_idx],
                            partial_numerators_3[i][lifted_idx]
                    }
            };
            qm31 full_numerator = sub(
                    partial_numerator,
                    mul_by_scalar(first_linear_term_accs[i], domain_point.y)
            );
            quotient = add(quotient, mul(full_numerator, denominator_inverses[i]));
        }

        result_column_0[row] = quotient.a.a;
        result_column_1[row] = quotient.a.b;
        result_column_2[row] = quotient.b.a;
        result_column_3[row] = quotient.b.b;
    }
}

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
) {
    int domain_log_size = log_2((int)domain_size);

    auto sample_batches = (column_sample_batch *)malloc(sizeof(column_sample_batch) * sample_size);
    memset(sample_batches, 0, sizeof(column_sample_batch) * sample_size);

    column_sample_batch *sample_batches_device = cuda_proving_malloc<column_sample_batch>(sample_size);
    cm31* denominator_inverses = cuda_proving_malloc<cm31>(sample_size * domain_size);

    uint32_t *sample_column_indexes_device =
        cuda_proving_clone_to_device<uint32_t>(sample_column_indexes, sample_column_indexes_size);
    qm31 *sample_column_values_device =
        cuda_proving_clone_to_device<qm31>(sample_column_values, sample_column_indexes_size);

    column_sample_batches_for(
            sample_points,
            sample_column_indexes_device,
            sample_column_values_device,
            sample_column_and_values_sizes,
            sample_size,
            sample_batches
    );

    cuda_mem_copy_host_to_device(sample_batches, sample_batches_device, sample_size);
    qm31 *batch_random_coeffs_device = cuda_proving_malloc<qm31>(sample_size);
    uint32_t *line_coeffs_sizes_device = cuda_proving_malloc<uint32_t>(sample_size);
    qm31 *flattened_line_coeffs_device = cuda_proving_malloc<qm31>(flattened_line_coeffs_size);

    // Accumulate Quotient Constants
    int block_dim = sample_size < THREAD_COUNT_MAX ? sample_size : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (sample_size + block_dim - 1) / block_dim;
    column_line_and_batch_random_coeffs<<<num_blocks, block_dim>>>(
            sample_batches_device,
            sample_size,
            random_coefficient,
            flattened_line_coeffs_device,
            line_coeffs_sizes_device,
            batch_random_coeffs_device
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // TODO: set to 1024
    block_dim = 512;
    num_blocks = (domain_size + block_dim - 1) / block_dim;
    accumulate_quotients_in_gpu<<<num_blocks, block_dim>>>(
            half_coset_initial_index,
            half_coset_step_size,
            domain_size,
            domain_log_size,
            columns,
            number_of_columns,
            random_coefficient,
            sample_batches_device,
            sample_size,
            result_column_0,
            result_column_1,
            result_column_2,
            result_column_3,
            flattened_line_coeffs_device,
            line_coeffs_sizes_device,
            batch_random_coeffs_device,
            denominator_inverses
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    free(sample_batches);
    cuda_proving_free(sample_batches_device);
    cuda_proving_free(denominator_inverses);
    cuda_proving_free(sample_column_indexes_device);
    cuda_proving_free(sample_column_values_device);
    cuda_proving_free(batch_random_coeffs_device);
    cuda_proving_free(line_coeffs_sizes_device);
    cuda_proving_free(flattened_line_coeffs_device);
}

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
) {
    if (sample_column_indexes_size == 0) {
        return;
    }

    uint32_t *sample_column_indexes_device =
        cuda_proving_clone_to_device<uint32_t>(sample_column_indexes, sample_column_indexes_size);
    qm31 *line_coeffs_b_device =
        cuda_proving_clone_to_device<qm31>(line_coeffs_b, sample_column_indexes_size);
    qm31 *line_coeffs_c_device =
        cuda_proving_clone_to_device<qm31>(line_coeffs_c, sample_column_indexes_size);

    int block_dim = 512;
    int num_blocks = (domain_size + block_dim - 1) / block_dim;
    accumulate_partial_quotient_numerators_in_gpu<<<num_blocks, block_dim>>>(
            domain_size,
            columns,
            sample_column_indexes_device,
            sample_column_indexes_size,
            line_coeffs_b_device,
            line_coeffs_c_device,
            result_column_0,
            result_column_1,
            result_column_2,
            result_column_3
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_proving_free(sample_column_indexes_device);
    cuda_proving_free(line_coeffs_b_device);
    cuda_proving_free(line_coeffs_c_device);
}

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
) {
    if (sample_size == 0) {
        return;
    }

    secure_field_point *sample_points_device =
        cuda_proving_clone_to_device<secure_field_point>(sample_points, sample_size);
    qm31 *first_linear_term_accs_device =
        cuda_proving_clone_to_device<qm31>(first_linear_term_accs, sample_size);
    uint32_t *partial_numerator_log_sizes_device =
        cuda_proving_clone_to_device<uint32_t>(partial_numerator_log_sizes, sample_size);
    cm31 *denominator_inverses = cuda_proving_malloc<cm31>(sample_size * domain_size);

    int block_dim = 512;
    int num_blocks = (domain_size + block_dim - 1) / block_dim;
    combine_quotients_from_numerators_in_gpu<<<num_blocks, block_dim>>>(
            half_coset_initial_index,
            half_coset_step_size,
            domain_size,
            domain_log_size,
            sample_points_device,
            sample_size,
            first_linear_term_accs_device,
            partial_numerator_log_sizes_device,
            partial_numerators_0,
            partial_numerators_1,
            partial_numerators_2,
            partial_numerators_3,
            result_column_0,
            result_column_1,
            result_column_2,
            result_column_3,
            denominator_inverses
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_proving_free(sample_points_device);
    cuda_proving_free(first_linear_term_accs_device);
    cuda_proving_free(partial_numerator_log_sizes_device);
    cuda_proving_free(denominator_inverses);
}
