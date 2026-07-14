#include "quotient_numerator_single_write.cuh"

#include <cuda_runtime.h>

namespace {

constexpr uint32_t TERM_WORDS = 3;
constexpr uint32_t BLOCK_THREADS = 256;

} // namespace

// One thread owns one (group, row) numerator. The descriptor builder preserves
// the legacy batch order inside every group, so this changes memory ownership,
// not field semantics: no zero pass and no global read/modify/write cascade.
extern "C" __global__ void stwo_quotient_numerator_single_write_kernel(
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
        uint32_t *const *outputs_3) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t group = blockIdx.y;
    if (group >= group_count || row >= max_output_size) {
        return;
    }
    const uint32_t group_log_size = group_log_sizes[group];
    if (row >= (1u << group_log_size)) {
        return;
    }

    qm31 numerator = qm31{cm31{0, 0}, cm31{0, 0}};
    for (uint32_t index = group_offsets[group];
         index < group_offsets[group + 1]; ++index) {
        const uint32_t *descriptor =
            term_descriptors + static_cast<size_t>(index) * TERM_WORDS;
        const uint32_t source = descriptor[0];
        const uint32_t term = descriptor[1];
        const uint32_t source_log_size = descriptor[2];
        const uint32_t log_ratio = group_log_size - source_log_size;
        const uint32_t source_row =
            (row >> (log_ratio + 1) << 1) + (row & 1);
        const qm31 b = line_coefficients[static_cast<size_t>(term) * 3 + 1];
        const qm31 c = line_coefficients[static_cast<size_t>(term) * 3 + 2];
        numerator = add(
            numerator,
            sub(mul_by_scalar(c, source_evaluations[source][source_row]), b));
    }

    outputs_0[group][row] = numerator.a.a;
    outputs_1[group][row] = numerator.a.b;
    outputs_2[group][row] = numerator.b.a;
    outputs_3[group][row] = numerator.b.b;
}

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
        void *stream) {
    if (group_offsets == nullptr || term_descriptors == nullptr ||
        group_count == 0 || group_count > 65535 || max_output_size == 0 ||
        source_evaluations == nullptr || line_coefficients == nullptr ||
        group_log_sizes == nullptr || outputs_0 == nullptr ||
        outputs_1 == nullptr || outputs_2 == nullptr ||
        outputs_3 == nullptr || stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    const uint32_t blocks =
        (max_output_size + BLOCK_THREADS - 1) / BLOCK_THREADS;
    stwo_quotient_numerator_single_write_kernel<<<
        dim3(blocks, group_count), BLOCK_THREADS, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            group_offsets, term_descriptors, group_count, max_output_size,
            source_evaluations, line_coefficients, group_log_sizes, outputs_0,
            outputs_1, outputs_2, outputs_3);
    return cudaGetLastError();
}
