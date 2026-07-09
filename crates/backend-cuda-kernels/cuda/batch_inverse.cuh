#ifndef BATCH_INVERSE_H
#define BATCH_INVERSE_H

#include <cuda_runtime.h>
#include "fields.cuh"

extern "C"
void batch_inverse_base_field(m31 *from, m31 *dst, int size);

extern "C"
void batch_inverse_secure_field(qm31 *from, qm31 *dst, int size);

// Allocation/sync/default-stream-free body for prepared graph execution.
cudaError_t batch_inverse_secure_field_on(
    cudaStream_t stream, qm31 *from, qm31 *dst, int size);

#endif // BATCH_INVERSE_H
