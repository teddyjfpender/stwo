// Allocation-free generated CommonLookupElements execution. Every pointer is
// caller-owned arena storage and every launch uses the explicit proof stream.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "batch_inverse.cuh"
#include "fields.cuh"
#include "prefix_sum.cuh"

namespace {

constexpr uint32_t BLOCK = 256;
constexpr uint32_t DESC_WORDS = 16;
constexpr uint32_t USE_WORDS = 7;
constexpr uint32_t LARGE_MEMORY_VALUE_ID_BASE = 0x40000000u;
constexpr uint32_t XOR12_LIMB_BITS = 10;
constexpr uint32_t XOR12_EXPAND_BITS = 2;

// Expand the channel's exact LookupElements draw order `[z, alpha]` into the
// persistent relation slots consumed by every generated relation instance.
// One thread is intentional: at most 128 QM31 powers are produced and this
// launch sits on a serial Fiat-Shamir boundary.
__global__ void relation_expand_challenges_kernel(const qm31 *drawn,
                                                   qm31 *alpha_powers,
                                                   uint32_t n_alpha_powers,
                                                   qm31 *z) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  z[0] = drawn[0];
  qm31 alpha = drawn[1];
  qm31 power = {{1, 0}, {0, 0}};
  for (uint32_t i = 0; i < n_alpha_powers; ++i) {
    alpha_powers[i] = power;
    power = mul(power, alpha);
  }
}

__device__ __forceinline__ m31 tuple_word(const uint32_t *const *sources,
                                          uint32_t n_rows, uint32_t row,
                                          uint32_t source_offset_rows,
                                          const uint32_t *use, uint32_t word) {
  uint32_t kind = use[0];
  uint32_t arg = use[1];
  if (word == 0) {
    return use[3];
  }
  if (kind == 0u) { // LookupWords.
    return sources[0][(arg + word) * n_rows + row];
  }
  if (kind == 1u) { // MemoryAddressChunk.
    if (word == 1) {
      return row + 1u + arg * n_rows;
    }
    return sources[arg * 2u][row];
  }
  if (kind == 2u || kind == 4u) { // Memory big/small limb pair.
    return sources[arg + word - 1u][row];
  }
  if (kind == 3u) { // Full big memory value.
    if (word == 1) {
      return (row + source_offset_rows) | LARGE_MEMORY_VALUE_ID_BASE;
    }
    return sources[word - 2u][row];
  }
  if (kind == 5u) { // Full small memory value.
    if (word == 1) {
      return row + source_offset_rows;
    }
    return sources[word - 2u][row];
  }
  // BitwiseXor12: multiplicity-column index encodes high parts of a,b;
  // row encodes their two low 10-bit limbs.
  uint32_t ah = arg >> XOR12_EXPAND_BITS;
  uint32_t bh = arg & ((1u << XOR12_EXPAND_BITS) - 1u);
  uint32_t a = (ah << XOR12_LIMB_BITS) | (row >> XOR12_LIMB_BITS);
  uint32_t b = (bh << XOR12_LIMB_BITS) | (row & ((1u << XOR12_LIMB_BITS) - 1u));
  return word == 1 ? a : (word == 2 ? b : (a ^ b));
}

__device__ __forceinline__ qm31 combine_use(const uint32_t *const *sources,
                                            uint32_t n_rows, uint32_t row,
                                            uint32_t source_offset_rows,
                                            const uint32_t *use,
                                            const qm31 *alphas, qm31 z) {
  qm31 zero = {{0, 0}, {0, 0}};
  qm31 acc = sub(zero, z);
  for (uint32_t word = 0; word < use[2]; ++word) {
    acc =
        add(acc,
            mul(tuple_word(sources, n_rows, row, source_offset_rows, use, word),
                alphas[word]));
  }
  return acc;
}

__device__ __forceinline__ m31 multiplicity(const uint32_t *const *sources,
                                            uint32_t n_rows, uint32_t row,
                                            uint32_t n_real,
                                            const uint32_t *use) {
  uint32_t kind = use[4];
  uint32_t arg = use[5];
  m31 value;
  if (kind == 0u) {
    value = 1u;
  } else if (kind == 1u) {
    value = row < n_real ? 1u : 0u;
  } else if (kind == 2u) {
    value = sources[0][arg * n_rows + row];
  } else if (kind == 3u) {
    value = sources[arg * 2u + 1u][row];
  } else {
    // Big/small multiplicity and XOR12 all carry their source-column index.
    value = sources[arg][row];
  }
  return use[6] != 0u && value != 0u ? P - value : value;
}

__global__ void relation_pairs_kernel(const uint32_t *const *sources,
                                      uint32_t n_rows, uint32_t n_real,
                                      uint32_t source_offset_rows,
                                      const uint32_t *descriptors,
                                      const qm31 *alphas, const qm31 *z_ptr,
                                      m31 *const *outputs, qm31 *denominators) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t column = blockIdx.y;
  if (row >= n_rows) {
    return;
  }
  const uint32_t *descriptor = descriptors + column * DESC_WORDS;
  const uint32_t *use_a = descriptor + 1;
  qm31 z = z_ptr[0];
  qm31 denominator_a =
      combine_use(sources, n_rows, row, source_offset_rows, use_a, alphas, z);
  m31 mult_a = multiplicity(sources, n_rows, row, n_real, use_a);
  qm31 numerator;
  qm31 denominator;
  if (descriptor[0] == 2u) {
    const uint32_t *use_b = descriptor + 1 + USE_WORDS;
    qm31 denominator_b =
        combine_use(sources, n_rows, row, source_offset_rows, use_b, alphas, z);
    m31 mult_b = multiplicity(sources, n_rows, row, n_real, use_b);
    numerator = add(mul(mult_b, denominator_a), mul(mult_a, denominator_b));
    denominator = mul(denominator_a, denominator_b);
  } else {
    numerator = qm31{{mult_a, 0}, {0, 0}};
    denominator = denominator_a;
  }
  uint32_t output_base = column * 4u;
  outputs[output_base][row] = numerator.a.a;
  outputs[output_base + 1u][row] = numerator.a.b;
  outputs[output_base + 2u][row] = numerator.b.a;
  outputs[output_base + 3u][row] = numerator.b.b;
  denominators[column * n_rows + row] = denominator;
}

__global__ void fraction_chain_kernel(m31 *const *outputs, const qm31 *inverse,
                                      uint32_t n_rows, uint32_t column) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }
  uint32_t base = column * 4u;
  qm31 numerator = {{outputs[base][row], outputs[base + 1u][row]},
                    {outputs[base + 2u][row], outputs[base + 3u][row]}};
  qm31 value = mul(numerator, inverse[row]);
  if (column != 0u) {
    uint32_t previous = base - 4u;
    value =
        add(value,
            qm31{{outputs[previous][row], outputs[previous + 1u][row]},
                 {outputs[previous + 2u][row], outputs[previous + 3u][row]}});
  }
  outputs[base][row] = value.a.a;
  outputs[base + 1u][row] = value.a.b;
  outputs[base + 2u][row] = value.b.a;
  outputs[base + 3u][row] = value.b.b;
}

__global__ void reduce_coordinates_kernel(const m31 *c0, const m31 *c1,
                                          const m31 *c2, const m31 *c3,
                                          uint32_t n_rows, qm31 *partials) {
  extern __shared__ qm31 shared[];
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = row < n_rows
                            ? qm31{{c0[row], c1[row]}, {c2[row], c3[row]}}
                            : qm31{{0, 0}, {0, 0}};
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          add(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void reduce_qm31_kernel(const qm31 *input, uint32_t size,
                                   qm31 *partials) {
  extern __shared__ qm31 shared[];
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = index < size ? input[index] : qm31{{0, 0}, {0, 0}};
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          add(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void shift_last_column_kernel(m31 *c0, m31 *c1, m31 *c2, m31 *c3,
                                         uint32_t n_rows, const qm31 *sum,
                                         qm31 *claimed_sum, m31 inverse_rows) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row == 0u) {
    claimed_sum[0] = sum[0];
  }
  if (row >= n_rows) {
    return;
  }
  qm31 shift = mul(inverse_rows, sum[0]);
  c0[row] = sub(c0[row], shift.a.a);
  c1[row] = sub(c1[row], shift.a.b);
  c2[row] = sub(c2[row], shift.b.a);
  c3[row] = sub(c3[row], shift.b.b);
}

} // namespace

extern "C" int stwo_relation_expand_challenges_on(
    const uint32_t *drawn_z_alpha, uint32_t *alpha_powers,
    uint32_t n_alpha_powers, uint32_t *z, void *stream_raw) {
  if (drawn_z_alpha == nullptr || alpha_powers == nullptr || z == nullptr ||
      n_alpha_powers == 0u) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  relation_expand_challenges_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const qm31 *>(drawn_z_alpha),
      reinterpret_cast<qm31 *>(alpha_powers), n_alpha_powers,
      reinterpret_cast<qm31 *>(z));
  return (int)cudaGetLastError();
}

extern "C" int stwo_relation_pairs_on(
    const uint32_t *const *sources, uint32_t n_sources, uint32_t n_rows,
    uint32_t n_real, uint32_t source_offset_rows, const uint32_t *descriptors,
    uint32_t n_columns, const uint32_t *alpha_powers, uint32_t n_alpha_powers,
    const uint32_t *z, uint32_t *const *outputs, uint32_t *denominators,
    void *stream_raw) {
  if (n_sources == 0u || n_rows == 0u || n_real > n_rows || n_columns == 0u ||
      n_alpha_powers == 0u) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  dim3 grid((n_rows + BLOCK - 1u) / BLOCK, n_columns);
  relation_pairs_kernel<<<grid, BLOCK, 0, stream>>>(
      sources, n_rows, n_real, source_offset_rows, descriptors,
      reinterpret_cast<const qm31 *>(alpha_powers),
      reinterpret_cast<const qm31 *>(z),
      reinterpret_cast<m31 *const *>(outputs),
      reinterpret_cast<qm31 *>(denominators));
  return (int)cudaGetLastError();
}

extern "C" int stwo_relation_fraction_chain_on(uint32_t *const *outputs,
                                               const uint32_t *denominators,
                                               uint32_t *inverse_scratch,
                                               uint32_t n_rows,
                                               uint32_t n_columns,
                                               void *stream_raw) {
  if (n_rows == 0u || n_columns == 0u) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  qm31 *inverse = reinterpret_cast<qm31 *>(inverse_scratch);
  const qm31 *denominator = reinterpret_cast<const qm31 *>(denominators);
  uint32_t blocks = (n_rows + BLOCK - 1u) / BLOCK;
  for (uint32_t column = 0; column < n_columns; ++column) {
    cudaError_t error = batch_inverse_secure_field_on(
        stream, const_cast<qm31 *>(denominator + column * n_rows), inverse,
        (int)n_rows);
    if (error != cudaSuccess) {
      return (int)error;
    }
    fraction_chain_kernel<<<blocks, BLOCK, 0, stream>>>(
        reinterpret_cast<m31 *const *>(outputs), inverse, n_rows, column);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return (int)error;
    }
  }
  return (int)cudaSuccess;
}

extern "C" int stwo_relation_reduce_shift_on(
    uint32_t *output_0, uint32_t *output_1, uint32_t *output_2,
    uint32_t *output_3, uint32_t n_rows, uint32_t *reduction_a,
    uint32_t *reduction_b, uint32_t reduction_capacity, uint32_t *claimed_sum,
    uint32_t inverse_rows, void *stream_raw) {
  if (n_rows == 0u || reduction_capacity == 0u) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  qm31 *a = reinterpret_cast<qm31 *>(reduction_a);
  qm31 *b = reinterpret_cast<qm31 *>(reduction_b);
  uint32_t size = (n_rows + BLOCK - 1u) / BLOCK;
  if (size > reduction_capacity) {
    return (int)cudaErrorInvalidValue;
  }
  reduce_coordinates_kernel<<<size, BLOCK, BLOCK * sizeof(qm31), stream>>>(
      output_0, output_1, output_2, output_3, n_rows, a);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return (int)error;
  }
  qm31 *current = a;
  qm31 *next = b;
  while (size > 1u) {
    uint32_t next_size = (size + BLOCK - 1u) / BLOCK;
    reduce_qm31_kernel<<<next_size, BLOCK, BLOCK * sizeof(qm31), stream>>>(
        current, size, next);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return (int)error;
    }
    qm31 *swap = current;
    current = next;
    next = swap;
    size = next_size;
  }
  uint32_t blocks = (n_rows + BLOCK - 1u) / BLOCK;
  shift_last_column_kernel<<<blocks, BLOCK, 0, stream>>>(
      output_0, output_1, output_2, output_3, n_rows, current,
      reinterpret_cast<qm31 *>(claimed_sum), inverse_rows);
  return (int)cudaGetLastError();
}

extern "C" int stwo_relation_prefix_scan_on(uint32_t *output, uint32_t n_rows,
                                            uint32_t *eval_scratch,
                                            void *scan_temp,
                                            size_t scan_temp_bytes,
                                            void *stream_raw) {
  if (n_rows == 0u) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  return (int)inclusive_prefix_sum_prepared_on(
      stream, output, eval_scratch, scan_temp, scan_temp_bytes, n_rows);
}
