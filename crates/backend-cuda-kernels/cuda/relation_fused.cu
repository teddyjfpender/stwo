// Fused relation pipeline: one kernel, one proof-wide launch, replacing the
// 3-stage `relation_pairs_global` -> `batch_inverse_secure_field_ragged` ->
// `fraction_chain_global` sequence for every fused-eligible instance. The
// denominator slab is never materialized — every denominator is recomputed
// from the arena source columns via the shared helpers in relation_fused.cuh,
// so the only global traffic is: source reads (twice), one staging write plus
// one read/write of the committed output coordinates. The instance's
// `denominators` binding is a one-word aligned sentinel in a mode-sealed fused
// preparation, and the proof-wide `inverse_scratch` slot stays allocated but
// untouched. Neither is passed to or dereferenced by this kernel.
//
// Per thread (= one row of one instance), with C = columns:
//   1. Backward pass c = C-1..0: recompute (num_c, d_c); stage
//      `num_c * suffix_c` (suffix_c = prod_{j>c} d_j) into output column c;
//      accumulate `suffix = prod d_j`.
//   2. ONE inversion: `running = inv(prod d_j)`.
//   3. Forward pass c = 0..C-1: recompute d_c (denominator only); now
//      `running = 1 / prod_{j>=c} d_j`, so
//      `staged_c * running = num_c * prod_{j>c} d_j / prod_{j>=c} d_j
//                          = num_c / d_c`;
//      accumulate and overwrite output column c with the partial-fraction
//      chain value, then `running *= d_c`.
//
// Byte identity with the 3-stage lane: M31/QM31 arithmetic is exact modular
// arithmetic through the same canonical-form primitives (fields.cu), so the
// committed value at column c — the exact field element
// sum_{j<=c} num_j * d_j^{-1} — has one canonical word encoding regardless of
// how the inverses are materialized (per-element `inv`, 1024-leaf Montgomery
// tree, or this single-inversion peel). Zero denominators are outside the
// contract in both lanes (the SIMD reference asserts them away). Parity is
// gated by prepared_relation_native.rs (eager/captured/mutated-replay).
//
// Dispatch reuses the immutable 11-word geometry records and the
// RELATION_ROW_FIRST/RELATION_ROW_BLOCKS row-tile layout that
// fraction_chain_global_kernel already binary-searches; instances whose bit
// is clear in the by-value eligibility mask are skipped here and executed by
// the host on the existing per-instance 3-stage path.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "relation_fused.cuh"

namespace {

__device__ __forceinline__ void store_coordinates(m31 *const *outputs,
                                                  uint32_t column, uint32_t row,
                                                  qm31 value) {
  uint32_t base = column * 4u;
  outputs[base][row] = value.a.a;
  outputs[base + 1u][row] = value.a.b;
  outputs[base + 2u][row] = value.b.a;
  outputs[base + 3u][row] = value.b.b;
}

__global__ void relation_fused_kernel(
    const uint32_t *const *const *source_tables,
    const uint32_t *const *descriptors, m31 *const *const *output_tables,
    const uint32_t *geometry, uint32_t n_instances, const qm31 *alphas,
    const qm31 *z_ptr, relation_fused_mask mask) {
  uint32_t local_block = 0u;
  uint32_t instance = relation_instance_for_block(
      geometry, n_instances, blockIdx.x, RELATION_ROW_FIRST,
      RELATION_ROW_BLOCKS, &local_block);
  if (instance == n_instances || !relation_fused_mask_test(mask, instance)) {
    return;
  }
  const uint32_t *g = geometry + instance * RELATION_GEOMETRY_WORDS;
  uint32_t rows = g[RELATION_ROWS];
  uint32_t row = local_block * RELATION_LAUNCH_BLOCK + threadIdx.x;
  if (row >= rows) {
    return;
  }
  uint32_t columns = g[RELATION_COLUMNS];
  uint32_t n_real = g[RELATION_REAL_ROWS];
  uint32_t source_offset_rows = g[RELATION_SOURCE_OFFSET];
  const uint32_t *const *sources = source_tables[instance];
  const uint32_t *instance_descriptors = descriptors[instance];
  m31 *const *outputs = output_tables[instance];
  qm31 z = z_ptr[0];

  // Backward pass: stage numerator * denominator-suffix into the committed
  // output columns (they are rewritten below, exactly like the 3-stage lane
  // rewrites its pairs output) and fold the whole-chain product in registers.
  qm31 suffix = {{1, 0}, {0, 0}};
  for (uint32_t column = columns; column-- > 0u;) {
    qm31 numerator;
    qm31 denominator;
    relation_column_fraction(sources, rows, row, n_real, source_offset_rows,
                             instance_descriptors +
                                 column * RELATION_DESC_WORDS,
                             alphas, z, &numerator, &denominator);
    store_coordinates(outputs, column, row, mul(numerator, suffix));
    suffix = mul(suffix, denominator);
  }

  // The only inversion this thread performs.
  qm31 running = inv(suffix);

  // Forward pass: recompute denominators, peel per-column inverses out of the
  // running prefix, and write the committed partial-fraction chain in place.
  qm31 accumulated = {{0, 0}, {0, 0}};
  for (uint32_t column = 0; column < columns; ++column) {
    qm31 denominator = relation_column_denominator(
        sources, rows, row, source_offset_rows,
        instance_descriptors + column * RELATION_DESC_WORDS, alphas, z);
    uint32_t base = column * 4u;
    qm31 staged = {{outputs[base][row], outputs[base + 1u][row]},
                   {outputs[base + 2u][row], outputs[base + 3u][row]}};
    accumulated = add(accumulated, mul(staged, running));
    store_coordinates(outputs, column, row, accumulated);
    running = mul(running, denominator);
  }
}

} // namespace

extern "C" int stwo_relation_fused_on(
    const uint32_t *const *const *source_tables,
    const uint32_t *const *descriptors, uint32_t *const *const *output_tables,
    const uint32_t *geometry, uint32_t n_instances, uint32_t total_row_blocks,
    const uint32_t *alpha_powers, uint32_t n_alpha_powers, const uint32_t *z,
    const uint32_t *eligible_mask_words, void *stream_raw) {
  if (source_tables == nullptr || descriptors == nullptr ||
      output_tables == nullptr || geometry == nullptr || n_instances == 0u ||
      n_instances > RELATION_FUSED_MAX_INSTANCES || total_row_blocks == 0u ||
      total_row_blocks > 0x7fffffffu || alpha_powers == nullptr ||
      n_alpha_powers == 0u || z == nullptr || eligible_mask_words == nullptr ||
      stream_raw == nullptr) {
    return (int)cudaErrorInvalidValue;
  }
  // Host-side mask words are copied into the by-value kernel parameter here,
  // before launch/capture; nothing device-resident carries eligibility.
  relation_fused_mask mask;
  for (uint32_t word = 0; word < RELATION_FUSED_MASK_WORDS; ++word) {
    mask.bits[word] = eligible_mask_words[word];
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  relation_fused_kernel<<<total_row_blocks, RELATION_LAUNCH_BLOCK, 0, stream>>>(
      source_tables, descriptors,
      reinterpret_cast<m31 *const *const *>(output_tables), geometry,
      n_instances, reinterpret_cast<const qm31 *>(alpha_powers),
      reinterpret_cast<const qm31 *>(z), mask);
  return (int)cudaGetLastError();
}
