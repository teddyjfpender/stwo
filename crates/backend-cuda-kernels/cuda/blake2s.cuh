#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 256;

// Shared device primitive for the transcript engine.  It is defined by
// blake2s.cu so the channel and Merkle paths use one compression
// implementation.  Inputs are byte strings exactly as consumed by the Rust
// `Blake2sHasher`; no domain tag is inserted here.
#ifdef __CUDACC__
__device__ void stwo_blake2s_hash2_device(
    const uint8_t *first,
    size_t first_len,
    const uint8_t *second,
    size_t second_len,
    Blake2sHash *out);
#endif

// Occupancy lever for the register-capped commit/leaf-hash kernels. The
// STWO_COMMIT_PROBE measured them pinned at 255 registers => 1 block/SM => 12.5%
// occupancy, already spilling to local memory (blake2s full-unroll register
// explosion). The minBlocksPerMultiprocessor argument to __launch_bounds__ forces
// the compiler to cap registers so >=N blocks co-reside on an SM, trading register
// spilling for latency-hiding occupancy. Compile-time tunable to sweep the
// trade-off; 1 reproduces the prior (uncapped, 255-reg) behavior. Byte-identical
// either way — this is purely a scheduling/occupancy hint.
//
// MEASURED 2026-07-06: minBlocks=2 doubles occupancy (12.5%→25%, regs 255→128, no
// added spilling) but is FLAT on total prove within run-to-run noise (~0.8s on
// SN_PIE_2): the leaf-hash is blake2s-COMPUTE-bound, so latency-hiding occupancy
// doesn't help, and the commit-Merkle slice is small. Default kept at 1 (no forced
// cap = original behavior); the knob + STWO_COMMIT_PROBE stay as diagnostics.
#ifndef STWO_LEAF_MIN_BLOCKS
#define STWO_LEAF_MIN_BLOCKS 1
#endif

extern "C"
void commit_on_first_layer(uint32_t size, uint32_t number_of_columns, uint32_t **columns, Blake2sHash* result);

extern "C"
void commit_on_first_layer_lifted(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **columns,
    const uint32_t *column_log_sizes,
    uint32_t lifting_log_size,
    Blake2sHash* result
);

extern "C"
void commit_on_layer_with_previous(uint32_t size, uint32_t number_of_columns, uint32_t **columns, Blake2sHash* previous_layer, Blake2sHash* result);

// Streaming leaf commit (VRAM diet): the lifted first layer split so the caller
// LDEs base columns one group at a time. `state` holds h[8] per leaf (init ->
// update per group -> finalize). Non-final groups pass a multiple-of-16
// `group_n_cols`; `cols_done` is the running column offset. Byte-identical to
// commit_on_first_layer_lifted.
extern "C"
void stream_leaf_init(uint32_t size, Blake2sHash *state);
extern "C"
void stream_leaf_update(uint32_t size, uint32_t group_n_cols, uint32_t **columns, const uint32_t *column_log_sizes, uint32_t lifting_log_size, uint32_t cols_done, Blake2sHash *state);
extern "C"
void stream_leaf_finalize(uint32_t size, uint32_t rem_cols, uint32_t **columns, const uint32_t *column_log_sizes, uint32_t lifting_log_size, uint32_t cols_done, Blake2sHash *result);

// Allocation-free explicit-stream variants used by transcript-bounded graph
// segments. All pointer tables, state, and outputs are caller-owned device memory.
extern "C"
int stwo_blake2s_leaf_init_on(uint32_t size, Blake2sHash *state, void *stream);
extern "C"
int stwo_blake2s_leaf_update_on(uint32_t size, uint32_t group_n_cols, uint32_t **columns, const uint32_t *column_log_sizes, uint32_t lifting_log_size, uint32_t cols_done, Blake2sHash *state, void *stream);
extern "C"
int stwo_blake2s_leaf_finalize_on(uint32_t size, uint32_t rem_cols, uint32_t **columns, const uint32_t *column_log_sizes, uint32_t lifting_log_size, uint32_t cols_done, Blake2sHash *result, void *stream);
extern "C"
int stwo_blake2s_layer_on(const Blake2sHash *previous_layer, uint32_t output_size, Blake2sHash *result, void *stream);
extern "C"
int stwo_blake2s_tail_on(const Blake2sHash *first, uint32_t first_size, Blake2sHash *const *out_levels, uint32_t n_levels, void *stream);

// FRI leaf hashing directly from four coordinate columns. `log_rows_per_leaf`
// is exactly 0 (four-word leaf) or 2 (the verifier's 4-row packed leaf). This
// avoids materializing the 16 packed columns while preserving their byte order.
extern "C"
int stwo_blake2s_fri_leaf_on(uint32_t evaluation_size,
                             uint32_t **coordinate_columns,
                             uint32_t log_rows_per_leaf,
                             Blake2sHash *result,
                             void *stream);

// Workstream D layer-pair fusion: hash two internal (column-free) tree levels per
// launch. `size` = number of grandparent hashes; `previous_layer` holds 4*size.
extern "C"
void commit_on_two_layers_with_previous(uint32_t size, Blake2sHash* previous_layer, Blake2sHash* result);

#endif // BLAKE2S_H
