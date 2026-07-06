#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 256;

// Occupancy lever for the register-capped commit/leaf-hash kernels. The
// STWO_COMMIT_PROBE measured them pinned at 255 registers => 1 block/SM => 12.5%
// occupancy, already spilling to local memory (blake2s full-unroll register
// explosion). The minBlocksPerMultiprocessor argument to __launch_bounds__ forces
// the compiler to cap registers so >=N blocks co-reside on an SM, trading register
// spilling for latency-hiding occupancy. Compile-time tunable to sweep the
// trade-off; 1 reproduces the prior (uncapped, 255-reg) behavior. Byte-identical
// either way — this is purely a scheduling/occupancy hint.
#ifndef STWO_LEAF_MIN_BLOCKS
#define STWO_LEAF_MIN_BLOCKS 2
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

// Workstream D layer-pair fusion: hash two internal (column-free) tree levels per
// launch. `size` = number of grandparent hashes; `previous_layer` holds 4*size.
extern "C"
void commit_on_two_layers_with_previous(uint32_t size, Blake2sHash* previous_layer, Blake2sHash* result);

#endif // BLAKE2S_H
