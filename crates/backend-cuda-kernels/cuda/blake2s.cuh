#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 256;

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
