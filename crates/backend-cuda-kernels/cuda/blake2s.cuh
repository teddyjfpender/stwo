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

// Workstream D layer-pair fusion: hash two internal (column-free) tree levels per
// launch. `size` = number of grandparent hashes; `previous_layer` holds 4*size.
extern "C"
void commit_on_two_layers_with_previous(uint32_t size, Blake2sHash* previous_layer, Blake2sHash* result);

#endif // BLAKE2S_H
