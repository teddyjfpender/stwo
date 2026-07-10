// Capture-safe native witness writer for Cairo's ec_op_builtin.
//
// One thread owns one EC-op instance and emits both the 273 committed base
// columns and the exact round-major input columns consumed by
// partial_ec_mul_generic.  The latter are written directly into their final
// 127-column arena layout; there is no rows*252*state host/sub-input staging
// buffer.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "ec_ops.cuh"

namespace {

constexpr uint32_t EC_OP_BLOCK = 64;
constexpr uint32_t EC_OP_TRACE_COLUMNS = 273;
constexpr uint32_t PARTIAL_INPUT_COLUMNS = 127;
constexpr uint32_t PARTIAL_ROUNDS = 252;
constexpr uint32_t PARTIAL_PADDED_ROUNDS = 256;
constexpr uint32_t FELT_LIMBS = 28;
constexpr uint32_t W27_WORDS = 10;

constexpr uint32_t MEMORY_ADDRESS_RELATION = 1444891767u;
constexpr uint32_t MEMORY_BIG_RELATION = 1662111297u;
constexpr uint32_t RANGE_CHECK_8_RELATION = 1420243005u;
constexpr uint32_t PARTIAL_EC_MUL_RELATION = 183619546u;

struct EcOpTraceColumns {
    uint32_t *columns[EC_OP_TRACE_COLUMNS];
};

struct PartialInputColumns {
    uint32_t *columns[PARTIAL_INPUT_COLUMNS];
};

__device__ __forceinline__ bool felt_equal(const felt252 &a, const felt252 &b) {
    for (uint32_t limb = 0; limb < 8; ++limb) {
        if (a.limbs[limb] != b.limbs[limb]) {
            return false;
        }
    }
    return true;
}

// Complete for every non-infinity result accepted by the host writer.  The
// generated witness panics if doubling or addition produces infinity, so the
// opposite-y equal-x case is outside the valid proof-input domain.
__device__ __forceinline__ void ec_double_affine_exact(
    const AffinePointCuda &point, AffinePointCuda &result) {
    felt252 x = felt_to_mont(point.x);
    felt252 y = felt_to_mont(point.y);
    felt252 x2 = felt_mul(x, x);
    felt252 numerator = felt_add(felt_add(x2, x2), x2);
    numerator = felt_add(
        numerator, ff_dispatch_st<ff_config_starknet>::get_one()); // curve alpha = 1
    felt252 denominator = felt_add(y, y);
    felt252 lambda = felt_mul(numerator, felt_inverse(denominator));
    felt252 x3 = felt_sub(felt_mul(lambda, lambda), felt_add(x, x));
    felt252 y3 = felt_sub(felt_mul(lambda, felt_sub(x, x3)), y);
    result.x = felt_from_mont(x3);
    result.y = felt_from_mont(y3);
}

__device__ __forceinline__ void ec_add_affine_exact(
    const AffinePointCuda &left,
    const AffinePointCuda &right,
    AffinePointCuda &result) {
    if (felt_equal(left.x, right.x) && felt_equal(left.y, right.y)) {
        ec_double_affine_exact(left, result);
    } else {
        ec_add_affine(left, right, result);
    }
}

__device__ __forceinline__ bool load_memory_value(
    const uint32_t *const *tables,
    uint32_t n_big,
    uint32_t n_small,
    uint32_t id,
    uint32_t *limbs) {
    uint32_t tag = id >> 30;
    uint32_t index = id & 0x3fffffffu;
    if (tag == 1u) {
        if (index >= n_big) {
            return false;
        }
        for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
            limbs[limb] = tables[1u + limb][index];
        }
        return true;
    }
    if (tag == 0u && index < n_small) {
        for (uint32_t limb = 0; limb < 8; ++limb) {
            limbs[limb] = tables[29u + limb][index];
        }
        for (uint32_t limb = 8; limb < FELT_LIMBS; ++limb) {
            limbs[limb] = 0;
        }
        return true;
    }
    return false;
}

__device__ __forceinline__ void store_trace_limbs(
    EcOpTraceColumns trace,
    uint32_t first_column,
    uint32_t row,
    const uint32_t *limbs) {
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        trace.columns[first_column + limb][row] = limbs[limb];
    }
}

__device__ __forceinline__ void store_lookup_word(
    uint32_t *lookup,
    uint32_t rows,
    uint32_t row,
    uint32_t word,
    uint32_t value) {
    lookup[static_cast<size_t>(word) * rows + row] = value;
}

__device__ __forceinline__ void store_memory_address_lookup(
    uint32_t *lookup,
    uint32_t rows,
    uint32_t row,
    uint32_t word,
    uint32_t address,
    uint32_t id) {
    store_lookup_word(lookup, rows, row, word, MEMORY_ADDRESS_RELATION);
    store_lookup_word(lookup, rows, row, word + 1u, address);
    store_lookup_word(lookup, rows, row, word + 2u, id);
}

__device__ __forceinline__ void store_memory_big_lookup(
    uint32_t *lookup,
    uint32_t rows,
    uint32_t row,
    uint32_t word,
    uint32_t id,
    const uint32_t *limbs) {
    store_lookup_word(lookup, rows, row, word, MEMORY_BIG_RELATION);
    store_lookup_word(lookup, rows, row, word + 1u, id);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        store_lookup_word(lookup, rows, row, word + 2u + limb, limbs[limb]);
    }
}

__device__ __forceinline__ void count_memory_input(
    uint32_t address,
    uint32_t id,
    uint32_t *address_counts,
    uint32_t *big_counts,
    uint32_t *small_counts) {
    // memory_address_to_id's canonical row key is address - 1.  The encoded
    // memory-id tag selects the exact big/small runtime multiplicity slab.
    atomicAdd(&address_counts[address - 1u], 1u);
    uint32_t tag = id >> 30;
    uint32_t index = id & 0x3fffffffu;
    if (tag == 1u) {
        atomicAdd(&big_counts[index], 1u);
    } else if (tag == 0u) {
        atomicAdd(&small_counts[index], 1u);
    }
}

__device__ __forceinline__ void felt_to_limbs(
    const felt252 &value, uint32_t *limbs) {
    felt252_to_m31_limbs(value, reinterpret_cast<m31 *>(limbs));
}

__device__ __forceinline__ void store_partial_input(
    PartialInputColumns output,
    uint32_t destination,
    uint32_t chain,
    uint32_t round,
    const uint32_t *m,
    const AffinePointCuda &q,
    const AffinePointCuda &accumulator,
    uint32_t counter,
    uint32_t enabler) {
    uint32_t limbs[FELT_LIMBS];
    output.columns[0][destination] = chain;
    output.columns[1][destination] = round;
    for (uint32_t word = 0; word < W27_WORDS; ++word) {
        output.columns[2u + word][destination] = m[word];
    }
    felt_to_limbs(q.x, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        output.columns[12u + limb][destination] = limbs[limb];
    }
    felt_to_limbs(q.y, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        output.columns[40u + limb][destination] = limbs[limb];
    }
    felt_to_limbs(accumulator.x, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        output.columns[68u + limb][destination] = limbs[limb];
    }
    felt_to_limbs(accumulator.y, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        output.columns[96u + limb][destination] = limbs[limb];
    }
    output.columns[124][destination] = counter;
    output.columns[125][destination] = enabler;
    output.columns[126][destination] = destination;
}

__device__ __forceinline__ void store_partial_lookup(
    uint32_t *lookup,
    uint32_t rows,
    uint32_t row,
    uint32_t first_word,
    uint32_t chain,
    uint32_t round,
    const uint32_t *m,
    const AffinePointCuda &q,
    const AffinePointCuda &accumulator,
    uint32_t counter) {
    uint32_t limbs[FELT_LIMBS];
    uint32_t word = first_word;
    store_lookup_word(lookup, rows, row, word++, PARTIAL_EC_MUL_RELATION);
    store_lookup_word(lookup, rows, row, word++, chain);
    store_lookup_word(lookup, rows, row, word++, round);
    for (uint32_t i = 0; i < W27_WORDS; ++i) {
        store_lookup_word(lookup, rows, row, word++, m[i]);
    }
    felt_to_limbs(q.x, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        store_lookup_word(lookup, rows, row, word++, limbs[limb]);
    }
    felt_to_limbs(q.y, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        store_lookup_word(lookup, rows, row, word++, limbs[limb]);
    }
    felt_to_limbs(accumulator.x, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        store_lookup_word(lookup, rows, row, word++, limbs[limb]);
    }
    felt_to_limbs(accumulator.y, limbs);
    for (uint32_t limb = 0; limb < FELT_LIMBS; ++limb) {
        store_lookup_word(lookup, rows, row, word++, limbs[limb]);
    }
    store_lookup_word(lookup, rows, row, word, counter);
}

__global__ void __launch_bounds__(EC_OP_BLOCK) ec_op_witness_kernel(
    const uint32_t *const *tables,
    uint32_t n_addresses,
    uint32_t n_big,
    uint32_t n_small,
    const uint32_t *segment_start_source,
    uint32_t rows,
    EcOpTraceColumns trace,
    uint32_t *lookup,
    PartialInputColumns partial,
    uint32_t *address_counts,
    uint32_t *big_counts,
    uint32_t *small_counts,
    uint32_t *range_check_8_counts) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const uint32_t *addr_to_id = tables[0];
    uint32_t segment_start = *segment_start_source;
    uint64_t end = static_cast<uint64_t>(segment_start) +
                   static_cast<uint64_t>(rows) * 7u;
    if (segment_start == 0u || end > n_addresses) {
        return;
    }
    uint32_t base = segment_start + 7u * row;
    uint32_t limbs[FELT_LIMBS];
    uint32_t m[W27_WORDS];
    AffinePointCuda accumulator;
    AffinePointCuda q;

    // Five input cells: P.x, P.y, Q.x, Q.y, scalar m.
    uint32_t id = addr_to_id[base];
    if (!load_memory_value(tables, n_big, n_small, id, limbs)) return;
    trace.columns[0][row] = id;
    count_memory_input(base, id, address_counts, big_counts, small_counts);
    store_trace_limbs(trace, 1, row, limbs);
    store_memory_address_lookup(lookup, rows, row, 0, base, id);
    store_memory_big_lookup(lookup, rows, row, 3, id, limbs);
    felt252_from_m31_limbs(accumulator.x, reinterpret_cast<m31 *>(limbs));

    id = addr_to_id[base + 1u];
    if (!load_memory_value(tables, n_big, n_small, id, limbs)) return;
    trace.columns[29][row] = id;
    count_memory_input(base + 1u, id, address_counts, big_counts, small_counts);
    store_trace_limbs(trace, 30, row, limbs);
    store_memory_address_lookup(lookup, rows, row, 33, base + 1u, id);
    store_memory_big_lookup(lookup, rows, row, 36, id, limbs);
    felt252_from_m31_limbs(accumulator.y, reinterpret_cast<m31 *>(limbs));

    id = addr_to_id[base + 2u];
    if (!load_memory_value(tables, n_big, n_small, id, limbs)) return;
    trace.columns[58][row] = id;
    count_memory_input(base + 2u, id, address_counts, big_counts, small_counts);
    store_trace_limbs(trace, 59, row, limbs);
    store_memory_address_lookup(lookup, rows, row, 66, base + 2u, id);
    store_memory_big_lookup(lookup, rows, row, 69, id, limbs);
    felt252_from_m31_limbs(q.x, reinterpret_cast<m31 *>(limbs));

    id = addr_to_id[base + 3u];
    if (!load_memory_value(tables, n_big, n_small, id, limbs)) return;
    trace.columns[87][row] = id;
    count_memory_input(base + 3u, id, address_counts, big_counts, small_counts);
    store_trace_limbs(trace, 88, row, limbs);
    store_memory_address_lookup(lookup, rows, row, 99, base + 3u, id);
    store_memory_big_lookup(lookup, rows, row, 102, id, limbs);
    felt252_from_m31_limbs(q.y, reinterpret_cast<m31 *>(limbs));

    id = addr_to_id[base + 4u];
    if (!load_memory_value(tables, n_big, n_small, id, limbs)) return;
    trace.columns[116][row] = id;
    count_memory_input(base + 4u, id, address_counts, big_counts, small_counts);
    store_trace_limbs(trace, 117, row, limbs);
    store_memory_address_lookup(lookup, rows, row, 132, base + 4u, id);
    store_memory_big_lookup(lookup, rows, row, 135, id, limbs);
    for (uint32_t word = 0; word < 9; ++word) {
        m[word] = limbs[3u * word] | (limbs[3u * word + 1u] << 9) |
                  (limbs[3u * word + 2u] << 18);
    }
    m[9] = limbs[27];

    uint32_t ms_is_max = limbs[27] == 256u;
    uint32_t ms_and_mid_are_max = ms_is_max && limbs[21] == 136u;
    uint32_t rc0 = limbs[27] - ms_is_max;
    uint32_t rc1 = ms_is_max * (120u + limbs[21] - ms_and_mid_are_max);
    trace.columns[145][row] = ms_is_max;
    trace.columns[146][row] = ms_and_mid_are_max;
    trace.columns[147][row] = rc1;
    store_lookup_word(lookup, rows, row, 165, RANGE_CHECK_8_RELATION);
    store_lookup_word(lookup, rows, row, 166, rc0);
    store_lookup_word(lookup, rows, row, 167, RANGE_CHECK_8_RELATION);
    store_lookup_word(lookup, rows, row, 168, rc1);
    atomicAdd(&range_check_8_counts[rc0], 1u);
    atomicAdd(&range_check_8_counts[rc1], 1u);

    uint32_t counter = 26;
    store_partial_lookup(lookup, rows, row, 169, row, 0, m, q, accumulator, counter);

    for (uint32_t round = 0; round < PARTIAL_ROUNDS; ++round) {
        uint32_t destination = round * rows + row;
        store_partial_input(partial, destination, row, round, m, q, accumulator, counter, 1);

        if ((m[0] & 1u) != 0) {
            AffinePointCuda sum;
            ec_add_affine_exact(accumulator, q, sum);
            accumulator = sum;
        }
        AffinePointCuda doubled;
        ec_double_affine_exact(q, doubled);
        q = doubled;

        if (counter == 0) {
            for (uint32_t word = 0; word + 1u < W27_WORDS; ++word) {
                m[word] = m[word + 1u];
            }
            m[W27_WORDS - 1u] = 0;
            counter = 26;
        } else {
            m[0] >>= 1;
            --counter;
        }
    }

    for (uint32_t word = 0; word < W27_WORDS; ++word) {
        trace.columns[148u + word][row] = m[word];
    }
    felt_to_limbs(q.x, limbs);
    store_trace_limbs(trace, 158, row, limbs);
    felt_to_limbs(q.y, limbs);
    store_trace_limbs(trace, 186, row, limbs);
    felt_to_limbs(accumulator.x, limbs);
    store_trace_limbs(trace, 214, row, limbs);
    felt_to_limbs(accumulator.y, limbs);
    store_trace_limbs(trace, 242, row, limbs);
    trace.columns[270][row] = counter;
    store_partial_lookup(
        lookup, rows, row, 295, row, PARTIAL_ROUNDS, m, q, accumulator, counter);

    uint32_t result_x_id = addr_to_id[base + 5u];
    trace.columns[271][row] = result_x_id;
    count_memory_input(base + 5u, result_x_id, address_counts, big_counts, small_counts);
    store_memory_address_lookup(lookup, rows, row, 421, base + 5u, result_x_id);
    felt_to_limbs(accumulator.x, limbs);
    store_memory_big_lookup(lookup, rows, row, 424, result_x_id, limbs);

    uint32_t result_y_id = addr_to_id[base + 6u];
    trace.columns[272][row] = result_y_id;
    count_memory_input(base + 6u, result_y_id, address_counts, big_counts, small_counts);
    store_memory_address_lookup(lookup, rows, row, 454, base + 6u, result_y_id);
    felt_to_limbs(accumulator.y, limbs);
    store_memory_big_lookup(lookup, rows, row, 457, result_y_id, limbs);
    store_lookup_word(lookup, rows, row, 487, 1);
}

// The generated partial_ec_mul_generic writer pads its packed-input vector to
// the next power of two by repeating packed input 0.  Since an EC-op component
// has a power-of-two row count, 252 rounds pad to exactly 256 rounds.  Every
// appended SIMD pack therefore repeats source rows 0..15 from round zero.
__global__ void partial_input_padding_kernel(
    uint32_t rows, PartialInputColumns partial) {
    uint32_t pad_row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t padding_rows = (PARTIAL_PADDED_ROUNDS - PARTIAL_ROUNDS) * rows;
    if (pad_row >= padding_rows) {
        return;
    }
    uint32_t destination = PARTIAL_ROUNDS * rows + pad_row;
    uint32_t source = pad_row & 15u;
    for (uint32_t column = 0; column < 125; ++column) {
        partial.columns[column][destination] = partial.columns[column][source];
    }
    partial.columns[125][destination] = 0;
    partial.columns[126][destination] = destination;
}

} // namespace

extern "C" int ec_op_builtin_witness_on(
    const uint32_t *const *execution_tables,
    uint32_t n_addresses,
    uint32_t n_big,
    uint32_t n_small,
    const uint32_t *segment_start_source,
    uint32_t row_count,
    uint32_t *const *trace_columns_host,
    uint32_t *lookup_words,
    uint32_t *const *partial_input_columns_host,
    uint32_t partial_row_count,
    uint32_t *address_counts,
    uint32_t address_count_words,
    uint32_t *big_counts,
    uint32_t big_count_words,
    uint32_t *small_counts,
    uint32_t small_count_words,
    uint32_t *range_check_8_counts,
    uint32_t range_check_8_count_words,
    cudaStream_t stream) {
    if (execution_tables == nullptr || trace_columns_host == nullptr ||
        lookup_words == nullptr || partial_input_columns_host == nullptr ||
        segment_start_source == nullptr || address_counts == nullptr ||
        big_counts == nullptr || small_counts == nullptr ||
        range_check_8_counts == nullptr || stream == nullptr || row_count < 16u ||
        address_count_words < n_addresses - 1u || big_count_words < n_big ||
        small_count_words < n_small || range_check_8_count_words < 256u ||
        partial_row_count != PARTIAL_PADDED_ROUNDS * row_count) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    EcOpTraceColumns trace = {};
    PartialInputColumns partial = {};
    for (uint32_t column = 0; column < EC_OP_TRACE_COLUMNS; ++column) {
        if (trace_columns_host[column] == nullptr) {
            return static_cast<int>(cudaErrorInvalidDevicePointer);
        }
        trace.columns[column] = trace_columns_host[column];
    }
    for (uint32_t column = 0; column < PARTIAL_INPUT_COLUMNS; ++column) {
        if (partial_input_columns_host[column] == nullptr) {
            return static_cast<int>(cudaErrorInvalidDevicePointer);
        }
        partial.columns[column] = partial_input_columns_host[column];
    }

    uint32_t blocks = (row_count + EC_OP_BLOCK - 1u) / EC_OP_BLOCK;
    ec_op_witness_kernel<<<blocks, EC_OP_BLOCK, 0, stream>>>(
        execution_tables, n_addresses, n_big, n_small, segment_start_source,
        row_count, trace, lookup_words, partial, address_counts, big_counts,
        small_counts, range_check_8_counts);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return static_cast<int>(error);
    }

    uint32_t padding_rows = partial_row_count - PARTIAL_ROUNDS * row_count;
    blocks = (padding_rows + EC_OP_BLOCK - 1u) / EC_OP_BLOCK;
    partial_input_padding_kernel<<<blocks, EC_OP_BLOCK, 0, stream>>>(row_count, partial);
    return static_cast<int>(cudaGetLastError());
}
