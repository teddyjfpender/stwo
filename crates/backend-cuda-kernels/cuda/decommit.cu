#include "decommit.cuh"
#include "blake2s.cuh"

#include <cuda_runtime.h>

namespace {

constexpr uint32_t BLOCK = 256;
constexpr uint32_t HASH_WORDS = 8;
constexpr uint32_t AUX_NODE_WORDS = 10; // level, index, hash[8]
constexpr uint32_t M31_P = 0x7fffffffu;

enum HeaderWord : uint32_t {
    H_MAGIC = 0,
    H_VERSION = 1,
    H_TREE_COUNT = 2,
    H_RAW_COUNT = 3,
    H_UNIQUE_COUNT = 4,
    H_RAW_OFFSET = 5,
    H_UNIQUE_OFFSET = 6,
    H_USED_WORDS = 7,
};

enum TreeMetaWord : uint32_t {
    M_KIND = 0,
    M_ROLE = 1,
    M_QUERY_OFFSET = 2,
    M_QUERY_COUNT = 3,
    M_VALUES_OFFSET = 4,
    M_VALUES_COUNT = 5,
    M_FRI_WITNESS_OFFSET = 6,
    M_FRI_WITNESS_COUNT = 7,
    M_HASH_WITNESS_OFFSET = 8,
    M_HASH_WITNESS_COUNT = 9,
    M_AUX_OFFSET = 10,
    M_AUX_COUNT = 11,
    M_ALL_VALUES_OFFSET = 12,
    M_ALL_VALUES_COUNT = 13,
    M_LEAF_LOG_SIZE = 14,
    M_USED_WORDS = 15,
};

__device__ __forceinline__ uint32_t lifted_index(
    uint32_t position,
    uint32_t lifting_log_size,
    uint32_t column_log_size) {
    const uint32_t shift = lifting_log_size - column_log_size;
    return shift == 0
        ? position
        : ((position >> (shift + 1)) << 1) + (position & 1u);
}

__device__ __forceinline__ uint32_t canonical_m31(uint32_t value) {
    // Resident columns may retain the valid unreduced zero representation P.
    // Proof values use Column::at semantics; sparse leaf hashing intentionally
    // continues reading the untouched raw word stream.
    return value < M31_P ? value : value % M31_P;
}

__device__ uint32_t sort_unique(uint32_t *values, uint32_t count) {
    // Query counts are protocol constants (currently O(10^2)); serial insertion
    // sort avoids CUB temporary storage and remains graph-address-stable.
    for (uint32_t i = 1; i < count; ++i) {
        const uint32_t value = values[i];
        uint32_t j = i;
        while (j != 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
    uint32_t unique = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (unique == 0 || values[i] != values[unique - 1]) {
            values[unique++] = values[i];
        }
    }
    return unique;
}

__device__ __forceinline__ uint32_t map_query_log(
    uint32_t position,
    uint32_t source_log_size,
    uint32_t target_log_size) {
    if (source_log_size < target_log_size) {
        return ((position >> 1) << (target_log_size - source_log_size + 1)) |
               (position & 1u);
    }
    return ((position >> (source_log_size - target_log_size + 1)) << 1) |
           (position & 1u);
}

__device__ __forceinline__ uint32_t *tree_meta(uint32_t *assembly, uint32_t tree_index) {
    return assembly + STWO_DECOMMIT_HEADER_WORDS +
           tree_index * STWO_DECOMMIT_TREE_META_WORDS;
}

__device__ bool reserve_words(
    uint32_t *assembly,
    uint32_t capacity,
    uint32_t words,
    uint32_t *offset) {
    const uint32_t cursor = assembly[H_USED_WORDS];
    if (cursor > capacity || words > capacity - cursor) {
        assembly[H_USED_WORDS] = 0;
        return false;
    }
    *offset = cursor;
    assembly[H_USED_WORDS] = cursor + words;
    return true;
}

__device__ const Blake2sHash *find_sparse(
    uint32_t level,
    uint32_t index,
    uint32_t leaf_log_size,
    const uint32_t *indices,
    const Blake2sHash *hashes,
    const uint32_t *level_offsets,
    const uint32_t *level_counts,
    uint32_t level_count) {
    const uint32_t distance = leaf_log_size - level;
    if (distance >= level_count) return nullptr;
    const uint32_t offset = level_offsets[distance];
    uint32_t lo = 0;
    uint32_t hi = level_counts[distance];
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1);
        const uint32_t current = indices[offset + mid];
        if (current < index) lo = mid + 1;
        else hi = mid;
    }
    return (lo < level_counts[distance] && indices[offset + lo] == index)
        ? &hashes[offset + lo]
        : nullptr;
}

__device__ const Blake2sHash *node_hash(
    uint32_t level,
    uint32_t index,
    uint32_t leaf_log_size,
    uint32_t first_retained_log_size,
    const Blake2sHash *const *retained,
    const uint32_t *sparse_indices,
    const Blake2sHash *sparse_hashes,
    const uint32_t *sparse_offsets,
    const uint32_t *sparse_counts,
    uint32_t sparse_level_count) {
    if (level <= first_retained_log_size) {
        return &retained[level][index];
    }
    return find_sparse(level, index, leaf_log_size, sparse_indices, sparse_hashes,
                       sparse_offsets, sparse_counts, sparse_level_count);
}

__device__ void write_hash(uint32_t *destination, const Blake2sHash &hash) {
    #pragma unroll
    for (uint32_t i = 0; i < HASH_WORDS; ++i) destination[i] = hash.s[i];
}

__global__ void normalize_queries_kernel(
    const uint32_t *raw,
    uint32_t raw_count,
    uint32_t query_log_size,
    uint32_t tree_count,
    uint32_t *unique,
    uint32_t *unique_count,
    uint32_t *assembly,
    uint32_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t mask = (1u << query_log_size) - 1u;
    for (uint32_t i = 0; i < raw_count; ++i) unique[i] = raw[i] & mask;
    const uint32_t count = sort_unique(unique, raw_count);
    *unique_count = count;

    const uint32_t raw_offset = STWO_DECOMMIT_HEADER_WORDS +
        tree_count * STWO_DECOMMIT_TREE_META_WORDS;
    const uint32_t unique_offset = raw_offset + raw_count;
    const uint32_t used = unique_offset + count;
    if (used > capacity) {
        assembly[H_USED_WORDS] = 0;
        return;
    }
    assembly[H_MAGIC] = STWO_DECOMMIT_MAGIC;
    assembly[H_VERSION] = STWO_DECOMMIT_VERSION;
    assembly[H_TREE_COUNT] = tree_count;
    assembly[H_RAW_COUNT] = raw_count;
    assembly[H_UNIQUE_COUNT] = count;
    assembly[H_RAW_OFFSET] = raw_offset;
    assembly[H_UNIQUE_OFFSET] = unique_offset;
    assembly[H_USED_WORDS] = used;
    for (uint32_t i = 0; i < tree_count * STWO_DECOMMIT_TREE_META_WORDS; ++i) {
        assembly[STWO_DECOMMIT_HEADER_WORDS + i] = 0;
    }
    for (uint32_t i = 0; i < raw_count; ++i) assembly[raw_offset + i] = raw[i] & mask;
    for (uint32_t i = 0; i < count; ++i) assembly[unique_offset + i] = unique[i];
}

__global__ void prepare_trace_queries_kernel(
    const uint32_t *unique,
    const uint32_t *unique_count,
    uint32_t max_queries,
    uint32_t source_log,
    uint32_t tree_log,
    uint32_t leaf_log,
    uint32_t unretained,
    uint32_t *mapped,
    uint32_t *mapped_count,
    uint32_t *walk,
    uint32_t *walk_count,
    uint32_t *leaf_indices,
    uint32_t *leaf_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t count = min(*unique_count, max_queries);
    for (uint32_t i = 0; i < count; ++i) {
        mapped[i] = map_query_log(unique[i], source_log, tree_log);
        walk[i] = mapped[i];
    }
    *mapped_count = count;
    const uint32_t dedup = sort_unique(walk, count);
    *walk_count = dedup;

    if (unretained == 0) {
        *leaf_count = 0;
        return;
    }
    const uint32_t span = 1u << unretained;
    uint32_t leaves = 0;
    for (uint32_t i = 0; i < dedup; ++i) {
        const uint32_t base = (walk[i] >> unretained) << unretained;
        for (uint32_t j = 0; j < span; ++j) leaf_indices[leaves++] = base + j;
    }
    *leaf_count = sort_unique(leaf_indices, leaves);
}

__global__ void gather_trace_values_kernel(
    const uint32_t *const *columns,
    const uint32_t *column_logs,
    uint32_t n_columns,
    uint32_t lifting_log,
    const uint32_t *queries,
    const uint32_t *query_count,
    uint32_t max_queries,
    uint32_t first_column,
    uint32_t stride,
    uint32_t *output) {
    const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (column >= n_columns || q >= min(*query_count, max_queries)) return;
    const uint32_t row = lifted_index(queries[q], lifting_log, column_logs[column]);
    output[(size_t)(first_column + column) * stride + q] =
        canonical_m31(columns[column][row]);
}

__global__ void sparse_parent_kernel(
    const uint32_t *child_indices,
    const Blake2sHash *child_hashes,
    const uint32_t *child_count,
    uint32_t max_child_count,
    uint32_t *parent_indices,
    Blake2sHash *parent_hashes,
    uint32_t *parent_count) {
    const uint32_t parent = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = min(*child_count, max_child_count);
    const uint32_t parents = count >> 1;
    if (parent == 0) *parent_count = parents;
    if (parent >= parents) return;
    const uint32_t left = 2 * parent;
    parent_indices[parent] = child_indices[left] >> 1;
    stwo_blake2s_hash2_device(
        reinterpret_cast<const uint8_t *>(&child_hashes[left]), sizeof(Blake2sHash),
        reinterpret_cast<const uint8_t *>(&child_hashes[left + 1]), sizeof(Blake2sHash),
        &parent_hashes[parent]);
}

__global__ void assemble_trace_kernel(
    uint32_t tree_index,
    uint32_t role,
    uint32_t leaf_log,
    uint32_t first_retained_log,
    uint32_t column_count,
    const uint32_t *mapped,
    const uint32_t *mapped_count_ptr,
    uint32_t max_queries,
    uint32_t *walk,
    uint32_t *scratch,
    const uint32_t *walk_count_ptr,
    const uint32_t *values,
    const Blake2sHash *const *retained,
    const uint32_t *sparse_indices,
    const Blake2sHash *sparse_hashes,
    const uint32_t *sparse_offsets,
    const uint32_t *sparse_counts,
    uint32_t sparse_level_count,
    uint32_t *assembly,
    uint32_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t *meta = tree_meta(assembly, tree_index);
    const uint32_t tree_start = assembly[H_USED_WORDS];
    const uint32_t mapped_count = min(*mapped_count_ptr, max_queries);

    uint32_t offset = 0;
    if (!reserve_words(assembly, capacity, mapped_count, &offset)) return;
    meta[M_QUERY_OFFSET] = offset;
    meta[M_QUERY_COUNT] = mapped_count;
    for (uint32_t i = 0; i < mapped_count; ++i) assembly[offset + i] = mapped[i];

    const uint32_t value_words = column_count * mapped_count;
    if (!reserve_words(assembly, capacity, value_words, &offset)) return;
    meta[M_VALUES_OFFSET] = offset;
    meta[M_VALUES_COUNT] = value_words;
    for (uint32_t c = 0; c < column_count; ++c) {
        for (uint32_t q = 0; q < mapped_count; ++q) {
            assembly[offset + c * mapped_count + q] = values[(size_t)c * max_queries + q];
        }
    }

    uint32_t current_count = min(*walk_count_ptr, max_queries);
    uint32_t *current = walk;
    uint32_t *next = scratch;
    const uint32_t hash_offset = assembly[H_USED_WORDS];
    uint32_t hash_count = 0;
    const uint32_t aux_offset = hash_offset + leaf_log * current_count * HASH_WORDS;
    // Reserve conservative bounds once; compact the aux region after the walk.
    const uint32_t reserve = leaf_log * current_count * (HASH_WORDS + 2 * AUX_NODE_WORDS);
    if (!reserve_words(assembly, capacity, reserve, &offset)) return;
    uint32_t aux_count = 0;
    for (int32_t layer = (int32_t)leaf_log - 1; layer >= 0; --layer) {
        const uint32_t previous_level = (uint32_t)layer + 1;
        uint32_t next_count = 0;
        for (uint32_t i = 0; i < current_count;) {
            const uint32_t first = current[i];
            const bool pair = i + 1 < current_count && current[i + 1] == (first ^ 1u);
            if (!pair) {
                const Blake2sHash *hash = node_hash(
                    previous_level, first ^ 1u, leaf_log, first_retained_log,
                    retained, sparse_indices, sparse_hashes, sparse_offsets, sparse_counts,
                    sparse_level_count);
                if (hash == nullptr) { assembly[H_USED_WORDS] = 0; return; }
                write_hash(assembly + hash_offset + hash_count * HASH_WORDS, *hash);
                ++hash_count;
            }
            const uint32_t parent = first >> 1;
            next[next_count++] = parent;
            for (uint32_t child = 2 * parent; child <= 2 * parent + 1; ++child) {
                const Blake2sHash *hash = node_hash(
                    previous_level, child, leaf_log, first_retained_log,
                    retained, sparse_indices, sparse_hashes, sparse_offsets, sparse_counts,
                    sparse_level_count);
                if (hash == nullptr) { assembly[H_USED_WORDS] = 0; return; }
                uint32_t *entry = assembly + aux_offset + aux_count * AUX_NODE_WORDS;
                entry[0] = previous_level;
                entry[1] = child;
                write_hash(entry + 2, *hash);
                ++aux_count;
            }
            i += pair ? 2 : 1;
        }
        uint32_t *swap = current; current = next; next = swap;
        current_count = next_count;
    }

    // Move aux immediately after the exact hash witness, eliminating the
    // conservative gap before publishing offsets and cursor.
    const uint32_t compact_aux = hash_offset + hash_count * HASH_WORDS;
    for (uint32_t i = 0; i < aux_count * AUX_NODE_WORDS; ++i) {
        assembly[compact_aux + i] = assembly[aux_offset + i];
    }
    assembly[H_USED_WORDS] = compact_aux + aux_count * AUX_NODE_WORDS;
    meta[M_KIND] = 0;
    meta[M_ROLE] = role;
    meta[M_FRI_WITNESS_OFFSET] = 0;
    meta[M_FRI_WITNESS_COUNT] = 0;
    meta[M_HASH_WITNESS_OFFSET] = hash_offset;
    meta[M_HASH_WITNESS_COUNT] = hash_count;
    meta[M_AUX_OFFSET] = compact_aux;
    meta[M_AUX_COUNT] = aux_count;
    meta[M_ALL_VALUES_OFFSET] = 0;
    meta[M_ALL_VALUES_COUNT] = 0;
    meta[M_LEAF_LOG_SIZE] = leaf_log;
    meta[M_USED_WORDS] = assembly[H_USED_WORDS] - tree_start;
}

__global__ void prepare_fri_queries_kernel(
    const uint32_t *unique,
    const uint32_t *unique_count,
    uint32_t max_queries,
    uint32_t cumulative_fold,
    uint32_t fold_step,
    uint32_t packed_log,
    uint32_t *tree_queries,
    uint32_t *tree_count,
    uint32_t *expanded,
    uint32_t *expanded_count,
    uint32_t *walk,
    uint32_t *walk_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t count = min(*unique_count, max_queries);
    for (uint32_t i = 0; i < count; ++i) tree_queries[i] = unique[i] >> cumulative_fold;
    const uint32_t queries = sort_unique(tree_queries, count);
    *tree_count = queries;

    uint32_t out = 0;
    uint32_t previous_coset = 0xffffffffu;
    const uint32_t coset_size = 1u << fold_step;
    for (uint32_t i = 0; i < queries; ++i) {
        const uint32_t coset = tree_queries[i] >> fold_step;
        if (coset == previous_coset) continue;
        previous_coset = coset;
        const uint32_t start = coset << fold_step;
        for (uint32_t j = 0; j < coset_size; ++j) expanded[out++] = start + j;
    }
    *expanded_count = out;
    for (uint32_t i = 0; i < out; ++i) walk[i] = expanded[i] >> packed_log;
    *walk_count = sort_unique(walk, out);
}

__global__ void gather_fri_values_kernel(
    const uint32_t *const *coordinates,
    const uint32_t *positions,
    const uint32_t *count_ptr,
    uint32_t max_positions,
    uint32_t *values) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= min(*count_ptr, max_positions)) return;
    const uint32_t position = positions[i];
    #pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        values[4 * i + c] = canonical_m31(coordinates[c][position]);
    }
}

__device__ bool contains_sorted(const uint32_t *values, uint32_t count, uint32_t target) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1);
        if (values[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo < count && values[lo] == target;
}

__global__ void assemble_fri_kernel(
    uint32_t tree_index,
    uint32_t leaf_log,
    const uint32_t *tree_queries,
    const uint32_t *tree_count_ptr,
    const uint32_t *expanded,
    const uint32_t *expanded_count_ptr,
    const uint32_t *values,
    uint32_t *walk,
    uint32_t *scratch,
    const uint32_t *walk_count_ptr,
    const Blake2sHash *const *retained,
    uint32_t *assembly,
    uint32_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t *meta = tree_meta(assembly, tree_index);
    const uint32_t tree_start = assembly[H_USED_WORDS];
    const uint32_t query_count = *tree_count_ptr;
    const uint32_t expanded_count = *expanded_count_ptr;
    uint32_t offset = 0;
    if (!reserve_words(assembly, capacity, query_count, &offset)) return;
    meta[M_QUERY_OFFSET] = offset;
    meta[M_QUERY_COUNT] = query_count;
    for (uint32_t i = 0; i < query_count; ++i) assembly[offset + i] = tree_queries[i];

    uint32_t witness_count = 0;
    for (uint32_t i = 0; i < expanded_count; ++i) {
        if (!contains_sorted(tree_queries, query_count, expanded[i])) ++witness_count;
    }
    if (!reserve_words(assembly, capacity, 4 * witness_count, &offset)) return;
    meta[M_FRI_WITNESS_OFFSET] = offset;
    meta[M_FRI_WITNESS_COUNT] = witness_count;
    uint32_t witness = 0;
    for (uint32_t i = 0; i < expanded_count; ++i) {
        if (!contains_sorted(tree_queries, query_count, expanded[i])) {
            #pragma unroll
            for (uint32_t c = 0; c < 4; ++c) assembly[offset + 4 * witness + c] = values[4 * i + c];
            ++witness;
        }
    }

    uint32_t current_count = *walk_count_ptr;
    uint32_t *current = walk;
    uint32_t *next = scratch;
    const uint32_t hash_offset = assembly[H_USED_WORDS];
    const uint32_t aux_offset = hash_offset + leaf_log * current_count * HASH_WORDS;
    const uint32_t reserve = leaf_log * current_count * (HASH_WORDS + 2 * AUX_NODE_WORDS);
    if (!reserve_words(assembly, capacity, reserve, &offset)) return;
    uint32_t hash_count = 0, aux_count = 0;
    for (int32_t layer = (int32_t)leaf_log - 1; layer >= 0; --layer) {
        const uint32_t previous_level = (uint32_t)layer + 1;
        uint32_t next_count = 0;
        for (uint32_t i = 0; i < current_count;) {
            const uint32_t first = current[i];
            const bool pair = i + 1 < current_count && current[i + 1] == (first ^ 1u);
            if (!pair) {
                write_hash(assembly + hash_offset + hash_count * HASH_WORDS,
                           retained[previous_level][first ^ 1u]);
                ++hash_count;
            }
            const uint32_t parent = first >> 1;
            next[next_count++] = parent;
            for (uint32_t child = 2 * parent; child <= 2 * parent + 1; ++child) {
                uint32_t *entry = assembly + aux_offset + aux_count * AUX_NODE_WORDS;
                entry[0] = previous_level;
                entry[1] = child;
                write_hash(entry + 2, retained[previous_level][child]);
                ++aux_count;
            }
            i += pair ? 2 : 1;
        }
        uint32_t *swap = current; current = next; next = swap;
        current_count = next_count;
    }
    const uint32_t compact_aux = hash_offset + hash_count * HASH_WORDS;
    for (uint32_t i = 0; i < aux_count * AUX_NODE_WORDS; ++i) {
        assembly[compact_aux + i] = assembly[aux_offset + i];
    }
    assembly[H_USED_WORDS] = compact_aux + aux_count * AUX_NODE_WORDS;

    if (!reserve_words(assembly, capacity, expanded_count * 5, &offset)) return;
    meta[M_ALL_VALUES_OFFSET] = offset;
    meta[M_ALL_VALUES_COUNT] = expanded_count;
    for (uint32_t i = 0; i < expanded_count; ++i) {
        assembly[offset + 5 * i] = expanded[i];
        #pragma unroll
        for (uint32_t c = 0; c < 4; ++c) assembly[offset + 5 * i + 1 + c] = values[4 * i + c];
    }
    meta[M_KIND] = 1;
    meta[M_ROLE] = tree_index;
    meta[M_VALUES_OFFSET] = 0;
    meta[M_VALUES_COUNT] = 0;
    meta[M_HASH_WITNESS_OFFSET] = hash_offset;
    meta[M_HASH_WITNESS_COUNT] = hash_count;
    meta[M_AUX_OFFSET] = compact_aux;
    meta[M_AUX_COUNT] = aux_count;
    meta[M_LEAF_LOG_SIZE] = leaf_log;
    meta[M_USED_WORDS] = assembly[H_USED_WORDS] - tree_start;
}

} // namespace

extern "C" int stwo_decommit_normalize_queries_on(
    const uint32_t *raw, uint32_t raw_count, uint32_t log_size, uint32_t tree_count,
    uint32_t *unique, uint32_t *unique_count, uint32_t *assembly,
    uint32_t capacity, void *stream) {
    if (!raw || raw_count == 0 || log_size == 0 || log_size >= 31 || !unique ||
        !unique_count || !assembly || !stream) return cudaErrorInvalidValue;
    normalize_queries_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        raw, raw_count, log_size, tree_count, unique, unique_count, assembly, capacity);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_prepare_trace_queries_on(
    const uint32_t *unique, const uint32_t *unique_count, uint32_t max_queries,
    uint32_t source_log, uint32_t tree_log, uint32_t leaf_log, uint32_t unretained,
    uint32_t *mapped, uint32_t *mapped_count, uint32_t *walk, uint32_t *walk_count,
    uint32_t *leaves, uint32_t *leaf_count, void *stream) {
    if (!unique || !unique_count || max_queries == 0 || source_log >= 31 || tree_log >= 31 ||
        leaf_log >= 31 || unretained > leaf_log || !mapped || !mapped_count || !walk ||
        !walk_count || !leaves || !leaf_count || !stream) return cudaErrorInvalidValue;
    prepare_trace_queries_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        unique, unique_count, max_queries, source_log, tree_log, leaf_log, unretained,
        mapped, mapped_count, walk, walk_count, leaves, leaf_count);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_gather_trace_values_on(
    const uint32_t *const *columns, const uint32_t *logs, uint32_t n_columns,
    uint32_t lifting_log, const uint32_t *queries, const uint32_t *query_count,
    uint32_t max_queries, uint32_t first_column, uint32_t stride, uint32_t *output,
    void *stream) {
    if (!columns || !logs || n_columns == 0 || lifting_log >= 31 || !queries ||
        !query_count || max_queries == 0 || stride < max_queries || !output || !stream)
        return cudaErrorInvalidValue;
    const dim3 grid((max_queries + BLOCK - 1) / BLOCK, n_columns);
    gather_trace_values_kernel<<<grid, BLOCK, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        columns, logs, n_columns, lifting_log, queries, query_count, max_queries,
        first_column, stride, output);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_sparse_parent_on(
    const uint32_t *child_indices, const Blake2sHash *child_hashes,
    const uint32_t *child_count, uint32_t max_child_count, uint32_t *parent_indices,
    Blake2sHash *parent_hashes, uint32_t *parent_count, void *stream) {
    if (!child_indices || !child_hashes || !child_count || max_child_count < 2 ||
        !parent_indices || !parent_hashes || !parent_count || !stream)
        return cudaErrorInvalidValue;
    const uint32_t max_parents = max_child_count >> 1;
    sparse_parent_kernel<<<(max_parents + BLOCK - 1) / BLOCK, BLOCK, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(child_indices, child_hashes, child_count,
        max_child_count, parent_indices, parent_hashes, parent_count);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_assemble_trace_on(
    uint32_t tree_index, uint32_t role, uint32_t leaf_log, uint32_t first_retained_log,
    uint32_t column_count, const uint32_t *mapped, const uint32_t *mapped_count,
    uint32_t max_queries, uint32_t *walk, uint32_t *walk_scratch,
    const uint32_t *walk_count, const uint32_t *values,
    const Blake2sHash *const *retained, const uint32_t *sparse_indices,
    const Blake2sHash *sparse_hashes, const uint32_t *sparse_offsets,
    const uint32_t *sparse_counts, uint32_t sparse_level_count, uint32_t *assembly,
    uint32_t capacity, void *stream) {
    if (leaf_log >= 31 || first_retained_log > leaf_log || column_count == 0 || !mapped ||
        !mapped_count || max_queries == 0 || !walk || !walk_scratch || !walk_count ||
        !values || !retained || !sparse_indices || !sparse_hashes || !sparse_offsets ||
        !sparse_counts || !assembly || !stream) return cudaErrorInvalidValue;
    assemble_trace_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        tree_index, role, leaf_log, first_retained_log, column_count, mapped, mapped_count,
        max_queries, walk, walk_scratch, walk_count, values, retained, sparse_indices,
        sparse_hashes, sparse_offsets, sparse_counts, sparse_level_count, assembly, capacity);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_prepare_fri_queries_on(
    const uint32_t *unique, const uint32_t *unique_count, uint32_t max_queries,
    uint32_t cumulative_fold, uint32_t fold_step, uint32_t packed_log,
    uint32_t *tree_queries, uint32_t *tree_count, uint32_t *expanded,
    uint32_t *expanded_count, uint32_t *walk, uint32_t *walk_count, void *stream) {
    if (!unique || !unique_count || max_queries == 0 || fold_step >= 31 || packed_log >= 31 ||
        !tree_queries || !tree_count || !expanded || !expanded_count || !walk ||
        !walk_count || !stream) return cudaErrorInvalidValue;
    prepare_fri_queries_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        unique, unique_count, max_queries, cumulative_fold, fold_step, packed_log,
        tree_queries, tree_count, expanded, expanded_count, walk, walk_count);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_gather_fri_values_on(
    const uint32_t *const *coordinates, const uint32_t *positions,
    const uint32_t *count, uint32_t max_positions, uint32_t *values, void *stream) {
    if (!coordinates || !positions || !count || max_positions == 0 || !values || !stream)
        return cudaErrorInvalidValue;
    gather_fri_values_kernel<<<(max_positions + BLOCK - 1) / BLOCK, BLOCK, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(coordinates, positions, count,
        max_positions, values);
    return cudaGetLastError();
}

extern "C" int stwo_decommit_assemble_fri_on(
    uint32_t tree_index, uint32_t leaf_log, const uint32_t *tree_queries,
    const uint32_t *tree_count, const uint32_t *expanded, const uint32_t *expanded_count,
    const uint32_t *values, uint32_t *walk, uint32_t *walk_scratch,
    const uint32_t *walk_count, const Blake2sHash *const *retained, uint32_t *assembly,
    uint32_t capacity, void *stream) {
    if (leaf_log >= 31 || !tree_queries || !tree_count || !expanded || !expanded_count ||
        !values || !walk || !walk_scratch || !walk_count || !retained || !assembly || !stream)
        return cudaErrorInvalidValue;
    assemble_fri_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        tree_index, leaf_log, tree_queries, tree_count, expanded, expanded_count, values,
        walk, walk_scratch, walk_count, retained, assembly, capacity);
    return cudaGetLastError();
}
