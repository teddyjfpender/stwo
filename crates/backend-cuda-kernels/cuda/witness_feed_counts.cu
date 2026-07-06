// Generalized multiplicity count feed for the witness-JIT lane (the device
// component DAG, B2): consumes a witness kernel's WORD-MAJOR sub-input buffer
// directly on device (`sub[w * column_length + row]`) — deleting the sub D2H,
// the host packed-vector rebuild, and the DashMap `add_input` storm for every
// COUNT-STYLE relation (range checks, xor tables, points-table mults).
//
// Per descriptor: key = fold of the tuple's words (key = (key << bits[i]) |
// word_i), optionally mapped through the consumer's input->row LUT (the actual
// preprocessed layout, uploaded from the host generator — never a closed form),
// then atomicAdd into counts[rel_index * table_size + row]. Equivalence to the
// host path: the same multiset of increments the host `add_input` calls
// produce, merged commutatively — the certified `blake_g_xor_count_kernel`
// argument, generalized.
//
// Descriptor ABI is FLAT u32 (stride WFC_DESC_STRIDE), no structs across FFI:
//   [0] word_base   [1] n_words (1..=WFC_MAX_WORDS)
//   [2..7] bits[5]  [7] rel_index
//   [8] table_size  [9] lut_index (WFC_NO_LUT = identity)
//   [10] counts_index
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cuda_mem_pool.cuh"

#define WFC_DESC_STRIDE 14u
#define WFC_MAX_WORDS 5u
#define WFC_NO_LUT 0xFFFFFFFFu

__global__ void witness_feed_counts_kernel(
    const uint32_t *sub_words,
    uint32_t column_length,
    const uint32_t *descs,
    uint32_t n_descs,
    const uint32_t *const *luts,
    uint32_t *const *counts
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }
    for (uint32_t d = 0; d < n_descs; ++d) {
        const uint32_t *e = descs + (size_t)d * WFC_DESC_STRIDE;
        uint32_t word_base = e[0];
        uint32_t n_words = e[1];
        uint32_t rel_index = e[7];
        uint32_t table_size = e[8];
        uint32_t lut_index = e[9];
        uint32_t kind = e[11];

        if (kind == 1u) {
            // MEM-ID DECODE (memory_id_to_big): tag = id >> 30 (0 = small,
            // 1 = big), val = id & 0x3FFFFFFF; DEFAULT_ID (empty cell) skipped
            // defensively — the host feed panics on it, a valid trace never
            // produces one. e[8] = big table size, e[12] = small table size,
            // e[10] = big counts slot, e[13] = small counts slot.
            uint32_t v = sub_words[(size_t)word_base * column_length + row];
            if (v == ((1u << 30) - 1u)) {
                continue;
            }
            uint32_t tag = v >> 30;
            uint32_t val = v & 0x3FFFFFFFu;
            if (tag == 1u) {
                if (val < table_size) {
                    atomicAdd(&counts[e[10]][(size_t)rel_index * table_size + val], 1u);
                }
            } else if (tag == 0u) {
                uint32_t small_size = e[12];
                if (val < small_size) {
                    atomicAdd(&counts[e[13]][(size_t)rel_index * small_size + val], 1u);
                }
            }
            continue;
        }

        // FOLD (+ optional signed key offset e[12], e.g. addr - 1; + optional LUT).
        uint32_t key = 0;
        for (uint32_t i = 0; i < n_words; ++i) {
            key = (key << e[2 + i]) |
                  sub_words[(size_t)(word_base + i) * column_length + row];
        }
        long long keyed = (long long)key + (long long)(int32_t)e[12];
        // Memory safety FIRST: the key must be in the LUT/table domain BEFORE
        // any dereference (an out-of-width tuple — impossible on a valid trace,
        // where the host feed would panic — must never become an OOB read).
        // LUT domains equal table_size for every registered family (the LUT
        // covers the full tuple space).
        if (keyed < 0 || (uint64_t)keyed >= table_size) {
            continue;
        }
        uint32_t k = (uint32_t)keyed;
        uint32_t idx = (lut_index == WFC_NO_LUT) ? k : luts[lut_index][k];
        if (idx < table_size) {
            atomicAdd(&counts[e[10]][(size_t)rel_index * table_size + idx], 1u);
        }
    }
}

// Launch over every padded row (the host feeds padding rows too — mults_0 = 1
// everywhere; truncating at the real count is an invalid-proof bug). All
// pointer arrays are DEVICE arrays built by the caller. Returns 0 on success.
extern "C" int stwo_witness_feed_counts(
    const uint32_t *sub_words_dev,
    uint32_t column_length,
    const uint32_t *descs_dev,
    uint32_t n_descs,
    const uint32_t *const *luts_dev,
    uint32_t *const *counts_dev
) {
    if (n_descs == 0 || column_length == 0) {
        return 0;
    }
    const uint32_t block = 256;
    uint32_t grid = (column_length + block - 1) / block;
    witness_feed_counts_kernel<<<grid, block>>>(
        sub_words_dev, column_length, descs_dev, n_descs, luts_dev, counts_dev);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "stwo_witness_feed_counts: launch failed\n");
        return 1;
    }
    return 0;
}
