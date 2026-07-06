#include "blake2s.cuh"
#include <cstdio>
#include "utils.cuh"

__device__ __constant__ uint32_t blake2s_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__device__ __constant__ uint8_t blake2s_sigma[10][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};


#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define G(r,i,a,b,c,d) \
    do { \
        a = a + b + m[blake2s_sigma[r][2*i+0]]; \
        d = ROTR32(d ^ a, 16); \
        c = c + d; \
        b = ROTR32(b ^ c, 12); \
        a = a + b + m[blake2s_sigma[r][2*i+1]]; \
        d = ROTR32(d ^ a, 8); \
        c = c + d; \
        b = ROTR32(b ^ c, 7); \
    } while(0)

typedef struct {
    uint32_t h[8];      // hash state
    uint32_t t;         // total bytes so far
    uint8_t  buf[64];   // buffer
    size_t   buflen;    // buffer usage
} Blake2sState;

__device__ void blake2s_compress(
    Blake2sState* S,
    const uint8_t block[64],
    uint32_t t,         // total bytes so far
    uint32_t lastblock  // 0 for normal, 0xFFFFFFFF for last block
) {
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        m[i] =  ((uint32_t)block[4*i+0]      ) |
                ((uint32_t)block[4*i+1] << 8 ) |
                ((uint32_t)block[4*i+2] << 16) |
                ((uint32_t)block[4*i+3] << 24);
    }

    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = S->h[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i+8] = blake2s_IV[i];

    v[12] ^= t;         // low 32 bits of offset
    v[13] ^= 0;         // high 32 bits (always 0 for <2^32 bytes)
    v[14] ^= lastblock; // 0xFFFFFFFF for last block

    // 10 rounds
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        G(r,0,v[0],v[4],v[8],v[12]);
        G(r,1,v[1],v[5],v[9],v[13]);
        G(r,2,v[2],v[6],v[10],v[14]);
        G(r,3,v[3],v[7],v[11],v[15]);
        G(r,4,v[0],v[5],v[10],v[15]);
        G(r,5,v[1],v[6],v[11],v[12]);
        G(r,6,v[2],v[7],v[8],v[13]);
        G(r,7,v[3],v[4],v[9],v[14]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++)
        S->h[i] ^= v[i] ^ v[i+8];
}


__device__ void blake2s_init(Blake2sState* S) {
    const uint32_t IV[8] = {
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };
    S->h[0] = IV[0] ^ 0x01010020; // digest len = 32
    S->h[1] = IV[1];
    S->h[2] = IV[2];
    S->h[3] = IV[3];
    S->h[4] = IV[4];
    S->h[5] = IV[5];
    S->h[6] = IV[6];
    S->h[7] = IV[7];
    S->t = 0;
    S->buflen = 0;
}
__device__ void blake2s_update(Blake2sState* S, const uint8_t* in, size_t inlen) {
    size_t left = S->buflen;
    size_t fill = 64 - left;

    if (inlen > fill) {
        memcpy(S->buf + left, in, fill);
        S->t += 64;
        blake2s_compress(S, S->buf, S->t, 0);
        in += fill;
        inlen -= fill;
        while (inlen > 64) {
            S->t += 64;
            blake2s_compress(S, in, S->t, 0);
            in += 64;
            inlen -= 64;
        }
        left = 0;
    }
    memcpy(S->buf + left, in, inlen);
    S->buflen = left + inlen;
}

__device__ void blake2s_finalize(Blake2sState* S, Blake2sHash* out) {
    S->t += S->buflen;
    memset(S->buf + S->buflen, 0, 64 - S->buflen); // pad
    blake2s_compress(S, S->buf, S->t, 0xFFFFFFFF); // lastblock = 0xFFFFFFFF
    for (int i = 0; i < 8; i++) {
        out->s[i] = S->h[i];
    }
}

// ---------------------------------------------------------------------------
// WORD-BLOCK blake2s (the commit-path fast lane). The tree hashes M31 words
// and 32-byte child digests — both are sequences of little-endian u32s, which
// ARE blake2s message words: the byte-buffered state machine above (64-byte
// local buffer, per-4-byte memcpy, byte->word re-decode every compress) is
// pure overhead for them. These helpers keep h[8] and m[16] in REGISTERS
// (callers must index m with compile-time constants — unrolled 16-word
// groups), producing BIT-IDENTICAL digests: same word stream, same byte
// counts, same lastblock flag.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void blake2s_compress_words(
    uint32_t h[8],
    const uint32_t m[16],
    uint32_t t,         // total bytes so far
    uint32_t lastblock  // 0 for normal, 0xFFFFFFFF for last block
) {
    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = h[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i+8] = blake2s_IV[i];

    v[12] ^= t;
    v[14] ^= lastblock;

    #pragma unroll
    for (int r = 0; r < 10; r++) {
        G(r,0,v[0],v[4],v[8],v[12]);
        G(r,1,v[1],v[5],v[9],v[13]);
        G(r,2,v[2],v[6],v[10],v[14]);
        G(r,3,v[3],v[7],v[11],v[15]);
        G(r,4,v[0],v[5],v[10],v[15]);
        G(r,5,v[1],v[6],v[11],v[12]);
        G(r,6,v[2],v[7],v[8],v[13]);
        G(r,7,v[3],v[4],v[9],v[14]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++)
        h[i] ^= v[i] ^ v[i+8];
}

__device__ __forceinline__ void blake2s_init_words(uint32_t h[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) h[i] = blake2s_IV[i];
    h[0] ^= 0x01010020; // digest len = 32, fanout/depth 1 — same as blake2s_init
}

// Hash a row's column words: unrolled 16-word groups (m stays in registers),
// zero-padded tail block with the exact byte count + lastblock flag.
__device__ __forceinline__ void blake2s_hash_column_words(
    uint32_t h[8],
    uint32_t t,                 // bytes already consumed (0, or 64 after children)
    uint32_t **data,
    uint32_t number_of_columns,
    uint32_t index,
    Blake2sHash *out
) {
    uint32_t m[16];
    uint32_t col = 0;
    // Lazy like blake2s_update (`inlen > fill`): a block is compressed with
    // last=0 only when more words follow it, so the block holding the final
    // word is the one flagged last — also when the stream is an exact
    // multiple of 16 words (rem == 16 below, never a zero-padded extra block).
    while (col + 16 < number_of_columns) {
        #pragma unroll
        for (int k = 0; k < 16; k++) m[k] = data[col + k][index];
        t += 64;
        blake2s_compress_words(h, m, t, 0);
        col += 16;
    }
    uint32_t rem = number_of_columns - col;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        m[k] = (k < rem) ? data[col + k][index] : 0;
    }
    t += 4 * rem;
    blake2s_compress_words(h, m, t, 0xFFFFFFFF);
    #pragma unroll
    for (int i = 0; i < 8; i++) out->s[i] = h[i];
}




__global__ void __launch_bounds__(BLOCK_SIZE) commit_on_first_layer_in_gpu(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    Blake2sHash *result
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    // Word-block lane: bit-identical digests, no byte buffer (see
    // blake2s_hash_column_words).
    uint32_t h[8];
    blake2s_init_words(h);
    blake2s_hash_column_words(h, 0, data, number_of_columns, index, &result[index]);
}

__device__ __forceinline__ uint32_t lifted_column_index(
    uint32_t lifted_index,
    uint32_t log_ratio
) {
    if (log_ratio == 0) {
        return lifted_index;
    }
    return ((lifted_index >> (log_ratio + 1)) << 1) + (lifted_index & 1);
}

__global__ void __launch_bounds__(BLOCK_SIZE) commit_on_first_layer_lifted_in_gpu(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    const uint32_t *column_log_sizes,
    uint32_t lifting_log_size,
    Blake2sHash *result
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    // Word-block lane with per-column lifted source indices.
    uint32_t h[8];
    blake2s_init_words(h);
    uint32_t m[16];
    uint32_t t = 0;
    uint32_t col = 0;
    // Lazy full-block loop — see blake2s_hash_column_words.
    while (col + 16 < number_of_columns) {
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            uint32_t log_ratio = lifting_log_size - column_log_sizes[col + k];
            m[k] = data[col + k][lifted_column_index(index, log_ratio)];
        }
        t += 64;
        blake2s_compress_words(h, m, t, 0);
        col += 16;
    }
    uint32_t rem = number_of_columns - col;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        if (k < rem) {
            uint32_t log_ratio = lifting_log_size - column_log_sizes[col + k];
            m[k] = data[col + k][lifted_column_index(index, log_ratio)];
        } else {
            m[k] = 0;
        }
    }
    t += 4 * rem;
    blake2s_compress_words(h, m, t, 0xFFFFFFFF);
    #pragma unroll
    for (int i = 0; i < 8; i++) result[index].s[i] = h[i];
}
__global__ void __launch_bounds__(BLOCK_SIZE) commit_on_layer_using_previous_in_gpu(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    Blake2sHash *prev_layer,
    Blake2sHash *result
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    // Word-block lane: children = exactly one full block (t = 64), then the
    // optional layer columns continue block-wise. Bit-identical digests.
    uint32_t h[8];
    blake2s_init_words(h);
    uint32_t m[16];
    Blake2sHash left = prev_layer[2*index];
    Blake2sHash right = prev_layer[2*index+1];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = left.s[i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[8 + i] = right.s[i];
    if (number_of_columns == 0) {
        blake2s_compress_words(h, m, 64, 0xFFFFFFFF);
        #pragma unroll
        for (int i = 0; i < 8; i++) result[index].s[i] = h[i];
        return;
    }
    blake2s_compress_words(h, m, 64, 0);
    blake2s_hash_column_words(h, 64, data, number_of_columns, index, &result[index]);
}

// ---------------------------------------------------------------------------
// Commit-path fusion (Workstream D) — layer-pair Merkle hashing.
//
// Additive: the kernels above are untouched. This hashes TWO internal tree levels
// per launch. For output index i it produces
//     out[i] = H( H(prev[4i], prev[4i+1]), H(prev[4i+2], prev[4i+3]) )
// which is *exactly* two sequential applications of
// `commit_on_layer_using_previous_in_gpu` with number_of_columns == 0 — i.e.
// byte-identical to the reference by construction, no hash-function or ordering
// change. It is only valid where BOTH fused levels inject zero columns (the common
// internal-tree case in the lifted Merkle tree); the caller must not use it across
// a level that injects columns. Gated OFF behind STWO_CUDA_FUSED_COMMIT +
// STWO_CUDA_FUSED_COMMIT_LAYER_PAIR (default OFF) until the pod gates pass.
// ---------------------------------------------------------------------------

// Hash a pair of child digests into one, matching the column-free byte stream of
// `commit_on_layer_using_previous_in_gpu` (left.s[0..8] then right.s[0..8], each
// word little-endian) exactly.
__device__ __forceinline__ Blake2sHash blake2s_hash_children_device(
    const Blake2sHash& left,
    const Blake2sHash& right
) {
    // Word-block lane: one full block (t = 64) with the lastblock flag.
    uint32_t h[8];
    blake2s_init_words(h);
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = left.s[i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[8 + i] = right.s[i];
    blake2s_compress_words(h, m, 64, 0xFFFFFFFF);
    Blake2sHash out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) out.s[i] = h[i];
    return out;
}

__global__ void __launch_bounds__(BLOCK_SIZE) commit_on_two_layers_using_previous_in_gpu(
    uint32_t size,                 // number of OUTPUT (grandparent) hashes
    Blake2sHash *prev_layer,       // size == 4 * `size`
    Blake2sHash *result
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    Blake2sHash left_mid =
        blake2s_hash_children_device(prev_layer[4 * index + 0], prev_layer[4 * index + 1]);
    Blake2sHash right_mid =
        blake2s_hash_children_device(prev_layer[4 * index + 2], prev_layer[4 * index + 3]);
    result[index] = blake2s_hash_children_device(left_mid, right_mid);
}

uint32_t number_of_blocks_for(uint32_t size) {
    return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}
void commit_on_first_layer(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **device_columns,
    Blake2sHash* result
) {
    commit_on_first_layer_in_gpu<<<number_of_blocks_for(size), BLOCK_SIZE>>>(
        size, number_of_columns, device_columns, result);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void commit_on_first_layer_lifted(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **device_columns,
    const uint32_t *column_log_sizes,
    uint32_t lifting_log_size,
    Blake2sHash* result
) {
    commit_on_first_layer_lifted_in_gpu<<<number_of_blocks_for(size), BLOCK_SIZE>>>(
        size, number_of_columns, device_columns, column_log_sizes, lifting_log_size, result);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void commit_on_layer_with_previous(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **device_columns,
    Blake2sHash* previous_layer,
    Blake2sHash* result
) {
    commit_on_layer_using_previous_in_gpu<<<number_of_blocks_for(size), BLOCK_SIZE>>>(
        size, number_of_columns, device_columns, previous_layer, result);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// Layer-pair fusion (Workstream D). `size` is the number of grandparent hashes to
// produce; `previous_layer` must hold `4 * size` child hashes. Column-free only.
void commit_on_two_layers_with_previous(
    uint32_t size,
    Blake2sHash* previous_layer,
    Blake2sHash* result
) {
    commit_on_two_layers_using_previous_in_gpu<<<number_of_blocks_for(size), BLOCK_SIZE>>>(
        size, previous_layer, result);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Merkle TAIL fusion (commit fusion, C2): the top K levels in ONE launch. The
// per-level launches this replaces are tiny (<= 4096 nodes) — their cost is
// the launch gap, not the hashing — so a single block with __syncthreads
// between levels replaces ~K launches. Levels write into caller-provided
// per-level buffers (the same layout the per-layer path produces, so tree
// readers are unchanged). Hashing delegates to blake2s_hash_children_device —
// the SAME routine the per-layer kernel uses: a scheduling change, not a new
// hash.
// 256 threads, bounded: the inlined children hash is register-fat and a
// 1024-thread block exceeds the SM register file (launch fails with
// out-of-resources on H100). Width is irrelevant here — the whole tail is
// <= 2^12 hashes and the per-level grid-stride loop covers any level width.
#define STWO_TAIL_BLOCK 256u
__global__ void __launch_bounds__(STWO_TAIL_BLOCK) blake2s_tail_kernel(
    const Blake2sHash *first,
    uint32_t first_size,
    Blake2sHash *const *out_levels,
    uint32_t n_levels
) {
    const Blake2sHash *prev = first;
    uint32_t size = first_size;
    for (uint32_t l = 0; l < n_levels; ++l) {
        uint32_t next = size / 2;
        for (uint32_t i = threadIdx.x; i < next; i += blockDim.x) {
            out_levels[l][i] =
                blake2s_hash_children_device(prev[2 * i], prev[2 * i + 1]);
        }
        __syncthreads();
        prev = out_levels[l];
        size = next;
    }
}

// Returns 0 on success. `out_levels_dev` is a DEVICE array of n_levels device
// pointers; level l holds first_size >> (l+1) hashes. first_size must be a
// power of two with first_size >> n_levels >= 1.
extern "C" int stwo_blake2s_tail(
    const Blake2sHash *first_dev,
    uint32_t first_size,
    Blake2sHash *const *out_levels_dev,
    uint32_t n_levels
) {
    if (n_levels == 0) {
        return 0;
    }
    if (first_size == 0 || (first_size & (first_size - 1)) != 0 ||
        (first_size >> n_levels) == 0) {
        fprintf(stderr, "stwo_blake2s_tail: bad sizes (first=%u levels=%u)\n",
                first_size, n_levels);
        return 1;
    }
    blake2s_tail_kernel<<<1, STWO_TAIL_BLOCK>>>(first_dev, first_size, out_levels_dev, n_levels);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "stwo_blake2s_tail: launch failed\n");
        return 1;
    }
    return 0;
}
