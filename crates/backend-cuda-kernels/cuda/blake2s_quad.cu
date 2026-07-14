#include "blake2s.cuh"

// Low-register streaming Blake2s leaf update. The scalar implementation keeps
// h[8], m[16], and v[16] in one thread and reaches the SM90 255-register
// ceiling. Here one four-lane subgroup owns one leaf. Lane q owns v[q],
// v[q+4], v[q+8], and v[q+12]; column G functions are local, while the
// diagonal half-round is a fixed shuffle permutation inside the quad.

namespace {

constexpr uint32_t kBlockThreads = 256;
constexpr uint32_t kQuadWidth = 4;
constexpr uint32_t kLeavesPerBlock = kBlockThreads / kQuadWidth;
#ifndef STWO_BLAKE2S_QUAD_MIN_BLOCKS
#define STWO_BLAKE2S_QUAD_MIN_BLOCKS 6
#endif
static_assert(kBlockThreads % 32 == 0, "quad block must contain complete warps");
static_assert(32 % kQuadWidth == 0, "quad width must partition a warp");
static_assert(STWO_BLAKE2S_QUAD_MIN_BLOCKS >= 1, "invalid occupancy target");

__device__ __constant__ uint32_t kIv[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__device__ __forceinline__ uint32_t rotate_right(uint32_t value, uint32_t bits) {
    return __funnelshift_r(value, value, bits);
}

__device__ __forceinline__ void mix(
    uint32_t &a,
    uint32_t &b,
    uint32_t &c,
    uint32_t &d,
    uint32_t first,
    uint32_t second) {
    a = a + b + first;
    d = rotate_right(d ^ a, 16);
    c += d;
    b = rotate_right(b ^ c, 12);
    a = a + b + second;
    d = rotate_right(d ^ a, 8);
    c += d;
    b = rotate_right(b ^ c, 7);
}

__device__ __forceinline__ uint32_t quad_shuffle(
    uint32_t mask, uint32_t value, uint32_t source_lane) {
    return __shfl_sync(mask, value, source_lane, kQuadWidth);
}

// The 16 message words are staged once in shared memory so every lane can
// consume the round permutation without retaining the full block in registers.
__device__ __forceinline__ void compress_quad(
    uint32_t mask,
    uint32_t quad_lane,
    uint32_t &h_low,
    uint32_t &h_high,
    const uint32_t *message,
    uint32_t total_bytes,
    uint32_t last_block) {
    const uint32_t original_low = h_low;
    const uint32_t original_high = h_high;
    uint32_t a = h_low;
    uint32_t b = h_high;
    uint32_t c = kIv[quad_lane];
    uint32_t d = kIv[quad_lane + 4];
    if (quad_lane == 0) d ^= total_bytes;  // v[12]
    if (quad_lane == 2) d ^= last_block;   // v[14]

    // Spell sigma indices as immediates. Dynamic constant-memory indexing
    // serializes the four distinct lane addresses; the select below becomes
    // ordinary predicates followed by one shared-memory message load.
#define QUAD_PICK(i0, i1, i2, i3)                                           \
    message[quad_lane == 0 ? (i0) : quad_lane == 1 ? (i1)                   \
                                  : quad_lane == 2 ? (i2) : (i3)]
#define QUAD_ROUND(s0, s1, s2, s3, s4, s5, s6, s7,                         \
                   s8, s9, s10, s11, s12, s13, s14, s15) do {              \
    mix(a, b, c, d, QUAD_PICK(s0, s2, s4, s6),                             \
         QUAD_PICK(s1, s3, s5, s7));                                        \
    uint32_t diagonal_b = quad_shuffle(mask, b, (quad_lane + 1) & 3u);      \
    uint32_t diagonal_c = quad_shuffle(mask, c, (quad_lane + 2) & 3u);      \
    uint32_t diagonal_d = quad_shuffle(mask, d, (quad_lane + 3) & 3u);      \
    mix(a, diagonal_b, diagonal_c, diagonal_d,                              \
         QUAD_PICK(s8, s10, s12, s14), QUAD_PICK(s9, s11, s13, s15));      \
    b = quad_shuffle(mask, diagonal_b, (quad_lane + 3) & 3u);               \
    c = quad_shuffle(mask, diagonal_c, (quad_lane + 2) & 3u);               \
    d = quad_shuffle(mask, diagonal_d, (quad_lane + 1) & 3u);               \
} while (0)

    QUAD_ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    QUAD_ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3);
    QUAD_ROUND(11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4);
    QUAD_ROUND(7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8);
    QUAD_ROUND(9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13);
    QUAD_ROUND(2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9);
    QUAD_ROUND(12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11);
    QUAD_ROUND(13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10);
    QUAD_ROUND(6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5);
    QUAD_ROUND(10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0);

#undef QUAD_ROUND
#undef QUAD_PICK

    h_low = original_low ^ a ^ c;
    h_high = original_high ^ b ^ d;
}

__device__ __forceinline__ uint32_t lifted_index(
    uint32_t leaf, uint32_t log_ratio) {
    if (log_ratio == 0) return leaf;
    return ((leaf >> (log_ratio + 1)) << 1) + (leaf & 1);
}

__global__ __launch_bounds__(kBlockThreads, STWO_BLAKE2S_QUAD_MIN_BLOCKS)
void stream_leaf_update_quad(
    uint32_t size,
    uint32_t group_columns,
    uint32_t **columns,
    const uint32_t *column_log_sizes,
    uint32_t lifting_log_size,
    uint32_t columns_done,
    Blake2sHash *states) {
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t leaf = thread / kQuadWidth;
    const uint32_t quad_lane = threadIdx.x & 3u;
    if (leaf >= size) return;

    const uint32_t lane_in_warp = threadIdx.x & 31u;
    const uint32_t mask = 0xFu << (lane_in_warp & ~3u);
    const uint32_t local_leaf = threadIdx.x / kQuadWidth;
    __shared__ uint32_t messages[kLeavesPerBlock][16];
    uint32_t h_low = states[leaf].s[quad_lane];
    uint32_t h_high = states[leaf].s[quad_lane + 4];
    uint32_t total_bytes = 4u * columns_done;

    for (uint32_t first = 0; first < group_columns; first += 16) {
        const uint32_t i0 = first + quad_lane;
        const uint32_t i1 = i0 + 4;
        const uint32_t i2 = i0 + 8;
        const uint32_t i3 = i0 + 12;
        const uint32_t r0 = lifting_log_size - column_log_sizes[i0];
        const uint32_t r1 = lifting_log_size - column_log_sizes[i1];
        const uint32_t r2 = lifting_log_size - column_log_sizes[i2];
        const uint32_t r3 = lifting_log_size - column_log_sizes[i3];
        const uint32_t m0 = columns[i0][lifted_index(leaf, r0)];
        const uint32_t m1 = columns[i1][lifted_index(leaf, r1)];
        const uint32_t m2 = columns[i2][lifted_index(leaf, r2)];
        const uint32_t m3 = columns[i3][lifted_index(leaf, r3)];
        messages[local_leaf][quad_lane] = m0;
        messages[local_leaf][quad_lane + 4] = m1;
        messages[local_leaf][quad_lane + 8] = m2;
        messages[local_leaf][quad_lane + 12] = m3;
        __syncwarp(mask);
        total_bytes += 64;
        compress_quad(mask, quad_lane, h_low, h_high, messages[local_leaf],
                      total_bytes, 0);
    }

    states[leaf].s[quad_lane] = h_low;
    states[leaf].s[quad_lane + 4] = h_high;
}

}  // namespace

extern "C" int stwo_blake2s_leaf_update_quad_on(
    uint32_t size,
    uint32_t group_columns,
    uint32_t **columns,
    const uint32_t *column_log_sizes,
    uint32_t lifting_log_size,
    uint32_t columns_done,
    Blake2sHash *states,
    void *stream) {
    if (size == 0 || group_columns == 0 || (group_columns % 16) != 0 ||
        columns == nullptr || column_log_sizes == nullptr ||
        lifting_log_size >= 31 || size != (1u << lifting_log_size) ||
        (columns_done % 16) != 0 || states == nullptr || stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    const uint32_t blocks = (size + kLeavesPerBlock - 1) / kLeavesPerBlock;
    stream_leaf_update_quad<<<blocks, kBlockThreads, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        size, group_columns, columns, column_log_sizes, lifting_log_size,
        columns_done, states);
    return cudaGetLastError();
}
