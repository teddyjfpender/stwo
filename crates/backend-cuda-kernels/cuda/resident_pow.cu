#include "resident_pow.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

#include "blake2s.cuh"

namespace {

constexpr uint32_t POW_PREFIX = 0x12345678U;
constexpr uint32_t POW_BLOCK_SIZE = 256U;
constexpr uint32_t POW_GRID_SIZE = 1024U;

// SIMD grind lattice (crates/stwo/src/prover/backend/simd/grind.rs). The
// reference scans nonces of the form (hi << 32) | low with 0 <= low < 2^20,
// hi ascending and low ascending within each hi, and returns the first hit.
// Because low < 2^20 < 2^32, that (hi, low) scan order IS numeric order on
// the lattice values, so the reference answer is exactly the numeric minimum
// of the qualifying lattice nonces. This kernel therefore enumerates a linear
// index i, maps it monotonically onto the lattice, and takes an atomicMin of
// the MAPPED nonce: the published minimum is byte-identical to SIMD's result.
constexpr uint32_t POW_GRIND_LOW_BITS = 20U;
constexpr unsigned long long POW_GRIND_LOW_MASK =
    (1ULL << POW_GRIND_LOW_BITS) - 1ULL;
// SIMD asserts the found hi is < 2^31 - 1 (the M31 prime). Mirror that bound
// as the give-up limit: indices span [0, P << 20) ~ 2^51, astronomically more
// than any realistic grind (pow_bits <= 32 needs ~2^32 attempts). On
// exhaustion best_nonce stays UINT64_MAX (not a lattice value, since lattice
// values have zero bits 20..31) and the published nonce fails verification
// downstream: fail-closed, never a silent wrap.
constexpr unsigned long long POW_INDEX_LIMIT =
    static_cast<unsigned long long>(0x7FFFFFFFU) << POW_GRIND_LOW_BITS;

// Monotone index -> lattice nonce map: i -> ((i >> 20) << 32) | (i & 0xFFFFF).
// Strictly increasing in i (the hi field takes the index's high bits, the low
// field its low 20 bits, and nonce bits 20..31 are always zero), so numeric
// comparisons on mapped nonces order exactly like the underlying indices.
__device__ __forceinline__ unsigned long long pow_index_to_nonce(
    unsigned long long index) {
  return ((index >> POW_GRIND_LOW_BITS) << 32U) | (index & POW_GRIND_LOW_MASK);
}

// Fixed 40-byte Blake2s candidate block, shared semantically with the legacy
// grind kernel but kept local so the persistent loop stays fully in registers.
static __device__ __constant__ uint32_t POW_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

static __device__ __constant__ uint8_t POW_SIGMA[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}};

#define POW_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define POW_G(r, i, a, b, c, d)                                  \
  do {                                                            \
    a = a + b + m[POW_SIGMA[r][2 * i + 0]];                      \
    d = POW_ROTR32(d ^ a, 16);                                    \
    c = c + d;                                                    \
    b = POW_ROTR32(b ^ c, 12);                                    \
    a = a + b + m[POW_SIGMA[r][2 * i + 1]];                      \
    d = POW_ROTR32(d ^ a, 8);                                     \
    c = c + d;                                                    \
    b = POW_ROTR32(b ^ c, 7);                                     \
  } while (0)

__device__ __forceinline__ uint32_t candidate_hash_word(
    const Blake2sHash &prefixed_digest,
    unsigned long long nonce) {
  uint32_t m[16];
#pragma unroll
  for (uint32_t i = 0; i < 8U; ++i) {
    m[i] = prefixed_digest.s[i];
  }
  m[8] = static_cast<uint32_t>(nonce);
  m[9] = static_cast<uint32_t>(nonce >> 32U);
#pragma unroll
  for (uint32_t i = 10U; i < 16U; ++i) {
    m[i] = 0U;
  }

  uint32_t v[16];
  v[0] = POW_IV[0] ^ 0x01010020U;
#pragma unroll
  for (uint32_t i = 1U; i < 8U; ++i) {
    v[i] = POW_IV[i];
  }
#pragma unroll
  for (uint32_t i = 0U; i < 8U; ++i) {
    v[i + 8U] = POW_IV[i];
  }
  v[12] ^= 40U;
  v[14] ^= 0xffffffffU;

#pragma unroll
  for (uint32_t round = 0U; round < 10U; ++round) {
    POW_G(round, 0, v[0], v[4], v[8], v[12]);
    POW_G(round, 1, v[1], v[5], v[9], v[13]);
    POW_G(round, 2, v[2], v[6], v[10], v[14]);
    POW_G(round, 3, v[3], v[7], v[11], v[15]);
    POW_G(round, 4, v[0], v[5], v[10], v[15]);
    POW_G(round, 5, v[1], v[6], v[11], v[12]);
    POW_G(round, 6, v[2], v[7], v[8], v[13]);
    POW_G(round, 7, v[3], v[4], v[9], v[14]);
  }
  return (POW_IV[0] ^ 0x01010020U) ^ v[0] ^ v[8];
}

__device__ __forceinline__ uint32_t trailing_zeros(uint32_t value) {
  return value == 0U ? 32U : static_cast<uint32_t>(__clz(__brev(value)));
}

__global__ void persistent_pow_search(
    const uint32_t *transcript_state,
    uint32_t pow_bits,
    unsigned long long *best_nonce,
    uint32_t *completed_blocks,
    uint32_t *transcript_nonce) {
  __shared__ Blake2sHash prefixed_digest;
  if (threadIdx.x == 0U) {
    uint8_t prefix_input[52] = {0};
    prefix_input[0] = static_cast<uint8_t>(POW_PREFIX);
    prefix_input[1] = static_cast<uint8_t>(POW_PREFIX >> 8U);
    prefix_input[2] = static_cast<uint8_t>(POW_PREFIX >> 16U);
    prefix_input[3] = static_cast<uint8_t>(POW_PREFIX >> 24U);
    const uint8_t *digest =
        reinterpret_cast<const uint8_t *>(transcript_state);
#pragma unroll
    for (uint32_t i = 0; i < 32U; ++i) {
      prefix_input[16U + i] = digest[i];
    }
    prefix_input[48] = static_cast<uint8_t>(pow_bits);
    prefix_input[49] = static_cast<uint8_t>(pow_bits >> 8U);
    prefix_input[50] = static_cast<uint8_t>(pow_bits >> 16U);
    prefix_input[51] = static_cast<uint8_t>(pow_bits >> 24U);
    stwo_blake2s_hash2_device(
        prefix_input, 52U, nullptr, 0U, &prefixed_digest);
  }
  __syncthreads();

  const unsigned long long worker =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  // Workers stride over the linear INDEX space; every hashed candidate is the
  // mapped lattice nonce, and best_nonce always holds a mapped value (or the
  // UINT64_MAX sentinel). Because the map is strictly monotone, comparing a
  // mapped candidate against best is equivalent to comparing indices, so the
  // early-exit reasoning below is unchanged from the pre-lattice kernel.
  // index + stride cannot overflow: index < POW_INDEX_LIMIT ~ 2^51 and
  // stride = gridDim.x * blockDim.x = 2^18.
  unsigned long long index = worker;
  while (index < POW_INDEX_LIMIT) {
    const unsigned long long candidate = pow_index_to_nonce(index);
    // An atomic read prevents a worker from terminating on an out-of-date
    // larger bound. A stale smaller bound is impossible because best only falls.
    const unsigned long long best = atomicAdd(best_nonce, 0ULL);
    if (candidate >= best) {
      break;
    }

    if (trailing_zeros(candidate_hash_word(prefixed_digest, candidate)) >=
        pow_bits) {
      atomicMin(best_nonce, candidate);
    }

    index += stride;
  }

  // Every worker exhausts its index residue class for all lattice nonces below
  // the observed minimum. The last block to retire therefore knows every
  // lattice candidate below the final atomic minimum was checked, and alone
  // publishes the transcript nonce -- the numerically smallest qualifying
  // lattice nonce, i.e. exactly the SIMD grind's answer.
  __syncthreads();
  if (threadIdx.x == 0U) {
    __threadfence();
    const uint32_t completed = atomicAdd(completed_blocks, 1U) + 1U;
    if (completed == gridDim.x) {
      const unsigned long long nonce = atomicAdd(best_nonce, 0ULL);
      transcript_nonce[0] = static_cast<uint32_t>(nonce);
      transcript_nonce[1] = static_cast<uint32_t>(nonce >> 32U);
      __threadfence();
    }
  }
}

}  // namespace

extern "C" int stwo_blake2s_pow_persistent_on(
    const uint32_t *transcript_state,
    uint32_t pow_bits,
    unsigned long long *best_nonce,
    uint32_t *completed_blocks,
    uint32_t *transcript_nonce,
    void *stream_raw) {
  if (transcript_state == nullptr || best_nonce == nullptr ||
      completed_blocks == nullptr || transcript_nonce == nullptr ||
      stream_raw == nullptr || pow_bits > 32U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  persistent_pow_search<<<POW_GRID_SIZE, POW_BLOCK_SIZE, 0,
                          reinterpret_cast<cudaStream_t>(stream_raw)>>>(
      transcript_state, pow_bits, best_nonce, completed_blocks,
      transcript_nonce);
  return static_cast<int>(cudaGetLastError());
}
