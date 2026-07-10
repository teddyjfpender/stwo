#ifndef STWO_RESIDENT_POW_H
#define STWO_RESIDENT_POW_H

#include <cstdint>

// Search the numeric u64 nonce space with one persistent kernel. The state is
// the 16-word device Blake2s transcript state; its first eight words are the
// current digest. `best_nonce` is initialized to UINT64_MAX and
// `completed_blocks` to zero by the prepared caller.
extern "C" int stwo_blake2s_pow_persistent_on(
    const uint32_t *transcript_state,
    uint32_t pow_bits,
    unsigned long long *best_nonce,
    uint32_t *completed_blocks,
    uint32_t *transcript_nonce,
    void *stream);

#endif  // STWO_RESIDENT_POW_H
