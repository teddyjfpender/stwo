// Opcode base-trace kernels for the witness-on-GPU cohort ports (round 12).
//
// Each kernel is the per-row device analogue of one generated SIMD writer's
// row math (transcribed u32-for-u32; every PackedM31 add/sub/mul becomes the
// fields.cuh modular op), with the memory deduce lookups fused as table
// gathers against the prove-wide device memory tables the backend uploads
// once per prove:
//   addr -> id:    the address-ordered raw id table (index addr-1)
//   id -> limbs:   tag 1 = F252 (8 raw words per value), tag 0 = Small
//                  (u128 = 4 raw words per value, upper words zero), split
//                  into 9-bit limbs exactly like memory_id_to_big's
//                  deduce_output.
// Trace columns and the staged tuple columns (lookup expressions that are not
// plain trace columns) feed the witness_logup.cu lane unchanged.

#include "fields.cuh"
#include "utils.cuh"

namespace {

constexpr uint32_t OP_BLOCK = 256;
constexpr uint32_t OP_LIMB_BITS = 9;
constexpr uint32_t OP_LIMB_MASK = (1u << OP_LIMB_BITS) - 1;

// Little-endian 9-bit limb split over N_WORDS words; line-for-line the
// generic `split` in stwo-cairo-common prover_types/felt.rs (and the copy in
// memory_witness.cu's memory_limb_split_kernel).
template <int N_WORDS, int N_LIMBS>
DEVICE_FORCEINLINE void op_split_le_9bit(const uint32_t *words, uint32_t *limbs) {
    uint32_t n_bits_in_word = 32;
    uint32_t word_i = 0;
    uint32_t word = words[0];
    for (int e = 0; e < N_LIMBS; ++e) {
        if (n_bits_in_word > OP_LIMB_BITS) {
            limbs[e] = word & OP_LIMB_MASK;
            word >>= OP_LIMB_BITS;
            n_bits_in_word -= OP_LIMB_BITS;
            continue;
        }
        limbs[e] = word;
        word_i += 1;
        word = word_i < N_WORDS ? words[word_i] : 0;
        if (n_bits_in_word < OP_LIMB_BITS) {
            limbs[e] |= (word << n_bits_in_word) & OP_LIMB_MASK;
            word >>= OP_LIMB_BITS - n_bits_in_word;
        }
        n_bits_in_word += 32 - OP_LIMB_BITS;
    }
}

// addr -> raw encoded id: AddressToId is address-ordered from address 1
// (host Index impl reads data[addr - 1]).
DEVICE_FORCEINLINE uint32_t mem_addr_to_id(const uint32_t *addr_table, uint32_t addr) {
    return addr_table[addr - 1];
}

// id -> first N_LIMBS 9-bit limbs, mirroring memory_id_to_big::deduce_output.
template <int N_LIMBS>
DEVICE_FORCEINLINE void mem_id_to_limbs(
    uint32_t id,
    const uint32_t *big_words,    // 8 words per value
    const uint32_t *small_words,  // 4 words per value
    uint32_t *limbs
) {
    uint32_t tag = id >> 30;
    uint32_t val = id & 0x3FFFFFFFu;
    uint32_t words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (tag == 1) {
        #pragma unroll
        for (int w = 0; w < 8; ++w) words[w] = big_words[(size_t)val * 8 + w];
    } else {
        #pragma unroll
        for (int w = 0; w < 4; ++w) words[w] = small_words[(size_t)val * 4 + w];
    }
    op_split_le_9bit<8, N_LIMBS>(words, limbs);
}

// ret_opcode (16 trace columns): two 29-bit memory reads at fp-1 / fp-2.
// Staged columns: the two read addresses (lookup tuple slots that are not
// trace columns) and the next_pc / next_fp limb recombinations of the
// `opcodes` yield tuple.
__global__ void ret_opcode_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                 // 16 trace columns
    uint32_t *addr0, uint32_t *addr1,       // staged: fp-1, fp-2
    uint32_t *next_pc, uint32_t *next_fp    // staged: limb recombinations
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Read Positive Num Bits 29 at fp - 1 (modular sub, like PackedM31).
    uint32_t a0 = sub(v_fp, 1u);
    uint32_t id0 = mem_addr_to_id(addr_table, a0);
    trace[3][row] = id0;
    uint32_t l0[4];
    mem_id_to_limbs<4>(id0, big_words, small_words, l0);
    trace[4][row] = l0[0];
    trace[5][row] = l0[1];
    trace[6][row] = l0[2];
    trace[7][row] = l0[3];
    trace[8][row] = (l0[3] & 2u) >> 1;  // partial_limb_msb (u16 bit math)

    // Read Positive Num Bits 29 at fp - 2.
    uint32_t a1 = sub(v_fp, 2u);
    uint32_t id1 = mem_addr_to_id(addr_table, a1);
    trace[9][row] = id1;
    uint32_t l1[4];
    mem_id_to_limbs<4>(id1, big_words, small_words, l1);
    trace[10][row] = l1[0];
    trace[11][row] = l1[1];
    trace[12][row] = l1[2];
    trace[13][row] = l1[3];
    trace[14][row] = (l1[3] & 2u) >> 1;

    trace[15][row] = row < n_rows ? 1u : 0u;  // enabler

    addr0[row] = a0;
    addr1[row] = a1;
    // next_pc = l0 + l1*2^9 + l2*2^18 + l3*2^27 (modular adds/muls, exactly
    // the host's PackedM31 expression).
    next_pc[row] = add(
        add(l0[0], mul(l0[1], 512u)),
        add(mul(l0[2], 262144u), mul(l0[3], 134217728u)));
    next_fp[row] = add(
        add(l1[0], mul(l1[1], 512u)),
        add(mul(l1[2], 262144u), mul(l1[3], 134217728u)));
}

}  // namespace

extern "C" void ret_opcode_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *addr0, uint32_t *addr1,
    uint32_t *next_pc, uint32_t *next_fp
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    ret_opcode_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace), addr0, addr1, next_pc, next_fp);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
