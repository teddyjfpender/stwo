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

// jnz_opcode_taken (47 trace columns): decodes the jnz instruction, reads the
// dst felt (full 28-limb f252) at the fp/ap-based address, and reads the
// small-signed next_pc immediate at pc+1. Staged columns are every lookup
// tuple slot that is not a plain trace column (mirroring the host writer's
// LookupData arrays element-for-element).
__global__ void jnz_opcode_taken_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                 // 47 trace columns
    uint32_t *vi_off1, uint32_t *vi_off2,   // staged: verify_instruction tuple exprs
    uint32_t *addr_dst,                     // staged: mem_address_to_id_1 read addr
    uint32_t *next_pc_addr,                 // staged: mem_address_to_id_3 read addr (pc+1)
    uint32_t *m4_s4, uint32_t *m4_s5,       // staged: memory_id_to_big_4 slots
    uint32_t *m4_s22, uint32_t *m4_s28,     // staged: memory_id_to_big_4 slots
    uint32_t *next_pc_out, uint32_t *next_ap_out // staged: opcodes-out yield exprs
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: read the instruction felt at pc and split limbs.
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);

    // offset0 = il[0] + (il[1] & 127) << 9  (PackedUInt16, plain u32 ops).
    uint32_t offset0 = il[0] + (((il[1] & 127u) << 9));
    trace[3][row] = offset0;
    // flags from limb 5 >> 3 + limb 6 << 6.
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);
    uint32_t dst_base_fp = (flags >> 0) & 1u;
    trace[4][row] = dst_base_fp;
    uint32_t ap_update_add_1 = (flags >> 11) & 1u;
    trace[5][row] = ap_update_add_1;

    // verify_instruction tuple staged exprs (M31 modular ops).
    vi_off1[row] = add(add(mul(dst_base_fp, 8u), 16u), 32u);
    vi_off2[row] = add(8u, mul(ap_update_add_1, 32u));

    // decode_instruction_ad440.0[0] = offset0 - 32768 (M31).
    uint32_t dec_off0 = sub(offset0, 32768u);

    // mem_dst_base = dst_base_fp*fp + (1 - dst_base_fp)*ap.
    uint32_t mem_dst_base = add(mul(dst_base_fp, v_fp),
                               mul(sub(1u, dst_base_fp), v_ap));
    trace[6][row] = mem_dst_base;

    // Read Id at mem_dst_base + dec_off0.
    uint32_t dst_addr = add(mem_dst_base, dec_off0);
    addr_dst[row] = dst_addr;
    uint32_t dst_id = mem_addr_to_id(addr_table, dst_addr);
    trace[7][row] = dst_id;

    // Read full 28-limb f252 value of dst_id into trace cols 8..35.
    uint32_t dl[28];
    mem_id_to_limbs<28>(dst_id, big_words, small_words, dl);
    #pragma unroll
    for (int i = 0; i < 28; ++i) trace[8 + i][row] = dl[i];

    // dst_sum_inv = inverse of full limb sum. dst_sum_squares_inv = inverse of
    // (sum without limbs 0,21,27 plus squared diffs-from-p). The host writes the
    // inverses to cols 36/37 but they are NOT lookup tuple slots; the device
    // backend recomputes them only if needed. Here we replicate the values via
    // batch inverse on host? No — they are trace columns, so compute on device.
    // dst_sum_p_zero = sum of limbs 1..20,22..26 (excludes 0,21,27).
    uint32_t sum_pz = 0u;
    sum_pz = add(sum_pz, dl[1]);
    #pragma unroll
    for (int i = 2; i <= 20; ++i) sum_pz = add(sum_pz, dl[i]);
    sum_pz = add(sum_pz, dl[22]);
    sum_pz = add(sum_pz, dl[23]);
    sum_pz = add(sum_pz, dl[24]);
    sum_pz = add(sum_pz, dl[25]);
    sum_pz = add(sum_pz, dl[26]);
    uint32_t full_sum = add(sum_pz, add(add(dl[0], dl[21]), dl[27]));
    trace[36][row] = inv(full_sum);
    uint32_t d12 = sub(dl[0], 1u);
    uint32_t d13 = sub(dl[21], 136u);
    uint32_t d14 = sub(dl[27], 256u);
    uint32_t squares = add(add(mul(d12, d12), mul(d13, d13)), mul(d14, d14));
    trace[37][row] = inv(add(sum_pz, squares));

    // Read Small at pc + 1.
    uint32_t npc_addr = add(v_pc, 1u);
    next_pc_addr[row] = npc_addr;
    uint32_t next_pc_id = mem_addr_to_id(addr_table, npc_addr);
    trace[38][row] = next_pc_id;
    uint32_t nl[28];
    mem_id_to_limbs<28>(next_pc_id, big_words, small_words, nl);

    // Decode Small Sign.
    uint32_t msb = (nl[27] == 256u) ? 1u : 0u;
    trace[39][row] = msb;
    uint32_t mid_limbs_set = ((nl[20] == 511u) && (msb != 0u)) ? 1u : 0u;
    trace[40][row] = mid_limbs_set;
    // decode_small_sign outputs (M31 ops).
    uint32_t dss2 = mul(mid_limbs_set, 508u);
    uint32_t dss3 = mul(mid_limbs_set, 511u);
    uint32_t dss4 = sub(mul(msb, 136u), mid_limbs_set);
    uint32_t dss5 = mul(msb, 256u);

    uint32_t npc0 = nl[0];
    trace[41][row] = npc0;
    uint32_t npc1 = nl[1];
    trace[42][row] = npc1;
    uint32_t npc2 = nl[2];
    trace[43][row] = npc2;
    uint32_t remainder_bits = nl[3] & 3u;  // PackedUInt16
    trace[44][row] = remainder_bits;
    uint32_t partial_limb_msb = (remainder_bits & 2u) >> 1;
    trace[45][row] = partial_limb_msb;

    // memory_id_to_big_4 staged slots.
    m4_s4[row] = add(remainder_bits, dss2);
    m4_s5[row] = dss3;
    m4_s22[row] = dss4;
    m4_s28[row] = dss5;

    trace[46][row] = row < n_rows ? 1u : 0u;  // enabler

    // read_small_output.0 = ((((npc0 + npc1*512) + npc2*262144)
    //   + remainder_bits*134217728) - msb) - 536870912*mid_limbs_set  (M31).
    uint32_t read_small =
        sub(sub(add(add(add(npc0, mul(npc1, 512u)), mul(npc2, 262144u)),
                    mul(remainder_bits, 134217728u)),
                msb),
            mul(536870912u, mid_limbs_set));
    // opcodes-out yield staged exprs.
    next_pc_out[row] = add(v_pc, read_small);
    next_ap_out[row] = add(v_ap, ap_update_add_1);
}

// add_ap_opcode (17 trace columns): decodes the add_ap instruction at pc, reads
// the small-signed operand (op1, which may be the immediate at pc, or an
// fp/ap-based cell), forms next_ap = ap + read_small, and range-checks next_ap
// (RangeCheck29 = bottom 11 bits via range_check_11 + the remaining high bits
// scaled by 2^20 via range_check_18). Staged columns are every lookup tuple
// slot that is not a plain trace column (mirroring the host writer's LookupData
// arrays element-for-element):
//   [0] vi_felt   = (24 + op1_imm*32) + op1_base_fp*64 + op1_base_ap*128
//                                                   (verify_instruction slot)
//   [1] op1_addr  = mem1_base + (offset2 - 32768)   (mem_address_to_id_1 addr)
//   [2] m_s5      = remainder_bits + mid_limbs_set*508  (memory_id_to_big_2)
//   [3] m_s6      = mid_limbs_set*511
//   [4] m_s22     = msb*136 - mid_limbs_set
//   [5] m_s28     = msb*256
//   [6] rc18_val  = (next_ap - rc29_bot11bits) * 1048576  (range_check_18 slot)
//   [7] next_pc   = pc + (1 + op1_imm)              (opcodes-out yield)
//   [8] next_ap   = ap + read_small                 (opcodes-out yield)
// All PackedM31 expressions use the modular fields.cuh ops; the instruction
// bit-extractions, the limb (op_split_le_9bit) values, remainder_bits,
// partial_limb_msb and the range_check_29 bottom-11-bit split are PackedUInt16 /
// PackedUInt32 (plain u32). range_check_11's lookup value is the trace column
// rc29_bot11bits itself (col 15), so it is not staged.
__global__ void add_ap_opcode_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                 // 17 trace columns
    uint32_t *vi_felt,                      // staged: verify_instruction tuple expr
    uint32_t *op1_addr_out,                 // staged: mem_address_to_id_1 read addr
    uint32_t *m_s5, uint32_t *m_s6,         // staged: memory_id_to_big_2 slots
    uint32_t *m_s22, uint32_t *m_s28,       // staged: memory_id_to_big_2 slots
    uint32_t *rc18_val,                     // staged: range_check_18 tuple value
    uint32_t *next_pc_out, uint32_t *next_ap_out // staged: opcodes-out yield exprs
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: read the instruction felt at pc and split limbs.
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);

    // offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7) << 13)  (PackedUInt16).
    uint32_t offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7u) << 13);
    trace[3][row] = offset2;
    // flags from limb 5 >> 3 + limb 6 << 6.
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);
    uint32_t op1_imm = (flags >> 2) & 1u;
    trace[4][row] = op1_imm;
    uint32_t op1_base_fp = (flags >> 3) & 1u;
    trace[5][row] = op1_base_fp;
    // op1_base_ap = (1 - op1_imm) - op1_base_fp  (M31).
    uint32_t op1_base_ap = sub(sub(1u, op1_imm), op1_base_fp);

    // verify_instruction tuple staged expr (M31 modular ops):
    // ((24 + op1_imm*32) + op1_base_fp*64) + op1_base_ap*128.
    vi_felt[row] = add(add(add(24u, mul(op1_imm, 32u)), mul(op1_base_fp, 64u)),
                       mul(op1_base_ap, 128u));

    // mem1_base = op1_imm*pc + op1_base_fp*fp + op1_base_ap*ap  (M31).
    uint32_t mem1_base = add(add(mul(op1_imm, v_pc), mul(op1_base_fp, v_fp)),
                             mul(op1_base_ap, v_ap));
    trace[6][row] = mem1_base;

    // Read Small: Read Id at mem1_base + (offset2 - 32768).
    uint32_t op1_addr = add(mem1_base, sub(offset2, 32768u));
    op1_addr_out[row] = op1_addr;
    uint32_t op1_id = mem_addr_to_id(addr_table, op1_addr);
    trace[7][row] = op1_id;
    uint32_t ol[28];
    mem_id_to_limbs<28>(op1_id, big_words, small_words, ol);

    // Decode Small Sign.
    uint32_t msb = (ol[27] == 256u) ? 1u : 0u;
    trace[8][row] = msb;
    uint32_t mid_limbs_set = ((ol[20] == 511u) && (msb != 0u)) ? 1u : 0u;
    trace[9][row] = mid_limbs_set;
    // decode_small_sign outputs (M31 ops).
    uint32_t dss2 = mul(mid_limbs_set, 508u);
    uint32_t dss3 = mul(mid_limbs_set, 511u);
    uint32_t dss4 = sub(mul(msb, 136u), mid_limbs_set);
    uint32_t dss5 = mul(msb, 256u);

    uint32_t op1_limb_0 = ol[0];
    trace[10][row] = op1_limb_0;
    uint32_t op1_limb_1 = ol[1];
    trace[11][row] = op1_limb_1;
    uint32_t op1_limb_2 = ol[2];
    trace[12][row] = op1_limb_2;
    uint32_t remainder_bits = ol[3] & 3u;  // PackedUInt16
    trace[13][row] = remainder_bits;
    uint32_t partial_limb_msb = (remainder_bits & 2u) >> 1;  // PackedUInt16
    trace[14][row] = partial_limb_msb;

    // memory_id_to_big_2 staged sign slots.
    m_s5[row] = add(remainder_bits, dss2);
    m_s6[row] = dss3;
    m_s22[row] = dss4;
    m_s28[row] = dss5;

    // read_small_output.0 = ((((op1_limb_0 + op1_limb_1*512) + op1_limb_2*262144)
    //   + remainder_bits*134217728) - msb) - 536870912*mid_limbs_set  (M31).
    uint32_t read_small =
        sub(sub(add(add(add(op1_limb_0, mul(op1_limb_1, 512u)),
                        mul(op1_limb_2, 262144u)),
                    mul(remainder_bits, 134217728u)),
                msb),
            mul(536870912u, mid_limbs_set));

    // next_ap = ap + read_small  (M31).
    uint32_t next_ap = add(v_ap, read_small);

    // Range Check 29: bottom 11 bits of next_ap (PackedUInt32, plain u32 mask).
    uint32_t rc29_bot11bits = next_ap & 2047u;
    trace[15][row] = rc29_bot11bits;
    // range_check_18 tuple value = (next_ap - rc29_bot11bits) * 1048576  (M31).
    rc18_val[row] = mul(sub(next_ap, rc29_bot11bits), 1048576u);

    trace[16][row] = row < n_rows ? 1u : 0u;  // enabler

    // opcodes-out yield staged exprs (M31).
    next_pc_out[row] = add(v_pc, add(1u, op1_imm));
    next_ap_out[row] = next_ap;
}

// add_opcode_small (39 trace columns): decodes the instruction at pc (a fused
// pc->id->limbs gather, limbs NOT staged — only the derived offsets/flags are
// trace columns), then three "Read Small" memory reads (dst/op0/op1) each
// producing an id trace column and a small-sign decode. The staged columns are
// the lookup-tuple expressions that are not plain trace columns:
//   [0]  vi_felt5  = dst_base_fp*8 + op0_base_fp*16 + op1_imm*32
//                     + op1_base_fp*64 + op1_base_ap*128 + 256
//   [1]  vi_felt6  = ap_update_add_1*32 + 256
//   [2]  next_pc   = pc + 1 + op1_imm           (opcodes-out yield)
//   [3]  next_ap   = ap + ap_update_add_1       (opcodes-out yield)
//   per read r in {dst, op0, op1} at staged base 4 + 5*r:
//     [+0] addr  = mem_base + (offset_signed)   (offset - 32768, modular)
//     [+1] s5    = remainder_bits + mid_limbs_set*508
//     [+2] s6    = mid_limbs_set*511
//     [+3] s23   = msb*136 - mid_limbs_set
//     [+4] s29   = msb*256
// All PackedM31 expressions use the modular fields.cuh ops; the instruction
// bit-extractions are PackedUInt16 (plain u32).

// One "Read Small" decode: gathers id->limbs (up to limb 27), writes the seven
// trace columns starting at trace[col_base] and the four staged sign columns
// starting at staged[staged_base] (slots +1..+4; +0 holds the read address,
// written by the caller).
DEVICE_FORCEINLINE void add_small_read(
    uint32_t id, uint32_t row,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t *const *trace, uint32_t col_base,
    uint32_t *const *staged, uint32_t staged_base
) {
    uint32_t l[28];
    mem_id_to_limbs<28>(id, big_words, small_words, l);
    // msb = (limb27 == 256), mid_limbs_set = (limb20 == 511) & msb (M31 eq -> 0/1).
    uint32_t msb = (l[27] == 256u) ? 1u : 0u;
    uint32_t mid_limbs_set = ((l[20] == 511u) ? 1u : 0u) & msb;
    uint32_t limb0 = l[0];
    uint32_t limb1 = l[1];
    uint32_t limb2 = l[2];
    uint32_t remainder_bits = l[3] & 3u;          // limb3 & 3 (u16 bit math)
    uint32_t partial_limb_msb = (remainder_bits & 2u) >> 1;
    trace[col_base + 0][row] = msb;
    trace[col_base + 1][row] = mid_limbs_set;
    trace[col_base + 2][row] = limb0;
    trace[col_base + 3][row] = limb1;
    trace[col_base + 4][row] = limb2;
    trace[col_base + 5][row] = remainder_bits;
    trace[col_base + 6][row] = partial_limb_msb;
    // Staged sign columns (modular M31 ops).
    staged[staged_base + 1][row] = add(remainder_bits, mul(mid_limbs_set, 508u));
    staged[staged_base + 2][row] = mul(mid_limbs_set, 511u);
    staged[staged_base + 3][row] = sub(mul(msb, 136u), mid_limbs_set);
    staged[staged_base + 4][row] = mul(msb, 256u);
}

__global__ void add_opcode_small_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,    // 39 trace columns
    uint32_t *const *staged    // 19 staged columns (4 + 5*3)
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);
    uint32_t offset0 = il[0] + ((il[1] & 127u) << 9);
    uint32_t offset1 = (il[1] >> 7) + (il[2] << 2) + ((il[3] & 31u) << 11);
    uint32_t offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7u) << 13);
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);  // shared (limb5>>3)+(limb6<<6)
    uint32_t dst_base_fp = (flags >> 0) & 1u;
    uint32_t op0_base_fp = (flags >> 1) & 1u;
    uint32_t op1_imm = (flags >> 2) & 1u;
    uint32_t op1_base_fp = (flags >> 3) & 1u;
    uint32_t ap_update_add_1 = (flags >> 11) & 1u;
    trace[3][row] = offset0;
    trace[4][row] = offset1;
    trace[5][row] = offset2;
    trace[6][row] = dst_base_fp;
    trace[7][row] = op0_base_fp;
    trace[8][row] = op1_imm;
    trace[9][row] = op1_base_fp;
    trace[10][row] = ap_update_add_1;
    // op1_base_ap = 1 - op1_imm - op1_base_fp (modular M31).
    uint32_t op1_base_ap = sub(sub(1u, op1_imm), op1_base_fp);

    // Memory bases (modular M31): base_fp ? fp : ap, etc.
    uint32_t mem_dst_base = add(mul(dst_base_fp, v_fp), mul(sub(1u, dst_base_fp), v_ap));
    uint32_t mem0_base = add(mul(op0_base_fp, v_fp), mul(sub(1u, op0_base_fp), v_ap));
    uint32_t mem1_base = add(add(mul(op1_imm, v_pc), mul(op1_base_fp, v_fp)),
                             mul(op1_base_ap, v_ap));
    trace[11][row] = mem_dst_base;
    trace[12][row] = mem0_base;
    trace[13][row] = mem1_base;

    // Signed offsets: offset - 32768 (modular M31).
    uint32_t off0_signed = sub(offset0, 32768u);
    uint32_t off1_signed = sub(offset1, 32768u);
    uint32_t off2_signed = sub(offset2, 32768u);

    // dst read at mem_dst_base + off0_signed.
    uint32_t dst_addr = add(mem_dst_base, off0_signed);
    uint32_t dst_id = mem_addr_to_id(addr_table, dst_addr);
    trace[14][row] = dst_id;
    add_small_read(dst_id, row, big_words, small_words, trace, 15, staged, 4);
    staged[4][row] = dst_addr;

    // op0 read at mem0_base + off1_signed.
    uint32_t op0_addr = add(mem0_base, off1_signed);
    uint32_t op0_id = mem_addr_to_id(addr_table, op0_addr);
    trace[22][row] = op0_id;
    add_small_read(op0_id, row, big_words, small_words, trace, 23, staged, 9);
    staged[9][row] = op0_addr;

    // op1 read at mem1_base + off2_signed.
    uint32_t op1_addr = add(mem1_base, off2_signed);
    uint32_t op1_id = mem_addr_to_id(addr_table, op1_addr);
    trace[30][row] = op1_id;
    add_small_read(op1_id, row, big_words, small_words, trace, 31, staged, 14);
    staged[14][row] = op1_addr;

    trace[38][row] = row < n_rows ? 1u : 0u;  // enabler

    // verify_instruction staged felts (modular M31).
    staged[0][row] = add(
        add(add(mul(dst_base_fp, 8u), mul(op0_base_fp, 16u)),
            add(mul(op1_imm, 32u), mul(op1_base_fp, 64u))),
        add(mul(op1_base_ap, 128u), 256u));
    staged[1][row] = add(mul(ap_update_add_1, 32u), 256u);
    // opcodes-out yield staged: next_pc = pc + 1 + op1_imm, next_ap = ap + ap_update_add_1.
    staged[2][row] = add(add(v_pc, 1u), op1_imm);
    staged[3][row] = add(v_ap, ap_update_add_1);
}

// assert_eq_opcode (12 trace columns, the smallest opcode): decodes the
// instruction at pc, computes the two memory bases (dst / op1), and performs
// the "Mem Verify Equal" read of dst at mem_dst_base + (offset0 - 32768). The
// op1 read (memory_address_to_id_2) shares the same id (dst_id) but a different
// address (mem1_base + (offset2 - 32768)) — that's the equality being asserted.
// Staged columns (lookup tuple slots that are not plain trace columns):
//   [0] vi_felt5  = ((dst_base_fp*8 + 16) + op1_base_fp*64) + (1-op1_base_fp)*128
//   [1] vi_felt6  = ap_update_add_1*32 + 256
//   [2] dst_addr  = mem_dst_base + (offset0 - 32768)   (mem_address_to_id_1 addr)
//   [3] op1_addr  = mem1_base    + (offset2 - 32768)   (mem_address_to_id_2 addr)
//   [4] next_pc   = pc + 1                              (opcodes-out yield)
//   [5] next_ap   = ap + ap_update_add_1               (opcodes-out yield)
// PackedM31 expressions use the modular fields.cuh ops; the instruction
// bit-extractions are PackedUInt16 (plain u32).
__global__ void assert_eq_opcode_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                 // 12 trace columns
    uint32_t *vi_felt5, uint32_t *vi_felt6, // staged: verify_instruction exprs
    uint32_t *dst_addr_out,                 // staged: mem_address_to_id_1 read addr
    uint32_t *op1_addr_out,                 // staged: mem_address_to_id_2 read addr
    uint32_t *next_pc_out, uint32_t *next_ap_out // staged: opcodes-out yield exprs
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: read the instruction felt at pc and split limbs.
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);

    // offset0 = il[0] + ((il[1] & 127) << 9)  (PackedUInt16, plain u32 ops).
    uint32_t offset0 = il[0] + (((il[1] & 127u) << 9));
    trace[3][row] = offset0;
    // offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7) << 13).
    uint32_t offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7u) << 13);
    trace[4][row] = offset2;
    // flags from limb 5 >> 3 + limb 6 << 6.
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);
    uint32_t dst_base_fp = (flags >> 0) & 1u;
    trace[5][row] = dst_base_fp;
    uint32_t op1_base_fp = (flags >> 3) & 1u;
    trace[6][row] = op1_base_fp;
    uint32_t ap_update_add_1 = (flags >> 11) & 1u;
    trace[7][row] = ap_update_add_1;

    // verify_instruction tuple staged exprs (M31 modular ops).
    // vi_felt5 = (((dst_base_fp*8) + 16) + op1_base_fp*64) + (1-op1_base_fp)*128.
    vi_felt5[row] = add(
        add(add(mul(dst_base_fp, 8u), 16u), mul(op1_base_fp, 64u)),
        mul(sub(1u, op1_base_fp), 128u));
    // vi_felt6 = ap_update_add_1*32 + 256.
    vi_felt6[row] = add(mul(ap_update_add_1, 32u), 256u);

    // Signed offsets: offset - 32768 (modular M31).
    uint32_t off0_signed = sub(offset0, 32768u);
    uint32_t off2_signed = sub(offset2, 32768u);

    // mem_dst_base = dst_base_fp*fp + (1 - dst_base_fp)*ap.
    uint32_t mem_dst_base = add(mul(dst_base_fp, v_fp),
                               mul(sub(1u, dst_base_fp), v_ap));
    trace[8][row] = mem_dst_base;
    // mem1_base = op1_base_fp*fp + (1 - op1_base_fp)*ap.
    uint32_t mem1_base = add(mul(op1_base_fp, v_fp),
                            mul(sub(1u, op1_base_fp), v_ap));
    trace[9][row] = mem1_base;

    // Mem Verify Equal — Read Id at mem_dst_base + off0_signed.
    uint32_t dst_addr = add(mem_dst_base, off0_signed);
    dst_addr_out[row] = dst_addr;
    uint32_t dst_id = mem_addr_to_id(addr_table, dst_addr);
    trace[10][row] = dst_id;

    // memory_address_to_id_2 reads the same id (dst_id) at op1_addr.
    uint32_t op1_addr = add(mem1_base, off2_signed);
    op1_addr_out[row] = op1_addr;

    trace[11][row] = row < n_rows ? 1u : 0u;  // enabler

    // opcodes-out yield staged: next_pc = pc + 1, next_ap = ap + ap_update_add_1.
    next_pc_out[row] = add(v_pc, 1u);
    next_ap_out[row] = add(v_ap, ap_update_add_1);
}

// assert_eq_opcode_imm (9 trace columns): the immediate-operand sibling of
// assert_eq_opcode. Decodes the instruction at pc, computes mem_dst_base, and
// performs the "Mem Verify Equal" read of dst at mem_dst_base + (offset0 -
// 32768). The second memory read (memory_address_to_id_2) reads the SAME id
// (dst_id) at pc+1 (the immediate operand) — that is the equality asserted.
// Unlike assert_eq_opcode there is no op1 base / offset2 decode (op1 is the
// immediate), so the instruction shape is simpler: only offset0, dst_base_fp,
// ap_update_add_1 are extracted, and the verify_instruction tuple uses the
// constants 32767 / 32769 for offset1 / offset2.
// Staged columns (lookup tuple slots that are not plain trace columns):
//   [0] vi_felt5  = (dst_base_fp*8 + 16) + 32   (verify_instruction expr)
//   [1] vi_felt6  = ap_update_add_1*32 + 256    (verify_instruction expr)
//   [2] dst_addr  = mem_dst_base + (offset0 - 32768)  (mem_address_to_id_1 addr)
//   [3] imm_addr  = pc + 1                        (mem_address_to_id_2 addr)
//   [4] next_pc   = pc + 2                        (opcodes-out yield)
//   [5] next_ap   = ap + ap_update_add_1          (opcodes-out yield)
// PackedM31 expressions use the modular fields.cuh ops; the instruction
// bit-extractions are PackedUInt16 (plain u32).
__global__ void assert_eq_opcode_imm_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                 // 9 trace columns
    uint32_t *vi_felt5, uint32_t *vi_felt6, // staged: verify_instruction exprs
    uint32_t *dst_addr_out,                 // staged: mem_address_to_id_1 read addr
    uint32_t *imm_addr_out,                 // staged: mem_address_to_id_2 read addr (pc+1)
    uint32_t *next_pc_out, uint32_t *next_ap_out // staged: opcodes-out yield exprs
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: read the instruction felt at pc and split limbs.
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);

    // offset0 = il[0] + ((il[1] & 127) << 9)  (PackedUInt16, plain u32 ops).
    uint32_t offset0 = il[0] + (((il[1] & 127u) << 9));
    trace[3][row] = offset0;
    // flags from limb 5 >> 3 + limb 6 << 6.
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);
    uint32_t dst_base_fp = (flags >> 0) & 1u;
    trace[4][row] = dst_base_fp;
    uint32_t ap_update_add_1 = (flags >> 11) & 1u;
    trace[5][row] = ap_update_add_1;

    // verify_instruction tuple staged exprs (M31 modular ops).
    // vi_felt5 = ((dst_base_fp*8) + 16) + 32.
    vi_felt5[row] = add(add(mul(dst_base_fp, 8u), 16u), 32u);
    // vi_felt6 = ap_update_add_1*32 + 256.
    vi_felt6[row] = add(mul(ap_update_add_1, 32u), 256u);

    // Signed offset: offset0 - 32768 (modular M31).
    uint32_t off0_signed = sub(offset0, 32768u);

    // mem_dst_base = dst_base_fp*fp + (1 - dst_base_fp)*ap.
    uint32_t mem_dst_base = add(mul(dst_base_fp, v_fp),
                               mul(sub(1u, dst_base_fp), v_ap));
    trace[6][row] = mem_dst_base;

    // Mem Verify Equal — Read Id at mem_dst_base + off0_signed.
    uint32_t dst_addr = add(mem_dst_base, off0_signed);
    dst_addr_out[row] = dst_addr;
    uint32_t dst_id = mem_addr_to_id(addr_table, dst_addr);
    trace[7][row] = dst_id;

    // memory_address_to_id_2 reads the same id (dst_id) at pc+1 (the immediate).
    imm_addr_out[row] = add(v_pc, 1u);

    trace[8][row] = row < n_rows ? 1u : 0u;  // enabler

    // opcodes-out yield staged: next_pc = pc + 2, next_ap = ap + ap_update_add_1.
    next_pc_out[row] = add(v_pc, 2u);
    next_ap_out[row] = add(v_ap, ap_update_add_1);
}

// assert_eq_opcode_double_deref (19 trace columns): the double-dereference
// sibling of assert_eq_opcode. Decodes the instruction at pc (offset0/1/2,
// dst_base_fp, op0_base_fp, ap_update_add_1 — the add_opcode_small instruction
// shape), computes the dst / op0 memory bases, then performs a DOUBLE
// dereference of the op0 operand:
//   1. Read Id at mem0_base + (offset1 - 32768): the POINTER's id
//      (mem1_base_id), a 29-bit (4-limb) read whose limbs recombine to a memory
//      address.
//   2. Read Id at (limb0 + limb1*512 + limb2*262144 + limb3*134217728)
//      + (offset2 - 32768): the doubly-dereferenced cell. Its id is the SAME id
//      as dst (dst_id) — that is the equality asserted (dst == [[op0] + off2]).
// The dst read (Read Id at mem_dst_base + (offset0 - 32768)) yields dst_id.
// Staged columns (lookup tuple slots that are not plain trace columns):
//   [0] vi_felt5  = dst_base_fp*8 + op0_base_fp*16    (verify_instruction expr)
//   [1] vi_felt6  = ap_update_add_1*32 + 256          (verify_instruction expr)
//   [2] mem1_base_addr = mem0_base + (offset1 - 32768) (mem_address_to_id_1 addr)
//   [3] dst_addr  = mem_dst_base + (offset0 - 32768)   (mem_address_to_id_3 addr)
//   [4] ddref_addr = ptr + (offset2 - 32768)           (mem_address_to_id_4 addr)
//   [5] next_pc   = pc + 1                             (opcodes-out yield)
//   [6] next_ap   = ap + ap_update_add_1               (opcodes-out yield)
// PackedM31 expressions use the modular fields.cuh ops; the instruction
// bit-extractions and partial_limb_msb are PackedUInt16 (plain u32).
__global__ void assert_eq_opcode_double_deref_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                 // 19 trace columns
    uint32_t *vi_felt5, uint32_t *vi_felt6, // staged: verify_instruction exprs
    uint32_t *mem1_base_addr_out,           // staged: mem_address_to_id_1 read addr
    uint32_t *dst_addr_out,                 // staged: mem_address_to_id_3 read addr
    uint32_t *ddref_addr_out,               // staged: mem_address_to_id_4 read addr
    uint32_t *next_pc_out, uint32_t *next_ap_out // staged: opcodes-out yield exprs
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: read the instruction felt at pc and split limbs.
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);

    // offset0/1/2 (PackedUInt16, plain u32 ops).
    uint32_t offset0 = il[0] + ((il[1] & 127u) << 9);
    trace[3][row] = offset0;
    uint32_t offset1 = (il[1] >> 7) + (il[2] << 2) + ((il[3] & 31u) << 11);
    trace[4][row] = offset1;
    uint32_t offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7u) << 13);
    trace[5][row] = offset2;
    // flags from limb 5 >> 3 + limb 6 << 6.
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);
    uint32_t dst_base_fp = (flags >> 0) & 1u;
    trace[6][row] = dst_base_fp;
    uint32_t op0_base_fp = (flags >> 1) & 1u;
    trace[7][row] = op0_base_fp;
    uint32_t ap_update_add_1 = (flags >> 11) & 1u;
    trace[8][row] = ap_update_add_1;

    // verify_instruction tuple staged exprs (M31 modular ops).
    // vi_felt5 = dst_base_fp*8 + op0_base_fp*16.
    vi_felt5[row] = add(mul(dst_base_fp, 8u), mul(op0_base_fp, 16u));
    // vi_felt6 = ap_update_add_1*32 + 256.
    vi_felt6[row] = add(mul(ap_update_add_1, 32u), 256u);

    // Signed offsets: offset - 32768 (modular M31).
    uint32_t off0_signed = sub(offset0, 32768u);
    uint32_t off1_signed = sub(offset1, 32768u);
    uint32_t off2_signed = sub(offset2, 32768u);

    // mem_dst_base = dst_base_fp*fp + (1 - dst_base_fp)*ap.
    uint32_t mem_dst_base = add(mul(dst_base_fp, v_fp),
                               mul(sub(1u, dst_base_fp), v_ap));
    trace[9][row] = mem_dst_base;
    // mem0_base = op0_base_fp*fp + (1 - op0_base_fp)*ap.
    uint32_t mem0_base = add(mul(op0_base_fp, v_fp),
                            mul(sub(1u, op0_base_fp), v_ap));
    trace[10][row] = mem0_base;

    // First deref — Read Id at mem0_base + off1_signed: the pointer's id.
    uint32_t mem1_base_addr = add(mem0_base, off1_signed);
    mem1_base_addr_out[row] = mem1_base_addr;
    uint32_t mem1_base_id = mem_addr_to_id(addr_table, mem1_base_addr);
    trace[11][row] = mem1_base_id;

    // Read Positive Known Id Num Bits 29: the pointer value (4 limbs).
    uint32_t ml[4];
    mem_id_to_limbs<4>(mem1_base_id, big_words, small_words, ml);
    trace[12][row] = ml[0];
    trace[13][row] = ml[1];
    trace[14][row] = ml[2];
    trace[15][row] = ml[3];
    trace[16][row] = (ml[3] & 2u) >> 1;  // partial_limb_msb (u16 bit math)

    // Mem Verify Equal — Read Id at mem_dst_base + off0_signed: dst_id.
    uint32_t dst_addr = add(mem_dst_base, off0_signed);
    dst_addr_out[row] = dst_addr;
    uint32_t dst_id = mem_addr_to_id(addr_table, dst_addr);
    trace[17][row] = dst_id;

    // Second deref — memory_address_to_id_4 reads the SAME id (dst_id) at the
    // pointer address recombined from the 4 limbs plus off2_signed (M31 ops).
    uint32_t ptr = add(add(ml[0], mul(ml[1], 512u)),
                       add(mul(ml[2], 262144u), mul(ml[3], 134217728u)));
    ddref_addr_out[row] = add(ptr, off2_signed);

    trace[18][row] = row < n_rows ? 1u : 0u;  // enabler

    // opcodes-out yield staged: next_pc = pc + 1, next_ap = ap + ap_update_add_1.
    next_pc_out[row] = add(v_pc, 1u);
    next_ap_out[row] = add(v_ap, ap_update_add_1);
}

// call_opcode_rel_imm (24 trace columns): a `call rel imm` instruction. The
// fixed instruction shape means the verify_instruction tuple is all constants
// except pc (no instruction-felt decode of offsets/flags is needed). Reads two
// 29-bit felts (stored_fp at ap, stored_ret_pc at ap+1) and one small-signed
// felt (the relative immediate distance_to_next_pc at pc+1). Staged columns are
// the lookup-tuple slots that are not plain trace columns:
//   [0] ret_pc_addr   = ap + 1   (memory_address_to_id_3 read addr)
//   [1] next_pc_addr  = pc + 1   (memory_address_to_id_5 read addr)
//   [2] m6_s5  = remainder_bits + mid_limbs_set*508   (memory_id_to_big_6)
//   [3] m6_s6  = mid_limbs_set*511
//   [4] m6_s22 = msb*136 - mid_limbs_set
//   [5] m6_s28 = msb*256
//   [6] next_pc_out = pc + read_small   (opcodes-out yield)
//   [7] next_ap_out = ap + 2            (opcodes-out yield)
__global__ void call_opcode_rel_imm_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,                       // 24 trace columns
    uint32_t *ret_pc_addr, uint32_t *next_pc_addr, // staged read addrs (ap+1, pc+1)
    uint32_t *m6_s5, uint32_t *m6_s6,             // staged: memory_id_to_big_6 slots
    uint32_t *m6_s22, uint32_t *m6_s28,           // staged: memory_id_to_big_6 slots
    uint32_t *next_pc_out, uint32_t *next_ap_out  // staged: opcodes-out yield exprs
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction at pc: the verify_instruction tuple is all constants
    // for this opcode, so the instruction felt is not split into trace columns.

    // Read Positive Num Bits 29: stored_fp at ap.
    uint32_t stored_fp_id = mem_addr_to_id(addr_table, v_ap);
    trace[3][row] = stored_fp_id;
    uint32_t fl[4];
    mem_id_to_limbs<4>(stored_fp_id, big_words, small_words, fl);
    trace[4][row] = fl[0];
    trace[5][row] = fl[1];
    trace[6][row] = fl[2];
    trace[7][row] = fl[3];
    trace[8][row] = (fl[3] & 2u) >> 1;  // partial_limb_msb (u16 bit math)

    // Read Positive Num Bits 29: stored_ret_pc at ap + 1.
    uint32_t ret_addr = add(v_ap, 1u);
    ret_pc_addr[row] = ret_addr;
    uint32_t stored_ret_pc_id = mem_addr_to_id(addr_table, ret_addr);
    trace[9][row] = stored_ret_pc_id;
    uint32_t rl[4];
    mem_id_to_limbs<4>(stored_ret_pc_id, big_words, small_words, rl);
    trace[10][row] = rl[0];
    trace[11][row] = rl[1];
    trace[12][row] = rl[2];
    trace[13][row] = rl[3];
    trace[14][row] = (rl[3] & 2u) >> 1;  // partial_limb_msb

    // Read Small: distance_to_next_pc (the relative immediate) at pc + 1.
    uint32_t npc_addr = add(v_pc, 1u);
    next_pc_addr[row] = npc_addr;
    uint32_t dist_id = mem_addr_to_id(addr_table, npc_addr);
    trace[15][row] = dist_id;
    uint32_t dl[28];
    mem_id_to_limbs<28>(dist_id, big_words, small_words, dl);

    // Decode Small Sign.
    uint32_t msb = (dl[27] == 256u) ? 1u : 0u;
    trace[16][row] = msb;
    uint32_t mid_limbs_set = ((dl[20] == 511u) && (msb != 0u)) ? 1u : 0u;
    trace[17][row] = mid_limbs_set;
    // decode_small_sign outputs (M31 ops).
    uint32_t dss2 = mul(mid_limbs_set, 508u);
    uint32_t dss3 = mul(mid_limbs_set, 511u);
    uint32_t dss4 = sub(mul(msb, 136u), mid_limbs_set);
    uint32_t dss5 = mul(msb, 256u);

    uint32_t d0 = dl[0];
    trace[18][row] = d0;
    uint32_t d1 = dl[1];
    trace[19][row] = d1;
    uint32_t d2 = dl[2];
    trace[20][row] = d2;
    uint32_t remainder_bits = dl[3] & 3u;  // PackedUInt16
    trace[21][row] = remainder_bits;
    trace[22][row] = (remainder_bits & 2u) >> 1;  // partial_limb_msb

    // memory_id_to_big_6 staged slots.
    m6_s5[row] = add(remainder_bits, dss2);
    m6_s6[row] = dss3;
    m6_s22[row] = dss4;
    m6_s28[row] = dss5;

    trace[23][row] = row < n_rows ? 1u : 0u;  // enabler

    // read_small_output.0 = ((((d0 + d1*512) + d2*262144)
    //   + remainder_bits*134217728) - msb) - 536870912*mid_limbs_set  (M31).
    uint32_t read_small =
        sub(sub(add(add(add(d0, mul(d1, 512u)), mul(d2, 262144u)),
                    mul(remainder_bits, 134217728u)),
                msb),
            mul(536870912u, mid_limbs_set));
    // opcodes-out yield staged exprs: next_pc = pc + read_small, next_ap = ap + 2.
    next_pc_out[row] = add(v_pc, read_small);
    next_ap_out[row] = add(v_ap, 2u);
}

// add_opcode (103 trace columns): decodes the instruction at pc, then three
// FULL "Read Positive Num Bits 252" memory reads (dst/op0/op1) each producing
// an id trace column and a 28-limb f252 decode (all limbs are trace columns,
// no small-sign decode), then a carry-chain add check producing sub_p_bit.
// Staged columns (lookup-tuple expressions that are not plain trace columns):
//   [0] vi_felt5 = dst_base_fp*8 + op0_base_fp*16 + op1_imm*32
//                   + op1_base_fp*64 + op1_base_ap*128 + 256
//   [1] vi_felt6 = ap_update_add_1*32 + 256
//   [2] dst_addr = mem_dst_base + (offset0 - 32768)
//   [3] op0_addr = mem0_base + (offset1 - 32768)
//   [4] op1_addr = mem1_base + (offset2 - 32768)
//   [5] next_pc  = pc + 1 + op1_imm      (opcodes-out yield)
//   [6] next_ap  = ap + ap_update_add_1  (opcodes-out yield)
// All PackedM31 expressions use the modular fields.cuh ops; the instruction
// bit-extractions are PackedUInt16 (plain u32).
__global__ void add_opcode_trace_kernel(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    uint32_t *const *trace,    // 103 trace columns
    uint32_t *const *staged    // 7 staged columns
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) return;
    uint32_t v_pc = pc[row], v_ap = ap[row], v_fp = fp[row];
    trace[0][row] = v_pc;
    trace[1][row] = v_ap;
    trace[2][row] = v_fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    uint32_t instr_id = mem_addr_to_id(addr_table, v_pc);
    uint32_t il[7];
    mem_id_to_limbs<7>(instr_id, big_words, small_words, il);
    uint32_t offset0 = il[0] + ((il[1] & 127u) << 9);
    uint32_t offset1 = (il[1] >> 7) + (il[2] << 2) + ((il[3] & 31u) << 11);
    uint32_t offset2 = (il[3] >> 5) + (il[4] << 4) + ((il[5] & 7u) << 13);
    uint32_t flags = (il[5] >> 3) + (il[6] << 6);
    uint32_t dst_base_fp = (flags >> 0) & 1u;
    uint32_t op0_base_fp = (flags >> 1) & 1u;
    uint32_t op1_imm = (flags >> 2) & 1u;
    uint32_t op1_base_fp = (flags >> 3) & 1u;
    uint32_t ap_update_add_1 = (flags >> 11) & 1u;
    trace[3][row] = offset0;
    trace[4][row] = offset1;
    trace[5][row] = offset2;
    trace[6][row] = dst_base_fp;
    trace[7][row] = op0_base_fp;
    trace[8][row] = op1_imm;
    trace[9][row] = op1_base_fp;
    trace[10][row] = ap_update_add_1;
    // op1_base_ap = 1 - op1_imm - op1_base_fp (modular M31).
    uint32_t op1_base_ap = sub(sub(1u, op1_imm), op1_base_fp);

    // Memory bases (modular M31).
    uint32_t mem_dst_base = add(mul(dst_base_fp, v_fp), mul(sub(1u, dst_base_fp), v_ap));
    uint32_t mem0_base = add(mul(op0_base_fp, v_fp), mul(sub(1u, op0_base_fp), v_ap));
    uint32_t mem1_base = add(add(mul(op1_imm, v_pc), mul(op1_base_fp, v_fp)),
                             mul(op1_base_ap, v_ap));
    trace[11][row] = mem_dst_base;
    trace[12][row] = mem0_base;
    trace[13][row] = mem1_base;

    // Signed offsets: offset - 32768 (modular M31).
    uint32_t off0_signed = sub(offset0, 32768u);
    uint32_t off1_signed = sub(offset1, 32768u);
    uint32_t off2_signed = sub(offset2, 32768u);

    // dst read (full 252) at mem_dst_base + off0_signed.
    uint32_t dst_addr = add(mem_dst_base, off0_signed);
    uint32_t dst_id = mem_addr_to_id(addr_table, dst_addr);
    trace[14][row] = dst_id;
    uint32_t dstl[28];
    mem_id_to_limbs<28>(dst_id, big_words, small_words, dstl);
    #pragma unroll
    for (int i = 0; i < 28; ++i) trace[15 + i][row] = dstl[i];

    // op0 read (full 252) at mem0_base + off1_signed.
    uint32_t op0_addr = add(mem0_base, off1_signed);
    uint32_t op0_id = mem_addr_to_id(addr_table, op0_addr);
    trace[43][row] = op0_id;
    uint32_t op0l[28];
    mem_id_to_limbs<28>(op0_id, big_words, small_words, op0l);
    #pragma unroll
    for (int i = 0; i < 28; ++i) trace[44 + i][row] = op0l[i];

    // op1 read (full 252) at mem1_base + off2_signed.
    uint32_t op1_addr = add(mem1_base, off2_signed);
    uint32_t op1_id = mem_addr_to_id(addr_table, op1_addr);
    trace[72][row] = op1_id;
    uint32_t op1l[28];
    mem_id_to_limbs<28>(op1_id, big_words, small_words, op1l);
    #pragma unroll
    for (int i = 0; i < 28; ++i) trace[73 + i][row] = op1l[i];

    // Verify Add 252: sub_p_bit = 1 & (op0_limb0 ^ op1_limb0 ^ dst_limb0)
    // (PackedUInt16 bit math, plain u32).
    uint32_t sub_p_bit = 1u & ((op0l[0] ^ op1l[0]) ^ dstl[0]);
    trace[101][row] = sub_p_bit;

    trace[102][row] = row < n_rows ? 1u : 0u;  // enabler

    // verify_instruction staged felts (modular M31).
    staged[0][row] = add(
        add(add(mul(dst_base_fp, 8u), mul(op0_base_fp, 16u)),
            add(mul(op1_imm, 32u), mul(op1_base_fp, 64u))),
        add(mul(op1_base_ap, 128u), 256u));
    staged[1][row] = add(mul(ap_update_add_1, 32u), 256u);
    // memory_address_to_id read addresses (staged tuple slots).
    staged[2][row] = dst_addr;
    staged[3][row] = op0_addr;
    staged[4][row] = op1_addr;
    // opcodes-out yield staged: next_pc = pc + 1 + op1_imm, next_ap = ap + ap_update_add_1.
    staged[5][row] = add(add(v_pc, 1u), op1_imm);
    staged[6][row] = add(v_ap, ap_update_add_1);
}

}  // namespace

extern "C" void assert_eq_opcode_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *vi_felt5, uint32_t *vi_felt6,
    uint32_t *dst_addr_out, uint32_t *op1_addr_out,
    uint32_t *next_pc_out, uint32_t *next_ap_out
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    assert_eq_opcode_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace),
        vi_felt5, vi_felt6, dst_addr_out, op1_addr_out, next_pc_out, next_ap_out);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void assert_eq_opcode_imm_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *vi_felt5, uint32_t *vi_felt6,
    uint32_t *dst_addr_out, uint32_t *imm_addr_out,
    uint32_t *next_pc_out, uint32_t *next_ap_out
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    assert_eq_opcode_imm_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace),
        vi_felt5, vi_felt6, dst_addr_out, imm_addr_out, next_pc_out, next_ap_out);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void assert_eq_opcode_double_deref_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *vi_felt5, uint32_t *vi_felt6,
    uint32_t *mem1_base_addr_out, uint32_t *dst_addr_out, uint32_t *ddref_addr_out,
    uint32_t *next_pc_out, uint32_t *next_ap_out
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    assert_eq_opcode_double_deref_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace),
        vi_felt5, vi_felt6, mem1_base_addr_out, dst_addr_out, ddref_addr_out,
        next_pc_out, next_ap_out);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void call_opcode_rel_imm_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *ret_pc_addr, uint32_t *next_pc_addr,
    uint32_t *m6_s5, uint32_t *m6_s6,
    uint32_t *m6_s22, uint32_t *m6_s28,
    uint32_t *next_pc_out, uint32_t *next_ap_out
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    call_opcode_rel_imm_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace),
        ret_pc_addr, next_pc_addr, m6_s5, m6_s6, m6_s22, m6_s28,
        next_pc_out, next_ap_out);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void add_opcode_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    const uint32_t *const *staged
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    add_opcode_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace), const_cast<uint32_t *const *>(staged));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void jnz_opcode_taken_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *vi_off1, uint32_t *vi_off2,
    uint32_t *addr_dst, uint32_t *next_pc_addr,
    uint32_t *m4_s4, uint32_t *m4_s5,
    uint32_t *m4_s22, uint32_t *m4_s28,
    uint32_t *next_pc_out, uint32_t *next_ap_out
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    jnz_opcode_taken_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace),
        vi_off1, vi_off2, addr_dst, next_pc_addr,
        m4_s4, m4_s5, m4_s22, m4_s28, next_pc_out, next_ap_out);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void add_ap_opcode_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    uint32_t *vi_felt,
    uint32_t *op1_addr_out,
    uint32_t *m_s5, uint32_t *m_s6,
    uint32_t *m_s22, uint32_t *m_s28,
    uint32_t *rc18_val,
    uint32_t *next_pc_out, uint32_t *next_ap_out
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    add_ap_opcode_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace),
        vi_felt, op1_addr_out, m_s5, m_s6, m_s22, m_s28, rc18_val,
        next_pc_out, next_ap_out);
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

extern "C" void add_opcode_small_trace(
    const uint32_t *pc, const uint32_t *ap, const uint32_t *fp,
    const uint32_t *addr_table,
    const uint32_t *big_words, const uint32_t *small_words,
    uint32_t n_rows,
    uint32_t column_length,
    const uint32_t *const *trace,
    const uint32_t *const *staged
) {
    uint32_t blocks = (column_length + OP_BLOCK - 1) / OP_BLOCK;
    add_opcode_small_trace_kernel<<<blocks, OP_BLOCK>>>(
        pc, ap, fp, addr_table, big_words, small_words, n_rows, column_length,
        const_cast<uint32_t *const *>(trace), const_cast<uint32_t *const *>(staged));
    stwo_maybe_debug_sync();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

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
