#include "secure_field_support.h"

constant uint STWO_METAL_LOGUP_BLOCK_DIM = 256u;
constant uint STWO_METAL_LOGUP_LANE_BLOCK = 16u;

static inline StwoMetalQm31 stwo_metal_load_packed_lane_qm31(
    device const uint *packed,
    uint index
) {
    uint lane = index & (STWO_METAL_LOGUP_LANE_BLOCK - 1u);
    uint base = (index / STWO_METAL_LOGUP_LANE_BLOCK) *
        (4u * STWO_METAL_LOGUP_LANE_BLOCK) + lane;
    return StwoMetalQm31 {
        packed[base],
        packed[base + STWO_METAL_LOGUP_LANE_BLOCK],
        packed[base + 2u * STWO_METAL_LOGUP_LANE_BLOCK],
        packed[base + 3u * STWO_METAL_LOGUP_LANE_BLOCK],
    };
}

kernel void logup_fraction_chain_u32x4(
    device uint *num0 [[buffer(0)]],
    device uint *num1 [[buffer(1)]],
    device uint *num2 [[buffer(2)]],
    device uint *num3 [[buffer(3)]],
    device const uint *denom_packed [[buffer(4)]],
    device const uint *prev0 [[buffer(5)]],
    device const uint *prev1 [[buffer(6)]],
    device const uint *prev2 [[buffer(7)]],
    device const uint *prev3 [[buffer(8)]],
    constant uint &n_elements [[buffer(9)]],
    constant bool &has_prev [[buffer(10)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= n_elements) {
        return;
    }

    StwoMetalQm31 numerator = StwoMetalQm31 {
        num0[index],
        num1[index],
        num2[index],
        num3[index],
    };
    StwoMetalQm31 denom_inv =
        stwo_metal_qm31_inverse(stwo_metal_load_packed_lane_qm31(denom_packed, index));
    StwoMetalQm31 value = stwo_metal_qm31_mul(numerator, denom_inv);

    if (has_prev) {
        StwoMetalQm31 prev = StwoMetalQm31 {
            prev0[index],
            prev1[index],
            prev2[index],
            prev3[index],
        };
        value = stwo_metal_qm31_add(value, prev);
    }

    num0[index] = value.a;
    num1[index] = value.b;
    num2[index] = value.c;
    num3[index] = value.d;
}
