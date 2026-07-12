//! Prepared resident Blake2s proof-of-work.
//!
//! The persistent search kernel reads the current device transcript digest and
//! publishes the numerically smallest nonce on the SIMD grind lattice
//! `{(hi << 32) | low : 0 <= low < 2^POW_GRIND_LOW_BITS}` satisfying the
//! ordinary Blake2s channel check. The SIMD reference
//! (`stwo::prover::backend::simd::grind`) scans hi ascending, then low
//! ascending within each hi; because `low < 2^20 < 2^32`, that scan order IS
//! numeric order on the lattice, so the kernel's numeric minimum over mapped
//! lattice nonces is byte-identical to `SimdBackend::grind`. Replay contains
//! only stream memsets and that one kernel.

use std::collections::BTreeSet;

use super::device_transcript::BLAKE2S_TRANSCRIPT_STATE_WORDS;
use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
pub const POW_NONCE_WORDS: usize = 2;
pub const POW_U64_ALIGNMENT_WORDS: usize = core::mem::align_of::<u64>() / WORD_BYTES;
/// Low-bit width of the SIMD grind lattice (GRIND_LOW_BITS in
/// `stwo::prover::backend::simd::grind`).
pub const POW_GRIND_LOW_BITS: u32 = 20;

/// Host mirror of the kernel's monotone index -> nonce map
/// (`pow_index_to_nonce` in `cuda/resident_pow.cu`): linear search index `i`
/// covers exactly the SIMD lattice nonce `(hi << 32) | low` with
/// `hi = i >> 20` and `low = i & 0xFFFFF`. Strictly increasing in `i` (nonce
/// bits 20..31 are always zero), so numeric order on mapped nonces equals
/// index order equals the SIMD (hi ascending, low ascending) scan order.
pub const fn pow_index_to_nonce(index: u64) -> u64 {
    ((index >> POW_GRIND_LOW_BITS) << 32) | (index & ((1 << POW_GRIND_LOW_BITS) - 1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Blake2sPowWorkspaceRequirements {
    pub state_words: usize,
    pub nonce_words: usize,
    pub best_nonce_words: usize,
    pub completed_blocks_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Blake2sPowWorkspaceSlots {
    pub best_nonce: ArenaSlotId,
    pub completed_blocks: ArenaSlotId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Blake2sPowArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedBlake2sPowError {
    InvalidPowBits(u32),
    ContextMismatch(ArenaSlotId),
    AliasedSlot(ArenaSlotId),
    StateTooSmall {
        required: usize,
        actual: usize,
    },
    NonceTooSmall {
        required: usize,
        actual: usize,
    },
    SlotTooSmall {
        slot: ArenaSlotId,
        required: usize,
        actual: usize,
    },
    MisalignedBestNonce(ArenaSlotId),
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedBlake2sPowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid prepared CUDA Blake2s PoW graph: {self:?}")
    }
}

impl std::error::Error for PreparedBlake2sPowError {}

impl From<ArenaError> for PreparedBlake2sPowError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedBlake2sPowError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

pub const fn blake2s_pow_workspace_requirements() -> Blake2sPowWorkspaceRequirements {
    Blake2sPowWorkspaceRequirements {
        state_words: BLAKE2S_TRANSCRIPT_STATE_WORDS,
        nonce_words: POW_NONCE_WORDS,
        best_nonce_words: POW_NONCE_WORDS,
        completed_blocks_words: 1,
    }
}

impl Blake2sPowWorkspaceRequirements {
    pub fn arena_slot_requirements(
        self,
        slots: Blake2sPowWorkspaceSlots,
    ) -> Result<[Blake2sPowArenaSlotRequirement; 2], PreparedBlake2sPowError> {
        if slots.best_nonce == slots.completed_blocks {
            return Err(PreparedBlake2sPowError::AliasedSlot(slots.best_nonce));
        }
        Ok([
            Blake2sPowArenaSlotRequirement {
                id: slots.best_nonce,
                len_words: self.best_nonce_words,
                alignment_words: POW_U64_ALIGNMENT_WORDS,
            },
            Blake2sPowArenaSlotRequirement {
                id: slots.completed_blocks,
                len_words: self.completed_blocks_words,
                alignment_words: 1,
            },
        ])
    }
}

pub struct PreparedBlake2sPowGraph<'a> {
    arena: &'a DeviceArena,
    pow_bits: u32,
    transcript_state: ArenaSlice,
    best_nonce: ArenaSlice,
    completed_blocks: ArenaSlice,
    transcript_nonce: ArenaSlice,
}

impl<'a> PreparedBlake2sPowGraph<'a> {
    pub fn prepare(
        arena: &'a DeviceArena,
        transcript_state: ArenaSlice,
        pow_bits: u32,
        transcript_nonce: ArenaSlice,
        slots: Blake2sPowWorkspaceSlots,
    ) -> Result<Self, PreparedBlake2sPowError> {
        validate_pow_bits(pow_bits)?;
        let requirements = blake2s_pow_workspace_requirements();
        if transcript_state.len_words() < requirements.state_words {
            return Err(PreparedBlake2sPowError::StateTooSmall {
                required: requirements.state_words,
                actual: transcript_state.len_words(),
            });
        }
        if transcript_nonce.len_words() < requirements.nonce_words {
            return Err(PreparedBlake2sPowError::NonceTooSmall {
                required: requirements.nonce_words,
                actual: transcript_nonce.len_words(),
            });
        }
        let slot_requirements = requirements.arena_slot_requirements(slots)?;
        let best_nonce = bind_slot(arena, slot_requirements[0])?;
        let completed_blocks = bind_slot(arena, slot_requirements[1])?;
        if (best_nonce.as_u32_ptr() as usize) % core::mem::align_of::<u64>() != 0 {
            return Err(PreparedBlake2sPowError::MisalignedBestNonce(
                best_nonce.id(),
            ));
        }

        let bindings = [
            transcript_state,
            transcript_nonce,
            best_nonce,
            completed_blocks,
        ];
        let context = arena.context().identity_token();
        let mut identities = BTreeSet::new();
        for binding in bindings {
            if binding.context_token() != context {
                return Err(PreparedBlake2sPowError::ContextMismatch(binding.id()));
            }
            if !identities.insert(binding.id()) {
                return Err(PreparedBlake2sPowError::AliasedSlot(binding.id()));
            }
        }

        Ok(Self {
            arena,
            pow_bits,
            transcript_state,
            best_nonce,
            completed_blocks,
            transcript_nonce,
        })
    }

    pub fn nonce_destination(&self) -> ArenaSlice {
        self.transcript_nonce
    }

    /// Initialize caller-owned scratch and enqueue the one persistent search
    /// kernel. No digest or nonce crosses the host during replay.
    pub fn launch(&self) -> Result<(), PreparedBlake2sPowError> {
        unsafe {
            self.arena.context().memset_async(
                self.best_nonce.as_void_ptr(),
                0xff,
                POW_NONCE_WORDS * WORD_BYTES,
            )?;
            self.arena.context().memset_async(
                self.completed_blocks.as_void_ptr(),
                0,
                WORD_BYTES,
            )?;
            self.arena.context().memset_async(
                self.transcript_nonce.as_void_ptr(),
                0xff,
                POW_NONCE_WORDS * WORD_BYTES,
            )?;
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_blake2s_pow_persistent_on(
                self.transcript_state.as_u32_ptr().cast_const(),
                self.pow_bits,
                self.best_nonce.as_u32_ptr().cast::<u64>(),
                self.completed_blocks.as_u32_ptr(),
                self.transcript_nonce.as_u32_ptr(),
                self.arena.context().stream_raw().as_ptr(),
            )
        };
        check_cuda("prepared_blake2s_pow", code)?;
        Ok(())
    }
}

/// Mirrors the SIMD reference's `pow_bits <= 32` assertion. The lattice
/// enumeration keeps this bound sound: `pow_bits = 32` needs ~2^32 attempts in
/// expectation, i.e. ~2^12 hi blocks of 2^20 lows each, far below the kernel's
/// `hi < 2^31 - 1` give-up limit (which itself mirrors the SIMD post-grind
/// assertion that the found hi is reduced modulo the M31 prime).
fn validate_pow_bits(pow_bits: u32) -> Result<(), PreparedBlake2sPowError> {
    if pow_bits > 32 {
        return Err(PreparedBlake2sPowError::InvalidPowBits(pow_bits));
    }
    Ok(())
}

fn bind_slot(
    arena: &DeviceArena,
    requirement: Blake2sPowArenaSlotRequirement,
) -> Result<ArenaSlice, PreparedBlake2sPowError> {
    let slice = arena.bind(requirement.id)?;
    if slice.len_words() < requirement.len_words {
        return Err(PreparedBlake2sPowError::SlotTooSmall {
            slot: requirement.id,
            required: requirement.len_words,
            actual: slice.len_words(),
        });
    }
    // Pooled slots may be larger than any single logical buffer; expose only
    // the logical extent so no consumer derives sizes from the pooled surplus.
    Ok(slice.truncated(requirement.len_words))
}

#[cfg(test)]
mod tests {
    use stwo::core::channel::{Blake2sChannelGeneric, Channel};
    use stwo::core::proof_of_work::GrindOps;
    use stwo::core::vcs::blake2_hash::Blake2sHasherGeneric;
    use stwo::prover::backend::simd::SimdBackend;

    use super::*;

    fn reference_valid_pow(
        channel: &Blake2sChannelGeneric<false>,
        pow_bits: u32,
        nonce: u64,
    ) -> bool {
        let mut prefix = Blake2sHasherGeneric::<false>::default();
        prefix.update(&Blake2sChannelGeneric::<false>::POW_PREFIX.to_le_bytes());
        prefix.update(&[0u8; 12]);
        prefix.update(&channel.digest().0);
        prefix.update(&pow_bits.to_le_bytes());
        let prefixed = prefix.finalize();

        let mut candidate = Blake2sHasherGeneric::<false>::default();
        candidate.update(prefixed.as_ref());
        candidate.update(&nonce.to_le_bytes());
        let hash = candidate.finalize();
        u32::from_le_bytes(hash.0[..4].try_into().unwrap()).trailing_zeros() >= pow_bits
    }

    #[test]
    fn exact_scratch_layout() {
        let requirements = blake2s_pow_workspace_requirements();
        assert_eq!(requirements.state_words, 16);
        assert_eq!(requirements.nonce_words, 2);
        assert_eq!(requirements.best_nonce_words, 2);
        assert_eq!(requirements.completed_blocks_words, 1);
        assert_eq!(POW_U64_ALIGNMENT_WORDS, 2);
        assert_eq!(
            requirements
                .arena_slot_requirements(Blake2sPowWorkspaceSlots {
                    best_nonce: ArenaSlotId(7),
                    completed_blocks: ArenaSlotId(8),
                })
                .unwrap(),
            [
                Blake2sPowArenaSlotRequirement {
                    id: ArenaSlotId(7),
                    len_words: 2,
                    alignment_words: 2,
                },
                Blake2sPowArenaSlotRequirement {
                    id: ArenaSlotId(8),
                    len_words: 1,
                    alignment_words: 1,
                },
            ]
        );
    }

    #[test]
    fn pow_prefix_and_validity_match_blake2s_channel() {
        let mut channel = Blake2sChannelGeneric::<false>::default();
        channel.mix_u32s(&[1, 0x1122_3344, 0xaabb_ccdd, 9]);
        for pow_bits in [0, 1, 7, 16, 26, 31, 32] {
            for nonce in [0, 1, 17, 0x1122_3344_5566_7788] {
                assert_eq!(
                    reference_valid_pow(&channel, pow_bits, nonce),
                    channel.verify_pow_nonce(pow_bits, nonce),
                    "pow_bits={pow_bits}, nonce={nonce}"
                );
            }
        }
    }

    #[test]
    fn index_to_nonce_mapping_pins_the_simd_lattice() {
        // Identity below the first hi block.
        assert_eq!(pow_index_to_nonce(0), 0);
        assert_eq!(pow_index_to_nonce(1), 1);
        assert_eq!(pow_index_to_nonce((1 << 20) - 1), (1 << 20) - 1);
        // Crossing a low-block boundary increments hi and resets low.
        assert_eq!(pow_index_to_nonce(1 << 20), 1 << 32);
        assert_eq!(pow_index_to_nonce((1 << 20) + 1), (1 << 32) | 1);

        // Equivalence with the lattice documented in grind_blake2s.cu and
        // implemented by the SIMD grind: index (hi << 20) | low covers
        // exactly the nonce (hi << 32) | low, 0 <= low < 2^20.
        for hi in [0u64, 1, 2, 41, (1 << 31) - 2] {
            for low in [0u64, 1, 0x12345, (1 << 20) - 1] {
                assert_eq!(pow_index_to_nonce((hi << 20) | low), (hi << 32) | low);
            }
        }

        // Strict monotonicity across block boundaries: numeric order on
        // mapped nonces equals index order, which is the SIMD scan order.
        let samples = [
            0u64,
            1,
            (1 << 20) - 1,
            1 << 20,
            (1 << 20) + 1,
            (5 << 20) + 7,
            u64::from(u32::MAX),
        ];
        for pair in samples.windows(2) {
            assert!(pow_index_to_nonce(pair[0]) < pow_index_to_nonce(pair[1]));
        }
    }

    #[test]
    fn mapped_strided_minimum_matches_simd_grind() {
        let mut channel = Blake2sChannelGeneric::<false>::default();
        channel.mix_u32s(&[42, 77, 99]);
        let expected = SimdBackend::grind(&channel, 10);
        // Invert the monotone map to bound the walk at the known answer.
        let expected_index =
            ((expected >> 32) << POW_GRIND_LOW_BITS) | (expected & ((1 << POW_GRIND_LOW_BITS) - 1));
        assert_eq!(pow_index_to_nonce(expected_index), expected);
        let workers = 37u64;
        let strided = (0..workers)
            .filter_map(|worker| {
                (worker..=expected_index)
                    .step_by(workers as usize)
                    .map(pow_index_to_nonce)
                    .find(|&nonce| reference_valid_pow(&channel, 10, nonce))
            })
            .min()
            .unwrap();
        assert_eq!(strided, expected);
    }

    #[test]
    fn lattice_enumeration_skips_dense_nonces_outside_the_simd_search_space() {
        // Synthetic qualifying set reproducing the divergence scenario: the
        // dense-u64 minimum 0x30_0000 has low-32 bits >= 2^20, so the SIMD
        // grind NEVER tests it; the reference answer is the lattice point.
        let lattice_hit = (3u64 << 32) | 7;
        let dense_only_hit = 0x30_0000u64;
        let qualifies = |nonce: u64| nonce == dense_only_hit || nonce == lattice_hit;
        assert!(dense_only_hit < lattice_hit, "dense minimum must differ");

        // SIMD scan-order reference (hi ascending, low ascending) == index
        // order under the monotone map.
        let simd_answer = (0u64..)
            .map(pow_index_to_nonce)
            .find(|&nonce| qualifies(nonce))
            .unwrap();
        assert_eq!(simd_answer, lattice_hit);

        // Kernel model: workers stride the index space, atomicMin over MAPPED
        // nonces. Each worker's minimum is its first hit (map is monotone per
        // residue class); the global minimum is the SIMD answer, and the
        // dense-only nonce is never enumerated.
        let workers = 5u64;
        let limit = (3u64 << POW_GRIND_LOW_BITS) + 8;
        let strided = (0..workers)
            .filter_map(|worker| {
                (worker..limit)
                    .step_by(workers as usize)
                    .map(pow_index_to_nonce)
                    .inspect(|&nonce| assert_ne!(nonce, dense_only_hit))
                    .find(|&nonce| qualifies(nonce))
            })
            .min()
            .unwrap();
        assert_eq!(strided, simd_answer);
    }

    #[test]
    fn pow_bits_above_supported_reference_range_fail_closed() {
        assert_eq!(
            validate_pow_bits(33),
            Err(PreparedBlake2sPowError::InvalidPowBits(33))
        );
        assert_eq!(validate_pow_bits(32), Ok(()));
        assert_eq!(validate_pow_bits(26), Ok(()));
    }
}
