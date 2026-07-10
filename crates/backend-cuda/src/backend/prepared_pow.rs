//! Prepared resident Blake2s proof-of-work.
//!
//! The persistent search kernel reads the current device transcript digest and
//! publishes the globally lowest numeric `u64` satisfying the ordinary Blake2s
//! channel check. Replay contains only stream memsets and that one kernel.

use std::collections::BTreeSet;

use super::device_transcript::BLAKE2S_TRANSCRIPT_STATE_WORDS;
use super::exec_context::{
    check_cuda, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
pub const POW_NONCE_WORDS: usize = 2;
pub const POW_U64_ALIGNMENT_WORDS: usize = core::mem::align_of::<u64>() / WORD_BYTES;

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
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use stwo::core::channel::{Blake2sChannelGeneric, Channel};
    use stwo::core::proof_of_work::GrindOps;
    use stwo::core::vcs::blake2_hash::Blake2sHasherGeneric;
    use stwo::prover::backend::CpuBackend;

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
        for pow_bits in [0, 1, 7, 16, 31, 32] {
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
    fn numeric_global_minimum_matches_sequential_reference() {
        let mut channel = Blake2sChannelGeneric::<false>::default();
        channel.mix_u32s(&[42, 77, 99]);
        let expected = CpuBackend::grind(&channel, 10);
        let workers = 37u64;
        let strided = (0..workers)
            .filter_map(|worker| {
                (worker..=expected)
                    .step_by(workers as usize)
                    .find(|&nonce| reference_valid_pow(&channel, 10, nonce))
            })
            .min()
            .unwrap();
        assert_eq!(strided, expected);
    }

    #[test]
    fn pow_bits_above_supported_reference_range_fail_closed() {
        assert_eq!(
            validate_pow_bits(33),
            Err(PreparedBlake2sPowError::InvalidPowBits(33))
        );
        assert_eq!(validate_pow_bits(32), Ok(()));
    }
}
