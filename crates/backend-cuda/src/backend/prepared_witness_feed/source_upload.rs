//! Runtime receipt for one statement-varying witness-feed source upload.
//!
//! Process-local arena and pointer tokens are admission evidence only. They
//! must never enter a semantic, artifact, program, or proof identity.

use super::{ArenaSlice, ArenaSlotId, DeviceArena, PreparedWitnessFeedError, WitnessFeedContract};

/// Process-local proof that one shape-checked host source was copied and
/// fenced.
///
/// The receipt is unforgeable outside this module. Writer exclusivity for its
/// source slice remains a proof-plan obligation; this receipt only tracks
/// uploads performed through its prepared feed graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WitnessFeedSourceUploadReceipt {
    contract_identity: [u8; 32],
    arena_identity: usize,
    exec_context_token: u64,
    source_slot: ArenaSlotId,
    source_address: usize,
    source_words: usize,
    generation: u64,
}

impl WitnessFeedSourceUploadReceipt {
    pub(super) fn prepare(
        binding: WitnessFeedSourceUploadBinding,
        words: &[u32],
        generation: u64,
    ) -> Result<Self, PreparedWitnessFeedError> {
        if generation == 0 {
            return Err(PreparedWitnessFeedError::SourceUploadGenerationOverflow);
        }
        if words.len() != binding.source_words {
            return Err(PreparedWitnessFeedError::SlotSizeMismatch {
                slot: binding.source_slot,
                expected_words: binding.source_words,
                actual_words: words.len(),
            });
        }
        Ok(Self {
            contract_identity: binding.contract_identity,
            arena_identity: binding.arena_identity,
            exec_context_token: binding.exec_context_token,
            source_slot: binding.source_slot,
            source_address: binding.source_address,
            source_words: binding.source_words,
            generation,
        })
    }

    pub(super) fn matches(
        self,
        binding: WitnessFeedSourceUploadBinding,
        current_generation: u64,
    ) -> bool {
        self.generation != 0
            && self.generation == current_generation
            && self.contract_identity == binding.contract_identity
            && self.arena_identity == binding.arena_identity
            && self.exec_context_token == binding.exec_context_token
            && self.source_slot == binding.source_slot
            && self.source_address == binding.source_address
            && self.source_words == binding.source_words
    }

    pub const fn contract_identity(self) -> [u8; 32] {
        self.contract_identity
    }

    pub const fn arena_identity(self) -> usize {
        self.arena_identity
    }

    pub const fn exec_context_token(self) -> u64 {
        self.exec_context_token
    }

    pub const fn source_slot(self) -> ArenaSlotId {
        self.source_slot
    }

    pub const fn source_address(self) -> usize {
        self.source_address
    }

    pub const fn source_words(self) -> usize {
        self.source_words
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WitnessFeedSourceUploadBinding {
    contract_identity: [u8; 32],
    arena_identity: usize,
    exec_context_token: u64,
    source_slot: ArenaSlotId,
    source_address: usize,
    source_words: usize,
}

impl WitnessFeedSourceUploadBinding {
    pub(super) fn new(
        arena: &DeviceArena,
        contract: &WitnessFeedContract,
        source: ArenaSlice,
    ) -> Self {
        Self {
            contract_identity: contract.identity(),
            arena_identity: arena.base_ptr().as_ptr() as usize,
            exec_context_token: arena.exec_context_token(),
            source_slot: source.id(),
            source_address: source.as_u32_ptr() as usize,
            source_words: source.len_words(),
        }
    }
}

pub(super) fn next_source_upload_generation(current: u64) -> Result<u64, PreparedWitnessFeedError> {
    current
        .checked_add(1)
        .ok_or(PreparedWitnessFeedError::SourceUploadGenerationOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> WitnessFeedSourceUploadBinding {
        WitnessFeedSourceUploadBinding {
            contract_identity: [7; 32],
            arena_identity: 11,
            exec_context_token: 13,
            source_slot: ArenaSlotId(17),
            source_address: 19,
            source_words: 3,
        }
    }

    #[test]
    fn receipt_binds_exact_runtime_source_and_generation() {
        let binding = binding();
        let first = WitnessFeedSourceUploadReceipt::prepare(binding, &[1, 2, 3], 1).unwrap();
        let second = WitnessFeedSourceUploadReceipt::prepare(binding, &[4, 5, 6], 2).unwrap();

        assert!(first.matches(binding, 1));
        assert!(!first.matches(binding, 2));
        assert!(second.matches(binding, 2));
        assert_eq!(second.contract_identity(), [7; 32]);
        assert_eq!(second.arena_identity(), 11);
        assert_eq!(second.exec_context_token(), 13);
        assert_eq!(second.source_slot(), ArenaSlotId(17));
        assert_eq!(second.source_address(), 19);
        assert_eq!(second.source_words(), 3);
        assert_eq!(second.generation(), 2);

        let mut wrong_source = binding;
        wrong_source.source_slot = ArenaSlotId(18);
        assert!(!second.matches(wrong_source, 2));
        assert_eq!(
            WitnessFeedSourceUploadReceipt::prepare(binding, &[1, 2], 3),
            Err(PreparedWitnessFeedError::SlotSizeMismatch {
                slot: ArenaSlotId(17),
                expected_words: 3,
                actual_words: 2,
            })
        );
    }

    #[test]
    fn generation_is_monotonic_and_overflow_fails_closed() {
        assert_eq!(next_source_upload_generation(0).unwrap(), 1);
        assert_eq!(next_source_upload_generation(41).unwrap(), 42);
        assert_eq!(
            next_source_upload_generation(u64::MAX),
            Err(PreparedWitnessFeedError::SourceUploadGenerationOverflow)
        );
        assert_eq!(
            WitnessFeedSourceUploadReceipt::prepare(binding(), &[1, 2, 3], 0),
            Err(PreparedWitnessFeedError::SourceUploadGenerationOverflow)
        );
    }
}
