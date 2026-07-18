//! Address-free compiler authority for the unfused direct-retained Base commit.
//!
//! The input programs are the only schedule authority. The compiler replaces
//! each ordinary `Lde` with an explicit B2N/N2B pair and preserves every
//! state, absorb, finalize, and Merkle operation in exact order. Fused leaf,
//! interior, and tail variants fail closed until they have equally explicit
//! effects and shard binders.

use super::*;
use crate::backend::prepared_decommit::TraceTreeRole;
use crate::backend::prepared_witness_input::static_build::{
    bind_static_build, StaticBuildBindError, StaticBuildBinding,
};

mod abi;
mod compiler;
mod effect;
mod encoding;
mod invocation;

pub use abi::{BaseCommitAbiAccess, BaseCommitAbiArgument, BaseCommitAbiArgumentKind};
use compiler::{u32_value, Compiler};
pub use effect::{
    BaseCommitAccess, BaseCommitAccessKind, BaseCommitAliasAuthority, BaseCommitAliasDiscipline,
    BaseCommitAliasRequirement, BaseCommitDependencyRange, BaseCommitDependencyRole,
    BaseCommitEffect, BaseCommitInstalledAccess, BaseCommitPointerBinding, BaseCommitPointerTarget,
};
use encoding::{linked_authority, program_identity, source_identity};
pub use invocation::{
    BaseCommitInvocation, BaseCommitInvocationArgument, BaseCommitInvocationValue,
};

const HASH_WORDS: usize = core::mem::size_of::<Blake2sHash>() / core::mem::size_of::<u32>();
const STATE_WORDS: usize = PROGRESSIVE_BLAKE2S_STATE_STRIDE_BYTES / core::mem::size_of::<u32>();
const ZERO_IDENTITY: [u8; 32] = [0; 32];
const SOURCE_DOMAIN: &[u8] = b"stwo-cuda-base-commit-source-v1\0";
const ABI_DOMAIN: &[u8] = b"stwo-cuda-base-commit-abi-v1\0";
const EFFECT_DOMAIN: &[u8] = b"stwo-cuda-base-commit-effect-v1\0";
const LAUNCH_DOMAIN: &[u8] = b"stwo-cuda-base-commit-launch-v1\0";
const OPERATION_DOMAIN: &[u8] = b"stwo-cuda-base-commit-operation-v1\0";
const PROGRAM_DOMAIN: &[u8] = b"stwo-cuda-base-commit-program-v1\0";
const STATIC_BUILD_DOMAIN: &[u8] = b"stwo-cuda-base-commit-static-build-v1\0";
const LINKED_DOMAIN: &[u8] = b"stwo-cuda-base-commit-linked-v1\0";

const AUTHORITY_SOURCE: &[u8] = include_bytes!("base_commit_authority.rs");
const ABI_AUTHORITY_SOURCE: &[u8] = include_bytes!("base_commit_authority/abi.rs");
const COMPILER_AUTHORITY_SOURCE: &[u8] = include_bytes!("base_commit_authority/compiler.rs");
const EFFECT_AUTHORITY_SOURCE: &[u8] = include_bytes!("base_commit_authority/effect.rs");
const EFFECT_VALIDATION_SOURCE: &[u8] =
    include_bytes!("base_commit_authority/effect/validation.rs");
const ENCODING_AUTHORITY_SOURCE: &[u8] = include_bytes!("base_commit_authority/encoding.rs");
const INVOCATION_AUTHORITY_SOURCE: &[u8] = include_bytes!("base_commit_authority/invocation.rs");
const COMMIT_SOURCE: &[u8] = include_bytes!("program.rs");
const DIRECT_SOURCE: &[u8] = include_bytes!("direct_retained_b2n.rs");
const DIRECT_LAUNCH_SOURCE: &[u8] = include_bytes!("direct_retained_b2n/launch.rs");
const PREPARED_COMMIT_BINDER_SOURCE: &[u8] = include_bytes!("../prepared_progressive_commit.rs");
const IN_PLACE_PLANNER_SOURCE: &[u8] = include_bytes!("../progressive_commit_in_place.rs");
const PROGRESSIVE_COMMIT_SOURCE: &[u8] = include_bytes!("../progressive_commit.rs");
const NTT_LEAF_FUSION_SOURCE: &[u8] = include_bytes!("../progressive_ntt_leaf_fusion.rs");
const RAW_FFI_SOURCE: &[u8] = include_bytes!("../../../../backend-cuda-kernels/src/raw.rs");
const B2N_SOURCE: &[u8] = include_bytes!("../../../../backend-cuda-kernels/cuda/ifft.cu");
const N2B_SOURCE: &[u8] = include_bytes!("../../../../backend-cuda-kernels/cuda/rfft.cu");
const BLAKE_SOURCE: &[u8] = include_bytes!("../../../../backend-cuda-kernels/cuda/blake2s.cu");
const IN_PLACE_SOURCE: &[u8] =
    include_bytes!("../../../../backend-cuda-kernels/cuda/progressive_commit_in_place.cu");

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BaseCommitValueRole {
    SourceEvaluation { canonical_column: u32 },
    RetainedStageTwo { canonical_column: u32 },
    RetainedEvaluation { canonical_column: u32 },
    State { version: u32, log_size: u32 },
    HashLayer { log_size: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaseCommitLayout {
    pub role: BaseCommitValueRole,
    pub rows: usize,
    pub words_per_row: usize,
    pub logical_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BaseCommitPartitionAuthority {
    Monolithic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BaseCommitAbi {
    DirectB2nV1,
    DirectN2bV1,
    StateInitV1,
    StateExpandInPlaceV1,
    StateAbsorbV1,
    StateFinalizeInPlaceV1,
    MerkleLayerInPlaceV1,
    MerkleLayerV1,
}

impl BaseCommitAbi {
    const ALL: [Self; 8] = [
        Self::DirectB2nV1,
        Self::DirectN2bV1,
        Self::StateInitV1,
        Self::StateExpandInPlaceV1,
        Self::StateAbsorbV1,
        Self::StateFinalizeInPlaceV1,
        Self::MerkleLayerInPlaceV1,
        Self::MerkleLayerV1,
    ];

    pub const fn wrapper_symbol(self) -> &'static str {
        match self {
            Self::DirectB2nV1 => "stwo_ntt_b2n_columns_to_retained_on",
            Self::DirectN2bV1 => "stwo_ntt_n2b_columns_from_stage_two_on",
            Self::StateInitV1 => "stwo_blake2s_progressive_init_on",
            Self::StateExpandInPlaceV1 => "stwo_blake2s_progressive_expand_in_place_on",
            Self::StateAbsorbV1 => "stwo_blake2s_progressive_absorb_on",
            Self::StateFinalizeInPlaceV1 => "stwo_blake2s_progressive_finalize_in_place_on",
            Self::MerkleLayerInPlaceV1 => "stwo_blake2s_layer_in_place_on",
            Self::MerkleLayerV1 => "stwo_blake2s_layer_on",
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::DirectB2nV1 => 1,
            Self::DirectN2bV1 => 2,
            Self::StateInitV1 => 3,
            Self::StateExpandInPlaceV1 => 4,
            Self::StateAbsorbV1 => 5,
            Self::StateFinalizeInPlaceV1 => 6,
            Self::MerkleLayerInPlaceV1 => 7,
            Self::MerkleLayerV1 => 8,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BaseCommitOperationKind {
    DirectB2n {
        batch_index: u32,
        source_log_size: u32,
        retained_log_size: u32,
        canonical_columns: Vec<u32>,
    },
    DirectN2b {
        batch_index: u32,
        source_log_size: u32,
        retained_log_size: u32,
        canonical_columns: Vec<u32>,
    },
    StateInit {
        log_size: u32,
    },
    StateExpandInPlace {
        from_log_size: u32,
        to_log_size: u32,
        absorbed_columns: u32,
        bands: u32,
    },
    StateAbsorb {
        batch_index: u32,
        log_size: u32,
        absorbed_columns_before: u32,
        canonical_columns: Vec<u32>,
    },
    StateFinalizeInPlace {
        log_size: u32,
        absorbed_columns: u32,
        bands: u32,
    },
    MerkleLayerInPlace {
        level: u32,
        output_hashes: u32,
        bands: u32,
    },
    MerkleLayer {
        level: u32,
        output_hashes: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BaseCommitOperation {
    pub kind: BaseCommitOperationKind,
    pub abi: BaseCommitAbi,
    pub effect: BaseCommitEffect,
    pub invocation: BaseCommitInvocation,
    pub partition: BaseCommitPartitionAuthority,
    pub nested_kernel_launches: u32,
    pub nested_device_copies: u32,
    pub source_identity: [u8; 32],
    pub abi_identity: [u8; 32],
    pub launch_identity: [u8; 32],
    pub identity: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaseCommitRetainedEvaluation {
    pub canonical_column: u32,
    pub role: BaseCommitValueRole,
    pub words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaseCommitRetainedLayer {
    pub log_size: u32,
    pub role: BaseCommitValueRole,
    pub words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BaseCommitProgramAuthority {
    commit: CommitProgram,
    direct: DirectRetainedB2nProgram,
    layouts: Vec<BaseCommitLayout>,
    operations: Vec<BaseCommitOperation>,
    retained_evaluations: Vec<BaseCommitRetainedEvaluation>,
    retained_layers_bottom_up: Vec<BaseCommitRetainedLayer>,
    root: BaseCommitValueRole,
    source_identity: [u8; 32],
    identity: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaseCommitLinkedAuthority {
    program_identity: [u8; 32],
    module_build_identity: [u8; 32],
    static_build_source_identity: [u8; 32],
    static_build_identity: [u8; 32],
    target_sm: u32,
    sm_identity: [u8; 32],
    identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BaseCommitAuthorityError {
    UnsupportedRole(TraceTreeRole),
    ProgramMismatch,
    UnsupportedStorage,
    UnsupportedLeafFusion,
    UnsupportedInteriorFusion,
    UnsupportedTailFusion,
    UnsupportedOperation(CommitProgramOperation),
    InvalidProgramOrder,
    InvalidRetainedOutput,
    MissingStaticSourceIdentity,
    InvalidStaticAbi(&'static str),
    InvalidEffect,
    InvalidInvocation,
    SizeOverflow,
    StaticBuildUnavailable,
    StaticBuildMismatch,
    UnsupportedTargetSm(u32),
}

impl core::fmt::Display for BaseCommitAuthorityError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid direct Base commit authority: {self:?}")
    }
}

impl std::error::Error for BaseCommitAuthorityError {}

impl From<StaticBuildBindError> for BaseCommitAuthorityError {
    fn from(error: StaticBuildBindError) -> Self {
        match error {
            StaticBuildBindError::UnsupportedTargetSm(sm) => Self::UnsupportedTargetSm(sm),
            StaticBuildBindError::BuildReceiptUnavailable => Self::StaticBuildUnavailable,
            StaticBuildBindError::InconsistentBuildMetadata
            | StaticBuildBindError::BuildReceiptMismatch => Self::StaticBuildMismatch,
        }
    }
}

impl BaseCommitProgramAuthority {
    pub fn compile(
        commit: &CommitProgram,
        direct: &DirectRetainedB2nProgram,
    ) -> Result<Self, BaseCommitAuthorityError> {
        if direct.role() != TraceTreeRole::Base {
            return Err(BaseCommitAuthorityError::UnsupportedRole(direct.role()));
        }
        if direct.commit_cache_key() != commit.identity().cache_key {
            return Err(BaseCommitAuthorityError::ProgramMismatch);
        }
        if commit.identity().storage != ProgressiveCommitStorageMode::InPlaceSlab {
            return Err(BaseCommitAuthorityError::UnsupportedStorage);
        }
        if commit.identity().ntt_leaf_fusion != ProgressiveNttLeafFusionMode::Separate {
            return Err(BaseCommitAuthorityError::UnsupportedLeafFusion);
        }
        if commit.identity().interior4_fused {
            return Err(BaseCommitAuthorityError::UnsupportedInteriorFusion);
        }
        if commit.identity().config.max_fused_tail_levels != 0 {
            return Err(BaseCommitAuthorityError::UnsupportedTailFusion);
        }
        for abi in BaseCommitAbi::ALL {
            if !abi.source_declares_entry(abi.wrapper_source()) {
                return Err(BaseCommitAuthorityError::InvalidStaticAbi(
                    abi.wrapper_symbol(),
                ));
            }
        }

        let source_identity = source_identity(commit, direct)?;
        let mut compiler = Compiler::new(commit, direct, source_identity);
        compiler.compile_steps()?;
        let (layouts, retained_evaluations, retained_layers_bottom_up, root) =
            compiler.finish_values()?;
        let identity = program_identity(
            commit,
            direct,
            source_identity,
            &layouts,
            &compiler.operations,
            &retained_evaluations,
            &retained_layers_bottom_up,
            root,
        )?;
        Ok(Self {
            commit: commit.clone(),
            direct: direct.clone(),
            layouts,
            operations: compiler.operations,
            retained_evaluations,
            retained_layers_bottom_up,
            root,
            source_identity,
            identity,
        })
    }

    pub fn validate(&self) -> Result<(), BaseCommitAuthorityError> {
        (Self::compile(&self.commit, &self.direct)? == *self)
            .then_some(())
            .ok_or(BaseCommitAuthorityError::ProgramMismatch)
    }

    pub fn bind_static_build(
        &self,
        target_sm: u32,
    ) -> Result<Option<BaseCommitLinkedAuthority>, BaseCommitAuthorityError> {
        self.validate()?;
        bind_static_build(STATIC_BUILD_DOMAIN, self.identity, target_sm)
            .map(|binding| binding.map(|binding| linked_authority(self.identity, binding)))
            .map_err(Into::into)
    }

    pub const fn commit(&self) -> &CommitProgram {
        &self.commit
    }
    pub const fn direct(&self) -> &DirectRetainedB2nProgram {
        &self.direct
    }
    pub fn layouts(&self) -> &[BaseCommitLayout] {
        &self.layouts
    }
    pub fn operations(&self) -> &[BaseCommitOperation] {
        &self.operations
    }
    pub fn retained_evaluations(&self) -> &[BaseCommitRetainedEvaluation] {
        &self.retained_evaluations
    }
    pub fn retained_layers_bottom_up(&self) -> &[BaseCommitRetainedLayer] {
        &self.retained_layers_bottom_up
    }
    pub const fn root(&self) -> BaseCommitValueRole {
        self.root
    }
    pub const fn source_identity(&self) -> [u8; 32] {
        self.source_identity
    }
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Project the explicit B2N/N2B pair back to the source program's single
    /// LDE entry. This is a parity view only; execution remains decomposed.
    pub fn commit_operation_view(
        &self,
    ) -> Result<Vec<CommitProgramOperation>, BaseCommitAuthorityError> {
        let mut view = Vec::new();
        for operation in &self.operations {
            let source = match &operation.kind {
                BaseCommitOperationKind::DirectB2n { .. } => continue,
                BaseCommitOperationKind::DirectN2b {
                    batch_index,
                    retained_log_size,
                    canonical_columns,
                    ..
                } => CommitProgramOperation::Lde {
                    batch_index: *batch_index,
                    segment_offset: 0,
                    columns: u32_value(canonical_columns.len())?,
                    log_size: *retained_log_size,
                },
                BaseCommitOperationKind::StateInit { log_size } => {
                    CommitProgramOperation::StateInit {
                        log_size: *log_size,
                    }
                }
                BaseCommitOperationKind::StateExpandInPlace {
                    from_log_size,
                    to_log_size,
                    absorbed_columns,
                    bands,
                } => CommitProgramOperation::StateExpandInPlace {
                    from_log_size: *from_log_size,
                    to_log_size: *to_log_size,
                    absorbed_columns: *absorbed_columns,
                    bands: *bands,
                },
                BaseCommitOperationKind::StateAbsorb {
                    batch_index,
                    log_size,
                    absorbed_columns_before,
                    canonical_columns,
                } => CommitProgramOperation::Absorb {
                    batch_index: *batch_index,
                    segment_offset: 0,
                    columns: u32_value(canonical_columns.len())?,
                    log_size: *log_size,
                    absorbed_columns_before: *absorbed_columns_before,
                },
                BaseCommitOperationKind::StateFinalizeInPlace {
                    log_size,
                    absorbed_columns,
                    bands,
                } => CommitProgramOperation::FinalizeInPlace {
                    log_size: *log_size,
                    absorbed_columns: *absorbed_columns,
                    bands: *bands,
                },
                BaseCommitOperationKind::MerkleLayerInPlace {
                    level,
                    output_hashes,
                    bands,
                } => CommitProgramOperation::MerkleLayerInPlace {
                    level: *level,
                    output_hashes: *output_hashes,
                    bands: *bands,
                },
                BaseCommitOperationKind::MerkleLayer {
                    level,
                    output_hashes,
                } => CommitProgramOperation::MerkleLayer {
                    level: *level,
                    output_hashes: *output_hashes,
                },
            };
            view.push(source);
        }
        Ok(view)
    }
}

impl BaseCommitLinkedAuthority {
    pub fn validate(
        &self,
        program: &BaseCommitProgramAuthority,
    ) -> Result<(), BaseCommitAuthorityError> {
        let expected = program
            .bind_static_build(self.target_sm)?
            .ok_or(BaseCommitAuthorityError::StaticBuildUnavailable)?;
        (*self == expected)
            .then_some(())
            .ok_or(BaseCommitAuthorityError::StaticBuildMismatch)
    }

    pub const fn program_identity(&self) -> [u8; 32] {
        self.program_identity
    }
    pub const fn module_build_identity(&self) -> [u8; 32] {
        self.module_build_identity
    }
    pub const fn static_build_source_identity(&self) -> [u8; 32] {
        self.static_build_source_identity
    }
    pub const fn static_build_identity(&self) -> [u8; 32] {
        self.static_build_identity
    }
    pub const fn target_sm(&self) -> u32 {
        self.target_sm
    }
    pub const fn sm_identity(&self) -> [u8; 32] {
        self.sm_identity
    }
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

#[cfg(test)]
#[path = "base_commit_authority_tests.rs"]
mod tests;
