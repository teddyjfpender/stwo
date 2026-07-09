//! CUDA proving backend for stwo, ported from the stwo-cuda prototype and adapted to
//! this repository's backend extension points. Compile-gated: without nvcc the kernels
//! crate provides panicking stubs, so this crate builds everywhere but only proves on
//! a CUDA machine. Conformance gate: `stwo_backend_testkit::assert_backend_conformance`
//! (proof byte-equality vs CpuBackend on both Blake2s channels), run on a CUDA box.

mod backend;

pub use backend::aot;
pub use backend::commit_graph::{
    CommitGraphError, CommitGraphPlan, CommitLaunchKind, CommitLdeBatch, CommitLeafGroup,
    CommitTailPlan,
};
pub use backend::decommit_gather::{
    column_row_gather_requirements, gather_column_rows_host, ColumnRowGatherError,
    ColumnRowGatherRequirements, ColumnRowGatherSlots, PreparedColumnRowGather,
};
pub use backend::device_transcript::{
    replay_blake2s_reference, Blake2sTranscriptRequirements, Blake2sTranscriptSchedule,
    Blake2sTranscriptWorkspaceSlots, DeviceTranscriptError, PreparedBlake2sTranscript,
    TranscriptArenaSlotRequirement, TranscriptBoundaryId, TranscriptBoundaryState,
    TranscriptInputBinding, TranscriptInputId, TranscriptIoRequirement, TranscriptMirrorReport,
    TranscriptOperation, TranscriptOutputBinding, TranscriptOutputId, TranscriptReferenceTrace,
    TranscriptSegmentCursor, TranscriptSegmentStart, TranscriptStart,
    BLAKE2S_TRANSCRIPT_ALIGNMENT_WORDS, BLAKE2S_TRANSCRIPT_PROTOCOL_TAG,
    BLAKE2S_TRANSCRIPT_STATE_WORDS,
};
pub use backend::exec_context::{
    ArenaError, ArenaLayout, ArenaSlice, ArenaSlotId, ArenaSlotSpec, CudaExecContext,
    CudaGraphCapture, CudaGraphExec, CudaRuntimeError, DeviceArena,
};
pub use backend::pcs_driver::{
    prove_values_with_config as prove_cuda_pcs_values, CudaPcsDriverConfig, CudaPcsDriverError,
    CudaPcsDriverOutput, CudaPcsDriverTelemetry, CudaPcsGraphHookError, CudaPcsGraphHooks,
    CudaPcsRuntimeMode,
};
pub use backend::prepared_commit::{
    commit_workspace_requirements, CommitArenaSlotRequirement, CommitBatchRequirements,
    CommitBatchSlots, CommitCoefficientColumn, CommitCoefficientGroup, CommitGroupRequirements,
    CommitGroupSlots, CommitLayerRequirements, CommitWorkspaceConfig, CommitWorkspaceRequirements,
    CommitWorkspaceSlots, PreparedCommitError, PreparedCommitGraph, COMMIT_HASH_ALIGNMENT_WORDS,
    COMMIT_POINTER_ALIGNMENT_WORDS,
};
pub use backend::prepared_fri::{
    fri_workspace_requirements, FriArenaSlotRequirement, FriMerkleLayerRequirements,
    FriMerkleTreeRequirements, FriMerkleTreeSlots, FriRoundRequirements, FriWorkspaceConfig,
    FriWorkspaceRequirements, FriWorkspaceSlots, PreparedFriError, PreparedFriEvaluation,
    PreparedFriGraph, FRI_CHALLENGE_WORDS, FRI_HASH_ALIGNMENT_WORDS, FRI_POINTER_ALIGNMENT_WORDS,
};
pub use backend::prepared_quotient::{
    quotient_workspace_requirements, PreparedQuotientError, PreparedQuotientGraph,
    QuotientArenaSlotRequirement, QuotientNumeratorSource, QuotientSampleConstants,
    QuotientWorkspaceConfig, QuotientWorkspaceRequirements, QuotientWorkspaceSlots,
    QUOTIENT_POINTER_ALIGNMENT_WORDS,
};
pub use backend::relation_graph::{
    relation_graph_requirements, PreparedRelationGraph, PreparedRelationOutput,
    RelationArenaSlotRequirement, RelationBatchProgram, RelationChallenges,
    RelationColumnDescriptor, RelationGraphError, RelationGraphRequirements, RelationGraphSlots,
    RelationInstanceRequirement, RelationInstanceSlots, RelationInstanceSources,
    RelationKernelProgram, RelationMultiplicityKind, RelationRowExtent, RelationSourceLayout,
    RelationTupleKind, RelationUseDescriptor, RELATION_POINTER_ALIGNMENT_WORDS,
};
mod columns;

pub use backend::{
    blake_witness, exec_tables, finalize_raw_logup, jit_witness, logup_pairs, memory_witness,
    pedersen_table, pedersen_witness, CudaBackend,
};
pub use columns::{BaseFieldVec, Blake2sHashVec, SecureFieldVec};

/// (free_bytes, total_bytes) of GPU memory; (0, 0) without CUDA.
pub fn gpu_memory_info() -> (usize, usize) {
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return (0, 0);
    }
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe { columns::bindings::cuda_get_memory_info(&mut free, &mut total) };
    (free, total)
}

/// Never-release-pool high-water marks since process start, in bytes:
/// (peak allocated in flight, peak reserved from the device). Driver-maintained
/// and exact, unlike sampler-based probes (the harness's 25ms sampler measured
/// up to 11GB low on SN_PIE_2); (0, 0) without CUDA. The VRAM-diet metric.
pub fn gpu_pool_highwater() -> (usize, usize) {
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return (0, 0);
    }
    let mut used = 0usize;
    let mut reserved = 0usize;
    unsafe { columns::bindings::cuda_pool_highwater(&mut used, &mut reserved) };
    (used, reserved)
}

/// Reset the pool high-water marks to current usage — per-phase VRAM
/// attribution (design §1.1 R5): read + reset at phase boundaries.
pub fn gpu_pool_highwater_reset() {
    if stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        unsafe { columns::bindings::cuda_pool_highwater_reset() };
    }
}

/// Fence work issued by the migration-era CUDA backend default stream before
/// handing its buffers to an isolated [`CudaExecContext`].
///
/// The resident prover does not launch proof work on the legacy stream.  This
/// boundary exists solely while witness producers still return owning
/// `BaseFieldVec`s allocated by the old backend; once those producers target
/// arena slices directly this function and the associated D2D staging copy are
/// removed.  Stub builds have no device work to fence.
pub fn synchronize_legacy_stream_for_arena_handoff() {
    if stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        // The native wrapper checks the CUDA status and aborts rather than
        // allowing a failed producer to race an arena consumer.
        unsafe { columns::bindings::stwo_legacy_stream_sync() };
    }
}
