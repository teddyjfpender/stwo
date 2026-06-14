//! CUDA proving backend for stwo, ported from the stwo-cuda prototype and adapted to
//! this repository's backend extension points. Compile-gated: without nvcc the kernels
//! crate provides panicking stubs, so this crate builds everywhere but only proves on
//! a CUDA machine. Conformance gate: `stwo_backend_testkit::assert_backend_conformance`
//! (proof byte-equality vs CpuBackend on both Blake2s channels), run on a CUDA box.

mod backend;
mod columns;

pub use backend::{finalize_raw_logup, memory_witness, CudaBackend};
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

/// True peak reserved VRAM (high-water mark) of this process, in bytes; 0 without CUDA.
///
/// Unlike [`gpu_memory_info`] (which reports the instantaneous footprint and so
/// only sees end-of-run state), this is the maximum the allocator ever reserved,
/// so it captures the real peak even after a [`release_mem_pool`] trim has shrunk
/// the live footprint.
pub fn gpu_peak_vram_bytes() -> u64 {
    columns::bindings::vram_peak_bytes()
}

/// Releases pooled-but-unused device memory back to the OS (`cudaMemPoolTrimTo`).
///
/// The mem pool's never-release threshold is unchanged: this is an EXPLICIT,
/// opt-in trim. It is invoked from the low-memory spill point (see
/// [`stwo::prover::backend::ColumnOps::release_pooled_memory`]) and must run after
/// the freed columns' stream-ordered frees are enqueued. No-op without CUDA.
pub fn release_mem_pool() {
    columns::bindings::trim_mem_pool();
}
