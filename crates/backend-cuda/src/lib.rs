//! CUDA proving backend for stwo, ported from the stwo-cuda prototype and adapted to
//! this repository's backend extension points. Compile-gated: without nvcc the kernels
//! crate provides panicking stubs, so this crate builds everywhere but only proves on
//! a CUDA machine. Conformance gate: `stwo_backend_testkit::assert_backend_conformance`
//! (proof byte-equality vs CpuBackend on both Blake2s channels), run on a CUDA box.

mod backend;
mod columns;

pub use backend::{
    blake_witness, exec_tables, finalize_raw_logup, jit_witness, logup_pairs, memory_witness,
    pedersen_witness, CudaBackend,
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
