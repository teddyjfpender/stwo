// Foundation primitive (design §19); wired into the streamful commit island +
// two-proof scheduler in the following increments.
#![allow(dead_code)]
//! Resource-owning CUDA execution context (design §19) — one per resident proof.
//!
//! The core prove path is stream-0-centric: every kernel and every pool alloc/free
//! is ordered on the legacy default stream. To run two proofs concurrently (M6) —
//! and, later, to give a single proof's streamful commit island its own stream — each
//! proof needs an independent stream AND its own stream-ordered pool. A context is
//! NOT a bare stream pointer: it owns the stream and a never-release memory pool, so
//! buffers allocated in its scope free stream-ordered AFTER their last use on that
//! stream. Two contexts allocate from DISJOINT pools, so there is no cross-stream
//! reuse hazard from the shared global default pool (the central concurrency hazard),
//! and no stream-0 `cudaFreeAsync` ever races another stream's kernels.

use core::ffi::c_void;

/// An owned CUDA stream + its own memory pool. Driven by a single host thread (the
/// proof's owner); `Drop` synchronizes the stream and tears down stream + pool.
pub struct CudaExecContext {
    handle: *mut c_void,
}

impl CudaExecContext {
    /// Create a context (non-blocking stream + own never-release pool). Returns
    /// `None` on the stub build (no CUDA) or if the native side cannot create the
    /// stream — callers fall back to the stream-0 path.
    pub fn new() -> Option<Self> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return None;
        }
        // SAFETY: FFI; returns null on failure (guarded below).
        let handle = unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_create() };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// The context's CUDA stream as an opaque handle, to pass to `_on(stream)` kernel
    /// variants (`null` = the legacy default stream on those entry points).
    pub fn stream(&self) -> *mut c_void {
        // SAFETY: FFI on a valid handle.
        unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_stream(self.handle) }
    }

    /// Block the host until all work enqueued on the context's stream has completed.
    pub fn sync(&self) {
        // SAFETY: FFI on a valid handle.
        unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_sync(self.handle) };
    }

    /// Allocate `count` u32 from the context's pool, ordered on its stream. The
    /// pointer is usable by later work on the SAME stream. Returns null on failure.
    /// Must be released via [`Self::free_u32`] on this context so the free is
    /// stream-ordered on the same stream.
    pub fn alloc_u32(&self, count: usize) -> *mut u32 {
        // SAFETY: FFI on a valid handle.
        unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_alloc_u32(self.handle, count) }
    }

    /// Free a context-allocated buffer, ordered on the context's stream.
    ///
    /// # Safety
    /// `ptr` must have come from [`Self::alloc_u32`] on this same context and not
    /// have been freed already.
    pub unsafe fn free_u32(&self, ptr: *mut u32) {
        stwo_backend_cuda_kernels::raw::stwo_exec_context_free_u32(self.handle, ptr)
    }
}

impl Drop for CudaExecContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Native side synchronizes the stream before destroying stream + pool, so
            // no pending free/kernel outlives the resources it references.
            // SAFETY: FFI on a valid handle; called exactly once (null-checked).
            unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_destroy(self.handle) };
            self.handle = core::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_none_without_cuda() {
        // On the stub build there is no CUDA runtime; `new` must return None (guarded
        // before any FFI) rather than abort in the stub symbol.
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            assert!(CudaExecContext::new().is_none());
        }
    }
}
