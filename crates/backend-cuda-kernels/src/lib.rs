//! Compile-gated CUDA kernel library — the staging crate for a future `stwo-backend-cuda`.
//!
//! See the crate README for the kernel-to-trait mapping, the API delta against the
//! prototype these kernels came from, and the known issues to address while wiring up.
//!
//! On a machine with `nvcc`, building this crate compiles every kernel under `cuda/`
//! into a static archive (the first validation gate for the staged sources). Without
//! `nvcc` it builds as a stub, so the workspace never requires a CUDA toolkit.
//!
//! No FFI bindings are exposed yet: bindings should be written together with the
//! `stwo-backend-cuda` trait implementations and validated against
//! `stwo-backend-testkit` (proof byte-equality on both Blake2s channels), mirroring
//! `stwo-backend-metal`.

pub mod aot_pack;
pub mod raw;
#[cfg(not(stwo_cuda_link))]
mod stubs;

/// True when the CUDA kernels were compiled and linked into this build.
pub const CUDA_KERNELS_BUILT: bool = cfg!(stwo_cuda_link);

/// `"cuda"` when the kernels were compiled, `"no-cuda"` for the stub build.
pub const BUILD_MODE: &str = env!("STWO_CUDA_BUILD_MODE");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_mode_is_consistent() {
        assert_eq!(BUILD_MODE == "cuda", CUDA_KERNELS_BUILT);
    }

    #[test]
    fn prepared_quotient_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_combine_quotients_from_numerators_on as usize, 0);
        assert_ne!(raw::stwo_ntt_b2n_columns_on as usize, 0);
        assert_ne!(raw::stwo_lde_n2b_columns_on as usize, 0);
    }
}
