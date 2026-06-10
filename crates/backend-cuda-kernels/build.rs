//! Build script for `stwo-backend-cuda-kernels`.
//!
//! Compile-gated: when `nvcc` is available (on `PATH`, or via the `STWO_CUDA_NVCC` env
//! var), every kernel under `cuda/` is compiled into a static archive and linked into the
//! crate, so `cargo build -p stwo-backend-cuda-kernels` on a CUDA machine validates that
//! the staged kernels compile. Without `nvcc` the crate builds as a stub and the rest of
//! the workspace is unaffected — no CUDA toolkit is required to build or test stwo.
//!
//! Tunables:
//! - `STWO_CUDA_NVCC`: path to the nvcc binary (default: `nvcc` from `PATH`)
//! - `STWO_CUDA_ARCH`: value for `-arch` (default: `native`)
//! - `STWO_CUDA_NVCC_FLAGS`: extra whitespace-separated flags appended to every call
//!
//! The kernels are compiled with `-rdc=true` (they cross-reference `fields.cu` across
//! translation units) and `-dlto`, so device link-time optimization can inline field ops
//! at the final device link — see "Known issues" in the README.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(stwo_cuda_link)");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_NVCC_FLAGS");

    let sources = kernel_sources();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    let nvcc = env::var("STWO_CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let nvcc_available = Command::new(&nvcc)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    if !nvcc_available {
        println!("cargo:rustc-env=STWO_CUDA_BUILD_MODE=no-cuda");
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let arch = env::var("STWO_CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    let extra_flags: Vec<String> = env::var("STWO_CUDA_NVCC_FLAGS")
        .map(|flags| flags.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();

    let archive = out_dir.join("libstwo_cuda_kernels.a");
    let mut compile = Command::new(&nvcc);
    compile
        .arg("-lib")
        .arg("-rdc=true")
        .arg("-dlto")
        .arg("-O3")
        .arg("--std=c++17")
        .arg(format!("-arch={arch}"))
        .args(&extra_flags)
        .args(&sources)
        .arg("-o")
        .arg(&archive);
    let output = compile
        .output()
        .expect("nvcc was detected but could not be launched");
    assert!(
        output.status.success(),
        "nvcc failed to compile the CUDA kernels:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    println!("cargo:rustc-env=STWO_CUDA_BUILD_MODE=cuda");
    println!("cargo:rustc-cfg=stwo_cuda_link");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=stwo_cuda_kernels");
    println!("cargo:rustc-link-lib=cudart");
}

fn kernel_sources() -> Vec<PathBuf> {
    let cuda_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("cuda");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&cuda_dir)
        .expect("cuda/ kernel directory must exist")
        .filter_map(|entry| {
            let path = entry.expect("readable cuda/ directory entry").path();
            (path.extension().is_some_and(|ext| ext == "cu")).then_some(path)
        })
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no .cu kernels found under cuda/");
    sources
}
