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
    // Track the whole cuda/ tree (cargo traverses directories recursively): headers
    // (.cuh) are compiled into every translation unit, so an edit there must dirty
    // the build just like a .cu edit — otherwise stale objects with old symbol
    // signatures survive in the archive.
    println!("cargo:rerun-if-changed=cuda");

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
    // STWO_CUDA_ARCH accepts a comma list (e.g. "sm_86,sm_90") to build a fat binary
    // that runs on multiple GPU generations — one release artifact for 3090 and H100.
    let archs: Vec<String> = env::var("STWO_CUDA_ARCH")
        .unwrap_or_else(|_| detect_arch())
        .split(',')
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    let gencode_flags: Vec<String> = archs
        .iter()
        .flat_map(|arch| {
            let num = arch.trim_start_matches("sm_");
            [
                "-gencode".to_string(),
                format!("arch=compute_{num},code=sm_{num}"),
            ]
        })
        .collect();
    let extra_flags: Vec<String> = env::var("STWO_CUDA_NVCC_FLAGS")
        .map(|flags| flags.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();

    // Separable compilation (-rdc=true) requires an explicit device-link step: the
    // final Rust link knows nothing about CUDA, so the device-link object must be in
    // the archive. Pipeline: each .cu -> .o (-dc), then nvcc -dlink over all objects,
    // then everything into one archive.
    let run_nvcc = |args: &mut Command| {
        let output = args
            .output()
            .expect("nvcc was detected but could not be launched");
        assert!(
            output.status.success(),
            "nvcc failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };

    // Every kernel directory is an include root: the generated headers include each
    // other by bare name regardless of subdirectory.
    let mut include_dirs: Vec<String> = vec!["cuda".to_string()];
    collect_dirs(std::path::Path::new("cuda"), &mut include_dirs);
    include_dirs.sort();

    let mut objects: Vec<PathBuf> = Vec::with_capacity(sources.len() + 1);
    for source in &sources {
        let object = out_dir.join(format!(
            "{}.o",
            source
                .file_stem()
                .expect("kernel file stem")
                .to_string_lossy()
        ));
        run_nvcc(
            Command::new(&nvcc)
                .arg("-dc")
                .arg("-O3")
                .arg("--std=c++17")
                // The fp256/poseidon252 stack calls `constexpr __host__` accessors from
                // device code (sppark lineage); nvcc requires this flag for that pattern.
                .arg("--expt-relaxed-constexpr")
                .args(
                    include_dirs
                        .iter()
                        .flat_map(|dir| ["-I".to_string(), dir.clone()]),
                )
                .arg("-Xcompiler")
                .arg("-fPIC")
                .args(&gencode_flags)
                .args(&extra_flags)
                .arg(source)
                .arg("-o")
                .arg(&object),
        );
        objects.push(object);
    }
    let dlink = out_dir.join("stwo_cuda_kernels_dlink.o");
    run_nvcc(
        Command::new(&nvcc)
            .arg("-dlink")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .args(&gencode_flags)
            .args(&objects)
            .arg("-o")
            .arg(&dlink),
    );
    objects.push(dlink);

    let archive = out_dir.join("libstwo_cuda_kernels.a");
    let _ = std::fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let output = Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .output()
        .expect("ar should be available to archive the kernel objects");
    assert!(
        output.status.success(),
        "ar failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    println!("cargo:rustc-env=STWO_CUDA_BUILD_MODE=cuda");
    println!("cargo:rustc-cfg=stwo_cuda_link");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if let Some(lib_dir) = cuda_lib_dir(&nvcc) {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=static=stwo_cuda_kernels");
    println!("cargo:rustc-link-lib=cudart");
    // The JIT runtime (runtime_jit.cu) compiles generated kernels at runtime.
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=cuda");
    // The .cu host code uses C++ exceptions and the C++ runtime.
    println!("cargo:rustc-link-lib=stdc++");
}

/// The compute capability of the local GPU as an `-arch` value (e.g. `sm_86`), queried
/// via `nvidia-smi`. Falls back to `native` — but note `-arch=native` segfaults nvcc
/// 11.8's detection path (validated on RunPod), which is why the explicit query is the
/// default and `STWO_CUDA_ARCH` exists as an override.
fn detect_arch() -> String {
    Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            let cap = String::from_utf8(output.stdout).ok()?;
            let cap = cap.lines().next()?.trim().replace('.', "");
            (!cap.is_empty()).then(|| format!("sm_{cap}"))
        })
        .unwrap_or_else(|| "native".to_string())
}

/// The toolkit's library directory (for `-lcudart`), from `CUDA_HOME`/`CUDA_PATH` or
/// derived from the nvcc binary's location (`<root>/bin/nvcc` -> `<root>/lib64`).
fn cuda_lib_dir(nvcc: &str) -> Option<PathBuf> {
    let root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .or_else(|| {
            let which = Command::new("which").arg(nvcc).output().ok()?;
            let path = String::from_utf8(which.stdout).ok()?;
            Some(PathBuf::from(path.trim()).parent()?.parent()?.to_path_buf())
        })?;
    [root.join("lib64"), root.join("lib")]
        .into_iter()
        .find(|dir| dir.is_dir())
}

fn kernel_sources() -> Vec<PathBuf> {
    let cuda_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("cuda");
    let mut sources = Vec::new();
    collect_cu(&cuda_dir, &mut sources);
    sources.sort();
    assert!(!sources.is_empty(), "no .cu kernels found under cuda/");
    sources
}

fn collect_cu(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("cuda/ kernel directory must exist") {
        let path = entry.expect("readable cuda/ directory entry").path();
        if path.is_dir() {
            collect_cu(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "cu") {
            out.push(path);
        }
    }
}

fn collect_dirs(dir: &std::path::Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("cuda/ directory must exist") {
        let path = entry.expect("readable directory entry").path();
        if path.is_dir() {
            out.push(path.to_string_lossy().into_owned());
            collect_dirs(&path, out);
        }
    }
}
