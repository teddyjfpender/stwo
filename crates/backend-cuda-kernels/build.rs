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
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let aot_constraint_max_instrs = generated_aot_constraint_max_instrs();
    if !nvcc_available {
        write_aot_pack(&out_dir, &[], aot_constraint_max_instrs);
        println!("cargo:rustc-env=STWO_CUDA_BUILD_MODE=no-cuda");
        return;
    }
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

    // Archive objects compile on the same bounded pool (independent TUs) with an
    // mtime cache vs the SOURCE file only — header edits still dirty the build
    // via cargo's rerun-if-changed on cuda/, which reruns this script into a
    // fresh OUT_DIR fingerprint. A build.rs edit likewise re-fingerprints.
    let mut objects: Vec<PathBuf> = Vec::with_capacity(sources.len() + 1);
    let mut obj_jobs: Vec<(PathBuf, PathBuf)> = Vec::new();
    for source in &sources {
        let object = out_dir.join(format!(
            "{}.o",
            source
                .file_stem()
                .expect("kernel file stem")
                .to_string_lossy()
        ));
        let fresh = match (
            std::fs::metadata(&object).and_then(|m| m.modified()),
            std::fs::metadata(source).and_then(|m| m.modified()),
        ) {
            (Ok(o), Ok(s)) => o >= s,
            _ => false,
        };
        if !fresh {
            obj_jobs.push((source.clone(), object.clone()));
        }
        objects.push(object);
    }
    if !obj_jobs.is_empty() {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16)
            .min(obj_jobs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let jobs_ref = &obj_jobs;
        let next_ref = &next;
        let include_dirs_ref = &include_dirs;
        let gencode_ref = &gencode_flags;
        let extra_ref = &extra_flags;
        let nvcc_ref = &nvcc;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(move || loop {
                    let i = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((source, object)) = jobs_ref.get(i) else {
                        break;
                    };
                    let output = Command::new(nvcc_ref)
                        .arg("-dc")
                        .arg("-O3")
                        .arg("--std=c++17")
                        // The fp256/poseidon252 stack calls `constexpr __host__`
                        // accessors from device code (sppark lineage); nvcc
                        // requires this flag for that pattern.
                        .arg("--expt-relaxed-constexpr")
                        .args(
                            include_dirs_ref
                                .iter()
                                .flat_map(|dir| ["-I".to_string(), dir.clone()]),
                        )
                        .arg("-Xcompiler")
                        .arg("-fPIC")
                        .args(gencode_ref.iter())
                        .args(extra_ref.iter())
                        .arg(source)
                        .arg("-o")
                        .arg(object)
                        .output()
                        .expect("nvcc was detected but could not be launched");
                    assert!(
                        output.status.success(),
                        "nvcc failed:\nstdout:\n{}\nstderr:\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                });
            }
        });
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

    build_aot_pack(
        &nvcc,
        &archs,
        &extra_flags,
        &out_dir,
        aot_constraint_max_instrs,
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
            // generated/ holds SELF-CONTAINED AOT module sources (kernel_emit):
            // they redefine the field helpers, so they never join the archive —
            // they compile to standalone cubins embedded in the AOT pack.
            if path.file_name().is_some_and(|n| n == "generated") {
                continue;
            }
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

/// Compile every `cuda/generated/*.cu` (kernel_emit output: self-contained AOT
/// module sources named `<kind>_<label>_<cache_key:016x>.cu`) to a standalone
/// cubin per arch at -O3, and embed them as one pack + index. The runtime's
/// `stwo_aot_lookup` (src/aot_pack.rs) serves `get_or_compile`'s tier-0 — a
/// cache-key miss falls back to NVRTC, which IS the drift check (design §4).
///
/// Cubins are cached in OUT_DIR by (source mtime): an unchanged kernel never
/// recompiles. SASS generation for the biggest fused kernels is the expensive
/// step — paid per AIR revision at build time, never at prove time.
fn build_aot_pack(
    nvcc: &str,
    archs: &[String],
    extra_flags: &[String],
    out_dir: &PathBuf,
    constraint_max_instrs: usize,
) {
    let gen_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("cuda")
        .join("generated");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&gen_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "cu"))
                .collect()
        })
        .unwrap_or_default();
    sources.sort();

    let cubin_dir = out_dir.join("aot_cubins");
    std::fs::create_dir_all(&cubin_dir).expect("create aot cubin cache dir");
    // Collect jobs, then compile stale ones on a bounded worker pool — cubins
    // are independent TUs and SASS -O3 on the big fp256 kernels takes minutes
    // each; serial nvcc dominated the first pod build.
    let mut entries: Vec<(u64, u32, PathBuf)> = Vec::new();
    let mut jobs: Vec<(PathBuf, String, PathBuf)> = Vec::new();
    for source in &sources {
        let stem = source.file_stem().unwrap().to_string_lossy().to_string();
        let key = u64::from_str_radix(stem.rsplit('_').next().unwrap(), 16)
            .expect("generated kernel file names end in _<cache_key:016x>");
        let src_mtime = std::fs::metadata(source).and_then(|m| m.modified()).ok();
        for arch in archs {
            let num: u32 = arch
                .trim_start_matches("sm_")
                .parse()
                .expect("STWO_CUDA_ARCH entries look like sm_90");
            let cubin = cubin_dir.join(format!("{stem}_{arch}.cubin"));
            let fresh = match (
                std::fs::metadata(&cubin).and_then(|m| m.modified()),
                src_mtime,
            ) {
                (Ok(c), Some(s)) => c >= s,
                _ => false,
            };
            if !fresh {
                jobs.push((source.clone(), arch.clone(), cubin.clone()));
            }
            entries.push((key, num, cubin));
        }
    }
    if !jobs.is_empty() {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16)
            .min(jobs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let jobs_ref = &jobs;
        let next_ref = &next;
        let nvcc_ref = &nvcc;
        let extra_ref = &extra_flags;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(move || loop {
                    let i = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((source, arch, cubin)) = jobs_ref.get(i) else {
                        break;
                    };
                    let output = Command::new(nvcc_ref)
                        .arg("-cubin")
                        .arg("-O3")
                        .arg("--std=c++17")
                        .arg("--expt-relaxed-constexpr")
                        .arg(format!("-arch={arch}"))
                        .args(extra_ref.iter())
                        .arg(source)
                        .arg("-o")
                        .arg(cubin)
                        .output()
                        .expect("nvcc launch for AOT cubin");
                    assert!(
                        output.status.success(),
                        "nvcc -cubin failed for {}:\n{}",
                        source.display(),
                        String::from_utf8_lossy(&output.stderr)
                    );
                });
            }
        });
    }
    let refs: Vec<(u64, u32, &std::path::Path)> = entries
        .iter()
        .map(|(k, a, p)| (*k, *a, p.as_path()))
        .collect();
    write_aot_pack(out_dir, &refs, constraint_max_instrs);
}

/// Concatenate cubins into `aot_pack.bin` + emit `aot_index.rs` (sorted by
/// (cache_key, sm)). Always written — an empty pack keeps the stub build and
/// include_bytes! happy.
fn write_aot_pack(
    out_dir: &std::path::Path,
    entries: &[(u64, u32, &std::path::Path)],
    constraint_max_instrs: usize,
) {
    let mut pack: Vec<u8> = Vec::new();
    let mut index: Vec<(u64, u32, usize, usize)> = Vec::new();
    for (key, sm, path) in entries {
        let blob = std::fs::read(path).expect("read cubin");
        index.push((*key, *sm, pack.len(), blob.len()));
        pack.extend_from_slice(&blob);
    }
    index.sort();
    std::fs::write(out_dir.join("aot_pack.bin"), &pack).expect("write aot pack");
    let mut rs = String::from(
        "// Generated by build.rs — (cache_key, sm, offset, len) into aot_pack.bin.\n         pub(crate) static AOT_INDEX: &[(u64, u32, usize, usize)] = &[\n",
    );
    for (key, sm, off, len) in &index {
        rs.push_str(&format!("    (0x{key:016x}, {sm}, {off}, {len}),\n"));
    }
    rs.push_str("];\n");
    rs.push_str(&format!(
        "pub(crate) const AOT_CONSTRAINT_MAX_INSTRS: usize = {constraint_max_instrs};\n"
    ));
    std::fs::write(out_dir.join("aot_index.rs"), rs).expect("write aot index");
}

fn generated_aot_constraint_max_instrs() -> usize {
    let path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("cuda")
        .join("generated")
        .join("aot_constraint_max_instrs.txt");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "generated AOT constraint cap is missing at {}: {error}",
            path.display()
        )
    });
    let cap = raw.trim().parse::<usize>().unwrap_or_else(|error| {
        panic!(
            "invalid generated AOT constraint cap at {}: {error}",
            path.display()
        )
    });
    assert!(cap > 0, "generated AOT constraint cap must be non-zero");
    cap
}
