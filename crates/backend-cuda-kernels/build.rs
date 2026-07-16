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
//! - `STWO_CUDA_BUILD_JOBS`: maximum concurrent nvcc processes (default: host parallelism)
//! - `STWO_CUDA_HOST_COMPILER`: explicit nvcc host compiler (default: `c++` from `PATH`)
//!
//! The kernels are compiled with `-rdc=true` (they cross-reference `fields.cu` across
//! translation units) and `-dlto`, so device link-time optimization can inline field ops
//! at the final device link — see "Known issues" in the README.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn cuda_build_workers(job_count: usize) -> usize {
    env::var("STWO_CUDA_BUILD_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
        .min(16)
        .min(job_count)
}

#[path = "build_cache.rs"]
mod build_cache;
use build_cache::*;

fn object_fixed_flags(include_dirs: &[String]) -> Vec<String> {
    // The fp256/poseidon252 stack calls constexpr host accessors from device
    // code (sppark lineage), which requires relaxed constexpr compilation.
    let mut flags = ["-dc", "-O3", "--std=c++17", "--expt-relaxed-constexpr"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    flags.extend(
        include_dirs
            .iter()
            .flat_map(|dir| ["-I".to_string(), dir.clone()]),
    );
    flags.extend(["-Xcompiler".to_string(), "-fPIC".to_string()]);
    flags
}

fn aot_fixed_flags() -> Vec<String> {
    ["-cubin", "-O3", "--std=c++17", "--expt-relaxed-constexpr"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(stwo_cuda_link)");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_NVCC_FLAGS");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_BUILD_JOBS");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_HOST_COMPILER");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed=build_cache.rs");
    println!("cargo:rerun-if-changed=build_fingerprint_tests.rs");

    let sources = kernel_sources();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    // Track the whole cuda/ tree (cargo traverses directories recursively): headers
    // (.cuh) are compiled into every translation unit, so an edit there must dirty
    // the build just like a .cu edit — otherwise stale objects with old symbol
    // signatures survive in the archive.
    println!("cargo:rerun-if-changed=cuda");

    let nvcc_requested = env::var("STWO_CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let nvcc_path = command_path(&nvcc_requested);
    emit_command_reruns(nvcc_path.as_deref());
    let nvcc_available = command_version(&nvcc_requested, true).is_some();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let aot_constraint_max_instrs = generated_aot_constraint_max_instrs();
    let aot_constraint_max_live_u32_lanes = generated_aot_constraint_max_live_u32_lanes();
    if !nvcc_available {
        write_aot_pack(
            &out_dir,
            &[],
            aot_constraint_max_instrs,
            aot_constraint_max_live_u32_lanes,
        );
        println!("cargo:rustc-env=STWO_CUDA_BUILD_MODE=no-cuda");
        return;
    }
    let nvcc_path = nvcc_path
        .and_then(|path| std::fs::canonicalize(path).ok())
        .unwrap_or_else(|| PathBuf::from(&nvcc_requested));
    let nvcc = nvcc_path.to_string_lossy().into_owned();
    let nvcc_identity =
        command_version(&nvcc, true).expect("re-probe resolved nvcc version successfully");
    let nvcc_command_identity = command_identity(&nvcc, Some(&nvcc_path));
    let host_compiler = env::var("STWO_CUDA_HOST_COMPILER").unwrap_or_else(|_| "c++".to_string());
    let host_path = command_path(&host_compiler)
        .unwrap_or_else(|| panic!("nvcc host compiler not found: {host_compiler}"));
    emit_command_reruns(Some(&host_path));
    let host_executable = std::fs::canonicalize(&host_path).unwrap_or_else(|_| host_path.clone());
    let host_executable = host_executable.to_string_lossy().into_owned();
    let host_version_identity = command_version(&host_executable, false)
        .expect("query resolved nvcc host compiler version");
    let host_command_identity = command_identity(
        &host_executable,
        Some(std::path::Path::new(&host_executable)),
    );
    let host_compiler_flag = format!("-ccbin={host_executable}");
    let compiler_identity = CompilerIdentity {
        executable: &nvcc,
        command: &nvcc_command_identity,
        version: &nvcc_identity,
        host_executable: &host_executable,
        host_command: &host_command_identity,
        host_version: &host_version_identity,
        host_flag: &host_compiler_flag,
    };
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
    validate_extra_flags(&extra_flags).unwrap_or_else(|error| panic!("{error}"));

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
    let mut object_compile_flags = object_fixed_flags(&include_dirs);
    object_compile_flags.push(compiler_identity.host_flag.to_string());
    object_compile_flags.extend(gencode_flags.iter().cloned());
    object_compile_flags.extend(extra_flags.iter().cloned());
    let object_compiler_fingerprint = compiler_fingerprint(
        compiler_identity,
        &object_compile_flags,
        OBJECT_COMPILER_POLICY,
    );
    let header_fingerprint = headers_digest(&include_dirs);

    // Archive objects compile on the same bounded pool (independent TUs). Their
    // local and persistent paths are content-addressed by source, headers and
    // compiler policy; mtimes are never trusted for generated-code freshness.
    let obj_cache: Option<PathBuf> = env::var("STWO_CUDA_OBJ_CACHE").ok().map(PathBuf::from);
    if let Some(dir) = &obj_cache {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut objects: Vec<PathBuf> = Vec::with_capacity(sources.len() + 1);
    let mut obj_jobs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut staged_objects: Vec<(PathBuf, PathBuf, Option<PathBuf>)> = Vec::new();
    let mut source_identities: Vec<(PathBuf, PathBuf, u128)> = Vec::with_capacity(sources.len());
    for source in &sources {
        let source_arg = normalized_source_path(source);
        let source_bytes = std::fs::read(source).expect("read CUDA source for cache identity");
        let key = artifact_fingerprint(
            object_compiler_fingerprint,
            &source_arg,
            &source_bytes,
            Some(header_fingerprint),
        );
        let object = out_dir.join(format!(
            "{}-{key:032x}.o",
            source
                .file_stem()
                .expect("kernel file stem")
                .to_string_lossy()
        ));
        let cache_path = obj_cache.as_ref().map(|dir| {
            dir.join(format!(
                "{}-{key:032x}.o",
                source
                    .file_stem()
                    .expect("kernel file stem")
                    .to_string_lossy()
            ))
        });
        if !object.is_file() {
            let restored_staging = cache_path
                .as_ref()
                .and_then(|cached| stage_cached_artifact(cached, &object));
            let restored = restored_staging.is_some();
            let staging = restored_staging.unwrap_or_else(|| staging_path(&object));
            if !restored {
                obj_jobs.push((source_arg.clone(), staging.clone()));
            }
            let publish_cache = if restored { None } else { cache_path };
            staged_objects.push((staging, object.clone(), publish_cache));
        }
        objects.push(object);
        source_identities.push((source.clone(), source_arg, key));
    }
    if !obj_jobs.is_empty() {
        let workers = cuda_build_workers(obj_jobs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let jobs_ref = &obj_jobs;
        let next_ref = &next;
        let compile_flags_ref = &object_compile_flags;
        let nvcc_ref = &nvcc;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(move || loop {
                    let i = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((source, staging)) = jobs_ref.get(i) else {
                        break;
                    };
                    let output = nvcc_command(nvcc_ref)
                        .args(compile_flags_ref.iter())
                        .arg(source)
                        .arg("-o")
                        .arg(staging)
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
    let object_publications: Vec<(PathBuf, PathBuf)> = staged_objects
        .iter()
        .map(|(staging, object, _)| (staging.clone(), object.clone()))
        .collect();
    publish_validated_artifacts(&object_publications, || {
        if headers_digest(&include_dirs) != header_fingerprint {
            return Err("CUDA headers changed while nvcc was running; retry the build".to_string());
        }
        validate_source_identities(
            object_compiler_fingerprint,
            Some(header_fingerprint),
            &source_identities,
        )?;
        if !compiler_identity_is_current(compiler_identity) {
            return Err(
                "nvcc or its host compiler changed while compiling CUDA objects; retry the build"
                    .to_string(),
            );
        }
        Ok(())
    })
    .unwrap_or_else(|error| panic!("{error}"));
    for (_staging, object, cache_path) in staged_objects {
        if let Some(cached) = cache_path {
            let cache_staging = staging_path(&cached);
            if std::fs::copy(&object, &cache_staging).is_ok() {
                let _ = std::fs::rename(&cache_staging, cached);
            }
        }
    }
    let dlink = out_dir.join("stwo_cuda_kernels_dlink.o");
    run_nvcc(
        nvcc_command(&nvcc)
            .arg("-dlink")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg(compiler_identity.host_flag)
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

    let aot_source_identities = build_aot_pack(
        &nvcc,
        compiler_identity,
        &archs,
        &extra_flags,
        &out_dir,
        aot_constraint_max_instrs,
        aot_constraint_max_live_u32_lanes,
    );

    assert_eq!(
        kernel_sources(),
        sources,
        "CUDA source set changed while nvcc was running; retry the build"
    );
    let expected_aot_sources: Vec<PathBuf> = aot_source_identities
        .iter()
        .map(|(source, ..)| source.clone())
        .collect();
    assert_eq!(
        aot_sources_for_build(),
        expected_aot_sources,
        "generated AOT source set changed while nvcc was running; retry the build"
    );
    let mut final_include_dirs = vec!["cuda".to_string()];
    collect_dirs(std::path::Path::new("cuda"), &mut final_include_dirs);
    final_include_dirs.sort();
    assert_eq!(
        final_include_dirs, include_dirs,
        "CUDA include-directory set changed while nvcc was running; retry the build"
    );
    assert_eq!(
        headers_digest(&final_include_dirs),
        header_fingerprint,
        "CUDA headers changed before build completion; retry the build"
    );
    validate_source_identities(
        object_compiler_fingerprint,
        Some(header_fingerprint),
        &source_identities,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    validate_source_identities(0, None, &aot_source_identities)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        generated_aot_constraint_max_instrs(),
        aot_constraint_max_instrs,
        "generated AOT instruction cap changed during build"
    );
    assert_eq!(
        generated_aot_constraint_max_live_u32_lanes(),
        aot_constraint_max_live_u32_lanes,
        "generated AOT live-lane cap changed during build"
    );
    assert!(
        compiler_identity_is_current(compiler_identity),
        "nvcc or its host compiler changed before build completion; retry the build"
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

fn generated_aot_sources() -> Vec<PathBuf> {
    let gen_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("cuda")
        .join("generated");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(gen_dir)
        .expect("generated AOT source directory must exist")
        .map(|entry| entry.expect("read generated AOT source entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
        .collect();
    sources.sort();
    sources
}

fn aot_sources_for_build() -> Vec<PathBuf> {
    if cfg!(feature = "test-only-empty-aot-pack") {
        Vec::new()
    } else {
        generated_aot_sources()
    }
}

/// Compile every `cuda/generated/*.cu` (kernel_emit output: self-contained AOT
/// module sources named `<kind>_<label>_<cache_key:016x>.cu`) to a standalone
/// cubin per arch at -O3, and embed them as one pack + index. The runtime's
/// `stwo_aot_lookup` (src/aot_pack.rs) serves `get_or_compile`'s tier-0 — a
/// cache-key miss falls back to NVRTC, which IS the drift check (design §4).
///
/// Cubins are cached in OUT_DIR by source content and the exact nvcc invocation:
/// an unchanged kernel never recompiles, while compiler or policy drift selects
/// a new path. SASS generation for the biggest fused kernels is the expensive
/// step — paid per AIR revision at build time, never at prove time.
fn build_aot_pack(
    nvcc: &str,
    compiler_identity: CompilerIdentity<'_>,
    archs: &[String],
    extra_flags: &[String],
    out_dir: &std::path::Path,
    constraint_max_instrs: usize,
    constraint_max_live_u32_lanes: usize,
) -> Vec<(PathBuf, PathBuf, u128)> {
    let sources = aot_sources_for_build();

    let cubin_dir = out_dir.join("aot_cubins");
    std::fs::create_dir_all(&cubin_dir).expect("create aot cubin cache dir");
    // Collect jobs, then compile stale ones on a bounded worker pool — cubins
    // are independent TUs and SASS -O3 on the big fp256 kernels takes minutes
    // each; serial nvcc dominated the first pod build.
    let mut entries: Vec<(u64, u32, PathBuf)> = Vec::new();
    let mut jobs: Vec<(PathBuf, PathBuf, PathBuf, Vec<String>)> = Vec::new();
    let mut source_identities: Vec<(PathBuf, PathBuf, u128)> = Vec::with_capacity(sources.len());
    for source in &sources {
        let source_arg = normalized_source_path(source);
        let stem = source.file_stem().unwrap().to_string_lossy().to_string();
        let key = u64::from_str_radix(stem.rsplit('_').next().unwrap(), 16)
            .expect("generated kernel file names end in _<cache_key:016x>");
        let source_bytes = std::fs::read(source).expect("read generated AOT source identity");
        let source_identity = artifact_fingerprint(0, &source_arg, &source_bytes, None);
        source_identities.push((source.clone(), source_arg.clone(), source_identity));
        for arch in archs {
            let num: u32 = arch
                .trim_start_matches("sm_")
                .parse()
                .expect("STWO_CUDA_ARCH entries look like sm_90");
            // The fully fused Poseidon partial-round witness reaches the CUDA
            // 11.8 sm_90 register ceiling, where ptxas miscompiles this TU. Keep
            // its generated CUDA and fp256 math unchanged while selecting a
            // conservative ptxas schedule. The flag is part of the exact
            // compiler identity below.
            let ptxas_o0 =
                arch == "sm_90" && stem.starts_with("witness_poseidon_3_partial_rounds_chain_");
            let mut compile_flags = aot_fixed_flags();
            compile_flags.push(compiler_identity.host_flag.to_string());
            compile_flags.push(format!("-arch={arch}"));
            compile_flags.extend(extra_flags.iter().cloned());
            if ptxas_o0 {
                compile_flags.push("-Xptxas=-O0".to_string());
            }
            let compiler =
                compiler_fingerprint(compiler_identity, &compile_flags, AOT_COMPILER_POLICY);
            let artifact = artifact_fingerprint(compiler, &source_arg, &source_bytes, None);
            let cubin = cubin_dir.join(format!("{stem}_{arch}_{artifact:032x}.cubin"));
            if !cubin.is_file() {
                let staging = staging_path(&cubin);
                let _ = std::fs::remove_file(&staging);
                jobs.push((source_arg.clone(), staging, cubin.clone(), compile_flags));
            }
            entries.push((key, num, cubin));
        }
    }
    if !jobs.is_empty() {
        let workers = cuda_build_workers(jobs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let jobs_ref = &jobs;
        let next_ref = &next;
        let nvcc_ref = &nvcc;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(move || loop {
                    let i = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((source, staging, _cubin, compile_flags)) = jobs_ref.get(i) else {
                        break;
                    };
                    let output = nvcc_command(nvcc_ref)
                        .args(compile_flags.iter())
                        .arg(source)
                        .arg("-o")
                        .arg(staging)
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
    let aot_publications: Vec<(PathBuf, PathBuf)> = jobs
        .iter()
        .map(|(_source, staging, cubin, _flags)| (staging.clone(), cubin.clone()))
        .collect();
    publish_validated_artifacts(&aot_publications, || {
        validate_source_identities(0, None, &source_identities)?;
        if !compiler_identity_is_current(compiler_identity) {
            return Err(
                "nvcc or its host compiler changed while compiling AOT cubins; retry the build"
                    .to_string(),
            );
        }
        Ok(())
    })
    .unwrap_or_else(|error| panic!("{error}"));
    let refs: Vec<(u64, u32, &std::path::Path)> = entries
        .iter()
        .map(|(k, a, p)| (*k, *a, p.as_path()))
        .collect();
    write_aot_pack(
        out_dir,
        &refs,
        constraint_max_instrs,
        constraint_max_live_u32_lanes,
    );
    source_identities
}

/// Concatenate cubins into `aot_pack.bin` + emit `aot_index.rs` (sorted by
/// (cache_key, sm)). Always written — an empty pack keeps the stub build and
/// include_bytes! happy.
fn write_aot_pack(
    out_dir: &std::path::Path,
    entries: &[(u64, u32, &std::path::Path)],
    constraint_max_instrs: usize,
    constraint_max_live_u32_lanes: usize,
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
    rs.push_str(&format!(
        "pub(crate) const AOT_CONSTRAINT_MAX_LIVE_U32_LANES: usize = \
         {constraint_max_live_u32_lanes};\n"
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

fn generated_aot_constraint_max_live_u32_lanes() -> usize {
    let path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("cuda")
        .join("generated")
        .join("aot_constraint_max_live_u32_lanes.txt");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "generated AOT constraint live-lane cap is missing at {}: {error}",
            path.display()
        )
    });
    let cap = raw.trim().parse::<usize>().unwrap_or_else(|error| {
        panic!(
            "invalid generated AOT constraint live-lane cap at {}: {error}",
            path.display()
        )
    });
    assert!(
        cap > 0,
        "generated AOT constraint live-lane cap must be non-zero"
    );
    cap
}

#[cfg(test)]
#[path = "build_fingerprint_tests.rs"]
mod tests;
