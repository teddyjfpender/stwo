#!/usr/bin/env bash
#
# cuda_compile_check.sh — catch nvcc compile errors without a GPU.
#
# CUDA kernels are compiled offline; a GPU is only needed to execute them. This
# runs the pod's exact compiler (CUDA 11.8, sm_90) in an x86_64 container under
# emulation, so header/template/type/arch errors surface locally in minutes
# instead of costing an H100 round trip. Runtime classes (illegal accesses,
# races, capture/replay behavior, occupancy) still require the hardware gates.
#
# Usage:
#   scripts/cuda_compile_check.sh                 # kernels changed vs HEAD
#   scripts/cuda_compile_check.sh --all           # every hand-written kernel
#   scripts/cuda_compile_check.sh --all --generated  # + generated AOT pack
#   scripts/cuda_compile_check.sh cuda/foo.cu ... # explicit files
#   scripts/cuda_compile_check.sh --build         # exact: run build.rs via
#                                                 # cargo in the container
#   scripts/cuda_compile_check.sh --link          # build the real nvcc archive
#                                                 # AND link every native test
#                                                 # binary (catches stub-masked
#                                                 # FFI/symbol drift; no GPU)
#   scripts/cuda_compile_check.sh --resources F.. # -Xptxas -v resource table;
#                                                 # FAILS on spills and known
#                                                 # SM90 launch-contract drift
#
# Environment:
#   CUDA_IMAGE  container image (default nvidia/cuda:11.8.0-devel-ubuntu22.04,
#               matching the H100 pod)
#   CUDA_ARCH   -arch value for the fast mode (default sm_90)
#   JOBS        parallel nvcc processes (default 2: emulated cicc segfaults
#               spuriously under heavier parallelism — retry a FAIL solo before
#               trusting it)

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${CUDA_IMAGE:-nvidia/cuda:11.8.0-devel-ubuntu22.04}"
ARCH="${CUDA_ARCH:-sm_90}"
JOBS="${JOBS:-2}"
KERNELS_DIR="crates/backend-cuda-kernels"

MODE="changed"
INCLUDE_GENERATED=0
FILES=()
for arg in "$@"; do
  case "$arg" in
    --all) MODE="all" ;;
    --generated) INCLUDE_GENERATED=1 ;;
    --build) MODE="build" ;;
    --link) MODE="link" ;;
    --symbols) MODE="symbols" ;;
    --resources) MODE="resources" ;;
    *) [[ "$MODE" == "changed" ]] && MODE="explicit"; FILES+=("$arg") ;;
  esac
done

docker_run() {
  docker run --rm --platform linux/amd64 \
    -v "$PWD:/workspace" -w /workspace "$IMAGE" bash -lc "$1"
}

if [[ "$MODE" == "symbols" ]]; then
  # Instant symbol-coherence audit: every extern "C" declaration in raw.rs
  # must have a definition in some hand-written .cu and a no-CUDA stub. Catches
  # the class the macOS stubs mask (missing/renamed symbol discovered only at
  # pod link time) without compiling anything. The real --link gate remains
  # authoritative for C linkage and signatures.
  python3 - "$KERNELS_DIR" <<'PY'
import pathlib, re, sys

kd = sys.argv[1]
root = pathlib.Path(kd)

def without_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//.*', '', text)

raw = without_comments((root / "src/raw.rs").read_text())
extern_blocks = re.findall(r'(?ms)^extern\s+"C"\s*\{(.*?)^\}', raw)
declared = set(re.findall(
    r'\bpub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', "\n".join(extern_blocks)
))
sources = [
    without_comments(path.read_text(errors="replace"))
    for path in (root / "cuda").glob("**/*.cu")
    if "generated" not in path.parts
]

def has_definition(name):
    signature = re.compile(
        rf'\b{re.escape(name)}\s*\('
        rf'(?:[^(){{}};]|\([^(){{}};]*\))*\)\s*'
        rf'(?:noexcept\s*)?\{{',
        re.S,
    )
    return any(signature.search(source) for source in sources)

defined = {name for name in declared if has_definition(name)}
stubs_source = without_comments((root / "src/stubs.rs").read_text())
stubs = set(re.findall(
    r'\bpub\s+unsafe\s+extern\s+"C"\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
    stubs_source,
))
missing_def = sorted(declared - defined)
missing_stub = sorted(declared - stubs)
extra_stub = sorted(stubs - declared)
ok = True
if missing_def:
    print(f"FAIL: declared in raw.rs but no .cu definition: {missing_def}"); ok = False
if missing_stub:
    print(f"FAIL: declared in raw.rs but no stub: {missing_stub}"); ok = False
if extra_stub:
    print(f"FAIL: stub has no raw.rs declaration: {extra_stub}"); ok = False
print(f"[cuda_compile_check] symbols: {len(declared)} declared, {len(defined)} defined, "
      + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
  exit $?
fi

RUST_VOLUME="stwo-cuda-rustup"

container_cargo() {
  # Rust toolchain lives in a named volume so repeat runs skip the install.
  docker run --rm --platform linux/amd64 \
    -v "$PWD:/workspace" -v "${RUST_VOLUME}:/root/.rustup" \
    -v "${RUST_VOLUME}-cargo:/root/.cargo" \
    -e STWO_CUDA_OBJ_CACHE=/workspace/.docker_cuda_obj_cache \
    -e STWO_CUDA_ARCH="${ARCH}" -e STWO_CUDA_BUILD_JOBS="${JOBS}" \
    -e CARGO_BUILD_JOBS="${JOBS}" \
    -e RUST_MIN_STACK=33554432 \
    -w /workspace "$IMAGE" bash -lc "
      set -e
      export PATH=\$HOME/.cargo/bin:\$PATH
      command -v cargo >/dev/null 2>&1 || {
        apt-get update -qq >/dev/null && apt-get install -y -qq curl build-essential >/dev/null
        curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null
      }
      $1
    "
}

if [[ "$MODE" == "link" ]]; then
  # The macOS stub build cannot catch FFI declarations whose symbol is missing
  # from (or mismatched in) the real nvcc archive — only producing the archive
  # and LINKING the cfg(stwo_cuda_link) test binaries proves symbol coherence.
  # No GPU needed: linking resolves symbols without executing kernels.
  container_cargo "cargo build --locked -p stwo-backend-cuda-kernels && cargo test --locked -p stwo-backend-cuda --no-run"
  echo '[cuda_compile_check] link-mode PASS (archive built, all native test binaries linked)'
  exit 0
fi

if [[ "$MODE" == "resources" ]]; then
  # Per-kernel register/spill/shared-memory report via ptxas. Spills are the
  # plan's hard failure criterion for the fused kernels.
  [[ "${#FILES[@]}" -gt 0 ]] || { echo 'usage: --resources <file.cu>...'; exit 2; }
  out=$(printf '%s\0' "${FILES[@]}" | docker run --rm -i --platform linux/amd64 \
    -v "$PWD:/workspace" -w /workspace "$IMAGE" bash -lc "
      xargs -0 -P ${JOBS} -I{} sh -c '
        echo \"=== {} ===\";
        nvcc -std=c++17 -arch=${ARCH} -dc --expt-relaxed-constexpr -Xptxas -v \
          -I ${KERNELS_DIR}/cuda -I ${KERNELS_DIR}/cuda/generated \
          -o /dev/null {} 2>&1 | grep -E \"Function|registers|spill|smem\"
      '
    ")
  echo "$out"
  if [[ "$ARCH" == "sm_90" ]] && printf '%s\n' "${FILES[@]}" \
      | grep -Fxq "${KERNELS_DIR}/cuda/ifft.cu"; then
    # The exact composition/B2N launchers use (32, 2^LOG_VALUES_PER_THREAD).
    # Keep this static seam paired with the ptxas receipt: if launch geometry
    # changes, update both rather than silently checking a stale thread count.
    source_pattern='dim3 block{warp, 1u << LOG_VALUES_PER_THREAD, 1};'
    if [[ "$(grep -Fc "$source_pattern" "${KERNELS_DIR}/cuda/ifft.cu")" -ne 2 ]]; then
      echo '[cuda_compile_check] resources FAIL: ifft launch geometry drifted'
      exit 1
    fi
    receipt=$(mktemp)
    trap 'rm -f "$receipt"' EXIT
    printf '%s\n' "$out" > "$receipt"
    gpu-lab/tools/check-cuda-resources "$receipt" \
      --kernel '_Z22b2n_noinit_block_batchILj4ELb0EE' \
      --launch-threads 512 --required-blocks-per-sm 1 \
      --registers-per-sm 65536 --max-registers 128 \
      --max-stack-bytes 0 --max-spill-store-bytes 0 \
      --max-spill-load-bytes 0 \
      --max-static-shared-bytes 34816
    gpu-lab/tools/check-cuda-resources "$receipt" \
      --kernel '_Z32composition_split_boundary_batchILj3ELb1EE' \
      --launch-threads 256 --registers-per-sm 65536
    if grep -Fq '_Z32composition_split_boundary_batchILj4E' "$receipt"; then
      echo '[cuda_compile_check] resources FAIL: unsupported 512-thread composition kernel was emitted'
      exit 1
    fi
    gpu-lab/tools/check-cuda-resources "$receipt" \
      --kernel '_Z32composition_split_boundary_batchILj3ELb0EE' \
      --launch-threads 256 --registers-per-sm 65536
    rm -f "$receipt"
    trap - EXIT
  fi
  if [[ "$ARCH" == "sm_90" ]] && printf '%s\n' "${FILES[@]}" \
      | grep -Fxq "${KERNELS_DIR}/cuda/quotients.cu"; then
    receipt=$(mktemp)
    trap 'rm -f "$receipt"' EXIT
    printf '%s\n' "$out" > "$receipt"
    # The kernel launches 128 threads with __launch_bounds__(128, 4). Checking
    # the four-block register envelope as 512 threads fails closed above 128
    # registers/thread; ptxas must additionally report zero stack and spills.
    gpu-lab/tools/check-cuda-resources "$receipt" \
      --kernel '_Z34combine_quotients_b2n_init7_in_gpu' \
      --launch-threads 128 --required-blocks-per-sm 4 \
      --registers-per-sm 65536 --max-registers 128 \
      --max-stack-bytes 0 --max-spill-store-bytes 0 \
      --max-spill-load-bytes 0 \
      --max-static-shared-bytes 2048
    rm -f "$receipt"
    trap - EXIT
  fi
  if [[ "$ARCH" == "sm_90" ]] && printf '%s\n' "${FILES[@]}" \
      | grep -Fxq "${KERNELS_DIR}/cuda/oods_collapsed.cu"; then
    receipt=$(mktemp)
    trap 'rm -f "$receipt"' EXIT
    printf '%s\n' "$out" > "$receipt"
    # The collapsed OODS launch has one 512-thread CTA per 1024-row partition;
    # its small-domain sibling uses 256 threads. Keep both executable shapes
    # inside Hopper's per-SM register file and fail closed on missing ptxas
    # entries. The global spill check below remains authoritative for spills.
    gpu-lab/tools/check-cuda-resources "$receipt" \
      --kernel 'barycentric_weights_collapsed_1024_kernel' \
      --launch-threads 512 --registers-per-sm 65536
    gpu-lab/tools/check-cuda-resources "$receipt" \
      --kernel 'barycentric_weights_collapsed_small_kernel' \
      --launch-threads 256 --registers-per-sm 65536
    rm -f "$receipt"
    trap - EXIT
  fi
  if echo "$out" | grep -E '[1-9][0-9]* bytes spill (stores|loads)' >/dev/null; then
    echo '[cuda_compile_check] resources FAIL: spills detected'
    exit 1
  fi
  echo '[cuda_compile_check] resources PASS (zero spills)'
  exit 0
fi

if [[ "$MODE" == "build" ]]; then
  # Authoritative: the crate's own build.rs drives nvcc exactly as on the pod.
  docker_run "
    set -e
    command -v cargo >/dev/null 2>&1 || {
      apt-get update -qq && apt-get install -y -qq curl build-essential >/dev/null
      curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null
    }
    . \$HOME/.cargo/env
    STWO_CUDA_ARCH=${ARCH} cargo build -p stwo-backend-cuda-kernels
  "
  echo '[cuda_compile_check] build-mode PASS'
  exit 0
fi

if [[ "$MODE" == "changed" ]]; then
  mapfile -t FILES < <(git diff --name-only HEAD -- "${KERNELS_DIR}/cuda" \
    | grep -E '\.(cu|cuh)$' || true)
  # A changed header rechecks every hand-written translation unit.
  if printf '%s\n' "${FILES[@]}" | grep -q '\.cuh$'; then
    MODE="all"
  else
    mapfile -t FILES < <(printf '%s\n' "${FILES[@]}" | grep '\.cu$' || true)
  fi
fi

if [[ "$MODE" == "all" ]]; then
  mapfile -t FILES < <(find "${KERNELS_DIR}/cuda" -type f -name '*.cu' \
    ! -path '*/generated/*' -print | sort)
  if [[ "$INCLUDE_GENERATED" == "1" ]]; then
    mapfile -t -O "${#FILES[@]}" FILES < <(find "${KERNELS_DIR}/cuda/generated" \
      -maxdepth 1 -type f -name '*.cu' -print | sort)
  fi
fi

if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo '[cuda_compile_check] no kernel changes to check'
  exit 0
fi

echo "[cuda_compile_check] ${#FILES[@]} translation unit(s), ${IMAGE}, -arch=${ARCH}"
printf '%s\0' "${FILES[@]}" | docker run --rm -i --platform linux/amd64 \
  -v "$PWD:/workspace" -w /workspace "$IMAGE" bash -lc "
    xargs -0 -P ${JOBS} -I{} sh -c '
      nvcc -std=c++17 -arch=${ARCH} --ptx \
        -I ${KERNELS_DIR}/cuda -I ${KERNELS_DIR}/cuda/generated \
        -o /dev/null {} && echo \"  OK {}\" || { echo \"  FAIL {}\"; exit 255; }
    '
  "
echo '[cuda_compile_check] PASS'
