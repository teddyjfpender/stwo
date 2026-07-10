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
    *) MODE="explicit"; FILES+=("$arg") ;;
  esac
done

docker_run() {
  docker run --rm --platform linux/amd64 \
    -v "$PWD:/workspace" -w /workspace "$IMAGE" bash -lc "$1"
}

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
  mapfile -t FILES < <(ls "${KERNELS_DIR}"/cuda/*.cu)
  if [[ "$INCLUDE_GENERATED" == "1" ]]; then
    mapfile -t -O "${#FILES[@]}" FILES < <(ls "${KERNELS_DIR}"/cuda/generated/*.cu)
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
