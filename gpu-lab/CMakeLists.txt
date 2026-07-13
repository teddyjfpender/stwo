cmake_minimum_required(VERSION 3.22)
project(stwo_gpu_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(GPU_LAB_SM "86" CACHE STRING "Single CUDA SM target, for example 86, 90, or 100")
if(NOT GPU_LAB_SM MATCHES "^[0-9][0-9][0-9]?$")
  message(FATAL_ERROR "GPU_LAB_SM must be two or three digits, for example 86, 90, or 100")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit 11.8 REQUIRED)

add_executable(stwo-gpu-lab
  harness/budget.cpp
  harness/buffers.cpp
  harness/effects.cpp
  harness/identity.cpp
  harness/main.cpp
  harness/replay.cpp
  harness/result.cpp
  harness/runner.cpp)
target_link_libraries(stwo-gpu-lab PRIVATE CUDA::cuda_driver)

add_executable(stwo-gpu-lab-cpu-self-test
  harness/budget.cpp
  harness/cpu_selftest.cpp
  harness/effects.cpp
  harness/fri_round6.cpp
  harness/identity.cpp
  harness/replay.cpp)
add_executable(stwo-gpu-lab-fri-round6
  harness/fri_round6.cpp
  harness/fri_round6_main.cpp
  harness/fri_round6_runner.cpp
  harness/identity.cpp)
target_link_libraries(stwo-gpu-lab-fri-round6 PRIVATE CUDA::cuda_driver)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stwo-gpu-lab PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(stwo-gpu-lab-cpu-self-test
                         PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(stwo-gpu-lab-fri-round6
                         PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

set(LAB_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/tools/lab.py")
file(GLOB LAB_TOOL_MODULES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/*.py")
set(LAB_TOOL_SOURCES "${LAB_TOOL}" ${LAB_TOOL_MODULES})
set(FRI_MODULE_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/tools/fri-module")
set(FRI_MODULE_TOOL_SOURCES
    "${FRI_MODULE_TOOL}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/fri_module_cli.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/fri_discovery.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/fri_module.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/fri_publication.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/fri_staging.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/fri_tool_identity.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/common.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/identity.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_lab/immutable_output.py")
set(PEDERSEN_ABI "${CMAKE_CURRENT_SOURCE_DIR}/manifests/witness_pedersen_builtin.abi.json")
set(PEDERSEN_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda-kernels/cuda/generated/witness_pedersen_builtin_d7b6e6ef82833ff3.cu")
set(PEDERSEN_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda/src/backend/jit_witness/codegen.rs")
set(PEDERSEN_ISA
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda/src/backend/jit_witness/isa.rs")
set(PEDERSEN_RECORDER
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda/src/backend/jit_witness/recording.rs")
set(PEDERSEN_PROGRAMS
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda/src/backend/jit_witness/programs.rs")
set(KERNEL_EMIT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../stwo-cairo/stwo_cairo_prover/crates/gpu-prover/src/bin/kernel_emit.rs")
set(PEDERSEN_REGISTRY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/jit_prove_backend.rs")
set(PEDERSEN_COMPONENT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin.rs")
set(WITNESS_EVAL
    "${CMAKE_CURRENT_SOURCE_DIR}/../../stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/witness_eval/mod.rs")
set(WITNESS_RECORDER
    "${CMAKE_CURRENT_SOURCE_DIR}/../../stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/witness_eval/recording.rs")
set(AOT_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda-kernels/cuda/generated/aot_manifest.json")
set(MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modules")
set(TOOLCHAIN_IDENTITY "${CMAKE_CURRENT_BINARY_DIR}/toolchain-identity.json")
set(PEDERSEN_INDEX "${MODULE_DIR}/witness_pedersen_builtin.sm${GPU_LAB_SM}.module.json")
set(PEDERSEN_STAMP "${MODULE_DIR}/witness_pedersen_builtin.sm${GPU_LAB_SM}.stamp")

add_custom_target(gpu_lab_toolchain_identity
  COMMAND "${Python3_EXECUTABLE}" "${LAB_TOOL}" toolchain-identity
          --nvcc "${CUDAToolkit_NVCC_EXECUTABLE}"
          --host-compiler "${CMAKE_CXX_COMPILER}"
          --output "${TOOLCHAIN_IDENTITY}"
  BYPRODUCTS "${TOOLCHAIN_IDENTITY}"
  COMMENT "Refreshing immutable CUDA toolchain identity"
  VERBATIM)

add_custom_command(
  OUTPUT "${PEDERSEN_STAMP}"
  COMMAND "${Python3_EXECUTABLE}" "${LAB_TOOL}" build-module
          --nvcc "${CUDAToolkit_NVCC_EXECUTABLE}"
          --host-compiler "${CMAKE_CXX_COMPILER}"
          --sm "${GPU_LAB_SM}"
          --source "${PEDERSEN_SOURCE}"
          --abi "${PEDERSEN_ABI}"
          --repo-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
          --workspace-root "${CMAKE_CURRENT_SOURCE_DIR}/../.."
          --generator "${PEDERSEN_GENERATOR}"
          --generator "${PEDERSEN_ISA}"
          --generator "${PEDERSEN_RECORDER}"
          --generator "${PEDERSEN_PROGRAMS}"
          --generator "${KERNEL_EMIT}"
          --generator "${PEDERSEN_REGISTRY}"
          --generator "${PEDERSEN_COMPONENT}"
          --generator "${WITNESS_EVAL}"
          --generator "${WITNESS_RECORDER}"
          --generator "${AOT_MANIFEST}"
          --output-dir "${MODULE_DIR}"
          --index "${PEDERSEN_INDEX}"
          --stamp "${PEDERSEN_STAMP}"
  DEPENDS "${PEDERSEN_SOURCE}" "${PEDERSEN_ABI}" "${PEDERSEN_GENERATOR}"
          "${PEDERSEN_ISA}" "${PEDERSEN_RECORDER}" "${PEDERSEN_PROGRAMS}"
          "${KERNEL_EMIT}" "${PEDERSEN_REGISTRY}" "${PEDERSEN_COMPONENT}"
          "${WITNESS_EVAL}" "${WITNESS_RECORDER}" "${AOT_MANIFEST}"
          ${LAB_TOOL_SOURCES} "${TOOLCHAIN_IDENTITY}"
  COMMENT "Building one immutable witness_pedersen_builtin sm_${GPU_LAB_SM} cubin"
  VERBATIM)

add_custom_target(witness_pedersen_builtin_cubin DEPENDS "${PEDERSEN_STAMP}")
add_dependencies(witness_pedersen_builtin_cubin gpu_lab_toolchain_identity)

set(FRI_ROUND6_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fri_round6.cu")
set(FRI_ROUND6_ABI "${CMAKE_CURRENT_SOURCE_DIR}/manifests/fri_round6.abi.json")
set(CUDA_KERNEL_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../crates/backend-cuda-kernels/cuda")
set(FRI_ROUND6_INDEX
    "${MODULE_DIR}/fri_round6.sm${GPU_LAB_SM}.locator.non-evidence.json")
set(FRI_ROUND6_STAMP
    "${MODULE_DIR}/fri_round6.sm${GPU_LAB_SM}.stamp")
set(FRI_ROUND6_DEPFILE
    "${MODULE_DIR}/fri_round6.sm${GPU_LAB_SM}.d")
set(FRI_ROUND6_PRODUCTION_SOURCES
    "${CUDA_KERNEL_ROOT}/fields.cu"
    "${CUDA_KERNEL_ROOT}/fold_line.cu"
    "${CUDA_KERNEL_ROOT}/blake2s.cu"
    "${CUDA_KERNEL_ROOT}/device_transcript.cu")
add_custom_command(
  OUTPUT "${FRI_ROUND6_STAMP}"
  BYPRODUCTS "${FRI_ROUND6_INDEX}"
  COMMAND "${Python3_EXECUTABLE}" "${FRI_MODULE_TOOL}" build
          --nvcc "${CUDAToolkit_NVCC_EXECUTABLE}"
          --host-compiler "${CMAKE_CXX_COMPILER}"
          --sm "${GPU_LAB_SM}"
          --source "${FRI_ROUND6_SOURCE}"
          --abi "${FRI_ROUND6_ABI}"
          --repo-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
          --output-dir "${MODULE_DIR}"
          --index "${FRI_ROUND6_INDEX}"
          --stamp "${FRI_ROUND6_STAMP}"
          --depfile "${FRI_ROUND6_DEPFILE}"
  DEPENDS "${FRI_ROUND6_SOURCE}" "${FRI_ROUND6_ABI}"
          ${FRI_ROUND6_PRODUCTION_SOURCES} ${FRI_MODULE_TOOL_SOURCES}
          "${TOOLCHAIN_IDENTITY}"
  DEPFILE "${FRI_ROUND6_DEPFILE}"
  COMMENT "Building exact content-addressed FRI round-6 sm_${GPU_LAB_SM} cubin"
  VERBATIM)
add_custom_target(fri_round6_cubin DEPENDS "${FRI_ROUND6_STAMP}")
add_dependencies(fri_round6_cubin gpu_lab_toolchain_identity)

add_custom_target(fri_round6_module_validate
  COMMAND "${Python3_EXECUTABLE}" "${FRI_MODULE_TOOL}" validate
          --nvcc "${CUDAToolkit_NVCC_EXECUTABLE}"
          --host-compiler "${CMAKE_CXX_COMPILER}"
          --sm "${GPU_LAB_SM}"
          --source "${FRI_ROUND6_SOURCE}"
          --abi "${FRI_ROUND6_ABI}"
          --repo-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
          --output-dir "${MODULE_DIR}"
          --index "${FRI_ROUND6_INDEX}"
          --stamp "${FRI_ROUND6_STAMP}"
          --depfile "${FRI_ROUND6_DEPFILE}"
  DEPENDS "${FRI_ROUND6_STAMP}"
  COMMENT "Validating FRI round-6 module, recipe, ABI, SM, and source closure"
  VERBATIM)
