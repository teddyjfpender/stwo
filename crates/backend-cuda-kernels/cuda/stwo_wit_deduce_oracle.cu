// Truth-oracle launcher for the witness-JIT computed-deduce device functions.
//
// Runs the EXACT `stwo_wit_deduce_*` functions the NVRTC witness kernels embed
// (same header, same table-globals fill mechanics) from a precompiled kernel,
// so the pod validation ladder can compare device results against the host
// `fast_deduction` reference over real rows BEFORE any JIT kernel is trusted:
// a mismatch here is fp256 math / table layout, a mismatch only in the JIT
// differential is codegen. Also doubles as the device-table spot check — kind 3
// IS a raw table read, compared against host `PEDERSEN_TABLE_18` coordinates.
//
// NOTE: `stwo_wit_deduce.cuh` DEFINES module-level device globals; it must be
// included by exactly one translation unit of this archive (this one).

#include <cuda_runtime.h>
#include <cstdio>

#include "ec_ops.cuh"
#include "stwo_wit_deduce.cuh"

// pedersen_table_init.cu exports (same archive).
extern "C" bool is_pedersen_table_initialized();
extern "C" void initialize_pedersen_table();
extern "C" void get_pedersen_table_column_ptrs(m31** output_ptrs, uint32_t* out_n_rows);

namespace {

constexpr unsigned KIND_PARTIAL_EC_MUL_W18 = 2;
constexpr unsigned KIND_PEDERSEN_POINTS_W18 = 3;

__global__ void oracle_partial_ec_mul_w18_kernel(
    const unsigned* in, unsigned* out, unsigned n_items) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_items) {
        return;
    }
    stwo_wit_deduce_partial_ec_mul_w18(in + (size_t)idx * 72, out + (size_t)idx * 72);
}

__global__ void oracle_pedersen_points_w18_kernel(
    const unsigned* in, unsigned* out, unsigned n_items) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_items) {
        return;
    }
    stwo_wit_deduce_pedersen_points_w18(in + idx, out + (size_t)idx * 56);
}

// Fill this module's copy of the witness table globals from the runtime table
// state. NO self-heal generation: the oracle itself falsified the GPU-generated
// table (run 20260705T113615Z) — the host-built table must be registered first
// (borrowed mode); an unregistered table is a hard oracle failure (rc=2).
bool fill_oracle_table_globals() {
    if (!is_pedersen_table_initialized()) {
        fprintf(stderr,
                "stwo deduce oracle: pedersen table not registered (host-table "
                "registration required; GPU generation is quarantined)\n");
        return false;
    }
    m31* ptrs[56];
    uint32_t n_rows = 0;
    get_pedersen_table_column_ptrs(ptrs, &n_rows);
    if (n_rows == 0 || (n_rows & (n_rows - 1)) != 0) {
        fprintf(stderr, "stwo deduce oracle: table row count %u not a power of two\n", n_rows);
        return false;
    }
    if (cudaMemcpyToSymbol(g_stwo_wit_pedersen_cols, ptrs, sizeof(ptrs)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(g_stwo_wit_pedersen_n_rows, &n_rows, sizeof(n_rows)) !=
        cudaSuccess) {
        return false;
    }
    // The generation kernels ran on the legacy stream; make their completion
    // (and the symbol copies) visible before any oracle launch.
    return cudaDeviceSynchronize() == cudaSuccess;
}

} // namespace

// Run `n_items` deduces of `kind` on device. `h_in`/`h_out` are host buffers of
// n_items * (in words) / n_items * (out words) per the recorder shapes
// (kind 2: 72 -> 72; kind 3: 1 -> 56). Returns 0 on success, nonzero on any
// failure (unknown kind, table init failure, CUDA error) — callers must treat
// nonzero as "no data", never as zeros.
extern "C" int stwo_wit_deduce_oracle_run(
    unsigned kind, const unsigned* h_in, unsigned* h_out, unsigned n_items) {
    if (n_items == 0) {
        return 0;
    }
    unsigned in_words;
    unsigned out_words;
    if (kind == KIND_PARTIAL_EC_MUL_W18) {
        in_words = 72;
        out_words = 72;
    } else if (kind == KIND_PEDERSEN_POINTS_W18) {
        in_words = 1;
        out_words = 56;
    } else {
        return 1;
    }
    if (!fill_oracle_table_globals()) {
        return 2;
    }

    unsigned* d_in = nullptr;
    unsigned* d_out = nullptr;
    size_t in_bytes = (size_t)n_items * in_words * sizeof(unsigned);
    size_t out_bytes = (size_t)n_items * out_words * sizeof(unsigned);
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        return 3;
    }
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_in);
        return 3;
    }
    int rc = 0;
    if (cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = 4;
    }
    if (rc == 0) {
        const unsigned block = 256;
        unsigned grid = (n_items + block - 1) / block;
        if (kind == KIND_PARTIAL_EC_MUL_W18) {
            oracle_partial_ec_mul_w18_kernel<<<grid, block>>>(d_in, d_out, n_items);
        } else {
            oracle_pedersen_points_w18_kernel<<<grid, block>>>(d_in, d_out, n_items);
        }
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            rc = 5;
        }
    }
    if (rc == 0 &&
        cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        rc = 6;
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return rc;
}
