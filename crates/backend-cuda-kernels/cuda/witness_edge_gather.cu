// Device component edges (the device DAG, B3): gather a CONSUMER component's
// input columns directly from a PRODUCER's word-major sub buffer — the
// device-to-device replacement for the host input-list feed (D2H → packed
// rebuild → add_packed_inputs → consumer re-reads).
//
// Layout contract (matches the host feed semantics exactly):
//   producer sub word (word_base + j*words_per_instance + k) at producer row r
//     == consumer input column k at consumer row (j*producer_rows + r)
// because the host feeds instance j's full column before instance j+1
// (`.iter().for_each(add_packed_inputs)` in declaration order) and each feed
// appends producer_rows rows.
//
// Consumer padding: the consumer's writer pads the STACKED input set to the
// next power of two by repeating its FIRST PACKED ROW (16 lanes). The gather
// replicates rows [0, 16) of the stacked set into every padding row group,
// byte-identical to the host `resize(packed_size, first)` on packed inputs.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void witness_edge_gather_kernel(
    const uint32_t *producer_sub,
    uint32_t producer_rows,
    uint32_t word_base,
    uint32_t words_per_instance,
    uint32_t n_instances,
    uint32_t consumer_rows, // padded (power of two, >= n_instances*producer_rows)
    uint32_t *const *consumer_cols // words_per_instance column pointers
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= consumer_rows) {
        return;
    }
    uint32_t real_rows = n_instances * producer_rows;
    // Padding rows replicate the first PACKED row's lanes (row % 16 of rows 0..16).
    uint32_t src_row = row < real_rows ? row : (row & 15u);
    uint32_t j = src_row / producer_rows;
    uint32_t r = src_row % producer_rows;
    for (uint32_t k = 0; k < words_per_instance; ++k) {
        consumer_cols[k][row] =
            producer_sub[(size_t)(word_base + (size_t)j * words_per_instance + k) *
                             producer_rows +
                         r];
    }
}

// Returns 0 on success. All pointers are DEVICE pointers; `consumer_cols_dev`
// is a device array of `words_per_instance` column pointers, each
// `consumer_rows` long.
extern "C" int stwo_witness_edge_gather(
    const uint32_t *producer_sub_dev,
    uint32_t producer_rows,
    uint32_t word_base,
    uint32_t words_per_instance,
    uint32_t n_instances,
    uint32_t consumer_rows,
    uint32_t *const *consumer_cols_dev
) {
    if (consumer_rows == 0 || words_per_instance == 0) {
        return 0;
    }
    if ((size_t)n_instances * producer_rows > consumer_rows) {
        fprintf(stderr, "stwo_witness_edge_gather: consumer_rows too small\n");
        return 1;
    }
    const uint32_t block = 256;
    uint32_t grid = (consumer_rows + block - 1) / block;
    witness_edge_gather_kernel<<<grid, block>>>(
        producer_sub_dev, producer_rows, word_base, words_per_instance, n_instances,
        consumer_rows, consumer_cols_dev);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "stwo_witness_edge_gather: launch failed\n");
        return 1;
    }
    return 0;
}
