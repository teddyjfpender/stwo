#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <new>

// ---------------------------------------------------------------------------
// Resource-owning execution context for ONE resident proof (design §19).
//
// The core prove path is stream-0-centric: every kernel + every pool alloc/free
// is ordered on the legacy default stream (see cuda_mem_pool.cuh). To run two
// proofs concurrently, each must own an independent CUDA stream AND its own
// stream-ordered memory pool: two contexts allocating from DISJOINT pools have no
// cross-stream reuse hazard, whereas two streams sharing the single global default
// pool have no ordering guarantee between one's cudaFreeAsync and the other's
// kernels (the central concurrency hazard in the recon).
//
// A context is NOT just a stream pointer — it owns the stream and the pool, and all
// alloc / free / kernel work for the proof's streamful island routes through its
// stream, so buffers free stream-ordered AFTER their last use on that stream. No
// stream-0 free ever races another stream's kernels (the resource-lifetime bug).
//
// Handles are opaque `void*` (StwoExecContext*), matching the FFI style of the rest
// of this crate. Pool creation fails closed: a live context is always isolated.
// ---------------------------------------------------------------------------

namespace {
struct StwoExecContext {
    cudaStream_t stream;
    cudaMemPool_t pool;
};

StwoExecContext *context_from(void *handle) {
    return static_cast<StwoExecContext *>(handle);
}

cudaError_t first_error(cudaError_t current, cudaError_t candidate) {
    return current == cudaSuccess ? candidate : current;
}

__global__ void fill_u32_kernel(uint32_t *dst, uint32_t value, size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = value;
    }
}
}  // namespace

// Create a context: a non-blocking stream + its own never-release memory pool on
// the current device. Isolation is part of the contract, so pool creation fails
// closed instead of silently falling back to the process-wide default pool.
extern "C" int stwo_exec_context_create(void **out_handle) {
    if (out_handle == nullptr) {
        return cudaErrorInvalidValue;
    }
    *out_handle = nullptr;

    StwoExecContext *ctx = new (std::nothrow) StwoExecContext();
    if (ctx == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    ctx->stream = nullptr;
    ctx->pool = nullptr;

    cudaError_t err = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        delete ctx;
        return err;
    }

    int device_id = 0;
    err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return err;
    }

    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device_id;
    err = cudaMemPoolCreate(&ctx->pool, &props);
    if (err != cudaSuccess || ctx->pool == nullptr) {
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return err == cudaSuccess ? cudaErrorMemoryAllocation : err;
    }

    // Never release back to the OS between warm proves, so a workspace can retain
    // its slab for the full capture epoch without allocator churn.
    uint64_t threshold = UINT64_MAX;
    err = cudaMemPoolSetAttribute(ctx->pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (err != cudaSuccess) {
        cudaMemPoolDestroy(ctx->pool);
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return err;
    }

    *out_handle = ctx;
    return cudaSuccess;
}

// Sync + tear down. Synchronizes the stream first so no pending free/kernel outlives
// the pool/stream it references.
extern "C" int stwo_exec_context_destroy(void *handle) {
    if (handle == nullptr) {
        return cudaSuccess;
    }
    StwoExecContext *ctx = context_from(handle);
    cudaError_t err = cudaSuccess;
    if (ctx->stream != nullptr) {
        err = first_error(err, cudaStreamSynchronize(ctx->stream));
        err = first_error(err, cudaStreamDestroy(ctx->stream));
    }
    if (ctx->pool != nullptr) {
        err = first_error(err, cudaMemPoolDestroy(ctx->pool));
    }
    delete ctx;
    return err;
}

// Block the host until all work enqueued on the context's stream has completed.
extern "C" int stwo_exec_context_sync(void *handle) {
    if (handle == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    return cudaStreamSynchronize(context_from(handle)->stream);
}

// The context's stream as an opaque handle, to pass to `_on(stream)` kernel variants.
extern "C" int stwo_exec_context_stream(void *handle, void **out_stream) {
    if (handle == nullptr || out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    *out_stream = reinterpret_cast<void *>(context_from(handle)->stream);
    return *out_stream == nullptr ? cudaErrorInvalidResourceHandle : cudaSuccess;
}

// Allocate `count` u32 from the context's pool, ordered on its stream. The returned
// pointer is usable by later work on the SAME stream (stream-ordered allocation
// guarantees the memory is backed before any later same-stream consumer runs).
// Returns a checked CUDA status and writes a non-null pointer on success.
extern "C" int stwo_exec_context_alloc_u32(
    void *handle, size_t count, uint32_t **out_ptr
) {
    if (handle == nullptr || out_ptr == nullptr || count == 0 ||
        count > SIZE_MAX / sizeof(uint32_t)) {
        return cudaErrorInvalidValue;
    }
    *out_ptr = nullptr;
    StwoExecContext *ctx = context_from(handle);
    uint32_t *ptr = nullptr;
    size_t bytes = count * sizeof(uint32_t);
    cudaError_t err =
        cudaMallocFromPoolAsync(reinterpret_cast<void **>(&ptr), bytes, ctx->pool, ctx->stream);
    if (err != cudaSuccess) {
        return err;
    }
    if (ptr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    *out_ptr = ptr;
    return cudaSuccess;
}

// Free a buffer on the context's stream: ordered AFTER all prior same-stream users,
// so no use-after-free for same-stream work and no cross-stream reuse hazard.
extern "C" int stwo_exec_context_free_u32(void *handle, uint32_t *ptr) {
    if (handle == nullptr || ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    return cudaFreeAsync(ptr, context_from(handle)->stream);
}

// Stream-explicit memory operations used to seed/read stable arena slots around
// graph launches. They enqueue only; callers fence with stwo_exec_context_sync.
extern "C" int stwo_exec_context_memset_async(
    void *handle, void *dst, int value, size_t bytes
) {
    if (handle == nullptr || (dst == nullptr && bytes != 0)) {
        return cudaErrorInvalidValue;
    }
    if (bytes == 0) {
        return cudaSuccess;
    }
    return cudaMemsetAsync(dst, value, bytes, context_from(handle)->stream);
}

extern "C" int stwo_exec_context_fill_u32_async(
    void *handle, uint32_t *dst, uint32_t value, size_t count
) {
    if (handle == nullptr || (dst == nullptr && count != 0)) {
        return cudaErrorInvalidValue;
    }
    if (count == 0) {
        return cudaSuccess;
    }
    constexpr uint32_t block = 256;
    size_t blocks = (count + block - 1u) / block;
    if (blocks > static_cast<size_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    fill_u32_kernel<<<static_cast<uint32_t>(blocks), block, 0,
                      context_from(handle)->stream>>>(dst, value, count);
    return cudaGetLastError();
}

extern "C" int stwo_exec_context_memcpy_d2d_async(
    void *handle, void *dst, const void *src, size_t bytes
) {
    if (handle == nullptr || ((dst == nullptr || src == nullptr) && bytes != 0)) {
        return cudaErrorInvalidValue;
    }
    if (bytes == 0) {
        return cudaSuccess;
    }
    return cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyDeviceToDevice, context_from(handle)->stream);
}

extern "C" int stwo_exec_context_memcpy_h2d_async(
    void *handle, void *dst, const void *src, size_t bytes
) {
    if (handle == nullptr || ((dst == nullptr || src == nullptr) && bytes != 0)) {
        return cudaErrorInvalidValue;
    }
    if (bytes == 0) {
        return cudaSuccess;
    }
    return cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyHostToDevice, context_from(handle)->stream);
}

extern "C" int stwo_exec_context_memcpy_d2h_async(
    void *handle, void *dst, const void *src, size_t bytes
) {
    if (handle == nullptr || ((dst == nullptr || src == nullptr) && bytes != 0)) {
        return cudaErrorInvalidValue;
    }
    if (bytes == 0) {
        return cudaSuccess;
    }
    return cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyDeviceToHost, context_from(handle)->stream);
}

// ---------------------------------------------------------------------------
// Single-stream CUDA graph lifecycle. A graph exec contains only work captured
// from the context stream; transcript boundaries and host reads remain outside.
// ---------------------------------------------------------------------------

extern "C" int stwo_graph_capture_begin(void *handle) {
    if (handle == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    return cudaStreamBeginCapture(
        context_from(handle)->stream, cudaStreamCaptureModeThreadLocal);
}

extern "C" int stwo_graph_capture_end(void *handle, void **out_exec) {
    if (handle == nullptr || out_exec == nullptr) {
        return cudaErrorInvalidValue;
    }
    *out_exec = nullptr;

    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(context_from(handle)->stream, &graph);
    if (err != cudaSuccess) {
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        return err;
    }
    if (graph == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaError_t destroy_err = cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        return err;
    }
    if (destroy_err != cudaSuccess) {
        cudaGraphExecDestroy(exec);
        return destroy_err;
    }
    if (exec == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    *out_exec = reinterpret_cast<void *>(exec);
    return cudaSuccess;
}

// CUDA has no separate abort primitive. Ending capture exits capture mode; any
// successfully produced graph is immediately discarded. On an invalidated
// capture cudaStreamEndCapture returns that status after restoring the stream.
extern "C" int stwo_graph_capture_abort(void *handle) {
    if (handle == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(context_from(handle)->stream, &graph);
    if (graph != nullptr) {
        cudaError_t destroy_err = cudaGraphDestroy(graph);
        err = first_error(err, destroy_err);
    }
    return err;
}

extern "C" int stwo_graph_launch(void *exec_handle, void *context_handle) {
    if (exec_handle == nullptr || context_handle == nullptr) {
        return cudaErrorInvalidValue;
    }
    return cudaGraphLaunch(
        reinterpret_cast<cudaGraphExec_t>(exec_handle), context_from(context_handle)->stream);
}

extern "C" int stwo_graph_destroy(void *exec_handle) {
    if (exec_handle == nullptr) {
        return cudaSuccess;
    }
    return cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec_handle));
}
