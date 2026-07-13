#include "runner.hpp"

#include "budget.hpp"
#include "cuda_check.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gpu_lab {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double predicted_wall_ms(const std::vector<double> &wall_ms) {
  const std::size_t first = wall_ms.size() > 3 ? wall_ms.size() - 3 : 0;
  return std::max(0.01,
                  *std::max_element(wall_ms.begin() + first, wall_ms.end()));
}

double checked_budget_used_ms(Clock::time_point start, double limit_ms,
                              const char *phase) {
  const double used_ms = elapsed_ms(start);
  if (used_ms > limit_ms)
    throw std::runtime_error(
        std::string("shared benchmark budget exhausted during ") + phase);
  return used_ms;
}

} // namespace

#define CU_CHECK(call) GPU_LAB_CU_CHECK(call)

Runner::Runner(const Options &options, const Plan &plan, const Replay &replay)
    : options_(options), plan_(plan), replay_(replay) {
  try {
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device_, options_.device));
    int major = 0;
    int minor = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    CU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
    if (major * 10 + minor != static_cast<int>(plan_.target_sm))
      throw std::runtime_error(
          "actual device SM does not match execution plan target");
    validate_launch_limits();

    CU_CHECK(cuCtxGetCurrent(&previous_context_));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
    primary_retained_ = true;
    CU_CHECK(cuCtxSetCurrent(context_));
    CU_CHECK(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
    CU_CHECK(cuModuleLoadData(&module_, plan_.module_image.data()));
    CU_CHECK(cuModuleGetFunction(&function_, module_, plan_.symbol.c_str()));
    upload();
  } catch (...) {
    cleanup();
    throw;
  }
}

Runner::~Runner() { cleanup(); }

void Runner::cleanup() noexcept {
  if (context_)
    cuCtxSetCurrent(context_);
  if (stream_)
    cuStreamSynchronize(stream_);
  if (graph_executable_) {
    cuGraphExecDestroy(graph_executable_);
    graph_executable_ = nullptr;
  }
  if (graph_) {
    cuGraphDestroy(graph_);
    graph_ = nullptr;
  }
  if (stream_) {
    cuStreamDestroy(stream_);
    stream_ = nullptr;
  }
  for (auto pointer = allocations_.rbegin(); pointer != allocations_.rend();
       ++pointer)
    if (*pointer)
      cuMemFree(*pointer);
  allocations_.clear();
  guarded_regions_.clear();
  if (module_) {
    cuModuleUnload(module_);
    module_ = nullptr;
  }
  if (context_) {
    cuCtxSetCurrent(previous_context_);
    context_ = nullptr;
  }
  previous_context_ = nullptr;
  if (primary_retained_) {
    cuDevicePrimaryCtxRelease(device_);
    primary_retained_ = false;
  }
}

Validation Runner::eager_correctness(bool mutated) {
  bind_inputs(mutated);
  reset_outputs();
  CU_CHECK(cuStreamSynchronize(stream_));
  launch_eager();
  CU_CHECK(cuStreamSynchronize(stream_));
  return validate_outputs(mutated);
}

Validation Runner::graph_correctness(bool mutated) {
  ensure_graph();
  bind_inputs(mutated);
  reset_outputs();
  CU_CHECK(cuStreamSynchronize(stream_));
  CU_CHECK(cuGraphLaunch(graph_executable_, stream_));
  CU_CHECK(cuStreamSynchronize(stream_));
  return validate_outputs(mutated);
}

Benchmark Runner::benchmark() {
  ensure_graph();
  bind_inputs(false);
  Benchmark result;
  result.budget_ms = static_cast<double>(options_.budget_ms);
  const auto budget_start = Clock::now();
  const double eager_limit_ms = result.budget_ms / 2.0;
  result.eager = measure_series(false, budget_start, eager_limit_ms);
  result.post_eager = validate_outputs(false);
  checked_budget_used_ms(budget_start, eager_limit_ms, "eager post-validation");
  if (!result.post_eager.passed) {
    result.post_graph =
        Validation{false, 0, "suppressed after eager benchmark failure"};
    return result;
  }
  result.graph = measure_series(true, budget_start, result.budget_ms);
  result.post_graph = validate_outputs(false);
  result.budget_used_ms = checked_budget_used_ms(budget_start, result.budget_ms,
                                                 "graph post-validation");
  result.budget_remaining_ms =
      remaining_ms(result.budget_ms, result.budget_used_ms);
  return result;
}

std::string Runner::device_name() const {
  char name[256]{};
  CU_CHECK(cuDeviceGetName(name, sizeof(name), device_));
  return name;
}

std::string Runner::device_uuid() const {
  CUuuid uuid{};
  CU_CHECK(cuDeviceGetUuid(&uuid, device_));
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (char byte : uuid.bytes)
    output << std::setw(2)
           << static_cast<unsigned>(static_cast<unsigned char>(byte));
  return output.str();
}

std::size_t Runner::total_memory_bytes() const {
  std::size_t bytes = 0;
  CU_CHECK(cuDeviceTotalMem(&bytes, device_));
  return bytes;
}

int Runner::device_attribute(CUdevice_attribute attribute) const {
  int value = 0;
  CU_CHECK(cuDeviceGetAttribute(&value, attribute, device_));
  return value;
}

int Runner::function_attribute(CUfunction_attribute attribute) const {
  int value = 0;
  CU_CHECK(cuFuncGetAttribute(&value, attribute, function_));
  return value;
}

int Runner::driver_version() const {
  int value = 0;
  CU_CHECK(cuDriverGetVersion(&value));
  return value;
}

std::uint32_t Runner::grid_x() const {
  const std::uint64_t grid =
      (static_cast<std::uint64_t>(replay_.rows) + plan_.block_x - 1) /
      plan_.block_x;
  if (grid == 0 || grid > static_cast<std::uint64_t>(device_attribute(
                              CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)))
    throw std::runtime_error("execution plan grid exceeds device limit");
  return static_cast<std::uint32_t>(grid);
}

double Runner::graph_capture_ms() const { return graph_capture_ms_; }

double Runner::graph_instantiate_ms() const { return graph_instantiate_ms_; }

void Runner::validate_launch_limits() {
  auto attribute = [&](CUdevice_attribute key) {
    int value = 0;
    CU_CHECK(cuDeviceGetAttribute(&value, key, device_));
    return value;
  };
  const std::uint64_t threads =
      static_cast<std::uint64_t>(plan_.block_x) * plan_.block_y * plan_.block_z;
  if (threads > static_cast<std::uint64_t>(
                    attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)) ||
      plan_.block_x > static_cast<std::uint32_t>(
                          attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)) ||
      plan_.block_y > static_cast<std::uint32_t>(
                          attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)) ||
      plan_.block_z > static_cast<std::uint32_t>(
                          attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)) ||
      plan_.dynamic_shared_bytes >
          static_cast<std::uint32_t>(
              attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)))
    throw std::runtime_error("execution plan launch exceeds device limits");
}

void Runner::launch_eager() {
  std::uint32_t rows = replay_.rows;
  static_assert(sizeof(CUdeviceptr) == 8 && sizeof(rows) == 4,
                "reviewed binding requires seven u64 pointers then one u32");
  void *arguments[] = {
      &input_pointer_table_,
      &table_pointer_table_,
      &table_strides_,
      &output_pointer_table_,
      &mult_count_pointer_table_,
      &lookup_,
      &sub_,
      &rows,
  };
  static_assert(sizeof(arguments) / sizeof(arguments[0]) == 8,
                "reviewed first-slice binding has exactly eight arguments");
  CU_CHECK(cuLaunchKernel(
      function_, grid_x(), 1, 1, plan_.block_x, plan_.block_y, plan_.block_z,
      plan_.dynamic_shared_bytes, stream_, arguments, nullptr));
}

void Runner::ensure_graph() {
  if (graph_executable_)
    return;
  CUgraph captured = nullptr;
  bool capture_active = false;
  const auto capture_start = std::chrono::steady_clock::now();
  CU_CHECK(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  capture_active = true;
  try {
    launch_eager();
    const CUresult end_result = cuStreamEndCapture(stream_, &captured);
    capture_active = false;
    CU_CHECK(end_result);
    graph_capture_ms_ = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - capture_start)
                            .count();
    const auto instantiate_start = std::chrono::steady_clock::now();
#if CUDA_VERSION >= 12000
    CU_CHECK(cuGraphInstantiate(&graph_executable_, captured, 0));
#else
    CU_CHECK(cuGraphInstantiateWithFlags(&graph_executable_, captured, 0));
#endif
    graph_instantiate_ms_ =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - instantiate_start)
            .count();
    graph_ = captured;
  } catch (...) {
    if (capture_active) {
      CUgraph abandoned = nullptr;
      cuStreamEndCapture(stream_, &abandoned);
      if (abandoned)
        cuGraphDestroy(abandoned);
    }
    if (graph_executable_) {
      cuGraphExecDestroy(graph_executable_);
      graph_executable_ = nullptr;
    }
    if (captured)
      cuGraphDestroy(captured);
    throw;
  }
}

Sample Runner::measure_once(bool graph, CUevent start, CUevent stop) {
  const auto wall_start = std::chrono::steady_clock::now();
  reset_outputs();
  CU_CHECK(cuStreamSynchronize(stream_));
  CU_CHECK(cuEventRecord(start, stream_));
  const auto submit_start = std::chrono::steady_clock::now();
  if (graph)
    CU_CHECK(cuGraphLaunch(graph_executable_, stream_));
  else
    launch_eager();
  const auto submit_stop = std::chrono::steady_clock::now();
  CU_CHECK(cuEventRecord(stop, stream_));
  CU_CHECK(cuEventSynchronize(stop));
  const auto wall_stop = std::chrono::steady_clock::now();
  float gpu_ms = 0.0f;
  CU_CHECK(cuEventElapsedTime(&gpu_ms, start, stop));
  return {
      static_cast<double>(gpu_ms),
      std::chrono::duration<double, std::micro>(submit_stop - submit_start)
          .count(),
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count(),
  };
}

Series Runner::measure_series(bool graph, Clock::time_point budget_start,
                              double series_limit_ms) {
  CUevent start = nullptr;
  CUevent stop = nullptr;
  CU_CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
  try {
    CU_CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    Series result;
    const auto warmup_start = std::chrono::steady_clock::now();
    std::vector<double> warmup_gpu_ms;
    std::vector<double> warmup_wall_ms;
    while (true) {
      if (!warmup_gpu_ms.empty()) {
        const double predicted_ms = predicted_wall_ms(warmup_wall_ms);
        const double warmup_used_ms = elapsed_ms(warmup_start);
        const double shared_used_ms = elapsed_ms(budget_start);
        const double required_ms =
            predicted_ms * (static_cast<double>(options_.min_iterations) + 1.0);
        if (result.warmup_stable ||
            !sample_fits(options_.warmup_budget_ms, warmup_used_ms,
                         predicted_ms) ||
            !sample_fits(series_limit_ms, shared_used_ms, required_ms))
          break;
      }
      const Sample sample = measure_once(graph, start, stop);
      warmup_gpu_ms.push_back(sample.gpu_ms);
      warmup_wall_ms.push_back(sample.wall_ms);
      ++result.warmup_iterations;
      if (warmup_gpu_ms.size() >= 3) {
        const auto begin = warmup_gpu_ms.end() - 3;
        const auto [low, high] =
            std::minmax_element(begin, warmup_gpu_ms.end());
        result.warmup_stable = *low > 0.0 && (*high / *low) <= 1.05;
      }
      checked_budget_used_ms(budget_start, series_limit_ms,
                             graph ? "graph warmup" : "eager warmup");
    }

    std::vector<double> observed_wall_ms = warmup_wall_ms;
    while (static_cast<int>(result.samples.size()) < options_.max_iterations) {
      const double predicted_ms = predicted_wall_ms(observed_wall_ms);
      if (!sample_fits(series_limit_ms, elapsed_ms(budget_start),
                       predicted_ms)) {
        if (static_cast<int>(result.samples.size()) < options_.min_iterations)
          throw std::runtime_error(
              std::string("shared benchmark budget cannot fit ") +
              (graph ? "graph" : "eager") + " minimum samples");
        break;
      }
      const Sample sample = measure_once(graph, start, stop);
      result.samples.push_back(sample);
      observed_wall_ms.push_back(sample.wall_ms);
      checked_budget_used_ms(budget_start, series_limit_ms,
                             graph ? "graph measurement" : "eager measurement");
    }
    cuEventDestroy(stop);
    cuEventDestroy(start);
    return result;
  } catch (...) {
    if (stop)
      cuEventDestroy(stop);
    if (start)
      cuEventDestroy(start);
    throw;
  }
}

} // namespace gpu_lab
