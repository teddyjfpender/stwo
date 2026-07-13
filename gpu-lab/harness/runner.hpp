#pragma once

#include "effects.hpp"
#include "replay.hpp"

#include <cuda.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_lab {

struct Validation {
  bool passed = false;
  std::size_t checked_words = 0;
  std::string error;
};

struct Sample {
  double gpu_ms = 0.0;
  double submit_us = 0.0;
  double wall_ms = 0.0;
};

struct Series {
  std::vector<Sample> samples;
  int warmup_iterations = 0;
  bool warmup_stable = false;
};

struct Benchmark {
  Series eager;
  Series graph;
  Validation post_eager;
  Validation post_graph;
  double budget_ms = 0.0;
  double budget_used_ms = 0.0;
  double budget_remaining_ms = 0.0;
  std::string error;

  bool passed() const {
    return error.empty() && post_eager.passed && post_graph.passed;
  }
};

class Runner {
public:
  Runner(const Options &options, const Plan &plan, const Replay &replay);
  Runner(const Runner &) = delete;
  Runner &operator=(const Runner &) = delete;
  ~Runner();

  Validation eager_correctness(bool mutated);
  Validation graph_correctness(bool mutated);
  Benchmark benchmark();
  std::string device_name() const;
  std::string device_uuid() const;
  std::size_t total_memory_bytes() const;
  int device_attribute(CUdevice_attribute attribute) const;
  int function_attribute(CUfunction_attribute attribute) const;
  int driver_version() const;
  std::uint32_t grid_x() const;
  double graph_capture_ms() const;
  double graph_instantiate_ms() const;

private:
  struct GuardedRegion {
    CUdeviceptr base = 0;
    CUdeviceptr data = 0;
    EffectExpectation expectation;
  };

  void cleanup() noexcept;
  void validate_launch_limits();
  CUdeviceptr allocate_bytes(std::size_t bytes, const std::string &label);
  CUdeviceptr upload_bytes(const void *bytes, std::size_t size,
                           const std::string &label);
  CUdeviceptr upload_words(const std::uint32_t *words, std::size_t count,
                           const std::string &label);
  CUdeviceptr upload_pointers(const std::vector<CUdeviceptr> &pointers,
                              const std::string &label);
  void upload();
  void bind_inputs(bool mutated);
  void reset_outputs();
  void launch_eager();
  void ensure_graph();
  Sample measure_once(bool graph, CUevent start, CUevent stop);
  Series measure_series(bool graph,
                        std::chrono::steady_clock::time_point budget_start,
                        double series_limit_ms);
  Validation validate_outputs(bool mutated);
  bool validate_effects(Validation &validation);

  const Options &options_;
  const Plan &plan_;
  const Replay &replay_;
  CUdevice device_ = 0;
  CUcontext previous_context_ = nullptr;
  CUcontext context_ = nullptr;
  CUstream stream_ = nullptr;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  CUgraph graph_ = nullptr;
  CUgraphExec graph_executable_ = nullptr;
  bool primary_retained_ = false;
  double graph_capture_ms_ = 0.0;
  double graph_instantiate_ms_ = 0.0;
  std::vector<CUdeviceptr> allocations_;
  std::vector<GuardedRegion> guarded_regions_;
  std::vector<CUdeviceptr> input_columns_;
  CUdeviceptr input_pointer_table_ = 0;
  CUdeviceptr table_ = 0;
  CUdeviceptr table_pointer_table_ = 0;
  CUdeviceptr table_strides_ = 0;
  std::vector<CUdeviceptr> output_columns_;
  CUdeviceptr output_pointer_table_ = 0;
  CUdeviceptr mult_count_pointer_table_ = 0;
  CUdeviceptr lookup_ = 0;
  CUdeviceptr sub_ = 0;
};

} // namespace gpu_lab
