#include "result.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace gpu_lab {
namespace fs = std::filesystem;
namespace {

std::string json_escape(const std::string &input) {
  std::ostringstream output;
  for (unsigned char character : input) {
    switch (character) {
    case '\\':
      output << "\\\\";
      break;
    case '"':
      output << "\\\"";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20) {
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<int>(character) << std::dec;
      } else {
        output << character;
      }
    }
  }
  return output.str();
}

std::vector<double> sample_values(const Series &series, int field) {
  std::vector<double> result;
  result.reserve(series.samples.size());
  for (const Sample &sample : series.samples) {
    if (field == 0)
      result.push_back(sample.gpu_ms);
    else if (field == 1)
      result.push_back(sample.submit_us);
    else
      result.push_back(sample.wall_ms);
  }
  return result;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double standard_deviation(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  double squared = 0.0;
  for (double value : values)
    squared += (value - mean) * (value - mean);
  return std::sqrt(squared / static_cast<double>(values.size()));
}

void append_raw(std::ostringstream &output, const std::vector<double> &values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      output << ',';
    output << values[index];
  }
  output << ']';
}

void append_series(std::ostringstream &output, const Series &series) {
  const auto gpu_ms = sample_values(series, 0);
  const auto submit_us = sample_values(series, 1);
  const auto wall_ms = sample_values(series, 2);
  output << "{\"warmup_iterations\":" << series.warmup_iterations
         << ",\"warmup_stable\":" << (series.warmup_stable ? "true" : "false")
         << ",\"iterations\":" << series.samples.size()
         << ",\"p5_gpu_ms\":" << percentile(gpu_ms, 0.05)
         << ",\"p50_gpu_ms\":" << percentile(gpu_ms, 0.50)
         << ",\"p95_gpu_ms\":" << percentile(gpu_ms, 0.95)
         << ",\"stddev_gpu_ms\":" << standard_deviation(gpu_ms)
         << ",\"p50_submit_us\":" << percentile(submit_us, 0.50)
         << ",\"p95_submit_us\":" << percentile(submit_us, 0.95)
         << ",\"p50_wall_ms\":" << percentile(wall_ms, 0.50)
         << ",\"p95_wall_ms\":" << percentile(wall_ms, 0.95)
         << ",\"p95_exploratory\":"
         << (series.samples.size() < 30 ? "true" : "false")
         << ",\"raw_gpu_ms\":";
  append_raw(output, gpu_ms);
  output << ",\"raw_submit_us\":";
  append_raw(output, submit_us);
  output << ",\"raw_wall_ms\":";
  append_raw(output, wall_ms);
  output << '}';
}

std::string benchmark_failure(const Benchmark &benchmark) {
  if (!benchmark.error.empty())
    return benchmark.error;
  if (!benchmark.post_eager.passed)
    return benchmark.post_eager.error.empty()
               ? "post-eager output validation failed"
               : benchmark.post_eager.error;
  if (!benchmark.post_graph.passed)
    return benchmark.post_graph.error.empty()
               ? "post-graph output validation failed"
               : benchmark.post_graph.error;
  return "benchmark did not satisfy its acceptance contract";
}

} // namespace

std::string make_result(const Options &options, const Plan &plan,
                        Runner &runner, const Validation &eager,
                        const Validation &graph,
                        const Validation &mutated_eager,
                        const Validation &mutated_graph,
                        const Benchmark *benchmark) {
  const bool post_benchmark_passed =
      benchmark == nullptr || benchmark->passed();
  const bool passed = eager.passed && graph.passed && mutated_eager.passed &&
                      mutated_graph.passed && post_benchmark_passed;
  std::ostringstream output;
  output << std::fixed << std::setprecision(6);
  output
      << "{\n"
      << "  \"schema_version\": \"stwo.gpu-lab.result.v1\",\n"
      << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
      << "  \"post_benchmark_passed\": "
      << (post_benchmark_passed ? "true" : "false") << ",\n"
      << "  \"mode\": \"" << options.mode << "\",\n"
      << "  \"semantic_fixture_sha256\": \"" << digest_hex(plan.fixture_sha256)
      << "\",\n"
      << "  \"module_content_sha256\": \"" << digest_hex(plan.module_sha256)
      << "\",\n"
      << "  \"build_recipe_hash\": \"" << digest_hex(plan.build_recipe_hash)
      << "\",\n"
      << "  \"host_oracle_index_sha256\": \""
      << digest_hex(plan.oracle_index_sha256) << "\",\n"
      << "  \"execution_manifest_sha256\": \"" << options.manifest_sha256
      << "\",\n"
      << "  \"plan_sha256\": \"" << options.plan_sha256 << "\",\n"
      << "  \"replay_sha256\": \"" << digest_hex(plan.replay_sha256) << "\",\n"
      << "  \"harness_executable_sha256\": \""
      << digest_hex(current_executable_sha256()) << "\",\n"
      << "  \"device\": {\"name\": \"" << json_escape(runner.device_name())
      << "\", \"uuid\": \"" << runner.device_uuid()
      << "\", \"ordinal\": " << options.device << ", \"sm_major\": "
      << runner.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
      << ", \"sm_minor\": "
      << runner.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
      << ", \"target_sm\": " << plan.target_sm
      << ", \"driver_version\": " << runner.driver_version()
      << ", \"total_memory_bytes\": " << runner.total_memory_bytes()
      << ", \"multiprocessor_count\": "
      << runner.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
      << ", \"clock_rate_khz\": "
      << runner.device_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
      << ", \"memory_clock_rate_khz\": "
      << runner.device_attribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
      << ", \"memory_bus_width_bits\": "
      << runner.device_attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
      << ", \"ecc_enabled\": "
      << (runner.device_attribute(CU_DEVICE_ATTRIBUTE_ECC_ENABLED) ? "true"
                                                                   : "false")
      << "},\n"
      << "  \"launch\": {\"symbol\": \"" << json_escape(plan.symbol)
      << "\", \"grid\": [" << runner.grid_x() << ",1,1], \"block\": ["
      << plan.block_x << ',' << plan.block_y << ',' << plan.block_z
      << "], \"dynamic_shared_bytes\": " << plan.dynamic_shared_bytes << "},\n"
      << "  \"correctness\": {\"eager_passed\": "
      << (eager.passed ? "true" : "false")
      << ", \"graph_passed\": " << (graph.passed ? "true" : "false")
      << ", \"mutated_eager_passed\": "
      << (mutated_eager.passed ? "true" : "false")
      << ", \"mutated_graph_passed\": "
      << (mutated_graph.passed ? "true" : "false")
      << ", \"eager_checked_words\": " << eager.checked_words
      << ", \"graph_checked_words\": " << graph.checked_words
      << ", \"mutated_eager_checked_words\": "
      << mutated_eager.checked_words
      << ", \"mutated_graph_checked_words\": "
      << mutated_graph.checked_words
      << ", \"eager_error\": \"" << json_escape(eager.error)
      << "\", \"graph_error\": \"" << json_escape(graph.error) << '"';
  output << ", \"mutated_eager_error\": \""
         << json_escape(mutated_eager.error)
         << "\", \"mutated_graph_error\": \""
         << json_escape(mutated_graph.error) << '"';
  if (benchmark) {
    output << ", \"post_eager_passed\": "
           << (benchmark->post_eager.passed ? "true" : "false")
           << ", \"post_graph_passed\": "
           << (benchmark->post_graph.passed ? "true" : "false")
           << ", \"post_eager_checked_words\": "
           << benchmark->post_eager.checked_words
           << ", \"post_graph_checked_words\": "
           << benchmark->post_graph.checked_words
           << ", \"post_eager_error\": \""
           << json_escape(benchmark->post_eager.error)
           << "\", \"post_graph_error\": \""
           << json_escape(benchmark->post_graph.error) << '"';
  }
  output << "},\n"
         << "  \"resources\": {\"registers_per_thread\": "
         << runner.function_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS)
         << ", \"static_shared_bytes\": "
         << runner.function_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
         << ", \"local_bytes_per_thread\": "
         << runner.function_attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
         << "},\n";
  if (benchmark && benchmark->passed()) {
    output << "  \"timing\": {\"post_benchmark_passed\": true"
           << ", \"performance_admissible\": true"
           << ", \"budget_ms\": " << benchmark->budget_ms
           << ", \"budget_used_ms\": " << benchmark->budget_used_ms
           << ", \"budget_remaining_ms\": " << benchmark->budget_remaining_ms
           << ", \"graph_capture_ms\": " << runner.graph_capture_ms()
           << ", \"graph_instantiate_ms\": " << runner.graph_instantiate_ms()
           << ", \"eager\": ";
    append_series(output, benchmark->eager);
    output << ", \"graph\": ";
    append_series(output, benchmark->graph);
    output << "}\n";
  } else if (benchmark) {
    output << "  \"timing\": {\"post_benchmark_passed\": false"
           << ", \"performance_admissible\": false, \"error\": \""
           << json_escape(benchmark_failure(*benchmark)) << "\"}\n";
  } else {
    output << "  \"timing\": {}\n";
  }
  output << "}\n";
  return output.str();
}

void remove_old_result(const fs::path &path) {
  std::error_code error;
  fs::remove(path, error);
  if (error)
    throw std::runtime_error("cannot remove old result: " + error.message());
}

void write_result_atomic(const fs::path &destination,
                         const std::string &content) {
  fs::path temporary = destination;
  temporary += ".tmp";
  std::error_code error;
  fs::remove(temporary, error);
  if (error)
    throw std::runtime_error("cannot remove stale result temporary: " +
                             error.message());
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot open result temporary for writing");
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    output.flush();
    if (!output) {
      output.close();
      fs::remove(temporary, error);
      throw std::runtime_error("failed to write complete result");
    }
    output.close();
    if (!output) {
      fs::remove(temporary, error);
      throw std::runtime_error("failed to close complete result");
    }
  }
  fs::rename(temporary, destination, error);
  if (error) {
    fs::remove(temporary);
    throw std::runtime_error("cannot atomically install result: " +
                             error.message());
  }
}

} // namespace gpu_lab
