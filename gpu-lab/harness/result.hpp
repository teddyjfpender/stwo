#pragma once

#include "replay.hpp"
#include "runner.hpp"

#include <filesystem>
#include <string>

namespace gpu_lab {

std::string make_result(const Options &options, const Plan &plan,
                        Runner &runner, const Validation &eager,
                        const Validation &graph,
                        const Validation &mutated_eager,
                        const Validation &mutated_graph,
                        const Benchmark *benchmark);
void remove_old_result(const std::filesystem::path &path);
void write_result_atomic(const std::filesystem::path &destination,
                         const std::string &content);

} // namespace gpu_lab
