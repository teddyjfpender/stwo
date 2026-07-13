#pragma once

#include "identity.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_lab {

struct Plan {
  std::uint32_t binding_version = 0;
  std::uint32_t target_sm = 0;
  std::uint32_t input_columns = 0;
  std::uint32_t table_slots = 0;
  std::uint32_t stride_words = 0;
  std::uint32_t output_columns = 0;
  std::uint32_t lookup_columns = 0;
  std::uint32_t sub_columns = 0;
  std::uint32_t block_x = 0;
  std::uint32_t block_y = 0;
  std::uint32_t block_z = 0;
  std::uint32_t dynamic_shared_bytes = 0;
  std::string symbol;
  std::filesystem::path module_path;
  std::vector<std::uint8_t> module_image;
  Digest module_sha256{};
  Digest build_recipe_hash{};
  Digest fixture_sha256{};
  Digest replay_sha256{};
  Digest abi_sha256{};
  Digest oracle_index_sha256{};
};

struct Options {
  std::string plan;
  std::string plan_sha256;
  std::string replay;
  std::string manifest;
  std::string manifest_sha256;
  std::string result;
  std::string mode;
  int device = 0;
  int warmup_budget_ms = 5000;
  int budget_ms = 30000;
  int min_iterations = 5;
  int max_iterations = 500;
};

struct Replay {
  std::uint32_t rows = 0;
  std::uint32_t table_len = 0;
  std::uint32_t input_columns = 0;
  std::uint32_t output_columns = 0;
  std::uint32_t lookup_columns = 0;
  std::uint32_t sub_columns = 0;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> table;
  std::vector<std::uint32_t> expected_output;
  std::vector<std::uint32_t> expected_lookup;
  std::vector<std::uint32_t> expected_sub;
  std::vector<std::uint32_t> mutated_inputs;
  std::vector<std::uint32_t> mutated_expected_output;
  std::vector<std::uint32_t> mutated_expected_lookup;
  std::vector<std::uint32_t> mutated_expected_sub;
};

Plan load_plan(const std::filesystem::path &path,
               const std::string &expected_sha256);
Options parse_options(int argc, char **argv);
Replay load_replay(const std::filesystem::path &path, const Plan &plan);
void plan_binding_self_test();
std::size_t checked_words(std::uint32_t columns, std::uint32_t rows,
                          const char *label);

} // namespace gpu_lab
