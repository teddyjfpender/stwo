#include "fri_round6.hpp"
#include "fri_round6_runner.hpp"
#include "identity.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace gpu_lab {
namespace {

struct FriOptions {
  std::filesystem::path module;
  std::string module_sha256;
  std::filesystem::path build_recipe;
  std::string build_recipe_sha256;
  std::filesystem::path primary;
  std::string primary_sha256;
  std::filesystem::path primary_index;
  std::string primary_index_sha256;
  std::filesystem::path hostile;
  std::string hostile_sha256;
  std::filesystem::path hostile_index;
  std::string hostile_index_sha256;
  std::filesystem::path result;
  int device = 0;
  std::uint32_t target_sm = 0;
};

int parse_int(const std::string &value, const char *label) {
  std::size_t consumed = 0;
  int parsed = std::stoi(value, &consumed);
  if (consumed != value.size())
    throw std::runtime_error(std::string("invalid ") + label);
  return parsed;
}

FriOptions parse_options(int argc, char **argv) {
  static const std::vector<std::string> allowed = {
      "module",          "module-sha256",        "build-recipe",
      "build-recipe-sha256", "primary",          "primary-sha256",
      "primary-index",   "primary-index-sha256", "hostile",
      "hostile-sha256",  "hostile-index",        "hostile-index-sha256",
      "result",          "device",               "target-sm",
  };
  std::map<std::string, std::string> values;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::strncmp(argv[index], "--", 2) != 0)
      throw std::runtime_error("arguments must be --name value pairs");
    const std::string name = argv[index] + 2;
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
      throw std::runtime_error("unknown option --" + name);
    if (!values.emplace(name, argv[index + 1]).second)
      throw std::runtime_error("duplicate option --" + name);
  }
  auto require = [&](const char *name) -> std::string {
    const auto found = values.find(name);
    if (found == values.end() || found->second.empty())
      throw std::runtime_error(std::string("missing --") + name);
    return found->second;
  };
  FriOptions options;
  options.module = require("module");
  options.module_sha256 = require("module-sha256");
  options.build_recipe = require("build-recipe");
  options.build_recipe_sha256 = require("build-recipe-sha256");
  options.primary = require("primary");
  options.primary_sha256 = require("primary-sha256");
  options.primary_index = require("primary-index");
  options.primary_index_sha256 = require("primary-index-sha256");
  options.hostile = require("hostile");
  options.hostile_sha256 = require("hostile-sha256");
  options.hostile_index = require("hostile-index");
  options.hostile_index_sha256 = require("hostile-index-sha256");
  options.result = require("result");
  options.device = parse_int(require("device"), "device");
  const int target_sm = parse_int(require("target-sm"), "target-sm");
  if (options.device < 0 || target_sm < 50 || target_sm > 999)
    throw std::runtime_error("unsupported device ordinal or target SM");
  for (const std::string *digest :
       {&options.module_sha256, &options.build_recipe_sha256,
        &options.primary_sha256, &options.primary_index_sha256,
        &options.hostile_sha256, &options.hostile_index_sha256})
    if (!is_sha256_hex(*digest))
      throw std::runtime_error("all identities must be lowercase SHA256");
  options.target_sm = static_cast<std::uint32_t>(target_sm);
  return options;
}

std::string json_escape(const std::string &text) {
  std::ostringstream output;
  for (unsigned char character : text) {
    if (character == '\\' || character == '"')
      output << '\\' << character;
    else if (character == '\n')
      output << "\\n";
    else if (character < 0x20)
      throw std::runtime_error("result text contains an unsupported control byte");
    else
      output << character;
  }
  return output.str();
}

void append_validation(std::ostringstream &output, const char *name,
                       const FriValidation &validation, bool comma = true) {
  output << "    \"" << name << "\": {\"passed\": "
         << (validation.passed ? "true" : "false")
         << ", \"checked_words\": " << validation.checked_words
         << ", \"error\": \"" << json_escape(validation.error) << "\"}"
         << (comma ? "," : "") << "\n";
}

std::string make_result(const FriOptions &options, FriRound6Runner &runner,
                        const FriValidation &primary_eager,
                        const FriValidation &primary_graph,
                        const FriValidation &hostile_eager,
                        const FriValidation &hostile_graph,
                        bool stale_cursor_rejected,
                        const std::string &stale_cursor_error,
                        const FriValidation &reset_replay) {
  const bool passed = primary_eager.passed && primary_graph.passed &&
                      hostile_eager.passed && hostile_graph.passed &&
                      stale_cursor_rejected && reset_replay.passed;
  std::ostringstream output;
  output << "{\n"
         << "  \"schema_version\": \"stwo.gpu-lab.fri-round6-result.v1\",\n"
         << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
         << "  \"performance_admissible\": false,\n"
         << "  \"segment\": \"GraphSegment::FriLayer(7)/FriRound(6)\",\n"
         << "  \"module_content_sha256\": \"" << options.module_sha256
         << "\",\n"
         << "  \"build_recipe_sha256\": \"" << options.build_recipe_sha256
         << "\",\n"
         << "  \"primary_fixture_sha256\": \"" << options.primary_sha256
         << "\",\n"
         << "  \"primary_fixture_index_sha256\": \""
         << options.primary_index_sha256 << "\",\n"
         << "  \"hostile_fixture_sha256\": \"" << options.hostile_sha256
         << "\",\n"
         << "  \"hostile_fixture_index_sha256\": \""
         << options.hostile_index_sha256 << "\",\n"
         << "  \"harness_executable_sha256\": \""
         << digest_hex(current_executable_sha256()) << "\",\n"
         << "  \"device\": {\"name\": \"" << json_escape(runner.device_name())
         << "\", \"uuid\": \"" << runner.device_uuid()
         << "\", \"ordinal\": " << options.device
         << ", \"target_sm\": " << options.target_sm
         << ", \"driver_version\": " << runner.driver_version() << "},\n"
         << "  \"graph_contract\": {\"kernels\": "
         << runner.graph_kernel_nodes() << ", \"device_copies\": "
         << runner.graph_copy_nodes() << ", "
            "\"entry_log\": 6, \"exit_log\": 3, \"packed_leaf_log\": 2},\n"
         << "  \"correctness\": {\n";
  append_validation(output, "primary_eager", primary_eager);
  append_validation(output, "primary_graph", primary_graph);
  append_validation(output, "hostile_eager", hostile_eager);
  append_validation(output, "hostile_graph", hostile_graph);
  output << "    \"stale_cursor_status_order\": {\"passed\": "
         << (stale_cursor_rejected ? "true" : "false")
         << ", \"error\": \"" << json_escape(stale_cursor_error) << "\"},\n";
  append_validation(output, "reset_replay", reset_replay, false);
  output << "  }\n}\n";
  return output.str();
}

void write_atomic(const std::filesystem::path &destination,
                  const std::string &content) {
  std::filesystem::path temporary = destination;
  temporary += ".tmp";
  std::error_code error;
  std::filesystem::remove(temporary, error);
  if (error)
    throw std::runtime_error("cannot remove stale result temporary");
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    output.close();
    if (!output)
      throw std::runtime_error("cannot write complete FRI result");
  }
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("cannot atomically install FRI result: " +
                             error.message());
  }
}

void remove_stale_result(const std::filesystem::path &result) {
  std::error_code error;
  std::filesystem::remove(result, error);
  if (error)
    throw std::runtime_error("cannot remove stale FRI result: " +
                             error.message());
  auto temporary = result;
  temporary += ".tmp";
  std::filesystem::remove(temporary, error);
  if (error)
    throw std::runtime_error("cannot remove stale FRI result temporary: " +
                             error.message());
}

void verify_bound_file(const std::filesystem::path &path,
                       const std::string &expected_sha256,
                       const char *label) {
  if (digest_hex(sha256_file(path)) != expected_sha256)
    throw std::runtime_error(std::string(label) + " SHA256 mismatch");
}

} // namespace
} // namespace gpu_lab

int main(int argc, char **argv) {
  using namespace gpu_lab;
  try {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
      primitive_self_test();
      fri_round6_layout_self_test();
      std::cout << "stwo-gpu-lab FRI round-6 self-test: PASS\n";
      return 0;
    }
    const FriOptions options = parse_options(argc, argv);
    remove_stale_result(options.result);
    verify_bound_file(options.build_recipe, options.build_recipe_sha256,
                      "build recipe");
    verify_bound_file(options.primary_index, options.primary_index_sha256,
                      "primary fixture index");
    verify_bound_file(options.hostile_index, options.hostile_index_sha256,
                      "hostile fixture index");
    const FriRound6Fixture primary =
        load_fri_round6_fixture(options.primary, options.primary_sha256);
    const FriRound6Fixture hostile =
        load_fri_round6_fixture(options.hostile, options.hostile_sha256);
    validate_fri_round6_pair(primary, hostile);
    FriRound6Runner runner(options.device, options.target_sm, options.module,
                           options.module_sha256, primary);
    const FriValidation primary_eager = runner.run_eager(primary);
    const FriValidation primary_graph = runner.run_graph(primary);
    const FriValidation hostile_eager = runner.run_eager(hostile);
    const FriValidation hostile_graph = runner.run_graph(hostile);
    std::string stale_cursor_error;
    const bool stale_cursor =
        runner.stale_cursor_sets_order_status(primary, stale_cursor_error);
    const FriValidation reset_replay = runner.run_graph(primary);
    const std::string result =
        make_result(options, runner, primary_eager, primary_graph,
                    hostile_eager, hostile_graph, stale_cursor,
                    stale_cursor_error, reset_replay);
    write_atomic(options.result, result);
    std::cout << result;
    return primary_eager.passed && primary_graph.passed &&
                   hostile_eager.passed && hostile_graph.passed &&
                   stale_cursor && reset_replay.passed
               ? 0
               : 2;
  } catch (const std::exception &error) {
    std::cerr << "stwo-gpu-lab-fri-round6: " << error.what() << '\n';
    return 1;
  }
}
