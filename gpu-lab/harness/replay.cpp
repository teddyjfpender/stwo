#include "replay.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>

namespace gpu_lab {
namespace fs = std::filesystem;
namespace {

constexpr std::uint32_t kPlanVersion = 2;
constexpr std::uint32_t kBindingVersion = 1;
constexpr const char *kFirstSliceAbiSha256 =
    "0c0302965abd6c325519b73823fe9351c174c89bbf9cb809be02214833581dce";
constexpr const char *kFirstSliceOracleIndexSha256 =
    "8f85f4a879a246b8efc88dd20615d998431456eed85835ac7a030875c5d03050";
// Fail-closed integration hook: populate only after the production wrapper and
// exporter closure pass review. Python preparation rejects the same empty seal.
constexpr std::string_view kIndexedOracleWrapperSha256 =
    "209b3022c874125699d81c6e3df3c9bfa5ffc5507a57302dd9c05fa8c9ceee0b";
constexpr const char *kFirstSliceSymbol = "stwo_jit_witness_da729390f0d884f9";

void require_reversed_rows(const std::vector<std::uint32_t> &primary,
                           const std::vector<std::uint32_t> &mutated,
                           std::uint32_t columns, std::uint32_t rows,
                           const char *label) {
  if (primary.size() != mutated.size())
    throw std::runtime_error(std::string(label) + " mutation size mismatch");
  for (std::uint32_t column = 0; column < columns; ++column)
    for (std::uint32_t row = 0; row < rows; ++row)
      if (mutated[std::size_t(column) * rows + row] !=
          primary[std::size_t(column) * rows + (rows - 1 - row)])
        throw std::runtime_error(std::string(label) +
                                 " is not the reviewed reverse-row mutation");
}

int parse_int(const std::string &value, const char *name) {
  std::size_t consumed = 0;
  int parsed = std::stoi(value, &consumed);
  if (consumed != value.size())
    throw std::runtime_error(std::string("invalid ") + name);
  return parsed;
}

void validate_first_slice_binding(const Plan &plan) {
  bool reviewed_oracle =
      plan.oracle_index_sha256 == digest_from_hex(kFirstSliceOracleIndexSha256);
  if (!reviewed_oracle && kIndexedOracleWrapperSha256.size() == 64)
    reviewed_oracle = plan.oracle_index_sha256 ==
                      digest_from_hex(std::string(kIndexedOracleWrapperSha256));
  if (plan.binding_version != kBindingVersion ||
      plan.abi_sha256 != digest_from_hex(kFirstSliceAbiSha256) ||
      !reviewed_oracle)
    throw std::runtime_error("execution plan has an unreviewed ABI binding");
  if (plan.symbol != kFirstSliceSymbol || plan.input_columns != 3 ||
      plan.table_slots != 37 || plan.stride_words != 3 ||
      plan.output_columns != 3 || plan.lookup_columns != 14 ||
      plan.sub_columns != 6 || plan.block_x != 256 || plan.block_y != 1 ||
      plan.block_z != 1 || plan.dynamic_shared_bytes != 0)
    throw std::runtime_error(
        "execution plan differs from the reviewed first-slice binding");
}

} // namespace

Plan load_plan(const fs::path &path, const std::string &expected_sha256) {
  const auto bytes = read_small_file(path, 2 * 1024 * 1024);
  if (digest_hex(sha256_bytes(bytes)) != expected_sha256)
    throw std::runtime_error("execution plan SHA256 mismatch");

  ByteCursor cursor(bytes);
  cursor.expect_magic("STWOPLN1", 8);
  const auto version = cursor.u32("version");
  Plan plan;
  plan.binding_version = cursor.u32("binding_version");
  plan.target_sm = cursor.u32("target_sm");
  plan.input_columns = cursor.u32("input_columns");
  plan.table_slots = cursor.u32("table_slots");
  plan.stride_words = cursor.u32("stride_words");
  plan.output_columns = cursor.u32("output_columns");
  plan.lookup_columns = cursor.u32("lookup_columns");
  plan.sub_columns = cursor.u32("sub_columns");
  plan.block_x = cursor.u32("block_x");
  plan.block_y = cursor.u32("block_y");
  plan.block_z = cursor.u32("block_z");
  plan.dynamic_shared_bytes = cursor.u32("dynamic_shared_bytes");
  const auto symbol_length = cursor.u32("symbol_length");
  const auto module_path_length = cursor.u32("module_path_length");
  if (version != kPlanVersion)
    throw std::runtime_error("unsupported execution plan version");
  if (symbol_length == 0 || symbol_length > 4096 || module_path_length == 0 ||
      module_path_length > 1024 * 1024)
    throw std::runtime_error("invalid execution plan string length");
  plan.symbol = cursor.string(symbol_length, "kernel symbol");
  plan.module_path = cursor.string(module_path_length, "module path");
  plan.module_sha256 = cursor.digest();
  plan.build_recipe_hash = cursor.digest();
  plan.fixture_sha256 = cursor.digest();
  plan.replay_sha256 = cursor.digest();
  plan.abi_sha256 = cursor.digest();
  plan.oracle_index_sha256 = cursor.digest();
  cursor.require_end();

  if (plan.target_sm < 50 || plan.target_sm > 999)
    throw std::runtime_error("execution plan target SM is unsupported");
  validate_first_slice_binding(plan);

  if (plan.module_path.is_relative())
    plan.module_path = fs::absolute(path).parent_path() / plan.module_path;
  plan.module_path = plan.module_path.lexically_normal();
  plan.module_image = read_small_file(plan.module_path, 256 * 1024 * 1024);
  if (plan.module_image.empty() ||
      sha256_bytes(plan.module_image) != plan.module_sha256)
    throw std::runtime_error("module content does not match execution plan");
  return plan;
}

void plan_binding_self_test() {
  Plan plan;
  plan.binding_version = kBindingVersion;
  plan.abi_sha256 = digest_from_hex(kFirstSliceAbiSha256);
  plan.oracle_index_sha256 = digest_from_hex(kFirstSliceOracleIndexSha256);
  plan.symbol = kFirstSliceSymbol;
  plan.input_columns = 3;
  plan.table_slots = 37;
  plan.stride_words = 3;
  plan.output_columns = 3;
  plan.lookup_columns = 14;
  plan.sub_columns = 6;
  plan.block_x = 256;
  plan.block_y = 1;
  plan.block_z = 1;
  validate_first_slice_binding(plan);

  auto rejected = [](const Plan &candidate) {
    try {
      validate_first_slice_binding(candidate);
      return false;
    } catch (const std::runtime_error &) {
      return true;
    }
  };
  Plan mutated = plan;
  ++mutated.binding_version;
  if (!rejected(mutated))
    throw std::runtime_error("mutated binding version was accepted");
  mutated = plan;
  mutated.abi_sha256[0] ^= 1;
  if (!rejected(mutated))
    throw std::runtime_error("mutated ABI binding was accepted");
  mutated = plan;
  mutated.oracle_index_sha256[0] ^= 1;
  if (!rejected(mutated))
    throw std::runtime_error("mutated oracle binding was accepted");
  mutated = plan;
  ++mutated.block_x;
  if (!rejected(mutated))
    throw std::runtime_error("mutated launch binding was accepted");
}

Options parse_options(int argc, char **argv) {
  static const std::vector<std::string> allowed = {
      "plan",
      "plan-sha256",
      "replay",
      "manifest-sha256",
      "manifest",
      "result",
      "mode",
      "device",
      "warmup-budget-ms",
      "budget-ms",
      "min-iterations",
      "max-iterations",
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
  auto required = [&](const char *name) -> std::string {
    auto found = values.find(name);
    if (found == values.end() || found->second.empty())
      throw std::runtime_error(std::string("missing --") + name);
    return found->second;
  };

  Options options;
  options.plan = required("plan");
  options.plan_sha256 = required("plan-sha256");
  options.replay = required("replay");
  options.manifest = required("manifest");
  options.manifest_sha256 = required("manifest-sha256");
  options.result = required("result");
  options.mode = required("mode");
  if (values.count("device"))
    options.device = parse_int(values["device"], "device");
  if (values.count("warmup-budget-ms"))
    options.warmup_budget_ms =
        parse_int(values["warmup-budget-ms"], "warmup-budget-ms");
  if (values.count("budget-ms"))
    options.budget_ms = parse_int(values["budget-ms"], "budget-ms");
  if (values.count("min-iterations"))
    options.min_iterations =
        parse_int(values["min-iterations"], "min-iterations");
  if (values.count("max-iterations"))
    options.max_iterations =
        parse_int(values["max-iterations"], "max-iterations");

  if (options.mode != "correctness" && options.mode != "benchmark")
    throw std::runtime_error("--mode must be correctness or benchmark");
  if (!is_sha256_hex(options.plan_sha256))
    throw std::runtime_error(
        "--plan-sha256 must be 64 lowercase hex characters");
  if (!is_sha256_hex(options.manifest_sha256))
    throw std::runtime_error(
        "--manifest-sha256 must be 64 lowercase hex characters");
  if (options.device < 0 || options.warmup_budget_ms <= 0 ||
      options.budget_ms <= 0 || options.min_iterations < 5 ||
      options.max_iterations < options.min_iterations)
    throw std::runtime_error("invalid device or benchmark budget");
  return options;
}

std::size_t checked_words(std::uint32_t columns, std::uint32_t rows,
                          const char *label) {
  const std::uint64_t count = static_cast<std::uint64_t>(columns) * rows;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
    throw std::runtime_error(std::string(label) +
                             " word count overflows host size");
  return static_cast<std::size_t>(count);
}

Replay load_replay(const fs::path &path, const Plan &plan) {
  // Stage 0 deliberately snapshots the tiny replay before parsing so the
  // authenticated bytes cannot be replaced between hashing and execution.
  const auto bytes = read_small_file(path, 512 * 1024 * 1024);
  if (sha256_bytes(bytes) != plan.replay_sha256)
    throw std::runtime_error("replay content does not match execution plan");
  ByteCursor cursor(bytes);
  cursor.expect_magic("STWOLAB1", 8);
  const auto version = cursor.u32("version");
  Replay replay;
  replay.rows = cursor.u32("rows");
  replay.table_len = cursor.u32("table_len");
  replay.input_columns = cursor.u32("input_columns");
  replay.output_columns = cursor.u32("output_columns");
  replay.lookup_columns = cursor.u32("lookup_columns");
  replay.sub_columns = cursor.u32("sub_columns");
  const auto reserved = cursor.u32("reserved");
  if (version != 2 || replay.rows == 0 || replay.table_len == 0 ||
      reserved != 0)
    throw std::runtime_error("unsupported replay shape/version");
  if (replay.input_columns != plan.input_columns ||
      replay.output_columns != plan.output_columns ||
      replay.lookup_columns != plan.lookup_columns ||
      replay.sub_columns != plan.sub_columns)
    throw std::runtime_error("replay layout does not match execution plan");

  replay.inputs = cursor.words(
      checked_words(replay.input_columns, replay.rows, "inputs"), "inputs");
  replay.table = cursor.words(replay.table_len, "table");
  replay.expected_output = cursor.words(
      checked_words(replay.output_columns, replay.rows, "outputs"), "outputs");
  replay.expected_lookup = cursor.words(
      checked_words(replay.lookup_columns, replay.rows, "lookup"), "lookup");
  replay.expected_sub = cursor.words(
      checked_words(replay.sub_columns, replay.rows, "sub"), "sub");
  replay.mutated_inputs = cursor.words(
      checked_words(replay.input_columns, replay.rows, "mutated inputs"),
      "mutated inputs");
  replay.mutated_expected_output = cursor.words(
      checked_words(replay.output_columns, replay.rows, "mutated outputs"),
      "mutated outputs");
  replay.mutated_expected_lookup = cursor.words(
      checked_words(replay.lookup_columns, replay.rows, "mutated lookup"),
      "mutated lookup");
  replay.mutated_expected_sub = cursor.words(
      checked_words(replay.sub_columns, replay.rows, "mutated sub"),
      "mutated sub");
  cursor.require_end();
  require_reversed_rows(replay.inputs, replay.mutated_inputs,
                        replay.input_columns, replay.rows, "input case");
  require_reversed_rows(replay.expected_output,
                        replay.mutated_expected_output,
                        replay.output_columns, replay.rows, "output case");
  require_reversed_rows(replay.expected_lookup,
                        replay.mutated_expected_lookup,
                        replay.lookup_columns, replay.rows, "lookup case");
  require_reversed_rows(replay.expected_sub, replay.mutated_expected_sub,
                        replay.sub_columns, replay.rows, "sub case");
  if (replay.inputs == replay.mutated_inputs)
    throw std::runtime_error("reviewed proof-varying mutation changed no input");
  return replay;
}

} // namespace gpu_lab
