#include "fri_round6.hpp"
#include "fri_round6_runner.hpp"
#include "identity.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gpu_lab {
namespace {

struct FriOptions {
  std::filesystem::path module;
  std::string module_sha256;
  std::filesystem::path module_index;
  std::string module_index_sha256;
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
  std::string harness_sha256;
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
      "module",         "module-sha256",
      "module-index",   "module-index-sha256",
      "build-recipe",   "build-recipe-sha256",
      "primary",        "primary-sha256",
      "primary-index",  "primary-index-sha256",
      "hostile",        "hostile-sha256",
      "hostile-index",  "hostile-index-sha256",
      "harness-sha256", "result",
      "device",         "target-sm",
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
  options.module_index = require("module-index");
  options.module_index_sha256 = require("module-index-sha256");
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
  options.harness_sha256 = require("harness-sha256");
  options.result = require("result");
  options.device = parse_int(require("device"), "device");
  const int target_sm = parse_int(require("target-sm"), "target-sm");
  if (options.device < 0 || target_sm < 50 || target_sm > 999)
    throw std::runtime_error("unsupported device ordinal or target SM");
  for (const std::string *digest :
       {&options.module_sha256, &options.module_index_sha256,
        &options.build_recipe_sha256, &options.primary_sha256,
        &options.primary_index_sha256, &options.hostile_sha256,
        &options.hostile_index_sha256, &options.harness_sha256})
    if (!is_sha256_hex(*digest))
      throw std::runtime_error("all identities must be lowercase SHA256");
  options.target_sm = static_cast<std::uint32_t>(target_sm);
  return options;
}

std::filesystem::path
normalized_destination(const std::filesystem::path &path) {
  std::error_code error;
  const auto absolute = std::filesystem::absolute(path, error);
  if (error || absolute.filename().empty())
    throw std::runtime_error("invalid result path");
  const auto parent =
      std::filesystem::weakly_canonical(absolute.parent_path(), error);
  if (error)
    throw std::runtime_error("cannot resolve result parent");
  return parent / absolute.filename();
}

void reject_result_input_alias(const FriOptions &options) {
  const auto result = normalized_destination(options.result);
  auto temporary = result;
  temporary += ".tmp";
  const std::vector<std::filesystem::path> protected_paths = {
      options.module,        options.module_index,      options.build_recipe,
      options.primary,       options.primary_index,     options.hostile,
      options.hostile_index, current_executable_path(),
  };
  for (const auto &path : protected_paths) {
    const auto protected_path = std::filesystem::canonical(path);
    if (result == protected_path || temporary == protected_path)
      throw std::runtime_error("result path aliases a sealed launch input");
  }
}

std::string json_escape(const std::string &text) {
  std::ostringstream output;
  for (unsigned char character : text) {
    if (character == '\\' || character == '"')
      output << '\\' << character;
    else if (character == '\n')
      output << "\\n";
    else if (character < 0x20)
      throw std::runtime_error(
          "result text contains an unsupported control byte");
    else
      output << character;
  }
  return output.str();
}

void append_validation(std::ostringstream &output, const char *name,
                       const FriValidation &validation, bool comma = true) {
  output << "    \"" << name
         << "\": {\"passed\": " << (validation.passed ? "true" : "false")
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
         << "  \"standalone_admissible\": false,\n"
         << "  \"performance_admissible\": false,\n"
         << "  \"segment\": \"GraphSegment::FriLayer(7)/FriRound(6)\",\n"
         << "  \"module_content_sha256\": \"" << options.module_sha256
         << "\",\n"
         << "  \"module_index_sha256\": \"" << options.module_index_sha256
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
         << "  \"harness_executable_sha256\": \"" << options.harness_sha256
         << "\",\n"
         << "  \"device\": {\"name\": \"" << json_escape(runner.device_name())
         << "\", \"uuid\": \"" << runner.device_uuid()
         << "\", \"ordinal\": " << options.device
         << ", \"target_sm\": " << options.target_sm
         << ", \"driver_version\": " << runner.driver_version() << "},\n"
         << "  \"graph_contract\": {\"kernels\": "
         << runner.graph_kernel_nodes()
         << ", \"device_copies\": " << runner.graph_copy_nodes()
         << ", "
            "\"entry_log\": 6, \"exit_log\": 3, \"packed_leaf_log\": 2},\n"
         << "  \"correctness\": {\n";
  append_validation(output, "primary_eager", primary_eager);
  append_validation(output, "primary_graph", primary_graph);
  append_validation(output, "hostile_eager", hostile_eager);
  append_validation(output, "hostile_graph", hostile_graph);
  output << "    \"stale_cursor_status_order\": {\"passed\": "
         << (stale_cursor_rejected ? "true" : "false") << ", \"error\": \""
         << json_escape(stale_cursor_error) << "\"},\n";
  append_validation(output, "reset_replay", reset_replay, false);
  output << "  }\n}\n";
  return output.str();
}

using ResultPublishHook = void (*)(const std::filesystem::path &);

void write_atomic(const std::filesystem::path &destination,
                  const std::string &content,
                  ResultPublishHook before_publish = nullptr) {
  const auto parent = destination.parent_path();
  std::string pattern =
      (parent / ("." + destination.filename().string() + ".XXXXXX")).string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  int descriptor = ::mkstemp(name.data());
  if (descriptor < 0)
    throw std::system_error(errno, std::generic_category(),
                            "cannot create unique FRI result temporary");
  const std::filesystem::path temporary(name.data());
  struct stat identity{};
  bool identity_known = false;
  bool temporary_live = true;
  try {
    // Restrict the inode before writing. The already-open descriptor remains
    // writable, while another process cannot open this result inode for write.
    if (::fchmod(descriptor, S_IRUSR) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot protect FRI result temporary");
    const char *cursor = content.data();
    std::size_t remaining = content.size();
    while (remaining != 0) {
      const ssize_t written = ::write(descriptor, cursor, remaining);
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        throw std::system_error(errno, std::generic_category(),
                                "cannot write complete FRI result");
      cursor += written;
      remaining -= static_cast<std::size_t>(written);
    }
    if (::fsync(descriptor) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot sync FRI result temporary");
    if (::fstat(descriptor, &identity) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot identify FRI result temporary");
    identity_known = true;
    if (!S_ISREG(identity.st_mode) || (identity.st_mode & 0777) != S_IRUSR ||
        identity.st_size != static_cast<off_t>(content.size()) ||
        identity.st_nlink != 1)
      throw std::runtime_error("FRI result temporary identity is unsafe");
    const auto verify_content = [&]() {
      std::vector<char> actual(content.size());
      std::size_t offset = 0;
      while (offset != actual.size()) {
        const ssize_t count =
            ::pread(descriptor, actual.data() + offset, actual.size() - offset,
                    static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR)
          continue;
        if (count <= 0)
          throw std::system_error(errno, std::generic_category(),
                                  "cannot read back complete FRI result");
        offset += static_cast<std::size_t>(count);
      }
      char extra = 0;
      if (::pread(descriptor, &extra, 1, static_cast<off_t>(actual.size())) !=
              0 ||
          !std::equal(actual.begin(), actual.end(), content.begin()))
        throw std::runtime_error(
            "FRI result readback differs from generated bytes");
    };
    verify_content();
    if (before_publish != nullptr)
      before_publish(temporary);
    if (::link(temporary.c_str(), destination.c_str()) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot atomically install unique FRI result");
    struct stat installed{};
    if (::lstat(destination.c_str(), &installed) != 0 ||
        installed.st_dev != identity.st_dev ||
        installed.st_ino != identity.st_ino ||
        installed.st_size != identity.st_size || !S_ISREG(installed.st_mode) ||
        (installed.st_mode & 0777) != S_IRUSR)
      throw std::runtime_error(
          "published FRI result inode differs from written inode");
    struct stat named_temporary{};
    if (::lstat(temporary.c_str(), &named_temporary) != 0 ||
        named_temporary.st_dev != identity.st_dev ||
        named_temporary.st_ino != identity.st_ino)
      throw std::runtime_error("FRI result temporary was substituted");
    if (::unlink(temporary.c_str()) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot remove installed FRI result temporary");
    temporary_live = false;
    if (::fstat(descriptor, &installed) != 0 || installed.st_nlink != 1 ||
        installed.st_dev != identity.st_dev ||
        installed.st_ino != identity.st_ino)
      throw std::runtime_error("published FRI result link count differs");
    int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
    directory_flags |= O_DIRECTORY;
#endif
    const int directory = ::open(parent.c_str(), directory_flags);
    if (directory < 0 || ::fsync(directory) != 0) {
      const int failure = errno;
      if (directory >= 0)
        ::close(directory);
      throw std::system_error(failure, std::generic_category(),
                              "cannot sync FRI result directory");
    }
    if (::close(directory) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot close FRI result directory");
    if (::lstat(destination.c_str(), &installed) != 0 ||
        installed.st_dev != identity.st_dev ||
        installed.st_ino != identity.st_ino ||
        installed.st_size != identity.st_size || !S_ISREG(installed.st_mode) ||
        (installed.st_mode & 0777) != S_IRUSR)
      throw std::runtime_error("FRI result changed after directory sync");
    verify_content();
    const int completed_descriptor = descriptor;
    descriptor = -1;
    if (::close(completed_descriptor) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "cannot close FRI result temporary");
  } catch (...) {
    if (descriptor >= 0)
      ::close(descriptor);
    if (temporary_live) {
      struct stat candidate{};
      if (!identity_known || (::lstat(temporary.c_str(), &candidate) == 0 &&
                              candidate.st_dev == identity.st_dev &&
                              candidate.st_ino == identity.st_ino))
        ::unlink(temporary.c_str());
    }
    // A published result is immutable and hash-checkable even if a later
    // durability check fails. Never delete it here; the caller sees failure
    // and the admission orchestrator rejects the run.
    throw;
  }
}

void substitute_result_temporary(const std::filesystem::path &temporary) {
  if (::unlink(temporary.c_str()) != 0 ||
      ::symlink("/dev/null", temporary.c_str()) != 0)
    throw std::system_error(errno, std::generic_category(),
                            "cannot substitute FRI result temporary");
}

void result_io_self_test() {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "stwo-fri-result.XXXXXX")
          .string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const char *created = ::mkdtemp(name.data());
  if (created == nullptr)
    throw std::system_error(errno, std::generic_category(),
                            "cannot create FRI result self-test directory");
  const std::filesystem::path directory(created);
  const auto result = directory / "result.json";
  try {
    const std::string first = "{\"passed\":true}\n";
    write_atomic(result, first);
    const auto bytes = read_small_file(result, 1024);
    if (std::string(bytes.begin(), bytes.end()) != first)
      throw std::runtime_error("atomic FRI result content changed");
    bool duplicate_rejected = false;
    try {
      write_atomic(result, "replacement");
    } catch (const std::system_error &) {
      duplicate_rejected = true;
    }
    if (!duplicate_rejected || read_small_file(result, 1024) != bytes)
      throw std::runtime_error("atomic FRI result allowed a replacement race");
    const auto mode = std::filesystem::status(result).permissions();
    if ((mode & (std::filesystem::perms::owner_write |
                 std::filesystem::perms::group_write |
                 std::filesystem::perms::others_write)) !=
        std::filesystem::perms::none)
      throw std::runtime_error("published FRI result remained writable");
    const auto occupied = directory / "occupied";
    std::filesystem::create_directory(occupied);
    bool directory_rejected = false;
    try {
      write_atomic(occupied, "replacement");
    } catch (const std::exception &) {
      directory_rejected = true;
    }
    if (!directory_rejected || !std::filesystem::is_directory(occupied))
      throw std::runtime_error(
          "FRI result publication replaced an occupied path");
    bool substitution_rejected = false;
    try {
      write_atomic(directory / "substituted", first,
                   substitute_result_temporary);
    } catch (const std::exception &) {
      substitution_rejected = true;
    }
    if (!substitution_rejected)
      throw std::runtime_error("FRI result accepted a substituted temporary");
  } catch (...) {
    std::filesystem::remove_all(directory);
    throw;
  }
  std::filesystem::remove_all(directory);
}

void verify_bound_file(const std::filesystem::path &path,
                       const std::string &expected_sha256, const char *label) {
  if (digest_hex(sha256_file(path)) != expected_sha256)
    throw std::runtime_error(std::string(label) + " SHA256 mismatch");
}

void verify_launch_inputs(const FriOptions &options) {
  verify_bound_file(options.module, options.module_sha256, "module");
  verify_bound_file(options.module_index, options.module_index_sha256,
                    "module index");
  verify_bound_file(options.build_recipe, options.build_recipe_sha256,
                    "build recipe");
  verify_bound_file(options.primary, options.primary_sha256, "primary fixture");
  verify_bound_file(options.primary_index, options.primary_index_sha256,
                    "primary fixture index");
  verify_bound_file(options.hostile, options.hostile_sha256, "hostile fixture");
  verify_bound_file(options.hostile_index, options.hostile_index_sha256,
                    "hostile fixture index");
  if (digest_hex(current_executable_sha256()) != options.harness_sha256)
    throw std::runtime_error("harness executable SHA256 mismatch");
}

} // namespace
} // namespace gpu_lab

int main(int argc, char **argv) {
  using namespace gpu_lab;
  try {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
      primitive_self_test();
      fri_round6_layout_self_test();
      result_io_self_test();
      std::cout << "stwo-gpu-lab FRI round-6 self-test: PASS\n";
      return 0;
    }
    FriOptions options = parse_options(argc, argv);
    options.result = normalized_destination(options.result);
    reject_result_input_alias(options);
    verify_launch_inputs(options);
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
    verify_launch_inputs(options);
    const std::string result = make_result(
        options, runner, primary_eager, primary_graph, hostile_eager,
        hostile_graph, stale_cursor, stale_cursor_error, reset_replay);
    write_atomic(options.result, result);
    verify_launch_inputs(options);
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
