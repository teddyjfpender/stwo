#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_lab {

using Digest = std::array<std::uint8_t, 32>;

Digest sha256_bytes(const std::vector<std::uint8_t> &bytes);
Digest sha256_file(const std::filesystem::path &path);
std::filesystem::path current_executable_path();
Digest current_executable_sha256();
Digest digest_from_hex(const std::string &value);
std::string digest_hex(const Digest &digest);
bool is_sha256_hex(const std::string &value);
std::vector<std::uint8_t> read_small_file(const std::filesystem::path &path,
                                          std::size_t maximum_size);

class ByteCursor {
public:
  explicit ByteCursor(const std::vector<std::uint8_t> &bytes);

  void expect_magic(const char *magic, std::size_t size);
  std::uint32_t u32(const char *label);
  std::string string(std::size_t size, const char *label);
  Digest digest();
  std::vector<std::uint32_t> words(std::size_t count, const char *label);
  void require_end() const;

private:
  void require(std::size_t size) const;

  const std::vector<std::uint8_t> &bytes_;
  std::size_t position_ = 0;
};

void primitive_self_test();

} // namespace gpu_lab
