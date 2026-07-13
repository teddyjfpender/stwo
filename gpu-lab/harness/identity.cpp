#include "identity.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace gpu_lab {
namespace {

class Sha256 {
public:
  Sha256()
      : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

  void update(const std::uint8_t *data, std::size_t size) {
    for (std::size_t index = 0; index < size; ++index) {
      buffer_[buffer_size_++] = data[index];
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        bit_count_ += 512;
        buffer_size_ = 0;
      }
    }
  }

  Digest finish() {
    std::size_t index = buffer_size_;
    buffer_[index++] = 0x80;
    if (index > 56) {
      std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(index),
                buffer_.end(), 0);
      transform(buffer_.data());
      index = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(index),
              buffer_.begin() + 56, 0);
    bit_count_ += static_cast<std::uint64_t>(buffer_size_) * 8;
    for (int byte = 0; byte < 8; ++byte)
      buffer_[63 - byte] = static_cast<std::uint8_t>(bit_count_ >> (byte * 8));
    transform(buffer_.data());

    Digest digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      digest[word * 4] = static_cast<std::uint8_t>(state_[word] >> 24);
      digest[word * 4 + 1] = static_cast<std::uint8_t>(state_[word] >> 16);
      digest[word * 4 + 2] = static_cast<std::uint8_t>(state_[word] >> 8);
      digest[word * 4 + 3] = static_cast<std::uint8_t>(state_[word]);
    }
    return digest;
  }

private:
  static std::uint32_t rotate_right(std::uint32_t value, unsigned count) {
    return (value >> count) | (value << (32 - count));
  }

  void transform(const std::uint8_t *block) {
    static constexpr std::array<std::uint32_t, 64> constants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                     (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                     (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                     static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t s0 = rotate_right(words[index - 15], 7) ^
                               rotate_right(words[index - 15], 18) ^
                               (words[index - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[index - 2], 17) ^
                               rotate_right(words[index - 2], 19) ^
                               (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sigma1 =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sigma1 + choose + constants[index] + words[index];
      const std::uint32_t sigma0 =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

} // namespace

Digest sha256_bytes(const std::vector<std::uint8_t> &bytes) {
  Sha256 sha;
  sha.update(bytes.data(), bytes.size());
  return sha.finish();
}

Digest sha256_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open file for hashing: " + path.string());
  Sha256 sha;
  std::array<char, 1024 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      sha.update(reinterpret_cast<const std::uint8_t *>(buffer.data()),
                 static_cast<std::size_t>(count));
    }
  }
  if (!input.eof())
    throw std::runtime_error("failed while hashing: " + path.string());
  return sha.finish();
}

std::filesystem::path current_executable_path() {
#if defined(__linux__)
  return std::filesystem::canonical("/proc/self/exe");
#elif defined(__APPLE__)
  std::uint32_t size = 0;
  if (_NSGetExecutablePath(nullptr, &size) != -1 || size == 0)
    throw std::runtime_error("cannot size current executable path");
  std::vector<char> buffer(size);
  if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    throw std::runtime_error("cannot resolve current executable path");
  return std::filesystem::canonical(buffer.data());
#else
#error "current executable identity is required on this platform"
#endif
}

Digest current_executable_sha256() {
  return sha256_file(current_executable_path());
}

std::string digest_hex(const Digest &digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::uint8_t byte : digest)
    output << std::setw(2) << static_cast<unsigned>(byte);
  return output.str();
}

Digest digest_from_hex(const std::string &value) {
  if (!is_sha256_hex(value))
    throw std::runtime_error("digest must be 64 lowercase hex characters");
  auto nibble = [](char character) -> std::uint8_t {
    if (character <= '9')
      return static_cast<std::uint8_t>(character - '0');
    return static_cast<std::uint8_t>(character - 'a' + 10);
  };
  Digest digest{};
  for (std::size_t index = 0; index < digest.size(); ++index)
    digest[index] = static_cast<std::uint8_t>((nibble(value[index * 2]) << 4) |
                                              nibble(value[index * 2 + 1]));
  return digest;
}

bool is_sha256_hex(const std::string &value) {
  if (value.size() != 64)
    return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

std::vector<std::uint8_t> read_small_file(const std::filesystem::path &path,
                                          std::size_t maximum_size) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open file: " + path.string());
  const std::streamoff end = input.tellg();
  if (end < 0 || static_cast<std::uint64_t>(end) > maximum_size)
    throw std::runtime_error("file exceeds size limit: " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!bytes.empty())
    input.read(reinterpret_cast<char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!input)
    throw std::runtime_error("cannot read complete file: " + path.string());
  return bytes;
}

ByteCursor::ByteCursor(const std::vector<std::uint8_t> &bytes)
    : bytes_(bytes) {}

void ByteCursor::expect_magic(const char *magic, std::size_t size) {
  require(size);
  if (std::memcmp(bytes_.data() + position_, magic, size) != 0)
    throw std::runtime_error("invalid execution plan magic");
  position_ += size;
}

std::uint32_t ByteCursor::u32(const char *label) {
  require(4);
  const auto *value = bytes_.data() + position_;
  position_ += 4;
  const std::uint32_t decoded = static_cast<std::uint32_t>(value[0]) |
                                (static_cast<std::uint32_t>(value[1]) << 8) |
                                (static_cast<std::uint32_t>(value[2]) << 16) |
                                (static_cast<std::uint32_t>(value[3]) << 24);
  (void)label;
  return decoded;
}

std::string ByteCursor::string(std::size_t size, const char *label) {
  require(size);
  std::string value(reinterpret_cast<const char *>(bytes_.data() + position_),
                    size);
  position_ += size;
  if (value.find('\0') != std::string::npos)
    throw std::runtime_error(std::string(label) + " contains a null byte");
  return value;
}

Digest ByteCursor::digest() {
  require(32);
  Digest result{};
  std::copy_n(bytes_.data() + position_, result.size(), result.begin());
  position_ += result.size();
  return result;
}

std::vector<std::uint32_t> ByteCursor::words(std::size_t count,
                                             const char *label) {
  if (count > (bytes_.size() - position_) / sizeof(std::uint32_t))
    throw std::runtime_error(std::string("truncated replay payload at ") +
                             label);
  std::vector<std::uint32_t> result(count);
  for (std::size_t index = 0; index < count; ++index)
    result[index] = u32(label);
  return result;
}

void ByteCursor::require_end() const {
  if (position_ != bytes_.size())
    throw std::runtime_error(
        "execution plan contains undeclared trailing bytes");
}

void ByteCursor::require(std::size_t size) const {
  if (size > bytes_.size() - position_)
    throw std::runtime_error("truncated execution plan");
}

void primitive_self_test() {
  const auto digest = [](const char *text) {
    const auto *begin = reinterpret_cast<const std::uint8_t *>(text);
    return digest_hex(sha256_bytes(
        std::vector<std::uint8_t>(begin, begin + std::strlen(text))));
  };
  if (digest("") !=
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ||
      digest("abc") !=
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    throw std::runtime_error("internal SHA256 self-test failed");
  if (digest_hex(digest_from_hex(digest("abc"))) != digest("abc"))
    throw std::runtime_error("internal digest hex self-test failed");
  const auto executable = current_executable_path();
  if (!std::filesystem::is_regular_file(executable) ||
      current_executable_sha256() != sha256_file(executable))
    throw std::runtime_error("current executable identity self-test failed");

  std::vector<std::uint8_t> encoded = {
      'T', 'E',  'S',  'T',  0x78, 0x56, 0x34, 0x12, 'o',
      'k', 0x04, 0x03, 0x02, 0x01, 0xEF, 0xCD, 0xAB, 0x89,
  };
  for (std::uint8_t value = 0; value < 32; ++value)
    encoded.push_back(value);
  ByteCursor cursor(encoded);
  cursor.expect_magic("TEST", 4);
  if (cursor.u32("word") != 0x12345678u || cursor.string(2, "text") != "ok")
    throw std::runtime_error("internal byte-cursor self-test failed");
  if (cursor.words(2, "words") !=
      std::vector<std::uint32_t>{0x01020304u, 0x89ABCDEFu})
    throw std::runtime_error("internal word parser self-test failed");
  const Digest expected = cursor.digest();
  cursor.require_end();
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (expected[index] != index)
      throw std::runtime_error("internal digest parser self-test failed");
}

} // namespace gpu_lab
