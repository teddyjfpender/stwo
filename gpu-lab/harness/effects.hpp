#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_lab {

constexpr std::size_t kEffectGuardBytes = 64;
constexpr std::uint8_t kEffectGuardByte = 0xA5;

struct EffectExpectation {
  std::string label;
  std::size_t payload_bytes = 0;
  bool immutable = false;
  std::vector<std::uint8_t> expected;
};

bool validate_effect_snapshot(const std::vector<std::uint8_t> &snapshot,
                              const EffectExpectation &expectation,
                              std::string &error);
void effect_snapshot_self_test();

} // namespace gpu_lab
