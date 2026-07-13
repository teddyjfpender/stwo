#include "effects.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpu_lab {
namespace {

bool fail(std::string &error, const std::string &message) {
  if (error.empty())
    error = message;
  return false;
}

} // namespace

bool validate_effect_snapshot(const std::vector<std::uint8_t> &snapshot,
                              const EffectExpectation &expectation,
                              std::string &error) {
  const std::size_t expected_size =
      expectation.payload_bytes + 2 * kEffectGuardBytes;
  if (snapshot.size() != expected_size)
    return fail(error, expectation.label + ": effect snapshot size mismatch");
  const auto payload = snapshot.begin() + kEffectGuardBytes;
  const auto suffix =
      payload + static_cast<std::ptrdiff_t>(expectation.payload_bytes);
  if (!std::all_of(snapshot.begin(), payload, [](std::uint8_t value) {
        return value == kEffectGuardByte;
      }))
    return fail(error, expectation.label + ": prefix guard changed");
  if (!std::all_of(suffix, snapshot.end(), [](std::uint8_t value) {
        return value == kEffectGuardByte;
      }))
    return fail(error, expectation.label + ": suffix guard changed");
  if (expectation.immutable &&
      (expectation.expected.size() != expectation.payload_bytes ||
       !std::equal(payload, suffix, expectation.expected.begin())))
    return fail(error, expectation.label + ": read-only payload changed");
  return true;
}

void effect_snapshot_self_test() {
  EffectExpectation expectation{"read-only", 4, true, {1, 2, 3, 4}};
  std::vector<std::uint8_t> snapshot(
      expectation.payload_bytes + 2 * kEffectGuardBytes, kEffectGuardByte);
  std::copy(expectation.expected.begin(), expectation.expected.end(),
            snapshot.begin() + kEffectGuardBytes);
  std::string error;
  if (!validate_effect_snapshot(snapshot, expectation, error))
    throw std::runtime_error("valid effect snapshot was rejected: " + error);
  for (std::size_t offset :
       {std::size_t{0}, kEffectGuardBytes, snapshot.size() - 1}) {
    auto mutated = snapshot;
    mutated[offset] ^= 1;
    error.clear();
    if (validate_effect_snapshot(mutated, expectation, error))
      throw std::runtime_error("mutated effect snapshot was accepted");
  }
  expectation.immutable = false;
  snapshot[kEffectGuardBytes] ^= 1;
  error.clear();
  if (!validate_effect_snapshot(snapshot, expectation, error))
    throw std::runtime_error("declared writable payload was rejected: " +
                             error);
}

} // namespace gpu_lab
