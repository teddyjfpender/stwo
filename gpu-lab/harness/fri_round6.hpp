#pragma once

#include "identity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_lab {

constexpr std::size_t kFriRound6PayloadBytes = 1936;
constexpr std::uint32_t kFriRound6EntryLog = 6;
constexpr std::uint32_t kFriRound6ExitLog = 3;
constexpr std::uint32_t kFriPackedLeafLog = 2;

struct FriRound6Fixture {
  std::array<std::uint32_t, 256> entry_pong{};
  std::array<std::uint32_t, 56> inverse_twiddles{};
  std::array<std::uint32_t, 4> alpha6{};
  std::array<std::uint32_t, 16> entry_state{};
  std::array<std::uint32_t, 32> expected_final_ping{};
  std::array<std::uint32_t, 8> expected_root{};
  std::array<std::uint32_t, 16> expected_exit_state{};
  std::array<std::uint32_t, 4> expected_challenge7{};
  std::array<std::uint32_t, 32> expected_retained{};
  std::array<std::uint32_t, 16> expected_leaves{};
  std::array<std::uint32_t, 8> expected_mix_input{};
  std::array<std::uint32_t, 4> expected_draw_output{};
  std::array<std::uint32_t, 16> expected_boundary_mix{};
  std::array<std::uint32_t, 16> expected_boundary_draw{};
  Digest payload_sha256{};

  std::uint64_t chain34() const;
  std::uint64_t chain35() const;
  std::uint64_t chain36() const;
};

FriRound6Fixture load_fri_round6_fixture(const std::filesystem::path &path,
                                         const std::string &expected_sha256);
void validate_fri_round6_pair(const FriRound6Fixture &primary,
                              const FriRound6Fixture &hostile);
void fri_round6_layout_self_test();

} // namespace gpu_lab
