#include "fri_round6.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpu_lab {
namespace {

constexpr std::uint32_t kM31Prime = 0x7fffffffu;
constexpr std::size_t kCursorWord = 9;
constexpr std::size_t kStatusWord = 10;
constexpr std::size_t kReservedWord = 11;
constexpr std::size_t kChainLowWord = 12;
constexpr std::size_t kPadding0Word = 14;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
constexpr std::array<std::uint8_t, 12> kAbsorbRoot7Encoding{
    0x1e, 0x00, 0x01, 0x00, // boundary 0x1001e
    0x04, 0x00, 0x00, 0x00, // AbsorbRoot opcode
    0x1c, 0x00, 0x01, 0x00, // input 0x1001c
};
constexpr std::array<std::uint8_t, 12> kDrawAlpha7Encoding{
    0x1f, 0x00, 0x01, 0x00, // boundary 0x1001f
    0x06, 0x00, 0x00, 0x00, // DrawSecureFelt opcode
    0x1d, 0x00, 0x01, 0x00, // output 0x1001d
};

template <std::size_t N>
constexpr std::uint64_t fnv1a_continue(
    std::uint64_t chain, const std::array<std::uint8_t, N> &bytes) {
  for (const std::uint8_t byte : bytes)
    chain = (chain ^ byte) * kFnvPrime;
  return chain;
}

void validate_operation_chains(std::uint64_t chain34, std::uint64_t chain35,
                               std::uint64_t chain36) {
  const std::uint64_t expected35 =
      fnv1a_continue(chain34, kAbsorbRoot7Encoding);
  if (chain35 != expected35)
    throw std::runtime_error("chain35 does not encode FRI tree-7 root absorb");
  const std::uint64_t expected36 =
      fnv1a_continue(expected35, kDrawAlpha7Encoding);
  if (chain36 != expected36)
    throw std::runtime_error("chain36 does not encode FRI alpha-7 draw");
}

bool rejects_operation_chains(std::uint64_t chain34, std::uint64_t chain35,
                              std::uint64_t chain36) {
  try {
    validate_operation_chains(chain34, chain35, chain36);
    return false;
  } catch (const std::runtime_error &) {
    return true;
  }
}

template <std::size_t N>
void read_words(ByteCursor &cursor, std::array<std::uint32_t, N> &out,
                const char *label) {
  const auto words = cursor.words(N, label);
  std::copy(words.begin(), words.end(), out.begin());
}

template <std::size_t N>
void require_m31(const std::array<std::uint32_t, N> &words,
                 const char *label) {
  if (std::any_of(words.begin(), words.end(),
                  [](std::uint32_t word) { return word >= kM31Prime; }))
    throw std::runtime_error(std::string(label) +
                             " contains a non-canonical M31 word");
}

template <std::size_t N>
void require_equal(const std::array<std::uint32_t, N> &left,
                   const std::array<std::uint32_t, N> &right,
                   const char *label) {
  if (left != right)
    throw std::runtime_error(std::string(label) + " fixture invariant failed");
}

std::uint64_t decode_chain(const std::array<std::uint32_t, 16> &state) {
  return static_cast<std::uint64_t>(state[kChainLowWord]) |
         (static_cast<std::uint64_t>(state[kChainLowWord + 1]) << 32U);
}

void validate_state(const std::array<std::uint32_t, 16> &state,
                    std::uint32_t cursor, const char *label) {
  if (state[kCursorWord] != cursor || state[kStatusWord] != 0U ||
      state[kReservedWord] != 0U || state[kPadding0Word] != 0U ||
      state[kPadding0Word + 1] != 0U)
    throw std::runtime_error(std::string(label) +
                             " has an invalid transcript control state");
}

void validate_fixture(const FriRound6Fixture &fixture) {
  require_m31(fixture.entry_pong, "entry_pong");
  require_m31(fixture.inverse_twiddles, "inverse_twiddles");
  require_m31(fixture.alpha6, "alpha6");
  require_m31(fixture.expected_final_ping, "expected_final_ping");
  require_m31(fixture.expected_challenge7, "expected_challenge7");
  require_m31(fixture.expected_retained, "expected_retained");
  require_m31(fixture.expected_draw_output, "expected_draw_output");

  validate_state(fixture.entry_state, 34U, "entry_state");
  validate_state(fixture.expected_boundary_mix, 35U,
                 "expected_boundary_mix");
  validate_state(fixture.expected_boundary_draw, 36U,
                 "expected_boundary_draw");
  validate_state(fixture.expected_exit_state, 36U, "expected_exit_state");
  require_equal(fixture.expected_final_ping, fixture.expected_retained,
                "final_ping/retained");
  require_equal(fixture.expected_root, fixture.expected_mix_input,
                "root/mix_input");
  require_equal(fixture.expected_challenge7, fixture.expected_draw_output,
                "challenge/draw_output");
  require_equal(fixture.expected_exit_state, fixture.expected_boundary_draw,
                "exit_state/boundary_draw");
  validate_operation_chains(fixture.chain34(), fixture.chain35(),
                            fixture.chain36());
}

} // namespace

std::uint64_t FriRound6Fixture::chain34() const {
  return decode_chain(entry_state);
}

std::uint64_t FriRound6Fixture::chain35() const {
  return decode_chain(expected_boundary_mix);
}

std::uint64_t FriRound6Fixture::chain36() const {
  return decode_chain(expected_boundary_draw);
}

FriRound6Fixture load_fri_round6_fixture(const std::filesystem::path &path,
                                         const std::string &expected_sha256) {
  if (!is_sha256_hex(expected_sha256))
    throw std::runtime_error("fixture SHA256 must be lowercase hexadecimal");
  const auto bytes = read_small_file(path, kFriRound6PayloadBytes);
  if (bytes.size() != kFriRound6PayloadBytes)
    throw std::runtime_error("FRI round-6 fixture must be exactly 1,936 bytes");
  const Digest digest = sha256_bytes(bytes);
  if (digest_hex(digest) != expected_sha256)
    throw std::runtime_error("FRI round-6 fixture SHA256 mismatch");

  ByteCursor cursor(bytes);
  FriRound6Fixture fixture;
  read_words(cursor, fixture.entry_pong, "entry_pong");
  read_words(cursor, fixture.inverse_twiddles, "inverse_twiddles");
  read_words(cursor, fixture.alpha6, "alpha6");
  read_words(cursor, fixture.entry_state, "entry_state");
  read_words(cursor, fixture.expected_final_ping, "expected_final_ping");
  read_words(cursor, fixture.expected_root, "expected_root");
  read_words(cursor, fixture.expected_exit_state, "expected_exit_state");
  read_words(cursor, fixture.expected_challenge7, "expected_challenge7");
  read_words(cursor, fixture.expected_retained, "expected_retained");
  read_words(cursor, fixture.expected_leaves, "expected_leaves");
  read_words(cursor, fixture.expected_mix_input, "expected_mix_input");
  read_words(cursor, fixture.expected_draw_output, "expected_draw_output");
  read_words(cursor, fixture.expected_boundary_mix, "expected_boundary_mix");
  read_words(cursor, fixture.expected_boundary_draw, "expected_boundary_draw");
  cursor.require_end();
  fixture.payload_sha256 = digest;
  validate_fixture(fixture);
  return fixture;
}

void validate_fri_round6_pair(const FriRound6Fixture &primary,
                              const FriRound6Fixture &hostile) {
  if (primary.chain34() != hostile.chain34() ||
      primary.chain35() != hostile.chain35() ||
      primary.chain36() != hostile.chain36())
    throw std::runtime_error("primary and hostile fixtures change graph chains");
  if (primary.entry_pong == hostile.entry_pong &&
      primary.alpha6 == hostile.alpha6 &&
      primary.entry_state == hostile.entry_state)
    throw std::runtime_error("hostile fixture does not mutate a semantic input");
  if (primary.expected_root == hostile.expected_root ||
      primary.expected_challenge7 == hostile.expected_challenge7)
    throw std::runtime_error(
        "hostile fixture does not propagate through root and challenge");
}

void fri_round6_layout_self_test() {
  constexpr std::size_t words = 256 + 56 + 4 + 16 + 32 + 8 + 16 + 4 + 32 +
                                16 + 8 + 4 + 16 + 16;
  static_assert(words * sizeof(std::uint32_t) == kFriRound6PayloadBytes);
  static_assert(kFriRound6EntryLog - kFriRound6ExitLog == 3);
  static_assert((1U << kFriRound6ExitLog) >> kFriPackedLeafLog == 2);

  constexpr std::uint64_t chain34 = 0x0123456789abcdefULL;
  constexpr std::uint64_t chain35 =
      fnv1a_continue(chain34, kAbsorbRoot7Encoding);
  constexpr std::uint64_t chain36 =
      fnv1a_continue(chain35, kDrawAlpha7Encoding);
  validate_operation_chains(chain34, chain35, chain36);

  for (unsigned byte = 0; byte < sizeof(std::uint64_t); ++byte) {
    const std::uint64_t bit = 1ULL << (byte * 8U);
    if (!rejects_operation_chains(chain34 ^ bit, chain35, chain36) ||
        !rejects_operation_chains(chain34, chain35 ^ bit, chain36) ||
        !rejects_operation_chains(chain34, chain35, chain36 ^ bit))
      throw std::runtime_error("FRI transcript chain byte mutation survived");
  }

  for (std::size_t byte = 0; byte < kAbsorbRoot7Encoding.size(); ++byte) {
    auto encoding = kAbsorbRoot7Encoding;
    encoding[byte] ^= 1U;
    const std::uint64_t mutated35 = fnv1a_continue(chain34, encoding);
    const std::uint64_t mutated36 =
        fnv1a_continue(mutated35, kDrawAlpha7Encoding);
    if (!rejects_operation_chains(chain34, mutated35, mutated36))
      throw std::runtime_error("FRI root operation byte mutation survived");
  }
  for (std::size_t byte = 0; byte < kDrawAlpha7Encoding.size(); ++byte) {
    auto encoding = kDrawAlpha7Encoding;
    encoding[byte] ^= 1U;
    const std::uint64_t mutated36 = fnv1a_continue(chain35, encoding);
    if (!rejects_operation_chains(chain34, chain35, mutated36))
      throw std::runtime_error("FRI draw operation byte mutation survived");
  }
}

} // namespace gpu_lab
