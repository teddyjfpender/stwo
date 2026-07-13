#include "fri_round6_runner.hpp"

#include "cuda_check.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace gpu_lab {
namespace {

constexpr std::size_t kGuardBytes = 32;
constexpr std::uint8_t kGuardByte = 0xa5;
constexpr std::uint32_t kPoison = 0xdeadbeefu;

std::vector<CUdeviceptr> coordinate_pointers(CUdeviceptr base,
                                             std::size_t stride_words) {
  std::vector<CUdeviceptr> result;
  for (std::size_t coordinate = 0; coordinate < 4; ++coordinate)
    result.push_back(base + coordinate * stride_words * sizeof(std::uint32_t));
  return result;
}

template <std::size_t N>
std::vector<std::uint32_t> to_vector(const std::array<std::uint32_t, N> &words) {
  return {words.begin(), words.end()};
}

} // namespace

#define CU_CHECK(call) GPU_LAB_CU_CHECK(call)

FriRound6Runner::FriRound6Runner(
    int device_ordinal, std::uint32_t target_sm,
    const std::filesystem::path &module_path, const std::string &module_sha256,
    const FriRound6Fixture &binding_fixture)
    : device_ordinal_(device_ordinal), target_sm_(target_sm),
      chain34_(binding_fixture.chain34()), chain35_(binding_fixture.chain35()),
      chain36_(binding_fixture.chain36()) {
  try {
    if (!is_sha256_hex(module_sha256))
      throw std::runtime_error("module SHA256 must be lowercase hexadecimal");
    module_image_ = read_small_file(module_path, 64 * 1024 * 1024);
    if (module_image_.empty() ||
        digest_hex(sha256_bytes(module_image_)) != module_sha256)
      throw std::runtime_error("FRI round-6 module SHA256 mismatch");
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device_, device_ordinal_));
    int major = 0;
    int minor = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    CU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
    if (major * 10 + minor != static_cast<int>(target_sm_))
      throw std::runtime_error("actual device SM does not match module target");
    CU_CHECK(cuCtxGetCurrent(&previous_context_));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
    primary_retained_ = true;
    CU_CHECK(cuCtxSetCurrent(context_));
    CU_CHECK(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
    CU_CHECK(cuModuleLoadData(&module_, module_image_.data()));
    CU_CHECK(cuModuleGetFunction(
        &fold_, module_, "stwo_gpu_lab_fold_line_device_alpha"));
    CU_CHECK(cuModuleGetFunction(
        &leaf_, module_, "stwo_gpu_lab_blake2s_fri_leaf"));
    CU_CHECK(cuModuleGetFunction(&parent_, module_,
                                 "stwo_gpu_lab_blake2s_layer"));
    CU_CHECK(cuModuleGetFunction(
        &mix_, module_, "stwo_gpu_lab_blake2s_transcript_mix_words"));
    CU_CHECK(cuModuleGetFunction(
        &draw_, module_, "stwo_gpu_lab_blake2s_transcript_draw_secure"));
    allocate_buffers();
  } catch (...) {
    cleanup();
    throw;
  }
}

FriRound6Runner::~FriRound6Runner() { cleanup(); }

void FriRound6Runner::cleanup() noexcept {
  if (context_)
    cuCtxSetCurrent(context_);
  if (stream_)
    cuStreamSynchronize(stream_);
  if (graph_executable_)
    cuGraphExecDestroy(graph_executable_);
  if (graph_)
    cuGraphDestroy(graph_);
  if (stream_)
    cuStreamDestroy(stream_);
  for (auto region = regions_.rbegin(); region != regions_.rend(); ++region)
    if (region->base)
      cuMemFree(region->base);
  regions_.clear();
  if (module_)
    cuModuleUnload(module_);
  if (context_)
    cuCtxSetCurrent(previous_context_);
  context_ = nullptr;
  if (primary_retained_)
    cuDevicePrimaryCtxRelease(device_);
  primary_retained_ = false;
}

CUdeviceptr FriRound6Runner::allocate(std::size_t bytes, const char *label) {
  if (bytes == 0)
    throw std::runtime_error("zero-sized FRI device allocation");
  CUdeviceptr base = 0;
  CU_CHECK(cuMemAlloc(&base, bytes + 2 * kGuardBytes));
  CU_CHECK(cuMemsetD8(base, kGuardByte, bytes + 2 * kGuardBytes));
  const CUdeviceptr data = base + kGuardBytes;
  regions_.push_back({base, data, bytes, label});
  return data;
}

CUdeviceptr FriRound6Runner::upload_pointer_table(
    const std::vector<CUdeviceptr> &pointers, const char *label) {
  static_assert(sizeof(CUdeviceptr) == 8,
                "FRI replay requires 64-bit CUDA device pointers");
  const std::size_t bytes = pointers.size() * sizeof(CUdeviceptr);
  const CUdeviceptr table = allocate(bytes, label);
  CU_CHECK(cuMemcpyHtoD(table, pointers.data(), bytes));
  return table;
}

void FriRound6Runner::allocate_buffers() {
  pong_ = allocate(4 * 64 * 4, "pong");
  ping_ = allocate(4 * 32 * 4, "ping");
  retained_ = allocate(4 * 8 * 4, "retained");
  twiddles_ = allocate(56 * 4, "inverse_twiddles");
  alpha_ = allocate(4 * 4, "alpha6");
  pong_table_ =
      upload_pointer_table(coordinate_pointers(pong_, 64), "pong_table");
  ping_table_ =
      upload_pointer_table(coordinate_pointers(ping_, 32), "ping_table");
  retained_table_ = upload_pointer_table(coordinate_pointers(retained_, 8),
                                         "retained_table");
  leaves_ = allocate(2 * 8 * 4, "tree7_leaves");
  root_ = allocate(8 * 4, "tree7_root");
  root_input_ = allocate(8 * 4, "transcript_root_input");
  state_ = allocate(16 * 4, "transcript_state");
  mix_input_ = allocate(8 * 4, "mix_input_snapshot");
  boundary_mix_ = allocate(16 * 4, "boundary_after_mix");
  challenge_ = allocate(4 * 4, "alpha7");
  draw_output_ = allocate(4 * 4, "draw_output_snapshot");
  boundary_draw_ = allocate(16 * 4, "boundary_after_draw");
  challenge_copy_ = allocate(4 * 4, "round7_challenge_slot");
}

void FriRound6Runner::reset(const FriRound6Fixture &fixture) {
  CU_CHECK(cuMemcpyHtoDAsync(pong_, fixture.entry_pong.data(), 4 * 64 * 4,
                             stream_));
  CU_CHECK(cuMemcpyHtoDAsync(twiddles_, fixture.inverse_twiddles.data(),
                             56 * 4, stream_));
  CU_CHECK(cuMemcpyHtoDAsync(alpha_, fixture.alpha6.data(), 4 * 4, stream_));
  CU_CHECK(cuMemcpyHtoDAsync(state_, fixture.entry_state.data(), 16 * 4,
                             stream_));
  for (const auto buffer : {ping_, retained_, leaves_, root_, root_input_,
                            mix_input_, boundary_mix_, challenge_, draw_output_,
                            boundary_draw_, challenge_copy_}) {
    const auto region = std::find_if(
        regions_.begin(), regions_.end(),
        [buffer](const Region &candidate) { return candidate.data == buffer; });
    if (region == regions_.end() || region->bytes % 4 != 0)
      throw std::runtime_error("unregistered FRI output buffer");
    CU_CHECK(cuMemsetD32Async(buffer, kPoison, region->bytes / 4, stream_));
  }
}

void FriRound6Runner::launch_segment() {
  const auto launch_fold = [&](std::uint32_t offset, std::uint32_t n,
                               std::uint32_t squarings, CUdeviceptr input,
                               CUdeviceptr output) {
    void *arguments[] = {&twiddles_, &offset, &n, &alpha_, &squarings, &input,
                         &output};
    CU_CHECK(cuLaunchKernel(fold_, 1, 1, 1, 256, 1, 1, 0, stream_, arguments,
                            nullptr));
  };
  launch_fold(0, 64, 0, pong_table_, ping_table_);
  launch_fold(32, 32, 1, ping_table_, pong_table_);
  launch_fold(48, 16, 2, pong_table_, ping_table_);
  for (std::size_t coordinate = 0; coordinate < 4; ++coordinate)
    CU_CHECK(cuMemcpyDtoDAsync(retained_ + coordinate * 8 * 4,
                               ping_ + coordinate * 32 * 4, 8 * 4, stream_));

  std::uint32_t evaluation_size = 8;
  std::uint32_t packed_leaf_log = kFriPackedLeafLog;
  void *leaf_arguments[] = {&evaluation_size, &retained_table_,
                            &packed_leaf_log, &leaves_};
  CU_CHECK(cuLaunchKernel(leaf_, 1, 1, 1, 256, 1, 1, 0, stream_,
                          leaf_arguments, nullptr));
  std::uint32_t parent_size = 1;
  std::uint32_t columns = 0;
  CUdeviceptr no_columns = 0;
  void *parent_arguments[] = {&parent_size, &columns, &no_columns, &leaves_,
                              &root_};
  CU_CHECK(cuLaunchKernel(parent_, 1, 1, 1, 256, 1, 1, 0, stream_,
                          parent_arguments, nullptr));
  CU_CHECK(cuMemcpyDtoDAsync(root_input_, root_, 8 * 4, stream_));

  std::uint32_t mix_step = 34;
  std::uint32_t root_words = 8;
  std::uint32_t validate_m31 = 0;
  void *mix_arguments[] = {&state_,      &mix_step,    &chain34_,
                           &chain35_,    &root_input_, &root_words,
                           &validate_m31, &mix_input_,  &boundary_mix_};
  CU_CHECK(cuLaunchKernel(mix_, 1, 1, 1, 1, 1, 1, 0, stream_, mix_arguments,
                          nullptr));
  std::uint32_t draw_step = 35;
  std::uint32_t n_felts = 1;
  std::uint32_t max_rejections = 64;
  void *draw_arguments[] = {&state_,          &draw_step,
                            &chain35_,        &chain36_,
                            &n_felts,         &max_rejections,
                            &challenge_,      &draw_output_,
                            &boundary_draw_};
  CU_CHECK(cuLaunchKernel(draw_, 1, 1, 1, 1, 1, 1, 0, stream_, draw_arguments,
                          nullptr));
  CU_CHECK(cuMemcpyDtoDAsync(challenge_copy_, challenge_, 4 * 4, stream_));
}

void FriRound6Runner::ensure_graph() {
  if (graph_executable_)
    return;
  CUgraph captured = nullptr;
  bool capture_active = false;
  CU_CHECK(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  capture_active = true;
  try {
    launch_segment();
    CU_CHECK(cuStreamEndCapture(stream_, &captured));
    capture_active = false;
#if CUDA_VERSION >= 12000
    CU_CHECK(cuGraphInstantiate(&graph_executable_, captured, 0));
#else
    CU_CHECK(cuGraphInstantiateWithFlags(&graph_executable_, captured, 0));
#endif
    graph_ = captured;
    captured = nullptr;
    validate_graph_contract();
  } catch (...) {
    if (graph_executable_) {
      cuGraphExecDestroy(graph_executable_);
      graph_executable_ = nullptr;
    }
    if (graph_) {
      cuGraphDestroy(graph_);
      graph_ = nullptr;
    }
    if (capture_active) {
      CUgraph abandoned = nullptr;
      cuStreamEndCapture(stream_, &abandoned);
      if (abandoned)
        cuGraphDestroy(abandoned);
    }
    if (captured)
      cuGraphDestroy(captured);
    graph_kernel_nodes_ = 0;
    graph_copy_nodes_ = 0;
    throw;
  }
}

void FriRound6Runner::validate_graph_contract() {
  std::size_t count = 0;
  CU_CHECK(cuGraphGetNodes(graph_, nullptr, &count));
  std::vector<CUgraphNode> nodes(count);
  CU_CHECK(cuGraphGetNodes(graph_, nodes.data(), &count));
  graph_kernel_nodes_ = 0;
  graph_copy_nodes_ = 0;
  for (CUgraphNode node : nodes) {
    CUgraphNodeType type{};
    CU_CHECK(cuGraphNodeGetType(node, &type));
    if (type == CU_GRAPH_NODE_TYPE_KERNEL)
      ++graph_kernel_nodes_;
    else if (type == CU_GRAPH_NODE_TYPE_MEMCPY)
      ++graph_copy_nodes_;
    else
      throw std::runtime_error("FRI graph contains an unreviewed node type");
  }
  if (count != 13 || graph_kernel_nodes_ != 7 || graph_copy_nodes_ != 6)
    throw std::runtime_error("FRI graph topology differs from 7 kernels + 6 copies");
}

FriValidation FriRound6Runner::run_eager(const FriRound6Fixture &fixture) {
  reset(fixture);
  CU_CHECK(cuStreamSynchronize(stream_));
  launch_segment();
  CU_CHECK(cuStreamSynchronize(stream_));
  return validate(fixture);
}

FriValidation FriRound6Runner::run_graph(const FriRound6Fixture &fixture) {
  ensure_graph();
  reset(fixture);
  CU_CHECK(cuStreamSynchronize(stream_));
  CU_CHECK(cuGraphLaunch(graph_executable_, stream_));
  CU_CHECK(cuStreamSynchronize(stream_));
  return validate(fixture);
}

FriValidation FriRound6Runner::validate(const FriRound6Fixture &fixture) {
  FriValidation result{true, 0, {}};
  auto compare = [&](CUdeviceptr pointer,
                     const std::vector<std::uint32_t> &expected,
                     const char *label) {
    std::vector<std::uint32_t> got(expected.size());
    CU_CHECK(cuMemcpyDtoH(got.data(), pointer, got.size() * 4));
    result.checked_words += expected.size();
    const auto mismatch =
        std::mismatch(got.begin(), got.end(), expected.begin(), expected.end());
    if (mismatch.first == got.end())
      return;
    result.passed = false;
    if (result.error.empty()) {
      const auto index = static_cast<std::size_t>(mismatch.first - got.begin());
      result.error = std::string(label) + " word " + std::to_string(index) +
                     " differs from the CPU oracle";
    }
  };

  std::vector<std::uint32_t> ping(4 * 32);
  CU_CHECK(cuMemcpyDtoH(ping.data(), ping_, ping.size() * 4));
  std::vector<std::uint32_t> final_ping;
  for (std::size_t coordinate = 0; coordinate < 4; ++coordinate)
    final_ping.insert(final_ping.end(), ping.begin() + coordinate * 32,
                      ping.begin() + coordinate * 32 + 8);
  result.checked_words += final_ping.size();
  if (final_ping != to_vector(fixture.expected_final_ping)) {
    result.passed = false;
    if (result.error.empty())
      result.error = "final_ping differs from the CPU oracle";
  }
  compare(root_, to_vector(fixture.expected_root), "root");
  compare(state_, to_vector(fixture.expected_exit_state), "exit_state");
  compare(challenge_, to_vector(fixture.expected_challenge7), "challenge7");
  compare(retained_, to_vector(fixture.expected_retained), "retained");
  compare(leaves_, to_vector(fixture.expected_leaves), "leaves");
  compare(mix_input_, to_vector(fixture.expected_mix_input), "mix_input");
  compare(draw_output_, to_vector(fixture.expected_draw_output), "draw_output");
  compare(boundary_mix_, to_vector(fixture.expected_boundary_mix),
          "boundary_mix");
  compare(boundary_draw_, to_vector(fixture.expected_boundary_draw),
          "boundary_draw");
  compare(root_input_, to_vector(fixture.expected_root), "root_input_copy");
  compare(challenge_copy_, to_vector(fixture.expected_challenge7),
          "challenge_copy");
  compare(twiddles_, to_vector(fixture.inverse_twiddles), "inverse_twiddles");
  compare(alpha_, to_vector(fixture.alpha6), "alpha6");

  std::vector<std::uint32_t> pong(4 * 64);
  CU_CHECK(cuMemcpyDtoH(pong.data(), pong_, pong.size() * 4));
  for (std::size_t coordinate = 0; coordinate < 4; ++coordinate) {
    const auto got = pong.begin() + coordinate * 64 + 16;
    const auto expected = fixture.entry_pong.begin() + coordinate * 64 + 16;
    result.checked_words += 48;
    if (!std::equal(got, got + 48, expected)) {
      result.passed = false;
      if (result.error.empty())
        result.error = "pong dead tail was modified";
    }
  }
  if (!validate_guards(result.error))
    result.passed = false;
  return result;
}

bool FriRound6Runner::validate_guards(std::string &error) {
  std::array<std::uint8_t, kGuardBytes> guard{};
  for (const Region &region : regions_) {
    for (CUdeviceptr address : {region.base, region.data + region.bytes}) {
      CU_CHECK(cuMemcpyDtoH(guard.data(), address, guard.size()));
      if (!std::all_of(guard.begin(), guard.end(),
                       [](std::uint8_t value) { return value == kGuardByte; })) {
        if (error.empty())
          error = region.label + " guard was modified";
        return false;
      }
    }
  }
  for (const auto &[table, expected, label] :
       std::array{
           std::tuple{pong_table_, coordinate_pointers(pong_, 64),
                      "pong pointer table"},
           std::tuple{ping_table_, coordinate_pointers(ping_, 32),
                      "ping pointer table"},
           std::tuple{retained_table_, coordinate_pointers(retained_, 8),
                      "retained pointer table"},
       }) {
    std::vector<CUdeviceptr> got(expected.size());
    CU_CHECK(cuMemcpyDtoH(got.data(), table,
                          got.size() * sizeof(CUdeviceptr)));
    if (got != expected) {
      if (error.empty())
        error = std::string(label) + " was modified";
      return false;
    }
  }
  return true;
}

bool FriRound6Runner::stale_cursor_sets_order_status(
    const FriRound6Fixture &fixture, std::string &error) {
  ensure_graph();
  reset(fixture);
  const std::uint32_t stale_cursor = 33;
  CU_CHECK(cuMemcpyHtoDAsync(state_ + 9 * 4, &stale_cursor, 4, stream_));
  CU_CHECK(cuStreamSynchronize(stream_));
  CU_CHECK(cuGraphLaunch(graph_executable_, stream_));
  CU_CHECK(cuStreamSynchronize(stream_));
  std::array<std::uint32_t, 16> state{};
  CU_CHECK(cuMemcpyDtoH(state.data(), state_, state.size() * 4));
  auto expected_state = fixture.entry_state;
  expected_state[9] = stale_cursor;
  expected_state[10] = 1U;
  if (state != expected_state) {
    error = "stale cursor changed transcript state beyond STATUS_ORDER";
    return false;
  }
  const std::array<std::uint32_t, 4> poison4{kPoison, kPoison, kPoison,
                                              kPoison};
  const std::array<std::uint32_t, 8> poison8{kPoison, kPoison, kPoison,
                                              kPoison, kPoison, kPoison,
                                              kPoison, kPoison};
  std::array<std::uint32_t, 8> mix_input{};
  std::array<std::uint32_t, 4> challenge{};
  std::array<std::uint32_t, 4> draw_output{};
  std::array<std::uint32_t, 4> challenge_copy{};
  std::array<std::uint32_t, 16> boundary_mix{};
  std::array<std::uint32_t, 16> boundary_draw{};
  CU_CHECK(cuMemcpyDtoH(mix_input.data(), mix_input_, mix_input.size() * 4));
  CU_CHECK(cuMemcpyDtoH(challenge.data(), challenge_, challenge.size() * 4));
  CU_CHECK(cuMemcpyDtoH(draw_output.data(), draw_output_, draw_output.size() * 4));
  CU_CHECK(cuMemcpyDtoH(challenge_copy.data(), challenge_copy_,
                        challenge_copy.size() * 4));
  CU_CHECK(cuMemcpyDtoH(boundary_mix.data(), boundary_mix_,
                        boundary_mix.size() * 4));
  CU_CHECK(cuMemcpyDtoH(boundary_draw.data(), boundary_draw_,
                        boundary_draw.size() * 4));
  if (mix_input != poison8 || challenge != poison4 ||
      draw_output != poison4 || challenge_copy != poison4 ||
      boundary_mix != expected_state || boundary_draw != expected_state) {
    error = "stale cursor transcript no-write/snapshot contract failed";
    return false;
  }
  return validate_guards(error);
}

std::string FriRound6Runner::device_name() const {
  char name[256]{};
  CU_CHECK(cuDeviceGetName(name, sizeof(name), device_));
  return name;
}

std::string FriRound6Runner::device_uuid() const {
  CUuuid uuid{};
  CU_CHECK(cuDeviceGetUuid(&uuid, device_));
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (char byte : uuid.bytes)
    output << std::setw(2)
           << static_cast<unsigned>(static_cast<unsigned char>(byte));
  return output.str();
}

int FriRound6Runner::driver_version() const {
  int version = 0;
  CU_CHECK(cuDriverGetVersion(&version));
  return version;
}

std::size_t FriRound6Runner::graph_kernel_nodes() const {
  return graph_kernel_nodes_;
}

std::size_t FriRound6Runner::graph_copy_nodes() const {
  return graph_copy_nodes_;
}

} // namespace gpu_lab
