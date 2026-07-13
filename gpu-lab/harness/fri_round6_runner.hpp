#pragma once

#include "fri_round6.hpp"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_lab {

struct FriValidation {
  bool passed = false;
  std::size_t checked_words = 0;
  std::string error;
};

class FriRound6Runner {
public:
  FriRound6Runner(int device_ordinal, std::uint32_t target_sm,
                  const std::filesystem::path &module_path,
                  const std::string &module_sha256,
                  const FriRound6Fixture &binding_fixture);
  FriRound6Runner(const FriRound6Runner &) = delete;
  FriRound6Runner &operator=(const FriRound6Runner &) = delete;
  ~FriRound6Runner();

  FriValidation run_eager(const FriRound6Fixture &fixture);
  FriValidation run_graph(const FriRound6Fixture &fixture);
  bool stale_cursor_sets_order_status(const FriRound6Fixture &fixture,
                                      std::string &error);
  std::string device_name() const;
  std::string device_uuid() const;
  int driver_version() const;
  std::size_t graph_kernel_nodes() const;
  std::size_t graph_copy_nodes() const;

private:
  struct Region {
    CUdeviceptr base = 0;
    CUdeviceptr data = 0;
    std::size_t bytes = 0;
    std::string label;
  };

  void cleanup() noexcept;
  CUdeviceptr allocate(std::size_t bytes, const char *label);
  CUdeviceptr upload_pointer_table(const std::vector<CUdeviceptr> &pointers,
                                   const char *label);
  void allocate_buffers();
  void reset(const FriRound6Fixture &fixture);
  void launch_segment();
  void ensure_graph();
  void validate_graph_contract();
  FriValidation validate(const FriRound6Fixture &fixture);
  bool validate_guards(std::string &error);

  int device_ordinal_ = 0;
  std::uint32_t target_sm_ = 0;
  std::vector<std::uint8_t> module_image_;
  CUdevice device_ = 0;
  CUcontext previous_context_ = nullptr;
  CUcontext context_ = nullptr;
  CUstream stream_ = nullptr;
  CUmodule module_ = nullptr;
  CUfunction fold_ = nullptr;
  CUfunction leaf_ = nullptr;
  CUfunction parent_ = nullptr;
  CUfunction mix_ = nullptr;
  CUfunction draw_ = nullptr;
  CUgraph graph_ = nullptr;
  CUgraphExec graph_executable_ = nullptr;
  bool primary_retained_ = false;
  std::uint64_t chain34_ = 0;
  std::uint64_t chain35_ = 0;
  std::uint64_t chain36_ = 0;
  std::size_t graph_kernel_nodes_ = 0;
  std::size_t graph_copy_nodes_ = 0;
  std::vector<Region> regions_;

  CUdeviceptr pong_ = 0;
  CUdeviceptr ping_ = 0;
  CUdeviceptr retained_ = 0;
  CUdeviceptr twiddles_ = 0;
  CUdeviceptr alpha_ = 0;
  CUdeviceptr pong_table_ = 0;
  CUdeviceptr ping_table_ = 0;
  CUdeviceptr retained_table_ = 0;
  CUdeviceptr leaves_ = 0;
  CUdeviceptr root_ = 0;
  CUdeviceptr root_input_ = 0;
  CUdeviceptr state_ = 0;
  CUdeviceptr mix_input_ = 0;
  CUdeviceptr boundary_mix_ = 0;
  CUdeviceptr challenge_ = 0;
  CUdeviceptr draw_output_ = 0;
  CUdeviceptr boundary_draw_ = 0;
  CUdeviceptr challenge_copy_ = 0;
};

} // namespace gpu_lab
