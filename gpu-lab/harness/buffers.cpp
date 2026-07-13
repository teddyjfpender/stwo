#include "runner.hpp"

#include "cuda_check.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gpu_lab {

CUdeviceptr Runner::allocate_bytes(std::size_t bytes,
                                   const std::string &label) {
  if (bytes == 0 ||
      bytes > std::numeric_limits<std::size_t>::max() - 2 * kEffectGuardBytes)
    throw std::runtime_error("invalid guarded device allocation: " + label);
  CUdeviceptr base = 0;
  const std::size_t allocation_bytes = bytes + 2 * kEffectGuardBytes;
  GPU_LAB_CU_CHECK(cuMemAlloc(&base, allocation_bytes));
  allocations_.push_back(base);
  GPU_LAB_CU_CHECK(cuMemsetD8(base, kEffectGuardByte, allocation_bytes));
  const CUdeviceptr data = base + kEffectGuardBytes;
  guarded_regions_.push_back(
      {base, data, EffectExpectation{label, bytes, false, {}}});
  return data;
}

CUdeviceptr Runner::upload_bytes(const void *bytes, std::size_t size,
                                 const std::string &label) {
  CUdeviceptr pointer = allocate_bytes(size, label);
  GPU_LAB_CU_CHECK(cuMemcpyHtoD(pointer, bytes, size));
  auto &expectation = guarded_regions_.back().expectation;
  expectation.immutable = true;
  const auto *begin = static_cast<const std::uint8_t *>(bytes);
  expectation.expected.assign(begin, begin + size);
  return pointer;
}

CUdeviceptr Runner::upload_words(const std::uint32_t *words, std::size_t count,
                                 const std::string &label) {
  return upload_bytes(words, count * sizeof(std::uint32_t), label);
}

CUdeviceptr Runner::upload_pointers(const std::vector<CUdeviceptr> &pointers,
                                    const std::string &label) {
  static_assert(sizeof(CUdeviceptr) == 8,
                "reviewed first-slice ABI requires 64-bit device pointers");
  return upload_bytes(pointers.data(), pointers.size() * sizeof(CUdeviceptr),
                      label);
}

void Runner::upload() {
  const std::size_t rows = replay_.rows;
  for (std::uint32_t column = 0; column < plan_.input_columns; ++column) {
    const std::string label = "input_cols[" + std::to_string(column) + "]";
    input_columns_.push_back(
        upload_words(replay_.inputs.data() + column * rows, rows, label));
  }
  input_pointer_table_ = upload_pointers(input_columns_, "input_cols table");

  table_ = upload_words(replay_.table.data(), replay_.table.size(),
                        "table_bases[0]");
  std::vector<CUdeviceptr> table_pointers(plan_.table_slots, 0);
  table_pointers[0] = table_;
  table_pointer_table_ = upload_pointers(table_pointers, "table_bases table");
  std::vector<std::uint32_t> table_strides(plan_.stride_words, 0);
  table_strides[0] = replay_.table_len;
  table_strides_ =
      upload_words(table_strides.data(), table_strides.size(), "table_strides");

  for (std::uint32_t column = 0; column < plan_.output_columns; ++column) {
    const std::string label = "out_cols[" + std::to_string(column) + "]";
    output_columns_.push_back(
        allocate_bytes(rows * sizeof(std::uint32_t), label));
  }
  output_pointer_table_ = upload_pointers(output_columns_, "out_cols table");

  const std::uint32_t zero = 0;
  CUdeviceptr dummy_count = upload_words(&zero, 1, "mult_counts[0]");
  mult_count_pointer_table_ =
      upload_pointers({dummy_count}, "mult_counts table");
  lookup_ = allocate_bytes(
      checked_words(plan_.lookup_columns, replay_.rows, "lookup") *
          sizeof(std::uint32_t),
      "lookup_words");
  sub_ = allocate_bytes(checked_words(plan_.sub_columns, replay_.rows, "sub") *
                            sizeof(std::uint32_t),
                        "sub_words");
}

void Runner::bind_inputs(bool mutated) {
  const auto &inputs = mutated ? replay_.mutated_inputs : replay_.inputs;
  const std::size_t column_bytes = replay_.rows * sizeof(std::uint32_t);
  if (inputs.size() != checked_words(plan_.input_columns, replay_.rows, "inputs") ||
      input_columns_.size() != plan_.input_columns)
    throw std::runtime_error("proof-varying input binding shape mismatch");
  for (std::uint32_t column = 0; column < plan_.input_columns; ++column) {
    const auto *begin = reinterpret_cast<const std::uint8_t *>(
        inputs.data() + std::size_t(column) * replay_.rows);
    GPU_LAB_CU_CHECK(
        cuMemcpyHtoD(input_columns_[column], begin, column_bytes));
    const auto region = std::find_if(
        guarded_regions_.begin(), guarded_regions_.end(),
        [&](const GuardedRegion &candidate) {
          return candidate.data == input_columns_[column];
        });
    if (region == guarded_regions_.end() || !region->expectation.immutable)
      throw std::runtime_error("input binding lacks an immutable effect region");
    region->expectation.expected.assign(begin, begin + column_bytes);
  }
}

void Runner::reset_outputs() {
  for (CUdeviceptr column : output_columns_)
    GPU_LAB_CU_CHECK(
        cuMemsetD32Async(column, 0xDEADBEEFu, replay_.rows, stream_));
  GPU_LAB_CU_CHECK(cuMemsetD32Async(
      lookup_, 0xDEADBEEFu,
      checked_words(plan_.lookup_columns, replay_.rows, "lookup"), stream_));
  GPU_LAB_CU_CHECK(cuMemsetD32Async(
      sub_, 0xDEADBEEFu, checked_words(plan_.sub_columns, replay_.rows, "sub"),
      stream_));
}

bool Runner::validate_effects(Validation &validation) {
  bool passed = true;
  for (const GuardedRegion &region : guarded_regions_) {
    const std::size_t bytes =
        region.expectation.payload_bytes + 2 * kEffectGuardBytes;
    std::vector<std::uint8_t> snapshot(bytes);
    GPU_LAB_CU_CHECK(cuMemcpyDtoH(snapshot.data(), region.base, bytes));
    std::string error;
    if (!validate_effect_snapshot(snapshot, region.expectation, error)) {
      passed = false;
      if (validation.error.empty())
        validation.error = error;
    }
  }
  return passed;
}

Validation Runner::validate_outputs(bool mutated) {
  Validation result;
  std::vector<std::uint32_t> output(
      checked_words(plan_.output_columns, replay_.rows, "output"));
  std::vector<std::uint32_t> lookup(
      checked_words(plan_.lookup_columns, replay_.rows, "lookup"));
  std::vector<std::uint32_t> sub(
      checked_words(plan_.sub_columns, replay_.rows, "sub"));
  for (std::uint32_t column = 0; column < plan_.output_columns; ++column)
    GPU_LAB_CU_CHECK(cuMemcpyDtoH(
        output.data() + std::size_t(column) * replay_.rows,
        output_columns_[column], replay_.rows * sizeof(std::uint32_t)));
  GPU_LAB_CU_CHECK(cuMemcpyDtoH(lookup.data(), lookup_,
                                lookup.size() * sizeof(std::uint32_t)));
  GPU_LAB_CU_CHECK(
      cuMemcpyDtoH(sub.data(), sub_, sub.size() * sizeof(std::uint32_t)));
  auto compare = [&](const std::vector<std::uint32_t> &got,
                     const std::vector<std::uint32_t> &expected,
                     const char *label) {
    const auto mismatch =
        std::mismatch(got.begin(), got.end(), expected.begin(), expected.end());
    result.checked_words += expected.size();
    if (mismatch.first == got.end())
      return true;
    if (result.error.empty()) {
      const std::size_t index =
          static_cast<std::size_t>(mismatch.first - got.begin());
      std::ostringstream message;
      message << label << " word " << index << ": got " << *mismatch.first
              << ", expected " << *mismatch.second;
      result.error = message.str();
    }
    return false;
  };
  const auto &expected_output =
      mutated ? replay_.mutated_expected_output : replay_.expected_output;
  const auto &expected_lookup =
      mutated ? replay_.mutated_expected_lookup : replay_.expected_lookup;
  const auto &expected_sub =
      mutated ? replay_.mutated_expected_sub : replay_.expected_sub;
  const bool output_passed = compare(output, expected_output, "output");
  const bool lookup_passed = compare(lookup, expected_lookup, "lookup");
  const bool sub_passed = compare(sub, expected_sub, "sub");
  const bool effects_passed = validate_effects(result);
  result.passed =
      output_passed && lookup_passed && sub_passed && effects_passed;
  return result;
}

} // namespace gpu_lab
