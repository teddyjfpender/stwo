#include "budget.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpu_lab {

double remaining_ms(double limit_ms, double used_ms) {
  return std::max(0.0, limit_ms - used_ms);
}

bool sample_fits(double limit_ms, double used_ms, double predicted_ms) {
  return predicted_ms <= remaining_ms(limit_ms, used_ms);
}

void benchmark_budget_self_test() {
  if (remaining_ms(30.0, 7.5) != 22.5 || !sample_fits(30.0, 25.0, 5.0) ||
      sample_fits(30.0, 25.001, 5.0) || remaining_ms(30.0, 31.0) != 0.0)
    throw std::runtime_error("benchmark budget arithmetic self-test failed");
}

} // namespace gpu_lab
