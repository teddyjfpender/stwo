#pragma once

namespace gpu_lab {

double remaining_ms(double limit_ms, double used_ms);
bool sample_fits(double limit_ms, double used_ms, double predicted_ms);
void benchmark_budget_self_test();

} // namespace gpu_lab
