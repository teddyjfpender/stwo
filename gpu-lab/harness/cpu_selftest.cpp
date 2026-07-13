#include "budget.hpp"
#include "effects.hpp"
#include "fri_round6.hpp"
#include "identity.hpp"
#include "replay.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char **argv) {
  using namespace gpu_lab;
  try {
    primitive_self_test();
    plan_binding_self_test();
    effect_snapshot_self_test();
    benchmark_budget_self_test();
    fri_round6_layout_self_test();
    if (argc == 7 && std::string_view(argv[1]) == "--plan" &&
        std::string_view(argv[3]) == "--plan-sha256" &&
        std::string_view(argv[5]) == "--replay") {
      const Plan plan = load_plan(argv[2], argv[4]);
      const Replay replay = load_replay(argv[6], plan);
      std::cout << "validated replay rows=" << replay.rows
                << " table_len=" << replay.table_len << "\n";
    } else if (argc != 1) {
      throw std::runtime_error(
          "usage: cpu-self-test [--plan PATH --plan-sha256 SHA256 "
          "--replay PATH]");
    }
    std::cout << "stwo-gpu-lab CPU parser/effect self-test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "stwo-gpu-lab CPU parser/effect self-test: " << error.what()
              << '\n';
    return 1;
  }
}
