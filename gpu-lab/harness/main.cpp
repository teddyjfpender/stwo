#include "budget.hpp"
#include "effects.hpp"
#include "identity.hpp"
#include "replay.hpp"
#include "result.hpp"
#include "runner.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  using namespace gpu_lab;
  try {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
      primitive_self_test();
      plan_binding_self_test();
      effect_snapshot_self_test();
      benchmark_budget_self_test();
      std::cout << "stwo-gpu-lab primitive self-test: PASS\n";
      return 0;
    }
    const Options options = parse_options(argc, argv);
    remove_old_result(options.result);
    if (digest_hex(sha256_file(options.manifest)) != options.manifest_sha256)
      throw std::runtime_error("execution manifest SHA256 mismatch");
    const Plan plan = load_plan(options.plan, options.plan_sha256);
    const Replay replay = load_replay(options.replay, plan);
    Runner runner(options, plan, replay);
    const Validation not_run{false, 0, "not run"};
    const Validation eager = runner.eager_correctness(false);
    if (!eager.passed) {
      const std::string json =
          make_result(options, plan, runner, eager, not_run, not_run, not_run,
                      nullptr);
      write_result_atomic(options.result, json);
      std::cout << json;
      return 2;
    }
    const Validation graph = runner.graph_correctness(false);
    if (!graph.passed) {
      const std::string json = make_result(options, plan, runner, eager, graph,
                                           not_run, not_run, nullptr);
      write_result_atomic(options.result, json);
      std::cout << json;
      return 2;
    }
    const Validation mutated_eager = runner.eager_correctness(true);
    if (!mutated_eager.passed) {
      const std::string json = make_result(options, plan, runner, eager, graph,
                                           mutated_eager, not_run, nullptr);
      write_result_atomic(options.result, json);
      std::cout << json;
      return 2;
    }
    const Validation mutated_graph = runner.graph_correctness(true);
    if (!mutated_graph.passed) {
      const std::string json = make_result(options, plan, runner, eager, graph,
                                           mutated_eager, mutated_graph,
                                           nullptr);
      write_result_atomic(options.result, json);
      std::cout << json;
      return 2;
    }
    Benchmark benchmark;
    Benchmark *benchmark_pointer = nullptr;
    if (options.mode == "benchmark") {
      benchmark_pointer = &benchmark;
      try {
        benchmark = runner.benchmark();
      } catch (const std::exception &error) {
        benchmark.error = error.what();
        benchmark.post_eager.error = "not completed after performance failure";
        benchmark.post_graph.error = "not completed after performance failure";
      }
    }
    const std::string json = make_result(options, plan, runner, eager, graph,
                                         mutated_eager, mutated_graph,
                                         benchmark_pointer);
    write_result_atomic(options.result, json);
    std::cout << json;
    return benchmark_pointer && !benchmark_pointer->passed() ? 2 : 0;
  } catch (const std::exception &error) {
    std::cerr << "stwo-gpu-lab: " << error.what() << '\n';
    return 1;
  }
}
