#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda_runtime_api.h>

#include "megamoe/api/errors.hpp"
#include "megamoe/cuda/nccl_count_exchange.hpp"
#include "megamoe/metadata/serialize.hpp"
#include "cuda/device_notify_internal.hpp"
#include "cuda/nccl_protocol_internal.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

megamoe::RuntimeConfig make_config(std::uint32_t nodes, std::uint32_t gpus,
                                   std::uint32_t experts,
                                   std::uint32_t top_k,
                                   std::uint32_t mtile) {
  megamoe::RuntimeConfig config;
  config.num_experts = experts;
  config.top_k = top_k;
  config.m_tile_rows = mtile;
  config.topology.num_nodes = nodes;
  config.topology.world_size = nodes * gpus;
  for (std::uint32_t node = 0; node < nodes; ++node)
    for (std::uint32_t local = 0; local < gpus; ++local) {
      config.topology.rank_to_node.push_back(node);
      config.topology.rank_to_local_rank.push_back(local);
    }
  for (std::uint32_t expert = 0; expert < experts; ++expert)
    config.topology.expert_to_rank.push_back(
        expert % config.topology.world_size);
  return config;
}

std::vector<megamoe::RankRoutingInput>
seeded_inputs(std::uint32_t world, std::uint32_t experts,
              std::uint32_t top_k, std::uint64_t seed,
              std::uint64_t max_tokens = 10) {
  const auto next = [&seed]() mutable {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };
  std::vector<megamoe::RankRoutingInput> result(world);
  auto random = next;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    result[rank].num_tokens = random() % (max_tokens + 1);
    for (std::uint64_t token = 0; token < result[rank].num_tokens; ++token) {
      std::vector<std::uint32_t> choices(experts);
      for (std::uint32_t expert = 0; expert < experts; ++expert)
        choices[expert] = expert;
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        const auto selected = slot + static_cast<std::uint32_t>(
                                         random() % (experts - slot));
        std::swap(choices[slot], choices[selected]);
        result[rank].selected_experts.push_back(
            static_cast<std::int32_t>(choices[slot]));
      }
    }
  }
  return result;
}

std::vector<std::uint64_t> fixed_seeds() {
  return {0x5eed1234ULL, 0x10203040ULL, 0xdeadbeefULL, 0xc001d00dULL,
          0x01234567ULL, 0x89abcdefULL, 0x31415926ULL, 0x27182818ULL,
          0x0badf00dULL, 0xfeedfaceULL, 0x13579bdfULL, 0x2468ace0ULL,
          0xa5a5a5a5ULL, 0x5a5a5a5aULL, 0x11223344ULL, 0x55667788ULL,
          0x99aabbccULL, 0xddeeff00ULL, 0x42424242ULL, 0x77777777ULL,
          0xabcdef01ULL, 0x10fedcbaULL, 0x600dbeefULL, 0x7f4a7c15ULL,
          0xf00dcafeULL};
}

struct RunResult {
  std::string metadata;
  megamoe::cuda::CollectiveProtocolAudit audit;
  megamoe::cuda::NcclWorkspaceUsage workspace;
  megamoe::cuda::GenerationSetupTimings setup;
  megamoe::cuda::GenerationExecutionTimings execution;
  double session_setup_us = 0;
  double wall_us = 0;
  std::exception_ptr error;
};

std::vector<RunResult> run_fixture(
    const megamoe::RuntimeConfig &config,
    const std::vector<megamoe::RankRoutingInput> &routes,
    const std::string &identity, std::uint64_t generation,
    std::uint64_t max_tokens, bool stagger = false) {
  const auto world = config.topology.world_size;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<RunResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        auto execution = megamoe::cuda::make_cuda_execution(
            identity + "-exec-" + std::to_string(rank), device);
        const auto session_start = Clock::now();
        auto session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        const auto session_end = Clock::now();
        results[rank].session_setup_us =
            std::chrono::duration<double, std::micro>(session_end -
                                                      session_start)
                .count();
        if (stagger && rank % 3U != 0)
          std::this_thread::sleep_for(
              std::chrono::microseconds(20U * (rank % 3U)));
        const auto start = Clock::now();
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                    execution);
        auto result = coordinator.run({config, routes[rank], max_tokens});
        const auto finish = Clock::now();
        results[rank].metadata = megamoe::serialize_dispatch_metadata(
            result.dispatch().host());
        results[rank].audit = result.audit();
        results[rank].workspace = result.workspace_usage();
        results[rank].setup = result.setup_timings();
        results[rank].execution = result.execution_timings();
        results[rank].wall_us =
            std::chrono::duration<double, std::micro>(finish - start).count();
        megamoe::cuda::validate_cuda_dispatch_metadata(result.dispatch(),
                                                       execution);
      } catch (...) {
        results[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  return results;
}

struct ReusableRankResources {
  megamoe::cuda::CudaExecution execution;
  megamoe::cuda::NcclSession base_session;
  double session_setup_us = 0;
  std::exception_ptr error;
};

std::vector<ReusableRankResources>
make_reusable_resources(const megamoe::RuntimeConfig &config,
                        const std::string &identity,
                        std::uint64_t generation) {
  const auto world = config.topology.world_size;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<ReusableRankResources> resources(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        resources[rank].execution = megamoe::cuda::make_cuda_execution(
            identity + "-exec-" + std::to_string(rank), device);
        const auto start = Clock::now();
        resources[rank].base_session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        resources[rank].session_setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - start)
                .count();
      } catch (...) {
        resources[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &resource : resources)
    if (resource.error)
      std::rethrow_exception(resource.error);
  return resources;
}

std::vector<RunResult> run_reused_fixture(
    const megamoe::RuntimeConfig &config,
    const std::vector<megamoe::RankRoutingInput> &routes,
    const std::vector<ReusableRankResources> &resources,
    const std::string &identity, std::uint64_t generation,
    std::uint64_t max_tokens) {
  const auto world = config.topology.world_size;
  std::vector<RunResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto session = megamoe::cuda::make_nccl_generation(
            resources[rank].base_session, identity, generation);
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, resources[rank].execution);
        auto result = coordinator.run({config, routes[rank], max_tokens});
        results[rank].metadata = megamoe::serialize_dispatch_metadata(
            result.dispatch().host());
        results[rank].audit = result.audit();
        results[rank].workspace = result.workspace_usage();
        results[rank].setup = result.setup_timings();
        results[rank].execution = result.execution_timings();
      } catch (...) {
        results[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  return results;
}

void rethrow_errors(const std::vector<RunResult> &results) {
  for (const auto &result : results)
    if (result.error)
      std::rethrow_exception(result.error);
}

void test_local(std::uint32_t world) {
  const auto experts = std::max<std::uint32_t>(16, world * 2);
  const auto top_k = std::min<std::uint32_t>(6, experts);
  const auto config = make_config(1, world, experts, top_k, 7);
  for (const auto seed : fixed_seeds()) {
    const auto routes = seeded_inputs(world, experts, top_k, seed);
    const auto actual = run_fixture(config, routes, "local-fixture", seed + 1,
                                    10, true);
    rethrow_errors(actual);
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, routes, "local-fixture", seed + 1);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      require(actual[rank].metadata ==
                  megamoe::serialize_dispatch_metadata(expected[rank]),
              "CUDA/NCCL full-field mismatch rank=" +
                  std::to_string(rank) + " seed=" + std::to_string(seed));
      const auto audit = actual[rank].audit;
      require(audit.control_starts == 1 && audit.config_starts == 1 &&
                  audit.expert_count_starts == 1 &&
                  audit.packet_summary_starts == 1,
              "successful audit is not 1/1/1/1");
    }
  }
  const auto repeated = seeded_inputs(world, experts, top_k, 0x1234abcdULL);
  std::vector<std::string> baseline(world);
  for (std::uint64_t iteration = 0; iteration < 20; ++iteration) {
    const auto actual = run_fixture(config, repeated, "repeat", 900,
                                    10, iteration % 2 != 0);
    rethrow_errors(actual);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      if (iteration == 0)
        baseline[rank] = actual[rank].metadata;
      else
        require(actual[rank].metadata == baseline[rank],
                "20-run byte determinism mismatch");
    }
  }
  std::cout << "[PASS] coordinator local world=" << world
            << " 25 fixed seeds + 20 byte-identical repeats\n";
}

struct FailureResult {
  std::string error;
  megamoe::cuda::CollectiveProtocolAudit audit;
  std::exception_ptr unexpected;
};

void mutate_scenario(const std::string &scenario, std::uint32_t rank,
                     megamoe::RuntimeConfig &config,
                     megamoe::RankRoutingInput &routes,
                     std::uint64_t &max_tokens, std::string &identity,
                     std::uint64_t &generation) {
  if (scenario == "two_failures") {
    routes.num_tokens = 1;
    routes.selected_experts.clear();
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
      routes.selected_experts.push_back(static_cast<std::int32_t>(slot));
    if (rank == 0)
      routes.selected_experts.back() =
          static_cast<std::int32_t>(config.num_experts);
    else
      routes.selected_experts[1] = routes.selected_experts[0];
    return;
  }
  if (rank != 1)
    return;
  if (scenario == "out_of_range") {
    routes.num_tokens = 1;
    routes.selected_experts.clear();
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
      routes.selected_experts.push_back(
          slot + 1U == config.top_k
              ? static_cast<std::int32_t>(config.num_experts)
              : static_cast<std::int32_t>(slot));
  } else if (scenario == "duplicate") {
    routes.num_tokens = 1;
    routes.selected_experts.clear();
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
      routes.selected_experts.push_back(
          static_cast<std::int32_t>(slot == 1 ? 0 : slot));
  } else if (scenario == "max_mismatch") {
    max_tokens = 11;
  } else if (scenario == "over_capacity") {
    routes.num_tokens = 11;
    routes.selected_experts.clear();
    for (std::uint64_t token = 0; token < 11; ++token)
      for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
        routes.selected_experts.push_back(static_cast<std::int32_t>(slot));
  } else if (scenario == "generation_mismatch") {
    ++generation;
  } else if (scenario == "session_mismatch") {
    identity += "-different";
  } else if (scenario == "mtile_mismatch") {
    ++config.m_tile_rows;
  } else if (scenario == "multi_field_mismatch") {
    ++config.m_tile_rows;
    max_tokens = 11;
  } else if (scenario == "topology_mismatch") {
    std::swap(config.topology.expert_to_rank[0],
              config.topology.expert_to_rank[1]);
  } else if (scenario == "expert_capacity_mismatch") {
    config.capacity.expert_rows.assign(config.num_experts, 100);
    config.capacity.expert_rows[0] = 101;
  } else if (scenario == "workspace_capacity_mismatch") {
    config.capacity.workspace_rows_per_rank.assign(
        config.topology.world_size, 1000);
    config.capacity.workspace_rows_per_rank[0] = 1001;
  }
}

void test_failure(const std::string &scenario) {
  constexpr std::uint32_t world = 2;
  auto base_config = make_config(1, world, 8, 3, 4);
  if (scenario == "expert_capacity_mismatch")
    base_config.capacity.expert_rows.assign(base_config.num_experts, 100);
  if (scenario == "workspace_capacity_mismatch")
    base_config.capacity.workspace_rows_per_rank.assign(world, 1000);
  const auto base_routes = seeded_inputs(world, 8, 3, 0x55aaULL);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "failure-exec-" + std::to_string(rank), rank);
        auto config = base_config;
        auto routes = base_routes[rank];
        std::uint64_t max_tokens = 10;
        std::string identity = "failure-" + scenario;
        std::uint64_t generation = 1001;
        mutate_scenario(scenario, rank, config, routes, max_tokens, identity,
                        generation);
        auto session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, rank, bootstrap);
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                    execution);
        try {
          (void)coordinator.run({config, routes, max_tokens});
          throw std::runtime_error("invalid generation unexpectedly passed");
        } catch (const megamoe::NotifyDispatchError &error) {
          require(error.code() ==
                      megamoe::NotifyErrorCode::ControlProtocolFailure,
                  "logical mismatch was not a control protocol failure");
          results[rank].error = error.what();
          results[rank].audit = coordinator.audit();
        }
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  require(!results[0].error.empty() &&
              results[0].error == results[1].error,
          "ranks reported different structured errors");
  const bool config_failure = scenario == "topology_mismatch" ||
                              scenario == "expert_capacity_mismatch" ||
                              scenario == "workspace_capacity_mismatch";
  for (const auto &result : results) {
    require(result.audit.control_starts == 1,
            "failed generation omitted fixed control");
    require(result.audit.config_starts == (config_failure ? 1U : 0U),
            "failed generation has wrong config audit");
    require(result.audit.expert_count_starts == 0 &&
                result.audit.packet_summary_starts == 0,
            "failed generation entered data collective");
  }
  std::cout << "[PASS] coordinator rejection scenario=" << scenario
            << " error='" << results[0].error << "'\n";
}

void test_protocol_regressions() {
  for (const auto *scenario :
       {"out_of_range", "duplicate", "max_mismatch", "over_capacity",
        "generation_mismatch", "session_mismatch", "mtile_mismatch",
        "two_failures", "multi_field_mismatch",
        "topology_mismatch", "expert_capacity_mismatch",
        "workspace_capacity_mismatch"})
    test_failure(scenario);
}

void test_twenty_failure_recovery_rounds() {
  constexpr std::uint32_t world = 2;
  const auto config = make_config(1, world, 8, 3, 4);
  const std::vector<std::string> scenarios{
      "generation_mismatch", "session_mismatch", "topology_mismatch",
      "max_mismatch", "out_of_range"};
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::vector<std::vector<std::string>>> structured(
      scenarios.size(),
      std::vector<std::vector<std::string>>(
          20, std::vector<std::string>(world)));
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "recovery-exec-" + std::to_string(rank), rank);
        auto base = megamoe::cuda::make_nccl_session(
            "recovery-base", world, rank, 2000, rank, bootstrap);
        for (std::size_t category = 0; category < scenarios.size();
             ++category) {
          for (std::uint64_t round = 0; round < 20; ++round) {
            auto bad_config = config;
            auto routes = megamoe::RankRoutingInput{1, {0, 1, 2}};
            auto max_tokens = std::uint64_t{10};
            auto identity = std::string("recovery-") + scenarios[category];
            auto generation = 3000U +
                              static_cast<std::uint64_t>(category) * 100U +
                              round * 2U;
            mutate_scenario(scenarios[category], rank, bad_config, routes,
                            max_tokens, identity, generation);
            auto bad_session = megamoe::cuda::make_nccl_generation(
                base, identity, generation);
            megamoe::cuda::CudaNotifyGenerationCoordinator bad(bad_session,
                                                                execution);
            bool rejected = false;
            try {
              auto unexpected = bad.run({bad_config, routes, max_tokens});
              require(!unexpected,
                      "invalid generation published validated result");
            } catch (const megamoe::NotifyDispatchError &error) {
              rejected = true;
              structured[category][round][rank] = error.what();
            }
            require(rejected, "invalid recovery round was not rejected");
            const auto bad_audit = bad.audit();
            const bool config_failure =
                scenarios[category] == "topology_mismatch";
            require(bad_audit.control_starts == 1 &&
                        bad_audit.config_starts ==
                            (config_failure ? 1U : 0U) &&
                        bad_audit.expert_count_starts == 0 &&
                        bad_audit.packet_summary_starts == 0,
                    "invalid recovery audit crossed protocol boundary");

            auto good_session = megamoe::cuda::make_nccl_generation(
                base, "recovery-good-" + scenarios[category],
                3001U + static_cast<std::uint64_t>(category) * 100U +
                    round * 2U);
            megamoe::cuda::CudaNotifyGenerationCoordinator good(good_session,
                                                                 execution);
            auto result = good.run({config, {1, {0, 1, 2}}, 10});
            require(result.audit().control_starts == 1 &&
                        result.audit().config_starts == 1 &&
                        result.audit().expert_count_starts == 1 &&
                        result.audit().packet_summary_starts == 1,
                    "recovery audit is not 1/1/1/1");
            require(result && !megamoe::serialize_dispatch_metadata(
                                   result.dispatch().host()).empty(),
                    "recovery did not publish checked metadata");
          }
        }
      } catch (...) {
        errors[rank] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &error : errors)
    if (error)
      std::rethrow_exception(error);
  for (std::size_t category = 0; category < scenarios.size(); ++category)
    for (std::size_t round = 0; round < 20; ++round)
      require(!structured[category][round][0].empty() &&
                  structured[category][round][0] ==
                      structured[category][round][1],
              "recovery ranks selected different structured errors");
  std::cout << "[PASS] same ncclComm_t 5 categories x 20 invalid->valid recovery rounds\n";
}

void test_fault_injection() {
  using Point = megamoe::cuda::testing::NcclFaultPoint;
  struct FaultExpectation {
    Point point;
    megamoe::cuda::CollectiveProtocolAudit audit;
    const char *stage;
  };
  const std::vector<FaultExpectation> points{
      {Point::ControlStart, {0, 0, 0, 0}, "control-start"},
      {Point::ControlEnqueue, {0, 0, 0, 0}, "control-enqueue"},
      {Point::D2HStaging, {1, 0, 0, 0}, "control-d2h-staging"},
      {Point::EventRecord, {1, 0, 0, 0}, "control-event-record"},
      {Point::EventWait, {1, 0, 0, 0}, "control-event-wait"},
      {Point::ConfigSlotAllocation, {1, 0, 0, 0},
       "config-slot-allocation"},
      {Point::CountSlotAllocation, {1, 1, 0, 0},
       "count-slot-allocation"},
      {Point::SummarySlotAllocation, {1, 1, 0, 0},
       "summary-slot-allocation"},
      {Point::BackendCompletion, {1, 1, 1, 1},
       "summary-backend-completion"}};
  const auto config = make_config(1, 1, 8, 3, 4);
  for (std::size_t index = 0; index < points.size(); ++index) {
    megamoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto execution = megamoe::cuda::make_cuda_execution("fault-exec", 0);
    auto session = megamoe::cuda::make_nccl_session(
        "fault", 1, 0, 4000 + index, 0, bootstrap);
    megamoe::cuda::CollectiveProtocolAudit audit;
    {
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                  execution);
      megamoe::cuda::testing::inject_next_nccl_fault(points[index].point);
      bool failed = false;
      try {
        auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
        require(!unexpected, "fault generation published validated result");
      } catch (const megamoe::NotifyDispatchError &error) {
        failed = true;
        const std::string message = error.what();
        require(message.find(points[index].stage) != std::string::npos,
                "fault error lost exact stage");
        require(message.find("controlled test fault") != std::string::npos,
                "fault error lost original backend text");
      }
      require(failed, "fault injection did not fail generation");
      audit = coordinator.audit();
    }
    const auto expected = points[index].audit;
    require(audit.control_starts == expected.control_starts &&
                audit.config_starts == expected.config_starts &&
                audit.expert_count_starts == expected.expert_count_starts &&
                audit.packet_summary_starts ==
                    expected.packet_summary_starts,
            "fault audit does not match enqueue boundary");
    require(audit.control_starts <= 1 && audit.config_starts <= 1 &&
                audit.expert_count_starts <= 1 &&
                audit.packet_summary_starts <= 1,
            "fault audit regressed or double-incremented");
    const auto lifecycle =
        megamoe::cuda::testing::nccl_lifecycle_snapshot();
    require(lifecycle.slot_allocations == lifecycle.slot_destructions,
            "fault leaked a collective slot allocation");
    require(lifecycle.completion_lease_creates ==
                lifecycle.completion_lease_destroys,
            "fault leaked a completion lease");
    require(lifecycle.validated_publishes == 0,
            "fault published a validated generation");
    require(lifecycle.operation_cancels <= 1,
            "coordinator fault double-cancelled");
    require(lifecycle.operation_waits <= 4 &&
                lifecycle.destructor_drains <= 1,
            "fault lifecycle repeated wait/drain");
  }
  std::cout << "[PASS] 9-point NCCL enqueue audit/fault lifecycle table\n";
}

void test_config_wire_corruption(std::uint32_t world,
                                 std::uint32_t corrupt_rank) {
  const auto config = make_config(1, world, 8, 3, 4);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "wire-corruption-exec-" + std::to_string(rank), rank);
        auto session = megamoe::cuda::make_nccl_session(
            "wire-corruption", world, rank, 5100 + corrupt_rank, rank,
            bootstrap);
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                    execution);
        if (rank == corrupt_rank)
          megamoe::cuda::testing::corrupt_next_config_wire(64);
        try {
          auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
          require(!unexpected, "corrupt config published validated result");
        } catch (const megamoe::NotifyDispatchError &error) {
          results[rank].error = error.what();
          results[rank].audit = coordinator.audit();
        }
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  for (const auto &result : results) {
    require(!result.error.empty() &&
                result.error.find("checksum") != std::string::npos,
            "config corruption did not report integrity failure");
    require(result.audit.control_starts == 1 &&
                result.audit.config_starts == 1 &&
                result.audit.expert_count_starts == 0 &&
                result.audit.packet_summary_starts == 0,
            "config corruption audit is not 1/1/0/0");
  }
  for (std::uint32_t rank = 1; rank < world; ++rank)
    require(results[rank].error == results[0].error,
            "config corruption errors differ by validation rank");
  std::cout << "[PASS] config envelope corruption world=" << world
            << " origin_rank=" << corrupt_rank << " audit=1/1/0/0\n";
}

void test_exact_session_identity_agreement(bool different_length) {
  constexpr std::uint32_t world = 2;
  const auto config = make_config(1, world, 8, 3, 4);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "exact-session-exec-" + std::to_string(rank), rank);
        // Display name and header summary are deliberately equal. Only one raw
        // byte differs, proving that exact identity bytes are the final gate.
        auto session = megamoe::cuda::make_nccl_session(
            "same-display", world, rank, different_length ? 5201 : 5200,
            rank, bootstrap);
        std::vector<std::byte> raw_identity{
            std::byte{0x41}, std::byte{0x42},
            static_cast<std::byte>(0x43U + rank)};
        if (different_length && rank == 1)
          raw_identity.push_back(std::byte{0x44});
        megamoe::cuda::testing::override_next_protocol_identity(
            std::move(raw_identity),
            {0x1111111111111111ULL, 0x2222222222222222ULL});
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                    execution);
        try {
          auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
          require(!unexpected,
                  "different exact session identity published result");
        } catch (const megamoe::NotifyDispatchError &error) {
          results[rank].error = error.what();
          results[rank].audit = coordinator.audit();
        }
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  require(!results[0].error.empty() && results[0].error == results[1].error &&
              results[0].error.find("field=session_identity") !=
                  std::string::npos &&
              results[0].error.find("origin_rank=1") != std::string::npos,
          "exact session identity error is not globally deterministic");
  for (const auto &result : results)
    require(result.audit.control_starts == 1 &&
                result.audit.config_starts == 1 &&
                result.audit.expert_count_starts == 0 &&
                result.audit.packet_summary_starts == 0,
            "exact session identity audit is not 1/1/0/0");
  std::cout << "[PASS] equal summary/display but different raw session identity"
            << (different_length ? " length" : " byte")
            << " audit=1/1/0/0\n";
}

void test_device_only_sanitizer_paths() {
  const auto config = make_config(1, 1, 8, 3, 4);
  auto execution = megamoe::cuda::make_cuda_execution("device-only", 0);
  auto session = megamoe::make_collective_session("device-only", 1, 5001);
  auto context = megamoe::make_rank_collective_context(
      session, 0, execution.token());
  const auto run = [&](megamoe::RankRoutingInput input, bool valid) {
    auto routes = megamoe::cuda::upload_cuda_routes(input, config.top_k,
                                                     execution);
    auto plan = megamoe::cuda::query_cuda_notify_workspace(
        config, 0, input.num_tokens, 4);
    auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        plan, context, execution);
    auto operation = megamoe::cuda::start_cuda_rank_local_notify(
        config, context, execution, routes, std::move(workspace));
    const auto result = operation.wait_preflight();
    require(result.valid() == valid,
            "device-only validation status mismatch");
    if (!valid)
      require(result.status().error ==
                  megamoe::cuda::ControlLocalError::ExpertOutOfRange,
              "device error path lost coordinates");
  };
  run({3, {5, 1, 7, 1, 2, 3, 0, 7, 4}}, true);
  run({1, {0, 1, 8}}, false);
  std::cout << "[PASS] device-only normal/error preflight sanitizer paths\n";
}

void run_rank(std::uint32_t rank, std::uint32_t nodes,
              std::uint32_t gpus_per_node, std::uint32_t local_rank,
              std::uint64_t seed, int bootstrap_input_fd,
              int bootstrap_output_fd,
              const std::string &scenario) {
  const auto world = nodes * gpus_per_node;
  const auto experts = std::max<std::uint32_t>(16, world * 2);
  const auto top_k = std::min<std::uint32_t>(6, experts);
  const auto common_config = make_config(nodes, gpus_per_node, experts, top_k, 7);
  const auto common_routes = seeded_inputs(world, experts, top_k, seed)[rank];
  megamoe::cuda::NcclBootstrapId bootstrap;
  if (rank == 0) {
    require(rank == 0, "only rank 0 may generate bootstrap ID");
    require(bootstrap_output_fd >= 0 && bootstrap_input_fd < 0,
            "rank 0 requires only a bootstrap output descriptor");
    bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    megamoe::cuda::NcclBootstrapIo::write_fd(bootstrap,
                                              bootstrap_output_fd);
    require(::close(bootstrap_output_fd) == 0,
            "failed to close bootstrap output descriptor");
  } else {
    require(bootstrap_input_fd >= 0 && bootstrap_output_fd < 0,
            "nonzero rank requires only a bootstrap input descriptor");
    bootstrap = megamoe::cuda::NcclBootstrapIo::read_fd(bootstrap_input_fd);
    require(::close(bootstrap_input_fd) == 0,
            "failed to close bootstrap input descriptor");
  }
  auto execution = megamoe::cuda::make_cuda_execution(
      "multinode-exec-" + std::to_string(rank), local_rank);
  auto base = megamoe::cuda::make_nccl_session(
      "cuda-multinode-base", world, rank, 700, local_rank, bootstrap);
  if (scenario == "valid") {
    auto session = megamoe::cuda::make_nccl_generation(
        base, "cuda-multinode", 701);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                execution);
    auto result = coordinator.run({common_config, common_routes, 10});
    std::cout << megamoe::serialize_dispatch_metadata(result.dispatch().host())
              << '\n';
    return;
  }

  for (std::uint64_t round = 0; round < 20; ++round) {
    auto config = common_config;
    if (scenario == "expert_capacity_mismatch")
      config.capacity.expert_rows.assign(experts, 100);
    if (scenario == "workspace_capacity_mismatch")
      config.capacity.workspace_rows_per_rank.assign(world, 1000);
    auto routes = common_routes;
    std::uint64_t max_tokens = 10;
    std::string identity = "cuda-multinode-invalid-" + scenario;
    std::uint64_t generation = 800 + round * 2;
    mutate_scenario(scenario, rank, config, routes, max_tokens, identity,
                    generation);
    auto session = megamoe::cuda::make_nccl_generation(
        base, identity, generation);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                execution);
    try {
      (void)coordinator.run({config, routes, max_tokens});
      throw std::runtime_error("invalid cross-host generation passed");
    } catch (const megamoe::NotifyDispatchError &error) {
      const auto audit = coordinator.audit();
      require(audit.control_starts == 1 &&
                  audit.expert_count_starts == 0 &&
                  audit.packet_summary_starts == 0,
              "failed cross-host generation entered data collective");
      std::cout << "CONTROL_ERROR round=" << round << " " << error.what()
                << std::endl;
    }

    auto recovery_session = megamoe::cuda::make_nccl_generation(
        base, "cuda-multinode-recovery-" + scenario, 801 + round * 2);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(recovery_session,
                                                             execution);
    std::vector<std::int32_t> recovery_experts;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      recovery_experts.push_back(static_cast<std::int32_t>(slot));
    auto recovered = recovery.run({common_config,
                                   {1, std::move(recovery_experts)}, 10});
    require(recovered.audit().control_starts == 1 &&
                recovered.audit().config_starts == 1 &&
                recovered.audit().expert_count_starts == 1 &&
                recovered.audit().packet_summary_starts == 1,
            "cross-host recovery failed");
    require(recovered && !megamoe::serialize_dispatch_metadata(
                              recovered.dispatch().host()).empty(),
            "cross-host recovery metadata missing");
    std::cout << "RECOVERY_OK rank=" << rank << " round=" << round
              << std::endl;
  }
  throw megamoe::NotifyDispatchError(
      megamoe::NotifyErrorCode::ControlProtocolFailure,
      "expected invalid cross-host generation");
}

struct Statistics {
  double mean = 0, p50 = 0, p95 = 0, minimum = 0, maximum = 0;
};

Statistics summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  double sum = 0;
  for (const auto value : values)
    sum += value;
  const auto percentile = [&](double fraction) {
    const auto ordinal = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, ordinal) - 1];
  };
  return {sum / values.size(), percentile(.5), percentile(.95),
          values.front(), values.back()};
}

void benchmark(std::uint32_t world) {
  const auto config = make_config(1, world, 384, 6, 64);
  std::vector<megamoe::RankRoutingInput> routes(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    routes[rank].num_tokens = 8192;
    for (std::uint64_t token = 0; token < 8192; ++token) {
      const auto base = (token * 17U + rank * 31U) % 384U;
      for (std::uint32_t slot = 0; slot < 6; ++slot)
        routes[rank].selected_experts.push_back(static_cast<std::int32_t>(
            (base + slot * 53U) % 384U));
    }
  }
  constexpr std::size_t warmup = 10;
  constexpr std::size_t measured = 50;
  std::vector<double> steady;
  std::vector<double> setup;
  std::vector<double> generation_end_to_end;
  using SetupMember = double megamoe::cuda::GenerationSetupTimings::*;
  const std::vector<std::pair<const char *, SetupMember>> setup_stages{
      {"coordinator_backend_create_us", &megamoe::cuda::GenerationSetupTimings::coordinator_backend_create_us},
      {"control_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::control_slot_allocation_us},
      {"config_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::config_slot_allocation_us},
      {"count_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::count_slot_allocation_us},
      {"summary_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::summary_slot_allocation_us},
      {"event_allocation_us", &megamoe::cuda::GenerationSetupTimings::event_allocation_us},
      {"pinned_allocation_us", &megamoe::cuda::GenerationSetupTimings::pinned_allocation_us},
      {"device_allocation_us", &megamoe::cuda::GenerationSetupTimings::device_allocation_us}};
  std::vector<std::vector<double>> setup_stage_samples(setup_stages.size());
  megamoe::cuda::NcclWorkspaceUsage peak;
  // One communicator/session establishment per rank, once for the complete
  // 10+50 run. Samples are the ranks participating in that one establishment.
  const auto resources =
      make_reusable_resources(config, "benchmark-session", 2999);
  std::vector<double> session_setup;
  for (const auto &resource : resources)
    session_setup.push_back(resource.session_setup_us);
  const auto session_setup_stats = summarize(session_setup);
  for (std::size_t iteration = 0; iteration < warmup + measured; ++iteration) {
    const auto result = run_reused_fixture(
        config, routes, resources, "benchmark-session",
        3000 + iteration, 8192);
    rethrow_errors(result);
    if (iteration >= warmup) {
      double slow_wall = 0;
      double slow_setup = 0;
      double slow_steady = 0;
      std::vector<double> slow_setup_stages(setup_stages.size(), 0.0);
      for (const auto &rank : result) {
        const auto rank_steady = rank.execution.steady_execution_us;
        const auto rank_setup = rank.setup.per_generation_setup_total_us;
        // Compute setup-inclusive E2E per rank first; only then select the
        // iteration's slowest rank. No cross-rank max subtraction is used.
        slow_wall = std::max(slow_wall, rank_setup + rank_steady);
        slow_setup = std::max(slow_setup, rank_setup);
        slow_steady = std::max(slow_steady, rank_steady);
        for (std::size_t stage = 0; stage < setup_stages.size(); ++stage)
          slow_setup_stages[stage] = std::max(
              slow_setup_stages[stage], rank.setup.*setup_stages[stage].second);
        peak.device_bytes = std::max(peak.device_bytes,
                                     rank.workspace.device_bytes);
        peak.pinned_host_bytes = std::max(
            peak.pinned_host_bytes, rank.workspace.pinned_host_bytes);
        peak.control_slot_bytes = std::max(
            peak.control_slot_bytes, rank.workspace.control_slot_bytes);
        peak.config_wire_bytes = std::max(
            peak.config_wire_bytes, rank.workspace.config_wire_bytes);
        peak.config_slot_bytes = std::max(
            peak.config_slot_bytes, rank.workspace.config_slot_bytes);
        peak.expert_slot_bytes = std::max(
            peak.expert_slot_bytes, rank.workspace.expert_slot_bytes);
        peak.summary_slot_bytes = std::max(
            peak.summary_slot_bytes, rank.workspace.summary_slot_bytes);
        peak.route_input_device_bytes = std::max(
            peak.route_input_device_bytes,
            rank.workspace.route_input_device_bytes);
        peak.device_notify_arena_bytes = std::max(
            peak.device_notify_arena_bytes,
            rank.workspace.device_notify_arena_bytes);
        peak.device_notify_pinned_staging_bytes = std::max(
            peak.device_notify_pinned_staging_bytes,
            rank.workspace.device_notify_pinned_staging_bytes);
        peak.final_device_metadata_arena_view_capacity_bytes = std::max(
            peak.final_device_metadata_arena_view_capacity_bytes,
            rank.workspace.final_device_metadata_arena_view_capacity_bytes);
        peak.event_count = std::max(peak.event_count,
                                    rank.workspace.event_count);
        peak.quantified_event_bytes = std::max(
            peak.quantified_event_bytes,
            rank.workspace.quantified_event_bytes);
      }
      steady.push_back(slow_steady);
      setup.push_back(slow_setup);
      generation_end_to_end.push_back(slow_wall);
      for (std::size_t stage = 0; stage < setup_stages.size(); ++stage)
        setup_stage_samples[stage].push_back(slow_setup_stages[stage]);
    }
  }
  const auto steady_stats = summarize(steady);
  const auto setup_stats = summarize(setup);
  const auto generation_stats = summarize(generation_end_to_end);
  std::vector<Statistics> setup_stage_stats;
  for (auto &samples : setup_stage_samples)
    setup_stage_stats.push_back(summarize(std::move(samples)));
  std::cout << "BENCH world=" << world
            << " warmup=10 measured=50 aggregation=slowest-rank"
               " session_reused=true\n"
            << "  one_time_session_setup_us mean="
            << session_setup_stats.mean << " p50=" << session_setup_stats.p50
            << " p95=" << session_setup_stats.p95
            << " min=" << session_setup_stats.minimum
            << " max=" << session_setup_stats.maximum << '\n'
            << "  per_generation_setup_us mean=" << setup_stats.mean
            << " p50=" << setup_stats.p50 << " p95=" << setup_stats.p95
            << " min=" << setup_stats.minimum
            << " max=" << setup_stats.maximum << '\n'
            << "  setup_inclusive_generation_end_to_end_us mean="
            << generation_stats.mean << " p50=" << generation_stats.p50
            << " p95=" << generation_stats.p95
            << " min=" << generation_stats.minimum
            << " max=" << generation_stats.maximum << '\n'
            << "  steady_state_execution_us mean=" << steady_stats.mean
            << " p50=" << steady_stats.p50
            << " p95=" << steady_stats.p95
            << " min=" << steady_stats.minimum
            << " max=" << steady_stats.maximum << '\n';
  for (std::size_t stage = 0; stage < setup_stages.size(); ++stage) {
    const auto &stats = setup_stage_stats[stage];
    std::cout << "  " << setup_stages[stage].first << " mean=" << stats.mean
              << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  }
  std::cout << "BENCH_JSON {\"schema\":3,\"warmup\":10,\"measured\":50,"
               "\"aggregation\":\"slowest-rank\",\"session_reused\":true,"
               "\"session_setup_sample\":\"one establishment; samples are ranks\","
               "\"one_time_session_setup_us\":{\"mean\":"
            << session_setup_stats.mean << ",\"p50\":"
            << session_setup_stats.p50 << ",\"p95\":"
            << session_setup_stats.p95 << ",\"min\":"
            << session_setup_stats.minimum << ",\"max\":"
            << session_setup_stats.maximum
            << "},\"per_generation_setup_us\":{\"mean\":"
            << setup_stats.mean << ",\"p50\":" << setup_stats.p50
            << ",\"p95\":" << setup_stats.p95 << ",\"min\":"
            << setup_stats.minimum << ",\"max\":" << setup_stats.maximum
            << "},\"setup_inclusive_generation_end_to_end_us\":{\"mean\":"
            << generation_stats.mean << ",\"p50\":"
            << generation_stats.p50 << ",\"p95\":"
            << generation_stats.p95 << ",\"min\":"
            << generation_stats.minimum << ",\"max\":"
            << generation_stats.maximum
            << "},\"steady_state_execution_us\":{\"mean\":"
            << steady_stats.mean << ",\"p50\":" << steady_stats.p50
            << ",\"p95\":" << steady_stats.p95 << ",\"min\":"
            << steady_stats.minimum << ",\"max\":" << steady_stats.maximum
            << "},\"setup_stages\":{";
  for (std::size_t stage = 0; stage < setup_stages.size(); ++stage) {
    if (stage != 0)
      std::cout << ',';
    const auto &stats = setup_stage_stats[stage];
    std::cout << '\"' << setup_stages[stage].first << "\":{\"mean\":"
              << stats.mean << ",\"p50\":" << stats.p50
              << ",\"p95\":" << stats.p95 << ",\"min\":"
              << stats.minimum << ",\"max\":" << stats.maximum << '}';
  }
  std::cout << "},\"workspace\":{\"control_slot\":"
            << peak.control_slot_bytes << ",\"config_wire\":"
            << peak.config_wire_bytes << ",\"config_slot\":"
            << peak.config_slot_bytes << ",\"expert_slot\":"
            << peak.expert_slot_bytes << ",\"summary_slot\":"
            << peak.summary_slot_bytes << ",\"device\":"
            << peak.device_bytes << ",\"pinned\":"
            << peak.pinned_host_bytes
            << ",\"route_input_device\":"
            << peak.route_input_device_bytes
            << ",\"device_notify_arena\":"
            << peak.device_notify_arena_bytes
            << ",\"device_notify_pinned_staging\":"
            << peak.device_notify_pinned_staging_bytes
            << ",\"final_device_metadata_arena_view_capacity\":"
            << peak.final_device_metadata_arena_view_capacity_bytes
            << ",\"final_metadata_is_arena_subview\":true"
            << ",\"event_count\":" << peak.event_count
            << ",\"quantified_event_bytes\":"
            << peak.quantified_event_bytes
            << "},\"hot_path\":{\"per_route_copy\":false,"
               "\"per_route_allocation\":false,\"per_route_sync\":false}}\n";
}

void print_nccl_runtime_info() {
  const auto info = megamoe::cuda::nccl_runtime_info();
  std::cout << "NCCL_RUNTIME compiled=" << info.compiled_version
            << " runtime=" << info.runtime_version
            << " library=" << info.loaded_library_path
            << " NCCL_SOCKET_IFNAME=" << info.socket_ifname
            << " NCCL_IB_DISABLE=" << info.ib_disable << '\n';
}

std::uint64_t parse_u64(const char *value) {
  return static_cast<std::uint64_t>(std::stoull(value));
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--local") {
      const auto world = static_cast<std::uint32_t>(parse_u64(argv[2]));
      print_nccl_runtime_info();
      test_local(world);
      if (world == 2) {
        test_protocol_regressions();
        test_twenty_failure_recovery_rounds();
        test_config_wire_corruption(2, 0);
        test_config_wire_corruption(2, 1);
        test_exact_session_identity_agreement(false);
        test_exact_session_identity_agreement(true);
      }
      if (world == 1) {
        test_fault_injection();
        test_config_wire_corruption(1, 0);
      }
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--protocol-regressions") {
      test_protocol_regressions();
      test_twenty_failure_recovery_rounds();
      test_config_wire_corruption(2, 0);
      test_config_wire_corruption(2, 1);
      test_exact_session_identity_agreement(false);
      test_exact_session_identity_agreement(true);
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--smoke") {
      test_local(1);
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--device-smoke") {
      test_device_only_sanitizer_paths();
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--bench") {
      benchmark(static_cast<std::uint32_t>(parse_u64(argv[2])));
      return 0;
    }
    if (argc >= 14 && std::string(argv[1]) == "--rank-runner") {
      std::uint32_t rank = 0, nodes = 0, gpus = 0, local = 0;
      std::uint64_t seed = 0;
      int bootstrap_input_fd = -1;
      int bootstrap_output_fd = -1;
      std::string scenario = "valid";
      for (int index = 2; index + 1 < argc; index += 2) {
        const std::string key = argv[index];
        if (key == "--rank") rank = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--nodes") nodes = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--gpus-per-node") gpus = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--local-rank") local = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--seed") seed = parse_u64(argv[index + 1]);
        else if (key == "--bootstrap-input-fd") bootstrap_input_fd = std::stoi(argv[index + 1]);
        else if (key == "--bootstrap-output-fd") bootstrap_output_fd = std::stoi(argv[index + 1]);
        else if (key == "--scenario") scenario = argv[index + 1];
      }
      print_nccl_runtime_info();
      run_rank(rank, nodes, gpus, local, seed, bootstrap_input_fd,
               bootstrap_output_fd, scenario);
      return 0;
    }
    throw std::runtime_error("invalid arguments");
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
