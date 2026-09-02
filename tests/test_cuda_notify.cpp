#include "cuda/testing/api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda_runtime_api.h>

#include "megamoe/api/errors.hpp"
#include "megamoe/metadata/serialize.hpp"
#include "cuda/activation_transport_internal.hpp"
#include "cuda/device_notify_internal.hpp"
#include "cuda/nccl_protocol_internal.hpp"
#include "cuda/testing/generation_scheduler.hpp"

namespace {

using Clock = std::chrono::steady_clock;

#if MEGAMOE_DEVICE_TIMELINE
void write_device_timeline(
    const megamoe::cuda::profiling::DeviceTimelineSnapshot &snapshot,
    const std::string &prefix) {
  const auto phase = "forward";
  const auto path = prefix + ".rank" + std::to_string(snapshot.rank) + "." +
                    phase + ".json";
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    throw std::runtime_error("cannot create device timeline: " + path);
  output << "{\"schema\":" << snapshot.schema_version
         << ",\"rank\":" << snapshot.rank
         << ",\"generation\":" << snapshot.generation
         << ",\"phase\":\"" << phase << "\""
         << ",\"max_ctas\":" << snapshot.max_ctas
         << ",\"capacity_per_cta\":" << snapshot.capacity_per_cta
         << ",\"overflow_events\":" << snapshot.overflow_events
         << ",\"events\":[";
  for (std::size_t index = 0; index < snapshot.records.size(); ++index) {
    const auto &event = snapshot.records[index];
    if (index != 0)
      output << ',';
    output << "{\"timestamp_ns\":" << event.timestamp_ns
           << ",\"generation\":" << event.generation
           << ",\"rank\":" << event.rank << ",\"cta\":" << event.cta
           << ",\"channel\":" << event.channel
           << ",\"phase\":" << ((event.metadata >> 24U) & 0xffU)
           << ",\"role\":" << ((event.metadata >> 16U) & 0xffU)
           << ",\"event\":" << (event.metadata & 0xffffU)
           << ",\"arg0\":" << event.arg0 << ",\"arg1\":" << event.arg1
           << '}';
  }
  output << "]}\n";
  if (!output)
    throw std::runtime_error("failed writing device timeline: " + path);
  std::cout << "DEVICE_TIMELINE rank=" << snapshot.rank
            << " phase=" << phase << " events=" << snapshot.records.size()
            << " overflow=" << snapshot.overflow_events
            << " path=" << path << std::endl;
}
#endif

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

megamoe::cuda::CudaGateInput make_unit_gates(
    const megamoe::cuda::CudaRouteInput &routes,
    const megamoe::RankCollectiveContext &context,
    const megamoe::cuda::CudaExecution &execution) {
  return megamoe::cuda::testing::upload_fp32_gates_for_testing(
      std::vector<float>(static_cast<std::size_t>(routes.route_count()),
                         1.0F),
      routes, context, execution);
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

std::vector<float> fixed_random_selected_gates(
    std::uint64_t tokens, std::uint32_t top_k, std::uint32_t rank,
    std::uint64_t seed) {
  std::vector<float> gates(
      static_cast<std::size_t>(tokens * static_cast<std::uint64_t>(top_k)));
  for (std::uint64_t token = 0; token < tokens; ++token) {
    std::uint64_t total = 0;
    std::vector<std::uint32_t> scores(top_k);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const auto mixed = seed + token * 1315423911ULL +
                         static_cast<std::uint64_t>(slot) * 2654435761ULL +
                         static_cast<std::uint64_t>(rank) * 97531ULL;
      scores[slot] = 1U + static_cast<std::uint32_t>(mixed % 65521ULL);
      total += scores[slot];
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      gates[static_cast<std::size_t>(token * top_k + slot)] =
          static_cast<float>(static_cast<double>(scores[slot]) /
                             static_cast<double>(total));
  }
  return gates;
}

std::uint64_t hash_fp32(const std::vector<float> &values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      hash = (hash ^ ((bits >> (byte * 8U)) & 0xffU)) *
             1099511628211ULL;
  }
  return hash;
}

std::uint64_t hash_i32(const std::vector<std::int32_t> &values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto value : values) {
    const auto bits = static_cast<std::uint32_t>(value);
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      hash = (hash ^ ((bits >> (byte * 8U)) & 0xffU)) *
             1099511628211ULL;
  }
  return hash;
}

std::uint64_t hash_bf16_indexed(
    const std::vector<megamoe::transport::BFloat16> &values) {
  std::uint64_t hash = 0;
  for (std::size_t index = 0; index < values.size(); ++index)
    hash += static_cast<std::uint64_t>(values[index].bits) *
            (static_cast<std::uint64_t>(index) + 1ULL);
  return hash;
}

std::uint32_t fixture_mix32_host(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

std::uint16_t fixture_weight_bits(std::uint32_t expert,
                                  std::uint32_t tensor_kind,
                                  std::uint64_t row,
                                  std::uint64_t column) {
  auto key = fixture_mix32_host(expert ^ 0x9e3779b9U);
  key ^= fixture_mix32_host(tensor_kind ^ 0x85ebca6bU);
  key ^= fixture_mix32_host(static_cast<std::uint32_t>(row) ^ 0xc2b2ae35U);
  key ^=
      fixture_mix32_host(static_cast<std::uint32_t>(column) ^ 0x27d4eb2fU);
  const auto signed_value =
      static_cast<int>(fixture_mix32_host(key) & 1U) * 2 - 1;
  return megamoe::transport::float_to_bfloat16(
             static_cast<float>(signed_value) / 256.0F)
      .bits;
}

std::uint64_t forward_weight_sample_hash(std::uint32_t expert_begin,
                                         std::uint32_t expert_count,
                                         std::uint64_t hidden,
                                         std::uint64_t intermediate) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint32_t local = 0; local < expert_count; ++local) {
    const auto expert = expert_begin + local;
    for (std::uint64_t sample = 0; sample < 32ULL; ++sample) {
      const auto row = (sample * 7919ULL + 5ULL) % (2ULL * intermediate);
      const auto column = (sample * 104729ULL + 17ULL) % hidden;
      const auto bits = fixture_weight_bits(expert, 1U, row, column);
      hash = (hash ^ static_cast<std::uint8_t>(bits)) * 1099511628211ULL;
      hash = (hash ^ static_cast<std::uint8_t>(bits >> 8U)) *
             1099511628211ULL;
    }
    for (std::uint64_t sample = 0; sample < 32ULL; ++sample) {
      const auto row = (sample * 65537ULL + 11ULL) % hidden;
      const auto column = (sample * 8191ULL + 3ULL) % intermediate;
      const auto bits = fixture_weight_bits(expert, 2U, row, column);
      hash = (hash ^ static_cast<std::uint8_t>(bits)) * 1099511628211ULL;
      hash = (hash ^ static_cast<std::uint8_t>(bits >> 8U)) *
             1099511628211ULL;
    }
  }
  return hash;
}

struct RunResult {
  std::string metadata;
  megamoe::cuda::CollectiveProtocolAudit audit;
  megamoe::cuda::DeviceMaterializationAudit device_audit;
  megamoe::cuda::NcclWorkspaceUsage workspace;
  megamoe::cuda::GenerationSetupTimings setup;
  megamoe::cuda::GenerationExecutionTimings execution;
  double session_setup_us = 0;
  double pool_setup_us = 0;
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
        auto pool = megamoe::cuda::make_device_notify_resource_pool(
            session, execution, {config, max_tokens, 4096, 2});
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, execution, pool);
        auto result = coordinator.run({config, routes[rank], max_tokens});
        const auto finish = Clock::now();
        results[rank].metadata = megamoe::serialize_dispatch_metadata(
            megamoe::cuda::download_device_metadata_for_validation(
                result.dispatch(), execution));
        results[rank].audit = result.audit();
        results[rank].device_audit = result.dispatch().audit();
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
  megamoe::cuda::DeviceNotifyResourcePool pool;
  megamoe::cuda::CudaRouteInput device_routes;
  megamoe::cuda::CudaGateInput device_gates;
  double session_setup_us = 0;
  double pool_setup_us = 0;
  std::exception_ptr error;
};

std::vector<ReusableRankResources>
make_reusable_resources(const megamoe::RuntimeConfig &config,
                        const std::string &identity,
                        std::uint64_t generation,
                        std::uint64_t max_tokens) {
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
        resources[rank].pool = megamoe::cuda::make_device_notify_resource_pool(
            resources[rank].base_session, resources[rank].execution,
            {config, max_tokens, 4096, 2});
        resources[rank].pool_setup_us =
            resources[rank].pool.usage().one_time_setup_us;
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
        const auto context = megamoe::make_rank_collective_context(
            session.session(), rank, resources[rank].execution.token());
        const auto exact_routes = resources[rank].device_routes
                                      ? megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
                                            resources[rank].device_routes,
                                            context,
                                            resources[rank].execution)
                                      : megamoe::cuda::CudaRouteInput{};
        const auto exact_gates = exact_routes
            ? megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
                  resources[rank].device_gates, exact_routes, context,
                  resources[rank].execution)
            : megamoe::cuda::CudaGateInput{};
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, resources[rank].execution, resources[rank].pool);
        auto result = exact_routes
                          ? coordinator.run_device(
                                {config, exact_routes, exact_gates, max_tokens})
                          : coordinator.run(
                                {config, routes[rank], max_tokens});
        results[rank].metadata = megamoe::serialize_dispatch_metadata(
            megamoe::cuda::download_device_metadata_for_validation(
                result.dispatch(), resources[rank].execution));
        results[rank].audit = result.audit();
        results[rank].device_audit = result.dispatch().audit();
        results[rank].workspace = result.workspace_usage();
        results[rank].setup = result.setup_timings();
        results[rank].execution = result.execution_timings();
        const auto release_start = Clock::now();
        result = {};
        results[rank].execution.pool_acquire_release_us +=
            std::chrono::duration<double, std::micro>(Clock::now() -
                                                      release_start)
                .count();
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
      const auto staging = actual[rank].device_audit;
      require(staging.bulk_route_d2h == 0 &&
                  staging.bulk_collective_d2h == 0 &&
                  staging.host_materializations == 0 &&
                  staging.bulk_metadata_h2d == 0,
              "production generation used a retired host roundtrip");
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
        auto coordinator =
            megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
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

void test_device_capacity_failure_and_recovery() {
  constexpr std::uint32_t world = 2;
  auto bad_config = make_config(1, world, 4, 1, 4);
  bad_config.capacity.expert_rows.assign(4, 100);
  bad_config.capacity.expert_rows[0] = 1;
  auto good_config = bad_config;
  good_config.capacity.expert_rows[0] = 2;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "capacity-exec-" + std::to_string(rank), rank);
        auto base = megamoe::cuda::make_nccl_session(
            "capacity-base", world, rank, 6200, rank, bootstrap);
        auto pool = megamoe::cuda::make_device_notify_resource_pool(
            base, execution, {good_config, 1, 4096, 1, 2});
        auto bad_session = megamoe::cuda::make_nccl_generation(
            base, "capacity-bad", 6201);
        megamoe::cuda::CudaNotifyGenerationCoordinator bad(
            bad_session, execution, pool);
        bool rejected = false;
        try {
          (void)bad.run({bad_config, {1, {0}}, 1});
        } catch (const megamoe::NotifyDispatchError &error) {
          rejected = error.code() ==
                     megamoe::NotifyErrorCode::ExpertCapacityExceeded;
        }
        require(rejected, "device expert capacity did not reject generation");
        const auto bad_audit = bad.audit();
        require(bad_audit.control_starts == 1 &&
                    bad_audit.config_starts == 1 &&
                    bad_audit.expert_count_starts == 1 &&
                    bad_audit.packet_summary_starts == 1,
                "device capacity failure occurred before complete collectives");
        auto good_session = megamoe::cuda::make_nccl_generation(
            base, "capacity-good", 6202);
        megamoe::cuda::CudaNotifyGenerationCoordinator good(
            good_session, execution, pool);
        auto result = good.run({good_config, {1, {0}}, 1});
        require(static_cast<bool>(result),
                "valid exact-capacity recovery did not publish");
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
  std::cout << "[PASS] device capacity failure and exact-capacity recovery\n";
}

#if !MEGAMOE_DEBUG_VALIDATE
void test_release_fused_status_and_config_fallback() {
  constexpr std::uint32_t world = 2;
  const auto config = make_config(1, world, 8, 3, 4);
  const std::vector<megamoe::RankRoutingInput> good_routes(
      world, megamoe::RankRoutingInput{1, {0, 1, 2}});
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> unexpected(world);
  std::array<std::array<std::string, world>, 2> failures;
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "release-contract-exec-" + std::to_string(rank), rank);
        auto base = megamoe::cuda::make_nccl_session(
            "release-contract", world, rank, 6600, rank, bootstrap);
        auto pool = megamoe::cuda::make_device_notify_resource_pool(
            base, execution, {config, 10, 4096, 1, 2});

        const auto run_good = [&](std::uint64_t generation) {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", generation);
          megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
              session, execution, pool);
          auto result = coordinator.run({config, good_routes[rank], 10});
          const auto audit = coordinator.audit();
          require(result && audit.fused_notify_starts == 1U &&
                      audit.control_starts == 0U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "release recovery did not use one fused collective");
          const auto actual = megamoe::serialize_dispatch_metadata(
              megamoe::cuda::download_device_metadata_for_validation(
                  result.dispatch(), execution));
          const auto expected = megamoe::build_notify_dispatch_for_testing(
              config, good_routes, "release-contract", generation);
          require(actual ==
                      megamoe::serialize_dispatch_metadata(expected[rank]),
                  "release recovery metadata/hash differs from oracle");
        };

        // Establish the immutable session config. The prime itself uses the
        // complete protocol; every following base-config generation is fused.
        {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", 6601);
          megamoe::cuda::CudaNotifyGenerationCoordinator prime(
              session, execution, pool);
          auto result = prime.run({config, good_routes[rank], 10});
          const auto audit = prime.audit();
          require(result && audit.fused_notify_starts == 0U &&
                      audit.control_starts == 1U &&
                      audit.config_starts == 1U &&
                      audit.expert_count_starts == 1U &&
                      audit.packet_summary_starts == 1U,
                  "release config prime did not use the complete protocol");
        }

        auto invalid_routes = good_routes[rank];
        if (rank == 1U)
          invalid_routes.selected_experts.back() =
              static_cast<std::int32_t>(config.num_experts);
        {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", 6602);
          megamoe::cuda::CudaNotifyGenerationCoordinator invalid(
              session, execution, pool);
          try {
            (void)invalid.run({config, invalid_routes, 10});
            throw std::runtime_error(
                "fused invalid-route generation unexpectedly passed");
          } catch (const megamoe::NotifyDispatchError &error) {
            failures[0][rank] = error.what();
          }
          const auto audit = invalid.audit();
          require(audit.fused_notify_starts == 1U &&
                      audit.control_starts == 0U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "same-config invalid route crossed the fused boundary");
        }
        run_good(6603);

        auto changed_config = config;
        changed_config.top_k = changed_config.num_experts + 1U;
        {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", 6604);
          megamoe::cuda::CudaNotifyGenerationCoordinator changed(
              session, execution, pool);
          try {
            (void)changed.run({changed_config, good_routes[rank], 10});
            throw std::runtime_error(
                "layout-changing invalid generation unexpectedly passed");
          } catch (const megamoe::NotifyDispatchError &error) {
            failures[1][rank] = error.what();
          }
          const auto audit = changed.audit();
          require(audit.fused_notify_starts == 0U &&
                      audit.control_starts == 1U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "layout-changing config entered the cached fused path");
        }
        run_good(6605);
      } catch (...) {
        unexpected[rank] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &error : unexpected)
    if (error)
      std::rethrow_exception(error);
  for (const auto &failure : failures)
    require(!failure[0].empty() && failure[0] == failure[1],
            "release protocol failure was not identical on every rank");
  std::cout << "[PASS] release fused invalid-status + full config fallback + "
               "same-communicator oracle recoveries\n";
}
#endif

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
            auto bad =
                megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                    bad_session, execution);
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
            auto good =
                megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                    good_session, execution);
            auto result = good.run({config, {1, {0, 1, 2}}, 10});
            require(result.audit().control_starts == 1 &&
                        result.audit().config_starts == 1 &&
                        result.audit().expert_count_starts == 1 &&
                        result.audit().packet_summary_starts == 1,
                    "recovery audit is not 1/1/1/1");
            require(result && !megamoe::serialize_dispatch_metadata(
                                   megamoe::cuda::download_device_metadata_for_validation(
                                       result.dispatch(), execution)).empty(),
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
  const auto require_recovery = [](const auto &recovered, const auto &audit,
                                   const auto &config, const auto &routes,
                                   const auto &execution,
                                   const std::string &identity,
                                   std::uint64_t generation,
                                   const char *failure) {
#if MEGAMOE_DEBUG_VALIDATE
    const bool protocol_ok =
        audit.fused_notify_starts == 0 && audit.control_starts == 1 &&
        audit.config_starts == 1 && audit.expert_count_starts == 1 &&
        audit.packet_summary_starts == 1;
#else
    const bool protocol_ok =
        audit.fused_notify_starts == 1 && audit.control_starts == 0 &&
        audit.config_starts == 0 && audit.expert_count_starts == 0 &&
        audit.packet_summary_starts == 0;
#endif
    require(recovered && protocol_ok, failure);
    const auto actual = megamoe::serialize_dispatch_metadata(
        megamoe::cuda::download_device_metadata_for_validation(
            recovered.dispatch(), execution));
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, {routes}, identity, generation);
    require(actual == megamoe::serialize_dispatch_metadata(expected.front()),
            "fault recovery produced incorrect dispatch metadata/hash");
  };
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
      auto pool = megamoe::cuda::make_device_notify_resource_pool(
          session, execution, {config, 10, 4096, 1, 2});
      // Establish the session-static config on this exact pooled backend.
      // Release must then recover through the fused protocol; debug must
      // continue to execute the complete protocol without this release-only
      // setup prime. Lifecycle assertions below distinguish both modes.
#if !MEGAMOE_DEBUG_VALIDATE
      {
        auto prime_session = megamoe::cuda::make_nccl_generation(
            session, "fault", 3900 + index);
        megamoe::cuda::CudaNotifyGenerationCoordinator prime(
            prime_session, execution, pool);
        require(static_cast<bool>(
                    prime.run({config, {1, {0, 1, 2}}, 10})),
                "failed to prime fault-recovery static config");
      }
#endif
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, pool);
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
      require(failed, std::string("fault generation published validated result at ") +
                          points[index].stage);
      audit = coordinator.audit();
      require(pool.usage().slots_in_use == 0,
              "pooled fault did not return failed generation slot");
      auto recovery_session = megamoe::cuda::make_nccl_generation(
          session, "fault", 5000 + index);
      megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
          recovery_session, execution, pool);
      const megamoe::RankRoutingInput routes{1, {0, 1, 2}};
      auto recovered = recovery.run({config, routes, 10});
      require_recovery(recovered, recovery.audit(), config, routes, execution,
                       "fault", 5000 + index,
                       "pooled fault did not recover with the exact protocol "
                       "on the same communicator");
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
#if MEGAMOE_DEBUG_VALIDATE
    require(lifecycle.validated_publishes == 1,
            "debug fault/recovery published other than recovery");
#else
    require(lifecycle.validated_publishes == 2,
            "release fault/recovery published other than prime + recovery");
#endif
    require(lifecycle.operation_cancels <= 1,
            "coordinator fault double-cancelled");
    require(lifecycle.operation_waits <= 8 &&
                lifecycle.destructor_drains <= 2,
            "fault lifecycle repeated wait/drain");
  }
  const std::vector<FaultExpectation> device_stage_points{
      {Point::CountKernelCompletion, {1, 1, 1, 0},
       "count-kernel-completion"},
      {Point::SummaryKernelCompletion, {1, 1, 1, 1},
       "summary-kernel-completion"},
      {Point::MaterializationKernel, {1, 1, 1, 1},
       "materialization-kernel"}};
  for (std::size_t index = 0; index < device_stage_points.size(); ++index) {
    megamoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto execution = megamoe::cuda::make_cuda_execution(
        "device-stage-fault-exec", 0);
    auto session = megamoe::cuda::make_nccl_session(
        "device-stage-fault", 1, 0, 4050 + index, 0, bootstrap);
    auto pool = megamoe::cuda::make_device_notify_resource_pool(
        session, execution, {config, 10, 4096, 1, 2});
#if !MEGAMOE_DEBUG_VALIDATE
    {
      auto prime_session = megamoe::cuda::make_nccl_generation(
          session, "device-stage-fault", 3950 + index);
      megamoe::cuda::CudaNotifyGenerationCoordinator prime(
          prime_session, execution, pool);
      require(static_cast<bool>(
                  prime.run({config, {1, {0, 1, 2}}, 10})),
              "failed to prime device-stage recovery static config");
    }
#endif
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    megamoe::cuda::testing::inject_next_nccl_fault(
        device_stage_points[index].point);
    bool failed = false;
    try {
      (void)coordinator.run({config, {1, {0, 1, 2}}, 10});
    } catch (const megamoe::NotifyDispatchError &error) {
      failed = std::string(error.what()).find(
                   device_stage_points[index].stage) != std::string::npos;
    }
    require(failed && pool.usage().slots_in_use == 0,
            "pooled device-stage fault leaked/published its slot");
    const auto actual = coordinator.audit();
    const auto expected = device_stage_points[index].audit;
    require(actual.control_starts == expected.control_starts &&
                actual.config_starts == expected.config_starts &&
                actual.expert_count_starts == expected.expert_count_starts &&
                actual.packet_summary_starts ==
                    expected.packet_summary_starts,
            "device-stage fault audit crossed the wrong boundary");
    auto recovery_session = megamoe::cuda::make_nccl_generation(
        session, "device-stage-fault", 4150 + index);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    const megamoe::RankRoutingInput routes{1, {0, 1, 2}};
    auto recovered = recovery.run({config, routes, 10});
    require_recovery(recovered, recovery.audit(), config, routes, execution,
                     "device-stage-fault", 4150 + index,
                     "device-stage fault did not recover with the exact "
                     "protocol on the same communicator");
    const auto lifecycle =
        megamoe::cuda::testing::nccl_lifecycle_snapshot();
#if MEGAMOE_DEBUG_VALIDATE
    require(lifecycle.validated_publishes == 1 &&
                lifecycle.operation_cancels == 0,
            "debug device-stage fault published/cancelled an invalid "
            "generation");
#else
    require(lifecycle.validated_publishes == 2 &&
                lifecycle.operation_cancels == 0,
            "release device-stage fault published other than prime + "
            "recovery or cancelled an invalid generation");
#endif
  }
  for (const bool deadline : {false, true}) {
    megamoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto execution = megamoe::cuda::make_cuda_execution(
        deadline ? "deadline-exec" : "cancel-exec", 0);
    auto session = megamoe::cuda::make_nccl_session(
        deadline ? "deadline" : "cancel", 1, 0,
        deadline ? 4091 : 4090, 0, bootstrap);
    auto pool = megamoe::cuda::make_device_notify_resource_pool(
        session, execution, {config, 10, 4096, 1, 2});
    megamoe::cuda::GenerationRunOptions options;
    if (deadline)
      options.deadline_after = std::chrono::microseconds(1);
    else
      options.cancellation.request_cancel();
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    bool failed = false;
    try {
      (void)coordinator.run({config, {1, {0, 1, 2}}, 10}, options);
    } catch (const megamoe::NotifyDispatchError &error) {
      failed = std::string(error.what()).find("generation-cancel") !=
               std::string::npos;
    }
    require(failed, "production cancellation/deadline path did not fail");
    const auto audit = coordinator.audit();
    require(audit.control_starts == 1 && audit.config_starts == 1 &&
                audit.expert_count_starts == 1 &&
                audit.packet_summary_starts == 0,
            "cancellation audit is not 1/1/1/0");
    const auto lifecycle =
        megamoe::cuda::testing::nccl_lifecycle_snapshot();
    require(lifecycle.operation_cancels == 1 &&
                lifecycle.validated_publishes == 0,
            "cancellation was not executed exactly once or published a handle");
    auto recovery_session = megamoe::cuda::make_nccl_generation(
        session, deadline ? "deadline" : "cancel",
        deadline ? 4191 : 4190);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    const megamoe::RankRoutingInput routes{1, {0, 1, 2}};
    auto recovered = recovery.run({config, routes, 10});
    require_recovery(recovered, recovery.audit(), config, routes, execution,
                     deadline ? "deadline" : "cancel",
                     deadline ? 4191 : 4190,
                     "communicator did not recover with the exact protocol "
                     "after cancellation/deadline");
  }
  std::cout << "[PASS] 9-point NCCL + 3 device-stage pooled fault matrix + "
               "production cancel/deadline exact cancel=1\n";
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
        auto coordinator =
            megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
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
        auto coordinator =
            megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
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
    auto routes = megamoe::cuda::upload_cuda_routes(
        input, config.top_k, context, execution);
    auto plan = megamoe::cuda::query_cuda_notify_workspace(
        config, 0, input.num_tokens, 4);
    auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        plan, context, execution);
    auto gates = make_unit_gates(routes, context, execution);
    auto operation = megamoe::cuda::start_cuda_rank_local_notify(
        config, context, execution, routes, gates, std::move(workspace));
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
  const megamoe::RankRoutingInput handoff_input{1, {0, 2, 4}};
  auto producer_routes = megamoe::cuda::upload_cuda_routes(
      handoff_input, config.top_k, context, execution);
  auto consumer_execution =
      megamoe::cuda::make_cuda_execution("device-only-consumer", 0);
  auto consumer_context = megamoe::make_rank_collective_context(
      session, 0, consumer_execution.token());
  auto consumer_plan = megamoe::cuda::query_cuda_notify_workspace(
      config, 0, handoff_input.num_tokens, 4);
  bool no_wait_failed = false;
  try {
    auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        consumer_plan, consumer_context, consumer_execution);
    auto gates = make_unit_gates(producer_routes, context, execution);
    (void)megamoe::cuda::start_cuda_rank_local_notify(
        config, consumer_context, consumer_execution, producer_routes,
        gates, std::move(workspace));
  } catch (const megamoe::NotifyDispatchError &) {
    no_wait_failed = true;
  }
  require(no_wait_failed,
          "cross-stream route consumption without event handoff succeeded");
  auto imported = megamoe::cuda::import_selected_expert_routes(
      producer_routes, consumer_context, consumer_execution);
  auto imported_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      consumer_plan, consumer_context, consumer_execution);
  auto imported_gates = make_unit_gates(
      imported, consumer_context, consumer_execution);
  auto imported_operation = megamoe::cuda::start_cuda_rank_local_notify(
      config, consumer_context, consumer_execution, imported,
      imported_gates, std::move(imported_workspace));
  require(imported_operation.wait_preflight().valid(),
          "legal cross-stream event handoff failed");
  auto foreign_session =
      megamoe::make_collective_session("foreign-device-only", 1, 5001);
  auto foreign_context = megamoe::make_rank_collective_context(
      foreign_session, 0, consumer_execution.token());
  bool foreign_failed = false;
  try {
    (void)megamoe::cuda::import_selected_expert_routes(
        producer_routes, foreign_context, consumer_execution);
  } catch (const megamoe::NotifyDispatchError &) {
    foreign_failed = true;
  }
  require(foreign_failed,
          "foreign route pipeline session handoff did not fail closed");
  auto alternate_routes = megamoe::cuda::upload_cuda_routes(
      handoff_input, config.top_k, context, execution);
  auto original_gates = make_unit_gates(producer_routes, context, execution);
  bool route_gate_binding_failed = false;
  try {
    megamoe::cuda::DeviceGateProvenanceBridge::validate(
        original_gates, alternate_routes, context, execution);
  } catch (const megamoe::NotifyDispatchError &) {
    route_gate_binding_failed = true;
  }
  require(route_gate_binding_failed,
          "gate lease bound to another route allocation was accepted");
  const auto wide_config = make_config(73, 1, 146, 6, 4);
  auto wide_session =
      megamoe::make_collective_session("device-wide-mask", 73, 5002);
  auto wide_context = megamoe::make_rank_collective_context(
      wide_session, 0, execution.token());
  std::vector<megamoe::RankRoutingInput> wide_inputs(73);
  for (std::uint32_t rank = 0; rank < 73; ++rank) {
    wide_inputs[rank].num_tokens = rank % 5U == 0 ? 0 : 1;
    if (wide_inputs[rank].num_tokens == 0)
      continue;
    for (std::uint32_t slot = 0; slot < wide_config.top_k; ++slot)
      wide_inputs[rank].selected_experts.push_back(
          static_cast<std::int32_t>((rank * 2U + slot * 23U) % 146U));
  }
  wide_inputs[0] = {1, {0, 2, 64, 128, 144, 145}};
  std::vector<std::uint64_t> count_matrix(73U * 146U, 0);
  std::vector<std::uint64_t> summary_matrix(73U * 73U, 0);
  for (std::uint32_t source = 0; source < 73; ++source) {
    std::vector<bool> seen_node(73, false);
    for (const auto selected : wide_inputs[source].selected_experts) {
      const auto expert = static_cast<std::uint32_t>(selected);
      ++count_matrix[static_cast<std::size_t>(source) * 146U + expert];
      const auto destination_rank =
          wide_config.topology.expert_to_rank[expert];
      seen_node[wide_config.topology.rank_to_node[destination_rank]] = true;
    }
    for (std::uint32_t node = 0; node < 73; ++node)
      summary_matrix[static_cast<std::size_t>(source) * 73U + node] =
          seen_node[node] ? wide_inputs[source].num_tokens : 0;
  }
  const auto expected_wide = megamoe::build_notify_dispatch_for_testing(
      wide_config, wide_inputs, "device-wide-mask", 5002);
  std::string wide_baseline;
  megamoe::cuda::DeviceDispatchHandle wide_handle;
  for (std::uint32_t repeat = 0; repeat < 20; ++repeat) {
    wide_handle = megamoe::cuda::testing::materialize_fake_device_collectives(
        wide_config, wide_context, execution, wide_inputs[0], 4,
        count_matrix, summary_matrix);
    const auto downloaded =
        megamoe::cuda::download_device_metadata_for_validation(
            wide_handle, execution);
    const auto serialized = megamoe::serialize_dispatch_metadata(downloaded);
    require(serialized == megamoe::serialize_dispatch_metadata(expected_wide[0]),
            "73-node full device materializer/oracle mismatch");
    if (repeat == 0)
      wide_baseline = serialized;
    else
      require(serialized == wide_baseline,
              "73-node materializer is not byte deterministic");
  }
  auto wide_plan = megamoe::cuda::query_cuda_notify_workspace(
      wide_config, 0, wide_inputs[0].num_tokens, 4);
  require(wide_plan.node_mask_words == 2,
          "73-node workspace did not allocate multiword mask");
  require(wide_handle.sizes().node_mask_words == 2 &&
              wide_handle.sizes().packet_count == 6 &&
              wide_handle.sizes().fanout_count == 6,
          "73-node packet dedup/fan-out shape mismatch");

  // The transport-only persistent worker consumes the immutable device
  // dispatch handle directly. No activation bulk download occurs until the
  // explicit validation call below.
  const megamoe::RankRoutingInput transport_routes{
      3, {5, 1, 7, 1, 2, 3, 0, 7, 4}};
  std::vector<std::uint64_t> transport_counts(8, 0);
  for (const auto selected : transport_routes.selected_experts)
    ++transport_counts[static_cast<std::size_t>(selected)];
  const std::vector<std::uint64_t> transport_summary{3};
  auto transport_dispatch =
      megamoe::cuda::testing::materialize_fake_device_collectives(
          config, context, execution, transport_routes, 4, transport_counts,
          transport_summary);
  const auto host_dispatch = megamoe::build_notify_dispatch_for_testing(
      config, {transport_routes}, "device-only", 5001);
  megamoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = 33;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 2;
  std::vector<megamoe::transport::BFloat16> activations(3U * 33U);
  for (std::size_t i = 0; i < activations.size(); ++i)
    activations[i] = megamoe::transport::float_to_bfloat16(
        static_cast<float>((i * 13U) % 97U) / 16.0F);
  const auto oracle = megamoe::transport::simulate_activation_transport(
      host_dispatch[0], activations,
      std::vector<float>(transport_routes.selected_experts.size(), 1.0F),
      transport_config, 1);
  megamoe::cuda::ActivationTransportPoolConfig transport_pool_config;
  transport_pool_config.max_tokens_per_rank = 4;
  transport_pool_config.max_hidden_size = 33;
  transport_pool_config.max_top_k = 3;
  transport_pool_config.max_world_size = 1;
  transport_pool_config.max_nodes = 1;
  transport_pool_config.max_destination_rows_per_rank = 12;
  transport_pool_config.channels = 1;
  transport_pool_config.ring_slots_per_source_channel = 2;
  transport_pool_config.max_inflight_generations = 2;
  auto transport_pool = megamoe::cuda::make_activation_transport_pool(
      session, execution, transport_pool_config);
  auto activation_lease =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, context, execution, transport_pool);
  megamoe::cuda::DeviceActivationTransport transport(
      session, execution, transport_pool);
  const auto api_before = megamoe::cuda::testing::cuda_api_audit_snapshot();
  auto transport_result = transport.run_device_simulation_synthetic(
      transport_dispatch, activation_lease, transport_config);
  const auto api_after = megamoe::cuda::testing::cuda_api_audit_snapshot();
  require(api_after.device_allocations == api_before.device_allocations &&
              api_after.device_frees == api_before.device_frees &&
              api_after.pinned_allocations == api_before.pinned_allocations &&
              api_after.pinned_frees == api_before.pinned_frees &&
              api_after.event_creates == api_before.event_creates &&
              api_after.event_destroys == api_before.event_destroys,
          "steady activation transport allocated CUDA resources");
  const auto downloaded_transport =
      megamoe::cuda::download_activation_transport_for_validation(
          transport_result, execution);
  require(downloaded_transport.contribution_by_route ==
              oracle.contribution_by_route &&
              downloaded_transport.combined_tokens == oracle.combined_tokens,
          "device transport synthetic contribution/combine oracle mismatch");
  require(std::equal(oracle.canonical_expert_rows.begin(),
                     oracle.canonical_expert_rows.end(),
                     downloaded_transport.canonical_expert_rows.begin()),
          "device transport canonical expert rows mismatch");
  require(downloaded_transport.protocol.dispatch_packets == 3 &&
              downloaded_transport.protocol.relayed_fanout == 9 &&
              downloaded_transport.protocol.reduced_tokens == 3,
          "device transport phase counters mismatch");
  require(transport_result.audit().backend ==
              megamoe::transport::ActivationTransportBackend::DeviceSimulation,
          "simulation backend was mislabeled as P2P");
  const auto transport_baseline = downloaded_transport.combined_tokens;
  transport_result = {};
  activation_lease = {};
  for (std::uint32_t repeat = 0; repeat < 20; ++repeat) {
    auto repeated_activation =
        megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, 3, 33, context, execution, transport_pool);
    auto repeated_result = transport.run_device_simulation_synthetic(
        transport_dispatch, repeated_activation, transport_config);
    const auto repeated =
        megamoe::cuda::download_activation_transport_for_validation(
            repeated_result, execution);
    require(repeated.combined_tokens == transport_baseline,
            "ring epoch reuse changed device transport bytes");
  }
  require(transport_pool.usage().slots_available == 2 &&
              transport_pool.usage().slots_in_use == 0,
          "transport slots were not returned after final lease destruction");

  auto next_transport_session =
      megamoe::make_collective_session("device-only", 1, 5002);
  auto next_transport_context = megamoe::make_rank_collective_context(
      next_transport_session, 0, execution.token());
  auto wrong_generation_activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, next_transport_context, execution,
          transport_pool);
  bool generation_failed = false;
  try {
    (void)transport.run_device_simulation_synthetic(transport_dispatch,
                                  wrong_generation_activation,
                                  transport_config);
  } catch (const megamoe::NotifyDispatchError &error) {
    generation_failed = error.code() == megamoe::NotifyErrorCode::WorkspaceMismatch;
  }
  require(generation_failed,
          "cross-generation activation lease was accepted");
  wrong_generation_activation = {};

  auto exhausted_config = transport_pool_config;
  exhausted_config.max_inflight_generations = 1;
  auto exhausted_pool = megamoe::cuda::make_activation_transport_pool(
      session, execution, exhausted_config);
  auto held_activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, context, execution, exhausted_pool);
  bool exhausted = false;
  try {
    (void)megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
        activations, 3, 33, context, execution, exhausted_pool);
  } catch (const megamoe::NotifyDispatchError &error) {
    exhausted = error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
  }
  require(exhausted && exhausted_pool.usage().slots_in_use == 1,
          "transport pool exhaustion did not fail closed");
  held_activation = {};
  require(exhausted_pool.usage().slots_available == 1,
          "transport pool did not recover after final input lease");

  megamoe::transport::ActivationTransportConfig wide_transport_config;
  wide_transport_config.hidden_size = 7;
  wide_transport_config.channels = 1;
  wide_transport_config.ring_slots_per_source_channel = 1;
  megamoe::cuda::ActivationTransportPoolConfig wide_transport_pool_config;
  wide_transport_pool_config.max_tokens_per_rank = 4;
  wide_transport_pool_config.max_hidden_size = 7;
  wide_transport_pool_config.max_top_k = 6;
  wide_transport_pool_config.max_world_size = 73;
  wide_transport_pool_config.max_nodes = 73;
  wide_transport_pool_config.max_destination_rows_per_rank = 32;
  wide_transport_pool_config.channels = 1;
  wide_transport_pool_config.ring_slots_per_source_channel = 1;
  wide_transport_pool_config.max_inflight_generations = 1;
  auto wide_transport_pool = megamoe::cuda::make_activation_transport_pool(
      wide_session, execution, wide_transport_pool_config);
  std::vector<megamoe::transport::BFloat16> wide_activations(7);
  for (std::size_t i = 0; i < wide_activations.size(); ++i)
    wide_activations[i] = megamoe::transport::float_to_bfloat16(
        static_cast<float>(i + 1U) / 8.0F);
  auto wide_activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          wide_activations, 1, 7, wide_context, execution,
          wide_transport_pool);
  megamoe::cuda::DeviceActivationTransport wide_transport(
      wide_session, execution, wide_transport_pool);
  auto wide_transport_result = wide_transport.run_device_simulation_synthetic(
      wide_handle, wide_activation, wide_transport_config);
  auto gate_before = wide_transport_config;
  gate_before.gate_placement =
      megamoe::transport::ActivationGatePlacement::DestinationBeforeUnsupported;
  bool gate_before_failed = false;
  try {
    (void)wide_transport.run_device_simulation_synthetic(
        wide_handle, wide_activation, gate_before);
  } catch (const megamoe::NotifyDispatchError &) {
    gate_before_failed = true;
  }
  require(gate_before_failed,
          "unsupported gate-before placement was silently accepted");
  const auto wide_transport_download =
      megamoe::cuda::download_activation_transport_for_validation(
          wide_transport_result, execution);
  const auto wide_transport_oracle =
      megamoe::transport::simulate_activation_transport(
          expected_wide[0], wide_activations,
          std::vector<float>(
              expected_wide[0]->outgoing.expert_id.size(), 1.0F),
          wide_transport_config, 1);
  require(wide_transport_download.contribution_by_route ==
              wide_transport_oracle.contribution_by_route &&
              wide_transport_download.combined_tokens ==
                  wide_transport_oracle.combined_tokens &&
              wide_transport_download.protocol.dispatch_packets == 6,
          "73-node full device transport/oracle mismatch");
  const auto capability =
      megamoe::cuda::probe_activation_transport_capability(2);
  require(capability.nccl_device_headers &&
              capability.gin_headers_available &&
              capability.gin_backend_implemented && !capability.gin_usable,
          "GIN header/runtime capability boundary was not detected");
  require(megamoe::cuda::testing::run_local_p2p_put_signal_probe(0, 1, 4096),
          "real CUDA P2P put+system-signal probe failed");
  std::cout << "[PASS] transport-only persistent device worker, deterministic "
               "synthetic combine, pooled steady path, P2P put/signal, GIN "
               "fail-closed\n";
  auto short_plan = wide_plan;
  --short_plan.device_bytes;
  bool short_failed = false;
  try {
    (void)megamoe::cuda::allocate_cuda_notify_workspace(
        short_plan, wide_context, execution);
  } catch (const megamoe::NotifyDispatchError &) {
    short_failed = true;
  }
  require(short_failed, "workspace capacity minus one byte did not fail");
  std::cout << "[PASS] full fake device materializer including 73-node "
               "multiword descriptors, 20 byte-identical repeats\n";
}

void test_delayed_route_producer_readiness() {
  const auto config = make_config(1, 1, 8, 1, 4);
  auto producer = megamoe::cuda::make_cuda_execution("delay-producer", 0);
  auto consumer = megamoe::cuda::make_cuda_execution("delay-consumer", 0);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto base = megamoe::cuda::make_nccl_session(
      "delay-session", 1, 0, 6900, 0, bootstrap);
  const auto producer_context = megamoe::make_rank_collective_context(
      base.session(), 0, producer.token());
  auto storage = megamoe::cuda::upload_cuda_routes(
      {1, {0}}, 1, producer_context, producer);
  auto delayed = megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
      storage, 5, 5000000, producer_context, producer);
  const auto consumer_context = megamoe::make_rank_collective_context(
      base.session(), 0, consumer.token());
  auto imported = megamoe::cuda::import_selected_expert_routes(
      delayed, consumer_context, consumer);
  auto pool = megamoe::cuda::make_device_notify_resource_pool(
      base, consumer, {config, 2, 4096, 1, 2});
  megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
      base, consumer, pool);
  auto imported_gates = make_unit_gates(imported, consumer_context, consumer);
  auto result = coordinator.run_device(
      {config, imported, imported_gates, 2});
  const auto downloaded =
      megamoe::cuda::download_device_metadata_for_validation(
          result.dispatch(), consumer);
  require(downloaded->outgoing.expert_id == std::vector<std::uint32_t>{5},
          "cross-stream Notify observed poison instead of ready-event result");

  auto next = megamoe::cuda::make_nccl_generation(
      base, "delay-session", 6901);
  const auto next_producer_context = megamoe::make_rank_collective_context(
      next.session(), 0, producer.token());
  auto delayed_without_ready =
      megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
          storage, 4, 100000000, next_producer_context, producer, false);
  const auto next_consumer_context = megamoe::make_rank_collective_context(
      next.session(), 0, consumer.token());
  bool missing_ready_failed = false;
  try {
    (void)megamoe::cuda::import_selected_expert_routes(
        delayed_without_ready, next_consumer_context, consumer);
  } catch (const megamoe::NotifyDispatchError &) {
    missing_ready_failed = true;
  }
  require(missing_ready_failed,
          "unrecorded producer-ready event was accepted");

  auto unsafe_delayed =
      megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
          storage, 4, 100000000, next_producer_context, producer);
  auto unsafe =
      megamoe::cuda::DeviceRouteProvenanceBridge::unsafe_import_without_wait_for_testing(
          unsafe_delayed, next_consumer_context, consumer);
  bool unsafe_failed = false;
  try {
    megamoe::cuda::CudaNotifyGenerationCoordinator unsafe_coordinator(
        next, consumer, pool);
    auto unsafe_gates = make_unit_gates(unsafe, next_consumer_context,
                                        consumer);
    (void)unsafe_coordinator.run_device(
        {config, unsafe, unsafe_gates, 2});
  } catch (const megamoe::NotifyDispatchError &) {
    unsafe_failed = true;
  }
  require(unsafe_failed,
          "negative no-wait seam did not demonstrate producer race");

  auto same_session = megamoe::cuda::make_nccl_session(
      "delay-same-stream", 1, 0, 6920, 0,
      megamoe::cuda::make_nccl_bootstrap_id());
  const auto same_context = megamoe::make_rank_collective_context(
      same_session.session(), 0, producer.token());
  auto same_storage = megamoe::cuda::upload_cuda_routes(
      {1, {0}}, 1, same_context, producer);
  auto same_delayed = megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
      same_storage, 6, 5000000, same_context, producer);
  auto same_pool = megamoe::cuda::make_device_notify_resource_pool(
      same_session, producer, {config, 2, 4096, 1, 2});
  megamoe::cuda::CudaNotifyGenerationCoordinator same_coordinator(
      same_session, producer, same_pool);
  auto same_gates = make_unit_gates(same_delayed, same_context, producer);
  auto same_result = same_coordinator.run_device(
      {config, same_delayed, same_gates, 2});
  const auto same_download =
      megamoe::cuda::download_device_metadata_for_validation(
          same_result.dispatch(), producer);
  require(same_download->outgoing.expert_id ==
              std::vector<std::uint32_t>{6},
          "same-stream producer order was not preserved");
  std::cout << "[PASS] delayed producer ready-event cross/same-stream + "
               "no-wait negative seam\n";
}

void test_device_dispatch_handle_lifetime() {
  const auto config = make_config(1, 1, 8, 3, 4);
  const megamoe::RankRoutingInput routes{2, {7, 1, 4, 0, 6, 2}};
  auto execution = megamoe::cuda::make_cuda_execution("handle-exec", 0);
  megamoe::cuda::DeviceDispatchHandle retained;
  {
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto session = megamoe::cuda::make_nccl_session(
        "handle-lifetime", 1, 0, 6100, 0, bootstrap);
    auto coordinator =
        megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
            session, execution);
    auto result = coordinator.run({config, routes, 10});
    auto first_copy = result.dispatch();
    retained = first_copy;
    auto moved = std::move(first_copy);
    require(!first_copy && moved, "device handle move state is unsafe");
    const auto &identity = moved.identity();
    const auto expectation = megamoe::cuda::DeviceDispatchViewExpectation{
        6100, 0, 1, routes.num_tokens, config.top_k,
        identity.config_digest_word0, identity.config_digest_word1};
    auto consumer = moved.acquire_consumer(execution, expectation);
    require(consumer && consumer.state() ==
                            megamoe::cuda::DeviceDispatchConsumerState::Acquired,
            "owning device consumer was not acquired");
    consumer.finish();
  }
  require(static_cast<bool>(retained),
          "device handle did not outlive coordinator/result");
  const auto actual = megamoe::cuda::download_device_metadata_for_validation(
      retained, execution);
  const auto expected = megamoe::build_notify_dispatch_for_testing(
      config, {routes}, "handle-lifetime", 6100);
  require(megamoe::serialize_dispatch_metadata(actual) ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "retained device handle metadata changed after owner destruction");
  bool mismatch_failed = false;
  try {
    (void)retained.acquire_consumer(
        execution,
        {6101, 0, 1, routes.num_tokens, config.top_k,
         retained.identity().config_digest_word0,
         retained.identity().config_digest_word1});
  } catch (const megamoe::NotifyDispatchError &) {
    mismatch_failed = true;
  }
  require(mismatch_failed,
          "cross-generation device view mismatch did not fail closed");
  auto second_execution =
      megamoe::cuda::make_cuda_execution("handle-consumer-stream-2", 0);
  const auto retained_id = retained.identity();
  const megamoe::cuda::DeviceDispatchViewExpectation retained_expectation{
      retained_id.generation, retained_id.calling_rank, retained_id.world_size,
      retained_id.num_tokens, retained_id.top_k,
      retained_id.config_digest_word0, retained_id.config_digest_word1};
  auto first_stream_consumer =
      retained.acquire_consumer(execution, retained_expectation);
  auto second_stream_consumer =
      retained.acquire_consumer(second_execution, retained_expectation);
  megamoe::cuda::testing::submit_dispatch_probe(first_stream_consumer, 10000);
  megamoe::cuda::testing::submit_dispatch_probe(second_stream_consumer, 10000);
  retained = {};
  (void)megamoe::cuda::testing::finish_dispatch_probe(first_stream_consumer);
  (void)megamoe::cuda::testing::finish_dispatch_probe(second_stream_consumer);
  megamoe::cuda::DeviceDispatchHandle empty;
  bool empty_failed = false;
  try {
    (void)empty.sizes();
  } catch (const megamoe::NotifyDispatchError &) {
    empty_failed = true;
  }
  require(empty_failed, "empty device handle accessor did not fail closed");
  std::cout << "[PASS] immutable device handle copy/move/lifetime/consumer checks\n";
}

void test_resource_pool_and_cuda_api_audit() {
  const auto config = make_config(1, 1, 16, 4, 7);
  const auto routes = seeded_inputs(1, 16, 4, 0xabcddcbaULL).front();
  auto execution = megamoe::cuda::make_cuda_execution("pool-exec", 0);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto base = megamoe::cuda::make_nccl_session(
      "pool-base", 1, 0, 6200, 0, bootstrap);
  auto pool = megamoe::cuda::make_device_notify_resource_pool(
      base, execution,
      {config, 10, 4096, 2});
  const auto initial_usage = pool.usage();
  require(initial_usage.slot_count == 2 && initial_usage.slots_in_use == 0 &&
              initial_usage.device_bytes != 0 &&
              initial_usage.pinned_host_bytes != 0 &&
              initial_usage.event_count == 30,
          "session resource pool capacity/accounting mismatch");

  auto device_routes = megamoe::cuda::upload_cuda_routes(
      routes, config.top_k,
      megamoe::make_rank_collective_context(base.session(), 0,
                                            execution.token()),
      execution);
  auto device_gates = make_unit_gates(
      device_routes,
      megamoe::make_rank_collective_context(base.session(), 0,
                                            execution.token()),
      execution);
  const auto exact_routes = [&](const megamoe::cuda::NcclSession &session) {
    return megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
        device_routes,
        megamoe::make_rank_collective_context(session.session(), 0,
                                              execution.token()),
        execution);
  };
  const auto exact_request = [&](const megamoe::cuda::NcclSession &session) {
    auto routes_for_generation = exact_routes(session);
    auto gates_for_generation =
        megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
            device_gates, routes_for_generation,
            megamoe::make_rank_collective_context(
                session.session(), 0, execution.token()),
            execution);
    return megamoe::cuda::DeviceNotifyGenerationRequest{
        config, std::move(routes_for_generation),
        std::move(gates_for_generation), 10};
  };
  {
    auto next = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6201);
    auto next_routes = exact_routes(next);
    bool stale_gate_failed = false;
    try {
      megamoe::cuda::DeviceGateProvenanceBridge::validate(
          device_gates, next_routes,
          megamoe::make_rank_collective_context(
              next.session(), 0, execution.token()),
          execution);
    } catch (const megamoe::NotifyDispatchError &) {
      stale_gate_failed = true;
    }
    require(stale_gate_failed,
            "generation-6200 gate lease was accepted at generation 6201");
  }
  for (std::uint64_t generation = 6201; generation <= 6210; ++generation) {
    bool rejected = false;
    try {
      auto mismatched = megamoe::cuda::make_nccl_generation(
          base, "pool-base", generation);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          mismatched, execution, pool);
      (void)coordinator.run_device(
          {config, device_routes, device_gates, 10});
    } catch (const megamoe::NotifyDispatchError &error) {
      const std::string message = error.what();
      rejected = message.find("field=generation") != std::string::npos &&
                 message.find("expected=" + std::to_string(generation)) !=
                     std::string::npos &&
                 message.find("actual=6200") != std::string::npos;
    }
    require(rejected,
            "generation-6200 route lease was accepted by a later generation");
    const auto rebound = exact_routes(megamoe::cuda::make_nccl_generation(
        base, "pool-base", generation));
    require(rebound.route_generation() == generation &&
                device_routes.route_generation() == 6200,
            "private exact-generation reissue mutated or forged provenance");
  }
  const auto before = megamoe::cuda::testing::cuda_api_audit_snapshot();
  megamoe::cuda::DeviceDispatchHandle retained;
  for (std::uint64_t iteration = 0; iteration < 10; ++iteration) {
    auto session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6201 + iteration);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    auto result = coordinator.run_device(exact_request(session));
    require(result.dispatch().audit().bulk_route_d2h == 0 &&
                result.dispatch().audit().bulk_collective_d2h == 0 &&
                result.dispatch().audit().host_materializations == 0 &&
                result.dispatch().audit().bulk_metadata_h2d == 0,
            "pooled device generation used bulk host staging");
    if (iteration >= 2U) {
      const auto audit = result.audit();
#if MEGAMOE_DEBUG_VALIDATE
      require(audit.fused_notify_starts == 0U &&
                  audit.control_starts == 1U && audit.config_starts == 1U &&
                  audit.expert_count_starts == 1U &&
                  audit.packet_summary_starts == 1U,
              "debug generation did not retain the full Notify protocol");
#else
      require(audit.fused_notify_starts == 1U &&
                  audit.control_starts == 0U && audit.config_starts == 0U &&
                  audit.expert_count_starts == 0U &&
                  audit.packet_summary_starts == 0U,
              "steady single-node release generation did not use one fused "
              "Notify collective");
#endif
    }
    if (iteration == 0)
      retained = result.dispatch();
  }
  const auto after = megamoe::cuda::testing::cuda_api_audit_snapshot();
  require(after.device_allocations == before.device_allocations &&
              after.device_frees == before.device_frees &&
              after.pinned_allocations == before.pinned_allocations &&
              after.pinned_frees == before.pinned_frees &&
              after.event_creates == before.event_creates &&
              after.event_destroys == before.event_destroys &&
              after.device_synchronizes == before.device_synchronizes,
          "steady generation performed forbidden CUDA resource API call");
  require(after.h2d_bytes - before.h2d_bytes < 1024U * 1024U &&
              after.d2h_bytes - before.d2h_bytes < 1024U * 1024U &&
              after.d2d_bytes - before.d2d_bytes < 1024U * 1024U,
          "steady generation performed an unexpected bulk memcpy");
  require(pool.usage().slots_in_use == 1,
          "retained handle did not keep exactly one pool slot leased");

#if !MEGAMOE_DEBUG_VALIDATE
  const auto require_base_recovery = [&](std::uint64_t generation,
                                         const char *failure) {
    auto recovery_session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", generation);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    auto recovered = recovery.run_device(exact_request(recovery_session));
    const auto audit = recovery.audit();
    require(recovered && audit.fused_notify_starts == 1U &&
                audit.control_starts == 0U && audit.config_starts == 0U &&
                audit.expert_count_starts == 0U &&
                audit.packet_summary_starts == 0U,
            failure);
    const auto actual = megamoe::serialize_dispatch_metadata(
        megamoe::cuda::download_device_metadata_for_validation(
            recovered.dispatch(), execution));
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, {routes}, "pool-base", generation);
    require(actual == megamoe::serialize_dispatch_metadata(expected.front()),
            "release protocol recovery metadata/hash mismatch");
  };

  // A predictable route error does not change the cached static layout.  It
  // must therefore be dataized in the fixed-width fused record instead of
  // returning locally before the collective.
  auto invalid_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6299);
  megamoe::cuda::CudaNotifyGenerationCoordinator invalid_coordinator(
      invalid_session, execution, pool);
  auto invalid_routes = routes;
  invalid_routes.selected_experts.front() =
      static_cast<std::int32_t>(config.num_experts);
  bool fused_status_failed = false;
  try {
    (void)invalid_coordinator.run({config, invalid_routes, 10});
  } catch (const megamoe::NotifyDispatchError &error) {
    fused_status_failed =
        error.code() == megamoe::NotifyErrorCode::ControlProtocolFailure;
  }
  const auto invalid_audit = invalid_coordinator.audit();
  require(fused_status_failed && invalid_audit.fused_notify_starts == 1U &&
              invalid_audit.control_starts == 0U &&
              invalid_audit.config_starts == 0U &&
              invalid_audit.expert_count_starts == 0U &&
              invalid_audit.packet_summary_starts == 0U,
          "invalid local status did not fail after exactly one fused Notify "
          "collective");
  require(pool.usage().slots_in_use == 1,
          "failed fused Notify generation did not return its pool slot");
  require_base_recovery(
      6300, "same-communicator recovery after fused route failure did not "
            "use the cached release protocol");

  // A request that changes a layout-relevant field cannot use a record laid
  // out from the cached config.  It must enter the complete protocol and fail
  // at control validation; forcing it through fused would interpret the old
  // layout.  Returning to the cached config on the same communicator must
  // recover through fused and reproduce the oracle bytes.
  auto changed_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6301);
  megamoe::cuda::CudaNotifyGenerationCoordinator changed_coordinator(
      changed_session, execution, pool);
  auto changed_config = config;
  changed_config.top_k = changed_config.num_experts + 1U;
  bool changed_config_failed = false;
  try {
    (void)changed_coordinator.run({changed_config, routes, 10});
  } catch (const megamoe::NotifyDispatchError &error) {
    changed_config_failed =
        error.code() == megamoe::NotifyErrorCode::ControlProtocolFailure;
  }
  const auto changed_audit = changed_coordinator.audit();
  require(changed_config_failed && changed_audit.fused_notify_starts == 0U &&
              changed_audit.control_starts == 1U &&
              changed_audit.config_starts == 0U &&
              changed_audit.expert_count_starts == 0U &&
              changed_audit.packet_summary_starts == 0U,
          "layout-changing invalid config did not fail at the complete "
          "protocol control boundary");
  require(pool.usage().slots_in_use == 1,
          "failed full-protocol config generation did not return its slot");
  require_base_recovery(
      6302, "same-communicator recovery after config fallback did not return "
            "to the cached release protocol");
#endif

  auto second_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6310);
  megamoe::cuda::CudaNotifyGenerationCoordinator second_coordinator(
      second_session, execution, pool);
  auto second = second_coordinator.run_device(exact_request(second_session));
  require(pool.usage().slots_in_use == 2,
          "two live handles did not occupy two exclusive pool slots");
  bool exhausted = false;
  try {
    auto third_session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6311);
    megamoe::cuda::CudaNotifyGenerationCoordinator third_coordinator(
        third_session, execution, pool);
    (void)third_coordinator.run_device(exact_request(third_session));
  } catch (const megamoe::NotifyDispatchError &error) {
    exhausted = error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
  }
  require(exhausted, "pool exhaustion did not fail with ResourceExhausted");
  second = {};
  require(pool.usage().slots_in_use == 1,
          "last handle destruction did not return its slot");
  megamoe::cuda::testing::inject_next_nccl_fault(
      megamoe::cuda::testing::NcclFaultPoint::ControlStart);
  bool injected_failed = false;
  try {
    auto failed_session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6312);
    megamoe::cuda::CudaNotifyGenerationCoordinator failed_coordinator(
        failed_session, execution, pool);
    (void)failed_coordinator.run_device(exact_request(failed_session));
  } catch (const megamoe::NotifyDispatchError &) {
    injected_failed = true;
  }
  require(injected_failed && pool.usage().slots_in_use == 1,
          "failed generation leaked or published a pool slot");
  auto recovery_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6313);
  megamoe::cuda::CudaNotifyGenerationCoordinator recovery_coordinator(
      recovery_session, execution, pool);
  auto recovered = recovery_coordinator.run_device(
      exact_request(recovery_session));
  require(recovered && pool.usage().slots_in_use == 2,
          "released pool slot was not reusable");

  const auto actual = megamoe::cuda::download_device_metadata_for_validation(
      recovered.dispatch(), execution);
  const auto expected = megamoe::build_notify_dispatch_for_testing(
      config, {routes}, "pool-base", 6313);
  require(megamoe::serialize_dispatch_metadata(actual) ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "pooled device input metadata mismatch");
  auto ownerless_handle = recovered.dispatch();
  recovered = {};
  retained = {};
  pool = {};
  require(megamoe::serialize_dispatch_metadata(
              megamoe::cuda::download_device_metadata_for_validation(
                  ownerless_handle, execution)) ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "device handle did not safely outlive the public pool owner");
  ownerless_handle = {};

  for (const auto slot_count : {1U, 3U}) {
    auto sized_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, slot_count});
    std::vector<megamoe::cuda::ValidatedCudaGeneration> live;
    for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
      const auto generation = 6400U + slot_count * 10U + slot;
      auto session = megamoe::cuda::make_nccl_generation(
          base, "pool-base", generation);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      live.push_back(coordinator.run_device(exact_request(session)));
    }
    require(sized_pool.usage().slots_in_use == slot_count,
            "pool size 1/N did not retain every live slot");
    bool sized_exhausted = false;
    try {
      auto session = megamoe::cuda::make_nccl_generation(
          base, "pool-base",
          6500U + slot_count);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      (void)coordinator.run_device(exact_request(session));
    } catch (const megamoe::NotifyDispatchError &error) {
      sized_exhausted =
          error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(sized_exhausted,
            "pool size 1/N did not reject one generation beyond capacity");
    live.clear();
    require(sized_pool.usage().slots_in_use == 0,
            "pool size 1/N did not release all final handle leases");
    {
      auto session = megamoe::cuda::make_nccl_generation(
          base, "pool-base",
          6600U + slot_count);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      auto reused = coordinator.run_device(exact_request(session));
      require(static_cast<bool>(reused),
              "pool size 1/N released slot was not reusable");
    }
    require(sized_pool.usage().slots_in_use == 0,
            "pool size 1/N reuse leaked a lease");
  }

  {
    auto stale_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 1});
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6700);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, stale_pool);
    auto gen1 = coordinator.run_device(exact_request(generation));
    auto handle = gen1.dispatch();
    const auto &id = handle.identity();
    auto consumer = handle.acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    bool consumer_exhausted = false;
    try {
      (void)handle.acquire_consumer(
          execution, {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, id.top_k, id.config_digest_word0,
                      id.config_digest_word1});
    } catch (const megamoe::NotifyDispatchError &error) {
      consumer_exhausted =
          error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(consumer_exhausted,
            "consumer event pool exhaustion did not fail closed");
    auto moved_consumer = std::move(consumer);
    require(!consumer && moved_consumer,
            "consumer lease move/moved-from state is unsafe");
    moved_consumer = std::move(moved_consumer);
    require(moved_consumer &&
                moved_consumer.state() ==
                    megamoe::cuda::DeviceDispatchConsumerState::Acquired,
            "consumer self-move corrupted the owning lease");
    megamoe::cuda::testing::submit_dispatch_probe(moved_consumer, 1000000);
    require(moved_consumer.state() ==
                megamoe::cuda::DeviceDispatchConsumerState::Submitted,
            "consumer probe was not submitted");
    handle = {};
    gen1 = {};
    require(stale_pool.usage().slots_in_use == 1,
            "submitted consumer did not retain generation slot");
    bool stale_exhausted = false;
    try {
      auto blocked_generation = megamoe::cuda::make_nccl_generation(
          base, "pool-base", 6701);
      megamoe::cuda::CudaNotifyGenerationCoordinator blocked(
          blocked_generation, execution, stale_pool);
      (void)blocked.run_device(exact_request(blocked_generation));
    } catch (const megamoe::NotifyDispatchError &error) {
      stale_exhausted =
          error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(stale_exhausted,
            "pool reused slot while stale consumer remained live");
    const auto stale_value =
        megamoe::cuda::testing::finish_dispatch_probe(moved_consumer);
    require(stale_value == expected.front()->outgoing.route_dst_row.front(),
            "old consumer did not read generation-1 canonical row");
    require(stale_pool.usage().slots_in_use == 0,
            "finished consumer did not release final slot lease");
    auto recovered_generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6702);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovered_coordinator(
        recovered_generation, execution, stale_pool);
    auto gen2 = recovered_coordinator.run_device(
        exact_request(recovered_generation));
    require(gen2 && stale_pool.usage().slots_in_use == 1,
            "slot was not reusable after final consumer completion");
  }


  {
    auto quarantine_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 2});
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6800);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, quarantine_pool);
    auto result = coordinator.run_device(exact_request(generation));
    auto handle = result.dispatch();
    const auto &id = handle.identity();
    auto consumer = handle.acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    megamoe::cuda::testing::submit_dispatch_probe(consumer, 1000);
    megamoe::cuda::testing::inject_next_consumer_fault(
        megamoe::cuda::testing::ConsumerFaultPoint::EventAndStreamSynchronize);
    bool finish_failed = false;
    try {
      consumer.finish();
    } catch (const megamoe::NotifyDispatchError &) {
      finish_failed = true;
    }
    require(finish_failed, "consumer event+stream failure was not surfaced");
    const auto quarantined = quarantine_pool.usage();
    require(quarantined.slots_quarantined == 1 &&
                quarantined.consumer_events_quarantined == 1 &&
                quarantined.consumer_events_available == 1 &&
                quarantined.consumer_events_in_use == 0 &&
                quarantined.slots_available == 0 &&
                quarantined.slots_in_use == 0,
            "consumer failure did not independently quarantine event and slot");
    bool handle_failed = false;
    try {
      (void)handle.acquire_consumer(
          execution, {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, id.top_k, id.config_digest_word0,
                      id.config_digest_word1});
    } catch (const megamoe::NotifyDispatchError &) {
      handle_failed = true;
    }
    require(handle_failed,
            "quarantined generation handle accepted another consumer");
    bool next_failed = false;
    try {
      auto next = megamoe::cuda::make_nccl_generation(
          base, "pool-base", 6801);
      megamoe::cuda::CudaNotifyGenerationCoordinator blocked(
          next, execution, quarantine_pool);
      (void)blocked.run_device(exact_request(next));
    } catch (const megamoe::NotifyDispatchError &error) {
      next_failed = error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(next_failed,
            "quarantined slot was allocated to a later generation");
  }

  for (const auto fault : {
           megamoe::cuda::testing::ConsumerFaultPoint::EventRecord,
           megamoe::cuda::testing::ConsumerFaultPoint::EventSynchronize}) {
    auto event_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 2});
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base",
        fault == megamoe::cuda::testing::ConsumerFaultPoint::EventRecord
            ? 6810
            : 6811);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, event_pool);
    auto result = coordinator.run_device(exact_request(generation));
    const auto &id = result.dispatch().identity();
    auto consumer = result.dispatch().acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    megamoe::cuda::testing::inject_next_consumer_fault(fault);
    bool failed = false;
    try {
      megamoe::cuda::testing::submit_dispatch_probe(consumer, 1000);
      consumer.finish();
    } catch (const megamoe::NotifyDispatchError &) {
      failed = true;
    }
    require(failed, "consumer event-only fault was not surfaced");
    const auto usage = event_pool.usage();
    require(usage.slots_quarantined == 0 &&
                usage.slots_in_use == 1 &&
                usage.consumer_events_quarantined == 1 &&
                usage.consumer_events_available == 1 &&
                usage.consumer_events_in_use == 0,
            "event-only failure incorrectly recycled event or quarantined slot");
  }

  std::cout << "[PASS] exact route generation 6200->6201..6210 rejection + "
               "resource pool size=1/2/3 lease/exhaustion/reuse/quarantine + "
               "10-generation "
               "CUDA API audit alloc/free/pinned/event/device-sync=0/0/0/0/0, "
               "H2D_bytes="
            << after.h2d_bytes - before.h2d_bytes << ", D2H_bytes="
            << after.d2h_bytes - before.d2h_bytes << ", D2D_bytes="
            << after.d2d_bytes - before.d2d_bytes << '\n';
}

#include "cuda/distributed_runner.inc"
#include "cuda/transport_benchmarks.inc"

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

void test_randomized_generation_scheduler() {
  using megamoe::cuda::testing::GenerationSchedulerModel;
  using megamoe::cuda::transport_state::TaskClass;
  constexpr auto kinds = megamoe::cuda::transport_state::kTaskClassCount;
  for (std::uint64_t seed = 1; seed <= 1000; ++seed) {
    auto random = seed * 0x9e3779b97f4a7c15ULL;
    auto next = [&random]() {
      random ^= random >> 12U;
      random ^= random << 25U;
      random ^= random >> 27U;
      return random * 0x2545f4914f6cdd1dULL;
    };
    std::array<std::uint32_t, kinds> expected{};
    for (auto &count : expected)
      count = static_cast<std::uint32_t>(next() % 17U);
    const auto contexts = 1U + static_cast<std::uint32_t>(next() % 8U);
    GenerationSchedulerModel model(expected, contexts, 3U);
    for (std::uint32_t context = 0; context < contexts; ++context)
      model.publish_directory(context);

    std::array<std::uint32_t, kinds> next_task{};
    std::vector<std::array<std::uint32_t, 3>> inflight;
    std::uint64_t steps = 0;
    while (!model.terminal_ready()) {
      require(++steps < 100000U,
              "randomized generation scheduler did not terminate");
      if (!inflight.empty() && (next() & 1U) != 0U) {
        const auto choice = static_cast<std::size_t>(next() % inflight.size());
        const auto item = inflight[choice];
        require(model.complete(static_cast<TaskClass>(item[0]), item[1],
                               item[2]),
                "claimed task failed completion");
        inflight.erase(inflight.begin() + static_cast<std::ptrdiff_t>(choice));
        continue;
      }
      const auto kind = static_cast<std::uint32_t>(next() % kinds);
      if (next_task[kind] >= expected[kind])
        continue;
      const auto context = static_cast<std::uint32_t>(next() % contexts);
      if (model.claim(static_cast<TaskClass>(kind), next_task[kind], context)) {
        inflight.push_back(
            std::array<std::uint32_t, 3>{kind, next_task[kind], context});
        ++next_task[kind];
      }
    }
    require(inflight.empty(), "terminal published with inflight work");
    require(model.publish_terminal_once(),
            "elected terminal checker failed first publication");
    require(!model.publish_terminal_once() &&
                model.terminal_publish_count() == 1U,
            "terminal was published more than once");
    for (std::uint32_t context = 0; context < contexts; ++context)
      require(model.context_credit(context) == 3U,
              "context credit was not restored");
  }
  std::cout << "[PASS] randomized static generation scheduler seeds=1000\n";
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
        test_device_capacity_failure_and_recovery();
#if !MEGAMOE_DEBUG_VALIDATE
        test_release_fused_status_and_config_fallback();
#endif
        test_twenty_failure_recovery_rounds();
        test_config_wire_corruption(2, 0);
        test_config_wire_corruption(2, 1);
        test_exact_session_identity_agreement(false);
        test_exact_session_identity_agreement(true);
      }
      if (world == 1) {
        test_randomized_generation_scheduler();
        test_device_dispatch_handle_lifetime();
        test_delayed_route_producer_readiness();
        test_resource_pool_and_cuda_api_audit();
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
    if (argc == 2 && std::string(argv[1]) == "--v3-protocol-smoke") {
      const auto probe =
          megamoe::cuda::testing::run_persistent_v3_protocol_probe();
      require(probe.rdma_ctas == 8U && probe.relay_ctas == 8U &&
                  probe.compute_ctas + probe.rdma_ctas + probe.relay_ctas ==
                      probe.sm_count,
              "V3 fixed CTA role partition mismatch");
      require(probe.tasks_completed == 8ULL * probe.tasks_per_context &&
                  probe.observed_checksum == probe.expected_checksum &&
                  probe.terminal_epoch != 0ULL &&
                  probe.terminal_publish_count == 1ULL &&
                  probe.error_code == 0ULL,
              "V3 ticket/scoreboard protocol probe failed");
      std::cout << "[PASS] V3 fixed roles rdma=" << probe.rdma_ctas
                << " relay=" << probe.relay_ctas
                << " compute=" << probe.compute_ctas
                << " tasks=" << probe.tasks_completed
                << " terminal_publish_count="
                << probe.terminal_publish_count << '\n';
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--resource-pool-test") {
      print_nccl_runtime_info();
      test_resource_pool_and_cuda_api_audit();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--scheduler-state-test") {
      test_randomized_generation_scheduler();
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--bench") {
      benchmark(static_cast<std::uint32_t>(parse_u64(argv[2])));
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--transport-bench") {
      transport_benchmark();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--transport-capability") {
      const auto capability =
          megamoe::cuda::probe_activation_transport_capability(8);
      std::cout << "TRANSPORT_CAPABILITY compiled_nccl="
                << capability.compiled_nccl_version
                << " runtime_nccl=" << capability.runtime_nccl_version
                << " device_headers=" << capability.nccl_device_headers
                << " gin_headers_available="
                << capability.gin_headers_available
                << " gin_symbols_present=" << capability.gin_symbols_present
                << " gin_runtime_initializable="
                << capability.gin_runtime_initializable
                << " gin_backend_implemented="
                << capability.gin_backend_implemented
                << " gin_usable=" << capability.gin_usable
                << " system_scope_atomics="
                << capability.system_scope_atomics
                << " reason='" << capability.gin_unavailable_reason << "'\n";
      for (std::size_t i = 0; i < capability.peer_access_matrix.size(); ++i)
        std::cout << static_cast<unsigned int>(
            capability.peer_access_matrix[i]);
      std::cout << '\n';
      const auto ordering_ok =
          megamoe::cuda::testing::run_local_p2p_put_signal_probe(
              0, 1, 64U * 1024U * 1024U);
      std::cout << "LOCAL_P2P_ORDERING bytes="
                << (64U * 1024U * 1024U)
                << " last_block_signal_verified=" << ordering_ok
                << " early_signal_rejected=" << ordering_ok << '\n';
      std::cout << "LOCAL_P2P_DEVICE_PUT_GBPS gpu0_to_gpu1="
                << megamoe::cuda::testing::measure_local_p2p_put_bandwidth_gbps(
                       0, 1, 64U * 1024U * 1024U, 5, 20)
                << " bytes_per_sample=" << (64U * 1024U * 1024U)
                << " warmup=5 measured=20\n";
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
