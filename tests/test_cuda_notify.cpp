#include "cuda/testing/api.hpp"
#include "cuda/forward/tuning_config.hpp"

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

#include "weftmoe/api/errors.hpp"
#include "weftmoe/metadata/serialize.hpp"
#include "cuda/activation_transport_internal.hpp"
#include "cuda/device_notify_internal.hpp"
#include "cuda/nccl_protocol_internal.hpp"
#include "cuda/testing/generation_scheduler.hpp"

namespace {

using Clock = std::chrono::steady_clock;

#if WEFTMOE_DEVICE_TIMELINE
void write_device_timeline(
    const weftmoe::cuda::profiling::DeviceTimelineSnapshot &snapshot,
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
weftmoe::cuda::CudaGateInput make_unit_gates(
    const weftmoe::cuda::CudaRouteInput &routes,
    const weftmoe::RankCollectiveContext &context,
    const weftmoe::cuda::CudaExecution &execution) {
  return weftmoe::cuda::testing::upload_fp32_gates_for_testing(
      std::vector<float>(static_cast<std::size_t>(routes.route_count()),
                         1.0F),
      routes, context, execution);
}

weftmoe::RuntimeConfig make_config(std::uint32_t nodes, std::uint32_t gpus,
                                   std::uint32_t experts,
                                   std::uint32_t top_k,
                                   std::uint32_t mtile) {
  weftmoe::RuntimeConfig config;
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

std::vector<weftmoe::RankRoutingInput>
seeded_inputs(std::uint32_t world, std::uint32_t experts,
              std::uint32_t top_k, std::uint64_t seed,
              std::uint64_t max_tokens = 10) {
  const auto next = [&seed]() mutable {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };
  std::vector<weftmoe::RankRoutingInput> result(world);
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
    const std::vector<weftmoe::transport::BFloat16> &values) {
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
  return weftmoe::transport::float_to_bfloat16(
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
  weftmoe::cuda::CollectiveProtocolAudit audit;
  weftmoe::cuda::DeviceMaterializationAudit device_audit;
  weftmoe::cuda::NcclWorkspaceUsage workspace;
  weftmoe::cuda::GenerationSetupTimings setup;
  weftmoe::cuda::GenerationExecutionTimings execution;
  double session_setup_us = 0;
  double pool_setup_us = 0;
  double wall_us = 0;
  std::exception_ptr error;
};

std::vector<RunResult> run_fixture(
    const weftmoe::RuntimeConfig &config,
    const std::vector<weftmoe::RankRoutingInput> &routes,
    const std::string &identity, std::uint64_t generation,
    std::uint64_t max_tokens, bool stagger = false) {
  const auto world = config.topology.world_size;
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<RunResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        auto execution = weftmoe::cuda::make_cuda_execution(
            identity + "-exec-" + std::to_string(rank), device);
        const auto session_start = Clock::now();
        auto session = weftmoe::cuda::make_nccl_session(
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
        auto pool = weftmoe::cuda::make_device_notify_resource_pool(
            session, execution, {config, max_tokens, 4096, 2});
        weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, execution, pool);
        auto result = coordinator.run({config, routes[rank], max_tokens});
        const auto finish = Clock::now();
        results[rank].metadata = weftmoe::serialize_dispatch_metadata(
            weftmoe::cuda::download_device_metadata_for_validation(
                result.dispatch(), execution));
        results[rank].audit = result.audit();
        results[rank].device_audit = result.dispatch().audit();
        results[rank].workspace = result.workspace_usage();
        results[rank].setup = result.setup_timings();
        results[rank].execution = result.execution_timings();
        results[rank].wall_us =
            std::chrono::duration<double, std::micro>(finish - start).count();
        weftmoe::cuda::validate_cuda_dispatch_metadata(result.dispatch(),
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
  weftmoe::cuda::CudaExecution execution;
  weftmoe::cuda::NcclSession base_session;
  weftmoe::cuda::DeviceNotifyResourcePool pool;
  weftmoe::cuda::CudaRouteInput device_routes;
  weftmoe::cuda::CudaGateInput device_gates;
  double session_setup_us = 0;
  double pool_setup_us = 0;
  std::exception_ptr error;
};

std::vector<ReusableRankResources>
make_reusable_resources(const weftmoe::RuntimeConfig &config,
                        const std::string &identity,
                        std::uint64_t generation,
                        std::uint64_t max_tokens) {
  const auto world = config.topology.world_size;
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<ReusableRankResources> resources(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        resources[rank].execution = weftmoe::cuda::make_cuda_execution(
            identity + "-exec-" + std::to_string(rank), device);
        const auto start = Clock::now();
        resources[rank].base_session = weftmoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        resources[rank].session_setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - start)
                .count();
        resources[rank].pool = weftmoe::cuda::make_device_notify_resource_pool(
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
    const weftmoe::RuntimeConfig &config,
    const std::vector<weftmoe::RankRoutingInput> &routes,
    const std::vector<ReusableRankResources> &resources,
    const std::string &identity, std::uint64_t generation,
    std::uint64_t max_tokens) {
  const auto world = config.topology.world_size;
  std::vector<RunResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto session = weftmoe::cuda::make_nccl_generation(
            resources[rank].base_session, identity, generation);
        const auto context = weftmoe::make_rank_collective_context(
            session.session(), rank, resources[rank].execution.token());
        const auto exact_routes = resources[rank].device_routes
                                      ? weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
                                            resources[rank].device_routes,
                                            context,
                                            resources[rank].execution)
                                      : weftmoe::cuda::CudaRouteInput{};
        const auto exact_gates = exact_routes
            ? weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
                  resources[rank].device_gates, exact_routes, context,
                  resources[rank].execution)
            : weftmoe::cuda::CudaGateInput{};
        weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, resources[rank].execution, resources[rank].pool);
        auto result = exact_routes
                          ? coordinator.run_device(
                                {config, exact_routes, exact_gates, max_tokens})
                          : coordinator.run(
                                {config, routes[rank], max_tokens});
        results[rank].metadata = weftmoe::serialize_dispatch_metadata(
            weftmoe::cuda::download_device_metadata_for_validation(
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
    const auto expected = weftmoe::build_notify_dispatch_for_testing(
        config, routes, "local-fixture", seed + 1);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      require(actual[rank].metadata ==
                  weftmoe::serialize_dispatch_metadata(expected[rank]),
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
  weftmoe::cuda::CollectiveProtocolAudit audit;
  std::exception_ptr unexpected;
};

void mutate_scenario(const std::string &scenario, std::uint32_t rank,
                     weftmoe::RuntimeConfig &config,
                     weftmoe::RankRoutingInput &routes,
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
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = weftmoe::cuda::make_cuda_execution(
            "failure-exec-" + std::to_string(rank), rank);
        auto config = base_config;
        auto routes = base_routes[rank];
        std::uint64_t max_tokens = 10;
        std::string identity = "failure-" + scenario;
        std::uint64_t generation = 1001;
        mutate_scenario(scenario, rank, config, routes, max_tokens, identity,
                        generation);
        auto session = weftmoe::cuda::make_nccl_session(
            identity, world, rank, generation, rank, bootstrap);
        auto coordinator =
            weftmoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
        try {
          (void)coordinator.run({config, routes, max_tokens});
          throw std::runtime_error("invalid generation unexpectedly passed");
        } catch (const weftmoe::NotifyDispatchError &error) {
          require(error.code() ==
                      weftmoe::NotifyErrorCode::ControlProtocolFailure,
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
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = weftmoe::cuda::make_cuda_execution(
            "capacity-exec-" + std::to_string(rank), rank);
        auto base = weftmoe::cuda::make_nccl_session(
            "capacity-base", world, rank, 6200, rank, bootstrap);
        auto pool = weftmoe::cuda::make_device_notify_resource_pool(
            base, execution, {good_config, 1, 4096, 1, 2});
        auto bad_session = weftmoe::cuda::make_nccl_generation(
            base, "capacity-bad", 6201);
        weftmoe::cuda::CudaNotifyGenerationCoordinator bad(
            bad_session, execution, pool);
        bool rejected = false;
        try {
          (void)bad.run({bad_config, {1, {0}}, 1});
        } catch (const weftmoe::NotifyDispatchError &error) {
          rejected = error.code() ==
                     weftmoe::NotifyErrorCode::ExpertCapacityExceeded;
        }
        require(rejected, "device expert capacity did not reject generation");
        const auto bad_audit = bad.audit();
        require(bad_audit.control_starts == 1 &&
                    bad_audit.config_starts == 1 &&
                    bad_audit.expert_count_starts == 1 &&
                    bad_audit.packet_summary_starts == 1,
                "device capacity failure occurred before complete collectives");
        auto good_session = weftmoe::cuda::make_nccl_generation(
            base, "capacity-good", 6202);
        weftmoe::cuda::CudaNotifyGenerationCoordinator good(
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

#if !WEFTMOE_DEBUG_VALIDATE
void test_release_fused_status_and_config_fallback() {
  constexpr std::uint32_t world = 2;
  const auto config = make_config(1, world, 8, 3, 4);
  const std::vector<weftmoe::RankRoutingInput> good_routes(
      world, weftmoe::RankRoutingInput{1, {0, 1, 2}});
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> unexpected(world);
  std::array<std::array<std::string, world>, 2> failures;
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = weftmoe::cuda::make_cuda_execution(
            "release-contract-exec-" + std::to_string(rank), rank);
        auto base = weftmoe::cuda::make_nccl_session(
            "release-contract", world, rank, 6600, rank, bootstrap);
        auto pool = weftmoe::cuda::make_device_notify_resource_pool(
            base, execution, {config, 10, 4096, 1, 2});

        const auto run_good = [&](std::uint64_t generation) {
          auto session = weftmoe::cuda::make_nccl_generation(
              base, "release-contract", generation);
          weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
              session, execution, pool);
          auto result = coordinator.run({config, good_routes[rank], 10});
          const auto audit = coordinator.audit();
          require(result && audit.fused_notify_starts == 1U &&
                      audit.control_starts == 0U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "release recovery did not use one fused collective");
          const auto actual = weftmoe::serialize_dispatch_metadata(
              weftmoe::cuda::download_device_metadata_for_validation(
                  result.dispatch(), execution));
          const auto expected = weftmoe::build_notify_dispatch_for_testing(
              config, good_routes, "release-contract", generation);
          require(actual ==
                      weftmoe::serialize_dispatch_metadata(expected[rank]),
                  "release recovery metadata/hash differs from oracle");
        };

        // Establish the immutable session config. The prime itself uses the
        // complete protocol; every following base-config generation is fused.
        {
          auto session = weftmoe::cuda::make_nccl_generation(
              base, "release-contract", 6601);
          weftmoe::cuda::CudaNotifyGenerationCoordinator prime(
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
          auto session = weftmoe::cuda::make_nccl_generation(
              base, "release-contract", 6602);
          weftmoe::cuda::CudaNotifyGenerationCoordinator invalid(
              session, execution, pool);
          try {
            (void)invalid.run({config, invalid_routes, 10});
            throw std::runtime_error(
                "fused invalid-route generation unexpectedly passed");
          } catch (const weftmoe::NotifyDispatchError &error) {
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
          auto session = weftmoe::cuda::make_nccl_generation(
              base, "release-contract", 6604);
          weftmoe::cuda::CudaNotifyGenerationCoordinator changed(
              session, execution, pool);
          try {
            (void)changed.run({changed_config, good_routes[rank], 10});
            throw std::runtime_error(
                "layout-changing invalid generation unexpectedly passed");
          } catch (const weftmoe::NotifyDispatchError &error) {
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

void test_failure_recovery_rounds() {
  constexpr std::uint32_t world = 2;
  constexpr std::size_t recovery_rounds = 4;
  const auto config = make_config(1, world, 8, 3, 4);
  const std::vector<std::string> scenarios{
      "generation_mismatch", "session_mismatch", "topology_mismatch",
      "max_mismatch", "out_of_range"};
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::vector<std::vector<std::string>>> structured(
      scenarios.size(),
      std::vector<std::vector<std::string>>(
          recovery_rounds, std::vector<std::string>(world)));
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = weftmoe::cuda::make_cuda_execution(
            "recovery-exec-" + std::to_string(rank), rank);
        auto base = weftmoe::cuda::make_nccl_session(
            "recovery-base", world, rank, 2000, rank, bootstrap);
        for (std::size_t category = 0; category < scenarios.size();
             ++category) {
          for (std::uint64_t round = 0; round < recovery_rounds; ++round) {
            auto bad_config = config;
            auto routes = weftmoe::RankRoutingInput{1, {0, 1, 2}};
            auto max_tokens = std::uint64_t{10};
            auto identity = std::string("recovery-") + scenarios[category];
            auto generation = 3000U +
                              static_cast<std::uint64_t>(category) * 100U +
                              round * 2U;
            mutate_scenario(scenarios[category], rank, bad_config, routes,
                            max_tokens, identity, generation);
            auto bad_session = weftmoe::cuda::make_nccl_generation(
                base, identity, generation);
            auto bad =
                weftmoe::cuda::testing::make_unpooled_coordinator_for_testing(
                    bad_session, execution);
            bool rejected = false;
            try {
              auto unexpected = bad.run({bad_config, routes, max_tokens});
              require(!unexpected,
                      "invalid generation published validated result");
            } catch (const weftmoe::NotifyDispatchError &error) {
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

            auto good_session = weftmoe::cuda::make_nccl_generation(
                base, "recovery-good-" + scenarios[category],
                3001U + static_cast<std::uint64_t>(category) * 100U +
                    round * 2U);
            auto good =
                weftmoe::cuda::testing::make_unpooled_coordinator_for_testing(
                    good_session, execution);
            auto result = good.run({config, {1, {0, 1, 2}}, 10});
            require(result.audit().control_starts == 1 &&
                        result.audit().config_starts == 1 &&
                        result.audit().expert_count_starts == 1 &&
                        result.audit().packet_summary_starts == 1,
                    "recovery audit is not 1/1/1/1");
            require(result && !weftmoe::serialize_dispatch_metadata(
                                   weftmoe::cuda::download_device_metadata_for_validation(
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
    for (std::size_t round = 0; round < recovery_rounds; ++round)
      require(!structured[category][round][0].empty() &&
                  structured[category][round][0] ==
                      structured[category][round][1],
              "recovery ranks selected different structured errors");
  std::cout << "[PASS] same ncclComm_t 5 categories x " << recovery_rounds
            << " invalid->valid recovery rounds\n";
}

void test_fault_injection() {
  using Point = weftmoe::cuda::testing::NcclFaultPoint;
  const auto require_recovery = [](const auto &recovered, const auto &audit,
                                   const auto &config, const auto &routes,
                                   const auto &execution,
                                   const std::string &identity,
                                   std::uint64_t generation,
                                   const char *failure) {
#if WEFTMOE_DEBUG_VALIDATE
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
    const auto actual = weftmoe::serialize_dispatch_metadata(
        weftmoe::cuda::download_device_metadata_for_validation(
            recovered.dispatch(), execution));
    const auto expected = weftmoe::build_notify_dispatch_for_testing(
        config, {routes}, identity, generation);
    require(actual == weftmoe::serialize_dispatch_metadata(expected.front()),
            "fault recovery produced incorrect dispatch metadata/hash");
  };
  struct FaultExpectation {
    Point point;
    weftmoe::cuda::CollectiveProtocolAudit audit;
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
    weftmoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
    auto execution = weftmoe::cuda::make_cuda_execution("fault-exec", 0);
    auto session = weftmoe::cuda::make_nccl_session(
        "fault", 1, 0, 4000 + index, 0, bootstrap);
    weftmoe::cuda::CollectiveProtocolAudit audit;
    {
      auto pool = weftmoe::cuda::make_device_notify_resource_pool(
          session, execution, {config, 10, 4096, 1, 2});
      // Establish the session-static config on this exact pooled backend.
      // Release must then recover through the fused protocol; debug must
      // continue to execute the complete protocol without this release-only
      // setup prime. Lifecycle assertions below distinguish both modes.
#if !WEFTMOE_DEBUG_VALIDATE
      {
        auto prime_session = weftmoe::cuda::make_nccl_generation(
            session, "fault", 3900 + index);
        weftmoe::cuda::CudaNotifyGenerationCoordinator prime(
            prime_session, execution, pool);
        require(static_cast<bool>(
                    prime.run({config, {1, {0, 1, 2}}, 10})),
                "failed to prime fault-recovery static config");
      }
#endif
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, pool);
      weftmoe::cuda::testing::inject_next_nccl_fault(points[index].point);
      bool failed = false;
      try {
        auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
        require(!unexpected, "fault generation published validated result");
      } catch (const weftmoe::NotifyDispatchError &error) {
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
      auto recovery_session = weftmoe::cuda::make_nccl_generation(
          session, "fault", 5000 + index);
      weftmoe::cuda::CudaNotifyGenerationCoordinator recovery(
          recovery_session, execution, pool);
      const weftmoe::RankRoutingInput routes{1, {0, 1, 2}};
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
        weftmoe::cuda::testing::nccl_lifecycle_snapshot();
    require(lifecycle.slot_allocations == lifecycle.slot_destructions,
            "fault leaked a collective slot allocation");
    require(lifecycle.completion_lease_creates ==
                lifecycle.completion_lease_destroys,
            "fault leaked a completion lease");
#if WEFTMOE_DEBUG_VALIDATE
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
    weftmoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
    auto execution = weftmoe::cuda::make_cuda_execution(
        "device-stage-fault-exec", 0);
    auto session = weftmoe::cuda::make_nccl_session(
        "device-stage-fault", 1, 0, 4050 + index, 0, bootstrap);
    auto pool = weftmoe::cuda::make_device_notify_resource_pool(
        session, execution, {config, 10, 4096, 1, 2});
#if !WEFTMOE_DEBUG_VALIDATE
    {
      auto prime_session = weftmoe::cuda::make_nccl_generation(
          session, "device-stage-fault", 3950 + index);
      weftmoe::cuda::CudaNotifyGenerationCoordinator prime(
          prime_session, execution, pool);
      require(static_cast<bool>(
                  prime.run({config, {1, {0, 1, 2}}, 10})),
              "failed to prime device-stage recovery static config");
    }
#endif
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    weftmoe::cuda::testing::inject_next_nccl_fault(
        device_stage_points[index].point);
    bool failed = false;
    try {
      (void)coordinator.run({config, {1, {0, 1, 2}}, 10});
    } catch (const weftmoe::NotifyDispatchError &error) {
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
    auto recovery_session = weftmoe::cuda::make_nccl_generation(
        session, "device-stage-fault", 4150 + index);
    weftmoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    const weftmoe::RankRoutingInput routes{1, {0, 1, 2}};
    auto recovered = recovery.run({config, routes, 10});
    require_recovery(recovered, recovery.audit(), config, routes, execution,
                     "device-stage-fault", 4150 + index,
                     "device-stage fault did not recover with the exact "
                     "protocol on the same communicator");
    const auto lifecycle =
        weftmoe::cuda::testing::nccl_lifecycle_snapshot();
#if WEFTMOE_DEBUG_VALIDATE
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
    weftmoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
    auto execution = weftmoe::cuda::make_cuda_execution(
        deadline ? "deadline-exec" : "cancel-exec", 0);
    auto session = weftmoe::cuda::make_nccl_session(
        deadline ? "deadline" : "cancel", 1, 0,
        deadline ? 4091 : 4090, 0, bootstrap);
    auto pool = weftmoe::cuda::make_device_notify_resource_pool(
        session, execution, {config, 10, 4096, 1, 2});
    weftmoe::cuda::GenerationRunOptions options;
    if (deadline)
      options.deadline_after = std::chrono::microseconds(1);
    else
      options.cancellation.request_cancel();
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    bool failed = false;
    try {
      (void)coordinator.run({config, {1, {0, 1, 2}}, 10}, options);
    } catch (const weftmoe::NotifyDispatchError &error) {
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
        weftmoe::cuda::testing::nccl_lifecycle_snapshot();
    require(lifecycle.operation_cancels == 1 &&
                lifecycle.validated_publishes == 0,
            "cancellation was not executed exactly once or published a handle");
    auto recovery_session = weftmoe::cuda::make_nccl_generation(
        session, deadline ? "deadline" : "cancel",
        deadline ? 4191 : 4190);
    weftmoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    const weftmoe::RankRoutingInput routes{1, {0, 1, 2}};
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
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = weftmoe::cuda::make_cuda_execution(
            "wire-corruption-exec-" + std::to_string(rank), rank);
        auto session = weftmoe::cuda::make_nccl_session(
            "wire-corruption", world, rank, 5100 + corrupt_rank, rank,
            bootstrap);
        auto coordinator =
            weftmoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
        if (rank == corrupt_rank)
          weftmoe::cuda::testing::corrupt_next_config_wire(64);
        try {
          auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
          require(!unexpected, "corrupt config published validated result");
        } catch (const weftmoe::NotifyDispatchError &error) {
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
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = weftmoe::cuda::make_cuda_execution(
            "exact-session-exec-" + std::to_string(rank), rank);
        // Display name and header summary are deliberately equal. Only one raw
        // byte differs, proving that exact identity bytes are the final gate.
        auto session = weftmoe::cuda::make_nccl_session(
            "same-display", world, rank, different_length ? 5201 : 5200,
            rank, bootstrap);
        std::vector<std::byte> raw_identity{
            std::byte{0x41}, std::byte{0x42},
            static_cast<std::byte>(0x43U + rank)};
        if (different_length && rank == 1)
          raw_identity.push_back(std::byte{0x44});
        weftmoe::cuda::testing::override_next_protocol_identity(
            std::move(raw_identity),
            {0x1111111111111111ULL, 0x2222222222222222ULL});
        auto coordinator =
            weftmoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
        try {
          auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
          require(!unexpected,
                  "different exact session identity published result");
        } catch (const weftmoe::NotifyDispatchError &error) {
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

void test_protocol_failure_suite() {
  test_protocol_regressions();
  test_failure_recovery_rounds();
  test_config_wire_corruption(2, 0);
  test_config_wire_corruption(2, 1);
  test_exact_session_identity_agreement(false);
  test_exact_session_identity_agreement(true);
}
void test_device_only_sanitizer_paths() {
  const auto config = make_config(1, 1, 8, 3, 4);
  auto execution = weftmoe::cuda::make_cuda_execution("device-only", 0);
  auto session = weftmoe::make_collective_session("device-only", 1, 5001);
  auto context = weftmoe::make_rank_collective_context(
      session, 0, execution.token());
  const auto run = [&](weftmoe::RankRoutingInput input, bool valid) {
    auto routes = weftmoe::cuda::upload_cuda_routes(
        input, config.top_k, context, execution);
    auto plan = weftmoe::cuda::query_cuda_notify_workspace(
        config, 0, input.num_tokens, 4);
    auto workspace = weftmoe::cuda::allocate_cuda_notify_workspace(
        plan, context, execution);
    auto gates = make_unit_gates(routes, context, execution);
    auto operation = weftmoe::cuda::start_cuda_rank_local_notify(
        config, context, execution, routes, gates, std::move(workspace));
    const auto result = operation.wait_preflight();
    require(result.valid() == valid,
            "device-only validation status mismatch");
    if (!valid)
      require(result.status().error ==
                  weftmoe::cuda::ControlLocalError::ExpertOutOfRange,
              "device error path lost coordinates");
  };
  run({3, {5, 1, 7, 1, 2, 3, 0, 7, 4}}, true);
  run({1, {0, 1, 8}}, false);
  const weftmoe::RankRoutingInput handoff_input{1, {0, 2, 4}};
  auto producer_routes = weftmoe::cuda::upload_cuda_routes(
      handoff_input, config.top_k, context, execution);
  auto consumer_execution =
      weftmoe::cuda::make_cuda_execution("device-only-consumer", 0);
  auto consumer_context = weftmoe::make_rank_collective_context(
      session, 0, consumer_execution.token());
  auto consumer_plan = weftmoe::cuda::query_cuda_notify_workspace(
      config, 0, handoff_input.num_tokens, 4);
  bool no_wait_failed = false;
  try {
    auto workspace = weftmoe::cuda::allocate_cuda_notify_workspace(
        consumer_plan, consumer_context, consumer_execution);
    auto gates = make_unit_gates(producer_routes, context, execution);
    (void)weftmoe::cuda::start_cuda_rank_local_notify(
        config, consumer_context, consumer_execution, producer_routes,
        gates, std::move(workspace));
  } catch (const weftmoe::NotifyDispatchError &) {
    no_wait_failed = true;
  }
  require(no_wait_failed,
          "cross-stream route consumption without event handoff succeeded");
  auto imported = weftmoe::cuda::import_selected_expert_routes(
      producer_routes, consumer_context, consumer_execution);
  auto imported_workspace = weftmoe::cuda::allocate_cuda_notify_workspace(
      consumer_plan, consumer_context, consumer_execution);
  auto imported_gates = make_unit_gates(
      imported, consumer_context, consumer_execution);
  auto imported_operation = weftmoe::cuda::start_cuda_rank_local_notify(
      config, consumer_context, consumer_execution, imported,
      imported_gates, std::move(imported_workspace));
  require(imported_operation.wait_preflight().valid(),
          "legal cross-stream event handoff failed");
  auto foreign_session =
      weftmoe::make_collective_session("foreign-device-only", 1, 5001);
  auto foreign_context = weftmoe::make_rank_collective_context(
      foreign_session, 0, consumer_execution.token());
  bool foreign_failed = false;
  try {
    (void)weftmoe::cuda::import_selected_expert_routes(
        producer_routes, foreign_context, consumer_execution);
  } catch (const weftmoe::NotifyDispatchError &) {
    foreign_failed = true;
  }
  require(foreign_failed,
          "foreign route pipeline session handoff did not fail closed");
  auto alternate_routes = weftmoe::cuda::upload_cuda_routes(
      handoff_input, config.top_k, context, execution);
  auto original_gates = make_unit_gates(producer_routes, context, execution);
  bool route_gate_binding_failed = false;
  try {
    weftmoe::cuda::DeviceGateProvenanceBridge::validate(
        original_gates, alternate_routes, context, execution);
  } catch (const weftmoe::NotifyDispatchError &) {
    route_gate_binding_failed = true;
  }
  require(route_gate_binding_failed,
          "gate lease bound to another route allocation was accepted");
  const auto wide_config = make_config(73, 1, 146, 6, 4);
  auto wide_session =
      weftmoe::make_collective_session("device-wide-mask", 73, 5002);
  auto wide_context = weftmoe::make_rank_collective_context(
      wide_session, 0, execution.token());
  std::vector<weftmoe::RankRoutingInput> wide_inputs(73);
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
  const auto expected_wide = weftmoe::build_notify_dispatch_for_testing(
      wide_config, wide_inputs, "device-wide-mask", 5002);
  std::string wide_baseline;
  weftmoe::cuda::DeviceDispatchHandle wide_handle;
  for (std::uint32_t repeat = 0; repeat < 20; ++repeat) {
    wide_handle = weftmoe::cuda::testing::materialize_fake_device_collectives(
        wide_config, wide_context, execution, wide_inputs[0], 4,
        count_matrix, summary_matrix);
    const auto downloaded =
        weftmoe::cuda::download_device_metadata_for_validation(
            wide_handle, execution);
    const auto serialized = weftmoe::serialize_dispatch_metadata(downloaded);
    require(serialized == weftmoe::serialize_dispatch_metadata(expected_wide[0]),
            "73-node full device materializer/oracle mismatch");
    if (repeat == 0)
      wide_baseline = serialized;
    else
      require(serialized == wide_baseline,
              "73-node materializer is not byte deterministic");
  }
  auto wide_plan = weftmoe::cuda::query_cuda_notify_workspace(
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
  const weftmoe::RankRoutingInput transport_routes{
      3, {5, 1, 7, 1, 2, 3, 0, 7, 4}};
  std::vector<std::uint64_t> transport_counts(8, 0);
  for (const auto selected : transport_routes.selected_experts)
    ++transport_counts[static_cast<std::size_t>(selected)];
  const std::vector<std::uint64_t> transport_summary{3};
  auto transport_dispatch =
      weftmoe::cuda::testing::materialize_fake_device_collectives(
          config, context, execution, transport_routes, 4, transport_counts,
          transport_summary);
  const auto host_dispatch = weftmoe::build_notify_dispatch_for_testing(
      config, {transport_routes}, "device-only", 5001);
  weftmoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = 33;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 2;
  std::vector<weftmoe::transport::BFloat16> activations(3U * 33U);
  for (std::size_t i = 0; i < activations.size(); ++i)
    activations[i] = weftmoe::transport::float_to_bfloat16(
        static_cast<float>((i * 13U) % 97U) / 16.0F);
  const auto oracle = weftmoe::transport::simulate_activation_transport(
      host_dispatch[0], activations,
      std::vector<float>(transport_routes.selected_experts.size(), 1.0F),
      transport_config, 1);
  weftmoe::cuda::ActivationTransportPoolConfig transport_pool_config;
  transport_pool_config.max_tokens_per_rank = 4;
  transport_pool_config.max_hidden_size = 33;
  transport_pool_config.max_top_k = 3;
  transport_pool_config.max_world_size = 1;
  transport_pool_config.max_nodes = 1;
  transport_pool_config.max_destination_rows_per_rank = 12;
  transport_pool_config.channels = 1;
  transport_pool_config.ring_slots_per_source_channel = 2;
  transport_pool_config.max_inflight_generations = 2;
  auto transport_pool = weftmoe::cuda::make_activation_transport_pool(
      session, execution, transport_pool_config);
  auto activation_lease =
      weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, context, execution, transport_pool);
  weftmoe::cuda::DeviceActivationTransport transport(
      session, execution, transport_pool);
  const auto api_before = weftmoe::cuda::testing::cuda_api_audit_snapshot();
  auto transport_result = transport.run_device_simulation_synthetic(
      transport_dispatch, activation_lease, transport_config);
  const auto api_after = weftmoe::cuda::testing::cuda_api_audit_snapshot();
  require(api_after.device_allocations == api_before.device_allocations &&
              api_after.device_frees == api_before.device_frees &&
              api_after.pinned_allocations == api_before.pinned_allocations &&
              api_after.pinned_frees == api_before.pinned_frees &&
              api_after.event_creates == api_before.event_creates &&
              api_after.event_destroys == api_before.event_destroys,
          "steady activation transport allocated CUDA resources");
  const auto downloaded_transport =
      weftmoe::cuda::download_activation_transport_for_validation(
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
              weftmoe::transport::ActivationTransportBackend::DeviceSimulation,
          "simulation backend was mislabeled as P2P");
  const auto transport_baseline = downloaded_transport.combined_tokens;
  transport_result = {};
  activation_lease = {};
  for (std::uint32_t repeat = 0; repeat < 20; ++repeat) {
    auto repeated_activation =
        weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, 3, 33, context, execution, transport_pool);
    auto repeated_result = transport.run_device_simulation_synthetic(
        transport_dispatch, repeated_activation, transport_config);
    const auto repeated =
        weftmoe::cuda::download_activation_transport_for_validation(
            repeated_result, execution);
    require(repeated.combined_tokens == transport_baseline,
            "ring epoch reuse changed device transport bytes");
  }
  require(transport_pool.usage().slots_available == 2 &&
              transport_pool.usage().slots_in_use == 0,
          "transport slots were not returned after final lease destruction");

  auto next_transport_session =
      weftmoe::make_collective_session("device-only", 1, 5002);
  auto next_transport_context = weftmoe::make_rank_collective_context(
      next_transport_session, 0, execution.token());
  auto wrong_generation_activation =
      weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, next_transport_context, execution,
          transport_pool);
  bool generation_failed = false;
  try {
    (void)transport.run_device_simulation_synthetic(transport_dispatch,
                                  wrong_generation_activation,
                                  transport_config);
  } catch (const weftmoe::NotifyDispatchError &error) {
    generation_failed = error.code() == weftmoe::NotifyErrorCode::WorkspaceMismatch;
  }
  require(generation_failed,
          "cross-generation activation lease was accepted");
  wrong_generation_activation = {};

  auto exhausted_config = transport_pool_config;
  exhausted_config.max_inflight_generations = 1;
  auto exhausted_pool = weftmoe::cuda::make_activation_transport_pool(
      session, execution, exhausted_config);
  auto held_activation =
      weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, context, execution, exhausted_pool);
  bool exhausted = false;
  try {
    (void)weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
        activations, 3, 33, context, execution, exhausted_pool);
  } catch (const weftmoe::NotifyDispatchError &error) {
    exhausted = error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
  }
  require(exhausted && exhausted_pool.usage().slots_in_use == 1,
          "transport pool exhaustion did not fail closed");
  held_activation = {};
  require(exhausted_pool.usage().slots_available == 1,
          "transport pool did not recover after final input lease");

  weftmoe::transport::ActivationTransportConfig wide_transport_config;
  wide_transport_config.hidden_size = 7;
  wide_transport_config.channels = 1;
  wide_transport_config.ring_slots_per_source_channel = 1;
  weftmoe::cuda::ActivationTransportPoolConfig wide_transport_pool_config;
  wide_transport_pool_config.max_tokens_per_rank = 4;
  wide_transport_pool_config.max_hidden_size = 7;
  wide_transport_pool_config.max_top_k = 6;
  wide_transport_pool_config.max_world_size = 73;
  wide_transport_pool_config.max_nodes = 73;
  wide_transport_pool_config.max_destination_rows_per_rank = 32;
  wide_transport_pool_config.channels = 1;
  wide_transport_pool_config.ring_slots_per_source_channel = 1;
  wide_transport_pool_config.max_inflight_generations = 1;
  auto wide_transport_pool = weftmoe::cuda::make_activation_transport_pool(
      wide_session, execution, wide_transport_pool_config);
  std::vector<weftmoe::transport::BFloat16> wide_activations(7);
  for (std::size_t i = 0; i < wide_activations.size(); ++i)
    wide_activations[i] = weftmoe::transport::float_to_bfloat16(
        static_cast<float>(i + 1U) / 8.0F);
  auto wide_activation =
      weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
          wide_activations, 1, 7, wide_context, execution,
          wide_transport_pool);
  weftmoe::cuda::DeviceActivationTransport wide_transport(
      wide_session, execution, wide_transport_pool);
  auto wide_transport_result = wide_transport.run_device_simulation_synthetic(
      wide_handle, wide_activation, wide_transport_config);
  auto gate_before = wide_transport_config;
  gate_before.gate_placement =
      weftmoe::transport::ActivationGatePlacement::DestinationBeforeUnsupported;
  bool gate_before_failed = false;
  try {
    (void)wide_transport.run_device_simulation_synthetic(
        wide_handle, wide_activation, gate_before);
  } catch (const weftmoe::NotifyDispatchError &) {
    gate_before_failed = true;
  }
  require(gate_before_failed,
          "unsupported gate-before placement was silently accepted");
  const auto wide_transport_download =
      weftmoe::cuda::download_activation_transport_for_validation(
          wide_transport_result, execution);
  const auto wide_transport_oracle =
      weftmoe::transport::simulate_activation_transport(
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
      weftmoe::cuda::probe_activation_transport_capability(2);
  require(capability.nccl_device_headers &&
              capability.gin_headers_available &&
              capability.gin_backend_implemented && !capability.gin_usable,
          "GIN header/runtime capability boundary was not detected");
  require(weftmoe::cuda::testing::run_local_p2p_put_signal_probe(0, 1, 4096),
          "real CUDA P2P put+system-signal probe failed");
  std::cout << "[PASS] transport-only persistent device worker, deterministic "
               "synthetic combine, pooled steady path, P2P put/signal, GIN "
               "fail-closed\n";
  auto short_plan = wide_plan;
  --short_plan.device_bytes;
  bool short_failed = false;
  try {
    (void)weftmoe::cuda::allocate_cuda_notify_workspace(
        short_plan, wide_context, execution);
  } catch (const weftmoe::NotifyDispatchError &) {
    short_failed = true;
  }
  require(short_failed, "workspace capacity minus one byte did not fail");
  std::cout << "[PASS] full fake device materializer including 73-node "
               "multiword descriptors, 20 byte-identical repeats\n";
}
void test_delayed_route_producer_readiness() {
  const auto config = make_config(1, 1, 8, 1, 4);
  auto producer = weftmoe::cuda::make_cuda_execution("delay-producer", 0);
  auto consumer = weftmoe::cuda::make_cuda_execution("delay-consumer", 0);
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  auto base = weftmoe::cuda::make_nccl_session(
      "delay-session", 1, 0, 6900, 0, bootstrap);
  const auto producer_context = weftmoe::make_rank_collective_context(
      base.session(), 0, producer.token());
  auto storage = weftmoe::cuda::upload_cuda_routes(
      {1, {0}}, 1, producer_context, producer);
  auto delayed = weftmoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
      storage, 5, 5000000, producer_context, producer);
  const auto consumer_context = weftmoe::make_rank_collective_context(
      base.session(), 0, consumer.token());
  auto imported = weftmoe::cuda::import_selected_expert_routes(
      delayed, consumer_context, consumer);
  auto pool = weftmoe::cuda::make_device_notify_resource_pool(
      base, consumer, {config, 2, 4096, 1, 2});
  weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
      base, consumer, pool);
  auto imported_gates = make_unit_gates(imported, consumer_context, consumer);
  auto result = coordinator.run_device(
      {config, imported, imported_gates, 2});
  const auto downloaded =
      weftmoe::cuda::download_device_metadata_for_validation(
          result.dispatch(), consumer);
  require(downloaded->outgoing.expert_id == std::vector<std::uint32_t>{5},
          "cross-stream Notify observed poison instead of ready-event result");

  auto next = weftmoe::cuda::make_nccl_generation(
      base, "delay-session", 6901);
  const auto next_producer_context = weftmoe::make_rank_collective_context(
      next.session(), 0, producer.token());
  auto delayed_without_ready =
      weftmoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
          storage, 4, 100000000, next_producer_context, producer, false);
  const auto next_consumer_context = weftmoe::make_rank_collective_context(
      next.session(), 0, consumer.token());
  bool missing_ready_failed = false;
  try {
    (void)weftmoe::cuda::import_selected_expert_routes(
        delayed_without_ready, next_consumer_context, consumer);
  } catch (const weftmoe::NotifyDispatchError &) {
    missing_ready_failed = true;
  }
  require(missing_ready_failed,
          "unrecorded producer-ready event was accepted");

  auto unsafe_delayed =
      weftmoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
          storage, 4, 100000000, next_producer_context, producer);
  auto unsafe =
      weftmoe::cuda::DeviceRouteProvenanceBridge::unsafe_import_without_wait_for_testing(
          unsafe_delayed, next_consumer_context, consumer);
  bool unsafe_failed = false;
  try {
    weftmoe::cuda::CudaNotifyGenerationCoordinator unsafe_coordinator(
        next, consumer, pool);
    auto unsafe_gates = make_unit_gates(unsafe, next_consumer_context,
                                        consumer);
    (void)unsafe_coordinator.run_device(
        {config, unsafe, unsafe_gates, 2});
  } catch (const weftmoe::NotifyDispatchError &) {
    unsafe_failed = true;
  }
  require(unsafe_failed,
          "negative no-wait seam did not demonstrate producer race");

  auto same_session = weftmoe::cuda::make_nccl_session(
      "delay-same-stream", 1, 0, 6920, 0,
      weftmoe::cuda::make_nccl_bootstrap_id());
  const auto same_context = weftmoe::make_rank_collective_context(
      same_session.session(), 0, producer.token());
  auto same_storage = weftmoe::cuda::upload_cuda_routes(
      {1, {0}}, 1, same_context, producer);
  auto same_delayed = weftmoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
      same_storage, 6, 5000000, same_context, producer);
  auto same_pool = weftmoe::cuda::make_device_notify_resource_pool(
      same_session, producer, {config, 2, 4096, 1, 2});
  weftmoe::cuda::CudaNotifyGenerationCoordinator same_coordinator(
      same_session, producer, same_pool);
  auto same_gates = make_unit_gates(same_delayed, same_context, producer);
  auto same_result = same_coordinator.run_device(
      {config, same_delayed, same_gates, 2});
  const auto same_download =
      weftmoe::cuda::download_device_metadata_for_validation(
          same_result.dispatch(), producer);
  require(same_download->outgoing.expert_id ==
              std::vector<std::uint32_t>{6},
          "same-stream producer order was not preserved");
  std::cout << "[PASS] delayed producer ready-event cross/same-stream + "
               "no-wait negative seam\n";
}

void test_device_dispatch_handle_lifetime() {
  const auto config = make_config(1, 1, 8, 3, 4);
  const weftmoe::RankRoutingInput routes{2, {7, 1, 4, 0, 6, 2}};
  auto execution = weftmoe::cuda::make_cuda_execution("handle-exec", 0);
  weftmoe::cuda::DeviceDispatchHandle retained;
  {
    auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
    auto session = weftmoe::cuda::make_nccl_session(
        "handle-lifetime", 1, 0, 6100, 0, bootstrap);
    auto coordinator =
        weftmoe::cuda::testing::make_unpooled_coordinator_for_testing(
            session, execution);
    auto result = coordinator.run({config, routes, 10});
    auto first_copy = result.dispatch();
    retained = first_copy;
    auto moved = std::move(first_copy);
    require(!first_copy && moved, "device handle move state is unsafe");
    const auto &identity = moved.identity();
    const auto expectation = weftmoe::cuda::DeviceDispatchViewExpectation{
        6100, 0, 1, routes.num_tokens, config.top_k,
        identity.config_digest_word0, identity.config_digest_word1};
    auto consumer = moved.acquire_consumer(execution, expectation);
    require(consumer && consumer.state() ==
                            weftmoe::cuda::DeviceDispatchConsumerState::Acquired,
            "owning device consumer was not acquired");
    consumer.finish();
  }
  require(static_cast<bool>(retained),
          "device handle did not outlive coordinator/result");
  const auto actual = weftmoe::cuda::download_device_metadata_for_validation(
      retained, execution);
  const auto expected = weftmoe::build_notify_dispatch_for_testing(
      config, {routes}, "handle-lifetime", 6100);
  require(weftmoe::serialize_dispatch_metadata(actual) ==
              weftmoe::serialize_dispatch_metadata(expected.front()),
          "retained device handle metadata changed after owner destruction");
  bool mismatch_failed = false;
  try {
    (void)retained.acquire_consumer(
        execution,
        {6101, 0, 1, routes.num_tokens, config.top_k,
         retained.identity().config_digest_word0,
         retained.identity().config_digest_word1});
  } catch (const weftmoe::NotifyDispatchError &) {
    mismatch_failed = true;
  }
  require(mismatch_failed,
          "cross-generation device view mismatch did not fail closed");
  auto second_execution =
      weftmoe::cuda::make_cuda_execution("handle-consumer-stream-2", 0);
  const auto retained_id = retained.identity();
  const weftmoe::cuda::DeviceDispatchViewExpectation retained_expectation{
      retained_id.generation, retained_id.calling_rank, retained_id.world_size,
      retained_id.num_tokens, retained_id.top_k,
      retained_id.config_digest_word0, retained_id.config_digest_word1};
  auto first_stream_consumer =
      retained.acquire_consumer(execution, retained_expectation);
  auto second_stream_consumer =
      retained.acquire_consumer(second_execution, retained_expectation);
  weftmoe::cuda::testing::submit_dispatch_probe(first_stream_consumer, 10000);
  weftmoe::cuda::testing::submit_dispatch_probe(second_stream_consumer, 10000);
  retained = {};
  (void)weftmoe::cuda::testing::finish_dispatch_probe(first_stream_consumer);
  (void)weftmoe::cuda::testing::finish_dispatch_probe(second_stream_consumer);
  weftmoe::cuda::DeviceDispatchHandle empty;
  bool empty_failed = false;
  try {
    (void)empty.sizes();
  } catch (const weftmoe::NotifyDispatchError &) {
    empty_failed = true;
  }
  require(empty_failed, "empty device handle accessor did not fail closed");
  std::cout << "[PASS] immutable device handle copy/move/lifetime/consumer checks\n";
}

void test_resource_pool_and_cuda_api_audit() {
  const auto config = make_config(1, 1, 16, 4, 7);
  const auto routes = seeded_inputs(1, 16, 4, 0xabcddcbaULL).front();
  auto execution = weftmoe::cuda::make_cuda_execution("pool-exec", 0);
  auto bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
  auto base = weftmoe::cuda::make_nccl_session(
      "pool-base", 1, 0, 6200, 0, bootstrap);
  auto pool = weftmoe::cuda::make_device_notify_resource_pool(
      base, execution,
      {config, 10, 4096, 2});
  const auto initial_usage = pool.usage();
  require(initial_usage.slot_count == 2 && initial_usage.slots_in_use == 0 &&
              initial_usage.device_bytes != 0 &&
              initial_usage.pinned_host_bytes != 0 &&
              initial_usage.event_count == 30,
          "session resource pool capacity/accounting mismatch");

  auto device_routes = weftmoe::cuda::upload_cuda_routes(
      routes, config.top_k,
      weftmoe::make_rank_collective_context(base.session(), 0,
                                            execution.token()),
      execution);
  auto device_gates = make_unit_gates(
      device_routes,
      weftmoe::make_rank_collective_context(base.session(), 0,
                                            execution.token()),
      execution);
  const auto exact_routes = [&](const weftmoe::cuda::NcclSession &session) {
    return weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
        device_routes,
        weftmoe::make_rank_collective_context(session.session(), 0,
                                              execution.token()),
        execution);
  };
  const auto exact_request = [&](const weftmoe::cuda::NcclSession &session) {
    auto routes_for_generation = exact_routes(session);
    auto gates_for_generation =
        weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
            device_gates, routes_for_generation,
            weftmoe::make_rank_collective_context(
                session.session(), 0, execution.token()),
            execution);
    return weftmoe::cuda::DeviceNotifyGenerationRequest{
        config, std::move(routes_for_generation),
        std::move(gates_for_generation), 10};
  };
  {
    auto next = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6201);
    auto next_routes = exact_routes(next);
    bool stale_gate_failed = false;
    try {
      weftmoe::cuda::DeviceGateProvenanceBridge::validate(
          device_gates, next_routes,
          weftmoe::make_rank_collective_context(
              next.session(), 0, execution.token()),
          execution);
    } catch (const weftmoe::NotifyDispatchError &) {
      stale_gate_failed = true;
    }
    require(stale_gate_failed,
            "generation-6200 gate lease was accepted at generation 6201");
  }
  for (std::uint64_t generation = 6201; generation <= 6210; ++generation) {
    bool rejected = false;
    try {
      auto mismatched = weftmoe::cuda::make_nccl_generation(
          base, "pool-base", generation);
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          mismatched, execution, pool);
      (void)coordinator.run_device(
          {config, device_routes, device_gates, 10});
    } catch (const weftmoe::NotifyDispatchError &error) {
      const std::string message = error.what();
      rejected = message.find("field=generation") != std::string::npos &&
                 message.find("expected=" + std::to_string(generation)) !=
                     std::string::npos &&
                 message.find("actual=6200") != std::string::npos;
    }
    require(rejected,
            "generation-6200 route lease was accepted by a later generation");
    const auto rebound = exact_routes(weftmoe::cuda::make_nccl_generation(
        base, "pool-base", generation));
    require(rebound.route_generation() == generation &&
                device_routes.route_generation() == 6200,
            "private exact-generation reissue mutated or forged provenance");
  }
  const auto before = weftmoe::cuda::testing::cuda_api_audit_snapshot();
  weftmoe::cuda::DeviceDispatchHandle retained;
  for (std::uint64_t iteration = 0; iteration < 10; ++iteration) {
    auto session = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6201 + iteration);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    auto result = coordinator.run_device(exact_request(session));
    require(result.dispatch().audit().bulk_route_d2h == 0 &&
                result.dispatch().audit().bulk_collective_d2h == 0 &&
                result.dispatch().audit().host_materializations == 0 &&
                result.dispatch().audit().bulk_metadata_h2d == 0,
            "pooled device generation used bulk host staging");
    if (iteration >= 2U) {
      const auto audit = result.audit();
#if WEFTMOE_DEBUG_VALIDATE
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
  const auto after = weftmoe::cuda::testing::cuda_api_audit_snapshot();
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

#if !WEFTMOE_DEBUG_VALIDATE
  const auto require_base_recovery = [&](std::uint64_t generation,
                                         const char *failure) {
    auto recovery_session = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", generation);
    weftmoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    auto recovered = recovery.run_device(exact_request(recovery_session));
    const auto audit = recovery.audit();
    require(recovered && audit.fused_notify_starts == 1U &&
                audit.control_starts == 0U && audit.config_starts == 0U &&
                audit.expert_count_starts == 0U &&
                audit.packet_summary_starts == 0U,
            failure);
    const auto actual = weftmoe::serialize_dispatch_metadata(
        weftmoe::cuda::download_device_metadata_for_validation(
            recovered.dispatch(), execution));
    const auto expected = weftmoe::build_notify_dispatch_for_testing(
        config, {routes}, "pool-base", generation);
    require(actual == weftmoe::serialize_dispatch_metadata(expected.front()),
            "release protocol recovery metadata/hash mismatch");
  };

  // A predictable route error does not change the cached static layout.  It
  // must therefore be dataized in the fixed-width fused record instead of
  // returning locally before the collective.
  auto invalid_session = weftmoe::cuda::make_nccl_generation(
      base, "pool-base", 6299);
  weftmoe::cuda::CudaNotifyGenerationCoordinator invalid_coordinator(
      invalid_session, execution, pool);
  auto invalid_routes = routes;
  invalid_routes.selected_experts.front() =
      static_cast<std::int32_t>(config.num_experts);
  bool fused_status_failed = false;
  try {
    (void)invalid_coordinator.run({config, invalid_routes, 10});
  } catch (const weftmoe::NotifyDispatchError &error) {
    fused_status_failed =
        error.code() == weftmoe::NotifyErrorCode::ControlProtocolFailure;
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
  auto changed_session = weftmoe::cuda::make_nccl_generation(
      base, "pool-base", 6301);
  weftmoe::cuda::CudaNotifyGenerationCoordinator changed_coordinator(
      changed_session, execution, pool);
  auto changed_config = config;
  changed_config.top_k = changed_config.num_experts + 1U;
  bool changed_config_failed = false;
  try {
    (void)changed_coordinator.run({changed_config, routes, 10});
  } catch (const weftmoe::NotifyDispatchError &error) {
    changed_config_failed =
        error.code() == weftmoe::NotifyErrorCode::ControlProtocolFailure;
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

  auto second_session = weftmoe::cuda::make_nccl_generation(
      base, "pool-base", 6310);
  weftmoe::cuda::CudaNotifyGenerationCoordinator second_coordinator(
      second_session, execution, pool);
  auto second = second_coordinator.run_device(exact_request(second_session));
  require(pool.usage().slots_in_use == 2,
          "two live handles did not occupy two exclusive pool slots");
  bool exhausted = false;
  try {
    auto third_session = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6311);
    weftmoe::cuda::CudaNotifyGenerationCoordinator third_coordinator(
        third_session, execution, pool);
    (void)third_coordinator.run_device(exact_request(third_session));
  } catch (const weftmoe::NotifyDispatchError &error) {
    exhausted = error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
  }
  require(exhausted, "pool exhaustion did not fail with ResourceExhausted");
  second = {};
  require(pool.usage().slots_in_use == 1,
          "last handle destruction did not return its slot");
  weftmoe::cuda::testing::inject_next_nccl_fault(
      weftmoe::cuda::testing::NcclFaultPoint::ControlStart);
  bool injected_failed = false;
  try {
    auto failed_session = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6312);
    weftmoe::cuda::CudaNotifyGenerationCoordinator failed_coordinator(
        failed_session, execution, pool);
    (void)failed_coordinator.run_device(exact_request(failed_session));
  } catch (const weftmoe::NotifyDispatchError &) {
    injected_failed = true;
  }
  require(injected_failed && pool.usage().slots_in_use == 1,
          "failed generation leaked or published a pool slot");
  auto recovery_session = weftmoe::cuda::make_nccl_generation(
      base, "pool-base", 6313);
  weftmoe::cuda::CudaNotifyGenerationCoordinator recovery_coordinator(
      recovery_session, execution, pool);
  auto recovered = recovery_coordinator.run_device(
      exact_request(recovery_session));
  require(recovered && pool.usage().slots_in_use == 2,
          "released pool slot was not reusable");

  const auto actual = weftmoe::cuda::download_device_metadata_for_validation(
      recovered.dispatch(), execution);
  const auto expected = weftmoe::build_notify_dispatch_for_testing(
      config, {routes}, "pool-base", 6313);
  require(weftmoe::serialize_dispatch_metadata(actual) ==
              weftmoe::serialize_dispatch_metadata(expected.front()),
          "pooled device input metadata mismatch");
  auto ownerless_handle = recovered.dispatch();
  recovered = {};
  retained = {};
  pool = {};
  require(weftmoe::serialize_dispatch_metadata(
              weftmoe::cuda::download_device_metadata_for_validation(
                  ownerless_handle, execution)) ==
              weftmoe::serialize_dispatch_metadata(expected.front()),
          "device handle did not safely outlive the public pool owner");
  ownerless_handle = {};

  for (const auto slot_count : {1U, 3U}) {
    auto sized_pool = weftmoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, slot_count});
    std::vector<weftmoe::cuda::ValidatedCudaGeneration> live;
    for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
      const auto generation = 6400U + slot_count * 10U + slot;
      auto session = weftmoe::cuda::make_nccl_generation(
          base, "pool-base", generation);
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      live.push_back(coordinator.run_device(exact_request(session)));
    }
    require(sized_pool.usage().slots_in_use == slot_count,
            "pool size 1/N did not retain every live slot");
    bool sized_exhausted = false;
    try {
      auto session = weftmoe::cuda::make_nccl_generation(
          base, "pool-base",
          6500U + slot_count);
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      (void)coordinator.run_device(exact_request(session));
    } catch (const weftmoe::NotifyDispatchError &error) {
      sized_exhausted =
          error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
    }
    require(sized_exhausted,
            "pool size 1/N did not reject one generation beyond capacity");
    live.clear();
    require(sized_pool.usage().slots_in_use == 0,
            "pool size 1/N did not release all final handle leases");
    {
      auto session = weftmoe::cuda::make_nccl_generation(
          base, "pool-base",
          6600U + slot_count);
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      auto reused = coordinator.run_device(exact_request(session));
      require(static_cast<bool>(reused),
              "pool size 1/N released slot was not reusable");
    }
    require(sized_pool.usage().slots_in_use == 0,
            "pool size 1/N reuse leaked a lease");
  }

  {
    auto stale_pool = weftmoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 1});
    auto generation = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6700);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
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
    } catch (const weftmoe::NotifyDispatchError &error) {
      consumer_exhausted =
          error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
    }
    require(consumer_exhausted,
            "consumer event pool exhaustion did not fail closed");
    auto moved_consumer = std::move(consumer);
    require(!consumer && moved_consumer,
            "consumer lease move/moved-from state is unsafe");
    moved_consumer = std::move(moved_consumer);
    require(moved_consumer &&
                moved_consumer.state() ==
                    weftmoe::cuda::DeviceDispatchConsumerState::Acquired,
            "consumer self-move corrupted the owning lease");
    weftmoe::cuda::testing::submit_dispatch_probe(moved_consumer, 1000000);
    require(moved_consumer.state() ==
                weftmoe::cuda::DeviceDispatchConsumerState::Submitted,
            "consumer probe was not submitted");
    handle = {};
    gen1 = {};
    require(stale_pool.usage().slots_in_use == 1,
            "submitted consumer did not retain generation slot");
    bool stale_exhausted = false;
    try {
      auto blocked_generation = weftmoe::cuda::make_nccl_generation(
          base, "pool-base", 6701);
      weftmoe::cuda::CudaNotifyGenerationCoordinator blocked(
          blocked_generation, execution, stale_pool);
      (void)blocked.run_device(exact_request(blocked_generation));
    } catch (const weftmoe::NotifyDispatchError &error) {
      stale_exhausted =
          error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
    }
    require(stale_exhausted,
            "pool reused slot while stale consumer remained live");
    const auto stale_value =
        weftmoe::cuda::testing::finish_dispatch_probe(moved_consumer);
    require(stale_value == expected.front()->outgoing.route_dst_row.front(),
            "old consumer did not read generation-1 canonical row");
    require(stale_pool.usage().slots_in_use == 0,
            "finished consumer did not release final slot lease");
    auto recovered_generation = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6702);
    weftmoe::cuda::CudaNotifyGenerationCoordinator recovered_coordinator(
        recovered_generation, execution, stale_pool);
    auto gen2 = recovered_coordinator.run_device(
        exact_request(recovered_generation));
    require(gen2 && stale_pool.usage().slots_in_use == 1,
            "slot was not reusable after final consumer completion");
  }

  {
    auto quarantine_pool = weftmoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 2});
    auto generation = weftmoe::cuda::make_nccl_generation(
        base, "pool-base", 6800);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, quarantine_pool);
    auto result = coordinator.run_device(exact_request(generation));
    auto handle = result.dispatch();
    const auto &id = handle.identity();
    auto consumer = handle.acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    weftmoe::cuda::testing::submit_dispatch_probe(consumer, 1000);
    weftmoe::cuda::testing::inject_next_consumer_fault(
        weftmoe::cuda::testing::ConsumerFaultPoint::EventAndStreamSynchronize);
    bool finish_failed = false;
    try {
      consumer.finish();
    } catch (const weftmoe::NotifyDispatchError &) {
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
    } catch (const weftmoe::NotifyDispatchError &) {
      handle_failed = true;
    }
    require(handle_failed,
            "quarantined generation handle accepted another consumer");
    bool next_failed = false;
    try {
      auto next = weftmoe::cuda::make_nccl_generation(
          base, "pool-base", 6801);
      weftmoe::cuda::CudaNotifyGenerationCoordinator blocked(
          next, execution, quarantine_pool);
      (void)blocked.run_device(exact_request(next));
    } catch (const weftmoe::NotifyDispatchError &error) {
      next_failed = error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
    }
    require(next_failed,
            "quarantined slot was allocated to a later generation");
  }

  for (const auto fault : {
           weftmoe::cuda::testing::ConsumerFaultPoint::EventRecord,
           weftmoe::cuda::testing::ConsumerFaultPoint::EventSynchronize}) {
    auto event_pool = weftmoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 2});
    auto generation = weftmoe::cuda::make_nccl_generation(
        base, "pool-base",
        fault == weftmoe::cuda::testing::ConsumerFaultPoint::EventRecord
            ? 6810
            : 6811);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, event_pool);
    auto result = coordinator.run_device(exact_request(generation));
    const auto &id = result.dispatch().identity();
    auto consumer = result.dispatch().acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    weftmoe::cuda::testing::inject_next_consumer_fault(fault);
    bool failed = false;
    try {
      weftmoe::cuda::testing::submit_dispatch_probe(consumer, 1000);
      consumer.finish();
    } catch (const weftmoe::NotifyDispatchError &) {
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
void run_cached_backward_rank(
    std::uint32_t rank, std::uint32_t nodes, std::uint32_t world,
    std::uint32_t experts, std::uint32_t top_k, std::uint64_t max_tokens,
    const weftmoe::RuntimeConfig &config,
    const weftmoe::RankRoutingInput &routes,
    const weftmoe::cuda::NcclSession &base,
    const weftmoe::cuda::CudaExecution &execution, bool validation) {
  constexpr std::uint64_t hidden = weftmoe::cuda::forward_tuning::kMainHidden;
  constexpr std::uint64_t intermediate = weftmoe::cuda::forward_tuning::kMainIntermediate;
  require((nodes == 1U && (world == 1U || world == 2U || world == 8U)) ||
              (nodes == 2U && world == 16U),
          "cached Backward requires local 1/2/8 GPUs or GIN 2x8 GPUs");
  require(experts % world == 0U && routes.num_tokens == max_tokens,
          "cached Backward fixture shape is inconsistent");

  auto generation = weftmoe::cuda::make_nccl_generation(
      base, "cuda-cached-backward", 701U);
  auto context = weftmoe::make_rank_collective_context(
      generation.session(), rank, execution.token());
  auto notify_pool = weftmoe::cuda::make_device_notify_resource_pool(
      base, execution, {config, max_tokens, 8192, 3U});
  auto route_storage = weftmoe::cuda::upload_cuda_routes(
      routes, top_k, context, execution);
  const auto route_gates = fixed_random_selected_gates(
      max_tokens, top_k, rank, 0x4d4547414d4f4547ULL);
  auto gate_storage = weftmoe::cuda::testing::upload_fp32_gates_for_testing(
      route_gates, route_storage, context, execution);

  weftmoe::cuda::ActivationTransportPoolConfig pool_config;
  pool_config.max_tokens_per_rank = max_tokens;
  pool_config.max_hidden_size = hidden;
  pool_config.max_top_k = top_k;
  pool_config.max_world_size = world;
  pool_config.max_nodes = nodes;
  pool_config.max_destination_rows_per_rank =
      std::max<std::uint64_t>(
          max_tokens * (top_k + 1ULL) * (world == 2U ? 2ULL : 1ULL),
          static_cast<std::uint64_t>(world) * top_k);
  pool_config.channels = 1U;
  pool_config.ring_slots_per_source_channel = 1U;
  pool_config.max_inflight_generations = 1U;
  pool_config.max_intermediate_size = intermediate;
  pool_config.max_owned_experts = experts / world;
  pool_config.enable_backward_state = true;
  pool_config.enable_gin_rdma = nodes == 2U;
  auto activation_pool = weftmoe::cuda::make_activation_transport_pool(
      generation.session(), execution, pool_config);

  std::vector<weftmoe::transport::BFloat16> activation_host(
      static_cast<std::size_t>(max_tokens * hidden));
  std::vector<weftmoe::transport::BFloat16> dy_host(activation_host.size());
  for (std::size_t index = 0; index < activation_host.size(); ++index) {
    activation_host[index] = weftmoe::transport::float_to_bfloat16(
        static_cast<float>(static_cast<int>((index + rank * 13U) % 127U) -
                           63) /
        64.0F);
    dy_host[index] = weftmoe::transport::float_to_bfloat16(
        static_cast<float>(static_cast<int>((index * 29U + rank * 17U) %
                                            113U) -
                           56) /
        64.0F);
  }
  auto activations =
      weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activation_host, max_tokens, hidden, context, execution,
          activation_pool);
  auto output_gradients =
      weftmoe::cuda::BackwardValidationBridge::upload_for_testing(
          dy_host, max_tokens, hidden, context, execution);
  auto weights =
      weftmoe::cuda::testing::make_deterministic_forward_weights_for_testing(
          experts / world, rank * (experts / world), hidden, intermediate,
          generation.session(), execution);
  weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
      generation, execution, notify_pool);
  auto notify = coordinator.run_device(
      {config, route_storage, gate_storage, max_tokens});
  const auto progress = [&](const char *phase) {
    if (std::getenv("WEFTMOE_BACKWARD_PROGRESS") != nullptr)
      std::cerr << "BACKWARD_PROGRESS rank=" << rank << " phase=" << phase
                << std::endl;
  };
  progress("notify_done");
  weftmoe::cuda::ActivationTransportRunOptions forward_options;
  forward_options.deadline_after = std::chrono::seconds(300);
  auto forward = [&] {
    if (nodes == 2U) {
      weftmoe::cuda::GinRdmaForwardMegaKernel forward_kernel(
          generation, execution, activation_pool);
      progress("forward_constructed");
    return forward_kernel.enqueue(notify.dispatch(), activations, weights,
                                {hidden, intermediate, true},
                                forward_options);
    }
    weftmoe::cuda::LocalCudaP2pForwardMegaKernel forward_kernel(
        generation, execution, activation_pool);
    progress("forward_constructed");
    return forward_kernel.enqueue(notify.dispatch(), activations, weights,
                              {hidden, intermediate, true}, forward_options);
  }();
  progress("forward_done");

  const auto rounds_text = std::getenv("WEFTMOE_BACKWARD_ROUNDS");
  const auto rounds = rounds_text
                          ? static_cast<std::uint32_t>(std::stoul(rounds_text))
                          : (validation ? 2U : 30U);
  const auto warmups_text = std::getenv("WEFTMOE_BACKWARD_WARMUPS");
  const auto warmups = warmups_text
                           ? static_cast<std::uint32_t>(
                                 std::stoul(warmups_text))
                           : (validation ? 0U : 10U);
  require(rounds > warmups, "Backward requires at least one measured round");
  std::vector<double> wall_samples;
  std::vector<double> total_samples;
  std::vector<double> prepare_samples;
  std::vector<double> compute_samples;
  std::vector<double> reduce_samples;
  std::vector<double> transport_samples;
  std::vector<double> dispatch_samples;
  std::vector<double> dab_samples;
  std::vector<double> dx_samples;
  std::vector<double> dw2_samples;
  std::vector<double> dw1_samples;
  std::vector<double> combine_samples;
  std::vector<double> overlap_samples;
  weftmoe::cuda::DeviceBackwardHandle last;
  weftmoe::cuda::BackwardMegaKernel backward;
  for (std::uint32_t round = 0; round < rounds; ++round) {
    progress("backward_enter");
    const auto begin = Clock::now();
    auto output = backward.enqueue(
        forward, output_gradients, {hidden, intermediate}, {});
    if (validation) {
      bool rejected = false;
      try {
        (void)backward.enqueue(forward, output_gradients, {hidden, intermediate}, {});
      } catch (const weftmoe::NotifyDispatchError &error) {
        rejected = error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
      }
      require(rejected, "overlapping Backward reused the same workspace");
    }
    if (validation) {
      cudaStream_t output_stream = nullptr;
      require(cudaStreamCreateWithFlags(&output_stream, cudaStreamNonBlocking) == cudaSuccess,
              "create async consumer stream");
      output.wait_on(output_stream);
      require(cudaStreamSynchronize(output_stream) == cudaSuccess,
              "wait async output dependency");
      require(cudaStreamDestroy(output_stream) == cudaSuccess,
              "destroy async consumer stream");
    }
    output.wait();
    if (validation) output.wait();  // completion is idempotent
    const auto end = Clock::now();
    if (round >= warmups) {
      wall_samples.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      total_samples.push_back(output.timings().total_us);
      prepare_samples.push_back(output.timings().prepare_us);
      compute_samples.push_back(output.timings().compute_us);
      reduce_samples.push_back(output.timings().reduce_us);
      transport_samples.push_back(output.timings().transport_us);
      dispatch_samples.push_back(output.timings().dispatch_us);
      dab_samples.push_back(output.timings().dab_us);
      dx_samples.push_back(output.timings().dx_us);
      dw2_samples.push_back(output.timings().dw2_us);
      dw1_samples.push_back(output.timings().dw1_us);
      combine_samples.push_back(output.timings().combine_us);
      overlap_samples.push_back(output.timings().dw_combine_overlap_us);
    }
    last = std::move(output);
    progress("backward_done");
  }

  bool repeat_equal = true;
  std::uint64_t dx_hash = 0ULL;
  std::uint64_t dgate_hash = 0ULL;
  std::uint64_t dw1_hash = 0ULL;
  std::uint64_t dw2_hash = 0ULL;
  if (validation) {
    progress("validation_download_enter");
    auto first = weftmoe::cuda::BackwardValidationBridge::download(
        last, execution);
    last = backward.run(forward, output_gradients,
                        {hidden, intermediate}, {});
    progress("validation_repeat_done");
    auto repeated = weftmoe::cuda::BackwardValidationBridge::download(
        last, execution);
    const auto equal_bytes = [](const auto &left, const auto &right) {
      return left.size() == right.size() &&
             (left.empty() ||
              std::memcmp(left.data(), right.data(),
                          left.size() * sizeof(left.front())) == 0);
    };
    repeat_equal = equal_bytes(first.dx, repeated.dx) &&
                   equal_bytes(first.dgate, repeated.dgate) &&
                   equal_bytes(first.dw1, repeated.dw1) &&
                   equal_bytes(first.dw2, repeated.dw2);
    dx_hash = hash_bf16_indexed(repeated.dx);
    dgate_hash = hash_fp32(repeated.dgate);
    dw1_hash = hash_bf16_indexed(repeated.dw1);
    dw2_hash = hash_bf16_indexed(repeated.dw2);
    if (const auto *artifact_dir = std::getenv("WEFTMOE_BACKWARD_OUTPUT_DIR")) {
      const auto prefix = std::string(artifact_dir) + "/rank-" +
                          std::to_string(rank) + "-";
      const auto write = [&](const std::string &name, const auto &values) {
        std::ofstream output(prefix + name,
                             std::ios::binary | std::ios::trunc);
        require(output.good(), "failed opening Backward artifact " + name);
        output.write(reinterpret_cast<const char *>(values.data()),
                     static_cast<std::streamsize>(values.size() *
                                                  sizeof(values.front())));
        require(output.good(), "failed writing Backward artifact " + name);
      };
      const auto retained =
          weftmoe::cuda::download_forward_backward_state_for_validation(
              forward, execution);
      write("routes.i32", routes.selected_experts);
      write("gates.f32", route_gates);
      write("source-x.bf16", activation_host);
      write("source-dy.bf16", dy_host);
      write("canonical-x.bf16", retained.canonical_x);
      write("canonical-ab.bf16", retained.fc1_ab);
      write("dx.bf16", repeated.dx);
      write("dgate.f32", repeated.dgate);
      write("dw1.bf16", repeated.dw1);
      write("dw2.bf16", repeated.dw2);
    }
    progress("validation_done");
  }
  require(repeat_equal, "cached Backward outputs changed across executions");

  const auto emit = [](const std::vector<double> &values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0U)
        std::cout << ',';
      std::cout << values[index];
    }
  };
  const auto usage = activation_pool.usage();
  std::cout << "BACKWARD_RESULT rank=" << rank << " world=" << world
            << " mode=" << (validation ? "validation" : "benchmark")
            << " backend=" << (nodes == 2U ? "gin-rdma" : "local-p2p")
            << " tokens=" << max_tokens << " experts=" << experts
            << " top_k=" << top_k << " hidden=" << hidden
            << " intermediate=" << intermediate
            << " incoming_rows="
            << notify.dispatch().sizes().incoming_row_count
            << " gemm_launches=" << last.audit().gemm_launches
            << " dispatch_peer_bytes="
            << last.audit().actual_dispatch_peer_bytes
            << " return_peer_bytes=" << last.audit().actual_return_peer_bytes
            << " gin_return_puts=" << last.audit().gin_return_put_count
            << " gin_return_payload_bytes="
            << last.audit().gin_return_payload_bytes
            << " workspace_bytes=" << last.audit().workspace_bytes
            << " pool_bytes=" << usage.device_bytes
            << " repeat_equal=" << repeat_equal << " dx_hash=" << dx_hash
            << " dgate_hash=" << dgate_hash << " dw1_hash=" << dw1_hash
            << " dw2_hash=" << dw2_hash << " wall=";
  emit(wall_samples);
  std::cout << " total=";
  emit(total_samples);
  std::cout << " prepare=";
  emit(prepare_samples);
  std::cout << " compute=";
  emit(compute_samples);
  std::cout << " reduce=";
  emit(reduce_samples);
  std::cout << " transport=";
  emit(transport_samples);
  std::cout << " dispatch=";
  emit(dispatch_samples);
  std::cout << " dab=";
  emit(dab_samples);
  std::cout << " dx=";
  emit(dx_samples);
  std::cout << " dw2=";
  emit(dw2_samples);
  std::cout << " dw1=";
  emit(dw1_samples);
  std::cout << " combine=";
  emit(combine_samples);
  std::cout << " dw_combine_overlap=";
  emit(overlap_samples);
  std::cout << std::endl;
}
void run_rank(std::uint32_t rank, std::uint32_t nodes,
              std::uint32_t gpus_per_node, std::uint32_t local_rank,
              std::uint64_t seed, int bootstrap_input_fd,
              int bootstrap_output_fd,
              const std::string &scenario) {
  const auto world = nodes * gpus_per_node;
  constexpr std::array<std::uint32_t, 12> kCrossEpExperts{
      1U, 68U, 135U, 202U, 269U, 336U,
      145U, 289U, 10U, 90U, 230U, 350U};
  const auto benchmark_p2p = scenario == "activation_p2p_bench";
  const auto benchmark_integrated =
      scenario == "activation_integrated_bench";
  const auto gin_forward_smoke = scenario == "gin_forward_smoke";
  const auto gin_forward_benchmark = scenario == "gin_forward_bench";
  const auto socket_forward_benchmark =
      scenario == "socket_forward_bench";
  const auto gin_backward_smoke = scenario == "gin_backward_smoke";
  const auto gin_backward_benchmark = scenario == "gin_backward_bench";
  const auto backward_validation =
      scenario == "backward_validate" || gin_backward_smoke;
  const auto backward_benchmark =
      scenario == "backward_bench" || gin_backward_benchmark;
  const auto forward_smoke = scenario == "forward_smoke" || gin_forward_smoke;
  const auto benchmark_forward =
      scenario == "forward_bench" || gin_forward_benchmark ||
      socket_forward_benchmark;
  const auto benchmark_shape = benchmark_p2p || benchmark_integrated ||
                               benchmark_forward || backward_benchmark;
  auto experts = std::max<std::uint32_t>(16, world * 2);
  auto top_k = std::min<std::uint32_t>(6, experts);
  if (const auto *p2p_top_k = std::getenv("WEFTMOE_P2P_TOP_K"))
    top_k = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(std::stoul(p2p_top_k)), experts);
  auto common_config = make_config(nodes, gpus_per_node, experts, top_k, 7);
  auto common_routes = seeded_inputs(world, experts, top_k, seed)[rank];
  std::uint64_t max_tokens = 10;
  // Optional fixed-global-batch fixture: repartition the original eight
  // source shards across EP8/EP16 without changing any token's data.
  std::uint32_t fixture_rank = rank;
  std::uint64_t fixture_token_offset = 0;
  std::uint64_t fixture_tokens = 0;
  if (benchmark_shape) {
    const auto cross_host_benchmark = gin_forward_benchmark ||
                                      socket_forward_benchmark ||
                                      gin_backward_benchmark;
    require((!cross_host_benchmark && nodes == 1 &&
             ((benchmark_p2p && world == 8) ||
              ((benchmark_forward || backward_benchmark) &&
               (world == 1 || world == 8)) ||
              (benchmark_integrated && (world == 2 || world == 8)))) ||
                (cross_host_benchmark && nodes == 2 &&
                 (world == 2 || world == 16)),
            "CUDA benchmark requires local 1/2/8 GPUs or GIN 2x1/2x8 GPUs");
    experts = std::getenv("WEFTMOE_BENCH_EXPERTS")
                  ? std::stoul(std::getenv("WEFTMOE_BENCH_EXPERTS")) : 384U;
    require(experts > 0 && experts % world == 0, "experts must divide EP");
    const auto benchmark_top_k =
        std::getenv("WEFTMOE_INTEGRATED_TOP_K")
            ? std::getenv("WEFTMOE_INTEGRATED_TOP_K")
            : std::getenv("WEFTMOE_P2P_TOP_K");
    top_k = benchmark_top_k
                ? static_cast<std::uint32_t>(std::stoul(benchmark_top_k))
                : 6U;
    require(top_k == 1 || top_k == 6 || top_k == 8,
            "integrated benchmark top-k must be 1, 6, or 8");
    const auto bench_tokens_text = std::getenv("WEFTMOE_P2P_BENCH_TOKENS");
    max_tokens = bench_tokens_text ? std::stoull(bench_tokens_text) : 8192U;
    fixture_tokens = max_tokens;
    if (std::getenv("WEFTMOE_FORWARD_REPARTITION_EP8")) {
      require(benchmark_forward && (world == 8U || world == 16U) &&
                  max_tokens > 0U,
              "EP repartition fixture requires Forward EP8/EP16");
      fixture_tokens = max_tokens * (world / 8U);
      fixture_rank = rank / (world / 8U);
      fixture_token_offset = (rank % (world / 8U)) * max_tokens;
    }
    common_config = make_config(nodes, gpus_per_node, experts, top_k, 64);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      common_config.topology.expert_to_rank[expert] =
          expert / (experts / world);
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    common_routes.selected_experts.reserve(
        static_cast<std::size_t>(max_tokens * top_k));
    for (std::uint64_t token = 0; token < max_tokens; ++token) {
      const auto base = static_cast<std::uint32_t>(
          ((token + fixture_token_offset) * 17U + fixture_rank * 31U) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            (base + slot * 53U) % experts));
    }
    // Diagnostic only: hold per-expert row counts and GEMM dimensions fixed
    // while changing where tokens originate. This changes routing semantics
    // and must not be used as an output-equivalence fixture.
    if (const char *scope = std::getenv("WEFTMOE_FORWARD_DIAGNOSTIC_ROUTE_SCOPE")) {
      require(gin_forward_benchmark && world == 16U && top_k == 1U &&
                  max_tokens % experts == 0U,
              "route-scope diagnostic requires EP16 top-1 and whole expert cycles");
      const std::string mode(scope);
      require(mode == "self" || mode == "node" || mode == "remote",
              "diagnostic route scope must be self, node, or remote");
      const auto width = mode == "self" ? experts / world : experts / nodes;
      const auto owner = mode == "self" ? rank :
          (mode == "node" ? rank / gpus_per_node : 1U - rank / gpus_per_node);
      for (auto &expert : common_routes.selected_experts)
        expert = static_cast<std::int32_t>(
            owner * width + static_cast<std::uint32_t>(expert) % width);
    }
  }
  if (forward_smoke) {
    const auto cross_host_megakernel_smoke = gin_forward_smoke;
    require((!cross_host_megakernel_smoke && nodes == 1 &&
             (world == 1 || world == 2 || world == 8)) ||
                (cross_host_megakernel_smoke && nodes == 2 &&
                 (world == 2 || world == 16)),
            "forward smoke requires local 1/2/8 GPUs or cross-host "
            "2x1/2x8 GPUs");
    // Cross-topology fixture: all EP sizes retain the same 384 global experts,
    // source activation, selected IDs and gates. Only contiguous ownership
    // changes. The selected experts span the complete ownership range.
    experts = 384U;
    max_tokens = 2;
    common_config = make_config(nodes, gpus_per_node, experts, top_k, 64);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      common_config.topology.expert_to_rank[expert] =
          expert / (experts / world);
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    for (std::uint64_t token = 0; token < max_tokens; ++token)
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            kCrossEpExperts[token * 6U + slot]));
  }
  if (scenario == "activation_gin") {
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    common_routes.selected_experts.reserve(
        static_cast<std::size_t>(max_tokens * top_k));
    for (std::uint64_t token = 0; token < max_tokens; ++token) {
      const auto base_expert = static_cast<std::uint32_t>(
          (token * 17U + rank * 31U) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            (base_expert + slot * 5U) % experts));
    }
  }
  if (backward_validation) {
    require((!gin_backward_smoke && nodes == 1U &&
             (world == 1U || world == 2U || world == 8U)) ||
                (gin_backward_smoke && nodes == 2U && world == 16U),
            "Backward validation requires local 1/2/8 GPUs or GIN 2x8 "
            "GPUs");
    // Keep at least the small topology already exercised by the Forward
    // scheduler; one-expert fixtures do not provide its split-role terminal
    // accounting with a representative task domain.
    experts = std::max<std::uint32_t>(16U, world * 2U);
    const auto *tokens_text = std::getenv("WEFTMOE_BACKWARD_TOKENS");
    max_tokens = tokens_text ? std::stoull(tokens_text) : 8ULL;
    // Forward publishes 64-row Dispatch arrival tiles. Backward later groups
    // and pads those immutable rows into its own 128-row dependency blocks.
    common_config = make_config(nodes, gpus_per_node, experts, top_k, 64U);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      common_config.topology.expert_to_rank[expert] =
          expert / (experts / world);
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    common_routes.selected_experts.reserve(
        static_cast<std::size_t>(max_tokens * top_k));
    for (std::uint64_t token = 0; token < max_tokens; ++token) {
      const auto base = static_cast<std::uint32_t>(
          (token * world + rank) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            (base + slot * 53U) % experts));
    }
  }
  weftmoe::cuda::NcclBootstrapId bootstrap;
  if (rank == 0) {
    require(rank == 0, "only rank 0 may generate bootstrap ID");
    require(bootstrap_output_fd >= 0 && bootstrap_input_fd < 0,
            "rank 0 requires only a bootstrap output descriptor");
    bootstrap = weftmoe::cuda::make_nccl_bootstrap_id();
    weftmoe::cuda::NcclBootstrapOutputChannel channel(bootstrap_output_fd);
    channel.write(bootstrap);
    require(channel.state() == weftmoe::cuda::BootstrapChannelState::Consumed,
            "bootstrap output channel did not reach Consumed");
  } else {
    require(bootstrap_input_fd >= 0 && bootstrap_output_fd < 0,
            "nonzero rank requires only a bootstrap input descriptor");
    weftmoe::cuda::NcclBootstrapInputChannel channel(bootstrap_input_fd);
    bootstrap = channel.read();
    require(channel.state() == weftmoe::cuda::BootstrapChannelState::Consumed,
            "bootstrap input channel did not reach Consumed");
  }
  auto execution = weftmoe::cuda::make_cuda_execution(
      "multinode-exec-" + std::to_string(rank), local_rank);
  auto base = weftmoe::cuda::make_nccl_session(
      "cuda-multinode-base", world, rank, 700, local_rank, bootstrap);
  if (backward_validation || backward_benchmark) {
    run_cached_backward_rank(rank, nodes, world, experts, top_k, max_tokens,
                             common_config, common_routes, base, execution,
                             backward_validation);
    return;
  }
  if (scenario == "gin_probe") {
    const auto capability =
        weftmoe::cuda::probe_activation_transport_capability(base, execution);
    std::cout << "GIN_DISTRIBUTED_PROBE rank=" << rank
              << " world=" << world
              << " headers=" << capability.gin_headers_available
              << " symbols=" << capability.gin_symbols_present
              << " runtime_initializable="
              << capability.gin_runtime_initializable
              << " backend_implemented="
              << capability.gin_backend_implemented
              << " usable=" << capability.gin_usable
              << " reason=" << capability.gin_unavailable_reason
              << std::endl;
    return;
  }
  if (scenario == "gin_put_probe") {
    const auto probe =
        weftmoe::cuda::testing::run_distributed_gin_put_signal_probe(
            base, execution, 1024);
    std::cout << "GIN_PUT_SIGNAL rank=" << rank << " world=" << world
              << " rail_rank=" << probe.rail_rank
              << " rail_world=" << probe.rail_world
              << " local_rank=" << probe.local_rank
              << " bytes=" << probe.payload_bytes
              << " received_hash=" << probe.received_hash
              << " expected_hash=" << probe.expected_hash
              << " match=" << (probe.received_hash == probe.expected_hash)
              << std::endl;
    return;
  }
  if (scenario == "gin_link_bench") {
    const auto *mode_text = std::getenv("WEFTMOE_GIN_BENCH_MODE");
    const auto full = mode_text != nullptr && std::string(mode_text) == "full";
    const auto *shift_text = std::getenv("WEFTMOE_GIN_BENCH_SHIFT");
    const auto shift = shift_text == nullptr
                           ? 0U
                           : static_cast<std::uint32_t>(std::stoul(shift_text));
    const auto *warmup_text = std::getenv("WEFTMOE_GIN_BENCH_WARMUP");
    const auto warmup = warmup_text == nullptr
                            ? 4U
                            : static_cast<std::uint32_t>(std::stoul(warmup_text));
    const auto *measured_text = std::getenv("WEFTMOE_GIN_BENCH_MEASURED");
    const auto measured =
        measured_text == nullptr
            ? 20U
            : static_cast<std::uint32_t>(std::stoul(measured_text));
    const auto *batch_text = std::getenv("WEFTMOE_GIN_BENCH_BATCH");
    const auto batch = batch_text == nullptr
                           ? 32U
                           : static_cast<std::uint32_t>(std::stoul(batch_text));
    const std::vector<std::uint64_t> payloads{
        1ULL * 7168ULL * sizeof(weftmoe::transport::BFloat16),
        2ULL * 7168ULL * sizeof(weftmoe::transport::BFloat16),
        64ULL * 1024ULL,
        64ULL * 7168ULL * sizeof(weftmoe::transport::BFloat16),
        128ULL * 7168ULL * sizeof(weftmoe::transport::BFloat16),
        4ULL * 1024ULL * 1024ULL};
    const auto benchmark =
        weftmoe::cuda::testing::run_distributed_gin_link_benchmark(
            base, execution,
            full ? weftmoe::cuda::testing::GinConnectionMode::Full
                 : weftmoe::cuda::testing::GinConnectionMode::Rail,
            shift, payloads, warmup, measured, batch);
    for (const auto &sample : benchmark.cases) {
      std::cout << "GIN_LINK_BENCH rank=" << rank
                << " mode=" << (full ? "full" : "rail")
                << " shift=" << benchmark.target_shift
                << " local_rank=" << benchmark.local_rank
                << " local_world=" << benchmark.local_world
                << " gin_type=" << benchmark.gin_type
                << " railed_gin_type=" << benchmark.railed_gin_type
                << " payload=" << sample.payload_bytes
                << " puts=" << sample.puts_per_sample
                << " durations_ns=";
      for (std::size_t index = 0; index < sample.duration_ns.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << sample.duration_ns[index];
      }
      std::cout << " received_hash=" << benchmark.received_hash
                << " expected_hash=" << benchmark.expected_hash
                << " match="
                << (benchmark.received_hash == benchmark.expected_hash)
                << std::endl;
    }
    return;
  }
  if (scenario == "gin_team_pool") {
    auto generation = weftmoe::cuda::make_nccl_generation(
        base, "cuda-gin-team-pool", 701);
    weftmoe::cuda::ActivationTransportPoolConfig config;
    config.max_tokens_per_rank = 2;
    config.max_hidden_size = 17;
    config.max_top_k = 2;
    config.max_world_size = world;
    config.max_nodes = nodes;
    config.max_destination_rows_per_rank = world * 4ULL;
    config.channels = 2;
    config.ring_slots_per_source_channel = 2;
    config.max_inflight_generations = 2;
    config.enable_gin_rdma = true;
    auto activation_pool = weftmoe::cuda::make_activation_transport_pool(
        generation.session(), execution, config);
    weftmoe::cuda::GinRdmaActivationTransport transport(
        generation, execution, activation_pool);
    const auto usage = activation_pool.usage();
    require(usage.compute_device_bytes != 0 &&
                usage.gin_registered_bytes != 0 &&
                usage.device_bytes > usage.compute_device_bytes,
            "GIN pool did not separate compute and registered storage");
    std::cout << "GIN_TEAM_POOL rank=" << rank << " world=" << world
              << " nodes=" << nodes
              << " local_world=" << gpus_per_node
              << " local_rank=" << local_rank
              << " compute_bytes=" << usage.compute_device_bytes
              << " registered_bytes=" << usage.gin_registered_bytes
              << " slots=" << config.max_inflight_generations << std::endl;
    return;
  }
  auto pool = weftmoe::cuda::make_device_notify_resource_pool(
      base, execution,
      {common_config, max_tokens, benchmark_shape ? 8192U : 4096U,
       forward_smoke ? 3U : 2U});
  if (benchmark_forward) {
    constexpr std::uint64_t hidden = weftmoe::cuda::forward_tuning::kMainHidden;
    constexpr std::uint64_t intermediate = weftmoe::cuda::forward_tuning::kMainIntermediate;
    const bool retain_backward_state =
        std::getenv("WEFTMOE_FORWARD_BENCH_RETAIN_STATE") != nullptr;
    auto setup_generation = weftmoe::cuda::make_nccl_generation(
        base, "cuda-forward-bench", 701);
    auto setup_context = weftmoe::make_rank_collective_context(
        setup_generation.session(), rank, execution.token());
    auto route_storage = weftmoe::cuda::upload_cuda_routes(
        common_routes, top_k, setup_context, execution);
    const auto fixture_gates = fixed_random_selected_gates(
        fixture_tokens, top_k, fixture_rank, 0x4d4547414d4f4547ULL);
    const std::vector<float> route_gates(
        fixture_gates.begin() + fixture_token_offset * top_k,
        fixture_gates.begin() +
            (fixture_token_offset + common_routes.num_tokens) * top_k);
    auto gate_storage =
        weftmoe::cuda::testing::upload_fp32_gates_for_testing(
            route_gates, route_storage, setup_context, execution);
    weftmoe::cuda::ActivationTransportPoolConfig transport_config;
    transport_config.max_tokens_per_rank = max_tokens;
    transport_config.max_hidden_size = hidden;
    transport_config.max_top_k = top_k;
    transport_config.max_world_size = world;
    transport_config.max_nodes = nodes;
    // The deterministic fixture is close to uniform but integer ownership
    // boundaries can put slightly more than route_count/world rows on a rank.
    transport_config.max_destination_rows_per_rank =
        std::max<std::uint64_t>(
            max_tokens * (top_k + 1U) * (world == 2U ? 2U : 1U),
            static_cast<std::uint64_t>(world) * top_k);
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel = 1;
    transport_config.max_inflight_generations = 1U;
    transport_config.max_intermediate_size = intermediate;
    transport_config.max_owned_experts = experts / world;
    transport_config.enable_backward_state = retain_backward_state;
    transport_config.enable_gin_rdma = gin_forward_benchmark;
    transport_config.enable_socket_fallback = socket_forward_benchmark;
    const auto setup_start = Clock::now();
    auto transport_pool = weftmoe::cuda::make_activation_transport_pool(
        setup_generation.session(), execution, transport_config);
    std::vector<weftmoe::transport::BFloat16> activation_host(
        static_cast<std::size_t>(max_tokens * hidden));
    for (std::size_t index = 0; index < activation_host.size(); ++index)
      activation_host[index] = weftmoe::transport::float_to_bfloat16(
          static_cast<float>(static_cast<int>(
                                 (index + fixture_token_offset * hidden +
                                  fixture_rank * 13U) % 127U) - 63) /
          64.0F);
    auto activation_storage =
        weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activation_host, max_tokens, hidden, setup_context, execution,
            transport_pool);
    const auto experts_per_rank = experts / world;
    const auto activation_hash = hash_bf16_indexed(activation_host);
    const auto route_hash = hash_i32(common_routes.selected_experts);
    const auto gate_hash = hash_fp32(route_gates);
    const auto weight_hash = forward_weight_sample_hash(
        rank * experts_per_rank, experts_per_rank, hidden, intermediate);
    auto weights =
        weftmoe::cuda::testing::make_deterministic_forward_weights_for_testing(
            experts_per_rank, rank * experts_per_rank, hidden, intermediate,
            setup_generation.session(), execution);
    const auto actual_weight_hashes =
        weftmoe::cuda::testing::hash_forward_weight_storage_for_validation(
            weights, execution);
    const auto all_distinct = [](const auto &hashes) {
      for (std::size_t left = 0; left < hashes.size(); ++left)
        for (std::size_t right = left + 1; right < hashes.size(); ++right)
          if (hashes[left].low == hashes[right].low &&
              hashes[left].high == hashes[right].high)
            return false;
      return true;
    };
    require(all_distinct(actual_weight_hashes.sampled_w1_experts),
            "actual W1 device storage is not expert-distinct");
    require(all_distinct(actual_weight_hashes.sampled_w2_experts),
            "actual W2 device storage is not expert-distinct");
    require(all_distinct(actual_weight_hashes.sampled_w2_output_rows),
            "actual W2 device output rows are not distinct");
    std::unique_ptr<weftmoe::cuda::LocalCudaP2pForwardMegaKernel>
        local_forward;
    std::unique_ptr<weftmoe::cuda::GinRdmaForwardMegaKernel> gin_forward;
    std::unique_ptr<weftmoe::cuda::NcclSocketFallbackForwardMegaKernel>
        socket_forward;
    if (gin_forward_benchmark)
      gin_forward =
          std::make_unique<weftmoe::cuda::GinRdmaForwardMegaKernel>(
              setup_generation, execution, transport_pool);
    else if (socket_forward_benchmark)
      socket_forward = std::make_unique<
          weftmoe::cuda::NcclSocketFallbackForwardMegaKernel>(
          setup_generation, execution, transport_pool);
    else
      local_forward =
          std::make_unique<weftmoe::cuda::LocalCudaP2pForwardMegaKernel>(
              setup_generation, execution, transport_pool);
    const auto run_forward = [&](const auto &dispatch,
                                 const auto &activations,
                                 const auto &forward_weights,
                                 const weftmoe::cuda::ForwardMegaKernelConfig
                                     &forward_config,
                                 const weftmoe::cuda::
                                     ActivationTransportRunOptions
                                         &run_options) {
      if (gin_forward)
        return gin_forward->run(dispatch, activations, forward_weights,
                                forward_config, run_options);
      if (socket_forward)
        return socket_forward->run(dispatch, activations, forward_weights,
                                   forward_config, run_options);
      return local_forward->run(dispatch, activations, forward_weights,
                                forward_config, run_options);
    };
    const auto one_time_setup_us =
        std::chrono::duration<double, std::micro>(Clock::now() - setup_start)
            .count();
    const auto rounds_text = std::getenv("WEFTMOE_FORWARD_BENCH_ROUNDS");
    const auto rounds = rounds_text
                            ? static_cast<std::uint32_t>(
                                  std::stoul(rounds_text))
                            : 110U;
    require(rounds > 10, "forward benchmark requires >10 rounds");
    const auto generation_base_text =
        std::getenv("WEFTMOE_FORWARD_BENCH_GENERATION_BASE");
    const auto generation_base =
        generation_base_text
            ? static_cast<std::uint64_t>(std::stoull(generation_base_text))
            : 1000ULL;
    require(generation_base <=
                std::numeric_limits<std::uint64_t>::max() - rounds,
            "forward benchmark generation range overflows");
    std::vector<double> wall_samples;
    std::vector<double> notify_samples;
    std::vector<double> forward_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> fc1_samples;
    std::vector<double> fc2_samples;
    std::vector<double> return_samples;
    std::vector<double> reduce_samples;
    std::vector<double> fc1_fc2_overlap_samples;
    std::vector<double> fc2_return_overlap_samples;
    std::vector<double> return_reduce_overlap_samples;
    std::vector<double> terminal_status_samples;
    weftmoe::cuda::ForwardMegaKernelAudit last_audit;
    weftmoe::cuda::DeviceForwardHandle last_output;
    const auto forward_api_before =
        weftmoe::cuda::testing::cuda_api_audit_snapshot();
    for (std::uint32_t round = 0; round < rounds; ++round) {
      if (std::getenv("WEFTMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_PROGRESS rank=" << rank << " round=" << round
                  << std::endl;
      auto generation = weftmoe::cuda::make_nccl_generation(
          base, "cuda-forward-bench", generation_base + round);
      auto context = weftmoe::make_rank_collective_context(
          generation.session(), rank, execution.token());
      auto routes =
          weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
              route_storage, context, execution);
      auto gates =
          weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
              gate_storage, routes, context, execution);
      auto activations =
          weftmoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
              activation_storage, context, execution);
      const auto wall_start = Clock::now();
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          generation, execution, pool);
      auto notify = coordinator.run_device(
          {common_config, routes, gates, max_tokens});
      const auto notify_end = Clock::now();
      if (std::getenv("WEFTMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_HOST_PROGRESS rank=" << rank
                  << " round=" << round << " phase=notify_done"
                  << std::endl;
      weftmoe::cuda::ActivationTransportRunOptions run_options;
      auto progress_deadline_seconds = 5U;
      if (const auto *deadline =
              std::getenv("WEFTMOE_FORWARD_PROGRESS_DEADLINE_SECONDS"))
        progress_deadline_seconds =
            static_cast<std::uint32_t>(std::stoul(deadline));
      require(progress_deadline_seconds != 0U,
              "Forward progress deadline must be positive");
      run_options.deadline_after =
          std::getenv("WEFTMOE_FORWARD_PROGRESS")
              ? std::chrono::seconds(progress_deadline_seconds)
              : std::chrono::seconds(300);
      if (std::getenv("WEFTMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_HOST_PROGRESS rank=" << rank
                  << " round=" << round << " phase=forward_enter"
                  << std::endl;
      auto output = run_forward(
          notify.dispatch(), activations, weights,
          {hidden, intermediate, retain_backward_state}, run_options);
      if (std::getenv("WEFTMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_HOST_PROGRESS rank=" << rank
                  << " round=" << round << " phase=forward_done"
                  << std::endl;
      const auto wall_end = Clock::now();
      last_audit = output.audit();
      last_output = output;
      if (round >= 10U) {
        wall_samples.push_back(
            std::chrono::duration<double, std::micro>(wall_end - wall_start)
                .count());
        notify_samples.push_back(
            std::chrono::duration<double, std::micro>(notify_end - wall_start)
                .count());
        forward_samples.push_back(output.timings().total_us);
        dispatch_samples.push_back(output.timings().dispatch_us);
        fc1_samples.push_back(output.timings().fc1_swiglu_us);
        fc2_samples.push_back(output.timings().fc2_us);
        return_samples.push_back(output.timings().combine_return_us);
        reduce_samples.push_back(output.timings().reduce_us);
        fc1_fc2_overlap_samples.push_back(
            output.timings().fc1_fc2_overlap_us);
        fc2_return_overlap_samples.push_back(
            output.timings().fc2_return_overlap_us);
        return_reduce_overlap_samples.push_back(
            output.timings().return_reduce_overlap_us);
        terminal_status_samples.push_back(
            output.timings().terminal_status_agreement_us);
      }
    }
    const auto forward_api_after =
        weftmoe::cuda::testing::cuda_api_audit_snapshot();
#if WEFTMOE_DEVICE_TIMELINE
    if (std::getenv("WEFTMOE_DEVICE_TIMELINE_PREFIX") != nullptr) {
      const std::string prefix =
          std::getenv("WEFTMOE_DEVICE_TIMELINE_PREFIX");
      write_device_timeline(
          weftmoe::cuda::testing::download_forward_device_timeline(
              last_output),
          prefix);
    }
#endif
    require(forward_api_after.device_allocations ==
                    forward_api_before.device_allocations &&
                forward_api_after.device_frees ==
                    forward_api_before.device_frees &&
                forward_api_after.pinned_allocations ==
                    forward_api_before.pinned_allocations &&
                forward_api_after.pinned_frees ==
                    forward_api_before.pinned_frees &&
                forward_api_after.event_creates ==
                    forward_api_before.event_creates &&
                forward_api_after.event_destroys ==
                    forward_api_before.event_destroys &&
                forward_api_after.device_synchronizes ==
                    forward_api_before.device_synchronizes,
            "Forward steady path allocated resources or device-synchronized");
    const auto emit_values = [](const std::vector<double> &values) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << values[index];
      }
    };
    const auto downloaded_output =
        weftmoe::cuda::download_forward_output_for_validation(last_output,
                                                               execution);
    std::uint64_t output_hash = 1469598103934665603ULL;
    std::vector<std::uint16_t> first_token_unique;
    for (const auto value : downloaded_output) {
      output_hash = (output_hash ^ static_cast<std::uint8_t>(value.bits)) *
                    1099511628211ULL;
      output_hash =
          (output_hash ^ static_cast<std::uint8_t>(value.bits >> 8U)) *
          1099511628211ULL;
    }
    for (std::uint64_t column = 0;
         column < hidden && first_token_unique.size() < 32U; ++column) {
      const auto bits = downloaded_output[static_cast<std::size_t>(column)].bits;
      if (std::find(first_token_unique.begin(), first_token_unique.end(), bits) ==
          first_token_unique.end())
        first_token_unique.push_back(bits);
    }
    require(first_token_unique.size() > 1U,
            "Forward output fixture collapsed to one repeated value");
    if (const auto *output_dir = std::getenv("WEFTMOE_FORWARD_OUTPUT_DIR")) {
      const auto path = std::string(output_dir) + "/weftmoe-rank-" +
                        std::to_string(rank) + ".bf16";
      std::ofstream destination(path, std::ios::binary | std::ios::trunc);
      require(destination.good(), "failed to open forward output artifact");
      destination.write(
          reinterpret_cast<const char *>(downloaded_output.data()),
          static_cast<std::streamsize>(downloaded_output.size() *
                                       sizeof(downloaded_output.front())));
      require(destination.good(), "failed to write forward output artifact");
      if (std::getenv("WEFTMOE_FORWARD_DUMP_INPUTS")) {
        const auto prefix = std::string(output_dir) + "/weftmoe-rank-" +
                            std::to_string(rank);
        const auto dump = [&](const std::string &suffix, const auto &values) {
          std::ofstream file(prefix + suffix, std::ios::binary);
          file.write(reinterpret_cast<const char *>(values.data()),
                     static_cast<std::streamsize>(values.size() *
                                                 sizeof(values.front())));
          require(file.good(), "failed to write Forward input artifact");
        };
        dump(".input.bf16", activation_host);
        dump(".routes.i32", common_routes.selected_experts);
        dump(".gates.f32", route_gates);
      }
    }
    const auto usage = transport_pool.usage();
    const char *reduce_scheduler =
        std::getenv("WEFTMOE_FORWARD_DYNAMIC_REDUCE_SCAN") != nullptr
            ? "scan"
            : (std::getenv("WEFTMOE_FORWARD_LEGACY_REDUCE") != nullptr
                   ? "legacy"
                   : "queue");
    std::cout << "FORWARD_BENCH rank=" << rank
              << " world=" << world
              << " tokens=" << common_routes.num_tokens
              << " experts=" << experts << " top_k=" << top_k
              << " hidden=" << hidden << " intermediate=" << intermediate
              << " backend="
              << (gin_forward_benchmark
                      ? "gin-rdma"
                      : socket_forward_benchmark ? "nccl-socket-fallback"
                                                 : "local-cuda-p2p")
              << " specialization=sm90-wgmma-tma-" << hidden << "x" << intermediate
              << " retained_backward_state=" << retain_backward_state
              << " reduce_scheduler=" << reduce_scheduler
              << " setup_us=" << one_time_setup_us
              << " device_pool_bytes=" << usage.device_bytes
              << " compute_pool_bytes=" << usage.compute_device_bytes
              << " gin_registered_bytes=" << usage.gin_registered_bytes
              << " pinned_pool_bytes=" << usage.pinned_bytes
              << " pool_events=" << usage.event_count
              << " weight_bytes=" << weights.device_bytes()
              << " persistent_launches="
              << last_audit.persistent_kernel_launches
              << " wgmma=" << last_audit.wgmma_instructions
              << " tma=" << last_audit.tma_loads
              << " multicast=" << last_audit.tma_multicast_loads
              << " dispatch_peer_bytes="
              << last_audit.actual_dispatch_peer_bytes
              << " return_peer_bytes="
              << last_audit.actual_return_peer_bytes
              << " gin_dispatch_puts="
              << last_audit.gin_dispatch_put_count
              << " raw_remote_route_bytes="
              << last_audit.raw_cross_node_route_bytes
              << " gin_dispatch_payload_bytes="
              << last_audit.gin_dispatch_payload_bytes
              << " gin_return_payload_bytes="
              << last_audit.gin_return_payload_bytes
              << " gin_return_puts=" << last_audit.gin_return_put_count
              << " gin_signals=" << last_audit.gin_signal_count
              << " gin_metadata_bytes=" << last_audit.gin_metadata_bytes
              << " activation_hash=" << activation_hash
              << " route_hash=" << route_hash
              << " gate_hash=" << gate_hash
              << " weight_generator_version=3"
              << " weight_hash=" << weight_hash
              << " actual_w1_hash_low=" << actual_weight_hashes.w1.low
              << " actual_w1_hash_high=" << actual_weight_hashes.w1.high
              << " actual_w2_hash_low=" << actual_weight_hashes.w2.low
              << " actual_w2_hash_high=" << actual_weight_hashes.w2.high
              << " first_token_output_unique_at_least="
              << first_token_unique.size()
              << " cluster_supported="
              << static_cast<unsigned int>(
                     last_audit.runtime_cluster_supported)
              << " cluster_requested=" << last_audit.requested_cluster_size
              << " cluster_actual=" << last_audit.runtime_cluster_size
              << " cluster_rank_mask="
              << last_audit.observed_cluster_rank_mask
              << " max_active_clusters=" << last_audit.max_active_clusters
              << " steady_forbidden_cuda_api_calls=0"
              << " output_hash=" << output_hash << " wall=";
    emit_values(wall_samples);
    std::cout << " notify=";
    emit_values(notify_samples);
    std::cout << " forward=";
    emit_values(forward_samples);
    std::cout << " dispatch=";
    emit_values(dispatch_samples);
    std::cout << " fc1=";
    emit_values(fc1_samples);
    std::cout << " fc2=";
    emit_values(fc2_samples);
    std::cout << " return=";
    emit_values(return_samples);
    std::cout << " reduce=";
    emit_values(reduce_samples);
    std::cout << " fc1_fc2_overlap=";
    emit_values(fc1_fc2_overlap_samples);
    std::cout << " fc2_return_overlap=";
    emit_values(fc2_return_overlap_samples);
    std::cout << " return_reduce_overlap=";
    emit_values(return_reduce_overlap_samples);
    std::cout << " terminal_status_agreement=";
    emit_values(terminal_status_samples);
    std::cout << std::endl;
    return;
  }
  if (forward_smoke) {
    constexpr std::uint64_t hidden = 7168;
    constexpr std::uint64_t intermediate = 3072;
    auto generation = weftmoe::cuda::make_nccl_generation(
        base, "cuda-forward-smoke", 701);
    auto context = weftmoe::make_rank_collective_context(
        generation.session(), rank, execution.token());
    auto routes = weftmoe::cuda::upload_cuda_routes(
        common_routes, top_k, context, execution);
    const auto gate_mode = std::getenv("WEFTMOE_GATE_MODE")
                               ? std::getenv("WEFTMOE_GATE_MODE")
                               : "fixed-random";
    const auto gates = std::string(gate_mode) == "unit"
                           ? std::vector<float>(max_tokens * top_k, 1.0F)
                           : fixed_random_selected_gates(
                                 max_tokens, top_k, rank,
                                 0x4d4547414d4f4547ULL);
    auto device_gates =
        weftmoe::cuda::testing::upload_fp32_gates_for_testing(
            gates, routes, context, execution);
    weftmoe::cuda::ActivationTransportPoolConfig transport_config;
    transport_config.max_tokens_per_rank = max_tokens;
    transport_config.max_hidden_size = hidden;
    transport_config.max_top_k = top_k;
    transport_config.max_world_size = world;
    transport_config.max_nodes = nodes;
    transport_config.max_destination_rows_per_rank =
        world * max_tokens * top_k;
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel = 1;
    transport_config.max_inflight_generations = world == 1U ? 2U : 1U;
    transport_config.max_intermediate_size = intermediate;
    transport_config.max_owned_experts = experts / world;
    transport_config.enable_backward_state = true;
    transport_config.enable_gin_rdma = gin_forward_smoke;
    auto transport_pool = weftmoe::cuda::make_activation_transport_pool(
        generation.session(), execution, transport_config);
    std::vector<weftmoe::transport::BFloat16> input(max_tokens * hidden);
    for (std::uint64_t index = 0; index < max_tokens * hidden; ++index)
      input[static_cast<std::size_t>(index)] =
          weftmoe::transport::float_to_bfloat16(
              static_cast<float>(static_cast<int>((index + rank * 13U) %
                                                   127U) -
                                 63) /
              64.0F);
    auto activations =
        weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
            input, max_tokens, hidden, context, execution, transport_pool);
    auto weights =
        weftmoe::cuda::testing::make_deterministic_forward_weights_for_testing(
            experts / world, rank * (experts / world), hidden, intermediate,
            generation.session(), execution);
    std::unique_ptr<weftmoe::cuda::LocalCudaP2pForwardMegaKernel>
        local_forward;
    std::unique_ptr<weftmoe::cuda::GinRdmaForwardMegaKernel> gin_forward;
    if (gin_forward_smoke)
      gin_forward =
          std::make_unique<weftmoe::cuda::GinRdmaForwardMegaKernel>(
              generation, execution, transport_pool);
    else
      local_forward =
          std::make_unique<weftmoe::cuda::LocalCudaP2pForwardMegaKernel>(
              generation, execution, transport_pool);
    const auto run_forward = [&](const auto &dispatch_handle,
                                 const auto &activation_handle,
                                 const auto &weight_handle,
                                 const weftmoe::cuda::ForwardMegaKernelConfig
                                     &forward_config,
                                 const auto &run_options_value) {
      return gin_forward_smoke
                 ? gin_forward->run(dispatch_handle, activation_handle,
                                    weight_handle, forward_config,
                                    run_options_value)
                 : local_forward->run(dispatch_handle, activation_handle,
                                      weight_handle, forward_config,
                                      run_options_value);
    };
    weftmoe::cuda::ActivationTransportRunOptions run_options;
    run_options.deadline_after = std::chrono::seconds(120);
    // Two physical transport slots and two live Forward handles must remain
    // isolated. Drop the producer leases after submission so the handles are
    // the final owners, then prove pool exhaustion and exact release.
    if (world == 1U) {
      auto second_input = input;
      for (auto &value : second_input)
        value = weftmoe::transport::float_to_bfloat16(
            -weftmoe::transport::bfloat16_to_float(value));
      auto second_generation = weftmoe::cuda::make_nccl_generation(
          base, "cuda-forward-smoke", 702U);
      auto second_context = weftmoe::make_rank_collective_context(
          second_generation.session(), rank, execution.token());
      auto second_activations =
          weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
              second_input, max_tokens, hidden, second_context, execution,
              transport_pool);
      const auto run_exact = [&](std::uint64_t exact_generation,
                                 const auto &source) {
        auto current_generation = weftmoe::cuda::make_nccl_generation(
            base, "cuda-forward-smoke", exact_generation);
        auto current_context = weftmoe::make_rank_collective_context(
            current_generation.session(), rank, execution.token());
        auto current_routes =
            weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
                routes, current_context, execution);
        auto current_gates =
            weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
                device_gates, current_routes, current_context, execution);
        auto current_activation =
            weftmoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
                source, current_context, execution);
        weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            current_generation, execution, pool);
        auto notify = coordinator.run_device(
            {common_config, current_routes, current_gates, max_tokens});
        return run_forward(
            notify.dispatch(), current_activation, weights,
            {hidden, intermediate}, run_options);
      };
      auto live_first = run_exact(780U, activations);
      auto first_before =
          weftmoe::cuda::download_forward_output_for_validation(live_first,
                                                                 execution);
      activations = {};
      auto live_second = run_exact(781U, second_activations);
      second_activations = {};
      const auto second_output =
          weftmoe::cuda::download_forward_output_for_validation(live_second,
                                                                 execution);
      const auto first_after =
          weftmoe::cuda::download_forward_output_for_validation(live_first,
                                                                 execution);
      require(first_before == first_after && first_before != second_output,
              "two live generations aliased output storage");
      auto live_usage = transport_pool.usage();
      require(live_usage.slots_in_use == 2 && live_usage.slots_available == 0,
              "two live Forward handles did not retain both slots");
      bool exhausted = false;
      try {
        (void)weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
            input, max_tokens, hidden, context, execution, transport_pool);
      } catch (const weftmoe::NotifyDispatchError &error) {
        exhausted =
            error.code() == weftmoe::NotifyErrorCode::ResourceExhausted;
      }
      require(exhausted, "pool=2 accepted a third live generation");
      live_first = {};
      live_second = {};
      live_usage = transport_pool.usage();
      require(live_usage.slots_available == 2 && live_usage.slots_in_use == 0,
              "last Forward handle did not release its slot");
      activations =
          weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
              input, max_tokens, hidden, context, execution, transport_pool);
    }
    std::vector<weftmoe::transport::BFloat16> reference_output;
    weftmoe::cuda::ForwardBackwardStateValidationDownload
        reference_backward_state;
    bool saw_backward_state = false;
    weftmoe::cuda::ForwardMegaKernelAudit reference_audit;
    weftmoe::cuda::ForwardMegaKernelTimings reference_timings;
    const auto forward_smoke_repeats =
        std::getenv("WEFTMOE_FORWARD_SMOKE_REPEATS")
            ? static_cast<std::uint32_t>(std::stoul(
                  std::getenv("WEFTMOE_FORWARD_SMOKE_REPEATS")))
            : 20U;
    require(forward_smoke_repeats >= 1U,
            "Forward smoke repeat count must be nonzero");
    for (std::uint32_t repeat = 0; repeat < forward_smoke_repeats; ++repeat) {
      if (std::getenv("WEFTMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_SMOKE_PROGRESS rank=" << rank
                  << " repeat=" << repeat << std::endl;
      auto repeat_generation = weftmoe::cuda::make_nccl_generation(
          base, "cuda-forward-smoke", 800U + repeat);
      auto repeat_context = weftmoe::make_rank_collective_context(
          repeat_generation.session(), rank, execution.token());
      auto repeat_routes =
          weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
              routes, repeat_context, execution);
      auto repeat_gates =
          weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
              device_gates, repeat_routes, repeat_context, execution);
      auto repeat_activations =
          weftmoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
              activations, repeat_context, execution);
      weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          repeat_generation, execution, pool);
      auto notify = coordinator.run_device(
          {common_config, repeat_routes, repeat_gates, max_tokens});
      const bool retain_backward_state = (repeat & 1U) != 0U;
      auto retained_metadata =
          retain_backward_state && !saw_backward_state
              ? weftmoe::cuda::download_device_metadata_for_validation(
                    notify.dispatch(), execution)
              : weftmoe::DispatchHandle{};
      auto current = run_forward(notify.dispatch(), repeat_activations, weights,
                                 {hidden, intermediate,
                                  retain_backward_state},
                                 run_options);
      const auto current_output =
          weftmoe::cuda::download_forward_output_for_validation(current,
                                                                 execution);
      if (retain_backward_state) {
        const auto state =
            weftmoe::cuda::download_forward_backward_state_for_validation(
                current, execution);
        require(state.fc1_ab.size() % (2ULL * intermediate) == 0ULL &&
                    state.canonical_x.size() % hidden == 0ULL &&
                    state.fc1_ab.size() / (2ULL * intermediate) ==
                        state.canonical_x.size() / hidden,
                "retained Forward state has inconsistent compact row shapes");
        if (!state.fc1_ab.empty())
          require(std::any_of(state.fc1_ab.begin(), state.fc1_ab.end(),
                              [](const auto value) {
                                return value.bits != 0U;
                              }),
                  "retained BF16 FC1 A/B checkpoint is all zero");
        if (!saw_backward_state) {
          require(retained_metadata &&
                      retained_metadata->incoming.incoming_rows.size() ==
                          state.canonical_x.size() / hidden,
                  "retained canonical X metadata shape mismatch");
          for (const auto &row :
               retained_metadata->incoming.incoming_rows) {
            const auto route = std::find(kCrossEpExperts.begin(),
                                         kCrossEpExperts.end(),
                                         row.expert_id);
            require(route != kCrossEpExperts.end(),
                    "retained canonical X referenced an unknown smoke expert");
            const auto token =
                static_cast<std::uint64_t>(route - kCrossEpExperts.begin()) /
                6ULL;
            for (std::uint64_t column = 0; column < hidden; ++column) {
              const auto linear = token * hidden + column;
              const auto expected =
                  weftmoe::transport::float_to_bfloat16(
                      static_cast<float>(static_cast<int>(
                          (linear + row.source_rank * 13U) % 127U) - 63) /
                      64.0F);
              require(state.canonical_x[static_cast<std::size_t>(
                          row.dst_row * hidden + column)] == expected,
                      "retained canonical X was overwritten by FC2 output");
            }
          }
          reference_backward_state = state;
          saw_backward_state = true;
        } else {
          require(state.canonical_x == reference_backward_state.canonical_x &&
                      state.fc1_ab == reference_backward_state.fc1_ab,
                  "retained Forward state changed across identical runs");
        }
      }
      if (std::getenv("WEFTMOE_FORWARD_PROGRESS")) {
        std::uint64_t repeat_hash = 1469598103934665603ULL;
        for (const auto value : current_output)
          repeat_hash = (repeat_hash ^ value.bits) * 1099511628211ULL;
        std::cerr << "FORWARD_SMOKE_HASH rank=" << rank
                  << " repeat=" << repeat << " hash=" << repeat_hash
                  << std::endl;
      }
      if (repeat == 0U) {
        reference_output = current_output;
        reference_audit = current.audit();
        reference_timings = current.timings();
      } else {
        if (current_output != reference_output &&
            std::getenv("WEFTMOE_FORWARD_PROGRESS")) {
          const auto mismatch = std::mismatch(reference_output.begin(),
                                              reference_output.end(),
                                              current_output.begin());
          const auto index = static_cast<std::size_t>(
              mismatch.first - reference_output.begin());
          std::cerr << "FORWARD_SMOKE_MISMATCH rank=" << rank
                    << " repeat=" << repeat << " index=" << index
                    << " expected_bits=" << mismatch.first->bits
                    << " actual_bits=" << mismatch.second->bits << std::endl;
        }
        require(current_output == reference_output,
                "Forward output changed across identical repetitions");
      }
    }
    require(reference_audit.persistent_kernel_launches == 1 &&
                ((reference_audit.wgmma_instructions != 0 &&
                  reference_audit.tma_loads != 0) ||
                 reference_audit.fc1_tasks == 0),
            "forward smoke did not execute one WGMMA/TMA persistent kernel");
    require(saw_backward_state,
            "forward smoke did not exercise retained Backward state");
    std::uint64_t output_hash = 1469598103934665603ULL;
    for (const auto value : reference_output)
      output_hash = (output_hash ^ value.bits) * 1099511628211ULL;
    if (const auto *output_dir = std::getenv("WEFTMOE_FORWARD_OUTPUT_DIR")) {
      const auto prefix = std::string(output_dir) + "/weftmoe-rank-" +
                          std::to_string(rank);
      std::ofstream output_file(prefix + ".bf16",
                                std::ios::binary | std::ios::trunc);
      require(output_file.good(), "failed to open Forward smoke output");
      output_file.write(reinterpret_cast<const char *>(reference_output.data()),
                        static_cast<std::streamsize>(
                            reference_output.size() * sizeof(reference_output[0])));
      std::ofstream cache_file(prefix + ".ab.bf16",
                               std::ios::binary | std::ios::trunc);
      require(cache_file.good(),
              "failed to open Forward A/B checkpoint artifact");
      cache_file.write(
          reinterpret_cast<const char *>(
              reference_backward_state.fc1_ab.data()),
          static_cast<std::streamsize>(
              reference_backward_state.fc1_ab.size() *
              sizeof(reference_backward_state.fc1_ab[0])));
    }
    std::cout << "FORWARD_SMOKE rank=" << rank
              << " specialization=sm90-wgmma-tma-7168x3072"
              << " persistent_launches="
              << reference_audit.persistent_kernel_launches
              << " wgmma=" << reference_audit.wgmma_instructions
              << " tma=" << reference_audit.tma_loads
              << " fc1_tasks=" << reference_audit.fc1_tasks
              << " fc2_tasks=" << reference_audit.fc2_tasks
              << " repeat_equal=1 repeats="
              << forward_smoke_repeats << " gate_mode="
              << gate_mode << " top_k=" << top_k
              << " output_hash=" << output_hash
              << " pool2_isolated=1"
              << " backward_state_equal=1"
              << " total_us=" << reference_timings.total_us
              << " dispatch_us=" << reference_timings.dispatch_us
              << " fc1_us=" << reference_timings.fc1_swiglu_us
              << " fc2_us=" << reference_timings.fc2_us
              << " return_us=" << reference_timings.combine_return_us
              << " reduce_us=" << reference_timings.reduce_us << std::endl;
    return;
  }
  if (benchmark_integrated) {
    require(nodes == 1 && (world == 2 || world == 8),
            "integrated benchmark requires one 2/8-GPU node");
    constexpr std::uint64_t hidden = 7168;
    const auto setup_start = Clock::now();
    auto setup_generation = weftmoe::cuda::make_nccl_generation(
        base, "cuda-multinode-base", 701);
    auto setup_context = weftmoe::make_rank_collective_context(
        setup_generation.session(), rank, execution.token());
    auto route_storage = weftmoe::cuda::upload_cuda_routes(
        common_routes, top_k, setup_context, execution);
    const auto gate_mode_text = std::getenv("WEFTMOE_GATE_MODE");
    const std::string gate_mode = gate_mode_text ? gate_mode_text
                                                  : "fixed-random";
    require(gate_mode == "fixed-random" || gate_mode == "unit",
            "gate mode must be fixed-random or unit");
    auto route_gates = gate_mode == "unit"
        ? std::vector<float>(common_routes.selected_experts.size(), 1.0F)
        : fixed_random_selected_gates(common_routes.num_tokens, top_k, rank,
                                      0x4d4547414d4f4547ULL);
    if (gate_mode == "fixed-random")
      for (std::uint64_t token = 0; token < common_routes.num_tokens; ++token) {
        float sum = 0.0F;
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
          const auto gate = route_gates[
              static_cast<std::size_t>(token * top_k + slot)];
          require(gate > 0.0F && gate <= 1.0F,
                  "fixed-random gate is outside (0,1]");
          sum += gate;
        }
        require(std::abs(sum - 1.0F) <= 2.0e-6F,
                "fixed-random selected gates do not sum to one");
      }
    auto gate_storage = weftmoe::cuda::testing::upload_fp32_gates_for_testing(
        route_gates, route_storage, setup_context, execution);
    const auto gate_hash = hash_fp32(route_gates);
    weftmoe::cuda::ActivationTransportPoolConfig transport_pool_config;
    transport_pool_config.max_tokens_per_rank = max_tokens;
    transport_pool_config.max_hidden_size = hidden;
    transport_pool_config.max_top_k = top_k;
    transport_pool_config.max_world_size = world;
    transport_pool_config.max_nodes = 1;
    std::vector<std::uint64_t> destination_rows(world, 0);
    for (std::uint32_t source = 0; source < world; ++source)
      for (std::uint64_t token = 0; token < max_tokens; ++token) {
        const auto base_expert = static_cast<std::uint32_t>(
            (token * 17U + source * 31U) % experts);
        for (std::uint32_t route_slot = 0; route_slot < top_k; ++route_slot) {
          const auto expert =
              (base_expert + route_slot * 53U) % experts;
          ++destination_rows[expert / (experts / world)];
        }
      }
    transport_pool_config.max_destination_rows_per_rank =
        *std::max_element(destination_rows.begin(), destination_rows.end());
    transport_pool_config.channels = 1;
    transport_pool_config.ring_slots_per_source_channel =
        static_cast<std::uint32_t>(max_tokens * top_k);
    transport_pool_config.max_inflight_generations = 1;
    auto transport_pool = weftmoe::cuda::make_activation_transport_pool(
        setup_generation.session(), execution, transport_pool_config);
    std::vector<weftmoe::transport::BFloat16> activations(
        static_cast<std::size_t>(common_routes.num_tokens * hidden));
    for (std::size_t index = 0; index < activations.size(); ++index) {
      const auto value = static_cast<float>(rank * 101U + index % 97U) / 32.0F;
      activations[index] = weftmoe::transport::float_to_bfloat16(value);
    }
    auto activation_storage =
        weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, common_routes.num_tokens, hidden, setup_context,
            execution, transport_pool);
    weftmoe::cuda::LocalCudaP2pActivationTransport transport(
        setup_generation, execution, transport_pool);
    const auto one_time_setup_us =
        std::chrono::duration<double, std::micro>(Clock::now() - setup_start)
            .count();

    weftmoe::transport::ActivationTransportConfig transport_config;
    transport_config.hidden_size = hidden;
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel =
        transport_pool_config.ring_slots_per_source_channel;
    transport_config.expert_transform =
        weftmoe::transport::ActivationExpertTransform::Identity;
    transport_config.gate_placement =
        weftmoe::transport::ActivationGatePlacement::SourceAfter;
    const auto rounds_text =
        std::getenv("WEFTMOE_INTEGRATED_BENCH_ROUNDS");
    const auto rounds = rounds_text
                            ? static_cast<std::uint32_t>(
                                  std::stoul(rounds_text))
                            : 110U;
    require(rounds > 10,
            "integrated benchmark requires 10 warmups plus measurements");
    std::vector<double> wall_samples;
    std::vector<double> notify_samples;
    std::vector<double> mapping_samples;
    std::vector<double> control_samples;
    std::vector<double> count_summary_samples;
    std::vector<double> materialize_samples;
    std::vector<double> transport_samples;
    std::vector<double> transport_setup_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> relay_return_samples;
    std::vector<double> reduce_samples;
    std::vector<double> terminal_samples;
    weftmoe::cuda::ActivationTransportDeviceAudit last_audit;
    weftmoe::cuda::CollectiveProtocolAudit last_notify_audit;
    const auto api_before =
        weftmoe::cuda::testing::cuda_api_audit_snapshot();
    for (std::uint32_t round = 0; round < rounds; ++round) {
      const auto generation = 10000ULL + round;
      auto generation_session = weftmoe::cuda::make_nccl_generation(
          base, "cuda-multinode-base", generation);
      auto context = weftmoe::make_rank_collective_context(
          generation_session.session(), rank, execution.token());
      auto exact_routes =
          weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
              route_storage, context, execution);
      auto exact_gates =
          weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
              gate_storage, exact_routes, context, execution);
      auto exact_activation =
          weftmoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
              activation_storage, context, execution);
      weftmoe::cuda::GenerationExecutionTimings notify_timing;
      weftmoe::cuda::ActivationTransportTimings transport_timing;
      const auto wall_start = Clock::now();
      {
        weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            generation_session, execution, pool);
        auto notify = coordinator.run_device(
            {common_config, exact_routes, exact_gates, max_tokens});
        notify_timing = notify.execution_timings();
        last_notify_audit = notify.audit();
        weftmoe::cuda::ActivationTransportRunOptions options;
        options.deadline_after = std::chrono::seconds(120);
        auto transported = transport.run_transport_slice(
            notify.dispatch(), exact_activation, transport_config, options);
        transport_timing = transported.timings();
        last_audit = transported.audit();
        require(last_audit.backend ==
                    weftmoe::transport::ActivationTransportBackend::LocalCudaP2p,
                "integrated benchmark did not run LocalCudaP2p");
        require(last_audit.activation_d2h_bytes == 0 &&
                    last_audit.activation_h2d_bytes == 0 &&
                    last_audit.hot_path_allocations == 0,
                "integrated benchmark staged or allocated activation data");
        transported = {};
        notify = {};
      }
      const auto wall_us =
          std::chrono::duration<double, std::micro>(Clock::now() - wall_start)
              .count();
      if (round >= 10U) {
        wall_samples.push_back(wall_us);
        notify_samples.push_back(notify_timing.coordinator_run_wall_us);
        mapping_samples.push_back(notify_timing.local_mapping_us);
        control_samples.push_back(notify_timing.control_config_us);
        count_summary_samples.push_back(notify_timing.expert_count_us +
                                        notify_timing.packet_summary_us);
        materialize_samples.push_back(
            notify_timing.device_materialization_us);
        transport_samples.push_back(transport_timing.total_us);
        transport_setup_samples.push_back(
            transport_timing.generation_setup_us);
        dispatch_samples.push_back(transport_timing.dispatch_put_us);
        relay_return_samples.push_back(
            transport_timing.synthetic_return_us);
        reduce_samples.push_back(transport_timing.combine_reduce_us);
        terminal_samples.push_back(transport_timing.terminal_completion_us);
      }
    }
    const auto api_after =
        weftmoe::cuda::testing::cuda_api_audit_snapshot();
    require(api_after.device_allocations == api_before.device_allocations &&
                api_after.device_frees == api_before.device_frees &&
                api_after.pinned_allocations == api_before.pinned_allocations &&
                api_after.pinned_frees == api_before.pinned_frees &&
                api_after.event_creates == api_before.event_creates &&
                api_after.event_destroys == api_before.event_destroys &&
                api_after.device_synchronizes == api_before.device_synchronizes,
            "integrated steady generations performed dynamic CUDA setup");

    // One explicit validation generation is outside every timed sample.
    auto validation_session = weftmoe::cuda::make_nccl_generation(
        base, "cuda-multinode-base", 50000);
    auto validation_context = weftmoe::make_rank_collective_context(
        validation_session.session(), rank, execution.token());
    auto validation_routes =
        weftmoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
            route_storage, validation_context, execution);
    auto validation_gates =
        weftmoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
            gate_storage, validation_routes, validation_context, execution);
    auto validation_activation =
        weftmoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
            activation_storage, validation_context, execution);
    weftmoe::cuda::CudaNotifyGenerationCoordinator validation_coordinator(
        validation_session, execution, pool);
    auto validation_notify = validation_coordinator.run_device(
        {common_config, validation_routes, validation_gates, max_tokens});
    auto validation_transport = transport.run_transport_slice(
        validation_notify.dispatch(), validation_activation, transport_config,
        {});
    const auto output =
        weftmoe::cuda::download_activation_output_for_validation(
            validation_transport, execution);
    require(output.size() == activations.size(),
            "integrated output shape mismatch");
    std::vector<weftmoe::transport::BFloat16> expected_by_token_value(
        static_cast<std::size_t>(common_routes.num_tokens * 97U));
    for (std::uint64_t token = 0; token < common_routes.num_tokens; ++token)
      for (std::uint32_t value_index = 0; value_index < 97U; ++value_index) {
        const auto input = weftmoe::transport::float_to_bfloat16(
            static_cast<float>(rank * 101U + value_index) / 32.0F);
        float expected = 0.0F;
        std::vector<std::uint32_t> order(top_k);
        for (std::uint32_t slot = 0; slot < top_k; ++slot)
          order[slot] = slot;
        std::sort(order.begin(), order.end(), [&](std::uint32_t lhs,
                                                  std::uint32_t rhs) {
          return common_routes.selected_experts[
                     static_cast<std::size_t>(token * top_k + lhs)] <
                 common_routes.selected_experts[
                     static_cast<std::size_t>(token * top_k + rhs)];
        });
        for (const auto slot : order) {
          const auto route = static_cast<std::size_t>(token * top_k + slot);
          volatile float product =
              route_gates[route] *
              weftmoe::transport::bfloat16_to_float(input);
          expected += product;
        }
        expected_by_token_value[
            static_cast<std::size_t>(token * 97U + value_index)] =
            weftmoe::transport::float_to_bfloat16(expected);
      }
    std::uint64_t output_hash = 1469598103934665603ULL;
    bool output_equal = true;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const auto token = index / static_cast<std::size_t>(hidden);
      output_equal = output_equal &&
                     output[index] == expected_by_token_value[
                         token * 97U + index % 97U];
      output_hash = (output_hash ^ output[index].bits) * 1099511628211ULL;
    }
    require(output_equal,
            "integrated identity/gated output differs from common reference");

    const auto emit_values = [](const std::vector<double> &values) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << values[index];
      }
    };
    const auto usage = transport_pool.usage();
    std::cout << "INTEGRATED_BENCH rank=" << rank
              << " backend=local-cuda-p2p"
              << " setup_us=" << one_time_setup_us
              << " device_bytes=" << usage.device_bytes
              << " pinned_bytes=" << usage.pinned_bytes
              << " route_input_bytes=" << route_storage.device_bytes()
              << " local_gate_bytes=" << gate_storage.device_bytes()
              << " peer_gate_bytes=0"
              << " gate_mode=" << gate_mode
              << " gate_hash=" << gate_hash
              << " output_equal=1 output_hash=" << output_hash
              << " raw_bytes=" << last_audit.raw_cross_node_payload_bytes
              << " dedup_bytes="
              << last_audit.deduplicated_cross_node_payload_bytes
              << " actual_dispatch_peer_bytes="
              << last_audit.actual_dispatch_peer_bytes
              << " actual_return_peer_bytes="
              << last_audit.actual_return_peer_bytes
              << " derived_route_bytes="
              << last_audit.derived_route_input_bytes
              << " derived_local_gate_bytes="
              << last_audit.derived_local_gate_bytes
              << " derived_peer_gate_bytes="
              << last_audit.derived_peer_gate_bytes
              << " generation_memset_calls="
              << last_audit.generation_memset_calls
              << " generation_memset_bytes="
              << last_audit.generation_memset_bytes
              << " kernel_launches=" << last_audit.kernel_launches
              << " control_enqueues=" << last_notify_audit.control_starts
              << " config_enqueues=" << last_notify_audit.config_starts
              << " count_enqueues="
              << last_notify_audit.expert_count_starts
              << " summary_enqueues="
              << last_notify_audit.packet_summary_starts
              << " dispatch_kernel_blocks="
              << last_audit.dispatch_kernel_blocks
              << " return_kernel_blocks="
              << last_audit.return_kernel_blocks
              << " reduce_kernel_blocks="
              << last_audit.reduce_kernel_blocks
              << " steady_allocs="
              << api_after.device_allocations - api_before.device_allocations
              << " steady_h2d_bytes="
              << api_after.h2d_bytes - api_before.h2d_bytes
              << " steady_d2h_bytes="
              << api_after.d2h_bytes - api_before.d2h_bytes
              << " steady_d2d_bytes="
              << api_after.d2d_bytes - api_before.d2d_bytes
              << " wall=";
    emit_values(wall_samples);
    std::cout << " notify=";
    emit_values(notify_samples);
    std::cout << " mapping=";
    emit_values(mapping_samples);
    std::cout << " control=";
    emit_values(control_samples);
    std::cout << " count_summary=";
    emit_values(count_summary_samples);
    std::cout << " materialize=";
    emit_values(materialize_samples);
    std::cout << " transport=";
    emit_values(transport_samples);
    std::cout << " transport_setup=";
    emit_values(transport_setup_samples);
    std::cout << " dispatch=";
    emit_values(dispatch_samples);
    std::cout << " relay_return=";
    emit_values(relay_return_samples);
    std::cout << " reduce=";
    emit_values(reduce_samples);
    std::cout << " terminal=";
    emit_values(terminal_samples);
    std::cout << std::endl;
    return;
  }
  if (scenario == "activation_p2p" || scenario == "activation_p2p_cancel" ||
      scenario == "activation_socket" || scenario == "activation_gin" ||
      benchmark_p2p) {
    const bool socket_fallback = scenario == "activation_socket";
    const bool gin_rdma = scenario == "activation_gin";
    require(socket_fallback || gin_rdma ? nodes == 2 : nodes == 1,
            socket_fallback || gin_rdma
                ? "cross-host activation scenario requires exactly two nodes"
                : "local CUDA P2P scenario requires exactly one node");
    auto session = weftmoe::cuda::make_nccl_generation(
        base, "cuda-local-p2p", 701);
    auto context = weftmoe::make_rank_collective_context(
        session.session(), rank, execution.token());
    auto device_routes = weftmoe::cuda::upload_cuda_routes(
        common_routes, top_k, context, execution);
    const auto gate_mode_text = std::getenv("WEFTMOE_GATE_MODE");
    const std::string gate_mode = gate_mode_text ? gate_mode_text
                                                  : "fixed-random";
    require(gate_mode == "fixed-random" || gate_mode == "unit",
            "gate mode must be fixed-random or unit");
    auto route_gates = gate_mode == "unit"
        ? std::vector<float>(common_routes.selected_experts.size(), 1.0F)
        : fixed_random_selected_gates(common_routes.num_tokens, top_k, rank,
                                      0x4d4547414d4f4547ULL);
    auto device_gates =
        weftmoe::cuda::testing::upload_fp32_gates_for_testing(
            route_gates, device_routes, context, execution);
    const auto gate_hash = hash_fp32(route_gates);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    auto notify = coordinator.run_device(
        {common_config, device_routes, device_gates, max_tokens});
    weftmoe::DispatchHandle cross_host_oracle_dispatch;
    if (socket_fallback || gin_rdma)
      cross_host_oracle_dispatch =
          weftmoe::cuda::download_device_metadata_for_validation(
              notify.dispatch(), execution);
    const std::uint64_t hidden = benchmark_p2p ? 7168U : 17U;
    weftmoe::cuda::ActivationTransportPoolConfig transport_pool_config;
    transport_pool_config.max_tokens_per_rank = max_tokens;
    transport_pool_config.max_hidden_size = hidden;
    transport_pool_config.max_top_k = top_k;
    transport_pool_config.max_world_size = world;
    transport_pool_config.max_nodes = nodes;
    transport_pool_config.max_destination_rows_per_rank = benchmark_p2p
        ? max_tokens * top_k
        : static_cast<std::uint64_t>(world) * max_tokens * top_k;
    transport_pool_config.channels = 1;
    const auto no_wrap_test = std::getenv("WEFTMOE_P2P_NO_WRAP") != nullptr;
    const auto ring_slots_text = std::getenv("WEFTMOE_P2P_RING_SLOTS");
    transport_pool_config.ring_slots_per_source_channel =
        benchmark_p2p
            ? static_cast<std::uint32_t>(max_tokens * top_k)
            : (no_wrap_test
                   ? static_cast<std::uint32_t>(
                         transport_pool_config.max_destination_rows_per_rank)
                   : (ring_slots_text
                          ? static_cast<std::uint32_t>(
                                std::stoul(ring_slots_text))
                          : 2U));
    transport_pool_config.max_inflight_generations = 1;
    transport_pool_config.enable_socket_fallback = socket_fallback;
    transport_pool_config.enable_gin_rdma = gin_rdma;
    const auto setup_start = Clock::now();
    auto transport_pool = weftmoe::cuda::make_activation_transport_pool(
        session.session(), execution, transport_pool_config);
    std::vector<weftmoe::transport::BFloat16> activations(
        static_cast<std::size_t>(common_routes.num_tokens * hidden));
    for (std::size_t index = 0; index < activations.size(); ++index) {
      const auto value = static_cast<float>(rank * 101U + index % 97U) / 32.0F;
      activations[index] = weftmoe::transport::float_to_bfloat16(value);
    }
    auto activation =
        weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, common_routes.num_tokens, hidden, context, execution,
            transport_pool);
    std::unique_ptr<weftmoe::cuda::LocalCudaP2pActivationTransport>
        local_transport;
    std::unique_ptr<weftmoe::cuda::NcclSocketFallbackActivationTransport>
        socket_transport;
    std::unique_ptr<weftmoe::cuda::GinRdmaActivationTransport> gin_transport;
    if (socket_fallback)
      socket_transport = std::make_unique<
          weftmoe::cuda::NcclSocketFallbackActivationTransport>(
          session, execution, transport_pool);
    else if (gin_rdma)
      gin_transport =
          std::make_unique<weftmoe::cuda::GinRdmaActivationTransport>(
              session, execution, transport_pool);
    else
      local_transport = std::make_unique<
          weftmoe::cuda::LocalCudaP2pActivationTransport>(
          session, execution, transport_pool);
    const auto setup_us = std::chrono::duration<double, std::micro>(
                              Clock::now() - setup_start)
                              .count();
    weftmoe::transport::ActivationTransportConfig transport_config;
    transport_config.hidden_size = hidden;
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel =
        transport_pool_config.ring_slots_per_source_channel;
    transport_config.expert_transform =
        weftmoe::transport::ActivationExpertTransform::SyntheticValidation;
    transport_config.gate_placement =
        weftmoe::transport::ActivationGatePlacement::SourceAfter;
    if (scenario == "activation_p2p_cancel") {
      weftmoe::cuda::ActivationTransportRunOptions cancel_options;
      cancel_options.cancellation.request_cancel();
      bool cancelled = false;
      try {
        (void)local_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, cancel_options);
      } catch (const weftmoe::NotifyDispatchError &error) {
        cancelled = error.code() == weftmoe::NotifyErrorCode::TransportTimeout;
      }
      require(cancelled, "all-rank local P2P cancellation was not observed");
      activation = {};
      const auto usage = transport_pool.usage();
      require(usage.slots_quarantined == 1 && usage.slots_available == 0,
              "cancelled local P2P slot was not quarantined");
      std::cout << "P2P_CANCEL rank=" << rank
                << " cancelled=1 quarantined=" << usage.slots_quarantined
                << std::endl;
      return;
    }
    const auto rounds_text = std::getenv("WEFTMOE_P2P_ROUNDS");
    const auto benchmark_rounds_text =
        std::getenv("WEFTMOE_P2P_BENCH_ROUNDS");
    const auto rounds = benchmark_p2p
                            ? (benchmark_rounds_text
                                   ? static_cast<std::uint32_t>(
                                         std::stoul(benchmark_rounds_text))
                                   : 110U)
                            : (rounds_text ? static_cast<std::uint32_t>(
                                                 std::stoul(rounds_text))
                                           : (socket_fallback || gin_rdma
                                                  ? 20U
                                                  : 1U));
    require(rounds > 0, "local P2P rounds must be positive");
    require(!benchmark_p2p || rounds > 10,
            "local P2P benchmark requires 10 warmups plus measurements");
    weftmoe::cuda::DeviceActivationTransportHandle transported;
    weftmoe::cuda::ActivationTransportValidationDownload downloaded;
    std::vector<weftmoe::transport::BFloat16> baseline(
        static_cast<std::size_t>(common_routes.num_tokens * hidden));
    std::vector<double> total_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> relay_samples;
    std::vector<double> synthetic_samples;
    std::vector<double> reduce_samples;
    const auto steady_api_before =
        weftmoe::cuda::testing::cuda_api_audit_snapshot();
    for (std::uint32_t round = 0; round < rounds; ++round) {
      weftmoe::cuda::ActivationTransportRunOptions run_options;
      if (benchmark_p2p)
        run_options.deadline_after = std::chrono::seconds(120);
      if (socket_fallback)
        transported = socket_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, run_options);
      else if (gin_rdma)
        transported = gin_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, run_options);
      else
        transported = local_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, run_options);
      const auto warmup = benchmark_p2p ? 10U : 0U;
      if (benchmark_p2p && round >= warmup) {
        total_samples.push_back(transported.timings().total_us);
        dispatch_samples.push_back(transported.timings().dispatch_put_us);
        relay_samples.push_back(transported.timings().relay_us);
        synthetic_samples.push_back(
            transported.timings().synthetic_return_us);
        reduce_samples.push_back(transported.timings().combine_reduce_us);
      }
      if (benchmark_p2p) {
        if (round + 1U != rounds)
          transported = {};
        continue;
      }
      downloaded = weftmoe::cuda::download_activation_transport_for_validation(
          transported, execution);
      if (round == 0)
        std::copy(downloaded.combined_tokens.begin(),
                  downloaded.combined_tokens.end(), baseline.begin());
      else
        require(downloaded.combined_tokens == baseline,
                "local P2P repeated generation changed output bytes");
      if (round + 1U != rounds) {
        downloaded = {};
        transported = {};
      }
    }
    require(transported.audit().backend ==
                (socket_fallback
                     ? weftmoe::transport::ActivationTransportBackend::
                           NcclSocketFallback
                     : gin_rdma
                           ? weftmoe::transport::ActivationTransportBackend::Gin
                     : weftmoe::transport::ActivationTransportBackend::
                           LocalCudaP2p),
            "concrete activation backend audit mismatch");
    require(transported.audit().activation_d2h_bytes == 0 &&
                transported.audit().activation_h2d_bytes == 0 &&
                transported.audit().hot_path_allocations == 0,
            "real local P2P path staged activation or allocated hot resources");
    if (socket_fallback || gin_rdma) {
      const auto oracle = weftmoe::transport::simulate_activation_transport(
          cross_host_oracle_dispatch, activations, route_gates,
          transport_config, 1);
      if (downloaded.contribution_by_route != oracle.contribution_by_route) {
        const auto mismatch = std::mismatch(
            downloaded.contribution_by_route.begin(),
            downloaded.contribution_by_route.end(),
            oracle.contribution_by_route.begin());
        const auto index = static_cast<std::size_t>(
            mismatch.first - downloaded.contribution_by_route.begin());
        throw std::runtime_error(
            "cross-host route contributions differ from host oracle: "
            "index=" + std::to_string(index) + ", actual=" +
            std::to_string(mismatch.first->bits) + ", expected=" +
            std::to_string(mismatch.second->bits));
      }
      require(downloaded.combined_tokens == oracle.combined_tokens,
              "cross-host combined output differs from host oracle");
    }
    const auto steady_api_after =
        weftmoe::cuda::testing::cuda_api_audit_snapshot();
    require(steady_api_after.device_allocations ==
                steady_api_before.device_allocations &&
                steady_api_after.device_frees == steady_api_before.device_frees &&
                steady_api_after.pinned_allocations ==
                    steady_api_before.pinned_allocations &&
                steady_api_after.pinned_frees == steady_api_before.pinned_frees &&
                steady_api_after.event_creates == steady_api_before.event_creates &&
                steady_api_after.event_destroys ==
                    steady_api_before.event_destroys &&
                steady_api_after.device_synchronizes ==
                    steady_api_before.device_synchronizes,
            "real local P2P steady path allocated or device-synchronized");
    if (benchmark_p2p) {
      const auto emit_values = [](const std::vector<double> &values) {
        for (std::size_t index = 0; index < values.size(); ++index) {
          if (index != 0)
            std::cout << ',';
          std::cout << values[index];
        }
      };
      const auto usage = transport_pool.usage();
      std::cout << "P2P_BENCH rank=" << rank
                << " backend=local-cuda-p2p setup_us=" << setup_us
                << " device_bytes=" << usage.device_bytes
                << " pinned_bytes=" << usage.pinned_bytes
                << " raw_bytes="
                << transported.audit().raw_cross_node_payload_bytes
                << " dedup_bytes="
                << transported.audit().deduplicated_cross_node_payload_bytes
                << " actual_dispatch_peer_bytes="
                << transported.audit().actual_dispatch_peer_bytes
                << " actual_return_peer_bytes="
                << transported.audit().actual_return_peer_bytes
                << " stalls=" << transported.audit().credit_stalls
                << " peak_ring=" << transported.audit().peak_ring_slots
                << " total=";
      emit_values(total_samples);
      std::cout << " dispatch=";
      emit_values(dispatch_samples);
      std::cout << " relay=";
      emit_values(relay_samples);
      std::cout << " synthetic=";
      emit_values(synthetic_samples);
      std::cout << " reduce=";
      emit_values(reduce_samples);
      std::cout << std::endl;
      return;
    }
    if (socket_fallback) {
      std::cout << "SOCKET_RESULT rank=" << rank
                << " backend="
                << weftmoe::transport::activation_transport_backend_name(
                       transported.audit().backend)
                << " tokens=" << common_routes.num_tokens
                << " top_k=" << top_k << " hidden=" << hidden
                << " packets=" << transported.audit().packet_count
                << " fanout=" << transported.audit().fanout_count
                << " raw_bytes="
                << transported.audit().raw_cross_node_payload_bytes
                << " dedup_bytes="
                << transported.audit().deduplicated_cross_node_payload_bytes
                << " gate_mode=" << gate_mode
                << " gate_hash=" << gate_hash
                << " output_hash="
                << hash_bf16_indexed(downloaded.combined_tokens)
                << " contribution_hash="
                << hash_bf16_indexed(downloaded.contribution_by_route)
                << " rounds=" << rounds
                << " oracle=1 total_us=" << transported.timings().total_us
                << std::endl;
      return;
    }
    if (gin_rdma) {
      std::cout << "GIN_RESULT rank=" << rank
                << " backend="
                << weftmoe::transport::activation_transport_backend_name(
                       transported.audit().backend)
                << " tokens=" << common_routes.num_tokens
                << " top_k=" << top_k << " hidden=" << hidden
                << " packets=" << transported.audit().packet_count
                << " fanout=" << transported.audit().fanout_count
                << " raw_remote_bytes="
                << transported.audit().raw_cross_node_payload_bytes
                << " dedup_remote_bytes="
                << transported.audit().deduplicated_cross_node_payload_bytes
                << " return_remote_bytes="
                << transported.audit().actual_return_peer_bytes
                << " gate_mode=" << gate_mode
                << " gate_hash=" << gate_hash
                << " output_hash="
                << hash_bf16_indexed(downloaded.combined_tokens)
                << " contribution_hash="
                << hash_bf16_indexed(downloaded.contribution_by_route)
                << " rounds=" << rounds
                << " oracle=1 total_us=" << transported.timings().total_us
                << std::endl;
      return;
    }
    const auto emit_bits = [](const auto &values) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << values[index].bits;
      }
    };
    std::cout << "P2P_RESULT rank=" << rank
              << " backend="
              << weftmoe::transport::activation_transport_backend_name(
                     transported.audit().backend)
              << " tokens=" << common_routes.num_tokens
              << " top_k=" << top_k << " hidden=" << hidden
              << " packets=" << transported.audit().packet_count
              << " fanout=" << transported.audit().fanout_count
              << " stalls=" << transported.audit().credit_stalls
              << " peak_ring=" << transported.audit().peak_ring_slots
              << " raw_bytes="
              << transported.audit().raw_cross_node_payload_bytes
              << " dedup_bytes="
              << transported.audit().deduplicated_cross_node_payload_bytes
              << " gate_mode=" << gate_mode
              << " gate_hash=" << gate_hash
              << " rounds=" << rounds
              << " total_us=" << transported.timings().total_us
              << " combined=";
    emit_bits(downloaded.combined_tokens);
    std::cout << " contribution=";
    emit_bits(downloaded.contribution_by_route);
    std::cout << " canonical=";
    emit_bits(downloaded.canonical_expert_rows);
    std::cout << std::endl;
    return;
  }
  if (scenario == "valid") {
    auto session = weftmoe::cuda::make_nccl_generation(
        base, "cuda-multinode", 701);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                execution,
                                                                pool);
    auto result = coordinator.run({common_config, common_routes, 10});
    std::cout << weftmoe::serialize_dispatch_metadata(
                     weftmoe::cuda::download_device_metadata_for_validation(
                         result.dispatch(), execution))
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
    auto session = weftmoe::cuda::make_nccl_generation(
        base, identity, generation);
    weftmoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                execution,
                                                                pool);
    try {
      (void)coordinator.run({config, routes, max_tokens});
      throw std::runtime_error("invalid cross-host generation passed");
    } catch (const weftmoe::NotifyDispatchError &error) {
      const auto audit = coordinator.audit();
      require(audit.control_starts == 1 &&
                  audit.expert_count_starts == 0 &&
                  audit.packet_summary_starts == 0,
              "failed cross-host generation entered data collective");
      std::cout << "CONTROL_ERROR round=" << round << " " << error.what()
                << std::endl;
    }

    auto recovery_session = weftmoe::cuda::make_nccl_generation(
        base, "cuda-multinode-recovery-" + scenario, 801 + round * 2);
    weftmoe::cuda::CudaNotifyGenerationCoordinator recovery(recovery_session,
                                                             execution,
                                                             pool);
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
    require(recovered && !weftmoe::serialize_dispatch_metadata(
                              weftmoe::cuda::download_device_metadata_for_validation(
                                  recovered.dispatch(), execution)).empty(),
            "cross-host recovery metadata missing");
    std::cout << "RECOVERY_OK rank=" << rank << " round=" << round
              << std::endl;
  }
  throw weftmoe::NotifyDispatchError(
      weftmoe::NotifyErrorCode::ControlProtocolFailure,
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

void transport_benchmark() {
  constexpr std::uint32_t world = 8;
  constexpr std::uint64_t tokens = 8192;
  constexpr std::uint64_t hidden = 7168;
  constexpr std::uint32_t experts = 384;
  constexpr std::uint32_t top_k = 6;
  const auto config = make_config(8, 1, experts, top_k, 64);
  // Use the exact requested shape and deterministic, duplicate-free tuples.
  std::vector<weftmoe::RankRoutingInput> exact(world);
  std::vector<std::uint64_t> counts(
      static_cast<std::size_t>(world) * experts, 0);
  std::vector<std::uint64_t> summaries(
      static_cast<std::size_t>(world) * world, 0);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    exact[rank].num_tokens = tokens;
    for (std::uint64_t token = 0; token < tokens; ++token) {
      bool seen_node[world]{};
      const auto base = static_cast<std::uint32_t>(
          (token * 17U + rank * 31U) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        // Distinct experts on one destination node make the node-level
        // payload-dedup traffic reduction directly measurable.
        const auto expert = (base + slot * 8U) % experts;
        exact[rank].selected_experts.push_back(
            static_cast<std::int32_t>(expert));
        ++counts[static_cast<std::size_t>(rank) * experts + expert];
        seen_node[config.topology.expert_to_rank[expert]] = true;
      }
      for (std::uint32_t node = 0; node < world; ++node)
        summaries[static_cast<std::size_t>(rank) * world + node] +=
            seen_node[node] ? 1U : 0U;
    }
  }
  auto execution = weftmoe::cuda::make_cuda_execution("transport-bench", 0);
  auto session = weftmoe::make_collective_session("transport-bench", world,
                                                   9100);
  auto context = weftmoe::make_rank_collective_context(
      session, 0, execution.token());
  auto dispatch = weftmoe::cuda::testing::materialize_fake_device_collectives(
      config, context, execution, exact[0], tokens, counts, summaries);
  weftmoe::cuda::ActivationTransportPoolConfig pool_config;
  pool_config.max_tokens_per_rank = tokens;
  pool_config.max_hidden_size = hidden;
  pool_config.max_top_k = top_k;
  pool_config.max_world_size = world;
  pool_config.max_nodes = world;
  pool_config.max_destination_rows_per_rank = tokens * top_k;
  pool_config.channels = 1;
  pool_config.ring_slots_per_source_channel = 8;
  pool_config.max_inflight_generations = 1;
  const auto setup_start = Clock::now();
  auto pool = weftmoe::cuda::make_activation_transport_pool(
      session, execution, pool_config);
  const auto setup_us = std::chrono::duration<double, std::micro>(
                            Clock::now() - setup_start)
                            .count();
  std::vector<weftmoe::transport::BFloat16> host(
      static_cast<std::size_t>(tokens * hidden));
  for (std::size_t i = 0; i < host.size(); ++i)
    host[i].bits = static_cast<std::uint16_t>(0x3f00U + (i % 127U));
  auto activation =
      weftmoe::cuda::testing::upload_bfloat16_activations_for_testing(
          host, tokens, hidden, context, execution, pool);
  weftmoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = hidden;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 8;
  weftmoe::cuda::DeviceActivationTransport transport(session, execution, pool);
  std::vector<double> total;
  std::vector<double> dispatch_relay;
  std::vector<double> combine_reduce;
  weftmoe::cuda::ActivationTransportDeviceAudit last_audit;
  for (std::uint32_t iteration = 0; iteration < 60; ++iteration) {
    auto result = transport.run_device_simulation_synthetic(
        dispatch, activation, transport_config);
    if (iteration >= 10) {
      total.push_back(result.timings().total_us);
      dispatch_relay.push_back(result.timings().dispatch_relay_us);
      combine_reduce.push_back(result.timings().combine_reduce_us);
    }
    last_audit = result.audit();
  }
  const auto total_stats = summarize(total);
  const auto dispatch_stats = summarize(dispatch_relay);
  const auto reduce_stats = summarize(combine_reduce);
  const auto usage = pool.usage();
  const auto print_stats = [](const char *name, const Statistics &stats) {
    std::cout << "  " << name << " mean=" << stats.mean
              << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  };
  std::cout << "Activation transport benchmark: tokens/rank=" << tokens
            << " hidden=" << hidden << " top_k=" << top_k
            << " logical_world=" << world
            << " backend=device-simulation (no GIN/RDMA)\n"
            << "  one_time_pool_setup_us=" << setup_us << '\n'
            << "  raw_cross_node_payload_bytes="
            << last_audit.raw_cross_node_payload_bytes << '\n'
            << "  deduplicated_cross_node_payload_bytes="
            << last_audit.deduplicated_cross_node_payload_bytes << '\n'
            << "  device_workspace_bytes=" << usage.device_bytes << '\n'
            << "  pinned_workspace_bytes=" << usage.pinned_bytes << '\n';
  print_stats("dispatch_relay_us", dispatch_stats);
  print_stats("combine_reduce_us", reduce_stats);
  print_stats("transport_steady_total_us", total_stats);
  std::cout
      << "TRANSPORT_BENCH_JSON {\"schema\":1,\"warmup\":10,\"measured\":50,"
         "\"backend\":\"device-simulation\",\"one_time_pool_setup_us\":"
      << setup_us << ",\"raw_cross_node_payload_bytes\":"
      << last_audit.raw_cross_node_payload_bytes
      << ",\"deduplicated_cross_node_payload_bytes\":"
      << last_audit.deduplicated_cross_node_payload_bytes
      << ",\"device_workspace_bytes\":" << usage.device_bytes
      << ",\"pinned_workspace_bytes\":" << usage.pinned_bytes
      << ",\"dispatch_relay_us\":{\"mean\":" << dispatch_stats.mean
      << ",\"p50\":" << dispatch_stats.p50 << ",\"p95\":"
      << dispatch_stats.p95 << ",\"min\":" << dispatch_stats.minimum
      << ",\"max\":" << dispatch_stats.maximum
      << "},\"combine_reduce_us\":{\"mean\":" << reduce_stats.mean
      << ",\"p50\":" << reduce_stats.p50 << ",\"p95\":"
      << reduce_stats.p95 << ",\"min\":" << reduce_stats.minimum
      << ",\"max\":" << reduce_stats.maximum
      << "},\"steady_total_us\":{\"mean\":" << total_stats.mean
      << ",\"p50\":" << total_stats.p50 << ",\"p95\":"
      << total_stats.p95 << ",\"min\":" << total_stats.minimum
      << ",\"max\":" << total_stats.maximum << "}}\n";
}

void benchmark(std::uint32_t world) {
  const auto config = make_config(1, world, 384, 6, 64);
  std::vector<weftmoe::RankRoutingInput> routes(world);
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
  constexpr std::size_t measured = 100;
  std::vector<double> steady;
  std::vector<double> setup;
  std::vector<double> generation_end_to_end;
  using SetupMember = double weftmoe::cuda::GenerationSetupTimings::*;
  const std::vector<std::pair<const char *, SetupMember>> setup_stages{
      {"pool_acquire_us", &weftmoe::cuda::GenerationSetupTimings::pool_acquire_us},
      {"coordinator_backend_create_us", &weftmoe::cuda::GenerationSetupTimings::coordinator_backend_create_us},
      {"control_slot_allocation_us", &weftmoe::cuda::GenerationSetupTimings::control_slot_allocation_us},
      {"config_slot_allocation_us", &weftmoe::cuda::GenerationSetupTimings::config_slot_allocation_us},
      {"count_slot_allocation_us", &weftmoe::cuda::GenerationSetupTimings::count_slot_allocation_us},
      {"summary_slot_allocation_us", &weftmoe::cuda::GenerationSetupTimings::summary_slot_allocation_us},
      {"event_allocation_us", &weftmoe::cuda::GenerationSetupTimings::event_allocation_us},
      {"pinned_allocation_us", &weftmoe::cuda::GenerationSetupTimings::pinned_allocation_us},
      {"device_allocation_us", &weftmoe::cuda::GenerationSetupTimings::device_allocation_us}};
  std::vector<std::vector<double>> setup_stage_samples(setup_stages.size());
  using ExecutionMember =
      double weftmoe::cuda::GenerationExecutionTimings::*;
  const std::vector<std::pair<const char *, ExecutionMember>> execution_stages{
      {"local_mapping_us",
       &weftmoe::cuda::GenerationExecutionTimings::local_mapping_us},
      {"control_config_us",
       &weftmoe::cuda::GenerationExecutionTimings::control_config_us},
      {"expert_count_us",
       &weftmoe::cuda::GenerationExecutionTimings::expert_count_us},
      {"packet_summary_us",
       &weftmoe::cuda::GenerationExecutionTimings::packet_summary_us},
      {"device_materialization_us",
       &weftmoe::cuda::GenerationExecutionTimings::device_materialization_us},
      {"prefix_planning_us",
       &weftmoe::cuda::GenerationExecutionTimings::prefix_planning_us},
      {"remaining_parallel_materialization_us",
       &weftmoe::cuda::GenerationExecutionTimings::remaining_parallel_materialization_us},
      {"pool_acquire_release_us",
       &weftmoe::cuda::GenerationExecutionTimings::pool_acquire_release_us}};
  std::vector<std::vector<double>> execution_stage_samples(
      execution_stages.size());
  weftmoe::cuda::NcclWorkspaceUsage peak;
  // One communicator/session establishment per rank, once for the complete
  // 10+50 run. Samples are the ranks participating in that one establishment.
  auto resources =
      make_reusable_resources(config, "benchmark-session", 2999, 8192);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    const auto context = weftmoe::make_rank_collective_context(
        resources[rank].base_session.session(), rank,
        resources[rank].execution.token());
    resources[rank].device_routes =
        weftmoe::cuda::upload_cuda_routes(
            routes[rank], config.top_k,
            context,
            resources[rank].execution);
    resources[rank].device_gates =
        weftmoe::cuda::testing::upload_fp32_gates_for_testing(
            fixed_random_selected_gates(routes[rank].num_tokens,
                                        config.top_k, rank,
                                        0x4d4547414d4f4547ULL),
            resources[rank].device_routes, context,
            resources[rank].execution);
  }
  std::vector<double> session_setup;
  std::vector<double> pool_setup;
  for (const auto &resource : resources)
    session_setup.push_back(resource.session_setup_us);
  for (const auto &resource : resources)
    pool_setup.push_back(resource.pool_setup_us);
  const auto session_setup_stats = summarize(session_setup);
  const auto pool_setup_stats = summarize(pool_setup);
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
      std::vector<double> slow_execution_stages(execution_stages.size(), 0.0);
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
        for (std::size_t stage = 0; stage < execution_stages.size(); ++stage)
          slow_execution_stages[stage] = std::max(
              slow_execution_stages[stage],
              rank.execution.*execution_stages[stage].second);
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
      for (std::size_t stage = 0; stage < execution_stages.size(); ++stage)
        execution_stage_samples[stage].push_back(
            slow_execution_stages[stage]);
    }
  }
  const auto steady_stats = summarize(steady);
  const auto setup_stats = summarize(setup);
  const auto generation_stats = summarize(generation_end_to_end);
  std::vector<Statistics> setup_stage_stats;
  for (auto &samples : setup_stage_samples)
    setup_stage_stats.push_back(summarize(std::move(samples)));
  std::vector<Statistics> execution_stage_stats;
  for (auto &samples : execution_stage_samples)
    execution_stage_stats.push_back(summarize(std::move(samples)));
  std::cout << "BENCH world=" << world
            << " warmup=" << warmup << " measured=" << measured
            << " aggregation=slowest-rank"
               " session_reused=true\n"
            << "  one_time_session_setup_us mean="
            << session_setup_stats.mean << " p50=" << session_setup_stats.p50
            << " p95=" << session_setup_stats.p95
            << " min=" << session_setup_stats.minimum
            << " max=" << session_setup_stats.maximum << '\n'
            << "  one_time_pool_setup_us mean=" << pool_setup_stats.mean
            << " p50=" << pool_setup_stats.p50
            << " p95=" << pool_setup_stats.p95
            << " min=" << pool_setup_stats.minimum
            << " max=" << pool_setup_stats.maximum << '\n'
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
  for (std::size_t stage = 0; stage < execution_stages.size(); ++stage) {
    const auto &stats = execution_stage_stats[stage];
    std::cout << "  " << execution_stages[stage].first << " mean="
              << stats.mean << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  }
  std::cout << "BENCH_JSON {\"schema\":7,\"warmup\":" << warmup
            << ",\"measured\":" << measured << ','
            << "\"aggregation\":\"slowest-rank\",\"session_reused\":true,"
               "\"session_setup_sample\":\"one establishment; samples are ranks\","
               "\"one_time_session_setup_us\":{\"mean\":"
            << session_setup_stats.mean << ",\"p50\":"
            << session_setup_stats.p50 << ",\"p95\":"
            << session_setup_stats.p95 << ",\"min\":"
            << session_setup_stats.minimum << ",\"max\":"
            << session_setup_stats.maximum
            << "},\"one_time_pool_setup_us\":{\"mean\":"
            << pool_setup_stats.mean << ",\"p50\":"
            << pool_setup_stats.p50 << ",\"p95\":"
            << pool_setup_stats.p95 << ",\"min\":"
            << pool_setup_stats.minimum << ",\"max\":"
            << pool_setup_stats.maximum
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
  std::cout << "},\"execution_stages\":{";
  for (std::size_t stage = 0; stage < execution_stages.size(); ++stage) {
    if (stage != 0)
      std::cout << ',';
    const auto &stats = execution_stage_stats[stage];
    std::cout << '\"' << execution_stages[stage].first
              << "\":{\"mean\":" << stats.mean << ",\"p50\":" << stats.p50
              << ",\"p95\":" << stats.p95 << ",\"min\":" << stats.minimum
              << ",\"max\":" << stats.maximum << '}';
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
  const auto info = weftmoe::cuda::nccl_runtime_info();
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
  using weftmoe::cuda::testing::GenerationSchedulerModel;
  using weftmoe::cuda::transport_state::TaskClass;
  constexpr auto kinds = weftmoe::cuda::transport_state::kTaskClassCount;
  constexpr std::uint64_t seed_count = 256;
  for (std::uint64_t seed = 1; seed <= seed_count; ++seed) {
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
  std::cout << "[PASS] randomized static generation scheduler seeds="
            << seed_count << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--local") {
      const auto world = static_cast<std::uint32_t>(parse_u64(argv[2]));
      print_nccl_runtime_info();
      test_local(world);
      if (world == 2) {
        test_device_capacity_failure_and_recovery();
#if !WEFTMOE_DEBUG_VALIDATE
        test_release_fused_status_and_config_fallback();
#endif
        test_protocol_failure_suite();
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
      test_protocol_failure_suite();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--device-smoke") {
      test_device_only_sanitizer_paths();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--v3-protocol-smoke") {
      const auto probe =
          weftmoe::cuda::testing::run_persistent_v3_protocol_probe();
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
      const auto backward_probe =
          weftmoe::cuda::testing::run_backward_scheduler_probe();
      require(backward_probe.dx_completed == 36ULL &&
                  backward_probe.dab_completed == 60ULL &&
                  backward_probe.dw2_completed == 28ULL &&
                  backward_probe.dw1_completed == 268ULL &&
                  backward_probe.total_completed == 392ULL &&
                  backward_probe.duplicate_claims == 0ULL &&
                  backward_probe.dependency_errors == 0ULL &&
                  backward_probe.priority_errors == 0ULL &&
                  backward_probe.timeout_errors == 0ULL &&
                  backward_probe.terminal_epoch != 0ULL &&
                  backward_probe.terminal_publish_count == 1ULL,
              "BWD four-queue scheduler probe failed");
      std::cout << "[PASS] BWD scheduler compute="
                << backward_probe.compute_ctas
                << " participating=" << backward_probe.participating_ctas
                << " tasks=" << backward_probe.total_completed
                << " priority=dx>dab>dw2>dw1\n";
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
          weftmoe::cuda::probe_activation_transport_capability(8);
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
          weftmoe::cuda::testing::run_local_p2p_put_signal_probe(
              0, 1, 64U * 1024U * 1024U);
      std::cout << "LOCAL_P2P_ORDERING bytes="
                << (64U * 1024U * 1024U)
                << " last_block_signal_verified=" << ordering_ok
                << " early_signal_rejected=" << ordering_ok << '\n';
      std::cout << "LOCAL_P2P_DEVICE_PUT_GBPS gpu0_to_gpu1="
                << weftmoe::cuda::testing::measure_local_p2p_put_bandwidth_gbps(
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
