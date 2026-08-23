#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "megamoe/api/errors.hpp"
#include "megamoe/cuda/device_notify.hpp"
#include "megamoe/cuda/nccl_count_exchange.hpp"
#include "megamoe/metadata/serialize.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

megamoe::RuntimeConfig make_config(std::uint32_t nodes,
                                   std::uint32_t gpus_per_node,
                                   std::uint32_t experts,
                                   std::uint32_t top_k,
                                   std::uint32_t mtile) {
  megamoe::RuntimeConfig config;
  config.num_experts = experts;
  config.top_k = top_k;
  config.m_tile_rows = mtile;
  config.topology.num_nodes = nodes;
  config.topology.world_size = nodes * gpus_per_node;
  for (std::uint32_t node = 0; node < nodes; ++node) {
    for (std::uint32_t local = 0; local < gpus_per_node; ++local) {
      config.topology.rank_to_node.push_back(node);
      config.topology.rank_to_local_rank.push_back(local);
    }
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
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };
  std::vector<megamoe::RankRoutingInput> routes(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    routes[rank].num_tokens = next() % (max_tokens + 1);
    for (std::uint64_t token = 0; token < routes[rank].num_tokens; ++token) {
      std::vector<std::uint32_t> candidates(experts);
      for (std::uint32_t expert = 0; expert < experts; ++expert)
        candidates[expert] = expert;
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        const auto selected = slot +
            static_cast<std::uint32_t>(next() % (experts - slot));
        std::swap(candidates[slot], candidates[selected]);
        routes[rank].selected_experts.push_back(
            static_cast<std::int32_t>(candidates[slot]));
      }
    }
  }
  return routes;
}

megamoe::RankRoutingInput seeded_rank_input(
    std::uint32_t world, std::uint32_t experts, std::uint32_t top_k,
    std::uint64_t seed, std::uint32_t target_rank,
    std::uint64_t max_tokens = 10) {
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };
  megamoe::RankRoutingInput result;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    const auto tokens = next() % (max_tokens + 1);
    if (rank == target_rank)
      result.num_tokens = tokens;
    for (std::uint64_t token = 0; token < tokens; ++token) {
      std::vector<std::uint32_t> candidates(experts);
      for (std::uint32_t expert = 0; expert < experts; ++expert)
        candidates[expert] = expert;
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        const auto selected =
            slot + static_cast<std::uint32_t>(next() % (experts - slot));
        std::swap(candidates[slot], candidates[selected]);
        if (rank == target_rank)
          result.selected_experts.push_back(
              static_cast<std::int32_t>(candidates[slot]));
      }
    }
  }
  return result;
}

struct Timings {
  double mapping_us = 0;
  double collective_us = 0;
  double materialize_us = 0;
  double end_to_end_us = 0;
  std::uint64_t workspace_bytes = 0;
};

struct StageSample {
  double mapping_cuda_us = 0;
  double mapping_wall_us = 0;
  double control_completion_us = 0;
  double count_completion_us = 0;
  double summary_completion_us = 0;
  double collective_completion_us = 0;
  double materialize_wall_us = 0;
  double end_to_end_wall_us = 0;
};

struct WorkspaceSample {
  std::uint64_t route_input_device_bytes = 0;
  std::uint64_t device_arena_bytes = 0;
  std::uint64_t pinned_staging_bytes = 0;
  std::uint64_t nccl_device_bytes = 0;
  std::uint64_t nccl_pinned_bytes = 0;
  std::uint64_t final_metadata_resident_arena_bytes = 0;
};

struct RankResult {
  std::vector<std::string> serialized;
  std::vector<StageSample> samples;
  WorkspaceSample workspace;
  Timings timings;
  std::exception_ptr error;
};

std::vector<std::uint64_t> fixed_seeds() {
  return {0x5eed1234ULL, 0x10203040ULL, 0xdeadbeefULL, 0xc001d00dULL,
          0x01234567ULL, 0x89abcdefULL, 0x31415926ULL, 0x27182818ULL,
          0x0badf00dULL, 0xfeedfaceULL, 0x13579bdfULL, 0x2468ace0ULL,
          0xa5a5a5a5ULL, 0x5a5a5a5aULL, 0x11223344ULL, 0x55667788ULL,
          0x99aabbccULL, 0xddeeff00ULL, 0x42424242ULL, 0x77777777ULL,
          0xabcdef01ULL, 0x10fedcbaULL, 0x600dbeefULL, 0x7f4a7c15ULL,
          0xf00dcafeULL};
}

std::vector<RankResult> run_local_collective(
    const megamoe::RuntimeConfig &config,
    const std::vector<std::vector<megamoe::RankRoutingInput>> &fixtures,
    const std::string &identity, std::uint64_t generation,
    bool stagger_enqueue, std::size_t timing_skip = 0,
    bool capture_serialized = true) {
  const auto world = config.topology.world_size;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<RankResult> results(world);
  std::vector<std::thread> threads;
  threads.reserve(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        auto execution = megamoe::cuda::make_cuda_execution(
            identity + "-stream-" + std::to_string(rank), device);
        auto base_session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        for (std::size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
          // Each fixture is a fresh framework generation over the same healthy
          // communicator. The wire generation intentionally remains stable so
          // the repeated-input serialization check is byte-for-byte.
          auto session = fixture == 0
                             ? base_session
                             : megamoe::cuda::make_nccl_generation(
                                   base_session, identity, generation);
          auto context = megamoe::make_rank_collective_context(
              session.session(), rank, execution.token());
          megamoe::cuda::NcclAllGatherCountExchange backend(
              session, execution, config.num_experts,
              config.topology.num_nodes);
          const auto all_start = Clock::now();
          auto routes = megamoe::cuda::upload_cuda_routes(
              fixtures[fixture][rank], config.top_k, execution);
          const auto plan = megamoe::cuda::query_cuda_notify_workspace(
              config, rank, fixtures[fixture][rank].num_tokens,
              stagger_enqueue ? 10 : 8192);
          results[rank].timings.workspace_bytes =
              std::max(results[rank].timings.workspace_bytes,
                       plan.device_bytes + plan.host_staging_bytes);
          auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
              plan, context, execution);
          results[rank].workspace.route_input_device_bytes = std::max(
              results[rank].workspace.route_input_device_bytes,
              routes.device_bytes());
          results[rank].workspace.device_arena_bytes = std::max(
              results[rank].workspace.device_arena_bytes, plan.device_bytes);
          results[rank].workspace.pinned_staging_bytes = std::max(
              results[rank].workspace.pinned_staging_bytes,
              plan.host_staging_bytes);
          const auto map_start = Clock::now();
          auto local_operation = megamoe::cuda::start_cuda_rank_local_notify(
              config, context, execution, routes, std::move(workspace));
          auto preflight = local_operation.wait_preflight();
          const auto map_end = Clock::now();
          if (stagger_enqueue && ((rank + fixture) % 3U) != 0)
            std::this_thread::sleep_for(std::chrono::microseconds(
                20U * ((rank + static_cast<std::uint32_t>(fixture)) % 3U)));
          const auto collective_start = Clock::now();
          const auto record = megamoe::cuda::make_control_record(
              config, context, fixtures[fixture][rank].num_tokens,
              fixtures[fixture][rank].selected_experts.size(),
              stagger_enqueue ? 10U : 8192U, preflight.status());
          auto control = backend.start_control(context, record);
          const auto control_done = control.wait();
          const auto control_end = Clock::now();
          require(static_cast<bool>(control_done),
                  "control completion handle is empty");
          require(preflight.valid(),
                  "control accepted an invalid local preflight");
          const auto &local = preflight.local();
          auto count = megamoe::start_count_exchange(local.local(), backend);
          auto count_done = count.wait();
          const auto count_end = Clock::now();
          auto summary =
              megamoe::start_notify_summary_exchange(local.local(), backend);
          auto summary_done = summary.wait();
          const auto collective_end = Clock::now();
          const auto nccl_usage = backend.workspace_usage();
          results[rank].workspace.nccl_device_bytes = std::max(
              results[rank].workspace.nccl_device_bytes,
              nccl_usage.device_bytes);
          results[rank].workspace.nccl_pinned_bytes = std::max(
              results[rank].workspace.nccl_pinned_bytes,
              nccl_usage.pinned_host_bytes);
          const auto materialize_start = Clock::now();
          auto host = megamoe::materialize_rank_metadata(
              local.local(), count_done, summary_done);
          auto device_handle = megamoe::cuda::stage_cuda_dispatch_metadata(
              std::move(host), local, execution);
          // Final metadata is staged into the already-owned notify arena; this
          // is a resident logical category, not an additional allocation.
          results[rank].workspace.final_metadata_resident_arena_bytes =
              std::max(results[rank].workspace.final_metadata_resident_arena_bytes,
                       plan.device_bytes);
          const auto materialize_end = Clock::now();
          if (stagger_enqueue)
            megamoe::cuda::validate_cuda_dispatch_metadata(device_handle,
                                                            execution);
          require(static_cast<bool>(device_handle.device().expert_id) ||
                      fixtures[fixture][rank].num_tokens == 0,
                  "device route metadata lease is empty");
          const auto audit = backend.audit();
          require(audit.control_starts == 1 &&
                      audit.expert_count_starts == 1 &&
                      audit.packet_summary_starts == 1,
                  "collective protocol audit does not equal control/count/summary");
          if (capture_serialized)
            results[rank].serialized.push_back(
                megamoe::serialize_dispatch_metadata(device_handle.host()));
          if (fixture >= timing_skip) {
            results[rank].samples.push_back(StageSample{
                preflight.device_mapping_us(),
                std::chrono::duration<double, std::micro>(map_end - map_start)
                    .count(),
                std::chrono::duration<double, std::micro>(control_end -
                                                          collective_start)
                    .count(),
                std::chrono::duration<double, std::micro>(count_end -
                                                          control_end)
                    .count(),
                std::chrono::duration<double, std::micro>(collective_end -
                                                          count_end)
                    .count(),
                std::chrono::duration<double, std::micro>(collective_end -
                                                          collective_start)
                    .count(),
                std::chrono::duration<double, std::micro>(materialize_end -
                                                          materialize_start)
                    .count(),
                std::chrono::duration<double, std::micro>(materialize_end -
                                                          all_start)
                    .count()});
            results[rank].timings.mapping_us +=
                std::chrono::duration<double, std::micro>(map_end - map_start)
                    .count();
            results[rank].timings.collective_us +=
                std::chrono::duration<double, std::micro>(collective_end -
                                                          collective_start)
                    .count();
            results[rank].timings.materialize_us +=
                std::chrono::duration<double, std::micro>(materialize_end -
                                                          materialize_start)
                    .count();
            results[rank].timings.end_to_end_us +=
                std::chrono::duration<double, std::micro>(materialize_end -
                                                          all_start)
                    .count();
          }
        }
      } catch (...) {
        results[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    if (results[rank].error)
      std::rethrow_exception(results[rank].error);
  }
  return results;
}

void test_local(std::uint32_t world) {
  const auto experts = std::max<std::uint32_t>(16, world * 2);
  const auto top_k = std::min<std::uint32_t>(6, experts);
  const auto config = make_config(1, world, experts, top_k, 7);
  std::vector<std::vector<megamoe::RankRoutingInput>> fixtures;
  for (const auto seed : fixed_seeds())
    fixtures.push_back(seeded_inputs(world, experts, top_k, seed));
  fixtures.emplace_back(world); // Every rank empty.
  std::vector<megamoe::RankRoutingInput> skew(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    skew[rank].num_tokens = 10;
    for (std::uint64_t token = 0; token < skew[rank].num_tokens; ++token) {
      skew[rank].selected_experts.push_back(0);
      for (std::uint32_t slot = 1; slot < top_k; ++slot)
        skew[rank].selected_experts.push_back(static_cast<std::int32_t>(
            1U + ((rank * 7U + static_cast<std::uint32_t>(token) + slot) %
                  (experts - 1U))));
    }
  }
  fixtures.push_back(std::move(skew));
  auto repeat = seeded_inputs(world, experts, top_k, 0x1234abcdULL);
  const auto repeat_begin = fixtures.size();
  for (unsigned iteration = 0; iteration < 20; ++iteration)
    fixtures.push_back(repeat);
  const std::string identity = "cuda-local-" + std::to_string(world);
  constexpr std::uint64_t generation = 501;
  const auto actual = run_local_collective(config, fixtures, identity,
                                           generation, true);
  for (std::size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, fixtures[fixture], identity, generation);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      const auto reference = megamoe::serialize_dispatch_metadata(expected[rank]);
      require(actual[rank].serialized[fixture] == reference,
              "CUDA/NCCL full-field mismatch fixture=" +
                  std::to_string(fixture) + ", rank=" +
                  std::to_string(rank));
    }
  }
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    for (std::size_t fixture = repeat_begin + 1; fixture < fixtures.size();
         ++fixture) {
      require(actual[rank].serialized[fixture] ==
                  actual[rank].serialized[repeat_begin],
              "20-run byte determinism mismatch rank=" +
                  std::to_string(rank));
    }
  }
  std::cout << "[PASS] real CUDA/NCCL local world=" << world
            << ": 25 fixed seeds + empty/skew/partial-tile targets + "
               "20 byte-identical repeats\n";
}

void test_smoke() {
  const auto config = make_config(1, 1, 8, 3, 4);
  const auto fixtures = std::vector<std::vector<megamoe::RankRoutingInput>>{
      seeded_inputs(1, 8, 3, 0x5eedULL, 3)};
  constexpr const char *identity = "cuda-sanitizer-smoke";
  constexpr std::uint64_t generation = 502;
  const auto actual = run_local_collective(config, fixtures, identity,
                                           generation, true);
  const auto expected = megamoe::build_notify_dispatch_for_testing(
      config, fixtures.front(), identity, generation);
  require(actual.front().serialized.front() ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "CUDA sanitizer smoke full-field mismatch");
  std::cout << "[PASS] CUDA/NCCL sanitizer smoke full-field cross-check\n";
}

void test_device_kernel_smoke() {
  const auto config = make_config(1, 1, 8, 3, 4);
  auto execution = megamoe::cuda::make_cuda_execution(
      "cuda-device-sanitizer-stream", 0);
  auto session = megamoe::make_collective_session(
      "cuda-device-sanitizer", 1, 503);
  auto context = megamoe::make_rank_collective_context(
      session, 0, execution.token());
  megamoe::RankRoutingInput input{3, {5, 1, 7, 1, 2, 3, 0, 7, 4}};
  auto routes = megamoe::cuda::upload_cuda_routes(input, config.top_k,
                                                   execution);
  auto plan = megamoe::cuda::query_cuda_notify_workspace(config, 0, 3, 3);
  auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      plan, context, execution);
  auto operation = megamoe::cuda::start_cuda_rank_local_notify(
      config, context, execution, routes, std::move(workspace));
  auto actual = operation.wait();
  auto expected = megamoe::build_rank_local_histogram(config, context, input);
  const auto &lhs = actual.local().outgoing();
  const auto &rhs = expected.outgoing();
  require(lhs.expert_id == rhs.expert_id &&
              lhs.local_offset == rhs.local_offset &&
              lhs.route_dst_rank == rhs.route_dst_rank &&
              lhs.route_dst_node == rhs.route_dst_node &&
              lhs.route_dst_local_rank == rhs.route_dst_local_rank &&
              lhs.combine_order == rhs.combine_order &&
              lhs.destination_node_mask == rhs.destination_node_mask &&
              actual.local().local_expert_count().values() ==
                  expected.local_expert_count().values() &&
              actual.local().packet_count_by_destination_node().values() ==
                  expected.packet_count_by_destination_node().values(),
          "device kernel sanitizer smoke mismatch");
  std::cout << "[PASS] CUDA device-only stable mapping sanitizer smoke\n";
}

void test_invalid_inputs() {
  const auto config = make_config(1, 1, 8, 3, 4);
  auto execution = megamoe::cuda::make_cuda_execution("invalid-stream", 0);
  auto session = megamoe::make_collective_session("cuda-invalid", 1, 91);
  auto context = megamoe::make_rank_collective_context(
      session, 0, execution.token());
  const auto expect_code = [&](megamoe::RankRoutingInput input,
                               megamoe::NotifyErrorCode code) {
    bool failed = false;
    try {
      auto routes = megamoe::cuda::upload_cuda_routes(input, config.top_k,
                                                       execution);
      auto plan = megamoe::cuda::query_cuda_notify_workspace(
          config, 0, input.num_tokens, input.num_tokens);
      auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
          plan, context, execution);
      auto operation = megamoe::cuda::start_cuda_rank_local_notify(
          config, context, execution, routes, std::move(workspace));
      (void)operation.wait();
    } catch (const megamoe::NotifyDispatchError &error) {
      failed = error.code() == code;
      require(std::string(error.what()).find("phase=cuda-device-validation") !=
                  std::string::npos,
              "device validation error lacks phase: " +
                  std::string(error.what()));
    }
    require(failed, "invalid device routes did not fail with expected code");
  };
  expect_code({1, {1, 1, 2}}, megamoe::NotifyErrorCode::DuplicateExpert);
  expect_code({1, {1, 8, 2}}, megamoe::NotifyErrorCode::ExpertOutOfRange);

  auto plan = megamoe::cuda::query_cuda_notify_workspace(config, 0, 2, 2);
  --plan.device_bytes;
  bool short_plan_failed = false;
  try {
    (void)megamoe::cuda::allocate_cuda_notify_workspace(plan, context,
                                                         execution);
  } catch (const megamoe::NotifyDispatchError &) {
    short_plan_failed = true;
  }
  require(short_plan_failed, "short CUDA workspace plan was accepted");
  std::cout << "[PASS] CUDA duplicate/out-of-range/workspace capacity rejection\n";
}

void test_device_masks_dedup_and_capacity() {
  auto execution = megamoe::cuda::make_cuda_execution("mask-stream", 0);
  auto many = make_config(73, 1, 73, 2, 8);
  auto many_session = megamoe::make_collective_session("many-device", 73, 92);
  auto many_context = megamoe::make_rank_collective_context(
      many_session, 0, execution.token());
  megamoe::RankRoutingInput many_routes{1, {0, 72}};
  auto routes = megamoe::cuda::upload_cuda_routes(many_routes, 2, execution);
  auto plan = megamoe::cuda::query_cuda_notify_workspace(many, 0, 1, 1);
  auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      plan, many_context, execution);
  auto operation = megamoe::cuda::start_cuda_rank_local_notify(
      many, many_context, execution, routes, std::move(workspace));
  auto local = operation.wait();
  require(local.local().outgoing().node_mask_words == 2,
          "73-node mask did not use two words");
  require(local.local().outgoing().destination_node_mask ==
              std::vector<std::uint64_t>({1, 1ULL << 8U}),
          "73-node device mask mismatch");

  auto dedup = make_config(2, 1, 4, 2, 4);
  dedup.topology.expert_to_rank = {0, 0, 1, 1};
  auto dedup_session = megamoe::make_collective_session("dedup-device", 2, 93);
  auto dedup_context = megamoe::make_rank_collective_context(
      dedup_session, 0, execution.token());
  megamoe::RankRoutingInput dedup_routes{1, {2, 3}};
  auto dedup_input = megamoe::cuda::upload_cuda_routes(dedup_routes, 2,
                                                        execution);
  auto dedup_plan = megamoe::cuda::query_cuda_notify_workspace(dedup, 0, 1, 1);
  auto dedup_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      dedup_plan, dedup_context, execution);
  auto dedup_operation = megamoe::cuda::start_cuda_rank_local_notify(
      dedup, dedup_context, execution, dedup_input,
      std::move(dedup_workspace));
  auto dedup_local = dedup_operation.wait();
  require(dedup_local.local().packet_count_by_destination_node().values() ==
              std::vector<std::uint64_t>({0, 1}),
          "token-node device packet summary did not deduplicate fan-out");

  auto capacity = make_config(1, 1, 8, 3, 4);
  capacity.capacity.expert_rows.assign(8, 100);
  capacity.capacity.expert_rows[1] = 0;
  const auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto capacity_session = megamoe::cuda::make_nccl_session(
      "capacity-device", 1, 0, 94, 0, bootstrap);
  auto capacity_context = megamoe::make_rank_collective_context(
      capacity_session.session(), 0, execution.token());
  megamoe::cuda::NcclAllGatherCountExchange backend(
      capacity_session, execution, 8, 1);
  megamoe::RankRoutingInput capacity_routes{1, {1, 2, 3}};
  auto capacity_input = megamoe::cuda::upload_cuda_routes(
      capacity_routes, 3, execution);
  auto capacity_plan = megamoe::cuda::query_cuda_notify_workspace(
      capacity, 0, 1, 1);
  auto capacity_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      capacity_plan, capacity_context, execution);
  auto capacity_notify = megamoe::cuda::start_cuda_rank_local_notify(
      capacity, capacity_context, execution, capacity_input,
      std::move(capacity_workspace));
  auto capacity_preflight = capacity_notify.wait_preflight();
  auto capacity_control = backend.start_control(
      capacity_context,
      megamoe::cuda::make_control_record(
          capacity, capacity_context, capacity_routes.num_tokens,
          capacity_routes.selected_experts.size(), 1,
          capacity_preflight.status()));
  (void)capacity_control.wait();
  require(capacity_preflight.valid(), "capacity preflight unexpectedly failed");
  const auto &capacity_local = capacity_preflight.local();
  auto counts = megamoe::start_count_exchange(capacity_local.local(), backend);
  auto counts_done = counts.wait();
  auto summary = megamoe::start_notify_summary_exchange(capacity_local.local(),
                                                         backend);
  auto summary_done = summary.wait();
  bool capacity_failed = false;
  try {
    (void)megamoe::materialize_rank_metadata(
        capacity_local.local(), counts_done, summary_done);
  } catch (const megamoe::NotifyDispatchError &error) {
    capacity_failed =
        error.code() == megamoe::NotifyErrorCode::ExpertCapacityExceeded;
  }
  require(capacity_failed, "device/NCCL path accepted insufficient expert capacity");
  std::cout << "[PASS] CUDA 73-node mask, token-node dedup, and capacity rejection\n";
}

void test_nccl_context_rejections() {
  auto execution = megamoe::cuda::make_cuda_execution(
      "nccl-context-stream", 0);
  const auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto session = megamoe::cuda::make_nccl_session(
      "nccl-context", 1, 0, 95, 0, bootstrap);
  megamoe::cuda::NcclAllGatherCountExchange backend(
      session, execution, 8, 1);
  const auto config = make_config(1, 1, 8, 3, 4);
  const megamoe::RankRoutingInput input{1, {1, 2, 3}};
  const auto expect_exchange_failure = [&](const auto &local,
                                           const std::string &needle) {
    bool failed = false;
    try {
      auto operation = megamoe::start_count_exchange(local, backend);
      (void)operation.wait();
    } catch (const megamoe::NotifyDispatchError &error) {
      failed = error.code() == megamoe::NotifyErrorCode::CountExchangeFailure;
      require(std::string(error.what()).find(needle) != std::string::npos,
              "NCCL context failure lacks field '" + needle + "': " +
                  error.what());
    }
    require(failed, "NCCL context mismatch escaped exchange failure mapping");
  };

  auto foreign_session = megamoe::make_collective_session(
      "foreign-nccl-context", 1, 95);
  auto foreign_context = megamoe::make_rank_collective_context(
      foreign_session, 0, execution.token());
  auto routes = megamoe::cuda::upload_cuda_routes(input, 3, execution);
  auto plan = megamoe::cuda::query_cuda_notify_workspace(config, 0, 1, 1);
  auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      plan, foreign_context, execution);
  auto notify = megamoe::cuda::start_cuda_rank_local_notify(
      config, foreign_context, execution, routes, std::move(workspace));
  auto foreign_local = notify.wait();
  bool foreign_control_failed = false;
  try {
    auto operation = backend.start_control(
        foreign_context,
        megamoe::cuda::make_control_record(config, foreign_context, 1, 3, 1));
    (void)operation.wait();
  } catch (const megamoe::NotifyDispatchError &error) {
    foreign_control_failed =
        error.code() == megamoe::NotifyErrorCode::NcclFailure &&
        std::string(error.what()).find("field=context") != std::string::npos;
  }
  require(foreign_control_failed,
          "foreign context was accepted by NCCL control protocol");

  auto host_context = megamoe::make_rank_collective_context(
      session.session(), 0, execution.token());
  auto host_local = megamoe::build_rank_local_histogram(
      config, host_context, input);

  auto proper_plan =
      megamoe::cuda::query_cuda_notify_workspace(config, 0, 1, 1);
  auto proper_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      proper_plan, host_context, execution);
  auto proper_notify = megamoe::cuda::start_cuda_rank_local_notify(
      config, host_context, execution, routes, std::move(proper_workspace));
  auto proper_local = proper_notify.wait();
  auto control = backend.start_control(
      host_context,
      megamoe::cuda::make_control_record(config, host_context, 1, 3, 1));
  (void)control.wait();
  expect_exchange_failure(host_local, "field=device_input");
  {
    auto cancelled =
        megamoe::start_count_exchange(proper_local.local(), backend);
    cancelled.cancel();
    cancelled.cancel();
  }
  auto completed = megamoe::start_count_exchange(proper_local.local(), backend);
  (void)completed.wait();
  {
    auto abandoned = megamoe::start_notify_summary_exchange(
        proper_local.local(), backend);
    (void)abandoned.state();
  }
  auto summary = megamoe::start_notify_summary_exchange(
      proper_local.local(), backend);
  (void)summary.wait();

  bool empty_identity_failed = false;
  bool zero_generation_failed = false;
  try {
    (void)megamoe::cuda::make_nccl_session("", 1, 0, 95, 0, bootstrap);
  } catch (const megamoe::NotifyDispatchError &error) {
    empty_identity_failed = error.code() == megamoe::NotifyErrorCode::NcclFailure;
  }
  try {
    (void)megamoe::cuda::make_nccl_session(
        "zero-generation", 1, 0, 0, 0, bootstrap);
  } catch (const megamoe::NotifyDispatchError &error) {
    zero_generation_failed =
        error.code() == megamoe::NotifyErrorCode::NcclFailure;
  }
  require(empty_identity_failed && zero_generation_failed,
          "NCCL session accepted empty identity or generation zero");
  std::cout << "[PASS] NCCL session/device/context and cancel/destructor boundaries\n";
}

void test_nccl_control_operation_lifecycle() {
  const auto config = make_config(1, 1, 8, 3, 4);
  const auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto execution = megamoe::cuda::make_cuda_execution(
      "control-lifecycle-stream", 0);
  auto session = megamoe::cuda::make_nccl_session(
      "control-lifecycle", 1, 0, 96, 0, bootstrap);
  auto context = megamoe::make_rank_collective_context(
      session.session(), 0, execution.token());
  megamoe::cuda::NcclAllGatherCountExchange backend(session, execution, 8, 1);
  const auto record = megamoe::cuda::make_control_record(
      config, context, 1, 3, 10);
  {
    auto first = backend.start_control(context, record);
    auto moved = std::move(first);
    moved = std::move(moved);
    moved.cancel();
    moved.cancel();
    require(moved.state() == megamoe::ExchangeState::Cancelled,
            "control cancel did not reach cancelled state");
  }
  {
    auto abandoned = backend.start_control(context, record);
    require(static_cast<bool>(abandoned), "control start returned empty operation");
  }
  auto completed = backend.start_control(context, record);
  auto handle = completed.wait();
  auto handle_copy = handle;
  require(handle_copy.matrix().size() == 1 &&
              handle_copy.context().calling_rank() == 0,
          "validated control handle copy lost completion state");
  bool repeated_wait_failed = false;
  try {
    (void)completed.wait();
  } catch (const megamoe::NotifyDispatchError &) {
    repeated_wait_failed = true;
  }
  require(repeated_wait_failed, "control operation accepted repeated wait");
  completed.cancel();
  const auto audit = backend.audit();
  require(audit.control_starts == 3 && audit.expert_count_starts == 0 &&
              audit.packet_summary_starts == 0,
          "control lifecycle audit count mismatch");
  std::cout << "[PASS] NCCL control move/cancel/destructor/repeated-wait lifecycle\n";
}

struct ProtocolCaseRankResult {
  std::string control_error;
  megamoe::cuda::CollectiveProtocolAudit failed_audit;
  megamoe::cuda::CollectiveProtocolAudit recovery_audit;
  std::exception_ptr unexpected;
};

void test_control_protocol_failure_and_recovery(const std::string &scenario) {
  constexpr std::uint32_t world = 2;
  constexpr std::uint64_t failed_generation = 610;
  constexpr std::uint64_t recovery_generation = 611;
  const std::string identity = "control-regression-" + scenario;
  const auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<ProtocolCaseRankResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            identity + "-stream-" + std::to_string(rank), rank);
        auto base_session = megamoe::cuda::make_nccl_session(
            identity, world, rank, failed_generation, rank, bootstrap);
        auto failed_config = make_config(1, world, 8, 3, 4);
        std::uint64_t max_tokens = 10;
        megamoe::RankRoutingInput input{1, {1, 2, 3}};
        bool can_launch_preflight = true;
        if (rank == 1 && scenario == "out_of_range")
          input = {1, {1, 8, 3}};
        else if (rank == 1 && scenario == "duplicate")
          input = {1, {1, 1, 3}};
        else if (rank == 1 && scenario == "max_mismatch")
          max_tokens = 11;
        else if (rank == 1 && scenario == "over_capacity") {
          input.num_tokens = 11;
          input.selected_experts.clear();
          for (std::uint64_t token = 0; token < input.num_tokens; ++token)
            input.selected_experts.insert(input.selected_experts.end(),
                                          {1, 2, 3});
          can_launch_preflight = false;
        } else if (rank == 1 && scenario == "experts_mismatch") {
          failed_config = make_config(1, world, 9, 3, 4);
        } else if (rank == 1 && scenario == "topk_mismatch") {
          failed_config = make_config(1, world, 8, 2, 4);
          input = {1, {1, 2}};
        } else if (rank == 1 && scenario == "invalid_topk") {
          failed_config = make_config(1, world, 2, 3, 4);
          can_launch_preflight = false;
        }

        auto failed_context = megamoe::make_rank_collective_context(
            base_session.session(), rank, execution.token());
        megamoe::cuda::NcclAllGatherCountExchange failed_backend(
            base_session, execution);
        megamoe::cuda::LocalPreflightStatus status;
        std::optional<megamoe::cuda::CudaNotifyPreflight> preflight;
        if (can_launch_preflight) {
          auto routes = megamoe::cuda::upload_cuda_routes(
              input, failed_config.top_k, execution);
          auto plan = megamoe::cuda::query_cuda_notify_workspace(
              failed_config, rank, input.num_tokens, max_tokens);
          auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
              plan, failed_context, execution);
          auto operation = megamoe::cuda::start_cuda_rank_local_notify(
              failed_config, failed_context, execution, routes,
              std::move(workspace));
          preflight.emplace(operation.wait_preflight());
          status = preflight->status();
        }
        const auto record = megamoe::cuda::make_control_record(
            failed_config, failed_context, input.num_tokens,
            input.selected_experts.size(), max_tokens, status);
        try {
          auto control = failed_backend.start_control(failed_context, record);
          (void)control.wait();
          throw std::runtime_error(
              "invalid distributed generation passed control agreement");
        } catch (const megamoe::NotifyDispatchError &error) {
          require(error.code() ==
                      megamoe::NotifyErrorCode::ControlProtocolFailure,
                  "invalid generation did not fail as control protocol error");
          results[rank].control_error = error.what();
        }
        results[rank].failed_audit = failed_backend.audit();

        // Reuse the exact native communicator through a new immutable
        // generation view. No bootstrap or communicator rebuild is allowed.
        auto recovery_session = megamoe::cuda::make_nccl_generation(
            base_session, identity, recovery_generation);
        auto recovery_context = megamoe::make_rank_collective_context(
            recovery_session.session(), rank, execution.token());
        megamoe::cuda::NcclAllGatherCountExchange recovery_backend(
            recovery_session, execution);
        const auto recovery_config = make_config(1, world, 8, 3, 4);
        const megamoe::RankRoutingInput recovery_input{1, {1, 2, 3}};
        auto recovery_routes = megamoe::cuda::upload_cuda_routes(
            recovery_input, recovery_config.top_k, execution);
        auto recovery_plan = megamoe::cuda::query_cuda_notify_workspace(
            recovery_config, rank, recovery_input.num_tokens, 10);
        auto recovery_workspace =
            megamoe::cuda::allocate_cuda_notify_workspace(
                recovery_plan, recovery_context, execution);
        auto recovery_operation =
            megamoe::cuda::start_cuda_rank_local_notify(
                recovery_config, recovery_context, execution,
                recovery_routes, std::move(recovery_workspace));
        auto recovery_preflight = recovery_operation.wait_preflight();
        auto recovery_control = recovery_backend.start_control(
            recovery_context,
            megamoe::cuda::make_control_record(
                recovery_config, recovery_context,
                recovery_input.num_tokens,
                recovery_input.selected_experts.size(), 10,
                recovery_preflight.status()));
        (void)recovery_control.wait();
        require(recovery_preflight.valid(), "valid recovery preflight failed");
        const auto &local = recovery_preflight.local();
        auto counts = megamoe::start_count_exchange(local.local(),
                                                     recovery_backend);
        auto counts_done = counts.wait();
        auto summary = megamoe::start_notify_summary_exchange(
            local.local(), recovery_backend);
        auto summary_done = summary.wait();
        (void)megamoe::materialize_rank_metadata(local.local(), counts_done,
                                                  summary_done);
        results[rank].recovery_audit = recovery_backend.audit();
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results) {
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  }
  require(!results[0].control_error.empty(), "rank 0 did not report control error");
  require(results[0].control_error == results[1].control_error,
          "ranks selected different control errors");
  for (const auto &result : results) {
    require(result.failed_audit.control_starts == 1 &&
                result.failed_audit.expert_count_starts == 0 &&
                result.failed_audit.packet_summary_starts == 0,
            "failed generation entered a variable-width collective");
    require(result.recovery_audit.control_starts == 1 &&
                result.recovery_audit.expert_count_starts == 1 &&
                result.recovery_audit.packet_summary_starts == 1,
            "recovery generation did not complete exact protocol sequence");
  }
  require(results[0].control_error.find("origin_rank=1") != std::string::npos,
          "control error did not identify rank 1 as origin");
  std::cout << "[PASS] control failure/recovery scenario=" << scenario
            << " error='" << results[0].control_error << "'\n";
}

void test_all_control_protocol_regressions() {
  for (const auto *scenario : {"out_of_range", "duplicate", "max_mismatch",
                               "over_capacity", "experts_mismatch",
                               "topk_mismatch", "invalid_topk"})
    test_control_protocol_failure_and_recovery(scenario);
}

void run_rank(std::uint32_t rank, std::uint32_t nodes,
              std::uint32_t gpus_per_node, std::uint32_t local_rank,
              std::uint64_t seed, const std::string &id_hex,
              const std::string &scenario) {
  const auto world = nodes * gpus_per_node;
  const auto experts = std::max<std::uint32_t>(16, world * 2);
  const auto top_k = std::min<std::uint32_t>(6, experts);
  auto config = make_config(nodes, gpus_per_node, experts, top_k, 7);
  auto local_routes =
      seeded_rank_input(world, experts, top_k, seed, rank);
  std::uint64_t max_tokens = 10;
  bool can_launch_preflight = true;
  if (rank == 1 && scenario == "out_of_range") {
    local_routes.num_tokens = 1;
    local_routes.selected_experts = {0, static_cast<std::int32_t>(experts)};
    for (std::uint32_t slot = 2; slot < top_k; ++slot)
      local_routes.selected_experts.push_back(
          static_cast<std::int32_t>(slot));
  } else if (rank == 1 && scenario == "duplicate") {
    local_routes.num_tokens = 1;
    local_routes.selected_experts = {0, 0};
    for (std::uint32_t slot = 2; slot < top_k; ++slot)
      local_routes.selected_experts.push_back(
          static_cast<std::int32_t>(slot));
  } else if (rank == 1 && scenario == "max_mismatch") {
    max_tokens = 11;
  } else if (rank == 1 && scenario == "over_capacity") {
    local_routes.num_tokens = 11;
    local_routes.selected_experts.clear();
    for (std::uint64_t token = 0; token < local_routes.num_tokens; ++token)
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        local_routes.selected_experts.push_back(
            static_cast<std::int32_t>(slot));
    can_launch_preflight = false;
  }
  megamoe::cuda::NcclBootstrapId bootstrap;
  if (id_hex == "GENERATE") {
    require(rank == 0, "only rank 0 may generate NCCL bootstrap ID");
    bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    std::cout << "BOOTSTRAP " << bootstrap.hex() << std::endl;
  } else {
    bootstrap = megamoe::cuda::parse_nccl_bootstrap_id(id_hex);
  }
  auto execution = megamoe::cuda::make_cuda_execution(
      "multinode-stream-" + std::to_string(rank), local_rank);
  auto session = megamoe::cuda::make_nccl_session(
      "cuda-multinode", world, rank, 701, local_rank, bootstrap);
  auto context = megamoe::make_rank_collective_context(
      session.session(), rank, execution.token());
  megamoe::cuda::NcclAllGatherCountExchange backend(session, execution);
  std::optional<megamoe::cuda::CudaNotifyPreflight> preflight;
  megamoe::cuda::LocalPreflightStatus preflight_status;
  if (can_launch_preflight) {
    auto routes = megamoe::cuda::upload_cuda_routes(local_routes, top_k,
                                                     execution);
    auto plan = megamoe::cuda::query_cuda_notify_workspace(
        config, rank, local_routes.num_tokens, max_tokens);
    auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        plan, context, execution);
    auto local_operation = megamoe::cuda::start_cuda_rank_local_notify(
        config, context, execution, routes, std::move(workspace));
    preflight.emplace(local_operation.wait_preflight());
    preflight_status = preflight->status();
  }
  if ((rank % 3U) != 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(rank % 3U));
  auto control = backend.start_control(
      context, megamoe::cuda::make_control_record(
                   config, context, local_routes.num_tokens,
                   local_routes.selected_experts.size(), max_tokens,
                   preflight_status));
  try {
    (void)control.wait();
  } catch (const megamoe::NotifyDispatchError &error) {
    const auto audit = backend.audit();
    require(audit.control_starts == 1 && audit.expert_count_starts == 0 &&
                audit.packet_summary_starts == 0,
            "rank-runner entered data collective after control failure");
    std::cout << "CONTROL_ERROR " << error.what() << std::endl;

    auto recovery_session = megamoe::cuda::make_nccl_generation(
        session, "cuda-multinode", 702);
    auto recovery_context = megamoe::make_rank_collective_context(
        recovery_session.session(), rank, execution.token());
    megamoe::cuda::NcclAllGatherCountExchange recovery_backend(
        recovery_session, execution);
    const auto recovery_config = make_config(nodes, gpus_per_node, experts,
                                             top_k, 7);
    megamoe::RankRoutingInput recovery_input;
    recovery_input.num_tokens = 1;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      recovery_input.selected_experts.push_back(
          static_cast<std::int32_t>(slot));
    auto recovery_routes = megamoe::cuda::upload_cuda_routes(
        recovery_input, top_k, execution);
    auto recovery_plan = megamoe::cuda::query_cuda_notify_workspace(
        recovery_config, rank, 1, 10);
    auto recovery_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        recovery_plan, recovery_context, execution);
    auto recovery_operation = megamoe::cuda::start_cuda_rank_local_notify(
        recovery_config, recovery_context, execution, recovery_routes,
        std::move(recovery_workspace));
    auto recovery_preflight = recovery_operation.wait_preflight();
    auto recovery_control = recovery_backend.start_control(
        recovery_context,
        megamoe::cuda::make_control_record(
            recovery_config, recovery_context, 1,
            recovery_input.selected_experts.size(), 10,
            recovery_preflight.status()));
    (void)recovery_control.wait();
    require(recovery_preflight.valid(), "rank-runner recovery preflight failed");
    const auto &recovery_local = recovery_preflight.local();
    auto recovery_counts = megamoe::start_count_exchange(
        recovery_local.local(), recovery_backend);
    auto recovery_counts_done = recovery_counts.wait();
    auto recovery_summary = megamoe::start_notify_summary_exchange(
        recovery_local.local(), recovery_backend);
    auto recovery_summary_done = recovery_summary.wait();
    (void)megamoe::materialize_rank_metadata(
        recovery_local.local(), recovery_counts_done, recovery_summary_done);
    const auto recovery_audit = recovery_backend.audit();
    require(recovery_audit.control_starts == 1 &&
                recovery_audit.expert_count_starts == 1 &&
                recovery_audit.packet_summary_starts == 1,
            "rank-runner recovery sequence incomplete");
    std::cout << "RECOVERY_OK rank=" << rank << std::endl;
    throw;
  }
  require(preflight && preflight->valid(),
          "control accepted invalid rank-runner preflight");
  const auto &local = preflight->local();
  auto count = megamoe::start_count_exchange(local.local(), backend);
  auto count_done = count.wait();
  auto summary = megamoe::start_notify_summary_exchange(local.local(), backend);
  auto summary_done = summary.wait();
  auto host = megamoe::materialize_rank_metadata(local.local(), count_done,
                                                  summary_done);
  auto device = megamoe::cuda::stage_cuda_dispatch_metadata(
      std::move(host), local, execution);
  megamoe::cuda::validate_cuda_dispatch_metadata(device, execution);
  std::cout << megamoe::serialize_dispatch_metadata(device.host()) << '\n';
}

void benchmark(std::uint32_t world) {
  const auto config = make_config(1, world, 384, 6, 64);
  std::vector<megamoe::RankRoutingInput> workload(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    workload[rank].num_tokens = 8192;
    workload[rank].selected_experts.clear();
    for (std::uint64_t token = 0; token < 8192; ++token) {
      const auto base = static_cast<std::uint32_t>(
          (token * 17U + rank * 31U) % 384U);
      for (std::uint32_t slot = 0; slot < 6; ++slot)
        workload[rank].selected_experts.push_back(
            static_cast<std::int32_t>((base + slot * 53U) % 384U));
    }
  }
  constexpr std::size_t warmup_iterations = 10;
  constexpr std::size_t measured_iterations = 50;
  std::vector<std::vector<megamoe::RankRoutingInput>> fixtures(
      warmup_iterations + measured_iterations, workload);
  const auto results = run_local_collective(config, fixtures, "cuda-bench",
                                             801, false, warmup_iterations,
                                             false);
  struct Statistics {
    double mean = 0;
    double p50 = 0;
    double p95 = 0;
    double minimum = 0;
    double maximum = 0;
  };
  const auto summarize = [](std::vector<double> values) {
    require(!values.empty(), "benchmark statistic has no values");
    std::sort(values.begin(), values.end());
    double sum = 0;
    for (const auto value : values)
      sum += value;
    const auto percentile = [&](double fraction) {
      const auto ordinal = static_cast<std::size_t>(
          std::ceil(fraction * static_cast<double>(values.size())));
      return values[std::max<std::size_t>(ordinal, 1) - 1];
    };
    return Statistics{sum / static_cast<double>(values.size()),
                      percentile(0.50), percentile(0.95), values.front(),
                      values.back()};
  };
  using SampleMember = double StageSample::*;
  const std::vector<std::pair<const char *, SampleMember>> stages{
      {"mapping_cuda_event_us", &StageSample::mapping_cuda_us},
      {"mapping_host_wall_us", &StageSample::mapping_wall_us},
      {"control_nccl_completion_us", &StageSample::control_completion_us},
      {"expert_count_nccl_completion_us", &StageSample::count_completion_us},
      {"packet_summary_nccl_completion_us", &StageSample::summary_completion_us},
      {"collective_completion_us", &StageSample::collective_completion_us},
      {"d2h_materialize_h2d_wall_us", &StageSample::materialize_wall_us},
      {"end_to_end_host_wall_us", &StageSample::end_to_end_wall_us}};
  std::vector<std::pair<std::string, Statistics>> slowest_statistics;
  for (const auto &[name, member] : stages) {
    std::vector<double> slowest;
    slowest.reserve(measured_iterations);
    for (std::size_t iteration = 0; iteration < measured_iterations;
         ++iteration) {
      double value = 0;
      for (const auto &rank : results) {
        require(rank.samples.size() == measured_iterations,
                "benchmark rank sample count mismatch");
        value = std::max(value, rank.samples[iteration].*member);
      }
      slowest.push_back(value);
    }
    slowest_statistics.emplace_back(name, summarize(std::move(slowest)));
  }

  WorkspaceSample workspace;
  for (const auto &rank : results) {
    workspace.route_input_device_bytes = std::max(
        workspace.route_input_device_bytes,
        rank.workspace.route_input_device_bytes);
    workspace.device_arena_bytes = std::max(
        workspace.device_arena_bytes, rank.workspace.device_arena_bytes);
    workspace.pinned_staging_bytes = std::max(
        workspace.pinned_staging_bytes, rank.workspace.pinned_staging_bytes);
    workspace.nccl_device_bytes = std::max(
        workspace.nccl_device_bytes, rank.workspace.nccl_device_bytes);
    workspace.nccl_pinned_bytes = std::max(
        workspace.nccl_pinned_bytes, rank.workspace.nccl_pinned_bytes);
    workspace.final_metadata_resident_arena_bytes = std::max(
        workspace.final_metadata_resident_arena_bytes,
        rank.workspace.final_metadata_resident_arena_bytes);
  }
  const auto unique_peak = workspace.route_input_device_bytes +
                           workspace.device_arena_bytes +
                           workspace.pinned_staging_bytes +
                           workspace.nccl_device_bytes +
                           workspace.nccl_pinned_bytes;
  std::cout << "BENCH world=" << world
            << " tokens_per_rank=8192 experts=384 top_k=6 warmup="
            << warmup_iterations << " measured=" << measured_iterations
            << " aggregation=slowest-rank-per-iteration\n";
  for (const auto &[name, stats] : slowest_statistics)
    std::cout << "  " << name << " mean=" << stats.mean
              << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  std::cout << "  workspace route_input_device="
            << workspace.route_input_device_bytes << " device_arena="
            << workspace.device_arena_bytes << " pinned_staging="
            << workspace.pinned_staging_bytes << " nccl_device="
            << workspace.nccl_device_bytes << " nccl_pinned="
            << workspace.nccl_pinned_bytes
            << " final_metadata_resident_in_arena="
            << workspace.final_metadata_resident_arena_bytes
            << " unique_peak_bytes=" << unique_peak << '\n';
  std::cout << "BENCH_JSON {\"schema\":1,\"world\":" << world
            << ",\"tokens_per_rank\":8192,\"experts\":384,\"top_k\":6"
            << ",\"warmup\":" << warmup_iterations
            << ",\"measured\":" << measured_iterations
            << ",\"aggregation\":\"slowest-rank-per-iteration\",\"stages\":{";
  for (std::size_t index = 0; index < slowest_statistics.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    const auto &[name, stats] = slowest_statistics[index];
    std::cout << '\"' << name << "\":{\"mean\":" << stats.mean
              << ",\"p50\":" << stats.p50 << ",\"p95\":" << stats.p95
              << ",\"min\":" << stats.minimum << ",\"max\":"
              << stats.maximum << '}';
  }
  std::cout << "},\"workspace_bytes\":{\"route_input_device\":"
            << workspace.route_input_device_bytes << ",\"device_arena\":"
            << workspace.device_arena_bytes << ",\"pinned_staging\":"
            << workspace.pinned_staging_bytes << ",\"nccl_device\":"
            << workspace.nccl_device_bytes << ",\"nccl_pinned\":"
            << workspace.nccl_pinned_bytes
            << ",\"final_metadata_resident_in_arena\":"
            << workspace.final_metadata_resident_arena_bytes
            << ",\"unique_peak\":" << unique_peak
            << "},\"hot_path\":{\"per_route_copy\":false,"
               "\"per_route_allocation\":false,\"per_route_sync\":false}}\n";
}

std::uint64_t parse_u64(const char *value) {
  return static_cast<std::uint64_t>(std::stoull(value));
}

void print_nccl_runtime_info() {
  const auto info = megamoe::cuda::nccl_runtime_info();
  std::cout << "NCCL_RUNTIME compiled=" << info.compiled_version
            << " runtime=" << info.runtime_version
            << " library=" << info.loaded_library_path
            << " NCCL_SOCKET_IFNAME=" << info.socket_ifname
            << " NCCL_IB_DISABLE=" << info.ib_disable << '\n';
  if (info.compiled_version != info.runtime_version)
    std::cout << "[WARN] NCCL header/runtime version mismatch: header="
              << info.compiled_version << " runtime=" << info.runtime_version
              << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--emit-id") {
      std::cout << megamoe::cuda::make_nccl_bootstrap_id().hex() << '\n';
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--local") {
      const auto world = static_cast<std::uint32_t>(parse_u64(argv[2]));
      cudaDeviceProp properties{};
      const auto property_status = cudaGetDeviceProperties(&properties, 0);
      require(property_status == cudaSuccess,
              std::string("cudaGetDeviceProperties failed: ") +
                  cudaGetErrorString(property_status));
      std::cout << "[INFO] CUDA runtime device0=" << properties.name
                << " cc=" << properties.major << '.' << properties.minor
                << '\n';
      print_nccl_runtime_info();
      test_invalid_inputs();
      test_device_masks_dedup_and_capacity();
      test_nccl_context_rejections();
      test_nccl_control_operation_lifecycle();
      test_local(world);
      std::cout << "[PASS] NCCL compiled version "
                << megamoe::cuda::compiled_nccl_version() << '\n';
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--bench") {
      benchmark(static_cast<std::uint32_t>(parse_u64(argv[2])));
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--protocol-regressions") {
      test_all_control_protocol_regressions();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--smoke") {
      test_invalid_inputs();
      test_smoke();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--device-smoke") {
      test_invalid_inputs();
      test_device_kernel_smoke();
      return 0;
    }
    if (argc >= 14 && std::string(argv[1]) == "--rank-runner") {
      std::uint32_t rank = 0, nodes = 0, gpus = 0, local = 0;
      std::uint64_t seed = 0;
      std::string id;
      std::string scenario = "valid";
      for (int i = 2; i < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--rank") rank = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--nodes") nodes = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--gpus-per-node") gpus = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--local-rank") local = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--seed") seed = parse_u64(argv[i + 1]);
        else if (key == "--id") id = argv[i + 1];
        else if (key == "--scenario") scenario = argv[i + 1];
      }
      print_nccl_runtime_info();
      run_rank(rank, nodes, gpus, local, seed, id, scenario);
      return 0;
    }
    throw std::runtime_error(
        "usage: test_cuda_notify --local N | --smoke | --device-smoke | --bench N | "
        "--protocol-regressions | --emit-id | --rank-runner ...");
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
