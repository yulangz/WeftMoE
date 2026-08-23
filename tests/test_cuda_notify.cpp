#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
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

struct Timings {
  double mapping_us = 0;
  double collective_us = 0;
  double materialize_us = 0;
  double end_to_end_us = 0;
  std::uint64_t workspace_bytes = 0;
};

struct RankResult {
  std::vector<std::string> serialized;
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
    bool stagger_enqueue, std::size_t timing_skip = 0) {
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
        auto session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        auto context = megamoe::make_rank_collective_context(
            session.session(), rank, execution.token());
        megamoe::cuda::NcclAllGatherCountExchange backend(
            session, execution, config.num_experts,
            config.topology.num_nodes);
        for (std::size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
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
          const auto map_start = Clock::now();
          auto local_operation = megamoe::cuda::start_cuda_rank_local_notify(
              config, context, execution, routes, std::move(workspace));
          auto local = local_operation.wait();
          const auto map_end = Clock::now();
          if (stagger_enqueue && ((rank + fixture) % 3U) != 0)
            std::this_thread::sleep_for(std::chrono::microseconds(
                20U * ((rank + static_cast<std::uint32_t>(fixture)) % 3U)));
          const auto collective_start = Clock::now();
          auto count = megamoe::start_count_exchange(local.local(), backend);
          auto summary =
              megamoe::start_notify_summary_exchange(local.local(), backend);
          auto count_done = count.wait();
          auto summary_done = summary.wait();
          const auto collective_end = Clock::now();
          const auto materialize_start = Clock::now();
          auto host = megamoe::materialize_rank_metadata(
              local.local(), count_done, summary_done);
          auto device_handle = megamoe::cuda::stage_cuda_dispatch_metadata(
              std::move(host), local, execution);
          const auto materialize_end = Clock::now();
          if (stagger_enqueue)
            megamoe::cuda::validate_cuda_dispatch_metadata(device_handle,
                                                            execution);
          require(static_cast<bool>(device_handle.device().expert_id) ||
                      fixtures[fixture][rank].num_tokens == 0,
                  "device route metadata lease is empty");
          results[rank].serialized.push_back(
              megamoe::serialize_dispatch_metadata(device_handle.host()));
          if (fixture >= timing_skip) {
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
  auto capacity_local = capacity_notify.wait();
  auto counts = megamoe::start_count_exchange(capacity_local.local(), backend);
  auto summary = megamoe::start_notify_summary_exchange(capacity_local.local(),
                                                         backend);
  auto counts_done = counts.wait();
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
  expect_exchange_failure(foreign_local.local(), "field=context");

  auto host_context = megamoe::make_rank_collective_context(
      session.session(), 0, execution.token());
  auto host_local = megamoe::build_rank_local_histogram(
      config, host_context, input);
  expect_exchange_failure(host_local, "field=device_input");

  auto proper_plan =
      megamoe::cuda::query_cuda_notify_workspace(config, 0, 1, 1);
  auto proper_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      proper_plan, host_context, execution);
  auto proper_notify = megamoe::cuda::start_cuda_rank_local_notify(
      config, host_context, execution, routes, std::move(proper_workspace));
  auto proper_local = proper_notify.wait();
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

void run_rank(std::uint32_t rank, std::uint32_t nodes,
              std::uint32_t gpus_per_node, std::uint32_t local_rank,
              std::uint64_t seed, const std::string &id_hex) {
  const auto world = nodes * gpus_per_node;
  const auto experts = std::max<std::uint32_t>(16, world * 2);
  const auto top_k = std::min<std::uint32_t>(6, experts);
  const auto config = make_config(nodes, gpus_per_node, experts, top_k, 7);
  auto all_routes = seeded_inputs(world, experts, top_k, seed);
  auto local_routes = std::move(all_routes.at(rank));
  all_routes.clear();
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
  megamoe::cuda::NcclAllGatherCountExchange backend(
      session, execution, config.num_experts, config.topology.num_nodes);
  auto routes = megamoe::cuda::upload_cuda_routes(local_routes, top_k,
                                                   execution);
  auto plan = megamoe::cuda::query_cuda_notify_workspace(
      config, rank, local_routes.num_tokens, 10);
  auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      plan, context, execution);
  auto local_operation = megamoe::cuda::start_cuda_rank_local_notify(
      config, context, execution, routes, std::move(workspace));
  auto local = local_operation.wait();
  if ((rank % 3U) != 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(rank % 3U));
  auto count = megamoe::start_count_exchange(local.local(), backend);
  auto summary = megamoe::start_notify_summary_exchange(local.local(), backend);
  auto count_done = count.wait();
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
  std::vector<std::vector<megamoe::RankRoutingInput>> fixtures(6, workload);
  const auto results = run_local_collective(config, fixtures, "cuda-bench",
                                             801, false, 1);
  Timings maximum;
  for (const auto &result : results) {
    maximum.mapping_us = std::max(maximum.mapping_us, result.timings.mapping_us);
    maximum.collective_us =
        std::max(maximum.collective_us, result.timings.collective_us);
    maximum.materialize_us =
        std::max(maximum.materialize_us, result.timings.materialize_us);
    maximum.end_to_end_us =
        std::max(maximum.end_to_end_us, result.timings.end_to_end_us);
    maximum.workspace_bytes =
        std::max(maximum.workspace_bytes, result.timings.workspace_bytes);
  }
  constexpr double measured_iterations = 5.0;
  std::cout << "BENCH world=" << world << " tokens_per_rank=8192 experts=384 top_k=6"
            << " mapping_us=" << maximum.mapping_us / measured_iterations
            << " collective_us=" << maximum.collective_us / measured_iterations
            << " d2h_materialize_h2d_us=" << maximum.materialize_us / measured_iterations
            << " end_to_end_us=" << maximum.end_to_end_us / measured_iterations
            << " workspace_peak_bytes=" << maximum.workspace_bytes << '\n';
}

std::uint64_t parse_u64(const char *value) {
  return static_cast<std::uint64_t>(std::stoull(value));
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
      test_invalid_inputs();
      test_device_masks_dedup_and_capacity();
      test_nccl_context_rejections();
      test_local(world);
      std::cout << "[PASS] NCCL compiled version "
                << megamoe::cuda::compiled_nccl_version() << '\n';
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--bench") {
      benchmark(static_cast<std::uint32_t>(parse_u64(argv[2])));
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
    if (argc == 14 && std::string(argv[1]) == "--rank-runner") {
      std::uint32_t rank = 0, nodes = 0, gpus = 0, local = 0;
      std::uint64_t seed = 0;
      std::string id;
      for (int i = 2; i < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--rank") rank = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--nodes") nodes = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--gpus-per-node") gpus = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--local-rank") local = static_cast<std::uint32_t>(parse_u64(argv[i + 1]));
        else if (key == "--seed") seed = parse_u64(argv[i + 1]);
        else if (key == "--id") id = argv[i + 1];
      }
      run_rank(rank, nodes, gpus, local, seed, id);
      return 0;
    }
    throw std::runtime_error(
        "usage: test_cuda_notify --local N | --smoke | --device-smoke | --bench N | "
        "--emit-id | --rank-runner ...");
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
