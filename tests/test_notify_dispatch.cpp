#include "metadata/testing_api.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <signal.h>
#include <unistd.h>

#include "cuda/nccl_protocol_internal.hpp"
#include "cuda/persistent_v3/backward/cluster_task.cuh"
#include "cuda/persistent_v3/forward/pipeline_state.cuh"
#include "cuda/static_v4/ring_layout.hpp"
#include "metadata/notify_corruption_hook.hpp"
#include "transport/activation_protocol_testing.hpp"
#include "weftmoe/api/errors.hpp"
#include "weftmoe/cuda/control_protocol.hpp"
#include "weftmoe/kernels/persistent_contract.hpp"
#include "weftmoe/metadata/serialize.hpp"
#include "weftmoe/scoreboard/workspace_plan.hpp"
#include "weftmoe/transport/count_exchange.hpp"

namespace {

using weftmoe::BackendCompletion;
using weftmoe::BackendCountExchangeOperation;
using weftmoe::CollectiveOperation;
using weftmoe::CountExchange;
using weftmoe::CountExchangeRequest;
using weftmoe::CountMatrix;
using weftmoe::DispatchHandle;
using weftmoe::ExchangeKind;
using weftmoe::ExchangeState;
using weftmoe::HostFakeCountExchange;
using weftmoe::LocalHistogramHandle;
using weftmoe::NotifyDispatchError;
using weftmoe::NotifyErrorCode;
using weftmoe::RankCollectiveContext;
using weftmoe::RankRoutingInput;
using weftmoe::RuntimeConfig;
using weftmoe::Topology;
using weftmoe::ValidatedCollectiveHandle;

struct BootstrapIoScript {
  std::vector<std::uint8_t> input;
  std::vector<std::uint8_t> output;
  std::size_t input_offset = 0;
  std::size_t read_chunk = std::numeric_limits<std::size_t>::max();
  std::size_t write_chunk = std::numeric_limits<std::size_t>::max();
  bool read_eintr = false;
  bool write_eintr = false;
  bool write_zero = false;
  bool write_broken_pipe = false;
  std::size_t close_count = 0;
  std::vector<int> closed_fds;
};

BootstrapIoScript *bootstrap_io_script = nullptr;

std::ptrdiff_t scripted_bootstrap_read(int, void *destination,
                                       std::size_t capacity) {
  if (bootstrap_io_script == nullptr)
    throw std::runtime_error("bootstrap read hook has no script");
  if (bootstrap_io_script->read_eintr) {
    bootstrap_io_script->read_eintr = false;
    errno = EINTR;
    return -1;
  }
  const auto remaining = bootstrap_io_script->input.size() -
                         bootstrap_io_script->input_offset;
  if (remaining == 0)
    return 0;
  const auto count = std::min(
      {capacity, remaining, bootstrap_io_script->read_chunk});
  std::memcpy(destination,
              bootstrap_io_script->input.data() +
                  bootstrap_io_script->input_offset,
              count);
  bootstrap_io_script->input_offset += count;
  return static_cast<std::ptrdiff_t>(count);
}
std::ptrdiff_t scripted_bootstrap_write(int, const void *source,
                                        std::size_t size) {
  if (bootstrap_io_script == nullptr)
    throw std::runtime_error("bootstrap write hook has no script");
  if (bootstrap_io_script->write_eintr) {
    bootstrap_io_script->write_eintr = false;
    errno = EINTR;
    return -1;
  }
  if (bootstrap_io_script->write_broken_pipe) {
    bootstrap_io_script->write_broken_pipe = false;
    errno = EPIPE;
    return -1;
  }
  if (bootstrap_io_script->write_zero) {
    bootstrap_io_script->write_zero = false;
    return 0;
  }
  const auto count = std::min(size, bootstrap_io_script->write_chunk);
  const auto *bytes = static_cast<const std::uint8_t *>(source);
  bootstrap_io_script->output.insert(bootstrap_io_script->output.end(), bytes,
                                     bytes + count);
  return static_cast<std::ptrdiff_t>(count);
}

int scripted_bootstrap_close(int fd) {
  if (bootstrap_io_script == nullptr)
    return -1;
  ++bootstrap_io_script->close_count;
  bootstrap_io_script->closed_fds.push_back(fd);
  return 0;
}

class BootstrapIoHookGuard {
public:
  explicit BootstrapIoHookGuard(BootstrapIoScript &script) {
    bootstrap_io_script = &script;
    weftmoe::cuda::testing::set_bootstrap_io_hooks(
        scripted_bootstrap_read, scripted_bootstrap_write,
        scripted_bootstrap_close);
  }
  BootstrapIoHookGuard(const BootstrapIoHookGuard &) = delete;
  BootstrapIoHookGuard &operator=(const BootstrapIoHookGuard &) = delete;
  ~BootstrapIoHookGuard() {
    weftmoe::cuda::testing::clear_bootstrap_io_hooks();
    bootstrap_io_script = nullptr;
  }
};

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      throw std::runtime_error(std::string("requirement failed: ") +           \
                               #condition + " at line " +                      \
                               std::to_string(__LINE__));                      \
    }                                                                          \
  } while (false)

static_assert(!std::is_aggregate_v<weftmoe::HostIntegerRow>);
static_assert(!std::is_constructible_v<weftmoe::HostIntegerRow, std::uintptr_t,
                                       std::shared_ptr<const void>>);
static_assert(std::is_same_v<
              decltype(std::declval<const LocalHistogramHandle &>().outgoing()),
              const weftmoe::SourceRouteMetadata &>);
static_assert(std::is_copy_constructible_v<LocalHistogramHandle>);
static_assert(std::is_copy_constructible_v<ValidatedCollectiveHandle>);
static_assert(!std::is_copy_constructible_v<CollectiveOperation>);
static_assert(std::is_move_constructible_v<CollectiveOperation>);
static_assert(
    !std::is_copy_constructible_v<weftmoe::cuda::NcclBootstrapInputChannel>);
static_assert(
    std::is_nothrow_move_constructible_v<
        weftmoe::cuda::NcclBootstrapInputChannel>);
static_assert(
    !std::is_copy_constructible_v<weftmoe::cuda::NcclBootstrapOutputChannel>);
static_assert(
    std::is_nothrow_move_constructible_v<
        weftmoe::cuda::NcclBootstrapOutputChannel>);

Topology homogeneous_topology(std::uint32_t nodes, std::uint32_t gpus,
                              std::uint32_t experts) {
  Topology topology;
  topology.num_nodes = nodes;
  topology.world_size = nodes * gpus;
  for (std::uint32_t node = 0; node < nodes; ++node) {
    for (std::uint32_t local = 0; local < gpus; ++local) {
      topology.rank_to_node.push_back(node);
      topology.rank_to_local_rank.push_back(local);
    }
  }
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    topology.expert_to_rank.push_back(expert % topology.world_size);
  }
  return topology;
}

RuntimeConfig make_config(std::uint32_t nodes, std::uint32_t gpus,
                          std::uint32_t experts, std::uint32_t top_k,
                          std::uint32_t tile) {
  RuntimeConfig config;
  config.num_experts = experts;
  config.top_k = top_k;
  config.m_tile_rows = tile;
  config.topology = homogeneous_topology(nodes, gpus, experts);
  return config;
}

RankCollectiveContext make_context(const RuntimeConfig &config,
                                   std::uint32_t rank,
                                   const std::string &identity = "test",
                                   std::uint64_t generation = 1,
                                   const std::string &execution = "exec") {
  auto session = weftmoe::make_collective_session(
      identity, config.topology.world_size, generation);
  return weftmoe::make_rank_collective_context(
      std::move(session), rank, weftmoe::make_execution_token(execution));
}

void expect_error(NotifyErrorCode code, const std::function<void()> &action,
                  const std::vector<std::string> &text = {}) {
  try {
    action();
  } catch (const NotifyDispatchError &error) {
    REQUIRE(error.code() == code);
    for (const auto &needle : text) {
      REQUIRE(std::string(error.what()).find(needle) != std::string::npos);
    }
    return;
  }
  throw std::runtime_error("expected NotifyDispatchError was not raised");
}

std::vector<RankRoutingInput> seeded_inputs(std::uint32_t world,
                                            std::uint32_t experts,
                                            std::uint32_t top_k,
                                            std::uint64_t seed) {
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };
  std::vector<RankRoutingInput> inputs(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    inputs[rank].num_tokens = next() % 6U;
    for (std::uint64_t token = 0; token < inputs[rank].num_tokens; ++token) {
      std::vector<std::uint32_t> pool(experts);
      for (std::uint32_t i = 0; i < experts; ++i)
        pool[i] = i;
      for (std::uint32_t i = 0; i < top_k; ++i) {
        const auto selected =
            i + static_cast<std::uint32_t>(next() % (experts - i));
        std::swap(pool[i], pool[selected]);
        inputs[rank].selected_experts.push_back(
            static_cast<std::int32_t>(pool[i]));
      }
    }
  }
  return inputs;
}

void verify_mapping(const std::vector<DispatchHandle> &handles) {
  std::vector<std::set<std::pair<std::uint32_t, std::uint64_t>>> seen(
      handles.size());
  for (std::size_t source = 0; source < handles.size(); ++source) {
    const auto &out = handles[source]->outgoing;
    REQUIRE(handles[source]->calling_rank == source);
    for (std::size_t route = 0; route < out.expert_id.size(); ++route) {
      REQUIRE(seen[out.route_dst_rank[route]]
                  .insert({out.expert_id[route], out.route_dst_row[route]})
                  .second);
      REQUIRE(out.reverse_routes[route].dst_row == out.route_dst_row[route]);
    }
  }
  for (std::size_t rank = 0; rank < handles.size(); ++rank) {
    const auto &incoming = handles[rank]->incoming;
    REQUIRE(incoming.expert_row_prefix.back() == incoming.incoming_rows.size());
    for (std::size_t row = 0; row < incoming.incoming_rows.size(); ++row) {
      REQUIRE(incoming.incoming_rows[row].dst_row == row);
    }
  }
}

enum class ThrowMode { None, Notify, Standard, Unknown };

[[noreturn]] void throw_mode(ThrowMode mode, const char *phase) {
  if (mode == ThrowMode::Notify) {
    throw NotifyDispatchError(NotifyErrorCode::InvalidHandle,
                              std::string("injected-") + phase + "-notify");
  }
  if (mode == ThrowMode::Standard) {
    throw std::runtime_error(std::string("injected-") + phase + "-std");
  }
  throw 17;
}

struct BackendBehavior {
  ThrowMode start = ThrowMode::None;
  ThrowMode wait = ThrowMode::None;
  ThrowMode cancel = ThrowMode::None;
  CountMatrix expert_rows;
  CountMatrix summary_rows;
  bool override_context = false;
  RankCollectiveContext returned_context;
  bool override_kind = false;
  ExchangeKind returned_kind = ExchangeKind::ExpertCounts;
  bool return_device = false;
  std::size_t wait_calls = 0;
  std::size_t cancel_calls = 0;
  std::size_t operation_destructs = 0;
  std::size_t completion_lease_destructs = 0;
  std::weak_ptr<const void> input_lifetime;
};

class TestDeviceLease final : public weftmoe::BackendDeviceLeaseState {};

class TestCompletionLease final : public weftmoe::BackendCompletionLeaseState {
public:
  explicit TestCompletionLease(std::shared_ptr<BackendBehavior> behavior)
      : behavior_(std::move(behavior)) {}
  ~TestCompletionLease() noexcept override {
    ++behavior_->completion_lease_destructs;
  }

private:
  std::shared_ptr<BackendBehavior> behavior_;
};

class ScriptedOperation final : public BackendCountExchangeOperation {
public:
  ScriptedOperation(std::shared_ptr<BackendBehavior> behavior,
                    CountExchangeRequest request)
      : behavior_(std::move(behavior)), request_(std::move(request)) {}

  ~ScriptedOperation() noexcept override { ++behavior_->operation_destructs; }

  ExchangeState state() const noexcept override {
    return ExchangeState::Complete;
  }

  BackendCompletion wait() override {
    ++behavior_->wait_calls;
    if (behavior_->wait != ThrowMode::None) {
      throw_mode(behavior_->wait, "wait");
    }
    auto context = behavior_->override_context ? behavior_->returned_context
                                               : request_.context;
    const auto kind =
        behavior_->override_kind ? behavior_->returned_kind : request_.kind;
    const auto &rows = request_.kind == ExchangeKind::ExpertCounts
                           ? behavior_->expert_rows
                           : behavior_->summary_rows;
    if (behavior_->return_device) {
      return weftmoe::make_device_backend_completion(
          std::move(context), kind,
          weftmoe::make_device_buffer_lease(
              static_cast<std::uint32_t>(rows.size()), request_.row_width,
              static_cast<std::uint64_t>(rows.size()) * request_.row_width,
              std::make_shared<TestDeviceLease>()),
          std::make_shared<TestCompletionLease>(behavior_));
    }
    return weftmoe::make_host_backend_completion(
        std::move(context), kind, weftmoe::make_host_count_matrix(rows),
        std::make_shared<TestCompletionLease>(behavior_));
  }

  void cancel() override {
    ++behavior_->cancel_calls;
    if (behavior_->cancel != ThrowMode::None) {
      throw_mode(behavior_->cancel, "cancel");
    }
  }

private:
  std::shared_ptr<BackendBehavior> behavior_;
  CountExchangeRequest request_;
};

class ScriptedBackend final : public CountExchange {
public:
  explicit ScriptedBackend(std::shared_ptr<BackendBehavior> behavior)
      : behavior_(std::move(behavior)) {}

  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &request) override {
    behavior_->input_lifetime =
        weftmoe::testing::HostIntegerRowTestAccess::lifetime(request.input);
    if (behavior_->start != ThrowMode::None) {
      throw_mode(behavior_->start, "start");
    }
    return std::make_unique<ScriptedOperation>(behavior_, request);
  }

private:
  std::shared_ptr<BackendBehavior> behavior_;
};

ValidatedCollectiveHandle complete_scripted(const LocalHistogramHandle &local,
                                            ExchangeKind kind,
                                            const CountMatrix &rows) {
  auto behavior = std::make_shared<BackendBehavior>();
  if (kind == ExchangeKind::ExpertCounts)
    behavior->expert_rows = rows;
  else
    behavior->summary_rows = rows;
  ScriptedBackend backend(behavior);
  auto operation = kind == ExchangeKind::ExpertCounts
                       ? weftmoe::start_count_exchange(local, backend)
                       : weftmoe::start_notify_summary_exchange(local, backend);
  return operation.wait();
}
void test_rank_local_async_and_deterministic_mapping() {
  const auto config = make_config(2, 2, 8, 3, 4);
  const auto inputs = seeded_inputs(4, 8, 3, 0x1234ULL);
  const auto session = weftmoe::make_collective_session("async", 4, 9);
  std::vector<LocalHistogramHandle> locals;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    auto context = weftmoe::make_rank_collective_context(
        session, rank,
        weftmoe::make_execution_token("exec-" + std::to_string(rank)));
    locals.push_back(
        weftmoe::build_rank_local_histogram(config, context, inputs[rank]));
  }
  HostFakeCountExchange backend(4, "async");
  std::vector<CollectiveOperation> count_ops;
  count_ops.reserve(4);
  count_ops.push_back(weftmoe::start_count_exchange(locals[0], backend));
  REQUIRE(count_ops[0].state() == ExchangeState::Pending);
  auto wait0 =
      std::async(std::launch::async, [&] { return count_ops[0].wait(); });
  REQUIRE(wait0.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
  for (std::uint32_t rank = 1; rank < 4; ++rank) {
    count_ops.push_back(weftmoe::start_count_exchange(locals[rank], backend));
  }
  std::vector<ValidatedCollectiveHandle> counts;
  counts.push_back(wait0.get());
  for (std::uint32_t rank = 1; rank < 4; ++rank) {
    counts.push_back(count_ops[rank].wait());
  }
  std::vector<CollectiveOperation> summary_ops;
  summary_ops.reserve(4);
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    summary_ops.push_back(
        weftmoe::start_notify_summary_exchange(locals[rank], backend));
  }
  std::vector<DispatchHandle> handles;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const auto summary = summary_ops[rank].wait();
    handles.push_back(weftmoe::materialize_rank_metadata(
        locals[rank], counts[rank], summary));
  }
  verify_mapping(handles);
}
void test_existing_shapes_descriptors_and_repeatability() {
  auto config = make_config(2, 2, 8, 2, 4);
  std::vector<RankRoutingInput> inputs(4);
  inputs[0] = {3, {0, 1, 0, 2, 0, 3}};
  inputs[1] = {2, {0, 4, 0, 5}};
  inputs[3] = {1, {0, 6}};
  const auto first =
      weftmoe::build_notify_dispatch_for_testing(config, inputs, "repeat", 7);
  const auto second =
      weftmoe::build_notify_dispatch_for_testing(config, inputs, "repeat", 7);
  for (std::size_t rank = 0; rank < first.size(); ++rank) {
    REQUIRE(weftmoe::serialize_dispatch_metadata(first[rank]) ==
            weftmoe::serialize_dispatch_metadata(second[rank]));
  }
  REQUIRE(first[0]->incoming.mtile_expected_rows[0] ==
          std::vector<std::uint32_t>({4, 2}));
  REQUIRE(first[2]->outgoing.num_tokens == 0);
  verify_mapping(first);

  auto packet_config = make_config(2, 2, 6, 3, 4);
  packet_config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3};
  std::vector<RankRoutingInput> packet_inputs(4);
  packet_inputs[0] = {2, {2, 4, 3, 0, 2, 5}};
  const auto packet_handles = weftmoe::build_notify_dispatch_for_testing(
      packet_config, packet_inputs, "packet", 3);
  const auto &transport = packet_handles[0]->transport;
  REQUIRE(transport.outgoing_packets.size() == 3);
  REQUIRE(transport.outgoing_packets[1].identity.destination_node == 1);
  REQUIRE(transport.outgoing_packets[1].fanout.size() == 3);
  REQUIRE(transport.outgoing_packets[1].proxy_rank == 2);
  REQUIRE(transport.outgoing_signals[1].expected_packets == 2);
  REQUIRE(
      packet_handles[2]->transport.incoming_proxy_signals[0].expected_signals ==
      2);

  auto many = make_config(73, 1, 73, 2, 8);
  std::vector<RankRoutingInput> many_inputs(73);
  many_inputs[0] = {1, {0, 72}};
  const auto many_handles =
      weftmoe::build_notify_dispatch_for_testing(many, many_inputs, "many", 11);
  REQUIRE(many_handles[0]->outgoing.node_mask_words == 2);
  REQUIRE(many_handles[0]->outgoing.destination_node_mask ==
          std::vector<std::uint64_t>({1, std::uint64_t{1} << 8U}));
  REQUIRE(many_handles[0]->transport.outgoing_signals.size() == 73);
}

void test_activation_transport_protocol_oracle() {
  auto config = make_config(2, 2, 8, 3, 4);
  config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3, 0, 1};
  std::vector<RankRoutingInput> inputs(4);
  inputs[0] = {3, {4, 1, 5, 0, 3, 6, 2, 7, 4}};
  const auto handles = weftmoe::build_notify_dispatch_for_testing(
      config, inputs, "transport-oracle", 77);
  weftmoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = 33;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 2;
  std::vector<weftmoe::transport::BFloat16> activations(3U * 33U);
  for (std::size_t i = 0; i < activations.size(); ++i)
    activations[i] = weftmoe::transport::float_to_bfloat16(
        static_cast<float>((i * 17U) % 101U) / 32.0F);
  const std::vector<float> gates{0.2F, 0.3F, 0.5F, 0.1F, 0.6F,
                                 0.3F, 0.7F, 0.2F, 0.1F};
  const auto canonical = weftmoe::transport::simulate_activation_transport(
      handles[0], activations, gates, transport_config, 1);
  std::vector<std::uint64_t> reversed(
      handles[0]->transport.outgoing_packets.size());
  std::iota(reversed.rbegin(), reversed.rend(), std::uint64_t{0});
  const auto reordered = weftmoe::transport::simulate_activation_transport(
      handles[0], activations, gates, transport_config, 1, reversed);
  REQUIRE(canonical.combined_tokens.size() == activations.size());
  REQUIRE(canonical.contribution_by_route == reordered.contribution_by_route);
  REQUIRE(canonical.combined_tokens == reordered.combined_tokens);
  REQUIRE(canonical.canonical_expert_rows == reordered.canonical_expert_rows);
  REQUIRE(canonical.audit.dispatch_packets ==
          handles[0]->transport.outgoing_packets.size());
  REQUIRE(canonical.audit.relayed_fanout == 9);
  REQUIRE(canonical.audit.synthetic_contributions == 9);
  const auto bad_sequence =
      weftmoe::testing::NotifyCorruptionHook::with_packet_sequence(
          handles[0], 0, 7);
  expect_error(NotifyErrorCode::InvalidHandle,
               [&] {
                 (void)weftmoe::transport::simulate_activation_transport(
                     bad_sequence, activations, gates, transport_config, 1,
                     reversed);
               },
               {"phase=dispatch-validation",
                "field=transport.packet.identity"});

  auto many = make_config(73, 1, 73, 2, 8);
  std::vector<RankRoutingInput> many_inputs(73);
  many_inputs[0] = {1, {0, 72}};
  const auto many_handles = weftmoe::build_notify_dispatch_for_testing(
      many, many_inputs, "transport-73", 78);
  transport_config.hidden_size = 7;
  std::vector<weftmoe::transport::BFloat16> many_activations(7);
  const auto many_result = weftmoe::transport::simulate_activation_transport(
      many_handles[0], many_activations, {0.25F, 0.75F}, transport_config, 2);
  REQUIRE(many_handles[0]->outgoing.node_mask_words == 2);
  REQUIRE(many_result.audit.dispatch_packets == 2);
  REQUIRE(many_result.audit.deduplicated_cross_node_payload_bytes == 14);
  REQUIRE(many_result.audit.raw_cross_node_payload_bytes == 14);

  // Descriptor/team logic is runtime-shaped and is not restricted to the
  // physically available two-node, 1/8-GPU acceptance topology.
  const auto verify_runtime_topology = [&](std::uint32_t nodes,
                                            std::uint32_t gpus) {
    const auto world = nodes * gpus;
    const auto top_k = std::min<std::uint32_t>(6U, world);
    auto runtime = make_config(nodes, gpus, world, top_k, 4);
    const auto source_rank = std::min<std::uint32_t>(2U, gpus - 1U);
    std::vector<RankRoutingInput> runtime_inputs(world);
    runtime_inputs[source_rank].num_tokens = 2;
    for (std::uint64_t token = 0; token < 2; ++token)
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        runtime_inputs[source_rank].selected_experts.push_back(
            static_cast<std::int32_t>((token * top_k + slot) % world));
    const auto runtime_handles = weftmoe::build_notify_dispatch_for_testing(
        runtime, runtime_inputs,
        "runtime-topology-" + std::to_string(nodes) + "x" +
            std::to_string(gpus),
        79 + nodes * 10U + gpus);
    const auto &descriptor = runtime_handles[source_rank]->transport;
    REQUIRE(descriptor.outgoing_signals.size() == nodes);
    REQUIRE(descriptor.outgoing_packets.size() <= 2ULL * nodes);
    const auto source_local = source_rank % gpus;
    for (const auto &packet : descriptor.outgoing_packets) {
      REQUIRE(packet.proxy_rank ==
              packet.identity.destination_node * gpus + source_local);
      REQUIRE(packet.channel == 0U);
      for (const auto &fanout : packet.fanout) {
        const auto owner = runtime.topology.expert_to_rank[fanout.expert_id];
        REQUIRE(fanout.destination_local_rank == owner % gpus);
      }
    }
    weftmoe::transport::ActivationTransportConfig runtime_transport;
    runtime_transport.hidden_size = 7;
    runtime_transport.channels = 4;
    runtime_transport.ring_slots_per_source_channel = 2;
    std::vector<weftmoe::transport::BFloat16> runtime_activations(14);
    std::vector<float> runtime_gates(2ULL * top_k,
                                     1.0F / static_cast<float>(top_k));
    const auto result = weftmoe::transport::simulate_activation_transport(
        runtime_handles[source_rank], runtime_activations, runtime_gates,
        runtime_transport, 3);
    REQUIRE(result.audit.dispatch_packets ==
            descriptor.outgoing_packets.size());
    REQUIRE(result.audit.relayed_fanout == 2ULL * top_k);
    REQUIRE(result.audit.peak_ring_slots <=
            runtime_transport.ring_slots_per_source_channel);
    verify_mapping(runtime_handles);
  };
  verify_runtime_topology(3, 1);
  verify_runtime_topology(3, 4);
  verify_runtime_topology(4, 2);

  auto invalid = transport_config;
  invalid.ring_slots_per_source_channel = 0;
  expect_error(NotifyErrorCode::TransportProtocolFailure,
               [&] {
                 (void)weftmoe::transport::simulate_activation_transport(
                     many_handles[0], many_activations, {0.25F, 0.75F},
                     invalid, 2);
               },
               {"phase=activation-transport",
                "field=ring_slots_per_source_channel"});
}

void test_topology_route_capacity_and_overflow() {
  auto bad = make_config(1, 6, 8, 2, 4);
  bad.topology.num_nodes = 4;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 weftmoe::build_rank_local_histogram(bad, make_context(bad, 0),
                                                     {});
               },
               {"heterogeneous"});
  auto sparse = make_config(2, 2, 8, 2, 4);
  sparse.topology.rank_to_local_rank[1] = 2;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 weftmoe::build_rank_local_histogram(
                     sparse, make_context(sparse, 0), {});
               },
               {"local coordinate"});
  auto duplicate = make_config(2, 2, 8, 2, 4);
  duplicate.topology.rank_to_node[2] = 0;
  duplicate.topology.rank_to_local_rank[2] = 0;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 weftmoe::build_rank_local_histogram(
                     duplicate, make_context(duplicate, 0), {});
               },
               {"duplicate topology coordinate", "node=0", "local_rank=0"});
  auto bad_node = make_config(2, 2, 8, 2, 4);
  bad_node.topology.rank_to_node[0] = 2;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 weftmoe::build_rank_local_histogram(
                     bad_node, make_context(bad_node, 0), {});
               },
               {"out-of-range node coordinate", "rank=0", "node=2"});
  auto topk = make_config(1, 1, 2, 3, 4);
  expect_error(NotifyErrorCode::InvalidRuntimeConfig,
               [&] {
                 weftmoe::build_rank_local_histogram(topk,
                                                     make_context(topk, 0), {});
               },
               {"top_k exceeds"});

  auto config = make_config(1, 2, 4, 2, 4);
  const auto context = make_context(config, 0);
  expect_error(NotifyErrorCode::DuplicateExpert,
               [&] {
                 weftmoe::build_rank_local_histogram(config, context,
                                                     {1, {2, 2}});
               },
               {"rank=0", "topk_slot=1"});
  expect_error(NotifyErrorCode::ExpertOutOfRange,
               [&] {
                 weftmoe::build_rank_local_histogram(config, context,
                                                     {1, {2, 4}});
               },
               {"actual=4"});
  expect_error(NotifyErrorCode::ArithmeticOverflow,
               [&] {
                 weftmoe::build_rank_local_histogram(
                     config, context,
                     {std::numeric_limits<std::uint64_t>::max(), {}});
               },
               {"route count"});
  config.capacity.expert_rows = {1, 100, 100, 100};
  const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {0, 2}}};
  expect_error(NotifyErrorCode::ExpertCapacityExceeded,
               [&] {
                 weftmoe::build_notify_dispatch_for_testing(config, inputs,
                                                            "capacity", 1);
               },
               {"expert=0"});
  config.capacity.expert_rows.clear();
  config.capacity.workspace_rows_per_rank = {2, 100};
  expect_error(NotifyErrorCode::WorkspaceCapacityExceeded,
               [&] {
                 weftmoe::build_notify_dispatch_for_testing(config, inputs,
                                                            "workspace-cap", 1);
               },
               {"destination_rank=0", "actual_rows=3", "capacity=2"});
}
void test_operation_move_and_terminal_state_machine() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = weftmoe::build_rank_local_histogram(
      config, make_context(config, 0, "state", 1), {1, {0}});

  auto success = std::make_shared<BackendBehavior>();
  success->expert_rows = {{1, 0}};
  ScriptedBackend success_backend(success);
  auto original = weftmoe::start_count_exchange(local, success_backend);
  CollectiveOperation moved(std::move(original));
  REQUIRE(!original);
  auto *self_alias = &moved;
  moved = std::move(*self_alias);
  REQUIRE(static_cast<bool>(moved));
  const auto validated = moved.wait();
  REQUIRE(validated.rows() == 1);
  REQUIRE(success->wait_calls == 1);
  REQUIRE(success->cancel_calls == 0);
  moved.cancel();
  moved.cancel();
  REQUIRE(success->cancel_calls == 0);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)moved.wait(); },
               {"phase=wait", "expected=pending", "actual=terminal"});
  REQUIRE(success->wait_calls == 1);

  auto cancelled = std::make_shared<BackendBehavior>();
  cancelled->expert_rows = {{1, 0}};
  ScriptedBackend cancelled_backend(cancelled);
  {
    auto operation = weftmoe::start_count_exchange(local, cancelled_backend);
    operation.cancel();
    operation.cancel();
    REQUIRE(cancelled->cancel_calls == 1);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)operation.wait(); },
                 {"phase=wait", "actual=terminal"});
    REQUIRE(cancelled->wait_calls == 0);
  }
  REQUIRE(cancelled->cancel_calls == 1);
  REQUIRE(cancelled->operation_destructs == 1);

  auto destination = std::make_shared<BackendBehavior>();
  destination->expert_rows = {{1, 0}};
  ScriptedBackend destination_backend(destination);
  auto source = std::make_shared<BackendBehavior>();
  source->expert_rows = {{1, 0}};
  ScriptedBackend source_backend(source);
  {
    auto destination_operation =
        weftmoe::start_count_exchange(local, destination_backend);
    auto source_operation =
        weftmoe::start_count_exchange(local, source_backend);
    destination_operation = std::move(source_operation);
    REQUIRE(destination->cancel_calls == 1);
    REQUIRE(!source_operation);
    destination_operation.cancel();
    REQUIRE(source->cancel_calls == 1);
  }
  REQUIRE(destination->cancel_calls == 1);
  REQUIRE(source->cancel_calls == 1);
  REQUIRE(destination->operation_destructs == 1);
  REQUIRE(source->operation_destructs == 1);
}
void test_typed_storage_lifetimes_and_unwaited_destruction() {
  struct SessionState final : weftmoe::BackendSessionState {};
  struct ExecutionState final : weftmoe::BackendExecutionState {};
  const auto config = make_config(1, 1, 2, 1, 4);
  auto session_state = std::make_shared<SessionState>();
  auto execution_state = std::make_shared<ExecutionState>();
  const std::weak_ptr<SessionState> session_lifetime = session_state;
  const std::weak_ptr<ExecutionState> execution_lifetime = execution_state;
  auto session =
      weftmoe::make_collective_session("lifetime", 1, 1, session_state);
  auto execution = weftmoe::make_execution_token("exec", execution_state);
  auto context = weftmoe::make_rank_collective_context(session, 0, execution);
  auto local = weftmoe::build_rank_local_histogram(config, context, {1, {0}});
  auto behavior = std::make_shared<BackendBehavior>();
  behavior->expert_rows = {{1, 0}};
  ScriptedBackend backend(behavior);
  {
    auto operation = weftmoe::start_count_exchange(local, backend);
    local = {};
    context = {};
    execution = {};
    session = {};
    session_state.reset();
    execution_state.reset();
    REQUIRE(!behavior->input_lifetime.expired());
    REQUIRE(!session_lifetime.expired());
    REQUIRE(!execution_lifetime.expired());
  }
  REQUIRE(behavior->cancel_calls == 1);
  REQUIRE(behavior->input_lifetime.expired());
  REQUIRE(session_lifetime.expired());
  REQUIRE(execution_lifetime.expired());

  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto throwing = std::make_shared<BackendBehavior>();
    throwing->expert_rows = {{1, 0}};
    throwing->cancel = mode;
    ScriptedBackend throwing_backend(throwing);
    const auto throwing_local = weftmoe::build_rank_local_histogram(
        config, make_context(config, 0, "destructor", 2), {1, {0}});
    {
      auto operation =
          weftmoe::start_count_exchange(throwing_local, throwing_backend);
      (void)operation;
    }
    REQUIRE(throwing->cancel_calls == 1);
    REQUIRE(throwing->operation_destructs == 1);
  }

  auto retained_session_state = std::make_shared<SessionState>();
  auto retained_execution_state = std::make_shared<ExecutionState>();
  const std::weak_ptr<SessionState> retained_session = retained_session_state;
  const std::weak_ptr<ExecutionState> retained_execution =
      retained_execution_state;
  auto retained_context = weftmoe::make_rank_collective_context(
      weftmoe::make_collective_session("validated-life", 1, 3,
                                       retained_session_state),
      0,
      weftmoe::make_execution_token("validated-exec",
                                    retained_execution_state));
  auto retained_local =
      weftmoe::build_rank_local_histogram(config, retained_context, {1, {0}});
  auto retained_behavior = std::make_shared<BackendBehavior>();
  retained_behavior->expert_rows = {{1, 0}};
  ScriptedBackend retained_backend(retained_behavior);
  auto retained_operation =
      weftmoe::start_count_exchange(retained_local, retained_backend);
  auto validated = retained_operation.wait();
  ValidatedCollectiveHandle validated_copy = validated;
  ValidatedCollectiveHandle validated_move = std::move(validated_copy);
  retained_session_state.reset();
  retained_execution_state.reset();
  retained_context = {};
  retained_local = {};
  retained_operation = {};
  REQUIRE(!retained_behavior->input_lifetime.expired());
  REQUIRE(!retained_session.expired());
  REQUIRE(!retained_execution.expired());
  REQUIRE(retained_behavior->wait_calls == 1);
  REQUIRE(retained_behavior->cancel_calls == 0);
  REQUIRE(retained_behavior->operation_destructs == 0);
  REQUIRE(retained_behavior->completion_lease_destructs == 0);
  REQUIRE(validated_move.rows() == 1);
  validated = {};
  REQUIRE(retained_behavior->operation_destructs == 0);
  REQUIRE(retained_behavior->completion_lease_destructs == 0);
  validated_move = {};
  REQUIRE(retained_behavior->input_lifetime.expired());
  REQUIRE(retained_session.expired());
  REQUIRE(retained_execution.expired());
  REQUIRE(retained_behavior->wait_calls == 1);
  REQUIRE(retained_behavior->cancel_calls == 0);
  REQUIRE(retained_behavior->operation_destructs == 1);
  REQUIRE(retained_behavior->completion_lease_destructs == 1);
}

void test_backend_exception_boundaries_and_device_rejection() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = weftmoe::build_rank_local_histogram(
      config, make_context(config, 0, "errors", 3), {1, {0}});
  LocalHistogramHandle empty_local;
  auto unused_behavior = std::make_shared<BackendBehavior>();
  ScriptedBackend unused_backend(unused_behavior);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)weftmoe::start_count_exchange(empty_local, unused_backend); },
      {"phase=start", "actual=empty/default"});
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->start = mode;
    ScriptedBackend backend(behavior);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)weftmoe::start_count_exchange(local, backend); },
                 {"phase=start", "kind=expert_counts", "rank=0", "errors",
                  "generation=3", "backend_error"});
  }
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->wait = mode;
    behavior->expert_rows = {{1, 0}};
    ScriptedBackend backend(behavior);
    auto operation = weftmoe::start_count_exchange(local, backend);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)operation.wait(); },
                 {"phase=wait", "backend_error"});
    REQUIRE(behavior->wait_calls == 1);
    REQUIRE(behavior->cancel_calls == 1);
    operation = {};
    REQUIRE(behavior->operation_destructs == 1);
    REQUIRE(behavior->cancel_calls == 1);
  }
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->cancel = mode;
    behavior->expert_rows = {{1, 0}};
    ScriptedBackend backend(behavior);
    auto operation = weftmoe::start_count_exchange(local, backend);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { operation.cancel(); },
                 {"phase=cancel", "backend_error"});
    REQUIRE(behavior->cancel_calls == 1);
    operation = {};
    REQUIRE(behavior->operation_destructs == 1);
    REQUIRE(behavior->cancel_calls == 1);
  }
  auto device_behavior = std::make_shared<BackendBehavior>();
  device_behavior->expert_rows = {{1, 0}};
  device_behavior->return_device = true;
  ScriptedBackend device_backend(device_behavior);
  auto device_op = weftmoe::start_count_exchange(local, device_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)device_op.wait(); },
               {"typed host matrix", "opaque device lease"});

  auto kind_behavior = std::make_shared<BackendBehavior>();
  kind_behavior->expert_rows = {{1, 0}};
  kind_behavior->override_kind = true;
  kind_behavior->returned_kind = ExchangeKind::NotifyPacketSummary;
  ScriptedBackend kind_backend(kind_behavior);
  auto kind_op = weftmoe::start_count_exchange(local, kind_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)kind_op.wait(); },
               {"collective kind mismatch", "expected=expert_counts",
                "actual=notify_packet_summary"});

  auto shape_behavior = std::make_shared<BackendBehavior>();
  shape_behavior->expert_rows = {};
  ScriptedBackend shape_backend(shape_behavior);
  auto shape_op = weftmoe::start_count_exchange(local, shape_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)shape_op.wait(); },
               {"field=matrix.rows", "expected=1", "actual=0"});

  const auto valid_device = weftmoe::make_device_buffer_lease(
      2, 3, 8, std::make_shared<TestDeviceLease>());
  REQUIRE(valid_device.rows() == 2);
  REQUIRE(valid_device.row_width() == 3);
  REQUIRE(valid_device.element_count() == 6);
  REQUIRE(valid_device.capacity_elements() == 8);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::make_device_buffer_lease(
                     0, 3, 3, std::make_shared<TestDeviceLease>());
               },
               {"phase=device-lease-validation", "field=rows", "expected=>0",
                "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::make_device_buffer_lease(
                     1, 0, 1, std::make_shared<TestDeviceLease>());
               },
               {"field=row_width", "expected=>0", "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::make_device_buffer_lease(
                     2, 4, 7, std::make_shared<TestDeviceLease>());
               },
               {"field=capacity_elements", "expected=>=8", "actual=7"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::make_device_buffer_lease(
                     1, 1, 0, std::make_shared<TestDeviceLease>());
               },
               {"field=capacity_elements", "expected=>0", "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::make_device_buffer_lease(
                     2, std::numeric_limits<std::uint64_t>::max(),
                     std::numeric_limits<std::uint64_t>::max(),
                     std::make_shared<TestDeviceLease>());
               },
               {"field=element_count.uint64", "actual=overflow"});
}
void test_context_kind_and_provenance_mismatch_messages() {
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)weftmoe::make_collective_session("", 1, 1); },
               {"identity", "actual=empty"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)weftmoe::make_collective_session("bad", 1, 0); },
               {"generation", "actual=0"});

  const auto fake_config = make_config(1, 2, 4, 1, 4);
  const auto fake_session_a =
      weftmoe::make_collective_session("fake-cross", 2, 1);
  const auto fake_session_b =
      weftmoe::make_collective_session("fake-cross", 2, 1);
  const auto fake_local_a = weftmoe::build_rank_local_histogram(
      fake_config,
      weftmoe::make_rank_collective_context(
          fake_session_a, 0, weftmoe::make_execution_token("fake-a")),
      {});
  const auto fake_local_b = weftmoe::build_rank_local_histogram(
      fake_config,
      weftmoe::make_rank_collective_context(
          fake_session_b, 1, weftmoe::make_execution_token("fake-b")),
      {});
  HostFakeCountExchange cross_backend(2, "fake-cross");
  auto first_fake_operation =
      weftmoe::start_count_exchange(fake_local_a, cross_backend);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)weftmoe::start_count_exchange(fake_local_b, cross_backend); },
      {"phase=start", "cross-session mix", "different session instance"});
  first_fake_operation.cancel();

  const auto config = make_config(1, 2, 4, 2, 4);
  const auto session = weftmoe::make_collective_session("base", 2, 4);
  const auto execution0 = weftmoe::make_execution_token("exec0");
  const auto context0 =
      weftmoe::make_rank_collective_context(session, 0, execution0);
  const auto local0 =
      weftmoe::build_rank_local_histogram(config, context0, {1, {0, 1}});
  const auto count0 = complete_scripted(local0, ExchangeKind::ExpertCounts,
                                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  const auto summary0 =
      complete_scripted(local0, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
  (void)weftmoe::materialize_rank_metadata(local0, count0, summary0);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)weftmoe::materialize_rank_metadata(local0, count0, count0); },
      {"field=collective.kind", "expected=notify_packet_summary",
       "actual=expert_counts"});

  const auto local_rank1 = weftmoe::build_rank_local_histogram(
      config,
      weftmoe::make_rank_collective_context(
          session, 1, weftmoe::make_execution_token("exec1")),
      {});
  const auto count_rank1 = complete_scripted(
      local_rank1, ExchangeKind::ExpertCounts, {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(local0, count_rank1,
                                                          summary0);
               },
               {"field=calling_rank", "expected=0", "actual=1"});

  const auto other_execution = weftmoe::build_rank_local_histogram(
      config,
      weftmoe::make_rank_collective_context(
          session, 0, weftmoe::make_execution_token("other-exec")),
      {1, {0, 1}});
  const auto other_exec_count =
      complete_scripted(other_execution, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(
                     local0, other_exec_count, summary0);
               },
               {"field=execution.provenance", "exec0", "other-exec"});

  const auto generation_context = make_context(config, 0, "base", 5, "exec0");
  const auto generation_local = weftmoe::build_rank_local_histogram(
      config, generation_context, {1, {0, 1}});
  const auto generation_count =
      complete_scripted(generation_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(
                     local0, generation_count, summary0);
               },
               {"field=session.generation", "expected=4", "actual=5"});

  const auto session_local = weftmoe::build_rank_local_histogram(
      config, make_context(config, 0, "other-session", 4, "exec0"),
      {1, {0, 1}});
  const auto session_count = complete_scripted(
      session_local, ExchangeKind::ExpertCounts, {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(local0, session_count,
                                                          summary0);
               },
               {"field=session.identity", "base", "other-session"});

  const auto same_fields_local =
      weftmoe::build_rank_local_histogram(config, context0, {1, {0, 1}});
  const auto same_fields_count =
      complete_scripted(same_fields_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(
                     local0, same_fields_count, summary0);
               },
               {"field=local.provenance"});

  const auto same_values_session =
      weftmoe::make_collective_session("base", 2, 4);
  const auto other_session_instance_local = weftmoe::build_rank_local_histogram(
      config,
      weftmoe::make_rank_collective_context(same_values_session, 0, execution0),
      {1, {0, 1}});
  const auto other_session_instance_count = complete_scripted(
      other_session_instance_local, ExchangeKind::ExpertCounts,
      {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(
                     local0, other_session_instance_count, summary0);
               },
               {"field=session.provenance", "different session instance"});

  const auto world3 = make_config(1, 3, 4, 2, 4);
  const auto world3_local = weftmoe::build_rank_local_histogram(
      world3, make_context(world3, 0, "base", 4, "exec0"), {1, {0, 1}});
  const auto world3_count =
      complete_scripted(world3_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(local0, world3_count,
                                                          summary0);
               },
               {"field=world_size", "expected=2", "actual=3"});
}
void test_malicious_completion_context_and_corruption_hooks() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = weftmoe::build_rank_local_histogram(
      config, make_context(config, 0, "malicious", 2), {1, {0}});
  auto behavior = std::make_shared<BackendBehavior>();
  behavior->expert_rows = {{1, 0}};
  behavior->override_context = true;
  behavior->returned_context = weftmoe::make_rank_collective_context(
      local.context().session(), 0, weftmoe::make_execution_token("other"));
  ScriptedBackend backend(behavior);
  auto operation = weftmoe::start_count_exchange(local, backend);
  expect_error(
      NotifyErrorCode::CountExchangeFailure, [&] { (void)operation.wait(); },
      {"execution token mismatch", "expected='exec'", "actual='other'"});

  const auto count =
      complete_scripted(local, ExchangeKind::ExpertCounts, {{1, 0}});
  const auto summary =
      complete_scripted(local, ExchangeKind::NotifyPacketSummary, {{1}});
  const auto bad_expert =
      weftmoe::testing::NotifyCorruptionHook::with_expert_id(local, 0, 99);
  expect_error(NotifyErrorCode::ExpertOutOfRange,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(bad_expert, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.expert_id", "rank=0",
                "expected=[0,2)", "actual=99"});
  const auto truncated =
      weftmoe::testing::NotifyCorruptionHook::with_truncated_expert_ids(local);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(truncated, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=expert_id", "rank=0",
                "expected=1", "actual=0"});

  const auto bad_offset =
      weftmoe::testing::NotifyCorruptionHook::with_local_offset(local, 0, 7);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(bad_offset, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.local_offset",
                "rank=0", "expected=0", "actual=7"});
  const auto bad_combine =
      weftmoe::testing::NotifyCorruptionHook::with_combine_order(local, 0, 9);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(bad_combine, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.combine_order",
                "rank=0", "actual=9"});
  const auto bad_mask =
      weftmoe::testing::NotifyCorruptionHook::with_node_mask(local, 0, 0);
  expect_error(
      NotifyErrorCode::InvalidRouteShape,
      [&] {
        (void)weftmoe::materialize_rank_metadata(bad_mask, count, summary);
      },
      {"phase=materialize-local", "field=outgoing.destination_node_mask",
       "rank=0", "expected=recomputed node mask", "actual=content mismatch"});
  const auto bad_local_count =
      weftmoe::testing::NotifyCorruptionHook::with_local_expert_count(local, 0,
                                                                      8);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(bad_local_count,
                                                          count, summary);
               },
               {"phase=materialize-local", "field=local_expert_count", "rank=0",
                "actual=content mismatch"});

  const auto bad_matrix_shape =
      weftmoe::testing::NotifyCorruptionHook::with_collective_matrix(count, {});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(
                     local, bad_matrix_shape, summary);
               },
               {"phase=materialize", "field=matrix.rows", "rank=0",
                "expected=1", "actual=0"});
  const auto bad_matrix_width =
      weftmoe::testing::NotifyCorruptionHook::with_collective_matrix(count,
                                                                     {{1}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(
                     local, bad_matrix_width, summary);
               },
               {"phase=materialize", "field=matrix.row_width", "rank=0",
                "expected=2", "actual=1"});
  const auto bad_matrix_row =
      weftmoe::testing::NotifyCorruptionHook::with_collective_matrix(count,
                                                                     {{0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)weftmoe::materialize_rank_metadata(local, bad_matrix_row,
                                                          summary);
               },
               {"phase=materialize", "field=expert_counts.calling_rank_row",
                "rank=0", "expected=local histogram count row",
                "actual=content mismatch"});

  const auto handle = weftmoe::materialize_rank_metadata(local, count, summary);
  const auto validate_corrupt = [&](const DispatchHandle &corrupt,
                                    const std::string &field) {
    expect_error(NotifyErrorCode::InvalidHandle,
                 [&] { (void)weftmoe::make_workspace_plan(corrupt); },
                 {"phase=dispatch-validation", "field=" + field, "rank=0",
                  "expected=", "actual="});
  };
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_expert_row_count(handle, 0,
                                                                    9),
      "incoming.expert_row_count");
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_expert_row_prefix(handle, 1,
                                                                     9),
      "incoming.expert_row_prefix");
  validate_corrupt(weftmoe::testing::NotifyCorruptionHook::
                       with_packet_identity_destination_node(handle, 0, 9),
                   "transport.packet.identity");
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_packet_proxy(handle, 0, 9),
      "transport.packet.proxy_rank");
  validate_corrupt(weftmoe::testing::NotifyCorruptionHook::with_fanout_expert(
                       handle, 0, 0, 9),
                   "transport.fanout.expert_id");
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_fanout_local_rank(handle, 0,
                                                                     0, 9),
      "transport.fanout.destination_local_rank");
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_fanout_row(handle, 0, 0, 9),
      "transport.fanout.dst_row");
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_fanout_slot(handle, 0, 0, 9),
      "transport.fanout.topk_slot");
  validate_corrupt(
      weftmoe::testing::NotifyCorruptionHook::with_mtile_row(handle, 0, 0, 9),
      "incoming.mtile_expected_rows");
}

void test_handle_copy_move_workspace_and_overflow() {
  const auto config = make_config(1, 2, 4, 2, 4);
  const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {2, 3}}};
  const auto handles = weftmoe::build_notify_dispatch_for_testing(
      config, inputs, "workspace", 5);
  DispatchHandle empty;
  expect_error(NotifyErrorCode::InvalidHandle, [&] { (void)empty.metadata(); });
  const auto plan = weftmoe::make_workspace_plan(handles[0]);
  weftmoe::validate_workspace_plan(handles[0], plan);
  (void)weftmoe::ForwardPersistentInputs::checked(handles[0], plan);
  weftmoe::WorkspacePlan empty_plan;
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.requirements(); }, {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.shape(); }, {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.identity(); }, {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.epoch(); }, {"default/invalid"});
  expect_error(
      NotifyErrorCode::WorkspaceMismatch,
      [&] { weftmoe::validate_workspace_plan(handles[0], empty_plan); },
      {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { weftmoe::validate_workspace_plan(handles[1], plan); },
               {"another DispatchHandle"});
  const auto other = weftmoe::build_notify_dispatch_for_testing(config, inputs,
                                                                "workspace", 5);
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { weftmoe::validate_workspace_plan(other[0], plan); },
               {"another DispatchHandle"});

  const auto local = weftmoe::build_rank_local_histogram(
      config, make_context(config, 0, "overflow", 1), {1, {0, 1}});
  const auto counts = complete_scripted(
      local, ExchangeKind::ExpertCounts,
      {{1, 1, 0, 0}, {std::numeric_limits<std::uint64_t>::max(), 0, 0, 0}});
  const auto summary =
      complete_scripted(local, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
  expect_error(
      NotifyErrorCode::ArithmeticOverflow,
      [&] { (void)weftmoe::materialize_rank_metadata(local, counts, summary); },
      {"source prefix"});
}
void test_fixed_control_protocol_agreement() {
  const auto config = make_config(1, 2, 8, 3, 4);
  const auto context0 = make_context(config, 0, "control", 17, "control-exec-0");
  const auto context1 = make_context(config, 1, "control", 17, "control-exec-1");
  const auto record0 = weftmoe::cuda::make_control_record(
      config, context0, 2, 6, 10);
  const auto record1 = weftmoe::cuda::make_control_record(
      config, context1, 3, 9, 10);
  weftmoe::cuda::validate_control_matrix({record0, record1});

  const auto expect_control_error = [&](const weftmoe::cuda::ControlMatrix &rows,
                                        const std::vector<std::string> &fields) {
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] { weftmoe::cuda::validate_control_matrix(rows); },
                 fields);
  };

  const auto max_mismatch = weftmoe::cuda::make_control_record(
      config, context1, 3, 9, 11);
  expect_control_error({record0, max_mismatch},
                       {"phase=control-agreement", "field=max_tokens_per_rank",
                        "origin_rank=1", "expected=10", "actual=11"});

  auto expert_config = make_config(1, 2, 9, 3, 4);
  const auto expert_mismatch = weftmoe::cuda::make_control_record(
      expert_config, context1, 3, 9, 10);
  expect_control_error({record0, expert_mismatch},
                       {"field=num_experts", "origin_rank=1"});

  weftmoe::cuda::LocalPreflightStatus invalid_route;
  invalid_route.error = weftmoe::cuda::ControlLocalError::ExpertOutOfRange;
  invalid_route.field = weftmoe::cuda::ControlField::LocalValidation;
  invalid_route.token = 4;
  invalid_route.slot = 2;
  invalid_route.expert = 99;
  invalid_route.expected = 8;
  invalid_route.actual = 99;
  const auto bad_route = weftmoe::cuda::make_control_record(
      config, context1, 5, 15, 10, invalid_route);
  expect_control_error({record0, bad_route},
                       {"field=local_validation", "origin_rank=1",
                        "token=4", "slot=2", "expert=99"});

  const auto over_capacity = weftmoe::cuda::make_control_record(
      config, context1, 11, 33, 10);
  expect_control_error({record0, over_capacity},
                       {"field=num_tokens", "origin_rank=1",
                        "expected=10", "actual=11"});

  auto generation_mismatch = record1;
  ++generation_mismatch.generation;
  expect_control_error({record0, generation_mismatch},
                       {"field=generation", "origin_rank=1",
                        "expected=17", "actual=18"});

  auto session_mismatch = record1;
  ++session_mismatch.session_id_word1;
  expect_control_error({record0, session_mismatch},
                       {"field=session_id_word1", "origin_rank=1"});

  // Validation has no calling-rank context: the same gathered matrix always
  // produces an identical structured error.
  std::string first_error;
  std::string second_error;
  try {
    weftmoe::cuda::validate_control_matrix({record0, generation_mismatch});
  } catch (const NotifyDispatchError &error) {
    first_error = error.what();
  }
  try {
    weftmoe::cuda::validate_control_matrix({record0, generation_mismatch});
  } catch (const NotifyDispatchError &error) {
    second_error = error.what();
  }
  REQUIRE(!first_error.empty() && first_error == second_error);

  const auto wire0 = weftmoe::cuda::encode_control_header(record0);
  REQUIRE(weftmoe::cuda::decode_control_header(wire0).generation == 17);
  REQUIRE(static_cast<unsigned>(wire0[0]) == 0x43U &&
          static_cast<unsigned>(wire0[7]) == 0x4dU);
  auto corrupt_wire = wire0;
  corrupt_wire[80] ^= std::byte{1};
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] { (void)weftmoe::cuda::decode_control_header(corrupt_wire); },
               {"field=checksum"});
  auto byteswapped = wire0;
  for (std::size_t word = 0; word < weftmoe::cuda::kControlSchemaWords;
       ++word)
    std::reverse(byteswapped.begin() + static_cast<std::ptrdiff_t>(word * 8),
                 byteswapped.begin() +
                     static_cast<std::ptrdiff_t>((word + 1) * 8));
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] { (void)weftmoe::cuda::decode_control_header(byteswapped); },
               {"phase=control-agreement"});

  const auto config0 = weftmoe::cuda::encode_canonical_config(
      config, 10, context0.session().stable_identity_bytes());
  const auto decoded_config =
      weftmoe::cuda::decode_canonical_config(config0);
  REQUIRE(decoded_config.rank_to_node == config.topology.rank_to_node);
  REQUIRE(decoded_config.expert_to_rank == config.topology.expert_to_rank);
  REQUIRE(decoded_config.stable_session_identity ==
          context0.session().stable_identity_bytes());
  REQUIRE(config0.size() >= weftmoe::cuda::kCanonicalConfigHeaderBytes);
  const std::array<std::byte, 16> config_golden{
      std::byte{0x52}, std::byte{0x49}, std::byte{0x57}, std::byte{0x47},
      std::byte{0x46}, std::byte{0x43}, std::byte{0x4d}, std::byte{0x4d},
      std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
      std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
  REQUIRE(std::equal(config_golden.begin(), config_golden.end(),
                     config0.begin()));
  {
    auto byteswapped_config = config0;
    for (std::size_t word = 0;
         word * 8U < weftmoe::cuda::kCanonicalConfigHeaderBytes; ++word)
      std::reverse(
          byteswapped_config.begin() +
              static_cast<std::ptrdiff_t>(word * 8U),
          byteswapped_config.begin() +
              static_cast<std::ptrdiff_t>((word + 1U) * 8U));
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   (void)weftmoe::cuda::decode_canonical_config(
                       byteswapped_config);
                 },
                 {"magic"});
  }

  // The envelope is self-validating even at world=1. Every corruption class
  // must fail before payload values can be used.
  auto world1_header = record0;
  world1_header.calling_rank = 0;
  world1_header.world_size = 1;
  world1_header.config_blob_bytes = config0.size();
  world1_header.config_digest = weftmoe::cuda::config_wire_digest(config0);
  weftmoe::cuda::validate_config_matrix({config0}, {world1_header});
  const auto corrupt_config = [&](std::size_t offset,
                                  const std::string &needle) {
    auto damaged = config0;
    damaged.at(offset) ^= std::byte{1};
    auto header = world1_header;
    header.config_digest = weftmoe::cuda::config_wire_digest(damaged);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   weftmoe::cuda::validate_config_matrix({damaged}, {header});
                 },
                 {needle});
  };
  corrupt_config(0, "magic");
  corrupt_config(8, "protocol_version");
  corrupt_config(16, "header_length");
  corrupt_config(24, "config_blob_bytes");
  corrupt_config(32, "reserved");
  corrupt_config(48, "checksum");
  corrupt_config(64, "checksum");
  {
    auto truncated = config0;
    truncated.pop_back();
    auto header = world1_header;
    header.config_blob_bytes = truncated.size();
    header.config_digest = weftmoe::cuda::config_wire_digest(truncated);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   weftmoe::cuda::validate_config_matrix({truncated}, {header});
                 },
                 {"truncated"});
  }
  {
    auto trailing = config0;
    trailing.push_back(std::byte{0});
    auto header = world1_header;
    header.config_blob_bytes = trailing.size();
    header.config_digest = weftmoe::cuda::config_wire_digest(trailing);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   weftmoe::cuda::validate_config_matrix({trailing}, {header});
                 },
                 {"trailing_bytes"});
  }
  {
    auto wrong_header_digest = world1_header;
    ++wrong_header_digest.config_digest;
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   weftmoe::cuda::validate_config_matrix(
                       {config0}, {wrong_header_digest});
                 },
                 {"field=config_digest", "control_header_vs_wire"});
  }
  auto config_changed = config;
  std::swap(config_changed.topology.expert_to_rank[0],
            config_changed.topology.expert_to_rank[1]);
  const auto config1 = weftmoe::cuda::encode_canonical_config(
      config_changed, 10, context1.session().stable_identity_bytes());
  auto same_digest0 = record0;
  auto same_digest1 = record1;
  same_digest0.config_blob_bytes = config0.size();
  same_digest1.config_blob_bytes = config1.size();
  same_digest0.config_digest = weftmoe::cuda::config_wire_digest(config0);
  same_digest1.config_digest = weftmoe::cuda::config_wire_digest(config1);
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 weftmoe::cuda::validate_config_matrix(
                     {config0, config1}, {same_digest0, same_digest1});
               },
               {"phase=config-agreement", "field=expert_to_rank",
                "origin_rank=1"});

  // A forged equal diagnostic digest cannot bypass per-rank integrity. Once
  // both diagnostic digests are correctly recomputed, exact decoded/byte
  // agreement still rejects the legitimate-but-different payload.
  auto forged_digest1 = same_digest1;
  forged_digest1.config_digest = same_digest0.config_digest;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 weftmoe::cuda::validate_config_matrix(
                     {config0, config1}, {same_digest0, forged_digest1});
               },
               {"field=config_digest", "origin_rank=1"});

  // Original stable identity bytes, not their two header summaries or display
  // names, are the final session agreement boundary.
  const std::vector<std::byte> raw_identity0{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x30}};
  const std::vector<std::byte> raw_identity1{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x31}};
  const auto raw_session0 = weftmoe::make_collective_session(
      "same-display", raw_identity0, 2, 91);
  const auto raw_session1 = weftmoe::make_collective_session(
      "same-display", raw_identity1, 2, 91);
  const auto raw_context0 = weftmoe::make_rank_collective_context(
      raw_session0, 0, weftmoe::make_execution_token("raw-exec-0"));
  const auto raw_context1 = weftmoe::make_rank_collective_context(
      raw_session1, 1, weftmoe::make_execution_token("raw-exec-1"));
  const auto raw_wire0 = weftmoe::cuda::encode_canonical_config(
      config, 10, raw_identity0);
  const auto raw_wire1 = weftmoe::cuda::encode_canonical_config(
      config, 10, raw_identity1);
  auto raw_header0 = weftmoe::cuda::make_control_record(
      config, raw_context0, 1, 3, 10);
  auto raw_header1 = weftmoe::cuda::make_control_record(
      config, raw_context1, 1, 3, 10);
  // Simulate an adversarial summary collision: fixed control passes, exact raw
  // identity comparison in config agreement still fails.
  raw_header1.session_id_word0 = raw_header0.session_id_word0;
  raw_header1.session_id_word1 = raw_header0.session_id_word1;
  weftmoe::cuda::validate_control_matrix({raw_header0, raw_header1});
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 weftmoe::cuda::validate_config_matrix(
                     {raw_wire0, raw_wire1}, {raw_header0, raw_header1});
               },
               {"field=session_identity", "origin_rank=1", "coordinate=2"});

  const auto config3 = make_config(1, 3, 9, 3, 4);
  const auto context30 = make_context(config3, 0, "common", 44, "e0");
  const auto context31 = make_context(config3, 1, "common", 44, "e1");
  const auto context32 = make_context(config3, 2, "common", 44, "e2");
  const auto row30 = weftmoe::cuda::make_control_record(
      config3, context30, 1, 3, 10);
  auto row31 = weftmoe::cuda::make_control_record(
      config3, context31, 1, 3, 10);
  auto row32 = weftmoe::cuda::make_control_record(
      config3, context32, 1, 3, 10);
  ++row31.session_id_word1;
  ++row32.generation;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 weftmoe::cuda::validate_control_matrix(
                     {row30, row31, row32});
               },
               {"field=session_id_word1", "origin_rank=1"});

  // For one rank with several differences, fixed field order selects
  // generation before session and config fields.
  ++row31.generation;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 weftmoe::cuda::validate_control_matrix(
                     {row30, row31, row32});
               },
               {"field=generation", "origin_rank=1"});
}
void test_bootstrap_exact_frame_io() {
  std::vector<std::uint8_t> frame(
      weftmoe::cuda::NcclBootstrapInputChannel::kWireBytes);
  for (std::size_t index = 0; index < frame.size(); ++index)
    frame[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);

  const auto read_valid = [&](std::size_t chunk, bool interrupted) {
    BootstrapIoScript script;
    script.input = frame;
    script.read_chunk = chunk;
    script.read_eintr = interrupted;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapInputChannel channel(7);
    auto result = channel.read();
    REQUIRE(static_cast<bool>(result));
    REQUIRE(script.input_offset == frame.size());
    REQUIRE(script.close_count == 1);
    REQUIRE(channel.state() ==
            weftmoe::cuda::BootstrapChannelState::Consumed);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"field=read-state", "expected=state=Open",
                  "actual=state=Consumed"});
    REQUIRE(script.close_count == 1);
    return result;
  };

  auto bootstrap = read_valid(frame.size(), false);
  (void)read_valid(1, false);
  (void)read_valid(17, true);
  (void)read_valid(127, false);

  for (const auto short_size : {std::size_t{0}, std::size_t{1},
                                std::size_t{127}}) {
    BootstrapIoScript script;
    script.input.assign(frame.begin(), frame.begin() + short_size);
    script.read_chunk = 17;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapInputChannel channel(7);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"phase=bootstrap-channel-read", "field=bytes",
                  "expected=128", "actual=" + std::to_string(short_size)});
    REQUIRE(channel.state() == weftmoe::cuda::BootstrapChannelState::Failed);
    REQUIRE(script.close_count == 1);
  }

  for (const auto oversized : {std::size_t{129}, std::size_t{257}}) {
    BootstrapIoScript script;
    script.input.resize(oversized, 0x5aU);
    script.read_chunk = 17;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapInputChannel channel(7);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"phase=bootstrap-channel-read", "field=frame-length",
                  "expected=128", "actual=>=129"});
    REQUIRE(script.input_offset == 129);
    REQUIRE(channel.state() == weftmoe::cuda::BootstrapChannelState::Failed);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"field=read-state", "actual=state=Failed"});
    REQUIRE(script.input_offset == 129);
    REQUIRE(script.close_count == 1);
  }

  for (const auto chunk : {std::size_t{1}, std::size_t{17},
                           std::size_t{127}}) {
    BootstrapIoScript script;
    script.write_chunk = chunk;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapOutputChannel channel(8);
    channel.write(bootstrap);
    REQUIRE(script.output == frame);
    REQUIRE(script.close_count == 1);
    REQUIRE(channel.state() ==
            weftmoe::cuda::BootstrapChannelState::Consumed);
  }

  {
    BootstrapIoScript script;
    script.write_chunk = 17;
    script.write_eintr = true;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapOutputChannel channel(8);
    channel.write(bootstrap);
    REQUIRE(script.output == frame);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    script.write_zero = true;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapOutputChannel channel(8);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { channel.write(bootstrap); },
                 {"phase=bootstrap-channel-write", "field=write-progress",
                  "expected=>0", "actual=0"});
    REQUIRE(channel.state() == weftmoe::cuda::BootstrapChannelState::Failed);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    script.write_broken_pipe = true;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapOutputChannel channel(8);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { channel.write(bootstrap); },
                 {"phase=bootstrap-channel-write", "field=write",
                  "actual=I/O failure"});
    REQUIRE(channel.state() == weftmoe::cuda::BootstrapChannelState::Failed);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    script.input = frame;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapInputChannel source(7);
    weftmoe::cuda::NcclBootstrapInputChannel moved(std::move(source));
    REQUIRE(source.state() == weftmoe::cuda::BootstrapChannelState::Closed);
    REQUIRE(source.native_fd_for_testing() == -1);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)source.read(); }, {"actual=state=Closed"});
    auto *self_alias = &moved;
    moved = std::move(*self_alias);
    (void)moved.read();
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    {
      weftmoe::cuda::NcclBootstrapOutputChannel channel(8);
      channel.cancel();
      channel.cancel();
      REQUIRE(channel.state() == weftmoe::cuda::BootstrapChannelState::Closed);
    }
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    {
      weftmoe::cuda::NcclBootstrapOutputChannel source(8);
      weftmoe::cuda::NcclBootstrapOutputChannel moved(std::move(source));
      REQUIRE(source.state() == weftmoe::cuda::BootstrapChannelState::Closed);
      REQUIRE(source.native_fd_for_testing() == -1);
      moved.write(bootstrap);
    }
    REQUIRE(script.output == frame);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    weftmoe::cuda::NcclBootstrapInputChannel source(7);
    weftmoe::cuda::NcclBootstrapInputChannel destination(9);
    destination = std::move(source);
    REQUIRE(source.state() == weftmoe::cuda::BootstrapChannelState::Closed);
    REQUIRE(destination.native_fd_for_testing() == 7);
    REQUIRE(script.close_count == 1);
    REQUIRE(script.closed_fds == std::vector<int>{9});
    destination.cancel();
    REQUIRE(script.close_count == 2);
    REQUIRE(script.closed_fds == (std::vector<int>{9, 7}));
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    { weftmoe::cuda::NcclBootstrapInputChannel abandoned(7); }
    REQUIRE(script.close_count == 1);
    REQUIRE(script.closed_fds == std::vector<int>{7});
  }
}

void write_pipe_frame(int fd, const std::vector<std::uint8_t> &bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto result = ::write(fd, bytes.data() + offset,
                                bytes.size() - offset);
    if (result < 0 && errno == EINTR)
      continue;
    if (result <= 0)
      throw std::runtime_error("pipe fixture write failed");
    offset += static_cast<std::size_t>(result);
  }
  if (::close(fd) != 0)
    throw std::runtime_error("pipe fixture close failed");
}

void test_bootstrap_real_pipe_one_shot() {
  std::vector<std::uint8_t> frame(
      weftmoe::cuda::NcclBootstrapInputChannel::kWireBytes);
  for (std::size_t index = 0; index < frame.size(); ++index)
    frame[index] = static_cast<std::uint8_t>((index * 19U + 5U) & 0xffU);

  for (const auto size : {std::size_t{0}, std::size_t{1}, std::size_t{127},
                          std::size_t{128}, std::size_t{129},
                          std::size_t{257}}) {
    int descriptors[2] = {-1, -1};
    REQUIRE(::pipe(descriptors) == 0);
    std::vector<std::uint8_t> bytes(size, 0x6dU);
    if (size == 128)
      bytes = frame;
    write_pipe_frame(descriptors[1], bytes);
    const int original_read_fd = descriptors[0];
    weftmoe::cuda::NcclBootstrapInputChannel channel(original_read_fd);
    bool first_rejected = false;
    if (size == 128) {
      auto id = channel.read();
      REQUIRE(static_cast<bool>(id));
      REQUIRE(channel.state() ==
              weftmoe::cuda::BootstrapChannelState::Consumed);
    } else {
      try {
        (void)channel.read();
      } catch (const NotifyDispatchError &) {
        first_rejected = true;
      }
      REQUIRE(first_rejected);
      REQUIRE(channel.state() ==
              weftmoe::cuda::BootstrapChannelState::Failed);
    }
    errno = 0;
    std::uint8_t probe = 0;
    REQUIRE(::read(original_read_fd, &probe, 1) == -1);
    REQUIRE(errno == EBADF);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"field=read-state", "expected=state=Open"});
    if (size == 257) {
      std::cout << "first_oversize_rejected = " << (first_rejected ? 1 : 0)
                << "\nremaining_128_accepted_as_next_frame = 0\n"
                << "channel_terminal = "
                << (channel.state() ==
                            weftmoe::cuda::BootstrapChannelState::Failed
                        ? 1
                        : 0)
                << '\n';
    }
  }

  // A reader rejects at byte 129 and closes immediately; it never drains an
  // attacker that keeps the write end open.
  struct sigaction ignore_action {};
  struct sigaction old_action {};
  ignore_action.sa_handler = SIG_IGN;
  REQUIRE(::sigemptyset(&ignore_action.sa_mask) == 0);
  REQUIRE(::sigaction(SIGPIPE, &ignore_action, &old_action) == 0);
  int streaming[2] = {-1, -1};
  REQUIRE(::pipe(streaming) == 0);
  std::atomic<bool> saw_epipe{false};
  std::thread writer([&] {
    std::array<std::uint8_t, 256> bytes{};
    while (true) {
      const auto result = ::write(streaming[1], bytes.data(), bytes.size());
      if (result < 0 && errno == EINTR)
        continue;
      if (result < 0 && errno == EPIPE) {
        saw_epipe = true;
        break;
      }
      if (result <= 0)
        break;
    }
    (void)::close(streaming[1]);
  });
  weftmoe::cuda::NcclBootstrapInputChannel streaming_channel(streaming[0]);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { (void)streaming_channel.read(); },
               {"field=frame-length", "actual=>=129"});
  writer.join();
  REQUIRE(saw_epipe.load());
  REQUIRE(::sigaction(SIGPIPE, &old_action, nullptr) == 0);

  // Obtain an opaque ID through the input channel, then prove that output is
  // exact, closes on completion, and cannot be reused.
  int source_pipe[2] = {-1, -1};
  REQUIRE(::pipe(source_pipe) == 0);
  write_pipe_frame(source_pipe[1], frame);
  weftmoe::cuda::NcclBootstrapInputChannel source(source_pipe[0]);
  auto id = source.read();
  int output_pipe[2] = {-1, -1};
  REQUIRE(::pipe(output_pipe) == 0);
  const int original_write_fd = output_pipe[1];
  weftmoe::cuda::NcclBootstrapOutputChannel output(original_write_fd);
  output.write(id);
  REQUIRE(output.state() == weftmoe::cuda::BootstrapChannelState::Consumed);
  errno = 0;
  REQUIRE(::write(original_write_fd, frame.data(), 1) == -1);
  REQUIRE(errno == EBADF);
  std::vector<std::uint8_t> received(frame.size());
  std::size_t offset = 0;
  while (offset < received.size()) {
    const auto count =
        ::read(output_pipe[0], received.data() + offset,
               received.size() - offset);
    REQUIRE(count > 0);
    offset += static_cast<std::size_t>(count);
  }
  std::uint8_t trailing = 0;
  REQUIRE(::read(output_pipe[0], &trailing, 1) == 0);
  REQUIRE(::close(output_pipe[0]) == 0);
  REQUIRE(received == frame);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { output.write(id); },
               {"field=write-state", "actual=state=Consumed"});

  // Once a terminal channel has relinquished its number, it cannot touch a
  // later object that the OS assigns that same descriptor number.
  int stale_pipe[2] = {-1, -1};
  REQUIRE(::pipe(stale_pipe) == 0);
  std::vector<std::uint8_t> oversized(129, 0xaaU);
  write_pipe_frame(stale_pipe[1], oversized);
  const int stale_number = stale_pipe[0];
  weftmoe::cuda::NcclBootstrapInputChannel stale(stale_number);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { (void)stale.read(); }, {"actual=>=129"});
  int replacement[2] = {-1, -1};
  REQUIRE(::pipe(replacement) == 0);
  if (replacement[0] != stale_number) {
    REQUIRE(::dup2(replacement[0], stale_number) == stale_number);
    REQUIRE(::close(replacement[0]) == 0);
    replacement[0] = stale_number;
  }
  const std::uint8_t marker = 0x4cU;
  REQUIRE(::write(replacement[1], &marker, 1) == 1);
  REQUIRE(::close(replacement[1]) == 0);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { (void)stale.read(); }, {"actual=state=Failed"});
  std::uint8_t observed = 0;
  REQUIRE(::read(replacement[0], &observed, 1) == 1);
  REQUIRE(observed == marker);
  REQUIRE(::close(replacement[0]) == 0);
}

void test_static_forward_window_packets() {
  namespace plan = weftmoe::cuda::static_forward;
  for (unsigned tokens : {1U, 64U}) {
    const plan::WindowShape roomy{16U, 8U, 128U, 4U};
    auto clipped = roomy;
    clipped.tokens = tokens;
    REQUIRE(plan::return_partition(roomy, tokens).rows ==
            plan::return_partition(clipped, tokens).rows);
  }
  // Include a partial source group and unequal/empty sources. This exercises
  // three nodes without assuming the available two-node GPU test topology.
  for (const unsigned world : {8U, 16U, 24U}) {
    constexpr unsigned local_world = 8U, owned = 4U, top_k = 4U;
    const plan::WindowShape shape{world, 16U, 8U, top_k, 8U};
    const auto experts = world * owned;
    const auto windows = shape.windows(42U);
    using Key = std::tuple<std::uint64_t, unsigned, unsigned>;
    using Token = std::pair<std::uint64_t, unsigned>;
    std::map<Key, std::vector<Token>> packets;
    std::vector<unsigned> counts(windows * experts, 0U);
    std::uint64_t route_count = 0U, fanout_count = 0U;
    for (unsigned source = 0; source < world; ++source) {
      const auto tokens = (source * 17U) % 43U;
      unsigned covered_tokens = 0U;
      for (std::uint64_t w = 0; w < windows; ++w)
        covered_tokens += shape.token_count(w, source, tokens);
      REQUIRE(covered_tokens == tokens);
      for (unsigned token = 0; token < tokens; ++token) {
        const auto w = shape.window(source, token);
        REQUIRE(w < windows);
        REQUIRE(source >= shape.source_begin(w));
        REQUIRE(source < shape.source_end(w));
        REQUIRE(token >= shape.token_begin(w));
        REQUIRE(token - shape.token_begin(w) <
                shape.token_count(w, source, tokens));
        std::array<plan::TokenRoute, top_k> routes{};
        std::map<unsigned, unsigned> expected;
        for (unsigned k = 0; k < top_k; ++k) {
          const auto expert = token % 3U == 0U
                                  ? experts - 1U
                                  : (source * 13U + token * 7U + k / 2U) % experts;
          routes[k] = {expert, !(k == 3U && token % 5U == 0U)};
          if (routes[k].valid) {
            ++expected[expert / (local_world * owned)];
            ++counts[w * experts + expert];
            ++route_count;
          }
        }
        plan::for_each_token_node(
            routes.data(), top_k, local_world * owned,
            [&](unsigned node, unsigned fanouts) {
              const auto found = expected.find(node);
              REQUIRE(found != expected.end());
              REQUIRE(found->second == fanouts);
              packets[{w, source, node}].emplace_back(token, fanouts);
              fanout_count += fanouts;
              expected.erase(found);
            });
        REQUIRE(expected.empty());
      }
    }
    REQUIRE(route_count == fanout_count);
    for (std::uint64_t w = 0; w < windows; ++w) {
      for (unsigned rank = 0; rank < world; ++rank) {
        std::uint64_t blocks = 0;
        for (unsigned e = rank * owned; e < (rank + 1U) * owned; ++e) {
          const auto rows = counts[w * experts + e];
          const auto n = plan::ceil_div(rows, shape.tile_m);
          blocks += n;
          unsigned covered = 0;
          for (std::uint64_t b = 0; b < n; ++b)
            covered += plan::block_rows(rows, b, shape.tile_m);
          REQUIRE(covered == rows);
        }
        REQUIRE(blocks <= shape.max_mblocks(owned));
      }
    }
    const plan::PacketPartition partition{3U, 2U};
    std::uint64_t packed_fanouts = 0;
    for (const auto &entry : packets) {
      const auto &tokens = entry.second;
      std::set<std::uint64_t> packed_tokens;
      std::uint64_t packet_count = 0;
      for (unsigned lane = 0; lane < partition.lanes; ++lane) {
        const auto n = partition.lane_packets(tokens.size(), lane);
        packet_count += n;
        for (std::uint64_t sequence = 0; sequence < n; ++sequence) {
          const auto begin = partition.token_begin(lane, sequence);
          const auto rows = partition.token_count(tokens.size(), lane, sequence);
          unsigned expected = 0;
          for (unsigned i = 0; i < rows; ++i) {
            REQUIRE(packed_tokens.insert(tokens[begin + i].first).second);
            expected += tokens[begin + i].second;
          }
          REQUIRE(rows && expected >= rows && expected <= rows * top_k);
          packed_fanouts += expected;
        }
      }
      REQUIRE(packet_count == partition.packets(tokens.size()));
      REQUIRE(packed_tokens.size() == tokens.size());
    }
    REQUIRE(packed_fanouts == route_count);
  }
}

void test_static_forward_slot_lifetimes() {
  namespace plan = weftmoe::cuda::static_forward;
  using weftmoe::cuda::communication::reached;
  std::array<plan::Retirement, 2> retirement{};
  std::array<std::uint64_t, 2> done{}, credit{}, local_done{}, local_target{};
  for (std::uint64_t batch = 0; batch < 17U; ++batch) {
    std::array<std::uint64_t, 2> target{};
    std::array<unsigned, 2> expected{};
    for (unsigned i = 0; i < 2U; ++i) {
      const plan::SlotSequence seq{batch * 2U + i, 2U};
      REQUIRE(seq.slot() == i && seq.generation() == batch);
      REQUIRE(seq.reusable(credit[i], local_done[i], local_target[i]));
      expected[i] = static_cast<unsigned>((batch * 7U + i * 3U) % 13U + 1U);
      target[i] = retirement[i].publish(expected[i]);
      ++local_target[i];
    }
    // Slot 1 finishes first; neither its ACK nor its local completion retires 0.
    for (const unsigned i : {1U, 0U}) {
      const plan::SlotSequence next{(batch + 1U) * 2U + i, 2U};
      REQUIRE(!next.reusable(credit[i], local_done[i], local_target[i]));
      done[i] += expected[i] - 1U;
      REQUIRE(!reached(done[i], target[i], 64U));
      ++done[i];
      REQUIRE(reached(done[i], target[i], 64U));
      credit[i] = batch + 1U;
      REQUIRE(!next.reusable(credit[i], local_done[i], local_target[i]));
      ++local_done[i];
      REQUIRE(next.reusable(credit[i], local_done[i], local_target[i]));
      if (i == 1U)
        REQUIRE(!reached(done[0], target[0], 64U));
    }
  }
  // Row-based C retirement is invariant to Return fragmentation.
  plan::Retirement c;
  REQUIRE(c.publish(128U) == 128U);
  std::uint64_t copied = 0;
  for (unsigned rows : {40U, 56U}) {
    copied += rows;
    REQUIRE(!reached(copied, c.target, 64U));
  }
  copied += 32U;
  REQUIRE(reached(copied, c.target, 64U));
  REQUIRE(c.publish(3U) == 131U); // Next-generation tail; no reset.
  std::array<plan::RingCapacity, 3> planes{{{6U, 4U, 8U},
                                          {6U, 0U, 8U},
                                          {6U, 3U, 8U}}};
  REQUIRE(planes[0].fits(3U) && !planes[1].fits(3U) && planes[2].fits(3U));
  // No partial reservation while S is full.
  for (const auto &plane : planes)
    REQUIRE(plane.head == 6U);
  planes[1].tail = 1U;
  for (auto &plane : planes) {
    REQUIRE(plane.fits(3U));
    plane.head += 3U;
  }
}

void test_static_forward_network_layout() {
  namespace plan = weftmoe::cuda::static_forward;
  constexpr unsigned rows = 32U, hidden = 2048U;
  const auto dispatch = plan::packet_bytes(rows, hidden);
  const auto returning = plan::packet_bytes(rows / 2U, hidden);
  REQUIRE(dispatch == rows * hidden * 2U);
  for (unsigned n = 1; n <= rows; ++n) {
    const auto compact = plan::packet_bytes(n, hidden);
    REQUIRE(compact % 128U == 0U);
    REQUIRE(compact <= dispatch);
    REQUIRE(compact == n * hidden * 2U);
  }
  for (const unsigned nodes : {1U, 2U, 3U, 70U}) {
    const plan::NetworkRings rings{nodes, 2U, 4U, dispatch, returning};
    for (unsigned self = 0; self < nodes; ++self) {
      std::set<std::uint64_t> channels;
      std::vector<std::pair<std::size_t, std::size_t>> ranges;
      for (unsigned peer = 0; peer < nodes; ++peer) {
        if (peer == self)
          continue;
        for (unsigned lane = 0; lane < rings.lanes; ++lane) {
          const auto channel = rings.channel(self, peer, lane);
          REQUIRE(channels.insert(channel).second);
          REQUIRE(rings.peer(self, channel) == peer);
          for (bool ret : {false, true}) {
            const auto state = rings.state(ret, channel);
            ranges.emplace_back(state, state + sizeof(plan::NetworkChannel));
          }
          for (unsigned slot = 0; slot < rings.depth; ++slot) {
            for (bool ret : {false, true}) {
              const auto control = rings.control(ret, channel, slot);
              ranges.emplace_back(control, control + sizeof(plan::NetworkSlot));
              for (bool rx : {false, true}) {
                if (ret && rx)
                  continue; // Return lands in the source contribution window.
                const auto payload = [&](unsigned sequence) {
                  return ret ? rings.return_payload(channel, sequence)
                             : rings.dispatch_payload(rx, channel, sequence);
                };
                const auto begin = payload(slot);
                const auto bytes = ret ? returning : dispatch;
                REQUIRE(begin % 128U == 0U);
                REQUIRE(begin == payload(slot + rings.depth));
                ranges.emplace_back(begin, begin + bytes);
              }
            }
          }
        }
      }
      REQUIRE(channels.size() == rings.channels());
      std::sort(ranges.begin(), ranges.end());
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        REQUIRE(ranges[i].second <= rings.bytes());
        if (i)
          REQUIRE(ranges[i - 1U].second <= ranges[i].first);
      }
    }
  }
}

void test_forward_multicast_task_pairs() {
  using weftmoe::cuda::persistent_v3::fc1_group_task;
  for (unsigned mblocks = 1; mblocks <= 8; ++mblocks) {
    for (unsigned ntiles : {2U, 6U, 8U, 12U}) {
      std::set<std::pair<unsigned, unsigned>> tasks;
      for (unsigned position = 0; position < mblocks * ntiles; position += 2U) {
        const auto a = fc1_group_task(position, mblocks, ntiles);
        const auto b = fc1_group_task(position + 1U, mblocks, ntiles);
        REQUIRE(a.member == b.member || a.ntile == b.ntile);
        REQUIRE(a.member < mblocks && b.member < mblocks);
        REQUIRE(a.ntile < ntiles && b.ntile < ntiles);
        REQUIRE(tasks.emplace(a.member, a.ntile).second);
        REQUIRE(tasks.emplace(b.member, b.ntile).second);
      }
      REQUIRE(tasks.size() == mblocks * ntiles);
    }
  }
}

void test_backward_multicast_task_pairs() {
  using weftmoe::cuda::persistent_v3::backward::cluster_tile;
  for (unsigned mtiles = 1; mtiles <= 21; ++mtiles) {
    for (unsigned ntiles = 1; ntiles <= 13; ++ntiles) {
      for (bool share_a : {false, true}) {
        std::set<std::pair<unsigned, unsigned>> tasks;
        unsigned tails = 0;
        for (unsigned pair = 0; pair < (mtiles * ntiles + 1U) / 2U; ++pair) {
          const auto a = cluster_tile(pair, 0, mtiles, ntiles, share_a);
          const auto b = cluster_tile(pair, 1, mtiles, ntiles, share_a);
          REQUIRE(a.valid && a.share_a == b.share_a);
          REQUIRE(a.share_a ? a.m == b.m : a.n == b.n);
          for (const auto tile : {a, b}) {
            REQUIRE(tile.m < mtiles && tile.n < ntiles);
            if (tile.valid)
              REQUIRE(tasks.emplace(tile.m, tile.n).second);
            else
              ++tails;
          }
        }
        REQUIRE(tasks.size() == mtiles * ntiles);
        REQUIRE(tails == (mtiles * ntiles) % 2U);
      }
    }
  }
}

} // namespace

int main() {
  const std::vector<std::pair<const char *, std::function<void()>>> tests{
      {"static_forward_window_packets", test_static_forward_window_packets},
      {"static_forward_slot_lifetimes", test_static_forward_slot_lifetimes},
      {"static_forward_network_layout", test_static_forward_network_layout},
      {"forward_multicast_task_pairs", test_forward_multicast_task_pairs},
      {"backward_multicast_task_pairs", test_backward_multicast_task_pairs},
      {"rank_local_async_and_deterministic_mapping",
       test_rank_local_async_and_deterministic_mapping},
      {"existing_shapes_descriptors_and_repeatability",
       test_existing_shapes_descriptors_and_repeatability},
      {"activation_transport_protocol_oracle",
       test_activation_transport_protocol_oracle},
      {"topology_route_capacity_and_overflow",
       test_topology_route_capacity_and_overflow},
      {"typed_storage_lifetimes_and_unwaited_destruction",
       test_typed_storage_lifetimes_and_unwaited_destruction},
      {"operation_move_and_terminal_state_machine",
       test_operation_move_and_terminal_state_machine},
      {"backend_exception_boundaries_and_device_rejection",
       test_backend_exception_boundaries_and_device_rejection},
      {"context_kind_and_provenance_mismatch_messages",
       test_context_kind_and_provenance_mismatch_messages},
      {"malicious_completion_context_and_corruption_hooks",
       test_malicious_completion_context_and_corruption_hooks},
      {"handle_copy_move_workspace_and_overflow",
       test_handle_copy_move_workspace_and_overflow},
      {"fixed_control_protocol_agreement",
       test_fixed_control_protocol_agreement},
      {"bootstrap_exact_frame_io", test_bootstrap_exact_frame_io},
      {"bootstrap_real_pipe_one_shot", test_bootstrap_real_pipe_one_shot},
  };
  std::size_t passed = 0;
  for (const auto &[name, test] : tests) {
    try {
      test();
      ++passed;
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
      return 1;
    } catch (...) {
      std::cerr << "[FAIL] " << name << ": unknown exception\n";
      return 1;
    }
  }
  std::cout << passed << "/" << tests.size() << " tests passed\n";
  return 0;
}
