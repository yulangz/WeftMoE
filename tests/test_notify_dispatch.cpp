#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "megamoe/api/errors.hpp"
#include "megamoe/api/notify_dispatch.hpp"
#include "megamoe/cuda/control_protocol.hpp"
#include "megamoe/kernels/persistent_contract.hpp"
#include "megamoe/metadata/serialize.hpp"
#include "megamoe/scoreboard/workspace_plan.hpp"
#include "megamoe/transport/count_exchange.hpp"
#include "metadata/notify_corruption_hook.hpp"

namespace {

using megamoe::BackendCompletion;
using megamoe::BackendCountExchangeOperation;
using megamoe::CollectiveOperation;
using megamoe::CountExchange;
using megamoe::CountExchangeRequest;
using megamoe::CountMatrix;
using megamoe::DispatchHandle;
using megamoe::ExchangeKind;
using megamoe::ExchangeState;
using megamoe::HostFakeCountExchange;
using megamoe::LocalHistogramHandle;
using megamoe::NotifyDispatchError;
using megamoe::NotifyErrorCode;
using megamoe::RankCollectiveContext;
using megamoe::RankRoutingInput;
using megamoe::RuntimeConfig;
using megamoe::Topology;
using megamoe::ValidatedCollectiveHandle;

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      throw std::runtime_error(std::string("requirement failed: ") +           \
                               #condition + " at line " +                      \
                               std::to_string(__LINE__));                      \
    }                                                                          \
  } while (false)

static_assert(!std::is_aggregate_v<megamoe::HostIntegerRow>);
static_assert(!std::is_constructible_v<megamoe::HostIntegerRow, std::uintptr_t,
                                       std::shared_ptr<const void>>);
static_assert(std::is_same_v<
              decltype(std::declval<const LocalHistogramHandle &>().outgoing()),
              const megamoe::SourceRouteMetadata &>);
static_assert(std::is_copy_constructible_v<LocalHistogramHandle>);
static_assert(std::is_copy_constructible_v<ValidatedCollectiveHandle>);
static_assert(!std::is_copy_constructible_v<CollectiveOperation>);
static_assert(std::is_move_constructible_v<CollectiveOperation>);

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
  auto session = megamoe::make_collective_session(
      identity, config.topology.world_size, generation);
  return megamoe::make_rank_collective_context(
      std::move(session), rank, megamoe::make_execution_token(execution));
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

class TestDeviceLease final : public megamoe::BackendDeviceLeaseState {};

class TestCompletionLease final : public megamoe::BackendCompletionLeaseState {
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
      return megamoe::make_device_backend_completion(
          std::move(context), kind,
          megamoe::make_device_buffer_lease(
              static_cast<std::uint32_t>(rows.size()), request_.row_width,
              static_cast<std::uint64_t>(rows.size()) * request_.row_width,
              std::make_shared<TestDeviceLease>()),
          std::make_shared<TestCompletionLease>(behavior_));
    }
    return megamoe::make_host_backend_completion(
        std::move(context), kind, megamoe::make_host_count_matrix(rows),
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
    behavior_->input_lifetime = request.input.lifetime_for_testing();
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
                       ? megamoe::start_count_exchange(local, backend)
                       : megamoe::start_notify_summary_exchange(local, backend);
  return operation.wait();
}

void test_rank_local_async_and_deterministic_mapping() {
  const auto config = make_config(2, 2, 8, 3, 4);
  const auto inputs = seeded_inputs(4, 8, 3, 0x1234ULL);
  const auto session = megamoe::make_collective_session("async", 4, 9);
  std::vector<LocalHistogramHandle> locals;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    auto context = megamoe::make_rank_collective_context(
        session, rank,
        megamoe::make_execution_token("exec-" + std::to_string(rank)));
    locals.push_back(
        megamoe::build_rank_local_histogram(config, context, inputs[rank]));
  }
  HostFakeCountExchange backend(4, "async");
  std::vector<CollectiveOperation> count_ops;
  count_ops.reserve(4);
  count_ops.push_back(megamoe::start_count_exchange(locals[0], backend));
  REQUIRE(count_ops[0].state() == ExchangeState::Pending);
  auto wait0 =
      std::async(std::launch::async, [&] { return count_ops[0].wait(); });
  REQUIRE(wait0.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
  for (std::uint32_t rank = 1; rank < 4; ++rank) {
    count_ops.push_back(megamoe::start_count_exchange(locals[rank], backend));
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
        megamoe::start_notify_summary_exchange(locals[rank], backend));
  }
  std::vector<DispatchHandle> handles;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const auto summary = summary_ops[rank].wait();
    handles.push_back(megamoe::materialize_rank_metadata(
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
      megamoe::build_notify_dispatch_for_testing(config, inputs, "repeat", 7);
  const auto second =
      megamoe::build_notify_dispatch_for_testing(config, inputs, "repeat", 7);
  for (std::size_t rank = 0; rank < first.size(); ++rank) {
    REQUIRE(megamoe::serialize_dispatch_metadata(first[rank]) ==
            megamoe::serialize_dispatch_metadata(second[rank]));
  }
  REQUIRE(first[0]->incoming.mtile_expected_rows[0] ==
          std::vector<std::uint32_t>({4, 2}));
  REQUIRE(first[2]->outgoing.num_tokens == 0);
  verify_mapping(first);

  auto packet_config = make_config(2, 2, 6, 3, 4);
  packet_config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3};
  std::vector<RankRoutingInput> packet_inputs(4);
  packet_inputs[0] = {2, {2, 4, 3, 0, 2, 5}};
  const auto packet_handles = megamoe::build_notify_dispatch_for_testing(
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
      megamoe::build_notify_dispatch_for_testing(many, many_inputs, "many", 11);
  REQUIRE(many_handles[0]->outgoing.node_mask_words == 2);
  REQUIRE(many_handles[0]->outgoing.destination_node_mask ==
          std::vector<std::uint64_t>({1, std::uint64_t{1} << 8U}));
  REQUIRE(many_handles[0]->transport.outgoing_signals.size() == 73);
}

void test_topology_route_capacity_and_overflow() {
  auto bad = make_config(1, 6, 8, 2, 4);
  bad.topology.num_nodes = 4;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(bad, make_context(bad, 0),
                                                     {});
               },
               {"heterogeneous"});
  auto sparse = make_config(2, 2, 8, 2, 4);
  sparse.topology.rank_to_local_rank[1] = 2;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(
                     sparse, make_context(sparse, 0), {});
               },
               {"local coordinate"});
  auto duplicate = make_config(2, 2, 8, 2, 4);
  duplicate.topology.rank_to_node[2] = 0;
  duplicate.topology.rank_to_local_rank[2] = 0;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(
                     duplicate, make_context(duplicate, 0), {});
               },
               {"duplicate topology coordinate", "node=0", "local_rank=0"});
  auto bad_node = make_config(2, 2, 8, 2, 4);
  bad_node.topology.rank_to_node[0] = 2;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(
                     bad_node, make_context(bad_node, 0), {});
               },
               {"out-of-range node coordinate", "rank=0", "node=2"});
  auto topk = make_config(1, 1, 2, 3, 4);
  expect_error(NotifyErrorCode::InvalidRuntimeConfig,
               [&] {
                 megamoe::build_rank_local_histogram(topk,
                                                     make_context(topk, 0), {});
               },
               {"top_k exceeds"});

  auto config = make_config(1, 2, 4, 2, 4);
  const auto context = make_context(config, 0);
  expect_error(NotifyErrorCode::DuplicateExpert,
               [&] {
                 megamoe::build_rank_local_histogram(config, context,
                                                     {1, {2, 2}});
               },
               {"rank=0", "topk_slot=1"});
  expect_error(NotifyErrorCode::ExpertOutOfRange,
               [&] {
                 megamoe::build_rank_local_histogram(config, context,
                                                     {1, {2, 4}});
               },
               {"actual=4"});
  expect_error(NotifyErrorCode::ArithmeticOverflow,
               [&] {
                 megamoe::build_rank_local_histogram(
                     config, context,
                     {std::numeric_limits<std::uint64_t>::max(), {}});
               },
               {"route count"});
  config.capacity.expert_rows = {1, 100, 100, 100};
  const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {0, 2}}};
  expect_error(NotifyErrorCode::ExpertCapacityExceeded,
               [&] {
                 megamoe::build_notify_dispatch_for_testing(config, inputs,
                                                            "capacity", 1);
               },
               {"expert=0"});
  config.capacity.expert_rows.clear();
  config.capacity.workspace_rows_per_rank = {2, 100};
  expect_error(NotifyErrorCode::WorkspaceCapacityExceeded,
               [&] {
                 megamoe::build_notify_dispatch_for_testing(config, inputs,
                                                            "workspace-cap", 1);
               },
               {"destination_rank=0", "actual_rows=3", "capacity=2"});
}

void test_operation_move_and_terminal_state_machine() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "state", 1), {1, {0}});

  auto success = std::make_shared<BackendBehavior>();
  success->expert_rows = {{1, 0}};
  ScriptedBackend success_backend(success);
  auto original = megamoe::start_count_exchange(local, success_backend);
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
    auto operation = megamoe::start_count_exchange(local, cancelled_backend);
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
        megamoe::start_count_exchange(local, destination_backend);
    auto source_operation =
        megamoe::start_count_exchange(local, source_backend);
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
  struct SessionState final : megamoe::BackendSessionState {};
  struct ExecutionState final : megamoe::BackendExecutionState {};
  const auto config = make_config(1, 1, 2, 1, 4);
  auto session_state = std::make_shared<SessionState>();
  auto execution_state = std::make_shared<ExecutionState>();
  const std::weak_ptr<SessionState> session_lifetime = session_state;
  const std::weak_ptr<ExecutionState> execution_lifetime = execution_state;
  auto session =
      megamoe::make_collective_session("lifetime", 1, 1, session_state);
  auto execution = megamoe::make_execution_token("exec", execution_state);
  auto context = megamoe::make_rank_collective_context(session, 0, execution);
  auto local = megamoe::build_rank_local_histogram(config, context, {1, {0}});
  auto behavior = std::make_shared<BackendBehavior>();
  behavior->expert_rows = {{1, 0}};
  ScriptedBackend backend(behavior);
  {
    auto operation = megamoe::start_count_exchange(local, backend);
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
    const auto throwing_local = megamoe::build_rank_local_histogram(
        config, make_context(config, 0, "destructor", 2), {1, {0}});
    {
      auto operation =
          megamoe::start_count_exchange(throwing_local, throwing_backend);
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
  auto retained_context = megamoe::make_rank_collective_context(
      megamoe::make_collective_session("validated-life", 1, 3,
                                       retained_session_state),
      0,
      megamoe::make_execution_token("validated-exec",
                                    retained_execution_state));
  auto retained_local =
      megamoe::build_rank_local_histogram(config, retained_context, {1, {0}});
  auto retained_behavior = std::make_shared<BackendBehavior>();
  retained_behavior->expert_rows = {{1, 0}};
  ScriptedBackend retained_backend(retained_behavior);
  auto retained_operation =
      megamoe::start_count_exchange(retained_local, retained_backend);
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
  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "errors", 3), {1, {0}});
  LocalHistogramHandle empty_local;
  auto unused_behavior = std::make_shared<BackendBehavior>();
  ScriptedBackend unused_backend(unused_behavior);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)megamoe::start_count_exchange(empty_local, unused_backend); },
      {"phase=start", "actual=empty/default"});
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->start = mode;
    ScriptedBackend backend(behavior);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)megamoe::start_count_exchange(local, backend); },
                 {"phase=start", "kind=expert_counts", "rank=0", "errors",
                  "generation=3", "backend_error"});
  }
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->wait = mode;
    behavior->expert_rows = {{1, 0}};
    ScriptedBackend backend(behavior);
    auto operation = megamoe::start_count_exchange(local, backend);
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
    auto operation = megamoe::start_count_exchange(local, backend);
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
  auto device_op = megamoe::start_count_exchange(local, device_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)device_op.wait(); },
               {"typed host matrix", "opaque device lease"});

  auto kind_behavior = std::make_shared<BackendBehavior>();
  kind_behavior->expert_rows = {{1, 0}};
  kind_behavior->override_kind = true;
  kind_behavior->returned_kind = ExchangeKind::NotifyPacketSummary;
  ScriptedBackend kind_backend(kind_behavior);
  auto kind_op = megamoe::start_count_exchange(local, kind_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)kind_op.wait(); },
               {"collective kind mismatch", "expected=expert_counts",
                "actual=notify_packet_summary"});

  auto shape_behavior = std::make_shared<BackendBehavior>();
  shape_behavior->expert_rows = {};
  ScriptedBackend shape_backend(shape_behavior);
  auto shape_op = megamoe::start_count_exchange(local, shape_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)shape_op.wait(); },
               {"field=matrix.rows", "expected=1", "actual=0"});

  const auto valid_device = megamoe::make_device_buffer_lease(
      2, 3, 8, std::make_shared<TestDeviceLease>());
  REQUIRE(valid_device.rows() == 2);
  REQUIRE(valid_device.row_width() == 3);
  REQUIRE(valid_device.element_count() == 6);
  REQUIRE(valid_device.capacity_elements() == 8);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     0, 3, 3, std::make_shared<TestDeviceLease>());
               },
               {"phase=device-lease-validation", "field=rows", "expected=>0",
                "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     1, 0, 1, std::make_shared<TestDeviceLease>());
               },
               {"field=row_width", "expected=>0", "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     2, 4, 7, std::make_shared<TestDeviceLease>());
               },
               {"field=capacity_elements", "expected=>=8", "actual=7"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     1, 1, 0, std::make_shared<TestDeviceLease>());
               },
               {"field=capacity_elements", "expected=>0", "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     2, std::numeric_limits<std::uint64_t>::max(),
                     std::numeric_limits<std::uint64_t>::max(),
                     std::make_shared<TestDeviceLease>());
               },
               {"field=element_count.uint64", "actual=overflow"});
}

void test_context_kind_and_provenance_mismatch_messages() {
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)megamoe::make_collective_session("", 1, 1); },
               {"identity", "actual=empty"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)megamoe::make_collective_session("bad", 1, 0); },
               {"generation", "actual=0"});

  const auto fake_config = make_config(1, 2, 4, 1, 4);
  const auto fake_session_a =
      megamoe::make_collective_session("fake-cross", 2, 1);
  const auto fake_session_b =
      megamoe::make_collective_session("fake-cross", 2, 1);
  const auto fake_local_a = megamoe::build_rank_local_histogram(
      fake_config,
      megamoe::make_rank_collective_context(
          fake_session_a, 0, megamoe::make_execution_token("fake-a")),
      {});
  const auto fake_local_b = megamoe::build_rank_local_histogram(
      fake_config,
      megamoe::make_rank_collective_context(
          fake_session_b, 1, megamoe::make_execution_token("fake-b")),
      {});
  HostFakeCountExchange cross_backend(2, "fake-cross");
  auto first_fake_operation =
      megamoe::start_count_exchange(fake_local_a, cross_backend);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)megamoe::start_count_exchange(fake_local_b, cross_backend); },
      {"phase=start", "cross-session mix", "different session instance"});
  first_fake_operation.cancel();

  const auto config = make_config(1, 2, 4, 2, 4);
  const auto session = megamoe::make_collective_session("base", 2, 4);
  const auto execution0 = megamoe::make_execution_token("exec0");
  const auto context0 =
      megamoe::make_rank_collective_context(session, 0, execution0);
  const auto local0 =
      megamoe::build_rank_local_histogram(config, context0, {1, {0, 1}});
  const auto count0 = complete_scripted(local0, ExchangeKind::ExpertCounts,
                                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  const auto summary0 =
      complete_scripted(local0, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
  (void)megamoe::materialize_rank_metadata(local0, count0, summary0);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)megamoe::materialize_rank_metadata(local0, count0, count0); },
      {"field=collective.kind", "expected=notify_packet_summary",
       "actual=expert_counts"});

  const auto local_rank1 = megamoe::build_rank_local_histogram(
      config,
      megamoe::make_rank_collective_context(
          session, 1, megamoe::make_execution_token("exec1")),
      {});
  const auto count_rank1 = complete_scripted(
      local_rank1, ExchangeKind::ExpertCounts, {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local0, count_rank1,
                                                          summary0);
               },
               {"field=calling_rank", "expected=0", "actual=1"});

  const auto other_execution = megamoe::build_rank_local_histogram(
      config,
      megamoe::make_rank_collective_context(
          session, 0, megamoe::make_execution_token("other-exec")),
      {1, {0, 1}});
  const auto other_exec_count =
      complete_scripted(other_execution, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, other_exec_count, summary0);
               },
               {"field=execution.provenance", "exec0", "other-exec"});

  const auto generation_context = make_context(config, 0, "base", 5, "exec0");
  const auto generation_local = megamoe::build_rank_local_histogram(
      config, generation_context, {1, {0, 1}});
  const auto generation_count =
      complete_scripted(generation_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, generation_count, summary0);
               },
               {"field=session.generation", "expected=4", "actual=5"});

  const auto session_local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "other-session", 4, "exec0"),
      {1, {0, 1}});
  const auto session_count = complete_scripted(
      session_local, ExchangeKind::ExpertCounts, {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local0, session_count,
                                                          summary0);
               },
               {"field=session.identity", "base", "other-session"});

  const auto same_fields_local =
      megamoe::build_rank_local_histogram(config, context0, {1, {0, 1}});
  const auto same_fields_count =
      complete_scripted(same_fields_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, same_fields_count, summary0);
               },
               {"field=local.provenance"});

  const auto same_values_session =
      megamoe::make_collective_session("base", 2, 4);
  const auto other_session_instance_local = megamoe::build_rank_local_histogram(
      config,
      megamoe::make_rank_collective_context(same_values_session, 0, execution0),
      {1, {0, 1}});
  const auto other_session_instance_count = complete_scripted(
      other_session_instance_local, ExchangeKind::ExpertCounts,
      {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, other_session_instance_count, summary0);
               },
               {"field=session.provenance", "different session instance"});

  const auto world3 = make_config(1, 3, 4, 2, 4);
  const auto world3_local = megamoe::build_rank_local_histogram(
      world3, make_context(world3, 0, "base", 4, "exec0"), {1, {0, 1}});
  const auto world3_count =
      complete_scripted(world3_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local0, world3_count,
                                                          summary0);
               },
               {"field=world_size", "expected=2", "actual=3"});
}

void test_malicious_completion_context_and_corruption_hooks() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "malicious", 2), {1, {0}});
  auto behavior = std::make_shared<BackendBehavior>();
  behavior->expert_rows = {{1, 0}};
  behavior->override_context = true;
  behavior->returned_context = megamoe::make_rank_collective_context(
      local.context().session(), 0, megamoe::make_execution_token("other"));
  ScriptedBackend backend(behavior);
  auto operation = megamoe::start_count_exchange(local, backend);
  expect_error(
      NotifyErrorCode::CountExchangeFailure, [&] { (void)operation.wait(); },
      {"execution token mismatch", "expected='exec'", "actual='other'"});

  const auto count =
      complete_scripted(local, ExchangeKind::ExpertCounts, {{1, 0}});
  const auto summary =
      complete_scripted(local, ExchangeKind::NotifyPacketSummary, {{1}});
  const auto bad_expert =
      megamoe::testing::NotifyCorruptionHook::with_expert_id(local, 0, 99);
  expect_error(NotifyErrorCode::ExpertOutOfRange,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_expert, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.expert_id", "rank=0",
                "expected=[0,2)", "actual=99"});
  const auto truncated =
      megamoe::testing::NotifyCorruptionHook::with_truncated_expert_ids(local);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(truncated, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=expert_id", "rank=0",
                "expected=1", "actual=0"});

  const auto bad_offset =
      megamoe::testing::NotifyCorruptionHook::with_local_offset(local, 0, 7);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_offset, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.local_offset",
                "rank=0", "expected=0", "actual=7"});
  const auto bad_combine =
      megamoe::testing::NotifyCorruptionHook::with_combine_order(local, 0, 9);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_combine, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.combine_order",
                "rank=0", "actual=9"});
  const auto bad_mask =
      megamoe::testing::NotifyCorruptionHook::with_node_mask(local, 0, 0);
  expect_error(
      NotifyErrorCode::InvalidRouteShape,
      [&] {
        (void)megamoe::materialize_rank_metadata(bad_mask, count, summary);
      },
      {"phase=materialize-local", "field=outgoing.destination_node_mask",
       "rank=0", "expected=recomputed node mask", "actual=content mismatch"});
  const auto bad_local_count =
      megamoe::testing::NotifyCorruptionHook::with_local_expert_count(local, 0,
                                                                      8);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_local_count,
                                                          count, summary);
               },
               {"phase=materialize-local", "field=local_expert_count", "rank=0",
                "actual=content mismatch"});

  const auto bad_matrix_shape =
      megamoe::testing::NotifyCorruptionHook::with_collective_matrix(count, {});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local, bad_matrix_shape, summary);
               },
               {"phase=materialize", "field=matrix.rows", "rank=0",
                "expected=1", "actual=0"});
  const auto bad_matrix_width =
      megamoe::testing::NotifyCorruptionHook::with_collective_matrix(count,
                                                                     {{1}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local, bad_matrix_width, summary);
               },
               {"phase=materialize", "field=matrix.row_width", "rank=0",
                "expected=2", "actual=1"});
  const auto bad_matrix_row =
      megamoe::testing::NotifyCorruptionHook::with_collective_matrix(count,
                                                                     {{0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local, bad_matrix_row,
                                                          summary);
               },
               {"phase=materialize", "field=expert_counts.calling_rank_row",
                "rank=0", "expected=local histogram count row",
                "actual=content mismatch"});

  const auto handle = megamoe::materialize_rank_metadata(local, count, summary);
  const auto validate_corrupt = [&](const DispatchHandle &corrupt,
                                    const std::string &field) {
    expect_error(NotifyErrorCode::InvalidHandle,
                 [&] { (void)megamoe::make_workspace_plan(corrupt); },
                 {"phase=dispatch-validation", "field=" + field, "rank=0",
                  "expected=", "actual="});
  };
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_expert_row_count(handle, 0,
                                                                    9),
      "incoming.expert_row_count");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_expert_row_prefix(handle, 1,
                                                                     9),
      "incoming.expert_row_prefix");
  validate_corrupt(megamoe::testing::NotifyCorruptionHook::
                       with_packet_identity_destination_node(handle, 0, 9),
                   "transport.packet.identity");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_packet_proxy(handle, 0, 9),
      "transport.packet.proxy_rank");
  validate_corrupt(megamoe::testing::NotifyCorruptionHook::with_fanout_expert(
                       handle, 0, 0, 9),
                   "transport.fanout.expert_id");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_fanout_local_rank(handle, 0,
                                                                     0, 9),
      "transport.fanout.destination_local_rank");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_fanout_row(handle, 0, 0, 9),
      "transport.fanout.dst_row");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_fanout_slot(handle, 0, 0, 9),
      "transport.fanout.topk_slot");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_mtile_row(handle, 0, 0, 9),
      "incoming.mtile_expected_rows");
}

void test_handle_copy_move_workspace_and_overflow() {
  const auto config = make_config(1, 2, 4, 2, 4);
  const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {2, 3}}};
  const auto handles = megamoe::build_notify_dispatch_for_testing(
      config, inputs, "workspace", 5);
  DispatchHandle empty;
  expect_error(NotifyErrorCode::InvalidHandle, [&] { (void)empty.metadata(); });
  const auto plan = megamoe::make_workspace_plan(handles[0]);
  megamoe::validate_workspace_plan(handles[0], plan);
  (void)megamoe::ForwardPersistentInputs::checked(handles[0], plan);
  (void)megamoe::BackwardPersistentInputs::checked(handles[0], plan);
  megamoe::WorkspacePlan empty_plan;
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
      [&] { megamoe::validate_workspace_plan(handles[0], empty_plan); },
      {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { megamoe::validate_workspace_plan(handles[1], plan); },
               {"another DispatchHandle"});
  const auto other = megamoe::build_notify_dispatch_for_testing(config, inputs,
                                                                "workspace", 5);
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { megamoe::validate_workspace_plan(other[0], plan); },
               {"another DispatchHandle"});

  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "overflow", 1), {1, {0, 1}});
  const auto counts = complete_scripted(
      local, ExchangeKind::ExpertCounts,
      {{1, 1, 0, 0}, {std::numeric_limits<std::uint64_t>::max(), 0, 0, 0}});
  const auto summary =
      complete_scripted(local, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
  expect_error(
      NotifyErrorCode::ArithmeticOverflow,
      [&] { (void)megamoe::materialize_rank_metadata(local, counts, summary); },
      {"source prefix"});
}

void test_fixed_control_protocol_agreement() {
  const auto config = make_config(1, 2, 8, 3, 4);
  const auto context0 = make_context(config, 0, "control", 17, "control-exec-0");
  const auto context1 = make_context(config, 1, "control", 17, "control-exec-1");
  const auto record0 = megamoe::cuda::make_control_record(
      config, context0, 2, 6, 10);
  const auto record1 = megamoe::cuda::make_control_record(
      config, context1, 3, 9, 10);
  megamoe::cuda::validate_control_matrix({record0, record1});

  const auto expect_control_error = [&](const megamoe::cuda::ControlMatrix &rows,
                                        const std::vector<std::string> &fields) {
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] { megamoe::cuda::validate_control_matrix(rows); },
                 fields);
  };

  const auto max_mismatch = megamoe::cuda::make_control_record(
      config, context1, 3, 9, 11);
  expect_control_error({record0, max_mismatch},
                       {"phase=control-agreement", "field=max_tokens_per_rank",
                        "origin_rank=1", "expected=10", "actual=11"});

  auto expert_config = make_config(1, 2, 9, 3, 4);
  const auto expert_mismatch = megamoe::cuda::make_control_record(
      expert_config, context1, 3, 9, 10);
  expect_control_error({record0, expert_mismatch},
                       {"field=num_experts", "origin_rank=1"});

  megamoe::cuda::LocalPreflightStatus invalid_route;
  invalid_route.error = megamoe::cuda::ControlLocalError::ExpertOutOfRange;
  invalid_route.field = megamoe::cuda::ControlField::LocalValidation;
  invalid_route.token = 4;
  invalid_route.slot = 2;
  invalid_route.expert = 99;
  invalid_route.expected = 8;
  invalid_route.actual = 99;
  const auto bad_route = megamoe::cuda::make_control_record(
      config, context1, 5, 15, 10, invalid_route);
  expect_control_error({record0, bad_route},
                       {"field=local_validation", "origin_rank=1",
                        "token=4", "slot=2", "expert=99"});

  const auto over_capacity = megamoe::cuda::make_control_record(
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
    megamoe::cuda::validate_control_matrix({record0, generation_mismatch});
  } catch (const NotifyDispatchError &error) {
    first_error = error.what();
  }
  try {
    megamoe::cuda::validate_control_matrix({record0, generation_mismatch});
  } catch (const NotifyDispatchError &error) {
    second_error = error.what();
  }
  REQUIRE(!first_error.empty() && first_error == second_error);

  const auto wire0 = megamoe::cuda::encode_control_header(record0);
  REQUIRE(megamoe::cuda::decode_control_header(wire0).generation == 17);
  REQUIRE(static_cast<unsigned>(wire0[0]) == 0x43U &&
          static_cast<unsigned>(wire0[7]) == 0x4dU);
  auto corrupt_wire = wire0;
  corrupt_wire[80] ^= std::byte{1};
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] { (void)megamoe::cuda::decode_control_header(corrupt_wire); },
               {"field=checksum"});
  auto byteswapped = wire0;
  for (std::size_t word = 0; word < megamoe::cuda::kControlSchemaWords;
       ++word)
    std::reverse(byteswapped.begin() + static_cast<std::ptrdiff_t>(word * 8),
                 byteswapped.begin() +
                     static_cast<std::ptrdiff_t>((word + 1) * 8));
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] { (void)megamoe::cuda::decode_control_header(byteswapped); },
               {"phase=control-agreement"});

  const auto config0 = megamoe::cuda::encode_canonical_config(
      config, 10, context0.session().stable_identity_bytes());
  const auto decoded_config =
      megamoe::cuda::decode_canonical_config(config0);
  REQUIRE(decoded_config.rank_to_node == config.topology.rank_to_node);
  REQUIRE(decoded_config.expert_to_rank == config.topology.expert_to_rank);
  REQUIRE(decoded_config.stable_session_identity ==
          context0.session().stable_identity_bytes());
  REQUIRE(config0.size() >= megamoe::cuda::kCanonicalConfigHeaderBytes);
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
         word * 8U < megamoe::cuda::kCanonicalConfigHeaderBytes; ++word)
      std::reverse(
          byteswapped_config.begin() +
              static_cast<std::ptrdiff_t>(word * 8U),
          byteswapped_config.begin() +
              static_cast<std::ptrdiff_t>((word + 1U) * 8U));
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   (void)megamoe::cuda::decode_canonical_config(
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
  world1_header.config_digest = megamoe::cuda::config_wire_digest(config0);
  megamoe::cuda::validate_config_matrix({config0}, {world1_header});
  const auto corrupt_config = [&](std::size_t offset,
                                  const std::string &needle) {
    auto damaged = config0;
    damaged.at(offset) ^= std::byte{1};
    auto header = world1_header;
    header.config_digest = megamoe::cuda::config_wire_digest(damaged);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix({damaged}, {header});
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
    header.config_digest = megamoe::cuda::config_wire_digest(truncated);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix({truncated}, {header});
                 },
                 {"truncated"});
  }
  {
    auto trailing = config0;
    trailing.push_back(std::byte{0});
    auto header = world1_header;
    header.config_blob_bytes = trailing.size();
    header.config_digest = megamoe::cuda::config_wire_digest(trailing);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix({trailing}, {header});
                 },
                 {"trailing_bytes"});
  }
  {
    auto wrong_header_digest = world1_header;
    ++wrong_header_digest.config_digest;
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix(
                       {config0}, {wrong_header_digest});
                 },
                 {"field=config_digest", "control_header_vs_wire"});
  }
  auto config_changed = config;
  std::swap(config_changed.topology.expert_to_rank[0],
            config_changed.topology.expert_to_rank[1]);
  const auto config1 = megamoe::cuda::encode_canonical_config(
      config_changed, 10, context1.session().stable_identity_bytes());
  auto same_digest0 = record0;
  auto same_digest1 = record1;
  same_digest0.config_blob_bytes = config0.size();
  same_digest1.config_blob_bytes = config1.size();
  same_digest0.config_digest = megamoe::cuda::config_wire_digest(config0);
  same_digest1.config_digest = megamoe::cuda::config_wire_digest(config1);
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_config_matrix(
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
                 megamoe::cuda::validate_config_matrix(
                     {config0, config1}, {same_digest0, forged_digest1});
               },
               {"field=config_digest", "origin_rank=1"});

  // Original stable identity bytes, not their two header summaries or display
  // names, are the final session agreement boundary.
  const std::vector<std::byte> raw_identity0{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x30}};
  const std::vector<std::byte> raw_identity1{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x31}};
  const auto raw_session0 = megamoe::make_collective_session(
      "same-display", raw_identity0, 2, 91);
  const auto raw_session1 = megamoe::make_collective_session(
      "same-display", raw_identity1, 2, 91);
  const auto raw_context0 = megamoe::make_rank_collective_context(
      raw_session0, 0, megamoe::make_execution_token("raw-exec-0"));
  const auto raw_context1 = megamoe::make_rank_collective_context(
      raw_session1, 1, megamoe::make_execution_token("raw-exec-1"));
  const auto raw_wire0 = megamoe::cuda::encode_canonical_config(
      config, 10, raw_identity0);
  const auto raw_wire1 = megamoe::cuda::encode_canonical_config(
      config, 10, raw_identity1);
  auto raw_header0 = megamoe::cuda::make_control_record(
      config, raw_context0, 1, 3, 10);
  auto raw_header1 = megamoe::cuda::make_control_record(
      config, raw_context1, 1, 3, 10);
  // Simulate an adversarial summary collision: fixed control passes, exact raw
  // identity comparison in config agreement still fails.
  raw_header1.session_id_word0 = raw_header0.session_id_word0;
  raw_header1.session_id_word1 = raw_header0.session_id_word1;
  megamoe::cuda::validate_control_matrix({raw_header0, raw_header1});
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_config_matrix(
                     {raw_wire0, raw_wire1}, {raw_header0, raw_header1});
               },
               {"field=session_identity", "origin_rank=1", "coordinate=2"});

  const auto config3 = make_config(1, 3, 9, 3, 4);
  const auto context30 = make_context(config3, 0, "common", 44, "e0");
  const auto context31 = make_context(config3, 1, "common", 44, "e1");
  const auto context32 = make_context(config3, 2, "common", 44, "e2");
  const auto row30 = megamoe::cuda::make_control_record(
      config3, context30, 1, 3, 10);
  auto row31 = megamoe::cuda::make_control_record(
      config3, context31, 1, 3, 10);
  auto row32 = megamoe::cuda::make_control_record(
      config3, context32, 1, 3, 10);
  ++row31.session_id_word1;
  ++row32.generation;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_control_matrix(
                     {row30, row31, row32});
               },
               {"field=session_id_word1", "origin_rank=1"});

  // For one rank with several differences, fixed field order selects
  // generation before session and config fields.
  ++row31.generation;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_control_matrix(
                     {row30, row31, row32});
               },
               {"field=generation", "origin_rank=1"});
}

} // namespace

int main() {
  const std::vector<std::pair<const char *, std::function<void()>>> tests{
      {"rank_local_async_and_deterministic_mapping",
       test_rank_local_async_and_deterministic_mapping},
      {"existing_shapes_descriptors_and_repeatability",
       test_existing_shapes_descriptors_and_repeatability},
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
