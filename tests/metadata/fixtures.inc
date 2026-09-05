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
    megamoe::cuda::testing::set_bootstrap_io_hooks(
        scripted_bootstrap_read, scripted_bootstrap_write,
        scripted_bootstrap_close);
  }
  BootstrapIoHookGuard(const BootstrapIoHookGuard &) = delete;
  BootstrapIoHookGuard &operator=(const BootstrapIoHookGuard &) = delete;
  ~BootstrapIoHookGuard() {
    megamoe::cuda::testing::clear_bootstrap_io_hooks();
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
static_assert(
    !std::is_copy_constructible_v<megamoe::cuda::NcclBootstrapInputChannel>);
static_assert(
    std::is_nothrow_move_constructible_v<
        megamoe::cuda::NcclBootstrapInputChannel>);
static_assert(
    !std::is_copy_constructible_v<megamoe::cuda::NcclBootstrapOutputChannel>);
static_assert(
    std::is_nothrow_move_constructible_v<
        megamoe::cuda::NcclBootstrapOutputChannel>);

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
    behavior_->input_lifetime =
        megamoe::testing::HostIntegerRowTestAccess::lifetime(request.input);
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
