struct DeviceNotifyResourcePool::Impl
    : std::enable_shared_from_this<DeviceNotifyResourcePool::Impl> {
  struct SlotState {
    enum class State : std::uint8_t { Available, InUse, Quarantined };
    State state = State::Available;
    std::uint64_t epoch = 0;
    std::shared_ptr<CudaNotifyPoolAllocation> notify;
    std::shared_ptr<NcclAllGatherCountExchange> backend;
  };

  struct Lease {
    std::shared_ptr<Impl> pool;
    std::size_t index = 0;
    std::uint64_t epoch = 0;
    ~Lease() noexcept {
      if (!pool)
        return;
      auto &slot = pool->slots[index];
      const auto drained =
          DeviceNotifyPoolBridge::drain(slot.notify, pool->execution);
      std::lock_guard<std::mutex> lock(pool->mutex);
      const bool quarantined =
          DeviceNotifyPoolBridge::quarantined(slot.notify);
      if (slot.state == SlotState::State::InUse && slot.epoch == epoch) {
        if (pool->in_use != 0)
          --pool->in_use;
        slot.state = drained && !quarantined
                         ? SlotState::State::Available
                         : SlotState::State::Quarantined;
      }
    }
  };

  NcclSession base_session;
  CudaExecution execution;
  DeviceNotifyResourcePoolConfig config;
  CudaNotifyWorkspacePlan max_plan;
  std::vector<SlotState> slots;
  mutable std::mutex mutex;
  std::uint32_t in_use = 0;
  std::uint64_t epoch_high = 0;
  double setup_us = 0;

  std::shared_ptr<Lease> acquire(const NcclSession &generation) {
    if (!generation || !generation.shares_communicator_with(base_session))
      throw NotifyDispatchError(
          NotifyErrorCode::InvalidHandle,
          "phase=device-pool-acquire, field=session, rank=n/a, "
          "expected=same native communicator, actual=mismatch");
    std::lock_guard<std::mutex> lock(mutex);
    for (std::size_t index = 0; index < slots.size(); ++index) {
      auto &slot = slots[index];
      if (slot.state != SlotState::State::Available)
        continue;
      slot.state = SlotState::State::InUse;
      slot.epoch = ++epoch_high;
      ++in_use;
      try {
        slot.backend->prepare_generation(generation);
      } catch (...) {
        slot.state = SlotState::State::Available;
        --in_use;
        throw;
      }
      auto lease = std::make_shared<Lease>();
      lease->pool = shared_from_this();
      lease->index = index;
      lease->epoch = slot.epoch;
      return lease;
    }
    throw NotifyDispatchError(
        NotifyErrorCode::ResourceExhausted,
        "phase=device-pool-acquire, field=available_slots, rank=" +
            std::to_string(generation.rank()) + ", expected>=1, actual=0");
  }
};

DeviceNotifyResourcePool make_device_notify_resource_pool(
    const NcclSession &base_session, const CudaExecution &execution,
    DeviceNotifyResourcePoolConfig config) {
  if (!base_session || !execution || config.max_tokens_per_rank == 0 ||
      config.max_inflight_generations == 0 ||
      config.max_inflight_consumers == 0)
    invalid("phase=device-pool-create, field=input, rank=n/a, "
            "expected=valid nonzero capacity, actual=invalid");
  validate_runtime_config(config.capacity_config);
  if (config.capacity_config.topology.world_size !=
          base_session.session().world_size() ||
      base_session.device_ordinal() != execution.device_ordinal())
    invalid("phase=device-pool-create, field=session/config/execution, rank=" +
            std::to_string(base_session.rank()) +
            ", expected=matching world/device, actual=mismatch");
  ConfigWire wire;
  LocalPreflightStatus status;
  if (!try_encode_canonical_config(
          config.capacity_config, config.max_tokens_per_rank, wire, status,
          base_session.session().stable_identity_bytes()))
    invalid("phase=device-pool-create, field=config_wire, rank=" +
            std::to_string(base_session.rank()) +
            ", expected=encodable capacity config, actual=invalid");
  if (config.max_config_wire_bytes == 0)
    config.max_config_wire_bytes = wire.size();
  if (config.max_config_wire_bytes < wire.size() ||
      config.max_config_wire_bytes > kMaxCanonicalConfigBytes ||
      config.max_config_wire_bytes > SIZE_MAX)
    invalid("phase=device-pool-create, field=max_config_wire_bytes, rank=" +
            std::to_string(base_session.rank()) + ", expected=[" +
            std::to_string(wire.size()) + "," +
            std::to_string(kMaxCanonicalConfigBytes) + "], actual=" +
            std::to_string(config.max_config_wire_bytes));

  const auto start = std::chrono::steady_clock::now();
  auto impl = std::make_shared<DeviceNotifyResourcePool::Impl>();
  impl->base_session = base_session;
  impl->execution = execution;
  impl->config = config;
  impl->max_plan = query_cuda_notify_workspace(
      config.capacity_config, base_session.rank(),
      config.max_tokens_per_rank, config.max_tokens_per_rank);
  impl->slots.resize(config.max_inflight_generations);
  for (auto &slot : impl->slots) {
    slot.notify = DeviceNotifyPoolBridge::allocate(
        impl->max_plan, execution, config.max_inflight_consumers);
    slot.backend = std::make_shared<NcclAllGatherCountExchange>(
        base_session, execution, config.capacity_config.num_experts,
        config.capacity_config.topology.num_nodes,
        static_cast<std::size_t>(config.max_config_wire_bytes));
  }
  impl->setup_us =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - start)
          .count();
  return DeviceNotifyResourcePool(std::move(impl));
}

DeviceNotifyResourcePoolUsage DeviceNotifyResourcePool::usage() const {
  if (!impl_)
    invalid("phase=device-pool-usage, field=pool, rank=n/a, "
            "expected=valid, actual=empty");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceNotifyResourcePoolUsage usage;
  usage.slot_count = static_cast<std::uint32_t>(impl_->slots.size());
  usage.slot_epoch_high_watermark = impl_->epoch_high;
  usage.one_time_setup_us = impl_->setup_us;
  for (const auto &slot : impl_->slots) {
    const bool runtime_quarantined =
        DeviceNotifyPoolBridge::quarantined(slot.notify);
    if (runtime_quarantined ||
        slot.state == DeviceNotifyResourcePool::Impl::SlotState::State::Quarantined)
      ++usage.slots_quarantined;
    else if (slot.state == DeviceNotifyResourcePool::Impl::SlotState::State::Available)
      ++usage.slots_available;
    else
      ++usage.slots_in_use;
    const auto events = DeviceNotifyPoolBridge::consumer_event_usage(slot.notify);
    usage.consumer_events_available += events[0];
    usage.consumer_events_in_use += events[1];
    usage.consumer_events_quarantined += events[2];
    usage.device_bytes += DeviceNotifyPoolBridge::device_bytes(slot.notify);
    usage.pinned_host_bytes +=
        DeviceNotifyPoolBridge::pinned_bytes(slot.notify);
    const auto backend = slot.backend->workspace_usage();
    usage.device_bytes += backend.device_bytes;
    usage.pinned_host_bytes += backend.pinned_host_bytes;
    usage.event_count +=
        5U + DeviceNotifyPoolBridge::event_count(slot.notify);
  }
  return usage;
}

struct ValidatedCudaGeneration::Impl {
  CudaDispatchHandle dispatch;
  CollectiveProtocolAudit audit;
  NcclWorkspaceUsage workspace;
  GenerationSetupTimings setup;
  GenerationExecutionTimings execution;
  std::shared_ptr<const void> resource_lease;
};

const CudaDispatchHandle &ValidatedCudaGeneration::dispatch() const {
  if (!impl_)
    invalid("phase=generation-result, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->dispatch;
}

const CollectiveProtocolAudit &ValidatedCudaGeneration::audit() const {
  if (!impl_)
    invalid("phase=generation-result, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->audit;
}

const NcclWorkspaceUsage &ValidatedCudaGeneration::workspace_usage() const {
  if (!impl_)
    invalid("phase=generation-result, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->workspace;
}

const GenerationSetupTimings &ValidatedCudaGeneration::setup_timings() const {
  if (!impl_)
    invalid("phase=generation-result, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->setup;
}

const GenerationExecutionTimings &
ValidatedCudaGeneration::execution_timings() const {
  if (!impl_)
    invalid("phase=generation-result, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->execution;
}

struct CudaNotifyGenerationCoordinator::Impl {
  NcclSession session;
  CudaExecution execution;
  RankCollectiveContext context;
  std::shared_ptr<NcclAllGatherCountExchange> backend;
  std::shared_ptr<DeviceNotifyResourcePool::Impl> pool;
  GenerationSetupTimings setup;
  bool used = false;
  CudaRouteInput direct_routes;
  CudaGateInput direct_gates;
};

struct GenerationCancellationToken::State {
  std::atomic<bool> requested{false};
};

GenerationCancellationToken::GenerationCancellationToken()
    : state_(std::make_shared<State>()) {}

void GenerationCancellationToken::request_cancel() const noexcept {
  if (state_)
    state_->requested.store(true, std::memory_order_release);
}

bool GenerationCancellationToken::cancellation_requested() const noexcept {
  return state_ && state_->requested.load(std::memory_order_acquire);
}

