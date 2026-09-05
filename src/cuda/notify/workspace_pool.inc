CudaNotifyWorkspacePlan query_cuda_notify_workspace(
    const RuntimeConfig &config, std::uint32_t calling_rank,
    std::uint64_t num_tokens, std::uint64_t max_tokens_per_rank) {
  validate_runtime_config(config);
  if (calling_rank >= config.topology.world_size)
    invalid("phase=cuda-workspace-query, field=calling_rank, rank=" +
            std::to_string(calling_rank) + ", expected=[0," +
            std::to_string(config.topology.world_size) + "), actual=" +
            std::to_string(calling_rank));
  if (max_tokens_per_rank < num_tokens)
    invalid("phase=cuda-workspace-query, field=max_tokens_per_rank, rank=" +
            std::to_string(calling_rank) + ", expected>=" +
            std::to_string(num_tokens) + ", actual=" +
            std::to_string(max_tokens_per_rank));
  CudaNotifyWorkspacePlan plan;
  plan.calling_rank = calling_rank;
  plan.num_tokens = num_tokens;
  plan.max_tokens_per_rank = max_tokens_per_rank;
  plan.top_k = config.top_k;
  plan.num_experts = config.num_experts;
  plan.world_size = config.topology.world_size;
  plan.num_nodes = config.topology.num_nodes;
  plan.node_mask_words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(plan.num_nodes) + 63U) / 64U);
  plan.route_count = checked_mul(num_tokens, config.top_k, "route_count");
  plan.max_incoming_rows = checked_mul(
      checked_mul(config.topology.world_size, max_tokens_per_rank,
                  "max_global_tokens"),
      config.top_k, "max_incoming_rows");
  if (plan.route_count > static_cast<std::uint64_t>(INT_MAX))
    invalid("phase=cuda-workspace-query, field=route_count, rank=n/a, expected<=INT_MAX, actual=" +
            std::to_string(plan.route_count));
  if (plan.route_count != 0 && config.num_experts >
                                   std::numeric_limits<std::uint64_t>::max() /
                                       plan.route_count)
    invalid("phase=cuda-workspace-query, field=sort_key, rank=n/a, expected=expert*route_count within uint64, actual=overflow");
  const auto cub_items = static_cast<int>(std::max<std::uint64_t>(plan.route_count, 1));
  std::size_t sort_bytes = 0;
  std::size_t scan_bytes = 0;
  (void)cub::DeviceRadixSort::SortPairs(
      nullptr, sort_bytes, static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr), cub_items);
  (void)cub::DeviceScan::InclusiveSumByKey(
      nullptr, scan_bytes, static_cast<std::uint32_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr), cub_items);
  std::size_t plain_scan_bytes = 0;
  (void)cub::DeviceScan::InclusiveSum(
      nullptr, plain_scan_bytes, static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr), cub_items);
  scan_bytes = std::max(scan_bytes, plain_scan_bytes);
  plan.sort_temp_bytes = sort_bytes;
  plan.scan_temp_bytes = scan_bytes;
  const auto device = make_device_layout(plan);
  const auto host = make_host_layout(plan);
  plan.device_bytes = device.total;
  plan.host_staging_bytes = host.total;
  return plan;
}

struct CudaNotifyWorkspace::Impl {
  std::shared_ptr<WorkspaceState> state;
};

CudaNotifyWorkspace::CudaNotifyWorkspace() = default;
CudaNotifyWorkspace::CudaNotifyWorkspace(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaNotifyWorkspace::~CudaNotifyWorkspace() noexcept = default;
CudaNotifyWorkspace::CudaNotifyWorkspace(CudaNotifyWorkspace &&) noexcept = default;
CudaNotifyWorkspace &CudaNotifyWorkspace::operator=(CudaNotifyWorkspace &&) noexcept = default;

const CudaNotifyWorkspacePlan &CudaNotifyWorkspace::plan() const {
  if (!impl_ || !impl_->state)
    invalid("attempted to access empty CUDA notify workspace");
  return impl_->state->plan;
}

CudaNotifyWorkspace allocate_cuda_notify_workspace(
    const CudaNotifyWorkspacePlan &plan,
    const RankCollectiveContext &context, const CudaExecution &execution) {
  if (!context || !execution)
    invalid("phase=cuda-workspace-allocate, field=context, rank=n/a, expected=valid, actual=empty");
  if (plan.calling_rank != context.calling_rank())
    invalid("phase=cuda-workspace-allocate, field=calling_rank, rank=" +
            std::to_string(context.calling_rank()) + ", expected=" +
            std::to_string(context.calling_rank()) + ", actual=" +
            std::to_string(plan.calling_rank));
  if (plan.world_size != context.session().world_size())
    invalid("phase=cuda-workspace-allocate, field=world_size, rank=" +
            std::to_string(context.calling_rank()) + ", expected=" +
            std::to_string(context.session().world_size()) + ", actual=" +
            std::to_string(plan.world_size));
  if (!context.execution().same_instance(execution.token()))
    invalid("phase=cuda-workspace-allocate, field=execution, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=context execution instance, actual=different");
  const auto state_exec = internal::checked_execution_state(execution);
  auto state = std::make_shared<WorkspaceState>();
  state->plan = plan;
  state->device_layout = make_device_layout(plan);
  state->host_layout = make_host_layout(plan);
  if (state->device_layout.total != plan.device_bytes ||
      state->host_layout.total != plan.host_staging_bytes)
    invalid("phase=cuda-workspace-allocate, field=plan.bytes, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=recomputed layout, actual=mismatch");
  state->context = context;
  state->epoch = context.session().generation();
  state->execution = execution;
  state->allocation = std::make_shared<CudaAllocation>();
  auto &allocation = *state->allocation;
  allocation.device = state_exec->device;
  allocation.device_bytes = checked_size(plan.device_bytes, "device_bytes");
  allocation.host_bytes = checked_size(plan.host_staging_bytes, "host_bytes");
  internal::throw_cuda("workspace-allocate", cudaSetDevice(static_cast<int>(allocation.device)),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda("workspace-allocate",
                       cudaMalloc(&allocation.device_ptr,
                                  std::max<std::size_t>(allocation.device_bytes, 1)),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaMalloc(workspace)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda("workspace-allocate",
                       cudaHostAlloc(&allocation.host_ptr,
                                     std::max<std::size_t>(allocation.host_bytes, 1),
                                     cudaHostAllocPortable |
                                         cudaHostAllocMapped),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaHostAlloc(staging)");
  internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
  internal::throw_cuda(
      "workspace-allocate",
      cudaHostGetDevicePointer(&allocation.host_device_ptr,
                               allocation.host_ptr, 0),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaHostGetDevicePointer(staging)");
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.mapping_start),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate(mapping start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.mapping_complete),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate(mapping complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreateWithFlags(&allocation.completion,
                                                cudaEventDisableTiming),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.planning_start),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventCreate(planning start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.planning_complete),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventCreate(planning complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  allocation.consumer_events.reserve(4);
  for (std::uint32_t i = 0; i < 4; ++i) {
    cudaEvent_t event = nullptr;
    internal::throw_cuda("workspace-allocate",
                         cudaEventCreateWithFlags(&event,
                                                  cudaEventDisableTiming),
                         context.calling_rank(), context.session().world_size(),
                         context.session().identity(), context.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "cudaEventCreate(consumer completion)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    allocation.consumer_events.push_back(event);
    allocation.consumer_event_states.push_back(
        internal::ConsumerEventSlotState::Available);
  }
  auto impl = std::make_shared<CudaNotifyWorkspace::Impl>();
  impl->state = std::move(state);
  return CudaNotifyWorkspace(std::move(impl));
}

std::shared_ptr<CudaNotifyPoolAllocation> DeviceNotifyPoolBridge::allocate(
    const CudaNotifyWorkspacePlan &max_plan,
    const CudaExecution &execution,
    std::uint32_t max_inflight_consumers) {
  if (!execution)
    invalid("phase=device-pool-allocate, field=execution, rank=n/a, "
            "expected=valid, actual=empty");
  const auto exec = internal::checked_execution_state(execution);
  auto result = std::make_shared<CudaNotifyPoolAllocation>();
  result->max_plan = max_plan;
  result->route = std::make_shared<CudaAllocation>();
  result->workspace = std::make_shared<CudaAllocation>();
  result->route->device = exec->device;
  result->workspace->device = exec->device;
  result->route->device_bytes = checked_size(
      checked_mul(max_plan.route_count, sizeof(std::int32_t),
                  "pool route bytes"),
      "pool route bytes");
  result->workspace->device_bytes =
      checked_size(max_plan.device_bytes, "pool workspace device bytes");
  result->workspace->host_bytes = checked_size(
      max_plan.host_staging_bytes, "pool workspace pinned bytes");
  internal::throw_cuda("device-pool-allocate",
                       cudaSetDevice(static_cast<int>(exec->device)),
                       max_plan.calling_rank, max_plan.world_size,
                       execution.token().identity(), 1,
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda(
      "device-pool-allocate",
      cudaMalloc(&result->route->device_ptr,
                 std::max<std::size_t>(result->route->device_bytes, 1)),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaMalloc(pool route slot)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaMalloc(&result->workspace->device_ptr,
                 std::max<std::size_t>(result->workspace->device_bytes, 1)),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaMalloc(pool notify arena)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaHostAlloc(&result->workspace->host_ptr,
                    std::max<std::size_t>(result->workspace->host_bytes, 1),
                    cudaHostAllocPortable | cudaHostAllocMapped),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaHostAlloc(pool status staging)");
  internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaHostGetDevicePointer(&result->workspace->host_device_ptr,
                               result->workspace->host_ptr, 0),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaHostGetDevicePointer(pool status staging)");
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->mapping_start),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool mapping start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->mapping_complete),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool mapping complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreateWithFlags(&result->workspace->completion,
                               cudaEventDisableTiming),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool completion)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->planning_start),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool planning start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->planning_complete),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool planning complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreateWithFlags(&result->route->producer_ready,
                               cudaEventDisableTiming),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool producer ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  result->workspace->consumer_events.reserve(max_inflight_consumers);
  for (std::uint32_t i = 0; i < max_inflight_consumers; ++i) {
    cudaEvent_t event = nullptr;
    internal::throw_cuda(
        "device-pool-allocate",
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
        max_plan.calling_rank, max_plan.world_size,
        execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaEventCreate(pool consumer completion)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    result->workspace->consumer_events.push_back(event);
    result->workspace->consumer_event_states.push_back(
        internal::ConsumerEventSlotState::Available);
  }
  return result;
}

CudaRouteInput DeviceNotifyPoolBridge::upload(
    const RankRoutingInput &routes, std::uint32_t top_k,
    const RankCollectiveContext &context,
    const CudaExecution &execution,
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    std::shared_ptr<const void> slot_lease) {
  if (!allocation || !slot_lease)
    invalid("phase=device-pool-route-upload, field=pool_slot, rank=n/a, "
            "expected=valid, actual=empty");
  const auto count = checked_mul(routes.num_tokens, top_k, "route_count");
  if (routes.selected_experts.size() != checked_size(count, "route_count") ||
      count > allocation->max_plan.route_count)
    invalid("phase=device-pool-route-upload, field=shape, rank=n/a, "
            "expected=valid route count within pool, actual=" +
            std::to_string(routes.selected_experts.size()));
  const auto exec = internal::checked_execution_state(execution);
  const auto bytes = checked_size(
      checked_mul(count, sizeof(std::int32_t), "route bytes"),
      "route bytes");
  if (bytes != 0)
    internal::throw_cuda(
        "device-pool-route-upload",
        cudaMemcpyAsync(allocation->route->device_ptr,
                        routes.selected_experts.data(), bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        allocation->max_plan.calling_rank, allocation->max_plan.world_size,
        execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(host convenience route upload)");
  if (bytes != 0)
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  internal::throw_cuda(
      "device-pool-route-upload",
      cudaEventRecord(allocation->route->producer_ready, exec->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(producer ready)");
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = allocation->route;
  impl->slot_lease = std::move(slot_lease);
  impl->device_ptr = allocation->route->device_ptr;
  impl->producer_session = context.session();
  impl->producer_execution = execution.token();
  impl->consumer_execution = execution.token();
  impl->device = exec->device;
  impl->generation = context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = routes.num_tokens;
  impl->top_k = top_k;
  impl->routes = count;
  return CudaRouteInput(std::move(impl));
}

CudaNotifyWorkspace DeviceNotifyPoolBridge::workspace(
    const CudaNotifyWorkspacePlan &actual_plan,
    const RankCollectiveContext &context,
    const CudaExecution &execution,
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    std::shared_ptr<const void> slot_lease) {
  if (!allocation || !slot_lease || !context || !execution)
    invalid("phase=device-pool-workspace, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto device_layout = make_device_layout(actual_plan);
  const auto host_layout = make_host_layout(actual_plan);
  if (device_layout.total != actual_plan.device_bytes ||
      host_layout.total != actual_plan.host_staging_bytes ||
      device_layout.total > allocation->workspace->device_bytes ||
      host_layout.total > allocation->workspace->host_bytes ||
      actual_plan.route_count > allocation->max_plan.route_count ||
      actual_plan.max_incoming_rows >
          allocation->max_plan.max_incoming_rows)
    invalid("phase=device-pool-workspace, field=capacity, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=actual layout within pool slot, actual=short");
  auto state = std::make_shared<WorkspaceState>();
  state->plan = actual_plan;
  state->device_layout = device_layout;
  state->host_layout = host_layout;
  state->context = context;
  state->epoch = context.session().generation();
  state->execution = execution;
  state->allocation = allocation->workspace;
  state->slot_lease = std::move(slot_lease);
  auto impl = std::make_shared<CudaNotifyWorkspace::Impl>();
  impl->state = std::move(state);
  return CudaNotifyWorkspace(std::move(impl));
}

bool DeviceNotifyPoolBridge::drain(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    const CudaExecution &execution) noexcept {
  if (!allocation || !execution)
    return false;
  try {
    const auto exec = internal::checked_execution_state(execution);
    (void)cudaSetDevice(static_cast<int>(exec->device));
    if (allocation->workspace && allocation->workspace->completion) {
      if (cudaEventRecord(allocation->workspace->completion, exec->stream) !=
          cudaSuccess)
        return false;
      if (cudaEventSynchronize(allocation->workspace->completion) !=
          cudaSuccess)
        return false;
    } else {
      if (cudaStreamSynchronize(exec->stream) != cudaSuccess)
        return false;
    }
    return true;
  } catch (...) {
    // Slot lease destruction is fail-closed and cannot throw. The pool owns
    // the allocation until teardown, whose allocation destructor drains again.
    return false;
  }
}

std::uint64_t DeviceNotifyPoolBridge::device_bytes(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  if (!allocation)
    return 0;
  return static_cast<std::uint64_t>(allocation->route->device_bytes) +
         static_cast<std::uint64_t>(allocation->workspace->device_bytes);
}

std::uint64_t DeviceNotifyPoolBridge::pinned_bytes(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  return allocation
             ? static_cast<std::uint64_t>(allocation->workspace->host_bytes)
             : 0;
}

std::uint64_t DeviceNotifyPoolBridge::event_count(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  if (!allocation)
    return 0;
  return 6U + allocation->workspace->consumer_events.size();
}

bool DeviceNotifyPoolBridge::quarantined(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  if (!allocation || !allocation->workspace)
    return true;
  std::lock_guard<std::mutex> lock(allocation->workspace->consumer_mutex);
  return allocation->workspace->dispatch_slot_quarantined.load(
      std::memory_order_acquire);
}

std::array<std::uint32_t, 3> DeviceNotifyPoolBridge::consumer_event_usage(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  std::array<std::uint32_t, 3> usage{};
  if (!allocation || !allocation->workspace)
    return usage;
  std::lock_guard<std::mutex> lock(allocation->workspace->consumer_mutex);
  for (const auto state : allocation->workspace->consumer_event_states) {
    if (state == internal::ConsumerEventSlotState::Available)
      ++usage[0];
    else if (state == internal::ConsumerEventSlotState::InUse)
      ++usage[1];
    else
      ++usage[2];
  }
  return usage;
}

