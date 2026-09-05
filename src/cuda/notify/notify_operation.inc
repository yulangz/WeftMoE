struct CudaLocalNotifyHandle::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  std::uint64_t num_tokens = 0;
  DeviceDispatchMetadata device;
  DeviceBufferLease expert_count;
  DeviceBufferLease packet_count;
  CudaRouteInput routes;
  CudaGateInput gates;
  std::shared_ptr<WorkspaceState> workspace;
};

std::uint64_t CudaLocalNotifyHandle::num_tokens() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->num_tokens;
}
const RuntimeConfig &CudaLocalNotifyHandle::config() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->config;
}
const RankCollectiveContext &CudaLocalNotifyHandle::context() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->context;
}
const DeviceBufferLease &
CudaLocalNotifyHandle::device_expert_count_for_backend() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local expert count");
  return impl_->expert_count;
}
const DeviceBufferLease &
CudaLocalNotifyHandle::device_packet_count_for_backend() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local packet count");
  return impl_->packet_count;
}

struct CudaNotifyPreflight::Impl {
  LocalPreflightStatus status;
  CudaLocalNotifyHandle local;
  double device_mapping_us = 0;
};

const LocalPreflightStatus &CudaNotifyPreflight::status() const {
  if (!impl_)
    invalid("phase=cuda-preflight, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->status;
}

double CudaNotifyPreflight::device_mapping_us() const {
  if (!impl_)
    invalid("phase=cuda-preflight, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->device_mapping_us;
}

bool CudaNotifyPreflight::valid() const {
  return impl_ && impl_->status.error == ControlLocalError::None &&
         static_cast<bool>(impl_->local);
}

const CudaLocalNotifyHandle &CudaNotifyPreflight::local() const {
  if (!valid())
    invalid("phase=cuda-preflight, field=local, rank=n/a, expected=successful preflight, actual=failed/empty");
  return impl_->local;
}

struct CudaNotifyOperation::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  CudaExecution execution;
  CudaRouteInput routes;
  CudaGateInput gates;
  std::shared_ptr<WorkspaceState> workspace;
  bool terminal = false;
  bool waited = false;
};

CudaNotifyOperation start_cuda_rank_local_notify(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    const CudaGateInput &gates, CudaNotifyWorkspace workspace) {
  validate_runtime_config(config);
  if (!context || !execution || !routes || !gates || !workspace)
    invalid("phase=cuda-notify-start, field=input, rank=n/a, expected=all handles valid, actual=empty");
  const auto rank = context.calling_rank();
  if (context.session().world_size() != config.topology.world_size ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=cuda-notify-start, field=context, rank=" +
            std::to_string(rank) +
            ", expected=config world and exact execution, actual=mismatch");
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  DeviceGateProvenanceBridge::validate(gates, routes, context, execution);
  const auto expected = query_cuda_notify_workspace(
      config, context.calling_rank(), routes.num_tokens(),
      workspace.plan().max_tokens_per_rank);
  const auto &actual = workspace.plan();
  if (expected.calling_rank != actual.calling_rank ||
      expected.num_tokens != actual.num_tokens ||
      expected.max_tokens_per_rank != actual.max_tokens_per_rank ||
      expected.top_k != actual.top_k ||
      expected.num_experts != actual.num_experts ||
      expected.world_size != actual.world_size ||
      expected.num_nodes != actual.num_nodes ||
      expected.node_mask_words != actual.node_mask_words ||
      expected.route_count != actual.route_count ||
      expected.max_incoming_rows != actual.max_incoming_rows ||
      expected.device_bytes != actual.device_bytes ||
      expected.host_staging_bytes != actual.host_staging_bytes ||
      expected.sort_temp_bytes != actual.sort_temp_bytes ||
      expected.scan_temp_bytes != actual.scan_temp_bytes)
    invalid("phase=cuda-notify-start, field=workspace.plan, rank=" +
            std::to_string(rank) + ", expected=exact queried plan, actual=mismatch");
  if (!workspace.impl_->state->context.same_instance(context))
    invalid("phase=cuda-notify-start, field=workspace.context, rank=" +
            std::to_string(rank) + ", expected=exact context, actual=different");
  auto state = workspace.impl_->state;
  auto exec = internal::checked_execution_state(execution);
  const auto &dl = state->device_layout;
  void *base = state->allocation->device_ptr;
  internal::throw_cuda("notify-start", cudaSetDevice(static_cast<int>(exec->device)),
                       rank, context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.expert_to_rank),
                                       config.topology.expert_to_rank.data(),
                                       config.num_experts * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(expert_to_rank)");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::HostToDevice,
      static_cast<std::uint64_t>(config.num_experts) * sizeof(std::uint32_t));
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.rank_to_node),
                                       config.topology.rank_to_node.data(),
                                       config.topology.world_size * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(rank_to_node)");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::HostToDevice,
      static_cast<std::uint64_t>(config.topology.world_size) *
          sizeof(std::uint32_t));
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.rank_to_local),
                                       config.topology.rank_to_local_rank.data(),
                                       config.topology.world_size * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(rank_to_local)");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::HostToDevice,
      static_cast<std::uint64_t>(config.topology.world_size) *
          sizeof(std::uint32_t));
  internal::throw_cuda("notify-start", cudaMemsetAsync(at<void>(base, dl.expert_count), 0,
                                                        config.num_experts * sizeof(std::uint64_t), exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemsetAsync(expert_count)");
  internal::throw_cuda("notify-start", cudaMemsetAsync(at<void>(base, dl.packet_count), 0,
                                                        config.topology.num_nodes * sizeof(std::uint64_t), exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::NotifyPacketSummary,
                       "cudaMemsetAsync(packet_count)");
  internal::throw_cuda("notify-start", cudaMemsetAsync(at<void>(base, dl.status), 0xff,
                                                        sizeof(DeviceStatus), exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemsetAsync(status)");
  const auto route_count = state->plan.route_count;
  internal::throw_cuda(
      "notify-start",
      cudaEventRecord(state->allocation->mapping_start, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(mapping start)");
  internal::throw_cuda(
      "notify-start",
      cudaMemsetAsync(
          at<void>(base, dl.local_offset), 0,
          checked_size(checked_mul(route_count, sizeof(std::uint64_t),
                                   "local_offset_bytes"),
                       "local_offset_bytes"),
          exec->stream),
      rank, context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(local_offset)");
  const auto route_blocks = static_cast<unsigned>((route_count + 255U) / 256U);
  const auto token_blocks = static_cast<unsigned>((state->plan.num_tokens + 255U) / 256U);
  if (route_count != 0) {
    prepare_routes_kernel<<<route_blocks, 256, 0, exec->stream>>>(
        static_cast<const std::int32_t *>(routes.impl_->device_ptr),
        route_count, route_count, config.num_experts,
        at<const std::uint32_t>(base, dl.expert_to_rank),
        at<const std::uint32_t>(base, dl.rank_to_node),
        at<const std::uint32_t>(base, dl.rank_to_local),
        at<std::uint64_t>(base, dl.keys_in),
        at<std::uint64_t>(base, dl.indices_in),
        at<std::uint32_t>(base, dl.expert_id),
        at<std::uint32_t>(base, dl.route_dst_rank),
        at<std::uint32_t>(base, dl.route_dst_node),
        at<std::uint32_t>(base, dl.route_dst_local_rank),
        at<std::uint64_t>(base, dl.ones), at<DeviceStatus>(base, dl.status));
    validate_and_describe_tokens_kernel<<<token_blocks, 256, 0, exec->stream>>>(
        static_cast<const std::int32_t *>(routes.impl_->device_ptr),
        state->plan.num_tokens, config.top_k, config.num_experts,
        at<const std::uint32_t>(base, dl.expert_to_rank),
        at<const std::uint32_t>(base, dl.rank_to_node), config.topology.num_nodes,
        state->plan.node_mask_words, at<std::uint32_t>(base, dl.combine_order),
        at<std::uint64_t>(base, dl.masks),
        at<std::uint64_t>(base, dl.packet_count),
        at<DeviceStatus>(base, dl.status));
    internal::throw_cuda("notify-start", cudaGetLastError(), rank,
                         context.session().world_size(), context.session().identity(),
                         context.session().generation(), ExchangeKind::ExpertCounts,
                         "route kernels launch");
    std::size_t sort_bytes = checked_size(state->plan.sort_temp_bytes, "sort_temp_bytes");
    internal::throw_cuda(
        "notify-start",
        cub::DeviceRadixSort::SortPairs(
            at<void>(base, dl.temp), sort_bytes,
            at<std::uint64_t>(base, dl.keys_in),
            at<std::uint64_t>(base, dl.keys_out),
            at<std::uint64_t>(base, dl.indices_in),
            at<std::uint64_t>(base, dl.indices_out),
            static_cast<int>(route_count), 0, 64, exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cub::DeviceRadixSort::SortPairs");
    extract_sorted_experts_kernel<<<route_blocks, 256, 0, exec->stream>>>(
        at<const std::uint64_t>(base, dl.keys_out),
        at<std::uint32_t>(base, dl.sorted_experts), route_count, route_count);
    std::size_t scan_bytes = checked_size(state->plan.scan_temp_bytes, "scan_temp_bytes");
    internal::throw_cuda(
        "notify-start",
        cub::DeviceScan::InclusiveSumByKey(
            at<void>(base, dl.temp), scan_bytes,
            at<std::uint32_t>(base, dl.sorted_experts),
            at<std::uint64_t>(base, dl.ones),
            at<std::uint64_t>(base, dl.inclusive),
            static_cast<int>(route_count), std::equal_to<std::uint32_t>(),
            exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cub::DeviceScan::InclusiveSumByKey");
    histogram_and_scatter_kernel<<<route_blocks, 256, 0, exec->stream>>>(
        at<const std::uint32_t>(base, dl.sorted_experts),
        at<const std::uint64_t>(base, dl.indices_out),
        at<const std::uint64_t>(base, dl.inclusive), route_count,
        config.num_experts, at<std::uint64_t>(base, dl.expert_count),
        at<std::uint64_t>(base, dl.local_offset));
    internal::throw_cuda("notify-start", cudaGetLastError(), rank,
                         context.session().world_size(), context.session().identity(),
                         context.session().generation(), ExchangeKind::ExpertCounts,
                         "mapping kernels launch");
  }
  internal::throw_cuda(
      "notify-start",
      cudaEventRecord(state->allocation->mapping_complete, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(mapping complete)");
  const auto &hl = state->host_layout;
  void *host = state->allocation->host_ptr;
  const auto copy = [&](std::size_t host_offset, std::size_t device_offset,
                        std::size_t bytes, const char *name) {
    if (bytes == 0)
      return;
    internal::throw_cuda("notify-start",
                         cudaMemcpyAsync(at<void>(host, host_offset),
                                         at<void>(base, device_offset), bytes,
                                         cudaMemcpyDeviceToHost, exec->stream),
                         rank, context.session().world_size(),
                         context.session().identity(),
                         context.session().generation(),
                         ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              bytes);
  };
  copy(hl.status, dl.status, sizeof(DeviceStatus), "D2H status");
  internal::throw_cuda("notify-start",
                       cudaEventRecord(state->allocation->completion, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaEventRecord");
  auto impl = std::make_shared<CudaNotifyOperation::Impl>();
  impl->config = config;
  impl->context = context;
  impl->execution = execution;
  impl->routes = routes;
  impl->gates = gates;
  impl->workspace = std::move(state);
  return CudaNotifyOperation(std::move(impl));
}

CudaNotifyOperation::CudaNotifyOperation() = default;
CudaNotifyOperation::CudaNotifyOperation(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaNotifyOperation::~CudaNotifyOperation() noexcept {
  if (impl_ && !impl_->terminal && impl_->workspace) {
    (void)cudaSetDevice(static_cast<int>(impl_->execution.device_ordinal()));
    (void)cudaEventSynchronize(impl_->workspace->allocation->completion);
    impl_->terminal = true;
  }
}
CudaNotifyOperation::CudaNotifyOperation(CudaNotifyOperation &&) noexcept = default;
CudaNotifyOperation &CudaNotifyOperation::operator=(CudaNotifyOperation &&other) noexcept {
  if (this == &other)
    return *this;
  if (impl_ && !impl_->terminal && impl_->workspace) {
    (void)cudaSetDevice(static_cast<int>(impl_->execution.device_ordinal()));
    (void)cudaEventSynchronize(impl_->workspace->allocation->completion);
  }
  impl_ = std::move(other.impl_);
  return *this;
}

CudaNotifyPreflight CudaNotifyOperation::wait_preflight() {
  if (!impl_)
    invalid("phase=cuda-notify-wait, field=operation, rank=n/a, expected=valid, actual=empty");
  if (impl_->waited || impl_->terminal)
    invalid("phase=cuda-notify-wait, field=state, rank=" +
            std::to_string(impl_->context.calling_rank()) +
            ", expected=pending, actual=terminal");
  const auto rank = impl_->context.calling_rank();
  internal::throw_cuda("notify-wait",
                       cudaEventSynchronize(impl_->workspace->allocation->completion),
                       rank, impl_->context.session().world_size(),
                       impl_->context.session().identity(),
                       impl_->context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventSynchronize");
  impl_->terminal = true;
  impl_->waited = true;
  const auto &state = impl_->workspace;
  const auto *status = at<const DeviceStatus>(state->allocation->host_ptr,
                                              state->host_layout.status);
  auto preflight_impl = std::make_shared<CudaNotifyPreflight::Impl>();
  float mapping_ms = 0;
  internal::throw_cuda(
      "notify-wait",
      cudaEventElapsedTime(&mapping_ms,
                           impl_->workspace->allocation->mapping_start,
                           impl_->workspace->allocation->mapping_complete),
      rank, impl_->context.session().world_size(),
      impl_->context.session().identity(), impl_->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(mapping)");
  preflight_impl->device_mapping_us =
      static_cast<double>(mapping_ms) * 1000.0;
  const auto unpack_status = [&](unsigned long long packed,
                                 ControlLocalError error,
                                 std::uint64_t expected,
                                 std::uint64_t actual) {
    const auto flat = static_cast<std::uint64_t>(packed >> 32U);
    const auto expert_bits = static_cast<std::uint32_t>(packed);
    std::int32_t raw_expert = 0;
    std::memcpy(&raw_expert, &expert_bits, sizeof(raw_expert));
    preflight_impl->status = LocalPreflightStatus{
        error, ControlField::LocalValidation, flat / impl_->config.top_k,
        flat % impl_->config.top_k, raw_expert, expected, actual};
  };
  if (status->out_of_range_packed != ~0ULL) {
    const auto expert_bits =
        static_cast<std::uint32_t>(status->out_of_range_packed);
    unpack_status(status->out_of_range_packed,
                  ControlLocalError::ExpertOutOfRange,
                  impl_->config.num_experts, expert_bits);
    return CudaNotifyPreflight(std::move(preflight_impl));
  }
  if (status->duplicate_packed != ~0ULL) {
    unpack_status(status->duplicate_packed, ControlLocalError::DuplicateExpert,
                  0, 1);
    return CudaNotifyPreflight(std::move(preflight_impl));
  }
  auto result = std::make_shared<CudaLocalNotifyHandle::Impl>();
  result->config = impl_->config;
  result->context = impl_->context;
  result->num_tokens = state->plan.num_tokens;
  result->routes = impl_->routes;
  result->gates = impl_->gates;
  result->workspace = state;
  const auto route_width = state->plan.route_count;
  result->device.expert_id = make_lease(state, state->device_layout.expert_id, route_width);
  result->device.local_offset = make_lease(state, state->device_layout.local_offset, route_width);
  result->device.route_dst_row = make_lease(state, state->device_layout.route_dst_row, route_width);
  result->device.route_dst_mtile =
      make_lease(state, state->device_layout.route_dst_mtile, route_width);
  result->device.route_dst_rank = make_lease(state, state->device_layout.route_dst_rank, route_width);
  result->device.route_dst_node = make_lease(state, state->device_layout.route_dst_node, route_width);
  result->device.route_dst_local_rank = make_lease(state, state->device_layout.route_dst_local_rank, route_width);
  result->device.combine_order = make_lease(state, state->device_layout.combine_order, route_width);
  result->device.destination_node_mask = make_lease(
      state, state->device_layout.masks,
      checked_mul(state->plan.num_tokens, state->plan.node_mask_words, "mask lease"));
  result->expert_count = make_lease(
      state, state->device_layout.expert_count, state->plan.num_experts);
  result->packet_count = make_lease(
      state, state->device_layout.packet_count, state->plan.num_nodes);
  preflight_impl->local = CudaLocalNotifyHandle(std::move(result));
  return CudaNotifyPreflight(std::move(preflight_impl));
}

CudaLocalNotifyHandle CudaNotifyOperation::wait() {
  auto preflight = wait_preflight();
  if (preflight.valid())
    return preflight.local();
  const auto &status = preflight.status();
  const auto rank = impl_->context.calling_rank();
  const auto flat = status.token * impl_->config.top_k + status.slot;
  if (status.error == ControlLocalError::ExpertOutOfRange) {
    throw NotifyDispatchError(
        NotifyErrorCode::ExpertOutOfRange,
        "phase=cuda-device-validation, field=selected_experts, rank=" +
            std::to_string(rank) + ", expected=[0," +
            std::to_string(impl_->config.num_experts) + "), actual=" +
            std::to_string(status.expert) + ", " +
            route_coordinates(flat, impl_->config.top_k));
  }
  throw NotifyDispatchError(
      NotifyErrorCode::DuplicateExpert,
      "phase=cuda-device-validation, field=selected_experts, rank=" +
          std::to_string(rank) +
          ", expected=unique within token, actual=duplicate expert=" +
          std::to_string(status.expert) + ", " +
          route_coordinates(flat, impl_->config.top_k));
}

void CudaNotifyOperation::cancel() {
  if (!impl_ || impl_->terminal)
    return;
  const auto rank = impl_->context.calling_rank();
  internal::throw_cuda("notify-cancel",
                       cudaEventSynchronize(impl_->workspace->allocation->completion),
                       rank, impl_->context.session().world_size(),
                       impl_->context.session().identity(),
                       impl_->context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventSynchronize");
  impl_->terminal = true;
}


