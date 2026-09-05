
ActivationTransportPoolUsage ActivationTransportResourcePool::usage() const {
  if (!impl_ || !impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-usage",
                    "pool", 0, "valid", "empty");
  ActivationTransportPoolUsage result;
  auto state = impl_->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (const auto &slot : state->slots) {
    if (slot.state == SlotState::Available)
      ++result.slots_available;
    else if (slot.state == SlotState::InUse)
      ++result.slots_in_use;
    else
      ++result.slots_quarantined;
  }
  result.device_bytes = checked_mul(
      checked_add(
          checked_add(state->layout.total, state->gin_layout.total,
                      "pool compute plus GIN bytes"),
          sizeof(std::uint64_t), "pool slot device bytes"),
      state->slots.size(), "pool device bytes");
#if MEGAMOE_DEVICE_TIMELINE
  result.device_bytes = checked_add(
      result.device_bytes,
      checked_mul(state->timeline_layout.total, state->slots.size(),
                  "pool timeline bytes"),
      "pool device plus timeline bytes");
#endif
  result.compute_device_bytes = checked_mul(
      state->layout.total, state->slots.size(), "pool compute device bytes");
  result.gin_registered_bytes = checked_mul(
      state->gin_layout.total, state->slots.size(),
      "pool GIN registered bytes");
  const auto progress_bytes = checked_mul(
      state->slots.empty() ? 0ULL
                           : state->slots.front().forward_progress_words,
      sizeof(std::uint64_t), "pool mapped progress bytes");
  const auto pinned_per_slot = checked_add(
      checked_add(2ULL * sizeof(TransportDeviceStatus),
                  sizeof(std::uint64_t), "pool pinned status/cancel bytes"),
      progress_bytes, "pool pinned bytes per slot");
  const auto socket_count_bytes = state->config.enable_socket_fallback
      ? checked_mul(checked_mul(state->config.max_world_size,
                                state->config.max_world_size,
                                "Socket host count matrix"),
                    sizeof(std::uint64_t),
                    "Socket host count matrix bytes")
      : 0ULL;
  result.pinned_bytes = checked_mul(
      checked_add(pinned_per_slot, socket_count_bytes,
                  "pool pinned plus Socket bytes"),
      state->slots.size(),
                                    "pool pinned bytes");
  const auto events_per_slot =
      11ULL + (state->config.enable_backward_state ? 8ULL : 0ULL);
  result.event_count =
      checked_mul(events_per_slot, state->slots.size(), "pool events");
  result.epoch_high_watermark = state->epoch_high;
  return result;
}

std::uint64_t BFloat16ActivationLease::num_tokens() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->tokens;
}
std::uint64_t BFloat16ActivationLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t BFloat16ActivationLease::element_count() const {
  return checked_mul(num_tokens(), hidden_size(), "activation elements");
}
std::uint64_t BFloat16ActivationLease::generation() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->generation;
}
std::uint32_t BFloat16ActivationLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->device;
}
std::uint32_t BFloat16ActivationLease::layout_version() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->layout;
}

std::uint64_t BFloat16OutputGradientLease::num_tokens() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "output-gradient-lease",
                    "lease", 0, "valid", "empty");
  return impl_->tokens;
}

std::uint64_t BFloat16OutputGradientLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "output-gradient-lease",
                    "lease", 0, "valid", "empty");
  return impl_->hidden;
}

std::uint64_t BFloat16OutputGradientLease::element_count() const {
  return checked_mul(num_tokens(), hidden_size(), "output gradient elements");
}

std::uint64_t BFloat16OutputGradientLease::generation() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "output-gradient-lease",
                    "lease", 0, "valid", "empty");
  return impl_->generation;
}

std::uint32_t BFloat16OutputGradientLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "output-gradient-lease",
                    "lease", 0, "valid", "empty");
  return impl_->device;
}

BFloat16OutputGradientLease BackwardValidationBridge::upload_for_testing(
    const std::vector<transport::BFloat16> &host,
    std::uint64_t tokens, std::uint64_t hidden,
    const RankCollectiveContext &context, const CudaExecution &execution) {
  if (!context || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "output-gradient-test-upload", "context/execution", 0,
                    "valid", "empty");
  const auto &identity = context.session();
  if (tokens == 0 || hidden == 0 ||
      host.size() != checked_size(checked_mul(tokens, hidden,
                                             "output gradient elements"),
                                  "output gradient elements"))
    transport_error(NotifyErrorCode::InvalidRouteShape,
                    "output-gradient-test-upload", "host shape",
                    context.calling_rank(), "exact nonempty [tokens, hidden]",
                    std::to_string(host.size()));
  const auto bytes = checked_size(
      checked_mul(host.size(), sizeof(transport::BFloat16),
                  "output gradient upload bytes"),
      "output gradient upload bytes");
  auto exec = internal::checked_execution_state(execution);
  auto allocation = std::make_shared<BackwardGradientAllocation>();
  allocation->device = exec->device;
  internal::throw_cuda(
      "output-gradient-test-upload", cudaMalloc(&allocation->data, bytes),
      context.calling_rank(), identity.world_size(), identity.identity(),
      identity.generation(), ExchangeKind::ExpertCounts,
      "cudaMalloc(output gradient)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda(
      "output-gradient-test-upload",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      context.calling_rank(), identity.world_size(), identity.identity(),
      identity.generation(), ExchangeKind::ExpertCounts,
      "cudaEventCreate(output gradient ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "output-gradient-test-upload",
      cudaMemcpyAsync(allocation->data, host.data(), bytes,
                      cudaMemcpyHostToDevice, exec->stream),
      context.calling_rank(), identity.world_size(), identity.identity(),
      identity.generation(), ExchangeKind::ExpertCounts,
      "cudaMemcpyAsync(output gradient)");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  internal::throw_cuda(
      "output-gradient-test-upload",
      cudaEventRecord(allocation->ready, exec->stream), context.calling_rank(),
      identity.world_size(), identity.identity(), identity.generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(output gradient ready)");
  auto impl = std::make_shared<BFloat16OutputGradientLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->pointer = static_cast<const transport::BFloat16 *>(impl->allocation->data);
  impl->session_identity = identity.stable_identity_bytes();
  impl->producer_execution = execution.token();
  impl->generation = identity.generation();
  impl->tokens = tokens;
  impl->hidden = hidden;
  impl->device = exec->device;
  return BFloat16OutputGradientLease(std::move(impl));
}

BFloat16OutputGradientLease BackwardValidationBridge::reissue_for_benchmark(
    const BFloat16OutputGradientLease &storage,
    const RankCollectiveContext &context, const CudaExecution &execution) {
  if (!storage.impl_ || !context || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "output-gradient-benchmark-reissue", "input", 0,
                    "valid", "empty");
  const auto &source = *storage.impl_;
  if (!source.allocation || !source.allocation->ready ||
      source.session_identity != context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() ||
      !source.producer_execution.same_instance(execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "output-gradient-benchmark-reissue", "provenance",
                    context.calling_rank(),
                    "same storage/session/execution/device/ready", "mismatch");
  auto impl = std::make_shared<BFloat16OutputGradientLease::Impl>(source);
  impl->generation = context.session().generation();
  return BFloat16OutputGradientLease(std::move(impl));
}

BFloat16ActivationLease ActivationTransportBridge::upload_for_testing(
    const std::vector<transport::BFloat16> &host, std::uint64_t num_tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!context || !execution || !pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "activation-test-upload", "context/execution/pool", 0,
                    "valid", "empty");
  auto state = pool.impl_->state;
  if (!context.execution().same_instance(execution.token()) ||
      context.session().stable_identity_bytes() != state->session_identity ||
      execution.device_ordinal() != state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "activation-test-upload", "provenance",
                    context.calling_rank(), "exact pool session/execution/device",
                    "mismatch");
  if (num_tokens > state->config.max_tokens_per_rank ||
      hidden == 0 || hidden > state->config.max_hidden_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "activation-test-upload", "shape", context.calling_rank(),
                    "within pool capacities", "exceeded/zero");
  const auto count = checked_mul(num_tokens, hidden, "activation upload count");
  if (host.size() != checked_size(count, "activation upload count"))
    transport_error(NotifyErrorCode::InvalidRouteShape,
                    "activation-test-upload", "host.size",
                    context.calling_rank(), std::to_string(count),
                    std::to_string(host.size()));
  auto lease = acquire_slot(state, context.session().generation(),
                            context.calling_rank());
  auto &slot = state->slots[lease->index];
  auto exec = internal::checked_execution_state(execution);
  if (count != 0) {
    const auto bytes = checked_size(checked_mul(count, sizeof(transport::BFloat16),
                                                "activation upload bytes"),
                                    "activation upload bytes");
    internal::throw_cuda(
        "activation-test-upload",
        cudaMemcpyAsync(at<void>(slot.device, state->layout.input), host.data(),
                        bytes, cudaMemcpyHostToDevice, exec->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemcpyAsync(test activation)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  }
  internal::throw_cuda(
      "activation-test-upload", cudaEventRecord(slot.producer_ready, exec->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(producer-ready)");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>();
  impl->slot = std::move(lease);
  impl->pool = state;
  impl->pointer = at<const transport::BFloat16>(slot.device, state->layout.input);
  impl->session_identity = context.session().stable_identity_bytes();
  impl->producer_execution = execution.token();
  impl->generation = context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden;
  impl->device = execution.device_ordinal();
  impl->ready_recorded = true;
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16ActivationLease ActivationTransportBridge::reissue_for_benchmark(
    const BFloat16ActivationLease &storage,
    const RankCollectiveContext &context, const CudaExecution &execution) {
  if (!storage.impl_ || !context || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "activation-benchmark-reissue", "input", 0, "valid",
                    "empty");
  const auto &source = *storage.impl_;
  if (!context.execution().same_instance(execution.token()) ||
      !source.producer_execution.same_instance(execution.token()) ||
      source.session_identity != context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      !source.slot || !source.pool)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "activation-benchmark-reissue", "provenance",
                    context.calling_rank(),
                    "same immutable storage/session/execution/device/ready",
                    "mismatch");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>(source);
  impl->generation = context.session().generation();
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16ActivationLease testing::upload_bfloat16_activations_for_testing(
    const std::vector<transport::BFloat16> &host, std::uint64_t num_tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  return ActivationTransportBridge::upload_for_testing(
      host, num_tokens, hidden, context, execution, pool);
}

BFloat16ActivationLease ExternalCudaProducer::publish_activations(
    const transport::BFloat16 *device_data,
    std::uint64_t capacity_elements, std::uint64_t num_tokens,
    std::uint64_t hidden_size, const ActivationTransportResourcePool &pool,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !pool.impl_ || !pool.impl_->state || !allocation_owner)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-activation-publish", "producer/pool/owner", 0,
                    "valid", "empty");
  auto pool_state = pool.impl_->state;
  if (state_->backend->context.session().stable_identity_bytes() !=
          pool_state->session_identity ||
      state_->backend->device != pool_state->device || hidden_size == 0 ||
      hidden_size > pool_state->config.max_hidden_size ||
      num_tokens > pool_state->config.max_tokens_per_rank)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "external-activation-publish", "provenance/shape", 0,
                    "matching pool session/device and bounded shape",
                    "mismatch");
  const auto elements = checked_mul(num_tokens, hidden_size,
                                    "external activation elements");
  validate_external_cuda_range(device_data, elements, capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "activations");
  auto lease = acquire_slot(pool_state,
                            state_->backend->context.session().generation(),
                            state_->backend->context.calling_rank());
  auto &slot = pool_state->slots[lease->index];
  internal::throw_cuda(
      "external-activation-publish",
      cudaEventRecord(slot.producer_ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(activation ready)");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>();
  impl->slot = std::move(lease);
  impl->pool = std::move(pool_state);
  impl->pointer = device_data;
  impl->session_identity = state_->backend->context.session().stable_identity_bytes();
  impl->producer_execution = state_->backend->execution;
  impl->generation = state_->backend->context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden_size;
  impl->device = state_->backend->device;
  impl->ready_recorded = true;
  impl->external_owner = std::move(allocation_owner);
  impl->external_producer = state_->backend;
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16OutputGradientLease ExternalCudaProducer::publish_output_gradients(
    const transport::BFloat16 *device_data,
    std::uint64_t capacity_elements, std::uint64_t num_tokens,
    std::uint64_t hidden_size,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !allocation_owner || num_tokens == 0 || hidden_size == 0)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-output-gradient-publish",
                    "producer/owner/shape", 0, "valid and nonzero", "empty");
  const auto elements = checked_mul(num_tokens, hidden_size,
                                    "external output gradient elements");
  validate_external_cuda_range(device_data, elements, capacity_elements,
                               sizeof(transport::BFloat16),
                               state_->backend->device, "output gradients");
  auto allocation = std::make_shared<BackwardGradientAllocation>();
  allocation->device = state_->backend->device;
  allocation->data = const_cast<transport::BFloat16 *>(device_data);
  allocation->owns_data = false;
  allocation->external_owner = std::move(allocation_owner);
  allocation->external_producer = state_->backend;
  internal::throw_cuda(
      "external-output-gradient-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts,
      "cudaEventCreate(output gradient ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "external-output-gradient-publish",
      cudaEventRecord(allocation->ready, state_->backend->stream),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(output gradient ready)");
  auto impl = std::make_shared<BFloat16OutputGradientLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->pointer = device_data;
  impl->session_identity =
      state_->backend->context.session().stable_identity_bytes();
  impl->producer_execution = state_->backend->execution;
  impl->generation = state_->backend->context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden_size;
  impl->device = state_->backend->device;
  return BFloat16OutputGradientLease(std::move(impl));
}

DeviceActivationTransport::DeviceActivationTransport(
    const CollectiveSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-create",
                    "session/execution/pool", 0, "valid", "empty");
  if (session.stable_identity_bytes() != pool.impl_->state->session_identity ||
      session.world_size() != pool.impl_->state->world ||
      execution.device_ordinal() != pool.impl_->state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-create",
                    "pool provenance", 0, "matching session/world/device",
                    "mismatch");
  impl_ = std::make_shared<Impl>();
  impl_->session = session;
  impl_->execution = execution;
  impl_->pool = pool;
}

LocalCudaP2pActivationTransport::LocalCudaP2pActivationTransport(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create",
                    "pool", 0, "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend =
      create_local_p2p_backend(session, execution, pool.impl_->state);
}

NcclSocketFallbackActivationTransport::
    NcclSocketFallbackActivationTransport(
        const NcclSession &session, const CudaExecution &execution,
        const ActivationTransportResourcePool &pool) {
  if (!session || !execution || !pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "socket-create",
                    "session/execution/pool", 0, "valid", "empty");
  auto state = pool.impl_->state;
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (!state->config.enable_socket_fallback)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-create",
                    "pool.enable_socket_fallback", session.rank(), "true",
                    "false");
  const auto *ib_disabled = std::getenv("NCCL_IB_DISABLE");
  if (ib_disabled == nullptr || std::string(ib_disabled) != "1")
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=socket-create, field=NCCL_IB_DISABLE, rank=" +
            std::to_string(session.rank()) +
            ", expected=1 before communicator creation, actual=" +
            (ib_disabled ? std::string(ib_disabled) : "unset") +
            "; NcclSocketFallback refuses an unaudited network selection");
  if (session.session().stable_identity_bytes() != state->session_identity ||
      session.session().world_size() != state->world ||
      session.device_ordinal() != state->device ||
      execution.device_ordinal() != state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "socket-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  impl_ = std::make_shared<Impl>();
  impl_->backend = std::make_shared<SocketBackendState>();
  impl_->backend->pool = std::move(state);
  impl_->backend->nccl = nccl;
  impl_->backend->session = session;
  impl_->backend->execution = execution;
}

GinRdmaActivationTransport::GinRdmaActivationTransport(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-create", "pool", 0,
                    "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend = create_gin_backend(session, execution, pool.impl_->state);
}

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (pool.impl_ && pool.impl_->state &&
      pool.impl_->state->config.enable_gin_rdma)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-create",
                    "backend", session.rank(), "LocalCudaP2p",
                    "GIN-enabled pool requires GinRdmaForwardMegaKernel");
  *this = LocalCudaP2pForwardMegaKernel(session, execution, pool, false);
}

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool, bool use_gin) {
  *this = LocalCudaP2pForwardMegaKernel(session, execution, pool, use_gin,
                                        false);
}

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool, bool use_gin,
    bool use_socket) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-create", "pool",
                    0, "valid", "empty");
  if (use_gin && use_socket)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-create",
                    "backend", session.rank(), "one backend",
                    "GIN and Socket both requested");
  impl_ = std::make_shared<Impl>();
  if (use_gin) {
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_gin_backend(session, execution, pool.impl_->state));
  } else if (use_socket) {
    auto socket = std::make_shared<SocketBackendState>();
    socket->pool = pool.impl_->state;
    socket->nccl = internal::NcclSessionAccess::checked(session);
    socket->session = session;
    socket->execution = execution;
    impl_->backend = socket;
    impl_->socket_transport =
        std::make_shared<NcclSocketFallbackActivationTransport>(
            session, execution, pool);
  } else {
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_local_p2p_backend(session, execution, pool.impl_->state));
  }
}

GinRdmaForwardMegaKernel::GinRdmaForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pForwardMegaKernel>(
          new LocalCudaP2pForwardMegaKernel(session, execution, pool, true))) {}

DeviceForwardHandle GinRdmaForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(dispatch, activations, weights, config, options);
}

NcclSocketFallbackForwardMegaKernel::NcclSocketFallbackForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pForwardMegaKernel>(
          new LocalCudaP2pForwardMegaKernel(session, execution, pool, false,
                                            true))) {}

DeviceForwardHandle NcclSocketFallbackForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(dispatch, activations, weights, config, options);
}

DeviceActivationTransportHandle
