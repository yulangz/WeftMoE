
struct CudaNotifyPoolAllocation {
  CudaNotifyWorkspacePlan max_plan;
  std::shared_ptr<internal::CudaAllocation> route;
  std::shared_ptr<internal::CudaAllocation> workspace;
  ~CudaNotifyPoolAllocation() noexcept {
    if (!workspace || !workspace->dispatch_slot_quarantined.load(
                          std::memory_order_acquire))
      return;
    (void)cudaSetDevice(static_cast<int>(workspace->device));
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceSynchronize);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      // A broken context cannot prove that consumer work ended. Fail closed by
      // abandoning native resources to the process instead of freeing storage
      // that a device may still reference.
      if (route)
        route->abandon_on_teardown = true;
      workspace->abandon_on_teardown = true;
    }
  }
};

namespace internal {

std::unique_ptr<BackendCountExchangeOperation>
CudaDeviceLeaseAccess::start(const CountExchangeRequest &) {
  invalid("phase=cuda-device-lease-access, field=start, rank=n/a, "
          "expected=never called, actual=called");
}

std::shared_ptr<const CudaBufferState>
CudaDeviceLeaseAccess::checked(const DeviceBufferLease &lease,
                               std::uint32_t expected_device) {
  if (!lease)
    invalid("phase=cuda-device-lease-access, field=lease, rank=n/a, "
            "expected=valid, actual=empty");
  auto state = std::dynamic_pointer_cast<const CudaBufferState>(
      backend_device_state(lease));
  if (!state || !state->pointer || state->device != expected_device)
    invalid("phase=cuda-device-lease-access, field=backend, rank=n/a, "
            "expected=CUDA lease on execution device, actual=foreign/null");
  return state;
}

CudaExecutionState::~CudaExecutionState() noexcept {
  if (!stream)
    return;
  (void)cudaSetDevice(static_cast<int>(device));
  (void)cudaStreamSynchronize(stream);
  (void)cudaStreamDestroy(stream);
}

CudaAllocation::~CudaAllocation() noexcept {
  if (abandon_on_teardown)
    return;
  (void)cudaSetDevice(static_cast<int>(device));
  if (completion)
    (void)cudaEventSynchronize(completion);
  if (mapping_start)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (mapping_start)
    (void)cudaEventDestroy(mapping_start);
  if (mapping_complete)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (mapping_complete)
    (void)cudaEventDestroy(mapping_complete);
  if (planning_start)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (planning_start)
    (void)cudaEventDestroy(planning_start);
  if (planning_complete)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (planning_complete)
    (void)cudaEventDestroy(planning_complete);
  if (producer_ready && owns_producer_event)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (producer_ready && owns_producer_event)
    (void)cudaEventDestroy(producer_ready);
  for (auto event : consumer_events) {
    if (event) {
      record_cuda_api(CudaApiAuditKind::EventDestroy);
      (void)cudaEventDestroy(event);
    }
  }
  if (completion)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (completion)
    (void)cudaEventDestroy(completion);
  if (host_ptr)
    record_cuda_api(CudaApiAuditKind::PinnedFree);
  if (host_ptr)
    (void)cudaFreeHost(host_ptr);
  if (device_ptr && owns_device_memory)
    record_cuda_api(CudaApiAuditKind::DeviceFree);
  if (device_ptr && owns_device_memory)
    (void)cudaFree(device_ptr);
}

void throw_cuda(const char *phase, cudaError_t error, std::uint32_t rank,
                std::uint32_t world, const std::string &session,
                std::uint64_t generation, ExchangeKind kind,
                const char *operation) {
  if (error == cudaSuccess)
    return;
  std::ostringstream stream;
  const char *kind_text = kind == ExchangeKind::Control
                              ? "control"
                              : (kind == ExchangeKind::ExpertCounts
                                     ? "expert_counts"
                                     : "notify_packet_summary");
  stream << "phase=" << phase << ", kind=" << kind_text
         << ", rank=" << rank << ", world=" << world << ", session='"
         << session << "', generation=" << generation << ", operation="
         << operation << ", cuda_error=" << static_cast<int>(error) << " ("
         << cudaGetErrorName(error) << ": " << cudaGetErrorString(error)
         << ")";
  throw NotifyDispatchError(NotifyErrorCode::CudaFailure, stream.str());
}

} // namespace internal

struct CudaExecution::Impl {
  std::shared_ptr<CudaExecutionState> state;
  ExecutionToken token;
};

namespace internal {
std::unique_ptr<BackendCountExchangeOperation>
CudaExecutionStateAccess::start(const CountExchangeRequest &) {
  invalid("phase=cuda-execution-state-access, field=start, rank=n/a, "
          "expected=never called, actual=called");
}

std::shared_ptr<const CudaExecutionState>
CudaExecutionStateAccess::checked(const CudaExecution &execution) {
  if (!execution)
    invalid("phase=cuda-context, field=execution, rank=n/a, "
            "expected=valid, actual=empty");
  auto state = std::dynamic_pointer_cast<const CudaExecutionState>(
      backend_execution_state(execution.token()));
  if (!state || !state->stream)
    invalid("phase=cuda-context, field=execution_backend, rank=n/a, "
            "expected=live CUDA execution, actual=foreign/null");
  return state;
}

std::shared_ptr<const CudaExecutionState>
checked_execution_state(const CudaExecution &execution) {
  return CudaExecutionStateAccess::checked(execution);
}
} // namespace internal

CudaExecution::CudaExecution(const CudaExecution &) = default;
CudaExecution::CudaExecution(CudaExecution &&) noexcept = default;
CudaExecution &CudaExecution::operator=(const CudaExecution &) = default;
CudaExecution &CudaExecution::operator=(CudaExecution &&) noexcept = default;
CudaExecution::~CudaExecution() = default;

CudaExecution make_cuda_execution(std::string identity,
                                  std::uint32_t device_ordinal) {
  int count = 0;
  internal::throw_cuda("execution-create", cudaGetDeviceCount(&count), 0, 1,
                       identity, 1, ExchangeKind::ExpertCounts,
                       "cudaGetDeviceCount");
  if (device_ordinal >= static_cast<std::uint32_t>(count))
    invalid("phase=execution-create, field=device_ordinal, rank=n/a, expected=[0," +
            std::to_string(count) + "), actual=" +
            std::to_string(device_ordinal));
  internal::throw_cuda("execution-create", cudaSetDevice(static_cast<int>(device_ordinal)),
                       0, 1, identity, 1, ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
  auto state = std::make_shared<CudaExecutionState>();
  state->device = device_ordinal;
  internal::throw_cuda("execution-create",
                       cudaStreamCreateWithFlags(&state->stream,
                                                 cudaStreamNonBlocking),
                       0, 1, identity, 1, ExchangeKind::ExpertCounts,
                       "cudaStreamCreateWithFlags");
  auto impl = std::make_shared<CudaExecution::Impl>();
  impl->state = state;
  impl->token = make_execution_token(std::move(identity), state);
  return CudaExecution(std::move(impl));
}

const ExecutionToken &CudaExecution::token() const {
  if (!impl_)
    invalid("attempted to access empty CUDA execution");
  return impl_->token;
}

std::uint32_t CudaExecution::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty CUDA execution");
  return impl_->state->device;
}

struct SelectedExpertRouteLease<std::int32_t>::Impl {
  std::shared_ptr<CudaAllocation> allocation;
  std::shared_ptr<const void> slot_lease;
  const void *device_ptr = nullptr;
  CollectiveSession producer_session;
  ExecutionToken producer_execution;
  ExecutionToken consumer_execution;
  std::uint32_t layout = kSelectedExpertRouteLayoutVersion;
  std::uint32_t device = 0;
  std::uint64_t generation = 0;
  bool ready_recorded = false;
  std::uint64_t tokens = 0;
  std::uint32_t top_k = 0;
  std::uint64_t routes = 0;
};

struct SelectedExpertGateLease<float>::Impl {
  std::shared_ptr<CudaAllocation> allocation;
  const float *device_ptr = nullptr;
  CudaRouteInput bound_routes;
  CollectiveSession producer_session;
  ExecutionToken producer_execution;
  ExecutionToken consumer_execution;
  std::uint32_t layout = kSelectedExpertGateLayoutVersion;
  std::uint32_t device = 0;
  std::uint64_t generation = 0;
  bool ready_recorded = false;
  std::uint64_t tokens = 0;
  std::uint32_t top_k = 0;
  std::uint64_t routes = 0;
};

ExternalCudaProducer make_external_cuda_producer(
    RankCollectiveContext context, std::uint32_t device_ordinal,
    CUstream_st *producer_stream, std::shared_ptr<const void> stream_owner,
    std::string identity) {
  if (!context || !producer_stream || !stream_owner || identity.empty())
    invalid("phase=external-cuda-producer-create, field=context/stream/owner/identity, rank=n/a, expected=all valid, actual=empty");
  int device_count = 0;
  internal::throw_cuda("external-cuda-producer-create",
                       cudaGetDeviceCount(&device_count),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaGetDeviceCount");
  if (device_ordinal >= static_cast<std::uint32_t>(device_count))
    invalid("phase=external-cuda-producer-create, field=device, rank=" +
            std::to_string(context.calling_rank()) + ", expected=[0," +
            std::to_string(device_count) + "), actual=" +
            std::to_string(device_ordinal));
  auto backend = std::make_shared<internal::ExternalCudaProducerState>();
  backend->context = std::move(context);
  backend->device = device_ordinal;
  backend->stream = reinterpret_cast<cudaStream_t>(producer_stream);
  backend->stream_owner = std::move(stream_owner);
  backend->execution = make_execution_token(std::move(identity), backend);
  auto state = std::make_shared<ExternalCudaProducer::State>();
  state->backend = std::move(backend);
  return ExternalCudaProducer(std::move(state));
}

ExternalCudaProducer::ExternalCudaProducer(const ExternalCudaProducer &) =
    default;
ExternalCudaProducer::ExternalCudaProducer(ExternalCudaProducer &&) noexcept =
    default;
ExternalCudaProducer &ExternalCudaProducer::operator=(
    const ExternalCudaProducer &) = default;
ExternalCudaProducer &ExternalCudaProducer::operator=(
    ExternalCudaProducer &&) noexcept = default;
ExternalCudaProducer::~ExternalCudaProducer() = default;

const RankCollectiveContext &ExternalCudaProducer::context() const {
  if (!state_)
    invalid("phase=external-cuda-producer, field=handle, rank=n/a, expected=valid, actual=empty");
  return state_->backend->context;
}

std::uint32_t ExternalCudaProducer::device_ordinal() const {
  if (!state_)
    invalid("phase=external-cuda-producer, field=handle, rank=n/a, expected=valid, actual=empty");
  return state_->backend->device;
}

CudaRouteInput ExternalCudaProducer::publish_selected_experts(
    const std::int32_t *device_data, std::uint64_t capacity_elements,
    std::uint64_t num_tokens, std::uint32_t top_k,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !allocation_owner || top_k == 0)
    invalid("phase=external-route-publish, field=producer/owner/top_k, rank=n/a, expected=valid/nonzero, actual=invalid");
  const auto routes = checked_mul(num_tokens, top_k, "external route count");
  validate_external_cuda_range(device_data, routes, capacity_elements,
                               sizeof(std::int32_t), state_->backend->device,
                               "selected_experts");
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state_->backend->device;
  allocation->device_ptr = const_cast<std::int32_t *>(device_data);
  allocation->device_bytes = checked_size(
      checked_mul(capacity_elements, sizeof(std::int32_t),
                  "external route capacity"),
      "external route capacity");
  allocation->owns_device_memory = false;
  allocation->external_owner = std::make_shared<ExternalTensorOwners>(
      ExternalTensorOwners{std::move(allocation_owner), state_->backend});
  internal::throw_cuda(
      "external-route-publish",
      cudaEventCreateWithFlags(&allocation->producer_ready,
                               cudaEventDisableTiming),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventCreate(producer ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "external-route-publish",
      cudaEventRecord(allocation->producer_ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(producer ready)");
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = device_data;
  impl->producer_session = state_->backend->context.session();
  impl->producer_execution = state_->backend->execution;
  impl->consumer_execution = state_->backend->execution;
  impl->device = state_->backend->device;
  impl->generation = state_->backend->context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = num_tokens;
  impl->top_k = top_k;
  impl->routes = routes;
  return CudaRouteInput(std::move(impl));
}

CudaGateInput ExternalCudaProducer::publish_gates(
    const float *device_data, std::uint64_t capacity_elements,
    const CudaRouteInput &exact_routes,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !allocation_owner || !exact_routes.impl_)
    invalid("phase=external-gate-publish, field=producer/owner/routes, rank=n/a, expected=valid, actual=empty");
  const auto &route = *exact_routes.impl_;
  if (route.producer_session.stable_identity_bytes() !=
          state_->backend->context.session().stable_identity_bytes() ||
      route.generation != state_->backend->context.session().generation() ||
      route.device != state_->backend->device ||
      !route.producer_execution.same_instance(state_->backend->execution))
    invalid("phase=external-gate-publish, field=route_provenance, rank=" +
            std::to_string(state_->backend->context.calling_rank()) +
            ", expected=exact producer route lease, actual=mismatch");
  validate_external_cuda_range(device_data, route.routes, capacity_elements,
                               sizeof(float), state_->backend->device, "gates");
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state_->backend->device;
  allocation->device_ptr = const_cast<float *>(device_data);
  allocation->device_bytes = checked_size(
      checked_mul(capacity_elements, sizeof(float), "external gate capacity"),
      "external gate capacity");
  allocation->owns_device_memory = false;
  allocation->external_owner = std::make_shared<ExternalTensorOwners>(
      ExternalTensorOwners{std::move(allocation_owner), state_->backend});
  // Gate and route readiness share the immutable route provenance event. A
  // second record extends it to cover all producer work enqueued so far.
  internal::throw_cuda(
      "external-gate-publish",
      cudaEventRecord(route.allocation->producer_ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(route+gate ready)");
  auto impl = std::make_shared<CudaGateInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = device_data;
  impl->bound_routes = exact_routes;
  impl->producer_session = state_->backend->context.session();
  impl->producer_execution = state_->backend->execution;
  impl->consumer_execution = state_->backend->execution;
  impl->device = state_->backend->device;
  impl->generation = state_->backend->context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = route.tokens;
  impl->top_k = route.top_k;
  impl->routes = route.routes;
  return CudaGateInput(std::move(impl));
}

CudaRouteInput upload_cuda_routes(const RankRoutingInput &routes,
                                  std::uint32_t top_k,
                                  const RankCollectiveContext &context,
                                  const CudaExecution &execution) {
  if (!context || !execution ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=cuda-route-upload, field=producer_context, rank=n/a, "
            "expected=matching session/execution, actual=mismatch/empty");
  const auto state = internal::checked_execution_state(execution);
  const auto count = checked_mul(routes.num_tokens, top_k, "route_count");
  if (routes.selected_experts.size() != checked_size(count, "route_count"))
    invalid("phase=cuda-route-upload, field=selected_experts.size, rank=n/a, expected=" +
            std::to_string(count) + ", actual=" +
            std::to_string(routes.selected_experts.size()));
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state->device;
  const auto route_bytes = checked_size(
      checked_mul(count, sizeof(std::int32_t), "route_input_bytes"),
      "route_input_bytes");
  allocation->device_bytes = std::max<std::size_t>(route_bytes, 1);
  internal::throw_cuda("route-upload", cudaSetDevice(static_cast<int>(state->device)),
                       0, 1, execution.token().identity(), 1,
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda("route-upload",
                       cudaMalloc(&allocation->device_ptr,
                                  allocation->device_bytes),
                       0, 1, execution.token().identity(), 1,
                       ExchangeKind::ExpertCounts, "cudaMalloc(route input)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda("route-upload",
                       cudaEventCreateWithFlags(&allocation->producer_ready,
                                                cudaEventDisableTiming),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventCreate(producer ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  if (count != 0) {
    internal::throw_cuda(
        "route-upload",
        cudaMemcpyAsync(allocation->device_ptr, routes.selected_experts.data(),
                        route_bytes,
                        cudaMemcpyHostToDevice, state->stream),
        0, 1, execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(route input)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              route_bytes);
  }
  internal::throw_cuda("route-upload",
                       cudaEventRecord(allocation->producer_ready,
                                       state->stream),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventRecord(producer ready)");
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = impl->allocation->device_ptr;
  impl->producer_session = context.session();
  impl->producer_execution = execution.token();
  impl->consumer_execution = execution.token();
  impl->device = state->device;
  impl->generation = context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = routes.num_tokens;
  impl->top_k = top_k;
  impl->routes = count;
  return CudaRouteInput(std::move(impl));
}

std::uint64_t CudaRouteInput::num_tokens() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->tokens;
}
std::uint32_t CudaRouteInput::top_k() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->top_k;
}
std::uint64_t CudaRouteInput::route_count() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->routes;
}
std::uint64_t CudaRouteInput::device_bytes() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return checked_mul(impl_->routes, sizeof(std::int32_t),
                     "route input bytes");
}

std::uint32_t CudaRouteInput::layout_version() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->layout;
}
std::uint32_t CudaRouteInput::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->device;
}
std::uint64_t CudaRouteInput::route_generation() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->generation;
}

std::uint64_t CudaGateInput::num_tokens() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->tokens;
}
std::uint32_t CudaGateInput::top_k() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->top_k;
}
std::uint64_t CudaGateInput::route_count() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->routes;
}
std::uint64_t CudaGateInput::device_bytes() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return checked_mul(impl_->routes, sizeof(float), "gate input bytes");
}
std::uint32_t CudaGateInput::layout_version() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->layout;
}
std::uint32_t CudaGateInput::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->device;
}
std::uint64_t CudaGateInput::gate_generation() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->generation;
}

CudaGateInput DeviceGateProvenanceBridge::upload_internal(
    const std::vector<float> &host, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  if (host.size() != checked_size(routes.route_count(), "gate route count"))
    invalid("phase=cuda-gate-upload, field=shape, rank=" +
            std::to_string(context.calling_rank()) + ", expected=" +
            std::to_string(routes.route_count()) + ", actual=" +
            std::to_string(host.size()));
  for (std::size_t index = 0; index < host.size(); ++index)
    if (!std::isfinite(host[index]) || host[index] < 0.0F)
      invalid("phase=cuda-gate-upload, field=value, rank=" +
              std::to_string(context.calling_rank()) + ", expected=finite>=0, actual=invalid, route=" +
              std::to_string(index));
  const auto state = internal::checked_execution_state(execution);
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state->device;
  const auto bytes = checked_size(
      checked_mul(routes.route_count(), sizeof(float), "gate input bytes"),
      "gate input bytes");
  allocation->device_bytes = std::max<std::size_t>(bytes, 1);
  internal::throw_cuda(
      "gate-upload", cudaSetDevice(static_cast<int>(state->device)),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda(
      "gate-upload", cudaMalloc(&allocation->device_ptr,
                                  allocation->device_bytes),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaMalloc(gate input)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  if (bytes != 0) {
    internal::throw_cuda(
        "gate-upload",
        cudaMemcpyAsync(allocation->device_ptr, host.data(), bytes,
                        cudaMemcpyHostToDevice, state->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemcpyAsync(gate input)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  }
  internal::throw_cuda(
      "gate-upload",
      cudaEventRecord(routes.impl_->allocation->producer_ready, state->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(gate producer ready)");
  auto impl = std::make_shared<CudaGateInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = static_cast<const float *>(impl->allocation->device_ptr);
  impl->bound_routes = routes;
  impl->producer_session = context.session();
  impl->producer_execution = execution.token();
  impl->consumer_execution = execution.token();
  impl->device = state->device;
  impl->generation = context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = routes.num_tokens();
  impl->top_k = routes.top_k();
  impl->routes = routes.route_count();
  return CudaGateInput(std::move(impl));
}

CudaGateInput testing::upload_fp32_gates_for_testing(
    const std::vector<float> &host, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  return DeviceGateProvenanceBridge::upload_internal(
      host, routes, context, execution);
}

CudaRouteInput import_selected_expert_routes(
    const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution) {
  if (!routes.impl_ || !consumer_context || !consumer_execution)
    invalid("phase=route-stream-handoff, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &source = *routes.impl_;
  if (!consumer_context.execution().same_instance(consumer_execution.token()) ||
      consumer_context.calling_rank() >= consumer_context.session().world_size() ||
      source.producer_session.stable_identity_bytes() !=
          consumer_context.session().stable_identity_bytes() ||
      source.generation != consumer_context.session().generation())
    invalid("phase=route-stream-handoff, field=session/generation/execution, rank=" +
            std::to_string(consumer_context.calling_rank()) +
            ", expected=matching immutable producer provenance, actual=mismatch");
  const auto exec = internal::checked_execution_state(consumer_execution);
  if (exec->device != source.device || !source.ready_recorded ||
      !source.allocation || !source.allocation->producer_ready)
    invalid("phase=route-stream-handoff, field=device/ready_event, rank=" +
            std::to_string(consumer_context.calling_rank()) +
            ", expected=recorded producer event on matching device, actual=mismatch");
  internal::throw_cuda("route-stream-handoff",
                       cudaStreamWaitEvent(exec->stream,
                                           source.allocation->producer_ready),
                       consumer_context.calling_rank(),
                       consumer_context.session().world_size(),
                       consumer_context.session().identity(),
                       consumer_context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaStreamWaitEvent");
  auto impl = std::make_shared<CudaRouteInput::Impl>(source);
  impl->consumer_execution = consumer_execution.token();
  return CudaRouteInput(std::move(impl));
}

CudaGateInput import_selected_expert_gates(
    const CudaGateInput &gates, const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution) {
  if (!gates.impl_ || !routes.impl_ || !consumer_context ||
      !consumer_execution)
    invalid("phase=gate-stream-handoff, field=input, rank=n/a, expected=valid, actual=empty");
  const auto &source = *gates.impl_;
  if (!consumer_context.execution().same_instance(consumer_execution.token()) ||
      source.producer_session.stable_identity_bytes() !=
          consumer_context.session().stable_identity_bytes() ||
      source.generation != consumer_context.session().generation() ||
      source.device != consumer_execution.device_ordinal() ||
      source.tokens != routes.num_tokens() || source.top_k != routes.top_k() ||
      !source.ready_recorded || !source.bound_routes.impl_ ||
      source.bound_routes.impl_->allocation != routes.impl_->allocation ||
      !routes.impl_->allocation || !routes.impl_->allocation->producer_ready)
    invalid("phase=gate-stream-handoff, field=provenance/shape/ready, rank=" +
            std::to_string(consumer_context.calling_rank()) +
            ", expected=matching exact routes/session/generation/device, actual=mismatch");
  const auto exec = internal::checked_execution_state(consumer_execution);
  internal::throw_cuda(
      "gate-stream-handoff",
      cudaStreamWaitEvent(exec->stream,
                          routes.impl_->allocation->producer_ready),
      consumer_context.calling_rank(), consumer_context.session().world_size(),
      consumer_context.session().identity(),
      consumer_context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(route+gate ready)");
  auto impl = std::make_shared<CudaGateInput::Impl>(source);
  impl->bound_routes = routes;
  impl->consumer_execution = consumer_execution.token();
  return CudaGateInput(std::move(impl));
}

CudaRouteInput DeviceRouteProvenanceBridge::reissue_for_testing(
    const CudaRouteInput &storage, const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!storage.impl_ || !context || !execution)
    invalid("phase=test-route-reissue, field=input, rank=n/a, "
            "expected=valid private storage/context/execution, actual=empty");
  const auto &source = *storage.impl_;
  if (!context.execution().same_instance(execution.token()) ||
      !source.producer_execution.same_instance(execution.token()) ||
      source.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      !source.allocation || !source.allocation->producer_ready)
    invalid("phase=test-route-reissue, field=provenance, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=same immutable storage pipeline/execution/device, "
            "actual=mismatch");
  auto impl = std::make_shared<CudaRouteInput::Impl>(source);
  impl->producer_session = context.session();
  impl->generation = context.session().generation();
  return CudaRouteInput(std::move(impl));
}

CudaRouteInput DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
    const CudaRouteInput &storage, std::int32_t first_expert,
    std::uint64_t delay_cycles, const RankCollectiveContext &context,
    const CudaExecution &producer_execution, bool record_ready) {
  auto exact = reissue_for_testing(storage, context, producer_execution);
  auto impl = std::make_shared<CudaRouteInput::Impl>(*exact.impl_);
  auto exec = internal::checked_execution_state(producer_execution);
  const auto bytes = checked_size(
      checked_mul(impl->routes, sizeof(std::int32_t), "test delayed routes"),
      "test delayed routes");
  if (bytes != 0) {
    internal::throw_cuda(
        "test-route-producer", cudaMemset(
            const_cast<void *>(impl->device_ptr), 0xff, bytes),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemset(route poison)");
    const auto blocks = static_cast<unsigned int>((impl->routes + 127U) / 128U);
    delayed_route_fill_kernel<<<blocks, 128, 0, exec->stream>>>(
        static_cast<std::int32_t *>(const_cast<void *>(impl->device_ptr)),
        impl->routes, first_expert, delay_cycles);
    internal::throw_cuda(
        "test-route-producer", cudaGetLastError(), context.calling_rank(),
        context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "delayed route kernel launch");
  }
  impl->ready_recorded = false;
  if (record_ready) {
    internal::throw_cuda(
        "test-route-producer",
        cudaEventRecord(impl->allocation->producer_ready, exec->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaEventRecord(producer ready)");
    impl->ready_recorded = true;
  }
  return CudaRouteInput(std::move(impl));
}

CudaRouteInput DeviceRouteProvenanceBridge::unsafe_import_without_wait_for_testing(
    const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution) {
  if (!routes.impl_ || !consumer_context || !consumer_execution)
    invalid("phase=test-route-import-no-wait, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  auto impl = std::make_shared<CudaRouteInput::Impl>(*routes.impl_);
  impl->consumer_execution = consumer_execution.token();
  return CudaRouteInput(std::move(impl));
}

void DeviceRouteProvenanceBridge::validate(
    const CudaRouteInput &routes, const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!routes.impl_ || !context || !execution)
    invalid("phase=device-route-provenance, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &route = *routes.impl_;
  const auto required_bytes = checked_mul(
      route.routes, sizeof(std::int32_t), "route provenance bytes");
  if (route.layout != kSelectedExpertRouteLayoutVersion ||
      route.device != execution.device_ordinal() ||
      !route.consumer_execution.same_instance(execution.token()) ||
      route.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      route.generation != context.session().generation() ||
      !route.ready_recorded || !route.allocation ||
      !route.allocation->producer_ready ||
      required_bytes > route.allocation->device_bytes)
    invalid("phase=device-route-provenance, field=dtype/layout/shape/device/session/generation/execution/ready/capacity, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact immutable producer provenance, actual=mismatch");
}

bool DeviceGateProvenanceBridge::same_route_storage(
    const CudaGateInput &gates, const CudaRouteInput &routes) {
  return gates.impl_ && routes.impl_ && gates.impl_->bound_routes.impl_ &&
         gates.impl_->bound_routes.impl_->allocation == routes.impl_->allocation;
}

void DeviceGateProvenanceBridge::validate(
    const CudaGateInput &gates, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  if (!gates.impl_)
    invalid("phase=device-gate-provenance, field=input, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=valid FP32 gate lease, actual=empty");
  const auto &gate = *gates.impl_;
  const auto required_bytes = checked_mul(
      gate.routes, sizeof(float), "gate provenance bytes");
  if (gate.layout != kSelectedExpertGateLayoutVersion ||
      gate.device != execution.device_ordinal() ||
      !gate.consumer_execution.same_instance(execution.token()) ||
      gate.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      gate.generation != context.session().generation() ||
      gate.tokens != routes.num_tokens() || gate.top_k != routes.top_k() ||
      gate.routes != routes.route_count() || !gate.ready_recorded ||
      !gate.allocation || !gate.bound_routes.impl_ ||
      !gate.bound_routes.impl_->allocation ||
      !gate.bound_routes.impl_->allocation->producer_ready ||
      required_bytes > gate.allocation->device_bytes ||
      !same_route_storage(gates, routes))
    invalid("phase=device-gate-provenance, field=dtype/layout/shape/device/session/generation/execution/ready/capacity/route_binding, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact immutable route+gate provenance, actual=mismatch");
}

CudaGateInput DeviceGateProvenanceBridge::reissue_for_testing(
    const CudaGateInput &storage, const CudaRouteInput &exact_routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!storage.impl_ || !exact_routes || !context || !execution)
    invalid("phase=test-gate-reissue, field=input, rank=n/a, expected=valid, actual=empty");
  const auto &source = *storage.impl_;
  if (!context.execution().same_instance(execution.token()) ||
      !source.producer_execution.same_instance(execution.token()) ||
      source.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      !source.allocation || !source.bound_routes.impl_ ||
      !source.bound_routes.impl_->allocation ||
      !source.bound_routes.impl_->allocation->producer_ready ||
      !same_route_storage(storage, exact_routes) ||
      source.tokens != exact_routes.num_tokens() ||
      source.top_k != exact_routes.top_k())
    invalid("phase=test-gate-reissue, field=provenance, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=same immutable route storage/execution/device, actual=mismatch");
  auto impl = std::make_shared<CudaGateInput::Impl>(source);
  impl->bound_routes = exact_routes;
  impl->producer_session = context.session();
  impl->generation = context.session().generation();
  return CudaGateInput(std::move(impl));
}

const float *
DeviceGateProvenanceBridge::device_pointer(const CudaGateInput &gates) {
  if (!gates.impl_ || !gates.impl_->device_ptr)
    invalid("phase=device-gate-view, field=gate, rank=n/a, expected=valid, actual=empty");
  return gates.impl_->device_ptr;
}

