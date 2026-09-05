namespace {

class NcclAllGatherOperation final : public BackendCountExchangeOperation {
public:
  NcclAllGatherOperation(std::shared_ptr<NcclAllGatherCountExchange::Impl> impl,
                         Slot *slot, CountExchangeRequest request)
      : impl_(std::move(impl)), slot_(slot), request_(std::move(request)) {}

  ~NcclAllGatherOperation() noexcept override {
    if (!terminal_) {
      testing::g_destructor_drains.fetch_add(1);
      (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
      (void)cudaEventSynchronize(slot_->completion);
      release();
      restore_protocol();
    }
  }

  ExchangeState state() const noexcept override {
    if (cancelled_)
      return ExchangeState::Cancelled;
    if (terminal_)
      return ExchangeState::Complete;
    (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
    const auto event = cudaEventQuery(slot_->completion);
    if (event == cudaSuccess)
      return ExchangeState::Complete;
    return event == cudaErrorNotReady ? ExchangeState::Pending
                                     : ExchangeState::Failed;
  }

  BackendCompletion wait() override {
    if (terminal_)
      invalid("phase=nccl-wait, kind=" + std::string(kind_name(request_.kind)) +
              ", rank=" + std::to_string(request_.context.calling_rank()) +
              ", expected=pending, actual=terminal");
    testing::g_operation_waits.fetch_add(1);
    const auto &session = request_.context.session();
    internal::throw_cuda("nccl-wait", cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())),
                         request_.context.calling_rank(), session.world_size(),
                         session.identity(), session.generation(), request_.kind,
                         "cudaSetDevice");
    internal::throw_cuda("nccl-wait", cudaEventSynchronize(slot_->completion),
                         request_.context.calling_rank(), session.world_size(),
                         session.identity(), session.generation(), request_.kind,
                         "cudaEventSynchronize");
    ncclResult_t async = ncclSuccess;
    internal::throw_nccl("nccl-wait",
                         ncclCommGetAsyncError(impl_->comm, &async),
                         request_.context.calling_rank(), session.world_size(),
                         session.identity(), session.generation(), request_.kind,
                         "ncclCommGetAsyncError");
    internal::throw_nccl("nccl-wait", async, request_.context.calling_rank(),
                         session.world_size(), session.identity(),
                         session.generation(), request_.kind,
                         "NCCL asynchronous completion");
    if (request_.kind == ExchangeKind::NotifyPacketSummary)
      maybe_fault(testing::NcclFaultPoint::BackendCompletion,
                  "summary-backend-completion", request_.context);
    terminal_ = true;
    release();
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      if (request_.kind == ExchangeKind::ExpertCounts &&
          impl_->protocol == NcclAllGatherCountExchange::Impl::ProtocolState::CountStarted)
        impl_->protocol = NcclAllGatherCountExchange::Impl::ProtocolState::CountComplete;
      if (request_.kind == ExchangeKind::NotifyPacketSummary)
        impl_->protocol = NcclAllGatherCountExchange::Impl::ProtocolState::Complete;
    }
    auto lease = std::make_shared<internal::CudaCompletionLease>();
    lease->owner = impl_;
    auto buffer = std::make_shared<internal::CudaBufferState>();
    buffer->owner = impl_;
    buffer->pointer = slot_->device_matrix;
    buffer->device = impl_->session.device_ordinal();
    const auto elements = checked_elements(session.world_size(),
                                           request_.row_width,
                                           "completion_matrix_elements");
    return make_device_backend_completion(
        request_.context, request_.kind,
        make_device_buffer_lease(session.world_size(), request_.row_width,
                                 elements, std::move(buffer)),
        std::move(lease));
  }

  void cancel() override {
    if (terminal_)
      return;
    testing::g_operation_cancels.fetch_add(1);
    const auto &session = request_.context.session();
    internal::throw_cuda("nccl-cancel", cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())),
                         request_.context.calling_rank(), session.world_size(),
                         session.identity(), session.generation(), request_.kind,
                         "cudaSetDevice");
    // NCCL has no per-operation cancellation. Synchronizing the recorded event
    // is the safe cleanup boundary and does not abort the shared communicator.
    internal::throw_cuda("nccl-cancel", cudaEventSynchronize(slot_->completion),
                         request_.context.calling_rank(), session.world_size(),
                         session.identity(), session.generation(), request_.kind,
                         "cudaEventSynchronize");
    cancelled_ = true;
    terminal_ = true;
    release();
    restore_protocol();
  }

private:
  void release() noexcept {
    if (released_)
      return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    slot_->in_use = false;
    released_ = true;
  }

  void restore_protocol() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (request_.kind == ExchangeKind::ExpertCounts &&
        impl_->protocol ==
            NcclAllGatherCountExchange::Impl::ProtocolState::CountStarted)
      impl_->protocol =
          NcclAllGatherCountExchange::Impl::ProtocolState::DataReady;
    if (request_.kind == ExchangeKind::NotifyPacketSummary &&
        impl_->protocol ==
            NcclAllGatherCountExchange::Impl::ProtocolState::SummaryStarted)
      impl_->protocol =
          NcclAllGatherCountExchange::Impl::ProtocolState::CountComplete;
  }

  std::shared_ptr<NcclAllGatherCountExchange::Impl> impl_;
  Slot *slot_ = nullptr;
  CountExchangeRequest request_;
  bool terminal_ = false;
  bool cancelled_ = false;
  bool released_ = false;
};

} // namespace

NcclAllGatherCountExchange::NcclAllGatherCountExchange(
    const NcclSession &session, const CudaExecution &execution) {
  if (!session || !execution)
    invalid("phase=nccl-backend-create, field=input, rank=n/a, expected=valid session/execution, actual=empty");
  if (session.device_ordinal() != execution.device_ordinal())
    invalid("phase=nccl-backend-create, field=device, rank=" +
            std::to_string(session.rank()) + ", expected=" +
            std::to_string(session.device_ordinal()) + ", actual=" +
            std::to_string(execution.device_ordinal()));
  impl_ = std::make_shared<Impl>();
  impl_->session = session;
  impl_->execution = execution;
  impl_->execution_state = internal::checked_execution_state(execution);
  impl_->comm = session.impl_->state->comm;
  impl_->control = std::make_unique<ByteSlot>();
  const auto control_start = std::chrono::steady_clock::now();
  allocate_byte_slot(*impl_->control, kControlWireBytes, session,
                     "nccl-control-workspace", &impl_->setup);
  const auto control_end = std::chrono::steady_clock::now();
  impl_->setup.control_slot_allocation_us =
      std::chrono::duration<double, std::micro>(control_end - control_start)
          .count();
}

NcclAllGatherCountExchange::NcclAllGatherCountExchange(
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t max_expert_row_width,
    std::uint64_t max_summary_row_width)
    : NcclAllGatherCountExchange(session, execution) {
  auto expert = std::make_unique<Slot>();
  auto summary = std::make_unique<Slot>();
  allocate_slot(*expert, max_expert_row_width, session,
                ExchangeKind::ExpertCounts, &impl_->setup);
  allocate_slot(*summary, max_summary_row_width, session,
                ExchangeKind::NotifyPacketSummary, &impl_->setup);
  impl_->expert = std::move(expert);
  impl_->summary = std::move(summary);
}

NcclAllGatherCountExchange::NcclAllGatherCountExchange(
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t max_expert_row_width,
    std::uint64_t max_summary_row_width, std::size_t max_config_wire_bytes)
    : NcclAllGatherCountExchange(session, execution, max_expert_row_width,
                                 max_summary_row_width) {
  impl_->config = std::make_unique<ByteSlot>();
  allocate_byte_slot(*impl_->config, max_config_wire_bytes, session,
                     "nccl-config-pool", &impl_->setup);
  constexpr std::size_t kFastHeaderBytes = kControlWireBytes;
  if (max_expert_row_width >
          (std::numeric_limits<std::size_t>::max() - kFastHeaderBytes) /
              sizeof(std::uint64_t) ||
      max_summary_row_width >
          (std::numeric_limits<std::size_t>::max() - kFastHeaderBytes -
           static_cast<std::size_t>(max_expert_row_width) *
               sizeof(std::uint64_t)) /
              sizeof(std::uint64_t))
    invalid("phase=nccl-fast-notify-pool, field=record_bytes, rank=" +
            std::to_string(session.rank()) +
            ", expected=size_t sum, actual=overflow");
  const auto fast_bytes =
      kFastHeaderBytes +
      static_cast<std::size_t>(max_expert_row_width) * sizeof(std::uint64_t) +
      static_cast<std::size_t>(max_summary_row_width) * sizeof(std::uint64_t);
  impl_->fast_notify = std::make_unique<ByteSlot>();
  allocate_byte_slot(*impl_->fast_notify, fast_bytes, session,
                     "nccl-fast-notify-pool", &impl_->setup);
}

void NcclAllGatherCountExchange::prepare_generation(
    const NcclSession &session) {
  if (!impl_ || !session ||
      session.impl_->state.get() != impl_->session.impl_->state.get())
    invalid("phase=nccl-pool-prepare, field=session, rank=n/a, "
            "expected=same native communicator, actual=mismatch");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto busy = impl_->control->in_use ||
                    (impl_->config && impl_->config->in_use) ||
                    (impl_->expert && impl_->expert->in_use) ||
                    (impl_->summary && impl_->summary->in_use) ||
                    (impl_->fast_notify && impl_->fast_notify->in_use);
  if (busy)
    invalid("phase=nccl-pool-prepare, field=slot, rank=" +
            std::to_string(session.rank()) +
            ", expected=idle, actual=in-use");
  impl_->session = session;
  impl_->audit = {};
  impl_->agreed_expert_width = 0;
  impl_->agreed_summary_width = 0;
  impl_->protocol = Impl::ProtocolState::ControlReady;
}

NcclAllGatherCountExchange::~NcclAllGatherCountExchange() noexcept = default;
NcclAllGatherCountExchange::NcclAllGatherCountExchange(
    NcclAllGatherCountExchange &&) noexcept = default;
NcclAllGatherCountExchange &NcclAllGatherCountExchange::operator=(
    NcclAllGatherCountExchange &&) noexcept = default;

NcclControlOperation NcclAllGatherCountExchange::start_control(
    const RankCollectiveContext &context, const ControlHeader &record) {
  if (!impl_ || !context)
    invalid("phase=control-start, field=context/backend, rank=n/a, expected=valid, actual=empty");
  const auto rank = context.calling_rank();
  const auto &session = context.session();
  maybe_fault(testing::NcclFaultPoint::ControlStart, "control-start",
              context);
  if (!session.same_instance(impl_->session.session()) ||
      !context.execution().same_instance(impl_->execution.token()))
    invalid("phase=control-start, field=context, rank=" +
            std::to_string(rank) +
            ", expected=backend session/execution instances, actual=mismatch");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->protocol != Impl::ProtocolState::ControlReady ||
      impl_->control->in_use)
    invalid("phase=control-start, field=protocol_state, rank=" +
            std::to_string(rank) +
            ", expected=control-ready, actual=busy/advanced");
  impl_->protocol = Impl::ProtocolState::ControlPending;
  impl_->control->in_use = true;
  const auto wire = encode_control_header(record);
  std::copy(wire.begin(), wire.end(), impl_->control->host_send);
  const auto bytes = kControlWireBytes;
  try {
    internal::throw_cuda(
        "control-start",
        cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())), rank,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaSetDevice");
    internal::throw_cuda(
        "control-start",
        cudaMemsetAsync(impl_->control->device_matrix, 0,
                        impl_->control->width *
                            static_cast<std::size_t>(session.world_size()),
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemsetAsync(ControlMatrix capacity)");
    internal::throw_cuda(
        "control-start",
        cudaMemcpyAsync(impl_->control->device_send,
                        impl_->control->host_send, bytes,
                        cudaMemcpyHostToDevice, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemcpyAsync(H2D ControlWire)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
    maybe_fault(testing::NcclFaultPoint::ControlEnqueue,
                "control-enqueue", context);
    internal::throw_nccl(
        "control-start",
        ncclAllGather(impl_->control->device_send,
                      impl_->control->device_matrix,
                      kControlWireBytes, ncclUint8,
                      impl_->comm, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "ncclAllGather(ControlWire bytes)");
    // Audit means the NCCL call returned success and the collective is now
    // enqueued. It must not depend on later D2H/event/completion success.
    ++impl_->audit.control_starts;
    const auto matrix_bytes =
        static_cast<std::size_t>(session.world_size()) * kControlWireBytes;
    maybe_fault(testing::NcclFaultPoint::D2HStaging,
                "control-d2h-staging", context);
    internal::throw_cuda(
        "control-start",
        cudaMemcpyAsync(impl_->control->host_matrix,
                        impl_->control->device_matrix, matrix_bytes,
                        cudaMemcpyDeviceToHost, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemcpyAsync(D2H ControlMatrix)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              matrix_bytes);
    maybe_fault(testing::NcclFaultPoint::EventRecord,
                "control-event-record", context);
    internal::throw_cuda(
        "control-start",
        cudaEventRecord(impl_->control->completion,
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaEventRecord");
  } catch (...) {
    (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
    (void)cudaStreamSynchronize(impl_->execution_state->stream);
    impl_->control->in_use = false;
    impl_->protocol = Impl::ProtocolState::ControlReady;
    throw;
  }
  auto operation = std::make_shared<NcclControlOperation::Impl>();
  operation->backend = impl_;
  operation->context = context;
  operation->input = record;
  operation->slot = impl_->control.get();
  return NcclControlOperation(std::move(operation));
}

NcclConfigOperation NcclAllGatherCountExchange::start_config(
    const RankCollectiveContext &context, const ConfigWire &wire,
    const ValidatedControlHandle &control) {
  if (!impl_ || !context || !control)
    invalid("phase=config-start, field=input, rank=n/a, expected=valid context/control/backend, actual=empty");
  const auto rank = context.calling_rank();
  const auto &session = context.session();
  if (!session.same_instance(impl_->session.session()) ||
      !context.execution().same_instance(impl_->execution.token()) ||
      !control.context().same_instance(context))
    invalid("phase=config-start, field=provenance, rank=" +
            std::to_string(rank) +
            ", expected=exact coordinator session/execution/control, actual=mismatch");
  if (control.matrix().empty() || rank >= control.matrix().size() ||
      wire.size() != control.matrix()[rank].config_blob_bytes)
    invalid("phase=config-start, field=config_blob_bytes, rank=" +
            std::to_string(rank) + ", expected=" +
            (control.matrix().empty()
                 ? std::string("validated header")
                 : std::to_string(control.matrix()[rank].config_blob_bytes)) +
            ", actual=" + std::to_string(wire.size()));
  std::size_t collective_width = 0;
  for (const auto &header : control.matrix()) {
    if (header.config_blob_bytes > kMaxCanonicalConfigBytes ||
        header.config_blob_bytes > SIZE_MAX)
      invalid("phase=config-start, field=config_blob_bytes, rank=" +
              std::to_string(rank) +
              ", expected=bounded size_t, actual=" +
              std::to_string(header.config_blob_bytes));
    collective_width = std::max(
        collective_width, static_cast<std::size_t>(header.config_blob_bytes));
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->protocol != Impl::ProtocolState::ConfigReady)
    invalid("phase=config-start, field=protocol_state, rank=" +
            std::to_string(rank) +
            ", expected=config-ready, actual=not-ready");
  maybe_fault(testing::NcclFaultPoint::ConfigSlotAllocation,
              "config-slot-allocation", context);
  if (!impl_->config || impl_->config->width < collective_width) {
    impl_->config = std::make_unique<ByteSlot>();
    const auto allocation_start = std::chrono::steady_clock::now();
    allocate_byte_slot(*impl_->config, collective_width, impl_->session,
                       "nccl-config-workspace", &impl_->setup);
    const auto allocation_end = std::chrono::steady_clock::now();
    impl_->setup.config_slot_allocation_us +=
        std::chrono::duration<double, std::micro>(allocation_end -
                                                  allocation_start)
            .count();
  }
  if (impl_->config->in_use)
    invalid("phase=config-start, field=slot, rank=" + std::to_string(rank) +
            ", expected=available, actual=in-use");
  impl_->config->in_use = true;
  impl_->protocol = Impl::ProtocolState::ConfigPending;
  std::fill_n(impl_->config->host_send, impl_->config->width, std::byte{0});
  std::copy(wire.begin(), wire.end(), impl_->config->host_send);
  try {
    internal::throw_cuda(
        "config-start",
        cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())), rank,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaSetDevice");
    internal::throw_cuda(
        "config-start",
        cudaMemsetAsync(impl_->config->device_matrix, 0,
                        impl_->config->width *
                            static_cast<std::size_t>(session.world_size()),
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemsetAsync(ConfigMatrix capacity)");
    internal::throw_cuda(
        "config-start",
        cudaMemcpyAsync(impl_->config->device_send, impl_->config->host_send,
                        collective_width, cudaMemcpyHostToDevice,
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemcpyAsync(H2D ConfigWire)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              collective_width);
    internal::throw_nccl(
        "config-start",
        ncclAllGather(impl_->config->device_send, impl_->config->device_matrix,
                      collective_width, ncclUint8, impl_->comm,
                      impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "ncclAllGather(ConfigWire bytes)");
    ++impl_->audit.config_starts;
    const auto matrix_bytes =
        collective_width * static_cast<std::size_t>(session.world_size());
    internal::throw_cuda(
        "config-start",
        cudaMemcpyAsync(impl_->config->host_matrix,
                        impl_->config->device_matrix, matrix_bytes,
                        cudaMemcpyDeviceToHost, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemcpyAsync(D2H ConfigMatrix)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              matrix_bytes);
    internal::throw_cuda(
        "config-start",
        cudaEventRecord(impl_->config->completion,
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaEventRecord");
  } catch (...) {
    (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
    (void)cudaStreamSynchronize(impl_->execution_state->stream);
    impl_->config->in_use = false;
    impl_->protocol = Impl::ProtocolState::ConfigReady;
    throw;
  }
  auto operation = std::make_shared<NcclConfigOperation::Impl>();
  operation->backend = impl_;
  operation->context = context;
  operation->input = wire;
  operation->headers = control.matrix();
  operation->slot = impl_->config.get();
  operation->collective_width = collective_width;
  return NcclConfigOperation(std::move(operation));
}

bool NcclAllGatherCountExchange::can_reuse_verified_config(
    const ConfigWire &wire) const {
  if (!impl_)
    return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return !impl_->verified_config.empty() && impl_->verified_config == wire;
}

bool NcclAllGatherCountExchange::fast_path_ready() const {
  if (!impl_)
    return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return !impl_->verified_config.empty() && impl_->fast_notify &&
         impl_->expert && impl_->summary;
}

bool NcclAllGatherCountExchange::fast_config_matches(
    const ConfigWire &wire) const {
  if (!impl_)
    return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return !impl_->verified_config.empty() && impl_->verified_config == wire;
}

NcclAllGatherCountExchange::FastNotifyResult
NcclAllGatherCountExchange::run_fast_notify(
    const RankCollectiveContext &context, const ControlHeader &header,
    const DeviceBufferLease *expert_input,
    const DeviceBufferLease *summary_input) {
  if (!impl_ || !context)
    invalid("phase=fast-notify, field=context/backend, rank=n/a, "
            "expected=valid, actual=empty");
  const auto rank = context.calling_rank();
  const auto &session = context.session();
  if (!session.same_instance(impl_->session.session()) ||
      !context.execution().same_instance(impl_->execution.token()))
    invalid("phase=fast-notify, field=context, rank=" +
            std::to_string(rank) +
            ", expected=backend session/execution, actual=mismatch");

  std::shared_ptr<const internal::CudaBufferState> expert_state;
  std::shared_ptr<const internal::CudaBufferState> summary_state;
  if ((expert_input == nullptr) != (summary_input == nullptr))
    invalid("phase=fast-notify, field=device_inputs, rank=" +
            std::to_string(rank) +
            ", expected=both present or both absent, actual=partial");
  if (expert_input) {
    if (!*expert_input || !*summary_input || expert_input->rows() != 1 ||
        summary_input->rows() != 1 ||
        expert_input->row_width() != header.expert_row_width ||
        summary_input->row_width() != header.summary_row_width)
      invalid("phase=fast-notify, field=device_input.shape, rank=" +
              std::to_string(rank) +
              ", expected=agreed local rows, actual=mismatch");
    expert_state =
        std::dynamic_pointer_cast<const internal::CudaBufferState>(
            backend_device_state(*expert_input));
    summary_state =
        std::dynamic_pointer_cast<const internal::CudaBufferState>(
            backend_device_state(*summary_input));
    if (!expert_state || !summary_state || !expert_state->pointer ||
        !summary_state->pointer ||
        expert_state->device != impl_->session.device_ordinal() ||
        summary_state->device != impl_->session.device_ordinal())
      invalid("phase=fast-notify, field=device_input.backend, rank=" +
              std::to_string(rank) +
              ", expected=CUDA leases on backend device, actual=foreign/null");
  }
  ByteSlot *fast = nullptr;
  Slot *expert = nullptr;
  Slot *summary = nullptr;
  std::uint64_t expert_width = 0;
  std::uint64_t summary_width = 0;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->protocol != Impl::ProtocolState::ControlReady ||
        impl_->verified_config.empty() || !impl_->fast_notify ||
        !impl_->expert || !impl_->summary || impl_->fast_notify->in_use ||
        impl_->expert->in_use || impl_->summary->in_use)
      invalid("phase=fast-notify, field=protocol_state, rank=" +
              std::to_string(rank) +
              ", expected=verified/idle, actual=not-ready/busy");
    expert_width = impl_->verified_expert_width;
    summary_width = impl_->verified_summary_width;
    fast = impl_->fast_notify.get();
    expert = impl_->expert.get();
    summary = impl_->summary.get();
    fast->in_use = true;
    expert->in_use = true;
    summary->in_use = true;
    impl_->protocol = Impl::ProtocolState::FastNotifyStarted;
  }
  const auto release = [&]() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    fast->in_use = false;
    expert->in_use = false;
    summary->in_use = false;
  };
  if (expert_width > SIZE_MAX / sizeof(std::uint64_t) ||
      summary_width > SIZE_MAX / sizeof(std::uint64_t)) {
    release();
    invalid("phase=fast-notify, field=record_width, rank=" +
            std::to_string(rank) +
            ", expected=size_t widths, actual=overflow");
  }
  const auto expert_bytes =
      static_cast<std::size_t>(expert_width) * sizeof(std::uint64_t);
  const auto summary_bytes =
      static_cast<std::size_t>(summary_width) * sizeof(std::uint64_t);
  if (expert_bytes > SIZE_MAX - kControlWireBytes ||
      summary_bytes > SIZE_MAX - kControlWireBytes - expert_bytes) {
    release();
    invalid("phase=fast-notify, field=record_bytes, rank=" +
            std::to_string(rank) +
            ", expected=size_t sum, actual=overflow");
  }
  const auto record_bytes = kControlWireBytes + expert_bytes + summary_bytes;
  if (record_bytes > fast->width || expert_width > expert->width ||
      summary_width > summary->width) {
    release();
    invalid("phase=fast-notify, field=slot_capacity, rank=" +
            std::to_string(rank) + ", expected=preallocated record, actual=short");
  }

  const auto wire = encode_control_header(header);
  const auto enqueue_begin = std::chrono::steady_clock::now();
  try {
    internal::throw_cuda(
        "fast-notify", cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaSetDevice");
    internal::throw_cuda(
        "fast-notify",
        cudaMemsetAsync(fast->device_send, 0, record_bytes,
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemsetAsync(fast send)");
    internal::throw_cuda(
        "fast-notify",
        cudaMemsetAsync(fast->device_matrix, 0,
                        record_bytes * static_cast<std::size_t>(session.world_size()),
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemsetAsync(fast receive)");
    internal::throw_cuda(
        "fast-notify",
        cudaMemcpyAsync(fast->device_send, wire.data(), kControlWireBytes,
                        cudaMemcpyHostToDevice, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemcpyAsync(H2D fast header)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              kControlWireBytes);
    if (expert_state) {
      internal::throw_cuda(
          "fast-notify",
          cudaMemcpyAsync(fast->device_send + kControlWireBytes,
                          expert_state->pointer, expert_bytes,
                          cudaMemcpyDeviceToDevice,
                          impl_->execution_state->stream),
          rank, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMemcpyAsync(D2D expert row)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToDevice,
                                expert_bytes);
      internal::throw_cuda(
          "fast-notify",
          cudaMemcpyAsync(fast->device_send + kControlWireBytes + expert_bytes,
                          summary_state->pointer, summary_bytes,
                          cudaMemcpyDeviceToDevice,
                          impl_->execution_state->stream),
          rank, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::NotifyPacketSummary,
          "cudaMemcpyAsync(D2D summary row)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToDevice,
                                summary_bytes);
    }
    internal::throw_nccl(
        "fast-notify",
        ncclAllGather(fast->device_send, fast->device_matrix, record_bytes,
                      ncclUint8, impl_->comm, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "ncclAllGather(NotifyFastRecord)");
    ++impl_->audit.fused_notify_starts;
    internal::throw_cuda(
        "fast-notify",
        cudaMemcpy2DAsync(expert->device_matrix, expert_bytes,
                          fast->device_matrix + kControlWireBytes, record_bytes,
                          expert_bytes, session.world_size(),
                          cudaMemcpyDeviceToDevice,
                          impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts,
        "cudaMemcpy2DAsync(unpack expert matrix)");
    internal::record_cuda_api(
        internal::CudaApiAuditKind::DeviceToDevice,
        expert_bytes * static_cast<std::size_t>(session.world_size()));
    internal::throw_cuda(
        "fast-notify",
        cudaMemcpy2DAsync(summary->device_matrix, summary_bytes,
                          fast->device_matrix + kControlWireBytes + expert_bytes,
                          record_bytes, summary_bytes, session.world_size(),
                          cudaMemcpyDeviceToDevice,
                          impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::NotifyPacketSummary,
        "cudaMemcpy2DAsync(unpack summary matrix)");
    internal::record_cuda_api(
        internal::CudaApiAuditKind::DeviceToDevice,
        summary_bytes * static_cast<std::size_t>(session.world_size()));
    internal::throw_cuda(
        "fast-notify",
        cudaMemcpy2DAsync(fast->host_matrix, kControlWireBytes,
                          fast->device_matrix, record_bytes, kControlWireBytes,
                          session.world_size(), cudaMemcpyDeviceToHost,
                          impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaMemcpy2DAsync(D2H fast headers)");
    internal::record_cuda_api(
        internal::CudaApiAuditKind::DeviceToHost,
        kControlWireBytes * static_cast<std::size_t>(session.world_size()));
    internal::throw_cuda(
        "fast-notify",
        cudaEventRecord(fast->completion, impl_->execution_state->stream), rank,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaEventRecord(fast notify)");
  } catch (...) {
    (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
    (void)cudaStreamSynchronize(impl_->execution_state->stream);
    release();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->protocol = Impl::ProtocolState::ControlReady;
    throw;
  }
  const auto enqueue_end = std::chrono::steady_clock::now();

  auto make_completion = [&](Slot *slot, ExchangeKind kind,
                             std::uint64_t width) {
    auto lease = std::make_shared<internal::CudaCompletionLease>();
    lease->owner = impl_;
    auto buffer = std::make_shared<internal::CudaBufferState>();
    buffer->owner = impl_;
    buffer->pointer = slot->device_matrix;
    buffer->device = impl_->session.device_ordinal();
    const auto elements = checked_elements(session.world_size(), width,
                                           "fast completion matrix");
    return make_device_backend_completion(
        context, kind,
        make_device_buffer_lease(session.world_size(), width, elements,
                                 std::move(buffer)),
        std::move(lease));
  };
  FastNotifyResult result;
  result.expert_counts =
      make_completion(expert, ExchangeKind::ExpertCounts, expert_width);
  result.packet_summary = make_completion(
      summary, ExchangeKind::NotifyPacketSummary, summary_width);
  result.enqueue_us =
      std::chrono::duration<double, std::micro>(enqueue_end - enqueue_begin)
          .count();
  // The coordinator deliberately defers the only host wait until device
  // materialization has also been enqueued on this stream.
  result.wait_us = 0.0;
  return result;
}

double NcclAllGatherCountExchange::finish_fast_notify() {
  if (!impl_)
    invalid("phase=fast-notify-finish, field=backend, rank=n/a, "
            "expected=valid, actual=empty");
  ByteSlot *fast = nullptr;
  Slot *expert = nullptr;
  Slot *summary = nullptr;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->protocol != Impl::ProtocolState::FastNotifyStarted ||
        !impl_->fast_notify || !impl_->expert || !impl_->summary ||
        !impl_->fast_notify->in_use || !impl_->expert->in_use ||
        !impl_->summary->in_use)
      invalid("phase=fast-notify-finish, field=state, rank=" +
              std::to_string(impl_->session.rank()) +
              ", expected=pending fast notify, actual=not-pending");
    fast = impl_->fast_notify.get();
    expert = impl_->expert.get();
    summary = impl_->summary.get();
  }
  const auto release = [&]() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    fast->in_use = false;
    expert->in_use = false;
    summary->in_use = false;
  };
  const auto rank = impl_->session.rank();
  const auto &session = impl_->session.session();
  try {
    // Device materialization is ordered after this event on the same stream,
    // so in the normal path its completion wait has already completed the
    // collective. Keep this checked wait for standalone/failure paths.
    internal::throw_cuda(
        "fast-notify-finish", cudaEventSynchronize(fast->completion), rank,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "cudaEventSynchronize(fast notify)");
    ncclResult_t async = ncclSuccess;
    internal::throw_nccl(
        "fast-notify-finish", ncclCommGetAsyncError(impl_->comm, &async),
        rank, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::Control, "ncclCommGetAsyncError");
    internal::throw_nccl(
        "fast-notify-finish", async, rank, session.world_size(),
        session.identity(), session.generation(), ExchangeKind::Control,
        "NCCL asynchronous completion");
    ControlWireMatrix wires(session.world_size());
    for (std::uint32_t source = 0; source < session.world_size(); ++source)
      std::copy_n(fast->host_matrix +
                      static_cast<std::size_t>(source) * kControlWireBytes,
                  kControlWireBytes, wires[source].begin());
    (void)validate_control_wires(wires);
    release();
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->protocol = Impl::ProtocolState::Complete;
      ++impl_->audit.config_cache_hits;
    }
    return 0.0;
  } catch (...) {
    release();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->protocol = Impl::ProtocolState::ControlReady;
    throw;
  }
}

void NcclAllGatherCountExchange::reuse_verified_config(
    const RankCollectiveContext &context, const ConfigWire &wire,
    const ValidatedControlHandle &control) {
  if (!impl_ || !context || !control)
    invalid("phase=config-cache, field=input, rank=n/a, expected=valid, actual=empty");
  const auto rank = context.calling_rank();
  if (!context.session().same_instance(impl_->session.session()) ||
      !context.execution().same_instance(impl_->execution.token()) ||
      !control.context().same_instance(context))
    invalid("phase=config-cache, field=provenance, rank=" +
            std::to_string(rank) + ", expected=exact, actual=mismatch");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->protocol != Impl::ProtocolState::ConfigReady ||
      impl_->verified_config.empty() || impl_->verified_config != wire ||
      control.matrix().empty() ||
      control.matrix().front().expert_row_width !=
          impl_->verified_expert_width ||
      control.matrix().front().summary_row_width !=
          impl_->verified_summary_width)
    invalid("phase=config-cache, field=verified_config, rank=" +
            std::to_string(rank) +
            ", expected=setup-verified exact bytes/widths, actual=mismatch");
  impl_->agreed_expert_width = impl_->verified_expert_width;
  impl_->agreed_summary_width = impl_->verified_summary_width;
  impl_->protocol = Impl::ProtocolState::DataReady;
  ++impl_->audit.config_cache_hits;
}

CollectiveProtocolAudit NcclAllGatherCountExchange::audit() const {
  if (!impl_)
    invalid("phase=protocol-audit, field=backend, rank=n/a, expected=valid, actual=empty");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->audit;
}

NcclWorkspaceUsage NcclAllGatherCountExchange::workspace_usage() const {
  if (!impl_)
    invalid("phase=workspace-usage, field=backend, rank=n/a, expected=valid, actual=empty");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  NcclWorkspaceUsage usage;
  const auto slot_bytes = [](const Slot *slot) -> std::uint64_t {
    if (!slot)
      return 0;
    return static_cast<std::uint64_t>(slot->world) * slot->width *
           sizeof(std::uint64_t);
  };
  const auto byte_slot_matrix = [](const ByteSlot *slot) -> std::uint64_t {
    if (!slot)
      return 0;
    return static_cast<std::uint64_t>(slot->world) * slot->width;
  };
  const auto control_matrix = byte_slot_matrix(impl_->control.get());
  const auto control_row = kControlWireBytes;
  usage.control_slot_bytes = 2 * (control_matrix + control_row);
  const auto config_matrix = byte_slot_matrix(impl_->config.get());
  const auto config_row = impl_->config ? impl_->config->width : 0;
  usage.config_wire_bytes = config_row;
  usage.config_slot_bytes = 2 * (config_matrix + config_row);
  usage.expert_slot_bytes = slot_bytes(impl_->expert.get());
  usage.summary_slot_bytes = slot_bytes(impl_->summary.get());
  const auto fast_matrix = byte_slot_matrix(impl_->fast_notify.get());
  const auto fast_row = impl_->fast_notify ? impl_->fast_notify->width : 0;
  usage.fused_notify_slot_bytes = 2 * (fast_matrix + fast_row);
  const auto control_side = control_matrix + control_row;
  const auto config_side = config_matrix + config_row;
  const auto fast_side = fast_matrix + fast_row;
  usage.device_bytes = control_matrix + control_row + config_matrix +
                       config_row +
                       slot_bytes(impl_->expert.get()) +
                       slot_bytes(impl_->summary.get()) + fast_side;
  usage.pinned_host_bytes = control_side + config_side + fast_side;
  return usage;
}

std::unique_ptr<BackendCountExchangeOperation>
NcclAllGatherCountExchange::start(const CountExchangeRequest &request) {
  if (!impl_ || !request.context)
    invalid("phase=nccl-start, field=context/backend, rank=n/a, expected=valid, actual=empty");
  const auto rank = request.context.calling_rank();
  const auto &session = request.context.session();
  if (!session.same_instance(impl_->session.session()) ||
      !request.context.execution().same_instance(impl_->execution.token()))
    invalid("phase=nccl-start, field=context, rank=" + std::to_string(rank) +
            ", expected=backend session/execution instances, actual=mismatch");
  if (rank != impl_->session.rank() ||
      session.world_size() != impl_->session.session().world_size())
    invalid("phase=nccl-start, field=rank/world, rank=" + std::to_string(rank) +
            ", expected=backend rank/world, actual=mismatch");
  if (request.kind == ExchangeKind::Control)
    invalid("phase=nccl-start, field=kind, rank=" + std::to_string(rank) +
            ", expected=expert_counts/notify_packet_summary, actual=control");
  if (!request.device_input)
    invalid("phase=nccl-start, field=device_input, rank=" +
            std::to_string(rank) + ", expected=opaque CUDA lease, actual=empty");
  if (request.device_input.rows() != 1 ||
      request.device_input.row_width() != request.row_width)
    invalid("phase=nccl-start, field=device_input.shape, rank=" +
            std::to_string(rank) + ", expected=[1," +
            std::to_string(request.row_width) + "], actual=[" +
            std::to_string(request.device_input.rows()) + "," +
            std::to_string(request.device_input.row_width()) + "]");
  auto device = std::dynamic_pointer_cast<const internal::CudaBufferState>(
      backend_device_state(request.device_input));
  if (!device || !device->pointer ||
      device->device != impl_->session.device_ordinal())
    invalid("phase=nccl-start, field=device_input.backend, rank=" +
            std::to_string(rank) +
            ", expected=CUDA lease on backend device, actual=foreign/null");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto is_expert = request.kind == ExchangeKind::ExpertCounts;
  if (is_expert && impl_->protocol != Impl::ProtocolState::DataReady)
    invalid("phase=nccl-start, field=protocol_state, rank=" +
            std::to_string(rank) +
            ", expected=control-passed/data-ready, actual=not-ready");
  if (!is_expert &&
      impl_->protocol != Impl::ProtocolState::CountStarted &&
      impl_->protocol != Impl::ProtocolState::CountComplete)
    invalid("phase=nccl-start, field=protocol_state, rank=" +
            std::to_string(rank) +
            ", expected=expert-count-started/completed, actual=not-ready");
  Slot *slot = is_expert ? impl_->expert.get() : impl_->summary.get();
  if (!slot)
    invalid("phase=nccl-start, field=data_slot, rank=" +
            std::to_string(rank) +
            ", expected=allocated after control, actual=empty");
  if (request.row_width == 0 || request.row_width > slot->width)
    invalid("phase=nccl-start, field=row_width, rank=" +
            std::to_string(rank) + ", expected=(0," +
            std::to_string(slot->width) + "], actual=" +
            std::to_string(request.row_width));
  const auto agreed_width = is_expert ? impl_->agreed_expert_width
                                     : impl_->agreed_summary_width;
  if (request.row_width != agreed_width)
    invalid("phase=nccl-start, field=row_width.agreement, rank=" +
            std::to_string(rank) + ", expected=" +
            std::to_string(agreed_width) + ", actual=" +
            std::to_string(request.row_width));
  if (slot->in_use)
    invalid("phase=nccl-start, field=slot, rank=" + std::to_string(rank) +
            ", expected=available, actual=in-use, kind=" + kind_name(request.kind));
  slot->in_use = true;
  const auto prior_protocol = impl_->protocol;
  if (is_expert) {
    impl_->protocol = Impl::ProtocolState::CountStarted;
  } else {
    impl_->protocol = Impl::ProtocolState::SummaryStarted;
  }
  try {
    internal::throw_cuda(
        "nccl-start",
        cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())), rank,
        session.world_size(), session.identity(), session.generation(),
        request.kind, "cudaSetDevice");
    internal::throw_cuda(
        "nccl-start",
        cudaMemsetAsync(slot->device_matrix, 0,
                        checked_matrix_bytes(
                            checked_elements(slot->world, slot->width,
                                             "slot_capacity_elements"),
                            "slot_capacity_bytes"),
                        impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        request.kind, "cudaMemsetAsync(AllGather receive capacity)");
    internal::throw_nccl(
        "nccl-start",
        ncclAllGather(device->pointer, slot->device_matrix,
                      static_cast<std::size_t>(request.row_width), ncclUint64,
                      impl_->comm, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        request.kind, "ncclAllGather");
    if (is_expert)
      ++impl_->audit.expert_count_starts;
    else
      ++impl_->audit.packet_summary_starts;
    internal::throw_cuda(
        "nccl-start",
        cudaEventRecord(slot->completion, impl_->execution_state->stream), rank,
        session.world_size(), session.identity(), session.generation(),
        request.kind, "cudaEventRecord");
  } catch (...) {
    // No framework operation exists yet, so start itself must retain and drain
    // any partially enqueued work before the caller's typed input may unwind.
    (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
    (void)cudaStreamSynchronize(impl_->execution_state->stream);
    slot->in_use = false;
    impl_->protocol = prior_protocol;
    throw;
  }
  return std::make_unique<NcclAllGatherOperation>(impl_, slot, request);
}

#define MEGAMOE_STRINGIFY_DETAIL(x) #x
#define MEGAMOE_STRINGIFY(x) MEGAMOE_STRINGIFY_DETAIL(x)
const char *compiled_nccl_version() noexcept {
  return MEGAMOE_STRINGIFY(NCCL_MAJOR) "." MEGAMOE_STRINGIFY(NCCL_MINOR) "." MEGAMOE_STRINGIFY(NCCL_PATCH);
}

NcclRuntimeInfo nccl_runtime_info() {
  NcclRuntimeInfo info;
  info.compiled_version = compiled_nccl_version();
  int version = 0;
  internal::throw_nccl("runtime-info", ncclGetVersion(&version), 0, 1,
                       "runtime-info", 1, ExchangeKind::Control,
                       "ncclGetVersion");
  info.runtime_version = std::to_string(version / 10000) + "." +
                         std::to_string((version / 100) % 100) + "." +
                         std::to_string(version % 100);
  Dl_info library{};
  if (dladdr(reinterpret_cast<const void *>(&ncclGetVersion), &library) != 0 &&
      library.dli_fname)
    info.loaded_library_path = library.dli_fname;
  else
    info.loaded_library_path = "unknown";
  const auto environment = [](const char *name) {
    const auto *value = std::getenv(name);
    return value ? std::string(value) : std::string("<inherited/unset>");
  };
  info.socket_ifname = environment("NCCL_SOCKET_IFNAME");
  info.ib_disable = environment("NCCL_IB_DISABLE");
  return info;
}

