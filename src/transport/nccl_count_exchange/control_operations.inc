namespace {
std::vector<std::byte> make_nccl_stable_identity(
    const std::vector<std::uint8_t> &bootstrap, const std::string &identity) {
  constexpr char kDomain[] = "MegaMOE/NCCL/session/v1";
  std::vector<std::byte> result;
  const auto total = sizeof(kDomain) - 1U + 16U + bootstrap.size() +
                     identity.size();
  if (total > 4096U)
    invalid("phase=nccl-session-create, field=stable_identity_bytes, rank=n/a, expected<=4096, actual=" +
            std::to_string(total));
  result.reserve(total);
  for (std::size_t index = 0; index + 1U < sizeof(kDomain); ++index)
    result.push_back(static_cast<std::byte>(kDomain[index]));
  const auto append_length = [&](std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8U)
      result.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  };
  append_length(bootstrap.size());
  for (const auto byte : bootstrap)
    result.push_back(static_cast<std::byte>(byte));
  append_length(identity.size());
  for (const auto byte : identity)
    result.push_back(static_cast<std::byte>(byte));
  return result;
}
} // namespace

NcclSession make_nccl_session(std::string identity, std::uint32_t world_size,
                              std::uint32_t rank, std::uint64_t generation,
                              std::uint32_t device_ordinal,
                              const NcclBootstrapId &bootstrap_id) {
  if (identity.empty() || world_size == 0 || rank >= world_size ||
      generation == 0 || bootstrap_id.bytes_.size() != sizeof(ncclUniqueId))
    invalid("phase=nccl-session-create, field=arguments, rank=" +
            std::to_string(rank) +
            ", expected=valid identity/world/rank/generation/bootstrap, actual=invalid");
  int device_count = 0;
  internal::throw_cuda("session-create", cudaGetDeviceCount(&device_count),
                       rank, world_size, identity, generation,
                       ExchangeKind::ExpertCounts, "cudaGetDeviceCount");
  if (device_ordinal >= static_cast<std::uint32_t>(device_count))
    invalid("phase=nccl-session-create, field=device_ordinal, rank=" +
            std::to_string(rank) + ", expected=[0," +
            std::to_string(device_count) + "), actual=" +
            std::to_string(device_ordinal));
  internal::throw_cuda("session-create",
                       cudaSetDevice(static_cast<int>(device_ordinal)), rank,
                       world_size, identity, generation,
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  ncclUniqueId id{};
  std::memcpy(&id, bootstrap_id.bytes_.data(), sizeof(id));
  auto state = std::make_shared<internal::NcclSessionState>();
  state->device = device_ordinal;
  state->rank = rank;
  state->world = world_size;
  state->bootstrap_identity = bootstrap_id.bytes_;
  internal::throw_nccl("session-create",
                       ncclCommInitRank(&state->comm,
                                        static_cast<int>(world_size), id,
                                        static_cast<int>(rank)),
                       rank, world_size, identity, generation,
                       ExchangeKind::ExpertCounts, "ncclCommInitRank");
  auto impl = std::make_shared<NcclSession::Impl>();
  impl->state = state;
  auto stable_identity =
      make_nccl_stable_identity(bootstrap_id.bytes_, identity);
  impl->session = make_collective_session(
      std::move(identity), std::move(stable_identity), world_size, generation,
      state);
  return NcclSession(std::move(impl));
}

NcclSession make_nccl_generation(const NcclSession &base,
                                 std::string identity,
                                 std::uint64_t generation) {
  if (!base || identity.empty() || generation == 0)
    invalid("phase=nccl-generation-create, field=arguments, rank=n/a, expected=valid base/identity/generation, actual=invalid");
  auto impl = std::make_shared<NcclSession::Impl>();
  impl->state = base.impl_->state;
  auto stable_identity = make_nccl_stable_identity(
      impl->state->bootstrap_identity, identity);
  impl->session = make_collective_session(
      std::move(identity), std::move(stable_identity), impl->state->world,
      generation, impl->state);
  return NcclSession(std::move(impl));
}

const CollectiveSession &NcclSession::session() const {
  if (!impl_)
    invalid("attempted to access empty NCCL session");
  return impl_->session;
}
std::uint32_t NcclSession::rank() const {
  if (!impl_)
    invalid("attempted to access empty NCCL session");
  return impl_->state->rank;
}
std::uint32_t NcclSession::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty NCCL session");
  return impl_->state->device;
}
bool NcclSession::shares_communicator_with(
    const NcclSession &other) const noexcept {
  return impl_ && other.impl_ && impl_->state.get() == other.impl_->state.get();
}

struct NcclAllGatherCountExchange::Impl {
  NcclSession session;
  CudaExecution execution;
  std::shared_ptr<const internal::CudaExecutionState> execution_state;
  ncclComm_t comm = nullptr;
  std::unique_ptr<ByteSlot> control;
  std::unique_ptr<ByteSlot> config;
  std::unique_ptr<Slot> expert;
  std::unique_ptr<Slot> summary;
  std::unique_ptr<ByteSlot> fast_notify;
  std::uint64_t agreed_expert_width = 0;
  std::uint64_t agreed_summary_width = 0;
  ConfigWire verified_config;
  std::uint64_t verified_expert_width = 0;
  std::uint64_t verified_summary_width = 0;
  std::mutex mutex;
  enum class ProtocolState {
    ControlReady,
    ControlPending,
    ConfigReady,
    ConfigPending,
    DataReady,
    CountStarted,
    CountComplete,
    SummaryStarted,
    FastNotifyStarted,
    Complete,
  } protocol = ProtocolState::ControlReady;
  CollectiveProtocolAudit audit;
  GenerationSetupTimings setup;
};

namespace {

void allocate_slot(Slot &slot, std::uint64_t width,
                   const NcclSession &session, ExchangeKind kind,
                   GenerationSetupTimings *metrics = nullptr) {
  slot.width = width;
  slot.world = session.session().world_size();
  slot.device = session.device_ordinal();
  const auto elements = checked_elements(slot.world, width, "matrix_elements");
  const auto bytes = checked_matrix_bytes(elements, "matrix_bytes");
  internal::throw_cuda("nccl-workspace", cudaSetDevice(static_cast<int>(slot.device)),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind, "cudaSetDevice");
  const auto device_start = std::chrono::steady_clock::now();
  internal::throw_cuda("nccl-workspace",
                       cudaMalloc(reinterpret_cast<void **>(&slot.device_matrix),
                                  bytes),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind,
                       "cudaMalloc(AllGather matrix)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  const auto device_end = std::chrono::steady_clock::now();
  const auto event_start = device_end;
  internal::throw_cuda("nccl-workspace",
                       cudaEventCreateWithFlags(&slot.completion,
                                                cudaEventDisableTiming),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind,
                       "cudaEventCreate(AllGather)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  const auto event_end = std::chrono::steady_clock::now();
  slot.tracked = true;
  testing::g_slot_allocations.fetch_add(1);
  if (metrics) {
    metrics->device_allocation_us +=
        std::chrono::duration<double, std::micro>(device_end - device_start)
            .count();
    metrics->event_allocation_us +=
        std::chrono::duration<double, std::micro>(event_end - event_start)
            .count();
  }
}

void allocate_byte_slot(ByteSlot &slot, std::size_t width,
                        const NcclSession &session, const char *phase,
                        GenerationSetupTimings *metrics = nullptr) {
  if (width == 0 || width > kMaxCanonicalConfigBytes)
    invalid(std::string("phase=") + phase +
            ", field=width, rank=" + std::to_string(session.rank()) +
            ", expected=(0," + std::to_string(kMaxCanonicalConfigBytes) +
            "], actual=" + std::to_string(width));
  if (session.session().world_size() > SIZE_MAX / width)
    invalid(std::string("phase=") + phase +
            ", field=matrix_bytes, rank=" + std::to_string(session.rank()) +
            ", expected=size_t product, actual=overflow");
  slot.width = width;
  slot.world = session.session().world_size();
  slot.device = session.device_ordinal();
  const auto matrix_bytes = width * slot.world;
  internal::throw_cuda(phase, cudaSetDevice(static_cast<int>(slot.device)),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), ExchangeKind::Control,
                       "cudaSetDevice");
  const auto device_start = std::chrono::steady_clock::now();
  internal::throw_cuda(phase,
      cudaMalloc(reinterpret_cast<void **>(&slot.device_matrix), matrix_bytes),
      session.rank(), slot.world, session.session().identity(),
      session.session().generation(), ExchangeKind::Control,
      "cudaMalloc(byte matrix)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda(phase,
      cudaMalloc(reinterpret_cast<void **>(&slot.device_send), width),
      session.rank(), slot.world, session.session().identity(),
      session.session().generation(), ExchangeKind::Control,
      "cudaMalloc(byte send)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  const auto device_end = std::chrono::steady_clock::now();
  const auto pinned_start = device_end;
  internal::throw_cuda(phase,
      cudaHostAlloc(reinterpret_cast<void **>(&slot.host_matrix), matrix_bytes,
                    cudaHostAllocPortable),
      session.rank(), slot.world, session.session().identity(),
      session.session().generation(), ExchangeKind::Control,
      "cudaHostAlloc(byte matrix)");
  internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
  internal::throw_cuda(phase,
      cudaHostAlloc(reinterpret_cast<void **>(&slot.host_send), width,
                    cudaHostAllocPortable),
      session.rank(), slot.world, session.session().identity(),
      session.session().generation(), ExchangeKind::Control,
      "cudaHostAlloc(byte send)");
  internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
  const auto pinned_end = std::chrono::steady_clock::now();
  const auto event_start = pinned_end;
  internal::throw_cuda(phase,
      cudaEventCreateWithFlags(&slot.completion, cudaEventDisableTiming),
      session.rank(), slot.world, session.session().identity(),
      session.session().generation(), ExchangeKind::Control,
      "cudaEventCreate(byte slot)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  const auto event_end = std::chrono::steady_clock::now();
  slot.tracked = true;
  testing::g_slot_allocations.fetch_add(1);
  if (metrics) {
    metrics->device_allocation_us +=
        std::chrono::duration<double, std::micro>(device_end - device_start)
            .count();
    metrics->pinned_allocation_us +=
        std::chrono::duration<double, std::micro>(pinned_end - pinned_start)
            .count();
    metrics->event_allocation_us +=
        std::chrono::duration<double, std::micro>(event_end - event_start)
            .count();
  }
}

} // namespace

struct ValidatedControlHandle::Impl {
  ControlMatrix matrix;
  RankCollectiveContext context;
  std::shared_ptr<const void> owner;
};

const ControlMatrix &ValidatedControlHandle::matrix() const {
  if (!impl_)
    invalid("phase=control-handle, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->matrix;
}

const RankCollectiveContext &ValidatedControlHandle::context() const {
  if (!impl_)
    invalid("phase=control-handle, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->context;
}

struct NcclControlOperation::Impl {
  std::shared_ptr<NcclAllGatherCountExchange::Impl> backend;
  RankCollectiveContext context;
  ControlHeader input;
  ByteSlot *slot = nullptr;
  bool terminal = false;
  bool cancelled = false;
  bool released = false;
};

NcclControlOperation::NcclControlOperation() = default;
NcclControlOperation::~NcclControlOperation() noexcept {
  if (!impl_ || impl_->terminal)
    return;
  testing::g_destructor_drains.fetch_add(1);
  (void)cudaSetDevice(
      static_cast<int>(impl_->backend->session.device_ordinal()));
  (void)cudaEventSynchronize(impl_->slot->completion);
  std::lock_guard<std::mutex> lock(impl_->backend->mutex);
  impl_->slot->in_use = false;
  impl_->backend->protocol =
      NcclAllGatherCountExchange::Impl::ProtocolState::ControlReady;
  impl_->released = true;
  impl_->terminal = true;
}
NcclControlOperation::NcclControlOperation(NcclControlOperation &&) noexcept =
    default;
NcclControlOperation &NcclControlOperation::operator=(
    NcclControlOperation &&other) noexcept {
  if (this != &other) {
    if (impl_ && !impl_->terminal) {
      (void)cudaSetDevice(
          static_cast<int>(impl_->backend->session.device_ordinal()));
      (void)cudaEventSynchronize(impl_->slot->completion);
      std::lock_guard<std::mutex> lock(impl_->backend->mutex);
      impl_->slot->in_use = false;
      impl_->backend->protocol =
          NcclAllGatherCountExchange::Impl::ProtocolState::ControlReady;
      impl_->released = true;
      impl_->terminal = true;
    }
    impl_ = std::move(other.impl_);
  }
  return *this;
}

ExchangeState NcclControlOperation::state() const noexcept {
  if (!impl_)
    return ExchangeState::Failed;
  if (impl_->cancelled)
    return ExchangeState::Cancelled;
  if (impl_->terminal)
    return ExchangeState::Complete;
  (void)cudaSetDevice(static_cast<int>(impl_->backend->session.device_ordinal()));
  const auto status = cudaEventQuery(impl_->slot->completion);
  if (status == cudaSuccess)
    return ExchangeState::Complete;
  return status == cudaErrorNotReady ? ExchangeState::Pending
                                     : ExchangeState::Failed;
}

ValidatedControlHandle NcclControlOperation::wait() {
  if (!impl_ || impl_->terminal)
    invalid("phase=control-wait, field=operation, rank=n/a, expected=pending, actual=empty/terminal");
  testing::g_operation_waits.fetch_add(1);
  const auto &session = impl_->context.session();
  const auto rank = impl_->context.calling_rank();
  maybe_fault(testing::NcclFaultPoint::EventWait, "control-event-wait",
              impl_->context);
  internal::throw_cuda(
      "control-wait",
      cudaSetDevice(static_cast<int>(impl_->backend->session.device_ordinal())),
      rank, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::Control, "cudaSetDevice");
  internal::throw_cuda("control-wait",
                       cudaEventSynchronize(impl_->slot->completion), rank,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::Control,
                       "cudaEventSynchronize");
  ncclResult_t async = ncclSuccess;
  internal::throw_nccl("control-wait",
                       ncclCommGetAsyncError(impl_->backend->comm, &async),
                       rank, session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::Control,
                       "ncclCommGetAsyncError");
  internal::throw_nccl("control-wait", async, rank, session.world_size(),
                       session.identity(), session.generation(),
                       ExchangeKind::Control,
                       "NCCL asynchronous completion");
  ControlWireMatrix wires(session.world_size());
  for (std::uint32_t source = 0; source < session.world_size(); ++source) {
    const auto offset = static_cast<std::size_t>(source) * kControlWireBytes;
    std::copy_n(impl_->slot->host_matrix + offset, kControlWireBytes,
                wires[source].begin());
  }
  {
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    impl_->slot->in_use = false;
    impl_->released = true;
  }
  impl_->terminal = true;
  try {
    (void)validate_control_wires(wires);
  } catch (...) {
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    impl_->backend->protocol =
        NcclAllGatherCountExchange::Impl::ProtocolState::ControlReady;
    throw;
  }

  {
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    impl_->backend->protocol =
        NcclAllGatherCountExchange::Impl::ProtocolState::ConfigReady;
  }
  auto result = std::make_shared<ValidatedControlHandle::Impl>();
  result->matrix = validate_control_wires(wires);
  result->context = impl_->context;
  result->owner = impl_->backend;
  return ValidatedControlHandle(std::move(result));
}

void NcclControlOperation::cancel() {
  if (!impl_ || impl_->terminal)
    return;
  testing::g_operation_cancels.fetch_add(1);
  const auto &session = impl_->context.session();
  const auto rank = impl_->context.calling_rank();
  internal::throw_cuda(
      "control-cancel",
      cudaSetDevice(static_cast<int>(impl_->backend->session.device_ordinal())),
      rank, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::Control, "cudaSetDevice");
  internal::throw_cuda("control-cancel",
                       cudaEventSynchronize(impl_->slot->completion), rank,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::Control,
                       "cudaEventSynchronize");
  std::lock_guard<std::mutex> lock(impl_->backend->mutex);
  impl_->slot->in_use = false;
  impl_->backend->protocol =
      NcclAllGatherCountExchange::Impl::ProtocolState::ControlReady;
  impl_->released = true;
  impl_->cancelled = true;
  impl_->terminal = true;
}

struct NcclConfigOperation::Impl {
  std::shared_ptr<NcclAllGatherCountExchange::Impl> backend;
  RankCollectiveContext context;
  ConfigWire input;
  ControlMatrix headers;
  ByteSlot *slot = nullptr;
  std::size_t collective_width = 0;
  bool terminal = false;
};

NcclConfigOperation::NcclConfigOperation() = default;
NcclConfigOperation::~NcclConfigOperation() noexcept {
  if (!impl_ || impl_->terminal)
    return;
  testing::g_destructor_drains.fetch_add(1);
  (void)cudaSetDevice(
      static_cast<int>(impl_->backend->session.device_ordinal()));
  (void)cudaEventSynchronize(impl_->slot->completion);
  std::lock_guard<std::mutex> lock(impl_->backend->mutex);
  impl_->slot->in_use = false;
  impl_->backend->protocol =
      NcclAllGatherCountExchange::Impl::ProtocolState::ConfigReady;
  impl_->terminal = true;
}
NcclConfigOperation::NcclConfigOperation(NcclConfigOperation &&) noexcept =
    default;
NcclConfigOperation &
NcclConfigOperation::operator=(NcclConfigOperation &&other) noexcept {
  if (this == &other)
    return *this;
  if (impl_ && !impl_->terminal) {
    (void)cudaSetDevice(
        static_cast<int>(impl_->backend->session.device_ordinal()));
    (void)cudaEventSynchronize(impl_->slot->completion);
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    impl_->slot->in_use = false;
    impl_->backend->protocol =
        NcclAllGatherCountExchange::Impl::ProtocolState::ConfigReady;
    impl_->terminal = true;
  }
  impl_ = std::move(other.impl_);
  return *this;
}

ValidatedConfigHandle NcclConfigOperation::wait() {
  if (!impl_ || impl_->terminal)
    invalid("phase=config-wait, field=operation, rank=n/a, expected=pending, actual=empty/terminal");
  testing::g_operation_waits.fetch_add(1);
  const auto &session = impl_->context.session();
  const auto rank = impl_->context.calling_rank();
  internal::throw_cuda("config-wait",
                       cudaSetDevice(static_cast<int>(
                           impl_->backend->session.device_ordinal())),
                       rank, session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::Control,
                       "cudaSetDevice");
  internal::throw_cuda("config-wait",
                       cudaEventSynchronize(impl_->slot->completion), rank,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::Control,
                       "cudaEventSynchronize");
  ncclResult_t async = ncclSuccess;
  internal::throw_nccl("config-wait",
                       ncclCommGetAsyncError(impl_->backend->comm, &async),
                       rank, session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::Control,
                       "ncclCommGetAsyncError");
  internal::throw_nccl("config-wait", async, rank, session.world_size(),
                       session.identity(), session.generation(),
                       ExchangeKind::Control, "NCCL asynchronous completion");
  std::vector<ConfigWire> matrix(session.world_size());
  for (std::uint32_t source = 0; source < session.world_size(); ++source) {
    const auto offset = static_cast<std::size_t>(source) *
                        impl_->collective_width;
    const auto row_bytes = impl_->headers[source].config_blob_bytes;
    if (row_bytes > impl_->collective_width || row_bytes > SIZE_MAX)
      invalid("phase=config-wait, field=config_blob_bytes, rank=" +
              std::to_string(rank) +
              ", expected<=collective slot width, actual=" +
              std::to_string(row_bytes));
    matrix[source].resize(static_cast<std::size_t>(row_bytes));
    std::copy_n(impl_->slot->host_matrix + offset,
                static_cast<std::size_t>(row_bytes),
                matrix[source].begin());
  }
  {
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    impl_->slot->in_use = false;
  }
  impl_->terminal = true;
  try {
    validate_config_matrix(matrix, impl_->headers);
  } catch (...) {
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    impl_->backend->protocol =
        NcclAllGatherCountExchange::Impl::ProtocolState::ControlReady;
    throw;
  }

  std::unique_ptr<Slot> expert;
  std::unique_ptr<Slot> summary;
  maybe_fault(testing::NcclFaultPoint::CountSlotAllocation,
              "count-slot-allocation", impl_->context);
  maybe_fault(testing::NcclFaultPoint::SummarySlotAllocation,
              "summary-slot-allocation", impl_->context);
  if (!impl_->backend->expert || !impl_->backend->summary) {
    expert = std::make_unique<Slot>();
    const auto count_start = std::chrono::steady_clock::now();
    allocate_slot(*expert, impl_->headers.front().expert_row_width,
                  impl_->backend->session, ExchangeKind::ExpertCounts,
                  &impl_->backend->setup);
    const auto count_end = std::chrono::steady_clock::now();
    impl_->backend->setup.count_slot_allocation_us +=
        std::chrono::duration<double, std::micro>(count_end - count_start)
            .count();
    summary = std::make_unique<Slot>();
    const auto summary_start = std::chrono::steady_clock::now();
    allocate_slot(*summary, impl_->headers.front().summary_row_width,
                  impl_->backend->session,
                  ExchangeKind::NotifyPacketSummary,
                  &impl_->backend->setup);
    const auto summary_end = std::chrono::steady_clock::now();
    impl_->backend->setup.summary_slot_allocation_us +=
        std::chrono::duration<double, std::micro>(summary_end - summary_start)
            .count();
  } else if (impl_->headers.front().expert_row_width >
                 impl_->backend->expert->width ||
             impl_->headers.front().summary_row_width >
                 impl_->backend->summary->width) {
    invalid("phase=config-wait, field=preallocated_data_slot, rank=" +
            std::to_string(rank) +
            ", expected=capacity >= exact agreed widths, actual=short");
  }
  {
    std::lock_guard<std::mutex> lock(impl_->backend->mutex);
    if (expert)
      impl_->backend->expert = std::move(expert);
    if (summary)
      impl_->backend->summary = std::move(summary);
    impl_->backend->agreed_expert_width =
        impl_->headers.front().expert_row_width;
    impl_->backend->agreed_summary_width =
        impl_->headers.front().summary_row_width;
    impl_->backend->verified_config = impl_->input;
    impl_->backend->verified_expert_width =
        impl_->headers.front().expert_row_width;
    impl_->backend->verified_summary_width =
        impl_->headers.front().summary_row_width;
    impl_->backend->protocol =
        NcclAllGatherCountExchange::Impl::ProtocolState::DataReady;
  }
  ValidatedConfigHandle result;
  result.owner_ = impl_->backend;
  return result;
}

void NcclConfigOperation::cancel() {
  if (!impl_ || impl_->terminal)
    return;
  testing::g_operation_cancels.fetch_add(1);
  (void)cudaSetDevice(
      static_cast<int>(impl_->backend->session.device_ordinal()));
  (void)cudaEventSynchronize(impl_->slot->completion);
  std::lock_guard<std::mutex> lock(impl_->backend->mutex);
  impl_->slot->in_use = false;
  impl_->backend->protocol =
      NcclAllGatherCountExchange::Impl::ProtocolState::ConfigReady;
  impl_->terminal = true;
}

