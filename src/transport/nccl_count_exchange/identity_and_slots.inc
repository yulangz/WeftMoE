
#if !MEGAMOE_BUILD_TESTING
// The release object graph keeps only the counters needed by the production
// coordinator. Repository fault-control entry points are section-collected;
// using a distinct namespace also makes accidental retention auditable.
namespace release_instrumentation {
using NcclFaultPoint = ::megamoe::cuda::testing::NcclFaultPoint;
using NcclLifecycleSnapshot =
    ::megamoe::cuda::testing::NcclLifecycleSnapshot;
} // namespace release_instrumentation
#define testing release_instrumentation
#endif

namespace testing {
namespace {
std::atomic<int> g_next_fault{0};
thread_local std::size_t g_config_corruption_offset = SIZE_MAX;
struct IdentityOverride {
  std::vector<std::byte> raw;
  StableSessionId summary;
};
thread_local std::unique_ptr<IdentityOverride> g_identity_override;
std::atomic<std::uint64_t> g_slot_allocations{0};
std::atomic<std::uint64_t> g_slot_destructions{0};
std::atomic<std::uint64_t> g_operation_waits{0};
std::atomic<std::uint64_t> g_operation_cancels{0};
std::atomic<std::uint64_t> g_destructor_drains{0};
std::atomic<std::uint64_t> g_completion_lease_creates{0};
std::atomic<std::uint64_t> g_completion_lease_destroys{0};
std::atomic<std::uint64_t> g_validated_publishes{0};
}
void inject_next_nccl_fault(NcclFaultPoint point) noexcept {
  g_next_fault.store(static_cast<int>(point), std::memory_order_release);
}
void corrupt_next_config_wire(std::size_t offset) noexcept {
  g_config_corruption_offset = offset;
}
void override_next_protocol_identity(
    std::vector<std::byte> raw_identity,
    StableSessionId diagnostic_summary) noexcept {
  try {
    g_identity_override = std::make_unique<IdentityOverride>();
    g_identity_override->raw = std::move(raw_identity);
    g_identity_override->summary = diagnostic_summary;
  } catch (...) {
    g_identity_override.reset();
  }
}
void reset_nccl_lifecycle_counters() noexcept {
  g_slot_allocations.store(0);
  g_slot_destructions.store(0);
  g_operation_waits.store(0);
  g_operation_cancels.store(0);
  g_destructor_drains.store(0);
  g_completion_lease_creates.store(0);
  g_completion_lease_destroys.store(0);
  g_validated_publishes.store(0);
}
NcclLifecycleSnapshot nccl_lifecycle_snapshot() noexcept {
  return {g_slot_allocations.load(), g_slot_destructions.load(),
          g_operation_waits.load(), g_operation_cancels.load(),
          g_destructor_drains.load(), g_completion_lease_creates.load(),
          g_completion_lease_destroys.load(), g_validated_publishes.load()};
}
} // namespace testing

namespace {

[[noreturn]] void invalid(const std::string &message) {
  throw NotifyDispatchError(NotifyErrorCode::NcclFailure, message);
}

#if MEGAMOE_BUILD_TESTING
void maybe_fault(testing::NcclFaultPoint point, const char *stage,
                 const RankCollectiveContext &context) {
  auto expected = static_cast<int>(point);
  if (!testing::g_next_fault.compare_exchange_strong(
          expected, 0, std::memory_order_acq_rel))
    return;
  std::ostringstream message;
  message << "phase=fault-injection, stage=" << stage
          << ", rank=" << context.calling_rank()
          << ", world=" << context.session().world_size() << ", session='"
          << context.session().identity() << "', generation="
          << context.session().generation()
          << ", backend_error=controlled test fault";
  throw NotifyDispatchError(NotifyErrorCode::NcclFailure, message.str());
}
#else
template <typename Point>
constexpr void maybe_fault(Point, const char *,
                           const RankCollectiveContext &) noexcept {}
#endif

std::size_t checked_elements(std::uint32_t world, std::uint64_t width,
                             const char *field) {
  if (width == 0 || width > std::numeric_limits<std::size_t>::max() ||
      world > std::numeric_limits<std::size_t>::max() /
                  static_cast<std::size_t>(width)) {
    invalid(std::string("phase=nccl-workspace, field=") + field +
            ", rank=n/a, expected=nonzero product within size_t, actual=overflow/zero");
  }
  return static_cast<std::size_t>(world) * static_cast<std::size_t>(width);
}

std::size_t checked_matrix_bytes(std::size_t elements, const char *field) {
  if (elements > std::numeric_limits<std::size_t>::max() /
                     sizeof(std::uint64_t))
    invalid(std::string("phase=nccl-workspace, field=") + field +
            ", rank=n/a, expected=byte size within size_t, actual=overflow");
  return elements * sizeof(std::uint64_t);
}

struct Slot {
  std::uint64_t width = 0;
  std::uint32_t world = 0;
  std::uint32_t device = 0;
  std::uint64_t *device_matrix = nullptr;
  std::uint64_t *device_send = nullptr;
  cudaEvent_t completion = nullptr;
  bool in_use = false;
  bool tracked = false;

  ~Slot() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (completion)
      (void)cudaEventSynchronize(completion);
    if (completion)
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
    if (completion)
      (void)cudaEventDestroy(completion);
    if (device_matrix)
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
    if (device_matrix)
      (void)cudaFree(device_matrix);
    if (device_send)
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
    if (device_send)
      (void)cudaFree(device_send);
    if (tracked)
      testing::g_slot_destructions.fetch_add(1);
  }
};

struct ByteSlot {
  std::size_t width = 0;
  std::uint32_t world = 0;
  std::uint32_t device = 0;
  std::byte *device_matrix = nullptr;
  std::byte *device_send = nullptr;
  std::byte *host_matrix = nullptr;
  std::byte *host_send = nullptr;
  cudaEvent_t completion = nullptr;
  bool in_use = false;
  bool tracked = false;

  ~ByteSlot() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (completion)
      (void)cudaEventSynchronize(completion);
    if (completion)
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
    if (completion)
      (void)cudaEventDestroy(completion);
    if (host_matrix)
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
    if (host_matrix)
      (void)cudaFreeHost(host_matrix);
    if (host_send)
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
    if (host_send)
      (void)cudaFreeHost(host_send);
    if (device_matrix)
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
    if (device_matrix)
      (void)cudaFree(device_matrix);
    if (device_send)
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
    if (device_send)
      (void)cudaFree(device_send);
    if (tracked)
      testing::g_slot_destructions.fetch_add(1);
  }
};

const char *kind_name(ExchangeKind kind) noexcept {
  switch (kind) {
  case ExchangeKind::Control: return "control";
  case ExchangeKind::ExpertCounts: return "expert_counts";
  case ExchangeKind::NotifyPacketSummary: return "notify_packet_summary";
  }
  return "unknown";
}

} // namespace

namespace internal {

CudaCompletionLease::CudaCompletionLease() noexcept {
  testing::g_completion_lease_creates.fetch_add(1);
}
CudaCompletionLease::~CudaCompletionLease() noexcept {
  testing::g_completion_lease_destroys.fetch_add(1);
}

NcclSessionState::~NcclSessionState() noexcept {
  if (!comm)
    return;
  (void)cudaSetDevice(static_cast<int>(device));
  ncclResult_t async = ncclSuccess;
  const auto query = ncclCommGetAsyncError(comm, &async);
  if (query == ncclSuccess && async == ncclSuccess)
    (void)ncclCommFinalize(comm);
  else
    (void)ncclCommAbort(comm);
  (void)ncclCommDestroy(comm);
}

void throw_nccl(const char *phase, ncclResult_t error, std::uint32_t rank,
                std::uint32_t world, const std::string &session,
                std::uint64_t generation, ExchangeKind kind,
                const char *operation) {
  if (error == ncclSuccess)
    return;
  std::ostringstream stream;
  stream << "phase=" << phase << ", kind=" << kind_name(kind)
         << ", rank=" << rank << ", world=" << world << ", session='"
         << session << "', generation=" << generation << ", operation="
         << operation << ", nccl_error=" << static_cast<int>(error) << " ("
         << ncclGetErrorString(error) << ")";
  throw NotifyDispatchError(NotifyErrorCode::NcclFailure, stream.str());
}

} // namespace internal

NcclBootstrapId make_nccl_bootstrap_id() {
  static_assert(sizeof(ncclUniqueId) ==
                    NcclBootstrapInputChannel::kWireBytes,
                "bootstrap frame width must match ncclUniqueId");
  ncclUniqueId id{};
  internal::throw_nccl("bootstrap", ncclGetUniqueId(&id), 0, 1,
                       "bootstrap", 1, ExchangeKind::ExpertCounts,
                       "ncclGetUniqueId");
  std::vector<std::uint8_t> bytes(sizeof(id));
  std::memcpy(bytes.data(), &id, sizeof(id));
  return NcclBootstrapId(std::move(bytes));
}

struct NcclSession::Impl {
  std::shared_ptr<internal::NcclSessionState> state;
  CollectiveSession session;
};

std::unique_ptr<BackendCountExchangeOperation>
internal::NcclSessionAccess::start(const CountExchangeRequest &) {
  invalid("phase=nccl-session-access, field=start, rank=n/a, "
          "expected=never called, actual=called");
}

std::shared_ptr<internal::NcclSessionState>
internal::NcclSessionAccess::checked(const NcclSession &session) {
  if (!session)
    throw NotifyDispatchError(
        NotifyErrorCode::NcclFailure,
        "phase=nccl-session-access, field=session, rank=n/a, "
        "expected=live NCCL session, actual=empty");
  auto state = std::dynamic_pointer_cast<const internal::NcclSessionState>(
      backend_session_state(session.session()));
  if (!state || !state->comm)
    throw NotifyDispatchError(
        NotifyErrorCode::NcclFailure,
        "phase=nccl-session-access, field=session_backend, rank=n/a, "
        "expected=live NCCL session, actual=foreign/null");
  return std::const_pointer_cast<internal::NcclSessionState>(state);
}

NcclSession::NcclSession(const NcclSession &) = default;
NcclSession::NcclSession(NcclSession &&) noexcept = default;
NcclSession &NcclSession::operator=(const NcclSession &) = default;
NcclSession &NcclSession::operator=(NcclSession &&) noexcept = default;
NcclSession::~NcclSession() = default;

