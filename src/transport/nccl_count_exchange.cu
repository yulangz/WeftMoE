#include "megamoe/cuda/nccl_count_exchange.hpp"

#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cuda/cuda_internal.hpp"
#include "cuda/device_notify_internal.hpp"
#include "cuda/nccl_protocol_internal.hpp"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {

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
                       ncclCommInitRank(&state->comm, static_cast<int>(world_size),
                                        id, static_cast<int>(rank)),
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
  std::uint64_t agreed_expert_width = 0;
  std::uint64_t agreed_summary_width = 0;
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
                    (impl_->summary && impl_->summary->in_use);
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
  const auto control_side = control_matrix + control_row;
  const auto config_side = config_matrix + config_row;
  usage.device_bytes = control_matrix + control_row + config_matrix +
                       config_row +
                       slot_bytes(impl_->expert.get()) +
                       slot_bytes(impl_->summary.get());
  usage.pinned_host_bytes = control_side + config_side;
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

struct DeviceNotifyResourcePool::Impl
    : std::enable_shared_from_this<DeviceNotifyResourcePool::Impl> {
  struct SlotState {
    bool in_use = false;
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
      if (drained && slot.in_use && slot.epoch == epoch) {
        slot.in_use = false;
        if (pool->in_use != 0)
          --pool->in_use;
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
      if (slot.in_use)
        continue;
      slot.in_use = true;
      slot.epoch = ++epoch_high;
      ++in_use;
      try {
        slot.backend->prepare_generation(generation);
      } catch (...) {
        slot.in_use = false;
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
  usage.slots_in_use = impl_->in_use;
  usage.slot_epoch_high_watermark = impl_->epoch_high;
  usage.one_time_setup_us = impl_->setup_us;
  for (const auto &slot : impl_->slots) {
    usage.device_bytes += DeviceNotifyPoolBridge::device_bytes(slot.notify);
    usage.pinned_host_bytes +=
        DeviceNotifyPoolBridge::pinned_bytes(slot.notify);
    const auto backend = slot.backend->workspace_usage();
    usage.device_bytes += backend.device_bytes;
    usage.pinned_host_bytes += backend.pinned_host_bytes;
    usage.event_count +=
        4U + DeviceNotifyPoolBridge::event_count(slot.notify);
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
};

namespace {

LocalPreflightStatus host_preflight_status(
    const RuntimeConfig &config, const RankRoutingInput &routes,
    std::uint64_t max_tokens_per_rank,
    const CudaRouteInput *device_routes = nullptr) noexcept {
  LocalPreflightStatus status;
  const auto fail = [&](ControlLocalError error, ControlField field,
                        std::uint64_t token, std::uint64_t slot,
                        std::int64_t expert, std::uint64_t expected,
                        std::uint64_t actual) {
    status = {error, field, token, slot, expert, expected, actual};
  };
  try {
    validate_runtime_config(config);
  } catch (...) {
    fail(ControlLocalError::InvalidRuntimeConfig,
         ControlField::LocalValidation, UINT64_MAX, UINT64_MAX, 0, 1, 0);
    return status;
  }
  if (routes.num_tokens > max_tokens_per_rank) {
    fail(ControlLocalError::CapacityExceeded, ControlField::NumTokens,
         UINT64_MAX, UINT64_MAX, 0, max_tokens_per_rank, routes.num_tokens);
    return status;
  }
  if (config.top_k != 0 && routes.num_tokens > UINT64_MAX / config.top_k) {
    fail(ControlLocalError::ArithmeticOverflow, ControlField::RouteCount,
         UINT64_MAX, UINT64_MAX, 0, UINT64_MAX, UINT64_MAX);
    return status;
  }
  const auto expected = routes.num_tokens * config.top_k;
  const auto actual_route_count = device_routes
                                      ? device_routes->route_count()
                                      : routes.selected_experts.size();
  if (actual_route_count != expected ||
      (device_routes &&
       (device_routes->num_tokens() != routes.num_tokens ||
        device_routes->top_k() != config.top_k))) {
    fail(ControlLocalError::InvalidRouteShape, ControlField::RouteCount,
         UINT64_MAX, UINT64_MAX, 0, expected,
         actual_route_count);
    return status;
  }
  if (device_routes)
    return status;
  for (std::uint64_t token = 0; token < routes.num_tokens; ++token) {
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
      const auto index = token * config.top_k + slot;
      const auto expert = routes.selected_experts[static_cast<std::size_t>(index)];
      if (expert < 0 || static_cast<std::uint64_t>(expert) >= config.num_experts) {
        fail(ControlLocalError::ExpertOutOfRange,
             ControlField::LocalValidation, token, slot, expert,
             config.num_experts, static_cast<std::uint64_t>(expert));
        return status;
      }
      for (std::uint32_t earlier = 0; earlier < slot; ++earlier) {
        const auto earlier_index = token * config.top_k + earlier;
        if (routes.selected_experts[static_cast<std::size_t>(earlier_index)] ==
            expert) {
          fail(ControlLocalError::DuplicateExpert,
               ControlField::LocalValidation, token, slot, expert, 0,
               static_cast<std::uint64_t>(expert));
          return status;
        }
      }
    }
  }
  return status;
}

} // namespace

CudaNotifyGenerationCoordinator::CudaNotifyGenerationCoordinator(
    const NcclSession &session, const CudaExecution &execution) {
  if (!session || !execution)
    invalid("phase=generation-coordinator-create, field=input, rank=n/a, expected=valid session/execution, actual=empty");
  const auto start = std::chrono::steady_clock::now();
  auto impl = std::make_shared<Impl>();
  impl->session = session;
  impl->execution = execution;
  impl->context = make_rank_collective_context(
      session.session(), session.rank(), execution.token());
  const auto backend_start = std::chrono::steady_clock::now();
  impl->backend =
      std::make_shared<NcclAllGatherCountExchange>(session, execution);
  const auto backend_end = std::chrono::steady_clock::now();
  impl->setup = impl->backend->impl_->setup;
  impl->setup.coordinator_backend_create_us =
      std::chrono::duration<double, std::micro>(backend_end - backend_start)
          .count();
  impl->setup.per_generation_setup_total_us =
      std::chrono::duration<double, std::micro>(backend_end - start).count();
  impl_ = std::move(impl);
}

CudaNotifyGenerationCoordinator::CudaNotifyGenerationCoordinator(
    const NcclSession &session, const CudaExecution &execution,
    const DeviceNotifyResourcePool &pool) {
  if (!session || !execution || !pool)
    invalid("phase=generation-coordinator-create, field=input, rank=n/a, "
            "expected=valid session/execution/pool, actual=empty");
  if (!session.shares_communicator_with(pool.impl_->base_session) ||
      !execution.token().same_instance(pool.impl_->execution.token()))
    invalid("phase=generation-coordinator-create, field=pool_provenance, rank=" +
            std::to_string(session.rank()) +
            ", expected=same communicator/execution, actual=mismatch");
  auto impl = std::make_shared<Impl>();
  impl->session = session;
  impl->execution = execution;
  impl->context = make_rank_collective_context(
      session.session(), session.rank(), execution.token());
  impl->pool = pool.impl_;
  impl->setup.one_time_pool_setup_us = pool.impl_->setup_us;
  impl_ = std::move(impl);
}

CudaNotifyGenerationCoordinator::~CudaNotifyGenerationCoordinator() noexcept =
    default;
CudaNotifyGenerationCoordinator::CudaNotifyGenerationCoordinator(
    CudaNotifyGenerationCoordinator &&) noexcept = default;
CudaNotifyGenerationCoordinator &
CudaNotifyGenerationCoordinator::operator=(
    CudaNotifyGenerationCoordinator &&) noexcept = default;

namespace testing {
CudaNotifyGenerationCoordinator make_unpooled_coordinator_for_testing(
    const NcclSession &session, const CudaExecution &execution) {
  return CudaNotifyGenerationCoordinator(session, execution);
}
} // namespace testing

ValidatedCudaGeneration CudaNotifyGenerationCoordinator::run(
    NotifyGenerationRequest request) {
  if (!impl_ || impl_->used)
    invalid("phase=generation-coordinator-run, field=state, rank=n/a, expected=fresh coordinator, actual=empty/already-used");
  impl_->used = true;
  const auto run_start = std::chrono::steady_clock::now();

  std::shared_ptr<DeviceNotifyResourcePool::Impl::Lease> pool_lease;
  std::shared_ptr<CudaNotifyPoolAllocation> pool_allocation;
  if (impl_->pool) {
    const auto acquire_start = std::chrono::steady_clock::now();
    pool_lease = impl_->pool->acquire(impl_->session);
    auto &slot = impl_->pool->slots[pool_lease->index];
    pool_allocation = slot.notify;
    impl_->backend = slot.backend;
    impl_->setup.pool_acquire_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - acquire_start)
            .count();
  }
  if (!impl_->backend)
    invalid("phase=generation-coordinator-run, field=backend, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=available, actual=empty");

  ConfigWire config_wire;
  LocalPreflightStatus config_status;
  auto protocol_identity = impl_->context.session().stable_identity_bytes();
  std::unique_ptr<testing::IdentityOverride> identity_override;
  if (testing::g_identity_override) {
    identity_override = std::move(testing::g_identity_override);
    protocol_identity = identity_override->raw;
  }
  (void)try_encode_canonical_config(
      request.config, request.max_tokens_per_rank, config_wire, config_status,
      protocol_identity);
  const auto *direct_routes = impl_->direct_routes
                                  ? &impl_->direct_routes
                                  : static_cast<const CudaRouteInput *>(nullptr);
  auto status = host_preflight_status(request.config, request.local_routes,
                                      request.max_tokens_per_rank,
                                      direct_routes);
  if (status.error == ControlLocalError::None)
    status = config_status;
  if (status.error == ControlLocalError::None && direct_routes) {
    try {
      DeviceRouteProvenanceBridge::validate(
          *direct_routes, impl_->context, impl_->execution);
    } catch (const NotifyDispatchError &) {
      status = {ControlLocalError::InvalidRouteShape,
                ControlField::LocalValidation, UINT64_MAX, UINT64_MAX, 0,
                1, 0};
    }
  }
  if (status.error == ControlLocalError::None && impl_->pool) {
    const auto &capacity = impl_->pool->config;
    if (request.config.num_experts > capacity.capacity_config.num_experts ||
        request.config.top_k > capacity.capacity_config.top_k ||
        request.config.topology.world_size !=
            capacity.capacity_config.topology.world_size ||
        request.config.topology.num_nodes >
            capacity.capacity_config.topology.num_nodes ||
        request.max_tokens_per_rank > capacity.max_tokens_per_rank ||
        config_wire.size() > capacity.max_config_wire_bytes) {
      status = {ControlLocalError::CapacityExceeded,
                ControlField::LocalValidation, UINT64_MAX, UINT64_MAX, 0,
                capacity.max_tokens_per_rank,
                request.max_tokens_per_rank};
    }
  }

  CudaLocalNotifyHandle local;
  double local_mapping_us = 0;
  std::uint64_t route_input_bytes = 0;
  std::uint64_t notify_arena_bytes = 0;
  std::uint64_t notify_pinned_bytes = 0;
  if (status.error == ControlLocalError::None) {
    try {
      auto routes = direct_routes
                        ? *direct_routes
                        : pool_allocation
                        ? DeviceNotifyPoolBridge::upload(
                              request.local_routes, request.config.top_k,
                              impl_->context,
                              impl_->execution, pool_allocation, pool_lease)
                        : upload_cuda_routes(request.local_routes,
                                             request.config.top_k,
                                             impl_->context,
                                             impl_->execution);
      route_input_bytes = routes.device_bytes();
      auto plan = query_cuda_notify_workspace(
          request.config, impl_->session.rank(),
          request.local_routes.num_tokens, request.max_tokens_per_rank);
      notify_arena_bytes = plan.device_bytes;
      notify_pinned_bytes = plan.host_staging_bytes;
      auto workspace = pool_allocation
                           ? DeviceNotifyPoolBridge::workspace(
                                 plan, impl_->context, impl_->execution,
                                 pool_allocation, pool_lease)
                           : allocate_cuda_notify_workspace(
                                 plan, impl_->context, impl_->execution);
      auto operation = start_cuda_rank_local_notify(
          request.config, impl_->context, impl_->execution, routes,
          std::move(workspace));
      auto preflight = operation.wait_preflight();
      local_mapping_us = preflight.device_mapping_us();
      status = preflight.status();
      if (preflight.valid())
        local = preflight.local();
    } catch (const NotifyDispatchError &) {
      // Predictable input/config failures must still join fixed control. A
      // CUDA runtime/context failure is not predictable and remains fatal.
      status = host_preflight_status(request.config, request.local_routes,
                                     request.max_tokens_per_rank,
                                     direct_routes);
      if (status.error == ControlLocalError::None)
        throw;
    }
  }

  auto header = make_control_header(
      request.config, impl_->context, request.local_routes.num_tokens,
      direct_routes ? direct_routes->route_count()
                    : request.local_routes.selected_experts.size(),
      request.max_tokens_per_rank, config_wire.size(),
      config_wire_digest(config_wire), status);
  if (identity_override) {
    header.session_id_word0 = identity_override->summary.word0;
    header.session_id_word1 = identity_override->summary.word1;
  }
  const auto control_config_start = std::chrono::steady_clock::now();
  auto control = impl_->backend->start_control(impl_->context, header);
  auto control_done = control.wait();

  if (testing::g_config_corruption_offset != SIZE_MAX) {
    const auto offset = testing::g_config_corruption_offset;
    testing::g_config_corruption_offset = SIZE_MAX;
    if (offset < config_wire.size())
      config_wire[offset] ^= std::byte{1};
  }

  auto config = impl_->backend->start_config(impl_->context, config_wire,
                                             control_done);
  (void)config.wait();
  const auto control_config_end = std::chrono::steady_clock::now();
  const auto coordinator_create = impl_->setup.coordinator_backend_create_us;
  const auto initial_total = impl_->setup.per_generation_setup_total_us;
  const auto pool_setup = impl_->setup.one_time_pool_setup_us;
  const auto pool_acquire = impl_->setup.pool_acquire_us;
  if (impl_->pool) {
    impl_->setup = {};
    impl_->setup.one_time_pool_setup_us = pool_setup;
    impl_->setup.pool_acquire_us = pool_acquire;
    impl_->setup.per_generation_setup_total_us = pool_acquire;
  } else {
    impl_->setup = impl_->backend->impl_->setup;
    impl_->setup.coordinator_backend_create_us = coordinator_create;
    impl_->setup.per_generation_setup_total_us =
        initial_total + impl_->setup.config_slot_allocation_us +
        impl_->setup.count_slot_allocation_us +
        impl_->setup.summary_slot_allocation_us;
  }

  if (!local)
    invalid("phase=generation-coordinator, field=local_handle, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=valid after successful agreement, actual=empty");
  const auto count_start = std::chrono::steady_clock::now();
  auto count = impl_->backend->start(CountExchangeRequest{
      impl_->context, ExchangeKind::ExpertCounts, request.config.num_experts,
      {}, local.device_expert_count_for_backend()});
  auto count_done = count->wait();
  const auto count_end = std::chrono::steady_clock::now();
  const auto summary_start = std::chrono::steady_clock::now();
  auto summary = impl_->backend->start(CountExchangeRequest{
      impl_->context, ExchangeKind::NotifyPacketSummary,
      request.config.topology.num_nodes, {},
      local.device_packet_count_for_backend()});
  auto summary_done = summary->wait();
  const auto summary_end = std::chrono::steady_clock::now();
  const auto materialize_start = std::chrono::steady_clock::now();
  DeviceMetadataBridge::Timings materialize_timings;
  auto device = DeviceMetadataBridge::materialize(
      local, std::move(count_done), std::move(summary_done), impl_->execution,
      &materialize_timings);
  const auto materialize_end = std::chrono::steady_clock::now();
  const auto audit_value = impl_->backend->audit();
  if (audit_value.control_starts != 1 || audit_value.config_starts != 1 ||
      audit_value.expert_count_starts != 1 ||
      audit_value.packet_summary_starts != 1)
    invalid("phase=generation-publish, field=audit, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=1/1/1/1, actual=" +
            std::to_string(audit_value.control_starts) + "/" +
            std::to_string(audit_value.config_starts) + "/" +
            std::to_string(audit_value.expert_count_starts) + "/" +
            std::to_string(audit_value.packet_summary_starts));
  auto result = std::make_shared<ValidatedCudaGeneration::Impl>();
  result->dispatch = std::move(device);
  result->audit = audit_value;
  result->workspace = impl_->backend->workspace_usage();
  result->workspace.route_input_device_bytes = route_input_bytes;
  result->workspace.device_notify_arena_bytes = notify_arena_bytes;
  result->workspace.device_notify_pinned_staging_bytes = notify_pinned_bytes;
  result->workspace.final_device_metadata_arena_view_capacity_bytes =
      notify_arena_bytes;
  result->workspace.device_bytes += route_input_bytes + notify_arena_bytes;
  result->workspace.pinned_host_bytes += notify_pinned_bytes;
  // Notify workspace/route/consumer events plus four collective events.
  result->workspace.event_count = 14;
  result->workspace.quantified_event_bytes = 0;
  if (impl_->pool) {
    result->workspace = {};
    for (const auto &slot : impl_->pool->slots) {
      const auto backend_usage = slot.backend->workspace_usage();
      result->workspace.device_bytes +=
          DeviceNotifyPoolBridge::device_bytes(slot.notify) +
          backend_usage.device_bytes;
      result->workspace.pinned_host_bytes +=
          DeviceNotifyPoolBridge::pinned_bytes(slot.notify) +
          backend_usage.pinned_host_bytes;
      result->workspace.route_input_device_bytes +=
          static_cast<std::uint64_t>(impl_->pool->max_plan.route_count) *
          sizeof(std::int32_t);
      result->workspace.device_notify_arena_bytes +=
          impl_->pool->max_plan.device_bytes;
      result->workspace.device_notify_pinned_staging_bytes +=
          impl_->pool->max_plan.host_staging_bytes;
      result->workspace.control_slot_bytes += backend_usage.control_slot_bytes;
      result->workspace.config_wire_bytes += backend_usage.config_wire_bytes;
      result->workspace.config_slot_bytes += backend_usage.config_slot_bytes;
      result->workspace.expert_slot_bytes += backend_usage.expert_slot_bytes;
      result->workspace.summary_slot_bytes += backend_usage.summary_slot_bytes;
      result->workspace.event_count +=
          4U + DeviceNotifyPoolBridge::event_count(slot.notify);
    }
    result->workspace.final_device_metadata_arena_view_capacity_bytes =
        impl_->pool->max_plan.device_bytes;
    // A production router-owned input is outside the pool arenas and is a
    // unique live allocation. Account it once in addition to all pool slots;
    // the host-convenience path uses a pool route slot and adds nothing here.
    if (direct_routes) {
      result->workspace.route_input_device_bytes += route_input_bytes;
      result->workspace.device_bytes += route_input_bytes;
    }
  }
  result->setup = impl_->setup;
  const auto run_end = std::chrono::steady_clock::now();
  result->execution.coordinator_run_wall_us =
      std::chrono::duration<double, std::micro>(run_end - run_start).count();
  const auto dynamic_setup = result->setup.config_slot_allocation_us +
                             result->setup.count_slot_allocation_us +
                             result->setup.summary_slot_allocation_us +
                             result->setup.pool_acquire_us;
  result->execution.steady_execution_us =
      std::max(0.0, result->execution.coordinator_run_wall_us - dynamic_setup);
  result->execution.local_mapping_us = local_mapping_us;
  result->execution.control_config_us =
      std::chrono::duration<double, std::micro>(control_config_end -
                                                control_config_start)
          .count();
  result->execution.expert_count_us =
      std::chrono::duration<double, std::micro>(count_end - count_start).count();
  result->execution.packet_summary_us =
      std::chrono::duration<double, std::micro>(summary_end - summary_start)
          .count();
  result->execution.device_materialization_us =
      std::chrono::duration<double, std::micro>(materialize_end -
                                                materialize_start)
          .count();
  result->execution.prefix_planning_us =
      materialize_timings.prefix_planning_us;
  result->execution.remaining_parallel_materialization_us =
      materialize_timings.remaining_parallel_us;
  result->execution.pool_acquire_release_us = result->setup.pool_acquire_us;
  result->resource_lease = pool_lease
                               ? std::static_pointer_cast<const void>(pool_lease)
                               : std::static_pointer_cast<const void>(impl_);
  testing::g_validated_publishes.fetch_add(1);
  return ValidatedCudaGeneration(std::move(result));
}

ValidatedCudaGeneration CudaNotifyGenerationCoordinator::run_device(
    DeviceNotifyGenerationRequest request) {
  if (!impl_ || !impl_->pool)
    invalid("phase=generation-coordinator-device, field=resource_pool, "
            "rank=n/a, expected=explicit session pool, actual=missing");
  if (!request.local_routes)
    invalid("phase=generation-coordinator-device, field=routes, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=valid device lease, actual=empty");
  impl_->direct_routes = request.local_routes;
  NotifyGenerationRequest adapted;
  adapted.config = std::move(request.config);
  adapted.local_routes.num_tokens = request.local_routes.num_tokens();
  adapted.max_tokens_per_rank = request.max_tokens_per_rank;
  return run(std::move(adapted));
}

CollectiveProtocolAudit CudaNotifyGenerationCoordinator::audit() const {
  if (!impl_)
    invalid("phase=generation-audit, field=coordinator, rank=n/a, expected=valid, actual=empty");
  return impl_->backend->audit();
}

NcclWorkspaceUsage CudaNotifyGenerationCoordinator::workspace_usage() const {
  if (!impl_)
    invalid("phase=generation-workspace, field=coordinator, rank=n/a, expected=valid, actual=empty");
  return impl_->backend->workspace_usage();
}

GenerationSetupTimings CudaNotifyGenerationCoordinator::setup_timings() const {
  if (!impl_)
    invalid("phase=generation-setup, field=coordinator, rank=n/a, expected=valid, actual=empty");
  return impl_->setup;
}

} // namespace megamoe::cuda
