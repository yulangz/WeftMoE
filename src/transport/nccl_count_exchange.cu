#include "megamoe/cuda/nccl_count_exchange.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <array>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cuda/cuda_internal.hpp"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
namespace {

[[noreturn]] void invalid(const std::string &message) {
  throw NotifyDispatchError(NotifyErrorCode::NcclFailure, message);
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
  std::uint64_t *host_matrix = nullptr;
  cudaEvent_t completion = nullptr;
  bool in_use = false;

  ~Slot() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (completion)
      (void)cudaEventSynchronize(completion);
    if (completion)
      (void)cudaEventDestroy(completion);
    if (host_matrix)
      (void)cudaFreeHost(host_matrix);
    if (device_matrix)
      (void)cudaFree(device_matrix);
  }
};

const char *kind_name(ExchangeKind kind) noexcept {
  return kind == ExchangeKind::ExpertCounts ? "expert_counts"
                                            : "notify_packet_summary";
}

} // namespace

namespace internal {

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
  ncclUniqueId id{};
  internal::throw_nccl("bootstrap", ncclGetUniqueId(&id), 0, 1,
                       "bootstrap", 1, ExchangeKind::ExpertCounts,
                       "ncclGetUniqueId");
  std::vector<std::uint8_t> bytes(sizeof(id));
  std::memcpy(bytes.data(), &id, sizeof(id));
  return NcclBootstrapId(std::move(bytes));
}

std::string NcclBootstrapId::hex() const {
  if (bytes_.size() != sizeof(ncclUniqueId))
    invalid("phase=bootstrap-id, field=bytes, rank=n/a, expected=" +
            std::to_string(sizeof(ncclUniqueId)) + ", actual=" +
            std::to_string(bytes_.size()));
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto byte : bytes_)
    output << std::setw(2) << static_cast<unsigned>(byte);
  return output.str();
}

NcclBootstrapId parse_nccl_bootstrap_id(const std::string &hex) {
  if (hex.size() != sizeof(ncclUniqueId) * 2)
    invalid("phase=bootstrap-id-parse, field=hex.size, rank=n/a, expected=" +
            std::to_string(sizeof(ncclUniqueId) * 2) + ", actual=" +
            std::to_string(hex.size()));
  std::vector<std::uint8_t> bytes(sizeof(ncclUniqueId));
  const auto digit = [](char value) -> unsigned {
    if (value >= '0' && value <= '9')
      return static_cast<unsigned>(value - '0');
    if (value >= 'a' && value <= 'f')
      return 10U + static_cast<unsigned>(value - 'a');
    if (value >= 'A' && value <= 'F')
      return 10U + static_cast<unsigned>(value - 'A');
    invalid("phase=bootstrap-id-parse, field=hex, rank=n/a, expected=hex digits, actual=invalid character");
  };
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::uint8_t>((digit(hex[2 * i]) << 4U) |
                                         digit(hex[2 * i + 1]));
  return NcclBootstrapId(std::move(bytes));
}

struct NcclSession::Impl {
  std::shared_ptr<internal::NcclSessionState> state;
  CollectiveSession session;
};

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
  internal::throw_nccl("session-create",
                       ncclCommInitRank(&state->comm, static_cast<int>(world_size),
                                        id, static_cast<int>(rank)),
                       rank, world_size, identity, generation,
                       ExchangeKind::ExpertCounts, "ncclCommInitRank");
  auto impl = std::make_shared<NcclSession::Impl>();
  impl->state = state;
  impl->session = make_collective_session(std::move(identity), world_size,
                                          generation, state);
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

struct NcclAllGatherCountExchange::Impl {
  NcclSession session;
  CudaExecution execution;
  std::shared_ptr<const internal::CudaExecutionState> execution_state;
  ncclComm_t comm = nullptr;
  std::unique_ptr<Slot> expert;
  std::unique_ptr<Slot> summary;
  std::mutex mutex;
};

namespace {

void allocate_slot(Slot &slot, std::uint64_t width,
                   const NcclSession &session, ExchangeKind kind) {
  slot.width = width;
  slot.world = session.session().world_size();
  slot.device = session.device_ordinal();
  const auto elements = checked_elements(slot.world, width, "matrix_elements");
  const auto bytes = checked_matrix_bytes(elements, "matrix_bytes");
  internal::throw_cuda("nccl-workspace", cudaSetDevice(static_cast<int>(slot.device)),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind, "cudaSetDevice");
  internal::throw_cuda("nccl-workspace",
                       cudaMalloc(reinterpret_cast<void **>(&slot.device_matrix),
                                  bytes),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind,
                       "cudaMalloc(AllGather matrix)");
  internal::throw_cuda("nccl-workspace",
                       cudaHostAlloc(reinterpret_cast<void **>(&slot.host_matrix),
                                     bytes,
                                     cudaHostAllocPortable),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind,
                       "cudaHostAlloc(AllGather staging)");
  internal::throw_cuda("nccl-workspace",
                       cudaEventCreateWithFlags(&slot.completion,
                                                cudaEventDisableTiming),
                       session.rank(), slot.world,
                       session.session().identity(),
                       session.session().generation(), kind,
                       "cudaEventCreate(AllGather)");
}

class NcclAllGatherOperation final : public BackendCountExchangeOperation {
public:
  NcclAllGatherOperation(std::shared_ptr<NcclAllGatherCountExchange::Impl> impl,
                         Slot *slot, CountExchangeRequest request)
      : impl_(std::move(impl)), slot_(slot), request_(std::move(request)) {}

  ~NcclAllGatherOperation() noexcept override {
    if (!terminal_) {
      (void)cudaSetDevice(static_cast<int>(impl_->session.device_ordinal()));
      (void)cudaEventSynchronize(slot_->completion);
      release();
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
    CountMatrix rows(session.world_size(),
                     std::vector<std::uint64_t>(
                         static_cast<std::size_t>(request_.row_width)));
    for (std::uint32_t rank = 0; rank < session.world_size(); ++rank) {
      const auto offset = static_cast<std::size_t>(rank) *
                          static_cast<std::size_t>(request_.row_width);
      std::copy_n(slot_->host_matrix + offset,
                  static_cast<std::size_t>(request_.row_width),
                  rows[rank].begin());
    }
    terminal_ = true;
    release();
    auto lease = std::make_shared<internal::CudaCompletionLease>();
    lease->owner = impl_;
    return make_host_backend_completion(
        request_.context, request_.kind, make_host_count_matrix(rows),
        std::move(lease));
  }

  void cancel() override {
    if (terminal_)
      return;
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
  }

private:
  void release() noexcept {
    if (released_)
      return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    slot_->in_use = false;
    released_ = true;
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
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t max_expert_row_width,
    std::uint64_t max_summary_row_width) {
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
  impl_->expert = std::make_unique<Slot>();
  impl_->summary = std::make_unique<Slot>();
  allocate_slot(*impl_->expert, max_expert_row_width, session,
                ExchangeKind::ExpertCounts);
  allocate_slot(*impl_->summary, max_summary_row_width, session,
                ExchangeKind::NotifyPacketSummary);
}

NcclAllGatherCountExchange::~NcclAllGatherCountExchange() noexcept = default;
NcclAllGatherCountExchange::NcclAllGatherCountExchange(
    NcclAllGatherCountExchange &&) noexcept = default;
NcclAllGatherCountExchange &NcclAllGatherCountExchange::operator=(
    NcclAllGatherCountExchange &&) noexcept = default;

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
  Slot *slot = request.kind == ExchangeKind::ExpertCounts
                   ? impl_->expert.get()
                   : impl_->summary.get();
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (request.row_width == 0 || request.row_width > slot->width)
    invalid("phase=nccl-start, field=row_width, rank=" +
            std::to_string(rank) + ", expected=(0," +
            std::to_string(slot->width) + "], actual=" +
            std::to_string(request.row_width));
  if (slot->in_use)
    invalid("phase=nccl-start, field=slot, rank=" + std::to_string(rank) +
            ", expected=available, actual=in-use, kind=" + kind_name(request.kind));
  slot->in_use = true;
  try {
    internal::throw_cuda(
        "nccl-start",
        cudaSetDevice(static_cast<int>(impl_->session.device_ordinal())), rank,
        session.world_size(), session.identity(), session.generation(),
        request.kind, "cudaSetDevice");
    internal::throw_nccl(
        "nccl-start",
        ncclAllGather(device->pointer, slot->device_matrix,
                      static_cast<std::size_t>(request.row_width), ncclUint64,
                      impl_->comm, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        request.kind, "ncclAllGather");
    const auto elements = checked_elements(
        session.world_size(), request.row_width, "active_matrix_elements");
    const auto bytes = checked_matrix_bytes(elements, "active_matrix_bytes");
    internal::throw_cuda(
        "nccl-start",
        cudaMemcpyAsync(slot->host_matrix, slot->device_matrix, bytes,
                        cudaMemcpyDeviceToHost, impl_->execution_state->stream),
        rank, session.world_size(), session.identity(), session.generation(),
        request.kind, "cudaMemcpyAsync(D2H collective matrix)");
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
    throw;
  }
  return std::make_unique<NcclAllGatherOperation>(impl_, slot, request);
}

#define MEGAMOE_STRINGIFY_DETAIL(x) #x
#define MEGAMOE_STRINGIFY(x) MEGAMOE_STRINGIFY_DETAIL(x)
const char *compiled_nccl_version() noexcept {
  return MEGAMOE_STRINGIFY(NCCL_MAJOR) "." MEGAMOE_STRINGIFY(NCCL_MINOR) "." MEGAMOE_STRINGIFY(NCCL_PATCH);
}

} // namespace megamoe::cuda
