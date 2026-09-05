} // namespace

#include "cuda/forward/transport_launch.hpp"

struct ActivationTransportResourcePool::Impl {
  std::shared_ptr<ActivationPoolState> state;
};

struct SelectedActivationLease<transport::BFloat16>::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  const transport::BFloat16 *pointer = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken producer_execution;
  std::uint64_t generation = 0;
  std::uint64_t tokens = 0;
  std::uint64_t hidden = 0;
  std::uint32_t device = 0;
  std::uint32_t layout = kSelectedActivationLayoutVersion;
  bool ready_recorded = false;
  std::shared_ptr<const void> external_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;
};

struct BackwardGradientAllocation {
  void *data = nullptr;
  cudaEvent_t ready = nullptr;
  std::uint32_t device = 0;
  bool owns_data = true;
  std::shared_ptr<const void> external_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;

  ~BackwardGradientAllocation() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (ready) {
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      (void)cudaEventDestroy(ready);
    }
    if (data && owns_data) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(data);
    }
  }
};

struct BFloat16OutputGradientLease::Impl {
  std::shared_ptr<BackwardGradientAllocation> allocation;
  const transport::BFloat16 *pointer = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken producer_execution;
  std::uint64_t generation = 0;
  std::uint64_t tokens = 0;
  std::uint64_t hidden = 0;
  std::uint32_t device = 0;
};

struct LocalP2pBackendState;
struct SocketBackendState;
struct GinBackendState;

struct ActivationMegaKernelBackendState {
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<internal::NcclSessionState> nccl;
  NcclSession session;
  CudaExecution execution;
  virtual bool requires_collective_terminal_status() const noexcept = 0;
  virtual ~ActivationMegaKernelBackendState() noexcept = default;
};

struct DeviceActivationTransportHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  ActivationTransportExpectation identity;
  ActivationTransportDeviceAudit audit;
  ActivationTransportTimings timings;
  std::uint64_t route_count = 0;
  std::uint64_t destination_rows = 0;
  std::shared_ptr<LocalP2pBackendState> local_p2p_backend;
  std::shared_ptr<SocketBackendState> socket_backend;
  std::shared_ptr<GinBackendState> gin_backend;
};

struct DeviceActivationTransport::Impl {
  CollectiveSession session;
  CudaExecution execution;
  ActivationTransportResourcePool pool;
};

struct LocalP2pBackendState : ActivationMegaKernelBackendState {
  std::vector<void *> peer_slot_bases;
  std::vector<std::uint8_t> owns_mapping;

  bool requires_collective_terminal_status() const noexcept override {
    return false;
  }

  ~LocalP2pBackendState() noexcept {
    if (!pool)
      return;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    for (std::size_t index = 0; index < peer_slot_bases.size(); ++index) {
      if (owns_mapping[index] != 0 && peer_slot_bases[index])
        (void)cudaIpcCloseMemHandle(peer_slot_bases[index]);
    }
  }
};

struct LocalCudaP2pActivationTransport::Impl {
  std::shared_ptr<LocalP2pBackendState> backend;
};

struct SocketBackendState : ActivationMegaKernelBackendState {
  bool requires_collective_terminal_status() const noexcept override {
    return true;
  }
};

struct NcclSocketFallbackActivationTransport::Impl {
  std::shared_ptr<SocketBackendState> backend;
};

struct GinBackendState : ActivationMegaKernelBackendState {
  ncclDevComm_t dev_comm{};
  std::vector<ncclWindow_t> windows;
  std::vector<void *> peer_slot_bases;
  std::vector<std::uint8_t> owns_mapping;
  std::uint32_t num_nodes = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t node = 0;
  std::uint32_t local_rank = 0;
  std::uint32_t gin_context_count = 1;
  std::uint32_t gin_allocated_context_count = 1;
  bool full_connections = false;
  bool dev_comm_created = false;

  bool requires_collective_terminal_status() const noexcept override {
    return false;
  }

  ~GinBackendState() noexcept {
    if (!nccl)
      return;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    for (std::size_t index = 0; index < peer_slot_bases.size(); ++index)
      if (owns_mapping[index] != 0 && peer_slot_bases[index] != nullptr)
        (void)cudaIpcCloseMemHandle(peer_slot_bases[index]);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    if (window_deregister != nullptr)
      for (const auto window : windows)
        if (window != nullptr)
          (void)window_deregister(nccl->comm, window);
    if (dev_comm_created && dev_comm_destroy != nullptr)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  }
};

struct GinRdmaActivationTransport::Impl {
  std::shared_ptr<GinBackendState> backend;
};

namespace {

struct ProductionStatusAgreementResult {
  bool has_error = false;
  ProductionStatusRecord error{};
  double elapsed_us = 0.0;
};

template <typename Status>
ProductionStatusAgreementResult agree_production_terminal_status(
    const std::shared_ptr<ActivationMegaKernelBackendState> &backend,
    PoolSlot &slot, ProductionStatusPhase phase, std::uint64_t generation,
    std::uint64_t epoch, std::uint32_t rank, std::uint32_t world,
    const Status &status, cudaEvent_t completion_event) {
  const auto begin = std::chrono::steady_clock::now();
  if (!backend->requires_collective_terminal_status()) {
    ProductionStatusAgreementResult result;
    if (status.error != 0U) {
      result.has_error = true;
      result.error = ProductionStatusRecord{
          generation,
          static_cast<unsigned long long>(phase),
          static_cast<unsigned long long>(status.error_origin_rank),
          static_cast<unsigned long long>(status.error_code),
          static_cast<unsigned long long>(status.error_field),
          status.error_expected,
          status.error_actual,
          epoch};
    }
    result.elapsed_us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
    return result;
  }
  auto &layout = backend->pool->layout;
  const auto exec = internal::checked_execution_state(backend->execution);
  auto *host_records = slot.host_production_status;
  const auto error = status.error != 0U;
  host_records[rank] = ProductionStatusRecord{
      generation,
      static_cast<unsigned long long>(phase),
      error ? static_cast<unsigned long long>(status.error_origin_rank)
            : static_cast<unsigned long long>(rank),
      error ? static_cast<unsigned long long>(status.error_code) : 0ULL,
      error ? static_cast<unsigned long long>(status.error_field) : 0ULL,
      error ? status.error_expected : 0ULL,
      error ? status.error_actual : 0ULL,
      epoch};
  auto *send = at<ProductionStatusRecord>(
      slot.device, layout.production_error_records) + rank;
  auto *receive = at<unsigned long long>(
      slot.device, layout.production_status_matrix);
  internal::throw_cuda(
      "production-terminal-status",
      cudaMemcpyAsync(send, host_records + rank,
                      sizeof(ProductionStatusRecord), cudaMemcpyHostToDevice,
                      exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "H2D local production status record");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(ProductionStatusRecord));
  internal::throw_nccl(
      "production-terminal-status",
      ncclAllGather(send, receive, 8, ncclUint64, backend->nccl->comm,
                    exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts,
      "ncclAllGather(ProductionStatusRecord)");
  const auto matrix_bytes = checked_size(
      checked_mul(world, sizeof(ProductionStatusRecord),
                  "production terminal status matrix"),
      "production terminal status matrix");
  internal::throw_cuda(
      "production-terminal-status",
      cudaMemcpyAsync(host_records, receive, matrix_bytes,
                      cudaMemcpyDeviceToHost, exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "D2H production status matrix");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            matrix_bytes);
  internal::throw_cuda(
      "production-terminal-status",
      cudaEventRecord(completion_event, exec->stream), rank, world,
      backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(production terminal status)");
  internal::throw_cuda(
      "production-terminal-status", cudaEventSynchronize(completion_event),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts,
      "cudaEventSynchronize(production terminal status)");

  ProductionStatusAgreementResult result;
  const auto expected_phase = static_cast<unsigned long long>(phase);
  const auto malformed = [&](std::uint32_t origin, std::uint64_t field,
                             std::uint64_t expected,
                             std::uint64_t actual) {
    result.has_error = true;
    result.error = ProductionStatusRecord{
        generation, expected_phase, origin, 2ULL, field, expected, actual,
        epoch};
  };
  for (std::uint32_t source = 0; source < world && !result.has_error;
       ++source) {
    const auto &record = host_records[source];
    if (record.generation != generation)
      malformed(source, 1ULL, generation, record.generation);
    else if (record.phase != expected_phase)
      malformed(source, 2ULL, expected_phase, record.phase);
    else if (record.origin_rank != source && record.error_code == 0ULL)
      malformed(source, 3ULL, source, record.origin_rank);
    else if (record.epoch != epoch)
      malformed(source, 4ULL, epoch, record.epoch);
  }
  if (!result.has_error) {
    for (std::uint32_t source = 0; source < world; ++source) {
      const auto &record = host_records[source];
      if (record.error_code == 0ULL)
        continue;
      if (record.origin_rank >= world) {
        malformed(source, 5ULL, world - 1ULL, record.origin_rank);
        break;
      }
      if (!result.has_error || record.origin_rank < result.error.origin_rank ||
          (record.origin_rank == result.error.origin_rank &&
           record.field < result.error.field)) {
        result.has_error = true;
        result.error = record;
      }
    }
  }
  result.elapsed_us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  return result;
}

void throw_production_status_error(
    const ProductionStatusAgreementResult &agreement) {
  if (!agreement.has_error)
    return;
  const auto &error = agreement.error;
  throw NotifyDispatchError(
      NotifyErrorCode::TransportProtocolFailure,
      std::string("phase=production-terminal-agreement, kind=Forward") +
          ", origin_rank=" + std::to_string(error.origin_rank) +
          ", field=" + std::to_string(error.field) +
          ", code=" + std::to_string(error.error_code) +
          ", expected=" + std::to_string(error.expected) +
          ", actual=" + std::to_string(error.actual) +
          ", generation=" + std::to_string(error.generation));
}

} // namespace

struct ForwardAllocation {
  void *w1 = nullptr;
  void *w2 = nullptr;
  std::uint64_t bytes = 0;
  std::uint32_t device = 0;
  cudaEvent_t ready = nullptr;
  bool owns_w1 = true;
  bool owns_w2 = true;
  std::shared_ptr<const void> w1_owner;
  std::shared_ptr<const void> w2_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;
  ~ForwardAllocation() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (ready) {
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      (void)cudaEventDestroy(ready);
    }
    if (w2 && owns_w2) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w2);
    }
    if (w1 && owns_w1) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w1);
    }
  }
};

struct ForwardWeightLease::Impl {
  std::shared_ptr<ForwardAllocation> allocation;
  const transport::BFloat16 *w1 = nullptr;
  const transport::BFloat16 *w2 = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken execution;
  std::uint32_t owned_experts = 0;
  std::uint32_t global_expert_begin = 0;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint32_t device = 0;
  CUtensorMap w1_map{};
  CUtensorMap w2_map{};
  CUtensorMap backward_w1_map{};
  CUtensorMap backward_w2_map{};
};

struct ForwardWeightStorageHashAccess {
  static const ForwardWeightLease::Impl &get(
      const ForwardWeightLease &weights) {
    return *weights.impl_;
  }
};

struct DeviceForwardHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
  DeviceDispatchHandle dispatch;
  BFloat16ActivationLease activations;
  ForwardWeightLease weights;
  ActivationTransportExpectation identity;
  ForwardMegaKernelAudit audit;
  ForwardMegaKernelTimings timings;
  bool retained_backward_state = false;
  std::uint64_t incoming_rows = 0;
  std::uint64_t intermediate = 0;
  // Backward reuses Forward's immutable FULL-return row mapping, including
  // the chunk boundaries encoded in each entry.
  std::uint64_t forward_return_chunk_rows = 0;
  // The compact expert counts are immutable Dispatch metadata. Backward
  // lazily snapshots the tiny vector before its first timed launch and reuses
  // it for repeated executions of the same retained Forward state.
  mutable std::mutex backward_plan_mutex;
  mutable std::vector<std::uint64_t> backward_expert_rows;
  mutable std::vector<std::uint64_t> backward_expert_offsets;
  mutable std::vector<std::uint64_t> backward_padded_expert_rows;
  mutable std::vector<std::uint64_t> backward_padded_expert_offsets;
  mutable std::vector<std::uint32_t> backward_compact_to_padded;
  mutable std::uint64_t backward_padded_rows = 0;
  mutable bool backward_plan_uploaded = false;
  // Host-enqueued Socket transfers execute outside the persistent kernel.
  std::uint64_t socket_dispatch_peer_bytes = 0;
  std::uint64_t socket_return_peer_bytes = 0;
  std::uint64_t socket_raw_cross_node_route_bytes = 0;
};

struct DeviceBackwardHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  DeviceForwardHandle forward;
  BFloat16OutputGradientLease output_gradients;
  ActivationTransportExpectation identity;
  BackwardMegaKernelAudit audit;
  BackwardMegaKernelTimings timings;
  std::uint32_t owned_experts = 0;
  std::uint64_t incoming_rows = 0;
  std::uint64_t intermediate = 0;
};

struct LocalCudaP2pForwardMegaKernel::Impl {
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
  std::shared_ptr<NcclSocketFallbackActivationTransport> socket_transport;
};


#if MEGAMOE_DEVICE_TIMELINE
namespace {
profiling::DeviceTimelineSnapshot download_device_timeline(
    const std::shared_ptr<ActivationPoolState> &pool,
    const std::shared_ptr<SlotLease> &lease, void *allocation,
    cudaEvent_t completion, const ActivationTransportExpectation &identity,
    profiling::TimelinePhase phase) {
  if (!pool || !lease || lease->index >= pool->slots.size() || !allocation)
    transport_error(NotifyErrorCode::InvalidHandle, "device-timeline",
                    "handle/storage", identity.rank, "valid", "empty");
  internal::throw_cuda(
      "device-timeline", cudaEventSynchronize(completion), identity.rank,
      identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "wait timeline producer completion");
  const auto max_ctas = pool->timeline_max_ctas;
  std::vector<std::uint32_t> counts(max_ctas);
  std::vector<std::uint32_t> overflow(max_ctas);
  const auto count_bytes = checked_size(
      checked_mul(max_ctas, sizeof(std::uint32_t), "timeline count download"),
      "timeline count download");
  internal::throw_cuda(
      "device-timeline",
      cudaMemcpyAsync(counts.data(),
                      at<const void>(allocation, pool->timeline_layout.counts),
                      count_bytes, cudaMemcpyDeviceToHost, pool->cancel_stream),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "download timeline counts");
  internal::throw_cuda(
      "device-timeline",
      cudaMemcpyAsync(
          overflow.data(),
          at<const void>(allocation, pool->timeline_layout.overflow),
          count_bytes, cudaMemcpyDeviceToHost, pool->cancel_stream),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "download timeline overflow");
  const auto record_count = checked_mul(max_ctas, profiling::kEventsPerCta,
                                        "timeline record download");
  std::vector<profiling::DeviceTimelineRecord> raw(
      checked_size(record_count, "timeline record vector"));
  const auto record_bytes = checked_size(
      checked_mul(record_count, sizeof(profiling::DeviceTimelineRecord),
                  "timeline record bytes"),
      "timeline record bytes");
  internal::throw_cuda(
      "device-timeline",
      cudaMemcpyAsync(
          raw.data(), at<const void>(allocation, pool->timeline_layout.records),
          record_bytes, cudaMemcpyDeviceToHost, pool->cancel_stream),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "download timeline records");
  // All producer streams are ordered before the completion event above.
  // Download on the pool's non-blocking diagnostic stream rather than using
  // legacy synchronous memcpy semantics, which can form a driver-side
  // liveness cycle with resident device GIN work on the peer host.
  internal::throw_cuda(
      "device-timeline", cudaStreamSynchronize(pool->cancel_stream),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "drain timeline download");
  profiling::DeviceTimelineSnapshot result;
  result.rank = identity.rank;
  result.generation = identity.generation;
  result.phase = phase;
  result.max_ctas = max_ctas;
  result.capacity_per_cta = profiling::kEventsPerCta;
  for (std::uint32_t cta = 0; cta < max_ctas; ++cta) {
    result.overflow_events += overflow[cta];
    const auto stored = std::min(counts[cta], profiling::kEventsPerCta);
    const auto begin = raw.begin() +
                       static_cast<std::ptrdiff_t>(cta) *
                           profiling::kEventsPerCta;
    result.records.insert(result.records.end(), begin, begin + stored);
  }
  std::stable_sort(result.records.begin(), result.records.end(),
                   [](const auto &left, const auto &right) {
                     if (left.timestamp_ns != right.timestamp_ns)
                       return left.timestamp_ns < right.timestamp_ns;
                     if (left.cta != right.cta)
                       return left.cta < right.cta;
                     return left.metadata < right.metadata;
                   });
  return result;
}
} // namespace

profiling::DeviceTimelineSnapshot testing::download_forward_device_timeline(
    const DeviceForwardHandle &handle) {
  if (!handle.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "device-timeline",
                    "Forward handle", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  const auto &slot = impl.pool->slots[impl.slot->index];
  return download_device_timeline(
      impl.pool, impl.slot, slot.forward_timeline, slot.forward_complete,
      impl.identity, profiling::TimelinePhase::Forward);
}
#endif

std::uint32_t ForwardWeightLease::owned_experts() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->owned_experts;
}
std::uint64_t ForwardWeightLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t ForwardWeightLease::intermediate_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->intermediate;
}
std::uint64_t ForwardWeightLease::device_bytes() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->allocation->bytes;
}
std::uint32_t ForwardWeightLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->device;
}

ForwardWeightLease testing::make_deterministic_forward_weights_for_testing(
    std::uint32_t owned_experts, std::uint32_t global_expert_begin,
    std::uint64_t hidden_size, std::uint64_t intermediate_size,
    const CollectiveSession &session, const CudaExecution &execution) {
  if (!session || !execution || owned_experts == 0 || hidden_size == 0 ||
      intermediate_size == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-weight-create", "shape/session/execution", 0,
                    "all valid and nonzero", "invalid");
  const auto w1_rows = checked_mul(
      checked_mul(owned_experts, 2ULL, "W1 paired expert rows"),
      intermediate_size, "W1 expert rows");
  const auto w1_elements = checked_mul(
      w1_rows,
      hidden_size, "W1 elements");
  const auto w2_elements = checked_mul(
      checked_mul(owned_experts, hidden_size, "W2 expert rows"),
      intermediate_size, "W2 elements");
  const auto w1_bytes = checked_size(
      checked_mul(w1_elements, sizeof(transport::BFloat16), "W1 bytes"),
      "W1 bytes");
  const auto w2_bytes = checked_size(
      checked_mul(w2_elements, sizeof(transport::BFloat16), "W2 bytes"),
      "W2 bytes");
  const auto state = internal::checked_execution_state(execution);
  auto allocation = std::make_shared<ForwardAllocation>();
  allocation->device = state->device;
  allocation->bytes = static_cast<std::uint64_t>(w1_bytes) + w2_bytes;
  internal::throw_cuda(
      "forward-weight-create", cudaSetDevice(static_cast<int>(state->device)),
      0, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda(
      "forward-weight-create", cudaMalloc(&allocation->w1, w1_bytes), 0,
      session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts, "cudaMalloc(W1)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  try {
    internal::throw_cuda(
        "forward-weight-create", cudaMalloc(&allocation->w2, w2_bytes), 0,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaMalloc(W2)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    internal::throw_cuda(
        "forward-weight-create",
        cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaEventCreate(weight ready)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    initialize_forward_weights_kernel<<<4096, 256, 0, state->stream>>>(
        static_cast<transport::BFloat16 *>(allocation->w1), w1_elements,
        static_cast<transport::BFloat16 *>(allocation->w2), w2_elements,
        global_expert_begin, hidden_size, intermediate_size);
    internal::throw_cuda(
        "forward-weight-create", cudaGetLastError(), 0, session.world_size(),
        session.identity(), session.generation(), ExchangeKind::ExpertCounts,
        "initialize_forward_weights_kernel");
    internal::throw_cuda(
        "forward-weight-create",
        cudaEventRecord(allocation->ready, state->stream), 0,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaEventRecord(weight ready)");
  } catch (...) {
    allocation.reset();
    throw;
  }
  auto impl = std::make_shared<ForwardWeightLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->w1 = static_cast<const transport::BFloat16 *>(impl->allocation->w1);
  impl->w2 = static_cast<const transport::BFloat16 *>(impl->allocation->w2);
  impl->session_identity = session.stable_identity_bytes();
  impl->execution = execution.token();
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state->device;
  if (hidden_size == 7168 && intermediate_size == 3072) {
    impl->w1_map = make_bf16_tma_map(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        forward_tuning::kFc1TileN, "W1", forward_tuning::kTileK);
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        forward_tuning::kFc2TileN, "W2", forward_tuning::kTileK);
    impl->backward_w1_map = make_bf16_tma_map_mn(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        "Backward W1 MN-major");
    impl->backward_w2_map = make_bf16_tma_map_mn(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        "Backward W2 MN-major");
  }
  return ForwardWeightLease(std::move(impl));
}

ForwardWeightLease ExternalCudaProducer::publish_weights(
    const transport::BFloat16 *w1, std::uint64_t w1_capacity_elements,
    std::shared_ptr<const void> w1_owner,
    const transport::BFloat16 *w2, std::uint64_t w2_capacity_elements,
    std::shared_ptr<const void> w2_owner, std::uint32_t owned_experts,
    std::uint32_t global_expert_begin, std::uint64_t hidden_size,
    std::uint64_t intermediate_size) const {
  if (!state_ || !w1_owner || !w2_owner || owned_experts == 0 ||
      hidden_size == 0 || intermediate_size == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "external-weight-publish", "producer/owner/shape", 0,
                    "valid and nonzero", "invalid");
  const auto w1_elements = checked_mul(
      checked_mul(owned_experts, 2ULL * intermediate_size,
                  "external W1 rows"),
      hidden_size, "external W1 elements");
  const auto w2_elements = checked_mul(
      checked_mul(owned_experts, hidden_size, "external W2 rows"),
      intermediate_size, "external W2 elements");
  validate_external_cuda_range(w1, w1_elements, w1_capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "W1");
  validate_external_cuda_range(w2, w2_elements, w2_capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "W2");
  auto allocation = std::make_shared<ForwardAllocation>();
  allocation->device = state_->backend->device;
  allocation->w1 = const_cast<transport::BFloat16 *>(w1);
  allocation->w2 = const_cast<transport::BFloat16 *>(w2);
  allocation->owns_w1 = false;
  allocation->owns_w2 = false;
  allocation->w1_owner = std::move(w1_owner);
  allocation->w2_owner = std::move(w2_owner);
  allocation->external_producer = state_->backend;
  allocation->bytes = checked_add(
      checked_mul(w1_elements, sizeof(transport::BFloat16), "W1 bytes"),
      checked_mul(w2_elements, sizeof(transport::BFloat16), "W2 bytes"),
      "weight bytes");
  internal::throw_cuda(
      "external-weight-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventCreate(weight ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "external-weight-publish",
      cudaEventRecord(allocation->ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(weight ready)");
  auto impl = std::make_shared<ForwardWeightLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->w1 = w1;
  impl->w2 = w2;
  impl->session_identity = state_->backend->context.session().stable_identity_bytes();
  impl->execution = state_->backend->execution;
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state_->backend->device;
  if (hidden_size == 7168 && intermediate_size == 3072) {
    impl->w1_map = make_bf16_tma_map(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        forward_tuning::kFc1TileN, "W1", forward_tuning::kTileK);
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        forward_tuning::kFc2TileN, "W2", forward_tuning::kTileK);
    impl->backward_w1_map = make_bf16_tma_map_mn(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        "Backward W1 MN-major");
    impl->backward_w2_map = make_bf16_tma_map_mn(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        "Backward W2 MN-major");
  }
  return ForwardWeightLease(std::move(impl));
}

ForwardWeightStorageHashes
testing::hash_forward_weight_storage_for_validation(
    const ForwardWeightLease &weights, const CudaExecution &execution) {
  if (!weights || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weight-hash",
                    "weights/execution", 0, "valid", "empty");
  const auto &impl = ForwardWeightStorageHashAccess::get(weights);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-weight-hash", "device", 0,
                    std::to_string(impl.device),
                    std::to_string(state->device));
  internal::throw_cuda(
      "forward-weight-hash",
      cudaStreamWaitEvent(state->stream, impl.allocation->ready, 0), 0, 1,
      "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(weight ready)");

  const auto hash_one = [&](const transport::BFloat16 *values,
                            std::uint64_t elements,
                            const char *field) -> DeviceStorageHash128 {
    const auto chunk_count =
        (elements + kWeightHashChunkElements - 1ULL) /
        kWeightHashChunkElements;
    const auto chunk_bytes = checked_size(
        checked_mul(chunk_count, sizeof(WeightHashPair),
                    "weight hash chunk bytes"),
        "weight hash chunk bytes");
    WeightHashPair *device_chunks = nullptr;
    WeightHashPair *device_result = nullptr;
    try {
      internal::throw_cuda(
          "forward-weight-hash", cudaMalloc(&device_chunks, chunk_bytes), 0,
          1, "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(hash chunks)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "forward-weight-hash",
          cudaMalloc(&device_result, sizeof(WeightHashPair)), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(hash result)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      hash_forward_weight_chunks_kernel<<<
          static_cast<unsigned>(chunk_count), 256, 0, state->stream>>>(
          values, elements, device_chunks);
      internal::throw_cuda(
          "forward-weight-hash", cudaGetLastError(), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          field);
      finalize_forward_weight_hash_kernel<<<1, 1, 0, state->stream>>>(
          device_chunks, chunk_count, elements, device_result);
      internal::throw_cuda(
          "forward-weight-hash", cudaGetLastError(), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "finalize_forward_weight_hash_kernel");
      WeightHashPair host{};
      internal::throw_cuda(
          "forward-weight-hash",
          cudaMemcpyAsync(&host, device_result, sizeof(host),
                          cudaMemcpyDeviceToHost, state->stream),
          0, 1, "forward-weight-validation", 1,
          ExchangeKind::ExpertCounts, "cudaMemcpyAsync(hash result)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                sizeof(host));
      internal::throw_cuda(
          "forward-weight-hash", cudaStreamSynchronize(state->stream), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaStreamSynchronize(hash result)");
      (void)cudaFree(device_result);
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(device_chunks);
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      return {host.low, host.high, elements};
    } catch (...) {
      if (device_result) {
        (void)cudaFree(device_result);
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      }
      if (device_chunks) {
        (void)cudaFree(device_chunks);
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      }
      throw;
    }
  };

  const auto w1_elements = checked_mul(
      checked_mul(impl.owned_experts, 2ULL, "W1 hash expert rows"),
      checked_mul(impl.intermediate, impl.hidden, "W1 hash expert elements"),
      "W1 hash elements");
  const auto w2_elements = checked_mul(
      impl.owned_experts,
      checked_mul(impl.hidden, impl.intermediate,
                  "W2 hash expert elements"),
      "W2 hash elements");
  ForwardWeightStorageHashes hashes;
  hashes.w1 = hash_one(impl.w1, w1_elements,
                       "hash_forward_weight_chunks_kernel(W1)");
  hashes.w2 = hash_one(impl.w2, w2_elements,
                       "hash_forward_weight_chunks_kernel(W2)");
  const auto w1_expert_elements = checked_mul(
      2ULL, checked_mul(impl.intermediate, impl.hidden,
                        "sample W1 expert elements"),
      "sample W1 expert elements");
  const auto w2_expert_elements = checked_mul(
      impl.hidden, impl.intermediate, "sample W2 expert elements");
  std::array<std::uint32_t, 3> sampled_experts{
      0U, impl.owned_experts / 2U, impl.owned_experts - 1U};
  for (const auto local_expert : sampled_experts) {
    hashes.sampled_w1_experts.push_back(hash_one(
        impl.w1 + local_expert * w1_expert_elements, w1_expert_elements,
        "hash_forward_weight_chunks_kernel(W1 expert)"));
    hashes.sampled_w2_experts.push_back(hash_one(
        impl.w2 + local_expert * w2_expert_elements, w2_expert_elements,
        "hash_forward_weight_chunks_kernel(W2 expert)"));
  }
  std::array<std::uint64_t, 4> sampled_rows{
      0ULL, 1ULL, impl.hidden / 2ULL, impl.hidden - 1ULL};
  for (const auto row : sampled_rows)
    hashes.sampled_w2_output_rows.push_back(hash_one(
        impl.w2 + row * impl.intermediate, impl.intermediate,
        "hash_forward_weight_chunks_kernel(W2 output row)"));
  return hashes;
}

constexpr std::uint64_t kLocalP2pBootstrapMagic = 0x31504350454f4d4dULL;
constexpr std::uint64_t kLocalP2pBootstrapSchema = 1;
constexpr std::size_t kLocalP2pBootstrapHeaderBytes = 96;

void put_u64_le(std::vector<std::byte> &wire, std::size_t offset,
                std::uint64_t value) {
  if (offset > wire.size() || wire.size() - offset < sizeof(value))
    transport_error(NotifyErrorCode::ArithmeticOverflow,
                    "local-p2p-bootstrap", "wire offset", 0,
                    "8 writable bytes", "out of range");
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    wire[offset + byte] =
        static_cast<std::byte>((value >> (byte * 8U)) & 0xffU);
}

std::uint64_t get_u64_le(const std::byte *wire, std::size_t bytes,
                         std::size_t offset, std::uint32_t rank,
                         const char *field) {
  if (offset > bytes || bytes - offset < sizeof(std::uint64_t))
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "local-p2p-bootstrap", field, rank, "8 readable bytes",
                    "truncated");
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(wire[offset + byte]) << (byte * 8U);
  return value;
}

std::array<std::uint64_t, 2>
session_summary(const std::vector<std::byte> &identity) {
  std::array<std::uint64_t, 2> hash{1469598103934665603ULL,
                                    1099511628211ULL};
  for (const auto byte : identity) {
    const auto value = static_cast<std::uint8_t>(byte);
    hash[0] = (hash[0] ^ value) * 1099511628211ULL;
    hash[1] = (hash[1] ^ static_cast<std::uint8_t>(value + 0x9dU)) *
              1469598103934665603ULL;
  }
  return hash;
}

