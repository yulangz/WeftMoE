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
  // The compact expert counts are immutable Dispatch metadata. Backward
  // lazily snapshots the tiny vector before its first timed launch and reuses
  // it for repeated executions of the same retained Forward state.
  mutable std::mutex backward_plan_mutex;
  mutable std::vector<std::uint64_t> backward_expert_rows;
  mutable std::vector<std::uint64_t> backward_expert_offsets;
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

std::shared_ptr<LocalP2pBackendState> create_local_p2p_backend(
    const NcclSession &session, const CudaExecution &execution,
    const std::shared_ptr<ActivationPoolState> &pool) {
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create",
                    "session/execution/pool", 0, "valid", "empty");
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (session.session().stable_identity_bytes() != pool->session_identity ||
      session.session().world_size() != pool->world ||
      session.device_ordinal() != pool->device ||
      execution.device_ordinal() != pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  int device_count = 0;
  internal::throw_cuda(
      "local-p2p-create", cudaGetDeviceCount(&device_count), session.rank(),
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "cudaGetDeviceCount");
  if (nccl->world > static_cast<std::uint32_t>(device_count))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-create, field=local_world, rank=" +
            std::to_string(session.rank()) + ", expected<=device_count=" +
            std::to_string(device_count) + ", actual=" +
            std::to_string(nccl->world));

  const auto slot_count = pool->slots.size();
  const auto handles_bytes = checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                                         "IPC handle bytes");
  const auto record_bytes = checked_size(
      kLocalP2pBootstrapHeaderBytes + handles_bytes, "IPC record bytes");
  const auto matrix_bytes = checked_size(
      checked_mul(record_bytes, nccl->world, "IPC matrix bytes"),
      "IPC matrix bytes");
  std::vector<std::byte> local(record_bytes);
  put_u64_le(local, 0, kLocalP2pBootstrapMagic);
  put_u64_le(local, 8, kLocalP2pBootstrapSchema);
  put_u64_le(local, 16, record_bytes);
  put_u64_le(local, 24, session.rank());
  put_u64_le(local, 32, nccl->world);
  put_u64_le(local, 40, pool->device);
  put_u64_le(local, 48, slot_count);
  put_u64_le(local, 56, pool->layout.total);
  put_u64_le(local, 64, session.session().generation());
  const auto identity_hash =
      session_summary(session.session().stable_identity_bytes());
  put_u64_le(local, 72, identity_hash[0]);
  put_u64_le(local, 80, identity_hash[1]);
  put_u64_le(local, 88, 0);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    cudaIpcMemHandle_t handle{};
    internal::throw_cuda(
        "local-p2p-create",
        cudaIpcGetMemHandle(&handle, pool->slots[slot].device), session.rank(),
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaIpcGetMemHandle(pool slot)");
    std::memcpy(local.data() + kLocalP2pBootstrapHeaderBytes +
                    slot * sizeof(handle),
                &handle, sizeof(handle));
  }

  void *device_send = nullptr;
  void *device_receive = nullptr;
  std::vector<std::byte> matrix(matrix_bytes);
  auto exec = internal::checked_execution_state(execution);
  try {
    internal::throw_cuda(
        "local-p2p-create", cudaMalloc(&device_send, record_bytes),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(IPC bootstrap send)");
    internal::throw_cuda(
        "local-p2p-create", cudaMalloc(&device_receive, matrix_bytes),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(IPC bootstrap receive)");
    internal::throw_cuda(
        "local-p2p-create",
        cudaMemcpyAsync(device_send, local.data(), record_bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(IPC bootstrap send)");
    internal::throw_nccl(
        "local-p2p-create",
        ncclAllGather(device_send, device_receive, record_bytes, ncclUint8,
                      nccl->comm, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllGather(IPC bootstrap)");
    internal::throw_cuda(
        "local-p2p-create",
        cudaMemcpyAsync(matrix.data(), device_receive, matrix_bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(IPC bootstrap receive)");
    internal::throw_cuda(
        "local-p2p-create", cudaStreamSynchronize(exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(IPC bootstrap)");
  } catch (...) {
    if (device_receive)
      (void)cudaFree(device_receive);
    if (device_send)
      (void)cudaFree(device_send);
    throw;
  }
  (void)cudaFree(device_receive);
  (void)cudaFree(device_send);

  auto backend = std::make_shared<LocalP2pBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  const auto mapping_count = checked_size(
      checked_mul(slot_count, nccl->world, "IPC mapping count"),
      "IPC mapping count");
  backend->peer_slot_bases.assign(mapping_count, nullptr);
  backend->owns_mapping.assign(mapping_count, 0);
  try {
    std::vector<std::uint8_t> seen_devices(
        static_cast<std::size_t>(device_count), 0);
    for (std::uint32_t rank = 0; rank < nccl->world; ++rank) {
      const auto *record = matrix.data() +
                           static_cast<std::size_t>(rank) * record_bytes;
      const auto check = [&](std::size_t offset, std::uint64_t expected,
                             const char *field) {
        const auto actual = get_u64_le(record, record_bytes, offset,
                                       session.rank(), field);
        if (actual != expected)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "local-p2p-bootstrap", field, session.rank(),
                          std::to_string(expected), std::to_string(actual));
      };
      check(0, kLocalP2pBootstrapMagic, "magic");
      check(8, kLocalP2pBootstrapSchema, "schema");
      check(16, record_bytes, "record_bytes");
      check(24, rank, "rank");
      check(32, nccl->world, "world");
      check(48, slot_count, "slot_count");
      check(56, pool->layout.total, "arena_bytes");
      check(64, session.session().generation(), "generation");
      check(72, identity_hash[0], "session_digest0");
      check(80, identity_hash[1], "session_digest1");
      check(88, 0, "reserved");
      const auto peer_device = get_u64_le(record, record_bytes, 40,
                                          session.rank(), "device");
      if (peer_device >= static_cast<std::uint64_t>(device_count) ||
          seen_devices[static_cast<std::size_t>(peer_device)] != 0)
        transport_error(NotifyErrorCode::TransportProtocolFailure,
                        "local-p2p-bootstrap", "device", session.rank(),
                        "unique local device ordinal",
                        std::to_string(peer_device));
      seen_devices[static_cast<std::size_t>(peer_device)] = 1;
      int access = 0;
      if (peer_device != pool->device) {
        internal::throw_cuda(
            "local-p2p-create",
            cudaDeviceCanAccessPeer(&access, static_cast<int>(pool->device),
                                    static_cast<int>(peer_device)),
            session.rank(), nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaDeviceCanAccessPeer");
        if (access == 0)
          throw NotifyDispatchError(
              NotifyErrorCode::BackendUnavailable,
              "phase=local-p2p-create, field=peer_access, rank=" +
                  std::to_string(session.rank()) + ", expected=1, actual=0, "
                  "peer_rank=" + std::to_string(rank));
      }
      for (std::size_t slot = 0; slot < slot_count; ++slot) {
        const auto index = slot * nccl->world + rank;
        if (rank == session.rank()) {
          backend->peer_slot_bases[index] = pool->slots[slot].device;
          continue;
        }
        cudaIpcMemHandle_t handle{};
        std::memcpy(&handle,
                    record + kLocalP2pBootstrapHeaderBytes +
                        slot * sizeof(handle),
                    sizeof(handle));
        internal::throw_cuda(
            "local-p2p-create",
            cudaIpcOpenMemHandle(&backend->peer_slot_bases[index], handle,
                                 cudaIpcMemLazyEnablePeerAccess),
            session.rank(), nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaIpcOpenMemHandle(peer slot)");
        backend->owns_mapping[index] = 1;
      }
    }
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      internal::throw_cuda(
          "local-p2p-create",
          cudaMemcpyAsync(
              at<void *>(pool->slots[slot].device, pool->layout.peer_bases),
              backend->peer_slot_bases.data() + slot * nccl->world,
              nccl->world * sizeof(void *), cudaMemcpyHostToDevice,
              exec->stream),
          session.rank(), nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "cudaMemcpyAsync(peer base table)");
    }
    internal::throw_cuda(
        "local-p2p-create", cudaStreamSynchronize(exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(peer table)");
  } catch (...) {
    backend.reset();
    throw;
  }
  return backend;
}

std::shared_ptr<GinBackendState> create_gin_backend(
    const NcclSession &session, const CudaExecution &execution,
    const std::shared_ptr<ActivationPoolState> &pool) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)pool;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-create, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                    "session/execution/pool", 0, "valid", "empty");
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (!pool->config.enable_gin_rdma)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-create",
                    "pool.enable_gin_rdma", session.rank(), "true", "false");
  if (session.session().stable_identity_bytes() != pool->session_identity ||
      session.session().world_size() != pool->world ||
      session.device_ordinal() != pool->device ||
      execution.device_ordinal() != pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    if (auto cached_base = pool->gin_backend.lock()) {
      auto cached = std::dynamic_pointer_cast<GinBackendState>(cached_base);
      if (!cached || cached->nccl.get() != nccl.get() ||
          !cached->execution.token().same_instance(execution.token()))
        transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                        "cached backend provenance", session.rank(),
                        "same communicator/execution", "mismatch");
      return cached;
    }
  }

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [&](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-create, field=symbol, rank=") +
              std::to_string(session.rank()) + ", expected=" + name +
              ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  internal::throw_nccl(
      "gin-create", query_properties(nccl->comm, &properties), nccl->rank,
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "ncclCommQueryProperties");
  if (properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=railed_gin, rank=" +
            std::to_string(session.rank()) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (pool->config.max_nodes == 0 ||
      nccl->world % pool->config.max_nodes != 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-create",
                    "homogeneous topology", nccl->rank,
                    "world divisible by runtime nodes", "mismatch");
  const auto num_nodes = pool->config.max_nodes;
  const auto local_world = nccl->world / num_nodes;
  const auto node = nccl->rank / local_world;
  const auto local_rank = nccl->rank % local_world;
  if (rail.nRanks != static_cast<int>(num_nodes) ||
      lsa.nRanks != static_cast<int>(local_world) ||
      pool->device != local_rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=topology, rank=" +
            std::to_string(session.rank()) +
            ", expected=rail=nodes, lsa=local_world, device=local_rank, "
            "actual=world=" + std::to_string(nccl->world) + ", rail=" +
            std::to_string(rail.nRanks) + ", lsa=" +
            std::to_string(lsa.nRanks) + ", device=" +
            std::to_string(pool->device) + ", local_rank=" +
            std::to_string(local_rank));

  auto backend = std::make_shared<GinBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  backend->num_nodes = num_nodes;
  backend->local_world_size = local_world;
  backend->node = node;
  backend->local_rank = local_rank;

  // Exchange every compute-arena IPC handle collectively, but open only the
  // ranks in this node. The full-world pointer table keeps global rank as the
  // stable index consumed by Notify fan-out descriptors.
  const auto slot_count = pool->slots.size();
  const auto handles_per_rank_bytes = checked_size(
      checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                  "GIN IPC handles per rank"),
      "GIN IPC handles per rank");
  const auto matrix_bytes = checked_size(
      checked_mul(handles_per_rank_bytes, nccl->world,
                  "GIN IPC handle matrix"),
      "GIN IPC handle matrix");
  std::vector<cudaIpcMemHandle_t> local_handles(slot_count);
  std::vector<cudaIpcMemHandle_t> handle_matrix(slot_count * nccl->world);
  for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index)
    internal::throw_cuda(
        "gin-create",
        cudaIpcGetMemHandle(&local_handles[slot_index],
                            pool->slots[slot_index].device),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaIpcGetMemHandle(GIN local compute arena)");
  void *device_handles = nullptr;
  void *device_matrix = nullptr;
  try {
    internal::throw_cuda(
        "gin-create", cudaMalloc(&device_handles, handles_per_rank_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN IPC handles)");
    internal::throw_cuda(
        "gin-create", cudaMalloc(&device_matrix, matrix_bytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN IPC matrix)");
    const auto exec = internal::checked_execution_state(execution);
    internal::throw_cuda(
        "gin-create",
        cudaMemcpyAsync(device_handles, local_handles.data(),
                        handles_per_rank_bytes, cudaMemcpyHostToDevice,
                        exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "H2D GIN IPC handles");
    internal::throw_nccl(
        "gin-create",
        ncclAllGather(device_handles, device_matrix, handles_per_rank_bytes,
                      ncclUint8, nccl->comm, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllGather(GIN IPC handles)");
    internal::throw_cuda(
        "gin-create",
        cudaMemcpyAsync(handle_matrix.data(), device_matrix, matrix_bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "D2H GIN IPC matrix");
    internal::throw_cuda(
        "gin-create", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "wait GIN IPC bootstrap");
  } catch (...) {
    if (device_matrix != nullptr)
      (void)cudaFree(device_matrix);
    if (device_handles != nullptr)
      (void)cudaFree(device_handles);
    throw;
  }
  (void)cudaFree(device_matrix);
  (void)cudaFree(device_handles);
  backend->peer_slot_bases.assign(slot_count * nccl->world, nullptr);
  backend->owns_mapping.assign(slot_count * nccl->world, 0);
  try {
    for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
      for (std::uint32_t peer = node * local_world;
           peer < (node + 1U) * local_world; ++peer) {
        const auto index = slot_index * nccl->world + peer;
        if (peer == nccl->rank) {
          backend->peer_slot_bases[index] = pool->slots[slot_index].device;
        } else {
          internal::throw_cuda(
              "gin-create",
              cudaIpcOpenMemHandle(
                  &backend->peer_slot_bases[index],
                  handle_matrix[static_cast<std::size_t>(peer) * slot_count +
                                slot_index],
                  cudaIpcMemLazyEnablePeerAccess),
              nccl->rank, nccl->world, session.session().identity(),
              session.session().generation(), ExchangeKind::ExpertCounts,
              "cudaIpcOpenMemHandle(GIN local peer arena)");
          backend->owns_mapping[index] = 1;
        }
      }
      const auto exec = internal::checked_execution_state(execution);
      internal::throw_cuda(
          "gin-create",
          cudaMemcpyAsync(
              at<void *>(pool->slots[slot_index].device,
                         pool->layout.peer_bases),
              backend->peer_slot_bases.data() + slot_index * nccl->world,
              nccl->world * sizeof(void *), cudaMemcpyHostToDevice,
              exec->stream),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "H2D GIN local peer table");
    }
    internal::throw_cuda(
        "gin-create",
        cudaStreamSynchronize(
            internal::checked_execution_state(execution)->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "wait GIN local peer table");
  } catch (...) {
    backend.reset();
    throw;
  }
  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  backend->full_connections = true;
  constexpr std::uint32_t kGinContextCount = 8;
  // FULL Return must not share an exclusive context with an outstanding
  // same-rail
  // Dispatch queue.  Keep the logical topology at eight contexts/CTAs, but
  // give every FULL owner a second physical QP: [0, 8) is Dispatch and
  // [8, 16) is Return.  DeepEP routinely allocates more than this, and the
  // separation lets a single permanent RDMA CTA progress both queues without
  // mixing teams on one context.
  const auto allocated_context_count = 2U * kGinContextCount;
  requirements.ginContextCount = allocated_context_count;
  // One permanent RDMA CTA owns a same-rail Dispatch QP and an independent
  // FULL Return QP. Compute CTAs publish byte-range descriptors and never
  // touch a GIN context.
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 64;
  // 0=dispatch arrival, 1=return arrival.
  // Signal 0/1 are the data-plane dispatch/return epochs. Signal 2 is the
  // fixed terminal production-status epoch and is touched only once per
  // generation, never on the per-task success path.
  requirements.ginSignalCount = 3;
  // Counter 0 belongs to Dispatch; direct/stream Return overlaps it and uses
  // counter 1.  Request both explicitly (the previous count of one made every
  // counter-1 access formally out of bounds).
  requirements.ginCounterCount = 2;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  internal::throw_nccl(
      "gin-create",
      dev_comm_create(nccl->comm, &requirements, &backend->dev_comm),
      nccl->rank, nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "ncclDevCommCreate(activation GIN)");
  backend->dev_comm_created = true;
  backend->gin_context_count = kGinContextCount;
  backend->gin_allocated_context_count = allocated_context_count;
  try {
    backend->windows.reserve(pool->slots.size());
    for (auto &slot : pool->slots) {
      ncclWindow_t window = nullptr;
      if (slot.gin_window == nullptr || pool->gin_layout.total == 0)
        transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                        "registered window storage", nccl->rank,
                        "preallocated ncclMemAlloc window", "empty");
      internal::throw_nccl(
          "gin-create",
          window_register(nccl->comm, slot.gin_window,
                          pool->gin_layout.total, &window,
                          NCCL_WIN_STRICT_ORDERING),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "ncclCommWindowRegister(activation pool slot)");
      backend->windows.push_back(window);
    }
  } catch (...) {
    backend.reset();
    throw;
  }
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    pool->gin_backend = backend;
  }
  return backend;
#endif
}

ActivationTransportResourcePool make_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config) {
  if (!session || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-create",
                    "session/execution", 0, "valid", "empty");
  if (config.max_world_size < session.world_size())
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "max_world_size", 0,
                    ">= session world", std::to_string(config.max_world_size));
  auto state = std::make_shared<ActivationPoolState>();
  state->config = config;
  state->layout = make_layout(config);
  state->gin_layout = make_gin_window_layout(config);
  state->session_identity = session.stable_identity_bytes();
  state->world = session.world_size();
  state->execution = execution;
  state->device = execution.device_ordinal();
  internal::throw_cuda("transport-pool-create",
                       cudaSetDevice(static_cast<int>(state->device)), 0,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
  if (config.enable_backward_state) {
    throw_cublas("transport-pool-create",
                 cublasCreate(&state->backward_blas), 0,
                 "cublasCreate(Backward)");
    throw_cublas("transport-pool-create",
                 cublasSetAtomicsMode(state->backward_blas,
                                      CUBLAS_ATOMICS_NOT_ALLOWED),
                 0, "cublasSetAtomicsMode(Backward deterministic)");
  }
#if MEGAMOE_DEVICE_TIMELINE
  cudaDeviceProp timeline_device_properties{};
  internal::throw_cuda(
      "transport-pool-create",
      cudaGetDeviceProperties(&timeline_device_properties,
                              static_cast<int>(state->device)),
      0, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts,
      "cudaGetDeviceProperties(device timeline)");
  if (timeline_device_properties.multiProcessorCount <= 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "timeline CTA capacity", 0,
                    "positive SM count",
                    std::to_string(timeline_device_properties.multiProcessorCount));
  // Forward's transport, FC1 and FC2 grids execute concurrently.  Give their
  // lock-free trace writers disjoint logical CTA ranges; one-SM capacity was
  // sufficient only while FC1 and FC2 were accidentally stream-serialized.
  state->timeline_max_ctas =
      3U * static_cast<std::uint32_t>(
               timeline_device_properties.multiProcessorCount);
  state->timeline_layout =
      make_timeline_allocation_layout(state->timeline_max_ctas);
#endif
  state->slots.resize(config.max_inflight_generations);
  try {
    internal::throw_cuda(
        "transport-pool-create",
        cudaStreamCreateWithFlags(&state->cancel_stream, cudaStreamNonBlocking),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaStreamCreate(cancel)");
    for (auto &slot : state->slots) {
      internal::throw_cuda(
          "transport-pool-create", cudaMalloc(&slot.device,
                                                state->layout.total),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(transport slot)");
      if (config.enable_gin_rdma) {
        using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
        const auto mem_alloc = reinterpret_cast<MemAllocFn>(
            dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
        if (mem_alloc == nullptr)
          throw NotifyDispatchError(
              NotifyErrorCode::BackendUnavailable,
              "phase=transport-pool-create, field=ncclMemAlloc, rank=0, "
              "expected=runtime symbol, actual=missing");
        internal::throw_nccl(
            "transport-pool-create",
            mem_alloc(&slot.gin_window, state->gin_layout.total), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, "ncclMemAlloc(GIN window slot)");
      }
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
#if MEGAMOE_DEVICE_TIMELINE
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(&slot.forward_timeline, state->timeline_layout.total),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(Forward timeline)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.forward_timeline, 0,
                          state->timeline_layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize Forward timeline");
#endif
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device, 0, state->layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize complete transport slot storage");
      if (slot.gin_window)
        internal::throw_cuda(
            "transport-pool-create",
            cudaMemsetAsync(slot.gin_window, 0, state->gin_layout.total,
                            internal::checked_execution_state(execution)->stream),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "one-time initialize GIN window slot storage");
      if (config.max_intermediate_size != 0 &&
          config.max_hidden_size == 7168 &&
          config.max_intermediate_size == 3072) {
        slot.forward_x_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank,
            forward_tuning::kTileM, "canonical X", forward_tuning::kTileK);
        slot.forward_s_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size, config.max_destination_rows_per_rank,
            forward_tuning::kTileM, "gated SwiGLU U",
            forward_tuning::kTileK);
        slot.forward_x_m64_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank, 64,
            "canonical X M64");
        slot.forward_s_m64_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size, config.max_destination_rows_per_rank,
            64, "gated SwiGLU U M64");
        slot.forward_output_store_map = make_bf16_tma_store_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank,
            "Forward canonical output");
        if (config.enable_backward_state)
          slot.forward_training_output_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Forward training output");
        if (slot.gin_window != nullptr)
          slot.forward_staging_store_map = make_bf16_tma_store_map(
              at<void>(slot.gin_window, state->gin_layout.return_staging),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Forward GIN return staging");
        slot.forward_maps_valid = true;
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(transport status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_forward_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(forward status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(
              reinterpret_cast<void **>(&slot.host_production_status),
              checked_size(checked_mul(config.max_world_size,
                                       sizeof(ProductionStatusRecord),
                                       "production status host matrix"),
                           "production status host matrix"),
              cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(production terminal status matrix)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      std::fill_n(slot.host_production_status, config.max_world_size,
                  ProductionStatusRecord{});
      slot.generation_debug_contexts = static_cast<std::uint32_t>(
          (config.max_nodes > 0 ? config.max_nodes - 1ULL : 0ULL) * 8ULL);
      slot.generation_debug_words = transport_state::completion_debug_words(
          slot.generation_debug_contexts);
      slot.forward_progress_words =
          kForwardProgressWords + 2ULL * slot.generation_debug_words;
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(
              reinterpret_cast<void **>(&slot.host_forward_progress),
              slot.forward_progress_words * sizeof(std::uint64_t),
              cudaHostAllocPortable | cudaHostAllocMapped),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(mapped Forward progress)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostGetDevicePointer(
              reinterpret_cast<void **>(&slot.device_forward_progress),
              slot.host_forward_progress, 0),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostGetDevicePointer(Forward progress)");
      std::fill_n(slot.host_forward_progress, slot.forward_progress_words,
                  0ULL);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_cancel_epoch),
                        sizeof(std::uint64_t),
                        cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      *slot.host_cancel_epoch = 0;
      if (config.enable_socket_fallback) {
        const auto socket_count_bytes = checked_size(
            checked_mul(checked_mul(config.max_world_size,
                                    config.max_world_size,
                                    "Socket host count matrix"),
                        sizeof(std::uint64_t),
                        "Socket host count matrix bytes"),
            "Socket host count matrix bytes");
        internal::throw_cuda(
            "transport-pool-create",
            cudaHostAlloc(reinterpret_cast<void **>(&slot.host_socket_counts),
                          socket_count_bytes, cudaHostAllocPortable),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "cudaHostAlloc(Socket count matrix)");
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
        std::memset(slot.host_socket_counts, 0, socket_count_bytes);
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(reinterpret_cast<void **>(&slot.device_cancel_epoch),
                     sizeof(std::uint64_t)),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device_cancel_epoch, 0, sizeof(std::uint64_t),
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize device cancel epoch");
      auto create_event = [&](cudaEvent_t *event, const char *name) {
        internal::throw_cuda(
            "transport-pool-create",
            cudaEventCreateWithFlags(event, cudaEventDisableTiming), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      };
      auto create_timing_event = [&](cudaEvent_t *event, const char *name) {
        internal::throw_cuda(
            "transport-pool-create", cudaEventCreate(event), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      };
      int least_stream_priority = 0;
      int greatest_stream_priority = 0;
      internal::throw_cuda(
          "transport-pool-create",
          cudaDeviceGetStreamPriorityRange(&least_stream_priority,
                                           &greatest_stream_priority),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaDeviceGetStreamPriorityRange(Forward roles)");
      internal::throw_cuda("transport-pool-create",
                           cudaStreamCreateWithPriority(
                               &slot.forward_transport_stream,
                               cudaStreamNonBlocking, greatest_stream_priority),
                           0, session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaStreamCreateWithPriority(Forward transport)");
      internal::throw_cuda(
          "transport-pool-create",
          cudaStreamCreateWithPriority(&slot.forward_m64_head_stream,
                                       cudaStreamNonBlocking,
                                       greatest_stream_priority),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaStreamCreateWithPriority(Forward M64 head)");
      internal::throw_cuda(
          "transport-pool-create",
          cudaStreamCreateWithPriority(&slot.forward_compute_stream,
                                       cudaStreamNonBlocking,
                                       least_stream_priority),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaStreamCreateWithPriority(Forward compute)");
      create_event(&slot.producer_ready, "cudaEventCreate(producer-ready)");
      create_event(&slot.forward_roles_ready,
                   "cudaEventCreate(Forward roles ready)");
      create_event(&slot.forward_transport_done,
                   "cudaEventCreate(Forward transport done)");
      create_event(&slot.forward_m64_head_done,
                   "cudaEventCreate(Forward M64 head done)");
      create_event(&slot.forward_compute_done,
                   "cudaEventCreate(Forward compute done)");
      if (config.enable_backward_state) {
        create_timing_event(&slot.backward_start,
                            "cudaEventCreate(Backward start)");
        create_timing_event(&slot.backward_dispatch_done,
                            "cudaEventCreate(Backward dispatch done)");
        create_timing_event(&slot.backward_dab_done,
                            "cudaEventCreate(Backward dAB done)");
        create_timing_event(&slot.backward_dx_done,
                            "cudaEventCreate(Backward dX done)");
        create_timing_event(&slot.backward_dw2_done,
                            "cudaEventCreate(Backward dW2 done)");
        create_timing_event(&slot.backward_dw1_done,
                            "cudaEventCreate(Backward dW1 done)");
        create_timing_event(&slot.backward_return_done,
                            "cudaEventCreate(Backward return done)");
        create_timing_event(&slot.backward_complete,
                            "cudaEventCreate(Backward complete)");
      }
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.forward_start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaEventCreate(&slot.forward_output_ready), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaEventCreate(forward output ready)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.forward_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.dispatch_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(dispatch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    }
  } catch (...) {
    throw;
  }
  auto impl = std::make_shared<ActivationTransportResourcePool::Impl>();
  impl->state = std::move(state);
  return ActivationTransportResourcePool(std::move(impl));
}

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
