#include "cuda/activation_transport_internal.hpp"
#include "cuda/communication/chunk_ring.cuh"
#include "cuda/communication/local_notification.cuh"
#include "cuda/forward/transport_launch.hpp"
#include "cuda/primitives/hopper.cuh"
#include "cuda/primitives/programmatic_launch.cuh"

#include <cuda.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <nccl.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#if __has_include(<nccl_device/core.h>)
#include <nccl_device/core.h>
#include <nccl_device/impl/comm__types.h>
#endif
#if __has_include(<nccl_device.h>)
#include <nccl_device.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <functional>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

#include "cuda/cuda_internal.hpp"
#include "cuda/backward/tuning_config.hpp"
#include "cuda/persistent_v3/backward/pipeline_state.cuh"
#include "cuda/persistent_v3/backward/scheduler.cuh"
#include "cuda/persistent_v3/contracts.hpp"
#include "cuda/persistent_v3/forward/pipeline_state.cuh"
#include "cuda/persistent_v3/queues.cuh"
#include "cuda/persistent_v3/roles/common.cuh"
#include "cuda/profiling/device_timeline.cuh"
#include "cuda/transport_generation_completion.cuh"
#include "weftmoe/api/errors.hpp"

#ifndef WEFTMOE_FORWARD_LIGHT_PROGRESS
#define WEFTMOE_FORWARD_LIGHT_PROGRESS 0
#endif

namespace weftmoe::cuda {
struct ActivationMegaKernelBackendState;
namespace {

#include "cuda/forward/device_kernels.cuh"

#include "cuda/backward/device_kernels.cuh"

#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
__global__ void fill_transport_benchmark_output(transport::BFloat16 *output,
                                               unsigned long long count) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += gridDim.x * blockDim.x)
    output[i] = transport::BFloat16{
        static_cast<std::uint16_t>(0x3c00U + (i & 255U))};
}
#endif

} // namespace

#include "cuda/tensor_storage.cuh"

struct ActivationTransportResourcePool::Impl {
  std::shared_ptr<ActivationPoolState> state;
};

struct InputTensorAllocation;

struct SelectedActivationLease<transport::BFloat16>::Impl {
  std::shared_ptr<InputTensorAllocation> allocation;
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
};

struct InputTensorAllocation {
  void *data = nullptr;
  cudaEvent_t ready = nullptr;
  unsigned long long ready_capture_id = 0;
  std::uint32_t device = 0;
  bool owns_data = true;
  std::shared_ptr<const void> external_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;

  ~InputTensorAllocation() noexcept {
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
  std::shared_ptr<InputTensorAllocation> allocation;
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

  // A backend owns one device/context lifetime. Cache successful checks by
  // entry pointer and phase quotas so local/GIN and retained-state variants
  // are checked independently, without a CUDA query on every execution.
  std::mutex unified_register_mutex;
  struct UnifiedRegisterKey {
    const void *kernel;
    unsigned threads, math, producer, control;
    bool operator==(const UnifiedRegisterKey &other) const {
      return kernel == other.kernel && threads == other.threads &&
             math == other.math && producer == other.producer &&
             control == other.control;
    }
  };
  std::vector<UnifiedRegisterKey> validated_unified_register_kernels;

  std::mutex unified_launch_mutex;
  struct UnifiedLaunchKey {
    const void *kernel;
    std::size_t dynamic_smem;
    unsigned grid_blocks, block_threads, cluster_size;
    bool programmatic, cooperative;
    bool operator==(const UnifiedLaunchKey &other) const {
      return kernel == other.kernel && dynamic_smem == other.dynamic_smem &&
             grid_blocks == other.grid_blocks &&
             block_threads == other.block_threads &&
             cluster_size == other.cluster_size &&
             programmatic == other.programmatic &&
             cooperative == other.cooperative;
    }
  };
  struct UnifiedLaunchPreparation {
    UnifiedLaunchKey key;
    int max_active_clusters;
  };
  std::vector<UnifiedLaunchPreparation> prepared_unified_launches;

  template <class Resources>
  cudaError_t validate_unified_register_pool(const void *kernel) {
    const UnifiedRegisterKey key{
        kernel, Resources::kUnifiedThreads, Resources::kUnifiedMathRegisters,
        Resources::kUnifiedProducerRegisters, Resources::kUnifiedControlRegisters};
    std::lock_guard<std::mutex> lock(unified_register_mutex);
    if (std::find(validated_unified_register_kernels.begin(),
                  validated_unified_register_kernels.end(), key) !=
        validated_unified_register_kernels.end())
      return cudaSuccess;
    cudaFuncAttributes attributes{};
    const auto error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess)
      return error;
    if (attributes.numRegs <= 0 || attributes.numRegs > 256)
      return cudaErrorLaunchOutOfResources;
    // CUDA's Hopper occupancy calculation allocates 256 registers per warp:
    // round the compiled count UP to eight registers per thread. The launch
    // bound's theoretical maximum is not proof that ptxas used that pool.
    const unsigned initial =
        (static_cast<unsigned>(attributes.numRegs) + 7U) / 8U * 8U;
    constexpr unsigned requested =
        256U * Resources::kUnifiedMathRegisters +
        128U * Resources::kUnifiedProducerRegisters +
        (Resources::kUnifiedThreads - 384U) *
            Resources::kUnifiedControlRegisters;
    // PTX .inc/.dec must also respect the previous per-thread maximum.
    // An insufficient CTA pool can stall .inc indefinitely even when the
    // ordinary occupancy check says that the kernel can launch.
    if (Resources::kUnifiedMathRegisters < initial ||
        Resources::kUnifiedProducerRegisters > initial ||
        Resources::kUnifiedControlRegisters > initial ||
        requested > initial * Resources::kUnifiedThreads)
      return cudaErrorLaunchOutOfResources;
    validated_unified_register_kernels.push_back(key);
    return cudaSuccess;
  }

  // Function attributes, stack limits and cooperative-cluster occupancy are
  // invariant for a backend's device/context and a fixed kernel geometry.
  // Avoid repeating these driver queries after forward_start on every small
  // Forward invocation.
  template <class Resources>
  cudaError_t prepare_unified_launch(const void *kernel,
                                     std::size_t dynamic_smem,
                                     unsigned grid_blocks,
                                     unsigned block_threads,
                                     unsigned cluster_size,
                                     bool programmatic,
                                     bool cooperative_launch,
                                     int *max_active_clusters) {
    auto error = validate_unified_register_pool<Resources>(kernel);
    if (error != cudaSuccess)
      return error;
    const UnifiedLaunchKey key{kernel, dynamic_smem, grid_blocks,
                               block_threads, cluster_size, programmatic,
                               cooperative_launch};
    std::lock_guard<std::mutex> lock(unified_launch_mutex);
    const auto found = std::find_if(
        prepared_unified_launches.begin(), prepared_unified_launches.end(),
        [&](const UnifiedLaunchPreparation &entry) {
          return entry.key == key;
        });
    if (found != prepared_unified_launches.end()) {
      *max_active_clusters = found->max_active_clusters;
      return cudaSuccess;
    }
    error = internal::prepare_kernel_stack(kernel);
    if (error != cudaSuccess)
      return error;
    error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem));
    if (error != cudaSuccess)
      return error;
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(grid_blocks, 1, 1);
    launch.blockDim = dim3(block_threads, 1, 1);
    launch.dynamicSmemBytes = dynamic_smem;
    cudaLaunchAttribute cluster{};
    cluster.id = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim = {cluster_size, 1U, 1U};
    cudaLaunchAttribute cooperative{};
    cooperative.id = cudaLaunchAttributeCooperative;
    cooperative.val.cooperative = 1;
    cudaLaunchAttribute attributes[3]{};
    unsigned attribute_count = 0U;
    attributes[attribute_count++] = cluster;
    if (cooperative_launch)
      attributes[attribute_count++] = cooperative;
    if (programmatic)
      attributes[attribute_count++] = programmatic_launch::attribute();
    launch.attrs = attributes;
    launch.numAttrs = attribute_count;
    int clusters = 0;
    error = cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch);
    if (error != cudaSuccess)
      return error;
    prepared_unified_launches.push_back({key, clusters});
    *max_active_clusters = clusters;
    return cudaSuccess;
  }

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
  std::shared_ptr<PeerMappings> mappings;

  bool requires_collective_terminal_status() const noexcept override {
    return false;
  }

  ~LocalP2pBackendState() noexcept {
    if (mappings || !pool)
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
  std::shared_ptr<PeerMappings> mappings;
  std::uint32_t num_nodes = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t node = 0;
  std::uint32_t local_rank = 0;
  std::uint32_t gin_context_count = 1;
  std::uint32_t gin_allocated_context_count = 1;
  bool full_connections = false;
  std::shared_ptr<GinResources> resources;

  bool requires_collective_terminal_status() const noexcept override {
    return false;
  }

  ~GinBackendState() noexcept {
    if (mappings || !nccl)
      return;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    for (std::size_t index = 0; index < peer_slot_bases.size(); ++index)
      if (owns_mapping[index] != 0 && peer_slot_bases[index] != nullptr)
        (void)cudaIpcCloseMemHandle(peer_slot_bases[index]);
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
  void *w1_interleaved = nullptr;
  void *w2 = nullptr;
  std::uint64_t bytes = 0;
  std::uint32_t device = 0;
  cudaEvent_t ready = nullptr;
  unsigned long long ready_capture_id = 0;
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
    if (w1_interleaved) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w1_interleaved);
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
  const transport::BFloat16 *w1_interleaved = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken execution;
  std::uint32_t owned_experts = 0;
  std::uint32_t global_expert_begin = 0;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint32_t device = 0;
  CUtensorMap w1_map{};
  CUtensorMap w1_interleaved_map{};
  CUtensorMap w2_map{};
  CUtensorMap prefill_w1_map{};
  CUtensorMap prefill_w2_map{};
  CUtensorMap backward_w1_map{};
  CUtensorMap backward_w2_map{};

  void initialize_prefill_maps() {
    if constexpr (forward_tuning::kStaticV4PrefillMinTokens) {
      using Gemm = forward_tuning::StaticV4Gemm<true>;
      prefill_w1_map = make_bf16_tma_map(
          const_cast<transport::BFloat16 *>(w1_interleaved), hidden,
          owned_experts * 2ULL * intermediate, Gemm::kFc1N,
          "prefill interleaved W1", forward_tuning::kTileK);
      prefill_w2_map = make_bf16_tma_map(
          const_cast<transport::BFloat16 *>(w2), intermediate,
          owned_experts * hidden, Gemm::kFc2N, "prefill W2",
          forward_tuning::kTileK);
    }
  }
};

struct ForwardWeightStorageHashAccess {
  static const ForwardWeightLease::Impl &get(
      const ForwardWeightLease &weights) {
    return *weights.impl_;
  }
};

// One slot can chain FWD -> BWD, but cannot execute two FWDs or two
// BWDs concurrently. The completion action retains this guard until drained.
struct OperatorFlight {
  std::shared_ptr<SlotLease> lease;
  bool backward;
  OperatorFlight(std::shared_ptr<SlotLease> value, bool is_backward)
      : lease(std::move(value)), backward(is_backward) {
    std::lock_guard<std::mutex> lock(lease->pool->mutex);
    if (lease->backward_pending || (!backward && lease->forward_pending))
      transport_error(NotifyErrorCode::ResourceExhausted, "operator-enqueue",
                      "slot", 0, "wait for the previous operation", "operation pending");
    (backward ? lease->backward_pending : lease->forward_pending) = true;
  }
  ~OperatorFlight() noexcept {
    std::lock_guard<std::mutex> lock(lease->pool->mutex);
    (backward ? lease->backward_pending : lease->forward_pending) = false;
  }
};

// Compute handles can remain alive across arena growth. Reuse the wire only
// after its previous execution has finished, without waiting on the host or
// tying the communication lease to the lifetime of a saved activation.
struct GinExecution {
  std::shared_ptr<GinResources> resources;
  std::unique_lock<std::mutex> lock;
  cudaStream_t stream;
  bool serialized;
  explicit GinExecution(std::shared_ptr<GinResources> value,
                        cudaStream_t target, bool serialized_replay = false)
      : resources(std::move(value)), stream(target),
        serialized(serialized_replay) {
    if (!resources)
      return;
    lock = std::unique_lock<std::mutex>(resources->mutex);
    // Serialized graph plans share one stream-owned workspace. The caller
    // drains it before changing streams; no per-layer event belongs in capture.
    if (serialized)
      return;
    const auto error = internal::wait_input_ready(stream, resources->last_use,
                                                  resources->last_capture_id);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("GIN reuse wait: ") +
                               cudaGetErrorString(error));
  }
  void record() {
    if (!resources || serialized)
      return;
    const auto error = internal::record_input_ready(resources->last_use, stream,
                                                    resources->last_capture_id);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("GIN reuse event: ") +
                               cudaGetErrorString(error));
  }
};

// Diagnostic completion is explicit. Destroying an unfinished handle drains it
// before its captured allocations/consumer lease are released.
struct OperatorCompletion {
  std::mutex mutex;
  std::function<void()> finish;
  std::exception_ptr error;
  void wait() {
    std::lock_guard<std::mutex> lock(mutex);
    if (finish) {
      auto action = std::move(finish);
      try { action(); } catch (...) { error = std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
  }
  ~OperatorCompletion() noexcept {
    try { wait(); } catch (...) {}
  }
};

struct DeviceForwardHandle::Impl {
  mutable std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
  DeviceDispatchHandle dispatch;
  BFloat16ActivationLease activations;
  ForwardWeightLease weights;
  CudaTensorBuffer inference_routes;
  CudaTensorBuffer inference_gates;
  ForwardTensorStorage tensors;
  std::size_t workspace_index = 0;
  ActivationTransportExpectation identity;
  ForwardMegaKernelAudit audit;
  ForwardMegaKernelTimings timings;
  bool retained_backward_state = false;
  bool serialized_graph_replay = false;
  bool compact_return = false;
  std::uint64_t incoming_rows = 0;
  std::uint64_t intermediate = 0;
  // Host-enqueued Socket transfers execute outside the persistent kernel.
  std::uint64_t socket_dispatch_peer_bytes = 0;
  std::uint64_t socket_return_peer_bytes = 0;
  std::uint64_t socket_raw_cross_node_route_bytes = 0;
  std::unique_ptr<OperatorCompletion> completion;
};

struct DeviceBackwardHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  DeviceForwardHandle forward;
  BFloat16OutputGradientLease output_gradients;
  BackwardTensorStorage tensors;
  std::size_t workspace_index = 0;
  ActivationTransportExpectation identity;
  BackwardMegaKernelAudit audit;
  BackwardMegaKernelTimings timings;
  std::uint32_t owned_experts = 0;
  std::uint64_t incoming_rows = 0;
  std::uint64_t intermediate = 0;
  std::unique_ptr<OperatorCompletion> completion;
};

struct LocalCudaP2pForwardMegaKernel::Impl {
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
};

#if WEFTMOE_DEVICE_TIMELINE
namespace {
profiling::DeviceTimelineSnapshot
download_device_timeline(const std::shared_ptr<ActivationPoolState> &pool,
                         std::size_t index, void *allocation,
                         cudaEvent_t completion,
                         const ActivationTransportExpectation &identity,
                         profiling::TimelinePhase phase) {
  if (!pool || index >= pool->slots.size() || !allocation)
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
  const auto &slot = impl.pool->slots[impl.workspace_index];
  return download_device_timeline(
      impl.pool, impl.workspace_index, slot.forward_timeline,
      slot.complete, impl.identity, profiling::TimelinePhase::Forward);
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
  allocation->bytes =
      static_cast<std::uint64_t>(2ULL * w1_bytes) + w2_bytes;
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
        "forward-weight-create",
        cudaMalloc(&allocation->w1_interleaved, w1_bytes), 0,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaMalloc(interleaved W1)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
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
        static_cast<transport::BFloat16 *>(allocation->w1),
        static_cast<transport::BFloat16 *>(allocation->w1_interleaved),
        w1_elements,
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
  impl->w1_interleaved = static_cast<const transport::BFloat16 *>(
      impl->allocation->w1_interleaved);
  impl->w2 = static_cast<const transport::BFloat16 *>(impl->allocation->w2);
  impl->session_identity = session.stable_identity_bytes();
  impl->execution = execution.token();
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state->device;
  if (hidden_size == forward_tuning::kMainHidden && intermediate_size == forward_tuning::kMainIntermediate) {
    impl->w1_map = make_bf16_tma_map(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        forward_tuning::kFc1TileN, "W1", forward_tuning::kTileK);
    impl->w1_interleaved_map = make_bf16_tma_map(
        impl->allocation->w1_interleaved, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        forward_tuning::kFc1TileN, "interleaved W1",
        forward_tuning::kTileK);
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        forward_tuning::kFc2TileN, "W2", forward_tuning::kTileK);
    impl->initialize_prefill_maps();
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
  const auto w1_bytes = checked_size(
      checked_mul(w1_elements, sizeof(transport::BFloat16), "W1 bytes"),
      "W1 bytes");
  internal::throw_cuda(
      "external-weight-publish",
      cudaMalloc(&allocation->w1_interleaved, w1_bytes),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaMalloc(interleaved W1)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  allocation->bytes = checked_add(
      checked_mul(2ULL, w1_bytes, "W1 plus interleaved W1 bytes"),
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
  interleave_forward_w1_kernel<<<4096, 256, 0, state_->backend->stream>>>(
      w1, static_cast<transport::BFloat16 *>(allocation->w1_interleaved),
      w1_elements, hidden_size, intermediate_size);
  internal::throw_cuda(
      "external-weight-publish", cudaGetLastError(),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts, "interleave W1");
  internal::throw_cuda(
      "external-weight-publish",
      internal::record_input_ready(allocation->ready, state_->backend->stream,
                                   allocation->ready_capture_id),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(weight ready)");
  auto impl = std::make_shared<ForwardWeightLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->w1 = w1;
  impl->w1_interleaved = static_cast<const transport::BFloat16 *>(
      impl->allocation->w1_interleaved);
  impl->w2 = w2;
  impl->session_identity = state_->backend->context.session().stable_identity_bytes();
  impl->execution = state_->execution;
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state_->backend->device;
  if (hidden_size == forward_tuning::kMainHidden && intermediate_size == forward_tuning::kMainIntermediate) {
    impl->w1_map = make_bf16_tma_map(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        forward_tuning::kFc1TileN, "W1", forward_tuning::kTileK);
    impl->w1_interleaved_map = make_bf16_tma_map(
        impl->allocation->w1_interleaved, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        forward_tuning::kFc1TileN, "interleaved W1",
        forward_tuning::kTileK);
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        forward_tuning::kFc2TileN, "W2", forward_tuning::kTileK);
    impl->initialize_prefill_maps();
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
  for (std::size_t index = 0; index < sampled_experts.size(); ++index) {
    const auto local_expert = sampled_experts[index];
    // With one or two owned experts, midpoint and endpoint repeat. Hash each
    // expert once so the distinctness check compares different weight tensors.
    if (index != 0 && local_expert == sampled_experts[index - 1])
      continue;
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
constexpr std::uint64_t kLocalP2pBootstrapSchema = 2;
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
  // Device ordinals are process-local: schedulers commonly expose only one
  // GPU to each actor, making every actor's ordinal zero. CUDA IPC can still
  // import peer allocations; identify the physical devices by UUID instead.
  cudaDeviceProp device_properties{};
  internal::throw_cuda(
      "local-p2p-create",
      cudaGetDeviceProperties(&device_properties, static_cast<int>(pool->device)),
      session.rank(),
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "cudaGetDeviceProperties(UUID)");

  std::lock_guard<std::mutex> mapping_lock(pool->mutex);
  auto backend = std::make_shared<LocalP2pBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  if (pool->peer_mappings) {
    if (pool->peer_mappings->nccl.get() != nccl.get())
      transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-create",
                      "IPC cache communicator", nccl->rank, "same communicator",
                      "mismatch");
    backend->mappings = pool->peer_mappings;
    backend->peer_slot_bases = backend->mappings->bases;
    return backend;
  }
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
  std::memcpy(local.data() + 40, device_properties.uuid.bytes, 8);
  put_u64_le(local, 48, slot_count);
  put_u64_le(local, 56, pool->layout.total);
  put_u64_le(local, 64, session.session().generation());
  const auto identity_hash =
      session_summary(session.session().stable_identity_bytes());
  put_u64_le(local, 72, identity_hash[0]);
  put_u64_le(local, 80, identity_hash[1]);
  std::memcpy(local.data() + 88, device_properties.uuid.bytes + 8, 8);
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

  const auto mapping_count =
      checked_size(checked_mul(slot_count, nccl->world, "IPC mapping count"),
                   "IPC mapping count");
  backend->peer_slot_bases.assign(mapping_count, nullptr);
  backend->owns_mapping.assign(mapping_count, 0);
  try {
    std::vector<std::array<std::byte, 16>> seen_devices;
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
      std::array<std::byte, 16> peer_device{};
      std::memcpy(peer_device.data(), record + 40, 8);
      std::memcpy(peer_device.data() + 8, record + 88, 8);
      if (std::find(seen_devices.begin(), seen_devices.end(), peer_device) !=
          seen_devices.end())
        transport_error(NotifyErrorCode::TransportProtocolFailure,
                        "local-p2p-bootstrap", "device UUID", session.rank(),
                        "unique physical GPU", "duplicate at peer rank " +
                            std::to_string(rank));
      seen_devices.push_back(peer_device);
      // The peer may be hidden by CUDA_VISIBLE_DEVICES. The IPC import below
      // resolves its physical allocation and validates/enables peer access.
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
  auto mappings = std::make_shared<PeerMappings>();
  mappings->device = pool->device;
  mappings->nccl = nccl;
  mappings->bases = backend->peer_slot_bases;
  mappings->owned = std::move(backend->owns_mapping);
  backend->mappings = pool->peer_mappings = std::move(mappings);
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
      rail.rank != static_cast<int>(node) ||
      lsa.rank != static_cast<int>(local_rank) || lsa.stride != 1 ||
      rail.stride != static_cast<int>(local_world))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=topology, rank=" +
            std::to_string(session.rank()) +
            ", expected=contiguous LSA ranks and matching rail ranks, "
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

  std::unique_lock<std::mutex> mapping_lock(pool->mutex);
  if (pool->peer_mappings) {
    if (pool->peer_mappings->nccl.get() != nccl.get())
      transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                      "IPC cache communicator", nccl->rank, "same communicator",
                      "mismatch");
    backend->mappings = pool->peer_mappings;
    backend->peer_slot_bases = backend->mappings->bases;
  } else {
    // Exchange every compute-arena IPC handle collectively, but open only the
    // ranks in this node. The full-world pointer table keeps global rank as the
    // stable index consumed by Notify fan-out descriptors.
    const auto slot_count = pool->slots.size();
    const auto handles_per_rank_bytes =
        checked_size(checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                                 "GIN IPC handles per rank"),
                     "GIN IPC handles per rank");
    const auto matrix_bytes =
        checked_size(checked_mul(handles_per_rank_bytes, nccl->world,
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
            cudaMemcpyAsync(at<void *>(pool->slots[slot_index].device,
                                       pool->layout.peer_bases),
                            backend->peer_slot_bases.data() +
                                slot_index * nccl->world,
                            nccl->world * sizeof(void *),
                            cudaMemcpyHostToDevice, exec->stream),
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
    auto mappings = std::make_shared<PeerMappings>();
    mappings->device = pool->device;
    mappings->nccl = nccl;
    mappings->bases = backend->peer_slot_bases;
    mappings->owned = std::move(backend->owns_mapping);
    backend->mappings = pool->peer_mappings = std::move(mappings);
  }
  mapping_lock.unlock();
  backend->resources = pool->gin_resources;
  auto &resources = *backend->resources;
  std::lock_guard<std::mutex> resource_lock(resources.mutex);
  if (resources.nccl && resources.nccl.get() != nccl.get())
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                    "communication resources", nccl->rank, "same communicator",
                    "mismatch");
  resources.nccl = nccl;
  backend->full_connections = true;
  constexpr std::uint32_t kGinContextCount = 8;
  int context_sms = 0;
  internal::throw_cuda("gin-create", cudaDeviceGetAttribute(
      &context_sms, cudaDevAttrMultiProcessorCount, pool->device), nccl->rank,
      nccl->world, session.session().identity(), session.session().generation(),
      ExchangeKind::ExpertCounts, "query GIN context participants");
  unsigned allocated_context_count = std::max((1U + communication::kReturnPorts) * kGinContextCount,
      (forward_tuning::kUnifiedWorkers || backward_tuning::kUnifiedWorkers)
          ? 1U + unsigned(context_sms) : 0U);
  if constexpr (WEFTMOE_GIN_CONTEXT_COUNT > 0)
    allocated_context_count = WEFTMOE_GIN_CONTEXT_COUNT;
  const bool static_inference_pool = forward_tuning::kUnifiedWorkers &&
      pool->config.max_intermediate_size && !pool->config.enable_backward_state &&
      std::getenv("WEFTMOE_DISABLE_STATIC_FORWARD_V4") == nullptr;
  if (static_inference_pool)
    allocated_context_count = pool->gin_layout.static_forward.contexts;
  backend->gin_context_count = kGinContextCount;
  backend->gin_allocated_context_count = allocated_context_count;
  if (!resources.dev_comm_created) {
    ncclDevCommRequirements_t requirements =
        NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    // Allocate a private pool for this devComm. Static CTA/channel ownership
    // is independent of this host allocation policy; when the backend returns
    // fewer contexts, device-side GPU sharing arbitrates concurrent issuers.
    requirements.ginContextCount = allocated_context_count;
    requirements.barrierCount = 1;
    requirements.ginExclusiveContexts = true;
    requirements.ginQueueDepth = 1024;
    // Legacy transport diagnostics use signals 0/1; production terminal status
    // uses signal 2. Ring readiness and credits use window-addressed signals.
    requirements.ginSignalCount = 3U;
    // Every wire slot has its own local-completion counter. Counters 0/1 are
    // reserved for bootstrap and terminal-status collectives.
    requirements.ginCounterCount = 2U + pool->gin_layout.rings.channels() *
                                            communication::kRingDepth *
                                            pool->slots.size();
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    internal::throw_nccl(
        "gin-create",
        dev_comm_create(nccl->comm, &requirements, &resources.dev_comm),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclDevCommCreate(activation GIN)");
    resources.dev_comm_created = true;
    if ((static_inference_pool &&
         resources.dev_comm.ginContextCount < int(allocated_context_count)) ||
        resources.dev_comm.ginContextCount < 1 ||
        (!static_inference_pool &&
         !(forward_tuning::kUnifiedWorkers && backward_tuning::kUnifiedWorkers) &&
         resources.dev_comm.ginContextCount < int((1U + communication::kReturnPorts) * kGinContextCount)))
      transport_error(NotifyErrorCode::BackendUnavailable, "gin-create",
                      "actual GIN contexts", nccl->rank, "enough contexts for static owners or legacy Return streams",
                      std::to_string(resources.dev_comm.ginContextCount));
    allocated_context_count = unsigned(resources.dev_comm.ginContextCount);
    backend->gin_allocated_context_count = allocated_context_count;
    try {
      resources.windows.reserve(pool->slots.size());
      for (auto &slot : pool->slots) {
        ncclWindow_t window = nullptr;
        if (slot.gin_window == nullptr || pool->gin_layout.total == 0)
          transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                          "registered window storage", nccl->rank,
                          "preallocated ncclMemAlloc window", "empty");
        internal::throw_nccl(
            "gin-create",
            window_register(nccl->comm, slot.gin_window, pool->gin_layout.total,
                            &window, NCCL_WIN_STRICT_ORDERING),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "ncclCommWindowRegister(activation pool slot)");
        resources.windows.push_back(window);
        auto ring_layout = pool->gin_layout.rings;
        ring_layout.counter_base += (&slot - pool->slots.data()) *
                                    ring_layout.channels() *
                                    communication::kRingDepth;
        communication::initialize<<<
            allocated_context_count, 1, 0,
            internal::checked_execution_state(execution)->stream>>>(
            resources.dev_comm, window, ring_layout, allocated_context_count);
        internal::throw_cuda(
            "gin-create", cudaGetLastError(), nccl->rank, nccl->world,
            session.session().identity(), session.session().generation(),
            ExchangeKind::ExpertCounts, "initialize GIN chunk signals");
      }
    } catch (...) {
      backend.reset();
      throw;
    }
  }
  backend->dev_comm = resources.dev_comm;
  backend->gin_allocated_context_count = unsigned(resources.dev_comm.ginContextCount);
  backend->windows = resources.windows;
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    pool->gin_backend = backend;
  }
  return backend;
#endif
}

namespace {
std::shared_ptr<ActivationPoolState>
make_activation_pool_state(const CollectiveSession &, const CudaExecution &,
                           ActivationTransportPoolConfig,
                           std::shared_ptr<GinResources> = {});
}

ActivationTransportResourcePool prepare_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config, const DeviceDispatchHandle &dispatch,
    const ActivationTransportResourcePool &existing) {
  // Explicit capacity permits capture preparation before Notify. Eager callers
  // supply the current Notify and reuse/grow the same pool as needed.
  if (dispatch) {
    if (dispatch.identity().world_size != session.world_size() ||
        ActivationTransportBridge::context(dispatch)
                .session()
                .stable_identity_bytes() != session.stable_identity_bytes())
      transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-prepare",
                      "dispatch", 0, "Notify for the current EP group",
                      "mismatched");
    config.max_tokens_per_rank = std::max<std::uint64_t>(
        config.max_tokens_per_rank, dispatch.capacity().tokens_per_rank);
    config.max_destination_rows_per_rank =
        std::max<std::uint64_t>(config.max_destination_rows_per_rank,
                                dispatch.capacity().destination_rows_per_rank);
  } else if (!config.max_destination_rows_per_rank) {
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-prepare",
                    "capacity", 0,
                    "Notify or explicit destination row capacity", "missing");
  }
  config.max_tokens_per_rank =
      std::max<std::uint64_t>(config.max_tokens_per_rank, 1);
  config.max_destination_rows_per_rank =
      std::max<std::uint64_t>(config.max_destination_rows_per_rank, 1);
  std::shared_ptr<GinResources> resources;
  if (existing.impl_ && existing.impl_->state) {
    const auto &state = *existing.impl_->state;
    const auto &old = state.config;
    const bool compatible =
        state.session_identity == session.stable_identity_bytes() &&
        state.device == execution.device_ordinal() &&
        old.max_hidden_size == config.max_hidden_size &&
        old.max_intermediate_size == config.max_intermediate_size &&
        old.max_top_k == config.max_top_k &&
        old.max_world_size == config.max_world_size &&
        old.max_nodes == config.max_nodes &&
        old.max_owned_experts == config.max_owned_experts &&
        old.channels == config.channels &&
        old.ring_slots_per_source_channel ==
            config.ring_slots_per_source_channel &&
        old.max_inflight_generations == config.max_inflight_generations &&
        old.enable_backward_state == config.enable_backward_state &&
        old.static_forward_window_tokens == config.static_forward_window_tokens &&
        old.enable_gin_rdma == config.enable_gin_rdma &&
        old.enable_socket_fallback == config.enable_socket_fallback;
    if (compatible) {
      if (old.max_tokens_per_rank >= config.max_tokens_per_rank &&
          old.max_destination_rows_per_rank >=
              config.max_destination_rows_per_rank)
        return existing;
      if (config.max_tokens_per_rank > old.max_tokens_per_rank)
        config.max_tokens_per_rank = std::max(
            config.max_tokens_per_rank,
            checked_add(old.max_tokens_per_rank, old.max_tokens_per_rank / 4,
                        "source arena growth"));
      else
        config.max_tokens_per_rank = old.max_tokens_per_rank;
      if (config.max_destination_rows_per_rank >
          old.max_destination_rows_per_rank)
        config.max_destination_rows_per_rank =
            std::max(config.max_destination_rows_per_rank,
                     checked_add(old.max_destination_rows_per_rank,
                                 old.max_destination_rows_per_rank / 4ULL,
                                 "compute arena growth"));
      else
        config.max_destination_rows_per_rank =
            old.max_destination_rows_per_rank;
      if (state.gin_layout.total == make_gin_window_layout(config).total)
        resources = state.gin_resources;
    }
  }
  auto impl = std::make_shared<ActivationTransportResourcePool::Impl>();
  impl->state = make_activation_pool_state(session, execution, config,
                                           std::move(resources));
  return ActivationTransportResourcePool(std::move(impl));
}

namespace {
std::shared_ptr<ActivationPoolState>
make_activation_pool_state(const CollectiveSession &session,
                           const CudaExecution &execution,
                           ActivationTransportPoolConfig config,
                           std::shared_ptr<GinResources> resources) {
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
#if WEFTMOE_DEVICE_TIMELINE
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
  const bool new_communication = !resources;
  if (config.enable_gin_rdma) {
    if (!resources) {
      resources = std::make_shared<GinResources>();
      resources->device = state->device;
      resources->bytes_per_slot = state->gin_layout.total;
      resources->allocations.resize(config.max_inflight_generations, nullptr);
      internal::throw_cuda("transport-pool-create",
                           cudaEventCreateWithFlags(&resources->last_use,
                                                    cudaEventDisableTiming),
                           0, session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaEventCreate(GIN resource reuse)");
    }
    state->gin_resources = resources;
  }
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
      if (config.enable_gin_rdma && new_communication) {
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
        resources->allocations[&slot - state->slots.data()] = slot.gin_window;
      } else if (resources) {
        slot.gin_window = resources->allocations[&slot - state->slots.data()];
      }
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
#if WEFTMOE_DEVICE_TIMELINE
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
      if (slot.gin_window && new_communication)
        internal::throw_cuda(
            "transport-pool-create",
            cudaMemsetAsync(slot.gin_window, 0, state->gin_layout.total,
                            internal::checked_execution_state(execution)->stream),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "one-time initialize GIN window slot storage");
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
      // The communication probe has no FC2. Initialize its fixture once,
      // outside steady-state wall timing, while retaining first-use cost.
      fill_transport_benchmark_output<<<
          4096, 256, 0, internal::checked_execution_state(execution)->stream>>>(
          at<transport::BFloat16>(slot.device,
                                  config.enable_backward_state
                                      ? state->layout.forward_training_output
                                      : state->layout.canonical_rows),
          config.max_destination_rows_per_rank * config.max_hidden_size);
      internal::throw_cuda("transport-benchmark-prefill", cudaGetLastError(), 0,
                           session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "one-time Return fixture initialization");
#endif
      if (config.max_intermediate_size != 0 &&
          config.max_hidden_size == forward_tuning::kMainHidden &&
          config.max_intermediate_size == forward_tuning::kMainIntermediate) {
        const auto activation_rows = state->layout.static_ring.depth
            ? std::uint64_t(state->layout.static_ring.depth) * forward_tuning::kTileM
            : config.max_destination_rows_per_rank;
        slot.forward_x_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, activation_rows,
            forward_tuning::kTileM, "canonical X", forward_tuning::kTileK);
        slot.forward_s_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size, activation_rows,
            forward_tuning::kTileM, "gated SwiGLU U",
            forward_tuning::kTileK);
        slot.forward_output_store_map = make_bf16_tma_store_map(
            at<void>(slot.device, state->layout.static_ring.depth
                                     ? state->layout.static_ring.output
                                     : state->layout.canonical_rows),
            config.max_hidden_size, activation_rows,
            "Forward canonical output");
        if (config.enable_backward_state)
          slot.forward_training_output_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Forward training output");
        if (config.enable_backward_state) {
          slot.backward_dr_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              backward_tuning::kActivationTileM, "Backward compact dR",
              backward_tuning::kTileK);
          slot.backward_dab_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.backward_dab_rows),
              2ULL * config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              backward_tuning::kActivationTileM, "Backward padded dAB",
              backward_tuning::kTileK);
          slot.backward_dr_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Backward compact dW2 dR MN-major");
          slot.backward_u_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.backward_u),
              config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              "Backward dW2 U MN-major");
          slot.backward_dab_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.backward_dab_rows),
              2ULL * config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              "Backward dW1 dAB MN-major");

          slot.backward_u_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.backward_u),
              config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows, "Backward dU output",
              0ULL, backward_tuning::kActivationTileM);
          slot.backward_dx_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.backward_dx_rows),
              config.max_hidden_size,
              state->layout.backward_padded_capacity_rows, "Backward dX output",
              0ULL, backward_tuning::kActivationTileM);

          slot.backward_maps_valid = true;
        }
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
                        sizeof(TransportDeviceStatus) + sizeof(std::uint32_t),
                        cudaHostAllocPortable | cudaHostAllocMapped),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(forward status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      slot.host_forward_done =
          reinterpret_cast<std::uint32_t *>(slot.host_forward_status + 1);
      *slot.host_forward_done = 0;
      void *mapped_forward_status = nullptr;
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostGetDevicePointer(&mapped_forward_status,
                                   slot.host_forward_status, 0),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostGetDevicePointer(Forward completion)");
      slot.device_forward_done = at<std::uint32_t>(
          mapped_forward_status, sizeof(TransportDeviceStatus));
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
      create_event(&slot.forward_compute_done,
                   "cudaEventCreate(Forward compute done)");
      if (config.enable_backward_state) {
        create_timing_event(&slot.backward_start,
                            "cudaEventCreate(Backward start)");
        create_timing_event(&slot.backward_compute_start,
                            "cudaEventCreate(Backward compute start)");
        create_timing_event(&slot.backward_compute_done,
                            "cudaEventCreate(Backward compute done)");
        create_timing_event(&slot.backward_transport_done,
                            "cudaEventCreate(Backward transport done)");
        create_timing_event(&slot.backward_reduce_done,
                            "cudaEventCreate(Backward reduce done)");
        create_timing_event(&slot.backward_late_compute_done,
                            "cudaEventCreate(Backward late compute done)");
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
  return state;
}
} // namespace

ActivationTransportResourcePool
make_activation_transport_pool(const CollectiveSession &session,
                               const CudaExecution &execution,
                               ActivationTransportPoolConfig config) {
  auto state = make_activation_pool_state(session, execution, config);
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
#if WEFTMOE_DEVICE_TIMELINE
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
      checked_add(2ULL * sizeof(TransportDeviceStatus) + sizeof(std::uint32_t),
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
  auto allocation = std::make_shared<InputTensorAllocation>();
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
      internal::record_input_ready(allocation->ready, exec->stream,
                                   allocation->ready_capture_id),
      context.calling_rank(), identity.world_size(), identity.identity(),
      identity.generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(output gradient ready)");
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
  if (state->config.max_intermediate_size != 0) {
    auto data = std::make_shared<internal::CudaAllocation>();
    data->device = execution.device_ordinal();
    data->device_bytes =
        checked_size(checked_mul(count, 2ULL, "input bytes"), "input bytes");
    auto exec = internal::checked_execution_state(execution);
    internal::throw_cuda(
        "activation-test-upload",
        cudaMalloc(&data->device_ptr,
                   std::max<std::size_t>(data->device_bytes, 128)),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "allocate test input");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    if (data->device_bytes)
      internal::throw_cuda(
          "activation-test-upload",
          cudaMemcpyAsync(data->device_ptr, host.data(), data->device_bytes,
                          cudaMemcpyHostToDevice, exec->stream),
          context.calling_rank(), context.session().world_size(),
          context.session().identity(), context.session().generation(),
          ExchangeKind::ExpertCounts, "upload test input");
    auto producer = make_external_cuda_producer(
        context, data->device, exec->stream, exec, "test input");
    return producer.publish_activations(
        static_cast<const transport::BFloat16 *>(data->device_ptr), count,
        num_tokens, hidden, data);
  }
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
      "activation-test-upload",
      internal::record_input_ready(slot.producer_ready, exec->stream,
                                   slot.producer_capture_id),
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
      (!source.allocation &&
       !source.producer_execution.same_instance(execution.token())) ||
      source.session_identity != context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      (!source.allocation && (!source.slot || !source.pool)))
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
    const transport::BFloat16 *device_data, std::uint64_t capacity_elements,
    std::uint64_t num_tokens, std::uint64_t hidden_size,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !allocation_owner || hidden_size == 0)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-activation-publish", "producer/owner/hidden", 0,
                    "valid", "empty");
  const auto &producer = *state_->backend;
  const auto elements = checked_mul(num_tokens, hidden_size,
                                    "external activation elements");
  validate_external_cuda_range(device_data, elements, capacity_elements,
                               sizeof(transport::BFloat16), producer.device,
                               "activations");
  auto allocation = std::make_shared<InputTensorAllocation>();
  allocation->device = producer.device;
  allocation->data = const_cast<transport::BFloat16 *>(device_data);
  allocation->owns_data = false;
  allocation->external_owner = std::move(allocation_owner);
  allocation->external_producer = state_->backend;
  internal::throw_cuda(
      "external-activation-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      producer.context.calling_rank(), producer.context.session().world_size(),
      producer.context.session().identity(),
      producer.context.session().generation(), ExchangeKind::ExpertCounts,
      "create input readiness event");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "external-activation-publish",
      internal::record_input_ready(allocation->ready, producer.stream,
                                   allocation->ready_capture_id),
      producer.context.calling_rank(), producer.context.session().world_size(),
      producer.context.session().identity(),
      producer.context.session().generation(), ExchangeKind::ExpertCounts,
      "record input readiness");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->pointer = device_data;
  impl->session_identity = producer.context.session().stable_identity_bytes();
  impl->producer_execution = state_->execution;
  impl->generation = producer.context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden_size;
  impl->device = producer.device;
  impl->ready_recorded = true;
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
  auto allocation = std::make_shared<InputTensorAllocation>();
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
      internal::record_input_ready(allocation->ready, state_->backend->stream,
                                   allocation->ready_capture_id),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(output gradient ready)");
  auto impl = std::make_shared<BFloat16OutputGradientLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->pointer = device_data;
  impl->session_identity =
      state_->backend->context.session().stable_identity_bytes();
  impl->producer_execution = state_->execution;
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
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-create", "pool",
                    0, "valid", "empty");
  impl_ = std::make_shared<Impl>();
  if (use_gin)
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_gin_backend(session, execution, pool.impl_->state));
  else
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_local_p2p_backend(session, execution, pool.impl_->state));
}

GinRdmaForwardMegaKernel::GinRdmaForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pForwardMegaKernel>(
          new LocalCudaP2pForwardMegaKernel(session, execution, pool, true))) {}

DeviceForwardHandle GinRdmaForwardMegaKernel::enqueue(
    const DeviceDispatchHandle &dispatch, const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->enqueue(dispatch, activations, weights, config, options);
}

DeviceForwardHandle GinRdmaForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(dispatch, activations, weights, config, options);
}

DeviceActivationTransportHandle
DeviceActivationTransport::run_device_simulation_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config) {
  if (!impl_ || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-run",
                    "transport/dispatch/activation", 0, "valid", "empty");
  const auto &id = dispatch.identity();
  auto state = impl_->pool.impl_->state;
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  const auto &activation = *activations.impl_;
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens || activation.hidden != config.hidden_size ||
      activation.device != impl_->execution.device_ordinal() ||
      activation.session_identity != impl_->session.stable_identity_bytes() ||
      !activation.producer_execution.same_instance(impl_->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size ||
      id.num_nodes > state->config.max_nodes)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "transport-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "transport-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");
  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(impl_->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(impl_->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  const auto links = checked_mul(id.num_nodes, config.channels, "actual links");
  const auto ring_slots = checked_mul(
      checked_mul(id.num_nodes, config.channels, "actual ring links"),
      config.ring_slots_per_source_channel, "actual ring slots");
  const auto route_elements = checked_mul(view.sizes.route_count,
                                          config.hidden_size, "route elements");
  const auto combined_elements = checked_mul(id.num_tokens, config.hidden_size,
                                             "combined elements");
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0,
                        checked_size(checked_mul(count, width, name), name),
                        exec->stream),
        id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
  };
  internal::throw_cuda(
      "transport-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  zero_async(layout.ring_headers, ring_slots,
             sizeof(transport::TransportSlotHeader), "ring headers");
  zero_async(layout.ring_ready_sequence, ring_slots, sizeof(std::uint64_t),
             "ring sequences");
  zero_async(layout.link_next_sequence, links, sizeof(std::uint64_t),
             "link sequences");
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits), links,
                      config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "transport-run", cudaGetLastError(), id.calling_rank, id.world_size,
      impl_->session.identity(), id.generation, ExchangeKind::ExpertCounts,
      "initialize transport credits kernel");
  zero_async(layout.packet_claim, view.sizes.packet_count,
             sizeof(std::uint32_t), "packet claims");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "canonical elements"),
             sizeof(transport::BFloat16), "canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "combined");
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "status");
  internal::throw_cuda("transport-run", cudaEventRecord(slot.start, exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(start)");
  int sms = 0;
  internal::throw_cuda(
      "transport-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  TransportKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, impl_->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
  args.ring_headers = at<transport::TransportSlotHeader>(slot.device,
                                                          layout.ring_headers);
  args.ring_payload = at<transport::BFloat16>(slot.device, layout.ring_payload);
  args.ring_ready = at<unsigned long long>(slot.device,
                                            layout.ring_ready_sequence);
  args.link_next = at<unsigned long long>(slot.device,
                                           layout.link_next_sequence);
  args.link_credits = at<unsigned int>(slot.device, layout.link_credits);
  args.packet_claim = at<unsigned int>(slot.device, layout.packet_claim);
  args.canonical_rows = at<transport::BFloat16>(slot.device,
                                                 layout.canonical_rows);
  args.contribution = at<transport::BFloat16>(slot.device, layout.contribution);
  args.combine_return = at<transport::BFloat16>(slot.device,
                                                 layout.combine_return);
  args.combined = at<transport::BFloat16>(slot.device, layout.combined);
  args.status = at<TransportDeviceStatus>(slot.device, layout.status);
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  activation_transport_persistent_kernel<<<std::max(sms, 1), 128, 0,
                                           exec->stream>>>(args);
  internal::throw_cuda("transport-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, impl_->session.identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "activation transport persistent kernel launch");
  internal::throw_cuda(
      "transport-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(dispatch complete)");
  internal::throw_cuda(
      "transport-run",
      cudaMemcpyAsync(slot.host_status, args.status,
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "transport-run",
      internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda("transport-run", cudaEventSynchronize(slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventSynchronize(transport complete)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "transport-kernel", "descriptor", id.calling_rank,
                    "valid packet/fanout", "device validation error packet=" +
                        std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda("transport-timing",
                       cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventElapsedTime(total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  // The concrete backend object, never a caller-provided enum, owns this
  // label. This path is deliberately and permanently simulation-only.
  result->audit.backend =
      transport::ActivationTransportBackend::DeviceSimulation;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    result->timings.combine_reduce_us =
        static_cast<double>(complete_clock - dispatch_clock) / 1000.0;
  }
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle LocalCudaP2pActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=one node and exact local NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "local-p2p-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "local-p2p-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  [[maybe_unused]] const auto route_elements =
      checked_mul(view.sizes.route_count, config.hidden_size,
                  "P2P route elements");
  [[maybe_unused]] const auto combined_elements =
      checked_mul(id.num_tokens, config.hidden_size,
                  "P2P combined elements");
  [[maybe_unused]] const auto rank_links =
      checked_mul(id.world_size, config.channels, "P2P rank links");
  [[maybe_unused]] const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "P2P ingress slots");
  [[maybe_unused]] const auto node_links =
      static_cast<std::uint64_t>(config.channels);
  [[maybe_unused]] const auto node_ring_slots = checked_mul(
      node_links, config.ring_slots_per_source_channel, "P2P proxy slots");
  std::uint64_t generation_memset_calls = 0;
  std::uint64_t generation_memset_bytes = 0;
  std::uint64_t kernel_launches = 0;
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    const auto bytes = checked_size(checked_mul(count, width, name), name);
    internal::throw_cuda(
        "local-p2p-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0, bytes, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
    ++generation_memset_calls;
    generation_memset_bytes += bytes;
  };
  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested()
                       ? id.generation
                       : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "local-p2p-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  // The integrated timing starts after the router-owned input is ready and
  // includes every generation-specific transport initialization below.
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.start, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P start)");
#if WEFTMOE_DEBUG_VALIDATE
  zero_async(layout.ring_headers, node_ring_slots,
             sizeof(transport::TransportSlotHeader), "P2P proxy headers");
  zero_async(layout.ring_payload,
             checked_mul(node_ring_slots, config.hidden_size,
                         "P2P proxy payload"),
             sizeof(transport::BFloat16), "P2P proxy payload");
  zero_async(layout.ring_ready_sequence, node_ring_slots,
             sizeof(std::uint64_t), "P2P proxy ready");
  zero_async(layout.link_next_sequence, node_links, sizeof(std::uint64_t),
             "P2P proxy sequence");
  zero_async(layout.ingress_headers, ingress_slots,
             sizeof(LocalP2pRouteHeader), "P2P ingress headers");
  zero_async(layout.ingress_ready_sequence, ingress_slots,
             sizeof(std::uint64_t), "P2P ingress ready");
  zero_async(layout.ingress_next_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress next");
  zero_async(layout.ingress_published_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress published");
  zero_async(layout.outgoing_route_sequence, rank_links,
             sizeof(std::uint64_t), "P2P outgoing sequence");
  zero_async(layout.direct_route_headers,
             state->config.max_destination_rows_per_rank,
             sizeof(LocalP2pRouteHeader), "P2P direct route headers");
  zero_async(layout.return_ready_epoch, view.sizes.route_count,
             sizeof(std::uint64_t), "P2P return ready");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "P2P canonical rows"),
             sizeof(transport::BFloat16), "P2P canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "P2P contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "P2P combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "P2P combined");
#else
  // The trusted single-node bulk path overwrites every canonical row,
  // contribution and output element that it later reads. Epoch-valued peer
  // signals make all arena and signal clears unnecessary. Clearing a peer
  // signal here would race a faster rank's release store.
#endif
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "P2P status");
#if WEFTMOE_DEBUG_VALIDATE
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((node_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits),
                      node_links, config.ring_slots_per_source_channel);
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((rank_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.ingress_credits),
                      rank_links, config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P credits");

  // A fixed-width device AllGather is a setup barrier only. It exchanges no
  // activation and ensures every peer ring/credit/sequence region is fully
  // initialized before any persistent worker publishes to it.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, activation.slot->epoch);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
  validate_p2p_barrier_kernel<<<1, 256, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_receive),
      id.world_size, activation.slot->index, activation.slot->epoch,
      at<TransportDeviceStatus>(slot.device, layout.status));
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "validate P2P setup barrier");
#else
  // This small collective is a real per-generation dependency: peer writes
  // must not begin until every process has reached the transport phase. It
  // carries only 16 bytes/rank and replaces the old multi-gigabyte clears.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, id.generation);
  ++kernel_launches;
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
#endif
  int sms = 0;
  internal::throw_cuda(
      "local-p2p-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  if (sms <= static_cast<int>(id.world_size))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=resident_workers, rank=" +
            std::to_string(id.calling_rank) + ", expected>world=" +
            std::to_string(id.world_size) + ", actual_sms=" +
            std::to_string(sms));
  LocalP2pKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
#if WEFTMOE_DEBUG_VALIDATE
  args.epoch = activation.slot->epoch;
#else
  // Every rank invokes the trusted team in the same generation order. A
  // monotonically increasing per-slot epoch supports both new generations and
  // repeated same-generation correctness runs without clearing peer signals.
  if (slot.transport_epoch == std::numeric_limits<std::uint64_t>::max())
    transport_error(NotifyErrorCode::ArithmeticOverflow, "local-p2p-run",
                    "transport_epoch", id.calling_rank, "<UINT64_MAX",
                    "UINT64_MAX");
  args.epoch = ++slot.transport_epoch;
#endif
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.local_base = slot.device;
  args.ring_headers_offset = layout.ring_headers;
  args.ring_payload_offset = layout.ring_payload;
  args.ring_ready_offset = layout.ring_ready_sequence;
  args.link_next_offset = layout.link_next_sequence;
  args.link_credits_offset = layout.link_credits;
  args.ingress_headers_offset = layout.ingress_headers;
  args.ingress_ready_offset = layout.ingress_ready_sequence;
  args.ingress_next_offset = layout.ingress_next_sequence;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.ingress_credits_offset = layout.ingress_credits;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.direct_headers_offset = layout.direct_route_headers;
  args.return_ready_offset = layout.return_ready_epoch;
  args.canonical_rows_offset = layout.canonical_rows;
  args.contribution_offset = layout.contribution;
  args.combine_return_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  internal::throw_cuda(
      "local-p2p-run",
      cudaEventRecord(slot.dispatch_complete, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(P2P generation setup complete)");
#if WEFTMOE_DEBUG_VALIDATE
  local_p2p_persistent_kernel<<<sms, 128, 0, exec->stream>>>(args);
  ++kernel_launches;
#else
  local_p2p_bulk_dispatch_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(
          view.sizes.route_count, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_return_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(
          view.sizes.incoming_row_count, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_return_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_return_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_reduce_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(id.num_tokens, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_finish_kernel<<<1, 1, 0, exec->stream>>>(args);
  ++kernel_launches;
#endif
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "local P2P persistent kernel launch");
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H P2P transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "local-p2p-run",
      internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(P2P complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
  while (true) {
    const auto query = cudaEventQuery(slot.complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda(
          "local-p2p-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(P2P complete)");
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >=
            options.deadline_after;
    if (!cancelled && (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, id.generation,
                       __ATOMIC_RELEASE);
      internal::throw_cuda(
          "local-p2p-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "H2D publish cancel epoch");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                sizeof(std::uint64_t));
      internal::throw_cuda(
          "local-p2p-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaStreamSynchronize(cancel)");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  // Debug/cancellation paths retain the explicit team drain. In the trusted
  // release path every source waits for a generation-valued return signal
  // from every destination before producing output. The next Notify control
  // collective is the natural inter-generation barrier.
#if WEFTMOE_DEBUG_VALIDATE
  internal::throw_nccl(
      "local-p2p-drain",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P terminal drain)");
  internal::throw_cuda(
      "local-p2p-drain", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(P2P terminal drain)");
#else
  if (cancelled) {
    internal::throw_nccl(
        "local-p2p-drain",
        ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                      at<void>(slot.device, layout.p2p_barrier_receive), 2,
                      ncclUint64, backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "ncclAllGather(P2P cancellation drain)");
    internal::throw_cuda(
        "local-p2p-drain", cudaStreamSynchronize(exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(P2P cancellation drain)");
  }
#endif
  consumer.finish();
  if (cancelled || slot.host_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=local-p2p-run, field=cancel_epoch, rank=" +
            std::to_string(id.calling_rank) + ", expected=completion, actual=" +
            (options.cancellation.cancellation_requested() ? "cancelled"
                                                           : "deadline") +
            ", packets=" + std::to_string(slot.host_status->packets_done) +
            ", incoming=" + std::to_string(slot.host_status->incoming_done) +
            ", combine=" + std::to_string(slot.host_status->combine_done));
  if (slot.host_status->error != 0)
    transport_error(
        NotifyErrorCode::TransportProtocolFailure, "local-p2p-kernel",
        "ring/sequence/descriptor", id.calling_rank, "valid",
        "device error field=" + std::to_string(slot.host_status->error_field) +
            ", packet=" + std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(P2P total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->local_p2p_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows =
      state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::LocalCudaP2p;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.peak_ring_slots = slot.host_status->peak_ring_slots;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.derived_route_input_bytes =
      view.sizes.route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes =
      view.sizes.route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.generation_memset_calls = generation_memset_calls;
  result->audit.generation_memset_bytes = generation_memset_bytes;
  result->audit.kernel_launches = kernel_launches;
  result->audit.dispatch_kernel_blocks = view.sizes.route_count;
  result->audit.return_kernel_blocks = view.sizes.incoming_row_count;
  result->audit.reduce_kernel_blocks = id.num_tokens;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  float setup_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&setup_ms, slot.start, slot.dispatch_complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(P2P generation setup)");
  result->timings.generation_setup_us =
      static_cast<double>(setup_ms) * 1000.0;
  result->timings.dispatch_put_us =
      static_cast<double>(slot.host_status->dispatch_put_ticks) / 1000.0;
  result->timings.relay_us =
      static_cast<double>(slot.host_status->relay_ticks) / 1000.0;
  result->timings.synthetic_return_us =
      static_cast<double>(slot.host_status->synthetic_ticks) / 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    const auto post_dispatch = complete_clock - dispatch_clock;
    const auto relay_return = slot.host_status->synthetic_ticks;
    result->timings.combine_reduce_us = static_cast<double>(
        post_dispatch >= relay_return ? post_dispatch - relay_return : 0ULL) /
        1000.0;
  }
  const auto kernel_us = complete_clock >= start_clock
                             ? static_cast<double>(complete_clock - start_clock) /
                                   1000.0
                             : 0.0;
  const auto accounted = result->timings.generation_setup_us + kernel_us;
  result->timings.terminal_completion_us =
      std::max(0.0, result->timings.total_us - accounted);
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle
NcclSocketFallbackActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "socket-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  if (id.num_nodes < 2 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "topology", id.calling_rank,
                    "multi-node exact NCCL world/rank", "mismatch");
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "socket-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (!state->config.enable_socket_fallback ||
      config.hidden_size > state->config.max_hidden_size ||
      id.world_size > state->config.max_world_size ||
      id.top_k > state->config.max_top_k)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "socket-run",
                    "pool geometry", id.calling_rank, "within capacity",
                    "exceeded/disabled");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=socket-run, field=cancellation, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  auto *local_counts = at<unsigned long long>(slot.device,
                                               layout.terminal_epoch);
  auto *count_matrix = at<unsigned long long>(slot.device,
                                               layout.socket_count_matrix);
  auto *prefix = at<unsigned long long>(slot.device,
                                        layout.p2p_barrier_receive);
  auto *route_ids = at<unsigned int>(slot.device, layout.packet_claim);
  auto *send_headers = at<SocketRouteHeader>(
      slot.device, layout.socket_header_send);
  auto *receive_headers = at<SocketRouteHeader>(
      slot.device, layout.socket_header_receive);
  auto *send_payload = at<transport::BFloat16>(slot.device,
                                                layout.contribution);
  auto *receive_payload = at<transport::BFloat16>(
      slot.device, layout.socket_payload_receive);
  auto *canonical = at<transport::BFloat16>(slot.device,
                                             layout.canonical_rows);
  auto *by_route = at<transport::BFloat16>(slot.device,
                                            layout.combine_return);
  auto *combined = at<transport::BFloat16>(slot.device, layout.combined);
  auto *status = at<TransportDeviceStatus>(slot.device, layout.status);
  const auto world = id.world_size;
  const auto route_count = view.sizes.route_count;

  internal::throw_cuda(
      "socket-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(Socket start)");
  internal::throw_cuda(
      "socket-run", cudaMemsetAsync(local_counts, 0,
                                     world * sizeof(std::uint64_t),
                                     exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(Socket counts)");
  internal::throw_cuda(
      "socket-run", cudaMemsetAsync(status, 0, sizeof(*status), exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(Socket status)");
  if (route_count != 0) {
    socket_route_count_kernel<<<
        static_cast<unsigned int>((route_count + 255ULL) / 256ULL), 256, 0,
        exec->stream>>>(view, local_counts);
  }
  socket_route_prefix_kernel<<<1, 1, 0, exec->stream>>>(local_counts, prefix,
                                                        world);
  if (route_count != 0) {
    socket_stable_route_ids_kernel<<<world, 256, 0, exec->stream>>>(
        view, prefix, route_ids);
    socket_pack_routes_kernel<<<static_cast<unsigned int>(route_count), 256, 0,
                                exec->stream>>>(
        view, activation.pointer,
        ActivationTransportBridge::gate_pointer(dispatch), config.hidden_size,
        activation.slot->epoch, route_ids, send_headers, send_payload);
  }
  internal::throw_cuda(
      "socket-run", cudaGetLastError(), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "Socket pack kernels");
  internal::throw_nccl(
      "socket-run",
      ncclAllGather(local_counts, count_matrix, world, ncclUint64,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "ncclAllGather(Socket route counts)");
  const auto count_bytes = checked_size(
      checked_mul(checked_mul(world, world, "Socket actual count matrix"),
                  sizeof(std::uint64_t), "Socket actual count bytes"),
      "Socket actual count bytes");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_socket_counts, count_matrix, count_bytes,
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "D2H Socket route counts");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            count_bytes);
  internal::throw_cuda(
      "socket-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(Socket counts)");

  std::vector<std::uint64_t> send_prefix(static_cast<std::size_t>(world) + 1U,
                                         0);
  std::vector<std::uint64_t> receive_prefix(
      static_cast<std::size_t>(world) + 1U, 0);
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    send_prefix[peer + 1U] =
        send_prefix[peer] +
        slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                    world +
                                peer];
    receive_prefix[peer + 1U] =
        receive_prefix[peer] +
        slot.host_socket_counts[static_cast<std::size_t>(peer) * world +
                                id.calling_rank];
  }
  const auto incoming = receive_prefix[world];
  if (send_prefix[world] != route_count ||
      incoming > state->config.max_destination_rows_per_rank)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "socket-run",
                    "route totals", id.calling_rank,
                    "send=route_count and incoming<=capacity",
                    "send=" + std::to_string(send_prefix[world]) +
                        ", incoming=" + std::to_string(incoming));

  const auto enqueue_dispatch = [&] {
    internal::throw_nccl("socket-dispatch", ncclGroupStart(), id.calling_rank,
                         world, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "ncclGroupStart(Socket dispatch)");
    for (std::uint32_t peer = 0; peer < world; ++peer) {
      const auto send_rows = send_prefix[peer + 1U] - send_prefix[peer];
      const auto receive_rows =
          receive_prefix[peer + 1U] - receive_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != receive_rows)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "socket-dispatch", "self rows", id.calling_rank,
                          std::to_string(send_rows),
                          std::to_string(receive_rows));
        if (send_rows != 0) {
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(receive_headers + receive_prefix[peer],
                              send_headers + send_prefix[peer],
                              checked_size(send_rows * sizeof(*send_headers),
                                           "Socket self header bytes"),
                              cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket self headers");
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(
                  receive_payload + receive_prefix[peer] * config.hidden_size,
                  send_payload + send_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket self payload bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket self payload");
        }
        continue;
      }
      if (send_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_headers + send_prefix[peer],
                     send_rows * sizeof(*send_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_payload + send_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket payload)");
      }
      if (receive_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_headers + receive_prefix[peer],
                     receive_rows * sizeof(*receive_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_payload +
                         receive_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket payload)");
      }
    }
    internal::throw_nccl("socket-dispatch", ncclGroupEnd(), id.calling_rank,
                         world, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "ncclGroupEnd(Socket dispatch)");
  };
  enqueue_dispatch();
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket dispatch complete)");
  if (incoming != 0)
    socket_relay_transport_kernel<<<static_cast<unsigned int>(incoming), 256,
                                     0, exec->stream>>>(
        receive_headers, receive_payload, canonical,
        at<float>(slot.device, layout.canonical_gates), incoming,
        config.hidden_size, id.generation, id.calling_rank,
        state->config.max_destination_rows_per_rank,
        config.expert_transform ==
            transport::ActivationExpertTransform::Identity,
        status);

  internal::throw_nccl("socket-return", ncclGroupStart(), id.calling_rank,
                       world, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "ncclGroupStart(Socket return)");
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    const auto send_rows = receive_prefix[peer + 1U] - receive_prefix[peer];
    const auto receive_rows = send_prefix[peer + 1U] - send_prefix[peer];
    if (peer == id.calling_rank) {
      if (send_rows != 0)
        internal::throw_cuda(
            "socket-return",
            cudaMemcpyAsync(
                send_payload + send_prefix[peer] * config.hidden_size,
                receive_payload + receive_prefix[peer] * config.hidden_size,
                checked_size(send_rows * config.hidden_size *
                                 sizeof(transport::BFloat16),
                             "Socket self return bytes"),
                cudaMemcpyDeviceToDevice, exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "D2D Socket self return");
      continue;
    }
    if (send_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclSend(receive_payload +
                       receive_prefix[peer] * config.hidden_size,
                   send_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "ncclSend(Socket return)");
    if (receive_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                   receive_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "ncclRecv(Socket return)");
  }
  internal::throw_nccl("socket-return", ncclGroupEnd(), id.calling_rank,
                       world, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "ncclGroupEnd(Socket return)");
  if (route_count != 0)
    socket_scatter_returns_kernel<<<static_cast<unsigned int>(route_count),
                                    256, 0, exec->stream>>>(
        send_headers, send_payload, by_route, route_count,
        config.hidden_size);
  if (id.num_tokens != 0)
    socket_gate_reduce_kernel<<<static_cast<unsigned int>(id.num_tokens), 256,
                                0, exec->stream>>>(
        view, ActivationTransportBridge::gate_pointer(dispatch), by_route,
        combined, config.hidden_size);
  socket_finalize_status_kernel<<<1, 1, 0, exec->stream>>>(
      status, route_count, incoming, id.num_tokens);
  internal::throw_cuda(
      "socket-run", cudaGetLastError(), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "Socket relay/return/reduce kernels");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_status, status, sizeof(*status),
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "D2H Socket status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(*status));
  internal::throw_cuda(
      "socket-run",
      internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "socket-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(Socket complete)");
  consumer.finish();
  const auto production_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Forward, id.generation,
      id.generation, id.calling_rank, id.world_size, *slot.host_status,
      slot.complete);
  if (production_agreement.has_error) {
    failure_guard.committed = true;
    throw_production_status_error(production_agreement);
  }

  float total_ms = 0.0F;
  float dispatch_ms = 0.0F;
  internal::throw_cuda(
      "socket-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Socket total)");
  internal::throw_cuda(
      "socket-timing",
      cudaEventElapsedTime(&dispatch_ms, slot.start, slot.dispatch_complete),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Socket dispatch)");
  const auto gpus_per_node = world / id.num_nodes;
  const auto source_node = id.calling_rank / gpus_per_node;
  std::uint64_t remote_rows = 0;
  for (std::uint32_t peer = 0; peer < world; ++peer)
    if (peer / gpus_per_node != source_node)
      remote_rows += send_prefix[peer + 1U] - send_prefix[peer];
  const auto remote_bytes = checked_mul(
      checked_mul(remote_rows, config.hidden_size, "Socket remote elements"),
      sizeof(transport::BFloat16), "Socket remote bytes");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->socket_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::NcclSocketFallback;
  result->audit.packet_count = route_count;
  result->audit.fanout_count = incoming;
  result->audit.raw_cross_node_payload_bytes = remote_bytes;
  result->audit.deduplicated_cross_node_payload_bytes = remote_bytes;
  result->audit.actual_dispatch_peer_bytes = remote_bytes;
  result->audit.actual_return_peer_bytes = remote_bytes;
  result->audit.derived_route_input_bytes =
      route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes = route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.kernel_launches = 7;
  result->audit.dispatch_kernel_blocks = route_count;
  result->audit.return_kernel_blocks = incoming;
  result->audit.reduce_kernel_blocks = id.num_tokens;
  // Only fixed-width control/count/status metadata is staged. No activation
  // payload crosses the host boundary.
  result->audit.activation_d2h_bytes = 0;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.dispatch_relay_us =
      static_cast<double>(dispatch_ms) * 1000.0;
  result->timings.combine_reduce_us =
      result->timings.total_us - result->timings.dispatch_relay_us;
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle GinRdmaActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
#if !__has_include(<nccl_device.h>)
  (void)dispatch;
  (void)activations;
  (void)config;
  (void)options;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-run, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  const auto &activation = *activations.impl_;
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  if (id.num_nodes != backend->num_nodes ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=runtime GIN team topology and exact rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (!state->config.enable_gin_rdma ||
      config.hidden_size > state->config.max_hidden_size ||
      id.world_size > state->config.max_world_size ||
      id.top_k > state->config.max_top_k ||
      activation.slot->index >= backend->windows.size())
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "pool geometry/window", id.calling_rank,
                    "within registered GIN pool capacity",
                    "exceeded/disabled");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=gin-run, field=cancellation, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  if (dispatch.received_row_capacity() >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count > checked_mul(state->config.max_tokens_per_rank,
                                           state->config.max_top_k,
                                           "GIN route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");
  SlotFailureGuard failure_guard{activation.slot, false};
  const auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  internal::throw_cuda(
      "gin-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "gin-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(GIN start)");
  internal::throw_cuda(
      "gin-run", cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                                  sizeof(TransportDeviceStatus), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN status)");
  internal::throw_cuda(
      "gin-run",
      cudaMemsetAsync(at<void>(slot.gin_window,
                               state->gin_layout.barrier), 0,
                      sizeof(std::uint64_t), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN reset barrier)");
  gin_signal_reset_kernel<<<backend->gin_allocated_context_count, 1, 0,
                            exec->stream>>>(
      backend->dev_comm, backend->gin_allocated_context_count);
  internal::throw_cuda(
      "gin-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "GIN reset kernel launch");
  auto *barrier = at<std::uint64_t>(slot.gin_window,
                                    state->gin_layout.barrier);
  internal::throw_nccl(
      "gin-run",
      ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllReduce(after GIN reset)");

  GinTransportSliceArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.epoch = activation.slot->epoch;
  args.dev_comm = backend->dev_comm;
  args.window = backend->windows[activation.slot->index];
  args.compute_base = slot.device;
  args.window_base = slot.gin_window;
  args.send_header_offset = state->gin_layout.send_packet_headers;
  args.receive_header_offset = state->gin_layout.receive_packet_headers;
  args.send_fanout_offset = state->gin_layout.send_fanout;
  args.receive_fanout_offset = state->gin_layout.receive_fanout;
  args.send_payload_offset = state->gin_layout.send_payload;
  args.receive_payload_offset = state->gin_layout.receive_payload;
  args.canonical_offset = layout.canonical_rows;
  args.return_staging_offset = state->gin_layout.return_staging;
  args.return_receive_offset = state->gin_layout.return_receive;
  args.combine_compute_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.peer_bases_offset = layout.peer_bases;
  args.local_world_size = backend->local_world_size;
  args.node = backend->node;
  args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  gin_transport_slice_kernel<<<1, 256, 0, exec->stream>>>(args);
  internal::throw_cuda(
      "gin-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "GIN activation persistent kernel launch");
  internal::throw_cuda(
      "gin-run",
      cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H GIN status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "gin-run", internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(GIN complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "gin-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(GIN complete)");
  // A rank may destroy its registered window immediately after its result and
  // transport object leave scope. Rendezvous only after every device worker
  // has completed all remote puts so no peer can deregister early.
  internal::throw_nccl(
      "gin-run",
      ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllReduce(GIN terminal window lease)");
  internal::throw_cuda(
      "gin-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaStreamSynchronize(GIN terminal window lease)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure, "gin-run",
                    "device data plane", id.calling_rank, "valid",
                    "device error field=" +
                        std::to_string(slot.host_status->error_field));

  float total_ms = 0.0F;
  internal::throw_cuda(
      "gin-timing", cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(GIN total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->gin_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend = transport::ActivationTransportBackend::Gin;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.derived_route_input_bytes =
      view.sizes.route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes =
      view.sizes.route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.generation_memset_calls = 2;
  result->audit.generation_memset_bytes =
      sizeof(TransportDeviceStatus) + sizeof(std::uint64_t);
  result->audit.kernel_launches = 2;
  result->audit.dispatch_kernel_blocks = 1;
  result->audit.return_kernel_blocks = 1;
  result->audit.reduce_kernel_blocks = 1;
  result->audit.activation_d2h_bytes = 0;
  result->audit.activation_h2d_bytes = 0;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
#endif
}

DeviceForwardHandle LocalCudaP2pForwardMegaKernel::enqueue(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights,
    const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations || !weights)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-run",
                    "backend/dispatch/activation/weights", 0, "valid",
                    "empty");
  const auto backend = impl_->backend;
  const auto gin_backend =
      std::dynamic_pointer_cast<GinBackendState>(backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  if (id.top_k > 32U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "top_k", id.calling_rank,
                    "at most one route per warp lane (32)",
                    std::to_string(id.top_k));
  const auto &activation = *activations.impl_;
  const auto &weight = *weights.impl_;
  if ((!use_gin && id.num_nodes != 1) ||
      (use_gin && id.num_nodes != gin_backend->num_nodes) ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=forward-run, field=topology, rank=" +
            std::to_string(id.calling_rank) + ", expected=" +
            (use_gin ? "runtime GIN topology" : "one local node") +
            " and exact NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) +
            " nodes, world=" + std::to_string(id.world_size));
  // The hot GIN fanout record keeps its established 32-byte width by storing
  // the node-local rank and global expert ID in 16 bits.  Reject wider
  // topologies before any device-side narrowing instead of silently wrapping.
  if (use_gin &&
      (id.world_size > std::numeric_limits<std::uint16_t>::max() ||
       id.num_experts > std::numeric_limits<std::uint16_t>::max()))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "GIN fanout identity width", id.calling_rank,
                    "world_size and num_experts <= 65535",
                    std::to_string(id.world_size) + "/" +
                        std::to_string(id.num_experts));
  if ((activation.pool && activation.pool.get() != state.get()) ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (weight.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      weight.device != backend->execution.device_ordinal() ||
      weight.hidden != config.hidden_size ||
      weight.intermediate != config.intermediate_size || !weight.allocation ||
      !weight.allocation->ready)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "weight provenance", id.calling_rank,
                    "exact session/execution/device/shape/ready", "mismatch");
  auto lease = activation.slot
                   ? activation.slot
                   : acquire_slot(state, id.generation, id.calling_rank);
  if (state->config.max_hidden_size != config.hidden_size ||
      state->config.max_intermediate_size != config.intermediate_size ||
      !state->slots[lease->index].forward_maps_valid ||
      (config.retain_backward_state &&
       (!state->config.enable_backward_state ||
        state->layout.forward_training_output == 0)))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "forward pool geometry/backward state", id.calling_rank,
                    "Hopper 7168x3072 TMA maps and requested retained state",
                    "mismatch/missing");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  auto view = ActivationTransportBridge::view(consumer);
  // Notify materializes the route permutation and destination M-tile map for
  // every topology.  The split one-node transport consumes the same metadata
  // as GIN Dispatch even though it has no RDMA waves of its own.
  const bool split_local_metadata = !use_gin && id.num_nodes == 1U;
  auto dispatch_waves =
      use_gin || split_local_metadata
          ? ActivationTransportBridge::wave_view(consumer)
          : DispatchWaveKernelView{};
  const bool fused_inference_routes =
      ActivationTransportBridge::fused_inference_routes(dispatch);
  const bool static_v4_candidate =
      forward_tuning::kUnifiedWorkers &&
      (!use_gin || fused_inference_routes) &&
      !config.retain_backward_state &&
      std::getenv("WEFTMOE_DISABLE_STATIC_FORWARD_V4") == nullptr;
  if (state->layout.static_ring.depth &&
      (!static_v4_candidate || !fused_inference_routes || use_gin))
    throw std::invalid_argument("ring workspace requires fused local static inference");
  const bool serialized_graph_replay =
      static_v4_candidate && config.serialized_graph_replay;
  const bool direct_inference_routes = config.inference_routes.data != nullptr ||
                                       config.inference_gates.data != nullptr;
  if (direct_inference_routes) {
    if (!static_v4_candidate || !serialized_graph_replay ||
        !config.inference_routes.data || !config.inference_gates.data)
      throw std::invalid_argument(
          "direct inference routes require both buffers and serialized "
          "static inference");
    validate_external_cuda_range(
        config.inference_routes.data, view.sizes.route_count,
        config.inference_routes.bytes / sizeof(std::int32_t),
        sizeof(std::int32_t), backend->execution.device_ordinal(),
        "inference_routes");
    validate_external_cuda_range(
        config.inference_gates.data, view.sizes.route_count,
        config.inference_gates.bytes / sizeof(float), sizeof(float),
        backend->execution.device_ordinal(), "inference_gates");
  }
  if (weight.owned_experts != view.sizes.owned_expert_count)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "owned_experts", id.calling_rank,
                    std::to_string(view.sizes.owned_expert_count),
                    std::to_string(weight.owned_experts));
  auto tensors =
      config.tensors
          ? config.tensors
          : prepare_forward_tensors(dispatch, backend->execution, config);
  const auto &data = *tensors.impl_;
  if (data.tokens != id.num_tokens || data.hidden != config.hidden_size ||
      data.intermediate != config.intermediate_size ||
      data.rows < dispatch.received_row_capacity() ||
      data.training != config.retain_backward_state ||
      data.completion.device != backend->execution.device_ordinal())
    throw std::runtime_error(
        "Forward tensor storage does not match this invocation");
  const auto writes = {data.output, data.x, data.ab};
  tensor_storage_detail::validate_input_disjoint(
      writes, activation.pointer,
      id.num_tokens * config.hidden_size * sizeof(transport::BFloat16));
  const auto w2_bytes = static_cast<std::uint64_t>(weight.owned_experts) *
                        config.hidden_size * config.intermediate_size *
                        sizeof(transport::BFloat16);
  tensor_storage_detail::validate_input_disjoint(writes, weight.w1,
                                                 2 * w2_bytes);
  tensor_storage_detail::validate_input_disjoint(writes, weight.w2, w2_bytes);

  auto flight = std::make_shared<OperatorFlight>(lease, false);
  SlotFailureGuard failure_guard{lease, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[lease->index];
  GinExecution communication(state->gin_resources, exec->stream,
                             serialized_graph_replay);
  const auto &layout = state->layout;
#if WEFTMOE_DEBUG_VALIDATE
  const auto debug_forward_host_phase = [&](const char *phase) {
    if (std::getenv("WEFTMOE_FORWARD_PROGRESS") != nullptr)
      std::cerr << "FORWARD_RUN_HOST rank=" << id.calling_rank
                << " generation=" << id.generation << " phase=" << phase
                << std::endl;
  };
  debug_forward_host_phase("consumer_acquired");
#endif
  if (dispatch.received_row_capacity() >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count > checked_mul(state->config.max_tokens_per_rank,
                                           state->config.max_top_k,
                                           "forward route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");

  if (slot.transport_epoch >= (1ULL << 32U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "forward-run",
                    "transport_epoch", id.calling_rank, "<2^32-1",
                    std::to_string(slot.transport_epoch));
  const auto generation_epoch = ++slot.transport_epoch;
  const bool cancellation_requested =
      options.cancellation.cancellation_requested();
  __atomic_store_n(
      slot.host_cancel_epoch,
      cancellation_requested ? generation_epoch : 0ULL,
      __ATOMIC_RELEASE);
  if (cancellation_requested || slot.cancel_epoch_dirty) {
    internal::throw_cuda(
        "forward-run",
        cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                        sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                        exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "H2D initialize cancel epoch");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              sizeof(std::uint64_t));
    slot.cancel_epoch_dirty = cancellation_requested;
  }
  const bool activation_on_execution_stream =
      activation.producer_execution.same_instance(backend->execution.token()) ||
      (activation.allocation && activation.allocation->external_producer &&
       activation.allocation->external_producer->stream == exec->stream);
  if (!activation_on_execution_stream)
    internal::throw_cuda(
        "forward-run",
        internal::wait_input_ready(
            exec->stream,
            activation.allocation ? activation.allocation->ready
                                  : slot.producer_ready,
            activation.allocation ? activation.allocation->ready_capture_id
                                  : slot.producer_capture_id),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(activation ready)");
  const bool weights_on_execution_stream =
      weight.execution.same_instance(backend->execution.token()) ||
      (weight.allocation && weight.allocation->external_producer &&
       weight.allocation->external_producer->stream == exec->stream);
  if (!weights_on_execution_stream)
    internal::throw_cuda(
        "forward-run",
        internal::wait_input_ready(exec->stream, weight.allocation->ready,
                                   weight.allocation->ready_capture_id),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(weights ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("stream_dependencies_enqueued");
#endif

  if (!static_v4_candidate)
    internal::throw_cuda(
        "forward-run",
        internal::record_observable_event(slot.forward_start, exec->stream,
                                          &slot.forward_capture_id),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaEventRecord(forward start)");
#if WEFTMOE_DEVICE_TIMELINE
  reset_timeline("forward-run", slot.forward_timeline, state->timeline_layout,
                 state->timeline_max_ctas, exec->stream, id.calling_rank,
                 id.world_size, id.generation, backend->session);
#endif
  // Unified V3/V4 initialize status, generation signals and peer epochs in
  // forward_initialize_generation inside the resident grid.  Launching the
  // legacy split-grid initializer for V4 duplicates that work and inserts an
  // extra kernel between forward_start and the main kernel.
  if constexpr (!forward_tuning::kUnifiedWorkers) {
    internal::throw_cuda(
        "forward-run",
        cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                        sizeof(TransportDeviceStatus), exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts, "clear forward status");

    // Notify has already completed the rank-wide generation agreement on this
    // communicator.  The IPC team and per-slot peer mappings are
    // session-static, while payload visibility is ordered by the device
    // release/acquire signals below, so a second host-enqueued AllGather here
    // only serialized E2E.

    if (use_gin) {
#if __has_include(<nccl_device.h>)
    auto *gin_barrier =
        at<std::uint64_t>(slot.gin_window, state->gin_layout.barrier);
    internal::throw_cuda(
        "forward-run",
        cudaMemsetAsync(gin_barrier, 0, sizeof(std::uint64_t), exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts, "reset Forward GIN barrier");
    gin_signal_reset_kernel<<<gin_backend->gin_allocated_context_count, 1, 0,
                              exec->stream>>>(
        gin_backend->dev_comm, gin_backend->gin_allocated_context_count,
        3U + id.world_size);
    internal::throw_cuda("forward-run", cudaGetLastError(), id.calling_rank,
                         id.world_size, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "reset Forward GIN signal/counter");
    internal::throw_nccl(
        "forward-run",
        ncclAllReduce(gin_barrier, gin_barrier, 1, ncclUint64, ncclSum,
                      backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "ncclAllReduce(Forward GIN reset agreement)");
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("gin_reset_enqueued");
#endif
#endif
    }

  initialize_forward_generation_signals_kernel<<<1, 256, 0, exec->stream>>>(
      slot.device, at<void *>(slot.device, layout.peer_bases),
      layout.ingress_published_sequence, layout.outgoing_route_sequence,
      layout.p2p_barrier_receive, layout.gin_local_compute_ready,
      layout.forward_x_arrived_rows, view.sizes.mtile_count,
      layout.forward_completion, layout.forward_context_completion,
      static_cast<std::uint32_t>((state->config.max_nodes - 1U) * 8U),
      id.generation, id.calling_rank, id.world_size, state->config.channels,
      generation_epoch,
      use_gin ? gin_backend->node * gin_backend->local_world_size : 0U,
      use_gin ? gin_backend->local_world_size : id.world_size,
      at<unsigned long long>(slot.device, layout.execution_epoch),
      view.runtime);

  internal::throw_cuda(
      "forward-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "initialize owner completion signals and wait peer setup epoch");
  }
#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("generation_init_enqueued");
#endif
  ForwardKernelArgs args{};
  if (fused_inference_routes) {
    view.runtime = at<const DeviceDispatchRuntime>(
        slot.device, layout.fused_dispatch_runtime);
    view.expert_id =
        at<const std::uint32_t>(slot.device, layout.fused_route_expert);
    view.local_offset = at<const std::uint64_t>(
        slot.device, layout.fused_route_local_offset);
    view.return_global_expert_base = nullptr;
    view.route_dst_row =
        at<const std::uint64_t>(slot.device, layout.fused_route_dst_row);
    view.route_dst_rank =
        at<const std::uint32_t>(slot.device, layout.fused_route_dst_rank);
    view.route_dst_node = nullptr;
    view.route_dst_local_rank = nullptr;
    view.combine_order =
        at<const std::uint32_t>(slot.device, layout.fused_combine_order);
    view.source_expert_count = at<const std::uint64_t>(
        slot.device, layout.fused_source_expert_count);
    view.owned_expert_ids =
        at<const std::uint32_t>(slot.device, layout.fused_owned_expert_ids);
    view.expert_row_count = at<const std::uint64_t>(
        slot.device, layout.fused_expert_row_count);
    view.expert_row_prefix = at<const std::uint64_t>(
        slot.device, layout.fused_expert_row_prefix);
    view.mtile_row_offsets =
        at<const std::uint64_t>(slot.device, layout.fused_mtile_offsets);
    view.mtile_expected_rows =
        at<const std::uint32_t>(slot.device, layout.fused_mtile_rows);
    dispatch_waves.local_route_order = nullptr;
    dispatch_waves.route_dst_mtile =
        at<const std::uint32_t>(slot.device, layout.fused_route_dst_mtile);
  }
  args.dispatch = view;
  args.dispatch_waves = dispatch_waves;
  args.fused_inference_routes = fused_inference_routes;
  args.selected_experts =
      fused_inference_routes
          ? (direct_inference_routes
                 ? static_cast<const std::int32_t *>(
                       config.inference_routes.data)
                 : ActivationTransportBridge::route_pointer(dispatch))
          : nullptr;
  args.fused_local_expert_count =
      fused_inference_routes
          ? at<std::uint64_t>(slot.device, layout.fused_local_expert_count)
          : nullptr;
  args.fused_source_routes =
      fused_inference_routes
          ? at<std::uint32_t>(slot.device, layout.fused_source_routes)
          : nullptr;
  args.max_routes_per_rank =
      checked_mul(state->config.max_tokens_per_rank,
                  state->config.max_top_k,
                  "single-node source route capacity");
  args.source_input_offset = layout.input;
  args.fused_source_gates_offset = layout.fused_source_gates;
  args.input = activation.pointer;
  args.gates = direct_inference_routes
                   ? static_cast<const float *>(config.inference_gates.data)
                   : ActivationTransportBridge::gate_pointer(dispatch);
  args.w1 = weight.w1;
  args.w2 = weight.w2;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = state->config.channels;
  args.epoch = generation_epoch;
  args.runtime_epoch =
      at<unsigned long long>(slot.device, layout.execution_epoch);
  args.generation_setup_offset = layout.p2p_barrier_receive;
  args.completion_offset = layout.forward_completion;
  args.context_completion_offset = layout.forward_context_completion;
  args.context_completion_capacity = (state->config.max_nodes - 1U) * 8U;
  args.gin_allocated_context_count =
      use_gin ? gin_backend->gin_allocated_context_count : 0;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.use_gin = use_gin;
  args.local_world_size = id.world_size / id.num_nodes;
  args.node = id.calling_rank / args.local_world_size;
#if WEFTMOE_DEVICE_TIMELINE
  args.timeline = make_timeline_context(
      slot.forward_timeline, state->timeline_layout,
      state->timeline_max_ctas, id.calling_rank,
      profiling::TimelinePhase::Forward, id.generation);
#endif
#if WEFTMOE_BUILD_TESTING
  if (const auto *fault_rank =
          std::getenv("WEFTMOE_INTERNAL_FORWARD_STATUS_FAULT_RANK")) {
    const auto *fault_generation =
        std::getenv("WEFTMOE_INTERNAL_FORWARD_STATUS_FAULT_GENERATION");
    if (fault_generation == nullptr ||
        std::stoull(fault_generation) == id.generation)
      args.injected_status_error_rank = std::stoi(fault_rank);
  }
#endif
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[lease->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.static_network = state->gin_layout.static_forward;
    args.gin_rings = state->gin_layout.rings;
    args.gin_rings.counter_base +=
        lease->index * args.gin_rings.channels() * communication::kRingDepth;
    args.gin_send_header_offset = state->gin_layout.send_packet_headers;
    args.gin_receive_header_offset = state->gin_layout.receive_packet_headers;
    args.gin_send_dispatch_directory_offset =
        state->gin_layout.send_dispatch_directory;
    args.gin_receive_dispatch_directory_offset =
        state->gin_layout.receive_dispatch_directory;
    args.gin_send_fanout_offset = state->gin_layout.send_fanout;
    args.gin_receive_fanout_offset = state->gin_layout.receive_fanout;
    args.gin_send_payload_offset = state->gin_layout.send_payload;
    args.gin_receive_payload_offset = state->gin_layout.receive_payload;
    args.gin_return_staging_offset = state->gin_layout.return_staging;
    args.gin_return_receive_offset = state->gin_layout.return_receive;
    args.gin_return_send_rows_offset = state->gin_layout.return_send_rows;
    args.gin_return_receive_rows_offset = state->gin_layout.return_receive_rows;
    args.gin_production_error_records_offset =
        state->gin_layout.production_error_records;
    args.node = gin_backend->node;
    args.local_world_size = gin_backend->local_world_size;
    args.gin_context_count = gin_backend->gin_context_count;
    args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  }
  args.canonical_rows_offset = layout.canonical_rows;
  if (layout.static_ring.depth) {
    const auto &l = layout.static_ring;
    auto &r = args.ring;
    r.depth = l.depth;
    r.contribution_depth = l.contribution_depth;
    r.contribution_stride = std::uint64_t(l.shape.tokens) * state->config.max_top_k;
    r.state = at<static_forward::RingState>(slot.device, l.state);
    r.source_epochs = at<static_forward::SourceEpoch>(slot.device, l.source_epochs);
    r.slots = at<static_forward::ComputeSlot>(slot.device, l.slots);
    r.contributions = at<static_forward::ContributionSlot>(slot.device, l.contributions);
    r.contribution_ready = at<std::uint64_t>(slot.device, l.contribution_ready);
    r.window_prefix = at<std::uint64_t>(slot.device, l.window_prefix);
    r.expert_rows = at<std::uint64_t>(slot.device, l.expert_rows);
    r.expert_blocks = at<std::uint64_t>(slot.device, l.expert_blocks);
    r.blocks = at<static_forward::RingBlock>(slot.device, l.blocks);
    r.rows = at<static_forward::RingRow>(slot.device, l.rows);
    auto &p = r.source;
    p.shape = l.shape;
    p.shape.world = id.world_size;
    p.shape.sources = std::min(id.world_size, 16U);
    p.shape.top_k = id.top_k;
    p.packets = {32U, 2U};
    p.source_rank = id.calling_rank;
    p.experts = id.num_experts;
    p.nodes = id.num_nodes;
    p.tokens = id.num_tokens;
    p.selected_experts = args.selected_experts;
    p.valid = nullptr;
    p.expert_prefix = at<std::uint64_t>(slot.device, l.expert_prefix);
    p.node_prefix = at<std::uint64_t>(slot.device, l.node_prefix);
    p.route_order = at<std::uint64_t>(slot.device, l.route_order);
    p.source_slot = at<std::uint32_t>(slot.device, l.source_slot);
    p.node_tokens = at<static_forward::NodeToken>(slot.device, l.node_tokens);
    p.packet_prefix = at<std::uint64_t>(slot.device, l.packet_prefix);
    p.error = &r.state->error;
  }
  args.canonical_gates_offset = layout.canonical_gates;
  args.forward_output_rows_offset = config.retain_backward_state
                                        ? layout.forward_training_output
                                        : (layout.static_ring.depth ? layout.static_ring.output
                                                                    : layout.canonical_rows);
  args.direct_headers_offset = layout.direct_route_headers;
  args.contribution_offset = layout.contribution;
  args.return_ready_epoch_offset = layout.return_ready_epoch;
  args.output = static_cast<transport::BFloat16 *>(data.output->data);
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.status_offset = layout.status;
  args.forward_s_offset = layout.forward_s;
  args.saved_x = data.training
                     ? static_cast<transport::BFloat16 *>(data.x->data)
                     : nullptr;
#if !defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
  if (data.training && data.rows)
    args.saved_x_map = make_bf16_tma_store_map(
        data.x->data, config.hidden_size, data.rows, "saved FC1 X");
#endif
  args.saved_ab = data.training
                      ? static_cast<transport::BFloat16 *>(data.ab->data)
                      : nullptr;
  args.saved_rows = data.rows;
  args.retain_backward_state = config.retain_backward_state;
  args.forward_task_claim_offset = layout.forward_task_claim;
  args.forward_fc1_done_offset = layout.forward_fc1_done;
  args.forward_input_groups_offset = layout.forward_input_groups;
  args.x_arrived_rows_offset = layout.forward_x_arrived_rows;
  args.forward_reduce_route_count_offset = layout.forward_reduce_route_count;
  args.forward_reduce_ready_token_offset =
      layout.forward_reduce_ready_token;
  args.forward_return_rows_offset = layout.forward_return_rows;
  args.forward_return_queues_offset = layout.forward_return_queues;
  args.local_notification_offset = layout.local_notifications;
  args.local_notification_ctas = layout.local_notification_ctas;
  // Cross-node Forward has a single Return transport: FULL GIN. Dispatch
  // intentionally remains same-rail, while the second physical context set
  // gives each permanent RDMA CTA an independent FULL Return QP.
  if (use_gin && !gin_backend->full_connections)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "FULL Return connections", id.calling_rank,
                    "NCCL_GIN_CONNECTION_FULL", "unavailable");
  // Scheduler selection belongs to the generated tuning header. Runtime
  // environment overrides made one binary silently represent several kernels
  // and defeated the content-addressed JIT cache.
  args.forward_dynamic_reduce_scan = forward_tuning::kReduceScheduler == 1U;
  args.forward_dynamic_reduce_queue = forward_tuning::kReduceScheduler == 0U;
  args.forward_scheduler_offset = layout.forward_scheduler;
  args.forward_v3_control_offset = layout.forward_v3_control;
  if constexpr (forward_tuning::kUnifiedWorkers)
    args.gin_sender_done = at<void>(slot.device, layout.gin_sender_done);
  args.gin_local_compute_ready_offset = layout.gin_local_compute_ready;
  args.terminal_epoch_offset = layout.terminal_epoch;
  args.production_error_records_offset = layout.production_error_records;
  args.cancel_epoch =
      reinterpret_cast<const unsigned long long *>(slot.device_cancel_epoch);
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
  constexpr std::size_t forward_progress_base = 0;
  if (std::getenv("WEFTMOE_FORWARD_PROGRESS_DUMP") != nullptr) {
    std::fill_n(slot.host_forward_progress + forward_progress_base,
                kForwardProgressWords + 2ULL * slot.generation_debug_words,
                0ULL);
    args.progress = reinterpret_cast<unsigned long long *>(
        slot.device_forward_progress + forward_progress_base);
    args.progress_contexts = slot.generation_debug_contexts;
  }
#endif

  int cluster_supported = 0;
  int max_active_clusters = 0;
  internal::throw_cuda(
      "forward-run",
      cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaDevAttrClusterLaunch");
  int sms = 0;
    internal::throw_cuda(
        "forward-run",
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               static_cast<int>(exec->device)),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaDeviceGetAttribute(SM count)");
    const auto grid_blocks = static_cast<unsigned int>(sms);
    if (grid_blocks == 0)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          "phase=forward-run, field=SM count, rank=" +
              std::to_string(id.calling_rank) +
              ", expected>=2, actual=" + std::to_string(sms));
    const std::size_t forward_compute_smem =
        forward_tuning::compute_smem_bytes(id.world_size);
    const std::size_t forward_transport_smem =
        forward_tuning::transport_smem_bytes(
            static_cast<std::uint32_t>(args.hidden), id.world_size);
    const bool remote_transport = use_gin;
    const bool split_gin_roles =
        use_gin && view.identity.num_nodes >= 2U &&
        args.gin_context_count == persistent_v3::kRdmaCtaCount &&
        forward_tuning::kRdmaCtas == persistent_v3::kRdmaCtaCount;
    const bool split_local_roles =
        !remote_transport && view.identity.num_nodes == 1U;
    const bool split_forward_roles = split_gin_roles || split_local_roles;
    const bool static_v4_enabled = static_v4_candidate;
    if (static_v4_enabled) {
      // Direct FC2 push is made visible by one rank collective. Combine owns
      // immutable token stripes and does not consume V3 route-ready queues.
      args.forward_dynamic_reduce_queue = false;
      args.forward_dynamic_reduce_scan = false;
    }
    cudaError_t launch_forward = cudaSuccess;
    if (split_forward_roles || static_v4_candidate) {
      const auto &ordinary_output_store_map =
          config.retain_backward_state
              ? slot.forward_training_output_store_map
              : slot.forward_output_store_map;
      args.forward_relay_cta_count = split_local_roles
                                         ? forward_tuning::kRelayCtas
                                         : forward_tuning::kRemoteRelayCtas;
      const bool valid_relay_count =
          split_local_roles
              ? args.forward_relay_cta_count >=
                        std::max(4U, args.local_world_size) &&
                    (args.forward_relay_cta_count % 4U) == 0U
              : args.forward_relay_cta_count >=
                        persistent_v3::kRdmaCtaCount &&
                    args.forward_relay_cta_count %
                            persistent_v3::kRdmaCtaCount ==
                        0U;
      if (!valid_relay_count)
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "WEFTMOE_FORWARD_RELAY_CTAS", id.calling_rank,
                        split_local_roles
                            ? "multiple of 4, at least max(4, local_world_size)"
                            : "positive multiple of the RDMA context count",
                        std::to_string(args.forward_relay_cta_count));
      const unsigned int kTransportBlocks =
          (split_gin_roles ? persistent_v3::kRdmaCtaCount : 0U) +
          args.forward_relay_cta_count;
      if (!forward_tuning::kUnifiedWorkers && grid_blocks <= kTransportBlocks)
        throw NotifyDispatchError(
            NotifyErrorCode::BackendUnavailable,
            "phase=forward-run, field=split role residency, rank=" +
                std::to_string(id.calling_rank) + ", expected SMs>" +
                std::to_string(kTransportBlocks) +
                ", actual=" + std::to_string(grid_blocks));
      args.forward_compute_cta_count = forward_tuning::kUnifiedWorkers
          ? (grid_blocks & ~1U) : grid_blocks - kTransportBlocks;
      if constexpr (forward_tuning::kUnifiedWorkers)
        args.forward_relay_cta_count = args.forward_compute_cta_count -
            forward_tuning::unified_rdma_ctas(args.forward_compute_cta_count,
                                              args.use_gin);
      if constexpr (forward_tuning::kUnifiedWorkers) {
        if (use_gin) {
          args.gin_total_ctas = args.forward_compute_cta_count;
          args.gin_sender_count = args.forward_compute_cta_count - args.forward_relay_cta_count;
          // A zero dedicated-send budget retains one static sender per CTA.
          if (!args.gin_sender_count) args.gin_sender_count = args.gin_total_ctas;
          args.gin_receiver_base = args.forward_compute_cta_count - args.forward_relay_cta_count;
          args.gin_receiver_count = args.forward_relay_cta_count;
        }
      }
      args.forward_stream_local_dispatch = split_local_roles || static_v4_candidate;
      args.forward_compact_return = forward_tuning::kUnifiedWorkers;
      if (fused_inference_routes)
        args.forward_compact_return = false;
      args.forward_relay_local_return = forward_tuning::kRelayLocalReturn;
      // Eight independent row streams per CTA are the Hopper default.
      // Twelve streams slightly improve isolated NVLink
      // copy throughput, but contend more heavily with the paired FC1/FC2
      // pipeline and produce a longer end-to-end Return tail.  A value of
      // zero retains the exact vector implementation for autotuner/regression
      // A/B runs.
      args.forward_relay_tma_warps = forward_tuning::kRelayTmaWarps;
      if (args.forward_relay_tma_warps != 0U &&
          args.forward_relay_tma_warps != 4U &&
          args.forward_relay_tma_warps != 8U &&
          args.forward_relay_tma_warps != persistent_v3::kWarpsPerCta)
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "WEFTMOE_FORWARD_RELAY_TMA_WARPS", id.calling_rank,
                        "one of 0, 4, 8, or 12",
                        std::to_string(args.forward_relay_tma_warps));
      args.forward_local_return_batch_rows =
          forward_tuning::kLocalReturnBatchRows;
      if (args.forward_local_return_batch_rows < 32U ||
          args.forward_local_return_batch_rows > 1024U ||
          args.forward_local_return_batch_rows % 32U != 0U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "WEFTMOE_FORWARD_LOCAL_RETURN_BATCH_ROWS", id.calling_rank,
            "multiple of 32 in [32, 1024]",
            std::to_string(args.forward_local_return_batch_rows));
      args.forward_dispatch_publish_batch_rows =
          forward_tuning::kDispatchPublishBatchRows;
      if (args.forward_dispatch_publish_batch_rows != 4U &&
          args.forward_dispatch_publish_batch_rows != 8U &&
          args.forward_dispatch_publish_batch_rows != 16U &&
          args.forward_dispatch_publish_batch_rows != 32U &&
          args.forward_dispatch_publish_batch_rows != 64U &&
          args.forward_dispatch_publish_batch_rows != 128U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "WEFTMOE_FORWARD_DISPATCH_PUBLISH_BATCH_ROWS", id.calling_rank,
            "one of 4, 8, 16, 32, 64, or 128",
            std::to_string(args.forward_dispatch_publish_batch_rows));
      // A two-block base cadence keeps FC2/Return readiness fine-grained. The
      // saturated middle window widens to four for stronger W1 L2 reuse;
      // keeping its bounds tunable is important because widening near the
      // tail improves isolated GEMM time but lengthens Return/Reduce sharply.
      args.forward_fc1_locality_group = forward_tuning::kFc1LocalityGroup;
      if (args.forward_fc1_locality_group != 1U &&
          args.forward_fc1_locality_group != 2U &&
          args.forward_fc1_locality_group != 4U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "WEFTMOE_FORWARD_FC1_LOCALITY_GROUP", id.calling_rank,
            "one of 1, 2, or 4",
            std::to_string(args.forward_fc1_locality_group));
      const std::uint32_t locality_group4_begin =
          forward_tuning::kFc1LocalityGroup4Begin;
      const std::uint32_t locality_group4_end =
          forward_tuning::kFc1LocalityGroup4End;
      const auto locality_group4_span =
          locality_group4_end - locality_group4_begin;
      if (locality_group4_begin < 2U || locality_group4_end > UINT8_MAX ||
          (locality_group4_begin & 1U) != 0U ||
          locality_group4_end < locality_group4_begin ||
          (locality_group4_span & 3U) != 0U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "WEFTMOE_FORWARD_FC1_LOCALITY_GROUP4_BEGIN/END",
            id.calling_rank,
            "begin>=2 and even; end<=255; end>=begin; span divisible by 4",
            std::to_string(locality_group4_begin) + ":" +
                std::to_string(locality_group4_end));
      args.forward_fc1_locality_group4_begin =
          static_cast<std::uint8_t>(locality_group4_begin);
      args.forward_fc1_locality_group4_end =
          static_cast<std::uint8_t>(locality_group4_end);
      const auto forward_main_compute_ctas = args.forward_compute_cta_count;
      const bool prefill = static_v4_enabled &&
                           forward_tuning::kStaticV4PrefillMinTokens &&
                           id.num_tokens >=
                               forward_tuning::kStaticV4PrefillMinTokens;
      using PrefillGemm = forward_tuning::StaticV4Gemm<true>;
      const auto fc1_n = prefill ? PrefillGemm::kFc1N
                                 : forward_tuning::kFc1TileN;
      const auto fc2_n = prefill ? PrefillGemm::kFc2N
                                 : forward_tuning::kFc2TileN;
      const auto fc1_ntiles =
          (static_v4_enabled ? 2ULL : 1ULL) * args.intermediate / fc1_n;
      const auto fc2_ntiles = args.hidden / fc2_n;
      if ((forward_main_compute_ctas & 1U) || (fc1_ntiles & 1U) ||
          (fc2_ntiles & 1U))
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "Forward cluster extent", id.calling_rank,
            "even compute CTA and FC1/FC2 N-tile counts",
            "incompatible two-CTA multicast shape");
      if (static_v4_enabled) {
        const auto clusters =
            forward_main_compute_ctas / forward_tuning::kClusterSize;
        const auto max_mblocks =
            (state->config.max_destination_rows_per_rank +
             forward_tuning::kTileM - 1ULL) /
                forward_tuning::kTileM +
            state->config.max_owned_experts;
        const auto fc1_pairs = max_mblocks * fc1_ntiles / 2ULL;
        const auto fc2_units_per_mblock =
            (fc2_ntiles / 2U + forward_tuning::kFc2ClaimTiles - 1U) /
            forward_tuning::kFc2ClaimTiles;
        const auto fc2_units =
            max_mblocks * fc2_units_per_mblock;
        if (clusters == 0U ||
            fc1_pairs > 65535ULL * clusters ||
            fc2_units > 65535ULL * clusters)
          transport_error(
              NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
              "V4 per-cluster static task extent", id.calling_rank,
              "at most 65535 FC1 pairs and FC2 units per cluster",
              std::to_string(fc1_pairs) + "/" +
                  std::to_string(fc2_units));
      }
      // Every homogeneous CTA contributes its eight math warps to terminal
      // Reduce after its statically owned GEMM residues retire.
      args.forward_reduce_cta_count = forward_main_compute_ctas;
      if constexpr (forward_tuning::kUnifiedWorkers) {
        if (static_v4_enabled) {
          auto kStaticUnified = use_gin
              ? unified_forward_entry<false, false, true>()
              : unified_forward_entry<true, false, true>();
          if (args.ring.depth)
            kStaticUnified = unified_forward_entry<true, false, true, false, true>();
          if constexpr (forward_tuning::kStaticV4PrefillMinTokens) {
            if (prefill) {
              if (args.ring.depth)
                kStaticUnified = unified_forward_entry<true, false, true, true, true>();
              else
                kStaticUnified = use_gin
                    ? unified_forward_entry<false, false, true, true>()
                    : unified_forward_entry<true, false, true, true>();
            }
          }
          const auto smem =
              forward_unified_detail::shared_bytes<true>(id.world_size);
          const bool pdl = programmatic_launch::enabled();
          internal::throw_cuda(
              "forward-run",
              backend->prepare_unified_launch<
                  forward_tuning::StaticV4Resources>(
                  reinterpret_cast<const void *>(kStaticUnified), smem,
                  forward_main_compute_ctas, forward_tuning::kUnifiedThreads,
                  forward_tuning::kClusterSize, pdl, false,
                  &max_active_clusters),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts,
              "prepare cached V4 unified launch");
          cudaLaunchConfig_t launch{};
          launch.gridDim = dim3(forward_main_compute_ctas, 1, 1);
          launch.blockDim =
              dim3(forward_tuning::kUnifiedThreads, 1, 1);
          launch.dynamicSmemBytes = smem;
          launch.stream = exec->stream;
          cudaLaunchAttribute cluster{};
          cluster.id = cudaLaunchAttributeClusterDimension;
          cluster.val.clusterDim = {
              forward_tuning::kClusterSize, 1U, 1U};
          cudaLaunchAttribute attributes[2]{};
          unsigned attribute_count = 0U;
          attributes[attribute_count++] = cluster;
          if (pdl)
            attributes[attribute_count++] = programmatic_launch::attribute();
          launch.attrs = attributes;
          launch.numAttrs = attribute_count;
          if (static_cast<unsigned>(max_active_clusters) *
                  forward_tuning::kClusterSize <
              forward_main_compute_ctas)
            transport_error(
                NotifyErrorCode::BackendUnavailable, "forward-run",
                "V4 unified resident grid", id.calling_rank,
                std::to_string(forward_main_compute_ctas) + " resident CTAs",
                std::to_string(max_active_clusters) + " resident clusters");
          launch_forward = cudaLaunchKernelEx(
              &launch, kStaticUnified, args, slot.forward_x_map,
              slot.forward_s_map,
              prefill ? weight.prefill_w1_map : weight.w1_interleaved_map,
              prefill ? weight.prefill_w2_map : weight.w2_map,
              ordinary_output_store_map);
        } else {
        const auto launch_unified = [&](auto kernel) {
          const auto smem = forward_unified_detail::shared_bytes(id.world_size);
          const bool pdl = programmatic_launch::enabled();
#if !defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
          auto error = backend->prepare_unified_launch<
              forward_tuning::UnifiedResources>(
              reinterpret_cast<const void *>(kernel), smem,
              forward_main_compute_ctas, forward_tuning::kUnifiedThreads,
              forward_tuning::kClusterSize, pdl, true,
              &max_active_clusters);
#else
          // The communication diagnostic entry has no setmaxnreg partition.
          auto error = internal::prepare_kernel_stack(
              reinterpret_cast<const void *>(kernel));
          if (error == cudaSuccess)
            error = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem));
#endif
          if (error != cudaSuccess)
            return error;
          cudaLaunchConfig_t launch{};
          launch.gridDim = dim3(forward_main_compute_ctas, 1, 1);
          launch.blockDim = dim3(forward_tuning::kUnifiedThreads, 1, 1);
          launch.dynamicSmemBytes = smem;
          launch.stream = exec->stream;
          cudaLaunchAttribute cluster{};
          cluster.id = cudaLaunchAttributeClusterDimension;
          cluster.val.clusterDim = {forward_tuning::kClusterSize, 1U, 1U};
          cudaLaunchAttribute cooperative{};
          cooperative.id = cudaLaunchAttributeCooperative;
          cooperative.val.cooperative = 1;
          cudaLaunchAttribute attributes[] = {cluster, cooperative,
                                              programmatic_launch::attribute()};
          launch.attrs = attributes;
          launch.numAttrs = pdl ? 3U : 2U;
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
          error = cudaOccupancyMaxActiveClusters(&max_active_clusters, kernel,
                                                  &launch);
          if (error != cudaSuccess)
            return error;
#endif
          if (static_cast<unsigned>(max_active_clusters) *
                  forward_tuning::kClusterSize < forward_main_compute_ctas)
            return cudaErrorLaunchOutOfResources;
          return cudaLaunchKernelEx(
              &launch, kernel, args, slot.forward_x_map, slot.forward_s_map,
              weight.w1_map, weight.w2_map, ordinary_output_store_map);
        };
        if constexpr (forward_tuning::kRdmaCtas == 0U)
          launch_forward = config.retain_backward_state
              ? launch_unified(unified_forward_entry<true, true>())
              : launch_unified(unified_forward_entry<true, false>());
        else if (split_local_roles)
          launch_forward = config.retain_backward_state
              ? launch_unified(unified_forward_entry<true, true>())
              : launch_unified(unified_forward_entry<true, false>());
        else
          launch_forward = config.retain_backward_state
              ? launch_unified(unified_forward_entry<false, true>())
              : launch_unified(unified_forward_entry<false, false>());
        }
      } else {
      internal::throw_cuda(
          "forward-run",
          split_local_roles
              ? forward_transport_launch::set_local_max_dynamic_shared_memory(
                    static_cast<int>(forward_transport_smem))
              : forward_transport_launch::set_max_dynamic_shared_memory(
                    static_cast<int>(forward_transport_smem)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(split transport dynamic smem)");
      const auto set_compute_attribute = [&](auto kernel) {
        const auto error = internal::prepare_kernel_stack(
            reinterpret_cast<const void *>(kernel));
        if (error != cudaSuccess)
          return error;
        return cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
      };
      const auto set_tma_compute_attribute =
          [&](auto inference_kernel, auto training_kernel) {
            return set_compute_attribute(config.retain_backward_state
                                             ? training_kernel
                                             : inference_kernel);
          };
      cudaError_t compute_attribute = cudaSuccess;
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<true, forward_tuning::kFc1Stages,
                                               forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>);
      else
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                false, forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>);
      internal::throw_cuda(
          "forward-run", compute_attribute,
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(queued FC1/FC2 dynamic smem)");

      local_notification_detail::reset<<<
          (args.local_world_size * args.local_notification_ctas + 127U) / 128U,
          128, 0, exec->stream>>>(args);
      if (use_gin) {
        activation_ring_detail::prepare<<<
            (args.gin_rings.channels() + 127U) / 128U, 128, 0, exec->stream>>>(
            args);
        internal::throw_cuda(
            "forward-run", cudaGetLastError(), id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "prepare bounded GIN rings");
      }
      internal::throw_cuda(
          "forward-run",
          cudaEventRecord(slot.forward_roles_ready, exec->stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaEventRecord(split Forward roles ready)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(slot.forward_transport_stream,
                              slot.forward_roles_ready),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward transport ready)");
      internal::throw_cuda("forward-run",
                           cudaStreamWaitEvent(slot.forward_compute_stream,
                                               slot.forward_roles_ready),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaStreamWaitEvent(split Forward compute ready)");

      cudaLaunchConfig_t transport_launch{};
      transport_launch.gridDim = dim3(kTransportBlocks, 1, 1);
      transport_launch.blockDim = dim3(
          split_local_roles ? forward_tuning::kLocalTransportThreads
                            : forward_tuning::kComputeThreads,
          1, 1);
      transport_launch.dynamicSmemBytes = forward_transport_smem;
      transport_launch.stream = slot.forward_transport_stream;
      launch_forward = split_local_roles
                           ? forward_transport_launch::launch_local(
                                 &transport_launch, &args, sizeof(args))
                           : forward_transport_launch::launch(
                                 &transport_launch, &args, sizeof(args));
      internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           split_local_roles
                               ? "persistent 0-RDMA/tunable-Relay Forward launch"
                               : "persistent 8-RDMA/tunable-Relay Forward launch");
      internal::throw_cuda("forward-run",
                           cudaEventRecord(slot.forward_transport_done,
                                           slot.forward_transport_stream),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventRecord(split Forward transport done)");
      auto compute_args = args;
#if WEFTMOE_DEVICE_TIMELINE
      compute_args.timeline.cta_base = kTransportBlocks;
#endif
      cudaLaunchConfig_t compute_launch{};
      compute_launch.gridDim = dim3(forward_main_compute_ctas, 1, 1);
      compute_launch.blockDim =
          dim3(forward_tuning::kComputeThreads, 1, 1);
      compute_launch.dynamicSmemBytes = forward_compute_smem;
      compute_launch.stream = slot.forward_compute_stream;
      cudaLaunchAttribute compute_cluster_attribute{};
      compute_cluster_attribute.id = cudaLaunchAttributeClusterDimension;
      compute_cluster_attribute.val.clusterDim = {forward_tuning::kClusterSize,
                                                  1U, 1U};
      compute_launch.attrs = &compute_cluster_attribute;
      compute_launch.numAttrs = 1U;
      const auto launch_tma_compute =
          [&](auto inference_kernel, auto training_kernel,
              const CUtensorMap &output_store_map) {
            const auto kernel = config.retain_backward_state
                                    ? training_kernel
                                    : inference_kernel;
            return cudaLaunchKernelEx(
                &compute_launch, kernel, compute_args, slot.forward_x_map,
                slot.forward_s_map, weight.w1_map, weight.w2_map,
                output_store_map);
          };
      const auto &output_store_map = ordinary_output_store_map;
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<true, forward_tuning::kFc1Stages,
                                               forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            output_store_map);
      else
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                false, forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            output_store_map);
      internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           "persistent queued FC1/FC2 WGMMA/TMA Forward launch");

      internal::throw_cuda("forward-run",
                           cudaEventRecord(slot.forward_compute_done,
                                           slot.forward_compute_stream),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventRecord(split Forward compute done)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(exec->stream, slot.forward_transport_done),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward transport done)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(exec->stream, slot.forward_compute_done),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward compute done)");
      }
    } else {
      transport_error(NotifyErrorCode::BackendUnavailable, "forward-run",
                      "multicast transport", id.calling_rank,
                      "single-node NVLink or multi-node GIN",
                      "unsupported backend");
    }
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("kernel_launch_returned");
#endif
    internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                         id.world_size, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "single persistent WGMMA/TMA forward launch");

  if (config.retain_backward_state) {
    const auto save =
        [&](const std::shared_ptr<tensor_storage_detail::Allocation> &target,
            std::size_t offset, std::uint64_t row_bytes) {
          const auto bytes = dispatch.received_row_capacity() * row_bytes;
          if (!bytes)
            return;
          internal::throw_cuda(
              "forward-save",
              cudaMemcpyAsync(target->data, at<void>(slot.device, offset),
                              bytes, cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "save caller-owned Forward state");
        };
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
    // The communication probe skips FC1 and cannot checkpoint X in compute.
    save(data.x, layout.canonical_rows,
         config.hidden_size * sizeof(transport::BFloat16));
#endif
    save(data.gates, layout.canonical_gates, sizeof(float));
    save(data.headers, layout.direct_route_headers,
         sizeof(LocalP2pRouteHeader));
  }
  if (!serialized_graph_replay)
    tensors.impl_->completion.record(exec->stream);
  if (!static_v4_candidate)
    internal::throw_cuda("forward-run",
                         internal::record_observable_event(
                             slot.forward_output_ready, exec->stream),
                         id.calling_rank, id.world_size,
                         backend->session.session().identity(), id.generation,
                         ExchangeKind::ExpertCounts,
                         "cudaEventRecord(Forward output ready)");
  // Local P2P relies on the peer-visible terminal epoch. GIN payload and
  // return consumers have already acquired their per-generation signals;
  // the next Notify/reset agreement is the registered-window reuse boundary.
  if (!static_v4_candidate)
    internal::throw_cuda(
        "forward-run",
        internal::record_observable_event(slot.forward_complete, exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaEventRecord(forward complete)");
  if (!serialized_graph_replay) {
    internal::throw_cuda(
        "forward-run",
        internal::record_observable_event(slot.complete, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "record workspace completion");
    slot.completion_recorded = true;
  }

#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_enter");
#endif
  auto forward_stream_done = std::make_shared<std::atomic<bool>>(false);
  if (!static_v4_candidate)
    ActivationTransportBridge::submit(consumer);

#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_done");
#endif

  auto result = std::make_shared<DeviceForwardHandle::Impl>();
  result->slot = lease;
  result->workspace_index = lease->index;
  result->tensors = tensors;
  result->pool = state;
  result->backend = backend;
  result->dispatch = dispatch;
  result->activations = activations;
  result->weights = weights;
  result->inference_routes = config.inference_routes;
  result->inference_gates = config.inference_gates;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->retained_backward_state = config.retain_backward_state;
  result->serialized_graph_replay = serialized_graph_replay;
  result->compact_return = args.forward_compact_return;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->completion = std::make_unique<OperatorCompletion>();
  result->completion
      ->finish = [backend, state, flight, lease, id, options, args, use_gin,
                  cluster_supported, max_active_clusters, forward_stream_done,
                  static_v4_candidate, serialized_graph_replay,
#if WEFTMOE_DEBUG_VALIDATE
                  layout,
#endif
                  result = result.get(),
                  consumer_owner =
                      std::make_shared<DeviceDispatchConsumerLease>(
                          std::move(consumer))]() {
    auto &slot = state->slots[lease->index];
    auto exec = internal::checked_execution_state(backend->execution);
    auto &consumer = *consumer_owner;
    SlotFailureGuard failure_guard{lease, false};
#if WEFTMOE_DEBUG_VALIDATE
    const auto debug_forward_host_phase = [](const char *) {};
#endif
    // The slot lease keeps this mapped word unique until finish completes.
    // Publish only after the status copies and CUDA completion events above.
    // The default stream write includes a system-scope fence; it does not
    // depend on scheduling a driver callback thread to wake the consumer.
    __atomic_store_n(slot.host_forward_done, 0U, __ATOMIC_RELEASE);
    const auto write_status = cuStreamWriteValue32(
        exec->stream, reinterpret_cast<CUdeviceptr>(slot.device_forward_done),
        1U, CU_STREAM_WRITE_VALUE_DEFAULT);
    const bool mapped_completion = write_status == CUDA_SUCCESS;
    if (!mapped_completion && write_status != CUDA_ERROR_NOT_SUPPORTED) {
      const char *text = nullptr;
      (void)cuGetErrorString(write_status, &text);
      transport_error(NotifyErrorCode::CudaFailure, "forward-run",
                      "stream completion write", id.calling_rank,
                      "cuStreamWriteValue32 success",
                      text ? text : "unknown CUDA driver error");
    }
    if (!mapped_completion)
      internal::throw_cuda(
          "forward-run",
          cudaLaunchHostFunc(exec->stream, publish_stream_completion,
                             forward_stream_done.get()),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaLaunchHostFunc(Forward completion fallback)");
    const auto wait_start = std::chrono::steady_clock::now();
    bool cancelled = false;
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
    unsigned int progress_dump_count = 0U;
#endif
    while (true) {
      if (mapped_completion
              ? __atomic_load_n(slot.host_forward_done, __ATOMIC_ACQUIRE) != 0U
              : forward_stream_done->load(std::memory_order_acquire))
        break;
      const auto expired =
          options.deadline_after.count() > 0 &&
          std::chrono::steady_clock::now() - wait_start >= options.deadline_after;
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
      const auto stalled_for = std::chrono::steady_clock::now() - wait_start;
      const auto progress_dump_after = progress_dump_count == 0U
                                           ? std::chrono::seconds(3)
                                           : std::chrono::seconds(30);
      if (progress_dump_count < 2U && use_gin && args.progress != nullptr &&
          stalled_for >= progress_dump_after) {
        constexpr std::size_t forward_progress_base = 0;
        const auto progress = [&](std::uint32_t word) {
          return __atomic_load_n(slot.host_forward_progress +
                                     forward_progress_base + word,
                                 __ATOMIC_ACQUIRE);
        };
        // Emit the mapped coarse milestones before attempting a device-memory
        // snapshot. If a platform cannot schedule the diagnostic D2H while the
        // persistent grid is resident, this line still identifies the stalled
        // phase instead of losing the entire diagnostic.
        std::cerr << "FORWARD_PROGRESS_COARSE rank=" << id.calling_rank
                  << " generation=" << id.generation
                  << " sample=" << progress_dump_count
                  << " entered=" << progress(kForwardEntered)
                  << " setup=" << progress(kForwardSetupComplete)
                  << " dispatch_stored=" << progress(kForwardDispatchStored)
                  << " dispatch_observed="
                  << progress(kForwardDispatchObserved)
                  << " fc1=" << progress(kForwardFc1Complete)
                  << " fc2=" << progress(kForwardFc2Complete)
                  << " return=" << progress(kForwardReturnObserved)
                  << " reduce=" << progress(kForwardReduceComplete)
                  << " terminal=" << progress(kForwardTerminal) << std::endl;
        // Keep production completion atomics in device memory even in debug
        // builds. Redirecting the per-task counters to mapped host memory
        // perturbs the Forward schedule by hundreds of milliseconds and can
        // hide the race being diagnosed. A pair of copy-engine snapshots on the
        // independent cancel stream remains observable while the persistent
        // kernel occupies every SM.
        auto *completion_snapshot = slot.host_forward_progress +
                                    forward_progress_base + kForwardProgressWords;
        internal::throw_cuda(
            "forward-progress-snapshot",
            cudaMemcpyAsync(completion_snapshot,
                            at<void>(slot.device, layout.forward_completion),
                            sizeof(transport_state::GenerationCompletionVector),
                            cudaMemcpyDeviceToHost, state->cancel_stream),
            id.calling_rank, id.world_size, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "D2H Forward generation completion snapshot");
        internal::throw_cuda(
            "forward-progress-snapshot",
            cudaMemcpyAsync(
                completion_snapshot + transport_state::kGenerationCompletionWords,
                at<void>(slot.device, layout.forward_context_completion),
                static_cast<std::size_t>(slot.generation_debug_contexts) *
                    sizeof(transport_state::ContextCompletionRecord),
                cudaMemcpyDeviceToHost, state->cancel_stream),
            id.calling_rank, id.world_size, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "D2H Forward context completion snapshot");
        internal::throw_cuda(
            "forward-progress-snapshot",
            cudaStreamSynchronize(state->cancel_stream), id.calling_rank,
            id.world_size, backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "drain Forward completion snapshot");
        std::cerr << "FORWARD_PROGRESS rank=" << id.calling_rank
                  << " generation=" << id.generation
                  << " sample=" << progress_dump_count;
        dump_generation_completion(std::cerr, completion_snapshot,
                                   slot.generation_debug_contexts,
                                   "forward_completion");
        std::cerr << std::endl;
        ++progress_dump_count;
      }
#endif
      if (!cancelled &&
          (expired || options.cancellation.cancellation_requested())) {
        __atomic_store_n(slot.host_cancel_epoch, args.epoch, __ATOMIC_RELEASE);
        internal::throw_cuda(
            "forward-cancel",
            cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                            sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                            state->cancel_stream),
            id.calling_rank, id.world_size, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "H2D publish forward cancel");
        internal::throw_cuda(
            "forward-cancel", cudaStreamSynchronize(state->cancel_stream),
            id.calling_rank, id.world_size, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "drain forward cancel stream");
        slot.cancel_epoch_dirty = true;
        cancelled = true;
      }
      // Do not call CUDA event-query APIs while the resident GIN kernel is
      // active. On the remote host that can hold a driver-side progress lock and
      // form a liveness cycle with device GIN. The stream-ordered mapped word
      // (or callback fallback) signals completion without querying that lock.
      // Synchronous wait stays runnable: a nominal 50us nanosleep can delay
      // completion by tens of milliseconds under host scheduling jitter.
      // enqueue() remains asynchronous; only a caller waiting for its result
      // occupies a host thread here. Cancellation/deadline checks stay above.
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause");
#elif defined(__aarch64__)
      asm volatile("yield");
#else
      std::atomic_signal_fence(std::memory_order_acq_rel);
#endif
    }
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("stream_completion_observed");
#endif
    internal::throw_cuda(
        "forward-run",
        cudaMemcpyAsync(slot.host_forward_status,
                        at<void>(slot.device, state->layout.status),
                        sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                        exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "D2H forward status");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              sizeof(TransportDeviceStatus));
    internal::throw_cuda(
        "forward-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "complete D2H forward status");
    if (serialized_graph_replay) {
      result->tensors.impl_->completion.record(exec->stream);
      internal::throw_cuda(
          "forward-run",
          internal::record_observable_event(slot.complete, exec->stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts,
          "record serialized graph workspace completion");
      slot.completion_recorded = true;
      internal::throw_cuda(
          "forward-run", cudaEventSynchronize(slot.complete), id.calling_rank,
          id.world_size, backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts,
          "complete serialized graph workspace event");
    }
    const auto completion_event =
        static_v4_candidate ? slot.complete : slot.forward_complete;
    internal::throw_cuda(
        "forward-run", cudaEventQuery(completion_event), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "cudaEventQuery(completed Forward event)");
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("event_verified");
#endif
    consumer.finish();
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("consumer_finished");
    debug_forward_host_phase("terminal_agreement_enter");
#endif
    const auto production_agreement = agree_production_terminal_status(
        backend, slot, ProductionStatusPhase::Forward, id.generation,
        id.generation, id.calling_rank, id.world_size, *slot.host_forward_status,
        completion_event);
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("terminal_agreement_done");
#endif
    if (cancelled || slot.host_forward_status->error == 2U)
      throw NotifyDispatchError(
          NotifyErrorCode::TransportTimeout,
          "phase=forward-run, field=cancel/deadline, rank=" +
              std::to_string(id.calling_rank) +
              ", expected=completed forward, actual=cancelled");
    if (production_agreement.has_error) {
      // A logical device-status failure reached the fixed terminal agreement;
      // all peer work is drained and this slot is recoverable. Backend/CUDA
      // failures still leave the guard uncommitted and therefore quarantine.
      failure_guard.committed = true;
      throw_production_status_error(production_agreement);
    }

    float total_ms = 0.0F;
    if (!static_v4_candidate)
      internal::throw_cuda(
          "forward-timing",
          cudaEventElapsedTime(&total_ms, slot.forward_start,
                               slot.forward_output_ready),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventElapsedTime(forward total)");
    result->audit.specialization =
        ForwardKernelSpecialization::Sm90WgmmaTma7168x3072;
    result->audit.persistent_kernel_launches = 1;
    result->audit.wgmma_instructions =
        slot.host_forward_status->wgmma_instructions;
    result->audit.tma_loads = slot.host_forward_status->tma_loads;
    result->audit.tma_multicast_loads =
        slot.host_forward_status->tma_multicast_loads;
    result->audit.runtime_cluster_supported = cluster_supported != 0;
    result->audit.requested_cluster_size = forward_tuning::kClusterSize;
    result->audit.runtime_cluster_size =
        slot.host_forward_status->runtime_cluster_size;
    result->audit.observed_cluster_rank_mask =
        slot.host_forward_status->cluster_rank_mask;
    result->audit.max_active_clusters =
        static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
    result->audit.fc1_tasks = slot.host_forward_status->fc1_tasks_done;
    result->audit.fc2_tasks = slot.host_forward_status->fc2_tasks_done;
    result->audit.actual_dispatch_peer_bytes =
        slot.host_forward_status->actual_dispatch_peer_bytes;
    result->audit.actual_return_peer_bytes =
        slot.host_forward_status->actual_return_peer_bytes;
    result->audit.gin_dispatch_put_count =
        slot.host_forward_status->gin_dispatch_put_count;
    result->audit.raw_cross_node_route_bytes =
        slot.host_forward_status->raw_cross_node_payload_bytes;
    result->audit.gin_dispatch_payload_bytes =
        slot.host_forward_status->deduplicated_cross_node_payload_bytes;
    result->audit.gin_return_payload_bytes =
        slot.host_forward_status->gin_return_payload_bytes;
    result->audit.gin_return_put_count =
        slot.host_forward_status->gin_return_put_count;
    result->audit.gin_signal_count =
        slot.host_forward_status->gin_signal_count;
    result->audit.gin_metadata_bytes =
        slot.host_forward_status->gin_metadata_bytes;
    result->audit.workspace_bytes = state->layout.total;
    result->timings.terminal_status_agreement_us =
        production_agreement.elapsed_us;
    const auto &status = *slot.host_forward_status;
    const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
      return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
    };
    result->timings.device_kernel_us =
        elapsed(status.start_clock, status.complete_clock);
    result->timings.total_us = static_v4_candidate
                                   ? result->timings.device_kernel_us
                                   : static_cast<double>(total_ms) * 1000.0;
    result->timings.route_exchange_us =
        elapsed(status.route_exchange_start_clock,
                status.route_exchange_complete_clock);
    result->timings.dispatch_us =
        elapsed(status.route_exchange_complete_clock
                    ? status.route_exchange_complete_clock
                    : status.start_clock,
                status.dispatch_complete_clock);
    result->timings.fc1_swiglu_us =
        elapsed(status.fc1_start_clock, status.fc1_complete_clock);
    result->timings.fc2_us =
        elapsed(status.dispatch_blocks_done, status.fc2_complete_clock);
    result->timings.before_combine_us =
        elapsed(status.before_combine_start_clock,
                status.before_combine_complete_clock);
    result->timings.combine_return_us =
        elapsed(status.relay_blocks_done, status.return_complete_clock);
    result->timings.reduce_us =
        elapsed(status.reduced_tokens, status.reduce_blocks_done);
    result->timings.after_combine_us =
        elapsed(status.after_combine_start_clock,
                status.after_combine_complete_clock);
    result->timings.fc1_fc2_overlap_us =
        elapsed(status.dispatch_blocks_done, status.fc1_complete_clock);
    result->timings.fc2_return_overlap_us =
        elapsed(status.relay_blocks_done, status.fc2_complete_clock);
    result->timings.return_reduce_overlap_us =
        elapsed(status.reduced_tokens, status.return_complete_clock);
    failure_guard.committed = true;
    result->activations = {};
    if (!result->tensors.impl_->completion.capture)
      result->slot.reset();
  };
  failure_guard.committed = true;
  communication.record();
  return DeviceForwardHandle(std::move(result));
}

DeviceForwardHandle LocalCudaP2pForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch, const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  auto handle = enqueue(dispatch, activations, weights, config, options);
  handle.wait();
  return handle;
}

const ForwardTensorStorage &DeviceForwardHandle::tensors() const {
  (void)identity();
  return impl_->tensors;
}

void DeviceForwardHandle::wait() const {
  (void)identity();
  impl_->completion->wait();
}

void DeviceForwardHandle::wait_on(CUstream_st *stream) const {
  (void)identity();
  impl_->tensors.impl_->completion.wait_on(stream);
}

void DeviceForwardHandle::wait_workspace_on(CUstream_st *stream) const {
  (void)identity();
  const auto &slot = impl_->pool->slots[impl_->workspace_index];
  internal::throw_cuda(
      "forward-workspace-wait", cudaStreamWaitEvent(stream, slot.complete),
      impl_->identity.rank, impl_->identity.world,
      impl_->backend->session.session().identity(), impl_->identity.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(workspace complete)");
}

void DeviceForwardHandle::release_serialized_graph_workspace() const {
  (void)identity();
  if (impl_->retained_backward_state || !impl_->serialized_graph_replay)
    throw std::runtime_error("serialized graph workspace release requires captured inference");
  wait();
  impl_->slot.reset();
}

const ActivationTransportExpectation &DeviceForwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->identity;
}

const ForwardMegaKernelAudit &DeviceForwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  wait();
  return impl_->audit;
}

const ForwardMegaKernelTimings &DeviceForwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  wait();
  return impl_->timings;
}

const transport::BFloat16 *
forward_output_device_pointer_for_testing(const DeviceForwardHandle &handle) {
  return static_cast<const transport::BFloat16 *>(
      handle.impl_->tensors.impl_->output->data);
}

std::vector<transport::BFloat16> download_forward_output_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  internal::throw_cuda(
      "forward-download",
      cudaEventSynchronize(impl.tensors.impl_->completion.ready),
      impl.identity.rank, impl.identity.world, "forward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait Forward output");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "forward download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "forward download elements"));
  if (!output.empty()) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16),
                    "forward download bytes"),
        "forward download bytes");
    internal::throw_cuda(
        "forward-download",
        cudaMemcpy(output.data(), impl.tensors.impl_->output->data, bytes,
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(forward output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}

std::vector<transport::BFloat16>
download_forward_routes_for_validation(const DeviceForwardHandle &handle,
                                       const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-routes-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (impl.pool->layout.static_ring.depth)
    throw std::invalid_argument(
        "static ring Forward does not retain a full route contribution snapshot");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-routes-download", "device", impl.identity.rank,
                    "handle device", "foreign");
  const auto check = [&](cudaError_t status) {
    internal::throw_cuda("forward-routes-download", status, impl.identity.rank,
                         impl.identity.world, "forward-validation",
                         impl.identity.generation, ExchangeKind::ExpertCounts,
                         "snapshot route contributions");
  };
  check(cudaEventSynchronize(impl.tensors.impl_->completion.ready));
  const auto elements = checked_mul(
      checked_mul(impl.identity.tokens, impl.dispatch.identity().top_k,
                  "forward route count"),
      impl.identity.hidden, "forward route elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "route elements"));
  if (!output.empty()) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(output.front()), "forward route bytes"),
        "route bytes");
    check(cudaMemcpy(output.data(),
                     at<void>(impl.pool->slots[impl.workspace_index].device,
                              impl.pool->layout.contribution),
                     bytes, cudaMemcpyDeviceToHost));
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  if (impl.compact_return && !output.empty()) {
    // Preserve the route-major validation contract independently of the
    // production payload layout. Metadata and CPU gathering are debug-only.
    const auto &id = impl.dispatch.identity();
    const DeviceDispatchViewExpectation expected{
        id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
        id.config_digest_word0, id.config_digest_word1};
    auto consumer = impl.dispatch.acquire_consumer(execution, expected);
    const auto view = ActivationTransportBridge::view(consumer);
    const auto routes = impl.identity.tokens * id.top_k;
    std::vector<std::uint32_t> experts(routes);
    std::vector<std::uint64_t> offsets(routes), prefixes(id.num_experts);
    const auto copy_metadata = [&](auto &destination, const auto *source) {
      const auto bytes = destination.size() * sizeof(destination.front());
      check(cudaMemcpy(destination.data(), source, bytes, cudaMemcpyDeviceToHost));
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
    };
    copy_metadata(experts, view.expert_id);
    copy_metadata(offsets, view.local_offset);
    copy_metadata(prefixes, view.return_global_expert_base);
    consumer.finish();
    std::vector<transport::BFloat16> ordered(output.size());
    for (std::uint64_t route = 0; route < routes; ++route) {
      if (experts[route] >= prefixes.size())
        throw std::runtime_error("invalid compact Return expert in validation snapshot");
      const auto row = prefixes[experts[route]] + offsets[route];
      if (row >= routes)
        throw std::runtime_error("invalid compact Return row in validation snapshot");
      std::copy_n(output.data() + row * impl.identity.hidden, impl.identity.hidden,
                  ordered.data() + route * impl.identity.hidden);
    }
    return ordered;
  }
  return output;
}

ForwardBackwardStateValidationDownload
download_forward_backward_state_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "forward-backward-state-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (!impl.retained_backward_state || !impl.tensors)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-backward-state-download", "retained state",
                    impl.identity.rank, "enabled", "disabled");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-backward-state-download", "device",
                    impl.identity.rank, "handle device", "foreign");

  ForwardBackwardStateValidationDownload result;
  const auto x_elements = checked_mul(
      impl.incoming_rows, impl.identity.hidden,
      "forward retained canonical X elements");
  const auto plane_elements = checked_mul(
      impl.incoming_rows, impl.intermediate,
      "forward retained A/B plane elements");
  result.canonical_x.resize(
      checked_size(x_elements, "forward retained canonical X elements"));
  result.fc1_ab.resize(checked_size(
      checked_mul(plane_elements, 2ULL, "forward retained A/B elements"),
      "forward retained A/B elements"));
  if (impl.incoming_rows == 0)
    return result;

  const auto &saved = *impl.tensors.impl_;
  const auto copy = [&](void *destination, const void *source,
                        std::uint64_t elements, const char *name) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16), name), name);
    internal::throw_cuda(
        "forward-backward-state-download",
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.canonical_x.data(), saved.x->data, x_elements,
       "cudaMemcpy(retained canonical X)");
  const auto *ab = static_cast<const transport::BFloat16 *>(saved.ab->data);
  copy(result.fc1_ab.data(), ab, plane_elements,
       "cudaMemcpy(retained FC1 A)");
  const auto b_plane_stride = checked_mul(saved.rows, impl.intermediate,
                                          "forward retained B plane stride");
  copy(result.fc1_ab.data() + plane_elements, ab + b_plane_stride,
       plane_elements, "cudaMemcpy(retained FC1 B)");
  return result;
}
const ActivationTransportExpectation &
DeviceActivationTransportHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}
const ActivationTransportDeviceAudit &
DeviceActivationTransportHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}
const ActivationTransportTimings &
DeviceActivationTransportHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

ActivationTransportValidationDownload ActivationTransportBridge::download_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto hidden = impl.identity.hidden;
  const auto route_elements = checked_mul(impl.route_count, hidden,
                                          "download route elements");
  const auto combined_elements = checked_mul(impl.identity.tokens, hidden,
                                             "download combined elements");
  const auto canonical_elements =
      checked_mul(impl.destination_rows, hidden, "download canonical elements");
  ActivationTransportValidationDownload result;
  result.contribution_by_route.resize(
      checked_size(route_elements, "download route elements"));
  result.combined_tokens.resize(
      checked_size(combined_elements, "download combined elements"));
  result.canonical_expert_rows.resize(
      checked_size(canonical_elements, "download canonical elements"));
  auto copy = [&](void *destination, std::size_t offset, std::uint64_t count,
                  const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-download",
        cudaMemcpy(destination, at<const void>(slot.device, offset),
                   checked_size(checked_mul(count,
                                            sizeof(transport::BFloat16), name),
                                name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
  };
  const auto contribution_offset =
      impl.audit.backend ==
                  transport::ActivationTransportBackend::NcclSocketFallback ||
              impl.audit.backend ==
                  transport::ActivationTransportBackend::Gin
          ? impl.pool->layout.combine_return
          : impl.pool->layout.contribution;
  copy(result.contribution_by_route.data(), contribution_offset,
       route_elements, "download contributions");
  copy(result.combined_tokens.data(), impl.pool->layout.combined,
       combined_elements, "download combined");
  copy(result.canonical_expert_rows.data(), impl.pool->layout.canonical_rows,
       canonical_elements, "download canonical rows");
  result.protocol.dispatch_packets = slot.host_status->packets_done;
  result.protocol.relayed_fanout = slot.host_status->fanout_done;
  result.protocol.synthetic_contributions = slot.host_status->synthetic_done;
  result.protocol.combine_packets = slot.host_status->combine_done;
  result.protocol.reduced_tokens = slot.host_status->reduced_tokens;
  result.protocol.credit_stalls = slot.host_status->credit_stalls;
  result.protocol.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result.protocol.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  return result;
}

ActivationTransportValidationDownload download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  return ActivationTransportBridge::download_for_validation(handle, execution);
}

std::vector<transport::BFloat16> download_activation_output_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "transport-output-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "transport-output-download", "device",
                    impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "output download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "output download elements"));
  if (!output.empty()) {
    const auto bytes = checked_mul(elements,
                                   sizeof(transport::BFloat16),
                                   "output download bytes");
    internal::throw_cuda(
        "transport-output-download",
        cudaMemcpy(output.data(),
                   at<const void>(
                       impl.pool->slots[impl.slot->index].device,
                       impl.pool->layout.combined),
                   checked_size(bytes, "output download bytes"),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(final output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}
namespace {

double backward_elapsed_us(cudaEvent_t begin, cudaEvent_t end,
                           const ActivationTransportExpectation &identity,
                           const char *name) {
  float milliseconds = 0.0F;
  internal::throw_cuda(
      "backward-timing", cudaEventElapsedTime(&milliseconds, begin, end),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, name);
  return static_cast<double>(milliseconds) * 1000.0;
}

} // namespace

DeviceBackwardHandle BackwardMegaKernel::enqueue(
    const DeviceForwardHandle &forward,
    const BFloat16OutputGradientLease &output_gradients,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) const {
  if (!forward.impl_ || !output_gradients.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "forward/output gradients", 0, "valid", "empty");
  const auto &retained = *forward.impl_;
  const auto &gradient = *output_gradients.impl_;
  const auto &identity = retained.identity;
  if (!retained.retained_backward_state || !retained.tensors ||
      !retained.pool || !retained.backend || !retained.weights.impl_ ||
      !gradient.allocation || !gradient.allocation->ready)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "retained Forward state", identity.rank,
                    "complete cached X/A/B state and ready dY", "missing");
  if (config.hidden_size != backward_tuning::kMainHidden ||
      config.intermediate_size != backward_tuning::kMainIntermediate ||
      config.hidden_size != identity.hidden ||
      config.intermediate_size != retained.intermediate)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "specialization shape", identity.rank, "compiled Backward dimensions",
                    std::to_string(config.hidden_size) + "x" +
                        std::to_string(config.intermediate_size));
  if (gradient.generation != identity.generation ||
      gradient.tokens != identity.tokens || gradient.hidden != identity.hidden ||
      gradient.device != retained.pool->device ||
      gradient.session_identity != retained.pool->session_identity)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "dY provenance", identity.rank,
                    "same generation/session/device/[tokens,H] as Forward",
                    "mismatch");
  if (options.cancellation.cancellation_requested() ||
      options.deadline_after.count() != 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "cancellation/deadline", identity.rank,
                    "disabled for local correctness milestone", "requested");
  const auto local_backend =
      std::dynamic_pointer_cast<LocalP2pBackendState>(retained.backend);
  const auto gin_backend =
      std::dynamic_pointer_cast<GinBackendState>(retained.backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  const bool valid_local =
      local_backend && identity.world == local_backend->peer_slot_bases.size();
  const bool valid_gin =
      gin_backend && gin_backend->num_nodes >= 2U &&
      gin_backend->local_world_size == backward_tuning::kRdmaCtas &&
      gin_backend->gin_context_count == backward_tuning::kRdmaCtas &&
      gin_backend->full_connections;
  if (!valid_local && !valid_gin)
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "transport backend", identity.rank,
                    "one-node LocalCudaP2p or multi-node 8-rail GIN retained "
                    "Forward",
                    "unsupported");
  const auto &pool = retained.pool;
  const auto &layout = pool->layout;
  auto lease = retained.slot
                   ? retained.slot
                   : acquire_slot(pool, identity.generation, identity.rank);
  if (!pool->config.enable_backward_state || layout.backward_dab_rows == 0 ||
      layout.backward_dx_rows == 0 ||
      !pool->slots[lease->index].backward_maps_valid)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "Backward workspace", identity.rank,
                    "training-enabled pool with owned expert capacity",
                    "missing");
  const auto &dispatch_id = retained.dispatch.identity();
  if (dispatch_id.top_k > 32U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "top_k", identity.rank, "at most one route per warp lane (32)",
                    std::to_string(dispatch_id.top_k));
  const auto expectation = DeviceDispatchViewExpectation{
      dispatch_id.generation, dispatch_id.calling_rank,
      dispatch_id.world_size, dispatch_id.num_tokens, dispatch_id.top_k,
      dispatch_id.config_digest_word0, dispatch_id.config_digest_word1};
  auto consumer = retained.dispatch.acquire_consumer(retained.backend->execution,
                                                      expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  const auto dispatch_waves = ActivationTransportBridge::wave_view(consumer);
  if (view.sizes.incoming_row_count != retained.incoming_rows ||
      view.sizes.owned_expert_count != retained.weights.impl_->owned_experts)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "retained Dispatch geometry", identity.rank,
                    "same rows/experts as Forward", "mismatch");

  auto exec = internal::checked_execution_state(retained.backend->execution);

  auto tensors =
      config.tensors
          ? config.tensors
          : prepare_backward_tensors(retained.dispatch,
                                     retained.backend->execution, config);
  const auto &data = *tensors.impl_;
  const auto &saved = *retained.tensors.impl_;
  if (data.tokens != identity.tokens || data.hidden != config.hidden_size ||
      data.intermediate != config.intermediate_size ||
      data.top_k != dispatch_id.top_k ||
      data.experts != view.sizes.owned_expert_count ||
      data.completion.device != pool->device)
    throw std::runtime_error(
        "Backward tensor storage does not match this invocation");
  const auto writes = {data.dx, data.dgate, data.dw1, data.dw2};
  for (const auto &input :
       {saved.output, saved.x, saved.ab, saved.gates, saved.headers})
    tensor_storage_detail::validate_input_disjoint(writes, input->data,
                                                   input->bytes);
  tensor_storage_detail::validate_input_disjoint(writes, gradient.pointer,
                                                 data.dx->bytes);
  tensor_storage_detail::validate_input_disjoint(
      writes, retained.weights.impl_->w1, data.dw1->bytes);
  tensor_storage_detail::validate_input_disjoint(
      writes, retained.weights.impl_->w2, data.dw2->bytes);

  auto &slot = pool->slots[lease->index];
  GinExecution communication(pool->gin_resources, exec->stream);
  auto flight = std::make_shared<OperatorFlight>(lease, true);
  SlotFailureGuard failure_guard{lease, false};
  const auto scheduler_mblock_capacity =
      (layout.backward_padded_capacity_rows + backward_tuning::kTileM - 1ULL) /
      backward_tuning::kTileM;
  if (scheduler_mblock_capacity > UINT32_MAX)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "scheduler M-block capacity",
                    identity.rank, "uint32",
                    std::to_string(scheduler_mblock_capacity));
  const auto scheduler_layout =
      persistent_v3::backward::make_backward_scheduler_layout(
          scheduler_mblock_capacity,
          static_cast<std::uint32_t>(config.hidden_size /
                                     backward_tuning::kDxTileN /
                                     backward_tuning::kActivationClusterN),
          pool->config.max_owned_experts);
  const auto scheduler_capacity_words =
      (layout.forward_fc1_done - layout.forward_task_claim) /
      sizeof(unsigned long long);
  if (scheduler_layout.total_words > scheduler_capacity_words)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "aliased Backward scheduler words",
                    identity.rank, std::to_string(scheduler_capacity_words),
                    std::to_string(scheduler_layout.total_words));
  BackwardKernelArgs args{};
  args.dispatch = view;
  args.dispatch_waves = dispatch_waves;
  args.dy = gradient.pointer;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows = pool->config.max_destination_rows_per_rank;
  args.padded_capacity_rows = layout.backward_padded_capacity_rows;
  // Actual padded rows and active experts are loaded from the device directory.
  args.scheduler_words = scheduler_layout.total_words;
  args.scheduler_mblock_capacity =
      static_cast<std::uint32_t>(scheduler_mblock_capacity);
  args.scheduler_expert_capacity = pool->config.max_owned_experts;
  args.rank = identity.rank;
  args.world = identity.world;
  args.node = use_gin ? gin_backend->node : 0U;
  args.local_world_size =
      use_gin ? gin_backend->local_world_size : identity.world;
  args.gin_context_count =
      use_gin ? gin_backend->gin_context_count : 0U;
  args.max_tokens_per_rank = pool->config.max_tokens_per_rank;
  args.backward_return_chunk_rows = backward_tuning::kTileM;
  args.local_notification_offset = layout.local_notifications;
  args.local_notification_ctas = layout.local_notification_ctas;
  if constexpr (backward_tuning::kUnifiedWorkers)
    args.gin_sender_done = at<void>(slot.device, layout.gin_sender_done);
  args.use_gin = use_gin;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[lease->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.gin_rings = pool->gin_layout.rings;
    args.gin_rings.counter_base +=
        lease->index * args.gin_rings.channels() * communication::kRingDepth;
    args.forward_return_queues_offset = layout.forward_return_queues;
    args.forward_return_rows_offset = layout.forward_return_rows;
  }
  args.saved_gates = static_cast<const float *>(saved.gates->data);
  args.saved_ab = static_cast<const transport::BFloat16 *>(saved.ab->data);
  args.saved_rows = saved.rows;
  // FC2 output is dead after Forward reduction. Reuse its compact plane for
  // the reverse dY dispatch instead of allocating and filling padded dR.
  args.compact_dr_offset = layout.forward_training_output;
  args.recomputed_u_offset = layout.backward_u;
  args.dab_offset = layout.backward_dab_rows;
  args.dx_rows_offset = layout.backward_dx_rows;
  args.dgate_rows_offset = layout.backward_dgate_rows;
  args.dgate_output_offset = layout.backward_dgate_output;
  args.dw1 = static_cast<transport::BFloat16 *>(data.dw1->data);
  args.dw2 = static_cast<transport::BFloat16 *>(data.dw2->data);
  args.contribution_offset = layout.contribution;
  args.dx = static_cast<transport::BFloat16 *>(data.dx->data);
  args.direct_headers_offset = layout.direct_route_headers;
  args.compact_to_padded_offset = layout.backward_compact_to_padded;
  args.padded_expert_rows_offset = layout.backward_padded_expert_rows;
  args.padded_expert_offsets_offset = layout.backward_padded_expert_offsets;
  args.setup_epoch_offset = layout.backward_setup_epoch;
  args.dispatch_ready_offset = layout.backward_dispatch_ready;
  args.return_ready_offset = layout.backward_return_ready;
  args.scheduler_offset = layout.forward_task_claim;
  args.status_offset = layout.status;
  // Compute queues use an execution epoch. Wire-slot sequences are separate
  // and remain monotonic across Forward, Backward and graph replays.
  if (slot.transport_epoch >= (1ULL << 32U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "backward-run",
                    "transport_epoch", identity.rank, "<2^32-1", "exhausted");
  const auto backward_epoch = ++slot.transport_epoch;
  args.epoch = backward_epoch;
  args.runtime_epoch =
      at<unsigned long long>(slot.device, layout.execution_epoch);
  if (std::getenv("WEFTMOE_BACKWARD_PROGRESS") != nullptr) {
    std::cerr << "BACKWARD_ADDR rank=" << identity.rank
              << " epoch=" << backward_epoch
              << " status=" << at<void>(slot.device, layout.status)
              << " host_status=" << static_cast<void *>(slot.host_status)
              << " scheduler=" << at<void>(slot.device, args.scheduler_offset)
              << " scheduler_words=" << args.scheduler_words
              << " mblocks=" << args.mblocks
              << " mblock_capacity=" << args.scheduler_mblock_capacity
              << " expert_capacity=" << args.scheduler_expert_capacity;
    if (use_gin)
      std::cerr << " ring_channels=" << args.gin_rings.channels()
                << " ring_depth=" << communication::kRingDepth;
    std::cerr << std::endl;
  }

  auto transport_stream = slot.forward_transport_stream;
  auto compute_stream = slot.forward_compute_stream;
  int sm_count = 0;
  internal::throw_cuda(
      "backward-run",
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "query Backward SM count");
  const auto transport_ctas =
      (use_gin ? backward_tuning::kRemoteRelayCtas
               : backward_tuning::kRelayCtas) +
      (use_gin ? backward_tuning::kRdmaCtas : 0U);
  if (!backward_tuning::kUnifiedWorkers && sm_count <= static_cast<int>(transport_ctas))
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "split Backward role residency", identity.rank,
                    "more SMs than dedicated transport CTAs",
                    std::to_string(sm_count));
  constexpr std::uint32_t kActivationStorageTileN =
      backward_tuning::kDabTileN > backward_tuning::kDxTileN
          ? backward_tuning::kDabTileN
          : backward_tuning::kDxTileN;
  constexpr std::uint32_t kStorageTileN =
      kActivationStorageTileN > backward_tuning::kDwTileN
          ? kActivationStorageTileN
          : backward_tuning::kDwTileN;
  constexpr auto kPersistentSmem =
      backward_compute_detail::dynamic_shared_bytes<kStorageTileN>();
  internal::throw_cuda(
      "backward-run",
      cudaFuncSetAttribute(
          backward_local_transport_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kPersistentSmem)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts,
      "reserve one SM per Backward Relay CTA");
  internal::throw_cuda(
      "backward-run",
      cudaFuncSetAttribute(
          backward_local_compute_kernel<backward_tuning::kDabTileN,
                                        backward_tuning::kDxTileN,
                                        backward_tuning::kDwTileN>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kPersistentSmem)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts,
      "set persistent Backward compute shared memory");
  const auto x_map = make_bf16_tma_map_mn(
      saved.rows ? saved.x->data : at<void>(slot.device, layout.canonical_rows),
      config.hidden_size, std::max<std::uint64_t>(saved.rows, 1),
      "saved X for dW1");
  const auto dw1_map =
      make_bf16_tma_store_map(data.dw1->data, config.hidden_size,
                              static_cast<std::uint64_t>(data.experts) * 2ULL *
                                  config.intermediate_size,
                              "caller dW1");
  const auto dw2_map = make_bf16_tma_store_map(
      data.dw2->data, config.intermediate_size,
      static_cast<std::uint64_t>(data.experts) * config.hidden_size,
      "caller dW2");
  const auto compute_ctas = backward_tuning::kUnifiedWorkers
      ? (static_cast<std::uint32_t>(sm_count) & ~1U)
      : ((static_cast<std::uint32_t>(sm_count) - transport_ctas) & ~1U);
  if constexpr (backward_tuning::kUnifiedWorkers) {
    args.local_notification_ctas = compute_ctas;
    if (use_gin) {
      args.gin_total_ctas = compute_ctas;
      args.gin_sender_count = forward_tuning::unified_rdma_ctas(compute_ctas, true);
      if (!args.gin_sender_count) args.gin_sender_count = 1U;
      args.gin_receiver_base = args.gin_sender_count;
      args.gin_receiver_count = compute_ctas - args.gin_sender_count;
    }
  }
  int cluster_supported = 0;
  internal::throw_cuda(
      "backward-run",
      cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                             static_cast<int>(exec->device)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "query Backward cluster support");
  if (!cluster_supported || compute_ctas < 2U)
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "Backward cluster extent", identity.rank,
                    "cluster launch support and at least two compute SMs",
                    std::to_string(compute_ctas));
  const auto launch_compute_workers =
      [&](std::uint32_t ctas, cudaStream_t stream, const char *detail) {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {2U, 1U, 1U};
        cudaLaunchConfig_t launch{};
        launch.gridDim = dim3(ctas);
        launch.blockDim = dim3(backward_tuning::kComputeThreads);
        launch.dynamicSmemBytes = kPersistentSmem;
        launch.stream = stream;
        launch.attrs = &attribute;
        launch.numAttrs = 1;
        internal::throw_cuda(
            "backward-run",
            cudaLaunchKernelEx(
                &launch,
                backward_local_compute_kernel<backward_tuning::kDabTileN,
                                              backward_tuning::kDxTileN,
                                              backward_tuning::kDwTileN>,
                args, slot.backward_dr_map, slot.backward_dab_map,
                retained.weights.impl_->backward_w2_map,
                retained.weights.impl_->backward_w1_map,
                slot.backward_dr_mn_map, slot.backward_u_mn_map,
                slot.backward_dab_mn_map, x_map, slot.backward_u_store_map,
                slot.backward_dx_store_map, dw2_map, dw1_map),
            identity.rank, identity.world, "backward", identity.generation,
            ExchangeKind::ExpertCounts, detail);
      };
  saved.completion.wait_on(exec->stream);
  if (retained.dispatch.received_row_capacity())
    internal::throw_cuda(
        "backward-restore-routing",
        cudaMemcpyAsync(at<void>(slot.device, layout.direct_route_headers),
                        saved.headers->data,
                        retained.dispatch.received_row_capacity() *
                            sizeof(LocalP2pRouteHeader),
                        cudaMemcpyDeviceToDevice, exec->stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts,
        "restore retained routing into execution workspace");
  local_notification_detail::reset<<<
      (args.local_world_size * args.local_notification_ctas + 127U) / 128U, 128,
      0, exec->stream>>>(args);
  // Respect all preceding caller-stream work, including captured consumers of
  // the Forward output, before the transport stream reuses its output plane.
  if (use_gin) {
    if constexpr (backward_tuning::kUnifiedWorkers)
      internal::throw_cuda("backward-run",
          cudaMemsetAsync(args.gin_sender_done, 0,
                          args.gin_sender_count * sizeof(unsigned),
                          exec->stream),
          identity.rank, identity.world, "backward", identity.generation,
           ExchangeKind::ExpertCounts, "initialize unified Backward GIN sender completion");
    activation_ring_detail::prepare<<<(args.gin_rings.channels() + 127U) / 128U,
                                      128, 0, exec->stream>>>(args);
    internal::throw_cuda("backward-run", cudaGetLastError(), identity.rank,
                         identity.world, "backward", identity.generation,
                         ExchangeKind::ExpertCounts,
                         "prepare bounded GIN rings");
  }
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.forward_roles_ready, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward caller-stream dependency");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(transport_stream, slot.forward_roles_ready),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait Backward caller-stream dependency");
  internal::throw_cuda(
      "backward-run",
      internal::wait_input_ready(transport_stream, gradient.allocation->ready,
                                 gradient.allocation->ready_capture_id),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait dY producer");
  backward_build_row_directory_kernel<<<1, 256, 0, transport_stream>>>(args);
  internal::throw_cuda("backward-plan", cudaGetLastError(), identity.rank,
      identity.world, "backward", identity.generation, ExchangeKind::ExpertCounts,
      "build Backward expert directory on device");
  backward_setup_pipeline_kernel<<<1, backward_tuning::kSetupThreads, 0,
                                   transport_stream>>>(args);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), identity.rank, identity.world,
      "backward", identity.generation, ExchangeKind::ExpertCounts,
      "initialize fixed Backward pipeline state");
  internal::throw_cuda(
      "backward-run",
      internal::record_observable_event(slot.backward_start, transport_stream,
                                        &slot.backward_capture_id),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward start");
  if (view.sizes.owned_expert_count != 0U) {
    backward_clear_inactive_gradients_kernel<<<
        dim3(128, view.sizes.owned_expert_count), 256, 0, transport_stream>>>(args);
    internal::throw_cuda("backward-run", cudaGetLastError(), identity.rank,
        identity.world, "backward", identity.generation, ExchangeKind::ExpertCounts,
        "clear inactive expert gradients on device");
  }

  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.forward_roles_ready,
                                       transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "publish Backward launch prerequisites");
  internal::throw_cuda("backward-run",
                       internal::record_observable_event(
                           slot.backward_compute_start, transport_stream),
                       identity.rank, identity.world, "backward",
                       identity.generation, ExchangeKind::ExpertCounts,
                       "record Backward compute start");
  if constexpr (backward_tuning::kUnifiedWorkers) {
    const auto kernel = [&] {
      if constexpr (backward_tuning::kRdmaCtas == 0U)
        return unified_backward_kernel<backward_tuning::kDabTileN,
            backward_tuning::kDxTileN, backward_tuning::kDwTileN, false>;
      else
        return use_gin
            ? unified_backward_kernel<backward_tuning::kDabTileN,
                backward_tuning::kDxTileN, backward_tuning::kDwTileN, true>
            : unified_backward_kernel<backward_tuning::kDabTileN,
                backward_tuning::kDxTileN, backward_tuning::kDwTileN, false>;
    }();
    const auto smem = backward_unified_detail::shared_bytes();
    internal::throw_cuda("backward-run",
        retained.backend->validate_unified_register_pool<backward_tuning::UnifiedResources>(
            reinterpret_cast<const void *>(kernel)),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "validate unified Backward register pool");
    internal::throw_cuda("backward-run",
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem)),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "set unified Backward shared memory");
    cudaLaunchAttribute cluster{};
    cluster.id = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim = {2U, 1U, 1U};
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(compute_ctas);
    launch.blockDim = dim3(backward_tuning::kUnifiedThreads);
    launch.dynamicSmemBytes = smem;
    launch.stream = transport_stream;
    launch.attrs = &cluster;
    launch.numAttrs = 1U;
    int clusters = 0;
    internal::throw_cuda("backward-run",
        cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "query unified Backward residency");
    if (2U * static_cast<unsigned>(clusters) < compute_ctas)
      transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                      "unified Backward residency", identity.rank,
                      "all worker CTAs resident", std::to_string(clusters));
    internal::throw_cuda("backward-run",
        cudaLaunchKernelEx(&launch, kernel, args,
            slot.backward_dr_map, slot.backward_dab_map,
            retained.weights.impl_->backward_w2_map,
            retained.weights.impl_->backward_w1_map, slot.backward_dr_mn_map,
            slot.backward_u_mn_map, slot.backward_dab_mn_map, x_map,
            slot.backward_u_store_map, slot.backward_dx_store_map, dw2_map, dw1_map),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "launch unified Backward workers");
    for (auto event : {slot.backward_transport_done, slot.backward_compute_done,
                       slot.forward_compute_done})
      internal::throw_cuda("backward-run",
          internal::record_observable_event(event, transport_stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "record unified Backward completion");
  } else {
  backward_local_transport_kernel<<<transport_ctas,
                                    backward_tuning::kComputeThreads,
                                    kPersistentSmem,
                                    transport_stream>>>(args);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), identity.rank, identity.world,
      "backward", identity.generation, ExchangeKind::ExpertCounts,
      "persistent local Backward Relay launch");
  internal::throw_cuda("backward-run",
                       internal::record_observable_event(
                           slot.backward_transport_done, transport_stream),
                       identity.rank, identity.world, "backward",
                       identity.generation, ExchangeKind::ExpertCounts,
                       "record Backward transport completion");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(compute_stream, slot.forward_roles_ready),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "start Backward compute roles");
  launch_compute_workers(compute_ctas, compute_stream,
                         "persistent Backward cluster multicast launch");
  internal::throw_cuda("backward-run",
                       internal::record_observable_event(
                           slot.backward_compute_done, compute_stream),
                       identity.rank, identity.world, "backward",
                       identity.generation, ExchangeKind::ExpertCounts,
                       "record Backward compute completion");

  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.forward_compute_done, compute_stream), identity.rank,
      identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "publish Backward compute dependency");

  }

  // Unified communication warps reduce dX before they retire. Legacy grids
  // keep the standalone reducer on the transport stream to overlap dW.
  if constexpr (!backward_tuning::kUnifiedWorkers) {
    // Queue route reduction behind the persistent transport grid. The compute
    // CTAs remain resident, while reducers use SMs released by communication.
    // Each token warp waits independently and retains the fixed route order.
    if (identity.tokens != 0ULL) {
      constexpr auto kReduceWarps = backward_tuning::kReduceThreads / 32U;
      const auto reduce_ctas = static_cast<unsigned int>(
          (identity.tokens + kReduceWarps - 1ULL) / kReduceWarps);
      backward_reduce_dx_kernel<<<reduce_ctas, backward_tuning::kReduceThreads,
                                  0, transport_stream>>>(args);
      internal::throw_cuda(
          "backward-run", cudaGetLastError(), identity.rank, identity.world,
          "backward", identity.generation, ExchangeKind::ExpertCounts,
          "streaming Backward dX reduce launch");
    }
    internal::throw_cuda("backward-run",
                         internal::record_observable_event(
                             slot.backward_reduce_done, transport_stream),
                         identity.rank, identity.world, "backward",
                         identity.generation, ExchangeKind::ExpertCounts,
                         "record Backward reduction completion");
    // Add workers to the dW queues after transport and reduction release their
    // SMs, preserving the initial residency needed for activation progress.
    launch_compute_workers(transport_ctas, transport_stream,
                           "late Backward compute-worker launch");
    internal::throw_cuda("backward-run",
                         internal::record_observable_event(
                             slot.backward_late_compute_done, transport_stream),
                         identity.rank, identity.world, "backward",
                         identity.generation, ExchangeKind::ExpertCounts,
                         "record late Backward compute-worker completion");
  }
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.forward_transport_done, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "publish Backward transport dependency");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.forward_compute_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "join Backward compute roles");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.forward_transport_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts,
      "join late Backward compute workers and reduction");
  if (data.dgate->bytes)
    internal::throw_cuda(
        "backward-output",
        cudaMemcpyAsync(data.dgate->data,
                        at<void>(slot.device, layout.backward_dgate_output),
                        data.dgate->bytes, cudaMemcpyDeviceToDevice,
                        exec->stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "publish caller-owned dGate");
  tensors.impl_->completion.record(exec->stream);
  internal::throw_cuda(
      "backward-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "download Backward pipeline clocks");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "backward-run",
      internal::record_observable_event(slot.backward_complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward complete");
  internal::throw_cuda(
      "backward-run",
      internal::record_observable_event(slot.complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record pool slot complete");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  auto result = std::make_shared<DeviceBackwardHandle::Impl>();
  result->slot = lease;
  result->workspace_index = lease->index;
  result->tensors = tensors;
  result->pool = pool;
  result->forward = forward;
  result->output_gradients = output_gradients;
  result->identity = identity;
  result->owned_experts = view.sizes.owned_expert_count;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->completion = std::make_unique<OperatorCompletion>();
  result->completion
      ->finish = [forward, pool, flight, lease, identity, config, args, use_gin,
                  backward_epoch, result = result.get(),
                  consumer_owner =
                      std::make_shared<DeviceDispatchConsumerLease>(
                          std::move(consumer))]() {
    forward.wait();
    const auto &retained = *forward.impl_;
    auto &slot = pool->slots[lease->index];
    const auto &layout = pool->layout;
    auto &consumer = *consumer_owner;
    SlotFailureGuard failure_guard{lease, false};
    const bool diagnostic_progress =
        std::getenv("WEFTMOE_BACKWARD_PROGRESS") != nullptr;
    const auto diagnostic_status = [&](const TransportDeviceStatus &s,
                                       bool pending) {
      std::cerr << "BACKWARD_STATUS rank=" << identity.rank
                << " epoch=" << backward_epoch << " pending=" << pending
                << " return_puts=" << s.gin_return_put_count
                << " return_bytes=" << s.gin_return_payload_bytes
                << " dispatch_complete=" << s.dispatch_complete_clock
                << " dx_complete=" << s.fc2_complete_clock
                << " return_complete=" << s.return_complete_clock
                << std::endl;
    };
    if (diagnostic_progress) {
      const auto started = std::chrono::steady_clock::now();
      bool sampled = false;
      while (cudaEventQuery(slot.backward_complete) == cudaErrorNotReady) {
        if (!sampled && std::chrono::steady_clock::now() - started >
                            std::chrono::seconds(3) &&
            slot.forward_progress_words * sizeof(std::uint64_t) >=
                sizeof(TransportDeviceStatus)) {
          internal::throw_cuda(
              "backward-diagnostic",
              cudaMemcpyAsync(slot.host_forward_progress,
                              at<void>(slot.device, layout.status),
                              sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                              pool->cancel_stream),
              identity.rank, identity.world, "backward", identity.generation,
              ExchangeKind::ExpertCounts, "snapshot pending Backward status");
          internal::throw_cuda(
              "backward-diagnostic", cudaStreamSynchronize(pool->cancel_stream),
              identity.rank, identity.world, "backward", identity.generation,
              ExchangeKind::ExpertCounts, "drain Backward status snapshot");
          diagnostic_status(*reinterpret_cast<const TransportDeviceStatus *>(
                                slot.host_forward_progress), true);
          for (std::uint64_t offset = 0; offset < args.scheduler_words;) {
            const auto count = std::min<std::uint64_t>(
                slot.forward_progress_words, args.scheduler_words - offset);
            internal::throw_cuda(
                "backward-diagnostic",
                cudaMemcpyAsync(slot.host_forward_progress,
                    at<std::uint64_t>(slot.device, args.scheduler_offset) + offset,
                    count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                    pool->cancel_stream),
                identity.rank, identity.world, "backward", identity.generation,
                ExchangeKind::ExpertCounts, "snapshot Backward scheduler");
            internal::throw_cuda(
                "backward-diagnostic", cudaStreamSynchronize(pool->cancel_stream),
                identity.rank, identity.world, "backward", identity.generation,
                ExchangeKind::ExpertCounts, "drain Backward scheduler snapshot");
            std::cerr << "BACKWARD_SCHED rank=" << identity.rank
                      << " epoch=" << backward_epoch << " offset=" << offset
                      << " values=";
            for (std::uint64_t i = 0; i < count; ++i)
              std::cerr << (i ? "," : "") << slot.host_forward_progress[i];
            std::cerr << std::endl;
            offset += count;
          }
          sampled = true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    internal::throw_cuda(
        "backward-run", cudaEventSynchronize(slot.backward_complete),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "wait Backward completion");
    if (diagnostic_progress)
      diagnostic_status(*slot.host_status, false);
    consumer.finish();

    result->audit.specialization =
        BackwardKernelSpecialization::Sm90CachedXAb7168x3072;
    result->audit.gemm_launches = 2ULL;
    result->audit.actual_dispatch_peer_bytes =
        retained.audit.actual_dispatch_peer_bytes;
    const auto remote_return_rows =
        retained.audit.actual_return_peer_bytes /
        (config.hidden_size * sizeof(transport::BFloat16));
    result->audit.actual_return_peer_bytes =
        use_gin ? remote_return_rows * args.gin_rings.return_row_bytes
                : retained.audit.actual_return_peer_bytes +
                      remote_return_rows * sizeof(float);
    result->audit.gin_return_put_count = slot.host_status->gin_return_put_count;
    result->audit.gin_return_payload_bytes =
        slot.host_status->gin_return_payload_bytes;
    result->audit.workspace_bytes = pool->layout.total;
    result->audit.hot_path_allocations = 0;
    const auto device_span_us = [](unsigned long long begin,
                                   unsigned long long end) {
      return begin == ~0ULL || end == 0ULL || end < begin
                 ? 0.0
                 : static_cast<double>(end - begin) / 1000.0;
    };
    result->timings.prepare_us = backward_elapsed_us(
        slot.backward_start, slot.backward_compute_start, identity,
        "Backward cache preparation elapsed");
    const auto primary_compute_us = backward_elapsed_us(
        slot.backward_compute_start, slot.backward_compute_done, identity,
        "Backward primary persistent compute elapsed");
    if constexpr (backward_tuning::kUnifiedWorkers) {
      result->timings.compute_us = primary_compute_us;
      // The embedded reduction has device clocks; an adjacent host event
      // would measure only the empty stream interval after the kernel.
      result->timings.reduce_us = device_span_us(
          slot.host_status->backward_reduce_start_clock,
          slot.host_status->backward_reduce_complete_clock);
    } else {
      const auto late_compute_us = backward_elapsed_us(
          slot.backward_compute_start, slot.backward_late_compute_done, identity,
          "Backward late compute elapsed");
      result->timings.compute_us = std::max(primary_compute_us, late_compute_us);
      result->timings.reduce_us = backward_elapsed_us(
          slot.backward_transport_done, slot.backward_reduce_done, identity,
          "Backward dX reduction elapsed");
    }
    result->timings.transport_us = backward_elapsed_us(
        slot.backward_compute_start, slot.backward_transport_done, identity,
        "Backward persistent transport elapsed");
    result->timings.dispatch_us = device_span_us(
        slot.host_status->start_clock,
        slot.host_status->dispatch_complete_clock);
    result->timings.dab_us = device_span_us(
        slot.host_status->fc1_start_clock,
        slot.host_status->fc1_complete_clock);
    result->timings.dx_us = device_span_us(
        slot.host_status->relay_blocks_done,
        slot.host_status->fc2_complete_clock);
    result->timings.dw2_us = device_span_us(
        slot.host_status->backward_dw2_start_clock,
        slot.host_status->backward_dw2_complete_clock);
    result->timings.dw1_us = device_span_us(
        slot.host_status->backward_dw1_start_clock,
        slot.host_status->backward_dw1_complete_clock);
    result->timings.combine_us = device_span_us(
        slot.host_status->fc2_complete_clock,
        slot.host_status->return_complete_clock);
    const auto dw_after_dx = device_span_us(
        slot.host_status->fc2_complete_clock,
        slot.host_status->backward_dw1_complete_clock);
    result->timings.dw_combine_overlap_us =
        std::min(dw_after_dx, result->timings.combine_us);
    result->timings.total_us = backward_elapsed_us(
        slot.backward_start, slot.backward_complete, identity,
        "Backward total elapsed");
    failure_guard.committed = true;
    if (!result->tensors.impl_->completion.capture) {
      result->slot.reset();
      result->forward = {};
      result->output_gradients = {};
#if !WEFTMOE_DEVICE_TIMELINE
      result->pool.reset();
#endif
    }
  };
  failure_guard.committed = true;
  communication.record();
  return DeviceBackwardHandle(std::move(result));
}

DeviceBackwardHandle BackwardMegaKernel::run(
    const DeviceForwardHandle &forward, const BFloat16OutputGradientLease &gradients,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) const {
  auto handle = enqueue(forward, gradients, config, options);
  handle.wait();
  return handle;
}

const BackwardTensorStorage &DeviceBackwardHandle::tensors() const {
  (void)identity();
  return impl_->tensors;
}

void DeviceBackwardHandle::wait() const {
  (void)identity();
  impl_->completion->wait();
}

void DeviceBackwardHandle::wait_on(CUstream_st *stream) const {
  (void)identity();
  impl_->tensors.impl_->completion.wait_on(stream);
}

const ActivationTransportExpectation &DeviceBackwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}

const BackwardMegaKernelAudit &DeviceBackwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  wait();
  return impl_->audit;
}

const BackwardMegaKernelTimings &DeviceBackwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  wait();
  return impl_->timings;
}

BackwardValidationDownload BackwardValidationBridge::download(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  if (!handle.impl_ || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.tensors.impl_->completion.device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  internal::throw_cuda(
      "backward-download",
      cudaEventSynchronize(impl.tensors.impl_->completion.ready),
      impl.identity.rank, impl.identity.world, "backward",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait Backward output");
  const auto &data = *impl.tensors.impl_;
  BackwardValidationDownload result;
  result.dx.resize(checked_size(
      checked_mul(impl.identity.tokens, impl.identity.hidden,
                  "Backward dX download elements"),
      "Backward dX download elements"));
  result.dgate.resize(
      checked_size(checked_mul(impl.identity.tokens, data.top_k,
                               "Backward dGate download elements"),
                   "Backward dGate download elements"));
  result.dw1.resize(checked_size(
      checked_mul(checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                              "Backward dW1 download rows"),
                  impl.identity.hidden, "Backward dW1 download elements"),
      "Backward dW1 download elements"));
  result.dw2.resize(checked_size(
      checked_mul(checked_mul(impl.owned_experts, impl.identity.hidden,
                              "Backward dW2 download rows"),
                  impl.intermediate, "Backward dW2 download elements"),
      "Backward dW2 download elements"));
  const auto copy = [&](void *host, const void *device, std::uint64_t bytes,
                        const char *name) {
    if (bytes == 0ULL)
      return;
    internal::throw_cuda(
        "backward-download",
        cudaMemcpy(host, device, checked_size(bytes, name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "backward",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.dx.data(), data.dx->data,
       result.dx.size() * sizeof(transport::BFloat16), "download dX");
  copy(result.dgate.data(), data.dgate->data,
       result.dgate.size() * sizeof(float), "download dGate");
  copy(result.dw1.data(), data.dw1->data,
       result.dw1.size() * sizeof(transport::BFloat16), "download dW1");
  copy(result.dw2.data(), data.dw2->data,
       result.dw2.size() * sizeof(transport::BFloat16), "download dW2");
  return result;
}
testing::GinPutSignalProbeResult
testing::run_distributed_gin_put_signal_probe(
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t payload_bytes) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)payload_bytes;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-put-probe, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-put-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(exec->device));
  constexpr std::size_t kAllocationBytes = 4096;
  constexpr std::size_t kReceiveOffset = 2048;
  constexpr std::size_t kBarrierOffset = 3072;
  constexpr std::size_t kErrorOffset = 4088;
  if (payload_bytes == 0 || payload_bytes > kReceiveOffset)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-put-probe",
                    "payload_bytes", nccl->rank, "1..2048",
                    std::to_string(payload_bytes));

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                             const ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-put-probe, field=symbol, rank=0, expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
      load("ncclDevCommDestroy"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
      load("ncclCommWindowDeregister"));
  const auto mem_alloc =
      reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto result = query_properties(nccl->comm, &properties);
  if (result != ncclSuccess || properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-put-probe, field=railed_gin, rank=" +
            std::to_string(nccl->rank) +
            ", expected=available, actual=" +
            (result == ncclSuccess
                 ? std::string("NCCL_GIN_TYPE_NONE")
                 : std::string(ncclGetErrorString(result))));
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0)
    transport_error(NotifyErrorCode::BackendUnavailable, "gin-put-probe",
                    "rail topology", nccl->rank, "rail.nRanks>=2, lsa>0",
                    "rail=" + std::to_string(rail.nRanks) + ", lsa=" +
                        std::to_string(lsa.nRanks));

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = 1;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 32;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };
  try {
    internal::throw_nccl(
        "gin-put-probe",
        dev_comm_create(nccl->comm, &requirements, &dev_comm), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclDevCommCreate(rail GIN)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-put-probe", mem_alloc(&allocation, kAllocationBytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN probe window)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemsetAsync(allocation, 0, kAllocationBytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN probe window)");
    std::vector<std::uint8_t> send(
        checked_size(payload_bytes, "GIN probe payload"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(nccl->rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpyAsync(allocation, send.data(), send.size(),
                        cudaMemcpyHostToDevice, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(GIN probe source)");
    internal::throw_cuda(
        "gin-put-probe", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(before GIN window register)");
    internal::throw_nccl(
        "gin-put-probe",
        window_register(nccl->comm, allocation, kAllocationBytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN probe)");
    gin_probe_reset_kernel<<<1, 1, 0, exec->stream>>>(dev_comm);
    auto *barrier = static_cast<std::uint64_t *>(allocation) +
                    kBarrierOffset / sizeof(std::uint64_t);
    internal::throw_nccl(
        "gin-put-probe",
        ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum, nccl->comm,
                      exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllReduce(after GIN reset)");
    gin_probe_put_signal_kernel<<<1, 1, 0, exec->stream>>>(
        dev_comm, window, checked_size(payload_bytes, "GIN probe payload"),
        kReceiveOffset,
        reinterpret_cast<unsigned int *>(
            static_cast<std::byte *>(allocation) + kErrorOffset));
    internal::throw_cuda(
        "gin-put-probe", cudaGetLastError(), nccl->rank, nccl->world,
        session.session().identity(), session.session().generation(),
        ExchangeKind::ExpertCounts, "GIN put/signal kernel launch");
    internal::throw_cuda(
        "gin-put-probe", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(GIN put/signal)");
    unsigned int device_error = 0;
    std::vector<std::uint8_t> received(send.size());
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(received.data(),
                   static_cast<std::byte *>(allocation) + kReceiveOffset,
                   received.size(), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN received payload)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(&device_error,
                   static_cast<std::byte *>(allocation) + kErrorOffset,
                   sizeof(device_error), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN device error)");
    if (device_error != 0)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "device_error", nccl->rank, "0",
                      std::to_string(device_error));
    const auto source_rail = (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_rank = source_rail * lsa.nRanks + lsa.rank;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(source_rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    testing::GinPutSignalProbeResult probe;
    probe.payload_bytes = payload_bytes;
    probe.rail_rank = static_cast<std::uint32_t>(rail.rank);
    probe.rail_world = static_cast<std::uint32_t>(rail.nRanks);
    probe.local_rank = static_cast<std::uint32_t>(lsa.rank);
    probe.received_hash = hash(received);
    probe.expected_hash = hash(expected);
    if (received != expected)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "received payload hash", nccl->rank,
                      std::to_string(probe.expected_hash),
                      std::to_string(probe.received_hash));
    cleanup();
    return probe;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

testing::GinLinkBenchmarkResult testing::run_distributed_gin_link_benchmark(
    const NcclSession &session, const CudaExecution &execution,
    GinConnectionMode mode, std::uint32_t target_shift,
    const std::vector<std::uint64_t> &payload_bytes, std::uint32_t warmup,
    std::uint32_t measured, std::uint32_t throughput_puts_per_sample) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)mode;
  (void)target_shift;
  (void)payload_bytes;
  (void)warmup;
  (void)measured;
  (void)throughput_puts_per_sample;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-link-benchmark, field=headers, rank=0, "
      "expected=nccl_device.h, actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "gin-link-benchmark", "device", nccl->rank,
                    std::to_string(nccl->device),
                    std::to_string(exec->device));
  if (payload_bytes.empty() || measured == 0U ||
      throughput_puts_per_sample == 0U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-link-benchmark", "benchmark shape", nccl->rank,
                    "non-empty payloads and positive samples/puts", "invalid");
  const auto maximum_payload =
      *std::max_element(payload_bytes.begin(), payload_bytes.end());
  if (maximum_payload == 0ULL)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-link-benchmark", "payload_bytes", nccl->rank,
                    "positive", "0");
  for (const auto bytes : payload_bytes)
    if (bytes == 0ULL || bytes > maximum_payload)
      transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                      "gin-link-benchmark", "payload_bytes", nccl->rank,
                      "1..max_payload", std::to_string(bytes));

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                             const ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-link-benchmark, field=symbol, rank=0, "
                      "expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
      load("ncclDevCommDestroy"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
      load("ncclCommWindowDeregister"));
  const auto mem_alloc = reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto nccl_result = query_properties(nccl->comm, &properties);
  if (nccl_result != ncclSuccess)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-link-benchmark, field=properties, rank=" +
            std::to_string(nccl->rank) + ", actual=" +
            ncclGetErrorString(nccl_result));
  const auto selected_type =
      mode == GinConnectionMode::Rail ? properties.railedGinType
                                      : properties.ginType;
  if (selected_type == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-link-benchmark, field=" +
            std::string(mode == GinConnectionMode::Rail ? "railedGinType"
                                                        : "ginType") +
            ", rank=" + std::to_string(nccl->rank) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");

  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0 ||
      nccl->world % static_cast<std::uint32_t>(lsa.nRanks) != 0U)
    transport_error(NotifyErrorCode::BackendUnavailable,
                    "gin-link-benchmark", "homogeneous topology", nccl->rank,
                    "at least two nodes and positive LSA", "rail=" +
                        std::to_string(rail.nRanks) + ",lsa=" +
                        std::to_string(lsa.nRanks));
  target_shift %= static_cast<std::uint32_t>(lsa.nRanks);
  const auto destination_node = (rail.rank + 1) % rail.nRanks;
  const auto destination_local =
      (lsa.rank + static_cast<int>(target_shift)) % lsa.nRanks;
  const auto destination =
      mode == GinConnectionMode::Rail
          ? destination_node
          : destination_node * lsa.nRanks + destination_local;
  const bool fanout_full = mode == GinConnectionMode::Full &&
                           std::getenv("WEFTMOE_GIN_BENCH_FANOUT") != nullptr;

  const auto align_4k = [](std::uint64_t value, const char *field) {
    return checked_add(value, 4095ULL, field) / 4096ULL * 4096ULL;
  };
  const auto receive_offset = align_4k(maximum_payload, "GIN receive offset");
  const auto barrier_offset = align_4k(
      checked_add(receive_offset, maximum_payload,
                  "GIN benchmark receive extent"),
      "GIN benchmark barrier offset");
  const auto allocation_bytes = checked_size(
      checked_add(barrier_offset, 4096ULL, "GIN benchmark window bytes"),
      "GIN benchmark window bytes");
  const auto duration_bytes = checked_size(
      checked_mul(measured, sizeof(unsigned long long),
                  "GIN benchmark duration bytes"),
      "GIN benchmark duration bytes");

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = 1;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 64;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType =
      mode == GinConnectionMode::Rail ? NCCL_GIN_CONNECTION_RAIL
                                      : NCCL_GIN_CONNECTION_FULL;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  unsigned long long *device_duration = nullptr;
  unsigned int *device_error = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (device_error != nullptr)
      (void)cudaFree(device_error);
    if (device_duration != nullptr)
      (void)cudaFree(device_duration);
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };

  try {
    internal::throw_nccl(
        "gin-link-benchmark",
        dev_comm_create(nccl->comm, &requirements, &dev_comm), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        mode == GinConnectionMode::Rail ? "ncclDevCommCreate(RAIL)"
                                        : "ncclDevCommCreate(FULL)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-link-benchmark", mem_alloc(&allocation, allocation_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN link benchmark)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaMalloc(&device_duration, duration_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN duration samples)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaMalloc(&device_error, sizeof(*device_error)),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN device error)");
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemsetAsync(allocation, 0, allocation_bytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN benchmark window)");
    std::vector<std::uint8_t> send(
        checked_size(maximum_payload, "GIN benchmark source"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          ((fanout_full ? 0ULL : static_cast<std::uint64_t>(nccl->rank)) *
               17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemcpyAsync(allocation, send.data(), send.size(),
                        cudaMemcpyHostToDevice, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(GIN benchmark source)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(before GIN benchmark registration)");
    internal::throw_nccl(
        "gin-link-benchmark",
        window_register(nccl->comm, allocation, allocation_bytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN link benchmark)");

    GinLinkBenchmarkResult benchmark;
    benchmark.mode = mode;
    benchmark.target_shift = target_shift;
    benchmark.rank = nccl->rank;
    benchmark.world = nccl->world;
    benchmark.local_rank = static_cast<std::uint32_t>(lsa.rank);
    benchmark.local_world = static_cast<std::uint32_t>(lsa.nRanks);
    benchmark.gin_type = static_cast<std::uint32_t>(properties.ginType);
    benchmark.railed_gin_type =
        static_cast<std::uint32_t>(properties.railedGinType);

    std::vector<std::uint32_t> batches{1U};
    if (throughput_puts_per_sample != 1U)
      batches.push_back(throughput_puts_per_sample);
    auto *barrier = reinterpret_cast<std::uint64_t *>(
        static_cast<std::byte *>(allocation) + barrier_offset);
    for (const auto bytes : payload_bytes) {
      for (const auto puts : batches) {
        gin_probe_reset_kernel<<<1, 1, 0, exec->stream>>>(dev_comm);
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemsetAsync(device_error, 0, sizeof(*device_error),
                            exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemsetAsync(GIN benchmark error)");
        internal::throw_nccl(
            "gin-link-benchmark",
            ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                          nccl->comm, exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "ncclAllReduce(after GIN benchmark reset)");
        gin_link_benchmark_kernel<<<1, 1, 0, exec->stream>>>(
            dev_comm, window, mode == GinConnectionMode::Full, fanout_full,
            destination, lsa.nRanks,
            checked_size(bytes, "GIN benchmark payload"),
            checked_size(receive_offset, "GIN benchmark receive offset"),
            warmup, measured, puts, device_duration, device_error);
        internal::throw_cuda(
            "gin-link-benchmark", cudaGetLastError(), nccl->rank, nccl->world,
            session.session().identity(), session.session().generation(),
            ExchangeKind::ExpertCounts, "GIN benchmark kernel launch");
        internal::throw_cuda(
            "gin-link-benchmark", cudaStreamSynchronize(exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaStreamSynchronize(GIN link benchmark)");
        unsigned int host_error = 0U;
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemcpy(&host_error, device_error, sizeof(host_error),
                       cudaMemcpyDeviceToHost),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemcpy(GIN benchmark error)");
        if (host_error != 0U)
          transport_error(NotifyErrorCode::NcclFailure,
                          "gin-link-benchmark", "device_error", nccl->rank,
                          "0", std::to_string(host_error));
        GinLinkBenchmarkCaseResult sample;
        sample.payload_bytes = bytes;
        sample.puts_per_sample = puts;
        sample.duration_ns.resize(measured);
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemcpy(sample.duration_ns.data(), device_duration,
                       duration_bytes, cudaMemcpyDeviceToHost),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemcpy(GIN benchmark durations)");
        benchmark.cases.push_back(std::move(sample));
      }
    }

    std::vector<std::uint8_t> received(send.size());
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemcpy(received.data(),
                   static_cast<std::byte *>(allocation) + receive_offset,
                   received.size(), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN benchmark received payload)");
    const auto source_node =
        (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_local =
        mode == GinConnectionMode::Rail
            ? lsa.rank
            : (lsa.rank + lsa.nRanks - static_cast<int>(target_shift)) %
                  lsa.nRanks;
    const auto source_rank = source_node * lsa.nRanks + source_local;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          ((fanout_full ? 0ULL : static_cast<std::uint64_t>(source_rank)) *
               17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    benchmark.received_hash = hash(received);
    benchmark.expected_hash = hash(expected);
    if (received != expected)
      transport_error(NotifyErrorCode::NcclFailure,
                      "gin-link-benchmark", "received payload hash",
                      nccl->rank, std::to_string(benchmark.expected_hash),
                      std::to_string(benchmark.received_hash));
    cleanup();
    return benchmark;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

ActivationTransportCapability probe_activation_transport_capability(
    std::uint32_t world_size) {
  ActivationTransportCapability result;
  result.compiled_nccl_version = NCCL_VERSION_CODE;
  int runtime = 0;
  if (ncclGetVersion(&runtime) == ncclSuccess)
    result.runtime_nccl_version = static_cast<std::uint32_t>(runtime);
#if __has_include(<nccl_device.h>)
  result.nccl_device_headers = true;
#endif
#if __has_include(<nccl_device/gin.h>)
  result.gin_headers_available = true;
#endif
  // Device put/signal/wait/flush are header-inline APIs. Their required host
  // lifecycle surface must all exist in the actually loaded runtime.
  constexpr const char *required_symbols[] = {
      "ncclCommQueryProperties",
      "ncclDevCommCreate",
      "ncclDevCommDestroy",
      "ncclMemAlloc",
      "ncclMemFree",
      "ncclCommWindowRegister",
      "ncclCommWindowDeregister",
      "ncclGetPeerDevicePointer",
      "ncclTeamRail",
      "ncclGinOutboxCreateRequirement",
      "ncclGinInboxA2ACreateRequirement",
      "ncclGinBarrierCreateRequirement",
  };
  result.gin_symbols_present = result.gin_headers_available;
  std::string first_missing;
  for (const auto *symbol : required_symbols) {
    if (dlsym(RTLD_DEFAULT, symbol) == nullptr) {
      result.gin_symbols_present = false;
      if (first_missing.empty())
        first_missing = symbol;
    }
  }
  result.gin_backend_implemented = true;

#if __has_include(<nccl_device/core.h>)
  if (result.gin_headers_available && result.gin_symbols_present) {
    using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                                ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                               ncclWindow_t *, int);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
    using MemFreeFn = ncclResult_t (*)(void *);
    const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
        dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
    const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommCreate"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto window_register = reinterpret_cast<WindowRegisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowRegister"));
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto mem_alloc = reinterpret_cast<MemAllocFn>(
        dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free = reinterpret_cast<MemFreeFn>(
        dlsym(RTLD_DEFAULT, "ncclMemFree"));
    ncclComm_t probe_comm = nullptr;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    void *allocation = nullptr;
    bool dev_comm_created = false;
    std::string init_error;
    ncclUniqueId id{};
    auto nccl_result = ncclGetUniqueId(&id);
    if (nccl_result == ncclSuccess)
      nccl_result = ncclCommInitRank(&probe_comm, 1, id, 0);
    if (nccl_result != ncclSuccess) {
      init_error = std::string("ncclCommInitRank: ") +
                   ncclGetErrorString(nccl_result);
    } else {
      ncclCommProperties properties{};
      properties.size = sizeof(properties);
      properties.magic = NCCL_API_MAGIC;
      properties.version = NCCL_VERSION_CODE;
      nccl_result = query_properties(probe_comm, &properties);
      if (nccl_result != ncclSuccess) {
        init_error = std::string("ncclCommQueryProperties: ") +
                     ncclGetErrorString(nccl_result);
      } else if (properties.ginType == NCCL_GIN_TYPE_NONE &&
                 properties.railedGinType == NCCL_GIN_TYPE_NONE) {
        init_error = "ncclCommQueryProperties: no GIN context type";
      } else {
        ncclDevCommRequirements_t requirements{};
        requirements.size = sizeof(requirements);
        requirements.magic = NCCL_API_MAGIC;
        requirements.version = NCCL_VERSION_CODE;
        requirements.ginContextCount = 1;
        requirements.ginExclusiveContexts = true;
        requirements.ginQueueDepth = 32;
        requirements.ginSignalCount = 4;
        requirements.ginConnectionType =
            properties.railedGinType != NCCL_GIN_TYPE_NONE
                ? NCCL_GIN_CONNECTION_RAIL
                : NCCL_GIN_CONNECTION_FULL;
        nccl_result = dev_comm_create(probe_comm, &requirements, &dev_comm);
        if (nccl_result != ncclSuccess) {
          init_error = std::string("ncclDevCommCreate(GIN): ") +
                       ncclGetErrorString(nccl_result);
        } else {
          dev_comm_created = true;
          nccl_result = mem_alloc(&allocation, 4096);
          if (nccl_result != ncclSuccess) {
            init_error = std::string("ncclMemAlloc(window): ") +
                         ncclGetErrorString(nccl_result);
          } else {
            nccl_result = window_register(
                probe_comm, allocation, 4096, &window,
                NCCL_WIN_STRICT_ORDERING);
            if (nccl_result != ncclSuccess)
              init_error = std::string("ncclCommWindowRegister: ") +
                           ncclGetErrorString(nccl_result);
            else
              result.gin_runtime_initializable = true;
          }
        }
      }
    }
    if (window)
      (void)window_deregister(probe_comm, window);
    if (allocation)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(probe_comm, &dev_comm);
    if (probe_comm)
      (void)ncclCommDestroy(probe_comm);
    if (!result.gin_runtime_initializable)
      result.gin_unavailable_reason = std::move(init_error);
  }
#endif
  result.gin_usable = result.gin_headers_available &&
                      result.gin_symbols_present &&
                      result.gin_runtime_initializable &&
                      result.gin_backend_implemented;
  if (!result.gin_usable) {
    std::ostringstream message;
    message << "GIN unavailable: headers=" << result.gin_headers_available
            << ", symbols=" << result.gin_symbols_present
            << ", runtime_initializable="
            << result.gin_runtime_initializable
            << ", backend_implemented=" << result.gin_backend_implemented
            << ", compiled_nccl=" << result.compiled_nccl_version
            << ", runtime_nccl=" << result.runtime_nccl_version;
    if (!first_missing.empty())
      message << ", first_missing_symbol=" << first_missing;
    if (!result.gin_unavailable_reason.empty())
      message << ", initialization_error=" << result.gin_unavailable_reason;
    result.gin_unavailable_reason = message.str();
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess)
    devices = 0;
  const auto checked_world = std::min<std::uint32_t>(
      world_size, static_cast<std::uint32_t>(std::max(devices, 0)));
  result.peer_access_matrix.resize(
      static_cast<std::size_t>(checked_world) * checked_world, 0);
  for (std::uint32_t source = 0; source < checked_world; ++source) {
    for (std::uint32_t destination = 0; destination < checked_world;
         ++destination) {
      int access = source == destination ? 1 : 0;
      if (source != destination)
        (void)cudaDeviceCanAccessPeer(&access, static_cast<int>(source),
                                      static_cast<int>(destination));
      result.peer_access_matrix[static_cast<std::size_t>(source) *
                                    checked_world +
                                destination] =
          static_cast<std::uint8_t>(access != 0);
    }
  }
  if (devices > 0) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
      result.system_scope_atomics = properties.major >= 6;
  }
  return result;
}

ActivationTransportCapability probe_activation_transport_capability(
    const NcclSession &session, const CudaExecution &execution) {
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto cuda = internal::checked_execution_state(execution);
  if (nccl->device != cuda->device)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-distributed-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(cuda->device));

  auto result = probe_activation_transport_capability(nccl->world);
  // The world=1 probe above is useful for loader diagnostics but cannot prove
  // that a rail GIN team can be formed.  Replace only its initialization
  // result with the collective probe below.
  result.gin_runtime_initializable = false;
  result.gin_backend_implemented = true;
  result.gin_usable = false;
  result.gin_unavailable_reason.clear();

#if __has_include(<nccl_device/core.h>)
  if (result.gin_headers_available && result.gin_symbols_present) {
    using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                                ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                               ncclWindow_t *, int);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
    using MemFreeFn = ncclResult_t (*)(void *);
    const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
        dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
    const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommCreate"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto window_register = reinterpret_cast<WindowRegisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowRegister"));
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto mem_alloc = reinterpret_cast<MemAllocFn>(
        dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free = reinterpret_cast<MemFreeFn>(
        dlsym(RTLD_DEFAULT, "ncclMemFree"));

    ncclCommProperties_t properties{};
    properties.size = sizeof(properties);
    properties.magic = NCCL_API_MAGIC;
    properties.version = NCCL_VERSION_CODE;
    ncclDevCommRequirements_t requirements{};
    requirements.size = sizeof(requirements);
    requirements.magic = NCCL_API_MAGIC;
    requirements.version = NCCL_VERSION_CODE;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    void *allocation = nullptr;
    bool dev_comm_created = false;
    std::string error;

    auto nccl_result = query_properties(nccl->comm, &properties);
    if (nccl_result != ncclSuccess) {
      error = std::string("ncclCommQueryProperties: ") +
              ncclGetErrorString(nccl_result);
    } else if (properties.ginType == NCCL_GIN_TYPE_NONE &&
               properties.railedGinType == NCCL_GIN_TYPE_NONE) {
      error = "ncclCommQueryProperties: no GIN context type";
    } else {
      requirements.ginContextCount = 1;
      requirements.ginExclusiveContexts = true;
      requirements.ginQueueDepth = 32;
      requirements.ginSignalCount = 4;
      requirements.ginConnectionType =
          properties.railedGinType != NCCL_GIN_TYPE_NONE
              ? NCCL_GIN_CONNECTION_RAIL
              : NCCL_GIN_CONNECTION_FULL;
      nccl_result = dev_comm_create(nccl->comm, &requirements, &dev_comm);
      if (nccl_result != ncclSuccess) {
        error = std::string("ncclDevCommCreate(GIN): ") +
                ncclGetErrorString(nccl_result);
      } else {
        dev_comm_created = true;
        nccl_result = mem_alloc(&allocation, 4096);
        if (nccl_result != ncclSuccess) {
          error = std::string("ncclMemAlloc(window): ") +
                  ncclGetErrorString(nccl_result);
        } else {
          nccl_result = window_register(nccl->comm, allocation, 4096, &window,
                                        NCCL_WIN_STRICT_ORDERING);
          if (nccl_result != ncclSuccess)
            error = std::string("ncclCommWindowRegister: ") +
                    ncclGetErrorString(nccl_result);
          else
            result.gin_runtime_initializable = true;
        }
      }
    }
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
    if (!result.gin_runtime_initializable)
      result.gin_unavailable_reason = std::move(error);
  }
#endif

  result.gin_usable = result.gin_headers_available &&
                      result.gin_symbols_present &&
                      result.gin_runtime_initializable &&
                      result.gin_backend_implemented;

  std::ostringstream message;
  message << "GIN distributed probe: headers="
          << result.gin_headers_available << ", symbols="
          << result.gin_symbols_present << ", runtime_initializable="
          << result.gin_runtime_initializable << ", backend_implemented="
          << result.gin_backend_implemented << ", rank=" << nccl->rank
          << ", world=" << nccl->world << ", compiled_nccl="
          << result.compiled_nccl_version << ", runtime_nccl="
          << result.runtime_nccl_version;
  if (!result.gin_unavailable_reason.empty())
    message << ", initialization_error=" << result.gin_unavailable_reason;
  result.gin_unavailable_reason = message.str();
  return result;
}

bool testing::run_local_p2p_put_signal_probe(std::uint32_t source_device,
                                             std::uint32_t destination_device,
                                             std::uint64_t elements) {
  if (elements == 0)
    return false;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return false;
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
  bool ok = false;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(checked_mul(
                                   elements, sizeof(std::uint64_t),
                                   "peer destination bytes"),
                               "peer destination bytes")) != cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemset(signal, 0, sizeof(*signal));
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable = cudaDeviceEnablePeerAccess(
        static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(checked_mul(
                              elements, sizeof(std::uint64_t),
                              "peer source bytes"),
                          "peer source bytes")) != cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  if (cudaMemsetAsync(source, 0x5a,
                      checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                               "peer copy bytes"),
                                   "peer copy bytes"),
                      source_stream) != cudaSuccess ||
      cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
  const auto blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
  const auto run = [&](bool early_signal, unsigned int &observed_errors) {
    (void)cudaSetDevice(static_cast<int>(destination_device));
    if (cudaMemsetAsync(destination, 0,
                        checked_size(checked_mul(elements,
                                                 sizeof(std::uint64_t),
                                                 "peer reset bytes"),
                                     "peer reset bytes"),
                        destination_stream) != cudaSuccess ||
        cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
            cudaSuccess ||
        cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(destination_stream) != cudaSuccess)
      return false;
    peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
        destination, signal, errors, elements, expected);
    if (cudaGetLastError() != cudaSuccess)
      return false;
    (void)cudaSetDevice(static_cast<int>(source_device));
    if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                        source_stream) != cudaSuccess ||
        cudaStreamSynchronize(source_stream) != cudaSuccess)
      return false;
    peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
        source, destination, signal, completed_blocks, elements, early_signal);
    if (cudaGetLastError() != cudaSuccess)
      return false;
    // The destination independently observes the signal and validates every
    // word before the source stream is synchronized.
    (void)cudaSetDevice(static_cast<int>(destination_device));
    if (cudaStreamSynchronize(destination_stream) != cudaSuccess ||
        cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    (void)cudaSetDevice(static_cast<int>(source_device));
    return cudaStreamSynchronize(source_stream) == cudaSuccess;
  };
  {
    unsigned int positive_errors = 0;
    unsigned int early_signal_errors = 0;
    ok = run(false, positive_errors) && positive_errors == 0U &&
         run(true, early_signal_errors) && early_signal_errors != 0U;
  }
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return ok;
}

double testing::measure_local_p2p_put_bandwidth_gbps(
    std::uint32_t source_device, std::uint32_t destination_device,
    std::uint64_t bytes, std::uint32_t warmup, std::uint32_t measured) {
  if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0 || measured == 0)
    return 0.0;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return 0.0;
  const auto elements = bytes / sizeof(std::uint64_t);
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  double result = 0.0;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable = cudaDeviceEnablePeerAccess(
        static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemsetAsync(source, 0x5a, checked_size(bytes, "P2P bandwidth bytes"),
                        source_stream);
  if (cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
    constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
    const auto blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
    double measured_seconds = 0.0;
    for (std::uint32_t iteration = 0; iteration < warmup + measured;
         ++iteration) {
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
              cudaSuccess ||
          cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
              cudaSuccess ||
          cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
          destination, signal, errors, elements, expected);
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                          source_stream) != cudaSuccess ||
          cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      const auto start = std::chrono::steady_clock::now();
      peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
          source, destination, signal, completed_blocks, elements, false);
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      const auto finish = std::chrono::steady_clock::now();
      unsigned int observed_errors = 0;
      if (cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          observed_errors != 0U)
        goto cleanup;
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      if (iteration >= warmup)
        measured_seconds +=
            std::chrono::duration<double>(finish - start).count();
    }
    if (measured_seconds > 0.0)
      result = static_cast<double>(bytes) * measured /
               measured_seconds / 1.0e9;
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return result;
}

bool backward_pipeline_available() noexcept { return true; }

} // namespace weftmoe::cuda
