#include "cuda/activation_transport_internal.hpp"
#include "cuda/communication/chunk_ring.cuh"
#include "cuda/communication/local_notification.cuh"
#include "cuda/communication/nccl_device.cuh"
#include "cuda/communication/return_layout.hpp"
#include "cuda/primitives/address_range.cuh"
#include "cuda/primitives/hopper.cuh"
#include "cuda/primitives/programmatic_launch.cuh"
#include "cuda/fp8/tile.cuh"
#include "cuda/fp8/column.cuh"
#include "cuda/fp8/source.cuh"

#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

#include "cuda/backward/pipeline_state.cuh"
#include "cuda/backward/scheduler.cuh"
#include "cuda/backward/tuning_config.hpp"
#include "cuda/cuda_internal.hpp"
#include "cuda/device_communicator.cuh"
#include "cuda/pipeline/atomics.cuh"
#include "cuda/pipeline/contracts.hpp"
#include "cuda/profiling/device_timeline.cuh"
#include "cuda/transport_generation_completion.cuh"
#include "weftmoe/api/errors.hpp"

#ifndef WEFTMOE_FORWARD_LIGHT_PROGRESS
#define WEFTMOE_FORWARD_LIGHT_PROGRESS 0
#endif

namespace weftmoe::cuda {
#ifndef WEFTMOE_FORWARD_TUNING_ID
#define WEFTMOE_FORWARD_TUNING_ID "untuned"
#endif
const char *forward_tuning_id() { return WEFTMOE_FORWARD_TUNING_ID; }

struct ActivationMegaKernelBackendState;
namespace {

#include "cuda/forward/kernels/primitives.cuh"

#include "cuda/weftmoe_dt/kernel.cuh"
#include "cuda/weftmoe_pf/local_kernel.cuh"
#include "cuda/weftmoe_pf/network_kernel.cuh"
#include "cuda/weftmoe_pf/split_notify.cuh"

#include "cuda/backward/device_kernels.cuh"
#include "cuda/weftmoe_pf/backward_kernel.cuh"

#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
__global__ void fill_transport_benchmark_output(transport::BFloat16 *output,
                                                unsigned long long count) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    output[i] =
        transport::BFloat16{static_cast<std::uint16_t>(0x3c00U + (i & 255U))};
}
#endif

} // namespace

#include "cuda/tensor_storage.cuh"
#include "cuda/tensor_preparation.cuh"

std::pair<std::uint32_t, std::uint32_t> forward_shape() {
  return {forward_tuning::kMainHidden, forward_tuning::kMainIntermediate};
}

namespace {
#include "cuda/backward/peer_workspace.cuh"
}

struct ActivationTransportResourcePool::Impl {
  std::shared_ptr<ActivationPoolState> state;
};

struct InputTensorAllocation;

struct SelectedActivationLease<transport::BFloat16>::Impl {
  std::shared_ptr<InputTensorAllocation> allocation;
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  const transport::BFloat16 *pointer = nullptr;
  FP8SourceBuffers fp8;
  bool prequantized = false;
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
  FP8SourceBuffers fp8;
  bool prequantized = false;
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
    const UnifiedRegisterKey key{kernel, Resources::kUnifiedThreads,
                                 Resources::kUnifiedMathRegisters,
                                 Resources::kUnifiedProducerRegisters,
                                 Resources::kUnifiedControlRegisters};
    std::lock_guard<std::mutex> lock(unified_register_mutex);
    if (std::find(validated_unified_register_kernels.begin(),
                  validated_unified_register_kernels.end(),
                  key) != validated_unified_register_kernels.end())
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
    constexpr unsigned requested = 256U * Resources::kUnifiedMathRegisters +
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
  // Avoid repeating these driver queries on every Forward/Backward invocation.
  template <class Resources = void>
  cudaError_t
  prepare_unified_launch(const void *kernel, std::size_t dynamic_smem,
                         unsigned grid_blocks, unsigned block_threads,
                         unsigned cluster_size, bool programmatic,
                         bool cooperative_launch, int *max_active_clusters) {
    // Only entries using setmaxnreg need a dynamic quota audit. Ordinary
    // register allocation is checked by the occupancy query below.
    auto error = cudaSuccess;
    if constexpr (!std::is_void_v<Resources>) {
      error = validate_unified_register_pool<Resources>(kernel);
      if (error != cudaSuccess)
        return error;
    }
    const UnifiedLaunchKey key{
        kernel,       dynamic_smem, grid_blocks,       block_threads,
        cluster_size, programmatic, cooperative_launch};
    std::lock_guard<std::mutex> lock(unified_launch_mutex);
    const auto found = std::find_if(prepared_unified_launches.begin(),
                                    prepared_unified_launches.end(),
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
    error = cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
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
  auto *send =
      at<ProductionStatusRecord>(slot.device, layout.production_error_records) +
      rank;
  auto *receive =
      at<unsigned long long>(slot.device, layout.production_status_matrix);
  internal::throw_cuda(
      "production-terminal-status",
      cudaMemcpyAsync(send, host_records + rank, sizeof(ProductionStatusRecord),
                      cudaMemcpyHostToDevice, exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "H2D local production status record");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(ProductionStatusRecord));
  internal::throw_nccl("production-terminal-status",
                       ncclAllGather(send, receive, 8, ncclUint64,
                                     backend->nccl->comm, exec->stream),
                       rank, world, backend->session.session().identity(),
                       generation, ExchangeKind::ExpertCounts,
                       "ncclAllGather(ProductionStatusRecord)");
  const auto matrix_bytes =
      checked_size(checked_mul(world, sizeof(ProductionStatusRecord),
                               "production terminal status matrix"),
                   "production terminal status matrix");
  internal::throw_cuda("production-terminal-status",
                       cudaMemcpyAsync(host_records, receive, matrix_bytes,
                                       cudaMemcpyDeviceToHost, exec->stream),
                       rank, world, backend->session.session().identity(),
                       generation, ExchangeKind::ExpertCounts,
                       "D2H production status matrix");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            matrix_bytes);
  internal::throw_cuda("production-terminal-status",
                       cudaEventRecord(completion_event, exec->stream), rank,
                       world, backend->session.session().identity(), generation,
                       ExchangeKind::ExpertCounts,
                       "cudaEventRecord(production terminal status)");
  internal::throw_cuda("production-terminal-status",
                       cudaEventSynchronize(completion_event), rank, world,
                       backend->session.session().identity(), generation,
                       ExchangeKind::ExpertCounts,
                       "cudaEventSynchronize(production terminal status)");

  ProductionStatusAgreementResult result;
  const auto expected_phase = static_cast<unsigned long long>(phase);
  const auto malformed = [&](std::uint32_t origin, std::uint64_t field,
                             std::uint64_t expected, std::uint64_t actual) {
    result.has_error = true;
    result.error =
        ProductionStatusRecord{generation, expected_phase, origin, 2ULL,
                               field,      expected,       actual, epoch};
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
    if (w1 && owns_w1) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w1);
    }
  }
};

struct ForwardWeightLease::Impl {
  std::shared_ptr<ForwardAllocation> allocation;
  FP8WeightBuffers fp8_buffers;
  fp8::Weights fp8;
  const transport::BFloat16 *w1 = nullptr;
  const transport::BFloat16 *w2 = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken execution;
  std::uint32_t owned_experts = 0;
  std::uint32_t global_expert_begin = 0;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint32_t device = 0;
  CUtensorMap fc1_w1_map{};
  CUtensorMap w2_map{};
  CUtensorMap decode_w1_map{};
  CUtensorMap decode_w2_map{};
  CUtensorMap backward_w1_map{};
  CUtensorMap backward_w2_map{};

  void initialize_decode_maps() {
    decode_w1_map = make_gate_up_w1_tma_map(
        const_cast<transport::BFloat16 *>(w1), hidden, intermediate,
        owned_experts, weftmoe_dt::kN, weftmoe_dt::kK, "decode gate/up TMA W1");
    decode_w2_map = make_bf16_tma_map(
        const_cast<transport::BFloat16 *>(w2), intermediate,
        owned_experts * hidden, weftmoe_dt::kN, "decode W2", weftmoe_dt::kK);
  }
};

struct ForwardWeightStorageHashAccess {
  static const ForwardWeightLease::Impl &
  get(const ForwardWeightLease &weights) {
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
                      "slot", 0, "wait for the previous operation",
                      "operation pending");
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
  std::shared_ptr<TrainingCompletionPool::Entry> record;
  bool ready() const { return record ? record->ready() : !finish; }
  void wait() {
    std::lock_guard<std::mutex> lock(mutex);
    if (finish) {
      auto action = std::move(finish);
      try {
        action();
      } catch (...) {
        error = std::current_exception();
      }
    }
    record.reset();
    if (error)
      std::rethrow_exception(error);
  }
  ~OperatorCompletion() noexcept {
    try {
      wait();
    } catch (...) {
    }
  }
};

struct DeviceForwardHandle::Impl {
  ForwardKernelArgs pf_args{};
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
  internal::throw_cuda("device-timeline", cudaEventSynchronize(completion),
                       identity.rank, identity.world, "timeline",
                       identity.generation, ExchangeKind::ExpertCounts,
                       "wait timeline producer completion");
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
    const auto begin = raw.begin() + static_cast<std::ptrdiff_t>(cta) *
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

profiling::DeviceTimelineSnapshot
testing::download_forward_device_timeline(const DeviceForwardHandle &handle) {
  if (!handle.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "device-timeline",
                    "Forward handle", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  const auto &slot = impl.pool->slots[impl.workspace_index];
  return download_device_timeline(
      impl.pool, impl.workspace_index, slot.forward_timeline, slot.complete,
      impl.identity, profiling::TimelinePhase::Forward);
}
#endif

std::uint32_t ForwardWeightLease::owned_experts() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights", "handle",
                    0, "valid", "empty");
  return impl_->owned_experts;
}
std::uint64_t ForwardWeightLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights", "handle",
                    0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t ForwardWeightLease::intermediate_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights", "handle",
                    0, "valid", "empty");
  return impl_->intermediate;
}
std::uint64_t ForwardWeightLease::device_bytes() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights", "handle",
                    0, "valid", "empty");
  return impl_->allocation->bytes;
}
std::uint32_t ForwardWeightLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights", "handle",
                    0, "valid", "empty");
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
  const auto w1_rows =
      checked_mul(checked_mul(owned_experts, 2ULL, "W1 paired expert rows"),
                  intermediate_size, "W1 expert rows");
  const auto w1_elements = checked_mul(w1_rows, hidden_size, "W1 elements");
  const auto w2_elements =
      checked_mul(checked_mul(owned_experts, hidden_size, "W2 expert rows"),
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
        cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming), 0,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaEventCreate(weight ready)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    initialize_forward_weights_kernel<<<4096, 256, 0, state->stream>>>(
        static_cast<transport::BFloat16 *>(allocation->w1),
        w1_elements, static_cast<transport::BFloat16 *>(allocation->w2),
        w2_elements, global_expert_begin, hidden_size, intermediate_size);
    internal::throw_cuda("forward-weight-create", cudaGetLastError(), 0,
                         session.world_size(), session.identity(),
                         session.generation(), ExchangeKind::ExpertCounts,
                         "initialize_forward_weights_kernel");
    internal::throw_cuda("forward-weight-create",
                         cudaEventRecord(allocation->ready, state->stream), 0,
                         session.world_size(), session.identity(),
                         session.generation(), ExchangeKind::ExpertCounts,
                         "cudaEventRecord(weight ready)");
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
  if (hidden_size == forward_tuning::kMainHidden &&
      intermediate_size == forward_tuning::kMainIntermediate) {
    impl->fc1_w1_map = make_gate_up_w1_tma_map(
        impl->allocation->w1, hidden_size, intermediate_size, owned_experts,
        forward_tuning::kFc1TileN, forward_tuning::kTileK, "gate/up TMA W1");
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        forward_tuning::kFc2TileN, "W2", forward_tuning::kTileK);
    impl->initialize_decode_maps();
    impl->backward_w1_map = make_bf16_tma_map_mn(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        "Backward W1 MN-major", backward_tuning::kActivationTileK);
    impl->backward_w2_map = make_bf16_tma_map_mn(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        "Backward W2 MN-major", backward_tuning::kActivationTileK);
  }
  return ForwardWeightLease(std::move(impl));
}

ForwardWeightLease ExternalCudaProducer::publish_fp8_weights(
    const FP8WeightBuffers &buffers, std::uint32_t owned_experts,
    std::uint32_t global_expert_begin, std::uint64_t hidden_size,
    std::uint64_t intermediate_size) const {
  if (!state_ || !owned_experts || !hidden_size || !intermediate_size ||
      hidden_size % 128U || intermediate_size % 128U)
    throw std::invalid_argument("FP8 weights require nonempty experts and H/I multiples of 128");
  const auto w2_bytes = checked_mul(checked_mul(owned_experts, hidden_size,
      "FP8 weight rows"), intermediate_size, "FP8 weight bytes");
  const auto w1_bytes = checked_mul(w2_bytes, 2U, "FP8 W1 bytes");
  auto impl = std::make_shared<ForwardWeightLease::Impl>();
  auto allocation = std::make_shared<ForwardAllocation>();
  allocation->device = state_->backend->device;
  allocation->external_producer = state_->backend;
  const auto validate = [&](const CudaTensorBuffer &buffer, std::uint64_t bytes,
                            std::uint64_t alignment = 1U) {
    if (!buffer.owner)
      throw std::invalid_argument("FP8 weights and scales require allocation owners");
    if (reinterpret_cast<std::uintptr_t>(buffer.data) % alignment)
      throw std::invalid_argument("FP8 weights/scales have invalid pointer alignment");
    validate_external_cuda_range(buffer.data, bytes, buffer.bytes, 1U,
                                 allocation->device, "FP8 weights/scales");
    allocation->bytes = checked_add(allocation->bytes, bytes, "FP8 weight allocation");
  };
  validate(buffers.w1, w1_bytes);
  validate(buffers.s1, w1_bytes / (128U * 128U) * sizeof(float), alignof(float));
  validate(buffers.w2, w2_bytes);
  validate(buffers.s2, w2_bytes / (128U * 128U) * sizeof(float), alignof(float));
  const bool transposed = buffers.w1_t.data || buffers.s1_t.data ||
                          buffers.w2_t.data || buffers.s2_t.data;
  if (transposed) {
    validate(buffers.w1_t, w1_bytes);
    validate(buffers.s1_t, w1_bytes / (128U * 128U) * sizeof(float), alignof(float));
    validate(buffers.w2_t, w2_bytes);
    validate(buffers.s2_t, w2_bytes / (128U * 128U) * sizeof(float), alignof(float));
  }
  const auto &context = state_->backend->context;
  internal::throw_cuda("fp8-weight-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "create FP8 weight readiness");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("fp8-weight-publish",
      internal::record_input_ready(allocation->ready, state_->backend->stream,
                                  allocation->ready_capture_id),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "record FP8 weight readiness");
  impl->allocation = std::move(allocation);
  impl->fp8_buffers = buffers;
  impl->fp8 = {static_cast<const std::uint8_t *>(buffers.w1.data),
               static_cast<const std::uint8_t *>(buffers.w2.data),
               static_cast<const float *>(buffers.s1.data),
               static_cast<const float *>(buffers.s2.data),
               static_cast<const std::uint8_t *>(buffers.w1_t.data),
               static_cast<const std::uint8_t *>(buffers.w2_t.data),
               static_cast<const float *>(buffers.s1_t.data),
               static_cast<const float *>(buffers.s2_t.data)};
  impl->session_identity = context.session().stable_identity_bytes();
  impl->execution = state_->execution;
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state_->backend->device;
  impl->fc1_w1_map = make_gate_up_w1_tma_map(
      buffers.w1.data, hidden_size, intermediate_size, owned_experts,
      forward_tuning::kFc1TileN, 128U, "FP8 W1", true);
  impl->w2_map = make_fp8_tma_map(buffers.w2.data, intermediate_size,
      std::uint64_t(owned_experts) * hidden_size, forward_tuning::kFc2TileN, "FP8 W2");
  impl->decode_w1_map = make_gate_up_w1_tma_map(buffers.w1.data, hidden_size, intermediate_size,
      owned_experts, 256U, 128U, "DT FP8 W1", true);
  impl->decode_w2_map = make_fp8_tma_map(buffers.w2.data, intermediate_size,
      std::uint64_t(owned_experts) * hidden_size, 256U, "DT FP8 W2");
  if (transposed) {
    impl->backward_w1_map = make_fp8_tma_map(buffers.w1_t.data, 2U * intermediate_size,
        std::uint64_t(owned_experts) * hidden_size, backward_tuning::kDxTileN, "FP8 W1 transposed");
    impl->backward_w2_map = make_fp8_tma_map(buffers.w2_t.data, hidden_size,
        std::uint64_t(owned_experts) * intermediate_size, backward_tuning::kDabTileN, "FP8 W2 transposed");
  }
  return ForwardWeightLease(std::move(impl));
}

ForwardWeightLease ExternalCudaProducer::publish_weights(
    const transport::BFloat16 *w1, std::uint64_t w1_capacity_elements,
    std::shared_ptr<const void> w1_owner, const transport::BFloat16 *w2,
    std::uint64_t w2_capacity_elements, std::shared_ptr<const void> w2_owner,
    std::uint32_t owned_experts, std::uint32_t global_expert_begin,
    std::uint64_t hidden_size, std::uint64_t intermediate_size) const {
  if (!state_ || !w1_owner || !w2_owner || owned_experts == 0 ||
      hidden_size == 0 || intermediate_size == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "external-weight-publish", "producer/owner/shape", 0,
                    "valid and nonzero", "invalid");
  const auto w1_elements = checked_mul(
      checked_mul(owned_experts, 2ULL * intermediate_size, "external W1 rows"),
      hidden_size, "external W1 elements");
  const auto w2_elements =
      checked_mul(checked_mul(owned_experts, hidden_size, "external W2 rows"),
                  intermediate_size, "external W2 elements");
  validate_external_cuda_range(w1, w1_elements, w1_capacity_elements,
                               sizeof(transport::BFloat16),
                               state_->backend->device, "W1");
  validate_external_cuda_range(w2, w2_elements, w2_capacity_elements,
                               sizeof(transport::BFloat16),
                               state_->backend->device, "W2");
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
  allocation->bytes = checked_add(
      w1_bytes,
      checked_mul(w2_elements, sizeof(transport::BFloat16), "W2 bytes"),
      "weight bytes");
  internal::throw_cuda(
      "external-weight-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventCreate(weight ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
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
  impl->w2 = w2;
  impl->session_identity =
      state_->backend->context.session().stable_identity_bytes();
  impl->execution = state_->execution;
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state_->backend->device;
  if (hidden_size == forward_tuning::kMainHidden &&
      intermediate_size == forward_tuning::kMainIntermediate) {
    impl->fc1_w1_map = make_gate_up_w1_tma_map(
        impl->allocation->w1, hidden_size, intermediate_size, owned_experts,
        forward_tuning::kFc1TileN, forward_tuning::kTileK, "gate/up TMA W1");
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        forward_tuning::kFc2TileN, "W2", forward_tuning::kTileK);
    impl->initialize_decode_maps();
    impl->backward_w1_map = make_bf16_tma_map_mn(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        "Backward W1 MN-major", backward_tuning::kActivationTileK);
    impl->backward_w2_map = make_bf16_tma_map_mn(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size,
        "Backward W2 MN-major", backward_tuning::kActivationTileK);
  }
  return ForwardWeightLease(std::move(impl));
}

ForwardWeightStorageHashes testing::hash_forward_weight_storage_for_validation(
    const ForwardWeightLease &weights, const CudaExecution &execution) {
  if (!weights || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weight-hash",
                    "weights/execution", 0, "valid", "empty");
  const auto &impl = ForwardWeightStorageHashAccess::get(weights);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-weight-hash",
                    "device", 0, std::to_string(impl.device),
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
        (elements + kWeightHashChunkElements - 1ULL) / kWeightHashChunkElements;
    const auto chunk_bytes =
        checked_size(checked_mul(chunk_count, sizeof(WeightHashPair),
                                 "weight hash chunk bytes"),
                     "weight hash chunk bytes");
    WeightHashPair *device_chunks = nullptr;
    WeightHashPair *device_result = nullptr;
    try {
      internal::throw_cuda(
          "forward-weight-hash", cudaMalloc(&device_chunks, chunk_bytes), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(hash chunks)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda("forward-weight-hash",
                           cudaMalloc(&device_result, sizeof(WeightHashPair)),
                           0, 1, "forward-weight-validation", 1,
                           ExchangeKind::ExpertCounts,
                           "cudaMalloc(hash result)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      hash_forward_weight_chunks_kernel<<<static_cast<unsigned>(chunk_count),
                                          256, 0, state->stream>>>(
          values, elements, device_chunks);
      internal::throw_cuda("forward-weight-hash", cudaGetLastError(), 0, 1,
                           "forward-weight-validation", 1,
                           ExchangeKind::ExpertCounts, field);
      finalize_forward_weight_hash_kernel<<<1, 1, 0, state->stream>>>(
          device_chunks, chunk_count, elements, device_result);
      internal::throw_cuda("forward-weight-hash", cudaGetLastError(), 0, 1,
                           "forward-weight-validation", 1,
                           ExchangeKind::ExpertCounts,
                           "finalize_forward_weight_hash_kernel");
      WeightHashPair host{};
      internal::throw_cuda(
          "forward-weight-hash",
          cudaMemcpyAsync(&host, device_result, sizeof(host),
                          cudaMemcpyDeviceToHost, state->stream),
          0, 1, "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaMemcpyAsync(hash result)");
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
      checked_mul(impl.hidden, impl.intermediate, "W2 hash expert elements"),
      "W2 hash elements");
  ForwardWeightStorageHashes hashes;
  hashes.w1 =
      hash_one(impl.w1, w1_elements, "hash_forward_weight_chunks_kernel(W1)");
  hashes.w2 =
      hash_one(impl.w2, w2_elements, "hash_forward_weight_chunks_kernel(W2)");
  const auto w1_expert_elements = checked_mul(
      2ULL,
      checked_mul(impl.intermediate, impl.hidden, "sample W1 expert elements"),
      "sample W1 expert elements");
  const auto w2_expert_elements =
      checked_mul(impl.hidden, impl.intermediate, "sample W2 expert elements");
  std::array<std::uint32_t, 3> sampled_experts{0U, impl.owned_experts / 2U,
                                               impl.owned_experts - 1U};
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
  std::array<std::uint64_t, 4> sampled_rows{0ULL, 1ULL, impl.hidden / 2ULL,
                                            impl.hidden - 1ULL};
  for (const auto row : sampled_rows)
    hashes.sampled_w2_output_rows.push_back(
        hash_one(impl.w2 + row * impl.intermediate, impl.intermediate,
                 "hash_forward_weight_chunks_kernel(W2 output row)"));
  return hashes;
}

constexpr std::uint64_t kLocalP2pBootstrapMagic = 0x31504350454f4d4dULL;
constexpr std::uint64_t kLocalP2pBootstrapSchema = 2;
constexpr std::size_t kLocalP2pBootstrapHeaderBytes = 96;

void put_u64_le(std::vector<std::byte> &wire, std::size_t offset,
                std::uint64_t value) {
  if (offset > wire.size() || wire.size() - offset < sizeof(value))
    transport_error(NotifyErrorCode::ArithmeticOverflow, "local-p2p-bootstrap",
                    "wire offset", 0, "8 writable bytes", "out of range");
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
  std::array<std::uint64_t, 2> hash{1469598103934665603ULL, 1099511628211ULL};
  for (const auto byte : identity) {
    const auto value = static_cast<std::uint8_t>(byte);
    hash[0] = (hash[0] ^ value) * 1099511628211ULL;
    hash[1] = (hash[1] ^ static_cast<std::uint8_t>(value + 0x9dU)) *
              1469598103934665603ULL;
  }
  return hash;
}

std::shared_ptr<LocalP2pBackendState>
create_local_p2p_backend(const NcclSession &session,
                         const CudaExecution &execution,
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
                    "provenance", session.rank(), "same session/world/device",
                    "mismatch");
  // Device ordinals are process-local: schedulers commonly expose only one
  // GPU to each actor, making every actor's ordinal zero. CUDA IPC can still
  // import peer allocations; identify the physical devices by UUID instead.
  cudaDeviceProp device_properties{};
  internal::throw_cuda(
      "local-p2p-create",
      cudaGetDeviceProperties(&device_properties,
                              static_cast<int>(pool->device)),
      session.rank(), nccl->world, session.session().identity(),
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
  const auto handles_bytes =
      checked_mul(slot_count, sizeof(cudaIpcMemHandle_t), "IPC handle bytes");
  const auto record_bytes = checked_size(
      kLocalP2pBootstrapHeaderBytes + handles_bytes, "IPC record bytes");
  const auto matrix_bytes =
      checked_size(checked_mul(record_bytes, nccl->world, "IPC matrix bytes"),
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
        "local-p2p-create", cudaStreamSynchronize(exec->stream), session.rank(),
        nccl->world, session.session().identity(),
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
      const auto *record =
          matrix.data() + static_cast<std::size_t>(rank) * record_bytes;
      const auto check = [&](std::size_t offset, std::uint64_t expected,
                             const char *field) {
        const auto actual =
            get_u64_le(record, record_bytes, offset, session.rank(), field);
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
                        "unique physical GPU",
                        "duplicate at peer rank " + std::to_string(rank));
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
        "local-p2p-create", cudaStreamSynchronize(exec->stream), session.rank(),
        nccl->world, session.session().identity(),
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

std::shared_ptr<GinBackendState>
create_gin_backend(const NcclSession &session, const CudaExecution &execution,
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
                    "provenance", session.rank(), "same session/world/device",
                    "mismatch");
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

  using QueryPropertiesFn =
      ncclResult_t (*)(ncclComm_t, ncclCommProperties_t *);
  using WindowRegisterFn =
      ncclResult_t (*)(ncclComm_t, void *, std::size_t, ncclWindow_t *, int);
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
  const auto query_properties =
      reinterpret_cast<QueryPropertiesFn>(load("ncclCommQueryProperties"));
  const auto window_register =
      reinterpret_cast<WindowRegisterFn>(load("ncclCommWindowRegister"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  internal::throw_nccl("gin-create", query_properties(nccl->comm, &properties),
                       nccl->rank, nccl->world, session.session().identity(),
                       session.session().generation(),
                       ExchangeKind::ExpertCounts, "ncclCommQueryProperties");
  if (properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=railed_gin, rank=" +
            std::to_string(session.rank()) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (pool->config.max_nodes == 0 || nccl->world % pool->config.max_nodes != 0)
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
            "actual=world=" +
            std::to_string(nccl->world) +
            ", rail=" + std::to_string(rail.nRanks) +
            ", lsa=" + std::to_string(lsa.nRanks) +
            ", device=" + std::to_string(pool->device) +
            ", local_rank=" + std::to_string(local_rank));

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
  internal::throw_cuda(
      "gin-create",
      cudaDeviceGetAttribute(&context_sms, cudaDevAttrMultiProcessorCount,
                             pool->device),
      nccl->rank, nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "query GIN context participants");
  unsigned allocated_context_count = std::max(
      (1U + communication::kReturnPorts) * kGinContextCount,
      (forward_tuning::kUnifiedWorkers || backward_tuning::kUnifiedWorkers)
          ? 1U + unsigned(context_sms)
          : 0U);
  if constexpr (WEFTMOE_GIN_CONTEXT_COUNT > 0)
    allocated_context_count = WEFTMOE_GIN_CONTEXT_COUNT;
  const bool static_inference_pool = forward_tuning::kUnifiedWorkers &&
                                     pool->config.max_intermediate_size &&
                                     !pool->config.enable_backward_state;
  if (static_inference_pool)
    allocated_context_count = pool->gin_layout.static_forward.contexts;
  backend->gin_context_count = kGinContextCount;
  backend->gin_allocated_context_count = allocated_context_count;
  if (!resources.connection) {
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
    // Legacy wire slots retain per-slot completion counters. Static Forward
    // uses an owner-private flushed prefix; 0/1 serve bootstrap/terminal
    // status.
    const auto wire_counters =
        pool->gin_layout.rings.channels() * communication::kRingDepth;
    requirements.ginCounterCount = 2U + wire_counters * pool->slots.size();
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    requirements.ginType = NCCL_GIN_TYPE_GDAKI;
    resources.connection =
        internal::lease_device_communicator(session, requirements);
    resources.dev_comm = resources.connection->value;
    if ((static_inference_pool &&
         resources.dev_comm.ginContextCount < allocated_context_count) ||
        resources.dev_comm.ginContextCount < 1U ||
        (!static_inference_pool &&
         !(forward_tuning::kUnifiedWorkers &&
           backward_tuning::kUnifiedWorkers) &&
         resources.dev_comm.ginContextCount <
             (1U + communication::kReturnPorts) * kGinContextCount))
      transport_error(
          NotifyErrorCode::BackendUnavailable, "gin-create",
          "actual GIN contexts", nccl->rank,
          "enough contexts for static owners or legacy Return streams",
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
        const auto stream =
            internal::checked_execution_state(execution)->stream;
        if (pool->gin_layout.static_forward.rings.depth) {
          weftmoe_pf::netring::
              initialize<<<allocated_context_count, 1, 0, stream>>>(
                  resources.dev_comm, window, pool->gin_layout.static_forward);
        }
        if (pool->gin_layout.rings.channels()) {
          auto ring_layout = pool->gin_layout.rings;
          ring_layout.counter_base += (&slot - pool->slots.data()) *
                                      ring_layout.channels() *
                                      communication::kRingDepth;
          communication::initialize<<<allocated_context_count, 1, 0, stream>>>(
              resources.dev_comm, window, ring_layout, allocated_context_count);
        }
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
  backend->gin_allocated_context_count =
      unsigned(resources.dev_comm.ginContextCount);
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
        old.stream_ordered_training == config.stream_ordered_training &&
        old.decode_only == config.decode_only &&
        old.fp8_workspace == config.fp8_workspace &&
        old.static_forward_window_tokens ==
            config.static_forward_window_tokens &&
        old.static_forward_contribution_windows ==
            config.static_forward_contribution_windows &&
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
  if (config.fp8_workspace && (!WEFTMOE_FP8_SOURCE_X || !WEFTMOE_FP8_SOURCE_DY ||
      !WEFTMOE_FP8_PACKED_RING || (!config.decode_only &&
      (forward_tuning::kFc1TileN != 256U ||
       (!local_prefill_streaming(config) && !network_prefill_streaming(config)))) || config.enable_socket_fallback))
    throw std::invalid_argument("Compact FP8 workspace requires PF N256 with source quantization");
  auto state = std::make_shared<ActivationPoolState>();
  state->config = config;
  state->layout = make_layout(config);
  state->gin_layout = make_gin_window_layout(config);
  state->session_identity = session.stable_identity_bytes();
  state->world = session.world_size();
  state->execution = execution;
  state->device = execution.device_ordinal();
  internal::throw_cuda(
      "transport-pool-create", cudaSetDevice(static_cast<int>(state->device)),
      0, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts, "cudaSetDevice");
  if (config.enable_backward_state) {
    throw_cublas("transport-pool-create", cublasCreate(&state->backward_blas),
                 0, "cublasCreate(Backward)");
    throw_cublas(
        "transport-pool-create",
        cublasSetAtomicsMode(state->backward_blas, CUBLAS_ATOMICS_NOT_ALLOWED),
        0, "cublasSetAtomicsMode(Backward deterministic)");
  }
#if WEFTMOE_DEVICE_TIMELINE
  cudaDeviceProp timeline_device_properties{};
  internal::throw_cuda("transport-pool-create",
                       cudaGetDeviceProperties(&timeline_device_properties,
                                               static_cast<int>(state->device)),
                       0, session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::ExpertCounts,
                       "cudaGetDeviceProperties(device timeline)");
  if (timeline_device_properties.multiProcessorCount <= 0)
    transport_error(
        NotifyErrorCode::InvalidRuntimeConfig, "transport-pool-create",
        "timeline CTA capacity", 0, "positive SM count",
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
      internal::throw_cuda("transport-pool-create",
                           cudaMalloc(&slot.device, state->layout.total), 0,
                           session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaMalloc(transport slot)");
      if (config.enable_gin_rdma && new_communication) {
        using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
        const auto mem_alloc =
            reinterpret_cast<MemAllocFn>(dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
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
          cudaMalloc(&slot.forward_timeline, state->timeline_layout.total), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(Forward timeline)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.forward_timeline, 0,
                          state->timeline_layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "one-time initialize Forward timeline");
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
            cudaMemsetAsync(
                slot.gin_window, 0, state->gin_layout.total,
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
        const auto activation_rows =
            state->layout.static_ring.depth
                ? std::uint64_t(state->layout.static_ring.depth) *
                      forward_tuning::kTileM
                : config.max_destination_rows_per_rank;
        slot.forward_x_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, activation_rows, forward_tuning::kTileM,
            "canonical X", forward_tuning::kTileK);
        slot.forward_s_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size, activation_rows,
            forward_tuning::kTileM, "gated SwiGLU U", forward_tuning::kTileK);
        slot.forward_fp8_x_map = make_fp8_tma_map(
            at<void>(slot.device, state->layout.canonical_rows), config.max_hidden_size,
            activation_rows, 128U, "FP8 ring X", config.fp8_workspace ? fp8::source_pitch(config.max_hidden_size) : config.max_hidden_size * 2U,
            WEFTMOE_FP8_PACKED_RING && forward_tuning::kFc1TileN == 256U &&
            state->layout.static_ring.depth != 0U);
        slot.forward_fp8_s_map = make_fp8_tma_map(
            at<void>(slot.device, state->layout.forward_s), config.max_intermediate_size,
            activation_rows, 128U, "FP8 ring S", config.fp8_workspace ? fp8::source_pitch(config.max_intermediate_size) : config.max_intermediate_size * 2U,
            WEFTMOE_FP8_PACKED_RING && forward_tuning::kFc1TileN == 256U &&
            state->layout.static_ring.depth != 0U);
        if (weftmoe_dt::eligible(config, state->layout.static_ring)) {
          const auto rows = weftmoe_dt::kM;
          slot.decode_fp8_x_map = make_fp8_tma_map(
              at<void>(slot.device, state->layout.canonical_rows), config.max_hidden_size,
              activation_rows, rows, "DT FP8 X", config.fp8_workspace ? fp8::source_pitch(config.max_hidden_size) : config.max_hidden_size * 2U, true);
          slot.decode_fp8_s_map = make_fp8_tma_map(
              at<void>(slot.device, state->layout.forward_s), config.max_intermediate_size,
              activation_rows, rows, "DT FP8 S", config.fp8_workspace ? fp8::source_pitch(config.max_intermediate_size) : config.max_intermediate_size * 2U, true);
          slot.decode_x_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.canonical_rows),
              config.max_hidden_size, activation_rows, rows, "decode X",
              weftmoe_dt::kK);
          slot.decode_s_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.forward_s),
              config.max_intermediate_size, activation_rows, rows, "decode S",
              weftmoe_dt::kK);
        }
        // Ring FC2 writes contribution/Return TX directly. Reuse its unused
        // epilogue descriptor for FC1 S; arena entries still describe C.
        const bool ring = state->layout.static_ring.depth != 0U;
        slot.forward_output_store_map = make_bf16_tma_store_map(
            at<void>(slot.device, ring ? state->layout.forward_s
                                       : state->layout.canonical_rows),
            ring ? config.max_intermediate_size : config.max_hidden_size,
            activation_rows,
            ring ? "Forward ring SwiGLU" : "Forward canonical output");
        if (config.enable_backward_state && !state->layout.static_ring.depth)
          slot.forward_training_output_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Forward training output");
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
          ExchangeKind::ExpertCounts, "cudaHostAlloc(forward status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      slot.host_forward_done =
          reinterpret_cast<std::uint32_t *>(slot.host_forward_status + 1);
      *slot.host_forward_done = 0;
      void *mapped_forward_status = nullptr;
      internal::throw_cuda("transport-pool-create",
                           cudaHostGetDevicePointer(&mapped_forward_status,
                                                    slot.host_forward_status,
                                                    0),
                           0, session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
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
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_forward_progress),
                        slot.forward_progress_words * sizeof(std::uint64_t),
                        cudaHostAllocPortable | cudaHostAllocMapped),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(mapped Forward progress)");
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
                        sizeof(std::uint64_t), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      *slot.host_cancel_epoch = 0;
      if (config.enable_socket_fallback) {
        const auto socket_count_bytes = checked_size(
            checked_mul(
                checked_mul(config.max_world_size, config.max_world_size,
                            "Socket host count matrix"),
                sizeof(std::uint64_t), "Socket host count matrix bytes"),
            "Socket host count matrix bytes");
        internal::throw_cuda(
            "transport-pool-create",
            cudaHostAlloc(reinterpret_cast<void **>(&slot.host_socket_counts),
                          socket_count_bytes, cudaHostAllocPortable),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, "cudaHostAlloc(Socket count matrix)");
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
        internal::throw_cuda("transport-pool-create", cudaEventCreate(event), 0,
                             session.world_size(), session.identity(),
                             session.generation(), ExchangeKind::ExpertCounts,
                             name);
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
                               &slot.backward_transport_stream,
                               cudaStreamNonBlocking, greatest_stream_priority),
                           0, session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaStreamCreateWithPriority(Forward transport)");
      create_event(&slot.producer_ready, "cudaEventCreate(producer-ready)");
      create_event(&slot.backward_roles_ready,
                   "cudaEventCreate(Forward roles ready)");
      create_event(&slot.backward_join_transport,
                   "cudaEventCreate(Forward transport done)");
      create_event(&slot.backward_join_compute,
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
          "transport-pool-create", cudaEventCreate(&slot.forward_output_ready),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward output ready)");
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

void ActivationTransportResourcePool::retire_peer_mappings() {
  if (!impl_)
    return;
  auto &state = *impl_->state;
  for (auto &slot : state.slots) {
    if (slot.backward_peer_workspace)
      slot.backward_peer_workspace->close_mappings();
  }
  if (state.peer_mappings)
    state.peer_mappings->close();
}

bool ActivationTransportResourcePool::separates_forward_notify() const {
  return impl_ && forward_tuning::kUnifiedWorkers &&
         (local_prefill_streaming(impl_->state->config) ||
          network_prefill_streaming(impl_->state->config));
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
      checked_add(checked_add(state->layout.total, state->gin_layout.total,
                              "pool compute plus GIN bytes"),
                  sizeof(std::uint64_t), "pool slot device bytes"),
      state->slots.size(), "pool device bytes");
#if WEFTMOE_DEVICE_TIMELINE
  result.device_bytes =
      checked_add(result.device_bytes,
                  checked_mul(state->timeline_layout.total, state->slots.size(),
                              "pool timeline bytes"),
                  "pool device plus timeline bytes");
#endif
  result.compute_device_bytes = checked_mul(
      state->layout.total, state->slots.size(), "pool compute device bytes");
  for (const auto &slot : state->slots)
    if (slot.backward_peer_workspace) {
      result.device_bytes += slot.backward_peer_workspace->bytes();
      result.compute_device_bytes += slot.backward_peer_workspace->bytes();
    }
  result.gin_registered_bytes =
      checked_mul(state->gin_layout.total, state->slots.size(),
                  "pool GIN registered bytes");
  const auto progress_bytes = checked_mul(
      state->slots.empty() ? 0ULL : state->slots.front().forward_progress_words,
      sizeof(std::uint64_t), "pool mapped progress bytes");
  const auto pinned_per_slot = checked_add(
      checked_add(2ULL * sizeof(TransportDeviceStatus) + sizeof(std::uint32_t),
                  sizeof(std::uint64_t), "pool pinned status/cancel bytes"),
      progress_bytes, "pool pinned bytes per slot");
  const auto socket_count_bytes =
      state->config.enable_socket_fallback
          ? checked_mul(checked_mul(state->config.max_world_size,
                                    state->config.max_world_size,
                                    "Socket host count matrix"),
                        sizeof(std::uint64_t), "Socket host count matrix bytes")
          : 0ULL;
  result.pinned_bytes =
      checked_mul(checked_add(pinned_per_slot, socket_count_bytes,
                              "pool pinned plus Socket bytes"),
                  state->slots.size(), "pool pinned bytes");
  const auto events_per_slot =
      11ULL + (state->config.enable_backward_state ? 8ULL : 0ULL);
  result.event_count =
      checked_mul(events_per_slot, state->slots.size(), "pool events");
  result.epoch_high_watermark = state->epoch_high;
  return result;
}

std::uint64_t BFloat16ActivationLease::num_tokens() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease", "lease",
                    0, "valid", "empty");
  return impl_->tokens;
}
std::uint64_t BFloat16ActivationLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease", "lease",
                    0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t BFloat16ActivationLease::element_count() const {
  return checked_mul(num_tokens(), hidden_size(), "activation elements");
}
std::uint64_t BFloat16ActivationLease::generation() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease", "lease",
                    0, "valid", "empty");
  return impl_->generation;
}
std::uint32_t BFloat16ActivationLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease", "lease",
                    0, "valid", "empty");
  return impl_->device;
}
std::uint32_t BFloat16ActivationLease::layout_version() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease", "lease",
                    0, "valid", "empty");
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
    const std::vector<transport::BFloat16> &host, std::uint64_t tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!context || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "output-gradient-test-upload", "context/execution", 0,
                    "valid", "empty");
  const auto &identity = context.session();
  if (tokens == 0 || hidden == 0 ||
      host.size() !=
          checked_size(checked_mul(tokens, hidden, "output gradient elements"),
                       "output gradient elements"))
    transport_error(NotifyErrorCode::InvalidRouteShape,
                    "output-gradient-test-upload", "host shape",
                    context.calling_rank(), "exact nonempty [tokens, hidden]",
                    std::to_string(host.size()));
  const auto bytes =
      checked_size(checked_mul(host.size(), sizeof(transport::BFloat16),
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
  internal::throw_cuda("output-gradient-test-upload",
                       cudaMemcpyAsync(allocation->data, host.data(), bytes,
                                       cudaMemcpyHostToDevice, exec->stream),
                       context.calling_rank(), identity.world_size(),
                       identity.identity(), identity.generation(),
                       ExchangeKind::ExpertCounts,
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
  impl->pointer =
      static_cast<const transport::BFloat16 *>(impl->allocation->data);
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
                    "output-gradient-benchmark-reissue", "input", 0, "valid",
                    "empty");
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
    transport_error(NotifyErrorCode::InvalidHandle, "activation-test-upload",
                    "context/execution/pool", 0, "valid", "empty");
  auto state = pool.impl_->state;
  if (!context.execution().same_instance(execution.token()) ||
      context.session().stable_identity_bytes() != state->session_identity ||
      execution.device_ordinal() != state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "activation-test-upload", "provenance",
                    context.calling_rank(),
                    "exact pool session/execution/device", "mismatch");
  if (num_tokens > state->config.max_tokens_per_rank || hidden == 0 ||
      hidden > state->config.max_hidden_size)
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
    const auto bytes =
        checked_size(checked_mul(count, sizeof(transport::BFloat16),
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
  impl->pointer =
      at<const transport::BFloat16>(slot.device, state->layout.input);
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
    transport_error(
        NotifyErrorCode::WorkspaceMismatch, "activation-benchmark-reissue",
        "provenance", context.calling_rank(),
        "same immutable storage/session/execution/device/ready", "mismatch");
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
  const auto elements =
      checked_mul(num_tokens, hidden_size, "external activation elements");
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
    const transport::BFloat16 *device_data, std::uint64_t capacity_elements,
    std::uint64_t num_tokens, std::uint64_t hidden_size,
    std::shared_ptr<const void> allocation_owner) const {
  // A rank with no source tokens still participates in expert computation and
  // collectives. Its empty dY lease must carry the same readiness/provenance.
  if (!state_ || !allocation_owner || hidden_size == 0)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-output-gradient-publish", "producer/owner/shape",
                    0, "valid and nonzero", "empty");
  const auto elements =
      checked_mul(num_tokens, hidden_size, "external output gradient elements");
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
      ExchangeKind::ExpertCounts, "cudaEventCreate(output gradient ready)");
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

BFloat16ActivationLease ExternalCudaProducer::publish_fp8_activations(
    const FP8SourceBuffers &buffers, std::uint64_t num_tokens,
    std::uint64_t hidden_size) const {
  if (!state_ || !buffers.data.owner || !buffers.scales.owner ||
      hidden_size == 0 || hidden_size % 128U)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-activation-publish", "producer/owner/hidden", 0,
                    "valid", "empty");
  const auto &producer = *state_->backend;
  const auto elements =
      checked_mul(num_tokens, hidden_size, "external activation elements");
  if (reinterpret_cast<std::uintptr_t>(buffers.data.data) % 16U)
    transport_error(NotifyErrorCode::InvalidHandle, "external-activation-publish",
                    "FP8 source alignment", 0, "16-byte aligned", "misaligned");
  validate_external_cuda_range(buffers.data.data, elements, buffers.data.bytes,
                               1U, state_->backend->device, "FP8 source data");
  validate_external_cuda_range(buffers.scales.data, elements / 128U,
                               buffers.scales.bytes / sizeof(float), sizeof(float),
                               state_->backend->device, "FP8 source scales");
  auto allocation = std::make_shared<InputTensorAllocation>();
  allocation->device = producer.device;
  allocation->data = buffers.data.data;
  allocation->owns_data = false;
  allocation->external_owner = buffers.data.owner;
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
  impl->fp8 = buffers;
  impl->prequantized = true;
  impl->session_identity = producer.context.session().stable_identity_bytes();
  impl->producer_execution = state_->execution;
  impl->generation = producer.context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden_size;
  impl->device = producer.device;
  impl->ready_recorded = true;
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16OutputGradientLease ExternalCudaProducer::publish_fp8_output_gradients(
    const FP8SourceBuffers &buffers, std::uint64_t num_tokens,
    std::uint64_t hidden_size) const {
  if (!state_ || !buffers.data.owner || !buffers.scales.owner ||
      hidden_size == 0 || hidden_size % 128U)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-output-gradient-publish", "producer/owner/shape",
                    0, "valid and nonzero", "empty");
  const auto elements =
      checked_mul(num_tokens, hidden_size, "external output gradient elements");
  if (reinterpret_cast<std::uintptr_t>(buffers.data.data) % 16U)
    transport_error(NotifyErrorCode::InvalidHandle, "external-output-gradient-publish",
                    "FP8 source alignment", 0, "16-byte aligned", "misaligned");
  validate_external_cuda_range(buffers.data.data, elements, buffers.data.bytes,
                               1U, state_->backend->device, "FP8 source data");
  validate_external_cuda_range(buffers.scales.data, elements / 128U,
                               buffers.scales.bytes / sizeof(float), sizeof(float),
                               state_->backend->device, "FP8 source scales");
  auto allocation = std::make_shared<InputTensorAllocation>();
  allocation->device = state_->backend->device;
  allocation->data = buffers.data.data;
  allocation->owns_data = false;
  allocation->external_owner = buffers.data.owner;
  allocation->external_producer = state_->backend;
  internal::throw_cuda(
      "external-output-gradient-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      state_->backend->context.calling_rank(),
      state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventCreate(output gradient ready)");
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
  impl->fp8 = buffers;
  impl->prequantized = true;
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
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create", "pool",
                    0, "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend =
      create_local_p2p_backend(session, execution, pool.impl_->state);
}

NcclSocketFallbackActivationTransport::NcclSocketFallbackActivationTransport(
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
                    "provenance", session.rank(), "same session/world/device",
                    "mismatch");
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
    transport_error(NotifyErrorCode::InvalidHandle, "forward-create", "pool", 0,
                    "valid", "empty");
  impl_ = std::make_shared<Impl>();
  if (use_gin)
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_gin_backend(session, execution, pool.impl_->state));
  else
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_local_p2p_backend(session, execution, pool.impl_->state));
  const auto &capacity = pool.impl_->state->config;
  const auto local_world =
      use_gin ? std::static_pointer_cast<GinBackendState>(impl_->backend)
                    ->local_world_size
              : impl_->backend->nccl->world;
  if (capacity.enable_backward_state)
    for (auto &slot : pool.impl_->state->slots)
      if (!slot.backward_peer_workspace) {
        slot.backward_peer_workspace = std::make_shared<BackwardPeerWorkspace>(
            session, execution, local_world, capacity);
        // Reserve IPC storage before application activations fill the Torch
        // caching allocator. cudaMalloc cannot reclaim its unused segments.
        if (!use_gin || !weftmoe_pf::kNetworkPrivateDr) {
          auto &workspace = *slot.backward_peer_workspace;
          workspace.prepare(0, workspace.initial_rows, workspace.max_routes,
                            capacity.max_hidden_size, !capacity.fp8_workspace);
        }
      }
}

GinRdmaForwardMegaKernel::GinRdmaForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pForwardMegaKernel>(
          new LocalCudaP2pForwardMegaKernel(session, execution, pool, true))) {}

DeviceForwardHandle GinRdmaForwardMegaKernel::enqueue(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->enqueue(dispatch, activations, weights, config, options);
}

DeviceForwardHandle
GinRdmaForwardMegaKernel::run(const DeviceDispatchHandle &dispatch,
                              const BFloat16ActivationLease &activations,
                              const ForwardWeightLease &weights,
                              const ForwardMegaKernelConfig &config,
                              const ActivationTransportRunOptions &options) {
  return delegate_->run(dispatch, activations, weights, config, options);
}

// Keep execution and validation separate from resource/backend preparation.
#include "cuda/activation_transport/execution.cuh"
#include "cuda/activation_transport/validation.cuh"

ActivationTransportCapability
probe_activation_transport_capability(std::uint32_t world_size) {
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
    using QueryPropertiesFn =
        ncclResult_t (*)(ncclComm_t, ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn =
        ncclResult_t (*)(ncclComm_t, const ncclDevComm_t *);
    using WindowRegisterFn =
        ncclResult_t (*)(ncclComm_t, void *, std::size_t, ncclWindow_t *, int);
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
    const auto mem_alloc =
        reinterpret_cast<MemAllocFn>(dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free =
        reinterpret_cast<MemFreeFn>(dlsym(RTLD_DEFAULT, "ncclMemFree"));
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
      init_error =
          std::string("ncclCommInitRank: ") + ncclGetErrorString(nccl_result);
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
        requirements.ginType = NCCL_GIN_TYPE_GDAKI;
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
            nccl_result = window_register(probe_comm, allocation, 4096, &window,
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
  result.gin_usable =
      result.gin_headers_available && result.gin_symbols_present &&
      result.gin_runtime_initializable && result.gin_backend_implemented;
  if (!result.gin_usable) {
    std::ostringstream message;
    message << "GIN unavailable: headers=" << result.gin_headers_available
            << ", symbols=" << result.gin_symbols_present
            << ", runtime_initializable=" << result.gin_runtime_initializable
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
      result
          .peer_access_matrix[static_cast<std::size_t>(source) * checked_world +
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

ActivationTransportCapability
probe_activation_transport_capability(const NcclSession &session,
                                      const CudaExecution &execution) {
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto cuda = internal::checked_execution_state(execution);
  if (nccl->device != cuda->device)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-distributed-probe", "device", nccl->rank,
                    std::to_string(nccl->device), std::to_string(cuda->device));

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
    using QueryPropertiesFn =
        ncclResult_t (*)(ncclComm_t, ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn =
        ncclResult_t (*)(ncclComm_t, const ncclDevComm_t *);
    using WindowRegisterFn =
        ncclResult_t (*)(ncclComm_t, void *, std::size_t, ncclWindow_t *, int);
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
    const auto mem_alloc =
        reinterpret_cast<MemAllocFn>(dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free =
        reinterpret_cast<MemFreeFn>(dlsym(RTLD_DEFAULT, "ncclMemFree"));

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
      requirements.ginType = NCCL_GIN_TYPE_GDAKI;
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

  result.gin_usable =
      result.gin_headers_available && result.gin_symbols_present &&
      result.gin_runtime_initializable && result.gin_backend_implemented;

  std::ostringstream message;
  message << "GIN distributed probe: headers=" << result.gin_headers_available
          << ", symbols=" << result.gin_symbols_present
          << ", runtime_initializable=" << result.gin_runtime_initializable
          << ", backend_implemented=" << result.gin_backend_implemented
          << ", rank=" << nccl->rank << ", world=" << nccl->world
          << ", compiled_nccl=" << result.compiled_nccl_version
          << ", runtime_nccl=" << result.runtime_nccl_version;
  if (!result.gin_unavailable_reason.empty())
    message << ", initialization_error=" << result.gin_unavailable_reason;
  result.gin_unavailable_reason = message.str();
  return result;
}

bool backward_pipeline_available() noexcept { return true; }

} // namespace weftmoe::cuda
