#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "weftmoe/cuda/nccl_count_exchange.hpp"

namespace weftmoe::cuda::internal {

// Prepare before launching mutually dependent persistent kernels. Growing the
// per-thread stack at a later kernel launch can wait for an already-running
// kernel, which may itself be waiting for the kernel being launched. The local
// storage requirement is a conservative bound for these nonrecursive kernels.
// Preserve a larger limit established by the application.
inline cudaError_t prepare_kernel_stack(const void *kernel) {
  cudaFuncAttributes attributes{};
  auto error = cudaFuncGetAttributes(&attributes, kernel);
  if (error != cudaSuccess || attributes.localSizeBytes == 0U)
    return error;
  std::size_t current = 0;
  error = cudaDeviceGetLimit(&current, cudaLimitStackSize);
  if (error != cudaSuccess || current >= attributes.localSizeBytes)
    return error;
  return cudaDeviceSetLimit(cudaLimitStackSize, attributes.localSizeBytes);
}

enum class CudaApiAuditKind {
  DeviceAllocate,
  DeviceFree,
  PinnedAllocate,
  PinnedFree,
  EventCreate,
  EventDestroy,
  DeviceSynchronize,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
};
void record_cuda_api(CudaApiAuditKind kind,
                     std::uint64_t bytes = 0) noexcept;

// Timing and host-visible completion need actual event nodes during capture.
// Dependency-only events continue to use cudaEventRecord directly.
inline cudaError_t
record_observable_event(cudaEvent_t event, cudaStream_t stream,
                        unsigned long long *recorded_capture = nullptr) {
  cudaStreamCaptureStatus status;
  unsigned long long capture_id = 0;
  auto error = cudaStreamGetCaptureInfo(stream, &status, &capture_id);
  if (error != cudaSuccess)
    return error;
  if (recorded_capture)
    *recorded_capture =
        status == cudaStreamCaptureStatusActive ? capture_id : 0;
  return status == cudaStreamCaptureStatusActive
             ? cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal)
             : cudaEventRecord(event, stream);
}

// Observable event nodes do not enroll a new stream into a capture. Use the
// producer's dependency events within that capture, and its observable event
// for consumers outside it (including another graph).
inline cudaError_t
wait_output_ready(cudaStream_t stream, cudaStream_t producer_stream,
                  cudaEvent_t observable, unsigned long long recorded_capture,
                  cudaEvent_t first, cudaEvent_t second = nullptr) {
  cudaStreamCaptureStatus status;
  unsigned long long capture_id = 0;
  auto error = cudaStreamGetCaptureInfo(stream, &status, &capture_id);
  if (error != cudaSuccess)
    return error;
  cudaStreamCaptureStatus producer_status;
  unsigned long long producer_capture = 0;
  error = cudaStreamGetCaptureInfo(producer_stream, &producer_status,
                                   &producer_capture);
  if (error != cudaSuccess)
    return error;
  if (producer_status == cudaStreamCaptureStatusActive &&
      producer_capture == recorded_capture &&
      (status == cudaStreamCaptureStatusNone ||
       capture_id == recorded_capture)) {
    error = cudaStreamWaitEvent(stream, first);
    return error != cudaSuccess || !second
               ? error
               : cudaStreamWaitEvent(stream, second);
  }
  return cudaStreamWaitEvent(stream, observable,
                             status == cudaStreamCaptureStatusActive
                                 ? cudaEventWaitExternal
                                 : cudaEventWaitDefault);
}

// Input events can precede capture or be produced inside the enclosing graph.
// Keep internal graph edges when both ends were recorded in the same capture.
inline cudaError_t record_input_ready(cudaEvent_t event, cudaStream_t stream,
                                      unsigned long long &capture_id) {
  cudaStreamCaptureStatus status;
  auto error = cudaStreamGetCaptureInfo(stream, &status, &capture_id);
  if (error != cudaSuccess)
    return error;
  if (status != cudaStreamCaptureStatusActive)
    capture_id = 0;
  return cudaEventRecord(event, stream);
}

inline cudaError_t wait_input_ready(cudaStream_t stream, cudaEvent_t event,
                                    unsigned long long recorded_capture) {
  cudaStreamCaptureStatus status;
  unsigned long long capture_id = 0;
  auto error = cudaStreamGetCaptureInfo(stream, &status, &capture_id);
  if (error != cudaSuccess)
    return error;
  const auto flags =
      status == cudaStreamCaptureStatusActive && capture_id != recorded_capture
          ? cudaEventWaitExternal
          : cudaEventWaitDefault;
  return cudaStreamWaitEvent(stream, event, flags);
}

struct CudaExecutionState final : BackendExecutionState {
  std::uint32_t device = 0;
  cudaStream_t stream = nullptr;
  bool owns_stream = true;
  std::shared_ptr<const void> stream_owner;
  ~CudaExecutionState() noexcept override;
};

struct ExternalCudaProducerState final : BackendExecutionState {
  RankCollectiveContext context;
  std::shared_ptr<const void> stream_owner;
  cudaStream_t stream = nullptr;
  std::uint32_t device = 0;
};

struct NcclSessionState final : BackendSessionState {
  std::uint32_t device = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  std::vector<std::uint8_t> bootstrap_identity;
  ncclComm_t comm = nullptr;
  ~NcclSessionState() noexcept override;
};

class NcclSessionAccess final : public CountExchange {
public:
  static std::shared_ptr<NcclSessionState> checked(const NcclSession &session);

private:
  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &) override;
};

class CudaExecutionStateAccess final : public CountExchange {
public:
  static std::shared_ptr<const CudaExecutionState>
  checked(const CudaExecution &execution);

private:
  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &) override;
};

enum class ConsumerEventSlotState : std::uint8_t {
  Available,
  InUse,
  Quarantined,
};

struct CudaAllocation {
  std::uint32_t device = 0;
  void *device_ptr = nullptr;
  void *host_ptr = nullptr;
  // Device alias for the small mapped status page. Generic/public metadata
  // never exposes this address; release Notify uses it only to avoid a host
  // scheduler wake between device materialization and Forward launch.
  void *host_device_ptr = nullptr;
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
  cudaEvent_t mapping_start = nullptr;
  cudaEvent_t mapping_complete = nullptr;
  cudaEvent_t planning_start = nullptr;
  cudaEvent_t planning_complete = nullptr;
  cudaEvent_t producer_ready = nullptr;
  unsigned long long producer_capture_id = 0;
  cudaEvent_t completion = nullptr;
  unsigned long long completion_capture_id = 0;
  std::vector<cudaEvent_t> consumer_events;
  mutable std::mutex consumer_mutex;
  std::vector<ConsumerEventSlotState> consumer_event_states;
  std::atomic<bool> dispatch_slot_quarantined{false};
  bool abandon_on_teardown = false;
  bool owns_device_memory = true;
  bool owns_producer_event = true;
  std::shared_ptr<const void> external_owner;
  ~CudaAllocation() noexcept;
};

struct CudaBufferState final : BackendDeviceLeaseState {
  std::shared_ptr<const void> owner;
  void *pointer = nullptr;
  std::uint32_t device = 0;
};

struct CudaCompletionLease final : BackendCompletionLeaseState {
  CudaCompletionLease() noexcept;
  ~CudaCompletionLease() noexcept override;
  std::shared_ptr<const void> owner;
};

// Private CUDA-only access to an opaque generic lease. This type is not part of
// any installed/public header.
class CudaDeviceLeaseAccess final : public CountExchange {
public:
  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &) override;
  static std::shared_ptr<const CudaBufferState>
  checked(const DeviceBufferLease &lease, std::uint32_t expected_device);
};

void throw_cuda(const char *phase, cudaError_t error, std::uint32_t rank,
                std::uint32_t world, const std::string &session,
                std::uint64_t generation, ExchangeKind kind,
                const char *operation);
void throw_nccl(const char *phase, ncclResult_t error, std::uint32_t rank,
                std::uint32_t world, const std::string &session,
                std::uint64_t generation, ExchangeKind kind,
                const char *operation);

std::shared_ptr<const CudaExecutionState>
checked_execution_state(const CudaExecution &execution);

} // namespace weftmoe::cuda::internal
