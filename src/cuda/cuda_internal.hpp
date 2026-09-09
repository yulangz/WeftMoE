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

struct CudaExecutionState final : BackendExecutionState {
  std::uint32_t device = 0;
  cudaStream_t stream = nullptr;
  ~CudaExecutionState() noexcept override;
};

struct ExternalCudaProducerState final : BackendExecutionState {
  RankCollectiveContext context;
  ExecutionToken execution;
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
  cudaEvent_t completion = nullptr;
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
