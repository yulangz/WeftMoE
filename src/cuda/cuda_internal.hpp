#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "megamoe/cuda/nccl_count_exchange.hpp"

namespace megamoe::cuda::internal {

struct CudaExecutionState final : BackendExecutionState {
  std::uint32_t device = 0;
  cudaStream_t stream = nullptr;
  ~CudaExecutionState() noexcept override;
};

struct NcclSessionState final : BackendSessionState {
  std::uint32_t device = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  ncclComm_t comm = nullptr;
  ~NcclSessionState() noexcept override;
};

struct CudaAllocation {
  std::uint32_t device = 0;
  void *device_ptr = nullptr;
  void *host_ptr = nullptr;
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
  cudaEvent_t mapping_start = nullptr;
  cudaEvent_t mapping_complete = nullptr;
  cudaEvent_t completion = nullptr;
  ~CudaAllocation() noexcept;
};

struct CudaBufferState final : BackendDeviceLeaseState {
  std::shared_ptr<const void> owner;
  void *pointer = nullptr;
  std::uint32_t device = 0;
};

struct CudaCompletionLease final : BackendCompletionLeaseState {
  std::shared_ptr<const void> owner;
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

} // namespace megamoe::cuda::internal
