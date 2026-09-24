#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#include "weftmoe/version.hpp"

WEFTMOE_PUBLIC_API_BEGIN
namespace weftmoe::cuda {
class CudaExecution;
class NcclSession;
class DeviceDispatchHandle;
class LocalCudaP2pForwardMegaKernel;
class BackwardMegaKernel;
class BackwardValidationBridge;
class DeviceForwardHandle;
class DeviceBackwardHandle;
struct ForwardMegaKernelConfig;
struct BackwardMegaKernelConfig;

// Caller-owned CUDA storage, for example an at::empty tensor whose owner keeps
// its caching-allocator allocation alive. Buffers are contiguous and must not
// alias one another or any live input. Nonempty buffers require 128-byte
// alignment. Empty bindings use the native allocator. Reusing an explicit
// storage object overwrites its previous results; retain separate storage for
// each Forward whose tensors will still be read by Backward.
struct CudaTensorBuffer {
  void *data = nullptr;
  std::uint64_t bytes = 0;
  std::shared_ptr<const void> owner;
};

// An adapter can use at::empty for every unbound tensor, including saved route
// metadata. Allocation is performed during preparation, outside kernel timing.
using CudaTensorAllocator =
    std::function<CudaTensorBuffer(std::uint64_t bytes)>;

// A completed collective rejected preparation; no main kernel was entered.
// CUDA/NCCL failures of the agreement itself do not use this exception type.
class TensorPreparationFailure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Collective host-allocation boundary, outside CUDA Graph capture. Construct
// before allocating per-call tensors; every rank then calls run in the same
// order. The callback performs rank-local preparation, not collectives. No
// rank may enter execution when any rank's preparation throws.
// The small status buffer and NCCL collective are prepared once, before an
// allocation failure could prevent creating the agreement's own resources.
// Calls on one instance must be serialized. This does not recover a failed
// CUDA context or a broken communicator.
class CollectiveTensorPreparation {
public:
  CollectiveTensorPreparation(const NcclSession &, const CudaExecution &);
  void run(const std::function<void()> &prepare) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

struct ForwardTensorBuffers {
  CudaTensorBuffer output;     // BF16 [tokens, hidden]
  CudaTensorBuffer fc1_input;  // BF16 [received_rows, hidden], training only
  CudaTensorBuffer fc1_output; // BF16 [2, received_rows, intermediate]
};

struct BackwardTensorBuffers {
  CudaTensorBuffer dx;    // BF16 [tokens, hidden]
  CudaTensorBuffer dgate; // FP32 [tokens, top_k]
  CudaTensorBuffer dw1;   // BF16 [local_experts, 2 * intermediate, hidden]
  CudaTensorBuffer dw2;   // BF16 [local_experts, hidden, intermediate]
};

// These objects own per-call tensors, rank-private Backward intermediates,
// and their completion event. They never register an NCCL window or lease a
// shared transport workspace. Prepare
// them before capture; a captured execution reuses its explicitly bound
// tensors.
class ForwardTensorStorage {
public:
  ForwardTensorStorage() = default;
  ForwardTensorBuffers buffers() const;
  std::uint64_t row_capacity() const;
  std::uint64_t bytes() const;
  explicit operator bool() const noexcept { return bool(impl_); }

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend ForwardTensorStorage
  prepare_forward_tensors(const DeviceDispatchHandle &, const CudaExecution &,
                          const ForwardMegaKernelConfig &,
                          const ForwardTensorBuffers &,
                          const CudaTensorAllocator &);
  friend class LocalCudaP2pForwardMegaKernel;
  friend class BackwardMegaKernel;
  friend class BackwardValidationBridge;
  friend class DeviceForwardHandle;
  friend class DeviceBackwardHandle;
};

class BackwardTensorStorage {
public:
  BackwardTensorStorage() = default;
  BackwardTensorBuffers buffers() const;
  std::uint64_t bytes() const;
  explicit operator bool() const noexcept { return bool(impl_); }

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend BackwardTensorStorage
  prepare_backward_tensors(const DeviceDispatchHandle &, const CudaExecution &,
                           const BackwardMegaKernelConfig &,
                           const BackwardTensorBuffers &,
                           const CudaTensorAllocator &);
  friend class BackwardMegaKernel;
  friend class BackwardValidationBridge;
  friend class DeviceBackwardHandle;
};

ForwardTensorStorage prepare_forward_tensors(const DeviceDispatchHandle &,
                                             const CudaExecution &,
                                             const ForwardMegaKernelConfig &,
                                             const ForwardTensorBuffers & = {},
                                             const CudaTensorAllocator & = {});
BackwardTensorStorage
prepare_backward_tensors(const DeviceDispatchHandle &, const CudaExecution &,
                         const BackwardMegaKernelConfig &,
                         const BackwardTensorBuffers & = {},
                         const CudaTensorAllocator & = {});

} // namespace weftmoe::cuda
WEFTMOE_PUBLIC_API_END
