#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "weftmoe/cuda/activation_transport.hpp"

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe::cuda {

// Readiness-bearing, caller-owned source-token dY. Like a Forward activation
// lease, this never consumes an execution-workspace slot.
class BFloat16OutputGradientLease {
public:
  BFloat16OutputGradientLease() = default;
  std::uint64_t num_tokens() const;
  std::uint64_t hidden_size() const;
  std::uint64_t element_count() const;
  std::uint64_t generation() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit BFloat16OutputGradientLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class BackwardMegaKernel;
  friend class ExternalCudaProducer;
  friend class BackwardValidationBridge;
};

enum class BackwardKernelSpecialization : std::uint8_t {
  Sm90CachedXAb7168x3072,
};

struct BackwardMegaKernelConfig {
  std::uint64_t hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  BackwardTensorStorage tensors{};
};

struct BackwardMegaKernelTimings {
  double prepare_us = 0;
  double compute_us = 0;
  double reduce_us = 0;
  double transport_us = 0;
  double dispatch_us = 0;
  double dab_us = 0;
  double dx_us = 0;
  double dw2_us = 0;
  double dw1_us = 0;
  double combine_us = 0;
  double dw_combine_overlap_us = 0;
  double total_us = 0;
};

struct BackwardMegaKernelAudit {
  BackwardKernelSpecialization specialization =
      BackwardKernelSpecialization::Sm90CachedXAb7168x3072;
  std::uint64_t gemm_launches = 0;
  std::uint64_t actual_dispatch_peer_bytes = 0;
  std::uint64_t actual_return_peer_bytes = 0;
  std::uint64_t gin_return_put_count = 0;
  std::uint64_t gin_return_payload_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t hot_path_allocations = 0;
};

class DeviceBackwardHandle {
public:
  DeviceBackwardHandle() = default;
  const ActivationTransportExpectation &identity() const;
  const BackwardMegaKernelAudit &audit() const;
  const BackwardMegaKernelTimings &timings() const;
  const BackwardTensorStorage &tensors() const;
  void wait() const;
  bool ready() const;
  void wait_on(CUstream_st *stream) const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceBackwardHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class BackwardMegaKernel;
  friend class BackwardValidationBridge;
};

// The exact transport backend and immutable routing directory come from the
// retained Forward handle, so Backward cannot accidentally bind a different
// session or route generation.
class BackwardMegaKernel {
public:
  // Uses the retained Forward's bounded communication geometry; the private
  // dR/U/dAB workspace and caller-visible gradients follow actual tape sizes.
  BackwardTensorStorage prepare_tensors(
      const DeviceForwardHandle &forward, const BackwardMegaKernelConfig &config,
      const BackwardTensorBuffers &bindings = {},
      const CudaTensorAllocator &allocator = {}) const;

  DeviceBackwardHandle enqueue(
      const DeviceForwardHandle &forward,
      const BFloat16OutputGradientLease &output_gradients,
      const BackwardMegaKernelConfig &config,
      const ActivationTransportRunOptions &options = {}) const;

  DeviceBackwardHandle run(
      const DeviceForwardHandle &forward,
      const BFloat16OutputGradientLease &output_gradients,
      const BackwardMegaKernelConfig &config,
      const ActivationTransportRunOptions &options = {}) const;
};

// Reports whether this binary contains the cached-state CUDA implementation
// that passed the maintained local and GIN qualification.
bool backward_pipeline_available() noexcept;

} // namespace weftmoe::cuda

WEFTMOE_PUBLIC_API_END
