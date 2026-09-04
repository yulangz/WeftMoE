#pragma once

#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/cuda/device_notify.hpp"
#include "megamoe/cuda/nccl_count_exchange.hpp"
#include "megamoe/transport/activation_protocol.hpp"
#include "megamoe/version.hpp"

MEGAMOE_PUBLIC_API_BEGIN

struct CUstream_st;

namespace megamoe::cuda {

template <class T> class SelectedActivationLease;
using BFloat16ActivationLease =
    SelectedActivationLease<transport::BFloat16>;
class BFloat16OutputGradientLease;
class BackwardMegaKernel;
class BackwardValidationBridge;
class ForwardWeightLease;
class ExternalCudaProducer;

inline constexpr std::uint32_t kSelectedActivationLayoutVersion = 1;

// Immutable router -> transport handoff. No public accessor returns a native
// device address. The producer event and execution provenance are retained by
// the implementation until the transport generation completes.
template <> class SelectedActivationLease<transport::BFloat16> {
public:
  SelectedActivationLease() = default;
  std::uint64_t num_tokens() const;
  std::uint64_t hidden_size() const;
  std::uint64_t element_count() const;
  std::uint64_t generation() const;
  std::uint32_t device_ordinal() const;
  std::uint32_t layout_version() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit SelectedActivationLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class GinRdmaForwardMegaKernel;
  friend class NcclSocketFallbackForwardMegaKernel;
  friend class ExternalCudaProducer;
  friend class BackwardMegaKernel;
  friend class BackwardValidationBridge;
};

struct ActivationTransportPoolConfig {
  std::uint64_t max_tokens_per_rank = 0;
  std::uint64_t max_hidden_size = 0;
  std::uint32_t max_top_k = 0;
  std::uint32_t max_world_size = 0;
  std::uint32_t max_nodes = 0;
  std::uint64_t max_destination_rows_per_rank = 0;
  std::uint32_t channels = 1;
  std::uint32_t ring_slots_per_source_channel = 0;
  std::uint32_t max_inflight_generations = 2;
  std::uint64_t max_intermediate_size = 0;
  std::uint32_t max_owned_experts = 0;
  // Reserves compact route/payload staging for the explicitly host-enqueued
  // NCCL Socket correctness backend. Disabled for the frozen local P2P path.
  bool enable_socket_fallback = false;
  // Registers each preallocated slot as an NCCL window for the explicitly
  // constructed GinRdma backend. It never enables or selects that backend by
  // itself.
  bool enable_gin_rdma = false;
  // Reserve generation-local state consumed by the cached-state Backward:
  // immutable canonical X, BF16 FC1 A/B checkpoints, and a Forward output
  // plane that does not alias X. The ordinary inference footprint is
  // unchanged unless this is enabled explicitly. Kept last for aggregate
  // source compatibility with the pre-checkpoint configuration.
  bool enable_backward_state = false;
};

struct ActivationTransportPoolUsage {
  std::uint32_t slots_available = 0;
  std::uint32_t slots_in_use = 0;
  std::uint32_t slots_quarantined = 0;
  std::uint64_t device_bytes = 0;
  std::uint64_t compute_device_bytes = 0;
  std::uint64_t gin_registered_bytes = 0;
  std::uint64_t pinned_bytes = 0;
  std::uint64_t event_count = 0;
  std::uint64_t epoch_high_watermark = 0;
};

class ActivationTransportResourcePool {
public:
  ActivationTransportResourcePool() = default;
  ActivationTransportPoolUsage usage() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ActivationTransportResourcePool(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend ActivationTransportResourcePool make_activation_transport_pool(
      const CollectiveSession &, const CudaExecution &,
      ActivationTransportPoolConfig);
  friend class LocalCudaP2pForwardMegaKernel;
  friend class GinRdmaForwardMegaKernel;
  friend class NcclSocketFallbackForwardMegaKernel;
  friend class ExternalCudaProducer;
};

ActivationTransportResourcePool make_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config);

struct ActivationTransportExpectation {
  std::uint64_t generation = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  std::uint64_t tokens = 0;
  std::uint64_t hidden = 0;
};


struct ActivationTransportRunOptions {
  GenerationCancellationToken cancellation;
  // Zero means no deadline. Multi-rank cancellation/deadline must be requested
  // consistently by every healthy rank in the local P2P team.
  std::chrono::microseconds deadline_after{0};
};

// A real, node-local, multi-process CUDA IPC backend. Construction exchanges
// and validates pool IPC handles over the existing node-local NCCL session.
// It fails closed unless every local peer mapping is established.

enum class ForwardKernelSpecialization : std::uint8_t {
  Sm90WgmmaTma7168x3072,
};

struct ForwardMegaKernelConfig {
  std::uint64_t hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  // Preserve the post-Dispatch canonical X and the exact BF16 A/B values
  // consumed by Forward's SwiGLU. The pool must have been constructed with
  // enable_backward_state=true.
  bool retain_backward_state = false;
};

class ForwardWeightLease {
public:
  ForwardWeightLease() = default;
  std::uint32_t owned_experts() const;
  std::uint64_t hidden_size() const;
  std::uint64_t intermediate_size() const;
  std::uint64_t device_bytes() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ForwardWeightLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class ExternalCudaProducer;
};


struct ForwardMegaKernelTimings {
  double dispatch_us = 0;
  double fc1_swiglu_us = 0;
  double fc2_us = 0;
  double combine_return_us = 0;
  double reduce_us = 0;
  double fc1_fc2_overlap_us = 0;
  double fc2_return_overlap_us = 0;
  double return_reduce_overlap_us = 0;
  double terminal_status_agreement_us = 0;
  double total_us = 0;
};

struct ForwardMegaKernelAudit {
  ForwardKernelSpecialization specialization =
      ForwardKernelSpecialization::Sm90WgmmaTma7168x3072;
  std::uint64_t persistent_kernel_launches = 0;
  std::uint64_t wgmma_instructions = 0;
  std::uint64_t tma_loads = 0;
  std::uint64_t tma_multicast_loads = 0;
  bool runtime_cluster_supported = false;
  std::uint32_t requested_cluster_size = 1;
  std::uint32_t runtime_cluster_size = 1;
  std::uint64_t observed_cluster_rank_mask = 1;
  std::uint32_t max_active_clusters = 0;
  std::uint64_t fc1_tasks = 0;
  std::uint64_t fc2_tasks = 0;
  std::uint64_t actual_dispatch_peer_bytes = 0;
  std::uint64_t actual_return_peer_bytes = 0;
  std::uint64_t gin_dispatch_put_count = 0;
  std::uint64_t raw_cross_node_route_bytes = 0;
  std::uint64_t gin_dispatch_payload_bytes = 0;
  std::uint64_t gin_return_payload_bytes = 0;
  std::uint64_t gin_return_put_count = 0;
  std::uint64_t gin_signal_count = 0;
  std::uint64_t gin_metadata_bytes = 0;
  std::uint64_t hot_path_allocations = 0;
  std::uint64_t activation_d2h_bytes = 0;
  std::uint64_t activation_h2d_bytes = 0;
  std::uint64_t workspace_bytes = 0;
};

class DeviceForwardHandle {
public:
  DeviceForwardHandle() = default;
  const ActivationTransportExpectation &identity() const;
  const ForwardMegaKernelAudit &audit() const;
  const ForwardMegaKernelTimings &timings() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceForwardHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class GinRdmaForwardMegaKernel;
  friend class NcclSocketFallbackForwardMegaKernel;
  friend class ExternalCudaProducer;
};

// CUDA-specific zero-copy producer boundary. It records readiness on the
// caller-owned producer stream, checks the typed allocation range/device, and
// retains both allocation and stream owners through consumer completion.
class ExternalCudaProducer {
public:
  ExternalCudaProducer() = default;
  ExternalCudaProducer(const ExternalCudaProducer &);
  ExternalCudaProducer(ExternalCudaProducer &&) noexcept;
  ExternalCudaProducer &operator=(const ExternalCudaProducer &);
  ExternalCudaProducer &operator=(ExternalCudaProducer &&) noexcept;
  ~ExternalCudaProducer();
  const RankCollectiveContext &context() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(state_); }

  CudaRouteInput publish_selected_experts(
      const std::int32_t *device_data, std::uint64_t capacity_elements,
      std::uint64_t num_tokens, std::uint32_t top_k,
      std::shared_ptr<const void> allocation_owner) const;
  CudaGateInput publish_gates(
      const float *device_data, std::uint64_t capacity_elements,
      const CudaRouteInput &exact_routes,
      std::shared_ptr<const void> allocation_owner) const;
  BFloat16ActivationLease publish_activations(
      const transport::BFloat16 *device_data,
      std::uint64_t capacity_elements, std::uint64_t num_tokens,
      std::uint64_t hidden_size, const ActivationTransportResourcePool &pool,
      std::shared_ptr<const void> allocation_owner) const;
  // dY is caller-owned and does not reserve a second transport-pool slot.
  // The returned lease records producer-stream readiness and retains the
  // allocation until Backward has consumed it.
  BFloat16OutputGradientLease publish_output_gradients(
      const transport::BFloat16 *device_data,
      std::uint64_t capacity_elements, std::uint64_t num_tokens,
      std::uint64_t hidden_size,
      std::shared_ptr<const void> allocation_owner) const;
  ForwardWeightLease publish_weights(
      const transport::BFloat16 *w1, std::uint64_t w1_capacity_elements,
      std::shared_ptr<const void> w1_owner,
      const transport::BFloat16 *w2, std::uint64_t w2_capacity_elements,
      std::shared_ptr<const void> w2_owner, std::uint32_t owned_experts,
      std::uint32_t global_expert_begin, std::uint64_t hidden_size,
      std::uint64_t intermediate_size) const;
private:
  struct State;
  explicit ExternalCudaProducer(std::shared_ptr<const State> state)
      : state_(std::move(state)) {}
  std::shared_ptr<const State> state_;
  friend ExternalCudaProducer make_external_cuda_producer(
      RankCollectiveContext, std::uint32_t, CUstream_st *,
      std::shared_ptr<const void>, std::string);
};

ExternalCudaProducer make_external_cuda_producer(
    RankCollectiveContext context, std::uint32_t device_ordinal,
    CUstream_st *producer_stream, std::shared_ptr<const void> stream_owner,
    std::string identity);

class LocalCudaP2pForwardMegaKernel {
public:
  LocalCudaP2pForwardMegaKernel(
      const NcclSession &session, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);

  DeviceForwardHandle run(
      const DeviceDispatchHandle &dispatch,
      const BFloat16ActivationLease &activations,
      const ForwardWeightLease &weights,
      const ForwardMegaKernelConfig &config,
      const ActivationTransportRunOptions &options = {});

private:
  LocalCudaP2pForwardMegaKernel(
      const NcclSession &, const CudaExecution &,
      const ActivationTransportResourcePool &, bool use_gin);
  LocalCudaP2pForwardMegaKernel(
      const NcclSession &, const CudaExecution &,
      const ActivationTransportResourcePool &, bool use_gin,
      bool use_socket);
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class GinRdmaForwardMegaKernel;
  friend class NcclSocketFallbackForwardMegaKernel;
};

// Cross-node same-rail GIN Forward. It shares the frozen WGMMA/TMA compute
// specialization with LocalCudaP2pForwardMegaKernel, but has a distinct
// construction path and never falls back to local simulation or Socket.
class GinRdmaForwardMegaKernel {
public:
  GinRdmaForwardMegaKernel(const NcclSession &, const CudaExecution &,
                           const ActivationTransportResourcePool &);
  DeviceForwardHandle run(
      const DeviceDispatchHandle &, const BFloat16ActivationLease &,
      const ForwardWeightLease &, const ForwardMegaKernelConfig &,
      const ActivationTransportRunOptions & = {});

private:
  std::shared_ptr<LocalCudaP2pForwardMegaKernel> delegate_;
};

// Host-enqueued NCCL Socket correctness fallback for the complete Forward
// semantics. Dispatch and return have explicit host phase boundaries; the FC
// body reuses the frozen WGMMA/TMA compute core, gate-before-FC2 semantics, and
// deterministic unweighted reduce. This type never reports a GIN or
// single-persistent audit label.
class NcclSocketFallbackForwardMegaKernel {
public:
  NcclSocketFallbackForwardMegaKernel(
      const NcclSession &, const CudaExecution &,
      const ActivationTransportResourcePool &);
  DeviceForwardHandle run(
      const DeviceDispatchHandle &, const BFloat16ActivationLease &,
      const ForwardWeightLease &, const ForwardMegaKernelConfig &,
      const ActivationTransportRunOptions & = {});

private:
  std::shared_ptr<LocalCudaP2pForwardMegaKernel> delegate_;
};


struct ActivationTransportCapability {
  std::uint32_t compiled_nccl_version = 0;
  std::uint32_t runtime_nccl_version = 0;
  bool nccl_device_headers = false;
  bool gin_headers_available = false;
  bool gin_symbols_present = false;
  bool gin_runtime_initializable = false;
  bool gin_backend_implemented = false;
  bool gin_usable = false;
  bool system_scope_atomics = false;
  std::vector<std::uint8_t> peer_access_matrix;
  std::string gin_unavailable_reason;
};

ActivationTransportCapability probe_activation_transport_capability(
    std::uint32_t world_size);

// Run the device-communicator/window portion of the GIN capability probe on
// an already established communicator.  Window registration is collective,
// so every rank in `session` must call this overload in the same order.  The
// returned value contains no native NCCL/CUDA handles.
ActivationTransportCapability probe_activation_transport_capability(
    const NcclSession &session, const CudaExecution &execution);

} // namespace megamoe::cuda

MEGAMOE_PUBLIC_API_END
