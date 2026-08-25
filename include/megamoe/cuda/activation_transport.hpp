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

namespace megamoe::cuda {

template <class T> class SelectedActivationLease;
using BFloat16ActivationLease =
    SelectedActivationLease<transport::BFloat16>;
class ActivationTransportBridge;
class ForwardMegaKernelBridge;
class ForwardWeightLease;
class SelectedOutputGradientLease;
struct BackwardValidationDownload;
namespace testing {
ForwardWeightLease make_deterministic_forward_weights_for_testing(
    std::uint32_t owned_experts, std::uint32_t global_expert_begin,
    std::uint64_t hidden_size, std::uint64_t intermediate_size,
    const CollectiveSession &session, const CudaExecution &execution);
} // namespace testing

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
  friend class ActivationTransportBridge;
  friend class DeviceActivationTransport;
  friend class LocalCudaP2pActivationTransport;
  friend class NcclSocketFallbackActivationTransport;
  friend class GinRdmaActivationTransport;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class LocalCudaP2pBackwardMegaKernel;
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
  bool enable_forward_cache = false;
  // Reserves the generation-local Backward arena.  It is deliberately opt-in
  // so the frozen Forward-only pool footprint is unchanged.
  bool enable_backward = false;
  // Reserves compact route/payload staging for the explicitly host-enqueued
  // NCCL Socket correctness backend. Disabled for the frozen local P2P path.
  bool enable_socket_fallback = false;
  // Registers each preallocated slot as an NCCL window for the explicitly
  // constructed GinRdma backend. It never enables or selects that backend by
  // itself.
  bool enable_gin_rdma = false;
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
  friend class ActivationTransportBridge;
  friend class DeviceActivationTransport;
  friend class LocalCudaP2pActivationTransport;
  friend class NcclSocketFallbackActivationTransport;
  friend class GinRdmaActivationTransport;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class LocalCudaP2pBackwardMegaKernel;
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

struct ActivationTransportTimings {
  double generation_setup_us = 0;
  double dispatch_put_us = 0;
  double relay_us = 0;
  double synthetic_return_us = 0;
  double dispatch_relay_us = 0;
  double combine_reduce_us = 0;
  double terminal_completion_us = 0;
  double total_us = 0;
};

struct ActivationTransportDeviceAudit {
  transport::ActivationTransportBackend backend =
      transport::ActivationTransportBackend::DeviceSimulation;
  std::uint64_t raw_cross_node_payload_bytes = 0;
  std::uint64_t deduplicated_cross_node_payload_bytes = 0;
  // Physical peer-link traffic. Unlike the logical node-dedup number, these
  // counters include every destination-GPU fan-out and exclude local writes.
  std::uint64_t actual_dispatch_peer_bytes = 0;
  std::uint64_t actual_return_peer_bytes = 0;
  // Shape-derived retained-storage/semantic traffic sizes. Unlike the peer
  // counters above, these are not incremented by the device data plane.
  std::uint64_t derived_route_input_bytes = 0;
  std::uint64_t derived_local_gate_bytes = 0;
  std::uint64_t derived_peer_gate_bytes = 0;
  std::uint64_t generation_memset_calls = 0;
  std::uint64_t generation_memset_bytes = 0;
  std::uint64_t kernel_launches = 0;
  std::uint64_t dispatch_kernel_blocks = 0;
  std::uint64_t return_kernel_blocks = 0;
  std::uint64_t reduce_kernel_blocks = 0;
  std::uint64_t packet_count = 0;
  std::uint64_t fanout_count = 0;
  std::uint64_t credit_stalls = 0;
  std::uint64_t peak_ring_slots = 0;
  std::uint64_t activation_d2h_bytes = 0;
  std::uint64_t activation_h2d_bytes = 0;
  std::uint64_t hot_path_allocations = 0;
};

class DeviceActivationTransportHandle {
public:
  DeviceActivationTransportHandle() = default;
  const ActivationTransportExpectation &identity() const;
  const ActivationTransportDeviceAudit &audit() const;
  const ActivationTransportTimings &timings() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceActivationTransportHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class DeviceActivationTransport;
  friend class LocalCudaP2pActivationTransport;
  friend class NcclSocketFallbackActivationTransport;
  friend class GinRdmaActivationTransport;
  friend class ActivationTransportBridge;
  friend struct ActivationTransportValidationDownload;
  friend std::vector<transport::BFloat16>
  download_activation_output_for_validation(
      const DeviceActivationTransportHandle &, const CudaExecution &);
};

class DeviceActivationTransport {
public:
  DeviceActivationTransport(const CollectiveSession &session,
                            const CudaExecution &execution,
                            const ActivationTransportResourcePool &pool);

  // Explicit correctness simulation. It can only publish a
  // DeviceSimulation audit and never selects a production backend by enum.
  DeviceActivationTransportHandle run_device_simulation_synthetic(
      const DeviceDispatchHandle &dispatch,
      const BFloat16ActivationLease &activations,
      const transport::ActivationTransportConfig &config);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

struct ActivationTransportRunOptions {
  GenerationCancellationToken cancellation;
  // Zero means no deadline. Multi-rank cancellation/deadline must be requested
  // consistently by every healthy rank in the local P2P team.
  std::chrono::microseconds deadline_after{0};
  // Training-step fast path: enqueue Forward and transfer its completion/cache
  // lease directly to Backward. The Forward handle remains owning, but audit
  // and timing access is valid only after Backward completes the handoff.
  bool defer_forward_completion_to_backward = false;
};

// A real, node-local, multi-process CUDA IPC backend. Construction exchanges
// and validates pool IPC handles over the existing node-local NCCL session.
// It fails closed unless every local peer mapping is established.
class LocalCudaP2pActivationTransport {
public:
  LocalCudaP2pActivationTransport(
      const NcclSession &session, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);

  DeviceActivationTransportHandle run_synthetic(
      const DeviceDispatchHandle &dispatch,
      const BFloat16ActivationLease &activations,
      const transport::ActivationTransportConfig &config,
      const ActivationTransportRunOptions &options = {});

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// Cross-host correctness/lower-bound backend. Payload movement is explicitly
// host-enqueued NCCL Send/Recv; this class is not a device-side GIN backend and
// never publishes a Gin audit label.
class NcclSocketFallbackActivationTransport {
public:
  NcclSocketFallbackActivationTransport(
      const NcclSession &session, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);

  DeviceActivationTransportHandle run_synthetic(
      const DeviceDispatchHandle &dispatch,
      const BFloat16ActivationLease &activations,
      const transport::ActivationTransportConfig &config,
      const ActivationTransportRunOptions &options = {});

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// True device-issued NCCL GIN data plane. The transport slice derives the
// homogeneous rail/local team at runtime, partitions ingress by source node,
// and performs same-rail ingress plus node-local CUDA-IPC fan-out. The current
// public operation remains the synthetic validation transform; cross-node
// Forward/Backward FC integration is not implied by this type.
class GinRdmaActivationTransport {
public:
  GinRdmaActivationTransport(
      const NcclSession &session, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);

  DeviceActivationTransportHandle run_synthetic(
      const DeviceDispatchHandle &dispatch,
      const BFloat16ActivationLease &activations,
      const transport::ActivationTransportConfig &config,
      const ActivationTransportRunOptions &options = {});

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

enum class ForwardCacheMode : std::uint8_t {
  CacheFc1InputOutput,
  NoFc1OutputCache,
};

enum class ForwardKernelSpecialization : std::uint8_t {
  Sm90WgmmaTma7168x3072,
  CudaCoreCorrectnessFallback,
};

struct ForwardMegaKernelConfig {
  std::uint64_t hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  ForwardCacheMode cache_mode = ForwardCacheMode::NoFc1OutputCache;
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
  friend class ForwardMegaKernelBridge;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class LocalCudaP2pBackwardMegaKernel;
  friend ForwardWeightLease
  testing::make_deterministic_forward_weights_for_testing(
      std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t,
      const CollectiveSession &, const CudaExecution &);
  friend struct ForwardWeightStorageHashAccess;
  friend struct ForwardBackwardWeightAccess;
};

struct DeviceStorageHash128 {
  std::uint64_t low = 0;
  std::uint64_t high = 0;
  std::uint64_t element_count = 0;
};

struct ForwardWeightStorageHashes {
  DeviceStorageHash128 w1;
  DeviceStorageHash128 w2;
  std::vector<DeviceStorageHash128> sampled_w1_experts;
  std::vector<DeviceStorageHash128> sampled_w2_experts;
  std::vector<DeviceStorageHash128> sampled_w2_output_rows;
};

struct BackwardStorageHashes {
  DeviceStorageHash128 dx;
  DeviceStorageHash128 dw1;
  DeviceStorageHash128 dw2;
  DeviceStorageHash128 dgate;
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
  double total_us = 0;
};

struct ForwardMegaKernelAudit {
  ForwardKernelSpecialization specialization =
      ForwardKernelSpecialization::CudaCoreCorrectnessFallback;
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
  // Owning one-shot handoff used by Backward. It transfers the already
  // acquired Forward dispatch consumer; it never exposes a kernel view or
  // acquires a second consumer event/slot lease.
  DeviceDispatchConsumerLease transfer_consumer_to_backward() const;
  struct Impl;
  explicit DeviceForwardHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class ForwardMegaKernelBridge;
  friend std::vector<transport::BFloat16>
  download_forward_output_for_validation(const DeviceForwardHandle &,
                                         const CudaExecution &);
  friend std::vector<transport::BFloat16>
  download_forward_cache_for_validation(const DeviceForwardHandle &,
                                        const CudaExecution &);
  friend bool testing_forward_cache_ranges_disjoint(
      const DeviceForwardHandle &, const DeviceForwardHandle &);
  friend class SelectedOutputGradientLease;
  friend class LocalCudaP2pBackwardMegaKernel;
  friend struct BackwardMegaKernelBridge;
  friend SelectedOutputGradientLease
  testing_upload_output_gradient_for_backward(
      const DeviceForwardHandle &, const std::vector<transport::BFloat16> &,
      const CudaExecution &);
  friend SelectedOutputGradientLease
  testing_reissue_output_gradient_for_benchmark(
      const DeviceForwardHandle &, const SelectedOutputGradientLease &,
      const CudaExecution &);
};

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
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

std::vector<transport::BFloat16>
download_forward_output_for_validation(const DeviceForwardHandle &handle,
                                       const CudaExecution &execution);

std::vector<transport::BFloat16>
download_forward_cache_for_validation(const DeviceForwardHandle &handle,
                                      const CudaExecution &execution);

// Test-only predicate. It validates real device address intervals internally
// without exposing either native pointer to the caller.
bool testing_forward_cache_ranges_disjoint(const DeviceForwardHandle &left,
                                           const DeviceForwardHandle &right);

// Immutable autograd -> Backward handoff.  dY is BF16 [tokens, hidden] and is
// bound to the exact Forward generation/slot.  No native pointer is exposed.
class SelectedOutputGradientLease {
public:
  SelectedOutputGradientLease() = default;
  std::uint64_t num_tokens() const;
  std::uint64_t hidden_size() const;
  std::uint64_t generation() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit SelectedOutputGradientLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pBackwardMegaKernel;
  friend struct BackwardMegaKernelBridge;
  friend SelectedOutputGradientLease
  testing_upload_output_gradient_for_backward(
      const DeviceForwardHandle &, const std::vector<transport::BFloat16> &,
      const CudaExecution &);
  friend SelectedOutputGradientLease
  testing_reissue_output_gradient_for_benchmark(
      const DeviceForwardHandle &, const SelectedOutputGradientLease &,
      const CudaExecution &);
};

enum class BackwardWeightGradientDType : std::uint8_t { BFloat16 };

struct BackwardMegaKernelConfig {
  std::uint64_t hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  BackwardWeightGradientDType weight_gradient_dtype =
      BackwardWeightGradientDType::BFloat16;
};

enum class BackwardKernelSpecialization : std::uint8_t {
  Sm90WgmmaTma7168x3072,
  CudaCoreCorrectnessFallback,
};

struct BackwardMegaKernelTimings {
  double forward_to_backward_handoff_us = 0;
  double backward_setup_to_launch_host_us = 0;
  double forward_consumer_drain_host_us = 0;
  double final_consumer_drain_host_us = 0;
  double dy_dispatch_us = 0;
  double dv_swiglu_backward_us = 0;
  double dgate_reduce_us = 0;
  double transpose_us = 0;
  double dx_us = 0;
  double dw1_us = 0;
  double dw2_us = 0;
  double dw1_ready_to_start_us = 0;
  double dw2_ready_to_start_us = 0;
  double dw1_max_expert_ready_to_start_us = 0;
  double dw2_max_expert_ready_to_start_us = 0;
  double dw1_tail_us = 0;
  double dw2_tail_us = 0;
  double dw1_overlap_dv_dx_us = 0;
  double dw2_overlap_dv_dx_us = 0;
  double dx_combine_reduce_us = 0;
  double total_us = 0;
};

struct BackwardMegaKernelAudit {
  BackwardKernelSpecialization specialization =
      BackwardKernelSpecialization::CudaCoreCorrectnessFallback;
  std::uint64_t persistent_kernel_launches = 0;
  std::uint64_t wgmma_instructions = 0;
  std::uint64_t tma_loads = 0;
  std::uint64_t dv_tasks = 0;
  std::uint64_t dx_tasks = 0;
  std::uint64_t dw1_tasks = 0;
  std::uint64_t dw2_tasks = 0;
  std::uint64_t actual_dy_peer_bytes = 0;
  std::uint64_t actual_dx_peer_bytes = 0;
  std::uint64_t hot_path_allocations = 0;
  std::uint64_t workspace_bytes = 0;
  bool recomputed_fc1 = false;
  std::uint64_t dispatch_consumer_acquires = 0;
  std::uint64_t dispatch_consumer_releases = 0;
  BackwardWeightGradientDType weight_gradient_dtype =
      BackwardWeightGradientDType::BFloat16;
};

class DeviceBackwardHandle {
public:
  DeviceBackwardHandle() = default;
  const ActivationTransportExpectation &identity() const;
  const BackwardMegaKernelAudit &audit() const;
  const BackwardMegaKernelTimings &timings() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceBackwardHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pBackwardMegaKernel;
  friend struct BackwardValidationDownload;
  friend struct BackwardStorageHashAccess;
  friend BackwardValidationDownload download_backward_for_validation(
      const DeviceBackwardHandle &, const CudaExecution &);
  friend bool testing_backward_storage_ranges_disjoint(
      const DeviceBackwardHandle &, const DeviceBackwardHandle &);
  friend std::size_t
  testing_backward_slot_index(const DeviceBackwardHandle &);
};

struct BackwardValidationDownload {
  std::vector<transport::BFloat16> dx;
  std::vector<float> dgate;
  std::vector<transport::BFloat16> dw1;
  std::vector<transport::BFloat16> dw2;
  std::vector<transport::BFloat16> sampled_ab;
  std::vector<transport::BFloat16> sampled_dab;
};

class LocalCudaP2pBackwardMegaKernel {
public:
  LocalCudaP2pBackwardMegaKernel(
      const NcclSession &session, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);

  DeviceBackwardHandle run(
      const DeviceForwardHandle &forward,
      const SelectedOutputGradientLease &output_gradient,
      const BackwardMegaKernelConfig &config,
      const ActivationTransportRunOptions &options = {});

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

BackwardValidationDownload download_backward_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution);

// Test-only predicates expose slot identity/isolation without exposing native
// device addresses or an owner-free kernel view.
bool testing_backward_storage_ranges_disjoint(
    const DeviceBackwardHandle &left, const DeviceBackwardHandle &right);
std::size_t testing_backward_slot_index(const DeviceBackwardHandle &handle);

// Explicit test/convenience upload into the Forward generation's preallocated
// dY slot.  Production autograd/router integration will issue the same typed
// lease without a host roundtrip.
SelectedOutputGradientLease testing_upload_output_gradient_for_backward(
    const DeviceForwardHandle &forward,
    const std::vector<transport::BFloat16> &host,
    const CudaExecution &execution);

SelectedOutputGradientLease testing_reissue_output_gradient_for_benchmark(
    const DeviceForwardHandle &forward,
    const SelectedOutputGradientLease &storage,
    const CudaExecution &execution);

struct ActivationTransportValidationDownload {
  std::vector<transport::BFloat16> contribution_by_route;
  std::vector<transport::BFloat16> combined_tokens;
  std::vector<transport::BFloat16> canonical_expert_rows;
  transport::TransportProtocolAudit protocol;
};

ActivationTransportValidationDownload download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution);

std::vector<transport::BFloat16> download_activation_output_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution);

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

namespace testing {
struct GinPutSignalProbeResult {
  std::uint64_t payload_bytes = 0;
  std::uint32_t rail_rank = 0;
  std::uint32_t rail_world = 0;
  std::uint32_t local_rank = 0;
  std::uint64_t received_hash = 0;
  std::uint64_t expected_hash = 0;
};

// Real device-issued GIN rail put with remote indexed signal and local
// completion counter. This is a lifecycle/data-movement probe, not the full
// activation backend.
GinPutSignalProbeResult run_distributed_gin_put_signal_probe(
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t payload_bytes = 1024);

BFloat16ActivationLease upload_bfloat16_activations_for_testing(
    const std::vector<transport::BFloat16> &host,
    std::uint64_t num_tokens, std::uint64_t hidden,
    const RankCollectiveContext &context, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool);

ForwardWeightLease make_deterministic_forward_weights_for_testing(
    std::uint32_t owned_experts, std::uint32_t global_expert_begin,
    std::uint64_t hidden_size, std::uint64_t intermediate_size,
    const CollectiveSession &session, const CudaExecution &execution);

// Validation-only hash of the complete BF16 storage which the Forward kernel
// will consume. The device performs a fixed-size chunk reduction over actual
// bytes; only the two 64-bit results per tensor are copied to the host.
ForwardWeightStorageHashes hash_forward_weight_storage_for_validation(
    const ForwardWeightLease &weights, const CudaExecution &execution);

// Validation-only fixed-order hashes over the complete device gradient
// buffers produced by the persistent Backward kernel. No bulk gradient
// download is performed.
BackwardStorageHashes hash_backward_storage_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution);

// Recomputes the hash from the live device allocation even when the handle
// carries a completion-time hash. This is intentionally validation-only and
// is used to prove that a later live generation did not overwrite the slot.
BackwardStorageHashes rehash_backward_storage_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution);

enum class BackwardStorageField : std::uint8_t { Dx, Dw1, Dw2, DGate };

// Corruption sensitivity check for actual device gradient storage. Validation
// only; the offset is measured in bytes and the caller must restore mutations.
void flip_backward_storage_byte_for_validation(
    const DeviceBackwardHandle &handle, BackwardStorageField field,
    std::uint64_t byte_offset, const CudaExecution &execution);

// Compatibility wrapper retained inside the validation namespace.
void flip_backward_dx_byte_for_validation(
    const DeviceBackwardHandle &handle, std::uint64_t byte_offset,
    const CudaExecution &execution);

struct EmptyExpertGradientZeroCheck {
  std::uint64_t dw1_nonzero_elements = 0;
  std::uint64_t dw2_nonzero_elements = 0;
  std::uint64_t dw1_zero_store_tasks_expected = 0;
  std::uint64_t dw1_zero_store_tasks_completed = 0;
  std::uint64_t dw2_zero_store_tasks_expected = 0;
  std::uint64_t dw2_zero_store_tasks_completed = 0;
};

// Scans the complete dW intervals of one known-empty local expert on device.
EmptyExpertGradientZeroCheck check_empty_expert_gradients_for_validation(
    const DeviceBackwardHandle &handle, std::uint32_t local_expert,
    const CudaExecution &execution);

// One-time Backward session setup. It builds device-resident W1^T/W2^T and
// their TMA descriptors; Forward-only callers pay neither memory nor setup.
void prepare_forward_weights_for_backward(
    const ForwardWeightLease &weights, const CudaExecution &execution);

// Executes a true device-issued store through a peer mapping and observes a
// system-scope release/acquire signal. It is a capability/correctness probe,
// not the cross-node backend.
bool run_local_p2p_put_signal_probe(std::uint32_t source_device,
                                    std::uint32_t destination_device,
                                    std::uint64_t elements);
double measure_local_p2p_put_bandwidth_gbps(std::uint32_t source_device,
                                            std::uint32_t destination_device,
                                            std::uint64_t bytes,
                                            std::uint32_t warmup,
                                            std::uint32_t measured);
// NCCL-free sanitizer harness. It launches the exact production main-shape
// Forward persistent kernel, replacing only Notify/bootstrap with an explicit
// initialized world-1 dispatch fixture.
bool run_forward_world1_sanitizer_harness();
bool run_backward_world1_sanitizer_harness();
} // namespace testing

} // namespace megamoe::cuda
