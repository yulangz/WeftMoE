#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "weftmoe/cuda/device_notify.hpp"
#include "weftmoe/cuda/nccl_count_exchange.hpp"
#include "weftmoe/cuda/tensor_storage.hpp"
#include "weftmoe/transport/activation_protocol.hpp"
#include "weftmoe/version.hpp"

WEFTMOE_PUBLIC_API_BEGIN

struct CUstream_st;

namespace weftmoe::cuda {

// Identity of the immutable generated Forward profile, checked before pool
// preparation when applying an offline autotune table.
const char *forward_tuning_id();
std::pair<std::uint32_t, std::uint32_t> forward_shape();

template <class T> class SelectedActivationLease;
using BFloat16ActivationLease = SelectedActivationLease<transport::BFloat16>;
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
  // Allocate fixed communication rings for the explicitly constructed GinRdma
  // backend. Their windows survive compatible compute-arena growth. This flag
  // never selects the backend by itself.
  bool enable_gin_rdma = false;
  // Enable reusable Backward metadata and capacity-sized peer workspaces.
  // Long-lived X/A/B checkpoints and final outputs belong to
  // ForwardTensorStorage, independently of these execution slots.
  bool enable_backward_state = false;
  // Opt-in bounded static Forward workspace. Zero retains the arena path;
  // otherwise this is the source-token window size. Training uses an eager
  // Notify/allocation boundary and independent canonical checkpoint storage.
  std::uint32_t static_forward_window_tokens = 0;
  // Source-side Return/Reduce capacity, independent of destination X/S/C.
  std::uint32_t static_forward_contribution_windows = 2;
  // Initial dR capacity, independent of worst-case routing metadata. Zero
  // reserves twice the source-route capacity. Payloads grow only on overflow;
  // changing the actual row count never shrinks their allocation or IPC maps.
  std::uint64_t backward_peer_capacity_rows = 0;
  // Select DT explicitly for single-node inference. False keeps the PF path;
  // a small pool never opts into DT. The complete capacity must fit one window.
  bool decode_only = false;
  // Compact source packets and packed compute rings; single-node PF FP8 only.
  bool fp8_workspace = false;
  // Eager training submits successive uses of a workspace on its execution
  // stream. Handles retain tensors independently until GPU completion.
  bool stream_ordered_training = false;
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
  // Whether this compiled PF entry plans in a separate Forward kernel.
  bool separates_forward_notify() const;
  // Terminal phase boundary: first quiesce all users and discard captured
  // graphs on every rank, then retire imports, then collectively agree before
  // destroying any exporting pool. No further execution may use this pool.
  void retire_peer_mappings();
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ActivationTransportResourcePool(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend ActivationTransportResourcePool
  make_activation_transport_pool(const CollectiveSession &,
                                 const CudaExecution &,
                                 ActivationTransportPoolConfig);
  friend ActivationTransportResourcePool prepare_activation_transport_pool(
      const CollectiveSession &, const CudaExecution &,
      ActivationTransportPoolConfig, const DeviceDispatchHandle &,
      const ActivationTransportResourcePool &);
  friend class LocalCudaP2pForwardMegaKernel;
  friend class GinRdmaForwardMegaKernel;
  friend class ExternalCudaProducer;
};

ActivationTransportResourcePool
make_activation_transport_pool(const CollectiveSession &session,
                               const CudaExecution &execution,
                               ActivationTransportPoolConfig config);

// Combine explicit capacities with the current Notify requirements. Capture
// preparation without Notify must provide a destination-row capacity. Reuse
// compatible storage; live Forward/Backward handles retain older pools.
ActivationTransportResourcePool prepare_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config,
    const DeviceDispatchHandle &dispatch = {},
    const ActivationTransportResourcePool &existing = {});

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
  // Preserve X and A/B for Backward (BF16 or row-scaled FP8). Forward
  // consumes the same represented A/B values. Requires enable_backward_state.
  bool retain_backward_state = false;
  // Internal preparation policy selected by the FP8 PF backend. X is retained
  // as a row-scaled FP8 checkpoint; fc1_input is absent.
  bool fp8_x_checkpoint = false;
  // The FP8 checkpoint omits the BF16 fc1_output binding too.
  bool fp8_ab_checkpoint = false;
  // Caller-owned storage is prepared before entering Forward. Allocation
  // exceptions propagate locally; recovery belongs to the application.
  // With no binding, the training-ring backend allocates the tape.
  ForwardTensorStorage tensors{};
  // The replay stream itself orders output consumers and serialized workspace
  // reuse, so observable events can be recorded once during graph release.
  bool serialized_graph_replay = false;
  // A serialized inference graph may bind its stable application route inputs
  // directly while reusing a shape-compatible Dispatch handle.  This avoids a
  // second set of route/gate staging copies in CUDA Graph replay.  Both buffers
  // must be provided together; training and the legacy split paths reject them.
  CudaTensorBuffer inference_routes{}; // INT32 [tokens, top_k]
  CudaTensorBuffer inference_gates{};  // FP32 [tokens, top_k]
  // Eager split planning knows the exact tape extent only inside run().
  std::function<ForwardTensorStorage(const ForwardMegaKernelConfig &)> tensor_factory{};
};

// Caller-prequantized E4M3FN weights. Logical W1=[E,2I,H], W2=[E,H,I].
// Each FP32 scale multiplies a logical 128x128 block. Optional transposes
// are required for Backward and contain the same quantized values, transposed.
// Buffers are immutable while any Forward/Backward handle retains the lease.
struct FP8WeightBuffers {
  CudaTensorBuffer w1, s1, w2, s2;
  CudaTensorBuffer w1_t, s1_t, w2_t, s2_t;
  std::uint64_t version = 0;
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
  double device_kernel_us = 0;
  double route_exchange_us = 0;
  double dispatch_us = 0;
  double fc1_swiglu_us = 0;
  double fc2_us = 0;
  double before_combine_us = 0;
  double combine_return_us = 0;
  double reduce_us = 0;
  double after_combine_us = 0;
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
  // Zero when standalone cluster occupancy was not queried for this launch.
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
  const ForwardTensorStorage &tensors() const;
  // enqueue() returns before GPU completion. wait() collects diagnostics;
  // saved tensors remain valid. Stream-ordered training hands the workspace
  // back at submission, otherwise wait() releases it. Captured handles reserve
  // their slot until destruction. wait_on() inserts only a dependency.
  void wait() const;
  // Nonblocking host completion check for stream-ordered training handles.
  bool ready() const;
  void wait_on(CUstream_st *stream) const;
  // Serialize reuse of the persistent execution workspace on another stream.
  // Unlike wait_on(), this also waits for status copies and the terminal
  // workspace-complete event, but it never blocks the calling host thread.
  void wait_workspace_on(CUstream_st *stream) const;
  // After instantiating and executing an inference graph once, relinquish its
  // workspace lease. The graph and this handle must remain alive. The caller
  // MUST serialize every replay and every other use of this pool on one stream;
  // concurrent replay is forbidden. Saved training state cannot use this mode.
  void release_serialized_graph_workspace() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceForwardHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class LocalCudaP2pForwardMegaKernel;
  friend class GinRdmaForwardMegaKernel;
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

  CudaRouteInput
  publish_selected_experts(const std::int32_t *device_data,
                           std::uint64_t capacity_elements,
                           std::uint64_t num_tokens, std::uint32_t top_k,
                           std::shared_ptr<const void> allocation_owner) const;
  CudaGateInput
  publish_gates(const float *device_data, std::uint64_t capacity_elements,
                const CudaRouteInput &exact_routes,
                std::shared_ptr<const void> allocation_owner) const;
  BFloat16ActivationLease
  publish_activations(const transport::BFloat16 *device_data,
                      std::uint64_t capacity_elements, std::uint64_t num_tokens,
                      std::uint64_t hidden_size,
                      std::shared_ptr<const void> allocation_owner) const;
  // dY is caller-owned and does not reserve a second transport-pool slot.
  // The returned lease records producer-stream readiness and retains the
  // allocation until Backward has consumed it.
  BFloat16OutputGradientLease
  publish_output_gradients(const transport::BFloat16 *device_data,
                           std::uint64_t capacity_elements,
                           std::uint64_t num_tokens, std::uint64_t hidden_size,
                           std::shared_ptr<const void> allocation_owner) const;
  ForwardWeightLease publish_weights(
      const transport::BFloat16 *w1, std::uint64_t w1_capacity_elements,
      std::shared_ptr<const void> w1_owner, const transport::BFloat16 *w2,
      std::uint64_t w2_capacity_elements, std::shared_ptr<const void> w2_owner,
      std::uint32_t owned_experts, std::uint32_t global_expert_begin,
      std::uint64_t hidden_size, std::uint64_t intermediate_size) const;
  // Legacy caller-owned packing scratch is validated and retained by the lease
  // for compatibility. Forward's TMA maps now read W1 directly and form the
  // gate/up layout in shared memory; no packed weight values are cached.
  ForwardWeightLease publish_weights(
      const transport::BFloat16 *w1, std::uint64_t w1_capacity_elements,
      std::shared_ptr<const void> w1_owner, const transport::BFloat16 *w2,
      std::uint64_t w2_capacity_elements, std::shared_ptr<const void> w2_owner,
      std::uint32_t owned_experts, std::uint32_t global_expert_begin,
      std::uint64_t hidden_size, std::uint64_t intermediate_size,
      const CudaTensorBuffer &packed_w1) const;
  ForwardWeightLease publish_fp8_weights(
      const FP8WeightBuffers &, std::uint32_t owned_experts,
      std::uint32_t global_expert_begin, std::uint64_t hidden_size,
      std::uint64_t intermediate_size) const;

private:
  struct State;
  explicit ExternalCudaProducer(std::shared_ptr<const State> state)
      : state_(std::move(state)) {}
  std::shared_ptr<const State> state_;
  friend ExternalCudaProducer
  make_external_cuda_producer(RankCollectiveContext, std::uint32_t,
                              CUstream_st *, std::shared_ptr<const void>,
                              std::string);
};

ExternalCudaProducer make_external_cuda_producer(
    RankCollectiveContext context, std::uint32_t device_ordinal,
    CUstream_st *producer_stream, std::shared_ptr<const void> stream_owner,
    std::string identity);

class LocalCudaP2pForwardMegaKernel {
public:
  LocalCudaP2pForwardMegaKernel(const NcclSession &session,
                                const CudaExecution &execution,
                                const ActivationTransportResourcePool &pool);

  DeviceForwardHandle
  enqueue(const DeviceDispatchHandle &dispatch,
          const BFloat16ActivationLease &activations,
          const ForwardWeightLease &weights,
          const ForwardMegaKernelConfig &config,
          const ActivationTransportRunOptions &options = {});

  DeviceForwardHandle run(const DeviceDispatchHandle &dispatch,
                          const BFloat16ActivationLease &activations,
                          const ForwardWeightLease &weights,
                          const ForwardMegaKernelConfig &config,
                          const ActivationTransportRunOptions &options = {});

private:
  LocalCudaP2pForwardMegaKernel(const NcclSession &, const CudaExecution &,
                                const ActivationTransportResourcePool &,
                                bool use_gin);
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class GinRdmaForwardMegaKernel;
};

// Cross-node same-rail GIN Forward. It shares the frozen WGMMA/TMA compute
// specialization with LocalCudaP2pForwardMegaKernel, but has a distinct
// construction path and never falls back to local simulation or Socket.
class GinRdmaForwardMegaKernel {
public:
  GinRdmaForwardMegaKernel(const NcclSession &, const CudaExecution &,
                           const ActivationTransportResourcePool &);
  DeviceForwardHandle enqueue(const DeviceDispatchHandle &,
                              const BFloat16ActivationLease &,
                              const ForwardWeightLease &,
                              const ForwardMegaKernelConfig &,
                              const ActivationTransportRunOptions & = {});

  DeviceForwardHandle run(const DeviceDispatchHandle &,
                          const BFloat16ActivationLease &,
                          const ForwardWeightLease &,
                          const ForwardMegaKernelConfig &,
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

ActivationTransportCapability
probe_activation_transport_capability(std::uint32_t world_size);

// Run the device-communicator/window portion of the GIN capability probe on
// an already established communicator.  Window registration is collective,
// so every rank in `session` must call this overload in the same order.  The
// returned value contains no native NCCL/CUDA handles.
ActivationTransportCapability
probe_activation_transport_capability(const NcclSession &session,
                                      const CudaExecution &execution);

} // namespace weftmoe::cuda

WEFTMOE_PUBLIC_API_END
