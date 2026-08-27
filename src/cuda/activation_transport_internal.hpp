#pragma once

#include "cuda/device_notify_internal.hpp"
#include "cuda/profiling/device_timeline_types.hpp"
#include "transport/activation_protocol_testing.hpp"
#define private public
#include "megamoe/cuda/activation_transport.hpp"
#undef private

namespace megamoe::cuda {

namespace internal {
struct ExternalCudaProducerState;
}

struct ExternalCudaProducer::State {
  std::shared_ptr<const internal::ExternalCudaProducerState> backend;
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
  std::uint64_t actual_dispatch_peer_bytes = 0;
  std::uint64_t actual_return_peer_bytes = 0;
  std::uint64_t raw_cross_node_route_bytes = 0;
  std::uint64_t gin_dispatch_payload_bytes = 0;
  std::uint64_t gin_return_payload_bytes = 0;
  std::uint64_t gin_dispatch_put_count = 0;
  std::uint64_t gin_return_put_count = 0;
  std::uint64_t gin_signal_count = 0;
  std::uint64_t gin_metadata_bytes = 0;
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
  friend std::vector<transport::BFloat16>
  download_activation_output_for_validation(
      const DeviceActivationTransportHandle &, const CudaExecution &);
};

class DeviceActivationTransport {
public:
  DeviceActivationTransport(const CollectiveSession &, const CudaExecution &,
                            const ActivationTransportResourcePool &);
  DeviceActivationTransportHandle run_device_simulation_synthetic(
      const DeviceDispatchHandle &, const BFloat16ActivationLease &,
      const transport::ActivationTransportConfig &);
private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

class LocalCudaP2pActivationTransport {
public:
  LocalCudaP2pActivationTransport(const NcclSession &, const CudaExecution &,
                                  const ActivationTransportResourcePool &);
  DeviceActivationTransportHandle run_transport_slice(
      const DeviceDispatchHandle &, const BFloat16ActivationLease &,
      const transport::ActivationTransportConfig &,
      const ActivationTransportRunOptions & = {});
private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

class NcclSocketFallbackActivationTransport {
public:
  NcclSocketFallbackActivationTransport(
      const NcclSession &, const CudaExecution &,
      const ActivationTransportResourcePool &);
  DeviceActivationTransportHandle run_transport_slice(
      const DeviceDispatchHandle &, const BFloat16ActivationLease &,
      const transport::ActivationTransportConfig &,
      const ActivationTransportRunOptions & = {});
private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

class GinRdmaActivationTransport {
public:
  GinRdmaActivationTransport(const NcclSession &, const CudaExecution &,
                             const ActivationTransportResourcePool &);
  DeviceActivationTransportHandle run_transport_slice(
      const DeviceDispatchHandle &, const BFloat16ActivationLease &,
      const transport::ActivationTransportConfig &,
      const ActivationTransportRunOptions & = {});
private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
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
struct BackwardStageStorageHashes {
  DeviceStorageHash128 canonical_x;
  DeviceStorageHash128 canonical_dy;
  DeviceStorageHash128 dy_dw2_reduction_extent;
  DeviceStorageHash128 active_ab;
  DeviceStorageHash128 s;
  DeviceStorageHash128 dv;
  DeviceStorageHash128 u_transpose;
  DeviceStorageHash128 dab;
  DeviceStorageHash128 dab_transpose;
  DeviceStorageHash128 dgate_partial;
  DeviceStorageHash128 dgate_canonical;
  DeviceStorageHash128 dgate_final;
  DeviceStorageHash128 dw2;
  DeviceStorageHash128 dw1;
  DeviceStorageHash128 return_signal_epoch;
  DeviceStorageHash128 route_return_coverage;
  std::uint64_t incoming_rows = 0;
  std::uint64_t padded_reduction_rows = 0;
  std::uint64_t route_return_coverage_count = 0;
};
struct BackwardHashAgreement {
  bool matches = false;
  std::uint32_t origin_rank = 0;
  std::uint32_t field = 0;
  DeviceStorageHash128 expected;
  DeviceStorageHash128 actual;
};

struct BackwardValidationDownload {
  std::vector<transport::BFloat16> dx;
  std::vector<float> dgate;
  std::vector<transport::BFloat16> dw1;
  std::vector<transport::BFloat16> dw2;
  std::vector<transport::BFloat16> sampled_ab;
  std::vector<transport::BFloat16> sampled_dab;
};

struct ActivationTransportValidationDownload {
  std::vector<transport::BFloat16> contribution_by_route;
  std::vector<transport::BFloat16> combined_tokens;
  std::vector<transport::BFloat16> canonical_expert_rows;
  transport::TransportProtocolAudit protocol;
};

class ActivationTransportBridge {
public:
  static DeviceDispatchKernelView
  view(DeviceDispatchConsumerLease &consumer);
  static const float *gate_pointer(const DeviceDispatchHandle &dispatch);
  static void wait_for_gates(const DeviceDispatchHandle &dispatch,
                             const CudaExecution &execution);
  static void submit(DeviceDispatchConsumerLease &consumer);
  static const RankCollectiveContext &
  context(const DeviceDispatchConsumerLease &consumer);
  static BFloat16ActivationLease upload_for_testing(
      const std::vector<transport::BFloat16> &host,
      std::uint64_t num_tokens, std::uint64_t hidden,
      const RankCollectiveContext &context, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);
  static BFloat16ActivationLease reissue_for_benchmark(
      const BFloat16ActivationLease &storage,
      const RankCollectiveContext &context, const CudaExecution &execution);
  static ActivationTransportValidationDownload download_for_validation(
      const DeviceActivationTransportHandle &handle,
      const CudaExecution &execution);
};

std::vector<transport::BFloat16> download_forward_output_for_validation(
    const DeviceForwardHandle &, const CudaExecution &);
std::vector<transport::BFloat16> download_forward_cache_for_validation(
    const DeviceForwardHandle &, const CudaExecution &);
bool testing_forward_cache_ranges_disjoint(const DeviceForwardHandle &,
                                           const DeviceForwardHandle &);
BackwardValidationDownload download_backward_for_validation(
    const DeviceBackwardHandle &, const CudaExecution &);
bool testing_backward_storage_ranges_disjoint(const DeviceBackwardHandle &,
                                              const DeviceBackwardHandle &);
std::size_t testing_backward_slot_index(const DeviceBackwardHandle &);
SelectedOutputGradientLease testing_upload_output_gradient_for_backward(
    const DeviceForwardHandle &, const std::vector<transport::BFloat16> &,
    const CudaExecution &);
SelectedOutputGradientLease testing_reissue_output_gradient_for_benchmark(
    const DeviceForwardHandle &, const SelectedOutputGradientLease &,
    const CudaExecution &);
ActivationTransportValidationDownload
download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &, const CudaExecution &);
std::vector<transport::BFloat16>
download_activation_output_for_validation(
    const DeviceActivationTransportHandle &, const CudaExecution &);

namespace testing {
#if MEGAMOE_DEVICE_TIMELINE
profiling::DeviceTimelineSnapshot
download_forward_device_timeline(const DeviceForwardHandle &);
profiling::DeviceTimelineSnapshot
download_backward_device_timeline(const DeviceBackwardHandle &);
#endif
struct GinPutSignalProbeResult {
  std::uint64_t payload_bytes = 0;
  std::uint32_t rail_rank = 0;
  std::uint32_t rail_world = 0;
  std::uint32_t local_rank = 0;
  std::uint64_t received_hash = 0;
  std::uint64_t expected_hash = 0;
};
GinPutSignalProbeResult run_distributed_gin_put_signal_probe(
    const NcclSession &, const CudaExecution &, std::uint64_t = 1024);
BFloat16ActivationLease upload_bfloat16_activations_for_testing(
    const std::vector<transport::BFloat16> &, std::uint64_t, std::uint64_t,
    const RankCollectiveContext &, const CudaExecution &,
    const ActivationTransportResourcePool &);
ForwardWeightLease make_deterministic_forward_weights_for_testing(
    std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t,
    const CollectiveSession &, const CudaExecution &);
ForwardWeightStorageHashes hash_forward_weight_storage_for_validation(
    const ForwardWeightLease &, const CudaExecution &);
BackwardStorageHashes hash_backward_storage_for_validation(
    const DeviceBackwardHandle &, const CudaExecution &);
BackwardStorageHashes rehash_backward_storage_for_validation(
    const DeviceBackwardHandle &, const CudaExecution &);
BackwardStageStorageHashes hash_backward_stage_storage_for_validation(
    const DeviceBackwardHandle &, const CudaExecution &);
BackwardHashAgreement agree_backward_hashes_for_validation(
    const DeviceBackwardHandle &, const BackwardStorageHashes &,
    const BackwardStorageHashes &, const CudaExecution &);
enum class BackwardStorageField : std::uint8_t { Dx, Dw1, Dw2, DGate };
void flip_backward_storage_byte_for_validation(
    const DeviceBackwardHandle &, BackwardStorageField, std::uint64_t,
    const CudaExecution &);
void flip_backward_dx_byte_for_validation(
    const DeviceBackwardHandle &, std::uint64_t, const CudaExecution &);
struct EmptyExpertGradientZeroCheck {
  std::uint64_t dw1_nonzero_elements = 0;
  std::uint64_t dw2_nonzero_elements = 0;
  std::uint64_t dw1_zero_store_tasks_expected = 0;
  std::uint64_t dw1_zero_store_tasks_completed = 0;
  std::uint64_t dw2_zero_store_tasks_expected = 0;
  std::uint64_t dw2_zero_store_tasks_completed = 0;
};
EmptyExpertGradientZeroCheck check_empty_expert_gradients_for_validation(
    const DeviceBackwardHandle &, std::uint32_t, const CudaExecution &);
bool run_local_p2p_put_signal_probe(std::uint32_t, std::uint32_t,
                                    std::uint64_t);
double measure_local_p2p_put_bandwidth_gbps(
    std::uint32_t, std::uint32_t, std::uint64_t, std::uint32_t,
    std::uint32_t);
} // namespace testing

} // namespace megamoe::cuda
