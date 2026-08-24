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
};

struct ActivationTransportPoolUsage {
  std::uint32_t slots_available = 0;
  std::uint32_t slots_in_use = 0;
  std::uint32_t slots_quarantined = 0;
  std::uint64_t device_bytes = 0;
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

namespace testing {
BFloat16ActivationLease upload_bfloat16_activations_for_testing(
    const std::vector<transport::BFloat16> &host,
    std::uint64_t num_tokens, std::uint64_t hidden,
    const RankCollectiveContext &context, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool);

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
} // namespace testing

} // namespace megamoe::cuda
