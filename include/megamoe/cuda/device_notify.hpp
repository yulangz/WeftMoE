#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/api/notify_dispatch.hpp"
#include "megamoe/cuda/control_protocol.hpp"

namespace megamoe::cuda {

class CudaExecution;
class ExternalCudaProducer;
class CudaLocalNotifyHandle;
class DeviceDispatchHandle;
using CudaDispatchHandle = DeviceDispatchHandle;
template <class T> class SelectedExpertRouteLease;
using CudaRouteInput = SelectedExpertRouteLease<std::int32_t>;
template <class T> class SelectedExpertGateLease;
using CudaGateInput = SelectedExpertGateLease<float>;
class DeviceDispatchConsumerLease;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
class DeviceDispatchTestBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
class ActivationTransportBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
class DeviceRouteProvenanceBridge;
class DeviceGateProvenanceBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
class CudaNotifyWorkspace;
class CudaNotifyOperation;
class CudaNotifyPreflight;
#endif
namespace internal {
struct CudaExecutionState;
struct ExternalCudaProducerState;
std::shared_ptr<const CudaExecutionState>
checked_execution_state(const CudaExecution &execution);
}

// CUDA/NCCL native handles remain private to the implementation.
class CudaExecution {
public:
  CudaExecution() = default;
  const ExecutionToken &token() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaExecution(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend CudaExecution make_cuda_execution(std::string, std::uint32_t);
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class DeviceNotifyBridge;
#endif
  friend class SelectedExpertRouteLease<std::int32_t>;
  friend class DeviceDispatchConsumerLease;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class CudaNotifyWorkspace;
  friend class CudaNotifyOperation;
#endif
  friend class NcclAllGatherCountExchange;
  friend std::shared_ptr<const internal::CudaExecutionState>
  internal::checked_execution_state(const CudaExecution &);
};

CudaExecution make_cuda_execution(std::string identity,
                                  std::uint32_t device_ordinal);

// Production routing input. The selected-expert matrix stays in backend-owned
// device storage; this handle adds immutable shape, capacity, execution and
// owner provenance without exposing a native address.
template <> class SelectedExpertRouteLease<std::int32_t> {
public:
  SelectedExpertRouteLease() = default;
  std::uint64_t num_tokens() const;
  std::uint32_t top_k() const;
  std::uint64_t route_count() const;
  std::uint64_t device_bytes() const;
  std::uint32_t layout_version() const;
  std::uint32_t device_ordinal() const;
  std::uint64_t route_generation() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit SelectedExpertRouteLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend CudaRouteInput upload_cuda_routes(
      const RankRoutingInput &, std::uint32_t,
      const RankCollectiveContext &, const CudaExecution &);
  friend CudaRouteInput import_selected_expert_routes(
      const CudaRouteInput &, const RankCollectiveContext &,
      const CudaExecution &);
  friend CudaGateInput import_selected_expert_gates(
      const CudaGateInput &, const CudaRouteInput &,
      const RankCollectiveContext &, const CudaExecution &);
  friend class DeviceNotifyPoolBridge;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class DeviceRouteProvenanceBridge;
  friend class DeviceGateProvenanceBridge;
#endif
  friend class SelectedExpertGateLease<float>;
  friend class ExternalCudaProducer;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class ActivationTransportBridge;
  friend class CudaNotifyOperation;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, const CudaGateInput &,
      CudaNotifyWorkspace);
#endif
};

inline constexpr std::uint32_t kSelectedExpertRouteLayoutVersion = 1;

// Explicit cross-stream handoff. This enqueues cudaStreamWaitEvent on the
// consumer stream and preserves the immutable producer provenance.
CudaRouteInput import_selected_expert_routes(
    const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution);

CudaGateInput import_selected_expert_gates(
    const CudaGateInput &gates, const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution);

inline constexpr std::uint32_t kSelectedExpertGateLayoutVersion = 1;

// Immutable FP32 router gate tensor in the exact [token, top-k] route order.
// No public API exposes its device address or permits provenance rebinding.
template <> class SelectedExpertGateLease<float> {
public:
  SelectedExpertGateLease() = default;
  std::uint64_t num_tokens() const;
  std::uint32_t top_k() const;
  std::uint64_t route_count() const;
  std::uint64_t device_bytes() const;
  std::uint32_t layout_version() const;
  std::uint32_t device_ordinal() const;
  std::uint64_t gate_generation() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit SelectedExpertGateLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class ExternalCudaProducer;
  friend CudaGateInput import_selected_expert_gates(
      const CudaGateInput &, const CudaRouteInput &,
      const RankCollectiveContext &, const CudaExecution &);
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class DeviceGateProvenanceBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class CudaNotifyOperation;
  friend class DeviceMetadataBridge;
  friend class ActivationTransportBridge;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, const CudaGateInput &,
      CudaNotifyWorkspace);
#endif
};

class CudaLocalNotifyHandle {
public:
  CudaLocalNotifyHandle() = default;
  std::uint64_t num_tokens() const;
  const RuntimeConfig &config() const;
  const RankCollectiveContext &context() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  const DeviceBufferLease &device_expert_count_for_backend() const;
  const DeviceBufferLease &device_packet_count_for_backend() const;
  struct Impl;
  explicit CudaLocalNotifyHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class CudaNotifyOperation;
  friend class DeviceMetadataBridge;
#endif
  friend class CudaNotifyGenerationCoordinator;
  friend class DeviceDispatchConsumerLease;
};

// Versioned POD fields directly consumed by future forward/backward kernels.
// 64-bit fields precede 32-bit fields so the CUDA/host ABI has no hidden
// interior padding; fixed sizes below are part of layout version 1.
struct DeviceReverseRouteRecord {
  std::uint64_t local_token_id;
  std::uint64_t dst_row;
  std::uint32_t source_rank;
  std::uint32_t topk_slot;
  std::uint32_t expert_id;
  std::uint32_t destination_rank;
};

struct DeviceIncomingRowRecord {
  std::uint64_t source_expert_offset;
  std::uint64_t dst_row;
  std::uint32_t source_rank;
  std::uint32_t expert_id;
};

struct DevicePacketHeaderRecord {
  std::uint64_t sequence_on_link;
  std::uint64_t local_token_id;
  std::uint32_t source_rank;
  std::uint32_t destination_node;
  std::uint32_t proxy_rank;
  std::uint32_t channel;
};

struct DeviceFanoutRecord {
  std::uint64_t dst_row;
  std::uint32_t topk_slot;
  std::uint32_t destination_local_rank;
  std::uint32_t expert_id;
  std::uint32_t reserved = 0;
};

struct DeviceSignalRecord {
  std::uint64_t expected_packets;
  std::uint64_t expected_signals;
  std::uint32_t source_node;
  std::uint32_t destination_node;
  std::uint32_t source_rank;
  std::uint32_t destination_proxy_rank;
  std::uint32_t rail_local_rank;
  std::uint32_t channel;
};

static_assert(sizeof(DeviceReverseRouteRecord) == 32);
static_assert(sizeof(DeviceIncomingRowRecord) == 24);
static_assert(sizeof(DevicePacketHeaderRecord) == 32);
static_assert(sizeof(DeviceFanoutRecord) == 24);
static_assert(sizeof(DeviceSignalRecord) == 40);

inline constexpr std::uint32_t kDeviceDispatchLayoutVersion = 1;

struct DeviceDispatchIdentity {
  std::uint32_t layout_version = kDeviceDispatchLayoutVersion;
  std::uint64_t generation = 0;
  std::uint32_t calling_rank = 0;
  std::uint32_t world_size = 0;
  std::uint64_t num_tokens = 0;
  std::uint32_t top_k = 0;
  std::uint32_t num_experts = 0;
  std::uint32_t num_nodes = 0;
  std::uint64_t config_digest_word0 = 0;
  std::uint64_t config_digest_word1 = 0;
};

struct DeviceDispatchSizes {
  std::uint64_t route_count = 0;
  std::uint32_t node_mask_words = 0;
  std::uint32_t owned_expert_count = 0;
  std::uint64_t incoming_row_count = 0;
  std::uint64_t mtile_count = 0;
  std::uint64_t packet_count = 0;
  std::uint64_t fanout_count = 0;
  std::uint32_t outgoing_signal_count = 0;
  std::uint32_t incoming_signal_count = 0;
};

// Auditable production-boundary counters. A normal generation must publish
// 0/0/0/0. Explicit validation download is a separate opt-in API.
struct DeviceMaterializationAudit {
  std::uint64_t bulk_route_d2h = 0;
  std::uint64_t bulk_collective_d2h = 0;
  std::uint64_t host_materializations = 0;
  std::uint64_t bulk_metadata_h2d = 0;
};

struct DeviceDispatchViewExpectation {
  std::uint64_t generation = 0;
  std::uint32_t calling_rank = 0;
  std::uint32_t world_size = 0;
  std::uint64_t num_tokens = 0;
  std::uint32_t top_k = 0;
  std::uint64_t config_digest_word0 = 0;
  std::uint64_t config_digest_word1 = 0;
};

enum class DeviceDispatchConsumerState : std::uint8_t {
  Acquired,
  Submitted,
  Completed,
  Released,
};

// Owning public consumer boundary. It never exposes native pointers. A trusted
// launch adapter submits work and records the pooled completion event.
class DeviceDispatchConsumerLease {
public:
  DeviceDispatchConsumerLease() = default;
  ~DeviceDispatchConsumerLease() noexcept;
  DeviceDispatchConsumerLease(const DeviceDispatchConsumerLease &) = delete;
  DeviceDispatchConsumerLease &operator=(const DeviceDispatchConsumerLease &) = delete;
  DeviceDispatchConsumerLease(DeviceDispatchConsumerLease &&) noexcept;
  DeviceDispatchConsumerLease &operator=(DeviceDispatchConsumerLease &&) noexcept;
  DeviceDispatchConsumerState state() const;
  void finish();
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceDispatchConsumerLease(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend class DeviceDispatchHandle;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class DeviceDispatchTestBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class ActivationTransportBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend DispatchHandle download_device_metadata_for_validation(
      const DeviceDispatchHandle &, const CudaExecution &);
#endif
};

class DeviceDispatchHandle {
public:
  DeviceDispatchHandle() = default;
  const DeviceDispatchIdentity &identity() const;
  const DeviceDispatchSizes &sizes() const;
  const DeviceMaterializationAudit &audit() const;
  DeviceDispatchConsumerLease acquire_consumer(
      const CudaExecution &execution,
      const DeviceDispatchViewExpectation &expected) const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceDispatchHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class DeviceMetadataBridge;
#endif
  friend class CudaNotifyGenerationCoordinator;
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class ActivationTransportBridge;
#endif
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend DispatchHandle download_device_metadata_for_validation(
      const DeviceDispatchHandle &, const CudaExecution &);
#endif
};

#if defined(MEGAMOE_INTERNAL_TESTING_API)
// Explicit diagnostic/test-only bulk download. The production coordinator
// never calls this function.
DispatchHandle download_device_metadata_for_validation(
    const DeviceDispatchHandle &handle, const CudaExecution &execution);

void validate_cuda_dispatch_metadata(const DeviceDispatchHandle &handle,
                                      const CudaExecution &execution);

namespace testing {
CudaGateInput upload_fp32_gates_for_testing(
    const std::vector<float> &host, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution);
struct CudaApiAuditSnapshot {
  std::uint64_t device_allocations = 0;
  std::uint64_t device_frees = 0;
  std::uint64_t pinned_allocations = 0;
  std::uint64_t pinned_frees = 0;
  std::uint64_t event_creates = 0;
  std::uint64_t event_destroys = 0;
  std::uint64_t device_synchronizes = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t d2d_bytes = 0;
};
CudaApiAuditSnapshot cuda_api_audit_snapshot() noexcept;
DeviceDispatchHandle materialize_fake_device_collectives(
    const RuntimeConfig &config,
    const RankCollectiveContext &context,
    const CudaExecution &execution,
    const RankRoutingInput &local_routes,
    std::uint64_t max_tokens_per_rank,
    const std::vector<std::uint64_t> &expert_count_matrix,
    const std::vector<std::uint64_t> &packet_summary_matrix);
void submit_dispatch_probe(DeviceDispatchConsumerLease &consumer,
                           std::uint64_t delay_cycles);
std::uint64_t finish_dispatch_probe(DeviceDispatchConsumerLease &consumer);
} // namespace testing
#endif

} // namespace megamoe::cuda
