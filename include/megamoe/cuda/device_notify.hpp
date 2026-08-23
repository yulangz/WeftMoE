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
class CudaLocalNotifyHandle;
class DeviceDispatchHandle;
using CudaDispatchHandle = DeviceDispatchHandle;
class DeviceRouteInput;
using CudaRouteInput = DeviceRouteInput;
class CudaNotifyWorkspace;
class CudaNotifyOperation;
class CudaNotifyPreflight;
namespace internal {
struct CudaExecutionState;
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
  friend class DeviceNotifyBridge;
  friend class DeviceRouteInput;
  friend class CudaNotifyWorkspace;
  friend class CudaNotifyOperation;
  friend class NcclAllGatherCountExchange;
  friend std::shared_ptr<const internal::CudaExecutionState>
  internal::checked_execution_state(const CudaExecution &);
};

CudaExecution make_cuda_execution(std::string identity,
                                  std::uint32_t device_ordinal);

// Production routing input. The selected-expert matrix stays in backend-owned
// device storage; this handle adds immutable shape, capacity, execution and
// owner provenance without exposing a native address.
class DeviceRouteInput {
public:
  DeviceRouteInput() = default;
  std::uint64_t num_tokens() const;
  std::uint32_t top_k() const;
  std::uint64_t route_count() const;
  std::uint64_t device_bytes() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceRouteInput(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend DeviceRouteInput make_device_route_input(
      DeviceBufferLease, std::uint64_t, std::uint32_t,
      const CudaExecution &);
  friend DeviceRouteInput upload_cuda_routes(
      const RankRoutingInput &, std::uint32_t, const CudaExecution &);
  friend class DeviceNotifyPoolBridge;
  friend class CudaNotifyOperation;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, CudaNotifyWorkspace);
};

DeviceRouteInput make_device_route_input(
    DeviceBufferLease selected_experts, std::uint64_t num_tokens,
    std::uint32_t top_k, const CudaExecution &execution);

// Explicit host convenience for tests/tools. It allocates and uploads once;
// production training should pass the router's existing DeviceBufferLease to
// make_device_route_input instead.
DeviceRouteInput upload_device_routes_for_testing(
    const RankRoutingInput &routes, std::uint32_t top_k,
    const CudaExecution &execution);

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
  friend class CudaNotifyOperation;
  friend class DeviceMetadataBridge;
  friend class CudaNotifyGenerationCoordinator;
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

struct DeviceDispatchMetadata {
  DeviceBufferLease expert_id;
  DeviceBufferLease local_offset;
  DeviceBufferLease route_dst_row;
  DeviceBufferLease route_dst_rank;
  DeviceBufferLease route_dst_node;
  DeviceBufferLease route_dst_local_rank;
  DeviceBufferLease combine_order;
  DeviceBufferLease destination_node_mask;
  DeviceBufferLease source_expert_count;
  DeviceBufferLease packet_summary_count;
  DeviceBufferLease owned_expert_ids;
  DeviceBufferLease expert_row_count;
  DeviceBufferLease expert_row_prefix;
  DeviceBufferLease mtile_row_offsets;
  DeviceBufferLease mtile_expected_rows;
  DeviceBufferLease incoming_rows;
  DeviceBufferLease reverse_routes;
  DeviceBufferLease packet_headers;
  DeviceBufferLease packet_fanout_offsets;
  DeviceBufferLease packet_fanout;
  DeviceBufferLease outgoing_signals;
  DeviceBufferLease incoming_proxy_signals;
};

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

// CUDA-specific typed view for forward/backward kernels. Native pointers never
// enter the transport-independent metadata API.
struct DeviceDispatchView {
  DeviceDispatchIdentity identity;
  DeviceDispatchSizes sizes;
  const std::uint32_t *expert_id = nullptr;
  const std::uint64_t *local_offset = nullptr;
  const std::uint64_t *route_dst_row = nullptr;
  const std::uint32_t *route_dst_rank = nullptr;
  const std::uint32_t *route_dst_node = nullptr;
  const std::uint32_t *route_dst_local_rank = nullptr;
  const std::uint32_t *combine_order = nullptr;
  const std::uint64_t *destination_node_mask = nullptr;
  const std::uint64_t *source_expert_count = nullptr;
  const std::uint32_t *owned_expert_ids = nullptr;
  const std::uint64_t *expert_row_count = nullptr;
  const std::uint64_t *expert_row_prefix = nullptr;
  const std::uint64_t *mtile_row_offsets = nullptr;
  const std::uint32_t *mtile_expected_rows = nullptr;
  const DeviceIncomingRowRecord *incoming_rows = nullptr;
  const DeviceReverseRouteRecord *reverse_routes = nullptr;
  const DevicePacketHeaderRecord *packet_headers = nullptr;
  const std::uint64_t *packet_fanout_offsets = nullptr;
  const DeviceFanoutRecord *packet_fanout = nullptr;
  const DeviceSignalRecord *outgoing_signals = nullptr;
  const DeviceSignalRecord *incoming_proxy_signals = nullptr;
};

class DeviceDispatchHandle {
public:
  DeviceDispatchHandle() = default;
  const DeviceDispatchMetadata &device() const;
  const DeviceDispatchIdentity &identity() const;
  const DeviceDispatchSizes &sizes() const;
  const DeviceMaterializationAudit &audit() const;
  DeviceDispatchView checked_view(
      const CudaExecution &execution,
      const DeviceDispatchViewExpectation &expected) const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceDispatchHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class DeviceMetadataBridge;
  friend class CudaNotifyGenerationCoordinator;
  friend DispatchHandle download_device_metadata_for_validation(
      const DeviceDispatchHandle &, const CudaExecution &);
};

// Explicit diagnostic/test-only bulk download. The production coordinator
// never calls this function.
DispatchHandle download_device_metadata_for_validation(
    const DeviceDispatchHandle &handle, const CudaExecution &execution);

void validate_cuda_dispatch_metadata(const DeviceDispatchHandle &handle,
                                      const CudaExecution &execution);

namespace testing {
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
} // namespace testing

} // namespace megamoe::cuda
