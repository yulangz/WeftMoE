#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define private public
#include "weftmoe/cuda/device_notify.hpp"
#undef private

namespace weftmoe::cuda {

class CudaNotifyWorkspace;
class CudaNotifyOperation;
class CudaNotifyPreflight;

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
CudaGateInput upload_fp32_gates_for_testing(
    const std::vector<float> &, const CudaRouteInput &,
    const RankCollectiveContext &, const CudaExecution &);
DeviceDispatchHandle materialize_fake_device_collectives(
    const RuntimeConfig &, const RankCollectiveContext &,
    const CudaExecution &, const RankRoutingInput &, std::uint64_t,
    const std::vector<std::uint64_t> &,
    const std::vector<std::uint64_t> &);
void submit_dispatch_probe(DeviceDispatchConsumerLease &, std::uint64_t);
std::uint64_t finish_dispatch_probe(DeviceDispatchConsumerLease &);
} // namespace testing

DispatchHandle download_device_metadata_for_validation(
    const DeviceDispatchHandle &, const CudaExecution &);
void validate_cuda_dispatch_metadata(const DeviceDispatchHandle &,
                                     const CudaExecution &);

// Backend-private opaque slices. Generic DeviceBufferLease remains confined to
// CUDA/NCCL implementation code and never enters the public dispatch API.
struct DeviceDispatchMetadata {
  DeviceBufferLease expert_id;
  DeviceBufferLease local_offset;
  DeviceBufferLease route_dst_row;
  DeviceBufferLease route_dst_mtile;
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
  DeviceBufferLease local_route_order;
  DeviceBufferLease dispatch_wave_packet_order;
  DeviceBufferLease dispatch_wave_offsets;
  DeviceBufferLease dispatch_wave_destination_node;
  DeviceBufferLease dispatch_wave_channel;
  DeviceBufferLease dispatch_wave_first_primary_tile;
  DeviceBufferLease dispatch_wave_last_primary_tile;
  DeviceBufferLease dispatch_wave_packet_count;
  DeviceBufferLease dispatch_wave_payload_bytes;
  DeviceBufferLease outgoing_signals;
  DeviceBufferLease incoming_proxy_signals;
};

class DeviceMetadataBridge {
public:
  struct Timings {
    double prefix_planning_us = 0;
    double remaining_parallel_us = 0;
  };
  static DeviceDispatchHandle materialize(
      const CudaLocalNotifyHandle &local,
      BackendCompletion expert_counts,
      BackendCompletion packet_summary,
      const CudaExecution &execution, Timings *timings = nullptr);
};

CudaRouteInput upload_cuda_routes(const RankRoutingInput &routes,
                                  std::uint32_t top_k,
                                  const RankCollectiveContext &context,
                                  const CudaExecution &execution);

namespace testing {
enum class ConsumerFaultPoint : std::uint8_t {
  None,
  EventRecord,
  EventSynchronize,
  EventAndStreamSynchronize,
};
void inject_next_consumer_fault(ConsumerFaultPoint point) noexcept;
} // namespace testing

class DeviceRouteProvenanceBridge {
public:
  static void validate(const CudaRouteInput &routes,
                       const RankCollectiveContext &context,
                       const CudaExecution &execution);
  // Private test/benchmark adapters. Immutable storage may be shared, but
  // every lease has exact generation provenance. These declarations are not
  // installed with the public API.
  static CudaRouteInput reissue_for_testing(
      const CudaRouteInput &storage, const RankCollectiveContext &context,
      const CudaExecution &execution);
  static CudaRouteInput schedule_delayed_for_testing(
      const CudaRouteInput &storage, std::int32_t first_expert,
      std::uint64_t delay_cycles, const RankCollectiveContext &context,
      const CudaExecution &producer_execution, bool record_ready = true);
  static CudaRouteInput unsafe_import_without_wait_for_testing(
      const CudaRouteInput &routes,
      const RankCollectiveContext &consumer_context,
      const CudaExecution &consumer_execution);
};

class DeviceGateProvenanceBridge {
public:
  static CudaGateInput upload_internal(
      const std::vector<float> &host, const CudaRouteInput &routes,
      const RankCollectiveContext &context,
      const CudaExecution &execution);
  static void validate(const CudaGateInput &gates,
                       const CudaRouteInput &routes,
                       const RankCollectiveContext &context,
                       const CudaExecution &execution);
  static bool same_route_storage(const CudaGateInput &gates,
                                 const CudaRouteInput &routes);
  static CudaGateInput reissue_for_testing(
      const CudaGateInput &storage, const CudaRouteInput &exact_routes,
      const RankCollectiveContext &context,
      const CudaExecution &execution);
  static const float *device_pointer(const CudaGateInput &gates);
};

struct DeviceDispatchKernelView {
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

// GIN-only scheduling view.  Keeping the wavefront permutation outside
// DeviceDispatchKernelView means the frozen LocalCudaP2p Forward
// launch contract and its generated SASS do not grow when the remote backend
// learns a new packet scheduling policy.
struct DispatchWaveKernelView {
  std::uint64_t wave_count = 0;
  // Source routes sorted by (local-node first, destination local M tile,
  // expert, original route).  Relay CTAs use the local prefix to complete
  // destination M-blocks in the same wavefront order as remote packets.
  const std::uint64_t *local_route_order = nullptr;
  const std::uint64_t *packet_order = nullptr;
  const std::uint64_t *offsets = nullptr;
  const std::uint32_t *destination_node = nullptr;
  const std::uint32_t *channel = nullptr;
  const std::uint64_t *first_primary_tile = nullptr;
  const std::uint64_t *last_primary_tile = nullptr;
  const std::uint64_t *packet_count = nullptr;
  const std::uint64_t *descriptor_bytes = nullptr;
  const std::uint32_t *route_dst_mtile = nullptr;
};

struct CudaNotifyWorkspacePlan {
  std::uint32_t calling_rank = 0;
  std::uint64_t num_tokens = 0;
  std::uint64_t max_tokens_per_rank = 0;
  std::uint32_t top_k = 0;
  std::uint32_t num_experts = 0;
  std::uint32_t world_size = 0;
  std::uint32_t num_nodes = 0;
  std::uint32_t node_mask_words = 0;
  std::uint64_t route_count = 0;
  std::uint64_t max_incoming_rows = 0;
  std::uint64_t device_bytes = 0;
  std::uint64_t host_staging_bytes = 0;
  std::uint64_t sort_temp_bytes = 0;
  std::uint64_t scan_temp_bytes = 0;
};

CudaNotifyWorkspacePlan query_cuda_notify_workspace(
    const RuntimeConfig &config, std::uint32_t calling_rank,
    std::uint64_t num_tokens, std::uint64_t max_tokens_per_rank);

class CudaNotifyWorkspace {
public:
  CudaNotifyWorkspace();
  ~CudaNotifyWorkspace() noexcept;
  CudaNotifyWorkspace(const CudaNotifyWorkspace &) = delete;
  CudaNotifyWorkspace &operator=(const CudaNotifyWorkspace &) = delete;
  CudaNotifyWorkspace(CudaNotifyWorkspace &&) noexcept;
  CudaNotifyWorkspace &operator=(CudaNotifyWorkspace &&) noexcept;
  const CudaNotifyWorkspacePlan &plan() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaNotifyWorkspace(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend CudaNotifyWorkspace
  allocate_cuda_notify_workspace(const CudaNotifyWorkspacePlan &,
                                 const RankCollectiveContext &,
                                 const CudaExecution &);
  friend class CudaNotifyOperation;
  friend class DeviceNotifyPoolBridge;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, const CudaGateInput &,
      CudaNotifyWorkspace);
};

CudaNotifyWorkspace allocate_cuda_notify_workspace(
    const CudaNotifyWorkspacePlan &plan,
    const RankCollectiveContext &context, const CudaExecution &execution);

struct CudaNotifyPoolAllocation;

class DeviceNotifyPoolBridge {
public:
  static std::shared_ptr<CudaNotifyPoolAllocation> allocate(
      const CudaNotifyWorkspacePlan &max_plan,
      const CudaExecution &execution,
      std::uint32_t max_inflight_consumers = 4);
  static CudaRouteInput upload(
      const RankRoutingInput &routes, std::uint32_t top_k,
      const RankCollectiveContext &context,
      const CudaExecution &execution,
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
      std::shared_ptr<const void> slot_lease);
  static CudaNotifyWorkspace workspace(
      const CudaNotifyWorkspacePlan &actual_plan,
      const RankCollectiveContext &context,
      const CudaExecution &execution,
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
      std::shared_ptr<const void> slot_lease);
  static bool drain(
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
      const CudaExecution &execution) noexcept;
  static std::uint64_t device_bytes(
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept;
  static std::uint64_t pinned_bytes(
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept;
  static std::uint64_t event_count(
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept;
  static bool quarantined(
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept;
  static std::array<std::uint32_t, 3> consumer_event_usage(
      const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept;
};

class CudaNotifyPreflight {
public:
  CudaNotifyPreflight() = default;
  const LocalPreflightStatus &status() const;
  double device_mapping_us() const;
  bool valid() const;
  const CudaLocalNotifyHandle &local() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaNotifyPreflight(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class CudaNotifyOperation;
};

class CudaNotifyOperation {
public:
  CudaNotifyOperation();
  ~CudaNotifyOperation() noexcept;
  CudaNotifyOperation(const CudaNotifyOperation &) = delete;
  CudaNotifyOperation &operator=(const CudaNotifyOperation &) = delete;
  CudaNotifyOperation(CudaNotifyOperation &&) noexcept;
  CudaNotifyOperation &operator=(CudaNotifyOperation &&) noexcept;
  CudaNotifyPreflight wait_preflight();
  CudaLocalNotifyHandle wait();
  void cancel();
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaNotifyOperation(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, const CudaGateInput &,
      CudaNotifyWorkspace);
};

CudaNotifyOperation start_cuda_rank_local_notify(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    const CudaGateInput &gates, CudaNotifyWorkspace workspace);

} // namespace weftmoe::cuda
