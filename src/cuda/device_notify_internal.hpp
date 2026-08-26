#pragma once

#ifndef MEGAMOE_INTERNAL_TESTING_API
#define MEGAMOE_INTERNAL_TESTING_API 1
#endif

#include "megamoe/cuda/device_notify.hpp"

namespace megamoe::cuda {

// Backend-private opaque slices. Generic DeviceBufferLease remains confined to
// CUDA/NCCL implementation code and never enters the public dispatch API.
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
  static CudaGateInput upload_for_testing(
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

} // namespace megamoe::cuda
