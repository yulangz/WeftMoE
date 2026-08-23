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
class CudaDispatchHandle;
class CudaRouteInput;
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
  friend class CudaRouteInput;
  friend class CudaNotifyWorkspace;
  friend class CudaNotifyOperation;
  friend class NcclAllGatherCountExchange;
  friend std::shared_ptr<const internal::CudaExecutionState>
  internal::checked_execution_state(const CudaExecution &);
};

CudaExecution make_cuda_execution(std::string identity,
                                  std::uint32_t device_ordinal);

class CudaRouteInput {
public:
  CudaRouteInput() = default;
  std::uint64_t num_tokens() const;
  std::uint32_t top_k() const;
  std::uint64_t route_count() const;
  std::uint64_t device_bytes() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaRouteInput(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend CudaRouteInput upload_cuda_routes(const RankRoutingInput &,
                                           std::uint32_t,
                                           const CudaExecution &);
  friend class CudaNotifyOperation;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, CudaNotifyWorkspace);
};

CudaRouteInput upload_cuda_routes(const RankRoutingInput &routes,
                                  std::uint32_t top_k,
                                  const CudaExecution &execution);

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
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, CudaNotifyWorkspace);
};

CudaNotifyWorkspace allocate_cuda_notify_workspace(
    const CudaNotifyWorkspacePlan &plan,
    const RankCollectiveContext &context, const CudaExecution &execution);

class CudaLocalNotifyHandle {
public:
  CudaLocalNotifyHandle() = default;
  const LocalHistogramHandle &local() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaLocalNotifyHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class CudaNotifyOperation;
  friend CudaDispatchHandle stage_cuda_dispatch_metadata(
      DispatchHandle, const CudaLocalNotifyHandle &, const CudaExecution &);
  friend void validate_cuda_dispatch_metadata(
      const CudaDispatchHandle &, const CudaExecution &);
};

// Completion of local CUDA validation. Predictable route/config errors are
// represented as data so the rank can still join the fixed control collective.
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
      const CudaExecution &, const CudaRouteInput &, CudaNotifyWorkspace);
};

CudaNotifyOperation start_cuda_rank_local_notify(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    CudaNotifyWorkspace workspace);

// Device copies of the fields directly consumed by future forward/backward
// kernels. The host handle was produced through checked typed staging.
struct DeviceReverseRouteRecord {
  std::uint32_t source_rank;
  std::uint64_t local_token_id;
  std::uint32_t topk_slot;
  std::uint32_t expert_id;
  std::uint32_t destination_rank;
  std::uint64_t dst_row;
};

struct DeviceIncomingRowRecord {
  std::uint32_t source_rank;
  std::uint32_t expert_id;
  std::uint64_t source_expert_offset;
  std::uint64_t dst_row;
};

struct DevicePacketHeaderRecord {
  std::uint32_t source_rank;
  std::uint32_t destination_node;
  std::uint64_t sequence_on_link;
  std::uint64_t local_token_id;
  std::uint32_t proxy_rank;
  std::uint32_t channel;
};

struct DeviceFanoutRecord {
  std::uint32_t topk_slot;
  std::uint32_t destination_local_rank;
  std::uint32_t expert_id;
  std::uint64_t dst_row;
};

struct DeviceSignalRecord {
  std::uint32_t source_node;
  std::uint32_t destination_node;
  std::uint32_t source_rank;
  std::uint32_t destination_proxy_rank;
  std::uint32_t rail_local_rank;
  std::uint32_t channel;
  std::uint64_t expected_packets;
  std::uint64_t expected_signals;
};

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

class CudaDispatchHandle {
public:
  CudaDispatchHandle() = default;
  const DispatchHandle &host() const;
  const DeviceDispatchMetadata &device() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaDispatchHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend CudaDispatchHandle stage_cuda_dispatch_metadata(
      DispatchHandle, const CudaLocalNotifyHandle &, const CudaExecution &);
  friend void validate_cuda_dispatch_metadata(
      const CudaDispatchHandle &, const CudaExecution &);
};

CudaDispatchHandle stage_cuda_dispatch_metadata(
    DispatchHandle host, const CudaLocalNotifyHandle &local,
    const CudaExecution &execution);

// Controlled diagnostic/test staging. The CUDA module interprets its private
// leases, downloads every device field, and compares it to the immutable host
// handle. Transport-independent metadata never sees a native device address.
void validate_cuda_dispatch_metadata(const CudaDispatchHandle &handle,
                                     const CudaExecution &execution);

} // namespace megamoe::cuda
