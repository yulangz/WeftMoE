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
