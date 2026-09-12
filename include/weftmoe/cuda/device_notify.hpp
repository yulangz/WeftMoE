#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "weftmoe/api/notify_dispatch.hpp"
#include "weftmoe/cuda/control_protocol.hpp"

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe::cuda {

// Dynamic receives are sized after Notify. Static receives reserve the maximum
// possible routing fan-in before capture; excess token shapes require a new
// pool.
enum class BufferMode : std::uint8_t { Dynamic, Static };

// Common capacities derived from the current count exchange on every rank.
struct DeviceDispatchCapacity {
  std::uint64_t tokens_per_rank = 0;
  std::uint64_t destination_rows_per_rank = 0;
};

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
class NcclSession;

// CUDA/NCCL native handles remain private to the implementation.
class CudaExecution {
public:
  CudaExecution() = default;
  CudaExecution(const CudaExecution &);
  CudaExecution(CudaExecution &&) noexcept;
  CudaExecution &operator=(const CudaExecution &);
  CudaExecution &operator=(CudaExecution &&) noexcept;
  ~CudaExecution();
  const ExecutionToken &token() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CudaExecution(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend CudaExecution make_cuda_execution(std::string, std::uint32_t);
  friend class SelectedExpertRouteLease<std::int32_t>;
  friend class DeviceDispatchConsumerLease;
  friend class NcclAllGatherCountExchange;
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
  friend class SelectedExpertGateLease<float>;
  friend class ExternalCudaProducer;
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
  friend class CudaNotifyGenerationCoordinator;
  friend class DeviceDispatchConsumerLease;
};

// Versioned POD fields directly consumed by persistent kernels.
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
  // Dense within one (source rank, destination node) pair and ordered by
  // destination local rank then canonical row. Forward reuses
  // this immutable identity so Return wire groups follow compute M blocks
  // instead of the token/packet dispatch order.
  std::uint64_t return_compact_row;
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
static_assert(sizeof(DeviceFanoutRecord) == 32);
static_assert(sizeof(DeviceSignalRecord) == 40);

inline constexpr std::uint32_t kDeviceDispatchLayoutVersion = 2;

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
};

class DeviceDispatchHandle {
public:
  DeviceDispatchHandle() = default;
  const DeviceDispatchIdentity &identity() const;
  // Host snapshots from preparation. Static replay refreshes the corresponding
  // device counts without a host readback; these snapshots remain unchanged.
  const DeviceDispatchSizes &sizes() const;
  const DeviceDispatchCapacity &capacity() const;
  // Required per-call receive storage: exact rows in dynamic mode, declared
  // maximum fan-in in static mode, including future routing changes.
  std::uint64_t received_row_capacity() const;
  // Enqueue a fresh Notify into static storage using the retained input
  // tensors. Their values may change, their shape and addresses may not.
  // The caller must finish the prior FWD/BWD before reusing this storage.
  void enqueue_notify(const NcclSession &session) const;
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
  friend class CudaNotifyGenerationCoordinator;
};

} // namespace weftmoe::cuda

WEFTMOE_PUBLIC_API_END
