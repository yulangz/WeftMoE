#pragma once

#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "weftmoe/cuda/device_notify.hpp"

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe::cuda {

class NcclSession;
class NcclBootstrapInputChannel;
class NcclBootstrapOutputChannel;
class CudaNotifyGenerationCoordinator;

class NcclBootstrapId {
public:
  NcclBootstrapId() = default;
  explicit operator bool() const noexcept { return !bytes_.empty(); }
  const std::vector<std::uint8_t> &bytes() const noexcept { return bytes_; }
  static NcclBootstrapId from_bytes(std::vector<std::uint8_t> bytes);

private:
  explicit NcclBootstrapId(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}
  std::vector<std::uint8_t> bytes_;
  friend NcclBootstrapId make_nccl_bootstrap_id();
  friend class NcclBootstrapInputChannel;
  friend class NcclBootstrapOutputChannel;
  friend NcclSession make_nccl_session(std::string, std::uint32_t,
                                       std::uint32_t, std::uint64_t,
                                       std::uint32_t,
                                       const NcclBootstrapId &);
};

NcclBootstrapId make_nccl_bootstrap_id();

class NcclSession {
public:
  NcclSession() = default;
  NcclSession(const NcclSession &);
  NcclSession(NcclSession &&) noexcept;
  NcclSession &operator=(const NcclSession &);
  NcclSession &operator=(NcclSession &&) noexcept;
  ~NcclSession();
  const CollectiveSession &session() const;
  std::uint32_t rank() const;
  std::uint32_t device_ordinal() const;
  bool shares_communicator_with(const NcclSession &other) const noexcept;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit NcclSession(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend NcclSession make_nccl_session(std::string, std::uint32_t,
                                       std::uint32_t, std::uint64_t,
                                       std::uint32_t,
                                       const NcclBootstrapId &);
  friend NcclSession make_nccl_generation(const NcclSession &, std::string,
                                          std::uint64_t);
  friend class CudaNotifyGenerationCoordinator;
  friend class NcclAllGatherCountExchange;
};

NcclSession make_nccl_session(std::string identity, std::uint32_t world_size,
                              std::uint32_t rank, std::uint64_t generation,
                              std::uint32_t device_ordinal,
                              const NcclBootstrapId &bootstrap_id);
NcclSession make_nccl_generation(const NcclSession &base,
                                 std::string stable_identity,
                                 std::uint64_t generation);

struct CollectiveProtocolAudit {
  std::uint64_t control_starts = 0;
  std::uint64_t config_starts = 0;
  std::uint64_t expert_count_starts = 0;
  std::uint64_t packet_summary_starts = 0;
  std::uint64_t config_cache_hits = 0;
  // Single-node trusted release path: one record carries control/status,
  // expert counts and packet summary in one AllGather.
  std::uint64_t fused_notify_starts = 0;
};

struct NcclWorkspaceUsage {
  // Unique underlying allocations. Subviews are reported separately and are
  // never added twice to device_bytes/pinned_host_bytes.
  std::uint64_t device_bytes = 0;
  std::uint64_t pinned_host_bytes = 0;
  std::uint64_t route_input_device_bytes = 0;
  std::uint64_t device_notify_arena_bytes = 0;
  std::uint64_t device_notify_pinned_staging_bytes = 0;
  // Capacity of the final metadata view within device_notify_arena_bytes.
  // It is a subview and is therefore not added to device_bytes again.
  std::uint64_t final_device_metadata_arena_view_capacity_bytes = 0;
  std::uint64_t event_count = 0;
  std::uint64_t quantified_event_bytes = 0;
  std::uint64_t control_slot_bytes = 0;
  std::uint64_t config_wire_bytes = 0;
  std::uint64_t config_slot_bytes = 0;
  std::uint64_t expert_slot_bytes = 0;
  std::uint64_t summary_slot_bytes = 0;
  std::uint64_t fused_notify_slot_bytes = 0;
};

struct GenerationSetupTimings {
  double one_time_pool_setup_us = 0;
  double pool_acquire_us = 0;
  double coordinator_backend_create_us = 0;
  double control_slot_allocation_us = 0;
  double config_slot_allocation_us = 0;
  double count_slot_allocation_us = 0;
  double summary_slot_allocation_us = 0;
  double event_allocation_us = 0;
  double pinned_allocation_us = 0;
  double device_allocation_us = 0;
  double per_generation_setup_total_us = 0;
};

struct GenerationExecutionTimings {
  // Direct same-rank coordinator measurement. setup-inclusive is defined as
  // this value plus that rank's GenerationSetupTimings total.
  double steady_execution_us = 0;
  double coordinator_run_wall_us = 0;
  double local_mapping_us = 0;
  double control_config_us = 0;
  double expert_count_us = 0;
  double packet_summary_us = 0;
  double device_materialization_us = 0;
  double prefix_planning_us = 0;
  double remaining_parallel_materialization_us = 0;
  double pool_acquire_release_us = 0;
  // Same-clock-domain offsets from coordinator entry. Route/gate readiness is
  // the entry boundary for run_device(); all remaining values are monotonic
  // host timestamps around the asynchronous device/NCCL work.
  double route_gate_ready_offset_us = 0;
  double pool_acquire_begin_offset_us = 0;
  double pool_acquire_end_offset_us = 0;
  double local_histogram_begin_offset_us = 0;
  double local_histogram_end_offset_us = 0;
  double collective_enqueue_begin_offset_us = 0;
  double collective_enqueue_end_offset_us = 0;
  double collective_complete_offset_us = 0;
  double device_materialization_end_offset_us = 0;
  double notify_handle_ready_offset_us = 0;
  double fused_collective_enqueue_us = 0;
  double fused_collective_wait_us = 0;
};

struct NotifyGenerationRequest {
  RuntimeConfig config;
  RankRoutingInput local_routes;
  std::uint64_t max_tokens_per_rank = 0;
};

struct DeviceNotifyGenerationRequest {
  RuntimeConfig config;
  CudaRouteInput local_routes;
  CudaGateInput local_gates;
  std::uint64_t max_tokens_per_rank = 0;
};

class GenerationCancellationToken {
public:
  GenerationCancellationToken();
  void request_cancel() const noexcept;
  bool cancellation_requested() const noexcept;

private:
  struct State;
  std::shared_ptr<State> state_;
};

struct GenerationRunOptions {
  GenerationCancellationToken cancellation;
  // Zero means no deadline. The deadline is measured from coordinator entry.
  std::chrono::microseconds deadline_after{0};
};

struct DeviceNotifyResourcePoolConfig {
  RuntimeConfig capacity_config;
  std::uint64_t max_tokens_per_rank = 0;
  std::uint64_t max_config_wire_bytes = 0;
  std::uint32_t max_inflight_generations = 2;
  std::uint32_t max_inflight_consumers = 4;
  // Single-node inference fast path. Route rows are allocated atomically and
  // are intentionally not stable across invocations. GEMM does not reduce in
  // M and the fixed combine order is retained, so this does not affect FWD
  // numerics. Training must leave this disabled because dW reduces over M.
  bool enable_unstable_inference_rows = false;
};

struct DeviceNotifyResourcePoolUsage {
  std::uint32_t slot_count = 0;
  std::uint32_t slots_available = 0;
  std::uint32_t slots_in_use = 0;
  std::uint32_t slots_quarantined = 0;
  std::uint32_t consumer_events_available = 0;
  std::uint32_t consumer_events_in_use = 0;
  std::uint32_t consumer_events_quarantined = 0;
  std::uint64_t slot_epoch_high_watermark = 0;
  std::uint64_t device_bytes = 0;
  std::uint64_t pinned_host_bytes = 0;
  std::uint64_t event_count = 0;
  double one_time_setup_us = 0;
};

class DeviceNotifyResourcePool {
public:
  DeviceNotifyResourcePool() = default;
  DeviceNotifyResourcePoolUsage usage() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceNotifyResourcePool(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend DeviceNotifyResourcePool make_device_notify_resource_pool(
      const NcclSession &, const CudaExecution &,
      DeviceNotifyResourcePoolConfig);
  friend class CudaNotifyGenerationCoordinator;
};

DeviceNotifyResourcePool make_device_notify_resource_pool(
    const NcclSession &base_session, const CudaExecution &execution,
    DeviceNotifyResourcePoolConfig config);

// A successful result can only be published after control, an exact config
// agreement (or a release-mode hit on the session's already-validated config),
// count, and summary complete in that order.
class ValidatedCudaGeneration {
public:
  ValidatedCudaGeneration() = default;
  const CudaDispatchHandle &dispatch() const;
  const CollectiveProtocolAudit &audit() const;
  const NcclWorkspaceUsage &workspace_usage() const;
  const GenerationSetupTimings &setup_timings() const;
  const GenerationExecutionTimings &execution_timings() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ValidatedCudaGeneration(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class CudaNotifyGenerationCoordinator;
};

// The sole production entry for one distributed generation. Predictable
// config/shape/route/capacity failures are dataized before the fixed control
// AllGather; callers cannot separately wait preflight or start data collectives.
class CudaNotifyGenerationCoordinator {
public:
  CudaNotifyGenerationCoordinator(const NcclSession &session,
                                  const CudaExecution &execution,
                                  const DeviceNotifyResourcePool &pool);
  ~CudaNotifyGenerationCoordinator() noexcept;
  CudaNotifyGenerationCoordinator(
      const CudaNotifyGenerationCoordinator &) = delete;
  CudaNotifyGenerationCoordinator &operator=(
      const CudaNotifyGenerationCoordinator &) = delete;
  CudaNotifyGenerationCoordinator(CudaNotifyGenerationCoordinator &&) noexcept;
  CudaNotifyGenerationCoordinator &operator=(
      CudaNotifyGenerationCoordinator &&) noexcept;

  ValidatedCudaGeneration run(NotifyGenerationRequest request,
                              GenerationRunOptions options = {});
  ValidatedCudaGeneration run_device(DeviceNotifyGenerationRequest request,
                                     GenerationRunOptions options = {});
  CollectiveProtocolAudit audit() const;
  NcclWorkspaceUsage workspace_usage() const;
  GenerationSetupTimings setup_timings() const;

private:
  CudaNotifyGenerationCoordinator(const NcclSession &session,
                                  const CudaExecution &execution);
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

const char *compiled_nccl_version() noexcept;

struct NcclRuntimeInfo {
  std::string compiled_version;
  std::string runtime_version;
  std::string loaded_library_path;
  std::string socket_ifname;
  std::string ib_disable;
};

NcclRuntimeInfo nccl_runtime_info();

} // namespace weftmoe::cuda

WEFTMOE_PUBLIC_API_END
