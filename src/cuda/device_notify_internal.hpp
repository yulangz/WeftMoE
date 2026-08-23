#pragma once

#include "megamoe/cuda/device_notify.hpp"

namespace megamoe::cuda {

class DeviceMetadataBridge {
public:
  static DeviceDispatchHandle materialize(
      const CudaLocalNotifyHandle &local,
      BackendCompletion expert_counts,
      BackendCompletion packet_summary,
      const CudaExecution &execution);
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
  friend class DeviceNotifyPoolBridge;
  friend CudaNotifyOperation start_cuda_rank_local_notify(
      const RuntimeConfig &, const RankCollectiveContext &,
      const CudaExecution &, const CudaRouteInput &, CudaNotifyWorkspace);
};

CudaNotifyWorkspace allocate_cuda_notify_workspace(
    const CudaNotifyWorkspacePlan &plan,
    const RankCollectiveContext &context, const CudaExecution &execution);

struct CudaNotifyPoolAllocation;

class DeviceNotifyPoolBridge {
public:
  static std::shared_ptr<CudaNotifyPoolAllocation> allocate(
      const CudaNotifyWorkspacePlan &max_plan,
      const CudaExecution &execution);
  static CudaRouteInput upload(
      const RankRoutingInput &routes, std::uint32_t top_k,
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
      const CudaExecution &, const CudaRouteInput &, CudaNotifyWorkspace);
};

CudaNotifyOperation start_cuda_rank_local_notify(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    CudaNotifyWorkspace workspace);

} // namespace megamoe::cuda
