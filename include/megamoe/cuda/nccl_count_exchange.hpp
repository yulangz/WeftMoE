#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/cuda/device_notify.hpp"

namespace megamoe::cuda {

class NcclSession;

class NcclBootstrapId {
public:
  NcclBootstrapId() = default;
  std::string hex() const;
  explicit operator bool() const noexcept { return !bytes_.empty(); }

private:
  explicit NcclBootstrapId(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}
  std::vector<std::uint8_t> bytes_;
  friend NcclBootstrapId make_nccl_bootstrap_id();
  friend NcclBootstrapId parse_nccl_bootstrap_id(const std::string &);
  friend NcclSession make_nccl_session(std::string, std::uint32_t,
                                       std::uint32_t, std::uint64_t,
                                       std::uint32_t,
                                       const NcclBootstrapId &);
};

NcclBootstrapId make_nccl_bootstrap_id();
NcclBootstrapId parse_nccl_bootstrap_id(const std::string &hex);

class NcclSession {
public:
  NcclSession() = default;
  const CollectiveSession &session() const;
  std::uint32_t rank() const;
  std::uint32_t device_ordinal() const;
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
};

struct NcclWorkspaceUsage {
  std::uint64_t device_bytes = 0;
  std::uint64_t pinned_host_bytes = 0;
  std::uint64_t control_slot_bytes = 0;
  std::uint64_t config_wire_bytes = 0;
  std::uint64_t config_slot_bytes = 0;
  std::uint64_t expert_slot_bytes = 0;
  std::uint64_t summary_slot_bytes = 0;
};

struct GenerationSetupTimings {
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

struct NotifyGenerationRequest {
  RuntimeConfig config;
  RankRoutingInput local_routes;
  std::uint64_t max_tokens_per_rank = 0;
};

// A successful result can only be published after the exact sequence
// control/config/count/summary has completed once each.
class ValidatedCudaGeneration {
public:
  ValidatedCudaGeneration() = default;
  const CudaDispatchHandle &dispatch() const;
  const CollectiveProtocolAudit &audit() const;
  const NcclWorkspaceUsage &workspace_usage() const;
  const GenerationSetupTimings &setup_timings() const;
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
                                  const CudaExecution &execution);
  ~CudaNotifyGenerationCoordinator() noexcept;
  CudaNotifyGenerationCoordinator(
      const CudaNotifyGenerationCoordinator &) = delete;
  CudaNotifyGenerationCoordinator &operator=(
      const CudaNotifyGenerationCoordinator &) = delete;
  CudaNotifyGenerationCoordinator(CudaNotifyGenerationCoordinator &&) noexcept;
  CudaNotifyGenerationCoordinator &operator=(
      CudaNotifyGenerationCoordinator &&) noexcept;

  ValidatedCudaGeneration run(NotifyGenerationRequest request);
  CollectiveProtocolAudit audit() const;
  NcclWorkspaceUsage workspace_usage() const;
  GenerationSetupTimings setup_timings() const;

private:
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

} // namespace megamoe::cuda
