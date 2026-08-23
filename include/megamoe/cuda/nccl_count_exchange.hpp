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
  friend class NcclSession;
  friend NcclSession make_nccl_session(std::string, std::uint32_t,
                                       std::uint32_t, std::uint64_t,
                                       std::uint32_t,
                                       const NcclBootstrapId &);
  friend NcclSession make_nccl_generation(const NcclSession &, std::string,
                                          std::uint64_t);
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
  friend class NcclAllGatherCountExchange;
};

NcclSession make_nccl_session(std::string identity, std::uint32_t world_size,
                              std::uint32_t rank, std::uint64_t generation,
                              std::uint32_t device_ordinal,
                              const NcclBootstrapId &bootstrap_id);

// Creates a new immutable generation/session view over the same live NCCL
// communicator. This allows a healthy communicator to recover after a rejected
// control generation without re-bootstrap or native-handle exposure.
NcclSession make_nccl_generation(const NcclSession &base,
                                 std::string identity,
                                 std::uint64_t generation);

class ValidatedControlHandle {
public:
  ValidatedControlHandle() = default;
  const ControlMatrix &matrix() const;
  const RankCollectiveContext &context() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ValidatedControlHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class NcclControlOperation;
};

class NcclControlOperation {
public:
  NcclControlOperation();
  ~NcclControlOperation() noexcept;
  NcclControlOperation(const NcclControlOperation &) = delete;
  NcclControlOperation &operator=(const NcclControlOperation &) = delete;
  NcclControlOperation(NcclControlOperation &&) noexcept;
  NcclControlOperation &operator=(NcclControlOperation &&) noexcept;
  ExchangeState state() const noexcept;
  ValidatedControlHandle wait();
  void cancel();
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit NcclControlOperation(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend class NcclAllGatherCountExchange;
};

struct CollectiveProtocolAudit {
  std::uint64_t control_starts = 0;
  std::uint64_t expert_count_starts = 0;
  std::uint64_t packet_summary_starts = 0;
};

struct NcclWorkspaceUsage {
  std::uint64_t device_bytes = 0;
  std::uint64_t pinned_host_bytes = 0;
  std::uint64_t control_slot_bytes = 0;
  std::uint64_t expert_slot_bytes = 0;
  std::uint64_t summary_slot_bytes = 0;
};

class NcclAllGatherCountExchange final : public CountExchange {
public:
  struct Impl;
  NcclAllGatherCountExchange(const NcclSession &session,
                             const CudaExecution &execution);
  NcclAllGatherCountExchange(const NcclSession &session,
                             const CudaExecution &execution,
                             std::uint64_t max_expert_row_width,
                             std::uint64_t max_summary_row_width);
  ~NcclAllGatherCountExchange() noexcept override;
  NcclAllGatherCountExchange(const NcclAllGatherCountExchange &) = delete;
  NcclAllGatherCountExchange &
  operator=(const NcclAllGatherCountExchange &) = delete;
  NcclAllGatherCountExchange(NcclAllGatherCountExchange &&) noexcept;
  NcclAllGatherCountExchange &
  operator=(NcclAllGatherCountExchange &&) noexcept;

  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &request) override;
  NcclControlOperation start_control(const RankCollectiveContext &context,
                                     const ControlRecord &record);
  CollectiveProtocolAudit audit() const;
  NcclWorkspaceUsage workspace_usage() const;

private:
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
