#pragma once

#include <memory>
#include <cstddef>
#include <vector>

#include "megamoe/cuda/nccl_count_exchange.hpp"

namespace megamoe::cuda {

enum class BootstrapChannelState { Open, Consumed, Failed, Closed };

// Private launcher/test bridge. Construction transfers descriptor ownership.
// Both directions are move-only one-shot channels and close before entering a
// terminal state; no text representation is exposed by the public NCCL API.
class NcclBootstrapInputChannel {
public:
  static constexpr std::size_t kWireBytes = 128U;
  explicit NcclBootstrapInputChannel(int owned_fd);
  ~NcclBootstrapInputChannel() noexcept;
  NcclBootstrapInputChannel(const NcclBootstrapInputChannel &) = delete;
  NcclBootstrapInputChannel &
  operator=(const NcclBootstrapInputChannel &) = delete;
  NcclBootstrapInputChannel(NcclBootstrapInputChannel &&) noexcept;
  NcclBootstrapInputChannel &
  operator=(NcclBootstrapInputChannel &&) noexcept;

  NcclBootstrapId read();
  void cancel() noexcept;
  BootstrapChannelState state() const noexcept { return state_; }
  int native_fd_for_testing() const noexcept { return fd_; }

private:
  int finish(BootstrapChannelState) noexcept;
  void require_open(const char *operation) const;
  int fd_ = -1;
  BootstrapChannelState state_ = BootstrapChannelState::Closed;
};

class NcclBootstrapOutputChannel {
public:
  static constexpr std::size_t kWireBytes = 128U;
  explicit NcclBootstrapOutputChannel(int owned_fd);
  ~NcclBootstrapOutputChannel() noexcept;
  NcclBootstrapOutputChannel(const NcclBootstrapOutputChannel &) = delete;
  NcclBootstrapOutputChannel &
  operator=(const NcclBootstrapOutputChannel &) = delete;
  NcclBootstrapOutputChannel(NcclBootstrapOutputChannel &&) noexcept;
  NcclBootstrapOutputChannel &
  operator=(NcclBootstrapOutputChannel &&) noexcept;

  void write(const NcclBootstrapId &);
  void cancel() noexcept;
  BootstrapChannelState state() const noexcept { return state_; }
  int native_fd_for_testing() const noexcept { return fd_; }

private:
  int finish(BootstrapChannelState) noexcept;
  void require_open(const char *operation) const;
  int fd_ = -1;
  BootstrapChannelState state_ = BootstrapChannelState::Closed;
};

namespace testing {
using BootstrapReadHook = std::ptrdiff_t (*)(int, void *, std::size_t);
using BootstrapWriteHook = std::ptrdiff_t (*)(int, const void *, std::size_t);
using BootstrapCloseHook = int (*)(int);

// Process-local test seam for exercising EINTR, short I/O, and zero progress.
// Production calls leave all hooks null and use POSIX read/write/close.
void set_bootstrap_io_hooks(BootstrapReadHook, BootstrapWriteHook,
                            BootstrapCloseHook = nullptr) noexcept;
void clear_bootstrap_io_hooks() noexcept;

enum class NcclFaultPoint {
  None,
  ControlStart,
  ControlEnqueue,
  EventRecord,
  EventWait,
  D2HStaging,
  ConfigSlotAllocation,
  CountSlotAllocation,
  SummarySlotAllocation,
  BackendCompletion,
  CountKernelCompletion,
  SummaryKernelCompletion,
  MaterializationKernel,
};
void inject_next_nccl_fault(NcclFaultPoint point) noexcept;
// Controlled wire/provenance corruption hooks. They are declared only in the
// private test header and are consumed once by the sole coordinator.
void corrupt_next_config_wire(std::size_t offset) noexcept;
void override_next_protocol_identity(
    std::vector<std::byte> raw_identity,
    StableSessionId diagnostic_summary) noexcept;
struct NcclLifecycleSnapshot {
  std::uint64_t slot_allocations = 0;
  std::uint64_t slot_destructions = 0;
  std::uint64_t operation_waits = 0;
  std::uint64_t operation_cancels = 0;
  std::uint64_t destructor_drains = 0;
  std::uint64_t completion_lease_creates = 0;
  std::uint64_t completion_lease_destroys = 0;
  std::uint64_t validated_publishes = 0;
};
void reset_nccl_lifecycle_counters() noexcept;
NcclLifecycleSnapshot nccl_lifecycle_snapshot() noexcept;
} // namespace testing

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

private:
  struct Impl;
  explicit NcclControlOperation(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend class NcclAllGatherCountExchange;
};

class ValidatedConfigHandle {
public:
  ValidatedConfigHandle() = default;
  explicit operator bool() const noexcept { return static_cast<bool>(owner_); }

private:
  std::shared_ptr<const void> owner_;
  friend class NcclConfigOperation;
  friend class NcclAllGatherCountExchange;
};

class NcclConfigOperation {
public:
  NcclConfigOperation();
  ~NcclConfigOperation() noexcept;
  NcclConfigOperation(const NcclConfigOperation &) = delete;
  NcclConfigOperation &operator=(const NcclConfigOperation &) = delete;
  NcclConfigOperation(NcclConfigOperation &&) noexcept;
  NcclConfigOperation &operator=(NcclConfigOperation &&) noexcept;
  ValidatedConfigHandle wait();
  void cancel();

private:
  struct Impl;
  explicit NcclConfigOperation(std::shared_ptr<Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
  friend class NcclAllGatherCountExchange;
};

class NcclAllGatherCountExchange final : public CountExchange {
public:
  struct Impl;
  NcclAllGatherCountExchange(const NcclSession &, const CudaExecution &);
  NcclAllGatherCountExchange(const NcclSession &, const CudaExecution &,
                             std::uint64_t, std::uint64_t);
  NcclAllGatherCountExchange(const NcclSession &, const CudaExecution &,
                             std::uint64_t, std::uint64_t, std::size_t);
  ~NcclAllGatherCountExchange() noexcept override;
  NcclAllGatherCountExchange(const NcclAllGatherCountExchange &) = delete;
  NcclAllGatherCountExchange &
  operator=(const NcclAllGatherCountExchange &) = delete;
  NcclAllGatherCountExchange(NcclAllGatherCountExchange &&) noexcept;
  NcclAllGatherCountExchange &
  operator=(NcclAllGatherCountExchange &&) noexcept;

  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &) override;
  NcclControlOperation start_control(const RankCollectiveContext &,
                                     const ControlHeader &);
  NcclConfigOperation start_config(const RankCollectiveContext &,
                                   const ConfigWire &,
                                   const ValidatedControlHandle &);
  bool can_reuse_verified_config(const ConfigWire &) const;
  void reuse_verified_config(const RankCollectiveContext &,
                             const ConfigWire &,
                             const ValidatedControlHandle &);
  CollectiveProtocolAudit audit() const;
  NcclWorkspaceUsage workspace_usage() const;
  void prepare_generation(const NcclSession &);

private:
  struct FastNotifyResult {
    BackendCompletion expert_counts;
    BackendCompletion packet_summary;
    double enqueue_us = 0;
    double wait_us = 0;
  };
  bool fast_path_ready() const;
  bool fast_config_matches(const ConfigWire &) const;
  FastNotifyResult run_fast_notify(
      const RankCollectiveContext &, const ControlHeader &,
      const DeviceBufferLease *expert_counts,
      const DeviceBufferLease *packet_summary);
  // Completes the fixed-record agreement after downstream device work on the
  // same execution stream has reached its own completion event. This folds
  // the fast collective and materialization into one host wait while keeping
  // control validation before a generation is published.
  double finish_fast_notify();
  std::shared_ptr<Impl> impl_;
  friend class CudaNotifyGenerationCoordinator;
};

} // namespace megamoe::cuda
