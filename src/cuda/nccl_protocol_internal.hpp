#pragma once

#include <memory>
#include <cstddef>
#include <vector>

#include "megamoe/cuda/nccl_count_exchange.hpp"

namespace megamoe::cuda {

namespace testing {
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
  CollectiveProtocolAudit audit() const;
  NcclWorkspaceUsage workspace_usage() const;

private:
  std::shared_ptr<Impl> impl_;
  friend class CudaNotifyGenerationCoordinator;
};

} // namespace megamoe::cuda
