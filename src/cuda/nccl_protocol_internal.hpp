#pragma once

#include <memory>

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
