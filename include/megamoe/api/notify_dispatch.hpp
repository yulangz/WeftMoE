#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/config/runtime_config.hpp"
#include "megamoe/metadata/dispatch_handle.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace megamoe {
namespace testing {
class NotifyCorruptionHook;
}
namespace cuda {
class DeviceNotifyBridge;
}

// Immutable phase-1 result. It owns the calling rank's routes and typed count
// rows, and binds them to one rank/session/generation/execution provenance.
class LocalHistogramHandle {
public:
  LocalHistogramHandle() = default;
  const RuntimeConfig &config() const;
  const RankCollectiveContext &context() const;
  const SourceRouteMetadata &outgoing() const;
  const HostIntegerRow &local_expert_count() const;
  const HostIntegerRow &packet_count_by_destination_node() const;
  const DeviceBufferLease &device_expert_count() const;
  const DeviceBufferLease &device_packet_count() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit LocalHistogramHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;

  friend LocalHistogramHandle
  build_rank_local_histogram(const RuntimeConfig &,
                             const RankCollectiveContext &,
                             const RankRoutingInput &);
  friend class CollectiveOperation;
  friend DispatchHandle
  materialize_rank_metadata(const LocalHistogramHandle &,
                            const class ValidatedCollectiveHandle &,
                            const class ValidatedCollectiveHandle &);
  friend class testing::NotifyCorruptionHook;
  friend class cuda::DeviceNotifyBridge;
};

LocalHistogramHandle
build_rank_local_histogram(const RuntimeConfig &config,
                           const RankCollectiveContext &context,
                           const RankRoutingInput &local_routes);

// Successful framework-validated phase-2 result. Backends cannot construct it.
class ValidatedCollectiveHandle {
public:
  ValidatedCollectiveHandle() = default;
  const RankCollectiveContext &context() const;
  ExchangeKind kind() const;
  std::uint32_t rows() const;
  std::uint64_t row_width() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ValidatedCollectiveHandle(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend class CollectiveOperation;
  friend DispatchHandle
  materialize_rank_metadata(const LocalHistogramHandle &,
                            const ValidatedCollectiveHandle &,
                            const ValidatedCollectiveHandle &);
  friend class testing::NotifyCorruptionHook;
};

// Move-only framework adaptor. It retains local storage, session/execution
// tokens, and the backend operation until wait/cancel reaches a terminal state.
// The same object does not support concurrent wait/cancel/move calls.
class CollectiveOperation {
public:
  CollectiveOperation();
  ~CollectiveOperation() noexcept;
  CollectiveOperation(const CollectiveOperation &) = delete;
  CollectiveOperation &operator=(const CollectiveOperation &) = delete;
  CollectiveOperation(CollectiveOperation &&) noexcept;
  CollectiveOperation &operator=(CollectiveOperation &&) noexcept;

  ExchangeState state() const noexcept;
  ValidatedCollectiveHandle wait();
  void cancel();
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CollectiveOperation(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend CollectiveOperation start_count_exchange(const LocalHistogramHandle &,
                                                  CountExchange &);
  friend CollectiveOperation
  start_notify_summary_exchange(const LocalHistogramHandle &, CountExchange &);
};

CollectiveOperation start_count_exchange(const LocalHistogramHandle &local,
                                         CountExchange &backend);
CollectiveOperation
start_notify_summary_exchange(const LocalHistogramHandle &local,
                              CountExchange &backend);

// Pure rank-local phase 3. Both completions must have been created from this
// exact local handle and its exact rank/session/execution context.
DispatchHandle
materialize_rank_metadata(const LocalHistogramHandle &local,
                          const ValidatedCollectiveHandle &expert_counts,
                          const ValidatedCollectiveHandle &packet_summary);

// Reference/test convenience only; production callers use the rank-local API.
std::vector<DispatchHandle> build_notify_dispatch_for_testing(
    const RuntimeConfig &config,
    const std::vector<RankRoutingInput> &routes_by_rank,
    const std::string &communicator_id = "host-fake",
    std::uint64_t generation = 1);

} // namespace megamoe
