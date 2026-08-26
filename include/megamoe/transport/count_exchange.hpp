#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace megamoe {

#if defined(MEGAMOE_INTERNAL_TESTING_API)
namespace testing { class HostIntegerRowTestAccess; }
#endif

using CountMatrix = std::vector<std::vector<std::uint64_t>>;

enum class ExchangeKind { Control, ExpertCounts, NotifyPacketSummary };
enum class ExchangeState { Pending, Complete, Failed, Cancelled };

class BackendSessionState {
public:
  virtual ~BackendSessionState() noexcept = default;
};

class BackendExecutionState {
public:
  virtual ~BackendExecutionState() noexcept = default;
};

class CollectiveSession {
public:
  CollectiveSession() = default;
  const std::string &identity() const;
  // Stable protocol identity shared by every process in the communicator.
  // This is immutable bounded storage; native communicator addresses are never
  // used as a distributed identity.
  const std::vector<std::byte> &stable_identity_bytes() const;
  std::uint32_t world_size() const;
  std::uint64_t generation() const;
  bool same_instance(const CollectiveSession &other) const noexcept;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit CollectiveSession(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend CollectiveSession
  make_collective_session(std::string, std::uint32_t, std::uint64_t,
                          std::shared_ptr<const BackendSessionState>);
  friend CollectiveSession make_collective_session(
      std::string, std::vector<std::byte>, std::uint32_t, std::uint64_t,
      std::shared_ptr<const BackendSessionState>);
  friend class CountExchange;
};

CollectiveSession make_collective_session(
    std::string identity, std::uint32_t world_size, std::uint64_t generation,
    std::shared_ptr<const BackendSessionState> backend_state = {});

CollectiveSession make_collective_session(
    std::string display_identity, std::vector<std::byte> stable_identity_bytes,
    std::uint32_t world_size, std::uint64_t generation,
    std::shared_ptr<const BackendSessionState> backend_state = {});

class ExecutionToken {
public:
  ExecutionToken() = default;
  const std::string &identity() const;
  bool same_instance(const ExecutionToken &other) const noexcept;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit ExecutionToken(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend ExecutionToken
      make_execution_token(std::string,
                           std::shared_ptr<const BackendExecutionState>);
  friend class CountExchange;
};

ExecutionToken make_execution_token(
    std::string identity,
    std::shared_ptr<const BackendExecutionState> backend_state = {});

class RankCollectiveContext {
public:
  RankCollectiveContext() = default;
  std::uint32_t calling_rank() const;
  const CollectiveSession &session() const;
  const ExecutionToken &execution() const;
  bool same_instance(const RankCollectiveContext &other) const noexcept;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit RankCollectiveContext(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend RankCollectiveContext make_rank_collective_context(CollectiveSession,
                                                            std::uint32_t,
                                                            ExecutionToken);
};

RankCollectiveContext make_rank_collective_context(CollectiveSession session,
                                                   std::uint32_t calling_rank,
                                                   ExecutionToken execution);

// Typed immutable host storage: allocation and ownership cannot be separated.
class HostIntegerRow {
public:
  HostIntegerRow() = default;
  std::size_t size() const;
  std::uint64_t at(std::size_t index) const;
  const std::vector<std::uint64_t> &values() const;
  explicit operator bool() const noexcept { return static_cast<bool>(values_); }

private:
  explicit HostIntegerRow(
      std::shared_ptr<const std::vector<std::uint64_t>> values)
      : values_(std::move(values)) {}
  std::shared_ptr<const std::vector<std::uint64_t>> values_;
  friend HostIntegerRow make_host_integer_row(std::vector<std::uint64_t>);
#if defined(MEGAMOE_INTERNAL_TESTING_API)
  friend class testing::HostIntegerRowTestAccess;
#endif
};

HostIntegerRow make_host_integer_row(std::vector<std::uint64_t> values);

class HostCountMatrix {
public:
  HostCountMatrix() = default;
  std::uint32_t rows() const;
  std::uint64_t row_width() const;
  std::uint64_t at(std::uint32_t row, std::uint64_t column) const;
  CountMatrix copy_rows() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit HostCountMatrix(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend HostCountMatrix make_host_count_matrix(const CountMatrix &);
};

HostCountMatrix make_host_count_matrix(const CountMatrix &rows);

class BackendDeviceLeaseState {
public:
  virtual ~BackendDeviceLeaseState() noexcept = default;
};

// Optional backend-owned resources on which a returned completion depends.
// The framework retains this lease through every validated-handle copy.
class BackendCompletionLeaseState {
public:
  virtual ~BackendCompletionLeaseState() noexcept = default;
};

// No native-address accessor exists in the transport-independent layer.
// element_count is checked rows*row_width and cannot exceed size_t or the
// backend-declared capacity_elements.
class DeviceBufferLease {
public:
  DeviceBufferLease() = default;
  std::uint32_t rows() const;
  std::uint64_t row_width() const;
  std::uint64_t element_count() const;
  std::uint64_t capacity_elements() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit DeviceBufferLease(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend DeviceBufferLease
  make_device_buffer_lease(std::uint32_t, std::uint64_t, std::uint64_t,
                           std::shared_ptr<const BackendDeviceLeaseState>);
  friend class CountExchange;
};

DeviceBufferLease make_device_buffer_lease(
    std::uint32_t rows, std::uint64_t row_width,
    std::uint64_t capacity_elements,
    std::shared_ptr<const BackendDeviceLeaseState> backend_lease);

struct CountExchangeRequest {
  RankCollectiveContext context;
  ExchangeKind kind = ExchangeKind::ExpertCounts;
  std::uint64_t row_width = 0;
  HostIntegerRow input;
  // Optional backend-opaque device row. Generic metadata never interprets it.
  // A backend must reject an unsupported or foreign lease before enqueue.
  DeviceBufferLease device_input;
};

class BackendCompletion {
public:
  BackendCompletion() = default;
  const RankCollectiveContext &context() const;
  ExchangeKind kind() const;
  bool has_host_result() const noexcept;
  bool has_device_result() const noexcept;
  const HostCountMatrix &host_result() const;
  const DeviceBufferLease &device_result() const;

private:
  struct Impl;
  explicit BackendCompletion(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend BackendCompletion make_host_backend_completion(
      RankCollectiveContext, ExchangeKind, HostCountMatrix,
      std::shared_ptr<const BackendCompletionLeaseState>);
  friend BackendCompletion make_device_backend_completion(
      RankCollectiveContext, ExchangeKind, DeviceBufferLease,
      std::shared_ptr<const BackendCompletionLeaseState>);
};

BackendCompletion make_host_backend_completion(
    RankCollectiveContext context, ExchangeKind kind, HostCountMatrix result,
    std::shared_ptr<const BackendCompletionLeaseState> completion_lease = {});
BackendCompletion make_device_backend_completion(
    RankCollectiveContext context, ExchangeKind kind, DeviceBufferLease result,
    std::shared_ptr<const BackendCompletionLeaseState> completion_lease = {});

class BackendCountExchangeOperation {
public:
  virtual ~BackendCountExchangeOperation() noexcept = default;
  virtual ExchangeState state() const noexcept = 0;
  virtual BackendCompletion wait() = 0;
  virtual void cancel() = 0;
};

class CountExchange {
public:
  virtual ~CountExchange() noexcept = default;
  virtual std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &request) = 0;

protected:
  static std::shared_ptr<const BackendSessionState>
  backend_session_state(const RankCollectiveContext &context);
  static std::shared_ptr<const BackendExecutionState>
  backend_execution_state(const RankCollectiveContext &context);
  static std::shared_ptr<const BackendDeviceLeaseState>
  backend_device_state(const DeviceBufferLease &lease);
};

// Deterministic host test backend; not an NCCL/GIN/LSA/RDMA implementation.
class HostFakeCountExchange final : public CountExchange {
public:
  struct SharedState;
  HostFakeCountExchange(std::uint32_t world_size, std::string session_identity);
  ~HostFakeCountExchange() noexcept override;
  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &request) override;
  void fail_next_start(std::string message);
  void fail_next_completion(std::string message);

private:
  std::shared_ptr<SharedState> state_;
};

} // namespace megamoe
