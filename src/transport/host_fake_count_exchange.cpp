#include "megamoe/transport/count_exchange.hpp"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "megamoe/api/errors.hpp"

namespace megamoe {
namespace {

[[noreturn]] void invalid_exchange(const std::string &message) {
  throw NotifyDispatchError(NotifyErrorCode::CountExchangeFailure, message);
}

const char *kind_name(ExchangeKind kind) noexcept {
  switch (kind) {
  case ExchangeKind::Control: return "control";
  case ExchangeKind::ExpertCounts: return "expert_counts";
  case ExchangeKind::NotifyPacketSummary: return "notify_packet_summary";
  }
  return "unknown";
}

} // namespace

struct CollectiveSession::Impl {
  std::string identity;
  std::vector<std::byte> stable_identity_bytes;
  std::uint32_t world_size = 0;
  std::uint64_t generation = 0;
  std::shared_ptr<const BackendSessionState> backend_state;
};

CollectiveSession make_collective_session(
    std::string identity, std::uint32_t world_size, std::uint64_t generation,
    std::shared_ptr<const BackendSessionState> backend_state) {
  std::vector<std::byte> stable(identity.size());
  std::transform(identity.begin(), identity.end(), stable.begin(),
                 [](char value) { return static_cast<std::byte>(value); });
  return make_collective_session(std::move(identity), std::move(stable),
                                 world_size, generation,
                                 std::move(backend_state));
}

CollectiveSession make_collective_session(
    std::string identity, std::vector<std::byte> stable_identity_bytes,
    std::uint32_t world_size, std::uint64_t generation,
    std::shared_ptr<const BackendSessionState> backend_state) {
  if (identity.empty()) {
    invalid_exchange("collective session identity/communicator is empty: "
                     "expected=non-empty, actual=empty");
  }
  if (world_size == 0) {
    invalid_exchange(
        "collective session world_size is invalid: expected=>0, actual=0");
  }
  if (generation == 0) {
    invalid_exchange(
        "collective session generation is invalid: expected=>0, actual=0");
  }
  constexpr std::size_t kMaxStableIdentityBytes = 4096;
  if (stable_identity_bytes.empty() ||
      stable_identity_bytes.size() > kMaxStableIdentityBytes) {
    invalid_exchange(
        "collective stable identity bytes are invalid: expected=(0,4096], actual=" +
        std::to_string(stable_identity_bytes.size()));
  }
  auto impl = std::make_shared<CollectiveSession::Impl>();
  impl->identity = std::move(identity);
  impl->stable_identity_bytes = std::move(stable_identity_bytes);
  impl->world_size = world_size;
  impl->generation = generation;
  impl->backend_state = std::move(backend_state);
  return CollectiveSession(std::move(impl));
}

const std::vector<std::byte> &CollectiveSession::stable_identity_bytes() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty collective session");
  return impl_->stable_identity_bytes;
}

const std::string &CollectiveSession::identity() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty collective session");
  return impl_->identity;
}

std::uint32_t CollectiveSession::world_size() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty collective session");
  return impl_->world_size;
}

std::uint64_t CollectiveSession::generation() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty collective session");
  return impl_->generation;
}

bool CollectiveSession::same_instance(
    const CollectiveSession &other) const noexcept {
  return impl_ && impl_ == other.impl_;
}

struct ExecutionToken::Impl {
  std::string identity;
  std::shared_ptr<const BackendExecutionState> backend_state;
};

ExecutionToken make_execution_token(
    std::string identity,
    std::shared_ptr<const BackendExecutionState> backend_state) {
  if (identity.empty()) {
    invalid_exchange(
        "execution token identity is empty: expected=non-empty, actual=empty");
  }
  auto impl = std::make_shared<ExecutionToken::Impl>();
  impl->identity = std::move(identity);
  impl->backend_state = std::move(backend_state);
  return ExecutionToken(std::move(impl));
}

const std::string &ExecutionToken::identity() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty execution token");
  return impl_->identity;
}

bool ExecutionToken::same_instance(const ExecutionToken &other) const noexcept {
  return impl_ && impl_ == other.impl_;
}

struct RankCollectiveContext::Impl {
  CollectiveSession session;
  std::uint32_t calling_rank = 0;
  ExecutionToken execution;
};

RankCollectiveContext make_rank_collective_context(CollectiveSession session,
                                                   std::uint32_t calling_rank,
                                                   ExecutionToken execution) {
  if (!session)
    invalid_exchange("rank context has empty collective session");
  if (!execution)
    invalid_exchange("rank context has empty execution token");
  if (calling_rank >= session.world_size()) {
    invalid_exchange("rank context calling_rank mismatch: expected=[0," +
                     std::to_string(session.world_size()) +
                     "), actual=" + std::to_string(calling_rank));
  }
  auto impl = std::make_shared<RankCollectiveContext::Impl>();
  impl->session = std::move(session);
  impl->calling_rank = calling_rank;
  impl->execution = std::move(execution);
  return RankCollectiveContext(std::move(impl));
}

std::uint32_t RankCollectiveContext::calling_rank() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty rank collective context");
  return impl_->calling_rank;
}

const CollectiveSession &RankCollectiveContext::session() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty rank collective context");
  return impl_->session;
}

const ExecutionToken &RankCollectiveContext::execution() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty rank collective context");
  return impl_->execution;
}

bool RankCollectiveContext::same_instance(
    const RankCollectiveContext &other) const noexcept {
  return impl_ && impl_ == other.impl_;
}

HostIntegerRow make_host_integer_row(std::vector<std::uint64_t> values) {
  return HostIntegerRow(
      std::make_shared<const std::vector<std::uint64_t>>(std::move(values)));
}

std::size_t HostIntegerRow::size() const {
  if (!values_)
    invalid_exchange("attempted to access an empty host integer row");
  return values_->size();
}

std::uint64_t HostIntegerRow::at(std::size_t index) const {
  if (!values_)
    invalid_exchange("attempted to access an empty host integer row");
  if (index >= values_->size()) {
    invalid_exchange(
        "host integer row index out of range: actual=" + std::to_string(index) +
        ", expected=[0," + std::to_string(values_->size()) + ")");
  }
  return values_->at(index);
}

const std::vector<std::uint64_t> &HostIntegerRow::values() const {
  if (!values_)
    invalid_exchange("attempted to access an empty host integer row");
  return *values_;
}

std::weak_ptr<const void>
HostIntegerRow::lifetime_for_testing() const noexcept {
  return std::weak_ptr<const void>(values_);
}

struct HostCountMatrix::Impl {
  std::uint32_t rows = 0;
  std::uint64_t row_width = 0;
  std::shared_ptr<const std::vector<std::uint64_t>> flat;
};

HostCountMatrix make_host_count_matrix(const CountMatrix &rows) {
  if (rows.size() > std::numeric_limits<std::uint32_t>::max()) {
    invalid_exchange("host count matrix rows overflow uint32: actual=" +
                     std::to_string(rows.size()));
  }
  const std::size_t width = rows.empty() ? 0 : rows.front().size();
  for (std::size_t rank = 0; rank < rows.size(); ++rank) {
    if (rows[rank].size() != width) {
      invalid_exchange(
          "host count matrix is ragged: rank=" + std::to_string(rank) +
          ", actual_width=" + std::to_string(rows[rank].size()) +
          ", expected_width=" + std::to_string(width));
    }
  }
  if (width != 0 &&
      rows.size() > std::numeric_limits<std::size_t>::max() / width) {
    invalid_exchange("host count matrix element count overflows size_t");
  }
  auto flat = std::make_shared<std::vector<std::uint64_t>>();
  flat->reserve(rows.size() * width);
  for (const auto &row : rows)
    flat->insert(flat->end(), row.begin(), row.end());
  auto impl = std::make_shared<HostCountMatrix::Impl>();
  impl->rows = static_cast<std::uint32_t>(rows.size());
  impl->row_width = static_cast<std::uint64_t>(width);
  impl->flat = std::move(flat);
  return HostCountMatrix(std::move(impl));
}

std::uint32_t HostCountMatrix::rows() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty host count matrix");
  return impl_->rows;
}

std::uint64_t HostCountMatrix::row_width() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty host count matrix");
  return impl_->row_width;
}

std::uint64_t HostCountMatrix::at(std::uint32_t row,
                                  std::uint64_t column) const {
  if (!impl_)
    invalid_exchange("attempted to access an empty host count matrix");
  if (row >= impl_->rows || column >= impl_->row_width) {
    invalid_exchange("host count matrix coordinate out of range: row=" +
                     std::to_string(row) +
                     ", column=" + std::to_string(column) +
                     ", expected_rows=" + std::to_string(impl_->rows) +
                     ", expected_width=" + std::to_string(impl_->row_width));
  }
  if (impl_->row_width > std::numeric_limits<std::size_t>::max()) {
    invalid_exchange("host count matrix row width exceeds size_t");
  }
  const auto width = static_cast<std::size_t>(impl_->row_width);
  const auto column_index = static_cast<std::size_t>(column);
  if (row != 0 && width > std::numeric_limits<std::size_t>::max() / row) {
    invalid_exchange("host count matrix offset overflows size_t");
  }
  const auto offset = static_cast<std::size_t>(row) * width + column_index;
  if (!impl_->flat || offset >= impl_->flat->size()) {
    invalid_exchange("host count matrix internal storage is inconsistent");
  }
  return impl_->flat->at(offset);
}

CountMatrix HostCountMatrix::copy_rows() const {
  const auto row_count = rows();
  const auto width_u64 = row_width();
  if (width_u64 > std::numeric_limits<std::size_t>::max()) {
    invalid_exchange("host count matrix width exceeds host size_t");
  }
  const auto width = static_cast<std::size_t>(width_u64);
  CountMatrix result(row_count, std::vector<std::uint64_t>(width));
  for (std::uint32_t row = 0; row < row_count; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      result[row][column] = at(row, static_cast<std::uint64_t>(column));
    }
  }
  return result;
}

struct DeviceBufferLease::Impl {
  std::uint32_t rows = 0;
  std::uint64_t row_width = 0;
  std::uint64_t element_count = 0;
  std::uint64_t capacity_elements = 0;
  std::shared_ptr<const BackendDeviceLeaseState> backend_lease;
};

DeviceBufferLease make_device_buffer_lease(
    std::uint32_t rows, std::uint64_t row_width,
    std::uint64_t capacity_elements,
    std::shared_ptr<const BackendDeviceLeaseState> backend_lease) {
  if (!backend_lease)
    invalid_exchange("phase=device-lease-validation, field=backend_lease, "
                     "rank=n/a, expected=non-null, actual=null");
  if (rows == 0)
    invalid_exchange("phase=device-lease-validation, field=rows, rank=n/a, "
                     "expected=>0, actual=0");
  if (row_width == 0)
    invalid_exchange("phase=device-lease-validation, field=row_width, "
                     "rank=n/a, expected=>0, actual=0");
  if (row_width > std::numeric_limits<std::uint64_t>::max() / rows) {
    invalid_exchange(
        "phase=device-lease-validation, field=element_count.uint64, rank=n/a, "
        "expected=rows*row_width within uint64, actual=overflow(rows=" +
        std::to_string(rows) + ", row_width=" + std::to_string(row_width) +
        ")");
  }
  const auto element_count = static_cast<std::uint64_t>(rows) * row_width;
  if (element_count > std::numeric_limits<std::size_t>::max()) {
    invalid_exchange(
        "phase=device-lease-validation, field=element_count.size_t, rank=n/a, "
        "expected=rows*row_width within size_t, actual=" +
        std::to_string(element_count));
  }
  if (capacity_elements == 0) {
    invalid_exchange(
        "phase=device-lease-validation, field=capacity_elements, rank=n/a, "
        "expected=>0, actual=0");
  }
  if (element_count > capacity_elements) {
    invalid_exchange(
        "phase=device-lease-validation, field=capacity_elements, rank=n/a, "
        "expected=>=" +
        std::to_string(element_count) +
        ", actual=" + std::to_string(capacity_elements));
  }
  auto impl = std::make_shared<DeviceBufferLease::Impl>();
  impl->rows = rows;
  impl->row_width = row_width;
  impl->element_count = element_count;
  impl->capacity_elements = capacity_elements;
  impl->backend_lease = std::move(backend_lease);
  return DeviceBufferLease(std::move(impl));
}

std::uint32_t DeviceBufferLease::rows() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty device buffer lease");
  return impl_->rows;
}

std::uint64_t DeviceBufferLease::row_width() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty device buffer lease");
  return impl_->row_width;
}

std::uint64_t DeviceBufferLease::element_count() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty device buffer lease");
  return impl_->element_count;
}

std::uint64_t DeviceBufferLease::capacity_elements() const {
  if (!impl_)
    invalid_exchange("attempted to access an empty device buffer lease");
  return impl_->capacity_elements;
}

struct BackendCompletion::Impl {
  RankCollectiveContext context;
  ExchangeKind kind = ExchangeKind::ExpertCounts;
  HostCountMatrix host;
  DeviceBufferLease device;
  std::shared_ptr<const BackendCompletionLeaseState> completion_lease;
};

BackendCompletion make_host_backend_completion(
    RankCollectiveContext context, ExchangeKind kind, HostCountMatrix result,
    std::shared_ptr<const BackendCompletionLeaseState> completion_lease) {
  if (!context)
    invalid_exchange("backend completion has empty rank context");
  if (!result)
    invalid_exchange("backend completion has empty host result");
  auto impl = std::make_shared<BackendCompletion::Impl>();
  impl->context = std::move(context);
  impl->kind = kind;
  impl->host = std::move(result);
  impl->completion_lease = std::move(completion_lease);
  return BackendCompletion(std::move(impl));
}

BackendCompletion make_device_backend_completion(
    RankCollectiveContext context, ExchangeKind kind, DeviceBufferLease result,
    std::shared_ptr<const BackendCompletionLeaseState> completion_lease) {
  if (!context)
    invalid_exchange("backend completion has empty rank context");
  if (!result)
    invalid_exchange("backend completion has empty device result");
  auto impl = std::make_shared<BackendCompletion::Impl>();
  impl->context = std::move(context);
  impl->kind = kind;
  impl->device = std::move(result);
  impl->completion_lease = std::move(completion_lease);
  return BackendCompletion(std::move(impl));
}

const RankCollectiveContext &BackendCompletion::context() const {
  if (!impl_)
    invalid_exchange("backend returned an empty completion");
  return impl_->context;
}

ExchangeKind BackendCompletion::kind() const {
  if (!impl_)
    invalid_exchange("backend returned an empty completion");
  return impl_->kind;
}

bool BackendCompletion::has_host_result() const noexcept {
  return impl_ && static_cast<bool>(impl_->host);
}

bool BackendCompletion::has_device_result() const noexcept {
  return impl_ && static_cast<bool>(impl_->device);
}

const HostCountMatrix &BackendCompletion::host_result() const {
  if (!impl_ || !impl_->host)
    invalid_exchange("completion has no host typed result");
  return impl_->host;
}

const DeviceBufferLease &BackendCompletion::device_result() const {
  if (!impl_ || !impl_->device)
    invalid_exchange("completion has no device lease");
  return impl_->device;
}

std::shared_ptr<const BackendSessionState>
CountExchange::backend_session_state(const RankCollectiveContext &context) {
  if (!context)
    invalid_exchange("backend requested state from empty rank context");
  return context.session().impl_->backend_state;
}

std::shared_ptr<const BackendExecutionState>
CountExchange::backend_execution_state(const RankCollectiveContext &context) {
  if (!context)
    invalid_exchange("backend requested state from empty rank context");
  return context.execution().impl_->backend_state;
}

std::shared_ptr<const BackendDeviceLeaseState>
CountExchange::backend_device_state(const DeviceBufferLease &lease) {
  if (!lease)
    invalid_exchange("backend requested state from empty device lease");
  return lease.impl_->backend_lease;
}

namespace {

struct RoundKey {
  std::uint64_t generation = 0;
  ExchangeKind kind = ExchangeKind::ExpertCounts;
  bool operator<(const RoundKey &other) const noexcept {
    if (generation != other.generation)
      return generation < other.generation;
    return static_cast<int>(kind) < static_cast<int>(other.kind);
  }
};

struct Round {
  std::uint64_t row_width = 0;
  CollectiveSession session;
  std::vector<std::optional<std::vector<std::uint64_t>>> rows;
  std::string completion_failure;
};

} // namespace

struct HostFakeCountExchange::SharedState {
  SharedState(std::uint32_t size, std::string identity_value)
      : world_size(size), identity(std::move(identity_value)) {}
  std::uint32_t world_size;
  std::string identity;
  std::mutex mutex;
  std::condition_variable completion;
  std::map<RoundKey, Round> rounds;
  std::string next_start_failure;
  std::string next_completion_failure;
};

namespace {

class HostFakeOperation final : public BackendCountExchangeOperation {
public:
  HostFakeOperation(std::shared_ptr<HostFakeCountExchange::SharedState> state,
                    CountExchangeRequest request)
      : state_(std::move(state)), request_(std::move(request)) {}

  ExchangeState state() const noexcept override {
    try {
      std::lock_guard<std::mutex> lock(state_->mutex);
      if (cancelled_)
        return ExchangeState::Cancelled;
      const RoundKey key{request_.context.session().generation(),
                         request_.kind};
      const auto it = state_->rounds.find(key);
      if (it == state_->rounds.end())
        return ExchangeState::Failed;
      const auto complete =
          std::all_of(it->second.rows.begin(), it->second.rows.end(),
                      [](const auto &row) { return row.has_value(); });
      if (!complete)
        return ExchangeState::Pending;
      return it->second.completion_failure.empty() ? ExchangeState::Complete
                                                   : ExchangeState::Failed;
    } catch (...) {
      return ExchangeState::Failed;
    }
  }

  BackendCompletion wait() override {
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (cancelled_)
      throw std::runtime_error("fake operation was cancelled");
    const RoundKey key{request_.context.session().generation(), request_.kind};
    const auto it = state_->rounds.find(key);
    if (it == state_->rounds.end()) {
      throw std::runtime_error("fake collective round disappeared");
    }
    state_->completion.wait(lock, [&] {
      return cancelled_ ||
             std::all_of(it->second.rows.begin(), it->second.rows.end(),
                         [](const auto &row) { return row.has_value(); });
    });
    if (cancelled_)
      throw std::runtime_error("fake operation was cancelled");
    if (!it->second.completion_failure.empty()) {
      throw std::runtime_error(it->second.completion_failure);
    }
    CountMatrix rows;
    rows.reserve(it->second.rows.size());
    for (const auto &row : it->second.rows)
      rows.push_back(*row);
    return make_host_backend_completion(request_.context, request_.kind,
                                        make_host_count_matrix(rows));
  }

  void cancel() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    cancelled_ = true;
    state_->completion.notify_all();
  }

private:
  std::shared_ptr<HostFakeCountExchange::SharedState> state_;
  CountExchangeRequest request_;
  bool cancelled_ = false;
};

} // namespace

HostFakeCountExchange::HostFakeCountExchange(std::uint32_t world_size,
                                             std::string session_identity)
    : state_(std::make_shared<SharedState>(world_size,
                                           std::move(session_identity))) {
  if (world_size == 0)
    invalid_exchange("fake collective world_size must be positive");
  if (state_->identity.empty()) {
    invalid_exchange("fake collective session identity must not be empty");
  }
}

HostFakeCountExchange::~HostFakeCountExchange() noexcept = default;

std::unique_ptr<BackendCountExchangeOperation>
HostFakeCountExchange::start(const CountExchangeRequest &request) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (!state_->next_start_failure.empty()) {
    auto message = std::move(state_->next_start_failure);
    state_->next_start_failure.clear();
    throw std::runtime_error(message);
  }
  if (!request.context)
    invalid_exchange("fake start received empty rank context");
  const auto &session = request.context.session();
  if (session.world_size() != state_->world_size) {
    invalid_exchange("fake collective world_size mismatch: expected=" +
                     std::to_string(state_->world_size) +
                     ", actual=" + std::to_string(session.world_size()));
  }
  if (session.identity() != state_->identity) {
    invalid_exchange("fake collective session identity mismatch: expected='" +
                     state_->identity + "', actual='" + session.identity() +
                     "'");
  }
  if (!request.input)
    invalid_exchange("fake collective input storage is empty");
  if (request.row_width == 0 || request.input.size() != request.row_width) {
    invalid_exchange("fake collective input width mismatch: rank=" +
                     std::to_string(request.context.calling_rank()) +
                     ", expected=" + std::to_string(request.row_width) +
                     ", actual=" + std::to_string(request.input.size()));
  }

  const RoundKey key{session.generation(), request.kind};
  auto [it, inserted] = state_->rounds.try_emplace(key);
  auto &round = it->second;
  if (inserted) {
    round.row_width = request.row_width;
    round.session = session;
    round.rows.resize(state_->world_size);
    round.completion_failure = std::move(state_->next_completion_failure);
    state_->next_completion_failure.clear();
  } else {
    if (!round.session.same_instance(session)) {
      invalid_exchange(
          "fake collective cross-session mix: expected=same session instance, "
          "actual=different session instance, kind=" +
          std::string(kind_name(request.kind)));
    }
    if (round.row_width != request.row_width) {
      invalid_exchange("fake collective row_width mismatch within kind=" +
                       std::string(kind_name(request.kind)) +
                       ": expected=" + std::to_string(round.row_width) +
                       ", actual=" + std::to_string(request.row_width));
    }
  }
  const auto rank = request.context.calling_rank();
  if (round.rows.at(rank).has_value()) {
    invalid_exchange(
        "rank started fake collective twice: rank=" + std::to_string(rank) +
        ", kind=" + kind_name(request.kind));
  }
  round.rows.at(rank) = request.input.values();
  state_->completion.notify_all();
  return std::make_unique<HostFakeOperation>(state_, request);
}

void HostFakeCountExchange::fail_next_start(std::string message) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->next_start_failure = std::move(message);
}

void HostFakeCountExchange::fail_next_completion(std::string message) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->next_completion_failure = std::move(message);
}

} // namespace megamoe
