#include "megamoe/transport/count_exchange.hpp"

#include <algorithm>
#include <condition_variable>
#include <map>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace megamoe {
namespace {

struct RoundKey {
    std::uint64_t generation = 0;
    ExchangeKind kind = ExchangeKind::ExpertCounts;

    bool operator<(const RoundKey& other) const {
        if (generation != other.generation) return generation < other.generation;
        return static_cast<int>(kind) < static_cast<int>(other.kind);
    }
};

struct Round {
    std::uint64_t row_width = 0;
    std::vector<std::optional<std::vector<std::uint64_t>>> rows;
    std::string completion_failure;
};

}  // namespace

struct HostFakeCountExchange::SharedState {
    SharedState(std::uint32_t size, std::string identity)
        : world_size(size), communicator_id(std::move(identity)) {}

    std::uint32_t world_size;
    std::string communicator_id;
    std::mutex mutex;
    std::condition_variable completion;
    std::map<RoundKey, Round> rounds;
    std::string next_start_failure;
    std::string next_completion_failure;
};

namespace {

class HostFakeOperation final : public CountExchangeOperation {
public:
    HostFakeOperation(std::shared_ptr<HostFakeCountExchange::SharedState> state,
                      CountExchangeRequest request)
        : state_(std::move(state)), request_(std::move(request)) {}

    ExchangeState state() const noexcept override {
        try {
            std::lock_guard<std::mutex> lock(state_->mutex);
            const auto it = state_->rounds.find(
                RoundKey{request_.collective.generation, request_.kind});
            if (it == state_->rounds.end()) return ExchangeState::Failed;
            const auto complete = std::all_of(
                it->second.rows.begin(), it->second.rows.end(),
                [](const auto& row) { return row.has_value(); });
            if (!complete) return ExchangeState::Pending;
            return it->second.completion_failure.empty() ? ExchangeState::Complete
                                                         : ExchangeState::Failed;
        } catch (...) {
            return ExchangeState::Failed;
        }
    }

    std::string error_message() const override {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto it = state_->rounds.find(
            RoundKey{request_.collective.generation, request_.kind});
        if (it == state_->rounds.end()) return "fake collective round disappeared";
        return it->second.completion_failure;
    }

    CountExchangeResult wait() override {
        std::unique_lock<std::mutex> lock(state_->mutex);
        const auto it = state_->rounds.find(
            RoundKey{request_.collective.generation, request_.kind});
        if (it == state_->rounds.end()) {
            throw std::runtime_error("fake collective round disappeared");
        }
        state_->completion.wait(lock, [&] {
            return std::all_of(it->second.rows.begin(), it->second.rows.end(),
                               [](const auto& row) { return row.has_value(); });
        });
        if (!it->second.completion_failure.empty()) {
            throw std::runtime_error(it->second.completion_failure);
        }

        CountMatrix rows;
        rows.reserve(it->second.rows.size());
        for (const auto& row : it->second.rows) rows.push_back(*row);
        return make_host_exchange_result(request_.collective, request_.kind, rows);
    }

private:
    std::shared_ptr<HostFakeCountExchange::SharedState> state_;
    // Retains input buffer ownership through completion as required by contract.
    CountExchangeRequest request_;
};

}  // namespace

HostFakeCountExchange::HostFakeCountExchange(std::uint32_t world_size,
                                             std::string communicator_id)
    : state_(std::make_shared<SharedState>(world_size, std::move(communicator_id))) {
    if (world_size == 0) throw std::invalid_argument("fake collective world_size must be positive");
    if (state_->communicator_id.empty()) {
        throw std::invalid_argument("fake collective communicator identity must not be empty");
    }
}

HostFakeCountExchange::~HostFakeCountExchange() = default;

std::unique_ptr<CountExchangeOperation> HostFakeCountExchange::start(
    const CountExchangeRequest& request) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->next_start_failure.empty()) {
        auto message = std::move(state_->next_start_failure);
        state_->next_start_failure.clear();
        throw std::runtime_error(message);
    }
    if (request.collective.world_size != state_->world_size) {
        std::ostringstream message;
        message << "fake collective world mismatch: actual="
                << request.collective.world_size << ", expected=" << state_->world_size;
        throw std::invalid_argument(message.str());
    }
    if (request.collective.communicator_id != state_->communicator_id) {
        throw std::invalid_argument("fake collective communicator mismatch: actual='" +
                                    request.collective.communicator_id + "', expected='" +
                                    state_->communicator_id + "'");
    }
    if (request.collective.calling_rank >= state_->world_size) {
        throw std::invalid_argument("fake collective calling rank is out of range");
    }
    if (request.input.location != MemoryLocation::Host) {
        throw std::invalid_argument("host fake collective accepts host input only");
    }
    if (!request.input.owner || request.input.address == 0) {
        throw std::invalid_argument("collective input buffer address/owner is null");
    }
    if (request.row_width == 0 || request.input.elements != request.row_width) {
        std::ostringstream message;
        message << "collective input width mismatch for rank="
                << request.collective.calling_rank << ": actual="
                << request.input.elements << ", expected=" << request.row_width;
        throw std::invalid_argument(message.str());
    }

    const RoundKey key{request.collective.generation, request.kind};
    auto [it, inserted] = state_->rounds.try_emplace(key);
    auto& round = it->second;
    if (inserted) {
        round.row_width = request.row_width;
        round.rows.resize(state_->world_size);
        round.completion_failure = std::move(state_->next_completion_failure);
        state_->next_completion_failure.clear();
    } else if (round.row_width != request.row_width) {
        throw std::invalid_argument("collective row width differs within one round");
    }
    const auto rank = request.collective.calling_rank;
    if (round.rows[rank].has_value()) {
        throw std::invalid_argument("rank started the same fake collective round twice");
    }
    const auto input = reinterpret_cast<const std::uint64_t*>(request.input.address);
    round.rows[rank] = std::vector<std::uint64_t>(
        input, input + static_cast<std::size_t>(request.input.elements));
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

IntegerBuffer make_host_integer_buffer(
    std::shared_ptr<const std::vector<std::uint64_t>> values) {
    if (!values) throw std::invalid_argument("host integer buffer owner is null");
    IntegerBuffer buffer;
    buffer.location = MemoryLocation::Host;
    buffer.elements = values->size();
    buffer.address = values->empty()
                         ? 0
                         : reinterpret_cast<std::uintptr_t>(values->data());
    buffer.owner = std::static_pointer_cast<const void>(std::move(values));
    return buffer;
}

CountExchangeResult make_host_exchange_result(
    CollectiveContext context, ExchangeKind kind, const CountMatrix& rows) {
    if (rows.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("host exchange result row count exceeds uint32");
    }
    const std::size_t width = rows.empty() ? 0 : rows.front().size();
    for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        if (rows[rank].size() != width) {
            throw std::invalid_argument("host exchange result has ragged row at rank=" +
                                        std::to_string(rank));
        }
    }
    if (width != 0 && rows.size() >
                          std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("host exchange result element count overflow");
    }
    auto flat = std::make_shared<std::vector<std::uint64_t>>();
    flat->reserve(rows.size() * width);
    for (const auto& row : rows) flat->insert(flat->end(), row.begin(), row.end());
    CountExchangeResult result;
    result.collective = std::move(context);
    result.kind = kind;
    result.rows = static_cast<std::uint32_t>(rows.size());
    result.row_width = width;
    result.output = make_host_integer_buffer(flat);
    return result;
}

CountMatrix CountExchangeResult::copy_host_rows() const {
    if (output.location != MemoryLocation::Host) {
        throw std::invalid_argument("cannot copy device collective output in host materializer");
    }
    if (rows != 0 && row_width >
                         std::numeric_limits<std::uint64_t>::max() / rows) {
        throw std::overflow_error("collective result element count overflow");
    }
    const auto expected = static_cast<std::uint64_t>(rows) * row_width;
    if (expected > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("collective output exceeds host addressable size");
    }
    if (output.elements != expected || !output.owner ||
        (expected != 0 && output.address == 0)) {
        throw std::invalid_argument("collective output descriptor is inconsistent");
    }
    if (row_width > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("collective row width exceeds size_t");
    }
    const auto* data = reinterpret_cast<const std::uint64_t*>(output.address);
    CountMatrix matrix(rows,
                       std::vector<std::uint64_t>(static_cast<std::size_t>(row_width)));
    for (std::uint32_t rank = 0; rank < rows; ++rank) {
        const auto offset = static_cast<std::size_t>(rank) *
                            static_cast<std::size_t>(row_width);
        std::copy_n(data + offset, static_cast<std::size_t>(row_width),
                    matrix[rank].begin());
    }
    return matrix;
}

}  // namespace megamoe
