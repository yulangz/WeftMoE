#include "megamoe/transport/count_exchange.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

#include "megamoe/api/errors.hpp"

namespace megamoe {

InMemoryCountExchange::InMemoryCountExchange(std::vector<std::uint32_t> arrival_order)
    : arrival_order_(std::move(arrival_order)) {}

CountMatrix InMemoryCountExchange::exchange(const CountMatrix& local_counts) {
    const auto world_size = local_counts.size();
    std::vector<std::uint32_t> order = arrival_order_;
    if (order.empty()) {
        order.resize(world_size);
        std::iota(order.begin(), order.end(), 0U);
    }
    if (order.size() != world_size) {
        throw NotifyDispatchError(NotifyErrorCode::CountExchangeFailure,
                                  "arrival order does not contain every source rank");
    }

    std::vector<bool> seen(world_size, false);
    CountMatrix exchanged(world_size);
    for (const auto source_rank : order) {
        if (source_rank >= world_size || seen[source_rank]) {
            throw NotifyDispatchError(NotifyErrorCode::CountExchangeFailure,
                                      "arrival order is not a rank permutation");
        }
        seen[source_rank] = true;
        exchanged[source_rank] = local_counts[source_rank];
    }
    return exchanged;
}

}  // namespace megamoe
