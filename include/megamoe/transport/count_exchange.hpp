#pragma once

#include <cstdint>
#include <vector>

namespace megamoe {

using CountMatrix = std::vector<std::vector<std::uint64_t>>;

// Transport owns count movement only. Its result is always indexed by
// canonical source rank, independent of delivery order.
class CountExchange {
public:
    virtual ~CountExchange() = default;
    virtual CountMatrix exchange(const CountMatrix& local_counts) = 0;
};

// Development transport. arrival_order models arbitrary message delivery but
// writes each payload into its source-indexed row.
class InMemoryCountExchange final : public CountExchange {
public:
    explicit InMemoryCountExchange(std::vector<std::uint32_t> arrival_order = {});
    CountMatrix exchange(const CountMatrix& local_counts) override;

private:
    std::vector<std::uint32_t> arrival_order_;
};

}  // namespace megamoe
