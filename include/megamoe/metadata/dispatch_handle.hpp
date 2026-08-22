#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "megamoe/config/runtime_config.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace megamoe {

struct ReverseRouteMetadata {
    std::uint32_t source_rank = 0;
    std::uint64_t local_token_id = 0;
    std::uint32_t topk_slot = 0;
    std::uint32_t expert_id = 0;
    std::uint32_t destination_rank = 0;
    std::uint64_t dst_row = 0;
};

struct SourceRouteMetadata {
    std::uint64_t num_tokens = 0;
    std::uint32_t top_k = 0;
    std::uint32_t node_mask_words = 0;

    // All route arrays are row-major [num_tokens, top_k].
    std::vector<std::uint32_t> expert_id;
    std::vector<std::uint64_t> local_offset;
    std::vector<std::uint64_t> route_dst_row;
    std::vector<std::uint32_t> route_dst_rank;
    std::vector<std::uint32_t> route_dst_node;
    std::vector<std::uint32_t> route_dst_local_rank;

    // [num_tokens, top_k], where each value is the original top-k slot at
    // that ascending-expert combine position.
    std::vector<std::uint32_t> combine_order;

    // [num_tokens, node_mask_words].
    std::vector<std::uint64_t> destination_node_mask;
};

struct DestinationMetadata {
    std::uint32_t rank = 0;
    // Owned global expert IDs in ascending order.
    std::vector<std::uint32_t> expert_ids;
    std::vector<std::uint64_t> expert_row_count;
    // Exclusive prefix, length expert_ids.size() + 1.
    std::vector<std::uint64_t> expert_row_prefix;
    // Per local expert, valid rows in each M tile. Empty experts have no tiles.
    std::vector<std::vector<std::uint32_t>> mtile_expected_rows;
    // Indexed directly by destination-local dst_row.
    std::vector<ReverseRouteMetadata> reverse_routes;
};

struct DispatchMetadata {
    RuntimeConfig config;
    CountMatrix source_expert_count;
    std::vector<SourceRouteMetadata> sources;
    std::vector<DestinationMetadata> destinations;
};

// Cheap-copy, immutable metadata shared by forward and backward launches.
class DispatchHandle {
public:
    DispatchHandle() = default;
    explicit DispatchHandle(std::shared_ptr<const DispatchMetadata> metadata)
        : metadata_(std::move(metadata)) {}

    const DispatchMetadata& metadata() const { return *metadata_; }
    const DispatchMetadata* operator->() const { return metadata_.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(metadata_); }

private:
    std::shared_ptr<const DispatchMetadata> metadata_;
};

}  // namespace megamoe
