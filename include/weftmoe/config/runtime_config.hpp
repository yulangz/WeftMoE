#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "weftmoe/version.hpp"

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe {

// All topology values are runtime data.  Expert ownership may be non-contiguous.
struct Topology {
    std::uint32_t world_size = 0;
    std::uint32_t num_nodes = 0;
    std::vector<std::uint32_t> rank_to_node;
    std::vector<std::uint32_t> rank_to_local_rank;
    std::vector<std::uint32_t> expert_to_rank;
};

struct CapacityLimits {
    // Empty means unbounded. Otherwise these arrays must match num_experts and
    // world_size respectively.
    std::vector<std::uint64_t> expert_rows;
    std::vector<std::uint64_t> workspace_rows_per_rank;
};

struct RuntimeConfig {
    std::uint32_t num_experts = 0;
    std::uint32_t top_k = 0;
    std::uint32_t m_tile_rows = 0;
    Topology topology;
    CapacityLimits capacity;
};

struct RankRoutingInput {
    std::uint64_t num_tokens = 0;
    // Row-major [num_tokens, top_k].
    std::vector<std::int32_t> selected_experts;
};

// Shared production validation used by host and optional device frontends.
void validate_runtime_config(const RuntimeConfig &config);

}  // namespace weftmoe

WEFTMOE_PUBLIC_API_END
