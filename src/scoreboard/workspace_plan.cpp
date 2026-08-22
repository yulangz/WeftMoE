#include "megamoe/scoreboard/workspace_plan.hpp"

#include <stdexcept>

namespace megamoe {

WorkspacePlan make_workspace_plan(const DispatchHandle& handle) {
    if (!handle) {
        throw std::invalid_argument("dispatch handle is empty");
    }
    const auto& metadata = handle.metadata();
    WorkspacePlan plan;
    plan.world_size = metadata.config.topology.world_size;
    plan.m_tile_rows = metadata.config.m_tile_rows;
    plan.ranks.resize(metadata.destinations.size());
    for (std::size_t rank = 0; rank < metadata.destinations.size(); ++rank) {
        const auto& destination = metadata.destinations[rank];
        auto& rank_plan = plan.ranks[rank];
        rank_plan.canonical_rows = destination.expert_row_prefix.back();
        rank_plan.route_metadata_entries = destination.reverse_routes.size();
        for (const auto& tiles : destination.mtile_expected_rows) {
            rank_plan.expert_mtiles += tiles.size();
        }
    }
    return plan;
}

}  // namespace megamoe
