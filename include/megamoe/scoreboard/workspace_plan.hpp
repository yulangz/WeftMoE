#pragma once

#include <cstdint>
#include <vector>

#include "megamoe/metadata/dispatch_handle.hpp"

namespace megamoe {

struct RankWorkspacePlan {
    std::uint64_t canonical_rows = 0;
    std::uint64_t expert_mtiles = 0;
    std::uint64_t route_metadata_entries = 0;
};

struct WorkspacePlan {
    std::uint32_t world_size = 0;
    std::uint32_t m_tile_rows = 0;
    std::vector<RankWorkspacePlan> ranks;
};

// Shape-only plan. Byte sizing is deliberately deferred until activation and
// scoreboard dtypes are selected by a device backend.
WorkspacePlan make_workspace_plan(const DispatchHandle& handle);

}  // namespace megamoe
