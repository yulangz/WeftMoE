#pragma once

#include <cstdint>
#include <memory>

#include "megamoe/metadata/dispatch_handle.hpp"

MEGAMOE_PUBLIC_API_BEGIN

namespace megamoe {

struct WorkspaceRequirements {
    std::uint64_t canonical_rows = 0;
    std::uint64_t expert_mtiles = 0;
    std::uint64_t outgoing_routes = 0;
    std::uint64_t outgoing_packets = 0;
};

struct WorkspaceShape {
    std::uint32_t calling_rank = 0;
    std::uint32_t world_size = 0;
    std::uint32_t num_experts = 0;
    std::uint32_t top_k = 0;
    std::uint32_t m_tile_rows = 0;
};

class WorkspacePlan {
public:
    WorkspacePlan() = default;

    const WorkspaceRequirements& requirements() const;
    const WorkspaceShape& shape() const;
    const MetadataIdentity& identity() const;
    std::uint64_t epoch() const;
    explicit operator bool() const noexcept { return valid_; }

private:
    bool valid_ = false;
    MetadataIdentity identity_;
    std::uint64_t epoch_ = 0;
    WorkspaceRequirements requirements_;
    WorkspaceShape shape_;
    std::weak_ptr<const DispatchHandle::ProvenanceToken> provenance_;

    friend WorkspacePlan make_workspace_plan(const DispatchHandle&);
    friend void validate_workspace_plan(const DispatchHandle&, const WorkspacePlan&);
};

WorkspacePlan make_workspace_plan(const DispatchHandle& handle);
void validate_workspace_plan(const DispatchHandle& handle,
                             const WorkspacePlan& plan);

}  // namespace megamoe

MEGAMOE_PUBLIC_API_END
