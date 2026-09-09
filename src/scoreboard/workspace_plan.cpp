#include "weftmoe/scoreboard/workspace_plan.hpp"

#include <string>

#include "weftmoe/api/errors.hpp"

namespace weftmoe {
namespace {

[[noreturn]] void invalid_plan(const std::string& message) {
    throw NotifyDispatchError(NotifyErrorCode::WorkspaceMismatch, message);
}

WorkspaceRequirements derive_requirements(const DispatchMetadata& metadata) {
    WorkspaceRequirements requirements;
    requirements.canonical_rows = metadata.incoming.expert_row_prefix.back();
    for (const auto& tiles : metadata.incoming.mtile_expected_rows) {
        requirements.expert_mtiles += tiles.size();
    }
    requirements.outgoing_routes = metadata.outgoing.expert_id.size();
    requirements.outgoing_packets = metadata.transport.outgoing_packets.size();
    return requirements;
}

WorkspaceShape derive_shape(const DispatchMetadata& metadata) {
    return WorkspaceShape{metadata.calling_rank,
                          metadata.config.topology.world_size,
                          metadata.config.num_experts,
                          metadata.config.top_k,
                          metadata.config.m_tile_rows};
}

bool same_requirements(const WorkspaceRequirements& lhs,
                       const WorkspaceRequirements& rhs) {
    return lhs.canonical_rows == rhs.canonical_rows &&
           lhs.expert_mtiles == rhs.expert_mtiles &&
           lhs.outgoing_routes == rhs.outgoing_routes &&
           lhs.outgoing_packets == rhs.outgoing_packets;
}

bool same_shape(const WorkspaceShape& lhs, const WorkspaceShape& rhs) {
    return lhs.calling_rank == rhs.calling_rank && lhs.world_size == rhs.world_size &&
           lhs.num_experts == rhs.num_experts && lhs.top_k == rhs.top_k &&
           lhs.m_tile_rows == rhs.m_tile_rows;
}

}  // namespace

const WorkspaceRequirements& WorkspacePlan::requirements() const {
    if (!valid_) invalid_plan("attempted to access a default/invalid WorkspacePlan");
    return requirements_;
}

const WorkspaceShape& WorkspacePlan::shape() const {
    if (!valid_) invalid_plan("attempted to access shape of a default/invalid WorkspacePlan");
    return shape_;
}

const MetadataIdentity& WorkspacePlan::identity() const {
    if (!valid_) invalid_plan("attempted to access identity of a default/invalid WorkspacePlan");
    return identity_;
}

std::uint64_t WorkspacePlan::epoch() const {
    if (!valid_) invalid_plan("attempted to access epoch of a default/invalid WorkspacePlan");
    return epoch_;
}

WorkspacePlan make_workspace_plan(const DispatchHandle& handle) {
    validate_dispatch_metadata(handle);
    const auto& metadata = handle.metadata();
    WorkspacePlan plan;
    plan.valid_ = true;
    plan.identity_ = metadata.identity;
    plan.epoch_ = metadata.identity.generation;
    plan.requirements_ = derive_requirements(metadata);
    plan.shape_ = derive_shape(metadata);
    plan.provenance_ = handle.provenance_;
    return plan;
}

void validate_workspace_plan(const DispatchHandle& handle,
                             const WorkspacePlan& plan) {
    validate_dispatch_metadata(handle);
    const auto& metadata = handle.metadata();
    if (!plan.valid_) invalid_plan("persistent launch rejected a default/invalid WorkspacePlan");
    const auto provenance = plan.provenance_.lock();
    if (!provenance || provenance.get() != handle.provenance_.get()) {
        invalid_plan("persistent launch rejected a WorkspacePlan from another DispatchHandle instance");
    }
    if (plan.identity_.content_hash != metadata.identity.content_hash ||
        plan.identity_.generation != metadata.identity.generation) {
        invalid_plan("workspace metadata identity mismatch for rank=" +
                     std::to_string(metadata.calling_rank));
    }
    if (plan.epoch_ == 0 || plan.epoch_ != metadata.identity.generation) {
        invalid_plan("workspace epoch mismatch for rank=" +
                     std::to_string(metadata.calling_rank) + ": workspace=" +
                     std::to_string(plan.epoch_) + ", metadata=" +
                     std::to_string(metadata.identity.generation));
    }
    const auto expected_shape = derive_shape(metadata);
    if (!same_shape(plan.shape_, expected_shape)) {
        invalid_plan("workspace runtime shape mismatch for rank=" +
                     std::to_string(metadata.calling_rank));
    }
    const auto expected_requirements = derive_requirements(metadata);
    if (!same_requirements(plan.requirements_, expected_requirements)) {
        invalid_plan("workspace capacity mismatch for rank=" +
                     std::to_string(metadata.calling_rank));
    }
}

}  // namespace weftmoe
