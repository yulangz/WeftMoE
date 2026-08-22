#pragma once

#include <utility>

#include "megamoe/metadata/dispatch_handle.hpp"
#include "megamoe/scoreboard/workspace_plan.hpp"

namespace megamoe {

class ForwardPersistentInputs {
public:
    static ForwardPersistentInputs checked(const DispatchHandle& dispatch,
                                           const WorkspacePlan& workspace) {
        validate_workspace_plan(dispatch, workspace);
        return ForwardPersistentInputs(dispatch, workspace);
    }

    const DispatchHandle& dispatch() const { return dispatch_; }
    const WorkspacePlan& workspace() const { return workspace_; }

private:
    ForwardPersistentInputs(DispatchHandle dispatch, WorkspacePlan workspace)
        : dispatch_(std::move(dispatch)), workspace_(std::move(workspace)) {}
    DispatchHandle dispatch_;
    WorkspacePlan workspace_;
};

class BackwardPersistentInputs {
public:
    static BackwardPersistentInputs checked(const DispatchHandle& dispatch,
                                            const WorkspacePlan& workspace) {
        validate_workspace_plan(dispatch, workspace);
        return BackwardPersistentInputs(dispatch, workspace);
    }

private:
    BackwardPersistentInputs(DispatchHandle dispatch, WorkspacePlan workspace)
        : dispatch_(std::move(dispatch)), workspace_(std::move(workspace)) {}
    DispatchHandle dispatch_;
    WorkspacePlan workspace_;
};

}  // namespace megamoe
