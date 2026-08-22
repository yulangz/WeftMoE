#pragma once

#include "megamoe/metadata/dispatch_handle.hpp"
#include "megamoe/scoreboard/workspace_plan.hpp"

namespace megamoe {

// These launch contracts reserve the module boundary for the SM90 persistent
// kernels. No launch function is exposed in phase 1: forward/backward kernels
// and RDMA transport are explicitly not implemented yet.
struct ForwardPersistentInputs {
    DispatchHandle dispatch;
    WorkspacePlan workspace;
};

struct BackwardPersistentInputs {
    DispatchHandle dispatch;
    WorkspacePlan workspace;
};

}  // namespace megamoe
