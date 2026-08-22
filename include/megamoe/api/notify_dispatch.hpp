#pragma once

#include <vector>

#include "megamoe/config/runtime_config.hpp"
#include "megamoe/metadata/dispatch_handle.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace megamoe {

// Builds the complete deterministic control-plane metadata. No destination
// row is allocated by an arrival-order atomic.
DispatchHandle build_notify_dispatch(
    const RuntimeConfig& config,
    const std::vector<RankRoutingInput>& routes_by_rank,
    CountExchange& count_exchange);

}  // namespace megamoe
