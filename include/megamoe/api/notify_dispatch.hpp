#pragma once

#include <memory>
#include <string>
#include <vector>

#include "megamoe/config/runtime_config.hpp"
#include "megamoe/metadata/dispatch_handle.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace megamoe {

struct LocalNotifyMetadata {
    RuntimeConfig config;
    std::uint32_t calling_rank = 0;
    SourceRouteMetadata outgoing;
    std::shared_ptr<const std::vector<std::uint64_t>> local_expert_count;
    std::shared_ptr<const std::vector<std::uint64_t>> packet_count_by_destination_node;
};

struct CompletedCollectives {
    CountExchangeResult expert_counts;
    CountExchangeResult packet_summary;
};

// Stage 1: reads only this calling rank's routes.
LocalNotifyMetadata build_rank_local_histogram(
    const RuntimeConfig& config,
    std::uint32_t calling_rank,
    const RankRoutingInput& local_routes);

// Stage 2: asynchronous integer collectives. Backend failures are normalized
// by these wrappers to NotifyDispatchError(CountExchangeFailure) with context.
std::unique_ptr<CountExchangeOperation> start_count_exchange(
    const LocalNotifyMetadata& local,
    CountExchange& backend,
    const CollectiveContext& context);

std::unique_ptr<CountExchangeOperation> start_notify_summary_exchange(
    const LocalNotifyMetadata& local,
    CountExchange& backend,
    const CollectiveContext& context);

CountExchangeResult complete_exchange(
    CountExchangeOperation& operation,
    const CollectiveContext& expected_context,
    ExchangeKind expected_kind,
    std::uint64_t expected_row_width);

// Stage 3: pure rank-local materialization from local histogram plus exchanged
// integer summaries. No backend calls and no other rank's routes are required.
DispatchHandle materialize_rank_metadata(
    const LocalNotifyMetadata& local,
    const CompletedCollectives& collectives);

// Reference/test convenience only. It composes the rank-local production path
// and must not be used as a device-side entry point.
std::vector<DispatchHandle> build_notify_dispatch_for_testing(
    const RuntimeConfig& config,
    const std::vector<RankRoutingInput>& routes_by_rank,
    const std::string& communicator_id = "host-fake",
    std::uint64_t generation = 1);

}  // namespace megamoe
