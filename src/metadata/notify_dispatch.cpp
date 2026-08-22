#include "megamoe/api/notify_dispatch.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "megamoe/api/errors.hpp"

namespace megamoe {
namespace {

[[noreturn]] void fail(NotifyErrorCode code, const std::string& message) {
    throw NotifyDispatchError(code, message);
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          const char* context) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        fail(NotifyErrorCode::ArithmeticOverflow,
             std::string("uint64 overflow while computing ") + context);
    }
    return lhs + rhs;
}

std::size_t checked_size(std::uint64_t value, const char* context) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        fail(NotifyErrorCode::ArithmeticOverflow,
             std::string("size_t overflow while computing ") + context);
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs,
                          const char* context) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        fail(NotifyErrorCode::ArithmeticOverflow,
             std::string("uint64 overflow while computing ") + context);
    }
    return lhs * rhs;
}

void validate_config(const RuntimeConfig& config,
                     const std::vector<RankRoutingInput>& routes_by_rank) {
    const auto& topology = config.topology;
    if (config.num_experts == 0 || config.top_k == 0 || config.m_tile_rows == 0) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "num_experts, top_k, and m_tile_rows must all be positive");
    }
    if (topology.world_size == 0 || topology.num_nodes == 0) {
        fail(NotifyErrorCode::InvalidTopology,
             "world_size and num_nodes must both be positive");
    }
    if (routes_by_rank.size() != topology.world_size ||
        topology.rank_to_node.size() != topology.world_size ||
        topology.rank_to_local_rank.size() != topology.world_size ||
        topology.expert_to_rank.size() != config.num_experts) {
        fail(NotifyErrorCode::InvalidTopology,
             "rank routes or topology arrays do not match runtime dimensions");
    }
    if (!config.capacity.expert_rows.empty() &&
        config.capacity.expert_rows.size() != config.num_experts) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "expert capacity must be empty or have num_experts entries");
    }
    if (!config.capacity.workspace_rows_per_rank.empty() &&
        config.capacity.workspace_rows_per_rank.size() != topology.world_size) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "workspace capacity must be empty or have world_size entries");
    }

    std::unordered_set<std::uint64_t> rank_coordinates;
    for (std::uint32_t rank = 0; rank < topology.world_size; ++rank) {
        if (topology.rank_to_node[rank] >= topology.num_nodes) {
            fail(NotifyErrorCode::InvalidTopology, "rank_to_node contains an out-of-range node");
        }
        const auto coordinate =
            (static_cast<std::uint64_t>(topology.rank_to_node[rank]) << 32U) |
            topology.rank_to_local_rank[rank];
        if (!rank_coordinates.insert(coordinate).second) {
            fail(NotifyErrorCode::InvalidTopology,
                 "two ranks share the same (node, local_rank) coordinate");
        }
    }
    for (const auto owner : topology.expert_to_rank) {
        if (owner >= topology.world_size) {
            fail(NotifyErrorCode::InvalidTopology,
                 "expert_to_rank contains an out-of-range rank");
        }
    }
}

std::string route_context(std::uint32_t rank, std::uint64_t token,
                          std::uint32_t slot) {
    std::ostringstream stream;
    stream << "rank=" << rank << ", token=" << token << ", topk_slot=" << slot;
    return stream.str();
}

}  // namespace

DispatchHandle build_notify_dispatch(
    const RuntimeConfig& config,
    const std::vector<RankRoutingInput>& routes_by_rank,
    CountExchange& count_exchange) {
    validate_config(config, routes_by_rank);
    const auto world_size = config.topology.world_size;
    const auto num_experts = config.num_experts;
    const auto top_k = config.top_k;

    auto output = std::make_shared<DispatchMetadata>();
    output->config = config;
    output->source_expert_count.assign(
        world_size, std::vector<std::uint64_t>(num_experts, 0));
    output->sources.resize(world_size);

    const auto node_mask_words = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(config.topology.num_nodes) + 63U) / 64U);
    for (std::uint32_t source_rank = 0; source_rank < world_size; ++source_rank) {
        const auto& input = routes_by_rank[source_rank];
        const auto num_routes = checked_mul(input.num_tokens, top_k, "route count");
        if (input.selected_experts.size() != checked_size(num_routes, "route array")) {
            fail(NotifyErrorCode::InvalidRouteShape,
                 "selected_experts shape mismatch for source rank " +
                     std::to_string(source_rank));
        }

        auto& source = output->sources[source_rank];
        source.num_tokens = input.num_tokens;
        source.top_k = top_k;
        source.node_mask_words = node_mask_words;
        source.expert_id.resize(input.selected_experts.size());
        source.local_offset.resize(input.selected_experts.size());
        source.route_dst_row.resize(input.selected_experts.size());
        source.route_dst_rank.resize(input.selected_experts.size());
        source.route_dst_node.resize(input.selected_experts.size());
        source.route_dst_local_rank.resize(input.selected_experts.size());
        source.combine_order.resize(input.selected_experts.size());
        source.destination_node_mask.assign(
            checked_size(checked_mul(input.num_tokens, node_mask_words,
                                     "destination node masks"),
                         "destination node masks"),
            0);

        auto& local_counts = output->source_expert_count[source_rank];
        for (std::uint64_t token = 0; token < input.num_tokens; ++token) {
            std::unordered_set<std::uint32_t> selected_for_token;
            std::vector<std::uint32_t> slots(top_k);
            std::iota(slots.begin(), slots.end(), 0U);

            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                const auto flat = checked_size(checked_add(checked_mul(token, top_k, "route index"),
                                                          slot, "route index"),
                                               "route index");
                const auto selected = input.selected_experts[flat];
                if (selected < 0 || static_cast<std::uint32_t>(selected) >= num_experts) {
                    fail(NotifyErrorCode::ExpertOutOfRange,
                         "expert ID is out of range at " +
                             route_context(source_rank, token, slot));
                }
                const auto expert = static_cast<std::uint32_t>(selected);
                if (!selected_for_token.insert(expert).second) {
                    fail(NotifyErrorCode::DuplicateExpert,
                         "duplicate expert in one token at " +
                             route_context(source_rank, token, slot));
                }

                const auto destination_rank = config.topology.expert_to_rank[expert];
                const auto destination_node = config.topology.rank_to_node[destination_rank];
                source.expert_id[flat] = expert;
                // This is the stable (local_token_id, topk_slot) scan. The
                // counter value is produced serially in canonical route order,
                // never by concurrent data-arrival atomics.
                source.local_offset[flat] = local_counts[expert]++;
                source.route_dst_rank[flat] = destination_rank;
                source.route_dst_node[flat] = destination_node;
                source.route_dst_local_rank[flat] =
                    config.topology.rank_to_local_rank[destination_rank];

                const auto mask_base = checked_size(
                    checked_mul(token, node_mask_words, "node mask index"),
                    "node mask index");
                source.destination_node_mask[mask_base + destination_node / 64U] |=
                    std::uint64_t{1} << (destination_node % 64U);
            }

            std::stable_sort(slots.begin(), slots.end(), [&](const auto lhs, const auto rhs) {
                const auto base = checked_size(checked_mul(token, top_k, "combine order"),
                                               "combine order");
                return source.expert_id[base + lhs] < source.expert_id[base + rhs];
            });
            const auto base = checked_size(checked_mul(token, top_k, "combine order"),
                                           "combine order");
            for (std::uint32_t position = 0; position < top_k; ++position) {
                source.combine_order[base + position] = slots[position];
            }
        }
    }

    const auto exchanged_counts = count_exchange.exchange(output->source_expert_count);
    if (exchanged_counts.size() != world_size) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "count exchange returned the wrong number of source rows");
    }
    for (std::uint32_t source = 0; source < world_size; ++source) {
        if (exchanged_counts[source].size() != num_experts ||
            exchanged_counts[source] != output->source_expert_count[source]) {
            fail(NotifyErrorCode::CountExchangeFailure,
                 "count exchange did not preserve canonical source-indexed counts");
        }
    }

    output->destinations.resize(world_size);
    std::vector<std::uint32_t> expert_local_index(num_experts, 0);
    for (std::uint32_t destination_rank = 0; destination_rank < world_size;
         ++destination_rank) {
        auto& destination = output->destinations[destination_rank];
        destination.rank = destination_rank;
        for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
            if (config.topology.expert_to_rank[expert] == destination_rank) {
                expert_local_index[expert] =
                    static_cast<std::uint32_t>(destination.expert_ids.size());
                destination.expert_ids.push_back(expert);
            }
        }

        destination.expert_row_prefix.push_back(0);
        for (const auto expert : destination.expert_ids) {
            std::uint64_t row_count = 0;
            for (std::uint32_t source = 0; source < world_size; ++source) {
                row_count = checked_add(row_count, exchanged_counts[source][expert],
                                        "expert row count");
            }
            if (!config.capacity.expert_rows.empty() &&
                row_count > config.capacity.expert_rows[expert]) {
                fail(NotifyErrorCode::ExpertCapacityExceeded,
                     "expert " + std::to_string(expert) + " requires " +
                         std::to_string(row_count) + " rows but capacity is " +
                         std::to_string(config.capacity.expert_rows[expert]));
            }
            destination.expert_row_count.push_back(row_count);
            destination.expert_row_prefix.push_back(
                checked_add(destination.expert_row_prefix.back(), row_count,
                            "expert row prefix"));

            std::vector<std::uint32_t> tile_rows;
            for (std::uint64_t first = 0; first < row_count;) {
                const auto remaining = row_count - first;
                tile_rows.push_back(static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(remaining, config.m_tile_rows)));
                if (remaining <= config.m_tile_rows) break;
                first = checked_add(first, config.m_tile_rows, "M-tile offset");
            }
            destination.mtile_expected_rows.push_back(std::move(tile_rows));
        }
        const auto workspace_rows = destination.expert_row_prefix.back();
        if (!config.capacity.workspace_rows_per_rank.empty() &&
            workspace_rows > config.capacity.workspace_rows_per_rank[destination_rank]) {
            fail(NotifyErrorCode::WorkspaceCapacityExceeded,
                 "destination rank " + std::to_string(destination_rank) +
                     " canonical rows exceed workspace capacity");
        }
        destination.reverse_routes.resize(checked_size(workspace_rows, "reverse routes"));
    }

    CountMatrix source_expert_prefix(
        static_cast<std::size_t>(world_size) + 1,
        std::vector<std::uint64_t>(num_experts, 0));
    for (std::uint32_t source = 0; source < world_size; ++source) {
        for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
            source_expert_prefix[source + 1][expert] = checked_add(
                source_expert_prefix[source][expert], exchanged_counts[source][expert],
                "source expert prefix");
        }
    }

    std::vector<std::vector<bool>> reverse_filled(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        reverse_filled[rank].assign(output->destinations[rank].reverse_routes.size(), false);
    }

    for (std::uint32_t source_rank = 0; source_rank < world_size; ++source_rank) {
        auto& source = output->sources[source_rank];
        for (std::uint64_t token = 0; token < source.num_tokens; ++token) {
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                const auto flat = checked_size(checked_add(checked_mul(token, top_k, "route index"),
                                                          slot, "route index"),
                                               "route index");
                const auto expert = source.expert_id[flat];
                const auto destination_rank = source.route_dst_rank[flat];
                const auto local_expert = expert_local_index[expert];
                const auto rank_base = source_expert_prefix[source_rank][expert];
                const auto within_expert = checked_add(
                    rank_base, source.local_offset[flat], "row within expert");
                const auto dst_row = checked_add(
                    output->destinations[destination_rank]
                        .expert_row_prefix[local_expert],
                    within_expert, "canonical destination row");
                source.route_dst_row[flat] = dst_row;

                auto& destination = output->destinations[destination_rank];
                const auto row_index = checked_size(dst_row, "reverse route row");
                if (row_index >= destination.reverse_routes.size() ||
                    reverse_filled[destination_rank][row_index]) {
                    fail(NotifyErrorCode::CountExchangeFailure,
                         "canonical row generation produced a collision");
                }
                reverse_filled[destination_rank][row_index] = true;
                destination.reverse_routes[row_index] = ReverseRouteMetadata{
                    source_rank, token, slot, expert, destination_rank, dst_row};
            }
        }
    }

    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        if (!std::all_of(reverse_filled[rank].begin(), reverse_filled[rank].end(),
                         [](bool value) { return value; })) {
            fail(NotifyErrorCode::CountExchangeFailure,
                 "canonical row generation left a reverse-route hole");
        }
    }

    return DispatchHandle(std::move(output));
}

}  // namespace megamoe
