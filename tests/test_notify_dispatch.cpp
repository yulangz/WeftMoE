#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/api/errors.hpp"
#include "megamoe/api/notify_dispatch.hpp"
#include "megamoe/kernels/persistent_contract.hpp"
#include "megamoe/scoreboard/workspace_plan.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace {

using megamoe::DispatchHandle;
using megamoe::InMemoryCountExchange;
using megamoe::NotifyDispatchError;
using megamoe::NotifyErrorCode;
using megamoe::RankRoutingInput;
using megamoe::RuntimeConfig;
using megamoe::Topology;

#define REQUIRE(condition)                                                        \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(std::string("requirement failed: ") +        \
                                     #condition + " at line " +                  \
                                     std::to_string(__LINE__));                   \
        }                                                                         \
    } while (false)

Topology make_topology(std::uint32_t world_size, std::uint32_t num_nodes,
                       std::uint32_t num_experts) {
    Topology topology;
    topology.world_size = world_size;
    topology.num_nodes = num_nodes;
    topology.rank_to_node.resize(world_size);
    topology.rank_to_local_rank.resize(world_size);
    std::vector<std::uint32_t> next_local(num_nodes, 0);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        const auto node = rank * num_nodes / world_size;
        topology.rank_to_node[rank] = node;
        topology.rank_to_local_rank[rank] = next_local[node]++;
    }
    topology.expert_to_rank.resize(num_experts);
    for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
        topology.expert_to_rank[expert] = expert % world_size;
    }
    return topology;
}

RuntimeConfig make_config(std::uint32_t world_size, std::uint32_t num_nodes,
                          std::uint32_t num_experts, std::uint32_t top_k,
                          std::uint32_t m_tile_rows) {
    RuntimeConfig config;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.m_tile_rows = m_tile_rows;
    config.topology = make_topology(world_size, num_nodes, num_experts);
    return config;
}

template <typename T>
void append_vector(std::ostringstream& out, const std::vector<T>& values) {
    out << values.size() << ':';
    for (const auto& value : values) out << value << ',';
    out << ';';
}

std::string fingerprint(const DispatchHandle& handle) {
    const auto& metadata = handle.metadata();
    std::ostringstream out;
    out << metadata.config.num_experts << ',' << metadata.config.top_k << ','
        << metadata.config.m_tile_rows << ','
        << metadata.config.topology.world_size << ','
        << metadata.config.topology.num_nodes << ';';
    append_vector(out, metadata.config.topology.rank_to_node);
    append_vector(out, metadata.config.topology.rank_to_local_rank);
    append_vector(out, metadata.config.topology.expert_to_rank);
    for (const auto& row : metadata.source_expert_count) append_vector(out, row);
    for (const auto& source : metadata.sources) {
        out << source.num_tokens << ',' << source.top_k << ',' << source.node_mask_words << ';';
        append_vector(out, source.expert_id);
        append_vector(out, source.local_offset);
        append_vector(out, source.route_dst_row);
        append_vector(out, source.route_dst_rank);
        append_vector(out, source.route_dst_node);
        append_vector(out, source.route_dst_local_rank);
        append_vector(out, source.combine_order);
        append_vector(out, source.destination_node_mask);
    }
    for (const auto& destination : metadata.destinations) {
        out << destination.rank << ';';
        append_vector(out, destination.expert_ids);
        append_vector(out, destination.expert_row_count);
        append_vector(out, destination.expert_row_prefix);
        for (const auto& tile_rows : destination.mtile_expected_rows) {
            append_vector(out, tile_rows);
        }
        for (const auto& reverse : destination.reverse_routes) {
            out << reverse.source_rank << ',' << reverse.local_token_id << ','
                << reverse.topk_slot << ',' << reverse.expert_id << ','
                << reverse.destination_rank << ',' << reverse.dst_row << ';';
        }
    }
    return out.str();
}

void verify_global_invariants(const DispatchHandle& handle) {
    const auto& metadata = handle.metadata();
    std::vector<std::set<std::pair<std::uint32_t, std::uint64_t>>> seen(
        metadata.destinations.size());
    for (std::uint32_t source_rank = 0; source_rank < metadata.sources.size();
         ++source_rank) {
        const auto& source = metadata.sources[source_rank];
        for (std::size_t route = 0; route < source.expert_id.size(); ++route) {
            const auto destination_rank = source.route_dst_rank[route];
            REQUIRE(seen[destination_rank]
                        .insert({source.expert_id[route], source.route_dst_row[route]})
                        .second);
            const auto& reverse = metadata.destinations[destination_rank]
                                      .reverse_routes[source.route_dst_row[route]];
            REQUIRE(reverse.source_rank == source_rank);
            REQUIRE(reverse.local_token_id == route / source.top_k);
            REQUIRE(reverse.topk_slot == route % source.top_k);
            REQUIRE(reverse.expert_id == source.expert_id[route]);
        }

        for (std::uint64_t token = 0; token < source.num_tokens; ++token) {
            const auto base = static_cast<std::size_t>(token * source.top_k);
            std::uint32_t previous_expert = 0;
            for (std::uint32_t position = 0; position < source.top_k; ++position) {
                const auto slot = source.combine_order[base + position];
                REQUIRE(slot < source.top_k);
                const auto expert = source.expert_id[base + slot];
                if (position != 0) REQUIRE(previous_expert < expert);
                previous_expert = expert;
            }
        }
    }
    for (const auto& destination : metadata.destinations) {
        REQUIRE(destination.expert_row_prefix.size() == destination.expert_ids.size() + 1);
        REQUIRE(destination.expert_row_prefix.back() == destination.reverse_routes.size());
        for (std::size_t i = 0; i < destination.expert_ids.size(); ++i) {
            REQUIRE(destination.expert_row_prefix[i + 1] -
                        destination.expert_row_prefix[i] ==
                    destination.expert_row_count[i]);
            std::uint64_t tile_sum = 0;
            for (const auto rows : destination.mtile_expected_rows[i]) {
                REQUIRE(rows > 0 && rows <= metadata.config.m_tile_rows);
                tile_sum += rows;
            }
            REQUIRE(tile_sum == destination.expert_row_count[i]);
        }
    }
}

void expect_error(NotifyErrorCode expected, const std::function<void()>& action) {
    try {
        action();
    } catch (const NotifyDispatchError& error) {
        REQUIRE(error.code() == expected);
        return;
    }
    throw std::runtime_error("expected NotifyDispatchError was not raised");
}

void test_single_rank_exact_layout() {
    auto config = make_config(1, 1, 5, 2, 4);
    std::vector<RankRoutingInput> inputs{{3, {3, 1, 1, 4, 3, 0}}};
    InMemoryCountExchange transport;
    const auto handle = megamoe::build_notify_dispatch(config, inputs, transport);
    const auto& source = handle->sources[0];
    REQUIRE(source.local_offset == std::vector<std::uint64_t>({0, 0, 1, 0, 1, 0}));
    REQUIRE(source.route_dst_row == std::vector<std::uint64_t>({3, 1, 2, 5, 4, 0}));
    REQUIRE(source.combine_order == std::vector<std::uint32_t>({1, 0, 0, 1, 1, 0}));
    REQUIRE(source.destination_node_mask == std::vector<std::uint64_t>({1, 1, 1}));

    const auto& destination = handle->destinations[0];
    REQUIRE(destination.expert_row_count ==
            std::vector<std::uint64_t>({1, 2, 0, 2, 1}));
    REQUIRE(destination.expert_row_prefix ==
            std::vector<std::uint64_t>({0, 1, 3, 3, 5, 6}));
    REQUIRE(destination.mtile_expected_rows[2].empty());
    verify_global_invariants(handle);

    const auto workspace = megamoe::make_workspace_plan(handle);
    REQUIRE(workspace.world_size == 1);
    REQUIRE(workspace.ranks[0].canonical_rows == 6);
    REQUIRE(workspace.ranks[0].route_metadata_entries == 6);
}

std::vector<RankRoutingInput> generated_inputs(std::uint32_t world_size,
                                               std::uint32_t num_experts,
                                               std::uint32_t top_k) {
    std::vector<RankRoutingInput> inputs(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        inputs[rank].num_tokens = rank % 4 + 1;
        for (std::uint64_t token = 0; token < inputs[rank].num_tokens; ++token) {
            const auto base = static_cast<std::uint32_t>((rank * 3 + token * 5) % num_experts);
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                inputs[rank].selected_experts.push_back(
                    static_cast<std::int32_t>((base + slot * 7) % num_experts));
            }
        }
    }
    return inputs;
}

void test_single_node_eight_rank_and_repeatability() {
    auto config = make_config(8, 1, 23, 3, 4);
    const auto inputs = generated_inputs(8, 23, 3);
    InMemoryCountExchange first_transport;
    InMemoryCountExchange second_transport;
    const auto first = megamoe::build_notify_dispatch(config, inputs, first_transport);
    const auto second = megamoe::build_notify_dispatch(config, inputs, second_transport);
    REQUIRE(fingerprint(first) == fingerprint(second));
    verify_global_invariants(first);

    bool found_partial_tile = false;
    for (const auto& destination : first->destinations) {
        for (const auto& tiles : destination.mtile_expected_rows) {
            if (!tiles.empty() && tiles.back() < config.m_tile_rows) found_partial_tile = true;
        }
    }
    REQUIRE(found_partial_tile);
}

void test_multinode_arrival_order_independence() {
    auto config = make_config(6, 3, 17, 3, 4);
    config.topology.rank_to_node = {0, 0, 0, 1, 2, 2};
    config.topology.rank_to_local_rank = {0, 1, 2, 0, 0, 1};
    config.topology.expert_to_rank = {5, 0, 3, 1, 4, 2, 0, 5, 2,
                                      4, 1, 3, 5, 2, 4, 0, 1};
    auto inputs = generated_inputs(6, 17, 3);
    inputs[2] = RankRoutingInput{};  // A rank with zero local tokens is valid.
    InMemoryCountExchange ordered({0, 1, 2, 3, 4, 5});
    InMemoryCountExchange shuffled({4, 1, 5, 0, 3, 2});
    const auto canonical = megamoe::build_notify_dispatch(config, inputs, ordered);
    const auto reordered = megamoe::build_notify_dispatch(config, inputs, shuffled);
    REQUIRE(fingerprint(canonical) == fingerprint(reordered));
    verify_global_invariants(canonical);

    for (const auto& source : canonical->sources) {
        for (std::size_t i = 0; i < source.route_dst_rank.size(); ++i) {
            const auto rank = source.route_dst_rank[i];
            REQUIRE(source.route_dst_node[i] == config.topology.rank_to_node[rank]);
            REQUIRE(source.route_dst_local_rank[i] ==
                    config.topology.rank_to_local_rank[rank]);
        }
    }
}

void test_nonuniform_empty_expert_and_partial_tile() {
    auto config = make_config(2, 1, 4, 2, 4);
    config.topology.expert_to_rank = {0, 0, 1, 1};
    std::vector<RankRoutingInput> inputs{
        {3, {0, 1, 0, 2, 0, 2}},
        {2, {0, 1, 0, 2}},
    };
    InMemoryCountExchange transport;
    const auto handle = megamoe::build_notify_dispatch(config, inputs, transport);
    REQUIRE(handle->destinations[0].expert_row_count ==
            std::vector<std::uint64_t>({5, 2}));
    REQUIRE(handle->destinations[0].mtile_expected_rows[0] ==
            std::vector<std::uint32_t>({4, 1}));
    REQUIRE(handle->destinations[1].expert_row_count ==
            std::vector<std::uint64_t>({3, 0}));
    REQUIRE(handle->destinations[1].mtile_expected_rows[1].empty());
    REQUIRE(handle->sources[0].route_dst_row[0] == 0);
    REQUIRE(handle->sources[0].route_dst_row[2] == 1);
    REQUIRE(handle->sources[0].route_dst_row[4] == 2);
    REQUIRE(handle->sources[1].route_dst_row[0] == 3);
    REQUIRE(handle->sources[1].route_dst_row[2] == 4);
    verify_global_invariants(handle);
}

void test_invalid_routes_and_capacity_fail_explicitly() {
    InMemoryCountExchange transport;
    auto duplicate_config = make_config(1, 1, 4, 2, 4);
    expect_error(NotifyErrorCode::DuplicateExpert, [&] {
        megamoe::build_notify_dispatch(duplicate_config, {{1, {2, 2}}}, transport);
    });
    expect_error(NotifyErrorCode::ExpertOutOfRange, [&] {
        megamoe::build_notify_dispatch(duplicate_config, {{1, {2, 4}}}, transport);
    });

    auto capacity_config = make_config(2, 1, 4, 2, 4);
    capacity_config.capacity.expert_rows = {2, 10, 10, 10};
    const std::vector<RankRoutingInput> capacity_inputs{
        {2, {0, 1, 0, 2}},
        {1, {0, 3}},
    };
    expect_error(NotifyErrorCode::ExpertCapacityExceeded, [&] {
        megamoe::build_notify_dispatch(capacity_config, capacity_inputs, transport);
    });

    capacity_config.capacity.expert_rows.clear();
    capacity_config.capacity.workspace_rows_per_rank = {3, 100};
    expect_error(NotifyErrorCode::WorkspaceCapacityExceeded, [&] {
        megamoe::build_notify_dispatch(capacity_config, capacity_inputs, transport);
    });
}

void test_bad_arrival_order_fails() {
    auto config = make_config(2, 1, 4, 2, 4);
    const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {2, 3}}};
    InMemoryCountExchange bad_transport({0, 0});
    expect_error(NotifyErrorCode::CountExchangeFailure, [&] {
        megamoe::build_notify_dispatch(config, inputs, bad_transport);
    });
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests{
        {"single_rank_exact_layout", test_single_rank_exact_layout},
        {"single_node_eight_rank_and_repeatability", test_single_node_eight_rank_and_repeatability},
        {"multinode_arrival_order_independence", test_multinode_arrival_order_independence},
        {"nonuniform_empty_expert_and_partial_tile", test_nonuniform_empty_expert_and_partial_tile},
        {"invalid_routes_and_capacity_fail_explicitly", test_invalid_routes_and_capacity_fail_explicitly},
        {"bad_arrival_order_fails", test_bad_arrival_order_fails},
    };

    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << "/" << tests.size() << " tests passed\n";
    return 0;
}
