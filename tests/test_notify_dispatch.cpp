#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/api/errors.hpp"
#include "megamoe/api/notify_dispatch.hpp"
#include "megamoe/kernels/persistent_contract.hpp"
#include "megamoe/metadata/serialize.hpp"
#include "megamoe/scoreboard/workspace_plan.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace {

using megamoe::CollectiveContext;
using megamoe::CompletedCollectives;
using megamoe::DispatchHandle;
using megamoe::ExchangeKind;
using megamoe::ExchangeState;
using megamoe::HostFakeCountExchange;
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

Topology homogeneous_topology(std::uint32_t num_nodes, std::uint32_t gpus_per_node,
                              std::uint32_t num_experts) {
    Topology topology;
    topology.num_nodes = num_nodes;
    topology.world_size = num_nodes * gpus_per_node;
    for (std::uint32_t node = 0; node < num_nodes; ++node) {
        for (std::uint32_t local = 0; local < gpus_per_node; ++local) {
            topology.rank_to_node.push_back(node);
            topology.rank_to_local_rank.push_back(local);
        }
    }
    for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
        topology.expert_to_rank.push_back(expert % topology.world_size);
    }
    return topology;
}

RuntimeConfig make_config(std::uint32_t nodes, std::uint32_t gpus_per_node,
                          std::uint32_t experts, std::uint32_t top_k,
                          std::uint32_t m_tile) {
    RuntimeConfig config;
    config.num_experts = experts;
    config.top_k = top_k;
    config.m_tile_rows = m_tile;
    config.topology = homogeneous_topology(nodes, gpus_per_node, experts);
    return config;
}

void expect_error(NotifyErrorCode code, const std::function<void()>& action,
                  const std::string& required_text = {}) {
    try {
        action();
    } catch (const NotifyDispatchError& error) {
        REQUIRE(error.code() == code);
        if (!required_text.empty()) {
            REQUIRE(std::string(error.what()).find(required_text) != std::string::npos);
        }
        return;
    }
    throw std::runtime_error("expected NotifyDispatchError was not raised");
}

std::vector<RankRoutingInput> seeded_inputs(std::uint32_t world_size,
                                            std::uint32_t num_experts,
                                            std::uint32_t top_k,
                                            std::uint64_t seed) {
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed;
    };
    std::vector<RankRoutingInput> inputs(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        inputs[rank].num_tokens = next() % 6U;
        for (std::uint64_t token = 0; token < inputs[rank].num_tokens; ++token) {
            std::vector<std::uint32_t> experts(num_experts);
            for (std::uint32_t i = 0; i < num_experts; ++i) experts[i] = i;
            for (std::uint32_t i = 0; i < top_k; ++i) {
                const auto selected = i + static_cast<std::uint32_t>(next() % (num_experts - i));
                std::swap(experts[i], experts[selected]);
                inputs[rank].selected_experts.push_back(
                    static_cast<std::int32_t>(experts[i]));
            }
        }
    }
    return inputs;
}

void verify_all_rank_mapping(const std::vector<DispatchHandle>& handles) {
    const auto world_size = handles.size();
    std::vector<std::set<std::pair<std::uint32_t, std::uint64_t>>> seen(world_size);
    std::vector<std::set<std::pair<std::uint32_t, std::uint64_t>>> packet_targets(
        world_size);
    for (std::uint32_t source = 0; source < world_size; ++source) {
        const auto& metadata = handles[source].metadata();
        REQUIRE(metadata.calling_rank == source);
        const auto& outgoing = metadata.outgoing;
        for (std::size_t i = 0; i < outgoing.expert_id.size(); ++i) {
            const auto destination = outgoing.route_dst_rank[i];
            REQUIRE(seen[destination]
                        .insert({outgoing.expert_id[i], outgoing.route_dst_row[i]})
                        .second);
            REQUIRE(outgoing.reverse_routes[i].source_rank == source);
            REQUIRE(outgoing.reverse_routes[i].dst_row == outgoing.route_dst_row[i]);
        }
        for (const auto& packet : metadata.transport.outgoing_packets) {
            for (const auto& target : packet.fanout) {
                const auto destination_rank =
                    metadata.config.topology.expert_to_rank[target.expert_id];
                REQUIRE(packet_targets[destination_rank]
                            .insert({target.expert_id, target.dst_row})
                            .second);
            }
        }
    }
    REQUIRE(seen == packet_targets);
    for (std::uint32_t destination = 0; destination < world_size; ++destination) {
        const auto& incoming = handles[destination]->incoming;
        REQUIRE(incoming.expert_row_prefix.back() == incoming.incoming_rows.size());
        for (std::size_t row = 0; row < incoming.incoming_rows.size(); ++row) {
            REQUIRE(incoming.incoming_rows[row].dst_row == row);
        }
    }
}

void test_rank_local_three_stage_and_async_completion() {
    auto config = make_config(2, 2, 8, 3, 4);
    const auto inputs = seeded_inputs(4, 8, 3, 0x1234ULL);
    std::vector<megamoe::LocalNotifyMetadata> locals;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        locals.push_back(megamoe::build_rank_local_histogram(config, rank, inputs[rank]));
    }
    HostFakeCountExchange backend(4, "rank-local-test");
    std::vector<CollectiveContext> contexts;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        contexts.push_back(CollectiveContext{rank, 4, "rank-local-test", 9,
                                             {0x1000U + rank, 0x2000U}});
    }
    std::vector<std::unique_ptr<megamoe::CountExchangeOperation>> count_ops;
    count_ops.push_back(
        megamoe::start_count_exchange(locals[0], backend, contexts[0]));
    REQUIRE(count_ops[0]->state() == ExchangeState::Pending);
    auto rank0_wait = std::async(std::launch::async, [&] {
        return megamoe::complete_exchange(*count_ops[0], contexts[0],
                                          ExchangeKind::ExpertCounts, 8);
    });
    REQUIRE(rank0_wait.wait_for(std::chrono::milliseconds(20)) ==
            std::future_status::timeout);
    for (std::uint32_t rank = 1; rank < 4; ++rank) {
        count_ops.push_back(
            megamoe::start_count_exchange(locals[rank], backend, contexts[rank]));
    }
    for (const auto& op : count_ops) REQUIRE(op->state() == ExchangeState::Complete);
    std::vector<megamoe::CountExchangeResult> counts;
    counts.push_back(rank0_wait.get());
    for (std::uint32_t rank = 1; rank < 4; ++rank) {
        counts.push_back(megamoe::complete_exchange(
            *count_ops[rank], contexts[rank], ExchangeKind::ExpertCounts, 8));
    }
    std::vector<std::unique_ptr<megamoe::CountExchangeOperation>> summary_ops;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        summary_ops.push_back(megamoe::start_notify_summary_exchange(
            locals[rank], backend, contexts[rank]));
    }
    std::vector<DispatchHandle> handles;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        auto summary = megamoe::complete_exchange(
            *summary_ops[rank], contexts[rank], ExchangeKind::NotifyPacketSummary, 2);
        handles.push_back(megamoe::materialize_rank_metadata(
            locals[rank], CompletedCollectives{counts[rank], std::move(summary)}));
    }
    verify_all_rank_mapping(handles);
}

void test_wrapper_composes_rank_local_path_and_is_repeatable() {
    const auto config = make_config(1, 8, 23, 3, 4);
    const auto inputs = seeded_inputs(8, 23, 3, 0xdecafbadULL);
    const auto first = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "repeat", 7);
    const auto second = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "repeat", 7);
    REQUIRE(first.size() == second.size());
    for (std::size_t rank = 0; rank < first.size(); ++rank) {
        REQUIRE(megamoe::serialize_dispatch_metadata(first[rank]) ==
                megamoe::serialize_dispatch_metadata(second[rank]));
    }
    verify_all_rank_mapping(first);
}

void test_packet_node_dedup_and_signal_cardinality() {
    auto config = make_config(2, 2, 6, 3, 4);
    config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3};
    std::vector<RankRoutingInput> inputs(4);
    inputs[0] = RankRoutingInput{2, {2, 4, 3, 0, 2, 5}};
    const auto handles = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "packet", 3);
    const auto& transport = handles[0]->transport;
    REQUIRE(transport.outgoing_packets.size() == 3);
    REQUIRE(transport.outgoing_packets[0].identity.destination_node == 0);
    REQUIRE(transport.outgoing_packets[0].local_token_id == 1);
    REQUIRE(transport.outgoing_packets[1].identity.destination_node == 1);
    REQUIRE(transport.outgoing_packets[1].local_token_id == 0);
    REQUIRE(transport.outgoing_packets[1].fanout.size() == 3);
    REQUIRE(transport.outgoing_packets[1].fanout[0].expert_id == 2);
    REQUIRE(transport.outgoing_packets[1].fanout[1].expert_id == 3);
    REQUIRE(transport.outgoing_packets[1].fanout[2].expert_id == 4);
    REQUIRE(transport.outgoing_packets[1].proxy_rank == 2);
    REQUIRE(transport.outgoing_signals[0].expected_packets == 1);
    REQUIRE(transport.outgoing_signals[1].expected_packets == 2);
    REQUIRE(handles[2]->transport.incoming_proxy_signals[0].source_rank == 0);
    REQUIRE(handles[2]->transport.incoming_proxy_signals[0].expected_signals == 2);
    verify_all_rank_mapping(handles);
}

void test_multiword_node_mask_over_seventy_nodes() {
    const auto config = make_config(73, 1, 73, 2, 8);
    std::vector<RankRoutingInput> inputs(73);
    inputs[0] = RankRoutingInput{1, {0, 72}};
    const auto handles = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "many-nodes", 11);
    const auto& outgoing = handles[0]->outgoing;
    REQUIRE(outgoing.node_mask_words == 2);
    REQUIRE(outgoing.destination_node_mask ==
            std::vector<std::uint64_t>({1, std::uint64_t{1} << 8U}));
    REQUIRE(handles[0]->transport.outgoing_packets.size() == 2);
    REQUIRE(handles[0]->transport.outgoing_signals.size() == 73);
    verify_all_rank_mapping(handles);
}

void test_empty_rank_expert_partial_tile_and_extreme_skew() {
    auto config = make_config(2, 2, 8, 2, 4);
    std::vector<RankRoutingInput> inputs(4);
    inputs[0] = RankRoutingInput{3, {0, 1, 0, 2, 0, 3}};
    inputs[1] = RankRoutingInput{2, {0, 4, 0, 5}};
    inputs[3] = RankRoutingInput{1, {0, 6}};
    const auto handles = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "skew", 2);
    REQUIRE(handles[2]->outgoing.num_tokens == 0);
    REQUIRE(handles[0]->incoming.expert_row_count[0] == 6);
    REQUIRE(handles[0]->incoming.mtile_expected_rows[0] ==
            std::vector<std::uint32_t>({4, 2}));
    bool found_empty_expert = false;
    for (const auto count : handles[3]->incoming.expert_row_count) {
        if (count == 0) found_empty_expert = true;
    }
    REQUIRE(found_empty_expert);
    verify_all_rank_mapping(handles);
}

void test_topology_rejections_and_topk_config() {
    auto non_homogeneous = make_config(1, 6, 8, 2, 4);
    non_homogeneous.topology.num_nodes = 4;
    expect_error(NotifyErrorCode::InvalidTopology, [&] {
        megamoe::build_rank_local_histogram(non_homogeneous, 0, {});
    }, "heterogeneous");
    auto sparse = make_config(2, 2, 8, 2, 4);
    sparse.topology.rank_to_local_rank[1] = 2;
    expect_error(NotifyErrorCode::InvalidTopology, [&] {
        megamoe::build_rank_local_histogram(sparse, 0, {});
    }, "local coordinate");
    auto duplicate = make_config(2, 2, 8, 2, 4);
    duplicate.topology.rank_to_node[2] = 0;
    expect_error(NotifyErrorCode::InvalidTopology, [&] {
        megamoe::build_rank_local_histogram(duplicate, 0, {});
    });
    auto bad_node = make_config(2, 2, 8, 2, 4);
    bad_node.topology.rank_to_node[0] = 2;
    expect_error(NotifyErrorCode::InvalidTopology, [&] {
        megamoe::build_rank_local_histogram(bad_node, 0, {});
    }, "node coordinate");
    auto topk = make_config(1, 1, 2, 3, 4);
    expect_error(NotifyErrorCode::InvalidRuntimeConfig, [&] {
        megamoe::build_rank_local_histogram(topk, 0, RankRoutingInput{0, {}});
    }, "top_k exceeds");
}

void test_route_capacity_and_integer_overflow_errors() {
    auto config = make_config(1, 2, 4, 2, 4);
    expect_error(NotifyErrorCode::DuplicateExpert, [&] {
        megamoe::build_rank_local_histogram(config, 0, {1, {2, 2}});
    }, "rank=0, token=0, topk_slot=1");
    expect_error(NotifyErrorCode::ExpertOutOfRange, [&] {
        megamoe::build_rank_local_histogram(config, 0, {1, {2, 4}});
    }, "actual=4");
    expect_error(NotifyErrorCode::ArithmeticOverflow, [&] {
        megamoe::build_rank_local_histogram(
            config, 0,
            RankRoutingInput{std::numeric_limits<std::uint64_t>::max(), {}});
    }, "route count");
    config.capacity.expert_rows = {1, 100, 100, 100};
    const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {0, 2}}};
    expect_error(NotifyErrorCode::ExpertCapacityExceeded, [&] {
        megamoe::build_notify_dispatch_for_testing(config, inputs, "capacity", 1);
    }, "expert=0");
    config.capacity.expert_rows.clear();
    config.capacity.workspace_rows_per_rank = {2, 100};
    expect_error(NotifyErrorCode::WorkspaceCapacityExceeded, [&] {
        megamoe::build_notify_dispatch_for_testing(config, inputs, "workspace", 1);
    }, "destination_rank=0");

    auto base_config = make_config(1, 2, 4, 2, 4);
    auto local = megamoe::build_rank_local_histogram(base_config, 0, {1, {0, 1}});
    const CollectiveContext context{0, 2, "overflow", 1, {}};
    auto counts = megamoe::make_host_exchange_result(
        context, ExchangeKind::ExpertCounts,
        {{1, 1, 0, 0},
         {std::numeric_limits<std::uint64_t>::max(), 0, 0, 0}});
    auto summary = megamoe::make_host_exchange_result(
        context, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
    expect_error(NotifyErrorCode::ArithmeticOverflow, [&] {
        megamoe::materialize_rank_metadata(
            local, CompletedCollectives{counts, summary});
    }, "source prefix");
}

void test_exchange_context_and_backend_failures() {
    const auto config = make_config(1, 2, 4, 2, 4);
    const auto local = megamoe::build_rank_local_histogram(config, 0, {1, {0, 1}});
    HostFakeCountExchange backend(2, "good");
    expect_error(NotifyErrorCode::CountExchangeFailure, [&] {
        megamoe::start_count_exchange(
            local, backend, CollectiveContext{0, 3, "good", 1, {}});
    }, "world size mismatch");
    expect_error(NotifyErrorCode::CountExchangeFailure, [&] {
        megamoe::start_count_exchange(
            local, backend, CollectiveContext{0, 2, "bad", 1, {}});
    }, "communicator");
    backend.fail_next_start("injected-start-failure");
    expect_error(NotifyErrorCode::CountExchangeFailure, [&] {
        megamoe::start_count_exchange(
            local, backend, CollectiveContext{0, 2, "good", 2, {}});
    }, "injected-start-failure");

    const auto local1 = megamoe::build_rank_local_histogram(config, 1, {});
    backend.fail_next_completion("injected-completion-failure");
    const CollectiveContext context0{0, 2, "good", 3, {}};
    const CollectiveContext context1{1, 2, "good", 3, {}};
    auto op0 = megamoe::start_count_exchange(local, backend, context0);
    auto op1 = megamoe::start_count_exchange(local1, backend, context1);
    REQUIRE(op0->state() == ExchangeState::Failed);
    REQUIRE(op1->state() == ExchangeState::Failed);
    expect_error(NotifyErrorCode::CountExchangeFailure, [&] {
        megamoe::complete_exchange(*op0, context0, ExchangeKind::ExpertCounts, 4);
    }, "injected-completion-failure");
}

void test_handle_workspace_identity_and_checked_access() {
    DispatchHandle empty;
    expect_error(NotifyErrorCode::InvalidHandle, [&] { (void)empty.metadata(); });
    megamoe::WorkspacePlan empty_plan;
    expect_error(NotifyErrorCode::WorkspaceMismatch,
                 [&] { (void)empty_plan.requirements(); });
    const auto config = make_config(1, 2, 4, 2, 4);
    const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {2, 3}}};
    const auto first = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "identity", 5);
    const auto second = megamoe::build_notify_dispatch_for_testing(
        config, inputs, "identity", 5);
    const auto plan = megamoe::make_workspace_plan(first[0]);
    megamoe::validate_workspace_plan(first[0], plan);
    (void)megamoe::ForwardPersistentInputs::checked(first[0], plan);
    (void)megamoe::BackwardPersistentInputs::checked(first[0], plan);
    expect_error(NotifyErrorCode::WorkspaceMismatch, [&] {
        megamoe::validate_workspace_plan(second[0], plan);
    }, "another DispatchHandle");
    expect_error(NotifyErrorCode::WorkspaceMismatch, [&] {
        megamoe::validate_workspace_plan(first[1], plan);
    }, "another DispatchHandle");
    expect_error(NotifyErrorCode::WorkspaceMismatch, [&] {
        megamoe::validate_workspace_plan(first[0], empty_plan);
    }, "default");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests{
        {"rank_local_three_stage_and_async_completion",
         test_rank_local_three_stage_and_async_completion},
        {"wrapper_composes_rank_local_path_and_is_repeatable",
         test_wrapper_composes_rank_local_path_and_is_repeatable},
        {"packet_node_dedup_and_signal_cardinality",
         test_packet_node_dedup_and_signal_cardinality},
        {"multiword_node_mask_over_seventy_nodes",
         test_multiword_node_mask_over_seventy_nodes},
        {"empty_rank_expert_partial_tile_and_extreme_skew",
         test_empty_rank_expert_partial_tile_and_extreme_skew},
        {"topology_rejections_and_topk_config",
         test_topology_rejections_and_topk_config},
        {"route_capacity_and_integer_overflow_errors",
         test_route_capacity_and_integer_overflow_errors},
        {"exchange_context_and_backend_failures",
         test_exchange_context_and_backend_failures},
        {"handle_workspace_identity_and_checked_access",
         test_handle_workspace_identity_and_checked_access},
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
