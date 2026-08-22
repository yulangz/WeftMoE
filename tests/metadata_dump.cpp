#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/api/notify_dispatch.hpp"
#include "megamoe/metadata/serialize.hpp"

namespace {

megamoe::RuntimeConfig make_config(std::uint32_t nodes, std::uint32_t gpus,
                                   std::uint32_t experts, std::uint32_t top_k,
                                   std::uint32_t mtile) {
    megamoe::RuntimeConfig config;
    config.num_experts = experts;
    config.top_k = top_k;
    config.m_tile_rows = mtile;
    config.topology.num_nodes = nodes;
    config.topology.world_size = nodes * gpus;
    for (std::uint32_t node = 0; node < nodes; ++node) {
        for (std::uint32_t local = 0; local < gpus; ++local) {
            config.topology.rank_to_node.push_back(node);
            config.topology.rank_to_local_rank.push_back(local);
        }
    }
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        config.topology.expert_to_rank.push_back(expert %
                                                 config.topology.world_size);
    }
    return config;
}

std::vector<megamoe::RankRoutingInput> seeded_inputs(std::uint32_t world_size,
                                                     std::uint32_t experts,
                                                     std::uint32_t top_k,
                                                     std::uint64_t seed) {
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed;
    };
    std::vector<megamoe::RankRoutingInput> inputs(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        inputs[rank].num_tokens = next() % 6U;
        for (std::uint64_t token = 0; token < inputs[rank].num_tokens;
             ++token) {
            std::vector<std::uint32_t> candidates(experts);
            for (std::uint32_t i = 0; i < experts; ++i)
                candidates[i] = i;
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                const auto selected = slot + static_cast<std::uint32_t>(
                                                 next() % (experts - slot));
                std::swap(candidates[slot], candidates[selected]);
                inputs[rank].selected_experts.push_back(
                    static_cast<std::int32_t>(candidates[slot]));
            }
        }
    }
    return inputs;
}

void emit_handles(const std::vector<megamoe::DispatchHandle>& handles) {
    std::cout << '[';
    for (std::size_t i = 0; i < handles.size(); ++i) {
        if (i != 0)
            std::cout << ',';
        std::cout << megamoe::serialize_dispatch_metadata(handles[i]);
    }
    std::cout << ']';
}

std::vector<megamoe::DispatchHandle>
build_repeat_checked(const megamoe::RuntimeConfig& config,
                     const std::vector<megamoe::RankRoutingInput>& routes,
                     const std::string& communicator,
                     std::uint64_t generation) {
    auto first = megamoe::build_notify_dispatch_for_testing(
        config, routes, communicator, generation);
    const auto second = megamoe::build_notify_dispatch_for_testing(
        config, routes, communicator, generation);
    if (first.size() != second.size()) {
        throw std::runtime_error("repeat handle count mismatch");
    }
    for (std::size_t rank = 0; rank < first.size(); ++rank) {
        if (megamoe::serialize_dispatch_metadata(first[rank]) !=
            megamoe::serialize_dispatch_metadata(second[rank])) {
            throw std::runtime_error("byte-for-byte repeat mismatch at rank=" +
                                     std::to_string(rank) + ", communicator='" +
                                     communicator + "'");
        }
    }
    return first;
}

} // namespace

int main() {
    std::cout << '{';

    auto targeted_config = make_config(2, 2, 6, 3, 4);
    targeted_config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3};
    std::vector<megamoe::RankRoutingInput> targeted_routes(4);
    targeted_routes[0] = {2, {2, 4, 3, 0, 2, 5}};
    std::cout << "\"targeted\":";
    emit_handles(build_repeat_checked(targeted_config, targeted_routes,
                                      "cross-targeted", 3));

    const auto random_config = make_config(2, 2, 11, 4, 3);
    constexpr std::array<std::uint64_t, 25> random_seeds{
        0x5eed1234ULL, 0x10203040ULL, 0xdeadbeefULL, 0xc001d00dULL,
        0x01234567ULL, 0x89abcdefULL, 0x31415926ULL, 0x27182818ULL,
        0x0badf00dULL, 0xfeedfaceULL, 0x13579bdfULL, 0x2468ace0ULL,
        0xa5a5a5a5ULL, 0x5a5a5a5aULL, 0x11223344ULL, 0x55667788ULL,
        0x99aabbccULL, 0xddeeff00ULL, 0x42424242ULL, 0x77777777ULL,
        0xabcdef01ULL, 0x10fedcbaULL, 0x600dbeefULL, 0x7f4a7c15ULL,
        0xf00dcafeULL};
    for (std::size_t index = 0; index < random_seeds.size(); ++index) {
        const auto random_routes = seeded_inputs(4, 11, 4, random_seeds[index]);
        const auto suffix =
            index < 10 ? "0" + std::to_string(index) : std::to_string(index);
        std::cout << ",\"seeded_" << suffix << "\":";
        emit_handles(build_repeat_checked(
            random_config, random_routes, "cross-random-" + suffix,
            100U + static_cast<std::uint64_t>(index)));
    }

    const auto many_config = make_config(73, 1, 73, 2, 8);
    std::vector<megamoe::RankRoutingInput> many_routes(73);
    many_routes[0] = {1, {0, 72}};
    std::cout << ",\"many_nodes\":";
    emit_handles(
        build_repeat_checked(many_config, many_routes, "cross-nodes", 23));

    auto skew_config = make_config(2, 2, 8, 2, 4);
    std::vector<megamoe::RankRoutingInput> skew_routes(4);
    skew_routes[0] = {3, {0, 1, 0, 2, 0, 3}};
    skew_routes[1] = {2, {0, 4, 0, 5}};
    skew_routes[3] = {1, {0, 6}};
    std::cout << ",\"skew_empty\":";
    emit_handles(
        build_repeat_checked(skew_config, skew_routes, "cross-skew", 29));

    std::cout << "}\n";
    return 0;
}
