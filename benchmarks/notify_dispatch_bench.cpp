#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "megamoe/api/notify_dispatch.hpp"
#include "megamoe/transport/count_exchange.hpp"

int main() {
    constexpr std::uint32_t world_size = 8;
    constexpr std::uint32_t num_experts = 384;
    constexpr std::uint32_t top_k = 6;
    constexpr std::uint64_t tokens_per_rank = 8192;

    megamoe::RuntimeConfig config;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.m_tile_rows = 64;
    config.topology.world_size = world_size;
    config.topology.num_nodes = 1;
    config.topology.rank_to_node.assign(world_size, 0);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        config.topology.rank_to_local_rank.push_back(rank);
    }
    for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
        config.topology.expert_to_rank.push_back(expert % world_size);
    }

    std::vector<megamoe::RankRoutingInput> inputs(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        inputs[rank].num_tokens = tokens_per_rank;
        inputs[rank].selected_experts.reserve(tokens_per_rank * top_k);
        for (std::uint64_t token = 0; token < tokens_per_rank; ++token) {
            const auto base = (rank * 13 + token * 17) % num_experts;
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                inputs[rank].selected_experts.push_back(
                    static_cast<std::int32_t>((base + slot * 53) % num_experts));
            }
        }
    }

    megamoe::InMemoryCountExchange transport;
    const auto start = std::chrono::steady_clock::now();
    const auto handle = megamoe::build_notify_dispatch(config, inputs, transport);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start);
    std::uint64_t total_rows = 0;
    for (const auto& destination : handle->destinations) {
        total_rows += destination.expert_row_prefix.back();
    }
    std::cout << "routes=" << total_rows << " elapsed_ms=" << elapsed.count() << '\n';
}
