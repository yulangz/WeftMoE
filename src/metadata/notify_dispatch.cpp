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
                          const std::string& context) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        fail(NotifyErrorCode::ArithmeticOverflow,
             "uint64 overflow while computing " + context + ": lhs=" +
                 std::to_string(lhs) + ", rhs=" + std::to_string(rhs));
    }
    return lhs + rhs;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs,
                          const std::string& context) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        fail(NotifyErrorCode::ArithmeticOverflow,
             "uint64 overflow while computing " + context + ": lhs=" +
                 std::to_string(lhs) + ", rhs=" + std::to_string(rhs));
    }
    return lhs * rhs;
}

std::size_t checked_size(std::uint64_t value, const std::string& context) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        fail(NotifyErrorCode::ArithmeticOverflow,
             "size_t overflow while computing " + context + ": value=" +
                 std::to_string(value));
    }
    return static_cast<std::size_t>(value);
}

struct DerivedTopology {
    std::uint32_t gpus_per_node = 0;
    // [node, local_rank] -> global rank.
    std::vector<std::uint32_t> rank_at;
};

DerivedTopology validate_config(const RuntimeConfig& config) {
    const auto& topology = config.topology;
    if (config.num_experts == 0 || config.top_k == 0 || config.m_tile_rows == 0) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "num_experts, top_k, and m_tile_rows must be positive: experts=" +
                 std::to_string(config.num_experts) + ", top_k=" +
                 std::to_string(config.top_k) + ", m_tile_rows=" +
                 std::to_string(config.m_tile_rows));
    }
    if (config.top_k > config.num_experts) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "top_k exceeds num_experts even for zero-token input: top_k=" +
                 std::to_string(config.top_k) + ", num_experts=" +
                 std::to_string(config.num_experts));
    }
    if (topology.world_size == 0 || topology.num_nodes == 0) {
        fail(NotifyErrorCode::InvalidTopology,
             "world_size and num_nodes must be positive: world_size=" +
                 std::to_string(topology.world_size) + ", num_nodes=" +
                 std::to_string(topology.num_nodes));
    }
    if (topology.world_size % topology.num_nodes != 0) {
        fail(NotifyErrorCode::InvalidTopology,
             "heterogeneous topology is unsupported: world_size=" +
                 std::to_string(topology.world_size) + " is not divisible by num_nodes=" +
                 std::to_string(topology.num_nodes));
    }
    if (topology.rank_to_node.size() != topology.world_size ||
        topology.rank_to_local_rank.size() != topology.world_size) {
        fail(NotifyErrorCode::InvalidTopology,
             "rank coordinate arrays must contain every rank exactly once: node_entries=" +
                 std::to_string(topology.rank_to_node.size()) + ", local_entries=" +
                 std::to_string(topology.rank_to_local_rank.size()) + ", expected=" +
                 std::to_string(topology.world_size));
    }
    if (topology.expert_to_rank.size() != config.num_experts) {
        fail(NotifyErrorCode::InvalidTopology,
             "expert ownership size mismatch: actual=" +
                 std::to_string(topology.expert_to_rank.size()) + ", expected=" +
                 std::to_string(config.num_experts));
    }
    if (!config.capacity.expert_rows.empty() &&
        config.capacity.expert_rows.size() != config.num_experts) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "expert capacity size mismatch: actual=" +
                 std::to_string(config.capacity.expert_rows.size()) + ", expected=" +
                 std::to_string(config.num_experts));
    }
    if (!config.capacity.workspace_rows_per_rank.empty() &&
        config.capacity.workspace_rows_per_rank.size() != topology.world_size) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "workspace capacity size mismatch: actual=" +
                 std::to_string(config.capacity.workspace_rows_per_rank.size()) +
                 ", expected=" + std::to_string(topology.world_size));
    }

    DerivedTopology derived;
    derived.gpus_per_node = topology.world_size / topology.num_nodes;
    derived.rank_at.assign(topology.world_size, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> ranks_per_node(topology.num_nodes, 0);
    for (std::uint32_t rank = 0; rank < topology.world_size; ++rank) {
        const auto node = topology.rank_to_node[rank];
        const auto local_rank = topology.rank_to_local_rank[rank];
        if (node >= topology.num_nodes) {
            fail(NotifyErrorCode::InvalidTopology,
                 "rank has out-of-range node coordinate: rank=" + std::to_string(rank) +
                     ", node=" + std::to_string(node) + ", expected node in [0," +
                     std::to_string(topology.num_nodes) + ")");
        }
        if (local_rank >= derived.gpus_per_node) {
            fail(NotifyErrorCode::InvalidTopology,
                 "rank has sparse/out-of-range local coordinate: rank=" +
                     std::to_string(rank) + ", node=" + std::to_string(node) +
                     ", local_rank=" + std::to_string(local_rank) + ", expected [0," +
                     std::to_string(derived.gpus_per_node) + ")");
        }
        const auto coordinate = static_cast<std::size_t>(node) * derived.gpus_per_node +
                                local_rank;
        if (derived.rank_at[coordinate] != std::numeric_limits<std::uint32_t>::max()) {
            fail(NotifyErrorCode::InvalidTopology,
                 "duplicate topology coordinate: node=" + std::to_string(node) +
                     ", local_rank=" + std::to_string(local_rank) + ", ranks=" +
                     std::to_string(derived.rank_at[coordinate]) + " and " +
                     std::to_string(rank));
        }
        derived.rank_at[coordinate] = rank;
        ++ranks_per_node[node];
    }
    for (std::uint32_t node = 0; node < topology.num_nodes; ++node) {
        if (ranks_per_node[node] != derived.gpus_per_node) {
            fail(NotifyErrorCode::InvalidTopology,
                 "homogeneous topology violation at node=" + std::to_string(node) +
                     ": ranks=" + std::to_string(ranks_per_node[node]) +
                     ", expected=" + std::to_string(derived.gpus_per_node));
        }
        for (std::uint32_t local = 0; local < derived.gpus_per_node; ++local) {
            const auto coordinate = static_cast<std::size_t>(node) *
                                        derived.gpus_per_node +
                                    local;
            if (derived.rank_at[coordinate] == std::numeric_limits<std::uint32_t>::max()) {
                fail(NotifyErrorCode::InvalidTopology,
                     "missing same-rail coordinate: node=" + std::to_string(node) +
                         ", local_rank=" + std::to_string(local));
            }
        }
    }
    for (std::uint32_t expert = 0; expert < config.num_experts; ++expert) {
        const auto owner = topology.expert_to_rank[expert];
        if (owner >= topology.world_size) {
            fail(NotifyErrorCode::InvalidTopology,
                 "expert owner out of range: expert=" + std::to_string(expert) +
                     ", owner=" + std::to_string(owner) + ", world_size=" +
                     std::to_string(topology.world_size));
        }
    }
    return derived;
}

std::string route_context(std::uint32_t rank, std::uint64_t token,
                          std::uint32_t slot) {
    return "rank=" + std::to_string(rank) + ", token=" + std::to_string(token) +
           ", topk_slot=" + std::to_string(slot);
}

bool same_execution(const ExecutionContext& lhs, const ExecutionContext& rhs) {
    return lhs.stream == rhs.stream && lhs.backend_context == rhs.backend_context;
}

void validate_collective_context(const CollectiveContext& context,
                                 const LocalNotifyMetadata& local,
                                 const char* operation) {
    if (context.calling_rank != local.calling_rank) {
        fail(NotifyErrorCode::CountExchangeFailure,
             std::string(operation) + " calling rank mismatch: actual=" +
                 std::to_string(context.calling_rank) + ", expected=" +
                 std::to_string(local.calling_rank));
    }
    if (context.world_size != local.config.topology.world_size) {
        fail(NotifyErrorCode::CountExchangeFailure,
             std::string(operation) + " world size mismatch for rank=" +
                 std::to_string(local.calling_rank) + ": actual=" +
                 std::to_string(context.world_size) + ", expected=" +
                 std::to_string(local.config.topology.world_size));
    }
    if (context.communicator_id.empty()) {
        fail(NotifyErrorCode::CountExchangeFailure,
             std::string(operation) + " communicator identity is empty for rank=" +
                 std::to_string(local.calling_rank));
    }
    if (context.generation == 0) {
        fail(NotifyErrorCode::CountExchangeFailure,
             std::string(operation) + " generation/epoch must be nonzero for rank=" +
                 std::to_string(local.calling_rank));
    }
}

std::unique_ptr<CountExchangeOperation> start_exchange(
    const LocalNotifyMetadata& local, CountExchange& backend,
    const CollectiveContext& context, ExchangeKind kind,
    std::shared_ptr<const std::vector<std::uint64_t>> row,
    std::uint64_t row_width) {
    validate_collective_context(context, local, "collective start");
    CountExchangeRequest request;
    request.collective = context;
    request.kind = kind;
    request.row_width = row_width;
    request.input = make_host_integer_buffer(std::move(row));
    try {
        auto operation = backend.start(request);
        if (!operation) throw std::runtime_error("backend returned a null operation");
        return operation;
    } catch (const NotifyDispatchError&) {
        throw;
    } catch (const std::exception& error) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective start failed: rank=" + std::to_string(context.calling_rank) +
                 ", world_size=" + std::to_string(context.world_size) +
                 ", communicator='" + context.communicator_id + "', generation=" +
                 std::to_string(context.generation) + ", backend_error='" +
                 error.what() + "'");
    }
}

void validate_result_shape(const CountExchangeResult& result,
                           const CollectiveContext& context,
                           ExchangeKind kind, std::uint64_t row_width) {
    if (result.collective.calling_rank != context.calling_rank ||
        result.collective.world_size != context.world_size ||
        result.collective.communicator_id != context.communicator_id ||
        result.collective.generation != context.generation ||
        !same_execution(result.collective.execution, context.execution) ||
        result.kind != kind) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective completion context mismatch for rank=" +
                 std::to_string(context.calling_rank) + ", communicator='" +
                 context.communicator_id + "', generation=" +
                 std::to_string(context.generation));
    }
    if (result.output.location != MemoryLocation::Host) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "host metadata materialization requires host collective output for rank=" +
                 std::to_string(context.calling_rank));
    }
    if (result.rows != context.world_size) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective output world mismatch: rows=" +
                 std::to_string(result.rows) + ", expected=" +
                 std::to_string(context.world_size));
    }
    if (result.row_width != row_width) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective output row width mismatch: actual=" +
                 std::to_string(result.row_width) + ", expected=" +
                 std::to_string(row_width));
    }
    const auto expected_elements = checked_mul(context.world_size, row_width,
                                               "collective output elements");
    if (result.output.elements != expected_elements || !result.output.owner ||
        (expected_elements != 0 && result.output.address == 0)) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective output buffer descriptor mismatch: elements=" +
                 std::to_string(result.output.elements) + ", expected=" +
                 std::to_string(expected_elements));
    }
}

class Hash64 {
public:
    void add(std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value_ ^= (value >> shift) & 0xffU;
            value_ *= 1099511628211ULL;
        }
    }
    void add(const std::string& value) {
        add(value.size());
        for (const auto ch : value) {
            value_ ^= static_cast<unsigned char>(ch);
            value_ *= 1099511628211ULL;
        }
    }
    template <typename T>
    void add_vector(const std::vector<T>& values) {
        add(values.size());
        for (const auto value : values) add(static_cast<std::uint64_t>(value));
    }
    std::uint64_t value() const { return value_; }

private:
    std::uint64_t value_ = 1469598103934665603ULL;
};

std::uint64_t hash_metadata(const DispatchMetadata& metadata) {
    Hash64 hash;
    const auto& config = metadata.config;
    hash.add(config.num_experts);
    hash.add(config.top_k);
    hash.add(config.m_tile_rows);
    hash.add(config.topology.world_size);
    hash.add(config.topology.num_nodes);
    hash.add_vector(config.topology.rank_to_node);
    hash.add_vector(config.topology.rank_to_local_rank);
    hash.add_vector(config.topology.expert_to_rank);
    hash.add_vector(config.capacity.expert_rows);
    hash.add_vector(config.capacity.workspace_rows_per_rank);
    hash.add(metadata.calling_rank);
    hash.add(metadata.communicator_id);
    for (const auto& row : metadata.source_expert_count) hash.add_vector(row);
    const auto& outgoing = metadata.outgoing;
    hash.add(outgoing.num_tokens);
    hash.add(outgoing.top_k);
    hash.add(outgoing.node_mask_words);
    hash.add_vector(outgoing.expert_id);
    hash.add_vector(outgoing.local_offset);
    hash.add_vector(outgoing.route_dst_row);
    hash.add_vector(outgoing.route_dst_rank);
    hash.add_vector(outgoing.route_dst_node);
    hash.add_vector(outgoing.route_dst_local_rank);
    hash.add_vector(outgoing.combine_order);
    hash.add_vector(outgoing.destination_node_mask);
    for (const auto& reverse : outgoing.reverse_routes) {
        hash.add(reverse.source_rank);
        hash.add(reverse.local_token_id);
        hash.add(reverse.topk_slot);
        hash.add(reverse.expert_id);
        hash.add(reverse.destination_rank);
        hash.add(reverse.dst_row);
    }
    const auto& incoming = metadata.incoming;
    hash.add(incoming.rank);
    hash.add_vector(incoming.expert_ids);
    hash.add_vector(incoming.expert_row_count);
    hash.add_vector(incoming.expert_row_prefix);
    for (const auto& tiles : incoming.mtile_expected_rows) hash.add_vector(tiles);
    for (const auto& row : incoming.incoming_rows) {
        hash.add(row.source_rank);
        hash.add(row.expert_id);
        hash.add(row.source_expert_offset);
        hash.add(row.dst_row);
    }
    for (const auto& packet : metadata.transport.outgoing_packets) {
        hash.add(packet.identity.source_rank);
        hash.add(packet.identity.destination_node);
        hash.add(packet.identity.sequence_on_link);
        hash.add(packet.local_token_id);
        hash.add(packet.proxy_rank);
        hash.add(packet.channel);
        for (const auto& target : packet.fanout) {
            hash.add(target.topk_slot);
            hash.add(target.destination_local_rank);
            hash.add(target.expert_id);
            hash.add(target.dst_row);
        }
    }
    const auto add_signals = [&hash](const auto& signals) {
        for (const auto& signal : signals) {
            hash.add(signal.source_node);
            hash.add(signal.destination_node);
            hash.add(signal.source_rank);
            hash.add(signal.destination_proxy_rank);
            hash.add(signal.rail_local_rank);
            hash.add(signal.channel);
            hash.add(signal.expected_packets);
            hash.add(signal.expected_signals);
        }
    };
    add_signals(metadata.transport.outgoing_signals);
    add_signals(metadata.transport.incoming_proxy_signals);
    return hash.value();
}

}  // namespace

class DispatchHandleFactory {
public:
    static DispatchHandle make(std::shared_ptr<const DispatchMetadata> metadata) {
        if (!metadata) {
            fail(NotifyErrorCode::InvalidHandle,
                 "cannot construct DispatchHandle from null metadata");
        }
        return DispatchHandle(
            std::move(metadata),
            std::make_shared<const DispatchHandle::ProvenanceToken>());
    }
};

const DispatchMetadata& DispatchHandle::metadata() const {
    if (!metadata_) {
        fail(NotifyErrorCode::InvalidHandle,
             "attempted to access an empty/default DispatchHandle");
    }
    return *metadata_;
}

LocalNotifyMetadata build_rank_local_histogram(
    const RuntimeConfig& config, std::uint32_t calling_rank,
    const RankRoutingInput& local_routes) {
    validate_config(config);
    if (calling_rank >= config.topology.world_size) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "calling rank out of range: rank=" + std::to_string(calling_rank) +
                 ", world_size=" + std::to_string(config.topology.world_size));
    }
    const auto num_routes = checked_mul(local_routes.num_tokens, config.top_k,
                                        "rank-local route count for rank=" +
                                            std::to_string(calling_rank));
    if (local_routes.selected_experts.size() != checked_size(num_routes, "route array")) {
        fail(NotifyErrorCode::InvalidRouteShape,
             "selected_experts shape mismatch: rank=" + std::to_string(calling_rank) +
                 ", actual=" + std::to_string(local_routes.selected_experts.size()) +
                 ", expected=" + std::to_string(num_routes));
    }

    LocalNotifyMetadata local;
    local.config = config;
    local.calling_rank = calling_rank;
    auto counts = std::make_shared<std::vector<std::uint64_t>>(config.num_experts, 0);
    auto packet_counts =
        std::make_shared<std::vector<std::uint64_t>>(config.topology.num_nodes, 0);
    auto& outgoing = local.outgoing;
    outgoing.num_tokens = local_routes.num_tokens;
    outgoing.top_k = config.top_k;
    outgoing.node_mask_words = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(config.topology.num_nodes) + 63U) / 64U);
    outgoing.expert_id.resize(local_routes.selected_experts.size());
    outgoing.local_offset.resize(local_routes.selected_experts.size());
    outgoing.route_dst_row.resize(local_routes.selected_experts.size());
    outgoing.route_dst_rank.resize(local_routes.selected_experts.size());
    outgoing.route_dst_node.resize(local_routes.selected_experts.size());
    outgoing.route_dst_local_rank.resize(local_routes.selected_experts.size());
    outgoing.combine_order.resize(local_routes.selected_experts.size());
    outgoing.destination_node_mask.assign(
        checked_size(checked_mul(local_routes.num_tokens, outgoing.node_mask_words,
                                 "destination node masks for rank=" +
                                     std::to_string(calling_rank)),
                     "destination node masks"),
        0);

    for (std::uint64_t token = 0; token < local_routes.num_tokens; ++token) {
        std::unordered_set<std::uint32_t> selected;
        std::vector<std::uint32_t> slots(config.top_k);
        std::iota(slots.begin(), slots.end(), 0U);
        std::vector<bool> token_nodes(config.topology.num_nodes, false);
        const auto base = checked_size(checked_mul(token, config.top_k, "route base"),
                                       "route base");
        for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
            const auto flat = base + slot;
            const auto raw_expert = local_routes.selected_experts[flat];
            if (raw_expert < 0 ||
                static_cast<std::uint32_t>(raw_expert) >= config.num_experts) {
                fail(NotifyErrorCode::ExpertOutOfRange,
                     "expert ID out of range at " +
                         route_context(calling_rank, token, slot) + ": actual=" +
                         std::to_string(raw_expert) + ", expected [0," +
                         std::to_string(config.num_experts) + ")");
            }
            const auto expert = static_cast<std::uint32_t>(raw_expert);
            if (!selected.insert(expert).second) {
                fail(NotifyErrorCode::DuplicateExpert,
                     "duplicate expert at " + route_context(calling_rank, token, slot) +
                         ": expert=" + std::to_string(expert));
            }
            const auto destination_rank = config.topology.expert_to_rank[expert];
            const auto destination_node = config.topology.rank_to_node[destination_rank];
            outgoing.expert_id[flat] = expert;
            outgoing.local_offset[flat] = (*counts)[expert];
            (*counts)[expert] = checked_add((*counts)[expert], 1,
                                            "local expert count for rank=" +
                                                std::to_string(calling_rank) +
                                                ", expert=" + std::to_string(expert));
            outgoing.route_dst_rank[flat] = destination_rank;
            outgoing.route_dst_node[flat] = destination_node;
            outgoing.route_dst_local_rank[flat] =
                config.topology.rank_to_local_rank[destination_rank];
            const auto mask_base = checked_size(
                checked_mul(token, outgoing.node_mask_words, "node mask base"),
                "node mask base");
            outgoing.destination_node_mask[mask_base + destination_node / 64U] |=
                std::uint64_t{1} << (destination_node % 64U);
            token_nodes[destination_node] = true;
        }
        std::stable_sort(slots.begin(), slots.end(), [&](auto lhs, auto rhs) {
            return outgoing.expert_id[base + lhs] < outgoing.expert_id[base + rhs];
        });
        for (std::uint32_t position = 0; position < config.top_k; ++position) {
            outgoing.combine_order[base + position] = slots[position];
        }
        for (std::uint32_t node = 0; node < config.topology.num_nodes; ++node) {
            if (token_nodes[node]) {
                (*packet_counts)[node] = checked_add(
                    (*packet_counts)[node], 1,
                    "packet summary for rank=" + std::to_string(calling_rank) +
                        ", destination_node=" + std::to_string(node));
            }
        }
    }
    local.local_expert_count = std::move(counts);
    local.packet_count_by_destination_node = std::move(packet_counts);
    return local;
}

std::unique_ptr<CountExchangeOperation> start_count_exchange(
    const LocalNotifyMetadata& local, CountExchange& backend,
    const CollectiveContext& context) {
    return start_exchange(local, backend, context, ExchangeKind::ExpertCounts,
                          local.local_expert_count, local.config.num_experts);
}

std::unique_ptr<CountExchangeOperation> start_notify_summary_exchange(
    const LocalNotifyMetadata& local, CountExchange& backend,
    const CollectiveContext& context) {
    return start_exchange(local, backend, context, ExchangeKind::NotifyPacketSummary,
                          local.packet_count_by_destination_node,
                          local.config.topology.num_nodes);
}

CountExchangeResult complete_exchange(
    CountExchangeOperation& operation, const CollectiveContext& expected_context,
    ExchangeKind expected_kind, std::uint64_t expected_row_width) {
    try {
        auto result = operation.wait();
        validate_result_shape(result, expected_context, expected_kind,
                              expected_row_width);
        return result;
    } catch (const NotifyDispatchError&) {
        throw;
    } catch (const std::exception& error) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective completion failed: rank=" +
                 std::to_string(expected_context.calling_rank) + ", world_size=" +
                 std::to_string(expected_context.world_size) + ", communicator='" +
                 expected_context.communicator_id + "', generation=" +
                 std::to_string(expected_context.generation) + ", backend_error='" +
                 error.what() + "'");
    }
}

DispatchHandle materialize_rank_metadata(
    const LocalNotifyMetadata& local, const CompletedCollectives& collectives) {
    const auto derived = validate_config(local.config);
    const auto& config = local.config;
    const auto world_size = config.topology.world_size;
    const auto num_experts = config.num_experts;
    const auto rank = local.calling_rank;
    if (rank >= world_size) {
        fail(NotifyErrorCode::InvalidRuntimeConfig,
             "materialization calling rank out of range: rank=" + std::to_string(rank));
    }
    const auto& count_context = collectives.expert_counts.collective;
    const auto& summary_context = collectives.packet_summary.collective;
    validate_result_shape(collectives.expert_counts, count_context,
                          ExchangeKind::ExpertCounts, num_experts);
    validate_result_shape(collectives.packet_summary, summary_context,
                          ExchangeKind::NotifyPacketSummary,
                          config.topology.num_nodes);
    if (count_context.calling_rank != rank || summary_context.calling_rank != rank ||
        count_context.world_size != world_size ||
        summary_context.world_size != world_size ||
        count_context.communicator_id != summary_context.communicator_id ||
        count_context.generation != summary_context.generation ||
        !same_execution(count_context.execution, summary_context.execution)) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "count and notify-summary collectives disagree for calling rank=" +
                 std::to_string(rank));
    }
    CountMatrix counts;
    CountMatrix summaries;
    try {
        counts = collectives.expert_counts.copy_host_rows();
        summaries = collectives.packet_summary.copy_host_rows();
    } catch (const std::exception& error) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "collective output ownership/copy failed for rank=" +
                 std::to_string(rank) + ": " + error.what());
    }
    if (counts[rank] != *local.local_expert_count) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "count collective corrupted calling rank row: rank=" +
                 std::to_string(rank));
    }
    if (summaries[rank] != *local.packet_count_by_destination_node) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "notify summary collective corrupted calling rank row: rank=" +
                 std::to_string(rank));
    }

    CountMatrix source_prefix(static_cast<std::size_t>(world_size) + 1,
                              std::vector<std::uint64_t>(num_experts, 0));
    std::vector<std::uint64_t> expert_total(num_experts, 0);
    for (std::uint32_t source = 0; source < world_size; ++source) {
        for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
            source_prefix[source + 1][expert] = checked_add(
                source_prefix[source][expert], counts[source][expert],
                "source prefix for source_rank=" + std::to_string(source) +
                    ", expert=" + std::to_string(expert));
        }
    }
    expert_total = source_prefix.back();

    std::vector<std::uint64_t> expert_base(num_experts, 0);
    std::vector<std::uint64_t> rank_total_rows(world_size, 0);
    for (std::uint32_t destination_rank = 0; destination_rank < world_size;
         ++destination_rank) {
        std::uint64_t prefix = 0;
        for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
            if (config.topology.expert_to_rank[expert] != destination_rank) continue;
            expert_base[expert] = prefix;
            if (!config.capacity.expert_rows.empty() &&
                expert_total[expert] > config.capacity.expert_rows[expert]) {
                fail(NotifyErrorCode::ExpertCapacityExceeded,
                     "expert capacity exceeded: expert=" + std::to_string(expert) +
                         ", owner_rank=" + std::to_string(destination_rank) +
                         ", actual_rows=" + std::to_string(expert_total[expert]) +
                         ", capacity=" +
                         std::to_string(config.capacity.expert_rows[expert]));
            }
            prefix = checked_add(prefix, expert_total[expert],
                                 "destination expert prefix for rank=" +
                                     std::to_string(destination_rank));
        }
        rank_total_rows[destination_rank] = prefix;
        if (!config.capacity.workspace_rows_per_rank.empty() &&
            prefix > config.capacity.workspace_rows_per_rank[destination_rank]) {
            fail(NotifyErrorCode::WorkspaceCapacityExceeded,
                 "workspace row capacity exceeded: destination_rank=" +
                     std::to_string(destination_rank) + ", actual_rows=" +
                     std::to_string(prefix) + ", capacity=" +
                     std::to_string(
                         config.capacity.workspace_rows_per_rank[destination_rank]));
        }
    }

    auto metadata = std::make_shared<DispatchMetadata>();
    metadata->config = config;
    metadata->calling_rank = rank;
    metadata->communicator_id = count_context.communicator_id;
    metadata->identity.generation = count_context.generation;
    metadata->source_expert_count = counts;
    metadata->outgoing = local.outgoing;
    auto& outgoing = metadata->outgoing;
    outgoing.reverse_routes.resize(outgoing.expert_id.size());
    for (std::uint64_t token = 0; token < outgoing.num_tokens; ++token) {
        const auto base = checked_size(checked_mul(token, outgoing.top_k, "route base"),
                                       "route base");
        for (std::uint32_t slot = 0; slot < outgoing.top_k; ++slot) {
            const auto flat = base + slot;
            const auto expert = outgoing.expert_id[flat];
            const auto destination_rank = outgoing.route_dst_rank[flat];
            const auto within_expert = checked_add(
                source_prefix[rank][expert], outgoing.local_offset[flat],
                "canonical row within expert for " + route_context(rank, token, slot));
            const auto dst_row = checked_add(
                expert_base[expert], within_expert,
                "canonical destination row for " + route_context(rank, token, slot));
            if (dst_row >= rank_total_rows[destination_rank]) {
                fail(NotifyErrorCode::CountExchangeFailure,
                     "canonical row exceeds destination layout for " +
                         route_context(rank, token, slot));
            }
            outgoing.route_dst_row[flat] = dst_row;
            outgoing.reverse_routes[flat] = ReverseRouteMetadata{
                rank, token, slot, expert, destination_rank, dst_row};
        }
    }

    auto& incoming = metadata->incoming;
    incoming.rank = rank;
    incoming.expert_row_prefix.push_back(0);
    for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
        if (config.topology.expert_to_rank[expert] != rank) continue;
        incoming.expert_ids.push_back(expert);
        incoming.expert_row_count.push_back(expert_total[expert]);
        incoming.expert_row_prefix.push_back(checked_add(
            incoming.expert_row_prefix.back(), expert_total[expert],
            "local incoming expert prefix for rank=" + std::to_string(rank)));
        std::vector<std::uint32_t> tiles;
        for (std::uint64_t first = 0; first < expert_total[expert];) {
            const auto remaining = expert_total[expert] - first;
            tiles.push_back(static_cast<std::uint32_t>(
                std::min<std::uint64_t>(remaining, config.m_tile_rows)));
            if (remaining <= config.m_tile_rows) break;
            first = checked_add(first, config.m_tile_rows,
                                "M-tile offset for expert=" +
                                    std::to_string(expert));
        }
        incoming.mtile_expected_rows.push_back(std::move(tiles));
        for (std::uint32_t source = 0; source < world_size; ++source) {
            for (std::uint64_t offset = 0; offset < counts[source][expert]; ++offset) {
                const auto dst_row = checked_add(
                    expert_base[expert],
                    checked_add(source_prefix[source][expert], offset,
                                "incoming source offset"),
                    "incoming canonical row");
                incoming.incoming_rows.push_back(
                    IncomingRowMetadata{source, expert, offset, dst_row});
            }
        }
    }
    std::sort(incoming.incoming_rows.begin(), incoming.incoming_rows.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.dst_row < rhs.dst_row; });

    const auto source_node = config.topology.rank_to_node[rank];
    const auto source_local_rank = config.topology.rank_to_local_rank[rank];
    std::vector<std::uint64_t> sequence_by_node(config.topology.num_nodes, 0);
    for (std::uint32_t destination_node = 0;
         destination_node < config.topology.num_nodes; ++destination_node) {
        const auto proxy_rank = derived.rank_at[
            static_cast<std::size_t>(destination_node) * derived.gpus_per_node +
            source_local_rank];
        for (std::uint64_t token = 0; token < outgoing.num_tokens; ++token) {
            const auto route_base = checked_size(
                checked_mul(token, outgoing.top_k, "packet route base"),
                "packet route base");
            std::vector<FanoutTarget> fanout;
            for (std::uint32_t slot = 0; slot < outgoing.top_k; ++slot) {
                const auto flat = route_base + slot;
                if (outgoing.route_dst_node[flat] != destination_node) continue;
                fanout.push_back(FanoutTarget{slot,
                                              outgoing.route_dst_local_rank[flat],
                                              outgoing.expert_id[flat],
                                              outgoing.route_dst_row[flat]});
            }
            if (fanout.empty()) continue;
            std::stable_sort(fanout.begin(), fanout.end(), [](const auto& lhs,
                                                               const auto& rhs) {
                return lhs.expert_id < rhs.expert_id;
            });
            const auto sequence = sequence_by_node[destination_node]++;
            metadata->transport.outgoing_packets.push_back(OutgoingPacketDescriptor{
                PacketIdentity{rank, destination_node, sequence}, token, proxy_rank, 0,
                std::move(fanout)});
        }
        const auto expected = summaries[rank][destination_node];
        if (sequence_by_node[destination_node] != expected) {
            fail(NotifyErrorCode::CountExchangeFailure,
                 "packet summary disagrees with materialized descriptors: source_rank=" +
                     std::to_string(rank) + ", destination_node=" +
                     std::to_string(destination_node) + ", descriptors=" +
                     std::to_string(sequence_by_node[destination_node]) +
                     ", summary=" + std::to_string(expected));
        }
        metadata->transport.outgoing_signals.push_back(SignalCardinality{
            source_node, destination_node, rank, proxy_rank, source_local_rank, 0,
            expected, expected});
    }

    const auto destination_node = config.topology.rank_to_node[rank];
    const auto rail_local_rank = config.topology.rank_to_local_rank[rank];
    for (std::uint32_t remote_source_node = 0;
         remote_source_node < config.topology.num_nodes; ++remote_source_node) {
        const auto source_rank = derived.rank_at[
            static_cast<std::size_t>(remote_source_node) * derived.gpus_per_node +
            rail_local_rank];
        const auto expected = summaries[source_rank][destination_node];
        metadata->transport.incoming_proxy_signals.push_back(SignalCardinality{
            remote_source_node, destination_node, source_rank, rank, rail_local_rank, 0,
            expected, expected});
    }

    metadata->identity.content_hash = hash_metadata(*metadata);
    return DispatchHandleFactory::make(std::move(metadata));
}

std::vector<DispatchHandle> build_notify_dispatch_for_testing(
    const RuntimeConfig& config,
    const std::vector<RankRoutingInput>& routes_by_rank,
    const std::string& communicator_id, std::uint64_t generation) {
    validate_config(config);
    if (routes_by_rank.size() != config.topology.world_size) {
        fail(NotifyErrorCode::InvalidRouteShape,
             "test wrapper requires one rank-local input per rank: actual=" +
                 std::to_string(routes_by_rank.size()) + ", expected=" +
                 std::to_string(config.topology.world_size));
    }
    std::vector<LocalNotifyMetadata> locals;
    locals.reserve(routes_by_rank.size());
    for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
        locals.push_back(build_rank_local_histogram(config, rank,
                                                     routes_by_rank[rank]));
    }

    HostFakeCountExchange backend(config.topology.world_size, communicator_id);
    std::vector<CollectiveContext> contexts(config.topology.world_size);
    std::vector<std::unique_ptr<CountExchangeOperation>> count_operations;
    for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
        contexts[rank] = CollectiveContext{rank, config.topology.world_size,
                                           communicator_id, generation, {0, 0}};
        count_operations.push_back(
            start_count_exchange(locals[rank], backend, contexts[rank]));
    }
    std::vector<CountExchangeResult> count_results;
    for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
        count_results.push_back(complete_exchange(
            *count_operations[rank], contexts[rank], ExchangeKind::ExpertCounts,
            config.num_experts));
    }

    std::vector<std::unique_ptr<CountExchangeOperation>> summary_operations;
    for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
        summary_operations.push_back(
            start_notify_summary_exchange(locals[rank], backend, contexts[rank]));
    }
    std::vector<DispatchHandle> handles;
    handles.reserve(config.topology.world_size);
    for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
        auto summary = complete_exchange(
            *summary_operations[rank], contexts[rank],
            ExchangeKind::NotifyPacketSummary, config.topology.num_nodes);
        handles.push_back(materialize_rank_metadata(
            locals[rank], CompletedCollectives{count_results[rank], std::move(summary)}));
    }
    return handles;
}

}  // namespace megamoe
