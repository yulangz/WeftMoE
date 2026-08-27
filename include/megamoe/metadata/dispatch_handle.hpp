#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/config/runtime_config.hpp"
#include "megamoe/transport/count_exchange.hpp"

MEGAMOE_PUBLIC_API_BEGIN

namespace megamoe {

class WorkspacePlan;
class DispatchHandleFactory;

struct MetadataIdentity {
    std::uint64_t content_hash = 0;
    std::uint64_t generation = 0;
};

struct ReverseRouteMetadata {
    std::uint32_t source_rank = 0;
    std::uint64_t local_token_id = 0;
    std::uint32_t topk_slot = 0;
    std::uint32_t expert_id = 0;
    std::uint32_t destination_rank = 0;
    std::uint64_t dst_row = 0;
};

struct SourceRouteMetadata {
    std::uint64_t num_tokens = 0;
    std::uint32_t top_k = 0;
    std::uint32_t node_mask_words = 0;

    // Row-major [num_tokens, top_k], for this calling rank only.
    std::vector<std::uint32_t> expert_id;
    std::vector<std::uint64_t> local_offset;
    std::vector<std::uint64_t> route_dst_row;
    std::vector<std::uint32_t> route_dst_rank;
    std::vector<std::uint32_t> route_dst_node;
    std::vector<std::uint32_t> route_dst_local_rank;
    std::vector<std::uint32_t> combine_order;
    std::vector<std::uint64_t> destination_node_mask;
    std::vector<ReverseRouteMetadata> reverse_routes;
};

struct IncomingRowMetadata {
    std::uint32_t source_rank = 0;
    std::uint32_t expert_id = 0;
    std::uint64_t source_expert_offset = 0;
    std::uint64_t dst_row = 0;
};

struct DestinationMetadata {
    std::uint32_t rank = 0;
    std::vector<std::uint32_t> expert_ids;
    std::vector<std::uint64_t> expert_row_count;
    std::vector<std::uint64_t> expert_row_prefix;
    std::vector<std::vector<std::uint32_t>> mtile_expected_rows;

    // Geometry derived only from integer counts. Remote token/slot identity is
    // carried by packet fan-out descriptors, not copied into this handle.
    std::vector<IncomingRowMetadata> incoming_rows;
};

struct PacketIdentity {
    std::uint32_t source_rank = 0;
    std::uint32_t destination_node = 0;
    std::uint64_t sequence_on_link = 0;
};

struct FanoutTarget {
    std::uint32_t topk_slot = 0;
    std::uint32_t destination_local_rank = 0;
    std::uint32_t expert_id = 0;
    std::uint64_t dst_row = 0;
};

struct OutgoingPacketDescriptor {
    PacketIdentity identity;
    std::uint64_t local_token_id = 0;
    std::uint32_t proxy_rank = 0;
    std::uint32_t channel = 0;
    std::vector<FanoutTarget> fanout;
};

inline constexpr std::uint64_t kDispatchPacketHeaderDescriptorBytes = 32;
inline constexpr std::uint64_t kDispatchFanoutDescriptorBytes = 24;

struct SignalCardinality {
    std::uint32_t source_node = 0;
    std::uint32_t destination_node = 0;
    std::uint32_t source_rank = 0;
    std::uint32_t destination_proxy_rank = 0;
    std::uint32_t rail_local_rank = 0;
    std::uint32_t channel = 0;
    std::uint64_t expected_packets = 0;
    std::uint64_t expected_signals = 0;
};

struct TransportMetadata {
    // Packet identity/storage order remains (destination_node,
    // local_token_id).  dispatch_wave_packet_order is a permutation over that
    // immutable storage, ordered by
    // (destination_node, primary_mtile, expert_id, destination_rank,
    // local_token_id).  This lets a transport advance canonical M tiles while
    // preserving one payload per token-node.
    std::vector<OutgoingPacketDescriptor> outgoing_packets;
    std::vector<std::uint64_t> dispatch_wave_packet_order;
    std::vector<std::uint64_t> dispatch_wave_offsets;
    std::vector<std::uint32_t> dispatch_wave_destination_node;
    std::vector<std::uint32_t> dispatch_wave_channel;
    std::vector<std::uint64_t> dispatch_wave_first_primary_tile;
    std::vector<std::uint64_t> dispatch_wave_last_primary_tile;
    std::vector<std::uint64_t> dispatch_wave_packet_count;
    // Descriptor bytes only. Activation bytes are derived after transport
    // binds the runtime hidden size and dtype.
    std::vector<std::uint64_t> dispatch_wave_payload_bytes;
    std::vector<SignalCardinality> outgoing_signals;
    std::vector<SignalCardinality> incoming_proxy_signals;
};

struct DispatchMetadata {
    RuntimeConfig config;
    std::uint32_t calling_rank = 0;
    std::string communicator_id;
    MetadataIdentity identity;

    // The only global payload retained: integer [source_rank, expert] counts.
    CountMatrix source_expert_count;
    SourceRouteMetadata outgoing;
    DestinationMetadata incoming;
    TransportMetadata transport;
};

class DispatchHandle {
public:
    DispatchHandle() = default;

    const DispatchMetadata& metadata() const;
    const DispatchMetadata* operator->() const { return &metadata(); }
    explicit operator bool() const noexcept { return static_cast<bool>(metadata_); }

private:
    struct ProvenanceToken {};
    DispatchHandle(std::shared_ptr<const DispatchMetadata> metadata,
                   std::shared_ptr<const ProvenanceToken> provenance)
        : metadata_(std::move(metadata)), provenance_(std::move(provenance)) {}

    std::shared_ptr<const DispatchMetadata> metadata_;
    std::shared_ptr<const ProvenanceToken> provenance_;

    friend class DispatchHandleFactory;
    friend class WorkspacePlan;
    friend WorkspacePlan make_workspace_plan(const DispatchHandle&);
    friend void validate_workspace_plan(const DispatchHandle&, const WorkspacePlan&);
};

void validate_dispatch_metadata(const DispatchHandle& handle);

}  // namespace megamoe

MEGAMOE_PUBLIC_API_END
