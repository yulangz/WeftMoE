#include "weftmoe/metadata/serialize.hpp"

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace weftmoe {
namespace {

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const auto raw : value) {
        const auto ch = static_cast<unsigned char>(raw);
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    out << "\\u00" << digits[ch >> 4U] << digits[ch & 0x0fU];
                } else {
                    out << raw;
                }
        }
    }
    out << '"';
    return out.str();
}

template <typename T>
void write_vector(std::ostringstream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        if constexpr (std::is_same_v<T, std::string>) {
            out << json_string(values[i]);
        } else {
            out << values[i];
        }
    }
    out << ']';
}

void write_matrix(std::ostringstream& out, const CountMatrix& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        write_vector(out, values[i]);
    }
    out << ']';
}

template <typename F>
void write_objects(std::ostringstream& out, std::size_t size, F&& writer) {
    out << '[';
    for (std::size_t i = 0; i < size; ++i) {
        if (i != 0) out << ',';
        writer(i);
    }
    out << ']';
}

void write_signal(std::ostringstream& out, const SignalCardinality& signal) {
    out << "{\"source_node\":" << signal.source_node
        << ",\"destination_node\":" << signal.destination_node
        << ",\"source_rank\":" << signal.source_rank
        << ",\"destination_proxy_rank\":" << signal.destination_proxy_rank
        << ",\"rail_local_rank\":" << signal.rail_local_rank
        << ",\"channel\":" << signal.channel
        << ",\"expected_packets\":" << signal.expected_packets
        << ",\"expected_signals\":" << signal.expected_signals << '}';
}

}  // namespace

std::string serialize_dispatch_metadata(const DispatchHandle& handle) {
    const auto& metadata = handle.metadata();
    const auto& config = metadata.config;
    const auto& outgoing = metadata.outgoing;
    const auto& incoming = metadata.incoming;
    std::ostringstream out;
    out << "{\"schema\":1,\"config\":{\"num_experts\":" << config.num_experts
        << ",\"top_k\":" << config.top_k << ",\"m_tile_rows\":"
        << config.m_tile_rows << ",\"world_size\":"
        << config.topology.world_size << ",\"num_nodes\":"
        << config.topology.num_nodes << ",\"rank_to_node\":";
    write_vector(out, config.topology.rank_to_node);
    out << ",\"rank_to_local_rank\":";
    write_vector(out, config.topology.rank_to_local_rank);
    out << ",\"expert_to_rank\":";
    write_vector(out, config.topology.expert_to_rank);
    out << ",\"expert_capacity\":";
    write_vector(out, config.capacity.expert_rows);
    out << ",\"workspace_capacity\":";
    write_vector(out, config.capacity.workspace_rows_per_rank);
    out << "},\"calling_rank\":" << metadata.calling_rank
        << ",\"communicator_id\":" << json_string(metadata.communicator_id)
        << ",\"identity\":{\"content_hash\":" << metadata.identity.content_hash
        << ",\"generation\":" << metadata.identity.generation
        << "},\"source_expert_count\":";
    write_matrix(out, metadata.source_expert_count);

    out << ",\"outgoing\":{\"num_tokens\":" << outgoing.num_tokens
        << ",\"top_k\":" << outgoing.top_k << ",\"node_mask_words\":"
        << outgoing.node_mask_words << ",\"expert_id\":";
    write_vector(out, outgoing.expert_id);
    out << ",\"local_offset\":";
    write_vector(out, outgoing.local_offset);
    out << ",\"route_dst_row\":";
    write_vector(out, outgoing.route_dst_row);
    out << ",\"route_dst_rank\":";
    write_vector(out, outgoing.route_dst_rank);
    out << ",\"route_dst_node\":";
    write_vector(out, outgoing.route_dst_node);
    out << ",\"route_dst_local_rank\":";
    write_vector(out, outgoing.route_dst_local_rank);
    out << ",\"combine_order\":";
    write_vector(out, outgoing.combine_order);
    out << ",\"destination_node_mask\":";
    write_vector(out, outgoing.destination_node_mask);
    out << ",\"reverse_routes\":";
    write_objects(out, outgoing.reverse_routes.size(), [&](std::size_t i) {
        const auto& route = outgoing.reverse_routes[i];
        out << "{\"source_rank\":" << route.source_rank
            << ",\"local_token_id\":" << route.local_token_id
            << ",\"topk_slot\":" << route.topk_slot
            << ",\"expert_id\":" << route.expert_id
            << ",\"destination_rank\":" << route.destination_rank
            << ",\"dst_row\":" << route.dst_row << '}';
    });
    out << "},\"incoming\":{\"rank\":" << incoming.rank
        << ",\"expert_ids\":";
    write_vector(out, incoming.expert_ids);
    out << ",\"expert_row_count\":";
    write_vector(out, incoming.expert_row_count);
    out << ",\"expert_row_prefix\":";
    write_vector(out, incoming.expert_row_prefix);
    out << ",\"mtile_expected_rows\":";
    write_objects(out, incoming.mtile_expected_rows.size(), [&](std::size_t i) {
        write_vector(out, incoming.mtile_expected_rows[i]);
    });
    out << ",\"incoming_rows\":";
    write_objects(out, incoming.incoming_rows.size(), [&](std::size_t i) {
        const auto& row = incoming.incoming_rows[i];
        out << "{\"source_rank\":" << row.source_rank
            << ",\"expert_id\":" << row.expert_id
            << ",\"source_expert_offset\":" << row.source_expert_offset
            << ",\"dst_row\":" << row.dst_row << '}';
    });
    out << "},\"transport\":{\"outgoing_packets\":";
    write_objects(out, metadata.transport.outgoing_packets.size(), [&](std::size_t i) {
        const auto& packet = metadata.transport.outgoing_packets[i];
        out << "{\"identity\":{\"source_rank\":" << packet.identity.source_rank
            << ",\"destination_node\":" << packet.identity.destination_node
            << ",\"sequence_on_link\":" << packet.identity.sequence_on_link
            << "},\"local_token_id\":" << packet.local_token_id
            << ",\"proxy_rank\":" << packet.proxy_rank
            << ",\"channel\":" << packet.channel << ",\"fanout\":";
        write_objects(out, packet.fanout.size(), [&](std::size_t j) {
            const auto& target = packet.fanout[j];
            out << "{\"topk_slot\":" << target.topk_slot
                << ",\"destination_local_rank\":"
                << target.destination_local_rank << ",\"expert_id\":"
                << target.expert_id << ",\"dst_row\":" << target.dst_row << '}';
        });
        out << '}';
    });
    out << ",\"dispatch_wave_packet_order\":";
    write_vector(out, metadata.transport.dispatch_wave_packet_order);
    out << ",\"dispatch_wave_offsets\":";
    write_vector(out, metadata.transport.dispatch_wave_offsets);
    out << ",\"dispatch_wave_destination_node\":";
    write_vector(out, metadata.transport.dispatch_wave_destination_node);
    out << ",\"dispatch_wave_channel\":";
    write_vector(out, metadata.transport.dispatch_wave_channel);
    out << ",\"dispatch_wave_first_primary_tile\":";
    write_vector(out, metadata.transport.dispatch_wave_first_primary_tile);
    out << ",\"dispatch_wave_last_primary_tile\":";
    write_vector(out, metadata.transport.dispatch_wave_last_primary_tile);
    out << ",\"dispatch_wave_packet_count\":";
    write_vector(out, metadata.transport.dispatch_wave_packet_count);
    out << ",\"dispatch_wave_payload_bytes\":";
    write_vector(out, metadata.transport.dispatch_wave_payload_bytes);
    out << ",\"outgoing_signals\":";
    write_objects(out, metadata.transport.outgoing_signals.size(),
                  [&](std::size_t i) {
                      write_signal(out, metadata.transport.outgoing_signals[i]);
                  });
    out << ",\"incoming_proxy_signals\":";
    write_objects(out, metadata.transport.incoming_proxy_signals.size(),
                  [&](std::size_t i) {
                      write_signal(out, metadata.transport.incoming_proxy_signals[i]);
                  });
    out << "}}";
    return out.str();
}

}  // namespace weftmoe
