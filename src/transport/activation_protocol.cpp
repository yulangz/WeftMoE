#define MEGAMOE_INTERNAL_TESTING_API 1
#include "megamoe/transport/activation_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "megamoe/api/errors.hpp"

namespace megamoe::transport {
namespace {

[[noreturn]] void protocol_error(const std::string &field,
                                 std::uint32_t rank,
                                 const std::string &expected,
                                 const std::string &actual) {
  throw NotifyDispatchError(
      NotifyErrorCode::TransportProtocolFailure,
      "phase=activation-transport, field=" + field + ", rank=" +
          std::to_string(rank) + ", expected=" + expected + ", actual=" +
          actual);
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b,
                          const char *field, std::uint32_t rank) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    protocol_error(field, rank, "checked uint64 multiplication", "overflow");
  return a * b;
}

std::size_t checked_size(std::uint64_t value, const char *field,
                         std::uint32_t rank) {
  if (value > std::numeric_limits<std::size_t>::max())
    protocol_error(field, rank, "value <= size_t max", std::to_string(value));
  return static_cast<std::size_t>(value);
}

BFloat16 synthetic_value(BFloat16 input, std::uint32_t expert,
                          std::uint64_t hidden_index,
                          ActivationExpertTransform transform) noexcept {
  if (transform == ActivationExpertTransform::Identity)
    return input;
  const auto expert_term = static_cast<float>((expert % 251U) + 1U) / 128.0F;
  const auto lane_term = static_cast<float>(hidden_index % 17U) / 1024.0F;
  return float_to_bfloat16(bfloat16_to_float(input) + expert_term + lane_term);
}

} // namespace

float bfloat16_to_float(BFloat16 value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value.bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

BFloat16 float_to_bfloat16(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const auto lsb = (bits >> 16U) & 1U;
  bits += 0x7fffU + lsb;
  return BFloat16{static_cast<std::uint16_t>(bits >> 16U)};
}

const char *activation_transport_backend_name(
    ActivationTransportBackend backend) noexcept {
  switch (backend) {
  case ActivationTransportBackend::DeviceSimulation:
    return "device-simulation";
  case ActivationTransportBackend::LocalCudaP2p:
    return "local-cuda-p2p";
  case ActivationTransportBackend::NcclSocketFallback:
    return "nccl-socket-host-enqueued-fallback";
  case ActivationTransportBackend::Gin:
    return "nccl-gin-device";
  }
  return "unknown";
}

SyntheticTransportResult simulate_activation_transport(
    const DispatchHandle &dispatch,
    const std::vector<BFloat16> &token_activations,
    const std::vector<float> &route_gates,
    const ActivationTransportConfig &config, std::uint64_t epoch,
    const std::vector<std::uint64_t> &arrival_order) {
  validate_dispatch_metadata(dispatch);
  const auto &metadata = dispatch.metadata();
  const auto rank = metadata.calling_rank;
  if (config.hidden_size == 0)
    protocol_error("hidden_size", rank, "> 0", "0");
  if (config.channels == 0)
    protocol_error("channels", rank, "> 0", "0");
  if (config.gate_placement != ActivationGatePlacement::SourceAfter)
    protocol_error("gate_placement", rank, "SourceAfter", "unsupported");
  if (config.ring_slots_per_source_channel == 0)
    protocol_error("ring_slots_per_source_channel", rank, "> 0", "0");
  if (epoch == 0)
    protocol_error("epoch", rank, "> 0", "0");
  const auto activation_count = checked_mul(metadata.outgoing.num_tokens,
                                            config.hidden_size,
                                            "activation_count", rank);
  if (token_activations.size() != checked_size(activation_count,
                                               "activation_count", rank))
    protocol_error("token_activations.size", rank,
                   std::to_string(activation_count),
                   std::to_string(token_activations.size()));

  const auto &packets = metadata.transport.outgoing_packets;
  std::vector<std::uint64_t> order;
  if (arrival_order.empty()) {
    order.resize(packets.size());
    std::iota(order.begin(), order.end(), std::uint64_t{0});
  } else {
    order = arrival_order;
  }
  if (order.size() != packets.size())
    protocol_error("arrival_order.size", rank, std::to_string(packets.size()),
                   std::to_string(order.size()));
  std::vector<std::uint8_t> seen(packets.size(), 0);
  for (const auto packet_index : order) {
    if (packet_index >= packets.size())
      protocol_error("arrival_order.packet", rank,
                     "packet index < packet_count",
                     std::to_string(packet_index));
    if (seen[checked_size(packet_index, "packet index", rank)] != 0)
      protocol_error("arrival_order.permutation", rank, "unique packet",
                     std::to_string(packet_index));
    seen[checked_size(packet_index, "packet index", rank)] = 1;
  }

  std::uint64_t destination_rows = 0;
  for (const auto &route : metadata.outgoing.reverse_routes)
    destination_rows = std::max(destination_rows, route.dst_row + 1U);
  const auto route_count = checked_mul(metadata.outgoing.num_tokens,
                                       metadata.outgoing.top_k,
                                       "route_count", rank);
  if (route_gates.size() != checked_size(route_count, "route_count", rank))
    protocol_error("route_gates.size", rank, std::to_string(route_count),
                   std::to_string(route_gates.size()));
  for (std::size_t route = 0; route < route_gates.size(); ++route)
    if (!std::isfinite(route_gates[route]) || route_gates[route] < 0.0F)
      protocol_error("route_gates.value", rank, "finite>=0",
                     "invalid route=" + std::to_string(route));
  SyntheticTransportResult result;
  result.contribution_by_route.resize(checked_size(
      checked_mul(route_count, config.hidden_size, "contribution_count", rank),
      "contribution_count", rank));
  result.combined_tokens.resize(checked_size(activation_count,
                                             "combined_count", rank));
  result.canonical_expert_rows.resize(checked_size(
      checked_mul(checked_mul(metadata.config.topology.world_size,
                              destination_rows, "destination row arena", rank),
                  config.hidden_size, "expert_row_count", rank),
      "expert_row_count", rank));

  const auto source_node =
      metadata.config.topology.rank_to_node.at(metadata.calling_rank);
  std::vector<std::uint64_t> link_credits(
      checked_size(checked_mul(metadata.config.topology.num_nodes,
                               config.channels, "link_count", rank),
                   "link_count", rank),
      config.ring_slots_per_source_channel);
  // Validate the descriptor protocol independently of simulated arrival.
  // Every link must contain exactly the monotonic sequence 0..N-1. An
  // arrival permutation may delay a sequence, but cannot redefine or skip it.
  std::vector<std::vector<std::uint64_t>> sequence_by_link(link_credits.size());
  for (const auto &packet : packets) {
    if (packet.identity.destination_node >= metadata.config.topology.num_nodes ||
        packet.channel >= config.channels)
      protocol_error("packet_descriptor", rank, "valid node/channel",
                     "out of range");
    const auto link = checked_size(
        checked_mul(packet.identity.destination_node, config.channels,
                    "sequence link", rank) + packet.channel,
        "sequence link", rank);
    sequence_by_link[link].push_back(packet.identity.sequence_on_link);
  }
  for (std::size_t link = 0; link < sequence_by_link.size(); ++link) {
    auto sequences = sequence_by_link[link];
    std::sort(sequences.begin(), sequences.end());
    for (std::size_t index = 0; index < sequences.size(); ++index) {
      if (sequences[index] != index)
        protocol_error("packet.sequence", rank, std::to_string(index),
                       std::to_string(sequences[index]));
    }
  }
  std::vector<std::uint8_t> route_ready(
      checked_size(route_count, "route_count", rank), 0);

  for (const auto packet_index : order) {
    const auto &packet = packets[checked_size(packet_index, "packet", rank)];
    if (packet.identity.source_rank != rank ||
        packet.identity.destination_node >=
            metadata.config.topology.num_nodes ||
        packet.channel >= config.channels ||
        packet.local_token_id >= metadata.outgoing.num_tokens)
      protocol_error("packet_descriptor", rank,
                     "valid source/node/channel/token", "mismatch");
    const auto link = checked_size(
        checked_mul(packet.identity.destination_node, config.channels,
                    "link", rank) +
            packet.channel,
        "link", rank);
    if (link_credits[link] == 0) {
      ++result.audit.credit_stalls;
      // The simulator relays one packet immediately, then returns its credit.
      // This models bounded progress without using arrival order for rows.
      link_credits[link] = 1;
    }
    --link_credits[link];
    ++result.audit.dispatch_packets;
    result.audit.peak_ring_slots = std::max<std::uint64_t>(
        result.audit.peak_ring_slots,
        config.ring_slots_per_source_channel - link_credits[link]);
    if (packet.identity.destination_node != source_node) {
      result.audit.deduplicated_cross_node_payload_bytes +=
          checked_mul(config.hidden_size, sizeof(BFloat16), "payload bytes",
                      rank);
      result.audit.raw_cross_node_payload_bytes += checked_mul(
          checked_mul(packet.fanout.size(), config.hidden_size,
                      "raw payload elements", rank),
          sizeof(BFloat16), "raw payload bytes", rank);
    }
    std::uint32_t previous_expert = 0;
    bool first = true;
    for (const auto &target : packet.fanout) {
      if ((!first && target.expert_id <= previous_expert) ||
          target.topk_slot >= metadata.outgoing.top_k)
        protocol_error("packet.fanout_order", rank,
                       "strict ascending expert and valid slot", "invalid");
      first = false;
      previous_expert = target.expert_id;
      const auto route = checked_mul(packet.local_token_id,
                                     metadata.outgoing.top_k, "route", rank) +
                         target.topk_slot;
      if (route >= route_count || route_ready[checked_size(route, "route", rank)] != 0)
        protocol_error("route.arrival", rank, "one arrival per route",
                       std::to_string(route));
      route_ready[checked_size(route, "route", rank)] = 1;
      for (std::uint64_t h = 0; h < config.hidden_size; ++h) {
        const auto input_index = checked_size(
            checked_mul(packet.local_token_id, config.hidden_size,
                        "input index", rank) + h,
            "input index", rank);
        const auto destination_rank =
            metadata.outgoing.route_dst_rank[checked_size(route, "route", rank)];
        const auto global_row = checked_mul(destination_rank, destination_rows,
                                            "global row", rank) +
                                target.dst_row;
        const auto row_index = checked_size(
            checked_mul(global_row, config.hidden_size, "row index", rank) + h,
            "row index", rank);
        if (row_index >= result.canonical_expert_rows.size())
          protocol_error("fanout.dst_row", rank,
                         "row within canonical arena",
                         std::to_string(target.dst_row));
        result.canonical_expert_rows[row_index] = token_activations[input_index];
        result.contribution_by_route[checked_size(
            checked_mul(route, config.hidden_size, "contribution index", rank) +
                h,
            "contribution index", rank)] =
            synthetic_value(token_activations[input_index], target.expert_id, h,
                            config.expert_transform);
      }
      ++result.audit.relayed_fanout;
      ++result.audit.synthetic_contributions;
      ++result.audit.combine_packets;
    }
    ++link_credits[link];
  }
  if (std::find(route_ready.begin(), route_ready.end(), std::uint8_t{0}) !=
      route_ready.end())
    protocol_error("route.completion", rank, "all routes ready", "missing");

  for (std::uint64_t token = 0; token < metadata.outgoing.num_tokens; ++token) {
    for (std::uint64_t h = 0; h < config.hidden_size; ++h) {
      float sum = 0.0F;
      for (std::uint32_t order_slot = 0; order_slot < metadata.outgoing.top_k;
           ++order_slot) {
        const auto combine_index = checked_size(
            checked_mul(token, metadata.outgoing.top_k, "combine index", rank) +
                order_slot,
            "combine index", rank);
        const auto original_slot = metadata.outgoing.combine_order.at(combine_index);
        const auto route = checked_mul(token, metadata.outgoing.top_k,
                                       "combine route", rank) + original_slot;
        const auto contribution_index = checked_size(
            checked_mul(route, config.hidden_size, "reduce index", rank) + h,
            "reduce index", rank);
        volatile float product =
            route_gates[checked_size(route, "gate route", rank)] *
            bfloat16_to_float(
                result.contribution_by_route[contribution_index]);
        sum += product;
      }
      result.combined_tokens[checked_size(
          checked_mul(token, config.hidden_size, "combined index", rank) + h,
          "combined index", rank)] = float_to_bfloat16(sum);
    }
    ++result.audit.reduced_tokens;
  }
  return result;
}

} // namespace megamoe::transport
