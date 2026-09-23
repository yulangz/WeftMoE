#pragma once

#include "cuda/static_v4/source_plan.cuh"

namespace weftmoe::cuda::static_forward {

// Immutable metadata, not an RX header. Non-consumers inspect only this plan;
// they never touch reusable packet storage. A consumer reads its count before
// waiting for ready, then consumes that many complete fanout rows before ACK.
// One fixed owner after all channel prefixes have been built. No payload or
// slot state is published by these planning passes.
__device__ inline void build_channel_offsets(const std::uint64_t *prefix,
                                             unsigned channels,
                                             std::uint64_t windows,
                                             std::uint64_t *offsets) {
  std::uint64_t total = 0U;
  offsets[0] = 0U;
  for (unsigned channel = 0; channel < channels; ++channel) {
    total += prefix[channel * (windows + 1U) + windows];
    offsets[channel + 1U] = total;
  }
}

// Find the unique nonempty interval containing value. Repeated boundaries
// represent empty windows/channels and are intentionally skipped.
__device__ inline std::uint64_t containing_interval(const std::uint64_t *prefix,
                                                    std::uint64_t count,
                                                    std::uint64_t value) {
  std::uint64_t first = 0U, last = count;
  while (first < last) {
    const auto middle = first + (last - first) / 2U;
    if (prefix[middle + 1U] <= value)
      first = middle + 1U;
    else
      last = middle;
  }
  return first;
}

struct PacketLocation {
  unsigned node;
  unsigned lane;
  std::uint64_t window;
  std::uint64_t sequence; // Window-local sequence on this lane.
};

__device__ inline PacketLocation locate_packet(
    const SourcePlanView &a, const std::uint64_t *offsets,
    const std::uint64_t *prefixes, std::uint64_t packet) {
  const auto channel =
      containing_interval(offsets, a.nodes * a.packets.lanes, packet);
  const auto sequence = packet - offsets[channel];
  const auto *prefix = prefixes + channel * (a.local_windows() + 1U);
  const auto window = containing_interval(prefix, a.local_windows(), sequence);
  return {static_cast<unsigned>(channel / a.packets.lanes),
          static_cast<unsigned>(channel % a.packets.lanes), window,
          sequence - prefix[window]};
}

// A statically strided thread owns each packet directory entry. Q is small;
// the owner computes exact per-rank fanouts without an atomic counter or a
// fanout-fragment reference graph. Future packing reuses this token ordering.
__device__ inline void build_dispatch_packets(const SourcePlanView &a,
                                              DispatchPlanView out,
                                              std::uint64_t worker,
                                              std::uint64_t workers) {
  const auto channels = a.nodes * a.packets.lanes;
  const auto local_world = a.shape.world / a.nodes;
  const auto owned = a.experts / a.shape.world;
  for (auto packet = worker; packet < out.channel_offsets[channels];
       packet += workers) {
    const auto location =
        locate_packet(a, out.channel_offsets, a.packet_prefix, packet);
    const auto node = location.node;
    const auto window = location.window;
    const auto node_begin = a.node_prefix[window * (a.nodes + 1U) + node];
    const auto tokens =
        a.node_prefix[window * (a.nodes + 1U) + node + 1U] - node_begin;
    const auto begin = window * a.window_capacity() + node_begin +
                       a.packets.token_begin(location.lane, location.sequence);
    const auto rows = a.packets.token_count(tokens, location.lane,
                                           location.sequence);
    auto *consumers = out.consumer_rows + packet * local_world;
    for (unsigned local = 0; local < local_world; ++local)
      consumers[local] = 0U;
    unsigned fanouts = 0U;
    for (unsigned row = 0; row < rows; ++row) {
      const auto token = a.node_tokens[begin + row].token;
      const auto route_begin = token * a.shape.top_k;
      for (unsigned k = 0; k < a.shape.top_k; ++k) {
        if (a.valid && !a.valid[route_begin + k])
          continue;
        const auto expert = a.selected_experts[route_begin + k];
        if (expert < 0 || static_cast<unsigned>(expert) >= a.experts)
          continue; // The source planning error rejects the invocation.
        const auto rank = static_cast<unsigned>(expert) / owned;
        if (rank / local_world == node) {
          ++consumers[rank % local_world];
          ++fanouts;
          out.route_packets[route_begin + k] = packet * a.packets.rows + row;
        }
      }
    }
    out.packets[packet] = {
        a.shape.window(a.source_rank, window * a.shape.tokens), begin, rows,
        fanouts};
  }
}

// Return follows the already-frozen node/expert route intervals. A complete
// route is one contribution and one retirement unit, including duplicate
// expert slots. No C-buffer address or runtime fragment count is needed here.
__device__ inline void build_return_packet_prefix(const SourcePlanView &a,
                                                  ReturnPlanView out,
                                                  unsigned worker,
                                                  unsigned workers) {
  const auto windows = a.local_windows();
  const auto experts_per_node = a.experts / a.nodes;
  for (auto channel = worker; channel < a.nodes * a.packets.lanes;
       channel += workers) {
    const auto node = channel / a.packets.lanes;
    const auto lane = channel % a.packets.lanes;
    auto *prefix = out.packet_prefix + channel * (windows + 1U);
    std::uint64_t total = 0U;
    prefix[0] = 0U;
    for (std::uint64_t w = 0; w < windows; ++w) {
      const auto *experts = a.expert_prefix + w * (a.experts + 1U);
      if (node != a.source_rank / (a.shape.world / a.nodes))
        total += a.packets.lane_packets(
            experts[(node + 1U) * experts_per_node] -
                experts[node * experts_per_node], lane);
      prefix[w + 1U] = total;
    }
  }
}

__device__ inline void build_return_packets(const SourcePlanView &a,
                                            ReturnPlanView out,
                                            std::uint64_t worker,
                                            std::uint64_t workers) {
  const auto channels = a.nodes * a.packets.lanes;
  const auto experts_per_node = a.experts / a.nodes;
  for (auto packet = worker; packet < out.channel_offsets[channels];
       packet += workers) {
    const auto location =
        locate_packet(a, out.channel_offsets, out.packet_prefix, packet);
    const auto *experts =
        a.expert_prefix + location.window * (a.experts + 1U);
    const auto first = experts[location.node * experts_per_node];
    const auto routes = experts[(location.node + 1U) * experts_per_node] - first;
    out.packets[packet] = {
        a.shape.window(a.source_rank, location.window * a.shape.tokens),
        location.window * a.window_capacity() + first +
            a.packets.token_begin(location.lane, location.sequence),
        a.packets.token_count(routes, location.lane, location.sequence)};
  }
}

} // namespace weftmoe::cuda::static_forward
