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

// One statically strided warp owns each packet. Matching destination lanes
// combine their route counts before writing its private consumer directory.
// No atomic counter or runtime fanout-fragment reference graph is needed.
__device__ inline void build_dispatch_packets(const SourcePlanView &a,
                                              DispatchPlanView out,
                                              std::uint64_t worker,
                                              std::uint64_t workers) {
  const auto channels = a.nodes * a.packets.lanes;
  const auto local_world = a.shape.world / a.nodes;
  const auto owned = a.experts / a.shape.world;
  const auto lane = static_cast<unsigned>(worker % 32U);
  for (auto packet = worker / 32U; packet < out.channel_offsets[channels];
       packet += workers / 32U) {
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
    for (unsigned local = lane; local < local_world; local += 32U)
      consumers[local] = 0U;
    __syncwarp();
    unsigned fanouts = 0U;
    for (unsigned base = 0U; base < rows * a.shape.top_k; base += 32U) {
      const auto index = base + lane;
      unsigned destination = ~0U;
      if (index < rows * a.shape.top_k) {
        const auto token = a.node_tokens[begin + index / a.shape.top_k].token;
        const auto route = token * a.shape.top_k + index % a.shape.top_k;
        const auto expert = a.selected_experts[route];
        if ((!a.valid || a.valid[route]) && expert >= 0 &&
            static_cast<unsigned>(expert) < a.experts) {
          const auto rank = static_cast<unsigned>(expert) / owned;
          if (rank / local_world == node)
            destination = rank % local_world;
        }
      }
      const auto matching = __match_any_sync(0xffffffffU, destination);
      if (destination != ~0U) {
        ++fanouts;
        if (lane == static_cast<unsigned>(__ffs(matching) - 1))
          consumers[destination] += __popc(matching);
      }
      // A later batch may elect a different lane for the same destination.
      __syncwarp();
    }
    fanouts = __reduce_add_sync(0xffffffffU, fanouts);
    if (lane == 0U)
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
  build_packet_prefix(a, out.partition, a.expert_prefix, a.experts + 1U,
                       a.experts / a.nodes, out.packet_prefix, worker, workers);
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
            out.partition.token_begin(location.lane, location.sequence),
        out.partition.token_count(routes, location.lane, location.sequence)};
  }
}

} // namespace weftmoe::cuda::static_forward
