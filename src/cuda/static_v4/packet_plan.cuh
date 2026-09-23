#pragma once

#include "cuda/static_v4/source_plan.cuh"

namespace weftmoe::cuda::static_forward {

// Immutable metadata, not an RX header. Non-consumers inspect only this plan;
// they never touch reusable packet storage. A consumer reads its count before
// waiting for ready, then consumes that many complete fanout rows before ACK.
struct DispatchPacketPlan {
  std::uint64_t window;
  std::uint64_t token_begin; // Index into the source's frozen NodeToken array.
  std::uint32_t tokens;
  std::uint32_t fanouts;
};

struct DispatchPlanView {
  std::uint64_t *channel_offsets; // [nodes * lanes + 1], self channels empty.
  DispatchPacketPlan *packets;    // [sum(channel terminal sequences)]
  std::uint32_t *consumer_rows;   // [packets, local_world]
};

// One fixed owner after all channel prefixes have been built. No payload or
// slot state is published by these planning passes.
__device__ inline void build_dispatch_channel_offsets(const SourcePlanView &a,
                                                      DispatchPlanView out) {
  const auto windows = a.local_windows();
  std::uint64_t total = 0U;
  out.channel_offsets[0] = 0U;
  for (unsigned channel = 0; channel < a.nodes * a.packets.lanes; ++channel) {
    total += a.packet_prefix[channel * (windows + 1U) + windows];
    out.channel_offsets[channel + 1U] = total;
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

// A statically strided thread owns each packet directory entry. Q is small;
// the owner computes exact per-rank fanouts without an atomic counter or a
// fanout-fragment reference graph. Future packing reuses this token ordering.
__device__ inline void build_dispatch_packets(const SourcePlanView &a,
                                              DispatchPlanView out,
                                              std::uint64_t worker,
                                              std::uint64_t workers) {
  const auto channels = a.nodes * a.packets.lanes;
  const auto windows = a.local_windows();
  const auto local_world = a.shape.world / a.nodes;
  const auto owned = a.experts / a.shape.world;
  for (auto packet = worker; packet < out.channel_offsets[channels];
       packet += workers) {
    const auto channel =
        containing_interval(out.channel_offsets, channels, packet);
    const auto node = static_cast<unsigned>(channel / a.packets.lanes);
    const auto lane = static_cast<unsigned>(channel % a.packets.lanes);
    const auto sequence = packet - out.channel_offsets[channel];
    const auto *prefix = a.packet_prefix + channel * (windows + 1U);
    const auto window = containing_interval(prefix, windows, sequence);
    const auto node_begin = a.node_prefix[window * (a.nodes + 1U) + node];
    const auto tokens =
        a.node_prefix[window * (a.nodes + 1U) + node + 1U] - node_begin;
    const auto local_sequence = sequence - prefix[window];
    const auto begin = window * a.window_capacity() + node_begin +
                       a.packets.token_begin(lane, local_sequence);
    const auto rows = a.packets.token_count(tokens, lane, local_sequence);
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
        }
      }
    }
    out.packets[packet] = {
        a.shape.window(a.source_rank, window * a.shape.tokens), begin, rows,
        fanouts};
  }
}

} // namespace weftmoe::cuda::static_forward
