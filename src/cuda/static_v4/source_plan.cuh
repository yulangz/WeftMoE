#pragma once

#include "cuda/static_v4/plan.hpp"
#include <cuda_runtime.h>

namespace weftmoe::cuda::static_forward {

// Source-local planning storage. A local token window reserves B*K entries,
// which bounds both its expert-sorted routes and node-deduplicated tokens.
// Only metadata scales with invocation length; no activation payload lives here.
struct NodeToken {
  std::uint64_t token;
  std::uint32_t fanouts;
};

struct SourcePlanView {
  WindowShape shape;
  PacketPartition packets;
  std::uint32_t source_rank;
  std::uint32_t experts;
  std::uint32_t nodes;
  std::uint64_t tokens;
  const std::int32_t *selected_experts;
  const std::uint8_t *valid; // Optional explicit mask, never inferred from id.
  std::uint64_t *expert_prefix; // [local_windows, experts+1]
  std::uint64_t *node_prefix;   // [local_windows, nodes+1]
  std::uint64_t *route_order;   // [local_windows, B*K]
  std::uint32_t *source_slot;   // [tokens*K], window/expert-local ordinal
  NodeToken *node_tokens;      // [local_windows, B*K]
  std::uint64_t *packet_prefix; // [nodes, lanes, local_windows+1]
  unsigned *error;             // Zeroed before the planning phase.

  __host__ __device__ std::uint64_t local_windows() const {
    return ceil_div(tokens, shape.tokens);
  }
  __host__ __device__ std::uint64_t window_capacity() const {
    return std::uint64_t(shape.tokens) * shape.top_k;
  }
  __host__ __device__ unsigned scratch_words() const {
    return experts + nodes;
  }
};

template <unsigned Threads, unsigned Barrier>
__device__ __forceinline__ void source_plan_sync() {
  asm volatile("bar.sync %0, %1;" ::"n"(Barrier), "n"(Threads) : "memory");
}

// Called by a fixed CTA/control group for each of its source token windows.
// The reservation order is inference-only; the completed directory is immutable
// before packet enumeration. StableTape must use stable source ordinals instead.
template <unsigned Threads, unsigned BaseThread = 0U, unsigned Barrier = 0U>
__device__ void build_source_window(const SourcePlanView &a,
                                    std::uint64_t local_window,
                                    unsigned *scratch) {
  const auto tid = threadIdx.x - BaseThread;
  const auto begin = local_window * a.shape.tokens;
  const auto remaining = a.tokens - begin;
  const auto tokens = static_cast<unsigned>(
      remaining < a.shape.tokens ? remaining : a.shape.tokens);
  const auto route_begin = begin * a.shape.top_k;
  const auto routes = tokens * a.shape.top_k;
  const auto experts_per_node = a.experts / a.nodes;
  auto *expert_prefix = a.expert_prefix + local_window * (a.experts + 1U);
  auto *node_prefix = a.node_prefix + local_window * (a.nodes + 1U);
  auto *route_order = a.route_order + local_window * a.window_capacity();
  auto *node_tokens = a.node_tokens + local_window * a.window_capacity();

  for (auto index = tid; index < a.scratch_words(); index += Threads)
    scratch[index] = 0U;
  source_plan_sync<Threads, Barrier>();

  for (auto index = tid; index < routes; index += Threads) {
    const auto route = route_begin + index;
    a.source_slot[route] = UINT32_MAX;
    if (a.valid && !a.valid[route])
      continue;
    const auto expert = a.selected_experts[route];
    if (expert < 0 || static_cast<unsigned>(expert) >= a.experts) {
      atomicExch(a.error, 1U);
      continue;
    }
    atomicAdd(scratch + expert, 1U);
  }

  // Each token is owned by one lane. Count one X record for each actually hit
  // node, while retaining all valid route slots, including duplicate experts.
  for (auto token = tid; token < tokens; token += Threads) {
    const auto first_route = route_begin + std::uint64_t(token) * a.shape.top_k;
    for (unsigned k = 0; k < a.shape.top_k; ++k) {
      if (a.valid && !a.valid[first_route + k])
        continue;
      const auto expert = a.selected_experts[first_route + k];
      if (expert < 0 || static_cast<unsigned>(expert) >= a.experts)
        continue;
      const auto node = static_cast<unsigned>(expert) / experts_per_node;
      bool first = true;
      for (unsigned previous = 0; previous < k; ++previous) {
        const auto other = a.selected_experts[first_route + previous];
        if ((!a.valid || a.valid[first_route + previous]) && other >= 0 &&
            static_cast<unsigned>(other) < a.experts &&
            static_cast<unsigned>(other) / experts_per_node == node)
          first = false;
      }
      if (first)
        atomicAdd(scratch + a.experts + node, 1U);
    }
  }
  source_plan_sync<Threads, Barrier>();

  if (tid == 0U) {
    std::uint64_t prefix = 0U;
    for (unsigned expert = 0; expert < a.experts; ++expert) {
      expert_prefix[expert] = prefix;
      const auto count = scratch[expert];
      scratch[expert] = static_cast<unsigned>(prefix);
      prefix += count;
    }
    expert_prefix[a.experts] = prefix;
    prefix = 0U;
    for (unsigned node = 0; node < a.nodes; ++node) {
      node_prefix[node] = prefix;
      const auto count = scratch[a.experts + node];
      scratch[a.experts + node] = static_cast<unsigned>(prefix);
      prefix += count;
    }
    node_prefix[a.nodes] = prefix;
  }
  source_plan_sync<Threads, Barrier>();

  for (auto index = tid; index < routes; index += Threads) {
    const auto route = route_begin + index;
    if (a.valid && !a.valid[route])
      continue;
    const auto expert = a.selected_experts[route];
    if (expert < 0 || static_cast<unsigned>(expert) >= a.experts)
      continue;
    const auto position = atomicAdd(scratch + expert, 1U);
    route_order[position] = route;
    a.source_slot[route] = static_cast<unsigned>(position - expert_prefix[expert]);
  }
  for (auto token = tid; token < tokens; token += Threads) {
    const auto first_route = route_begin + std::uint64_t(token) * a.shape.top_k;
    for (unsigned k = 0; k < a.shape.top_k; ++k) {
      if (a.valid && !a.valid[first_route + k])
        continue;
      const auto expert = a.selected_experts[first_route + k];
      if (expert < 0 || static_cast<unsigned>(expert) >= a.experts)
        continue;
      const auto node = static_cast<unsigned>(expert) / experts_per_node;
      bool first = true;
      unsigned fanouts = 0U;
      for (unsigned other_slot = 0; other_slot < a.shape.top_k; ++other_slot) {
        const auto other = a.selected_experts[first_route + other_slot];
        if ((!a.valid || a.valid[first_route + other_slot]) && other >= 0 &&
            static_cast<unsigned>(other) < a.experts &&
            static_cast<unsigned>(other) / experts_per_node == node) {
          first &= other_slot >= k;
          ++fanouts;
        }
      }
      if (first) {
        const auto position = atomicAdd(scratch + a.experts + node, 1U);
        node_tokens[position] = {begin + token, fanouts};
      }
    }
  }
  source_plan_sync<Threads, Barrier>();
}

// All source windows must be published before this pass (a grid join in the
// fused entry, a kernel boundary in the isolated planning test). One fixed
// worker owns each node/lane; the prefix supplies exact terminal sequences and
// lets RX enumerate packets in network order instead of expert-major order.
__device__ inline void build_source_packet_prefix(const SourcePlanView &a,
                                                  unsigned worker,
                                                  unsigned workers) {
  const auto windows = a.local_windows();
  for (auto channel = worker; channel < a.nodes * a.packets.lanes;
       channel += workers) {
    const auto node = channel / a.packets.lanes;
    const auto lane = channel % a.packets.lanes;
    auto *prefix = a.packet_prefix + channel * (windows + 1U);
    std::uint64_t total = 0;
    prefix[0] = 0U;
    for (std::uint64_t w = 0; w < windows; ++w) {
      const auto *nodes = a.node_prefix + w * (a.nodes + 1U);
      if (node != a.source_rank / (a.shape.world / a.nodes))
        total += a.packets.lane_packets(nodes[node + 1U] - nodes[node], lane);
      prefix[w + 1U] = total;
    }
  }
}

} // namespace weftmoe::cuda::static_forward
