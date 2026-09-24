#pragma once

#include "cuda/static_v4/plan.hpp"
#include <cuda_runtime.h>

namespace weftmoe::cuda::static_forward {

template <unsigned Threads, unsigned Barrier>
__device__ __forceinline__ void source_plan_sync() {
  asm volatile("bar.sync %0, %1;" ::"n"(Barrier), "n"(Threads) : "memory");
}

// Called by a fixed CTA/control group for each of its source token windows.
// The default freezes atomic reservation order. The canonical specialization
// reuses Notify's stable expert/source ordinals: a contiguous token window
// selects a contiguous subsequence of each expert, with no masked rows.
template <unsigned Threads, unsigned BaseThread = 0U, unsigned Barrier = 0U,
          bool Canonical = false>
__device__ void build_source_window(const SourcePlanView &a,
                                    std::uint64_t local_window,
                                    unsigned *scratch,
                                    const std::uint64_t *canonical_rows = nullptr) {
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
  unsigned long long *canonical_begin = nullptr;
  if constexpr (Canonical) {
    canonical_begin = reinterpret_cast<unsigned long long *>(
        scratch + ((a.scratch_words() + 1U) & ~1U));
    for (auto expert = tid; expert < a.experts; expert += Threads)
      canonical_begin[expert] = ~0ULL;
  }
  source_plan_sync<Threads, Barrier>();

  for (auto index = tid; index < routes; index += Threads) {
    const auto route = route_begin + index;
    if (a.valid && !a.valid[route])
      continue;
    const auto expert = a.selected_experts[route];
    if (expert < 0 || static_cast<unsigned>(expert) >= a.experts) {
      atomicExch(a.error, 1U);
      continue;
    }
    atomicAdd(scratch + expert, 1U);
    if constexpr (Canonical)
      atomicMin(canonical_begin + expert,
                static_cast<unsigned long long>(canonical_rows[route]));
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
    std::uint64_t position;
    if constexpr (Canonical)
      position = expert_prefix[expert] + canonical_rows[route] -
                 canonical_begin[expert];
    else
      position = atomicAdd(scratch + expert, 1U);
    route_order[position] = route;
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
// warp owns each node/lane; the prefix supplies exact terminal sequences and
// lets RX enumerate packets in network order instead of expert-major order.
__device__ inline void build_packet_prefix(
    const SourcePlanView &a, PacketPartition packets,
    const std::uint64_t *row_prefix, unsigned row_stride, unsigned node_stride,
    std::uint64_t *output, unsigned worker, unsigned workers) {
  const auto windows = a.local_windows();
  const auto lane = worker % 32U;
  for (auto channel = worker / 32U; channel < a.nodes * packets.lanes;
       channel += workers / 32U) {
    const auto node = channel / packets.lanes;
    const auto packet_lane = channel % packets.lanes;
    auto *prefix = output + channel * (windows + 1U);
    std::uint64_t carry = 0U;
    if (lane == 0U)
      prefix[0] = 0U;
    for (std::uint64_t base = 0; base < windows; base += 32U) {
      const auto w = base + lane;
      std::uint64_t count = 0U;
      if (w < windows && node != a.source_rank / (a.shape.world / a.nodes)) {
        const auto *rows = row_prefix + w * row_stride + node * node_stride;
        count = packets.lane_packets(rows[node_stride] - rows[0], packet_lane);
      }
#pragma unroll
      for (unsigned delta = 1U; delta < 32U; delta *= 2U) {
        const auto previous = __shfl_up_sync(0xffffffffU, count, delta);
        if (lane >= delta)
          count += previous;
      }
      if (w < windows)
        prefix[w + 1U] = carry + count;
      carry += __shfl_sync(0xffffffffU, count, 31U);
    }
  }
}

__device__ inline void build_source_packet_prefix(const SourcePlanView &a,
                                                  unsigned worker,
                                                  unsigned workers) {
  build_packet_prefix(a, a.packets, a.node_prefix, a.nodes + 1U, 1U,
                       a.packet_prefix, worker, workers);
}

} // namespace weftmoe::cuda::static_forward
