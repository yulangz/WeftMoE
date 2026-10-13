#pragma once

#include "cuda/weftmoe_pf/plan.hpp"
#include <cuda_runtime.h>

namespace weftmoe::cuda::static_forward {

template <unsigned Threads, unsigned Barrier>
__device__ __forceinline__ void source_plan_sync() {
  asm volatile("bar.sync %0, %1;" ::"n"(Barrier), "n"(Threads) : "memory");
}

// All lanes participate, including zero-padded lanes in the final chunk.
__device__ __forceinline__ std::uint64_t warp_inclusive_sum(std::uint64_t value) {
  const auto lane = threadIdx.x % 32U;
#pragma unroll
  for (unsigned delta = 1U; delta < 32U; delta *= 2U) {
    const auto previous = __shfl_up_sync(0xffffffffU, value, delta);
    if (lane >= delta)
      value += previous;
  }
  return value;
}

// One warp publishes an exclusive directory and turns histogram entries into
// reservation cursors. Carry across warp-sized chunks without extra scratch.
__device__ __forceinline__ void
prefix_source_counts(unsigned *counts, unsigned size, std::uint64_t *prefix) {
  const auto lane = threadIdx.x % 32U;
  std::uint64_t carry = 0U;
  for (unsigned base = 0U; base < size; base += 32U) {
    const auto index = base + lane;
    const std::uint64_t count = index < size ? counts[index] : 0U;
    const auto sum = warp_inclusive_sum(count);
    if (index < size) {
      prefix[index] = carry + sum - count;
      counts[index] = static_cast<unsigned>(carry + sum - count);
    }
    carry += __shfl_sync(0xffffffffU, sum, 31U);
  }
  if (lane == 0U)
    prefix[size] = carry;
}

// Share route validation between grouped directories and direct publication.
// A second pass need not report an invalid id again after the counting pass.
template <unsigned Threads, unsigned BaseThread = 0U, bool ReportError = true,
          class Emit>
__device__ __forceinline__ void for_each_source_route(
    const SourcePlanView &a, std::uint64_t local_window, Emit emit) {
  const auto begin = local_window * a.shape.tokens;
  const auto remaining = a.tokens - begin;
  const auto tokens = static_cast<unsigned>(
      remaining < a.shape.tokens ? remaining : a.shape.tokens);
  const auto route_begin = begin * a.shape.top_k;
  for (auto index = threadIdx.x - BaseThread;
       index < tokens * a.shape.top_k; index += Threads) {
    const auto route = route_begin + index;
    if (a.valid && !a.valid[route])
      continue;
    const auto expert = a.selected_experts[route];
    if (expert < 0 || static_cast<unsigned>(expert) >= a.experts) {
      if constexpr (ReportError)
        atomicExch(a.error, 1U);
      continue;
    }
    emit(route, static_cast<unsigned>(expert));
  }
}

// Called by a fixed CTA/control group for each of its source token windows.
// The default freezes atomic reservation order. The canonical specialization
// reuses Notify's stable expert/source ordinals: a contiguous token window
// selects a contiguous subsequence of each expert, with no masked rows.
// Local inference needs only the expert directory, not node packet metadata.
template <unsigned Threads, unsigned BaseThread = 0U, unsigned Barrier = 0U,
          bool Canonical = false, bool NodeDirectory = true>
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

  for_each_source_route<Threads, BaseThread>(
      a, local_window, [&](std::uint64_t route, unsigned expert) {
        atomicAdd(scratch + expert, 1U);
        if constexpr (Canonical)
          atomicMin(canonical_begin + expert,
                    static_cast<unsigned long long>(canonical_rows[route]));
      });

  // Each token is owned by one lane. Count one X record for each actually hit
  // node, while retaining all valid route slots, including duplicate experts.
  if constexpr (NodeDirectory) {
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
  }
  source_plan_sync<Threads, Barrier>();

  if (tid < 32U) {
    prefix_source_counts(scratch, a.experts, expert_prefix);
    if constexpr (NodeDirectory)
      prefix_source_counts(scratch + a.experts, a.nodes, node_prefix);
  }
  source_plan_sync<Threads, Barrier>();

  for_each_source_route<Threads, BaseThread, false>(
      a, local_window, [&](std::uint64_t route, unsigned expert) {
        std::uint64_t position;
        if constexpr (Canonical)
          position = expert_prefix[expert] + canonical_rows[route] -
                     canonical_begin[expert];
        else
          position = atomicAdd(scratch + expert, 1U);
        route_order[position] = route;
      });
  if constexpr (NodeDirectory) {
#if defined(WEFTMOE_PF_ORDERED_DISPATCH) && WEFTMOE_PF_ORDERED_DISPATCH
    // The expert cursors are dead after route_order publication. Reuse their
    // storage for a CTA prefix scan, keeping node tokens in source order.
    source_plan_sync<Threads, Barrier>();
    for (unsigned node = 0U; node < a.nodes; ++node) {
      auto cursor = node_prefix[node];
      for (unsigned chunk = 0U; chunk < tokens; chunk += Threads) {
        const auto token = chunk + tid;
        unsigned fanouts = 0U;
        if (token < tokens)
          for (unsigned slot = 0U; slot < a.shape.top_k; ++slot) {
            const auto route =
                route_begin + std::uint64_t(token) * a.shape.top_k + slot;
            const auto expert = a.selected_experts[route];
            fanouts += (!a.valid || a.valid[route]) && expert >= 0 &&
                       unsigned(expert) < a.experts &&
                       unsigned(expert) / experts_per_node == node;
          }
        const auto scan = unsigned(warp_inclusive_sum(fanouts != 0U));
        if (tid % 32U == 31U)
          scratch[tid / 32U] = scan;
        source_plan_sync<Threads, Barrier>();
        unsigned prior = 0U, total = 0U;
#pragma unroll
        for (unsigned warp = 0U; warp < Threads / 32U; ++warp) {
          const auto count = scratch[warp];
          if (warp < tid / 32U)
            prior += count;
          total += count;
        }
        if (fanouts)
          node_tokens[cursor + prior + scan - 1U] = {begin + token, fanouts};
        cursor += total;
        source_plan_sync<Threads, Barrier>();
      }
    }
#else
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
#endif
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
