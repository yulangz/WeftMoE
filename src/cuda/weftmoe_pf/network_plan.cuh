#pragma once

#include "cuda/weftmoe_pf/return_plan.cuh"

namespace weftmoe_pf::ring {

__device__ __forceinline__ std::uint64_t completed_blocks(
    std::uint64_t rows, std::uint64_t total, unsigned m) {
  return rows / m + (rows == total && rows % m != 0U);
}

// The planner assigns slots, not runtime work. A reserved tail ring per expert
// prevents cold partial blocks from pinning the ordinary circular ring. Tails
// retire in their last arrival window; successors may start in that window.
__device__ void canonical_geometry(const ForwardKernelArgs &a) {
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto m = r.source.shape.tile_m;
  std::uint64_t canonical = 0U, max_tails = 0U;
  for (unsigned e = 0U; e < owned; ++e) {
    std::uint64_t total = 0U;
    for (std::uint64_t w = 0U; w < r.state->windows; ++w)
      total += r.expert_rows[w * owned + e];
    std::uint64_t row = 0U, tails = 0U;
    for (std::uint64_t w = 0U; w < r.state->windows; ++w) {
      auto &p = r.window_compute[w * owned + e];
      p.row_begin = row;
      p.total_rows = total;
      p.block_begin = canonical;
      p.tail_index = tails;
      const auto end = row + r.expert_rows[w * owned + e];
      const auto count = completed_blocks(end, total, m) -
                         completed_blocks(row, total, m);
      tails += count != 0U && row % m != 0U;
      row = end;
    }
    canonical += static_forward::ceil_div(total, m);
    max_tails = max(max_tails, tails);
  }
  std::uint64_t blocks = 0U, full = 0U, starts = 0U, rows = 0U;
  std::uint64_t max_full_window = 0U;
  for (std::uint64_t w = 0U; w < r.state->windows; ++w) {
    const auto full_begin = full;
    r.window_prefix[w] = blocks;
    r.start_prefix[w] = starts;
    r.pull_prefix[w] = rows;
    auto *prefix = r.expert_blocks + w * (owned + 1U);
    for (unsigned e = 0U; e < owned; ++e) {
      auto &p = r.window_compute[w * owned + e];
      const auto end = p.row_begin + r.expert_rows[w * owned + e];
      const auto count = completed_blocks(end, p.total_rows, m) -
                         completed_blocks(p.row_begin, p.total_rows, m);
      prefix[e] = blocks - r.window_prefix[w];
      p.full_begin = full;
      p.start_begin = starts;
      p.pull_begin = rows;
      full += count - (count != 0U && p.row_begin % m != 0U);
      starts += static_forward::ceil_div(end, m) -
                static_forward::ceil_div(p.row_begin, m);
      blocks += count;
      rows += r.expert_rows[w * owned + e];
    }
    prefix[owned] = blocks - r.window_prefix[w];
    max_full_window = max(max_full_window, full - full_begin);
  }
  // Balance the two banks' prefetch horizon, rather than fixing their ratio
  // for one model/window size. This is a static per-call slot assignment;
  // neither capacity nor runtime task ownership changes. Two tails/expert
  // allow a predecessor to finish in the same window its successor starts.
  r.state->tail_slots = max_tails == 0U ? 0U :
      min(max_tails, min((r.depth - max_full_window) / owned,
          max(min(std::uint64_t(2U), max_tails),
              r.depth * max_tails / (full + owned * max_tails))));
  r.window_prefix[r.state->windows] = blocks;
  r.start_prefix[r.state->windows] = starts;
  r.pull_prefix[r.state->windows] = rows;
  r.state->blocks = blocks;
  r.state->rows = rows;
  auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
  runtime->sizes = a.dispatch.sizes;
  runtime->sizes.incoming_row_count = rows;
  runtime->sizes.mtile_count = blocks;
}

template <bool Training, unsigned Threads>
__device__ __noinline__ void prepare_canonical(const ForwardKernelArgs &a,
                                               unsigned *counts) {
  const auto &r = a.ring;
  const auto &p = r.source;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto m = p.shape.tile_m;
  const auto tid = threadIdx.x;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + tid;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  if (!worker)
    canonical_geometry(a);
  grid_sync<0U, Threads>(a);
  const auto tail_slots = r.state->tail_slots;
  const auto full_depth = r.depth - tail_slots * owned;
  for (auto task = std::uint64_t(blockIdx.x); task < r.state->windows * owned;
       task += gridDim.x) {
    const auto w = task / owned;
    const auto expert = unsigned(task % owned);
    const auto plan = r.window_compute[task];
    const auto *prefix = r.expert_blocks + w * (owned + 1U);
    const auto count = prefix[expert + 1U] - prefix[expert];
    const bool tail = count && plan.row_begin % m;
    for (auto b = std::uint64_t(tid); b < count; b += Threads) {
      const auto ordinal = plan.row_begin / m + b;
      const auto block = r.window_prefix[w] + prefix[expert] + b;
      auto start = w;
      unsigned slot;
      std::uint64_t reuse;
      if (tail && b == 0U) {
        while (ordinal * m < r.window_compute[start * owned + expert].row_begin)
          --start;
        slot = unsigned(full_depth + tail_slots * expert + plan.tail_index % tail_slots);
        reuse = plan.tail_index / tail_slots;
      } else {
        const auto index = plan.full_begin + b - unsigned(tail);
        slot = unsigned(index % full_depth);
        reuse = index / full_depth;
      }
      const auto generation =
          r.slots[slot].fc1_target /
              (2U * forward_tuning::kMainIntermediate / forward_tuning::kFc1TileN) +
          reuse + 1U;
      r.blocks[block] = {
          {expert, unsigned(ordinal * m),
           unsigned(min(std::uint64_t(m), plan.total_rows - ordinal * m))},
          unsigned(w), slot, 0U, generation, 0U};
      const auto &arrival = r.window_compute[start * owned + expert];
      r.start_blocks[arrival.start_begin + ordinal -
                     static_forward::ceil_div(arrival.row_begin, m)] =
          unsigned(block);
      r.canonical_blocks[plan.block_begin + ordinal] = unsigned(block);
    }
  }
  grid_sync<0U, Threads>(a);
  for (auto task = std::uint64_t(blockIdx.x); task < r.state->windows * owned;
       task += gridDim.x) {
    const auto w = task / owned;
    const auto expert = unsigned(task % owned);
    const auto plan = r.window_compute[task];
    const auto first = p.shape.source_begin(w);
    const auto local_window = w / p.shape.groups();
    for (unsigned i = tid; i < p.shape.sources; i += Threads)
      counts[i] = first + i < p.shape.source_end(w)
                      ? unsigned(source_count(a, first + i, local_window,
                                               expert, false))
                      : 0U;
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    const static_forward::WindowGeometryView geometry{
        p.shape, w, 1U, counts, nullptr, nullptr, nullptr};
    std::uint64_t tape_begin = 0U;
    if constexpr (Training)
      tape_begin = a.dispatch.expert_row_prefix[expert] + plan.row_begin;
    for (auto source = first + tid / 32U; source < p.shape.source_end(w);
         source += Threads / 32U) {
      const auto *prefix = source_metadata(a, source, p.expert_prefix) +
                           local_window * (p.experts + 1U);
      const auto *order = source_metadata(a, source, p.route_order) +
                          local_window * p.window_capacity();
      const auto *gates = source_metadata(a, source, r.source_gates);
      for (auto ordinal = tid % 32U; ordinal < counts[source - first];
           ordinal += 32U) {
        const auto row = geometry.expert_row<true>(0U, source, ordinal);
        const auto canonical = plan.row_begin + row;
        const auto block = r.canonical_blocks[plan.block_begin + canonical / m];
        const auto &desc = r.blocks[block];
        const auto physical = block * m + canonical % m;
        const auto route = order[prefix[a.rank * owned + expert] + ordinal];
        const auto gate = gates[route];
        const auto contribution = window_source(a, w, source).row_base +
                                   prefix[a.rank * owned + expert] + ordinal;
        void *output;
        if (source / a.local_world_size == a.node)
          output = ncclGetLsaPointer(
              a.gin_window, a.static_network.contribution +
                                contribution * a.hidden * 2U,
              source % a.local_world_size);
        else
          output = nullptr; // Installed after the node-wide TX partition.
        r.rows[physical] = {unsigned(route), source, gate,
                             unsigned(route / p.shape.top_k), output,
                             contribution};
        // Spread a wave's local NVLink reads over sources, without changing
        // GEMM/checkpoint order or assigning a whole M block to one warp.
        r.pull_order[plan.pull_begin +
                     geometry.expert_row<false>(0U, source, ordinal)] =
            unsigned(physical);
        if constexpr (Training) {
          const auto saved = tape_begin + row;
          a.checkpoint.rows[physical] = saved;
          if (a.checkpoint.gates) {
            a.checkpoint.gates[saved] = gate;
            a.checkpoint.headers[saved] = {
                transport::kActivationTransportSchemaVersion, source, a.rank,
                0U, a.dispatch.identity.generation, execution_epoch(a), route,
                route, route / p.shape.top_k, saved, a.rank * owned + expert,
                unsigned(route % p.shape.top_k)};
          }
        }
        source_metadata(a, source, r.locations)[route] = {
            desc.generation, block, unsigned(desc.slot * m + canonical % m)};
      }
    }
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  }
  // Plan against the committed receive prefix. A tape allocation may fail
  // after Notify; retrying must overwrite this target, not count an unsent
  // invocation twice. Only the receive worker advances forward_rows.
  for (auto node = worker; node < p.nodes; node += workers) {
    if (node == a.node)
      continue;
    std::uint64_t rows = 0U;
    for (std::uint64_t w = 0U; w < p.local_windows(); ++w) {
      const auto *prefix = p.expert_prefix + w * (p.experts + 1U);
      rows += prefix[(node + 1U) * a.local_world_size * owned] -
              prefix[node * a.local_world_size * owned];
    }
    const auto channel = a.static_network.rings.channel(a.node, node, 0U);
    auto &state = *peer_at<static_forward::NetworkChannel>(
        a.gin_window_base, a.static_network.ring_region +
                               a.static_network.rings.state(true, channel));
    state.forward_target = state.forward_rows + rows;
  }
  grid_sync<0U, Threads>(a);
  prepare_return<Threads>(a);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanReady, Kernel,
                             0U, 0U, 0U);
}

} // namespace weftmoe_pf::ring
