#pragma once

namespace weftmoe_pf::ring {

__device__ __forceinline__ void finish_network(const ForwardKernelArgs &a) {
  constexpr unsigned Base = 256U, Threads = 128U;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (blockIdx.x == 0U && threadIdx.x == Base)
    status->after_combine_start_clock = device_globaltimer_ns();
  // All static owners have retired their packets, readers and local NIC
  // completions. Protect invocation metadata before a peer stages its next
  // plan. No data-plane Put remains to fence: one context suffices for this
  // collective rendezvous, after the complete grid has joined.
  network_barrier<Base, Threads>(a, 1U);
  if (blockIdx.x == 0U) {
    const auto &network = a.static_network;
    for (unsigned index = threadIdx.x - Base;
         index < 2U * network.rings.channels(); index += Threads) {
      auto &channel = *peer_at<static_forward::NetworkChannel>(
          a.gin_window_base,
          network.ring_region + index * sizeof(static_forward::NetworkChannel));
      channel.tx_base = channel.tx_cursor;
      channel.rx_base = channel.rx_cursor;
    }
    if (threadIdx.x == Base)
      status->after_combine_complete_clock = device_globaltimer_ns();
  }
}

// Freeze RX addressing and source directory pointers once. Runtime workers
// visit only their packet residue, not every source/channel prefix per window.
__device__ __forceinline__ void build_pull_packets(const ForwardKernelArgs &a,
                                                   std::uint64_t window,
                                                   unsigned tid,
                                                   unsigned threads) {
  const auto &r = a.ring;
  const auto &p = r.source;
  const auto &network = a.static_network;
  const auto stride = p.shape.sources * p.packets.packets(p.shape.tokens);
  const auto local_window = window / p.shape.groups();
  const auto rail = a.rank % a.local_world_size;
  for (auto index = std::uint64_t(tid); index < stride; index += threads) {
    auto &out = r.pull_packets[window * stride + index];
    out.rows = 0U;
    const auto source = p.shape.source_begin(window) + index % p.shape.sources;
    if (source >= p.shape.source_end(window) ||
        source / a.local_world_size == a.node)
      continue;
    const auto source_windows = static_forward::ceil_div(
        r.source_epochs[source].tokens, p.shape.tokens);
    if (local_window >= source_windows)
      continue;
    const auto ordinal = index / p.shape.sources;
    const auto lane = ordinal % p.packets.lanes;
    const auto directory = a.node * p.packets.lanes + lane;
    const auto *offsets =
        source_metadata(a, source, r.dispatch.channel_offsets);
    const auto *prefix = source_metadata(a, source, p.packet_prefix) +
                         directory * (source_windows + 1U);
    const auto packet =
        offsets[directory] + prefix[local_window] + ordinal / p.packets.lanes;
    if (packet >= offsets[directory] + prefix[local_window + 1U])
      continue;
    const auto count = source_metadata(
        a, source,
        r.dispatch.consumer_rows)[packet * a.local_world_size + rail];
    if (!count)
      continue;
    const auto channel =
        network.rings.channel(a.node, source / a.local_world_size, lane);
    const auto *state =
        static_cast<const static_forward::NetworkChannel *>(ncclGetLsaPointer(
            a.gin_window,
            network.ring_region + network.rings.state(false, channel),
            source % a.local_world_size));
    const auto sequence = state->rx_base + packet - offsets[directory];
    auto *control = static_cast<static_forward::NetworkSlot *>(
        ncclGetLsaPointer(a.gin_window,
                          network.ring_region +
                              network.rings.control(false, channel, sequence),
                          source % a.local_world_size));
    const auto desc = source_metadata(a, source, r.dispatch.packets)[packet];
    out = {source_metadata(a, source, p.node_tokens) + desc.token_begin,
           source_metadata(a, source, p.selected_experts),
           source_metadata(a, source, r.source_gates),
           source_metadata(a, source, r.locations),
           ncclGetLsaPointer(
               a.gin_window,
               network.ring_region +
                   network.rings.dispatch_payload(true, channel, sequence),
               source % a.local_world_size),
           &control->visible_generation,
           &control->consumed_total,
           sequence / network.rings.depth + 1U,
           desc.tokens,
           count,
           source % a.local_world_size != rail ? count : 0U};
  }
}

__device__ __forceinline__ void stage_input(const ForwardKernelArgs &a,
                                            std::uint64_t tid,
                                            std::uint64_t stride) {
  auto *input = peer_at<uint4>(a.local_base, a.source_input_offset);
  for (auto v = tid; v < a.ring.source.tokens * a.hidden / 8U; v += stride)
    input[v] = reinterpret_cast<const uint4 *>(a.input)[v];
}

// One capacity window needs only the destination directory. Shared counters
// assign source-expert ordinals directly, without a grouped source copy.
// Empty sources still overwrite every count before the source-ready barrier.
template <unsigned Threads, unsigned BaseThread>
__device__ __forceinline__ void build_single_window(
    const ForwardKernelArgs &a, unsigned *counts) {
  const auto &p = a.ring.source;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto tid = threadIdx.x - BaseThread;
  for (unsigned expert = tid; expert < p.experts; expert += Threads)
    counts[expert] = 0U;
  collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  static_forward::for_each_source_route<Threads, BaseThread>(
      p, 0U, [&](std::uint64_t route, unsigned expert) {
        const auto ordinal = atomicAdd(counts + expert, 1U);
        weftmoe_pf::source_routes<static_forward::SourceRoute>(
            a, expert / owned, a.rank, expert % owned)[ordinal] = {
                static_cast<unsigned>(route), a.gates[route]};
      });
  collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  for (unsigned expert = tid; expert < p.experts; expert += Threads)
    weftmoe_pf::source_counts(a, expert / owned, a.rank)[expert % owned] =
        counts[expert];
}

__device__ __forceinline__ void *local_row_output(const ForwardKernelArgs &a,
                                                   std::uint64_t window,
                                                   unsigned source,
                                                   std::uint64_t route) {
  const auto &address = window_source(a, window, source);
  const auto &shape = a.ring.source.shape;
  return peer_at<transport::BFloat16>(a.peer_bases[source],
                                       a.contribution_offset) +
         (address.row_base + route - shape.token_begin(window) * shape.top_k) *
             a.hidden;
}

__device__ __forceinline__ void build_blocks(const ForwardKernelArgs &a,
                                              std::uint64_t window,
                                              unsigned expert, unsigned tid,
                                              unsigned threads) {
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto *prefix = r.expert_blocks + window * (owned + 1U);
  for (auto b = std::uint64_t(tid); b < prefix[expert + 1U] - prefix[expert];
       b += threads) {
    const auto block = r.window_prefix[window] + prefix[expert] + b;
    const auto seq = r.state->block_base + block;
    r.blocks[block] = {
        {expert, static_cast<unsigned>(b * r.source.shape.tile_m),
         static_forward::block_rows(r.expert_rows[window * owned + expert], b,
                                    r.source.shape.tile_m)},
        static_cast<unsigned>(window), static_cast<unsigned>(seq % r.depth),
        seq / r.depth + 1U};
  }
}

// CTA 0 still owns the count table. Four lanes stripe each source/expert
// list, keeping sparse lists from consuming an entire warp or separate CTA.
template <unsigned Threads>
__device__ __forceinline__ void build_small_window_rows(
    const ForwardKernelArgs &a, const unsigned *counts, unsigned tid) {
  constexpr unsigned Group = 4U;
  const auto &r = a.ring;
  const auto &shape = r.source.shape;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const static_forward::WindowGeometryView geometry{
      shape, 0U, owned, counts, r.expert_rows, r.expert_blocks, nullptr};
  for (unsigned pair = tid / Group; pair < owned * a.world;
       pair += Threads / Group) {
    const auto expert = pair / a.world;
    const auto source = pair % a.world;
    const auto count = counts[std::uint64_t(expert) * shape.sources + source];
    const auto *routes = weftmoe_pf::source_routes<static_forward::SourceRoute>(
        a, a.rank, source, expert);
    for (unsigned ordinal = tid % Group; ordinal < count; ordinal += Group) {
      const auto row = geometry.expert_row(expert, source, ordinal);
      const auto identity = routes[ordinal];
      const auto route = identity.route;
      r.rows[r.expert_blocks[expert] * shape.tile_m + row] = {
          route, source, identity.gate, route / shape.top_k,
          local_row_output(a, 0U, source, route)};
    }
  }
}

template <bool Network, bool Training = false>
__device__ __noinline__ void prepare(const ForwardKernelArgs &a,
                                     unsigned char *scratch) {
  constexpr unsigned Math = 256U, SourceThreads = 128U, Threads = 384U;
  constexpr bool Canonical = Training && !Network;
  const auto &r = a.ring;
  const auto &p = r.source;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  // With one capacity window, CTA 0 owns both source counting and destination
  // geometry. Keep their intermediate publications CTA-local until other CTAs
  // need the finished prefix. Use capacity, not this rank's actual token count:
  // unequal peer lengths must never select different planning dependencies.
  const bool single_window = !Network && !Training &&
      p.shape.windows(a.max_tokens_per_rank) == 1U;
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    r.source_header->epoch = {p.tokens, r.state->source_base};
    r.source_header->error = 0U;
    r.state->admitted_windows = 0U;
    if (a.runtime_epoch)
      ++*a.runtime_epoch;
  }
  if (single_window) {
    if (blockIdx.x == 0U)
      collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  } else {
    grid_sync(a);
  }
  if constexpr (Training)
    if (a.checkpoint.counts.runtime)
      build_training_rows(a, scratch);
  const auto tid = threadIdx.x;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + tid;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  auto *counts = reinterpret_cast<unsigned *>(scratch);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanBegin,
                             Kernel, 0U, 0U, 0U);
  if (threadIdx.x < Math) {
    const auto tid = std::uint64_t(blockIdx.x) * Math + threadIdx.x;
    const auto stride = std::uint64_t(gridDim.x) * Math;
    stage_input(a, tid, stride);
    for (auto route = tid; route < p.tokens * p.shape.top_k; route += stride) {
      r.source_gates[route] = a.gates[route];
      if constexpr (Training)
        a.checkpoint.source_rows[route] = a.dispatch.route_dst_row[route];
      if constexpr (Network)
        const_cast<std::int32_t *>(p.selected_experts)[route] =
            a.selected_experts[route];
    }
    if (!Training || a.checkpoint.counts.runtime)
      build_combine_order(a, tid, stride);
  } else {
    if (single_window) {
      if (blockIdx.x == 0U)
        build_single_window<SourceThreads, Math>(a, counts);
    } else {
      auto planner = p;
      if constexpr (Network)
        planner.selected_experts = a.selected_experts;
      for (auto w = std::uint64_t(blockIdx.x); w < p.local_windows();
           w += gridDim.x)
        static_forward::build_source_window<SourceThreads, Math,
                                            kCollectiveScopeBarrier, Canonical,
                                            Network || Training>(
            planner, w, counts, a.dispatch.route_dst_row);
    }
  }
  // Input staging overlaps source planning in both modes. The host has
  // already allocated the training tape; math has no live GEMM state yet.
  collective_scope_sync<Threads, kCollectiveJoinBarrier>();
  if constexpr (Network) {
    const auto &network = a.static_network;
    for (auto index = worker; index < 2U * network.rings.channels();
         index += workers) {
      auto &channel = *peer_at<static_forward::NetworkChannel>(
          a.gin_window_base,
          network.ring_region + index * sizeof(static_forward::NetworkChannel));
      channel.tx_cursor = channel.tx_base;
      channel.tx_reserve_cursor = channel.tx_reserved_head = channel.tx_base;
      channel.tx_completed_head = channel.tx_base;
      channel.rx_cursor = channel.consume_cursor = channel.rx_base;
      channel.acknowledged_head = channel.rx_base;
    }
  }
  if constexpr (Network) {
    grid_sync<0U, Threads>(a);
    if constexpr (Training)
      if (a.checkpoint.counts.runtime)
        build_training_packets(a, scratch);
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanSourceReady,
                               Kernel, 0U, 0U, 0U);
    for (auto index = worker; index < p.local_windows() * p.window_capacity();
         index += workers) {
      const auto window = index / p.window_capacity();
      const auto ordinal = index % p.window_capacity();
      if (ordinal < p.expert_prefix[window * (p.experts + 1U) + p.experts])
        r.contribution_rows[p.route_order[index]] = static_cast<unsigned>(ordinal);
    }
    static_forward::build_source_packet_prefix(p, worker, workers);
    static_forward::build_return_packet_prefix(p, r.returning, worker, workers);
    grid_sync<0U, Threads>(a);
    if (worker == 0U) {
      static_forward::build_channel_offsets(
          p.packet_prefix, p.nodes * p.packets.lanes, p.local_windows(),
          r.dispatch.channel_offsets);
      static_forward::build_channel_offsets(
          r.returning.packet_prefix, p.nodes * p.packets.lanes,
          p.local_windows(), r.returning.channel_offsets);
    }
    grid_sync<0U, Threads>(a);
    static_forward::build_dispatch_packets(p, r.dispatch, worker, workers);
    static_forward::build_return_packets(p, r.returning, worker, workers);
    grid_sync<0U, Threads>(a);
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanPacketsReady,
                               Kernel, 0U, 0U, 0U);
    if (blockIdx.x < a.static_network.contexts && tid == 0U) {
      const auto base = source_offset(a, a.rank);
      for (unsigned node = 0U; node < p.nodes; ++node)
        if (node != a.node)
          network_put(a, node, base, base, a.static_network.source_bytes);
    }
    // network_put assigns contiguous chunks to contexts from zero. Fence
    // every owner with a Put, but do not signal on the unused contexts.
    const auto chunks = static_forward::ceil_div(a.static_network.source_bytes,
                                                 kNetworkChunkBytes);
    network_barrier<0U, Threads>(
        a, static_cast<unsigned>(min(chunks, std::uint64_t(a.static_network.contexts))));
  } else {
    if (single_window)
      rank_barrier<0U, Threads, false>(a);
    else
      rank_barrier<0U, Threads>(a);
  }
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanExchangeReady,
                             Kernel, 0U, 0U, 0U);

  if (blockIdx.x == 0U) {
    unsigned error = 0U;
    for (unsigned source = tid; source < a.world; source += Threads) {
      const auto *header = source_metadata(a, source, r.source_header);
      r.source_epochs[source] = header->epoch;
      error |= header->error;
    }
    error = __reduce_or_sync(0xffffffffU, error);
    if (tid % 32U == 0U)
      counts[tid / 32U] = error;
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    if (tid == 0U) {
      std::uint64_t tokens = 0U;
      for (unsigned source = 0; source < a.world; ++source)
        tokens = tokens > r.source_epochs[source].tokens
                     ? tokens
                     : r.source_epochs[source].tokens;
      for (unsigned warp = 0; warp < Threads / 32U; ++warp)
        error |= counts[warp];
      r.state->windows = error ? 0U : p.shape.windows(tokens);
      auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
      runtime->error = error;
      peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)->error =
          error;
    }
  }
  if (single_window) {
    if (blockIdx.x == 0U)
      collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  } else {
    grid_sync<0U, Threads>(a);
  }

  for (auto w = std::uint64_t(blockIdx.x);
       (!single_window || blockIdx.x == 0U) && w < r.state->windows;
       w += gridDim.x) {
    const auto first = p.shape.source_begin(w);
    const auto local_window = w / p.shape.groups();
    if (tid < p.shape.source_end(w) - first) {
      const auto seq = r.source_epochs[first + tid].base + local_window;
      const auto slot = static_cast<unsigned>(seq % r.contribution_depth);
      r.window_sources[w * p.shape.sources + tid] = {
          seq, seq / r.contribution_depth + 1U,
          std::uint64_t(slot) * r.contribution_stride, slot};
    }
    if constexpr (Network)
      build_pull_packets(a, w, tid, Threads);
    for (unsigned index = tid; index < owned * p.shape.sources;
         index += Threads) {
      const auto expert = index / p.shape.sources;
      const auto source = first + index % p.shape.sources;
      counts[index] =
          source < p.shape.source_end(w)
              ? static_cast<unsigned>(source_count(a, source, local_window,
                                                   expert, single_window))
              : 0U;
    }
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    static_forward::WindowGeometryView geometry{p.shape,
                                                w,
                                                owned,
                                                counts,
                                                r.expert_rows + w * owned,
                                                r.expert_blocks +
                                                    w * (owned + 1U),
                                                nullptr};
    static_forward::build_window_counts<Threads, 0U, kCollectiveScopeBarrier>(
        geometry);
  }
  if (!single_window)
    grid_sync<0U, Threads>(a);
  if (blockIdx.x == 0U && tid == 0U) {
    std::uint64_t blocks = 0U, rows = 0U;
    for (std::uint64_t w = 0; w < r.state->windows; ++w) {
      r.window_prefix[w] = blocks;
      blocks += r.expert_blocks[w * (owned + 1U) + owned];
      for (unsigned e = 0; e < owned; ++e)
        rows += r.expert_rows[w * owned + e];
    }
    r.window_prefix[r.state->windows] = blocks;
    r.state->blocks = blocks;
    r.state->rows = rows;
    auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
    runtime->sizes = a.dispatch.sizes;
    runtime->sizes.incoming_row_count = rows;
    runtime->sizes.mtile_count = blocks;
    auto *scheduler =
        peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
    scheduler[0] = scheduler[1] = scheduler[2] = 0U;
    scheduler[3] = blocks;
    scheduler[4U + a.world] = scheduler[11U + 2U * a.world] = 0U;
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    status->dispatch_blocks_done = status->reduced_tokens =
        status->fc1_start_clock = status->relay_blocks_done = ~0ULL;
  }
  if (single_window && blockIdx.x == 0U) {
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    if (r.state->windows == 1U && r.state->rows <= 2U * gridDim.x) {
      for (unsigned expert = tid; expert < owned; expert += Threads)
        build_blocks(a, 0U, expert, 0U, 1U);
      build_small_window_rows<Threads>(a, counts, tid);
    }
  }
  grid_sync<0U, Threads>(a);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanGeometryReady,
                             Kernel, 0U, 0U, 0U);
  if (single_window && r.state->windows == 1U &&
      r.state->rows <= 2U * gridDim.x) {
    // The existing geometry join also publishes the complete reverse table.
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanReady,
                               Kernel, 0U, 0U, 0U);
    return;
  }
  for (auto task = std::uint64_t(blockIdx.x); task < r.state->windows * owned;
       task += gridDim.x) {
    const auto w = task / owned;
    const auto expert = static_cast<unsigned>(task % owned);
    const auto *prefix = r.expert_blocks + w * (owned + 1U);
    const auto first = p.shape.source_begin(w);
    const auto local_window = w / p.shape.groups();
    for (unsigned index = tid; index < p.shape.sources; index += Threads) {
      const auto source = first + index;
      counts[index] =
          source < p.shape.source_end(w)
              ? static_cast<unsigned>(source_count(a, source, local_window,
                                                   expert, single_window))
              : 0U;
    }
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    const static_forward::WindowGeometryView geometry{
        p.shape,
        w,
        1U,
        counts,
        r.expert_rows + w * owned + expert,
        r.expert_blocks + w * (owned + 1U) + expert,
        nullptr};
    build_blocks(a, w, expert, tid, Threads);
    if constexpr (Network)
      collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    // Freeze reverse identity and gate before payload starts. Each warp reads
    // a contiguous source-expert route list; pull does no peer count/ordinal
    // search, and Return never depends on the lifetime of an X slot.
    for (auto source = first + tid / 32U; source < p.shape.source_end(w);
         source += Threads / 32U) {
      const auto count = counts[source - first];
      const auto *source_prefix = source_metadata(a, source, p.expert_prefix) +
                                  local_window * (p.experts + 1U);
      const auto *order = source_metadata(a, source, p.route_order) +
                          local_window * p.window_capacity();
      const auto *gates = source_metadata(a, source, r.source_gates);
      for (auto ordinal = tid % 32U; ordinal < count; ordinal += 32U) {
        const auto row = geometry.expert_row<Canonical>(0U, source, ordinal);
        std::uint64_t route;
        float gate;
        if (single_window) {
          const auto identity =
              weftmoe_pf::source_routes<static_forward::SourceRoute>(
                  a, a.rank, source, expert)[ordinal];
          route = identity.route;
          gate = identity.gate;
        } else {
          route = order[source_prefix[a.rank * owned + expert] + ordinal];
          gate = gates[route];
        }
        void *output;
        if (!Network || source / a.local_world_size == a.node) {
          const auto &address = window_source(a, w, source);
          if constexpr (Network) {
            const auto output_row = address.row_base +
                source_prefix[a.rank * owned + expert] + ordinal;
            output = ncclGetLsaPointer(
                a.gin_window, a.static_network.contribution +
                                  output_row * a.hidden * sizeof(transport::BFloat16),
                source % a.local_world_size);
          } else {
            output = local_row_output(a, w, source, route);
          }
        } else {
          const auto node_expert = a.node * a.local_world_size * owned;
          const auto node_row = source_prefix[a.rank * owned + expert] +
                                ordinal - source_prefix[node_expert];
          output = return_tx::row_output(a, source, local_window, node_row);
        }
        r.rows[(r.window_prefix[w] + prefix[expert]) * p.shape.tile_m + row] = {
            static_cast<unsigned>(route), source, gate,
            static_cast<unsigned>(route / p.shape.top_k), output};
        if constexpr (Training)
          a.checkpoint.rows[(r.window_prefix[w] + prefix[expert]) * p.shape.tile_m + row] =
              source_metadata(a, source, a.checkpoint.source_rows)[route];
        if constexpr (Network) {
          const auto block = r.window_prefix[w] + prefix[expert] +
                             row / forward_tuning::kTileM;
          const auto &desc = r.blocks[block];
          source_metadata(a, source, r.locations)[route] = {
              desc.generation, block,
              desc.slot * forward_tuning::kTileM +
                  static_cast<unsigned>(row % forward_tuning::kTileM)};
        }
      }
    }
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  }
  grid_sync<0U, Threads>(a);
  if constexpr (Network)
    rank_barrier<0U, Threads, true, false, true>(a);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, worker == 0U, RingPlanReady,
                             Kernel, 0U, 0U, 0U);
}

} // namespace weftmoe_pf::ring
