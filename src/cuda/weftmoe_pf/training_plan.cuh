#pragma once

namespace weftmoe_pf::ring {

// The count-only Notify leaves route-sized tables untouched. Recover stable
// expert/source/route order here using its immutable chunk-count prefixes.
__device__ __noinline__ void build_training_rows(const ForwardKernelArgs &a,
                                                 unsigned char *scratch) {
  constexpr unsigned Threads = 384U, Items = 4U, KeyBits = 11U;
  static_assert(Threads * Items == kTrainingCountChunkRoutes);
  using Sort = cub::BlockRadixSort<std::uint64_t, Threads, Items>;
  union Storage {
    typename Sort::TempStorage sort;
    std::uint64_t keys[kTrainingCountChunkRoutes];
  };
  auto &shared = *reinterpret_cast<Storage *>(scratch);
  auto *order_base =
      reinterpret_cast<std::uint64_t *>(scratch + sizeof(Storage));
  const auto experts = a.dispatch.identity.num_experts;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto routes =
      a.dispatch.identity.num_tokens * a.dispatch.identity.top_k;
  const auto chunks =
      (routes + kTrainingCountChunkRoutes - 1U) / kTrainingCountChunkRoutes;
  const auto *counts =
      a.dispatch.source_expert_count + std::uint64_t(a.rank) * experts;
  for (unsigned expert = threadIdx.x; expert < experts; expert += Threads) {
    const auto node = expert / owned / a.local_world_size;
    std::uint64_t prefix = 0U;
    for (unsigned prior = 0; prior < experts; ++prior) {
      const auto prior_node = prior / owned / a.local_world_size;
      if ((prior_node == a.node && node != a.node) ||
          ((prior_node == a.node) == (node == a.node) && prior < expert))
        prefix += counts[prior];
    }
    order_base[expert] = prefix;
  }
  __syncthreads();
  unsigned expert_bits = 0U;
  for (unsigned value = experts; value; value >>= 1U)
    ++expert_bits;
  for (auto chunk = std::uint64_t(blockIdx.x); chunk < chunks;
       chunk += gridDim.x) {
    std::uint64_t keys[Items];
#pragma unroll
    for (unsigned item = 0; item < Items; ++item) {
      const auto within = threadIdx.x * Items + item;
      const auto route = chunk * kTrainingCountChunkRoutes + within;
      const auto expert =
          route < routes ? unsigned(a.selected_experts[route]) : experts;
      keys[item] = (std::uint64_t(expert) << KeyBits) | within;
    }
    // Input ordinals already increase in blocked order. Stable expert sorting
    // preserves that order without radix passes over the ordinal bits.
    Sort(shared.sort).Sort(keys, KeyBits, expert_bits + KeyBits);
    __syncthreads();
#pragma unroll
    for (unsigned item = 0; item < Items; ++item)
      shared.keys[threadIdx.x * Items + item] = keys[item];
    __syncthreads();
    std::uint64_t ordinals[Items]{};
#pragma unroll
    for (unsigned item = 0; item < Items; ++item) {
      const auto position = threadIdx.x * Items + item;
      const auto expert = unsigned(keys[item] >> KeyBits);
      if (expert >= experts)
        continue;
      unsigned first = 0U, end = position;
      while (first < end) {
        const auto middle = (first + end) / 2U;
        if ((shared.keys[middle] >> KeyBits) < expert)
          first = middle + 1U;
        else
          end = middle;
      }
      ordinals[item] =
          a.checkpoint.counts
              .chunk_prefix[std::uint64_t(expert) * chunks + chunk] +
          position - first;
    }
    // Sorting is only needed for stable ordinals. Restore source-route order
    // in shared memory so the route-indexed tape tables get coalesced stores.
    __syncthreads();
#pragma unroll
    for (unsigned item = 0; item < Items; ++item)
      shared.keys[keys[item] & ((1U << KeyBits) - 1U)] = ordinals[item];
    __syncthreads();
#pragma unroll
    for (unsigned item = 0; item < Items; ++item) {
      const auto within = threadIdx.x + item * Threads;
      const auto route = chunk * kTrainingCountChunkRoutes + within;
      if (route >= routes)
        continue;
      const auto expert = unsigned(a.selected_experts[route]);
      const auto ordinal = shared.keys[within];
      const auto row = a.checkpoint.counts.expert_source_base[expert] + ordinal;
      const auto destination = expert / owned;
      const_cast<std::uint32_t *>(a.dispatch.expert_id)[route] = expert;
      const_cast<std::uint64_t *>(a.dispatch.local_offset)[route] = ordinal;
      const_cast<std::uint64_t *>(a.dispatch.route_dst_row)[route] = row;
      const_cast<std::uint32_t *>(a.dispatch.route_dst_rank)[route] =
          destination;
      const_cast<std::uint32_t *>(a.dispatch.route_dst_node)[route] =
          destination / a.local_world_size;
      const_cast<std::uint32_t *>(a.dispatch.route_dst_local_rank)[route] =
          destination % a.local_world_size;
      const_cast<std::uint64_t *>(
          a.dispatch_waves.packet_order)[order_base[expert] + ordinal] = route;
      const_cast<DeviceReverseRouteRecord *>(
          a.dispatch.reverse_routes)[route] = {
          route / a.dispatch.identity.top_k,           row,    a.rank,
          unsigned(route % a.dispatch.identity.top_k), expert, destination};
    }
    __syncthreads();
  }
  grid_sync<0U, Threads>(a);

  // Interleave experts in 32-row stripes for Backward dY dispatch. Draining
  // an entire expert first concentrates every sender CTA on one peer. Build
  // the permutation directly from counts, without a route-sized radix sort.
  // packet_order is scratch here; the network packet plan is built later.
  if (!threadIdx.x) {
    std::uint64_t stripes = 0U;
    for (unsigned expert = 0; expert < experts; ++expert)
      stripes += (counts[expert] + 31U) / 32U;
    shared.keys[0] = stripes;
  }
  __syncthreads();
  const auto lane = threadIdx.x % 32U;
  for (auto stripe = (std::uint64_t(blockIdx.x) * Threads + threadIdx.x) / 32U;
       stripe < shared.keys[0];
       stripe += std::uint64_t(gridDim.x) * Threads / 32U) {
    unsigned expert = 0U;
    std::uint64_t ordinal = 0U, position = 0U;
    if (!lane) {
      auto index = stripe;
      while (index >= (counts[expert] + 31U) / 32U)
        index -= (counts[expert++] + 31U) / 32U;
      ordinal = index * 32U;
      const bool local = expert / owned / a.local_world_size == a.node;
      for (unsigned prior = 0; prior < experts; ++prior) {
        const bool prior_local = prior / owned / a.local_world_size == a.node;
        const auto count = counts[prior];
        if (prior_local && !local)
          position += count;
        else if (prior_local == local) {
          position += count < ordinal ? count : ordinal;
          if (prior < expert && count > ordinal)
            position += count - ordinal < 32U ? count - ordinal : 32U;
        }
      }
    }
    expert = __shfl_sync(0xffffffffU, expert, 0U);
    ordinal = __shfl_sync(0xffffffffU, ordinal, 0U);
    position = __shfl_sync(0xffffffffU, position, 0U);
    if (ordinal + lane < counts[expert])
      const_cast<std::uint64_t *>(
          a.dispatch_waves.local_route_order)[position + lane] =
          a.dispatch_waves.packet_order[order_base[expert] + ordinal + lane];
  }
  grid_sync<0U, Threads>(a);
}

// Backward needs immutable source-token fanout, not V3's sorted dispatch
// waves. Reuse the PF node directory: packets are node-major and each token
// is sent once to each node it actually hits. No route-sized sort is needed.
__device__ __noinline__ void build_training_packets(const ForwardKernelArgs &a,
                                                    unsigned char *scratch) {
  constexpr unsigned Threads = 384U;
  using Scan = cub::BlockScan<unsigned long long, Threads>;
  auto &scan = *reinterpret_cast<Scan::TempStorage *>(scratch);
  const auto &p = a.ring.source;
  const auto windows = p.local_windows();
  const auto experts_per_node = p.experts / p.nodes;
  for (auto task = std::uint64_t(blockIdx.x); task < windows * p.nodes;
       task += gridDim.x) {
    const auto node = unsigned(task / windows);
    const auto window = task % windows;
    std::uint64_t packet_base = 0U, row_base = 0U;
    for (unsigned prior_node = 0; prior_node <= node; ++prior_node)
      for (std::uint64_t prior_window = 0;
           prior_window < (prior_node == node ? window : windows);
           ++prior_window) {
        const auto *np = p.node_prefix + prior_window * (p.nodes + 1U);
        const auto *ep = p.expert_prefix + prior_window * (p.experts + 1U);
        packet_base += np[prior_node + 1U] - np[prior_node];
        row_base += ep[(prior_node + 1U) * experts_per_node] -
                    ep[prior_node * experts_per_node];
      }
    const auto *prefix = p.node_prefix + window * (p.nodes + 1U);
    const auto tokens = prefix[node + 1U] - prefix[node];
    const auto *records =
        p.node_tokens + window * p.window_capacity() + prefix[node];
    for (std::uint64_t begin = 0; begin < tokens; begin += Threads) {
      const auto index = begin + threadIdx.x;
      const auto count = index < tokens ? records[index].fanouts : 0U;
      unsigned long long offset, total;
      Scan(scan).ExclusiveSum(static_cast<unsigned long long>(count), offset,
                              total);
      if (index < tokens) {
        const auto packet = packet_base + index;
        const auto token = records[index].token;
        const_cast<DevicePacketHeaderRecord *>(
            a.dispatch.packet_headers)[packet] = {
            index,
            token,
            a.rank,
            node,
            node * a.local_world_size + a.rank % a.local_world_size,
            0U};
        const_cast<std::uint64_t *>(a.dispatch_waves.packet_order)[packet] =
            packet;
        const_cast<std::uint64_t *>(a.dispatch.packet_fanout_offsets)[packet] =
            row_base + offset;
        unsigned fanout = 0U;
        for (unsigned slot = 0; slot < p.shape.top_k; ++slot) {
          const auto route = token * p.shape.top_k + slot;
          const auto expert = unsigned(a.selected_experts[route]);
          if (expert / experts_per_node == node)
            const_cast<DeviceFanoutRecord *>(
                a.dispatch.packet_fanout)[row_base + offset + fanout++] = {
                a.dispatch.route_dst_row[route],
                0U,
                slot,
                (expert / a.dispatch.sizes.owned_expert_count) %
                    a.local_world_size,
                expert,
                0U};
        }
      }
      row_base += total;
      __syncthreads();
    }
  }
  if (!blockIdx.x && !threadIdx.x) {
    std::uint64_t packets = 0U;
    for (std::uint64_t window = 0; window < windows; ++window)
      packets += p.node_prefix[window * (p.nodes + 1U) + p.nodes];
    const_cast<std::uint64_t *>(a.dispatch.packet_fanout_offsets)[packets] =
        p.tokens * p.shape.top_k;
    a.checkpoint.counts.runtime->sizes.packet_count = packets;
    a.checkpoint.counts.runtime->sizes.fanout_count = p.tokens * p.shape.top_k;
  }
  grid_sync<0U, Threads>(a);
}

} // namespace weftmoe_pf::ring
