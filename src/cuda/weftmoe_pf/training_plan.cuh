#pragma once

namespace weftmoe_pf::ring {

// The count phase leaves route-sized tables untouched. Recover stable
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
#if defined(WEFTMOE_PF_LOCAL_COMPACT_RETURN) && WEFTMOE_PF_LOCAL_COMPACT_RETURN
  auto *order_base =
      reinterpret_cast<std::uint64_t *>(scratch + sizeof(Storage));
#endif
  const auto experts = a.dispatch.identity.num_experts;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto routes =
      a.dispatch.identity.num_tokens * a.dispatch.identity.top_k;
  const auto chunks =
      (routes + kTrainingCountChunkRoutes - 1U) / kTrainingCountChunkRoutes;
#if defined(WEFTMOE_PF_LOCAL_COMPACT_RETURN) && WEFTMOE_PF_LOCAL_COMPACT_RETURN
  if (!a.use_gin) {
    const auto *counts =
        a.dispatch.source_expert_count + std::uint64_t(a.rank) * experts;
    for (unsigned expert = threadIdx.x; expert < experts; expert += Threads) {
      std::uint64_t prefix = 0U;
      for (unsigned prior = 0; prior < expert; ++prior)
        prefix += counts[prior];
      order_base[expert] = prefix;
    }
  }
#endif
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
#if defined(WEFTMOE_PF_LOCAL_COMPACT_RETURN) && WEFTMOE_PF_LOCAL_COMPACT_RETURN
      if (!a.use_gin)
        a.ring.contribution_rows[route] = order_base[expert] + ordinal;
#endif
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
      const_cast<DeviceReverseRouteRecord *>(
          a.dispatch.reverse_routes)[route] = {
          route / a.dispatch.identity.top_k,           row,    a.rank,
          unsigned(route % a.dispatch.identity.top_k), expert, destination};
    }
    __syncthreads();
  }
  grid_sync<0U, Threads>(a);

}

} // namespace weftmoe_pf::ring
