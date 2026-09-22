#pragma once

namespace static_v4 {

__device__ __forceinline__ void build_combine_order(const ForwardKernelArgs &a,
                                                    std::uint64_t tid,
                                                    std::uint64_t stride) {
  auto *order = const_cast<std::uint32_t *>(a.dispatch.combine_order);
  const auto top_k = a.dispatch.identity.top_k;
  for (auto token = tid; token < a.dispatch.identity.num_tokens;
       token += stride) {
    const auto base = token * top_k;
    // Stable insertion preserves the numerical contract (expert, slot) while
    // avoiding the previous cubic search through all already-selected slots.
    for (unsigned slot = 0; slot < top_k; ++slot) {
      auto position = slot;
      const auto expert = a.selected_experts[base + slot];
      while (position &&
             a.selected_experts[base + order[base + position - 1U]] > expert) {
        order[base + position] = order[base + position - 1U];
        --position;
      }
      order[base + position] = slot;
    }
  }
}

template <unsigned BaseThread, unsigned Threads>
__device__ __forceinline__ void build_geometry(const ForwardKernelArgs &a) {
  if (blockIdx.x == 0U) {
    const auto tid = threadIdx.x - BaseThread;
    const auto owned = a.dispatch.sizes.owned_expert_count;
    auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
    auto *ids = const_cast<std::uint32_t *>(a.dispatch.owned_expert_ids);
    auto *counts = const_cast<std::uint64_t *>(a.dispatch.expert_row_count);
    auto *prefix = const_cast<std::uint64_t *>(a.dispatch.expert_row_prefix);
    auto *offsets = const_cast<std::uint64_t *>(a.dispatch.mtile_row_offsets);
    auto *tile_rows =
        const_cast<std::uint32_t *>(a.dispatch.mtile_expected_rows);
    for (unsigned local = tid; local < owned; local += Threads) {
      std::uint64_t total = 0ULL;
      if (!runtime->error)
        for (unsigned source = 0; source < a.world; ++source)
          total +=
              a.fused_local_expert_count[std::uint64_t(source) * owned + local];
      ids[local] = a.rank * owned + local;
      counts[local] = total;
    }
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    if (tid == 0U) {
      std::uint64_t incoming = 0ULL;
      for (unsigned local = 0; local < owned; ++local)
        incoming += counts[local];
      if (incoming > a.max_destination_rows)
        runtime->error = 1ULL;
      if (runtime->error)
        for (unsigned local = 0; local < owned; ++local)
          counts[local] = 0ULL;
      std::uint64_t rows = 0ULL, tiles = 0ULL;
      for (unsigned local = 0; local < owned; ++local) {
        const auto total = counts[local];
        prefix[local] = rows;
        offsets[local] = tiles;
        const auto extent =
            (total + forward_tuning::kTileM - 1ULL) / forward_tuning::kTileM;
        for (std::uint64_t tile = 0; tile < extent; ++tile)
          tile_rows[tiles + tile] =
              static_cast<unsigned>(min(std::uint64_t(forward_tuning::kTileM),
                                        total - tile * forward_tuning::kTileM));
        rows += total;
        tiles += extent;
      }
      prefix[owned] = rows;
      offsets[owned] = tiles;
      runtime->sizes = a.dispatch.sizes;
      runtime->sizes.incoming_row_count = rows;
      runtime->sizes.mtile_count = tiles;
      runtime->sizes.packet_count = 0ULL;
      runtime->sizes.fanout_count = 0ULL;
      runtime->sizes.outgoing_signal_count = 0U;
      runtime->sizes.incoming_signal_count = 0U;
      runtime->wave_count = 0ULL;
    }
  }
  grid_sync<BaseThread, Threads>(a);
}

// Return the source and source-local ordinal of an expert's interleaved row.
// Lanes stripe the rank domain, so this remains valid beyond one warp of ranks.
__device__ __forceinline__ void
decode_source_row(const ForwardKernelArgs &a, unsigned expert,
                  std::uint64_t row, unsigned &source, unsigned &source_slot) {
  const auto lane = threadIdx.x & 31U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  unsigned peeled = 0U;
  auto slot = static_cast<unsigned>(row);
  while (true) {
    unsigned active = 0U, layer = ~0U;
    for (unsigned rank = lane; rank < a.world; rank += 32U) {
      const auto count =
          a.fused_local_expert_count[std::uint64_t(rank) * owned + expert];
      if (count > peeled) {
        ++active;
        layer = min(layer, static_cast<unsigned>(count) - peeled);
      }
    }
    active = __reduce_add_sync(0xffffffffU, active);
    layer = __reduce_min_sync(0xffffffffU, layer);
    const auto rectangle = std::uint64_t(active) * layer;
    if (slot < rectangle) {
      auto ordinal = slot % active;
      source_slot = peeled + slot / active;
      for (unsigned base = 0; base < a.world; base += 32U) {
        const auto rank = base + lane;
        const bool present =
            rank < a.world &&
            a.fused_local_expert_count[std::uint64_t(rank) * owned + expert] >
                peeled;
        const auto mask = __ballot_sync(0xffffffffU, present);
        const auto count = __popc(mask);
        if (ordinal < count) {
          source = base + __fns(mask, 0U, ordinal + 1U);
          return;
        }
        ordinal -= count;
      }
    }
    slot -= static_cast<unsigned>(rectangle);
    peeled += layer;
  }
}

} // namespace static_v4
