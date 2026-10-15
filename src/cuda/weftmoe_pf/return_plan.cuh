#pragma once

namespace weftmoe_pf::ring {

__device__ __forceinline__ unsigned
return_rail_width(const ForwardKernelArgs &a) {
  return WEFTMOE_PF_RETURN_RAILS
             ? min(unsigned(WEFTMOE_PF_RETURN_RAILS), a.local_world_size)
             : a.local_world_size;
}

struct ReturnRailGroup {
  unsigned first, size;
};

__device__ __forceinline__ ReturnRailGroup
return_rail_group(const ForwardKernelArgs &a, unsigned expert) {
  const auto width = return_rail_width(a);
  const auto first =
      expert / a.dispatch.sizes.owned_expert_count / width * width;
  return {first, min(width, a.local_world_size - first)};
}

// All rails use the same node-wide static partition. Hot experts receive a
// larger TX horizon; small experts cannot strand most of the bounded arena.
template <unsigned Threads>
__device__ void prepare_return_capacity(const ForwardKernelArgs &a) {
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto experts = owned * a.local_world_size;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + threadIdx.x;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  for (auto expert = worker; expert < experts; expert += workers) {
    const auto destination = a.node * a.local_world_size + expert / owned;
    const auto *state = peer(a, destination, r.state);
    const auto rows =
        state->windows
            ? peer(a, destination, r.window_compute)[expert % owned].total_rows
            : 0U;
    r.return_experts[expert].depth = static_forward::ceil_div(
        rows, r.source.shape.tile_m * return_rail_group(a, expert).size);
  }
  grid_sync<0U, Threads>(a);
  const auto groups =
      static_forward::ceil_div(a.local_world_size, return_rail_width(a));
  if (worker < groups) {
    const auto first = unsigned(worker) * return_rail_width(a) * owned;
    const auto end = min(experts, first + return_rail_width(a) * owned);
    std::uint64_t total = 0U, active = 0U;
    for (unsigned e = first; e < end; ++e) {
      total += r.return_experts[e].depth;
      active += r.return_experts[e].depth != 0U;
    }
    std::uint64_t prefix = 0U;
    unsigned slots = 0U;
    for (unsigned e = first; e < end; ++e) {
      auto &out = r.return_experts[e];
      const auto blocks = out.depth;
      const auto extra = r.return_capacity - active;
      const auto depth = total <= r.return_capacity
                             ? blocks
                             : min(std::uint64_t(blocks),
                                   std::uint64_t(blocks != 0U) +
                                       extra * (prefix + blocks) / total -
                                       extra * prefix / total);
      out = {slots, unsigned(depth)};
      prefix += blocks;
      slots += depth;
    }
  }
  grid_sync<0U, Threads>(a);
}

// Each whole M-block is striped over the node's rails. A block occupies just
// one rail, avoiding replication of its TX capacity across all source rails.
template <unsigned Threads>
__device__ __noinline__ void prepare_return(const ForwardKernelArgs &a) {
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto local_rank = a.rank % a.local_world_size;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + threadIdx.x;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  const auto lane = threadIdx.x % 32U;
  constexpr auto m = forward_tuning::kTileM;
  constexpr auto tiles =
      forward_tuning::kMainHidden / forward_tuning::kFc1TileN;
  if constexpr (!WEFTMOE_PF_RETURN_EXPERT_TX) {
    // Preserve the compute-slot address mapping when isolating TX lifetime.
    for (auto block = worker / 32U; block < r.state->blocks;
         block += workers / 32U) {
      auto &desc = r.blocks[block];
      const auto offset = a.static_network.ring_region +
                          a.static_network.rings.return_payload(0U, 0U) +
                          std::uint64_t(desc.slot) * m * a.hidden * 2U;
      unsigned local_rows = 0U;
      for (unsigned row = lane; row < desc.tile.rows; row += 32U) {
        auto &identity = r.rows[block * m + row];
        const bool local = identity.source / a.local_world_size == a.node;
        local_rows += local;
        if (!local)
          identity.output =
              peer_at<char>(a.gin_window_base, offset + row * a.hidden * 2U);
      }
      local_rows = __reduce_add_sync(0xffffffffU, local_rows);
      if (!lane) {
        desc.local_rows = local_rows;
        auto &slot = r.slots[desc.slot];
        r.return_tasks[block] = {
            r.rows + block * m,
            offset,
            &slot.fc2_published,
            &slot.returned,
            desc.generation * tiles,
            desc.tile.rows == local_rows ? 0U : desc.tile.rows,
            desc.tile.rows};
        if constexpr (WEFTMOE_PF_RETURN_INDEPENDENT_TX) {
          desc.return_slot = desc.slot * a.local_world_size + local_rank;
          auto &tx = r.return_slots[desc.return_slot];
          constexpr auto fc1_tiles =
              2U * forward_tuning::kMainIntermediate / forward_tuning::kFc1TileN;
          // Compute counters may also advance in BWD. Independent TX epochs
          // must start from this physical TX slot's own completed invocation.
          desc.return_generation = tx.published / tiles + desc.generation -
                                   slot.fc1_target / fc1_tiles;
          auto &task = r.return_tasks[block];
          task.published = &tx.published;
          task.returned = &tx.consumed;
          task.target = desc.return_generation * tiles;
          task.credit_rows = desc.tile.rows - local_rows;
        }
      }
    }
    if (!worker)
      r.state->return_tasks = r.state->blocks;
    rank_barrier<0U, Threads, true, true, true>(a);
    return;
  }
  rank_barrier<0U, Threads, true, true, true>(a);
  prepare_return_capacity<Threads>(a);
  for (auto block = worker / 32U; block < r.state->blocks;
       block += workers / 32U) {
    auto &desc = r.blocks[block];
    const auto expert = local_rank * owned + desc.tile.expert;
    const auto plan = r.return_experts[expert];
    const auto ordinal = desc.tile.expert_row / m;
    const auto group = return_rail_group(a, expert);
    const auto rail = group.first + (expert + ordinal) % group.size;
    const auto index = plan.begin + ordinal / group.size % plan.depth;
    unsigned local_rows = 0U;
    for (unsigned row = lane; row < desc.tile.rows; row += 32U) {
      auto &identity = r.rows[block * m + row];
      const bool local = identity.source / a.local_world_size == a.node;
      local_rows += local;
      if (!local)
        identity.output = ncclGetLsaPointer(
            a.gin_window,
            a.static_network.ring_region +
                a.static_network.rings.return_payload(0U, 0U) +
                (std::uint64_t(index) * m + row) * a.hidden * 2U,
            rail);
    }
    local_rows = __reduce_add_sync(0xffffffffU, local_rows);
    if (!lane) {
      desc.local_rows = local_rows;
      desc.return_slot = index * a.local_world_size + rail;
      desc.return_generation =
          r.return_slots[desc.return_slot].published / tiles +
          ordinal / group.size / plan.depth + 1U;
    }
  }
  rank_barrier<0U, Threads, true, true, true>(a);
  if (!worker) {
    std::uint64_t blocks = 0U;
    for (unsigned local = 0U; local < a.local_world_size; ++local)
      blocks =
          max(blocks,
              peer(a, a.node * a.local_world_size + local, r.state)->blocks);
    r.state->return_tasks = blocks * a.local_world_size;
  }
  grid_sync<0U, Threads>(a);
  for (auto task = worker / 32U; task < r.state->return_tasks;
       task += workers / 32U) {
    if (lane)
      continue;
    const auto destination =
        a.node * a.local_world_size + task % a.local_world_size;
    const auto block = task / a.local_world_size;
    if (block >= peer(a, destination, r.state)->blocks) {
      r.return_tasks[task].count = 0U;
      continue;
    }
    const auto desc = peer(a, destination, r.blocks)[block];
    const auto expert =
        destination % a.local_world_size * owned + desc.tile.expert;
    const auto ordinal = desc.tile.expert_row / m;
    const auto group = return_rail_group(a, expert);
    const auto rail = group.first + (expert + ordinal) % group.size;
    if (rail != local_rank || desc.tile.rows == desc.local_rows) {
      r.return_tasks[task].count = 0U;
      continue;
    }
    const auto plan = r.return_experts[expert];
    const auto index = plan.begin + ordinal / group.size % plan.depth;
    auto &slot =
        peer(a, destination, r.return_slots)[index * a.local_world_size + rail];
    r.return_tasks[task] = {peer(a, destination, r.rows) + block * m,
                            a.static_network.ring_region +
                                a.static_network.rings.return_payload(0U, 0U) +
                                std::uint64_t(index) * m * a.hidden * 2U,
                            &slot.published,
                            &slot.consumed,
                            desc.return_generation * tiles,
                            desc.tile.rows,
                            desc.tile.rows - desc.local_rows};
  }
  grid_sync<0U, Threads>(a);
}

} // namespace weftmoe_pf::ring
