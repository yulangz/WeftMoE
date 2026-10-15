#pragma once

namespace weftmoe_pf::ring {

// Size a rail slot from the exact maximum fanout to one rail, not all M rows.
// The arena remains bounded; all peers derive the same static expert partition.
template <unsigned Threads>
__device__ __noinline__ void prepare_return(const ForwardKernelArgs &a) {
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto local_rank = a.rank % a.local_world_size;
  const auto experts = owned * a.local_world_size;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + threadIdx.x;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  const auto lane = threadIdx.x % 32U;
  constexpr auto m = forward_tuning::kTileM;
  constexpr auto tiles =
      forward_tuning::kMainHidden / forward_tuning::kFc1TileN;
  for (auto e = worker; e < owned; e += workers) {
    const auto rows = r.state->windows ? r.window_compute[e].total_rows : 0U;
    r.return_experts[local_rank * owned + e] = {
        0U, unsigned(static_forward::ceil_div(rows, m)), 1U};
  }
  grid_sync<0U, Threads>(a);
  for (auto block = worker / 32U; block < r.state->blocks;
       block += workers / 32U) {
    unsigned rail_rows = 0U, local_rows = 0U;
    for (unsigned first = 0U; first < r.blocks[block].tile.rows; first += 32U) {
      const auto row = first + lane;
      const bool valid = row < r.blocks[block].tile.rows;
      const auto source = valid ? r.rows[block * m + row].source : 0U;
      const bool remote = valid && source / a.local_world_size != a.node;
      local_rows += __popc(__ballot_sync(0xffffffffU, valid && !remote));
      for (unsigned rail = 0U; rail < a.local_world_size; ++rail) {
        const auto mask = __ballot_sync(
            0xffffffffU, remote && source % a.local_world_size == rail);
        if (lane == rail)
          rail_rows += __popc(mask);
      }
    }
    const auto stride = __reduce_max_sync(0xffffffffU, rail_rows);
    if (!lane) {
      r.blocks[block].local_rows = local_rows;
      atomicMax(
          &r.return_experts[local_rank * owned + r.blocks[block].tile.expert]
               .stride,
          stride);
    }
  }
  rank_barrier<0U, Threads, true, true, true>(a);
  for (auto e = worker; e < experts; e += workers) {
    const auto destination = a.node * a.local_world_size + e / owned;
    if (destination != a.rank)
      r.return_experts[e] = peer(a, destination, r.return_experts)[e];
  }
  // Every peer must finish reading the count/stride snapshot before an owner
  // replaces it with its capacity partition.
  rank_barrier<0U, Threads, true, true, true>(a);
  if (!worker) {
    std::uint64_t demand = 0U, minimum = 0U;
    for (unsigned e = 0U; e < experts; ++e) {
      const auto plan = r.return_experts[e];
      demand += std::uint64_t(plan.depth) * plan.stride;
      minimum += plan.depth ? plan.stride : 0U;
    }
    unsigned rows = 0U;
    for (unsigned e = 0U; e < experts; ++e) {
      auto &plan = r.return_experts[e];
      const auto blocks = plan.depth;
      const auto depth =
          demand <= r.return_capacity
              ? blocks
              : (blocks ? 1U + (r.return_capacity - minimum) * (blocks - 1U) /
                                   (demand - minimum)
                        : 0U);
      plan.begin = rows;
      plan.depth = unsigned(depth);
      rows += plan.depth * plan.stride;
    }
  }
  grid_sync<0U, Threads>(a);
  for (auto block = worker / 32U; block < r.state->blocks;
       block += workers / 32U) {
    auto &desc = r.blocks[block];
    const auto plan = r.return_experts[local_rank * owned + desc.tile.expert];
    const auto ordinal = desc.tile.expert_row / m;
    const auto slot = plan.begin + (ordinal % plan.depth) * plan.stride;
    if (!lane)
      desc.return_generation =
          r.return_slots[slot].published / tiles + ordinal / plan.depth + 1U;
    unsigned previous = 0U; // Lane i owns the prefix for source rail i.
    for (unsigned first = 0U; first < desc.tile.rows; first += 32U) {
      const auto row = first + lane;
      const bool valid = row < desc.tile.rows;
      const auto source = valid ? r.rows[block * m + row].source : 0U;
      const bool remote = valid && source / a.local_world_size != a.node;
      const auto rail = source % a.local_world_size;
      const auto matching = __match_any_sync(0xffffffffU, remote ? rail : ~0U);
      const auto packed = __shfl_sync(0xffffffffU, previous, rail) +
                          __popc(matching & ((1U << lane) - 1U));
      if (remote)
        r.rows[block * m + row].output = ncclGetLsaPointer(
            a.gin_window,
            a.static_network.ring_region +
                a.static_network.rings.return_payload(0U, 0U) +
                std::uint64_t(slot + packed) * a.hidden * 2U,
            rail);
      for (unsigned destination = 0U; destination < a.local_world_size;
           ++destination) {
        const auto mask =
            __ballot_sync(0xffffffffU, remote && rail == destination);
        if (lane == destination)
          previous += __popc(mask);
      }
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
    const auto destination =
        a.node * a.local_world_size + task % a.local_world_size;
    const auto block = task / a.local_world_size;
    if (block >= peer(a, destination, r.state)->blocks) {
      if (!lane)
        r.return_tasks[task].count = 0U;
      continue;
    }
    const auto desc = peer(a, destination, r.blocks)[block];
    const auto *rows = peer(a, destination, r.rows) + block * m;
    unsigned count = 0U;
    for (unsigned row = lane; row < desc.tile.rows; row += 32U) {
      const auto source = rows[row].source;
      count += source / a.local_world_size != a.node &&
               source % a.local_world_size == local_rank;
    }
    count = __reduce_add_sync(0xffffffffU, count);
    if (!lane) {
      const auto plan =
          r.return_experts[destination % a.local_world_size * owned +
                           desc.tile.expert];
      const auto index =
          plan.begin + (desc.tile.expert_row / m % plan.depth) * plan.stride;
      auto &slot = peer(a, destination, r.return_slots)[index];
      r.return_tasks[task] = {
          rows,
          a.static_network.ring_region +
              a.static_network.rings.return_payload(0U, 0U) +
              std::uint64_t(index) * a.hidden * 2U,
          &slot.published,
          &slot.consumed,
          desc.return_generation * tiles,
          count ? desc.tile.rows : 0U};
    }
  }
  grid_sync<0U, Threads>(a);
}

} // namespace weftmoe_pf::ring
