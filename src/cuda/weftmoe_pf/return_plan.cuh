#pragma once

namespace weftmoe_pf::ring {

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
    r.return_experts[expert].depth =
        static_forward::ceil_div(rows, r.source.shape.tile_m);
  }
  grid_sync<0U, Threads>(a);
  if (!worker) {
    std::uint64_t total = 0U, active = 0U;
    for (unsigned e = 0U; e < experts; ++e) {
      total += r.return_experts[e].depth;
      active += r.return_experts[e].depth != 0U;
    }
    std::uint64_t prefix = 0U;
    unsigned slots = 0U;
    for (unsigned e = 0U; e < experts; ++e) {
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

template <unsigned Threads>
__device__ __noinline__ void prepare_return(const ForwardKernelArgs &a) {
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + threadIdx.x;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  const auto lane = threadIdx.x % 32U;
  constexpr auto m = forward_tuning::kTileM;
  constexpr auto tiles =
      forward_tuning::kMainHidden / forward_tuning::kFc1TileN;
  if (!worker) {
    std::uint64_t blocks = 0U;
    for (unsigned local = 0U; local < a.local_world_size; ++local)
      blocks =
          max(blocks,
              peer(a, a.node * a.local_world_size + local, r.state)->blocks);
    r.state->return_tasks = blocks * a.local_world_size;
  }
  for (auto block = worker / 32U; block < r.state->blocks;
       block += workers / 32U) {
    unsigned count = 0U;
    for (unsigned row = lane; row < r.blocks[block].tile.rows; row += 32U)
      count += r.rows[block * m + row].source / a.local_world_size == a.node;
    count = __reduce_add_sync(0xffffffffU, count);
    if (!lane)
      r.blocks[block].local_rows = count;
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
               source % a.local_world_size == a.rank % a.local_world_size;
    }
    count = __reduce_add_sync(0xffffffffU, count);
    if (!lane) {
      const auto plan =
          r.return_experts[destination % a.local_world_size * owned +
                           desc.tile.expert];
      const auto index = plan.begin + desc.tile.expert_row / m % plan.depth;
      auto &slot = peer(a, destination, r.return_slots)[index];
      r.return_tasks[task] = {
          rows,
          a.static_network.ring_region +
              a.static_network.rings.return_payload(0U, 0U) +
              std::uint64_t(index) * m * a.hidden * 2U,
          &slot.published,
          &slot.consumed,
          desc.return_generation * tiles,
          count ? desc.tile.rows : 0U};
    }
  }
  grid_sync<0U, Threads>(a);
}

} // namespace weftmoe_pf::ring
