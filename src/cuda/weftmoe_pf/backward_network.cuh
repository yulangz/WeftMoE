#pragma once

namespace weftmoe_pf::backward_stream {

// The immutable Forward window is also the admission boundary for dY/dAB/dX.
// Group all its experts into each phase, preserving paired N-tile ownership.
struct NetworkSchedule {
  const ForwardKernelArgs &a;
  unsigned window = 0U, index = blockIdx.x;

  __device__ explicit NetworkSchedule(const ForwardKernelArgs &args)
      : a(args) {}

  __device__ __forceinline__ bool next(Task &t) {
    while (window < a.ring.state->windows) {
      const auto begin = unsigned(a.ring.window_prefix[window]);
      const auto blocks = unsigned(a.ring.window_prefix[window + 1U]) - begin;
      const auto du = blocks * kDuTiles, epilogue = blocks * 2U;
      const auto total = du + epilogue + blocks * kDxTiles;
      if (index < total) {
        const auto phase = index < du              ? Phase::Du
                           : index < du + epilogue ? Phase::Swiglu
                                                   : Phase::Dx;
        const auto linear = index - (phase == Phase::Du       ? 0U
                                     : phase == Phase::Swiglu ? du
                                                              : du + epilogue);
        const auto columns = phase == Phase::Du   ? kDuTiles
                             : phase == Phase::Dx ? kDxTiles
                                                  : 2U;
        const auto block = begin + linear / columns;
        const auto &desc = a.ring.blocks[block];
        t = {desc.tile.expert,
             block,
             linear % columns,
             desc.tile.rows,
             desc.slot * kM,
             unsigned(desc.generation),
             phase};
        index += gridDim.x;
        return true;
      }
      index -= total;
      ++window;
    }
    return false;
  }
};

// Remains live during dW: RX forwarding and outstanding NIC requests must
// never alias the accumulator epilogue or either GEMM operand pipeline.
struct alignas(16) NetworkShared {
  unsigned char pull[3][4096];
  local::Barrier pull_full[3];
};
constexpr auto kNetworkCommOffset = (kSharedBytes + 15U) & ~std::size_t{15U};
constexpr auto kNetworkSharedBytes = kNetworkCommOffset + sizeof(NetworkShared);

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
__device__ __forceinline__ float *
gradient_output(const ForwardKernelArgs &f,
                const static_forward::RingRow &row) {
  const bool local = row.source / f.local_world_size == f.node;
  const auto &n = f.static_network;
  const auto payload =
      local ? n.contribution : n.ring_region + n.rings.return_payload(0U, 0U);
  const auto scalar = local ? n.gradient_contribution : n.gradient_tx;
  const auto *base = static_cast<const char *>(ncclGetLsaPointer(
      f.gin_window, payload, row.source % f.local_world_size));
  const auto index =
      (static_cast<const char *>(row.output) - base) / (f.hidden * 2U);
  return static_cast<float *>(ncclGetLsaPointer(
             f.gin_window, scalar, row.source % f.local_world_size)) +
         index;
}
#endif

// Rebind only mutable transport addresses. Source routes, packet membership,
// fanout counts and saved-row mappings come directly from the Forward tape.
__global__ void prepare_network(ForwardKernelArgs f) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  constexpr unsigned Threads = 256U;
  const auto tid = std::uint64_t(blockIdx.x) * Threads + threadIdx.x;
  const auto stride = std::uint64_t(gridDim.x) * Threads;
  const auto &r = f.ring;
  const auto &n = f.static_network;
  if (!tid) {
    r.state->admitted_windows = 0U;
  }
  // Advance payload generations without re-exchanging immutable routing.
  // Every destination already knows every source's exact window count.
  for (auto source = tid; source < f.world; source += stride)
    r.source_epochs[source].base += static_forward::ceil_div(
        r.source_epochs[source].tokens, r.source.shape.tokens);
  for (auto block = tid; block < r.state->blocks; block += stride) {
    auto &desc = r.blocks[block];
    const auto sequence = r.state->block_base + block;
    desc.slot = unsigned(sequence % r.depth);
    desc.generation = sequence / r.depth + 1U;
  }
  grid_sync<0U, Threads>(f);
  if (!tid)
    r.state->source_base = r.source_epochs[f.rank].base;
  for (auto i = tid; i < r.depth; i += stride) {
    if (!r.state->blocks)
      break;
    const auto first = (i + r.depth - r.blocks[0].slot) % r.depth;
    if (first >= r.state->blocks)
      continue;
    const auto prior = r.blocks[first].generation - 1U;
    auto &s = r.slots[i];
    s = {};
    s.x_full = prior * kM;
    s.fc1_done = s.fc1_target = prior * kDuTiles;
    s.fc2_done = s.fc2_target = s.fc2_published = prior * kDxTiles;
    s.activation_ready = prior * 2U;
  }
  for (auto i = tid; i < std::uint64_t(f.local_world_size) * r.depth;
       i += stride)
    r.return_ready[i] = 0U;
  for (auto i = tid; i < 2U * n.rings.channels(); i += stride) {
    auto &s = *peer_at<static_forward::NetworkChannel>(
        f.gin_window_base,
        n.ring_region + i * sizeof(static_forward::NetworkChannel));
    s.tx_cursor = s.tx_reserve_cursor = s.tx_reserved_head =
        s.tx_completed_head = s.tx_base;
    s.rx_cursor = s.consume_cursor = s.acknowledged_head = s.rx_base;
  }
  // Publish dY staging and initialization before any peer probes RX/TX state.
  network_barrier<0U, Threads>(f, 1U);
  for (auto task = tid; task < r.state->windows * r.source.shape.sources;
       task += stride) {
    const auto window = task / r.source.shape.sources;
    const auto source =
        r.source.shape.source_begin(window) + task % r.source.shape.sources;
    if (source >= r.source.shape.source_end(window))
      continue;
    const auto sequence =
        r.source_epochs[source].base + window / r.source.shape.groups();
    r.window_sources[task] = {sequence, sequence / r.contribution_depth + 1U,
                              sequence % r.contribution_depth *
                                  r.contribution_stride,
                              unsigned(sequence % r.contribution_depth)};
  }
  for (auto logical = tid; logical < r.state->blocks * kM; logical += stride) {
    const auto &desc = r.blocks[logical / kM];
    if (logical % kM >= desc.tile.rows)
      continue;
    auto &row = r.rows[logical];
    ring::source_metadata(f, row.source, r.locations)[row.route] = {
        desc.generation, logical / kM, desc.slot * kM + unsigned(logical % kM)};
    if (row.source / f.local_world_size == f.node) {
      auto *base = static_cast<char *>(ncclGetLsaPointer(
          f.gin_window, n.contribution, row.source % f.local_world_size));
      const auto ordinal = (static_cast<char *>(row.output) - base) /
                           (f.hidden * 2U) % r.contribution_stride;
      const auto sequence = r.source_epochs[row.source].base +
                            desc.window / r.source.shape.groups();
      row.output =
          base +
          (sequence % r.contribution_depth * r.contribution_stride + ordinal) *
              f.hidden * 2U;
    }
  }
  // Locations are rail-shared: all destinations finish their disjoint writes
  // before another rail decodes a Return packet containing those locations.
  rank_barrier<0U, Threads, true, true, true>(f);
  const auto packets =
      r.source.shape.sources * r.source.packets.packets(r.source.shape.tokens);
  for (auto i = tid; i < r.state->windows * packets; i += stride)
    ring::build_pull_packet(f, i / packets, i % packets);

  // Every source rail owns its own TX channels. Decode those immutable packet
  // rows once, rather than searching expert/source ordinals inside GEMM.
  const auto owned = f.dispatch.sizes.owned_expert_count;
  for (unsigned task = blockIdx.x; task < f.world * n.rings.lanes;
       task += gridDim.x) {
    const auto source = task / n.rings.lanes;
    if (source / f.local_world_size == f.node)
      continue;
    const auto lane = task % n.rings.lanes;
    const auto directory = f.node * n.rings.lanes + lane;
    const auto *offsets =
        ring::source_metadata(f, source, r.returning.channel_offsets);
    const auto *plans = ring::source_metadata(f, source, r.returning.packets);
    const auto *order = ring::source_metadata(f, source, r.source.route_order);
    const auto *experts =
        ring::source_metadata(f, source, r.source.selected_experts);
    const auto *locations = ring::source_metadata(f, source, r.locations);
    const auto base = return_tx::state(f, source, lane).tx_base;
    const auto channel = return_tx::channel(f, source, lane);
    for (auto p = offsets[directory]; p < offsets[directory + 1U]; ++p) {
      const auto &packet = plans[p];
      auto *output = static_cast<char *>(ncclGetLsaPointer(
          f.gin_window,
          n.ring_region +
              n.rings.return_payload(channel, base + p - offsets[directory]),
          source % f.local_world_size));
      for (unsigned row = threadIdx.x; row < packet.rows; row += Threads) {
        const auto route = order[packet.route_begin + row];
        if (unsigned(experts[route]) / owned != f.rank)
          continue;
        const auto location = locations[route];
        r.rows[location.block * kM + location.row % kM].output =
            output + row * f.hidden * 2U;
      }
    }
  }
#endif
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
__device__ __forceinline__ void finish_network(const ForwardKernelArgs &f,
                                               const BackwardKernelArgs &a) {
  constexpr unsigned Base = 288U, Threads = 96U;
  rank_barrier<Base, Threads, true, true, true>(f);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (threadIdx.x == Base) {
    atomicMax(&status->return_complete_clock, device_globaltimer_ns());
    atomicMin(&status->backward_reduce_start_clock, device_globaltimer_ns());
  }
  local::combine<true, Base, Threads>(f);
  const auto *input = peer_at<const float>(
      f.gin_window_base, f.static_network.gradient_contribution);
  for (auto route = std::uint64_t(blockIdx.x) * Threads + threadIdx.x - Base;
       route < f.dispatch.identity.num_tokens * f.dispatch.identity.top_k;
       route += std::uint64_t(gridDim.x) * Threads) {
    const auto token = route / f.dispatch.identity.top_k;
    const auto sequence =
        f.ring.state->source_base + token / f.ring.source.shape.tokens;
    const auto row =
        sequence % f.ring.contribution_depth * f.ring.contribution_stride +
        f.ring.contribution_rows[route];
    a.dgate[route] = input[row];
  }
  if (threadIdx.x == Base)
    atomicMax(&status->backward_reduce_complete_clock, device_globaltimer_ns());
  grid_sync<Base, Threads>(f);
  if (!blockIdx.x && threadIdx.x == Base) {
    for (unsigned i = 0U; i < f.ring.contribution_depth; ++i)
      f.ring.contributions[i].consumed = f.ring.contributions[i].retire_target;
    f.ring.state->source_base += f.ring.source.local_windows();
    f.ring.state->block_base += f.ring.state->blocks;
    // Return the shared slots to Forward's counter units. The absolute block
    // generations advance once for this Backward, just as for Forward.
    for (unsigned i = 0U; i < f.ring.depth && f.ring.state->blocks; ++i) {
      const auto first =
          (i + f.ring.depth - f.ring.blocks[0].slot) % f.ring.depth;
      if (first >= f.ring.state->blocks)
        continue;
      const auto last = first + (f.ring.state->blocks - 1U - first) /
                                    f.ring.depth * f.ring.depth;
      const auto gen = f.ring.blocks[last].generation;
      auto &s = f.ring.slots[i];
      s.fc1_done = s.fc1_target = gen * local::kFc1Tiles;
      s.fc2_done = s.fc2_target = s.fc2_published = gen * local::kFc2Tiles;
    }
    for (unsigned channel = 0U; channel < f.static_network.rings.channels();
         ++channel)
      for (unsigned direction = 0U; direction < 2U; ++direction) {
        auto &s = *peer_at<static_forward::NetworkChannel>(
            f.gin_window_base,
            f.static_network.ring_region +
                f.static_network.rings.state(direction, channel));
        s.tx_base = s.tx_cursor;
        s.rx_base = s.rx_cursor;
      }
  }
  network_barrier<Base, Threads>(f, 1U);
}
#endif

} // namespace weftmoe_pf::backward_stream
