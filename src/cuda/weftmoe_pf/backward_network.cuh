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

// Notify rebuilds the current tape's physical plan in shared execution storage.
// Its slot generations are in Forward tile units; convert only the slots used
// by this Backward before any producer or communication worker runs.
__global__ void prepare_network(ForwardKernelArgs f) {
  const auto &r = f.ring;
  const auto tid = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = std::uint64_t(gridDim.x) * blockDim.x;
  for (auto i = tid; i < r.depth && r.state->blocks; i += stride) {
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
}
#endif

} // namespace weftmoe_pf::backward_stream
