#pragma once

namespace static_v4 {

// Context i has exactly one CTA owner on every rail. Each context performs
// its own Put fence: a signal on context zero cannot order another QP's puts.
// The node-local barrier then imports visibility from every receiving rail.
template <unsigned BaseThread, unsigned Threads>
__device__ __forceinline__ void network_barrier(const ForwardKernelArgs &a) {
  grid_sync<BaseThread, Threads>(a);
  if (blockIdx.x < kNetworkContexts && threadIdx.x == BaseThread) {
    const ncclGin gin(a.gin_dev_comm, blockIdx.x);
    ncclGinBarrier(ncclCoopThread{}, gin, ncclTeamTagRail{}, 0U,
                   ::cuda::memory_order_acq_rel, ncclGinFenceLevel::Put);
  }
  rank_barrier<BaseThread, Threads, true, true, true>(a);
}

__device__ __forceinline__ void
network_put(const ForwardKernelArgs &a, unsigned destination_node,
            std::size_t destination, std::size_t source, std::size_t bytes) {
  const ncclGin gin(a.gin_dev_comm, blockIdx.x);
  const auto peer =
      destination_node * a.local_world_size + a.rank % a.local_world_size;
  for (std::size_t offset = blockIdx.x * kNetworkChunkBytes; offset < bytes;
       offset += kNetworkContexts * kNetworkChunkBytes) {
    const auto extent = min(kNetworkChunkBytes, bytes - offset);
    gin.put(ncclTeamWorld(gin.comm), peer, a.gin_window, destination + offset,
            a.gin_window, source + offset, extent, ncclGin_None{},
            ncclGin_None{}, ncclCoopThread{});
  }
}

__device__ __noinline__ void
network_before_combine(const ForwardKernelArgs &a) {
  constexpr auto kThreads = forward_tuning::kMathWarpgroups * 128U;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (blockIdx.x == 0U && threadIdx.x == 0U)
    status->before_combine_start_clock = device_globaltimer_ns();
  // FC2 on every local GPU has finished writing the rail's TX arena.
  rank_barrier<0U, kThreads, true, true, true>(a);
  if (blockIdx.x < kNetworkContexts && threadIdx.x == 0U) {
    const auto nodes = a.dispatch.identity.num_nodes;
    const auto experts_per_node = a.dispatch.identity.num_experts / nodes;
    for (unsigned node = 0; node < nodes; ++node) {
      if (node == a.node)
        continue;
      const auto source =
          node * a.local_world_size + a.rank % a.local_world_size;
      const auto *prefix = source_prefix(a, source);
      const auto rows = prefix[(a.node + 1U) * experts_per_node] -
                        prefix[a.node * experts_per_node];
      network_put(
          a, node,
          a.static_network.return_rx + a.node * a.static_network.return_stride,
          a.static_network.return_tx + node * a.static_network.return_stride,
          rows * a.hidden * sizeof(transport::BFloat16));
    }
  }
  network_barrier<0U, kThreads>(a);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    const auto now = device_globaltimer_ns();
    status->before_combine_complete_clock = now;
    status->return_complete_clock = now;
  }
}

__device__ __forceinline__ void
network_after_combine(const ForwardKernelArgs &a) {
  constexpr auto kBase = forward_tuning::kMathWarpgroups * 128U;
  constexpr auto kThreads = forward_tuning::kUnifiedThreads - kBase;
  // The entry joins math readers before allowing another invocation to
  // overwrite RX. Every context drains before its arena can be recycled.
  network_barrier<kBase, kThreads>(a);
}

} // namespace static_v4
