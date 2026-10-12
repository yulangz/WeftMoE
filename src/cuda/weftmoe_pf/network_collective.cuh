#pragma once

namespace weftmoe_pf {

// Context i has exactly one CTA owner on every rail. Each context performs
// its own Put fence: a signal on context zero cannot order another QP's puts.
// The node-local barrier then imports visibility from every receiving rail.
template <unsigned BaseThread, unsigned Threads>
__device__ __forceinline__ void network_barrier(const ForwardKernelArgs &a,
                                                unsigned contexts) {
  grid_sync<BaseThread, Threads>(a);
  if (threadIdx.x == BaseThread)
    for (unsigned context = blockIdx.x; context < contexts;
         context += gridDim.x) {
      const ncclGin gin(a.gin_dev_comm, context);
      ncclGinBarrier(ncclCoopThread{}, gin, ncclTeamTagRail{}, 0U,
                     ::cuda::memory_order_acq_rel, ncclGinFenceLevel::Put);
    }
  rank_barrier<BaseThread, Threads, true, true, true>(a);
}

__device__ __forceinline__ void
network_put(const ForwardKernelArgs &a, unsigned destination_node,
            std::size_t destination, std::size_t source, std::size_t bytes) {
  const auto peer =
      destination_node * a.local_world_size + a.rank % a.local_world_size;
  for (unsigned context = blockIdx.x; context < a.static_network.contexts;
       context += gridDim.x) {
    const ncclGin gin(a.gin_dev_comm, context);
    for (std::size_t offset = context * kNetworkChunkBytes; offset < bytes;
         offset += a.static_network.contexts * kNetworkChunkBytes) {
      const auto extent = min(kNetworkChunkBytes, bytes - offset);
      gin.put(ncclTeamWorld(gin.comm), peer, a.gin_window, destination + offset,
              a.gin_window, source + offset, extent, ncclGin_None{},
              ncclGin_None{}, ncclCoopThread{});
    }
  }
}

} // namespace weftmoe_pf
