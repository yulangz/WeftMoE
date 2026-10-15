#pragma once

#include "cuda/weftmoe_pf/network_schedule.cuh"
#include "cuda/weftmoe_pf/network_workers.cuh"
#include "cuda/weftmoe_pf/network_return.cuh"

namespace weftmoe_pf::network {

constexpr std::size_t kSharedBytes = sizeof(local::Shared);

template <class Storage>
__device__ __forceinline__ void workers(const ForwardKernelArgs &a,
                                        Storage &s) {
  const auto warp = (threadIdx.x - 288U) / 32U;
  const auto role =
      warp == 0U && blockIdx.x < kNetworkOwners ? blockIdx.x : ~0U;
  auto *pending = reinterpret_cast<Pending *>(s.pull[0]);
  if (role < kGetWorkers) {
    dispatch(a, pending, role);
  } else if (role < kGetWorkers + kPutWorkers) {
    return_blocks(a, reinterpret_cast<ReturnPending *>(pending),
                   role - kGetWorkers);
  } else if (role < kNetworkOwners - 1U) {
    receive_blocks(a, role - kGetWorkers - kPutWorkers);
  } else if (role == kNetworkOwners - 1U) {
    ring::admit<true>(a, local::kFc1Tiles, local::kFc2Tiles);
  } else {
    const auto worker = blockIdx.x < kNetworkOwners
                            ? 2U * blockIdx.x + warp - 1U
                            : 3U * blockIdx.x - kNetworkOwners + warp;
    pull(a, s, warp, worker, gridDim.x * 3U - kNetworkOwners);
  }
}

__global__ __launch_bounds__(kThreads, 1) void kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<local::Shared *>(storage);
  programmatic_launch::wait();
  WEFTMOE_TIMELINE_RECORD(a.timeline, MainEnter, Kernel, 0U, 0U, 0U);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  // The standalone planner owns route publication, ring setup and status.
  if (threadIdx.x == 256U) {
    for (unsigned stage = 0U; stage < local::kStages; ++stage) {
      s.full[stage].init(1U);
      s.empty[stage].init(16U);
    }
    for (auto &barrier : s.pull_full)
      barrier.init(1U);

    cutlass::arch::fence_barrier_init();
  }
  if (threadIdx.x >= 256U)
    cutlass::arch::warpgroup_reg_dealloc<
        Resources::kUnifiedProducerRegisters>();
  __syncthreads();
  cute::cluster_sync();
  if (!a.dispatch.runtime->error) {
    if (threadIdx.x < 256U) {
      cutlass::arch::warpgroup_reg_alloc<Resources::kUnifiedMathRegisters>();
      local::compute<Schedule, true, Mma, forward_tuning::kWgmmaWaitGroup>(
          a, s, output_map);
    } else if (threadIdx.x < 288U) {
      local::produce<Schedule, true>(a, s, x_map, s_map, w1_map, w2_map);
    } else {
      workers(a, s);
    }
  }
  // Return RX is source-local, but local experts can still be writing this
  // source's contribution from another GPU. Join every local writer as well.
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, threadIdx.x % 32U == 0U, MainDataDone,
                             Kernel, threadIdx.x / 32U, 0U, 0U);
  rank_barrier<0U, kThreads, true, true, true>(a);
  WEFTMOE_TIMELINE_RECORD(a.timeline, MainJoinDone, Kernel, 0U, 0U, 0U);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    status->relay_blocks_done = status->route_exchange_complete_clock;
    status->return_complete_clock = device_globaltimer_ns();
  }
  WEFTMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U, 0U, 0U);
  if (threadIdx.x < 256U && !a.dispatch.runtime->error)
    local::combine<true>(a);
  WEFTMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U, 0U, 0U);
  grid_sync<0U, kThreads>(a);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    for (unsigned slot = 0U; slot < a.ring.contribution_depth; ++slot)
      a.ring.contributions[slot].consumed =
          a.ring.contributions[slot].retire_target;
    a.ring.state->block_base += a.ring.state->blocks;
    a.ring.state->source_base += a.ring.source.local_windows();
    for (unsigned channel = 0U; channel < a.static_network.rings.channels();
         ++channel)
      for (unsigned returning = 0U; returning < 2U; ++returning) {
        auto &state = *peer_at<static_forward::NetworkChannel>(
            a.gin_window_base,
            a.static_network.ring_region +
                a.static_network.rings.state(returning, channel));
        state.tx_base = state.tx_cursor;
        state.rx_base = state.rx_cursor;
      }
  }
  // All Gets, local RX readers, Return requests and Reduce readers drained.
  // Only invocation metadata/input reuse needs a global rendezvous here.
  network_barrier<0U, kThreads>(a, 1U);
  WEFTMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0U, 0U);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    status->fc1_tasks_done = a.ring.state->blocks * local::kFc1Tiles;
    status->fc2_tasks_done = a.ring.state->blocks * local::kFc2Tiles;
    status->complete_clock = device_globaltimer_ns();
  }
#endif
}

} // namespace weftmoe_pf::network
