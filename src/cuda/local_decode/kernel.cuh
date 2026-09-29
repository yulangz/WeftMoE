#pragma once

// Single-node decode follows MegaMoE's fixed pull/TMA/math dataflow, using the
// existing Weft numerical primitives and IPC allocation. Long-context and
// network scheduling remain in static_v4; no dynamic task queue is shared.
#include "cuda/local_decode/schedule.cuh"
#include "cuda/local_decode/compute.cuh"

namespace local_decode {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

__device__ __forceinline__ void geometry(const ForwardKernelArgs &a, Shared &s) {
  const auto lane = threadIdx.x % 32U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  bool error = false;
  if (lane < a.world)
    error = static_v4::ring::peer(a, lane, a.ring.source_header)->error != 0U;
  error = __any_sync(0xffffffffU, error);
  unsigned total_rows = 0U, total_blocks = 0U;
  for (unsigned base = 0U; base < owned; base += 32U) {
    const auto expert = base + lane;
    unsigned rows = 0U;
    if (expert < owned && !error)
      for (unsigned source = 0U; source < a.world; ++source)
        rows += a.fused_local_expert_count[std::uint64_t(source) * owned + expert];
    const auto blocks = (rows + kM - 1U) / kM;
    const auto prefix = static_forward::warp_inclusive_sum(blocks);
    if (expert < owned) {
      s.rows[expert] = rows;
      s.blocks[expert] = total_blocks + prefix - blocks;
    }
    total_rows += __reduce_add_sync(0xffffffffU, rows);
    total_blocks += __shfl_sync(0xffffffffU, prefix, 31U);
  }
  if (lane == 31U)
    s.blocks[owned] = total_blocks;
  if (blockIdx.x == 0U && lane == 0U) {
    auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    status->error = error;
    status->route_exchange_complete_clock = device_globaltimer_ns();
    status->dispatch_complete_clock = status->route_exchange_complete_clock;
    auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
    runtime->error = error;
    runtime->sizes = a.dispatch.sizes;
    runtime->sizes.incoming_row_count = total_rows;
    runtime->sizes.mtile_count = total_blocks;
  }
}

__device__ __forceinline__ void pull(const ForwardKernelArgs &a, Shared &s) {
  const auto warp = (threadIdx.x - 288U) / 32U;
  const auto lane = threadIdx.x % 32U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  unsigned expert = 0U, begin = 0U, phase = 0U, peer_rows = 0U, pulled = 0U;
  for (unsigned linear = blockIdx.x * 3U + warp;; linear += gridDim.x * 3U) {
    while (expert < owned && linear >= begin + s.rows[expert])
      begin += s.rows[expert++];
    if (expert == owned)
      break;
    unsigned source, ordinal;
    static_v4::decode_source_row(a, expert, linear - begin, source, ordinal);
    if (lane == 0U) {
      const auto identity = static_v4::source_routes<static_forward::SourceRoute>(
          a, a.rank, source, expert)[ordinal];
      const auto row = s.blocks[expert] * kM + linear - begin;
      const auto token = identity.route / a.dispatch.identity.top_k;
      const auto *input = peer_at<const char>(a.peer_bases[source],
                                              a.source_input_offset) +
                          std::uint64_t(token) * a.hidden * 2U;
      auto *output = peer_at<char>(a.local_base, a.canonical_rows_offset) +
                     std::uint64_t(row) * a.hidden * 2U;
      a.ring.rows[row] = {
          identity.route, source, identity.gate, token,
          peer_at<transport::BFloat16>(a.peer_bases[source],
                                       a.contribution_offset) +
              std::uint64_t(identity.route) * a.hidden};
      for (unsigned offset = 0U; offset < a.hidden * 2U; offset += 4096U) {
        const auto bytes = min(4096U, static_cast<unsigned>(a.hidden * 2U) - offset);
        cute::SM90_BULK_COPY_G2S::copy(input + offset,
            reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]),
            s.pull[warp], bytes);
        s.pull_full[warp].arrive_and_expect_tx(bytes);
        s.pull_full[warp].wait(phase);
        phase ^= 1U;
        cute::SM90_BULK_COPY_S2G::copy(s.pull[warp], output + offset, bytes);
        cute::tma_store_arrive();
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
      static_v4::reduce_add_release_device(&states(a)[row / kM].x_ready, 1ULL);
      peer_rows += source != a.rank;
      ++pulled;
    }
    __syncwarp();
  }
  if (lane == 0U && pulled) {
    auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
    if (peer_rows)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                std::uint64_t(peer_rows) * a.hidden * 2ULL);
  }
}

__device__ __forceinline__ void combine(const ForwardKernelArgs &a) {
  const auto *input = peer_at<const __nv_bfloat162>(a.local_base,
                                                   a.contribution_offset);
  auto *output = reinterpret_cast<__nv_bfloat162 *>(a.output);
  const auto pairs = a.hidden / 2U;
  const auto top_k = a.dispatch.identity.top_k;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  const bool active = std::uint64_t(blockIdx.x) * kMathThreads <
                      a.dispatch.identity.num_tokens * pairs;
  if (threadIdx.x == 0U && active)
    atomicMin(&status->reduced_tokens, device_globaltimer_ns());
  for (auto v = std::uint64_t(blockIdx.x) * kMathThreads + threadIdx.x;
       v < a.dispatch.identity.num_tokens * pairs;
       v += std::uint64_t(gridDim.x) * kMathThreads) {
    const auto token = v / pairs, column = v % pairs;
    float2 sum = {0.0F, 0.0F};
    for (unsigned position = 0U; position < top_k; ++position) {
      const auto slot = a.dispatch.combine_order[token * top_k + position];
      const auto value = __bfloat1622float2(
          input[(token * top_k + slot) * pairs + column]);
      sum.x = __fadd_rn(sum.x, value.x);
      sum.y = __fadd_rn(sum.y, value.y);
    }
    output[v] = __float22bfloat162_rn(sum);
  }
  static_v4::collective_scope_sync<kMathThreads, 7U>();
  if (threadIdx.x == 0U && active)
    atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
}

#endif

__global__ __launch_bounds__(kThreads, 1) void kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<Shared *>(storage);
  programmatic_launch::wait();
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *status = {};
    status->start_clock = device_globaltimer_ns();
    status->route_exchange_start_clock = status->start_clock;
    status->fc1_start_clock = ~0ULL;
    // V4 reuses these legacy counter fields for FC2, Return and Reduce starts.
    status->dispatch_blocks_done = ~0ULL;
    status->relay_blocks_done = ~0ULL;
    status->reduced_tokens = ~0ULL;
    a.ring.source_header->error = 0U;
    if (a.runtime_epoch)
      atomicAdd(a.runtime_epoch, 1ULL);
  }
  if (threadIdx.x == kMathThreads) {
    for (unsigned stage = 0U; stage < kStages; ++stage) {
      s.full[stage].init(1U);
      s.empty[stage].init(16U);
    }
    for (auto &barrier : s.pull_full)
      barrier.init(1U);
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(&x_map);
    cute::prefetch_tma_descriptor(&s_map);
    cute::prefetch_tma_descriptor(&w1_map);
    cute::prefetch_tma_descriptor(&w2_map);
    cute::prefetch_tma_descriptor(&output_map);
  }
  cute::cluster_sync();
  if (threadIdx.x < kMathThreads) {
    const auto tid = std::uint64_t(blockIdx.x) * kMathThreads + threadIdx.x;
    const auto stride = std::uint64_t(gridDim.x) * kMathThreads;
    static_v4::ring::stage_input(a, tid, stride);
    static_v4::build_combine_order(a, tid, stride);
    for (auto block = tid; block < a.ring.depth / 2U; block += stride)
      states(a)[block] = {};
  } else if (blockIdx.x == 0U) {
    static_v4::ring::build_single_window<128U, kMathThreads>(a, s.counts);
  }
  // CTA0's source builder uses the control-only collective barrier. Math
  // must not enter the all-thread grid collective until that phase retires.
  __syncthreads();
  static_v4::rank_barrier<0U, kThreads>(a);
  if (threadIdx.x >= kMathThreads && threadIdx.x < 288U)
    geometry(a, s);
  __syncthreads();
  forward_compute_detail::observe_compute_cluster(a);
  if (threadIdx.x < kMathThreads) {
    compute(a, s, output_map);
    static_v4::before_combine(a);
  } else if (threadIdx.x < 288U) {
    produce(a, s, x_map, s_map, w1_map, w2_map);
  } else {
    pull(a, s);
  }
  static_v4::collective_scope_sync<kThreads, static_v4::kCollectiveJoinBarrier>();
  if (threadIdx.x < kMathThreads) {
    if (!status->error)
      combine(a);
  } else {
    static_v4::after_combine(a);
    if (blockIdx.x == 0U && threadIdx.x == kMathThreads) {
      const auto blocks = s.blocks[a.dispatch.sizes.owned_expert_count];
      status->fc1_tasks_done = blocks * kFc1Tiles;
      status->fc2_tasks_done = blocks * kFc2Tiles;
      status->actual_return_peer_bytes = status->actual_dispatch_peer_bytes;
      if (!blocks) {
        status->fc1_start_clock = status->route_exchange_complete_clock;
        status->fc1_complete_clock = status->fc1_start_clock;
        status->dispatch_blocks_done = status->fc1_start_clock;
        status->relay_blocks_done = status->fc1_start_clock;
        status->fc2_complete_clock = status->fc1_start_clock;
      }
      if (a.injected_status_error_rank == static_cast<int>(a.rank))
        publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
      status->complete_clock = device_globaltimer_ns();
    }
  }
#endif
}

} // namespace local_decode
