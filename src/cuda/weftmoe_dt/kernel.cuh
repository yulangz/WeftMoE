#pragma once

// Single-node decode follows MegaMoE's fixed pull/TMA/math dataflow, using the
// existing Weft numerical primitives and IPC allocation. Long-context and
// network scheduling remain in weftmoe_pf; no dynamic task queue is shared.
#include "cuda/weftmoe_dt/compute.cuh"
#include "cuda/weftmoe_dt/fp8_compute.cuh"

namespace weftmoe_dt {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

template <bool FP8>
__device__ __forceinline__ void geometry(const ForwardKernelArgs &a,
                                         Shared<FP8> &s) {
  const auto lane = threadIdx.x % 32U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  bool error = false;
  if (lane < a.world)
    error = weftmoe_pf::ring::peer(a, lane, a.ring.source_header)->error != 0U;
  error = __any_sync(0xffffffffU, error);
  unsigned total_rows = 0U, total_blocks = 0U, total_slices = 0U;
  for (unsigned base = 0U; base < owned; base += 32U) {
    const auto expert = base + lane;
    unsigned rows = 0U;
    if (expert < owned && !error)
      for (unsigned source = 0U; source < a.world; ++source)
        rows +=
            a.fused_local_expert_count[std::uint64_t(source) * owned + expert];
    const auto blocks = (rows + kBlockM - 1U) / kBlockM;
    const auto prefix = static_forward::warp_inclusive_sum(blocks);
    if (expert < owned) {
      s.rows()[expert] = rows;
      s.blocks(owned)[expert] = total_blocks + prefix - blocks;
    }
    if constexpr (kSliceSchedule<FP8>) {
      const auto slices = (rows + kM - 1U) / kM;
      const auto slice_prefix = static_forward::warp_inclusive_sum(slices);
      if (expert < owned)
        s.slices(owned)[expert] = total_slices + slice_prefix - slices;
      total_slices += __shfl_sync(0xffffffffU, slice_prefix, 31U);
    }
    total_rows += __reduce_add_sync(0xffffffffU, rows);
    total_blocks += __shfl_sync(0xffffffffU, prefix, 31U);
  }
  if (lane == 31U) {
    s.blocks(owned)[owned] = total_blocks;
    if constexpr (kSliceSchedule<FP8>)
      s.slices(owned)[owned] = total_slices;
  }
  if (blockIdx.x == 0U && lane == 0U) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
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

template <bool FP8>
__device__ __forceinline__ void pull(const ForwardKernelArgs &a,
                                     Shared<FP8> &s) {
  constexpr auto kMathThreads = kComputeThreads<FP8>;
  const auto warp = (threadIdx.x - kMathThreads - 32U) / 32U;
  const auto lane = threadIdx.x % 32U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  unsigned expert = 0U, begin = 0U, phase = 0U, peer_rows = 0U, pulled = 0U;
  for (unsigned linear = blockIdx.x * 3U + warp;; linear += gridDim.x * 3U) {
    while (expert < owned && linear >= begin + s.rows()[expert])
      begin += s.rows()[expert++];
    if (expert == owned)
      break;
    unsigned source, ordinal;
    weftmoe_pf::decode_source_row(a, expert, linear - begin, source, ordinal);
    if constexpr (FP8) {
      const auto identity =
          weftmoe_pf::source_routes<static_forward::SourceRoute>(
              a, a.rank, source, expert)[ordinal];
      const unsigned row = s.blocks(owned)[expert] * kBlockM + linear - begin;
      const auto token = identity.route / a.dispatch.identity.top_k;
      const auto *input =
          peer_at<const std::uint8_t>(a.peer_bases[source],
                                      a.source_input_offset) +
          std::uint64_t(token) * fp8::source_pitch(unsigned(a.hidden));
      auto *base = peer_at<std::uint8_t>(a.local_base, a.canonical_rows_offset);
      auto *output =
          fp8::ring_row(base, row, unsigned(a.hidden), unsigned(a.fp8_x_pitch));
      for (unsigned col = lane * 16U; col < a.hidden; col += 512U)
        *reinterpret_cast<uint4 *>(output + col) =
            *reinterpret_cast<const uint4 *>(input + col);
      auto *scales = fp8::ring_scales(base, row, unsigned(a.hidden),
                                      unsigned(a.fp8_x_pitch));
      for (unsigned col = lane; col < a.hidden / 128U; col += 32U)
        scales[col * 128U] =
            reinterpret_cast<const float *>(input + a.hidden)[col];
      if (!lane)
        a.ring.rows[row] = {identity.route, source, identity.gate, token,
                            peer_at<transport::BFloat16>(
                                a.peer_bases[source], a.contribution_offset) +
                                std::uint64_t(identity.route) * a.hidden};
      __threadfence();
      __syncwarp();
      if (!lane) {
        weftmoe_pf::reduce_add_release_device(&states(a)[row / kBlockM].x_ready,
                                              1ULL);
        peer_rows += source != a.rank;
        ++pulled;
      }
    } else {
      if (lane == 0U) {
        const auto identity =
            weftmoe_pf::source_routes<static_forward::SourceRoute>(
                a, a.rank, source, expert)[ordinal];
        const auto row = s.blocks(owned)[expert] * kBlockM + linear - begin;
        const auto token = identity.route / a.dispatch.identity.top_k;
        const auto *input =
            peer_at<const char>(a.peer_bases[source], a.source_input_offset) +
            std::uint64_t(token) * a.hidden * 2U;
        auto *output = peer_at<char>(a.local_base, a.canonical_rows_offset) +
                       std::uint64_t(row) * a.hidden * 2U;
        a.ring.rows[row] = {identity.route, source, identity.gate, token,
                            peer_at<transport::BFloat16>(
                                a.peer_bases[source], a.contribution_offset) +
                                std::uint64_t(identity.route) * a.hidden};
        for (unsigned offset = 0U; offset < a.hidden * 2U; offset += 4096U) {
          const auto bytes =
              min(4096U, static_cast<unsigned>(a.hidden * 2U) - offset);
          cute::SM90_BULK_COPY_G2S::copy(
              input + offset,
              reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]),
              s.pull[warp], bytes);
          s.pull_full[warp].arrive_and_expect_tx(bytes);
          s.pull_full[warp].wait(phase);
          phase ^= 1U;
          cute::SM90_BULK_COPY_S2G::copy(s.pull[warp], output + offset, bytes);
          cute::tma_store_arrive();
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        weftmoe_pf::reduce_add_release_device(&states(a)[row / kBlockM].x_ready,
                                              1ULL);
        peer_rows += source != a.rank;
        ++pulled;
      }
    }
    __syncwarp();
  }
  if (lane == 0U && pulled) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
    if (peer_rows) {
      atomicAdd(
          &status->actual_dispatch_peer_bytes,
          std::uint64_t(peer_rows) *
              (FP8 ? fp8::source_pitch(unsigned(a.hidden)) : a.hidden * 2ULL));
      if constexpr (FP8)
        atomicAdd(&status->actual_return_peer_bytes,
                  std::uint64_t(peer_rows) * a.hidden * 2ULL);
    }
  }
}

template <unsigned kMathThreads>
__device__ __forceinline__ void combine(const ForwardKernelArgs &a) {
  const auto *input =
      peer_at<const __nv_bfloat162>(a.local_base, a.contribution_offset);
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
      const auto value =
          __bfloat1622float2(input[(token * top_k + slot) * pairs + column]);
      sum.x = __fadd_rn(sum.x, value.x);
      sum.y = __fadd_rn(sum.y, value.y);
    }
    output[v] = __float22bfloat162_rn(sum);
  }
  weftmoe_pf::collective_scope_sync<kMathThreads, 7U>();
  if (threadIdx.x == 0U && active) {
    const auto now = device_globaltimer_ns();
    atomicMax(&status->reduce_blocks_done, now);
    atomicMax(&status->complete_clock, now);
  }
}

#endif

template <bool FP8 = false>
__global__ __launch_bounds__(kKernelThreads<FP8>, FP8 ? 1 : 2) void kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  constexpr auto kMathThreads = kComputeThreads<FP8>;
  constexpr auto kThreads = kKernelThreads<FP8>;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<Shared<FP8> *>(storage);
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
    for (unsigned stage = 0U; stage < stages<FP8>; ++stage) {
      s.full[stage].init(1U);
      s.empty[stage].init(kMathThreads / 128U);
    }
    for (auto &barrier : s.pull_full)
      barrier.init(1U);
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(&x_map);
    cute::prefetch_tma_descriptor(&s_map);
    cute::prefetch_tma_descriptor(&w1_map);
    cute::prefetch_tma_descriptor(&w2_map);
  }
  cute::cluster_sync();
  if (threadIdx.x < kMathThreads) {
    const auto tid = std::uint64_t(blockIdx.x) * kMathThreads + threadIdx.x;
    const auto stride = std::uint64_t(gridDim.x) * kMathThreads;
    if constexpr (FP8)
      stage_fp8_source<kMathThreads>(a);
    else
      weftmoe_pf::ring::stage_input(a, tid, stride);
    weftmoe_pf::build_combine_order(a, tid, stride);
    for (auto block = tid; block < a.ring.depth / 2U; block += stride)
      states(a)[block] = {};
  } else if (blockIdx.x == 0U) {
    weftmoe_pf::ring::build_single_window<128U, kMathThreads>(a, s.rows());
  }
  // CTA0's source builder uses the control-only collective barrier. Math
  // must not enter the all-thread grid collective until that phase retires.
  __syncthreads();
  weftmoe_pf::rank_barrier<0U, kThreads>(a);
  if (threadIdx.x >= kMathThreads && threadIdx.x < kMathThreads + 32U)
    geometry(a, s);
  if constexpr (FP8) {
    if (threadIdx.x >= kMathThreads)
      cutlass::arch::warpgroup_reg_dealloc<fp8::kControlRegisters>();
  }
  __syncthreads();
  forward_compute_detail::observe_compute_cluster(a);
  if (threadIdx.x < kMathThreads) {
    if constexpr (FP8) {
      cutlass::arch::warpgroup_reg_alloc<fp8::kMathRegisters>();
      compute_fp8(a, s);
    } else
      compute(a, s);
    weftmoe_pf::before_combine<kMathThreads>(a);
  } else if (threadIdx.x < kMathThreads + 32U) {
    if constexpr (FP8)
      produce_fp8(a, s, x_map, s_map, w1_map, w2_map);
    else
      produce(a, s, x_map, s_map, w1_map, w2_map);
  } else {
    pull(a, s);
  }
  weftmoe_pf::collective_scope_sync<kThreads,
                                    weftmoe_pf::kCollectiveJoinBarrier>();
  if (threadIdx.x < kMathThreads) {
    if (!status->error)
      combine<kMathThreads>(a);
  } else {
    // No terminal cleanup: BeforeCombine retired every source/route reader.
    // The next invocation's BeforePull waits for all peers to finish this
    // kernel before any FC2 writer can reuse their contribution storage.
    if (blockIdx.x == 0U && threadIdx.x == kMathThreads) {
      const auto owned = a.dispatch.sizes.owned_expert_count;
      const auto blocks = s.blocks(owned)[owned];
      status->fc1_tasks_done =
          compute_units(s, owned) * (FP8 ? kFc1Tiles / 2U : kFc1Tiles);
      status->fc2_tasks_done =
          compute_units(s, owned) * (FP8 ? kFc2Tiles / 2U : kFc2Tiles);
      if constexpr (!FP8)
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
      atomicMax(&status->complete_clock, device_globaltimer_ns());
    }
  }
#endif
}

} // namespace weftmoe_dt
