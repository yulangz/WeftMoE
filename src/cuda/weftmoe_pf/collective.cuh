#pragma once

namespace weftmoe_pf {

// collective_words()[0] is the reusable grid counter. Word 1 advances the
// two-phase add/sub rank protocol. Words 4 and 5 are its signed signals; words
// 2 and 3 belong to the FC1 completion counter in the enclosing uint64 array.
// Alternating phases prevents a fast rank from reusing a signal still observed
// by a peer, and alternating the sign returns each signal to zero without a
// separate clear kernel.
template <unsigned BaseThread, unsigned Threads, bool ReleaseGrid = true,
          bool EnterGrid = true, bool NodeLocal = false>
__device__ __forceinline__ void rank_barrier(const ForwardKernelArgs &a) {
  // Static V4 kernels run a bounded graph invocation to completion. Keep the
  // cross-rank poll to the phase signal only; V3 retains cancellable waits.
  if constexpr (EnterGrid)
    grid_sync<BaseThread, Threads>(a);

  if (blockIdx.x == 0U) {
    if constexpr (!EnterGrid)
      collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    auto *words = collective_words(a);
    const auto state = words[1] & 3U;
    const auto phase = state & 1U;
    const auto decrement = state >> 1U;
    auto *local_signal = reinterpret_cast<int *>(words + 4U + phase);
    const auto local_thread = threadIdx.x - BaseThread;
    const auto peers = NodeLocal ? a.local_world_size : a.world;
    const auto peer_base = NodeLocal ? a.node * a.local_world_size : 0U;
    if (local_thread < peers) {
      auto *remote_words = reinterpret_cast<unsigned int *>(
          peer_at<unsigned long long>(a.peer_bases[peer_base + local_thread],
                                      a.forward_scheduler_offset) +
          10ULL + 2ULL * a.world);
      auto *remote_signal =
          reinterpret_cast<int *>(remote_words + 4U + phase);
      weftmoe_pf::reduce_add_release_system(remote_signal,
                                           decrement ? -1 : 1);
    }
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
    if (local_thread == 0U) {
      atomicAdd(words + 1U, 1U);
      const auto target = decrement ? 0 : static_cast<int>(peers);
      while (weftmoe_pf::load_acquire_system(local_signal) != target) {
      }
    }
  }

  if constexpr (ReleaseGrid) {
    grid_sync<BaseThread, Threads>(a);
  } else {
    // Without the epilogue grid broadcast, keep the participating lanes in
    // CTA 0 behind its leader's system-acquire poll.  Other CTAs execute the
    // same local join and may then wait at a caller-owned grid barrier.
    collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  }
}

template <unsigned MathThreads = forward_tuning::kMathWarpgroups * 128U>
__device__ __forceinline__ void before_combine(const ForwardKernelArgs &a) {
  constexpr auto kThreads = MathThreads;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (blockIdx.x == 0U && threadIdx.x == 0U)
    status->before_combine_start_clock = device_globaltimer_ns();
  rank_barrier<0U, kThreads>(a);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    const auto now = device_globaltimer_ns();
    status->before_combine_complete_clock = now;
    status->return_complete_clock = now;
  }
}

// Once every rank has published its FC2 contributions, no Dispatch writer can
// touch the destination-local arrival counters. Control lanes clear them while
// math lanes combine tokens. The terminal rank barrier below includes a CTA
// join, so a rank cannot enter the next generation until every peer has
// completed this cleanup.
__device__ __forceinline__ void
clear_dispatch_counters(const ForwardKernelArgs &a) {
  constexpr auto kMathThreads = forward_tuning::kMathWarpgroups * 128U;
  if (threadIdx.x < kMathThreads)
    return;
  auto *arrived = peer_at<unsigned long long>(
      a.local_base, a.x_arrived_rows_offset);
  const auto count =
      dispatch_sizes(a.dispatch).mtile_count * a.local_world_size;
  const auto lane = static_cast<std::uint64_t>(threadIdx.x - kMathThreads);
  const auto workers = blockDim.x - kMathThreads;
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * workers + lane;
       index < count;
       index += static_cast<std::uint64_t>(gridDim.x) * workers)
    arrived[index] = 0ULL;
  if (a.fused_inference_routes) {
    auto *local_counts = a.fused_local_expert_count;
    auto *send_counts =
        const_cast<std::uint64_t *>(a.dispatch.source_expert_count);
    for (auto expert = static_cast<std::uint64_t>(blockIdx.x) * workers + lane;
         expert < a.dispatch.identity.num_experts;
         expert += static_cast<std::uint64_t>(gridDim.x) * workers)
      local_counts[expert] = 0ULL;
    for (auto expert = static_cast<std::uint64_t>(blockIdx.x) * workers + lane;
         expert < a.dispatch.identity.num_experts;
         expert += static_cast<std::uint64_t>(gridDim.x) * workers)
      send_counts[expert] = 0ULL;
    if (blockIdx.x == 0U && lane == 0ULL)
      const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime)->error = 0ULL;
  }
}

// Do not let a fast rank recycle its slot while another rank still consumes
// the preceding epoch's contributions or source staging.
template <unsigned MathThreads = forward_tuning::kMathWarpgroups * 128U,
          unsigned ControlThreads = forward_tuning::kUnifiedThreads - MathThreads>
__device__ __forceinline__ void after_combine(const ForwardKernelArgs &a) {
  // The stream cannot launch the next generation until CTA 0 exits this
  // kernel. Other CTAs need only contribute their cleanup stores and may
  // retire after the local grid arrival, matching MegaMOE's AfterClean
  // barrier without a redundant grid-wide epilogue.
  constexpr auto kBase = MathThreads;
  constexpr auto kThreads = ControlThreads;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (blockIdx.x == 0U && threadIdx.x == kBase)
    status->after_combine_start_clock = device_globaltimer_ns();
  rank_barrier<kBase, kThreads, false>(a);
  if (blockIdx.x == 0U && threadIdx.x == kBase)
    status->after_combine_complete_clock = device_globaltimer_ns();
}

} // namespace weftmoe_pf
