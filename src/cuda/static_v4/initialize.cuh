#pragma once

namespace static_v4 {

constexpr unsigned kCollectiveScopeBarrier = 12U;
constexpr unsigned kCollectiveJoinBarrier = 13U;

template <unsigned Threads, unsigned Barrier>
__device__ __forceinline__ void collective_scope_sync() {
  asm volatile("bar.sync %0, %1;"
               :
               : "n"(Barrier), "r"(Threads)
               : "memory");
}

__device__ __forceinline__ unsigned int *
collective_words(const ForwardKernelArgs &a) {
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  return reinterpret_cast<unsigned int *>(
      scheduler + 10ULL + 2ULL * a.world);
}

// Reusable one-word resident-grid barrier. Every CTA contributes one release
// add; CTA 0 contributes the complementary value that flips the generation
// bit exactly when the last CTA arrives.
template <unsigned BaseThread = 0U,
          unsigned Threads = forward_tuning::kUnifiedThreads>
__device__ __forceinline__ void grid_sync(const ForwardKernelArgs &a) {
  auto *word = collective_words(a);
  constexpr unsigned kGeneration = 0x80000000U;
  collective_scope_sync<Threads, kCollectiveScopeBarrier>();
  if (threadIdx.x == BaseThread) {
    ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_device> counter(
        *word);
    const auto delta = blockIdx.x == 0U
                           ? kGeneration - (gridDim.x - 1U)
                           : 1U;
    const auto previous =
        counter.fetch_add(delta, ::cuda::memory_order_release);
    while (((counter.load(::cuda::memory_order_acquire) ^ previous) &
            kGeneration) == 0U) {
    }
  }
  collective_scope_sync<Threads, kCollectiveScopeBarrier>();
}

// Reset one inference generation and build its immutable M-block map. Dispatch
// counters are already zero: allocation initializes the first use, and the
// preceding generation clears them before its terminal rank barrier. CTA 0
// constructs the small expert map together with scalar state, so one grid join
// publishes the complete generation. V3's rings, dynamic queues and completion
// vectors are not used by either V4 topology specialization.
__device__ __noinline__ void initialize(const ForwardKernelArgs &a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_done =
      peer_at<unsigned long long>(a.local_base, a.forward_fc1_done_offset);
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);

  unsigned long long total_mblocks = 0ULL;
  if (blockIdx.x == 0U) {
    if (a.runtime_epoch) {
      if (threadIdx.x == 0U)
        atomicAdd(a.runtime_epoch, 1ULL);
      __syncthreads();
    }
    if (threadIdx.x == 0U && !a.fused_inference_routes)
      *status = {};

    if (threadIdx.x < 32U) {
      for (std::uint32_t expert = threadIdx.x;
           expert < dispatch_sizes(a.dispatch).owned_expert_count;
           expert += 32U)
        total_mblocks += forward_expert_mblock_count(a, expert);
      for (std::uint32_t delta = 16U; delta; delta >>= 1U)
        total_mblocks +=
            __shfl_down_sync(0xffffffffU, total_mblocks, delta);
    }

    if (threadIdx.x == 0U) {
      if (!a.fused_inference_routes)
        status->start_clock = device_globaltimer_ns();
      status->dispatch_blocks_done = ~0ULL;
      status->relay_blocks_done = ~0ULL;
      status->reduced_tokens = ~0ULL;
      status->fc1_start_clock = ~0ULL;
      scheduler[0] = scheduler[1] = scheduler[2] = 0ULL;
      scheduler[3] = total_mblocks;
      scheduler[4] = 0ULL;
      scheduler[4ULL + a.world] = 0ULL;
      scheduler[11ULL + 2ULL * a.world] = 0ULL;
      control->rdma_return_roles_complete.value = 0ULL;
      control->compute_reduce_roles_complete.value = 0ULL;
    }
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    __syncthreads();
    forward_initialize_mblocks<true>(
        a, [&](unsigned long long mblock, unsigned expert,
               unsigned local_mblock) {
          claims[mblock] =
              (static_cast<unsigned long long>(expert) << 32U) | local_mblock;
          fc1_done[mblock] = execution_epoch(a) << 32U;
        });
#endif
  }
  grid_sync(a);
}

} // namespace static_v4
