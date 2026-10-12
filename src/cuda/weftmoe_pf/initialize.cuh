#pragma once

namespace weftmoe_pf {

constexpr unsigned kCollectiveScopeBarrier = 12U;
constexpr unsigned kCollectiveJoinBarrier = 13U;

template <unsigned Threads, unsigned Barrier>
__device__ __forceinline__ void collective_scope_sync() {
  asm volatile("bar.sync %0, %1;" : : "n"(Barrier), "r"(Threads) : "memory");
}

__device__ __forceinline__ unsigned int *
collective_words(const ForwardKernelArgs &a) {
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  return reinterpret_cast<unsigned int *>(scheduler + 10ULL + 2ULL * a.world);
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
    const auto delta = blockIdx.x == 0U ? kGeneration - (gridDim.x - 1U) : 1U;
    const auto previous =
        counter.fetch_add(delta, ::cuda::memory_order_release);
    while (((counter.load(::cuda::memory_order_acquire) ^ previous) &
            kGeneration) == 0U) {
    }
  }
  collective_scope_sync<Threads, kCollectiveScopeBarrier>();
}

} // namespace weftmoe_pf
