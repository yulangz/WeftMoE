#pragma once

#ifndef WEFTMOE_FWD_GIN_LOCAL_PEER_QUEUES
#define WEFTMOE_FWD_GIN_LOCAL_PEER_QUEUES 0
#endif

// Included after the private kernel contracts and peer-address helpers.
namespace return_queue_detail {
using Queue = persistent_v3::ReturnRowQueue;

template <class Args> __device__ __forceinline__ Queue *queues(const Args &a) {
  return peer_at<Queue>(a.local_base, a.forward_return_queues_offset);
}

template <class Args>
__device__ __forceinline__ unsigned long long *ready_entries(const Args &a) {
  return peer_at<unsigned long long>(a.local_base, a.forward_return_ready_offset);
}

// Entries never wrap within an invocation. Producers do not wait for earlier
// tickets: each entry release-publishes its own completed row independently.
template <class Args>
__device__ __forceinline__ unsigned ready_rows(const Args &a, const Queue &q,
                                              unsigned long long begin, unsigned limit) {
  unsigned count = 0;
  while (count < limit && begin + count < q.capacity) {
    if (::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            ready_entries(a)[q.begin + begin + count])
            .load(::cuda::memory_order_acquire) != execution_epoch(a)) break;
    ++count;
  }
  return count;
}

template <class Args>
__device__ __forceinline__ unsigned index(const Args &a, unsigned source) {
  return source / a.local_world_size == a.node ? 0U : 1U + source;
}

__device__ __forceinline__ bool local_peer_queues(const ForwardKernelArgs &a) {
  return (a.forward_stream_local_dispatch ||
          (WEFTMOE_FWD_GIN_LOCAL_PEER_QUEUES && a.use_gin)) &&
         a.forward_relay_cta_count >= a.local_world_size;
}

__device__ __forceinline__ unsigned index(const ForwardKernelArgs &a,
                                          unsigned source) {
  return local_peer_queues(a)
             ? 1U + source
             : (source / a.local_world_size == a.node ? 0U : 1U + source);
}

template <class Args>
__device__ __forceinline__ void initialize(const Args &a) {
  if (blockIdx.x)
    return;
  auto *q = queues(a);
  for (unsigned i = threadIdx.x; i <= a.world; i += blockDim.x)
    q[i] = Queue{};
  __syncthreads();
  const auto lane = threadIdx.x % 32U;
  for (unsigned source = threadIdx.x / 32U; source < a.world;
       source += blockDim.x / 32U) {
    unsigned long long count = 0;
    for (unsigned e = lane; e < dispatch_sizes(a.dispatch).owned_expert_count;
         e += 32U)
      count +=
          a.dispatch
              .source_expert_count[source * a.dispatch.identity.num_experts +
                                   a.dispatch.owned_expert_ids[e]];
    for (unsigned delta = 16; delta; delta /= 2U)
      count += __shfl_down_sync(0xffffffffU, count, delta);
    if (!lane)
      atomicAdd(&q[index(a, source)].capacity, count);
  }
  __syncthreads();
  if (!threadIdx.x) {
    unsigned long long begin = 0;
    for (unsigned i = 0; i <= a.world; ++i) {
      q[i].begin = begin;
      begin += q[i].capacity;
    }
  }
  __syncthreads();
  const auto total = q[a.world].begin + q[a.world].capacity;
  for (unsigned long long i = threadIdx.x; i < total; i += blockDim.x)
    ready_entries(a)[i] = 0ULL;
  __syncthreads();
}

template <unsigned Threads, class Args, class Cancelled>
__device__ __noinline__ void publish(const Args &a, unsigned long long first,
                                     unsigned rows, unsigned *scratch,
                                     Cancelled cancelled) {
  constexpr auto warps = Threads / 32U;
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  auto *entries = peer_at<unsigned>(a.local_base, a.forward_return_rows_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto row = threadIdx.x < rows ? first + threadIdx.x : ~0ULL;
  auto pending = row != ~0ULL ? index(a, headers[row].source_rank) : UINT32_MAX;
  auto *base = reinterpret_cast<unsigned long long *>(scratch + 16U);
  while (true) {
    const auto next = __reduce_min_sync(0xffffffffU, pending);
    if (!lane)
      scratch[warp] = next;
    cutlass::arch::NamedBarrier::sync(Threads, 2U);
    if (!threadIdx.x) {
      auto selected = UINT32_MAX;
      for (unsigned w = 0; w < warps; ++w)
        selected = min(selected, scratch[w]);
      scratch[warps] = selected;
    }
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
    const auto selected = scratch[warps];
    if (selected == UINT32_MAX)
      break;
    const auto mask = __ballot_sync(0xffffffffU, pending == selected);
    if (!lane)
      scratch[warp] = __popc(mask);
    cutlass::arch::NamedBarrier::sync(Threads, 2U);
    auto &q = queues(a)[selected];
    if (!threadIdx.x) {
      unsigned count = 0;
      for (unsigned w = 0; w < warps; ++w) {
        const auto n = scratch[w];
        scratch[w] = count;
        count += n;
      }
      scratch[warps + 1U] = count;
      *base =
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
              q.reserved)
              .fetch_add(count, ::cuda::memory_order_relaxed);
      // Notify counted these rows exactly; overflow is a protocol
      // error, never a reason to truncate a skewed batch.
      if (*base + count > q.capacity)
        asm volatile("trap;");
    }
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
    if (pending == selected) {
      const auto offset = scratch[warp] + __popc(mask & ((1U << lane) - 1U));
      entries[q.begin + *base + offset] = static_cast<unsigned>(row);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
          ready_entries(a)[q.begin + *base + offset])
          .store(execution_epoch(a), ::cuda::memory_order_release);
      pending = UINT32_MAX;
    }
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
  }
}

} // namespace return_queue_detail
