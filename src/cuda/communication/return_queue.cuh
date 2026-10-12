#pragma once

#ifndef WEFTMOE_FWD_GIN_LOCAL_PEER_QUEUES
#define WEFTMOE_FWD_GIN_LOCAL_PEER_QUEUES 0
#endif

// Included after the private kernel contracts and peer-address helpers.
namespace return_queue_detail {
using Queue = communication::ReturnRowQueue;

template <class Args> __device__ __forceinline__ Queue *queues(const Args &a) {
  return peer_at<Queue>(a.local_base, a.forward_return_queues_offset);
}

// UINT32_MAX is not a valid compact row. Setup clears each finite queue only
// after the previous invocation drained, before releasing any new producers.
template <class Args>
__device__ __forceinline__ unsigned *entries(const Args &a) {
  return peer_at<unsigned>(a.local_base, a.forward_return_rows_offset);
}

inline constexpr unsigned kReadyBlockRows =
    communication::kReturnReadyBlockRows;

template <class Args>
__device__ __forceinline__ unsigned *ready_counts(const Args &a) {
  const auto &last = queues(a)[a.world];
  return entries(a) + last.begin + last.capacity;
}

template <class Args>
__device__ __forceinline__ bool block_ready(const Args &a, const Queue &q,
                                            unsigned long long position) {
  const auto block = position / kReadyBlockRows;
  const auto expected = unsigned(
      min(1ULL * kReadyBlockRows, q.capacity - block * kReadyBlockRows));
  return ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
             ready_counts(a)[q.ready_begin + block])
             .load(::cuda::memory_order_acquire) == expected;
}

// All producer threads join before this call. An atomic count describes one
// finite 32-row block, never a prefix or an ordering between different blocks.
// Producers add only their own disjoint reservation and never wait for peers.
template <class Args>
__device__ __forceinline__ void publish_ready(const Args &a, const Queue &q,
                                              unsigned long long begin,
                                              unsigned rows) {
  while (rows) {
    const auto block = begin / kReadyBlockRows;
    const auto n =
        min(rows, kReadyBlockRows - unsigned(begin % kReadyBlockRows));
    ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
        ready_counts(a)[q.ready_begin + block])
        .fetch_add(n, ::cuda::memory_order_acq_rel);
    begin += n;
    rows -= n;
  }
}

template <class Args>
__device__ __forceinline__ unsigned ready_rows(const Args &a, const Queue &q,
                                               unsigned long long begin,
                                               unsigned limit) {
  unsigned count = 0;
  while (count < limit && begin + count < q.capacity) {
    const auto position = begin + count;
    const auto n =
        unsigned(min(min(1ULL * (limit - count), q.capacity - position),
                     kReadyBlockRows - position % kReadyBlockRows));
    if (block_ready(a, q, position)) {
      count += n;
      continue;
    }
    for (unsigned i = 0; i < n; ++i) {
      if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
              entries(a)[q.begin + begin + count])
              .load(::cuda::memory_order_acquire) == UINT32_MAX)
        return count;
      ++count;
    }
  }
  return count;
}

template <class Args>
__device__ __forceinline__ bool ready_block(const Args &a, const Queue &q,
                                            unsigned long long begin,
                                            unsigned rows) {
  return ready_rows(a, q, begin, rows) == rows;
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

// Parallel callers must join the setup grid before releasing producers. Row
// storage follows Notify's exact incoming count; per-queue rounding needs at
// most one extra ready block per queue. Queue metadata still has one owner.
template <bool ParallelClear = false, class Args>
__device__ __forceinline__ void initialize(const Args &a) {
  if constexpr (ParallelClear) {
    const auto total = dispatch_sizes(a.dispatch).incoming_row_count;
    const auto blocks = total / kReadyBlockRows + 1ULL + a.world;
    const auto tid = 1ULL * blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = 1ULL * gridDim.x * blockDim.x;
    for (auto i = tid; i < total; i += stride)
      entries(a)[i] = UINT32_MAX;
    for (auto i = tid; i < blocks; i += stride)
      entries(a)[total + i] = 0U;
  }
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
    unsigned long long begin = 0, ready_begin = 0;
    for (unsigned i = 0; i <= a.world; ++i) {
      q[i].begin = begin;
      q[i].ready_begin = ready_begin;
      begin += q[i].capacity;
      ready_begin += (q[i].capacity + kReadyBlockRows - 1U) / kReadyBlockRows;
    }
  }
  __syncthreads();
  if constexpr (ParallelClear) {
    if (q[a.world].begin + q[a.world].capacity !=
        dispatch_sizes(a.dispatch).incoming_row_count)
      asm volatile("trap;");
    return;
  }
  const auto total = q[a.world].begin + q[a.world].capacity;
  for (unsigned long long i = threadIdx.x; i < total; i += blockDim.x)
    entries(a)[i] = UINT32_MAX;
  const auto blocks =
      q[a.world].ready_begin +
      (q[a.world].capacity + kReadyBlockRows - 1U) / kReadyBlockRows;
  for (unsigned long long i = threadIdx.x; i < blocks; i += blockDim.x)
    ready_counts(a)[i] = 0U;
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
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
          entries[q.begin + *base + offset])
          .store(static_cast<unsigned>(row), ::cuda::memory_order_release);
      pending = UINT32_MAX;
    }
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
    if (!threadIdx.x)
      publish_ready(a, q, *base, scratch[warps + 1U]);
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
  }
}

} // namespace return_queue_detail
