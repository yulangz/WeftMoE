#pragma once

// Included after the private kernel contracts and peer-address helpers.
namespace return_queue_detail {
using Queue = persistent_v3::ReturnRowQueue;

template <class Args> __device__ __forceinline__ Queue *queues(const Args &a) {
  return peer_at<Queue>(a.local_base, a.forward_return_queues_offset);
}

template <class Args>
__device__ __forceinline__ unsigned index(const Args &a, unsigned source) {
  return source / a.local_world_size == a.node ? 0U : 1U + source;
}

template <class Args>
__device__ __forceinline__ void initialize(const Args &a) {
  if (blockIdx.x || threadIdx.x)
    return;
  auto *q = queues(a);
  for (unsigned i = 0; i <= a.world; ++i)
    q[i] = Queue{};
  for (unsigned source = 0; source < a.world; ++source)
    for (unsigned e = 0; e < dispatch_sizes(a.dispatch).owned_expert_count; ++e)
      q[index(a, source)].capacity +=
          a.dispatch
              .source_expert_count[source * a.dispatch.identity.num_experts +
                                   a.dispatch.owned_expert_ids[e]];
  unsigned long long begin = 0;
  for (unsigned i = 0; i <= a.world; ++i) {
    q[i].begin = begin;
    begin += q[i].capacity;
  }
}

template <unsigned Threads, class Args, class Cancelled>
__device__ __noinline__ void publish(const Args &a, unsigned long long row,
                                     unsigned rows, unsigned *scratch,
                                     Cancelled cancelled) {
  constexpr auto warps = Threads / 32U;
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  auto *entries = peer_at<unsigned>(a.local_base, a.forward_return_rows_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  auto pending = threadIdx.x < rows
                     ? index(a, headers[row + threadIdx.x].source_rank)
                     : UINT32_MAX;
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
      // Notify counted these exact rows; a violation is a protocol error,
      // never a reason to truncate a skewed batch.
      if (*base + count > q.capacity)
        asm volatile("trap;");
    }
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
    if (pending == selected) {
      const auto offset = scratch[warp] + __popc(mask & ((1U << lane) - 1U));
      entries[q.begin + *base + offset] =
          static_cast<unsigned>(row + threadIdx.x);
      pending = UINT32_MAX;
    }
    __threadfence();
    cutlass::arch::NamedBarrier::sync(Threads, 2U);
    if (!threadIdx.x) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          confirm(q.confirmed);
      auto expected = *base;
      while (!confirm.compare_exchange_weak(
          expected, *base + scratch[warps + 1U], ::cuda::memory_order_acq_rel,
          ::cuda::memory_order_relaxed)) {
        if (cancelled())
          break;
        expected = *base;
        __nanosleep(32U);
      }
    }
    cutlass::arch::NamedBarrier::sync(Threads, 3U);
  }
}
} // namespace return_queue_detail
