#pragma once

#include "cuda/communication/notification_pipeline.cuh"
#include "cuda/communication/return_queue.cuh"

// FC2 publishes canonical row IDs. Relay consumes the local queue; each fixed
// GIN owner consumes its remote queue. A separate arrival queue carries compact
// row IDs back to the source. No worker searches compute tiles or chunk flags.
#if __has_include(<nccl_device.h>)
namespace forward_return_detail {

using ReturnCopyBarrier = ::cuda::barrier<::cuda::thread_scope_block>;
using RowQueue = persistent_v3::ReturnRowQueue;

__device__ __forceinline__ bool
cancelled(const ForwardKernelArgs &a,
          persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

// The fixed count matrix supplies the accounting total; Return scheduling
// itself consumes row queues and terminal chunks, without a compact directory.
__device__ __forceinline__ unsigned long long
source_compact_rows(const ForwardKernelArgs &a) {
  unsigned long long rows = 0;
  for (unsigned expert = 0; expert < a.dispatch.identity.num_experts; ++expert)
    if (a.dispatch.expert_to_rank[expert] / a.local_world_size != a.node)
      rows +=
          a.dispatch
              .source_expert_count[a.rank * a.dispatch.identity.num_experts +
                                   expert];
  return rows;
}

__device__ __forceinline__ unsigned long long
claim_rows(RowQueue &queue, unsigned int limit, unsigned int &count) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> published(
      queue.confirmed),
      head(queue.head);
  const auto end = published.load(::cuda::memory_order_acquire);
  auto begin = head.load(::cuda::memory_order_relaxed);
  count = 0U;
  if (begin >= end)
    return begin;
  const auto n = static_cast<unsigned int>(min(end - begin, 1ULL * limit));
  if (head.compare_exchange_strong(begin, begin + n,
                                   ::cuda::memory_order_relaxed))
    count = n;
  return begin;
}

// Called by all math threads after the last FC2 N-tile acquires every store.
// The dead GEMM operand storage supplies a few prefix words. Reserve/fill/
// confirm each queue before touching the next, so producers cannot form a
// cyclic wait between queues. CAS confirms exactly this producer's interval.
template <unsigned int MathThreads>
__device__ __noinline__ void
publish_rows(const ForwardKernelArgs &a, unsigned long long row,
             unsigned int rows, unsigned int *scratch) {
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  return_queue_detail::publish<MathThreads>(
      a, row, rows, scratch, [&] { return cancelled(a, *control); });
}

struct LocalReturnState {
  unsigned queue_index = 0;
  unsigned long long cursor = 0ULL;
  unsigned int selected = 0U;
};

__device__ __forceinline__ bool
local_return_terminal(const ForwardKernelArgs &a,
                      const LocalReturnState &state) {
  auto &queue = return_queue_detail::queues(a)[state.queue_index];
  // Notify supplies the exact row count. Every claim follows publication,
  // so no separate producer-completion counter is needed for an empty queue.
  return persistent_v3::load_acquire<::cuda::thread_scope_device>(
             queue.head) == queue.capacity;
}

__device__ __forceinline__ void copy_sync(unsigned threads) {
  asm volatile("bar.sync 6, %0;" ::"r"(threads) : "memory");
}

__device__ __forceinline__ bool
progress_local_return(const ForwardKernelArgs &a,
                      persistent_v3::ForwardPipelineControl &control,
                      unsigned int context,
                      LocalReturnState &state, std::byte *tma_storage,
                      ReturnCopyBarrier *tma_full, unsigned threads = 0U) {
  if (!threads)
    threads = blockDim.x;
  if (threadIdx.x == 0U) {
    state.selected = 0U;
    if (!cancelled(a, control))
      state.cursor =
          claim_rows(return_queue_detail::queues(a)[state.queue_index],
                     a.forward_local_return_batch_rows, state.selected);
  }
  copy_sync(threads);
  if (state.selected == 0U)
    return false;
  WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnBatchStart, LocalReturn,
                          context, state.selected, execution_epoch(a));
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (threadIdx.x == 0U)
    atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
  const auto *entries =
      peer_at<const unsigned int>(a.local_base, a.forward_return_rows_offset) +
      return_queue_detail::queues(a)[state.queue_index].begin;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *output = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  const auto copy_bytes = static_cast<unsigned int>(a.hidden * 2ULL);
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const bool tma = a.forward_relay_tma_warps != 0U && copy_bytes <= 16384U &&
                   (copy_bytes & 15U) == 0U;
  const auto workers = tma ? a.forward_relay_tma_warps : threads / 32U;
  bool copied = false;
  for (unsigned int item = warp; warp < workers && item < state.selected;
       item += workers) {
    const auto row = entries[state.cursor + item];
    const auto header = headers[row];
    const auto *src = output + row * a.hidden;
    auto *dst = peer_at<transport::BFloat16>(a.peer_bases[header.source_rank],
                                             a.contribution_offset) +
                header.route_index * a.hidden;
    if (tma) {
      auto *storage = tma_storage + warp * copy_bytes;
      if (lane == 0U) {
        auto &full = tma_full[warp];
        ::cuda::device::experimental::cp_async_bulk_global_to_shared(
            storage, src, copy_bytes, full);
        auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, copy_bytes);
        full.wait(static_cast<decltype(ticket) &&>(ticket));
        ::cuda::device::experimental::cp_async_bulk_shared_to_global(
            dst, storage, copy_bytes);
        ::cuda::device::experimental::cp_async_bulk_commit_group();
        ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
        copied = true;
      }
    } else {
      for (unsigned long long v = lane; v < a.hidden / 8ULL; v += 32ULL)
        reinterpret_cast<uint4 *>(dst)[v] =
            reinterpret_cast<const uint4 *>(src)[v];
    }
  }

  if (copied)
    ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
  __threadfence_system();
  copy_sync(threads);
  auto fill = [&](unsigned i) {
    const auto header = headers[entries[state.cursor + i]];
    return local_notification_detail::Entry{
        header.source_rank % a.local_world_size, header.route_index};
  };
  if (state.queue_index) {
    const auto incoming =
        (a.rank + a.local_world_size - blockIdx.x % a.local_world_size) %
        a.local_world_size;
    local_notification_detail::publish_peer_return(
        a, state.queue_index - 1U, state.selected, threads, fill, incoming);
  } else
    local_notification_detail::publish_return(a, state.selected, threads, fill);
  unsigned int peer_rows = 0U;
  for (unsigned int item = threadIdx.x; item < state.selected; item += threads)
    peer_rows += headers[entries[state.cursor + item]].source_rank != a.rank;
  const auto copied_peer_rows = __reduce_add_sync(0xffffffffU, peer_rows);
  if (lane == 0U && copied_peer_rows != 0U)
    atomicAdd(&status->actual_return_peer_bytes,
              copied_peer_rows * a.hidden * 2ULL);
  copy_sync(threads);
  WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnBatchEnd, LocalReturn, context,
                          state.selected, execution_epoch(a));
  return true;
}

} // namespace forward_return_detail
#endif
