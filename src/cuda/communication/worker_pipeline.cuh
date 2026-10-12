#pragma once
#include <type_traits>

#if __has_include(<nccl_device.h>)

#ifndef WEFTMOE_FWD_DISPATCH_ISSUER_BUFFERS
#define WEFTMOE_FWD_DISPATCH_ISSUER_BUFFERS 1
#endif

#include "cuda/communication/notification_pipeline.cuh"
#include "cuda/communication/return_queue.cuh"

// Included after the private Forward contract, peer helpers and input-group
// publisher. Every CTA owns an independent communication domain; no helper in
// this file waits for the math threads or reuses their operand storage.
namespace communication_worker {

using CopyBarrier = ::cuda::barrier<::cuda::thread_scope_block>;
#ifndef WEFTMOE_FWD_DEFER_LOCAL_RETURN_DURING_DISPATCH
#define WEFTMOE_FWD_DEFER_LOCAL_RETURN_DURING_DISPATCH 0
#endif

#include "cuda/communication/gin_sender_state.cuh"

// Three warp domains are useful when sharing the existing producer warpgroup;
// four or eight warps allow a separate communication warpgroup. The final warp
// publishes input groups and then receives notifications while copies proceed.
template <unsigned Threads, unsigned SegmentBytes = 2048U,
          unsigned Buffers = 1U, unsigned DispatchLanes = 1U>
struct alignas(128) Shared {
  static_assert(Threads >= 64U && Threads % 32U == 0U);
  static_assert(SegmentBytes >= 16U && SegmentBytes % 16U == 0U);
  static_assert(Buffers == 1U || Buffers == 2U);
  static_assert(DispatchLanes >= 1U && DispatchLanes <= 32U &&
                (DispatchLanes & (DispatchLanes - 1U)) == 0U);
  static constexpr unsigned kDispatchLanes = DispatchLanes;
  static constexpr unsigned kDispatchBuffers =
      DispatchLanes == 1U ? Buffers : WEFTMOE_FWD_DISPATCH_ISSUER_BUFFERS;
  static_assert(kDispatchBuffers == 1U || kDispatchBuffers == 2U);
  static constexpr unsigned kStorageSlots =
      Buffers > DispatchLanes *kDispatchBuffers
          ? Buffers
          : DispatchLanes *kDispatchBuffers;
  static constexpr unsigned kThreads = Threads;
  static constexpr unsigned kWarps = Threads / 32U;
  static constexpr unsigned kCopyWarps = Threads / 32U - 1U;
  static constexpr unsigned kCopyThreads = kCopyWarps * 32U;
  static constexpr unsigned kCopyBarrier = 8U;
  static constexpr unsigned kSegmentBytes = SegmentBytes;
  static constexpr unsigned kBuffers = Buffers;

  alignas(128) unsigned char staging[kCopyWarps * kStorageSlots * SegmentBytes];
  alignas(16) CopyBarrier full[kCopyWarps];
  alignas(16) CopyBarrier row_prefetch_full[kCopyWarps * kStorageSlots];
  unsigned long long dispatch_cursor;
  unsigned long long dispatch_end;
  unsigned long long return_cursor;
  unsigned long long notification_sequence;
  unsigned return_queue;
  unsigned return_rows;
  unsigned return_uniform;
  unsigned return_peer;
  unsigned dispatch_complete;
  unsigned stop;
  unsigned copies_done;
  unsigned pack_mailbox;  // 0: available, 1: immutable task ready, 2: stop.
  unsigned pack_observed; // Copy-domain broadcast; never written by control.

  GinSenderState gin_sender_state;
  activation_ring_detail::Transfer gin_task;
  unsigned gin_task_slot;
  unsigned gin_rx_dispatch;
  unsigned gin_rx_return;
  unsigned gin_progressed;
};

template <unsigned Threads, unsigned BarrierId = 8U>
__device__ __forceinline__ void copy_sync() {
  if constexpr (Threads == 32U)
    __syncwarp();
  else
    asm volatile("bar.sync %0, %1;"
                 :
                 : "n"(BarrierId), "r"(Threads)
                 : "memory");
}

template <unsigned Threads> __device__ __forceinline__ void domain_sync() {
  asm volatile("bar.sync 9, %0;" : : "r"(Threads) : "memory");
}

template <unsigned BaseThread>
__device__ __forceinline__ unsigned local_thread() {
  return threadIdx.x - BaseThread;
}

template <class Storage>
__device__ __forceinline__ void copy_row(Storage &s, unsigned warp,
                                         const void *source, void *target,
                                         unsigned bytes) {
  auto *buffer =
      s.staging + warp * Storage::kStorageSlots * Storage::kSegmentBytes;
  auto &full = s.full[warp];
  constexpr unsigned kStep = Storage::kBuffers * Storage::kSegmentBytes;
  for (unsigned offset = 0U; offset < bytes; offset += kStep) {
    const unsigned extent = min(kStep, bytes - offset);
#pragma unroll
    for (unsigned item = 0U; item < Storage::kBuffers; ++item) {
      const unsigned begin = item * Storage::kSegmentBytes;
      if (begin < extent)
        ::cuda::device::experimental::cp_async_bulk_global_to_shared(
            buffer + begin, static_cast<const char *>(source) + offset + begin,
            min(Storage::kSegmentBytes, extent - begin), full);
    }
    auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, extent);
    full.wait(static_cast<decltype(ticket) &&>(ticket));
#pragma unroll
    for (unsigned item = 0U; item < Storage::kBuffers; ++item) {
      const unsigned begin = item * Storage::kSegmentBytes;
      if (begin < extent)
        ::cuda::device::experimental::cp_async_bulk_shared_to_global(
            static_cast<char *>(target) + offset + begin, buffer + begin,
            min(Storage::kSegmentBytes, extent - begin));
    }
    ::cuda::device::experimental::cp_async_bulk_commit_group();
    ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
  }
}

// Dispatch fanouts share one immutable payload. Every store in a segment
// joins before staging reuse; callers still wait for global completion before
// publishing readiness or returning NIC credit.
template <class Storage, class Destination>
__device__ __forceinline__ void
copy_row_fanout(Storage &s, unsigned warp, const void *source, unsigned bytes,
                unsigned fanouts, Destination destination) {
  if (fanouts == 0U)
    return;
  auto *buffer =
      s.staging + warp * Storage::kStorageSlots * Storage::kSegmentBytes;
  auto &full = s.full[warp];
  constexpr unsigned kStep = Storage::kBuffers * Storage::kSegmentBytes;
  for (unsigned offset = 0U; offset < bytes; offset += kStep) {
    const unsigned extent = min(kStep, bytes - offset);
#pragma unroll
    for (unsigned item = 0U; item < Storage::kBuffers; ++item) {
      const unsigned begin = item * Storage::kSegmentBytes;
      if (begin < extent)
        ::cuda::device::experimental::cp_async_bulk_global_to_shared(
            buffer + begin, static_cast<const char *>(source) + offset + begin,
            min(Storage::kSegmentBytes, extent - begin), full);
    }
    auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, extent);
    full.wait(static_cast<decltype(ticket) &&>(ticket));
    for (unsigned fanout = 0U; fanout < fanouts; ++fanout) {
      auto *target = static_cast<char *>(destination(fanout));
#pragma unroll
      for (unsigned item = 0U; item < Storage::kBuffers; ++item) {
        const unsigned begin = item * Storage::kSegmentBytes;
        if (begin < extent)
          ::cuda::device::experimental::cp_async_bulk_shared_to_global(
              target + offset + begin, buffer + begin,
              min(Storage::kSegmentBytes, extent - begin));
      }
    }
    // All fanouts read the same immutable segment. One group joins every
    // destination before any subsequent G2S may overwrite that segment.
    ::cuda::device::experimental::cp_async_bulk_commit_group();
    ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
  }
}

#include "cuda/communication/row_prefetch.cuh"

// A lane owns a disjoint set of staging slots for the whole batch. The view
// lets the existing prefetch pipelines retain per-lane descriptors and tickets.
template <class Storage> struct LaneStagingView {
  static constexpr unsigned kBuffers = Storage::kDispatchBuffers;
  static constexpr unsigned kStorageSlots = kBuffers;
  static constexpr unsigned kSegmentBytes = Storage::kSegmentBytes;
  unsigned char *staging;
  CopyBarrier *row_prefetch_full;

  __device__ __forceinline__ LaneStagingView(Storage &s, unsigned warp,
                                             unsigned lane)
      : staging(s.staging + (warp * Storage::kStorageSlots + lane * kBuffers) *
                                kSegmentBytes),
        row_prefetch_full(s.row_prefetch_full + warp * Storage::kStorageSlots +
                          lane * kBuffers) {}
};

template <bool Fence = true> __device__ __forceinline__ void finish_copies() {
  ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
  if constexpr (Fence)
    __threadfence_system();
}

// Copy warps can publish disjoint contiguous input sub-batches independently.
// The two final domain joins protect the shared cursor. GIN's conservative
// policy retains the original interleaved rows and completion barrier.
template <unsigned BaseThread, unsigned CopyThreads = 0U,
          unsigned BarrierId = 8U, bool SingleConsumer = false, class Storage,
          class Args, class Fill>
__device__ __forceinline__ void publish_return(const Args &a, Storage &s,
                                               unsigned count, Fill fill,
                                               bool grouped = false) {
  constexpr auto threads = CopyThreads ? CopyThreads : Storage::kCopyThreads;
  const unsigned tid = local_thread<BaseThread>();
  const unsigned local = a.rank % a.local_world_size;
  // Canonical M-blocks often contain rows for a single source peer, even
  // when the queue also accepts other peers. Verify the actual immutable
  // entries before using dense notification positions; arbitrary mixed rows
  // keep the same complete partition below.
  if (count == 0U)
    return;
  unsigned selected_peer =
      grouped ? (s.return_queue - 1U) % a.local_world_size : 0U;
  if (!grouped) {
    if (tid == 0U) {
      s.return_uniform = 1U;
      s.return_peer = fill(0U).peer;
    }
    copy_sync<threads, BarrierId>();
    bool same_peer = true;
    for (unsigned i = tid; i < count; i += threads)
      same_peer = same_peer && fill(i).peer == s.return_peer;
    const unsigned mismatch = __ballot_sync(0xffffffffU, !same_peer);
    if ((tid & 31U) == 0U && mismatch)
      atomicExch(&s.return_uniform, 0U);
    copy_sync<threads, BarrierId>();
    grouped = s.return_uniform != 0U;
    selected_peer = s.return_peer;
    // Backward marks remote-only entries outside the NVLink peer domain.
    // Their GIN queue publication still runs after this helper returns.
    if (grouped && selected_peer >= a.local_world_size)
      return;
  }
  const unsigned first_peer = grouped ? selected_peer : 0U;
  const unsigned end_peer = grouped ? first_peer + 1U : a.local_world_size;
  for (unsigned peer = first_peer; peer < end_peer; ++peer) {
    for (unsigned begin = 0U; begin < count;
         begin += local_notification_detail::kRows) {
      const auto end = min(begin + local_notification_detail::kRows, count);
      if (tid == 0U) {
        s.return_rows = grouped ? end - begin : 0U;
        if (!grouped)
          for (unsigned i = begin; i < end; ++i)
            s.return_rows += fill(i).peer == peer;
        if (peer != local)
          s.notification_sequence = local_notification_detail::channel(
                                        a, a.local_base, peer, blockIdx.x)
                                        .sent;
      }
      copy_sync<threads, BarrierId>();
      const unsigned peer_rows = s.return_rows;
      copy_sync<threads, BarrierId>();
      if (peer_rows == 0U)
        continue;
      if (peer == local) {
        for (unsigned i = begin + tid; i < end; i += threads) {
          const auto entry = fill(i);
          if (entry.peer == peer)
            local_notification_detail::Receiver<Args>::apply(a, entry.row);
        }
        copy_sync<threads, BarrierId>();
        continue;
      }
      const auto sequence = s.notification_sequence;
      if (tid < 32U)
        local_notification_detail::wait_credit<Args, !SingleConsumer>(a, peer,
                                                                      sequence);
      copy_sync<threads, BarrierId>();
      auto &slot = local_notification_detail::channel(
                       a, a.peer_bases[a.node * a.local_world_size + peer],
                       local, blockIdx.x)
                       .slots[sequence % local_notification_detail::kDepth];
      // Peered queues have dense entries. For a mixed queue each lane counts
      // its immutable prefix; this is metadata-only and bounded by kRows.
      for (unsigned i = begin + tid; i < end; i += threads) {
        const auto entry = fill(i);
        if (entry.peer != peer)
          continue;
        unsigned position = grouped ? i - begin : 0U;
        if (!grouped)
          for (unsigned j = begin; j < i; ++j)
            position += fill(j).peer == peer;
        slot.rows[position] = entry.row;
      }
      __threadfence_system();
      copy_sync<threads, BarrierId>();
      if (tid == 0U) {
        slot.count = s.return_rows;
        slot.kind = local_notification_detail::Kind::Return;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
            slot.ready)
            .store(sequence / local_notification_detail::kDepth + 1ULL,
                   ::cuda::memory_order_release);
        local_notification_detail::channel(a, a.local_base, peer, blockIdx.x)
            .sent = sequence + 1ULL;
      }
      copy_sync<threads, BarrierId>();
    }
  }
}

template <unsigned BaseThread, bool AssistReceive = true, class Args,
          class Cancelled>
__device__ __forceinline__ void drain_notifications(const Args &a,
                                                    Cancelled cancelled) {
  const unsigned lane = local_thread<BaseThread>() % 32U;
  const unsigned local = a.rank % a.local_world_size;
  for (unsigned peer = 0U; peer < a.local_world_size; ++peer) {
    if (peer == local)
      continue;
    auto &state =
        local_notification_detail::channel(a, a.local_base, peer, blockIdx.x);
    const auto sequence = state.sent;
    local_notification_detail::wait_credit<Args, AssistReceive>(a, peer,
                                                                sequence);
    if (lane == 0U) {
      auto &slot = local_notification_detail::channel(
                       a, a.peer_bases[a.node * a.local_world_size + peer],
                       local, blockIdx.x)
                       .slots[sequence % local_notification_detail::kDepth];
      slot.count = 0U;
      slot.kind = local_notification_detail::Kind::End;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          slot.ready)
          .store(sequence / local_notification_detail::kDepth + 1ULL,
                 ::cuda::memory_order_release);
      state.sent = sequence + 1ULL;
    }
    __syncwarp();
  }
  while (true) {
    // Cancellation may arrive between lane loads; leave the warp uniformly
    // before entering progress and the full-mask completion shuffle below.
    const unsigned stop = lane == 0U ? unsigned(cancelled()) : 0U;
    if (__shfl_sync(0xffffffffU, stop, 0U))
      break;
    if constexpr (AssistReceive)
      local_notification_detail::progress(a);
    unsigned done = 1U;
    if (lane == 0U) {
      for (unsigned peer = 0U; peer < a.local_world_size; ++peer) {
        if (peer == local)
          continue;
        auto &state = local_notification_detail::channel(a, a.local_base, peer,
                                                         blockIdx.x);
        done &= ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
                    state.ended)
                    .load(::cuda::memory_order_acquire) != 0U;
        const auto claimed =
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
                state.claimed)
                .load(::cuda::memory_order_acquire);
        for (unsigned slot = 0U; slot < local_notification_detail::kDepth;
             ++slot) {
          constexpr unsigned depth = local_notification_detail::kDepth;
          done &= ::cuda::atomic_ref<unsigned long long,
                                     ::cuda::thread_scope_device>(
                      state.slots[slot].consumed)
                      .load(::cuda::memory_order_acquire) >=
                  (claimed + depth - 1U - slot) / depth;
          done &= ::cuda::atomic_ref<unsigned long long,
                                     ::cuda::thread_scope_system>(
                      state.slots[slot].credit)
                      .load(::cuda::memory_order_acquire) >=
                  (state.sent + depth - 1U - slot) / depth;
        }
      }
    }
    if (__shfl_sync(0xffffffffU, done, 0U))
      break;
    __nanosleep(50U);
  }
}

#include "cuda/communication/gin_sender.cuh"

// Alternate finite FC1 group attempts with Return receives. Missing Dispatch
// input must never prevent this warp from returning notification credits.
template <unsigned BaseThread, class Args, class Storage, class Cancelled,
          class PackDispatch, class PackReturn, class OnSend, class OnComplete>
__device__ __forceinline__ void
run_sender(const Args &a, Storage &s, Cancelled is_cancelled,
           PackDispatch pack_dispatch, PackReturn pack_return, OnSend on_send,
           OnComplete on_complete) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (warp == Storage::kCopyWarps) {
    if (lane == 0U) {
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block> mailbox(
          s.pack_mailbox);
      while (!is_cancelled()) {
        const bool available = mailbox.load(::cuda::memory_order_acquire) == 0U;
        activation_ring_detail::Transfer task;
        unsigned slot = 0U;
        bool progressed = false;
        // Publish the next immutable descriptor before calling GIN. Keeping
        // this after progress_sender serialized every pack behind the prior
        // PUT.
        if (available &&
            gin_detail::reserve_pack<(forward_tuning::kGinChannelPackLimit >
                                      1U)>(a, s.gin_sender_state, slot, task)) {
          s.gin_task_slot = slot;
          mailbox.store(1U, ::cuda::memory_order_release);
          progressed = true;
        }
        activation_ring_detail::Transfer unused;
        progressed =
            gin_detail::progress_sender<(forward_tuning::kGinChannelPackLimit >
                                         1U)>(a, s.gin_sender_state, slot,
                                              unused, on_send, on_complete,
                                              false) ||
            progressed;
        if (s.gin_sender_state.done)
          break;
        if (!progressed)
          __nanosleep(forward_tuning::kTransportIdleNanosleep);
      }
      // A cancelled copy may still finish its current finite TMA batch; its
      // independent cancellation poll also observes shutdown after that batch.
      mailbox.store(2U, ::cuda::memory_order_release);
    }
  } else {
    while (true) {
      if (tid == 0U)
        s.pack_observed =
            is_cancelled()
                ? 2U
                : ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                      s.pack_mailbox)
                      .load(::cuda::memory_order_acquire);
      copy_sync<Storage::kCopyThreads>();
      const unsigned phase = s.pack_observed;
      // Capture before the leader can overwrite the broadcast on an idle poll.
      copy_sync<Storage::kCopyThreads>();
      if (phase == 2U)
        break;
      if (phase == 0U) {
        __nanosleep(forward_tuning::kTransportIdleNanosleep);
        continue;
      }
      // Capture the reserved slot while the mailbox still belongs to us.
      // The descriptor itself lives in the pending-slot pool and remains
      // immutable until every packer has published its completion below.
      const unsigned slot = s.gin_task_slot;
      copy_sync<Storage::kCopyThreads>();
      if (tid == 0U)
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.pack_mailbox)
            .store(0U, ::cuda::memory_order_release);
      const auto &task = s.gin_sender_state.slots[slot].task;
      if (task.channel < a.gin_rings.dispatch_channels)
        pack_dispatch(task);
      else
        pack_return(task);
      copy_sync<Storage::kCopyThreads>();
      if (tid == 0U)
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
            s.gin_sender_state.slots[slot].pending)
            .store(2U, ::cuda::memory_order_release);
      copy_sync<Storage::kCopyThreads>();
    }
  }
  domain_sync<Storage::kThreads>();
}

} // namespace communication_worker
#endif
