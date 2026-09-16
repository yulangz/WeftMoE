#pragma once
#include <type_traits>

#if __has_include(<nccl_device.h>)

#ifndef WEFTMOE_FWD_DISPATCH_ISSUER_BUFFERS
#define WEFTMOE_FWD_DISPATCH_ISSUER_BUFFERS 1
#endif

#include "cuda/communication/notification_pipeline.cuh"
#include "cuda/communication/return_queue.cuh"
#include "cuda/persistent_v3/forward/return_pipeline.cuh"

// Included after the private Forward contract, peer helpers and input-group
// publisher. Every CTA owns an independent communication domain; no helper in
// this file waits for the math threads or reuses their operand storage.
namespace forward_unified_transport {

using CopyBarrier = ::cuda::barrier<::cuda::thread_scope_block>;
#ifndef WEFTMOE_FWD_LOCAL_BATCHES_PER_GIN_DISPATCH
#define WEFTMOE_FWD_LOCAL_BATCHES_PER_GIN_DISPATCH 1
#endif
inline constexpr unsigned kLocalBatchesPerGinDispatch =
    WEFTMOE_FWD_LOCAL_BATCHES_PER_GIN_DISPATCH;
static_assert(kLocalBatchesPerGinDispatch > 0U);
#ifndef WEFTMOE_FWD_DEFER_LOCAL_RETURN_DURING_DISPATCH
#define WEFTMOE_FWD_DEFER_LOCAL_RETURN_DURING_DISPATCH 0
#endif



#include "cuda/persistent_v3/forward/unified_gin_state.cuh"

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
  static constexpr unsigned kDispatchBuffers = DispatchLanes == 1U
      ? Buffers : WEFTMOE_FWD_DISPATCH_ISSUER_BUFFERS;
  static_assert(kDispatchBuffers == 1U || kDispatchBuffers == 2U);
  static constexpr unsigned kStorageSlots =
      Buffers > DispatchLanes * kDispatchBuffers
          ? Buffers : DispatchLanes * kDispatchBuffers;
  static constexpr unsigned kThreads = Threads;
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
  unsigned pack_mailbox; // 0: available, 1: immutable task ready, 2: stop.
  unsigned pack_observed; // Copy-domain broadcast; never written by control.

  GinSenderState gin_sender_state;
  activation_ring_detail::Transfer gin_task;
  unsigned gin_task_slot;
  unsigned gin_rx_dispatch;
  unsigned gin_rx_return;
  unsigned gin_progressed;
  unsigned local_dispatch_since_receive;
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

__device__ __forceinline__ bool
cancelled(const ForwardKernelArgs &a,
          persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

// One elected CTA publishes the generation-local clear. Communication-only
// barriers transfer its setup acquire to every copy warp without involving
// any math thread. All peer reverse maps must be clear before Dispatch stores.
template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void
init(const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
     Storage &s) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (warp < Storage::kCopyWarps && lane == 0U) {
    init(&s.full[warp], 1U);
    // Each slot keeps its own arrival phase across row batches. Initialize
    // once; the existing proxy fence and domain join cover both arrays.
#pragma unroll
    for (unsigned slot = 0U; slot < Storage::kStorageSlots; ++slot)
      init(&s.row_prefetch_full[warp * Storage::kStorageSlots + slot], 1U);
  }
  if (tid == 0U) {
    const unsigned rdma_ctas = gridDim.x - a.forward_relay_cta_count;
    const unsigned relay = blockIdx.x >= rdma_ctas ? blockIdx.x - rdma_ctas : 0U;
    s.dispatch_cursor = 1ULL * relay * a.forward_dispatch_publish_batch_rows;
    s.pack_mailbox = s.pack_observed = 0U;
    s.gin_sender_state = GinSenderState{};
    // local_route_order places local routes before remote ones.
    s.dispatch_end = forward_dispatch_detail::local_route_prefix(a);
    s.return_queue = 0U;
    s.return_rows = 0U;
    s.dispatch_complete = s.stop = s.copies_done = 0U;
    s.gin_rx_dispatch = s.gin_rx_return = blockIdx.x;
    s.gin_progressed = 0U;
    s.local_dispatch_since_receive = kLocalBatchesPerGinDispatch;
    s.gin_task.channel = activation_ring_detail::kNoChannel;
    if (blockIdx.x == 0U) {
      __threadfence_system();
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          control.generation.contexts[0].rx_published.value)
          .store(execution_epoch(a), ::cuda::memory_order_release);
    }
  }
  ::cuda::device::experimental::fence_proxy_async_shared_cta();
  domain_sync<Storage::kThreads>();
  if (tid == 0U) {
    for (unsigned local = 0U; local < a.local_world_size; ++local) {
      auto *peer_control = peer_at<persistent_v3::ForwardPipelineControl>(
          a.peer_bases[a.node * a.local_world_size + local],
          a.forward_v3_control_offset);
      while (
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
              peer_control->generation.contexts[0].rx_published.value)
                  .load(::cuda::memory_order_acquire) < execution_epoch(a) &&
          !cancelled(a, control))
        __nanosleep(forward_tuning::kSetupNanosleep);
    }
  }
  domain_sync<Storage::kThreads>();
}

// A contiguous row is split into a bounded shared-memory footprint. All
// payloads use bulk TMA, including wide rows. Ordinary instructions move only
// metadata. The issuing lane drains global completion before its caller may
// publish readiness; wait_group_read alone would only permit SMEM reuse.
template <class Storage>
__device__ __forceinline__ void copy_row(Storage &s, unsigned warp,
                                         const void *source, void *target,
                                         unsigned bytes) {
  auto *buffer = s.staging + warp * Storage::kStorageSlots * Storage::kSegmentBytes;
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
  auto *buffer = s.staging + warp * Storage::kStorageSlots * Storage::kSegmentBytes;
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

#include "cuda/persistent_v3/forward/row_prefetch_pipeline.cuh"

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

template <bool Fence = true>
__device__ __forceinline__ void finish_copies() {
  ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
  if constexpr (Fence)
    __threadfence_system();
}

// Copy warps can publish disjoint contiguous input sub-batches independently.
// The two final domain joins protect the shared cursor. GIN's conservative
// policy retains the original interleaved rows and completion barrier.
template <unsigned BaseThread, unsigned ActiveCopyWarps = 0U,
          unsigned SlotOffset = 0U, unsigned BarrierId = 8U,
          bool Optimize = true, class Storage>
__device__ __forceinline__ bool progress_dispatch(const ForwardKernelArgs &a,
                                                  Storage &s) {
  constexpr auto warps =
      ActiveCopyWarps ? ActiveCopyWarps : Storage::kCopyWarps;
  constexpr auto threads = warps * 32U;
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  const auto batch_begin = s.dispatch_cursor;
  if (batch_begin >= s.dispatch_end)
    return false;
  const auto batch_end =
      min(batch_begin + a.forward_dispatch_publish_batch_rows, s.dispatch_end);
  const auto warp_rows =
      (a.forward_dispatch_publish_batch_rows + warps - 1U) / warps;
  const auto begin = Optimize
                         ? min(batch_begin + warp * warp_rows, batch_end)
                         : batch_begin;
  const auto end = Optimize ? min(begin + warp_rows, batch_end) : batch_end;
  const unsigned metadata_tid = Optimize ? lane : tid;
  constexpr unsigned metadata_stride = Optimize ? 32U : threads;
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ProxyRelayStart, ProxyRelay,
                             blockIdx.x, batch_end - batch_begin,
                             execution_epoch(a));
  unsigned peer_rows = 0U;
  if (begin < end) {
    for (auto item = begin + metadata_tid; item < end; item += metadata_stride) {
      const auto route = a.dispatch_waves.local_route_order[item];
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      const auto token = route / a.dispatch.identity.top_k;
      auto *base = a.peer_bases[destination];
      peer_at<LocalP2pRouteHeader>(base, a.direct_headers_offset)[row] =
          LocalP2pRouteHeader{
              transport::kActivationTransportSchemaVersion,
              a.rank, destination, 0U, a.dispatch.identity.generation,
              execution_epoch(a), forward_contribution_row(a, route), route, token, row,
              a.dispatch.expert_id[route],
              static_cast<unsigned>(route % a.dispatch.identity.top_k)};
      peer_at<float>(base, a.canonical_gates_offset)[row] = a.gates[route];
      peer_rows += destination != a.rank;
    }
    if constexpr (Optimize)
      __syncwarp();
    if (lane < Storage::kDispatchLanes) {
      if constexpr (Storage::kDispatchLanes == 1U) {
        RowPrefetchPipeline<Storage,
                            Optimize && forward_tuning::kUnifiedRowPrefetch>
            row_pipeline(s, warp + SlotOffset);
        for (auto item = begin + (Optimize ? 0U : warp); item < end;
             item += Optimize ? 1U : warps) {
          const auto route = a.dispatch_waves.local_route_order[item];
          const auto destination = a.dispatch.route_dst_rank[route];
          auto *target = peer_at<transport::BFloat16>(a.peer_bases[destination],
                                                      a.canonical_rows_offset);
          row_pipeline.copy(
              a.input + (route / a.dispatch.identity.top_k) * a.hidden,
              target + a.dispatch.route_dst_row[route] * a.hidden,
              unsigned(a.hidden * sizeof(transport::BFloat16)));
        }
        row_pipeline.drain();
      } else {
        LaneStagingView<Storage> view(s, warp + SlotOffset, lane);
        RowPrefetchPipeline<LaneStagingView<Storage>> row_pipeline(view, 0U);
        for (auto item = begin + (Optimize ? lane : warp * Storage::kDispatchLanes + lane);
             item < end; item += (Optimize ? 1U : warps) * Storage::kDispatchLanes) {
          const auto route = a.dispatch_waves.local_route_order[item];
          const auto destination = a.dispatch.route_dst_rank[route];
          auto *target = peer_at<transport::BFloat16>(a.peer_bases[destination],
                                                     a.canonical_rows_offset);
          row_pipeline.copy(
              a.input + (route / a.dispatch.identity.top_k) * a.hidden,
              target + a.dispatch.route_dst_row[route] * a.hidden,
              unsigned(a.hidden * sizeof(transport::BFloat16)));
        }
        row_pipeline.drain();
      }
      // Every participating issuer completes its writes before readiness.
      finish_copies<false>();
    }
    __threadfence_system();
    if constexpr (Optimize)
      __syncwarp();
    else
      copy_sync<threads, BarrierId>();
    for (auto item = begin + metadata_tid; item < end; item += metadata_stride) {
      const auto route = a.dispatch_waves.local_route_order[item];
      local_notification_detail::Receiver<ForwardKernelArgs>::publish_dispatch(
          a, a.dispatch_waves.route_dst_mtile[route],
          a.dispatch.route_dst_rank[route] % a.local_world_size);
    }
    const auto peers = __reduce_add_sync(0xffffffffU, peer_rows);
    if (lane == 0U && peers)
      atomicAdd(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->actual_dispatch_peer_bytes,
                peers * a.hidden * sizeof(transport::BFloat16));
  }
  // No warp may advance the shared cursor until every warp has captured its
  // current sub-batch and published its completed copies. No Return channel
  // is touched by these independently progressing Dispatch warps.
  copy_sync<threads, BarrierId>();
  if (tid == 0U)
    s.dispatch_cursor +=
        1ULL * a.forward_relay_cta_count * a.forward_dispatch_publish_batch_rows;
  copy_sync<threads, BarrierId>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ProxyRelayEnd, ProxyRelay,
                             blockIdx.x, batch_end - batch_begin,
                             execution_epoch(a));
  return true;
}

// Single notification owner per CTA; copy warps may finish rows in parallel,
// and the independent publisher warp may concurrently consume incoming slots.
// This retains the existing bounded credit protocol and source-local reduce
// publication instead of introducing remote floating-point atomics.
template <unsigned BaseThread, unsigned CopyThreads = 0U,
          unsigned BarrierId = 8U, bool SingleConsumer = false,
          class Storage, class Args, class Fill>
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
  unsigned selected_peer = grouped ? (s.return_queue - 1U) % a.local_world_size : 0U;
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
        local_notification_detail::wait_credit<Args, !SingleConsumer>(
            a, peer, sequence);
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

template <unsigned BaseThread, unsigned ActiveCopyWarps = 0U,
          unsigned SlotOffset = 0U, unsigned BarrierId = 8U,
          bool Optimize = true, bool IndependentDispatch = false, class Storage>
__device__ __forceinline__ bool progress_return(const ForwardKernelArgs &a,
                                                Storage &s) {
  constexpr auto warps =
      ActiveCopyWarps ? ActiveCopyWarps : Storage::kCopyWarps;
  constexpr auto threads = warps * 32U;
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (tid == 0U) {
    // A nonblocking work claim does not lock a GIN context or an NVLink
    // channel. Wire queues retain their single producer and consumer.
    // Independent direction domains never read each other's live cursor.
    // A shared copy domain can enlarge Return only after its Dispatch ends.
    unsigned claim_limit = a.forward_local_return_batch_rows;
    if constexpr (Optimize && !IndependentDispatch)
      if (s.dispatch_cursor >= s.dispatch_end)
        claim_limit = max(claim_limit, forward_tuning::kUnifiedReturnDrainRows);
    s.return_rows = 0U;
    const bool peer_queues = return_queue_detail::local_peer_queues(a);
    const auto first = peer_queues ? 1U + a.node * a.local_world_size : 0U;
    const auto count = peer_queues ? a.local_world_size : 1U;
    const auto preferred = (a.rank + blockIdx.x) % count;
    for (unsigned probe = 0U; probe < count; ++probe) {
      s.return_queue = first + (preferred + probe) % count;
      s.return_cursor = forward_return_detail::claim_rows(
          a, return_queue_detail::queues(a)[s.return_queue],
          claim_limit, s.return_rows);
      if (s.return_rows)
        break;
      // Match the established peer affinity: while the preferred finite
      // queue is still producing, an empty poll or a lost CAS should not
      // send every CTA into the other peers' queues. Once its exact capacity
      // has been claimed, these workers may steal remaining peer work.
      if (Optimize && peer_queues && probe == 0U &&
          persistent_v3::load_acquire<::cuda::thread_scope_device>(
              return_queue_detail::queues(a)[s.return_queue].head) !=
              return_queue_detail::queues(a)[s.return_queue].capacity)
        break;
    }
  }
  copy_sync<threads, BarrierId>();
  const unsigned rows = s.return_rows;
  // Every copy warp must capture this result before an empty poll can return
  // and let the leader overwrite the shared descriptor on its next poll.
  copy_sync<threads, BarrierId>();
  if (!rows)
    return false;
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, LocalReturnBatchStart,
                             LocalReturn, blockIdx.x, rows, execution_epoch(a));
  const auto cursor = s.return_cursor;
  const auto *entries =
      peer_at<const unsigned>(a.local_base, a.forward_return_rows_offset) +
      return_queue_detail::queues(a)[s.return_queue].begin;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *output = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (tid == 0U)
    atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
  if (lane == 0U) {
    RowPrefetchPipeline<Storage,
                        Optimize && forward_tuning::kUnifiedRowPrefetch>
        row_pipeline(s, warp + SlotOffset);
    for (unsigned item = warp; item < rows; item += warps) {
      const auto row = entries[cursor + item];
      const auto header = headers[row];
      auto *target = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.contribution_offset);
      row_pipeline.copy(output + row * a.hidden,
                        target + header.transport_index * a.hidden,
                        unsigned(a.hidden * sizeof(transport::BFloat16)));
    }
    row_pipeline.drain();
    finish_copies();
  }
  copy_sync<threads, BarrierId>();
  publish_return<BaseThread, threads, BarrierId, true>(
      a, s, rows,
      [&](unsigned item) {
        const auto header = headers[entries[cursor + item]];
        return local_notification_detail::Entry{
            header.source_rank % a.local_world_size, header.route_index};
      },
      return_queue_detail::local_peer_queues(a));
  unsigned peer_rows = 0U;
  for (unsigned item = tid; item < rows; item += threads)
    peer_rows += headers[entries[cursor + item]].source_rank != a.rank;
  const auto peers = __reduce_add_sync(0xffffffffU, peer_rows);
  if (lane == 0U && peers)
    atomicAdd(&status->actual_return_peer_bytes,
              1ULL * peers * a.hidden * sizeof(transport::BFloat16));
  copy_sync<threads, BarrierId>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, LocalReturnBatchEnd,
                             LocalReturn, blockIdx.x, rows, execution_epoch(a));
  return true;
}

// Like the established local notification drain, but the warp may begin at a
// nonzero thread offset. End chunks belong only to this CTA's outgoing stream.
template <unsigned BaseThread, bool AssistReceive = true, class Args, class Cancelled>
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
    local_notification_detail::wait_credit<Args, AssistReceive>(a, peer, sequence);
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

template <unsigned BaseThread>
__device__ __forceinline__ void
drain_notifications(const ForwardKernelArgs &a) {
  drain_notifications<BaseThread, false>(a, [&] { return forward_cancelled(a); });
}

#include "cuda/persistent_v3/forward/unified_gin.cuh"

// Alternate finite FC1 group attempts with Return receives. Missing Dispatch
// input must never prevent this warp from returning notification credits.
template <class Storage>
__device__ __forceinline__ void
run_publisher(const ForwardKernelArgs &a,
              persistent_v3::ForwardPipelineControl &control,
              unsigned long long *scheduler, Storage &s) {
  const unsigned lane = threadIdx.x % 32U;
  forward_dispatch_detail::Fc1PublisherState publisher;
  if (lane == 0U)
    forward_dispatch_detail::init_fc1_publisher(
        a, blockIdx.x - (gridDim.x - a.forward_relay_cta_count),
        a.forward_relay_cta_count, scheduler, publisher);
  __syncwarp();
  while (true) {
    unsigned done = 0U, published = 0U;
    if (lane == 0U) {
#if WEFTMOE_DEVICE_TIMELINE
      gin_detail::observe_return_ready(a);
#endif
      published = forward_dispatch_detail::progress_fc1_publisher(
          a, scheduler, publisher);
      done = (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                  s.copies_done).load(::cuda::memory_order_acquire) != 0U &&
              publisher.pending.empty()) || cancelled(a, control);
    }
    if (__shfl_sync(0xffffffffU, done, 0U))
      break;
    const bool received = local_notification_detail::progress<ForwardKernelArgs, true>(a);
    if (!__shfl_sync(0xffffffffU, published, 0U) && !received)
      __nanosleep(50U);
  }
  if (lane == 0U)
    forward_dispatch_detail::finish_fc1_publisher(a, blockIdx.x);
}

// A CTA-local mailbox lets packing overlap GIN submission. The submitter
// is the sole owner of its channels and state; GIN arbitrates shared contexts.
template <unsigned BaseThread, class Args, class Storage, class Cancelled,
          class PackDispatch, class PackReturn, class OnSend, class OnComplete>
__device__ __forceinline__ void
run_sender(const Args &a, Storage &s, Cancelled is_cancelled,
           PackDispatch pack_dispatch, PackReturn pack_return,
           OnSend on_send, OnComplete on_complete) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (warp == Storage::kCopyWarps) {
    if (lane == 0U) {
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block> mailbox(s.pack_mailbox);
      while (!is_cancelled()) {
        const bool available = mailbox.load(::cuda::memory_order_acquire) == 0U;
        activation_ring_detail::Transfer task;
        unsigned slot = 0U;
#if WEFTMOE_DEVICE_TIMELINE
        const auto progress_start = profiling::read_globaltimer_ns();
#endif
        const bool progressed =
            gin_detail::progress_sender<(forward_tuning::kGinChannelPackLimit > 1U)>(
                a, s.gin_sender_state, slot, task,
                on_send, on_complete, available);
        if (task.channel != activation_ring_detail::kNoChannel) {
#if WEFTMOE_DEVICE_TIMELINE
          if constexpr (::std::is_same_v<Args, ForwardKernelArgs>)
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, GinSenderProgress,
                GinReturn, task.channel, profiling::read_globaltimer_ns() - progress_start,
                task.rows);
#endif
          s.gin_task = task;
          s.gin_task_slot = slot;
          mailbox.store(1U, ::cuda::memory_order_release);
        } else if (available && s.gin_sender_state.done) {
          break;
        }
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
        s.pack_observed = is_cancelled() ? 2U :
            ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.pack_mailbox)
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
      if (s.gin_task.channel < a.gin_rings.dispatch_channels)
        pack_dispatch(s.gin_task);
      else
        pack_return(s.gin_task);
      // Every copy warp has finished accessing the shared descriptor before
      // returning it to the independent controller.
      copy_sync<Storage::kCopyThreads>();
      if (tid == 0U) {
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
            s.gin_sender_state.slots[s.gin_task_slot].pending)
            .store(2U, ::cuda::memory_order_release);
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.pack_mailbox)
            .store(0U, ::cuda::memory_order_release);
      }
      copy_sync<Storage::kCopyThreads>();
    }
  }
  domain_sync<Storage::kThreads>();
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void
run_rdma(const ForwardKernelArgs &a,
         persistent_v3::ForwardPipelineControl &control, Storage &s) {
  run_sender<BaseThread>(a, s, [&] { return cancelled(a, control); },
      [&](const activation_ring_detail::Transfer &task) {
        gin_detail::pack_dispatch<BaseThread>(a, s, task);
      },
      [&](const activation_ring_detail::Transfer &task) {
        gin_detail::pack_return<BaseThread>(a, s, task);
      },
      [&](const activation_ring_detail::Transfer &task) { gin_detail::account_send(a, task); },
      [&](unsigned sender, GinSenderState &state) {
        gin_detail::complete_sender(a, control, sender, state);
      });
}

template <bool UseGin, unsigned BaseThread, class Storage>
__device__ __forceinline__ void
run_impl(const ForwardKernelArgs &a,
         persistent_v3::ForwardPipelineControl &control,
         unsigned long long *scheduler, Storage &s) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U;
  init<BaseThread>(a, control, s);
  const unsigned rdma_ctas = gridDim.x - a.forward_relay_cta_count;
  if constexpr (UseGin) {
    if (blockIdx.x < rdma_ctas) {
      run_rdma<BaseThread>(a, control, s);
      return;
    }
  }
  if (warp == Storage::kCopyWarps) {
    run_publisher(a, control, scheduler, s);
  } else {
    while (true) {
      // Capture before the existing join: the leader must not update the
      // phase at loop end while a slower warp is still deciding whether to
      // enter the conditional Return helper's named barriers.
      const bool local_return_eligible =
          !UseGin || !WEFTMOE_FWD_DEFER_LOCAL_RETURN_DURING_DISPATCH ||
          s.dispatch_complete != 0U;
      copy_sync<Storage::kCopyThreads>();
      bool remote_progress = false;
      if constexpr (UseGin) {
        remote_progress = gin_detail::progress_receive<BaseThread>(a, s, false);
        // A received token can fan out to several local routes. Bound its
        // share of the copy domain without enlarging local batches or delaying
        // Return service. Empty receives keep polling; once local work ends,
        // every iteration can drain received Dispatch again.
        if (s.local_dispatch_since_receive >= kLocalBatchesPerGinDispatch ||
            s.dispatch_cursor >= s.dispatch_end) {
          const bool received_dispatch = gin_detail::progress_receive<BaseThread>(a, s, true);
          if (received_dispatch && tid == 0U)
            s.local_dispatch_since_receive = 0U;
          remote_progress = received_dispatch || remote_progress;
        }
        if (rdma_ctas == 0U)
          remote_progress = gin_detail::progress_send<BaseThread>(a, control, s) ||
                            remote_progress;
      }
      constexpr bool optimize_local =
          !UseGin || forward_tuning::kUnifiedOptimizeGinLocalTransport;
      const bool dispatched =
          progress_dispatch<BaseThread, 0U, 0U, 8U, optimize_local>(a, s);
      // Local Return entries live in finite per-generation storage: compute
      // producers do not need these consumers to finish Dispatch. Keep GIN
      // Return receives above eligible on every iteration, including while
      // received input fanout still has work to publish.
      const bool returned =
          local_return_eligible
              ? progress_return<BaseThread, 0U, 0U, 8U, optimize_local>(a, s)
              : false;
      if (tid == 0U) {
        if (dispatched && s.local_dispatch_since_receive < kLocalBatchesPerGinDispatch)
          ++s.local_dispatch_since_receive;
        if (!s.dispatch_complete && s.dispatch_cursor >= s.dispatch_end &&
            (!UseGin ||
             activation_ring_detail::receive_complete(a, true))) {
          s.dispatch_complete = 1U;
          const auto previous =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control.relay_dispatch_roles_complete.value, 1ULL);
          if (previous + 1ULL == a.forward_relay_cta_count) {
            peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                ->dispatch_complete_clock = device_globaltimer_ns();
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, XDispatchComplete,
                                       ProxyRelay, 0U, s.dispatch_end,
                                       execution_epoch(a));
          }
        }
        forward_return_detail::LocalReturnState state;
        state.queue_index = s.return_queue;
        s.stop = cancelled(a, control) ||
                 (s.dispatch_complete &&
                  forward_return_detail::local_return_terminal(a, state) &&
                  (!UseGin ||
                   (activation_ring_detail::receive_complete(
                        a, false) &&
                    persistent_v3::load_acquire<::cuda::thread_scope_device>(
                        control.rdma_return_roles_complete.value) ==
                        a.gin_sender_count)));
      }
      copy_sync<Storage::kCopyThreads>();
      if (s.stop)
        break;
      if (!dispatched && !returned && !remote_progress)
        __nanosleep(forward_tuning::kTransportIdleNanosleep);
    }
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, LocalReturnTerminal,
                               LocalReturn, blockIdx.x, s.return_cursor,
                               execution_epoch(a));
    if (warp == 0U)
      drain_notifications<BaseThread>(a);
    copy_sync<Storage::kCopyThreads>();
    if (tid == 0U) {
      if constexpr (UseGin)
        activation_ring_detail::flush_receive(a);
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->return_complete_clock,
                device_globaltimer_ns());
      if constexpr (!UseGin)
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.rdma_return_roles_complete.value, 1ULL);
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.return_receive_roles_complete.value, 1ULL);
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.copies_done)
          .store(1U, ::cuda::memory_order_release);
    }
  }
  domain_sync<Storage::kThreads>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, TransportRoleDone,
                             LocalReturn, blockIdx.x, 0ULL, execution_epoch(a));
}

// Explicit thread-count spelling makes call sites auditable while preserving
// arbitrary segment/buffer tuning in the independent shared allocation.
template <unsigned Threads, unsigned BaseThread, unsigned SegmentBytes,
          unsigned Buffers, unsigned DispatchLanes>
__device__ __forceinline__ void
run(const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    unsigned long long *scheduler,
    Shared<Threads, SegmentBytes, Buffers, DispatchLanes> &storage) {
  run_impl<false, BaseThread>(a, control, scheduler, storage);
}

template <unsigned Threads, unsigned BaseThread, unsigned SegmentBytes,
          unsigned Buffers, unsigned DispatchLanes>
__device__ __forceinline__ void
run_gin(const ForwardKernelArgs &a,
        persistent_v3::ForwardPipelineControl &control,
        unsigned long long *scheduler,
        Shared<Threads, SegmentBytes, Buffers, DispatchLanes> &storage) {
  run_impl<true, BaseThread>(a, control, scheduler, storage);
}

#include "cuda/persistent_v3/forward/unified_local_warps.cuh"

} // namespace forward_unified_transport

#endif // __has_include(<nccl_device.h>)
