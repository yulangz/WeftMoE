#pragma once

#if __has_include(<nccl_device.h>)

#include "cuda/persistent_v3/forward/unified_transport.cuh"

// EP-local Backward payloads share the bounded TMA copier and notification
// protocol with Forward. Only queue adapters and readiness publication differ.
namespace backward_unified_transport {

template <unsigned Threads, unsigned SegmentBytes = 2048U,
          unsigned Buffers = 1U>
struct alignas(128) Shared
    : forward_unified_transport::Shared<Threads, SegmentBytes, Buffers> {
  using Base =
      forward_unified_transport::Shared<Threads, SegmentBytes, Buffers>;
  unsigned queue_scratch[Base::kCopyWarps];
  unsigned queue_selected;
  unsigned queue_rows;
  unsigned long long queue_base;
};

#include "cuda/persistent_v3/backward/unified_gin.cuh"

template <bool UseGin, unsigned BaseThread, class Storage>
__device__ __forceinline__ void init(const BackwardKernelArgs &a, Storage &s) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (warp < Storage::kCopyWarps && lane == 0U)
    init(&s.full[warp], 1U);
  if (tid == 0U) {
    s.dispatch_cursor = 1ULL * (blockIdx.x - (UseGin ? a.gin_receiver_base : 0U)) * backward_tuning::kDispatchBatchRows;
    s.pack_mailbox = s.pack_observed = 0U;
    s.gin_sender_state = forward_unified_transport::GinSenderState{};
    s.dispatch_end = backward_pipeline_detail::local_route_count(a);
    s.return_cursor = ~0ULL;
    s.return_queue = s.return_rows = 0U;
    s.dispatch_complete = s.stop = s.copies_done = 0U;
    if constexpr (UseGin) {
      s.gin_rx_dispatch = blockIdx.x % a.gin_rings.dispatch_channels;
      s.gin_rx_return = blockIdx.x % a.gin_rings.return_channels;
      s.gin_task.channel = activation_ring_detail::kNoChannel;
    }
    atomicMin(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                   ->start_clock,
              device_globaltimer_ns());
  }
  ::cuda::device::experimental::fence_proxy_async_shared_cta();
  forward_unified_transport::domain_sync<Storage::kThreads>();
}

template <unsigned BaseThread, class Storage, class Layout>
__device__ __forceinline__ bool
progress_dispatch(const BackwardKernelArgs &a, unsigned long long *words,
                  const Layout &layout, Storage &s) {
  constexpr auto threads = Storage::kCopyThreads;
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  const auto begin = s.dispatch_cursor;
  if (begin >= s.dispatch_end)
    return false;
  const unsigned batch = static_cast<unsigned>(
      min(1ULL * backward_tuning::kDispatchBatchRows, s.dispatch_end - begin));
  if (lane == 0U) {
    for (unsigned item = warp; item < batch; item += Storage::kCopyWarps) {
      const auto route = a.dispatch_waves.local_route_order[begin + item];
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto token = route / a.dispatch.identity.top_k;
      auto *target = peer_at<transport::BFloat16>(a.peer_bases[destination],
                                                  a.compact_dr_offset);
      forward_unified_transport::copy_row(
          s, warp, a.dy + token * a.hidden,
          target + a.dispatch.route_dst_row[route] * a.hidden,
          unsigned(a.hidden * sizeof(transport::BFloat16)));
    }
    forward_unified_transport::finish_copies();
  }
  forward_unified_transport::copy_sync<threads>();
  for (unsigned item = tid; item < batch; item += threads) {
    const auto route = a.dispatch_waves.local_route_order[begin + item];
    local_notification_detail::Receiver<BackwardKernelArgs>::publish_dispatch(
        a, a.dispatch.route_dst_row[route],
        a.dispatch.route_dst_rank[route] % a.local_world_size);
  }
  forward_unified_transport::copy_sync<threads>();
  if (tid == 0U) {
    auto *counter = reinterpret_cast<persistent_v3::PaddedCounter *>(
        words + layout.dispatched_routes);
    const auto done =
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            counter->value)
            .fetch_add(batch, ::cuda::memory_order_acq_rel) +
        batch;
    if (done == s.dispatch_end)
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->dispatch_complete_clock,
                device_globaltimer_ns());
    s.dispatch_cursor += 1ULL * (a.use_gin ? a.gin_receiver_count : gridDim.x) * backward_tuning::kDispatchBatchRows;
  }
  forward_unified_transport::copy_sync<threads>();
  return true;
}

template <bool UseGin, unsigned BaseThread, class Storage, class Layout>
__device__ __forceinline__ bool
progress_return(const BackwardKernelArgs &a, unsigned long long *words,
                const Layout &layout, std::uint32_t mblocks, Storage &s) {
  using namespace backward_pipeline_detail;
  constexpr auto threads = Storage::kCopyThreads;
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (tid == 0U) {
    s.return_cursor = ~0ULL;
    TileTask task{};
    if (persistent_v3::backward::try_claim_domain_once(
            make_return_domain(a, words, layout, mblocks), task)) {
      auto expected = 3ULL;
      if (!::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
               words[layout.phase + task.owner])
               .compare_exchange_strong(expected, 4ULL,
                                        ::cuda::memory_order_acq_rel,
                                        ::cuda::memory_order_relaxed))
        __trap(); // Each complete M-block has one immutable Return owner.
      s.return_cursor = task.owner;
      atomicMin(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->reduce_blocks_done,
                device_globaltimer_ns());
    }
  }
  forward_unified_transport::copy_sync<threads>();
  if (s.return_cursor == ~0ULL)
    return false;
  const auto *descriptors = reinterpret_cast<const BackwardMblockDescriptor *>(
      words + layout.descriptors);
  const auto descriptor = descriptors[s.return_cursor];
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *dx = a.dx_rows;
  const auto *dgate = a.dgate_rows;
  if (lane == 0U) {
    for (unsigned row = warp; row < descriptor.valid_rows;
         row += Storage::kCopyWarps) {
      const auto header = headers[descriptor.compact_row + row];
      if constexpr (UseGin)
        if (header.source_rank / a.local_world_size != a.node)
          continue;
      auto *target = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.contribution_offset);
      forward_unified_transport::copy_row(
          s, warp, dx + (1ULL * descriptor.padded_row + row) * a.hidden,
          target + header.route_index * a.hidden,
          unsigned(a.hidden * sizeof(transport::BFloat16)));
      peer_at<float>(a.peer_bases[header.source_rank],
                     a.dgate_output_offset)[header.route_index] =
          dgate[descriptor.padded_row + row];
    }
    forward_unified_transport::finish_copies();
  }
  forward_unified_transport::copy_sync<threads>();
  forward_unified_transport::publish_return<BaseThread>(
      a, s, descriptor.valid_rows, [&](unsigned item) {
        const auto header = headers[descriptor.compact_row + item];
        return local_notification_detail::Entry{
            header.source_rank / a.local_world_size == a.node
                ? header.source_rank % a.local_world_size
                : communication::kLocalPeers,
            header.route_index};
      });
  if constexpr (UseGin)
    gin_detail::publish_return_rows<BaseThread>(a, s, descriptor.compact_row,
                                                descriptor.valid_rows);
  if (tid == 0U) {
    auto *counter = reinterpret_cast<persistent_v3::PaddedCounter *>(
        words + layout.returned_mblocks);
    const auto done =
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            counter->value)
            .fetch_add(1ULL, ::cuda::memory_order_acq_rel) +
        1ULL;
    if (done == mblocks)
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->return_complete_clock,
                device_globaltimer_ns());
  }
  forward_unified_transport::copy_sync<threads>();
  return true;
}

template <bool UseGin, unsigned BaseThread, class Storage>
__device__ __forceinline__ void
run(const BackwardKernelArgs &a,
    const backward_unified_compute_detail::RowGeometry &geometry, Storage &s) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  auto *words = backward_pipeline_detail::scheduler_words(a);
  const auto layout = persistent_v3::backward::make_backward_scheduler_layout(
      a.scheduler_mblock_capacity,
      unsigned(a.hidden / backward_tuning::kDxTileN /
               backward_tuning::kActivationClusterN),
      a.scheduler_expert_capacity);
  init<UseGin, BaseThread>(a, s);
  if constexpr (UseGin) {
    if (blockIdx.x < a.gin_sender_count) {
      forward_unified_transport::run_sender<BaseThread>(a, s, [] { return false; },
          [&](const activation_ring_detail::Transfer &task) {
            gin_detail::pack_dispatch<BaseThread>(a, s, task);
          },
          [&](const activation_ring_detail::Transfer &task) {
            gin_detail::pack_return<BaseThread>(a, s, task);
          },
          [&](const activation_ring_detail::Transfer &task) {
            if (task.channel >= a.gin_rings.dispatch_channels) {
              auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
              atomicAdd(&status->gin_return_put_count, 1ULL);
              atomicAdd(&status->gin_return_payload_bytes,
                        task.rows * (a.hidden * 2ULL + sizeof(float)));
            }
          }, [](unsigned, forward_unified_transport::GinSenderState &) {});
      return;
    }
  }
  const unsigned publisher = blockIdx.x - (UseGin ? a.gin_receiver_base : 0U);
  const unsigned publishers = UseGin ? a.gin_receiver_count : gridDim.x;
  if (warp == Storage::kCopyWarps) {
    // This warp never waits for a copy-domain barrier. It can publish dU work
    // and return NIC/NVLink notification credit while all copy warps are busy.
    // Partition retained M-blocks across CTA leaders instead of making every
    // CTA contend on one global scan counter. A descriptor has one publisher;
    // phase CAS and the original system-acquire readiness check are unchanged.
    unsigned scan_cursor = publisher;
    unsigned pending = publisher < geometry.mblocks
                           ? 1U + (geometry.mblocks - 1U - publisher) / publishers
                           : 0U;
    while (true) {
      // A publisher warp must enter/leave its full-mask collectives together.
      unsigned done = 0U;
      if (lane == 0U)
        done = ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                   s.copies_done).load(::cuda::memory_order_acquire);
      if (__shfl_sync(0xffffffffU, done, 0U))
        break;
      const bool progressed = local_notification_detail::progress(a);
      if (lane == 0U && pending) {
        if (backward_pipeline_detail::try_publish_dispatch_mblock(
                a, words, layout, geometry.mblocks, scan_cursor))
          --pending;
        scan_cursor += publishers;
        if (scan_cursor >= geometry.mblocks)
          scan_cursor = publisher;
      }
      __syncwarp();
      if (!progressed)
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
    }
  } else {
    while (true) {
      const bool dispatched =
          progress_dispatch<BaseThread>(a, words, layout, s);
      bool gin_progressed = false;
      if constexpr (UseGin) {
        gin_progressed = gin_detail::progress_receive<BaseThread>(a, s, true);
        gin_progressed =
            gin_detail::progress_receive<BaseThread>(a, s, false) ||
            gin_progressed;
      }
      const bool returned = progress_return<UseGin, BaseThread>(
          a, words, layout, geometry.mblocks, s);
      if (tid == 0U) {
        auto *counter = reinterpret_cast<persistent_v3::PaddedCounter *>(
            words + layout.returned_mblocks);
        s.stop =
            s.dispatch_cursor >= s.dispatch_end &&
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
                counter->value)
                    .load(::cuda::memory_order_acquire) == geometry.mblocks;
        if constexpr (UseGin)
          if (s.stop && !gin_detail::terminal(a))
            s.stop = 0U;
      }
      forward_unified_transport::copy_sync<Storage::kCopyThreads>();
      if (s.stop)
        break;
      if (!dispatched && !returned && !gin_progressed)
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
    }
    if (warp == 0U)
      forward_unified_transport::drain_notifications<BaseThread>(
          a, [] { return false; });
    forward_unified_transport::copy_sync<Storage::kCopyThreads>();
    if (tid == 0U) {
      if constexpr (UseGin)
        activation_ring_detail::flush_receive(a);
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.copies_done)
          .store(1U, ::cuda::memory_order_release);
    }
  }
  forward_unified_transport::domain_sync<Storage::kThreads>();
}

} // namespace backward_unified_transport

#endif // __has_include(<nccl_device.h>)
