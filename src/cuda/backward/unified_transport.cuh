#pragma once

#if __has_include(<nccl_device.h>)

#include "cuda/communication/worker_pipeline.cuh"

// EP-local Backward payloads share the bounded TMA copier and notification
// protocol with Forward. Only queue adapters and readiness publication differ.
namespace backward_unified_transport {

template <unsigned Threads, unsigned SegmentBytes = 2048U,
          unsigned Buffers = 1U>
struct alignas(128) Shared
    : communication_worker::Shared<Threads, SegmentBytes, Buffers> {
  using Base = communication_worker::Shared<Threads, SegmentBytes, Buffers>;
  unsigned queue_scratch[Base::kCopyWarps];
  unsigned queue_selected;
  unsigned queue_rows;
  unsigned long long queue_base;
  unsigned static_return_next;
  unsigned reduced_warps;
};

#include "cuda/backward/unified_gin.cuh"
#include "cuda/weftmoe_pf/backward_return_transport.cuh"

template <bool UseGin, unsigned BaseThread, class Storage>
__device__ __forceinline__ void init(const BackwardKernelArgs &a, Storage &s) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (warp < Storage::kCopyWarps && lane == 0U) {
    init(&s.full[warp], 1U);
    for (unsigned slot = 0U; slot < Storage::kStorageSlots; ++slot)
      init(&s.row_prefetch_full[warp * Storage::kStorageSlots + slot], 1U);
  }
  if (tid == 0U) {
    const auto worker = blockIdx.x - (UseGin ? a.gin_receiver_base : 0U);
    const auto workers = UseGin ? a.gin_receiver_count : gridDim.x;
    const auto rows = dispatch_sizes(a.dispatch).incoming_row_count;
    const auto batch = backward_tuning::kDispatchBatchRows;
    const auto span =
        (rows + workers * batch - 1ULL) / (workers * batch) * batch;
    s.dispatch_cursor = worker * span;
    s.pack_mailbox = s.pack_observed = 0U;
    s.gin_sender_state = communication_worker::GinSenderState{};
    s.dispatch_end = min(1ULL * rows, s.dispatch_cursor + span);
    s.return_cursor = ~0ULL;
    s.static_return_next = blockIdx.x - (UseGin ? a.gin_receiver_base : 0U);
    s.return_queue = s.return_rows = 0U;
    s.dispatch_complete = s.stop = s.copies_done = 0U;
    s.reduced_warps = 0U;
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
  communication_worker::domain_sync<Storage::kThreads>();
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
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const unsigned batch = static_cast<unsigned>(
      min(1ULL * backward_tuning::kDispatchBatchRows, s.dispatch_end - begin));
  if (lane == 0U) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    communication_worker::RowPrefetchPipeline<Storage> prefetch(s, warp);
    for (unsigned item = warp; item < batch; item += Storage::kCopyWarps) {
      const auto header = headers[begin + item];
      if (header.source_rank / a.local_world_size != a.node)
        continue;
      const auto token = header.route_index / a.dispatch.identity.top_k;
      const auto *source = peer_at<const transport::BFloat16>(
          a.peer_bases[header.source_rank], a.input_offset);
      prefetch.copy(source + token * a.hidden, a.dr + (begin + item) * a.hidden,
                    unsigned(a.hidden * sizeof(transport::BFloat16)));
    }
    prefetch.drain();
    communication_worker::finish_copies();
  }
  communication_worker::copy_sync<threads>();
  for (unsigned item = tid; item < batch; item += threads) {
    if (headers[begin + item].source_rank / a.local_world_size == a.node)
      local_notification_detail::Receiver<BackwardKernelArgs>::publish_dispatch(
          a, begin + item, a.rank % a.local_world_size);
  }
  communication_worker::copy_sync<threads>();
  if (tid == 0U) {
    auto *counter = reinterpret_cast<pipeline::PaddedCounter *>(
        words + layout.dispatched_routes);
    const auto done =
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            counter->value)
            .fetch_add(batch, ::cuda::memory_order_acq_rel) +
        batch;
    if (done == dispatch_sizes(a.dispatch).incoming_row_count)
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->dispatch_complete_clock,
                device_globaltimer_ns());
    s.dispatch_cursor += backward_tuning::kDispatchBatchRows;
  }
  communication_worker::copy_sync<threads>();
  return true;
}

template <bool UseGin, unsigned BaseThread, class Storage, class Layout>
__device__ __forceinline__ bool
progress_return(const BackwardKernelArgs &a, unsigned long long *words,
                const Layout &layout, std::uint32_t mblocks, Storage &s) {
  if (a.return_ring.depth)
    return progress_ring_return<UseGin, BaseThread>(a, words, layout, s);
  using namespace backward_pipeline_detail;
  constexpr auto threads = Storage::kCopyThreads;
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  if (tid == 0U) {
    s.return_cursor = ~0ULL;

    const auto owner = s.static_return_next;
    constexpr auto tiles = backward_tuning::kMainHidden /
                           backward_tuning::kDxTileN /
                           backward_tuning::kActivationClusterN;
    if (owner < mblocks &&
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            words[layout.dx_completed + owner])
                .load(::cuda::memory_order_acquire) ==
            pipeline::backward::backward_dx_ready_value(tiles)) {
      s.return_cursor = owner;
      s.static_return_next += UseGin ? a.gin_receiver_count : gridDim.x;
      atomicMin(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->reduce_blocks_done,
                device_globaltimer_ns());
    }
  }
  communication_worker::copy_sync<threads>();
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
      auto *target = a.peer_payloads[header.source_rank].contribution;
      communication_worker::copy_row(
          s, warp, dx + (1ULL * descriptor.padded_row + row) * a.hidden,
          target + header.route_index * a.hidden,
          unsigned(a.hidden * sizeof(transport::BFloat16)));
      a.peer_payloads[header.source_rank].dgate[header.route_index] =
          dgate[descriptor.padded_row + row];
    }
    communication_worker::finish_copies();
  }
  communication_worker::copy_sync<threads>();
  communication_worker::publish_return<BaseThread>(
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
    auto *counter = reinterpret_cast<pipeline::PaddedCounter *>(
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
  communication_worker::copy_sync<threads>();
  return true;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool
progress_reduce(const BackwardKernelArgs &a, Storage &s, std::uint64_t &token) {
  const auto warp = (threadIdx.x - BaseThread) / 32U;
  const auto lane = threadIdx.x % 32U;
  constexpr auto warps = Storage::kThreads / 32U;
  const auto first = weftmoe_pf::backward_return::reduce_first(a, warps, warp);
  const auto start =
      lane == 0U && token == first ? device_globaltimer_ns() : 0ULL;
  if (!backward_reduce_dx_token<false>(a, token))
    return false;
  __syncwarp();
  if (start)
    atomicMin(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                   ->backward_reduce_start_clock,
              start);
  token +=
      std::uint64_t(weftmoe_pf::backward_return::reduce_workers(a)) * warps;
  if (a.return_ring.depth && token >= a.dispatch.identity.num_tokens &&
      lane == 0U)
    ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.reduced_warps)
        .fetch_add(1U, ::cuda::memory_order_release);
  return true;
}

template <bool UseGin, unsigned BaseThread, class Storage>
__device__ __forceinline__ void
run(const BackwardKernelArgs &a,
    const backward_unified_compute_detail::RowGeometry &geometry, Storage &s,
    std::uint64_t &reduce_token) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  auto *words = backward_pipeline_detail::scheduler_words(a);
  const auto layout = pipeline::backward::make_backward_scheduler_layout(
      a.scheduler_mblock_capacity,
      unsigned(a.hidden / backward_tuning::kDxTileN /
               backward_tuning::kActivationClusterN),
      a.scheduler_expert_capacity);
  init<UseGin, BaseThread>(a, s);
  if constexpr (UseGin) {
    if (blockIdx.x < a.gin_sender_count) {
      communication_worker::run_sender<BaseThread>(
          a, s, [] { return false; },
          [&](const activation_ring_detail::Transfer &task) {
            gin_detail::pack_dispatch<BaseThread>(a, s, task);
          },
          [&](const activation_ring_detail::Transfer &task) {
            gin_detail::pack_return<BaseThread>(a, s, task);
          },
          [&](const activation_ring_detail::Transfer &task) {
            if (task.channel >= a.gin_rings.dispatch_channels) {
              auto *status =
                  peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
              atomicAdd(&status->gin_return_put_count, 1ULL);
              atomicAdd(&status->gin_return_payload_bytes,
                        task.rows * (a.hidden * 2ULL + sizeof(float)));
              if (task.terminal)
                atomicMax(&status->backward_return_send_complete_clock,
                          device_globaltimer_ns());
            }
          },
          [](unsigned, communication_worker::GinSenderState &) {});
      if (tid == 0U)
        atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                       ->backward_sender_complete_clock,
                  device_globaltimer_ns());
      return;
    }
  }
  const unsigned publisher = blockIdx.x - (UseGin ? a.gin_receiver_base : 0U);
  const unsigned publishers = UseGin ? a.gin_receiver_count : gridDim.x;
  if (a.return_ring.depth && reduce_token >= a.dispatch.identity.num_tokens &&
      lane == 0U)
    ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.reduced_warps)
        .fetch_add(1U, ::cuda::memory_order_release);
  if (warp == Storage::kCopyWarps) {
    // This warp never waits for a copy-domain barrier. It can publish dU work
    // and return NIC/NVLink notification credit while all copy warps are busy.
    // Partition retained M-blocks across CTA leaders instead of making every
    // CTA contend on one global scan counter. A descriptor has one publisher;
    // both schedules share the same system-acquire readiness check. The
    // static path then needs only a release publication, without a phase CAS.
    unsigned scan_cursor = publisher;
    unsigned pending =
        publisher < geometry.mblocks
            ? 1U + (geometry.mblocks - 1U - publisher) / publishers
            : 0U;
    while (true) {
      // A publisher warp must enter/leave its full-mask collectives together.
      unsigned done = 0U;
      if (lane == 0U)
        done = ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                   s.copies_done)
                   .load(::cuda::memory_order_acquire);
      if (__shfl_sync(0xffffffffU, done, 0U))
        break;
      const bool progressed = local_notification_detail::progress(a);
      if (a.return_ring.depth && lane == 0U && publisher == 0U)
        weftmoe_pf::backward_return::advance_source(a);
      if (lane == 0U && pending) {
        if (backward_pipeline_detail::try_publish_dispatch_mblock(
                a, words, layout, geometry.mblocks, scan_cursor))
          --pending;
        scan_cursor += publishers;
        if (scan_cursor >= geometry.mblocks)
          scan_cursor = publisher;
      }
      __syncwarp();

      const bool idle = !__any_sync(0xffffffffU, progressed);
      const bool published = __shfl_sync(0xffffffffU, pending, 0U) == 0U;
      if (a.return_ring.depth || (idle && published))
        progress_reduce<BaseThread>(a, s, reduce_token);

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
      if (a.return_ring.depth ||
          !__any_sync(0xffffffffU, dispatched || returned || gin_progressed))
        progress_reduce<BaseThread>(a, s, reduce_token);
      if (tid == 0U) {
        auto *counter = reinterpret_cast<pipeline::PaddedCounter *>(
            words + layout.returned_mblocks);
        s.stop =
            s.dispatch_cursor >= s.dispatch_end &&
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
                counter->value)
                    .load(::cuda::memory_order_acquire) ==
                (a.return_ring.depth ? a.return_ring.state->blocks
                                     : geometry.mblocks);
        if (a.return_ring.depth && !a.return_ring.canonical &&
            publisher == 0U &&
            weftmoe_pf::ring::acquire(
                a.peer_payloads[a.rank].return_state->reserved) <
                static_forward::ceil_div(a.dispatch.identity.num_tokens,
                                         a.return_ring.shape.tokens))
          s.stop = 0U;
        // No receive/notification role may retire while one of this CTA's
        // tokens can still hold a source-window credit. Sender CTAs own no
        // Reduce tokens, so blocked future TX never prevents current Reduce.
        if (a.return_ring.depth &&
            ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                s.reduced_warps)
                    .load(::cuda::memory_order_acquire) != Storage::kWarps)
          s.stop = 0U;
        if constexpr (UseGin)
          if (s.stop && (!gin_detail::terminal(a) ||
                         !activation_ring_detail::receive_complete(a, true) ||
                         (a.return_ring.depth &&
                          !activation_ring_detail::receive_complete(a, false))))
            s.stop = 0U;
      }
      communication_worker::copy_sync<Storage::kCopyThreads>();
      if (s.stop)
        break;
      if (!dispatched && !returned && !gin_progressed)
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
    }
    if (warp == 0U)
      communication_worker::drain_notifications<BaseThread>(
          a, [] { return false; });
    communication_worker::copy_sync<Storage::kCopyThreads>();
    if (tid == 0U) {
      if constexpr (UseGin)
        activation_ring_detail::flush_receive(a);
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.copies_done)
          .store(1U, ::cuda::memory_order_release);
    }
  }
  communication_worker::domain_sync<Storage::kThreads>();
}

} // namespace backward_unified_transport

#endif // __has_include(<nccl_device.h>)
