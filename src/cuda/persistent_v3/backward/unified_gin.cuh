#pragma once

// Included inside backward_unified_transport. Context ownership and GIN
// submission are shared with Forward; these adapters describe Backward's
// compact dY Dispatch and padded dX/dGate Return payloads.
namespace gin_detail {
namespace wire = activation_ring_detail;
namespace ring = communication;
namespace shared_gin = forward_unified_transport::gin_detail;
using forward_unified_transport::copy_row;
using forward_unified_transport::copy_row_fanout;
using forward_unified_transport::copy_sync;
using forward_unified_transport::finish_copies;

__device__ __forceinline__ bool terminal(const BackwardKernelArgs &a) {
  for (unsigned context = 0U; context < a.gin_sender_count; ++context)
    if (!::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
             shared_gin::sender_done(a)[context])
             .load(::cuda::memory_order_acquire))
      return false;
  return true;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void pack_dispatch(const BackwardKernelArgs &a,
                                              Storage &s,
                                              const wire::Transfer &task) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  auto *chunk = static_cast<char *>(a.gin_window_base) +
                a.gin_rings.payload(task.channel, task.sequence, false);
  if (tid == 0U)
    *reinterpret_cast<ring::ChunkHeader *>(chunk) =
        ring::ChunkHeader{task.rows, task.terminal, 0};
  for (unsigned item = warp; item < task.rows; item += Storage::kCopyWarps) {
    const auto packet = a.dispatch_waves.packet_order[task.first + item];
    const auto token = a.dispatch.packet_headers[packet].local_token_id;
    const auto first = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    auto *record = chunk + sizeof(ring::ChunkHeader) +
                   item * a.gin_rings.dispatch_row_bytes;
    if (lane == 0U)
      *reinterpret_cast<ring::DispatchRow *>(record) =
          ring::DispatchRow{token, unsigned(end - first), 0};
    auto *fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow));
    for (auto index = first + lane; index < end; index += 32U) {
      const auto target = a.dispatch.packet_fanout[index];
      fanout[index - first] = ForwardDispatchFanoutRecord{
          target.dst_row,
          0,
          target.topk_slot,
          static_cast<std::uint16_t>(target.destination_local_rank),
          static_cast<std::uint16_t>(target.expert_id),
          target.reserved,
          0.0F};
    }
    if (lane == 0U)
      copy_row(s, warp, a.dy + token * a.hidden,
               record + a.gin_rings.dispatch_row_bytes - a.hidden * 2ULL,
               unsigned(a.hidden * 2ULL));
  }
  if (lane == 0U)
    finish_copies();
  __threadfence_system();
  copy_sync<Storage::kCopyThreads>();
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void pack_return(const BackwardKernelArgs &a,
                                            Storage &s,
                                            const wire::Transfer &task) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  auto *chunk = static_cast<char *>(a.gin_window_base) +
                a.gin_rings.payload(task.channel, task.sequence, false);
  if (tid == 0U)
    *reinterpret_cast<ring::ChunkHeader *>(chunk) =
        ring::ChunkHeader{task.rows, task.terminal, 0};
  const auto *entries =
      peer_at<const unsigned>(a.local_base, a.forward_return_rows_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  const auto *dx = a.dx_rows;
  const auto *dgate = a.dgate_rows;
  if (lane == 0U) {
    for (unsigned item = warp; item < task.rows; item += Storage::kCopyWarps) {
      const auto compact_row = entries[task.first + item];
      const auto padded_row = mapping[compact_row];
      auto *record = chunk + sizeof(ring::ChunkHeader) +
                     item * a.gin_rings.return_row_bytes;
      *reinterpret_cast<ring::ReturnRow *>(record) = ring::ReturnRow{
          headers[compact_row].route_index, dgate[padded_row], 0};
      copy_row(s, warp, dx + 1ULL * padded_row * a.hidden,
               record + sizeof(ring::ReturnRow), unsigned(a.hidden * 2ULL));
    }
    finish_copies();
  }
  __threadfence_system();
  copy_sync<Storage::kCopyThreads>();
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void receive_dispatch(const BackwardKernelArgs &a,
                                                 Storage &s,
                                                 const wire::Transfer &task) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  const unsigned items = task.rows * a.dispatch.identity.top_k;
  if (lane == 0U) {
    for (unsigned row = warp; row < task.rows; row += Storage::kCopyWarps) {
      const auto *record = chunk + row * a.gin_rings.dispatch_row_bytes;
      const auto header = *reinterpret_cast<const ring::DispatchRow *>(record);
      const unsigned fanouts =
          min(header.fanouts, unsigned(a.dispatch.identity.top_k));
      copy_row_fanout(
          s, warp, record + a.gin_rings.dispatch_row_bytes - a.hidden * 2ULL,
          unsigned(a.hidden * 2ULL), fanouts,
          [&, record](unsigned index) -> void * {
            const auto target =
                reinterpret_cast<const ForwardDispatchFanoutRecord *>(
                    record + sizeof(ring::DispatchRow))[index];
            const auto destination = a.node * a.local_world_size +
                                     target.destination_local_rank;
            return a.peer_payloads[destination].dr +
                   target.dst_row * a.hidden;
          });
    }
    finish_copies();
  }
  copy_sync<Storage::kCopyThreads>();
  for (unsigned item = tid; item < items; item += Storage::kCopyThreads) {
    const auto *record = chunk + (item / a.dispatch.identity.top_k) *
                                     a.gin_rings.dispatch_row_bytes;
    const unsigned index = item % a.dispatch.identity.top_k;
    if (index >= reinterpret_cast<const ring::DispatchRow *>(record)->fanouts)
      continue;
    const auto target = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow))[index];
    local_notification_detail::Receiver<BackwardKernelArgs>::publish_dispatch(
        a, target.dst_row, target.destination_local_rank);
  }
  copy_sync<Storage::kCopyThreads>();
  if (tid == 0U)
    wire::finish_receive(a, task);
  copy_sync<Storage::kCopyThreads>();
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void receive_return(const BackwardKernelArgs &a,
                                               Storage &s,
                                               const wire::Transfer &task) {
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  auto *output = a.peer_payloads[a.rank].contribution;
  auto *dgate = a.peer_payloads[a.rank].dgate;
  if (lane == 0U) {
    for (unsigned item = warp; item < task.rows; item += Storage::kCopyWarps) {
      const auto *record = chunk + item * a.gin_rings.return_row_bytes;
      const auto meta = *reinterpret_cast<const ring::ReturnRow *>(record);
      copy_row(s, warp, record + sizeof(ring::ReturnRow),
               output + meta.route * a.hidden, unsigned(a.hidden * 2ULL));
      dgate[meta.route] = meta.dgate;
    }
    finish_copies();
  }
  copy_sync<Storage::kCopyThreads>();
  for (unsigned item = tid; item < task.rows; item += Storage::kCopyThreads) {
    const auto *record = chunk + item * a.gin_rings.return_row_bytes;
    local_notification_detail::Receiver<BackwardKernelArgs>::apply(
        a, reinterpret_cast<const ring::ReturnRow *>(record)->route);
  }
  copy_sync<Storage::kCopyThreads>();
  if (tid == 0U)
    wire::finish_receive(a, task);
  copy_sync<Storage::kCopyThreads>();
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool progress_receive(const BackwardKernelArgs &a,
                                                 Storage &s, bool dispatch) {
  const unsigned tid = threadIdx.x - BaseThread;
  if (tid == 0U)
    wire::claim_receive(a, dispatch ? s.gin_rx_dispatch : s.gin_rx_return,
                        s.gin_task, dispatch);
  copy_sync<Storage::kCopyThreads>();
  const bool received = s.gin_task.channel != wire::kNoChannel;
  copy_sync<Storage::kCopyThreads>();
  if (!received)
    return false;
  if (dispatch)
    receive_dispatch<BaseThread>(a, s, s.gin_task);
  else
    receive_return<BaseThread>(a, s, s.gin_task);
  return true;
}

// The Return owner publishes immutable compact row IDs into the same bounded
// peer queues as the baseline. Each finite chunk is at most kCopyThreads rows,
// so layouts with only two copy warps still cover an entire M128 descriptor.
// This metadata path does not wait for any payload or NIC operation.
template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void publish_return_rows(const BackwardKernelArgs &a,
                                                    Storage &s, unsigned first,
                                                    unsigned rows) {
  constexpr auto threads = Storage::kCopyThreads;
  constexpr auto warps = Storage::kCopyWarps;
  const unsigned tid = threadIdx.x - BaseThread;
  const unsigned warp = tid / 32U, lane = tid % 32U;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  auto *entries = peer_at<unsigned>(a.local_base, a.forward_return_rows_offset);
  for (unsigned begin = 0U; begin < rows; begin += threads) {
    const auto row = begin + tid < rows ? first + begin + tid : UINT32_MAX;
    const auto source = row != UINT32_MAX ? headers[row].source_rank : a.rank;
    auto pending =
        source / a.local_world_size != a.node ? 1U + source : UINT32_MAX;
    while (true) {
      const auto next = __reduce_min_sync(0xffffffffU, pending);
      if (lane == 0U)
        s.queue_scratch[warp] = next;
      copy_sync<threads>();
      if (tid == 0U) {
        auto selected = UINT32_MAX;
        for (unsigned w = 0U; w < warps; ++w)
          selected = min(selected, s.queue_scratch[w]);
        s.queue_selected = selected;
      }
      copy_sync<threads>();
      const auto selected = s.queue_selected;
      if (selected == UINT32_MAX)
        break;
      const auto mask = __ballot_sync(0xffffffffU, pending == selected);
      if (lane == 0U)
        s.queue_scratch[warp] = __popc(mask);
      copy_sync<threads>();
      auto &queue = return_queue_detail::queues(a)[selected];
      if (tid == 0U) {
        unsigned count = 0U;
        for (unsigned w = 0U; w < warps; ++w) {
          const auto n = s.queue_scratch[w];
          s.queue_scratch[w] = count;
          count += n;
        }
        s.queue_rows = count;
        s.queue_base =
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
                queue.reserved)
                .fetch_add(count, ::cuda::memory_order_relaxed);
        if (s.queue_base + count > queue.capacity)
          __trap();
      }
      copy_sync<threads>();
      if (pending == selected) {
        const auto offset =
            s.queue_scratch[warp] + __popc(mask & ((1U << lane) - 1U));
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
            entries[queue.begin + s.queue_base + offset])
            .store(row, ::cuda::memory_order_release);
        pending = UINT32_MAX;
      }
      copy_sync<threads>();
      if (tid == 0U)
        return_queue_detail::publish_ready(a, queue, s.queue_base, s.queue_rows);
      copy_sync<threads>();
    }
  }
}
} // namespace gin_detail
