#pragma once

// Backward uses the same fixed wire rings as Forward. The retained Notify
// handle supplies routes; no data or metadata must survive in the GIN window.
#if __has_include(<nccl_device.h>)
namespace backward_gin_detail {
namespace wire = activation_ring_detail;
namespace ring = communication;

struct RelayInboundState {
  wire::Transfer task;
  unsigned cursor = 0;
  unsigned complete = 0;
};
using SourceCompletionState = RelayInboundState;

__device__ __forceinline__ bool all_received(const BackwardKernelArgs &a,
                                             bool dispatch) {
  const auto first = dispatch ? 0U : a.gin_rings.dispatch_channels;
  const auto end =
      dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.channels();
  auto *states = ring::channels(a.gin_window_base, a.gin_rings);
  for (auto c = first; c < end; ++c)
    if (!::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
             states[c].rx_terminal)
             .load(::cuda::memory_order_acquire))
      return false;
  return true;
}

__device__ __forceinline__ void
initialize_relay_state(const BackwardKernelArgs &, unsigned context,
                       RelayInboundState &inbound,
                       SourceCompletionState &completion) {
  if (!threadIdx.x) {
    inbound = {};
    completion = {};
    inbound.cursor = completion.cursor = context;
  }
  __syncthreads();
}
__device__ __forceinline__ bool
inbound_terminal(const RelayInboundState &state) {
  return state.complete != 0U;
}
__device__ __forceinline__ bool
source_completion_terminal(const SourceCompletionState &state) {
  return state.complete != 0U;
}

__device__ __forceinline__ bool
progress_inbound_once(const BackwardKernelArgs &a, RelayInboundState &state,
                      std::byte *storage,
                      ::cuda::barrier<::cuda::thread_scope_block> *full) {
  if (!threadIdx.x) {
    wire::claim_receive(a, state.cursor, state.task, true);
    state.complete = all_received(a, true);
  }
  __syncthreads();
  const auto task = state.task;
  if (task.channel == wire::kNoChannel)
    return false;
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  const auto bytes = unsigned(a.hidden * 2ULL);
  const auto items = task.rows * a.dispatch.identity.top_k;
  bool copied = false;
  for (unsigned item = warp; item < items; item += blockDim.x / 32U) {
    const auto *record = chunk + (item / a.dispatch.identity.top_k) *
                                     a.gin_rings.dispatch_row_bytes;
    const auto index = item % a.dispatch.identity.top_k;
    if (index >= reinterpret_cast<const ring::DispatchRow *>(record)->fanouts)
      continue;
    const auto target = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow))[index];
    auto *base = a.peer_bases[a.node * a.local_world_size +
                              target.destination_local_rank];
    auto *dst = peer_at<transport::BFloat16>(base, a.compact_dr_offset) +
                target.dst_row * a.hidden;
    const auto *src = record + a.gin_rings.dispatch_row_bytes - bytes;
    if (!lane) {
      auto *smem = storage + warp * bytes;
      ::cuda::device::experimental::cp_async_bulk_global_to_shared(
          smem, src, bytes, full[warp]);
      auto ticket = ::cuda::device::barrier_arrive_tx(full[warp], 1U, bytes);
      full[warp].wait(static_cast<decltype(ticket) &&>(ticket));
      ::cuda::device::experimental::cp_async_bulk_shared_to_global(dst, smem,
                                                                   bytes);
      ::cuda::device::experimental::cp_async_bulk_commit_group();
      ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
      copied = true;
    }
  }
  if (copied)
    ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
  __threadfence_system();
  __syncthreads();
  local_notification_detail::publish_dispatch(
      a, items, blockDim.x, [&](unsigned item) {
        const auto *record = chunk + (item / a.dispatch.identity.top_k) *
                                         a.gin_rings.dispatch_row_bytes;
        const auto index = item % a.dispatch.identity.top_k;
        if (index >=
            reinterpret_cast<const ring::DispatchRow *>(record)->fanouts)
          return local_notification_detail::Entry{communication::kLocalPeers,
                                                  0};
        const auto target =
            reinterpret_cast<const ForwardDispatchFanoutRecord *>(
                record + sizeof(ring::DispatchRow))[index];
        return local_notification_detail::Entry{target.destination_local_rank,
                                                target.dst_row};
      });
  if (!threadIdx.x)
    wire::finish_receive(a, task);
  __syncthreads();
  return true;
}

__device__ __forceinline__ bool
progress_source_completion_once(const BackwardKernelArgs &a,
                                SourceCompletionState &state) {
  if (!threadIdx.x) {
    wire::claim_receive(a, state.cursor, state.task, false);
    state.complete = all_received(a, false);
  }
  __syncthreads();
  const auto task = state.task;
  if (task.channel == wire::kNoChannel)
    return false;
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  auto *output =
      peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
  auto *gates = peer_at<float>(a.local_base, a.dgate_output_offset);
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  for (unsigned i = warp; i < task.rows; i += blockDim.x / 32U) {
    const auto *record = chunk + i * a.gin_rings.return_row_bytes;
    const auto meta = *reinterpret_cast<const ring::ReturnRow *>(record);
    auto *dst = reinterpret_cast<uint4 *>(output + meta.route * a.hidden);
    const auto *src =
        reinterpret_cast<const uint4 *>(record + sizeof(ring::ReturnRow));
    for (auto v = lane; v < a.hidden / 8ULL; v += 32U)
      dst[v] = src[v];
    if (!lane)
      gates[meta.route] = meta.dgate;
  }
  __threadfence();
  __syncthreads();
  auto *ready =
      peer_at<unsigned long long>(a.local_base, a.return_ready_offset);
  for (unsigned i = threadIdx.x; i < task.rows; i += blockDim.x) {
    const auto *record = chunk + i * a.gin_rings.return_row_bytes;
    const auto route = reinterpret_cast<const ring::ReturnRow *>(record)->route;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        ready[route])
        .store(execution_epoch(a), ::cuda::memory_order_release);
  }
  __syncthreads();
  if (!threadIdx.x)
    wire::finish_receive(a, task);
  __syncthreads();
  return true;
}

__device__ __forceinline__ void pack_return(const BackwardKernelArgs &a,
                                            const wire::Transfer &task) {
  auto *chunk = static_cast<char *>(a.gin_window_base) +
                a.gin_rings.payload(task.channel, task.sequence, false);
  if (!threadIdx.x)
    *reinterpret_cast<ring::ChunkHeader *>(chunk) =
        ring::ChunkHeader{task.rows, task.terminal, 0};
  const auto *entries =
      peer_at<const unsigned>(a.local_base, a.forward_return_rows_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  const auto *dx =
      peer_at<const transport::BFloat16>(a.local_base, a.dx_rows_offset);
  const auto *gates = peer_at<const float>(a.local_base, a.dgate_rows_offset);
  const auto warp = wire::transport_thread() / 32U, lane = threadIdx.x % 32U;
  for (unsigned i = warp; i < task.rows; i += blockDim.x / 32U - 1U) {
    const auto row = entries[task.first + i], padded = mapping[row];
    auto *record =
        chunk + sizeof(ring::ChunkHeader) + i * a.gin_rings.return_row_bytes;
    if (!lane)
      *reinterpret_cast<ring::ReturnRow *>(record) =
          ring::ReturnRow{headers[row].route_index, gates[padded], 0};
    auto *dst = reinterpret_cast<uint4 *>(record + sizeof(ring::ReturnRow));
    const auto *src =
        reinterpret_cast<const uint4 *>(dx + 1ULL * padded * a.hidden);
    for (auto v = lane; v < a.hidden / 8ULL; v += 32U)
      dst[v] = src[v];
  }
  __threadfence_system();
  wire::transport_sync();
}

__device__ __forceinline__ void rdma_role(const BackwardKernelArgs &a) {
  const auto context = unsigned(blockIdx.x);
  if (threadIdx.x == 32U) {
    auto *words = backward_pipeline_detail::scheduler_words(a);
    const auto layout = persistent_v3::backward::make_backward_scheduler_layout(
        a.scheduler_mblock_capacity,
        unsigned(a.hidden / backward_tuning::kDxTileN),
        a.scheduler_expert_capacity);
    unsigned mblock = 0;
    while (mblock < a.mblocks) {
      backward_pipeline_detail::publish_dispatch_if_ready(a, words, layout);
      if (::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
              words[layout.phase + mblock])
              .load(::cuda::memory_order_acquire))
        ++mblock;
      else
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
    }
  }
  if (threadIdx.x < 32U || threadIdx.x >= 64U) {
    __shared__ wire::Transfer task;
    __shared__ unsigned stop;
    unsigned dispatch_cursor = 0, return_cursor = 0;
    while (true) {
      if (!threadIdx.x) {
        wire::acknowledge(a, context);
        stop = wire::drained(a, context);
        task.channel = wire::kNoChannel;
        if (!stop && !wire::choose_return(a, context, return_cursor, task))
          wire::choose_dispatch(a, context, dispatch_cursor, task);
      }
      wire::transport_sync();
      if (stop)
        break;
      if (task.channel != wire::kNoChannel) {
        const bool dispatch = task.channel < a.gin_rings.dispatch_channels;
        if (dispatch)
          wire::pack_dispatch(a, task, a.dy, nullptr);
        else
          pack_return(a, task);
        if (!threadIdx.x) {
          wire::send(a, task);
          if (!dispatch) {
            auto *status =
                peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
            atomicAdd(&status->gin_return_put_count, 1ULL);
            atomicAdd(&status->gin_return_payload_bytes,
                      task.rows * (a.hidden * 2ULL + sizeof(float)));
          }
        }
      } else if (!threadIdx.x)
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
      wire::transport_sync();
    }
    if (!threadIdx.x) {
      const ncclGin gin(a.gin_dev_comm, context);
      gin.flush(ncclCoopThread{});
      for (unsigned port = 0; port < ring::kReturnPorts; ++port) {
        const ncclGin ret(a.gin_dev_comm,
                          a.gin_context_count +
                              ring::kReturnPorts *
                                  ((context + a.rank) % a.gin_context_count) +
                              port);
        ret.flush(ncclCoopThread{});
      }
    }
  }
  __syncthreads();
}
} // namespace backward_gin_detail
#endif
