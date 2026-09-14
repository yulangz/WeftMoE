#pragma once

// Included inside forward_dispatch_detail. The GIN owner services bounded
// chunks; the independent publisher warp continues releasing FC1 input groups.
namespace ring_detail {
namespace wire = activation_ring_detail;
namespace ring = communication;

__device__ __forceinline__ void pack_return(const ForwardKernelArgs &a,
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
  const auto *output = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  const auto warp = wire::transport_thread() / 32U, lane = threadIdx.x % 32U;
  for (unsigned i = warp; i < task.rows; i += blockDim.x / 32U - 1U) {
    const auto row = entries[task.first + i];
    auto *record =
        chunk + sizeof(ring::ChunkHeader) + i * a.gin_rings.return_row_bytes;
    if (!lane)
      *reinterpret_cast<ring::ReturnRow *>(record) =
          ring::ReturnRow{headers[row].route_index, 0.0F, 0};
    auto *target = reinterpret_cast<uint4 *>(record + sizeof(ring::ReturnRow));
    const auto *source =
        reinterpret_cast<const uint4 *>(output + row * a.hidden);
    for (auto v = lane; v < a.hidden / 8ULL; v += 32U)
      target[v] = source[v];
  }
  __threadfence_system();
  wire::transport_sync();
}

__device__ __forceinline__ void
receive_dispatch(const ForwardKernelArgs &a, const wire::Transfer &task,
                 transport_state::GenerationCompletionVector *completion,
                 std::byte *storage,
                 forward_return_detail::ReturnCopyBarrier *full) {
  WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay, task.channel,
                          task.rows, execution_epoch(a));
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  const auto source_rank = wire::peer(a, task.channel);
  const auto items = task.rows * a.dispatch.identity.top_k;
  const auto bytes = unsigned(a.hidden * 2ULL);
  bool copied = false;
  for (unsigned item = warp;
       warp < forward_tuning::kRelayScatterWarps && item < items;
       item += forward_tuning::kRelayScatterWarps) {
    const auto packet = item / a.dispatch.identity.top_k,
               index = item % a.dispatch.identity.top_k;
    const auto *record = chunk + packet * a.gin_rings.dispatch_row_bytes;
    const auto header = *reinterpret_cast<const ring::DispatchRow *>(record);
    if (index >= header.fanouts)
      continue;
    const auto target = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow))[index];
    const auto rank =
        a.node * a.local_world_size + target.destination_local_rank;
    auto *base = a.peer_bases[rank];
    if (!lane) {
      peer_at<LocalP2pRouteHeader>(
          base, a.direct_headers_offset)[target.dst_row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          source_rank,
          rank,
          0,
          a.dispatch.identity.generation,
          execution_epoch(a),
          0,
          header.token * a.dispatch.identity.top_k + target.topk_slot,
          header.token,
          target.dst_row,
          target.expert_id,
          target.topk_slot};
      peer_at<float>(base, a.canonical_gates_offset)[target.dst_row] =
          target.gate;
    }
    const auto *src = record + a.gin_rings.dispatch_row_bytes - bytes;
    auto *dst = peer_at<transport::BFloat16>(base, a.canonical_rows_offset) +
                target.dst_row * a.hidden;
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
                                                target.mtile};
      });
  if (!threadIdx.x) {
    claim_completion_tasks(completion, transport_state::TaskClass::RelayWave,
                           task.rows);
    complete_completion_tasks(completion, transport_state::TaskClass::RelayWave,
                              task.rows);
    wire::finish_receive(a, task);
  }
  __syncthreads();
  WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, task.channel,
                          task.rows, execution_epoch(a));
}

__device__ __forceinline__ void receive_return(const ForwardKernelArgs &a,
                                               const wire::Transfer &task) {
  WEFTMOE_TIMELINE_RECORD(a.timeline, SourceRelayStart, SourceRelay, task.channel,
                          task.rows, execution_epoch(a));
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  auto *output =
      peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  for (unsigned i = warp; i < task.rows; i += blockDim.x / 32U) {
    const auto *record = chunk + i * a.gin_rings.return_row_bytes;
    const auto route = reinterpret_cast<const ring::ReturnRow *>(record)->route;
    auto *dst = reinterpret_cast<uint4 *>(output + route * a.hidden);
    const auto *src =
        reinterpret_cast<const uint4 *>(record + sizeof(ring::ReturnRow));
    for (auto v = lane; v < a.hidden / 8ULL; v += 32U)
      dst[v] = src[v];
  }
  __threadfence();
  __syncthreads();
  for (unsigned i = threadIdx.x; i < task.rows; i += blockDim.x) {
    const auto *record = chunk + i * a.gin_rings.return_row_bytes;
    const auto route = reinterpret_cast<const ring::ReturnRow *>(record)->route;
    forward_reduce_detail::publish_ready_route(a, route, 3U);
  }
  __syncthreads();
  if (!threadIdx.x)
    wire::finish_receive(a, task);
  __syncthreads();
  WEFTMOE_TIMELINE_RECORD(a.timeline, SourceRelayEnd, SourceRelay, task.channel,
                          task.rows, execution_epoch(a));
}

__device__ __forceinline__ bool receive_complete(const ForwardKernelArgs &a,
                                                 bool dispatch) {
  const auto first = dispatch ? 0U : a.gin_rings.dispatch_channels;
  const auto end =
      dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.channels();
  auto *states = ring::channels(a.gin_window_base, a.gin_rings);
  for (auto channel = first; channel < end; ++channel)
    if (!::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
             states[channel].rx_terminal)
             .load(::cuda::memory_order_acquire))
      return false;
  return true;
}
} // namespace ring_detail

__device__ __forceinline__ void rdma_role(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    unsigned context, transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *, unsigned long long *scheduler) {
  namespace wire = activation_ring_detail;
  if (threadIdx.x == 32U)
    publish_ready_fc1_tasks(a, context, a.gin_context_count, scheduler);
  if (threadIdx.x < 32U || threadIdx.x >= 64U) {
    __shared__ wire::Transfer task;
    __shared__ unsigned stop;
    unsigned dispatch_cursor = 0, return_cursor = 0;
    bool dispatch_first = true;
    unsigned long long dispatch_puts = 0, return_puts = 0;
    while (true) {
      if (!threadIdx.x) {
        wire::acknowledge(a, context);
        stop = cancelled(a, control) || wire::drained(a, context);
        task.channel = wire::kNoChannel;
        if (!stop) {
          // When both directions are ready, alternate chunks. A continuous
          // Return stream must not starve inputs still needed by FC1.
          const bool selected =
              dispatch_first
                  ? wire::choose_dispatch(a, context, dispatch_cursor, task)
                  : wire::choose_return(a, context, return_cursor, task);
          if (!selected) {
            if (dispatch_first)
              wire::choose_return(a, context, return_cursor, task);
            else
              wire::choose_dispatch(a, context, dispatch_cursor, task);
          }
          if (task.channel != wire::kNoChannel)
            dispatch_first = task.channel >= a.gin_rings.dispatch_channels;
        }
      }
      wire::transport_sync();
      if (stop)
        break;
      if (task.channel != wire::kNoChannel) {
        const bool dispatch = task.channel < a.gin_rings.dispatch_channels;
        if (dispatch) {
          WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchPackStart, GinDispatch,
                                  task.channel, task.rows, execution_epoch(a));
          wire::pack_dispatch(a, task, a.input, a.gates);
          WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchPackEnd, GinDispatch,
                                  task.channel, task.rows, execution_epoch(a));
        } else {
          WEFTMOE_TIMELINE_RECORD(a.timeline, ReturnPackStart, GinReturn,
                                  task.channel, task.rows, execution_epoch(a));
          ring_detail::pack_return(a, task);
          WEFTMOE_TIMELINE_RECORD(a.timeline, ReturnPackEnd, GinReturn,
                                  task.channel, task.rows, execution_epoch(a));
        }
        if (!threadIdx.x) {
          wire::send(a, task);
          if (dispatch) {
            WEFTMOE_TIMELINE_RECORD(a.timeline, GinPutIssue, GinDispatch,
                                    task.channel, task.rows, execution_epoch(a));
          } else {
            WEFTMOE_TIMELINE_RECORD(a.timeline, ReturnGinPutIssue, GinReturn,
                                    task.channel, task.rows, execution_epoch(a));
          }
          auto *status =
              peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
          const auto bytes = 1ULL * task.rows * a.hidden * 2ULL;
          if (dispatch) {
            ++dispatch_puts;
            atomicAdd(&status->gin_dispatch_put_count, 1ULL);
            atomicAdd(&status->deduplicated_cross_node_payload_bytes, bytes);
            atomicAdd(&status->actual_dispatch_peer_bytes, bytes);
          } else {
            ++return_puts;
            atomicAdd(&status->gin_return_put_count, 1ULL);
            atomicAdd(&status->gin_return_payload_bytes, bytes);
          }
          atomicAdd(&status->gin_metadata_bytes,
                    sizeof(communication::ChunkHeader) +
                        1ULL * task.rows *
                            ((dispatch ? a.gin_rings.dispatch_row_bytes
                                       : a.gin_rings.return_row_bytes) -
                             a.hidden * 2ULL));
          atomicAdd(&status->gin_signal_count, 1ULL);
        }
      } else if (!threadIdx.x)
        __nanosleep(forward_tuning::kTransportIdleNanosleep);
      wire::transport_sync();
    }
    if (!threadIdx.x) {
      const ncclGin gin(a.gin_dev_comm, context);
      gin.flush(ncclCoopThread{});
      for (unsigned port = 0; port < communication::kReturnPorts; ++port) {
        const ncclGin ret(a.gin_dev_comm,
                          a.gin_context_count +
                              communication::kReturnPorts *
                                  ((context + a.rank) % a.gin_context_count) +
                              port);
        ret.flush(ncclCoopThread{});
      }
      atomicAdd(
          &completion_task(completion, transport_state::TaskClass::DispatchWave)
               .expected,
          dispatch_puts);
      claim_completion_tasks(
          completion, transport_state::TaskClass::DispatchWave, dispatch_puts);
      complete_completion_tasks(
          completion, transport_state::TaskClass::DispatchWave, dispatch_puts);
      atomicAdd(&completion_task(completion, transport_state::TaskClass::Return)
                     .expected,
                return_puts);
      claim_completion_tasks(completion, transport_state::TaskClass::Return,
                             return_puts);
      complete_completion_tasks(completion, transport_state::TaskClass::Return,
                                return_puts);
      atomicAdd(&completion->dispatch_local_completion_expected, dispatch_puts);
      atomicAdd(&completion->dispatch_local_completion_completed,
                dispatch_puts);
      atomicAdd(&completion->return_local_completion_expected, return_puts);
      atomicAdd(&completion->return_local_completion_completed, return_puts);
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.rdma_dispatch_roles_complete.value, 1ULL);
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.rdma_return_roles_complete.value, 1ULL);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void
relay_role(const ForwardKernelArgs &a,
           persistent_v3::ForwardPipelineControl &control, unsigned ordinal,
           transport_state::GenerationCompletionVector *completion,
           transport_state::ContextCompletionRecord *,
           unsigned char *storage) {
  namespace wire = activation_ring_detail;
  __shared__ wire::Transfer task;
  __shared__ forward_return_detail::LocalReturnState local_return;
  __shared__ unsigned done;
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ forward_return_detail::ReturnCopyBarrier
      full[persistent_v3::kWarpsPerCta];
  if (!threadIdx.x) {
    local_return = {};
    for (unsigned w = 0; w < persistent_v3::kWarpsPerCta; ++w)
      init(&full[w], 1U);
    ::cuda::device::experimental::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  auto local = 1ULL * ordinal * kLocalRouteBatch;
  const auto local_end = local_route_prefix(a);
  unsigned rx_dispatch = ordinal, rx_return = ordinal;
  bool dispatch_done = false;
  while (true) {
    if (threadIdx.x < 32)
      local_notification_detail::progress(a);
    __syncthreads();
    bool progressed = forward_return_detail::progress_local_return(
        a, control, ordinal, local_return,
        reinterpret_cast<std::byte *>(storage), full);
    if (!threadIdx.x)
      wire::claim_receive(a, rx_return, task, false);
    __syncthreads();
    if (task.channel != wire::kNoChannel) {
      ring_detail::receive_return(a, task);
      progressed = true;
    }
    if (!threadIdx.x)
      wire::claim_receive(a, rx_dispatch, task, true);
    __syncthreads();
    if (task.channel != wire::kNoChannel) {
      ring_detail::receive_dispatch(
          a, task, completion, reinterpret_cast<std::byte *>(storage), full);
      progressed = true;
    }
    // Service one local input batch alongside receive/return work. Waiting
    // for an entirely idle receive loop can starve this finite input stream.
    if (local < local_end) {
      const auto end = min(local + kLocalRouteBatch, 1ULL * local_end);
#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)
      publish_local_batch(a, local, end, completion,
                          reinterpret_cast<std::byte *>(storage), full);
#else
      publish_local_batch(a, local, end, completion);
#endif
      local += 1ULL * a.forward_relay_cta_count * kLocalRouteBatch;
      progressed = true;
    }
    if (!threadIdx.x) {
      if (!dispatch_done && local >= local_end &&
          ring_detail::receive_complete(a, true)) {
        const auto previous =
            persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                control.relay_dispatch_roles_complete.value, 1ULL);
        if (previous + 1ULL == a.forward_relay_cta_count) {
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
              ->dispatch_complete_clock = device_globaltimer_ns();
          WEFTMOE_TIMELINE_RECORD(a.timeline, XDispatchComplete, ProxyRelay,
                                  0U, local_end, execution_epoch(a));
        }
        dispatch_done = true;
      }
      done = cancelled(a, control) ||
             (dispatch_done && ring_detail::receive_complete(a, false) &&
              forward_return_detail::local_return_terminal(a, local_return));
    }
    __syncthreads();
    if (done)
      break;
    if (!progressed && threadIdx.x < 32U)
      forward_reduce_detail::try_reduce_one_token(a);
    if (!progressed && !threadIdx.x)
      __nanosleep(forward_tuning::kTransportIdleNanosleep);
    __syncthreads();
  }
  local_notification_detail::drain(a);
  __syncthreads();
  if (!threadIdx.x)
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.return_receive_roles_complete.value, 1ULL);
  __syncthreads();
}
