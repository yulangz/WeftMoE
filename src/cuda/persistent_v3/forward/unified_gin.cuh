#pragma once

// Included inside forward_unified_transport after the bounded TMA-copy and
// communication-domain barrier helpers.
namespace gin_detail {
namespace wire = activation_ring_detail;
namespace ring = communication;

template <class Args>
__device__ __forceinline__ GinSenderState *states(const Args &a) {
  return static_cast<GinSenderState *>(a.forward_unified_gin_state);
}

// A sender owns every S-th channel, independently of context allocation.
template <class Args>
__device__ __forceinline__ bool drained(const Args &a, unsigned sender) {
  for (unsigned channel = sender; channel < a.gin_rings.channels();
       channel += a.gin_sender_count) {
    const auto gin = wire::gin_for_cta(a, sender);
    if (!ring::drained(gin, a.gin_window, a.gin_window_base, a.gin_rings, channel))
      return false;
  }
  return true;
}

template <class Args>
__device__ __forceinline__ void send_packed(const Args &a,
                                         const wire::Transfer &task) {
  const bool dispatch = task.channel < a.gin_rings.dispatch_channels;
  const auto bytes = sizeof(ring::ChunkHeader) +
                         task.rows * (dispatch ? a.gin_rings.dispatch_row_bytes
                                               : a.gin_rings.return_row_bytes);
  const auto gin = wire::gin_for_cta(a, blockIdx.x);
  const unsigned slot = task.sequence % ring::kRingDepth;
  // The CTA's packers completed all TMA stores and system fences before
  // publishing this immutable task to their submitter.
  gin.put(
      ncclTeamWorld(gin.comm), wire::peer(a, task.channel), a.gin_window,
      a.gin_rings.payload(wire::remote_channel(a, task.channel), task.sequence,
                          true),
      a.gin_window, a.gin_rings.payload(task.channel, task.sequence, false),
      bytes,
      ncclGin_WeakVASignalAdd{
          a.gin_window,
          a.gin_rings.slot_control(wire::remote_channel(a, task.channel),
                                   slot) +
              offsetof(ring::ChunkSlot, ready_signal),
          1},
      ncclGin_WeakCounterInc{ring::counter(a.gin_rings, task.channel, slot)},
      ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_device,
      ::cuda::thread_scope_system);
  auto &channel = ring::channels(a.gin_window_base, a.gin_rings)[task.channel];
  ++channel.slots[slot].local_target;
  ++channel.sent;
  channel.tx_cursor += (dispatch ? ring::kDispatchChannels : ring::kReturnPorts) * ring::kChunkRows;
  channel.tx_terminal = task.terminal;
}

__device__ __forceinline__ void account_send(const ForwardKernelArgs &a,
                                             const wire::Transfer &task) {
  const bool dispatch = task.channel < a.gin_rings.dispatch_channels;
  const auto bytes = sizeof(ring::ChunkHeader) +
                     task.rows * (dispatch ? a.gin_rings.dispatch_row_bytes
                                           : a.gin_rings.return_row_bytes);
  if (dispatch) {
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, GinPutIssue, GinDispatch,
                               task.channel, task.rows, execution_epoch(a));
  } else {
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnGinPutIssue, GinReturn,
                               task.channel, task.rows, execution_epoch(a));
  }
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  const auto payload = 1ULL * task.rows * a.hidden * 2ULL;
  if (dispatch) {
    atomicAdd(&status->gin_dispatch_put_count, 1ULL);
    atomicAdd(&status->deduplicated_cross_node_payload_bytes, payload);
    atomicAdd(&status->actual_dispatch_peer_bytes, payload);
  } else {
    atomicAdd(&status->gin_return_put_count, 1ULL);
    atomicAdd(&status->gin_return_payload_bytes, payload);
  }
  atomicAdd(&status->gin_metadata_bytes, 1ULL * bytes - payload);
  atomicAdd(&status->gin_signal_count, 1ULL);
}

template <class Args>
__device__ __forceinline__ void flush_context(const Args &a, unsigned sender) {
  // Every slot's local counter and remote credit were checked by drained().
  wire::gin_for_cta(a, sender).flush(ncclCoopThread{});
}

__device__ __forceinline__ void
complete_context(const ForwardKernelArgs &a,
                 persistent_v3::ForwardPipelineControl &control,
                 unsigned context, GinSenderState &state) {
  (void)context;
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
  atomicAdd(
      &completion_task(completion, transport_state::TaskClass::DispatchWave)
           .expected,
      state.dispatch_puts);
  claim_completion_tasks(completion, transport_state::TaskClass::DispatchWave,
                         state.dispatch_puts);
  complete_completion_tasks(completion,
                            transport_state::TaskClass::DispatchWave,
                            state.dispatch_puts);
  atomicAdd(
      &completion_task(completion, transport_state::TaskClass::Return).expected,
      state.return_puts);
  claim_completion_tasks(completion, transport_state::TaskClass::Return,
                         state.return_puts);
  complete_completion_tasks(completion, transport_state::TaskClass::Return,
                            state.return_puts);
  atomicAdd(&completion->dispatch_local_completion_expected,
            state.dispatch_puts);
  atomicAdd(&completion->dispatch_local_completion_completed,
            state.dispatch_puts);
  atomicAdd(&completion->return_local_completion_expected, state.return_puts);
  atomicAdd(&completion->return_local_completion_completed, state.return_puts);
  persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
      control.rdma_dispatch_roles_complete.value, 1ULL);
  persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
      control.rdma_return_roles_complete.value, 1ULL);
}

// Backward retains one unfinished pack per channel. Forward derives additional
// reservations from immutable pending tasks within the bounded slot pool;
// durable sent/tx_cursor still advance only during ordered submission.
__device__ __forceinline__ bool channel_pending(GinSenderState &state,
                                                unsigned channel) {
#pragma unroll
  for (unsigned slot = 0; slot < forward_tuning::kUnifiedGinPackSlots; ++slot)
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
            state.slots[slot].pending)
                .load(::cuda::memory_order_relaxed) != 0U &&
        state.slots[slot].task.channel == channel)
      return true;
  return false;
}

template <bool Dispatch, bool SameChannel = false, class Args>
__device__ __forceinline__ bool choose_unpacked(const Args &a, unsigned context,
                                                GinSenderState &state,
                                                wire::Transfer &task,
                                                unsigned max_probes = 0U) {
  const unsigned first = Dispatch ? 0U : a.gin_rings.dispatch_channels;
  const unsigned end = Dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.channels();
  const unsigned stride = a.gin_sender_count;
  const unsigned start = first + (context + stride - first % stride) % stride;
  const unsigned count = start < end ? 1U + (end - 1U - start) / stride : 0U;
  auto &cursor = Dispatch ? state.dispatch_cursor : state.return_cursor;
  auto *channels = ring::channels(a.gin_window_base, a.gin_rings);
  const unsigned probes = max_probes == 0U ? count : min(count, max_probes);
  for (unsigned probe = 0; probe < probes; ++probe) {
    const unsigned channel = start + (cursor++ % count) * stride;
    if constexpr (!SameChannel)
      if (channel_pending(state, channel))
        continue;
    auto &wire_state = channels[channel];
    auto sequence = wire_state.sent;
    if constexpr (SameChannel) {
      bool terminal_reserved = false;
      unsigned channel_reservations = 0U;
      // The sole submitter owns reservations. Packers may publish pending
      // 1->2, but descriptors stay immutable until their ordered submission.
#pragma unroll
      for (unsigned slot = 0; slot < forward_tuning::kUnifiedGinPackSlots;
           ++slot) {
        const auto &entry = state.slots[slot];
        if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
                state.slots[slot].pending)
                    .load(::cuda::memory_order_relaxed) != 0U &&
            entry.task.channel == channel) {
          if constexpr (forward_tuning::kGinChannelPackLimit <
                        forward_tuning::kUnifiedGinPackSlots)
            ++channel_reservations;
          terminal_reserved |= entry.task.terminal != 0U;
          sequence = max(sequence, entry.task.sequence + 1ULL);
        }
      }
      // A pending terminal closes reservation immediately, even if earlier
      // payloads have not packed yet and prevent its ordered submission.
      if (terminal_reserved)
        continue;
      // The physical slot budget already limits larger configured values;
      // the default limit equal to that budget needs no extra counting.
      if constexpr (forward_tuning::kGinChannelPackLimit <
                    forward_tuning::kUnifiedGinPackSlots)
        if (channel_reservations >= forward_tuning::kGinChannelPackLimit)
          continue;
    }
    const ncclGin gin(a.gin_dev_comm, wire::data_context(a, channel));
    if (wire_state.tx_terminal ||
        !ring::can_send(gin, a.gin_window, a.gin_window_base, a.gin_rings, channel, sequence))
      continue;
    if constexpr (Dispatch) {
      auto reserved_cursor = wire_state.tx_cursor;
      if constexpr (SameChannel)
        reserved_cursor += (sequence - wire_state.sent) *
                           (1ULL * ring::kDispatchChannels * ring::kChunkRows);
      const auto rows = reserved_cursor < wire_state.tx_end
                            ? min(1ULL * ring::kChunkRows,
                                  wire_state.tx_end - reserved_cursor)
                            : 0ULL;
      task = wire::Transfer{channel, unsigned(rows),
                            reserved_cursor + 1ULL * ring::kDispatchChannels * ring::kChunkRows >= wire_state.tx_end,
                            sequence, reserved_cursor};
    } else {
      auto &queue = return_queue_detail::queues(a)[1U + wire::peer(a, channel)];
      const auto position = wire_state.tx_cursor +
          (sequence - wire_state.sent) * (1ULL * ring::kReturnPorts * ring::kChunkRows);
      const auto rows = position < queue.capacity
          ? unsigned(min(queue.capacity - position, 1ULL * ring::kChunkRows)) : 0U;
      if (!return_queue_detail::ready_block(a, queue, position, rows))
        continue;
      task = wire::Transfer{channel, rows,
          position + 1ULL * ring::kReturnPorts * ring::kChunkRows >= queue.capacity,
          sequence, queue.begin + position};
    }
    return true;
  }
  return false;
}

// Only this CTA advances its sender state; packers publish pending with release.
template <bool SameChannel = false, class Args, class OnSend, class OnComplete>
__device__ __forceinline__ bool
progress_context(const Args &a, unsigned &cursor, unsigned &selected_context,
                 unsigned &selected_slot, wire::Transfer &task, OnSend on_send,
                 OnComplete on_complete, bool reserve_pack = true) {
  task.channel = wire::kNoChannel;
  bool progressed = false;
  (void)cursor;
  const unsigned context = blockIdx.x;
  if (context >= a.gin_sender_count) return false;
  {
    auto &state = states(a)[context];
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(state.done)
            .load(::cuda::memory_order_relaxed) == 0U) {
      unsigned occupied = 0U;
      // Keep the large NCCL submission body out of an unrolled slot expansion.
#pragma unroll 1
      for (unsigned slot = 0; slot < forward_tuning::kUnifiedGinPackSlots;
           ++slot) {
        auto &entry = state.slots[slot];
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device> pending(
            entry.pending);
        auto phase = pending.load(::cuda::memory_order_acquire);
        bool may_submit = phase == 2U;
        if constexpr (SameChannel) {
          // Slot-array order is unrelated to a channel's sequence. Retain a
          // later packed slot until its predecessor has actually submitted.
          if (may_submit)
            may_submit = entry.task.sequence ==
                         ring::channels(a.gin_window_base,
                                        a.gin_rings)[entry.task.channel].sent;
        }
        if (may_submit) {
          send_packed(a, entry.task);
          on_send(entry.task);
          if (entry.task.channel < a.gin_rings.dispatch_channels)
            ++state.dispatch_puts;
          else
            ++state.return_puts;
          pending.store(0U, ::cuda::memory_order_relaxed);
          phase = 0U;
          progressed = true;
        }
        occupied += phase != 0U;
      }
      if (occupied == 0U && drained(a, context)) {
        flush_context(a, context);
        on_complete(context, state);
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(state.done)
            .store(1U, ::cuda::memory_order_release);
        progressed = true;
      } else if (reserve_pack && occupied < forward_tuning::kUnifiedGinPackSlots) {
        const bool dispatch_first = state.dispatch_first != 0U;
        bool selected =
            dispatch_first
                ? choose_unpacked<true, SameChannel>(a, context, state, task)
                : choose_unpacked<false, SameChannel>(a, context, state, task);
        if (!selected)
          selected =
              dispatch_first
                  ? choose_unpacked<false, SameChannel>(a, context, state, task)
                  : choose_unpacked<true, SameChannel>(a, context, state, task);
        if (selected) {
          state.dispatch_first = task.channel >= a.gin_rings.dispatch_channels;
          for (unsigned probe_slot = 0;
               probe_slot < forward_tuning::kUnifiedGinPackSlots;
               ++probe_slot) {
            const unsigned slot =
                state.slot_cursor++ % forward_tuning::kUnifiedGinPackSlots;
            auto &entry = state.slots[slot];
            ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device> pending(
                entry.pending);
            if (pending.load(::cuda::memory_order_relaxed) != 0U)
              continue;
            entry.task = task;
            pending.store(1U, ::cuda::memory_order_relaxed);
            selected_context = context;
            selected_slot = slot;
            break;
          }
        }
      }
    }
    if (task.channel != wire::kNoChannel)
      return true;
  }
  return progressed;
}

#ifndef WEFTMOE_FWD_GIN_PAYLOAD_ISSUER_LANES
#define WEFTMOE_FWD_GIN_PAYLOAD_ISSUER_LANES 1
#endif

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void pack_dispatch(const ForwardKernelArgs &a,
                                              Storage &s,
                                              const wire::Transfer &task) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  constexpr unsigned issuers = WEFTMOE_FWD_GIN_PAYLOAD_ISSUER_LANES;
  static_assert(issuers >= 1U && issuers <= Storage::kDispatchLanes);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, DispatchPackStart,
                             GinDispatch, task.channel, task.rows,
                             execution_epoch(a));
  auto *chunk = static_cast<char *>(a.gin_window_base) +
                a.gin_rings.payload(task.channel, task.sequence, false);
  if (tid == 0U)
    *reinterpret_cast<ring::ChunkHeader *>(chunk) =
        ring::ChunkHeader{task.rows, task.terminal, 0};
  // Materialize a complete chunk cooperatively before entering the TMA row
  // loop. Per-row warp reconvergence must not serialize metadata preparation
  // behind the elected issuer's payload wait.
  for (unsigned item = tid; item < task.rows; item += Storage::kCopyThreads) {
    const auto packet = a.dispatch_waves.packet_order[task.first + item];
    const auto token = a.dispatch.packet_headers[packet].local_token_id;
    const auto first = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    auto *record = chunk + sizeof(ring::ChunkHeader) +
                   item * a.gin_rings.dispatch_row_bytes;
    *reinterpret_cast<ring::DispatchRow *>(record) =
        ring::DispatchRow{token, unsigned(end - first), 0};
    auto *fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow));
    for (auto index = first; index < end; ++index) {
      const auto target = a.dispatch.packet_fanout[index];
      fanout[index - first] = ForwardDispatchFanoutRecord{
          target.dst_row, 0, target.topk_slot,
          static_cast<std::uint16_t>(target.destination_local_rank),
          static_cast<std::uint16_t>(target.expert_id), target.reserved,
          a.gates[token * a.dispatch.identity.top_k + target.topk_slot]};
    }
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (lane < issuers) {
    LaneStagingView<Storage> view(s, warp, lane);
    RowPrefetchPipeline<LaneStagingView<Storage>> row_pipeline(view, 0U);
    for (unsigned item = warp * issuers + lane; item < task.rows;
         item += Storage::kCopyWarps * issuers) {
      auto *record = chunk + sizeof(ring::ChunkHeader) +
                     item * a.gin_rings.dispatch_row_bytes;
      const auto token = reinterpret_cast<const ring::DispatchRow *>(record)->token;
      row_pipeline.copy(a.input + token * a.hidden,
                        record + a.gin_rings.dispatch_row_bytes - a.hidden * 2ULL,
                        unsigned(a.hidden * 2ULL));
    }
    row_pipeline.drain();
    finish_copies<false>();
  }
  __threadfence_system();
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, DispatchPackEnd,
                             GinDispatch, task.channel, task.rows,
                             execution_epoch(a));
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void pack_return(const ForwardKernelArgs &a,
                                            Storage &s,
                                            const wire::Transfer &task) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  constexpr unsigned issuers = WEFTMOE_FWD_GIN_PAYLOAD_ISSUER_LANES;
  static_assert(issuers >= 1U && issuers <= Storage::kDispatchLanes);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ReturnPackStart, GinReturn,
                             task.channel, task.rows, execution_epoch(a));
  auto *chunk = static_cast<char *>(a.gin_window_base) +
                a.gin_rings.payload(task.channel, task.sequence, false);
  if (tid == 0U)
    *reinterpret_cast<ring::ChunkHeader *>(chunk) =
        ring::ChunkHeader{task.rows, task.terminal, 0};
  const auto *entries =
      peer_at<const unsigned>(a.local_base, a.forward_return_rows_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *output = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  if (lane < issuers) {
    LaneStagingView<Storage> view(s, warp, lane);
    RowPrefetchPipeline<LaneStagingView<Storage>> row_pipeline(view, 0U);
    for (unsigned item = warp * issuers + lane; item < task.rows;
         item += Storage::kCopyWarps * issuers) {
      const unsigned row = entries[task.first + item];
      auto *record = chunk + sizeof(ring::ChunkHeader) +
                     item * a.gin_rings.return_row_bytes;
      *reinterpret_cast<ring::ReturnRow *>(record) =
          ring::ReturnRow{headers[row].route_index, 0.0F, 0};
      row_pipeline.copy(output + row * a.hidden,
               record + sizeof(ring::ReturnRow), unsigned(a.hidden * 2ULL));
    }
    row_pipeline.drain();
    finish_copies<false>();
  }
  __threadfence_system();
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ReturnPackEnd, GinReturn,
                             task.channel, task.rows, execution_epoch(a));
}

__device__ __forceinline__ bool
claim_dispatch_receive(const ForwardKernelArgs &a, unsigned &cursor,
                       wire::Transfer &task) {
  return wire::claim_receive(a, cursor, task, true);
}

__device__ __forceinline__ void
finish_dispatch_receive(const ForwardKernelArgs &a, const wire::Transfer &task) {
  wire::finish_receive(a, task);
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void receive_dispatch(const ForwardKernelArgs &a,
                                                 Storage &s,
                                                 const wire::Transfer &task) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ProxyRelayStart, ProxyRelay,
                             task.channel, task.rows, execution_epoch(a));
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader) +
                      task.first * a.gin_rings.dispatch_row_bytes;
  const auto source = wire::peer(a, task.channel);
  const unsigned items = task.rows * a.dispatch.identity.top_k;
  if (lane < Storage::kDispatchLanes) {
    auto destination = [&a](const char *record, unsigned index) -> void * {
      const auto target =
          reinterpret_cast<const ForwardDispatchFanoutRecord *>(
              record + sizeof(ring::DispatchRow))[index];
      auto *base = a.peer_bases[a.node * a.local_world_size +
                                target.destination_local_rank];
      return peer_at<transport::BFloat16>(base, a.canonical_rows_offset) +
             target.dst_row * a.hidden;
    };
    LaneStagingView<Storage> view(s, warp, lane);
    FanoutRowPrefetchPipeline<LaneStagingView<Storage>, const char *,
                             decltype(destination)>
        row_pipeline(view, 0U, destination);
    for (unsigned row = warp * Storage::kDispatchLanes + lane; row < task.rows;
         row += Storage::kCopyWarps * Storage::kDispatchLanes) {
      const auto *record = chunk + row * a.gin_rings.dispatch_row_bytes;
      const auto header = *reinterpret_cast<const ring::DispatchRow *>(record);
      const unsigned fanouts =
          min(header.fanouts, unsigned(a.dispatch.identity.top_k));
      // Write each fanout's metadata once, before readiness can be published.
      for (unsigned index = 0U; index < fanouts; ++index) {
        const auto target =
            reinterpret_cast<const ForwardDispatchFanoutRecord *>(
                record + sizeof(ring::DispatchRow))[index];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *base = a.peer_bases[destination];
        peer_at<LocalP2pRouteHeader>(base,
                                     a.direct_headers_offset)[target.dst_row] =
            LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                                source,
                                destination,
                                0,
                                a.dispatch.identity.generation,
                                execution_epoch(a),
                                0,
                                header.token * a.dispatch.identity.top_k +
                                    target.topk_slot,
                                header.token,
                                target.dst_row,
                                target.expert_id,
                                target.topk_slot};
        peer_at<float>(base, a.canonical_gates_offset)[target.dst_row] =
            target.gate;
      }
      row_pipeline.copy(
          record + a.gin_rings.dispatch_row_bytes - a.hidden * 2ULL, record,
          unsigned(a.hidden * 2ULL), fanouts);
    }
    row_pipeline.drain();
    finish_copies();
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  for (unsigned item = tid; item < items; item += Storage::kCopyThreads) {
    const auto *record = chunk + (item / a.dispatch.identity.top_k) *
                                     a.gin_rings.dispatch_row_bytes;
    const unsigned index = item % a.dispatch.identity.top_k;
    if (index >= reinterpret_cast<const ring::DispatchRow *>(record)->fanouts)
      continue;
    const auto target = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow))[index];
    local_notification_detail::Receiver<ForwardKernelArgs>::publish_dispatch(
        a, target.mtile, target.destination_local_rank);
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (tid == 0U) {
    auto *completion =
        static_cast<transport_state::GenerationCompletionVector *>(
            a.peer_bases[a.world]);
    claim_completion_tasks(completion, transport_state::TaskClass::RelayWave,
                           task.rows);
    complete_completion_tasks(completion, transport_state::TaskClass::RelayWave,
                              task.rows);
    finish_dispatch_receive(a, task);
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ProxyRelayEnd, ProxyRelay,
                             task.channel, task.rows, execution_epoch(a));
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void receive_return(const ForwardKernelArgs &a,
                                               Storage &s,
                                               const wire::Transfer &task) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
  constexpr unsigned issuers = WEFTMOE_FWD_GIN_PAYLOAD_ISSUER_LANES;
  static_assert(issuers >= 1U && issuers <= Storage::kDispatchLanes);
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, SourceRelayStart,
                             SourceRelay, task.channel, task.rows,
                             execution_epoch(a));
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  auto *output =
      peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
  if (lane < issuers) {
    LaneStagingView<Storage> view(s, warp, lane);
    RowPrefetchPipeline<LaneStagingView<Storage>> row_pipeline(view, 0U);
    for (unsigned item = warp * issuers + lane; item < task.rows;
         item += Storage::kCopyWarps * issuers) {
      const auto *record = chunk + item * a.gin_rings.return_row_bytes;
      const auto route =
          reinterpret_cast<const ring::ReturnRow *>(record)->route;
      row_pipeline.copy(record + sizeof(ring::ReturnRow),
               output + forward_contribution_row(a, route) * a.hidden, unsigned(a.hidden * 2ULL));
    }
    row_pipeline.drain();
    finish_copies();
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  for (unsigned item = tid; item < task.rows; item += Storage::kCopyThreads) {
    const auto *record = chunk + item * a.gin_rings.return_row_bytes;
    forward_reduce_detail::publish_ready_route(
        a, reinterpret_cast<const ring::ReturnRow *>(record)->route, 3U);
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (tid == 0U)
    wire::finish_receive(a, task);
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, SourceRelayEnd, SourceRelay,
                             task.channel, task.rows, execution_epoch(a));
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool
progress_send(const ForwardKernelArgs &a,
              persistent_v3::ForwardPipelineControl &control, Storage &s) {
  const unsigned tid = local_thread<BaseThread>();
  if (tid == 0U)
    s.gin_progressed =
        progress_context<(forward_tuning::kGinChannelPackLimit > 1U)>(
            a, s.gin_context_cursor, s.gin_context, s.gin_task_slot, s.gin_task,
            [&](const wire::Transfer &task) { account_send(a, task); },
            [&](unsigned context, GinSenderState &state) {
              complete_context(a, control, context, state);
            });
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (s.gin_task.channel != wire::kNoChannel) {
    if (s.gin_task.channel < a.gin_rings.dispatch_channels)
      pack_dispatch<BaseThread>(a, s, s.gin_task);
    else
      pack_return<BaseThread>(a, s, s.gin_task);
    if (tid == 0U)
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
          states(a)[s.gin_context].slots[s.gin_task_slot].pending)
          .store(2U, ::cuda::memory_order_release);
    copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  }
  return s.gin_progressed != 0U;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool progress_receive(const ForwardKernelArgs &a,
                                                 Storage &s, bool dispatch) {
  const unsigned tid = local_thread<BaseThread>();
  if (tid == 0U) {
    if (dispatch)
      claim_dispatch_receive(a, s.gin_rx_dispatch, s.gin_task);
    else
      wire::claim_receive(a, s.gin_rx_return, s.gin_task, false);
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  const bool received = s.gin_task.channel != wire::kNoChannel;
  // Every warp must capture an empty result before the leader can enter the
  // next receive poll and overwrite this shared task descriptor.
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (!received)
    return false;
  if (dispatch)
    receive_dispatch<BaseThread>(a, s, s.gin_task);
  else
    receive_return<BaseThread>(a, s, s.gin_task);
  return true;
}

} // namespace gin_detail
