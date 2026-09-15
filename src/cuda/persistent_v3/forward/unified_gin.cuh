#pragma once

// Included inside forward_unified_transport after the bounded TMA-copy and
// communication-domain barrier helpers.
namespace gin_detail {
namespace wire = activation_ring_detail;
namespace ring = communication;

template <class Args>
__device__ __forceinline__ GinContextState *states(const Args &a) {
  return static_cast<GinContextState *>(a.forward_unified_gin_state);
}

// Enumerate the three owned channels directly instead of repeatedly scanning
// all ranks' channels in every CTA's progress loop. This remains valid with
// more than one remote node and any fixed number of Return ports.
template <class Args>
__device__ __forceinline__ unsigned
owned_channel(const Args &a, unsigned context, unsigned index) {
  constexpr unsigned width = 1U + ring::kReturnPorts;
  const unsigned node = index / width;
  const unsigned slot = index % width;
  return slot == 0U
             ? node * ring::kDispatchChannels + context
             : a.gin_rings.dispatch_channels +
                   (node * a.local_world_size + context) * ring::kReturnPorts +
                   slot - 1U;
}

template <class Args>
__device__ __forceinline__ bool acknowledge(const Args &a, unsigned context) {
  const ncclGin gin(a.gin_dev_comm, context);
  const unsigned count =
      (a.dispatch.identity.num_nodes - 1U) * (1U + ring::kReturnPorts);
  bool progressed = false;
  for (unsigned index = 0U; index < count; ++index) {
    const unsigned channel = owned_channel(a, context, index);
    progressed =
        ring::acknowledge(gin, a.gin_window, a.gin_window_base, a.gin_rings,
                          channel, wire::remote_channel(a, channel),
                          wire::peer(a, channel)) ||
        progressed;
  }
  return progressed;
}

template <class Args>
__device__ __forceinline__ bool drained(const Args &a, unsigned context) {
  const unsigned count =
      (a.dispatch.identity.num_nodes - 1U) * (1U + ring::kReturnPorts);
  for (unsigned index = 0U; index < count; ++index) {
    const unsigned channel = owned_channel(a, context, index);
    const ncclGin gin(a.gin_dev_comm, wire::data_context(a, channel));
    if (!ring::drained(gin, a.gin_window, a.gin_window_base, a.gin_rings,
                       channel))
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
  const ncclGin gin(a.gin_dev_comm, wire::data_context(a, task.channel));
  const unsigned slot = task.sequence % ring::kRingDepth;
  // The packed release can originate from another CTA. All packer lanes have
  // completed TMA stores and system fences before the device release/acquire.
  gin.put(
      ncclTeamWorld(gin.comm), wire::peer(a, task.channel), a.gin_window,
      a.gin_rings.payload(wire::remote_channel(a, task.channel), task.sequence,
                          true),
      a.gin_window, a.gin_rings.payload(task.channel, task.sequence, false),
      bytes,
      ncclGin_StrongVASignalAdd{
          a.gin_window,
          a.gin_rings.slot_control(wire::remote_channel(a, task.channel),
                                   slot) +
              offsetof(ring::ChunkSlot, ready_signal),
          1},
      ncclGin_WeakCounterInc{ring::counter(a.gin_rings, task.channel, slot)},
      ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_device,
      ::cuda::thread_scope_system);
  auto &channel = ring::channels(a.gin_window_base, a.gin_rings)[task.channel];
  ++channel.sent;
  if (dispatch)
    channel.tx_cursor += ring::kDispatchChannels * ring::kChunkRows;
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
__device__ __forceinline__ void flush_context(const Args &a, unsigned context) {
  // drained() has already proved every data counter and returned credit.
  // Flushing here only joins outstanding ACK submissions for this context.
  ncclGin(a.gin_dev_comm, context).flush(ncclCoopThread{});
  for (unsigned port = 0; port < ring::kReturnPorts; ++port)
    ncclGin(a.gin_dev_comm, a.gin_context_count +
                                ring::kReturnPorts *
                                    ((context + a.rank) % a.gin_context_count) +
                                port)
        .flush(ncclCoopThread{});
}

__device__ __forceinline__ void
complete_context(const ForwardKernelArgs &a,
                 persistent_v3::ForwardPipelineControl &control,
                 unsigned context, GinContextState &state) {
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
__device__ __forceinline__ bool channel_pending(GinContextState &state,
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
                                                GinContextState &state,
                                                wire::Transfer &task,
                                                unsigned max_probes = 0U) {
  const auto nodes = a.dispatch.identity.num_nodes - 1U;
  const unsigned count = nodes * (Dispatch ? 1U : ring::kReturnPorts);
  auto &cursor = Dispatch ? state.dispatch_cursor : state.return_cursor;
  auto *channels = ring::channels(a.gin_window_base, a.gin_rings);
  const unsigned probes = max_probes == 0U ? count : min(count, max_probes);
  for (unsigned probe = 0; probe < probes; ++probe) {
    const auto index = cursor++ % count;
    const auto channel =
        Dispatch
            ? index * ring::kDispatchChannels + context
            : a.gin_rings.dispatch_channels +
                  (index / ring::kReturnPorts * a.local_world_size + context) *
                      ring::kReturnPorts +
                  index % ring::kReturnPorts;
    if constexpr (!SameChannel)
      if (channel_pending(state, channel))
        continue;
    auto &wire_state = channels[channel];
    auto sequence = wire_state.sent;
    if constexpr (SameChannel) {
      bool terminal_reserved = false;
      unsigned channel_reservations = 0U;
      // The context lease excludes reservation/submission mutations. Packers
      // may change pending 1->2, but each nonempty task is immutable and stays
      // occupied until this lease owner submits its exact sequence.
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
        !ring::can_send(gin, a.gin_window, a.gin_rings, channel, sequence))
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
      task = wire::Transfer{channel, unsigned(rows), rows == 0ULL,
                            sequence, reserved_cursor};
    } else {
      auto &queue = return_queue_detail::queues(a)[1U + wire::peer(a, channel)];
      const auto ready =
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
              queue.confirmed)
              .load(::cuda::memory_order_acquire);
      if (ready - queue.head < ring::kChunkRows && ready != queue.capacity)
        continue;
      const auto rows =
          unsigned(min(ready - queue.head, 1ULL * ring::kChunkRows));
      task = wire::Transfer{channel, rows, rows == 0U, sequence,
                            queue.begin + queue.head};
      // Both Return ports retain one common context lease. Their queue ranges
      // are disjoint even when their payload packers run in different CTAs.
      queue.head += rows;
    }
    return true;
  }
  return false;
}

// The context lease serializes only bounded metadata transitions and GIN
// submission. Every reserved slot hands packing to a different CTA outside the
// lease. The caller publishes completion to precisely that immutable slot.
template <bool SameChannel = false, class Args, class OnSend, class OnComplete>
__device__ __forceinline__ bool
progress_context(const Args &a, unsigned &cursor, unsigned &selected_context,
                 unsigned &selected_slot, wire::Transfer &task, OnSend on_send,
                 OnComplete on_complete, bool reserve_pack = true) {
  task.channel = wire::kNoChannel;
  bool progressed = false;
  static_assert(WEFTMOE_UNIFIED_GIN_CONTEXT_PROBES > 0);
  for (unsigned probe = 0;
       probe <
       min(a.gin_context_count, unsigned(WEFTMOE_UNIFIED_GIN_CONTEXT_PROBES));
       ++probe) {
    const unsigned context = cursor++ % a.gin_context_count;
    auto &state = states(a)[context];
    ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device> lease(
        state.lease);
    unsigned expected = 0U;
    if (!lease.compare_exchange_strong(expected, 1U,
                                       ::cuda::memory_order_acquire,
                                       ::cuda::memory_order_relaxed))
      continue;
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(state.done)
            .load(::cuda::memory_order_relaxed) == 0U) {
      progressed = acknowledge(a, context) || progressed;
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
    lease.store(0U, ::cuda::memory_order_release);
    if (task.channel != wire::kNoChannel)
      return true;
  }
  return progressed;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void pack_dispatch(const ForwardKernelArgs &a,
                                              Storage &s,
                                              const wire::Transfer &task) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U, lane = tid % 32U;
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
  if (lane == 0U) {
    RowPrefetchPipeline<Storage> row_pipeline(s, warp);
    for (unsigned item = warp; item < task.rows; item += Storage::kCopyWarps) {
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
  if (lane == 0U) {
    RowPrefetchPipeline<Storage> row_pipeline(s, warp);
    for (unsigned item = warp; item < task.rows; item += Storage::kCopyWarps) {
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
                      sizeof(ring::ChunkHeader);
  const auto source = wire::peer(a, task.channel);
  const unsigned items = task.rows * a.dispatch.identity.top_k;
  if (lane == 0U) {
    auto destination = [&a](const char *record, unsigned index) -> void * {
      const auto target =
          reinterpret_cast<const ForwardDispatchFanoutRecord *>(
              record + sizeof(ring::DispatchRow))[index];
      auto *base = a.peer_bases[a.node * a.local_world_size +
                                target.destination_local_rank];
      return peer_at<transport::BFloat16>(base, a.canonical_rows_offset) +
             target.dst_row * a.hidden;
    };
    FanoutRowPrefetchPipeline<Storage, const char *, decltype(destination)>
        row_pipeline(s, warp, destination);
    for (unsigned row = warp; row < task.rows; row += Storage::kCopyWarps) {
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
    wire::finish_receive(a, task);
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
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, SourceRelayStart,
                             SourceRelay, task.channel, task.rows,
                             execution_epoch(a));
  const auto *chunk = static_cast<const char *>(a.gin_window_base) +
                      a.gin_rings.payload(task.channel, task.sequence, true) +
                      sizeof(ring::ChunkHeader);
  auto *output =
      peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
  if (lane == 0U) {
    RowPrefetchPipeline<Storage> row_pipeline(s, warp);
    for (unsigned item = warp; item < task.rows; item += Storage::kCopyWarps) {
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
            [&](unsigned context, GinContextState &state) {
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
  if (tid == 0U)
    wire::claim_receive(a, dispatch ? s.gin_rx_dispatch : s.gin_rx_return,
                        s.gin_task, dispatch);
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
