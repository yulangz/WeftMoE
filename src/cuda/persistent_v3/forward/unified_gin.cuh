#pragma once

// Included inside forward_unified_transport after the bounded TMA-copy and
// communication-domain barrier helpers.
namespace gin_detail {
namespace wire = activation_ring_detail;
namespace ring = communication;

template <class Args>
__device__ __forceinline__ unsigned *sender_done(const Args &a) {
  return static_cast<unsigned *>(a.gin_sender_done);
}

// A sender owns every S-th channel, independently of sender allocation.
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
  // publishing this immutable task to their submitter. The acquire handoff
  // imports those releases, so GIN need not repeat the system release.
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
      ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_system,
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

__device__ __forceinline__ void
complete_sender(const ForwardKernelArgs &a,
                persistent_v3::ForwardPipelineControl &control,
                unsigned sender, GinSenderState &state) {
  (void)sender;
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
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
            state.slots[slot].pending)
                .load(::cuda::memory_order_relaxed) != 0U &&
        state.slots[slot].task.channel == channel)
      return true;
  return false;
}

template <bool Dispatch, bool SameChannel = false, class Args>
__device__ __forceinline__ bool choose_unpacked(const Args &a, unsigned sender,
                                                GinSenderState &state,
                                                wire::Transfer &task,
                                                unsigned max_probes = 0U) {
  const unsigned first = Dispatch ? 0U : a.gin_rings.dispatch_channels;
  const unsigned end = Dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.channels();
  const unsigned stride = a.gin_sender_count;
  const unsigned start = first + (sender + stride - first % stride) % stride;
  const unsigned count = start < end ? 1U + (end - 1U - start) / stride : 0U;
  auto &cursor = Dispatch ? state.dispatch_cursor : state.return_cursor;
  auto *channels = ring::channels(a.gin_window_base, a.gin_rings);
  const unsigned probes = max_probes == 0U ? count : min(count, max_probes);
  for (unsigned probe = 0; probe < probes; ++probe) {
    const unsigned channel = start + (cursor++ % count) * stride;
    auto &wire_state = channels[channel];
    if (wire_state.tx_terminal)
      continue;
    if constexpr (!SameChannel)
      if (channel_pending(state, channel))
        continue;
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
        if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
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
      task = wire::Hooks<Args>::returning(a, channel, sequence, position);
      if (!return_queue_detail::ready_block(a, queue, task.first - queue.begin,
                                            task.rows))
        continue;
    }
    return true;
  }
  return false;
}

// Reserve before submitting a previous packet, so the packer can execute
// concurrently with the expensive GIN call. The single submitter owns all
// descriptors; the mailbox release transfers their lifetime to the packer.
template <bool SameChannel = false, class Args>
__device__ __forceinline__ bool reserve_pack(const Args &a, GinSenderState &state,
                                             unsigned &selected_slot,
                                             wire::Transfer &task) {
  task.channel = wire::kNoChannel;
  if (state.done || blockIdx.x >= a.gin_sender_count)
    return false;
  unsigned free_slot = forward_tuning::kUnifiedGinPackSlots;
#pragma unroll
  for (unsigned probe = 0; probe < forward_tuning::kUnifiedGinPackSlots; ++probe) {
    const unsigned slot = (state.slot_cursor + probe) % forward_tuning::kUnifiedGinPackSlots;
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(state.slots[slot].pending)
            .load(::cuda::memory_order_relaxed) == 0U) {
      free_slot = slot;
      break;
    }
  }
  if (free_slot == forward_tuning::kUnifiedGinPackSlots)
    return false;
#if WEFTMOE_DEVICE_TIMELINE
  const auto select_start = profiling::read_globaltimer_ns();
#endif
  const bool dispatch_first = state.dispatch_first != 0U;
  bool selected = dispatch_first
      ? choose_unpacked<true, SameChannel>(a, blockIdx.x, state, task)
      : choose_unpacked<false, SameChannel>(a, blockIdx.x, state, task);
  if (!selected)
    selected = dispatch_first
        ? choose_unpacked<false, SameChannel>(a, blockIdx.x, state, task)
        : choose_unpacked<true, SameChannel>(a, blockIdx.x, state, task);
  if (!selected)
    return false;
#if WEFTMOE_DEVICE_TIMELINE
  if constexpr (::std::is_same_v<Args, ForwardKernelArgs>)
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, GinSenderSelect,
        GinReturn, task.channel, profiling::read_globaltimer_ns() - select_start,
        task.rows);
#endif
  state.dispatch_first = task.channel >= a.gin_rings.dispatch_channels;
  selected_slot = free_slot;
  state.slot_cursor = (free_slot + 1U) % forward_tuning::kUnifiedGinPackSlots;
  auto &entry = state.slots[free_slot];
  entry.task = task;
  ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(entry.pending)
      .store(1U, ::cuda::memory_order_relaxed);
  return true;
}

// Only this CTA advances its sender state; packers publish pending with release.
template <bool SameChannel = false, class Args, class OnSend, class OnComplete>
__device__ __forceinline__ bool
progress_sender(const Args &a, GinSenderState &state, unsigned &selected_slot,
                wire::Transfer &task, OnSend on_send, OnComplete on_complete,
                bool reserve_pack = true) {
  task.channel = wire::kNoChannel;
  bool progressed = false;
  const unsigned sender = blockIdx.x;
  if (sender >= a.gin_sender_count) return false;
  {
    if (state.done == 0U) {
      unsigned occupied = 0U;
      // Keep the large NCCL submission body out of an unrolled slot expansion.
#pragma unroll 1
      for (unsigned slot = 0; slot < forward_tuning::kUnifiedGinPackSlots;
           ++slot) {
        auto &entry = state.slots[slot];
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block> pending(
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
#if WEFTMOE_DEVICE_TIMELINE
          const auto submit_start = profiling::read_globaltimer_ns();
#endif
          send_packed(a, entry.task);
#if WEFTMOE_DEVICE_TIMELINE
          if constexpr (::std::is_same_v<Args, ForwardKernelArgs>)
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, GinSenderSubmit,
                GinReturn, entry.task.channel, profiling::read_globaltimer_ns() - submit_start,
                entry.task.rows);
#endif
          on_send(entry.task);
          state.terminal_puts += entry.task.terminal != 0U;
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
      const unsigned owned_channels = sender < a.gin_rings.channels()
          ? 1U + (a.gin_rings.channels() - 1U - sender) / a.gin_sender_count : 0U;
      // Do not scan old Dispatch slots on every Return submission. A sender
      // cannot drain before it has submitted the last packet of every channel.
      if (occupied == 0U && state.terminal_puts == owned_channels &&
          drained(a, sender)) {
        // Per-slot local counters and remote credits already prove that every
        // payload put is complete. This submitter issues no standalone ACKs.
        on_complete(sender, state);
        state.done = 1U;
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(sender_done(a)[sender])
            .store(1U, ::cuda::memory_order_release);
        progressed = true;
      } else if (reserve_pack && occupied < forward_tuning::kUnifiedGinPackSlots) {
        gin_detail::reserve_pack<SameChannel>(a, state, selected_slot, task);
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
  // Join every metadata writer and completed TMA issuer before importing
  // their writes into one system-scope release. The pending-slot release /
  // acquire then hands this visibility to the independent GIN submitter.
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (tid == 0U)
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
  // Join every metadata writer and completed TMA issuer before importing
  // their writes into one system-scope release. The pending-slot release /
  // acquire then hands this visibility to the independent GIN submitter.
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (tid == 0U)
    __threadfence_system();
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, ReturnPackEnd, GinReturn,
                             task.channel, task.rows, execution_epoch(a));
}

inline constexpr unsigned kReceiveParts =
    (ring::kChunkRows + forward_tuning::kGinReceiveRows - 1U) /
    forward_tuning::kGinReceiveRows;
inline constexpr bool kSplitReceive = kReceiveParts > 1U;

__device__ __forceinline__ bool
claim_dispatch_receive(const ForwardKernelArgs &a, unsigned &cursor,
                       wire::Transfer &task) {
  if constexpr (!kSplitReceive)
    return wire::claim_receive(a, cursor, task, true);
  const ncclGin gin(a.gin_dev_comm, 0);
  const unsigned probes = min(a.gin_rings.dispatch_channels,
                             unsigned(WEFTMOE_UNIFIED_GIN_CONTEXT_PROBES));
  for (unsigned probe = 0; probe < probes; ++probe) {
    const unsigned channel = cursor++ % a.gin_rings.dispatch_channels;
    auto &state = ring::channels(a.gin_window_base, a.gin_rings)[channel];
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(state.rx_terminal)
            .load(::cuda::memory_order_acquire))
      continue;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> head(
        state.dispatch_claimed_parts);
    auto ticket = head.load(::cuda::memory_order_relaxed);
    const auto sequence = state.dispatch_receive_base + ticket / kReceiveParts;
    const unsigned slot = sequence % ring::kRingDepth;
    if (!ring::reached(gin.readSignal(a.gin_window,
            a.gin_rings.slot_control(channel, slot) + offsetof(ring::ChunkSlot, ready_signal)),
            sequence / ring::kRingDepth + 1ULL, 64U))
      continue;
    if (!head.compare_exchange_strong(ticket, ticket + 1ULL,
                                     ::cuda::memory_order_relaxed))
      continue;
    const unsigned part = ticket % kReceiveParts;
    if (part + 1U == kReceiveParts)
      atomicMax(&state.claimed, sequence + 1ULL);
    // Every packet has the same number of tickets, including short/empty
    // tails. A successful ticket pins this physical slot until we finish.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const auto *header = reinterpret_cast<const ring::ChunkHeader *>(
        static_cast<const char *>(a.gin_window_base) +
        a.gin_rings.payload(channel, sequence, true));
    const unsigned begin = min(part * forward_tuning::kGinReceiveRows, header->rows);
    const unsigned rows = min(forward_tuning::kGinReceiveRows, header->rows - begin);
    task = wire::Transfer{channel, rows, header->terminal, sequence, begin};
    return true;
  }
  task.channel = wire::kNoChannel;
  return false;
}

__device__ __forceinline__ bool dispatch_receive_complete(const ForwardKernelArgs &a) {
  if constexpr (!kSplitReceive)
    return wire::receive_complete(a, true);
  auto &control = *peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  return persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.gin_dispatch_channels_complete.value) == a.gin_rings.dispatch_channels;
}

__device__ __forceinline__ void
finish_dispatch_receive(const ForwardKernelArgs &a, const wire::Transfer &task) {
  if constexpr (kSplitReceive) {
    auto &slot = ring::channels(a.gin_window_base, a.gin_rings)[task.channel]
                     .slots[task.sequence % ring::kRingDepth];
    const auto previous = ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        slot.consumed).fetch_add(1ULL, ::cuda::memory_order_acq_rel);
    if (previous % kReceiveParts + 1ULL != kReceiveParts)
      return;
    // The final acquire imports every reader's full TMA completion. Return
    // exactly one credit, only after all parts release this physical slot.
  }
  wire::finish_receive(a, task);
  if constexpr (kSplitReceive) {
    if (task.terminal) {
      auto &control = *peer_at<persistent_v3::ForwardPipelineControl>(
          a.local_base, a.forward_v3_control_offset);
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.gin_dispatch_channels_complete.value, 1ULL);
    }
  }
}

__device__ __forceinline__ void flush_receive(const ForwardKernelArgs &a) {
  wire::flush_receive(a);
  if constexpr (kSplitReceive) {
    // A stealing worker may credit any remote node's same-rail peer. Flush
    // only those extra connections, skipping the statically owned ones.
    const auto gin = wire::gin_for_cta(a, blockIdx.x);
    for (unsigned node = 0; node < a.dispatch.identity.num_nodes; ++node) {
      if (node == a.node) continue;
      const unsigned target = node * a.local_world_size + a.rank % a.local_world_size;
      bool owned = false;
      for (unsigned channel = blockIdx.x - a.gin_receiver_base;
           channel < a.gin_rings.channels(); channel += a.gin_receiver_count)
        owned |= wire::peer(a, channel) == target;
      if (owned) continue;
      ncclGinRequest_t request;
      gin.flushAsync(ncclTeamWorld(gin.comm), target, &request);
      gin.wait(request);
    }
  }
}

#ifndef WEFTMOE_FWD_GIN_FANOUT_LANES
#define WEFTMOE_FWD_GIN_FANOUT_LANES 1
#endif

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
  constexpr unsigned group_lanes =
      Storage::kDispatchLanes * WEFTMOE_FWD_GIN_FANOUT_LANES <= 32U
          ? WEFTMOE_FWD_GIN_FANOUT_LANES : 1U;
  const unsigned group = lane / group_lanes, member = lane % group_lanes;
  if (group < Storage::kDispatchLanes) {
    auto destination = [&a](const char *record, unsigned index) -> void * {
      const auto target =
          reinterpret_cast<const ForwardDispatchFanoutRecord *>(
              record + sizeof(ring::DispatchRow))[index];
      auto *base = a.peer_bases[a.node * a.local_world_size +
                                target.destination_local_rank];
      return peer_at<transport::BFloat16>(base, a.canonical_rows_offset) +
             target.dst_row * a.hidden;
    };
    LaneStagingView<Storage> view(s, warp, group);
    FanoutRowPrefetchPipeline<LaneStagingView<Storage>, const char *,
                             decltype(destination), group_lanes>
        row_pipeline(view, 0U, destination);
    for (unsigned row = warp * Storage::kDispatchLanes + group; row < task.rows;
         row += Storage::kCopyWarps * Storage::kDispatchLanes) {
      const auto *record = chunk + row * a.gin_rings.dispatch_row_bytes;
      const auto header = *reinterpret_cast<const ring::DispatchRow *>(record);
      const unsigned fanouts =
          min(header.fanouts, unsigned(a.dispatch.identity.top_k));
      // Write each fanout's metadata once, before readiness can be published.
      for (unsigned index = member; index < fanouts; index += group_lanes) {
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
    finish_copies<false>();
  }
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (tid == 0U)
    __threadfence_system();
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
        progress_sender<(forward_tuning::kGinChannelPackLimit > 1U)>(
            a, s.gin_sender_state, s.gin_task_slot, s.gin_task,
            [&](const wire::Transfer &task) { account_send(a, task); },
            [&](unsigned sender, GinSenderState &state) {
              complete_sender(a, control, sender, state);
            });
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  const bool packed = s.gin_task.channel != wire::kNoChannel;
  copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  if (packed) {
    if (s.gin_task.channel < a.gin_rings.dispatch_channels)
      pack_dispatch<BaseThread>(a, s, s.gin_task);
    else
      pack_return<BaseThread>(a, s, s.gin_task);
    if (tid == 0U)
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
          s.gin_sender_state.slots[s.gin_task_slot].pending)
          .store(2U, ::cuda::memory_order_release);
    copy_sync<Storage::kCopyThreads, Storage::kCopyBarrier>();
  }
  return s.gin_progressed != 0U;
}

#if WEFTMOE_DEVICE_TIMELINE
__device__ __forceinline__ void observe_return_ready(const ForwardKernelArgs &a) {
  if (!a.use_gin || !a.gin_receiver_count || blockIdx.x < a.gin_receiver_base) return;
  const unsigned first = a.gin_rings.dispatch_channels;
  const unsigned worker = blockIdx.x - a.gin_receiver_base;
  const unsigned stride = a.gin_receiver_count;
  const unsigned begin = first + (worker + stride - first % stride) % stride;
  const ncclGin gin(a.gin_dev_comm, 0U);
  for (unsigned c = begin; c < a.gin_rings.channels(); c += stride) {
    auto &state = ring::channels(a.gin_window_base, a.gin_rings)[c];
    const auto seq = ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(state.claimed)
        .load(::cuda::memory_order_relaxed);
    const unsigned slot = seq % ring::kRingDepth;
    const auto expected = seq / ring::kRingDepth + 1ULL;
    if (state.slots[slot].observed_ready == expected) continue;
    const auto ready = gin.readSignal(a.gin_window, a.gin_rings.slot_control(c, slot) +
        offsetof(ring::ChunkSlot, ready_signal));
    if (ring::reached(ready, expected, 64U)) {
      state.slots[slot].observed_ready = expected;
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, SourceSignalReady,
          SourceRelay, c, seq, 0ULL);
    }
  }
}
#endif

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool progress_receive(const ForwardKernelArgs &a,
                                                 Storage &s, bool dispatch) {
  // Ownership is uniform across this CTA. Workers without a channel in this
  // direction need neither a shared descriptor nor copy-domain rendezvous.
  if (a.gin_receiver_count && !(dispatch && kSplitReceive)) {
    const unsigned first = dispatch ? 0U : a.gin_rings.dispatch_channels;
    const unsigned count = dispatch ? a.gin_rings.dispatch_channels
                                    : a.gin_rings.return_channels;
    const unsigned offset = (blockIdx.x - a.gin_receiver_base +
        a.gin_receiver_count - first % a.gin_receiver_count) % a.gin_receiver_count;
    if (offset >= count)
      return false;
  }
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
  if (!dispatch)
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, SourcePacketClaimed,
        SourceRelay, s.gin_task.channel, s.gin_task.sequence, execution_epoch(a));
  if (dispatch)
    receive_dispatch<BaseThread>(a, s, s.gin_task);
  else
    receive_return<BaseThread>(a, s, s.gin_task);
  return true;
}

} // namespace gin_detail
