#pragma once

// Included inside communication_worker after the bounded TMA-copy and
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
      ncclGin_WeakVASignalAdd{a.gin_window,
                              a.gin_rings.slot_control(
                                  wire::remote_channel(a, task.channel), slot) +
                                  offsetof(ring::ChunkSlot, ready_signal),
                              1},
      ncclGin_WeakCounterInc{ring::counter(a.gin_rings, task.channel, slot)},
      ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_system,
      ::cuda::thread_scope_system);
  auto &channel = ring::channels(a.gin_window_base, a.gin_rings)[task.channel];
  ++channel.slots[slot].local_target;
  ++channel.sent;
  channel.tx_cursor +=
      (dispatch ? ring::kDispatchChannels : ring::kReturnPorts) *
      ring::kChunkRows;
  channel.tx_terminal = task.terminal;
}

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
__device__ __forceinline__ bool
choose_unpacked(const Args &a, unsigned sender, GinSenderState &state,
                wire::Transfer &task, unsigned max_probes = 0U) {
  const unsigned first = Dispatch ? 0U : a.gin_rings.dispatch_channels;
  const unsigned end =
      Dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.channels();
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
        !ring::can_send(gin, a.gin_window, a.gin_window_base, a.gin_rings,
                        channel, sequence))
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
                            reserved_cursor + 1ULL * ring::kDispatchChannels *
                                                  ring::kChunkRows >=
                                wire_state.tx_end,
                            sequence, reserved_cursor};
    } else {
      auto &queue = return_queue_detail::queues(a)[1U + wire::peer(a, channel)];
      const auto position = wire_state.tx_cursor +
                            (sequence - wire_state.sent) *
                                (1ULL * ring::kReturnPorts * ring::kChunkRows);
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
__device__ __forceinline__ bool
reserve_pack(const Args &a, GinSenderState &state, unsigned &selected_slot,
             wire::Transfer &task) {
  task.channel = wire::kNoChannel;
  if (state.done || blockIdx.x >= a.gin_sender_count)
    return false;
  unsigned free_slot = forward_tuning::kUnifiedGinPackSlots;
#pragma unroll
  for (unsigned probe = 0; probe < forward_tuning::kUnifiedGinPackSlots;
       ++probe) {
    const unsigned slot =
        (state.slot_cursor + probe) % forward_tuning::kUnifiedGinPackSlots;
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
            state.slots[slot].pending)
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
  bool selected =
      dispatch_first
          ? choose_unpacked<true, SameChannel>(a, blockIdx.x, state, task)
          : choose_unpacked<false, SameChannel>(a, blockIdx.x, state, task);
  if (!selected)
    selected =
        dispatch_first
            ? choose_unpacked<false, SameChannel>(a, blockIdx.x, state, task)
            : choose_unpacked<true, SameChannel>(a, blockIdx.x, state, task);
  if (!selected)
    return false;
#if WEFTMOE_DEVICE_TIMELINE
  if constexpr (::std::is_same_v<Args, ForwardKernelArgs>)
    WEFTMOE_TIMELINE_RECORD_IF(
        a.timeline, true, GinSenderSelect, GinReturn, task.channel,
        profiling::read_globaltimer_ns() - select_start, task.rows);
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

// Only this CTA advances its sender state; packers publish pending with
// release.
template <bool SameChannel = false, class Args, class OnSend, class OnComplete>
__device__ __forceinline__ bool
progress_sender(const Args &a, GinSenderState &state, unsigned &selected_slot,
                wire::Transfer &task, OnSend on_send, OnComplete on_complete,
                bool reserve_pack = true) {
  task.channel = wire::kNoChannel;
  bool progressed = false;
  const unsigned sender = blockIdx.x;
  if (sender >= a.gin_sender_count)
    return false;
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
                                        a.gin_rings)[entry.task.channel]
                             .sent;
        }
        if (may_submit) {
#if WEFTMOE_DEVICE_TIMELINE
          const auto submit_start = profiling::read_globaltimer_ns();
#endif
          send_packed(a, entry.task);
#if WEFTMOE_DEVICE_TIMELINE
          if constexpr (::std::is_same_v<Args, ForwardKernelArgs>)
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, GinSenderSubmit,
                                       GinReturn, entry.task.channel,
                                       profiling::read_globaltimer_ns() -
                                           submit_start,
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
      const unsigned owned_channels =
          sender < a.gin_rings.channels()
              ? 1U + (a.gin_rings.channels() - 1U - sender) / a.gin_sender_count
              : 0U;
      // Do not scan old Dispatch slots on every Return submission. A sender
      // cannot drain before it has submitted the last packet of every channel.
      if (occupied == 0U && state.terminal_puts == owned_channels &&
          drained(a, sender)) {
        // Per-slot local counters and remote credits already prove that every
        // payload put is complete. This submitter issues no standalone ACKs.
        on_complete(sender, state);
        state.done = 1U;
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
            sender_done(a)[sender])
            .store(1U, ::cuda::memory_order_release);
        progressed = true;
      } else if (reserve_pack &&
                 occupied < forward_tuning::kUnifiedGinPackSlots) {
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

} // namespace gin_detail
