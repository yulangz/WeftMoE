#pragma once

namespace static_v4::netring {

// One endpoint belongs to one static owner/context. No method claims work;
// cursors and exact terminal sequences come from the immutable packet plan.
struct Endpoint {
  const ForwardKernelArgs &a;
  ncclGin gin;

  __device__ Endpoint(const ForwardKernelArgs &args, unsigned context)
      : a(args), gin(args.gin_dev_comm, context,
                     NCCL_GIN_RESOURCE_SHARING_THREAD) {}

  __device__ const static_forward::NetworkRings &layout() const {
    return a.static_network.rings;
  }

  __device__ std::size_t control(bool returning, unsigned channel,
                                 std::uint64_t sequence) const {
    return a.static_network.ring_region +
           layout().control(returning, channel, sequence);
  }

  __device__ static_forward::NetworkSlot &slot(bool returning, unsigned channel,
                                               std::uint64_t sequence) const {
    return *peer_at<static_forward::NetworkSlot>(
        a.gin_window_base, control(returning, channel, sequence));
  }

  __device__ static_forward::NetworkChannel &state(bool returning,
                                                   unsigned channel) const {
    return *peer_at<static_forward::NetworkChannel>(
        a.gin_window_base,
        a.static_network.ring_region + layout().state(returning, channel));
  }

  __device__ std::size_t tx_payload(bool returning, unsigned channel,
                                    std::uint64_t sequence) const {
    return a.static_network.ring_region +
           (returning ? layout().return_payload(channel, sequence)
                      : layout().dispatch_payload(false, channel, sequence));
  }

  __device__ unsigned peer(unsigned channel) const {
    return layout().peer(a.node, channel) * a.local_world_size +
           a.rank % a.local_world_size;
  }

  __device__ std::size_t credit(bool returning, unsigned channel) const {
    return a.static_network.ring_region + layout().state(returning, channel) +
           offsetof(static_forward::NetworkChannel, credit_signal);
  }

  __device__ unsigned remote_channel(unsigned channel) const {
    return static_cast<unsigned>(layout().channel(
        layout().peer(a.node, channel), a.node, channel % layout().lanes));
  }

  // Only this owner issues on this context. A flush to the peer covers every
  // already-issued packet of this channel; other channel heads may lag safely.
  __device__ void complete_sends(bool returning, unsigned channel) const {
    auto &s = state(returning, channel);
    if (s.tx_completed_head == s.tx_cursor)
      return;
    ncclGinRequest_t request;
    gin.flushAsync(ncclTeamWorld(gin.comm), peer(channel), &request);
    gin.wait(request);
    s.tx_completed_head = s.tx_cursor;
  }

  __device__ bool can_send(bool returning, unsigned channel,
                           std::uint64_t sequence) const {
    const auto consumed = gin.readSignal(a.gin_window, credit(returning, channel));
    if (sequence >= layout().depth &&
        !communication::reached(consumed, sequence - layout().depth + 1U, 64U))
      return false;
    if (sequence >= layout().depth &&
        sequence - layout().depth >= state(returning, channel).tx_completed_head)
      complete_sends(returning, channel);
    // Reservation never runs beyond tx_cursor + depth: a previous occupant
    // is already issued and the completed head now covers its local read.
    return static_forward::SlotSequence{sequence, layout().depth}.reusable(
        consumed, state(returning, channel).tx_completed_head);
  }

  // The caller has completed its TX writes, including all asynchronous copies.
  // Reuse waits for remote consumed credit AND an explicit local flush.
  __device__ void send(bool returning, unsigned channel, std::uint64_t sequence,
                       std::size_t bytes) {
    const auto remote = remote_channel(channel);
    std::size_t destination;
    if (returning) {
      const auto source = peer(channel);
      const auto directory = a.node * layout().lanes + channel % layout().lanes;
      const auto begin = ring::source_metadata(
          a, source, a.ring.returning.channel_offsets)[directory];
      const auto &packet =
          ring::source_metadata(a, source, a.ring.returning.packets)[
              begin + sequence - state(true, channel).tx_base];
      destination = return_tx::destination(a, source, packet);
    } else {
      destination = a.static_network.ring_region +
                    layout().dispatch_payload(true, remote, sequence);
    }
    gin.put(ncclTeamWorld(gin.comm), peer(channel), a.gin_window, destination,
            a.gin_window,
            tx_payload(returning, channel, sequence), bytes,
            // Each slot imports its own bundled Put. No consumer relies on
            // this signal to order other packets on the same context.
            ncclGin_WeakVASignalAdd{
                a.gin_window,
                control(returning, remote, sequence) +
                    offsetof(static_forward::NetworkSlot, ready_signal),
                1U},
            ncclGin_None{},
            ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_block,
            ::cuda::thread_scope_system);
  }

  // readSignal acquires NIC payload visibility before the rail publishes the
  // generation to ordinary GPU/peer readers. A generic load is not a
  // substitute.
  __device__ bool import_ready(bool returning, unsigned channel,
                               std::uint64_t sequence,
                               const std::uint32_t &expected) {
    const auto generation = sequence / layout().depth + 1U;
    if (gin.readSignal(a.gin_window, control(returning, channel, sequence) +
                                         offsetof(static_forward::NetworkSlot,
                                                  ready_signal)) < generation)
      return false;
    auto &s = slot(returning, channel, sequence);
    s.retire_target += expected;
    ring::publish<::cuda::thread_scope_system>(s.visible_generation, generation);
    return true;
  }

  // Check retirement per slot, then coalesce the contiguous consumed prefix
  // into one ACK. TX issues in sequence, so an out-of-order credit cannot
  // unblock it before its head slot retires. Never infer another slot's
  // retirement from a ready signal or from the opposite-direction TX.
  __device__ void acknowledge(bool returning, unsigned channel) {
    auto &c = state(returning, channel);
    auto head = c.acknowledged_head;
    while (head < c.rx_cursor) {
      const auto &s = slot(returning, channel, head);
      if (ring::acquire<::cuda::thread_scope_system>(s.consumed_total) <
          s.retire_target)
        break;
      ++head;
    }
    if (head != c.acknowledged_head) {
      gin.signal(ncclTeamWorld(gin.comm), peer(channel),
                 ncclGin_StrongVASignalAdd{
                     a.gin_window,
                     credit(returning, remote_channel(channel)),
                     head - c.acknowledged_head},
                 ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_thread,
                 ::cuda::thread_scope_system);
      c.acknowledged_head = head;
    }
  }

  __device__ bool drained(bool returning, bool receive, unsigned channel,
                          std::uint64_t terminal) const {
    if (receive)
      return state(returning, channel).acknowledged_head == terminal;
    complete_sends(returning, channel);
    return communication::reached(
        gin.readSignal(a.gin_window, credit(returning, channel)), terminal, 64U);
  }
};

// Pool construction only. All generations and retirement counters persist
// through graph replay; no reset occurs while a packet or reader is in flight.
static __global__ void initialize(ncclDevComm_t dev, ncclWindow_t window,
                                  NetworkLayout layout) {
  const ncclGin gin(dev, blockIdx.x);
  const auto &r = layout.rings;
  for (unsigned channel = 0U; channel < r.channels(); ++channel) {
    for (unsigned index = 0U; index < r.depth; ++index) {
      for (unsigned direction = 0U; direction < 2U; ++direction) {
        const bool returning = direction != 0U;
        if (channel % layout.ring_workers == blockIdx.x) {
          const auto offset =
              layout.ring_region + r.control(returning, channel, index);
          gin.resetSignal(window, offset + offsetof(static_forward::NetworkSlot,
                                                    ready_signal));
          if (index == 0U)
            gin.resetSignal(
                window, layout.ring_region + r.state(returning, channel) +
                            offsetof(static_forward::NetworkChannel,
                                     credit_signal));
        }
      }
    }
  }
}

} // namespace static_v4::netring
