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

  __device__ std::size_t payload(bool returning, bool receive, unsigned channel,
                                 std::uint64_t sequence) const {
    return a.static_network.ring_region +
           layout().payload(returning, receive, channel, sequence);
  }

  __device__ unsigned counter(bool returning, unsigned channel,
                              std::uint64_t sequence) const {
    return a.static_network.counter_base +
           static_cast<unsigned>((returning ? layout().slots() : 0U) +
                                 channel * layout().depth +
                                 sequence % layout().depth);
  }

  __device__ unsigned peer(unsigned channel) const {
    return layout().peer(a.node, channel) * a.local_world_size +
           a.rank % a.local_world_size;
  }

  __device__ unsigned remote_channel(unsigned channel) const {
    return static_cast<unsigned>(layout().channel(
        layout().peer(a.node, channel), a.node, channel % layout().lanes));
  }

  __device__ bool can_send(bool returning, unsigned channel,
                           std::uint64_t sequence) const {
    const auto &s = slot(returning, channel, sequence);
    const auto credit = gin.readSignal(
        a.gin_window, control(returning, channel, sequence) +
                          offsetof(static_forward::NetworkSlot, credit_signal));
    const auto local = gin.readCounter(counter(returning, channel, sequence));
    return static_forward::SlotSequence{sequence, layout().depth}.reusable(
        credit, local, s.local_target);
  }

  // The caller has completed its TX writes, including all asynchronous copies.
  // Keep the remote consumed credit AND this context's local NIC completion.
  __device__ void send(bool returning, unsigned channel, std::uint64_t sequence,
                       std::size_t bytes) {
    const auto remote = remote_channel(channel);
    gin.put(ncclTeamWorld(gin.comm), peer(channel), a.gin_window,
            payload(returning, true, remote, sequence), a.gin_window,
            payload(returning, false, channel, sequence), bytes,
            // Each slot imports its own bundled Put. No consumer relies on
            // this signal to order other packets on the same context.
            ncclGin_WeakVASignalAdd{
                a.gin_window,
                control(returning, remote, sequence) +
                    offsetof(static_forward::NetworkSlot, ready_signal),
                1U},
            ncclGin_WeakCounterInc{counter(returning, channel, sequence)},
            ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_block,
            ::cuda::thread_scope_system);
    ++slot(returning, channel, sequence).local_target;
  }

  // readSignal acquires NIC payload visibility before the rail publishes the
  // generation to ordinary GPU/peer readers. A generic load is not a
  // substitute.
  __device__ bool import_ready(bool returning, unsigned channel,
                               std::uint64_t sequence) {
    const auto generation = sequence / layout().depth + 1U;
    if (gin.readSignal(a.gin_window, control(returning, channel, sequence) +
                                         offsetof(static_forward::NetworkSlot,
                                                  ready_signal)) < generation)
      return false;
    ring::publish<::cuda::thread_scope_system>(
        slot(returning, channel, sequence).visible_generation, generation);
    return true;
  }

  // Each physical RX slot retires independently, even when later slots finish
  // first. The target travels with that packet, never with a global done
  // prefix.
  __device__ void acknowledge(bool returning, unsigned channel) {
    for (unsigned index = 0U; index < layout().depth; ++index) {
      auto &s = slot(returning, channel, index);
      const auto generation = s.visible_generation;
      if (generation == s.acknowledged_generation)
        continue;
      const auto *header = peer_at<const static_forward::PacketHeader>(
          a.gin_window_base, payload(returning, true, channel, index));
      if (ring::acquire<::cuda::thread_scope_system>(s.consumed_total) <
          header->retire_target)
        continue;
      gin.signal(ncclTeamWorld(gin.comm), peer(channel),
                 ncclGin_StrongVASignalAdd{
                     a.gin_window,
                     control(returning, remote_channel(channel), index) +
                         offsetof(static_forward::NetworkSlot, credit_signal),
                     1U},
                 ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_thread,
                 ::cuda::thread_scope_system);
      s.acknowledged_generation = generation;
    }
  }

  __device__ bool drained(bool returning, bool receive, unsigned channel,
                          std::uint64_t terminal) const {
    for (unsigned index = 0U; index < layout().depth; ++index) {
      const auto target =
          terminal > index
              ? static_forward::ceil_div(terminal - index, layout().depth)
              : 0U;
      const auto &s = slot(returning, channel, index);
      if (receive) {
        if (s.acknowledged_generation < target)
          return false;
      } else if (!communication::reached(
                     gin.readSignal(a.gin_window,
                                    control(returning, channel, index) +
                                        offsetof(static_forward::NetworkSlot,
                                                 credit_signal)),
                     target, 64U) ||
                 !communication::reached(
                     gin.readCounter(counter(returning, channel, index)),
                     s.local_target)) {
        return false;
      }
    }
    return true;
  }
};

// Pool construction only. All generations and completion counters persist
// through graph replay; no reset occurs while a packet or reader is in flight.
static __global__ void initialize(ncclDevComm_t dev, ncclWindow_t window,
                                  NetworkLayout layout) {
  const ncclGin gin(dev, blockIdx.x);
  const auto &r = layout.rings;
  for (unsigned channel = 0U; channel < r.channels(); ++channel) {
    for (unsigned index = 0U; index < r.depth; ++index) {
      for (unsigned direction = 0U; direction < 2U; ++direction) {
        const bool returning = direction != 0U;
        gin.resetCounter(layout.counter_base +
                         static_cast<unsigned>((returning ? r.slots() : 0U) +
                                               channel * r.depth + index));
        if (channel % layout.ring_workers == blockIdx.x) {
          const auto offset =
              layout.ring_region + r.control(returning, channel, index);
          gin.resetSignal(window, offset + offsetof(static_forward::NetworkSlot,
                                                    ready_signal));
          gin.resetSignal(window, offset + offsetof(static_forward::NetworkSlot,
                                                    credit_signal));
        }
      }
    }
  }
}

} // namespace static_v4::netring
