#pragma once

namespace weftmoe_pf::netring {

// One endpoint belongs to one static owner/context. No method claims work;
// cursors and exact terminal sequences come from the immutable packet plan.
struct Endpoint {
  const ForwardKernelArgs &a;
  ncclGin gin;

  __device__ Endpoint(const ForwardKernelArgs &args, unsigned context)
      : a(args),
        gin(args.gin_dev_comm, context, NCCL_GIN_RESOURCE_SHARING_THREAD) {}

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

  __device__ unsigned peer(unsigned channel) const {
    return layout().peer(a.node, channel) * a.local_world_size +
           a.rank % a.local_world_size;
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
          if (returning)
            gin.resetSignal(window,
                            offset + offsetof(static_forward::NetworkSlot,
                                              gradient_signal));
          if (index == 0U)
            gin.resetSignal(window, layout.ring_region +
                                        r.state(returning, channel) +
                                        offsetof(static_forward::NetworkChannel,
                                                 credit_signal));
        }
      }
    }
  }
}

} // namespace weftmoe_pf::netring
