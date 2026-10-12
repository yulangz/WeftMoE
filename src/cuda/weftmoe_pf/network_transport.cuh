#pragma once

// Immutable packet addresses shared by dispatch and return workers.
namespace weftmoe_pf::network_ring_transport {

namespace ring = weftmoe_pf::ring;
using Endpoint = weftmoe_pf::netring::Endpoint;

struct ChannelPlan {
  unsigned source;
  std::uint64_t begin;
  std::uint64_t count;
};

template <bool Returning, bool Receive>
__device__ __forceinline__ ChannelPlan channel_plan(const ForwardKernelArgs &a,
                                                    unsigned channel) {
  const auto &network = a.static_network.rings;
  const auto peer = network.peer(a.node, channel);
  constexpr bool PeerSource = Returning != Receive;
  const auto source =
      PeerSource ? peer * a.local_world_size + a.rank % a.local_world_size
                 : a.rank;
  const auto node = PeerSource ? a.node : peer;
  const auto *offsets =
      ring::source_metadata(a, source,
                            Returning ? a.ring.returning.channel_offsets
                                      : a.ring.dispatch.channel_offsets);
  const auto index = node * network.lanes + channel % network.lanes;
  return {source, offsets[index], offsets[index + 1U] - offsets[index]};
}

__device__ __forceinline__ static_forward::ComputeSlot &
c_slot(const ForwardKernelArgs &a, unsigned destination,
       const static_forward::RouteLocation &location) {
  return ring::peer(a, destination,
                    a.ring.slots)[location.row / forward_tuning::kTileM];
}

} // namespace weftmoe_pf::network_ring_transport
