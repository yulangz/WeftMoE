#pragma once

namespace static_v4::return_tx {

__device__ __forceinline__ unsigned channel(const ForwardKernelArgs &a,
                                            unsigned source, unsigned lane) {
  return a.static_network.rings.channel(a.node, source / a.local_world_size,
                                        lane);
}

__device__ __forceinline__ const static_forward::NetworkChannel &
state(const ForwardKernelArgs &a, unsigned source, unsigned lane) {
  const auto &network = a.static_network;
  return *static_cast<const static_forward::NetworkChannel *>(ncclGetLsaPointer(
      a.gin_window,
      network.ring_region + network.rings.state(true, channel(a, source, lane)),
      source % a.local_world_size));
}

__device__ __forceinline__ std::uint64_t sequence(
    const ForwardKernelArgs &a, unsigned source, std::uint64_t local_window,
    std::uint64_t packet) {
  const auto &r = a.ring;
  const auto lanes = r.returning.partition.lanes;
  const auto lane = static_cast<unsigned>(packet % lanes);
  const auto windows = static_forward::ceil_div(r.source_epochs[source].tokens,
                                                r.source.shape.tokens);
  const auto *prefix = ring::source_metadata(a, source, r.returning.packet_prefix);
  return state(a, source, lane).tx_base +
         prefix[(a.node * lanes + lane) * (windows + 1U) + local_window] +
         packet / lanes;
}

// source node/expert intervals already define Return's exact row order.
// Freeze the physical TX address before the bounded data plane starts.
__device__ __forceinline__ void *row_output(
    const ForwardKernelArgs &a, unsigned source, std::uint64_t local_window,
    std::uint64_t node_row) {
  const auto &packets = a.ring.returning.partition;
  const auto packet = node_row / packets.rows;
  const auto lane = static_cast<unsigned>(packet % packets.lanes);
  const auto &network = a.static_network;
  const auto offset = network.ring_region +
      network.rings.payload(true, false, channel(a, source, lane),
                            sequence(a, source, local_window, packet)) +
      (node_row % packets.rows) * a.hidden * sizeof(transport::BFloat16);
  return ncclGetLsaPointer(a.gin_window, offset, source % a.local_world_size);
}

// Each source's routes to this GPU form a contiguous node-row interval.
// Its last packet on each lane is enough: the reservation head is FIFO, and
// no intersecting packet can be sent/reused before this window's FC2 writes.
__device__ __forceinline__ bool reserved(const ForwardKernelArgs &a,
                                         std::uint64_t window, unsigned worker,
                                         unsigned workers) {
  const auto &r = a.ring;
  const auto &p = r.source;
  const auto &packets = r.returning.partition;
  const auto local_window = window / p.shape.groups();
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto node_expert = a.node * a.local_world_size * owned;
  for (auto source = p.shape.source_begin(window);
       source < p.shape.source_end(window); ++source) {
    if (source / a.local_world_size == a.node ||
        local_window * p.shape.tokens >= r.source_epochs[source].tokens)
      continue;
    const auto *prefix = ring::source_metadata(a, source, p.expert_prefix) +
                         local_window * (p.experts + 1U);
    const auto begin = prefix[a.rank * owned] - prefix[node_expert];
    const auto end = prefix[(a.rank + 1U) * owned] - prefix[node_expert];
    if (begin == end)
      continue;
    const auto first = begin / packets.rows;
    const auto last = (end - 1U) / packets.rows;
    for (unsigned lane = worker; lane < packets.lanes; lane += workers) {
      if (last < lane)
        continue;
      const auto packet = (last - lane) / packets.lanes * packets.lanes + lane;
      if (packet < first)
        continue;
      const auto target = sequence(a, source, local_window, packet) + 1U;
      if (ring::acquire<::cuda::thread_scope_system>(
              state(a, source, lane).tx_reserved_head) < target)
        return false;
    }
  }
  return true;
}

} // namespace static_v4::return_tx
