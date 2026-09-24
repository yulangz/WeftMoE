#pragma once

namespace static_v4::return_tx {

// Return packets are contiguous intervals of the source's expert-major order.
// Dispatch of this source window implies its contribution slot is reserved.
// The slot remains live until every token is reduced, independently of the
// packet control-slot credits used by the remote TX owner.
__device__ __forceinline__ std::size_t destination(
    const ForwardKernelArgs &a, unsigned source,
    const static_forward::ReturnPacketPlan &packet) {
  const auto &r = a.ring;
  const auto window = packet.window / r.source.shape.groups();
  const auto sequence = r.source_epochs[source].base + window;
  const auto row = (sequence % r.contribution_depth) * r.contribution_stride +
                   packet.route_begin - window * r.source.window_capacity();
  return a.static_network.contribution +
         row * a.hidden * sizeof(transport::BFloat16);
}

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
  const auto packets = static_forward::return_partition(
      a.ring.source.shape, a.ring.source_epochs[source].tokens);
  const auto packet = node_row / packets.rows;
  const auto lane = static_cast<unsigned>(packet % packets.lanes);
  const auto &network = a.static_network;
  const auto offset = network.ring_region +
      network.rings.return_payload(channel(a, source, lane),
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
  const auto local_window = window / p.shape.groups();
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto node_expert = a.node * a.local_world_size * owned;
  // G <= 16 and admission owns a full warp. Probe different source rails
  // concurrently, assigning each source its own disjoint lane residue.
  // The per-source worker count also handles G that does not divide 32.
  const auto source_begin = p.shape.source_begin(window);
  const auto sources = p.shape.source_end(window) - source_begin;
  const auto source_index = worker % sources;
  const auto source = source_begin + source_index;
  const auto source_workers = (workers - 1U - source_index) / sources + 1U;
  if (source / a.local_world_size == a.node ||
      local_window * p.shape.tokens >= r.source_epochs[source].tokens)
    return true;
  const auto packets = static_forward::return_partition(
      p.shape, r.source_epochs[source].tokens);
  const auto *prefix = ring::source_metadata(a, source, p.expert_prefix) +
                       local_window * (p.experts + 1U);
  const auto begin = prefix[a.rank * owned] - prefix[node_expert];
  const auto end = prefix[(a.rank + 1U) * owned] - prefix[node_expert];
  if (begin == end)
    return true;
  const auto first = begin / packets.rows;
  const auto last = (end - 1U) / packets.rows;
  const auto lanes = min(last - first + 1U, std::uint64_t(packets.lanes));
  for (std::uint64_t index = worker / sources; index < lanes;
       index += source_workers) {
    const auto lane = static_cast<unsigned>((first + index) % packets.lanes);
    const auto packet = (last - lane) / packets.lanes * packets.lanes + lane;
    const auto target = sequence(a, source, local_window, packet) + 1U;
    if (ring::acquire<::cuda::thread_scope_system>(
            state(a, source, lane).tx_reserved_head) < target)
      return false;
  }
  return true;
}

} // namespace static_v4::return_tx
