#pragma once

namespace weftmoe_pf::network {

struct ReturnPending {
  ncclGinRequest_t request;
  std::uint64_t block;
};
static_assert(kNetworkPending * sizeof(ReturnPending) <= 4096U);

// FC2 writes its own bounded M-slot, not a source-window packet. A cold tail
// can therefore wait for a later input window without pinning unrelated TX
// packets. The last local NIC completion releases this slot for reuse.
__device__ __noinline__ void return_blocks(const ForwardKernelArgs &a,
                                           ReturnPending *pending,
                                           unsigned worker) {
  if (threadIdx.x % 32U)
    return;
  Endpoint endpoint(a, kGetWorkers + worker);
  const auto &r = a.ring;
  const auto bytes = a.hidden * 2U;
  const auto tx = a.static_network.ring_region +
                  endpoint.layout().return_payload(0U, 0U);
  unsigned issued = 0U, completed = 0U;
  std::uint64_t packets = 0U, payload = 0U;
  auto retire = [&]() {
    auto &entry = pending[completed % kNetworkPending];
    endpoint.gin.wait(entry.request);
    if (entry.block != ~0ULL)
      ring::arrive(ring::slot(a, entry.block).returned,
                   r.blocks[entry.block].tile.rows);
    ++completed;
  };
  for (auto block = std::uint64_t(worker); block < r.state->blocks;
       block += kPutWorkers) {
    const auto desc = r.blocks[block];
    auto &slot = r.slots[desc.slot];
    while (ring::acquire(slot.generation) != desc.generation ||
           ring::acquire(slot.fc2_published) < desc.generation * local::kFc2Tiles)
      if (completed != issued)
        retire();
    const auto first_request = issued;
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, Fc2MblockReturnReady, GinReturn,
                               worker, block, 0U);
    const auto *rows = r.rows + block * local::kM;
    for (unsigned row = 0U; row < desc.tile.rows;) {
      const auto identity = rows[row];
      unsigned count = 1U;
      while (row + count < desc.tile.rows &&
             rows[row + count].source == identity.source &&
             rows[row + count].contribution == identity.contribution + count)
        ++count;
      const auto node = identity.source / a.local_world_size;
      if (node != a.node) {
        if (issued - completed == kNetworkPending)
          retire();
        const auto channel = endpoint.layout().channel(
            node, a.node, worker % endpoint.layout().lanes);
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnPutStart, GinReturn,
                                   channel, block, count);
        endpoint.gin.put(
            ncclTeamWorld(endpoint.gin.comm), identity.source, a.gin_window,
            a.static_network.contribution + identity.contribution * bytes,
            a.gin_window,
            tx + (std::uint64_t(desc.slot) * local::kM + row) * bytes,
            count * bytes,
            ncclGin_WeakVASignalAdd{
                a.gin_window,
                endpoint.control(true, channel, 0U) +
                    offsetof(static_forward::NetworkSlot, ready_signal),
                count},
            ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_system);
        auto &entry = pending[issued++ % kNetworkPending];
        entry.block = ~0ULL;
        endpoint.gin.flushAsync(ncclTeamWorld(endpoint.gin.comm),
                                identity.source, &entry.request);
        ++packets;
        payload += count * bytes;
      }
      row += count;
    }
    if (first_request != issued)
      pending[(issued - 1U) % kNetworkPending].block = block;
    else
      ring::arrive(slot.returned, desc.tile.rows);
  }
  while (completed != issued)
    retire();
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_return_put_count, packets);
  atomicAdd(&status->gin_return_payload_bytes, payload);
}

// Immutable invocation-sized source C storage has no RX overwrite until the
// terminal barrier. Only the bounded sender M-slot needs per-fragment credit.
__device__ __noinline__ void receive_blocks(const ForwardKernelArgs &a,
                                            unsigned worker) {
  Endpoint endpoint(a, kGetWorkers + kPutWorkers + worker);
  const auto lane = threadIdx.x % 32U;
  for (auto peer = worker; peer < endpoint.layout().nodes;
       peer += kProgressWorkers) {
    if (peer == a.node)
      continue;
    const auto channel = endpoint.layout().channel(a.node, peer, 0U);
    const auto target = endpoint.state(true, channel).forward_rows;
    while (true) {
      unsigned long long rows = 0U;
      for (unsigned sender = lane;
           sender < min(kPutWorkers, endpoint.layout().lanes); sender += 32U) {
        const auto control = endpoint.control(
            true, endpoint.layout().channel(a.node, peer, sender), 0U);
        rows += endpoint.gin.readSignal(
            a.gin_window,
            control + offsetof(static_forward::NetworkSlot, ready_signal));
      }
      for (unsigned offset = 16U; offset; offset /= 2U)
        rows += __shfl_down_sync(0xffffffffU, rows, offset);
      if (__shfl_sync(0xffffffffU, rows >= target, 0U)) {
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, !lane, SourceSignalObserved,
                                   SourceRelay, channel, target, 0U);
        break;
      }
    }
  }
}

} // namespace weftmoe_pf::network
