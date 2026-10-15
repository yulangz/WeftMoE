#pragma once

namespace weftmoe_pf::network {

struct ReturnPending {
  ncclGinRequest_t request;
  std::uint64_t *consumed;
  unsigned rows;
};
struct ReturnScratch {
  ReturnPending pending[kNetworkPending];
  unsigned sources[local::kM];
  std::uint64_t contributions[local::kM];
};
static_assert(sizeof(ReturnScratch) <= 4096U);

// Each owner scans 32 interleaved expert blocks at once. One readiness probe
// per M-block replaces repeated remote probes for every source-ordered row.
// FC2 already wrote this rail's TX ring; only metadata is gathered here.
__device__ __noinline__ void return_blocks(const ForwardKernelArgs &a,
                                           ReturnScratch &s, unsigned worker) {
  Endpoint endpoint(a, kGetWorkers + worker);
  const auto &r = a.ring;
  const auto lane = threadIdx.x % 32U;
  const auto bytes = a.hidden * 2U;
  unsigned issued = 0U, completed = 0U;
  std::uint64_t packets = 0U, payload = 0U, remaining = 0U;
  for (auto base = std::uint64_t(worker) * 32U; base < r.state->return_tasks;
       base += kPutWorkers * 32U)
    remaining += min(std::uint64_t(32U), r.state->return_tasks - base);
  auto retire = [&]() {
    auto &entry = s.pending[completed++ % kNetworkPending];
    endpoint.gin.wait(entry.request);
    reduce_add_release_system(
        reinterpret_cast<unsigned long long *>(entry.consumed), entry.rows);
  };
  while (remaining) {
    for (auto base = std::uint64_t(worker) * 32U; base < r.state->return_tasks;
         base += kPutWorkers * 32U) {
      const auto index = base + lane;
      bool empty = false, ready = false;
      if (index < r.state->return_tasks) {
        auto &task = r.return_tasks[index];
        empty = task.count == 0U;
        if (empty)
          task.count = ~0U;
        else if (task.count != ~0U)
          ready = ring::acquire<::cuda::thread_scope_system>(*task.published) >=
                  task.target;
      }
      remaining -= __popc(__ballot_sync(0xffffffffU, empty));
      auto active = __ballot_sync(0xffffffffU, ready);
      while (active) {
        const auto task_id = base + unsigned(__ffs(active) - 1);
        auto &task = r.return_tasks[task_id];
        for (unsigned row = lane; row < task.count; row += 32U) {
          const auto identity = task.rows[row];
          s.sources[row] = identity.source;
          s.contributions[row] = identity.contribution;
        }
        __syncwarp();
        if (!lane) {
          for (unsigned row = 0U; row < task.count;) {
            const auto source = s.sources[row];
            if (source / a.local_world_size == a.node ||
                source % a.local_world_size != a.rank % a.local_world_size) {
              ++row;
              continue;
            }
            unsigned count = 1U;
            while (row + count < task.count &&
                   s.sources[row + count] == source &&
                   s.contributions[row + count] == s.contributions[row] + count)
              ++count;
            if (issued - completed == kNetworkPending)
              retire();
            const auto channel =
                endpoint.layout().channel(source / a.local_world_size, a.node,
                                          worker % endpoint.layout().lanes);
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnPutStart,
                                       GinReturn, channel, task_id, count);
            endpoint.gin.put(
                ncclTeamWorld(endpoint.gin.comm), source, a.gin_window,
                a.static_network.contribution + s.contributions[row] * bytes,
                a.gin_window, task.offset + std::uint64_t(row) * bytes,
                count * bytes,
                ncclGin_WeakVASignalAdd{
                    a.gin_window,
                    endpoint.control(true, channel, 0U) +
                        offsetof(static_forward::NetworkSlot, ready_signal),
                    count},
                ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_system);
            auto &entry = s.pending[issued++ % kNetworkPending];
            entry.consumed = task.returned;
            entry.rows = count;
            endpoint.gin.flushAsync(ncclTeamWorld(endpoint.gin.comm), source,
                                    &entry.request);
            ++packets;
            payload += count * bytes;
            row += count;
          }
          task.count = ~0U;
        }
        --remaining;
        active &= active - 1U;
        __syncwarp();
      }
    }
    if (!lane && completed != issued)
      retire();
    __syncwarp();
  }
  if (!lane) {
    while (completed != issued)
      retire();
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicAdd(&status->gin_return_put_count, packets);
    atomicAdd(&status->gin_return_payload_bytes, payload);
  }
}

// Immutable invocation-sized source C storage has no RX overwrite until the
// terminal barrier. Only bounded sender staging needs local NIC credit.
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
