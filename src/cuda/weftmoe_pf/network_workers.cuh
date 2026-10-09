#pragma once

namespace weftmoe_pf::network {

namespace transport = forward_unified_transport::network_ring_transport;
using Endpoint = netring::Endpoint;

// Request storage aliases an otherwise unused pull staging buffer in an RDMA
// CTA. Only one owner uses a context; at most four packet prefixes are pending.
struct Pending {
  ncclGinRequest_t request;
  std::uint64_t sequence;
  unsigned channel;
};
static_assert(4U * sizeof(Pending) <= 4096U);

template <bool Get>
__device__ __noinline__ void transfer(const ForwardKernelArgs &a,
                                      Pending *pending, unsigned worker) {
  if (threadIdx.x % 32U)
    return;
  constexpr unsigned Workers = Get ? kGetWorkers : kPutWorkers;
  Endpoint endpoint(a, (Get ? 0U : kGetWorkers) + worker);
  const auto row_bytes = a.hidden * sizeof(weftmoe::transport::BFloat16);
  const auto owned = a.dispatch.sizes.owned_expert_count;
  unsigned issued = 0U, completed = 0U;
  std::uint64_t bytes = 0U, packets = 0U;
  bool done = false;
  while (!done || completed != issued) {
    done = true;
    bool progress = false;
    for (unsigned channel = worker; channel < endpoint.layout().channels();
         channel += Workers) {
      auto &state = endpoint.state(!Get, channel);
      const auto plan = Get ? transport::channel_plan<false, true>(a, channel)
                            : transport::channel_plan<true, false>(a, channel);
      auto &cursor = Get ? state.rx_cursor : state.tx_cursor;
      const auto base = Get ? state.rx_base : state.tx_base;
      const auto terminal = base + plan.count;
      if constexpr (!Get) {
        // Reserve a whole bounded source window independently of GEMM. Both
        // remote retirement and local NIC completion protect each TX slot.
        const auto end = min(terminal, cursor + endpoint.layout().depth);
        const auto credit = endpoint.gin.readSignal(
            a.gin_window, endpoint.credit(true, channel));
        while (state.tx_reserve_cursor < end &&
               static_forward::SlotSequence{state.tx_reserve_cursor,
                   endpoint.layout().depth}.reusable(credit,
                                                     state.tx_completed_head)) {
          ++state.tx_reserve_cursor;
          ring::publish<::cuda::thread_scope_system>(state.tx_reserved_head,
                                                     state.tx_reserve_cursor);
          progress = true;
        }
      }
      if (cursor == terminal) {
        if constexpr (!Get)
          done &= endpoint.gin.readSignal(a.gin_window,
                      endpoint.credit(true, channel)) >= terminal;
        continue;
      }
      done = false;
      if (issued - completed == 4U)
        break;
      const auto sequence = cursor;
      const auto packet = plan.begin + sequence - base;
      unsigned rows;
      if constexpr (Get) {
        auto &slot = endpoint.slot(false, channel, sequence);
        if (ring::acquire<::cuda::thread_scope_system>(slot.consumed_total) <
            slot.retire_target)
          continue;
        const auto desc = ring::source_metadata(
            a, plan.source, a.ring.dispatch.packets)[packet];
        rows = desc.tokens;
        const auto *tokens = ring::source_metadata(
            a, plan.source, a.ring.source.node_tokens) + desc.token_begin;
        const auto destination = a.static_network.ring_region +
            endpoint.layout().dispatch_payload(true, channel, sequence);
        slot.retire_target += desc.fanouts;
        for (unsigned row = 0U; row < rows; ++row)
          endpoint.gin.get(ncclTeamWorld(endpoint.gin.comm), endpoint.peer(channel),
              a.gin_window, a.static_network.input + tokens[row].token * row_bytes,
              a.gin_window, destination + row * row_bytes, row_bytes);
      } else {
        if (sequence >= state.tx_reserve_cursor)
          continue;
        const auto desc = ring::source_metadata(
            a, plan.source, a.ring.returning.packets)[packet];
        rows = desc.rows;
        const auto *routes = ring::source_metadata(
            a, plan.source, a.ring.source.route_order) + desc.route_begin;
        const auto *experts = ring::source_metadata(
            a, plan.source, a.ring.source.selected_experts);
        const auto *locations = ring::source_metadata(a, plan.source,
                                                       a.ring.locations);
        bool ready = true;
        for (unsigned row = 0U; row < rows; ++row) {
          const auto route = routes[row];
          const auto rank = unsigned(experts[route]) / owned;
          const auto &location = locations[route];
          if (ring::acquire<::cuda::thread_scope_system>(
                  a.ring.return_ready[(rank % a.local_world_size) * a.ring.depth +
                                      location.row / local::kM]) !=
              location.generation) {
            ready = false;
            break;
          }
        }
        if (!ready)
          continue;
        endpoint.send(true, channel, sequence, rows * row_bytes);
        // FC2 no longer owns these rows. TX storage remains protected by the
        // independent NIC completion and remote notification-slot credit.
        static_forward::ComputeSlot *previous = nullptr;
        unsigned count = 0U;
        for (unsigned row = 0U; row < rows; ++row) {
          const auto route = routes[row];
          auto *slot = &transport::c_slot(a, unsigned(experts[route]) / owned,
                                          locations[route]);
          if (slot != previous) {
            if (count)
              ring::arrive<::cuda::thread_scope_system>(previous->returned, count);
            previous = slot;
            count = 0U;
          }
          ++count;
        }
        if (count)
          ring::arrive<::cuda::thread_scope_system>(previous->returned, count);
      }
      auto &request = pending[issued % 4U];
      request.channel = channel;
      request.sequence = sequence;
      endpoint.gin.flushAsync(ncclTeamWorld(endpoint.gin.comm),
                              endpoint.peer(channel), &request.request);
      ++issued;
      ++cursor;
      ++packets;
      bytes += rows * row_bytes;
      progress = true;
    }
    if (completed != issued &&
        (issued - completed == 4U || !progress || done)) {
      auto &request = pending[completed % 4U];
      endpoint.gin.wait(request.request);
      if constexpr (Get) {
        auto &slot = endpoint.slot(false, request.channel, request.sequence);
        ring::publish<::cuda::thread_scope_system>(slot.visible_generation,
            request.sequence / endpoint.layout().depth + 1U);
      } else {
        endpoint.state(true, request.channel).tx_completed_head =
            request.sequence + 1U;
      }
      ++completed;
    }
  }
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(Get ? &status->gin_dispatch_put_count : &status->gin_return_put_count,
            packets);
  atomicAdd(Get ? &status->deduplicated_cross_node_payload_bytes
                : &status->gin_return_payload_bytes, bytes);
}

// Nonblocking block publication shares four workers with Return RX progress.
// Never wait for a GEMM block while holding up the network's credit path.
__device__ __forceinline__ bool publish_block(const ForwardKernelArgs &a,
                                               std::uint64_t block) {
  auto &slot = ring::slot(a, block);
  if (!ring::admitted(a, block) ||
      ring::acquire(slot.fc2_published) < slot.fc2_target)
    return false;
  const auto &desc = a.ring.blocks[block];
  unsigned rails = 0U, local_rows = 0U;
  for (unsigned m = 0U; m < desc.tile.rows; ++m) {
    const auto source = a.ring.rows[block * local::kM + m].source;
    if (source / a.local_world_size == a.node)
      ++local_rows;
    else
      rails |= 1U << (source % a.local_world_size);
  }
  while (rails) {
    const auto rail = __ffs(rails) - 1U;
    ring::publish<::cuda::thread_scope_system>(
        ring::peer(a, a.node * a.local_world_size + rail, a.ring.return_ready)[
            (a.rank % a.local_world_size) * a.ring.depth + desc.slot],
        desc.generation);
    rails &= rails - 1U;
  }
  if (local_rows)
    ring::arrive<::cuda::thread_scope_system>(slot.returned, local_rows);
  return true;
}

__device__ __noinline__ void progress(const ForwardKernelArgs &a,
                                      unsigned worker) {
  if (threadIdx.x % 32U)
    return;
  Endpoint endpoint(a, kGetWorkers + kPutWorkers + worker);
  std::uint64_t block = worker;
  bool done;
  do {
    if (block < a.ring.state->blocks && publish_block(a, block))
      block += kProgressWorkers;
    done = block >= a.ring.state->blocks;
    for (unsigned channel = worker; channel < endpoint.layout().channels();
         channel += kProgressWorkers) {
      auto &state = endpoint.state(true, channel);
      const auto plan = transport::channel_plan<true, true>(a, channel);
      const auto terminal = state.rx_base + plan.count;
      if (state.rx_cursor < terminal) {
        const auto sequence = state.rx_cursor;
        const auto &desc = a.ring.returning.packets[
            plan.begin + sequence - state.rx_base];
        if (endpoint.import_ready(true, channel, sequence, desc.rows)) {
          // Payload goes directly to immutable invocation-sized contribution
          // storage. Only the notification slot is retired here.
          ring::arrive(endpoint.slot(true, channel, sequence).consumed_total,
                        desc.rows);
          ++state.rx_cursor;
        }
      }
      endpoint.acknowledge(true, channel);
      done &= state.rx_cursor == terminal;
    }
  } while (!done);
}

// The same 4 KiB bulk-copy staging used by local PF. Each warp owns its slot;
// every copy finishes before X-ready or RX-reader retirement is published.
struct Copy {
  local::Shared &s;
  unsigned warp, phase = 0U;

  __device__ void operator()(const char *input, char *output, unsigned bytes) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    for (unsigned offset = 0; offset < bytes; offset += 4096U) {
      const auto size = min(4096U, bytes - offset);
      cute::SM90_BULK_COPY_G2S::copy(input + offset,
          reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]), s.pull[warp], size);
      s.pull_full[warp].arrive_and_expect_tx(size);
      s.pull_full[warp].wait(phase);
      phase ^= 1U;
      cute::SM90_BULK_COPY_S2G::copy(s.pull[warp], output + offset, size);
      cute::tma_store_arrive();
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  }
};

__device__ __noinline__ void pull(const ForwardKernelArgs &a, local::Shared &s,
                                 unsigned warp, unsigned worker,
                                 unsigned workers) {
  if (threadIdx.x % 32U)
    return;
  Copy copy{s, warp};
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto bytes = unsigned(a.hidden * 2U);
  auto *output = peer_at<char>(a.local_base, a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  const auto publish_row = [&](std::uint64_t block, unsigned m, float gate,
                                const char *input) {
    const auto row = ring::row(a, block) + m;
    gates[row] = gate;
    copy(input, output + row * bytes, bytes);
    const auto valid = r.blocks[block].tile.rows;
    local::arrive(ring::slot(a, block).x_full,
                  m + 1U == valid ? local::kM - valid + 1U : 1U);
  };
  for (std::uint64_t window = 0; window < r.state->windows; ++window) {
    while (ring::acquire(r.state->admitted_windows) < window + 1U) {
    }
    const auto begin = r.window_prefix[window] * local::kM;
    const auto end = r.window_prefix[window + 1U] * local::kM;
    for (auto row = begin + worker; row < end; row += workers) {
      const auto block = row / local::kM;
      const auto m = unsigned(row % local::kM);
      if (m >= r.blocks[block].tile.rows)
        continue;
      const auto identity = r.rows[row];
      if (identity.source / a.local_world_size != a.node)
        continue;
      const auto *input = static_cast<const char *>(ncclGetLsaPointer(
          a.gin_window, a.static_network.input,
          identity.source % a.local_world_size));
      publish_row(block, m, identity.gate, input + identity.token * bytes);
    }
    const auto stride = r.source.shape.sources *
                        r.source.packets.packets(r.source.shape.tokens);
    for (auto task = worker; task < stride; task += workers) {
      const auto &packet = r.pull_packets[window * stride + task];
      if (!packet.rows)
        continue;
      while (ring::acquire<::cuda::thread_scope_system>(
                 *packet.visible_generation) != packet.generation) {
      }
      for (unsigned token = 0U; token < packet.token_count; ++token)
        for (unsigned k = 0U; k < r.source.shape.top_k; ++k) {
          const auto route = packet.tokens[token].token * r.source.shape.top_k + k;
          if (unsigned(packet.experts[route]) / owned != a.rank)
            continue;
          const auto &location = packet.locations[route];
          publish_row(location.block, location.row % local::kM,
                      packet.gates[route],
                      static_cast<const char *>(packet.payload) + token * bytes);
        }
      ring::arrive<::cuda::thread_scope_system>(*packet.consumed_total, packet.rows);
    }
  }
}

} // namespace weftmoe_pf::network
