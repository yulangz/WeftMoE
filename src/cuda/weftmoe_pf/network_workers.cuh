#pragma once

namespace weftmoe_pf::network {

namespace transport = weftmoe_pf::network_ring_transport;
using Endpoint = netring::Endpoint;

// Request storage aliases an otherwise unused pull staging buffer in an RDMA
// CTA. Only one owner uses a context; the request window is explicitly bounded.
struct Pending {
  ncclGinRequest_t request;
  std::uint64_t sequence;
  unsigned channel;
};
static_assert(kNetworkPending * sizeof(Pending) <= 4096U);

template <bool Get, bool Backward = false>
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
      if (!Get) {
        // Reserve a whole bounded source window independently of GEMM. Both
        // remote retirement and local NIC completion protect each TX slot.
        const auto end = min(terminal, cursor + endpoint.layout().depth);
        const auto credit = endpoint.gin.readSignal(
            a.gin_window, endpoint.credit(!Get, channel));
        while (state.tx_reserve_cursor < end &&
               static_forward::SlotSequence{state.tx_reserve_cursor,
                                            endpoint.layout().depth}
                   .reusable(credit, state.tx_completed_head)) {
          ++state.tx_reserve_cursor;
          ring::publish<::cuda::thread_scope_system>(state.tx_reserved_head,
                                                     state.tx_reserve_cursor);
          progress = true;
        }
      }
      if (cursor == terminal) {
        if (!Get)
          done &= endpoint.gin.readSignal(
                      a.gin_window, endpoint.credit(!Get, channel)) >= terminal;
        continue;
      }
      done = false;
      if (issued - completed == kNetworkPending)
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
        const auto *tokens =
            ring::source_metadata(a, plan.source, a.ring.source.node_tokens) +
            desc.token_begin;
        const auto destination =
            a.static_network.ring_region +
            endpoint.layout().dispatch_payload(true, channel, sequence);
        slot.retire_target += desc.fanouts;
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchGetStart,
                                   GinDispatch, channel, sequence, rows);
        for (unsigned row = 0U; row < rows; ++row)
          endpoint.gin.get(
              ncclTeamWorld(endpoint.gin.comm), endpoint.peer(channel),
              a.gin_window,
              a.static_network.input + tokens[row].token * row_bytes,
              a.gin_window, destination + row * row_bytes, row_bytes);
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchGetIssued,
                                   GinDispatch, channel, sequence, rows);

      } else {
        if (sequence >= state.tx_reserve_cursor)
          continue;
        const auto desc = ring::source_metadata(
            a, plan.source, a.ring.returning.packets)[packet];
        rows = desc.rows;
        const auto *routes =
            ring::source_metadata(a, plan.source, a.ring.source.route_order) +
            desc.route_begin;
        const auto *experts = ring::source_metadata(
            a, plan.source, a.ring.source.selected_experts);
        const auto *locations =
            ring::source_metadata(a, plan.source, a.ring.locations);
        bool ready = true;
        unsigned previous_index = ~0U;
        std::uint64_t previous_generation = 0U;
        for (unsigned row = 0U; row < rows; ++row) {
          const auto route = routes[row];
          const auto rank = unsigned(experts[route]) / owned;
          const auto &location = locations[route];
          const auto index = (rank % a.local_world_size) * a.ring.depth +
                             location.row / local::kM;
          // Reuse only a successful acquire for consecutive rows in this
          // immutable packet; never retain observations across polls.
          if (index == previous_index &&
              location.generation == previous_generation)
            continue;
          if (ring::acquire<::cuda::thread_scope_system>(
                  a.ring.return_ready[index]) != location.generation) {
            ready = false;
            break;
          }
          previous_index = index;
          previous_generation = location.generation;
        }
        if (!ready)
          continue;
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnPutStart, GinReturn,
                                   channel, sequence, rows);
        endpoint.send<Backward>(true, channel, sequence, rows * row_bytes);
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnGinPutIssue,
                                   GinReturn, channel, sequence, rows);
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
              ring::arrive<::cuda::thread_scope_system>(previous->returned,
                                                        count);
            previous = slot;
            count = 0U;
          }
          ++count;
        }
        if (count)
          ring::arrive<::cuda::thread_scope_system>(previous->returned, count);
      }
      auto &request = pending[issued % kNetworkPending];
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
        (issued - completed == kNetworkPending || !progress || done)) {
      auto &request = pending[completed % kNetworkPending];
      endpoint.gin.wait(request.request);
      if constexpr (Get) {
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchLocalCompletion,
                                   GinDispatch, request.channel,
                                   request.sequence, 0U);
      } else {
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnLocalCompletion,
                                   GinReturn, request.channel, request.sequence,
                                   0U);
      }
      if constexpr (Get) {

        auto &slot = endpoint.slot(false, request.channel, request.sequence);
        ring::publish<::cuda::thread_scope_system>(
            slot.visible_generation,
            request.sequence / endpoint.layout().depth + 1U);

      } else {
        endpoint.state(true, request.channel).tx_completed_head =
            request.sequence + 1U;
      }
      ++completed;
    }
  }
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(Get ? &status->gin_dispatch_put_count
                : &status->gin_return_put_count,
            packets);
  atomicAdd(Get ? &status->deduplicated_cross_node_payload_bytes
                : &status->gin_return_payload_bytes,
            bytes);
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
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, Fc2MblockReturnReady, GinReturn,
                             0U, block, 0U);
  unsigned rails = 0U, local_rows = 0U;

  local_rows = desc.local_rows;
  auto sources = unsigned(desc.remote_sources);
  const auto first = a.ring.source.shape.source_begin(desc.window);
  while (sources) {
    const auto source = first + __ffs(sources) - 1U;
    rails |= 1U << (source % a.local_world_size);
    sources &= sources - 1U;
  }

  while (rails) {
    const auto rail = __ffs(rails) - 1U;
    ring::publish<::cuda::thread_scope_system>(
        ring::peer(
            a, a.node * a.local_world_size + rail,
            a.ring.return_ready)[(a.rank % a.local_world_size) * a.ring.depth +
                                 desc.slot],
        desc.generation);
    rails &= rails - 1U;
  }
  if (local_rows)
    ring::arrive<::cuda::thread_scope_system>(slot.returned, local_rows);
  return true;
}

template <bool Backward = false>
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
        const auto &desc =
            a.ring.returning.packets[plan.begin + sequence - state.rx_base];
        if (endpoint.import_ready<Backward>(true, channel, sequence,
                                            desc.rows)) {
          WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, SourceSignalObserved,
                                     SourceRelay, channel, sequence, desc.rows);
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
  unsigned char *buffer;
  local::Barrier &full;
  unsigned phase = 0U;

  __device__ void operator()(const char *input, char *output, unsigned bytes) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    for (unsigned offset = 0; offset < bytes; offset += 4096U) {
      const auto size = min(4096U, bytes - offset);
      cute::SM90_BULK_COPY_G2S::copy(input + offset,
                                     reinterpret_cast<std::uint64_t *>(&full),
                                     buffer, size);
      full.arrive_and_expect_tx(size);
      full.wait(phase);
      phase ^= 1U;
      cute::SM90_BULK_COPY_S2G::copy(buffer, output + offset, size);
      cute::tma_store_arrive();
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  }
};

template <class Storage>
__device__ __noinline__ void pull(const ForwardKernelArgs &a, Storage &s,
                                  unsigned warp, unsigned worker,
                                  unsigned workers) {
  const auto lane = threadIdx.x % 32U;
  Copy copy{s.pull[warp], s.pull_full[warp]};
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto bytes = unsigned(a.hidden * 2U);
  auto *output = peer_at<char>(a.local_base, a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  const auto publish_row = [&](std::uint64_t block, unsigned m, float gate,
                               const char *input) {
    const auto row = ring::row(a, block) + m;
    if (!lane)
      copy(input, output + row * bytes, bytes);
    __syncwarp();
    if (!lane) {
      gates[row] = gate;
      const auto valid = r.blocks[block].tile.rows;
      local::arrive(ring::slot(a, block).x_full,
                    m + 1U == valid ? local::kM - valid + 1U : 1U);
    }
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
      const auto *input = static_cast<const char *>(
          ncclGetLsaPointer(a.gin_window, a.static_network.input,
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
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, !lane, ProxyRelayStart, ProxyRelay,
                                 worker, window * stride + task, packet.rows);

      // Route/metadata loads are warp-wide; only matching routes reach the
      // lane-0 bulk copier. Packet ownership and exact RX credit stay static.
      const auto top_k = unsigned(r.source.shape.top_k);
      for (unsigned first = 0U; first < packet.token_count * top_k;
           first += 32U) {
        const auto item = first + lane;
        const auto token = item / top_k;
        unsigned route = 0U;
        bool hit = false;
        if (token < packet.token_count) {
          route = packet.tokens[token].token * top_k + item % top_k;
          hit = unsigned(packet.experts[route]) / owned == a.rank;
        }
        unsigned long long block = 0U;
        unsigned row = 0U;
        float gate = 0.0f;
        if (hit) {
          block = packet.locations[route].block;
          row = packet.locations[route].row % local::kM;
          gate = packet.gates[route];
        }
        auto hits = __ballot_sync(0xffffffffU, hit);
        while (hits) {
          const auto owner = __ffs(hits) - 1U;
          publish_row(__shfl_sync(0xffffffffU, block, owner),
                      __shfl_sync(0xffffffffU, row, owner),
                      __shfl_sync(0xffffffffU, gate, owner),
                      static_cast<const char *>(packet.payload) +
                          __shfl_sync(0xffffffffU, token, owner) * bytes);
          hits &= hits - 1U;
        }
      }

      if (!lane)
        ring::arrive<::cuda::thread_scope_system>(*packet.consumed_total,
                                                  packet.rows);
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, !lane, ProxyRelayEnd, ProxyRelay,
                                 worker, window * stride + task, packet.rows);
    }
  }
}

} // namespace weftmoe_pf::network
