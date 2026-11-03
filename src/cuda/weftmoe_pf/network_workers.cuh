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

// Lane-major channel ownership interleaves remote nodes in each issue wave.
// It is a bijection of channels; each channel still has one FIFO owner.
__device__ __noinline__ void dispatch(const ForwardKernelArgs &a,
                                      Pending *pending, unsigned worker) {
  if (threadIdx.x % 32U)
    return;
  Endpoint endpoint(a, worker);
  const auto row_bytes = a.fp8_source_x ? fp8::source_pitch(unsigned(a.hidden)) : a.hidden * 2U;
  const auto peers = endpoint.layout().nodes - 1U;
  unsigned issued = 0U, completed = 0U;
  std::uint64_t bytes = 0U, packets = 0U;
  bool done = false;
  while (!done || completed != issued) {
    done = true;
    bool progress = false;
    for (unsigned index = worker; index < endpoint.layout().channels();
         index += kGetWorkers) {
      const auto channel =
          (index % peers) * endpoint.layout().lanes + index / peers;
      auto &state = endpoint.state(false, channel);
      const auto plan = transport::channel_plan<false, true>(a, channel);
      const auto terminal = state.rx_base + plan.count;
      if (state.rx_cursor == terminal)
        continue;
      done = false;
      if (issued - completed == kNetworkPending)
        break;
      const auto sequence = state.rx_cursor;
      auto &slot = endpoint.slot(false, channel, sequence);
      if (ring::acquire<::cuda::thread_scope_system>(slot.consumed_total) <
          slot.retire_target)
        continue;
      const auto packet = plan.begin + sequence - state.rx_base;
      const auto desc =
          ring::source_metadata(a, plan.source, a.ring.dispatch.packets)[packet];
      const auto *tokens =
          ring::source_metadata(a, plan.source, a.ring.source.node_tokens) +
          desc.token_begin;
      const auto destination =
          a.static_network.ring_region +
          endpoint.layout().dispatch_payload(channel, sequence);
      slot.retire_target += desc.fanouts;
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchGetStart,
                                 GinDispatch, channel, sequence, desc.tokens);
      for (unsigned row = 0U; row < desc.tokens;) {
        unsigned count = 1U;
#if defined(WEFTMOE_PF_COALESCE_GET) && WEFTMOE_PF_COALESCE_GET
        while (row + count < desc.tokens &&
               tokens[row + count].token == tokens[row].token + count)
          ++count;
#endif
        endpoint.gin.get(
            ncclTeamWorld(endpoint.gin.comm), endpoint.peer(channel),
            a.gin_window,
            a.static_network.input + tokens[row].token * row_bytes,
            a.gin_window, destination + row * row_bytes, count * row_bytes);
        row += count;
      }
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchGetIssued,
                                 GinDispatch, channel, sequence, desc.tokens);
      auto &request = pending[issued++ % kNetworkPending];
      request.channel = channel;
      request.sequence = sequence;
      endpoint.gin.flushAsync(ncclTeamWorld(endpoint.gin.comm),
                              endpoint.peer(channel), &request.request);
      ++state.rx_cursor;
      ++packets;
      bytes += desc.tokens * row_bytes;
      progress = true;
    }
    if (completed != issued &&
        (issued - completed == kNetworkPending || !progress || done)) {
      auto &request = pending[completed++ % kNetworkPending];
      endpoint.gin.wait(request.request);
      auto &slot = endpoint.slot(false, request.channel, request.sequence);
      ring::publish<::cuda::thread_scope_system>(
          slot.visible_generation,
          request.sequence / endpoint.layout().depth + 1U);
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchLocalCompletion,
                                 GinDispatch, request.channel,
                                 request.sequence, 0U);
    }
  }
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_dispatch_put_count, packets);
  atomicAdd(&status->deduplicated_cross_node_payload_bytes, bytes);
}

#ifndef WEFTMOE_PF_NETWORK_VECTOR_PULL
#define WEFTMOE_PF_NETWORK_VECTOR_PULL 0
#endif

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

  __device__ void warp_copy(const char *input, char *output, unsigned bytes) {
    const auto lane = threadIdx.x % 32U;
#if WEFTMOE_PF_NETWORK_VECTOR_PULL
    constexpr unsigned Batch = 4U;
    const auto *source = reinterpret_cast<const uint4 *>(input);
    auto *destination = reinterpret_cast<uint4 *>(output);
    for (unsigned column = lane; column < bytes / sizeof(uint4);
         column += 32U * Batch) {
      uint4 values[Batch];
#pragma unroll
      for (unsigned i = 0U; i < Batch; ++i)
        if (column + i * 32U < bytes / sizeof(uint4))
          values[i] = source[column + i * 32U];
#pragma unroll
      for (unsigned i = 0U; i < Batch; ++i)
        if (column + i * 32U < bytes / sizeof(uint4))
          destination[column + i * 32U] = values[i];
    }
    __threadfence();
#else
    if (!lane)
      (*this)(input, output, bytes);
#endif
    __syncwarp();
  }
};

template <bool WindowAdmission, class Publish>
__device__ __forceinline__ void pull_rows(const ForwardKernelArgs &a,
                                          unsigned worker, unsigned workers,
                                          Publish publish_row) {
  const auto lane = threadIdx.x % 32U;
  const auto &r = a.ring;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto bytes = a.fp8_source_x ? fp8::source_pitch(unsigned(a.hidden)) : unsigned(a.hidden * 2U);
  for (std::uint64_t window = 0; window < r.state->windows; ++window) {
    if constexpr (WindowAdmission)
      while (ring::acquire(r.state->admitted_windows) < window + 1U) {
      }
    const auto begin = WindowAdmission ? r.pull_prefix[window]
                                       : r.window_prefix[window] * local::kM;
    const auto end = WindowAdmission ? r.pull_prefix[window + 1U]
                                     : r.window_prefix[window + 1U] * local::kM;
    for (auto index = begin + worker; index < end; index += workers) {
      const auto row = WindowAdmission ? r.pull_order[index] : index;
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

// Packet rows are contiguous on the wire. Keep the payload on the bulk-copy
// path even when BF16's optional vector-copy experiment is enabled. The smaller
// profiles retain row-major source scales and copy the entire packet at once;
// their producer gathers K-block scales. Wider profiles retain column-major
// scales for TMA reuse across more GEMM tiles.
__device__ __forceinline__ void copy_fp8_row(
    Copy &copy, const char *input, std::uint8_t *output, unsigned row, unsigned hidden,
    unsigned pitch) {
  const unsigned lane = threadIdx.x % 32U;
  auto *q = fp8::ring_row(output, row, hidden, pitch);
  if constexpr (!local::kFP8NetworkRowScales) {
    if (!lane)
      copy(input, reinterpret_cast<char *>(q), hidden);
    __syncwarp();
    auto *s = fp8::ring_scales(output, row, hidden, pitch);
    for (unsigned c = lane; c < hidden / 128U; c += 32U)
      s[c * 128U] = reinterpret_cast<const float *>(input + hidden)[c];
    __threadfence();
    __syncwarp();
    return;
  }
  auto *s = reinterpret_cast<float *>(fp8::ring_row(output, row / 128U * 128U,
      hidden, pitch) + 128U * hidden) + row % 128U * (hidden / 128U);
  const unsigned scale_bytes = hidden / 128U * sizeof(float);
  if (hidden + scale_bytes <= local::kPullBufferBytes && scale_bytes % 16U == 0U) {
    if (!lane) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      cute::SM90_BULK_COPY_G2S::copy(input,
          reinterpret_cast<std::uint64_t *>(&copy.full), copy.buffer, hidden + scale_bytes);
      copy.full.arrive_and_expect_tx(hidden + scale_bytes);
      copy.full.wait(copy.phase);
      copy.phase ^= 1U;
      cute::SM90_BULK_COPY_S2G::copy(copy.buffer, q, hidden);
      cute::SM90_BULK_COPY_S2G::copy(copy.buffer + hidden, s, scale_bytes);
      cute::tma_store_arrive();
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  } else {
    if (!lane)
      copy(input, reinterpret_cast<char *>(q), hidden);
    __syncwarp();
    for (unsigned c = lane; c < hidden / 128U; c += 32U)
      s[c] = reinterpret_cast<const float *>(input + hidden)[c];
    __threadfence();
  }
  __syncwarp();
}

template <bool FP8, class Storage>
__device__ __noinline__ void pull(const ForwardKernelArgs &a, Storage &s,
                                  unsigned warp, unsigned worker,
                                  unsigned workers) {
  const auto lane = threadIdx.x % 32U;
  Copy copy{s.pull[warp], s.pull_full[warp]};
  const auto bytes = unsigned(a.hidden * 2U);
  auto *output = peer_at<char>(a.local_base, a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  pull_rows<true>(
      a, worker, workers,
      [&](std::uint64_t block, unsigned m, float gate, const char *input) {
        const auto row = ring::row(a, block) + m;
        if constexpr (FP8) {
          copy_fp8_row(copy, input, reinterpret_cast<std::uint8_t *>(output),
                       unsigned(row), unsigned(a.hidden), unsigned(a.fp8_x_pitch));
        } else
          copy.warp_copy(input, output + row * bytes, bytes);
        if (!lane) {
          gates[row] = gate;
          const auto valid = a.ring.blocks[block].tile.rows;
          local::arrive(ring::slot(a, block).x_full,
                        m + 1U == valid ? local::kM - valid + 1U : 1U);
        }
      });
}

} // namespace weftmoe_pf::network
