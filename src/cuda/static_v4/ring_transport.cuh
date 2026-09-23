#pragma once

#include "cuda/static_v4/network_transport.cuh"

// Included in forward_unified_transport. Pull and Return each have one issuer
// and two private staging slots, using the common row-prefetch pipeline.
// A batch stays within a fully admitted window and drains before publishing
// readiness/retirement; it cannot hold an earlier window across a capacity
// wait.
template <bool Return, bool Network, class Storage>
__device__ __noinline__ void ring_transfer(const ForwardKernelArgs &a,
                                           Storage &s, unsigned worker,
                                           unsigned workers) {
  namespace ring = static_v4::ring;
  const auto lane = threadIdx.x & 31U;
  if constexpr (Return || !Network)
    if (lane != 0U)
      return;
  const auto &r = a.ring;
  const auto &shape = r.source.shape;
  auto *local = peer_at<transport::BFloat16>(
      a.local_base,
      Return ? a.forward_output_rows_offset : a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  RowPrefetchPipeline<Storage> prefetch(s, Return ? 1U : 0U);
  unsigned peer_rows = 0U;
  bool started = false;
  constexpr auto TileM = forward_tuning::kTileM;
  for (std::uint64_t window = 0; window < r.state->windows; ++window) {
    const auto route_begin = shape.token_begin(window) * shape.top_k;
    const auto *sources = r.window_sources + window * shape.sources;
    const auto source_begin = shape.source_begin(window);
    const auto end = r.window_prefix[window + 1U] * TileM;
    if (lane == 0U)
      for (auto base = r.window_prefix[window] * TileM + 2ULL * worker;
           base < end; base += 2ULL * workers) {
        bool copied = false;
#pragma unroll
        for (unsigned item = 0; item < 2U; ++item) {
          const auto index = base + item;
          const auto block = index / TileM;
          const auto m = static_cast<unsigned>(index % TileM);
          if (index >= end || m >= r.blocks[block].tile.rows)
            continue;
          const auto identity = r.rows[index];
          if constexpr (Network)
            if (identity.source / a.local_world_size != a.node)
              continue;
          while (!ring::admitted(a, block)) {
          }
          const auto &address = sources[identity.source - source_begin];
          const void *source;
          void *destination;
          if constexpr (Return) {
            auto &slot = ring::slot(a, block);
            const auto target = slot.fc2_target;
            while (ring::acquire(slot.fc2_done) < target) {
            }
            if (!started) {
              atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
              started = true;
            }
            source = local + (ring::row(a, block) + m) * a.hidden;
            destination =
                peer_at<transport::BFloat16>(a.peer_bases[identity.source],
                                             a.contribution_offset) +
                (address.row_base + identity.route - route_begin) * a.hidden;
          } else {
            while (ring::acquire<::cuda::thread_scope_system>(
                       r.source_reserved[identity.source]) <
                   address.sequence + 1U) {
            }
            const auto target_row = ring::row(a, block) + m;
            gates[target_row] = identity.gate;
            source = peer_at<const transport::BFloat16>(
                         a.peer_bases[identity.source], a.source_input_offset) +
                     std::uint64_t(identity.token) * a.hidden;
            destination = local + target_row * a.hidden;
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
          if (!copied) {
            if constexpr (Return)
              WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true,
                                         LocalReturnBatchStart, LocalReturn, 0U,
                                         base, execution_epoch(a));
            else
              WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ProxyRelayStart,
                                         ProxyRelay, 0U, base,
                                         execution_epoch(a));
            copied = true;
          }
          prefetch.copy(
              source, destination,
              static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16)));
          peer_rows += identity.source != a.rank;
        }
        prefetch.drain();
        // Full bulk completion (not wait_group.read) imports async writes into
        // this issuer's generic proxy. The release publications below carry
        // payload visibility to readers; no additional SC fence is needed.
        finish_copies<false>();
        if (copied) {
          if constexpr (Return)
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, LocalReturnBatchEnd,
                                       LocalReturn, 0U, base,
                                       execution_epoch(a));
          else
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ProxyRelayEnd,
                                       ProxyRelay, 0U, base,
                                       execution_epoch(a));
        }
#pragma unroll
        for (unsigned item = 0; item < 2U; ++item) {
          const auto index = base + item;
          const auto block = index / TileM;
          const auto m = static_cast<unsigned>(index % TileM);
          if (index >= end || m >= r.blocks[block].tile.rows)
            continue;
          const auto identity = r.rows[index];
          if constexpr (Network)
            if (identity.source / a.local_world_size != a.node)
              continue;
          auto &slot = ring::slot(a, block);
          if constexpr (Return) {
            const auto &address = sources[identity.source - source_begin];
            const auto row = address.row_base + identity.route - route_begin;
            ring::publish<::cuda::thread_scope_system>(
                ring::peer(a, identity.source, r.contribution_ready)[row],
                address.generation);
            // Network Return may retire other rows of this same C slot from
            // a peer GPU. Both RMW scopes must include all of those writers.
            if constexpr (Network)
              static_v4::reduce_add_release_system(
                  reinterpret_cast<unsigned long long *>(&slot.returned), 1U);
            else
              static_v4::reduce_add_release_device(
                  reinterpret_cast<unsigned long long *>(&slot.returned), 1U);
          } else {
            const auto valid = r.blocks[block].tile.rows;
            const auto credit = m + 1U == valid ? TileM - valid + 1U : 1U;
            static_v4::reduce_add_release_device(
                reinterpret_cast<unsigned long long *>(&slot.x_full), credit);
          }
        }
      }
    if constexpr (Network && !Return) {
      __syncwarp();
      network_ring_transport::pull_window(a, s, worker, workers, window,
                                          peer_rows);
    }
  }
  if (lane != 0U)
    return;
  if (peer_rows)
    atomicAdd(Return ? &status->actual_return_peer_bytes
                     : &status->actual_dispatch_peer_bytes,
              std::uint64_t(peer_rows) * a.hidden *
                  sizeof(transport::BFloat16));
  atomicMax(Return ? &status->return_complete_clock
                   : &status->dispatch_complete_clock,
            device_globaltimer_ns());
}

__device__ __noinline__ void ring_reduce(const ForwardKernelArgs &a,
                                         unsigned worker, unsigned workers) {
  namespace ring = static_v4::ring;
  const auto &r = a.ring;
  const auto lane = threadIdx.x & 31U;
  auto view = forward_reduce_detail::reduce_token_view(a);
  bool started = false;
  const auto &shape = r.source.shape;
  for (std::uint64_t window = 0; window < r.source.local_windows(); ++window) {
    const auto &address = ring::window_source(
        a, window * shape.groups() + a.rank / shape.sources, a.rank);
    const auto begin = window * shape.tokens;
    const auto end = begin + static_forward::block_rows(r.source.tokens, window,
                                                        shape.tokens);
    unsigned completed = 0U;
    for (auto token = begin + worker; token < end; token += workers) {
      const auto first_row = address.row_base + (token - begin) * shape.top_k;
      if (lane == 0U)
        for (unsigned k = 0; k < shape.top_k; ++k)
          while (ring::acquire<::cuda::thread_scope_system>(
                     r.contribution_ready[first_row + k]) !=
                 address.generation) {
          }
      __syncwarp();
      if (!started && lane == 0U) {
        atomicMin(&view.status->reduced_tokens, device_globaltimer_ns());
        started = true;
      }
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, lane == 0U, ReduceStart, Reduce,
                                 0U, token, execution_epoch(a));
      forward_reduce_detail::reduce_token_payload<false>(view, token + 1U,
                                                         first_row);
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, lane == 0U, ReduceEnd, Reduce, 0U,
                                 token, execution_epoch(a));
      ++completed;
    }
    if (lane == 0U && completed)
      ring::arrive(r.contributions[address.slot].consumed, completed);
  }
  if (started && lane == 0U)
    atomicMax(&view.status->reduce_blocks_done, device_globaltimer_ns());
}

struct RingStaging
    : StaticV4Shared<128U, forward_tuning::kUnifiedSegmentBytes> {
  static constexpr unsigned kBuffers = 2U, kStorageSlots = 2U;
};

template <unsigned Fc1N, unsigned Fc2N, bool Network, class Storage>
__device__ __forceinline__ void ring_run(const ForwardKernelArgs &a,
                                         Storage &storage) {
  auto &s = *reinterpret_cast<RingStaging *>(&storage);
  static_assert(sizeof(s) <= sizeof(storage));
  const auto warp = (threadIdx.x - 288U) / 32U;
  const auto lane = threadIdx.x & 31U;
  if (warp < 2U && lane == 0U)
    for (unsigned slot = 0; slot < 2U; ++slot)
      init(&s.row_prefetch_full[2U * warp + slot], 1U);
  ::cuda::device::experimental::fence_proxy_async_shared_cta();
  domain_sync<96U>();
  if (a.dispatch.runtime->error)
    return;
  if constexpr (Network) {
    const auto network_workers = a.static_network.ring_workers;
    if (warp == 0U) {
      if (blockIdx.x < network_workers)
        network_ring_transport::transmit<false>(a, s, blockIdx.x);
      else
        ring_transfer<false, true>(a, s, blockIdx.x - network_workers,
                                   gridDim.x - network_workers);
    } else if (warp == 1U) {
      if (blockIdx.x < network_workers)
        network_ring_transport::transmit<true>(a, s, blockIdx.x);
      else if (blockIdx.x < 2U * network_workers)
        network_ring_transport::receive_return(a, s,
                                               blockIdx.x - network_workers);
      else
        ring_transfer<true, true>(a, s, blockIdx.x - 2U * network_workers,
                                  gridDim.x - 2U * network_workers);
    } else if (blockIdx.x == 0U) {
      static_v4::ring::admit<true>(a, 2U * a.intermediate / Fc1N,
                                   a.hidden / Fc2N);
    } else if (blockIdx.x <= network_workers) {
      network_ring_transport::receive_progress(a, blockIdx.x - 1U);
    } else {
      ring_reduce(a, blockIdx.x - network_workers - 1U,
                  gridDim.x - network_workers - 1U);
    }
    return;
  }
  if (warp == 0U)
    ring_transfer<false, false>(a, s, blockIdx.x, gridDim.x);
  else if (warp == 1U)
    ring_transfer<true, false>(a, s, blockIdx.x, gridDim.x);
  else if (blockIdx.x == 0U)
    static_v4::ring::admit<false>(a, 2U * a.intermediate / Fc1N,
                                  a.hidden / Fc2N);
  else
    ring_reduce(a, blockIdx.x - 1U, gridDim.x - 1U);
}
