#pragma once

#include "cuda/static_v4/network_transport.cuh"

// Pull has one issuer and two private staging slots. A batch stays within
// one Mblock and completes before publishing its aggregate X credit.
template <bool Network, class Storage>
__device__ __noinline__ void ring_pull(const ForwardKernelArgs &a, Storage &s,
                                       unsigned worker, unsigned workers) {
  namespace ring = static_v4::ring;
  const auto lane = threadIdx.x & 31U;
  if constexpr (!Network)
    if (lane != 0U)
      return;
  const auto &r = a.ring;
  const auto &shape = r.source.shape;
  auto *local =
      peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  RowPrefetchPipeline<Storage> prefetch(s, 0U);
  unsigned peer_rows = 0U;
  constexpr auto TileM = forward_tuning::kTileM;
  constexpr auto Batch = Network ? 2U : 8U;
  static_assert(TileM % Batch == 0U);
  for (std::uint64_t window = 0; window < r.state->windows; ++window) {
    const auto *sources = r.window_sources + window * shape.sources;
    const auto source_begin = shape.source_begin(window);
    const auto end = r.window_prefix[window + 1U] * TileM;
    if (lane == 0U)
      for (auto base = r.window_prefix[window] * TileM + Batch * worker;
           base < end; base += Batch * workers) {
        bool copied = false;
        unsigned credit = 0U;
#pragma unroll
        for (unsigned item = 0; item < Batch; ++item) {
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
          while (ring::acquire<::cuda::thread_scope_system>(
                     r.source_reserved[identity.source]) <
                 address.sequence + 1U) {
          }
          const auto target_row = ring::row(a, block) + m;
          gates[target_row] = identity.gate;
          const auto *source =
              peer_at<const transport::BFloat16>(a.peer_bases[identity.source],
                                                 a.source_input_offset) +
              std::uint64_t(identity.token) * a.hidden;
          asm volatile("fence.proxy.async.global;" ::: "memory");
          if (!copied) {
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ProxyRelayStart,
                                       ProxyRelay, 0U, base,
                                       execution_epoch(a));
            copied = true;
          }
          prefetch.copy(
              source, local + target_row * a.hidden,
              static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16)));
          peer_rows += identity.source != a.rank;
          const auto valid = r.blocks[block].tile.rows;
          credit += m + 1U == valid ? TileM - valid + 1U : 1U;
        }
        prefetch.drain();
        finish_copies<false>();
        if (copied)
          WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ProxyRelayEnd,
                                     ProxyRelay, 0U, base, execution_epoch(a));
        if (credit) {
          auto &slot = ring::slot(a, base / TileM);
          static_v4::reduce_add_release_device(
              reinterpret_cast<unsigned long long *>(&slot.x_full), credit);
        }
      }
    if constexpr (Network) {
      __syncwarp();
      network_ring_transport::pull_window(a, s, worker, workers, window,
                                          peer_rows);
    }
  }
  if (lane != 0U)
    return;
  if (peer_rows)
    atomicAdd(&status->actual_dispatch_peer_bytes,
              std::uint64_t(peer_rows) * a.hidden *
                  sizeof(transport::BFloat16));
  atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
}

// FC2 writes node-local contributions directly into their reserved source
// slots. This static owner publishes whole rows only after acquiring all N
// tiles; remote rows remain in C for the network Return owner to pack.
template <bool Network>
__device__ __noinline__ void ring_return(const ForwardKernelArgs &a,
                                         unsigned worker, unsigned workers) {
  namespace ring = static_v4::ring;
  const auto &r = a.ring;
  const auto lane = threadIdx.x & 31U;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  unsigned peer_rows = 0U;
  bool started = false;
  const auto blocks = r.window_prefix[r.state->windows];
  for (std::uint64_t block = worker; block < blocks; block += workers) {
    if constexpr (Network) {
      bool local = false;
      for (unsigned m = lane; m < r.blocks[block].tile.rows; m += 32U)
        local |= r.rows[block * forward_tuning::kTileM + m].source /
                     a.local_world_size ==
                 a.node;
      // A remote-only block may already be retired by its network owners.
      // We own no row in it and must not inspect its reusable C slot.
      if (!__any_sync(0xffffffffU, local))
        continue;
    }
    auto &slot = ring::slot(a, block);
    if (lane == 0U) {
      while (!ring::admitted(a, block)) {
      }
      while (ring::acquire(slot.fc2_done) < slot.fc2_target) {
      }
      // Acquire the device-local N-tile completion chain, then export those
      // peer stores before any lane publishes a source contribution flag.
      __threadfence_system();
      if (!started) {
        atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
        started = true;
      }
    }
    __syncwarp();
    const auto window = r.blocks[block].window;
    const auto route_begin =
        r.source.shape.token_begin(window) * r.source.shape.top_k;
    unsigned retired = 0U;
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, lane == 0U, LocalReturnBatchStart,
                               LocalReturn, 0U, block, execution_epoch(a));
    for (unsigned m = lane; m < r.blocks[block].tile.rows; m += 32U) {
      const auto identity = r.rows[block * forward_tuning::kTileM + m];
      if constexpr (Network)
        if (identity.source / a.local_world_size != a.node)
          continue;
      const auto &address = ring::window_source(a, window, identity.source);
      const auto row = address.row_base + identity.route - route_begin;
      ring::publish<::cuda::thread_scope_system>(
          ring::peer(a, identity.source, r.contribution_ready)[row],
          address.generation);
      ++retired;
      peer_rows += identity.source != a.rank;
    }
    // Every ready publication must finish before admission can reuse C.
    __syncwarp();
    retired = __reduce_add_sync(0xffffffffU, retired);
    if (lane == 0U && retired) {
      if constexpr (Network)
        static_v4::reduce_add_release_system(
            reinterpret_cast<unsigned long long *>(&slot.returned), retired);
      else
        static_v4::reduce_add_release_device(
            reinterpret_cast<unsigned long long *>(&slot.returned), retired);
    }
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, lane == 0U, LocalReturnBatchEnd,
                               LocalReturn, 0U, block, execution_epoch(a));
  }
  peer_rows = __reduce_add_sync(0xffffffffU, peer_rows);
  if (lane == 0U) {
    if (peer_rows)
      atomicAdd(&status->actual_return_peer_bytes,
                std::uint64_t(peer_rows) * a.hidden *
                    sizeof(transport::BFloat16));
    atomicMax(&status->return_complete_clock, device_globaltimer_ns());
  }
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
        ring_pull<true>(a, s, blockIdx.x - network_workers,
                        gridDim.x - network_workers);
    } else if (warp == 1U) {
      if (blockIdx.x < network_workers)
        network_ring_transport::transmit<true>(a, s, blockIdx.x);
      else if (blockIdx.x < 2U * network_workers)
        network_ring_transport::receive_return(a, s,
                                               blockIdx.x - network_workers);
      else
        ring_return<true>(a, blockIdx.x - 2U * network_workers,
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
    ring_pull<false>(a, s, blockIdx.x, gridDim.x);
  else if (warp == 1U)
    ring_return<false>(a, blockIdx.x, gridDim.x);
  else if (blockIdx.x == 0U)
    static_v4::ring::admit<false>(a, 2U * a.intermediate / Fc1N,
                                  a.hidden / Fc2N);
  else
    ring_reduce(a, blockIdx.x - 1U, gridDim.x - 1U);
}
