#pragma once

#include "cuda/communication/activation_ring.cuh"
#include "cuda/persistent_v3/forward/opportunistic_reduce.cuh"
#include "cuda/persistent_v3/forward/return_pipeline.cuh"

// Included by activation_transport.cu after the private Forward launch
// contract, GIN wire records, completion protocol, and peer helpers are
// visible.  This file contains only the RDMA/Relay Dispatch state
// machine; WGMMA lives in compute_kernel.cuh and cannot inherit these live
// ranges.

#if __has_include(<nccl_device.h>)

namespace forward_dispatch_detail {

inline constexpr std::uint32_t kLocalRouteBatch =
    forward_tuning::kLocalRouteBatchRows;
__device__ __forceinline__ bool
cancelled(const ForwardKernelArgs &a,
          persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

__device__ __forceinline__ std::uint64_t
local_route_prefix(const ForwardKernelArgs &a) {
  std::uint64_t begin = 0ULL;
  auto end = dispatch_sizes(a.dispatch).route_count;
  while (begin < end) {
    const auto middle = begin + (end - begin) / 2ULL;
    const auto route = a.dispatch_waves.local_route_order[middle];
    if (a.dispatch.route_dst_node[route] == a.node)
      begin = middle + 1ULL;
    else
      end = middle;
  }
  return begin;
}

// Incremental reverse-map stores target peer allocations during Dispatch.
// Before allowing any such store, prove that every GPU on this node has
// completed its generation-local map clear.  Block 0 publishes one epoch per
// GPU; each fixed transport CTA independently acquires the eight epochs, so
// this adds no grid-wide rendezvous and no compute CTA participates.
__device__ __forceinline__ void wait_for_local_map_clear(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    __threadfence_system();
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        control.generation.contexts[0].rx_published.value);
    ready.store(execution_epoch(a), ::cuda::memory_order_release);
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    const auto first_local = a.node * a.local_world_size;
    for (std::uint32_t local = 0U; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      auto *peer_control = peer_at<persistent_v3::ForwardPipelineControl>(
          a.peer_bases[peer], a.forward_v3_control_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          observed(peer_control->generation.contexts[0].rx_published.value);
      while (observed.load(::cuda::memory_order_acquire) < execution_epoch(a) &&
             !cancelled(a, control))
        __nanosleep(forward_tuning::kReduceQueueNanosleep);
    }
  }
  __syncthreads();
}

#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)

// Transport-only Hopper path. Each warp leader stages one 14-KiB row through
// the SMEM slot and mbarrier already reserved for local Return. Dispatch and
// Return progress serially in a Relay CTA, so this adds no shared memory and
// keeps twelve independent bulk-async streams in flight across the CTA.
__device__ __forceinline__ void publish_local_batch(
    const ForwardKernelArgs &a, std::uint64_t route_begin,
    std::uint64_t route_end,
    transport_state::GenerationCompletionVector *completion,
    std::byte *tma_storage,
    forward_return_detail::ReturnCopyBarrier *tma_full) {
  const auto routes = route_end - route_begin;
  constexpr std::uint32_t kWarps =
      forward_tuning::kLocalDispatchTmaWarps;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto copy_bytes = static_cast<std::uint32_t>(
      a.hidden * sizeof(transport::BFloat16));
  bool copied = false;
  for (std::uint64_t local = warp; warp < kWarps && local < routes;
       local += kWarps) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + local];
    if (lane == 0U) {
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      const auto token = route / a.dispatch.identity.top_k;
      const auto expert = a.dispatch.expert_id[route];
      const auto topk_slot = static_cast<std::uint32_t>(
          route % a.dispatch.identity.top_k);
      const auto gate = a.gates[route];
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      destination_headers[row] =
          LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                              a.rank,
                              destination,
                              0U,
                              a.dispatch.identity.generation,
                              execution_epoch(a),
                              route,
                              route,
                              token,
                              row,
                              expert,
                              topk_slot};
      auto *destination_gates = peer_at<float>(
          a.peer_bases[destination], a.canonical_gates_offset);
      destination_gates[row] = gate;
      auto *destination_rows = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.canonical_rows_offset);
      auto *warp_storage = tma_storage + warp * copy_bytes;
      auto &warp_full = tma_full[warp];
      ::cuda::device::experimental::cp_async_bulk_global_to_shared(
          warp_storage, a.input + token * a.hidden, copy_bytes, warp_full);
      auto token_barrier =
          ::cuda::device::barrier_arrive_tx(warp_full, 1U, copy_bytes);
      warp_full.wait(static_cast<decltype(token_barrier) &&>(token_barrier));
      ::cuda::device::experimental::cp_async_bulk_shared_to_global(
          destination_rows + row * a.hidden, warp_storage, copy_bytes);
      ::cuda::device::experimental::cp_async_bulk_commit_group();
      ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
      copied = true;
    }
  }
  if (copied) {
    ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
    __threadfence_system();
  }
  __syncthreads();
  local_notification_detail::publish_dispatch(
      a, unsigned(routes), blockDim.x, [&](unsigned i) {
        const auto route = a.dispatch_waves.local_route_order[route_begin + i];
        return local_notification_detail::Entry{
            a.dispatch.route_dst_rank[route] % a.local_world_size,
            a.dispatch_waves.route_dst_mtile[route]};
      });

  if (threadIdx.x == 0U) {
    unsigned long long peer_rows = 0ULL;
    for (auto position = route_begin; position < route_end; ++position) {
      const auto route = a.dispatch_waves.local_route_order[position];
      peer_rows += a.dispatch.route_dst_rank[route] != a.rank;
    }
    if (peer_rows != 0ULL) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(&status->actual_dispatch_peer_bytes,
                peer_rows * a.hidden * sizeof(transport::BFloat16));
    }
  }
}

#else

// Keep the established vector path in the compute translation unit.  The
// separately compiled transport object owns the evolving TMA implementation,
// so transport changes cannot perturb WGMMA register/stack allocation.
// Copy one node-local batch and release exact 128-row M-block counters only
// after every payload/header store made by the CTA is system-visible.  A warp
// owns a row, so the 14 KiB main-shape copy uses broad contiguous transactions
// without a per-row CTA barrier.
__device__ __forceinline__ void publish_local_batch(
    const ForwardKernelArgs &a, std::uint64_t route_begin,
    std::uint64_t route_end,
    transport_state::GenerationCompletionVector *completion) {
  const auto routes = route_end - route_begin;
  const auto vector_count = a.hidden / 8ULL;
  constexpr std::uint32_t kWarps = forward_tuning::kRelayScatterWarps;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  for (std::uint64_t local = warp; warp < kWarps && local < routes;
       local += kWarps) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + local];
    std::uint32_t destination = 0U;
    std::uint64_t row = 0ULL;
    std::uint64_t token = 0ULL;
    std::uint32_t expert = 0U;
    std::uint32_t topk_slot = 0U;
    if (lane == 0U) {
      destination = a.dispatch.route_dst_rank[route];
      row = a.dispatch.route_dst_row[route];
      token = route / a.dispatch.identity.top_k;
      expert = a.dispatch.expert_id[route];
      topk_slot = static_cast<std::uint32_t>(
          route % a.dispatch.identity.top_k);
      const auto gate = a.gates[route];
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      destination_headers[row] =
          LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                              a.rank,
                              destination,
                              0U,
                              a.dispatch.identity.generation,
                              execution_epoch(a),
                              route,
                              route,
                              token,
                              row,
                              expert,
                              topk_slot};
      auto *destination_gates = peer_at<float>(
          a.peer_bases[destination], a.canonical_gates_offset);
      destination_gates[row] = gate;
    }
    destination = __shfl_sync(0xffffffffU, destination, 0U);
    row = __shfl_sync(0xffffffffU, row, 0U);
    token = __shfl_sync(0xffffffffU, token, 0U);
    auto *destination_rows = peer_at<transport::BFloat16>(
        a.peer_bases[destination], a.canonical_rows_offset);
    auto *target = reinterpret_cast<uint4 *>(destination_rows) +
                   row * vector_count;
    const auto *source = reinterpret_cast<const uint4 *>(a.input) +
                         token * vector_count;
    for (std::uint64_t vector = lane; vector < vector_count; vector += 32ULL)
      target[vector] = source[vector];
  }

  __threadfence_system();
  __syncthreads();
  local_notification_detail::publish_dispatch(
      a, unsigned(routes), blockDim.x, [&](unsigned i) {
        const auto route = a.dispatch_waves.local_route_order[route_begin + i];
        return local_notification_detail::Entry{
            a.dispatch.route_dst_rank[route] % a.local_world_size,
            a.dispatch_waves.route_dst_mtile[route]};
      });

  if (threadIdx.x == 0U) {
    unsigned long long peer_rows = 0ULL;
    for (auto position = route_begin; position < route_end; ++position) {
      const auto route = a.dispatch_waves.local_route_order[position];
      peer_rows += a.dispatch.route_dst_rank[route] != a.rank;
    }
    if (peer_rows != 0ULL) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(&status->actual_dispatch_peer_bytes,
                peer_rows * a.hidden * sizeof(transport::BFloat16));
    }
  }
}

#endif // WEFTMOE_FORWARD_TRANSPORT_TU

// Notify tiles have a fixed row extent except for each expert's final tile.
// Cache the dependency interval once, alongside the immutable compute mapping.
__device__ __forceinline__ void
initialize_input_group(const ForwardKernelArgs &a, unsigned long long mblock,
                       std::uint32_t expert, std::uint32_t local_mblock) {
  auto &group = peer_at<persistent_v3::ForwardInputGroup>(
      a.local_base, a.forward_input_groups_offset)[mblock];
  group = persistent_v3::ForwardInputGroup{};
  const auto span = persistent_v3::input_group_span(
      local_mblock,
      static_cast<std::uint32_t>(forward_expert_mblock_count(a, expert)),
      forward_tuning::kStartupMblocks, a.forward_fc1_locality_group,
      a.forward_fc1_locality_group4_begin, a.forward_fc1_locality_group4_end);
  if (span.begin != local_mblock)
    return;
  const auto row_begin = forward_mblock_local_row(a, expert, local_mblock);
  const auto group_end = row_begin + 1ULL * span.count * a.forward_mblock_rows;
  const auto expert_rows = a.dispatch.expert_row_count[expert];
  const auto row_end = group_end < expert_rows ? group_end : expert_rows;
  const auto first_tile = a.dispatch.mtile_row_offsets[expert];
  const auto tile_rows = a.dispatch.mtile_expected_rows[first_tile];
  group.tile_cursor = first_tile + row_begin / tile_rows;
  group.tile_end = first_tile + (row_end + tile_rows - 1ULL) / tile_rows;
  group.mblocks = span.count;
}

// All reads are on this GPU. Failed polls need no acquire. Counters only grow
// during an execution, so once a tile is complete an acquire reload observes
// at least the same release sequence and makes every contributing row visible.
__device__ __forceinline__ bool
advance_input_group(const ForwardKernelArgs &a,
                    persistent_v3::ForwardInputGroup &group) {
  auto *arrived =
      peer_at<unsigned long long>(a.local_base, a.x_arrived_rows_offset);
  while (group.tile_cursor < group.tile_end) {
    unsigned long long rows = 0ULL;
    for (std::uint32_t rail = 0U; rail < a.local_world_size; ++rail) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> count(
          arrived[group.tile_cursor * a.local_world_size + rail]);
      rows += count.load(::cuda::memory_order_relaxed);
    }
    if (rows < a.dispatch.mtile_expected_rows[group.tile_cursor])
      return false;
    for (std::uint32_t rail = 0U; rail < a.local_world_size; ++rail) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> count(
          arrived[group.tile_cursor * a.local_world_size + rail]);
      (void)count.load(::cuda::memory_order_acquire);
    }
    ++group.tile_cursor;
  }
  return true;
}

// Each publisher owns disjoint input groups. Pending groups rotate through
// its local queue; completed groups are removed and expanded into the shared,
// device-local FC1 task queue. Copy workers only update the existing per-rail
// arrival counters, without consulting or modifying either queue.
__device__ __forceinline__ void
publish_ready_fc1_tasks(const ForwardKernelArgs &a, std::uint32_t context,
                        std::uint32_t publisher_count,
                        unsigned long long *scheduler) {
  const auto forward_mblocks = scheduler[3];
  const auto ntiles = static_cast<std::uint32_t>(
      (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
      forward_tuning::kFc1TileN);
  auto *groups = peer_at<persistent_v3::ForwardInputGroup>(
      a.local_base, a.forward_input_groups_offset);
  auto *claims =
      peer_at<unsigned long long>(a.local_base, a.forward_task_claim_offset);
  auto *ready_queue = claims + forward_mblocks;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
      scheduler[12ULL + 2ULL * a.world]);
  persistent_v3::ForwardInputPendingQueue pending;
  unsigned long long remaining = 0ULL;
  for (unsigned long long mblock = context; mblock < forward_mblocks;
       mblock += publisher_count) {
    if (groups[mblock].mblocks == 0U)
      continue;
    pending.push(groups, mblock);
    ++remaining;
  }
#if WEFTMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif
  while (!pending.empty() && !forward_cancelled(a)) {
    bool progressed = false;
    auto attempts = remaining;
    while (attempts-- != 0ULL && !pending.empty()) {
      const auto mblock = pending.head;
      auto &group = groups[mblock];
      const bool ready = advance_input_group(a, group);
      if (ready) {
        // Acquires for completed tiles survive queue rotations. The local
        // queue's release stores pass payload visibility to the compute CTA.
        const auto begin = tail.fetch_add(1ULL * group.mblocks * ntiles,
                                          ::cuda::memory_order_relaxed);
        for (std::uint32_t n = 0U; n < ntiles; ++n)
          for (std::uint32_t member = 0U; member < group.mblocks; ++member) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                entry(ready_queue[begin + 1ULL * n * group.mblocks + member]);
            entry.store(1ULL * (mblock + member) * ntiles + n + 1ULL,
                        ::cuda::memory_order_release);
          }
        --remaining;
        progressed = true;
#if WEFTMOE_DEBUG_VALIDATE
        atomicMax(&completion->bulk_pack_complete,
                  begin + 1ULL * group.mblocks * ntiles);
#endif
      }
      pending.advance(groups, ready);
    }
    if (!progressed)
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
#if WEFTMOE_DEBUG_VALIDATE
  atomicOr(&completion->bulk_contexts_issuing, 1ULL << context);
#endif
}
#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)

// One-node Forward has no GIN context owner. Each Relay CTA gives up to
// kLocalDispatchTmaWarps warps independent row-sized TMA streams and reserves
// the final warp's leader for destination-local FC1 queue publication. The
// two roles do not rendezvous until both their complete finite workspaces have
// drained, so incoming NVLink rows can release FC1 while this CTA is still
// sending later rows to its peers.
__device__ __forceinline__ void local_relay_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t relay_ordinal, unsigned long long *scheduler,
    unsigned char *storage) {
  wait_for_local_map_clear(a, control);
  WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay,
                          relay_ordinal, 0ULL, execution_epoch(a));
  constexpr std::uint32_t kPhysicalWarps =
      forward_tuning::kLocalTransportThreads / 32U;
  constexpr std::uint32_t kPublisherWarp = kPhysicalWarps - 1U;
  // The source default is the two-node production point and may use every
  // warp for rail-local Dispatch. A one-node launch has no RDMA owner, so it
  // must retain one publisher warp even when run without a world-8 generated
  // tuning header.
  constexpr std::uint32_t kCopyWarps =
      forward_tuning::kLocalDispatchTmaWarps < kPublisherWarp
          ? forward_tuning::kLocalDispatchTmaWarps
          : kPublisherWarp;
  constexpr std::uint32_t kCopyThreads = kCopyWarps * 32U;
  constexpr std::uint64_t kPublishBatchRows =
      forward_tuning::kDispatchPublishBatchRows;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto route_count = dispatch_sizes(a.dispatch).route_count;
  const auto copy_bytes = static_cast<std::uint32_t>(
      a.hidden * sizeof(transport::BFloat16));
  // Reuse the existing transport staging allocation. Wide rows or smaller
  // GEMM tiles retain one row per stream instead of growing shared memory.
  constexpr unsigned int kPrefetchRows = 2U;
  const auto rows_per_warp =
      kPrefetchRows * kCopyWarps * static_cast<std::size_t>(copy_bytes) <=
              forward_tuning::kGemmStorageBytes
          ? kPrefetchRows
          : 1U;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ forward_return_detail::ReturnCopyBarrier
      copy_full[kPhysicalWarps];
  __shared__ forward_return_detail::LocalReturnState
      local_return_state;
  __shared__ unsigned int local_return_done;
  if (warp < kCopyWarps && lane == 0U)
    init(&copy_full[warp], 1U);
  if (warp < kCopyWarps)
    ::cuda::device::experimental::fence_proxy_async_shared_cta();
  __syncwarp();

  unsigned long long peer_rows = 0ULL;
  if (warp < kCopyWarps) {
    const auto first_batch =
        static_cast<std::uint64_t>(relay_ordinal) * kPublishBatchRows;
    const auto batch_stride =
        static_cast<std::uint64_t>(a.forward_relay_cta_count) *
        kPublishBatchRows;
    for (auto batch_begin = first_batch; batch_begin < route_count;
         batch_begin += batch_stride) {
      const auto batch_end = min(batch_begin + kPublishBatchRows, route_count);

      // The ordering permutation is M-block major.  Give each TMA stream a
      // few rows from one contiguous batch, then publish that whole batch;
      // the former global warp-stride forced one system atomic per row and
      // destroyed the useful M-block runs already prepared by Notify.
      if (lane == 0U) {
        auto &full = copy_full[warp];
        for (auto position = batch_begin + warp; position < batch_end;
             position += rows_per_warp * kCopyWarps) {
          transport::BFloat16 *targets[kPrefetchRows] = {};
          unsigned int bytes = 0U;
#pragma unroll
          for (unsigned int item = 0U; item < kPrefetchRows; ++item) {
            const auto index = position + item * kCopyWarps;
            if (item < rows_per_warp && index < batch_end) {
              const auto route = a.dispatch_waves.local_route_order[index];
              const auto destination = a.dispatch.route_dst_rank[route];
              const auto row = a.dispatch.route_dst_row[route];
              const auto token = route / a.dispatch.identity.top_k;
              const auto gate = a.gates[route];
              auto *destination_headers = peer_at<LocalP2pRouteHeader>(
                  a.peer_bases[destination], a.direct_headers_offset);
              destination_headers[row] = LocalP2pRouteHeader{
                  transport::kActivationTransportSchemaVersion,
                  a.rank,
                  destination,
                  0U,
                  a.dispatch.identity.generation,
                  execution_epoch(a),
                  route,
                  route,
                  token,
                  row,
                  a.dispatch.expert_id[route],
                  static_cast<std::uint32_t>(route %
                                             a.dispatch.identity.top_k)};

              auto *destination_gates = peer_at<float>(
                  a.peer_bases[destination], a.canonical_gates_offset);
              destination_gates[row] = gate;

              auto *destination_rows = peer_at<transport::BFloat16>(
                  a.peer_bases[destination], a.canonical_rows_offset);
              targets[item] = destination_rows + row * a.hidden;
              ::cuda::device::experimental::cp_async_bulk_global_to_shared(
                  storage + (warp * rows_per_warp + item) * copy_bytes,
                  a.input + token * a.hidden, copy_bytes, full);
              bytes += copy_bytes;
            }
          }
          auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, bytes);
          full.wait(static_cast<decltype(ticket) &&>(ticket));
#pragma unroll
          for (unsigned int item = 0U; item < kPrefetchRows; ++item)
            if (targets[item])
              ::cuda::device::experimental::cp_async_bulk_shared_to_global(
                  targets[item],
                  storage + (warp * rows_per_warp + item) * copy_bytes,
                  copy_bytes);
          ::cuda::device::experimental::cp_async_bulk_commit_group();
          // The next prefetch may overwrite staging after the async reader
          // finishes; destination completion is only needed for publication.
          ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
        }
        ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
      }

      // Named barrier 2 contains only the copy warps.  The publisher warp can
      // keep polling ready M-blocks and appending FC1 tasks independently.
      asm volatile("bar.sync 2, %0;" : : "n"(kCopyThreads) : "memory");

      __threadfence_system();
      asm volatile("bar.sync 2, %0;" : : "n"(kCopyThreads) : "memory");
      local_notification_detail::publish_dispatch(
          a, unsigned(batch_end - batch_begin), kCopyThreads, [&](unsigned i) {
            const auto route =
                a.dispatch_waves.local_route_order[batch_begin + i];
            return local_notification_detail::Entry{
                a.dispatch.route_dst_rank[route] % a.local_world_size,
                a.dispatch_waves.route_dst_mtile[route]};
          });
      for (auto i = batch_begin + threadIdx.x; i < batch_end; i += kCopyThreads)
        peer_rows +=
            a.dispatch.route_dst_rank[a.dispatch_waves.local_route_order[i]] !=
            a.rank;
    }
  } else if (warp == kPublisherWarp && lane == 0U) {
    publish_ready_fc1_tasks(a, relay_ordinal, a.forward_relay_cta_count,
                            scheduler);
  }

  if (peer_rows != 0ULL) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicAdd(&status->actual_dispatch_peer_bytes,
              peer_rows * a.hidden * sizeof(transport::BFloat16));
  }
  __syncthreads();
  WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, relay_ordinal,
                          route_count, execution_epoch(a));
  WEFTMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay, relay_ordinal,
                          route_count, execution_epoch(a));
  if (threadIdx.x == 0U) {
    const auto old =
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.relay_dispatch_roles_complete.value, 1ULL);
    if (old + 1ULL == a.forward_relay_cta_count) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      status->dispatch_complete_clock = device_globaltimer_ns();
    }
  }

  // Once this CTA's finite Dispatch/publisher work is drained, reuse the
  // copy warps and the same row-sized SMEM slots for Return. All Relay CTAs
  // claim confirmed row batches from the local queue, independently of the
  // source rank. Compute can publish more rows concurrently.
  if (a.forward_relay_local_return) {
    const auto context = relay_ordinal % a.local_world_size;
    if (threadIdx.x == 0U) {
      local_return_state = forward_return_detail::LocalReturnState{};
      local_return_done = 0U;
      for (std::uint32_t return_warp = 0U;
           return_warp < kPhysicalWarps; ++return_warp)
        init(&copy_full[return_warp], 1U);
      ::cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    // Return copies need at most twelve warps. Reuse the independent
    // Dispatch publisher warp to consume notifications while copies run.
    constexpr unsigned kReturnThreads = (kPhysicalWarps - 1U) * 32U;
    if (warp == kPublisherWarp) {
      while (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                 local_return_done)
                     .load(::cuda::memory_order_acquire) == 0U &&
             !cancelled(a, control))
        if (!local_notification_detail::progress(a))
          __nanosleep(50);
    } else {
      while (local_return_done == 0U && !cancelled(a, control)) {
        const bool progressed = forward_return_detail::progress_local_return(
            a, control, context, scheduler, local_return_state,
            reinterpret_cast<std::byte *>(storage), copy_full, kReturnThreads);
        if (!progressed && threadIdx.x < 32U)
          forward_reduce_detail::try_reduce_one_token<true>(a);
        if (threadIdx.x == 0U) {
          if (forward_return_detail::local_return_terminal(
                  a, scheduler, local_return_state)) {
            ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(
                local_return_done)
                .store(1U, ::cuda::memory_order_release);
            WEFTMOE_TIMELINE_RECORD(
                a.timeline, LocalReturnTerminal, LocalReturn, context,
                local_return_state.cursor, execution_epoch(a));
          } else if (!progressed) {
            __nanosleep(forward_tuning::kReturnIdleNanosleep);
          }
        }
        forward_return_detail::copy_sync(kReturnThreads);
      }
    }
    __syncthreads();
  }
  local_notification_detail::drain(a);
  __syncthreads();
}

#endif
#include "cuda/persistent_v3/forward/gin_transport.cuh"

__device__ __forceinline__ void dispatch_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t forward_relay_cta_begin,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler, unsigned char *storage) {
  wait_for_local_map_clear(a, control);
  if (blockIdx.x < persistent_v3::kRdmaCtaCount) {
    rdma_role(a, control, static_cast<std::uint32_t>(blockIdx.x),
              completion, context_completion, scheduler);
    return;
  }
  const auto relay =
      static_cast<std::uint32_t>(blockIdx.x - forward_relay_cta_begin);
  relay_role(a, control, relay, completion, context_completion, scheduler,
             storage);
}

} // namespace forward_dispatch_detail

#endif // __has_include(<nccl_device.h>)
