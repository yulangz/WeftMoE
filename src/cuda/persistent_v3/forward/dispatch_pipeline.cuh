#pragma once

// Included by activation_transport.cu after the private Forward launch
// contract, GIN wire records, completion protocol, and peer helpers are
// visible.  This file contains only the fixed 8-RDMA/8-Relay Dispatch state
// machine; WGMMA lives in compute_kernel.cuh and cannot inherit these live
// ranges.

#if __has_include(<nccl_device.h>)

namespace forward_dispatch_detail {

inline constexpr std::uint32_t kLocalRouteBatch = 128U;
inline constexpr std::uint16_t kDispatchDirectoryTask = 1U;
// Warp 0 owns the GIN context and warp 1 contributes the scalar FC1-ready
// publisher.  The remaining ten complete warps used to sit idle while the
// paired Relay CTA both packed outbound packets and scattered inbound ones.
// Keep the two control warps out of every packing barrier so GIN progress and
// FC1 publication remain independent from the bulk HBM copy.
inline constexpr std::uint32_t kRdmaControlWarps = 2U;
inline constexpr std::uint32_t kRdmaPackWarps =
    persistent_v3::kWarpsPerCta - kRdmaControlWarps;
inline constexpr std::uint32_t kRdmaPackThreads = kRdmaPackWarps * 32U;

static_assert(kRdmaPackWarps == 10U);

__device__ __forceinline__ void rdma_pack_group_sync() {
  // Named barrier 1 belongs exclusively to threads [64, 384).  Warp 0 and
  // warp 1 execute long-lived scalar state machines and must never be forced
  // through a CTA-wide barrier while Dispatch is live.
  asm volatile("bar.sync 1, %0;" : : "n"(kRdmaPackThreads) : "memory");
}

__device__ __forceinline__ bool cancelled(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

__device__ __forceinline__ std::uint64_t
local_route_prefix(const ForwardKernelArgs &a) {
  std::uint64_t begin = 0ULL;
  auto end = a.dispatch.sizes.route_count;
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
    ready.store(a.epoch, ::cuda::memory_order_release);
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
      while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
             !cancelled(a, control))
        __nanosleep(128U);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ bool context_wave_span(
    const ForwardKernelArgs &a, std::uint32_t context,
    std::uint32_t remote_node, std::uint64_t &node_wave_begin,
    std::uint64_t &context_wave_begin, std::uint64_t &context_wave_end) {
  node_wave_begin = 0ULL;
  while (node_wave_begin < a.dispatch_waves.wave_count &&
         a.dispatch_waves.destination_node[node_wave_begin] < remote_node)
    ++node_wave_begin;
  auto node_wave_end = node_wave_begin;
  while (node_wave_end < a.dispatch_waves.wave_count &&
         a.dispatch_waves.destination_node[node_wave_end] == remote_node)
    ++node_wave_end;
  if (node_wave_begin == node_wave_end) {
    context_wave_begin = node_wave_begin;
    context_wave_end = node_wave_begin;
    return false;
  }
  const auto node_waves = node_wave_end - node_wave_begin;
  context_wave_begin =
      node_wave_begin + node_waves * context / persistent_v3::kRdmaCtaCount;
  context_wave_end = node_wave_begin +
                     node_waves * (context + 1ULL) /
                         persistent_v3::kRdmaCtaCount;
  return true;
}

#if defined(MEGAMOE_FORWARD_TRANSPORT_TU)

// Transport-only Hopper path. Each warp leader stages one 14-KiB row through
// the SMEM slot and mbarrier already reserved for local Return. Dispatch and
// Return progress serially in a Relay CTA, so this adds no shared memory and
// keeps twelve independent bulk-async streams in flight across the CTA.
__device__ __forceinline__ void publish_local_batch(
    const ForwardKernelArgs &a, std::uint64_t route_begin,
    std::uint64_t route_end,
    transport_state::GenerationCompletionVector *completion,
    std::byte *tma_storage,
    forward_return_detail::DirectLocalReturnBarrier *tma_full) {
  const auto routes = route_end - route_begin;
  constexpr std::uint32_t kWarps = persistent_v3::kWarpsPerCta;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto copy_bytes = static_cast<std::uint32_t>(
      a.hidden * sizeof(transport::BFloat16));
  bool copied = false;
  for (std::uint64_t local = warp; local < routes; local += kWarps) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + local];
    if (lane == 0U) {
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      const auto token = route / a.dispatch.identity.top_k;
      const auto expert = a.dispatch.expert_id[route];
      const auto topk_slot = static_cast<std::uint32_t>(
          route % a.dispatch.identity.top_k);
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      destination_headers[row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          a.rank,
          destination,
          0U,
          a.dispatch.identity.generation,
          a.epoch,
          route,
          route,
          token,
          row,
          expert,
          topk_slot};
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
  if (threadIdx.x < routes) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + threadIdx.x];
    const auto destination = a.dispatch.route_dst_rank[route];
    auto *arrived = peer_at<unsigned long long>(
        a.peer_bases[destination], a.x_arrived_rows_offset);
    const auto rail = a.rank % a.local_world_size;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        ready(arrived[a.dispatch_waves.route_dst_mtile[route] *
                          a.local_world_size +
                      rail]);
    ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
#if MEGAMOE_DEBUG_VALIDATE
    atomicAdd(&completion->local_rows_published, 1ULL);
    const auto tile = a.dispatch_waves.route_dst_mtile[route];
    if (tile < 64U)
      atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
  }
  __syncthreads();

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
  constexpr std::uint32_t kWarps = persistent_v3::kWarpsPerCta;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  for (std::uint64_t local = warp; local < routes; local += kWarps) {
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
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      destination_headers[row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          a.rank,
          destination,
          0U,
          a.dispatch.identity.generation,
          a.epoch,
          route,
          route,
          token,
          row,
          expert,
          topk_slot};
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
  if (threadIdx.x < routes) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + threadIdx.x];
    const auto destination = a.dispatch.route_dst_rank[route];
    auto *arrived = peer_at<unsigned long long>(
        a.peer_bases[destination], a.x_arrived_rows_offset);
    const auto rail = a.rank % a.local_world_size;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        ready(arrived[a.dispatch_waves.route_dst_mtile[route] *
                          a.local_world_size +
                      rail]);
    // Remote and local rows owned by the same rail can be released by
    // different Relay CTAs.  Chain their stores through the counter's
    // modification order so the final acquire sees every contributing row.
    ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
#if MEGAMOE_DEBUG_VALIDATE
    atomicAdd(&completion->local_rows_published, 1ULL);
    const auto tile = a.dispatch_waves.route_dst_mtile[route];
    if (tile < 64U)
      atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
  }
  __syncthreads();

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

#endif // MEGAMOE_FORWARD_TRANSPORT_TU

// RDMA-resident variant of pack_outbound_wave.  Threads [64, 384) form an
// independent ten-warp bulk group; the GIN owner in warp 0 can issue wave N
// while this group materializes wave N+1, and warp 1 can keep publishing FC1
// readiness.  The wire layout intentionally remains byte-for-byte identical
// to the compatibility Relay implementation; only the handoff is now a
// monotonic sequence prefix rather than a descriptor queue.
__device__ __forceinline__ void pack_outbound_wave_from_rdma(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context, std::uint64_t node_packet_begin,
    std::uint64_t wave, std::uint64_t packed_count) {
  const auto pack_thread = threadIdx.x - kRdmaControlWarps * 32U;
  const auto pack_warp = pack_thread / 32U;
  const auto lane = pack_thread & 31U;
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_send_header_offset);
  auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_send_fanout_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_send_payload_offset);
  auto *source_compact_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_source_compact_index_offset);
  auto *source_route_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_source_route_index_offset);
  auto *direct_source_compact_index = peer_at<unsigned long long>(
      a.local_base, a.forward_direct_source_compact_index_offset);
  const auto wave_begin = a.dispatch_waves.offsets[wave];
  const auto wave_end = a.dispatch_waves.offsets[wave + 1ULL];
  const auto wave_packets = wave_end - wave_begin;
  const auto vector_count = a.hidden / 8ULL;

  MEGAMOE_TIMELINE_RECORD_IF(a.timeline, pack_thread == 0U,
                             DispatchPackStart, GinDispatch, context,
                             wave_packets, a.epoch);

  for (std::uint64_t packet_in_wave = pack_warp;
       packet_in_wave < wave_packets; packet_in_wave += kRdmaPackWarps) {
    const auto position = wave_begin + packet_in_wave;
    const auto packet = a.dispatch_waves.packet_order[position];
    const auto descriptor = a.dispatch.packet_headers[packet];
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    const auto token = descriptor.local_token_id;

    auto *destination = reinterpret_cast<uint4 *>(send_payload) +
                        position * vector_count;
    const auto *source = reinterpret_cast<const uint4 *>(a.input) +
                         token * vector_count;
    for (std::uint64_t vector = lane; vector < vector_count; vector += 32ULL)
      destination[vector] = source[vector];

    for (std::uint64_t fanout_slot = lane;
         fanout_slot < a.dispatch.identity.top_k; fanout_slot += 32ULL) {
      if (fanout_begin + fanout_slot >= fanout_end)
        continue;
      const auto fanout = fanout_begin + fanout_slot;
      const auto target = a.dispatch.packet_fanout[fanout];
      send_fanout[position * a.dispatch.identity.top_k + fanout_slot] =
          target;
      source_compact_index[fanout] = target.return_compact_row;
      const auto route =
          token * a.dispatch.identity.top_k + target.topk_slot;
      source_route_index[target.return_compact_row] = route;
      // FULL Return lands directly in compact receive storage.  Its inverse
      // route map is generation-static, so publish it while Dispatch is already
      // touching this fanout record instead of rebuilding it on the Return
      // critical path after the payload marker arrives.
      direct_source_compact_index[route] = target.return_compact_row;
    }

    if (lane == 0U) {
      const auto wire_position = position - node_packet_begin;
      send_headers[position] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          a.rank,
          descriptor.proxy_rank,
          descriptor.channel,
          a.dispatch.identity.generation,
          0ULL,
          wire_position,
          fanout_end - fanout_begin,
          token,
          wire_position * a.dispatch.identity.top_k,
          0U,
          packet_in_wave == 0ULL
              ? static_cast<std::uint32_t>(wave_packets)
              : 0U};
    }
  }

  // Every writer establishes visibility to the GIN async proxy before the
  // elected producer publishes the descriptor.  This matches the old Relay
  // packer's conservative fence while the new subgroup protocol is proven.
  __threadfence_system();
  rdma_pack_group_sync();
  MEGAMOE_TIMELINE_RECORD_IF(
      a.timeline, pack_thread == 0U, DispatchPackEnd, GinDispatch, context,
      wave_packets * a.hidden * sizeof(transport::BFloat16), a.epoch);
  if (pack_thread == 0U) {
    if (wave_packets > UINT16_MAX) {
      persistent_v3::publish_error_once(control.generation, 201ULL);
    } else {
      persistent_v3::store_release<::cuda::thread_scope_device>(
          control.dispatch[context].outbound_waves_packed.value,
          packed_count);
    }
  }
  rdma_pack_group_sync();
}

// One permanent packing group owns the complete outbound wave interval for
// its GIN context.  The wire layout is immutable after Notify, so the scalar
// GIN owner needs only acquire a directory-ready epoch and a contiguous packed
// wave prefix.  No dynamic descriptor or terminal is passed through a ring.
__device__ __forceinline__ void rdma_pack_outbound_context(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context) {
  __shared__ std::uint64_t pack_node_wave_begin;
  __shared__ std::uint64_t pack_wave_begin;
  __shared__ std::uint64_t pack_wave_end;
  __shared__ std::uint64_t pack_node_packet_begin;
  __shared__ unsigned int pack_enabled;

  const auto pack_thread = threadIdx.x - kRdmaControlWarps * 32U;
  const auto remote_node = a.node == 0U ? 1U : 0U;
  if (pack_thread == 0U) {
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               control.generation.contexts[context]
                   .tx_local_complete.value) < a.epoch &&
           !cancelled(a, control))
      __nanosleep(64U);
    pack_enabled = cancelled(a, control) ? 0U : 1U;
    context_wave_span(a, context, remote_node, pack_node_wave_begin,
                      pack_wave_begin, pack_wave_end);
    pack_node_packet_begin =
        a.dispatch_waves.offsets[pack_node_wave_begin];
  }
  rdma_pack_group_sync();
  if (pack_enabled == 0U)
    return;

  if (pack_thread == 0U) {
    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *directories = reinterpret_cast<GinDispatchContextRecord *>(
        bytes + a.gin_send_dispatch_directory_offset);
    const auto remote_partition =
        remote_node < a.node ? remote_node : remote_node - 1U;
    const auto packet_begin =
        a.dispatch_waves.offsets[pack_wave_begin] - pack_node_packet_begin;
    const auto packet_end =
        a.dispatch_waves.offsets[pack_wave_end] - pack_node_packet_begin;
    const auto directory_index =
        remote_partition * persistent_v3::kRdmaCtaCount + context;
    const auto return_rows =
        a.dispatch.sizes.route_count - local_route_prefix(a);
    directories[directory_index] = GinDispatchContextRecord{
        a.dispatch.identity.generation,
        a.epoch,
        packet_begin,
        static_cast<std::uint32_t>(packet_end - packet_begin),
        static_cast<std::uint32_t>(pack_wave_end - pack_wave_begin),
        return_rows};
    __threadfence_system();
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].outbound_directory_ready.value,
        a.epoch);
  }
  rdma_pack_group_sync();

  for (auto wave = pack_wave_begin; wave < pack_wave_end; ++wave)
    pack_outbound_wave_from_rdma(a, control, context, pack_node_packet_begin,
                                 wave,
                                 wave - pack_wave_begin + 1ULL);

  if (pack_thread == 0U) {
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].outbound_pack_complete.value, a.epoch);
  }
  rdma_pack_group_sync();
}

__device__ __forceinline__ void relay_inbound_wave(
    const ForwardKernelArgs &a, const persistent_v3::PipelineTask &task,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion) {
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_receive_payload_offset);
  auto *fragment_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_fragment_index_offset);
  const auto source_partition = static_cast<std::uint32_t>(task.expert);
  const auto partition_packet_base =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank;
  const auto partition_fanout_base =
      partition_packet_base * a.dispatch.identity.top_k;
  const auto vector_count = a.hidden / 8ULL;
  constexpr std::uint32_t kWarps = persistent_v3::kWarpsPerCta;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto publish_batch_rows = min(
      static_cast<std::uint64_t>(a.forward_dispatch_publish_batch_rows),
      static_cast<std::uint64_t>(task.row_count));
  for (std::uint64_t batch_row_begin = 0ULL; batch_row_begin < task.row_count;
       batch_row_begin += publish_batch_rows) {
    const auto batch_row_end = min(batch_row_begin + publish_batch_rows,
                                   static_cast<std::uint64_t>(task.row_count));
    const auto batch_item_begin = batch_row_begin * a.dispatch.identity.top_k;
    const auto batch_item_end = batch_row_end * a.dispatch.identity.top_k;
    for (std::uint64_t item = batch_item_begin + warp;
         item < batch_item_end; item += kWarps) {
      const auto packet_in_wave = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto sequence = task.row_begin + packet_in_wave;
      const auto packet = partition_packet_base + sequence;
      std::uint32_t valid = 0U;
      std::uint32_t destination_local_rank = 0U;
      std::uint64_t destination_row = 0ULL;
      if (lane == 0U) {
        const auto header = receive_headers[packet];
        if (fanout_slot < header.route_index) {
          const auto packed =
              partition_fanout_base + header.dst_row + fanout_slot;
          const auto target = receive_fanout[packed];
          valid = 1U;
          destination_local_rank = target.destination_local_rank;
          destination_row = target.dst_row;
          const auto destination =
              a.node * a.local_world_size + destination_local_rank;
          auto *destination_headers = peer_at<LocalP2pRouteHeader>(
              a.peer_bases[destination], a.direct_headers_offset);
          const auto route = header.local_token_id * a.dispatch.identity.top_k +
                             target.topk_slot;
          destination_headers[destination_row] =
              LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                                  header.source_rank,
                                  destination,
                                  header.channel,
                                  header.generation,
                                  a.epoch,
                                  header.sequence,
                                  route,
                                  header.local_token_id,
                                  destination_row,
                                  target.expert_id,
                                  target.topk_slot};
        }
      }
      valid = __shfl_sync(0xffffffffU, valid, 0U);
      if (valid == 0U)
        continue;
      destination_local_rank =
          __shfl_sync(0xffffffffU, destination_local_rank, 0U);
      destination_row = __shfl_sync(0xffffffffU, destination_row, 0U);
      const auto destination =
          a.node * a.local_world_size + destination_local_rank;
      auto *destination_rows = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.canonical_rows_offset);
      auto *target = reinterpret_cast<uint4 *>(destination_rows) +
                     destination_row * vector_count;
      const auto *source = reinterpret_cast<const uint4 *>(receive_payload) +
                           packet * vector_count;
      for (std::uint64_t vector = lane; vector < vector_count; vector += 32ULL)
        target[vector] = source[vector];
    }

    __threadfence_system();
    __syncthreads();
    for (std::uint64_t item = batch_item_begin + threadIdx.x;
         item < batch_item_end; item += blockDim.x) {
      const auto packet_in_wave = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto packet =
          partition_packet_base + task.row_begin + packet_in_wave;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target =
          receive_fanout[partition_fanout_base + header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *arrived = peer_at<unsigned long long>(a.peer_bases[destination],
                                                  a.x_arrived_rows_offset);
      const auto rail = a.rank % a.local_world_size;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          arrived[target.reserved * a.local_world_size + rail]);
      ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
      // X readiness is the FC1 dependency; do not lengthen it with reverse-map
      // publication. FC2's descriptor is consumed only after the FULL mapping
      // helper acquires this release.
      const auto compact_row = target.return_compact_row;
      const auto chunk = compact_row / a.forward_return_chunk_rows;
      fragment_index[partition_fanout_base + compact_row] =
          header.dst_row + fanout_slot;
      auto *peer_map = peer_at<unsigned long long>(
          a.peer_bases[destination], a.forward_output_ready_epoch_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(peer_map[target.dst_row]);
      published.store(encode_return_row_mapping(
                          a.rank, chunk, partition_fanout_base + compact_row),
                      ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
      const auto record_index =
          source_partition * a.gin_context_count + task.context;
      atomicAdd(&context_completion[record_index].relay_fanout_rows, 1ULL);
      if (target.reserved == 0U)
        atomicAdd(&context_completion[record_index].relay_tile0_rows, 1ULL);
      atomicAdd(&completion->remote_rows_published, 1ULL);
      if (target.reserved < 64U)
        atomicOr(&completion->published_tile_mask, 1ULL << target.reserved);
#endif
    }
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    const auto record_index =
        source_partition * a.gin_context_count + task.context;
    auto &record = context_completion[record_index];
    record.relay_claimed += task.row_count;
    record.relay_completed += task.row_count;
    record.last_progress_globaltimer = device_globaltimer_ns();
    claim_completion_tasks(completion, transport_state::TaskClass::RelayWave,
                           task.row_count);
    complete_completion_tasks(
        completion, transport_state::TaskClass::RelayWave, task.row_count);
  }
  __syncthreads();
}

__device__ __forceinline__ void relay_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t relay_ordinal,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler, unsigned char *storage) {
  const auto context = relay_ordinal % persistent_v3::kRdmaCtaCount;
  const auto helper_lane = relay_ordinal / persistent_v3::kRdmaCtaCount;
  const auto helpers_per_context =
      a.forward_relay_cta_count / persistent_v3::kRdmaCtaCount;
  const bool owns_context_transport = helper_lane == 0U;
  const auto remote_node = a.node == 0U ? 1U : 0U;
  const auto source_partition =
      remote_node < a.node ? remote_node : remote_node - 1U;
  __shared__ std::uint64_t local_routes;
  __shared__ GinDispatchContextRecord inbound_directory;
  __shared__ std::uint64_t relay_packet_cursor;
  __shared__ std::uint64_t relay_waves;
  __shared__ persistent_v3::PipelineTask inbound_task;
  __shared__ forward_return_detail::DirectLocalReturnState local_return_state;
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ forward_return_detail::DirectLocalReturnBarrier
      local_return_tma_full[persistent_v3::kWarpsPerCta];
  __shared__ forward_return_detail::DirectFullCompletionState
      completion_state;
  __shared__ unsigned int inbound_directory_seen;
  __shared__ unsigned int inbound_ready;
  __shared__ unsigned int local_return_done;
  __shared__ unsigned int completion_done;
  if (threadIdx.x == 0U) {
    local_routes = local_route_prefix(a);
    inbound_directory = GinDispatchContextRecord{};
    relay_packet_cursor = 0ULL;
    relay_waves = 0ULL;
    inbound_directory_seen = 0U;
    inbound_ready = 0U;
    local_return_state = forward_return_detail::DirectLocalReturnState{};
    local_return_state.cursor = helper_lane;
    local_return_state.stride = helpers_per_context;
    completion_state = forward_return_detail::DirectFullCompletionState{};
    local_return_done = 0U;
    completion_done = owns_context_transport ? 0U : 1U;
    for (std::uint32_t warp = 0U; warp < persistent_v3::kWarpsPerCta; ++warp)
      init(&local_return_tma_full[warp], 1U);
    ::cuda::device::experimental::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *receive_directories = reinterpret_cast<GinDispatchContextRecord *>(
      bytes + a.gin_receive_dispatch_directory_offset);
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  const auto receive_directory_index =
      source_partition * persistent_v3::kRdmaCtaCount + context;
  const auto partition_packet_base =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank;

  auto local_begin = static_cast<std::uint64_t>(relay_ordinal) *
                     kLocalRouteBatch;
  const auto local_stride =
      static_cast<std::uint64_t>(a.forward_relay_cta_count) * kLocalRouteBatch;
  bool inbound_terminal = !owns_context_transport;
  bool dispatch_terminal_published = false;
  bool completion_timeline_started = false;
  bool completion_terminal_published = false;
  MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay, context,
                          0ULL, a.epoch);
  while ((!dispatch_terminal_published || local_return_done == 0U ||
          completion_done == 0U) &&
         !cancelled(a, control)) {
    bool progressed = false;

    if (!inbound_terminal) {
      if (threadIdx.x == 0U) {
        inbound_ready = 0U;
        if (inbound_directory_seen == 0U &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context]
                    .inbound_directory_ready.value) >= a.epoch) {
          // The GIN owner crossed the async-proxy boundary before releasing
          // this epoch.  Keep the fence here as well so this independent SM's
          // generic loads cannot observe an older receive-buffer image.
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto directory =
              receive_directories[receive_directory_index];
          if (directory.generation != a.dispatch.identity.generation ||
              directory.epoch != a.epoch) {
            persistent_v3::publish_error_once(control.generation, 203ULL);
          } else {
            inbound_directory = directory;
            relay_packet_cursor = directory.packet_begin;
            inbound_directory_seen = 1U;
          }
        }

        if (inbound_directory_seen != 0U) {
          const auto ready_waves =
              persistent_v3::load_acquire<::cuda::thread_scope_device>(
                  control.dispatch[context].inbound_waves_ready.value);
          if (relay_waves < ready_waves) {
            if (relay_waves >= inbound_directory.wave_count) {
              persistent_v3::publish_error_once(control.generation, 204ULL);
            } else {
              const auto packet =
                  partition_packet_base + relay_packet_cursor;
              asm volatile("fence.proxy.async.global;" : : : "memory");
              const auto wave_packets = receive_headers[packet].topk_slot;
              if (wave_packets == 0U || wave_packets > UINT16_MAX ||
                  relay_packet_cursor + wave_packets >
                      inbound_directory.packet_begin +
                          inbound_directory.packet_count) {
                persistent_v3::publish_error_once(control.generation, 205ULL);
              } else {
                inbound_task = persistent_v3::PipelineTask{};
                inbound_task.epoch = a.epoch;
                inbound_task.index = static_cast<std::uint32_t>(packet);
                inbound_task.expert =
                    static_cast<std::uint16_t>(source_partition);
                inbound_task.kind = static_cast<std::uint8_t>(
                    persistent_v3::TaskKind::DispatchRelay);
                inbound_task.context = static_cast<std::uint8_t>(context);
                inbound_task.row_begin =
                    static_cast<std::uint32_t>(relay_packet_cursor);
                inbound_task.row_count =
                    static_cast<std::uint16_t>(wave_packets);
                inbound_ready = 1U;
              }
            }
          }
        }
      }
      __syncthreads();
      if (inbound_ready != 0U) {
        progressed = true;
        relay_inbound_wave(a, inbound_task, completion,
                           context_completion);
        if (threadIdx.x == 0U) {
          relay_packet_cursor += inbound_task.row_count;
          ++relay_waves;
        }
      }
      __syncthreads();
      if (inbound_directory_seen != 0U &&
          relay_waves == inbound_directory.wave_count)
        inbound_terminal = true;
    }

#if defined(MEGAMOE_FORWARD_TRANSPORT_TU)
    // Mirror the compute scheduler's FC2-first policy at the node-local
    // transport boundary.  Once an FC2 M-block is available, retire one
    // bounded Return batch before issuing another Dispatch batch.  Skipping
    // Dispatch only in the iteration that actually consumed a descriptor
    // prevents the TMA streams from competing for the same Relay CTA and
    // NVLink path while keeping the scheduler work-conserving.
    bool local_return_progressed = false;
    if (local_return_done == 0U) {
      local_return_progressed =
          forward_return_detail::progress_direct_local_return_once(
              a, control, context, scheduler, local_return_state,
              reinterpret_cast<std::byte *>(storage),
              local_return_tma_full);
      progressed = local_return_progressed || progressed;
      if (threadIdx.x == 0U &&
          forward_return_detail::direct_local_return_terminal(
              a, scheduler, local_return_state))
        local_return_done = 1U;
      __syncthreads();
    }

    if (!local_return_progressed && local_begin < local_routes) {
      const auto local_end = min(
          local_begin + static_cast<std::uint64_t>(kLocalRouteBatch),
          local_routes);
      publish_local_batch(a, local_begin, local_end, completion,
                          reinterpret_cast<std::byte *>(storage),
                          local_return_tma_full);
      if (local_begin ==
          static_cast<std::uint64_t>(relay_ordinal) * kLocalRouteBatch)
        MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible,
                                GinDispatch, context, local_begin, a.epoch);
      local_begin += local_stride;
      progressed = true;
    }
#else
    if (local_begin < local_routes) {
      const auto local_end = min(
          local_begin + static_cast<std::uint64_t>(kLocalRouteBatch),
          local_routes);
      publish_local_batch(a, local_begin, local_end, completion);
      if (local_begin ==
          static_cast<std::uint64_t>(relay_ordinal) * kLocalRouteBatch)
        MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible,
                                GinDispatch, context, local_begin, a.epoch);
      local_begin += local_stride;
      progressed = true;
    }
#endif

    const bool dispatch_terminal =
        inbound_terminal && local_begin >= local_routes;
    if (dispatch_terminal && !dispatch_terminal_published) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, context,
                              local_routes, a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay, context,
                              local_routes, a.epoch);
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
      __syncthreads();
      dispatch_terminal_published = true;
      progressed = true;
    }

#if !defined(MEGAMOE_FORWARD_TRANSPORT_TU)
    // One bounded unit from each Return plane keeps Dispatch latency favored
    // without delaying already-computed FC2 rows until the complete Dispatch
    // or FC2 task spaces drain.
    if (local_return_done == 0U) {
      progressed = forward_return_detail::progress_direct_local_return_once(
                       a, control, context, scheduler, local_return_state,
                       reinterpret_cast<std::byte *>(storage),
                       local_return_tma_full) ||
                   progressed;
      if (threadIdx.x == 0U &&
          forward_return_detail::direct_local_return_terminal(
              a, scheduler, local_return_state))
        local_return_done = 1U;
      __syncthreads();
    }
#endif

    if (owns_context_transport && completion_done == 0U) {
      progressed =
          forward_return_detail::progress_direct_full_completion_once(
              a, control, context, scheduler, completion_state) ||
          progressed;
      if (!completion_timeline_started && completion_state.initialized != 0U) {
        MEGAMOE_TIMELINE_RECORD(
            a.timeline, SourceRelayStart, SourceRelay, context,
            completion_state.expected, a.epoch);
        completion_timeline_started = true;
      }
      if (forward_return_detail::direct_full_completion_terminal(
              completion_state)) {
        if (!completion_terminal_published) {
          MEGAMOE_TIMELINE_RECORD(a.timeline, SourceRelayEnd, SourceRelay,
                                  context, completion_state.completed, a.epoch);
          if (threadIdx.x == 0U) {
            persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                control.return_receive_roles_complete.value, 1ULL);
            completion_done = 1U;
          }
          __syncthreads();
          completion_terminal_published = true;
        }
      }
    }

    if (!progressed && threadIdx.x == 0U)
      __nanosleep(64U);
    __syncthreads();
  }
}

__device__ __forceinline__ void issue_outbound_task(
    const ForwardKernelArgs &a, const ncclGin &gin, ncclTeam rail,
    const persistent_v3::PipelineTask &task,
    unsigned long long &issued_completions) {
  const auto remote_node = static_cast<std::uint32_t>(task.expert);
  if ((task.flags & kDispatchDirectoryTask) != 0U) {
    gin.put(rail, static_cast<int>(remote_node), a.gin_window,
            a.gin_receive_dispatch_directory_offset +
                static_cast<std::size_t>(task.row_begin) *
                    sizeof(GinDispatchContextRecord),
            a.gin_window,
            a.gin_send_dispatch_directory_offset +
                static_cast<std::size_t>(task.index) *
                    sizeof(GinDispatchContextRecord),
            sizeof(GinDispatchContextRecord), ncclGin_SignalInc{0},
            ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_device);
    ++issued_completions;
    return;
  }

  const auto source_partition =
      a.node < remote_node ? a.node : a.node - 1U;
  const auto receive_packet =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank +
      task.row_begin;
  const auto packets = static_cast<std::uint64_t>(task.row_count);
  MEGAMOE_TIMELINE_RECORD(
      a.timeline, GinPutIssue, GinDispatch, task.context,
      packets * a.hidden * sizeof(transport::BFloat16), a.epoch);
  gin.put(rail, static_cast<int>(remote_node), a.gin_window,
          a.gin_receive_header_offset +
              receive_packet * sizeof(LocalP2pRouteHeader),
          a.gin_window,
          a.gin_send_header_offset +
              static_cast<std::size_t>(task.index) *
                  sizeof(LocalP2pRouteHeader),
          packets * sizeof(LocalP2pRouteHeader));
  gin.put(rail, static_cast<int>(remote_node), a.gin_window,
          a.gin_receive_fanout_offset +
              receive_packet * a.dispatch.identity.top_k *
                  sizeof(DeviceFanoutRecord),
          a.gin_window,
          a.gin_send_fanout_offset +
              static_cast<std::size_t>(task.index) *
                  a.dispatch.identity.top_k * sizeof(DeviceFanoutRecord),
          packets * a.dispatch.identity.top_k *
              sizeof(DeviceFanoutRecord));
  gin.put(rail, static_cast<int>(remote_node), a.gin_window,
          a.gin_receive_payload_offset +
              receive_packet * a.hidden * sizeof(transport::BFloat16),
          a.gin_window,
          a.gin_send_payload_offset +
              static_cast<std::size_t>(task.index) * a.hidden *
                  sizeof(transport::BFloat16),
          packets * a.hidden * sizeof(transport::BFloat16), ncclGin_None{},
          ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
          ::cuda::thread_scope_block, ::cuda::thread_scope_device);
  ++issued_completions;

  // This ordered 8-byte put is both the per-wave identity and its completion
  // action.  Signal N can therefore be consumed incrementally without
  // assuming that independent GIN WQEs complete in issue order.
  gin.putValue<std::uint64_t>(
      rail, static_cast<int>(remote_node), a.gin_window,
      a.gin_receive_header_offset +
          receive_packet * sizeof(LocalP2pRouteHeader) +
          offsetof(LocalP2pRouteHeader, source_epoch),
      a.epoch, ncclGin_SignalInc{0}, ncclCoopThread{}, ncclGin_None{},
      ::cuda::thread_scope_block, ::cuda::thread_scope_device);

  auto *status =
      peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_dispatch_put_count, 4ULL);
  atomicAdd(&status->gin_signal_count, 1ULL);
  atomicAdd(&status->deduplicated_cross_node_payload_bytes,
            packets * a.hidden * sizeof(transport::BFloat16));
  atomicAdd(&status->actual_dispatch_peer_bytes,
            packets * a.hidden * sizeof(transport::BFloat16));
  atomicAdd(&status->gin_metadata_bytes,
            packets * (sizeof(LocalP2pRouteHeader) +
                       a.dispatch.identity.top_k *
                           sizeof(DeviceFanoutRecord)));
}

// Ring-free Dispatch arrival publication.  The GIN owner remains responsible
// for acquiring signal 0 and crossing the async-proxy boundary; only after
// that fence does it release a contiguous wave prefix to the Relay CTA.  The
// Relay derives packet extents from the immutable directory and the first
// header in each wave, so no per-wave descriptor needs to be copied.
__device__ __forceinline__ bool publish_one_inbound_wave_sequence(
    const ForwardKernelArgs &a, const ncclGin &gin,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context, std::uint32_t source_partition,
    bool &directory_seen, GinDispatchContextRecord &directory,
    unsigned long long &packet_cursor, unsigned long long &waves_published,
    transport_state::ContextCompletionRecord *context_completion) {
  const auto signal_value =
      gin.readSignal(0, 64, ::cuda::memory_order_acquire);
  const auto observed = static_cast<unsigned long long>(directory_seen) +
                        waves_published;
  if (signal_value <= observed)
    return false;

  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *directories = reinterpret_cast<GinDispatchContextRecord *>(
      bytes + a.gin_receive_dispatch_directory_offset);
  auto *headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  if (!directory_seen) {
    asm volatile("fence.proxy.async.global;" : : : "memory");
    directory = directories[
        source_partition * persistent_v3::kRdmaCtaCount + context];
    if (directory.generation != a.dispatch.identity.generation ||
        directory.epoch != a.epoch)
      return false;
    packet_cursor = directory.packet_begin;
    directory_seen = true;
    auto &record = context_completion[
        source_partition * a.gin_context_count + context];
    record.directory_ready = 1ULL;
    record.relay_expected = directory.packet_count;
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].inbound_directory_ready.value,
        a.epoch);
    return true;
  }
  if (waves_published >= directory.wave_count)
    return false;

  const auto packet =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank +
      packet_cursor;
  ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> marker(
      headers[packet].source_epoch);
  if (marker.load(::cuda::memory_order_acquire) != a.epoch)
    return false;
  asm volatile("fence.proxy.async.global;" : : : "memory");
  const auto wave_packets = headers[packet].topk_slot;
  if (wave_packets == 0U || wave_packets > directory.packet_count ||
      packet_cursor + wave_packets >
          directory.packet_begin + directory.packet_count ||
      wave_packets > UINT16_MAX) {
    persistent_v3::publish_error_once(control.generation, 202ULL);
    return true;
  }

  packet_cursor += wave_packets;
  ++waves_published;
  persistent_v3::store_release<::cuda::thread_scope_device>(
      control.dispatch[context].inbound_waves_ready.value,
      waves_published);
  return true;
}

// Each fixed RDMA CTA contributes one otherwise-idle scalar lane to readiness
// publication. Eight lanes total scan the destination-local Forward M-blocks
// without a 116-compute-CTA system-atomic polling storm. A ready M-block is
// expanded once into its FC1 N-tile tasks in a device-local queue.
__device__ __forceinline__ void publish_ready_fc1_locality_groups(
    const ForwardKernelArgs &a, std::uint32_t context,
    unsigned long long *scheduler) {
  constexpr std::uint32_t kStartupMblocks = 2U;
  const auto locality_group = a.forward_fc1_locality_group;
  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *published_mblocks = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
#if MEGAMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif

  while (!forward_cancelled(a)) {
    bool pending = false;
    bool progressed = false;
    for (std::uint64_t mblock = context; mblock < forward_mblocks;
         mblock += persistent_v3::kRdmaCtaCount) {
      if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
        continue;
      pending = true;
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);

      std::uint32_t group_mblocks = 1U;
      if (local_mblock >= kStartupMblocks) {
        const auto group_offset =
            (local_mblock - kStartupMblocks) % locality_group;
        if (group_offset != 0U) {
          // Body mappings are expert-major and therefore contiguous. The
          // leader marks every follower before marking itself; observing the
          // leader published is consequently sufficient to retire this lane.
          const auto leader = mblock - group_offset;
          const auto leader_mapping = claims[leader];
          const bool valid_leader =
              static_cast<std::uint32_t>(leader_mapping >> 32U) == expert &&
              static_cast<std::uint32_t>(leader_mapping) + group_offset ==
                  local_mblock;
          if (valid_leader &&
              atomicAdd(&published_mblocks[leader], 0ULL) == 0ULL)
            continue;
          if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
            continue;
          // A malformed/tail mapping must make progress rather than leaving a
          // follower permanently owned by a leader that cannot include it.
          group_mblocks = 1U;
        } else {
          const auto expert_mblocks = static_cast<std::uint32_t>(
              forward_expert_mblock_count(a, expert));
          group_mblocks =
              min(locality_group, expert_mblocks - local_mblock);
          bool group_ready = true;
          for (std::uint32_t member = 0; member < group_mblocks; ++member) {
            const auto sibling = mblock + member;
            if (sibling >= forward_mblocks) {
              group_ready = false;
              break;
            }
            const auto sibling_mapping = claims[sibling];
            if (static_cast<std::uint32_t>(sibling_mapping >> 32U) != expert ||
                static_cast<std::uint32_t>(sibling_mapping) !=
                    local_mblock + member ||
                atomicAdd(&published_mblocks[sibling], 0ULL) != 0ULL) {
              group_ready = false;
              break;
            }
            const auto sibling_row =
                forward_mblock_local_row(a, expert, local_mblock + member);
            const auto sibling_rows =
                forward_mblock_valid_rows(a, expert, local_mblock + member);
            if (!forward_mblock_rows_ready<false>(
                    a, expert, sibling_row, sibling_rows)) {
              group_ready = false;
              break;
            }
          }
          if (!group_ready)
            continue;
        }
      } else {
        const auto local_row =
            forward_mblock_local_row(a, expert, local_mblock);
        const auto valid_rows =
            forward_mblock_valid_rows(a, expert, local_mblock);
        if (!forward_mblock_rows_ready<false>(a, expert, local_row,
                                               valid_rows))
          continue;
      }

      // Followers become visible before their leader. A follower lane that
      // acquires the leader flag can therefore never race a partial group.
      for (std::uint32_t member = group_mblocks; member != 0U; --member)
        atomicExch(&published_mblocks[mblock + member - 1U], 1ULL);
      const auto queue_base =
          atomicAdd(&scheduler[12ULL + 2ULL * a.world],
                    static_cast<unsigned long long>(group_mblocks) *
                        fc1_ntiles);
      for (std::uint32_t ntile = 0; ntile < fc1_ntiles; ++ntile) {
        for (std::uint32_t member = 0; member < group_mblocks; ++member) {
          const auto queue_slot =
              queue_base + static_cast<unsigned long long>(ntile) *
                               group_mblocks + member;
          const auto task = (mblock + member) * fc1_ntiles + ntile;
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              entry(fc1_ready_queue[queue_slot]);
          entry.store(task + 1ULL, ::cuda::memory_order_release);
        }
      }
      progressed = true;
#if MEGAMOE_DEBUG_VALIDATE
      atomicMax(&completion->bulk_pack_complete,
                queue_base +
                    static_cast<unsigned long long>(group_mblocks) *
                        fc1_ntiles);
#endif
    }
    if (!pending)
      break;
    if (!progressed)
      __nanosleep(256U);
  }
#if MEGAMOE_DEBUG_VALIDATE
  atomicOr(&completion->bulk_contexts_issuing, 1ULL << context);
#endif
}

__device__ __forceinline__ void publish_ready_fc1_tasks(
    const ForwardKernelArgs &a, std::uint32_t context,
    unsigned long long *scheduler) {
  if (a.forward_fc1_locality_group > 1U) {
    publish_ready_fc1_locality_groups(a, context, scheduler);
    return;
  }
  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *published_mblocks = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
  const auto head_mblocks = a.forward_m64_head_wavefront
                                ? *forward_mixed_scheduler_word(
                                      a, scheduler, kForwardHeadMblocks)
                                : forward_mblocks;
#if MEGAMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif
  auto remaining =
      context < forward_mblocks
          ? (forward_mblocks - 1ULL - context) /
                    persistent_v3::kRdmaCtaCount +
                1ULL
          : 0ULL;
  while (remaining != 0ULL && !forward_cancelled(a)) {
    bool progressed = false;
    for (std::uint64_t mblock = context; mblock < forward_mblocks;
         mblock += persistent_v3::kRdmaCtaCount) {
      if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
        continue;
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto local_row =
          forward_mblock_local_row(a, expert, local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      if (!forward_mblock_rows_ready<false>(a, expert, local_row, valid_rows))
        continue;
      atomicExch(&published_mblocks[mblock], 1ULL);
      const bool body_task =
          a.forward_m64_head_wavefront && mblock >= head_mblocks;
      auto *publish_tail =
          body_task
              ? forward_mixed_scheduler_word(a, scheduler,
                                             kForwardBodyFc1Publish)
              : &scheduler[12ULL + 2ULL * a.world];
      const auto partition_base =
          body_task ? head_mblocks * fc1_ntiles : 0ULL;
      const auto queue_base =
          partition_base + atomicAdd(
                               publish_tail,
                               static_cast<unsigned long long>(fc1_ntiles));
      for (std::uint32_t ntile = 0; ntile < fc1_ntiles; ++ntile) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            entry(fc1_ready_queue[queue_base + ntile]);
        const auto task = mblock * fc1_ntiles + ntile;
        entry.store(task + 1ULL, ::cuda::memory_order_release);
      }
      --remaining;
      progressed = true;
#if MEGAMOE_DEBUG_VALIDATE
      completion->bulk_pack_complete = queue_base + fc1_ntiles;
#endif
    }
    if (!progressed)
      __nanosleep(256U);
  }
#if MEGAMOE_DEBUG_VALIDATE
  atomicOr(&completion->bulk_contexts_issuing, 1ULL << context);
#endif
}

__device__ __forceinline__ void rdma_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler) {
  if (threadIdx.x == 0U) {
    const ncclGin gin(a.gin_dev_comm, context);
    const ncclGin return_gin(a.gin_dev_comm,
                             context + a.gin_context_count);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    const auto world = ncclTeamWorld(a.gin_dev_comm);
    // CTA-wide synchronization in the setup helper is unnecessary here: only
    // the permanent GIN thread touches this context or its queue consumer.
    const auto remote_node = a.node == 0U ? 1U : 0U;
    const auto remote_partition =
        remote_node < a.node ? remote_node : remote_node - 1U;
    const auto source_partition = remote_partition;

    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *send_directory = reinterpret_cast<GinDispatchContextRecord *>(
        bytes + a.gin_send_dispatch_directory_offset);
    const auto send_directory_index =
        remote_partition * persistent_v3::kRdmaCtaCount + context;

    // Inline the one-thread handshake so the remaining 383 threads never
    // acquire GIN state or register pressure.
    const auto source_directory_index =
        (a.node < remote_node ? a.node : a.node - 1U) *
            persistent_v3::kRdmaCtaCount +
        context;
    send_directory[send_directory_index] = GinDispatchContextRecord{
        a.dispatch.identity.generation, a.epoch, 0ULL, 0U, 0U};
    __threadfence_system();
    gin.put(rail, static_cast<int>(remote_node), a.gin_window,
            a.gin_receive_dispatch_directory_offset +
                source_directory_index * sizeof(GinDispatchContextRecord),
            a.gin_window,
            a.gin_send_dispatch_directory_offset +
                send_directory_index * sizeof(GinDispatchContextRecord),
            sizeof(GinDispatchContextRecord), ncclGin_SignalInc{2},
            ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_device);
    gin.waitCounter(ncclCoopThread{}, 0, 1ULL);
    gin.resetCounter(0);
    gin.waitSignal(ncclCoopThread{}, 2, 1ULL);
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.generation.contexts[context].tx_local_complete.value,
        a.epoch);

    std::uint64_t outbound_node_wave_begin = 0ULL;
    std::uint64_t outbound_wave_begin = 0ULL;
    std::uint64_t outbound_wave_end = 0ULL;
    context_wave_span(a, context, remote_node, outbound_node_wave_begin,
                      outbound_wave_begin, outbound_wave_end);
    const auto outbound_node_packet_begin =
        a.dispatch_waves.offsets[outbound_node_wave_begin];
    auto outbound_wave = outbound_wave_begin;
    unsigned long long issued_completions = 0ULL;
    bool outbound_directory_issued = false;
    bool outbound_terminal = false;
    bool inbound_terminal = false;
    bool directory_seen = false;
    GinDispatchContextRecord inbound_directory{};
    unsigned long long packet_cursor = 0ULL;
    unsigned long long waves_published = 0ULL;
    forward_return_detail::DirectFullRdmaState direct_full_state{};
    while ((!outbound_terminal || !inbound_terminal) &&
           !cancelled(a, control)) {
      bool progressed = false;
      if (!outbound_terminal) {
        if (!outbound_directory_issued &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context]
                    .outbound_directory_ready.value) >= a.epoch) {
          persistent_v3::PipelineTask directory_task{};
          directory_task.epoch = a.epoch;
          directory_task.index =
              static_cast<std::uint32_t>(send_directory_index);
          directory_task.expert = static_cast<std::uint16_t>(remote_node);
          directory_task.kind = static_cast<std::uint8_t>(
              persistent_v3::TaskKind::DispatchRelay);
          directory_task.context = static_cast<std::uint8_t>(context);
          directory_task.row_begin =
              static_cast<std::uint32_t>(source_directory_index);
          directory_task.flags = kDispatchDirectoryTask;
          issue_outbound_task(a, gin, rail, directory_task,
                              issued_completions);
          outbound_directory_issued = true;
          progressed = true;
        }

        if (outbound_directory_issued &&
            outbound_wave < outbound_wave_end) {
          const auto packed_prefix =
              persistent_v3::load_acquire<::cuda::thread_scope_device>(
                  control.dispatch[context]
                      .outbound_waves_packed.value);
          const auto wave_ordinal = outbound_wave - outbound_wave_begin;
          if (wave_ordinal < packed_prefix) {
            const auto wave_begin = a.dispatch_waves.offsets[outbound_wave];
            const auto wave_end =
                a.dispatch_waves.offsets[outbound_wave + 1ULL];
            const auto wave_packets = wave_end - wave_begin;
            if (wave_packets == 0ULL || wave_packets > UINT16_MAX) {
              persistent_v3::publish_error_once(control.generation, 206ULL);
            } else {
              persistent_v3::PipelineTask wave_task{};
              wave_task.epoch = a.epoch;
              wave_task.index = static_cast<std::uint32_t>(wave_begin);
              wave_task.expert = static_cast<std::uint16_t>(remote_node);
              wave_task.kind = static_cast<std::uint8_t>(
                  persistent_v3::TaskKind::DispatchRelay);
              wave_task.context = static_cast<std::uint8_t>(context);
              wave_task.row_begin = static_cast<std::uint32_t>(
                  wave_begin - outbound_node_packet_begin);
              wave_task.row_count =
                  static_cast<std::uint16_t>(wave_packets);
              issue_outbound_task(a, gin, rail, wave_task,
                                  issued_completions);
              ++outbound_wave;
              progressed = true;
            }
          }
        }

        if (outbound_directory_issued &&
            outbound_wave == outbound_wave_end &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context]
                    .outbound_pack_complete.value) >= a.epoch) {
          outbound_terminal = true;
          progressed = true;
        }
      }

      if (!inbound_terminal) {
        progressed = publish_one_inbound_wave_sequence(
                         a, gin, control, context, source_partition,
                         directory_seen, inbound_directory, packet_cursor,
                         waves_published, context_completion) ||
                     progressed;
        if (directory_seen &&
            waves_published == inbound_directory.wave_count) {
          inbound_terminal = true;
          atomicAdd(&scheduler[5ULL + 2ULL * a.world], 1ULL);
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, RemoteSignalObserved, GinDispatch, context,
              waves_published, a.epoch);
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, XFirstRemoteSignal, GinDispatch, context,
              waves_published, a.epoch);
          progressed = true;
        }
      }

      progressed =
          forward_return_detail::progress_direct_full_rdma_once(
              a, control, return_gin, world, context, completion,
              context_completion, scheduler, direct_full_state) ||
          progressed;

      if (!progressed)
        __nanosleep(64U);
    }

    if (issued_completions != 0ULL) {
      while (gin.readCounter(0, 56, ::cuda::memory_order_acquire) <
                 issued_completions &&
             !cancelled(a, control)) {
        if (!forward_return_detail::progress_direct_full_rdma_once(
                a, control, return_gin, world, context, completion,
                context_completion, scheduler, direct_full_state))
          __nanosleep(64U);
      }
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchLocalCompletion,
                            GinDispatch, context, issued_completions,
                            a.epoch);

    const auto directory = send_directory[send_directory_index];
    auto &record = context_completion[
        remote_partition * a.gin_context_count + context];
    record.directory_ready = 1ULL;
    record.dispatch_expected = directory.wave_count;
    record.dispatch_claimed = directory.wave_count;
    record.dispatch_completed = directory.wave_count;
    record.local_completion = 1ULL + directory.wave_count;
    record.credit = 1ULL;
    record.last_cursor = directory.packet_begin + directory.packet_count;
    record.last_progress_globaltimer = device_globaltimer_ns();
    atomicAdd(&completion->directory_ready, 1ULL);
    atomicAdd(&completion->dispatch_local_completion_completed,
              1ULL + directory.wave_count);
    if (directory.wave_count != 0U) {
      claim_completion_tasks(completion,
                             transport_state::TaskClass::DispatchWave,
                             directory.wave_count);
      complete_completion_tasks(completion,
                                transport_state::TaskClass::DispatchWave,
                                directory.wave_count);
    }
    // The same permanent scalar owner transitions directly from same-rail
    // Dispatch into its independent FULL Return QP. Lane 32 can continue
    // publishing FC1 work without participating in either network loop.
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.rdma_dispatch_roles_complete.value, 1ULL);
    forward_return_detail::direct_full_rdma_thread(
        a, control, context, completion, context_completion, scheduler,
        return_gin, world, direct_full_state);
  } else if (threadIdx.x == 32U) {
    publish_ready_fc1_tasks(a, context, scheduler);
  } else if (threadIdx.x >= kRdmaControlWarps * 32U) {
    rdma_pack_outbound_context(a, control, context);
  }
  if (threadIdx.x >= kRdmaControlWarps * 32U) {
    // The ten former packing warps now prefetch reverse-map readiness while
    // lane 0 retains exclusive ownership of this context's Return QP.
    const auto helper_warp =
        threadIdx.x / 32U - kRdmaControlWarps;
    forward_return_detail::direct_full_mapping_helper(
        a, control, context, scheduler, helper_warp, kRdmaPackWarps);
  }
  __syncthreads();
}

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
