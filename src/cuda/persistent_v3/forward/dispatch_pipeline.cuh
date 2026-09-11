#pragma once

// Included by activation_transport.cu after the private Forward launch
// contract, GIN wire records, completion protocol, and peer helpers are
// visible.  This file contains only the RDMA/Relay Dispatch state
// machine; WGMMA lives in compute_kernel.cuh and cannot inherit these live
// ranges.

#if __has_include(<nccl_device.h>)

namespace forward_dispatch_detail {

inline constexpr std::uint32_t kLocalRouteBatch =
    forward_tuning::kLocalRouteBatchRows;
inline constexpr std::uint16_t kDispatchDirectoryTask = 1U;
// Warp 0 owns the GIN context and warp 1 contributes the scalar FC1-ready
// publisher.  The remaining ten complete warps used to sit idle while the
// paired Relay CTA both packed outbound packets and scattered inbound ones.
// Keep the two control warps out of every packing barrier so GIN progress and
// FC1 publication remain independent from the bulk HBM copy.
inline constexpr std::uint32_t kRdmaControlWarps =
    forward_tuning::kRdmaControlWarps;
inline constexpr std::uint32_t kRdmaPackWarps =
    persistent_v3::kWarpsPerCta - kRdmaControlWarps;
inline constexpr std::uint32_t kRdmaPackThreads = kRdmaPackWarps * 32U;

static_assert(kRdmaPackWarps > 0U);

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

__device__ __forceinline__ bool context_wave_span(
    const ForwardKernelArgs &a, std::uint32_t context,
    std::uint32_t remote_node, std::uint64_t &node_wave_begin,
    std::uint64_t &context_wave_begin, std::uint64_t &context_wave_end) {
  node_wave_begin = 0ULL;
  while (node_wave_begin < dispatch_wave_count(a.dispatch, a.dispatch_waves) &&
         a.dispatch_waves.destination_node[node_wave_begin] < remote_node)
    ++node_wave_begin;
  auto node_wave_end = node_wave_begin;
  while (node_wave_end < dispatch_wave_count(a.dispatch, a.dispatch_waves) &&
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
#if WEFTMOE_DEBUG_VALIDATE
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
#if WEFTMOE_DEBUG_VALIDATE
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

#endif // WEFTMOE_FORWARD_TRANSPORT_TU

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
  auto *send_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
      bytes + a.gin_send_fanout_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_send_payload_offset);
  auto *source_route_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_row_to_source_route_offset);
  auto *direct_source_compact_index = peer_at<unsigned long long>(
      a.local_base, a.forward_source_route_to_return_row_offset);
  const auto wave_begin = a.dispatch_waves.offsets[wave];
  const auto wave_end = a.dispatch_waves.offsets[wave + 1ULL];
  const auto wave_packets = wave_end - wave_begin;
  const auto vector_count = a.hidden / 8ULL;

  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, pack_thread == 0U, DispatchPackStart,
                             GinDispatch, context, wave_packets,
                             execution_epoch(a));

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
      const auto route =
          token * a.dispatch.identity.top_k + target.topk_slot;
      send_fanout[position * a.dispatch.identity.top_k + fanout_slot] =
          ForwardDispatchFanoutRecord{
              target.dst_row,
              target.return_compact_row,
              target.topk_slot,
              static_cast<std::uint16_t>(target.destination_local_rank),
              static_cast<std::uint16_t>(target.expert_id),
              target.reserved,
              a.gates[route]};
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
  WEFTMOE_TIMELINE_RECORD_IF(
      a.timeline, pack_thread == 0U, DispatchPackEnd, GinDispatch, context,
      wave_packets * a.hidden * sizeof(transport::BFloat16),
      execution_epoch(a));
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
               control.generation.contexts[context].tx_local_complete.value) <
               execution_epoch(a) &&
           !cancelled(a, control))
      __nanosleep(forward_tuning::kTransportIdleNanosleep);
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
        dispatch_sizes(a.dispatch).route_count - local_route_prefix(a);
    directories[directory_index] = GinDispatchContextRecord{
        a.dispatch.identity.generation,
        execution_epoch(a),
        packet_begin,
        static_cast<std::uint32_t>(packet_end - packet_begin),
        static_cast<std::uint32_t>(pack_wave_end - pack_wave_begin),
        return_rows};
    __threadfence_system();
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].outbound_directory_ready.value,
        execution_epoch(a));
  }
  rdma_pack_group_sync();

  for (auto wave = pack_wave_begin; wave < pack_wave_end; ++wave)
    pack_outbound_wave_from_rdma(a, control, context, pack_node_packet_begin,
                                 wave,
                                 wave - pack_wave_begin + 1ULL);

  if (pack_thread == 0U) {
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].outbound_pack_complete.value,
        execution_epoch(a));
  }
  rdma_pack_group_sync();
}

__device__ __forceinline__ void relay_inbound_wave(
    const ForwardKernelArgs &a, const persistent_v3::PipelineTask &task,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    std::byte *tma_storage,
    forward_return_detail::ReturnCopyBarrier *tma_full) {
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *receive_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_receive_payload_offset);
  const auto source_partition = static_cast<std::uint32_t>(task.expert);
  const auto partition_packet_base =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank;
  const auto partition_fanout_base =
      partition_packet_base * a.dispatch.identity.top_k;
  const auto vector_count = a.hidden / 8ULL;
  constexpr std::uint32_t kWarps = forward_tuning::kRelayScatterWarps;
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
    bool copied = false;
    for (std::uint64_t item = batch_item_begin + warp;
         warp < kWarps && item < batch_item_end; item += kWarps) {
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
          const auto gate = target.gate;
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
                                  execution_epoch(a),
                                  header.sequence,
                                  route,
                                  header.local_token_id,
                                  destination_row,
                                  target.expert_id,
                                  target.topk_slot};
          auto *destination_gates = peer_at<float>(
              a.peer_bases[destination], a.canonical_gates_offset);
          destination_gates[destination_row] = gate;
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
#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)
      if (lane == 0U) {
        const auto copy_bytes = static_cast<std::uint32_t>(a.hidden * 2ULL);
        auto *warp_storage = tma_storage + warp * copy_bytes;
        auto &full = tma_full[warp];
        ::cuda::device::experimental::cp_async_bulk_global_to_shared(
            warp_storage, source, copy_bytes, full);
        auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, copy_bytes);
        full.wait(static_cast<decltype(ticket) &&>(ticket));
        ::cuda::device::experimental::cp_async_bulk_shared_to_global(
            target, warp_storage, copy_bytes);
        ::cuda::device::experimental::cp_async_bulk_commit_group();
        ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
        copied = true;
      }
#else
      for (std::uint64_t vector = lane; vector < vector_count; vector += 32ULL)
        target[vector] = source[vector];
#endif
    }

#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)
    if (copied)
      ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
#endif
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
      const auto compact_row = target.return_compact_row;
      auto *peer_map = peer_at<unsigned long long>(
          a.peer_bases[destination], a.forward_return_row_map_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(peer_map[target.dst_row]);
      published.store(
          encode_return_row_mapping(partition_fanout_base + compact_row),
          ::cuda::memory_order_release);
      // Input readiness also releases its immutable reverse map. FC2 and
      // Return consumers therefore never need a second mapping handshake.
      const auto rail = a.rank % a.local_world_size;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          arrived[target.mtile * a.local_world_size + rail]);
      ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);

#if WEFTMOE_DEBUG_VALIDATE
      const auto record_index =
          source_partition * a.gin_context_count + task.context;
      atomicAdd(&context_completion[record_index].relay_fanout_rows, 1ULL);
      if (target.mtile == 0U)
        atomicAdd(&context_completion[record_index].relay_tile0_rows, 1ULL);
      atomicAdd(&completion->remote_rows_published, 1ULL);
      if (target.mtile < 64U)
        atomicOr(&completion->published_tile_mask, 1ULL << target.mtile);
#endif
    }
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    const auto record_index =
        source_partition * a.gin_context_count + task.context;
    auto &record = context_completion[record_index];
    atomicAdd(&record.relay_claimed, static_cast<unsigned long long>(task.row_count));
    atomicAdd(&record.relay_completed, static_cast<unsigned long long>(task.row_count));
    atomicMax(&record.last_progress_globaltimer, device_globaltimer_ns());
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
  // Relay CTAs share the channel claim cursor; a slow copy never prevents
  // another worker from taking the next received wave.
  const auto remote_node = a.node == 0U ? 1U : 0U;
  const auto source_partition =
      remote_node < a.node ? remote_node : remote_node - 1U;
  __shared__ std::uint64_t local_routes;
  __shared__ GinDispatchContextRecord inbound_directory;
  __shared__ std::uint64_t relay_packet_cursor;
  __shared__ std::uint64_t relay_waves;
  __shared__ persistent_v3::PipelineTask inbound_task;
  __shared__ forward_return_detail::LocalReturnState local_return_state;
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ forward_return_detail::ReturnCopyBarrier
      local_return_tma_full[persistent_v3::kWarpsPerCta];
  __shared__ forward_return_detail::ReturnReceiveState
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
    local_return_state = forward_return_detail::LocalReturnState{};
    completion_state = forward_return_detail::ReturnReceiveState{};
    local_return_done = 0U;
    completion_done = 0U;
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
  bool inbound_terminal = false;
  bool dispatch_terminal_published = false;
  bool completion_timeline_started = false;
  bool completion_terminal_published = false;
  WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay, context,
                          0ULL, execution_epoch(a));
  while ((!dispatch_terminal_published || local_return_done == 0U ||
          completion_done == 0U) &&
         !cancelled(a, control)) {
    bool progressed = false;

    if (!inbound_terminal) {
      if (threadIdx.x == 0U) {
        inbound_ready = 0U;
        if (inbound_directory_seen == 0U &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context].inbound_directory_ready.value) >=
                execution_epoch(a)) {
          // The GIN owner crossed the async-proxy boundary before releasing
          // this epoch.  Keep the fence here as well so this independent SM's
          // generic loads cannot observe an older receive-buffer image.
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto directory =
              receive_directories[receive_directory_index];
          if (directory.generation != a.dispatch.identity.generation ||
              directory.epoch != execution_epoch(a)) {
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
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              claim(control.dispatch[context].inbound_claim.value);
          auto observed = claim.load(::cuda::memory_order_relaxed);
          relay_waves = observed >> 32U;
          relay_packet_cursor = inbound_directory.packet_begin +
                                (observed & 0xffffffffULL);
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
                inbound_task.epoch = execution_epoch(a);
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
                const auto next = ((relay_waves + 1ULL) << 32U) |
                    ((observed & 0xffffffffULL) + wave_packets);
                inbound_ready = claim.compare_exchange_strong(
                    observed, next, ::cuda::memory_order_relaxed,
                    ::cuda::memory_order_relaxed) ? 1U : 0U;
              }
            }
          }
        }
      }
      __syncthreads();
      if (inbound_ready != 0U) {
        progressed = true;
        relay_inbound_wave(a, inbound_task, completion,
                           context_completion, reinterpret_cast<std::byte *>(storage),
                           local_return_tma_full);
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

#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)
    // Mirror the compute scheduler's FC2-first policy at the node-local
    // transport boundary.  Once an FC2 M-block is available, retire one
    // bounded Return batch before issuing another Dispatch batch.  Skipping
    // Dispatch only in the iteration that actually consumed a row batch
    // prevents the TMA streams from competing for the same Relay CTA and
    // NVLink path while keeping the scheduler work-conserving.
    bool local_return_progressed = false;
    if (local_return_done == 0U) {
      local_return_progressed =
          forward_return_detail::progress_local_return(
              a, control, context, scheduler, local_return_state,
              reinterpret_cast<std::byte *>(storage),
              local_return_tma_full);
      progressed = local_return_progressed || progressed;
      if (threadIdx.x == 0U &&
          forward_return_detail::local_return_terminal(
              a, scheduler, local_return_state)) {
        local_return_done = 1U;
        WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnTerminal, LocalReturn,
                                context, local_return_state.cursor,
                                execution_epoch(a));
      }
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
        WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible, GinDispatch,
                                context, local_begin, execution_epoch(a));
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
        WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible, GinDispatch,
                                context, local_begin, execution_epoch(a));
      local_begin += local_stride;
      progressed = true;
    }
#endif

    const bool dispatch_terminal =
        inbound_terminal && local_begin >= local_routes;
    if (dispatch_terminal && !dispatch_terminal_published) {
      WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, context,
                              local_routes, execution_epoch(a));
      WEFTMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay, context,
                              local_routes, execution_epoch(a));
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

#if !defined(WEFTMOE_FORWARD_TRANSPORT_TU)
    // One bounded unit from each Return plane keeps Dispatch latency favored
    // without delaying already-computed FC2 rows until the complete Dispatch
    // or FC2 task spaces drain.
    if (local_return_done == 0U) {
      progressed = forward_return_detail::progress_local_return(
                       a, control, context, scheduler, local_return_state,
                       reinterpret_cast<std::byte *>(storage),
                       local_return_tma_full) ||
                   progressed;
      if (threadIdx.x == 0U &&
          forward_return_detail::local_return_terminal(
              a, scheduler, local_return_state)) {
        local_return_done = 1U;
        WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnTerminal, LocalReturn,
                                context, local_return_state.cursor,
                                execution_epoch(a));
      }
      __syncthreads();
    }
#endif

    if (completion_done == 0U) {
      for (std::uint32_t pass = 0U;
           pass < forward_tuning::kCompletionPeerProbes; ++pass) {
        const auto completion_progressed =
            forward_return_detail::progress_return_receive(
                a, control, context, scheduler, completion_state);
        progressed = completion_progressed || progressed;
        if (forward_return_detail::return_receive_terminal(
                completion_state))
          break;
      }
      if (!completion_timeline_started && completion_state.initialized != 0U) {
        WEFTMOE_TIMELINE_RECORD(a.timeline, SourceRelayStart, SourceRelay,
                                context, completion_state.expected,
                                execution_epoch(a));
        completion_timeline_started = true;
      }
      if (forward_return_detail::return_receive_terminal(
              completion_state)) {
        if (!completion_terminal_published) {
          WEFTMOE_TIMELINE_RECORD(a.timeline, SourceRelayEnd, SourceRelay,
                                  context, completion_state.completed,
                                  execution_epoch(a));
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

    if (!progressed && threadIdx.x < 32U)
      forward_reduce_detail::try_reduce_one_token<true>(a);
    if (!progressed && threadIdx.x == 0U)
      __nanosleep(forward_tuning::kTransportIdleNanosleep);
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
  WEFTMOE_TIMELINE_RECORD(a.timeline, GinPutIssue, GinDispatch, task.context,
                          packets * a.hidden * sizeof(transport::BFloat16),
                          execution_epoch(a));
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
                  sizeof(ForwardDispatchFanoutRecord),
          a.gin_window,
          a.gin_send_fanout_offset +
              static_cast<std::size_t>(task.index) *
                  a.dispatch.identity.top_k *
                  sizeof(ForwardDispatchFanoutRecord),
          packets * a.dispatch.identity.top_k *
              sizeof(ForwardDispatchFanoutRecord));
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
      execution_epoch(a), ncclGin_SignalInc{0}, ncclCoopThread{},
      ncclGin_None{}, ::cuda::thread_scope_block, ::cuda::thread_scope_device);

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
                           sizeof(ForwardDispatchFanoutRecord)));
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
        directory.epoch != execution_epoch(a))
      return false;
    packet_cursor = directory.packet_begin;
    directory_seen = true;
    auto &record = context_completion[
        source_partition * a.gin_context_count + context];
    record.directory_ready = 1ULL;
    record.relay_expected = directory.packet_count;
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].inbound_directory_ready.value,
        execution_epoch(a));
    return true;
  }
  if (waves_published >= directory.wave_count)
    return false;

  const auto packet =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank +
      packet_cursor;
  ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> marker(
      headers[packet].source_epoch);
  if (marker.load(::cuda::memory_order_acquire) != execution_epoch(a))
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
    std::uint32_t publisher_count, unsigned long long *scheduler) {
  constexpr std::uint32_t kStartupMblocks =
      forward_tuning::kStartupMblocks;
  const auto locality_group = a.forward_fc1_locality_group;
  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *published_mblocks = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
#if WEFTMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif

  while (!forward_cancelled(a)) {
    bool pending = false;
    bool progressed = false;
    for (std::uint64_t mblock = context; mblock < forward_mblocks;
         mblock += publisher_count) {
      if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
        continue;
      pending = true;
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);

      std::uint32_t group_mblocks = 1U;
      if (local_mblock >= kStartupMblocks) {
        // Four-way N-tile-major publication gives the best W1 L2 reuse, but
        // using it for an expert's first or last body blocks makes FC2 and
        // Return arrive in large bursts.  Keep the configured two-block
        // cadence at both ends and widen only the saturated middle window.
        const bool middle_group4 =
            locality_group == 2U &&
            local_mblock >= a.forward_fc1_locality_group4_begin &&
            local_mblock < a.forward_fc1_locality_group4_end;
        const auto effective_group = middle_group4 ? 4U : locality_group;
        const auto group_origin =
            middle_group4 ? a.forward_fc1_locality_group4_begin
                          : kStartupMblocks;
        const auto group_offset =
            (local_mblock - group_origin) % effective_group;
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
              min(effective_group, expert_mblocks - local_mblock);
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
#if WEFTMOE_DEBUG_VALIDATE
      atomicMax(&completion->bulk_pack_complete,
                queue_base +
                    static_cast<unsigned long long>(group_mblocks) *
                        fc1_ntiles);
#endif
    }
    if (!pending)
      break;
    if (!progressed)
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
#if WEFTMOE_DEBUG_VALIDATE
  atomicOr(&completion->bulk_contexts_issuing, 1ULL << context);
#endif
}

__device__ __forceinline__ void publish_ready_fc1_tasks(
    const ForwardKernelArgs &a, std::uint32_t context,
    std::uint32_t publisher_count, unsigned long long *scheduler) {
  if (a.forward_fc1_locality_group > 1U) {
    publish_ready_fc1_locality_groups(a, context, publisher_count, scheduler);
    return;
  }
  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *published_mblocks = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
#if WEFTMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif
  auto remaining =
      context < forward_mblocks
          ? (forward_mblocks - 1ULL - context) / publisher_count +
                1ULL
          : 0ULL;
  while (remaining != 0ULL && !forward_cancelled(a)) {
    bool progressed = false;
    for (std::uint64_t mblock = context; mblock < forward_mblocks;
         mblock += publisher_count) {
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
      const auto queue_base = atomicAdd(
          &scheduler[12ULL + 2ULL * a.world],
          static_cast<unsigned long long>(fc1_ntiles));
      for (std::uint32_t ntile = 0; ntile < fc1_ntiles; ++ntile) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            entry(fc1_ready_queue[queue_base + ntile]);
        const auto task = mblock * fc1_ntiles + ntile;
        entry.store(task + 1ULL, ::cuda::memory_order_release);
      }
      --remaining;
      progressed = true;
#if WEFTMOE_DEBUG_VALIDATE
      completion->bulk_pack_complete = queue_base + fc1_ntiles;
#endif
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

      // One release RMW per contiguous (destination,M-block) run publishes
      // every row copied above.  A block-scope barrier followed by this
      // system-scope release forms the transitive publication edge for all
      // issuing warps, while reducing the normal 128-row M-block from roughly
      // 128 system atomics to one per contributing source rank.
      for (auto local = static_cast<std::uint64_t>(threadIdx.x);
           local < batch_end - batch_begin; local += kCopyThreads) {
        const auto position = batch_begin + local;
        const auto route = a.dispatch_waves.local_route_order[position];
        const auto destination = a.dispatch.route_dst_rank[route];
        const auto mtile = a.dispatch_waves.route_dst_mtile[route];
        bool run_leader = local == 0U;
        if (!run_leader) {
          const auto previous_route =
              a.dispatch_waves.local_route_order[position - 1ULL];
          run_leader =
              a.dispatch.route_dst_rank[previous_route] != destination ||
              a.dispatch_waves.route_dst_mtile[previous_route] != mtile;
        }
        if (!run_leader)
          continue;
        auto run_end = position + 1ULL;
        while (run_end < batch_end) {
          const auto next_route = a.dispatch_waves.local_route_order[run_end];
          if (a.dispatch.route_dst_rank[next_route] != destination ||
              a.dispatch_waves.route_dst_mtile[next_route] != mtile)
            break;
          ++run_end;
        }
        const auto rows = run_end - position;
        auto *arrived = peer_at<unsigned long long>(
            a.peer_bases[destination], a.x_arrived_rows_offset);
        const auto rail = a.rank % a.local_world_size;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(arrived[static_cast<std::uint64_t>(mtile) *
                              a.local_world_size +
                          rail]);
        ready.fetch_add(rows, ::cuda::memory_order_release);
        if (destination != a.rank)
          peer_rows += rows;
        if (relay_ordinal == 0U && batch_begin == 0ULL && local == 0ULL)
          WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible,
                                  GinDispatch, 0U, route, execution_epoch(a));
      }
      asm volatile("bar.sync 2, %0;" : : "n"(kCopyThreads) : "memory");
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
  // whole CTA and the same row-sized SMEM slots for Return. All Relay CTAs
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
    while (local_return_done == 0U && !cancelled(a, control)) {
      const bool progressed =
          forward_return_detail::progress_local_return(
              a, control, context, scheduler, local_return_state,
              reinterpret_cast<std::byte *>(storage), copy_full);
      if (!progressed && threadIdx.x < 32U)
        forward_reduce_detail::try_reduce_one_token<true>(a);
      if (threadIdx.x == 0U) {
        if (forward_return_detail::local_return_terminal(
                a, scheduler, local_return_state)) {
          local_return_done = 1U;
          WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnTerminal, LocalReturn,
                                  context, local_return_state.cursor,
                                  execution_epoch(a));
        } else if (!progressed) {
          __nanosleep(forward_tuning::kReturnIdleNanosleep);
        }
      }
      __syncthreads();
    }
  }
}

#endif
__device__ __forceinline__ void rdma_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler) {
  if (threadIdx.x == 0U) {
    const ncclGin gin(a.gin_dev_comm, context);
    // Logical Return contexts own destination peers. Stagger their physical
    // GIN contexts across senders so fan-in does not concentrate the active
    // QPs on one RoCE LAG port. Receivers use this same permutation when
    // reading the sender-specific native signal.
    const auto return_context = (context + a.rank) % a.gin_context_count;
    const ncclGin return_gin(a.gin_dev_comm,
                             return_context + a.gin_context_count);
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
        a.dispatch.identity.generation, execution_epoch(a), 0ULL, 0U, 0U};
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
        execution_epoch(a));

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
    forward_return_detail::ReturnRdmaState direct_full_state{};
    while ((!outbound_terminal || !inbound_terminal) &&
           !cancelled(a, control)) {
      bool progressed = false;
      if (!outbound_terminal) {
        if (!outbound_directory_issued &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context].outbound_directory_ready.value) >=
                execution_epoch(a)) {
          persistent_v3::PipelineTask directory_task{};
          directory_task.epoch = execution_epoch(a);
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
              wave_task.epoch = execution_epoch(a);
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

        if (outbound_directory_issued && outbound_wave == outbound_wave_end &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context].outbound_pack_complete.value) >=
                execution_epoch(a)) {
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
          WEFTMOE_TIMELINE_RECORD(a.timeline, RemoteSignalObserved, GinDispatch,
                                  context, waves_published, execution_epoch(a));
          WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstRemoteSignal, GinDispatch,
                                  context, waves_published, execution_epoch(a));
          progressed = true;
        }
      }

      progressed =
          forward_return_detail::progress_return_send(
              a, control, return_gin, world, context, completion,
              context_completion, scheduler, direct_full_state) ||
          progressed;

      if (!progressed)
        __nanosleep(forward_tuning::kTransportIdleNanosleep);
    }

    if (issued_completions != 0ULL) {
      while (gin.readCounter(0, 56, ::cuda::memory_order_acquire) <
                 issued_completions &&
             !cancelled(a, control)) {
        if (!forward_return_detail::progress_return_send(
                a, control, return_gin, world, context, completion,
                context_completion, scheduler, direct_full_state))
          __nanosleep(forward_tuning::kTransportIdleNanosleep);
      }
    }
    WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchLocalCompletion, GinDispatch,
                            context, issued_completions, execution_epoch(a));

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
    forward_return_detail::drain_return_send(
        a, control, context, completion, context_completion, scheduler,
        return_gin, world, direct_full_state);
  } else if (threadIdx.x == 32U) {
    publish_ready_fc1_tasks(a, context, persistent_v3::kRdmaCtaCount,
                            scheduler);
  } else if (threadIdx.x >= kRdmaControlWarps * 32U) {
    rdma_pack_outbound_context(a, control, context);
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
