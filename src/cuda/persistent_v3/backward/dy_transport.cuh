#pragma once

// This header is included by activation_transport.cu after the private
// Backward launch contract and GIN wire records are visible.  It deliberately
// contains only communication-role code; WGMMA remains in the aggregation TU
// and is inlined into Compute CTAs.

#if __has_include(<nccl_device.h>)

// One batch is also one canonical scheduling quantum.  With 128 routes the
// eight Relay CTAs have enough independent vector traffic to amortize the
// system fence while publishing no coarser than an M-block.
inline constexpr std::uint32_t kBackwardV3LocalRouteBatch = 128U;
inline constexpr std::uint32_t kBackwardV3DyBulkStages = 12U;
inline constexpr std::uint32_t kBackwardV3DyMainHidden = 7168U;
inline constexpr std::uint32_t kBackwardV3DyRowBytes =
    kBackwardV3DyMainHidden * sizeof(transport::BFloat16);
static_assert(kBackwardV3DyRowBytes <= 16U * 1024U &&
              kBackwardV3DyRowBytes % 16U == 0U);
// GIN reaches its steady bandwidth only for multi-megabyte transfers. The
// fine-grained readiness catalog is therefore coalesced into the runtime
// v3_return_group_rows quantum (2048 rows by default at the main shape).

// Optional host-mapped diagnostics. The allocation is present for the legacy
// progress dump already; production passes nullptr, so these stores disappear
// behind one uniform branch and add no system traffic to measured runs. Slots
// 1..4 are owned by Relay and 5..7 by RDMA for each of the eight contexts.
__device__ __forceinline__ void backward_v3_debug_progress(
    const BackwardKernelArgs &a, std::uint32_t context,
    std::uint32_t slot, std::uint64_t value) {
  if (a.progress == nullptr || context >= a.progress_experts || slot == 0U ||
      slot >= 8U)
    return;
  constexpr std::uint64_t kProgressExpertBase = 32ULL;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> word(
      a.progress[kProgressExpertBase +
                 static_cast<std::uint64_t>(context) * 8ULL + slot]);
  word.store(value, ::cuda::memory_order_release);
}

// Mirror the public Backward cancellation word into the V3 generation state.
// The host updates that word from a separate stream, so it must be observed
// with a system-scope atomic; all role-local loops can then use the cheaper
// device-scope generation flag.
__device__ __forceinline__ bool backward_v3_generation_cancelled(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control) {
  if (persistent_v3::generation_cancelled(control.generation, a.epoch))
    return true;
  if (!backward_cancelled(a))
    return false;
  persistent_v3::store_release<::cuda::thread_scope_device>(
      control.generation.control.cancel_epoch.value, a.epoch);
  return true;
}

__device__ __forceinline__ void backward_v3_publish_local_dy_batch(
    const BackwardKernelArgs &a, std::uint64_t route_begin,
    std::uint64_t route_end, std::uint32_t relay_ordinal,
    unsigned char *bulk_storage,
    std::uint32_t *bulk_stage_epochs) {
  (void)relay_ordinal;
  const auto routes = route_end - route_begin;
  if (a.v3_generic_dy_copy) {
    constexpr std::uint64_t kVectorsPerRow =
        kBackwardV3DyRowBytes / sizeof(uint4);
    const auto warp = threadIdx.x / 32U;
    const auto lane = threadIdx.x % 32U;
    const auto warps = blockDim.x / 32U;
    for (std::uint64_t local_route = warp; local_route < routes;
         local_route += warps) {
      const auto route =
          a.dispatch_waves.local_route_order[route_begin + local_route];
      const auto token = route / a.dispatch.identity.top_k;
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      const auto *source = reinterpret_cast<const uint4 *>(
          a.dy_input + token * kBackwardV3DyMainHidden);
      auto *destination_dy = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.dy_canonical_offset);
      auto *target = reinterpret_cast<uint4 *>(
          destination_dy + row * kBackwardV3DyMainHidden);
      for (std::uint64_t vector = lane; vector < kVectorsPerRow;
           vector += 32ULL)
        target[vector] = source[vector];
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      if (lane == 0U)
        destination_gate[row] = a.gates[route];
    }
    // A system fence is expensive enough that issuing it once per lane and
    // route dominates the actual NVLink payload copy.  Every thread drains
    // all of its peer-global stores once for the complete 128-route batch;
    // the block rendezvous then makes it safe for any thread to publish an
    // exact destination row.  Readiness remains no coarser than one M-block.
    __threadfence_system();
    __syncthreads();
    for (std::uint64_t local_route = threadIdx.x; local_route < routes;
         local_route += blockDim.x) {
      const auto route =
          a.dispatch_waves.local_route_order[route_begin + local_route];
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      auto *arrived = peer_at<unsigned long long>(
          a.peer_bases[destination], a.dy_arrived_rows_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(arrived[row]);
      // Canonical rows have one transport producer.  A release store is the
      // required publication operation; an acquire RMW needlessly sends the
      // old zero back across NVLink.
      ready.store(1ULL, ::cuda::memory_order_release);
    }
  } else if (threadIdx.x == 0U) {
    auto *barriers =
        reinterpret_cast<cutlass::arch::ClusterTransactionBarrier *>(
            bulk_storage +
            kBackwardV3DyBulkStages * kBackwardV3DyRowBytes);
    for (std::uint64_t batch = 0; batch < routes;
         batch += kBackwardV3DyBulkStages) {
      const auto batch_rows = static_cast<std::uint32_t>(
          min(static_cast<std::uint64_t>(kBackwardV3DyBulkStages),
              routes - batch));
      for (std::uint32_t stage = 0; stage < batch_rows; ++stage) {
        const auto route = a.dispatch_waves.local_route_order[
            route_begin + batch + stage];
        const auto token = route / a.dispatch.identity.top_k;
        cute::SM90_BULK_COPY_G2S::copy(
            a.dy_input + token * kBackwardV3DyMainHidden,
            reinterpret_cast<std::uint64_t *>(&barriers[stage]),
            bulk_storage + stage * kBackwardV3DyRowBytes,
            kBackwardV3DyRowBytes);
        barriers[stage].arrive_and_expect_tx(kBackwardV3DyRowBytes);
      }
      for (std::uint32_t stage = 0; stage < batch_rows; ++stage) {
        barriers[stage].wait(bulk_stage_epochs[stage] & 1U);
        ++bulk_stage_epochs[stage];
        const auto route = a.dispatch_waves.local_route_order[
            route_begin + batch + stage];
        const auto destination = a.dispatch.route_dst_rank[route];
        const auto row = a.dispatch.route_dst_row[route];
        auto *destination_dy = peer_at<transport::BFloat16>(
            a.peer_bases[destination], a.dy_canonical_offset);
        cute::tma_store_fence();
        cute::SM90_BULK_COPY_S2G::copy(
            bulk_storage + stage * kBackwardV3DyRowBytes,
            destination_dy + row * kBackwardV3DyMainHidden,
            kBackwardV3DyRowBytes);
        cute::tma_store_arrive();
      }
      backward_v3_tma_store_wait_complete();
    }
    // The exact-row readiness epoch is published through the generic
    // proxy below.  Move the completed bulk-store writes out of the async
    // proxy before that release; a system fence alone does not bridge Hopper
    // memory proxies.
    asm volatile("fence.proxy.async.global;" : : : "memory");
  }
  __syncthreads();
  if (!a.v3_generic_dy_copy && threadIdx.x == 0U) {
    // In the bulk path thread 0 owns the async payload stores, gate stores,
    // and exact-row releases.
    for (std::uint64_t local_route = 0; local_route < routes; ++local_route) {
      const auto route =
          a.dispatch_waves.local_route_order[route_begin + local_route];
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      destination_gate[row] = a.gates[route];
    }
    __threadfence_system();
    for (std::uint64_t local_route = 0; local_route < routes; ++local_route) {
      const auto route =
          a.dispatch_waves.local_route_order[route_begin + local_route];
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      auto *arrived = peer_at<unsigned long long>(
          a.peer_bases[destination], a.dy_arrived_rows_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(arrived[row]);
      ready.store(1ULL, ::cuda::memory_order_release);
    }
  }
  if (threadIdx.x == 0U) {
    unsigned long long peer_rows = 0ULL;
    for (auto position = route_begin; position < route_end; ++position) {
      const auto route = a.dispatch_waves.local_route_order[position];
      const auto destination = a.dispatch.route_dst_rank[route];
      peer_rows += destination / a.local_world_size == a.node &&
                   destination != a.rank;
    }
    if (peer_rows != 0ULL) {
      auto *status =
          peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(&status->actual_dy_peer_bytes,
                peer_rows * a.hidden * sizeof(transport::BFloat16));
    }
  }
}

__device__ __forceinline__ std::uint64_t
backward_v3_local_route_prefix(const BackwardKernelArgs &a) {
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

__device__ __forceinline__ void backward_v3_publish_outbound_directory(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t context, std::uint32_t remote_node,
    std::uint64_t node_wave_begin, std::uint64_t context_wave_begin,
    std::uint64_t context_wave_end) {
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *directories = reinterpret_cast<GinDispatchContextRecord *>(
      bytes + a.gin_send_dispatch_directory_offset);
  const auto remote_partition =
      remote_node < a.node ? remote_node : remote_node - 1U;
  const auto node_packet_begin = a.dispatch_waves.offsets[node_wave_begin];
  const auto packet_begin =
      a.dispatch_waves.offsets[context_wave_begin] - node_packet_begin;
  const auto packet_end =
      a.dispatch_waves.offsets[context_wave_end] - node_packet_begin;
  const auto directory_index =
      remote_partition * persistent_v3::kRdmaCtaCount + context;
  if (threadIdx.x == 0U)
    directories[directory_index] = GinDispatchContextRecord{
        a.dispatch.identity.generation, a.epoch, packet_begin,
        static_cast<std::uint32_t>(packet_end - packet_begin),
        static_cast<std::uint32_t>(context_wave_end - context_wave_begin)};
  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0U) {
    persistent_v3::PipelineTask task{};
    task.epoch = a.epoch;
    task.index = directory_index;
    task.expert = static_cast<std::uint16_t>(remote_node);
    task.kind = static_cast<std::uint8_t>(
        persistent_v3::TaskKind::DispatchRelay);
    task.context = static_cast<std::uint8_t>(context);
    const auto source_partition =
        a.node < remote_node ? a.node : a.node - 1U;
    task.row_begin = source_partition * persistent_v3::kRdmaCtaCount +
                     context;
    task.flags = persistent_v3::kPipelineTaskDirectory;
    while (!persistent_v3::try_push(control.relay_to_rdma[context], task) &&
           !backward_v3_generation_cancelled(a, control))
      __nanosleep(64U);
  }
  __syncthreads();
}

// One main-shape dY row is 14 KiB, within Hopper's 16 KiB bulk-copy limit.
// A Relay CTA pipelines twelve complete rows through otherwise-unused dynamic
// shared memory.  This keeps the fixed eight-CTA Relay role while replacing
// hundreds of scalar instructions per row with the asynchronous copy engines.
__device__ __forceinline__ void backward_v3_bulk_copy_outbound_payload(
    const BackwardKernelArgs &a, transport::BFloat16 *send_payload,
    std::uint64_t wave_begin, std::uint64_t wave_end,
    unsigned char *bulk_storage, std::uint32_t *bulk_stage_epochs) {
  auto *stage_bytes = bulk_storage;
  auto *barriers =
      reinterpret_cast<cutlass::arch::ClusterTransactionBarrier *>(
          stage_bytes +
          kBackwardV3DyBulkStages * kBackwardV3DyRowBytes);

  const auto wave_packets = wave_end - wave_begin;
  for (std::uint64_t batch = 0; batch < wave_packets;
       batch += kBackwardV3DyBulkStages) {
    const auto batch_rows = static_cast<std::uint32_t>(
        min(static_cast<std::uint64_t>(kBackwardV3DyBulkStages),
            wave_packets - batch));
    for (std::uint32_t stage = 0; stage < batch_rows; ++stage) {
      const auto position = wave_begin + batch + stage;
      const auto packet = a.dispatch_waves.packet_order[position];
      const auto token = a.dispatch.packet_headers[packet].local_token_id;
      cute::SM90_BULK_COPY_G2S::copy(
          a.dy_input +
              static_cast<std::uint64_t>(token) *
                  kBackwardV3DyMainHidden,
          reinterpret_cast<std::uint64_t *>(&barriers[stage]),
          stage_bytes + stage * kBackwardV3DyRowBytes,
          kBackwardV3DyRowBytes);
      barriers[stage].arrive_and_expect_tx(kBackwardV3DyRowBytes);
    }
    for (std::uint32_t stage = 0; stage < batch_rows; ++stage) {
      barriers[stage].wait(bulk_stage_epochs[stage] & 1U);
      ++bulk_stage_epochs[stage];
      cute::tma_store_fence();
      cute::SM90_BULK_COPY_S2G::copy(
          stage_bytes + stage * kBackwardV3DyRowBytes,
          send_payload +
              (wave_begin + batch + stage) *
                  static_cast<std::uint64_t>(kBackwardV3DyMainHidden),
          kBackwardV3DyRowBytes);
      cute::tma_store_arrive();
    }
    backward_v3_tma_store_wait_complete();
  }
}

__device__ __forceinline__ void backward_v3_pack_outbound_wave(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t context, std::uint32_t remote_node,
    std::uint64_t node_packet_begin, std::uint64_t wave,
    unsigned char *bulk_storage, std::uint32_t *bulk_stage_epochs) {
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_send_header_offset);
  auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_send_fanout_offset);
  auto *send_gate =
      reinterpret_cast<float *>(bytes + a.gin_send_gate_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_send_payload_offset);
  const auto wave_begin = a.dispatch_waves.offsets[wave];
  const auto wave_end = a.dispatch_waves.offsets[wave + 1ULL];
  const auto wave_packets = wave_end - wave_begin;
  if (threadIdx.x == 0U && !a.v3_direct_dy_input)
    backward_v3_bulk_copy_outbound_payload(
        a, send_payload, wave_begin, wave_end, bulk_storage,
        bulk_stage_epochs);

  const auto fanout_items = wave_packets * a.dispatch.identity.top_k;
  for (std::uint64_t item = threadIdx.x; item < fanout_items;
       item += blockDim.x) {
    const auto position = wave_begin + item / a.dispatch.identity.top_k;
    const auto fanout_slot = item % a.dispatch.identity.top_k;
    const auto packet = a.dispatch_waves.packet_order[position];
    const auto descriptor = a.dispatch.packet_headers[packet];
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    if (fanout_begin + fanout_slot < fanout_end) {
      const auto target =
          a.dispatch.packet_fanout[fanout_begin + fanout_slot];
      const auto packed = position * a.dispatch.identity.top_k + fanout_slot;
      send_fanout[packed] = target;
      send_gate[packed] =
          a.gates[descriptor.local_token_id * a.dispatch.identity.top_k +
                  target.topk_slot];
    }
  }

  for (std::uint64_t position = wave_begin + threadIdx.x;
       position < wave_end; position += blockDim.x) {
    const auto packet = a.dispatch_waves.packet_order[position];
    const auto descriptor = a.dispatch.packet_headers[packet];
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    const auto wire_position = position - node_packet_begin;
    // source_epoch is the late readiness marker.  The header bulk put first
    // installs zero; the RDMA CTA overwrites only the first packet's marker
    // after payload, fanout and gate puts have settled remotely.
    send_headers[position] = LocalP2pRouteHeader{
        transport::kActivationTransportSchemaVersion,
        a.rank,
        descriptor.proxy_rank,
        descriptor.channel,
        a.dispatch.identity.generation,
        0ULL,
        wire_position,
        fanout_end - fanout_begin,
        descriptor.local_token_id,
        wire_position * a.dispatch.identity.top_k,
        0U,
        0U};
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    send_headers[wave_begin].topk_slot =
        static_cast<std::uint32_t>(wave_packets);
  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0U) {
    if (wave_packets > UINT16_MAX) {
      persistent_v3::publish_error_once(control.generation, 301ULL);
    } else {
      persistent_v3::PipelineTask task{};
      task.epoch = a.epoch;
      task.index = static_cast<std::uint32_t>(wave_begin);
      task.expert = static_cast<std::uint16_t>(remote_node);
      task.kind = static_cast<std::uint8_t>(
          persistent_v3::TaskKind::DispatchRelay);
      task.context = static_cast<std::uint8_t>(context);
      task.row_begin = static_cast<std::uint32_t>(
          wave_begin - node_packet_begin);
      task.row_count = static_cast<std::uint16_t>(wave_packets);
      while (!persistent_v3::try_push(control.relay_to_rdma[context], task) &&
             !backward_v3_generation_cancelled(a, control))
        __nanosleep(64U);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void backward_v3_relay_inbound_wave(
    const BackwardKernelArgs &a, const persistent_v3::PipelineTask &task,
    unsigned char *bulk_storage, std::uint32_t *bulk_stage_epochs) {
  // The RDMA CTA observed source_epoch and handed this wave off through a
  // device queue.  That acquire/release chain orders the task record, but it
  // does not cross from GIN's async proxy to the proxy used by TMA and normal
  // CUDA loads.  Perform the transition in the consuming/issuing thread.
  if (a.v3_generic_dy_copy || threadIdx.x == 0U)
    asm volatile("fence.proxy.async.global;" : : : "memory");
  __syncthreads();
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *receive_gate =
      reinterpret_cast<float *>(bytes + a.gin_receive_gate_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_receive_payload_offset);
  const auto source_partition = static_cast<std::uint32_t>(task.expert);
  const auto partition_packet_base =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank;
  const auto partition_fanout_base =
      partition_packet_base * a.dispatch.identity.top_k;
  const auto fanout_items =
      static_cast<std::uint64_t>(task.row_count) *
      a.dispatch.identity.top_k;
  if (a.v3_generic_dy_copy) {
    constexpr std::uint64_t kVectorsPerRow =
        kBackwardV3DyRowBytes / sizeof(uint4);
    const auto warp = threadIdx.x / 32U;
    const auto lane = threadIdx.x % 32U;
    const auto warps = blockDim.x / 32U;
    for (std::uint64_t item = warp; item < fanout_items; item += warps) {
      const auto packet_in_wave = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto sequence = task.row_begin + packet_in_wave;
      const auto packet = partition_packet_base + sequence;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target_record = receive_fanout[
          partition_fanout_base + header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size +
          target_record.destination_local_rank;
      const auto source_row =
          a.v3_direct_dy_input
              ? partition_packet_base + header.local_token_id
              : packet;
      const auto *source = reinterpret_cast<const uint4 *>(
          receive_payload + source_row * kBackwardV3DyMainHidden);
      auto *destination_dy = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.dy_canonical_offset);
      auto *target = reinterpret_cast<uint4 *>(
          destination_dy + target_record.dst_row * kBackwardV3DyMainHidden);
      for (std::uint64_t vector = lane; vector < kVectorsPerRow;
           vector += 32ULL)
        target[vector] = source[vector];
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      if (lane == 0U)
        destination_gate[target_record.dst_row] =
            receive_gate[partition_fanout_base + header.dst_row +
                         fanout_slot];
    }
    // As in local delivery, drain each lane's complete peer-store stream once
    // per wave, then publish exact rows in parallel.  This preserves the
    // system-scope producer/consumer contract without serializing every 14-KiB
    // row behind its own block of 32 system fences.
    __threadfence_system();
    __syncthreads();
    for (std::uint64_t item = threadIdx.x; item < fanout_items;
         item += blockDim.x) {
      const auto packet_in_wave = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto sequence = task.row_begin + packet_in_wave;
      const auto packet = partition_packet_base + sequence;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target_record = receive_fanout[
          partition_fanout_base + header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size +
          target_record.destination_local_rank;
      auto *arrived = peer_at<unsigned long long>(
          a.peer_bases[destination], a.dy_arrived_rows_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(arrived[target_record.dst_row]);
      ready.store(1ULL, ::cuda::memory_order_release);
    }
  } else if (threadIdx.x == 0U) {
    auto *barriers =
        reinterpret_cast<cutlass::arch::ClusterTransactionBarrier *>(
            bulk_storage +
            kBackwardV3DyBulkStages * kBackwardV3DyRowBytes);
    // dY is packet-major on the wire, while the local delivery is fanout
    // major.  Load each 14-KiB source row into shared memory once, then launch
    // every peer store for that packet from the same stage.  The previous
    // fanout-major loop redundantly reloaded the source row (2.6 times on the
    // main shape) and drained the async store pipeline every twelve fanouts.
    const auto wave_packets = static_cast<std::uint64_t>(task.row_count);
    for (std::uint64_t packet_batch = 0; packet_batch < wave_packets;
         packet_batch += kBackwardV3DyBulkStages) {
      const auto batch_packets = static_cast<std::uint32_t>(min(
          static_cast<std::uint64_t>(kBackwardV3DyBulkStages),
          wave_packets - packet_batch));
      for (std::uint32_t stage = 0; stage < batch_packets; ++stage) {
        const auto packet_in_wave = packet_batch + stage;
        const auto sequence = task.row_begin + packet_in_wave;
        const auto packet = partition_packet_base + sequence;
        const auto header = receive_headers[packet];
        const auto source_row =
            a.v3_direct_dy_input
                ? partition_packet_base + header.local_token_id
                : packet;
        cute::SM90_BULK_COPY_G2S::copy(
            receive_payload + source_row * kBackwardV3DyMainHidden,
            reinterpret_cast<std::uint64_t *>(&barriers[stage]),
            bulk_storage + stage * kBackwardV3DyRowBytes,
            kBackwardV3DyRowBytes);
        barriers[stage].arrive_and_expect_tx(kBackwardV3DyRowBytes);
      }
      for (std::uint32_t stage = 0; stage < batch_packets; ++stage) {
        barriers[stage].wait(bulk_stage_epochs[stage] & 1U);
        ++bulk_stage_epochs[stage];
        const auto packet_in_wave = packet_batch + stage;
        const auto packet = partition_packet_base + task.row_begin +
                            packet_in_wave;
        const auto header = receive_headers[packet];
        cute::tma_store_fence();
        for (std::uint32_t fanout_slot = 0U;
             fanout_slot < header.route_index; ++fanout_slot) {
          const auto target = receive_fanout[
              partition_fanout_base + header.dst_row + fanout_slot];
          const auto destination =
              a.node * a.local_world_size + target.destination_local_rank;
          auto *destination_dy = peer_at<transport::BFloat16>(
              a.peer_bases[destination], a.dy_canonical_offset);
          cute::SM90_BULK_COPY_S2G::copy(
              bulk_storage + stage * kBackwardV3DyRowBytes,
              destination_dy + target.dst_row * kBackwardV3DyMainHidden,
              kBackwardV3DyRowBytes);
        }
        // All fanouts from this immutable stage form one async store group.
        // One wait after the packet batch protects every stage from reuse.
        if (header.route_index != 0U)
          cute::tma_store_arrive();
      }
      backward_v3_tma_store_wait_complete();
    }
    // Publish the S2G copies to the generic proxy before the row counters can
    // release dV.  This pairs with the generic acquire and async-proxy fence
    // in the dV TMA producer without delaying unrelated M-blocks.
    asm volatile("fence.proxy.async.global;" : : : "memory");
  }
  if (!a.v3_generic_dy_copy)
    __threadfence_system();
  __syncthreads();
  if (!a.v3_generic_dy_copy && threadIdx.x == 0U) {
    for (std::uint64_t item = 0; item < fanout_items; ++item) {
      const auto packet_in_wave = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto sequence = task.row_begin + packet_in_wave;
      const auto packet = partition_packet_base + sequence;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto packed =
          partition_fanout_base + header.dst_row + fanout_slot;
      const auto target = receive_fanout[packed];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      destination_gate[target.dst_row] = receive_gate[packed];
    }
    __threadfence_system();
    for (std::uint64_t item = 0; item < fanout_items; ++item) {
      const auto packet_in_wave = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto sequence = task.row_begin + packet_in_wave;
      const auto packet = partition_packet_base + sequence;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target = receive_fanout[
          partition_fanout_base + header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *arrived = peer_at<unsigned long long>(
          a.peer_bases[destination], a.dy_arrived_rows_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(arrived[target.dst_row]);
      ready.store(1ULL, ::cuda::memory_order_release);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ bool backward_v3_context_wave_span(
    const BackwardKernelArgs &a, std::uint32_t context,
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
  if (node_wave_begin == node_wave_end)
    return false;
  const auto node_waves = node_wave_end - node_wave_begin;
  context_wave_begin =
      node_wave_begin +
      node_waves * context / persistent_v3::kRdmaCtaCount;
  context_wave_end =
      node_wave_begin +
      node_waves * (context + 1U) / persistent_v3::kRdmaCtaCount;
  return true;
}

__device__ __forceinline__ std::uint64_t
backward_v3_proxy_compact_rows(
    const ForwardReturnChunkRecord *chunks, std::uint64_t chunk_count,
    std::uint32_t source_node) {
  std::uint64_t rows = 0ULL;
  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    const auto record = chunks[chunk];
    if (record.source_node == source_node)
      rows = max(rows, record.row_begin + record.row_count);
  }
  return rows;
}

__device__ __forceinline__ std::uint64_t
backward_v3_outgoing_remote_fanout(const BackwardKernelArgs &a,
                                    std::uint32_t remote_node) {
  std::uint64_t rows = 0ULL;
  for (std::uint64_t packet = 0; packet < a.dispatch.sizes.packet_count;
       ++packet) {
    if (a.dispatch.packet_headers[packet].destination_node != remote_node)
      continue;
    rows += a.dispatch.packet_fanout_offsets[packet + 1ULL] -
            a.dispatch.packet_fanout_offsets[packet];
  }
  return rows;
}

// Source-side compact Return slices are disjoint by context.  A Relay CTA
// scatters one slice into canonical route slots and only then releases each
// token's readiness count.  The Compute role can therefore combine a token as
// soon as its exact top-k contributions have arrived, without a grid barrier.
__device__ __forceinline__ void backward_v3_scatter_return_slice(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    const persistent_v3::PipelineTask &task) {
  // The RDMA CTA acquired the GIN completion signal before publishing this
  // queue entry. Cross the Hopper async -> generic proxy boundary in every
  // thread that can load the receive window.  The queue release and the CTA
  // barrier distribute task metadata, but neither makes thread 0's proxy
  // fence apply to the generic loads issued by the other eleven warps.
  // Without the per-consumer transition those warps can intermittently read
  // an older Return payload even after the GIN completion signal is visible.
  asm volatile("fence.proxy.async.global;" : : : "memory");
  __syncthreads();
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *return_receive = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_return_receive_offset);
  auto *return_gate_receive = reinterpret_cast<float *>(
      bytes + a.gin_return_gate_receive_offset);
  // The V3 setup CTA builds this inverse directory in the dead Forward task
  // claim arena. Forward and the legacy oracle have both completed before V3
  // starts, so this is a phase-storage union rather than another allocation.
  const auto *source_route_index = peer_at<const unsigned long long>(
      a.local_base, a.claim_offset);
  auto *dx_return = peer_at<transport::BFloat16>(
      a.local_base, a.dx_return_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_offset);
  auto *token_state = peer_at<unsigned long long>(
      a.local_base, a.v3_token_state_offset);
  const auto row_begin = static_cast<std::uint64_t>(task.row_begin);
  const auto row_end = row_begin + task.row_count;
  const auto vector_count = a.hidden / 8ULL;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  constexpr std::uint32_t kWarpsPerCta = 12U;
  // Forward materializes the exact inverse compact-row directory once while
  // its immutable Dispatch metadata is hot. A Return slice can consequently
  // touch only its own rows instead of rescanning packet_count*top_k twice.
  // Direct consumption keeps the large dX plane in this contiguous receive
  // window; only the small dGate plane still needs route-major placement.
  if (!a.v3_direct_return_consume) {
    for (std::uint64_t compact = row_begin + warp; compact < row_end;
         compact += kWarpsPerCta) {
      auto route = lane == 0U ? source_route_index[compact] : 0ULL;
      route = __shfl_sync(0xffffffffU, route, 0U);
      auto *destination = reinterpret_cast<uint4 *>(dx_return) +
                          route * vector_count;
      const auto *source = reinterpret_cast<const uint4 *>(return_receive) +
                           compact * vector_count;
      for (std::uint64_t vector = lane; vector < vector_count;
           vector += 32ULL)
        destination[vector] = source[vector];
    }
  }
  for (std::uint64_t compact = row_begin + threadIdx.x; compact < row_end;
       compact += blockDim.x)
    dgate[source_route_index[compact]] = return_gate_receive[compact];

  __threadfence_system();
  __syncthreads();
  for (std::uint64_t compact = row_begin + threadIdx.x; compact < row_end;
       compact += blockDim.x) {
    const auto route = source_route_index[compact];
    const auto token = route / a.dispatch.identity.top_k;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        token_state[token]);
    // A token can receive routes from several Relay CTAs.  Preserve a
    // transitive publication of every route payload for the reducer that
    // acquires the final top-k count.
    ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.return_scatter_tasks_complete.value, 1ULL);
  __syncthreads();
}

__device__ __forceinline__ void backward_v3_relay_role(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t relay_ordinal, unsigned char *bulk_storage) {
  const auto context = relay_ordinal % persistent_v3::kRdmaCtaCount;
  // The Relay state machine interleaves local delivery, outbound packing and
  // inbound delivery in the same dynamic shared-memory arena.  A transaction
  // barrier must be initialized once and then advance monotonically across
  // all of those operations; reinitializing it after a TMA completion has
  // referenced the address is not a valid Hopper mbarrier lifetime transition.
  auto *bulk_barriers =
      reinterpret_cast<cutlass::arch::ClusterTransactionBarrier *>(
          bulk_storage +
          kBackwardV3DyBulkStages * kBackwardV3DyRowBytes);
  std::uint32_t bulk_stage_epochs[kBackwardV3DyBulkStages] = {};
  if (threadIdx.x == 0U) {
    for (std::uint32_t stage = 0; stage < kBackwardV3DyBulkStages; ++stage)
      bulk_barriers[stage].init(1);
    cutlass::arch::fence_barrier_init();
  }
  __shared__ std::uint64_t local_route_count;
  if (threadIdx.x == 0U)
    local_route_count = backward_v3_local_route_prefix(a);
  __syncthreads();

  // Publish every context directory before copying bulk data.  This lets the
  // peer RDMA CTA learn the exact wave count while the first local/outbound
  // batches are still being produced.  Empty context slices are intentional:
  // every receiving context expects one directory per active source node.
  for (std::uint32_t remote_node = 0;
       remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
    if (remote_node == a.node)
      continue;
    std::uint64_t node_wave_begin = 0ULL;
    std::uint64_t context_wave_begin = 0ULL;
    std::uint64_t context_wave_end = 0ULL;
    if (!backward_v3_context_wave_span(
            a, context, remote_node, node_wave_begin, context_wave_begin,
            context_wave_end))
      continue;
    backward_v3_publish_outbound_directory(
        a, control, context, remote_node, node_wave_begin,
        context_wave_begin, context_wave_end);
  }

  __shared__ persistent_v3::PipelineTask inbound_task;
  __shared__ unsigned int inbound_ready;
  __shared__ unsigned int return_publish_ready;
  __shared__ unsigned long long selected_return_chunk;
  auto *forward_return_chunks = peer_at<ForwardReturnChunkRecord>(
      a.local_base, a.forward_return_chunk_directory_offset);
  auto *forward_return_chunk_state = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_state_offset);
  auto *forward_return_chunk_ready_count = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_ready_count_offset);
  auto *forward_return_fragment_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_fragment_index_offset);
  const auto return_chunk_count = control.return_chunks;
  const auto owned_return_chunks =
      return_chunk_count > context
          ? 1ULL + (return_chunk_count - 1ULL - context) /
                       persistent_v3::kRdmaCtaCount
          : 0ULL;
  // The current V3 contract deliberately supports two nodes. Fine-grained
  // 128-row chunks remain independently packable by all Relay CTAs, while a
  // different static ownership partitions contiguous multi-MiB wire groups
  // over the eight GIN contexts. Thus no context waits for chunks it must
  // itself pack, and every wire operation starts as soon as its eight chunks
  // have reached system-visible staging memory.
  const auto return_source_node = a.node == 0U ? 1U : 0U;
  const auto return_compact_rows = backward_v3_proxy_compact_rows(
      forward_return_chunks, return_chunk_count, return_source_node);
  const auto return_group_count =
      (return_compact_rows + a.v3_return_group_rows - 1ULL) /
      a.v3_return_group_rows;
  const auto owned_return_groups =
      return_group_count > context
          ? 1ULL + (return_group_count - 1ULL - context) /
                       persistent_v3::kRdmaCtaCount
          : 0ULL;
  std::uint64_t return_chunk_cursor = 0ULL;
  std::uint64_t return_chunks_packed_by_role = 0ULL;
  std::uint64_t return_groups_issued = 0ULL;
  std::uint64_t return_debug_wait = ~0ULL;
  std::uint64_t return_debug_compute = ~0ULL;
  std::uint64_t return_debug_flags = ~0ULL;
  if (threadIdx.x == 0U)
    backward_v3_debug_progress(
        a, context, 1U,
        (owned_return_chunks << 32U) | owned_return_groups);
  auto local_route_begin =
      static_cast<std::uint64_t>(relay_ordinal) *
      kBackwardV3LocalRouteBatch;
  constexpr auto kLocalRouteStride =
      static_cast<std::uint64_t>(persistent_v3::kDefaultRelayCtaCount) *
      kBackwardV3LocalRouteBatch;
  std::uint32_t outbound_node_cursor = 0U;
  std::uint32_t outbound_remote_node = 0U;
  std::uint64_t outbound_wave = 0ULL;
  std::uint64_t outbound_wave_end = 0ULL;
  std::uint64_t outbound_node_packet_begin = 0ULL;
  bool local_complete_published = false;
  bool outbound_complete = false;
  bool outbound_complete_published = false;
  bool inbound_dy_terminal = false;
  bool inbound_return_terminal = false;
  bool outbound_return_terminal = false;

  // Relay is a cooperative state machine, not three global phases.  An
  // inbound wave has highest priority so received bytes immediately release
  // dV work.  One outbound wave and one local 128-route batch are then
  // produced per turn.  This preserves the SPSC ownership of each GIN context
  // while overlapping all three data paths without grid-wide barriers.
  while ((!local_complete_published || !outbound_complete_published ||
          !inbound_dy_terminal || !outbound_return_terminal ||
          !inbound_return_terminal) &&
         !backward_v3_generation_cancelled(a, control)) {
    bool progressed = false;

    if (!inbound_dy_terminal || !inbound_return_terminal) {
      if (threadIdx.x == 0U) {
        inbound_ready = persistent_v3::try_pop(
                            control.rdma_to_relay[context], inbound_task)
                            ? 1U
                            : 0U;
      }
      __syncthreads();
      if (inbound_ready != 0U) {
        progressed = true;
        const auto kind = static_cast<persistent_v3::TaskKind>(
            inbound_task.kind);
        if ((inbound_task.flags & persistent_v3::kPipelineTaskTerminal) !=
            0U) {
          if (kind == persistent_v3::TaskKind::DyRelay &&
              !inbound_dy_terminal) {
            inbound_dy_terminal = true;
            if (threadIdx.x == 0U) {
              // This completion is consumed by the diagnostic full-dY wait
              // from every GPU on the node.  Publish it at system scope only
              // after all peer payload stores from this Relay CTA are system
              // visible; a device-scope counter on the producer GPU cannot
              // order peer-global stores for a consumer on another GPU.
              __threadfence_system();
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  complete(control.inbound_dy_roles_complete.value);
              const auto completed =
                  complete.fetch_add(1ULL, ::cuda::memory_order_release) +
                  1ULL;
              if (completed == persistent_v3::kDefaultRelayCtaCount &&
                  persistent_v3::load_acquire<
                      ::cuda::thread_scope_device>(
                      control.local_route_roles_complete.value) ==
                      persistent_v3::kDefaultRelayCtaCount) {
                auto *status = peer_at<BackwardDeviceStatus>(
                    a.local_base, a.status_offset);
                status->dy_dispatch_complete_clock =
                    device_globaltimer_ns();
                MEGAMOE_TIMELINE_RECORD(a.timeline, DyDispatchComplete,
                                        Compute, context, completed, a.epoch);
              }
            }
          } else if (kind == persistent_v3::TaskKind::ReturnRelay &&
                     !inbound_return_terminal) {
            inbound_return_terminal = true;
            if (threadIdx.x == 0U)
              persistent_v3::fetch_add_release<
                  ::cuda::thread_scope_device>(
                  control.inbound_return_roles_complete.value, 1ULL);
          }
        } else if (kind == persistent_v3::TaskKind::DyRelay) {
          MEGAMOE_TIMELINE_RECORD(a.timeline, DyRelayStart, ProxyRelay,
                                  context, inbound_task.index, a.epoch);
          backward_v3_relay_inbound_wave(
              a, inbound_task, bulk_storage, bulk_stage_epochs);
          MEGAMOE_TIMELINE_RECORD(a.timeline, DyRelayEnd, ProxyRelay, context,
                                  inbound_task.index, a.epoch);
        } else if (kind == persistent_v3::TaskKind::ReturnRelay) {
          MEGAMOE_TIMELINE_RECORD(a.timeline, DxSourceRelay, SourceRelay,
                                  context, inbound_task.row_begin, a.epoch);
          backward_v3_scatter_return_slice(a, control, inbound_task);
          MEGAMOE_TIMELINE_RECORD(a.timeline, DGateReturn, SourceRelay,
                                  context, inbound_task.row_begin, a.epoch);
        } else if (threadIdx.x == 0U) {
          persistent_v3::publish_error_once(control.generation, 402ULL);
        }
      }
      __syncthreads();
    }

    // Gather one ready reverse chunk per turn.  Chunk ownership is static by
    // context, while producer counts arrive from all eight expert rails via
    // system release atomics.  A not-yet-ready chunk never head-of-line blocks
    // another chunk owned by this Relay CTA.
    if (threadIdx.x == 0U) {
      selected_return_chunk = ~0ULL;
      for (std::uint64_t attempt = 0;
           attempt < owned_return_chunks; ++attempt) {
        const auto local =
            owned_return_chunks == 0ULL
                ? 0ULL
                : (return_chunk_cursor++ % owned_return_chunks);
        const auto candidate =
            context + local * persistent_v3::kRdmaCtaCount;
        ::cuda::atomic_ref<unsigned long long,
                           ::cuda::thread_scope_device>
            state(forward_return_chunk_state[candidate]);
        if (state.load(::cuda::memory_order_acquire) !=
            kReturnChunkWaiting)
          continue;
        const auto chunk = forward_return_chunks[candidate];
        unsigned long long ready_rows = 0ULL;
        for (std::uint32_t producer_rail = 0;
             producer_rail < a.local_world_size; ++producer_rail) {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              ready(forward_return_chunk_ready_count[
                  candidate * a.local_world_size + producer_rail]);
          ready_rows += ready.load(::cuda::memory_order_acquire);
        }
        if (ready_rows != chunk.expected_fragments)
          continue;
        auto expected = kReturnChunkWaiting;
        if (state.compare_exchange_strong(
                expected, kReturnChunkPacking,
                ::cuda::memory_order_acq_rel,
                ::cuda::memory_order_acquire)) {
          selected_return_chunk = candidate;
          break;
        }
      }
    }
    __syncthreads();
    if (selected_return_chunk != ~0ULL) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, DxReturnPackStart, GinReturn,
                              context, selected_return_chunk, a.epoch);
      const auto return_chunk =
          forward_return_chunks[selected_return_chunk];
      if (!a.v3_direct_return_staging)
        pack_backward_return_chunk(a, return_chunk,
                                   forward_return_fragment_index);
      if (threadIdx.x == 0U) {
        // In direct mode the final producer count acquired above transitively
        // publishes every dX/dGate store already placed in this proxy's NCCL
        // window. The same chunk release then feeds the ordinary put-only GIN
        // path; no network get operation is introduced.
        persistent_v3::store_release<::cuda::thread_scope_device>(
            forward_return_chunk_state[selected_return_chunk],
            kReturnChunkComplete);
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.return_chunks_packed.value, 1ULL);
      }
      __syncthreads();
      ++return_chunks_packed_by_role;
      progressed = true;
    }

    // Packing stays at 128 rows so computation can release fragments early;
    // GIN submission is coalesced independently. A group is publishable only
    // after every overlapping catalog record has completed its system fence.
    // The coverage check makes a malformed/gapped directory fail closed
    // instead of putting uninitialised staging bytes.
    if (return_groups_issued < owned_return_groups) {
      const auto group =
          context + return_groups_issued * persistent_v3::kRdmaCtaCount;
      const auto row_begin = group * a.v3_return_group_rows;
      const auto row_end = min(row_begin + a.v3_return_group_rows,
                               return_compact_rows);
      if (threadIdx.x == 0U) {
        bool ready = row_begin < row_end;
        std::uint64_t covered_rows = 0ULL;
        std::uint64_t blocked_chunk = 0ULL;
        std::uint64_t blocked_ready = 0ULL;
        std::uint64_t blocked_expected = 0ULL;
        std::uint64_t blocked_state = 0ULL;
        for (std::uint64_t chunk_index = 0ULL;
             ready && chunk_index < return_chunk_count; ++chunk_index) {
          const auto chunk = forward_return_chunks[chunk_index];
          if (chunk.source_node != return_source_node)
            continue;
          const auto chunk_begin = chunk.row_begin;
          const auto chunk_end = chunk_begin + chunk.row_count;
          const auto overlap_begin = max(row_begin, chunk_begin);
          const auto overlap_end = min(row_end, chunk_end);
          if (overlap_begin >= overlap_end)
            continue;
          covered_rows += overlap_end - overlap_begin;
          const auto state =
              persistent_v3::load_acquire<::cuda::thread_scope_device>(
                  forward_return_chunk_state[chunk_index]);
          if (state != kReturnChunkComplete) {
            ready = false;
            blocked_chunk = chunk_index + 1ULL;
            blocked_expected = chunk.expected_fragments;
            blocked_state = state;
            for (std::uint32_t producer_rail = 0;
                 producer_rail < a.local_world_size; ++producer_rail) {
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  count(forward_return_chunk_ready_count[
                      chunk_index * a.local_world_size + producer_rail]);
              blocked_ready += count.load(::cuda::memory_order_acquire);
            }
          }
        }
        return_publish_ready =
            ready && covered_rows == row_end - row_begin ? 1U : 0U;
        const auto debug_wait =
            (return_chunks_packed_by_role & 0xffffULL) |
            ((blocked_chunk & 0xffffULL) << 16U) |
            ((blocked_ready & 0xffULL) << 32U) |
            ((blocked_expected & 0xffULL) << 40U) |
            ((blocked_state & 0xffULL) << 48U);
        if (debug_wait != return_debug_wait) {
          backward_v3_debug_progress(a, context, 2U, debug_wait);
          return_debug_wait = debug_wait;
        }
      }
      __syncthreads();
      if (return_publish_ready != 0U) {
        if (threadIdx.x == 0U) {
          const auto rows = row_end - row_begin;
          if (rows > UINT16_MAX) {
            persistent_v3::publish_error_once(control.generation, 403ULL);
          } else {
            persistent_v3::PipelineTask task{};
            task.epoch = a.epoch;
            task.index = static_cast<std::uint32_t>(group);
            task.expert = static_cast<std::uint16_t>(return_source_node);
            task.kind = static_cast<std::uint8_t>(
                persistent_v3::TaskKind::ReturnRelay);
            task.context = static_cast<std::uint8_t>(context);
            task.row_begin = static_cast<std::uint32_t>(row_begin);
            task.row_count = static_cast<std::uint16_t>(rows);
            while (!persistent_v3::try_push(control.relay_to_rdma[context],
                                            task) &&
                   !backward_v3_generation_cancelled(a, control))
              __nanosleep(64U);
          }
        }
        __syncthreads();
        ++return_groups_issued;
        progressed = true;
      }
    }

    if (!outbound_complete && outbound_wave == outbound_wave_end) {
      bool found_span = false;
      while (outbound_node_cursor < a.dispatch.identity.num_nodes) {
        const auto candidate_node = outbound_node_cursor++;
        if (candidate_node == a.node)
          continue;
        std::uint64_t node_wave_begin = 0ULL;
        std::uint64_t context_wave_begin = 0ULL;
        std::uint64_t context_wave_end = 0ULL;
        if (!backward_v3_context_wave_span(
                a, context, candidate_node, node_wave_begin,
                context_wave_begin, context_wave_end) ||
            context_wave_begin == context_wave_end)
          continue;
        outbound_remote_node = candidate_node;
        outbound_wave = context_wave_begin;
        outbound_wave_end = context_wave_end;
        outbound_node_packet_begin =
            a.dispatch_waves.offsets[node_wave_begin];
        found_span = true;
        break;
      }
      outbound_complete = !found_span;
    }

    if (!outbound_complete && outbound_wave < outbound_wave_end) {
      backward_v3_pack_outbound_wave(
          a, control, context, outbound_remote_node,
          outbound_node_packet_begin, outbound_wave++, bulk_storage,
          bulk_stage_epochs);
      progressed = true;
    }

    if (local_route_begin < local_route_count) {
      const auto route_end = min(
          local_route_begin +
              static_cast<std::uint64_t>(kBackwardV3LocalRouteBatch),
          local_route_count);
      backward_v3_publish_local_dy_batch(
          a, local_route_begin, route_end, relay_ordinal, bulk_storage,
          bulk_stage_epochs);
      local_route_begin += kLocalRouteStride;
      progressed = true;
    }

    if (!local_complete_published &&
        local_route_begin >= local_route_count) {
      if (threadIdx.x == 0U) {
        __threadfence_system();
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            complete(control.local_route_roles_complete.value);
        const auto completed =
            complete.fetch_add(1ULL, ::cuda::memory_order_release) + 1ULL;
        if (completed == persistent_v3::kDefaultRelayCtaCount) {
          auto *status =
              peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
          status->v3_local_delivery_complete_clock = device_globaltimer_ns();
          if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                  control.inbound_dy_roles_complete.value) ==
              persistent_v3::kDefaultRelayCtaCount) {
            status->dy_dispatch_complete_clock =
                status->v3_local_delivery_complete_clock;
            MEGAMOE_TIMELINE_RECORD(a.timeline, DyDispatchComplete, Compute,
                                    context, completed, a.epoch);
          }
        }
      }
      __syncthreads();
      local_complete_published = true;
      progressed = true;
    }

    if (!outbound_complete_published && outbound_complete) {
      if (threadIdx.x == 0U) {
        persistent_v3::PipelineTask terminal{};
        terminal.epoch = a.epoch;
        terminal.kind = static_cast<std::uint8_t>(
            persistent_v3::TaskKind::DispatchRelay);
        terminal.context = static_cast<std::uint8_t>(context);
        terminal.flags = persistent_v3::kPipelineTaskTerminal;
        while (!persistent_v3::try_push(control.relay_to_rdma[context],
                                        terminal) &&
               !backward_v3_generation_cancelled(a, control))
          __nanosleep(64U);
        const auto completed =
            persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                control.outbound_pack_roles_complete.value, 1ULL) +
            1ULL;
        if (completed == persistent_v3::kDefaultRelayCtaCount) {
          auto *status =
              peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
          status->v3_outbound_pack_complete_clock = device_globaltimer_ns();
        }
      }
      __syncthreads();
      outbound_complete_published = true;
      progressed = true;
    }

    // Chunk-pack and wire-group ownership are intentionally different. A
    // context may have issued all of its own groups while still owning a
    // 128-row chunk needed by another context's later group, so it must keep
    // servicing the pack catalog until both obligations are complete.
    if (!outbound_return_terminal &&
        return_chunks_packed_by_role == owned_return_chunks &&
        return_groups_issued == owned_return_groups) {
      if (threadIdx.x == 0U) {
        persistent_v3::PipelineTask terminal{};
        terminal.epoch = a.epoch;
        terminal.kind = static_cast<std::uint8_t>(
            persistent_v3::TaskKind::ReturnRelay);
        terminal.context = static_cast<std::uint8_t>(context);
        terminal.flags = persistent_v3::kPipelineTaskTerminal;
        while (!persistent_v3::try_push(control.relay_to_rdma[context],
                                        terminal) &&
               !backward_v3_generation_cancelled(a, control))
          __nanosleep(64U);
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.outbound_return_roles_complete.value, 1ULL);
      }
      __syncthreads();
      outbound_return_terminal = true;
      progressed = true;
    }

    if (!progressed && threadIdx.x == 0U)
      __nanosleep(64U);
    if (threadIdx.x == 0U) {
      const auto flags =
          static_cast<std::uint64_t>(local_complete_published) |
          (static_cast<std::uint64_t>(outbound_complete_published) << 1U) |
          (static_cast<std::uint64_t>(inbound_dy_terminal) << 2U) |
          (static_cast<std::uint64_t>(outbound_return_terminal) << 3U) |
          (static_cast<std::uint64_t>(inbound_return_terminal) << 4U);
      if (flags != return_debug_flags) {
        backward_v3_debug_progress(a, context, 4U, flags);
        return_debug_flags = flags;
      }
      const auto dx_complete =
          persistent_v3::load_acquire<::cuda::thread_scope_device>(
              control.dx_tasks_complete.value);
      const auto debug_compute =
          (dx_complete << 32U) | return_groups_issued;
      if (debug_compute != return_debug_compute) {
        backward_v3_debug_progress(a, context, 3U, debug_compute);
        return_debug_compute = debug_compute;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.relay_roles_complete.value, 1ULL);
    persistent_v3::wait_for_terminal(control.generation, a.epoch);
  }
  __syncthreads();
}

__device__ __forceinline__ void backward_v3_issue_rdma_task(
    const BackwardKernelArgs &a, const ncclGin &gin, ncclTeam rail,
    const persistent_v3::PipelineTask &task,
    unsigned long long &issued_completions) {
  const auto remote_node = static_cast<std::uint32_t>(task.expert);
  if ((task.flags & persistent_v3::kPipelineTaskDirectory) != 0U) {
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
          a.gin_receive_gate_offset +
              receive_packet * a.dispatch.identity.top_k * sizeof(float),
          a.gin_window,
          a.gin_send_gate_offset +
              static_cast<std::size_t>(task.index) *
                  a.dispatch.identity.top_k * sizeof(float),
          packets * a.dispatch.identity.top_k * sizeof(float));
  if (!a.v3_direct_dy_input) {
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
  }

  // NCCL GIN guarantees that a signal is visible only after this put and all
  // preceding puts on the same context/peer have settled.  A late 8-byte put
  // therefore turns source_epoch into an unambiguous per-wave ready marker,
  // even though signal 0 aggregates completions from several peers.
  gin.putValue<std::uint64_t>(
      rail, static_cast<int>(remote_node), a.gin_window,
      a.gin_receive_header_offset +
          receive_packet * sizeof(LocalP2pRouteHeader) +
          offsetof(LocalP2pRouteHeader, source_epoch),
      a.epoch, ncclGin_SignalInc{0}, ncclCoopThread{}, ncclGin_None{},
      ::cuda::thread_scope_block, ::cuda::thread_scope_device);

  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_dispatch_put_count,
            a.v3_direct_dy_input ? 4ULL : 5ULL);
  atomicAdd(&status->gin_signal_count, 1ULL);
  if (!a.v3_direct_dy_input) {
    atomicAdd(&status->gin_dispatch_payload_bytes,
              packets * a.hidden * sizeof(transport::BFloat16));
    atomicAdd(&status->raw_cross_node_dy_bytes,
              packets * a.hidden * sizeof(transport::BFloat16));
  }
  atomicAdd(&status->gin_metadata_bytes,
            packets * (sizeof(LocalP2pRouteHeader) +
                       a.dispatch.identity.top_k *
                           (sizeof(DeviceFanoutRecord) + sizeof(float))));
  MEGAMOE_TIMELINE_RECORD(a.timeline, DyGinPutIssue, GinDispatch,
                          task.context, packets, a.epoch);
}

// The producer-ready GIN plane is already token-major, so each RDMA CTA owns
// one contiguous token slice.  Local counter completion proves that every
// slice has settled remotely; a device-scope eight-CTA rendezvous then lets
// all contexts issue their ordered metadata markers.  This is a role-local
// dependency, not a grid barrier, and no Compute or Relay CTA participates.
__device__ __forceinline__ void backward_v3_issue_direct_dy_payload(
    const BackwardKernelArgs &a, const ncclGin &gin, ncclTeam rail,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t context, unsigned long long &issued_completions) {
  const auto token_begin =
      a.dispatch.identity.num_tokens * context /
      persistent_v3::kRdmaCtaCount;
  const auto token_end =
      a.dispatch.identity.num_tokens * (context + 1ULL) /
      persistent_v3::kRdmaCtaCount;
  const auto rows = token_end - token_begin;
  for (std::uint32_t remote_node = 0;
       remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
    if (remote_node == a.node || rows == 0ULL)
      continue;
    const auto source_partition =
        a.node < remote_node ? a.node : a.node - 1U;
    const auto receive_row =
        static_cast<std::uint64_t>(source_partition) *
            a.max_tokens_per_rank +
        token_begin;
    const auto payload_bytes =
        rows * a.hidden * sizeof(transport::BFloat16);
    gin.put(rail, static_cast<int>(remote_node), a.gin_window,
            a.gin_receive_payload_offset +
                receive_row * a.hidden * sizeof(transport::BFloat16),
            a.gin_window,
            a.gin_backward_dy_input_offset +
                token_begin * a.hidden * sizeof(transport::BFloat16),
            payload_bytes, ncclGin_None{}, ncclGin_CounterInc{0},
            ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_device);
    ++issued_completions;
    MEGAMOE_TIMELINE_RECORD(a.timeline, DyGinPutIssue, GinDispatch, context,
                            payload_bytes, a.epoch);
    auto *status =
        peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
    atomicAdd(&status->gin_dispatch_put_count, 1ULL);
    atomicAdd(&status->gin_dispatch_payload_bytes, payload_bytes);
    atomicAdd(&status->raw_cross_node_dy_bytes, payload_bytes);
  }
  if (issued_completions != 0ULL)
    gin.waitCounter(ncclCoopThread{}, 0, issued_completions);
  MEGAMOE_TIMELINE_RECORD(a.timeline, DyLocalCompletion, GinDispatch,
                          context, issued_completions, a.epoch);
  persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
      control.direct_dy_payload_roles_complete.value, 1ULL);
  while (!backward_v3_generation_cancelled(a, control) &&
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.direct_dy_payload_roles_complete.value) !=
             persistent_v3::kRdmaCtaCount)
    __nanosleep(64U);
}

__device__ __forceinline__ void backward_v3_issue_return_rdma_task(
    const BackwardKernelArgs &a, const ncclGin &gin, ncclTeam rail,
    const persistent_v3::PipelineTask &task,
    unsigned long long &issued_completions) {
  const auto source_node = static_cast<std::uint32_t>(task.expert);
  const auto source_partition =
      source_node < a.node ? source_node : source_node - 1U;
  const auto partition_base =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank *
      a.dispatch.identity.top_k;
  const auto packed_base = partition_base + task.row_begin;
  const auto rows = static_cast<std::uint64_t>(task.row_count);
  const auto payload_bytes =
      rows * a.hidden * sizeof(transport::BFloat16);

  // Signal 1 is a completion bitset, not a count. GIN operations from one
  // context may complete out of issue order, so a cumulative +1 cannot identify
  // which Return group is actually safe to consume. Each context has at most
  // 64 groups and assigns one immutable bit to every issued task. The signal
  // action is ordered after this payload and all preceding puts to the peer.
  if (issued_completions >= 64ULL) {
    auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
        a.local_base, a.v3_control_offset);
    persistent_v3::publish_error_once(control->generation, 408ULL);
    return;
  }
  const auto completion_bit = 1ULL << issued_completions;
  // Put the small gate plane first. The completion action belongs to the
  // large dX transfer itself, so observing its bit proves both this payload
  // and the preceding gate bytes have settled without relying on cross-WQE
  // completion ordering in the GIN backend.
  gin.put(rail, static_cast<int>(source_node), a.gin_window,
          a.gin_return_gate_receive_offset +
              static_cast<std::size_t>(task.row_begin) * sizeof(float),
          a.gin_window,
          a.gin_return_gate_staging_offset + packed_base * sizeof(float),
          rows * sizeof(float));
  gin.put(rail, static_cast<int>(source_node), a.gin_window,
          a.gin_return_receive_offset +
              static_cast<std::size_t>(task.row_begin) * a.hidden *
                  sizeof(transport::BFloat16),
          a.gin_window,
          a.gin_return_staging_offset +
              packed_base * a.hidden * sizeof(transport::BFloat16),
          payload_bytes, ncclGin_SignalAdd{1, completion_bit},
          ncclGin_CounterInc{1}, ncclCoopThread{}, ncclGin_None{},
          ::cuda::thread_scope_block, ::cuda::thread_scope_device);
  MEGAMOE_TIMELINE_RECORD(a.timeline, DxReturnGinPutIssue, GinReturn,
                          task.context, payload_bytes, a.epoch);
  ++issued_completions;

  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_return_put_count, 2ULL);
  atomicAdd(&status->gin_signal_count, 1ULL);
  atomicAdd(&status->gin_return_payload_bytes, payload_bytes);
  atomicAdd(&status->gin_metadata_bytes, rows * sizeof(float));
}

__device__ __forceinline__ std::uint64_t
backward_v3_inbound_return_tasks(const BackwardKernelArgs &a,
                                 std::uint32_t context) {
  std::uint64_t tasks = 0ULL;
  for (std::uint32_t remote_node = 0;
       remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
    if (remote_node == a.node)
      continue;
    const auto rows = backward_v3_outgoing_remote_fanout(a, remote_node);
    const auto groups = (rows + a.v3_return_group_rows - 1ULL) /
                        a.v3_return_group_rows;
    tasks += groups > context
                 ? 1ULL + (groups - 1ULL - context) /
                              persistent_v3::kRdmaCtaCount
                 : 0ULL;
  }
  return tasks;
}

__device__ __forceinline__ bool backward_v3_publish_one_inbound_return(
    const BackwardKernelArgs &a, const ncclGin &gin,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t context, unsigned long long &published_mask,
    unsigned long long &published_count) {
  const auto completed_mask =
      gin.readSignal(1, 64, ::cuda::memory_order_acquire);
  const auto available = completed_mask & ~published_mask;
  if (available == 0ULL)
    return false;
  const auto completed_ordinal = static_cast<std::uint64_t>(
      __ffsll(static_cast<long long>(available)) - 1);
  std::uint64_t ordinal_base = 0ULL;
  for (std::uint32_t remote_node = 0;
       remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
    if (remote_node == a.node)
      continue;
    const auto rows = backward_v3_outgoing_remote_fanout(a, remote_node);
    const auto groups = (rows + a.v3_return_group_rows - 1ULL) /
                        a.v3_return_group_rows;
    const auto context_groups =
        groups > context
            ? 1ULL + (groups - 1ULL - context) /
                         persistent_v3::kRdmaCtaCount
            : 0ULL;
    if (completed_ordinal >= ordinal_base + context_groups) {
      ordinal_base += context_groups;
      continue;
    }
    const auto local_group = completed_ordinal - ordinal_base;
    const auto group =
        context + local_group * persistent_v3::kRdmaCtaCount;
    const auto row_begin = group * a.v3_return_group_rows;
    const auto candidate_end = row_begin + a.v3_return_group_rows;
    const auto row_end =
        static_cast<unsigned long long>(rows) < candidate_end
            ? static_cast<unsigned long long>(rows)
            : candidate_end;
    if (row_end - row_begin > UINT16_MAX) {
      persistent_v3::publish_error_once(control.generation, 404ULL);
      return true;
    }
    persistent_v3::PipelineTask task{};
    task.epoch = a.epoch;
    task.expert = static_cast<std::uint16_t>(remote_node);
    task.kind =
        static_cast<std::uint8_t>(persistent_v3::TaskKind::ReturnRelay);
    task.context = static_cast<std::uint8_t>(context);
    task.row_begin = static_cast<std::uint32_t>(row_begin);
    task.row_count = static_cast<std::uint16_t>(row_end - row_begin);
    if (!persistent_v3::try_push(control.rdma_to_relay[context], task))
      return false;
    published_mask |= 1ULL << completed_ordinal;
    ++published_count;
    return true;
  }
  persistent_v3::publish_error_once(control.generation, 405ULL);
  return true;
}

__device__ __forceinline__ bool backward_v3_publish_one_inbound_wave(
    const BackwardKernelArgs &a, const ncclGin &gin,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t context, unsigned long long &directories_seen,
    unsigned long long &waves_expected,
    unsigned long long &waves_published) {
  const auto signal_value =
      gin.readSignal(0, 64, ::cuda::memory_order_acquire);
  if (signal_value <= directories_seen + waves_published)
    return false;

  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *directories = reinterpret_cast<GinDispatchContextRecord *>(
      bytes + a.gin_receive_dispatch_directory_offset);
  auto *headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  constexpr std::uint64_t kSeenBit = 1ULL << 63U;
  for (std::uint32_t signal_index = 0;
       signal_index < a.dispatch.sizes.incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node || signal.expected_packets == 0ULL)
      continue;
    const auto source_partition =
        signal.source_node < a.node ? signal.source_node
                                    : signal.source_node - 1U;
    auto &directory = directories[
        source_partition * persistent_v3::kRdmaCtaCount + context];
    const auto directory_generation = directory.generation & ~kSeenBit;
    if (directory_generation != a.dispatch.identity.generation ||
        directory.epoch != a.epoch)
      continue;
    if ((directory.generation & kSeenBit) == 0ULL) {
      waves_expected += directory.wave_count;
      ++directories_seen;
      directory.generation |= kSeenBit;
      return true;
    }
    if (directory.wave_count == 0U)
      continue;

    const auto packet =
        static_cast<std::uint64_t>(source_partition) *
            a.max_tokens_per_rank +
        directory.packet_begin;
    ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> marker(
        headers[packet].source_epoch);
    if (marker.load(::cuda::memory_order_acquire) != a.epoch)
      continue;
    const auto wave_packets = headers[packet].topk_slot;
    if (wave_packets == 0U || wave_packets > directory.packet_count ||
        wave_packets > UINT16_MAX) {
      persistent_v3::publish_error_once(control.generation, 401ULL);
      return true;
    }
    persistent_v3::PipelineTask task{};
    task.epoch = a.epoch;
    task.index = static_cast<std::uint32_t>(packet);
    task.expert = static_cast<std::uint16_t>(source_partition);
    task.kind =
        static_cast<std::uint8_t>(persistent_v3::TaskKind::DyRelay);
    task.context = static_cast<std::uint8_t>(context);
    task.row_begin = static_cast<std::uint32_t>(directory.packet_begin);
    task.row_count = static_cast<std::uint16_t>(wave_packets);
    if (!persistent_v3::try_push(control.rdma_to_relay[context], task))
      return false;
    directory.packet_begin += wave_packets;
    directory.packet_count -= wave_packets;
    --directory.wave_count;
    ++waves_published;
    return true;
  }
  return false;
}

__device__ __forceinline__ void backward_v3_rdma_role(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t context) {
  if (threadIdx.x == 0U) {
    const ncclGin gin(a.gin_dev_comm, context);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    unsigned long long incoming_directories_expected = 0ULL;
    for (std::uint32_t index = 0;
         index < a.dispatch.sizes.incoming_signal_count; ++index) {
      const auto signal = a.dispatch.incoming_proxy_signals[index];
      if (signal.source_node != a.node && signal.expected_packets != 0ULL)
        ++incoming_directories_expected;
    }

    const auto inbound_returns_expected =
        backward_v3_inbound_return_tasks(a, context);
    backward_v3_debug_progress(a, context, 7U,
                               inbound_returns_expected << 32U);
    unsigned long long dispatch_completions_issued = 0ULL;
    unsigned long long return_completions_issued = 0ULL;
    unsigned long long directories_seen = 0ULL;
    unsigned long long waves_expected = 0ULL;
    unsigned long long waves_published = 0ULL;
    unsigned long long inbound_returns_published_mask = 0ULL;
    unsigned long long inbound_returns_published = 0ULL;
#if MEGAMOE_DEVICE_TIMELINE
    unsigned long long timeline_dispatch_complete = 0ULL;
    unsigned long long timeline_return_complete = 0ULL;
#endif
    unsigned long long debug_inbound_returns_published = ~0ULL;
    unsigned long long debug_rdma_state = ~0ULL;
    bool outbound_dispatch_terminal = false;
    bool outbound_return_terminal = false;
    bool inbound_dispatch_terminal = false;
    bool inbound_return_terminal = false;
    if (a.v3_direct_dy_input)
      backward_v3_issue_direct_dy_payload(
          a, gin, rail, control, context, dispatch_completions_issued);
#if MEGAMOE_DEVICE_TIMELINE
    timeline_dispatch_complete = dispatch_completions_issued;
#endif
    while (!backward_v3_generation_cancelled(a, control)) {
      bool progressed = false;
      if (!outbound_dispatch_terminal || !outbound_return_terminal) {
        persistent_v3::PipelineTask outbound{};
        if (persistent_v3::try_pop(control.relay_to_rdma[context],
                                   outbound)) {
          progressed = true;
          const auto kind =
              static_cast<persistent_v3::TaskKind>(outbound.kind);
          if ((outbound.flags & persistent_v3::kPipelineTaskTerminal) != 0U) {
            if (kind == persistent_v3::TaskKind::DispatchRelay)
              outbound_dispatch_terminal = true;
            else if (kind == persistent_v3::TaskKind::ReturnRelay)
              outbound_return_terminal = true;
            else
              persistent_v3::publish_error_once(control.generation, 406ULL);
          } else if (kind == persistent_v3::TaskKind::DispatchRelay) {
            backward_v3_issue_rdma_task(a, gin, rail, outbound,
                                        dispatch_completions_issued);
          } else if (kind == persistent_v3::TaskKind::ReturnRelay) {
            backward_v3_issue_return_rdma_task(
                a, gin, rail, outbound, return_completions_issued);
            backward_v3_debug_progress(a, context, 5U,
                                       return_completions_issued);
          } else {
            persistent_v3::publish_error_once(control.generation, 407ULL);
          }
        }
      }

      if (!inbound_dispatch_terminal)
        progressed = backward_v3_publish_one_inbound_wave(
                         a, gin, control, context, directories_seen,
                         waves_expected, waves_published) ||
                     progressed;
      if (!inbound_dispatch_terminal &&
          directories_seen == incoming_directories_expected &&
          waves_published == waves_expected) {
        persistent_v3::PipelineTask terminal{};
        terminal.epoch = a.epoch;
        terminal.kind =
            static_cast<std::uint8_t>(persistent_v3::TaskKind::DyRelay);
        terminal.context = static_cast<std::uint8_t>(context);
        terminal.flags = persistent_v3::kPipelineTaskTerminal;
        if (persistent_v3::try_push(control.rdma_to_relay[context],
                                    terminal)) {
          inbound_dispatch_terminal = true;
          progressed = true;
        }
      }

      if (!inbound_return_terminal)
        progressed = backward_v3_publish_one_inbound_return(
                         a, gin, control, context,
                         inbound_returns_published_mask,
                         inbound_returns_published) ||
                     progressed;
      if (inbound_returns_published != debug_inbound_returns_published) {
        backward_v3_debug_progress(a, context, 6U,
                                   inbound_returns_published);
        debug_inbound_returns_published = inbound_returns_published;
      }
      if (!inbound_return_terminal &&
          inbound_returns_published == inbound_returns_expected) {
        persistent_v3::PipelineTask terminal{};
        terminal.epoch = a.epoch;
        terminal.kind = static_cast<std::uint8_t>(
            persistent_v3::TaskKind::ReturnRelay);
        terminal.context = static_cast<std::uint8_t>(context);
        terminal.flags = persistent_v3::kPipelineTaskTerminal;
        if (persistent_v3::try_push(control.rdma_to_relay[context],
                                    terminal)) {
          inbound_return_terminal = true;
          progressed = true;
        }
      }

      const auto dispatch_complete =
          gin.readCounter(0, 56, ::cuda::memory_order_acquire);
      const auto return_complete =
          gin.readCounter(1, 56, ::cuda::memory_order_acquire);
#if MEGAMOE_DEVICE_TIMELINE
      if (dispatch_complete != timeline_dispatch_complete) {
        MEGAMOE_TIMELINE_RECORD(a.timeline, DyLocalCompletion, GinDispatch,
                                context, dispatch_complete, a.epoch);
        timeline_dispatch_complete = dispatch_complete;
      }
      if (return_complete != timeline_return_complete) {
        MEGAMOE_TIMELINE_RECORD(a.timeline, DxReturnLocalCompletion,
                                GinReturn, context, return_complete, a.epoch);
        timeline_return_complete = return_complete;
      }
#endif
      const auto flags =
          static_cast<std::uint64_t>(outbound_dispatch_terminal) |
          (static_cast<std::uint64_t>(outbound_return_terminal) << 1U) |
          (static_cast<std::uint64_t>(inbound_dispatch_terminal) << 2U) |
          (static_cast<std::uint64_t>(inbound_return_terminal) << 3U);
      const auto rdma_state =
          (inbound_returns_expected << 32U) |
          ((return_complete & 0xffffULL) << 16U) | flags;
      if (rdma_state != debug_rdma_state) {
        backward_v3_debug_progress(a, context, 7U, rdma_state);
        debug_rdma_state = rdma_state;
      }
      if (outbound_dispatch_terminal && outbound_return_terminal &&
          inbound_dispatch_terminal && inbound_return_terminal &&
          dispatch_complete >= dispatch_completions_issued &&
          return_complete >= return_completions_issued)
        break;
      if (!progressed)
        __nanosleep(64U);
    }

    const auto completed =
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.rdma_roles_complete.value, 1ULL) +
        1ULL;
    if (completed == persistent_v3::kRdmaCtaCount) {
      auto *status =
          peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
      status->v3_rdma_complete_clock = device_globaltimer_ns();
    }
    persistent_v3::wait_for_terminal(control.generation, a.epoch);
  }
  __syncthreads();
}

#endif // __has_include(<nccl_device.h>)
