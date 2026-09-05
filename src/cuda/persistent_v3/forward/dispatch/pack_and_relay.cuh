
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
      source_compact_index[fanout] = target.return_compact_row;
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
  auto *receive_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
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
                                  a.epoch,
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
          arrived[target.mtile * a.local_world_size + rail]);
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
              a, scheduler, local_return_state)) {
        local_return_done = 1U;
        MEGAMOE_TIMELINE_RECORD(a.timeline, LocalReturnTerminal, LocalReturn,
                                context, local_return_state.cursor, a.epoch);
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
              a, scheduler, local_return_state)) {
        local_return_done = 1U;
        MEGAMOE_TIMELINE_RECORD(a.timeline, LocalReturnTerminal, LocalReturn,
                                context, local_return_state.cursor, a.epoch);
      }
      __syncthreads();
    }
#endif

    if (owns_context_transport && completion_done == 0U) {
      for (std::uint32_t pass = 0U;
           pass < forward_tuning::kCompletionChunksPerPass; ++pass) {
        const auto completion_progressed =
            forward_return_detail::progress_direct_full_completion_once(
                a, control, context, scheduler, completion_state);
        progressed = completion_progressed || progressed;
        if (forward_return_detail::direct_full_completion_terminal(
                completion_state))
          break;
      }
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
      __nanosleep(forward_tuning::kTransportIdleNanosleep);
    __syncthreads();
  }
}

