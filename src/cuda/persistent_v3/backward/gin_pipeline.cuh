#pragma once

// The Backward network path deliberately reuses Forward's immutable packet,
// fanout, canonical-header, and FULL-return maps.  Dispatch therefore sends
// only dY payloads over the rail team; Return sends dX plus its FP32 dGate
// companion over the independent FULL contexts.  No wrapping descriptor ring
// or grid-wide rendezvous is needed.

#if __has_include(<nccl_device.h>)

namespace backward_gin_detail {

using backward_pipeline_detail::BackwardMblockDescriptor;
using backward_pipeline_detail::PaddedCounter;

inline constexpr std::uint32_t kReturnChunkRows = 128U;
inline constexpr unsigned long long kMaxInflight = 24ULL;

struct RelayInboundState {
  unsigned long long packet_cursor = 0ULL;
  unsigned long long packet_end = 0ULL;
  std::uint32_t wave_remaining = 0U;
  std::uint32_t current_fanout = 0U;
  std::uint32_t source_partition = 0U;
  std::uint32_t progressed = 0U;
};

struct SourceCompletionState {
  unsigned long long compact_rows = 0ULL;
  unsigned long long chunk_count = 0ULL;
  unsigned long long scan_cursor = 0ULL;
  unsigned long long completed = 0ULL;
  unsigned long long expected = 0ULL;
  std::uint32_t context = 0U;
  std::uint32_t selected_rows = 0U;
  std::uint32_t selected = 0U;
  std::uint32_t progressed = 0U;
};

__device__ __forceinline__ std::uint32_t remote_node(
    const BackwardKernelArgs &a) {
  return a.node == 0U ? 1U : 0U;
}

__device__ __forceinline__ std::uint32_t remote_partition(
    const BackwardKernelArgs &a) {
  const auto remote = remote_node(a);
  return remote < a.node ? remote : remote - 1U;
}

__device__ __forceinline__ bool context_wave_span(
    const BackwardKernelArgs &a, std::uint32_t context,
    std::uint64_t &node_wave_begin, std::uint64_t &context_wave_begin,
    std::uint64_t &context_wave_end) {
  const auto remote = remote_node(a);
  node_wave_begin = 0ULL;
  while (node_wave_begin < a.dispatch_waves.wave_count &&
         a.dispatch_waves.destination_node[node_wave_begin] < remote)
    ++node_wave_begin;
  auto node_wave_end = node_wave_begin;
  while (node_wave_end < a.dispatch_waves.wave_count &&
         a.dispatch_waves.destination_node[node_wave_end] == remote)
    ++node_wave_end;
  const auto node_waves = node_wave_end - node_wave_begin;
  context_wave_begin =
      node_wave_begin + node_waves * context / a.gin_context_count;
  context_wave_end = node_wave_begin +
                     node_waves * (context + 1ULL) / a.gin_context_count;
  return node_wave_begin != node_wave_end;
}

__device__ __forceinline__ void initialize_relay_state(
    const BackwardKernelArgs &a, std::uint32_t context,
    RelayInboundState &inbound, SourceCompletionState &completion) {
  if (threadIdx.x == 0U) {
    const auto partition = remote_partition(a);
    const auto *receive_directories =
        reinterpret_cast<const GinDispatchContextRecord *>(
            static_cast<const std::byte *>(a.gin_window_base) +
            a.gin_receive_dispatch_directory_offset);
    const auto &receive =
        receive_directories[static_cast<std::uint64_t>(partition) *
                                a.gin_context_count +
                            context];
    inbound = RelayInboundState{};
    inbound.packet_cursor = receive.packet_begin;
    inbound.packet_end = receive.packet_begin + receive.packet_count;
    inbound.source_partition = partition;

    const auto *send_directories =
        reinterpret_cast<const GinDispatchContextRecord *>(
            static_cast<const std::byte *>(a.gin_window_base) +
            a.gin_send_dispatch_directory_offset);
    completion = SourceCompletionState{};
    completion.compact_rows =
        send_directories[static_cast<std::uint64_t>(partition) *
                             a.gin_context_count +
                         context]
            .return_row_count;
    completion.chunk_count =
        (completion.compact_rows + kReturnChunkRows - 1ULL) /
        kReturnChunkRows;
    completion.scan_cursor = context;
    completion.context = context;
    completion.expected =
        completion.chunk_count > context
            ? 1ULL + (completion.chunk_count - 1ULL - context) /
                         a.gin_context_count
            : 0ULL;
  }
  __syncthreads();
}

__device__ __forceinline__ bool inbound_terminal(
    const RelayInboundState &state) {
  return state.packet_cursor == state.packet_end &&
         state.wave_remaining == 0U;
}

__device__ __forceinline__ bool source_completion_terminal(
    const SourceCompletionState &state) {
  return state.completed >= state.expected;
}

// Consume one remotely arrived packet.  The first header in each Forward wave
// is reused as the ordered BWD payload marker; all packet/fanout identity stays
// immutable across repeated Backward executions.
__device__ __forceinline__ bool progress_inbound_once(
    const BackwardKernelArgs &a, RelayInboundState &state) {
  if (threadIdx.x == 0U)
    state.progressed = 0U;
  __syncthreads();
  if (inbound_terminal(state))
    return false;

  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  const auto *fanout = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  const auto *payload = reinterpret_cast<const transport::BFloat16 *>(
      bytes + a.gin_receive_payload_offset);
  const auto partition_packet_base =
      static_cast<std::uint64_t>(state.source_partition) *
      a.max_tokens_per_rank;
  const auto partition_fanout_base =
      partition_packet_base * a.dispatch.identity.top_k;

  if (threadIdx.x == 0U && state.wave_remaining == 0U) {
    const auto packet = partition_packet_base + state.packet_cursor;
    ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> marker(
        headers[packet].source_epoch);
    if (marker.load(::cuda::memory_order_acquire) == a.epoch) {
      asm volatile("fence.proxy.async.global;" : : : "memory");
      const auto wave_packets = headers[packet].topk_slot;
      if (wave_packets != 0U &&
          state.packet_cursor + wave_packets <= state.packet_end)
        state.wave_remaining = wave_packets;
    }
  }
  __syncthreads();
  if (state.wave_remaining == 0U)
    return false;

  const auto packet = partition_packet_base + state.packet_cursor;
  const auto header = headers[packet];
  if (threadIdx.x == 0U)
    state.current_fanout = static_cast<std::uint32_t>(header.route_index);
  __syncthreads();
  const auto vector_count = a.hidden / 8ULL;
  const auto items = static_cast<std::uint64_t>(state.current_fanout) *
                     vector_count;
  for (std::uint64_t item = threadIdx.x; item < items;
       item += blockDim.x) {
    const auto fanout_slot = item / vector_count;
    const auto vector = item % vector_count;
    const auto target =
        fanout[partition_fanout_base + header.dst_row + fanout_slot];
    const auto destination =
        a.node * a.local_world_size + target.destination_local_rank;
    const auto *compact_to_padded = peer_at<const unsigned int>(
        a.peer_bases[destination], a.compact_to_padded_offset);
    const auto destination_row = compact_to_padded[target.dst_row];
    auto *destination_dr = peer_at<transport::BFloat16>(
        a.peer_bases[destination], a.padded_dr_offset);
    reinterpret_cast<uint4 *>(destination_dr)[
        static_cast<std::uint64_t>(destination_row) * vector_count + vector] =
        reinterpret_cast<const uint4 *>(payload)[packet * vector_count +
                                                  vector];
  }
  __threadfence_system();
  __syncthreads();

  for (std::uint32_t fanout_slot = threadIdx.x;
       fanout_slot < state.current_fanout; fanout_slot += blockDim.x) {
    const auto target =
        fanout[partition_fanout_base + header.dst_row + fanout_slot];
    const auto destination =
        a.node * a.local_world_size + target.destination_local_rank;
    const auto *compact_to_padded = peer_at<const unsigned int>(
        a.peer_bases[destination], a.compact_to_padded_offset);
    const auto destination_row = compact_to_padded[target.dst_row];
    auto *peer_words = peer_at<unsigned long long>(
        a.peer_bases[destination], a.scheduler_offset);
    const auto scheduler_layout =
        megamoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
            a.scheduler_mblock_capacity,
            static_cast<std::uint32_t>(a.hidden /
                                       backward_tuning::kDxTileN),
            a.scheduler_expert_capacity);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> arrived(
        peer_words[scheduler_layout.dispatch_arrived +
                   destination_row / backward_tuning::kTileM]);
    arrived.fetch_add(1ULL, ::cuda::memory_order_release);
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    ++state.packet_cursor;
    --state.wave_remaining;
    state.progressed = 1U;
  }
  __syncthreads();
  return state.progressed != 0U;
}

// Publish one complete FULL-return chunk.  dX stays in the registered compact
// receive plane and is read there by Reduce; only the tiny dGate scalar is
// copied into the public route-order output.
__device__ __forceinline__ bool progress_source_completion_once(
    const BackwardKernelArgs &a, SourceCompletionState &state) {
  if (threadIdx.x == 0U) {
    state.progressed = 0U;
    state.selected = 0U;
  }
  __syncthreads();
  if (source_completion_terminal(state) || state.chunk_count == 0ULL)
    return false;

  if (threadIdx.x == 0U) {
    const auto chunk = state.scan_cursor;
    if (chunk < state.chunk_count) {
      const auto row_begin = chunk * kReturnChunkRows;
      const auto remaining = state.compact_rows - row_begin;
      state.selected_rows = static_cast<std::uint32_t>(
          remaining < kReturnChunkRows ? remaining : kReturnChunkRows);

      asm volatile("fence.proxy.async.global;" : : : "memory");
      const auto *markers = reinterpret_cast<const unsigned long long *>(
          static_cast<const std::byte *>(a.gin_window_base) +
          a.gin_return_fragment_ready_offset);
      unsigned long long ready_rows = 0ULL;
      for (std::uint32_t row = 0U; row < state.selected_rows; ++row)
        ready_rows += markers[row_begin + row];
      if (ready_rows >= state.selected_rows) {
        state.selected = static_cast<std::uint32_t>(chunk + 1ULL);
      }
    }
  }
  __syncthreads();
  if (state.selected == 0U)
    return false;

  const auto chunk = static_cast<unsigned long long>(state.selected - 1U);
  const auto row_begin = chunk * kReturnChunkRows;
  asm volatile("fence.proxy.async.global;" : : : "memory");
  const auto *route_index = peer_at<const unsigned long long>(
      a.local_base, a.forward_return_source_route_index_offset);
  const auto *gate_receive = reinterpret_cast<const float *>(
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_return_gate_receive_offset);
  auto *gate_output =
      peer_at<float>(a.local_base, a.dgate_output_offset);
  auto *route_ready = peer_at<unsigned long long>(
      a.local_base, a.return_ready_offset);
  for (std::uint32_t local = threadIdx.x; local < state.selected_rows;
       local += blockDim.x) {
    const auto compact = row_begin + local;
    const auto route = route_index[compact];
    gate_output[route] = gate_receive[compact];
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        route_ready[route]);
    ready.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    ++state.completed;
    state.scan_cursor += a.gin_context_count;
    state.progressed = 1U;
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->return_complete_clock, device_globaltimer_ns());
  }
  __syncthreads();
  return state.progressed != 0U;
}

struct RdmaSharedState {
  unsigned long long issued_dispatch = 0ULL;
  unsigned long long completed_dispatch = 0ULL;
  unsigned long long issued_return = 0ULL;
  unsigned long long completed_return = 0ULL;
  std::uint64_t node_packet_begin = 0ULL;
  std::uint64_t wave_begin = 0ULL;
  std::uint64_t wave_end = 0ULL;
  std::uint32_t mblock = 0U;
  std::uint32_t row_cursor = 0U;
  std::uint32_t run_canonical = 0U;
  std::uint32_t run_compact = 0U;
  std::uint32_t run_source = 0U;
  std::uint32_t run_rows = 0U;
};

__device__ __forceinline__ void rdma_role(const BackwardKernelArgs &a) {
  const auto context = static_cast<std::uint32_t>(blockIdx.x);
  __shared__ RdmaSharedState state;
  if (threadIdx.x == 0U) {
    state = RdmaSharedState{};
    context_wave_span(a, context, state.node_packet_begin, state.wave_begin,
                      state.wave_end);
    const ncclGin gin(a.gin_dev_comm, context);
    const ncclGin return_gin(a.gin_dev_comm,
                             context + a.gin_context_count);
    gin.resetCounter(0);
    return_gin.resetCounter(1);
  }
  __syncthreads();

  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_send_header_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_send_payload_offset);
  const auto vector_count = a.hidden / 8ULL;
  const auto remote = remote_node(a);
  const auto partition = remote_partition(a);
  const auto node_packet_base = a.dispatch_waves.offsets[state.node_packet_begin];

  // RAIL Dispatch: retain Forward's packet order and copy only dY.  One ordered
  // eight-byte header put closes each wave, so the receiver needs no signal
  // reset or descriptor queue.
  for (auto wave = state.wave_begin; wave < state.wave_end; ++wave) {
    const auto wave_begin = a.dispatch_waves.offsets[wave];
    const auto wave_end = a.dispatch_waves.offsets[wave + 1ULL];
    const auto packets = wave_end - wave_begin;
    for (std::uint64_t item = threadIdx.x;
         item < packets * vector_count; item += blockDim.x) {
      const auto packet_in_wave = item / vector_count;
      const auto vector = item % vector_count;
      const auto position = wave_begin + packet_in_wave;
      const auto packet = a.dispatch_waves.packet_order[position];
      const auto token = a.dispatch.packet_headers[packet].local_token_id;
      reinterpret_cast<uint4 *>(send_payload)[position * vector_count +
                                                vector] =
          reinterpret_cast<const uint4 *>(a.dy)[token * vector_count + vector];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U && packets != 0ULL) {
      const auto receive_packet =
          static_cast<std::uint64_t>(partition) * a.max_tokens_per_rank +
          wave_begin - node_packet_base;
      send_headers[wave_begin].source_epoch = a.epoch;
      __threadfence_system();
      const ncclGin gin(a.gin_dev_comm, context);
      const auto rail = ncclTeamRail(a.gin_dev_comm);
      gin.put(rail, static_cast<int>(remote), a.gin_window,
              a.gin_receive_payload_offset +
                  receive_packet * a.hidden *
                      sizeof(transport::BFloat16),
              a.gin_window,
              a.gin_send_payload_offset +
                  wave_begin * a.hidden * sizeof(transport::BFloat16),
              packets * a.hidden * sizeof(transport::BFloat16));
      gin.put(rail, static_cast<int>(remote), a.gin_window,
              a.gin_receive_header_offset +
                  receive_packet * sizeof(LocalP2pRouteHeader) +
                  offsetof(LocalP2pRouteHeader, source_epoch),
              a.gin_window,
              a.gin_send_header_offset +
                  wave_begin * sizeof(LocalP2pRouteHeader) +
                  offsetof(LocalP2pRouteHeader, source_epoch),
              sizeof(std::uint64_t), ncclGin_None{}, ncclGin_CounterInc{0},
              ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_block,
              ::cuda::thread_scope_device);
      ++state.issued_dispatch;
      if (state.issued_dispatch - state.completed_dispatch >= kMaxInflight) {
        gin.waitCounter(ncclCoopThread{}, 0, state.issued_dispatch);
        state.completed_dispatch = state.issued_dispatch;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    const ncclGin gin(a.gin_dev_comm, context);
    if (state.issued_dispatch != state.completed_dispatch)
      gin.waitCounter(ncclCoopThread{}, 0, state.issued_dispatch);
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
  }
  __syncthreads();

  auto *words = backward_pipeline_detail::scheduler_words(a);
  const auto scheduler_layout =
      megamoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity,
          static_cast<std::uint32_t>(a.hidden / backward_tuning::kDxTileN),
          a.scheduler_expert_capacity);
  const auto *descriptors = reinterpret_cast<const BackwardMblockDescriptor *>(
      words + scheduler_layout.descriptors);
  const auto *dx = peer_at<const transport::BFloat16>(
      a.local_base, a.dx_rows_offset);
  const auto *dgate =
      peer_at<const float>(a.local_base, a.dgate_rows_offset);
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  auto *mapping = peer_at<unsigned long long>(
      a.local_base, a.forward_output_ready_epoch_offset);
  auto *return_staging = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_return_staging_offset);
  auto *gate_staging = reinterpret_cast<float *>(
      bytes + a.gin_return_gate_staging_offset);

  // FULL Return: each context owns exactly the remote source local-rank with
  // the same ordinal.  It independently scans completed M128 blocks, packs
  // only its rows into the registered canonical plane, then emits coalesced
  // compact runs.  Compute and the other seven contexts never wait on it.
  while (state.mblock < a.mblocks) {
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          completed(words[scheduler_layout.dx_completed + state.mblock]);
      const auto expected =
          static_cast<unsigned long long>(a.hidden /
                                          backward_tuning::kDxTileN);
      while (completed.load(::cuda::memory_order_acquire) != expected)
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
      state.row_cursor = 0U;
    }
    __syncthreads();
    const auto descriptor = descriptors[state.mblock];

    while (state.row_cursor < descriptor.valid_rows) {
      if (threadIdx.x == 0U) {
        state.run_rows = 0U;
        while (state.row_cursor < descriptor.valid_rows) {
          const auto canonical = descriptor.compact_row + state.row_cursor;
          const auto header = headers[canonical];
          const auto source_node = header.source_rank / a.local_world_size;
          const auto owns =
              source_node != a.node &&
              header.source_rank % a.local_world_size == context;
          if (!owns) {
            if (state.run_rows != 0U)
              break;
            ++state.row_cursor;
            continue;
          }
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              published(mapping[canonical]);
          const auto encoded = published.load(::cuda::memory_order_acquire);
          if (encoded == 0ULL) {
            __nanosleep(backward_tuning::kQueueIdleNanosleep);
            continue;
          }
          const auto compact = return_mapping_compact_row(encoded);
          const auto chunk = return_mapping_chunk(encoded);
          if (state.run_rows == 0U) {
            state.run_canonical = static_cast<std::uint32_t>(canonical);
            state.run_compact = static_cast<std::uint32_t>(compact);
            state.run_source = header.source_rank;
            state.run_rows = 1U;
            ++state.row_cursor;
            continue;
          }
          const auto first_mapping = mapping[state.run_canonical];
          const auto extends =
              header.source_rank == state.run_source &&
              canonical == state.run_canonical + state.run_rows &&
              compact == state.run_compact + state.run_rows &&
              chunk == return_mapping_chunk(first_mapping);
          if (!extends)
            break;
          ++state.run_rows;
          ++state.row_cursor;
        }
      }
      __syncthreads();
      if (state.run_rows == 0U)
        continue;

      const auto padded_begin =
          descriptor.padded_row + state.run_canonical - descriptor.compact_row;
      for (std::uint64_t item = threadIdx.x;
           item < static_cast<std::uint64_t>(state.run_rows) * vector_count;
           item += blockDim.x) {
        const auto row = item / vector_count;
        const auto vector = item % vector_count;
        reinterpret_cast<uint4 *>(return_staging)[
            (static_cast<std::uint64_t>(state.run_canonical) + row) *
                vector_count +
            vector] = reinterpret_cast<const uint4 *>(dx)[
            (static_cast<std::uint64_t>(padded_begin) + row) * vector_count +
            vector];
      }
      for (std::uint32_t row = threadIdx.x; row < state.run_rows;
           row += blockDim.x)
        gate_staging[state.run_canonical + row] = dgate[padded_begin + row];
      __threadfence_system();
      __syncthreads();

      if (threadIdx.x == 0U) {
        const ncclGin return_gin(a.gin_dev_comm,
                                 context + a.gin_context_count);
        const auto world = ncclTeamWorld(a.gin_dev_comm);
        const auto payload_bytes =
            static_cast<std::uint64_t>(state.run_rows) * a.hidden *
            sizeof(transport::BFloat16);
        return_gin.put(
            world, static_cast<int>(state.run_source), a.gin_window,
            a.gin_return_receive_offset +
                static_cast<std::uint64_t>(state.run_compact) * a.hidden *
                    sizeof(transport::BFloat16),
            a.gin_window,
            a.gin_return_staging_offset +
                static_cast<std::uint64_t>(state.run_canonical) * a.hidden *
                    sizeof(transport::BFloat16),
            payload_bytes);
        return_gin.put(
            world, static_cast<int>(state.run_source), a.gin_window,
            a.gin_return_gate_receive_offset +
                static_cast<std::uint64_t>(state.run_compact) * sizeof(float),
            a.gin_window,
            a.gin_return_gate_staging_offset +
                static_cast<std::uint64_t>(state.run_canonical) * sizeof(float),
            static_cast<std::uint64_t>(state.run_rows) * sizeof(float),
            ncclGin_VASignalAdd{
                a.gin_window,
                a.gin_return_fragment_ready_offset +
                    static_cast<std::uint64_t>(state.run_compact) *
                        sizeof(std::uint64_t),
                static_cast<std::uint64_t>(state.run_rows)},
            ncclGin_CounterInc{1}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++state.issued_return;
        if (state.issued_return - state.completed_return >= kMaxInflight) {
          return_gin.waitCounter(ncclCoopThread{}, 1, state.issued_return);
          state.completed_return = state.issued_return;
        }
        auto *status =
            peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
        atomicAdd(&status->gin_return_put_count, 2ULL);
        atomicAdd(&status->gin_return_payload_bytes, payload_bytes);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U)
      ++state.mblock;
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    const ncclGin return_gin(a.gin_dev_comm,
                             context + a.gin_context_count);
    if (state.issued_return != state.completed_return)
      return_gin.waitCounter(ncclCoopThread{}, 1, state.issued_return);
  }
}

} // namespace backward_gin_detail

#endif // __has_include(<nccl_device.h>)
