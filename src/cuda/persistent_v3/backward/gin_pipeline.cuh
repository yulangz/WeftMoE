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

inline constexpr unsigned long long kMaxInflight = 24ULL;

struct RelayInboundState {
  unsigned long long packet_cursor = 0ULL;
  unsigned long long packet_end = 0ULL;
  std::uint32_t wave_remaining = 0U;
  std::uint32_t batch_packets = 0U;
  std::uint32_t source_partition = 0U;
  std::uint32_t progressed = 0U;
};

struct SourceCompletionState {
  unsigned long long compact_rows = 0ULL;
  unsigned long long chunk_count = 0ULL;
  unsigned long long scan_cursor = 0ULL;
  unsigned long long completed = 0ULL;
  unsigned long long expected = 0ULL;
  unsigned long long selected_chunk = 0ULL;
  std::uint32_t context = 0U;
  std::uint32_t selected_rows = 0U;
  std::uint32_t selected = 0U;
  std::uint32_t chunk_ready = 0U;
  std::uint32_t progressed = 0U;
  unsigned long long marker_ready[
      backward_tuning::kComputeThreads / 32U]{};
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
        (completion.compact_rows + a.forward_return_chunk_rows - 1ULL) /
        a.forward_return_chunk_rows;
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

// Consume one bounded batch from a remotely arrived wave. The first header in
// each Forward wave is reused as the ordered BWD payload marker; batching
// amortizes the system fence over many packet fanouts while retaining regular
// opportunities to service Return completion.
__device__ __forceinline__ bool progress_inbound_once(
    const BackwardKernelArgs &a, RelayInboundState &state) {
  if (threadIdx.x == 0U) {
    state.progressed = 0U;
    state.batch_packets = 0U;
  }
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

  if (threadIdx.x == 0U) {
    if (state.wave_remaining == 0U) {
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
    if (state.wave_remaining != 0U)
      state.batch_packets =
          state.wave_remaining < backward_tuning::kDispatchBatchRows
              ? state.wave_remaining
              : backward_tuning::kDispatchBatchRows;
  }
  __syncthreads();
  if (state.batch_packets == 0U)
    return false;

  const auto vector_count = a.hidden / 8ULL;
  const auto top_k = a.dispatch.identity.top_k;
  const auto items = static_cast<std::uint64_t>(state.batch_packets) * top_k;
  constexpr std::uint32_t kWarps =
      backward_tuning::kComputeThreads / 32U;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  for (std::uint64_t item = warp; warp < kWarps && item < items;
       item += kWarps) {
    const auto packet_offset = item / top_k;
    const auto fanout_slot = item % top_k;
    const auto packet =
        partition_packet_base + state.packet_cursor + packet_offset;
    std::uint32_t valid = 0U;
    std::uint32_t destination = 0U;
    std::uint32_t destination_row = 0U;
    if (lane == 0U) {
      const auto header = headers[packet];
      if (fanout_slot < header.route_index) {
        const auto target =
            fanout[partition_fanout_base + header.dst_row + fanout_slot];
        destination =
            a.node * a.local_world_size + target.destination_local_rank;
        const auto *compact_to_padded = peer_at<const unsigned int>(
            a.peer_bases[destination], a.compact_to_padded_offset);
        destination_row = compact_to_padded[target.dst_row];
        valid = 1U;
      }
    }
    valid = __shfl_sync(0xffffffffU, valid, 0U);
    if (valid == 0U)
      continue;
    destination = __shfl_sync(0xffffffffU, destination, 0U);
    destination_row = __shfl_sync(0xffffffffU, destination_row, 0U);
    auto *destination_dr = peer_at<transport::BFloat16>(
        a.peer_bases[destination], a.padded_dr_offset);
    auto *target = reinterpret_cast<uint4 *>(destination_dr) +
                   static_cast<std::uint64_t>(destination_row) * vector_count;
    const auto *source = reinterpret_cast<const uint4 *>(payload) +
                         packet * vector_count;
    for (std::uint64_t vector = lane; vector < vector_count; vector += 32ULL)
      target[vector] = source[vector];
  }
  __threadfence_system();
  __syncthreads();

  for (std::uint64_t item = threadIdx.x; item < items;
       item += blockDim.x) {
    const auto packet_offset = item / top_k;
    const auto fanout_slot = item % top_k;
    const auto packet =
        partition_packet_base + state.packet_cursor + packet_offset;
    const auto header = headers[packet];
    if (fanout_slot >= header.route_index)
      continue;
    const auto target = fanout[partition_fanout_base + header.dst_row +
                               fanout_slot];
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
    state.packet_cursor += state.batch_packets;
    state.wave_remaining -= state.batch_packets;
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
    state.chunk_ready = 0U;
  }
  __syncthreads();
  if (source_completion_terminal(state) || state.chunk_count == 0ULL)
    return false;

  auto *chunk_state = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_state_offset);
  if (threadIdx.x == 0U) {
    for (unsigned long long attempt = 0ULL; attempt < state.expected;
         ++attempt) {
      const auto chunk = state.scan_cursor;
      state.scan_cursor += a.gin_context_count;
      if (state.scan_cursor >= state.chunk_count)
        state.scan_cursor = state.context;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> done(
          chunk_state[chunk]);
      if (done.load(::cuda::memory_order_acquire) == a.epoch)
        continue;
      state.selected_chunk = chunk;
      const auto row_begin = chunk * a.forward_return_chunk_rows;
      const auto remaining = state.compact_rows - row_begin;
      state.selected_rows = static_cast<std::uint32_t>(
          remaining < a.forward_return_chunk_rows
              ? remaining
              : a.forward_return_chunk_rows);
      state.selected = 1U;
      break;
    }
  }
  __syncthreads();
  if (state.selected == 0U)
    return false;

  const auto row_begin =
      state.selected_chunk * a.forward_return_chunk_rows;
  const auto *markers = reinterpret_cast<const unsigned long long *>(
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_return_fragment_ready_offset);
  constexpr auto kWarps = backward_tuning::kComputeThreads / 32U;
  constexpr unsigned int kWarpMask = 0xffffffffU;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  asm volatile("fence.proxy.async.global;" : : : "memory");
  unsigned long long ready_rows = 0ULL;
  for (std::uint32_t row = threadIdx.x; row < state.selected_rows;
       row += blockDim.x)
    ready_rows += markers[row_begin + row];
  for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U)
    ready_rows += __shfl_down_sync(kWarpMask, ready_rows, offset);
  if (lane == 0U)
    state.marker_ready[warp] = ready_rows;
  __syncthreads();
  if (threadIdx.x == 0U) {
    ready_rows = 0ULL;
#pragma unroll
    for (std::uint32_t index = 0U; index < kWarps; ++index)
      ready_rows += state.marker_ready[index];
    state.chunk_ready = ready_rows >= state.selected_rows ? 1U : 0U;
  }
  __syncthreads();
  if (state.chunk_ready == 0U)
    return false;

  const auto *route_index = peer_at<const unsigned long long>(
      a.local_base, a.forward_return_source_route_index_offset);
  const auto *return_receive =
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_return_receive_offset;
  auto *gate_output =
      peer_at<float>(a.local_base, a.dgate_output_offset);
  auto *route_ready = peer_at<unsigned long long>(
      a.local_base, a.return_ready_offset);
  for (std::uint32_t local = threadIdx.x; local < state.selected_rows;
       local += blockDim.x) {
    const auto compact = row_begin + local;
    const auto route = route_index[compact];
    gate_output[route] = *reinterpret_cast<const float *>(
        return_receive + compact * a.gin_return_row_bytes +
        a.hidden * sizeof(transport::BFloat16));
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        route_ready[route]);
    ready.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> done(
        chunk_state[state.selected_chunk]);
    done.store(a.epoch, ::cuda::memory_order_release);
    ++state.completed;
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
  std::uint32_t mblock_cursor = 0U;
  std::uint32_t selected_mblock = 0U;
  std::uint32_t completed_mblocks = 0U;
  std::uint32_t selected = 0U;
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
  auto *return_staging = bytes + a.gin_return_staging_offset;

  // FULL Return: each context owns exactly the remote source local-rank with
  // the same ordinal.  Scan completed M128 blocks cyclically so an unfinished
  // low-numbered block cannot hold up return traffic from later ready blocks.
  // The fixed per-block bitmask records which contexts have consumed a block;
  // it is scheduler metadata rather than a producer/consumer ring.
  while (state.completed_mblocks < a.mblocks) {
    if (threadIdx.x == 0U) {
      state.selected = 0U;
      const auto expected =
          static_cast<unsigned long long>(a.hidden /
                                          backward_tuning::kDxTileN);
      for (std::uint32_t probe = 0U; probe < a.mblocks; ++probe) {
        auto mblock = state.mblock_cursor + probe;
        if (mblock >= a.mblocks)
          mblock -= a.mblocks;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            context_done(
                words[scheduler_layout.return_context_done + mblock]);
        if ((context_done.load(::cuda::memory_order_relaxed) &
             (1ULL << context)) != 0ULL)
          continue;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(words[scheduler_layout.dx_completed + mblock]);
        if (completed.load(::cuda::memory_order_acquire) != expected)
          continue;
        state.selected_mblock = mblock;
        state.mblock_cursor = mblock + 1U;
        if (state.mblock_cursor == a.mblocks)
          state.mblock_cursor = 0U;
        state.row_cursor = 0U;
        state.selected = 1U;
        break;
      }
      if (state.selected == 0U)
        __nanosleep(backward_tuning::kQueueIdleNanosleep);
    }
    __syncthreads();
    if (state.selected == 0U)
      continue;
    const auto descriptor = descriptors[state.selected_mblock];

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
        auto *target = reinterpret_cast<uint4 *>(
            return_staging +
            (static_cast<std::uint64_t>(state.run_canonical) + row) *
                a.gin_return_row_bytes);
        target[vector] = reinterpret_cast<const uint4 *>(dx)[
            (static_cast<std::uint64_t>(padded_begin) + row) * vector_count +
            vector];
      }
      for (std::uint32_t row = threadIdx.x; row < state.run_rows;
           row += blockDim.x) {
        auto *gate = reinterpret_cast<float *>(
            return_staging +
            (static_cast<std::uint64_t>(state.run_canonical) + row) *
                a.gin_return_row_bytes +
            a.hidden * sizeof(transport::BFloat16));
        *gate = dgate[padded_begin + row];
      }
      __threadfence_system();
      __syncthreads();

      if (threadIdx.x == 0U) {
        const ncclGin return_gin(a.gin_dev_comm,
                                 context + a.gin_context_count);
        const auto world = ncclTeamWorld(a.gin_dev_comm);
        const auto payload_bytes =
            static_cast<std::uint64_t>(state.run_rows) *
            a.gin_return_row_bytes;
        return_gin.put(
            world, static_cast<int>(state.run_source), a.gin_window,
            a.gin_return_receive_offset +
                static_cast<std::uint64_t>(state.run_compact) *
                    a.gin_return_row_bytes,
            a.gin_window,
            a.gin_return_staging_offset +
                static_cast<std::uint64_t>(state.run_canonical) *
                    a.gin_return_row_bytes,
            payload_bytes,
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
        atomicAdd(&status->gin_return_put_count, 1ULL);
        atomicAdd(&status->gin_return_payload_bytes, payload_bytes);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          context_done(words[scheduler_layout.return_context_done +
                             state.selected_mblock]);
      context_done.fetch_or(1ULL << context, ::cuda::memory_order_release);
      ++state.completed_mblocks;
    }
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
