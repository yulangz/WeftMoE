#pragma once

// FULL Return keeps GIN ownership in the fixed RDMA CTAs and lands payloads
// directly in the registered compact receive plane. All Relay CTAs share
// local-copy batches and receive-completion chunks. Receive completion only
// publishes token readiness; it never copies the registered Return payload.

#if __has_include(<nccl_device.h>)

namespace forward_return_detail {

using DirectLocalReturnBarrier =
    ::cuda::barrier<::cuda::thread_scope_block>;

__device__ __forceinline__ bool cancelled(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

__device__ __forceinline__ unsigned long long source_compact_rows(
    const ForwardKernelArgs &a, const unsigned long long *,
    std::uint32_t context = 0U) {
  const auto *directories = reinterpret_cast<const GinDispatchContextRecord *>(
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_send_dispatch_directory_offset);
  return directories[context].return_row_count;
}

// Scratch words after the parallel directory totals are reserved for the
// direct-return producer/consumer handshake.  Compute CTAs append immutable
// fragments and publish one terminal count per processed M-block.  The fixed
// RDMA CTAs are the sole GIN context owners and record their issued counts here.
__device__ __forceinline__ unsigned long long direct_scheduler_base(
    const ForwardKernelArgs &a) {
  const auto partitions = a.dispatch.identity.num_nodes - 1ULL;
  // Eight words after the directory arrays belong to mixed scheduling and the
  // transport-only startup rendezvous.  Start direct counters after that
  // aligned block so neither state can alias Return ownership metadata.
  return 16ULL + 2ULL * a.world +
         2ULL * partitions * a.gin_context_count + 8ULL;
}

__device__ __forceinline__ unsigned long long *direct_issued_counts(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  return scheduler + direct_scheduler_base(a);
}

__device__ __forceinline__ unsigned long long &direct_mblocks_complete(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  return scheduler[direct_scheduler_base(a) + a.gin_context_count + 1ULL];
}

__device__ __forceinline__ unsigned long long &direct_fragment_tail(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  return scheduler[direct_scheduler_base(a) + a.gin_context_count];
}

__device__ __noinline__ void enqueue_direct_full_mblock(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    std::uint64_t canonical_row,
    std::uint32_t row_count, std::uint32_t context_mask,
    std::uint32_t local_context_mask) {
  if (row_count == 0U || forward_cancelled(a))
    return;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
      direct_fragment_tail(a, scheduler));
  const auto slot = tail.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
  if (slot >= a.forward_direct_return_fragment_capacity) {
    auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
        a.local_base, a.forward_v3_control_offset);
    persistent_v3::publish_error_once(control->generation, 310ULL);
    return;
  }

  auto *fragments = peer_at<ForwardDirectReturnFragment>(
      a.local_base, a.forward_direct_return_fragment_offset);
  auto &fragment = fragments[slot];
  fragment.canonical_row = canonical_row;
  fragment.compact_row = 0ULL;
  // Queue records are immutable M-block descriptors, not wire fragments.
  // source_rank is otherwise unused until the RDMA owner materializes a local
  // wire fragment, so reuse it for the node-local Return consumer mask.
  fragment.source_rank = local_context_mask;
  fragment.row_count = row_count;
  fragment.context = UINT32_MAX;
  fragment.reserved = context_mask;
  ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device> published(
      fragment.epoch);
  published.store(a.epoch, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
  if (a.use_gin) {
    auto *completion =
        static_cast<transport_state::GenerationCompletionVector *>(
            a.peer_bases[a.world]);
    atomicMax(&completion->bulk_local_ready, 1001ULL + slot);
  }
#endif
}

// Called by lane 0 only after all FC2 N tiles have made canonical and registered
// staging stores system-visible.  Publishing one record per M-block keeps the
// row-map scan and fragment coalescing out of the WGMMA CTA altogether.
__device__ __noinline__ void issue_direct_full_mblock(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    std::uint64_t canonical_row, std::uint32_t valid_rows,
    std::uint64_t context_masks) {
  enqueue_direct_full_mblock(
      a, scheduler, canonical_row, valid_rows,
      static_cast<std::uint32_t>(context_masks),
      static_cast<std::uint32_t>(context_masks >> 32U));
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> done(
      direct_mblocks_complete(a, scheduler));
  const auto completed =
      done.fetch_add(1ULL, ::cuda::memory_order_release) + 1ULL;
#if MEGAMOE_DEBUG_VALIDATE
  if (a.use_gin) {
    auto *completion =
        static_cast<transport_state::GenerationCompletionVector *>(
            a.peer_bases[a.world]);
    atomicMax(&completion->progress_ctas_at_bulk_return, 1000ULL + completed);
  }
#endif
}

// Separate work planes share immutable FC2 descriptors and explicit completion
// counters. None of the local-copy/completion workers owns a GIN context.
// FULL Return already lands in this rank's registered compact receive plane.
// Once a chunk's ordered fragment markers are complete, publish only its
// route-to-compact metadata and readiness.  Reduce then reads that registered
// payload in place, eliminating the former 352 MiB receive->contribution copy
// while preserving large contiguous network puts.
__device__ __forceinline__ void
publish_direct_full_source_chunk(const ForwardKernelArgs &a,
                                 std::uint64_t row_begin,
                                 std::uint32_t row_count) {
  const auto *source_route_index = peer_at<const unsigned long long>(
      a.local_base, a.forward_return_source_route_index_offset);
  auto *route_ready =
      peer_at<unsigned long long>(a.local_base, a.return_ready_epoch_offset);

  // The marker is the completion action of the payload put.  Bring async-proxy
  // writes into the generic proxy before publishing a CUDA-visible ready word.
  asm volatile("fence.proxy.async.global;" : : : "memory");
  // Dispatch already materialized route->compact.  This fence therefore only
  // has to order the received payload before the route/queue release below.
  __threadfence_system();
  __syncthreads();
  for (std::uint64_t local_row = threadIdx.x; local_row < row_count;
       local_row += blockDim.x) {
    const auto compact_row = row_begin + local_row;
    const auto route = source_route_index[compact_row];
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        route_ready[route]);
    if (a.forward_dynamic_reduce_queue)
      forward_reduce_detail::publish_ready_route(a, a.local_base, route, 1U);
    else
      ready.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
}

struct DirectFullCompletionState {
  unsigned long long compact_rows = 0ULL;
  unsigned long long chunk_count = 0ULL;
  unsigned long long completed = 0ULL;
  unsigned long long expected = 0ULL;
  unsigned long long selected_chunk = 0ULL;
  std::uint32_t selected_rows = 0U;
  std::uint32_t initialized = 0U;
  std::uint32_t selected = 0U;
  std::uint32_t chunk_ready = 0U;
  std::uint32_t progressed = 0U;
  unsigned long long marker_ready[persistent_v3::kWarpsPerCta]{};
};

__device__ __forceinline__ bool
direct_full_completion_terminal(const DirectFullCompletionState &state) {
  return state.initialized != 0U && state.completed == state.expected;
}

// Inspect at most one receive chunk and never wait on it.  The scan cursor
// advances even on a miss, so a late fragment cannot head-of-line block later
// chunks. The epoch-tagged chunk word records ownership; a separate completed
// counter advances only after publication, so a claimed chunk is not confused
// with a completed one. Neither a fixed helper nor a blocking wait is required.
__device__ __forceinline__ bool progress_direct_full_completion_once(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context, unsigned long long *scheduler,
    DirectFullCompletionState &state) {
  if (threadIdx.x == 0U) {
    state.progressed = 0U;
    state.selected = 0U;
    state.chunk_ready = 0U;
    if (state.initialized == 0U &&
        persistent_v3::load_acquire<::cuda::thread_scope_device>(
            control.dispatch[0].outbound_directory_ready.value) >= a.epoch) {
      state.compact_rows = source_compact_rows(a, scheduler);
      state.chunk_count =
          (state.compact_rows + a.forward_return_chunk_rows - 1ULL) /
          a.forward_return_chunk_rows;
      state.expected = state.chunk_count;
      state.initialized = 1U;
    }
    state.completed = persistent_v3::load_acquire<::cuda::thread_scope_device>(
        control.return_chunks_complete.value);
  }
  __syncthreads();
  if (state.initialized == 0U || state.completed == state.expected ||
      cancelled(a, control))
    return false;

  auto *chunk_state = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_state_offset);
  if (threadIdx.x == 0U) {
    // One bounded probe. Atomic scan distribution avoids a fixed scanner per
    // rail; a slow/missing chunk is revisited without blocking later chunks.
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> scan(
        control.return_chunk_scan.value);
    const auto chunk =
        scan.fetch_add(1ULL, ::cuda::memory_order_relaxed) % state.chunk_count;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claimed(
        chunk_state[chunk]);
    if (claimed.load(::cuda::memory_order_acquire) != a.epoch) {
      state.selected_chunk = chunk;
      const auto remaining =
          state.compact_rows - chunk * a.forward_return_chunk_rows;
      state.selected_rows = static_cast<std::uint32_t>(
          min(remaining,
              static_cast<unsigned long long>(a.forward_return_chunk_rows)));
      state.selected = 1U;
    }
  }
  __syncthreads();
  if (state.selected == 0U)
    return false;

  const auto row_begin = state.selected_chunk * a.forward_return_chunk_rows;
  const auto *fragment_ready = reinterpret_cast<const unsigned long long *>(
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_return_fragment_ready_offset);
  constexpr auto kMarkerWarps = forward_tuning::kCompletionMarkerWarps;
  if (threadIdx.x < kMarkerWarps * 32U) {
    constexpr unsigned int kWarpMask = 0xffffffffU;
    const auto warp = threadIdx.x / 32U;
    const auto lane = threadIdx.x & 31U;
    // GIN writes arrive through the async proxy.  One coherence handoff and one
    // parallel marker pass are enough for this opportunistic probe.
    asm volatile("fence.proxy.async.global;" : : : "memory");
    unsigned long long ready_rows = 0ULL;
    for (std::uint32_t row = warp * 32U + lane; row < state.selected_rows;
         row += kMarkerWarps * 32U)
      ready_rows += fragment_ready[row_begin + row];
    for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U)
      ready_rows += __shfl_down_sync(kWarpMask, ready_rows, offset);
    if (lane == 0U)
      state.marker_ready[warp] = ready_rows;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    unsigned long long ready_rows = 0ULL;
#pragma unroll
    for (std::uint32_t warp = 0U; warp < kMarkerWarps; ++warp)
      ready_rows += state.marker_ready[warp];
    state.chunk_ready = ready_rows >= state.selected_rows ? 1U : 0U;
  }
  __syncthreads();
  if (state.chunk_ready == 0U)
    return false;

  // Mark ownership only after observing every payload marker. Multiple CTAs
  // may probe the same chunk, but exactly one publishes its route readiness.
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claimed(
        chunk_state[state.selected_chunk]);
    auto old = claimed.load(::cuda::memory_order_relaxed);
    state.selected =
        old != a.epoch && claimed.compare_exchange_strong(
                              old, a.epoch, ::cuda::memory_order_acq_rel,
                              ::cuda::memory_order_relaxed);
  }
  __syncthreads();
  if (state.selected == 0U)
    return false;

  MEGAMOE_TIMELINE_RECORD(a.timeline, FullReturnChunkReady, SourceRelay,
                          context, state.selected_rows, a.epoch);
  if (threadIdx.x == 0U) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
  }
  publish_direct_full_source_chunk(a, row_begin, state.selected_rows);
  MEGAMOE_TIMELINE_RECORD(a.timeline, FullReturnChunkPublished, SourceRelay,
                          context, state.selected_rows, a.epoch);
  if (threadIdx.x == 0U) {
    state.completed =
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.return_chunks_complete.value, 1ULL) +
        1ULL;
    state.progressed = 1U;
  }
  __syncthreads();
  return state.progressed != 0U;
}
inline constexpr std::uint32_t kMaxDirectLocalReturnBatch = 8U;

struct DirectLocalReturnState {
  unsigned long long cursor = 0ULL;
  unsigned int preferred_stripe = 0U;
  std::uint32_t context = 0U;
  std::uint32_t selected = 0U;
  std::uint32_t active = 0U;
  std::uint32_t copied_rows = 0U;
  ForwardDirectReturnFragment batch[kMaxDirectLocalReturnBatch]{};
};

__device__ __forceinline__ bool
direct_local_return_terminal(const ForwardKernelArgs &a,
                             unsigned long long *scheduler,
                             const DirectLocalReturnState &) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> produced(
      direct_mblocks_complete(a, scheduler));
  if (produced.load(::cuda::memory_order_acquire) < scheduler[3])
    return false;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
      direct_fragment_tail(a, scheduler));
  const auto end = tail.load(::cuda::memory_order_acquire);
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  const auto stripes = (a.forward_relay_cta_count + a.local_world_size - 1U) /
      a.local_world_size;
  for (unsigned int context = 0U; context < a.local_world_size; ++context)
    for (unsigned int stripe = 0U; stripe < stripes; ++stripe) {
      const auto count = persistent_v3::load_acquire<::cuda::thread_scope_device>(
          control->local_return_head[context * stripes + stripe].value);
      if (stripe + count * stripes < end)
        return false;
    }
  return true;
}

// Shared destination/stripe cursors retain interleaved descriptor batches
// without binding a CTA to a stripe. Prefer the home stripe and destination,
// then steal another ready batch. Never claim unpublished descriptors.
__device__ __forceinline__ bool progress_direct_local_return_once(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    std::uint32_t preferred_context, unsigned long long *scheduler,
    DirectLocalReturnState &state, std::byte *tma_storage,
    DirectLocalReturnBarrier *tma_full) {
  auto *fragments = peer_at<ForwardDirectReturnFragment>(
      a.local_base, a.forward_direct_return_fragment_offset);
  if (threadIdx.x == 0U) {
    state.selected = 0U;
    state.active = 0U;
    state.copied_rows = 0U;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
        direct_fragment_tail(a, scheduler));
    const auto published_tail = tail.load(::cuda::memory_order_acquire);
    const auto stripes = (a.forward_relay_cta_count + a.local_world_size - 1U) /
        a.local_world_size;
    for (unsigned int attempt = 0U; attempt < a.local_world_size * stripes; ++attempt) {
      const auto context = (preferred_context + attempt / stripes) % a.local_world_size;
      const auto stripe = (state.preferred_stripe + attempt % stripes) % stripes;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> head(
          control.local_return_head[context * stripes + stripe].value);
      auto begin = head.load(::cuda::memory_order_relaxed);
      unsigned int count = 0U;
      while (count < a.forward_local_return_batch_mblocks &&
             stripe + (begin + count) * stripes < published_tail) {
        ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device> ready(
            fragments[stripe + (begin + count) * stripes].epoch);
        if (ready.load(::cuda::memory_order_acquire) != a.epoch)
          break;
        ++count;
      }
      if (!count || !head.compare_exchange_strong(begin, begin + count,
                                                  ::cuda::memory_order_acq_rel,
                                                  ::cuda::memory_order_relaxed))
        continue;
      state.cursor = stripe + begin * stripes;
      state.context = context;
      state.selected = count;
      for (unsigned int item = 0U; item < count; ++item) {
        state.batch[item] = fragments[stripe + (begin + item) * stripes];
        if ((state.batch[item].source_rank & (1U << context)) != 0U)
          ++state.active;
      }
      break;
    }
  }
  __syncthreads();
  if (state.selected == 0U || cancelled(a, control))
    return false;

  const auto context = state.context;
  const auto local_source = a.node * a.local_world_size + context;
  if (state.active != 0U) {
    MEGAMOE_TIMELINE_RECORD(a.timeline, LocalReturnBatchStart, LocalReturn,
                            context, (state.cursor << 32U) | state.active,
                            a.epoch);
    if (threadIdx.x == 0U) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
    }
    const auto *headers = peer_at<const LocalP2pRouteHeader>(
        a.local_base, a.direct_headers_offset);
    const auto *forward_output =
        a.use_gin && a.forward_registered_output
            ? reinterpret_cast<const transport::BFloat16 *>(
                  static_cast<const std::byte *>(a.gin_window_base) +
                  a.gin_return_staging_offset)
            : peer_at<const transport::BFloat16>(a.local_base,
                                                 a.forward_output_rows_offset);
    auto *source_contribution = peer_at<transport::BFloat16>(
        a.peer_bases[local_source], a.contribution_offset);
    const auto copy_bytes =
        static_cast<std::uint32_t>(a.hidden * sizeof(transport::BFloat16));
    const auto tma_warps = a.forward_relay_tma_warps;
    const bool use_tma = tma_warps != 0U && copy_bytes != 0U &&
                         copy_bytes <= 16384U && (copy_bytes & 15U) == 0U;
    if (use_tma) {
      const auto warp = threadIdx.x / 32U;
      const auto lane = threadIdx.x & 31U;
      // Each warp leader owns an independent mbarrier and 14-KiB SMEM row.
      // This retains the twelve concurrent row streams of the vector path
      // while replacing 448 lane-issued peer stores with one bulk store.
      if (warp < tma_warps && lane == 0U) {
        auto *warp_storage = tma_storage + warp * copy_bytes;
        auto &warp_full = tma_full[warp];
        bool copied = false;
        for (std::uint32_t item = 0U; item < state.selected; ++item) {
          const auto fragment = state.batch[item];
          if ((fragment.source_rank & (1U << context)) == 0U)
            continue;
          for (std::uint32_t local_row = warp; local_row < fragment.row_count;
               local_row += tma_warps) {
            const auto canonical_row = fragment.canonical_row + local_row;
            const auto header = headers[canonical_row];
            if (header.source_rank != local_source)
              continue;
            const auto *source = forward_output + canonical_row * a.hidden;
            auto *target = source_contribution + header.route_index * a.hidden;
            ::cuda::device::experimental::cp_async_bulk_global_to_shared(
                warp_storage, source, copy_bytes, warp_full);
            auto token =
                ::cuda::device::barrier_arrive_tx(warp_full, 1U, copy_bytes);
            warp_full.wait(static_cast<decltype(token) &&>(token));
            ::cuda::device::experimental::cp_async_bulk_shared_to_global(
                target, warp_storage, copy_bytes);
            ::cuda::device::experimental::cp_async_bulk_commit_group();
            ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
            copied = true;
          }
        }
        if (copied) {
          ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
          __threadfence_system();
        }
      }
    } else {
      const auto vector_count = a.hidden / 8ULL;
      const auto warp = threadIdx.x / 32U;
      const auto lane = threadIdx.x & 31U;
      constexpr auto kWarps = persistent_v3::kWarpsPerCta;
      // A warp owns one row. Twelve independent copy streams expose enough
      // outstanding peer transactions to fill NVLink. Batching only changes
      // the fence cadence; each row keeps the same warp ownership.
      for (std::uint32_t item = 0U; item < state.selected; ++item) {
        const auto fragment = state.batch[item];
        if ((fragment.source_rank & (1U << context)) == 0U)
          continue;
        for (std::uint32_t local_row = warp; local_row < fragment.row_count;
             local_row += kWarps) {
          const auto canonical_row = fragment.canonical_row + local_row;
          const auto header = headers[canonical_row];
          if (header.source_rank != local_source)
            continue;
          const auto *source128 = reinterpret_cast<const uint4 *>(
              forward_output + canonical_row * a.hidden);
          auto *target128 = reinterpret_cast<uint4 *>(
              source_contribution + header.route_index * a.hidden);
          for (std::uint64_t vector = lane; vector < vector_count;
               vector += 32ULL)
            target128[vector] = source128[vector];
        }
      }
      __threadfence_system();
    }
    __syncthreads();
    std::uint32_t published_rows = 0U;
    for (std::uint32_t item = 0U; item < state.selected; ++item) {
      const auto fragment = state.batch[item];
      if ((fragment.source_rank & (1U << context)) == 0U)
        continue;
      for (std::uint32_t local_row = threadIdx.x;
           local_row < fragment.row_count; local_row += blockDim.x) {
        const auto canonical_row = fragment.canonical_row + local_row;
        const auto header = headers[canonical_row];
        if (header.source_rank != local_source)
          continue;
        auto *source_route_ready = peer_at<unsigned long long>(
            a.peer_bases[local_source], a.return_ready_epoch_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(source_route_ready[header.route_index]);
        if (a.forward_dynamic_reduce_queue)
          forward_reduce_detail::publish_ready_route(
              a, a.peer_bases[local_source], header.route_index, 2U);
        else
          ready.store(a.epoch, ::cuda::memory_order_release);
        ++published_rows;
      }
    }
    constexpr unsigned int kWarpMask = 0xffffffffU;
    const auto warp_rows = __reduce_add_sync(kWarpMask, published_rows);
    if ((threadIdx.x & 31U) == 0U && warp_rows != 0U)
      atomicAdd(&state.copied_rows, warp_rows);
    __syncthreads();

    if (threadIdx.x == 0U && state.copied_rows != 0U) {
      if (a.forward_stream_local_dispatch && local_source != a.rank) {
        auto *status =
            peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
        atomicAdd(&status->actual_return_peer_bytes,
                  static_cast<unsigned long long>(state.copied_rows) *
                      a.hidden * sizeof(transport::BFloat16));
      }
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          completed(scheduler[4U + local_source]);
      const auto old =
          completed.fetch_add(state.copied_rows, ::cuda::memory_order_acq_rel);
      const auto expected = old >> 32U;
      const auto actual = (old & 0xffffffffULL) + state.copied_rows;
      if (actual == expected) {
        auto *source_signal = peer_at<unsigned long long>(
            a.peer_bases[local_source], a.outgoing_sequence_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(source_signal[a.rank * a.channels]);
        ready.store(a.epoch, ::cuda::memory_order_release);
      }
    }
    __syncthreads();
    MEGAMOE_TIMELINE_RECORD(a.timeline, LocalReturnBatchEnd, LocalReturn,
                            context, (state.cursor << 32U) | state.copied_rows,
                            a.epoch);
  }

  return true;
}
// Only the permanent owner of `fragment.context` calls this function.  Keeping
// both payload and marker on the same exclusive QP gives the receiver a cheap
// per-fragment release flag without exposing GIN state to FC2 producers.
__device__ __forceinline__ void issue_direct_full_fragment_rdmacta(
    const ForwardKernelArgs &a, const ncclGin &gin, ncclTeam world,
    const ForwardDirectReturnFragment &fragment,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long &issued) {
  const auto context = fragment.context;
  const auto source_node = fragment.source_rank / a.local_world_size;
  const auto source_partition =
      source_node < a.node ? source_node : source_node - 1U;
  const auto payload_bytes =
      static_cast<std::uint64_t>(fragment.row_count) * a.hidden *
      sizeof(transport::BFloat16);

#if MEGAMOE_DEBUG_VALIDATE
  // Preserve the exact first WQE owned by this CTA.  If a local completion
  // stops advancing, the timeout dump then identifies the remote rank/context
  // and both source and destination row intervals without perturbing the GIN
  // polling loop itself.
  const auto diagnostic_key =
      static_cast<unsigned long long>(fragment.source_rank) *
          a.gin_context_count +
      context;
  if (atomicCAS(&completion->first_waiting_tile, ~0ULL, diagnostic_key) ==
      ~0ULL) {
    completion->first_wait_expected = fragment.canonical_row;
    completion->first_wait_arrived = fragment.compact_row;
    completion->tile_waiter_count = fragment.row_count;
  }
#endif

  const auto compact_capacity =
      static_cast<std::uint64_t>(a.dispatch.identity.num_nodes - 1U) *
      a.max_tokens_per_rank * a.dispatch.identity.top_k;
  const bool valid =
      fragment.source_rank < a.world && a.local_world_size != 0U &&
      source_node < a.dispatch.identity.num_nodes && source_node != a.node &&
      context < a.gin_context_count && fragment.row_count != 0U &&
      fragment.canonical_row < a.max_destination_rows &&
      fragment.row_count <=
          a.max_destination_rows - fragment.canonical_row &&
      fragment.compact_row < compact_capacity &&
      fragment.row_count <= compact_capacity - fragment.compact_row;
  if (!valid) {
    auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
        a.local_base, a.forward_v3_control_offset);
    persistent_v3::publish_error_once(control->generation, 312ULL);
    return;
  }

  auto &return_tasks =
      completion_task(completion, transport_state::TaskClass::Return);
  atomicAdd(&return_tasks.expected, 1ULL);
  claim_completion_tasks(completion, transport_state::TaskClass::Return);
  atomicAdd(&completion->return_local_completion_expected, 1ULL);
  auto &record = context_completion[
      static_cast<std::uint64_t>(source_partition) * a.gin_context_count +
      context];
  atomicAdd(&record.return_expected, 1ULL);
  atomicAdd(&record.return_claimed, 1ULL);
  atomicAdd(&record.inflight, 1ULL);

  MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnGinPutIssue, GinReturn, context,
                          payload_bytes, a.epoch);
  gin.put(
      world, static_cast<int>(fragment.source_rank), a.gin_window,
      a.gin_return_receive_offset +
          static_cast<std::size_t>(fragment.compact_row) * a.hidden *
              sizeof(transport::BFloat16),
      a.gin_window,
      a.gin_return_staging_offset +
          static_cast<std::size_t>(fragment.canonical_row) * a.hidden *
              sizeof(transport::BFloat16),
      payload_bytes,
      ncclGin_VASignalAdd{
          a.gin_window,
          a.gin_return_fragment_ready_offset +
              static_cast<std::size_t>(fragment.compact_row) *
                  sizeof(std::uint64_t),
          static_cast<std::uint64_t>(fragment.row_count)},
      ncclGin_CounterInc{1},
      ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_block,
      ::cuda::thread_scope_device);
  ++issued;

  auto *status =
      peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_return_put_count, 1ULL);
  atomicAdd(&status->gin_signal_count, 1ULL);
  atomicAdd(&status->gin_return_payload_bytes, payload_bytes);
}

struct DirectFullPendingRun {
  std::uint32_t source = 0U;
  std::uint64_t canonical = 0ULL;
  std::uint64_t compact = 0ULL;
  std::uint64_t chunk = 0ULL;
  std::uint32_t rows = 0U;
};

__device__ __forceinline__ void flush_direct_full_run(
    const ForwardKernelArgs &a, const ncclGin &gin, ncclTeam world,
    std::uint32_t context, DirectFullPendingRun &run,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long &issued) {
  if (run.rows == 0U)
    return;
  ForwardDirectReturnFragment fragment{};
  fragment.canonical_row = run.canonical;
  fragment.compact_row = run.compact;
  fragment.source_rank = run.source;
  fragment.row_count = run.rows;
  fragment.context = context;
  fragment.epoch = a.epoch;
  issue_direct_full_fragment_rdmacta(a, gin, world, fragment, completion,
                                     context_completion, issued);
  run.rows = 0U;
}

struct DirectFullRdmaState {
  forward_reduce_detail::RdmaReduceMailbox *reduce_mailbox = nullptr;
  unsigned long long cursor = 0ULL;
  unsigned long long issued = 0ULL;
  unsigned long long completed = 0ULL;
  std::uint32_t local_row = 0U;
  DirectFullPendingRun run{};
};

__device__ __forceinline__ bool direct_full_return_has_credit(
    const ForwardKernelArgs &a, const ncclGin &gin,
    DirectFullRdmaState &state) {
  const auto limit =
      static_cast<unsigned long long>(a.forward_return_max_inflight);
  if (limit == 0ULL || state.issued - state.completed < limit)
    return true;
  const auto completed =
      gin.readCounter(1, 56, ::cuda::memory_order_acquire);
  state.completed = completed < state.issued ? completed : state.issued;
  return state.issued - state.completed < limit;
}

// Scan one M-block incrementally and emit at most one coalesced run per call.
// The row cursor and pending run make a full Return QP non-blocking: Dispatch
// retains priority on the same RDMA CTA instead of waiting for queue credit or
// issuing every run in a 128-row block as one unbounded burst.
__device__ __forceinline__ bool progress_direct_full_mblock_rdmacta(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control, const ncclGin &gin,
    ncclTeam world, std::uint32_t context,
    const ForwardDirectReturnFragment &mblock,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    DirectFullRdmaState &state) {
  auto *mapping = peer_at<unsigned long long>(
      a.local_base, a.forward_output_ready_epoch_offset);
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);

  while (state.local_row < mblock.row_count) {
    const auto canonical_row = mblock.canonical_row + state.local_row;
    const auto source = headers[canonical_row].source_rank;
    if (source / a.local_world_size == a.node) {
      if (state.run.rows != 0U) {
        if (!direct_full_return_has_credit(a, gin, state))
          return false;
        flush_direct_full_run(a, gin, world, context, state.run, completion,
                              context_completion, state.issued);
        return true;
      }
      ++state.local_row;
      continue;
    }
    // The immutable canonical header already identifies the only FULL
    // connection/context that can own this row.  Filter here, before touching
    // the peer-published reverse mapping.  The previous order made all eight
    // RDMA CTAs acquire-poll every remote row and discarded seven copies only
    // after the expensive system-scope load completed.
    const auto direct_context = source % a.local_world_size;
    if (direct_context != context) {
      if (state.run.rows != 0U) {
        if (!direct_full_return_has_credit(a, gin, state))
          return false;
        flush_direct_full_run(a, gin, world, context, state.run, completion,
                              context_completion, state.issued);
        return true;
      }
      ++state.local_row;
      continue;
    }
#if MEGAMOE_DEBUG_VALIDATE
    atomicMax(&completion->fc1_scheduler_value, 4000ULL + canonical_row);
#endif
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(mapping[canonical_row]);
    const auto encoded = published.load(::cuda::memory_order_acquire);
#if MEGAMOE_DEBUG_VALIDATE
    if (encoded == 0ULL)
      atomicMax(&completion->fc2_scheduler_value, 5000ULL + canonical_row);
#endif
    // The helper's per-context release certificate is acquired before this
    // scan, so a zero mapping means that helper has not completed yet. Leave
    // the row cursor in place and let Dispatch make progress before retrying.
    if (encoded == 0ULL || cancelled(a, control))
      return false;
#if MEGAMOE_DEBUG_VALIDATE
    atomicMax(&completion->fc2_scheduler_value, 6000ULL + canonical_row);
#endif
    const auto compact = return_mapping_compact_row(encoded);
    const auto chunk = return_mapping_chunk(encoded);
    const bool extends =
        state.run.rows != 0U && source == state.run.source &&
        chunk == state.run.chunk &&
        compact == state.run.compact + state.run.rows &&
        canonical_row == state.run.canonical + state.run.rows;
    if (!extends) {
      if (state.run.rows != 0U) {
        if (!direct_full_return_has_credit(a, gin, state))
          return false;
        flush_direct_full_run(a, gin, world, context, state.run, completion,
                              context_completion, state.issued);
        return true;
      }
      state.run.source = source;
      state.run.canonical = canonical_row;
      state.run.compact = compact;
      state.run.chunk = chunk;
      state.run.rows = 1U;
    } else {
      ++state.run.rows;
    }
    ++state.local_row;
  }
  if (state.run.rows != 0U) {
    if (!direct_full_return_has_credit(a, gin, state))
      return false;
    flush_direct_full_run(a, gin, world, context, state.run, completion,
                          context_completion, state.issued);
  }
  state.local_row = 0U;
  ++state.cursor;
  return true;
}

// Once RDMA packing has drained, its ten data warps have no more useful work
// while lane 0 remains the permanent GIN owner.  Use those otherwise-idle
// warps to acquire-prefetch the eager reverse map.  Each warp owns a disjoint
// subsequence of immutable M-block records; each RDMA CTA contributes only its
// context bit.  This keeps system-scope polling and the first row scan out of
// the scalar GIN issue loop without transferring a QP or adding a CTA-wide
// rendezvous.
__device__ __forceinline__ void direct_full_mapping_helper(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control, std::uint32_t context,
    unsigned long long *scheduler, std::uint32_t helper_warp,
    std::uint32_t helper_warps) {
  constexpr unsigned int kWarpMask = 0xffffffffU;
  const auto lane = threadIdx.x & 31U;
  const auto mblocks = scheduler[3];
  if (mblocks > a.forward_direct_return_fragment_capacity) {
    if (lane == 0U)
      persistent_v3::publish_error_once(control.generation, 311ULL);
    return;
  }

  auto *fragments = peer_at<ForwardDirectReturnFragment>(
      a.local_base, a.forward_direct_return_fragment_offset);
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  auto *mapping = peer_at<unsigned long long>(
      a.local_base, a.forward_output_ready_epoch_offset);

  for (unsigned long long slot = helper_warp; slot < mblocks;
       slot += helper_warps) {
    auto &fragment = fragments[slot];
    ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device> published(
        fragment.epoch);
    while (published.load(::cuda::memory_order_acquire) != a.epoch &&
           !cancelled(a, control))
      __nanosleep(forward_tuning::kReturnIdleNanosleep);
    if (cancelled(a, control))
      return;

    const auto context_bit = 1U << context;
    if ((fragment.reserved & context_bit) == 0U)
      continue;

    bool ready = false;
    while (!ready && !cancelled(a, control)) {
      bool missing = false;
      for (std::uint32_t local_row = lane;
           local_row < fragment.row_count; local_row += 32U) {
        const auto canonical_row = fragment.canonical_row + local_row;
        const auto source = headers[canonical_row].source_rank;
        if (source >= a.world ||
            source / a.local_world_size == a.node ||
            source % a.local_world_size != context)
          continue;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            row_mapping(mapping[canonical_row]);
        missing = missing ||
                  row_mapping.load(::cuda::memory_order_acquire) == 0ULL;
      }
      ready = __ballot_sync(kWarpMask, missing) == 0U;
      if (!ready && lane == 0U)
        __nanosleep(forward_tuning::kReturnIdleNanosleep);
      __syncwarp(kWarpMask);
    }
    if (cancelled(a, control))
      return;

    if (lane == 0U) {
      // compact_row is unused by an M-block queue record.  Reuse it as the
      // per-context acquire certificate; emitted payload fragments are local
      // temporaries and continue to carry their real compact row there.
      ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device>
          ready_mask(fragment.compact_row);
      ready_mask.fetch_or(static_cast<std::uint64_t>(context_bit),
                          ::cuda::memory_order_release);
    }
    __syncwarp(kWarpMask);
  }
}

// FC2 can finish a complete M-block before Dispatch has drained.  The helper
// warp's release proves that every mapping owned by this context was observed
// non-zero.  The scalar owner then performs only the coalescing/issue scan.
__device__ __forceinline__ bool direct_full_mblock_mapping_ready(
    const ForwardKernelArgs &a, std::uint32_t context,
    ForwardDirectReturnFragment &mblock) {
  ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device>
      ready_mask(mblock.compact_row);
  return (ready_mask.load(::cuda::memory_order_acquire) & (1ULL << context)) !=
         0ULL;
}

// Consume at most one coalesced run. Calling this once from each Dispatch
// scheduler iteration gives Rail Dispatch priority while opportunistic FULL
// puts fill otherwise idle network time. Every context remains the sole owner
// of its GIN QP.
__device__ __forceinline__ bool progress_direct_full_rdma_once(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control, const ncclGin &gin,
    ncclTeam world, std::uint32_t context,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler, DirectFullRdmaState &state) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
      direct_fragment_tail(a, scheduler));
  if (state.cursor >= tail.load(::cuda::memory_order_acquire))
    return false;

  auto *fragments = peer_at<ForwardDirectReturnFragment>(
      a.local_base, a.forward_direct_return_fragment_offset);
  auto &fragment = fragments[state.cursor];
  ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device> published(
      fragment.epoch);
  if (published.load(::cuda::memory_order_acquire) != a.epoch)
    return false;

  if ((fragment.reserved & (1U << context)) != 0U) {
    if (!direct_full_mblock_mapping_ready(a, context, fragment))
      return false;
    return progress_direct_full_mblock_rdmacta(
        a, control, gin, world, context, fragment, completion,
        context_completion, state);
  }
  state.local_row = 0U;
  state.run.rows = 0U;
  ++state.cursor;
  return true;
}

__device__ __forceinline__ void direct_full_rdma_thread(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control, std::uint32_t context,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler, const ncclGin &gin, ncclTeam world,
    DirectFullRdmaState &state) {
  const auto mblocks = scheduler[3];
  if (mblocks > a.forward_direct_return_fragment_capacity)
    persistent_v3::publish_error_once(control.generation, 311ULL);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
      direct_fragment_tail(a, scheduler));
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
      producers_done(direct_mblocks_complete(a, scheduler));
  while (!cancelled(a, control)) {
    bool progressed = false;
    while (progress_direct_full_rdma_once(
        a, control, gin, world, context, completion, context_completion,
        scheduler, state)) {
      progressed = true;
#if MEGAMOE_DEBUG_VALIDATE
      atomicMax(&completion->bulk_pack_complete, 2000ULL + state.cursor);
#endif
    }
    if (producers_done.load(::cuda::memory_order_acquire) >= mblocks &&
        state.cursor >= tail.load(::cuda::memory_order_acquire))
      break;
    if (!progressed)
      if (!forward_reduce_detail::try_reduce_rdma_token(a, state.reduce_mailbox))
        __nanosleep(forward_tuning::kReturnIdleNanosleep);
  }

  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
      issued_count(direct_issued_counts(a, scheduler)[context]);
  issued_count.store(state.issued, ::cuda::memory_order_release);
  while (gin.readCounter(1, 56, ::cuda::memory_order_acquire) < state.issued &&
         !cancelled(a, control))
    if (!forward_reduce_detail::try_reduce_rdma_token(a, state.reduce_mailbox))
      __nanosleep(forward_tuning::kTransportIdleNanosleep);

  if (state.issued != 0ULL) {
    complete_completion_tasks(completion, transport_state::TaskClass::Return,
                              state.issued);
    atomicAdd(&completion->return_local_completion_completed, state.issued);
    const auto source_node = a.node == 0U ? 1U : 0U;
    const auto source_partition =
        source_node < a.node ? source_node : source_node - 1U;
    auto &record = context_completion[
        static_cast<std::uint64_t>(source_partition) * a.gin_context_count +
        context];
    atomicAdd(&record.return_completed, state.issued);
    atomicAdd(&record.inflight, 0ULL - state.issued);
    atomicAdd(&record.local_completion, state.issued);
    record.last_progress_globaltimer = device_globaltimer_ns();
  }
  MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnLocalCompletion, GinReturn,
                          context, state.issued, a.epoch);
  persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
      control.rdma_return_roles_complete.value, 1ULL);
}

} // namespace forward_return_detail

#endif // __has_include(<nccl_device.h>)
