#pragma once

// Included inside forward_unified_transport after its copy primitives.  The
// V4 single-node inference path owns Dispatch and Return by immutable CTA
// stripes.  Readiness is still published with release/acquire epochs, but no
// queue head, CAS, or work-stealing decision assigns a row to a worker.

// M128 is the compute/readiness unit, but it is too coarse as a Return copy
// unit for dense expert loads (eight local experts would activate only eight
// CTAs).  Choose an immutable power-of-two stripe count from the local plan's
// average M-block occupancy.  The choice depends only on frozen geometry, not
// peer distribution: sparse/decode plans retain one task per M block, while a
// nontrivial M block exposes 8- or 16-row copy tasks.  Dense M128 uses the
// wider stripe because doubling its notification count costs more than the
// extra copy parallelism saves.
__device__ __forceinline__ unsigned
static_v4_return_splits(const ForwardKernelArgs &a,
                        const unsigned long long *scheduler) {
  const auto mblocks = scheduler[3];
  if (!mblocks)
    return 1U;
  const auto average =
      (dispatch_sizes(a.dispatch).incoming_row_count + mblocks - 1ULL) /
      mblocks;
  return average > 64ULL ? 8U : average > 16ULL ? 16U : 1U;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool
static_v4_return_ready(const ForwardKernelArgs &a,
                       unsigned long long *scheduler, Storage &s) {
  constexpr auto threads = Storage::kCopyThreads;
  const unsigned tid = local_thread<BaseThread>();
  const auto splits = static_v4_return_splits(a, scheduler);
  const auto mblock = s.return_cursor / splits;
  const auto mblocks = scheduler[3];
  if (mblock >= mblocks)
    return false;

  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  const auto fc1_ntiles = static_cast<unsigned long long>(
      (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
      forward_tuning::kFc1TileN);
  auto *return_ready = claims + mblocks + mblocks * fc1_ntiles;
  if (tid == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
        return_ready[mblock]);
    s.return_rows =
        ready.load(::cuda::memory_order_acquire) >= execution_epoch(a) ? 1U
                                                                       : 0U;
  }
  copy_sync<threads>();
  const bool ready = s.return_rows != 0U;
  copy_sync<threads>();
  return ready;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void
static_v4_copy_return(const ForwardKernelArgs &a,
                      const unsigned long long *scheduler, Storage &s) {
  constexpr auto threads = Storage::kCopyThreads;
  constexpr auto warps = Storage::kCopyWarps;
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U;
  const unsigned lane = tid & 31U;
  const auto task = s.return_cursor;
  const auto splits = static_v4_return_splits(a, scheduler);
  const auto split_rows = forward_tuning::kTileM / splits;
  const auto mblock = task / splits;
  const auto split = static_cast<unsigned>(task % splits);

  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  const auto mapping = claims[mblock];
  const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
  const auto local_mblock = static_cast<std::uint32_t>(mapping);
  const auto block_rows = forward_mblock_valid_rows(a, expert, local_mblock);
  const auto split_begin = split * split_rows;
  const auto rows = split_begin < block_rows
                        ? min(split_rows, block_rows - split_begin)
                        : 0U;
  const auto row = a.dispatch.expert_row_prefix[expert] +
                   forward_mblock_local_row(local_mblock) + split_begin;
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  const auto *output = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);

  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, LocalReturnBatchStart,
                             LocalReturn, blockIdx.x, rows,
                             execution_epoch(a));
  if (tid == 0U)
    atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
  if (lane == 0U) {
    RowPrefetchPipeline<Storage, forward_tuning::kUnifiedRowPrefetch>
        pipeline(s, warp);
    for (unsigned item = warp; item < rows; item += warps) {
      const auto header = headers[row + item];
      auto *target = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.contribution_offset);
      pipeline.copy(output + (row + item) * a.hidden,
                    target + header.transport_index * a.hidden,
                    unsigned(a.hidden * sizeof(transport::BFloat16)));
    }
    pipeline.drain();
    finish_copies();
  }
  copy_sync<threads>();
  s.return_rows = rows;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ void
static_v4_publish_return(const ForwardKernelArgs &a,
                         const unsigned long long *scheduler, Storage &s) {
  constexpr auto threads = Storage::kCopyThreads;
  const unsigned tid = local_thread<BaseThread>();
  const auto task = s.return_cursor;
  const auto splits = static_v4_return_splits(a, scheduler);
  const auto split_rows = forward_tuning::kTileM / splits;
  const auto mblock = task / splits;
  const auto split = static_cast<unsigned>(task % splits);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  const auto mapping = claims[mblock];
  const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
  const auto local_mblock = static_cast<std::uint32_t>(mapping);
  const auto row = a.dispatch.expert_row_prefix[expert] +
                   forward_mblock_local_row(local_mblock) +
                   split * split_rows;
  const auto rows = s.return_rows;
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  const auto local = a.rank % a.local_world_size;
  if (!rows) {
    if (tid == 0U)
      s.return_cursor += a.forward_relay_cta_count;
    copy_sync<threads>();
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, LocalReturnBatchEnd,
                               LocalReturn, blockIdx.x, 0U,
                               execution_epoch(a));
    return;
  }

  // Reuse the copy-domain staging only after all TMA descriptors and tickets
  // from static_v4_copy_return are dead.  The M block contains at most 128
  // rows, so the compact peer partition fits without a second work queue.
  auto *positions = reinterpret_cast<unsigned *>(s.staging);
  auto *counts = positions + communication::kLocalChunkRows;
  auto *sequences = reinterpret_cast<unsigned long long *>(
      counts + communication::kLocalPeers);
  if (tid < communication::kLocalPeers)
    counts[tid] = 0U;
  copy_sync<threads>();
  for (unsigned begin = 0U; begin < rows; begin += threads) {
    const auto item = begin + tid;
    const auto peer = item < rows
                          ? headers[row + item].source_rank % a.local_world_size
                          : communication::kLocalPeers;
    const auto lane = tid & 31U;
    const auto peers = __match_any_sync(0xffffffffU, peer);
    const auto leader = unsigned(__ffs(peers) - 1);
    unsigned base = 0U;
    if (lane == leader && peer < a.local_world_size)
      base = atomicAdd(&counts[peer], unsigned(__popc(peers)));
    base = __shfl_sync(0xffffffffU, base, leader);
    if (item < rows)
      positions[item] = base + __popc(peers & ((1U << lane) - 1U));
  }
  copy_sync<threads>();
  if (tid < a.local_world_size && tid != local && counts[tid])
    sequences[tid] = local_notification_detail::channel(
                         a, a.local_base, tid, blockIdx.x)
                         .sent;
  copy_sync<threads>();
  if (tid < 32U) {
    for (unsigned peer = 0U; peer < a.local_world_size; ++peer)
      if (peer != local && counts[peer])
        local_notification_detail::wait_credit<ForwardKernelArgs, false>(
            a, peer, sequences[peer]);
  }
  copy_sync<threads>();
  for (unsigned item = tid; item < rows; item += threads) {
    const auto header = headers[row + item];
    const auto peer = header.source_rank % a.local_world_size;
    if (peer == local) {
      local_notification_detail::Receiver<ForwardKernelArgs>::apply(
          a, header.route_index);
    } else {
      auto &slot = local_notification_detail::channel(
                       a,
                       a.peer_bases[a.node * a.local_world_size + peer], local,
                       blockIdx.x)
                       .slots[sequences[peer] %
                              local_notification_detail::kDepth];
      slot.rows[positions[item]] = header.route_index;
    }
  }
  __threadfence_system();
  copy_sync<threads>();
  if (tid == 0U) {
    unsigned remote_rows = 0U;
    for (unsigned peer = 0U; peer < a.local_world_size; ++peer) {
      if (peer == local || !counts[peer])
        continue;
      const auto sequence = sequences[peer];
      auto &slot = local_notification_detail::channel(
                       a,
                       a.peer_bases[a.node * a.local_world_size + peer], local,
                       blockIdx.x)
                       .slots[sequence % local_notification_detail::kDepth];
      slot.count = counts[peer];
      slot.kind = local_notification_detail::Kind::Return;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          slot.ready)
          .store(sequence / local_notification_detail::kDepth + 1ULL,
                 ::cuda::memory_order_release);
      local_notification_detail::channel(a, a.local_base, peer, blockIdx.x)
          .sent = sequence + 1ULL;
      remote_rows += counts[peer];
    }
    if (remote_rows)
      atomicAdd(&status->actual_return_peer_bytes,
                1ULL * remote_rows * a.hidden * sizeof(transport::BFloat16));
  }
  copy_sync<threads>();
  if (tid == 0U)
    s.return_cursor += a.forward_relay_cta_count;
  copy_sync<threads>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, LocalReturnBatchEnd,
                             LocalReturn, blockIdx.x, rows,
                             execution_epoch(a));
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool
static_v4_progress_return(const ForwardKernelArgs &a,
                          unsigned long long *scheduler, Storage &s) {
  // Lexically separate readiness, payload movement, and notification
  // publication so the copy pipeline's descriptor/ticket registers are dead
  // before the per-peer notification walk begins.
  if (!static_v4_return_ready<BaseThread>(a, scheduler, s))
    return false;
  static_v4_copy_return<BaseThread>(a, scheduler, s);
  static_v4_publish_return<BaseThread>(a, scheduler, s);
  return true;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool
static_v4_progress_dispatch(const ForwardKernelArgs &a, Storage &s) {
  // Keep this wrapper as a distinct lexical phase: Dispatch route metadata,
  // its two-slot copy pipeline and Return state are never needed together.
  return progress_dispatch<BaseThread>(a, s);
}

template <class Storage>
__device__ __forceinline__ void static_v4_receive_returns(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control, Storage &s) {
  const unsigned lane = threadIdx.x & 31U;
  while (true) {
    const bool received =
        local_notification_detail::progress<ForwardKernelArgs, true>(a);
    const unsigned done = lane == 0U
                              ? ::cuda::atomic_ref<
                                    unsigned, ::cuda::thread_scope_block>(
                                    s.copies_done)
                                    .load(::cuda::memory_order_acquire)
                              : 0U;
    if (__shfl_sync(0xffffffffU, done, 0U))
      break;
    if (!received && !cancelled(a, control))
      __nanosleep(forward_tuning::kTransportIdleNanosleep);
  }
}

template <unsigned Threads, unsigned BaseThread, unsigned SegmentBytes,
          unsigned Buffers, unsigned DispatchLanes>
__device__ __forceinline__ void static_v4_run_single_node(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    unsigned long long *scheduler,
    Shared<Threads, SegmentBytes, Buffers, DispatchLanes> &s) {
  using Storage = Shared<Threads, SegmentBytes, Buffers, DispatchLanes>;
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U;
  init<BaseThread>(a, control, s);
  if (tid == 0U)
    s.return_cursor = blockIdx.x;
  domain_sync<Storage::kThreads>();

  if (warp == Storage::kCopyWarps) {
    static_v4_receive_returns(a, control, s);
  } else {
    while (true) {
      const bool dispatched = static_v4_progress_dispatch<BaseThread>(a, s);
      const bool returned = static_v4_progress_return<BaseThread>(a, scheduler, s);
      if (tid == 0U) {
        if (!s.dispatch_complete && s.dispatch_cursor >= s.dispatch_end) {
          s.dispatch_complete = 1U;
          const auto previous =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control.relay_dispatch_roles_complete.value, 1ULL);
          if (previous + 1ULL == a.forward_relay_cta_count)
            peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                ->dispatch_complete_clock = device_globaltimer_ns();
        }
        s.stop = cancelled(a, control) ||
                 (s.dispatch_complete &&
                  s.return_cursor >=
                      scheduler[3] * static_v4_return_splits(a, scheduler));
      }
      copy_sync<Storage::kCopyThreads>();
      if (s.stop)
        break;
      if (!dispatched && !returned)
        __nanosleep(forward_tuning::kTransportIdleNanosleep);
    }
    if (warp == 0U)
      drain_notifications<BaseThread, false>(
          a, [&] { return cancelled(a, control); });
    copy_sync<Storage::kCopyThreads>();
    if (tid == 0U) {
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->return_complete_clock,
                device_globaltimer_ns());
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.rdma_return_roles_complete.value, 1ULL);
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.return_receive_roles_complete.value, 1ULL);
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.copies_done)
          .store(1U, ::cuda::memory_order_release);
    }
  }
  domain_sync<Storage::kThreads>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, TransportRoleDone,
                             LocalReturn, blockIdx.x, 0ULL,
                             execution_epoch(a));
}
