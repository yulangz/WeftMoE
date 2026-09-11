#pragma once

// FC2 publishes canonical row IDs. Relay consumes the local queue; each fixed
// GIN owner consumes its remote queue. A separate arrival queue carries compact
// row IDs back to the source. No worker searches compute tiles or chunk flags.
#if __has_include(<nccl_device.h>)
namespace forward_return_detail {

using ReturnCopyBarrier = ::cuda::barrier<::cuda::thread_scope_block>;
using RowQueue = persistent_v3::ReturnRowQueue;
using PeerQueue = persistent_v3::ReturnPeerQueue;

__device__ __forceinline__ bool
cancelled(const ForwardKernelArgs &a,
          persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

__device__ __forceinline__ PeerQueue *peer_queues(const ForwardKernelArgs &a) {
  return peer_at<PeerQueue>(a.local_base, a.forward_return_peers_offset);
}

__device__ __forceinline__ unsigned long long
source_compact_rows(const ForwardKernelArgs &a, const unsigned long long *) {
  const auto *peers = peer_queues(a);
  unsigned long long rows = 0ULL;
  for (unsigned int peer = 0U; peer < a.world; ++peer)
    rows += peers[peer].receive_count;
  return rows;
}

__device__ __forceinline__ unsigned int queue_index(const ForwardKernelArgs &a,
                                                    unsigned int source) {
  return source / a.local_world_size == a.node
             ? 0U
             : 1U + source % a.local_world_size;
}

// Count-derived disjoint intervals fit exactly in incoming_rows and routes.
// The mapping supports arbitrary expert placement and uneven route counts.
__device__ __forceinline__ void
initialize_queues(const ForwardKernelArgs &a,
                  persistent_v3::ForwardPipelineControl &control,
                  unsigned long long *scheduler) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    unsigned long long begin = 0ULL;
    for (unsigned int q = 0U; q <= a.gin_context_count; ++q) {
      auto &queue = control.return_rows[q];
      queue.begin = begin;
      for (unsigned int source = 0U; source < a.world; ++source)
        if (queue_index(a, source) == q)
          queue.capacity += scheduler[4U + source] >> 32U;
      begin += queue.capacity;
    }
  }
  auto *peers = peer_queues(a);
  for (unsigned int peer = blockIdx.x * blockDim.x + threadIdx.x;
       peer < a.world; peer += gridDim.x * blockDim.x) {
    PeerQueue result{};
    if (a.use_gin && peer / a.local_world_size != a.node) {
      for (unsigned int expert = 0U; expert < a.dispatch.identity.num_experts;
           ++expert) {
        const auto owner = a.dispatch.expert_to_rank[expert];
        const auto received =
            a.dispatch
                .source_expert_count[a.rank * a.dispatch.identity.num_experts +
                                     expert];
        if (owner / a.local_world_size != a.node && owner < peer)
          result.receive_begin += received;
        if (owner == peer)
          result.receive_count += received;
        if (owner / a.local_world_size != peer / a.local_world_size &&
            owner < a.rank)
          result.send_begin +=
              a.dispatch
                  .source_expert_count[peer * a.dispatch.identity.num_experts +
                                       expert];
      }
    }
    peers[peer] = result;
  }
}

__device__ __forceinline__ unsigned long long
claim_rows(RowQueue &queue, unsigned int limit, unsigned int &count) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> published(
      queue.confirmed),
      head(queue.head);
  const auto end = published.load(::cuda::memory_order_acquire);
  auto begin = head.load(::cuda::memory_order_relaxed);
  count = 0U;
  if (begin >= end)
    return begin;
  const auto n = static_cast<unsigned int>(min(end - begin, 1ULL * limit));
  if (head.compare_exchange_strong(begin, begin + n,
                                   ::cuda::memory_order_relaxed))
    count = n;
  return begin;
}

// Called by all math threads after the last FC2 N-tile acquires every store.
// The dead GEMM operand storage supplies a few prefix words. Reserve/fill/
// confirm each queue before touching the next, so producers cannot form a
// cyclic wait between queues. CAS confirms exactly this producer's interval.
template <unsigned int MathThreads>
__device__ __noinline__ void
publish_rows(const ForwardKernelArgs &a, unsigned long long *,
             unsigned long long row, unsigned int rows, unsigned int *scratch) {
  constexpr auto warps = MathThreads / 32U;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  auto *entries =
      peer_at<unsigned int>(a.local_base, a.forward_return_rows_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto q = threadIdx.x < rows
                     ? queue_index(a, headers[row + threadIdx.x].source_rank)
                     : UINT32_MAX;
  // A notification holds a full 32-bit compact row and a contiguous suffix
  // length. The length is a transfer hint, never a task or readiness flag;
  // consumers may split confirmed intervals at arbitrary row boundaries.
  const auto *mapping = peer_at<const unsigned long long>(
      a.local_base, a.forward_return_row_map_offset);
  unsigned int compact = 0U;
  bool run_end = false;
  if (q != 0U && q != UINT32_MAX) {
    compact = static_cast<unsigned int>(
        return_mapping_compact_row(mapping[row + threadIdx.x]));
    run_end = threadIdx.x + 1U == rows ||
              headers[row + threadIdx.x + 1U].source_rank !=
                  headers[row + threadIdx.x].source_rank ||
              return_mapping_compact_row(mapping[row + threadIdx.x + 1U]) !=
                  compact + 1ULL;
  }
  const auto ends = __ballot_sync(0xffffffffU, run_end);
  if (lane == 0U)
    scratch[32U + warp] = ends;
  if (threadIdx.x == 0U)
    scratch[warps] = 0U;
  cutlass::arch::NamedBarrier::sync(MathThreads, 2U);
  const auto active =
      __reduce_or_sync(0xffffffffU, q != UINT32_MAX ? 1U << q : 0U);
  if (lane == 0U)
    atomicOr(scratch + warps, active);
  cutlass::arch::NamedBarrier::sync(MathThreads, 3U);
  const auto queues = scratch[warps];
  auto *base = reinterpret_cast<unsigned long long *>(scratch + 16U);
  for (unsigned int index = 0U; index <= a.gin_context_count; ++index) {
    if ((queues & (1U << index)) == 0U)
      continue;
    const auto mask = __ballot_sync(0xffffffffU, q == index);
    if (lane == 0U)
      scratch[warp] = __popc(mask);
    cutlass::arch::NamedBarrier::sync(MathThreads, 2U);
    auto &queue = control->return_rows[index];
    if (threadIdx.x == 0U) {
      unsigned int count = 0U;
      for (unsigned int w = 0U; w < warps; ++w) {
        const auto n = scratch[w];
        scratch[w] = count;
        count += n;
      }
      scratch[warps + 1U] = count;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          reserve(queue.reserved);
      *base = reserve.fetch_add(count, ::cuda::memory_order_relaxed);
      if (*base + count > queue.capacity)
        persistent_v3::publish_error_once(control->generation, 310ULL);
      // Every math warp must leave through the same barrier boundary, even
      // when host cancellation arrives while this interval is being reserved.
      scratch[warps + 2U] = cancelled(a, *control);
    }
    cutlass::arch::NamedBarrier::sync(MathThreads, 3U);
    if (scratch[warps + 2U] != 0U)
      return;
    if (q == index) {
      const auto offset = scratch[warp] + __popc(mask & ((1U << lane) - 1U));
      entries[queue.begin + *base + offset] =
          static_cast<unsigned int>(row + threadIdx.x);
      if (index != 0U) {
        auto boundary = ends & (0xffffffffU << lane);
        unsigned int next_warp = warp;
        while (boundary == 0U)
          boundary = scratch[32U + ++next_warp];
        const auto length = next_warp * 32U + __ffs(boundary) - threadIdx.x;
        auto *send_rows = reinterpret_cast<unsigned long long *>(
            static_cast<std::byte *>(a.gin_window_base) +
            a.gin_return_send_rows_offset);
        send_rows[queue.begin + *base + offset] =
            compact | (static_cast<unsigned long long>(length) << 32U);
      }
    }
    __threadfence();
    cutlass::arch::NamedBarrier::sync(MathThreads, 2U);
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          confirm(queue.confirmed);
      auto expected = *base;
      while (!confirm.compare_exchange_weak(
          expected, *base + scratch[warps + 1U], ::cuda::memory_order_acq_rel,
          ::cuda::memory_order_relaxed)) {
        if (cancelled(a, *control))
          break;
        expected = *base;
        __nanosleep(32U);
      }
    }
    cutlass::arch::NamedBarrier::sync(MathThreads, 3U);
  }
  if (threadIdx.x == 0U)
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control->return_producers_complete.value, 1ULL);
}

struct LocalReturnState {
  unsigned long long cursor = 0ULL;
  unsigned int selected = 0U;
};

__device__ __forceinline__ bool
local_return_terminal(const ForwardKernelArgs &a, unsigned long long *scheduler,
                      const LocalReturnState &) {
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  return persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control->return_producers_complete.value) >= scheduler[3] &&
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control->return_rows[0].head) == control->return_rows[0].capacity;
}

__device__ __forceinline__ bool progress_local_return(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    unsigned int context, unsigned long long *, LocalReturnState &state,
    std::byte *tma_storage, ReturnCopyBarrier *tma_full) {
  if (threadIdx.x == 0U) {
    state.selected = 0U;
    if (!cancelled(a, control))
      state.cursor =
          claim_rows(control.return_rows[0], a.forward_local_return_batch_rows,
                     state.selected);
  }
  __syncthreads();
  if (state.selected == 0U)
    return false;
  WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnBatchStart, LocalReturn,
                          context, state.selected, execution_epoch(a));
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (threadIdx.x == 0U)
    atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
  const auto *entries =
      peer_at<const unsigned int>(a.local_base, a.forward_return_rows_offset) +
      control.return_rows[0].begin;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *output =
      a.use_gin && a.forward_registered_output
          ? reinterpret_cast<const transport::BFloat16 *>(
                static_cast<const std::byte *>(a.gin_window_base) +
                a.gin_return_staging_offset)
          : peer_at<const transport::BFloat16>(a.local_base,
                                               a.forward_output_rows_offset);
  const auto copy_bytes = static_cast<unsigned int>(a.hidden * 2ULL);
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const bool tma = a.forward_relay_tma_warps != 0U && copy_bytes <= 16384U &&
                   (copy_bytes & 15U) == 0U;
  const auto workers = tma ? a.forward_relay_tma_warps : blockDim.x / 32U;
  bool copied = false;
  for (unsigned int item = warp; warp < workers && item < state.selected;
       item += workers) {
    const auto row = entries[state.cursor + item];
    const auto header = headers[row];
    const auto *src = output + row * a.hidden;
    auto *dst = peer_at<transport::BFloat16>(a.peer_bases[header.source_rank],
                                             a.contribution_offset) +
                header.route_index * a.hidden;
    if (tma) {
      if (lane == 0U) {
        auto *storage = tma_storage + warp * copy_bytes;
        auto &full = tma_full[warp];
        ::cuda::device::experimental::cp_async_bulk_global_to_shared(
            storage, src, copy_bytes, full);
        auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, copy_bytes);
        full.wait(static_cast<decltype(ticket) &&>(ticket));
        ::cuda::device::experimental::cp_async_bulk_shared_to_global(
            dst, storage, copy_bytes);
        ::cuda::device::experimental::cp_async_bulk_commit_group();
        ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
        copied = true;
      }
    } else {
      for (unsigned long long v = lane; v < a.hidden / 8ULL; v += 32ULL)
        reinterpret_cast<uint4 *>(dst)[v] =
            reinterpret_cast<const uint4 *>(src)[v];
    }
  }
  if (copied)
    ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
  __threadfence_system();
  __syncthreads();
  unsigned int peer_rows = 0U;
  for (unsigned int item = threadIdx.x; item < state.selected;
       item += blockDim.x) {
    const auto header = headers[entries[state.cursor + item]];
    auto *source_base = a.peer_bases[header.source_rank];
    if (a.forward_dynamic_reduce_queue)
      forward_reduce_detail::publish_ready_route(a, source_base,
                                                 header.route_index, 2U);
    else {
      auto *ready =
          peer_at<unsigned long long>(source_base, a.return_ready_epoch_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          ready[header.route_index])
          .store(execution_epoch(a), ::cuda::memory_order_release);
    }
    peer_rows += header.source_rank != a.rank;
  }
  const auto copied_peer_rows = __reduce_add_sync(0xffffffffU, peer_rows);
  if (lane == 0U && copied_peer_rows != 0U)
    atomicAdd(&status->actual_return_peer_bytes,
              copied_peer_rows * a.hidden * 2ULL);
  __syncthreads();
  WEFTMOE_TIMELINE_RECORD(a.timeline, LocalReturnBatchEnd, LocalReturn,
                          context, state.selected, execution_epoch(a));
  return true;
}

struct ReturnReceiveState {
  unsigned long long expected = 0ULL;
  unsigned long long completed = 0ULL;
  unsigned long long begin = 0ULL;
  unsigned int selected_rows = 0U;
  unsigned int peer = 0U;
  unsigned int initialized = 0U;
};

__device__ __forceinline__ bool
return_receive_terminal(const ReturnReceiveState &state) {
  return state.initialized && state.completed == state.expected;
}

__device__ __forceinline__ bool progress_return_receive(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    unsigned int, unsigned long long *, ReturnReceiveState &state) {
  auto *peers = peer_queues(a);
  if (threadIdx.x == 0U) {
    if (!state.initialized) {
      for (unsigned int peer = 0U; peer < a.world; ++peer)
        state.expected += peers[peer].receive_count;
      state.peer = blockIdx.x % a.world;
      state.initialized = 1U;
    }
    state.completed = persistent_v3::load_acquire<::cuda::thread_scope_device>(
        control.return_rows_complete.value);
    state.selected_rows = 0U;
    const auto sender = state.peer;
    state.peer = (sender + 1U) % a.world;
    auto &queue = peers[sender];
    if (queue.receive_count != 0ULL && state.completed < state.expected) {
      // Match the sender's physical-context permutation used for port balance.
      const auto physical =
          a.gin_context_count +
          (a.rank % a.local_world_size + sender) % a.gin_context_count;
      const ncclGin gin(a.gin_dev_comm, physical);
      const auto ready =
          gin.readSignal(3U + sender, 64, ::cuda::memory_order_acquire);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> head(
          queue.receive_head);
      auto begin = head.load(::cuda::memory_order_relaxed);
      if (ready > queue.receive_count)
        persistent_v3::publish_error_once(control.generation, 313ULL);
      else if (begin < ready) {
        const auto n = min(ready - begin, 1ULL * a.forward_return_batch_rows);
        if (head.compare_exchange_strong(begin, begin + n,
                                         ::cuda::memory_order_relaxed)) {
          state.begin = queue.receive_begin + begin;
          state.selected_rows = static_cast<unsigned int>(n);
        }
      }
    }
    if (cancelled(a, control))
      state.selected_rows = 0U;
  }
  __syncthreads();
  if (!state.selected_rows)
    return false;
  asm volatile("fence.proxy.async.global;" : : : "memory");
  const auto *entries = reinterpret_cast<const unsigned long long *>(
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_return_receive_rows_offset);
  const auto *routes = peer_at<const unsigned long long>(
      a.local_base, a.forward_return_row_to_source_route_offset);
  for (unsigned int item = threadIdx.x; item < state.selected_rows;
       item += blockDim.x) {
    const auto route = routes[entries[state.begin + item] & 0xffffffffULL];
    if (a.forward_dynamic_reduce_queue)
      forward_reduce_detail::publish_ready_route(a, a.local_base, route, 1U);
    else {
      auto *ready = peer_at<unsigned long long>(a.local_base,
                                                a.return_ready_epoch_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          ready[route])
          .store(execution_epoch(a), ::cuda::memory_order_release);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
    state.completed =
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.return_rows_complete.value, state.selected_rows) +
        state.selected_rows;
  }
  __syncthreads();
  return true;
}

struct ReturnRdmaState {
  unsigned long long cursor = 0ULL;
  unsigned long long end = 0ULL;
  unsigned long long issued = 0ULL;
  unsigned long long completed = 0ULL;
  unsigned long long notifications = 0ULL;
  unsigned long long notify_begin = 0ULL;
  unsigned int source = 0U;
};

// Each payload has its own local counter. Notification puts use counter 0,
// leaving return_max_inflight expressed in payloads as before this refactor.
__device__ __forceinline__ void
record_return_operation(const ForwardKernelArgs &a, unsigned int source,
                        unsigned int context,
                        transport_state::GenerationCompletionVector *completion,
                        transport_state::ContextCompletionRecord *contexts) {
  const auto node = source / a.local_world_size;
  const auto partition = node < a.node ? node : node - 1U;
  auto &record = contexts[partition * a.gin_context_count + context];
  atomicAdd(&record.return_expected, 1ULL);
  atomicAdd(&record.return_claimed, 1ULL);
  atomicAdd(&record.inflight, 1ULL);
  atomicAdd(
      &completion_task(completion, transport_state::TaskClass::Return).expected,
      1ULL);
  claim_completion_tasks(completion, transport_state::TaskClass::Return);
  atomicAdd(&completion->return_local_completion_expected, 1ULL);
}

__device__ __forceinline__ bool progress_return_send(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    const ncclGin &gin, ncclTeam world, unsigned int context,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *, ReturnRdmaState &state) {
  auto &queue = control.return_rows[1U + context];
  const auto *entries =
      peer_at<const unsigned int>(a.local_base, a.forward_return_rows_offset) +
      queue.begin;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *notifications =
      reinterpret_cast<const unsigned long long *>(
          static_cast<const std::byte *>(a.gin_window_base) +
          a.gin_return_send_rows_offset) +
      queue.begin;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  // One contiguous metadata put confirms several possibly discontiguous
  // payload puts. Its strong action covers only this context and peer.
  if (state.notify_begin < state.cursor &&
      (state.cursor == state.end ||
       headers[entries[state.cursor]].source_rank != state.source)) {
    auto &peer = peer_queues(a)[state.source];
    const auto rows = state.cursor - state.notify_begin;
    gin.put(world, state.source, a.gin_window,
            a.gin_return_receive_rows_offset +
                (peer.send_begin + peer.sent) * sizeof(unsigned long long),
            a.gin_window,
            a.gin_return_send_rows_offset +
                (queue.begin + state.notify_begin) * sizeof(unsigned long long),
            rows * sizeof(unsigned long long),
            ncclGin_StrongSignalAdd{3U + a.rank, rows}, ncclGin_CounterInc{0},
            ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_block,
            ::cuda::thread_scope_device);
    peer.sent += rows;
    ++state.notifications;
    state.notify_begin = state.cursor;
    record_return_operation(a, state.source, context, completion,
                            context_completion);
    atomicAdd(&status->gin_signal_count, 1ULL);
    atomicAdd(&status->gin_metadata_bytes,
              rows * sizeof(unsigned long long));
    return true;
  }
  if (a.forward_return_max_inflight != 0U &&
      state.issued - state.completed >= a.forward_return_max_inflight) {
    state.completed = gin.readCounter(1, 56, ::cuda::memory_order_acquire);
    if (state.issued - state.completed >= a.forward_return_max_inflight)
      return false;
  }
  if (state.cursor == state.end) {
    unsigned int count = 0U;
    state.cursor = claim_rows(queue, a.forward_return_batch_rows, count);
    state.end = state.cursor + count;
    state.notify_begin = state.cursor;
    if (!count)
      return false;
  }
  const auto first = entries[state.cursor];
  const auto source = headers[first].source_rank;
  const auto compact = notifications[state.cursor] & 0xffffffffULL;
  state.source = source;
  unsigned int rows = 0U;
  do {
    const auto entry = notifications[state.cursor + rows];
    const auto length = static_cast<unsigned int>(
        min(state.end - state.cursor - rows, 1ULL * (entry >> 32U)));
    if (length == 0U) {
      persistent_v3::publish_error_once(control.generation, 314ULL);
      return false;
    }
    rows += length;
    if (state.cursor + rows == state.end)
      break;
    const auto next = entries[state.cursor + rows];
    if (next != first + rows || headers[next].source_rank != source ||
        (notifications[state.cursor + rows] & 0xffffffffULL) != compact + rows)
      break;
  } while (true);
  const auto bytes = rows * a.hidden * sizeof(transport::BFloat16);
  WEFTMOE_TIMELINE_RECORD(a.timeline, ReturnGinPutIssue, GinReturn, context,
                          bytes, execution_epoch(a));
  gin.put(world, source, a.gin_window,
          a.gin_return_receive_offset + compact * a.hidden * 2ULL, a.gin_window,
          a.gin_return_staging_offset + first * a.hidden * 2ULL, bytes,
          ncclGin_None{}, ncclGin_CounterInc{1}, ncclCoopThread{},
          ncclGin_None{}, ::cuda::thread_scope_block,
          ::cuda::thread_scope_device);
  state.cursor += rows;
  ++state.issued;
  record_return_operation(a, source, context, completion, context_completion);
  atomicAdd(&status->gin_return_put_count, 1ULL);
  atomicAdd(&status->gin_return_payload_bytes, bytes);
  return true;
}

__device__ __forceinline__ void
drain_return_send(const ForwardKernelArgs &a,
                  persistent_v3::ForwardPipelineControl &control,
                  unsigned int context,
                  transport_state::GenerationCompletionVector *completion,
                  transport_state::ContextCompletionRecord *context_completion,
                  unsigned long long *scheduler, const ncclGin &gin,
                  ncclTeam world, ReturnRdmaState &state) {
  auto &queue = control.return_rows[1U + context];
  while (!cancelled(a, control)) {
    if (progress_return_send(a, control, gin, world, context, completion,
                             context_completion, scheduler, state))
      continue;
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            control.return_producers_complete.value) >= scheduler[3] &&
        state.cursor == state.end && state.notify_begin == state.cursor &&
        queue.head == queue.capacity)
      break;
    __nanosleep(forward_tuning::kReturnIdleNanosleep);
  }
  while (gin.readCounter(1, 56, ::cuda::memory_order_acquire) < state.issued &&
         !cancelled(a, control))
    __nanosleep(forward_tuning::kTransportIdleNanosleep);
  while (gin.readCounter(0, 56, ::cuda::memory_order_acquire) <
             state.notifications &&
         !cancelled(a, control))
    __nanosleep(forward_tuning::kTransportIdleNanosleep);
  const auto operations = state.issued + state.notifications;
  complete_completion_tasks(completion, transport_state::TaskClass::Return,
                            operations);
  atomicAdd(&completion->return_local_completion_completed, operations);
  for (unsigned int partition = 0U;
       partition + 1U < a.dispatch.identity.num_nodes; ++partition) {
    auto &record =
        context_completion[partition * a.gin_context_count + context];
    record.return_completed = record.return_expected;
    record.local_completion = record.return_expected;
    record.inflight = 0ULL;
  }
  persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
      control.rdma_return_roles_complete.value, 1ULL);
}

} // namespace forward_return_detail
#endif
