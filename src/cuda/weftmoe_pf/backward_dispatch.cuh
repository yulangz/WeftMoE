#pragma once

// Included in backward_unified_transport::gin_detail. Keep the registered NIC
// RX slot until every destination has pulled its rows into private dR. This is
// the same ready/consumed contract as Forward's packet fanout, with one static
// receiving CTA and one statically paired reader CTA per local rank.
template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool publish_dispatch(const BackwardKernelArgs &a,
                                                Storage &s) {
  const auto tid = threadIdx.x - BaseThread;
  auto *slots = a.peer_payloads[a.rank].dispatch_slots +
                blockIdx.x * ring::kRingDepth;
  // The ACKs live on different GPUs. Probe them concurrently instead of
  // serializing eight system-scope loads in the issuing lane.
  const auto tail = s.dispatch_tail;
  unsigned consumed = 0U;
  if (tid < 32U) {
    auto &slot = slots[tail % ring::kRingDepth];
    const bool pending = tail != s.dispatch_head && tid < a.local_world_size;
    const bool ready = !pending ||
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
            slot.consumed[tid]).load(::cuda::memory_order_acquire) == slot.ready;
    consumed = __all_sync(0xffffffffU, ready);
  }
  if (tid == 0U) {
    s.gin_progressed = 0U;
    if (tail != s.dispatch_head && consumed) {
      wire::finish_receive(a, s.dispatch_hold[tail % ring::kRingDepth]);
      ++s.dispatch_tail;
      s.gin_progressed = 1U;
    }
    const auto index = s.dispatch_head % ring::kRingDepth;
    auto &task = s.dispatch_hold[index];
    if (s.dispatch_head - s.dispatch_tail < ring::kRingDepth &&
        wire::claim_receive(a, s.gin_rx_dispatch, task, true)) {
      if (!task.rows) {
        wire::finish_receive(a, task);
      } else {
        auto &slot = slots[index];
        slot.channel = task.channel;
        slot.sequence = task.sequence;
        slot.rows = task.rows;
        const auto generation = slot.ready + 1ULL;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(slot.ready)
            .store(generation, ::cuda::memory_order_release);
        ++s.dispatch_head;
      }
      s.gin_progressed = 1U;
    }
    if (s.dispatch_head == s.dispatch_tail &&
        wire::receive_complete(a, true))
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(slots[0].finished)
          .store(execution_epoch(a), ::cuda::memory_order_release);
  }
  copy_sync<Storage::kCopyThreads>();
  const bool progressed = s.gin_progressed;
  copy_sync<Storage::kCopyThreads>();
  return progressed;
}

template <unsigned BaseThread, class Storage>
__device__ __forceinline__ bool pull_dispatch(const BackwardKernelArgs &a,
                                             Storage &s) {
  const auto tid = threadIdx.x - BaseThread;
  const auto warp = tid / 32U, lane = tid % 32U;
  const auto local = a.rank % a.local_world_size;
  if (warp == 0U) {
    const auto peer = (s.pull_peer + lane) % a.local_world_size;
    unsigned long long generation = 0ULL;
    if (lane < a.local_world_size) {
      auto &slot = a.peer_payloads[a.node * a.local_world_size + peer]
          .dispatch_slots[blockIdx.x * ring::kRingDepth +
                          s.pull_cursor[peer] % ring::kRingDepth];
      generation = ::cuda::atomic_ref<unsigned long long,
                                      ::cuda::thread_scope_system>(slot.ready)
                       .load(::cuda::memory_order_acquire);
      if (generation == slot.consumed[local])
        generation = 0ULL;
    }
    const auto mask = __ballot_sync(0xffffffffU, generation != 0ULL);
    const auto selected = mask ? __ffs(mask) - 1U : 0U;
    const auto sequence = __shfl_sync(0xffffffffU, generation, selected);
    const auto selected_peer = __shfl_sync(0xffffffffU, peer, selected);
    if (lane == 0U) {
      s.pull_sequence = sequence;
      s.pull_peer = selected_peer;
    }
  }
  copy_sync<Storage::kCopyThreads>();
  const auto generation = s.pull_sequence;
  if (generation) {
    const auto peer = a.node * a.local_world_size + s.pull_peer;
    auto &slot = a.peer_payloads[peer].dispatch_slots[
        blockIdx.x * ring::kRingDepth + s.pull_cursor[s.pull_peer] % ring::kRingDepth];
    const auto *chunk = static_cast<const char *>(ncclGetLsaPointer(
        a.gin_window, a.gin_rings.payload(slot.channel, slot.sequence, true),
        s.pull_peer)) +
        sizeof(ring::ChunkHeader);
    const auto items = slot.rows * a.dispatch.identity.top_k;
    asm volatile("fence.proxy.async.global;" ::: "memory");
    forward_unified_transport::RowPrefetchPipeline<Storage> prefetch(s, warp);
    for (unsigned begin = warp * 32U; begin < items;
         begin += Storage::kCopyThreads) {
      const auto item = begin + lane;
      const auto *record = chunk + (item / a.dispatch.identity.top_k) *
                                     a.gin_rings.dispatch_row_bytes;
      const auto index = item % a.dispatch.identity.top_k;
      bool selected = false;
      unsigned long long row = 0ULL;
      if (item < items &&
          index < reinterpret_cast<const ring::DispatchRow *>(record)->fanouts) {
        const auto target = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
            record + sizeof(ring::DispatchRow))[index];
        selected = target.destination_local_rank == local;
        row = target.dst_row;
      }
      auto mask = __ballot_sync(0xffffffffU, selected);
      while (mask) {
        const auto issuer = unsigned(__ffs(mask) - 1);
        const auto destination = __shfl_sync(0xffffffffU, row, issuer);
        const auto address = __shfl_sync(0xffffffffU,
            reinterpret_cast<unsigned long long>(record), issuer);
        if (lane == 0U)
          prefetch.copy(reinterpret_cast<const char *>(address) +
                            a.gin_rings.dispatch_row_bytes - a.hidden * 2ULL,
                        a.dr + destination * a.hidden, unsigned(a.hidden * 2ULL));
        mask &= mask - 1U;
      }
    }
    if (lane == 0U) {
      prefetch.drain();
      finish_copies();
    }
    copy_sync<Storage::kCopyThreads>();
    for (unsigned item = tid; item < items; item += Storage::kCopyThreads) {
      const auto *record = chunk + (item / a.dispatch.identity.top_k) *
                                     a.gin_rings.dispatch_row_bytes;
      const auto index = item % a.dispatch.identity.top_k;
      if (index >= reinterpret_cast<const ring::DispatchRow *>(record)->fanouts)
        continue;
      const auto target = reinterpret_cast<const ForwardDispatchFanoutRecord *>(
          record + sizeof(ring::DispatchRow))[index];
      if (target.destination_local_rank == local)
        local_notification_detail::Receiver<BackwardKernelArgs>::publish_dispatch(
            a, target.dst_row, local);
    }
    copy_sync<Storage::kCopyThreads>();
    if (tid == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(slot.consumed[local])
          .store(generation, ::cuda::memory_order_release);
      ++s.pull_cursor[s.pull_peer];
    }
  }
  copy_sync<Storage::kCopyThreads>();
  if (tid == 0U)
    s.pull_peer = (s.pull_peer + 1U) % a.local_world_size;
  copy_sync<Storage::kCopyThreads>();
  return generation != 0ULL;
}

__device__ __forceinline__ bool pull_terminal(const BackwardKernelArgs &a) {
  const auto first = a.node * a.local_world_size;
  for (unsigned peer = first; peer < first + a.local_world_size; ++peer)
    if (::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
            a.peer_payloads[peer].dispatch_slots[blockIdx.x * ring::kRingDepth].finished)
            .load(::cuda::memory_order_acquire) != execution_epoch(a))
      return false;
  return true;
}
