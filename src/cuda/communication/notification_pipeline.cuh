#pragma once

#include "cuda/persistent_v3/forward/reduce_queue.cuh"

// Private launch-contract adapter. Receive handlers touch only local memory.
namespace local_notification_detail {
using Kind = communication::NotificationKind;
using Channel = communication::NotificationChannel;
constexpr unsigned kRows = communication::kLocalChunkRows;
constexpr unsigned kDepth = communication::kLocalRingDepth;

template <class Args> struct Receiver;

template <class Args>
__device__ __forceinline__ Channel &channel(const Args &a, void *base,
                                            unsigned peer, unsigned cta) {
  return peer_at<Channel>(
      base,
      a.local_notification_offset)[peer * a.local_notification_ctas + cta];
}

template <class Args> __global__ void reset(Args a) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= a.local_world_size * a.local_notification_ctas)
    return;
  auto &state =
      peer_at<Channel>(a.local_base, a.local_notification_offset)[index];
  state.sent = state.claimed = 0;
  state.ended = 0;
  for (auto &slot : state.slots)
    slot.ready = slot.credit = slot.consumed = 0;
}

// Each CTA consumes its matching incoming channels. The default permits
// cooperating consumers via CAS. Exclusive ownership removes that arbitration;
// callers must keep exactly one complete receive warp live until channels drain.
// A transient activemask is not an ownership mask under independent thread
// scheduling: subsets of the same warp would otherwise elect extra consumers.
template <class Args, bool SingleConsumer = false>
__device__ __forceinline__ bool progress(const Args &a,
                                         unsigned peer = UINT32_MAX) {
  const auto mask = SingleConsumer ? 0xffffffffU : __activemask();
  const auto lane = threadIdx.x % 32;
  const auto leader = unsigned(__ffs(mask) - 1);
  const auto width = __popc(mask);
  const auto ordinal = __popc(mask & ((1U << lane) - 1));
  const auto local = a.rank % a.local_world_size;
  unsigned selected = a.local_world_size;
  unsigned long long seq = 0;
  if (lane == leader) {
    const auto begin = peer == UINT32_MAX ? 0U : peer;
    const auto end = peer == UINT32_MAX ? a.local_world_size : peer + 1U;
    for (unsigned p = begin; p < end; ++p) {
      if (p == local)
        continue;
      auto &state = channel(a, a.local_base, p, blockIdx.x);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> head(
          state.claimed);
      seq = head.load(::cuda::memory_order_relaxed);
      auto &slot = state.slots[seq % kDepth];
      if (::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
              slot.ready)
                  .load(::cuda::memory_order_acquire) >= seq / kDepth + 1) {
        bool claimed = true;
        if constexpr (SingleConsumer)
          head.store(seq + 1, ::cuda::memory_order_relaxed);
        else
          claimed = head.compare_exchange_strong(seq, seq + 1,
                                                   ::cuda::memory_order_relaxed);
        if (claimed) {
          selected = p;
          break;
        }
      }
    }
  }
  selected = __shfl_sync(mask, selected, leader);
  seq = __shfl_sync(mask, seq, leader);
  if (selected == a.local_world_size)
    return false;
  __syncwarp(mask);
  auto &state = channel(a, a.local_base, selected, blockIdx.x);
  auto &slot = state.slots[seq % kDepth];
  for (unsigned i = ordinal; i < slot.count; i += width)
    Receiver<Args>::apply(a, slot.rows[i]);
  __syncwarp(mask);
  if (lane == leader) {
    const auto turn = seq / kDepth + 1;
    if (slot.kind == Kind::End)
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(state.ended)
          .store(1, ::cuda::memory_order_release);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        slot.consumed)
        .store(turn, ::cuda::memory_order_release);
    auto &credit =
        channel(a, a.peer_bases[a.node * a.local_world_size + selected], local,
                blockIdx.x)
            .slots[seq % kDepth]
            .credit;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(credit)
        .store(turn, ::cuda::memory_order_release);
  }
  __syncwarp(mask);
  return true;
}

template <class Args, bool AssistReceive = true>
__device__ __forceinline__ void
wait_credit(const Args &a, unsigned p, unsigned long long seq,
            unsigned incoming_peer = UINT32_MAX) {
  const auto mask = __activemask();
  const auto leader = unsigned(__ffs(mask) - 1);
  while (true) {
    unsigned ready = 0;
    if (threadIdx.x % 32 == leader) {
      auto &credit =
          channel(a, a.local_base, p, blockIdx.x).slots[seq % kDepth].credit;
      ready =
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
              credit)
              .load(::cuda::memory_order_acquire) >= seq / kDepth;
    }
    if (__shfl_sync(mask, ready, leader))
      return;
    if constexpr (AssistReceive) {
      if (!progress(a, incoming_peer))
        __nanosleep(50);
    } else {
      __nanosleep(50);
    }
  }
}

struct Entry {
  unsigned peer;
  unsigned long long row;
};
struct Batch {
  Entry entries[kRows];
  unsigned positions[kRows];
  unsigned counts[communication::kLocalPeers];
  unsigned long long sequences[communication::kLocalPeers];
};

__device__ __forceinline__ void sync(unsigned threads) {
  asm volatile("bar.sync 5, %0;" ::"r"(threads) : "memory");
}

// Dispatch writes directly into immutable per-execution storage. A one-way
// arrival count releases the destination's local compute queue; no credit is
// needed because this memory is not recycled while computation uses it.
template <class Args, class Fill>
__device__ __forceinline__ void publish_dispatch(const Args &a, unsigned count,
                                                 unsigned threads, Fill fill) {
  for (unsigned i = threadIdx.x; i < count; i += threads) {
    const auto entry = fill(i);
    if (entry.peer < a.local_world_size)
      Receiver<Args>::publish_dispatch(a, entry.row, entry.peer);
  }
  sync(threads);
}

// Copies are complete before publishing these Return notifications. Ordinary
// peer stores and one release per chunk replace remote queue reservations.
template <class Args, class Fill>
__device__ __forceinline__ void publish_return(const Args &a, unsigned count,
                                               unsigned threads, Fill fill) {
  __shared__ Batch batch;
  const auto local = a.rank % a.local_world_size;
  for (unsigned begin = 0; begin < count; begin += kRows) {
    const auto rows = min(kRows, count - begin);
    for (unsigned i = threadIdx.x; i < rows; i += threads)
      batch.entries[i] = fill(begin + i);
    if (threadIdx.x < a.local_world_size)
      batch.counts[threadIdx.x] = 0;
    sync(threads);
    if (threadIdx.x < kRows) {
      const auto peer = threadIdx.x < rows ? batch.entries[threadIdx.x].peer
                                           : communication::kLocalPeers;
      const auto lane = threadIdx.x % 32;
      const auto mask = __match_any_sync(0xffffffffU, peer);
      const auto leader = unsigned(__ffs(mask) - 1);
      unsigned base = 0;
      if (lane == leader && peer < a.local_world_size)
        base = atomicAdd(&batch.counts[peer], unsigned(__popc(mask)));
      base = __shfl_sync(0xffffffffU, base, leader);
      if (threadIdx.x < rows)
        batch.positions[threadIdx.x] = base + __popc(mask & ((1U << lane) - 1));
    }
    sync(threads);
    if (!threadIdx.x) {
      for (unsigned p = 0; p < a.local_world_size; ++p) {
        if (!batch.counts[p] || p == local)
          continue;
        auto &state = channel(a, a.local_base, p, blockIdx.x);
        batch.sequences[p] = state.sent;
      }
    }
    sync(threads);
    if (threadIdx.x < 32) {
      for (unsigned p = 0; p < a.local_world_size; ++p)
        if (p != local && batch.counts[p])
          wait_credit(a, p, batch.sequences[p]);
    }
    sync(threads);
    for (unsigned i = threadIdx.x; i < rows; i += threads) {
      const auto entry = batch.entries[i];
      if (entry.peer >= a.local_world_size)
        continue;
      if (entry.peer == local)
        Receiver<Args>::apply(a, entry.row);
      else {
        const auto seq = batch.sequences[entry.peer];
        auto &slot =
            channel(a, a.peer_bases[a.node * a.local_world_size + entry.peer],
                    local, blockIdx.x)
                .slots[seq % kDepth];
        slot.rows[batch.positions[i]] = entry.row;
      }
    }
    __threadfence_system();
    sync(threads);
    if (!threadIdx.x) {
      for (unsigned p = 0; p < a.local_world_size; ++p) {
        if (!batch.counts[p] || p == local)
          continue;
        const auto seq = batch.sequences[p];
        auto &slot = channel(a, a.peer_bases[a.node * a.local_world_size + p],
                             local, blockIdx.x)
                         .slots[seq % kDepth];
        slot.count = batch.counts[p];
        slot.kind = Kind::Return;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
            slot.ready)
            .store(seq / kDepth + 1, ::cuda::memory_order_release);
        channel(a, a.local_base, p, blockIdx.x).sent = seq + 1;
      }
    }
    if (threadIdx.x < 32)
      progress(a);
    sync(threads);
  }
}

// A known destination avoids repartitioning the row IDs and probing unrelated
// incoming channels. Payload completion precedes every notification release.
template <class Args, class Fill>
__device__ __forceinline__ void
publish_peer_return(const Args &a, unsigned peer, unsigned count,
                    unsigned threads, Fill fill, unsigned incoming_peer) {
  __shared__ unsigned long long sequence;
  const unsigned local = a.rank % a.local_world_size;
  for (unsigned begin = 0; begin < count; begin += kRows) {
    const auto rows = min(kRows, count - begin);
    if (peer == local) {
      for (unsigned i = threadIdx.x; i < rows; i += threads)
        Receiver<Args>::apply(a, fill(begin + i).row);
      sync(threads);
      continue;
    }
    auto &outgoing = channel(a, a.local_base, peer, blockIdx.x);
    if (!threadIdx.x)
      sequence = outgoing.sent;
    sync(threads);
    if (threadIdx.x < 32U)
      wait_credit(a, peer, sequence, incoming_peer);
    sync(threads);
    auto &slot = channel(a, a.peer_bases[peer], local, blockIdx.x)
                     .slots[sequence % kDepth];
    for (unsigned i = threadIdx.x; i < rows; i += threads)
      slot.rows[i] = fill(begin + i).row;
    __threadfence_system();
    sync(threads);
    if (!threadIdx.x) {
      slot.count = rows;
      slot.kind = Kind::Return;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          slot.ready)
          .store(sequence / kDepth + 1U, ::cuda::memory_order_release);
      ++outgoing.sent;
    }
    sync(threads);
  }
}

// A worker must keep receiving after its own finite copy work ends. Explicit
// terminal chunks handle empty/skewed peers without predicting receive counts.
template <class Args>
__device__ __forceinline__ void drain(const Args &a,
                                      unsigned destination = UINT32_MAX,
                                      unsigned source = UINT32_MAX) {
  if (threadIdx.x >= 32U)
    return;
  const auto local = a.rank % a.local_world_size;
  const auto begin = destination == UINT32_MAX ? 0U : destination;
  const auto end =
      destination == UINT32_MAX ? a.local_world_size : destination + 1U;
  for (unsigned p = begin; p < end; ++p) {
    if (p == local)
      continue;
    auto &state = channel(a, a.local_base, p, blockIdx.x);
    const auto seq = state.sent;
    wait_credit(a, p, seq, source);
    auto &slot = channel(a, a.peer_bases[a.node * a.local_world_size + p],
                         local, blockIdx.x)
                     .slots[seq % kDepth];
    if (!threadIdx.x) {
      slot.count = 0;
      slot.kind = Kind::End;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          slot.ready)
          .store(seq / kDepth + 1U, ::cuda::memory_order_release);
      ++state.sent;
    }
    __syncwarp();
  }
  while (true) {
    progress(a, source);
    unsigned done = 1U;
    if (!threadIdx.x) {
      for (unsigned p = 0; p < a.local_world_size; ++p) {
        if (p == local)
          continue;
        auto &state = channel(a, a.local_base, p, blockIdx.x);
        if (source == UINT32_MAX || p == source) {
          done &= ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
                      state.ended)
                      .load(::cuda::memory_order_acquire) != 0U;
          const auto claimed =
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_device>(state.claimed)
                  .load(::cuda::memory_order_acquire);
          for (unsigned i = 0; i < kDepth; ++i)
            done &= ::cuda::atomic_ref<unsigned long long,
                                       ::cuda::thread_scope_device>(
                        state.slots[i].consumed)
                        .load(::cuda::memory_order_acquire) >=
                    (claimed + kDepth - 1U - i) / kDepth;
        }
        if (destination == UINT32_MAX || p == destination)
          for (unsigned i = 0; i < kDepth; ++i)
            done &= ::cuda::atomic_ref<unsigned long long,
                                       ::cuda::thread_scope_system>(
                        state.slots[i].credit)
                        .load(::cuda::memory_order_acquire) >=
                    (state.sent + kDepth - 1U - i) / kDepth;
      }
    }
    if (__shfl_sync(0xffffffffU, done, 0U))
      return;
    __nanosleep(50U);
  }
}

template <> struct Receiver<ForwardKernelArgs> {
  __device__ static void publish_dispatch(const ForwardKernelArgs &a,
                                          unsigned long long tile,
                                          unsigned peer) {
    const auto group = __match_any_sync(
        __activemask(), tile * communication::kLocalPeers + peer);
    if (threadIdx.x % 32 == unsigned(__ffs(group) - 1)) {
      auto *arrived = peer_at<unsigned long long>(
          a.peer_bases[a.node * a.local_world_size + peer],
          a.x_arrived_rows_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          arrived[tile * a.local_world_size + a.rank % a.local_world_size])
          .fetch_add(__popc(group), ::cuda::memory_order_release);
    }
  }
  __device__ static void apply(const ForwardKernelArgs &a,
                               unsigned long long row) {
    forward_reduce_detail::publish_ready_route(a, row, 2U);
  }
};
} // namespace local_notification_detail
