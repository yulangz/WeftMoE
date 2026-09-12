#pragma once

#include <cstdint>

#include "cuda/persistent_v3/epoch_state.cuh"

namespace weftmoe::cuda::persistent_v3 {

// Hopper exposes at most 132 SMs.  After the fixed RDMA/Relay reservation the
// Forward compute grid contains at most 58 two-CTA clusters.  Keep a little
// headroom so an experimental cluster scheduler can use one mailbox per pair
// without coupling the arena size to a particular device's SM count.
inline constexpr std::uint32_t kForwardComputePairCapacity = 64U;

// The leader assigns one same-kind task to each CTA.  Odd sequence values
// publish a pair; the follower changes the value to the next even sequence
// after finishing.  This thread-0 handshake avoids a full-cluster barrier at
// every GEMM boundary while retaining the lockstep required by TMA multicast.
struct alignas(32) ForwardComputePairMailbox {
  unsigned long long sequence = 0;
  unsigned long long tasks[2] = {0, 0};
  unsigned long long metadata = 0;
};

static_assert(sizeof(ForwardComputePairMailbox) == 32U);

inline constexpr unsigned long long kNoInputGroup = ~0ULL;

// Owned by one destination-local publisher for the entire execution. Advance
// tile_cursor only after all rows of that Notify tile have arrived; completed
// dependencies never need to be checked again.
struct ForwardInputGroup {
  unsigned long long tile_cursor = 0;
  unsigned long long tile_end = 0;
  unsigned long long next = kNoInputGroup;
  std::uint32_t mblocks = 0;
};

static_assert(sizeof(ForwardInputGroup) == 32U);

struct ForwardInputGroupSpan {
  std::uint32_t begin = 0;
  std::uint32_t count = 0;
};

// Keep startup blocks independent and clip groups at locality-window edges.
WEFTMOE_V3_HOST_DEVICE constexpr ForwardInputGroupSpan
input_group_span(std::uint32_t block, std::uint32_t blocks,
                 std::uint32_t startup, std::uint32_t locality,
                 std::uint32_t wide_begin, std::uint32_t wide_end) {
  if (block >= blocks)
    return {};
  if (locality <= 1U || block < startup)
    return {block, 1U};
  const auto begin = wide_begin > startup ? wide_begin : startup;
  auto origin = startup;
  auto limit = blocks;
  auto width = locality;
  if (locality == 2U && begin < wide_end) {
    if (block < begin) {
      limit = begin < blocks ? begin : blocks;
    } else if (block < wide_end) {
      origin = begin;
      limit = wide_end < blocks ? wide_end : blocks;
      width = 4U;
    } else {
      origin = wide_end;
    }
  }
  const auto leader = block - (block - origin) % width;
  return {leader, width < limit - leader ? width : limit - leader};
}

// A private circular queue of pending groups. Rotating a pending head changes
// only registers; a completed head is unlinked once. No peer accesses, shared
// queue head, atomics or per-entry epochs are needed for this work list.
struct ForwardInputPendingQueue {
  unsigned long long head = kNoInputGroup;
  unsigned long long tail = kNoInputGroup;

  WEFTMOE_V3_HOST_DEVICE bool empty() const { return head == kNoInputGroup; }

  WEFTMOE_V3_HOST_DEVICE void push(ForwardInputGroup *groups,
                                   unsigned long long group) {
    if (empty())
      head = group;
    else
      groups[tail].next = group;
    groups[group].next = head;
    tail = group;
  }

  WEFTMOE_V3_HOST_DEVICE void advance(ForwardInputGroup *groups, bool ready) {
    const auto next = groups[head].next;
    if (!ready) {
      tail = head;
    } else if (head == tail) {
      head = tail = kNoInputGroup;
      return;
    } else {
      groups[tail].next = next;
    }
    head = next;
  }
};

// Each producer reserves, fills and confirms one interval before reserving
// another. Consumers claim only confirmed entries. Storage is reset between
// executions; no ring wrap, per-entry epoch or descriptor scan is required.
struct alignas(128) ReturnRowQueue {
  unsigned long long reserved = 0;
  unsigned long long confirmed = 0;
  unsigned long long head = 0;
  unsigned long long begin = 0;
  unsigned long long capacity = 0;
};

struct alignas(128) ForwardPipelineControl {
  GenerationState generation;

  PaddedCounter rdma_dispatch_roles_complete;
  PaddedCounter relay_dispatch_roles_complete;
  PaddedCounter rdma_return_roles_complete;
  PaddedCounter return_receive_roles_complete;
  PaddedCounter source_ready_published;
  PaddedCounter return_producers_complete;
  // A route publisher increments one source-token counter after making its
  // contribution visible.  The publisher that completes top-k appends that
  // token to the ready queue.  Compute CTAs consume the queue in small batches
  // after FC2, avoiding a late route at the head of a fixed token stripe from
  // stalling otherwise-ready reductions on the same warp.
  PaddedCounter reduce_ready_tail;
  PaddedCounter reduce_ready_head;
  PaddedCounter reduce_tokens_complete;
  // Compute CTAs publish here after their disjoint token-reduction stripe.
  // The elected transport terminal CTA acquires the full count instead of
  // forcing either independent grid through a cross-role grid.sync().
  PaddedCounter compute_reduce_roles_complete;
};

static_assert(alignof(ForwardPipelineControl) == 128U);

} // namespace weftmoe::cuda::persistent_v3
