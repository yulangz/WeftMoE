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

// Dispatch storage is generation-static and every context owns a contiguous
// wave interval, so its hot handoffs need only monotonic prefixes.  Outbound
// fields connect the ten RDMA-resident packing warps to the scalar GIN owner;
// inbound fields connect that owner (after signal/proxy acquisition) to the
// shared Relay consumers for that channel.
struct alignas(128) ForwardDispatchSequence {
  PaddedCounter outbound_directory_ready;
  PaddedCounter outbound_waves_packed;
  PaddedCounter outbound_pack_complete;
  PaddedCounter inbound_directory_ready;
  PaddedCounter inbound_waves_ready;
  // Packed {claimed wave count, relative packet cursor}; CAS assigns each
  // variable-length wave exactly once to any Relay CTA on this channel.
  PaddedCounter inbound_claim;
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

// Notify's count matrix fixes disjoint arrival intervals for each sender.
// Only that sender writes the interval, and Relay workers share its head.
struct alignas(64) ReturnPeerQueue {
  unsigned long long receive_begin = 0;
  unsigned long long receive_count = 0;
  unsigned long long receive_head = 0;
  unsigned long long send_begin = 0;
  unsigned long long sent = 0;
};

struct alignas(128) ForwardPipelineControl {
  GenerationState generation;
  ForwardDispatchSequence dispatch[kRdmaCtaCount];

  PaddedCounter rdma_dispatch_roles_complete;
  PaddedCounter relay_dispatch_roles_complete;
  PaddedCounter rdma_return_roles_complete;
  PaddedCounter return_receive_roles_complete;
  PaddedCounter source_ready_published;
  // Queue 0 is local; queue 1+context belongs to its permanent GIN owner.
  ReturnRowQueue return_rows[1U + kRdmaCtaCount];
  PaddedCounter return_rows_complete;
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
