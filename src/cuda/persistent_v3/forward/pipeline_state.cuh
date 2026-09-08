#pragma once

#include <cstdint>

#include "cuda/persistent_v3/epoch_state.cuh"

namespace megamoe::cuda::persistent_v3 {

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
// paired Dispatch Relay CTA.
struct alignas(128) ForwardDispatchSequence {
  PaddedCounter outbound_directory_ready;
  PaddedCounter outbound_waves_packed;
  PaddedCounter outbound_pack_complete;
  PaddedCounter inbound_directory_ready;
  PaddedCounter inbound_waves_ready;
};

struct alignas(128) ForwardPipelineControl {
  GenerationState generation;
  ForwardDispatchSequence dispatch[kRdmaCtaCount];

  PaddedCounter rdma_dispatch_roles_complete;
  PaddedCounter relay_dispatch_roles_complete;
  PaddedCounter rdma_return_roles_complete;
  PaddedCounter return_receive_roles_complete;
  PaddedCounter source_ready_published;
  // Shared destination/stripe cursors. Host validation allows at most 32
  // Relay CTAs. Reserve padding for rounding up to the local world size.
  PaddedCounter local_return_head[32U + kRdmaCtaCount - 1U];
  PaddedCounter return_chunk_scan;
  PaddedCounter return_chunks_complete;
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

} // namespace megamoe::cuda::persistent_v3
