#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda/persistent_v3/queues.cuh"

namespace megamoe::cuda::persistent_v3 {

// A context normally owns only a handful of 128-row waves.  Sixty-four
// entries also covers the pessimistic main-shape expert-fragmentation bound
// while keeping every handoff queue resident in ordinary global memory.
inline constexpr std::uint32_t kBackwardQueueCapacity = 64U;
// Hopper tops out at 132 SMs.  With eight RDMA and eight Relay CTAs the
// production compute grid therefore has at most 58 two-CTA clusters.  Keep a
// little headroom so paired activation scheduling never aliases mailboxes.
inline constexpr std::uint32_t kBackwardActivationPairCapacity = 64U;

inline constexpr std::uint16_t kPipelineTaskDirectory = 1U << 0U;
inline constexpr std::uint16_t kPipelineTaskTerminal = 1U << 1U;

// Fixed-size, generation-local control plane.  Large runtime-sized arrays
// (M-block mappings/cursors and per-expert cursors) are separate arena slices
// so this type stays independent of the configured token capacity.
struct alignas(128) BackwardPipelineControl {
  GenerationState generation;
  SpscRing<PipelineTask, kBackwardQueueCapacity>
      relay_to_rdma[kRdmaCtaCount];
  SpscRing<PipelineTask, kBackwardQueueCapacity>
      rdma_to_relay[kRdmaCtaCount];

  PaddedCounter setup_epoch;
  // setup_epoch is system-visible and is observed by the one coordinator on
  // every same-node GPU.  roles_start_epoch is then the cheap device-local
  // fan-out that releases this GPU's resident role CTAs.  Splitting the two
  // avoids making all 132 CTAs poll seven peer allocations during startup.
  PaddedCounter roles_start_epoch;
  PaddedCounter rdma_roles_complete;
  PaddedCounter relay_roles_complete;
  PaddedCounter compute_roles_complete;
  PaddedCounter local_route_roles_complete;
  PaddedCounter outbound_pack_roles_complete;
  PaddedCounter inbound_dy_roles_complete;
  PaddedCounter dv_scan_cursor;
  PaddedCounter dx_scan_cursor;
  PaddedCounter dw1_scan_cursor;
  PaddedCounter dw2_scan_cursor;
  PaddedCounter direct_dy_payload_roles_complete;
  PaddedCounter dv_tasks_complete;
  PaddedCounter mblocks_ready;
  PaddedCounter dx_tasks_complete;
  PaddedCounter dw1_tasks_complete;
  PaddedCounter dw2_tasks_complete;
  PaddedCounter return_chunks_packed;
  PaddedCounter outbound_return_roles_complete;
  PaddedCounter inbound_return_roles_complete;
  PaddedCounter return_scatter_tasks_complete;
  PaddedCounter reduce_scan_cursor;
  PaddedCounter reduce_tasks_complete;

  // Written by the setup CTA, then immutable after setup_epoch is published.
  unsigned long long mblocks = 0;
  unsigned long long padded_rows = 0;
  unsigned long long dv_tasks = 0;
  unsigned long long dx_tasks = 0;
  unsigned long long dw1_tasks = 0;
  unsigned long long dw2_tasks = 0;
  unsigned long long return_chunks = 0;
  unsigned long long outbound_return_tasks = 0;
  unsigned long long inbound_return_tasks = 0;
  unsigned long long reduce_tasks = 0;
};

static_assert(alignof(BackwardPipelineControl) == 128U);

// A single leader claim assigns adjacent N tiles to the two CTAs in a
// cluster.  The odd/even sequence is a tiny device-scope producer/consumer
// protocol: odd publishes the next pair, even acknowledges that the follower
// finished it.  Keeping this in global memory avoids a per-tile full-cluster
// barrier while still making the task identity visible before TMA multicast.
struct alignas(32) BackwardActivationPairMailbox {
  unsigned long long sequence = 0;
  unsigned long long tasks[2] = {0, 0};
  unsigned long long reserved = 0;
};

static_assert(sizeof(BackwardActivationPairMailbox) == 32U);

#if defined(__CUDACC__)
MEGAMOE_V3_DEVICE inline bool bounded_ticket_claim(
    unsigned long long &cursor, unsigned long long limit,
    unsigned long long &ticket) {
  auto observed = load_relaxed<::cuda::thread_scope_device>(cursor);
  while (observed < limit) {
    auto expected = observed;
    if (compare_exchange_acq_rel<::cuda::thread_scope_device>(
            cursor, expected, observed + 1ULL)) {
      ticket = observed;
      return true;
    }
    observed = expected;
  }
  return false;
}

MEGAMOE_V3_DEVICE inline bool bounded_ticket_claim_span(
    unsigned long long &cursor, unsigned long long limit,
    unsigned long long span, unsigned long long &first) {
  auto observed = load_relaxed<::cuda::thread_scope_device>(cursor);
  while (observed + span <= limit) {
    auto expected = observed;
    if (compare_exchange_acq_rel<::cuda::thread_scope_device>(
            cursor, expected, observed + span)) {
      first = observed;
      return true;
    }
    observed = expected;
  }
  return false;
}

template <class Ring>
MEGAMOE_V3_DEVICE inline void initialize_spsc_ring(Ring &ring) {
  if (threadIdx.x == 0U) {
    ring.producer_ticket.value = 0ULL;
    ring.consumer_ticket.value = 0ULL;
  }
  for (std::uint32_t slot = threadIdx.x; slot < kBackwardQueueCapacity;
       slot += blockDim.x)
    ring.slots[slot].sequence = slot;
}
#endif

} // namespace megamoe::cuda::persistent_v3
