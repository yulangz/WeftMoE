#pragma once

// Included inside forward_unified_transport. Context leases protect only
// finite queue/control transitions. Payload packing proceeds outside the lease
// and publishes a device-scoped handoff to whichever CTA submits the message.
struct GinPendingSlot {
  unsigned pending; // 0: empty, 1: packing, 2: packed.
  activation_ring_detail::Transfer task;
};
static_assert(sizeof(GinPendingSlot) == 40U);

struct alignas(128) GinContextState {
  unsigned lease;
  unsigned done;
  unsigned dispatch_cursor;
  unsigned return_cursor;
  unsigned dispatch_first;
  unsigned slot_cursor;
  unsigned long long dispatch_puts;
  unsigned long long return_puts;
  GinPendingSlot slots[forward_tuning::kUnifiedGinPackSlots];
};
static_assert(offsetof(GinContextState, slots) == 40U);
static_assert(sizeof(GinContextState) == forward_tuning::kUnifiedGinStateBytes);
