#pragma once

// One sender CTA owns each state. Only its submitter mutates scheduling fields;
// its copy warps release-publish immutable packed tasks. No cross-CTA lease.
struct GinPendingSlot {
  unsigned pending; // 0: empty, 1: packing, 2: packed.
  activation_ring_detail::Transfer task;
};
static_assert(sizeof(GinPendingSlot) == 40U);

struct alignas(128) GinSenderState {
  unsigned done;
  unsigned dispatch_cursor;
  unsigned return_cursor;
  unsigned dispatch_first;
  unsigned slot_cursor;
  unsigned long long dispatch_puts;
  unsigned long long return_puts;
  GinPendingSlot slots[forward_tuning::kUnifiedGinPackSlots];
};
static_assert(offsetof(GinSenderState, slots) == 40U);
static_assert(sizeof(GinSenderState) == forward_tuning::kUnifiedGinStateBytes);
