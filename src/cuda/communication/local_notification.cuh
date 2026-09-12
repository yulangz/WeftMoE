#pragma once

#include <cstddef>
#include <cuda/atomic>

namespace weftmoe::cuda::communication {

// Payloads go directly to the execution arena. Only row IDs circulate here.
// Each transport CTA owns one outgoing channel per local peer. A slot is
// reusable after that peer has applied every notification in the chunk.
inline constexpr unsigned kLocalChunkRows = 128;
inline constexpr unsigned kLocalRingDepth = 4;
inline constexpr unsigned kLocalPeers = 8;
enum class NotificationKind : unsigned { Return, End };

struct alignas(128) NotificationSlot {
  unsigned long long ready;
  unsigned long long credit;
  unsigned long long consumed;
  unsigned count;
  NotificationKind kind;
  unsigned long long rows[kLocalChunkRows];
};

struct alignas(128) NotificationChannel {
  unsigned long long sent;
  unsigned long long claimed;
  unsigned ended;
  NotificationSlot slots[kLocalRingDepth];
};

} // namespace weftmoe::cuda::communication
