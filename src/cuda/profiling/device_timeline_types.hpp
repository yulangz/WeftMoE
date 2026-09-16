#pragma once

#include <cstdint>
#include <vector>

#ifndef WEFTMOE_DEVICE_TIMELINE
#define WEFTMOE_DEVICE_TIMELINE 0
#endif

namespace weftmoe::cuda::profiling {

inline constexpr std::uint32_t kTimelineSchemaVersion = 9;
// Main-shape compute CTAs can retire more than 256 FC tiles.  A truncated
// suffix makes the apparent FC2/Return tail shorter than the real one, so the
// profiling build deliberately spends a few extra MiB per GPU for a complete
// trace.  This allocation is compiled out of production builds.
// Per-task math, epilogue and scheduler boundaries need a larger capacity
// than the earlier phase-only traces, especially at 64K tokens per rank.
inline constexpr std::uint32_t kEventsPerCta = 16384;

enum class TimelinePhase : std::uint16_t { Forward = 1 };

enum class TimelineRole : std::uint16_t {
  Kernel = 0,
  GinDispatch = 1,
  ProxyRelay = 2,
  Compute = 3,
  GinReturn = 4,
  SourceRelay = 5,
  Reduce = 6,
  LocalReturn = 7,
};

enum class TimelineEvent : std::uint16_t {
  KernelEnter = 1,
  DispatchPacketReady = 2,
  GinPutIssue = 3,
  DispatchLocalCompletion = 4,
  RemoteSignalObserved = 5,
  ProxyRelayStart = 6,
  ProxyRelayEnd = 7,
  XTileReady = 8,
  Fc1First = 9,
  Fc1Last = 10,
  Fc2TileReady = 11,
  ReturnPackStart = 12,
  ReturnPackEnd = 13,
  ReturnGinPutIssue = 14,
  ReturnLocalCompletion = 15,
  SourceSignalObserved = 16,
  SourceRelayStart = 17,
  SourceRelayEnd = 18,
  ReduceReady = 19,
  ReduceStart = 20,
  ReduceEnd = 21,
  Terminal = 22,
  XFirstLocalRowVisible = 42,
  XFirstRemoteSignal = 43,
  XFirstProxyRelay = 44,
  XFirstMtileReady = 45,
  Fc1ClaimStart = 46,
  XLastDispatchPut = 47,
  XLastRelay = 48,
  XDispatchComplete = 49,
  ReturnPackCtaBytes = 57,
  ReturnDirectoryStart = 58,
  ReturnDirectoryEnd = 59,
  DispatchPackStart = 60,
  DispatchPackEnd = 61,
  DispatchPackCtaBytes = 62,
  ReturnGatherStart = 63,
  ReturnGatherEnd = 64,
  ReturnGatherCtaBytes = 65,
  Fc2First = 71,
  Fc2Last = 72,
  ReduceTokenReadyObserved = 73,
  LocalReturnBatchStart = 74,
  LocalReturnBatchEnd = 75,
  LocalReturnTerminal = 76,
  FullReturnChunkReady = 77,
  FullReturnChunkPublished = 78,
  Fc2MblockReturnReady = 79,
  ComputeSelectStart = 80,
  ComputeSelectEnd = 81,
  Fc1TileStart = 82,
  Fc1MathEnd = 83,
  Fc1TileEnd = 84,
  Fc2TileStart = 85,
  Fc2MathEnd = 86,
  Fc2TileEnd = 87,
  ComputePairJoinEnd = 88,
  TransportRoleDone = 89,
  // Sender samples: arg0 is elapsed ns, arg1 is rows. No event for idle polls.
  GinSenderSelect = 90,
  GinSenderSubmit = 91,
  GinSenderProgress = 92,
  SourceSignalReady = 93,
  // Ready and Claimed identify the same packet by (channel, arg0=sequence).
  SourcePacketClaimed = 94,
};

// Fixed-width, pointer-free host/device trace record. Timestamps are only
// comparable within one GPU's globaltimer domain.
struct DeviceTimelineRecord {
  std::uint64_t timestamp_ns = 0;
  std::uint64_t generation = 0;
  std::uint64_t arg0 = 0;
  std::uint64_t arg1 = 0;
  std::uint32_t rank = 0;
  std::uint32_t cta = 0;
  std::uint32_t channel = 0;
  // phase[31:24] | role[23:16] | event[15:0]
  std::uint32_t metadata = 0;
};
static_assert(sizeof(DeviceTimelineRecord) == 48);

struct DeviceTimelineContext {
  DeviceTimelineRecord *records = nullptr;
  std::uint32_t *counts = nullptr;
  std::uint32_t *overflow = nullptr;
  std::uint32_t max_ctas = 0;
  std::uint32_t capacity_per_cta = 0;
  std::uint32_t rank = 0;
  std::uint32_t phase = 0;
  // Split kernels share one allocation.  Give each grid a disjoint logical
  // CTA range so their lock-free per-CTA writers cannot alias.
  std::uint32_t cta_base = 0;
  std::uint64_t generation = 0;
};

struct DeviceTimelineSnapshot {
  std::uint32_t schema_version = kTimelineSchemaVersion;
  std::uint32_t rank = 0;
  std::uint64_t generation = 0;
  TimelinePhase phase = TimelinePhase::Forward;
  std::uint32_t max_ctas = 0;
  std::uint32_t capacity_per_cta = 0;
  std::uint64_t overflow_events = 0;
  std::vector<DeviceTimelineRecord> records;
};

} // namespace weftmoe::cuda::profiling
