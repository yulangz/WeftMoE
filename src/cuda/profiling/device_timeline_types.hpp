#pragma once

#include <cstdint>
#include <vector>

#ifndef MEGAMOE_DEVICE_TIMELINE
#define MEGAMOE_DEVICE_TIMELINE 0
#endif

namespace megamoe::cuda::profiling {

inline constexpr std::uint32_t kTimelineSchemaVersion = 3;
inline constexpr std::uint32_t kEventsPerCta = 256;

enum class TimelinePhase : std::uint16_t { Forward = 1, Backward = 2 };

enum class TimelineRole : std::uint16_t {
  Kernel = 0,
  GinDispatch = 1,
  ProxyRelay = 2,
  Compute = 3,
  GinReturn = 4,
  SourceRelay = 5,
  Reduce = 6,
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
  DyPacketReady = 23,
  DyGinPutIssue = 24,
  DyLocalCompletion = 25,
  DyRelayStart = 26,
  DyRelayEnd = 27,
  DvFirst = 28,
  DvLast = 29,
  DxTileReady = 30,
  Dw1Ready = 31,
  Dw1Start = 32,
  Dw1End = 33,
  Dw2Ready = 34,
  Dw2Start = 35,
  Dw2End = 36,
  DxReturnPackStart = 37,
  DxReturnGinPutIssue = 38,
  DxReturnLocalCompletion = 39,
  DxSourceRelay = 40,
  DGateReturn = 41,
  XFirstLocalRowVisible = 42,
  XFirstRemoteSignal = 43,
  XFirstProxyRelay = 44,
  XFirstMtileReady = 45,
  Fc1ClaimStart = 46,
  XLastDispatchPut = 47,
  XLastRelay = 48,
  XDispatchComplete = 49,
  DyFirstLocalRowVisible = 50,
  DyFirstRemoteSignal = 51,
  DyFirstMtileReady = 52,
  DvClaimStart = 53,
  DyLastDispatchPut = 54,
  DyLastRelay = 55,
  DyDispatchComplete = 56,
  ReturnPackCtaBytes = 57,
  ReturnDirectoryStart = 58,
  ReturnDirectoryEnd = 59,
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

} // namespace megamoe::cuda::profiling
