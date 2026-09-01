#pragma once

#include "cuda/profiling/device_timeline_types.hpp"

#if MEGAMOE_DEVICE_TIMELINE

namespace megamoe::cuda::profiling {

__device__ __forceinline__ std::uint64_t read_globaltimer_ns() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

// Most events are emitted by thread 0, but a warp-specialized transport CTA
// also has a bulk pack leader at thread 64.  Profiling builds use an atomic
// reservation within the CTA-private slice so both independent writers can be
// observed without a barrier between GIN progress and packing.
__device__ __forceinline__ void record_if(
    const DeviceTimelineContext &context, bool elected, TimelineEvent event,
    TimelineRole role, std::uint32_t channel = 0, std::uint64_t arg0 = 0,
    std::uint64_t arg1 = 0) {
  const auto logical_cta = context.cta_base + blockIdx.x;
  if (context.records == nullptr || context.counts == nullptr ||
      context.overflow == nullptr || !elected ||
      logical_cta >= context.max_ctas)
    return;
  auto &count = context.counts[logical_cta];
  const auto index = atomicAdd(&count, 1U);
  if (index >= context.capacity_per_cta) {
    atomicMax(&context.overflow[logical_cta],
              index + 1U - context.capacity_per_cta);
    return;
  }
  auto &dst = context.records[
      static_cast<std::uint64_t>(logical_cta) * context.capacity_per_cta +
      index];
  dst.timestamp_ns = read_globaltimer_ns();
  dst.generation = context.generation;
  dst.arg0 = arg0;
  dst.arg1 = arg1;
  dst.rank = context.rank;
  dst.cta = logical_cta;
  dst.channel = channel;
  dst.metadata = ((context.phase & 0xffU) << 24U) |
                 ((static_cast<std::uint32_t>(role) & 0xffU) << 16U) |
                 (static_cast<std::uint32_t>(event) & 0xffffU);
}

__device__ __forceinline__ void record(
    const DeviceTimelineContext &context, TimelineEvent event,
    TimelineRole role, std::uint32_t channel = 0, std::uint64_t arg0 = 0,
    std::uint64_t arg1 = 0) {
  record_if(context, threadIdx.x == 0U, event, role, channel, arg0, arg1);
}

} // namespace megamoe::cuda::profiling

#define MEGAMOE_TIMELINE_RECORD(context, event, role, channel, arg0, epoch) \
  ::megamoe::cuda::profiling::record(                                    \
      context, ::megamoe::cuda::profiling::TimelineEvent::event,          \
      ::megamoe::cuda::profiling::TimelineRole::role, channel, arg0, epoch)

#define MEGAMOE_TIMELINE_RECORD_IF(context, elected, event, role, channel, \
                                   arg0, epoch)                           \
  ::megamoe::cuda::profiling::record_if(                                  \
      context, elected, ::megamoe::cuda::profiling::TimelineEvent::event, \
      ::megamoe::cuda::profiling::TimelineRole::role, channel, arg0, epoch)

#else

#define MEGAMOE_TIMELINE_RECORD(context, event, role, channel, arg0, epoch) \
  do {                                                                     \
  } while (false)

#define MEGAMOE_TIMELINE_RECORD_IF(context, elected, event, role, channel, \
                                   arg0, epoch)                            \
  do {                                                                      \
  } while (false)

#endif
