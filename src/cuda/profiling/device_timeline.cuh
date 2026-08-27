#pragma once

#include "cuda/profiling/device_timeline_types.hpp"

#if MEGAMOE_DEVICE_TIMELINE

namespace megamoe::cuda::profiling {

__device__ __forceinline__ std::uint64_t read_globaltimer_ns() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

// One elected lane writes each CTA's private slice. Consequently reservation
// needs neither a global atomic nor ordering between unrelated workers.
__device__ __forceinline__ void record(
    const DeviceTimelineContext &context, TimelineEvent event,
    TimelineRole role, std::uint32_t channel = 0, std::uint64_t arg0 = 0,
    std::uint64_t arg1 = 0) {
  if (context.records == nullptr || context.counts == nullptr ||
      context.overflow == nullptr || threadIdx.x != 0U ||
      blockIdx.x >= context.max_ctas)
    return;
  auto &count = context.counts[blockIdx.x];
  const auto index = count++;
  if (index >= context.capacity_per_cta) {
    context.overflow[blockIdx.x] = index + 1U - context.capacity_per_cta;
    return;
  }
  auto &dst = context.records[
      static_cast<std::uint64_t>(blockIdx.x) * context.capacity_per_cta +
      index];
  dst.timestamp_ns = read_globaltimer_ns();
  dst.generation = context.generation;
  dst.arg0 = arg0;
  dst.arg1 = arg1;
  dst.rank = context.rank;
  dst.cta = blockIdx.x;
  dst.channel = channel;
  dst.metadata = ((context.phase & 0xffU) << 24U) |
                 ((static_cast<std::uint32_t>(role) & 0xffU) << 16U) |
                 (static_cast<std::uint32_t>(event) & 0xffffU);
}

} // namespace megamoe::cuda::profiling

#define MEGAMOE_TIMELINE_RECORD(context, event, role, channel, arg0, epoch) \
  ::megamoe::cuda::profiling::record(                                    \
      context, ::megamoe::cuda::profiling::TimelineEvent::event,          \
      ::megamoe::cuda::profiling::TimelineRole::role, channel, arg0, epoch)

#else

#define MEGAMOE_TIMELINE_RECORD(context, event, role, channel, arg0, epoch) \
  do {                                                                     \
  } while (false)

#endif
