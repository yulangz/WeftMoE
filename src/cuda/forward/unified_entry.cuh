#pragma once

#include "cuda/persistent_v3/forward/unified_kernel.cuh"
#include "cuda/weftmoe_pf/kernel.cuh"

#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY) && __has_include(<nccl_device.h>)
#include "unified_communication_benchmark.cuh"
#endif

template <bool StaticLocal, bool RetainBackwardState,
          bool StaticSchedule = false, bool Prefill = false,
          bool RingSchedule = false>
constexpr auto unified_forward_entry() {
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY) && __has_include(<nccl_device.h>)
  return unified_communication_benchmark_kernel<StaticLocal,
                                                RetainBackwardState>;
#else
  if constexpr (StaticSchedule)
    return weftmoe_pf::kernel<StaticLocal, RetainBackwardState, Prefill,
                              RingSchedule>;
  else
    return unified_forward_kernel<StaticLocal, RetainBackwardState>;
#endif
}
