#pragma once

#include "cuda/forward/unified_kernel.cuh"

namespace weftmoe_pf {

// General static Forward: bounded rings for single-/multi-node prefill and
// training, plus the existing non-ring and short multi-node specializations.
template <bool StaticLocal, bool RetainBackwardState, bool Prefill = false,
          bool RingSchedule = false>
__global__ __launch_bounds__(forward_tuning::kUnifiedThreads, 1) void kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_store_map) {
  forward_unified_detail::run<StaticLocal, RetainBackwardState, true, Prefill,
                              RingSchedule>(
      a, x_map, s_map, w1_map, w2_map, output_store_map);
}

} // namespace weftmoe_pf
