#pragma once

#include "cuda/forward/unified_kernel.cuh"

// Preserve the V3 fallback without duplicating the shared compute pipeline.
template <bool StaticLocal, bool RetainBackwardState>
__global__ __launch_bounds__(forward_tuning::kUnifiedThreads, 1)
void unified_forward_kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_store_map) {
  forward_unified_detail::run<StaticLocal, RetainBackwardState>(
      a, x_map, s_map, w1_map, w2_map, output_store_map);
}
