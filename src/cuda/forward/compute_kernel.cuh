#pragma once

// Shared Hopper fragment layout and epilogues for PF and DT.
// Keep the WGMMA roles separate from transport so math and TMA retain their
// independent register quotas without GIN/relay live ranges in the GEMM loop.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)

namespace forward_compute_detail {

// Record the active compute cluster for launch diagnostics.
__device__ __forceinline__ void
observe_compute_cluster(const ForwardKernelArgs &a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (threadIdx.x == 0U) {
    const auto cluster = cooperative_groups::this_cluster();
    const auto dimensions = cluster.dim_blocks();
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->runtime_cluster_size,
              dimensions.x * dimensions.y * dimensions.z);
    atomicOr(&status->cluster_rank_mask, 1ULL << cluster.block_rank());
  }
#endif
}

constexpr std::uint32_t kThreads = forward_tuning::kComputeThreads;
constexpr std::uint32_t kMathWarpGroups = forward_tuning::kMathWarpgroups;
constexpr std::uint32_t kMathThreads = kMathWarpGroups * 128U;
constexpr std::uint32_t kTaskReadyBarrier = 0U;
constexpr std::uint32_t kMathReadyBarrier = 2U;
constexpr std::uint32_t kMathDoneBarrier = 3U;
// The host validates these dimensions before admitting this specialization.
constexpr std::uint64_t kHidden = forward_tuning::kMainHidden;
constexpr std::uint64_t kIntermediate = forward_tuning::kMainIntermediate;

// FC1's WGMMA accumulator is FP32, but the training contract checkpoints A/B
// in BF16. Forward must consume those exact checkpoint values as well, or a
// later SwiGLU recomputation would silently use different operands. FC2 then
// consumes the separately rounded BF16 U = gate*S.
template <bool FastMath = false>
__device__ __forceinline__ float gated_swiglu_fc2_input(float a, float b,
                                                        float gate) {
  float sigmoid;
  if constexpr (FastMath) {
    // The optional inference profile matches MegaMoE's fast SFU path.
    // Projection/output BF16 rounding and multiplication order stay intact;
    // training and all existing callers retain the rounded FP32 division.
    float exponent;
    const auto scaled = -a * 1.4426950408889634F;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponent) : "f"(scaled));
    const auto denominator = 1.0F + exponent;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sigmoid) : "f"(denominator));
  } else {
    sigmoid = 1.0F / (1.0F + __expf(-a));
  }
  return __fmul_rn((a * sigmoid) * b, gate);
}

template <bool FastMath = false>
__device__ __forceinline__ float
gated_swiglu_fc2_input_from_bf16(std::uint16_t a_bits, std::uint16_t b_bits,
                                 float gate) {
  return gated_swiglu_fc2_input<FastMath>(bf16_bits_to_float(a_bits),
                                          bf16_bits_to_float(b_bits), gate);
}

// A 64-column atom uses the same 128-byte swizzle as the FC2 TMA epilogue.
// STSM packs the WGMMA fragments; vector loads below recover contiguous rows.
template <unsigned Rows = 128U>
__device__ __forceinline__ void
store_m_fragment(transport::BFloat16 *tile, unsigned index, __nv_bfloat162 row0,
                 __nv_bfloat162 row1) {
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x % 32U;
  const auto atom = index / 8U;
  const auto column = (index % 8U) ^ (lane % 8U);
  auto *target = reinterpret_cast<unsigned char *>(tile) + warp * 16U * 128U +
                 atom * Rows * 128U + lane * 128U + column * 16U;
  hopper::StoreMatrixX2<__nv_bfloat162>::copy(row0, row1, target);
}

template <unsigned Rows = 128U>
__device__ __forceinline__ unsigned m_fragment_vector_offset(unsigned row,
                                                             unsigned column) {
  return (column / 64U) * Rows * 64U + row * 64U +
         ((column % 64U) ^ ((row % 8U) * 8U));
}

// The integer CUTLASS overload adds eight to a barrier ID. Unified compute
// uses physical 1..4 for its task/math joins and 7 for token reduction.
// Communication owns 6/8/9; the math Return publisher separately uses 10.
template <bool Unified>
__device__ __forceinline__ auto physical_barrier(unsigned id) {
  // Physical barrier 0 belongs to full-CTA startup/retirement. Unused
  // control warps in a larger layout can reach retirement while compute
  // still publishes its first task, so its rendezvous must use another ID.
  return static_cast<cutlass::arch::ReservedNamedBarriers>(
      Unified ? (id == kTaskReadyBarrier ? 4U : id) : id + 8U);
}

#include "cuda/weftmoe_pf/checkpoint_copy.cuh"
#include "cuda/weftmoe_pf/fc1_epilogue.cuh"
#include "cuda/weftmoe_pf/fc2_epilogue.cuh"

} // namespace forward_compute_detail
#endif
