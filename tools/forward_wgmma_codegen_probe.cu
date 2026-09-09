#include "cuda/primitives/hopper.cuh"
#include <cuda_runtime.h>
#include <cutlass/arch/reg_reconfig.h>

using Mma128 = typename weftmoe::cuda::hopper::BF16MMASelector<128>::type;
using Mma256 = typename weftmoe::cuda::hopper::BF16MMASelector<256>::type;

template <class Mma, bool Dual>
__global__ __launch_bounds__(384, 1) void wgmma_codegen_probe(float *output) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) cutlass::bfloat16_t smem[];
  const auto warpgroup =
      __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup >= 2U) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<232>();
  float accum_a[Mma::kNumAccum] = {0.0F};
  float accum_b[Mma::kNumAccum] = {0.0F};
#pragma unroll
  for (int index = 0; index < Mma::kNumAccum; ++index) {
    cute::warpgroup_fence_operand(accum_a[index]);
    if constexpr (Dual)
      cute::warpgroup_fence_operand(accum_b[index]);
  }
  cute::warpgroup_arrive();
#pragma unroll
  for (unsigned k = 0; k < 64U / Mma::K; ++k) {
    const auto desc_a = weftmoe::cuda::hopper::make_smem_desc(
        smem + warpgroup * 64U * 64U + k * Mma::K, 1);
    const auto desc_b = weftmoe::cuda::hopper::make_smem_desc(
        smem + 2U * 64U * 64U + k * Mma::K, 1);
    const auto desc_b2 = weftmoe::cuda::hopper::make_smem_desc(
        smem + 4U * 64U * 64U + k * Mma::K, 1);
    Mma::wgmma(desc_a, desc_b, accum_a, 1);
    if constexpr (Dual)
      Mma::wgmma(desc_a, desc_b2, accum_b, 1);
  }
  cute::warpgroup_commit_batch();
#pragma unroll
  for (int index = 0; index < Mma::kNumAccum; ++index) {
    cute::warpgroup_fence_operand(accum_a[index]);
    if constexpr (Dual)
      cute::warpgroup_fence_operand(accum_b[index]);
  }
  cute::warpgroup_wait<0>();
  output[static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
         threadIdx.x] = accum_a[0] + (Dual ? accum_b[0] : 0.0F);
#else
  (void)output;
#endif
}

template __global__ void wgmma_codegen_probe<Mma128, true>(float *);
template __global__ void wgmma_codegen_probe<Mma128, false>(float *);
template __global__ void wgmma_codegen_probe<Mma256, false>(float *);
