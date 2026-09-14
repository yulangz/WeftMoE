// Diagnostic only: preallocated balanced expert GEMMs, with the production
// WGMMA/TMA math and FC2 epilogue extracted by run_gemm_diagnostics.py.
// FC1 stores A/B but excludes SwiGLU, routing, task queues and communication.
// Run arguments: op M N K groups device compute_ctas reverse_order share_b.
#include "tuning.hpp"

#include "cuda/forward/tuning_config.hpp"
#include "cuda/primitives/hopper.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cutlass/arch/reg_reconfig.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace weftmoe::cuda;
namespace transport {
using BFloat16 = cutlass::bfloat16_t;
}
using ForwardBarrier = cutlass::arch::ClusterTransactionBarrier;
using TunedForwardFc1Mma =
    typename hopper::BF16MMASelector<forward_tuning::kFc1TileN>::type;
using TunedForwardFc2Mma =
    typename hopper::BF16MMASelector<forward_tuning::kFc2TileN>::type;

__device__ unsigned float2_to_bf16x2_bits(float a, float b) {
  auto value = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<unsigned *>(&value);
}
template <int N, class F> __device__ void visit(float *accum, F function) {
  const auto wg = threadIdx.x / 128;
  const auto warp = (threadIdx.x % 128) / 32;
  const auto lane = threadIdx.x % 32;
#pragma unroll
  for (int i = 0; i < N / 8; ++i) {
    const auto col = lane % 4 * 2 + i * 8;
    const auto row = warp * 16 + lane / 4;
    function(wg, row, col, accum[i * 4], accum[i * 4 + 1]);
    function(wg, row + 8, col, accum[i * 4 + 2], accum[i * 4 + 3]);
  }
}
constexpr unsigned kMathThreads = 256, kMathReadyBarrier = 3,
                   kMathDoneBarrier = 4;
#include "epilogue_helpers.cuh"

// Static work assignment removes route readiness and task-queue costs. The
// math branches below are extracted verbatim from the current FWD kernel.
template <bool Fc1, bool ShareB = false>
__global__ __launch_bounds__(384, 1) void forward_gemm(
    const __grid_constant__ CUtensorMap am,
    const __grid_constant__ CUtensorMap bm,
    const __grid_constant__ CUtensorMap om, transport::BFloat16 *output, int M,
    int runtime_n, int runtime_k, int groups) {
  constexpr int N =
      Fc1 ? 2 * forward_tuning::kMainIntermediate : forward_tuning::kMainHidden;
  constexpr int K =
      Fc1 ? forward_tuning::kMainHidden : forward_tuning::kMainIntermediate;
  constexpr auto kHidden = forward_tuning::kMainHidden;
  constexpr auto kIntermediate = forward_tuning::kMainIntermediate;
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr auto Fc1StageCount = forward_tuning::kFc1Stages;
  constexpr auto Fc2StageCount = forward_tuning::kFc2Stages;
  constexpr auto kAStageElements = 128U * 64U;
  constexpr auto kFc1BStageElements = forward_tuning::kFc1TileN * 64U;
  constexpr auto kFc2BStageElements = forward_tuning::kFc2TileN * 64U;
  constexpr auto stages = Fc1 ? Fc1StageCount : Fc2StageCount;
  constexpr auto tile_n =
      Fc1 ? forward_tuning::kFc1TileN : forward_tuning::kFc2TileN;
  constexpr auto b_elements = tile_n * 64U;
  constexpr bool kFc2Pipelined = forward_tuning::kFc2ClaimTiles > 1U;
  constexpr auto kFc2OutputOffset =
      Fc2StageCount * (kAStageElements + kFc2BStageElements);
  extern __shared__ __align__(1024) unsigned char bytes[];
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(bytes);
  auto *smem_b = smem_a + stages * kAStageElements;
  auto *fc2_smem_b = smem_b;
  auto *fc1_b2 = smem_b + stages * b_elements;
  auto *full = reinterpret_cast<ForwardBarrier *>(
      smem_b + stages * b_elements * (Fc1 ? 2U : 1U) +
      (!Fc1 && kFc2Pipelined ? 128U * tile_n : 0U));
  auto *empty = full + stages;
  auto *fc1_b2_full = empty + stages;
  auto *fc1_full = full;
  auto *fc1_empty = empty;
  auto *fc2_full = full;
  auto *fc2_empty = empty;
  if (threadIdx.x == 256) {
    for (unsigned i = 0; i < stages; ++i) {
      full[i].init(1);
      empty[i].init(16);
      if constexpr (Fc1)
        fc1_b2_full[i].init(1);
    }
    cutlass::arch::fence_barrier_init();
  }
  cute::cluster_sync();
  const auto warpgroup = threadIdx.x / 128;
  if (warpgroup < 2)
    cutlass::arch::warpgroup_reg_alloc<forward_tuning::kMathRegisters>();
  else
    cutlass::arch::warpgroup_reg_dealloc<forward_tuning::kProducerRegisters>();
  unsigned fc1_pipeline = 0, fc2_pipeline = 0;
  constexpr auto kFc2Iterations = forward_tuning::kMainIntermediate / 64U;
  struct {
    unsigned hidden;
  } a{unsigned(K)};
  const int nt = (Fc1 ? N / 2 : N) / tile_n;
  const int mt = M / 128;
  constexpr int chunk = Fc1 ? 1 : forward_tuning::kFc2ClaimTiles;
  const int chunks_per_row = (nt / 2 + chunk - 1) / chunk;
  constexpr bool kUnevenClaims =
      ((Fc1 ? N / 2 : N) / tile_n) % (2 * chunk) != 0;
  const int clusters = gridDim.x / 2, cluster = blockIdx.x / 2;
  const int batches = groups * mt * chunks_per_row;
  const int limit =
      kUnevenClaims ? (batches + clusters - 1) / clusters * clusters : batches;
  for (int work = cluster, wave = 0; work < limit; work += clusters, ++wave) {
    int batch = work;
    if constexpr (kUnevenClaims) {
      // Rotate ownership so strided assignment cannot give the same CTAs
      // every short final group. Each wave still covers each batch once.
      const int rotation = chunks_per_row - clusters % chunks_per_row + 1;
      batch = work - cluster + (cluster + wave * rotation) % clusters;
      if (batch >= batches)
        break;
    }
    const int first_n = batch % chunks_per_row * 2 * chunk;
    const int count = std::min(chunk, (nt - first_n) / 2);
    for (int offset = 0; offset < count; ++offset) {
      const int task = (batch / chunks_per_row) * nt + first_n + 2 * offset +
                       (blockIdx.x & 1);
      const int expert = task / (mt * nt);
      const int local = task % (mt * nt);
      const int row =
          ShareB ? (local / (2 * nt) * 2 + local % 2) * 128 : local / nt * 128;
      const int col = (ShareB ? local / 2 % nt : local % nt) * tile_n;
      if (threadIdx.x == 256) {
        auto &pipeline = Fc1 ? fc1_pipeline : fc2_pipeline;
        for (unsigned iteration = 0; iteration < unsigned(K / 64);
             ++iteration) {
          const auto current = pipeline + iteration;
          const auto stage = current % stages;
          const auto phase = current / stages & 1U;
          empty[stage].wait(phase ^ 1U);
          hopper::tma_load<64, 128, 128, cutlass::bfloat16_t>(
              &am, &full[stage], smem_a + stage * kAStageElements,
              iteration * 64, expert * M + row, ShareB ? 1 : 2);
          hopper::tma_load<64, tile_n, 128, cutlass::bfloat16_t>(
              &bm, &full[stage], smem_b + stage * b_elements, iteration * 64,
              expert * N + col, ShareB ? 2 : 1,
              static_cast<uint64_t>(
                  Fc1 ? cute::TMA::CacheHintSm90::EVICT_NORMAL
                      : cute::TMA::CacheHintSm90::EVICT_LAST));
          full[stage].arrive_and_expect_tx((kAStageElements + b_elements) * 2);
          if constexpr (Fc1) {
            hopper::tma_load<64, tile_n, 128, cutlass::bfloat16_t>(
                &bm, &fc1_b2_full[stage], fc1_b2 + stage * b_elements,
                iteration * 64, expert * N + N / 2 + col, ShareB ? 2 : 1);
            fc1_b2_full[stage].arrive_and_expect_tx(b_elements * 2);
          }
        }
        pipeline += K / 64;
      } else if (warpgroup < 2) {
        if constexpr (Fc1) {
#include "fc1_math.cuh"
          // The production checkpoint epilogue packs fragments with STSM.
          // This diagnostic writes A/B and omits SwiGLU, gates and the U store.
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
#pragma unroll
          for (unsigned index = 0; index < Fc1TileMma::kNumAccum / 4U;
               ++index) {
            store_m128_fragment(
                smem_a, index,
                __float22bfloat162_rn(
                    make_float2(accum_a[index * 4], accum_a[index * 4 + 1])),
                __float22bfloat162_rn(make_float2(accum_a[index * 4 + 2],
                                                  accum_a[index * 4 + 3])));
            store_m128_fragment(
                smem_a + 128U * tile_n, index,
                __float22bfloat162_rn(
                    make_float2(accum_b[index * 4], accum_b[index * 4 + 1])),
                __float22bfloat162_rn(make_float2(accum_b[index * 4 + 2],
                                                  accum_b[index * 4 + 3])));
          }
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);
          for (unsigned v = threadIdx.x; v < 128U * tile_n / 8U;
               v += kMathThreads) {
            const auto r = v / (tile_n / 8U), c = (v % (tile_n / 8U)) * 8U;
            const auto offset = m128_fragment_vector_offset(r, c);
            const auto index =
                (std::uint64_t(expert) * M + row + r) * N + col + c;
            *reinterpret_cast<uint4 *>(output + index) =
                *reinterpret_cast<const uint4 *>(smem_a + offset);
            *reinterpret_cast<uint4 *>(output + index + N / 2) =
                *reinterpret_cast<const uint4 *>(smem_a + 128U * tile_n +
                                                 offset);
          }
        } else {
#include "fc2_math.cuh"
          const auto *output_store_map = &om;
          const auto n = col;
#include "fc2_store.cuh"
        }
      }
    }
    if constexpr (!Fc1 && kFc2Pipelined) {
      if (threadIdx.x < tile_n / 64)
        asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
    }
    cute::cluster_sync();
  }
  cute::cluster_sync();
#endif
}

#include "gemm_benchmark.cuh"
int main(int argc, char **argv) try {
  if (argc != 10)
    return 2;
  const std::string op = argv[1];
  const int M = std::atoi(argv[2]), N = std::atoi(argv[3]),
            K = std::atoi(argv[4]);
  const int groups = std::atoi(argv[5]), device = std::atoi(argv[6]),
            ctas = std::atoi(argv[7]);
  const bool reverse = std::atoi(argv[8]);
  const bool share_b = std::atoi(argv[9]);
  if (op != "fc1" && op != "fc2")
    return 2;
  if (share_b && op != "fc1")
    throw std::runtime_error(
        "share_b is only supported by the forward FC1 diagnostic");
  if (M <= 0 || M % (share_b ? 256 : 128) || N <= 0 || N % 256 || K <= 0 ||
      K % 64 || groups <= 0 || ctas <= 0 || ctas % 2)
    return 2;
  if ((op == "fc1" && (N != 2 * forward_tuning::kMainIntermediate ||
                       K != forward_tuning::kMainHidden)) ||
      (op == "fc2" && (N != forward_tuning::kMainHidden ||
                       K != forward_tuning::kMainIntermediate)))
    return 2;
  CK(cudaSetDevice(device));
  DR(cuInit(0));
  transport::BFloat16 *left, *right, *output, *sample;
  const std::size_t na = std::size_t(groups) * M * K,
                    nb = std::size_t(groups) * N * K,
                    no = std::size_t(groups) * M * N;
  CK(cudaMalloc(&left, na * 2));
  CK(cudaMalloc(&right, nb * 2));
  CK(cudaMalloc(&output, no * 2));
  CK(cudaMalloc(&sample, 1024));
  fill<<<256, 256>>>(left, na);
  fill<<<256, 256>>>(right, nb);
  CK(cudaMemset(output, 0, no * 2));
  const auto am = tensor_map(left, K, groups * M, 128);
  const auto bm = tensor_map(right, K, groups * N,
                             op == "fc1" ? forward_tuning::kFc1TileN
                                         : forward_tuning::kFc2TileN);
  const auto om = tensor_map(output, N, groups * M, 128);
  constexpr int fc1_smem = 2 * forward_tuning::kFc1Stages *
                               (128 + 2 * forward_tuning::kFc1TileN) * 64 +
                           3 * forward_tuning::kFc1Stages * 8;
  constexpr int fc2_smem =
      2 * forward_tuning::kFc2Stages * (128 + forward_tuning::kFc2TileN) * 64 +
      2 * forward_tuning::kFc2Stages * 8 +
      (forward_tuning::kFc2ClaimTiles > 1 ? 2 * 128 * forward_tuning::kFc2TileN
                                          : 0);
  CK(cudaFuncSetAttribute(forward_gemm<true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          fc1_smem));
  CK(cudaFuncSetAttribute(forward_gemm<true, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          fc1_smem));
  CK(cudaFuncSetAttribute(forward_gemm<false>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          fc2_smem));
  cudaLaunchAttribute cluster_attr{};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t config{};
  CK(cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking));
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(384);
  config.attrs = &cluster_attr;
  config.numAttrs = 1;
  config.dynamicSmemBytes = op == "fc1" ? fc1_smem : fc2_smem;
  const auto launch = [&] {
    if (op == "fc1") {
      if (share_b)
        CK(cudaLaunchKernelEx(&config, forward_gemm<true, true>, am, bm, om,
                              output, M, N, K, groups));
      else
        CK(cudaLaunchKernelEx(&config, forward_gemm<true>, am, bm, om, output,
                              M, N, K, groups));
    } else if (op == "fc2")
      CK(cudaLaunchKernelEx(&config, forward_gemm<false>, am, bm, om, output, M,
                            N, K, groups));
    else
      throw std::runtime_error("only fc1/fc2 are supported");
    CK(cudaGetLastError());
  };
  cublasHandle_t handle;
  BL(cublasCreate(&handle));
  BL(cublasSetStream(handle, config.stream));
  BL(cublasSetMathMode(handle,
                       CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  float alpha = 1, beta = 0;
  const auto reference = [&] {
    BL(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, right, CUDA_R_16BF,
        K, std::size_t(N) * K, left, CUDA_R_16BF, K, std::size_t(M) * K, &beta,
        output, CUDA_R_16BF, N, std::size_t(M) * N, groups, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  compare_gemms(launch, reference, output, sample, no, op, M, N, K, groups,
                ctas, device, reverse, config.stream);
  BL(cublasDestroy(handle));
  CK(cudaStreamDestroy(config.stream));
  CK(cudaFree(sample));
  CK(cudaFree(left));
  CK(cudaFree(right));
  CK(cudaFree(output));
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
