// Calls production BWD GEMMs on balanced, statically assigned expert tiles.
// Run arguments: op M N K groups device compute_ctas reverse_order share_b.
// share_b is reserved for the forward diagnostic and must be zero here.
#include "tuning.hpp"

#include "cuda/backward/tuning_config.hpp"
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

#include "cuda/persistent_v3/backward/cluster_task.cuh"
#include "cuda/persistent_v3/backward/compute_kernels.cuh"

template <int Op>
__global__ __launch_bounds__(384, 1) void backward_gemm(
    const __grid_constant__ CUtensorMap am,
    const __grid_constant__ CUtensorMap bm,
    const __grid_constant__ CUtensorMap om, transport::BFloat16 *output, int M,
    int N, int K, int groups) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using namespace backward_compute_detail;
  constexpr bool Dw = Op >= 2;
  constexpr auto storage_n =
      std::max(backward_tuning::kDabTileN, backward_tuning::kDxTileN);
  extern __shared__ __align__(1024) unsigned char storage[];
  initialize_k_mn_pipeline<storage_n>(storage);
  unsigned pipeline = 0;
  constexpr auto tile_m =
      Dw ? backward_tuning::kDwTileM : backward_tuning::kActivationTileM;
  constexpr auto tile_n =
      Dw ? backward_tuning::kDwTileN
         : (Op == 0 ? backward_tuning::kDabTileN : backward_tuning::kDxTileN);
  const int nt = N / tile_n, mt = M / tile_m;
  if constexpr (Dw) {
    constexpr int claim = Op == 2 ? backward_tuning::kDw1ClaimTiles
                                  : backward_tuning::kDw2ClaimTiles;
    const int pairs = (mt * nt + 1) / 2;
    const int chunks = (pairs + claim - 1) / claim;
    for (int task = blockIdx.x / 2; task < groups * chunks;
         task += gridDim.x / 2) {
      const int expert = task / chunks;
      const int first = task % chunks * claim;
      const int count = std::min(pairs - first, claim);
      run_mn_mn_chunk<tile_m, tile_n, storage_n>(
          am, bm, om, first, count, mt, nt, expert * K, expert * K, K, N,
          expert * M, storage, pipeline);
      cute::cluster_sync();
    }
  } else {
    constexpr int pair_n = backward_tuning::kActivationClusterN;
    const int np = nt / pair_n, mp = M / 128;
    for (int task = blockIdx.x / 2; task < groups * mp * np;
         task += gridDim.x / 2) {
      const int expert = task / (mp * np);
      const int row = task / np % mp * 128;
      const int col = task % np * pair_n * tile_n;
      run_k_mn_tile_tma_store<tile_n, storage_n>(
          am, bm, om, expert * M + row, expert * K, K, expert * M + row, col,
          storage, pipeline);
      cute::cluster_sync();
    }
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
  const bool dw = op == "dw1" || op == "dw2";
  if (share_b)
    throw std::runtime_error(
        "share_b is only supported by the forward FC1 diagnostic");
  if (op != "du" && op != "dx" && !dw)
    return 2;
  if (M <= 0 || M % 128 || N <= 0 || N % 256 || K <= 0 || K % 64 ||
      groups <= 0 || ctas <= 0 || ctas % 2)
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
  const auto am =
      dw ? tensor_map(left, M, groups * K, 64)
         : tensor_map(left, K, groups * M, backward_tuning::kActivationTileM);
  const auto bm = tensor_map(right, N, groups * K, 64);
  const auto om = tensor_map(output, N, groups * M,
                             dw ? 128 : backward_tuning::kActivationTileM);
  constexpr int storage_n =
      std::max(backward_tuning::kDabTileN, backward_tuning::kDxTileN);
  constexpr int shared_bytes =
      backward_compute_detail::dynamic_shared_bytes<storage_n>();
  CK(cudaFuncSetAttribute(backward_gemm<0>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          shared_bytes));
  CK(cudaFuncSetAttribute(backward_gemm<1>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          shared_bytes));
  CK(cudaFuncSetAttribute(backward_gemm<2>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          shared_bytes));
  CK(cudaFuncSetAttribute(backward_gemm<3>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          shared_bytes));
  cudaLaunchAttribute cluster_attr{};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t config{};
  CK(cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking));
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(384);
  config.attrs = &cluster_attr;
  config.numAttrs = 1;
  config.dynamicSmemBytes = shared_bytes;
  const auto launch = [&] {
    if (op == "du")
      CK(cudaLaunchKernelEx(&config, backward_gemm<0>, am, bm, om, output, M, N,
                            K, groups));
    else if (op == "dx")
      CK(cudaLaunchKernelEx(&config, backward_gemm<1>, am, bm, om, output, M, N,
                            K, groups));
    else if (op == "dw1")
      CK(cudaLaunchKernelEx(&config, backward_gemm<2>, am, bm, om, output, M, N,
                            K, groups));
    else
      CK(cudaLaunchKernelEx(&config, backward_gemm<3>, am, bm, om, output, M, N,
                            K, groups));
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
        handle, CUBLAS_OP_N, dw ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
        right, CUDA_R_16BF, N, std::size_t(N) * K, left, CUDA_R_16BF,
        dw ? M : K, std::size_t(M) * K, &beta, output, CUDA_R_16BF, N,
        std::size_t(M) * N, groups, CUBLAS_COMPUTE_32F,
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
