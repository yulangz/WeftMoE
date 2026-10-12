// Calls production BWD GEMMs on statically assigned expert tiles.
// Arguments: op M N K groups device compute_ctas reverse_order share_b [rows_file].
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
#include <fstream>
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

#include "cuda/backward/cluster_task.cuh"
#include "cuda/backward/compute_kernels.cuh"

#ifndef WEFTMOE_GEMM_MATH_REGISTERS
#define WEFTMOE_GEMM_MATH_REGISTERS 208
#endif
template <bool Math>
struct DiagnosticSynchronization : backward_compute_detail::FullCtaComputeSynchronization {
  static constexpr bool kMathEnabled = Math;
  static constexpr bool kProducerEnabled = !Math;
  static constexpr unsigned kActivationWaitGroup = 0;
};

template <int Op, bool Math>
__device__ __forceinline__ void gemm_tasks(
    const CUtensorMap &am, const CUtensorMap &bm, const CUtensorMap &om,
    int M, int N, int K, int groups, const unsigned *prefix,
    unsigned char *storage, transport::BFloat16 *output) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using namespace backward_compute_detail;
  constexpr bool Dw = Op >= 2;
  constexpr auto storage_n =
      std::max(backward_tuning::kDabTileN, backward_tuning::kDxTileN);
  unsigned pipeline = 0;
  constexpr auto tile_m =
      Dw ? backward_tuning::kDwTileM : backward_tuning::kActivationTileM;
  constexpr auto tile_n =
      Dw ? backward_tuning::kDwTileN
         : (Op == 0 ? backward_tuning::kDabTileN : backward_tuning::kDxTileN);
  const int nt = N / tile_n, mt = M / tile_m;
  if constexpr (Dw) {
#if defined(WEFTMOE_GEMM_DW_CLAIM)
    constexpr int claim = WEFTMOE_GEMM_DW_CLAIM;
#else
    constexpr int claim = Op == 2 ? backward_tuning::kDw1ClaimTiles
                                  : backward_tuning::kDw2ClaimTiles;
#endif
    const int pairs = (mt * nt + 1) / 2;
    const int chunks = (pairs + claim - 1) / claim;
    for (int task = blockIdx.x / 2; task < groups * chunks;
         task += gridDim.x / 2) {
      const int expert = task / chunks;
      const int first = task % chunks * claim;
      const int count = std::min(pairs - first, claim);
      const auto start = prefix ? prefix[expert] : expert * K;
      const auto reduction = prefix ? prefix[expert + 1] - start : K;
      run_mn_mn_chunk<tile_m, tile_n, storage_n, DiagnosticSynchronization<Math>>(
          am, bm, om, first, count, mt, nt, start, start, reduction, N,
          expert * M, storage, pipeline, output);
      cute::cluster_sync();
    }
  } else {
    constexpr int pair_n = backward_tuning::kActivationClusterN;
    const int np = nt / pair_n, mp = M / 128;
    const auto blocks = prefix ? prefix[groups] / 128U : groups * mp;
    unsigned expert_cursor = 0U;
    for (int task = blockIdx.x / 2; task < blocks * np;
         task += gridDim.x / 2) {
      const auto row = task / np * 128U;
      if (prefix)
        while (row >= prefix[expert_cursor + 1U])
          ++expert_cursor;
      const auto expert = prefix ? expert_cursor : task / (mp * np);
      const int col = task % np * pair_n * tile_n;
      run_k_mn_tile_tma_store<tile_n, storage_n, DiagnosticSynchronization<Math>>(
          am, bm, om, row, expert * K, K, row, col,
          storage, pipeline);
      cute::cluster_sync();
    }
  }
  cute::cluster_sync();
#endif
}

template <int Op>
__global__ __launch_bounds__(384, 1) void backward_gemm(
    const __grid_constant__ CUtensorMap am,
    const __grid_constant__ CUtensorMap bm,
    const __grid_constant__ CUtensorMap om, transport::BFloat16 *output, int M,
    int N, int K, int groups, const unsigned *prefix) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr auto storage_n = std::max(backward_tuning::kDabTileN,
                                      backward_tuning::kDxTileN);
  extern __shared__ __align__(1024) unsigned char storage[];
  backward_compute_detail::initialize_k_mn_barriers<storage_n>(storage);
  cute::cluster_sync();
  if (threadIdx.x < 256) {
    cutlass::arch::warpgroup_reg_alloc<WEFTMOE_GEMM_MATH_REGISTERS>();
    gemm_tasks<Op, true>(am, bm, om, M, N, K, groups, prefix, storage, output);
  } else {
    cutlass::arch::warpgroup_reg_dealloc<504 - 2 * WEFTMOE_GEMM_MATH_REGISTERS>();
    gemm_tasks<Op, false>(am, bm, om, M, N, K, groups, prefix, storage, output);
  }
#endif
}
#include "gemm_benchmark.cuh"

__global__ void clear_padding(transport::BFloat16 *p, unsigned width,
                              const unsigned *prefix, const unsigned *rows) {
  const auto expert = blockIdx.x;
  const auto start = std::size_t(prefix[expert] + rows[expert]) * width;
  const auto end = std::size_t(prefix[expert + 1U]) * width;
  for (auto i = start + threadIdx.x; i < end; i += blockDim.x)
    p[i] = transport::BFloat16(0.0F);
}

int main(int argc, char **argv) try {
  if (argc != 10 && argc != 11)
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
  std::vector<unsigned> rows, offsets;
  unsigned *device_prefix = nullptr, *device_rows = nullptr;
  if (argc == 11) {
    std::ifstream input(argv[10]);
    int value;
    while (input >> value) {
      if (value < 0)
        throw std::runtime_error("Expert rows must be nonnegative");
      rows.push_back(value);
    }
    if (rows.size() != std::size_t(groups))
      throw std::runtime_error("Expected one row count per expert");
    const unsigned alignment = dw ? 64U : 128U;
    offsets.push_back(0U);
    for (auto row : rows)
      offsets.push_back(offsets.back() + (row + alignment - 1U) / alignment * alignment);
    if (!offsets.back())
      throw std::runtime_error("The diagnostic needs at least one active expert");
    CK(cudaMalloc(&device_prefix, offsets.size() * sizeof(unsigned)));
    CK(cudaMalloc(&device_rows, rows.size() * sizeof(unsigned)));
    CK(cudaMemcpy(device_prefix, offsets.data(), offsets.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(device_rows, rows.data(), rows.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  }
  const auto total_rows = device_prefix ? offsets.back() : groups * (dw ? K : M);
  transport::BFloat16 *left, *right, *output, *sample;
  const std::size_t na = std::size_t(total_rows) * (dw ? M : K),
                    nb = std::size_t(dw ? total_rows : groups * K) * N,
                    no = std::size_t(dw ? groups * M : total_rows) * N;
  CK(cudaMalloc(&left, na * 2));
  CK(cudaMalloc(&right, nb * 2));
  CK(cudaMalloc(&output, no * 2));
  CK(cudaMalloc(&sample, 1024));
  fill<<<256, 256>>>(left, na);
  fill<<<256, 256>>>(right, nb);
  if (device_prefix) {
    clear_padding<<<groups, 256>>>(left, dw ? M : K, device_prefix, device_rows);
    if (dw)
      clear_padding<<<groups, 256>>>(right, N, device_prefix, device_rows);
  }
  CK(cudaMemset(output, 0, no * 2));
  const auto am =
      dw ? tensor_map(left, M, total_rows, 64)
         : tensor_map(left, K, total_rows, backward_tuning::kActivationTileM);
  const auto bm = tensor_map(right, N, dw ? total_rows : groups * K, 64);
  const auto om = tensor_map(output, N, dw ? groups * M : total_rows,
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
                            K, groups, device_prefix));
    else if (op == "dx")
      CK(cudaLaunchKernelEx(&config, backward_gemm<1>, am, bm, om, output, M, N,
                            K, groups, device_prefix));
    else if (op == "dw1")
      CK(cudaLaunchKernelEx(&config, backward_gemm<2>, am, bm, om, output, M, N,
                            K, groups, device_prefix));
    else
      CK(cudaLaunchKernelEx(&config, backward_gemm<3>, am, bm, om, output, M, N,
                            K, groups, device_prefix));
    CK(cudaGetLastError());
  };
  cublasHandle_t handle;
  BL(cublasCreate(&handle));
  BL(cublasSetStream(handle, config.stream));
  BL(cublasSetMathMode(handle,
                       CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  float alpha = 1, beta = 0;
  const auto reference = [&] {
    if (device_prefix) {
      // This is a graph of per-expert cuBLAS GEMMs, not TE grouped GEMM.
      for (int expert = 0; expert < groups; ++expert) {
        if (!rows[expert])
          continue;
        const auto start = std::size_t(offsets[expert]);
        BL(cublasGemmEx(
            handle, CUBLAS_OP_N, dw ? CUBLAS_OP_T : CUBLAS_OP_N, N,
            dw ? M : rows[expert], dw ? rows[expert] : K, &alpha,
            right + (dw ? start : std::size_t(expert) * K) * N,
            CUDA_R_16BF, N, left + start * (dw ? M : K), CUDA_R_16BF,
            dw ? M : K, &beta,
            output + (dw ? std::size_t(expert) * M : start) * N,
            CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      return;
    }
    BL(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, dw ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
        right, CUDA_R_16BF, N, std::size_t(N) * K, left, CUDA_R_16BF,
        dw ? M : K, std::size_t(M) * K, &beta, output, CUDA_R_16BF, N,
        std::size_t(M) * N, groups, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  const auto work = device_prefix
      ? 2.0 * std::accumulate(rows.begin(), rows.end(), std::uint64_t{0}) * N * (dw ? M : K)
      : 0.0;
  compare_gemms(launch, reference, output, sample, no, op, M, N, K, groups,
                ctas, device, reverse, config.stream, work);
  if (device_prefix) {
    std::printf("RAGGED_ROWS");
    for (auto row : rows)
      std::printf(" %u", row);
    std::printf("\n");
    CK(cudaFree(device_prefix));
    CK(cudaFree(device_rows));
  }
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
