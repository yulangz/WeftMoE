// Canonical local-X diagnostic: production math loops and SwiGLU epilogue,
// static paired tasks, no routing/communication. FC2 is a separate fixed GEMM.
// Args: packed|separated fc1|chain tokens world balanced|hotspot device
#include "cuda/forward/tuning_config.hpp"
#include "cuda/primitives/hopper.cuh"
#include "tuning.hpp"
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
constexpr unsigned kMathThreads = 256, kMathReadyBarrier = 3,
                   kMathDoneBarrier = 4;
template <bool Unified>
__device__ constexpr auto physical_barrier(unsigned id) {
  return static_cast<cutlass::arch::ReservedNamedBarriers>(id + 1U);
}
#include "static_gemm_helpers.cuh"
struct ForwardKernelArgs {
  unsigned hidden, intermediate;
};
#include "cuda/static_v4/fc1_epilogue.cuh"
#include "gemm_benchmark.cuh"

struct Mblock {
  unsigned row, expert, rows;
};
constexpr unsigned LogicalN = WEFTMOE_GEMM_LOGICAL_N;
constexpr unsigned Hidden = forward_tuning::kMainHidden;
constexpr unsigned Intermediate = forward_tuning::kMainIntermediate;
constexpr unsigned Ctas = 132;

template <bool Fc1, bool Packed> constexpr unsigned operand_bytes() {
  constexpr unsigned stages = Fc1 ? WEFTMOE_GEMM_STAGES : 2U;
  constexpr unsigned k = Fc1 ? WEFTMOE_GEMM_TILE_K : 128U;
  constexpr unsigned b = Fc1 ? 2U * LogicalN : 256U;
  return 2U * stages * (128U + b) * k;
}
template <bool Fc1, bool Packed> constexpr unsigned shared_bytes() {
  constexpr unsigned stages = Fc1 ? WEFTMOE_GEMM_STAGES : 2U;
  return operand_bytes<Fc1, Packed>() + 3U * stages * sizeof(ForwardBarrier);
}

template <bool Fc1, bool Packed, bool Producer>
__device__ __noinline__ void
gemm_role(const CUtensorMap &am, const CUtensorMap &bm, const Mblock *blocks,
          unsigned block_count, const float *gates, transport::BFloat16 *output,
          unsigned char *storage) {
  constexpr unsigned Fc1TileN = Packed ? 2U * LogicalN : LogicalN;
  constexpr unsigned Fc2TileN = 256U;
  constexpr unsigned TileK = Fc1 ? WEFTMOE_GEMM_TILE_K : 128U;
  constexpr unsigned Fc1StageCount = WEFTMOE_GEMM_STAGES;
  constexpr unsigned Fc2StageCount = 2U;
  constexpr unsigned Stages = Fc1 ? Fc1StageCount : Fc2StageCount;
  constexpr unsigned kHidden = Hidden;
  constexpr unsigned kAStageElements = 128U * TileK;
  constexpr unsigned kFc1BStageElements = Fc1TileN * TileK;
  constexpr unsigned kFc2BStageElements = Fc2TileN * TileK;
  constexpr unsigned kFc2Iterations = Intermediate / TileK;
  constexpr unsigned TileN = Fc1 ? Fc1TileN : Fc2TileN;
  constexpr unsigned Ntiles = Fc1 ? Intermediate / LogicalN : Hidden / Fc2TileN;
  constexpr unsigned K = Fc1 ? Hidden : Intermediate;
  static_assert(Ntiles % 2U == 0U);
  static_assert(shared_bytes<Fc1, Packed>() <= 227U * 1024U);
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + Stages * kAStageElements;
  auto *fc1_b2 = smem_b + Fc1StageCount * kFc1BStageElements;
  auto *fc2_smem_b = smem_b;
  auto *full = reinterpret_cast<ForwardBarrier *>(storage +
                                                  operand_bytes<Fc1, Packed>());
  auto *empty = full + Stages;
  auto *fc1_b2_full = empty + Stages;
  auto *fc1_full = full, *fc1_empty = empty;
  auto *fc2_full = full, *fc2_empty = empty;
  const auto warpgroup = threadIdx.x / 128U;
  unsigned fc1_pipeline = 0U, fc2_pipeline = 0U;
  const ForwardKernelArgs a{Hidden, Intermediate};
  for (unsigned work = blockIdx.x / 2U; work < block_count * (Ntiles / 2U);
       work += gridDim.x / 2U) {
    const auto block = blocks[work / (Ntiles / 2U)];
    const auto row = block.row, valid_rows = block.rows;
    const auto n_tile = 2U * (work % (Ntiles / 2U)) + (blockIdx.x & 1U);
    const auto n = n_tile * TileN;
    if constexpr (Producer) {
      if (threadIdx.x == 256U) {
        auto &pipeline = Fc1 ? fc1_pipeline : fc2_pipeline;
        for (unsigned iteration = 0U; iteration < K / TileK; ++iteration) {
          const auto current = pipeline + iteration;
          const auto stage = current % Stages;
          empty[stage].wait(((current / Stages) & 1U) ^ 1U);
          hopper::tma_load<TileK, 128, 128, cutlass::bfloat16_t>(
              &am, &full[stage], smem_a + stage * kAStageElements,
              iteration * TileK, row, 2U);
          hopper::tma_load<TileK, TileN, 128, cutlass::bfloat16_t>(
              &bm, &full[stage], smem_b + stage * TileN * TileK,
              iteration * TileK,
              block.expert * (Fc1 ? 2U * Intermediate : Hidden) + n);
          full[stage].arrive_and_expect_tx((kAStageElements + TileN * TileK) *
                                           2U);
          if constexpr (Fc1 && !Packed) {
            hopper::tma_load<TileK, Fc1TileN, 128, cutlass::bfloat16_t>(
                &bm, &fc1_b2_full[stage], fc1_b2 + stage * kFc1BStageElements,
                iteration * TileK,
                block.expert * 2U * Intermediate + Intermediate + n);
            fc1_b2_full[stage].arrive_and_expect_tx(kFc1BStageElements * 2U);
          }
        }
        pipeline += K / TileK;
      }
    } else {
      if constexpr (Fc1 && Packed) {
#include "fc1_packed_math.cuh"
        static_v4_store_fc1_tile<Fc1TileN>(a, smem_b, accum, output, gates, row,
                                           valid_rows, n / 2U);
      } else if constexpr (Fc1) {
#include "fc1_separated_math.cuh"
        // Present the two fragments to the same production SwiGLU epilogue.
        // This is a compile-time register permutation, not a GM layout change.
        float packed[2U * Fc1TileMma::kNumAccum];
#pragma unroll
        for (unsigned index = 0; index < Fc1TileMma::kNumAccum / 4U; ++index)
#pragma unroll
          for (unsigned value = 0; value < 4U; ++value) {
            packed[8U * index + value] = accum_a[4U * index + value];
            packed[8U * index + 4U + value] = accum_b[4U * index + value];
          }
        static_v4_store_fc1_tile<2U * Fc1TileN>(a, smem_b, packed, output,
                                                gates, row, valid_rows, n);
      } else {
#include "fc2_math.cuh"
        // Both readers finish before reusing CTA-private B, never multicast A.
        cutlass::arch::NamedBarrier::sync(
            kMathThreads, physical_barrier<true>(kMathReadyBarrier));
#pragma unroll
        for (unsigned index = 0; index < Fc2TileMma::kNumAccum / 4U; ++index)
          store_m128_fragment(
              smem_b, index,
              __float22bfloat162_rn(
                  make_float2(accum[4U * index], accum[4U * index + 1U])),
              __float22bfloat162_rn(
                  make_float2(accum[4U * index + 2U], accum[4U * index + 3U])));
        cutlass::arch::NamedBarrier::sync(
            kMathThreads, physical_barrier<true>(kMathDoneBarrier));
        for (unsigned vector = threadIdx.x; vector < valid_rows * Fc2TileN / 8U;
             vector += kMathThreads) {
          const auto m = vector / (Fc2TileN / 8U);
          const auto column = (vector % (Fc2TileN / 8U)) * 8U;
          *reinterpret_cast<uint4 *>(output + std::uint64_t(row + m) * Hidden +
                                     n + column) =
              *reinterpret_cast<const uint4 *>(
                  smem_b + m128_fragment_vector_offset(m, column));
        }
      }
    }
    __syncthreads();
  }
}

template <bool Fc1, bool Packed>
__global__ __launch_bounds__(384, 1) void static_gemm(
    const __grid_constant__ CUtensorMap am,
    const __grid_constant__ CUtensorMap bm, const Mblock *blocks,
    unsigned block_count, const float *gates, transport::BFloat16 *output) {
  extern __shared__ __align__(1024) unsigned char storage[];
  constexpr unsigned Stages = Fc1 ? WEFTMOE_GEMM_STAGES : 2U;
  if (threadIdx.x == 256U) {
    auto *full = reinterpret_cast<ForwardBarrier *>(
        storage + operand_bytes<Fc1, Packed>());
    for (unsigned stage = 0; stage < Stages; ++stage) {
      full[stage].init(1U);
      full[Stages + stage].init(16U);
      if constexpr (Fc1 && !Packed)
        full[2U * Stages + stage].init(1U);
    }
    cutlass::arch::fence_barrier_init();
  }
  cute::cluster_sync();
  // Separate role codegen, as in the unified production entry: producer
  // live ranges must not force spills in the asynchronous math fragments.
  if (threadIdx.x >= 256U) {
    cutlass::arch::warpgroup_reg_dealloc<64>();
    gemm_role<Fc1, Packed, true>(am, bm, blocks, block_count, gates, output,
                                 storage);
  } else {
    cutlass::arch::warpgroup_reg_alloc<216>();
    gemm_role<Fc1, Packed, false>(am, bm, blocks, block_count, gates, output,
                                  storage);
  }
  cute::cluster_sync();
}

__global__ void pack_weights(const transport::BFloat16 *source,
                             transport::BFloat16 *destination,
                             std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += gridDim.x * blockDim.x) {
    const auto row = (index / Hidden) % (2U * Intermediate);
    const auto packed =
        (row % Intermediate / 8U) * 16U + (row / Intermediate) * 8U + row % 8U;
    const auto expert = index / (2U * Intermediate * Hidden);
    destination[(expert * 2U * Intermediate + packed) * Hidden +
                index % Hidden] = source[index];
  }
}
__global__ void fill_gates(float *gates, unsigned rows) {
  for (unsigned row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x)
    gates[row] = 0.125F + float(row % 7U) / 32.0F;
}
__global__ void reference_swiglu(const transport::BFloat16 *ab,
                                 const float *gates, transport::BFloat16 *s,
                                 std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += gridDim.x * blockDim.x) {
    const auto row = index / Intermediate, col = index % Intermediate;
    s[index] = transport::BFloat16(gated_swiglu_fc2_input(
        float(ab[row * 2U * Intermediate + col]),
        float(ab[row * 2U * Intermediate + Intermediate + col]), gates[row]));
  }
}

int main(int argc, char **argv) try {
  if (argc != 7)
    throw std::runtime_error("args: packed|separated fc1|chain tokens world "
                             "balanced|hotspot device");
  const std::string layout = argv[1], op = argv[2], distribution = argv[5];
  const unsigned tokens = std::stoul(argv[3]), world = std::stoul(argv[4]);
  const int device = std::stoi(argv[6]);
  if ((layout != "packed" && layout != "separated") ||
      (op != "fc1" && op != "chain") ||
      (distribution != "balanced" && distribution != "hotspot") || !world ||
      64U % world || !tokens || (layout == "separated" && LogicalN < 64U))
    throw std::runtime_error("unsupported diagnostic shape/layout");
  CK(cudaSetDevice(device));
  DR(cuInit(0));
  const unsigned experts = 64U, top_k = 4U, owned = experts / world;
  const unsigned rank = distribution == "hotspot" ? world - 1U : 0U;
  std::vector<unsigned> counts(owned), bases(owned);
  for (unsigned source = 0; source < world; ++source)
    for (unsigned token = 0; token < tokens; ++token)
      for (unsigned slot = 0; slot < top_k; ++slot) {
        const auto expert =
            distribution == "hotspot"
                ? experts - 1U
                : (token * top_k + slot + source * 7U) % experts;
        if (expert / owned == rank)
          ++counts[expert % owned];
      }
  unsigned rows = 0U, valid = 0U;
  std::vector<Mblock> blocks;
  for (unsigned expert = 0; expert < owned; ++expert) {
    bases[expert] = rows;
    for (unsigned first = 0; first < counts[expert]; first += 128U) {
      blocks.push_back({rows, expert, std::min(128U, counts[expert] - first)});
      rows += 128U;
    }
    valid += counts[expert];
  }
  transport::BFloat16 *x, *w1, *packed_w1, *w2, *s, *c, *ab, *sample;
  float *gates;
  Mblock *device_blocks;
  const std::size_t w1_elements =
      std::size_t(owned) * 2U * Intermediate * Hidden;
  CK(cudaMalloc(&x, std::size_t(rows) * Hidden * 2U));
  CK(cudaMalloc(&w1, w1_elements * 2U));
  CK(cudaMalloc(&packed_w1, w1_elements * 2U));
  CK(cudaMalloc(&w2, std::size_t(owned) * Hidden * Intermediate * 2U));
  CK(cudaMalloc(&s, std::size_t(rows) * Intermediate * 2U));
  CK(cudaMalloc(&c, std::size_t(rows) * Hidden * 2U));
  CK(cudaMalloc(&ab, std::size_t(rows) * 2U * Intermediate * 2U));
  CK(cudaMalloc(&sample, 1024));
  CK(cudaMalloc(&gates, std::size_t(rows) * sizeof(float)));
  CK(cudaMalloc(&device_blocks, blocks.size() * sizeof(Mblock)));
  CK(cudaMemcpy(device_blocks, blocks.data(), blocks.size() * sizeof(Mblock),
                cudaMemcpyHostToDevice));
  fill<<<256, 256>>>(x, std::size_t(rows) * Hidden);
  fill<<<256, 256>>>(w1, w1_elements);
  fill<<<256, 256>>>(w2, std::size_t(owned) * Hidden * Intermediate);
  fill_gates<<<256, 256>>>(gates, rows);
  CK(cudaMemset(s, 0, std::size_t(rows) * Intermediate * 2U));
  CK(cudaMemset(c, 0, std::size_t(rows) * Hidden * 2U));
  CK(cudaMemset(ab, 0, std::size_t(rows) * 2U * Intermediate * 2U));
  cudaEvent_t pack_begin, pack_end;
  CK(cudaEventCreate(&pack_begin));
  CK(cudaEventCreate(&pack_end));
  CK(cudaEventRecord(pack_begin));
  pack_weights<<<256, 256>>>(w1, packed_w1, w1_elements);
  CK(cudaEventRecord(pack_end));
  CK(cudaEventSynchronize(pack_end));
  float pack_ms;
  CK(cudaEventElapsedTime(&pack_ms, pack_begin, pack_end));
  CK(cudaEventDestroy(pack_begin));
  CK(cudaEventDestroy(pack_end));
  const auto xm = tensor_map(x, Hidden, rows, 128);
  const auto sm = tensor_map(s, Intermediate, rows, 128);
  const auto wp =
      tensor_map(packed_w1, Hidden, owned * 2U * Intermediate, 2U * LogicalN);
  const auto ws = tensor_map(w1, Hidden, owned * 2U * Intermediate, LogicalN);
  const auto w2m = tensor_map(w2, Intermediate, owned * Hidden, 256);
  CK(cudaFuncSetAttribute(static_gemm<true, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          shared_bytes<true, true>()));
  if constexpr (LogicalN >= 64U)
    CK(cudaFuncSetAttribute(static_gemm<true, false>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            shared_bytes<true, false>()));
  CK(cudaFuncSetAttribute(static_gemm<false, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          shared_bytes<false, true>()));
  cudaLaunchAttribute cluster{};
  cluster.id = cudaLaunchAttributeClusterDimension;
  cluster.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t config{};
  CK(cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking));
  config.gridDim = dim3(Ctas);
  config.blockDim = dim3(384);
  config.attrs = &cluster;
  config.numAttrs = 1;
  const auto launch = [&] {
    config.dynamicSmemBytes = shared_bytes<true, true>();
    if (layout == "packed")
      CK(cudaLaunchKernelEx(&config, static_gemm<true, true>, xm, wp,
                            device_blocks, unsigned(blocks.size()), gates, s));
    else if constexpr (LogicalN >= 64U)
      CK(cudaLaunchKernelEx(&config, static_gemm<true, false>, xm, ws,
                            device_blocks, unsigned(blocks.size()), gates, s));
    if (op == "chain") {
      config.dynamicSmemBytes = shared_bytes<false, true>();
      CK(cudaLaunchKernelEx(&config, static_gemm<false, true>, sm, w2m,
                            device_blocks, unsigned(blocks.size()), gates, c));
    }
  };
  cublasHandle_t handle;
  BL(cublasCreate(&handle));
  BL(cublasSetStream(handle, config.stream));
  BL(cublasSetMathMode(handle,
                       CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  const float alpha = 1.0F, beta = 0.0F;
  const auto reference = [&] {
    for (unsigned expert = 0; expert < owned; ++expert) {
      if (!counts[expert])
        continue;
      BL(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, 2U * Intermediate, counts[expert],
          Hidden, &alpha, w1 + std::size_t(expert) * 2U * Intermediate * Hidden,
          CUDA_R_16BF, Hidden, x + std::size_t(bases[expert]) * Hidden,
          CUDA_R_16BF, Hidden, &beta,
          ab + std::size_t(bases[expert]) * 2U * Intermediate, CUDA_R_16BF,
          2U * Intermediate, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    reference_swiglu<<<256, 256, 0, config.stream>>>(
        ab, gates, s, std::size_t(rows) * Intermediate);
    if (op == "chain")
      for (unsigned expert = 0; expert < owned; ++expert) {
        if (!counts[expert])
          continue;
        BL(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, Hidden, counts[expert],
            Intermediate, &alpha,
            w2 + std::size_t(expert) * Hidden * Intermediate, CUDA_R_16BF,
            Intermediate, s + std::size_t(bases[expert]) * Intermediate,
            CUDA_R_16BF, Intermediate, &beta,
            c + std::size_t(bases[expert]) * Hidden, CUDA_R_16BF, Hidden,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
  };
  std::printf("STATIC_GEMM layout=%s op=%s tokens=%u world=%u rank=%u "
              "distribution=%s valid_rows=%u padded_rows=%u blocks=%zu "
              "logical_n=%u stages=%u tile_k=%u pack_ms=%.6f "
              "extra_weight_bytes=%zu fc1_smem=%u fc2_smem=%u\n",
              layout.c_str(), op.c_str(), tokens, world, rank,
              distribution.c_str(), valid, rows, blocks.size(), LogicalN,
              unsigned(WEFTMOE_GEMM_STAGES), unsigned(WEFTMOE_GEMM_TILE_K),
              pack_ms, w1_elements * 2U, shared_bytes<true, true>(),
              shared_bytes<false, true>());
  // Report useful GEMM FLOPs from valid rows, not padded M128 work. SwiGLU is
  // timed but its scalar arithmetic is not counted as tensor-core throughput.
  std::printf("VALID_FLOPS fc1=%.0f fc2=%.0f\n",
              4.0 * valid * Hidden * Intermediate,
              op == "chain" ? 2.0 * valid * Hidden * Intermediate : 0.0);
  compare_gemms(launch, reference, op == "fc1" ? s : c, sample,
                std::size_t(rows) * (op == "fc1" ? Intermediate : Hidden),
                layout + "-" + op, rows, Intermediate, Hidden, 1, Ctas, device,
                false, config.stream,
                (op == "chain" ? 6.0 : 4.0) * valid * Hidden * Intermediate);
  BL(cublasDestroy(handle));
  CK(cudaStreamDestroy(config.stream));
  for (auto *pointer : {x, w1, packed_w1, w2, s, c, ab, sample})
    CK(cudaFree(pointer));
  CK(cudaFree(gates));
  CK(cudaFree(device_blocks));
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
