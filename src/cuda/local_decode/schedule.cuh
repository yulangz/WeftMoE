#pragma once

namespace local_decode {

constexpr unsigned kM = 128U, kN = 128U, kK = 64U, kStages = 5U;
constexpr unsigned kSparseRows = 8U;
constexpr unsigned kThreads = 384U, kMathThreads = 256U;
constexpr unsigned kMaxExperts = 256U;
constexpr unsigned kFc1Tiles = 2U * forward_tuning::kMainIntermediate / kN;
constexpr unsigned kFc2Tiles = forward_tuning::kMainHidden / kN;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using Mma = hopper::BF16Mma<kN>;

// Selection uses collective capacity, never one source's actual batch size.
// One window covers even the all-to-one-expert case. Larger pools keep V4.
inline bool eligible(const ActivationTransportPoolConfig &config,
                     const static_forward::ComputeRingLayout &ring) {
  return forward_tuning::kStaticV4PrefillEnabled &&
         forward_tuning::kClusterSize == 2U &&
         forward_tuning::kUnifiedThreads == kThreads &&
         forward_tuning::kTileM == kM && forward_tuning::kTileK == kK &&
         forward_tuning::StaticV4Gemm<true>::kFc1N == kN &&
         forward_tuning::kFc2TileN == kN &&
         config.max_nodes == 1U && !config.enable_backward_state &&
         config.max_tokens_per_rank <= 64U && ring.depth &&
         ring.max_windows == 1U && config.max_world_size <= 32U &&
         config.max_owned_experts * config.max_world_size <= kMaxExperts;
}

// Alias only the invocation-private block directory, not cumulative ring
// credits. All payload fits one generation; the terminal collective protects
// reuse. There is no online admission or within-invocation slot overwrite.
struct BlockState {
  unsigned long long x_ready, s_ready;
};
static_assert(sizeof(BlockState) <= sizeof(static_forward::RingBlock));

struct Shared {
  // Source planning retires before geometry; reuse its scratch for local rows.
  union {
    unsigned counts[kMaxExperts];
    unsigned rows[kMaxExperts];
  };
  unsigned blocks[kMaxExperts + 1U];
  alignas(1024) transport::BFloat16 a[kStages][kM * kK];
  alignas(1024) transport::BFloat16 b[kStages][kN * kK];
  alignas(1024) transport::BFloat16 output[kM * kN];
  alignas(1024) unsigned char pull[3][4096];
  Barrier full[kStages], empty[kStages], pull_full[3];
};
static_assert(sizeof(Shared) <= 227U * 1024U);

__device__ __forceinline__ BlockState *states(const ForwardKernelArgs &a) {
  return reinterpret_cast<BlockState *>(a.ring.blocks);
}

struct Task {
  unsigned expert, block, column, rows;
  bool fc2;
};

// Producer and math independently replay this immutable grid-stride sequence.
// FC1 residues continue into FC2; adjacent CTAs always share M and adjacent N.
__device__ __forceinline__ Task task(const Shared &s, unsigned owned,
                                     unsigned index) {
  const auto fc1 = s.blocks[owned] * kFc1Tiles;
  const bool fc2 = index >= fc1;
  const auto linear = fc2 ? index - fc1 : index;
  const auto columns = fc2 ? kFc2Tiles : kFc1Tiles;
  const auto block = linear / columns;
  unsigned expert = 0U;
  while (s.blocks[expert + 1U] <= block)
    ++expert;
  return {expert, block, linear % columns,
          min(kM, s.rows[expert] - (block - s.blocks[expert]) * kM), fc2};
}

} // namespace local_decode
