#pragma once

namespace local_decode {

// Keep the shared M128 GM directory; decode arithmetic uses M16 or M64
// slices. Hot experts traverse every slice, including multi-block experts.
#ifndef WEFTMOE_FWD_LOCAL_DECODE_M
#define WEFTMOE_FWD_LOCAL_DECODE_M 0
#endif
#ifndef WEFTMOE_FWD_LOCAL_DECODE_STAGES
#define WEFTMOE_FWD_LOCAL_DECODE_STAGES 3
#endif
#ifndef WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER
#define WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER 0
#endif
constexpr unsigned kBlockM = 128U, kM = 16U, kN = 128U, kK = 64U;
constexpr unsigned kStages = WEFTMOE_FWD_LOCAL_DECODE_STAGES;
constexpr unsigned kSelectedM = WEFTMOE_FWD_LOCAL_DECODE_M;
constexpr unsigned kCtaMultiplier = WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER;
static_assert(kSelectedM == 0U || kSelectedM == 16U || kSelectedM == 64U);
static_assert(kStages == 2U || kStages == 3U);
static_assert(kCtaMultiplier <= 3U);
constexpr unsigned kThreads = 256U, kMathThreads = 128U;
constexpr unsigned kMaxExperts = 256U;
constexpr unsigned kFc1Tiles = 2U * forward_tuning::kMainIntermediate / kN;
constexpr unsigned kFc2Tiles = forward_tuning::kMainHidden / kN;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr unsigned kWarpN = kN / (kMathThreads / 32U);
using WarpMma = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using GroupMma = hopper::BF16Mma<kN>;

// Selection uses collective capacity, never one source's actual batch size.
// One window covers even the all-to-one-expert case. Larger pools keep V4.
inline bool eligible(const ActivationTransportPoolConfig &config,
                     const static_forward::ComputeRingLayout &ring) {
  return forward_tuning::kStaticV4PrefillEnabled &&
         forward_tuning::kClusterSize == 2U &&
         forward_tuning::kTileM == kBlockM && forward_tuning::kTileK == kK &&
         forward_tuning::StaticV4Gemm<true>::kFc1N == kN &&
         forward_tuning::kFc2TileN == kN && config.max_nodes == 1U &&
         !config.enable_backward_state && config.max_tokens_per_rank <= 64U &&
         ring.depth && ring.max_windows == 1U && config.max_world_size <= 32U &&
         config.max_owned_experts * config.max_world_size <= kMaxExperts;
}

// Alias only the invocation-private block directory, not cumulative ring
// credits. All payload fits one generation. BeforeCombine retires source reads;
// the next BeforePull protects contribution reuse until all local combines end.
// There is no online admission or within-invocation slot overwrite.
struct BlockState {
  unsigned long long x_ready, s_ready;
};
static_assert(sizeof(BlockState) <= sizeof(static_forward::RingBlock));

template <unsigned M> struct Shared {
  // Source planning retires before geometry; reuse its scratch for local rows.
  union {
    unsigned counts[kMaxExperts];
    unsigned rows[kMaxExperts];
  };
  unsigned blocks[kMaxExperts + 1U];
  alignas(1024) transport::BFloat16 a[kStages][M * kK];
  alignas(1024) transport::BFloat16 b[kStages][kN * kK];
  alignas(1024) transport::BFloat16 output[M * kN];
  alignas(1024) unsigned char pull[3][4096];
  Barrier full[kStages], empty[kStages], pull_full[3];
};
static_assert(sizeof(Shared<16U>) <= 76U * 1024U);
static_assert(sizeof(Shared<64U>) <= 112U * 1024U);

__device__ __forceinline__ BlockState *states(const ForwardKernelArgs &a) {
  return reinterpret_cast<BlockState *>(a.ring.blocks);
}

struct Task {
  unsigned expert, block, column, rows, row;
  bool fc2;
};

// Producer and math independently replay this immutable grid-stride sequence.
// FC1 residues continue into FC2. TMA stages and credits are CTA-local: each
// full barrier publishes A/B; one math-group arrival releases both operands.
template <unsigned M>
__device__ __forceinline__ Task task(const Shared<M> &s, unsigned owned,
                                     unsigned index) {
  const auto fc1 = s.blocks[owned] * kFc1Tiles;
  const bool fc2 = index >= fc1;
  const auto linear = fc2 ? index - fc1 : index;
  const auto columns = fc2 ? kFc2Tiles : kFc1Tiles;
  const auto block = linear / columns;
  unsigned expert = 0U;
  while (s.blocks[expert + 1U] <= block)
    ++expert;
  return {expert,
          block,
          linear % columns,
          min(kBlockM, s.rows[expert] - (block - s.blocks[expert]) * kBlockM),
          block * kBlockM,
          fc2};
}

template <unsigned M>
__device__ __forceinline__ Task slice(Task t, unsigned offset) {
  t.rows = min(M, t.rows - offset);
  t.row += offset;
  return t;
}

} // namespace local_decode
