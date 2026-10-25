#pragma once

namespace weftmoe_dt {

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
#ifndef WEFTMOE_DT_SLICE_SCHEDULE
#define WEFTMOE_DT_SLICE_SCHEDULE 0
#endif
#ifndef WEFTMOE_FWD_LOCAL_DECODE_M64_MIN_ROWS
#define WEFTMOE_FWD_LOCAL_DECODE_M64_MIN_ROWS 16
#endif
#ifndef WEFTMOE_FWD_LOCAL_DECODE_NARROW_CTA_MULTIPLIER
#define WEFTMOE_FWD_LOCAL_DECODE_NARROW_CTA_MULTIPLIER 0
#endif
// Bit 0 enables M16 slices, bit 1 enables M64 slices. The row storage and
// readiness directory stay M128; only the deterministic compute raster changes.
constexpr unsigned kSliceScheduleMask = WEFTMOE_DT_SLICE_SCHEDULE;
static_assert(kSliceScheduleMask <= 3U);
template <unsigned M>
inline constexpr bool kSliceSchedule =
    (kSliceScheduleMask & (M == 16U ? 1U : 2U)) != 0U;

// Keep production selection unchanged; diagnostic builds can compare larger
// batches provided the complete invocation still fits a single source window.
#ifndef WEFTMOE_DT_MAX_TOKENS
#define WEFTMOE_DT_MAX_TOKENS 64
#endif
constexpr unsigned kBlockM = 128U, kM = 16U, kN = 128U, kK = 64U;
constexpr unsigned kStages = WEFTMOE_FWD_LOCAL_DECODE_STAGES;
constexpr unsigned kSelectedM = WEFTMOE_FWD_LOCAL_DECODE_M;
constexpr unsigned kCtaMultiplier = WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER;
constexpr unsigned kM64MinRows = WEFTMOE_FWD_LOCAL_DECODE_M64_MIN_ROWS;
constexpr unsigned kNarrowCtaMultiplier =
    WEFTMOE_FWD_LOCAL_DECODE_NARROW_CTA_MULTIPLIER;
static_assert(kM64MinRows > 0U);
static_assert(kNarrowCtaMultiplier <= 3U);
static_assert(kSelectedM == 0U || kSelectedM == 16U || kSelectedM == 64U);
static_assert(kStages == 2U || kStages == 3U);
static_assert(kCtaMultiplier <= 3U);
constexpr unsigned kThreads = 256U, kMathThreads = 128U;
constexpr unsigned kFc1Tiles = 2U * forward_tuning::kMainIntermediate / kN;
constexpr unsigned kFc2Tiles = forward_tuning::kMainHidden / kN;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr unsigned kWarpN = kN / (kMathThreads / 32U);
using WarpMma = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using GroupMma = hopper::BF16Mma<kN>;

// Selection uses collective capacity, never one source's actual batch size.
// One window covers even the all-to-one-expert case. Explicit decode pools
// cover the server's decode capacity; legacy DT profiles retain their limit.
inline bool eligible(const ActivationTransportPoolConfig &config,
                     const static_forward::ComputeRingLayout &ring) {
  return (config.decode_only || forward_tuning::kStaticV4PrefillEnabled) &&
         forward_tuning::kClusterSize == 2U &&
         forward_tuning::kTileM == kBlockM && forward_tuning::kTileK == kK &&
         config.max_hidden_size == forward_tuning::kMainHidden &&
         config.max_intermediate_size == forward_tuning::kMainIntermediate &&
         config.max_nodes == 1U && !config.enable_backward_state &&
         (config.decode_only ||
          config.max_tokens_per_rank <= WEFTMOE_DT_MAX_TOKENS) &&
         ring.depth && ring.max_windows == 1U && config.max_world_size <= 32U;
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
  alignas(1024) transport::BFloat16 a[kStages][M * kK];
  alignas(1024) transport::BFloat16 b[kStages][kN * kK];
  alignas(1024) transport::BFloat16 output[M * kN];
  alignas(1024) unsigned char pull[3][4096];
  Barrier full[kStages], empty[kStages], pull_full[3];

  // The variable-size tail first holds one source count per global expert.
  // After the source-ready barrier it holds local row counts, physical block
  // offsets, and (when enabled) compute-slice offsets. Slices share the same
  // physical input directory and completion counters as the M128 schedule.
  __device__ __forceinline__ unsigned *rows() {
    return reinterpret_cast<unsigned *>(this + 1);
  }
  __device__ __forceinline__ unsigned *blocks(unsigned owned) {
    return rows() + owned;
  }
  __device__ __forceinline__ unsigned *slices(unsigned owned) {
    return rows() + 2U * owned + 1U;
  }
};
static_assert(sizeof(Shared<16U>) <= 76U * 1024U);
static_assert(sizeof(Shared<64U>) <= 112U * 1024U);

template <unsigned M>
std::size_t shared_bytes(unsigned experts, unsigned owned) {
  return sizeof(Shared<M>) +
         sizeof(unsigned) * std::max<std::size_t>(
             experts, kSliceSchedule<M> ? 3ULL * owned + 2U
                                       : 2ULL * owned + 1U);
}

__device__ __forceinline__ BlockState *states(const ForwardKernelArgs &a) {
  return reinterpret_cast<BlockState *>(a.ring.blocks);
}

struct Task {
  unsigned expert, block, column, rows, row;
  bool fc2;
};

template <unsigned M>
__device__ __forceinline__ unsigned compute_units(Shared<M> &s,
                                                 unsigned owned) {
  if constexpr (kSliceSchedule<M>)
    return s.slices(owned)[owned];
  return s.blocks(owned)[owned];
}

template <unsigned M>
__device__ __forceinline__ unsigned ready_rows(Shared<M> &s, unsigned owned,
                                              const Task &t) {
  if constexpr (kSliceSchedule<M>)
    return min(kBlockM, s.rows()[t.expert] -
                           (t.block - s.blocks(owned)[t.expert]) * kBlockM);
  return t.rows;
}

// Producer and math independently replay this immutable grid-stride sequence.
// FC1 residues continue into FC2. TMA stages and credits are CTA-local: each
// full barrier publishes A/B; one math-group arrival releases both operands.
template <unsigned M>
__device__ __forceinline__ Task task(Shared<M> &s, unsigned owned,
                                     unsigned index) {
  const auto *blocks = s.blocks(owned);
  const auto fc1 = compute_units(s, owned) * kFc1Tiles;
  const bool fc2 = index >= fc1;
  const auto linear = fc2 ? index - fc1 : index;
  const auto columns = fc2 ? kFc2Tiles : kFc1Tiles;
  const auto block = linear / columns;
  if constexpr (kSliceSchedule<M>) {
    const auto *slices = s.slices(owned);
    unsigned expert = 0U;
    while (slices[expert + 1U] <= block)
      ++expert;
    const auto offset = (block - slices[expert]) * M;
    return {expert, blocks[expert] + offset / kBlockM, linear % columns,
            min(M, s.rows()[expert] - offset),
            blocks[expert] * kBlockM + offset, fc2};
  }
  unsigned expert = 0U;
  while (blocks[expert + 1U] <= block)
    ++expert;
  return {expert,
          block,
          linear % columns,
          min(kBlockM, s.rows()[expert] - (block - blocks[expert]) * kBlockM),
          block * kBlockM,
          fc2};
}

template <unsigned M>
__device__ __forceinline__ Task slice(Task t, unsigned offset) {
  t.rows = min(M, t.rows - offset);
  t.row += offset;
  return t;
}

} // namespace weftmoe_dt
