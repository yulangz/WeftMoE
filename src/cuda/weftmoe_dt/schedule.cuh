#pragma once

namespace weftmoe_dt {

// Keep the shared M128 GM directory; all decode arithmetic uses M64
// slices. Hot experts traverse every slice, including multi-block experts.
#ifndef WEFTMOE_FWD_LOCAL_DECODE_STAGES
#define WEFTMOE_FWD_LOCAL_DECODE_STAGES 3
#endif
#ifndef WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER
#define WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER 0
#endif
#ifndef WEFTMOE_DT_SLICE_SCHEDULE
#define WEFTMOE_DT_SLICE_SCHEDULE 0
#endif
// Offline DT FP8 tuning overrides; defaults preserve the production policy.
// BF16 uses its existing configuration throughout comparative benchmarks.
#ifndef WEFTMOE_DT_FP8_STAGES
#define WEFTMOE_DT_FP8_STAGES 3
#endif
#ifndef WEFTMOE_DT_FP8_SLICE_SCHEDULE
#define WEFTMOE_DT_FP8_SLICE_SCHEDULE 1
#endif
#ifndef WEFTMOE_DT_FP8_CTAS
#define WEFTMOE_DT_FP8_CTAS 0
#endif
constexpr unsigned kFP8Stages = WEFTMOE_DT_FP8_STAGES;
constexpr bool kFP8SliceSchedule = WEFTMOE_DT_FP8_SLICE_SCHEDULE;
constexpr unsigned kFP8Ctas = WEFTMOE_DT_FP8_CTAS;
static_assert(kFP8Stages >= 1U && kFP8Stages <= 3U);
static_assert(WEFTMOE_DT_FP8_SLICE_SCHEDULE == 0 ||
              WEFTMOE_DT_FP8_SLICE_SCHEDULE == 1);
static_assert(kFP8Ctas % 2U == 0U);
// Slice scheduling changes only the deterministic compute raster.
// Row storage and the readiness directory remain M128.
constexpr bool kBF16SliceSchedule = WEFTMOE_DT_SLICE_SCHEDULE;
static_assert(WEFTMOE_DT_SLICE_SCHEDULE == 0 || WEFTMOE_DT_SLICE_SCHEDULE == 1);
template <bool FP8 = false>
inline constexpr bool kSliceSchedule =
    FP8 ? kFP8SliceSchedule : kBF16SliceSchedule;

constexpr unsigned kBlockM = 128U, kM = 64U, kN = 128U, kK = 64U;
constexpr unsigned kStages = WEFTMOE_FWD_LOCAL_DECODE_STAGES;
constexpr unsigned kCtaMultiplier = WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER;
static_assert(kStages == 2U || kStages == 3U);
static_assert(kCtaMultiplier <= 3U);
constexpr unsigned kMathThreads = 128U;
template <bool FP8>
inline constexpr unsigned kComputeThreads = FP8 ? 256U : 128U;
template <bool FP8>
inline constexpr unsigned kKernelThreads = kComputeThreads<FP8> + 128U;
constexpr unsigned kFc1Tiles = 2U * forward_tuning::kMainIntermediate / kN;
constexpr unsigned kFc2Tiles = forward_tuning::kMainHidden / kN;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using GroupMma = hopper::BF16Mma<kN>;

// DT is selected only by an explicit, collectively agreed decode pool.
// Its complete capacity must fit one window, including all-to-one routing;
// neither a small actual batch nor a PF profile implicitly selects DT.
inline bool eligible(const ActivationTransportPoolConfig &config,
                     const static_forward::ComputeRingLayout &ring) {
  return config.decode_only && forward_tuning::kClusterSize == 2U &&
         forward_tuning::kTileM == kBlockM && forward_tuning::kTileK == kK &&
         config.max_hidden_size == forward_tuning::kMainHidden &&
         config.max_intermediate_size == forward_tuning::kMainIntermediate &&
         config.max_nodes == 1U && !config.enable_backward_state &&
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

template <bool FP8>
inline constexpr unsigned stages = FP8 ? kFP8Stages : kStages;

template <bool FP8 = false> struct Shared {
  using Operand = std::conditional_t<FP8, std::uint8_t, transport::BFloat16>;
  alignas(1024) Operand a[stages<FP8>][kM * (FP8 ? 128U : kK)];
  alignas(1024) Operand b[stages<FP8>][(FP8 ? 256U : kN) * (FP8 ? 128U : kK)];
  alignas(1024) transport::BFloat16 output[kM * (FP8 ? 256U : kN)];
  alignas(1024) unsigned char pull[3][4096];
  Barrier full[stages<FP8>], empty[stages<FP8>], pull_full[3];

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
static_assert(sizeof(Shared<>) <= 112U * 1024U);

template <bool FP8 = false>
std::size_t shared_bytes(unsigned experts, unsigned owned) {
  return sizeof(Shared<FP8>) +
         sizeof(unsigned) *
             std::max<std::size_t>(experts, kSliceSchedule<FP8>
                                                ? 3ULL * owned + 2U
                                                : 2ULL * owned + 1U);
}

__device__ __forceinline__ BlockState *states(const ForwardKernelArgs &a) {
  return reinterpret_cast<BlockState *>(a.ring.blocks);
}

struct Task {
  unsigned expert, block, column, rows, row;
  bool fc2;
};

template <bool FP8>
__device__ __forceinline__ unsigned compute_units(Shared<FP8> &s,
                                                  unsigned owned) {
  if constexpr (kSliceSchedule<FP8>)
    return s.slices(owned)[owned];
  return s.blocks(owned)[owned];
}

template <bool FP8>
__device__ __forceinline__ unsigned ready_rows(Shared<FP8> &s, unsigned owned,
                                               const Task &t) {
  if constexpr (kSliceSchedule<FP8>)
    return min(kBlockM, s.rows()[t.expert] -
                            (t.block - s.blocks(owned)[t.expert]) * kBlockM);
  return t.rows;
}

// Producer and math independently replay this immutable grid-stride sequence.
// FC1 residues continue into FC2. TMA stages and credits are CTA-local: each
// full barrier publishes A/B; each math group arrives to release the operands.
template <bool FP8>
__device__ __forceinline__ Task task(Shared<FP8> &s, unsigned owned,
                                     unsigned index) {
  const auto *blocks = s.blocks(owned);
  constexpr unsigned Fc1Tiles = FP8 ? kFc1Tiles / 2U : kFc1Tiles;
  const auto fc1 = compute_units(s, owned) * Fc1Tiles;
  const bool fc2 = index >= fc1;
  const auto linear = fc2 ? index - fc1 : index;
  const auto columns = fc2 ? (FP8 ? kFc2Tiles / 2U : kFc2Tiles) : Fc1Tiles;
  const auto block = linear / columns;
  if constexpr (kSliceSchedule<FP8>) {
    const auto *slices = s.slices(owned);
    unsigned expert = 0U;
    while (slices[expert + 1U] <= block)
      ++expert;
    const auto offset = (block - slices[expert]) * kM;
    return {expert,
            blocks[expert] + offset / kBlockM,
            linear % columns,
            min(kM, s.rows()[expert] - offset),
            blocks[expert] * kBlockM + offset,
            fc2};
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

__device__ __forceinline__ Task slice(Task t, unsigned offset) {
  t.rows = min(kM, t.rows - offset);
  t.row += offset;
  return t;
}

} // namespace weftmoe_dt
