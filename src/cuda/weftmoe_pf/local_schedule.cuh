#pragma once

#ifndef WEFTMOE_FP8_FWD_WAVE_BLOCKS
#define WEFTMOE_FP8_FWD_WAVE_BLOCKS 0
#endif

namespace weftmoe_pf::local {

constexpr unsigned kM = 128U, kN = forward_tuning::kFc1TileN, kK = 64U;
constexpr unsigned kThreads = 384U, kMathThreads = 256U;
constexpr unsigned kPullBufferBytes = 4096U;
constexpr bool kFP8NetworkRowScales = forward_tuning::kMainHidden < 4096U &&
    forward_tuning::kMainHidden % 512U == 0U;
constexpr unsigned kStages = forward_tuning::kStaticV4Fc1Stages;
constexpr unsigned kFc1Tiles = 2U * forward_tuning::kMainIntermediate / kN;
constexpr unsigned kFc2Tiles = forward_tuning::kMainHidden / kN;
using Resources =
    unified_worker_tuning::Resources<384, 96, 288, 208, 80, 80, 4096, 1>;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using Mma = hopper::BF16Mma<kN>;

// Input stages never alias the epilogue. The producer can prefetch the next
// task while math writes the previous task's SwiGLU or remote contribution.
struct Shared {
  alignas(1024) transport::BFloat16 a[kStages][kM * kK];
  alignas(1024) transport::BFloat16 b[kStages][kN * kK];
  alignas(1024) transport::BFloat16 output[kM * kN];
  alignas(1024) unsigned char pull[3][kPullBufferBytes];
  Barrier full[kStages], empty[kStages], pull_full[3];
};

// Block weights need two scales per N128 for interleaved gate/up, rather
// than a sparse N-element scale plane.
inline constexpr unsigned kFP8ScaleElements = kM + 2U * (kN / 128U);
inline constexpr std::size_t kFP8SharedBytes = sizeof(Shared) + kStages * kFP8ScaleElements * sizeof(float);
__device__ __forceinline__ float *fp8_scales(Shared &s, unsigned stage) {
  return reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(&s) + sizeof(Shared)) +
      stage * kFP8ScaleElements;
}

struct Task {
  unsigned expert, block, column, rows, row, generation;
  bool fc2;
};

// Every CTA owns one residue of the same immutable task stream. Expert waves
// larger than the payload ring are split at full M-block boundaries, never
// at source-token windows. Producer and math replay this stream independently.
struct Schedule {
  const ForwardKernelArgs &a;
  unsigned expert = 0U, begin = 0U, end = 0U, index;
  unsigned cached_expert = ~0U, expert_begin = 0U, expert_end = 0U, rows = 0U;

  __device__ explicit Schedule(const ForwardKernelArgs &args)
      : a(args), index(blockIdx.x) {}

  __device__ __forceinline__ bool next(Task &t) {
    const auto owned = a.dispatch.sizes.owned_expert_count;
    while (expert < owned) {
      if (cached_expert != expert) {
        // Geometry is immutable within this invocation. Keep the current
        // expert in warp-uniform registers across all its FC1/FC2 tiles.
        expert_begin = __shfl_sync(
            0xffffffffU, unsigned(a.ring.expert_blocks[expert]), 0);
        expert_end = __shfl_sync(
            0xffffffffU, unsigned(a.ring.expert_blocks[expert + 1U]), 0);
        rows = __shfl_sync(0xffffffffU, unsigned(a.ring.expert_rows[expert]), 0);
        cached_expert = expert;
      }
      if (end == begin)
        end = min(expert_end, begin + a.ring.depth);
      const auto blocks = end - begin;
      const auto fc1 = blocks * kFc1Tiles;
      const auto total = fc1 + blocks * kFc2Tiles;
      if (index < total) {
        const bool fc2 = index >= fc1;
        const auto linear = fc2 ? index - fc1 : index;
        const auto columns = fc2 ? kFc2Tiles : kFc1Tiles;
        constexpr unsigned Group = 16U;
        const auto group = linear / (blocks * Group);
        const auto width = min(Group, columns - group * Group);
        const auto within = linear - group * blocks * Group;
        const auto block = begin + within / width;
        t = {expert,
             block,
             group * Group + within % width,
             min(kM, rows - (block - expert_begin) * kM),
             block % a.ring.depth * kM,
             block / a.ring.depth + 1U,
             fc2};
        index += gridDim.x;
        return true;
      }
      index -= total;
      begin = end;
      if (end == expert_end)
        ++expert;
    }
    return false;
  }
};

// Schedule FC1 across an entire bounded ring wave before FC2 so resident
// CTAs can process different experts while waiting for FC1 dependencies.
// Producer and math must replay this same stream, including cluster pairs.
struct FP8Schedule {
  const ForwardKernelArgs &a;
  unsigned begin = 0, index = blockIdx.x;
  __device__ explicit FP8Schedule(const ForwardKernelArgs &args) : a(args) {}
  __device__ __forceinline__ bool next(Task &t) {
    const auto total_blocks = unsigned(a.ring.state->blocks);
    while (begin < total_blocks) {
      // Small experts benefit from shorter peer-input dependencies. Larger
      // experts need a wider wave to keep enough independent GEMMs in flight.
      const auto wave = WEFTMOE_FP8_FWD_WAVE_BLOCKS
          ? unsigned(WEFTMOE_FP8_FWD_WAVE_BLOCKS)
          : max(1U, (kFc1Tiles <= 4U ? 128U : 256U) / kFc1Tiles);
      const auto capacity = min(a.ring.depth, wave);
      const auto blocks = min(capacity, total_blocks - begin);
      const auto first = blocks * kFc1Tiles;
      const auto total = first + blocks * kFc2Tiles;
      if (index >= total) {
        index -= total;
        begin += blocks;
        continue;
      }
      const bool fc2 = index >= first;
      const auto linear = fc2 ? index - first : index;
      const auto columns = fc2 ? kFc2Tiles : kFc1Tiles;
      const auto block = begin + linear / columns;
      unsigned lo = 0, hi = a.dispatch.sizes.owned_expert_count;
      while (lo + 1U < hi) {
        const auto middle = (lo + hi) / 2U;
        if (a.ring.expert_blocks[middle] <= block)
          lo = middle;
        else
          hi = middle;
      }
      const auto rows = unsigned(a.ring.expert_rows[lo]);
      t = {lo, block, linear % columns,
           min(kM, rows - (block - unsigned(a.ring.expert_blocks[lo])) * kM),
           block % a.ring.depth * kM, block / a.ring.depth + 1U, fc2};
      index += gridDim.x;
      return true;
    }
    return false;
  }
};

__device__ __forceinline__ void arrive(std::uint64_t &word, unsigned count) {
  reduce_add_release_device(reinterpret_cast<unsigned long long *>(&word),
                            count);
}

} // namespace weftmoe_pf::local
