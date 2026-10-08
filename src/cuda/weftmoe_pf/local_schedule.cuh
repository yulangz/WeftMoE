#pragma once

namespace weftmoe_pf::local {

constexpr unsigned kM = 128U, kN = forward_tuning::kFc1TileN, kK = 64U;
constexpr unsigned kThreads = 384U, kMathThreads = 256U;
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
  alignas(1024) unsigned char pull[3][4096];
  Barrier full[kStages], empty[kStages], pull_full[3];
};

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

__device__ __forceinline__ void arrive(std::uint64_t &word, unsigned count) {
  reduce_add_release_device(reinterpret_cast<unsigned long long *>(&word),
                            count);
}

} // namespace weftmoe_pf::local
