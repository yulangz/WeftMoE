#pragma once

namespace weftmoe_pf::network {

constexpr unsigned kThreads = 384U;
using Resources =
    unified_worker_tuning::Resources<384, 96, 288, 208, 88, 88, 4096, 1>;
using Mma = hopper::BF16Mma<local::kN>;

// Static residues of full M/N tiles, paired for TMA multicast. Windows order
// blocks by their last required arrival, never pad an expert at each boundary.
struct Schedule {
  const ForwardKernelArgs &a;
  unsigned window = 0U, expert = 0U, index = blockIdx.x;

  __device__ explicit Schedule(const ForwardKernelArgs &args) : a(args) {}

  __device__ __forceinline__ bool next(local::Task &t) {
    const auto owned = a.dispatch.sizes.owned_expert_count;
    while (window < a.ring.state->windows) {
      const auto *prefix = a.ring.expert_blocks + window * (owned + 1U);
      const auto begin = unsigned(prefix[expert]);
      const auto blocks = unsigned(prefix[expert + 1U]) - begin;
      const auto fc1 = blocks * local::kFc1Tiles;
      const auto total = fc1 + blocks * local::kFc2Tiles;
      if (index < total) {
        const bool fc2 = index >= fc1;
        const auto linear = fc2 ? index - fc1 : index;
        const auto columns = fc2 ? local::kFc2Tiles : local::kFc1Tiles;
        constexpr unsigned Group = 16U;
        const auto group = linear / (blocks * Group);
        const auto width = min(Group, columns - group * Group);
        const auto within = linear - group * blocks * Group;
        const auto block =
            unsigned(a.ring.window_prefix[window]) + begin + within / width;
        const auto &desc = a.ring.blocks[block];
        t = {expert,
             block,
             group * Group + within % width,
             desc.tile.rows,
             desc.slot * local::kM,
             unsigned(desc.generation),
             fc2};
        index += gridDim.x;
        return true;
      }
      index -= total;
      if (++expert == owned) {
        expert = 0U;
        ++window;
      }
    }
    return false;
  }
};

} // namespace weftmoe_pf::network
