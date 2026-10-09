#pragma once

namespace weftmoe_pf::network {

constexpr unsigned kThreads = kNetworkWorkerMode == 2U ? 512U : 384U;
using Resources = unified_worker_tuning::Resources<
    kThreads, 96, 288, 208, kThreads == 512U ? 48U : 80U,
    kThreads == 512U ? 48U : 80U, 4096, 1>;
// CUDA 12.9 rejects native N256 at a 512-thread entry's static REG128.
// Split only this candidate's instruction; preserve logical N and K order.
using Mma = hopper::BF16Mma<local::kN, hopper::Major::K, hopper::Major::K,
    local::kK, 128U, kThreads == 512U ? 128 : WEFTMOE_WGMMA_N256_ATOM_N>;

// Static residues of full M/N tiles, paired for TMA multicast. A source window
// bounds RX/Return lifetimes; it does not split GEMM at network packet edges.
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
        const auto block = unsigned(a.ring.window_prefix[window]) + begin +
                           within / width;
        const auto &desc = a.ring.blocks[block];
        t = {expert, block, group * Group + within % width, desc.tile.rows,
             desc.slot * local::kM, unsigned(desc.generation), fc2};
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
