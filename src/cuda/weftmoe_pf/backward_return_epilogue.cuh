#pragma once

namespace weftmoe_pf::backward_return {

// Local rows go straight from the WGMMA epilogue to a credited source
// window. Remote rows keep the bounded dX slot until the NIC sender packs
// them. The Return owner publishes dGate and readiness after the cluster join.
struct Output {
  const BackwardKernelArgs &a;
  unsigned block;

  template <unsigned TileM, unsigned TileN>
  __device__ __forceinline__ bool store(transport::BFloat16 *tile,
                                       unsigned output_row,
                                       unsigned column_base) const {
    if (!a.return_ring.depth)
      return false;
    constexpr auto threads = TileM == 64U ? 128U : 256U;
    constexpr auto row_lanes = 8U;
    const auto valid = a.return_ring.blocks[block].rows;
    const auto first = std::uint64_t(block) * backward_tuning::kTileM;
    const auto row_offset = output_row % backward_tuning::kTileM;
    for (unsigned row = threadIdx.x / row_lanes;
         row < TileM && row + row_offset < valid; row += threads / row_lanes) {
      const auto identity = a.return_ring.rows[first + row_offset + row];
      auto *destination = identity.source / a.local_world_size == a.node
          ? a.peer_payloads[identity.source].contribution +
                contribution_row(a, identity.route) * a.hidden
          : a.dx_rows + (std::uint64_t(output_row) + row) * a.hidden;
      for (unsigned column = (threadIdx.x % row_lanes) * 8U;
           column < TileN; column += row_lanes * 8U) {
        const auto offset = column / 64U * TileM * 64U + row * 64U +
                            ((column % 64U) ^ ((row % 8U) * 8U));
        *reinterpret_cast<uint4 *>(destination + column_base + column) =
            *reinterpret_cast<const uint4 *>(tile + offset);
      }
    }
    return true;
  }
};

} // namespace weftmoe_pf::backward_return
