#pragma once

// Included inside forward_compute_detail after the fragment-layout and named
// barrier helpers.  Keeping this specialization out of the generic epilogue
// avoids carrying static peer addressing through the dynamic V3 epilogue.
template <bool Network = false, unsigned TileN = forward_tuning::kFc2TileN,
          bool RingSchedule = false, unsigned TileRows = 128U,
          unsigned MathThreads = kMathThreads>
__device__ __forceinline__ void
static_v4_copy_fc2_tile(const ForwardKernelArgs &a, transport::BFloat16 *tile,
                        std::uint64_t canonical_row, std::uint32_t valid_rows,
                        std::uint32_t column_base, std::uint64_t mblock,
                        unsigned row_offset = 0U) {
  cutlass::arch::NamedBarrier::sync(MathThreads,
                                    physical_barrier<true>(kMathReadyBarrier));

  // Neighboring lanes write neighboring vectors of one contribution row.
  // Assigning one lane per row scatters every warp across H-strided peer
  // addresses and turns each 16-byte store into its own NVLink transaction.
  constexpr auto kRowLanes = 8U;
  constexpr auto kElements = sizeof(uint4) / sizeof(transport::BFloat16);
  const auto column_lane = threadIdx.x % kRowLanes;
  for (auto row = threadIdx.x / kRowLanes; row < valid_rows;
       row += MathThreads / kRowLanes) {
    transport::BFloat16 *destination;
    if constexpr (RingSchedule) {
      const auto identity =
          a.ring.rows[mblock * forward_tuning::kTileM + row_offset + row];
      destination = static_cast<transport::BFloat16 *>(identity.output);
    } else {
      const auto &header = weftmoe_pf::route_header(a, canonical_row + row);
      if constexpr (Network)
        destination = weftmoe_pf::network_fc2_destination(a, header);
      else
        destination =
            peer_at<transport::BFloat16>(a.peer_bases[header.source_rank],
                                         a.contribution_offset) +
            std::uint64_t(header.contribution_row) * a.hidden;
    }
    destination += column_base;
    for (std::uint32_t column = column_lane * kElements; column < TileN;
         column += kRowLanes * kElements) {
      if (column_base + column >= a.hidden)
        break;
      const auto remaining = a.hidden - column_base - column;
      const auto elements = remaining < kElements ? remaining : kElements;
      const auto *source =
          tile + m_fragment_vector_offset<TileRows>(row, column);
      if (elements == kElements) {
        *reinterpret_cast<uint4 *>(destination + column) =
            *reinterpret_cast<const uint4 *>(source);
      } else {
        for (std::uint32_t element = 0; element < elements; ++element)
          destination[column + element] = source[element];
      }
    }
  }
  cutlass::arch::NamedBarrier::sync(MathThreads,
                                    physical_barrier<true>(kMathDoneBarrier));
}

template <class TileMma, bool Network = false,
          unsigned TileN = forward_tuning::kFc2TileN, bool RingSchedule = false,
          unsigned TileRows = 128U, unsigned MathThreads = kMathThreads>
__device__ __forceinline__ void
static_v4_push_fc2_tile(const ForwardKernelArgs &a, transport::BFloat16 *tile,
                        float *accum, std::uint64_t canonical_row,
                        std::uint32_t valid_rows, std::uint32_t column_base,
                        std::uint64_t mblock, unsigned row_offset = 0U) {
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    store_m_fragment<TileRows>(
        tile, index,
        __float22bfloat162_rn(
            make_float2(accum[index * 4U], accum[index * 4U + 1U])),
        __float22bfloat162_rn(
            make_float2(accum[index * 4U + 2U], accum[index * 4U + 3U])));
  }
  static_v4_copy_fc2_tile<Network, TileN, RingSchedule, TileRows, MathThreads>(
      a, tile, canonical_row, valid_rows, column_base, mblock, row_offset);
}
