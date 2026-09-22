#pragma once

// Included inside forward_compute_detail after the fragment-layout and named
// barrier helpers.  Keeping this specialization out of the generic epilogue
// avoids carrying inference-only peer addressing through the training path.
template <class TileMma, bool Network = false>
__device__ __forceinline__ void static_v4_push_fc2_tile(
    const ForwardKernelArgs &a, transport::BFloat16 *tile, float *accum,
    std::uint64_t canonical_row, std::uint32_t valid_rows,
    std::uint32_t column_base) {
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    store_m128_fragment(
        tile, index,
        __float22bfloat162_rn(
            make_float2(accum[index * 4U], accum[index * 4U + 1U])),
        __float22bfloat162_rn(make_float2(accum[index * 4U + 2U],
                                          accum[index * 4U + 3U])));
  }
  cutlass::arch::NamedBarrier::sync(
      kMathThreads, physical_barrier<true>(kMathReadyBarrier));

  // Neighboring lanes write neighboring vectors of one contribution row.
  // Assigning one lane per row scatters every warp across H-strided peer
  // addresses and turns each 16-byte store into its own NVLink transaction.
  constexpr auto kRowLanes = 8U;
  constexpr auto kElements = sizeof(uint4) / sizeof(transport::BFloat16);
  const auto column_lane = threadIdx.x % kRowLanes;
  for (auto row = threadIdx.x / kRowLanes; row < valid_rows;
       row += kMathThreads / kRowLanes) {
    const auto &header = static_v4::route_header(a, canonical_row + row);
    transport::BFloat16 *destination;
    if constexpr (Network)
      destination = static_v4::network_fc2_destination(a, header);
    else
      destination = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.contribution_offset) +
          std::uint64_t(header.contribution_row) * a.hidden;
    destination += column_base;
    for (std::uint32_t column = column_lane * kElements;
         column < forward_tuning::kFc2TileN;
         column += kRowLanes * kElements) {
      if (column_base + column >= a.hidden)
        break;
      const auto remaining = a.hidden - column_base - column;
      const auto elements = remaining < kElements ? remaining : kElements;
      const auto *source = tile + m128_fragment_vector_offset(row, column);
      if (elements == kElements) {
        *reinterpret_cast<uint4 *>(destination + column) =
            *reinterpret_cast<const uint4 *>(source);
      } else {
        for (std::uint32_t element = 0; element < elements; ++element)
          destination[column + element] = source[element];
      }
    }
  }
  cutlass::arch::NamedBarrier::sync(
      kMathThreads, physical_barrier<true>(kMathDoneBarrier));
}
