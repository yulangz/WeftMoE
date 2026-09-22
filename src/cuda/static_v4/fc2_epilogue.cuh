#pragma once

// Included inside forward_compute_detail after the fragment-layout and named
// barrier helpers.  Keeping this specialization out of the generic epilogue
// avoids carrying peer addressing through the training and multi-node paths.
template <class TileMma>
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

  const auto row = threadIdx.x & 127U;
  if (row < valid_rows) {
    const auto &header = static_v4::route_header(a, canonical_row + row);
    auto *destination = peer_at<transport::BFloat16>(
        a.peer_bases[header.source_rank], a.contribution_offset);
    destination += header.contribution_row * a.hidden + column_base;
    const auto half = threadIdx.x >> 7U;
    constexpr auto kElements = sizeof(uint4) / sizeof(transport::BFloat16);
    const auto begin = half * (forward_tuning::kFc2TileN / 2U);
    const auto end = begin + forward_tuning::kFc2TileN / 2U;
    for (std::uint32_t column = begin; column < end; column += kElements) {
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
