#pragma once

// Included inside forward_compute_detail, alongside the shared M128 fragment
// layout helpers. W1's packed [gate8, up8] accumulator becomes an N/2 SwiGLU
// tile; the BF16 projection rounding remains identical to the V3 contract.
__device__ __forceinline__ float2
static_v4_swiglu_pair(const float *gate, const float *up, float route_weight) {
  const auto g = __bfloat1622float2(
      __float22bfloat162_rn(make_float2(gate[0], gate[1])));
  const auto u = __bfloat1622float2(
      __float22bfloat162_rn(make_float2(up[0], up[1])));
  return make_float2(gated_swiglu_fc2_input(g.x, u.x, route_weight),
                    gated_swiglu_fc2_input(g.y, u.y, route_weight));
}

template <unsigned Fc1TileN>
__device__ __forceinline__ void static_v4_store_fc1_tile(
    const ForwardKernelArgs &a, transport::BFloat16 *tile, const float *accum,
    transport::BFloat16 *output, const float *gates, std::uint64_t row,
    std::uint32_t valid_rows, std::uint32_t column_base) {
  constexpr auto kOutputN = Fc1TileN / 2U;
  const auto lane = threadIdx.x % 32U;
  const auto row0 = (threadIdx.x / 32U) * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
  // Keep the two invariant gate loads outside the fragment-store loop.
  const float gate0 = row0 < valid_rows ? gates[row + row0] : 0.0F;
  const float gate1 = row1 < valid_rows ? gates[row + row1] : 0.0F;

  if (valid_rows == forward_tuning::kTileM) {
    // Both math warpgroups must retire their TMA B reads before staging
    // into that CTA-private arena. A is multicast and must not be reused.
    // N32 uses half of an N64 swizzle atom, not a different
    // fragment layout; its vector gather below skips the unused columns.
    cutlass::arch::NamedBarrier::sync(
        kMathThreads, physical_barrier<true>(kMathReadyBarrier));
#pragma unroll
    for (unsigned index = 0; index < kOutputN / 8U; ++index) {
      const auto *g = accum + index * 8U;
      const auto *u = g + 4U;
      store_m128_fragment(
          tile, index,
          __float22bfloat162_rn(static_v4_swiglu_pair(g, u, gate0)),
          __float22bfloat162_rn(static_v4_swiglu_pair(g + 2U, u + 2U, gate1)));
    }
    cutlass::arch::NamedBarrier::sync(
        kMathThreads, physical_barrier<true>(kMathDoneBarrier));
    for (unsigned vector = threadIdx.x;
         vector < forward_tuning::kTileM * kOutputN / 8U;
         vector += kMathThreads) {
      const auto m = vector / (kOutputN / 8U);
      const auto n = (vector % (kOutputN / 8U)) * 8U;
      *reinterpret_cast<uint4 *>(output + (row + m) * a.intermediate +
                                 column_base + n) =
          *reinterpret_cast<const uint4 *>(
              tile + m128_fragment_vector_offset(m, n));
    }
    return;
  }

  // Decode and compact expert tails avoid staging and its extra CTA barriers.
#pragma unroll
  for (unsigned index = 0; index < kOutputN / 8U; ++index) {
    const auto *g = accum + index * 8U;
    const auto *u = g + 4U;
    const auto column = column_base + index * 8U + (lane % 4U) * 2U;
    if (row0 < valid_rows) {
      const auto value = static_v4_swiglu_pair(g, u, gate0);
      *reinterpret_cast<std::uint32_t *>(
          output + (row + row0) * a.intermediate + column) =
          float2_to_bf16x2_bits(value.x, value.y);
    }
    if (row1 < valid_rows) {
      const auto value = static_v4_swiglu_pair(g + 2U, u + 2U, gate1);
      *reinterpret_cast<std::uint32_t *>(
          output + (row + row1) * a.intermediate + column) =
          float2_to_bf16x2_bits(value.x, value.y);
    }
  }
}
