#pragma once

// Included inside forward_compute_detail, alongside the shared M128 fragment
// layout helpers. W1's packed [gate8, up8] accumulator becomes an N/2 SwiGLU
// tile; the BF16 projection rounding remains identical to the V3 contract.
template <bool FastMath = false>
__device__ __forceinline__ float2 static_v4_swiglu_pair(const float *gate,
                                                        const float *up,
                                                        float route_weight) {
  const auto g =
      __bfloat1622float2(__float22bfloat162_rn(make_float2(gate[0], gate[1])));
  const auto u =
      __bfloat1622float2(__float22bfloat162_rn(make_float2(up[0], up[1])));
  return make_float2(gated_swiglu_fc2_input<FastMath>(g.x, u.x, route_weight),
                     gated_swiglu_fc2_input<FastMath>(g.y, u.y, route_weight));
}

// Transpose four BF16x2 words across four lanes. Each lane then owns eight
// consecutive columns, without staging the WGMMA fragment through shared.
__device__ __forceinline__ uint4 static_v4_transpose_vector(uint4 v) {
  const auto lane = threadIdx.x % 4U;
  const auto mask = 0xfU << (threadIdx.x % 32U / 4U * 4U);
  auto t = __shfl_xor_sync(mask, (lane & 1U) ? v.x : v.y, 1, 4);
  v.x = (lane & 1U) ? t : v.x;
  v.y = (lane & 1U) ? v.y : t;
  t = __shfl_xor_sync(mask, (lane & 1U) ? v.z : v.w, 1, 4);
  v.z = (lane & 1U) ? t : v.z;
  v.w = (lane & 1U) ? v.w : t;
  t = __shfl_xor_sync(mask, (lane & 2U) ? v.x : v.z, 2, 4);
  v.x = (lane & 2U) ? t : v.x;
  v.z = (lane & 2U) ? v.z : t;
  t = __shfl_xor_sync(mask, (lane & 2U) ? v.y : v.w, 2, 4);
  v.y = (lane & 2U) ? t : v.y;
  v.w = (lane & 2U) ? v.w : t;
  return v;
}

// Checkpoint gate/up directly from the same WGMMA accumulators used by
// SwiGLU. Each four-lane group owns one row, including a partial M tail.
template <unsigned Fc1TileN, class Args>
__device__ __forceinline__ void
static_v4_save_fc1_tile(const Args &a, const float *accum,
                        std::uint32_t valid_rows, std::uint32_t column_base,
                        std::uint64_t mblock) {
  const auto lane = threadIdx.x % 32U;
  const auto row0 = (threadIdx.x / 32U) * 16U + lane / 4U;
  for (unsigned item = 0U; item < 2U; ++item) {
    const auto m = row0 + item * 8U;
    if (m >= valid_rows)
      continue;
    const auto saved = a.checkpoint.rows[mblock * forward_tuning::kTileM + m];
#pragma unroll
    for (unsigned index = 0U; index < Fc1TileN / 16U; index += 4U) {
      unsigned gate_words[4], up_words[4];
#pragma unroll
      for (unsigned j = 0U; j < 4U; ++j) {
        const auto *g = accum + (index + j) * 8U + item * 2U;
        const auto *u = g + 4U;
        gate_words[j] = float2_to_bf16x2_bits(g[0], g[1]);
        up_words[j] = float2_to_bf16x2_bits(u[0], u[1]);
      }
      const auto column = column_base + index * 8U + (lane % 4U) * 8U;
      auto *gate = a.saved_ab + saved * a.intermediate + column;
      auto *up = gate + a.saved_rows * a.intermediate;
      *reinterpret_cast<uint4 *>(gate) = static_v4_transpose_vector(
          make_uint4(gate_words[0], gate_words[1], gate_words[2], gate_words[3]));
      *reinterpret_cast<uint4 *>(up) = static_v4_transpose_vector(
          make_uint4(up_words[0], up_words[1], up_words[2], up_words[3]));
    }
  }
}

template <unsigned Fc1TileN, bool FastMath = false, class Args>
__device__ __forceinline__ void static_v4_store_fc1_register_vector_tile(
    const Args &a, const float *accum, transport::BFloat16 *output,
    const float *gates, std::uint64_t row, std::uint32_t valid_rows,
    std::uint32_t column_base) {
  static_assert(Fc1TileN % 64U == 0U);
  const auto lane = threadIdx.x % 32U;
  const auto row0 = (threadIdx.x / 32U) * 16U + lane / 4U;
#pragma unroll
  for (unsigned item = 0; item < 2U; ++item) {
    const auto m = row0 + item * 8U;
    if (m >= valid_rows)
      continue;
    const auto gate = gates[row + m];
#pragma unroll
    for (unsigned index = 0; index < Fc1TileN / 16U; index += 4U) {
      std::uint32_t words[4];
#pragma unroll
      for (unsigned j = 0; j < 4U; ++j) {
        const auto *g = accum + (index + j) * 8U + item * 2U;
        const auto value = static_v4_swiglu_pair<FastMath>(g, g + 4U, gate);
        words[j] = float2_to_bf16x2_bits(value.x, value.y);
      }
      const auto vector = static_v4_transpose_vector(
          make_uint4(words[0], words[1], words[2], words[3]));
      const auto column = column_base + index * 8U + (lane % 4U) * 8U;
      *reinterpret_cast<uint4 *>(output + (row + m) * a.intermediate + column) =
          vector;
    }
  }
}
