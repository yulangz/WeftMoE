#pragma once

// Included inside forward_compute_detail, alongside the shared M128 fragment
// layout helpers. W1's packed [gate8, up8] accumulator becomes an N/2 SwiGLU
// tile; the BF16 projection rounding remains identical to the V3 contract.
template <bool FastMath = false>
__device__ __forceinline__ float2
static_v4_swiglu_pair(const float *gate, const float *up, float route_weight) {
  const auto g = __bfloat1622float2(
      __float22bfloat162_rn(make_float2(gate[0], gate[1])));
  const auto u = __bfloat1622float2(
      __float22bfloat162_rn(make_float2(up[0], up[1])));
  return make_float2(gated_swiglu_fc2_input<FastMath>(g.x, u.x, route_weight),
                    gated_swiglu_fc2_input<FastMath>(g.y, u.y, route_weight));
}

// Save directly from WGMMA registers when the shared epilogue arena cannot
// hold three planes. In particular, local PF keeps its S-only TMA fast path.
template <unsigned Fc1TileN, class Args>
__device__ __forceinline__ void static_v4_save_fc1_tile(
    const Args &a, const float *accum, std::uint32_t valid_rows,
    std::uint32_t column_base, std::uint64_t mblock) {
  const auto lane = threadIdx.x % 32U;
  const auto row0 = (threadIdx.x / 32U) * 16U + lane / 4U;
  for (unsigned item = 0U; item < 2U; ++item) {
    const auto m = row0 + item * 8U;
    if (m >= valid_rows)
      continue;
    const auto saved = a.checkpoint.rows[mblock * forward_tuning::kTileM + m];
#pragma unroll
    for (unsigned index = 0U; index < Fc1TileN / 16U; ++index) {
      const auto *g = accum + index * 8U + item * 2U;
      const auto *u = g + 4U;
      const auto column = column_base + index * 8U + (lane % 4U) * 2U;
      auto *gate = a.saved_ab + saved * a.intermediate + column;
      auto *up = gate + a.saved_rows * a.intermediate;
      *reinterpret_cast<__nv_bfloat162 *>(gate) =
          __float22bfloat162_rn(make_float2(g[0], g[1]));
      *reinterpret_cast<__nv_bfloat162 *>(up) =
          __float22bfloat162_rn(make_float2(u[0], u[1]));
    }
  }
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

template <unsigned Fc1TileN, unsigned ScratchElements,
          bool RetainBackwardState = false, bool TmaStore = false,
          bool PrivateScratch = false, bool FastMath = false, class Args>
__device__ __forceinline__ void static_v4_store_fc1_tile(
    const Args &a, transport::BFloat16 *tile, const float *accum,
    transport::BFloat16 *output, const float *gates, std::uint64_t row,
    std::uint32_t valid_rows, std::uint32_t column_base, std::uint64_t mblock,
    const CUtensorMap *store_map) {
  constexpr auto kOutputN = Fc1TileN / 2U;
  const auto lane = threadIdx.x % 32U;
  const auto row0 = (threadIdx.x / 32U) * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
  // Keep the two invariant gate loads outside the fragment-store loop.
  const float gate0 = row0 < valid_rows ? gates[row + row0] : 0.0F;
  const float gate1 = row1 < valid_rows ? gates[row + row1] : 0.0F;

  constexpr auto kOutputElements = forward_tuning::kTileM * kOutputN;
  constexpr auto kPlanes = RetainBackwardState ? 3U : 1U;
  if (kOutputN >= 64U && kPlanes * kOutputElements <= ScratchElements &&
      valid_rows == forward_tuning::kTileM) {
    // Both math warpgroups must retire their WGMMA B reads before staging
    // into that CTA-private arena. A is multicast and must not be reused.
    // Stage complete N64 swizzle atoms. The narrow decode entry retains its
    // original direct epilogue rather than staging a partial atom.
    // Training packs G/U into two additional retired B planes for vector
    // checkpoint writeback in canonical row order.
    if constexpr (!PrivateScratch)
      cutlass::arch::NamedBarrier::sync(
          kMathThreads, physical_barrier<true>(kMathReadyBarrier));
#pragma unroll
    for (unsigned index = 0; index < kOutputN / 8U; ++index) {
      const auto *g = accum + index * 8U;
      const auto *u = g + 4U;
      store_m_fragment(
          tile, index,
          __float22bfloat162_rn(static_v4_swiglu_pair<FastMath>(g, u, gate0)),
          __float22bfloat162_rn(static_v4_swiglu_pair<FastMath>(g + 2U, u + 2U, gate1)));
      if constexpr (RetainBackwardState) {
        store_m_fragment(
            tile + kOutputElements, index,
            __float22bfloat162_rn(make_float2(g[0], g[1])),
            __float22bfloat162_rn(make_float2(g[2], g[3])));
        store_m_fragment(
            tile + 2U * kOutputElements, index,
            __float22bfloat162_rn(make_float2(u[0], u[1])),
            __float22bfloat162_rn(make_float2(u[2], u[3])));
      }
    }
    if constexpr (TmaStore)
      cute::tma_store_fence();
    cutlass::arch::NamedBarrier::sync(
        kMathThreads, physical_barrier<true>(kMathDoneBarrier));
    if constexpr (TmaStore && kOutputN >= 64U) {
      if (threadIdx.x < kOutputN / 64U) {
        const auto atom = threadIdx.x;
        cute::SM90_TMA_STORE_2D::copy(
            store_map, tile + atom * forward_tuning::kTileM * 64U,
            column_base + atom * 64U, static_cast<std::uint32_t>(row));
        cute::tma_store_arrive();
      }
    }
    if constexpr (RetainBackwardState || !TmaStore) {
      for (unsigned vector = threadIdx.x;
           vector < forward_tuning::kTileM * kOutputN / 8U;
           vector += kMathThreads) {
        const auto m = vector / (kOutputN / 8U);
        const auto n = (vector % (kOutputN / 8U)) * 8U;
        const auto offset = m_fragment_vector_offset(m, n);
        if constexpr (!TmaStore)
          *reinterpret_cast<uint4 *>(output + (row + m) * a.intermediate +
                                     column_base + n) =
              *reinterpret_cast<const uint4 *>(tile + offset);
        if constexpr (RetainBackwardState) {
          const auto saved =
              a.checkpoint.rows[mblock * forward_tuning::kTileM + m];
          auto *gate = a.saved_ab + saved * a.intermediate + column_base + n;
          auto *up = gate + a.saved_rows * a.intermediate;
          *reinterpret_cast<uint4 *>(gate) =
              *reinterpret_cast<const uint4 *>(tile + kOutputElements + offset);
          *reinterpret_cast<uint4 *>(up) =
              *reinterpret_cast<const uint4 *>(tile + 2U * kOutputElements + offset);
        }
      }
    }
    // The caller's math join includes every issuer before fc1_done and B
    // reuse. Read-only completion is insufficient for publishing ring S.
    if constexpr (TmaStore && kOutputN >= 64U)
      if (threadIdx.x < kOutputN / 64U)
        asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
    return;
  }

  if constexpr (RetainBackwardState)
    static_v4_save_fc1_tile<Fc1TileN>(a, accum, valid_rows, column_base, mblock);

  // Decode and compact expert tails avoid staging and its extra CTA barriers.
#pragma unroll
  for (unsigned index = 0; index < kOutputN / 8U; ++index) {
    const auto *g = accum + index * 8U;
    const auto *u = g + 4U;
    const auto column = column_base + index * 8U + (lane % 4U) * 2U;
    if (row0 < valid_rows) {
      const auto value = make_float2(
          gated_swiglu_fc2_input_from_bf16<FastMath>(
              float_to_bf16_bits(g[0]), float_to_bf16_bits(u[0]), gate0),
          gated_swiglu_fc2_input_from_bf16<FastMath>(
              float_to_bf16_bits(g[1]), float_to_bf16_bits(u[1]), gate0));
      *reinterpret_cast<std::uint32_t *>(
          output + (row + row0) * a.intermediate + column) =
          float2_to_bf16x2_bits(value.x, value.y);
    }
    if (row1 < valid_rows) {
      const auto value = make_float2(
          gated_swiglu_fc2_input_from_bf16<FastMath>(
              float_to_bf16_bits(g[2]), float_to_bf16_bits(u[2]), gate1),
          gated_swiglu_fc2_input_from_bf16<FastMath>(
              float_to_bf16_bits(g[3]), float_to_bf16_bits(u[3]), gate1));
      *reinterpret_cast<std::uint32_t *>(
          output + (row + row1) * a.intermediate + column) =
          float2_to_bf16x2_bits(value.x, value.y);
    }
  }
}
