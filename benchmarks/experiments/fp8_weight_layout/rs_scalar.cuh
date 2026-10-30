#pragma once

// C' = W' * G': original weights enter registers, gradients stay K-major SMEM.
template <size_t... I>
__device__ __forceinline__ void rs_mma(const unsigned (&v)[4], std::uint64_t b,
                                     float *d, bool accumulate,
                                     cute::index_sequence<I...>) {
  cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN<>::fma(
      v[0], v[1], v[2], v[3], b, d[I]...,
      accumulate ? cute::SM90::GMMA::ScaleOut::One : cute::SM90::GMMA::ScaleOut::Zero);
}

template <unsigned N>
__device__ __forceinline__ void weight_rs_tile(const std::uint8_t *g,
    const std::uint8_t *w, const float *sg, const float *sw, float *total) {
  const unsigned lane = threadIdx.x % 32U;
  const unsigned wr = threadIdx.x / 32U * 16U + lane / 4U;
  auto db = hopper::make_gmma_desc<hopper::Major::K, 128, 128, 128>(g, 0U, 0U);
  const auto bbase = __shfl_sync(0xffffffffU, db.reg32_[0], 0U);
  float partial[64];
#pragma unroll
  for (unsigned segment = 0; segment < N / 128U; ++segment) {
#pragma unroll
    for (auto &v : partial) cute::warpgroup_fence_operand(v);
    unsigned regs[4][4];
#pragma unroll
    for (unsigned k = 0; k < 128U; k += 32U) {
      auto &reg = regs[k / 32U];
#pragma unroll
      for (unsigned i = 0; i < 4U; ++i) {
        const unsigned row = wr + i % 2U * 8U;
        const unsigned col = k + lane % 4U * 4U + i / 2U * 16U;
        unsigned v = 0;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j)
          v |= unsigned(w[segment * 16384U + fp8::tile_offset(col + j, row)]) << (j * 8U);
        reg[i] = v;
      }
    }
    cute::warpgroup_arrive();
#pragma unroll
    for (unsigned k = 0; k < 128U; k += 32U) {
      db.reg32_[0] = bbase + k / 16U;
      rs_mma(regs[k / 32U], db, partial, k != 0U, cute::make_index_sequence<64>{});
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
#pragma unroll
    for (auto &v : partial) cute::warpgroup_fence_operand(v);
#pragma unroll
    for (unsigned j = 0; j < 16U; ++j) {
#pragma unroll
      for (unsigned e = 0; e < 4U; ++e) {
        const unsigned token = lane % 4U * 2U + j * 8U + e % 2U;
        const auto scale = sg[token] * sw[segment * 128U];
        const unsigned idx = segment * 64U + j * 4U + e;
        total[idx] = fmaf(partial[j * 4U + e], scale, total[idx]);
      }
    }
  }
}


template <unsigned N, bool Dx>
__device__ __forceinline__ void weight_rs_store(const ForwardKernelArgs &f,
    const BackwardKernelArgs &a, Shared &s, const Task &t, const float *accum) {
  static_assert(WEFTMOE_FP8_STREAMED_WGRAD);
  // The producer may already fill the next task's operand stages. Reuse only
  // the math-owned X conversion scratch, after its cp.async has been drained.
  auto *scratch = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<unsigned char *>(&s) + kFP8BaseSharedBytes);
  const unsigned lane = threadIdx.x % 32U;
  const unsigned wr = (threadIdx.x % 128U) / 32U * 16U + lane / 4U;
  const auto *mapping = peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
#pragma unroll
  for (unsigned part = 0; part < N / 64U; ++part) {
    fp8::MathSync::sync();
    if (threadIdx.x / 128U == part % 2U) {
#pragma unroll
      for (unsigned j = 0; j < 16U; ++j)
#pragma unroll
        for (unsigned e = 0; e < 4U; ++e) {
          const unsigned row = lane % 4U * 2U + j * 8U + e % 2U;
          const unsigned col = wr + e / 2U * 8U;
          scratch[(row * 64U + col) ^ ((row & 7U) * 8U)] =
              __float2bfloat16_rn(accum[part / 2U * 64U + j * 4U + e]);
        }
    }
    fp8::MathSync::sync();
    for (unsigned idx = threadIdx.x * 8U; idx < 128U * 64U; idx += 256U * 8U) {
      const unsigned row = idx / 64U, col = idx % 64U;
      if (row >= t.rows) continue;
      transport::BFloat16 *output;
      if constexpr (Dx)
        output = static_cast<transport::BFloat16 *>(f.ring.rows[t.block * kM + row].output);
      else
        output = a.recomputed_u + std::uint64_t(a.fp8_du_ring ? t.row + row
            : mapping[f.checkpoint.rows[t.block * kM + row]]) * a.intermediate;
      *reinterpret_cast<uint4 *>(output + t.column * N + part * 64U + col) =
          *reinterpret_cast<const uint4 *>(scratch + (idx ^ ((row & 7U) * 8U)));
    }
  }
}
