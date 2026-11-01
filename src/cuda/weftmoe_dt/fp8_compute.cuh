#pragma once

namespace weftmoe_dt {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

// Math warps prepare each source row once while CTA0's control warps run
// mini Notify. The existing BeforePull rank barrier publishes both results.
template <unsigned MathThreads>
__device__ __forceinline__ void stage_fp8_source(const ForwardKernelArgs &a) {
  constexpr unsigned H = forward_tuning::kMainHidden;
  const unsigned lane = threadIdx.x % 32U, warp = threadIdx.x / 32U;
  auto *base = peer_at<std::uint8_t>(a.local_base, a.source_input_offset);
  for (auto row = std::uint64_t(blockIdx.x) * (MathThreads / 32U) + warp;
       row < a.dispatch.identity.num_tokens;
       row += std::uint64_t(gridDim.x) * (MathThreads / 32U)) {
    auto *packet = base + row * fp8::source_pitch(H);
    auto *scales = reinterpret_cast<float *>(packet + H);
    if (a.prequantized_input) {
      for (unsigned col = lane * 16U; col < H; col += 512U)
        *reinterpret_cast<uint4 *>(packet + col) =
            *reinterpret_cast<const uint4 *>(a.prequantized_input + row * H +
                                             col);
      for (unsigned col = lane; col < H / 128U; col += 32U)
        scales[col] = a.prequantized_scales[row * (H / 128U) + col];
    } else {
      for (unsigned col = lane * 8U; col < H; col += 256U) {
        const auto v =
            *reinterpret_cast<const uint4 *>(a.input + row * H + col);
        fp8::store_fp8_eight(v, packet, H, col, scales);
      }
    }
  }
}

__device__ __forceinline__ void
produce_fp8(const ForwardKernelArgs &a, Shared<true> &s,
            const CUtensorMap &x_map, const CUtensorMap &s_map,
            const CUtensorMap &w1_map, const CUtensorMap &w2_map) {
  static_assert(sizeof(s.a[0]) == kM * 128U);
  static_assert(sizeof(s.b[0]) == 256U * 128U);
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto tasks =
      compute_units(s, owned) * (kFc1Tiles / 2U + kFc2Tiles / 2U);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto block = task(s, owned, index);
    const auto physical_rows = ready_rows(s, owned, block);
    const auto *ready = block.fc2 ? &states(a)[block.block].s_ready
                                  : &states(a)[block.block].x_ready;
    const auto target =
        block.fc2 ? ((physical_rows + kM - 1U) / kM) * (kFc1Tiles / 2U)
                  : physical_rows;
    const unsigned reduction = block.fc2 ? forward_tuning::kMainIntermediate
                                         : forward_tuning::kMainHidden;
    for (unsigned offset = 0; offset < block.rows; offset += kM) {
      const auto t = slice(block, offset);
      {
        for (unsigned k = 0; k < reduction; k += 128U) {
          s.empty[stage].wait(phase ^ 1U);
          if (cute::elect_one_sync()) {
            s.full[stage].expect_transaction((kM + 256U) * 128U);
            if (!t.fc2)
              cute::SM90_TMA_LOAD_5D::copy(
                  &w1_map, reinterpret_cast<std::uint64_t *>(&s.full[stage]),
                  static_cast<std::uint64_t>(
                      cute::TMA::CacheHintSm90::EVICT_NORMAL),
                  s.b[stage], k, 0, 0, t.column * 16U, t.expert);
            else
              hopper::tma_load<128U, 256U, 128U>(
                  &w2_map, &s.full[stage],
                  reinterpret_cast<std::uint8_t *>(s.b[stage]), k,
                  t.expert * forward_tuning::kMainHidden + t.column * 256U);
            if (offset == 0U && k == 0U)
              while (weftmoe_pf::load_acquire_device(ready) != target) {
              }
            cute::SM90_TMA_LOAD_3D::copy(
                t.fc2 ? &s_map : &x_map,
                reinterpret_cast<std::uint64_t *>(&s.full[stage]),
                static_cast<std::uint64_t>(
                    cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.a[stage], k, t.row % 128U, t.row / 128U);
            s.full[stage].arrive();
          }
          stage = (stage + 1U) % stages<true>;
          phase ^= stage == 0U;
        }
      }
    }
  }
}

template <unsigned N, bool FC2>
__device__ __forceinline__ void
mma_fp8(const ForwardKernelArgs &a, Shared<true> &s, const Task &t, unsigned k,
        unsigned stage, float *total) {
  const auto lane = threadIdx.x % 32U, warp = (threadIdx.x / 32U) % 4U;
  auto *qa = reinterpret_cast<std::uint8_t *>(s.a[stage]);
  auto *qb = reinterpret_cast<std::uint8_t *>(s.b[stage]);
  const unsigned columns = FC2 ? unsigned(a.intermediate) : unsigned(a.hidden);
  auto *base = peer_at<std::uint8_t>(
      a.local_base, FC2 ? a.forward_s_offset : a.canonical_rows_offset);
  const unsigned pitch =
      FC2 ? unsigned(a.fp8_s_pitch) : unsigned(a.fp8_x_pitch);
  const unsigned row = warp * 16U + lane / 4U;
  const float sa0 =
      fp8::ring_scales(base, t.row + row, columns, pitch)[k / 128U * 128U];
  const float sa1 =
      fp8::ring_scales(base, t.row + row + 8U, columns, pitch)[k / 128U * 128U];
  // A row block and the two gate/up (or adjacent output) weight blocks
  // need only four products; retire scale addressing before issuing MMA.
  const unsigned kb = k / 128U, stride = columns / 128U;
  const float *ws =
      FC2 ? a.fp8.s2 +
                (std::uint64_t(t.expert) * a.hidden / 128U +
                 t.column * (N / 128U)) *
                    stride +
                kb
          : a.fp8.s1 +
                (std::uint64_t(t.expert) * 2U * a.intermediate / 128U +
                 t.column) *
                    stride +
                kb;
  const float w0 = ws[0];
  const float w1 = FC2 ? (N == 256U ? ws[stride] : w0)
                       : ws[(a.intermediate / 128U) * stride];
  const float factors[4] = {sa0 * w0, sa1 * w0, sa0 * w1, sa1 * w1};
  // Two math warpgroups split N256 into independent N128 accumulators.
  using Mma = hopper::FP8Mma<128U>;
  const unsigned wg = threadIdx.x / 128U;
  auto da =
      hopper::make_gmma_desc<hopper::Major::K, kM, 128U, 128U>(qa, 0U, 0U);
  auto db = hopper::make_gmma_desc<hopper::Major::K, N, 128U, 128U>(qb, 0U, 0U);
  const auto abase = da.reg32_[0], bbase = db.reg32_[0];
  float partial[Mma::kNumAccum];
#pragma unroll
  for (auto &v : partial)
    cute::warpgroup_fence_operand(v);
  cute::warpgroup_arrive();
#pragma unroll
  for (unsigned atom = 0; atom < 4U; ++atom) {
    da.reg32_[0] = abase + atom * 2U;
    db.reg32_[0] = bbase + wg * 128U * 128U / 16U + atom * 2U;
    Mma::wgmma(da, db, partial, atom != 0U);
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
#pragma unroll
  for (auto &v : partial)
    cute::warpgroup_fence_operand(v);
  // Two consumer arrivals retire the stage. Each warpgroup can promote
  // its accumulator independently; the producer still waits for both.
  if (threadIdx.x % 128U == 0U)
    s.empty[stage].arrive();
#pragma unroll
  for (unsigned j = 0; j < Mma::kNumAccum / 4U; ++j) {
    const unsigned group = FC2 ? wg : j % 2U;
#pragma unroll
    for (unsigned e = 0; e < 4U; ++e)
      total[j * 4U + e] = fmaf(partial[j * 4U + e],
                               factors[group * 2U + e / 2U], total[j * 4U + e]);
  }
}

// Round the complete N256 gate/up tile before reducing each 128-channel amax.
// XOR row bits into the shared column to avoid eight-way accumulator stores.
// The FC2 output scratch aliases this epilogue; the TMA operand stages do not.
__device__ __forceinline__ void save_ab(Shared<true> &s, const float *accum) {
  const unsigned lane = threadIdx.x % 32U, warp = (threadIdx.x / 32U) % 4U;
#pragma unroll
  for (unsigned item = 0; item < 2U; ++item) {
    const unsigned row = warp * 16U + lane / 4U + item * 8U;
#pragma unroll
    for (unsigned i = 0; i < 128U / 16U; ++i) {
      const auto *g = accum + i * 8U + item * 2U;
      const unsigned c = (threadIdx.x / 128U) * 64U + i * 8U + (lane % 4U) * 2U;
      *reinterpret_cast<__nv_bfloat162 *>(s.output + row * 256U +
                                          (c ^ ((row & 7U) * 8U))) =
          __float22bfloat162_rn(make_float2(g[0], g[1]));
      *reinterpret_cast<__nv_bfloat162 *>(s.output + row * 256U + 128U +
                                          (c ^ ((row & 7U) * 8U))) =
          __float22bfloat162_rn(make_float2(g[4], g[5]));
    }
  }
}

__device__ __forceinline__ void store_s_fp8(const ForwardKernelArgs &a,
                                            Shared<true> &s, const Task &t) {
  const unsigned lane = threadIdx.x % 32U, warp = threadIdx.x / 32U;
  weftmoe_pf::collective_scope_sync<kComputeThreads<true>, 7U>();
  auto *out = peer_at<std::uint8_t>(a.local_base, a.forward_s_offset);
  for (unsigned row = warp; row < t.rows; row += kComputeThreads<true> / 32U) {
    float av[4], bv[4], ma = 0, mb = 0;
    bool invalid_a = false, invalid_b = false;
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
      av[j] = __bfloat162float(reinterpret_cast<__nv_bfloat16 *>(
          s.output)[row * 256U + ((lane * 4U + j) ^ ((row & 7U) * 8U))]);
      bv[j] = __bfloat162float(reinterpret_cast<__nv_bfloat16 *>(
          s.output)[row * 256U + 128U + ((lane * 4U + j) ^ ((row & 7U) * 8U))]);
      ma = fmaxf(ma, fabsf(av[j]));
      mb = fmaxf(mb, fabsf(bv[j]));
      invalid_a |= !isfinite(av[j]);
      invalid_b |= !isfinite(bv[j]);
    }
    invalid_a = __any_sync(0xffffffffU, invalid_a);
    invalid_b = __any_sync(0xffffffffU, invalid_b);
    const float sa = invalid_a ? nanf("") : fp8::block_scale(fp8::warp_max(ma));
    const float sb = invalid_b ? nanf("") : fp8::block_scale(fp8::warp_max(mb));
    const float ia =
        invalid_a ? sa : __uint_as_float(0x7f000000U - __float_as_uint(sa));
    const float ib =
        invalid_b ? sb : __uint_as_float(0x7f000000U - __float_as_uint(sb));
    unsigned values[2];
    float maximum = 0;
    bool invalid = false;
    const float gate = a.ring.rows[t.row + row].gate;
#pragma unroll
    for (unsigned j = 0; j < 2U; ++j) {
      const __nv_fp8x2_e4m3 aq(
          make_float2(av[j * 2U] * ia, av[j * 2U + 1U] * ia));
      const __nv_fp8x2_e4m3 bq(
          make_float2(bv[j * 2U] * ib, bv[j * 2U + 1U] * ib));
      const auto ax = static_cast<float2>(aq), bx = static_cast<float2>(bq);
      const float g[2] = {ax.x * sa, ax.y * sa}, u[2] = {bx.x * sb, bx.y * sb};
      const auto v = forward_compute_detail::static_v4_swiglu_pair<
          forward_tuning::kLocalPrefillFastMath>(g, u, gate);
      values[j] = float2_to_bf16x2_bits(v.x, v.y);
      const auto rounded = bf16x2_bits_to_float2(values[j]);
      maximum = fmaxf(maximum, fmaxf(fabsf(rounded.x), fabsf(rounded.y)));
      invalid |= !isfinite(rounded.x) || !isfinite(rounded.y);
    }
    invalid = __any_sync(0xffffffffU, invalid);
    const float scale =
        invalid ? nanf("") : fp8::block_scale(fp8::warp_max(maximum));
    const float inverse =
        invalid ? scale : __uint_as_float(0x7f000000U - __float_as_uint(scale));
    unsigned q = 0;
#pragma unroll
    for (unsigned j = 0; j < 2U; ++j) {
      const auto v = bf16x2_bits_to_float2(values[j]);
      q |= unsigned(
               __nv_fp8x2_e4m3(make_float2(v.x * inverse, v.y * inverse)).__x)
           << (j * 16U);
    }
    auto *row_out = fp8::ring_row(out, t.row + row, unsigned(a.intermediate),
                                  unsigned(a.fp8_s_pitch));
    *reinterpret_cast<unsigned *>(row_out + t.column * 128U + lane * 4U) = q;
    if (!lane)
      fp8::ring_scales(out, t.row + row, unsigned(a.intermediate),
                       unsigned(a.fp8_s_pitch))[t.column * 128U] = scale;
  }
  __threadfence();
  weftmoe_pf::collective_scope_sync<kComputeThreads<true>, 7U>();
  if (!threadIdx.x)
    weftmoe_pf::reduce_add_release_device(&states(a)[t.block].s_ready, 1ULL);
  weftmoe_pf::collective_scope_sync<kComputeThreads<true>, 7U>();
}

template <bool FC2>
__device__ __forceinline__ void
compute_fp8_tile(const ForwardKernelArgs &a, Shared<true> &s, const Task &t,
                 unsigned offset, unsigned &stage, unsigned &phase) {
  constexpr unsigned N = 256U;
  float accum[N / 4U] = {};
  const unsigned reduction =
      FC2 ? unsigned(a.intermediate) : unsigned(a.hidden);
  for (unsigned k = 0; k < reduction; k += 128U) {
    s.full[stage].wait(phase);
    mma_fp8<N, FC2>(a, s, t, k, stage, accum);
    stage = (stage + 1U) % stages<true>;
    phase ^= stage == 0U;
  }
  if constexpr (FC2)
    store_fc2<true, 256U>(a, s, t, accum, offset);
  else {
    save_ab(s, accum);
    store_s_fp8(a, s, t);
  }
}

__device__ __forceinline__ void compute_fp8(const ForwardKernelArgs &a,
                                            Shared<true> &s) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto fc1_tasks = compute_units(s, owned) * (kFc1Tiles / 2U);
  const auto tasks = fc1_tasks + compute_units(s, owned) * (kFc2Tiles / 2U);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto block = task(s, owned, index);
    const auto physical_rows = ready_rows(s, owned, block);
    const auto target =
        block.fc2 ? ((physical_rows + kM - 1U) / kM) * (kFc1Tiles / 2U)
                  : physical_rows;
    const auto *ready = block.fc2 ? &states(a)[block.block].s_ready
                                  : &states(a)[block.block].x_ready;
    while (weftmoe_pf::load_acquire_device(ready) != target) {
    }
    const bool first = index < (block.fc2 ? fc1_tasks : 0U) + gridDim.x;
    const bool last = index + gridDim.x >= (block.fc2 ? tasks : fc1_tasks);
    if (!threadIdx.x && first)
      atomicMin(block.fc2 ? &status->dispatch_blocks_done
                          : &status->fc1_start_clock,
                device_globaltimer_ns());
    for (unsigned offset = 0; offset < block.rows; offset += kM) {
      const auto t = slice(block, offset);
      if (t.fc2) {
        if (!threadIdx.x && first && offset == 0U)
          atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
        compute_fp8_tile<true>(
            a, s, t,
            offset +
                (kSliceSchedule<true> ? block.row - block.block * kBlockM : 0U),
            stage, phase);
      } else
        compute_fp8_tile<false>(a, s, t, 0U, stage, phase);
    }
    if (!threadIdx.x && last)
      atomicMax(block.fc2 ? &status->fc2_complete_clock
                          : &status->fc1_complete_clock,
                device_globaltimer_ns());
  }
}
#endif
} // namespace weftmoe_dt
