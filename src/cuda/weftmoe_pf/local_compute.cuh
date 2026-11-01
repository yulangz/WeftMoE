#pragma once

#include "cuda/weftmoe_pf/local_schedule.cuh"

namespace weftmoe_pf::local {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

// Gate/up and S retain the BF16 rounding boundaries of the unfused contract.
// Four-lane groups own complete rows, including partial M tails.
template <bool FastMath, bool Packed, class Args>
__device__ __forceinline__ void store_fp8_swiglu(
    const Args &a, const float *accum, transport::BFloat16 *output,
    const float *gates, unsigned row, unsigned rows, unsigned column, unsigned block) {
  const unsigned lane = threadIdx.x % 32U;
  const unsigned row0 = threadIdx.x / 32U * 16U + lane / 4U;
  const auto store_eight = [](std::uint8_t *destination, const unsigned (&words)[4], unsigned mask) {
    const unsigned lane = threadIdx.x % 4U;
    unsigned lo = words[0] | (words[1] << 16U);
    unsigned hi = words[2] | (words[3] << 16U);
    // A 4x4 halfword transpose: exchange packed pairs, then interleave
    // halfwords. Three shuffles suffice instead of four word transposes.
    const auto other = __shfl_xor_sync(mask, (lane & 2U) ? lo : hi, 2, 4);
    lo = (lane & 2U) ? other : lo;
    hi = (lane & 2U) ? hi : other;
    const unsigned selector = (lane & 1U) ? 0x3276U : 0x5410U;
    const auto x = __byte_perm(lo, __shfl_xor_sync(mask, lo, 1, 4), selector);
    const auto y = __byte_perm(hi, __shfl_xor_sync(mask, hi, 1, 4), selector);
    *reinterpret_cast<uint2 *>(destination) = make_uint2(x, y);
  };
#pragma unroll
  for (unsigned item = 0; item < 2U; ++item) {
    const auto m = row0 + item * 8U;
    if (m >= rows)
      continue;
    // All valid four-lane row groups participate with the same warp mask.
    // Width-four shuffles keep rows independent without dynamic subgroup masks.
    const unsigned valid = min(8U, rows - (threadIdx.x / 32U * 16U + item * 8U));
    const unsigned mask = 0xffffffffU >> ((8U - valid) * 4U);
    const auto row_max = [mask](unsigned x) {
      x = max(x, __shfl_xor_sync(mask, x, 1, 4));
      return max(x, __shfl_xor_sync(mask, x, 2, 4));
    };
    // Capture read-only row metadata before any output stores can alias it.
    const float route_gate = gates[row + m];
    const auto saved = a.fp8_ab_data ? a.checkpoint.rows[std::uint64_t(block) * kM + m] : 0U;
    auto *ab_row = a.fp8_ab_data ? a.fp8_ab_data + saved * a.intermediate + column : nullptr;
    const auto ab_plane = a.saved_rows * a.intermediate;
    float scale_a = 1.0F, scale_b = 1.0F;
    if (a.fp8_ab_quantized) {
      // Nonnegative IEEE bit patterns preserve magnitude order, including
      // NaN/Inf above finite values. This folds invalid detection into amax.
      unsigned max_a = 0U, max_b = 0U;
#pragma unroll
      for (unsigned j = 0; j < 16U; ++j) {
        const auto *g = accum + j * 8U + item * 2U;
        max_a = max(max_a, max(__float_as_uint(g[0]) & 0x7fffffffU,
                               __float_as_uint(g[1]) & 0x7fffffffU));
        max_b = max(max_b, max(__float_as_uint(g[4]) & 0x7fffffffU,
                               __float_as_uint(g[5]) & 0x7fffffffU));
      }
      max_a = row_max(max_a);
      max_b = row_max(max_b);
      scale_a = fp8::block_scale(__bfloat162float(__float2bfloat16_rn(__uint_as_float(max_a))));
      scale_b = fp8::block_scale(__bfloat162float(__float2bfloat16_rn(__uint_as_float(max_b))));
      if (ab_row && !(lane % 4U)) {
        const auto offset = saved * (a.intermediate / 128U) + column / 128U;
        a.fp8_ab_scales[offset] = scale_a;
        a.fp8_ab_scales[offset + a.saved_rows * (a.intermediate / 128U)] = scale_b;
      }
    }
    const float inv_a = isfinite(scale_a) ? __uint_as_float(0x7f000000U - __float_as_uint(scale_a)) : scale_a;
    const float inv_b = isfinite(scale_b) ? __uint_as_float(0x7f000000U - __float_as_uint(scale_b)) : scale_b;
    // S has already been rounded to BF16. Keep packed pairs until the final
    // FP8 cast instead of holding two FP32 registers per pair across amax.
    unsigned values[16], maximum = 0U;
#pragma unroll
    for (unsigned index = 0; index < 16U; index += 4U) {
      unsigned aq_words[4], bq_words[4];
#pragma unroll
      for (unsigned j = 0; j < 4U; ++j) {
        const auto *g = accum + (index + j) * 8U + item * 2U;
        float2 value;
        if (a.fp8_ab_quantized) {
          auto av = __bfloat1622float2(__float22bfloat162_rn(make_float2(g[0], g[1])));
          auto bv = __bfloat1622float2(__float22bfloat162_rn(make_float2(g[4], g[5])));
          const auto aq = __nv_fp8x2_e4m3(make_float2(av.x * inv_a, av.y * inv_a));
          const auto bq = __nv_fp8x2_e4m3(make_float2(bv.x * inv_b, bv.y * inv_b));
          aq_words[j] = aq.__x; bq_words[j] = bq.__x;
          av = static_cast<float2>(aq); bv = static_cast<float2>(bq);
          const float ga[2] = {av.x * scale_a, av.y * scale_a};
          const float up[2] = {bv.x * scale_b, bv.y * scale_b};
          value = forward_compute_detail::static_v4_swiglu_pair<FastMath>(ga, up, route_gate);
        } else {
          value = forward_compute_detail::static_v4_swiglu_pair<FastMath>(g, g + 4U, route_gate);
        }
        const auto bits = float2_to_bf16x2_bits(value.x, value.y);
        values[index + j] = bits;
        maximum = max(maximum, max(bits & 0x7fffU, (bits >> 16U) & 0x7fffU));
      }
      if (a.fp8_ab_quantized && ab_row) {
        const auto offset = index * 8U + lane % 4U * 8U;
        store_eight(ab_row + offset, aq_words, mask);
        store_eight(ab_row + ab_plane + offset, bq_words, mask);
      }
    }
    maximum = row_max(maximum);
    const auto scale = fp8::block_scale(__uint_as_float(maximum << 16U));
    const auto inverse = isfinite(scale) ? __uint_as_float(0x7f000000U - __float_as_uint(scale)) : scale;
    auto *buffer = reinterpret_cast<std::uint8_t *>(output);
    auto *base = Packed
        ? fp8::ring_row(buffer, row + m, unsigned(a.intermediate), unsigned(a.fp8_s_pitch))
        : buffer + std::uint64_t(row + m) * a.intermediate * 2U;
#pragma unroll
    for (unsigned index = 0; index < 16U; index += 4U) {
      unsigned words[4];
#pragma unroll
      for (unsigned j = 0; j < 4U; ++j) {
        const auto v = bf16x2_bits_to_float2(values[index + j]);
        words[j] = __nv_fp8x2_e4m3(make_float2(v.x * inverse, v.y * inverse)).__x;
      }
      store_eight(base + column + index * 8U + lane % 4U * 8U, words, mask);
    }
    if (!(lane % 4U)) {
      auto *sf = Packed
          ? fp8::ring_scales(buffer, row + m, unsigned(a.intermediate), unsigned(a.fp8_s_pitch))
          : reinterpret_cast<float *>(base + a.intermediate);
      sf[column / 128U * (Packed ? 128U : 1U)] = scale;
    }
  }
}

template <bool Network>
__device__ __forceinline__ void compute_fp8_tile(
    const ForwardKernelArgs &a, Shared &s, const Task &t, float *accum,
    unsigned &stage, unsigned &phase) {
  if constexpr (kN == 256U) {
    const auto reduction = t.fc2 ? forward_tuning::kMainIntermediate : forward_tuning::kMainHidden;
#if WEFTMOE_DEVICE_TIMELINE
    std::uint64_t tma_wait_ns = 0U;
#endif
    for (unsigned k = 0; k < reduction; k += 128U) {
#if WEFTMOE_DEVICE_TIMELINE
      const auto wait_begin = threadIdx.x == 0U ? profiling::read_globaltimer_ns() : 0U;
#endif
      s.full[stage].wait(phase);
#if WEFTMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U)
        tma_wait_ns += profiling::read_globaltimer_ns() - wait_begin;
#endif
      // Match BF16's boundary: first operand stage is ready, before MMA setup.
      if (k == 0U) {
        if (t.fc2) {
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2MmaBegin, Compute, t.column,
                                  t.block, 0U);
        } else {
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1MmaBegin, Compute, t.column,
                                  t.block, 0U);
        }
      }
      auto *scales = fp8_scales(s, stage);
      const bool cache_owner = !t.fc2 && a.fp8_x_data && t.column == (t.block + k / 128U) % kFc1Tiles;
      if constexpr (Network) {
        // Notify may precede dynamic tape allocation. Save reverse metadata
        // once, using the same first-K owner as X. A-ready publishes Pull's
        // gates; the FC1 epilogue fence publishes these checkpoint writes.
        if (cache_owner && k == 0U && threadIdx.x < t.rows) {
          const auto physical = std::uint64_t(t.block) * kM + threadIdx.x;
          const auto saved = a.checkpoint.rows[physical];
          const auto &identity = a.ring.rows[physical];
          a.checkpoint.gates[saved] = peer_at<const float>(
              a.local_base, a.canonical_gates_offset)[t.row + threadIdx.x];
          a.checkpoint.headers[saved] = {
              transport::kActivationTransportSchemaVersion,
              identity.source, a.rank, 0U, a.dispatch.identity.generation,
              execution_epoch(a), identity.route, identity.route,
              identity.token, saved,
              a.rank * a.dispatch.sizes.owned_expert_count + t.expert,
              unsigned(identity.route % a.dispatch.identity.top_k)};
        }
      }
      // A tail with fewer than 65 rows needs only the first math warpgroup.
      if (threadIdx.x / 128U * 64U < t.rows)
        fp8::multiply_tile<kN, fp8::NoSync, true, true>(reinterpret_cast<std::uint8_t *>(s.a[stage]),
            reinterpret_cast<std::uint8_t *>(s.b[stage]), scales, scales + kM, accum);
      // Stripe ownership over N tiles: each X K128 block is saved once.
      // The original row-scaled operand remains intact until all readers retire.
      if (cache_owner) {
        if constexpr (Network) {
          // Save each K128 operand once, after its GEMM load. Pull only
          // publishes the input and row metadata; checkpoint stores must not
          // delay the next row's arrival. Physical ring rows need not match
          // canonical expert rows, so remap both the row and its swizzle.
          std::uint64_t padded_begin = 0U;
          for (unsigned e = 0U; e < t.expert; ++e)
            padded_begin += (a.dispatch.expert_row_count[e] + 127U) / 128U * 128U;
          const auto expert_begin = a.dispatch.expert_row_prefix[t.expert];
          const auto *input = reinterpret_cast<const std::uint8_t *>(s.a[stage]);
          for (unsigned v = threadIdx.x; v < kM * 8U; v += kMathThreads) {
            const unsigned m = v / 8U, c = v % 8U * 16U;
            if (m < t.rows) {
              const auto saved = padded_begin + a.checkpoint.rows[std::uint64_t(t.block) * kM + m] - expert_begin;
              auto *destination = a.fp8_x_data + saved / 128U * 128U * reduction + k * 128U;
              *reinterpret_cast<uint4 *>(destination + fp8::tile_offset(saved % 128U, c)) =
                  *reinterpret_cast<const uint4 *>(input + fp8::tile_offset(m, c));
              if (!c)
                a.fp8_x_scales[saved / 128U * reduction + k + saved % 128U] = scales[m];
            }
          }
        } else {
          const auto offset = std::uint64_t(t.block) * kM * reduction + k * kM;
          // Retain the exact row-scaled operand. Column requantization belongs
          // to Backward, once per expert/token/channel block.
          const auto *input = reinterpret_cast<const uint4 *>(s.a[stage]);
          auto *output = reinterpret_cast<uint4 *>(a.fp8_x_data + offset);
          for (unsigned v = threadIdx.x; v < kM * 128U / sizeof(uint4); v += kMathThreads)
            output[v] = input[v];
          if (threadIdx.x < kM)
            a.fp8_x_scales[offset / 128U + threadIdx.x] = scales[threadIdx.x];
        }
      }
      if (threadIdx.x % 32U < 2U)
        s.empty[stage].arrive(threadIdx.x % 32U);
      stage = (stage + 1U) % kStages;
      phase ^= stage == 0U;
    }
    WEFTMOE_TIMELINE_RECORD(a.timeline, TmaWaitNs, Compute, unsigned(t.fc2),
                            tma_wait_ns, t.block);
    return;
  }

}

template <bool FP8 = false, class Scheduler = Schedule, bool Network = false>
__device__ __forceinline__ void
produce(const ForwardKernelArgs &a, Shared &s, const CUtensorMap &x_map,
        const CUtensorMap &s_map, const CUtensorMap &w1_map,
        const CUtensorMap &w2_map) {
  Scheduler schedule(a);
  Task t;
  unsigned stage = 0U, phase = 0U;
#if WEFTMOE_DEVICE_TIMELINE
  std::uint64_t tx_wait_ns = 0U, tx_wait_tiles = 0U;
#endif
  while (schedule.next(t)) {
    auto &slot = a.ring.slots[t.row / kM];
    if (cute::elect_one_sync()) {
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, InputWaitBegin, Compute,
                                 unsigned(t.fc2), t.block, t.column);
      if constexpr (Network)
        while (!ring::admitted(a, t.block)) {
        }
      const auto &ready = t.fc2 ? slot.fc1_done : slot.x_full;
      const auto target =
          std::uint64_t(t.generation) * (t.fc2 ? kFc1Tiles : kM);
      while (ring::acquire(ready) < target) {
      }
      if constexpr (Network && WEFTMOE_PF_RETURN_INDEPENDENT_TX) {
        if (t.fc2) {
          // Pass TX ownership through the existing input-stage barrier;
          // the math warpgroups need no extra epilogue rendezvous.
          const auto &desc = a.ring.blocks[t.block];
          const auto &consumed = a.ring.return_slots[desc.return_slot].consumed;
          const auto target = (desc.return_generation - 1U) * kM;
          if (ring::acquire<ring::kReturnScope>(consumed) < target) {
#if WEFTMOE_DEVICE_TIMELINE
            const auto begin = ::weftmoe::cuda::profiling::read_globaltimer_ns();
#endif
            while (ring::acquire<ring::kReturnScope>(consumed) < target) {
            }
#if WEFTMOE_DEVICE_TIMELINE
            tx_wait_ns += ::weftmoe::cuda::profiling::read_globaltimer_ns() - begin;
            ++tx_wait_tiles;
#endif
          }
        }
      }
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, InputWaitEnd, Compute,
                                 unsigned(t.fc2), t.block, t.column);
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __syncwarp();
    const auto reduction =
        t.fc2 ? forward_tuning::kMainIntermediate : forward_tuning::kMainHidden;
    for (unsigned k = 0; k < reduction; k += (FP8 ? 128U : kK)) {
      s.empty[stage].wait(phase ^ 1U);
      if constexpr (FP8) {
        auto *sa = fp8_scales(s, stage), *sb = sa + kM;
        const auto *input = peer_at<const std::uint8_t>(a.local_base,
            t.fc2 ? a.forward_s_offset : a.canonical_rows_offset) + std::uint64_t(t.row) * (t.fc2 ? a.fp8_s_pitch : a.fp8_x_pitch);
        if constexpr (Network && WEFTMOE_FP8_PACKED_RING && kFP8NetworkRowScales)
          if (!t.fc2)
            for (unsigned row = threadIdx.x % 32U; row < kM; row += 32U)
              sa[row] = row < t.rows ? reinterpret_cast<const float *>(input + kM * reduction)[row * (reduction / 128U) + k / 128U] : 1.0F;
        if constexpr (!WEFTMOE_FP8_PACKED_RING)
          for (unsigned row = threadIdx.x % 32U; row < kM; row += 32U)
            sa[row] = row < t.rows ? reinterpret_cast<const float *>(
                input + std::uint64_t(row) * reduction * 2U + reduction)[k / 128U] : 1.0F;
        const auto elements = t.fc2 ? a.hidden * a.intermediate : 2U * a.hidden * a.intermediate;
        fp8::WeightReader weight{nullptr,
            (t.fc2 ? a.fp8.s2 : a.fp8.s1) + t.expert * elements / (128U * 128U),
            unsigned(reduction), t.column * (t.fc2 ? kN : kN / 2U), unsigned(a.intermediate), !t.fc2};
        const auto lane = threadIdx.x % 32U;
        if (lane < 2U * (kN / 128U)) {
          const auto column = lane / 2U * 128U + lane % 2U * 8U;
          sb[lane] = weight.scale(column, k / 128U);
        }
        __syncwarp();
        if (cute::elect_one_sync()) {
          if constexpr (WEFTMOE_FP8_PACKED_RING) {
            auto *barrier = reinterpret_cast<std::uint64_t *>(&s.full[stage]);
            if (cute::block_rank_in_cluster() == 0U)
              cute::SM90_TMA_LOAD_MULTICAST_3D::copy(t.fc2 ? &s_map : &x_map, barrier, 3U,
                static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.a[stage], k, 0, t.row / kM);
            if (!Network || t.fc2 || !kFP8NetworkRowScales)
              cute::SM90_BULK_COPY_G2S::copy(input + kM * reduction + k * sizeof(float),
                  barrier, sa, kM * sizeof(float));
          } else
            hopper::tma_load<128U, kM, 128U>(t.fc2 ? &s_map : &x_map, &s.full[stage],
                reinterpret_cast<std::uint8_t *>(s.a[stage]), k, t.row, 2U);
          if (!t.fc2)
            cute::SM90_TMA_LOAD_5D::copy(&w1_map, reinterpret_cast<std::uint64_t *>(&s.full[stage]),
                static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.b[stage], k, 0, 0, t.column * (kN / 16U), t.expert);
          else
            hopper::tma_load<128U, kN, 128U>(&w2_map, &s.full[stage],
                reinterpret_cast<std::uint8_t *>(s.b[stage]), k,
                t.expert * forward_tuning::kMainHidden + t.column * kN);
          s.full[stage].arrive_and_expect_tx((kM + kN) * 128U +
              (WEFTMOE_FP8_PACKED_RING && (!Network || t.fc2 || !kFP8NetworkRowScales) ? kM * sizeof(float) : 0U));
        }
      } else {
        if (cute::elect_one_sync()) {
          hopper::tma_load<kK, kM, 128U>(t.fc2 ? &s_map : &x_map, &s.full[stage],
                                         s.a[stage], k, t.row, 2U);
          if (!t.fc2)
            cute::SM90_TMA_LOAD_5D::copy(
                &w1_map, reinterpret_cast<std::uint64_t *>(&s.full[stage]),
                static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.b[stage], k, 0, 0, t.column * (kN / 16U), t.expert);
          else
            hopper::tma_load<kK, kN, 128U>(&w2_map,
                                         &s.full[stage], s.b[stage], k,
                                         t.expert * forward_tuning::kMainHidden +
                                             t.column * kN);
          s.full[stage].arrive_and_expect_tx((kM + kN) * kK * 2U);
        }
      }
      stage = stage + 1U == kStages ? 0U : stage + 1U;
      phase ^= stage == 0U;
    }
  }
#if WEFTMOE_DEVICE_TIMELINE
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, cute::elect_one_sync(), TxCreditWaitNs,
                             Compute, 0U, tx_wait_ns, tx_wait_tiles);
#endif
}

template <bool FP8 = false, class Scheduler = Schedule, bool Network = false, class Matrix = Mma,
          unsigned WaitGroup = forward_tuning::kWgmmaWaitGroup>
__device__ __forceinline__ void compute(const ForwardKernelArgs &a, Shared &s,
                                        const CUtensorMap &output_map) {
  using Mma = Matrix;
  using namespace forward_compute_detail;
  Scheduler schedule(a);
  Task t;
  unsigned stage = 0U, phase = 0U;
  bool started_fc1 = false, started_fc2 = false;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  auto *output = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  auto da = hopper::make_gmma_desc<hopper::Major::K, kM, kK, 128U>(
      s.a[0], (threadIdx.x / 128U) * 64U, 0U);
  auto db =
      hopper::make_gmma_desc<hopper::Major::K, kN, kK, 128U>(s.b[0], 0U, 0U);
  const auto abase = __shfl_sync(0xffffffffU, da.reg32_[0], 0);
  const auto bbase = __shfl_sync(0xffffffffU, db.reg32_[0], 0);
  bool previous_fc2 = false;
  while (schedule.next(t)) {
    if (t.fc2) {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2TileStart, Compute, t.column,
                              t.block, 0U);
    } else {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1TileStart, Compute, t.column,
                              t.block, 0U);
    }
    auto &slot = a.ring.slots[t.row / kM];
    const bool started = t.fc2 ? started_fc2 : started_fc1;
    if (!started && threadIdx.x == 0U)
      atomicMin(t.fc2 ? &status->dispatch_blocks_done
                      : &status->fc1_start_clock,
                device_globaltimer_ns());
    started_fc1 |= !t.fc2;
    started_fc2 |= t.fc2;
    if (threadIdx.x == 0U && t.fc2 && !previous_fc2)
      atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
    previous_fc2 = t.fc2;
    float accum[Mma::kNumAccum] = {};
    const auto reduction =
        t.fc2 ? forward_tuning::kMainIntermediate : forward_tuning::kMainHidden;
    unsigned retirement = stage;
    constexpr auto kCheckpointRows = kM / 2U;
    const auto checkpoint_begin = (t.column & 1U) * kCheckpointRows;
    const auto checkpoint_rows =
        t.rows > checkpoint_begin
            ? min(t.rows - checkpoint_begin, kCheckpointRows)
            : 0U;
    const bool checkpoint_owner =
        a.saved_x && !t.fc2 && t.column / 2U == t.block % (kFc1Tiles / 2U);
    StaticV4XCheckpoint<kK, kCheckpointRows> checkpoint;
#if WEFTMOE_DEVICE_TIMELINE
    std::uint64_t tma_wait_ns = 0U;
#endif
    if constexpr (FP8) {
      compute_fp8_tile<Network>(a, s, t, accum, stage, phase);
    } else {
    for (unsigned k = 0; k < reduction; k += kK) {
#if WEFTMOE_DEVICE_TIMELINE
      const auto wait_begin =
          threadIdx.x == 0U ? profiling::read_globaltimer_ns() : 0U;
#endif
      s.full[stage].wait(phase);
#if WEFTMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U)
        tma_wait_ns += profiling::read_globaltimer_ns() - wait_begin;
#endif
      if (k == 0U) {
        if (t.fc2) {
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2MmaBegin, Compute, t.column,
                                  t.block, 0U);
        } else {
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1MmaBegin, Compute, t.column,
                                  t.block, 0U);
        }
      }
      const auto astart = abase + stage * (kM * kK / 8U);
      const auto bstart = bbase + stage * (kN * kK / 8U);
#pragma unroll
      for (auto &value : accum)
        cute::warpgroup_fence_operand(value);
      cute::warpgroup_arrive();
#pragma unroll
      for (unsigned atom = 0; atom < kK; atom += Mma::K) {
        da.reg32_[0] = astart + atom / 8U;
        db.reg32_[0] = bstart + atom / 8U;
        Mma::wgmma(da, db, accum, k != 0U || atom != 0U);
      }
      cute::warpgroup_commit_batch();
#pragma unroll
      for (auto &value : accum)
        cute::warpgroup_fence_operand(value);
      if (checkpoint_owner) {
        // The A-ready acquire also publishes Pull's logical-to-saved rows.
        // Reuse this GEMM load; do not add another global read of X.
        if (k == 0U) {
          checkpoint.prepare(a, t.block, checkpoint_rows, checkpoint_begin);
          if constexpr (Network)
            if (threadIdx.x < checkpoint_rows) {
              const auto physical =
                  t.block * kM + checkpoint_begin + threadIdx.x;
              const auto &row = a.ring.rows[physical];
              const auto saved = a.checkpoint.rows[physical];
              a.checkpoint.gates[saved] = row.gate;
              a.checkpoint.headers[saved] = {
                  transport::kActivationTransportSchemaVersion,
                  row.source,
                  a.rank,
                  0U,
                  a.dispatch.identity.generation,
                  execution_epoch(a),
                  row.route,
                  row.route,
                  row.token,
                  saved,
                  a.rank * a.dispatch.sizes.owned_expert_count + t.expert,
                  unsigned(row.route % a.dispatch.identity.top_k)};
            }
        }
        checkpoint.save(s.a[stage] + checkpoint_begin * kK, checkpoint_rows, k,
                        a.hidden, a.saved_x_map);
      }
      const auto lane = threadIdx.x & 31U;
      if constexpr (WaitGroup == 1U) {
        if (k != 0U) {
          cute::warpgroup_wait<1>();
          if (lane < 2U)
            s.empty[retirement].arrive(lane);
          retirement = retirement + 1U == kStages ? 0U : retirement + 1U;
        }
      } else {
        cute::warpgroup_wait<0>();
        if (lane < 2U)
          s.empty[stage].arrive(lane);
      }
      stage = stage + 1U == kStages ? 0U : stage + 1U;
      phase ^= stage == 0U;
    }
    if constexpr (WaitGroup == 1U) {
      cute::warpgroup_wait<0>();
      const auto lane = threadIdx.x & 31U;
      if (lane < 2U)
        s.empty[retirement].arrive(lane);
    }
    }
    if (t.fc2) {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2MathEnd, Compute, t.column,
                              t.block, 0U);
      if constexpr (Network) {
        static_v4_store_fc2_register_tile<kN>(a, accum, t.rows, t.column * kN,
                                              t.block);
      } else
        static_v4_push_fc2_tile<Mma, kN, kM, kMathThreads, true>(
            a, s.output, accum, t.row, t.rows, t.column * kN, t.block);
      // This is an S read-retirement credit, not contribution publication.
      // All TMA/WGMMA reads and the math join have finished. The terminal
      // rank barrier publishes remote stores once, without draining them at
      // every FC2 tile just to recycle the unrelated S input slot.
      if (threadIdx.x == 0U)
        atomicAdd(reinterpret_cast<unsigned long long *>(&slot.fc2_done), 1ULL);
      if constexpr (Network) {
        // Publish rail-visible output separately from X/S retirement. Every
        // remote row holds TX credit until its rail's local NIC completion.
        __threadfence_system();
        collective_scope_sync<kMathThreads, 7U>();
        if (threadIdx.x == 0U) {
          const auto &desc = a.ring.blocks[t.block];
          if constexpr (!WEFTMOE_PF_RETURN_INDEPENDENT_TX) {
            arrive(slot.fc2_published, 1U);
            if (t.column == 0U && desc.local_rows == t.rows)
              arrive(slot.returned, t.rows);
          } else {
            auto &tx_slot = a.ring.return_slots[desc.return_slot];
            if constexpr (WEFTMOE_PF_RETURN_RAILS == 1)
              arrive(tx_slot.published, 1U);
            else
              reduce_add_release_system(
                  reinterpret_cast<unsigned long long *>(&tx_slot.published), 1ULL);
            if (t.column == 0U) {
              if constexpr (WEFTMOE_PF_RETURN_RAILS == 1) {
                // One local owner retires the whole TX epoch. Blocks without
                // NIC readers can retire here; otherwise the Put owner does it.
                if (desc.local_rows == t.rows)
                  ring::publish(tx_slot.consumed, desc.return_generation * kM);
              } else {
                reduce_add_release_system(
                    reinterpret_cast<unsigned long long *>(&tx_slot.consumed),
                    desc.local_rows + kM - t.rows);
                arrive(slot.returned, t.rows);
              }
            }
          }
        }
        collective_scope_sync<kMathThreads, 7U>();
      }
    } else {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1MathEnd, Compute, t.column,
                              t.block, 0U);
      // X and S have independent lifetimes. Pull can recycle X as soon as
      // all FC1 tiles retire, but the new S must wait for the previous FC2.
      while (ring::acquire(slot.fc2_done) <
             std::uint64_t(t.generation - 1U) * kFc2Tiles) {
      }
      if (a.saved_ab)
        static_v4_save_fc1_tile<kN>(a, accum, t.rows, t.column * kN / 2U,
                                    t.block);
      if constexpr (FP8 && kN == 256U) {
        store_fp8_swiglu<forward_tuning::kLocalPrefillFastMath, WEFTMOE_FP8_PACKED_RING>(
            a, accum, output, gates, t.row, t.rows, t.column * kN / 2U, t.block);
      } else
      static_v4_store_fc1_register_vector_tile<
          kN, forward_tuning::kLocalPrefillFastMath>(
          a, accum, output, gates, t.row, t.rows, t.column * kN / 2U);
      // All writers complete their vector stores before S-ready publication.
      __threadfence();
      collective_scope_sync<kMathThreads, 7U>();
      if (threadIdx.x == 0U)
        arrive(slot.fc1_done, 1U);
      collective_scope_sync<kMathThreads, 7U>();
    }
    if (t.fc2) {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2TileEnd, Compute, t.column,
                              t.block, 0U);
    } else {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1TileEnd, Compute, t.column,
                              t.block, 0U);
    }
    if constexpr (!FP8) {
      WEFTMOE_TIMELINE_RECORD(a.timeline, TmaWaitNs, Compute, unsigned(t.fc2),
                              tma_wait_ns, t.block);
    }
  }
  if (threadIdx.x == 0U) {
    const auto now = device_globaltimer_ns();
    if (started_fc1 && !previous_fc2)
      atomicMax(&status->fc1_complete_clock, now);
    if (started_fc2)
      atomicMax(&status->fc2_complete_clock, now);
  }
}

#endif
} // namespace weftmoe_pf::local
