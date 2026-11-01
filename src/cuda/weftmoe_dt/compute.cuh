#pragma once

#include "cuda/weftmoe_dt/schedule.cuh"

namespace weftmoe_dt {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

__device__ __forceinline__ void produce(const ForwardKernelArgs &a, Shared<> &s,
                                        const CUtensorMap &x_map,
                                        const CUtensorMap &s_map,
                                        const CUtensorMap &w1_map,
                                        const CUtensorMap &w2_map) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto tasks = compute_units(s, owned) * (kFc1Tiles + kFc2Tiles);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto block = task(s, owned, index);
    const auto *ready = block.fc2 ? &states(a)[block.block].s_ready
                                  : &states(a)[block.block].x_ready;
    // Each x_ready counts unordered arrivals for a whole physical M128 block.
    // Every FC2 slice must observe completion of all FC1 slices in that block.
    const auto physical_rows = ready_rows(s, owned, block);
    const auto target = block.fc2 ? ((physical_rows + kM - 1U) / kM) * kFc1Tiles
                                  : physical_rows;
    const auto reduction = block.fc2 ? forward_tuning::kMainIntermediate
                                     : forward_tuning::kMainHidden;
    for (unsigned offset = 0U; offset < block.rows; offset += kM) {
      const auto t = slice(block, offset);
      for (unsigned k = 0U; k < reduction; k += kK) {
        s.empty[stage].wait(phase ^ 1U);
        if (cute::elect_one_sync()) {
          s.full[stage].expect_transaction((kM + kN) * kK * 2U);
          if (!t.fc2)
            cute::SM90_TMA_LOAD_5D::copy(
                &w1_map, reinterpret_cast<std::uint64_t *>(&s.full[stage]),
                static_cast<std::uint64_t>(
                    cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.b[stage], k, 0, 0, t.column * (kN / 16U), t.expert);
          else
            hopper::tma_load<kK, kN, 128U>(
                &w2_map, &s.full[stage], s.b[stage], k,
                t.expert * forward_tuning::kMainHidden + t.column * kN);
          if (offset == 0U && k == 0U)
            while (weftmoe_pf::load_acquire_device(ready) != target) {
            }
          hopper::tma_load<kK, kM, 128U>(t.fc2 ? &s_map : &x_map,
                                         &s.full[stage], s.a[stage], k, t.row);
          s.full[stage].arrive();
        }
        stage = (stage + 1U) % kStages;
        phase ^= stage == 0U;
      }
    }
  }
}

__device__ __forceinline__ void store_fc1(const ForwardKernelArgs &a,
                                          const Task &t, const float *accum) {
  using namespace forward_compute_detail;
  const auto lane = threadIdx.x % 32U;
  const auto warp = threadIdx.x / 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
  const auto *rows = a.ring.rows + t.row;
  auto *output = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  for (unsigned item = 0U; item < 2U; ++item) {
    const auto row = item ? row1 : row0;
    if (row >= t.rows)
      continue;
    const auto gate = rows[row].gate;
#pragma unroll
    for (unsigned i = 0; i < kN / 16U; ++i) {
      const auto *g = accum + i * 8U + item * 2U;
      const auto column = t.column * (kN / 2U) + i * 8U + (lane % 4U) * 2U;
      *reinterpret_cast<__nv_bfloat162 *>(
          output + (t.row + row) * a.intermediate + column) =
          __float22bfloat162_rn(
              static_v4_swiglu_pair<forward_tuning::kLocalPrefillFastMath>(
                  g, g + 4U, gate));
    }
  }
  __threadfence();
  weftmoe_pf::collective_scope_sync<kMathThreads, 7U>();
  if (threadIdx.x == 0U) {
    weftmoe_pf::reduce_add_release_device(&states(a)[t.block].s_ready, 1ULL);
  }
  weftmoe_pf::collective_scope_sync<kMathThreads, 7U>();
}

template <bool FP8, unsigned N = kN>
__device__ __forceinline__ void store_fc2(const ForwardKernelArgs &a,
                                          Shared<FP8> &s, const Task &t,
                                          const float *accum, unsigned offset) {
  using namespace forward_compute_detail;
  if constexpr (FP8) {
    // Each warpgroup owns one N128 half; keep the common N256 return layout.
    const auto wg = threadIdx.x / 128U;
    const auto warp = (threadIdx.x / 32U) % 4U;
    const auto lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned i = 0; i < 16U; ++i) {
      const auto atom = i / 8U;
      const auto col = (i % 8U) ^ (lane % 8U);
      auto *target = reinterpret_cast<unsigned char *>(s.output) +
                     wg * kM * 128U * 2U + warp * 16U * 128U +
                     atom * kM * 128U + lane * 128U + col * 16U;
      hopper::StoreMatrixX2<__nv_bfloat162>::copy(
          __float22bfloat162_rn(make_float2(accum[i * 4U], accum[i * 4U + 1U])),
          __float22bfloat162_rn(
              make_float2(accum[i * 4U + 2U], accum[i * 4U + 3U])),
          target);
    }
  } else {
#pragma unroll
    for (unsigned i = 0U; i < (N / 2U) / 4U; ++i)
      store_m_fragment<kM>(
          s.output, i,
          __float22bfloat162_rn(make_float2(accum[i * 4U], accum[i * 4U + 1U])),
          __float22bfloat162_rn(
              make_float2(accum[i * 4U + 2U], accum[i * 4U + 3U])));
  }
  static_v4_copy_fc2_tile<N, kM, kComputeThreads<FP8>>(
      a, s.output, t.row, t.rows, t.column * N, t.block, offset);
}

__device__ __forceinline__ void compute(const ForwardKernelArgs &a,
                                        Shared<> &s) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto fc1_tasks = compute_units(s, owned) * kFc1Tiles;
  const auto tasks = compute_units(s, owned) * (kFc1Tiles + kFc2Tiles);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto block = task(s, owned, index);
    const auto *ready = block.fc2 ? &states(a)[block.block].s_ready
                                  : &states(a)[block.block].x_ready;
    // Each x_ready counts unordered arrivals for a whole physical M128 block.
    // Every FC2 slice must observe completion of all FC1 slices in that block.
    const auto physical_rows = ready_rows(s, owned, block);
    const auto target = block.fc2 ? ((physical_rows + kM - 1U) / kM) * kFc1Tiles
                                  : physical_rows;
    while (weftmoe_pf::load_acquire_device(ready) != target) {
    }
    // Phase extrema need only this CTA's first/last task, not a contended
    // global atomic at every tile. The static sequence puts FC1 before FC2.
    const bool first = index < (block.fc2 ? fc1_tasks : 0U) + gridDim.x;
    const bool last = index + gridDim.x >= (block.fc2 ? tasks : fc1_tasks);
    if (threadIdx.x == 0U && first)
      atomicMin(block.fc2 ? &status->dispatch_blocks_done
                          : &status->fc1_start_clock,
                device_globaltimer_ns());
    for (unsigned offset = 0U; offset < block.rows; offset += kM) {
      const auto t = slice(block, offset);
      float accum[GroupMma::kNumAccum] = {};
      const auto reduction = t.fc2 ? forward_tuning::kMainIntermediate
                                   : forward_tuning::kMainHidden;
      unsigned retirement = stage;
      for (unsigned k = 0U; k < reduction; k += kK) {
        s.full[stage].wait(phase);
        auto da = hopper::make_gmma_desc<hopper::Major::K, kM, kK, 128U>(
            s.a[stage], 0U, 0U);
        auto db = hopper::make_gmma_desc<hopper::Major::K, kN, kK, 128U>(
            s.b[stage], 0U, 0U);
        const auto abase = da.reg32_[0], bbase = db.reg32_[0];
#pragma unroll
        for (auto &value : accum)
          cute::warpgroup_fence_operand(value);
        cute::warpgroup_arrive();
#pragma unroll
        for (unsigned atom = 0U; atom < kK / GroupMma::K; ++atom) {
          da.reg32_[0] = abase + atom * GroupMma::K * 2U / 16U;
          db.reg32_[0] = bbase + atom * GroupMma::K * 2U / 16U;
          GroupMma::wgmma(da, db, accum, k != 0U || atom != 0U);
        }
        cute::warpgroup_commit_batch();
#pragma unroll
        for (auto &value : accum)
          cute::warpgroup_fence_operand(value);
        if (k != 0U) {
          cute::warpgroup_wait<1>();
          if (threadIdx.x == 0U)
            s.empty[retirement].arrive();
          retirement = (retirement + 1U) % kStages;
        }

        stage = (stage + 1U) % kStages;
        phase ^= stage == 0U;
      }
      cute::warpgroup_wait<0>();
      if (threadIdx.x == 0U)
        s.empty[retirement].arrive();
      if (t.fc2) {
        if (threadIdx.x == 0U && first && offset == 0U)
          atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
        store_fc2(a, s, t, accum,
                  offset + (kSliceSchedule<> ? block.row - block.block * kBlockM
                                             : 0U));
      } else {
        store_fc1(a, t, accum);
      }
    }
    if (threadIdx.x == 0U && last)
      atomicMax(block.fc2 ? &status->fc2_complete_clock
                          : &status->fc1_complete_clock,
                device_globaltimer_ns());
  }
}

#endif
} // namespace weftmoe_dt
