#pragma once

#include "cuda/weftmoe_dt/schedule.cuh"

namespace weftmoe_dt {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

template <unsigned M>
__device__ __forceinline__ void
produce(const ForwardKernelArgs &a, Shared<M> &s, const CUtensorMap &x_map,
        const CUtensorMap &s_map, const CUtensorMap &w1_map,
        const CUtensorMap &w2_map) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto tasks = s.blocks[owned] * (kFc1Tiles + kFc2Tiles);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto block = task(s, owned, index);
    const auto *ready = block.fc2 ? &states(a)[block.block].s_ready
                                  : &states(a)[block.block].x_ready;
    const auto target =
        block.fc2 ? ((block.rows + M - 1U) / M) * kFc1Tiles : block.rows;
    const auto reduction = block.fc2 ? forward_tuning::kMainIntermediate
                                     : forward_tuning::kMainHidden;
    const auto columns = block.fc2 ? forward_tuning::kMainHidden
                                   : 2U * forward_tuning::kMainIntermediate;
    for (unsigned offset = 0U; offset < block.rows; offset += M) {
      const auto t = slice<M>(block, offset);
      for (unsigned k = 0U; k < reduction; k += kK) {
        s.empty[stage].wait(phase ^ 1U);
        if (cute::elect_one_sync()) {
          s.full[stage].expect_transaction((M + kN) * kK * 2U);
          hopper::tma_load<kK, kN, 128U>(t.fc2 ? &w2_map : &w1_map,
                                         &s.full[stage], s.b[stage], k,
                                         t.expert * columns + t.column * kN);
          if (offset == 0U && k == 0U)
            while (weftmoe_pf::load_acquire_device(ready) != target) {
            }
          hopper::tma_load<kK, M, 128U>(t.fc2 ? &s_map : &x_map, &s.full[stage],
                                        s.a[stage], k, t.row);
          s.full[stage].arrive();
        }
        stage = (stage + 1U) % kStages;
        phase ^= stage == 0U;
      }
    }
  }
}

template <unsigned M>
__device__ __forceinline__ void store_fc1(const ForwardKernelArgs &a,
                                          const Task &t, const float *accum) {
  using namespace forward_compute_detail;
  const auto lane = threadIdx.x % 32U;
  const auto warp = threadIdx.x / 32U;
  const auto row0 = (M == 16U ? 0U : warp * 16U) + lane / 4U;
  const auto row1 = row0 + 8U;
  const auto *rows = a.ring.rows + t.row;
  auto *output = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  for (unsigned item = 0U; item < 2U; ++item) {
    const auto row = item ? row1 : row0;
    if (row >= t.rows)
      continue;
    const auto gate = rows[row].gate;
#pragma unroll
    for (unsigned i = 0; i < (M == 16U ? kWarpN : kN) / 16U; ++i) {
      const auto *g = accum + i * 8U + item * 2U;
      const auto column = t.column * (kN / 2U) +
                          (M == 16U ? warp * (kWarpN / 2U) : 0U) + i * 8U +
                          (lane % 4U) * 2U;
      *reinterpret_cast<__nv_bfloat162 *>(
          output + (t.row + row) * a.intermediate + column) =
          __float22bfloat162_rn(static_v4_swiglu_pair(g, g + 4U, gate));
    }
  }
  __threadfence();
  weftmoe_pf::collective_scope_sync<kMathThreads, 7U>();
  if (threadIdx.x == 0U) {
    weftmoe_pf::reduce_add_release_device(&states(a)[t.block].s_ready, 1ULL);
  }
  weftmoe_pf::collective_scope_sync<kMathThreads, 7U>();
}

template <unsigned M>
__device__ __forceinline__ void store_fc2(const ForwardKernelArgs &a,
                                          Shared<M> &s, const Task &t,
                                          const float *accum, unsigned offset) {
  using namespace forward_compute_detail;
  if constexpr (M == 16U) {
    const auto lane = threadIdx.x % 32U, warp = threadIdx.x / 32U;
#pragma unroll
    for (unsigned i = 0U; i < kWarpN / 8U; ++i) {
      const auto column = warp * kWarpN + i * 8U + (lane % 4U) * 2U;
#pragma unroll
      for (unsigned item = 0U; item < 2U; ++item) {
        const auto row = lane / 4U + item * 8U;
        const auto pos = m_fragment_vector_offset<M>(row, column);
        *reinterpret_cast<__nv_bfloat162 *>(s.output + pos) =
            __float22bfloat162_rn(make_float2(accum[i * 4U + item * 2U],
                                              accum[i * 4U + item * 2U + 1U]));
      }
    }
  } else {
#pragma unroll
    for (unsigned i = 0U; i < GroupMma::kNumAccum / 4U; ++i)
      store_m_fragment<M>(
          s.output, i,
          __float22bfloat162_rn(make_float2(accum[i * 4U], accum[i * 4U + 1U])),
          __float22bfloat162_rn(
              make_float2(accum[i * 4U + 2U], accum[i * 4U + 3U])));
  }
  static_v4_copy_fc2_tile<kN, M, kMathThreads>(a, s.output, t.row, t.rows,
                                               t.column * kN, t.block, offset);
}

template <unsigned M>
__device__ __forceinline__ void compute(const ForwardKernelArgs &a,
                                        Shared<M> &s) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto fc1_tasks = s.blocks[owned] * kFc1Tiles;
  const auto tasks = s.blocks[owned] * (kFc1Tiles + kFc2Tiles);
  const auto lane = threadIdx.x % 32U;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto block = task(s, owned, index);
    const auto *ready = block.fc2 ? &states(a)[block.block].s_ready
                                  : &states(a)[block.block].x_ready;
    const auto target =
        block.fc2 ? ((block.rows + M - 1U) / M) * kFc1Tiles : block.rows;
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
    for (unsigned offset = 0U; offset < block.rows; offset += M) {
      const auto t = slice<M>(block, offset);
      float accum[M == 16U ? kWarpN / 2U : GroupMma::kNumAccum] = {};
      const auto reduction = t.fc2 ? forward_tuning::kMainIntermediate
                                   : forward_tuning::kMainHidden;
      const auto warp = threadIdx.x / 32U;
      unsigned retirement = stage;
      for (unsigned k = 0U; k < reduction; k += kK) {
        s.full[stage].wait(phase);
        if constexpr (M == 16U) {
#pragma unroll
          for (unsigned atom = 0U; atom < kK / 16U; ++atom) {
            const auto m = lane % 16U;
            const auto acol = atom * 16U + (lane / 16U) * 8U;
            const auto apos = m * kK + (acol ^ ((m % 8U) * 8U));
            uint32_t ar[4];
            cute::SM75_U32x4_LDSM_N::copy(
                *reinterpret_cast<const cute::uint128_t *>(s.a[stage] + apos),
                ar[0], ar[1], ar[2], ar[3]);
#pragma unroll
            for (unsigned i = 0U; i < kWarpN / 8U; ++i) {
              const auto n = warp * kWarpN + i * 8U + lane % 8U;
              const auto col = atom * 16U + ((lane / 8U) % 2U) * 8U;
              const auto pos = n * kK + (col ^ ((n % 8U) * 8U));
              uint32_t b0, b1;
              cute::SM75_U32x2_LDSM_N::copy(
                  *reinterpret_cast<const cute::uint128_t *>(s.b[stage] + pos),
                  b0, b1);
              auto *d = accum + i * 4U;
              WarpMma::fma(d[0], d[1], d[2], d[3], ar[0], ar[1], ar[2], ar[3],
                           b0, b1, d[0], d[1], d[2], d[3]);
            }
          }
          weftmoe_pf::collective_scope_sync<kMathThreads, 7U>();
          if (threadIdx.x == 0U)
            s.empty[stage].arrive();
        } else {
          auto da = hopper::make_gmma_desc<hopper::Major::K, M, kK, 128U>(
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
        }
        stage = (stage + 1U) % kStages;
        phase ^= stage == 0U;
      }
      if constexpr (M != 16U) {
        cute::warpgroup_wait<0>();
        if (threadIdx.x == 0U)
          s.empty[retirement].arrive();
      }
      if (t.fc2) {
        if (threadIdx.x == 0U && first && offset == 0U)
          atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
        store_fc2(a, s, t, accum, offset);
      } else {
        store_fc1<M>(a, t, accum);
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
