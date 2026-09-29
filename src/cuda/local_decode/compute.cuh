#pragma once

namespace local_decode {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

__device__ __forceinline__ void produce(
    const ForwardKernelArgs &a, Shared &s, const CUtensorMap &x_map,
    const CUtensorMap &s_map, const CUtensorMap &w1_map,
    const CUtensorMap &w2_map) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto tasks = s.blocks[owned] * (kFc1Tiles + kFc2Tiles);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto t = task(s, owned, index);
    const auto *ready = t.fc2 ? &states(a)[t.block].s_ready
                              : &states(a)[t.block].x_ready;
    const auto target = t.fc2 ? kFc1Tiles : t.rows;
    while (static_v4::load_acquire_device(ready) != target) {}
    const auto reduction = t.fc2 ? forward_tuning::kMainIntermediate
                                 : forward_tuning::kMainHidden;
    const auto columns = t.fc2 ? forward_tuning::kMainHidden
                               : 2U * forward_tuning::kMainIntermediate;
    for (unsigned k = 0U; k < reduction; k += kK) {
      s.empty[stage].wait(phase ^ 1U);
      if (cute::elect_one_sync()) {
        hopper::tma_load<kK, kM, 128U>(
            t.fc2 ? &s_map : &x_map, &s.full[stage], s.a[stage],
            k, t.block * kM, 2U);
        hopper::tma_load<kK, kN, 128U>(
            t.fc2 ? &w2_map : &w1_map, &s.full[stage], s.b[stage],
            k, t.expert * columns + t.column * kN);
        s.full[stage].arrive_and_expect_tx((kM + kN) * kK * 2U);
      }
      stage = (stage + 1U) % kStages;
      phase ^= stage == 0U;
    }
  }
}

__device__ __forceinline__ void store_fc1(
    const ForwardKernelArgs &a, Shared &s, const Task &t, const float *accum,
    const CUtensorMap &output_map) {
  using namespace forward_compute_detail;
  const auto lane = threadIdx.x % 32U;
  const auto row0 = (threadIdx.x / 32U) * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
  const auto *rows = a.ring.rows + t.block * kM;
  const auto gate0 = row0 < t.rows ? rows[row0].gate : 0.0F;
  const auto gate1 = row1 < t.rows ? rows[row1].gate : 0.0F;
#pragma unroll
  for (unsigned i = 0; i < kN / 16U; ++i) {
    const auto *g = accum + i * 8U;
    const auto *u = g + 4U;
    store_m128_fragment(
        s.output, i,
        __float22bfloat162_rn(static_v4_swiglu_pair(g, u, gate0)),
        __float22bfloat162_rn(static_v4_swiglu_pair(g + 2U, u + 2U, gate1)));
  }
  cute::tma_store_fence();
  static_v4::collective_scope_sync<kMathThreads, 7U>();
  if (threadIdx.x == 0U) {
    cute::SM90_TMA_STORE_2D::copy(
        &output_map, s.output, t.column * (kN / 2U), t.block * kM);
    cute::tma_store_arrive();
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    static_v4::reduce_add_release_device(&states(a)[t.block].s_ready, 1ULL);
  }
  static_v4::collective_scope_sync<kMathThreads, 7U>();
}

__device__ __forceinline__ void compute(const ForwardKernelArgs &a, Shared &s,
                                        const CUtensorMap &output_map) {
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto tasks = s.blocks[owned] * (kFc1Tiles + kFc2Tiles);
  const auto half = threadIdx.x / 128U;
  const auto lane = threadIdx.x % 32U;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  unsigned stage = 0U, phase = 0U;
  for (unsigned index = blockIdx.x; index < tasks; index += gridDim.x) {
    const auto t = task(s, owned, index);
    if (threadIdx.x == 0U)
      atomicMin(t.fc2 ? &status->dispatch_blocks_done : &status->fc1_start_clock,
                device_globaltimer_ns());
    float accum[Mma::kNumAccum] = {};
    const auto reduction = t.fc2 ? forward_tuning::kMainIntermediate
                                 : forward_tuning::kMainHidden;
    for (unsigned k = 0U; k < reduction; k += kK) {
      s.full[stage].wait(phase);
      auto da = hopper::make_gmma_desc<hopper::Major::K, kM, kK, 128U>(
          s.a[stage], half * 64U, 0U);
      auto db = hopper::make_gmma_desc<hopper::Major::K, kN, kK, 128U>(
          s.b[stage], 0U, 0U);
      const auto abase = da.reg32_[0], bbase = db.reg32_[0];
#pragma unroll
      for (unsigned i = 0U; i < Mma::kNumAccum; ++i)
        cute::warpgroup_fence_operand(accum[i]);
      cute::warpgroup_arrive();
#pragma unroll
      for (unsigned atom = 0U; atom < kK / Mma::K; ++atom) {
        da.reg32_[0] = abase + atom * Mma::K * 2U / 16U;
        db.reg32_[0] = bbase + atom * Mma::K * 2U / 16U;
        Mma::wgmma(da, db, accum, k != 0U || atom != 0U);
      }
      cute::warpgroup_commit_batch();
#pragma unroll
      for (unsigned i = 0U; i < Mma::kNumAccum; ++i)
        cute::warpgroup_fence_operand(accum[i]);
      cute::warpgroup_wait<0>();
      if (lane < 2U)
        s.empty[stage].arrive(lane);
      stage = (stage + 1U) % kStages;
      phase ^= stage == 0U;
    }
    if (t.fc2) {
      if (threadIdx.x == 0U)
        atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
      forward_compute_detail::static_v4_push_fc2_tile<Mma, false, kN, true>(
          a, s.output, accum, t.block * kM, t.rows, t.column * kN, t.block);
      if (threadIdx.x == 0U)
        atomicMax(&status->fc2_complete_clock, device_globaltimer_ns());
    } else {
      store_fc1(a, s, t, accum, output_map);
      if (threadIdx.x == 0U)
        atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
    }
  }
}

#endif
} // namespace local_decode
