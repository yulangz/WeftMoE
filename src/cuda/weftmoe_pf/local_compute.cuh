#pragma once

#include "cuda/weftmoe_pf/local_schedule.cuh"

namespace weftmoe_pf::local {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

template <class Scheduler = Schedule, bool Network = false>
__device__ __forceinline__ void produce(const ForwardKernelArgs &a, Shared &s,
                                        const CUtensorMap &x_map,
                                        const CUtensorMap &s_map,
                                        const CUtensorMap &w1_map,
                                        const CUtensorMap &w2_map) {
  Scheduler schedule(a);
  Task t;
  unsigned stage = 0U, phase = 0U;
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
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, InputWaitEnd, Compute,
          unsigned(t.fc2), t.block, t.column);
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __syncwarp();
    const auto reduction =
        t.fc2 ? forward_tuning::kMainIntermediate : forward_tuning::kMainHidden;
    const auto columns = t.fc2 ? forward_tuning::kMainHidden
                               : 2U * forward_tuning::kMainIntermediate;
    for (unsigned k = 0; k < reduction; k += kK) {
      s.empty[stage].wait(phase ^ 1U);
      if (cute::elect_one_sync()) {
        hopper::tma_load<kK, kM, 128U>(t.fc2 ? &s_map : &x_map, &s.full[stage],
                                       s.a[stage], k, t.row, 2U);
        hopper::tma_load<kK, kN, 128U>(t.fc2 ? &w2_map : &w1_map,
                                       &s.full[stage], s.b[stage], k,
                                       t.expert * columns + t.column * kN);
        s.full[stage].arrive_and_expect_tx((kM + kN) * kK * 2U);
      }
      stage = stage + 1U == kStages ? 0U : stage + 1U;
      phase ^= stage == 0U;
    }
  }
}

template <class Scheduler = Schedule, bool Network = false, class Matrix = Mma,
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
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2TileStart, Compute, t.column, t.block, 0U);
    } else {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1TileStart, Compute, t.column, t.block, 0U);
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
    const auto checkpoint_rows = t.rows > checkpoint_begin
                                     ? min(t.rows - checkpoint_begin,
                                           kCheckpointRows)
                                     : 0U;
    const bool checkpoint_owner =
        a.saved_x && !t.fc2 && t.column / 2U == t.block % (kFc1Tiles / 2U);
    StaticV4XCheckpoint<kK, false, kCheckpointRows> checkpoint;
#if WEFTMOE_DEVICE_TIMELINE
    std::uint64_t tma_wait_ns = 0U;
#endif
    for (unsigned k = 0; k < reduction; k += kK) {
#if WEFTMOE_DEVICE_TIMELINE
      const auto wait_begin = threadIdx.x == 0U ? profiling::read_globaltimer_ns() : 0U;
#endif
      s.full[stage].wait(phase);
#if WEFTMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U)
        tma_wait_ns += profiling::read_globaltimer_ns() - wait_begin;
#endif
      if (k == 0U) {
        if (t.fc2) {
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2MmaBegin, Compute, t.column, t.block, 0U);
        } else {
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1MmaBegin, Compute, t.column, t.block, 0U);
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
              const auto physical = t.block * kM + checkpoint_begin + threadIdx.x;
              const auto &row = a.ring.rows[physical];
              const auto saved = a.checkpoint.rows[physical];
              a.checkpoint.gates[saved] = row.gate;
              a.checkpoint.headers[saved] = {
                  transport::kActivationTransportSchemaVersion, row.source,
                  a.rank, 0U, a.dispatch.identity.generation, execution_epoch(a),
                  row.route, row.route, row.token, saved,
                  a.rank * a.dispatch.sizes.owned_expert_count + t.expert,
                  unsigned(row.route % a.dispatch.identity.top_k)};
            }
        }
        checkpoint.save(s.a[stage] + checkpoint_begin * kK, checkpoint_rows,
                        k, a.hidden, a.saved_x_map);
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
    if (t.fc2) {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2MathEnd, Compute, t.column, t.block, 0U);
      if constexpr (Network && WEFTMOE_PF_FC2_REGISTER_STORE)
        static_v4_store_fc2_register_tile<kN>(a, accum, t.rows, t.column * kN, t.block);
      else
        static_v4_push_fc2_tile<Mma, false, kN, true, kM, kMathThreads, true>(
            a, s.output, accum, t.row, t.rows, t.column * kN, t.block);
      // This is an S read-retirement credit, not contribution publication.
      // All TMA/WGMMA reads and the math join have finished. The terminal
      // rank barrier publishes remote stores once, without draining them at
      // every FC2 tile just to recycle the unrelated S input slot.
      if (threadIdx.x == 0U)
        atomicAdd(reinterpret_cast<unsigned long long *>(&slot.fc2_done), 1ULL);
      if constexpr (Network) {
        // Publish NIC-readable output separately from the S input credit.
        __threadfence_system();
        collective_scope_sync<kMathThreads, 7U>();
        if (threadIdx.x == 0U)
          arrive(slot.fc2_published, 1U);
        collective_scope_sync<kMathThreads, 7U>();
      }
    } else {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1MathEnd, Compute, t.column, t.block, 0U);
      // X and S have independent lifetimes. Pull can recycle X as soon as
      // all FC1 tiles retire, but the new S must wait for the previous FC2.
      while (ring::acquire(slot.fc2_done) <
             std::uint64_t(t.generation - 1U) * kFc2Tiles) {
      }
      if (a.saved_ab)
        static_v4_save_fc1_tile<kN>(a, accum, t.rows, t.column * kN / 2U,
                                  t.block);
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
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2TileEnd, Compute, t.column, t.block, 0U);
    } else {
      WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1TileEnd, Compute, t.column, t.block, 0U);
    }
    WEFTMOE_TIMELINE_RECORD(a.timeline, TmaWaitNs, Compute,
        unsigned(t.fc2), tma_wait_ns, t.block);
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
