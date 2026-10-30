#pragma once

namespace weftmoe_pf::backward_stream {

constexpr unsigned kM = 128U, kK = backward_tuning::kActivationTileK;
static_assert(kK == 64U || kK == 128U);
constexpr unsigned kStages = backward_tuning::kDabStages;
constexpr unsigned kDuN = backward_tuning::kDabTileN;
constexpr unsigned kDxN = backward_tuning::kDxTileN;
constexpr unsigned kN = kDuN > kDxN ? kDuN : kDxN;
constexpr unsigned kDuTiles = backward_tuning::kMainIntermediate / kDuN;
constexpr unsigned kDxTiles = backward_tuning::kMainHidden / kDxN;
#ifndef WEFTMOE_PF_BWD_NETWORK_MATH_REGISTERS
// GDAKI-only code keeps the activation cursors in registers at 192/120.
#define WEFTMOE_PF_BWD_NETWORK_MATH_REGISTERS 192
#endif
#ifndef WEFTMOE_PF_BWD_WAVE_BLOCKS
#define WEFTMOE_PF_BWD_WAVE_BLOCKS 64
#endif
#ifndef WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS
#define WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS 128
#endif
#ifndef WEFTMOE_FP8_BWD_WAVE_BLOCKS
#define WEFTMOE_FP8_BWD_WAVE_BLOCKS 0
#endif
constexpr unsigned kNetworkMathRegisters =
    WEFTMOE_PF_BWD_NETWORK_MATH_REGISTERS;
// The CTA initially owns 384 * 168 registers, not the entire 64K SM file.
constexpr unsigned kNetworkControlRegisters = 504U - 2U * kNetworkMathRegisters;
static_assert(kNetworkMathRegisters >= 168U && kNetworkMathRegisters <= 240U &&
              kNetworkMathRegisters % 8U == 0U);
constexpr bool kEnabled = backward_tuning::kActivationClusterN == 2U &&
                          backward_tuning::kActivationTileK == kK &&
                          backward_tuning::kUnifiedWorkers &&
                          backward_tuning::kUnifiedThreads == 384U &&
                          backward_tuning::kUnifiedCommBase == 288U &&
                          backward_tuning::kUnifiedCommThreads == 96U;

// Independent producer/math streams share only operand-stage barriers. The
// activation epilogue never aliases these stages; dW reuses the arena after
// the entire activation grid has drained.
struct Shared {
  alignas(1024) transport::BFloat16 a[kStages][kM * kK];
  alignas(1024) transport::BFloat16 b[kStages][kN * kK];
  alignas(1024) unsigned char pull[local::kPullWarps][4096];
  local::Barrier full[kStages], empty[kStages];
  local::Barrier pull_full[local::kPullWarps];
};
constexpr auto kDwBytes = backward_compute_detail::dynamic_shared_bytes<
    backward_weights::kStorageTileN>();
constexpr auto kSharedBytes = sizeof(Shared) > kDwBytes ? sizeof(Shared)
                                                        : kDwBytes;

constexpr auto kFP8ScaleBytes = kStages * (kM + kN) * sizeof(float);
// FP8 Wgrad has its own smaller pipeline; reserving the BF16 arena here
// wastes enough SMEM to prevent the concurrent GIN workers from launching.
constexpr auto kFP8BaseSharedBytes = backward_weights::kFP8PipelineBytes > sizeof(Shared) + kFP8ScaleBytes
    ? backward_weights::kFP8PipelineBytes : sizeof(Shared) + kFP8ScaleBytes;
constexpr auto kFP8SharedBytes = kFP8BaseSharedBytes +
    (WEFTMOE_FP8_STREAMED_WGRAD ? 128U * 132U : 0U);
__device__ __forceinline__ float *fp8_scales(Shared &s, unsigned stage) {
  return reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(&s) + sizeof(Shared)) + stage * (kM + kN);
}

enum class Phase : unsigned { Du, Swiglu, Dx };
struct Task {
  unsigned expert, block, column, rows, row, generation;
  Phase phase;
};


template <bool FP8 = false>
struct Schedule {
  const ForwardKernelArgs &a;
  unsigned expert = 0U, wave_expert = 0U, begin = 0U, end = 0U, index = blockIdx.x;
  const unsigned swiglu_tiles, wave_blocks;

  __device__ Schedule(const ForwardKernelArgs &args, const BackwardKernelArgs &b)
      : a(args), swiglu_tiles(b.fp8_x_data ? unsigned(b.intermediate / 128U) : 2U),
        // Streamed SwiGLU adds feature tasks. Shorter waves reduce the Du/Dx
        // critical path for larger workloads; keep small workloads together.
        wave_blocks(b.fp8_x_data && args.ring.state->blocks > WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS
            ? (WEFTMOE_FP8_BWD_WAVE_BLOCKS ? unsigned(WEFTMOE_FP8_BWD_WAVE_BLOCKS)
                : max(1U, 128U / kDuTiles))
            : unsigned(WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS)) {}

  __device__ __forceinline__ bool next(Task &t) {
    const auto all_blocks = unsigned(a.ring.state->blocks);
    while (begin < all_blocks) {
      if (end == begin) {
        end = min(all_blocks,
                  begin + min(a.ring.depth, wave_blocks));
        // Du, SwiGLU and Dx revisit this wave. Keep its first expert so a
        // phase change never scans the entire expert directory from zero.
        if constexpr (FP8)
          while (begin >= a.ring.expert_blocks[wave_expert + 1U])
            ++wave_expert;
      }
      const auto blocks = end - begin;
      const auto du = blocks * kDuTiles;
      const auto epilogue = blocks * swiglu_tiles;
      const auto total = du + epilogue + blocks * kDxTiles;
      if (index < total) {
        const auto phase = index < du ? Phase::Du
                           : index < du + epilogue ? Phase::Swiglu : Phase::Dx;
        const auto linear = index - (phase == Phase::Du ? 0U
                                     : phase == Phase::Swiglu ? du
                                     : du + epilogue);
        const auto columns = phase == Phase::Du   ? kDuTiles
                             : phase == Phase::Dx ? kDxTiles
                             : swiglu_tiles;
        const auto block = begin + linear / columns;
        if (block < a.ring.expert_blocks[expert])
          expert = FP8 ? wave_expert : 0U;
        while (block >= a.ring.expert_blocks[expert + 1U])
          ++expert;
        const auto expert_begin = unsigned(a.ring.expert_blocks[expert]);
        t = {expert,
             block,
             linear % columns,
             min(kM, unsigned(a.ring.expert_rows[expert]) -
                         (block - expert_begin) * kM),
             block % a.ring.depth * kM,
             block / a.ring.depth + 1U,
             phase};
        index += gridDim.x;
        return true;
      }
      index -= total;
      begin = end;
    }
    return false;
  }
};

template <bool Network = false>
__global__ void prepare(ForwardKernelArgs f, BackwardKernelArgs a) {
  const auto expert = blockIdx.y;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;
  auto *words = backward_pipeline_detail::scheduler_words(a);
  const auto layout = pipeline::backward::make_backward_scheduler_layout(
      a.scheduler_mblock_capacity,
      kDxTiles / backward_tuning::kActivationClusterN,
      a.scheduler_expert_capacity);
  const auto rows = a.dispatch.expert_row_count[expert];
  const auto padded =
      Network ? peer_at<const unsigned long long>(
                    a.local_base, a.padded_expert_offsets_offset)[expert]
              : f.ring.expert_blocks[expert] * kM;
  if (!tid) {
    if constexpr (!Network) {
      peer_at<unsigned long long>(
          a.local_base, a.padded_expert_offsets_offset)[expert] = padded;
      peer_at<unsigned long long>(a.local_base,
                                  a.padded_expert_rows_offset)[expert] =
          (f.ring.expert_blocks[expert + 1U] - f.ring.expert_blocks[expert]) *
          kM;
      if (expert + 1U == a.dispatch.sizes.owned_expert_count)
        peer_at<unsigned long long>(
            a.local_base, a.padded_expert_offsets_offset)[expert + 1U] =
            f.ring.expert_blocks[expert + 1U] * kM;
    }
    words[layout.dw_reduction_rows + expert] =
        (rows + backward_tuning::kTileK - 1U) / backward_tuning::kTileK *
        backward_tuning::kTileK;
  }
  auto *mapping = peer_at<unsigned>(a.local_base, a.compact_to_padded_offset);
  for (auto row = std::uint64_t(tid); row < rows; row += stride)
    mapping[a.dispatch.expert_row_prefix[expert] + row] =
        unsigned(padded + row);
  if constexpr (Network) {
    // Initialize canonical cache padding before the dW phase.
    const auto tail = (kM - rows % kM) % kM;
    for (auto v = std::uint64_t(tid); !a.fp8 && v < tail * a.intermediate; v += stride) {
      reinterpret_cast<unsigned short *>(
          a.recomputed_u)[(padded + rows) * a.intermediate + v] = 0U;
      auto *dab = reinterpret_cast<unsigned short *>(a.dab) +
                  (padded + rows) * 2U * a.intermediate;
      dab[v] = dab[tail * a.intermediate + v] = 0U;
    }
  }
  if (!rows) {
    const auto elements = a.hidden * a.intermediate;
    if (a.fp32_weight_gradients) {
      auto *dw1 = reinterpret_cast<float *>(a.dw1) + expert * 2U * elements;
      auto *dw2 = reinterpret_cast<float *>(a.dw2) + expert * elements;
      for (auto i = std::uint64_t(tid); i < 2U * elements; i += stride) {
        dw1[i] = 0.0F;
        if (i < elements)
          dw2[i] = 0.0F;
      }
    } else {
    auto *dw1 =
        reinterpret_cast<unsigned short *>(a.dw1 + expert * 2U * elements);
    auto *dw2 = reinterpret_cast<unsigned short *>(a.dw2 + expert * elements);
    for (auto i = std::uint64_t(tid); i < 2U * elements; i += stride) {
      dw1[i] = 0U;
      if (i < elements)
        dw2[i] = 0U;
    }
    }
  }
  if (!expert) {
    if (!tid && a.fp8_wgrad_ready)
      *a.fp8_wgrad_ready = 0;
    if constexpr (!Network)
      for (auto i = tid; i < f.ring.depth; i += stride)
        f.ring.slots[i] = {};
    if (!tid) {
      if (a.runtime_epoch)
        ++*a.runtime_epoch;
      words[layout.completed_dw1_tiles] = words[layout.completed_dw2_tiles] =
          0ULL;
      words[layout.completed_dx_tiles] = 0ULL;
      // FP8 always claims independent CTA tasks, even in BF16 static builds.
      words[layout.dw2_head] = 0ULL;
      backward_pipeline_detail::initialize_status(a);
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      status->before_combine_complete_clock = 0ULL;
      status->fc1_tasks_done = status->fc2_tasks_done = 0ULL;
      status->synthetic_ticks = status->dispatch_put_ticks =
          status->relay_ticks = 0ULL;
#endif
    }
  }
}

} // namespace weftmoe_pf::backward_stream

#include "cuda/weftmoe_pf/backward_network.cuh"
#if defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
#include "../../../benchmarks/pf_backward_identity.cuh"
#endif

namespace weftmoe_pf::backward_stream {

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
#include "cuda/weftmoe_pf/fp8_swiglu.cuh"
template <bool Network, bool FP8>
__device__ __forceinline__ void
produce(const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s,
        const CUtensorMap &dy, const CUtensorMap &dab, const CUtensorMap &w2,
        const CUtensorMap &w1) {
  using Scheduler = std::conditional_t<Network, NetworkSchedule, Schedule<FP8>>;
  Scheduler schedule(f, a);
  Task t;
  unsigned stage = 0U, phase = 0U;
  while (schedule.next(t)) {
    if (t.phase == Phase::Swiglu)
      continue;
    const bool dx = t.phase == Phase::Dx;
    auto &slot = f.ring.slots[t.row / kM];
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
    const auto ready_begin = device_globaltimer_ns();
#endif
    if (cute::elect_one_sync()) {
      if constexpr (Network) {
        if (!dx) {
          const auto *ready = backward_pipeline_detail::scheduler_words(a) +
                              directory_layout(a).dispatch_arrived;
          while (ring::acquire(*reinterpret_cast<const std::uint64_t *>(
                     ready + t.block)) < t.rows) {
          }
        } else {
          while (ring::acquire(slot.activation_ready) <
                     std::uint64_t(t.generation) * 2U ||
                 ring::acquire(slot.returned) < t.generation - 1U) {
          }
        }
      } else {
        const auto &ready = dx ? slot.activation_ready : slot.x_full;
        while (ring::acquire(ready) <
               std::uint64_t(t.generation) * (dx ? 2U : kM)) {
        }
      }
      if (!dx)
        while (ring::acquire(slot.fc2_done) <
               std::uint64_t(t.generation - 1U) * kDxTiles) {
        }
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __syncwarp();
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
    if (threadIdx.x == 256U) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(dx ? &status->relay_ticks : &status->dispatch_put_ticks,
                device_globaltimer_ns() - ready_begin);
    }
#endif
    const auto reduction = dx ? 2U * f.intermediate : f.hidden;
    const auto input_row =
        Network ? (FP8 ? (dx ? t.row : t.block * kM)
                       : (dx ? t.block * kM : directory(a)[t.block].compact_row)) : t.row;
    for (unsigned k = 0U; k < reduction; k += (FP8 ? 128U : kK)) {
      s.empty[stage].wait(phase ^ 1U);
      if constexpr (FP8) {
        auto *sa = fp8_scales(s, stage), *sb = sa + kM;
        [[maybe_unused]] const auto stride = dx ? a.activation_stride : f.hidden;
        const auto *input = reinterpret_cast<const std::uint8_t *>(dx
            ? a.dx_rows : Network ? a.dr : peer_at<const transport::BFloat16>(f.local_base, f.canonical_rows_offset)) +
            std::uint64_t(input_row) * (dx ? a.fp8_activation_pitch : Network ? fp8::source_pitch(unsigned(f.hidden)) : f.fp8_x_pitch);
        constexpr bool Packed = WEFTMOE_FP8_PACKED_RING && local::kN == 256U;
        if constexpr (Network && Packed && local::kFP8NetworkRowScales)
          if (!dx)
            for (unsigned row = threadIdx.x % 32U; row < kM; row += 32U)
              sa[row] = row < t.rows ? reinterpret_cast<const float *>(input + kM * reduction)[row * (reduction / 128U) + k / 128U] : 1.0F;
        if constexpr (!Packed)
          for (unsigned row = threadIdx.x % 32U; row < kM; row += 32U)
            sa[row] = row < t.rows ? reinterpret_cast<const float *>(
                input + std::uint64_t(row) * stride * 2U + reduction)[k / 128U] : 1.0F;
        const auto columns = dx ? kDxN : kDuN;
        const auto elements = f.hidden * f.intermediate * (dx ? 2U : 1U);
        fp8::WeightReader weight{nullptr,
            (dx ? f.fp8.s1_t : f.fp8.s2_t) + t.expert * elements / (128U * 128U),
            unsigned(reduction), t.column * columns, unsigned(f.intermediate), false};
        const auto lane = threadIdx.x % 32U;
        if (lane < 2U * (columns / 128U)) {
          const auto column = lane / 2U * 128U + lane % 2U * 8U;
          sb[lane] = weight.scale(column, k / 128U);
        }
        __syncwarp();
        if (cute::elect_one_sync()) {
          if constexpr (Packed) {
            auto *barrier = reinterpret_cast<std::uint64_t *>(&s.full[stage]);
            if (cute::block_rank_in_cluster() == 0U)
              cute::SM90_TMA_LOAD_MULTICAST_3D::copy(dx ? &dab : &dy, barrier, 3U,
                static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.a[stage], k, 0, input_row / kM);
            if (!Network || dx || !local::kFP8NetworkRowScales)
              cute::SM90_BULK_COPY_G2S::copy(input + kM * reduction + k * sizeof(float),
                  barrier, sa, kM * sizeof(float));
          } else
            hopper::tma_load<128U, kM, 128U>(dx ? &dab : &dy, &s.full[stage],
                reinterpret_cast<std::uint8_t *>(s.a[stage]), k, input_row, 2U);
          if (dx)
            hopper::tma_load<128U, kDxN, 128U>(&w1, &s.full[stage],
                reinterpret_cast<std::uint8_t *>(s.b[stage]), k,
                t.expert * unsigned(f.hidden) + t.column * kDxN);
          else
            hopper::tma_load<128U, kDuN, 128U>(&w2, &s.full[stage],
                reinterpret_cast<std::uint8_t *>(s.b[stage]), k,
                t.expert * unsigned(f.intermediate) + t.column * kDuN);
          s.full[stage].arrive_and_expect_tx((kM + columns) * 128U +
              (Packed && (!Network || dx || !local::kFP8NetworkRowScales) ? kM * sizeof(float) : 0U));
        }
      } else {
        if (cute::elect_one_sync()) {
          hopper::tma_load<kK, kM, 128U>(dx ? &dab : &dy, &s.full[stage],
                                         s.a[stage], k, input_row, 2U);
          if (dx)
            hopper::tma_load<kDxN, kK, 128U>(
                &w1, &s.full[stage], s.b[stage], t.column * kDxN,
                t.expert * unsigned(2U * f.intermediate) + k);
          else
            hopper::tma_load<kDuN, kK, 128U>(&w2, &s.full[stage], s.b[stage],
                                             t.column * kDuN,
                                             t.expert * unsigned(f.hidden) + k);
          s.full[stage].arrive_and_expect_tx((kM + (dx ? kDxN : kDuN)) * kK * 2U);
        }
      }
      stage = (stage + 1U) % kStages;
      phase ^= stage == 0U;
    }
  }
  // Drain every operand slot, including remote multicast readers, before
  // the activation barriers themselves become part of the dW output arena.
  for (unsigned i = 0U; i < kStages; ++i) {
    s.empty[stage].wait(phase ^ 1U);
    stage = (stage + 1U) % kStages;
    phase ^= stage == 0U;
  }
}

template <unsigned N, bool Dx, bool Network>
__device__ __forceinline__ void fp8_activation(
    const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s,
    const Task &t, float *accum, unsigned &stage, unsigned &phase) {
  {
    constexpr unsigned reduction = Dx ? 2U * backward_tuning::kMainIntermediate : backward_tuning::kMainHidden;
    for (unsigned k = 0; k < reduction; k += 128U) {
      s.full[stage].wait(phase);
      auto *scales = fp8_scales(s, stage);
      const bool cache_owner = !Dx && a.fp8_x_data &&
          t.column == (t.block + k / 128U) % kDuTiles;
      // Stripe ownership over dU N tiles. Each X block is converted once,
      // alongside dR; every dW1 tile then reuses the same global result.
      auto *xraw = reinterpret_cast<std::uint8_t *>(&s) + kFP8BaseSharedBytes;
      auto *xscales = reinterpret_cast<float *>(xraw + kM * 128U);
      const auto xoffset = std::uint64_t(t.block) * kM * reduction + k * kM;
      if (cache_owner && a.fp8_x_row_data) {
        for (unsigned v = threadIdx.x * 16U; v < kM * 128U; v += 256U * 16U) {
          const auto dest = unsigned(__cvta_generic_to_shared(xraw + v));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
              "r"(dest), "l"(a.fp8_x_row_data + xoffset + v) : "memory");
        }
        if (threadIdx.x < kM)
          xscales[threadIdx.x] = a.fp8_x_row_scales[xoffset / 128U + threadIdx.x];
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
      if (threadIdx.x / 128U * 64U < t.rows)
        fp8::multiply_tile<N, fp8::NoSync, true, true>(reinterpret_cast<std::uint8_t *>(s.a[stage]),
            reinterpret_cast<std::uint8_t *>(s.b[stage]), scales, scales + kM, accum);
      if (cache_owner) {
        const auto offset = std::uint64_t(t.block) * kM * reduction + k * kM;
        fp8::row_to_column(reinterpret_cast<const std::uint8_t *>(s.a[stage]),
            scales, t.rows, a.fp8_wgrad_data + offset, a.fp8_wgrad_scales + offset / 128U);
        if (a.fp8_x_row_data) {
          asm volatile("cp.async.wait_group 0;" ::: "memory");
          fp8::MathSync::sync();
          fp8::row_to_column(xraw, xscales, t.rows,
              a.fp8_x_data + xoffset,
              a.fp8_x_scales + xoffset / 128U);
          fp8::MathSync::sync();
        }
      }
      backward_compute_detail::release_stage(s.empty[stage]);
      stage = (stage + 1U) % kStages;
      phase ^= stage == 0U;
    }
    return;
  }

}

template <unsigned N, bool Dx, bool Network, bool FP8>
__device__ __forceinline__ void
gemm(const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s,
     const Task &t, unsigned &stage, unsigned &phase) {
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
  const auto task_begin = device_globaltimer_ns();
#endif
  constexpr auto MajorA = hopper::Major::K, MajorB = hopper::Major::MN;
  using Mma =
      typename hopper::BF16MMASelector<N, MajorA, MajorB, kK, 128U>::type;
  auto da = hopper::make_gmma_desc<MajorA, kM, kK, 128U>(
      s.a[0], threadIdx.x / 128U * 64U, 0U);
  auto db = hopper::make_gmma_desc<MajorB, N, kK, 128U>(s.b[0], 0U, 0U);
  const auto abase = __shfl_sync(0xffffffffU, da.reg32_[0], 0U);
  const auto bbase = __shfl_sync(0xffffffffU, db.reg32_[0], 0U);
  float accum[Mma::kNumAccum] = {};
  const auto reduction = Dx ? 2U * f.intermediate : f.hidden;
  const bool checkpoint_owner =
      !Dx && !Network && t.column / 2U == t.block % (kDuTiles / 2U);
  const auto half = (t.column % 2U) * 64U;
  const auto checkpoint_rows = t.rows > half ? min(64U, t.rows - half) : 0U;
  forward_compute_detail::StaticV4XCheckpoint<64U, 64U> checkpoint;
  if constexpr (FP8) {
    fp8_activation<N, Dx, Network>(f, a, s, t, accum, stage, phase);
  } else {
  for (unsigned k = 0U; k < reduction; k += kK) {
    s.full[stage].wait(phase);
    if (k == 0U && checkpoint_owner)
      checkpoint.prepare(f, t.block, checkpoint_rows, half);
    if (k == 0U && !threadIdx.x) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicMin(Dx ? &status->relay_blocks_done : &status->fc1_start_clock,
                device_globaltimer_ns());
    }
#pragma unroll
    for (auto &value : accum)
      cute::warpgroup_fence_operand(value);
    cute::warpgroup_arrive();
#pragma unroll
    for (unsigned atom = 0U; atom < kK; atom += Mma::K) {
      da.reg32_[0] = hopper::advance_gmma_desc_lo<MajorA, kM, kK, 128U,
                                                  transport::BFloat16>(
          abase + stage * kM * kK / 8U, 0U, atom);
      db.reg32_[0] = hopper::advance_gmma_desc_lo<MajorB, N, kK, 128U,
                                                  transport::BFloat16>(
          bbase + stage * kN * kK / 8U, 0U, atom);
      Mma::wgmma(da, db, accum, k != 0U || atom != 0U);
    }
    cute::warpgroup_commit_batch();
#pragma unroll
    for (auto &value : accum)
      cute::warpgroup_fence_operand(value);
    if (checkpoint_owner)
#pragma unroll
      for (unsigned atom = 0U; atom < kK / 64U; ++atom)
        checkpoint.save(s.a[stage] + atom * kM * 64U + half * 64U,
                        checkpoint_rows, k + atom * 64U, f.hidden,
                        f.saved_x_map);
    cute::warpgroup_wait<0>();
    backward_compute_detail::release_stage(s.empty[stage]);
    stage = (stage + 1U) % kStages;
    phase ^= stage == 0U;
  }
  }
  const auto lane = threadIdx.x % 32U;
  const auto row0 = threadIdx.x / 32U * 16U + lane / 4U;
  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
#pragma unroll
  for (unsigned item = 0U; item < 2U; ++item) {
    const auto row = row0 + item * 8U;
    if (row >= t.rows)
      continue;
    transport::BFloat16 *output;
    if constexpr (Dx)
      output = Network ? dx_output(f, a, t, row)
                       : static_cast<transport::BFloat16 *>(
                             f.ring.rows[t.block * kM + row].output);
    else
      output = a.recomputed_u +
               std::uint64_t(
                   a.fp8_du_ring ? t.row + row
                           : Network ? t.block * kM + row
                           : mapping[f.checkpoint.rows[t.block * kM + row]]) *
                   a.intermediate;
    output += t.column * N;
#pragma unroll
    for (unsigned index = 0U; index < N / 8U; index += 4U) {
      std::uint32_t bits[4];
#pragma unroll
      for (unsigned j = 0U; j < 4U; ++j) {
        const auto *value = accum + (index + j) * 4U + item * 2U;
        bits[j] = float2_to_bf16x2_bits(value[0], value[1]);
      }
      *reinterpret_cast<uint4 *>(output + index * 8U + lane % 4U * 8U) =
          forward_compute_detail::static_v4_transpose_vector(
              make_uint4(bits[0], bits[1], bits[2], bits[3]));
    }
  }
  if constexpr (Dx) {
    if constexpr (Network)
      __threadfence_system();
  } else {
    __threadfence();
  }
  backward_weights::UnifiedComputeSynchronization<
      true>::math_sync(256U, 0U);
  if (!threadIdx.x) {
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicAdd(Dx ? &status->fc2_tasks_done : &status->fc1_tasks_done,
              device_globaltimer_ns() - task_begin);
    if constexpr (!Dx)
      atomicMax(&status->before_combine_complete_clock,
                device_globaltimer_ns());
#endif
    auto &slot = f.ring.slots[t.row / kM];
    ring::arrive(Dx ? slot.fc2_done : slot.fc1_done, 1U);
    if constexpr (Dx && Network)
      ring::arrive(slot.fc2_published, 1U);
  }
}

template <bool Network, bool FP8>
__device__ __forceinline__ void
compute(const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s) {
  using Scheduler = std::conditional_t<Network, NetworkSchedule, Schedule<FP8>>;
  Scheduler schedule(f, a);
  Task t;
  unsigned stage = 0U, phase = 0U;
  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  while (schedule.next(t)) {
    if (t.phase == Phase::Du) {
      gemm<kDuN, false, Network, FP8>(f, a, s, t, stage, phase);
    } else if (t.phase == Phase::Dx) {
      gemm<kDxN, true, Network, FP8>(f, a, s, t, stage, phase);
    } else {
      auto &slot = f.ring.slots[t.row / kM];
      if (!threadIdx.x)
        while (ring::acquire(slot.fc1_done) <
                   std::uint64_t(t.generation) * kDuTiles ||
               (Network && ring::acquire(slot.returned) < t.generation - 1U)) {
        }
      using Sync =
          backward_weights::UnifiedComputeSynchronization<true>;
      Sync::math_sync(256U, 0U);
      if constexpr (FP8) {
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
          const auto task_begin = device_globaltimer_ns();
#endif
          fp8_swiglu_columns<Network>(f, a, t, reinterpret_cast<float *>(
              reinterpret_cast<unsigned char *>(&s) + kFP8BaseSharedBytes));
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
          if (!threadIdx.x)
            atomicAdd(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                           ->synthetic_ticks,
                      device_globaltimer_ns() - task_begin);
#endif
      } else {
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
      const auto task_begin = device_globaltimer_ns();
#endif
      for (unsigned row = t.column * 64U + threadIdx.x / 32U;
           row < (t.column + 1U) * 64U; row += 8U) {
        const bool valid = row < t.rows;
        const auto saved =
            valid ? (Network ? directory(a)[t.block].compact_row + row
                             : f.checkpoint.rows[t.block * kM + row])
                  : 0U;
        const auto padded =
            Network || !valid ? t.block * kM + row : mapping[saved];
        // All dU N tiles have retired above. The local dY ring can now
        // receive dAB directly from registers, alongside the dW1 cache.
        auto *staging = Network ? nullptr
            : a.dx_rows + std::uint64_t(t.row + row) * a.activation_stride;
        backward_swiglu_epilogue_warp_row<!Network, false>(
            a, padded, saved, valid, threadIdx.x % 32U, staging, nullptr);
        __syncwarp();
        const auto lane = threadIdx.x % 32U;
        if (valid && !lane) {
          if constexpr (Network)
            *gradient_output(f, a, t, row) = a.dgate_rows[padded];
          else {
            const auto identity = f.ring.rows[t.block * kM + row];
            a.peer_payloads[identity.source].dgate[identity.route] =
                a.dgate_rows[padded];
          }
        }
      }
      if constexpr (Network)
        __threadfence_system();
      else
        __threadfence();
      Sync::math_sync(256U, 0U);
      if (!threadIdx.x) {
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
        atomicAdd(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                       ->synthetic_ticks,
                  device_globaltimer_ns() - task_begin);
#endif
        ring::arrive(slot.activation_ready, 1U);
        atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                       ->fc1_complete_clock,
                  device_globaltimer_ns());
      }
      }
    }
  }
}

// Phase helpers keep math/producer scopes separate from the return roles.
__device__ __forceinline__ void retire_activation_barriers(Shared &s) {
  if (threadIdx.x != 256U)
    return;
  // These barriers may alias the dW store tile. PTX requires inval before
  // their storage can be used for ordinary data, even after the phase join.
  for (auto &barrier : s.full)
    local::Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&barrier));
  for (auto &barrier : s.empty)
    local::Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&barrier));
  for (auto &barrier : s.pull_full)
    local::Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&barrier));
}

template <bool Network, bool FP8 = false>
__device__ __forceinline__ void
begin_weights(const ForwardKernelArgs &f, const BackwardKernelArgs &a,
              unsigned char *storage,
              backward_weights::SharedState &state) {
  if constexpr (Network) {
    // Only compute joins here. Return owners keep their disjoint communication
    // arena alive and can drain dX/dr while the static dW stream executes.
    using Sync =
        backward_weights::UnifiedComputeSynchronization<true>;
    Sync::cta_sync();
    retire_activation_barriers(*reinterpret_cast<Shared *>(storage));
    backward_weights::initialize_compute<
        backward_weights::kStorageTileN, FP8>(storage, state);
    Sync::cta_sync();
    if (!threadIdx.x) {
      const auto layout = pipeline::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity, kDxTiles / 2U,
          a.scheduler_expert_capacity);
      auto *word = backward_pipeline_detail::scheduler_words(a) +
                   layout.completed_dx_tiles;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> done(
          *word);
      done.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
      while (done.load(::cuda::memory_order_acquire) != gridDim.x) {
      }
    }
    Sync::cta_sync();
  } else {
    // Local slot credits retire input reads, not peer contribution stores.
    // Every writer drains once before the grid/rank handoff to Reduce.
    __threadfence_system();
    grid_sync<0U, 384U>(f);
    retire_activation_barriers(*reinterpret_cast<Shared *>(storage));
    backward_weights::initialize_compute<
        backward_weights::kStorageTileN, FP8>(storage, state);
    __syncthreads();
    cute::cluster_sync();
  }
  if (!threadIdx.x)
    atomicMax(&peer_at<TransportDeviceStatus>(f.local_base, f.status_offset)
                   ->fc2_complete_clock,
              device_globaltimer_ns());
}

__device__ __forceinline__ void finish() {
  __syncthreads();
  cute::cluster_sync();
}
#endif

template <bool Network = false, bool FP8 = false>
__global__ __launch_bounds__(384U, 1) void kernel(
    const __grid_constant__ ForwardKernelArgs f,
    const __grid_constant__ BackwardKernelArgs a,
    const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap dab_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap dr_mn_map,
    const __grid_constant__ CUtensorMap u_mn_map,
    const __grid_constant__ CUtensorMap dab_mn_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    const __grid_constant__ CUtensorMap dw1_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<Shared *>(storage);
  __shared__ backward_weights::SharedState state;
  const auto geometry = backward_weights::load_row_geometry(a);
  if (!blockIdx.x && !threadIdx.x)
    peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)->start_clock =
        device_globaltimer_ns();
  if (threadIdx.x == 256U) {
    const auto readers = cute::block_rank_in_cluster() == 0U ? 16U : 8U;
    for (unsigned i = 0U; i < kStages; ++i) {
      s.full[i].init(1U);
      s.empty[i].init(readers);
    }
    for (auto &barrier : s.pull_full)
      barrier.init(1U);
    if constexpr (Network) {
      auto &comm =
          *reinterpret_cast<NetworkShared *>(storage + (FP8 ? kFP8NetworkCommOffset : kNetworkCommOffset));
      for (auto &barrier : comm.pull_full)
        barrier.init(1U);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();
  if (threadIdx.x < 256U) {
    cutlass::arch::warpgroup_reg_alloc<
        FP8 ? fp8::kMathRegisters : Network ? kNetworkMathRegisters
                : backward_tuning::kUnifiedMathRegisters>();
#if defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
    identity<Network>(f, a);
    begin_weights<Network, FP8>(f, a, storage, state);
#else
    compute<Network, FP8>(f, a, s);
    begin_weights<Network, FP8>(f, a, storage, state);
    if constexpr (FP8) {
      backward_weights::run_fp8_weights_pipeline<false>(a, geometry, storage, state, dw1_store_map, dw2_store_map);
    } else {
    backward_weights::UnifiedComputeSynchronization<true> sync{
        state};
    backward_weights::run_compute<backward_tuning::kDwTileN, FP8>(
        a, dr_mn_map, u_mn_map, dab_mn_map, x_mn_map, dw2_store_map, dw1_store_map,
        geometry, storage, state, sync);
    }
#endif
    finish();
  } else {
    cutlass::arch::warpgroup_reg_dealloc<
        FP8 ? fp8::kControlRegisters : Network ? kNetworkControlRegisters
                : backward_tuning::kUnifiedControlRegisters>();
    if (threadIdx.x < 288U) {
#if defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
      begin_weights<Network, FP8>(f, a, storage, state);
#else
      produce<Network, FP8>(f, a, s, dy_map, dab_map, w2_map, w1_map);
      begin_weights<Network, FP8>(f, a, storage, state);
      if constexpr (FP8)
        backward_weights::run_fp8_weights_pipeline<true>(a, geometry, storage, state, dw1_store_map, dw2_store_map);
      if constexpr (!FP8) {
      backward_weights::UnifiedComputeSynchronization<false>
          sync{state};
      backward_weights::run_compute<backward_tuning::kDwTileN, FP8>(
        a, dr_mn_map, u_mn_map, dab_mn_map, x_mn_map, dw2_store_map, dw1_store_map,
          geometry, storage, state, sync);
      }
#endif
      finish();
    } else {
      if constexpr (Network) {
        auto &comm =
            *reinterpret_cast<NetworkShared *>(storage + (FP8 ? kFP8NetworkCommOffset : kNetworkCommOffset));
        network_workers(f, a, comm);
        finish_network<>(f, a);
      } else {
        if ((threadIdx.x - 288U) / 32U < local::kPullWarps) {
          if constexpr (FP8) {
            if (f.fp8_source_x)
              local::pull_fp8<kDxTiles, true, true>(f, s, nullptr);
            else
              local::pull_fp8<kDxTiles, true>(f, s, a.fp8_x_data ? nullptr : a.dr);
          } else
            local::pull<kDxTiles, true>(f, s);
        }
        begin_weights<false, FP8>(f, a, storage, state);
        auto *status =
            peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
        rank_barrier<288U, 96U>(f);
        if (threadIdx.x == 288U)
          atomicMax(&status->return_complete_clock, device_globaltimer_ns());
        if (threadIdx.x == 288U)
          atomicMin(&status->backward_reduce_start_clock,
                    device_globaltimer_ns());
        local::combine<false, 288U, 96U>(f);
        if (threadIdx.x == 288U)
          atomicMax(&status->backward_reduce_complete_clock,
                    device_globaltimer_ns());
      }
      finish();
      // Communication-only Reduce overlaps dW. Its terminal barrier must
      // still protect metadata until every rank finishes weight gradients.
      if constexpr (Network) {
        network_barrier<288U, 96U>(f, 1U);
      } else
        rank_barrier<288U, 96U>(f);
    }
  }
#endif
}

} // namespace weftmoe_pf::backward_stream
