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
#define WEFTMOE_PF_BWD_WAVE_BLOCKS 128
#endif
#ifndef WEFTMOE_PF_BWD_LOCAL_EXPERT_WAVES
#define WEFTMOE_PF_BWD_LOCAL_EXPERT_WAVES 0
#endif
#ifndef WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS
#define WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS 0
#endif
#ifndef WEFTMOE_PF_BWD_GATHER_BATCH
#define WEFTMOE_PF_BWD_GATHER_BATCH 1
#endif
#ifndef WEFTMOE_PF_BWD_SWIGLU_VECTOR_WORDS
#define WEFTMOE_PF_BWD_SWIGLU_VECTOR_WORDS 1
#endif
#ifndef WEFTMOE_PF_BWD_SWIGLU_DIRECT_RING
#define WEFTMOE_PF_BWD_SWIGLU_DIRECT_RING 0
#endif
#ifndef WEFTMOE_PF_BWD_ACTIVATION_WAIT_GROUP
#define WEFTMOE_PF_BWD_ACTIVATION_WAIT_GROUP 0
#endif
#ifndef WEFTMOE_PF_BWD_DEFER_LOCAL_RETURN_FENCE
#define WEFTMOE_PF_BWD_DEFER_LOCAL_RETURN_FENCE 0
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
    backward_unified_detail::kStorageTileN>();
constexpr auto kSharedBytes = sizeof(Shared) > kDwBytes ? sizeof(Shared)
                                                        : kDwBytes;

enum class Phase : unsigned { Du, Swiglu, Dx };
struct Task {
  unsigned expert, block, column, rows, row, generation;
  Phase phase;
};

struct Schedule {
  const ForwardKernelArgs &a;
  unsigned expert = 0U, begin = 0U, end = 0U, index = blockIdx.x;
  unsigned wave_expert = 0U;

  __device__ Schedule(const ForwardKernelArgs &args, const BackwardKernelArgs &)
      : a(args) {}

  __device__ __forceinline__ bool next(Task &t) {
    const auto all_blocks = unsigned(a.ring.state->blocks);
    while (begin < all_blocks) {
      if (end == begin) {
        end = min(all_blocks, begin + a.ring.depth);
        if constexpr (WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS > 0)
          end = min(end, begin + WEFTMOE_PF_BWD_LOCAL_WAVE_BLOCKS);
        if constexpr (WEFTMOE_PF_BWD_LOCAL_EXPERT_WAVES) {
          while (begin >= a.ring.expert_blocks[wave_expert + 1U])
            ++wave_expert;
          end = min(end, unsigned(a.ring.expert_blocks[wave_expert + 1U]));
        }
      }
      const auto blocks = end - begin;
      const auto du = blocks * kDuTiles;
      const auto epilogue = blocks * 2U;
      const auto total = du + epilogue + blocks * kDxTiles;
      if (index < total) {
        const auto phase = index < du              ? Phase::Du
                           : index < du + epilogue ? Phase::Swiglu
                                                   : Phase::Dx;
        const auto linear = index - (phase == Phase::Du       ? 0U
                                     : phase == Phase::Swiglu ? du
                                                              : du + epilogue);
        const auto columns = phase == Phase::Du   ? kDuTiles
                             : phase == Phase::Dx ? kDxTiles
                                                  : 2U;
        const auto block = begin + linear / columns;
        if (block < a.ring.expert_blocks[expert])
          expert = 0U;
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
    for (auto block = std::uint64_t(tid); block < (rows + kM - 1U) / kM;
         block += stride)
      words[layout.phase + padded / kM + block] = padded / kM + block;
    // Initialize canonical cache padding before the dW phase.
    const auto tail = (kM - rows % kM) % kM;
    for (auto v = std::uint64_t(tid); v < tail * a.intermediate; v += stride) {
      reinterpret_cast<unsigned short *>(
          a.recomputed_u)[(padded + rows) * a.intermediate + v] = 0U;
      auto *dab = reinterpret_cast<unsigned short *>(a.dab) +
                  (padded + rows) * 2U * a.intermediate;
      dab[v] = dab[tail * a.intermediate + v] = 0U;
    }
  }
  if (!rows) {
    const auto elements = a.hidden * a.intermediate;
    auto *dw1 =
        reinterpret_cast<unsigned short *>(a.dw1 + expert * 2U * elements);
    auto *dw2 = reinterpret_cast<unsigned short *>(a.dw2 + expert * elements);
    for (auto i = std::uint64_t(tid); i < 2U * elements; i += stride) {
      dw1[i] = 0U;
      if (i < elements)
        dw2[i] = 0U;
    }
  }
  if (!expert) {
    if constexpr (!Network)
      for (auto i = tid; i < f.ring.depth; i += stride)
        f.ring.slots[i] = {};
    if (!tid) {
      if (a.runtime_epoch)
        ++*a.runtime_epoch;
      words[layout.completed_dw1_tiles] = words[layout.completed_dw2_tiles] =
          0ULL;
      words[layout.completed_dx_tiles] = 0ULL;
      backward_pipeline_detail::initialize_status(a);
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
      auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      status->before_combine_complete_clock = 0ULL;
      status->fc1_tasks_done = status->fc2_tasks_done = 0ULL;
      status->synthetic_ticks = status->dispatch_put_ticks = status->relay_ticks = 0ULL;
#endif
    }
  }
}

} // namespace weftmoe_pf::backward_stream

#include "cuda/weftmoe_pf/backward_network.cuh"
#include "cuda/weftmoe_pf/backward_epilogue.cuh"
#if defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
#include "../../../benchmarks/pf_backward_identity.cuh"
#endif

namespace weftmoe_pf::backward_stream {

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
template <bool Network>
__device__ __forceinline__ void
produce(const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s,
        const CUtensorMap &dy, const CUtensorMap &dab, const CUtensorMap &w2,
        const CUtensorMap &w1) {
  using Scheduler = std::conditional_t<Network, NetworkSchedule, Schedule>;
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
      auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(dx ? &status->relay_ticks : &status->dispatch_put_ticks,
                device_globaltimer_ns() - ready_begin);
    }
#endif
    const auto reduction = dx ? 2U * f.intermediate : f.hidden;
    const auto input_row =
        Network ? (dx ? t.block * kM : directory(a)[t.block].compact_row)
                : t.row;
    for (unsigned k = 0U; k < reduction; k += kK) {
      s.empty[stage].wait(phase ^ 1U);
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
      stage = (stage + 1U) % kStages;
      phase ^= stage == 0U;
    }
  }
}

template <unsigned N, bool Dx, bool Network>
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
    if constexpr (WEFTMOE_PF_BWD_ACTIVATION_WAIT_GROUP == 0) {
      cute::warpgroup_wait<0>();
      backward_compute_detail::release_stage(s.empty[stage]);
    } else if (k) {
      static_assert(kStages >= 2U);
      cute::warpgroup_wait<1>();
      backward_compute_detail::release_stage(s.empty[(stage + kStages - 1U) % kStages]);
    }
    stage = (stage + 1U) % kStages;
    phase ^= stage == 0U;
  }
#if WEFTMOE_PF_BWD_ACTIVATION_WAIT_GROUP != 0
  cute::warpgroup_wait<0>();
  backward_compute_detail::release_stage(s.empty[(stage + kStages - 1U) % kStages]);
#endif
#if defined(WEFTMOE_PF_BWD_FUSED_SWIGLU)
  if constexpr (!Dx) {
    // dU was BF16 before SwiGLU in the unfused path. Compact it before loading
    // G/U so the epilogue does not keep 128 FP32 accumulators live alongside
    // its activation vectors. This is a compiler fence, not a device barrier.
    unsigned packed[Mma::kNumAccum / 2U];
#pragma unroll
    for (unsigned i = 0U; i < Mma::kNumAccum / 2U; ++i) {
      packed[i] = float2_to_bf16x2_bits(accum[2U * i], accum[2U * i + 1U]);
      asm volatile("" : "+r"(packed[i]) : : "memory");
    }
    store_swiglu<N, Network>(f, a, t, packed);
  } else
#endif
  {
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
                   Network ? t.block * kM + row
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
  }
  if constexpr (Dx) {
    if constexpr (Network || !WEFTMOE_PF_BWD_DEFER_LOCAL_RETURN_FENCE)
      __threadfence_system();
  } else {
    __threadfence();
  }
  backward_unified_compute_detail::UnifiedComputeSynchronization<
      true>::math_sync(256U, 0U);
  if (!threadIdx.x) {
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
    auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicAdd(Dx ? &status->fc2_tasks_done : &status->fc1_tasks_done,
              device_globaltimer_ns() - task_begin);
    if constexpr (!Dx)
      atomicMax(&status->before_combine_complete_clock, device_globaltimer_ns());
#endif
    auto &slot = f.ring.slots[t.row / kM];
    ring::arrive(Dx ? slot.fc2_done : slot.fc1_done, 1U);
    if constexpr (Dx && Network)
      ring::arrive(slot.fc2_published, 1U);
  }
}

template <bool Network>
__device__ __forceinline__ void
compute(const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s) {
  using Scheduler = std::conditional_t<Network, NetworkSchedule, Schedule>;
  Scheduler schedule(f, a);
  Task t;
  unsigned stage = 0U, phase = 0U;
  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  while (schedule.next(t)) {
    if (t.phase == Phase::Du) {
      gemm<kDuN, false, Network>(f, a, s, t, stage, phase);
    } else if (t.phase == Phase::Dx) {
      gemm<kDxN, true, Network>(f, a, s, t, stage, phase);
    } else {
      auto &slot = f.ring.slots[t.row / kM];
      if (!threadIdx.x)
        while (ring::acquire(slot.fc1_done) <
                   std::uint64_t(t.generation) * kDuTiles ||
               (Network && ring::acquire(slot.returned) < t.generation - 1U)) {
        }
      using Sync =
          backward_unified_compute_detail::UnifiedComputeSynchronization<true>;
      Sync::math_sync(256U, 0U);
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
        const auto lane = threadIdx.x % 32U;
        float gate_gradient = 0.0F;
#if defined(WEFTMOE_PF_BWD_FUSED_SWIGLU)
        for (unsigned tile = lane; tile < kDuTiles; tile += 32U)
          gate_gradient += a.dgate_rows[std::uint64_t(padded) * kDuTiles + tile];
#pragma unroll
        for (unsigned stride = 16U; stride; stride >>= 1U)
          gate_gradient += __shfl_down_sync(0xffffffffU, gate_gradient, stride);
#else
        backward_swiglu_epilogue_warp_row<
            !Network && WEFTMOE_PF_BWD_SWIGLU_DIRECT_RING,
            WEFTMOE_PF_BWD_SWIGLU_VECTOR_WORDS>(
            a, padded, saved, valid, lane,
            Network ? nullptr : a.dx_rows +
                                    std::uint64_t(t.row + row) * a.activation_stride);
        __syncwarp();
        if (!lane)
          gate_gradient = a.dgate_rows[padded];
#endif
        if constexpr (!Network &&
#if defined(WEFTMOE_PF_BWD_FUSED_SWIGLU)
                      true
#else
                      !WEFTMOE_PF_BWD_SWIGLU_DIRECT_RING
#endif
        ) {
          auto *out = reinterpret_cast<uint4 *>(
              a.dx_rows + std::uint64_t(t.row + row) * a.activation_stride);
          const auto *in = reinterpret_cast<const uint4 *>(
              a.dab + std::uint64_t(padded) * 2U * a.intermediate);
          constexpr unsigned Batch = WEFTMOE_PF_BWD_GATHER_BATCH;
          for (unsigned column = lane; column < 2U * a.intermediate / 8U;
               column += 32U * Batch) {
            uint4 values[Batch];
#pragma unroll
            for (unsigned i = 0U; i < Batch; ++i)
              if (column + 32U * i < 2U * a.intermediate / 8U)
                values[i] = valid ? in[column + 32U * i] : make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
            for (unsigned i = 0U; i < Batch; ++i)
              if (column + 32U * i < 2U * a.intermediate / 8U)
                out[column + 32U * i] = values[i];
          }
        }
        if (valid && !lane) {
          if constexpr (Network)
            *gradient_output(f, a, t, row) = gate_gradient;
          else {
            const auto identity = f.ring.rows[t.block * kM + row];
            a.peer_payloads[identity.source].dgate[identity.route] =
                gate_gradient;
          }
        }
      }
      if constexpr (Network || !WEFTMOE_PF_BWD_DEFER_LOCAL_RETURN_FENCE)
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

// Phase helpers keep math/producer scopes separate from the return roles.
template <bool Network>
__device__ __forceinline__ void
begin_weights(const ForwardKernelArgs &f, const BackwardKernelArgs &a,
              unsigned char *storage,
              backward_unified_compute_detail::SharedState &state) {
  if constexpr (Network) {
    // Only compute joins here. Return owners keep their disjoint communication
    // arena alive and can drain dX/dr while the static dW stream executes.
    using Sync =
        backward_unified_compute_detail::UnifiedComputeSynchronization<true>;
    Sync::cta_sync();
    backward_unified_compute_detail::initialize_compute<
        backward_unified_detail::kStorageTileN>(storage, state);
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
    // Local ring credits retire dY/dAB reads, not remote contribution stores.
    // Drain every writer once before the grid hands publication to Return.
    if constexpr (WEFTMOE_PF_BWD_DEFER_LOCAL_RETURN_FENCE)
      __threadfence_system();
    grid_sync<0U, 384U>(f);
    backward_unified_compute_detail::initialize_compute<
        backward_unified_detail::kStorageTileN>(storage, state);
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

template <bool Network = false>
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
    const __grid_constant__ CUtensorMap u_store_map,
    const __grid_constant__ CUtensorMap dx_store_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    const __grid_constant__ CUtensorMap dw1_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<Shared *>(storage);
  __shared__ backward_unified_compute_detail::SharedState state;
  const auto geometry = backward_unified_compute_detail::load_row_geometry(a);
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
          *reinterpret_cast<NetworkShared *>(storage + kNetworkCommOffset);
      for (auto &barrier : comm.pull_full)
        barrier.init(1U);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();
  if (threadIdx.x < 256U) {
    cutlass::arch::warpgroup_reg_alloc<
        Network ? kNetworkMathRegisters
                : backward_tuning::kUnifiedMathRegisters>();
#if defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
    identity<Network>(f, a);
    begin_weights<Network>(f, a, storage, state);
#else
    compute<Network>(f, a, s);
    begin_weights<Network>(f, a, storage, state);
    backward_unified_compute_detail::UnifiedComputeSynchronization<true> sync{
        state};
    backward_unified_compute_detail::run_compute<
        kDuN, kDxN, backward_tuning::kDwTileN, true>(
        a, dy_map, dab_map, w2_map, w1_map, dr_mn_map, u_mn_map, dab_mn_map,
        x_mn_map, u_store_map, dx_store_map, dw2_store_map, dw1_store_map,
        geometry, storage, state, sync);
#endif
    finish();
  } else {
    cutlass::arch::warpgroup_reg_dealloc<
        Network ? kNetworkControlRegisters
                : backward_tuning::kUnifiedControlRegisters>();
    if (threadIdx.x < 288U) {
#if defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
      begin_weights<Network>(f, a, storage, state);
#else
      produce<Network>(f, a, s, dy_map, dab_map, w2_map, w1_map);
      begin_weights<Network>(f, a, storage, state);
      backward_unified_compute_detail::UnifiedComputeSynchronization<false>
          sync{state};
      backward_unified_compute_detail::run_compute<
          kDuN, kDxN, backward_tuning::kDwTileN, true>(
          a, dy_map, dab_map, w2_map, w1_map, dr_mn_map, u_mn_map, dab_mn_map,
          x_mn_map, u_store_map, dx_store_map, dw2_store_map, dw1_store_map,
          geometry, storage, state, sync);
#endif
      finish();
    } else {
      if constexpr (Network) {
        auto &comm =
            *reinterpret_cast<NetworkShared *>(storage + kNetworkCommOffset);
        network_workers(f, a, comm);
        finish_network(f, a);
      } else {
        if ((threadIdx.x - 288U) / 32U < local::kPullWarps)
          local::pull<kDxTiles, true>(f, s);
        begin_weights<false>(f, a, storage, state);
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
      // Return/reduce overlaps dW, but a peer must not reuse invocation
      // metadata until every rank has also finished its weight gradients.
      if constexpr (Network)
        network_barrier<288U, 96U>(f, 1U);
      else
        rank_barrier<288U, 96U>(f);
    }
  }
#endif
}

} // namespace weftmoe_pf::backward_stream
