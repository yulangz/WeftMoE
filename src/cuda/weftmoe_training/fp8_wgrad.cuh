#pragma once

#ifndef WEFTMOE_FP8_WGRAD_PIPELINED_PROMOTION
#define WEFTMOE_FP8_WGRAD_PIPELINED_PROMOTION 0
#endif

// Keep the existing producer warp active during Wgrad. Operand slots have
// independent full/empty barriers, so the two math warpgroups need no CTA
// rendezvous for every K128 promotion. Output storage never aliases operands.
template <bool Producer>
__device__ __forceinline__ void run_fp8_weights_pipeline(
    const BackwardKernelArgs &a, const RowGeometry &geometry,
    unsigned char *storage, SharedState &state,
    const CUtensorMap &dw1_map, const CUtensorMap &dw2_map) {
  constexpr unsigned N = backward_tuning::kDwTileN;
  constexpr unsigned M = backward_tuning::kDwTileM;
  constexpr unsigned Stages = WEFTMOE_FP8_WGRAD_PIPELINE_STAGES;
  constexpr unsigned Claim = WEFTMOE_FP8_WGRAD_CLAIM_TILES;
  static_assert(Claim > 0U);
  static_assert(Stages == 2U || Stages == 3U);
  constexpr unsigned StageBytes = kFP8StageBytes;
  constexpr unsigned StoreOffset = kFP8StoreOffset;
  auto *qa = storage;
  auto *qb = qa + Stages * 128U * 128U;
  auto *sa = reinterpret_cast<float *>(qb + Stages * N * 128U);
  auto *sb = sa + 128U;
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  auto *full = reinterpret_cast<Barrier *>(storage + Stages * StageBytes);
  auto *empty = full + Stages;
  if (threadIdx.x == 256U) {
    for (unsigned s = 0; s < Stages; ++s) {
      full[s].init(1U);
      empty[s].init(8U);
    }
    cutlass::arch::fence_barrier_init();
  }
  using Sync = UnifiedComputeSynchronization<true>;
  Sync::cta_sync();
  unsigned stage = 0U, phase = 0U;
  constexpr auto tiles2 = backward_pipeline_detail::kDw2TilesPerExpert;
  constexpr auto tiles1 = backward_pipeline_detail::kDw1TilesPerExpert;
  constexpr auto chunks2 = (tiles2 + Claim - 1U) / Claim;
  constexpr auto chunks1 = (tiles1 + Claim - 1U) / Claim;
  const auto experts = a.dispatch.sizes.owned_expert_count;
  const auto layout = pipeline::backward::make_backward_scheduler_layout(
      a.scheduler_mblock_capacity,
      backward_tuning::kMainHidden / backward_tuning::kDxTileN / 2U,
      a.scheduler_expert_capacity);
  auto *words = backward_pipeline_detail::scheduler_words(a);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  for (;;) {
    if (threadIdx.x == 256U)
      state.task_tile = unsigned(atomicAdd(
          reinterpret_cast<unsigned long long *>(words + layout.dw2_head), 1ULL));
    Sync::cta_sync();
    const auto index = state.task_tile;
    Sync::cta_sync();
    if (index >= experts * (chunks2 + chunks1)) break;
    const bool first = index >= experts * chunks2;
    const auto chunks = first ? chunks1 : chunks2;
    const auto relative = first ? index - experts * chunks2 : index;
    const auto expert = relative / chunks;
    const auto rows = unsigned(a.dispatch.expert_row_count[expert]);
    if (!rows) continue;
    const auto begin = relative % chunks * Claim;
    const auto count = min(Claim, (first ? tiles1 : tiles2) - begin);
    const auto padded = peer_at<const unsigned long long>(
        a.local_base, a.padded_expert_offsets_offset)[expert];
    const unsigned m = first ? 2U * a.intermediate : a.hidden;
    const unsigned n = first ? a.hidden : a.intermediate;
    const auto lp = first ? a.hidden + a.intermediate : 0U;
    const auto rp = first ? 0U : a.hidden;
    const auto *right_data = first ? a.fp8_x_data : a.fp8_wgrad_data;
    const auto *right_scales = first ? a.fp8_x_scales : a.fp8_wgrad_scales;
    if (!threadIdx.x)
      atomicMin(first ? &status->backward_dw1_start_clock : &status->backward_dw2_start_clock,
          device_globaltimer_ns());
    for (unsigned task = begin; task < begin + count; ++task) {
      const auto tile = pipeline::backward::cluster_tile(task / 2U, task % 2U, m / M, n / N);
      if (!tile.valid) continue;
      for (unsigned wave = 0; wave < M / 128U; ++wave) {
        const auto left_column = tile.m * M + wave * 128U;
        const auto right_column = tile.n * N;
        if constexpr (Producer) {
          for (unsigned k = 0; k < rows; k += 128U) {
            empty[stage].wait(phase ^ 1U);
            if (threadIdx.x == 256U) {
              const auto lo = std::uint64_t(lp) * a.fp8_wgrad_rows +
                  (padded + k) * m + left_column * 128U;
              const auto ro = std::uint64_t(rp) * a.fp8_wgrad_rows +
                  (padded + k) * n + right_column * 128U;
              auto *barrier = reinterpret_cast<std::uint64_t *>(full + stage);
              cute::SM90_BULK_COPY_G2S::copy(a.fp8_wgrad_data + lo, barrier,
                  qa + stage * 128U * 128U, 128U * 128U);
              cute::SM90_BULK_COPY_G2S::copy(right_data + ro, barrier,
                  qb + stage * N * 128U, N * 128U);
              cute::SM90_BULK_COPY_G2S::copy(a.fp8_wgrad_scales + lo / 128U, barrier,
                  sa + stage * (128U + N), 128U * sizeof(float));
              cute::SM90_BULK_COPY_G2S::copy(right_scales + ro / 128U, barrier,
                  sb + stage * (128U + N), N * sizeof(float));
              full[stage].arrive_and_expect_tx(StageBytes);
            }
            stage = (stage + 1U) % Stages;
            phase ^= stage == 0U;
          }
        } else {
          float accum[N / 2U] = {};
          for (unsigned k = 0; k < rows; k += 128U) {
            full[stage].wait(phase);
            if constexpr (N == 256U && WEFTMOE_FP8_WGRAD_PIPELINED_PROMOTION)
              fp8::multiply_tile_pipelined_256<fp8::NoSync>(
                  qa + stage * 128U * 128U, qb + stage * N * 128U,
                  sa + stage * (128U + N), sb + stage * (128U + N), accum);
            else
              fp8::multiply_tile<N, fp8::NoSync, false, true>(
                  qa + stage * 128U * 128U, qb + stage * N * 128U,
                  sa + stage * (128U + N), sb + stage * (128U + N), accum);
            if (threadIdx.x % 32U == 0U) empty[stage].arrive();
            stage = (stage + 1U) % Stages;
            phase ^= stage == 0U;
          }
          if (!a.fp32_weight_gradients) {
            backward_compute_detail::store_tma_accumulator<128U, N, StoreOffset / 2U, Sync>(
                accum, first ? dw1_map : dw2_map, expert * m + left_column,
                right_column, storage, true, false);
          } else {
            auto *output = reinterpret_cast<float *>(first ? a.dw1 : a.dw2);
            const auto r = threadIdx.x / 32U * 16U + threadIdx.x % 32U / 4U;
            const auto c = threadIdx.x % 4U * 2U;
#pragma unroll
            for (unsigned j = 0; j < N / 8U; ++j)
#pragma unroll
              for (unsigned e = 0; e < 4U; e += 2U) {
                const auto offset = (std::uint64_t(expert) * m + left_column + r + e / 2U * 8U) * n +
                    right_column + c + j * 8U;
                *reinterpret_cast<float2 *>(output + offset) = make_float2(accum[j * 4U + e], accum[j * 4U + e + 1U]);
              }
          }
        }
      }
    }
    if (!threadIdx.x) {
      auto *counter = reinterpret_cast<backward_pipeline_detail::PaddedCounter *>(words +
          (first ? layout.completed_dw1_tiles : layout.completed_dw2_tiles));
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> completed(counter->value);
      const auto done = completed.fetch_add(count, ::cuda::memory_order_acq_rel) + count;
      if (done == static_cast<unsigned long long>(geometry.active_experts) * (first ? tiles1 : tiles2))
        atomicMax(first ? &status->backward_dw1_complete_clock : &status->backward_dw2_complete_clock,
            device_globaltimer_ns());
    }
  }
  if constexpr (Producer) {
    for (unsigned i = 0; i < Stages; ++i) {
      empty[stage].wait(phase ^ 1U);
      stage = (stage + 1U) % Stages;
      phase ^= stage == 0U;
    }
  } else if (!a.fp32_weight_gradients && threadIdx.x < N / 64U) {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  Sync::cta_sync();
  if (threadIdx.x == 256U)
    for (unsigned i = 0; i < Stages; ++i) {
      Barrier::invalidate(reinterpret_cast<std::uint64_t *>(full + i));
      Barrier::invalidate(reinterpret_cast<std::uint64_t *>(empty + i));
    }
  Sync::cta_sync();
}
