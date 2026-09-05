    if constexpr (DoFc2) {
    while (true) {
      if (*task_kind_shared != kComputeTaskFc2)
        break;
#if MEGAMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U && !timeline_fc2_started) {
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2First, Compute, 0U,
                                *task_shared, a.epoch);
        timeline_fc2_started = true;
      }
#endif
      if (threadIdx.x == 0U)
        atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
      using Fc2TileMma =
          typename std::conditional<M64, TunedForwardFc2M64Mma,
                                    TunedForwardFc2Mma>::type;
      float accum[Fc2TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
        accum[index] = 0.0F;
      for (std::uint32_t wave = 0; wave < kConfiguredFc2Waves; ++wave) {
#pragma unroll
        for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
          const auto phase = wave & 1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements +
              (M64 ? 0U : warpgroup * 64U * forward_tuning::kTileK));
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kFc2BStageElements +
              (M64 ? warpgroup * (forward_tuning::kFc2TileN / 2U) *
                           forward_tuning::kTileK
                   : 0U));
          fc2_full[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t k = 0;
               k < forward_tuning::kTileK / Fc2TileMma::K; ++k) {
            const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                stage_a + k * Fc2TileMma::K, 1);
            const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                stage_b + k * Fc2TileMma::K, 1);
            Fc2TileMma::wgmma(desc_a, desc_b, accum, 1);
          }
          deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
          for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          // TMA remains four-stage asynchronous in the producer warpgroup.  The
          // math side completes one K=64 group before recycling that stage; the
          // previous wait<1> needed a second live accumulator generation and
          // made ptxas serialize every individual WGMMA instruction.
          deep_gemm::ptx::warpgroup_wait<
              forward_tuning::kWgmmaWaitGroup>();
          if constexpr (Cluster2) {
            const auto lane = threadIdx.x & 31U;
            if (lane < 2U)
              fc2_empty[stage].arrive(lane);
          } else if ((threadIdx.x & 31U) == 0U) {
            fc2_empty[stage].arrive();
          }
        }
      }
      fc2_pipeline += kFc2Iterations;

      // Reload immutable task metadata only after the WGMMA mainloop.  This is
      // the register analogue of a role union: task/return bookkeeping reuses
      // registers that the asynchronous accumulator pipeline no longer needs.
      const auto task = *task_shared;
      const auto mblock = task / fc2_ntiles;
      const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto row = a.dispatch.expert_row_prefix[expert] +
                       forward_mblock_local_row(a, expert, local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      const auto n = n_tile * forward_tuning::kFc2TileN;

      const bool full_tma_tile =
          TmaFc2Store && !M64 && valid_rows == forward_tuning::kTileM &&
                                 n + forward_tuning::kFc2TileN <= a.hidden;
      if constexpr (TmaFc2Store) {
        if (full_tma_tile) {
          // Reuse the 64 KiB FC2 input-stage union after its final consumer.
          // STSM converts the accumulator fragments directly to the 128-byte
          // swizzle expected by four 64x128 TMA stores.  Partial expert tails
          // deliberately stay on the checked scalar path: expert prefixes are
          // compact rather than padded, so a 128-row store there would clobber
          // the following expert.
          auto *output_tile = smem_a;
          const auto warp = __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
          const auto lane = threadIdx.x & 31U;
#pragma unroll
          for (std::uint32_t index = 0;
               index < Fc2TileMma::kNumAccum / 4U; ++index) {
            constexpr std::uint32_t kBankGroupBytes = 16U;
            const auto atom = index / (64U / 8U);
            const auto in_atom = index % (64U / 8U);
            const auto swizzle_row = lane;
            auto swizzle_column = in_atom;
            swizzle_column ^= swizzle_row % (128U / kBankGroupBytes);
            auto *smem_pointer =
                reinterpret_cast<std::uint8_t *>(output_tile) +
                warp * 16U * 128U + atom * 128U * 128U +
                swizzle_row * 128U + swizzle_column * kBankGroupBytes;
            deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
                __float22bfloat162_rn(
                    make_float2(accum[index * 4U], accum[index * 4U + 1U])),
                __float22bfloat162_rn(make_float2(accum[index * 4U + 2U],
                                                   accum[index * 4U + 3U])),
                smem_pointer);
          }
          cute::tma_store_fence();
          cutlass::arch::NamedBarrier::sync(kMathThreads,
                                            kMathReadyBarrier);
          if (threadIdx.x <
              forward_tuning::kFc2TileN / forward_tuning::kFc2StoreAtomN) {
            const auto atom = threadIdx.x;
            cute::SM90_TMA_STORE_2D::copy(
                output_store_map, output_tile + atom * 128U * 64U,
                static_cast<std::uint32_t>(
                    n + atom * forward_tuning::kFc2StoreAtomN),
                static_cast<std::uint32_t>(row));
            cute::tma_store_arrive();
            // The unqualified wait both protects the shared-memory union from
            // the next task and completes the async-to-generic proxy handoff
            // needed before FC2 publishes its completion counter.
            asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
          }
          cutlass::arch::NamedBarrier::sync(kMathThreads,
                                            kMathDoneBarrier);
        }
      }
      // When TMA targets a separate Return staging plane, retain the scalar
      // canonical store for local consumers. Production aliases its
      // canonical output with registered staging, so a full tile needs only
      // the TMA store; partial expert tails remain on the checked scalar path.
      if (!full_tma_tile || TmaFc2StoreToStaging) {
        const auto store_fc2 =
            [&](std::uint32_t wg, std::uint32_t m, std::uint32_t column,
                float v0, float v1) {
              const auto output_column = static_cast<std::uint64_t>(n) + column;
              const auto block_m = M64 ? m : wg * 64U + m;
              const auto output_row = row + block_m;
              if (block_m < valid_rows && output_column + 1ULL < a.hidden) {
                const auto packed = float2_to_bf16x2_bits(v0, v1);
                reinterpret_cast<std::uint32_t *>(
                    forward_output + output_row * a.hidden + output_column)[0] =
                    packed;
                if (!full_tma_tile && direct_return_staging != nullptr &&
                    direct_return_staging != forward_output &&
                    headers[output_row].source_rank / a.local_world_size !=
                        a.node)
                  reinterpret_cast<std::uint32_t *>(
                      direct_return_staging + output_row * a.hidden +
                      output_column)[0] = packed;
              } else if (block_m < valid_rows && output_column < a.hidden) {
                const auto value =
                    transport::BFloat16{float_to_bf16_bits(v0)};
                forward_output[output_row * a.hidden + output_column] = value;
                if (!full_tma_tile && direct_return_staging != nullptr &&
                    direct_return_staging != forward_output &&
                    headers[output_row].source_rank / a.local_world_size !=
                        a.node)
                  direct_return_staging[output_row * a.hidden + output_column] =
                      value;
              }
            };
      if constexpr (M64)
        visit_m64_accumulator<Fc2TileMma>(accum, store_fc2);
      else
        visit_m128_accumulator<Fc2TileMma>(accum, store_fc2);
      }
      // N-tiles synchronize only through this GPU-local completion chain.
      // A device fence is sufficient here; the final N-tile acquires the
      // chain and performs one system fence for the whole M-block below.
      __threadfence();
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            complete(fc1_done[mblock]);
        const auto old =
            complete.fetch_add(1ULL << 16U, ::cuda::memory_order_acq_rel);
        *return_mblock_shared =
            (((old >> 16U) & 0xffffULL) + 1ULL == fc2_ntiles) ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
        if constexpr (!StaticLocal) {
          if (a.use_gin)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc2);
        }
#endif
      }
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);

      if (threadIdx.x == 0U)
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2TileReady, Compute, 0U, task,
                                a.epoch);

      if (*return_mblock_shared != 0U) {
        // Thread 0's acq_rel RMW above acquired every earlier N-tile release.
        // Publish that transitive chain, plus this CTA's stores, to NVLink/GIN
        // once per M-block before any Return-ready counter becomes visible.
        __threadfence_system();
        if (threadIdx.x == 0U)
          *return_context_masks_shared = 0ULL;
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);

        // Exact masks prevent seven unrelated context owners from rescanning
        // this M-block. Build them with one header load per row and one shared
        // OR per warp instead of a 128-load scalar loop on the last FC2 tile.
        std::uint32_t remote_context_bit = 0U;
        std::uint32_t local_context_bit = 0U;
        if (threadIdx.x < valid_rows) {
          const auto source = headers[row + threadIdx.x].source_rank;
          if (source < a.world) {
            const auto context = source % a.local_world_size;
            if (source / a.local_world_size != a.node)
              remote_context_bit = 1U << context;
            else if (a.forward_relay_local_return)
              local_context_bit = 1U << context;
          }
        }
        const auto warp_remote =
            __reduce_or_sync(0xffffffffU, remote_context_bit);
        const auto warp_local =
            __reduce_or_sync(0xffffffffU, local_context_bit);
        if ((threadIdx.x & 31U) == 0U)
          atomicOr(return_context_masks_shared,
                   static_cast<unsigned long long>(warp_remote) |
                       (static_cast<unsigned long long>(warp_local) << 32U));
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);

        if (threadIdx.x == 0U) {
          const auto finished =
              atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
          if (finished == forward_mblocks) {
            status->fc2_complete_clock = device_globaltimer_ns();
            forward_publish_progress(a, kForwardFc2Complete, fc2_tasks);
          }
          if constexpr (!StaticLocal) {
            if (a.use_gin || (a.forward_stream_local_dispatch &&
                              a.forward_relay_local_return)) {
              // Lane 0 publishes one maximal contiguous descriptor to the
              // fixed FULL GIN owners. No context state or network register
              // live range enters this WGMMA kernel.
              forward_return_detail::issue_direct_full_mblock(
                  a, scheduler, row, valid_rows,
                  *return_context_masks_shared);
              MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2MblockReturnReady,
                                      Compute, 0U, mblock, a.epoch);
            }
          } else if (a.forward_stream_local_dispatch &&
                     a.forward_relay_local_return) {
            // One-node split Forward uses the identical immutable M-block
            // directory, but only its node-local context mask is consumed.
            // The compute CTA returns to the FC2-priority queue immediately;
            // permanent Relay CTAs own all peer copies and readiness stores.
            forward_return_detail::issue_direct_full_mblock(
                a, scheduler, row, valid_rows,
                *return_context_masks_shared);
            MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2MblockReturnReady,
                                    Compute, 0U, mblock, a.epoch);
          }
        }
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);

        // In the split GIN path Relay CTAs consume the same immutable M-block
        // descriptor as RDMA and own all node-local NVLink Return work.  Keep
        // this compatibility copy only for the non-GIN local P2P path; a math
        // CTA must become schedulable again immediately after publication.
        const bool split_relay_return =
            (a.use_gin || a.forward_stream_local_dispatch) &&
            a.forward_relay_local_return;
        if (!a.use_socket && !split_relay_return) {
          for (std::uint32_t source = threadIdx.x; source < a.world;
               source += kMathThreads)
            return_rows_by_source[source] = 0U;
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
          for (std::uint32_t local_row = 0; local_row < valid_rows;
               ++local_row) {
            const auto return_row = row + local_row;
            const auto header = headers[return_row];
            if (a.use_gin && header.source_rank / a.local_world_size != a.node)
              continue;
            auto *source_contribution = peer_at<transport::BFloat16>(
                a.peer_bases[header.source_rank], a.contribution_offset);
            const auto *source128 = reinterpret_cast<const uint4 *>(
                forward_output + return_row * a.hidden);
            auto *target128 = reinterpret_cast<uint4 *>(
                source_contribution + header.route_index * a.hidden);
            const auto vector_count = a.hidden / 8ULL;
            for (std::uint64_t vector = threadIdx.x; vector < vector_count;
                 vector += kMathThreads)
              target128[vector] = source128[vector];
            if (threadIdx.x == 0U)
              ++return_rows_by_source[header.source_rank];
          }
          __threadfence_system();
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);
          // Preserve the rank-wide signal below as the terminal protocol, but
          // expose each local-node contribution independently for Reduce.
          for (std::uint32_t local_row = threadIdx.x;
               local_row < valid_rows; local_row += kMathThreads) {
            const auto return_row = row + local_row;
            const auto header = headers[return_row];
            if (!a.use_gin ||
                header.source_rank / a.local_world_size == a.node) {
              auto *source_route_ready = peer_at<unsigned long long>(
                  a.peer_bases[header.source_rank],
                  a.return_ready_epoch_offset);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  route_ready(source_route_ready[header.route_index]);
              if (a.forward_dynamic_reduce_queue)
                forward_reduce_detail::publish_ready_route(
                    a, a.peer_bases[header.source_rank], header.route_index,
                    3U);
              else
                route_ready.store(a.epoch, ::cuda::memory_order_release);
            }
          }
          for (std::uint32_t source_rank = threadIdx.x; source_rank < a.world;
               source_rank += kMathThreads) {
            if (a.use_gin && source_rank / a.local_world_size != a.node)
              continue;
            const auto rows = return_rows_by_source[source_rank];
            if (rows == 0U)
              continue;
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                completed(scheduler[4U + source_rank]);
            const auto old =
                completed.fetch_add(rows, ::cuda::memory_order_acq_rel);
            const auto expected = old >> 32U;
            const auto actual = (old & 0xffffffffULL) + rows;
            if (actual == expected) {
              auto *source_signal = peer_at<unsigned long long>(
                  a.peer_bases[source_rank], a.outgoing_sequence_offset);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  ready(source_signal[a.rank * a.channels]);
              ready.store(a.epoch, ::cuda::memory_order_release);
            }
          }
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
        }
      }

#if MEGAMOE_DEBUG_VALIDATE
      if (a.use_gin && threadIdx.x == 0U &&
          blockIdx.x < transport_state::kForwardDebugCtas) {
        completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
            (5ULL << 32U) | fc2_pipeline;
        completion->forward_fc2_cta_tiles_completed[blockIdx.x] = task + 1ULL;
      }
#endif
      select_compute_task<Cluster2, StaticLocal, DoFc1, DoFc2, Partition>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
          pair_operand_shared, pair_round, fc2_streak);
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
    }
    }

    if (*task_kind_shared == kComputeTaskDone)
      break;
  }
#if MEGAMOE_DEVICE_TIMELINE
  if (threadIdx.x == 0U && timeline_fc2_started)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2Last, Compute, 0U, fc2_tasks,
                            a.epoch);
#endif
  if constexpr (Cluster2)
    cute::cluster_sync();
}

} // namespace forward_compute_detail
