#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  const auto fc1_ntiles =
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  const auto forward_mblocks = scheduler[3];
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  const auto fc2_ntiles =
      static_cast<std::uint32_t>(
          (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
          forward_tuning::kFc2TileN);
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  const bool rdma_worker_cta = KernelRole != ForwardKernelRole::ComputeOnly &&
                               remote_backend_enabled<StaticLocal>(a.use_gin) &&
                               blockIdx.x < forward_rdma_ctas;
  [[maybe_unused]] const bool relay_worker_cta =
      KernelRole != ForwardKernelRole::ComputeOnly &&
      remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x >= forward_relay_cta_begin &&
      blockIdx.x < forward_compute_cta_begin;
  // Keep transport and WGMMA roles disjoint for the complete persistent
  // interval.  In particular, a Relay CTA must not execute generic/proxy
  // traffic and then enter the dynamic WGMMA register-allocation protocol.
  const bool transport_worker_cta = rdma_worker_cta || relay_worker_cta;

  // Roles are immutable for the Dispatch/FC/streaming-Return interval:
  // [0,8) owns GIN contexts, [8,16) owns NVLink Relay/directory work, and the
  // remaining CTAs own WGMMA.  In particular no Relay CTA enters the GEMM
  // path and no RDMA CTA copies the Dispatch payload into expert arenas.
  if constexpr (KernelRole != ForwardKernelRole::TransportOnly) {
    if (!transport_worker_cta) {
      if (threadIdx.x == 256U) {
        for (std::uint32_t stage = 0; stage < 3U; ++stage) {
          fc1_full_barriers[stage].init(1);
          fc1_empty_barriers[stage].init(8);
        }
        for (std::uint32_t stage = 0; stage < 4U; ++stage) {
          fc2_full_barriers[stage].init(1);
          fc2_empty_barriers[stage].init(8);
        }
        cutlass::arch::fence_barrier_init();
      }
      __syncthreads();
      // Do not emit SETMAXNREG in the monolithic compatibility kernel. Its
      // static allocation is already 168 registers/thread (64,512
      // registers/CTA), and ptxas cannot lower the transport-capable producer
      // warpgroup to 48. Keeping the ignored alloc/dealloc pair still
      // introduces warpgroup dependency edges and can strand the TMA producer
      // behind math warpgroups waiting on its barrier. The split compute kernel
      // restores the intended 224/48 redistribution once its producer path has
      // a genuinely small live set.
      std::uint32_t pipeline_iteration = 0;
      __shared__ unsigned long long task_shared;
      __shared__ unsigned int claim_shared;
      auto *fc1_claims = claims + forward_mblocks;
      auto *fc2_claims = fc1_claims + fc1_tasks;
      constexpr unsigned int kTaskChunk = MEGAMOE_DEBUG_VALIDATE ? 1U : 4U;
      unsigned int task_in_chunk = kTaskChunk;
#if MEGAMOE_DEVICE_TIMELINE
      bool timeline_fc1_started = false;
#endif
#if MEGAMOE_DEBUG_VALIDATE
      if (completion != nullptr && threadIdx.x == 0U) {
        completion->fc1_scheduler_value =
            a.cancel_epoch
                ? atomicAdd_system(
                      const_cast<unsigned long long *>(a.cancel_epoch), 0ULL)
                : ~0ULL;
        completion->bulk_local_ready = 10ULL;
      }
      __syncthreads();
#endif
      while (!forward_cancelled(a)) {
        if (task_in_chunk == kTaskChunk) {
          if (threadIdx.x == 0U)
            task_shared = atomicAdd(
                &scheduler[0], static_cast<unsigned long long>(kTaskChunk));
          __syncthreads();
          task_in_chunk = 0U;
        }
        const auto task = task_shared + task_in_chunk++;
        if (task >= fc1_tasks)
          break;
        const auto mblock = task / fc1_ntiles;
        const auto n_tile = static_cast<std::uint32_t>(task % fc1_ntiles);
        const auto mapping = claims[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        if (threadIdx.x == 0U) {
          claim_shared =
              atomicCAS(&fc1_claims[task], 0ULL, 1ULL) == 0ULL ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
          if (completion != nullptr) {
            completion->fc1_scheduler_value = task;
            completion->bulk_local_ready = 11ULL;
          }
#endif
        }
        __syncthreads();
        if (claim_shared == 0U)
          continue;
        if (expert >= a.dispatch.sizes.owned_expert_count)
          break;
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            claim_completion_tasks(completion,
                                   transport_state::TaskClass::Fc1);
        const auto row =
            a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
        const auto remaining_rows =
            a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
        const auto valid_rows = static_cast<std::uint32_t>(
            remaining_rows < 128ULL ? remaining_rows : 128ULL);
        wait_forward_mblock_rows<StaticLocal>(
            a, expert, static_cast<std::uint64_t>(local_mblock) * 128ULL,
            valid_rows, completion);
        __syncthreads();
#if MEGAMOE_DEVICE_TIMELINE
        if (!timeline_fc1_started) {
          MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstMtileReady, Compute, 0U,
                                  task, a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1First, Compute, 0U, task,
                                  a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1ClaimStart, Compute, 0U, task,
                                  a.epoch);
          timeline_fc1_started = true;
        }
#endif
        const auto n = n_tile * 128U;
        {
          float accum_a[ForwardMma::kNumAccum];
          float accum_b[ForwardMma::kNumAccum];
          forward_wgmma_swiglu_tile(
              x_map, w1_map, static_cast<std::uint32_t>(row),
              static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n),
              static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                         a.intermediate + n),
              static_cast<std::uint32_t>(a.hidden), smem_a, smem_b,
              fc1_full_barriers, fc1_empty_barriers, accum_a, accum_b,
              pipeline_iteration);
          visit_forward_accumulator_pair(
              accum_a, accum_b,
              [&](std::uint32_t warpgroup, std::uint32_t m,
                  std::uint32_t column, float av0, float av1, float bv0,
                  float bv1) {
                const auto block_m = warpgroup * 64U + m;
                const auto output_column =
                    static_cast<std::uint64_t>(n) + column;
                if (block_m < valid_rows &&
                    output_column + 1ULL < a.intermediate &&
                    (a.intermediate & 1ULL) == 0ULL) {
                  const auto output_index =
                      (row + block_m) * a.intermediate + output_column;
                  const auto gate = canonical_gates[row + block_m];
                  // Each WGMMA lane owns two adjacent BF16 outputs.  Publish
                  // the pair as one naturally aligned store for gated U.
                  const auto a_pair = float2_to_bf16x2_bits(av0, av1);
                  const auto b_pair = float2_to_bf16x2_bits(bv0, bv1);
                  const auto a_values = bf16x2_bits_to_float2(a_pair);
                  const auto b_values = bf16x2_bits_to_float2(b_pair);
                  if (ab_cache != nullptr) {
                    reinterpret_cast<std::uint32_t *>(
                        ab_cache)[output_index / 2ULL] = a_pair;
                    const auto b_index =
                        a.max_destination_rows * a.intermediate +
                        output_index;
                    reinterpret_cast<std::uint32_t *>(
                        ab_cache)[b_index / 2ULL] = b_pair;
                  }
                  reinterpret_cast<std::uint32_t *>(s)[output_index / 2ULL] =
                      float2_to_bf16x2_bits(
                          forward_compute_detail::
                              gated_swiglu_fc2_input(a_values.x, b_values.x,
                                                    gate),
                          forward_compute_detail::
                              gated_swiglu_fc2_input(a_values.y, b_values.y,
                                                    gate));
                } else if (block_m < valid_rows) {
                  // Retain checked scalar stores for non-main odd-N fixtures.
                  const float av[2] = {av0, av1};
                  const float bv[2] = {bv0, bv1};
                  const auto gate = canonical_gates[row + block_m];
#pragma unroll
                  for (std::uint32_t value = 0; value < 2U; ++value) {
                    const auto tail_column = output_column + value;
                    if (tail_column < a.intermediate) {
                      const auto output_index =
                          (row + block_m) * a.intermediate + tail_column;
                      const auto a_bits = float_to_bf16_bits(av[value]);
                      const auto b_bits = float_to_bf16_bits(bv[value]);
                      if (ab_cache != nullptr) {
                        ab_cache[output_index] =
                            transport::BFloat16{a_bits};
                        ab_cache[a.max_destination_rows * a.intermediate +
                                 output_index] =
                            transport::BFloat16{b_bits};
                      }
                      s[output_index] = transport::BFloat16{float_to_bf16_bits(
                          forward_compute_detail::
                              gated_swiglu_fc2_input_from_bf16(
                                  a_bits, b_bits, gate))};
                    }
                  }
                }
              });
        }
        // FC2 may be claimed by another CTA as soon as this tile increments the
        // m-block counter. Order every lane's S/checkpoint stores first; a
        // fence from only the counter lane would not publish the other lanes'
        // writes.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              ready(fc1_done[mblock]);
          ready.fetch_add(1ULL, ::cuda::memory_order_release);
        }
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc1);
        if (threadIdx.x == 0U) {
          atomicExch(&fc1_claims[task], 2ULL);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0)
        atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
      MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1Last, Compute, 0U, fc1_tasks,
                              a.epoch);
      forward_publish_progress(a, kForwardFc1Complete, fc1_tasks);

      // FC1 transactions issued by this CTA are complete. FC2 consumes its
      // separately initialized barrier set, so no transaction-barrier object is
      // reinitialized in place. There is deliberately no grid-wide FC1/FC2
      // barrier: late FC1 tiles and ready FC2 tiles overlap across CTAs.
      __syncthreads();
      pipeline_iteration = 0;

      task_in_chunk = kTaskChunk;
      if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            directory_ready(scheduler[6ULL + 2ULL * a.world]);
        while (directory_ready.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
          if (threadIdx.x == 0U)
            __nanosleep(256U);
          __syncthreads();
        }
      }
      while (!forward_cancelled(a)) {
        if (task_in_chunk == kTaskChunk) {
          if (threadIdx.x == 0U)
            task_shared = atomicAdd(
                &scheduler[1], static_cast<unsigned long long>(kTaskChunk));
          __syncthreads();
          task_in_chunk = 0U;
        }
        const auto task = task_shared + task_in_chunk++;
        if (task >= fc2_tasks)
          break;
        const auto mblock = task / fc2_ntiles;
        const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
        if (threadIdx.x == 0U) {
          claim_shared =
              atomicCAS(&fc2_claims[task], 0ULL, 1ULL) == 0ULL ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
          if (completion != nullptr)
            completion->fc2_scheduler_value = task;
#endif
        }
        __syncthreads();
        if (claim_shared == 0U)
          continue;
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            claim_completion_tasks(completion,
                                   transport_state::TaskClass::Fc2);
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas) {
          completion->forward_fc2_cta_tiles_started[blockIdx.x] = task + 1ULL;
          completion->forward_fc2_cta_last_task[blockIdx.x] = task;
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (1ULL << 32U) | pipeline_iteration;
        }
#endif
        if (threadIdx.x == 0)
          atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
        const auto mapping = claims[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            fc1_ready(fc1_done[mblock]);
        while ((fc1_ready.load(::cuda::memory_order_acquire) & 0xffffffffULL) <
               fc1_ntiles) {
        }
        __syncthreads();
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas)
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (2ULL << 32U) | pipeline_iteration;
#endif
        const auto row =
            a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
        const auto remaining_rows =
            a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
        const auto valid_rows = static_cast<std::uint32_t>(
            remaining_rows < 128ULL ? remaining_rows : 128ULL);
        const auto n = n_tile * 256U;
        float accum[ForwardMmaWide::kNumAccum];
        forward_wgmma_tile<ForwardMmaWide, 256U>(
            s_map, w2_map, static_cast<std::uint32_t>(row),
            static_cast<std::uint32_t>(expert * a.hidden + n),
            static_cast<std::uint32_t>(a.intermediate), smem_a, smem_b,
            fc2_full_barriers, fc2_empty_barriers, accum, pipeline_iteration, 0U
#if MEGAMOE_DEBUG_VALIDATE
            ,
            a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas
                ? &completion->forward_fc2_cta_producer_checkpoint[blockIdx.x]
                : nullptr,
            a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas
                ? &completion->forward_fc2_cta_math0_checkpoint[blockIdx.x]
                : nullptr,
            a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas
                ? &completion->forward_fc2_cta_math1_checkpoint[blockIdx.x]
                : nullptr
#endif
        );
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas)
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (3ULL << 32U) | pipeline_iteration;
#endif
        visit_forward_accumulator_wide(accum, [&](std::uint32_t warpgroup,
                                                  std::uint32_t m,
                                                  std::uint32_t column,
                                                  float v0, float v1) {
          const auto output_column = static_cast<std::uint64_t>(n) + column;
          const auto block_m = warpgroup * 64U + m;
          if (block_m < valid_rows && output_column < a.hidden)
            forward_output[(row + block_m) * a.hidden + output_column] =
                transport::BFloat16{float_to_bf16_bits(v0)};
          if (block_m < valid_rows && output_column + 1U < a.hidden)
            forward_output[(row + block_m) * a.hidden + output_column + 1U] =
                transport::BFloat16{float_to_bf16_bits(v1)};
        });
        // Publish this N tile's canonical stores before exposing completion.
        // The CTA that observes the final N tile owns the M-block return; other
        // CTAs can continue FC2 for unrelated blocks in parallel.
        __threadfence();
        __syncthreads();
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas)
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (4ULL << 32U) | pipeline_iteration;
#endif
        __shared__ unsigned int return_mblock_shared;
        if (threadIdx.x == 0) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              complete(fc1_done[mblock]);
          const auto old =
              complete.fetch_add(1ULL << 16U, ::cuda::memory_order_acq_rel);
          return_mblock_shared =
              (((old >> 16U) & 0xffffULL) + 1ULL == fc2_ntiles) ? 1U : 0U;
        }
        __syncthreads();
        if (return_mblock_shared != 0U) {
          // Preserve the monolithic local/socket completion accounting. The
          // cross-node build always uses the split FULL descriptor path.
          __threadfence_system();
          __syncthreads();
          if (threadIdx.x == 0U) {
            const auto finished =
                atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
            if (finished == forward_mblocks) {
              status->fc2_complete_clock = device_globaltimer_ns();
              forward_publish_progress(a, kForwardFc2Complete, fc2_tasks);
            }
          }
          __syncthreads();
        }
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2TileReady, Compute, 0U, task,
                                a.epoch);
        if (return_mblock_shared != 0U &&
            !remote_backend_enabled<StaticLocal>(a.use_socket)) {
          // Only lane 0 produces and consumes the peer-row audit count. Keeping
          // it in CTA shared storage made independent CTAs alias through the
          // cluster-shared address space under racecheck. A thread-local scalar
          // has the same audit semantics and no cross-thread lifetime.
          std::uint32_t return_peer_rows = 0U;
          if (threadIdx.x == 0)
            atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
          for (std::uint32_t source = threadIdx.x; source < a.world;
               source += blockDim.x)
            return_rows_by_source[source] = 0U;
          __syncthreads();
          for (std::uint32_t local_row = 0; local_row < valid_rows;
               ++local_row) {
            const auto return_row = row + local_row;
            const auto header = headers[return_row];
            if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
                header.source_rank / a.local_world_size != a.node)
              continue;
            auto *source_contribution = peer_at<transport::BFloat16>(
                a.peer_bases[header.source_rank], a.contribution_offset);
            const auto *source128 = reinterpret_cast<const uint4 *>(
                forward_output + return_row * a.hidden);
            auto *target128 = reinterpret_cast<uint4 *>(
                source_contribution + header.route_index * a.hidden);
            const auto vector_count = a.hidden / 8ULL;
            for (std::uint64_t vector = threadIdx.x; vector < vector_count;
                 vector += blockDim.x * 2ULL) {
              target128[vector] = source128[vector];
              const auto second = vector + blockDim.x;
              if (second < vector_count)
                target128[second] = source128[second];
            }
            if (threadIdx.x == 0) {
              ++return_rows_by_source[header.source_rank];
              if (header.source_rank != a.rank)
                ++return_peer_rows;
            }
          }
          if (threadIdx.x == 0 && return_peer_rows != 0U)
            atomicAdd(&status->actual_return_peer_bytes,
                      static_cast<unsigned long long>(return_peer_rows) *
                          a.hidden * sizeof(transport::BFloat16));
          __threadfence_system();
          __syncthreads();
          for (std::uint32_t source_rank = threadIdx.x; source_rank < a.world;
               source_rank += blockDim.x) {
            if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
                source_rank / a.local_world_size != a.node)
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
          __syncthreads();
        }
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc2);
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas) {
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (5ULL << 32U) | pipeline_iteration;
          completion->forward_fc2_cta_tiles_completed[blockIdx.x] = task + 1ULL;
        }
#endif
        if (threadIdx.x == 0U) {
          atomicExch(&fc2_claims[task], 2ULL);
        }
        __syncthreads();
      }
      // Return chunks have a single owner: their GIN context CTA. Allowing
      // completed compute CTAs to race those owners for WAITING -> PACKING made
      // the release kernel timing-dependent and could strand a context in its
      // terminal wait. The fixed progress team is sufficient to gather while
      // FC2 continues and keeps pack, issue, and completion ordered in one CTA.
    }
  }
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    publish_local_structured_error(status, a.rank, 90U, 1U, 1ULL, 0ULL);
  }
#endif
  if constexpr (KernelRole != ForwardKernelRole::ComputeOnly) {
    if (remote_backend_enabled<StaticLocal>(a.use_socket)) {
      grid.sync();
      if (blockIdx.x == 0 && threadIdx.x == 0 &&
          a.injected_status_error_rank == static_cast<int>(a.rank))
        publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
      grid.sync();
      publish_production_terminal_status<StaticLocal>(
          a, status, ProductionStatusPhase::Forward,
          a.production_error_records_offset,
          a.gin_production_error_records_offset, 0ULL);
      grid.sync();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
        status->fc1_tasks_done = fc1_tasks;
        status->fc2_tasks_done = fc2_tasks;
        const auto fc1_k_blocks = (a.hidden + 63ULL) / 64ULL;
        const auto fc2_k_blocks = (a.intermediate + 63ULL) / 64ULL;
        status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                            fc2_tasks * fc2_k_blocks * 2ULL;
        status->wgmma_instructions =
            (fc1_tasks * 2ULL * fc1_k_blocks + fc2_tasks * fc2_k_blocks) *
            2ULL * (64ULL / ForwardMma::K);
#endif
        status->return_complete_clock = device_globaltimer_ns();
        status->complete_clock = status->return_complete_clock;
      }
      forward_publish_progress(a, kForwardFc2Complete, scheduler[3]);
      forward_publish_progress(a, kForwardTerminal, a.epoch);
      return;
    }
