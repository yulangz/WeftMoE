    if (threadIdx.x == 0)
      atomicMax(&status->return_complete_clock, device_globaltimer_ns());
    forward_publish_progress(a, kForwardReturnObserved, scheduler[3]);

    // A CTA may begin reducing tokens whose destination-rank bulk signals are
    // already visible while other CTAs finish returning later M-blocks.  The
    // per-destination acquire below is the dependency; a grid barrier here only
    // serialized independent return and reduce work.
    auto *contribution =
        peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
    auto *source_ready =
        peer_at<unsigned long long>(a.local_base, a.outgoing_sequence_offset);
    const auto *direct_source_compact_index =
        peer_at<const unsigned long long>(
            a.local_base, a.forward_direct_source_compact_index_offset);
    const auto *direct_return_receive =
        a.use_gin
            ? reinterpret_cast<const transport::BFloat16 *>(
                  static_cast<const std::byte *>(a.gin_window_base) +
                  a.gin_return_receive_offset)
            : nullptr;
    auto *combined =
        peer_at<transport::BFloat16>(a.local_base, a.combined_offset);
    constexpr std::uint32_t kWarpsPerCta = 12U;
    constexpr std::uint32_t kWarpsPerToken = 1U;
    constexpr std::uint32_t kTokensPerCta = kWarpsPerCta / kWarpsPerToken;
    const auto warp = threadIdx.x / 32U;
    const auto lane = threadIdx.x & 31U;
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceReady, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    const auto token_warp = warp % kWarpsPerToken;
    for (std::uint64_t token =
             static_cast<std::uint64_t>(blockIdx.x) * kTokensPerCta +
             warp / kWarpsPerToken;
         token < a.dispatch.identity.num_tokens && !forward_cancelled(a);
         token += static_cast<std::uint64_t>(gridDim.x) * kTokensPerCta) {
      if constexpr (!StaticLocal)
        if (a.use_gin && lane == 0U)
          claim_completion_tasks(completion,
                                 transport_state::TaskClass::Reduce);
      if (lane == 0U && status->reduced_tokens == ~0ULL)
        atomicMin(&status->reduced_tokens, device_globaltimer_ns());
      unsigned long long lane_route = 0ULL;
      unsigned long long lane_row = 0ULL;
      unsigned int lane_direct = 0U;
      if (lane < a.dispatch.identity.top_k) {
        const auto slot =
            a.dispatch.combine_order[token * a.dispatch.identity.top_k + lane];
        lane_route = token * a.dispatch.identity.top_k + slot;
        lane_row = lane_route;
        const auto destination = a.dispatch.route_dst_rank[lane_route];
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(source_ready[destination * a.channels]);
        while (ready.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
        }
        if (a.use_gin &&
            a.dispatch.route_dst_node[lane_route] != a.node) {
          lane_row = direct_source_compact_index[lane_route];
          lane_direct = 1U;
        }
      }
      __syncwarp();
      auto *output8 = reinterpret_cast<uint4 *>(combined + token * a.hidden);
      for (std::uint64_t vector = lane + token_warp * 32U;
           vector < a.hidden / 8ULL; vector += kWarpsPerToken * 32ULL) {
        float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
             ++order) {
          const auto row = __shfl_sync(0xffffffffU, lane_row, order);
          const auto direct =
              __shfl_sync(0xffffffffU, lane_direct, order) != 0U;
          const auto *source = direct ? direct_return_receive : contribution;
          const auto packed = reinterpret_cast<const uint4 *>(
              source + row * a.hidden)[vector];
          const unsigned int words[4] = {packed.x, packed.y, packed.z,
                                         packed.w};
#pragma unroll
          for (std::uint32_t word = 0; word < 4U; ++word) {
            sums[word * 2U] = __fadd_rn(
                sums[word * 2U],
                bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
            sums[word * 2U + 1U] = __fadd_rn(
                sums[word * 2U + 1U], bf16_bits_to_float(
                                          static_cast<std::uint16_t>(
                                              words[word] >> 16U)));
          }
        }
        unsigned int words[4];
#pragma unroll
        for (std::uint32_t word = 0; word < 4U; ++word)
          words[word] =
              static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U])) |
              (static_cast<unsigned int>(
                   float_to_bf16_bits(sums[word * 2U + 1U]))
               << 16U);
        output8[vector] = make_uint4(words[0], words[1], words[2], words[3]);
      }
      __syncwarp();
      if constexpr (!StaticLocal)
        if (a.use_gin && lane == 0U)
          complete_completion_tasks(completion,
                                    transport_state::TaskClass::Reduce);
    }
    if (lane == 0U)
      atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    __syncthreads();
    grid.sync();
    forward_publish_progress(a, kForwardReduceComplete,
                             a.dispatch.identity.num_tokens);
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        a.injected_status_error_rank == static_cast<int>(a.rank))
      publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
    grid.sync();
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) {
          completion->ring_credit_restored = completion->ring_credit_expected;
          publish_elected_terminal(completion, a.epoch, status->error != 0U);
        }
        __syncthreads();
        if (blockIdx.x != 0U) {
          // Elect one lane to poll.  forward_cancelled() contains warp-level
          // collectives used by the normal worker loops, so evaluating it from
          // independently reconverging lanes at terminal can strand the rest of
          // warp 0 in WARPSYNC even after the completion epoch is visible.
          if (threadIdx.x == 0U) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
                terminal(completion->kernel_complete_epoch);
            while (terminal.load(::cuda::memory_order_acquire) < a.epoch) {
            }
          }
          __syncthreads();
          return;
        }
      }
    }
    // Only the elected terminal CTA performs the production status agreement.
    // GIN progress CTAs and compute CTAs have already proved their static task
    // spaces complete and now leave after observing kernel_complete_epoch.  In
    // particular, they never participate in a second grid-wide rendezvous whose
    // progress would otherwise depend on CTA-local control-flow state.
    publish_production_terminal_status<StaticLocal>(
        a, status, ProductionStatusPhase::Forward,
        a.production_error_records_offset,
        a.gin_production_error_records_offset,
        a.use_gin ? a.dispatch.identity.num_nodes - 1ULL : 0ULL);
    if constexpr (StaticLocal) {
      grid.sync();
    } else if (!a.use_gin) {
      grid.sync();
    } else {
      __syncthreads();
    }
    // Every local return CTA has fenced before publishing its destination
    // signal, and every local Reduce warp acquired all of its destination
    // signals before reaching this grid completion.  Consequently no peer can
    // still write this slot and no local worker can still read it; a second
    // all-rank terminal rendezvous would only duplicate those dependencies.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
      status->fc1_tasks_done = fc1_tasks;
      status->fc2_tasks_done = fc2_tasks;
      const auto fc1_k_blocks = (a.hidden + 63ULL) / 64ULL;
      const auto fc2_k_blocks = (a.intermediate + 63ULL) / 64ULL;
      status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                          fc2_tasks * fc2_k_blocks * 2ULL;
      status->wgmma_instructions =
          (fc1_tasks * 2ULL * fc1_k_blocks + fc2_tasks * fc2_k_blocks) * 2ULL *
          (64ULL / ForwardMma::K);
#endif
      status->complete_clock = device_globaltimer_ns();
    }
    forward_publish_progress(a, kForwardTerminal, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
  }
}
