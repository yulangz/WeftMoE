// Adapted from DeepSeek MegaMoE (MIT), see LICENSE-DeepSeek.
#pragma once

#include "fused/comm/barrier.cuh"
#include "fused/ptx/utils.cuh"

namespace weftmoe::sm100::comm {

template <uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kNumRanks, uint32_t kNumSMs,
          uint32_t kNumDispatchThreads, uint32_t kDispatchBarrierIdx, uint32_t kDispatchGridSyncIndex,
          bool kSave>
CUTLASS_DEVICE void dispatch_pf(const layout::MegaMoEBuffer& buffer, const layout::SymBuffer<kNumRanks>& sym_buffer,
                                uint32_t num_tokens, uint32_t* counts, const layout::BF16Checkpoint& checkpoint) {
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
    const auto& workspace = buffer.workspace;
    const uint32_t sm_idx = blockIdx.x, thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync(), lane_idx = ptx::get_lane_idx();
    WEFT_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
    constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
    const auto read_topk_idx = [&](const auto& process) {
        // TODO: figure out better unrolling
        // Now, `unroll` is better than `unroll 8`
        #pragma unroll
        for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
             i < num_tokens;
             i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
            // Allocate slots for each token-topk
            int expert_idx = -1;
            if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                expert_idx = static_cast<int>(
                    buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                if (expert_idx >= 0)
                    process(i * kNumTopk + lane_idx, expert_idx);
            }
            __syncwarp();
        }
    };

    // Count experts' tokens
    read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
       atomicAdd_block(counts + expert_idx, 1);
    });
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

    // Get SM offset (~6.5 us)
    #pragma unroll
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
        const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(counts[i]);
        counts[i] = static_cast<uint32_t>(
            ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
    }
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

    // Write source indices (~2 us with 512 tokens)
    read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
        const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
        const auto dst_slot_idx = kSave ? checkpoint.source_ordinals[token_topk_idx] : atomicAdd_block(counts + expert_idx, 1);
        const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
            expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
        *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
    });

    // Grid sync
    comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
        workspace, sm_idx, thread_idx,
        [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
    );

    // Write expert count
    if (sm_idx == 0) {
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const auto dst_rank_idx = i / kNumExpertsPerRank;
            const auto dst_local_expert_idx = i % kNumExpertsPerRank;
            const auto expert_status = *workspace.get_expert_send_count_ptr(i);
            *sym_buffer.map(
                workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                dst_rank_idx) = expert_status & 0xffffffff;
            ptx::atomic_add_sys(
                sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                expert_status);
        }
    }
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
}

// State carried by a dispatch warp as it traverses the received row directory.
template <uint32_t kNumRanks>
struct PullCursor {
    static constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
    int current_expert_idx = -1;
    uint32_t expert_start_idx = 0, expert_end_idx = 0, expert_pool_block_offset = 0;
    uint32_t stored_rank_count[kNumRanksPerLane] = {};
    uint32_t current_rank_in_expert_idx, token_idx_in_rank, token_idx_in_expert;

    template <uint32_t BLOCK_M, uint32_t kNumExpertsPerRank, typename Scheduler>
    CUTLASS_DEVICE bool locate_round_robin(const Scheduler& scheduler, uint32_t token_idx) {
        const auto& workspace = scheduler.workspace;
        const uint32_t lane_idx = ptx::get_lane_idx();
        // Advance expert until within the range
        int old_expert_idx = current_expert_idx;
        while (token_idx >= expert_end_idx) {
            if (++ current_expert_idx >= kNumExpertsPerRank)
                break;

            // Update pool block offset for the new expert
            expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

            // Move start and end to the next expert
            expert_start_idx = expert_end_idx;
            expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
        }

        // Finish all tokens
        if (current_expert_idx >= kNumExpertsPerRank)
            return false;

        // Load per-rank counts when expert changes
        if (old_expert_idx != current_expert_idx) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                // TODO: this is not coalesced
                stored_rank_count[i] = j < kNumRanks ?
                    static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
            }
        }

        // Round-robin rank selection via iterative min-peeling
        uint32_t remaining[kNumRanksPerLane];
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
            remaining[i] = stored_rank_count[i];
        uint32_t offset = 0;
        token_idx_in_expert = token_idx - expert_start_idx;
        uint32_t slot_idx = token_idx_in_expert;
        while (true) {
            // Compute active count and min across all ranks
            // NOTES: reduce within each lane first, then warp-reduce once
            uint32_t num_actives_in_lane = 0;
            uint32_t min_in_lane = 0xffffffff;
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                num_actives_in_lane += remaining[i] > 0;
                if (remaining[i] > 0)
                    min_in_lane = cute::min(min_in_lane, remaining[i]);
            }
            const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
            const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

            // Hit in the current round
            const uint32_t num_round_tokens = length * num_active_ranks;
            if (slot_idx < num_round_tokens) {
                const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                uint32_t num_seen_ranks = 0;
                current_rank_in_expert_idx = 0;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                    const uint32_t num_active_lanes = __popc(mask);
                    if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                        current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                    num_seen_ranks += num_active_lanes;
                }
                token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                break;
            }

            // Move into the next round
            slot_idx -= num_round_tokens;
            offset += length;
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] -= cute::min(remaining[i], length);
        }
        return true;
    }
};

} // namespace weftmoe::sm100::comm
