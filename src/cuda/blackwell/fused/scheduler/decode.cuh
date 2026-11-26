#pragma once

#include "fused/comm/dispatch.cuh"
#include "fused/ptx/utils.cuh"

namespace weftmoe::sm100::decode {

// Slot ordinals are private to a source rank and support repeated expert IDs.
template <bool SingleCTA, uint32_t Experts, uint32_t TopK, uint32_t Ranks,
          uint32_t SMs, uint32_t Threads, uint32_t BarrierIdx, uint32_t GridSyncIdx>
CUTLASS_DEVICE void dispatch(const layout::MegaMoEBuffer& buffer,
                             const layout::SymBuffer<Ranks>& sym, uint32_t tokens,
                             uint32_t* counts, uint32_t epoch) {
    constexpr uint32_t owned = Experts / Ranks;
    const uint32_t tid = threadIdx.x, cta = blockIdx.x;
    const auto& ws = buffer.workspace;
    auto sync = [] { __barrier_sync_count(BarrierIdx, Threads); };
    // Short traces use a single shared histogram. For longer traces, assign
    // disjoint slots to CTAs and reserve contiguous ordinals per warp/expert.
    if (SingleCTA ? cta == 0 : tid < 32) {
        const uint32_t first = SingleCTA ? tid : cta * 32 + tid;
        const uint32_t stride = SingleCTA ? Threads : SMs * 32;
        for (uint32_t start = first - (tid % 32); start < tokens * TopK; start += stride) {
            const uint32_t slot = start + tid % 32;
            const uint32_t e = slot < tokens * TopK
                ? buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[slot] : Experts;
            const uint32_t valid = __ballot_sync(0xffffffff, e < Experts);
            if (e < Experts) {
                uint32_t ordinal;
                if constexpr (SingleCTA) {
                    ordinal = atomicAdd_block(counts + e, 1u);
                } else {
                    const uint32_t same = __match_any_sync(valid, e);
                    const uint32_t leader = __ffs(same) - 1;
                    const uint32_t lane = tid % 32;
                    uint32_t base = 0;
                    if (lane == leader)
                        base = ptx::atomic_add(ws.get_expert_send_count_ptr(e), static_cast<uint64_t>(__popc(same)));
                    ordinal = __shfl_sync(same, base, leader) + __popc(same & ((1u << lane) - 1));
                }
                auto* entry = ws.get_src_token_topk_idx_ptr(e % owned, sym.rank_idx, ordinal);
                *sym.map(entry, e / owned) = slot;
            }
        }
    }
    if constexpr (SingleCTA)
        sync();
    else
        comm::grid_sync<SMs, GridSyncIdx>(ws, cta, tid, sync);
    if (cta == 0) {
        for (uint32_t e = tid; e < Experts; e += Threads) {
            const uint32_t n = SingleCTA ? counts[e] : *ws.get_expert_send_count_ptr(e);
            *sym.map(ws.get_expert_recv_count_ptr(sym.rank_idx, e % owned), e / owned) = n;
        }
        sync();
        if (tid < Ranks)
            ptx::st_rel_sys(sym.map(ws.get_peer_grid_idx_ptr(sym.rank_idx), tid), epoch);
    }

    // One release per destination publishes source indices, counts, and the
    // staged input. Each workspace advances the epoch once per collective call.
    const uint32_t lane = ptx::get_lane_idx();
    for (uint32_t first = 0; first < Ranks; first += 32) {
        const uint32_t peer = first + lane;
        comm::wait_until([&] {
            return __all_sync(0xffffffff, peer >= Ranks || ptx::ld_acq_sys(ws.get_peer_grid_idx_ptr(peer)) == epoch);
        }, [&] { printf("WeftMoE decode source readiness timeout: rank=%u\n", sym.rank_idx); });
    }
    __syncwarp();
    // Reuse the source histogram as the CTA-local receive directory. Every
    // source overwrites its entire vector (including zero-count experts).
    for (uint32_t expert = tid; expert < owned; expert += Threads) {
        uint32_t total = 0;
        #pragma unroll
        for (uint32_t peer = 0; peer < Ranks; ++peer)
            total += *ws.get_expert_recv_count_ptr(peer, expert);
        counts[expert] = total;
    }
}

// For a warp-sized directory each lane owns an expert. Two prefix lookups
// map a flat row to its expert and to a source slot within that expert.
template <uint32_t Ranks, typename Scheduler>
CUTLASS_DEVICE bool locate(const Scheduler& scheduler, uint32_t token, comm::PullCursor<Ranks>& cursor) {
    const uint32_t lane = ptx::get_lane_idx();
    const uint32_t rows = scheduler.stored_num_tokens_per_expert[0];
    const uint32_t end = math::warp_inclusive_sum(rows, lane);
    const uint32_t owners = __ballot_sync(0xffffffff, end > token);
    if (!owners) return false;
    const uint32_t expert = __ffs(owners) - 1;
    cursor.current_expert_idx = expert;
    cursor.expert_end_idx = ptx::exchange(end, expert);
    cursor.token_idx_in_expert = token - ptx::exchange(end - rows, expert);
    cursor.expert_pool_block_offset = scheduler.get_pool_block_offset(expert);
    const uint32_t count = lane < Ranks ? *scheduler.workspace.get_expert_recv_count_ptr(lane, expert) : 0;
    const uint32_t rank_end = math::warp_inclusive_sum(count, lane);
    cursor.current_rank_in_expert_idx = __ffs(__ballot_sync(0xffffffff, rank_end > cursor.token_idx_in_expert)) - 1;
    cursor.token_idx_in_rank = cursor.token_idx_in_expert -
        ptx::exchange(rank_end - count, cursor.current_rank_in_expert_idx);
    return true;
}

// The combine signal acknowledges the epoch even when that source
// sent no tokens to this owner. The following launch may then publish new slots.
template <uint32_t Ranks>
CUTLASS_DEVICE void wait_for_owners(const layout::Workspace& ws, uint32_t epoch) {
    for (uint32_t base = 0; base < Ranks; base += 32) {
        const uint32_t peer = base + ptx::get_lane_idx();
        comm::wait_until([&] {
            return __all_sync(0xffffffff, peer >= Ranks ||
                ptx::ld_acq_sys(ws.get_combine_ready_grid_idx_ptr(peer)) == epoch);
        }, [&] { printf("WeftMoE decode owner acknowledgement timeout\n"); });
    }
}

// A warp reduces 256 adjacent BF16 columns. Short batches can use many
// warps per token, without a serial TMA load/barrier pair for every slot.
template <uint32_t Hidden, uint32_t TopK, uint32_t Owned, uint32_t SMs, uint32_t Warps>
CUTLASS_DEVICE void combine(const layout::MegaMoEBuffer& buffer, void* output,
                            uint32_t tokens, uint32_t warp, uint32_t epoch) {
    constexpr uint32_t columns = 32 * 8;
    static_assert(Hidden % columns == 0);
    const uint32_t lane = ptx::get_lane_idx();
    for (uint32_t piece = warp * SMs + blockIdx.x; piece < tokens * (Hidden / columns);
         piece += Warps * SMs) {
        const uint32_t token = piece / (Hidden / columns);
        const uint32_t vector = piece % (Hidden / columns) * 32 + lane;
        const int e = lane < TopK ? buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token * TopK + lane] : -1;
        const uint32_t mask = __ballot_sync(0xffffffff, e >= 0);
        const auto* ready = buffer.workspace.get_combine_ready_grid_idx_ptr(e >= 0 ? e / Owned : 0);
        comm::wait_until([&] { return __all_sync(0xffffffff, e < 0 || ptx::ld_acq_sys(ready) == epoch); },
                         [&] { printf("WeftMoE decode combine timeout: token=%u\n", token); });
        __syncwarp();
        float2 acc[4] = {};
        #pragma unroll
        for (uint32_t slot = 0; slot < TopK; ++slot) {
            if (mask & (1u << slot)) {
                auto ptr = buffer.combine_token_buffer.get_rank_buffer(slot).get_data_buffer(token).get_base_ptr<uint4>();
                const uint4 v = ptx::ld_evict_first(ptr + vector);
                const auto* pairs = reinterpret_cast<const nv_bfloat162*>(&v);
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    ptx::accumulate(acc[j], pairs[j]);
            }
        }
        uint4 out;
        auto* pairs = reinterpret_cast<nv_bfloat162*>(&out);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            pairs[j] = __float22bfloat162_rn(acc[j]);
        static_cast<uint4*>(output)[static_cast<size_t>(token) * (Hidden / 8) + vector] = out;
    }
}

} // namespace weftmoe::sm100::decode
