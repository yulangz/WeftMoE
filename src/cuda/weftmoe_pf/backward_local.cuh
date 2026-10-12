#pragma once

namespace weftmoe_pf::backward_stream {

#ifndef WEFTMOE_PF_BWD_CANONICAL_INTERLEAVE
#define WEFTMOE_PF_BWD_CANONICAL_INTERLEAVE 0
#endif

__device__ __forceinline__ unsigned compact_local_row(
    const ForwardKernelArgs &f, const BackwardKernelArgs &a, const Task &t) {
  return unsigned(a.dispatch.expert_row_prefix[t.expert]) +
         (t.block - unsigned(f.ring.expert_blocks[t.expert])) * kM;
}

__device__ __forceinline__ transport::BFloat16 *local_dx_output(
    const ForwardKernelArgs &f, const BackwardKernelArgs &a, const Task &t,
    unsigned row) {
  const auto identity = f.checkpoint.headers[compact_local_row(f, a, t) + row];
  return peer_at<transport::BFloat16>(f.peer_bases[identity.source_rank],
                                      f.contribution_offset) +
         (std::uint64_t(identity.topk_slot) * f.max_tokens_per_rank +
          identity.local_token_id) * f.hidden;
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
// dW2 already needs the complete compact dY cache. Pull directly into it using
// the tape-owned headers, never a physical ring directory left by another layer.
__device__ __forceinline__ void pull_local_canonical(
    const ForwardKernelArgs &f, const BackwardKernelArgs &a, Shared &s) {
  const auto warp = (threadIdx.x - 288U) / 32U;
  const auto lane = threadIdx.x % 32U;
  const auto rows = a.dispatch.expert_row_prefix[a.dispatch.sizes.owned_expert_count];
  const auto *mapping = peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  auto *ready = backward_pipeline_detail::scheduler_words(a) +
                directory_layout(a).dispatch_arrived;
  unsigned phase = 0U, peer_rows = 0U, expert = 0U;
  std::uint64_t begin = 0U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  for (auto linear = std::uint64_t(blockIdx.x) * local::kPullWarps + warp;
       linear < rows; linear += gridDim.x * local::kPullWarps) {
    auto row = linear;
    if constexpr (WEFTMOE_PF_BWD_CANONICAL_INTERLEAVE) {
      while (linear >= begin + f.ring.expert_rows[expert])
        begin += f.ring.expert_rows[expert++];
      unsigned source, ordinal;
      decode_source_row(f, expert, linear - begin, source, ordinal);
      unsigned prior = 0U;
      for (auto peer = lane; peer < source; peer += 32U)
        prior += unsigned(f.fused_local_expert_count[std::uint64_t(peer) * owned + expert]);
      row = a.dispatch.expert_row_prefix[expert] +
            __reduce_add_sync(0xffffffffU, prior) + ordinal;
    }
    if (!lane) {
      const auto identity = f.checkpoint.headers[row];
      const auto *input = peer_at<const char>(f.peer_bases[identity.source_rank],
                                             f.source_input_offset) +
                          identity.local_token_id * f.hidden * 2U;
      auto *output = reinterpret_cast<char *>(a.dr) + row * f.hidden * 2U;
      for (unsigned offset = 0U; offset < f.hidden * 2U; offset += 4096U) {
        const auto bytes = min(4096U, unsigned(f.hidden * 2U) - offset);
        cute::SM90_BULK_COPY_G2S::copy(
            input + offset, reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]),
            s.pull[warp], bytes);
        s.pull_full[warp].arrive_and_expect_tx(bytes);
        s.pull_full[warp].wait(phase);
        phase ^= 1U;
        cute::SM90_BULK_COPY_S2G::copy(s.pull[warp], output + offset, bytes);
        cute::tma_store_arrive();
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
      reduce_add_release_device(ready + mapping[row] / kM, 1ULL);
      peer_rows += identity.source_rank != f.rank;
    }
    __syncwarp();
  }
  if (!lane) {
    auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    if (peer_rows)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                std::uint64_t(peer_rows) * f.hidden * 2ULL);
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
  }
}
#endif
} // namespace weftmoe_pf::backward_stream
