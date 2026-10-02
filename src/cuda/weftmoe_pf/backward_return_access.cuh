#pragma once

namespace weftmoe_pf::backward_return {

__device__ __forceinline__ unsigned reduce_workers(const BackwardKernelArgs &a) {
  return gridDim.x - (a.return_ring.depth ? a.gin_sender_count : 0U);
}

__device__ __forceinline__ std::uint64_t reduce_first(
    const BackwardKernelArgs &a, unsigned warps, unsigned warp) {
  const auto first = a.return_ring.depth ? a.gin_sender_count : 0U;
  return blockIdx.x < first ? a.dispatch.identity.num_tokens
                           : std::uint64_t(blockIdx.x - first) * warps + warp;
}

// One nonblocking source owner runs independently of dX and transport. The
// first D windows are free; subsequent reuse needs every token's Reduce credit.
__device__ __forceinline__ void advance_source(const BackwardKernelArgs &a) {
  const auto &r = a.return_ring;
  auto &peer = a.peer_payloads[a.rank];
  const auto windows = static_forward::ceil_div(a.dispatch.identity.num_tokens,
                                                r.shape.tokens);
  auto next = peer.return_state->reserved;
  while (next < windows) {
    const auto target = (next / r.contribution_depth) * r.shape.tokens;
    if (ring::acquire(peer.consumed[next % r.contribution_depth]) < target)
      break;
    ring::publish<::cuda::thread_scope_system>(peer.return_state->reserved, ++next);
  }
}

__device__ __forceinline__ bool local_reserved(const BackwardKernelArgs &a,
                                               unsigned block) {
  const auto &r = a.return_ring;
  auto sources = r.blocks[block].local_sources;
  const auto window = r.blocks[block].window / r.shape.groups();
  while (sources) {
    const auto local = unsigned(__ffs(sources) - 1);
    const auto &peer = a.peer_payloads[a.node * a.local_world_size + local];
    if (ring::acquire<::cuda::thread_scope_system>(peer.return_state->reserved) <= window)
      return false;
    sources &= sources - 1U;
  }
  return true;
}

__device__ __forceinline__ bool slot_free(const BackwardKernelArgs &a,
                                          unsigned block) {
  const auto &r = a.return_ring;
  return ring::acquire(r.slots[block % r.depth].consumed) >=
         std::uint64_t(block / r.depth) * backward_tuning::kTileM;
}

__device__ __forceinline__ void retire_rows(const BackwardKernelArgs &a,
                                            unsigned block, unsigned rows) {
  ring::arrive(a.return_ring.slots[block % a.return_ring.depth].consumed, rows);
}

__device__ __forceinline__ bool input_ready(const BackwardKernelArgs &a,
    unsigned block) {
  const auto &tile = a.return_ring.blocks[block];
  return slot_free(a, block) && ring::acquire(tile.input_rows) == tile.rows &&
         local_reserved(a, block);
}

// The existing epilogue fence and cluster join precede this publication.
// Each valid canonical row contributes exactly once to its immutable window
// tile; warp aggregation avoids one contended atomic per row.
__device__ __forceinline__ void publish_input(const BackwardKernelArgs &a,
    unsigned compact, unsigned rows) {
  if (threadIdx.x >= rows)
    return;
  const auto block = a.return_ring.row_blocks[compact + threadIdx.x];
  const auto group = __match_any_sync(__activemask(), block);
  if ((threadIdx.x & 31U) == unsigned(__ffs(group) - 1))
    ring::arrive(a.return_ring.blocks[block].input_rows, __popc(group));
}

// Canonical dAB remains intact for the later full-expert dW1 GEMM. Only the
// dX input is gathered into one reusable M tile per resident cluster.
__device__ __forceinline__ unsigned gather_row() {
  return (blockIdx.x / 2U) * backward_tuning::kTileM;
}

template <class Synchronization>
__device__ __forceinline__ void gather(const BackwardKernelArgs &a,
    unsigned block, unsigned cluster_rank, Synchronization &sync) {
  if constexpr (Synchronization::kMathEnabled) {
    constexpr auto threads = Synchronization::kWorkWarps * 32U;
    if (threadIdx.x < threads) {
      constexpr auto vectors = 2ULL * backward_tuning::kMainIntermediate / 8U;
      const auto *mapping = peer_at<const unsigned>(a.local_base,
                                                    a.compact_to_padded_offset);
      const auto *rows = a.return_ring.rows +
                         std::uint64_t(block) * backward_tuning::kTileM;
      auto *output = reinterpret_cast<uint4 *>(a.return_ring.gathered_dab) +
          std::uint64_t(gather_row()) * vectors;
      for (auto i = std::uint64_t(cluster_rank) * threads + threadIdx.x;
           i < backward_tuning::kTileM * vectors; i += 2U * threads) {
        const auto row = i / vectors;
        output[i] = row < a.return_ring.blocks[block].rows
            ? reinterpret_cast<const uint4 *>(a.dab)[
                  std::uint64_t(mapping[rows[row].compact]) * vectors + i % vectors]
            : uint4{0U, 0U, 0U, 0U};
      }
      __threadfence();
    }
  }
  sync.cluster_sync();
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// The wire packet never crosses a source-token window. In particular a short
// final packet must not wait for a future window that needs its payload credit.
__device__ __forceinline__ activation_ring_detail::Transfer packet(
    const BackwardKernelArgs &a, unsigned channel, unsigned long long sequence,
    unsigned long long position) {
  const auto source = activation_ring_detail::peer(a, channel);
  const auto &q = return_queue_detail::queues(a)[1U + source];
  const auto index = position / communication::kChunkRows;
  if (index >= q.reserved)
    return {channel, 0U, 1U, sequence, q.begin + q.capacity};
  const auto &r = a.return_ring;
  std::uint64_t first = 0U, end = r.windows / r.shape.groups();
  while (first + 1U < end) {
    const auto middle = (first + end) / 2U;
    const auto &b = bucket(a, middle * r.shape.groups() + source / r.shape.sources,
                           source);
    if (b.packet_begin <= index)
      first = middle;
    else
      end = middle;
  }
  const auto &b = bucket(a, first * r.shape.groups() + source / r.shape.sources,
                         source);
  const auto row = (index - b.packet_begin) * communication::kChunkRows;
  return {channel, unsigned(min(1ULL * communication::kChunkRows, b.rows - row)),
          index + communication::kReturnPorts >= q.reserved, sequence, b.begin + row};
}

__device__ __forceinline__ bool receive_reserved(const BackwardKernelArgs &a,
    unsigned channel, const communication::ChunkHeader *chunk) {
  if (!a.return_ring.depth || channel < a.gin_rings.dispatch_channels || !chunk->rows)
    return true;
  const auto route = reinterpret_cast<const communication::ReturnRow *>(chunk + 1)->route;
  const auto window = route / a.dispatch.identity.top_k / a.return_ring.shape.tokens;
  return ring::acquire(a.peer_payloads[a.rank].return_state->reserved) > window;
}

} // namespace weftmoe_pf::backward_return
