#pragma once

namespace weftmoe_pf::backward_return {

__device__ __forceinline__ unsigned expert(const BackwardKernelArgs &a,
                                           std::uint64_t row) {
  unsigned first = 0U, end = a.dispatch.sizes.owned_expert_count;
  while (first + 1U < end) {
    const auto middle = (first + end) / 2U;
    if (row < a.dispatch.expert_row_prefix[middle])
      end = middle;
    else
      first = middle;
  }
  return first;
}

__device__ __forceinline__ backward_detail::ReturnBucket &bucket(
    const BackwardKernelArgs &a, std::uint64_t window, unsigned source) {
  const auto &r = a.return_ring;
  return r.buckets[window * r.shape.sources + source % r.shape.sources];
}

__device__ __forceinline__ std::uint64_t contribution_row(
    const BackwardKernelArgs &a, std::uint64_t route) {
  const auto &r = a.return_ring;
  return route % (std::uint64_t(r.shape.tokens) * r.contribution_depth *
                  r.shape.top_k);
}

__device__ __forceinline__ std::uint64_t payload_row(
    const BackwardKernelArgs &a, std::uint64_t row) {
  return row % (std::uint64_t(a.return_ring.depth) * backward_tuning::kTileM);
}

// Called by the existing one-CTA Backward setup, before its peer barrier.
__device__ __forceinline__ void initialize(const BackwardKernelArgs &a) {
  const auto &r = a.return_ring;
  const auto experts = a.dispatch.sizes.owned_expert_count;
  for (auto i = std::uint64_t(threadIdx.x); i < r.windows * experts;
       i += blockDim.x)
    r.expert_rows[i] = 0U;
  for (auto i = std::uint64_t(threadIdx.x); i < r.windows * r.shape.sources;
       i += blockDim.x)
    r.buckets[i] = {};
  for (unsigned i = threadIdx.x; i < r.depth; i += blockDim.x)
    r.slots[i].consumed = 0U;
  for (unsigned i = threadIdx.x; i < a.mblocks; i += blockDim.x)
    r.du_window[i] = UINT32_MAX;
  for (auto i = std::uint64_t(threadIdx.x); i <= r.windows; i += blockDim.x)
    r.du_offsets[i] = 0U;
  auto &source = a.peer_payloads[a.rank];
  for (unsigned i = threadIdx.x; i < r.contribution_depth; i += blockDim.x)
    source.consumed[i] = 0U;
  if (threadIdx.x == 0U)
    source.return_state->reserved = 0U;
}

__global__ void count(BackwardKernelArgs a) {
  const auto &r = a.return_ring;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto *mapping = peer_at<const unsigned>(a.local_base,
                                                a.compact_to_padded_offset);
  for (auto row = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       row < a.dispatch.sizes.incoming_row_count;
       row += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto header = headers[row];
    const auto window = r.shape.window(header.source_rank,
                                       header.route_index / r.shape.top_k);
    const auto e = expert(a, row);
    atomicMin(r.du_window + mapping[row] / backward_tuning::kTileM, unsigned(window));
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  r.expert_rows + window * a.dispatch.sizes.owned_expert_count +
                  e),
              1ULL);
    atomicAdd(&bucket(a, window, header.source_rank).rows, 1U);
  }
}

__global__ void geometry(BackwardKernelArgs a) {
  const auto &r = a.return_ring;
  const auto experts = a.dispatch.sizes.owned_expert_count;
  const auto *padded = peer_at<const unsigned long long>(a.local_base,
                                                        a.padded_expert_offsets_offset);
  const auto mblocks = unsigned(padded[experts] / backward_tuning::kTileM);
  for (unsigned block = threadIdx.x; block < mblocks; block += blockDim.x)
    atomicAdd(r.du_offsets + r.du_window[block], 1U);
  __syncthreads();
  if (threadIdx.x == 0U) {
    unsigned prefix = 0U;
    for (std::uint64_t w = 0; w <= r.windows; ++w) {
      const auto count = r.du_offsets[w];
      r.du_offsets[w] = prefix;
      prefix += count;
    }
    std::uint64_t blocks = 0U;
    for (std::uint64_t w = 0; w < r.windows; ++w) {
      r.window_prefix[w] = blocks;
      unsigned local = 0U;
      for (unsigned e = 0; e < experts; ++e) {
        r.expert_blocks[w * (experts + 1U) + e] = local;
        const auto rows = r.expert_rows[w * experts + e];
        const auto tiles = static_forward::ceil_div(rows, backward_tuning::kTileM);
        for (unsigned tile = 0; tile < tiles; ++tile)
          r.blocks[blocks + local + tile] = {
              0U, e, static_forward::block_rows(rows, tile, backward_tuning::kTileM),
              unsigned(w), 0U, 0U};
        local += tiles;
        r.expert_rows[w * experts + e] = 0U; // Scatter cursor after the prefix.
      }
      r.expert_blocks[w * (experts + 1U) + experts] = local;
      blocks += local;
    }
    r.window_prefix[r.windows] = blocks;
    r.state->blocks = blocks;
  }
  // Each source queue is window-major, including empty windows. This fixed
  // boundary also flushes a short last packet without waiting for future work.
  for (unsigned source = threadIdx.x; source < a.world; source += blockDim.x) {
    if (source / a.local_world_size == a.node)
      continue;
    auto &queue = return_queue_detail::queues(a)[1U + source];
    auto begin = queue.begin;
    std::uint64_t packets = 0U;
    for (auto w = std::uint64_t(source / r.shape.sources); w < r.windows;
         w += r.shape.groups()) {
      auto &b = bucket(a, w, source);
      b.begin = begin;
      b.packet_begin = packets;
      begin += b.rows;
      packets += static_forward::ceil_div(b.rows, communication::kChunkRows);
    }
    queue.reserved = packets;
  }
  __syncthreads();
  // Prioritize the earliest consumer window, not the expert-major tape
  // address. Otherwise a late dU tile can hold up every source's first credit.
  for (unsigned block = threadIdx.x; block < mblocks; block += blockDim.x) {
    const auto position = atomicAdd(r.du_offsets + r.du_window[block], 1U);
    r.du_order[position] = block;
  }
}

__global__ void rows(BackwardKernelArgs a) {
  const auto &r = a.return_ring;
  const auto experts = a.dispatch.sizes.owned_expert_count;
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  for (auto row = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       row < a.dispatch.sizes.incoming_row_count;
       row += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto header = headers[row];
    const auto window = r.shape.window(header.source_rank,
                                       header.route_index / r.shape.top_k);
    const auto e = expert(a, row);
    const auto ordinal = atomicAdd(reinterpret_cast<unsigned long long *>(
        r.expert_rows + window * experts + e), 1ULL);
    const auto first = r.window_prefix[window] +
                       r.expert_blocks[window * (experts + 1U) + e];
    unsigned position = UINT32_MAX;
    if (header.source_rank / a.local_world_size != a.node) {
      auto &b = bucket(a, window, header.source_rank);
      position = b.begin + atomicAdd(&b.cursor, 1U);
    }
    r.rows[first * backward_tuning::kTileM + ordinal] = {
        unsigned(row), header.source_rank, unsigned(header.route_index), position};
    r.row_blocks[row] = first + ordinal / backward_tuning::kTileM;
    if (header.source_rank / a.local_world_size == a.node)
      atomicOr(&r.blocks[first + ordinal / backward_tuning::kTileM].local_sources,
                1U << (header.source_rank % a.local_world_size));
  }
}

} // namespace weftmoe_pf::backward_return
