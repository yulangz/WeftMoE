// A stable counting pass over fixed route chunks. Block-local radix sorting
// gives each route its ordinal inside the chunk; an expert-wise scan of chunk
// counts supplies the global ordinal. No atomic operation assigns row order.
constexpr unsigned kNotifyChunkRoutes = 1024;
struct LocalNotifyArgs {
  const std::int32_t *selected;
  std::uint64_t tokens, routes;
  unsigned experts, top_k, nodes, mask_words;
  const unsigned *expert_to_rank, *rank_to_node, *rank_to_local;
  unsigned *expert_id, *dst_rank, *dst_node, *dst_local, *combine_order;
  std::uint64_t *masks, *histogram, *packet_count, *local_offset;
  std::uint64_t *chunk_counts;
  DeviceStatus *status;
};

__device__ void local_notify(LocalNotifyArgs a) {
  auto grid = cooperative_groups::this_grid();
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;
  const auto chunks = (a.routes + kNotifyChunkRoutes - 1) / kNotifyChunkRoutes;
  for (auto i = std::uint64_t(tid); i < chunks * a.experts; i += stride)
    a.chunk_counts[i] = 0;
  for (auto i = tid; i < a.nodes; i += stride)
    a.packet_count[i] = 0;
  if (tid == 0)
    *a.status = DeviceStatus{~0ULL, ~0ULL};
  grid.sync();
  for (auto token = std::uint64_t(tid); token < a.tokens; token += stride)
    describe_token(a.selected, token, a.tokens, a.top_k, a.experts,
                   a.expert_to_rank, a.rank_to_node, a.nodes, a.mask_words,
                   a.combine_order, a.masks, a.packet_count, a.status);

  using Sort = cub::BlockRadixSort<std::uint64_t, 256, 4>;
  __shared__ union {
    typename Sort::TempStorage sort;
    std::uint64_t keys[kNotifyChunkRoutes];
    typename cub::BlockScan<unsigned long long, 256>::TempStorage scan;
  } shared;
  unsigned bits = 0;
  for (unsigned value = a.experts; value; value >>= 1)
    ++bits;
  for (auto chunk = std::uint64_t(blockIdx.x); chunk < chunks;
       chunk += gridDim.x) {
    std::uint64_t keys[4];
#pragma unroll
    for (unsigned item = 0; item < 4; ++item) {
      const auto within = threadIdx.x * 4 + item;
      const auto route = chunk * kNotifyChunkRoutes + within;
      unsigned expert = a.experts;
      if (route < a.routes) {
        const auto raw = a.selected[route];
        if (raw >= 0 && unsigned(raw) < a.experts) {
          expert = unsigned(raw);
          const auto destination = a.expert_to_rank[expert];
          a.expert_id[route] = expert;
          a.dst_rank[route] = destination;
          a.dst_node[route] = a.rank_to_node[destination];
          a.dst_local[route] = a.rank_to_local[destination];
        } else {
          atomicMin(&a.status->out_of_range_packed,
                    (static_cast<unsigned long long>(route) << 32) |
                        unsigned(raw));
          a.expert_id[route] = a.experts;
          a.dst_rank[route] = a.dst_node[route] = a.dst_local[route] = 0;
          a.local_offset[route] = 0;
        }
      }
      keys[item] = (std::uint64_t(expert) << 10) | within;
    }
    Sort(shared.sort).Sort(keys, 0, bits + 10);
    __syncthreads();
#pragma unroll
    for (unsigned item = 0; item < 4; ++item)
      shared.keys[threadIdx.x * 4 + item] = keys[item];
    __syncthreads();
#pragma unroll
    for (unsigned item = 0; item < 4; ++item) {
      const auto position = threadIdx.x * 4 + item;
      const auto key = keys[item];
      const auto expert = unsigned(key >> 10);
      if (expert < a.experts) {
        unsigned lo = 0, hi = position;
        while (lo < hi) {
          const auto mid = (lo + hi) / 2;
          if ((shared.keys[mid] >> 10) < expert)
            lo = mid + 1;
          else
            hi = mid;
        }
        a.local_offset[chunk * kNotifyChunkRoutes + (key & 1023)] =
            position - lo;
        if (position + 1 == kNotifyChunkRoutes ||
            (shared.keys[position + 1] >> 10) != expert)
          a.chunk_counts[std::uint64_t(expert) * chunks + chunk] =
              position + 1 - lo;
      }
    }
    __syncthreads();
  }
  grid.sync();
  for (unsigned expert = blockIdx.x; expert < a.experts; expert += gridDim.x) {
    unsigned long long carry = 0;
    for (std::uint64_t first = 0; first < chunks; first += 256) {
      const auto chunk = first + threadIdx.x;
      const auto index = std::uint64_t(expert) * chunks + chunk;
      unsigned long long prefix, total;
      cub::BlockScan<unsigned long long, 256>(shared.scan)
          .ExclusiveSum(chunk < chunks ? a.chunk_counts[index] : 0ULL, prefix,
                        total);
      if (chunk < chunks)
        a.chunk_counts[index] = carry + prefix;
      carry += total;
      __syncthreads();
    }
    if (!threadIdx.x)
      a.histogram[expert] = carry;
  }
  grid.sync();
  for (auto route = std::uint64_t(tid); route < a.routes; route += stride) {
    const auto expert = a.expert_id[route];
    if (expert < a.experts)
      a.local_offset[route] += a.chunk_counts[std::uint64_t(expert) * chunks +
                                              route / kNotifyChunkRoutes];
  }
}

__global__ void __launch_bounds__(256) local_notify_kernel(LocalNotifyArgs a) {
  local_notify(a);
}
