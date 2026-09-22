#pragma once

namespace static_v4 {

__device__ __noinline__ void network_route_exchange(const ForwardKernelArgs &a,
                                                    unsigned char *scratch) {
  constexpr auto kMathThreads = forward_tuning::kMathWarpgroups * 128U;
  constexpr auto kControlThreads =
      forward_tuning::kUnifiedThreads - kMathThreads;
  const auto routes = a.dispatch.sizes.route_count;
  const auto experts = a.dispatch.identity.num_experts;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
  auto *prefix = source_pointer<std::uint64_t>(a, a.rank);

  if (threadIdx.x < kMathThreads) {
    const auto tid = std::uint64_t(blockIdx.x) * kMathThreads + threadIdx.x;
    const auto stride = std::uint64_t(gridDim.x) * kMathThreads;
    auto *input = source_pointer<uint4>(a, a.rank, a.static_network.input);
    for (auto vector = tid;
         vector < a.dispatch.identity.num_tokens * a.hidden / 8ULL;
         vector += stride)
      input[vector] = reinterpret_cast<const uint4 *>(a.input)[vector];
    build_combine_order(a, tid, stride);
    collective_scope_sync<forward_tuning::kUnifiedThreads,
                          kCollectiveJoinBarrier>();
    collective_scope_sync<forward_tuning::kUnifiedThreads,
                          kCollectiveJoinBarrier>();
    return;
  }

  const auto control_thread = threadIdx.x - kMathThreads;
  const auto tid = std::uint64_t(blockIdx.x) * kControlThreads + control_thread;
  const auto stride = std::uint64_t(gridDim.x) * kControlThreads;
  auto *histogram = reinterpret_cast<unsigned *>(scratch);
  auto *counts = const_cast<std::uint64_t *>(a.dispatch.source_expert_count);
  auto *source_gates = source_pointer<float>(a, a.rank, a.static_network.gates);
  for (unsigned expert = control_thread; expert < experts;
       expert += kControlThreads)
    histogram[expert] = 0U;
  collective_scope_sync<kControlThreads, kCollectiveScopeBarrier>();
  for (auto route = tid; route < routes; route += stride) {
    source_gates[route] = a.gates[route];
    const auto raw = a.selected_experts[route];
    const bool invalid = raw < 0 || unsigned(raw) >= experts;
    if (invalid)
      atomicExch(reinterpret_cast<unsigned long long *>(&runtime->error), 1ULL);
    atomicAdd(histogram + (invalid ? 0U : unsigned(raw)), 1U);
  }
  collective_scope_sync<kControlThreads, kCollectiveScopeBarrier>();
  for (unsigned expert = control_thread; expert < experts;
       expert += kControlThreads) {
    const auto count = histogram[expert];
    if (count)
      histogram[expert] = static_cast<unsigned>(
          atomicAdd(reinterpret_cast<unsigned long long *>(counts + expert),
                    static_cast<unsigned long long>(count)));
  }
  grid_sync<kMathThreads, kControlThreads>(a);
  if (tid == 0ULL) {
    prefix[0] = 0ULL;
    for (unsigned expert = 0; expert < experts; ++expert)
      prefix[expert + 1U] = prefix[expert] + counts[expert];
    prefix[experts + 1U] = runtime->error;
  }
  grid_sync<kMathThreads, kControlThreads>(a);

  const auto experts_per_node = experts / a.dispatch.identity.num_nodes;
  auto *order =
      source_pointer<std::uint32_t>(a, a.rank, a.static_network.routes);
  auto *return_index = const_cast<std::uint64_t *>(a.dispatch.local_offset);
  for (auto route = tid; route < routes; route += stride) {
    const auto raw = a.selected_experts[route];
    const auto expert =
        raw < 0 || unsigned(raw) >= experts ? 0U : unsigned(raw);
    const auto index = prefix[expert] + atomicAdd(histogram + expert, 1U);
    order[index] = static_cast<std::uint32_t>(route);
    return_index[route] =
        index - prefix[expert / experts_per_node * experts_per_node];
  }
  collective_scope_sync<forward_tuning::kUnifiedThreads,
                        kCollectiveJoinBarrier>();
  grid_sync<kMathThreads, kControlThreads>(a);

  if (blockIdx.x < kNetworkContexts && control_thread == 0U) {
    const auto base = source_offset(a, a.rank);
    for (unsigned node = 0; node < a.dispatch.identity.num_nodes; ++node) {
      if (node == a.node)
        continue;
      network_put(a, node, base, base,
                  (experts + 2ULL) * sizeof(std::uint64_t));
      network_put(a, node, base + a.static_network.routes,
                  base + a.static_network.routes,
                  routes * sizeof(std::uint32_t));
      network_put(a, node, base + a.static_network.gates,
                  base + a.static_network.gates, routes * sizeof(float));
      network_put(a, node, base + a.static_network.input,
                  base + a.static_network.input,
                  a.dispatch.identity.num_tokens * a.hidden *
                      sizeof(transport::BFloat16));
    }
  }
  network_barrier<kMathThreads, kControlThreads>(a);

  for (auto index = tid; index < std::uint64_t(a.world) * owned;
       index += stride) {
    const auto source = unsigned(index / owned);
    const auto expert = a.rank * owned + unsigned(index % owned);
    const auto *source_counts = source_prefix(a, source);
    a.fused_local_expert_count[index] =
        source_counts[expert + 1U] - source_counts[expert];
  }
  // All peers have the same capacity and source prefixes. Reject a route or
  // capacity error collectively before any canonical arena is written.
  if (blockIdx.x == 0U) {
    for (unsigned destination = control_thread; destination < a.world;
         destination += kControlThreads) {
      std::uint64_t rows = 0ULL;
      for (unsigned source = 0; source < a.world; ++source) {
        const auto *source_counts = source_prefix(a, source);
        rows += source_counts[(destination + 1U) * owned] -
                source_counts[destination * owned];
        if (source_counts[experts + 1U])
          atomicExch(reinterpret_cast<unsigned long long *>(&runtime->error),
                     1ULL);
      }
      if (rows > a.max_destination_rows)
        atomicExch(reinterpret_cast<unsigned long long *>(&runtime->error),
                   1ULL);
    }
  }
  grid_sync<kMathThreads, kControlThreads>(a);
  build_geometry<kMathThreads, kControlThreads>(a);
  collective_scope_sync<forward_tuning::kUnifiedThreads,
                        kCollectiveJoinBarrier>();
}

} // namespace static_v4
