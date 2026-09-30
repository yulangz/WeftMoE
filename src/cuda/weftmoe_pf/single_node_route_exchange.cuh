#pragma once

namespace weftmoe_pf {

// Publish the compact control plane used by destination-pull MoE. Every
// source owns one stable slot sequence per global expert and writes only the
// source route id into the owning rank's symmetric table. X and gates are
// staged once into source-local symmetric storage, then destination control
// warps pull the routed rows.
__device__ __noinline__ void
route_exchange(const ForwardKernelArgs &a, unsigned char *scratch) {
  constexpr auto kMathThreads = forward_tuning::kMathWarpgroups * 128U;
  constexpr auto kControlThreads =
      forward_tuning::kUnifiedThreads - kMathThreads;
  static_assert(kControlThreads == 128U);
  const auto routes = a.dispatch.sizes.route_count;
  const auto experts = a.dispatch.identity.num_experts;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  auto *send_counts =
      const_cast<std::uint64_t *>(a.dispatch.source_expert_count);
  auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
  auto *cta_expert_count = reinterpret_cast<std::uint32_t *>(scratch);

  if (threadIdx.x < kMathThreads) {
    const auto tid = static_cast<std::uint64_t>(blockIdx.x) * kMathThreads +
                     threadIdx.x;
    const auto stride =
        static_cast<std::uint64_t>(gridDim.x) * kMathThreads;
    auto *source_input = peer_at<transport::BFloat16>(
        a.local_base, a.source_input_offset);
    const auto input_vectors =
        a.dispatch.identity.num_tokens * a.hidden / 8ULL;
    for (auto vector = tid; vector < input_vectors; vector += stride)
      reinterpret_cast<uint4 *>(source_input)[vector] =
          reinterpret_cast<const uint4 *>(a.input)[vector];

    // The fixed reduction order depends only on source routes, so the idle
    // math role can build it while control publishes the route table.
    build_combine_order(a, tid, stride);
    // Control may publish the rank-ready signal only after both the route
    // table and source activation staging are complete.
    collective_scope_sync<forward_tuning::kUnifiedThreads,
                          kCollectiveJoinBarrier>();
    collective_scope_sync<forward_tuning::kUnifiedThreads,
                          kCollectiveJoinBarrier>();
    return;
  }

  const auto control_thread = threadIdx.x - kMathThreads;
  const auto tid = static_cast<std::uint64_t>(blockIdx.x) * kControlThreads +
                   control_thread;
  const auto stride =
      static_cast<std::uint64_t>(gridDim.x) * kControlThreads;
  if (tid == 0ULL) {
    runtime->sizes.route_count = routes;
    runtime->sizes.owned_expert_count = owned;
    runtime->sizes.packet_count = 0ULL;
    runtime->sizes.fanout_count = 0ULL;
    runtime->sizes.outgoing_signal_count = 0U;
    runtime->sizes.incoming_signal_count = 0U;
    runtime->wave_count = 0ULL;
  }
  for (auto expert = static_cast<std::uint64_t>(control_thread);
       expert < experts; expert += kControlThreads)
    cta_expert_count[expert] = 0U;
  collective_scope_sync<kControlThreads, kCollectiveScopeBarrier>();

  // Pass one forms a CTA-local histogram. One global atomic per
  // (CTA,expert) reserves a disjoint source slot interval; pass two then
  // reproduces the route enumeration to assign exact offsets.
  for (auto route = tid; route < routes; route += stride) {
    peer_at<float>(a.local_base, a.fused_source_gates_offset)[route] =
        a.gates[route];
    const auto raw = a.selected_experts[route];
    const auto invalid =
        raw < 0 || static_cast<std::uint32_t>(raw) >= experts;
    if (invalid)
      atomicExch(reinterpret_cast<unsigned long long *>(&runtime->error),
                 1ULL);
    const auto expert = invalid ? 0U : static_cast<std::uint32_t>(raw);
    atomicAdd(cta_expert_count + expert, 1U);
  }

  collective_scope_sync<kControlThreads, kCollectiveScopeBarrier>();

  for (auto expert = static_cast<std::uint64_t>(control_thread);
       expert < experts; expert += kControlThreads) {
    const auto count = cta_expert_count[expert];
    if (count)
      cta_expert_count[expert] = static_cast<std::uint32_t>(atomicAdd(
          reinterpret_cast<unsigned long long *>(send_counts + expert),
          static_cast<unsigned long long>(count)));
  }
  collective_scope_sync<kControlThreads, kCollectiveScopeBarrier>();

  for (auto route = tid; route < routes; route += stride) {
    const auto raw = a.selected_experts[route];
    const auto expert =
        raw < 0 || static_cast<std::uint32_t>(raw) >= experts
            ? 0U
            : static_cast<std::uint32_t>(raw);
    const auto source_slot = atomicAdd(cta_expert_count + expert, 1U);
    const auto destination = expert / owned;
    const auto local_expert = expert % owned;
    source_routes(a, destination, a.rank, local_expert)[source_slot] =
        static_cast<std::uint32_t>(route);
  }

  collective_scope_sync<forward_tuning::kUnifiedThreads,
                        kCollectiveJoinBarrier>();

  // First finish all source-local reservations and route-table stores.  CTA 0
  // then publishes this source's final counts directly into each destination's
  // source-major table, matching the route-table direction.  The following
  // release/acquire boundary makes both remote payloads visible together.
  grid_sync<kMathThreads, kControlThreads>(a);
  if (blockIdx.x == 0U) {
    for (auto expert = static_cast<std::uint64_t>(control_thread);
         expert < experts; expert += kControlThreads) {
      const auto destination = static_cast<std::uint32_t>(expert / owned);
      const auto local = static_cast<std::uint32_t>(expert % owned);
      source_counts(a, destination, a.rank)[local] = send_counts[expert];
    }
  }
  rank_barrier<kMathThreads, kControlThreads, false, false>(a);

  build_geometry<kMathThreads, kControlThreads>(a);
  collective_scope_sync<forward_tuning::kUnifiedThreads,
                        kCollectiveJoinBarrier>();
}

} // namespace weftmoe_pf
