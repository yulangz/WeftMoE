#pragma once

namespace static_v4 {

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
  auto *local_counts = a.fused_local_expert_count;
  auto *send_counts =
      const_cast<std::uint64_t *>(a.dispatch.source_expert_count);
  auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
  auto *combine_order = const_cast<std::uint32_t *>(a.dispatch.combine_order);
  auto *owned_ids = const_cast<std::uint32_t *>(a.dispatch.owned_expert_ids);
  auto *row_counts = const_cast<std::uint64_t *>(a.dispatch.expert_row_count);
  auto *row_prefix = const_cast<std::uint64_t *>(a.dispatch.expert_row_prefix);
  auto *mtile_offsets =
      const_cast<std::uint64_t *>(a.dispatch.mtile_row_offsets);
  auto *mtile_rows =
      const_cast<std::uint32_t *>(a.dispatch.mtile_expected_rows);
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
    for (auto token = tid; token < a.dispatch.identity.num_tokens;
         token += stride) {
      const auto base = token * a.dispatch.identity.top_k;
      for (std::uint32_t position = 0;
           position < a.dispatch.identity.top_k; ++position) {
        std::uint32_t best = a.dispatch.identity.top_k;
        for (std::uint32_t slot = 0; slot < a.dispatch.identity.top_k;
             ++slot) {
          bool used = false;
          for (std::uint32_t prior = 0; prior < position; ++prior)
            used = used || combine_order[base + prior] == slot;
          if (used)
            continue;
          if (best == a.dispatch.identity.top_k ||
              a.selected_experts[base + slot] <
                  a.selected_experts[base + best] ||
              (a.selected_experts[base + slot] ==
                   a.selected_experts[base + best] &&
               slot < best))
            best = slot;
        }
        combine_order[base + position] = best;
      }
    }
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

  const auto table_offset =
      reinterpret_cast<const char *>(a.fused_source_routes) -
      reinterpret_cast<const char *>(a.local_base);
  for (auto route = tid; route < routes; route += stride) {
    const auto raw = a.selected_experts[route];
    const auto expert =
        raw < 0 || static_cast<std::uint32_t>(raw) >= experts
            ? 0U
            : static_cast<std::uint32_t>(raw);
    const auto source_slot = atomicAdd(cta_expert_count + expert, 1U);
    const auto destination = expert / owned;
    const auto local_expert = expert % owned;
    auto *table =
        peer_at<std::uint32_t>(a.peer_bases[destination], table_offset);
    const auto index =
        (static_cast<std::uint64_t>(local_expert) * a.world + a.rank) *
            a.max_routes_per_rank +
        source_slot;
    table[index] = static_cast<std::uint32_t>(route);
  }

  collective_scope_sync<forward_tuning::kUnifiedThreads,
                        kCollectiveJoinBarrier>();

  // First finish all source-local reservations and route-table stores.  CTA 0
  // then publishes this source's final counts directly into each destination's
  // source-major table, matching the route-table direction.  The following
  // release/acquire boundary makes both remote payloads visible together.
  grid_sync<kMathThreads, kControlThreads>(a);
  if (blockIdx.x == 0U) {
    const auto count_offset =
        reinterpret_cast<const char *>(local_counts) -
        reinterpret_cast<const char *>(a.local_base);
    for (auto expert = static_cast<std::uint64_t>(control_thread);
         expert < experts; expert += kControlThreads) {
      const auto destination = static_cast<std::uint32_t>(expert / owned);
      const auto local = static_cast<std::uint32_t>(expert % owned);
      peer_at<std::uint64_t>(a.peer_bases[destination], count_offset)
          [static_cast<std::uint64_t>(a.rank) * owned + local] =
              send_counts[expert];
    }
  }
  rank_barrier<kMathThreads, kControlThreads, false, false>(a);

  if (blockIdx.x == 0U) {
    for (auto local = static_cast<std::uint64_t>(control_thread);
         local < owned; local += kControlThreads) {
      std::uint64_t total = 0ULL;
      for (std::uint32_t source = 0; source < a.world; ++source)
        total += local_counts[
            static_cast<std::uint64_t>(source) * owned + local];
      owned_ids[local] = a.rank * owned + local;
      row_counts[local] = total;
    }
    collective_scope_sync<kControlThreads, kCollectiveScopeBarrier>();
    if (control_thread == 0U) {
      std::uint64_t rows = 0ULL;
      std::uint64_t mtiles = 0ULL;
      for (std::uint32_t local = 0; local < owned; ++local) {
        const auto total = row_counts[local];
        row_prefix[local] = rows;
        mtile_offsets[local] = mtiles;
        const auto tiles =
            (total + forward_tuning::kTileM - 1ULL) /
            forward_tuning::kTileM;
        for (std::uint64_t tile = 0; tile < tiles; ++tile) {
          const auto remaining = total - tile * forward_tuning::kTileM;
          mtile_rows[mtiles + tile] = static_cast<std::uint32_t>(
              remaining < forward_tuning::kTileM
                  ? remaining
                  : forward_tuning::kTileM);
        }
        rows += total;
        mtiles += tiles;
      }
      row_prefix[owned] = rows;
      mtile_offsets[owned] = mtiles;
      runtime->sizes.incoming_row_count = rows;
      runtime->sizes.mtile_count = mtiles;
      if (rows > a.max_destination_rows)
        runtime->error = 1ULL;
    }
  }
  grid_sync<kMathThreads, kControlThreads>(a);
  collective_scope_sync<forward_tuning::kUnifiedThreads,
                        kCollectiveJoinBarrier>();
}

} // namespace static_v4
