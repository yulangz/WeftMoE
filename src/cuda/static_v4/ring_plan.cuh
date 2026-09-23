#pragma once

namespace static_v4::ring {

__device__ __noinline__ void prepare_local(const ForwardKernelArgs &a,
                                           unsigned char *scratch) {
  constexpr unsigned Math = 256U, Control = 128U;
  const auto &r = a.ring;
  const auto &p = r.source;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    r.state->tokens = p.tokens;
    r.state->error = 0U;
    if (a.runtime_epoch)
      ++*a.runtime_epoch;
  }
  grid_sync(a);
  if (threadIdx.x < Math) {
    const auto tid = std::uint64_t(blockIdx.x) * Math + threadIdx.x;
    const auto stride = std::uint64_t(gridDim.x) * Math;
    auto *input = peer_at<uint4>(a.local_base, a.source_input_offset);
    for (auto v = tid; v < p.tokens * a.hidden / 8U; v += stride)
      input[v] = reinterpret_cast<const uint4 *>(a.input)[v];
    auto *gates = peer_at<float>(a.local_base, a.fused_source_gates_offset);
    for (auto route = tid; route < p.tokens * p.shape.top_k; route += stride)
      gates[route] = a.gates[route];
    build_combine_order(a, tid, stride);
    collective_scope_sync<384U, kCollectiveJoinBarrier>();
    collective_scope_sync<384U, kCollectiveJoinBarrier>();
    return;
  }
  const auto tid = threadIdx.x - Math;
  auto *counts = reinterpret_cast<unsigned *>(scratch);
  for (auto w = std::uint64_t(blockIdx.x); w < p.local_windows();
       w += gridDim.x)
    static_forward::build_source_window<Control, Math, kCollectiveScopeBarrier>(
        p, w, counts);
  collective_scope_sync<384U, kCollectiveJoinBarrier>();
  rank_barrier<Math, Control>(a);

  if (blockIdx.x == 0U) {
    unsigned error = 0U;
    for (unsigned source = tid; source < a.world; source += Control) {
      const auto *state = peer(a, source, r.state);
      r.source_epochs[source] = {state->tokens, state->source_base};
      error |= state->error;
    }
    error = __reduce_or_sync(0xffffffffU, error);
    if (tid % 32U == 0U)
      counts[tid / 32U] = error;
    collective_scope_sync<Control, kCollectiveScopeBarrier>();
    if (tid == 0U) {
      std::uint64_t tokens = 0U;
      for (unsigned source = 0; source < a.world; ++source)
        tokens = tokens > r.source_epochs[source].tokens
                     ? tokens
                     : r.source_epochs[source].tokens;
      for (unsigned warp = 0; warp < Control / 32U; ++warp)
        error |= counts[warp];
      r.state->windows = error ? 0U : p.shape.windows(tokens);
      auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
      runtime->error = error;
      peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)->error =
          error;
    }
  }
  grid_sync<Math, Control>(a);

  for (auto w = std::uint64_t(blockIdx.x); w < r.state->windows;
       w += gridDim.x) {
    const auto first = p.shape.source_begin(w);
    const auto local_window = w / p.shape.groups();
    if (tid < p.shape.source_end(w) - first) {
      const auto seq = r.source_epochs[first + tid].base + local_window;
      const auto slot = static_cast<unsigned>(seq % r.contribution_depth);
      r.window_sources[w * p.shape.sources + tid] = {
          seq, seq / r.contribution_depth + 1U,
          std::uint64_t(slot) * r.contribution_stride, slot};
    }
    for (unsigned index = tid; index < owned * p.shape.sources;
         index += Control) {
      const auto expert = index / p.shape.sources;
      const auto source = first + index % p.shape.sources;
      counts[index] =
          source < p.shape.source_end(w)
              ? static_cast<unsigned>(source_count(a, source, local_window,
                                                   a.rank * owned + expert))
              : 0U;
    }
    collective_scope_sync<Control, kCollectiveScopeBarrier>();
    static_forward::WindowGeometryView geometry{p.shape,
                                                w,
                                                owned,
                                                counts,
                                                r.expert_rows + w * owned,
                                                r.expert_blocks +
                                                    w * (owned + 1U),
                                                nullptr};
    static_forward::build_window_counts<Control, Math, kCollectiveScopeBarrier>(
        geometry);
  }
  grid_sync<Math, Control>(a);
  if (blockIdx.x == 0U && tid == 0U) {
    std::uint64_t blocks = 0U, rows = 0U;
    for (std::uint64_t w = 0; w < r.state->windows; ++w) {
      r.window_prefix[w] = blocks;
      blocks += r.expert_blocks[w * (owned + 1U) + owned];
      for (unsigned e = 0; e < owned; ++e)
        rows += r.expert_rows[w * owned + e];
    }
    r.window_prefix[r.state->windows] = blocks;
    r.state->blocks = blocks;
    r.state->rows = rows;
    auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
    runtime->sizes = a.dispatch.sizes;
    runtime->sizes.incoming_row_count = rows;
    runtime->sizes.mtile_count = blocks;
    auto *scheduler =
        peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
    scheduler[0] = scheduler[1] = scheduler[2] = 0U;
    scheduler[3] = blocks;
    scheduler[4U + a.world] = scheduler[11U + 2U * a.world] = 0U;
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    status->dispatch_blocks_done = status->reduced_tokens =
        status->fc1_start_clock = status->relay_blocks_done = ~0ULL;
  }
  grid_sync<Math, Control>(a);
  for (auto task = std::uint64_t(blockIdx.x); task < r.state->windows * owned;
       task += gridDim.x) {
    const auto w = task / owned;
    const auto expert = static_cast<unsigned>(task % owned);
    const auto *prefix = r.expert_blocks + w * (owned + 1U);
    const auto first = p.shape.source_begin(w);
    const auto local_window = w / p.shape.groups();
    for (unsigned index = tid; index < p.shape.sources; index += Control) {
      const auto source = first + index;
      counts[index] =
          source < p.shape.source_end(w)
              ? static_cast<unsigned>(source_count(a, source, local_window,
                                                   a.rank * owned + expert))
              : 0U;
    }
    collective_scope_sync<Control, kCollectiveScopeBarrier>();
    const static_forward::WindowGeometryView geometry{
        p.shape,
        w,
        1U,
        counts,
        r.expert_rows + w * owned + expert,
        r.expert_blocks + w * (owned + 1U) + expert,
        nullptr};
    for (auto b = std::uint64_t(tid); b < prefix[expert + 1U] - prefix[expert];
         b += Control) {
      const auto block = r.window_prefix[w] + prefix[expert] + b;
      const auto seq = r.state->block_base + block;
      r.blocks[block] = {
          {expert, static_cast<unsigned>(b * p.shape.tile_m),
           static_forward::block_rows(r.expert_rows[w * owned + expert], b,
                                      p.shape.tile_m)},
          static_cast<unsigned>(w), static_cast<unsigned>(seq % r.depth),
          seq / r.depth + 1U};
    }
    // Freeze reverse identity and gate before payload starts. Each warp reads
    // a contiguous source-expert route list; pull does no peer count/ordinal
    // search, and Return never depends on the lifetime of an X slot.
    for (auto source = first + tid / 32U; source < p.shape.source_end(w);
         source += Control / 32U) {
      const auto count = counts[source - first];
      const auto *source_prefix =
          peer(a, source, p.expert_prefix) + local_window * (p.experts + 1U);
      const auto *order =
          peer(a, source, p.route_order) + local_window * p.window_capacity();
      const auto *gates = peer_at<const float>(a.peer_bases[source],
                                               a.fused_source_gates_offset);
      for (auto ordinal = tid % 32U; ordinal < count; ordinal += 32U) {
        const auto row = geometry.expert_row(0U, source, ordinal);
        const auto route =
            order[source_prefix[a.rank * owned + expert] + ordinal];
        r.rows[(r.window_prefix[w] + prefix[expert]) * p.shape.tile_m + row] = {
            static_cast<unsigned>(route), source, gates[route],
            static_cast<unsigned>(route / p.shape.top_k)};
      }
    }
    collective_scope_sync<Control, kCollectiveScopeBarrier>();
  }
  grid_sync<Math, Control>(a);
  collective_scope_sync<384U, kCollectiveJoinBarrier>();
}

} // namespace static_v4::ring
