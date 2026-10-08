#pragma once

#include "cuda/weftmoe_pf/local_compute.cuh"

namespace weftmoe_pf::local {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)

__device__ __forceinline__ void geometry(const ForwardKernelArgs &a) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    const auto owned = a.dispatch.sizes.owned_expert_count;
    auto *runtime = const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime);
    const auto runtime_offset = reinterpret_cast<const char *>(runtime) -
                                reinterpret_cast<const char *>(a.local_base);
    bool error = false;
    for (unsigned source = 0; source < a.world; ++source)
      error |=
          peer_at<DeviceDispatchRuntime>(a.peer_bases[source], runtime_offset)
              ->error != 0U;
    std::uint64_t blocks = 0U, rows = 0U;
    for (unsigned expert = 0; expert < owned; ++expert) {
      unsigned count = 0U;
      if (!error)
        for (unsigned source = 0; source < a.world; ++source)
          count += a.fused_local_expert_count[std::uint64_t(source) * owned +
                                              expert];
      a.ring.expert_rows[expert] = count;
      a.ring.expert_blocks[expert] = blocks;
      blocks += (count + kM - 1U) / kM;
      rows += count;
    }
    a.ring.expert_blocks[owned] = blocks;
    a.ring.state->blocks = blocks;
    a.ring.state->rows = rows;
    runtime->error = error;
    runtime->sizes = a.dispatch.sizes;
    runtime->sizes.incoming_row_count = rows;
    runtime->sizes.mtile_count = blocks;
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    status->error = error;
    status->route_exchange_complete_clock = device_globaltimer_ns();
    status->dispatch_complete_clock = status->route_exchange_complete_clock;
  }
  grid_sync(a);
}

__device__ __forceinline__ void pull(const ForwardKernelArgs &a, Shared &s) {
  const auto warp = (threadIdx.x - kMathThreads - 32U) / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  unsigned expert = 0U, begin = 0U, phase = 0U, peer_rows = 0U;
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  for (unsigned linear = blockIdx.x * 3U + warp;; linear += gridDim.x * 3U) {
    while (expert < owned && linear >= begin + a.ring.expert_rows[expert])
      begin += a.ring.expert_rows[expert++];
    if (expert == owned)
      break;
    unsigned source, ordinal;
    decode_source_row(a, expert, linear - begin, source, ordinal);
    if (lane == 0U) {
      const auto logical =
          unsigned(a.ring.expert_blocks[expert]) * kM + linear - begin;
      const auto block = logical / kM, generation = block / a.ring.depth + 1U;
      const auto row = block % a.ring.depth * kM + logical % kM;
      auto &slot = a.ring.slots[block % a.ring.depth];
      while (ring::acquire(slot.fc1_done) <
             std::uint64_t(generation - 1U) * kFc1Tiles) {
      }
      const auto route = source_routes(a, a.rank, source, expert)[ordinal];
      const auto token = route / a.dispatch.identity.top_k;
      const auto gate = peer_at<float>(a.peer_bases[source],
                                       a.fused_source_gates_offset)[route];
      if (a.saved_x) {
        const auto saved =
            ring::source_metadata(a, source, a.checkpoint.source_rows)[route];
        a.checkpoint.rows[logical] = saved;
        a.checkpoint.gates[saved] = gate;
        a.checkpoint.headers[saved] = {
            transport::kActivationTransportSchemaVersion, source, a.rank, 0U,
            a.dispatch.identity.generation, execution_epoch(a), route, route,
            token, saved, a.rank * owned + expert,
            unsigned(route % a.dispatch.identity.top_k)};
      }
      a.ring.rows[logical] = {
          route, source, gate, token,
          peer_at<transport::BFloat16>(a.peer_bases[source],
                                       a.contribution_offset) +
              (std::uint64_t(route % a.dispatch.identity.top_k) *
                   a.max_tokens_per_rank +
               token) *
                  a.hidden};
      gates[row] = gate;
      const auto *input =
          peer_at<const char>(a.peer_bases[source], a.source_input_offset) +
          std::uint64_t(token) * a.hidden * 2U;
      auto *output = peer_at<char>(a.local_base, a.canonical_rows_offset) +
                     std::uint64_t(row) * a.hidden * 2U;
      for (unsigned offset = 0; offset < a.hidden * 2U; offset += 4096U) {
        const auto bytes = min(4096U, unsigned(a.hidden * 2U) - offset);
        cute::SM90_BULK_COPY_G2S::copy(
            input + offset,
            reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]), s.pull[warp],
            bytes);
        s.pull_full[warp].arrive_and_expect_tx(bytes);
        s.pull_full[warp].wait(phase);
        phase ^= 1U;
        cute::SM90_BULK_COPY_S2G::copy(s.pull[warp], output + offset, bytes);
        cute::tma_store_arrive();
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
      const auto rows = unsigned(a.ring.expert_rows[expert]);
      const auto tail =
          linear - begin + 1U == rows ? (kM - rows % kM) % kM : 0U;
      arrive(slot.x_full, 1U + tail);
      peer_rows += source != a.rank;
    }
    __syncwarp();
  }
  if (lane == 0U) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    if (peer_rows)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                std::uint64_t(peer_rows) * a.hidden * 2ULL);
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
  }
}

__device__ __forceinline__ void combine(const ForwardKernelArgs &a) {
  const auto *input = peer_at<const uint4>(a.local_base, a.contribution_offset);
  auto *output = reinterpret_cast<uint4 *>(a.output);
  constexpr unsigned vectors = forward_tuning::kMainHidden / 8U;
  constexpr unsigned warps = kMathThreads / 32U;
  constexpr unsigned batch = 4U;
  const auto lane = threadIdx.x % 32U;
  const auto top_k = a.dispatch.identity.top_k;
  for (auto token = std::uint64_t(blockIdx.x) * warps + threadIdx.x / 32U;
       token < a.dispatch.identity.num_tokens; token += gridDim.x * warps) {
    // Keep the usual top-k directory in warp registers. Four independent
    // 16B vectors overlap global loads without changing each element's
    // expert-sorted FP32 summation order.
    const auto order =
        lane < top_k ? a.dispatch.combine_order[token * top_k + lane] : 0U;
    for (unsigned column = lane; column < vectors; column += batch * 32U) {
      float2 sum[batch][4] = {};
      for (unsigned position = 0; position < top_k; ++position) {
        const auto slot =
            position < 32U ? __shfl_sync(0xffffffffU, order, position)
                           : a.dispatch.combine_order[token * top_k + position];
        uint4 packed[batch];
#pragma unroll
        for (unsigned vector = 0; vector < batch; ++vector)
          if (column + vector * 32U < vectors)
            packed[vector] = input[(slot * a.max_tokens_per_rank + token) *
                                       vectors + column + vector * 32U];
#pragma unroll
        for (unsigned vector = 0; vector < batch; ++vector) {
          if (column + vector * 32U >= vectors)
            continue;
          const unsigned bits[4] = {packed[vector].x, packed[vector].y,
                                    packed[vector].z, packed[vector].w};
#pragma unroll
          for (unsigned i = 0; i < 4U; ++i) {
            const auto value = bf16x2_bits_to_float2(bits[i]);
            sum[vector][i].x = __fadd_rn(sum[vector][i].x, value.x);
            sum[vector][i].y = __fadd_rn(sum[vector][i].y, value.y);
          }
        }
      }
#pragma unroll
      for (unsigned vector = 0; vector < batch; ++vector)
        if (column + vector * 32U < vectors)
          output[token * vectors + column + vector * 32U] = {
              float2_to_bf16x2_bits(sum[vector][0].x, sum[vector][0].y),
              float2_to_bf16x2_bits(sum[vector][1].x, sum[vector][1].y),
              float2_to_bf16x2_bits(sum[vector][2].x, sum[vector][2].y),
              float2_to_bf16x2_bits(sum[vector][3].x, sum[vector][3].y)};
    }
  }
  collective_scope_sync<kMathThreads, 7U>();
  if (threadIdx.x == 0U) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
    atomicMax(&status->complete_clock, device_globaltimer_ns());
  }
}

#endif

__global__ __launch_bounds__(kThreads, 1) void kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<Shared *>(storage);
  programmatic_launch::wait();
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *status = {};
    status->start_clock = device_globaltimer_ns();
    status->route_exchange_start_clock = status->start_clock;
    status->fc1_start_clock = ~0ULL;
    status->dispatch_blocks_done = ~0ULL;
    status->relay_blocks_done = ~0ULL;
    status->reduced_tokens = ~0ULL;
    const_cast<DeviceDispatchRuntime *>(a.dispatch.runtime)->error = 0U;
    if (a.runtime_epoch)
      ++*a.runtime_epoch;
  }
  const auto tid = std::uint64_t(blockIdx.x) * kThreads + threadIdx.x;
  const auto stride = std::uint64_t(gridDim.x) * kThreads;
  for (auto slot = tid; slot < a.ring.depth; slot += stride)
    a.ring.slots[slot] = {};
  if (!a.retain_backward_state)
    for (auto expert = tid; expert < a.dispatch.identity.num_experts;
         expert += stride)
      const_cast<std::uint64_t *>(a.dispatch.source_expert_count)[expert] = 0U;
  grid_sync(a);
  if (a.retain_backward_state) {
    ring::build_training_rows(a, storage);
    route_exchange<false, true>(a, storage);
  } else {
    route_exchange<false>(a, storage);
  }
  geometry(a);
  if (threadIdx.x == kMathThreads) {
    for (unsigned stage = 0; stage < kStages; ++stage) {
      s.full[stage].init(1U);
      s.empty[stage].init(16U);
    }
    for (auto &barrier : s.pull_full)
      barrier.init(1U);
    cutlass::arch::fence_barrier_init();
  }
  if (threadIdx.x >= kMathThreads)
    cutlass::arch::warpgroup_reg_dealloc<Resources::kUnifiedControlRegisters>();
  __syncthreads();
  cute::cluster_sync();
  forward_compute_detail::observe_compute_cluster(a);
  if (threadIdx.x < kMathThreads) {
    cutlass::arch::warpgroup_reg_alloc<Resources::kUnifiedMathRegisters>();
    compute(a, s, output_map);
    before_combine<kMathThreads>(a);
  } else if (threadIdx.x < kMathThreads + 32U) {
    produce(a, s, x_map, s_map, w1_map, w2_map);
  } else {
    pull(a, s);
  }
  collective_scope_sync<kThreads, kCollectiveJoinBarrier>();
  if (threadIdx.x < kMathThreads) {
    if (threadIdx.x == 0U)
      atomicMin(&status->reduced_tokens, device_globaltimer_ns());
    if (!status->error)
      combine(a);
  } else if (blockIdx.x == 0U && threadIdx.x == kMathThreads) {
    status->fc1_tasks_done = a.ring.state->blocks * kFc1Tiles;
    status->fc2_tasks_done = a.ring.state->blocks * kFc2Tiles;
    status->actual_return_peer_bytes = status->actual_dispatch_peer_bytes;
    // No payload cleanup: the next invocation's route exchange waits for all
    // peers to leave this kernel before any new FC2 contribution can arrive.
    atomicMax(&status->complete_clock, device_globaltimer_ns());
  }
#endif
}

} // namespace weftmoe_pf::local
