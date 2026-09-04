#include "cuda/backward/kernel_contract.cuh"
#include "cuda/persistent_v3/backward/compute_kernels.cuh"

__global__ void backward_pack_cached_state_kernel(BackwardKernelArgs a,
                                                   std::uint32_t chunks) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x) / chunks;
  const auto chunk = static_cast<std::uint32_t>(blockIdx.x) % chunks;
  if (expert >= a.dispatch.sizes.owned_expert_count)
    return;
  const auto compact_begin = a.dispatch.expert_row_prefix[expert];
  const auto valid_rows = a.dispatch.expert_row_count[expert];
  const auto *padded_rows = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_rows_offset);
  const auto *padded_offsets = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  const auto padded_begin = padded_offsets[expert];
  const auto rows = padded_rows[expert];
  const auto *compact_x = peer_at<const transport::BFloat16>(
      a.local_base, a.canonical_x_offset);
  const auto *compact_ab = peer_at<const transport::BFloat16>(
      a.local_base, a.cached_ab_offset);
  const auto *compact_gate = peer_at<const float>(
      a.local_base, a.canonical_gate_offset);
  auto *padded_x = peer_at<transport::BFloat16>(a.local_base,
                                                a.padded_x_offset);
  auto *padded_ab = peer_at<transport::BFloat16>(a.local_base,
                                                 a.padded_ab_offset);
  auto *padded_dr = peer_at<transport::BFloat16>(a.local_base,
                                                 a.padded_dr_offset);
  auto *padded_gate = peer_at<float>(a.local_base, a.dgate_rows_offset);
  constexpr std::uint32_t kVectorsPerThreadGroup = 8U;
  static_assert(sizeof(uint4) ==
                kVectorsPerThreadGroup * sizeof(transport::BFloat16));
  for (std::uint64_t local_row = chunk; local_row < rows;
       local_row += chunks) {
    const auto padded_row = padded_begin + local_row;
    const auto valid = local_row < valid_rows;
    const auto compact_row = compact_begin + local_row;
    for (std::uint64_t vector = threadIdx.x;
         vector < a.hidden / kVectorsPerThreadGroup;
         vector += blockDim.x) {
      const auto target = reinterpret_cast<uint4 *>(
          padded_x + padded_row * a.hidden);
      if (valid) {
        const auto source = reinterpret_cast<const uint4 *>(
            compact_x + compact_row * a.hidden);
        target[vector] = source[vector];
      } else {
        target[vector] = uint4{};
        reinterpret_cast<uint4 *>(padded_dr + padded_row * a.hidden)[vector] =
            uint4{};
      }
    }
    for (std::uint64_t vector = threadIdx.x;
         vector < a.intermediate / kVectorsPerThreadGroup;
         vector += blockDim.x) {
      const auto compact_index = compact_row * a.intermediate;
      const auto padded_index = padded_row * a.intermediate;
      auto *target_a = reinterpret_cast<uint4 *>(padded_ab + padded_index);
      auto *target_b = reinterpret_cast<uint4 *>(
          padded_ab + a.padded_capacity_rows * a.intermediate + padded_index);
      if (valid) {
        const auto *source_a =
            reinterpret_cast<const uint4 *>(compact_ab + compact_index);
        const auto *source_b = reinterpret_cast<const uint4 *>(
            compact_ab + a.max_destination_rows * a.intermediate +
            compact_index);
        target_a[vector] = source_a[vector];
        target_b[vector] = source_b[vector];
      } else {
        target_a[vector] = uint4{};
        target_b[vector] = uint4{};
      }
    }
    if (threadIdx.x == 0U)
      padded_gate[padded_row] = valid ? compact_gate[compact_row] : 0.0F;
  }
}

__global__ void backward_pack_gate_kernel(BackwardKernelArgs a) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (expert >= a.dispatch.sizes.owned_expert_count)
    return;
  const auto compact_begin = a.dispatch.expert_row_prefix[expert];
  const auto valid_rows = a.dispatch.expert_row_count[expert];
  const auto *padded_rows = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_rows_offset);
  const auto *padded_offsets = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  const auto *compact_gate = peer_at<const float>(
      a.local_base, a.canonical_gate_offset);
  auto *padded_gate = peer_at<float>(a.local_base, a.dgate_rows_offset);
  for (std::uint64_t local_row = threadIdx.x;
       local_row < padded_rows[expert]; local_row += blockDim.x) {
    const auto padded_row = padded_offsets[expert] + local_row;
    padded_gate[padded_row] =
        local_row < valid_rows ? compact_gate[compact_begin + local_row]
                               : 0.0F;
  }
}

__global__ void backward_setup_local_kernel(BackwardKernelArgs a) {
  if (blockIdx.x != 0U)
    return;
  for (std::uint32_t peer = threadIdx.x; peer < a.world;
       peer += blockDim.x) {
    auto *setup = peer_at<unsigned long long>(a.peer_bases[peer],
                                               a.setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(setup[a.rank]);
    published.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_setup = peer_at<unsigned long long>(a.local_base,
                                                   a.setup_epoch_offset);
  for (std::uint32_t peer = threadIdx.x; peer < a.world;
       peer += blockDim.x) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(local_setup[peer]);
    while (published.load(::cuda::memory_order_acquire) != a.epoch)
      __nanosleep(backward_tuning::kSetupIdleNanosleep);
  }
}

__global__ void backward_dispatch_dy_local_kernel(BackwardKernelArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x);
  if (route >= a.dispatch.sizes.route_count)
    return;
  const auto destination = a.dispatch.route_dst_rank[route];
  const auto compact_destination_row = a.dispatch.route_dst_row[route];
  const auto *compact_to_padded = peer_at<const unsigned int>(
      a.peer_bases[destination], a.compact_to_padded_offset);
  const auto destination_row = compact_to_padded[compact_destination_row];
  const auto token = route / a.dispatch.identity.top_k;
  auto *destination_dr = peer_at<transport::BFloat16>(
      a.peer_bases[destination], a.padded_dr_offset);
  const auto *source = reinterpret_cast<const uint4 *>(
      a.dy + token * a.hidden);
  auto *target = reinterpret_cast<uint4 *>(
      destination_dr + destination_row * a.hidden);
  const auto vectors = a.hidden / 8ULL;
  for (std::uint64_t vector = threadIdx.x; vector < vectors;
       vector += blockDim.x)
    target[vector] = source[vector];
  __syncthreads();
  if (threadIdx.x == 0U) {
    __threadfence_system();
    auto *ready = peer_at<unsigned long long>(
        a.peer_bases[destination], a.dispatch_ready_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        marker(ready[compact_destination_row]);
    marker.store(a.epoch, ::cuda::memory_order_release);
  }
}

__global__ void backward_wait_for_dispatch_rows_kernel(BackwardKernelArgs a,
                                                        std::uint64_t rows) {
  const auto compact_row = static_cast<std::uint64_t>(blockIdx.x);
  if (compact_row >= rows || threadIdx.x != 0U)
    return;
  auto *ready = peer_at<unsigned long long>(
      a.local_base, a.dispatch_ready_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        marker(ready[compact_row]);
  // Each fixed row has one producer.  This kernel is only the correctness
  // scaffold's launch boundary; the persistent implementation consumes the
  // same marker directly when it claims a ready tile.
  while (marker.load(::cuda::memory_order_acquire) != a.epoch)
    __nanosleep(backward_tuning::kReduceIdleNanosleep);
}

__device__ __forceinline__ void backward_swiglu_epilogue_row(
    const BackwardKernelArgs &a, std::uint64_t row, bool valid_row,
    float *reduction) {
  constexpr std::uint32_t kEpilogueThreads = 256U;
  const bool active = threadIdx.x < kEpilogueThreads;
  auto *u = peer_at<transport::BFloat16>(a.local_base,
                                         a.recomputed_u_offset);
  const auto *du = u;
  const auto *ab = peer_at<const transport::BFloat16>(
      a.local_base, a.padded_ab_offset);
  auto *dab = peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_rows_offset);
  if (!valid_row) {
    for (std::uint64_t column = threadIdx.x;
         active && column < a.intermediate;
         column += kEpilogueThreads) {
      const auto index = row * a.intermediate + column;
      const auto output_base = row * 2ULL * a.intermediate;
      u[index] = transport::BFloat16{0U};
      dab[output_base + column] = transport::BFloat16{0U};
      dab[output_base + a.intermediate + column] =
          transport::BFloat16{0U};
    }
    if (threadIdx.x == 0U)
      dgate[row] = 0.0F;
    __syncthreads();
    return;
  }
  const auto gate = dgate[row];
  float partial = 0.0F;
  for (std::uint64_t column = threadIdx.x;
       active && column < a.intermediate;
       column += kEpilogueThreads) {
    const auto index = row * a.intermediate + column;
    const auto av = bf16_bits_to_float(ab[index].bits);
    const auto bv = bf16_bits_to_float(
        ab[a.padded_capacity_rows * a.intermediate + index].bits);
    const auto duv = bf16_bits_to_float(du[index].bits);
    const auto sigmoid = 1.0F / (1.0F + __expf(-av));
    const auto silu = av * sigmoid;
    const auto s = silu * bv;
    const auto ds = gate * duv;
    const auto silu_derivative =
        sigmoid * (1.0F + av * (1.0F - sigmoid));
    const auto output_base = row * 2ULL * a.intermediate;
    dab[output_base + column] = transport::BFloat16{
        float_to_bf16_bits(ds * bv * silu_derivative)};
    dab[output_base + a.intermediate + column] =
        transport::BFloat16{float_to_bf16_bits(ds * silu)};
    u[index] = transport::BFloat16{
        float_to_bf16_bits(gate * s)};
    partial = fmaf(duv, s, partial);
  }
  if (active)
    reduction[threadIdx.x] = partial;
  __syncthreads();
  for (std::uint32_t stride = kEpilogueThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U)
    dgate[row] = reduction[0];
}

__device__ __forceinline__ void backward_swiglu_epilogue_warp_row(
    const BackwardKernelArgs &a, std::uint64_t row, bool row_is_valid,
    std::uint32_t lane) {
  auto *u = peer_at<transport::BFloat16>(a.local_base,
                                         a.recomputed_u_offset);
  const auto *ab = peer_at<const transport::BFloat16>(
      a.local_base, a.padded_ab_offset);
  auto *dab = peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_rows_offset);
  const auto output_base = row * 2ULL * a.intermediate;
  if (!row_is_valid) {
    for (std::uint64_t column = lane; column < a.intermediate;
         column += 32U) {
      const auto index = row * a.intermediate + column;
      u[index] = transport::BFloat16{0U};
      dab[output_base + column] = transport::BFloat16{0U};
      dab[output_base + a.intermediate + column] =
          transport::BFloat16{0U};
    }
    if (lane == 0U)
      dgate[row] = 0.0F;
    return;
  }

  const auto gate = dgate[row];
  float partial = 0.0F;
  for (std::uint64_t column = lane; column < a.intermediate;
       column += 32U) {
    const auto index = row * a.intermediate + column;
    const auto av = bf16_bits_to_float(ab[index].bits);
    const auto bv = bf16_bits_to_float(
        ab[a.padded_capacity_rows * a.intermediate + index].bits);
    const auto duv = bf16_bits_to_float(u[index].bits);
    const auto sigmoid = 1.0F / (1.0F + __expf(-av));
    const auto silu = av * sigmoid;
    const auto s = silu * bv;
    const auto ds = gate * duv;
    const auto silu_derivative =
        sigmoid * (1.0F + av * (1.0F - sigmoid));
    dab[output_base + column] = transport::BFloat16{
        float_to_bf16_bits(ds * bv * silu_derivative)};
    dab[output_base + a.intermediate + column] =
        transport::BFloat16{float_to_bf16_bits(ds * silu)};
    u[index] = transport::BFloat16{float_to_bf16_bits(gate * s)};
    partial = fmaf(duv, s, partial);
  }
#pragma unroll
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U)
    partial += __shfl_down_sync(0xffffffffU, partial, stride);
  if (lane == 0U)
    dgate[row] = partial;
}

// A persistent compute CTA has twelve warps. Give rows to those warps in
// compile-time-unrolled rounds and keep one FP32 dGate accumulator per lane so
// the elementwise epilogue does not extend eight independent live ranges into
// the surrounding WGMMA scheduler.
template <std::uint32_t RowsPerTask>
__device__ __forceinline__ void backward_swiglu_epilogue_warp_rows(
    const BackwardKernelArgs &a, std::uint64_t first_row,
    std::uint32_t valid_rows, std::uint32_t rows) {
  constexpr std::uint32_t kWarps = backward_tuning::kComputeThreads / 32U;
  static_assert(RowsPerTask >= kWarps);
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  if constexpr (RowsPerTask == kWarps) {
    if (warp >= rows)
      return;
    backward_swiglu_epilogue_warp_row(
        a, first_row + warp, warp < valid_rows, lane);
  } else {
#pragma unroll
    for (std::uint32_t local_row = warp; local_row < RowsPerTask;
         local_row += kWarps) {
      if (local_row < rows)
        backward_swiglu_epilogue_warp_row(
            a, first_row + local_row, local_row < valid_rows, lane);
    }
  }
}

__global__ void backward_swiglu_epilogue_kernel(BackwardKernelArgs a,
                                                 std::uint64_t rows) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= rows)
    return;
  __shared__ unsigned int valid_row;
  __shared__ float reduction[256];
  if (threadIdx.x == 0U) {
    valid_row = 0U;
    const auto *padded_offsets = peer_at<const unsigned long long>(
        a.local_base, a.padded_expert_offsets_offset);
    for (std::uint32_t expert = 0U;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      const auto begin = padded_offsets[expert];
      const auto end = padded_offsets[expert + 1U];
      if (row >= begin && row < end) {
        valid_row = row - begin < a.dispatch.expert_row_count[expert] ? 1U : 0U;
        break;
      }
    }
  }
  __syncthreads();
  backward_swiglu_epilogue_row(a, row, valid_row != 0U, reduction);
}

__global__ void backward_return_local_kernel(BackwardKernelArgs a,
                                              std::uint64_t rows) {
  const auto compact_row = static_cast<std::uint64_t>(blockIdx.x);
  if (compact_row >= rows)
    return;
  const auto *compact_to_padded = peer_at<const unsigned int>(
      a.local_base, a.compact_to_padded_offset);
  const auto row = compact_to_padded[compact_row];
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  const auto header = headers[compact_row];
  const auto *dx = peer_at<const transport::BFloat16>(
      a.local_base, a.dx_rows_offset);
  const auto *dgate = peer_at<const float>(a.local_base,
                                           a.dgate_rows_offset);
  auto *source_contribution = peer_at<transport::BFloat16>(
      a.peer_bases[header.source_rank], a.contribution_offset);
  const auto *source = reinterpret_cast<const uint4 *>(
      dx + row * a.hidden);
  auto *target = reinterpret_cast<uint4 *>(
      source_contribution + header.route_index * a.hidden);
  const auto vectors = a.hidden / 8ULL;
  for (std::uint64_t vector = threadIdx.x; vector < vectors;
       vector += blockDim.x)
    target[vector] = source[vector];
  if (threadIdx.x == 0U) {
    auto *source_dgate = peer_at<float>(
        a.peer_bases[header.source_rank], a.dgate_output_offset);
    source_dgate[header.route_index] = dgate[row];
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    __threadfence_system();
    auto *ready = peer_at<unsigned long long>(
        a.peer_bases[header.source_rank], a.return_ready_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        marker(ready[header.route_index]);
    marker.store(a.epoch, ::cuda::memory_order_release);
  }
}

__global__ void backward_reduce_dx_kernel(BackwardKernelArgs a) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= a.dispatch.identity.num_tokens)
    return;
  constexpr std::uint32_t kTrainingTopK = 6U;
  __shared__ const transport::BFloat16 *source_rows[kTrainingTopK];
  const auto *contribution = peer_at<const transport::BFloat16>(
      a.local_base, a.contribution_offset);
#if __has_include(<nccl_device.h>)
  const auto *direct_receive =
      a.use_gin
          ? static_cast<const std::byte *>(a.gin_window_base) +
                a.gin_return_receive_offset
          : static_cast<const std::byte *>(nullptr);
  const auto *direct_compact =
      a.use_gin
          ? peer_at<const unsigned long long>(
                a.local_base, a.forward_direct_source_compact_index_offset)
          : static_cast<const unsigned long long *>(nullptr);
#endif
  auto *combined = peer_at<transport::BFloat16>(
      a.local_base, a.combined_dx_offset);
  auto *ready = peer_at<unsigned long long>(a.local_base,
                                             a.return_ready_offset);
  const auto route_base = token * a.dispatch.identity.top_k;
  if (a.dispatch.identity.top_k == kTrainingTopK) {
    if (threadIdx.x < kTrainingTopK) {
      const auto position = static_cast<std::uint32_t>(threadIdx.x);
      const auto slot = a.dispatch.combine_order[route_base + position];
      const auto route = route_base + slot;
      auto *source = contribution + route * a.hidden;
#if __has_include(<nccl_device.h>)
      if (a.use_gin && a.dispatch.route_dst_node[route] != a.node)
        source = reinterpret_cast<const transport::BFloat16 *>(
            direct_receive +
            direct_compact[route] * a.gin_return_row_bytes);
#endif
      source_rows[position] = source;
    }
    __syncthreads();
  }
  for (std::uint32_t position = 0U;
       position < a.dispatch.identity.top_k; ++position) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        marker(ready[route_base + position]);
    while (marker.load(::cuda::memory_order_acquire) != a.epoch)
      __nanosleep(backward_tuning::kReduceIdleNanosleep);
  }
  for (std::uint64_t column = threadIdx.x; column < a.hidden;
       column += blockDim.x) {
    float sum = 0.0F;
    if (a.dispatch.identity.top_k == kTrainingTopK) {
#pragma unroll
      for (std::uint32_t position = 0U; position < kTrainingTopK;
           ++position)
        sum += bf16_bits_to_float(source_rows[position][column].bits);
    } else {
      for (std::uint32_t position = 0U;
           position < a.dispatch.identity.top_k; ++position) {
        const auto slot = a.dispatch.combine_order[route_base + position];
        const auto route = route_base + slot;
#if __has_include(<nccl_device.h>)
        const auto remote =
            a.use_gin && a.dispatch.route_dst_node[route] != a.node;
        const auto *remote_row =
            remote
                ? reinterpret_cast<const transport::BFloat16 *>(
                      direct_receive + direct_compact[route] *
                                           a.gin_return_row_bytes)
                : static_cast<const transport::BFloat16 *>(nullptr);
        const auto value = remote ? remote_row[column]
                                  : contribution[route * a.hidden + column];
#else
        const auto value = contribution[route * a.hidden + column];
#endif
        sum += bf16_bits_to_float(value.bits);
      }
    }
    combined[token * a.hidden + column] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
}

#include "cuda/persistent_v3/backward/pipeline_kernels.cuh"
