#include "cuda/backward/kernel_contract.cuh"
#include "cuda/persistent_v3/backward/compute_kernels.cuh"

// Rebuilt for each invocation: routing counts may change even at fixed shape.
// Capacity was checked when the pool and dispatch handle were constructed.
__global__ void backward_build_row_directory_kernel(BackwardKernelArgs a) {
  auto *offsets = peer_at<unsigned long long>(a.local_base, a.padded_expert_offsets_offset);
  auto *rows = peer_at<unsigned long long>(a.local_base, a.padded_expert_rows_offset);
  auto *mapping = peer_at<unsigned int>(a.local_base, a.compact_to_padded_offset);
  if (threadIdx.x == 0U) {
    unsigned long long cursor = 0ULL;
    for (unsigned int e = 0; e < dispatch_sizes(a.dispatch).owned_expert_count;
         ++e) {
      offsets[e] = cursor;
      rows[e] = ((a.dispatch.expert_row_count[e] + 127ULL) / 128ULL) * 128ULL;
      cursor += rows[e];
    }
    offsets[dispatch_sizes(a.dispatch).owned_expert_count] = cursor;
  }
  __syncthreads();
  for (unsigned int e = 0; e < dispatch_sizes(a.dispatch).owned_expert_count;
       ++e)
    for (unsigned long long row = threadIdx.x; row < a.dispatch.expert_row_count[e]; row += blockDim.x)
      mapping[a.dispatch.expert_row_prefix[e] + row] =
          static_cast<unsigned int>(offsets[e] + row);
}

__global__ void backward_clear_inactive_gradients_kernel(BackwardKernelArgs a) {
  const auto expert = blockIdx.y;
  if (a.dispatch.expert_row_count[expert] != 0ULL) return;
  const auto elements = a.hidden * a.intermediate;
  auto *dw1 = peer_at<transport::BFloat16>(a.local_base, a.dw1_offset) + expert * 2ULL * elements;
  auto *dw2 = peer_at<transport::BFloat16>(a.local_base, a.dw2_offset) + expert * elements;
  for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < 2ULL * elements; i += gridDim.x * blockDim.x) {
    reinterpret_cast<unsigned short *>(dw1)[i] = 0;
    if (i < elements) reinterpret_cast<unsigned short *>(dw2)[i] = 0;
  }
}

__device__ __forceinline__ void backward_load_row_geometry(BackwardKernelArgs &a) {
  __shared__ unsigned long long geometry[2];
  if (threadIdx.x == 0U) {
    const auto *offsets = peer_at<const unsigned long long>(a.local_base, a.padded_expert_offsets_offset);
    geometry[0] = offsets[dispatch_sizes(a.dispatch).owned_expert_count];
    unsigned long long active = 0;
    for (unsigned int e = 0; e < dispatch_sizes(a.dispatch).owned_expert_count;
         ++e)
      active += a.dispatch.expert_row_count[e] != 0ULL;
    geometry[1] = active;
  }
  __syncthreads();
  a.padded_rows = geometry[0];
  a.mblocks = static_cast<unsigned int>(geometry[0] / backward_tuning::kTileM);
  a.active_experts = static_cast<unsigned int>(geometry[1]);
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
    published.store(execution_epoch(a), ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_setup = peer_at<unsigned long long>(a.local_base,
                                                   a.setup_epoch_offset);
  for (std::uint32_t peer = threadIdx.x; peer < a.world;
       peer += blockDim.x) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(local_setup[peer]);
    while (published.load(::cuda::memory_order_acquire) != execution_epoch(a))
      __nanosleep(backward_tuning::kSetupIdleNanosleep);
  }
}

__global__ void backward_dispatch_dy_local_kernel(BackwardKernelArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x);
  if (route >= dispatch_sizes(a.dispatch).route_count)
    return;
  const auto destination = a.dispatch.route_dst_rank[route];
  const auto compact_destination_row = a.dispatch.route_dst_row[route];
  const auto token = route / a.dispatch.identity.top_k;
  auto *destination_dr = peer_at<transport::BFloat16>(
      a.peer_bases[destination], a.compact_dr_offset);
  const auto *source = reinterpret_cast<const uint4 *>(
      a.dy + token * a.hidden);
  auto *target = reinterpret_cast<uint4 *>(
      destination_dr + compact_destination_row * a.hidden);
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
    marker.store(execution_epoch(a), ::cuda::memory_order_release);
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
    while (marker.load(::cuda::memory_order_acquire) != execution_epoch(a))
      __nanosleep(backward_tuning::kReduceIdleNanosleep);
}

__device__ __forceinline__ void backward_swiglu_epilogue_warp_row(
    const BackwardKernelArgs &a, std::uint64_t padded_row,
    std::uint64_t compact_row, bool row_is_valid, std::uint32_t lane) {
  auto *u = peer_at<transport::BFloat16>(a.local_base,
                                         a.recomputed_u_offset);
  const auto *ab = peer_at<const transport::BFloat16>(
      a.local_base, a.cached_ab_offset);
  const auto *gates = peer_at<const float>(a.local_base,
                                           a.canonical_gate_offset);
  auto *dab = peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_rows_offset);
  const auto b_base = a.max_destination_rows * a.intermediate;
  const auto output_base = padded_row * 2ULL * a.intermediate;
  if (!row_is_valid) {
    for (std::uint64_t column = 2ULL * lane; column < a.intermediate;
         column += 64U) {
      const auto index = padded_row * a.intermediate + column;
      *reinterpret_cast<std::uint32_t *>(u + index) = 0U;
      *reinterpret_cast<std::uint32_t *>(dab + output_base + column) = 0U;
      *reinterpret_cast<std::uint32_t *>(
          dab + output_base + a.intermediate + column) = 0U;
    }
    if (lane == 0U)
      dgate[padded_row] = 0.0F;
    return;
  }

  const auto gate = gates[compact_row];
  float partial = 0.0F;
  for (std::uint64_t column = 2ULL * lane; column < a.intermediate;
       column += 64U) {
    const auto output_index = padded_row * a.intermediate + column;
    const auto input_index = compact_row * a.intermediate + column;
    const auto a_pair =
        *reinterpret_cast<const std::uint32_t *>(ab + input_index);
    const auto b_pair =
        *reinterpret_cast<const std::uint32_t *>(ab + b_base + input_index);
    const auto du_pair =
        *reinterpret_cast<const std::uint32_t *>(u + output_index);
    std::uint32_t da_pair = 0U;
    std::uint32_t db_pair = 0U;
    std::uint32_t gated_u_pair = 0U;
#pragma unroll
    for (std::uint32_t element = 0U; element < 2U; ++element) {
      const auto shift = element * 16U;
      const auto av = __uint_as_float((a_pair >> shift) << 16U);
      const auto bv = __uint_as_float((b_pair >> shift) << 16U);
      const auto duv = __uint_as_float((du_pair >> shift) << 16U);
      const auto sigmoid = 1.0F / (1.0F + __expf(-av));
      const auto silu = av * sigmoid;
      const auto s = silu * bv;
      const auto ds = gate * duv;
      const auto silu_derivative =
          sigmoid * (1.0F + av * (1.0F - sigmoid));
      da_pair |= static_cast<std::uint32_t>(
                     float_to_bf16_bits(ds * bv * silu_derivative))
                 << shift;
      db_pair |= static_cast<std::uint32_t>(
                     float_to_bf16_bits(ds * silu))
                 << shift;
      gated_u_pair |= static_cast<std::uint32_t>(
                          float_to_bf16_bits(gate * s))
                      << shift;
      partial = fmaf(duv, s, partial);
    }
    *reinterpret_cast<std::uint32_t *>(dab + output_base + column) = da_pair;
    *reinterpret_cast<std::uint32_t *>(
        dab + output_base + a.intermediate + column) = db_pair;
    *reinterpret_cast<std::uint32_t *>(u + output_index) = gated_u_pair;
  }
#pragma unroll
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U)
    partial += __shfl_down_sync(0xffffffffU, partial, stride);
  if (lane == 0U)
    dgate[padded_row] = partial;
}

// A persistent compute CTA has twelve warps. Give rows to those warps in
// compile-time-unrolled rounds and keep one FP32 dGate accumulator per lane so
// the elementwise epilogue does not extend eight independent live ranges into
// the surrounding WGMMA scheduler.
template <std::uint32_t RowsPerTask>
__device__ __forceinline__ void backward_swiglu_epilogue_warp_rows(
    const BackwardKernelArgs &a, std::uint64_t first_padded_row,
    std::uint64_t first_compact_row, std::uint32_t valid_rows,
    std::uint32_t rows) {
  constexpr std::uint32_t kWarps = backward_tuning::kComputeThreads / 32U;
  static_assert(RowsPerTask >= kWarps);
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  if constexpr (RowsPerTask == kWarps) {
    if (warp >= rows)
      return;
    backward_swiglu_epilogue_warp_row(
        a, first_padded_row + warp, first_compact_row + warp,
        warp < valid_rows, lane);
  } else {
#pragma unroll
    for (std::uint32_t local_row = warp; local_row < RowsPerTask;
         local_row += kWarps) {
      if (local_row < rows)
        backward_swiglu_epilogue_warp_row(
            a, first_padded_row + local_row, first_compact_row + local_row,
            local_row < valid_rows, lane);
    }
  }
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
    marker.store(execution_epoch(a), ::cuda::memory_order_release);
  }
}

__global__ void backward_reduce_dx_kernel(BackwardKernelArgs a) {
  constexpr std::uint32_t kTrainingTopK = 6U;
  // One lane publishes each route pointer. The dynamic top-k path must not
  // index the six-entry storage used by the unrolled training fast path.
  constexpr std::uint32_t kMaxTopK = 32U;
  constexpr std::uint32_t kWarpsPerCta =
      backward_tuning::kReduceThreads / 32U;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto token = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerCta +
                     warp;
  if (token >= a.dispatch.identity.num_tokens)
    return;
  __shared__ const transport::BFloat16
      *source_rows[kWarpsPerCta][kMaxTopK];
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
  if (lane < a.dispatch.identity.top_k) {
    const auto position = lane;
    const auto slot = a.dispatch.combine_order[route_base + position];
    const auto route = route_base + slot;
    auto *source = contribution + route * a.hidden;
#if __has_include(<nccl_device.h>)
    if (a.use_gin && a.dispatch.route_dst_node[route] != a.node)
      source = reinterpret_cast<const transport::BFloat16 *>(
          direct_receive + direct_compact[route] * a.gin_return_row_bytes);
#endif
    source_rows[warp][position] = source;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        marker(ready[route_base + position]);
    while (marker.load(::cuda::memory_order_acquire) != execution_epoch(a))
      __nanosleep(backward_tuning::kReduceIdleNanosleep);
  }
  __syncwarp();

  auto *output = reinterpret_cast<uint4 *>(combined + token * a.hidden);
  for (std::uint64_t vector = lane; vector < a.hidden / 8ULL;
       vector += 32ULL) {
    float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 0.0F, 0.0F};
    if (a.dispatch.identity.top_k == kTrainingTopK) {
#pragma unroll
      for (std::uint32_t position = 0U; position < kTrainingTopK;
           ++position) {
        const auto packed = reinterpret_cast<const uint4 *>(
            source_rows[warp][position])[vector];
        const unsigned int words[4] = {packed.x, packed.y, packed.z,
                                       packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
          sums[word * 2U] = __fadd_rn(
              sums[word * 2U],
              bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
          sums[word * 2U + 1U] = __fadd_rn(
              sums[word * 2U + 1U],
              bf16_bits_to_float(
                  static_cast<std::uint16_t>(words[word] >> 16U)));
        }
      }
    } else {
      for (std::uint32_t position = 0U;
           position < a.dispatch.identity.top_k; ++position) {
        const auto packed = reinterpret_cast<const uint4 *>(
            source_rows[warp][position])[vector];
        const unsigned int words[4] = {packed.x, packed.y, packed.z,
                                       packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
          sums[word * 2U] = __fadd_rn(
              sums[word * 2U],
              bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
          sums[word * 2U + 1U] = __fadd_rn(
              sums[word * 2U + 1U],
              bf16_bits_to_float(
                  static_cast<std::uint16_t>(words[word] >> 16U)));
        }
      }
    }
    unsigned int words[4];
#pragma unroll
    for (std::uint32_t word = 0U; word < 4U; ++word)
      words[word] =
          static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U])) |
          (static_cast<unsigned int>(
               float_to_bf16_bits(sums[word * 2U + 1U]))
           << 16U);
    output[vector] = make_uint4(words[0], words[1], words[2], words[3]);
  }
}

#include "cuda/persistent_v3/backward/pipeline_kernels.cuh"
