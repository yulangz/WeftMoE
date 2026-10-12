#include "cuda/backward/kernel_contract.cuh"
#include "cuda/weftmoe_pf/backward_return_plan.cuh"
#include "cuda/weftmoe_pf/backward_return_access.cuh"
#include "cuda/weftmoe_pf/backward_return_hooks.cuh"
#include "cuda/backward/compute_kernels.cuh"

// Rebuilt for each invocation: routing counts may change even at fixed shape.
// Capacity was checked when the pool and dispatch handle were constructed.
__global__ void backward_build_row_directory_kernel(BackwardKernelArgs a) {
  auto *offsets =
      peer_at<unsigned long long>(a.local_base, a.padded_expert_offsets_offset);
  auto *rows =
      peer_at<unsigned long long>(a.local_base, a.padded_expert_rows_offset);
  auto *mapping =
      peer_at<unsigned int>(a.local_base, a.compact_to_padded_offset);
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
    for (unsigned long long row = threadIdx.x;
         row < a.dispatch.expert_row_count[e]; row += blockDim.x)
      mapping[a.dispatch.expert_row_prefix[e] + row] =
          static_cast<unsigned int>(offsets[e] + row);
}

__global__ void backward_clear_inactive_gradients_kernel(BackwardKernelArgs a) {
  const auto expert = blockIdx.y;
  if (a.dispatch.expert_row_count[expert] != 0ULL)
    return;
  const auto elements = a.hidden * a.intermediate;
  auto *dw1 = a.dw1 + expert * 2ULL * elements;
  auto *dw2 = a.dw2 + expert * elements;
  for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
                threadIdx.x;
       i < 2ULL * elements; i += gridDim.x * blockDim.x) {
    reinterpret_cast<unsigned short *>(dw1)[i] = 0;
    if (i < elements)
      reinterpret_cast<unsigned short *>(dw2)[i] = 0;
  }
}

__device__ __forceinline__ void
backward_load_row_geometry(BackwardKernelArgs &a) {
  __shared__ unsigned long long geometry[2];
  if (threadIdx.x == 0U) {
    const auto *offsets = peer_at<const unsigned long long>(
        a.local_base, a.padded_expert_offsets_offset);
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
  for (std::uint32_t peer = threadIdx.x; peer < a.world; peer += blockDim.x) {
    auto *setup =
        peer_at<unsigned long long>(a.peer_bases[peer], a.setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(setup[a.rank]);
    published.store(execution_epoch(a), ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_setup =
      peer_at<unsigned long long>(a.local_base, a.setup_epoch_offset);
  for (std::uint32_t peer = threadIdx.x; peer < a.world; peer += blockDim.x) {
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
  auto *destination_dr = a.peer_payloads[destination].dr;
  const auto *source = reinterpret_cast<const uint4 *>(a.dy + token * a.hidden);
  auto *target = reinterpret_cast<uint4 *>(destination_dr +
                                           compact_destination_row * a.hidden);
  const auto vectors = a.hidden / 8ULL;
  for (std::uint64_t vector = threadIdx.x; vector < vectors;
       vector += blockDim.x)
    target[vector] = source[vector];
  __syncthreads();
  if (threadIdx.x == 0U) {
    __threadfence_system();
    auto *ready = peer_at<unsigned long long>(a.peer_bases[destination],
                                              a.dispatch_ready_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> marker(
        ready[compact_destination_row]);
    marker.store(execution_epoch(a), ::cuda::memory_order_release);
  }
}

__global__ void backward_wait_for_dispatch_rows_kernel(BackwardKernelArgs a,
                                                       std::uint64_t rows) {
  const auto compact_row = static_cast<std::uint64_t>(blockIdx.x);
  if (compact_row >= rows || threadIdx.x != 0U)
    return;
  auto *ready =
      peer_at<unsigned long long>(a.local_base, a.dispatch_ready_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> marker(
      ready[compact_row]);
  // Each fixed row has one producer.  This kernel is only the correctness
  // scaffold's launch boundary; the persistent implementation consumes the
  // same marker directly when it claims a ready tile.
  while (marker.load(::cuda::memory_order_acquire) != execution_epoch(a))
    __nanosleep(backward_tuning::kReduceIdleNanosleep);
}

#include "cuda/backward/swiglu.cuh"

template <bool Stage = false, unsigned VectorWords = 1U>
__device__ __forceinline__ void backward_swiglu_epilogue_warp_row(
    const BackwardKernelArgs &a, std::uint64_t padded_row,
    std::uint64_t compact_row, bool row_is_valid, std::uint32_t lane,
    transport::BFloat16 *staging = nullptr) {
  auto *u = a.recomputed_u;
  const auto *ab = a.saved_ab;
  const auto *gates = a.saved_gates;
  auto *dab = a.dab;
  auto *dgate = a.dgate_rows;
  const auto b_base = a.saved_rows * a.intermediate;
  const auto output_base = padded_row * 2ULL * a.intermediate;
  static_assert(VectorWords == 1U || VectorWords == 4U);
  if constexpr (VectorWords == 4U) {
    float partial = 0.0F;
    const auto gate = row_is_valid ? gates[compact_row] : 0.0F;
    for (std::uint64_t column = 8ULL * lane; column < a.intermediate;
         column += 256U) {
      uint4 da{}, db{}, activation{};
      if (row_is_valid) {
        const auto output_index = padded_row * a.intermediate + column;
        const auto input_index = compact_row * a.intermediate + column;
        const auto av = *reinterpret_cast<const uint4 *>(ab + input_index);
        const auto bv = *reinterpret_cast<const uint4 *>(ab + b_base + input_index);
        const auto du = *reinterpret_cast<const uint4 *>(u + output_index);
        const unsigned as[4] = {av.x, av.y, av.z, av.w};
        const unsigned bs[4] = {bv.x, bv.y, bv.z, bv.w};
        const unsigned ds[4] = {du.x, du.y, du.z, du.w};
        unsigned das[4], dbs[4], us[4];
#pragma unroll
        for (unsigned word = 0U; word < 4U; ++word) {
          const auto result = backward_swiglu_pair(as[word], bs[word], ds[word],
                                                    gate, partial);
          das[word] = result.da;
          dbs[word] = result.db;
          us[word] = result.u;
        }
        da = make_uint4(das[0], das[1], das[2], das[3]);
        db = make_uint4(dbs[0], dbs[1], dbs[2], dbs[3]);
        activation = make_uint4(us[0], us[1], us[2], us[3]);
      }
      *reinterpret_cast<uint4 *>(dab + output_base + column) = da;
      *reinterpret_cast<uint4 *>(dab + output_base + a.intermediate + column) = db;
      *reinterpret_cast<uint4 *>(u + padded_row * a.intermediate + column) = activation;
      if constexpr (Stage) {
        *reinterpret_cast<uint4 *>(staging + column) = da;
        *reinterpret_cast<uint4 *>(staging + a.intermediate + column) = db;
      }
    }
#pragma unroll
    for (unsigned stride = 16U; stride; stride >>= 1U)
      partial += __shfl_down_sync(0xffffffffU, partial, stride);
    if (!lane)
      dgate[padded_row] = partial;
    return;
  }
  if (!row_is_valid) {
    for (std::uint64_t column = 2ULL * lane; column < a.intermediate;
         column += 64U) {
      const auto index = padded_row * a.intermediate + column;
      *reinterpret_cast<std::uint32_t *>(u + index) = 0U;
      *reinterpret_cast<std::uint32_t *>(dab + output_base + column) = 0U;
      *reinterpret_cast<std::uint32_t *>(dab + output_base + a.intermediate +
                                         column) = 0U;
      if constexpr (Stage) {
        *reinterpret_cast<unsigned *>(staging + column) = 0U;
        *reinterpret_cast<unsigned *>(staging + a.intermediate + column) = 0U;
      }
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
    const auto result = backward_swiglu_pair(a_pair, b_pair, du_pair, gate, partial);
    *reinterpret_cast<std::uint32_t *>(dab + output_base + column) = result.da;
    *reinterpret_cast<std::uint32_t *>(dab + output_base + a.intermediate +
                                       column) = result.db;
    *reinterpret_cast<std::uint32_t *>(u + output_index) = result.u;
    if constexpr (Stage) {
      *reinterpret_cast<unsigned *>(staging + column) = result.da;
      *reinterpret_cast<unsigned *>(staging + a.intermediate + column) = result.db;
    }
  }
#pragma unroll
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U)
    partial += __shfl_down_sync(0xffffffffU, partial, stride);
  if (lane == 0U)
    dgate[padded_row] = partial;
}

// Give rows to the arithmetic warps in compile-time-unrolled rounds. A unified
// CTA leaves its producer and communication warps out of this epilogue; the
// lane ownership and FP32 dGate shuffle tree within each row stay unchanged.
template <std::uint32_t RowsPerTask,
          std::uint32_t WorkWarps = backward_tuning::kComputeThreads / 32U>
__device__ __forceinline__ void backward_swiglu_epilogue_warp_rows(
    const BackwardKernelArgs &a, std::uint64_t first_padded_row,
    std::uint64_t first_compact_row, std::uint32_t valid_rows,
    std::uint32_t rows) {
  constexpr std::uint32_t kWarps = WorkWarps;
  static_assert(kWarps > 0U);
  static_assert(RowsPerTask >= kWarps);
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  if (warp >= kWarps)
    return;
  if constexpr (RowsPerTask == kWarps) {
    if (warp >= rows)
      return;
    backward_swiglu_epilogue_warp_row(a, first_padded_row + warp,
                                      first_compact_row + warp,
                                      warp < valid_rows, lane);
  } else {
#pragma unroll
    for (std::uint32_t local_row = warp; local_row < RowsPerTask;
         local_row += kWarps) {
      if (local_row < rows)
        backward_swiglu_epilogue_warp_row(a, first_padded_row + local_row,
                                          first_compact_row + local_row,
                                          local_row < valid_rows, lane);
    }
  }
}

__global__ void backward_return_local_kernel(BackwardKernelArgs a,
                                             std::uint64_t rows) {
  const auto compact_row = static_cast<std::uint64_t>(blockIdx.x);
  if (compact_row >= rows)
    return;
  const auto *compact_to_padded =
      peer_at<const unsigned int>(a.local_base, a.compact_to_padded_offset);
  const auto row = compact_to_padded[compact_row];
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  const auto header = headers[compact_row];
  const auto *dx = a.dx_rows;
  const auto *dgate = a.dgate_rows;
  auto *source_contribution = a.peer_payloads[header.source_rank].contribution;
  const auto *source = reinterpret_cast<const uint4 *>(dx + row * a.hidden);
  auto *target = reinterpret_cast<uint4 *>(source_contribution +
                                           header.route_index * a.hidden);
  const auto vectors = a.hidden / 8ULL;
  for (std::uint64_t vector = threadIdx.x; vector < vectors;
       vector += blockDim.x)
    target[vector] = source[vector];
  if (threadIdx.x == 0U) {
    auto *source_dgate = a.peer_payloads[header.source_rank].dgate;
    source_dgate[header.route_index] = dgate[row];
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    __threadfence_system();
    auto *ready = peer_at<unsigned long long>(a.peer_bases[header.source_rank],
                                              a.return_ready_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> marker(
        ready[header.route_index]);
    marker.store(execution_epoch(a), ::cuda::memory_order_release);
  }
}

// Shared by the standalone reducer and the retired communication warps.
// One pointer per lane replaces the shared table. Canonical route order,
// readiness acquire and output packing are identical.
template <bool Wait = true>
__device__ __forceinline__ bool
backward_reduce_dx_token(const BackwardKernelArgs &a, std::uint64_t token) {
  constexpr std::uint32_t kTrainingTopK = 6U;
  // Keep one route pointer per lane. Warp broadcasts preserve canonical
  // top-k order without a separate shared-memory pointer table.
  const auto lane = threadIdx.x & 31U;
  if (token >= a.dispatch.identity.num_tokens)
    return false;
  const auto *contribution = a.peer_payloads[a.rank].contribution;
  auto *combined = a.dx;
  auto *ready =
      peer_at<unsigned long long>(a.local_base, a.return_ready_offset);
  const auto route_base = token * a.dispatch.identity.top_k;
  bool available = true;
  unsigned long long source_address = 0ULL;
  if (lane < a.dispatch.identity.top_k) {
    const auto position = lane;
    const auto slot = a.dispatch.combine_order[route_base + position];
    const auto route = route_base + slot;
    const auto row =
        a.return_ring.depth
            ? weftmoe_pf::backward_return::contribution_row(a, route)
            : route;
    auto *source = contribution + row * a.hidden;
    source_address = reinterpret_cast<unsigned long long>(source);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> marker(
        ready[route_base + position]);
    if constexpr (Wait) {
      while (marker.load(::cuda::memory_order_acquire) != execution_epoch(a))
        __nanosleep(backward_tuning::kReduceIdleNanosleep);
    } else {
      available =
          marker.load(::cuda::memory_order_acquire) == execution_epoch(a);
    }
  }
  if constexpr (!Wait)
    if (!__all_sync(0xffffffffU, available))
      return false;
  __syncwarp();

  auto *output = reinterpret_cast<uint4 *>(combined + token * a.hidden);
  for (std::uint64_t vector = lane; vector < a.hidden / 8ULL; vector += 32ULL) {
    float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    if (a.dispatch.identity.top_k == kTrainingTopK) {
#pragma unroll
      for (std::uint32_t position = 0U; position < kTrainingTopK; ++position) {
        const auto packed = reinterpret_cast<const uint4 *>(
            __shfl_sync(0xffffffffU, source_address, position))[vector];
        const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
          sums[word * 2U] = __fadd_rn(
              sums[word * 2U],
              bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
          sums[word * 2U + 1U] =
              __fadd_rn(sums[word * 2U + 1U],
                        bf16_bits_to_float(
                            static_cast<std::uint16_t>(words[word] >> 16U)));
        }
      }
    } else {
      for (std::uint32_t position = 0U; position < a.dispatch.identity.top_k;
           ++position) {
        const auto packed = reinterpret_cast<const uint4 *>(
            __shfl_sync(0xffffffffU, source_address, position))[vector];
        const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
          sums[word * 2U] = __fadd_rn(
              sums[word * 2U],
              bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
          sums[word * 2U + 1U] =
              __fadd_rn(sums[word * 2U + 1U],
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
          (static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U + 1U]))
           << 16U);
    output[vector] = make_uint4(words[0], words[1], words[2], words[3]);
  }
  if (a.return_ring.depth) {
    if (lane < a.dispatch.identity.top_k) {
      const auto route = route_base + lane;
      a.dgate[route] =
          a.peer_payloads[a.rank]
              .dgate[weftmoe_pf::backward_return::contribution_row(a, route)];
    }
    __syncwarp();
    // Full source ranges retire at the invocation join, not once per token.
    if (lane == 0U && !a.return_ring.canonical) {
      const auto window = token / a.return_ring.shape.tokens;
      weftmoe_pf::ring::arrive(
          a.peer_payloads[a.rank]
              .consumed[window % a.return_ring.contribution_depth],
          1U);
    }
  }
  return true;
}

__global__ void backward_reduce_dx_kernel(BackwardKernelArgs a) {
  constexpr std::uint32_t kWarpsPerCta = backward_tuning::kReduceThreads / 32U;
  const auto warp = threadIdx.x / 32U;
  const auto token =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerCta + warp;
  backward_reduce_dx_token(a, token);
}

#include "cuda/backward/pipeline_kernels.cuh"
#include "cuda/backward/unified_kernel.cuh"
