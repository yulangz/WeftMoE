#include "cuda/backward/compute_kernels.cuh"
#include "cuda/backward/kernel_contract.cuh"

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

#include "cuda/backward/swiglu.cuh"

template <bool Stage = false, bool FP8 = false>
__device__ __forceinline__ void backward_swiglu_epilogue_warp_row(
    const BackwardKernelArgs &a, std::uint64_t padded_row,
    std::uint64_t compact_row, bool row_is_valid, std::uint32_t lane,
    transport::BFloat16 *staging = nullptr, float *staging_scales = nullptr) {
  auto *u = a.recomputed_u;
  const auto *ab = a.saved_ab;
  const auto *gates = a.saved_gates;
  auto *dab = a.dab;
  auto *dgate = a.dgate_rows;
  const auto b_base = a.saved_rows * a.intermediate;
  const auto output_base = padded_row * 2ULL * a.intermediate;

  float partial = 0.0F;
  const auto gate = row_is_valid ? gates[compact_row] : 0.0F;
  for (std::uint64_t column = 8ULL * lane; column < a.intermediate;
       column += 256U) {
    uint4 da{}, db{}, activation{};
    if (row_is_valid) {
      const auto output_index = padded_row * a.intermediate + column;
      const auto input_index = compact_row * a.intermediate + column;
      const auto av = *reinterpret_cast<const uint4 *>(ab + input_index);
      const auto bv =
          *reinterpret_cast<const uint4 *>(ab + b_base + input_index);
      const auto du = *reinterpret_cast<const uint4 *>(u + output_index);
      const unsigned as[4] = {av.x, av.y, av.z, av.w};
      const unsigned bs[4] = {bv.x, bv.y, bv.z, bv.w};
      const unsigned ds[4] = {du.x, du.y, du.z, du.w};
      unsigned das[4], dbs[4], us[4];
#pragma unroll
      for (unsigned word = 0U; word < 4U; ++word) {
        const auto result =
            backward_swiglu_pair(as[word], bs[word], ds[word], gate, partial);
        das[word] = result.da;
        dbs[word] = result.db;
        us[word] = result.u;
      }
      da = make_uint4(das[0], das[1], das[2], das[3]);
      db = make_uint4(dbs[0], dbs[1], dbs[2], dbs[3]);
      activation = make_uint4(us[0], us[1], us[2], us[3]);
    }
    *reinterpret_cast<uint4 *>(dab + output_base + column) = da;
    *reinterpret_cast<uint4 *>(dab + output_base + a.intermediate + column) =
        db;
    *reinterpret_cast<uint4 *>(u + padded_row * a.intermediate + column) =
        activation;
    if constexpr (Stage && FP8) {
      fp8::store_fp8_eight(da, reinterpret_cast<std::uint8_t *>(staging),
          unsigned(2U * a.intermediate), unsigned(column), staging_scales, staging_scales ? 128U : 1U);
      fp8::store_fp8_eight(db, reinterpret_cast<std::uint8_t *>(staging),
          unsigned(2U * a.intermediate), unsigned(a.intermediate + column), staging_scales, staging_scales ? 128U : 1U);
    } else if constexpr (Stage) {
      *reinterpret_cast<uint4 *>(staging + column) = da;
      *reinterpret_cast<uint4 *>(staging + a.intermediate + column) = db;
    }
  }
#pragma unroll
  for (unsigned stride = 16U; stride; stride >>= 1U)
    partial += __shfl_down_sync(0xffffffffU, partial, stride);
  if (!lane)
    dgate[padded_row] = partial;
}

#include "cuda/backward/pipeline_kernels.cuh"
#include "cuda/weftmoe_pf/backward_weights.cuh"
