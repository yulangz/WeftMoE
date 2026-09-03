#include "cuda/backward/kernel_contract.cuh"

__global__ void backward_setup_local_kernel(BackwardKernelArgs a) {
  if (blockIdx.x != 0U)
    return;
  auto *dispatch_arrived = peer_at<unsigned long long>(
      a.local_base, a.dispatch_arrived_offset);
  auto *return_arrived = peer_at<unsigned long long>(
      a.local_base, a.return_arrived_offset);
  if (threadIdx.x == 0U) {
    *dispatch_arrived = 0ULL;
    *return_arrived = 0ULL;
  }
  __syncthreads();
  __threadfence_system();
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
      __nanosleep(128U);
  }
}

__global__ void backward_dispatch_dy_local_kernel(BackwardKernelArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x);
  if (route >= a.dispatch.sizes.route_count)
    return;
  const auto destination = a.dispatch.route_dst_rank[route];
  const auto destination_row = a.dispatch.route_dst_row[route];
  const auto token = route / a.dispatch.identity.top_k;
  auto *destination_dr = peer_at<transport::BFloat16>(
      a.peer_bases[destination], a.canonical_dr_offset);
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
    auto *arrived = peer_at<unsigned long long>(
        a.peer_bases[destination], a.dispatch_arrived_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        count(*arrived);
    count.fetch_add(1ULL, ::cuda::memory_order_release);
  }
}

__global__ void backward_wait_for_dispatch_kernel(BackwardKernelArgs a,
                                                   std::uint64_t rows) {
  if (blockIdx.x != 0U || threadIdx.x != 0U)
    return;
  auto *arrived = peer_at<unsigned long long>(
      a.local_base, a.dispatch_arrived_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
      count(*arrived);
  while (count.load(::cuda::memory_order_acquire) < rows)
    __nanosleep(256U);
}

__global__ void backward_swiglu_epilogue_kernel(BackwardKernelArgs a,
                                                 std::uint64_t rows) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= rows)
    return;
  const auto *du = peer_at<const transport::BFloat16>(
      a.local_base, a.dx_rows_offset);
  const auto *ab = peer_at<const transport::BFloat16>(
      a.local_base, a.cached_ab_offset);
  const auto *gate = peer_at<const float>(a.local_base,
                                          a.canonical_gate_offset);
  auto *u = peer_at<transport::BFloat16>(a.local_base,
                                         a.recomputed_u_offset);
  auto *dab = peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_rows_offset);
  float partial = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < a.intermediate;
       column += blockDim.x) {
    const auto index = row * a.intermediate + column;
    const auto av = bf16_bits_to_float(ab[index].bits);
    const auto bv = bf16_bits_to_float(
        ab[a.max_destination_rows * a.intermediate + index].bits);
    const auto duv = bf16_bits_to_float(du[index].bits);
    const auto sigmoid = 1.0F / (1.0F + expf(-av));
    const auto silu = av * sigmoid;
    const auto s = silu * bv;
    const auto ds = gate[row] * duv;
    const auto silu_derivative =
        sigmoid * (1.0F + av * (1.0F - sigmoid));
    const auto output_base = row * 2ULL * a.intermediate;
    dab[output_base + column] = transport::BFloat16{
        float_to_bf16_bits(ds * bv * silu_derivative)};
    dab[output_base + a.intermediate + column] =
        transport::BFloat16{float_to_bf16_bits(ds * silu)};
    u[index] = transport::BFloat16{
        float_to_bf16_bits(gate[row] * s)};
    partial = fmaf(duv, s, partial);
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0U;
       stride >>= 1U) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U)
    dgate[row] = reduction[0];
}

__global__ void backward_return_local_kernel(BackwardKernelArgs a,
                                              std::uint64_t rows) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= rows)
    return;
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  const auto header = headers[row];
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
    auto *arrived = peer_at<unsigned long long>(
        a.peer_bases[header.source_rank], a.return_arrived_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        count(*arrived);
    count.fetch_add(1ULL, ::cuda::memory_order_release);
  }
}

__global__ void backward_wait_for_return_kernel(BackwardKernelArgs a) {
  if (blockIdx.x != 0U || threadIdx.x != 0U)
    return;
  auto *arrived = peer_at<unsigned long long>(
      a.local_base, a.return_arrived_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
      count(*arrived);
  while (count.load(::cuda::memory_order_acquire) <
         a.dispatch.sizes.route_count)
    __nanosleep(256U);
}

__global__ void backward_reduce_dx_kernel(BackwardKernelArgs a) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= a.dispatch.identity.num_tokens)
    return;
  const auto *contribution = peer_at<const transport::BFloat16>(
      a.local_base, a.contribution_offset);
  auto *combined = peer_at<transport::BFloat16>(
      a.local_base, a.combined_dx_offset);
  for (std::uint64_t column = threadIdx.x; column < a.hidden;
       column += blockDim.x) {
    float sum = 0.0F;
    const auto route_base = token * a.dispatch.identity.top_k;
    for (std::uint32_t position = 0U;
         position < a.dispatch.identity.top_k; ++position) {
      const auto slot = a.dispatch.combine_order[route_base + position];
      const auto route = route_base + slot;
      sum += bf16_bits_to_float(
          contribution[route * a.hidden + column].bits);
    }
    combined[token * a.hidden + column] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
}

