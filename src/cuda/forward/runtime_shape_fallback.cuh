// Copyright (c) MegaMOE contributors.
// Internal implementation detail: this file is included by
// activation_transport.cu after the production argument/status types exist.
// It deliberately has no public declarations.

__device__ std::uint32_t runtime_shape_owned_expert_for_row(
    const DeviceDispatchKernelView &dispatch, std::uint64_t row) {
  for (std::uint32_t local = 0; local < dispatch.sizes.owned_expert_count;
       ++local) {
    const auto begin = dispatch.expert_row_prefix[local];
    if (row >= begin && row < begin + dispatch.expert_row_count[local])
      return local;
  }
  return dispatch.sizes.owned_expert_count;
}

__device__ float runtime_shape_dot(const transport::BFloat16 *left,
                                   const transport::BFloat16 *right,
                                   std::uint64_t count,
                                   std::uint64_t right_stride) {
  float sum = 0.0F;
  for (std::uint64_t k = 0; k < count; ++k)
    sum = __fadd_rn(sum,
                    __fmul_rn(bf16_bits_to_float(left[k].bits),
                              bf16_bits_to_float(right[k * right_stride].bits)));
  return sum;
}

// Correctness-only runtime H/I Forward.  One cooperative CTA is intentional:
// it provides deterministic phase ordering without host GEMM launches.  The
// optimized 7168x3072 path never instantiates or calls this kernel.
__global__ void runtime_shape_forward_megakernel(ForwardKernelArgs a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *canonical = peer_at<transport::BFloat16>(a.local_base,
                                                  a.canonical_rows_offset);
  auto *headers = peer_at<LocalP2pRouteHeader>(a.local_base,
                                               a.direct_headers_offset);
  auto *s = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  auto *ab = a.forward_ab_cache_offset == 0
                 ? static_cast<transport::BFloat16 *>(nullptr)
                 : peer_at<transport::BFloat16>(a.local_base,
                                                 a.forward_ab_cache_offset);
  auto *expert_output = peer_at<transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  auto *contribution = peer_at<transport::BFloat16>(a.local_base,
                                                     a.contribution_offset);
  auto *combined = peer_at<transport::BFloat16>(a.local_base,
                                                 a.combined_offset);
  if (threadIdx.x == 0U) {
    status->start_clock = device_globaltimer_ns();
    status->runtime_cluster_size = 1U;
    status->cluster_rank_mask = 1ULL;
  }
  __syncthreads();

  if (a.use_gin) {
    if (threadIdx.x == 0U)
      publish_local_structured_error(status, a.rank, 201U, 1U, 0ULL, 1ULL);
  } else if (!a.use_socket) {
    const auto dispatch_elements = a.dispatch.sizes.route_count * a.hidden;
    for (std::uint64_t item = threadIdx.x; item < dispatch_elements;
         item += blockDim.x) {
      const auto route = item / a.hidden;
      const auto h = item % a.hidden;
      const auto destination = a.dispatch.route_dst_rank[route];
      auto *destination_rows = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.canonical_rows_offset);
      destination_rows[a.dispatch.route_dst_row[route] * a.hidden + h] =
          a.input[(route / a.dispatch.identity.top_k) * a.hidden + h];
    }
    __syncthreads();
    for (std::uint64_t route = threadIdx.x;
         route < a.dispatch.sizes.route_count; route += blockDim.x) {
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      destination_headers[row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          a.rank,
          destination,
          0U,
          a.dispatch.identity.generation,
          a.epoch,
          route,
          route,
          route / a.dispatch.identity.top_k,
          row,
          a.dispatch.expert_id[route],
          static_cast<std::uint32_t>(route % a.dispatch.identity.top_k)};
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
      for (std::uint32_t peer = 0; peer < a.world; ++peer) {
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[peer], a.ingress_published_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.ingress_published_offset);
      for (std::uint32_t source = 0; source < a.world; ++source) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(signals[source * a.channels]);
        while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    status->dispatch_complete_clock = device_globaltimer_ns();

  const auto incoming = a.dispatch.sizes.incoming_row_count;
  const auto fc1_elements = incoming * a.intermediate;
  for (std::uint64_t item = threadIdx.x; item < fc1_elements;
       item += blockDim.x) {
    const auto row = item / a.intermediate;
    const auto column = item % a.intermediate;
    const auto local_expert = runtime_shape_owned_expert_for_row(a.dispatch, row);
    if (local_expert >= a.dispatch.sizes.owned_expert_count) {
      publish_local_structured_error(status, a.rank, 202U, 1U,
                                     a.dispatch.sizes.owned_expert_count,
                                     local_expert);
      continue;
    }
    const auto *x = canonical + row * a.hidden;
    const auto weight_base = static_cast<std::uint64_t>(local_expert) *
                             2ULL * a.intermediate * a.hidden;
    const auto *wa = a.w1 + weight_base + column * a.hidden;
    const auto *wb = a.w1 + weight_base +
                     (a.intermediate + column) * a.hidden;
    const auto av_bits = float_to_bf16_bits(runtime_shape_dot(x, wa, a.hidden, 1));
    const auto bv_bits = float_to_bf16_bits(runtime_shape_dot(x, wb, a.hidden, 1));
    const auto av = bf16_bits_to_float(av_bits);
    const auto bv = bf16_bits_to_float(bv_bits);
    const auto sigmoid = 1.0F / (1.0F + __expf(-av));
    s[item] = transport::BFloat16{
        float_to_bf16_bits(__fmul_rn(__fmul_rn(av, sigmoid), bv))};
    if (a.cache_fc1 && ab != nullptr) {
      ab[item] = transport::BFloat16{av_bits};
      ab[a.max_destination_rows * a.intermediate + item] =
          transport::BFloat16{bv_bits};
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    status->fc1_complete_clock = device_globaltimer_ns();

  const auto fc2_elements = incoming * a.hidden;
  for (std::uint64_t item = threadIdx.x; item < fc2_elements;
       item += blockDim.x) {
    const auto row = item / a.hidden;
    const auto h = item % a.hidden;
    const auto local_expert = runtime_shape_owned_expert_for_row(a.dispatch, row);
    if (local_expert >= a.dispatch.sizes.owned_expert_count)
      continue;
    const auto *row_s = s + row * a.intermediate;
    const auto *weight = a.w2 +
        (static_cast<std::uint64_t>(local_expert) * a.hidden + h) *
            a.intermediate;
    expert_output[item] = transport::BFloat16{
        float_to_bf16_bits(runtime_shape_dot(row_s, weight,
                                             a.intermediate, 1))};
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    status->fc2_complete_clock = device_globaltimer_ns();

  if (!a.use_socket && !a.use_gin) {
    for (std::uint64_t item = threadIdx.x; item < fc2_elements;
         item += blockDim.x) {
      const auto row = item / a.hidden;
      const auto h = item % a.hidden;
      const auto header = headers[row];
      auto *source = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.contribution_offset);
      source[header.route_index * a.hidden + h] = expert_output[item];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
      for (std::uint32_t source = 0; source < a.world; ++source) {
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[source], a.outgoing_sequence_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.outgoing_sequence_offset);
      for (std::uint32_t destination = 0; destination < a.world;
           ++destination) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(signals[destination * a.channels]);
        while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
        }
      }
    }
    __syncthreads();
    const auto output_elements = a.dispatch.identity.num_tokens * a.hidden;
    for (std::uint64_t item = threadIdx.x; item < output_elements;
         item += blockDim.x) {
      const auto token = item / a.hidden;
      const auto h = item % a.hidden;
      float sum = 0.0F;
      for (std::uint32_t order = 0;
           order < a.dispatch.identity.top_k; ++order) {
        const auto slot = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + slot;
        sum = gate_after_accumulate(
            sum, a.gates[route], contribution[route * a.hidden + h].bits);
      }
      combined[item] = transport::BFloat16{float_to_bf16_bits(sum)};
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    status->fc1_tasks_done = incoming == 0 ? 0 : 1;
    status->fc2_tasks_done = incoming == 0 ? 0 : 1;
    status->return_complete_clock = device_globaltimer_ns();
    status->complete_clock = status->return_complete_clock;
  }
  if (a.injected_status_error_rank == static_cast<int>(a.rank) &&
      threadIdx.x == 0U)
    publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
  __syncthreads();
  publish_production_terminal_status(
      a, status, ProductionStatusPhase::Forward,
      a.production_error_records_offset,
      a.gin_production_error_records_offset);
}

// Correctness-only runtime H/I Backward.  The phase order is fixed inside one
// persistent CTA; all reductions follow canonical K/route order and never use
// floating-point atomics.
__global__ void runtime_shape_backward_megakernel(BackwardKernelArgs a) {
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *canonical_x = peer_at<transport::BFloat16>(a.local_base,
                                                    a.canonical_x_offset);
  auto *headers = peer_at<LocalP2pRouteHeader>(a.local_base,
                                               a.direct_headers_offset);
  auto *dy = peer_at<transport::BFloat16>(a.local_base,
                                          a.dy_canonical_offset);
  auto *canonical_gate = peer_at<float>(a.local_base,
                                         a.canonical_gate_offset);
  auto *ab = a.use_forward_cache
                 ? peer_at<transport::BFloat16>(a.local_base,
                                                a.forward_ab_cache_offset)
                 : peer_at<transport::BFloat16>(a.local_base,
                                                a.recompute_ab_offset);
  auto *s = peer_at<transport::BFloat16>(a.local_base, a.backward_s_offset);
  auto *dab = peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
  auto *dx_canonical = peer_at<transport::BFloat16>(a.local_base,
                                                     a.dx_canonical_offset);
  auto *dx_return = peer_at<transport::BFloat16>(a.local_base,
                                                  a.dx_return_offset);
  auto *dx_combined = peer_at<transport::BFloat16>(a.local_base,
                                                    a.dx_combined_offset);
  auto *dgate_canonical = peer_at<float>(a.local_base,
                                          a.dgate_canonical_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_offset);
  auto *dw1 = peer_at<transport::BFloat16>(a.local_base, a.dw1_offset);
  auto *dw2 = peer_at<transport::BFloat16>(a.local_base, a.dw2_offset);
  if (threadIdx.x == 0U)
    status->start_clock = device_globaltimer_ns();
  __syncthreads();

  if (a.use_gin) {
    if (threadIdx.x == 0U)
      publish_local_structured_error(status, a.rank, 301U, 1U, 0ULL, 1ULL);
  } else if (!a.use_socket) {
    const auto elements = a.dispatch.sizes.route_count * a.hidden;
    for (std::uint64_t item = threadIdx.x; item < elements;
         item += blockDim.x) {
      const auto route = item / a.hidden;
      const auto h = item % a.hidden;
      const auto destination = a.dispatch.route_dst_rank[route];
      auto *destination_dy = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.dy_canonical_offset);
      destination_dy[a.dispatch.route_dst_row[route] * a.hidden + h] =
          a.dy_input[(route / a.dispatch.identity.top_k) * a.hidden + h];
    }
    __syncthreads();
    for (std::uint64_t route = threadIdx.x;
         route < a.dispatch.sizes.route_count; route += blockDim.x) {
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      destination_gate[row] = a.gates[route];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
      for (std::uint32_t peer = 0; peer < a.world; ++peer) {
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[peer], a.incoming_signal_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.incoming_signal_offset);
      for (std::uint32_t source = 0; source < a.world; ++source) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(signals[source * a.channels]);
        while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
               !backward_cancelled(a)) {
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    status->dy_dispatch_complete_clock = device_globaltimer_ns();

  const auto incoming = a.dispatch.sizes.incoming_row_count;
  const auto fc1_elements = incoming * a.intermediate;
  if (!a.use_forward_cache) {
    for (std::uint64_t item = threadIdx.x; item < fc1_elements;
         item += blockDim.x) {
      const auto row = item / a.intermediate;
      const auto column = item % a.intermediate;
      const auto local_expert = runtime_shape_owned_expert_for_row(a.dispatch, row);
      if (local_expert >= a.dispatch.sizes.owned_expert_count)
        continue;
      const auto *x = canonical_x + row * a.hidden;
      const auto weight_base = static_cast<std::uint64_t>(local_expert) *
                               2ULL * a.intermediate * a.hidden;
      ab[item] = transport::BFloat16{float_to_bf16_bits(runtime_shape_dot(
          x, a.w1 + weight_base + column * a.hidden, a.hidden, 1))};
      ab[a.max_destination_rows * a.intermediate + item] =
          transport::BFloat16{float_to_bf16_bits(runtime_shape_dot(
              x, a.w1 + weight_base + (a.intermediate + column) * a.hidden,
              a.hidden, 1))};
    }
    __syncthreads();
  }

  for (std::uint64_t item = threadIdx.x; item < fc1_elements;
       item += blockDim.x) {
    const auto row = item / a.intermediate;
    const auto column = item % a.intermediate;
    const auto local_expert = runtime_shape_owned_expert_for_row(a.dispatch, row);
    if (local_expert >= a.dispatch.sizes.owned_expert_count)
      continue;
    const auto av = bf16_bits_to_float(ab[item].bits);
    const auto bv = bf16_bits_to_float(
        ab[a.max_destination_rows * a.intermediate + item].bits);
    const auto sigmoid = 1.0F / (1.0F + __expf(-av));
    const auto silu = __fmul_rn(av, sigmoid);
    const auto s_bits = float_to_bf16_bits(__fmul_rn(silu, bv));
    s[item] = transport::BFloat16{s_bits};
    float dv = 0.0F;
    const auto *row_dy = dy + row * a.hidden;
    for (std::uint64_t h = 0; h < a.hidden; ++h) {
      const auto *weight = a.w2 +
          (static_cast<std::uint64_t>(local_expert) * a.hidden + h) *
              a.intermediate + column;
      dv = __fadd_rn(dv,
                     __fmul_rn(bf16_bits_to_float(row_dy[h].bits),
                               bf16_bits_to_float(weight->bits)));
    }
    const auto dv_bf16 = bf16_bits_to_float(float_to_bf16_bits(dv));
    const auto ds = __fmul_rn(canonical_gate[row], dv_bf16);
    dab[row * 2ULL * a.intermediate + column] =
        transport::BFloat16{float_to_bf16_bits(__fmul_rn(
            ds, __fmul_rn(bv, sigmoid * (1.0F + av * (1.0F - sigmoid)))))};
    dab[row * 2ULL * a.intermediate + a.intermediate + column] =
        transport::BFloat16{float_to_bf16_bits(__fmul_rn(ds, silu))};
  }
  __syncthreads();
  for (std::uint64_t row = threadIdx.x; row < incoming;
       row += blockDim.x) {
    const auto local_expert = runtime_shape_owned_expert_for_row(a.dispatch, row);
    float sum = 0.0F;
    for (std::uint64_t column = 0; column < a.intermediate; ++column) {
      float dv = 0.0F;
      for (std::uint64_t h = 0; h < a.hidden; ++h) {
        const auto *weight = a.w2 +
            (static_cast<std::uint64_t>(local_expert) * a.hidden + h) *
                a.intermediate + column;
        dv = __fadd_rn(dv,
                       __fmul_rn(bf16_bits_to_float(
                                     dy[row * a.hidden + h].bits),
                                 bf16_bits_to_float(weight->bits)));
      }
      const auto dv_bits = float_to_bf16_bits(dv);
      sum = __fadd_rn(sum,
                      __fmul_rn(bf16_bits_to_float(dv_bits),
                                bf16_bits_to_float(
                                    s[row * a.intermediate + column].bits)));
    }
    dgate_canonical[row] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    status->dv_complete_clock = device_globaltimer_ns();
    status->dgate_complete_clock = status->dv_complete_clock;
    status->transpose_complete_clock = status->dv_complete_clock;
  }

  const auto dx_elements = incoming * a.hidden;
  for (std::uint64_t item = threadIdx.x; item < dx_elements;
       item += blockDim.x) {
    const auto row = item / a.hidden;
    const auto h = item % a.hidden;
    const auto local_expert = runtime_shape_owned_expert_for_row(a.dispatch, row);
    float sum = 0.0F;
    const auto weight_base = static_cast<std::uint64_t>(local_expert) *
                             2ULL * a.intermediate * a.hidden;
    for (std::uint64_t k = 0; k < 2ULL * a.intermediate; ++k)
      sum = __fadd_rn(sum,
                      __fmul_rn(bf16_bits_to_float(
                                    dab[row * 2ULL * a.intermediate + k].bits),
                                bf16_bits_to_float(
                                    a.w1[weight_base + k * a.hidden + h].bits)));
    dx_canonical[item] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    status->dx_complete_clock = device_globaltimer_ns();

  const auto dw1_per_expert = 2ULL * a.intermediate * a.hidden;
  const auto dw1_elements =
      a.dispatch.sizes.owned_expert_count * dw1_per_expert;
  for (std::uint64_t item = threadIdx.x; item < dw1_elements;
       item += blockDim.x) {
    const auto local_expert = item / dw1_per_expert;
    const auto within = item % dw1_per_expert;
    const auto output = within / a.hidden;
    const auto h = within % a.hidden;
    float sum = 0.0F;
    const auto begin = a.dispatch.expert_row_prefix[local_expert];
    const auto count = a.dispatch.expert_row_count[local_expert];
    for (std::uint64_t k = 0; k < count; ++k) {
      const auto row = begin + k;
      sum = __fadd_rn(sum,
                      __fmul_rn(bf16_bits_to_float(
                                    dab[row * 2ULL * a.intermediate + output]
                                        .bits),
                                bf16_bits_to_float(
                                    canonical_x[row * a.hidden + h].bits)));
    }
    dw1[item] = transport::BFloat16{float_to_bf16_bits(sum)};
  }
  const auto dw2_per_expert = a.hidden * a.intermediate;
  const auto dw2_elements =
      a.dispatch.sizes.owned_expert_count * dw2_per_expert;
  for (std::uint64_t item = threadIdx.x; item < dw2_elements;
       item += blockDim.x) {
    const auto local_expert = item / dw2_per_expert;
    const auto within = item % dw2_per_expert;
    const auto h = within / a.intermediate;
    const auto column = within % a.intermediate;
    float sum = 0.0F;
    const auto begin = a.dispatch.expert_row_prefix[local_expert];
    const auto count = a.dispatch.expert_row_count[local_expert];
    for (std::uint64_t k = 0; k < count; ++k) {
      const auto row = begin + k;
      const auto u_bits = float_to_bf16_bits(__fmul_rn(
          canonical_gate[row],
          bf16_bits_to_float(s[row * a.intermediate + column].bits)));
      sum = __fadd_rn(sum,
                      __fmul_rn(bf16_bits_to_float(
                                    dy[row * a.hidden + h].bits),
                                bf16_bits_to_float(u_bits)));
    }
    dw2[item] = transport::BFloat16{float_to_bf16_bits(sum)};
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    status->dw1_complete_clock = device_globaltimer_ns();
    status->dw2_complete_clock = status->dw1_complete_clock;
  }

  if (!a.use_socket && !a.use_gin) {
    for (std::uint64_t item = threadIdx.x; item < dx_elements;
         item += blockDim.x) {
      const auto row = item / a.hidden;
      const auto h = item % a.hidden;
      const auto header = headers[row];
      auto *source_dx = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.dx_return_offset);
      source_dx[header.route_index * a.hidden + h] = dx_canonical[item];
    }
    for (std::uint64_t row = threadIdx.x; row < incoming;
         row += blockDim.x) {
      const auto header = headers[row];
      auto *source_dgate = peer_at<float>(a.peer_bases[header.source_rank],
                                           a.dgate_offset);
      source_dgate[header.route_index] = dgate_canonical[row];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
      for (std::uint32_t source = 0; source < a.world; ++source) {
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[source], a.outgoing_signal_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.outgoing_signal_offset);
      for (std::uint32_t destination = 0; destination < a.world;
           ++destination) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(signals[destination * a.channels]);
        while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
               !backward_cancelled(a)) {
        }
      }
    }
    __syncthreads();
    const auto result_elements = a.dispatch.identity.num_tokens * a.hidden;
    for (std::uint64_t item = threadIdx.x; item < result_elements;
         item += blockDim.x) {
      const auto token = item / a.hidden;
      const auto h = item % a.hidden;
      float sum = 0.0F;
      for (std::uint32_t order = 0;
           order < a.dispatch.identity.top_k; ++order) {
        const auto slot = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + slot;
        sum = __fadd_rn(sum, bf16_bits_to_float(
                                  dx_return[route * a.hidden + h].bits));
      }
      dx_combined[item] = transport::BFloat16{float_to_bf16_bits(sum)};
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    status->dx_reduce_complete_clock = device_globaltimer_ns();
    status->complete_clock = status->dx_reduce_complete_clock;
    status->padded_reduction_rows = incoming;
  }
  if (a.injected_status_error_rank == static_cast<int>(a.rank) &&
      threadIdx.x == 0U)
    publish_local_structured_error(status, a.rank, 320U, 1U, 0ULL, 1ULL);
  __syncthreads();
  publish_production_terminal_status(
      a, status, ProductionStatusPhase::Backward,
      a.production_error_records_offset,
      a.gin_production_error_records_offset);
}
