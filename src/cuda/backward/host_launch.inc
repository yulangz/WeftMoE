namespace {

void launch_backward_gemm(cublasHandle_t handle, cublasOperation_t op_a,
                          cublasOperation_t op_b, int m, int n, int k,
                          const transport::BFloat16 *a, int lda,
                          const transport::BFloat16 *b, int ldb,
                          transport::BFloat16 *c, int ldc, float beta,
                          std::uint32_t rank, const char *name) {
  constexpr float alpha = 1.0F;
  throw_cublas(
      "backward-gemm",
      cublasGemmEx(handle, op_a, op_b, m, n, k, &alpha, a, CUDA_R_16BF, lda,
                   b, CUDA_R_16BF, ldb, &beta, c, CUDA_R_16BF, ldc,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      rank, name);
}

double backward_elapsed_us(cudaEvent_t begin, cudaEvent_t end,
                           const ActivationTransportExpectation &identity,
                           const char *name) {
  float milliseconds = 0.0F;
  internal::throw_cuda(
      "backward-timing", cudaEventElapsedTime(&milliseconds, begin, end),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, name);
  return static_cast<double>(milliseconds) * 1000.0;
}

} // namespace

DeviceBackwardHandle BackwardMegaKernel::run(
    const DeviceForwardHandle &forward,
    const BFloat16OutputGradientLease &output_gradients,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) const {
  if (!forward.impl_ || !output_gradients.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "forward/output gradients", 0, "valid", "empty");
  const auto &retained = *forward.impl_;
  const auto &gradient = *output_gradients.impl_;
  const auto &identity = retained.identity;
  if (!retained.retained_backward_state || !retained.slot || !retained.pool ||
      !retained.backend || !retained.weights.impl_ ||
      !gradient.allocation || !gradient.allocation->ready)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "retained Forward state", identity.rank,
                    "complete cached X/A/B state and ready dY", "missing");
  if (config.hidden_size != 7168ULL ||
      config.intermediate_size != 3072ULL ||
      config.hidden_size != identity.hidden ||
      config.intermediate_size != retained.intermediate)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "specialization shape", identity.rank, "7168x3072",
                    std::to_string(config.hidden_size) + "x" +
                        std::to_string(config.intermediate_size));
  if (gradient.generation != identity.generation ||
      gradient.tokens != identity.tokens || gradient.hidden != identity.hidden ||
      gradient.device != retained.pool->device ||
      gradient.session_identity != retained.pool->session_identity)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "dY provenance", identity.rank,
                    "same generation/session/device/[tokens,H] as Forward",
                    "mismatch");
  if (options.cancellation.cancellation_requested() ||
      options.deadline_after.count() != 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "cancellation/deadline", identity.rank,
                    "disabled for local correctness milestone", "requested");
  const auto local_backend =
      std::dynamic_pointer_cast<LocalP2pBackendState>(retained.backend);
  if (!local_backend || identity.world != local_backend->peer_slot_bases.size())
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "transport backend", identity.rank,
                    "one-node LocalCudaP2p retained Forward", "non-local");
  const auto &pool = retained.pool;
  const auto &layout = pool->layout;
  if (!pool->config.enable_backward_state || !pool->backward_blas ||
      layout.backward_dab_rows == 0 || layout.backward_dx_rows == 0 ||
      layout.backward_dw1 == 0 || layout.backward_dw2 == 0)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "Backward workspace", identity.rank,
                    "training-enabled pool with owned expert capacity",
                    "missing");
  const auto &dispatch_id = retained.dispatch.identity();
  const auto expectation = DeviceDispatchViewExpectation{
      dispatch_id.generation, dispatch_id.calling_rank,
      dispatch_id.world_size, dispatch_id.num_tokens, dispatch_id.top_k,
      dispatch_id.config_digest_word0, dispatch_id.config_digest_word1};
  auto consumer = retained.dispatch.acquire_consumer(retained.backend->execution,
                                                      expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  if (view.sizes.incoming_row_count != retained.incoming_rows ||
      view.sizes.owned_expert_count != retained.weights.impl_->owned_experts)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "retained Dispatch geometry", identity.rank,
                    "same rows/experts as Forward", "mismatch");

  auto exec = internal::checked_execution_state(retained.backend->execution);
  {
    std::lock_guard<std::mutex> plan_lock(retained.backward_plan_mutex);
    if (retained.backward_expert_rows.empty()) {
      retained.backward_expert_rows.resize(view.sizes.owned_expert_count);
      retained.backward_expert_offsets.resize(
          static_cast<std::size_t>(view.sizes.owned_expert_count) + 1ULL);
      const auto rows_bytes = checked_size(
          checked_mul(view.sizes.owned_expert_count, sizeof(std::uint64_t),
                      "Backward expert row plan bytes"),
          "Backward expert row plan bytes");
      const auto offsets_bytes = checked_size(
          checked_mul(static_cast<std::uint64_t>(view.sizes.owned_expert_count) +
                          1ULL,
                      sizeof(std::uint64_t),
                      "Backward expert offset plan bytes"),
          "Backward expert offset plan bytes");
      internal::throw_cuda(
          "backward-plan",
          cudaMemcpyAsync(retained.backward_expert_rows.data(),
                          view.expert_row_count, rows_bytes,
                          cudaMemcpyDeviceToHost, exec->stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "D2H expert row counts");
      internal::throw_cuda(
          "backward-plan",
          cudaMemcpyAsync(retained.backward_expert_offsets.data(),
                          view.expert_row_prefix, offsets_bytes,
                          cudaMemcpyDeviceToHost, exec->stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "D2H expert row offsets");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                rows_bytes + offsets_bytes);
      internal::throw_cuda(
          "backward-plan", cudaStreamSynchronize(exec->stream), identity.rank,
          identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "cache expert row plan");
      retained.backward_padded_expert_rows.resize(
          view.sizes.owned_expert_count);
      retained.backward_padded_expert_offsets.resize(
          static_cast<std::size_t>(view.sizes.owned_expert_count) + 1ULL);
      retained.backward_compact_to_padded.resize(
          checked_size(view.sizes.incoming_row_count,
                       "Backward compact-to-padded entries"));
      std::uint64_t padded_cursor = 0ULL;
      for (std::uint32_t expert = 0U;
           expert < view.sizes.owned_expert_count; ++expert) {
        const auto rows = retained.backward_expert_rows[expert];
        const auto padded = checked_mul((rows + 127ULL) / 128ULL, 128ULL,
                                        "Backward padded expert rows");
        retained.backward_padded_expert_offsets[expert] = padded_cursor;
        retained.backward_padded_expert_rows[expert] = padded;
        const auto compact_begin = retained.backward_expert_offsets[expert];
        for (std::uint64_t local_row = 0ULL; local_row < rows; ++local_row) {
          retained.backward_compact_to_padded[checked_size(
              compact_begin + local_row,
              "Backward compact-to-padded index")] =
              static_cast<std::uint32_t>(padded_cursor + local_row);
        }
        padded_cursor = checked_add(padded_cursor, padded,
                                    "Backward padded row count");
      }
      retained.backward_padded_expert_offsets[view.sizes.owned_expert_count] =
          padded_cursor;
      retained.backward_padded_rows = padded_cursor;
      if (padded_cursor > layout.backward_padded_capacity_rows ||
          padded_cursor > UINT32_MAX)
        transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                        "backward-plan", "padded expert rows", identity.rank,
                        std::to_string(layout.backward_padded_capacity_rows),
                        std::to_string(padded_cursor));
    }
    if (!retained.backward_plan_uploaded) {
      const auto copy_plan = [&](std::size_t offset, const auto &values,
                                 const char *name) {
        if (values.empty())
          return;
        const auto bytes = checked_mul(values.size(), sizeof(values.front()),
                                       name);
        internal::throw_cuda(
            "backward-plan",
            cudaMemcpyAsync(
                at<void>(pool->slots[retained.slot->index].device, offset),
                            values.data(), checked_size(bytes, name),
                            cudaMemcpyHostToDevice, exec->stream),
            identity.rank, identity.world, "backward", identity.generation,
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                  bytes);
      };
      copy_plan(layout.backward_compact_to_padded,
                retained.backward_compact_to_padded,
                "upload Backward compact-to-padded directory");
      copy_plan(layout.backward_padded_expert_rows,
                retained.backward_padded_expert_rows,
                "upload Backward padded expert rows");
      copy_plan(layout.backward_padded_expert_offsets,
                retained.backward_padded_expert_offsets,
                "upload Backward padded expert offsets");
      internal::throw_cuda(
          "backward-plan", cudaStreamSynchronize(exec->stream), identity.rank,
          identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "publish padded Backward directory");
      retained.backward_plan_uploaded = true;
    }
  }

  auto &slot = pool->slots[retained.slot->index];
  SlotFailureGuard failure_guard{retained.slot, false};
  BackwardKernelArgs args{};
  args.dispatch = view;
  args.dy = gradient.pointer;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows = pool->config.max_destination_rows_per_rank;
  args.padded_capacity_rows = layout.backward_padded_capacity_rows;
  args.padded_rows = retained.backward_padded_rows;
  args.rank = identity.rank;
  args.world = identity.world;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.canonical_x_offset = layout.canonical_rows;
  args.canonical_gate_offset = layout.canonical_gates;
  args.cached_ab_offset = layout.forward_ab_cache;
  args.padded_x_offset = layout.backward_padded_x;
  args.padded_ab_offset = layout.backward_padded_ab;
  args.padded_dr_offset = layout.backward_padded_dr;
  args.recomputed_u_offset = layout.backward_u;
  args.dab_offset = layout.backward_dab_rows;
  args.dx_rows_offset = layout.backward_dx_rows;
  args.dgate_rows_offset = layout.backward_dgate_rows;
  args.dgate_output_offset = layout.backward_dgate_output;
  args.contribution_offset = layout.contribution;
  args.combined_dx_offset = layout.combined;
  args.direct_headers_offset = layout.direct_route_headers;
  args.compact_to_padded_offset = layout.backward_compact_to_padded;
  args.padded_expert_rows_offset = layout.backward_padded_expert_rows;
  args.padded_expert_offsets_offset = layout.backward_padded_expert_offsets;
  args.setup_epoch_offset = layout.backward_setup_epoch;
  args.dispatch_ready_offset = layout.backward_dispatch_ready;
  args.return_ready_offset = layout.backward_return_ready;
  args.epoch = ++slot.backward_epoch;
  if (args.epoch == 0ULL)
    args.epoch = ++slot.backward_epoch;

  auto transport_stream = slot.forward_transport_stream;
  auto compute_stream = slot.forward_compute_stream;
  internal::throw_cuda(
      "backward-run", cudaStreamWaitEvent(transport_stream,
                                           slot.forward_complete),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait retained Forward completion");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(transport_stream, gradient.allocation->ready),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait dY producer");
  backward_setup_local_kernel<<<1, 256, 0, transport_stream>>>(args);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), identity.rank, identity.world,
      "backward", identity.generation, ExchangeKind::ExpertCounts,
      "Backward setup launch");
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.backward_start, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward start");
  if (view.sizes.route_count != 0ULL) {
    backward_dispatch_dy_local_kernel<<<
        static_cast<unsigned int>(view.sizes.route_count), 256, 0,
        transport_stream>>>(args);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "Backward dY dispatch launch");
  }
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_dispatch_done, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward dispatch done");

  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(compute_stream, slot.backward_start), identity.rank,
      identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "start Backward compute");
  constexpr std::uint32_t kPackChunksPerExpert = 4U;
  if (view.sizes.owned_expert_count != 0U) {
    backward_pack_cached_state_kernel<<<
        view.sizes.owned_expert_count * kPackChunksPerExpert, 256, 0,
        compute_stream>>>(args, kPackChunksPerExpert);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "pack cached X/A/B into padded Backward rows");
  }
  if (view.sizes.incoming_row_count != 0ULL) {
    backward_wait_for_dispatch_rows_kernel<<<
        static_cast<unsigned int>(view.sizes.incoming_row_count), 32, 0,
        compute_stream>>>(args, view.sizes.incoming_row_count);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "wait for fixed-slot dR rows launch");
  }

  const auto hidden = static_cast<int>(config.hidden_size);
  const auto intermediate = static_cast<int>(config.intermediate_size);
  auto *dr = at<transport::BFloat16>(slot.device, layout.backward_padded_dr);
  auto *du = at<transport::BFloat16>(slot.device, layout.backward_dx_rows);
  auto *u = at<transport::BFloat16>(slot.device, layout.backward_u);
  auto *dab = at<transport::BFloat16>(slot.device, layout.backward_dab_rows);
  auto *dx = at<transport::BFloat16>(slot.device, layout.backward_dx_rows);
  auto *x = at<const transport::BFloat16>(slot.device,
                                          layout.backward_padded_x);
  auto *dw1 = at<transport::BFloat16>(slot.device, layout.backward_dw1);
  auto *dw2 = at<transport::BFloat16>(slot.device, layout.backward_dw2);
  const auto *w1 = retained.weights.impl_->w1;
  const auto *w2 = retained.weights.impl_->w2;
  std::uint64_t gemm_launches = 0ULL;
  {
    std::lock_guard<std::mutex> blas_lock(pool->backward_blas_mutex);
    throw_cublas("backward-run",
                 cublasSetStream(pool->backward_blas, compute_stream),
                 identity.rank, "cublasSetStream(Backward)");
    for (std::uint32_t expert = 0;
         expert < view.sizes.owned_expert_count; ++expert) {
      const auto rows = retained.backward_padded_expert_rows[expert];
      const auto row = retained.backward_padded_expert_offsets[expert];
      if (rows == 0ULL)
        continue;
      launch_backward_gemm(
          pool->backward_blas, CUBLAS_OP_N, CUBLAS_OP_N, intermediate,
          static_cast<int>(rows), hidden, w2 + expert * config.hidden_size *
                                                   config.intermediate_size,
          intermediate, dr + row * config.hidden_size, hidden,
          du + row * config.intermediate_size, intermediate, 0.0F,
          identity.rank, "dU = dR * W2");
      ++gemm_launches;
    }
    if (retained.backward_padded_rows != 0ULL) {
      backward_swiglu_epilogue_kernel<<<
          static_cast<unsigned int>(retained.backward_padded_rows), 256, 0,
          compute_stream>>>(args, retained.backward_padded_rows);
      internal::throw_cuda(
          "backward-run", cudaGetLastError(), identity.rank, identity.world,
          "backward", identity.generation, ExchangeKind::ExpertCounts,
          "Backward SwiGLU epilogue launch");
    }
    internal::throw_cuda(
        "backward-run", cudaEventRecord(slot.backward_dab_done, compute_stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "record dAB done");
    for (std::uint32_t expert = 0;
         expert < view.sizes.owned_expert_count; ++expert) {
      const auto rows = retained.backward_padded_expert_rows[expert];
      const auto row = retained.backward_padded_expert_offsets[expert];
      if (rows == 0ULL)
        continue;
      launch_backward_gemm(
          pool->backward_blas, CUBLAS_OP_N, CUBLAS_OP_N, hidden,
          static_cast<int>(rows), 2 * intermediate,
          w1 + expert * 2ULL * config.intermediate_size * config.hidden_size,
          hidden, dab + row * 2ULL * config.intermediate_size,
          2 * intermediate, dx + row * config.hidden_size, hidden, 0.0F,
          identity.rank, "dX = dAB * W1");
      ++gemm_launches;
    }
    internal::throw_cuda(
        "backward-run", cudaEventRecord(slot.backward_dx_done, compute_stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "record dX done");
    for (std::uint32_t expert = 0;
         expert < view.sizes.owned_expert_count; ++expert) {
      const auto rows = retained.backward_padded_expert_rows[expert];
      const auto row = retained.backward_padded_expert_offsets[expert];
      auto *expert_dw2 =
          dw2 + expert * config.hidden_size * config.intermediate_size;
      if (rows == 0ULL) {
        internal::throw_cuda(
            "backward-run",
            cudaMemsetAsync(expert_dw2, 0,
                            checked_size(config.hidden_size *
                                             config.intermediate_size *
                                             sizeof(transport::BFloat16),
                                         "empty expert dW2 bytes"),
                            compute_stream),
            identity.rank, identity.world, "backward", identity.generation,
            ExchangeKind::ExpertCounts, "zero empty expert dW2");
        continue;
      }
      launch_backward_gemm(
          pool->backward_blas, CUBLAS_OP_N, CUBLAS_OP_T, intermediate, hidden,
          static_cast<int>(rows), u + row * config.intermediate_size,
          intermediate, dr + row * config.hidden_size, hidden, expert_dw2,
          intermediate, 0.0F, identity.rank, "dW2 = dR^T * U");
      ++gemm_launches;
    }
    internal::throw_cuda(
        "backward-run", cudaEventRecord(slot.backward_dw2_done, compute_stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "record dW2 done");
    for (std::uint32_t expert = 0;
         expert < view.sizes.owned_expert_count; ++expert) {
      const auto rows = retained.backward_padded_expert_rows[expert];
      const auto row = retained.backward_padded_expert_offsets[expert];
      auto *expert_dw1 = dw1 + expert * 2ULL * config.intermediate_size *
                                   config.hidden_size;
      if (rows == 0ULL) {
        internal::throw_cuda(
            "backward-run",
            cudaMemsetAsync(expert_dw1, 0,
                            checked_size(2ULL * config.intermediate_size *
                                             config.hidden_size *
                                             sizeof(transport::BFloat16),
                                         "empty expert dW1 bytes"),
                            compute_stream),
            identity.rank, identity.world, "backward", identity.generation,
            ExchangeKind::ExpertCounts, "zero empty expert dW1");
        continue;
      }
      launch_backward_gemm(
          pool->backward_blas, CUBLAS_OP_N, CUBLAS_OP_T, hidden,
          2 * intermediate, static_cast<int>(rows),
          x + row * config.hidden_size, hidden,
          dab + row * 2ULL * config.intermediate_size, 2 * intermediate,
          expert_dw1, hidden, 0.0F, identity.rank,
          "dW1 = dAB^T * X");
      ++gemm_launches;
    }
    internal::throw_cuda(
        "backward-run", cudaEventRecord(slot.backward_dw1_done, compute_stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "record dW1 done");
  }

  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(transport_stream, slot.backward_dx_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait dX for local return");
  if (view.sizes.incoming_row_count != 0ULL) {
    backward_return_local_kernel<<<
        static_cast<unsigned int>(view.sizes.incoming_row_count), 256, 0,
        transport_stream>>>(args, view.sizes.incoming_row_count);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "Backward local return launch");
  }
  if (identity.tokens != 0ULL) {
    backward_reduce_dx_kernel<<<static_cast<unsigned int>(identity.tokens),
                                256, 0, transport_stream>>>(args);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "Backward dX reduce launch");
  }
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_return_done, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward return done");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.backward_return_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "join Backward return");
  internal::throw_cuda(
      "backward-run", cudaStreamWaitEvent(exec->stream, slot.backward_dw1_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "join Backward dW");
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.backward_complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward complete");
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record pool slot complete");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "backward-run", cudaEventSynchronize(slot.backward_complete),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait Backward completion");
  consumer.finish();

  auto result = std::make_shared<DeviceBackwardHandle::Impl>();
  result->slot = retained.slot;
  result->pool = pool;
  result->forward = forward;
  result->output_gradients = output_gradients;
  result->identity = identity;
  result->owned_experts = view.sizes.owned_expert_count;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->audit.specialization =
      BackwardKernelSpecialization::Sm90CachedXAb7168x3072;
  result->audit.gemm_launches = gemm_launches;
  result->audit.actual_dispatch_peer_bytes =
      retained.audit.actual_dispatch_peer_bytes;
  const auto remote_return_rows =
      retained.audit.actual_return_peer_bytes /
      (config.hidden_size * sizeof(transport::BFloat16));
  result->audit.actual_return_peer_bytes =
      retained.audit.actual_return_peer_bytes + remote_return_rows * sizeof(float);
  result->audit.workspace_bytes = pool->layout.total;
  result->audit.hot_path_allocations = 0;
  result->timings.dispatch_us = backward_elapsed_us(
      slot.backward_start, slot.backward_dispatch_done, identity,
      "Backward dispatch elapsed");
  result->timings.dab_us = backward_elapsed_us(
      slot.backward_dispatch_done, slot.backward_dab_done, identity,
      "Backward dAB elapsed");
  result->timings.dx_us = backward_elapsed_us(
      slot.backward_dab_done, slot.backward_dx_done, identity,
      "Backward dX elapsed");
  result->timings.dw2_us = backward_elapsed_us(
      slot.backward_dx_done, slot.backward_dw2_done, identity,
      "Backward dW2 elapsed");
  result->timings.dw1_us = backward_elapsed_us(
      slot.backward_dw2_done, slot.backward_dw1_done, identity,
      "Backward dW1 elapsed");
  result->timings.combine_us = backward_elapsed_us(
      slot.backward_dx_done, slot.backward_return_done, identity,
      "Backward combine elapsed");
  const auto dw_after_dx = backward_elapsed_us(
      slot.backward_dx_done, slot.backward_dw1_done, identity,
      "Backward dW span");
  result->timings.dw_combine_overlap_us =
      std::min(dw_after_dx, result->timings.combine_us);
  result->timings.total_us = backward_elapsed_us(
      slot.backward_start, slot.backward_complete, identity,
      "Backward total elapsed");
  failure_guard.committed = true;
  return DeviceBackwardHandle(std::move(result));
}

const ActivationTransportExpectation &DeviceBackwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}

const BackwardMegaKernelAudit &DeviceBackwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}

const BackwardMegaKernelTimings &DeviceBackwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

BackwardValidationDownload BackwardValidationBridge::download(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  if (!handle.impl_ || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  internal::throw_cuda(
      "backward-download", cudaEventSynchronize(
                               impl.pool->slots[impl.slot->index].backward_complete),
      impl.identity.rank, impl.identity.world, "backward",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait Backward output");
  const auto &layout = impl.pool->layout;
  const auto &slot = impl.pool->slots[impl.slot->index];
  BackwardValidationDownload result;
  result.dx.resize(checked_size(
      checked_mul(impl.identity.tokens, impl.identity.hidden,
                  "Backward dX download elements"),
      "Backward dX download elements"));
  result.dgate.resize(checked_size(
      checked_mul(impl.identity.tokens,
                  impl.forward.impl_->dispatch.identity().top_k,
                  "Backward dGate download elements"),
      "Backward dGate download elements"));
  result.dw1.resize(checked_size(
      checked_mul(checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                              "Backward dW1 download rows"),
                  impl.identity.hidden, "Backward dW1 download elements"),
      "Backward dW1 download elements"));
  result.dw2.resize(checked_size(
      checked_mul(checked_mul(impl.owned_experts, impl.identity.hidden,
                              "Backward dW2 download rows"),
                  impl.intermediate, "Backward dW2 download elements"),
      "Backward dW2 download elements"));
  const auto copy = [&](void *host, const void *device, std::uint64_t bytes,
                        const char *name) {
    if (bytes == 0ULL)
      return;
    internal::throw_cuda(
        "backward-download",
        cudaMemcpy(host, device, checked_size(bytes, name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "backward",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.dx.data(), at<const void>(slot.device, layout.combined),
       result.dx.size() * sizeof(transport::BFloat16), "download dX");
  copy(result.dgate.data(),
       at<const void>(slot.device, layout.backward_dgate_output),
       result.dgate.size() * sizeof(float), "download dGate");
  copy(result.dw1.data(), at<const void>(slot.device, layout.backward_dw1),
       result.dw1.size() * sizeof(transport::BFloat16), "download dW1");
  copy(result.dw2.data(), at<const void>(slot.device, layout.backward_dw2),
       result.dw2.size() * sizeof(transport::BFloat16), "download dW2");
  return result;
}
