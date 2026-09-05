      // Reduction ownership is distributed across the resident compute CTAs;
      // transport CTAs remain dedicated throughout the generation.
      args.forward_reduce_cta_count =
          (forward_main_compute_ctas + forward_tuning::kReduceCtaDivisor -
           1U) /
          forward_tuning::kReduceCtaDivisor;
      internal::throw_cuda(
          "forward-run",
          split_local_roles
              ? forward_transport_launch::set_local_max_dynamic_shared_memory(
                    static_cast<int>(forward_transport_smem))
              : forward_transport_launch::set_max_dynamic_shared_memory(
                    static_cast<int>(forward_transport_smem)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(split transport dynamic smem)");
      const auto set_tma_compute_attribute =
          [&](auto inference_kernel, auto training_kernel) {
            return cudaFuncSetAttribute(
                config.retain_backward_state ? training_kernel
                                             : inference_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(forward_compute_smem));
          };
      cudaError_t compute_attribute = cudaSuccess;
#if MEGAMOE_FWD_FC2_TMA_STORE
#if MEGAMOE_FWD_CLUSTER_SIZE == 2
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
      else
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
      else
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
#endif
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<
                true, MEGAMOE_FWD_TILE_M == 64,
                ForwardTaskPartition::All, true, false, false,
                forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
      else
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<
                false, MEGAMOE_FWD_TILE_M == 64,
                ForwardTaskPartition::All, true, false, false,
                forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
#endif
      internal::throw_cuda(
          "forward-run", compute_attribute,
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(queued FC1/FC2 dynamic smem)");

      internal::throw_cuda(
          "forward-run",
          cudaEventRecord(slot.forward_roles_ready, exec->stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaEventRecord(split Forward roles ready)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(slot.forward_transport_stream,
                              slot.forward_roles_ready),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward transport ready)");
      internal::throw_cuda("forward-run",
                           cudaStreamWaitEvent(slot.forward_compute_stream,
                                               slot.forward_roles_ready),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaStreamWaitEvent(split Forward compute ready)");

      cudaLaunchConfig_t transport_launch{};
      transport_launch.gridDim = dim3(kTransportBlocks, 1, 1);
      transport_launch.blockDim =
          dim3(forward_tuning::kComputeThreads, 1, 1);
      transport_launch.dynamicSmemBytes = forward_transport_smem;
      transport_launch.stream = slot.forward_transport_stream;
      launch_forward =
          split_local_roles
              ? forward_transport_launch::launch_local(
                    &transport_launch, &args, sizeof(args), slot.forward_x_map,
                    slot.forward_s_map, weight.w1_map, weight.w2_map)
              : forward_transport_launch::launch(
                    &transport_launch, &args, sizeof(args), slot.forward_x_map,
                    slot.forward_s_map, weight.w1_map, weight.w2_map);
      internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           split_local_roles
                               ? "persistent 0-RDMA/tunable-Relay Forward launch"
                               : "persistent 8-RDMA/tunable-Relay Forward launch");
      internal::throw_cuda("forward-run",
                           cudaEventRecord(slot.forward_transport_done,
                                           slot.forward_transport_stream),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventRecord(split Forward transport done)");
      auto compute_args = args;
#if MEGAMOE_DEVICE_TIMELINE
      compute_args.timeline.cta_base = kTransportBlocks;
#endif
      cudaLaunchConfig_t compute_launch{};
      compute_launch.gridDim = dim3(forward_main_compute_ctas, 1, 1);
      compute_launch.blockDim =
          dim3(forward_tuning::kComputeThreads, 1, 1);
      compute_launch.dynamicSmemBytes = forward_compute_smem;
      compute_launch.stream = slot.forward_compute_stream;
      cudaLaunchAttribute compute_cluster_attribute{};
      if (forward_cluster2) {
        compute_cluster_attribute.id = cudaLaunchAttributeClusterDimension;
        compute_cluster_attribute.val.clusterDim = {
            forward_tuning::kClusterSize, 1U, 1U};
        compute_launch.attrs = &compute_cluster_attribute;
        compute_launch.numAttrs = 1U;
      }
      const auto launch_tma_compute =
          [&](auto inference_kernel, auto training_kernel,
              const CUtensorMap &output_store_map) {
            const auto kernel = config.retain_backward_state
                                    ? training_kernel
                                    : inference_kernel;
            return cudaLaunchKernelEx(
                &compute_launch, kernel, compute_args, slot.forward_x_map,
                slot.forward_s_map, weight.w1_map, weight.w2_map,
                output_store_map);
          };
#if MEGAMOE_FWD_FC2_TMA_STORE
      const auto &output_store_map =
          args.use_gin && args.forward_registered_output
              ? slot.forward_staging_store_map
              : ordinary_output_store_map;
#if MEGAMOE_FWD_CLUSTER_SIZE == 2
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
      else
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
      else
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
#endif
#elif MEGAMOE_FWD_TILE_M == 64
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                true, true, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_m64_map, slot.forward_s_m64_map,
            weight.w1_map, weight.w2_map);
      else
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                false, true, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_m64_map, slot.forward_s_m64_map,
            weight.w1_map, weight.w2_map);
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                true, false, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
            weight.w2_map);
      else
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                false, false, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
            weight.w2_map);
#endif
      internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           "persistent queued FC1/FC2 WGMMA/TMA Forward launch");

      internal::throw_cuda("forward-run",
                           cudaEventRecord(slot.forward_compute_done,
                                           slot.forward_compute_stream),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventRecord(split Forward compute done)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(exec->stream, slot.forward_transport_done),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward transport done)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(exec->stream, slot.forward_compute_done),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward compute done)");
    } else {
      if (!remote_transport)
        transport_error(NotifyErrorCode::BackendUnavailable, "forward-run",
                        "single-node kernel", id.calling_rank,
                        "split transport/compute architecture", "disabled");
      internal::throw_cuda(
          "forward-run",
          cudaFuncSetAttribute(local_p2p_forward_megakernel<false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(forward_smem)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(dynamic smem)");
      cudaLaunchAttribute attributes[1]{};
      attributes[0].id = cudaLaunchAttributeCooperative;
      attributes[0].val.cooperative = 1;
      cudaLaunchConfig_t launch{};
      launch.gridDim = dim3(grid_blocks, 1, 1);
      launch.blockDim = dim3(384, 1, 1);
      launch.dynamicSmemBytes = forward_smem;
      launch.stream = exec->stream;
      launch.attrs = attributes;
      launch.numAttrs = 1;
      internal::throw_cuda(
          "forward-run",
          cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                                 static_cast<int>(exec->device)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaDevAttrClusterLaunch");
      const auto *forward_kernel = reinterpret_cast<const void *>(
          local_p2p_forward_megakernel<false>);
      internal::throw_cuda("forward-run",
                           cudaOccupancyMaxActiveClusters(
                               &max_active_clusters, forward_kernel, &launch),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaOccupancyMaxActiveClusters");
      launch_forward = cudaLaunchKernelEx(
          &launch, local_p2p_forward_megakernel<false>, args,
          slot.forward_x_map, slot.forward_s_map, weight.w1_map,
          weight.w2_map);
    }
#if MEGAMOE_DEBUG_VALIDATE
    debug_forward_host_phase("kernel_launch_returned");
#endif
    internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                         id.world_size, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "single persistent WGMMA/TMA forward launch");
  if (use_socket) {
    std::vector<std::uint64_t> send_prefix(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    std::vector<std::uint64_t> receive_prefix(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      send_prefix[peer + 1U] =
          send_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                      id.world_size +
                                  peer];
      receive_prefix[peer + 1U] =
          receive_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(peer) *
                                      id.world_size +
                                  id.calling_rank];
    }
    if (send_prefix[id.world_size] != view.sizes.route_count ||
        receive_prefix[id.world_size] != socket_incoming)
      transport_error(NotifyErrorCode::TransportProtocolFailure,
                      "socket-forward-return", "route counts",
                      id.calling_rank, "exact preload counts", "mismatch");
    auto *send_headers = at<SocketRouteHeader>(
        slot.device, layout.socket_header_send);
    auto *receive_headers = at<SocketRouteHeader>(
        slot.device, layout.socket_header_receive);
    auto *send_payload = at<transport::BFloat16>(
        slot.device, layout.contribution);
    auto *receive_payload = at<transport::BFloat16>(
        slot.device, layout.socket_payload_receive);
    auto *forward_output = at<transport::BFloat16>(
        slot.device, args.forward_output_rows_offset);
    if (socket_incoming != 0)
      socket_gather_forward_returns_kernel<<<
          static_cast<unsigned int>(socket_incoming), 256, 0,
          exec->stream>>>(receive_headers, forward_output, receive_payload,
                          socket_incoming, config.hidden_size);
    internal::throw_cuda(
        "socket-forward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "gather canonical FC2 rows");
    internal::throw_nccl(
        "socket-forward-return", ncclGroupStart(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupStart(Socket FC return)");
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      const auto send_rows = receive_prefix[peer + 1U] - receive_prefix[peer];
      const auto receive_rows = send_prefix[peer + 1U] - send_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != receive_rows)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "socket-forward-return", "self rows",
                          id.calling_rank, std::to_string(send_rows),
                          std::to_string(receive_rows));
        if (send_rows != 0)
          internal::throw_cuda(
              "socket-forward-return",
              cudaMemcpyAsync(
                  send_payload + send_prefix[peer] * config.hidden_size,
                  receive_payload +
                      receive_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket Forward self return bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket Forward self return");
        continue;
      }
      socket_return_peer_bytes = checked_add(
          socket_return_peer_bytes,
          checked_mul(checked_mul(send_rows, config.hidden_size,
                                  "Socket Forward remote return elements"),
                      sizeof(transport::BFloat16),
                      "Socket Forward remote return bytes"),
          "Socket Forward total remote return bytes");
      if (send_rows != 0)
        internal::throw_nccl(
            "socket-forward-return",
            ncclSend(receive_payload +
                         receive_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket Forward return)");
      if (receive_rows != 0)
        internal::throw_nccl(
            "socket-forward-return",
            ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket Forward return)");
    }
    internal::throw_nccl(
        "socket-forward-return", ncclGroupEnd(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupEnd(Socket Forward return)");
    if (view.sizes.route_count != 0)
      socket_scatter_returns_kernel<<<
          static_cast<unsigned int>(view.sizes.route_count), 256, 0,
          exec->stream>>>(send_headers, send_payload,
                          at<transport::BFloat16>(slot.device,
                                                  layout.combine_return),
                          view.sizes.route_count, config.hidden_size);
    if (id.num_tokens != 0)
      socket_sum_forward_routes_kernel<<<
          static_cast<unsigned int>(id.num_tokens), 256, 0, exec->stream>>>(
          view,
          at<transport::BFloat16>(slot.device, layout.combine_return),
          at<transport::BFloat16>(slot.device, layout.combined),
          config.hidden_size);
    internal::throw_cuda(
        "socket-forward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "Socket scatter/unweighted reduce");
  }
  internal::throw_cuda(
      "forward-run",
      cudaEventRecord(slot.forward_output_ready, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(Forward output ready)");
  // Local P2P relies on the peer-visible terminal epoch. GIN payload and
  // return consumers have already acquired their per-generation signals;
  // the next Notify/reset agreement is the registered-window reuse boundary.
  internal::throw_cuda(
      "forward-run",
      cudaMemcpyAsync(slot.host_forward_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H forward status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.forward_complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(forward complete)");
  slot.completion_recorded = true;

#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_enter");
#endif
  std::atomic<bool> forward_stream_done{false};
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "forward-run",
      cudaLaunchHostFunc(exec->stream, publish_stream_completion,
                         &forward_stream_done),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaLaunchHostFunc(Forward completion notification)");
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_done");
#endif

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
  unsigned int progress_dump_count = 0U;
#endif
  while (true) {
    if (forward_stream_done.load(std::memory_order_acquire))
      break;
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >= options.deadline_after;
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
    const auto stalled_for = std::chrono::steady_clock::now() - wait_start;
    const auto progress_dump_after = progress_dump_count == 0U
                                         ? std::chrono::seconds(3)
                                         : std::chrono::seconds(30);
    if (progress_dump_count < 2U && use_gin && args.progress != nullptr &&
        stalled_for >= progress_dump_after) {
      constexpr std::size_t forward_progress_base = 0;
      const auto progress = [&](std::uint32_t word) {
        return __atomic_load_n(slot.host_forward_progress +
                                   forward_progress_base + word,
                               __ATOMIC_ACQUIRE);
      };
      // Emit the mapped coarse milestones before attempting a device-memory
      // snapshot. If a platform cannot schedule the diagnostic D2H while the
      // persistent grid is resident, this line still identifies the stalled
      // phase instead of losing the entire diagnostic.
      std::cerr << "FORWARD_PROGRESS_COARSE rank=" << id.calling_rank
                << " generation=" << id.generation
                << " sample=" << progress_dump_count
                << " entered=" << progress(kForwardEntered)
                << " setup=" << progress(kForwardSetupComplete)
                << " dispatch_stored=" << progress(kForwardDispatchStored)
                << " dispatch_observed="
                << progress(kForwardDispatchObserved)
                << " fc1=" << progress(kForwardFc1Complete)
                << " fc2=" << progress(kForwardFc2Complete)
                << " return=" << progress(kForwardReturnObserved)
                << " reduce=" << progress(kForwardReduceComplete)
                << " terminal=" << progress(kForwardTerminal) << std::endl;
      // Keep production completion atomics in device memory even in debug
      // builds. Redirecting the per-task counters to mapped host memory
      // perturbs the Forward schedule by hundreds of milliseconds and can
      // hide the race being diagnosed. A pair of copy-engine snapshots on the
      // independent cancel stream remains observable while the persistent
      // kernel occupies every SM.
      auto *completion_snapshot = slot.host_forward_progress +
                                  forward_progress_base + kForwardProgressWords;
      internal::throw_cuda(
          "forward-progress-snapshot",
          cudaMemcpyAsync(completion_snapshot,
                          at<void>(slot.device, layout.forward_completion),
                          sizeof(transport_state::GenerationCompletionVector),
                          cudaMemcpyDeviceToHost, state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "D2H Forward generation completion snapshot");
      internal::throw_cuda(
          "forward-progress-snapshot",
          cudaMemcpyAsync(
              completion_snapshot + transport_state::kGenerationCompletionWords,
              at<void>(slot.device, layout.forward_context_completion),
              static_cast<std::size_t>(slot.generation_debug_contexts) *
                  sizeof(transport_state::ContextCompletionRecord),
              cudaMemcpyDeviceToHost, state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "D2H Forward context completion snapshot");
      internal::throw_cuda(
          "forward-progress-snapshot",
          cudaStreamSynchronize(state->cancel_stream), id.calling_rank,
          id.world_size, backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "drain Forward completion snapshot");
      std::cerr << "FORWARD_PROGRESS rank=" << id.calling_rank
                << " generation=" << id.generation
                << " sample=" << progress_dump_count;
      dump_generation_completion(std::cerr, completion_snapshot,
                                 slot.generation_debug_contexts,
                                 "forward_completion");
      std::cerr << std::endl;
      ++progress_dump_count;
    }
#endif
    if (!cancelled &&
        (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, args.epoch, __ATOMIC_RELEASE);
      internal::throw_cuda(
          "forward-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "H2D publish forward cancel");
      internal::throw_cuda(
          "forward-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "drain forward cancel stream");
      cancelled = true;
    }
    // Do not call CUDA event-query APIs while the resident GIN kernel is
    // active. On the remote host that can hold a driver-side progress lock and
    // form a liveness cycle with device GIN. The stream callback above is the
    // completion notification; this sleep only bounds host polling overhead.
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("stream_completion_observed");
#endif
  internal::throw_cuda(
      "forward-run", cudaEventQuery(slot.forward_complete), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventQuery(completed Forward event)");
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("event_verified");
#endif
  consumer.finish();
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_finished");
  debug_forward_host_phase("terminal_agreement_enter");
#endif
  const auto production_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Forward, id.generation,
      id.generation, id.calling_rank, id.world_size, *slot.host_forward_status,
      slot.forward_complete);
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("terminal_agreement_done");
#endif
  if (cancelled || slot.host_forward_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=forward-run, field=cancel/deadline, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=completed forward, actual=cancelled");
  if (production_agreement.has_error) {
    // A logical device-status failure reached the fixed terminal agreement;
    // all peer work is drained and this slot is recoverable. Backend/CUDA
    // failures still leave the guard uncommitted and therefore quarantine.
    failure_guard.committed = true;
    throw_production_status_error(production_agreement);
  }

  float total_ms = 0.0F;
  internal::throw_cuda(
      "forward-timing",
      cudaEventElapsedTime(&total_ms, slot.forward_start,
                           slot.forward_output_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(forward total)");
  auto result = std::make_shared<DeviceForwardHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->backend = backend;
  result->dispatch = dispatch;
  result->activations = activations;
  result->weights = weights;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->retained_backward_state = config.retain_backward_state;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->forward_return_chunk_rows = args.forward_return_chunk_rows;
  result->audit.specialization =
      ForwardKernelSpecialization::Sm90WgmmaTma7168x3072;
  result->audit.persistent_kernel_launches = 1;
  result->audit.wgmma_instructions =
      slot.host_forward_status->wgmma_instructions;
  result->audit.tma_loads = slot.host_forward_status->tma_loads;
  result->audit.tma_multicast_loads =
      slot.host_forward_status->tma_multicast_loads;
  result->audit.runtime_cluster_supported = cluster_supported != 0;
  result->audit.requested_cluster_size = 2;
  result->audit.runtime_cluster_size =
      slot.host_forward_status->runtime_cluster_size;
  result->audit.observed_cluster_rank_mask =
      slot.host_forward_status->cluster_rank_mask;
  result->audit.max_active_clusters =
      static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
  result->audit.fc1_tasks = slot.host_forward_status->fc1_tasks_done;
  result->audit.fc2_tasks = slot.host_forward_status->fc2_tasks_done;
  result->audit.actual_dispatch_peer_bytes =
      use_socket ? socket_dispatch_peer_bytes
                 : slot.host_forward_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      use_socket ? socket_return_peer_bytes
                 : slot.host_forward_status->actual_return_peer_bytes;
  result->audit.gin_dispatch_put_count =
      slot.host_forward_status->gin_dispatch_put_count;
  result->audit.raw_cross_node_route_bytes =
      use_socket ? socket_raw_cross_node_route_bytes
                 : slot.host_forward_status->raw_cross_node_payload_bytes;
  result->audit.gin_dispatch_payload_bytes =
      slot.host_forward_status->deduplicated_cross_node_payload_bytes;
  result->audit.gin_return_payload_bytes =
      slot.host_forward_status->gin_return_payload_bytes;
  result->audit.gin_return_put_count =
      slot.host_forward_status->gin_return_put_count;
  result->audit.gin_signal_count =
      slot.host_forward_status->gin_signal_count;
  result->audit.gin_metadata_bytes =
      slot.host_forward_status->gin_metadata_bytes;
  result->audit.workspace_bytes = state->layout.total;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.terminal_status_agreement_us =
      production_agreement.elapsed_us;
  const auto &status = *slot.host_forward_status;
  const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
    return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
  };
  result->timings.dispatch_us =
      elapsed(status.start_clock, status.dispatch_complete_clock);
  result->timings.fc1_swiglu_us =
      elapsed(status.fc1_start_clock, status.fc1_complete_clock);
  result->timings.fc2_us =
      elapsed(status.dispatch_blocks_done, status.fc2_complete_clock);
  result->timings.combine_return_us =
      elapsed(status.relay_blocks_done, status.return_complete_clock);
  result->timings.reduce_us =
      elapsed(status.reduced_tokens, status.reduce_blocks_done);
  result->timings.fc1_fc2_overlap_us =
      elapsed(status.dispatch_blocks_done, status.fc1_complete_clock);
  result->timings.fc2_return_overlap_us =
      elapsed(status.relay_blocks_done, status.fc2_complete_clock);
  result->timings.return_reduce_overlap_us =
      elapsed(status.reduced_tokens, status.return_complete_clock);
  failure_guard.committed = true;
  return DeviceForwardHandle(std::move(result));
}
