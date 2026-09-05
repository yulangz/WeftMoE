std::shared_ptr<LocalP2pBackendState> create_local_p2p_backend(
    const NcclSession &session, const CudaExecution &execution,
    const std::shared_ptr<ActivationPoolState> &pool) {
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create",
                    "session/execution/pool", 0, "valid", "empty");
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (session.session().stable_identity_bytes() != pool->session_identity ||
      session.session().world_size() != pool->world ||
      session.device_ordinal() != pool->device ||
      execution.device_ordinal() != pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  int device_count = 0;
  internal::throw_cuda(
      "local-p2p-create", cudaGetDeviceCount(&device_count), session.rank(),
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "cudaGetDeviceCount");
  if (nccl->world > static_cast<std::uint32_t>(device_count))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-create, field=local_world, rank=" +
            std::to_string(session.rank()) + ", expected<=device_count=" +
            std::to_string(device_count) + ", actual=" +
            std::to_string(nccl->world));

  const auto slot_count = pool->slots.size();
  const auto handles_bytes = checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                                         "IPC handle bytes");
  const auto record_bytes = checked_size(
      kLocalP2pBootstrapHeaderBytes + handles_bytes, "IPC record bytes");
  const auto matrix_bytes = checked_size(
      checked_mul(record_bytes, nccl->world, "IPC matrix bytes"),
      "IPC matrix bytes");
  std::vector<std::byte> local(record_bytes);
  put_u64_le(local, 0, kLocalP2pBootstrapMagic);
  put_u64_le(local, 8, kLocalP2pBootstrapSchema);
  put_u64_le(local, 16, record_bytes);
  put_u64_le(local, 24, session.rank());
  put_u64_le(local, 32, nccl->world);
  put_u64_le(local, 40, pool->device);
  put_u64_le(local, 48, slot_count);
  put_u64_le(local, 56, pool->layout.total);
  put_u64_le(local, 64, session.session().generation());
  const auto identity_hash =
      session_summary(session.session().stable_identity_bytes());
  put_u64_le(local, 72, identity_hash[0]);
  put_u64_le(local, 80, identity_hash[1]);
  put_u64_le(local, 88, 0);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    cudaIpcMemHandle_t handle{};
    internal::throw_cuda(
        "local-p2p-create",
        cudaIpcGetMemHandle(&handle, pool->slots[slot].device), session.rank(),
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaIpcGetMemHandle(pool slot)");
    std::memcpy(local.data() + kLocalP2pBootstrapHeaderBytes +
                    slot * sizeof(handle),
                &handle, sizeof(handle));
  }

  void *device_send = nullptr;
  void *device_receive = nullptr;
  std::vector<std::byte> matrix(matrix_bytes);
  auto exec = internal::checked_execution_state(execution);
  try {
    internal::throw_cuda(
        "local-p2p-create", cudaMalloc(&device_send, record_bytes),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(IPC bootstrap send)");
    internal::throw_cuda(
        "local-p2p-create", cudaMalloc(&device_receive, matrix_bytes),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(IPC bootstrap receive)");
    internal::throw_cuda(
        "local-p2p-create",
        cudaMemcpyAsync(device_send, local.data(), record_bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(IPC bootstrap send)");
    internal::throw_nccl(
        "local-p2p-create",
        ncclAllGather(device_send, device_receive, record_bytes, ncclUint8,
                      nccl->comm, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllGather(IPC bootstrap)");
    internal::throw_cuda(
        "local-p2p-create",
        cudaMemcpyAsync(matrix.data(), device_receive, matrix_bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(IPC bootstrap receive)");
    internal::throw_cuda(
        "local-p2p-create", cudaStreamSynchronize(exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(IPC bootstrap)");
  } catch (...) {
    if (device_receive)
      (void)cudaFree(device_receive);
    if (device_send)
      (void)cudaFree(device_send);
    throw;
  }
  (void)cudaFree(device_receive);
  (void)cudaFree(device_send);

  auto backend = std::make_shared<LocalP2pBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  const auto mapping_count = checked_size(
      checked_mul(slot_count, nccl->world, "IPC mapping count"),
      "IPC mapping count");
  backend->peer_slot_bases.assign(mapping_count, nullptr);
  backend->owns_mapping.assign(mapping_count, 0);
  try {
    std::vector<std::uint8_t> seen_devices(
        static_cast<std::size_t>(device_count), 0);
    for (std::uint32_t rank = 0; rank < nccl->world; ++rank) {
      const auto *record = matrix.data() +
                           static_cast<std::size_t>(rank) * record_bytes;
      const auto check = [&](std::size_t offset, std::uint64_t expected,
                             const char *field) {
        const auto actual = get_u64_le(record, record_bytes, offset,
                                       session.rank(), field);
        if (actual != expected)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "local-p2p-bootstrap", field, session.rank(),
                          std::to_string(expected), std::to_string(actual));
      };
      check(0, kLocalP2pBootstrapMagic, "magic");
      check(8, kLocalP2pBootstrapSchema, "schema");
      check(16, record_bytes, "record_bytes");
      check(24, rank, "rank");
      check(32, nccl->world, "world");
      check(48, slot_count, "slot_count");
      check(56, pool->layout.total, "arena_bytes");
      check(64, session.session().generation(), "generation");
      check(72, identity_hash[0], "session_digest0");
      check(80, identity_hash[1], "session_digest1");
      check(88, 0, "reserved");
      const auto peer_device = get_u64_le(record, record_bytes, 40,
                                          session.rank(), "device");
      if (peer_device >= static_cast<std::uint64_t>(device_count) ||
          seen_devices[static_cast<std::size_t>(peer_device)] != 0)
        transport_error(NotifyErrorCode::TransportProtocolFailure,
                        "local-p2p-bootstrap", "device", session.rank(),
                        "unique local device ordinal",
                        std::to_string(peer_device));
      seen_devices[static_cast<std::size_t>(peer_device)] = 1;
      int access = 0;
      if (peer_device != pool->device) {
        internal::throw_cuda(
            "local-p2p-create",
            cudaDeviceCanAccessPeer(&access, static_cast<int>(pool->device),
                                    static_cast<int>(peer_device)),
            session.rank(), nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaDeviceCanAccessPeer");
        if (access == 0)
          throw NotifyDispatchError(
              NotifyErrorCode::BackendUnavailable,
              "phase=local-p2p-create, field=peer_access, rank=" +
                  std::to_string(session.rank()) + ", expected=1, actual=0, "
                  "peer_rank=" + std::to_string(rank));
      }
      for (std::size_t slot = 0; slot < slot_count; ++slot) {
        const auto index = slot * nccl->world + rank;
        if (rank == session.rank()) {
          backend->peer_slot_bases[index] = pool->slots[slot].device;
          continue;
        }
        cudaIpcMemHandle_t handle{};
        std::memcpy(&handle,
                    record + kLocalP2pBootstrapHeaderBytes +
                        slot * sizeof(handle),
                    sizeof(handle));
        internal::throw_cuda(
            "local-p2p-create",
            cudaIpcOpenMemHandle(&backend->peer_slot_bases[index], handle,
                                 cudaIpcMemLazyEnablePeerAccess),
            session.rank(), nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaIpcOpenMemHandle(peer slot)");
        backend->owns_mapping[index] = 1;
      }
    }
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      internal::throw_cuda(
          "local-p2p-create",
          cudaMemcpyAsync(
              at<void *>(pool->slots[slot].device, pool->layout.peer_bases),
              backend->peer_slot_bases.data() + slot * nccl->world,
              nccl->world * sizeof(void *), cudaMemcpyHostToDevice,
              exec->stream),
          session.rank(), nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "cudaMemcpyAsync(peer base table)");
    }
    internal::throw_cuda(
        "local-p2p-create", cudaStreamSynchronize(exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(peer table)");
  } catch (...) {
    backend.reset();
    throw;
  }
  return backend;
}

std::shared_ptr<GinBackendState> create_gin_backend(
    const NcclSession &session, const CudaExecution &execution,
    const std::shared_ptr<ActivationPoolState> &pool) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)pool;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-create, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                    "session/execution/pool", 0, "valid", "empty");
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (!pool->config.enable_gin_rdma)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-create",
                    "pool.enable_gin_rdma", session.rank(), "true", "false");
  if (session.session().stable_identity_bytes() != pool->session_identity ||
      session.session().world_size() != pool->world ||
      session.device_ordinal() != pool->device ||
      execution.device_ordinal() != pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    if (auto cached_base = pool->gin_backend.lock()) {
      auto cached = std::dynamic_pointer_cast<GinBackendState>(cached_base);
      if (!cached || cached->nccl.get() != nccl.get() ||
          !cached->execution.token().same_instance(execution.token()))
        transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                        "cached backend provenance", session.rank(),
                        "same communicator/execution", "mismatch");
      return cached;
    }
  }

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [&](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-create, field=symbol, rank=") +
              std::to_string(session.rank()) + ", expected=" + name +
              ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  internal::throw_nccl(
      "gin-create", query_properties(nccl->comm, &properties), nccl->rank,
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "ncclCommQueryProperties");
  if (properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=railed_gin, rank=" +
            std::to_string(session.rank()) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (pool->config.max_nodes == 0 ||
      nccl->world % pool->config.max_nodes != 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-create",
                    "homogeneous topology", nccl->rank,
                    "world divisible by runtime nodes", "mismatch");
  const auto num_nodes = pool->config.max_nodes;
  const auto local_world = nccl->world / num_nodes;
  const auto node = nccl->rank / local_world;
  const auto local_rank = nccl->rank % local_world;
  if (rail.nRanks != static_cast<int>(num_nodes) ||
      lsa.nRanks != static_cast<int>(local_world) ||
      pool->device != local_rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=topology, rank=" +
            std::to_string(session.rank()) +
            ", expected=rail=nodes, lsa=local_world, device=local_rank, "
            "actual=world=" + std::to_string(nccl->world) + ", rail=" +
            std::to_string(rail.nRanks) + ", lsa=" +
            std::to_string(lsa.nRanks) + ", device=" +
            std::to_string(pool->device) + ", local_rank=" +
            std::to_string(local_rank));

  auto backend = std::make_shared<GinBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  backend->num_nodes = num_nodes;
  backend->local_world_size = local_world;
  backend->node = node;
  backend->local_rank = local_rank;

  // Exchange every compute-arena IPC handle collectively, but open only the
  // ranks in this node. The full-world pointer table keeps global rank as the
  // stable index consumed by Notify fan-out descriptors.
  const auto slot_count = pool->slots.size();
  const auto handles_per_rank_bytes = checked_size(
      checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                  "GIN IPC handles per rank"),
      "GIN IPC handles per rank");
  const auto matrix_bytes = checked_size(
      checked_mul(handles_per_rank_bytes, nccl->world,
                  "GIN IPC handle matrix"),
      "GIN IPC handle matrix");
  std::vector<cudaIpcMemHandle_t> local_handles(slot_count);
  std::vector<cudaIpcMemHandle_t> handle_matrix(slot_count * nccl->world);
  for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index)
    internal::throw_cuda(
        "gin-create",
        cudaIpcGetMemHandle(&local_handles[slot_index],
                            pool->slots[slot_index].device),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaIpcGetMemHandle(GIN local compute arena)");
  void *device_handles = nullptr;
  void *device_matrix = nullptr;
  try {
    internal::throw_cuda(
        "gin-create", cudaMalloc(&device_handles, handles_per_rank_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN IPC handles)");
    internal::throw_cuda(
        "gin-create", cudaMalloc(&device_matrix, matrix_bytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN IPC matrix)");
    const auto exec = internal::checked_execution_state(execution);
    internal::throw_cuda(
        "gin-create",
        cudaMemcpyAsync(device_handles, local_handles.data(),
                        handles_per_rank_bytes, cudaMemcpyHostToDevice,
                        exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "H2D GIN IPC handles");
    internal::throw_nccl(
        "gin-create",
        ncclAllGather(device_handles, device_matrix, handles_per_rank_bytes,
                      ncclUint8, nccl->comm, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllGather(GIN IPC handles)");
    internal::throw_cuda(
        "gin-create",
        cudaMemcpyAsync(handle_matrix.data(), device_matrix, matrix_bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "D2H GIN IPC matrix");
    internal::throw_cuda(
        "gin-create", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "wait GIN IPC bootstrap");
  } catch (...) {
    if (device_matrix != nullptr)
      (void)cudaFree(device_matrix);
    if (device_handles != nullptr)
      (void)cudaFree(device_handles);
    throw;
  }
  (void)cudaFree(device_matrix);
  (void)cudaFree(device_handles);
  backend->peer_slot_bases.assign(slot_count * nccl->world, nullptr);
  backend->owns_mapping.assign(slot_count * nccl->world, 0);
  try {
    for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
      for (std::uint32_t peer = node * local_world;
           peer < (node + 1U) * local_world; ++peer) {
        const auto index = slot_index * nccl->world + peer;
        if (peer == nccl->rank) {
          backend->peer_slot_bases[index] = pool->slots[slot_index].device;
        } else {
          internal::throw_cuda(
              "gin-create",
              cudaIpcOpenMemHandle(
                  &backend->peer_slot_bases[index],
                  handle_matrix[static_cast<std::size_t>(peer) * slot_count +
                                slot_index],
                  cudaIpcMemLazyEnablePeerAccess),
              nccl->rank, nccl->world, session.session().identity(),
              session.session().generation(), ExchangeKind::ExpertCounts,
              "cudaIpcOpenMemHandle(GIN local peer arena)");
          backend->owns_mapping[index] = 1;
        }
      }
      const auto exec = internal::checked_execution_state(execution);
      internal::throw_cuda(
          "gin-create",
          cudaMemcpyAsync(
              at<void *>(pool->slots[slot_index].device,
                         pool->layout.peer_bases),
              backend->peer_slot_bases.data() + slot_index * nccl->world,
              nccl->world * sizeof(void *), cudaMemcpyHostToDevice,
              exec->stream),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "H2D GIN local peer table");
    }
    internal::throw_cuda(
        "gin-create",
        cudaStreamSynchronize(
            internal::checked_execution_state(execution)->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "wait GIN local peer table");
  } catch (...) {
    backend.reset();
    throw;
  }
  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  backend->full_connections = true;
  constexpr std::uint32_t kGinContextCount = 8;
  // FULL Return must not share an exclusive context with an outstanding
  // same-rail
  // Dispatch queue.  Keep the logical topology at eight contexts/CTAs, but
  // give every FULL owner a second physical QP: [0, 8) is Dispatch and
  // [8, 16) is Return.  DeepEP routinely allocates more than this, and the
  // separation lets a single permanent RDMA CTA progress both queues without
  // mixing teams on one context.
  const auto allocated_context_count = 2U * kGinContextCount;
  requirements.ginContextCount = allocated_context_count;
  // One permanent RDMA CTA owns a same-rail Dispatch QP and an independent
  // FULL Return QP. Compute CTAs publish byte-range descriptors and never
  // touch a GIN context.
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 64;
  // 0=dispatch arrival, 1=return arrival.
  // Signal 0/1 are the data-plane dispatch/return epochs. Signal 2 is the
  // fixed terminal production-status epoch and is touched only once per
  // generation, never on the per-task success path.
  requirements.ginSignalCount = 3;
  // Counter 0 belongs to Dispatch; direct/stream Return overlaps it and uses
  // counter 1.  Request both explicitly (the previous count of one made every
  // counter-1 access formally out of bounds).
  requirements.ginCounterCount = 2;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  internal::throw_nccl(
      "gin-create",
      dev_comm_create(nccl->comm, &requirements, &backend->dev_comm),
      nccl->rank, nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "ncclDevCommCreate(activation GIN)");
  backend->dev_comm_created = true;
  backend->gin_context_count = kGinContextCount;
  backend->gin_allocated_context_count = allocated_context_count;
  try {
    backend->windows.reserve(pool->slots.size());
    for (auto &slot : pool->slots) {
      ncclWindow_t window = nullptr;
      if (slot.gin_window == nullptr || pool->gin_layout.total == 0)
        transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                        "registered window storage", nccl->rank,
                        "preallocated ncclMemAlloc window", "empty");
      internal::throw_nccl(
          "gin-create",
          window_register(nccl->comm, slot.gin_window,
                          pool->gin_layout.total, &window,
                          NCCL_WIN_STRICT_ORDERING),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "ncclCommWindowRegister(activation pool slot)");
      backend->windows.push_back(window);
    }
  } catch (...) {
    backend.reset();
    throw;
  }
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    pool->gin_backend = backend;
  }
  return backend;
#endif
}

