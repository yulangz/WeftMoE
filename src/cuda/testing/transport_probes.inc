testing::GinPutSignalProbeResult
testing::run_distributed_gin_put_signal_probe(
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t payload_bytes) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)payload_bytes;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-put-probe, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-put-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(exec->device));
  constexpr std::size_t kAllocationBytes = 4096;
  constexpr std::size_t kReceiveOffset = 2048;
  constexpr std::size_t kBarrierOffset = 3072;
  constexpr std::size_t kErrorOffset = 4088;
  if (payload_bytes == 0 || payload_bytes > kReceiveOffset)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-put-probe",
                    "payload_bytes", nccl->rank, "1..2048",
                    std::to_string(payload_bytes));

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                             const ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-put-probe, field=symbol, rank=0, expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
      load("ncclDevCommDestroy"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
      load("ncclCommWindowDeregister"));
  const auto mem_alloc =
      reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto result = query_properties(nccl->comm, &properties);
  if (result != ncclSuccess || properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-put-probe, field=railed_gin, rank=" +
            std::to_string(nccl->rank) +
            ", expected=available, actual=" +
            (result == ncclSuccess
                 ? std::string("NCCL_GIN_TYPE_NONE")
                 : std::string(ncclGetErrorString(result))));
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0)
    transport_error(NotifyErrorCode::BackendUnavailable, "gin-put-probe",
                    "rail topology", nccl->rank, "rail.nRanks>=2, lsa>0",
                    "rail=" + std::to_string(rail.nRanks) + ", lsa=" +
                        std::to_string(lsa.nRanks));

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = 1;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 32;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };
  try {
    internal::throw_nccl(
        "gin-put-probe",
        dev_comm_create(nccl->comm, &requirements, &dev_comm), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclDevCommCreate(rail GIN)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-put-probe", mem_alloc(&allocation, kAllocationBytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN probe window)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemsetAsync(allocation, 0, kAllocationBytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN probe window)");
    std::vector<std::uint8_t> send(
        checked_size(payload_bytes, "GIN probe payload"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(nccl->rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpyAsync(allocation, send.data(), send.size(),
                        cudaMemcpyHostToDevice, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(GIN probe source)");
    internal::throw_cuda(
        "gin-put-probe", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(before GIN window register)");
    internal::throw_nccl(
        "gin-put-probe",
        window_register(nccl->comm, allocation, kAllocationBytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN probe)");
    gin_probe_reset_kernel<<<1, 1, 0, exec->stream>>>(dev_comm);
    auto *barrier = static_cast<std::uint64_t *>(allocation) +
                    kBarrierOffset / sizeof(std::uint64_t);
    internal::throw_nccl(
        "gin-put-probe",
        ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum, nccl->comm,
                      exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllReduce(after GIN reset)");
    gin_probe_put_signal_kernel<<<1, 1, 0, exec->stream>>>(
        dev_comm, window, checked_size(payload_bytes, "GIN probe payload"),
        kReceiveOffset,
        reinterpret_cast<unsigned int *>(
            static_cast<std::byte *>(allocation) + kErrorOffset));
    internal::throw_cuda(
        "gin-put-probe", cudaGetLastError(), nccl->rank, nccl->world,
        session.session().identity(), session.session().generation(),
        ExchangeKind::ExpertCounts, "GIN put/signal kernel launch");
    internal::throw_cuda(
        "gin-put-probe", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(GIN put/signal)");
    unsigned int device_error = 0;
    std::vector<std::uint8_t> received(send.size());
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(received.data(),
                   static_cast<std::byte *>(allocation) + kReceiveOffset,
                   received.size(), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN received payload)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(&device_error,
                   static_cast<std::byte *>(allocation) + kErrorOffset,
                   sizeof(device_error), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN device error)");
    if (device_error != 0)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "device_error", nccl->rank, "0",
                      std::to_string(device_error));
    const auto source_rail = (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_rank = source_rail * lsa.nRanks + lsa.rank;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(source_rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    testing::GinPutSignalProbeResult probe;
    probe.payload_bytes = payload_bytes;
    probe.rail_rank = static_cast<std::uint32_t>(rail.rank);
    probe.rail_world = static_cast<std::uint32_t>(rail.nRanks);
    probe.local_rank = static_cast<std::uint32_t>(lsa.rank);
    probe.received_hash = hash(received);
    probe.expected_hash = hash(expected);
    if (received != expected)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "received payload hash", nccl->rank,
                      std::to_string(probe.expected_hash),
                      std::to_string(probe.received_hash));
    cleanup();
    return probe;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

testing::GinLinkBenchmarkResult testing::run_distributed_gin_link_benchmark(
    const NcclSession &session, const CudaExecution &execution,
    GinConnectionMode mode, std::uint32_t target_shift,
    const std::vector<std::uint64_t> &payload_bytes, std::uint32_t warmup,
    std::uint32_t measured, std::uint32_t throughput_puts_per_sample) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)mode;
  (void)target_shift;
  (void)payload_bytes;
  (void)warmup;
  (void)measured;
  (void)throughput_puts_per_sample;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-link-benchmark, field=headers, rank=0, "
      "expected=nccl_device.h, actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "gin-link-benchmark", "device", nccl->rank,
                    std::to_string(nccl->device),
                    std::to_string(exec->device));
  if (payload_bytes.empty() || measured == 0U ||
      throughput_puts_per_sample == 0U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-link-benchmark", "benchmark shape", nccl->rank,
                    "non-empty payloads and positive samples/puts", "invalid");
  const auto maximum_payload =
      *std::max_element(payload_bytes.begin(), payload_bytes.end());
  if (maximum_payload == 0ULL)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-link-benchmark", "payload_bytes", nccl->rank,
                    "positive", "0");
  for (const auto bytes : payload_bytes)
    if (bytes == 0ULL || bytes > maximum_payload)
      transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                      "gin-link-benchmark", "payload_bytes", nccl->rank,
                      "1..max_payload", std::to_string(bytes));

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                             const ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-link-benchmark, field=symbol, rank=0, "
                      "expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
      load("ncclDevCommDestroy"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
      load("ncclCommWindowDeregister"));
  const auto mem_alloc = reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto nccl_result = query_properties(nccl->comm, &properties);
  if (nccl_result != ncclSuccess)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-link-benchmark, field=properties, rank=" +
            std::to_string(nccl->rank) + ", actual=" +
            ncclGetErrorString(nccl_result));
  const auto selected_type =
      mode == GinConnectionMode::Rail ? properties.railedGinType
                                      : properties.ginType;
  if (selected_type == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-link-benchmark, field=" +
            std::string(mode == GinConnectionMode::Rail ? "railedGinType"
                                                        : "ginType") +
            ", rank=" + std::to_string(nccl->rank) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");

  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0 ||
      nccl->world % static_cast<std::uint32_t>(lsa.nRanks) != 0U)
    transport_error(NotifyErrorCode::BackendUnavailable,
                    "gin-link-benchmark", "homogeneous topology", nccl->rank,
                    "at least two nodes and positive LSA", "rail=" +
                        std::to_string(rail.nRanks) + ",lsa=" +
                        std::to_string(lsa.nRanks));
  target_shift %= static_cast<std::uint32_t>(lsa.nRanks);
  const auto destination_node = (rail.rank + 1) % rail.nRanks;
  const auto destination_local =
      (lsa.rank + static_cast<int>(target_shift)) % lsa.nRanks;
  const auto destination =
      mode == GinConnectionMode::Rail
          ? destination_node
          : destination_node * lsa.nRanks + destination_local;
  const bool fanout_full = mode == GinConnectionMode::Full &&
                           std::getenv("MEGAMOE_GIN_BENCH_FANOUT") != nullptr;

  const auto align_4k = [](std::uint64_t value, const char *field) {
    return checked_add(value, 4095ULL, field) / 4096ULL * 4096ULL;
  };
  const auto receive_offset = align_4k(maximum_payload, "GIN receive offset");
  const auto barrier_offset = align_4k(
      checked_add(receive_offset, maximum_payload,
                  "GIN benchmark receive extent"),
      "GIN benchmark barrier offset");
  const auto allocation_bytes = checked_size(
      checked_add(barrier_offset, 4096ULL, "GIN benchmark window bytes"),
      "GIN benchmark window bytes");
  const auto duration_bytes = checked_size(
      checked_mul(measured, sizeof(unsigned long long),
                  "GIN benchmark duration bytes"),
      "GIN benchmark duration bytes");

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = 1;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 64;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType =
      mode == GinConnectionMode::Rail ? NCCL_GIN_CONNECTION_RAIL
                                      : NCCL_GIN_CONNECTION_FULL;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  unsigned long long *device_duration = nullptr;
  unsigned int *device_error = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (device_error != nullptr)
      (void)cudaFree(device_error);
    if (device_duration != nullptr)
      (void)cudaFree(device_duration);
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };

  try {
    internal::throw_nccl(
        "gin-link-benchmark",
        dev_comm_create(nccl->comm, &requirements, &dev_comm), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        mode == GinConnectionMode::Rail ? "ncclDevCommCreate(RAIL)"
                                        : "ncclDevCommCreate(FULL)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-link-benchmark", mem_alloc(&allocation, allocation_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN link benchmark)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaMalloc(&device_duration, duration_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN duration samples)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaMalloc(&device_error, sizeof(*device_error)),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN device error)");
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemsetAsync(allocation, 0, allocation_bytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN benchmark window)");
    std::vector<std::uint8_t> send(
        checked_size(maximum_payload, "GIN benchmark source"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          ((fanout_full ? 0ULL : static_cast<std::uint64_t>(nccl->rank)) *
               17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemcpyAsync(allocation, send.data(), send.size(),
                        cudaMemcpyHostToDevice, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(GIN benchmark source)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(before GIN benchmark registration)");
    internal::throw_nccl(
        "gin-link-benchmark",
        window_register(nccl->comm, allocation, allocation_bytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN link benchmark)");

    GinLinkBenchmarkResult benchmark;
    benchmark.mode = mode;
    benchmark.target_shift = target_shift;
    benchmark.rank = nccl->rank;
    benchmark.world = nccl->world;
    benchmark.local_rank = static_cast<std::uint32_t>(lsa.rank);
    benchmark.local_world = static_cast<std::uint32_t>(lsa.nRanks);
    benchmark.gin_type = static_cast<std::uint32_t>(properties.ginType);
    benchmark.railed_gin_type =
        static_cast<std::uint32_t>(properties.railedGinType);

    std::vector<std::uint32_t> batches{1U};
    if (throughput_puts_per_sample != 1U)
      batches.push_back(throughput_puts_per_sample);
    auto *barrier = reinterpret_cast<std::uint64_t *>(
        static_cast<std::byte *>(allocation) + barrier_offset);
    for (const auto bytes : payload_bytes) {
      for (const auto puts : batches) {
        gin_probe_reset_kernel<<<1, 1, 0, exec->stream>>>(dev_comm);
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemsetAsync(device_error, 0, sizeof(*device_error),
                            exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemsetAsync(GIN benchmark error)");
        internal::throw_nccl(
            "gin-link-benchmark",
            ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                          nccl->comm, exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "ncclAllReduce(after GIN benchmark reset)");
        gin_link_benchmark_kernel<<<1, 1, 0, exec->stream>>>(
            dev_comm, window, mode == GinConnectionMode::Full, fanout_full,
            destination, lsa.nRanks,
            checked_size(bytes, "GIN benchmark payload"),
            checked_size(receive_offset, "GIN benchmark receive offset"),
            warmup, measured, puts, device_duration, device_error);
        internal::throw_cuda(
            "gin-link-benchmark", cudaGetLastError(), nccl->rank, nccl->world,
            session.session().identity(), session.session().generation(),
            ExchangeKind::ExpertCounts, "GIN benchmark kernel launch");
        internal::throw_cuda(
            "gin-link-benchmark", cudaStreamSynchronize(exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaStreamSynchronize(GIN link benchmark)");
        unsigned int host_error = 0U;
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemcpy(&host_error, device_error, sizeof(host_error),
                       cudaMemcpyDeviceToHost),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemcpy(GIN benchmark error)");
        if (host_error != 0U)
          transport_error(NotifyErrorCode::NcclFailure,
                          "gin-link-benchmark", "device_error", nccl->rank,
                          "0", std::to_string(host_error));
        GinLinkBenchmarkCaseResult sample;
        sample.payload_bytes = bytes;
        sample.puts_per_sample = puts;
        sample.duration_ns.resize(measured);
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemcpy(sample.duration_ns.data(), device_duration,
                       duration_bytes, cudaMemcpyDeviceToHost),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemcpy(GIN benchmark durations)");
        benchmark.cases.push_back(std::move(sample));
      }
    }

    std::vector<std::uint8_t> received(send.size());
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemcpy(received.data(),
                   static_cast<std::byte *>(allocation) + receive_offset,
                   received.size(), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN benchmark received payload)");
    const auto source_node =
        (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_local =
        mode == GinConnectionMode::Rail
            ? lsa.rank
            : (lsa.rank + lsa.nRanks - static_cast<int>(target_shift)) %
                  lsa.nRanks;
    const auto source_rank = source_node * lsa.nRanks + source_local;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          ((fanout_full ? 0ULL : static_cast<std::uint64_t>(source_rank)) *
               17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    benchmark.received_hash = hash(received);
    benchmark.expected_hash = hash(expected);
    if (received != expected)
      transport_error(NotifyErrorCode::NcclFailure,
                      "gin-link-benchmark", "received payload hash",
                      nccl->rank, std::to_string(benchmark.expected_hash),
                      std::to_string(benchmark.received_hash));
    cleanup();
    return benchmark;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

ActivationTransportCapability probe_activation_transport_capability(
    std::uint32_t world_size) {
  ActivationTransportCapability result;
  result.compiled_nccl_version = NCCL_VERSION_CODE;
  int runtime = 0;
  if (ncclGetVersion(&runtime) == ncclSuccess)
    result.runtime_nccl_version = static_cast<std::uint32_t>(runtime);
#if __has_include(<nccl_device.h>)
  result.nccl_device_headers = true;
#endif
#if __has_include(<nccl_device/gin.h>)
  result.gin_headers_available = true;
#endif
  // Device put/signal/wait/flush are header-inline APIs. Their required host
  // lifecycle surface must all exist in the actually loaded runtime.
  constexpr const char *required_symbols[] = {
      "ncclCommQueryProperties",
      "ncclDevCommCreate",
      "ncclDevCommDestroy",
      "ncclMemAlloc",
      "ncclMemFree",
      "ncclCommWindowRegister",
      "ncclCommWindowDeregister",
      "ncclGetPeerDevicePointer",
      "ncclTeamRail",
      "ncclGinOutboxCreateRequirement",
      "ncclGinInboxA2ACreateRequirement",
      "ncclGinBarrierCreateRequirement",
  };
  result.gin_symbols_present = result.gin_headers_available;
  std::string first_missing;
  for (const auto *symbol : required_symbols) {
    if (dlsym(RTLD_DEFAULT, symbol) == nullptr) {
      result.gin_symbols_present = false;
      if (first_missing.empty())
        first_missing = symbol;
    }
  }
  result.gin_backend_implemented = true;

#if __has_include(<nccl_device/core.h>)
  if (result.gin_headers_available && result.gin_symbols_present) {
    using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                                ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                               ncclWindow_t *, int);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
    using MemFreeFn = ncclResult_t (*)(void *);
    const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
        dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
    const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommCreate"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto window_register = reinterpret_cast<WindowRegisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowRegister"));
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto mem_alloc = reinterpret_cast<MemAllocFn>(
        dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free = reinterpret_cast<MemFreeFn>(
        dlsym(RTLD_DEFAULT, "ncclMemFree"));
    ncclComm_t probe_comm = nullptr;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    void *allocation = nullptr;
    bool dev_comm_created = false;
    std::string init_error;
    ncclUniqueId id{};
    auto nccl_result = ncclGetUniqueId(&id);
    if (nccl_result == ncclSuccess)
      nccl_result = ncclCommInitRank(&probe_comm, 1, id, 0);
    if (nccl_result != ncclSuccess) {
      init_error = std::string("ncclCommInitRank: ") +
                   ncclGetErrorString(nccl_result);
    } else {
      ncclCommProperties properties{};
      properties.size = sizeof(properties);
      properties.magic = NCCL_API_MAGIC;
      properties.version = NCCL_VERSION_CODE;
      nccl_result = query_properties(probe_comm, &properties);
      if (nccl_result != ncclSuccess) {
        init_error = std::string("ncclCommQueryProperties: ") +
                     ncclGetErrorString(nccl_result);
      } else if (properties.ginType == NCCL_GIN_TYPE_NONE &&
                 properties.railedGinType == NCCL_GIN_TYPE_NONE) {
        init_error = "ncclCommQueryProperties: no GIN context type";
      } else {
        ncclDevCommRequirements_t requirements{};
        requirements.size = sizeof(requirements);
        requirements.magic = NCCL_API_MAGIC;
        requirements.version = NCCL_VERSION_CODE;
        requirements.ginContextCount = 1;
        requirements.ginExclusiveContexts = true;
        requirements.ginQueueDepth = 32;
        requirements.ginSignalCount = 4;
        requirements.ginConnectionType =
            properties.railedGinType != NCCL_GIN_TYPE_NONE
                ? NCCL_GIN_CONNECTION_RAIL
                : NCCL_GIN_CONNECTION_FULL;
        nccl_result = dev_comm_create(probe_comm, &requirements, &dev_comm);
        if (nccl_result != ncclSuccess) {
          init_error = std::string("ncclDevCommCreate(GIN): ") +
                       ncclGetErrorString(nccl_result);
        } else {
          dev_comm_created = true;
          nccl_result = mem_alloc(&allocation, 4096);
          if (nccl_result != ncclSuccess) {
            init_error = std::string("ncclMemAlloc(window): ") +
                         ncclGetErrorString(nccl_result);
          } else {
            nccl_result = window_register(
                probe_comm, allocation, 4096, &window,
                NCCL_WIN_STRICT_ORDERING);
            if (nccl_result != ncclSuccess)
              init_error = std::string("ncclCommWindowRegister: ") +
                           ncclGetErrorString(nccl_result);
            else
              result.gin_runtime_initializable = true;
          }
        }
      }
    }
    if (window)
      (void)window_deregister(probe_comm, window);
    if (allocation)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(probe_comm, &dev_comm);
    if (probe_comm)
      (void)ncclCommDestroy(probe_comm);
    if (!result.gin_runtime_initializable)
      result.gin_unavailable_reason = std::move(init_error);
  }
#endif
  result.gin_usable = result.gin_headers_available &&
                      result.gin_symbols_present &&
                      result.gin_runtime_initializable &&
                      result.gin_backend_implemented;
  if (!result.gin_usable) {
    std::ostringstream message;
    message << "GIN unavailable: headers=" << result.gin_headers_available
            << ", symbols=" << result.gin_symbols_present
            << ", runtime_initializable="
            << result.gin_runtime_initializable
            << ", backend_implemented=" << result.gin_backend_implemented
            << ", compiled_nccl=" << result.compiled_nccl_version
            << ", runtime_nccl=" << result.runtime_nccl_version;
    if (!first_missing.empty())
      message << ", first_missing_symbol=" << first_missing;
    if (!result.gin_unavailable_reason.empty())
      message << ", initialization_error=" << result.gin_unavailable_reason;
    result.gin_unavailable_reason = message.str();
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess)
    devices = 0;
  const auto checked_world = std::min<std::uint32_t>(
      world_size, static_cast<std::uint32_t>(std::max(devices, 0)));
  result.peer_access_matrix.resize(
      static_cast<std::size_t>(checked_world) * checked_world, 0);
  for (std::uint32_t source = 0; source < checked_world; ++source) {
    for (std::uint32_t destination = 0; destination < checked_world;
         ++destination) {
      int access = source == destination ? 1 : 0;
      if (source != destination)
        (void)cudaDeviceCanAccessPeer(&access, static_cast<int>(source),
                                      static_cast<int>(destination));
      result.peer_access_matrix[static_cast<std::size_t>(source) *
                                    checked_world +
                                destination] =
          static_cast<std::uint8_t>(access != 0);
    }
  }
  if (devices > 0) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
      result.system_scope_atomics = properties.major >= 6;
  }
  return result;
}

ActivationTransportCapability probe_activation_transport_capability(
    const NcclSession &session, const CudaExecution &execution) {
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto cuda = internal::checked_execution_state(execution);
  if (nccl->device != cuda->device)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-distributed-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(cuda->device));

  auto result = probe_activation_transport_capability(nccl->world);
  // The world=1 probe above is useful for loader diagnostics but cannot prove
  // that a rail GIN team can be formed.  Replace only its initialization
  // result with the collective probe below.
  result.gin_runtime_initializable = false;
  result.gin_backend_implemented = true;
  result.gin_usable = false;
  result.gin_unavailable_reason.clear();

#if __has_include(<nccl_device/core.h>)
  if (result.gin_headers_available && result.gin_symbols_present) {
    using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                                ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                               ncclWindow_t *, int);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
    using MemFreeFn = ncclResult_t (*)(void *);
    const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
        dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
    const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommCreate"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto window_register = reinterpret_cast<WindowRegisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowRegister"));
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto mem_alloc = reinterpret_cast<MemAllocFn>(
        dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free = reinterpret_cast<MemFreeFn>(
        dlsym(RTLD_DEFAULT, "ncclMemFree"));

    ncclCommProperties_t properties{};
    properties.size = sizeof(properties);
    properties.magic = NCCL_API_MAGIC;
    properties.version = NCCL_VERSION_CODE;
    ncclDevCommRequirements_t requirements{};
    requirements.size = sizeof(requirements);
    requirements.magic = NCCL_API_MAGIC;
    requirements.version = NCCL_VERSION_CODE;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    void *allocation = nullptr;
    bool dev_comm_created = false;
    std::string error;

    auto nccl_result = query_properties(nccl->comm, &properties);
    if (nccl_result != ncclSuccess) {
      error = std::string("ncclCommQueryProperties: ") +
              ncclGetErrorString(nccl_result);
    } else if (properties.ginType == NCCL_GIN_TYPE_NONE &&
               properties.railedGinType == NCCL_GIN_TYPE_NONE) {
      error = "ncclCommQueryProperties: no GIN context type";
    } else {
      requirements.ginContextCount = 1;
      requirements.ginExclusiveContexts = true;
      requirements.ginQueueDepth = 32;
      requirements.ginSignalCount = 4;
      requirements.ginConnectionType =
          properties.railedGinType != NCCL_GIN_TYPE_NONE
              ? NCCL_GIN_CONNECTION_RAIL
              : NCCL_GIN_CONNECTION_FULL;
      nccl_result = dev_comm_create(nccl->comm, &requirements, &dev_comm);
      if (nccl_result != ncclSuccess) {
        error = std::string("ncclDevCommCreate(GIN): ") +
                ncclGetErrorString(nccl_result);
      } else {
        dev_comm_created = true;
        nccl_result = mem_alloc(&allocation, 4096);
        if (nccl_result != ncclSuccess) {
          error = std::string("ncclMemAlloc(window): ") +
                  ncclGetErrorString(nccl_result);
        } else {
          nccl_result = window_register(nccl->comm, allocation, 4096, &window,
                                        NCCL_WIN_STRICT_ORDERING);
          if (nccl_result != ncclSuccess)
            error = std::string("ncclCommWindowRegister: ") +
                    ncclGetErrorString(nccl_result);
          else
            result.gin_runtime_initializable = true;
        }
      }
    }
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
    if (!result.gin_runtime_initializable)
      result.gin_unavailable_reason = std::move(error);
  }
#endif

  result.gin_usable = result.gin_headers_available &&
                      result.gin_symbols_present &&
                      result.gin_runtime_initializable &&
                      result.gin_backend_implemented;

  std::ostringstream message;
  message << "GIN distributed probe: headers="
          << result.gin_headers_available << ", symbols="
          << result.gin_symbols_present << ", runtime_initializable="
          << result.gin_runtime_initializable << ", backend_implemented="
          << result.gin_backend_implemented << ", rank=" << nccl->rank
          << ", world=" << nccl->world << ", compiled_nccl="
          << result.compiled_nccl_version << ", runtime_nccl="
          << result.runtime_nccl_version;
  if (!result.gin_unavailable_reason.empty())
    message << ", initialization_error=" << result.gin_unavailable_reason;
  result.gin_unavailable_reason = message.str();
  return result;
}

bool testing::run_local_p2p_put_signal_probe(std::uint32_t source_device,
                                             std::uint32_t destination_device,
                                             std::uint64_t elements) {
  if (elements == 0)
    return false;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return false;
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
  bool ok = false;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(checked_mul(
                                   elements, sizeof(std::uint64_t),
                                   "peer destination bytes"),
                               "peer destination bytes")) != cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemset(signal, 0, sizeof(*signal));
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable = cudaDeviceEnablePeerAccess(
        static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(checked_mul(
                              elements, sizeof(std::uint64_t),
                              "peer source bytes"),
                          "peer source bytes")) != cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  if (cudaMemsetAsync(source, 0x5a,
                      checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                               "peer copy bytes"),
                                   "peer copy bytes"),
                      source_stream) != cudaSuccess ||
      cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
  const auto blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
  const auto run = [&](bool early_signal, unsigned int &observed_errors) {
    (void)cudaSetDevice(static_cast<int>(destination_device));
    if (cudaMemsetAsync(destination, 0,
                        checked_size(checked_mul(elements,
                                                 sizeof(std::uint64_t),
                                                 "peer reset bytes"),
                                     "peer reset bytes"),
                        destination_stream) != cudaSuccess ||
        cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
            cudaSuccess ||
        cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(destination_stream) != cudaSuccess)
      return false;
    peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
        destination, signal, errors, elements, expected);
    if (cudaGetLastError() != cudaSuccess)
      return false;
    (void)cudaSetDevice(static_cast<int>(source_device));
    if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                        source_stream) != cudaSuccess ||
        cudaStreamSynchronize(source_stream) != cudaSuccess)
      return false;
    peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
        source, destination, signal, completed_blocks, elements, early_signal);
    if (cudaGetLastError() != cudaSuccess)
      return false;
    // The destination independently observes the signal and validates every
    // word before the source stream is synchronized.
    (void)cudaSetDevice(static_cast<int>(destination_device));
    if (cudaStreamSynchronize(destination_stream) != cudaSuccess ||
        cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    (void)cudaSetDevice(static_cast<int>(source_device));
    return cudaStreamSynchronize(source_stream) == cudaSuccess;
  };
  {
    unsigned int positive_errors = 0;
    unsigned int early_signal_errors = 0;
    ok = run(false, positive_errors) && positive_errors == 0U &&
         run(true, early_signal_errors) && early_signal_errors != 0U;
  }
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return ok;
}

double testing::measure_local_p2p_put_bandwidth_gbps(
    std::uint32_t source_device, std::uint32_t destination_device,
    std::uint64_t bytes, std::uint32_t warmup, std::uint32_t measured) {
  if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0 || measured == 0)
    return 0.0;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return 0.0;
  const auto elements = bytes / sizeof(std::uint64_t);
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  double result = 0.0;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable = cudaDeviceEnablePeerAccess(
        static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemsetAsync(source, 0x5a, checked_size(bytes, "P2P bandwidth bytes"),
                        source_stream);
  if (cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
    constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
    const auto blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
    double measured_seconds = 0.0;
    for (std::uint32_t iteration = 0; iteration < warmup + measured;
         ++iteration) {
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
              cudaSuccess ||
          cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
              cudaSuccess ||
          cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
          destination, signal, errors, elements, expected);
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                          source_stream) != cudaSuccess ||
          cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      const auto start = std::chrono::steady_clock::now();
      peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
          source, destination, signal, completed_blocks, elements, false);
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      const auto finish = std::chrono::steady_clock::now();
      unsigned int observed_errors = 0;
      if (cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          observed_errors != 0U)
        goto cleanup;
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      if (iteration >= warmup)
        measured_seconds +=
            std::chrono::duration<double>(finish - start).count();
    }
    if (measured_seconds > 0.0)
      result = static_cast<double>(bytes) * measured /
               measured_seconds / 1.0e9;
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return result;
}
