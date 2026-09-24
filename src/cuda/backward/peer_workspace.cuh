#pragma once

// Host ownership for rank-local Backward communication payloads. A workspace
// belongs to one execution slot, not to any retained Forward tape. Exact-shape
// reuse avoids reallocating warm calls without retaining historical hotspots.
struct BackwardPeerWorkspace {
  struct Export {
    cudaIpcMemHandle_t handle{};
    std::size_t contribution = 0, dgate = 0;
  };
  struct Buffer {
    std::shared_ptr<tensor_storage_detail::Allocation> allocation;
    Export descriptor;
    std::uint64_t rows = 0, routes = 0;
    std::size_t table = 0;
    CUtensorMap dr_map{}, dr_mn_map{};
  };

  CollectiveTensorPreparation preparation;
  std::shared_ptr<internal::NcclSessionState> nccl;
  CudaExecution execution;
  cudaStream_t stream;
  unsigned local_world;
  std::shared_ptr<tensor_storage_detail::Allocation> exchange;
  std::shared_ptr<Buffer> buffer;
  PeerMappings mappings;
  std::vector<Export> exports, known;
  std::vector<backward_detail::PeerPayload> payloads;
  std::uint64_t generation = 0;

  static void check(cudaError_t error) {
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("Backward peer workspace: ") +
                               cudaGetErrorString(error));
  }

  BackwardPeerWorkspace(const NcclSession &session, const CudaExecution &exec,
                         unsigned local_size)
      : preparation(session, exec),
        nccl(internal::NcclSessionAccess::checked(session)), execution(exec),
        stream(internal::checked_execution_state(exec)->stream),
        local_world(local_size) {
    preparation.run([&] {
      exchange = tensor_storage_detail::allocate(
          nccl->world * sizeof(Export), exec.device_ordinal());
      exports.resize(nccl->world);
      known.resize(nccl->world);
      payloads.resize(nccl->world);
      mappings.device = exec.device_ordinal();
      mappings.nccl = nccl;
      mappings.owned.resize(nccl->world);
      mappings.bases.resize(nccl->world);
    });
  }

  void close_mappings() noexcept {
    for (unsigned rank = 0; rank < nccl->world; ++rank) {
      if (mappings.owned[rank])
        (void)cudaIpcCloseMemHandle(mappings.bases[rank]);
      mappings.bases[rank] = nullptr;
      mappings.owned[rank] = 0;
    }
  }

  ~BackwardPeerWorkspace() noexcept {
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    (void)cudaStreamSynchronize(stream);
    close_mappings();
  }

  std::shared_ptr<Buffer> allocate(std::uint64_t rows, std::uint64_t routes,
                                  std::uint64_t hidden) {
    auto result = std::make_shared<Buffer>();
    result->rows = rows;
    result->routes = routes;
    std::size_t cursor = 0;
    // Empty destinations still need a valid, nonzero TMA descriptor.
    append(cursor, checked_mul(std::max(rows, std::uint64_t{1}), hidden,
                                "Backward dR elements"),
            sizeof(transport::BFloat16), "Backward compact dR");
    result->descriptor.contribution = append(
        cursor, checked_mul(routes, hidden, "Backward contribution elements"),
        sizeof(transport::BFloat16), "Backward source contributions");
    result->descriptor.dgate =
        append(cursor, routes, sizeof(float), "Backward source dGate");
    result->table = append(cursor, nccl->world,
                            sizeof(backward_detail::PeerPayload),
                            "Backward peer payload addresses");
    result->allocation =
        tensor_storage_detail::allocate(cursor, execution.device_ordinal());
    auto *base = result->allocation->data;
    check(cudaIpcGetMemHandle(&result->descriptor.handle, base));
    result->dr_map = make_bf16_tma_map(
        base, hidden, std::max(rows, std::uint64_t{1}),
        backward_tuning::kActivationTileM, "actual Backward dR",
        backward_tuning::kTileK);
    result->dr_mn_map = make_bf16_tma_map_mn(
        base, hidden, std::max(rows, std::uint64_t{1}), "actual Backward dR MN");
    return result;
  }

  void prepare(std::uint64_t next_generation, std::uint64_t rows,
                std::uint64_t routes, std::uint64_t hidden) {
    // A retained Notify plan is immutable. Repeating that Backward needs no
    // allocation, address exchange or host collective.
    if (buffer && generation == next_generation)
      return;
    std::shared_ptr<Buffer> next;
    preparation.run([&] {
      next = buffer && buffer->rows == rows && buffer->routes == routes
          ? buffer : allocate(rows, routes, hidden);
    });
    generation = 0;
    exports[nccl->rank] = next->descriptor;
    auto *device = static_cast<Export *>(exchange->data);
    check(cudaMemcpyAsync(device + nccl->rank, &exports[nccl->rank],
                           sizeof(Export), cudaMemcpyHostToDevice, stream));
    const auto error = ncclAllGather(device + nccl->rank, device, sizeof(Export),
                                     ncclUint8, nccl->comm, stream);
    if (error != ncclSuccess)
      throw std::runtime_error(std::string("Backward peer address exchange: ") +
                               ncclGetErrorString(error));
    check(cudaMemcpyAsync(exports.data(), device, exchange->bytes,
                           cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    try {
      preparation.run([&] {
        const auto first = nccl->rank / local_world * local_world;
        for (unsigned rank = first; rank < first + local_world; ++rank) {
          auto &base = mappings.bases[rank];
          if (rank == nccl->rank) {
            base = next->allocation->data;
          } else if (!base || std::memcmp(&known[rank].handle,
                                          &exports[rank].handle,
                                          sizeof(cudaIpcMemHandle_t))) {
            if (mappings.owned[rank]) {
              check(cudaIpcCloseMemHandle(base));
              mappings.owned[rank] = 0;
              base = nullptr;
            }
            check(cudaIpcOpenMemHandle(&base, exports[rank].handle,
                                        cudaIpcMemLazyEnablePeerAccess));
            mappings.owned[rank] = 1;
          }
          payloads[rank] = {
              static_cast<transport::BFloat16 *>(base),
              at<transport::BFloat16>(base, exports[rank].contribution),
              at<float>(base, exports[rank].dgate)};
        }
      });
    } catch (...) {
      // All ranks saw the failed import. Close even successful imports before
      // any exporter releases its candidate allocation; leave no stale cache
      // that a retry could mistake for a prepared immutable plan.
      close_mappings();
      preparation.run([] {});
      buffer.reset();
      throw;
    }
    // The successful agreement also proves old imports were closed. Only now
    // may replacing the cache release this rank's previous exported payload.
    buffer = std::move(next);
    known = exports;
    check(cudaMemcpyAsync(at<void>(buffer->allocation->data, buffer->table),
                           payloads.data(), payloads.size() * sizeof(payloads[0]),
                           cudaMemcpyHostToDevice, stream));
    generation = next_generation;
  }

  const backward_detail::PeerPayload *device_payloads() const {
    return at<backward_detail::PeerPayload>(buffer->allocation->data,
                                            buffer->table);
  }
  std::uint64_t bytes() const {
    return exchange->bytes + (buffer ? buffer->allocation->bytes : 0U);
  }
};
