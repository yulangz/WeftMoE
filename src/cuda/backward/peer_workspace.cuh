#pragma once

// Host ownership for rank-local Backward communication payloads. A workspace
// belongs to one execution slot, not to any retained Forward tape. Logical
// rows are independent of storage capacity: one residency keeps its payloads
// and IPC mappings across variable microbatches and expert routes.
struct BackwardPeerWorkspace {
  struct Export {
    cudaIpcMemHandle_t handle{};
    std::size_t contribution = 0, dgate = 0;
  };
  struct Buffer {
    std::shared_ptr<tensor_storage_detail::Allocation> allocation;
    Export descriptor;
    std::uint64_t row_capacity = 0, route_capacity = 0, hidden = 0;
    std::size_t table = 0;
    CUtensorMap dr_map{}, dr_mn_map{};
  };

  CollectiveTensorPreparation preparation;
  std::shared_ptr<internal::NcclSessionState> nccl;
  CudaExecution execution;
  cudaStream_t stream;
  unsigned local_world;
  std::uint64_t initial_rows, max_rows, max_routes;
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
                         unsigned local_size, const ActivationTransportPoolConfig &config)
      : preparation(session, exec),
        nccl(internal::NcclSessionAccess::checked(session)), execution(exec),
        stream(internal::checked_execution_state(exec)->stream),
        local_world(local_size),
        initial_rows(config.backward_peer_capacity_rows),
        max_rows(config.max_destination_rows_per_rank),
        max_routes(checked_mul(config.max_tokens_per_rank, config.max_top_k,
                               "Backward route capacity")) {
    if (!initial_rows)
      initial_rows = checked_mul(max_routes, 2ULL, "Backward initial row capacity");
    initial_rows = std::min(initial_rows, max_rows);
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
    mappings.close();
  }

  ~BackwardPeerWorkspace() noexcept {
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    (void)cudaStreamSynchronize(stream);
    close_mappings();
  }

  std::shared_ptr<Buffer> allocate(std::uint64_t rows, std::uint64_t routes,
                                  std::uint64_t hidden) {
    auto result = std::make_shared<Buffer>();
    result->row_capacity = rows;
    result->route_capacity = routes;
    result->hidden = hidden;
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
    return result;
  }

  void prepare(std::uint64_t next_generation, std::uint64_t rows,
                std::uint64_t routes, std::uint64_t hidden) {
    // A retained Notify plan is immutable. Repeating that Backward needs no
    // allocation, address exchange or host collective.
    if (buffer && generation == next_generation)
      return;
    generation = 0;
    std::shared_ptr<Buffer> next;
    preparation.run([&] {
      if (buffer && buffer->hidden == hidden && rows <= buffer->row_capacity &&
          routes <= buffer->route_capacity) {
        next = buffer;
      } else {
        // A rare hotspot may exceed the initial heuristic, but must not drop
        // routes. Grow geometrically; ordinary shape changes keep the handle.
        const auto reserved = buffer
            ? std::min(max_rows, checked_mul(buffer->row_capacity, 2ULL,
                                             "Backward grown row capacity"))
            : initial_rows;
        next = allocate(std::max(rows, reserved), std::max(routes, max_routes), hidden);
      }
      // Preserve the logical TMA bounds, including zero fill of tail tiles.
      // The descriptor may change without replacing or remapping the storage.
      next->dr_map = make_bf16_tma_map(
          next->allocation->data, hidden, std::max(rows, std::uint64_t{1}),
          backward_tuning::kActivationTileM, "actual Backward dR",
          backward_tuning::kTileK);
      next->dr_mn_map = make_bf16_tma_map_mn(
          next->allocation->data, hidden, std::max(rows, std::uint64_t{1}),
          "actual Backward dR MN");
    });
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
