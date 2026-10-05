#pragma once

// Host ownership for rank-local Backward communication payloads. A workspace
// belongs to one execution slot, not to any retained Forward tape. Logical
// rows are independent of storage capacity: one residency keeps its payloads
// and IPC mappings across variable microbatches and expert routes.
struct BackwardPeerWorkspace {
  struct Export {
    cudaIpcMemHandle_t handle{};
    std::size_t contribution = 0, dgate = 0;
    std::size_t return_slots = 0, return_state = 0, consumed = 0;
  };
  struct Buffer {
    std::shared_ptr<tensor_storage_detail::Allocation> allocation;
    Export descriptor;
    std::uint64_t row_capacity = 0, route_capacity = 0, hidden = 0;
    std::size_t table = 0, partitions = 0;
    CUtensorMap dr_map{}, dr_mn_map{};
    unsigned return_depth = 0;
  };

  CollectiveTensorPreparation preparation;
  std::shared_ptr<internal::NcclSessionState> nccl;
  CudaExecution execution;
  cudaStream_t stream;
  unsigned local_world;
  unsigned dispatch_ctas = 0;
  unsigned owned_experts;
  std::uint64_t max_tokens;
  bool network;
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
        owned_experts(config.max_owned_experts),
        max_tokens(config.max_tokens_per_rank),
        network(config.max_nodes > 1U),
        initial_rows(config.backward_peer_capacity_rows),
        max_rows(config.max_destination_rows_per_rank),
        max_routes(checked_mul(config.max_tokens_per_rank, config.max_top_k,
                               "Backward route capacity")) {
    if (!initial_rows)
      initial_rows = checked_mul(max_routes, 2ULL, "Backward initial row capacity");
    initial_rows = std::min(initial_rows, max_rows);
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                 exec.device_ordinal()));
    dispatch_ctas = static_cast<unsigned>(sms) & ~1U;
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

  static std::uint64_t return_budget(
      const static_forward::ComputeRingLayout &forward, std::uint64_t hidden) {
    constexpr std::uint64_t kExtraBytes = 448ULL << 20U;
    const auto routes = std::uint64_t(forward.shape.tokens) *
                        forward.contribution_depth * forward.shape.top_k;
    return checked_mul(routes, hidden * 2ULL + sizeof(float),
                       "Backward source capacity") + kExtraBytes;
  }

  bool full_dx_fits(const static_forward::ComputeRingLayout &forward,
                    std::uint64_t hidden, std::uint64_t padded_rows) const {
    // Local dX already writes directly to its source; staging would add a copy.
    if (!network)
      return false;
    const auto budget = return_budget(forward, hidden);
    const auto source = checked_mul(max_routes, hidden * 2ULL + sizeof(float),
                                    "Backward full source capacity");
    const auto dx = checked_mul(padded_rows, hidden * 2ULL, "Backward full dX capacity");
    return source <= budget && dx <= budget - source;
  }

  // Spend a bounded increment on source contributions, not worst-case expert
  // payloads. A complete source range preserves canonical GEMM tiles; larger
  // contexts retain window credits with up to four times the Forward depth.
  static_forward::ComputeRingLayout return_layout(
      const static_forward::ComputeRingLayout &forward, std::uint64_t hidden) {
    auto result = forward;
    const auto row_bytes = checked_mul(hidden, 2ULL, "Backward contribution row") + sizeof(float);
    const auto budget = return_budget(forward, hidden);
    const auto full_bytes = checked_mul(max_routes, row_bytes, "Backward full source capacity");
    const bool canonical = full_bytes <= budget;
    if (canonical) {
      result.shape.tokens = static_cast<unsigned>(max_tokens);
      result.shape.sources = result.shape.world;
      result.contribution_depth = 1U;
    } else {
      // Coarsen only the dX regrouping window within the same byte budget.
      // Larger windows reduce expert tails without growing Forward X/S/C or
      // replacing the bounded Return protocol with a full-context allocation.
      // A bounded wire packet can join rows from distant expert M-blocks.
      // Keep the whole worst-case window in dX slots so packing that packet
      // never awaits a future block that needs to recycle its first row.
      const auto slot_tokens =
          std::uint64_t(forward.depth - owned_experts) * forward.shape.tile_m /
          (std::uint64_t(std::min(forward.shape.sources, forward.shape.world)) *
           forward.shape.top_k);
      result.shape.tokens = static_cast<unsigned>(std::min(
          std::uint64_t(forward.shape.tokens) * backward_tuning::kReturnWindowScale,
          std::min(slot_tokens, budget / (forward.shape.top_k * row_bytes))));
      const auto window_bytes = checked_mul(
          std::uint64_t(result.shape.tokens) * result.shape.top_k,
          row_bytes, "Backward contribution window");
      result.contribution_depth = static_cast<unsigned>(std::min(
          std::uint64_t(forward.contribution_depth) * 4U, budget / window_bytes));
    }
    // Remote packets stop at an M-block boundary; their payload ring stays
    // bounded. Local dX is written directly, so only its counters grow.
    const auto depth = canonical && !network
        ? std::max(std::uint64_t(forward.depth), result.shape.max_mblocks(owned_experts))
        : std::uint64_t(forward.depth);
    if (depth > UINT32_MAX / backward_tuning::kTileM)
      throw std::invalid_argument("Backward payload rows exceed uint32");
    result.depth = static_cast<unsigned>(depth);
    result.max_windows = result.shape.windows(max_tokens);
    return result;
  }

  ~BackwardPeerWorkspace() noexcept {
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    (void)cudaStreamSynchronize(stream);
    close_mappings();
  }

  std::shared_ptr<Buffer> allocate(std::uint64_t rows, std::uint64_t routes,
                                  std::uint64_t hidden,
                                  const static_forward::ComputeRingLayout *ring) {
    auto result = std::make_shared<Buffer>();
    result->row_capacity = rows;
    result->route_capacity = routes;
    result->hidden = hidden;
    result->return_depth = ring ? ring->depth : 0U;
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
    if (ring) {
      result->partitions = append(cursor,
          ring->max_windows * owned_experts * ring->shape.sources,
          sizeof(backward_detail::ReturnPartition), "Backward Return intervals");
      result->descriptor.return_slots = append(cursor, ring->depth,
          sizeof(backward_detail::ReturnSlot), "Backward dX slot credits");
      result->descriptor.return_state = append(cursor, 1U,
          sizeof(backward_detail::SourceReturnState), "Backward source window head");
      result->descriptor.consumed = append(cursor, ring->contribution_depth,
          sizeof(std::uint64_t), "Backward source window credits");
    }
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
                std::uint64_t routes, std::uint64_t hidden,
                const static_forward::ComputeRingLayout *ring = nullptr) {
    // A retained Notify plan is immutable. Repeating that Backward needs no
    // allocation, address exchange or host collective.
    const auto depth = ring ? ring->depth : 0U;
    if (ring)
      routes = std::uint64_t(ring->shape.tokens) * ring->contribution_depth *
               ring->shape.top_k;
    if (buffer && generation == next_generation && buffer->return_depth == depth)
      return;
    generation = 0;
    std::shared_ptr<Buffer> next;
    const bool bindings_changed = preparation.run_with_change([&] {
      if (buffer && buffer->hidden == hidden && buffer->return_depth == depth &&
          rows <= buffer->row_capacity &&
          routes <= buffer->route_capacity) {
        next = buffer;
      } else {
        // A rare hotspot may exceed the initial heuristic, but must not drop
        // routes. Grow geometrically; ordinary shape changes keep the handle.
        const auto reserved = buffer
            ? std::min(max_rows, checked_mul(buffer->row_capacity, 2ULL,
                                             "Backward grown row capacity"))
            : initial_rows;
        next = allocate(std::max(rows, reserved),
                        ring ? routes : std::max(routes, max_routes), hidden, ring);
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
      return next != buffer;
    });
    // Logical rows and call generations do not change peer addresses. The
    // existing allocation agreement also tells unchanged ranks when a peer
    // grew, so all ranks enter the remapping path together only when needed.
    if (!bindings_changed) {
      generation = next_generation;
      return;
    }
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
              at<float>(base, exports[rank].dgate),
              ring ? at<backward_detail::ReturnSlot>(base, exports[rank].return_slots) : nullptr,
              ring ? at<backward_detail::SourceReturnState>(base, exports[rank].return_state) : nullptr,
              ring ? at<std::uint64_t>(base, exports[rank].consumed) : nullptr};
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
