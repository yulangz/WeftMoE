__global__ void peer_put_signal_kernel(const std::uint64_t *source,
                                       std::uint64_t *destination,
                                       unsigned int *signal,
                                       unsigned int *completed_blocks,
                                       std::uint64_t count,
                                       bool early_signal) {
  if (early_signal && blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    atomicExch_system(signal, 1U);
  }
  if (early_signal && blockIdx.x != 0) {
    const auto begin = clock64();
    while (clock64() - begin < 1000000ULL) {
    }
  }
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    destination[index] = source[index];
  __syncthreads();
  if (threadIdx.x == 0) {
    // Every producer block publishes its payload before joining the completion
    // counter. Only the last block may publish the remote signal.
    __threadfence_system();
    const auto prior = atomicAdd(completed_blocks, 1U);
    if (!early_signal && prior + 1U == gridDim.x) {
      __threadfence_system();
      atomicExch_system(signal, 1U);
    }
  }
}

__global__ void peer_signal_verify_kernel(const std::uint64_t *destination,
                                          const unsigned int *signal,
                                          unsigned int *errors,
                                          std::uint64_t count,
                                          std::uint64_t expected) {
  while (atomicAdd_system(const_cast<unsigned int *>(signal), 0U) == 0U) {
  }
  __threadfence_system();
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    if (destination[index] != expected)
      atomicAdd(errors, 1U);
}

__global__ void initialize_transport_credits_kernel(unsigned int *credits,
                                                    std::uint64_t count,
                                                    unsigned int value) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < count)
    credits[index] = value;
}

__global__ void initialize_p2p_barrier_kernel(unsigned long long *record,
                                               unsigned long long slot_index,
                                               unsigned long long epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    record[0] = slot_index;
    record[1] = epoch;
  }
}

#if WEFTMOE_DEBUG_VALIDATE
__global__ void validate_p2p_barrier_kernel(
    const unsigned long long *records, std::uint32_t world,
    unsigned long long expected_slot, unsigned long long expected_epoch,
    TransportDeviceStatus *status) {
  const auto rank = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (rank >= world)
    return;
  if (records[rank * 2U] != expected_slot ||
      records[rank * 2U + 1U] != expected_epoch) {
    status->error_packet = rank;
    status->error_field = 19U;
    atomicCAS(&status->error, 0U, 1U);
  }
}
#endif

template <typename T> __device__ T *peer_at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

#if WEFTMOE_DEBUG_VALIDATE
template <typename Args>
__device__ __forceinline__ void
production_terminal_debug_progress(const Args &a, ProductionStatusPhase phase,
                                   unsigned long long value) {
  if (phase == ProductionStatusPhase::Forward && a.progress != nullptr &&
      blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        progress(a.progress[kForwardTerminal]);
    progress.store(value, ::cuda::memory_order_release);
  }
}
#else
template <typename Args>
__device__ __forceinline__ void
production_terminal_debug_progress(const Args &, ProductionStatusPhase,
                                   unsigned long long) {}
#endif

template <bool StaticLocal, typename Args, typename Status>
__device__ __forceinline__ void publish_production_terminal_status(
    const Args &a, Status *status, ProductionStatusPhase phase,
    std::size_t ordinary_record_offset, std::size_t gin_record_offset,
    unsigned long long prior_gin_signal_count) {
  if (blockIdx.x != 0U)
    return;
  production_terminal_debug_progress(a, phase, 10ULL);
  const auto local_world =
      a.local_world_size == 0U ? a.world : a.local_world_size;
  const auto node = a.local_world_size == 0U ? 0U : a.node;
  const auto first_local = node * local_world;
  // Legacy cancellation and debug checks may have claimed the status word
  // before structured fields were written. Normalize that already-owned local
  // failure at the single production boundary so it cannot be mistaken for a
  // successful record by the host agreement.
  if (threadIdx.x == 0U && status->error != 0U && status->error_code == 0U) {
    status->error_origin_rank = a.rank;
    status->error_code = status->error;
    status->error_expected = 0ULL;
    status->error_actual = status->error;
    __threadfence_system();
  }
  __syncthreads();
  const ProductionStatusRecord local{
      a.dispatch.identity.generation,
      static_cast<unsigned long long>(phase),
      a.rank,
      status->error == 0U ? 0ULL
                          : static_cast<unsigned long long>(status->error_code),
      status->error == 0U ? 0ULL
                          : static_cast<unsigned long long>(status->error_field),
      status->error_expected,
      status->error_actual,
      a.epoch};

  // Publish this rail's local record to every node-local peer. No record is
  // consumed until its epoch is observed with acquire ordering.
  for (std::uint32_t local_peer = threadIdx.x; local_peer < local_world;
       local_peer += blockDim.x) {
    const auto peer = first_local + local_peer;
    auto *records = peer_at<ProductionStatusRecord>(
        a.peer_bases[peer], ordinary_record_offset);
    auto record = local;
    record.epoch = 0ULL;
    records[a.rank] = record;
    __threadfence_system();
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(records[a.rank].epoch);
    published.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  production_terminal_debug_progress(a, phase, 20ULL);

#if __has_include(<nccl_device.h>)
  if (!StaticLocal && a.use_gin) {
    auto *gin_records = reinterpret_cast<ProductionStatusRecord *>(
        static_cast<std::byte *>(a.gin_window_base) + gin_record_offset);
    if (threadIdx.x == 0U) {
      auto record = local;
      record.epoch = 0ULL;
      gin_records[a.rank] = record;
      __threadfence_system();
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(gin_records[a.rank].epoch);
      published.store(a.epoch, ::cuda::memory_order_release);
      const ncclGin gin(a.gin_dev_comm, 0);
      const auto rail = ncclTeamRail(a.gin_dev_comm);
      production_terminal_debug_progress(a, phase, 30ULL);
      // Indexed signals are context-scoped, not operation-scoped.  In the
      // specialized Forward path signal 2 already contains the fixed-width
      // generation-ready handshake.  Its terminal threshold must therefore
      // include that protocol-known prefix.  It cannot be derived by reading
      // the live signal here: a faster peer may already have delivered this
      // terminal wave, which would make a sampled-base-plus-delta wait for an
      // increment that will never exist.
      //
      unsigned long long status_peers = 0ULL;
      const unsigned long long words[8] = {
          local.generation, local.phase,    local.origin_rank, local.error_code,
          local.field,      local.expected, local.actual,      local.epoch};
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == node)
          continue;
        const auto destination =
            gin_record_offset +
            static_cast<std::size_t>(a.rank) * sizeof(local);
        // Inline values remove the generic-store -> async-proxy source
        // dependency entirely.  The final epoch write carries the signal;
        // GIN orders it after all preceding puts to this context/peer, so the
        // receiver cannot observe completion before all eight words settle.
#pragma unroll
        for (std::uint32_t word = 0; word < 7U; ++word) {
          gin.putValue<unsigned long long>(
              rail, static_cast<int>(remote_node), a.gin_window,
              destination + word * sizeof(unsigned long long), words[word],
              ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
              ::cuda::thread_scope_thread, ::cuda::thread_scope_device);
          production_terminal_debug_progress(a, phase, 40ULL + word);
        }
        gin.putValue<unsigned long long>(
            rail, static_cast<int>(remote_node), a.gin_window,
            destination + 7U * sizeof(unsigned long long), words[7],
            ncclGin_SignalInc{2}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_thread, ::cuda::thread_scope_device);
        production_terminal_debug_progress(a, phase, 47ULL);
        ++status_peers;
      }
      if (status_peers != 0ULL) {
        const auto expected = prior_gin_signal_count + status_peers;
        production_terminal_debug_progress(
            a, phase,
            0x5000000000000000ULL | (expected << 32U) | gin.readSignal(2));
        gin.waitSignal(ncclCoopThread{}, 2, expected);
      }
      production_terminal_debug_progress(a, phase, 60ULL);
    }
    __syncthreads();
    // GIN writes arrive through Hopper's async proxy. Signal observation
    // orders the remote operation, but generic CTA loads of the received
    // record require an explicit proxy-domain fence. Without it the synthetic
    // epoch publication below could expose a current epoch with a stale
    // generation-zero payload.
    asm volatile("fence.proxy.async.global;" : : : "memory");
    __syncthreads();
    production_terminal_debug_progress(a, phase, 70ULL);
    // Each same-rail proxy receives the record for its corresponding remote
    // rank, then fans it out through already-owned node-local IPC mappings.
    const auto local_rank = a.rank - first_local;
    for (std::uint32_t remote_node = 0;
         remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
      if (remote_node == node)
        continue;
      const auto source = remote_node * local_world + local_rank;
      const auto remote = gin_records[source];
      for (std::uint32_t local_peer = threadIdx.x; local_peer < local_world;
           local_peer += blockDim.x) {
        const auto peer = first_local + local_peer;
        auto *records = peer_at<ProductionStatusRecord>(
            a.peer_bases[peer], ordinary_record_offset);
        auto record = remote;
        record.epoch = 0ULL;
        records[source] = record;
        __threadfence_system();
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(records[source].epoch);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
    }
    __syncthreads();
    production_terminal_debug_progress(a, phase, 80ULL);
  }
#else
  (void)gin_record_offset;
#endif

  auto *observed_records = peer_at<ProductionStatusRecord>(
      a.local_base, ordinary_record_offset);
  const auto observed_world = a.world;
  const auto observed_first = 0U;
  for (std::uint32_t offset = threadIdx.x; offset < observed_world;
       offset += blockDim.x) {
    const auto source = observed_first + offset;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        observed(observed_records[source].epoch);
    while (observed.load(::cuda::memory_order_acquire) < a.epoch) {
    }
  }
  __syncthreads();
  production_terminal_debug_progress(a, phase, 90ULL);

  // Local P2P and GIN both complete the fixed status agreement on device.
  // For GIN, every rail has already put its record cross-node and fanned the
  // corresponding remote record through node-local IPC, so the matrix below
  // is the same full-world matrix a redundant host NCCL AllGather would
  // produce. Validate and select the deterministic error here before the
  // status D2H; Socket returned above and retains its host collective.
  if (StaticLocal || a.use_gin) {
    if (threadIdx.x == 0U) {
      ProductionStatusRecord selected{};
      bool has_error = false;
      const auto expected_phase = static_cast<unsigned long long>(phase);
      for (std::uint32_t source = 0; source < observed_world; ++source) {
        auto record = observed_records[source];
        if (record.generation != a.dispatch.identity.generation) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          1ULL,
                                          a.dispatch.identity.generation,
                                          record.generation,
                                          a.epoch};
        } else if (record.phase != expected_phase) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          2ULL,
                                          expected_phase,
                                          record.phase,
                                          a.epoch};
        } else if (record.origin_rank != source && record.error_code == 0ULL) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          3ULL,
                                          source,
                                          record.origin_rank,
                                          a.epoch};
        } else if (record.epoch != a.epoch) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          4ULL,
                                          a.epoch,
                                          record.epoch,
                                          a.epoch};
        }
        if (record.error_code == 0ULL)
          continue;
        if (!has_error || record.origin_rank < selected.origin_rank ||
            (record.origin_rank == selected.origin_rank &&
             record.field < selected.field)) {
          selected = record;
          has_error = true;
        }
      }
      if (has_error) {
        status->error = 1U;
        status->error_origin_rank =
            static_cast<std::uint32_t>(selected.origin_rank);
        status->error_code = static_cast<std::uint32_t>(selected.error_code);
        status->error_field = static_cast<std::uint32_t>(selected.field);
        status->error_expected = selected.expected;
        status->error_actual = selected.actual;
      }
      __threadfence_system();
    }
    __syncthreads();
  }
  production_terminal_debug_progress(a, phase, 100ULL);
}

// The storage is owned by the local rank: ingress_published is written by
// remote Dispatch producers and outgoing_sequence is written by remote Return
// producers. Initialize only the local allocation, then publish setup-ready
// to every peer. No peer may issue a put until every owner has published the
// same epoch, which prevents initialization racing a remote release store.
__global__ void initialize_forward_generation_signals_kernel(
    void *local_base, void **peer_bases, std::size_t ingress_published_offset,
    std::size_t outgoing_sequence_offset, std::size_t setup_epoch_offset,
    std::size_t gin_local_compute_ready_offset,
    std::size_t forward_x_arrived_rows_offset, std::uint64_t mtile_count,
    std::size_t forward_completion_offset,
    std::size_t forward_context_completion_offset,
    std::uint32_t context_completion_capacity, std::uint64_t generation,
    std::uint32_t rank, std::uint32_t world, std::uint32_t channels,
    unsigned long long epoch, std::uint32_t first_peer,
    std::uint32_t peer_count, unsigned long long *runtime_epoch,
    const DeviceDispatchRuntime *runtime) {
  if (runtime_epoch) {
    if (threadIdx.x == 0U)
      atomicAdd(runtime_epoch, 1ULL);
    __syncthreads();
    epoch = *runtime_epoch;
  }
  if (runtime)
    mtile_count = runtime->sizes.mtile_count;
  const auto index = static_cast<std::uint32_t>(threadIdx.x);
  auto *incoming = peer_at<unsigned long long>(local_base,
                                                ingress_published_offset);
  auto *outgoing = peer_at<unsigned long long>(local_base,
                                                outgoing_sequence_offset);
  auto *compute_ready = peer_at<unsigned long long>(
      local_base, gin_local_compute_ready_offset);
  auto *x_arrived = peer_at<unsigned long long>(
      local_base, forward_x_arrived_rows_offset);
  auto *forward_completion = peer_at<transport_state::GenerationCompletionVector>(
      local_base, forward_completion_offset);
  auto *forward_context = peer_at<transport_state::ContextCompletionRecord>(
      local_base, forward_context_completion_offset);
  if (index == 0U) {
    peer_bases[world] = forward_completion;
    peer_bases[world + 1U] = forward_context;
  }
  const auto initial = epoch - 1ULL;
  for (std::uint32_t link = index; link < world * channels;
       link += blockDim.x) {
    incoming[link] = initial;
    outgoing[link] = initial;
  }
  for (std::uint32_t peer = index; peer < world; peer += blockDim.x)
    compute_ready[peer] = initial;
  const auto x_arrived_count = mtile_count * peer_count;
  for (std::uint64_t slot = index; slot < x_arrived_count; slot += blockDim.x) {
    x_arrived[slot] = 0ULL;
  }
  if (index == 0U) {
    *forward_completion = transport_state::GenerationCompletionVector{};
    forward_completion->generation = generation;
    forward_completion->epoch = epoch;
    forward_completion->first_waiting_tile = ~0ULL;
  }
  for (std::uint32_t context = index; context < context_completion_capacity;
       context += blockDim.x) {
    forward_context[context] = transport_state::ContextCompletionRecord{};
    forward_context[context].generation = generation;
    forward_context[context].epoch = epoch;
  }
  __syncthreads();
  __threadfence_system();
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    auto *ready = peer_at<unsigned long long>(peer_bases[peer],
                                               setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ready(ready[rank]);
    peer_ready.store(epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_ready = peer_at<unsigned long long>(local_base,
                                                   setup_epoch_offset);
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ready(local_ready[peer]);
    while (peer_ready.load(::cuda::memory_order_acquire) < epoch) {
    }
  }
  __syncthreads();
  // A second monotonic acknowledgement prevents a fast rank from returning
  // and initializing the next epoch before every slow rank has observed this
  // epoch's ready publication.
  __threadfence_system();
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    auto *ack = peer_at<unsigned long long>(peer_bases[peer],
                                             setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ack(ack[world + rank]);
    peer_ack.store(epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ack(local_ready[world + peer]);
    while (peer_ack.load(::cuda::memory_order_acquire) < epoch) {
    }
  }
}

struct LocalP2pKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  std::uint64_t hidden;
  std::uint64_t max_destination_rows;
  std::uint32_t rank;
  std::uint32_t world;
  std::uint32_t channels;
  std::uint32_t ring_slots;
  std::uint64_t epoch;
  void **peer_bases;
  void *local_base;
  std::size_t ring_headers_offset;
  std::size_t ring_payload_offset;
  std::size_t ring_ready_offset;
  std::size_t link_next_offset;
  std::size_t link_credits_offset;
  std::size_t ingress_headers_offset;
  std::size_t ingress_ready_offset;
  std::size_t ingress_next_offset;
  std::size_t ingress_published_offset;
  std::size_t ingress_credits_offset;
  std::size_t outgoing_sequence_offset;
  std::size_t direct_headers_offset;
  std::size_t return_ready_offset;
  std::size_t canonical_rows_offset;
  std::size_t contribution_offset;
  std::size_t combine_return_offset;
  std::size_t combined_offset;
  std::size_t status_offset;
  const unsigned long long *cancel_epoch;
  bool identity_expert;
};

__device__ bool local_p2p_cancelled(const LocalP2pKernelArgs &a) {
  return atomicAdd_system(
             const_cast<unsigned long long *>(a.cancel_epoch), 0ULL) ==
         a.dispatch.identity.generation;
}

__device__ bool acquire_system_credit(unsigned int *credit,
                                      TransportDeviceStatus *status,
                                      const LocalP2pKernelArgs &a) {
  while (!local_p2p_cancelled(a)) {
    const auto observed = atomicAdd_system(credit, 0U);
    if (observed != 0U &&
        atomicCAS_system(credit, observed, observed - 1U) == observed)
      return true;
    atomicAdd(&status->credit_stalls, 1ULL);
  }
  return false;
}

[[maybe_unused]] __global__ void
local_p2p_persistent_kernel(LocalP2pKernelArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *local_ring_headers = peer_at<transport::TransportSlotHeader>(
      a.local_base, a.ring_headers_offset);
  auto *local_ring_payload = peer_at<transport::BFloat16>(
      a.local_base, a.ring_payload_offset);
  auto *local_ring_ready =
      peer_at<unsigned long long>(a.local_base, a.ring_ready_offset);
  auto *local_link_next =
      peer_at<unsigned long long>(a.local_base, a.link_next_offset);
  auto *local_link_credits =
      peer_at<unsigned int>(a.local_base, a.link_credits_offset);
  auto *local_ingress_headers = peer_at<LocalP2pRouteHeader>(
      a.local_base, a.ingress_headers_offset);
  auto *local_ingress_ready =
      peer_at<unsigned long long>(a.local_base, a.ingress_ready_offset);
  auto *local_ingress_next =
      peer_at<unsigned long long>(a.local_base, a.ingress_next_offset);
  auto *local_ingress_published =
      peer_at<unsigned long long>(a.local_base, a.ingress_published_offset);
  auto *local_ingress_credits =
      peer_at<unsigned int>(a.local_base, a.ingress_credits_offset);
  auto *outgoing_sequence =
      peer_at<unsigned long long>(a.local_base, a.outgoing_sequence_offset);
  auto *local_return_ready =
      peer_at<unsigned long long>(a.local_base, a.return_ready_offset);
  auto *local_canonical = peer_at<transport::BFloat16>(
      a.local_base, a.canonical_rows_offset);
  auto *local_contribution = peer_at<transport::BFloat16>(
      a.local_base, a.contribution_offset);
  auto *local_combine_return = peer_at<transport::BFloat16>(
      a.local_base, a.combine_return_offset);
  auto *local_combined =
      peer_at<transport::BFloat16>(a.local_base, a.combined_offset);
  __shared__ unsigned long long shared_value;
  __shared__ unsigned long long stage_clock;
  __shared__ unsigned long long synthetic_phase_clock;

  if (lane == 0)
    atomicCAS(&status->start_clock, 0ULL, device_globaltimer_ns());
  __syncthreads();
  if (atomicAdd(&status->error, 0U) != 0U)
    return;

  // CTA 0 is the source Comm+proxy Relay worker. Packets and fan-out are
  // already in canonical descriptor order, so one elected producer owns every
  // sequence transition and no competing atomic-return order is observable.
  if (blockIdx.x == 0) {
    for (std::uint64_t packet = 0;
         packet < dispatch_sizes(a.dispatch).packet_count &&
         !local_p2p_cancelled(a);
         ++packet) {
      const auto packet_header = a.dispatch.packet_headers[packet];
      const auto link = static_cast<std::uint64_t>(packet_header.channel);
      if (packet_header.destination_node != 0 ||
          packet_header.channel >= a.channels ||
          packet_header.sequence_on_link != local_link_next[link]) {
        if (lane == 0) {
          status->error_packet = static_cast<unsigned int>(packet);
          status->error_field = 20U;
          atomicCAS(&status->error, 0U, 1U);
        }
        break;
      }
      if (lane == 0)
        shared_value = acquire_system_credit(&local_link_credits[link], status,
                                             a)
                           ? 1ULL
                           : 0ULL;
      __syncthreads();
      if (shared_value == 0)
        break;
      if (lane == 0) {
        const auto available = atomicAdd_system(&local_link_credits[link], 0U);
        atomicMax(&status->peak_ring_slots,
                  static_cast<unsigned long long>(a.ring_slots - available));
      }
      const auto physical =
          link * a.ring_slots + packet_header.sequence_on_link % a.ring_slots;
      if (lane == 0)
        stage_clock = device_globaltimer_ns();
      if (lane == 0) {
        local_ring_headers[physical] = transport::TransportSlotHeader{
            transport::kActivationTransportSchemaVersion,
            a.rank,
            0,
            0,
            packet_header.proxy_rank,
            packet_header.channel,
            a.dispatch.identity.generation,
            a.epoch,
            packet_header.sequence_on_link,
            packet_header.local_token_id,
            a.dispatch.packet_fanout_offsets[packet],
            a.dispatch.packet_fanout_offsets[packet + 1U]};
      }
      for (std::uint64_t hidden = lane; hidden < a.hidden;
           hidden += blockDim.x)
        local_ring_payload[physical * a.hidden + hidden] =
            a.input[packet_header.local_token_id * a.hidden + hidden];
      __syncthreads();
      if (lane == 0) {
        __threadfence_system();
        atomicExch_system(&local_ring_ready[physical],
                          packet_header.sequence_on_link + 1U);
        atomicAdd(&status->packets_done, 1ULL);
        atomicAdd(&status->dispatch_put_ticks,
                  device_globaltimer_ns() - stage_clock);
        stage_clock = device_globaltimer_ns();
      }
      __syncthreads();

      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      if (lane == 0) {
        atomicAdd(&status->deduplicated_cross_node_payload_bytes,
                  a.hidden * sizeof(transport::BFloat16));
        atomicAdd(&status->raw_cross_node_payload_bytes,
                  (fanout_end - fanout_begin) * a.hidden *
                      sizeof(transport::BFloat16));
      }
      for (auto fanout = fanout_begin;
           fanout < fanout_end && !local_p2p_cancelled(a); ++fanout) {
        const auto target = a.dispatch.packet_fanout[fanout];
        const auto route = packet_header.local_token_id *
                               a.dispatch.identity.top_k +
                           target.topk_slot;
        if (route >= dispatch_sizes(a.dispatch).route_count) {
          if (lane == 0) {
            status->error_packet = static_cast<unsigned int>(packet);
            status->error_field = 21U;
            atomicCAS(&status->error, 0U, 1U);
          }
          break;
        }
        const auto destination_rank = a.dispatch.route_dst_rank[route];
        if (destination_rank >= a.world ||
            target.destination_local_rank != destination_rank ||
            target.dst_row >= a.max_destination_rows) {
          if (lane == 0) {
            status->error_packet = static_cast<unsigned int>(packet);
            status->error_field = 21U;
            atomicCAS(&status->error, 0U, 1U);
          }
          break;
        }
        auto *destination_base = a.peer_bases[destination_rank];
        auto *destination_credit = peer_at<unsigned int>(
            destination_base, a.ingress_credits_offset);
        const auto ingress_link =
            static_cast<std::uint64_t>(a.rank) * a.channels +
            packet_header.channel;
        const auto no_wrap_batch = a.ring_slots >= a.max_destination_rows;
        if (lane == 0)
          shared_value =
              (no_wrap_batch || acquire_system_credit(
                                    &destination_credit[ingress_link], status, a))
                  ? outgoing_sequence[destination_rank * a.channels +
                                      packet_header.channel]
                  : ~0ULL;
        __syncthreads();
        if (shared_value == ~0ULL)
          break;
        if (lane == 0 && !no_wrap_batch) {
          const auto available =
              atomicAdd_system(&destination_credit[ingress_link], 0U);
          atomicMax(&status->peak_ring_slots,
                    static_cast<unsigned long long>(a.ring_slots - available));
        }
        const auto sequence = shared_value;
        const auto ingress_physical =
            ingress_link * a.ring_slots + sequence % a.ring_slots;
        auto *destination_canonical = peer_at<transport::BFloat16>(
            destination_base, a.canonical_rows_offset);
        for (std::uint64_t hidden = lane; hidden < a.hidden;
             hidden += blockDim.x)
          destination_canonical[target.dst_row * a.hidden + hidden] =
              local_ring_payload[physical * a.hidden + hidden];
        __syncthreads();
        if (lane == 0) {
          auto *destination_headers = peer_at<LocalP2pRouteHeader>(
              destination_base, a.ingress_headers_offset);
          destination_headers[ingress_physical] = LocalP2pRouteHeader{
              transport::kActivationTransportSchemaVersion,
              a.rank,
              destination_rank,
              packet_header.channel,
              a.dispatch.identity.generation,
              a.epoch,
              sequence,
              route,
              packet_header.local_token_id,
              target.dst_row,
              target.expert_id,
              target.topk_slot};
          if (!no_wrap_batch) {
            __threadfence_system();
            auto *destination_ready = peer_at<unsigned long long>(
                destination_base, a.ingress_ready_offset);
            atomicExch_system(&destination_ready[ingress_physical],
                              sequence + 1U);
          } else if ((sequence + 1U) % 64U == 0U) {
            __threadfence_system();
            auto *destination_published = peer_at<unsigned long long>(
                destination_base, a.ingress_published_offset);
            atomicExch_system(&destination_published[ingress_link],
                              sequence + 1U);
          }
          outgoing_sequence[destination_rank * a.channels +
                            packet_header.channel] = sequence + 1U;
          atomicAdd(&status->fanout_done, 1ULL);
        }
        __syncthreads();
      }
      if (lane == 0) {
        local_link_next[link] = packet_header.sequence_on_link + 1U;
        atomicAdd_system(&local_link_credits[link], 1U);
        atomicAdd(&status->relay_ticks,
                  device_globaltimer_ns() - stage_clock);
      }
      __syncthreads();
    }
    if (a.ring_slots >= a.max_destination_rows) {
      for (std::uint32_t destination = lane; destination < a.world;
           destination += blockDim.x) {
        const auto final_sequence =
            outgoing_sequence[destination * a.channels];
        if (final_sequence != 0U) {
          const auto ingress_link =
              static_cast<std::uint64_t>(a.rank) * a.channels;
          auto *destination_published = peer_at<unsigned long long>(
              a.peer_bases[destination], a.ingress_published_offset);
          __threadfence_system();
          atomicExch_system(&destination_published[ingress_link],
                            final_sequence);
        }
      }
    }
  }

  // CTA 1+source consumes the independent bounded ingress link from source.
  // It observes only the release sequence before reading the canonical row.
  if (blockIdx.x > 0 && blockIdx.x <= a.world) {
    const auto source = static_cast<std::uint32_t>(blockIdx.x - 1U);
    std::uint64_t expected = 0;
    if (lane == 0)
      shared_value = 0;
    __syncthreads();
    for (std::uint64_t row = lane;
         row < dispatch_sizes(a.dispatch).incoming_row_count; row += blockDim.x)
      if (a.dispatch.incoming_rows[row].source_rank == source)
        atomicAdd(&shared_value, 1ULL);
    __syncthreads();
    expected = shared_value;
    const auto ingress_link = static_cast<std::uint64_t>(source) * a.channels;
    if (lane == 0)
      synthetic_phase_clock = device_globaltimer_ns();
    __syncthreads();
    for (std::uint64_t sequence = 0;
         sequence < expected && !local_p2p_cancelled(a); ++sequence) {
      const auto physical =
          ingress_link * a.ring_slots + sequence % a.ring_slots;
      if (lane == 0) {
        if (a.ring_slots >= a.max_destination_rows) {
          while (atomicAdd_system(&local_ingress_published[ingress_link],
                                  0ULL) < sequence + 1U &&
                 !local_p2p_cancelled(a)) {
          }
        } else {
          while (atomicAdd_system(&local_ingress_ready[physical], 0ULL) !=
                     sequence + 1U &&
                 !local_p2p_cancelled(a)) {
          }
        }
        shared_value = local_p2p_cancelled(a) ? 0ULL : 1ULL;
      }
      __syncthreads();
      if (shared_value == 0)
        break;
      const auto header = local_ingress_headers[physical];
      if (lane == 0)
        stage_clock = device_globaltimer_ns();
      const auto incoming =
          header.dst_row < dispatch_sizes(a.dispatch).incoming_row_count
              ? a.dispatch.incoming_rows[header.dst_row]
              : DeviceIncomingRowRecord{};
      if (header.schema_version !=
              transport::kActivationTransportSchemaVersion ||
          header.source_rank != source || header.destination_rank != a.rank ||
          header.generation != a.dispatch.identity.generation ||
          header.sequence != sequence ||
          header.dst_row >= dispatch_sizes(a.dispatch).incoming_row_count ||
          incoming.source_rank != source ||
          incoming.expert_id != header.expert_id) {
        if (lane == 0) {
          status->error_packet = static_cast<unsigned int>(sequence);
          status->error_field = 22U;
          atomicCAS(&status->error, 0U, 1U);
        }
        break;
      }
      auto *source_base = a.peer_bases[source];
      auto *source_contribution = peer_at<transport::BFloat16>(
          source_base, a.contribution_offset);
      auto *source_combine = peer_at<transport::BFloat16>(
          source_base, a.combine_return_offset);
      for (std::uint64_t hidden = lane; hidden < a.hidden;
           hidden += blockDim.x) {
        const auto contribution = synthetic_value(
            local_canonical[header.dst_row * a.hidden + hidden],
            header.expert_id, hidden, a.identity_expert);
        source_contribution[header.route_index * a.hidden + hidden] =
            contribution;
        source_combine[header.route_index * a.hidden + hidden] = contribution;
      }
      __syncthreads();
      if (lane == 0) {
        __threadfence_system();
        if (a.ring_slots < a.max_destination_rows) {
          auto *source_ready = peer_at<unsigned long long>(
              source_base, a.return_ready_offset);
          atomicExch_system(&source_ready[header.route_index],
                            header.source_epoch);
          auto *source_status = peer_at<TransportDeviceStatus>(
              source_base, a.status_offset);
          atomicAdd_system(&source_status->synthetic_done, 1ULL);
          atomicAdd_system(&source_status->combine_done, 1ULL);
        }
        local_ingress_next[ingress_link] = sequence + 1U;
        if (a.ring_slots < a.max_destination_rows) {
          atomicExch_system(&local_ingress_ready[physical], 0ULL);
          atomicAdd_system(&local_ingress_credits[ingress_link], 1U);
        }
        atomicAdd(&status->incoming_done, 1ULL);
      }
      __syncthreads();
    }
    if (a.ring_slots >= a.max_destination_rows && expected != 0U && lane == 0) {
      __threadfence_system();
      auto *source_status = peer_at<TransportDeviceStatus>(
          a.peer_bases[source], a.status_offset);
      atomicAdd_system(&source_status->synthetic_done, expected);
      atomicAdd_system(&source_status->combine_done, expected);
    }
    if (lane == 0)
      atomicMax(&status->synthetic_ticks,
                device_globaltimer_ns() - synthetic_phase_clock);
  }

  while (!local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U &&
         (atomicAdd(&status->packets_done, 0ULL) <
              dispatch_sizes(a.dispatch).packet_count ||
          atomicAdd(&status->incoming_done, 0ULL) <
              dispatch_sizes(a.dispatch).incoming_row_count ||
          atomicAdd_system(&status->combine_done, 0ULL) <
              dispatch_sizes(a.dispatch).route_count)) {
  }
  if (local_p2p_cancelled(a) && lane == 0)
    atomicCAS(&status->error, 0U, 2U);
  if (lane == 0)
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
  __threadfence_system();

  while (!local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U) {
    if (lane == 0)
      shared_value = atomicAdd(&status->token_claim_cursor, 1ULL);
    __syncthreads();
    const auto token = shared_value;
    if (token >= a.dispatch.identity.num_tokens)
      break;
    for (std::uint64_t hidden = lane; hidden < a.hidden;
         hidden += blockDim.x) {
      float sum = 0.0F;
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto original = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + original;
        if (a.ring_slots < a.max_destination_rows)
          while (atomicAdd_system(&local_return_ready[route], 0ULL) != a.epoch &&
                 !local_p2p_cancelled(a)) {
          }
        sum = gate_after_accumulate(
            sum, a.gates[route],
            local_combine_return[route * a.hidden + hidden].bits);
      }
      local_combined[token * a.hidden + hidden] =
          transport::BFloat16{float_to_bf16_bits(sum)};
    }
    __syncthreads();
    if (lane == 0)
      atomicAdd(&status->reduced_tokens, 1ULL);
  }
  if (lane == 0)
    atomicMax(&status->complete_clock, device_globaltimer_ns());
}

// Trusted-team release path. Notify has already assigned a unique canonical
// row to every route. Separate all-SM grids bulk-copy each phase, and tiny
// kernels publish one system-scope completion per peer rather than per route.
#if !WEFTMOE_DEBUG_VALIDATE
__global__ void local_p2p_bulk_dispatch_kernel(LocalP2pKernelArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x);
  if (route >= dispatch_sizes(a.dispatch).route_count)
    return;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (route == 0 && threadIdx.x == 0)
    status->start_clock = device_globaltimer_ns();
  const auto destination = a.dispatch.route_dst_rank[route];
  const auto row = a.dispatch.route_dst_row[route];
  if (destination >= a.world || row >= a.max_destination_rows) {
    if (threadIdx.x == 0) {
      status->error_packet = static_cast<unsigned int>(route);
      status->error_field = 30U;
      atomicCAS(&status->error, 0U, 1U);
    }
    return;
  }
  const auto token = route / a.dispatch.identity.top_k;
  auto *destination_base = a.peer_bases[destination];
  auto *destination_rows = peer_at<transport::BFloat16>(
      destination_base, a.canonical_rows_offset);
  const auto *source = a.input + token * a.hidden;
  auto *target = destination_rows + row * a.hidden;
  if ((a.hidden & 7ULL) == 0ULL) {
    const auto *source128 = reinterpret_cast<const uint4 *>(source);
    auto *target128 = reinterpret_cast<uint4 *>(target);
    const auto vector_count = a.hidden / 8ULL;
    for (std::uint64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x * 2ULL) {
      target128[vector] = source128[vector];
      const auto second = vector + blockDim.x;
      if (second < vector_count)
        target128[second] = source128[second];
    }
  } else {
    for (std::uint64_t hidden = threadIdx.x; hidden < a.hidden;
         hidden += blockDim.x)
      target[hidden] = source[hidden];
  }
  if (threadIdx.x == 0) {
    auto *destination_headers = peer_at<LocalP2pRouteHeader>(
        destination_base, a.direct_headers_offset);
    destination_headers[row] = LocalP2pRouteHeader{
        transport::kActivationTransportSchemaVersion,
        a.rank,
        destination,
        static_cast<std::uint32_t>(route % a.channels),
        a.dispatch.identity.generation,
        a.epoch,
        route,
        route,
        token,
        row,
        a.dispatch.expert_id[route],
        static_cast<std::uint32_t>(route % a.dispatch.identity.top_k)};
    if (destination != a.rank)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                a.hidden * sizeof(transport::BFloat16));
  }
  __threadfence_system();
}

__global__ void local_p2p_publish_dispatch_kernel(LocalP2pKernelArgs a) {
  const auto destination = static_cast<std::uint32_t>(threadIdx.x);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (destination < a.world) {
    auto *signal = peer_at<unsigned long long>(
        a.peer_bases[destination], a.ingress_published_offset);
    __threadfence_system();
    atomicExch_system(&signal[a.rank * a.channels], a.epoch);
  }
  if (threadIdx.x == 0) {
    status->packets_done = dispatch_sizes(a.dispatch).packet_count;
    status->fanout_done = dispatch_sizes(a.dispatch).route_count;
    status->raw_cross_node_payload_bytes =
        dispatch_sizes(a.dispatch).route_count * a.hidden *
        sizeof(transport::BFloat16);
    status->deduplicated_cross_node_payload_bytes =
        dispatch_sizes(a.dispatch).packet_count * a.hidden *
        sizeof(transport::BFloat16);
    status->dispatch_put_ticks = device_globaltimer_ns() - status->start_clock;
  }
}

__global__ void local_p2p_wait_dispatch_kernel(LocalP2pKernelArgs a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  for (std::uint32_t source = threadIdx.x; source < a.world;
       source += blockDim.x) {
    auto *signals = peer_at<unsigned long long>(
        a.local_base, a.ingress_published_offset);
    while (atomicAdd_system(&signals[source * a.channels], 0ULL) != a.epoch &&
           !local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (local_p2p_cancelled(a))
      atomicCAS(&status->error, 0U, 2U);
    __threadfence_system();
    status->dispatch_complete_clock = device_globaltimer_ns();
  }
}

__global__ void local_p2p_bulk_return_kernel(LocalP2pKernelArgs a) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= dispatch_sizes(a.dispatch).incoming_row_count)
    return;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (atomicAdd(&status->error, 0U) != 0U)
    return;
  auto *local_headers = peer_at<LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  auto *local_canonical = peer_at<transport::BFloat16>(
      a.local_base, a.canonical_rows_offset);
  const auto header = local_headers[row];
  auto *source_contribution = peer_at<transport::BFloat16>(
      a.peer_bases[header.source_rank], a.contribution_offset);
  for (std::uint64_t hidden = threadIdx.x; hidden < a.hidden;
       hidden += blockDim.x)
    source_contribution[header.route_index * a.hidden + hidden] =
        synthetic_value(local_canonical[row * a.hidden + hidden],
                        header.expert_id, hidden, a.identity_expert);
  if (threadIdx.x == 0 && header.source_rank != a.rank)
    atomicAdd(&status->actual_return_peer_bytes,
              a.hidden * sizeof(transport::BFloat16));
  __threadfence_system();
}

__global__ void local_p2p_publish_return_kernel(LocalP2pKernelArgs a) {
  const auto source = static_cast<std::uint32_t>(threadIdx.x);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (source < a.world) {
    auto *signal = peer_at<unsigned long long>(
        a.peer_bases[source], a.outgoing_sequence_offset);
    __threadfence_system();
    atomicExch_system(&signal[a.rank * a.channels], a.epoch);
  }
  if (threadIdx.x == 0)
    status->incoming_done = dispatch_sizes(a.dispatch).incoming_row_count;
}

__global__ void local_p2p_wait_return_kernel(LocalP2pKernelArgs a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  for (std::uint32_t destination = threadIdx.x; destination < a.world;
       destination += blockDim.x) {
    auto *signals = peer_at<unsigned long long>(
        a.local_base, a.outgoing_sequence_offset);
    while (atomicAdd_system(&signals[destination * a.channels], 0ULL) !=
               a.epoch &&
           !local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (local_p2p_cancelled(a))
      atomicCAS(&status->error, 0U, 2U);
    __threadfence_system();
    status->synthetic_ticks =
        device_globaltimer_ns() - status->dispatch_complete_clock;
    status->relay_ticks = status->synthetic_ticks;
    status->synthetic_done = dispatch_sizes(a.dispatch).route_count;
    status->combine_done = dispatch_sizes(a.dispatch).route_count;
  }
}

__global__ void local_p2p_bulk_reduce_kernel(LocalP2pKernelArgs a) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= a.dispatch.identity.num_tokens)
    return;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (atomicAdd(&status->error, 0U) != 0U)
    return;
  auto *contribution = peer_at<transport::BFloat16>(
      a.local_base, a.contribution_offset);
  auto *combined = peer_at<transport::BFloat16>(
      a.local_base, a.combined_offset);
  for (std::uint64_t hidden = threadIdx.x; hidden < a.hidden;
       hidden += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t order = 0; order < a.dispatch.identity.top_k; ++order) {
      const auto original = a.dispatch.combine_order[
          token * a.dispatch.identity.top_k + order];
      const auto route = token * a.dispatch.identity.top_k + original;
      sum = gate_after_accumulate(
          sum, a.gates[route],
          contribution[route * a.hidden + hidden].bits);
    }
    combined[token * a.hidden + hidden] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
  if (threadIdx.x == 0)
    atomicAdd(&status->reduced_tokens, 1ULL);
}

__global__ void local_p2p_finish_kernel(LocalP2pKernelArgs a) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    status->complete_clock = device_globaltimer_ns();
  }
}
#endif
