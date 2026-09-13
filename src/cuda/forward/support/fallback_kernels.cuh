__device__ std::uint16_t float_to_bf16_bits(float value) {
  auto bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ std::uint32_t float2_to_bf16x2_bits(float low,
                                                               float high) {
  const auto pair = static_cast<__nv_bfloat162_raw>(
      __float22bfloat162_rn(make_float2(low, high)));
  return static_cast<std::uint32_t>(pair.x) |
         (static_cast<std::uint32_t>(pair.y) << 16U);
}

__device__ __forceinline__ float2
bf16x2_bits_to_float2(std::uint32_t pair) {
  // A promoted BF16 is its 16 bits in the high half of an FP32 word. Decode
  // both packed lanes together so the FC1 epilogue needs only one shift and
  // one mask per pair instead of separately extracting two uint16_t values.
  return make_float2(__uint_as_float(pair << 16U),
                     __uint_as_float(pair & 0xffff0000U));
}

__device__ float bf16_bits_to_float(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ float gate_after_accumulate(float sum, float gate,
                                       std::uint16_t bf16_value) {
  const float product = __fmul_rn(gate, bf16_bits_to_float(bf16_value));
  return __fadd_rn(sum, product);
}

__device__ unsigned long long device_globaltimer_ns() {
  unsigned long long value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

template <typename Status>
__device__ void publish_local_structured_error(
    Status *status, std::uint32_t rank, std::uint32_t field,
    std::uint32_t code, unsigned long long expected,
    unsigned long long actual) {
  if (atomicCAS(&status->error, 0U, 1U) != 0U)
    return;
  status->error_origin_rank = rank;
  status->error_field = field;
  status->error_code = code;
  status->error_expected = expected;
  status->error_actual = actual;
  __threadfence_system();
}

__device__ transport::BFloat16 synthetic_value(
    transport::BFloat16 input, std::uint32_t expert,
    std::uint64_t hidden_index, bool identity) {
  if (identity)
    return input;
  const auto expert_term = static_cast<float>((expert % 251U) + 1U) / 128.0F;
  const auto lane_term = static_cast<float>(hidden_index % 17U) / 1024.0F;
  return transport::BFloat16{
      float_to_bf16_bits(bf16_bits_to_float(input.bits) + expert_term +
                         lane_term)};
}

struct TransportKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  std::uint64_t hidden;
  std::uint64_t max_destination_rows;
  std::uint32_t channels;
  std::uint32_t ring_slots;
  transport::TransportSlotHeader *ring_headers;
  transport::BFloat16 *ring_payload;
  unsigned long long *ring_ready;
  unsigned long long *link_next;
  unsigned int *link_credits;
  unsigned int *packet_claim;
  transport::BFloat16 *canonical_rows;
  transport::BFloat16 *contribution;
  transport::BFloat16 *combine_return;
  transport::BFloat16 *combined;
  TransportDeviceStatus *status;
  bool identity_expert;
};

__global__ void activation_transport_persistent_kernel(TransportKernelArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto route_count = dispatch_sizes(a.dispatch).route_count;
  const auto packet_count = dispatch_sizes(a.dispatch).packet_count;
  __shared__ unsigned long long selected_packet;
  __shared__ unsigned long long selected_token;
  __shared__ unsigned long long packet_scan_cursor;
  if (lane == 0)
    atomicCAS(&a.status->start_clock, 0ULL,
              device_globaltimer_ns());
  __syncthreads();

  // Rule-based link ownership: link L belongs to CTA L % gridDim.x. Each CTA
  // scans the canonical descriptor array once and processes every assigned
  // link in sequence order. This is bounded O(grid*packets), has no ready
  // queue, and does not use atomic return order to assign a packet or row.
  if (lane == 0)
    packet_scan_cursor = 0;
  __syncthreads();
  while (atomicAdd(&a.status->error, 0U) == 0U) {
    if (lane == 0) {
      selected_packet = ~0ULL;
      while (packet_scan_cursor < packet_count) {
        const auto candidate = packet_scan_cursor++;
        const auto candidate_header = a.dispatch.packet_headers[candidate];
        const auto candidate_link =
            static_cast<unsigned long long>(candidate_header.destination_node) *
                a.channels +
            candidate_header.channel;
        if (candidate_link % gridDim.x != blockIdx.x)
          continue;
        if (candidate_header.sequence_on_link !=
            atomicAdd(&a.link_next[candidate_link], 0ULL)) {
          a.status->error_packet = static_cast<unsigned int>(candidate);
          a.status->error_field = 2U;
          atomicExch(&a.status->error, 1U);
          break;
        }
        if (atomicCAS(&a.packet_claim[candidate], 0U, 1U) == 0U) {
          atomicAdd(&a.status->packet_claim_cursor, 1ULL);
          selected_packet = candidate;
          break;
        }
      }
    }
    __syncthreads();
    if (selected_packet == ~0ULL)
      break;
    const auto packet = selected_packet;
    const auto header = a.dispatch.packet_headers[packet];
    const auto link = static_cast<unsigned long long>(header.destination_node) *
                          a.channels +
                      header.channel;
    const auto physical =
        link * a.ring_slots + header.sequence_on_link % a.ring_slots;
    const auto gpus_per_node =
        a.dispatch.identity.world_size / a.dispatch.identity.num_nodes;
    const auto source_node =
        a.dispatch.identity.calling_rank / gpus_per_node;
    if (lane == 0) {
      const auto prior = atomicSub(&a.link_credits[link], 1U);
      if (prior == 0U) {
        atomicAdd(&a.link_credits[link], 1U);
        atomicAdd(&a.status->credit_stalls, 1ULL);
        a.status->error_packet = static_cast<unsigned int>(packet);
        a.status->error_field = 3U;
        atomicExch(&a.status->error, 1U);
      }
    }
    __syncthreads();
    if (atomicAdd(&a.status->error, 0U) != 0U)
      break;
    if (lane == 0) {
      a.ring_headers[physical] = transport::TransportSlotHeader{
          transport::kActivationTransportSchemaVersion,
          header.source_rank,
          source_node,
          header.destination_node,
          header.proxy_rank,
          header.channel,
          a.dispatch.identity.generation,
          a.dispatch.identity.generation,
          header.sequence_on_link,
          header.local_token_id,
          a.dispatch.packet_fanout_offsets[packet],
          a.dispatch.packet_fanout_offsets[packet + 1U]};
    }
    for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
      a.ring_payload[physical * a.hidden + h] =
          a.input[header.local_token_id * a.hidden + h];
    }
    __syncthreads();
    if (lane == 0) {
      __threadfence_system();
      atomicExch_system(&a.ring_ready[physical],
                        static_cast<unsigned long long>(
                            header.sequence_on_link + 1U));
      atomicAdd(&a.status->packets_done, 1ULL);
    }
    __syncthreads();
    while (atomicAdd_system(&a.ring_ready[physical], 0ULL) !=
           header.sequence_on_link + 1U) {
    }
    __threadfence_system();
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
    if (lane == 0 && header.destination_node != source_node) {
      atomicAdd(&a.status->deduplicated_cross_node_payload_bytes,
                static_cast<unsigned long long>(
                    a.hidden * sizeof(transport::BFloat16)));
      atomicAdd(&a.status->raw_cross_node_payload_bytes,
                static_cast<unsigned long long>(
                    (fanout_end - fanout_begin) * a.hidden *
                    sizeof(transport::BFloat16)));
    }
    for (auto fanout = fanout_begin; fanout < fanout_end; ++fanout) {
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = header.local_token_id * a.dispatch.identity.top_k +
                         target.topk_slot;
      if (route >= route_count) {
        if (lane == 0) {
          a.status->error_packet = static_cast<unsigned int>(packet);
          a.status->error_field = 1U;
          atomicExch(&a.status->error, 1U);
        }
        break;
      }
      for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
        const auto payload = a.ring_payload[physical * a.hidden + h];
        if (a.dispatch.route_dst_rank[route] ==
                a.dispatch.identity.calling_rank &&
            target.dst_row < a.max_destination_rows)
          a.canonical_rows[target.dst_row * a.hidden + h] = payload;
        const auto contribution = synthetic_value(
            payload, target.expert_id, h, a.identity_expert);
        a.contribution[route * a.hidden + h] = contribution;
        a.combine_return[route * a.hidden + h] = contribution;
      }
      if (lane == 0) {
        atomicAdd(&a.status->fanout_done, 1ULL);
        atomicAdd(&a.status->synthetic_done, 1ULL);
        atomicAdd(&a.status->combine_done, 1ULL);
      }
      __syncthreads();
    }
    if (lane == 0) {
      __threadfence_system();
      atomicAdd(&a.link_next[link], 1ULL);
      atomicAdd(&a.link_credits[link], 1U);
    }
    __syncthreads();
  }

  // Comm/Relay CTAs transition to the deterministic TopKReduce role only
  // after global packet/fan-out completion counters prove the boundary.
  while ((atomicAdd(&a.status->packets_done, 0ULL) < packet_count ||
          atomicAdd(&a.status->combine_done, 0ULL) < route_count) &&
         atomicAdd(&a.status->error, 0U) == 0U) {
  }
  if (lane == 0)
    atomicMax(&a.status->dispatch_complete_clock,
              device_globaltimer_ns());
  __threadfence();
  while (atomicAdd(&a.status->error, 0U) == 0U) {
    if (lane == 0)
      selected_token = atomicAdd(&a.status->token_claim_cursor, 1ULL);
    __syncthreads();
    const auto token = selected_token;
    if (token >= a.dispatch.identity.num_tokens)
      break;
    for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
      float sum = 0.0F;
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto original = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + original;
        sum = gate_after_accumulate(
            sum, a.gates[route],
            a.combine_return[route * a.hidden + h].bits);
      }
      a.combined[token * a.hidden + h] =
          transport::BFloat16{float_to_bf16_bits(sum)};
    }
    __syncthreads();
    if (lane == 0)
      atomicAdd(&a.status->reduced_tokens, 1ULL);
  }
  if (lane == 0)
    atomicMax(&a.status->complete_clock,
              device_globaltimer_ns());
}

__global__ void socket_route_count_kernel(DeviceDispatchKernelView dispatch,
                                          unsigned long long *counts) {
  for (std::uint64_t route =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < dispatch.sizes.route_count;
       route += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    const auto destination = dispatch.route_dst_rank[route];
    if (destination < dispatch.identity.world_size)
      atomicAdd(&counts[destination], 1ULL);
  }
}

__global__ void socket_route_prefix_kernel(const unsigned long long *counts,
                                           unsigned long long *prefix,
                                           std::uint32_t world) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  unsigned long long total = 0;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    prefix[rank] = total;
    total += counts[rank];
  }
  prefix[world] = total;
}

__global__ void socket_stable_route_ids_kernel(
    DeviceDispatchKernelView dispatch, const unsigned long long *prefix,
    unsigned int *route_ids) {
  const auto destination = static_cast<std::uint32_t>(blockIdx.x);
  if (destination >= dispatch.identity.world_size)
    return;
  __shared__ unsigned int scan[256];
  __shared__ unsigned long long carry;
  if (threadIdx.x == 0)
    carry = 0;
  __syncthreads();
  for (std::uint64_t base = 0; base < dispatch.sizes.route_count;
       base += blockDim.x) {
    const auto route = base + threadIdx.x;
    scan[threadIdx.x] =
        route < dispatch.sizes.route_count &&
                dispatch.route_dst_rank[route] == destination
            ? 1U
            : 0U;
    __syncthreads();
    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1U) {
      const auto add = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0U;
      __syncthreads();
      scan[threadIdx.x] += add;
      __syncthreads();
    }
    if (route < dispatch.sizes.route_count &&
        dispatch.route_dst_rank[route] == destination) {
      const auto position = prefix[destination] + carry +
                            static_cast<unsigned long long>(scan[threadIdx.x] - 1U);
      route_ids[position] = static_cast<unsigned int>(route);
    }
    __syncthreads();
    if (threadIdx.x == 0)
      carry += scan[blockDim.x - 1U];
    __syncthreads();
  }
}

__global__ void socket_pack_routes_kernel(
    DeviceDispatchKernelView dispatch, const transport::BFloat16 *input,
    const float *gates, std::uint64_t hidden, std::uint64_t epoch,
    const unsigned int *route_ids, SocketRouteHeader *headers,
    transport::BFloat16 *payload) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= dispatch.sizes.route_count)
    return;
  const auto route = static_cast<std::uint64_t>(route_ids[packed]);
  const auto token = route / dispatch.identity.top_k;
  const auto slot = static_cast<std::uint32_t>(route % dispatch.identity.top_k);
  if (threadIdx.x == 0) {
    headers[packed] = SocketRouteHeader{
        LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                            dispatch.identity.calling_rank,
                            dispatch.route_dst_rank[route], 0U,
                            dispatch.identity.generation, epoch, packed, route,
                            token, dispatch.route_dst_row[route],
                            dispatch.expert_id[route], slot},
        gates[route], 0U};
  }
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    payload[packed * hidden + h] = input[token * hidden + h];
}

__global__ void socket_relay_transport_kernel(
    SocketRouteHeader *headers, transport::BFloat16 *compact_payload,
    transport::BFloat16 *canonical_rows, float *canonical_gates,
    std::uint64_t incoming, std::uint64_t hidden, std::uint64_t generation,
    std::uint32_t destination_rank, std::uint64_t max_rows,
    bool identity_expert, TransportDeviceStatus *status) {
  const auto incoming_index = static_cast<std::uint64_t>(blockIdx.x);
  if (incoming_index >= incoming)
    return;
  const auto header = headers[incoming_index];
  const auto route = header.route;
  const bool valid =
      route.schema_version == transport::kActivationTransportSchemaVersion &&
      route.generation == generation &&
      route.destination_rank == destination_rank && route.dst_row < max_rows;
  if (!valid) {
    if (threadIdx.x == 0)
      publish_local_structured_error(status, destination_rank, 40U, 1U,
                                     max_rows, route.dst_row);
    return;
  }
  if (threadIdx.x == 0)
    canonical_gates[route.dst_row] = header.gate;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    const auto value = compact_payload[incoming_index * hidden + h];
    canonical_rows[route.dst_row * hidden + h] = value;
    compact_payload[incoming_index * hidden + h] =
        synthetic_value(value, route.expert_id, h, identity_expert);
  }
}



__global__ void socket_scatter_returns_kernel(
    const SocketRouteHeader *headers,
    const transport::BFloat16 *compact,
    transport::BFloat16 *by_route, std::uint64_t routes,
    std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= routes)
    return;
  const auto route = headers[packed].route.route_index;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    by_route[route * hidden + h] = compact[packed * hidden + h];
}

__global__ void socket_gate_reduce_kernel(
    DeviceDispatchKernelView dispatch, const float *gates,
    const transport::BFloat16 *by_route, transport::BFloat16 *combined,
    std::uint64_t hidden) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= dispatch.identity.num_tokens)
    return;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t order = 0; order < dispatch.identity.top_k; ++order) {
      const auto original = dispatch.combine_order[
          token * dispatch.identity.top_k + order];
      const auto route = token * dispatch.identity.top_k + original;
      sum = gate_after_accumulate(sum, gates[route],
                                  by_route[route * hidden + h].bits);
    }
    combined[token * hidden + h] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
}


__global__ void socket_finalize_status_kernel(TransportDeviceStatus *status,
                                              std::uint64_t routes,
                                              std::uint64_t incoming,
                                              std::uint64_t tokens) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  status->packets_done = routes;
  status->fanout_done = incoming;
  status->synthetic_done = incoming;
  status->combine_done = routes;
  status->incoming_done = incoming;
  status->reduced_tokens = tokens;
  status->complete_clock = device_globaltimer_ns();
}

#if __has_include(<nccl_device.h>)
__global__ void gin_probe_reset_kernel(ncclDevComm_t dev_comm) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const ncclGin gin(dev_comm, 0);
    gin.resetSignal(0);
    gin.resetCounter(0);
  }
}

__global__ void gin_probe_put_signal_kernel(
    ncclDevComm_t dev_comm, ncclWindow_t window, std::size_t payload_bytes,
    std::size_t receive_offset, unsigned int *device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const ncclGin gin(dev_comm, 0);
  const auto rail = ncclTeamRail(dev_comm);
  if (rail.nRanks < 2) {
    *device_error = 1U;
    return;
  }
  const auto destination = (rail.rank + 1) % rail.nRanks;
  gin.put(rail, destination, window, receive_offset, window, 0,
          payload_bytes, ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
          ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_thread,
          ::cuda::thread_scope_device);
  // Counter completion proves the local source region can be reused. The
  // remote indexed signal is ordered after the payload has settled.
  gin.waitCounter(ncclCoopThread{}, 0, 1);
  gin.waitSignal(ncclCoopThread{}, 0, 1);
}

__global__ void gin_link_benchmark_kernel(
    ncclDevComm_t dev_comm, ncclWindow_t window, bool full_connection,
    bool fanout_full, int destination, int local_world,
    std::size_t payload_bytes, std::size_t receive_offset,
    std::uint32_t warmup, std::uint32_t measured,
    std::uint32_t puts_per_sample, unsigned long long *duration_ns,
    unsigned int *device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const ncclGin gin(dev_comm, 0);
  const ncclTeam team = full_connection ? ncclTeamWorld(dev_comm)
                                        : ncclTeamRail(dev_comm);
  if (destination < 0 || destination >= team.nRanks || measured == 0U ||
      puts_per_sample == 0U) {
    *device_error = 1U;
    return;
  }

  unsigned long long expected = 0ULL;
  const auto samples = static_cast<unsigned long long>(warmup) + measured;
  for (unsigned long long sample = 0ULL; sample < samples; ++sample) {
    const auto begin = device_globaltimer_ns();
    for (std::uint32_t put = 0; put < puts_per_sample; ++put) {
      auto put_destination = destination;
      if (fanout_full) {
        const auto node = destination / local_world;
        const auto local = destination % local_world;
        put_destination =
            node * local_world + (local + static_cast<int>(put)) % local_world;
      }
      gin.put(team, put_destination, window, receive_offset, window, 0,
              payload_bytes, ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
              ncclCoopThread{}, ncclGin_None{},
              ::cuda::thread_scope_system, ::cuda::thread_scope_system);
    }
    expected += puts_per_sample;
    gin.waitCounter(ncclCoopThread{}, 0, expected, 64,
                    ::cuda::memory_order_acquire);
    const auto end = device_globaltimer_ns();
    if (sample >= warmup)
      duration_ns[sample - warmup] = end - begin;
  }
  // Every rank has exactly one inverse sender.  Waiting for the matching
  // remote signal makes the final payload validation independent of whether
  // local completion happens a little earlier than the receive-side action.
  gin.waitSignal(ncclCoopThread{}, 0, expected, 64,
                 ::cuda::memory_order_acquire);
}

struct GinTransportSliceArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  std::uint64_t hidden;
  std::uint64_t epoch;
  ncclDevComm_t dev_comm;
  ncclWindow_t window;
  void *compute_base;
  void *window_base;
  std::size_t send_header_offset;
  std::size_t receive_header_offset;
  std::size_t send_fanout_offset;
  std::size_t receive_fanout_offset;
  std::size_t send_payload_offset;
  std::size_t receive_payload_offset;
  std::size_t canonical_offset;
  std::size_t return_staging_offset;
  std::size_t return_receive_offset;
  std::size_t combine_compute_offset;
  std::size_t combined_offset;
  std::size_t status_offset;
  std::size_t peer_bases_offset;
  std::uint32_t local_world_size;
  std::uint32_t node;
  std::uint64_t max_tokens_per_rank;
  bool identity_expert;
};

__global__ void gin_signal_reset_kernel(ncclDevComm_t dev_comm,
                                           std::uint32_t context_count,
                                           std::uint32_t signal_count = 3U) {
  if (blockIdx.x < context_count && threadIdx.x == 0) {
    const ncclGin gin(dev_comm, blockIdx.x);
    for (std::uint32_t signal = 0U; signal < signal_count; ++signal)
      gin.resetSignal(signal);
    gin.resetCounter(0);
    gin.resetCounter(1);
  }
}

__global__ void gin_transport_slice_kernel(GinTransportSliceArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto rank = a.dispatch.identity.calling_rank;
  const ncclGin gin(a.dev_comm, 0);
  const auto rail = ncclTeamRail(a.dev_comm);
  auto *bytes = static_cast<std::byte *>(a.window_base);
  auto *compute_bytes = static_cast<std::byte *>(a.compute_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.send_header_offset);
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.receive_header_offset);
  auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.send_fanout_offset);
  auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.receive_fanout_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.send_payload_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.receive_payload_offset);
  auto *return_staging = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.return_staging_offset);
  auto *return_receive = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.return_receive_offset);
  auto *combine_compute = reinterpret_cast<transport::BFloat16 *>(
      compute_bytes + a.combine_compute_offset);
  auto *combined = reinterpret_cast<transport::BFloat16 *>(
      compute_bytes + a.combined_offset);
  auto *status = reinterpret_cast<TransportDeviceStatus *>(
      compute_bytes + a.status_offset);
  auto **peer_bases = reinterpret_cast<void **>(
      compute_bytes + a.peer_bases_offset);
  __shared__ unsigned long long remote_packets;
  __shared__ unsigned long long remote_routes;
  __shared__ unsigned long long remote_incoming;
  if (lane == 0) {
    remote_packets = 0;
    remote_routes = 0;
    remote_incoming = 0;
    status->start_clock = device_globaltimer_ns();
  }
  __syncthreads();

  // Notify packets are already in deterministic (destination node, token)
  // order. One packet carries one token payload and all expert fan-out rows.
  for (std::uint64_t packet = 0;
       packet < dispatch_sizes(a.dispatch).packet_count; ++packet) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
    const bool local = descriptor.destination_node == a.node;
    if (!local) {
      for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x)
        send_payload[packet * a.hidden + h] =
            a.input[descriptor.local_token_id * a.hidden + h];
      for (auto fanout = fanout_begin + lane; fanout < fanout_end;
           fanout += blockDim.x) {
        send_fanout[fanout] = a.dispatch.packet_fanout[fanout];
      }
      if (lane == 0)
        send_headers[packet] = LocalP2pRouteHeader{
            transport::kActivationTransportSchemaVersion,
            rank,
            descriptor.proxy_rank,
            descriptor.channel,
            a.dispatch.identity.generation,
            a.epoch,
            descriptor.sequence_on_link,
            fanout_end - fanout_begin,
            descriptor.local_token_id,
            fanout_begin,
            0,
            0};
      __syncthreads();
      if (lane == 0) {
        const auto destination =
            static_cast<int>(descriptor.destination_node);
        const auto source_partition =
            a.node < descriptor.destination_node ? a.node : a.node - 1U;
        const auto receive_packet =
            source_partition * a.max_tokens_per_rank +
            descriptor.sequence_on_link;
        gin.put(rail, destination, a.window,
                a.receive_header_offset + receive_packet *
                                              sizeof(LocalP2pRouteHeader),
                a.window,
                a.send_header_offset + packet * sizeof(LocalP2pRouteHeader),
                sizeof(LocalP2pRouteHeader));
        gin.put(rail, destination, a.window,
                a.receive_fanout_offset +
                    receive_packet *
                        a.dispatch.identity.top_k * sizeof(DeviceFanoutRecord),
                a.window,
                a.send_fanout_offset +
                    fanout_begin * sizeof(DeviceFanoutRecord),
                (fanout_end - fanout_begin) * sizeof(DeviceFanoutRecord));
        gin.put(rail, destination, a.window,
                a.receive_payload_offset +
                    receive_packet * a.hidden *
                        sizeof(transport::BFloat16),
                a.window,
                a.send_payload_offset + packet * a.hidden *
                                            sizeof(transport::BFloat16),
                a.hidden * sizeof(transport::BFloat16),
                ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
                ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++remote_packets;
        remote_routes += fanout_end - fanout_begin;
      }
    } else {
      for (auto fanout = fanout_begin; fanout < fanout_end; ++fanout) {
        const auto target = a.dispatch.packet_fanout[fanout];
        const auto route = descriptor.local_token_id *
                               a.dispatch.identity.top_k +
                           target.topk_slot;
        for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
          const auto value = a.input[descriptor.local_token_id * a.hidden + h];
          const auto destination_rank =
              a.node * a.local_world_size + target.destination_local_rank;
          auto *destination_canonical =
              reinterpret_cast<transport::BFloat16 *>(
                  static_cast<std::byte *>(peer_bases[destination_rank]) +
                  a.canonical_offset);
          destination_canonical[target.dst_row * a.hidden + h] = value;
          combine_compute[route * a.hidden + h] = synthetic_value(
              value, target.expert_id, h, a.identity_expert);
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }
  if (lane == 0 && remote_packets != 0)
    gin.waitCounter(ncclCoopThread{}, 0, remote_packets);
  __syncthreads();

  if (lane == 0) {
    for (std::uint32_t index = 0;
         index < dispatch_sizes(a.dispatch).incoming_signal_count; ++index) {
      const auto signal = a.dispatch.incoming_proxy_signals[index];
      if (signal.source_node != a.node)
        remote_incoming += signal.expected_packets;
    }
    if (remote_incoming != 0)
      gin.waitSignal(ncclCoopThread{}, 0, remote_incoming);
  }
  __syncthreads();

  unsigned long long return_count = 0;
  for (std::uint32_t signal_index = 0;
       signal_index < dispatch_sizes(a.dispatch).incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node)
      continue;
    const auto source_partition =
        signal.source_node < a.node ? signal.source_node
                                    : signal.source_node - 1U;
    for (std::uint64_t sequence = 0;
         sequence < signal.expected_packets; ++sequence) {
      const auto packet = source_partition * a.max_tokens_per_rank + sequence;
      const auto header = receive_headers[packet];
      const auto fanout_count = header.route_index;
      for (std::uint64_t fanout = 0; fanout < fanout_count; ++fanout) {
        const auto target = receive_fanout[
            packet * a.dispatch.identity.top_k + fanout];
        const auto destination_rank =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *destination_canonical =
            reinterpret_cast<transport::BFloat16 *>(
                static_cast<std::byte *>(peer_bases[destination_rank]) +
                a.canonical_offset);
        const auto staging =
            (packet * a.dispatch.identity.top_k + fanout) * a.hidden;
        for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
          const auto value = receive_payload[packet * a.hidden + h];
          destination_canonical[target.dst_row * a.hidden + h] = value;
          return_staging[staging + h] = synthetic_value(
              value, target.expert_id, h, a.identity_expert);
        }
        __syncthreads();
      }
      if (lane == 0 && fanout_count != 0) {
        gin.put(rail, static_cast<int>(signal.source_node), a.window,
                a.return_receive_offset +
                    header.dst_row * a.hidden *
                        sizeof(transport::BFloat16),
                a.window,
                a.return_staging_offset +
                    packet * a.dispatch.identity.top_k * a.hidden *
                        sizeof(transport::BFloat16),
                fanout_count * a.hidden * sizeof(transport::BFloat16),
                ncclGin_SignalInc{1}, ncclGin_CounterInc{0},
                ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++return_count;
      }
      __syncthreads();
    }
  }
  if (lane == 0 && return_count != 0)
    gin.waitCounter(ncclCoopThread{}, 0, remote_packets + return_count);
  __syncthreads();
  if (lane == 0 && remote_packets != 0)
    gin.waitSignal(ncclCoopThread{}, 1, remote_packets);
  __syncthreads();

  // Map the packed return payload back to the original route slots using the
  // same deterministic outgoing packet/fan-out order which formed the put.
  for (std::uint64_t packet = 0;
       packet < dispatch_sizes(a.dispatch).packet_count; ++packet) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
    for (auto fanout = fanout_begin; fanout < fanout_end; ++fanout) {
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = descriptor.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x)
        combine_compute[route * a.hidden + h] =
            return_receive[fanout * a.hidden + h];
      __syncthreads();
    }
  }

  for (std::uint64_t token = 0; token < a.dispatch.identity.num_tokens;
       ++token) {
    for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
      float sum = 0.0F;
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto slot = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + slot;
        sum = gate_after_accumulate(
            sum, a.gates[route], combine_compute[route * a.hidden + h].bits);
      }
      combined[token * a.hidden + h] =
          transport::BFloat16{float_to_bf16_bits(sum)};
    }
  }
  __syncthreads();
  if (lane == 0) {
    status->packets_done = dispatch_sizes(a.dispatch).packet_count;
    status->fanout_done = dispatch_sizes(a.dispatch).fanout_count;
    status->synthetic_done = dispatch_sizes(a.dispatch).incoming_row_count;
    status->combine_done = dispatch_sizes(a.dispatch).route_count;
    status->incoming_done = dispatch_sizes(a.dispatch).incoming_row_count;
    status->reduced_tokens = a.dispatch.identity.num_tokens;
    status->raw_cross_node_payload_bytes =
        remote_routes * a.hidden * sizeof(transport::BFloat16);
    status->deduplicated_cross_node_payload_bytes =
        remote_packets * a.hidden * sizeof(transport::BFloat16);
    status->actual_dispatch_peer_bytes =
        status->deduplicated_cross_node_payload_bytes;
    status->actual_return_peer_bytes =
        remote_routes * a.hidden * sizeof(transport::BFloat16);
    status->complete_clock = device_globaltimer_ns();
  }
}
#endif

