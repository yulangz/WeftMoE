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

__device__ void runtime_shape_backward_progress(
    const BackwardKernelArgs &a, std::uint32_t index,
    unsigned long long value) {
  if (a.progress != nullptr && threadIdx.x == 0U) {
    a.progress[index] = value;
    __threadfence_system();
  }
}

#if __has_include(<nccl_device.h>)
__device__ void runtime_shape_forward_gin_dispatch(
    const ForwardKernelArgs &a, TransportDeviceStatus *status) {
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_send_header_offset);
  auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_send_fanout_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_send_payload_offset);
  for (std::uint64_t item = threadIdx.x;
       item < a.dispatch.sizes.packet_count * a.hidden;
       item += blockDim.x) {
    const auto packet = item / a.hidden;
    const auto h = item % a.hidden;
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node != a.node)
      send_payload[item] =
          a.input[descriptor.local_token_id * a.hidden + h];
  }
  const auto fanout_count = a.dispatch.packet_fanout_offsets[
      a.dispatch.sizes.packet_count];
  for (std::uint64_t fanout = threadIdx.x; fanout < fanout_count;
       fanout += blockDim.x)
    send_fanout[fanout] = a.dispatch.packet_fanout[fanout];
  for (std::uint64_t packet = threadIdx.x;
       packet < a.dispatch.sizes.packet_count; packet += blockDim.x) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto begin = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1U];
    send_headers[packet] = LocalP2pRouteHeader{
        transport::kActivationTransportSchemaVersion,
        a.rank,
        descriptor.proxy_rank,
        descriptor.channel,
        a.dispatch.identity.generation,
        a.epoch,
        descriptor.sequence_on_link,
        end - begin,
        descriptor.local_token_id,
        begin,
        0,
        0};
  }
  __threadfence_system();
  __syncthreads();

  const ncclGin gin(a.gin_dev_comm, 0);
  const auto rail = ncclTeamRail(a.gin_dev_comm);
  if (threadIdx.x == 0U) {
    unsigned long long puts = 0ULL;
    for (std::uint32_t remote_node = 0;
         remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
      if (remote_node == a.node)
        continue;
      std::uint64_t first = a.dispatch.sizes.packet_count;
      std::uint64_t last = 0;
      std::uint64_t first_fanout = 0;
      std::uint64_t last_fanout = 0;
      std::uint64_t first_sequence = 0;
      for (std::uint64_t packet = 0;
           packet < a.dispatch.sizes.packet_count; ++packet) {
        const auto descriptor = a.dispatch.packet_headers[packet];
        if (descriptor.destination_node != remote_node)
          continue;
        if (first == a.dispatch.sizes.packet_count) {
          first = packet;
          first_sequence = descriptor.sequence_on_link;
          first_fanout = a.dispatch.packet_fanout_offsets[packet];
        }
        last = packet + 1U;
        last_fanout = a.dispatch.packet_fanout_offsets[packet + 1U];
      }
      if (first == a.dispatch.sizes.packet_count)
        continue;
      const auto source_partition =
          a.node < remote_node ? a.node : a.node - 1U;
      const auto receive_packet =
          source_partition * a.max_tokens_per_rank + first_sequence;
      const auto packets = last - first;
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_header_offset +
                  receive_packet * sizeof(LocalP2pRouteHeader),
              a.gin_window,
              a.gin_send_header_offset +
                  first * sizeof(LocalP2pRouteHeader),
              packets * sizeof(LocalP2pRouteHeader));
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_fanout_offset +
                  (source_partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first_fanout) * sizeof(DeviceFanoutRecord),
              a.gin_window,
              a.gin_send_fanout_offset +
                  first_fanout * sizeof(DeviceFanoutRecord),
              (last_fanout - first_fanout) * sizeof(DeviceFanoutRecord));
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_payload_offset +
                  receive_packet * a.hidden *
                      sizeof(transport::BFloat16),
              a.gin_window,
              a.gin_send_payload_offset +
                  first * a.hidden * sizeof(transport::BFloat16),
              packets * a.hidden * sizeof(transport::BFloat16),
              ncclGin_SignalInc{0}, ncclGin_CounterInc{0}, ncclCoopThread{},
              ncclGin_None{}, ::cuda::thread_scope_block,
              ::cuda::thread_scope_device);
      atomicAdd(&status->gin_dispatch_put_count, 3ULL);
      atomicAdd(&status->gin_signal_count, 1ULL);
      atomicAdd(&status->gin_metadata_bytes,
                packets * sizeof(LocalP2pRouteHeader) +
                    (last_fanout - first_fanout) *
                        sizeof(DeviceFanoutRecord));
      atomicAdd(&status->raw_cross_node_payload_bytes,
                (last_fanout - first_fanout) * a.hidden *
                    sizeof(transport::BFloat16));
      ++puts;
    }
    if (puts != 0ULL)
      gin.waitCounter(ncclCoopThread{}, 0, puts);
    gin.resetCounter(0);
    unsigned long long incoming = 0ULL;
    for (std::uint32_t index = 0;
         index < a.dispatch.sizes.incoming_signal_count; ++index) {
      const auto signal = a.dispatch.incoming_proxy_signals[index];
      if (signal.source_node != a.node && signal.expected_packets != 0ULL)
        ++incoming;
    }
    if (incoming != 0ULL)
      gin.waitSignal(ncclCoopThread{}, 0, incoming);
  }
  __syncthreads();

  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_receive_payload_offset);
  for (std::uint32_t signal_index = 0;
       signal_index < a.dispatch.sizes.incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node)
      continue;
    const auto source_partition =
        signal.source_node < a.node ? signal.source_node
                                    : signal.source_node - 1U;
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k *
                    a.hidden;
         item += blockDim.x) {
      const auto sequence = item /
                            (a.dispatch.identity.top_k * a.hidden);
      const auto within = item %
                          (a.dispatch.identity.top_k * a.hidden);
      const auto fanout = within / a.hidden;
      const auto h = within % a.hidden;
      const auto packet =
          source_partition * a.max_tokens_per_rank + sequence;
      const auto header = receive_headers[packet];
      if (fanout >= header.route_index)
        continue;
      const auto target = receive_fanout[
          source_partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *destination_rows = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.canonical_rows_offset);
      destination_rows[target.dst_row * a.hidden + h] =
          receive_payload[packet * a.hidden + h];
    }
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k;
         item += blockDim.x) {
      const auto sequence = item / a.dispatch.identity.top_k;
      const auto fanout = item % a.dispatch.identity.top_k;
      const auto packet =
          source_partition * a.max_tokens_per_rank + sequence;
      const auto header = receive_headers[packet];
      if (fanout >= header.route_index)
        continue;
      const auto target = receive_fanout[
          source_partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      const auto route = header.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      destination_headers[target.dst_row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          header.source_rank,
          destination,
          header.channel,
          header.generation,
          header.source_epoch,
          header.sequence,
          route,
          header.local_token_id,
          target.dst_row,
          target.expert_id,
          target.topk_slot};
    }
  }
  __threadfence_system();
  __syncthreads();
}

__device__ void runtime_shape_forward_gin_return(
    const ForwardKernelArgs &a, TransportDeviceStatus *status,
    const transport::BFloat16 *expert_output,
    transport::BFloat16 *contribution) {
  const auto first_local = a.node * a.local_world_size;
  if (threadIdx.x == 0U) {
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      auto *ready = peer_at<unsigned long long>(
          a.peer_bases[peer], a.gin_local_compute_ready_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(ready[a.rank]);
      published.store(a.epoch, ::cuda::memory_order_release);
    }
    auto *local_ready = peer_at<unsigned long long>(
        a.local_base, a.gin_local_compute_ready_offset);
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          observed(local_ready[peer]);
      while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
             !forward_cancelled(a)) {
      }
    }
  }
  __syncthreads();
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *staging = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_return_staging_offset);
  for (std::uint32_t signal_index = 0;
       signal_index < a.dispatch.sizes.incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node)
      continue;
    const auto partition = signal.source_node < a.node
                               ? signal.source_node
                               : signal.source_node - 1U;
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k *
                    a.hidden;
         item += blockDim.x) {
      const auto sequence = item /
                            (a.dispatch.identity.top_k * a.hidden);
      const auto within = item %
                          (a.dispatch.identity.top_k * a.hidden);
      const auto fanout_slot = within / a.hidden;
      const auto h = within % a.hidden;
      const auto packet = partition * a.max_tokens_per_rank + sequence;
      const auto header = headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target = fanout[
          partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      const auto *output = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.forward_output_rows_offset);
      staging[(partition * a.max_tokens_per_rank *
                   a.dispatch.identity.top_k +
               header.dst_row + fanout_slot) *
                  a.hidden +
              h] = output[target.dst_row * a.hidden + h];
    }
  }
  __threadfence_system();
  __syncthreads();
  const ncclGin gin(a.gin_dev_comm, 0);
  const auto rail = ncclTeamRail(a.gin_dev_comm);
  if (threadIdx.x == 0U) {
    unsigned long long puts = 0ULL;
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node || signal.expected_packets == 0ULL)
        continue;
      const auto partition = signal.source_node < a.node
                                 ? signal.source_node
                                 : signal.source_node - 1U;
      std::uint64_t first = 0;
      std::uint64_t last = 0;
      for (std::uint64_t sequence = 0;
           sequence < signal.expected_packets; ++sequence) {
        const auto header =
            headers[partition * a.max_tokens_per_rank + sequence];
        if (sequence == 0)
          first = header.dst_row;
        last = header.dst_row + header.route_index;
      }
      if (first == last)
        continue;
      gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
              a.gin_return_receive_offset +
                  first * a.hidden * sizeof(transport::BFloat16),
              a.gin_window,
              a.gin_return_staging_offset +
                  (partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first) *
                      a.hidden * sizeof(transport::BFloat16),
              (last - first) * a.hidden * sizeof(transport::BFloat16),
              ncclGin_SignalInc{1}, ncclGin_CounterInc{0}, ncclCoopThread{},
              ncclGin_None{}, ::cuda::thread_scope_block,
              ::cuda::thread_scope_device);
      atomicAdd(&status->gin_return_put_count, 1ULL);
      atomicAdd(&status->gin_signal_count, 1ULL);
      atomicAdd(&status->gin_return_payload_bytes,
                (last - first) * a.hidden *
                    sizeof(transport::BFloat16));
      ++puts;
    }
    if (puts != 0ULL)
      gin.waitCounter(ncclCoopThread{}, 0, puts);
    bool expects_return = false;
    for (std::uint64_t packet = 0;
         packet < a.dispatch.sizes.packet_count; ++packet)
      expects_return = expects_return ||
                       a.dispatch.packet_headers[packet].destination_node !=
                           a.node;
    if (expects_return)
      gin.waitSignal(ncclCoopThread{}, 1, 1);
  }
  __syncthreads();
  auto *receive = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_return_receive_offset);
  for (std::uint64_t item = threadIdx.x;
       item < a.dispatch.sizes.packet_count *
                  a.dispatch.identity.top_k * a.hidden;
       item += blockDim.x) {
    const auto packet = item /
                        (a.dispatch.identity.top_k * a.hidden);
    const auto within = item %
                        (a.dispatch.identity.top_k * a.hidden);
    const auto fanout_slot = within / a.hidden;
    const auto h = within % a.hidden;
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto begin = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1U];
    if (fanout_slot >= end - begin)
      continue;
    const auto fanout_index = begin + fanout_slot;
    const auto target = a.dispatch.packet_fanout[fanout_index];
    const auto route = descriptor.local_token_id *
                           a.dispatch.identity.top_k +
                       target.topk_slot;
    contribution[route * a.hidden + h] =
        receive[fanout_index * a.hidden + h];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0U) {
    auto *signals = peer_at<unsigned long long>(
        a.local_base, a.outgoing_sequence_offset);
    for (std::uint32_t destination = 0; destination < a.world;
         ++destination) {
      if (destination / a.local_world_size == a.node)
        continue;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(signals[destination * a.channels]);
      ready.store(a.epoch, ::cuda::memory_order_release);
    }
  }
  __syncthreads();
}

__device__ void runtime_shape_backward_gin_dispatch(
    const BackwardKernelArgs &a, BackwardDeviceStatus *status) {
  const auto first_local = a.node * a.local_world_size;
  if (threadIdx.x == 0U) {
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      auto *ready = peer_at<unsigned long long>(
          a.peer_bases[peer], a.gin_local_compute_ready_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(ready[a.rank]);
      published.store(a.epoch, ::cuda::memory_order_release);
    }
    auto *local_ready = peer_at<unsigned long long>(
        a.local_base, a.gin_local_compute_ready_offset);
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          observed(local_ready[peer]);
      while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
             !backward_cancelled(a)) {
      }
    }
  }
  __syncthreads();
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_send_header_offset);
  auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_send_fanout_offset);
  auto *send_gate = reinterpret_cast<float *>(bytes + a.gin_send_gate_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_send_payload_offset);
  for (std::uint64_t item = threadIdx.x;
       item < a.dispatch.sizes.packet_count * a.hidden;
       item += blockDim.x) {
    const auto packet = item / a.hidden;
    const auto h = item % a.hidden;
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node != a.node)
      send_payload[item] =
          a.dy_input[descriptor.local_token_id * a.hidden + h];
  }
  const auto fanout_count = a.dispatch.packet_fanout_offsets[
      a.dispatch.sizes.packet_count];
  for (std::uint64_t fanout = threadIdx.x; fanout < fanout_count;
       fanout += blockDim.x)
    send_fanout[fanout] = a.dispatch.packet_fanout[fanout];
  for (std::uint64_t packet = threadIdx.x;
       packet < a.dispatch.sizes.packet_count; packet += blockDim.x) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto begin = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1U];
    for (auto fanout = begin; fanout < end; ++fanout) {
      const auto target = a.dispatch.packet_fanout[fanout];
      send_gate[fanout] =
          a.gates[descriptor.local_token_id * a.dispatch.identity.top_k +
                  target.topk_slot];
    }
    send_headers[packet] = LocalP2pRouteHeader{
        transport::kActivationTransportSchemaVersion,
        a.rank,
        descriptor.proxy_rank,
        descriptor.channel,
        a.dispatch.identity.generation,
        a.epoch,
        descriptor.sequence_on_link,
        end - begin,
        descriptor.local_token_id,
        begin,
        0,
        0};
  }
  __threadfence_system();
  __syncthreads();
  const ncclGin gin(a.gin_dev_comm, 0);
  const auto rail = ncclTeamRail(a.gin_dev_comm);
  if (threadIdx.x == 0U) {
    unsigned long long puts = 0ULL;
    for (std::uint32_t remote_node = 0;
         remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
      if (remote_node == a.node)
        continue;
      std::uint64_t first = a.dispatch.sizes.packet_count;
      std::uint64_t last = 0;
      std::uint64_t first_fanout = 0;
      std::uint64_t last_fanout = 0;
      std::uint64_t first_sequence = 0;
      for (std::uint64_t packet = 0;
           packet < a.dispatch.sizes.packet_count; ++packet) {
        const auto descriptor = a.dispatch.packet_headers[packet];
        if (descriptor.destination_node != remote_node)
          continue;
        if (first == a.dispatch.sizes.packet_count) {
          first = packet;
          first_sequence = descriptor.sequence_on_link;
          first_fanout = a.dispatch.packet_fanout_offsets[packet];
        }
        last = packet + 1U;
        last_fanout = a.dispatch.packet_fanout_offsets[packet + 1U];
      }
      if (first == a.dispatch.sizes.packet_count)
        continue;
      const auto partition = a.node < remote_node ? a.node : a.node - 1U;
      const auto receive_packet =
          partition * a.max_tokens_per_rank + first_sequence;
      const auto packets = last - first;
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_header_offset +
                  receive_packet * sizeof(LocalP2pRouteHeader),
              a.gin_window,
              a.gin_send_header_offset +
                  first * sizeof(LocalP2pRouteHeader),
              packets * sizeof(LocalP2pRouteHeader));
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_fanout_offset +
                  (partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first_fanout) * sizeof(DeviceFanoutRecord),
              a.gin_window,
              a.gin_send_fanout_offset +
                  first_fanout * sizeof(DeviceFanoutRecord),
              (last_fanout - first_fanout) * sizeof(DeviceFanoutRecord));
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_gate_offset +
                  (partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first_fanout) * sizeof(float),
              a.gin_window,
              a.gin_send_gate_offset + first_fanout * sizeof(float),
              (last_fanout - first_fanout) * sizeof(float));
      gin.put(rail, static_cast<int>(remote_node), a.gin_window,
              a.gin_receive_payload_offset +
                  receive_packet * a.hidden *
                      sizeof(transport::BFloat16),
              a.gin_window,
              a.gin_send_payload_offset +
                  first * a.hidden * sizeof(transport::BFloat16),
              packets * a.hidden * sizeof(transport::BFloat16),
              ncclGin_SignalInc{0}, ncclGin_CounterInc{0}, ncclCoopThread{},
              ncclGin_None{}, ::cuda::thread_scope_block,
              ::cuda::thread_scope_device);
      atomicAdd(&status->gin_dispatch_put_count, 4ULL);
      atomicAdd(&status->gin_signal_count, 1ULL);
      atomicAdd(&status->gin_metadata_bytes,
                packets * sizeof(LocalP2pRouteHeader) +
                    (last_fanout - first_fanout) *
                        (sizeof(DeviceFanoutRecord) + sizeof(float)));
      atomicAdd(&status->gin_dispatch_payload_bytes,
                packets * a.hidden * sizeof(transport::BFloat16));
      ++puts;
    }
    if (puts != 0ULL)
      gin.waitCounter(ncclCoopThread{}, 0, puts);
    gin.resetCounter(0);
    unsigned long long incoming = 0ULL;
    for (std::uint32_t index = 0;
         index < a.dispatch.sizes.incoming_signal_count; ++index) {
      const auto signal = a.dispatch.incoming_proxy_signals[index];
      if (signal.source_node != a.node && signal.expected_packets != 0ULL)
        ++incoming;
    }
    if (incoming != 0ULL)
      gin.waitSignal(ncclCoopThread{}, 0, incoming);
  }
  __syncthreads();
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *receive_gate = reinterpret_cast<float *>(
      bytes + a.gin_receive_gate_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_receive_payload_offset);
  for (std::uint32_t signal_index = 0;
       signal_index < a.dispatch.sizes.incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node)
      continue;
    const auto partition = signal.source_node < a.node
                               ? signal.source_node
                               : signal.source_node - 1U;
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k *
                    a.hidden;
         item += blockDim.x) {
      const auto sequence = item /
                            (a.dispatch.identity.top_k * a.hidden);
      const auto within = item %
                          (a.dispatch.identity.top_k * a.hidden);
      const auto fanout_slot = within / a.hidden;
      const auto h = within % a.hidden;
      const auto packet = partition * a.max_tokens_per_rank + sequence;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target = receive_fanout[
          partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *destination_dy = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.dy_canonical_offset);
      destination_dy[target.dst_row * a.hidden + h] =
          receive_payload[packet * a.hidden + h];
    }
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k;
         item += blockDim.x) {
      const auto sequence = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto packet = partition * a.max_tokens_per_rank + sequence;
      const auto header = receive_headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto fanout_index =
          partition * a.max_tokens_per_rank *
              a.dispatch.identity.top_k +
          header.dst_row + fanout_slot;
      const auto target = receive_fanout[fanout_index];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      destination_gate[target.dst_row] = receive_gate[fanout_index];
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
      const auto route = header.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      destination_headers[target.dst_row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          header.source_rank,
          destination,
          header.channel,
          header.generation,
          header.source_epoch,
          header.sequence,
          route,
          header.local_token_id,
          target.dst_row,
          target.expert_id,
          target.topk_slot};
    }
  }
  __threadfence_system();
  __syncthreads();
}

__device__ void runtime_shape_backward_gin_return(
    const BackwardKernelArgs &a, BackwardDeviceStatus *status,
    const transport::BFloat16 *dx_canonical,
    const float *dgate_canonical, transport::BFloat16 *dx_return,
    float *dgate) {
  const auto first_local = a.node * a.local_world_size;
  const auto compute_epoch = a.epoch | (1ULL << 63U);
  if (threadIdx.x == 0U) {
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      auto *ready = peer_at<unsigned long long>(
          a.peer_bases[peer], a.gin_local_compute_ready_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(ready[a.rank]);
      published.store(compute_epoch, ::cuda::memory_order_release);
    }
    auto *local_ready = peer_at<unsigned long long>(
        a.local_base, a.gin_local_compute_ready_offset);
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          observed(local_ready[peer]);
      while (observed.load(::cuda::memory_order_acquire) < compute_epoch &&
             !backward_cancelled(a)) {
      }
    }
  }
  __syncthreads();
  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  auto *fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.gin_receive_fanout_offset);
  auto *staging = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_return_staging_offset);
  auto *gate_staging = reinterpret_cast<float *>(
      bytes + a.gin_return_gate_staging_offset);
  for (std::uint32_t signal_index = 0;
       signal_index < a.dispatch.sizes.incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node)
      continue;
    const auto partition = signal.source_node < a.node
                               ? signal.source_node
                               : signal.source_node - 1U;
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k *
                    a.hidden;
         item += blockDim.x) {
      const auto sequence = item /
                            (a.dispatch.identity.top_k * a.hidden);
      const auto within = item %
                          (a.dispatch.identity.top_k * a.hidden);
      const auto fanout_slot = within / a.hidden;
      const auto h = within % a.hidden;
      const auto packet = partition * a.max_tokens_per_rank + sequence;
      const auto header = headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto target = fanout[
          partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout_slot];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      const auto *source_dx = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.dx_canonical_offset);
      staging[(partition * a.max_tokens_per_rank *
                   a.dispatch.identity.top_k +
               header.dst_row + fanout_slot) *
                  a.hidden +
              h] = source_dx[target.dst_row * a.hidden + h];
    }
    for (std::uint64_t item = threadIdx.x;
         item < signal.expected_packets * a.dispatch.identity.top_k;
         item += blockDim.x) {
      const auto sequence = item / a.dispatch.identity.top_k;
      const auto fanout_slot = item % a.dispatch.identity.top_k;
      const auto packet = partition * a.max_tokens_per_rank + sequence;
      const auto header = headers[packet];
      if (fanout_slot >= header.route_index)
        continue;
      const auto fanout_index =
          partition * a.max_tokens_per_rank *
              a.dispatch.identity.top_k +
          header.dst_row + fanout_slot;
      const auto target = fanout[fanout_index];
      const auto destination =
          a.node * a.local_world_size + target.destination_local_rank;
      const auto *source_gate = peer_at<float>(
          a.peer_bases[destination], a.dgate_canonical_offset);
      gate_staging[fanout_index] = source_gate[target.dst_row];
    }
  }
  __threadfence_system();
  __syncthreads();
  const ncclGin gin(a.gin_dev_comm, 0);
  const auto rail = ncclTeamRail(a.gin_dev_comm);
  if (threadIdx.x == 0U) {
    unsigned long long puts = 0ULL;
    // The production topology is homogeneous and the current physical GIN
    // matrix has one remote node. Iterate the descriptor list directly so the
    // correctness fallback keeps no compile-time node bound or local array.
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node || signal.expected_packets == 0ULL)
        continue;
      const auto partition = signal.source_node < a.node
                                 ? signal.source_node
                                 : signal.source_node - 1U;
      std::uint64_t first = 0;
      std::uint64_t last = 0;
      for (std::uint64_t sequence = 0;
           sequence < signal.expected_packets; ++sequence) {
        const auto header =
            headers[partition * a.max_tokens_per_rank + sequence];
        if (sequence == 0)
          first = header.dst_row;
        last = header.dst_row + header.route_index;
      }
      if (first == last)
        continue;
      gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
              a.gin_return_receive_offset +
                  first * a.hidden * sizeof(transport::BFloat16),
              a.gin_window,
              a.gin_return_staging_offset +
                  (partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first) *
                      a.hidden * sizeof(transport::BFloat16),
              (last - first) * a.hidden * sizeof(transport::BFloat16));
      gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
              a.gin_return_gate_receive_offset + first * sizeof(float),
              a.gin_window,
              a.gin_return_gate_staging_offset +
                  (partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first) * sizeof(float),
              (last - first) * sizeof(float), ncclGin_SignalInc{1},
              ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
              ::cuda::thread_scope_block, ::cuda::thread_scope_device);
      atomicAdd(&status->gin_return_put_count, 2ULL);
      atomicAdd(&status->gin_signal_count, 1ULL);
      atomicAdd(&status->gin_metadata_bytes,
                (last - first) * sizeof(float));
      atomicAdd(&status->gin_return_payload_bytes,
                (last - first) * a.hidden *
                    sizeof(transport::BFloat16));
      // Only the gate put carries the local-completion counter increment;
      // the paired dX put is ordered before it on the same context.
      ++puts;
    }
    if (puts != 0ULL)
      gin.waitCounter(ncclCoopThread{}, 0, puts);
    unsigned long long scalar_puts = 0ULL;
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node || signal.expected_packets == 0ULL)
        continue;
      const auto partition = signal.source_node < a.node
                                 ? signal.source_node
                                 : signal.source_node - 1U;
      std::uint64_t first = 0;
      std::uint64_t last = 0;
      for (std::uint64_t sequence = 0;
           sequence < signal.expected_packets; ++sequence) {
        const auto header =
            headers[partition * a.max_tokens_per_rank + sequence];
        if (sequence == 0)
          first = header.dst_row;
        last = header.dst_row + header.route_index;
      }
      if (first == last)
        continue;
      gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
              a.gin_return_gate_receive_offset + first * sizeof(float),
              a.gin_window,
              a.gin_return_gate_staging_offset +
                  (partition * a.max_tokens_per_rank *
                       a.dispatch.identity.top_k +
                   first) * sizeof(float),
              (last - first) * sizeof(float), ncclGin_SignalInc{1},
              ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
              ::cuda::thread_scope_block, ::cuda::thread_scope_device);
      ++scalar_puts;
      atomicAdd(&status->gin_return_put_count, 1ULL);
      atomicAdd(&status->gin_signal_count, 1ULL);
    }
    if (scalar_puts != 0ULL)
      gin.waitCounter(ncclCoopThread{}, 0, puts + scalar_puts);
    bool expects_return = false;
    for (std::uint64_t packet = 0;
         packet < a.dispatch.sizes.packet_count; ++packet)
      expects_return = expects_return ||
                       a.dispatch.packet_headers[packet].destination_node !=
                           a.node;
    if (expects_return)
      gin.waitSignal(ncclCoopThread{}, 1, 2);
  }
  __syncthreads();
  auto *receive = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.gin_return_receive_offset);
  auto *gate_receive = reinterpret_cast<float *>(
      bytes + a.gin_return_gate_receive_offset);
  for (std::uint64_t item = threadIdx.x;
       item < a.dispatch.sizes.packet_count *
                  a.dispatch.identity.top_k * a.hidden;
       item += blockDim.x) {
    const auto packet = item /
                        (a.dispatch.identity.top_k * a.hidden);
    const auto within = item %
                        (a.dispatch.identity.top_k * a.hidden);
    const auto fanout_slot = within / a.hidden;
    const auto h = within % a.hidden;
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto begin = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1U];
    if (fanout_slot >= end - begin)
      continue;
    const auto fanout_index = begin + fanout_slot;
    const auto target = a.dispatch.packet_fanout[fanout_index];
    const auto route = descriptor.local_token_id *
                           a.dispatch.identity.top_k +
                       target.topk_slot;
    dx_return[route * a.hidden + h] =
        receive[fanout_index * a.hidden + h];
  }
  for (std::uint64_t packet = threadIdx.x;
       packet < a.dispatch.sizes.packet_count; packet += blockDim.x) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto begin = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1U];
    for (auto fanout_index = begin; fanout_index < end; ++fanout_index) {
      const auto target = a.dispatch.packet_fanout[fanout_index];
      const auto route = descriptor.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      dgate[route] = gate_receive[fanout_index];
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0U) {
    auto *signals = peer_at<unsigned long long>(
        a.local_base, a.outgoing_signal_offset);
    for (std::uint32_t destination = 0; destination < a.world;
         ++destination) {
      if (destination / a.local_world_size == a.node)
        continue;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(signals[destination * a.channels]);
      ready.store(a.epoch, ::cuda::memory_order_release);
    }
  }
  __syncthreads();
}
#endif

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

  if (!a.use_socket) {
    const auto dispatch_elements = a.dispatch.sizes.route_count * a.hidden;
    for (std::uint64_t item = threadIdx.x; item < dispatch_elements;
         item += blockDim.x) {
      const auto route = item / a.hidden;
      const auto h = item % a.hidden;
      const auto destination = a.dispatch.route_dst_rank[route];
      if (a.use_gin && destination / a.local_world_size != a.node)
        continue;
      auto *destination_rows = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.canonical_rows_offset);
      destination_rows[a.dispatch.route_dst_row[route] * a.hidden + h] =
          a.input[(route / a.dispatch.identity.top_k) * a.hidden + h];
    }
    __syncthreads();
    for (std::uint64_t route = threadIdx.x;
         route < a.dispatch.sizes.route_count; route += blockDim.x) {
      const auto destination = a.dispatch.route_dst_rank[route];
      if (a.use_gin && destination / a.local_world_size != a.node)
        continue;
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
#if __has_include(<nccl_device.h>)
    if (a.use_gin)
      runtime_shape_forward_gin_dispatch(a, status);
#else
    if (a.use_gin && threadIdx.x == 0U)
      publish_local_structured_error(status, a.rank, 201U, 1U, 0ULL, 1ULL);
    __syncthreads();
#endif
    if (threadIdx.x == 0U) {
      const auto first_peer = a.use_gin ? a.node * a.local_world_size : 0U;
      const auto peer_count = a.use_gin ? a.local_world_size : a.world;
      for (std::uint32_t local = 0; local < peer_count; ++local) {
        const auto peer = first_peer + local;
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[peer], a.ingress_published_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.ingress_published_offset);
      for (std::uint32_t local = 0; local < peer_count; ++local) {
        const auto source = first_peer + local;
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

  if (!a.use_socket) {
    for (std::uint64_t item = threadIdx.x; item < fc2_elements;
         item += blockDim.x) {
      const auto row = item / a.hidden;
      const auto h = item % a.hidden;
      const auto header = headers[row];
      if (a.use_gin &&
          header.source_rank / a.local_world_size != a.node)
        continue;
      auto *source = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.contribution_offset);
      source[header.route_index * a.hidden + h] = expert_output[item];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
      const auto first_source = a.use_gin ? a.node * a.local_world_size : 0U;
      const auto source_count = a.use_gin ? a.local_world_size : a.world;
      for (std::uint32_t local = 0; local < source_count; ++local) {
        const auto source = first_source + local;
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[source], a.outgoing_sequence_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
    }
#if __has_include(<nccl_device.h>)
    if (a.use_gin)
      runtime_shape_forward_gin_return(a, status, expert_output,
                                       contribution);
#endif
    __syncthreads();
    if (threadIdx.x == 0U) {
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
  publish_production_terminal_status<false>(
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
  runtime_shape_backward_progress(a, kBackwardEntered, a.epoch);

  if (!a.use_socket) {
    const auto elements = a.dispatch.sizes.route_count * a.hidden;
    for (std::uint64_t item = threadIdx.x; item < elements;
         item += blockDim.x) {
      const auto route = item / a.hidden;
      const auto h = item % a.hidden;
      const auto destination = a.dispatch.route_dst_rank[route];
      if (a.use_gin && destination / a.local_world_size != a.node)
        continue;
      auto *destination_dy = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.dy_canonical_offset);
      destination_dy[a.dispatch.route_dst_row[route] * a.hidden + h] =
          a.dy_input[(route / a.dispatch.identity.top_k) * a.hidden + h];
    }
    __syncthreads();
    for (std::uint64_t route = threadIdx.x;
         route < a.dispatch.sizes.route_count; route += blockDim.x) {
      const auto destination = a.dispatch.route_dst_rank[route];
      if (a.use_gin && destination / a.local_world_size != a.node)
        continue;
      const auto row = a.dispatch.route_dst_row[route];
      auto *destination_gate = peer_at<float>(
          a.peer_bases[destination], a.canonical_gate_offset);
      destination_gate[row] = a.gates[route];
    }
    __threadfence_system();
    __syncthreads();
#if __has_include(<nccl_device.h>)
    if (a.use_gin)
      runtime_shape_backward_gin_dispatch(a, status);
#else
    if (a.use_gin && threadIdx.x == 0U)
      publish_local_structured_error(status, a.rank, 301U, 1U, 0ULL, 1ULL);
    __syncthreads();
#endif
    if (threadIdx.x == 0U) {
      const auto first_peer = a.use_gin ? a.node * a.local_world_size : 0U;
      const auto peer_count = a.use_gin ? a.local_world_size : a.world;
      for (std::uint32_t local = 0; local < peer_count; ++local) {
        const auto peer = first_peer + local;
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[peer], a.incoming_signal_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.incoming_signal_offset);
      for (std::uint32_t local = 0; local < peer_count; ++local) {
        const auto source = first_peer + local;
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
  runtime_shape_backward_progress(a, kBackwardDyComplete, a.epoch);

  const auto incoming = a.dispatch.sizes.incoming_row_count;
  const auto fc1_elements = incoming * a.intermediate;
  if (threadIdx.x == 0U)
    status->dv_tasks = fc1_elements;
  runtime_shape_backward_progress(a, kBackwardDvStarted, fc1_elements);
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
  runtime_shape_backward_progress(a, kBackwardDvComplete, fc1_elements);

  const auto dx_elements = incoming * a.hidden;
  if (threadIdx.x == 0U)
    status->dx_tasks = dx_elements;
  runtime_shape_backward_progress(a, kBackwardDxStarted, dx_elements);
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
  runtime_shape_backward_progress(a, kBackwardDxComplete, dx_elements);

  const auto dw1_per_expert = 2ULL * a.intermediate * a.hidden;
  const auto dw1_elements =
      a.dispatch.sizes.owned_expert_count * dw1_per_expert;
  if (threadIdx.x == 0U)
    status->dw1_tasks = dw1_elements;
  runtime_shape_backward_progress(a, kBackwardDw1Started, dw1_elements);
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
  if (threadIdx.x == 0U)
    status->dw2_tasks = dw2_elements;
  runtime_shape_backward_progress(a, kBackwardDw2Started, dw2_elements);
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
  runtime_shape_backward_progress(a, kBackwardDw2Complete, dw2_elements);
  runtime_shape_backward_progress(a, kBackwardDw1Complete, dw1_elements);

  if (!a.use_socket) {
    for (std::uint64_t item = threadIdx.x; item < dx_elements;
         item += blockDim.x) {
      const auto row = item / a.hidden;
      const auto h = item % a.hidden;
      const auto header = headers[row];
      if (a.use_gin &&
          header.source_rank / a.local_world_size != a.node)
        continue;
      auto *source_dx = peer_at<transport::BFloat16>(
          a.peer_bases[header.source_rank], a.dx_return_offset);
      source_dx[header.route_index * a.hidden + h] = dx_canonical[item];
    }
    for (std::uint64_t row = threadIdx.x; row < incoming;
         row += blockDim.x) {
      const auto header = headers[row];
      if (a.use_gin &&
          header.source_rank / a.local_world_size != a.node)
        continue;
      auto *source_dgate = peer_at<float>(a.peer_bases[header.source_rank],
                                           a.dgate_offset);
      source_dgate[header.route_index] = dgate_canonical[row];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
      const auto first_source = a.use_gin ? a.node * a.local_world_size : 0U;
      const auto source_count = a.use_gin ? a.local_world_size : a.world;
      for (std::uint32_t local = 0; local < source_count; ++local) {
        const auto source = first_source + local;
        auto *signals = peer_at<unsigned long long>(
            a.peer_bases[source], a.outgoing_signal_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(signals[a.rank * a.channels]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
    }
#if __has_include(<nccl_device.h>)
    if (a.use_gin)
      runtime_shape_backward_gin_return(a, status, dx_canonical,
                                        dgate_canonical, dx_return, dgate);
#endif
    __syncthreads();
    if (threadIdx.x == 0U) {
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
  publish_production_terminal_status<false>(
      a, status, ProductionStatusPhase::Backward,
      a.production_error_records_offset,
      a.gin_production_error_records_offset);
}
