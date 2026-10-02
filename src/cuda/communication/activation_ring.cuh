#pragma once

#include "cuda/communication/return_queue.cuh"

namespace activation_ring_detail {
namespace ring = communication;

inline constexpr unsigned kNoChannel = UINT32_MAX;

__host__ __device__ inline unsigned partition(unsigned node, unsigned peer) {
  return peer < node ? peer : peer - 1U;
}
__host__ __device__ inline unsigned node_at(unsigned node, unsigned index) {
  return index < node ? index : index + 1U;
}

template <class Args>
__device__ __forceinline__ unsigned peer(const Args &a, unsigned channel) {
  if (channel < a.gin_rings.dispatch_channels)
    return node_at(a.node, channel / ring::kDispatchChannels) *
               a.local_world_size +
           a.rank % a.local_world_size;
  const auto target =
      (channel - a.gin_rings.dispatch_channels) / ring::kReturnPorts;
  return node_at(a.node, target / a.local_world_size) * a.local_world_size +
         target % a.local_world_size;
}

template <class Args>
__device__ __forceinline__ unsigned remote_channel(const Args &a,
                                                   unsigned channel) {
  const auto target = peer(a, channel);
  const auto remote_partition = partition(target / a.local_world_size, a.node);
  if (channel < a.gin_rings.dispatch_channels)
    return remote_partition * ring::kDispatchChannels +
           channel % ring::kDispatchChannels;
  const auto port =
      (channel - a.gin_rings.dispatch_channels) % ring::kReturnPorts;
  return a.gin_rings.dispatch_channels +
         (remote_partition * a.local_world_size + a.rank % a.local_world_size) *
             ring::kReturnPorts +
         port;
}

template <class Args>
__device__ __forceinline__ unsigned context_for_cta(const Args &a, unsigned cta) {
  const unsigned available = unsigned(a.gin_dev_comm.ginContextCount);
  return available > 1U ? 1U + cta % (available - 1U) : 0U;
}

template <class Args>
__device__ __forceinline__ ncclGin gin_for_cta(const Args &a, unsigned cta) {
  // GPU scope is valid for both private and shared contexts, and keeps the
  // backend's large submission path to one compile-time sharing mode.
  return ncclGin(a.gin_dev_comm, context_for_cta(a, cta), NCCL_GIN_RESOURCE_SHARING_GPU);
}

// Only receiver-owned channels issue standalone credits. Join their actual
// peer queues once at shutdown; unused FULL connections need no flush.
template <class Args>
__device__ __forceinline__ void flush_receive(const Args &a) {
  const auto gin = gin_for_cta(a, blockIdx.x);
  const unsigned worker = blockIdx.x - a.gin_receiver_base;
  for (unsigned channel = worker; channel < a.gin_rings.channels();
       channel += a.gin_receiver_count) {
    const auto target = peer(a, channel);
    bool seen = false;
    for (unsigned earlier = worker; earlier < channel; earlier += a.gin_receiver_count)
      seen |= peer(a, earlier) == target;
    if (seen) continue;
    ncclGinRequest_t request;
    gin.flushAsync(ncclTeamWorld(gin.comm), target, &request);
    gin.wait(request);
  }
}

template <class Args>
__device__ __forceinline__ unsigned owner(const Args &a, unsigned channel) {
  if (a.gin_sender_count) return channel % a.gin_sender_count;
  return channel < a.gin_rings.dispatch_channels
             ? (channel % ring::kDispatchChannels) % a.gin_context_count
             : peer(a, channel) % a.local_world_size;
}

template <class Args>
__device__ __forceinline__ unsigned data_context(const Args &a,
                                                 unsigned channel) {
  if (a.gin_sender_count) return context_for_cta(a, owner(a, channel));
  if (channel < a.gin_rings.dispatch_channels)
    return (channel % ring::kDispatchChannels) % a.gin_context_count;
  const auto logical =
      (peer(a, channel) % a.local_world_size + a.rank) % a.gin_context_count;
  return a.gin_context_count + ring::kReturnPorts * logical +
         (channel - a.gin_rings.dispatch_channels) % ring::kReturnPorts;
}

// Durable sequences and per-slot signals survive FWD, BWD, arena growth and
// graph replay. Only the finite work interval changes at an operation boundary.
template <class Args>
__device__ __forceinline__ void prepare_channel(const Args &a,
                                                unsigned channel) {
  if (channel >= a.gin_rings.channels())
    return;
  auto &state = ring::channels(a.gin_window_base, a.gin_rings)[channel];
  const ncclGin gin(a.gin_dev_comm, data_context(a, channel));
  state.dispatch_claimed_parts = 0ULL;
  state.dispatch_receive_base = state.claimed;
  for (unsigned slot = 0; slot < ring::kRingDepth; ++slot) {
    state.slots[slot].local_target = gin.readCounter(ring::counter(a.gin_rings, channel, slot));
    state.slots[slot].consumed = 0ULL;
  }
  state.tx_terminal = state.rx_terminal = 0U;
  state.tx_cursor = state.tx_end = 0ULL;
  if (channel >= a.gin_rings.dispatch_channels) {
    if (a.gin_sender_count)
      state.tx_cursor = ((channel - a.gin_rings.dispatch_channels) % ring::kReturnPorts) * ring::kChunkRows;
    return;
  }
  const auto target = peer(a, channel) / a.local_world_size;
  unsigned long long first = 0ULL,
                     end = dispatch_sizes(a.dispatch).packet_count;
  while (first < end) {
    const auto middle = first + (end - first) / 2ULL;
    if (a.dispatch.packet_headers[middle].destination_node < target)
      first = middle + 1ULL;
    else
      end = middle;
  }
  const auto begin = first;
  end = dispatch_sizes(a.dispatch).packet_count;
  while (first < end) {
    const auto middle = first + (end - first) / 2ULL;
    if (a.dispatch.packet_headers[middle].destination_node <= target)
      first = middle + 1ULL;
    else
      end = middle;
  }
  state.tx_cursor =
      begin + (channel % ring::kDispatchChannels) * ring::kChunkRows;
  state.tx_end = end;
}

template <class Args> __global__ void prepare(Args a) {
  prepare_channel(a, blockIdx.x * blockDim.x + threadIdx.x);
}

struct Transfer {
  unsigned channel = kNoChannel;
  unsigned rows = 0;
  unsigned terminal = 0;
  unsigned long long sequence = 0;
  unsigned long long first = 0;
};

struct DefaultHooks {
  template <class Args>
  __device__ __forceinline__ static Transfer returning(
      const Args &a, unsigned channel, unsigned long long sequence,
      unsigned long long position) {
    const auto &queue = return_queue_detail::queues(a)[1U + peer(a, channel)];
    const auto rows = position < queue.capacity
        ? unsigned(min(queue.capacity - position, 1ULL * ring::kChunkRows)) : 0U;
    return {channel, rows,
        position + 1ULL * ring::kReturnPorts * ring::kChunkRows >= queue.capacity,
        sequence, queue.begin + position};
  }
  template <class Args>
  __device__ __forceinline__ static bool receive(
      const Args &, unsigned, const ring::ChunkHeader *) { return true; }
};
template <class Args> struct Hooks : DefaultHooks {};

template <class Args>
__device__ __forceinline__ bool acknowledge(const Args &a, unsigned context) {
  const ncclGin gin(a.gin_dev_comm, context);
  bool progressed = false;
  for (unsigned channel = 0; channel < a.gin_rings.channels(); ++channel)
    if (owner(a, channel) == context)
      progressed = ring::acknowledge(
                       gin, a.gin_window, a.gin_window_base, a.gin_rings,
                       channel, remote_channel(a, channel), peer(a, channel)) ||
                   progressed;
  return progressed;
}

template <class Args>
__device__ __forceinline__ bool drained(const Args &a, unsigned context) {
  for (unsigned channel = 0; channel < a.gin_rings.channels(); ++channel) {
    if (owner(a, channel) != context)
      continue;
    const ncclGin gin(a.gin_dev_comm, data_context(a, channel));
    if (!ring::drained(gin, a.gin_window, a.gin_window_base, a.gin_rings,
                       channel))
      return false;
  }
  return true;
}

// With fixed receive ownership a CTA needs only its own channel terminals.
// The existing CTA completion counters provide the global join.
template <class Args>
__device__ __forceinline__ bool receive_complete(const Args &a, bool dispatch) {
  const unsigned first = dispatch ? 0U : a.gin_rings.dispatch_channels;
  const unsigned end = dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.channels();
  const unsigned worker = blockIdx.x - a.gin_receiver_base;
  const unsigned stride = a.gin_receiver_count;
  const unsigned begin = first + (worker + stride - first % stride) % stride;
  for (unsigned channel = begin; channel < end; channel += stride)
    if (!::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
        ring::channels(a.gin_window_base, a.gin_rings)[channel].rx_terminal)
        .load(::cuda::memory_order_acquire)) return false;
  return true;
}

template <class Args>
__device__ __forceinline__ bool claim_receive(const Args &a, unsigned &cursor,
                                              Transfer &task, bool dispatch,
                                              unsigned max_probes = 0U) {
  const auto first = dispatch ? 0U : a.gin_rings.dispatch_channels;
  const auto count =
      dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.return_channels;
  const ncclGin gin(a.gin_dev_comm, 0);
  const unsigned worker = blockIdx.x - a.gin_receiver_base;
  const unsigned stride = a.gin_receiver_count ? a.gin_receiver_count : 1U;
  const unsigned offset = a.gin_receiver_count
      ? (worker + stride - first % stride) % stride : 0U;
  const unsigned owned = offset < count ? 1U + (count - 1U - offset) / stride : 0U;
  const unsigned probes = max_probes == 0U ? owned : min(owned, max_probes);
  for (unsigned probe = 0; probe < probes; ++probe) {
    const auto channel = first + offset + (cursor++ % owned) * stride;
    auto &state = ring::channels(a.gin_window_base, a.gin_rings)[channel];
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
            state.rx_terminal)
            .load(::cuda::memory_order_acquire))
      continue;
    unsigned long long sequence;
    if (a.gin_receiver_count) {
      sequence = state.claimed;
      const auto slot = unsigned(sequence % ring::kRingDepth);
      if (!ring::reached(gin.readSignal(a.gin_window,
              a.gin_rings.slot_control(channel, slot) + offsetof(ring::ChunkSlot, ready_signal)),
              sequence / ring::kRingDepth + 1ULL, 64U)) continue;
    } else if (!ring::claim(gin, a.gin_window, a.gin_window_base, a.gin_rings, channel,
                           sequence)) continue;
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const auto *header = reinterpret_cast<const ring::ChunkHeader *>(
        static_cast<const char *>(a.gin_window_base) +
        a.gin_rings.payload(channel, sequence, true));
    if (!Hooks<Args>::receive(a, channel, header))
      continue;
    if (a.gin_receiver_count)
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(state.claimed)
          .store(sequence + 1ULL, ::cuda::memory_order_relaxed);
    task = Transfer{channel, header->rows, header->terminal, sequence, 0};
    return true;
  }
  task.channel = kNoChannel;
  return false;
}

template <class Args>
__device__ __forceinline__ void finish_receive(const Args &a,
                                               const Transfer &task) {
  if (a.gin_receiver_count) {
    const auto slot = unsigned(task.sequence % ring::kRingDepth);
    const auto gin = gin_for_cta(a, blockIdx.x);
    gin.signal(ncclTeamWorld(gin.comm), peer(a, task.channel),
        ncclGin_StrongVASignalAdd{a.gin_window,
            a.gin_rings.slot_control(remote_channel(a, task.channel), slot) +
                offsetof(ring::ChunkSlot, credit_signal), 1},
        ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_device,
        ::cuda::thread_scope_system);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        ring::channels(a.gin_window_base, a.gin_rings)[task.channel].slots[slot].acknowledged)
        .store(task.sequence / ring::kRingDepth + 1ULL, ::cuda::memory_order_release);
  } else {
    ring::consumed(a.gin_window_base, a.gin_rings, task.channel, task.sequence);
  }
  if (task.terminal)
    ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
        ring::channels(a.gin_window_base, a.gin_rings)[task.channel]
            .rx_terminal)
        .store(1U, ::cuda::memory_order_release);
}

template <class Args>
__device__ __forceinline__ bool choose_dispatch(const Args &a, unsigned context,
                                                unsigned &cursor,
                                                Transfer &task) {
  const auto nodes = a.dispatch.identity.num_nodes - 1U;
  auto *states = ring::channels(a.gin_window_base, a.gin_rings);
  const unsigned per_node = context < ring::kDispatchChannels
      ? 1U + (ring::kDispatchChannels - 1U - context) / a.gin_context_count : 0U;
  const unsigned count = nodes * per_node;
  for (unsigned probe = 0; probe < count; ++probe) {
    const auto index = cursor++ % count;
    const auto channel = (index / per_node) * ring::kDispatchChannels +
                         context + (index % per_node) * a.gin_context_count;
    auto &state = states[channel];
    const ncclGin gin(a.gin_dev_comm, context);
    if (state.tx_terminal ||
        !ring::can_send(gin, a.gin_window, a.gin_window_base, a.gin_rings, channel, state.sent))
      continue;
    const auto count =
        state.tx_cursor < state.tx_end
            ? min(1ULL * ring::kChunkRows, state.tx_end - state.tx_cursor)
            : 0ULL;
    task = Transfer{channel, unsigned(count), count == 0ULL, state.sent,
                    state.tx_cursor};
    return true;
  }
  return false;
}

template <class Args>
__device__ __forceinline__ bool choose_return(const Args &a, unsigned context,
                                              unsigned &cursor,
                                              Transfer &task) {
  const auto count = (a.dispatch.identity.num_nodes - 1U) * ring::kReturnPorts;
  auto *states = ring::channels(a.gin_window_base, a.gin_rings);
  for (unsigned probe = 0; probe < count; ++probe) {
    const auto index = cursor++ % count;
    const auto channel =
        a.gin_rings.dispatch_channels +
        (index / ring::kReturnPorts * a.local_world_size + context) *
            ring::kReturnPorts +
        index % ring::kReturnPorts;
    auto &state = states[channel];
    const ncclGin gin(a.gin_dev_comm, data_context(a, channel));
    if (state.tx_terminal ||
        !ring::can_send(gin, a.gin_window, a.gin_window_base, a.gin_rings, channel, state.sent))
      continue;
    auto &queue = return_queue_detail::queues(a)[1U + peer(a, channel)];
    const auto remaining = queue.capacity - queue.head;
    const auto rows = unsigned(min(remaining, 1ULL * ring::kChunkRows));
    if (return_queue_detail::ready_rows(a, queue, queue.head, rows) != rows)
      continue;
    task = Transfer{channel, rows, rows == 0U, state.sent,
                    queue.begin + queue.head};
    queue.head += rows; // One RDMA CTA owns both ports of this peer queue.
    return true;
  }
  return false;
}

__device__ __forceinline__ unsigned transport_thread() {
  return threadIdx.x < 32U ? threadIdx.x : threadIdx.x - 32U;
}
__device__ __forceinline__ void transport_sync() {
  asm volatile("bar.sync 1, %0;" ::"r"(blockDim.x - 32U) : "memory");
}

template <class Args>
__device__ __forceinline__ void
pack_dispatch(const Args &a, const Transfer &task,
              const transport::BFloat16 *input, const float *gates) {
  auto *chunk = static_cast<char *>(a.gin_window_base) +
                a.gin_rings.payload(task.channel, task.sequence, false);
  if (!threadIdx.x)
    *reinterpret_cast<ring::ChunkHeader *>(chunk) =
        ring::ChunkHeader{task.rows, task.terminal, 0};
  const auto warp = transport_thread() / 32U, lane = threadIdx.x % 32U;
  for (unsigned row = warp; row < task.rows; row += blockDim.x / 32U - 1U) {
    const auto packet = a.dispatch_waves.packet_order[task.first + row];
    const auto token = a.dispatch.packet_headers[packet].local_token_id;
    const auto begin = a.dispatch.packet_fanout_offsets[packet];
    const auto end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    auto *record = chunk + sizeof(ring::ChunkHeader) +
                   row * a.gin_rings.dispatch_row_bytes;
    if (!lane)
      *reinterpret_cast<ring::DispatchRow *>(record) =
          ring::DispatchRow{token, unsigned(end - begin), 0};
    auto *fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
        record + sizeof(ring::DispatchRow));
    for (auto item = begin + lane; item < end; item += 32U) {
      const auto target = a.dispatch.packet_fanout[item];
      fanout[item - begin] = ForwardDispatchFanoutRecord{
          target.dst_row,
          0,
          target.topk_slot,
          static_cast<std::uint16_t>(target.destination_local_rank),
          static_cast<std::uint16_t>(target.expert_id),
          target.reserved,
          gates ? gates[token * a.dispatch.identity.top_k + target.topk_slot]
                : 0.0F};
    }
    auto *payload = reinterpret_cast<uint4 *>(
        record + a.gin_rings.dispatch_row_bytes - a.hidden * 2ULL);
    const auto *source =
        reinterpret_cast<const uint4 *>(input + token * a.hidden);
    for (auto v = lane; v < a.hidden / 8ULL; v += 32U)
      payload[v] = source[v];
  }
  __threadfence_system();
  transport_sync();
}

template <class Args>
__device__ __forceinline__ void send(const Args &a, const Transfer &task) {
  const bool dispatch = task.channel < a.gin_rings.dispatch_channels;
  const auto row_bytes =
      dispatch ? a.gin_rings.dispatch_row_bytes : a.gin_rings.return_row_bytes;
  const ncclGin gin(a.gin_dev_comm, data_context(a, task.channel));
  ring::send(gin, a.gin_window, a.gin_window_base, a.gin_rings, task.channel,
             remote_channel(a, task.channel), peer(a, task.channel),
             sizeof(ring::ChunkHeader) + task.rows * row_bytes);
  auto &state = ring::channels(a.gin_window_base, a.gin_rings)[task.channel];
  if (dispatch)
    state.tx_cursor += ring::kDispatchChannels * ring::kChunkRows;
  state.tx_terminal = task.terminal;
}
} // namespace activation_ring_detail
