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
__device__ __forceinline__ unsigned owner(const Args &a, unsigned channel) {
  return channel < a.gin_rings.dispatch_channels
             ? channel % ring::kDispatchChannels
             : peer(a, channel) % a.local_world_size;
}

template <class Args>
__device__ __forceinline__ unsigned data_context(const Args &a,
                                                 unsigned channel) {
  if (channel < a.gin_rings.dispatch_channels)
    return channel % ring::kDispatchChannels;
  const auto logical =
      (peer(a, channel) % a.local_world_size + a.rank) % a.gin_context_count;
  return a.gin_context_count + ring::kReturnPorts * logical +
         (channel - a.gin_rings.dispatch_channels) % ring::kReturnPorts;
}

// Durable sequences and per-slot signals survive FWD, BWD, arena growth and
// graph replay. Only the finite work interval changes at an operation boundary.
template <class Args> __global__ void prepare(Args a) {
  const auto channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= a.gin_rings.channels())
    return;
  auto &state = ring::channels(a.gin_window_base, a.gin_rings)[channel];
  state.tx_terminal = state.rx_terminal = 0U;
  state.tx_cursor = state.tx_end = 0ULL;
  if (channel >= a.gin_rings.dispatch_channels)
    return;
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

struct Transfer {
  unsigned channel = kNoChannel;
  unsigned rows = 0;
  unsigned terminal = 0;
  unsigned long long sequence = 0;
  unsigned long long first = 0;
};

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

template <class Args>
__device__ __forceinline__ bool claim_receive(const Args &a, unsigned &cursor,
                                              Transfer &task, bool dispatch) {
  const auto first = dispatch ? 0U : a.gin_rings.dispatch_channels;
  const auto count =
      dispatch ? a.gin_rings.dispatch_channels : a.gin_rings.return_channels;
  const ncclGin gin(a.gin_dev_comm, 0);
  for (unsigned probe = 0; probe < count; ++probe) {
    const auto channel = first + (cursor++ % count);
    auto &state = ring::channels(a.gin_window_base, a.gin_rings)[channel];
    if (::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
            state.rx_terminal)
            .load(::cuda::memory_order_acquire))
      continue;
    unsigned long long sequence;
    if (!ring::claim(gin, a.gin_window, a.gin_window_base, a.gin_rings, channel,
                     sequence))
      continue;
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const auto *header = reinterpret_cast<const ring::ChunkHeader *>(
        static_cast<const char *>(a.gin_window_base) +
        a.gin_rings.payload(channel, sequence, true));
    task = Transfer{channel, header->rows, header->terminal, sequence, 0};
    return true;
  }
  task.channel = kNoChannel;
  return false;
}

template <class Args>
__device__ __forceinline__ void finish_receive(const Args &a,
                                               const Transfer &task) {
  ring::consumed(a.gin_window_base, a.gin_rings, task.channel, task.sequence);
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
  for (unsigned probe = 0; probe < nodes; ++probe) {
    const auto channel = (cursor++ % nodes) * ring::kDispatchChannels + context;
    auto &state = states[channel];
    const ncclGin gin(a.gin_dev_comm, context);
    if (state.tx_terminal ||
        !ring::can_send(gin, a.gin_window, a.gin_rings, channel, state.sent))
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
        !ring::can_send(gin, a.gin_window, a.gin_rings, channel, state.sent))
      continue;
    auto &queue = return_queue_detail::queues(a)[1U + peer(a, channel)];
    const auto ready =
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            queue.confirmed)
            .load(::cuda::memory_order_acquire);
    if (ready - queue.head < ring::kChunkRows && ready != queue.capacity)
      continue;
    const auto rows =
        unsigned(min(ready - queue.head, 1ULL * ring::kChunkRows));
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
