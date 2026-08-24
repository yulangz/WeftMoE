#include "cuda/activation_transport_internal.hpp"

#include <cuda_runtime.h>
#include <nccl.h>
#if __has_include(<nccl_device/core.h>)
#include <nccl_device/core.h>
#include <nccl_device/impl/comm__types.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

#include "cuda/cuda_internal.hpp"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
namespace {

constexpr std::size_t kAlign = 256;

[[noreturn]] void transport_error(NotifyErrorCode code, const char *phase,
                                  const char *field, std::uint32_t rank,
                                  const std::string &expected,
                                  const std::string &actual) {
  throw NotifyDispatchError(
      code, std::string("phase=") + phase + ", field=" + field +
                ", rank=" + std::to_string(rank) + ", expected=" + expected +
                ", actual=" + actual);
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b,
                          const char *field) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "uint64 product", "overflow");
  return a * b;
}

std::size_t checked_size(std::uint64_t value, const char *field) {
  if (value > std::numeric_limits<std::size_t>::max())
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "value <= size_t max", std::to_string(value));
  return static_cast<std::size_t>(value);
}

std::size_t append(std::size_t &cursor, std::uint64_t count,
                   std::size_t width, const char *field) {
  if (cursor > std::numeric_limits<std::size_t>::max() - (kAlign - 1U))
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "alignable offset", "overflow");
  const auto aligned = (cursor + kAlign - 1U) & ~(kAlign - 1U);
  const auto bytes = checked_size(checked_mul(count, width, field), field);
  if (bytes > std::numeric_limits<std::size_t>::max() - aligned)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "bounded arena", "overflow");
  cursor = aligned + bytes;
  return aligned;
}

struct TransportDeviceStatus {
  unsigned long long packet_claim_cursor;
  unsigned long long packets_done;
  unsigned long long fanout_done;
  unsigned long long synthetic_done;
  unsigned long long combine_done;
  unsigned long long incoming_done;
  unsigned long long token_claim_cursor;
  unsigned long long reduced_tokens;
  unsigned long long credit_stalls;
  unsigned long long peak_ring_slots;
  unsigned long long dispatch_put_ticks;
  unsigned long long relay_ticks;
  unsigned long long synthetic_ticks;
  unsigned long long raw_cross_node_payload_bytes;
  unsigned long long deduplicated_cross_node_payload_bytes;
  unsigned long long start_clock;
  unsigned long long dispatch_complete_clock;
  unsigned long long complete_clock;
  unsigned int error;
  unsigned int error_packet;
  unsigned int error_field;
  unsigned int reserved;
};

struct LocalP2pRouteHeader {
  std::uint32_t schema_version;
  std::uint32_t source_rank;
  std::uint32_t destination_rank;
  std::uint32_t channel;
  std::uint64_t generation;
  std::uint64_t source_epoch;
  std::uint64_t sequence;
  std::uint64_t route_index;
  std::uint64_t local_token_id;
  std::uint64_t dst_row;
  std::uint32_t expert_id;
  std::uint32_t topk_slot;
};
static_assert(sizeof(LocalP2pRouteHeader) == 72);

struct Layout {
  std::size_t input = 0;
  std::size_t ring_headers = 0;
  std::size_t ring_payload = 0;
  std::size_t ring_ready_sequence = 0;
  std::size_t link_next_sequence = 0;
  std::size_t link_credits = 0;
  std::size_t ingress_headers = 0;
  std::size_t ingress_ready_sequence = 0;
  std::size_t ingress_next_sequence = 0;
  std::size_t ingress_published_sequence = 0;
  std::size_t ingress_credits = 0;
  std::size_t outgoing_route_sequence = 0;
  std::size_t return_ready_epoch = 0;
  std::size_t peer_bases = 0;
  std::size_t p2p_barrier_send = 0;
  std::size_t p2p_barrier_receive = 0;
  std::size_t packet_claim = 0;
  std::size_t canonical_rows = 0;
  std::size_t contribution = 0;
  std::size_t combine_return = 0;
  std::size_t combined = 0;
  std::size_t status = 0;
  std::size_t total = 0;
};

Layout make_layout(const ActivationTransportPoolConfig &config) {
  if (config.max_tokens_per_rank == 0 || config.max_hidden_size == 0 ||
      config.max_top_k == 0 || config.max_world_size == 0 ||
      config.max_nodes == 0 || config.channels == 0 ||
      config.ring_slots_per_source_channel == 0 ||
      config.max_inflight_generations == 0 ||
      config.max_destination_rows_per_rank == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-query", "capacity", 0,
                    "all capacities > 0", "zero field");
  Layout layout;
  std::size_t cursor = 0;
  const auto routes = checked_mul(config.max_tokens_per_rank, config.max_top_k,
                                  "max routes");
  const auto links = checked_mul(config.max_nodes, config.channels, "links");
  const auto ring_slots = checked_mul(
      checked_mul(config.max_nodes, config.channels, "ring links"),
      config.ring_slots_per_source_channel, "ring slots");
  const auto rank_links = checked_mul(config.max_world_size, config.channels,
                                      "rank links");
  const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "ingress slots");
  layout.input = append(cursor,
                        checked_mul(config.max_tokens_per_rank,
                                    config.max_hidden_size, "input elements"),
                        sizeof(transport::BFloat16), "input");
  layout.ring_headers = append(cursor, ring_slots,
                               sizeof(transport::TransportSlotHeader),
                               "ring headers");
  layout.ring_payload = append(
      cursor, checked_mul(ring_slots, config.max_hidden_size,
                          "ring payload elements"),
      sizeof(transport::BFloat16), "ring payload");
  layout.ring_ready_sequence =
      append(cursor, ring_slots, sizeof(std::uint64_t), "ring ready sequence");
  layout.link_next_sequence =
      append(cursor, links, sizeof(std::uint64_t), "link next sequence");
  layout.link_credits =
      append(cursor, links, sizeof(std::uint32_t), "link credits");
  layout.ingress_headers = append(cursor, ingress_slots,
                                  sizeof(LocalP2pRouteHeader),
                                  "P2P ingress headers");
  layout.ingress_ready_sequence = append(
      cursor, ingress_slots, sizeof(std::uint64_t), "P2P ingress ready");
  layout.ingress_next_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P ingress next");
  layout.ingress_published_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P ingress published");
  layout.ingress_credits = append(cursor, rank_links, sizeof(std::uint32_t),
                                  "P2P ingress credits");
  layout.outgoing_route_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P outgoing sequence");
  layout.return_ready_epoch = append(cursor, routes, sizeof(std::uint64_t),
                                     "P2P combine ready epoch");
  layout.peer_bases = append(cursor, config.max_world_size, sizeof(void *),
                             "P2P peer base table");
  layout.p2p_barrier_send =
      append(cursor, 2, sizeof(std::uint64_t), "P2P barrier send");
  layout.p2p_barrier_receive = append(cursor,
                                      checked_mul(config.max_world_size, 2,
                                                  "P2P barrier words"),
                                      sizeof(std::uint64_t),
                                      "P2P barrier receive");
  layout.packet_claim = append(cursor, routes, sizeof(std::uint32_t),
                               "packet claim bitmap");
  layout.canonical_rows = append(
      cursor, checked_mul(config.max_destination_rows_per_rank,
                          config.max_hidden_size, "canonical rows"),
      sizeof(transport::BFloat16), "canonical rows");
  const auto route_elements =
      checked_mul(routes, config.max_hidden_size, "route elements");
  layout.contribution = append(cursor, route_elements,
                               sizeof(transport::BFloat16), "contribution");
  layout.combine_return = append(cursor, route_elements,
                                 sizeof(transport::BFloat16), "combine return");
  layout.combined = append(
      cursor, checked_mul(config.max_tokens_per_rank, config.max_hidden_size,
                          "combined elements"),
      sizeof(transport::BFloat16), "combined");
  layout.status = append(cursor, 1, sizeof(TransportDeviceStatus), "status");
  layout.total = cursor;
  return layout;
}

enum class SlotState : std::uint8_t { Available, InUse, Quarantined };

struct PoolSlot {
  void *device = nullptr;
  TransportDeviceStatus *host_status = nullptr;
  std::uint64_t *host_cancel_epoch = nullptr;
  std::uint64_t *device_cancel_epoch = nullptr;
  cudaEvent_t producer_ready = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t dispatch_complete = nullptr;
  cudaEvent_t complete = nullptr;
  SlotState state = SlotState::Available;
  std::uint64_t epoch = 0;
  bool completion_recorded = false;
};

template <typename T>
T *at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}
template <typename T>
const T *at(const void *base, std::size_t offset) {
  return reinterpret_cast<const T *>(
      static_cast<const unsigned char *>(base) + offset);
}

struct ActivationPoolState {
  ActivationTransportPoolConfig config;
  Layout layout;
  std::vector<std::byte> session_identity;
  std::uint32_t world = 0;
  CudaExecution execution;
  std::uint32_t device = 0;
  std::mutex mutex;
  std::vector<PoolSlot> slots;
  std::uint64_t epoch_high = 0;
  cudaStream_t cancel_stream = nullptr;

  ~ActivationPoolState() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    for (auto &slot : slots) {
      if (slot.completion_recorded && slot.complete)
        (void)cudaEventSynchronize(slot.complete);
      if (slot.producer_ready)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.producer_ready)
        (void)cudaEventDestroy(slot.producer_ready);
      if (slot.start)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.start)
        (void)cudaEventDestroy(slot.start);
      if (slot.dispatch_complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.dispatch_complete)
        (void)cudaEventDestroy(slot.dispatch_complete);
      if (slot.complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.complete)
        (void)cudaEventDestroy(slot.complete);
      if (slot.host_status)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_status)
        (void)cudaFreeHost(slot.host_status);
      if (slot.host_cancel_epoch)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_cancel_epoch)
        (void)cudaFreeHost(slot.host_cancel_epoch);
      if (slot.device_cancel_epoch)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device_cancel_epoch)
        (void)cudaFree(slot.device_cancel_epoch);
      if (slot.device)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device)
        (void)cudaFree(slot.device);
    }
    if (cancel_stream)
      (void)cudaStreamDestroy(cancel_stream);
  }
};

struct SlotLease {
  std::shared_ptr<ActivationPoolState> pool;
  std::size_t index = 0;
  std::uint64_t epoch = 0;
  bool quarantined = false;

  ~SlotLease() noexcept {
    if (!pool || index >= pool->slots.size())
      return;
    auto &slot = pool->slots[index];
    bool complete = true;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    if (slot.completion_recorded && slot.complete)
      complete = cudaEventSynchronize(slot.complete) == cudaSuccess;
    std::lock_guard<std::mutex> lock(pool->mutex);
    if (slot.epoch != epoch)
      return;
    slot.state = (!complete || quarantined) ? SlotState::Quarantined
                                           : SlotState::Available;
    slot.completion_recorded = false;
  }
};

struct SlotFailureGuard {
  std::shared_ptr<SlotLease> lease;
  bool committed = false;
  ~SlotFailureGuard() noexcept {
    if (lease && !committed)
      lease->quarantined = true;
  }
};

std::shared_ptr<SlotLease> acquire_slot(
    const std::shared_ptr<ActivationPoolState> &pool, std::uint64_t generation,
    std::uint32_t rank) {
  std::lock_guard<std::mutex> lock(pool->mutex);
  for (std::size_t i = 0; i < pool->slots.size(); ++i) {
    auto &slot = pool->slots[i];
    if (slot.state != SlotState::Available)
      continue;
    slot.state = SlotState::InUse;
    slot.epoch = ++pool->epoch_high;
    slot.completion_recorded = false;
    auto lease = std::make_shared<SlotLease>();
    lease->pool = pool;
    lease->index = i;
    lease->epoch = slot.epoch;
    (void)generation;
    return lease;
  }
  throw NotifyDispatchError(
      NotifyErrorCode::ResourceExhausted,
      "phase=activation-pool-acquire, field=slots, rank=" +
          std::to_string(rank) + ", expected>=1 available, actual=0");
}

__device__ std::uint16_t float_to_bf16_bits(float value) {
  auto bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ float bf16_bits_to_float(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ unsigned long long device_globaltimer_ns() {
  unsigned long long value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ transport::BFloat16 synthetic_value(
    transport::BFloat16 input, std::uint32_t expert,
    std::uint64_t hidden_index) {
  const auto expert_term = static_cast<float>((expert % 251U) + 1U) / 128.0F;
  const auto lane_term = static_cast<float>(hidden_index % 17U) / 1024.0F;
  return transport::BFloat16{
      float_to_bf16_bits(bf16_bits_to_float(input.bits) + expert_term +
                         lane_term)};
}

struct TransportKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
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
};

__global__ void activation_transport_persistent_kernel(TransportKernelArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto route_count = a.dispatch.sizes.route_count;
  const auto packet_count = a.dispatch.sizes.packet_count;
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
        const auto contribution = synthetic_value(payload, target.expert_id, h);
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
        sum += bf16_bits_to_float(a.combine_return[route * a.hidden + h].bits);
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

template <typename T>
__device__ T *peer_at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

struct LocalP2pKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
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
  std::size_t return_ready_offset;
  std::size_t canonical_rows_offset;
  std::size_t contribution_offset;
  std::size_t combine_return_offset;
  std::size_t combined_offset;
  std::size_t status_offset;
  const unsigned long long *cancel_epoch;
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

__global__ void local_p2p_persistent_kernel(LocalP2pKernelArgs a) {
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
         packet < a.dispatch.sizes.packet_count && !local_p2p_cancelled(a);
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
        if (route >= a.dispatch.sizes.route_count) {
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
    for (std::uint64_t row = lane; row < a.dispatch.sizes.incoming_row_count;
         row += blockDim.x)
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
      const auto incoming = header.dst_row < a.dispatch.sizes.incoming_row_count
                                ? a.dispatch.incoming_rows[header.dst_row]
                                : DeviceIncomingRowRecord{};
      if (header.schema_version != transport::kActivationTransportSchemaVersion ||
          header.source_rank != source || header.destination_rank != a.rank ||
          header.generation != a.dispatch.identity.generation ||
          header.sequence != sequence ||
          header.dst_row >= a.dispatch.sizes.incoming_row_count ||
          incoming.source_rank != source || incoming.expert_id != header.expert_id) {
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
            header.expert_id, hidden);
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
              a.dispatch.sizes.packet_count ||
          atomicAdd(&status->incoming_done, 0ULL) <
              a.dispatch.sizes.incoming_row_count ||
          atomicAdd_system(&status->combine_done, 0ULL) <
              a.dispatch.sizes.route_count)) {
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
        sum += bf16_bits_to_float(
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

} // namespace

struct ActivationTransportResourcePool::Impl {
  std::shared_ptr<ActivationPoolState> state;
};

struct SelectedActivationLease<transport::BFloat16>::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  const transport::BFloat16 *pointer = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken producer_execution;
  std::uint64_t generation = 0;
  std::uint64_t tokens = 0;
  std::uint64_t hidden = 0;
  std::uint32_t device = 0;
  std::uint32_t layout = kSelectedActivationLayoutVersion;
  bool ready_recorded = false;
};

struct LocalP2pBackendState;

struct DeviceActivationTransportHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  ActivationTransportExpectation identity;
  ActivationTransportDeviceAudit audit;
  ActivationTransportTimings timings;
  std::uint64_t route_count = 0;
  std::uint64_t destination_rows = 0;
  std::shared_ptr<LocalP2pBackendState> local_p2p_backend;
};

struct DeviceActivationTransport::Impl {
  CollectiveSession session;
  CudaExecution execution;
  ActivationTransportResourcePool pool;
};

struct LocalP2pBackendState {
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<internal::NcclSessionState> nccl;
  NcclSession session;
  CudaExecution execution;
  std::vector<void *> peer_slot_bases;
  std::vector<std::uint8_t> owns_mapping;

  ~LocalP2pBackendState() noexcept {
    if (!pool)
      return;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    for (std::size_t index = 0; index < peer_slot_bases.size(); ++index) {
      if (owns_mapping[index] != 0 && peer_slot_bases[index])
        (void)cudaIpcCloseMemHandle(peer_slot_bases[index]);
    }
  }
};

struct LocalCudaP2pActivationTransport::Impl {
  std::shared_ptr<LocalP2pBackendState> backend;
};

constexpr std::uint64_t kLocalP2pBootstrapMagic = 0x31504350454f4d4dULL;
constexpr std::uint64_t kLocalP2pBootstrapSchema = 1;
constexpr std::size_t kLocalP2pBootstrapHeaderBytes = 96;

void put_u64_le(std::vector<std::byte> &wire, std::size_t offset,
                std::uint64_t value) {
  if (offset > wire.size() || wire.size() - offset < sizeof(value))
    transport_error(NotifyErrorCode::ArithmeticOverflow,
                    "local-p2p-bootstrap", "wire offset", 0,
                    "8 writable bytes", "out of range");
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    wire[offset + byte] =
        static_cast<std::byte>((value >> (byte * 8U)) & 0xffU);
}

std::uint64_t get_u64_le(const std::byte *wire, std::size_t bytes,
                         std::size_t offset, std::uint32_t rank,
                         const char *field) {
  if (offset > bytes || bytes - offset < sizeof(std::uint64_t))
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "local-p2p-bootstrap", field, rank, "8 readable bytes",
                    "truncated");
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(wire[offset + byte]) << (byte * 8U);
  return value;
}

std::array<std::uint64_t, 2>
session_summary(const std::vector<std::byte> &identity) {
  std::array<std::uint64_t, 2> hash{1469598103934665603ULL,
                                    1099511628211ULL};
  for (const auto byte : identity) {
    const auto value = static_cast<std::uint8_t>(byte);
    hash[0] = (hash[0] ^ value) * 1099511628211ULL;
    hash[1] = (hash[1] ^ static_cast<std::uint8_t>(value + 0x9dU)) *
              1469598103934665603ULL;
  }
  return hash;
}

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

ActivationTransportResourcePool make_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config) {
  if (!session || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-create",
                    "session/execution", 0, "valid", "empty");
  if (config.max_world_size < session.world_size())
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "max_world_size", 0,
                    ">= session world", std::to_string(config.max_world_size));
  auto state = std::make_shared<ActivationPoolState>();
  state->config = config;
  state->layout = make_layout(config);
  state->session_identity = session.stable_identity_bytes();
  state->world = session.world_size();
  state->execution = execution;
  state->device = execution.device_ordinal();
  internal::throw_cuda("transport-pool-create",
                       cudaSetDevice(static_cast<int>(state->device)), 0,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
  state->slots.resize(config.max_inflight_generations);
  try {
    internal::throw_cuda(
        "transport-pool-create",
        cudaStreamCreateWithFlags(&state->cancel_stream, cudaStreamNonBlocking),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaStreamCreate(cancel)");
    for (auto &slot : state->slots) {
      internal::throw_cuda("transport-pool-create",
                           cudaMalloc(&slot.device, state->layout.total), 0,
                           session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaMalloc(transport slot)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(transport status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_cancel_epoch),
                        sizeof(std::uint64_t),
                        cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      *slot.host_cancel_epoch = 0;
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(reinterpret_cast<void **>(&slot.device_cancel_epoch),
                     sizeof(std::uint64_t)),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      auto create_event = [&](cudaEvent_t *event, const char *name) {
        internal::throw_cuda(
            "transport-pool-create",
            cudaEventCreateWithFlags(event, cudaEventDisableTiming), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      };
      create_event(&slot.producer_ready, "cudaEventCreate(producer-ready)");
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.dispatch_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(dispatch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    }
  } catch (...) {
    throw;
  }
  auto impl = std::make_shared<ActivationTransportResourcePool::Impl>();
  impl->state = std::move(state);
  return ActivationTransportResourcePool(std::move(impl));
}

ActivationTransportPoolUsage ActivationTransportResourcePool::usage() const {
  if (!impl_ || !impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-usage",
                    "pool", 0, "valid", "empty");
  ActivationTransportPoolUsage result;
  auto state = impl_->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (const auto &slot : state->slots) {
    if (slot.state == SlotState::Available)
      ++result.slots_available;
    else if (slot.state == SlotState::InUse)
      ++result.slots_in_use;
    else
      ++result.slots_quarantined;
  }
  result.device_bytes = checked_mul(
      state->layout.total + sizeof(std::uint64_t), state->slots.size(),
      "pool device bytes");
  result.pinned_bytes = checked_mul(
      sizeof(TransportDeviceStatus) + sizeof(std::uint64_t),
      state->slots.size(), "pool pinned bytes");
  result.event_count = checked_mul(4, state->slots.size(), "pool events");
  result.epoch_high_watermark = state->epoch_high;
  return result;
}

std::uint64_t BFloat16ActivationLease::num_tokens() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->tokens;
}
std::uint64_t BFloat16ActivationLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t BFloat16ActivationLease::element_count() const {
  return checked_mul(num_tokens(), hidden_size(), "activation elements");
}
std::uint64_t BFloat16ActivationLease::generation() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->generation;
}
std::uint32_t BFloat16ActivationLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->device;
}
std::uint32_t BFloat16ActivationLease::layout_version() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->layout;
}

BFloat16ActivationLease ActivationTransportBridge::upload_for_testing(
    const std::vector<transport::BFloat16> &host, std::uint64_t num_tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!context || !execution || !pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "activation-test-upload", "context/execution/pool", 0,
                    "valid", "empty");
  auto state = pool.impl_->state;
  if (!context.execution().same_instance(execution.token()) ||
      context.session().stable_identity_bytes() != state->session_identity ||
      execution.device_ordinal() != state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "activation-test-upload", "provenance",
                    context.calling_rank(), "exact pool session/execution/device",
                    "mismatch");
  if (num_tokens > state->config.max_tokens_per_rank ||
      hidden == 0 || hidden > state->config.max_hidden_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "activation-test-upload", "shape", context.calling_rank(),
                    "within pool capacities", "exceeded/zero");
  const auto count = checked_mul(num_tokens, hidden, "activation upload count");
  if (host.size() != checked_size(count, "activation upload count"))
    transport_error(NotifyErrorCode::InvalidRouteShape,
                    "activation-test-upload", "host.size",
                    context.calling_rank(), std::to_string(count),
                    std::to_string(host.size()));
  auto lease = acquire_slot(state, context.session().generation(),
                            context.calling_rank());
  auto &slot = state->slots[lease->index];
  auto exec = internal::checked_execution_state(execution);
  if (count != 0) {
    const auto bytes = checked_size(checked_mul(count, sizeof(transport::BFloat16),
                                                "activation upload bytes"),
                                    "activation upload bytes");
    internal::throw_cuda(
        "activation-test-upload",
        cudaMemcpyAsync(at<void>(slot.device, state->layout.input), host.data(),
                        bytes, cudaMemcpyHostToDevice, exec->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemcpyAsync(test activation)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  }
  internal::throw_cuda(
      "activation-test-upload", cudaEventRecord(slot.producer_ready, exec->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(producer-ready)");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>();
  impl->slot = std::move(lease);
  impl->pool = state;
  impl->pointer = at<const transport::BFloat16>(slot.device, state->layout.input);
  impl->session_identity = context.session().stable_identity_bytes();
  impl->producer_execution = execution.token();
  impl->generation = context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden;
  impl->device = execution.device_ordinal();
  impl->ready_recorded = true;
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16ActivationLease testing::upload_bfloat16_activations_for_testing(
    const std::vector<transport::BFloat16> &host, std::uint64_t num_tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  return ActivationTransportBridge::upload_for_testing(
      host, num_tokens, hidden, context, execution, pool);
}

DeviceActivationTransport::DeviceActivationTransport(
    const CollectiveSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-create",
                    "session/execution/pool", 0, "valid", "empty");
  if (session.stable_identity_bytes() != pool.impl_->state->session_identity ||
      session.world_size() != pool.impl_->state->world ||
      execution.device_ordinal() != pool.impl_->state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-create",
                    "pool provenance", 0, "matching session/world/device",
                    "mismatch");
  impl_ = std::make_shared<Impl>();
  impl_->session = session;
  impl_->execution = execution;
  impl_->pool = pool;
}

LocalCudaP2pActivationTransport::LocalCudaP2pActivationTransport(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create",
                    "pool", 0, "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend =
      create_local_p2p_backend(session, execution, pool.impl_->state);
}

DeviceActivationTransportHandle
DeviceActivationTransport::run_device_simulation_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config) {
  if (!impl_ || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-run",
                    "transport/dispatch/activation", 0, "valid", "empty");
  const auto &id = dispatch.identity();
  auto state = impl_->pool.impl_->state;
  const auto &activation = *activations.impl_;
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens || activation.hidden != config.hidden_size ||
      activation.device != impl_->execution.device_ordinal() ||
      activation.session_identity != impl_->session.stable_identity_bytes() ||
      !activation.producer_execution.same_instance(impl_->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size ||
      id.num_nodes > state->config.max_nodes)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "transport-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(impl_->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(impl_->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  const auto links = checked_mul(id.num_nodes, config.channels, "actual links");
  const auto ring_slots = checked_mul(
      checked_mul(id.num_nodes, config.channels, "actual ring links"),
      config.ring_slots_per_source_channel, "actual ring slots");
  const auto route_elements = checked_mul(view.sizes.route_count,
                                          config.hidden_size, "route elements");
  const auto combined_elements = checked_mul(id.num_tokens, config.hidden_size,
                                             "combined elements");
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0,
                        checked_size(checked_mul(count, width, name), name),
                        exec->stream),
        id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
  };
  internal::throw_cuda(
      "transport-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  zero_async(layout.ring_headers, ring_slots,
             sizeof(transport::TransportSlotHeader), "ring headers");
  zero_async(layout.ring_ready_sequence, ring_slots, sizeof(std::uint64_t),
             "ring sequences");
  zero_async(layout.link_next_sequence, links, sizeof(std::uint64_t),
             "link sequences");
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits), links,
                      config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "transport-run", cudaGetLastError(), id.calling_rank, id.world_size,
      impl_->session.identity(), id.generation, ExchangeKind::ExpertCounts,
      "initialize transport credits kernel");
  zero_async(layout.packet_claim, view.sizes.packet_count,
             sizeof(std::uint32_t), "packet claims");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "canonical elements"),
             sizeof(transport::BFloat16), "canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "combined");
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "status");
  internal::throw_cuda("transport-run", cudaEventRecord(slot.start, exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(start)");
  int sms = 0;
  internal::throw_cuda(
      "transport-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  TransportKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  args.hidden = config.hidden_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
  args.ring_headers = at<transport::TransportSlotHeader>(slot.device,
                                                          layout.ring_headers);
  args.ring_payload = at<transport::BFloat16>(slot.device, layout.ring_payload);
  args.ring_ready = at<unsigned long long>(slot.device,
                                            layout.ring_ready_sequence);
  args.link_next = at<unsigned long long>(slot.device,
                                           layout.link_next_sequence);
  args.link_credits = at<unsigned int>(slot.device, layout.link_credits);
  args.packet_claim = at<unsigned int>(slot.device, layout.packet_claim);
  args.canonical_rows = at<transport::BFloat16>(slot.device,
                                                 layout.canonical_rows);
  args.contribution = at<transport::BFloat16>(slot.device, layout.contribution);
  args.combine_return = at<transport::BFloat16>(slot.device,
                                                 layout.combine_return);
  args.combined = at<transport::BFloat16>(slot.device, layout.combined);
  args.status = at<TransportDeviceStatus>(slot.device, layout.status);
  activation_transport_persistent_kernel<<<std::max(sms, 1), 128, 0,
                                           exec->stream>>>(args);
  internal::throw_cuda("transport-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, impl_->session.identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "activation transport persistent kernel launch");
  internal::throw_cuda(
      "transport-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(dispatch complete)");
  internal::throw_cuda(
      "transport-run",
      cudaMemcpyAsync(slot.host_status, args.status,
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda("transport-run", cudaEventRecord(slot.complete,
                                                         exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda("transport-run", cudaEventSynchronize(slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventSynchronize(transport complete)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "transport-kernel", "descriptor", id.calling_rank,
                    "valid packet/fanout", "device validation error packet=" +
                        std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda("transport-timing",
                       cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventElapsedTime(total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  // The concrete backend object, never a caller-provided enum, owns this
  // label. This path is deliberately and permanently simulation-only.
  result->audit.backend =
      transport::ActivationTransportBackend::DeviceSimulation;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    result->timings.combine_reduce_us =
        static_cast<double>(complete_clock - dispatch_clock) / 1000.0;
  }
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle LocalCudaP2pActivationTransport::run_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=one node and exact local NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "local-p2p-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  const auto route_elements =
      checked_mul(view.sizes.route_count, config.hidden_size,
                  "P2P route elements");
  const auto combined_elements =
      checked_mul(id.num_tokens, config.hidden_size,
                  "P2P combined elements");
  const auto rank_links = checked_mul(id.world_size, config.channels,
                                      "P2P rank links");
  const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "P2P ingress slots");
  const auto node_links = static_cast<std::uint64_t>(config.channels);
  const auto node_ring_slots = checked_mul(
      node_links, config.ring_slots_per_source_channel, "P2P proxy slots");
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "local-p2p-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0,
                        checked_size(checked_mul(count, width, name), name),
                        exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
  };
  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested()
                       ? id.generation
                       : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "local-p2p-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  zero_async(layout.ring_headers, node_ring_slots,
             sizeof(transport::TransportSlotHeader), "P2P proxy headers");
  zero_async(layout.ring_payload,
             checked_mul(node_ring_slots, config.hidden_size,
                         "P2P proxy payload"),
             sizeof(transport::BFloat16), "P2P proxy payload");
  zero_async(layout.ring_ready_sequence, node_ring_slots,
             sizeof(std::uint64_t), "P2P proxy ready");
  zero_async(layout.link_next_sequence, node_links, sizeof(std::uint64_t),
             "P2P proxy sequence");
  zero_async(layout.ingress_headers, ingress_slots,
             sizeof(LocalP2pRouteHeader), "P2P ingress headers");
  zero_async(layout.ingress_ready_sequence, ingress_slots,
             sizeof(std::uint64_t), "P2P ingress ready");
  zero_async(layout.ingress_next_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress next");
  zero_async(layout.ingress_published_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress published");
  zero_async(layout.outgoing_route_sequence, rank_links,
             sizeof(std::uint64_t), "P2P outgoing sequence");
  zero_async(layout.return_ready_epoch, view.sizes.route_count,
             sizeof(std::uint64_t), "P2P return ready");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "P2P canonical rows"),
             sizeof(transport::BFloat16), "P2P canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "P2P contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "P2P combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "P2P combined");
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "P2P status");
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((node_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits),
                      node_links, config.ring_slots_per_source_channel);
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((rank_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.ingress_credits),
                      rank_links, config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P credits");

  // A fixed-width device AllGather is a setup barrier only. It exchanges no
  // activation and ensures every peer ring/credit/sequence region is fully
  // initialized before any persistent worker publishes to it.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, activation.slot->epoch);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
  validate_p2p_barrier_kernel<<<1, 256, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_receive),
      id.world_size, activation.slot->index, activation.slot->epoch,
      at<TransportDeviceStatus>(slot.device, layout.status));
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "validate P2P setup barrier");
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.start, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P start)");
  int sms = 0;
  internal::throw_cuda(
      "local-p2p-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  if (sms <= static_cast<int>(id.world_size))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=resident_workers, rank=" +
            std::to_string(id.calling_rank) + ", expected>world=" +
            std::to_string(id.world_size) + ", actual_sms=" +
            std::to_string(sms));
  LocalP2pKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  args.hidden = config.hidden_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
  args.epoch = activation.slot->epoch;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.local_base = slot.device;
  args.ring_headers_offset = layout.ring_headers;
  args.ring_payload_offset = layout.ring_payload;
  args.ring_ready_offset = layout.ring_ready_sequence;
  args.link_next_offset = layout.link_next_sequence;
  args.link_credits_offset = layout.link_credits;
  args.ingress_headers_offset = layout.ingress_headers;
  args.ingress_ready_offset = layout.ingress_ready_sequence;
  args.ingress_next_offset = layout.ingress_next_sequence;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.ingress_credits_offset = layout.ingress_credits;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.return_ready_offset = layout.return_ready_epoch;
  args.canonical_rows_offset = layout.canonical_rows;
  args.contribution_offset = layout.contribution;
  args.combine_return_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  local_p2p_persistent_kernel<<<sms, 128, 0, exec->stream>>>(args);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "local P2P persistent kernel launch");
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H P2P transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
  while (true) {
    const auto query = cudaEventQuery(slot.complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda(
          "local-p2p-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(P2P complete)");
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >=
            options.deadline_after;
    if (!cancelled && (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, id.generation,
                       __ATOMIC_RELEASE);
      internal::throw_cuda(
          "local-p2p-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "H2D publish cancel epoch");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                sizeof(std::uint64_t));
      internal::throw_cuda(
          "local-p2p-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaStreamSynchronize(cancel)");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  // Keep every process's IPC exports and mappings alive until every peer has
  // observed its persistent worker terminal event. This is especially
  // important on cancellation: a faster peer must not close an IPC mapping
  // while another worker is still polling that peer's credit/signal words.
  internal::throw_nccl(
      "local-p2p-drain",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P terminal drain)");
  internal::throw_cuda(
      "local-p2p-drain", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(P2P terminal drain)");
  consumer.finish();
  if (cancelled || slot.host_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=local-p2p-run, field=cancel_epoch, rank=" +
            std::to_string(id.calling_rank) + ", expected=completion, actual=" +
            (options.cancellation.cancellation_requested() ? "cancelled"
                                                           : "deadline") +
            ", packets=" + std::to_string(slot.host_status->packets_done) +
            ", incoming=" + std::to_string(slot.host_status->incoming_done) +
            ", combine=" + std::to_string(slot.host_status->combine_done));
  if (slot.host_status->error != 0)
    transport_error(
        NotifyErrorCode::TransportProtocolFailure, "local-p2p-kernel",
        "ring/sequence/descriptor", id.calling_rank, "valid",
        "device error field=" + std::to_string(slot.host_status->error_field) +
            ", packet=" + std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(P2P total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->local_p2p_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows =
      state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::LocalCudaP2p;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.peak_ring_slots = slot.host_status->peak_ring_slots;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.dispatch_put_us =
      static_cast<double>(slot.host_status->dispatch_put_ticks) / 1000.0;
  result->timings.relay_us =
      static_cast<double>(slot.host_status->relay_ticks) / 1000.0;
  result->timings.synthetic_return_us =
      static_cast<double>(slot.host_status->synthetic_ticks) / 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    result->timings.combine_reduce_us =
        static_cast<double>(complete_clock - dispatch_clock) / 1000.0;
  }
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

const ActivationTransportExpectation &
DeviceActivationTransportHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}
const ActivationTransportDeviceAudit &
DeviceActivationTransportHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}
const ActivationTransportTimings &
DeviceActivationTransportHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

ActivationTransportValidationDownload ActivationTransportBridge::download_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto hidden = impl.identity.hidden;
  const auto route_elements = checked_mul(impl.route_count, hidden,
                                          "download route elements");
  const auto combined_elements = checked_mul(impl.identity.tokens, hidden,
                                             "download combined elements");
  const auto canonical_elements =
      checked_mul(impl.destination_rows, hidden, "download canonical elements");
  ActivationTransportValidationDownload result;
  result.contribution_by_route.resize(
      checked_size(route_elements, "download route elements"));
  result.combined_tokens.resize(
      checked_size(combined_elements, "download combined elements"));
  result.canonical_expert_rows.resize(
      checked_size(canonical_elements, "download canonical elements"));
  auto copy = [&](void *destination, std::size_t offset, std::uint64_t count,
                  const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-download",
        cudaMemcpy(destination, at<const void>(slot.device, offset),
                   checked_size(checked_mul(count,
                                            sizeof(transport::BFloat16), name),
                                name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
  };
  copy(result.contribution_by_route.data(), impl.pool->layout.contribution,
       route_elements, "download contributions");
  copy(result.combined_tokens.data(), impl.pool->layout.combined,
       combined_elements, "download combined");
  copy(result.canonical_expert_rows.data(), impl.pool->layout.canonical_rows,
       canonical_elements, "download canonical rows");
  result.protocol.dispatch_packets = slot.host_status->packets_done;
  result.protocol.relayed_fanout = slot.host_status->fanout_done;
  result.protocol.synthetic_contributions = slot.host_status->synthetic_done;
  result.protocol.combine_packets = slot.host_status->combine_done;
  result.protocol.reduced_tokens = slot.host_status->reduced_tokens;
  result.protocol.credit_stalls = slot.host_status->credit_stalls;
  result.protocol.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result.protocol.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  return result;
}

ActivationTransportValidationDownload download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  return ActivationTransportBridge::download_for_validation(handle, execution);
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
      "ncclCommWindowRegister",
      "ncclCommWindowDeregister",
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
  result.gin_backend_implemented = false;

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
          const auto cuda_result = cudaMalloc(&allocation, 4096);
          if (cuda_result != cudaSuccess) {
            init_error = std::string("cudaMalloc(window): ") +
                         cudaGetErrorString(cuda_result);
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
      (void)cudaFree(allocation);
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

} // namespace megamoe::cuda
