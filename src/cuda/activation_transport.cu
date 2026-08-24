#include "cuda/activation_transport_internal.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>

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
  unsigned long long token_claim_cursor;
  unsigned long long reduced_tokens;
  unsigned long long credit_stalls;
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

struct Layout {
  std::size_t input = 0;
  std::size_t ring_headers = 0;
  std::size_t ring_payload = 0;
  std::size_t ring_ready_sequence = 0;
  std::size_t link_next_sequence = 0;
  std::size_t link_credits = 0;
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
      if (slot.device)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device)
        (void)cudaFree(slot.device);
    }
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
                                       std::uint64_t count) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < count)
    destination[index] = source[index];
  if (index == 0) {
    __threadfence_system();
    atomicExch_system(signal, 1U);
  }
}

__global__ void initialize_transport_credits_kernel(unsigned int *credits,
                                                    std::uint64_t count,
                                                    unsigned int value) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < count)
    credits[index] = value;
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

struct DeviceActivationTransportHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  ActivationTransportExpectation identity;
  ActivationTransportDeviceAudit audit;
  ActivationTransportTimings timings;
  std::uint64_t route_count = 0;
  std::uint64_t destination_rows = 0;
};

struct DeviceActivationTransport::Impl {
  CollectiveSession session;
  CudaExecution execution;
  ActivationTransportResourcePool pool;
};

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
  result.device_bytes = checked_mul(state->layout.total, state->slots.size(),
                                    "pool device bytes");
  result.pinned_bytes = checked_mul(sizeof(TransportDeviceStatus),
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

DeviceActivationTransportHandle DeviceActivationTransport::run_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    transport::ActivationTransportBackend backend) {
  if (!impl_ || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-run",
                    "transport/dispatch/activation", 0, "valid", "empty");
  if (backend == transport::ActivationTransportBackend::NcclGinDevice)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=transport-run, field=backend, rank=" +
            std::to_string(dispatch.identity().calling_rank) +
            ", expected=runtime GIN symbols and initialized device comm, "
            "actual=unavailable; NCCL 2.28 runtime is not GIN-capable");
  if (backend ==
      transport::ActivationTransportBackend::NcclSendRecvHostEnqueuedFallback)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=transport-run, field=backend, rank=" +
            std::to_string(dispatch.identity().calling_rank) +
            ", expected=explicit fallback coordinator, actual=not attached");
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
  result->audit.backend = backend;
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
  result.gin_header_api = true;
#endif
  result.gin_runtime_symbols =
      dlsym(RTLD_DEFAULT, "ncclGinOutboxCreateRequirement") != nullptr &&
      dlsym(RTLD_DEFAULT, "ncclGinInboxA2ACreateRequirement") != nullptr;
  result.gin_usable = result.gin_header_api && result.gin_runtime_symbols;
  if (!result.gin_usable) {
    std::ostringstream message;
    message << "GIN unavailable: header_api=" << result.gin_header_api
            << ", runtime_symbols=" << result.gin_runtime_symbols
            << ", compiled_nccl=" << result.compiled_nccl_version
            << ", runtime_nccl=" << result.runtime_nccl_version;
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
  cudaStream_t stream = nullptr;
  std::vector<std::uint64_t> host(checked_size(elements, "peer probe elements"));
  for (std::uint64_t i = 0; i < elements; ++i)
    host[checked_size(i, "peer probe index")] = i * 17U + 3U;
  bool ok = false;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(checked_mul(
                                   elements, sizeof(std::uint64_t),
                                   "peer destination bytes"),
                               "peer destination bytes")) != cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess)
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
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    goto cleanup;
  if (cudaMemcpyAsync(source, host.data(),
                      checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                               "peer copy bytes"),
                                   "peer copy bytes"),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess)
    goto cleanup;
  peer_put_signal_kernel<<<
      static_cast<unsigned int>((elements + 255U) / 256U), 256, 0, stream>>>(
      source, destination, signal, elements);
  if (cudaGetLastError() != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess)
    goto cleanup;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  {
    std::vector<std::uint64_t> observed(host.size());
    unsigned int observed_signal = 0;
    if (cudaMemcpy(observed.data(), destination,
                   checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                            "peer observe bytes"),
                                "peer observe bytes"),
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(&observed_signal, signal, sizeof(observed_signal),
                   cudaMemcpyDeviceToHost) == cudaSuccess)
      ok = observed == host && observed_signal == 1U;
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (stream)
    (void)cudaStreamDestroy(stream);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
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
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t complete = nullptr;
  double result = 0.0;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess)
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
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&complete) != cudaSuccess)
    goto cleanup;
  (void)cudaMemsetAsync(source, 0x5a, checked_size(bytes, "P2P bandwidth bytes"),
                        stream);
  for (std::uint32_t iteration = 0; iteration < warmup; ++iteration)
    peer_put_signal_kernel<<<
        static_cast<unsigned int>((elements + 255U) / 256U), 256, 0,
        stream>>>(source, destination, signal, elements);
  if (cudaStreamSynchronize(stream) != cudaSuccess ||
      cudaEventRecord(start, stream) != cudaSuccess)
    goto cleanup;
  for (std::uint32_t iteration = 0; iteration < measured; ++iteration)
    peer_put_signal_kernel<<<
        static_cast<unsigned int>((elements + 255U) / 256U), 256, 0,
        stream>>>(source, destination, signal, elements);
  if (cudaEventRecord(complete, stream) != cudaSuccess ||
      cudaEventSynchronize(complete) != cudaSuccess)
    goto cleanup;
  {
    float elapsed_ms = 0.0F;
    if (cudaEventElapsedTime(&elapsed_ms, start, complete) == cudaSuccess &&
        elapsed_ms > 0.0F)
      result = static_cast<double>(bytes) * measured /
               (static_cast<double>(elapsed_ms) * 1.0e6);
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (complete)
    (void)cudaEventDestroy(complete);
  if (start)
    (void)cudaEventDestroy(start);
  if (stream)
    (void)cudaStreamDestroy(stream);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return result;
}

} // namespace megamoe::cuda
