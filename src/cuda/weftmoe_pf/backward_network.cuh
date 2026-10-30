#pragma once

namespace weftmoe_pf::backward_stream {

#ifndef WEFTMOE_PF_BWD_RETURN_SPANS
#define WEFTMOE_PF_BWD_RETURN_SPANS 0
#endif
#ifndef WEFTMOE_PF_BWD_STRIPED_SIGNALS
#define WEFTMOE_PF_BWD_STRIPED_SIGNALS 0
#endif
static_assert(!WEFTMOE_PF_BWD_STRIPED_SIGNALS ||
                  kPutWorkers <= static_forward::kNetworkPackets.lanes,
              "Striped Return signals require one channel lane per Put worker");

// Dispatch fills the compact dY cache already needed by dW2. Communication
// windows do not pad GEMM; Return reuses Forward's registered TX as an M ring.
struct CanonicalRow {
  unsigned source;
  unsigned span;
  std::uint64_t contribution;
};

__host__ __device__ inline unsigned network_depth(const ForwardKernelArgs &f) {
  const auto &r = f.static_network.rings;
  const auto capacity = r.slots() * r.return_stride / (kM * f.hidden * 2U);
  return unsigned(capacity < f.ring.depth ? capacity : f.ring.depth);
}

__device__ __forceinline__ auto directory_layout(const BackwardKernelArgs &a) {
  return pipeline::backward::make_backward_scheduler_layout(
      a.scheduler_mblock_capacity, kDxTiles / 2U, a.scheduler_expert_capacity);
}

__device__ __forceinline__ auto *directory(const BackwardKernelArgs &a) {
  return reinterpret_cast<pipeline::backward::BackwardMblockDescriptor *>(
      backward_pipeline_detail::scheduler_words(a) +
      directory_layout(a).descriptors);
}

__device__ __forceinline__ auto *canonical_rows(const BackwardKernelArgs &a) {
  return peer_at<CanonicalRow>(a.local_base, a.direct_headers_offset);
}

__device__ __forceinline__ unsigned ordered_block(const BackwardKernelArgs &a,
                                                  unsigned position) {
  const auto layout = directory_layout(a);
  const auto offset = layout.du_completed;
  return unsigned(
      backward_pipeline_detail::scheduler_words(a)[offset + position]);
}

__device__ __forceinline__ unsigned
network_blocks(const BackwardKernelArgs &a) {
  return unsigned(
      peer_at<const unsigned long long>(
          a.local_base,
          a.padded_expert_offsets_offset)[a.dispatch.sizes.owned_expert_count] /
      kM);
}

struct NetworkSchedule {
  const BackwardKernelArgs &a;
  unsigned depth, all, swiglu_tiles, begin = 0U, index = blockIdx.x;

  __device__ NetworkSchedule(const ForwardKernelArgs &f,
                             const BackwardKernelArgs &args)
      : a(args), depth(network_depth(f)), all(network_blocks(a)),
        swiglu_tiles(args.fp8 ? unsigned(args.intermediate / 128U) : 2U) {}

  __device__ __forceinline__ bool next(Task &t) {
    while (begin < all) {
      constexpr unsigned Wave = WEFTMOE_PF_BWD_WAVE_BLOCKS;
      const auto blocks = min(Wave ? min(Wave, depth) : depth, all - begin);
      const auto du = blocks * kDuTiles, epilogue = blocks * swiglu_tiles;
      const auto total = du + epilogue + blocks * kDxTiles;
      if (index < total) {
        const auto phase = index < du              ? Phase::Du
                           : index < du + epilogue ? Phase::Swiglu
                                                   : Phase::Dx;
        const auto linear = index - (phase == Phase::Du       ? 0U
                                     : phase == Phase::Swiglu ? du
                                                              : du + epilogue);
        const auto columns = phase == Phase::Du   ? kDuTiles
                             : phase == Phase::Dx ? kDxTiles
                                                  : swiglu_tiles;
        const auto position = begin + linear / columns;
        const auto block = ordered_block(a, position);
        const auto desc = directory(a)[block];
        t = {desc.expert,
             block,
             linear % columns,
             desc.valid_rows,
             position % depth * kM,
             position / depth + 1U,
             phase};
        index += gridDim.x;
        return true;
      }
      index -= total;
      begin += blocks;
    }
    return false;
  }
};

// NIC requests and RX forwarding remain live during dW.
struct alignas(16) NetworkShared {
  unsigned char pull[3][local::kPullBufferBytes];
  local::Barrier pull_full[3];
};
constexpr auto kNetworkCommOffset = (kSharedBytes + 15U) & ~std::size_t{15U};
constexpr auto kNetworkSharedBytes = kNetworkCommOffset + sizeof(NetworkShared);
constexpr auto kFP8NetworkCommOffset = (kFP8SharedBytes + 15U) & ~std::size_t{15U};
constexpr auto kFP8NetworkSharedBytes = kFP8NetworkCommOffset + sizeof(NetworkShared);

__device__ __forceinline__ std::size_t tx_base(const ForwardKernelArgs &f) {
  return f.static_network.ring_region +
         f.static_network.rings.return_payload(0U, 0U);
}

// Keep the BWD planner's physical directory for RX fanout. A window-ordered
// tape shares this stable expert/source order with FWD. Canonical reverse rows
// reuse the idle route-header arena, never a second activation payload.
__global__ void prepare_network(ForwardKernelArgs f, BackwardKernelArgs a) {
  const auto tid = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = std::uint64_t(gridDim.x) * blockDim.x;
  const auto &r = f.ring;
  auto *words = backward_pipeline_detail::scheduler_words(a);
  const auto layout = directory_layout(a);
  const auto *padded = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  const auto owned = a.dispatch.sizes.owned_expert_count;
  for (auto block = tid; block < network_blocks(a); block += stride) {
    unsigned expert = 0U;
    while (block * kM >= padded[expert + 1U])
      ++expert;
    const auto row = block * kM - padded[expert];
    directory(a)[block] = {
        unsigned(a.dispatch.expert_row_prefix[expert] + row),
        unsigned(block * kM), expert,
        unsigned(min(static_cast<unsigned long long>(kM),
                     a.dispatch.expert_row_count[expert] - row))};
    words[layout.dispatch_arrived + block] = 0ULL;
  }
  for (auto slot = tid; slot < r.depth; slot += stride)
    r.slots[slot] = {};
  for (auto row = tid; row < r.state->blocks * kM; row += stride) {
    const auto &desc = r.blocks[row / kM];
    if (row % kM >= desc.tile.rows)
      continue;
    const auto identity = r.rows[row];
    const auto window = desc.window / r.source.shape.groups();
    const auto *prefix =
        ring::source_metadata(f, identity.source, r.source.expert_prefix) +
        window * (r.source.experts + 1U);
    auto ordinal = desc.tile.expert_row + unsigned(row % kM);
    for (auto source = r.source.shape.source_begin(desc.window);
         source < identity.source; ++source)
      ordinal -= unsigned(
          ring::source_count(f, source, window, desc.tile.expert, false));
    const auto saved = f.checkpoint.rows[row];
    const auto padded = peer_at<const unsigned>(
        a.local_base, a.compact_to_padded_offset)[saved];
    unsigned span = 0U;
#if WEFTMOE_PF_BWD_RETURN_SPANS
    // This source/expert interval is contiguous in the window-ordered tape.
    // Clip at the canonical M boundary, with no runtime row-by-row scan.
    span = min(kM - padded % kM,
               unsigned(ring::source_count(f, identity.source, window,
                                           desc.tile.expert, false)) -
                   ordinal);
#endif
    canonical_rows(a)[saved] = {
        identity.source, span,
        ring::window_source(f, desc.window, identity.source).row_base +
            prefix[f.rank * owned + desc.tile.expert] + ordinal};
    const auto block = padded / kM;
    // Tape rows are monotonic in window within each expert. Only the last
    // valid row of an M block publishes its last-required-window key.
    if ((padded + 1U) % kM == 0U ||
        saved + 1U == a.dispatch.expert_row_prefix[desc.tile.expert + 1U])
      words[layout.phase + block] =
          (static_cast<unsigned long long>(desc.window) << 32U) | block;
  }
  for (auto peer = tid; peer < f.static_network.rings.nodes; peer += stride) {
    if (peer == f.node)
      continue;
    const auto channel = f.static_network.rings.channel(f.node, peer, 0U);
    auto &state = *peer_at<static_forward::NetworkChannel>(
        f.gin_window_base, f.static_network.ring_region +
                               f.static_network.rings.state(true, channel));
    std::uint64_t rows = 0U;
    const auto first = peer * f.local_world_size * owned;
    const auto *counts =
        a.dispatch.source_expert_count +
        std::uint64_t(f.rank) * a.dispatch.identity.num_experts;
    for (unsigned expert = first; expert < first + f.local_world_size * owned;
         ++expert)
      rows += counts[expert];
    state.backward_rows += rows;
  }
}

// Each expert's window-ordered tape already has monotonically increasing
// last-window keys. Merge those sorted runs in one launch by finding each
// block's rank. The old dU counter arena holds the immutable task permutation;
// stream BWD uses ring slots for dU completion instead.
__global__ void order_windows(BackwardKernelArgs a) {
  const auto block = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= network_blocks(a))
    return;
  const auto layout = directory_layout(a);
  auto *words = backward_pipeline_detail::scheduler_words(a);
  const auto *keys = words + layout.phase;
  const auto key = keys[block];
  const auto *padded = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  unsigned position = 0U;
  for (unsigned expert = 0U; expert < a.dispatch.sizes.owned_expert_count;
       ++expert) {
    const auto begin = padded[expert] / kM;
    auto lo = begin, hi = padded[expert + 1U] / kM;
    while (lo < hi) {
      const auto mid = (lo + hi) / 2U;
      if (keys[mid] < key)
        lo = mid + 1U;
      else
        hi = mid;
    }
    position += unsigned(lo - begin);
  }
  words[layout.du_completed + position] = key;
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
__device__ __forceinline__ transport::BFloat16 *
dx_output(const ForwardKernelArgs &f, const BackwardKernelArgs &a,
          const Task &t, unsigned row) {
  const auto identity =
      canonical_rows(a)[directory(a)[t.block].compact_row + row];
  if (identity.source / f.local_world_size == f.node)
    return static_cast<transport::BFloat16 *>(ncclGetLsaPointer(
        f.gin_window,
        f.static_network.contribution + identity.contribution * f.hidden * 2U,
        identity.source % f.local_world_size));
  return peer_at<transport::BFloat16>(f.gin_window_base, tx_base(f)) +
         std::uint64_t(t.row + row) * f.hidden;
}

__device__ __forceinline__ float *gradient_output(const ForwardKernelArgs &f,
                                                  const BackwardKernelArgs &a,
                                                  const Task &t, unsigned row) {
  const auto identity =
      canonical_rows(a)[directory(a)[t.block].compact_row + row];
  if (identity.source / f.local_world_size == f.node)
    return static_cast<float *>(
        ncclGetLsaPointer(f.gin_window,
                          f.static_network.gradient_contribution +
                              identity.contribution * sizeof(float),
                          identity.source % f.local_world_size));
  return peer_at<float>(f.gin_window_base, f.static_network.gradient_tx) +
         t.row + row;
}

__device__ __noinline__ void pull_canonical(const ForwardKernelArgs &f,
                                            const BackwardKernelArgs &a,
                                            NetworkShared &s, unsigned warp,
                                            unsigned worker, unsigned workers) {
  const auto lane = threadIdx.x % 32U;
  network::Copy copy{s.pull[warp], s.pull_full[warp]};
  auto *ready = backward_pipeline_detail::scheduler_words(a) +
                directory_layout(a).dispatch_arrived;
  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  network::pull_rows<false>(
      f, worker, workers,
      [&](std::uint64_t block, unsigned row, float, const char *input) {
        const auto saved = f.checkpoint.rows[block * kM + row];
        if (a.fp8)
          network::copy_fp8_row(copy, input, reinterpret_cast<std::uint8_t *>(a.dr),
              mapping[saved], unsigned(a.hidden), fp8::source_pitch(unsigned(a.hidden)));
        else
          copy.warp_copy(input, reinterpret_cast<char *>(a.dr + saved * a.hidden),
                         unsigned(a.hidden * 2U));
        if (!lane) {
          ring::arrive(
              *reinterpret_cast<std::uint64_t *>(ready + mapping[saved] / kM),
              1U);
        }
        __syncwarp();
      });
#if defined(WEFTMOE_PF_BWD_TASK_TIMING) || defined(WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC)
  if (!lane)
    atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                   ->dispatch_complete_clock,
              device_globaltimer_ns());
#endif
}

#ifndef WEFTMOE_PF_BWD_RETURN_PENDING
#define WEFTMOE_PF_BWD_RETURN_PENDING 1
#endif
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
struct ReturnPending {
  ncclGinRequest_t request;
  std::uint64_t position;
};
static_assert(WEFTMOE_PF_BWD_RETURN_PENDING * sizeof(ReturnPending) <= 4096U);
#endif

__device__ __forceinline__ unsigned fragment_rows(const CanonicalRow *rows,
                                                  unsigned row, unsigned end,
                                                  CanonicalRow identity) {
#if WEFTMOE_PF_BWD_RETURN_SPANS
  return identity.span;
#else
  unsigned count = 1U;
  while (row + count < end && rows[row + count].source == identity.source &&
         rows[row + count].contribution == identity.contribution + count)
    ++count;
  return count;
#endif
}

// Consecutive rows to one source/contribution interval become one Put. An
// expert/source boundary splits only communication, never a GEMM tile.
__device__ __noinline__ void return_canonical(const ForwardKernelArgs &f,
                                              const BackwardKernelArgs &a,
                                              unsigned worker, void *scratch) {
  if (threadIdx.x % 32U)
    return;
  netring::Endpoint endpoint(f, kGetWorkers + worker);
  const auto depth = network_depth(f);
  const auto bytes = f.hidden * 2U;
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
  constexpr unsigned Pending = WEFTMOE_PF_BWD_RETURN_PENDING;
  auto *pending = static_cast<ReturnPending *>(scratch);
  unsigned issued = 0U, completed = 0U;
  const auto retire = [&]() {
    auto &entry = pending[completed % Pending];
    endpoint.gin.wait(entry.request);
    if (entry.position != ~0ULL)
      ring::publish(f.ring.slots[entry.position % depth].returned,
                    entry.position / depth + 1U);
    ++completed;
  };
#endif
  for (auto position = worker; position < network_blocks(a);
       position += kPutWorkers) {
    const auto block = ordered_block(a, position);
    auto &slot = f.ring.slots[position % depth];
    const auto generation = position / depth + 1U;
    while (ring::acquire(slot.fc2_done) <
           std::uint64_t(generation) * kDxTiles) {
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
      if (completed != issued)
        retire();
#endif
    }
    const auto desc = directory(a)[block];
    const auto *rows = canonical_rows(a) + desc.compact_row;
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
    const auto first_request = issued;
#endif
    for (unsigned row = 0U; row < desc.valid_rows;) {
      const auto identity = rows[row];
      const auto node = identity.source / f.local_world_size;
      const auto count = fragment_rows(rows, row, desc.valid_rows, identity);
      if (node != f.node) {
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
        if (issued - completed == Pending)
          retire();
#endif
        const auto channel = endpoint.layout().channel(
            node, f.node, WEFTMOE_PF_BWD_STRIPED_SIGNALS ? worker : 0U);
        const auto control = endpoint.control(true, channel, 0U);
        const auto source_row = std::uint64_t(position % depth) * kM + row;
        endpoint.gin.put(
            ncclTeamWorld(endpoint.gin.comm), identity.source, f.gin_window,
            f.static_network.contribution + identity.contribution * bytes,
            f.gin_window, tx_base(f) + source_row * bytes, count * bytes,
            ncclGin_WeakVASignalAdd{
                f.gin_window,
                control +
                    offsetof(static_forward::NetworkSlot, backward_dx_signal),
                count},
            ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_system);
        endpoint.gin.put(
            ncclTeamWorld(endpoint.gin.comm), identity.source, f.gin_window,
            f.static_network.gradient_contribution +
                identity.contribution * sizeof(float),
            f.gin_window,
            f.static_network.gradient_tx + source_row * sizeof(float),
            count * sizeof(float),
            ncclGin_WeakVASignalAdd{
                f.gin_window,
                control +
                    offsetof(static_forward::NetworkSlot, backward_gate_signal),
                count},
            ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_system);
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
        auto &entry = pending[issued++ % Pending];
        entry.position = ~0ULL;
        endpoint.gin.flushAsync(ncclTeamWorld(endpoint.gin.comm),
                                identity.source, &entry.request);
#else
        ncclGinRequest_t request;
        endpoint.gin.flushAsync(ncclTeamWorld(endpoint.gin.comm),
                                identity.source, &request);
        endpoint.gin.wait(request);
#endif
      }
      row += count;
    }
    // FIFO retirement includes requests to other peers. Only the last
    // fragment releases this M-slot's local NIC-read credit.
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
    if (issued != first_request)
      pending[(issued - 1U) % Pending].position = position;
    else
#endif
      ring::publish(slot.returned, generation);
  }
#if WEFTMOE_PF_BWD_RETURN_PENDING > 1
  while (completed != issued)
    retire();
#endif
}

__device__ __noinline__ void receive_canonical(const ForwardKernelArgs &f,
                                               unsigned worker) {
#if !WEFTMOE_PF_BWD_STRIPED_SIGNALS
  if (threadIdx.x % 32U)
    return;
#endif
  netring::Endpoint endpoint(f, kGetWorkers + kPutWorkers + worker);
  for (auto peer = worker; peer < endpoint.layout().nodes;
       peer += kProgressWorkers) {
    if (peer == f.node)
      continue;
    const auto channel = endpoint.layout().channel(f.node, peer, 0U);
    const auto target = endpoint.state(true, channel).backward_rows;
#if WEFTMOE_PF_BWD_STRIPED_SIGNALS
    // Sender-owned counters avoid a shared remote atomic hotspot. Their sum
    // uses the existing exact cumulative node target: no new expected-count
    // directory, reset protocol, or payload capacity is needed.
    const auto lane = threadIdx.x % 32U;
    while (true) {
      unsigned long long dx = 0U, gate = 0U;
      for (unsigned sender = lane; sender < kPutWorkers; sender += 32U) {
        const auto control = endpoint.control(
            true, endpoint.layout().channel(f.node, peer, sender), 0U);
        dx += endpoint.gin.readSignal(
            f.gin_window, control + offsetof(static_forward::NetworkSlot,
                                             backward_dx_signal));
        gate += endpoint.gin.readSignal(
            f.gin_window, control + offsetof(static_forward::NetworkSlot,
                                             backward_gate_signal));
      }
      for (unsigned offset = 16U; offset; offset /= 2U) {
        dx += __shfl_down_sync(0xffffffffU, dx, offset);
        gate += __shfl_down_sync(0xffffffffU, gate, offset);
      }
      if (__shfl_sync(0xffffffffU, dx >= target && gate >= target, 0U))
        break;
    }
#else
    const auto control = endpoint.control(true, channel, 0U);
    while (endpoint.gin.readSignal(
               f.gin_window, control + offsetof(static_forward::NetworkSlot,
                                                backward_dx_signal)) < target) {
    }
    while (endpoint.gin.readSignal(
               f.gin_window, control + offsetof(static_forward::NetworkSlot,
                                                backward_gate_signal)) <
           target) {
    }
#endif
  }
}

__device__ __forceinline__ void network_workers(const ForwardKernelArgs &f,
                                                const BackwardKernelArgs &a,
                                                NetworkShared &s) {
  const auto warp = (threadIdx.x - 288U) / 32U;
  const auto role =
      warp == 0U && blockIdx.x < kNetworkOwners ? blockIdx.x : ~0U;
  if (role < kGetWorkers)
    network::dispatch(f, reinterpret_cast<network::Pending *>(s.pull[0]), role);
  else if (role < kGetWorkers + kPutWorkers)
    return_canonical(f, a, role - kGetWorkers, s.pull[0]);
  else if (role < kNetworkOwners - 1U)
    receive_canonical(f, role - kGetWorkers - kPutWorkers);
  else if (role != kNetworkOwners - 1U) {
    const auto worker = blockIdx.x < kNetworkOwners
                            ? 2U * blockIdx.x + warp - 1U
                            : 3U * blockIdx.x - kNetworkOwners + warp;
    pull_canonical(f, a, s, warp, worker, gridDim.x * 3U - kNetworkOwners);
  }
}

template <unsigned Base = 288U, unsigned Threads = 96U>
__device__ __forceinline__ void finish_network(const ForwardKernelArgs &f,
                                               const BackwardKernelArgs &a) {
  rank_barrier<Base, Threads, true, true, true>(f);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (threadIdx.x == Base) {
    atomicMax(&status->return_complete_clock, device_globaltimer_ns());
    atomicMin(&status->backward_reduce_start_clock, device_globaltimer_ns());
  }
  local::combine<true, Base, Threads>(f);
  const auto *input = peer_at<const float>(
      f.gin_window_base, f.static_network.gradient_contribution);
  for (auto route = std::uint64_t(blockIdx.x) * Threads + threadIdx.x - Base;
       route < f.dispatch.identity.num_tokens * f.dispatch.identity.top_k;
       route += std::uint64_t(gridDim.x) * Threads) {
    const auto token = route / f.dispatch.identity.top_k;
    const auto sequence =
        f.ring.state->source_base + token / f.ring.source.shape.tokens;
    const auto row =
        sequence % f.ring.contribution_depth * f.ring.contribution_stride +
        f.ring.contribution_rows[route];
    a.dgate[route] = input[row];
  }
  if (threadIdx.x == Base)
    atomicMax(&status->backward_reduce_complete_clock, device_globaltimer_ns());
  grid_sync<Base, Threads>(f);
  if (!blockIdx.x && threadIdx.x == Base) {
    f.ring.state->source_base += f.ring.source.local_windows();
    f.ring.state->block_base += f.ring.state->blocks;
    // Restore Forward counter units for every residue, including unused ones.
    for (unsigned i = 0U; i < f.ring.depth; ++i) {
      const auto generation =
          (f.ring.state->block_base + f.ring.depth - 1U - i) / f.ring.depth;
      auto &slot = f.ring.slots[i];
      slot = {};
      slot.x_full = generation * kM;
      slot.fc1_done = slot.fc1_target = generation * local::kFc1Tiles;
      slot.fc2_done = slot.fc2_target = slot.fc2_published =
          generation * local::kFc2Tiles;
    }
    for (unsigned channel = 0U; channel < f.static_network.rings.channels();
         ++channel) {
      auto &state = *peer_at<static_forward::NetworkChannel>(
          f.gin_window_base, f.static_network.ring_region +
                                 f.static_network.rings.state(false, channel));
      state.rx_base = state.rx_cursor;
    }
  }
}

#endif

} // namespace weftmoe_pf::backward_stream
