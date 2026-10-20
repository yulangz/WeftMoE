#pragma once

#include "weftmoe/version.hpp"

struct NotifyCommunicationView {
  ncclDevComm_t dev{};
  ncclWindow_t window = nullptr;
  std::uint64_t *base = nullptr;
  unsigned columns = 0;
  bool remote = false;
};

struct NotifyCommunication {
  NcclSession session;
  unsigned device = 0;
  NotifyCommunicationView view;
  std::shared_ptr<internal::DeviceCommunicator> connection;
  std::size_t bytes = 0;
  bool abandon = false;
  ~NotifyCommunication() {
    if (abandon) {
      if (connection)
        connection->abandoned = true;
      return;
    }
    (void)cudaSetDevice(device);
    const auto state = internal::NcclSessionAccess::checked(session);
    if (view.window)
      (void)ncclCommWindowDeregister(state->comm, view.window);
    if (view.base)
      (void)ncclMemFree(view.base);
  }
};

// Every CTA remains resident; only CTA 0 participates in the small count
// exchange. NCCL owns all network arbitration and barrier state.
__device__ void notify_exchange(LocalNotifyArgs local, DeviceMaterializeArgs a,
                                NotifyCommunicationView c,
                                std::uint64_t digest0, std::uint64_t digest1) {
  auto grid = cooperative_groups::this_grid();
  if (blockIdx.x == 0) {
    const auto world = ncclTeamWorld(c.dev);
    const auto lsa = ncclTeamLsa(c.dev);
    if (threadIdx.x < 32) {
#if WEFTMOE_ENABLE_GIN_RDMA
      if (c.remote)
        ncclBarrier(ncclCoopWarp{}, ncclTeamTagWorld{},
                    ncclGin(c.dev, 0, NCCL_GIN_RESOURCE_SHARING_CTA), 0,
                    ::cuda::memory_order_acq_rel, ncclGinFenceLevel::None);
      else
#endif
      {
        ncclLsaBarrierSession bar(ncclCoopWarp{}, c.dev, ncclTeamTagLsa{}, 0);
        bar.sync(ncclCoopWarp{}, ::cuda::memory_order_acq_rel);
      }
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < c.columns; i += blockDim.x) {
      std::uint64_t value = 0;
      if (i < a.experts)
        value = local.histogram[i];
      else if (i < a.experts + a.nodes)
        value = local.packet_count[i - a.experts];
      else if (i == c.columns - 4)
        value = local.status->out_of_range_packed;
      else if (i == c.columns - 3)
        value = local.status->duplicate_packed;
      else if (i == c.columns - 2)
        value = digest0;
      else if (i == c.columns - 1)
        value = digest1;
      c.base[i] = value;
    }
    __syncthreads();
    // Cross-node visibility needs system scope; the LSA-only (single-node)
    // build exchanges through NVLink peer mappings and needs device scope.
    if (WEFTMOE_ENABLE_GIN_RDMA) {
      __threadfence_system();
    } else {
      __threadfence();
    }
    const auto destination =
        std::size_t(1 + a.rank) * c.columns * sizeof(std::uint64_t);
    for (unsigned i = threadIdx.x; i < unsigned(lsa.nRanks) * c.columns;
         i += blockDim.x) {
      const unsigned peer = i / c.columns, word = i % c.columns;
      auto *target = peer == unsigned(lsa.rank)
                         ? reinterpret_cast<std::uint64_t *>(
                               reinterpret_cast<char *>(c.base) + destination)
                         : static_cast<std::uint64_t *>(
                               ncclGetLsaPointer(c.window, destination, peer));
      target[word] = c.base[word];
    }
    if (WEFTMOE_ENABLE_GIN_RDMA && c.remote) {
      const ncclGin gin(c.dev, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
      for (unsigned peer = threadIdx.x; peer < a.world; peer += blockDim.x) {
        const int local_peer = ncclTeamRankToLsa(c.dev, world, peer);
        if (local_peer >= 0 && local_peer < lsa.nRanks)
          continue;
        gin.put(world, peer, c.window, destination, c.window, 0,
                c.columns * sizeof(std::uint64_t), ncclGin_None{},
                ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_system);
      }
      // The following CTA join publishes every submitter to the barrier
      // warp. Its Put fence orders remote visibility; its trailing flush
      // covers local completion before the shared send record is reused.
    }
    if (WEFTMOE_ENABLE_GIN_RDMA) {
      __threadfence_system();
    } else {
      __threadfence();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
#if WEFTMOE_ENABLE_GIN_RDMA
      if (c.remote)
        ncclBarrier(ncclCoopWarp{}, ncclTeamTagWorld{},
                    ncclGin(c.dev, 0, NCCL_GIN_RESOURCE_SHARING_CTA), 0);
      else
#endif
      {
        ncclLsaBarrierSession bar(ncclCoopWarp{}, c.dev, ncclTeamTagLsa{}, 0);
        bar.sync(ncclCoopWarp{}, ::cuda::memory_order_acq_rel);
      }
    }
  }
  grid.sync();
  asm volatile("fence.proxy.async.global;" : : : "memory");
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;
  for (auto i = std::uint64_t(tid); i < std::uint64_t(a.world) * a.experts;
       i += stride)
    const_cast<std::uint64_t *>(a.count_matrix)[i] =
        c.base[(1 + i / a.experts) * c.columns + i % a.experts];
  for (auto i = std::uint64_t(tid); i < std::uint64_t(a.world) * a.nodes;
       i += stride)
    const_cast<std::uint64_t *>(a.summary_matrix)[i] =
        c.base[(1 + i / a.nodes) * c.columns + a.experts + i % a.nodes];
  // Route and config errors are collectively visible before any row index is
  // consumed. Materialization initializes status itself, so retain the error
  // separately until after its plan initialization.
  for (unsigned peer = tid; peer < a.world; peer += stride) {
    const auto *record = c.base + (1ULL + peer) * c.columns;
    if (record[c.columns - 4] != ~0ULL || record[c.columns - 3] != ~0ULL ||
        record[c.columns - 2] != digest0 || record[c.columns - 1] != digest1)
      atomicMin(&local.status->out_of_range_packed, 0ULL);
  }
  grid.sync();
}

// A rank with invalid host input must join the same exchange as healthy ranks.
// Only the fixed-capacity workspace is touched; invalid producer pointers and
// invalid configuration extents never reach the device.
__global__ void reject_notify_kernel(LocalNotifyArgs local,
                                     DeviceMaterializeArgs a,
                                     NotifyCommunicationView comm) {
  for (unsigned i = threadIdx.x; i < a.experts; i += blockDim.x)
    local.histogram[i] = 0;
  for (unsigned i = threadIdx.x; i < a.nodes; i += blockDim.x)
    local.packet_count[i] = 0;
  if (!threadIdx.x)
    *local.status = DeviceStatus{0ULL, 0ULL};
  __syncthreads();
  notify_exchange(local, a, comm, 0, 0);
}
