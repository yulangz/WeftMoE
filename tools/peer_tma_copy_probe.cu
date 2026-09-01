#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace cde = cuda::device::experimental;
using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;

namespace {

constexpr std::uint32_t kRowBytes = 7168U * 2U;

void check(cudaError_t error, const char *where) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(error));
}

__global__ void vector_peer_copy(std::byte *destination,
                                 const std::byte *source,
                                 std::uint64_t rows) {
  constexpr std::uint32_t kVectors = kRowBytes / sizeof(uint4);
  for (std::uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto *input =
        reinterpret_cast<const uint4 *>(source + row * kRowBytes);
    auto *output = reinterpret_cast<uint4 *>(destination + row * kRowBytes);
    for (std::uint32_t vector = threadIdx.x; vector < kVectors;
         vector += blockDim.x)
      output[vector] = input[vector];
  }
}

__global__ void warp_vector_peer_copy(std::byte *destination,
                                      const std::byte *source,
                                      std::uint64_t rows) {
  constexpr std::uint32_t kVectors = kRowBytes / sizeof(uint4);
  constexpr std::uint32_t kWarps = 12;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  for (std::uint64_t row =
           static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
       row < rows; row += static_cast<std::uint64_t>(gridDim.x) * kWarps) {
    const auto *input =
        reinterpret_cast<const uint4 *>(source + row * kRowBytes);
    auto *output = reinterpret_cast<uint4 *>(destination + row * kRowBytes);
    for (std::uint32_t vector = lane; vector < kVectors; vector += 32U)
      output[vector] = input[vector];
  }
}

template <int Stages>
__global__ void tma_peer_copy(std::byte *destination, const std::byte *source,
                              std::uint64_t rows) {
  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto *storage = reinterpret_cast<std::byte *>(storage_bytes);
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier full[Stages];
  if (threadIdx.x == 0U) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage)
      init(&full[stage], 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (threadIdx.x == 0U) {
    for (std::uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
      const auto iteration = row / gridDim.x;
      const auto stage = static_cast<int>(iteration % Stages);
      auto *buffer = storage + stage * kRowBytes;
      cde::cp_async_bulk_global_to_shared(
          buffer, source + row * kRowBytes, kRowBytes, full[stage]);
      auto token =
          cuda::device::barrier_arrive_tx(full[stage], 1, kRowBytes);
      full[stage].wait(std::move(token));

      cde::cp_async_bulk_shared_to_global(
          destination + row * kRowBytes, buffer, kRowBytes);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<Stages - 1>();
    }
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    __threadfence_system();
  }
}

template <int Issuers>
__global__ void warp_tma_peer_copy(std::byte *destination,
                                   const std::byte *source,
                                   std::uint64_t rows) {
  extern __shared__ __align__(128) unsigned char storage_bytes[];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier full[Issuers];
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  if (threadIdx.x < Issuers) {
    init(&full[threadIdx.x], 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (warp < Issuers && lane == 0U) {
    auto *buffer = reinterpret_cast<std::byte *>(storage_bytes) +
                   warp * kRowBytes;
    for (std::uint64_t row =
             static_cast<std::uint64_t>(blockIdx.x) * Issuers + warp;
         row < rows;
         row += static_cast<std::uint64_t>(gridDim.x) * Issuers) {
      cde::cp_async_bulk_global_to_shared(
          buffer, source + row * kRowBytes, kRowBytes, full[warp]);
      auto token =
          cuda::device::barrier_arrive_tx(full[warp], 1, kRowBytes);
      full[warp].wait(std::move(token));
      cde::cp_async_bulk_shared_to_global(
          destination + row * kRowBytes, buffer, kRowBytes);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<0>();
    }
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    __threadfence_system();
  }
}

template <class Launch>
double measure(std::uint64_t bytes, Launch launch) {
  constexpr int kWarmup = 5;
  constexpr int kMeasured = 20;
  for (int iteration = 0; iteration < kWarmup; ++iteration)
    launch();
  check(cudaDeviceSynchronize(), "warmup synchronize");

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "cudaEventCreate(begin)");
  check(cudaEventCreate(&end), "cudaEventCreate(end)");
  check(cudaEventRecord(begin), "cudaEventRecord(begin)");
  for (int iteration = 0; iteration < kMeasured; ++iteration)
    launch();
  check(cudaEventRecord(end), "cudaEventRecord(end)");
  check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
  float elapsed_ms = 0.0F;
  check(cudaEventElapsedTime(&elapsed_ms, begin, end),
        "cudaEventElapsedTime");
  check(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
  check(cudaEventDestroy(end), "cudaEventDestroy(end)");
  return static_cast<double>(bytes) * kMeasured /
         (static_cast<double>(elapsed_ms) * 1.0e6);
}

template <int Stages>
double measure_tma(std::byte *destination, const std::byte *source,
                   std::uint64_t rows, int ctas) {
  check(cudaFuncSetAttribute(
            tma_peer_copy<Stages>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            Stages * kRowBytes),
        "cudaFuncSetAttribute(tma_peer_copy dynamic shared memory)");
  const auto launch = [=] {
    tma_peer_copy<Stages><<<ctas, 32, Stages * kRowBytes>>>(
        destination, source, rows);
    check(cudaGetLastError(), "tma_peer_copy launch");
  };
  return measure(rows * kRowBytes, launch);
}

template <int Issuers>
double measure_warp_tma(std::byte *destination, const std::byte *source,
                        std::uint64_t rows, int ctas) {
  check(cudaFuncSetAttribute(
            warp_tma_peer_copy<Issuers>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            Issuers * kRowBytes),
        "cudaFuncSetAttribute(warp_tma_peer_copy dynamic shared memory)");
  const auto launch = [=] {
    warp_tma_peer_copy<Issuers><<<ctas, 384, Issuers * kRowBytes>>>(
        destination, source, rows);
    check(cudaGetLastError(), "warp_tma_peer_copy launch");
  };
  return measure(rows * kRowBytes, launch);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int source_device = argc > 1 ? std::stoi(argv[1]) : 0;
    const int destination_device = argc > 2 ? std::stoi(argv[2]) : 1;
    const std::uint64_t rows =
        argc > 3 ? std::stoull(argv[3]) : 16384ULL;
    const auto bytes = rows * kRowBytes;

    int can_access = 0;
    check(cudaDeviceCanAccessPeer(&can_access, source_device,
                                  destination_device),
          "cudaDeviceCanAccessPeer");
    if (can_access == 0)
      throw std::runtime_error("selected devices do not support peer access");

    std::byte *source = nullptr;
    std::byte *destination = nullptr;
    check(cudaSetDevice(source_device), "cudaSetDevice(source)");
    check(cudaDeviceEnablePeerAccess(destination_device, 0),
          "cudaDeviceEnablePeerAccess");
    check(cudaMalloc(&source, bytes), "cudaMalloc(source)");
    check(cudaMemset(source, 0x5a, bytes), "cudaMemset(source)");
    check(cudaSetDevice(destination_device), "cudaSetDevice(destination)");
    check(cudaMalloc(&destination, bytes), "cudaMalloc(destination)");
    check(cudaMemset(destination, 0, bytes), "cudaMemset(destination)");
    check(cudaSetDevice(source_device), "cudaSetDevice(source launch)");

    std::cout << "peer_tma_copy_probe source=" << source_device
              << " destination=" << destination_device << " rows=" << rows
              << " row_bytes=" << kRowBytes << " total_bytes=" << bytes
              << '\n';
    for (const int ctas : {1, 2, 4, 8, 16}) {
      const auto vector_gbps = measure(bytes, [=] {
        vector_peer_copy<<<ctas, 384>>>(destination, source, rows);
        check(cudaGetLastError(), "vector_peer_copy launch");
      });
      const auto warp_vector_gbps = measure(bytes, [=] {
        warp_vector_peer_copy<<<ctas, 384>>>(destination, source, rows);
        check(cudaGetLastError(), "warp_vector_peer_copy launch");
      });
      std::cout << "ctas=" << ctas << " vector_GBps=" << vector_gbps
                << " warp_vector_GBps=" << warp_vector_gbps
                << std::flush;
      const auto tma1_gbps =
          measure_tma<1>(destination, source, rows, ctas);
      std::cout << " tma1_GBps=" << tma1_gbps << std::flush;
      const auto tma2_gbps =
          measure_tma<2>(destination, source, rows, ctas);
      std::cout << " tma2_GBps=" << tma2_gbps << std::flush;
      const auto tma4_gbps =
          measure_tma<4>(destination, source, rows, ctas);
      std::cout << " tma4_GBps=" << tma4_gbps << std::flush;
      const auto warp_tma12_gbps =
          measure_warp_tma<12>(destination, source, rows, ctas);
      const auto warp_tma4_gbps =
          measure_warp_tma<4>(destination, source, rows, ctas);
      const auto warp_tma8_gbps =
          measure_warp_tma<8>(destination, source, rows, ctas);
      std::cout << " warp_tma4_GBps=" << warp_tma4_gbps
                << " warp_tma8_GBps=" << warp_tma8_gbps
                << " warp_tma12_GBps=" << warp_tma12_gbps << '\n';
    }

    unsigned char first = 0;
    unsigned char last = 0;
    check(cudaSetDevice(destination_device), "cudaSetDevice(verify)");
    check(cudaMemcpy(&first, destination, 1, cudaMemcpyDeviceToHost),
          "verify first byte");
    check(cudaMemcpy(&last, destination + bytes - 1, 1,
                     cudaMemcpyDeviceToHost),
          "verify last byte");
    if (first != 0x5a || last != 0x5a)
      throw std::runtime_error("peer copy verification failed");
    check(cudaFree(destination), "cudaFree(destination)");
    check(cudaSetDevice(source_device), "cudaSetDevice(free source)");
    check(cudaFree(source), "cudaFree(source)");
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "peer_tma_copy_probe: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
