// Standalone device regression for the production cooperative stable sort.
// nvcc -std=c++17 -arch=sm_90a -Isrc tests/test_notify_radix.cu -o /tmp/test_notify_radix
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include "cuda/notify/radix.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}
__global__ void sort_test(std::uint64_t *keys, std::uint64_t *values,
                          std::uint64_t *scratch_keys,
                          std::uint64_t *scratch_values, std::uint64_t count,
                          std::uint64_t *histogram) {
  __shared__ NotifySortStorage shared;
  notify_radix_sort(keys, values, scratch_keys, scratch_values, count,
                    histogram, shared);
}
int main() {
  constexpr std::uint64_t capacity = 524291;
  std::uint64_t *keys, *values, *scratch_keys, *scratch_values, *histogram;
  for (auto ptr : {&keys, &values, &scratch_keys, &scratch_values})
    check(cudaMalloc(ptr, capacity * sizeof(std::uint64_t)));
  check(cudaMalloc(&histogram,
                   (((capacity + 1023) / 1024) * 256 + 256) * sizeof(std::uint64_t)));
  int sms, resident;
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, sort_test, 256, 0));
  std::mt19937_64 rng(719);
  unsigned cases = 0;
  for (std::uint64_t count : {0ULL, 1ULL, 513ULL, 1024ULL, 1025ULL, 2049ULL,
                              65539ULL, 524291ULL}) {
    for (unsigned pattern = 0; pattern < 6; ++pattern) {
      std::vector<std::uint64_t> input(count), order(count), actual(count), output(count);
      std::iota(order.begin(), order.end(), 0);
      for (std::uint64_t i = 0; i < count; ++i) {
        switch (pattern) {
        case 0: input[i] = 0; break;
        case 1: input[i] = ~std::uint64_t(0); break;
        case 2: input[i] = rng() % 3; break;
        case 3: input[i] = (count - i) % 257; break;
        case 4: input[i] = rng(); break;
        default: input[i] = (i / 1023) << 40; break;
        }
      }
      check(cudaMemcpy(keys, input.data(), count * 8, cudaMemcpyHostToDevice));
      check(cudaMemcpy(values, order.data(), count * 8, cudaMemcpyHostToDevice));
      std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return input[a] < input[b]; });
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(std::min<unsigned>(sms * resident, 132));
      config.blockDim = dim3(256);
      cudaLaunchAttribute attribute{};
      attribute.id = cudaLaunchAttributeCooperative;
      attribute.val.cooperative = 1;
      config.attrs = &attribute;
      config.numAttrs = 1;
      check(cudaLaunchKernelEx(&config, sort_test, keys, values, scratch_keys,
                               scratch_values, count, histogram));
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(actual.data(), values, count * 8, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(output.data(), keys, count * 8, cudaMemcpyDeviceToHost));
      for (std::uint64_t i = 0; i < count; ++i)
        if (actual[i] != order[i] || output[i] != input[order[i]]) {
          std::fprintf(stderr, "Mismatch count=%llu pattern=%u index=%llu\n",
                       (unsigned long long)count, pattern, (unsigned long long)i);
          return 1;
        }
      ++cases;
    }
  }
  for (auto ptr : {keys, values, scratch_keys, scratch_values, histogram}) check(cudaFree(ptr));
  std::printf("PASS: %u stable radix cases (keys and source ordinals)\n", cases);
}
