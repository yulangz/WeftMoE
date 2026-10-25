// nvcc -std=c++17 -O2 -arch=sm_90a -Isrc tests/cuda/source_row_decoder.cu -o
// /tmp/source_row_decoder
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
struct ForwardKernelArgs {
  unsigned world;
  const std::uint64_t *fused_local_expert_count;
  struct {
    struct {
      unsigned owned_expert_count;
    } sizes;
  } dispatch;
};
#include "cuda/weftmoe_pf/source_row_decoder.cuh"
using namespace weftmoe_pf;

__global__ void test_cursor(ForwardKernelArgs a, const unsigned *prefix,
                            unsigned total, unsigned *result) {
  SourceRowCursor cursor(a);
  unsigned expert = 0;
  for (unsigned linear = blockIdx.x; linear < total; linear += gridDim.x) {
    while (linear >= prefix[expert + 1])
      ++expert;
    unsigned source, ordinal;
    cursor.decode(expert, linear - prefix[expert], source, ordinal);
    if (threadIdx.x == 0) {
      result[2 * linear] = source;
      result[2 * linear + 1] = ordinal;
    }
  }
}
void check(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
int main() {
  std::mt19937 rng(147);
  constexpr unsigned experts = 32;
  unsigned cases = 0;
  std::uint64_t checked = 0;
  for (unsigned world : {1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64})
    for (unsigned pattern = 0; pattern < 3; ++pattern) {
      std::vector<std::uint64_t> counts(world * experts);
      std::vector<unsigned> prefix{0}, expected;
      for (unsigned e = 0; e < experts; ++e) {
        unsigned max = 0;
        for (unsigned s = 0; s < world; ++s) {
          unsigned n = pattern == 0   ? 17
                       : pattern == 1 ? (s == world - 1 ? 513 : (s % 2))
                                      : rng() % 130;
          if (e % 7 == 0)
            n = 0;
          counts[s * experts + e] = n;
          max = std::max(max, n);
        }
        // Independent oracle enumerates source rounds, with no rectangular
        // layers.
        for (unsigned ordinal = 0; ordinal < max; ++ordinal)
          for (unsigned s = 0; s < world; ++s)
            if (ordinal < counts[s * experts + e]) {
              expected.push_back(s);
              expected.push_back(ordinal);
            }
        prefix.push_back(expected.size() / 2);
      }
      std::uint64_t *dc;
      unsigned *dp, *dr;
      check(cudaMalloc(&dc, counts.size() * sizeof(*dc)));
      check(cudaMalloc(&dp, prefix.size() * sizeof(*dp)));
      check(cudaMalloc(&dr, expected.size() * sizeof(*dr)));
      check(cudaMemcpy(dc, counts.data(), counts.size() * sizeof(*dc),
                       cudaMemcpyHostToDevice));
      check(cudaMemcpy(dp, prefix.data(), prefix.size() * sizeof(*dp),
                       cudaMemcpyHostToDevice));
      ForwardKernelArgs a{world, dc, {{experts}}};
      for (unsigned warps : {1, 7, 66, 396}) {
        test_cursor<<<warps, 32>>>(a, dp, expected.size() / 2, dr);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        std::vector<unsigned> actual(expected.size());
        check(cudaMemcpy(actual.data(), dr, actual.size() * sizeof(*dr),
                         cudaMemcpyDeviceToHost));
        if (actual != expected)
          throw std::runtime_error(
              "source-row mismatch world=" + std::to_string(world) +
              " pattern=" + std::to_string(pattern) +
              " warps=" + std::to_string(warps));
        ++cases;
        checked += expected.size() / 2;
      }
      check(cudaFree(dc));
      check(cudaFree(dp));
      check(cudaFree(dr));
    }
  std::cout << "PASS " << cases << " cases, " << checked << " decoded rows\n";
}
