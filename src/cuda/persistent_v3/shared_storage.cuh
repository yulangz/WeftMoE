#pragma once

#include <cstdint>

#include "cuda/persistent_v3/contracts.hpp"

namespace weftmoe::cuda::persistent_v3 {

// The union intentionally applies to shared memory only. CUDA register
// allocation is a whole-function compiler decision, so V3 controls register
// pressure through 128x128 WGMMA accumulator shapes and short lexical scopes.
// Fixed CTA roles ensure only one member is live for a CTA's lifetime.
template <std::uint32_t CommBytes, std::uint32_t RelayBytes,
          std::uint32_t GemmBytes>
struct alignas(1024) SharedStorage {
  struct Control {
    unsigned long long task = 0;
    unsigned int predicate = 0;
    unsigned int reserved = 0;
  } control;

  union alignas(1024) PhaseStorage {
    unsigned char rdma[CommBytes];
    unsigned char relay[RelayBytes];
    unsigned char gemm[GemmBytes];
  } phase;
};

} // namespace weftmoe::cuda::persistent_v3
