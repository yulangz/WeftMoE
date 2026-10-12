#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cuda/forward/tuning_config.hpp"

#if defined(__CUDACC__)
#define WEFTMOE_PIPELINE_HOST_DEVICE __host__ __device__
#define WEFTMOE_PIPELINE_DEVICE __device__
#else
#define WEFTMOE_PIPELINE_HOST_DEVICE
#define WEFTMOE_PIPELINE_DEVICE
#endif

namespace weftmoe::cuda::pipeline {

// Compute and retained activations share a 128-row canonical M block.
// The final block of every expert is physically padded, not merely predicated,
// so dW can traverse a uniform, deterministic K extent in 64-row WGMMA steps.
inline constexpr std::uint32_t kMBlockRows = 128U;
// Wire/context cardinality, not a launch-policy knob.  A two-node H200 job has
// eight physical GIN contexts; a one-node specialization launches zero RDMA
// workers but must not turn protocol divisions into divisions by zero.
inline constexpr std::uint32_t kRdmaCtaCount = 8U;
inline constexpr std::uint32_t kThreadsPerCta = forward_tuning::kComputeThreads;
inline constexpr std::uint32_t kWarpsPerCta = kThreadsPerCta / 32U;
inline constexpr std::uint32_t kInvalidTask = ~0U;

template <class T>
WEFTMOE_PIPELINE_HOST_DEVICE constexpr T align_up(T value, T alignment) {
  static_assert(std::is_unsigned_v<T>);
  return value == 0 ? 0 : ((value - 1) / alignment + 1) * alignment;
}

WEFTMOE_PIPELINE_HOST_DEVICE constexpr std::uint64_t
padded_expert_rows(std::uint64_t real_rows) {
  return align_up(real_rows, static_cast<std::uint64_t>(kMBlockRows));
}

WEFTMOE_PIPELINE_HOST_DEVICE constexpr std::uint64_t
expert_mblocks(std::uint64_t real_rows) {
  return padded_expert_rows(real_rows) / kMBlockRows;
}

struct ExpertRowSpan {
  std::uint64_t canonical_begin = 0;
  std::uint64_t padded_begin = 0;
  std::uint64_t real_rows = 0;
  std::uint64_t padded_rows = 0;

  WEFTMOE_PIPELINE_HOST_DEVICE constexpr std::uint64_t mblocks() const {
    return padded_rows / kMBlockRows;
  }

  WEFTMOE_PIPELINE_HOST_DEVICE constexpr bool valid() const {
    return padded_rows == padded_expert_rows(real_rows) &&
           padded_begin % kMBlockRows == 0;
  }
};

} // namespace weftmoe::cuda::pipeline
