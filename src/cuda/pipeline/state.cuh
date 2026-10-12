#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda/pipeline/contracts.hpp"

namespace weftmoe::cuda::pipeline {

// Hopper's L2 sector and the system-visible atomics used by the transport
// path make false sharing between producer and consumer cursors expensive.
// Every independently polled counter therefore owns a cache line.
struct alignas(128) PaddedCounter {
  unsigned long long value = 0;
};

static_assert(sizeof(PaddedCounter) == 128U);

} // namespace weftmoe::cuda::pipeline
