#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda/persistent_v3/contracts.hpp"

namespace megamoe::cuda::persistent_v3 {

// Hopper's L2 sector and the system-visible atomics used by the transport
// path make false sharing between producer and consumer cursors expensive.
// Every independently polled counter therefore owns a cache line.
struct alignas(128) PaddedCounter {
  unsigned long long value = 0;
};

static_assert(sizeof(PaddedCounter) == 128U);

struct alignas(128) ContextProgress {
  PaddedCounter rx_published;
  PaddedCounter rx_consumed;
  PaddedCounter tx_published;
  PaddedCounter tx_local_complete;
};

struct alignas(128) PipelineProgress {
  PaddedCounter expected[kTaskKindCount];
  PaddedCounter completed[kTaskKindCount];
};

struct alignas(128) KernelControl {
  PaddedCounter terminal_epoch;
  PaddedCounter cancel_epoch;
  PaddedCounter error_code;
  PaddedCounter terminal_publish_count;
};

struct alignas(128) GenerationState {
  KernelControl control;
  ContextProgress contexts[kRdmaCtaCount];
  PipelineProgress pipeline;
};

} // namespace megamoe::cuda::persistent_v3
