#pragma once

#include "weftmoe/transport/activation_protocol.hpp"
#include "cuda/weftmoe_pf/backward_return.hpp"

namespace weftmoe::cuda::backward_detail {

// One receiver CTA owns the publication. Each local rank acknowledges once,
// including empty fanouts; no reader can race reuse of the packet descriptor.
struct alignas(128) DispatchSlot {
  unsigned long long ready = 0;
  unsigned long long consumed[8]{};
  unsigned long long sequence = 0;
  unsigned long long finished = 0;
  unsigned channel = 0, rows = 0;
};

// Addresses imported into the consuming process, not another process's
// original CUDA pointers. Payload geometry is rank-local; scheduler metadata
// continues to use the shared transport-pool layout.
struct PeerPayload {
  transport::BFloat16 *dr;
  transport::BFloat16 *contribution;
  float *dgate;
  DispatchSlot *dispatch_slots = nullptr;
  ReturnSlot *return_slots = nullptr;
  SourceReturnState *return_state = nullptr;
  std::uint64_t *consumed = nullptr;
};

} // namespace weftmoe::cuda::backward_detail
