#pragma once

#include "weftmoe/transport/activation_protocol.hpp"
#include "cuda/weftmoe_pf/backward_return.hpp"

namespace weftmoe::cuda::backward_detail {

// Addresses imported into the consuming process, not another process's
// original CUDA pointers. Payload geometry is rank-local; scheduler metadata
// continues to use the shared transport-pool layout.
struct PeerPayload {
  transport::BFloat16 *dr;
  transport::BFloat16 *contribution;
  float *dgate;
  ReturnSlot *return_slots = nullptr;
  SourceReturnState *return_state = nullptr;
  std::uint64_t *consumed = nullptr;
};

} // namespace weftmoe::cuda::backward_detail
