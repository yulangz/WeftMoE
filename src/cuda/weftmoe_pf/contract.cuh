#pragma once

namespace weftmoe_pf {

// Inference only needs the source contribution address and the FC1 gate.
// Keep that hot route metadata in one naturally aligned remote store.
struct alignas(8) RouteHeader {
  std::uint32_t contribution_row;
  std::uint32_t source_rank;
};
static_assert(sizeof(RouteHeader) == 8U);

__device__ __forceinline__ const RouteHeader &
route_header(const ForwardKernelArgs &a, std::uint64_t canonical_row) {
  return peer_at<const RouteHeader>(a.local_base,
                                    a.direct_headers_offset)[canonical_row];
}

} // namespace weftmoe_pf
