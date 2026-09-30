#pragma once

namespace weftmoe_pf {

__device__ __forceinline__ std::size_t source_offset(const ForwardKernelArgs &a,
                                                     unsigned source) {
  return a.static_network.source +
         (source / a.local_world_size) * a.static_network.source_stride;
}

template <class T>
__device__ __forceinline__ T *source_pointer(const ForwardKernelArgs &a,
                                             unsigned source,
                                             std::size_t offset = 0) {
  return static_cast<T *>(ncclGetLsaPointer(a.gin_window,
                                            source_offset(a, source) + offset,
                                            source % a.local_world_size));
}

// Expert prefixes also give exact per-node Return extents. The final prefix
// is the source's actual route count; no assumption of equal token counts is
// needed by either Dispatch or Return.
__device__ __forceinline__ const std::uint64_t *
source_prefix(const ForwardKernelArgs &a, unsigned source) {
  return source_pointer<const std::uint64_t>(a, source);
}

__device__ __forceinline__ transport::BFloat16 *
network_fc2_destination(const ForwardKernelArgs &a, const RouteHeader &header) {
  const auto source_node = header.source_rank / a.local_world_size;
  if (source_node == a.node)
    return peer_at<transport::BFloat16>(a.peer_bases[header.source_rank],
                                        a.contribution_offset) +
           std::uint64_t(header.contribution_row) * a.hidden;
  return static_cast<transport::BFloat16 *>(
             ncclGetLsaPointer(a.gin_window,
                               a.static_network.return_tx +
                                   source_node * a.static_network.return_stride,
                               header.source_rank % a.local_world_size)) +
         std::uint64_t(header.contribution_row) * a.hidden;
}

__device__ __forceinline__ const transport::BFloat16 *
network_contribution(const ForwardKernelArgs &a, std::uint64_t route) {
  const auto experts_per_node =
      a.dispatch.identity.num_experts / a.dispatch.identity.num_nodes;
  const auto destination_node = a.selected_experts[route] / experts_per_node;
  if (destination_node == a.node)
    return peer_at<const transport::BFloat16>(a.local_base,
                                              a.contribution_offset) +
           route * a.hidden;
  return peer_at<const transport::BFloat16>(
             a.gin_window_base,
             a.static_network.return_rx +
                 destination_node * a.static_network.return_stride) +
         a.dispatch.local_offset[route] * a.hidden;
}

} // namespace weftmoe_pf
