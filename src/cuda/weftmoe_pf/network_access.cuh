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

} // namespace weftmoe_pf
