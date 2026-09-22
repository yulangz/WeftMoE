#pragma once

namespace static_v4 {

// CUDA atomic_ref keeps C++'s generic read/modify/write contract even when the
// returned value is dead.  Static V4's publication counters only need the
// one-way release reduction used by MegaMOE; spelling it in PTX avoids an
// unnecessary return dependency in every destination-pull warp.
__device__ __forceinline__ void reduce_add_release_device(
    unsigned long long *address, unsigned long long value) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;"
               :
               : "l"(address), "l"(value));
}

__device__ __forceinline__ void reduce_add_release_system(int *address,
                                                           int value) {
  asm volatile("red.release.sys.global.add.s32 [%0], %1;"
               :
               : "l"(address), "r"(value));
}

__device__ __forceinline__ unsigned long long
load_acquire_device(const unsigned long long *address) {
  unsigned long long value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address));
  return value;
}

__device__ __forceinline__ int load_acquire_system(const int *address) {
  int value;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
               : "=r"(value)
               : "l"(address));
  return value;
}

} // namespace static_v4
