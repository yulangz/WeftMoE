#pragma once

// Bind every accumulator to the wait instruction itself.  A separate empty
// asm before wgmma.wait_group can make ptxas conservatively treat the
// accumulator as an early non-WGMMA use inside a large persistent kernel and
// serialize every HGMMA.  Read/write constraints keep the values live through
// wait_group 0 without introducing such an intervening pseudo-use.
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)

#define MEGAMOE_WGMMA_ACCUMULATORS_64(array, offset) \
    "+f"((array)[(offset) + 0]), \
    "+f"((array)[(offset) + 1]), \
    "+f"((array)[(offset) + 2]), \
    "+f"((array)[(offset) + 3]), \
    "+f"((array)[(offset) + 4]), \
    "+f"((array)[(offset) + 5]), \
    "+f"((array)[(offset) + 6]), \
    "+f"((array)[(offset) + 7]), \
    "+f"((array)[(offset) + 8]), \
    "+f"((array)[(offset) + 9]), \
    "+f"((array)[(offset) + 10]), \
    "+f"((array)[(offset) + 11]), \
    "+f"((array)[(offset) + 12]), \
    "+f"((array)[(offset) + 13]), \
    "+f"((array)[(offset) + 14]), \
    "+f"((array)[(offset) + 15]), \
    "+f"((array)[(offset) + 16]), \
    "+f"((array)[(offset) + 17]), \
    "+f"((array)[(offset) + 18]), \
    "+f"((array)[(offset) + 19]), \
    "+f"((array)[(offset) + 20]), \
    "+f"((array)[(offset) + 21]), \
    "+f"((array)[(offset) + 22]), \
    "+f"((array)[(offset) + 23]), \
    "+f"((array)[(offset) + 24]), \
    "+f"((array)[(offset) + 25]), \
    "+f"((array)[(offset) + 26]), \
    "+f"((array)[(offset) + 27]), \
    "+f"((array)[(offset) + 28]), \
    "+f"((array)[(offset) + 29]), \
    "+f"((array)[(offset) + 30]), \
    "+f"((array)[(offset) + 31]), \
    "+f"((array)[(offset) + 32]), \
    "+f"((array)[(offset) + 33]), \
    "+f"((array)[(offset) + 34]), \
    "+f"((array)[(offset) + 35]), \
    "+f"((array)[(offset) + 36]), \
    "+f"((array)[(offset) + 37]), \
    "+f"((array)[(offset) + 38]), \
    "+f"((array)[(offset) + 39]), \
    "+f"((array)[(offset) + 40]), \
    "+f"((array)[(offset) + 41]), \
    "+f"((array)[(offset) + 42]), \
    "+f"((array)[(offset) + 43]), \
    "+f"((array)[(offset) + 44]), \
    "+f"((array)[(offset) + 45]), \
    "+f"((array)[(offset) + 46]), \
    "+f"((array)[(offset) + 47]), \
    "+f"((array)[(offset) + 48]), \
    "+f"((array)[(offset) + 49]), \
    "+f"((array)[(offset) + 50]), \
    "+f"((array)[(offset) + 51]), \
    "+f"((array)[(offset) + 52]), \
    "+f"((array)[(offset) + 53]), \
    "+f"((array)[(offset) + 54]), \
    "+f"((array)[(offset) + 55]), \
    "+f"((array)[(offset) + 56]), \
    "+f"((array)[(offset) + 57]), \
    "+f"((array)[(offset) + 58]), \
    "+f"((array)[(offset) + 59]), \
    "+f"((array)[(offset) + 60]), \
    "+f"((array)[(offset) + 61]), \
    "+f"((array)[(offset) + 62]), \
    "+f"((array)[(offset) + 63])

__device__ __forceinline__ void
backward_v3_wgmma_wait0_accum128(float (&accum)[128]) {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n"
               : MEGAMOE_WGMMA_ACCUMULATORS_64(accum, 0),
                 MEGAMOE_WGMMA_ACCUMULATORS_64(accum, 64)
               :
               : "memory");
}

__device__ __forceinline__ void
backward_v3_wgmma_wait0_accum64(float (&accum)[64]) {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n"
               : MEGAMOE_WGMMA_ACCUMULATORS_64(accum, 0)
               :
               : "memory");
}

__device__ __forceinline__ void
backward_v3_wgmma_wait0_accum64x2(float (&accum_a)[64],
                                  float (&accum_b)[64]) {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n"
               : MEGAMOE_WGMMA_ACCUMULATORS_64(accum_a, 0),
                 MEGAMOE_WGMMA_ACCUMULATORS_64(accum_b, 0)
               :
               : "memory");
}

#undef MEGAMOE_WGMMA_ACCUMULATORS_64

#endif
