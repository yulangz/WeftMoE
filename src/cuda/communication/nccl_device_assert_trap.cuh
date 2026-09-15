#pragma once

// Parse CUDA and libc assertion declarations before temporarily redirecting
// only the failure calls expanded inside NCCL device headers. In particular,
// glibc's repeatable assert.h keeps its declarations under _ASSERT_H_DECLS.
#include <cuda_runtime.h>
#include <cassert>

// Keep CUDA's detailed assertion diagnostics in validation builds. The normal
// path retains each condition, but a failure raises an inline device trap
// instead of calling CUDA's external assertion reporter.
#if defined(__CUDA_ARCH__) && !WEFTMOE_DEBUG_VALIDATE
namespace weftmoe::cuda::nccl_assert_detail {
__device__ __forceinline__ void fail(const char *, const char *, unsigned,
                                    const char *) {
  asm volatile("trap;" ::: "memory");
}
} // namespace weftmoe::cuda::nccl_assert_detail

#pragma push_macro("__assertfail")
#pragma push_macro("__assert_fail")
#undef __assertfail
#undef __assert_fail
#define __assertfail(message, file, line, function, char_size) \
  ::weftmoe::cuda::nccl_assert_detail::fail(message, file, line, function)
#define __assert_fail(message, file, line, function) \
  ::weftmoe::cuda::nccl_assert_detail::fail(message, file, line, function)
#endif

#include <nccl_device.h>

#if defined(__CUDA_ARCH__) && !WEFTMOE_DEBUG_VALIDATE
#pragma pop_macro("__assert_fail")
#pragma pop_macro("__assertfail")
#endif
