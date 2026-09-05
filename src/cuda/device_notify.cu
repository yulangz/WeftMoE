#include "cuda/device_notify_internal.hpp"
#include "cuda/activation_transport_internal.hpp"

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cuda/cuda_internal.hpp"
#include "cuda/device_notify_bridge.hpp"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {

namespace {
std::atomic<std::uint64_t> g_cuda_device_allocations{0};
std::atomic<std::uint64_t> g_cuda_device_frees{0};
std::atomic<std::uint64_t> g_cuda_pinned_allocations{0};
std::atomic<std::uint64_t> g_cuda_pinned_frees{0};
std::atomic<std::uint64_t> g_cuda_event_creates{0};
std::atomic<std::uint64_t> g_cuda_event_destroys{0};
std::atomic<std::uint64_t> g_cuda_device_synchronizes{0};
std::atomic<std::uint64_t> g_cuda_h2d_bytes{0};
std::atomic<std::uint64_t> g_cuda_d2h_bytes{0};
std::atomic<std::uint64_t> g_cuda_d2d_bytes{0};
#if MEGAMOE_BUILD_TESTING
std::atomic<std::uint8_t> g_consumer_fault{0};

bool take_consumer_fault(testing::ConsumerFaultPoint point) noexcept {
  auto expected = static_cast<std::uint8_t>(point);
  return g_consumer_fault.compare_exchange_strong(
      expected,
      static_cast<std::uint8_t>(testing::ConsumerFaultPoint::None));
}
#else
template <typename Point>
constexpr bool take_consumer_fault(Point) noexcept {
  return false;
}
#endif
} // namespace

#if MEGAMOE_BUILD_TESTING
void testing::inject_next_consumer_fault(ConsumerFaultPoint point) noexcept {
  g_consumer_fault.store(static_cast<std::uint8_t>(point));
}
#endif

namespace internal {
void record_cuda_api(CudaApiAuditKind kind, std::uint64_t bytes) noexcept {
  switch (kind) {
  case CudaApiAuditKind::DeviceAllocate: ++g_cuda_device_allocations; break;
  case CudaApiAuditKind::DeviceFree: ++g_cuda_device_frees; break;
  case CudaApiAuditKind::PinnedAllocate: ++g_cuda_pinned_allocations; break;
  case CudaApiAuditKind::PinnedFree: ++g_cuda_pinned_frees; break;
  case CudaApiAuditKind::EventCreate: ++g_cuda_event_creates; break;
  case CudaApiAuditKind::EventDestroy: ++g_cuda_event_destroys; break;
  case CudaApiAuditKind::DeviceSynchronize:
    ++g_cuda_device_synchronizes;
    break;
  case CudaApiAuditKind::HostToDevice: g_cuda_h2d_bytes += bytes; break;
  case CudaApiAuditKind::DeviceToHost: g_cuda_d2h_bytes += bytes; break;
  case CudaApiAuditKind::DeviceToDevice: g_cuda_d2d_bytes += bytes; break;
  }
}
} // namespace internal

#if MEGAMOE_BUILD_TESTING
testing::CudaApiAuditSnapshot testing::cuda_api_audit_snapshot() noexcept {
  return {g_cuda_device_allocations.load(), g_cuda_device_frees.load(),
          g_cuda_pinned_allocations.load(), g_cuda_pinned_frees.load(),
          g_cuda_event_creates.load(), g_cuda_event_destroys.load(),
          g_cuda_device_synchronizes.load(), g_cuda_h2d_bytes.load(),
          g_cuda_d2h_bytes.load(), g_cuda_d2d_bytes.load()};
}
#endif

namespace {
#include "cuda/notify/workspace_layout.inc"
#include "cuda/notify/materialization_kernels.inc"
} // namespace

#include "cuda/notify/execution_and_inputs.inc"
#include "cuda/notify/workspace_pool.inc"
#include "cuda/notify/notify_operation.inc"
#include "cuda/notify/dispatch_handle.inc"
#include "cuda/notify/dispatch_bridge.inc"

} // namespace megamoe::cuda
