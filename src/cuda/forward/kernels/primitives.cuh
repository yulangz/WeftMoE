#include "cuda/forward/kernel_support.cuh"
#include "cuda/weftmoe_pf/memory_order.cuh"
#include "cuda/communication/activation_ring.cuh"

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) // compiled production MMA path only
// The paired Forward kernel gives FC1 and FC2 independent N shapes.
using TunedForwardFc1Mma =
    typename hopper::BF16MMASelector<forward_tuning::kFc1TileN>::type;
using TunedForwardFc2Mma =
    typename hopper::BF16MMASelector<forward_tuning::kFc2TileN>::type;
using ForwardBarrier = cutlass::arch::ClusterTransactionBarrier;

#endif

__device__ void forward_publish_progress(const ForwardKernelArgs &a,
                                         std::uint32_t word,
                                         unsigned long long value) {
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
  if (a.progress != nullptr && blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> field(
        a.progress[word]);
    field.store(value, ::cuda::memory_order_release);
  }
#else
  (void)a;
  (void)word;
  (void)value;
#endif
}

template <bool StaticLocal>
__device__ __forceinline__ bool remote_backend_enabled(bool enabled) {
  if constexpr (StaticLocal)
    return false;
  return enabled;
}

// Include order follows the device helpers' dependencies.
#include "cuda/weftmoe_pf/network_access.cuh"
#include "cuda/weftmoe_pf/ring_access.cuh"
#include "cuda/weftmoe_pf/return_tx.cuh"
#include "cuda/weftmoe_pf/network_ring.cuh"
#include "cuda/forward/compute_kernel.cuh"

#include "cuda/weftmoe_pf/initialize.cuh"
#include "cuda/weftmoe_pf/collective.cuh"
#include "cuda/weftmoe_pf/route_plan.cuh"
#include "cuda/weftmoe_pf/network_collective.cuh"
#include "cuda/weftmoe_pf/single_node_route_exchange.cuh"
#include "cuda/weftmoe_pf/training_plan.cuh"
#include "cuda/weftmoe_pf/ring_plan.cuh"
#include "cuda/weftmoe_pf/ring_admission.cuh"

#include "cuda/communication/worker_pipeline.cuh"
#include "cuda/weftmoe_pf/network_transport.cuh"
