#include "cuda/nccl_protocol_internal.hpp"

#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cuda/cuda_internal.hpp"
#include "cuda/device_notify_internal.hpp"
#include "cuda/nccl_protocol_internal.hpp"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
#include "transport/nccl_count_exchange/identity_and_slots.inc"
#include "transport/nccl_count_exchange/control_operations.inc"
#include "transport/nccl_count_exchange/allgather_backend.inc"
#include "transport/nccl_count_exchange/resource_pool.inc"
#include "transport/nccl_count_exchange/generation_coordinator.inc"

} // namespace megamoe::cuda
