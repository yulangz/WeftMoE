#pragma once

#include "cuda/profiling/device_timeline_types.hpp"
#include "weftmoe/cuda/activation_transport.hpp"

#if WEFTMOE_DEVICE_TIMELINE
WEFTMOE_PUBLIC_API_BEGIN
namespace weftmoe::cuda::testing {
profiling::DeviceTimelineSnapshot
download_forward_device_timeline(const DeviceForwardHandle &);
}
WEFTMOE_PUBLIC_API_END
#endif
