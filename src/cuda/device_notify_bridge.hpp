#pragma once

#include "megamoe/cuda/device_notify.hpp"

namespace megamoe::cuda {

class DeviceNotifyBridge {
public:
  static DispatchHandle make_validation_dispatch(DispatchMetadata metadata);
};

} // namespace megamoe::cuda
