#pragma once

#include "weftmoe/cuda/device_notify.hpp"

namespace weftmoe::cuda {

class DeviceNotifyBridge {
public:
  static DispatchHandle make_validation_dispatch(DispatchMetadata metadata);
};

} // namespace weftmoe::cuda
