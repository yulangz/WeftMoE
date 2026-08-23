#pragma once

#include <cstdint>
#include <vector>

#include "megamoe/cuda/device_notify.hpp"

namespace megamoe::cuda {

class DeviceNotifyBridge {
public:
  static LocalHistogramHandle make_local(
      const RuntimeConfig &config, const RankCollectiveContext &context,
      std::uint64_t num_tokens, SourceRouteMetadata outgoing,
      std::vector<std::uint64_t> expert_counts,
      std::vector<std::uint64_t> packet_counts,
      DeviceBufferLease device_expert_counts,
      DeviceBufferLease device_packet_counts);
};

} // namespace megamoe::cuda
