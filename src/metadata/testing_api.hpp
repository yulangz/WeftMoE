#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#define private public
#include "megamoe/api/notify_dispatch.hpp"
#undef private

namespace megamoe::testing {

class HostIntegerRowTestAccess {
public:
  static std::weak_ptr<const void> lifetime(
      const HostIntegerRow &row) noexcept;
};

} // namespace megamoe::testing

namespace megamoe {
std::vector<DispatchHandle> build_notify_dispatch_for_testing(
    const RuntimeConfig &, const std::vector<RankRoutingInput> &,
    const std::string &communicator_id = "host-fake",
    std::uint64_t generation = 1);
} // namespace megamoe
