#pragma once

#include <cstddef>
#include <cstdint>

#include "megamoe/api/notify_dispatch.hpp"

namespace megamoe::testing {

// Internal sanitizer-only/test-only hook. This header is intentionally outside
// the public include tree.
class NotifyCorruptionHook {
public:
  static LocalHistogramHandle with_expert_id(const LocalHistogramHandle &source,
                                             std::size_t route_index,
                                             std::uint32_t expert_id);
  static LocalHistogramHandle
  with_truncated_expert_ids(const LocalHistogramHandle &source);
};

} // namespace megamoe::testing
