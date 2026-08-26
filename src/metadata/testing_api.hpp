#pragma once

// Host oracle and corruption declarations are repository-internal.  This
// header is intentionally outside include/megamoe and is never installed.
#ifndef MEGAMOE_INTERNAL_TESTING_API
#define MEGAMOE_INTERNAL_TESTING_API 1
#endif

#include "megamoe/api/notify_dispatch.hpp"

namespace megamoe::testing {

class HostIntegerRowTestAccess {
public:
  static std::weak_ptr<const void> lifetime(
      const HostIntegerRow &row) noexcept;
};

} // namespace megamoe::testing
