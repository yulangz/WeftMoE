#include "metadata/testing_api.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "megamoe/api/errors.hpp"
#include "cuda/device_notify_bridge.hpp"
#include "metadata/notify_corruption_hook.hpp"

namespace megamoe {

#include "metadata/notify_dispatch/state.inc"
#include "metadata/notify_dispatch/validation.inc"
#include "metadata/notify_dispatch/handles.inc"
#include "metadata/notify_dispatch/materialization.inc"

} // namespace megamoe
