#include "metadata/testing_api.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <signal.h>
#include <unistd.h>

#include "megamoe/api/errors.hpp"
#include "megamoe/cuda/control_protocol.hpp"
#include "megamoe/kernels/persistent_contract.hpp"
#include "megamoe/metadata/serialize.hpp"
#include "megamoe/scoreboard/workspace_plan.hpp"
#include "megamoe/transport/count_exchange.hpp"
#include "transport/activation_protocol_testing.hpp"
#include "cuda/nccl_protocol_internal.hpp"
#include "metadata/notify_corruption_hook.hpp"

namespace {

#include "metadata/fixtures.inc"
#include "metadata/dispatch_scenarios.inc"
#include "metadata/lifecycle_scenarios.inc"
#include "metadata/validation_scenarios.inc"
#include "metadata/control_protocol_scenarios.inc"
#include "metadata/bootstrap_scenarios.inc"

} // namespace

#include "metadata/test_main.inc"
