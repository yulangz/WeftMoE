#include "cuda/testing/api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda_runtime_api.h>

#include "megamoe/api/errors.hpp"
#include "megamoe/metadata/serialize.hpp"
#include "cuda/activation_transport_internal.hpp"
#include "cuda/device_notify_internal.hpp"
#include "cuda/nccl_protocol_internal.hpp"
#include "cuda/testing/generation_scheduler.hpp"

namespace {

#include "cuda/notify_fixtures.inc"
#include "cuda/control_scenarios.inc"
#include "cuda/device_lifecycle_scenarios.inc"
#include "cuda/backward_runner.inc"
#include "cuda/distributed_runner.inc"
#include "cuda/transport_benchmarks.inc"
#include "cuda/scheduler_model_scenario.inc"

} // namespace

#include "cuda/notify_cli.inc"
