#pragma once

#include <stdexcept>
#include <string>

#include "weftmoe/version.hpp"

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe {

enum class NotifyErrorCode {
    InvalidRuntimeConfig,
    InvalidTopology,
    InvalidRouteShape,
    ExpertOutOfRange,
    DuplicateExpert,
    ExpertCapacityExceeded,
    WorkspaceCapacityExceeded,
    CountExchangeFailure,
    ArithmeticOverflow,
    InvalidHandle,
    WorkspaceMismatch,
    CudaFailure,
    NcclFailure,
    ControlProtocolFailure,
    ResourceExhausted,
    BackendUnavailable,
    TransportProtocolFailure,
    TransportTimeout,
};

class NotifyDispatchError final : public std::runtime_error {
public:
    NotifyDispatchError(NotifyErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    NotifyErrorCode code() const noexcept { return code_; }

private:
    NotifyErrorCode code_;
};

}  // namespace weftmoe

WEFTMOE_PUBLIC_API_END
