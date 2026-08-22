#pragma once

#include <stdexcept>
#include <string>

namespace megamoe {

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
};

class NotifyDispatchError final : public std::runtime_error {
public:
    NotifyDispatchError(NotifyErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    NotifyErrorCode code() const noexcept { return code_; }

private:
    NotifyErrorCode code_;
};

}  // namespace megamoe
