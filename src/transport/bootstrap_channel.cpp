#include "cuda/nccl_protocol_internal.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
namespace {

thread_local testing::BootstrapReadHook read_hook = nullptr;
thread_local testing::BootstrapWriteHook write_hook = nullptr;
thread_local testing::BootstrapCloseHook close_hook = nullptr;

[[noreturn]] void invalid_io(const std::string &phase,
                             const std::string &field,
                             const std::string &expected,
                             const std::string &actual) {
  throw NotifyDispatchError(
      NotifyErrorCode::NcclFailure,
      "phase=" + phase + ", field=" + field +
          ", rank=n/a, expected=" + expected + ", actual=" + actual);
}

std::ptrdiff_t invoke_read(int fd, void *data, std::size_t size) {
  if (read_hook != nullptr)
    return read_hook(fd, data, size);
  return static_cast<std::ptrdiff_t>(::read(fd, data, size));
}

std::ptrdiff_t invoke_write(int fd, const void *data, std::size_t size) {
  if (write_hook != nullptr)
    return write_hook(fd, data, size);
  return static_cast<std::ptrdiff_t>(::write(fd, data, size));
}

int invoke_close(int fd) noexcept {
  if (close_hook != nullptr)
    return close_hook(fd);
  return ::close(fd);
}

const char *state_name(BootstrapChannelState state) noexcept {
  switch (state) {
  case BootstrapChannelState::Open:
    return "Open";
  case BootstrapChannelState::Consumed:
    return "Consumed";
  case BootstrapChannelState::Failed:
    return "Failed";
  case BootstrapChannelState::Closed:
    return "Closed";
  }
  return "unknown";
}

} // namespace

namespace testing {

void set_bootstrap_io_hooks(BootstrapReadHook new_read,
                            BootstrapWriteHook new_write,
                            BootstrapCloseHook new_close) noexcept {
  read_hook = new_read;
  write_hook = new_write;
  close_hook = new_close;
}

void clear_bootstrap_io_hooks() noexcept {
  read_hook = nullptr;
  write_hook = nullptr;
  close_hook = nullptr;
}

} // namespace testing

NcclBootstrapInputChannel::NcclBootstrapInputChannel(int owned_fd)
    : fd_(owned_fd), state_(BootstrapChannelState::Open) {
  if (owned_fd < 0) {
    fd_ = -1;
    state_ = BootstrapChannelState::Closed;
    invalid_io("bootstrap-channel-read", "descriptor", "valid owned fd",
               std::to_string(owned_fd));
  }
}

NcclBootstrapInputChannel::~NcclBootstrapInputChannel() noexcept {
  if (state_ == BootstrapChannelState::Open)
    finish(BootstrapChannelState::Closed);
}

NcclBootstrapInputChannel::NcclBootstrapInputChannel(
    NcclBootstrapInputChannel &&other) noexcept
    : fd_(std::exchange(other.fd_, -1)), state_(other.state_) {
  other.state_ = BootstrapChannelState::Closed;
}

NcclBootstrapInputChannel &NcclBootstrapInputChannel::operator=(
    NcclBootstrapInputChannel &&other) noexcept {
  if (this == &other)
    return *this;
  if (state_ == BootstrapChannelState::Open)
    finish(BootstrapChannelState::Closed);
  fd_ = std::exchange(other.fd_, -1);
  state_ = other.state_;
  other.state_ = BootstrapChannelState::Closed;
  return *this;
}

int NcclBootstrapInputChannel::finish(
    BootstrapChannelState terminal_state) noexcept {
  const int owned_fd = std::exchange(fd_, -1);
  state_ = terminal_state;
  if (owned_fd >= 0)
    return invoke_close(owned_fd);
  return 0;
}

void NcclBootstrapInputChannel::require_open(const char *operation) const {
  if (state_ != BootstrapChannelState::Open || fd_ < 0)
    invalid_io("bootstrap-channel-read", operation, "state=Open",
               std::string("state=") + state_name(state_));
}

NcclBootstrapId NcclBootstrapInputChannel::read() {
  require_open("read-state");
  try {
    std::vector<std::uint8_t> bytes(kWireBytes);
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const auto received =
          invoke_read(fd_, bytes.data() + offset, bytes.size() - offset);
      if (received < 0 && errno == EINTR)
        continue;
      if (received < 0)
        invalid_io("bootstrap-channel-read", "read", "128 bytes",
                   "I/O failure errno=" + std::to_string(errno));
      if (received == 0)
        invalid_io("bootstrap-channel-read", "bytes", "128",
                   std::to_string(offset));
      const auto count = static_cast<std::size_t>(received);
      if (count > bytes.size() - offset)
        invalid_io("bootstrap-channel-read", "read-count",
                   "at most remaining frame bytes", std::to_string(count));
      offset += count;
    }

    std::uint8_t trailing = 0;
    while (true) {
      const auto received = invoke_read(fd_, &trailing, 1U);
      if (received < 0 && errno == EINTR)
        continue;
      if (received < 0)
        invalid_io("bootstrap-channel-read", "frame-eof", "EOF",
                   "I/O failure errno=" + std::to_string(errno));
      if (received == 0)
        break;
      invalid_io("bootstrap-channel-read", "frame-length", "128", ">=129");
    }
    const int close_result = finish(BootstrapChannelState::Consumed);
    const int close_error = errno;
    if (close_result != 0)
      invalid_io("bootstrap-channel-read", "close", "success",
                 "I/O failure errno=" + std::to_string(close_error));
    return NcclBootstrapId(std::move(bytes));
  } catch (...) {
    finish(BootstrapChannelState::Failed);
    throw;
  }
}

void NcclBootstrapInputChannel::cancel() noexcept {
  if (state_ == BootstrapChannelState::Open)
    finish(BootstrapChannelState::Closed);
}

NcclBootstrapOutputChannel::NcclBootstrapOutputChannel(int owned_fd)
    : fd_(owned_fd), state_(BootstrapChannelState::Open) {
  if (owned_fd < 0) {
    fd_ = -1;
    state_ = BootstrapChannelState::Closed;
    invalid_io("bootstrap-channel-write", "descriptor", "valid owned fd",
               std::to_string(owned_fd));
  }
}

NcclBootstrapOutputChannel::~NcclBootstrapOutputChannel() noexcept {
  if (state_ == BootstrapChannelState::Open)
    finish(BootstrapChannelState::Closed);
}

NcclBootstrapOutputChannel::NcclBootstrapOutputChannel(
    NcclBootstrapOutputChannel &&other) noexcept
    : fd_(std::exchange(other.fd_, -1)), state_(other.state_) {
  other.state_ = BootstrapChannelState::Closed;
}

NcclBootstrapOutputChannel &NcclBootstrapOutputChannel::operator=(
    NcclBootstrapOutputChannel &&other) noexcept {
  if (this == &other)
    return *this;
  if (state_ == BootstrapChannelState::Open)
    finish(BootstrapChannelState::Closed);
  fd_ = std::exchange(other.fd_, -1);
  state_ = other.state_;
  other.state_ = BootstrapChannelState::Closed;
  return *this;
}

int NcclBootstrapOutputChannel::finish(
    BootstrapChannelState terminal_state) noexcept {
  const int owned_fd = std::exchange(fd_, -1);
  state_ = terminal_state;
  if (owned_fd >= 0)
    return invoke_close(owned_fd);
  return 0;
}

void NcclBootstrapOutputChannel::require_open(const char *operation) const {
  if (state_ != BootstrapChannelState::Open || fd_ < 0)
    invalid_io("bootstrap-channel-write", operation, "state=Open",
               std::string("state=") + state_name(state_));
}

void NcclBootstrapOutputChannel::write(const NcclBootstrapId &id) {
  require_open("write-state");
  try {
    if (id.bytes_.size() != kWireBytes)
      invalid_io("bootstrap-channel-write", "bytes", "128",
                 std::to_string(id.bytes_.size()));
    std::size_t offset = 0;
    while (offset < id.bytes_.size()) {
      const auto written = invoke_write(
          fd_, id.bytes_.data() + offset, id.bytes_.size() - offset);
      if (written < 0 && errno == EINTR)
        continue;
      if (written < 0)
        invalid_io("bootstrap-channel-write", "write", "128 bytes",
                   "I/O failure errno=" + std::to_string(errno));
      if (written == 0)
        invalid_io("bootstrap-channel-write", "write-progress", ">0", "0");
      const auto count = static_cast<std::size_t>(written);
      if (count > id.bytes_.size() - offset)
        invalid_io("bootstrap-channel-write", "write-count",
                   "at most remaining frame bytes", std::to_string(count));
      offset += count;
    }
    const int close_result = finish(BootstrapChannelState::Consumed);
    const int close_error = errno;
    if (close_result != 0)
      invalid_io("bootstrap-channel-write", "close", "success",
                 "I/O failure errno=" + std::to_string(close_error));
  } catch (...) {
    finish(BootstrapChannelState::Failed);
    throw;
  }
}

void NcclBootstrapOutputChannel::cancel() noexcept {
  if (state_ == BootstrapChannelState::Open)
    finish(BootstrapChannelState::Closed);
}

} // namespace megamoe::cuda
