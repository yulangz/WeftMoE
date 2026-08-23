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

} // namespace

namespace testing {

void set_bootstrap_io_hooks(BootstrapReadHook new_read,
                            BootstrapWriteHook new_write) noexcept {
  read_hook = new_read;
  write_hook = new_write;
}

void clear_bootstrap_io_hooks() noexcept {
  read_hook = nullptr;
  write_hook = nullptr;
}

} // namespace testing

void NcclBootstrapIo::write_fd(const NcclBootstrapId &id, int fd) {
  if (fd < 0 || id.bytes_.size() != kWireBytes)
    invalid_io("bootstrap-channel-write", "descriptor/bytes",
               "valid fd and 128 bytes", "invalid");
  std::size_t offset = 0;
  while (offset < id.bytes_.size()) {
    const auto written = invoke_write(
        fd, id.bytes_.data() + offset, id.bytes_.size() - offset);
    if (written < 0 && errno == EINTR)
      continue;
    if (written < 0)
      invalid_io("bootstrap-channel-write", "write", "128 bytes",
                 "I/O failure");
    if (written == 0)
      invalid_io("bootstrap-channel-write", "write-progress", ">0", "0");
    const auto count = static_cast<std::size_t>(written);
    if (count > id.bytes_.size() - offset)
      invalid_io("bootstrap-channel-write", "write-count",
                 "at most remaining frame bytes", std::to_string(count));
    offset += count;
  }
}

NcclBootstrapId NcclBootstrapIo::read_fd(int fd) {
  if (fd < 0)
    invalid_io("bootstrap-channel-read", "descriptor", "valid fd",
               "invalid");
  std::vector<std::uint8_t> bytes(kWireBytes);
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto received =
        invoke_read(fd, bytes.data() + offset, bytes.size() - offset);
    if (received < 0 && errno == EINTR)
      continue;
    if (received < 0)
      invalid_io("bootstrap-channel-read", "read", "128 bytes",
                 "I/O failure");
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
    const auto received = invoke_read(fd, &trailing, 1U);
    if (received < 0 && errno == EINTR)
      continue;
    if (received < 0)
      invalid_io("bootstrap-channel-read", "frame-eof", "EOF",
                 "I/O failure");
    if (received == 0)
      break;
    invalid_io("bootstrap-channel-read", "frame-length", "128", ">128");
  }
  return NcclBootstrapId(std::move(bytes));
}

} // namespace megamoe::cuda
