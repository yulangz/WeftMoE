#pragma once

namespace weftmoe::cuda::communication {

inline constexpr unsigned kReturnReadyBlockRows = 32U;
struct alignas(128) ReturnRowQueue {
  unsigned long long reserved = 0;
  unsigned long long head = 0;
  unsigned long long begin = 0;
  unsigned long long capacity = 0;
  unsigned long long ready_begin = 0;
};

} // namespace weftmoe::cuda::communication
