#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace megamoe::cuda::transport_state {

enum class TaskClass : std::uint32_t {
  DispatchWave = 0,
  RelayWave = 1,
  Fc1OrDv = 2,
  Fc2OrDx = 3,
  Return = 4,
  SourceArrival = 5,
  Reduce = 6,
  Dw1 = 7,
  Dw2 = 8,
  Count = 9,
};

constexpr std::size_t kTaskClassCount =
    static_cast<std::size_t>(TaskClass::Count);

struct TaskCompletionRecord {
  unsigned long long expected;
  unsigned long long claimed;
  unsigned long long completed;
  unsigned long long inflight;
};

// One immutable task space is established before any worker can claim work.
// Counters are monotonically increasing within an epoch.  The elected terminal
// CTA is the only writer of kernel_complete_epoch and terminal_publish_count.
struct GenerationCompletionVector {
  unsigned long long generation;
  unsigned long long epoch;
  std::array<TaskCompletionRecord, kTaskClassCount> tasks;
  unsigned long long dispatch_local_completion_expected;
  unsigned long long dispatch_local_completion_completed;
  unsigned long long return_local_completion_expected;
  unsigned long long return_local_completion_completed;
  unsigned long long ring_credit_expected;
  unsigned long long ring_credit_restored;
  unsigned long long directory_expected;
  unsigned long long directory_ready;
  unsigned long long progress_ctas_at_bulk_return;
  unsigned long long compute_ctas_at_bulk_return;
  unsigned long long grid_ctas_expected;
  unsigned long long local_rows_published;
  unsigned long long remote_rows_published;
  unsigned long long published_tile_mask;
  unsigned long long kernel_complete_epoch;
  unsigned long long terminal_publish_count;
};

// Fixed per-(remote node, GIN context) directory and timeout diagnostics.
// A zero-task context is represented by expected==completed==0 after it has
// published directory_ready; it remains in the terminal protocol.
struct ContextCompletionRecord {
  unsigned long long generation;
  unsigned long long epoch;
  unsigned long long directory_ready;
  unsigned long long dispatch_expected;
  unsigned long long dispatch_claimed;
  unsigned long long dispatch_completed;
  unsigned long long relay_expected;
  unsigned long long relay_claimed;
  unsigned long long relay_completed;
  unsigned long long return_expected;
  unsigned long long return_claimed;
  unsigned long long return_completed;
  unsigned long long inflight;
  unsigned long long local_completion;
  unsigned long long credit;
  unsigned long long last_cursor;
  unsigned long long last_progress_globaltimer;
};

static_assert(sizeof(TaskCompletionRecord) == 4U * sizeof(std::uint64_t));
static_assert(sizeof(ContextCompletionRecord) == 17U * sizeof(std::uint64_t));
static_assert(sizeof(GenerationCompletionVector) % sizeof(std::uint64_t) == 0U);

constexpr std::size_t kGenerationCompletionWords =
    sizeof(GenerationCompletionVector) / sizeof(std::uint64_t);
constexpr std::size_t kContextCompletionWords =
    sizeof(ContextCompletionRecord) / sizeof(std::uint64_t);

constexpr std::size_t completion_debug_words(std::size_t contexts) {
  return kGenerationCompletionWords + contexts * kContextCompletionWords;
}

} // namespace megamoe::cuda::transport_state
