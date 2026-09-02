#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace megamoe::cuda::transport_state {

enum class TaskClass : std::uint32_t {
  DispatchWave = 0,
  RelayWave = 1,
  Fc1 = 2,
  Fc2 = 3,
  Return = 4,
  SourceArrival = 5,
  Reduce = 6,
  Count = 7,
};

constexpr std::size_t kTaskClassCount =
    static_cast<std::size_t>(TaskClass::Count);
#if MEGAMOE_DEBUG_VALIDATE
constexpr std::size_t kForwardDebugCtas = 160;
#endif

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
  unsigned long long bulk_local_ready;
  unsigned long long bulk_pack_complete;
  unsigned long long bulk_contexts_issuing;
  unsigned long long fc1_scheduler_value;
  unsigned long long fc2_scheduler_value;
  unsigned long long local_rows_published;
  unsigned long long remote_rows_published;
  unsigned long long published_tile_mask;
  unsigned long long first_waiting_tile;
  unsigned long long first_wait_expected;
  unsigned long long first_wait_arrived;
  unsigned long long tile_waiter_count;
#if MEGAMOE_DEBUG_VALIDATE
  std::array<unsigned long long, 8> first_wait_arrived_by_rail;
  // Aggregate CTA checkpoints isolate a Forward FC2 stall without adding any
  // instrumentation to release kernels. A count drop between adjacent phases
  // identifies the exact task interval that did not retire.
  std::array<unsigned long long, 8> forward_fc2_phase_counts;
  std::array<unsigned long long, 8> forward_fc2_phase_last_task;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_tiles_started;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_tiles_completed;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_last_task;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_pipeline_iteration;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_producer_checkpoint;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_math0_checkpoint;
  std::array<unsigned long long, kForwardDebugCtas>
      forward_fc2_cta_math1_checkpoint;
#endif
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
  unsigned long long dispatch_fanout_rows;
  unsigned long long dispatch_tile0_rows;
  unsigned long long relay_fanout_rows;
  unsigned long long relay_tile0_rows;
};

static_assert(sizeof(TaskCompletionRecord) == 4U * sizeof(std::uint64_t));
static_assert(sizeof(ContextCompletionRecord) == 21U * sizeof(std::uint64_t));
static_assert(sizeof(GenerationCompletionVector) % sizeof(std::uint64_t) == 0U);

constexpr std::size_t kGenerationCompletionWords =
    sizeof(GenerationCompletionVector) / sizeof(std::uint64_t);
constexpr std::size_t kContextCompletionWords =
    sizeof(ContextCompletionRecord) / sizeof(std::uint64_t);

constexpr std::size_t completion_debug_words(std::size_t contexts) {
  return kGenerationCompletionWords + contexts * kContextCompletionWords;
}

} // namespace megamoe::cuda::transport_state
