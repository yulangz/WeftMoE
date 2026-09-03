#pragma once

#include <cstdint>

namespace megamoe::cuda::testing {

struct PersistentV3ProtocolProbeResult {
  std::uint32_t sm_count = 0;
  std::uint32_t rdma_ctas = 0;
  std::uint32_t relay_ctas = 0;
  std::uint32_t compute_ctas = 0;
  std::uint32_t tasks_per_context = 0;
  std::uint64_t tasks_completed = 0;
  std::uint64_t expected_checksum = 0;
  std::uint64_t observed_checksum = 0;
  std::uint64_t terminal_epoch = 0;
  std::uint64_t terminal_publish_count = 0;
  std::uint64_t error_code = 0;
};

PersistentV3ProtocolProbeResult
run_persistent_v3_protocol_probe(std::uint32_t tasks_per_context = 79U);

struct BackwardSchedulerProbeResult {
  std::uint32_t sm_count = 0;
  std::uint32_t compute_ctas = 0;
  std::uint32_t participating_ctas = 0;
  std::uint64_t dx_completed = 0;
  std::uint64_t dab_completed = 0;
  std::uint64_t dw2_completed = 0;
  std::uint64_t dw1_completed = 0;
  std::uint64_t total_completed = 0;
  std::uint64_t duplicate_claims = 0;
  std::uint64_t dependency_errors = 0;
  std::uint64_t priority_errors = 0;
  std::uint64_t timeout_errors = 0;
  std::uint64_t terminal_epoch = 0;
  std::uint64_t terminal_publish_count = 0;
};

BackwardSchedulerProbeResult run_backward_scheduler_probe();

} // namespace megamoe::cuda::testing
