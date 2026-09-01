#include "cuda/persistent_v3/testing.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "cuda/persistent_v3/roles/compute_role.cuh"
#include "cuda/persistent_v3/roles/rdma_role.cuh"
#include "cuda/persistent_v3/roles/relay_role.cuh"
#include "cuda/persistent_v3/shared_storage.cuh"

namespace megamoe::cuda::persistent_v3 {
namespace {

inline constexpr std::uint32_t kProbeQueueCapacity = 16U;
inline constexpr std::uint32_t kProbeMaxTasksPerContext = 128U;
inline constexpr std::uint32_t kProbeMaxTasks =
    kRdmaCtaCount * kProbeMaxTasksPerContext;
inline constexpr unsigned long long kProbeEpoch = 17ULL;

struct alignas(128) ProtocolProbeWorkspace {
  GenerationState generation;
  SpscRing<PipelineTask, kProbeQueueCapacity> rdma_to_relay[kRdmaCtaCount];
  SpscRing<PipelineTask, kProbeQueueCapacity> relay_to_rdma[kRdmaCtaCount];
  ScoreboardTask tasks[kProbeMaxTasks];
  PaddedCounter compute_scan_cursor;
  PaddedCounter compute_completed;
  PaddedCounter context_compute_completed[kRdmaCtaCount];
  PaddedCounter context_checksum[kRdmaCtaCount];
  PaddedCounter rdma_roles_completed;
  PaddedCounter relay_roles_completed;
};

MEGAMOE_V3_HOST_DEVICE constexpr unsigned long long
probe_input_value(std::uint32_t context, std::uint32_t local_task,
                  std::uint32_t global_task) {
  return (static_cast<unsigned long long>(context + 1U) << 56U) ^
         (static_cast<unsigned long long>(local_task + 1U) << 24U) ^
         (static_cast<unsigned long long>(global_task) *
          0x9e3779b97f4a7c15ULL);
}

MEGAMOE_V3_HOST_DEVICE constexpr unsigned long long
probe_output_value(unsigned long long input, std::uint32_t global_task) {
  return (input ^ 0xd1b54a32d192ed03ULL) +
         static_cast<unsigned long long>(global_task + 1U) * 128ULL;
}

__global__ void initialize_protocol_probe_kernel(ProtocolProbeWorkspace *work) {
  const auto global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_threads = gridDim.x * blockDim.x;
  for (std::uint32_t context = 0; context < kRdmaCtaCount; ++context) {
    for (std::uint32_t slot = global_thread; slot < kProbeQueueCapacity;
         slot += global_threads) {
      work->rdma_to_relay[context].slots[slot].sequence = slot;
      work->relay_to_rdma[context].slots[slot].sequence = slot;
    }
  }
}

MEGAMOE_V3_DEVICE inline void run_protocol_rdma_role(
    ProtocolProbeWorkspace &work, std::uint32_t context,
    std::uint32_t tasks_per_context) {
  if (threadIdx.x == 0U) {
    for (std::uint32_t local_task = 0; local_task < tasks_per_context;
         ++local_task) {
      const auto global_task = context * tasks_per_context + local_task;
      PipelineTask task{};
      task.epoch = kProbeEpoch;
      task.index = global_task;
      task.kind = static_cast<std::uint8_t>(TaskKind::DispatchRelay);
      task.context = static_cast<std::uint8_t>(context);
      task.row_begin = local_task * kMBlockRows;
      task.row_count = kMBlockRows;
      if (!rdma_publish_to_relay(work.rdma_to_relay[context], work.generation,
                                 task))
        break;
      fetch_add_release<::cuda::thread_scope_device>(
          work.generation.contexts[context].rx_published.value, 1ULL);
    }

    PipelineTask completion{};
    if (!generation_cancelled(work.generation, kProbeEpoch) &&
        rdma_consume_from_relay(work.relay_to_rdma[context], work.generation,
                                kProbeEpoch, completion)) {
      if (completion.epoch != kProbeEpoch ||
          completion.context != context ||
          completion.index != tasks_per_context) {
        publish_error_once(work.generation, 101ULL);
      } else {
        fetch_add_release<::cuda::thread_scope_device>(
            work.generation.contexts[context].tx_local_complete.value, 1ULL);
      }
    }
    fetch_add_release<::cuda::thread_scope_device>(
        work.rdma_roles_completed.value, 1ULL);
    wait_for_terminal(work.generation, kProbeEpoch);
  }
  __syncthreads();
}

MEGAMOE_V3_DEVICE inline void run_protocol_relay_role(
    ProtocolProbeWorkspace &work, std::uint32_t context,
    std::uint32_t tasks_per_context) {
  if (threadIdx.x == 0U) {
    for (std::uint32_t local_task = 0; local_task < tasks_per_context;
         ++local_task) {
      PipelineTask task{};
      if (!relay_consume_from_rdma(work.rdma_to_relay[context],
                                   work.generation, kProbeEpoch, task))
        break;
      const auto expected_index = context * tasks_per_context + local_task;
      if (task.epoch != kProbeEpoch || task.context != context ||
          task.index != expected_index || task.row_count != kMBlockRows) {
        publish_error_once(work.generation, 201ULL);
        break;
      }
      auto &record = work.tasks[task.index];
      record.input = probe_input_value(context, local_task, task.index);
      record.context = context;
      publish_ready(record, kProbeEpoch);
      fetch_add_release<::cuda::thread_scope_device>(
          work.generation.contexts[context].rx_consumed.value, 1ULL);
      fetch_add_release<::cuda::thread_scope_device>(
          work.generation.pipeline
              .completed[static_cast<std::uint32_t>(TaskKind::DispatchRelay)]
              .value,
          1ULL);
    }

    while (!generation_cancelled(work.generation, kProbeEpoch) &&
           load_acquire<::cuda::thread_scope_device>(
               work.context_compute_completed[context].value) <
               tasks_per_context) {
      __nanosleep(64U);
    }

    unsigned long long checksum = 0ULL;
    if (!generation_cancelled(work.generation, kProbeEpoch)) {
      for (std::uint32_t local_task = 0; local_task < tasks_per_context;
           ++local_task) {
        auto &record =
            work.tasks[context * tasks_per_context + local_task];
        if (!task_complete(record, kProbeEpoch)) {
          publish_error_once(work.generation, 202ULL);
          break;
        }
        checksum += record.output;
      }
    }
    store_release<::cuda::thread_scope_device>(
        work.context_checksum[context].value, checksum);

    if (!generation_cancelled(work.generation, kProbeEpoch)) {
      PipelineTask completion{};
      completion.epoch = kProbeEpoch;
      completion.index = tasks_per_context;
      completion.kind = static_cast<std::uint8_t>(TaskKind::ReturnRelay);
      completion.context = static_cast<std::uint8_t>(context);
      if (relay_publish_to_rdma(work.relay_to_rdma[context], work.generation,
                                completion)) {
        fetch_add_release<::cuda::thread_scope_device>(
            work.generation.contexts[context].tx_published.value, 1ULL);
        fetch_add_release<::cuda::thread_scope_device>(
            work.generation.pipeline
                .completed[static_cast<std::uint32_t>(TaskKind::ReturnRelay)]
                .value,
            1ULL);
      }
    }
    fetch_add_release<::cuda::thread_scope_device>(
        work.relay_roles_completed.value, 1ULL);
    wait_for_terminal(work.generation, kProbeEpoch);
  }
  __syncthreads();
}

MEGAMOE_V3_DEVICE inline bool protocol_probe_terminal_ready(
    ProtocolProbeWorkspace &work, std::uint32_t total_tasks) {
  if (load_acquire<::cuda::thread_scope_device>(
          work.compute_completed.value) != total_tasks ||
      load_acquire<::cuda::thread_scope_device>(
          work.rdma_roles_completed.value) != kRdmaCtaCount ||
      load_acquire<::cuda::thread_scope_device>(
          work.relay_roles_completed.value) != kRdmaCtaCount)
    return false;

  for (std::uint32_t context = 0; context < kRdmaCtaCount; ++context) {
    if (load_acquire<::cuda::thread_scope_device>(
            work.generation.contexts[context].rx_published.value) !=
            total_tasks / kRdmaCtaCount ||
        load_acquire<::cuda::thread_scope_device>(
            work.generation.contexts[context].rx_consumed.value) !=
            total_tasks / kRdmaCtaCount ||
        load_acquire<::cuda::thread_scope_device>(
            work.generation.contexts[context].tx_published.value) != 1ULL ||
        load_acquire<::cuda::thread_scope_device>(
            work.generation.contexts[context].tx_local_complete.value) !=
            1ULL ||
        !ring_drained(work.rdma_to_relay[context]) ||
        !ring_drained(work.relay_to_rdma[context]))
      return false;
  }
  return true;
}

MEGAMOE_V3_DEVICE inline void publish_protocol_terminal(
    ProtocolProbeWorkspace &work, std::uint32_t total_tasks) {
  while (true) {
    const auto error = load_acquire<::cuda::thread_scope_device>(
        work.generation.control.error_code.value);
    if (error != 0ULL || protocol_probe_terminal_ready(work, total_tasks)) {
      auto expected = 0ULL;
      if (compare_exchange_acq_rel<::cuda::thread_scope_device>(
              work.generation.control.terminal_epoch.value, expected,
              kProbeEpoch)) {
        fetch_add_release<::cuda::thread_scope_device>(
            work.generation.control.terminal_publish_count.value, 1ULL);
      }
      return;
    }
    __nanosleep(64U);
  }
}

MEGAMOE_V3_DEVICE inline void run_protocol_compute_role(
    ProtocolProbeWorkspace &work, std::uint32_t compute_ordinal,
    std::uint32_t tasks_per_context) {
  const auto total_tasks = tasks_per_context * kRdmaCtaCount;
  if (threadIdx.x == 0U) {
    while (!generation_cancelled(work.generation, kProbeEpoch) &&
           load_acquire<::cuda::thread_scope_device>(
               work.compute_completed.value) < total_tasks) {
      const auto task_index = try_claim_ready_task(
          work.tasks, total_tasks, work.compute_scan_cursor, kProbeEpoch);
      if (task_index == kInvalidTask) {
        __nanosleep(64U);
        continue;
      }
      auto &record = work.tasks[task_index];
      const auto context = record.context;
      if (context >= kRdmaCtaCount) {
        publish_error_once(work.generation, 301ULL);
        break;
      }
      record.output = probe_output_value(record.input, task_index);
      publish_complete(record, kProbeEpoch);
      fetch_add_release<::cuda::thread_scope_device>(
          work.context_compute_completed[context].value, 1ULL);
      fetch_add_release<::cuda::thread_scope_device>(
          work.compute_completed.value, 1ULL);
      fetch_add_release<::cuda::thread_scope_device>(
          work.generation.pipeline
              .completed[static_cast<std::uint32_t>(TaskKind::Fc1)]
              .value,
          1ULL);
    }

    if (compute_ordinal == 0U)
      publish_protocol_terminal(work, total_tasks);
    else
      wait_for_terminal(work.generation, kProbeEpoch);
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreadsPerCta, 1)
void persistent_v3_protocol_probe_kernel(ProtocolProbeWorkspace *work,
                                         std::uint32_t tasks_per_context,
                                         RoleLayout roles) {
  __shared__ SharedStorage<64U, 64U, 64U> shared;
  if (threadIdx.x == 0U)
    shared.control.task = blockIdx.x;
  __syncthreads();

  const auto role = roles.role(static_cast<std::uint32_t>(shared.control.task));
  const auto ordinal =
      roles.role_ordinal(static_cast<std::uint32_t>(shared.control.task));
  if (role == WorkerRole::Rdma) {
    run_protocol_rdma_role(*work, ordinal, tasks_per_context);
  } else if (role == WorkerRole::Relay) {
    run_protocol_relay_role(*work, roles.context_for_relay(ordinal),
                            tasks_per_context);
  } else if (role == WorkerRole::Compute) {
    run_protocol_compute_role(*work, ordinal, tasks_per_context);
  } else if (threadIdx.x == 0U) {
    publish_error_once(work->generation, 401ULL);
  }
}

void throw_cuda_error(cudaError_t error, const char *operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

struct CudaWorkspaceDeleter {
  void operator()(ProtocolProbeWorkspace *pointer) const noexcept {
    if (pointer != nullptr)
      (void)cudaFree(pointer);
  }
};

} // namespace
} // namespace megamoe::cuda::persistent_v3

namespace megamoe::cuda::testing {

PersistentV3ProtocolProbeResult
run_persistent_v3_protocol_probe(std::uint32_t tasks_per_context) {
  using namespace persistent_v3;
  if (tasks_per_context == 0U ||
      tasks_per_context > kProbeMaxTasksPerContext)
    throw std::invalid_argument(
        "V3 protocol probe tasks_per_context must be in [1, 128]");

  int device = 0;
  throw_cuda_error(cudaGetDevice(&device), "cudaGetDevice(V3 probe)");
  cudaDeviceProp properties{};
  throw_cuda_error(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties(V3 probe)");
  const RoleLayout roles{static_cast<std::uint32_t>(
                             properties.multiProcessorCount),
                         kDefaultRelayCtaCount};
  if (!roles.valid())
    throw std::runtime_error(
        "V3 protocol probe requires at least 17 resident CTAs");

  int active_per_sm = 0;
  throw_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active_per_sm, persistent_v3_protocol_probe_kernel,
                       kThreadsPerCta, 0),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor(V3 probe)");
  if (active_per_sm < 1)
    throw std::runtime_error("V3 protocol probe kernel is not resident");

  ProtocolProbeWorkspace *raw_workspace = nullptr;
  throw_cuda_error(
      cudaMalloc(reinterpret_cast<void **>(&raw_workspace),
                 sizeof(ProtocolProbeWorkspace)),
      "cudaMalloc(V3 protocol probe)");
  std::unique_ptr<ProtocolProbeWorkspace, CudaWorkspaceDeleter> workspace(
      raw_workspace);
  throw_cuda_error(cudaMemset(raw_workspace, 0, sizeof(ProtocolProbeWorkspace)),
                   "cudaMemset(V3 protocol probe)");

  initialize_protocol_probe_kernel<<<1, 256>>>(raw_workspace);
  throw_cuda_error(cudaGetLastError(), "initialize V3 protocol probe launch");
  persistent_v3_protocol_probe_kernel<<<roles.total_ctas, kThreadsPerCta>>>(
      raw_workspace, tasks_per_context, roles);
  throw_cuda_error(cudaGetLastError(), "persistent V3 protocol probe launch");
  throw_cuda_error(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize(V3 protocol probe)");

  auto host = std::make_unique<ProtocolProbeWorkspace>();
  throw_cuda_error(cudaMemcpy(host.get(), raw_workspace, sizeof(*host),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(V3 protocol probe result)");

  PersistentV3ProtocolProbeResult result;
  result.sm_count = roles.total_ctas;
  result.rdma_ctas = kRdmaCtaCount;
  result.relay_ctas = roles.relay_ctas;
  result.compute_ctas = roles.compute_ctas();
  result.tasks_per_context = tasks_per_context;
  result.tasks_completed =
      host->generation.pipeline
          .completed[static_cast<std::uint32_t>(TaskKind::Fc1)]
          .value;
  result.terminal_epoch = host->generation.control.terminal_epoch.value;
  result.terminal_publish_count =
      host->generation.control.terminal_publish_count.value;
  result.error_code = host->generation.control.error_code.value;
  for (std::uint32_t context = 0; context < kRdmaCtaCount; ++context)
    result.observed_checksum += host->context_checksum[context].value;
  for (std::uint32_t context = 0; context < kRdmaCtaCount; ++context) {
    for (std::uint32_t local_task = 0; local_task < tasks_per_context;
         ++local_task) {
      const auto global_task = context * tasks_per_context + local_task;
      result.expected_checksum += probe_output_value(
          probe_input_value(context, local_task, global_task), global_task);
    }
  }
  return result;
}

} // namespace megamoe::cuda::testing
