#pragma once

#include "cuda/cuda_internal.hpp"
#include <nccl_device.h>
#include <tuple>

namespace weftmoe::cuda::internal {

// Keep connections/queues with the session, independently of registered payload
// windows. A live pool holds a lease; only idle communicators may be reused.
// Pool creation/release remains collective: ranks must have the same lease
// history, just as they must enter ncclDevCommCreate in the same order.
struct DeviceCommunicator {
  ncclComm_t host = nullptr;
  unsigned device = 0;
  ncclDevCommRequirements_t requirements{};
  ncclDevComm_t value{};
  bool created = false;
  bool abandoned = false;

  ~DeviceCommunicator() noexcept {
    if (created && !abandoned) {
      (void)cudaSetDevice(static_cast<int>(device));
      (void)ncclDevCommDestroy(host, &value);
    }
  }
};

inline auto device_communicator_key(const ncclDevCommRequirements_t &r) {
  // Both callers use the default requirements initializer and change only
  // these fields. No caller supplies linked resource/team requirement lists.
  return std::tie(r.barrierCount, r.lsaBarrierCount, r.ginContextCount,
                  r.ginSignalCount, r.ginCounterCount, r.ginConnectionType,
                  r.ginExclusiveContexts, r.ginQueueDepth, r.ginType);
}

inline std::shared_ptr<DeviceCommunicator>
lease_device_communicator(const NcclSession &session,
                          const ncclDevCommRequirements_t &requirements) {
  auto state = NcclSessionAccess::checked(session);
  std::lock_guard<std::mutex> lock(state->device_communicator_mutex);
  for (const auto &connection : state->device_communicators)
    if (connection.use_count() == 1 && !connection->abandoned &&
        device_communicator_key(connection->requirements) ==
            device_communicator_key(requirements))
      return connection;
  auto connection = std::make_shared<DeviceCommunicator>();
  connection->host = state->comm;
  connection->device = state->device;
  connection->requirements = requirements;
  throw_nccl("device-communicator",
             ncclDevCommCreate(state->comm, &requirements, &connection->value),
             state->rank, state->world, session.session().identity(),
             session.session().generation(), ExchangeKind::ExpertCounts,
             "create session-owned device communicator");
  connection->created = true;
  state->device_communicators.push_back(connection);
  return connection;
}

} // namespace weftmoe::cuda::internal
