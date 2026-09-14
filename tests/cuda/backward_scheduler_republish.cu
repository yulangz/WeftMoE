#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda/persistent_v3/backward/scheduler.cuh"

namespace bwd = weftmoe::cuda::persistent_v3::backward;

struct Case {
  unsigned old_prefix, new_prefix, owner;
};
constexpr unsigned kOwners = 4226;
constexpr unsigned kReadyWords = (kOwners + 63U) / 64U;
struct State {
  unsigned long long ready[kReadyWords];
  unsigned long long summary[(kReadyWords + 63U) / 64U];
  unsigned long long claims[2U * kOwners];
};

// Force the interleaving that stranded a Ling expert block: observe a drained
// dU phase, publish the epilogue, then finish retiring the old ready summary.
__global__ void check_republish(const Case *cases, State *states,
                                unsigned *failures) {
  auto *ready = states[blockIdx.x].ready;
  auto *claims = states[blockIdx.x].claims;
  auto *summary = states[blockIdx.x].summary;
  __shared__ unsigned drained;
  const auto c = cases[blockIdx.x];
  for (unsigned i = threadIdx.x; i < 2U * kOwners; i += blockDim.x)
    claims[i] = 0;
  for (unsigned i = threadIdx.x; i < kReadyWords; i += blockDim.x)
    ready[i] = 0;
  if (threadIdx.x < (kReadyWords + 63U) / 64U)
    summary[threadIdx.x] = 0;
  __syncthreads();
  bwd::TileDomain domain{ready,  claims, kOwners, 128, bwd::TaskKind::Dab,
                         summary};
  if (threadIdx.x == 0) {
    bwd::publish_owner_prefix(domain, c.owner, c.old_prefix);
    bwd::TileTask task;
    for (unsigned i = 0; i < c.old_prefix; ++i)
      if (!bwd::try_claim_owner_tile(domain, c.owner, task))
        failures[blockIdx.x] |= 1;
  }
  __syncthreads();
  if (threadIdx.x == 1)
    drained = bwd::owner_fully_claimed(domain, c.owner);
  __syncthreads();
  if (threadIdx.x == 0)
    bwd::publish_owner_prefix(domain, c.owner, c.new_prefix);
  __syncthreads();
  if (threadIdx.x == 1 && drained)
    bwd::retire_claimed_owner(domain, c.owner);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!drained)
      failures[blockIdx.x] |= 2;
    bool seen[128]{};
    unsigned count = 0;
    bwd::TileTask task;
    while (bwd::try_claim_domain_once(domain, task)) {
      if (task.owner != c.owner || task.tile >= c.new_prefix ||
          task.tile_count != 1 || seen[task.tile]) {
        failures[blockIdx.x] |= 4;
        break;
      }
      seen[task.tile] = true;
      ++count;
    }
    if (count != c.new_prefix)
      failures[blockIdx.x] |= 8;
    if (ready[c.owner / 64] & (1ULL << (c.owner % 64)))
      failures[blockIdx.x] |= 16;
    if (summary[c.owner / 4096] & (1ULL << ((c.owner / 64) % 64)))
      failures[blockIdx.x] |= 32;
    // A delayed empty-word retirement must not hide a newly published owner.
    bwd::publish_owner_prefix(domain, c.owner, c.new_prefix);
    bwd::retire_ready_word(domain, c.owner / 64);
    unsigned recovered = 0;
    while (bwd::try_claim_domain_once(domain, task))
      ++recovered;
    if (recovered != c.new_prefix)
      failures[blockIdx.x] |= 64;
    // The old phase's retire clears the owner, another worker retires the
    // now-empty summary word, then the first worker repairs the new phase.
    // Repairing only the owner bit would strand every remaining phase tile.
    bwd::publish_owner_prefix(domain, c.owner, c.new_prefix);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        ready[c.owner / 64U])
        .fetch_and(~(1ULL << (c.owner % 64U)), ::cuda::memory_order_acq_rel);
    bwd::retire_ready_word(domain, c.owner / 64U);
    bwd::repair_retired_owner(domain, c.owner);
    recovered = 0;
    while (bwd::try_claim_domain_once(domain, task))
      ++recovered;
    if (recovered != c.new_prefix)
      failures[blockIdx.x] |= 128;
  }
}

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(2);
  }
}

int main() {
  const Case cases[]{{3, 11, 30},   {8, 11, 0},   {12, 11, 63},  {3, 11, 65},
                     {64, 128, 1},  {128, 3, 65}, {65, 127, 65}, {1, 1, 63},
                     {3, 11, 4095}, {8, 1, 4096}, {128, 7, 4225}};
  constexpr unsigned count = sizeof(cases) / sizeof(cases[0]);
  Case *device_cases;
  State *states;
  unsigned *failures;
  check(cudaMalloc(&device_cases, sizeof(cases)));
  check(cudaMalloc(&states, count * sizeof(State)));
  check(cudaMallocManaged(&failures, count * sizeof(unsigned)));
  check(cudaMemset(failures, 0, count * sizeof(unsigned)));
  check(cudaMemcpy(device_cases, cases, sizeof(cases), cudaMemcpyHostToDevice));
  check_republish<<<count, 32>>>(device_cases, states, failures);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  unsigned failed = 0;
  for (unsigned i = 0; i < count; ++i)
    if (failures[i]) {
      std::fprintf(stderr, "case=%u failure_mask=%u\n", i, failures[i]);
      ++failed;
    }
  check(cudaFree(device_cases));
  check(cudaFree(states));
  check(cudaFree(failures));
  if (failed)
    return 1;
  std::printf("PASS backward scheduler phase republish: %u cases\n", count);
  return 0;
}
