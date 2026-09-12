#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "cuda/persistent_v3/backward/scheduler.cuh"

namespace bwd = weftmoe::cuda::persistent_v3::backward;

constexpr unsigned kExperts = 67;
constexpr unsigned kTiles = 73;
constexpr unsigned kActiveExperts = kExperts - (kExperts + 6) / 7;

struct State {
  unsigned long long entries[kExperts];
  unsigned long long claims[kExperts];
  unsigned long long reserved;
  unsigned long long head;
  unsigned long long completed;
  unsigned payload[kExperts];
  unsigned visits[kExperts * kTiles];
  unsigned errors;
};

__device__ bwd::ExpertTileQueue queue(State &state, unsigned granularity) {
  return {state.entries, state.claims, &state.reserved,    &state.head,
          kExperts,      kTiles,       bwd::TaskKind::Dw1, granularity};
}

// A later producer may publish before the reserved queue head. A consumer
// must return immediately without claiming tiles from the missing entry.
__global__ void publication_hole(State *state) {
  if (threadIdx.x != 0)
    return;
  auto q = queue(*state, 16);
  q.owner_count = 2;
  q.tiles_per_owner = 1;
  state->reserved = 2;
  state->entries[1] = 1;
  bwd::TileTask task;
  if (bwd::try_claim_domain_once(q, task) || state->head != 0)
    ++state->errors;
  state->entries[0] = 2;
  if (!bwd::try_claim_domain_once(q, task) || task.owner != 1 ||
      task.tile != 0 || task.tile_count != 1)
    ++state->errors;
  if (!bwd::try_claim_domain_once(q, task) || task.owner != 0 ||
      task.tile != 0 || task.tile_count != 1)
    ++state->errors;
  if (bwd::try_claim_domain_once(q, task))
    ++state->errors;
}

// Exercise simultaneous publication and consumption, inactive experts, mixed
// claim widths and non-divisible tails. A payload read also checks that the
// release/acquire edge precedes the claimed GEMM work.
__global__ void concurrent_claims(State *state, unsigned granularity,
                                  unsigned execution) {
  if (threadIdx.x != 0)
    return;
  auto q = queue(*state, blockIdx.x % 2 == 0 ? 1 : granularity);
  if (blockIdx.x < 8) {
    for (unsigned index = blockIdx.x; index < kExperts; index += 8) {
      const auto expert = kExperts - 1 - index;
      if (expert % 7 == 0)
        continue;
      state->payload[expert] = execution * 101 + expert;
      __nanosleep((expert % 5) * 64);
      bwd::publish_owner(q, expert);
    }
  }
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> completed(
      state->completed);
  const auto start = clock64();
  while (completed.load(::cuda::memory_order_acquire) <
         kActiveExperts * kTiles) {
    bwd::TileTask task;
    if (!bwd::try_claim_domain_once(q, task)) {
      if (clock64() - start > 10000000000ULL) {
        atomicAdd(&state->errors, 1U);
        return;
      }
      __nanosleep(64);
      continue;
    }
    if (task.owner >= kExperts || task.owner % 7 == 0 ||
        task.tile + task.tile_count > kTiles || task.tile_count == 0 ||
        task.kind != bwd::TaskKind::Dw1) {
      atomicAdd(&state->errors, 1U);
      return;
    }
    if (state->payload[task.owner] != execution * 101 + task.owner)
      atomicAdd(&state->errors, 1U);
    for (unsigned tile = task.tile; tile < task.tile + task.tile_count; ++tile)
      if (atomicAdd(&state->visits[task.owner * kTiles + tile], 1U) != 0)
        atomicAdd(&state->errors, 1U);
    completed.fetch_add(task.tile_count, ::cuda::memory_order_release);
  }
}

void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(2);
  }
}

int main() {
  State *state;
  check(cudaMallocManaged(&state, sizeof(State)));
  check(cudaMemset(state, 0, sizeof(State)));
  publication_hole<<<1, 32>>>(state);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  if (state->errors != 0)
    return 1;
  for (unsigned execution = 1; execution <= 100; ++execution) {
    const unsigned widths[] = {1, 4, 8, 16};
    check(cudaMemset(state, 0, sizeof(State)));
    concurrent_claims<<<64, 32>>>(state, widths[execution % 4], execution);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    for (unsigned expert = 0; expert < kExperts; ++expert)
      for (unsigned tile = 0; tile < kTiles; ++tile)
        if (state->visits[expert * kTiles + tile] != (expert % 7 != 0))
          ++state->errors;
    if (state->errors != 0 || state->reserved != kActiveExperts ||
        state->head != kActiveExperts) {
      std::fprintf(stderr, "execution=%u errors=%u reserved=%llu head=%llu\n",
                   execution, state->errors, state->reserved, state->head);
      return 1;
    }
  }
  check(cudaFree(state));
  std::puts("PASS backward expert queue: publication hole and 100 executions");
}
