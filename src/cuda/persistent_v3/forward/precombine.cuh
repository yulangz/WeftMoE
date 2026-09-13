#pragma once

namespace forward_precombine_detail {
using Group = persistent_v3::ForwardPrecombineGroup;

__device__ __forceinline__ void describe_local(const ForwardKernelArgs &a,
                                               unsigned long long route) {
  if (!a.pre_combine)
    return;
  const auto top_k = a.dispatch.identity.top_k;
  const auto begin = route / top_k * top_k;
  const auto destination = a.dispatch.route_dst_rank[route];
  const auto row = a.dispatch.route_dst_row[route];
  const auto route_at = [&](unsigned i) {
    return begin + a.dispatch.combine_order[begin + i];
  };
  peer_at<Group>(a.peer_bases[destination],
                 a.forward_precombine_groups_offset)[row] =
      persistent_v3::precombine_group(
          top_k, destination, row,
          [&](unsigned i) { return a.dispatch.route_dst_rank[route_at(i)]; },
          [&](unsigned i) { return a.dispatch.route_dst_row[route_at(i)]; });
}

__device__ __forceinline__ void
describe_remote(const ForwardKernelArgs &a, void *base,
                const ForwardDispatchFanoutRecord *fanout, unsigned count,
                unsigned index) {
  if (a.pre_combine)
    peer_at<Group>(base,
                   a.forward_precombine_groups_offset)[fanout[index].dst_row] =
        persistent_v3::precombine_group(
            count, fanout[index].destination_local_rank, fanout[index].dst_row,
            [&](unsigned i) { return fanout[i].destination_local_rank; },
            [&](unsigned i) { return fanout[i].dst_row; });
}

// One warp owns one token. Match destination ranks in canonical-order lanes,
// then retain the first lane of each group; no per-lane top-k search is needed.
__device__ __forceinline__ unsigned long long
leader_mask(const std::uint32_t *order, const std::uint32_t *destination,
            unsigned long long token, unsigned top_k) {
  const auto lane = threadIdx.x & 31U;
  const auto begin = token * top_k;
  const auto rank =
      lane < top_k ? destination[begin + order[begin + lane]] : UINT32_MAX;
  const auto members = __match_any_sync(0xffffffffU, rank);
  return __ballot_sync(0xffffffffU,
                       lane < top_k && unsigned(__ffs(members) - 1) == lane);
}

// Every participating math thread has finished its FC2 stores. The last row
// of a group acquires all earlier rows and publishes its sole Return entry.
// Which row wins changes scheduling only; the immutable links fix the sum
// order.
template <unsigned Threads>
__device__ __noinline__ void
complete_groups(const ForwardKernelArgs &a, unsigned long long first,
                unsigned rows, unsigned long long *leaders) {
  const auto *groups =
      peer_at<const Group>(a.local_base, a.forward_precombine_groups_offset);
  auto *arrived =
      peer_at<unsigned>(a.local_base, a.forward_precombine_arrived_offset);
  if (threadIdx.x < rows) {
    const auto group = groups[first + threadIdx.x];
    if (group.count == 1U) {
      leaders[threadIdx.x] = group.leader;
    } else {
      const auto count =
          ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
              arrived[group.leader])
              .fetch_add(1U, ::cuda::memory_order_acq_rel);
      leaders[threadIdx.x] = count + 1U == group.count ? group.leader : ~0ULL;
    }
  }
  cutlass::arch::NamedBarrier::sync(Threads, 2U);
  // Queue publication transfers the completion chain to the Return worker.
}

__device__ __forceinline__ bool needs_reduce(const ForwardKernelArgs &a,
                                             unsigned long long row) {
  return a.pre_combine &&
         peer_at<const Group>(a.local_base,
                              a.forward_precombine_groups_offset)[row]
                 .count > 1U;
}

// Return packs the partial sum directly into shared memory or the GIN ring.
// FC2 only publishes readiness; its compute CTA never waits for this arithmetic
// or overwrites a contribution still needed by another tensor consumer.
__device__ __forceinline__ uint4 sum_vector(const ForwardKernelArgs &a,
                                            unsigned long long leader,
                                            unsigned long long column) {
  const auto *groups =
      peer_at<const Group>(a.local_base, a.forward_precombine_groups_offset);
  const auto *output = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  float sum[8] = {};
  for (auto row = leader; row != ~0ULL; row = groups[row].next) {
    const auto packed =
        *reinterpret_cast<const uint4 *>(output + row * a.hidden + column);
    const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) {
      sum[2U * i] =
          __fadd_rn(sum[2U * i],
                    bf16_bits_to_float(static_cast<std::uint16_t>(words[i])));
      sum[2U * i + 1U] = __fadd_rn(
          sum[2U * i + 1U],
          bf16_bits_to_float(static_cast<std::uint16_t>(words[i] >> 16U)));
    }
  }
  unsigned words[4];
#pragma unroll
  for (unsigned i = 0; i < 4U; ++i)
    words[i] = unsigned(float_to_bf16_bits(sum[2U * i])) |
               (unsigned(float_to_bf16_bits(sum[2U * i + 1U])) << 16U);
  return make_uint4(words[0], words[1], words[2], words[3]);
}
} // namespace forward_precombine_detail
