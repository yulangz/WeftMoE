#pragma once

// Included inside backward_unified_transport. dX tiles stay in their bounded
// output slot until every row has either reached a source window or been
// copied into independently credited NIC TX storage.
template <bool UseGin, unsigned BaseThread, class Storage, class Layout>
__device__ __forceinline__ bool progress_ring_return(
    const BackwardKernelArgs &a, unsigned long long *words,
    const Layout &layout, Storage &s) {
  namespace returning = weftmoe_pf::backward_return;
  namespace counters = weftmoe_pf::ring;
  constexpr auto threads = Storage::kCopyThreads;
  const auto tid = threadIdx.x - BaseThread;
  const auto &r = a.return_ring;
  constexpr auto tiles = backward_tuning::kMainHidden /
      backward_tuning::kDxTileN / backward_tuning::kActivationClusterN;
  if (tid == 0U) {
    const auto block = s.static_return_next;
    s.return_cursor = ~0ULL;
    if (block < r.state->blocks && counters::acquire(r.blocks[block].ready) >= tiles) {
      s.return_cursor = block;
      s.queue_selected = r.blocks[block].ready == tiles;
      atomicMin(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->reduce_blocks_done, device_globaltimer_ns());
    }
  }
  forward_unified_transport::copy_sync<threads>();
  if (s.return_cursor == ~0ULL)
    return false;
  const auto block = unsigned(s.return_cursor);
  const auto valid = r.blocks[block].rows;
  const auto first = std::uint64_t(block) * backward_tuning::kTileM;
  const auto fresh = s.queue_selected;
  if constexpr (UseGin) {
    if (fresh) {
      auto *entries = return_queue_detail::entries(a);
      for (unsigned row = tid; row < valid; row += threads) {
        const auto identity = r.rows[first + row];
        if (identity.source / a.local_world_size == a.node)
          continue;
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
            entries[identity.queue_position])
            .store(unsigned(first + row), ::cuda::memory_order_release);
        const auto &q = return_queue_detail::queues(a)[1U + identity.source];
        return_queue_detail::publish_ready(a, q, identity.queue_position - q.begin, 1U);
      }
      forward_unified_transport::copy_sync<threads>();
    }
  }
  if (tid == 0U) {
    if (fresh)
      counters::publish(r.blocks[block].ready, tiles + 1U);
    s.return_uniform = returning::local_reserved(a, block);
  }
  forward_unified_transport::copy_sync<threads>();
  if (!s.return_uniform)
    return fresh;

  const auto *mapping =
      peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  for (unsigned row = tid; row < valid; row += threads) {
    const auto identity = r.rows[first + row];
    if (identity.source / a.local_world_size != a.node)
      continue;
    const auto &peer = a.peer_payloads[identity.source];
    peer.dgate[returning::contribution_row(a, identity.route)] =
        a.dgate_rows[mapping[identity.compact]];
  }
  __threadfence_system();
  forward_unified_transport::copy_sync<threads>();
  forward_unified_transport::publish_return<BaseThread>(a, s, valid, [&](unsigned item) {
    const auto identity = r.rows[first + item];
    return local_notification_detail::Entry{
        identity.source / a.local_world_size == a.node
            ? identity.source % a.local_world_size : communication::kLocalPeers,
        identity.route};
  });
  if (tid == 0U) {
    unsigned local = backward_tuning::kTileM - valid;
    for (unsigned row = 0U; row < valid; ++row)
      local += r.rows[first + row].source / a.local_world_size == a.node;
    returning::retire_rows(a, block, local);
    s.static_return_next += UseGin ? a.gin_receiver_count : gridDim.x;
    auto *counter = reinterpret_cast<persistent_v3::PaddedCounter *>(
        words + layout.returned_mblocks);
    const auto done =
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(counter->value)
            .fetch_add(1ULL, ::cuda::memory_order_acq_rel) + 1ULL;
    if (done == r.state->blocks)
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->return_complete_clock, device_globaltimer_ns());
  }
  forward_unified_transport::copy_sync<threads>();
  return true;
}
