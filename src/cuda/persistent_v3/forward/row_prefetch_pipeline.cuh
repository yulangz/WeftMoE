#pragma once

// Included inside forward_unified_transport. The caller keeps
// this object across its complete row loop. Source loads for every slot are
// issued before any of those loads is waited on; each slot has a separate
// barrier and retains its own destination, byte count, and arrival ticket.
template <class Storage, bool Enabled = true> class RowPrefetchPipeline;

template <class Storage> class RowPrefetchPipeline<Storage, true> {
  static constexpr unsigned depth = Storage::kBuffers;
  static_assert(depth == 1U || depth == 2U);

  Storage &storage_;
  unsigned warp_;
  unsigned queued_ = 0U;
  void *targets_[depth];
  unsigned extents_[depth];
  CopyBarrier::arrival_token tickets_[depth];

  template <unsigned Slot>
  __device__ __forceinline__ void enqueue(const void *source, void *target,
                                         unsigned bytes) {
    static_assert(Slot < depth);
    // Each previous full pair committed one store group per slot, in slot
    // order. Slot0 can be overwritten after all but the newest group release
    // their shared reads; slot1 must wait for that last group too. The next
    // pair's slot0 G2S may therefore overlap the previous slot1 S2G. During
    // the initial fill there are no old groups, so these waits return at once.
    ::cuda::device::experimental::cp_async_bulk_wait_group_read<depth - 1U - Slot>();
    auto *buffer = storage_.staging +
                   (warp_ * depth + Slot) * Storage::kSegmentBytes;
    auto &full = storage_.row_prefetch_full[warp_ * depth + Slot];
    ::cuda::device::experimental::cp_async_bulk_global_to_shared(
        buffer, source, bytes, full);
    tickets_[Slot] = ::cuda::device::barrier_arrive_tx(full, 1U, bytes);
    targets_[Slot] = target;
    extents_[Slot] = bytes;
  }

  template <unsigned Slot>
  __device__ __forceinline__ void retire() {
    static_assert(Slot < depth);
    auto &full = storage_.row_prefetch_full[warp_ * depth + Slot];
    full.wait(static_cast<CopyBarrier::arrival_token &&>(tickets_[Slot]));
    const auto *buffer = storage_.staging +
                         (warp_ * depth + Slot) * Storage::kSegmentBytes;
    ::cuda::device::experimental::cp_async_bulk_shared_to_global(
        targets_[Slot], buffer, extents_[Slot]);
    ::cuda::device::experimental::cp_async_bulk_commit_group();
  }

public:
  __device__ __forceinline__ RowPrefetchPipeline(Storage &storage, unsigned warp)
      : storage_(storage), warp_(warp) {}

  __device__ __forceinline__ void copy(const void *source, void *target,
                                       unsigned bytes) {
    for (unsigned offset = 0U; offset < bytes;
         offset += Storage::kSegmentBytes) {
      const unsigned extent = min(Storage::kSegmentBytes, bytes - offset);
      const auto *source_segment = static_cast<const char *>(source) + offset;
      auto *target_segment = static_cast<char *>(target) + offset;
      // Constant slot indices let the compiler keep the two descriptors in
      // registers instead of dynamically indexing a local-memory array.
      if constexpr (depth == 1U) {
        enqueue<0U>(source_segment, target_segment, extent);
        retire<0U>();
      } else {
        if (queued_ == 0U) {
          enqueue<0U>(source_segment, target_segment, extent);
          queued_ = 1U;
        } else {
          enqueue<1U>(source_segment, target_segment, extent);
          // Both loads are now in flight, including when slot0 and slot1
          // came from different rows, peers, or final partial segments.
          retire<0U>();
          retire<1U>();
          queued_ = 0U;
        }
      }
    }
  }

  __device__ __forceinline__ void drain() {
    if constexpr (depth == 2U) {
      if (queued_ != 0U) {
        retire<0U>();
        queued_ = 0U;
      }
    }
    // Drain only retires pending loads into stores. The caller must preserve
    // its subsequent finish_copies(): global wait0 + system fence is the
    // separate payload-before-notification ordering, including an odd tail.
  }
};

// Preserve the original segmented copier when prefetch is disabled. Both
// variants share the row-loop interface and the caller's final global wait.
template <class Storage> class RowPrefetchPipeline<Storage, false> {
  Storage &storage_;
  unsigned warp_;

public:
  __device__ __forceinline__ RowPrefetchPipeline(Storage &storage, unsigned warp)
      : storage_(storage), warp_(warp) {}

  __device__ __forceinline__ void copy(const void *source, void *target,
                                       unsigned bytes) {
    copy_row(storage_, warp_, source, target, bytes);
  }
  __device__ __forceinline__ void drain() {}
};

// One received row can feed several local peers. Keep its immutable metadata
// descriptor until the prefetched payload is ready, then issue every fanout
// from the same staging slot. A group contains all stores from that slot, so
// the same read-completion rule as the single-destination pipeline protects
// reuse without serializing each row behind all of its peer writes.
template <class Storage, class Descriptor, class Destination>
class FanoutRowPrefetchPipeline {
  static constexpr unsigned depth = Storage::kBuffers;
  static_assert(depth == 1U || depth == 2U);

  Storage &storage_;
  Destination destination_;
  unsigned warp_;
  unsigned queued_ = 0U;
  Descriptor descriptors_[depth];
  unsigned offsets_[depth];
  unsigned extents_[depth];
  unsigned fanouts_[depth];
  CopyBarrier::arrival_token tickets_[depth];

  template <unsigned Slot>
  __device__ __forceinline__ void enqueue(const void *source,
                                         Descriptor descriptor,
                                         unsigned offset, unsigned bytes,
                                         unsigned fanouts) {
    static_assert(Slot < depth);
    ::cuda::device::experimental::cp_async_bulk_wait_group_read<
        depth - 1U - Slot>();
    auto *buffer = storage_.staging +
                   (warp_ * depth + Slot) * Storage::kSegmentBytes;
    auto &full = storage_.row_prefetch_full[warp_ * depth + Slot];
    ::cuda::device::experimental::cp_async_bulk_global_to_shared(
        buffer, source, bytes, full);
    tickets_[Slot] = ::cuda::device::barrier_arrive_tx(full, 1U, bytes);
    descriptors_[Slot] = descriptor;
    offsets_[Slot] = offset;
    extents_[Slot] = bytes;
    fanouts_[Slot] = fanouts;
  }

  template <unsigned Slot>
  __device__ __forceinline__ void retire() {
    static_assert(Slot < depth);
    auto &full = storage_.row_prefetch_full[warp_ * depth + Slot];
    full.wait(static_cast<CopyBarrier::arrival_token &&>(tickets_[Slot]));
    const auto *buffer = storage_.staging +
                         (warp_ * depth + Slot) * Storage::kSegmentBytes;
    for (unsigned fanout = 0U; fanout < fanouts_[Slot]; ++fanout) {
      auto *target = static_cast<char *>(
          destination_(descriptors_[Slot], fanout));
      ::cuda::device::experimental::cp_async_bulk_shared_to_global(
          target + offsets_[Slot], buffer, extents_[Slot]);
    }
    ::cuda::device::experimental::cp_async_bulk_commit_group();
  }

public:
  __device__ __forceinline__ FanoutRowPrefetchPipeline(
      Storage &storage, unsigned warp, Destination destination)
      : storage_(storage), destination_(destination), warp_(warp) {}

  __device__ __forceinline__ void copy(const void *source,
                                       Descriptor descriptor, unsigned bytes,
                                       unsigned fanouts) {
    if (fanouts == 0U)
      return;
    for (unsigned offset = 0U; offset < bytes;
         offset += Storage::kSegmentBytes) {
      const unsigned extent = min(Storage::kSegmentBytes, bytes - offset);
      const auto *segment = static_cast<const char *>(source) + offset;
      if constexpr (depth == 1U) {
        enqueue<0U>(segment, descriptor, offset, extent, fanouts);
        retire<0U>();
      } else {
        if (queued_ == 0U) {
          enqueue<0U>(segment, descriptor, offset, extent, fanouts);
          queued_ = 1U;
        } else {
          enqueue<1U>(segment, descriptor, offset, extent, fanouts);
          retire<0U>();
          retire<1U>();
          queued_ = 0U;
        }
      }
    }
  }

  __device__ __forceinline__ void drain() {
    if constexpr (depth == 2U) {
      if (queued_ != 0U) {
        retire<0U>();
        queued_ = 0U;
      }
    }
    // The caller must still finish_copies() before either publishing any
    // ready notification or releasing the source ring slot's credit. The
    // descriptor and every fanout record must remain valid until then.
  }
};
