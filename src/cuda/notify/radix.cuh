// Stable LSD radix sort inside a cooperatively launched resident grid.
// The caller supplies pooled scratch. Chunk order is source order; no global
// atomic ticket decides a key's position, including equal keys.
struct NotifySortStorage {
  using Sort = cub::BlockRadixSort<std::uint64_t, 256, 4, std::uint64_t>;
  union {
    typename Sort::TempStorage sort;
    std::uint64_t digits[1024];
    typename cub::BlockScan<unsigned long long, 256>::TempStorage scan;
  } temp;
};

__device__ void notify_radix_sort(std::uint64_t *keys, std::uint64_t *values,
                                  std::uint64_t *scratch_keys,
                                  std::uint64_t *scratch_values,
                                  std::uint64_t count, std::uint64_t *histogram,
                                  NotifySortStorage &shared,
                                  unsigned end_bit = 64) {
  auto grid = cooperative_groups::this_grid();
  const auto chunks = (count + 1023) / 1024;
  const auto tid = blockIdx.x * 256 + threadIdx.x;
  const auto stride = gridDim.x * 256;
  // A single tile is already a complete stable sort; no histogram exchange
  // or grid-wide radix rounds are needed for this algorithmic base case.
  if (count <= 1024) {
    if (blockIdx.x == 0) {
      std::uint64_t k[4], v[4];
#pragma unroll
      for (unsigned item = 0; item < 4; ++item) {
        const auto pos = threadIdx.x * 4 + item;
        k[item] = pos < count ? keys[pos] : ~std::uint64_t(0);
        v[item] = pos < count ? values[pos] : 0;
      }
      NotifySortStorage::Sort(shared.temp.sort).Sort(k, v);
#pragma unroll
      for (unsigned item = 0; item < 4; ++item) {
        const auto pos = threadIdx.x * 4 + item;
        if (pos < count) {
          keys[pos] = k[item];
          values[pos] = v[item];
        }
      }
    }
    grid.sync();
    return;
  }
  // Runtime key extent avoids radix passes over invariant high zero bits.
  if (blockIdx.x == 0 && threadIdx.x == 0)
    histogram[0] = 0;
  grid.sync();
  unsigned long long maximum = 0;
  for (auto i = std::uint64_t(tid); i < count; i += stride)
    maximum = max(maximum, static_cast<unsigned long long>(keys[i]));
  for (unsigned delta = 16; delta; delta >>= 1)
    maximum = max(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  if (threadIdx.x % 32 == 0)
    atomicMax(reinterpret_cast<unsigned long long *>(histogram), maximum);
  grid.sync();
  end_bit = max(1U, 64U - unsigned(__clzll(histogram[0])));
  // Every CTA must finish reading the extent before histogram initialization.
  grid.sync();
  for (unsigned bit = 0; bit < end_bit; bit += 8) {
    const unsigned width = min(8U, end_bit - bit);
    const unsigned bins = 1U << width;
    for (auto i = std::uint64_t(tid); i < chunks * bins; i += stride)
      histogram[i] = 0;
    grid.sync();
    for (auto chunk = std::uint64_t(blockIdx.x); chunk < chunks;
         chunk += gridDim.x) {
      std::uint64_t k[4], v[4];
#pragma unroll
      for (unsigned item = 0; item < 4; ++item) {
        const auto pos = chunk * 1024 + threadIdx.x * 4 + item;
        k[item] = pos < count ? keys[pos] : ~std::uint64_t(0);
        v[item] = pos < count ? values[pos] : 0;
      }
      NotifySortStorage::Sort(shared.temp.sort).Sort(k, v, bit, bit + width);
      __syncthreads();
#pragma unroll
      for (unsigned item = 0; item < 4; ++item)
        shared.temp.digits[threadIdx.x * 4 + item] =
            (k[item] >> bit) & (bins - 1);
      __syncthreads();
#pragma unroll
      for (unsigned item = 0; item < 4; ++item) {
        const auto pos = chunk * 1024 + threadIdx.x * 4 + item;
        if (pos < count) {
          scratch_keys[pos] = k[item];
          scratch_values[pos] = v[item];
          const auto within = threadIdx.x * 4 + item;
          const unsigned digit = unsigned((k[item] >> bit) & (bins - 1));
          if (within == 0 || shared.temp.digits[within - 1] != digit) {
            unsigned end = within + 1;
            while (end < 1024 && chunk * 1024 + end < count &&
                   shared.temp.digits[end] == digit)
              ++end;
            histogram[std::uint64_t(digit) * chunks + chunk] = end - within;
          }
        }
      }
      __syncthreads();
    }
    grid.sync();
    // Scan each bucket with contiguous accesses across its chunks. A
    // single thread per bucket serializes long batches and issues strided
    // transactions; distributing buckets across CTAs removes both costs.
    auto *bucket_base = histogram + chunks * 256;
    for (unsigned bucket = blockIdx.x; bucket < bins; bucket += gridDim.x) {
      unsigned long long carry = 0;
      for (std::uint64_t first = 0; first < chunks; first += 256) {
        const auto chunk = first + threadIdx.x;
        const auto index = std::uint64_t(bucket) * chunks + chunk;
        unsigned long long prefix, total;
        cub::BlockScan<unsigned long long, 256>(shared.temp.scan)
            .ExclusiveSum(chunk < chunks ? histogram[index] : 0ULL, prefix,
                          total);
        if (chunk < chunks)
          histogram[index] = carry + prefix;
        carry += total;
        __syncthreads();
      }
      if (!threadIdx.x)
        bucket_base[bucket] = carry;
    }
    grid.sync();
    if (blockIdx.x == 0) {
      unsigned long long prefix;
      cub::BlockScan<unsigned long long, 256>(shared.temp.scan)
          .ExclusiveSum(threadIdx.x < bins ? bucket_base[threadIdx.x] : 0ULL,
                        prefix);
      if (threadIdx.x < bins)
        bucket_base[threadIdx.x] = prefix;
    }
    grid.sync();
    for (auto chunk = std::uint64_t(blockIdx.x); chunk < chunks;
         chunk += gridDim.x) {
#pragma unroll
      for (unsigned item = 0; item < 4; ++item) {
        const auto pos = chunk * 1024 + threadIdx.x * 4 + item;
        shared.temp.digits[threadIdx.x * 4 + item] =
            pos < count ? ((scratch_keys[pos] >> bit) & (bins - 1)) : bins;
      }
      __syncthreads();
#pragma unroll
      for (unsigned item = 0; item < 4; ++item) {
        const auto within = threadIdx.x * 4 + item;
        const auto pos = chunk * 1024 + within;
        if (pos < count) {
          const auto digit = shared.temp.digits[within];
          unsigned lo = 0, hi = within;
          while (lo < hi) {
            const auto mid = (lo + hi) / 2;
            if (shared.temp.digits[mid] < digit)
              lo = mid + 1;
            else
              hi = mid;
          }
          const auto dst = bucket_base[digit] +
                           histogram[digit * chunks + chunk] + within - lo;
          keys[dst] = scratch_keys[pos];
          values[dst] = scratch_values[pos];
        }
      }
      __syncthreads();
    }
    grid.sync();
  }
}
