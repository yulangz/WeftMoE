#pragma once

// Per-call status must outlive reuse of the shared execution workspace.
// Cache these small pinned records, and poll mapped memory rather than CUDA
// event queries while a persistent GIN kernel may be running.
struct TrainingCompletionRecord {
  TransportDeviceStatus status;
  std::uint32_t done;
};

struct TrainingCompletionPool
    : std::enable_shared_from_this<TrainingCompletionPool> {
  struct Entry {
    TrainingCompletionRecord *host = nullptr;
    CUdeviceptr done = 0;
    cudaEvent_t start = nullptr, compute = nullptr;
    cudaEvent_t computed = nullptr, complete = nullptr;
    ~Entry() {
      for (auto event : {start, compute, computed, complete})
        if (event)
          (void)cudaEventDestroy(event);
      if (host)
        (void)cudaFreeHost(host);
    }
    bool ready() const {
      return __atomic_load_n(&host->done, __ATOMIC_ACQUIRE) != 0U;
    }
    void wait() const {
      while (!ready())
        std::this_thread::yield();
    }
    void record(const TransportDeviceStatus *status, cudaStream_t stream) {
      const auto error =
          cudaMemcpyAsync(&host->status, status, sizeof(host->status),
                          cudaMemcpyDeviceToHost, stream);
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                sizeof(host->status));
    }
    void publish(cudaStream_t stream) {
      // Publish after status copies and timing events. The default write
      // includes a system-scope fence before the host observes completion.
      if (cuStreamWriteValue32(stream, done, 1U,
                               CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS)
        throw std::runtime_error("publish training completion");
    }
  };
  std::vector<std::unique_ptr<Entry>> entries;
  std::vector<Entry *> available;

  std::shared_ptr<Entry> acquire() {
    if (available.empty()) {
      auto entry = std::make_unique<Entry>();
      auto error = cudaHostAlloc(reinterpret_cast<void **>(&entry->host),
                                 sizeof(TrainingCompletionRecord),
                                 cudaHostAllocPortable | cudaHostAllocMapped);
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
      void *device = nullptr;
      error = cudaHostGetDevicePointer(&device, entry->host, 0);
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
      entry->done = reinterpret_cast<CUdeviceptr>(
          &static_cast<TrainingCompletionRecord *>(device)->done);
      for (auto *event : {&entry->start, &entry->compute, &entry->computed,
                          &entry->complete}) {
        error = cudaEventCreate(event);
        if (error != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(error));
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      }
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      available.push_back(entry.get());
      entries.push_back(std::move(entry));
    }
    auto *entry = available.back();
    available.pop_back();
    __atomic_store_n(&entry->host->done, 0U, __ATOMIC_RELEASE);
    return {entry, [pool = shared_from_this()](Entry *value) {
              pool->available.push_back(value);
            }};
  }
};
