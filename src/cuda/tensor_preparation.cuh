#pragma once

// Host-only allocation agreement. Keep it separate from per-tensor ownership
// and from the payload/slot protocols; inference kernels never consult it.
struct CollectiveTensorPreparation::Impl {
  std::shared_ptr<internal::NcclSessionState> nccl;
  CudaExecution execution;
  cudaStream_t stream;
  std::shared_ptr<tensor_storage_detail::Allocation> status;
  std::uint64_t *host = nullptr;

  static void check(cudaError_t error) {
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("tensor preparation: ") +
                               cudaGetErrorString(error));
  }

  Impl(const NcclSession &session, const CudaExecution &exec)
      : nccl(internal::NcclSessionAccess::checked(session)), execution(exec),
        stream(internal::checked_execution_state(exec)->stream) {
    if (nccl->device != exec.device_ordinal())
      throw std::invalid_argument("tensor preparation requires the NCCL device");
    check(cudaSetDevice(static_cast<int>(exec.device_ordinal())));
    status = tensor_storage_detail::allocate(2U * sizeof(std::uint64_t),
                                               exec.device_ordinal());
    check(cudaHostAlloc(reinterpret_cast<void **>(&host),
                         2U * sizeof(std::uint64_t), cudaHostAllocPortable));
  }

  ~Impl() noexcept {
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    (void)cudaStreamSynchronize(stream);
    (void)cudaFreeHost(host);
  }

  std::uint64_t agree(bool failed, bool changed) const {
    auto *device = static_cast<std::uint64_t *>(status->data);
    // Failure rank takes precedence. With no failure, all ranks have the
    // same high bits, so MIN also reports whether any rank changed storage.
    const std::uint64_t origin = failed ? nccl->rank : nccl->world;
    host[0] = (origin << 1U) | !changed;
    check(cudaMemcpyAsync(device, host, sizeof(*host), cudaMemcpyHostToDevice,
                           stream));
    const auto error = ncclAllReduce(device, device + 1U, 1U, ncclUint64,
                                     ncclMin, nccl->comm, stream);
    if (error != ncclSuccess)
      throw std::runtime_error(std::string("tensor preparation agreement: ") +
                               ncclGetErrorString(error));
    check(cudaMemcpyAsync(host + 1U, device + 1U, sizeof(*host),
                           cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host[1];
  }
};

CollectiveTensorPreparation::CollectiveTensorPreparation(
    const NcclSession &session, const CudaExecution &execution)
    : impl_(std::make_shared<Impl>(session, execution)) {
  // Warm the NCCL control path before a later tape/workspace OOM. This is not
  // a main-kernel barrier and does not reserve any activation payload.
  run([] {});
}

bool CollectiveTensorPreparation::run(
    const std::function<void()> &prepare, bool changed) const {
  auto &state = *impl_;
  Impl::check(cudaSetDevice(static_cast<int>(state.execution.device_ordinal())));
  cudaStreamCaptureStatus capture;
  Impl::check(cudaStreamIsCapturing(state.stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error("collective tensor preparation requires eager execution");
  std::exception_ptr failure;
  try {
    prepare();
  } catch (...) {
    failure = std::current_exception();
  }
  const auto agreement = state.agree(static_cast<bool>(failure), changed);
  const auto origin = agreement >> 1U;
  if (origin == state.nccl->world)
    return (agreement & 1U) == 0U;
  const auto error = TensorPreparationFailure(
      "tensor preparation failed on rank " + std::to_string(origin) +
      " before execution");
  if (failure) {
    try {
      std::rethrow_exception(failure);
    } catch (...) {
      std::throw_with_nested(error);
    }
  }
  throw error;
}
