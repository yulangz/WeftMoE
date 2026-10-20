#pragma once

// Host-only allocation agreement. Keep it separate from per-tensor ownership
// and from the payload/slot protocols; inference kernels never consult it.
struct CollectiveTensorPreparation::Impl {
  std::shared_ptr<internal::NcclSessionState> nccl;
  CudaExecution execution;
  cudaStream_t stream = nullptr;
  std::shared_ptr<tensor_storage_detail::Allocation> status;
  std::uint32_t *host = nullptr;

  static void check(cudaError_t error) {
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("tensor preparation: ") +
                               cudaGetErrorString(error));
  }

  Impl(const NcclSession &session, const CudaExecution &exec)
      : nccl(internal::NcclSessionAccess::checked(session)), execution(exec) {
    if (nccl->device != exec.device_ordinal())
      throw std::invalid_argument("tensor preparation requires the NCCL device");
    check(cudaSetDevice(static_cast<int>(exec.device_ordinal())));
    status = tensor_storage_detail::allocate(4U * sizeof(std::uint32_t),
                                               exec.device_ordinal());
    check(cudaHostAlloc(reinterpret_cast<void **>(&host),
                         4U * sizeof(std::uint32_t), cudaHostAllocPortable));
    // This agreement reads host allocation status, not tensor contents.
    // Keep it independent of the application's attention/router/DDP stream.
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  ~Impl() noexcept {
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
    (void)cudaFreeHost(host);
  }

  std::uint32_t agree(bool failed, bool changed) const {
    auto *device = static_cast<std::uint32_t *>(status->data);
    host[0] = failed ? nccl->rank : nccl->world;
    host[1] = changed ? 0U : 1U;
    check(cudaMemcpyAsync(device, host, 2U * sizeof(*host), cudaMemcpyHostToDevice,
                           stream));
    const auto error = ncclAllReduce(device, device + 2U, 2U, ncclUint32,
                                     ncclMin, nccl->comm, stream);
    if (error != ncclSuccess)
      throw std::runtime_error(std::string("tensor preparation agreement: ") +
                               ncclGetErrorString(error));
    check(cudaMemcpyAsync(host + 2U, device + 2U, 2U * sizeof(*host),
                           cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host[2];
  }
};

CollectiveTensorPreparation::CollectiveTensorPreparation(
    const NcclSession &session, const CudaExecution &execution)
    : impl_(std::make_shared<Impl>(session, execution)) {
  // Warm the NCCL control path before a later tape/workspace OOM. This is not
  // a main-kernel barrier and does not reserve any activation payload.
  run([] {});
}

void CollectiveTensorPreparation::run(
    const std::function<void()> &prepare) const {
  auto &state = *impl_;
  Impl::check(cudaSetDevice(static_cast<int>(state.execution.device_ordinal())));
  cudaStreamCaptureStatus capture;
  Impl::check(cudaStreamIsCapturing(
      internal::checked_execution_state(state.execution)->stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error("collective tensor preparation requires eager execution");
  std::exception_ptr failure;
  try {
    prepare();
  } catch (...) {
    failure = std::current_exception();
  }
  const auto origin = state.agree(static_cast<bool>(failure), false);
  if (origin == state.nccl->world)
    return;
  // A failed allocation may follow an already-enqueued Notify. Drain it on
  // every rank before releasing the lease or retrying the idle plan.
  Impl::check(cudaStreamSynchronize(
      internal::checked_execution_state(state.execution)->stream));
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

bool CollectiveTensorPreparation::run_with_change(
    const std::function<bool()> &prepare) const {
  auto &state = *impl_;
  // Only changing an exported pointer requires every peer to remap. Do not
  // turn a per-call allocation exception into a recoverable collective error.
  state.host[0] = prepare() ? 1U : 0U;
  auto *device = static_cast<std::uint32_t *>(state.status->data);
  Impl::check(cudaMemcpyAsync(device, state.host, sizeof(*device),
                             cudaMemcpyHostToDevice, state.stream));
  const auto error = ncclAllReduce(device, device + 1U, 1U, ncclUint32,
                                   ncclMax, state.nccl->comm, state.stream);
  if (error != ncclSuccess)
    throw std::runtime_error(std::string("peer binding agreement: ") +
                             ncclGetErrorString(error));
  Impl::check(cudaMemcpyAsync(state.host + 1U, device + 1U, sizeof(*device),
                             cudaMemcpyDeviceToHost, state.stream));
  Impl::check(cudaStreamSynchronize(state.stream));
  return state.host[1] != 0U;
}
