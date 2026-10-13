#pragma once

// Host-side ownership of application tensors. No tensor in this module is
// exported through CUDA IPC or registered with NCCL GIN.
namespace tensor_storage_detail {
struct Allocation {
  void *data = nullptr;
  std::uint64_t bytes = 0;
  std::uint32_t device = 0;
  bool native = false;
  std::shared_ptr<const void> owner;
  ~Allocation() noexcept {
    if (native && data) {
      (void)cudaSetDevice(static_cast<int>(device));
      (void)cudaFree(data);
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
    }
  }
};

struct Completion {
  cudaEvent_t ready = nullptr;
  cudaEvent_t dependency = nullptr;
  cudaStream_t stream = nullptr;
  unsigned long long capture = 0;
  std::uint32_t device = 0;
  explicit Completion(const CudaExecution &execution) {
    auto exec = internal::checked_execution_state(execution);
    device = execution.device_ordinal();
    stream = exec->stream;
    cudaStreamCaptureStatus status;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
        status != cudaStreamCaptureStatusNone)
      throw std::runtime_error(
          "prepare tensor storage before CUDA Graph capture");
    (void)cudaSetDevice(static_cast<int>(device));
    if (cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&dependency, cudaEventDisableTiming) !=
            cudaSuccess) {
      if (ready)
        (void)cudaEventDestroy(ready);
      if (dependency)
        (void)cudaEventDestroy(dependency);
      throw std::runtime_error("create tensor completion events");
    }
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  }
  ~Completion() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (ready) {
      (void)cudaEventDestroy(ready);
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
    }
    if (dependency) {
      (void)cudaEventDestroy(dependency);
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
    }
  }
  void record(cudaStream_t producer) {
    stream = producer;
    auto error = cudaEventRecord(dependency, stream);
    if (error == cudaSuccess)
      error = internal::record_observable_event(ready, stream, &capture);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("record tensor completion: ") +
                               cudaGetErrorString(error));
  }
  void wait_on(cudaStream_t consumer) const {
    const auto error = internal::wait_output_ready(consumer, stream, ready,
                                                   capture, dependency);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("wait tensor completion: ") +
                               cudaGetErrorString(error));
  }
};

std::shared_ptr<Allocation> allocate(std::uint64_t bytes, std::uint32_t device,
                                     const CudaTensorBuffer &binding = {}) {
  auto result = std::make_shared<Allocation>();
  result->bytes = bytes;
  result->device = device;
  if (binding.data || binding.bytes || binding.owner) {
    if (!binding.owner)
      throw std::runtime_error(
          "external tensor storage requires an allocation owner");
    validate_external_cuda_range(binding.data, bytes, binding.bytes, 1, device,
                                 "tensor storage");
    if (bytes && reinterpret_cast<std::uintptr_t>(binding.data) % 128U)
      throw std::runtime_error("tensor storage must be aligned to 128 bytes");
    result->data = binding.data;
    result->owner = binding.owner;
  } else {
    result->native = true;
    const auto error =
        cudaMalloc(&result->data, std::max<std::uint64_t>(bytes, 128));
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("allocate tensor storage: ") +
                               cudaGetErrorString(error));
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  }
  return result;
}

void validate_disjoint(
    std::initializer_list<std::shared_ptr<Allocation>> values) {
  for (auto a = values.begin(); a != values.end(); ++a) {
    if (!*a || !(*a)->bytes)
      continue;
    for (auto b = a + 1; b != values.end(); ++b) {
      if (!*b || !(*b)->bytes)
        continue;
      const auto x = reinterpret_cast<std::uintptr_t>((*a)->data);
      const auto y = reinterpret_cast<std::uintptr_t>((*b)->data);
      if (x <= y ? y - x < (*a)->bytes : x - y < (*b)->bytes)
        throw std::runtime_error("output and saved tensors must not alias");
    }
  }
}
void validate_input_disjoint(
    std::initializer_list<std::shared_ptr<Allocation>> outputs,
    const void *input, std::uint64_t bytes) {
  if (!bytes)
    return;
  const auto y = reinterpret_cast<std::uintptr_t>(input);
  for (const auto &output : outputs) {
    if (!output || !output->bytes)
      continue;
    const auto x = reinterpret_cast<std::uintptr_t>(output->data);
    if (x <= y ? y - x < output->bytes : x - y < bytes)
      throw std::runtime_error(
          "output and saved tensors must not alias a live input");
  }
}

CudaTensorBuffer binding(const std::shared_ptr<Allocation> &value) {
  return value ? CudaTensorBuffer{value->data, value->bytes, value}
               : CudaTensorBuffer{};
}
} // namespace tensor_storage_detail

struct ForwardTensorStorage::Impl {
  std::shared_ptr<tensor_storage_detail::Allocation> output, x, ab, gates,
      headers;
  tensor_storage_detail::Completion completion;
  std::uint64_t tokens, rows, hidden, intermediate;
  bool training;
  explicit Impl(const CudaExecution &execution) : completion(execution) {}
};

struct BackwardTensorStorage::Impl {
  std::shared_ptr<tensor_storage_detail::Allocation> dx, dgate, dw1, dw2,
      workspace;
  tensor_storage_detail::Completion completion;
  std::uint64_t tokens, hidden, intermediate, rows, padded_rows;
  std::size_t u_offset, dab_offset, dgate_rows_offset;
  unsigned pf_depth = 0;
  std::size_t pf_dab_offset = 0;
  CUtensorMap dab_map{}, u_mn_map{}, dab_mn_map{};
  std::uint32_t top_k, experts;
  explicit Impl(const CudaExecution &execution) : completion(execution) {}
  static std::shared_ptr<Impl>
  make(const DeviceDispatchHandle &, const CudaExecution &,
       const BackwardMegaKernelConfig &, const BackwardTensorBuffers &,
       const CudaTensorAllocator &, unsigned pf_depth);
};

ForwardTensorStorage prepare_forward_tensors(
    const DeviceDispatchHandle &dispatch, const CudaExecution &execution,
    const ForwardMegaKernelConfig &config, const ForwardTensorBuffers &bindings,
    const CudaTensorAllocator &allocator) {
  using namespace tensor_storage_detail;
  if (!dispatch || !execution || !config.hidden_size ||
      !config.intermediate_size)
    throw std::runtime_error(
        "prepare Forward tensors requires Notify and a model shape");
  ForwardTensorStorage result;
  auto state = std::make_shared<ForwardTensorStorage::Impl>(execution);
  state->tokens = dispatch.identity().num_tokens;
  state->rows = dispatch.received_row_capacity();
  state->hidden = config.hidden_size;
  state->intermediate = config.intermediate_size;
  state->training = config.retain_backward_state;
  const auto device = execution.device_ordinal();
  const auto make = [&](std::uint64_t bytes,
                        const CudaTensorBuffer &binding = CudaTensorBuffer{}) {
    if (allocator && !binding.data && !binding.bytes && !binding.owner) {
      auto allocated = allocator(bytes);
      if (!allocated.owner)
        throw std::runtime_error("tensor allocator returned no owner");
      return allocate(bytes, device, allocated);
    }
    return allocate(bytes, device, binding);
  };
  state->output = make(
      checked_mul(checked_mul(state->tokens, state->hidden, "output elements"),
                  2ULL, "output bytes"),
      bindings.output);
  if (state->training) {
    state->x = make(
        checked_mul(checked_mul(state->rows, state->hidden, "saved X elements"),
                    2ULL, "saved X bytes"),
        bindings.fc1_input);
    state->ab = make(checked_mul(checked_mul(state->rows, state->intermediate,
                                             "saved AB elements"),
                                 4ULL, "saved AB bytes"),
                     bindings.fc1_output);
    state->gates = make(checked_mul(state->rows, sizeof(float), "saved gates"));
    state->headers = make(
        checked_mul(state->rows, sizeof(LocalP2pRouteHeader), "saved routing"));
  } else if (bindings.fc1_input.owner || bindings.fc1_input.data ||
             bindings.fc1_output.owner || bindings.fc1_output.data) {
    throw std::runtime_error("FC1 save buffers require retain_backward_state");
  }
  validate_disjoint(
      {state->output, state->x, state->ab, state->gates, state->headers});
  result.impl_ = std::move(state);
  return result;
}

std::shared_ptr<BackwardTensorStorage::Impl> BackwardTensorStorage::Impl::make(
    const DeviceDispatchHandle &dispatch, const CudaExecution &execution,
    const BackwardMegaKernelConfig &config,
    const BackwardTensorBuffers &bindings, const CudaTensorAllocator &allocator,
    unsigned pf_depth) {
  using namespace tensor_storage_detail;
  if (!dispatch || !execution || !config.hidden_size ||
      !config.intermediate_size)
    throw std::runtime_error(
        "prepare Backward tensors requires Notify and a model shape");
  auto state = std::make_shared<BackwardTensorStorage::Impl>(execution);
  state->pf_depth = pf_depth;
  state->tokens = dispatch.identity().num_tokens;
  state->top_k = dispatch.identity().top_k;
  state->experts = dispatch.sizes().owned_expert_count;
  state->hidden = config.hidden_size;
  state->intermediate = config.intermediate_size;
  state->rows = dispatch.received_row_capacity();
  // Zero-row ranks still need valid TMA descriptors, but no compute task can
  // touch the single placeholder row. Every nonempty rank allocates exact P.
  static_assert(backward_tuning::kTileM == pipeline::kMBlockRows);
  state->padded_rows = std::max<std::uint64_t>(
      1U, ActivationTransportBridge::padded_received_rows(dispatch));
  const auto device = execution.device_ordinal();
  const auto make = [&](std::uint64_t bytes,
                        const CudaTensorBuffer &binding = CudaTensorBuffer{}) {
    if (allocator && !binding.data && !binding.bytes && !binding.owner) {
      auto allocated = allocator(bytes);
      if (!allocated.owner)
        throw std::runtime_error("tensor allocator returned no owner");
      return allocate(bytes, device, allocated);
    }
    return allocate(bytes, device, binding);
  };
  const auto weight = checked_mul(
      checked_mul(state->experts, state->hidden, "gradient weight rows"),
      state->intermediate, "gradient weight elements");
  state->dx =
      make(checked_mul(checked_mul(state->tokens, state->hidden, "dX elements"),
                       2ULL, "dX bytes"),
           bindings.dx);
  state->dgate = make(
      checked_mul(checked_mul(state->tokens, state->top_k, "dGate elements"),
                  sizeof(float), "dGate bytes"),
      bindings.dgate);
  state->dw1 = make(checked_mul(weight, 4ULL, "dW1 bytes"), bindings.dw1);
  state->dw2 = make(checked_mul(weight, 2ULL, "dW2 bytes"), bindings.dw2);
  // Notify already carries exact R/P. U/dAB remain private; communication dR
  // belongs to the execution slot and is shared by its successive calls.
  std::size_t cursor = 0;
  const auto intermediate_elements = checked_mul(
      state->padded_rows, state->intermediate, "Backward workspace elements");
  state->u_offset = append(cursor, intermediate_elements, 2U, "Backward U");
  state->dab_offset = append(cursor, intermediate_elements, 4U, "Backward dAB");
  state->dgate_rows_offset =
      append(cursor, state->padded_rows, sizeof(float), "Backward dGate rows");
  if (pf_depth && 2U * state->intermediate > state->hidden)
    state->pf_dab_offset = append(
        cursor, std::uint64_t(pf_depth) * backward_tuning::kTileM *
                    state->intermediate,
        4U, "PF Backward dAB ring");
  state->workspace = make(cursor);
#if !defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
  auto *base = state->workspace->data;
  if (pf_depth && 2U * state->intermediate > state->hidden)
    state->dab_map = make_bf16_tma_map(
        at<void>(base, state->pf_dab_offset), 2ULL * state->intermediate,
        std::uint64_t(pf_depth) * backward_tuning::kTileM,
        backward_tuning::kTileM, "PF Backward dAB ring", 64U);
  state->u_mn_map =
      make_bf16_tma_map_mn(at<void>(base, state->u_offset), state->intermediate,
                           state->padded_rows, "Backward dW2 U MN-major");
  state->dab_mn_map = make_bf16_tma_map_mn(
      at<void>(base, state->dab_offset), 2ULL * state->intermediate,
      state->padded_rows, "Backward dW1 dAB MN-major");
#endif
  validate_disjoint(
      {state->dx, state->dgate, state->dw1, state->dw2, state->workspace});
  return state;
}

std::uint64_t ForwardTensorStorage::row_capacity() const {
  if (!impl_)
    throw std::runtime_error("empty Forward tensor storage");
  return impl_->rows;
}
std::uint64_t ForwardTensorStorage::bytes() const {
  if (!impl_)
    return 0;
  return impl_->output->bytes +
         (impl_->training ? impl_->x->bytes + impl_->ab->bytes +
                                impl_->gates->bytes + impl_->headers->bytes
                          : 0);
}
std::uint64_t BackwardTensorStorage::bytes() const {
  if (!impl_)
    return 0;
  return impl_->dx->bytes + impl_->dgate->bytes + impl_->dw1->bytes +
         impl_->dw2->bytes + impl_->workspace->bytes;
}

ForwardTensorBuffers ForwardTensorStorage::buffers() const {
  if (!impl_)
    return {};
  using tensor_storage_detail::binding;
  return {binding(impl_->output), binding(impl_->x), binding(impl_->ab)};
}
BackwardTensorBuffers BackwardTensorStorage::buffers() const {
  if (!impl_)
    return {};
  using tensor_storage_detail::binding;
  return {binding(impl_->dx), binding(impl_->dgate), binding(impl_->dw1),
          binding(impl_->dw2)};
}
