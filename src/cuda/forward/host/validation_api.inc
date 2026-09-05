
const ActivationTransportExpectation &DeviceForwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->identity;
}

const ForwardMegaKernelAudit &DeviceForwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->audit;
}

const ForwardMegaKernelTimings &DeviceForwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->timings;
}

std::vector<transport::BFloat16> download_forward_output_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "forward download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "forward download elements"));
  if (!output.empty()) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16),
                    "forward download bytes"),
        "forward download bytes");
    internal::throw_cuda(
        "forward-download",
        cudaMemcpy(output.data(),
                   at<const void>(impl.pool->slots[impl.slot->index].device,
                                  impl.pool->layout.combined),
                   bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(forward output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}

ForwardBackwardStateValidationDownload
download_forward_backward_state_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "forward-backward-state-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (!impl.retained_backward_state ||
      impl.pool->layout.forward_ab_cache == 0 ||
      impl.pool->layout.forward_training_output == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-backward-state-download", "retained state",
                    impl.identity.rank, "enabled", "disabled");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-backward-state-download", "device",
                    impl.identity.rank, "handle device", "foreign");

  ForwardBackwardStateValidationDownload result;
  const auto x_elements = checked_mul(
      impl.incoming_rows, impl.identity.hidden,
      "forward retained canonical X elements");
  const auto plane_elements = checked_mul(
      impl.incoming_rows, impl.intermediate,
      "forward retained A/B plane elements");
  result.canonical_x.resize(
      checked_size(x_elements, "forward retained canonical X elements"));
  result.fc1_ab.resize(checked_size(
      checked_mul(plane_elements, 2ULL, "forward retained A/B elements"),
      "forward retained A/B elements"));
  if (impl.incoming_rows == 0)
    return result;

  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto copy = [&](void *destination, const void *source,
                        std::uint64_t elements, const char *name) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16), name), name);
    internal::throw_cuda(
        "forward-backward-state-download",
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.canonical_x.data(),
       at<const void>(slot.device, impl.pool->layout.canonical_rows),
       x_elements, "cudaMemcpy(retained canonical X)");
  const auto *ab = at<const transport::BFloat16>(
      slot.device, impl.pool->layout.forward_ab_cache);
  copy(result.fc1_ab.data(), ab, plane_elements,
       "cudaMemcpy(retained FC1 A)");
  const auto b_plane_stride = checked_mul(
      impl.pool->config.max_destination_rows_per_rank, impl.intermediate,
      "forward retained B plane stride");
  copy(result.fc1_ab.data() + plane_elements, ab + b_plane_stride,
       plane_elements, "cudaMemcpy(retained FC1 B)");
  return result;
}
const ActivationTransportExpectation &
DeviceActivationTransportHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}
const ActivationTransportDeviceAudit &
DeviceActivationTransportHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}
const ActivationTransportTimings &
DeviceActivationTransportHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

ActivationTransportValidationDownload ActivationTransportBridge::download_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto hidden = impl.identity.hidden;
  const auto route_elements = checked_mul(impl.route_count, hidden,
                                          "download route elements");
  const auto combined_elements = checked_mul(impl.identity.tokens, hidden,
                                             "download combined elements");
  const auto canonical_elements =
      checked_mul(impl.destination_rows, hidden, "download canonical elements");
  ActivationTransportValidationDownload result;
  result.contribution_by_route.resize(
      checked_size(route_elements, "download route elements"));
  result.combined_tokens.resize(
      checked_size(combined_elements, "download combined elements"));
  result.canonical_expert_rows.resize(
      checked_size(canonical_elements, "download canonical elements"));
  auto copy = [&](void *destination, std::size_t offset, std::uint64_t count,
                  const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-download",
        cudaMemcpy(destination, at<const void>(slot.device, offset),
                   checked_size(checked_mul(count,
                                            sizeof(transport::BFloat16), name),
                                name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
  };
  const auto contribution_offset =
      impl.audit.backend ==
                  transport::ActivationTransportBackend::NcclSocketFallback ||
              impl.audit.backend ==
                  transport::ActivationTransportBackend::Gin
          ? impl.pool->layout.combine_return
          : impl.pool->layout.contribution;
  copy(result.contribution_by_route.data(), contribution_offset,
       route_elements, "download contributions");
  copy(result.combined_tokens.data(), impl.pool->layout.combined,
       combined_elements, "download combined");
  copy(result.canonical_expert_rows.data(), impl.pool->layout.canonical_rows,
       canonical_elements, "download canonical rows");
  result.protocol.dispatch_packets = slot.host_status->packets_done;
  result.protocol.relayed_fanout = slot.host_status->fanout_done;
  result.protocol.synthetic_contributions = slot.host_status->synthetic_done;
  result.protocol.combine_packets = slot.host_status->combine_done;
  result.protocol.reduced_tokens = slot.host_status->reduced_tokens;
  result.protocol.credit_stalls = slot.host_status->credit_stalls;
  result.protocol.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result.protocol.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  return result;
}

ActivationTransportValidationDownload download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  return ActivationTransportBridge::download_for_validation(handle, execution);
}

std::vector<transport::BFloat16> download_activation_output_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "transport-output-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "transport-output-download", "device",
                    impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "output download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "output download elements"));
  if (!output.empty()) {
    const auto bytes = checked_mul(elements,
                                   sizeof(transport::BFloat16),
                                   "output download bytes");
    internal::throw_cuda(
        "transport-output-download",
        cudaMemcpy(output.data(),
                   at<const void>(
                       impl.pool->slots[impl.slot->index].device,
                       impl.pool->layout.combined),
                   checked_size(bytes, "output download bytes"),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(final output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}
