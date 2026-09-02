#!/usr/bin/env python3
"""Compile-negative proofs for every retired pre-coordinator protocol entry."""

import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cases = {
    "old_wait": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { megamoe::cuda::CudaNotifyOperation op; (void)op.wait(); }
''',
    "wait_preflight": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { megamoe::cuda::CudaNotifyOperation op; (void)op.wait_preflight(); }
''',
    "raw_notify_start": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { (void)megamoe::cuda::start_cuda_rank_local_notify; }
''',
    "control_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_control_collective; }
''',
    "config_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_config_collective; }
''',
    "count_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_count_collective; }
''',
    "summary_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_summary_collective; }
''',
    "internal_operation": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { megamoe::cuda::NcclControlOperation operation; }
''',
    "internal_backend": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { megamoe::cuda::NcclAllGatherCountExchange backend; }
''',
    "bootstrap_hex": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { megamoe::cuda::NcclBootstrapId id; (void)id.hex(); }
''',
    "bootstrap_parse": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::parse_nccl_bootstrap_id("00"); }
''',
    "unpooled_production_coordinator": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() {
  megamoe::cuda::NcclSession session;
  megamoe::cuda::CudaExecution execution;
  megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                               execution);
}
''',
    "generic_route_rebind": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::DeviceBufferLease generic;
  megamoe::cuda::CudaExecution execution;
  (void)megamoe::cuda::make_device_route_input(generic, 1, 1, execution);
}
''',
    "public_test_route_upload": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::RankRoutingInput routes;
  megamoe::RankCollectiveContext context;
  megamoe::cuda::CudaExecution execution;
  (void)megamoe::cuda::upload_device_routes_for_testing(routes, 1, context,
                                                         execution);
}
''',
    "public_generation_reissue": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::CudaRouteInput routes;
  megamoe::RankCollectiveContext context;
  megamoe::cuda::CudaExecution execution;
  (void)megamoe::cuda::reissue_cuda_routes_for_testing(routes, context,
                                                        execution);
}
''',
    "test_pipeline_handoff": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { (void)megamoe::cuda::test_pipeline_handoff; }
''',
    "unsafe_route_import_without_wait": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::CudaRouteInput routes;
  megamoe::RankCollectiveContext context;
  megamoe::cuda::CudaExecution execution;
  (void)megamoe::cuda::unsafe_import_cuda_routes_without_wait_for_testing(
      routes, context, execution);
}
''',
    "wrong_route_dtype": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::SelectedExpertRouteLease<std::uint64_t> wrong_dtype;
}
''',
    "wrong_gate_dtype": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::SelectedExpertGateLease<std::uint16_t> wrong_dtype;
}
''',
    "generic_gate_rebind": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::DeviceBufferLease generic;
  (void)megamoe::cuda::make_selected_expert_gate_input(generic);
}
''',
    "dispatch_checked_raw_view": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::DeviceDispatchHandle handle;
  megamoe::cuda::CudaExecution execution;
  megamoe::cuda::DeviceDispatchViewExpectation expected;
  auto view = handle.checked_view(execution, expected);
  (void)view.route_dst_row;
}
''',
    "public_kernel_view": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { megamoe::cuda::DeviceDispatchKernelView view; }
''',
    "public_generic_dispatch_buffers": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::DeviceDispatchHandle handle;
  (void)handle.device().route_dst_row;
}
''',
    "activation_wrong_dtype": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() {
  megamoe::cuda::SelectedActivationLease<std::uint16_t> wrong;
}
''',
    "generic_activation_rebind": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() {
  megamoe::DeviceBufferLease generic;
  (void)megamoe::cuda::make_activation_input(generic);
}
''',
    "raw_peer_pointer": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() {
  megamoe::cuda::LocalCudaP2pActivationTransport backend;
  (void)backend.peer_pointer(1);
}
''',
    "private_ipc_factory": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() { (void)megamoe::cuda::make_local_p2p_ipc_mapping; }
''',
    "activation_backend_enum_spoof": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() {
  megamoe::cuda::DeviceActivationTransport transport;
  megamoe::cuda::DeviceDispatchHandle dispatch;
  megamoe::cuda::BFloat16ActivationLease input;
  megamoe::transport::ActivationTransportConfig config;
  (void)transport.run_synthetic(
      dispatch, input, config,
      megamoe::transport::ActivationTransportBackend::LocalCudaP2p);
}
''',
    "installed_testing_namespace": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() { (void)megamoe::cuda::testing::cuda_api_audit_snapshot(); }
''',
    "retired_testing_macro_cannot_restore_api": r'''
#define MEGAMOE_INTERNAL_TESTING_API 1
#include "megamoe/cuda/activation_transport.hpp"
int main() { (void)megamoe::cuda::testing::cuda_api_audit_snapshot(); }
''',
    "installed_synthetic_transport": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() { megamoe::cuda::DeviceActivationTransport transport; }
''',
    "installed_validation_download": r'''
#include "megamoe/cuda/activation_transport.hpp"
int main() {
  megamoe::cuda::DeviceForwardHandle handle;
  megamoe::cuda::CudaExecution execution;
  (void)megamoe::cuda::download_forward_output_for_validation(handle,
                                                               execution);
}
''',
    "installed_host_oracle_wrapper": r'''
#include "megamoe/api/notify_dispatch.hpp"
int main() { (void)megamoe::build_notify_dispatch_for_testing; }
''',
    "host_integer_row_lifetime_hook": r'''
#include "megamoe/transport/count_exchange.hpp"
int main() {
  megamoe::HostIntegerRow row;
  (void)row.lifetime_for_testing();
}
''',
}
with tempfile.TemporaryDirectory() as directory:
    temporary = Path(directory)
    for name, source in cases.items():
        path = temporary / f"{name}.cpp"
        path.write_text(source)
        command = [os.environ.get("CXX", "c++"), "-std=c++17", "-Werror",
                   "-I", str(root / "include"), "-c", str(path),
                   "-o", str(temporary / f"{name}.o")]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            raise SystemExit(f"retired API unexpectedly compiled: {name}")
        if not result.stderr.strip():
            raise SystemExit(f"{name} failed without a compiler diagnostic")
print(f"[PASS] {len(cases)} retired/unpooled/raw-pointer APIs are compile-negative")
