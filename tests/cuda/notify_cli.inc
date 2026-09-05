int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--local") {
      const auto world = static_cast<std::uint32_t>(parse_u64(argv[2]));
      print_nccl_runtime_info();
      test_local(world);
      if (world == 2) {
        test_device_capacity_failure_and_recovery();
#if !MEGAMOE_DEBUG_VALIDATE
        test_release_fused_status_and_config_fallback();
#endif
        test_protocol_failure_suite();
      }
      if (world == 1) {
        test_randomized_generation_scheduler();
        test_device_dispatch_handle_lifetime();
        test_delayed_route_producer_readiness();
        test_resource_pool_and_cuda_api_audit();
        test_fault_injection();
        test_config_wire_corruption(1, 0);
      }
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--protocol-regressions") {
      test_protocol_failure_suite();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--device-smoke") {
      test_device_only_sanitizer_paths();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--v3-protocol-smoke") {
      const auto probe =
          megamoe::cuda::testing::run_persistent_v3_protocol_probe();
      require(probe.rdma_ctas == 8U && probe.relay_ctas == 8U &&
                  probe.compute_ctas + probe.rdma_ctas + probe.relay_ctas ==
                      probe.sm_count,
              "V3 fixed CTA role partition mismatch");
      require(probe.tasks_completed == 8ULL * probe.tasks_per_context &&
                  probe.observed_checksum == probe.expected_checksum &&
                  probe.terminal_epoch != 0ULL &&
                  probe.terminal_publish_count == 1ULL &&
                  probe.error_code == 0ULL,
              "V3 ticket/scoreboard protocol probe failed");
      std::cout << "[PASS] V3 fixed roles rdma=" << probe.rdma_ctas
                << " relay=" << probe.relay_ctas
                << " compute=" << probe.compute_ctas
                << " tasks=" << probe.tasks_completed
                << " terminal_publish_count="
                << probe.terminal_publish_count << '\n';
      const auto backward_probe =
          megamoe::cuda::testing::run_backward_scheduler_probe();
      require(backward_probe.dx_completed == 36ULL &&
                  backward_probe.dab_completed == 60ULL &&
                  backward_probe.dw2_completed == 28ULL &&
                  backward_probe.dw1_completed == 268ULL &&
                  backward_probe.total_completed == 392ULL &&
                  backward_probe.duplicate_claims == 0ULL &&
                  backward_probe.dependency_errors == 0ULL &&
                  backward_probe.priority_errors == 0ULL &&
                  backward_probe.timeout_errors == 0ULL &&
                  backward_probe.terminal_epoch != 0ULL &&
                  backward_probe.terminal_publish_count == 1ULL,
              "BWD four-queue scheduler probe failed");
      std::cout << "[PASS] BWD scheduler compute="
                << backward_probe.compute_ctas
                << " participating=" << backward_probe.participating_ctas
                << " tasks=" << backward_probe.total_completed
                << " priority=dx>dab>dw2>dw1\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--bench") {
      benchmark(static_cast<std::uint32_t>(parse_u64(argv[2])));
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--transport-bench") {
      transport_benchmark();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--transport-capability") {
      const auto capability =
          megamoe::cuda::probe_activation_transport_capability(8);
      std::cout << "TRANSPORT_CAPABILITY compiled_nccl="
                << capability.compiled_nccl_version
                << " runtime_nccl=" << capability.runtime_nccl_version
                << " device_headers=" << capability.nccl_device_headers
                << " gin_headers_available="
                << capability.gin_headers_available
                << " gin_symbols_present=" << capability.gin_symbols_present
                << " gin_runtime_initializable="
                << capability.gin_runtime_initializable
                << " gin_backend_implemented="
                << capability.gin_backend_implemented
                << " gin_usable=" << capability.gin_usable
                << " system_scope_atomics="
                << capability.system_scope_atomics
                << " reason='" << capability.gin_unavailable_reason << "'\n";
      for (std::size_t i = 0; i < capability.peer_access_matrix.size(); ++i)
        std::cout << static_cast<unsigned int>(
            capability.peer_access_matrix[i]);
      std::cout << '\n';
      const auto ordering_ok =
          megamoe::cuda::testing::run_local_p2p_put_signal_probe(
              0, 1, 64U * 1024U * 1024U);
      std::cout << "LOCAL_P2P_ORDERING bytes="
                << (64U * 1024U * 1024U)
                << " last_block_signal_verified=" << ordering_ok
                << " early_signal_rejected=" << ordering_ok << '\n';
      std::cout << "LOCAL_P2P_DEVICE_PUT_GBPS gpu0_to_gpu1="
                << megamoe::cuda::testing::measure_local_p2p_put_bandwidth_gbps(
                       0, 1, 64U * 1024U * 1024U, 5, 20)
                << " bytes_per_sample=" << (64U * 1024U * 1024U)
                << " warmup=5 measured=20\n";
      return 0;
    }
    if (argc >= 14 && std::string(argv[1]) == "--rank-runner") {
      std::uint32_t rank = 0, nodes = 0, gpus = 0, local = 0;
      std::uint64_t seed = 0;
      int bootstrap_input_fd = -1;
      int bootstrap_output_fd = -1;
      std::string scenario = "valid";
      for (int index = 2; index + 1 < argc; index += 2) {
        const std::string key = argv[index];
        if (key == "--rank") rank = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--nodes") nodes = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--gpus-per-node") gpus = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--local-rank") local = static_cast<std::uint32_t>(parse_u64(argv[index + 1]));
        else if (key == "--seed") seed = parse_u64(argv[index + 1]);
        else if (key == "--bootstrap-input-fd") bootstrap_input_fd = std::stoi(argv[index + 1]);
        else if (key == "--bootstrap-output-fd") bootstrap_output_fd = std::stoi(argv[index + 1]);
        else if (key == "--scenario") scenario = argv[index + 1];
      }
      print_nccl_runtime_info();
      run_rank(rank, nodes, gpus, local, seed, bootstrap_input_fd,
               bootstrap_output_fd, scenario);
      return 0;
    }
    throw std::runtime_error("invalid arguments");
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
