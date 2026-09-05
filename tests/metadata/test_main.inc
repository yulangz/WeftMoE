int main() {
  const std::vector<std::pair<const char *, std::function<void()>>> tests{
      {"rank_local_async_and_deterministic_mapping",
       test_rank_local_async_and_deterministic_mapping},
      {"existing_shapes_descriptors_and_repeatability",
       test_existing_shapes_descriptors_and_repeatability},
      {"activation_transport_protocol_oracle",
       test_activation_transport_protocol_oracle},
      {"topology_route_capacity_and_overflow",
       test_topology_route_capacity_and_overflow},
      {"typed_storage_lifetimes_and_unwaited_destruction",
       test_typed_storage_lifetimes_and_unwaited_destruction},
      {"operation_move_and_terminal_state_machine",
       test_operation_move_and_terminal_state_machine},
      {"backend_exception_boundaries_and_device_rejection",
       test_backend_exception_boundaries_and_device_rejection},
      {"context_kind_and_provenance_mismatch_messages",
       test_context_kind_and_provenance_mismatch_messages},
      {"malicious_completion_context_and_corruption_hooks",
       test_malicious_completion_context_and_corruption_hooks},
      {"handle_copy_move_workspace_and_overflow",
       test_handle_copy_move_workspace_and_overflow},
      {"fixed_control_protocol_agreement",
       test_fixed_control_protocol_agreement},
      {"bootstrap_exact_frame_io", test_bootstrap_exact_frame_io},
      {"bootstrap_real_pipe_one_shot", test_bootstrap_real_pipe_one_shot},
  };
  std::size_t passed = 0;
  for (const auto &[name, test] : tests) {
    try {
      test();
      ++passed;
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
      return 1;
    } catch (...) {
      std::cerr << "[FAIL] " << name << ": unknown exception\n";
      return 1;
    }
  }
  std::cout << passed << "/" << tests.size() << " tests passed\n";
  return 0;
}
