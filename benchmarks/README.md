# Benchmarks

Use [tools/run.py](../tools/run.py) to benchmark FWD or BWD with an explicit
configuration, or to autotune before benchmarking. Native CUDA checks and rank
launchers are maintained in [tests](../tests).

- `megatron_dispatch_baseline.py` and `megatron_backward_baseline.py` measure
  Megatron/Flex FWD and cached-state BWD.
- `run_multinode_megatron_rdma.py` launches the multi-node IB/RoCE baseline;
  `run_multinode_megatron_socket.py` provides the Socket comparison and shared
  process management.
- `run_integrated_comparison.py` compares complete host-wall execution boundaries.
- `compare_forward_outputs.py`, `compare_backward_outputs.py`, and
  `compare_forward_ep_outputs.py` provide numerical and cross-EP checks.
- `forward_fp64_reference.py` is the independent numerical reference.
- `nccl_library.py` resolves the NCCL runtime. `dynamic_gate_fixture.json`
  supplies routing gates for reference checks.
- `notify_dispatch_bench.cpp` is built with
  `python3 build_native.py metadata-bench`.

Generated results belong in `results/`, which is ignored by Git.
