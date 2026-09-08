# Maintained benchmark tools

性能入口优先使用 [FWD/BWD autotune](../docs/autotune.md) 与 `tests/run_local_activation_p2p.py` / `tests/run_multinode_nccl.py`。

- `megatron_dispatch_baseline.py`、`megatron_backward_baseline.py`：Megatron/Flex FWD 与 cached-state BWD 基线。
- `run_multinode_megatron_rdma.py`：两机 IB/RoCE launcher；`run_multinode_megatron_socket.py` 提供 Socket 对照以及 RDMA 复用的进程管理函数。
- `run_integrated_comparison.py`：完整 host-wall 边界的端到端对比，由 Makefile 和参考测试使用。
- `compare_forward_outputs.py`、`compare_backward_outputs.py`：独立输出数值对照；`compare_forward_ep_outputs.py` 检查跨 EP 一致性，三者验证范围不同。
- `forward_fp64_reference.py`：独立 FP64 数值 oracle。
- `nccl_library.py`：统一运行库加载；`dynamic_gate_fixture.json` 是参考测试输入；`notify_dispatch_bench.cpp` 是 Makefile 使用的 metadata benchmark。

旧模型矩阵计划生成器、专用 runner、报告脚本和 Socket/GIN hash 比较器已删除。历史矩阵表格保留在 docs，新实验直接使用维护中的默认 autotune 与 Flex launcher。

`results/` 仅用于本地结果，已从 Git 移除并加入忽略规则；报告中的历史 JSON 文件名仅作来源标识，不保证新 checkout 包含这些结果。
