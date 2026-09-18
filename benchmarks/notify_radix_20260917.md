**Notify 稳定排序优化，2026-09-17，1620MHz（测量配置更正）**

后续的[完成等待优化报告](completion_latency_20260917.md)记录了长尾定位及关闭校验后的完整复测；以下保留原始诊断数据与更正。

**更正：以下历史时间样本实际开启了逐次输出校验。** 测量脚本将 `WEFTMOE_BENCH_CHECK_REPEAT` 设为 `0`，但 native harness 按变量是否存在启用校验；原始记录的每 rank `checked_repetitions=40` 证实这一点。校验在计时区间外，仍可能扰动执行，故这些数据只能作为带校验条件下的诊断，不再作为正式性能收益证据。后续复测将完全移除变量并断言校验次数为零。数值一致性与 sanitizer 结果不受此更正影响。

本轮将 Notify radix sort 的逐元素区间扫描和逐输出二分查找，改为并行标记每个 digit 区间的起点，再直接计算长度和稳定输出位置。没有改动浮点计算、GEMM 累加或 routed contribution 归约顺序。

每个排序 chunk 内每个 digit 只有一个起点写入者；CTA 同步后，区间末端直接生成 histogram，散射位置仍为 `digit_base + prior_chunk_count + position - run_start`。因此相同 key 的源顺序和 chunk 顺序均保留。设备测试还覆盖了完整 64 位 key、全相同 key、跨 chunk 长区间和不满 tile 的尾块。

该对照从提交 `857e736fcb5a6c807f662ac7a9da54a38b29e58d` 重新构建基线，使用相同的 `configs/communication_recovery_20260916/{qwen,flash}.hpp`。两节点各八张 L20X 均锁定 1620MHz，测前测后读取全部 GPU 频率，测试后解除锁频。每轮 10 次预热、30 次测量，共两轮；第二轮反转 case 和实现顺序。以下为合并全部 60 个样本的中位数，每个样本先取各 rank 最大值。计入当次 tensor 申请、输入拷贝、Notify 和保留训练状态的 Forward。该时间矩阵未与编译或 profiler 并发，但实际开启了计时区间外的逐次校验。

| 模型 / EP / 每 rank tokens | Notify 前 → 后 (µs) | Notify 降低 | 完整 wall 前 → 后 (ms) | wall 降低 |
|---|---:|---:|---:|---:|
| qwen / 8 / 8192 | 496.7 → 465.0 | 6.4% | 3.266 → 3.235 | 0.96% |
| qwen / 8 / 65536 | 1035.3 → 1008.1 | 2.6% | 20.881 → 20.838 | 0.21% |
| qwen / 16 / 8192 | 887.7 → 595.5 | 32.9% | 5.861 → 5.556 | 5.19% |
| qwen / 16 / 65536 | 1710.8 → 1261.3 | 26.3% | 36.807 → 36.756 | 0.14% |
| flash / 8 / 8192 | 515.9 → 445.8 | 13.6% | 7.738 → 7.718 | 0.26% |
| flash / 8 / 65536 | 1065.0 → 1029.7 | 3.3% | 56.101 → 56.112 | -0.02% |
| flash / 16 / 8192 | 865.7 → 607.4 | 29.8% | 10.838 → 10.545 | 2.70% |
| flash / 16 / 65536 | 1597.6 → 1234.2 | 22.7% | 73.071 → 72.300 | 1.05% |

**端到端尾延迟仍未解决。** Notify 的中位数收益在两个运行顺序中均存在，但不能据此宣称整个算子的延迟稳定性改善或接近硬件上限。两版都有离群值，候选部分 case 的 wall 均值和 P95 更差；Flash EP8/64K 反序候选 wall P50 达 74.740ms，而同轮 Forward GPU P50 仍为 54.858ms，基线反序 wall P50 为 56.113ms。Qwen EP8/64K 基线反序也出现约 44ms 的 wall P95，GPU P50 仍约 19.629ms。host completion/callback 等待是待检查路径，当前没有定位证据证明根因。全部离群值保留，不过滤、不用 GPU 时间代替端到端 wall。

| 模型 / EP / tokens | wall 均值前 → 后 (ms) | wall P95 前 → 后 (ms) |
|---|---:|---:|
| qwen / 8 / 8192 | 3.251 → 4.932 | 3.298 → 33.368 |
| qwen / 8 / 65536 | 23.992 → 20.837 | 41.657 → 20.952 |
| qwen / 16 / 8192 | 5.857 → 6.049 | 6.017 → 6.531 |
| qwen / 16 / 65536 | 36.909 → 36.631 | 37.093 → 37.954 |
| flash / 8 / 8192 | 7.764 → 7.717 | 7.868 → 7.782 |
| flash / 8 / 65536 | 56.476 → 64.376 | 56.299 → 86.692 |
| flash / 16 / 8192 | 11.271 → 12.240 | 11.332 → 11.352 |
| flash / 16 / 65536 | 77.843 → 72.811 | 88.854 → 73.866 |

数值与协议验证：

- 独立 GPU radix 测试 48 组，完整 key 和源序号与 CPU `stable_sort` 完全相同；Compute Sanitizer memcheck 零错误，racecheck 零错误、零警告。
- EP1/EP2 元数据 oracle、非法输入到合法输入恢复、配置回退、非对称 retained slots、资源池复用和 CUDA API allocation audit 均通过。
- 两模型 × EP8/EP16 × 8K/64K，每 rank 20 次全输出逐字节重复校验通过。两轮配对运行和独立重复校验的输出、输入、路由、gate、权重哈希与基线相同。
- 两模型 EP8 的 129、258、8192、65536 batch 的共同 129-token 前缀逐字节相同；EP16 的 8192/65536 前缀相同。EP8×258 与 EP16×129 重新分片后，完整输出及输入/route/gate 字节一致。
- 两模型 EP16 的动态路由 inference graph 重放与跨 stream 消费通过；每个模型的 160 个 Forward/Backward dump 文件与已有参考逐文件一致，包括 dW。

复现与证据：

- [原始样本、构建信息与源文件哈希](notify_radix_20260917.json)。
- [稳定排序实现](../src/cuda/notify/radix.cuh)与[独立 GPU 回归](../tests/test_notify_radix.cu)。后者文件头包含独立编译命令。
- 原始构建、性能与验证日志在 `results/notify_radix_20260917/`；`baseline_source/` 保存基线，`measure.py`、`reverse.py`、`repeat.py` 记录准确执行参数，`correctness.log` 和两个 sanitizer 日志记录验证结果。

后续优先定位上述 host wall 长尾，再检查 local Notify 排序冗余位、radix 不变位轮次，以及 64K Forward 的通信和计算关键路径。当前结果只是有证据支持的一步优化，整体目标仍未完成。
