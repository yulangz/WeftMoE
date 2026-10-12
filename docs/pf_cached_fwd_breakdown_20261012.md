# 借用 warp PF：缓存 FWD 分段计时（2026-10-12）

结论：EP16、8K/20K 的 mini-notify 完整入口约 **0.429/0.537 ms**，
四块训练缓存的 allocator 调用合计 **0.052/0.040 ms**，main kernel
**11.749/24.934 ms**。主要耗时仍是 main，不是显存申请。本轮只测量，
没有改变通信、缓存布局、GEMM、CUDA kernel 或生产绑定。

## 同口径分段结果

双机 16 卡 L20X、SM1620MHz，BF16 H4096/I2048/E256/top-k6，B4096/D2，
沿用保留的 2-pull + 1-network、132 lanes、credit windows2、publication
plan/return-ready 去重配置。每卡 tokens 正态波动 10%，routes/gates 每步
变化，bias std=0.5，seed123；trace/hash 与上一轮及历史缓存 FWD 一致。
每次 W1 重排仍计入，固定池/IPC/GIN 初始化和随机输入生成不计入。

prime32/warm8/sample24，原绑定→计时绑定→计时绑定→原绑定。
各段每步取最慢 rank，再取 p50，表中平均两轮结果，单位 ms：

| 阶段 | 8K | 20K |
|---|---:|---:|
| mini-notify 完整 host 入口：提交、GPU 工作及等计数就绪 | 0.429 | 0.537 |
| 训练缓存申请：四次 `torch.empty` 调用合计 | 0.052 | 0.040 |
| 完整缓存准备：含上述申请、句柄/事件/地址检查 | 0.087 | 0.067 |
| 全部六次 allocator 调用：另含 packed W1 和 output | 0.088 | 0.073 |
| 分配成功确认/等待：不属于 allocator 调用 | 0.386 | 0.464 |
| main GPU kernel：含内部 planner、计算/通信、X/GU 写出 | 11.749 | 24.934 |
| 其中 main 内 planner，已包含在上一行 | 1.359 | 1.571 |
| 完整缓存 FWD，计时绑定 | 12.675 | 25.883 |
| 完整缓存 FWD，原绑定对照 | 12.668 | 25.918 |

表中若干行是包含关系；各段最慢 rank/p50 样本也可能不同，不能直接相加。
完整 FWD 从调用开始到 device synchronize、缓存仍存活且就绪为止，不含
tape 释放或 BWD。计时绑定相对同期原绑定仅 **+0.055%/−0.134%**，没有
观察到明显测量扰动。上一轮新路径为 12.625/25.919 ms，复测一致。

## mini-notify 纯 GPU 时间与分配确认

快速 count-only Notify 入口没有填充 `GenerationExecutionTimings` 的
`prefix_planning_us`；诊断原始记录中的 `notify_gpu_span_us=0` 是未填字段，
**不是 kernel 零耗时**。为此另跑一轮 CUDA-only CUPTI，仍 prime32/warm8/
sample24；每 rank、每档准确捕获 24 个 notify、24 个 W1 重排、24 个 main：

| 独立 CUPTI kernel 计时 | 8K ms | 20K ms |
|---|---:|---:|
| `training_count_notify_kernel` | 0.330 | 0.444 |
| W1 interleave/repack | 0.269 | 0.269 |
| `weftmoe_pf::network::kernel` | 11.724 | 24.954 |

notify GPU 时间包含它内部的跨 rank 计数交换/等待；不是孤立 histogram
计算时间。CUPTI 轮的完整 FWD 为 12.759/25.983 ms，相对轻量计时轮约
+0.66%/+0.39%，不拿它替换主性能结果。main 与既有设备时间戳测量接近。

缓存申请后，`CollectiveTensorPreparation::agree()` 在同一 stream 上做
小量 H2D、NCCL AllReduce、D2H 和 stream synchronize。等待会覆盖此前
尚未完成的 W1 重排，以及 peer 到达/分配成功确认；不能把 0.386/0.464 ms
全部记成显存申请，也不能简单减去另一次采样的 0.269 ms 当作纯通信。

四个缓存是 saved X、saved G/U、saved gates、saved route headers，按
实际收到行数申请；另外两次是 packed W1 和 output。`torch.empty` 时间
包含 PyTorch/Python 绑定及 allocator 调用，不等于孤立 CUDA driver malloc。
第二轮轻量诊断和 CUPTI 轮，两个规模、全部 16 rank 的 `num_device_alloc`、
`num_device_free`、`num_alloc_retries`、`num_ooms` 增量均为 0，说明这次
预热后的稳态样本命中 caching allocator。第一轮加载脚本后才补上此统计，
未记录该字段；不声称它也直接测到了这些计数。结果不代表冷启动分配开销。

## 可相加的耗时占比

为避免跨阶段 max/p50 不可加，另按每步完整 FWD 最慢的 rank，取该 rank
的全部阶段，再对 48 个样本做算术平均：

| 部分 | 8K 占比 | 20K 占比 |
|---|---:|---:|
| mini-notify 完整 host 入口 | 2.89% | 1.61% |
| 缓存准备整体（含 allocator） | 0.44% | 0.17% |
| main GPU kernel | 92.49% | 95.76% |
| 其它：权重准备、分配确认、提交/返回等 | 4.19% | 2.47% |

这张占比表使用同-rank 均值，不是上一张独立 rank-max p50 的商。main
内部 planner 已经含在 main 中，不能再算到前置 mini-notify。

## 复现与审计

CUDA 库冻结于 `d69dbab`，SHA256 为
`53ed3599af382f212dcd379270dfadbf82e53192adc55224952e9b2b7327095f`，
与上一轮逐字节相同。只重编译实验 host adapter（`d5c8e23`），使用
`steady_clock` 记录 CPU 段、`DeviceForwardHandle::timings()` 读取既有
main 设备计时；没有加 GPU 同步点。这是训练 FWD 诊断绑定，不用作生产绑定。

实验分支 `experiment/pf-cached-breakdown-20261012`，工作目录
`../WeftMoE-experiments/cached-breakdown-20261012`；库和日志分别位于
`build/profile/`、`build/results/`。执行 `benchmarks/run_cached_breakdown.py`
跑正反序对照，加 `--trace` 跑独立 CUPTI；主仓库 benchmark 的
`--profile-stages` 需要此诊断 adapter。`summarize_cached_breakdown.py`
生成[完整机器记录](../benchmarks/pf_cached_fwd_breakdown_20261012.json)。

五次双机作业全部 exit0；3520 条 launcher GPU 时钟观测均为 1620MHz。
连续 500ms 采样两个节点分别 8328/8344 条，同样全部 1620MHz，无查询错误；
轮询不保证逐周期恒频。结束并停止监控后，16 张 GPU 均为 0MiB。
