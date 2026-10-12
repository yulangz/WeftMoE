# PF：将 main 的 notify/planner 拆成独立 kernel

后续已在 `19ab646` 正式接入，见[正式实现与复测](pf_notify_production_20261012.md)；下文保留此前实验口径。

本文测试的是训练缓存 FWD；不带缓存的单机 inference 拆分另见[专门消融](pf_inference_split_notify_20261012.md)。

结论：可以替代 mini-notify，并保持训练缓存及 BWD 正确。完整 FWD 的收益不大：单机基本持平，双机 8K / 20K 耗时分别下降约 1.5% / 0.6%。本次保留为实验，不切换默认实现。

## 完整缓存 FWD

单位 ms。原版是当前保留的 PF（mini-notify + main 内 planner），不是 V3。

| 拓扑 | 每 rank 名义 tokens | 原版 | 独立 notify + main | 耗时下降 |
|---|---:|---:|---:|---:|
| 单机 EP8 | 8K | 8.263 | 8.202 | 0.74% |
| 单机 EP8 | 20K | 18.710 | 18.705 | 0.03% |
| 双机 EP16 | 8K | 12.684 | 12.488 | 1.54% |
| 双机 EP16 | 20K | 25.937 | 25.781 | 0.60% |

统计为每步取最慢 rank，再取 p50，最后平均两轮 p50。每种拓扑按旧→新→新→旧运行；不带 profiler 的完整 FWD 为主结果。单机 8K 的两次新结果为 8.266 / 8.138 ms，运行间波动与平均收益相近，不宜宣称稳定加速。

口径与前次缓存测试一致：

- BF16，H4096 / I2048 / E256 / top-6，SM 目标频率 1620 MHz。
- tokens 在 8192 / 20480 附近按正态分布随机，标准差 10%，截断 ±3σ；每步刷新 routes/gates，共享 expert bias std=0.5，seed=123。
- 两版使用相同的每 rank workload hash、token 序列及容量，汇总脚本逐项核对。
- 完整 trace 预热一次，再 warmup 8 次、采样 24 次；统计到缓存就绪，tape 尚未释放。包含每次 W1 重排、notify、实际缓存申请及确认、main 和完成同步；排除一次性固定工作区、IPC/GIN 初始化和编译。
- 单机 B2048/D2、每 CTA 两个 pull warp；双机 B4096/D2，保留 all-CTA 借用 warp、132 lanes、D32/R128、publication plan、return-ready 去重。

## GPU 分项

下面是另一次 CUDA-only CUPTI 采样，每项均为逐步最慢 rank 的 p50，单位 ms。不同分项的最慢 rank 可能不同，不能直接相加解释上一表的完整墙钟时间。

| 拓扑 / tokens | 原 mini-notify | 原 main（含 planner） | 新独立 notify | 新 main |
|---|---:|---:|---:|---:|
| EP8 / 8K | 0.265 | 7.205 | 0.370 | 6.906 |
| EP8 / 20K | 0.304 | 17.592 | 0.590 | 17.183 |
| EP16 / 8K | 0.325 | 11.743 | 1.299 | 10.376 |
| EP16 / 20K | 0.441 | 25.013 | 1.526 | 23.487 |

W1 重排约为 EP8 0.535 ms、EP16 0.268 ms，两版相近。CUPTI 轮完整 FWD 与正式两轮均值相差不超过 0.73%；因此小于 1% 的差异不做过度解读。

main 变短，主要因为工作搬到了独立 notify，不能将这个降幅直接当作整体收益。拆分还需要主机读取实际行数并分配 tape。当前实验沿用现有 host API：输入/权重/output 准备后确认一次，notify 后申请 saved tensors 再确认一次；原路径将准备工作合在一次确认里。这些开销都包含在完整时间内，没有为了实验跳过分配失败的集体确认。

本次没有逐项测量这两次确认的 CPU 时间，不能将所有端到端差异都归因于它们。原始记录额外保留了同一步、同一最慢墙钟 rank 的 GPU 时间和剩余时间；剩余项还包含拷贝、NCCL 及 host 调度，并不等于纯 allocator 时间。

## 实现边界

原路径：mini 计数 → 权重/output/tape 准备及确认 → main 内稳定行映射、ring planner、通信计算。

实验路径：权重/output 准备及确认 → **完整独立 notify** → 按实际接收行数申请 tape 并确认 → **main 消费既有计划**。

1. 通过 `WEFTMOE_PF_SPLIT_NOTIFY=1` 延后训练 notify 的执行，先建立 owning metadata handle，不发射旧 mini kernel。
2. 独立的 384-thread resident kernel 完成精确计数、稳定 chunk 前缀、canonical 行号和原有 PF planner。计数临时复用固定 source metadata 的第一段 expert-prefix 空间；单机走已有 LSA 同步，多机走已有同 rail metadata Put 和 barrier。所有读取方退出后才允许完整 planner 覆盖这段临时空间，没有新增持久通信 buffer。
3. 复用已有 `build_training_rows`、本地 route exchange/geometry、多机 `ring::prepare` 和 BWD packet/fanout 构建。训练所需的精确计数仍需计算；删除的是独立 mini kernel 及其旧控制路径，不是省略这些数据依赖。
4. notify 完成后冻结精确接收行数，按它申请 saved X、G/U、gate/header；main 的 `plan_ready` 路径跳过重新规划。单机仍在 pull 时写 gate/header；多机由保存 X 的 FC1 tile 在第一次 A-stage ready 后顺带写 gate/header，避免提前申请最坏大小的 tape。
5. FC1/FC2 数学、X/GU checkpoint 数据写出方式、payload dispatch/return worker 和 ring credit 协议保持原实现。主机分配边界后仍使用原有通信计算流水线。

本次只验证了单机 PF 和 384-thread 多机借用 warp 路径；没有将实验开关推广为所有 backend 的生产 API。

## 正确性及资源

- EP8、EP16 均验证 256-token smoke，8K/20K 普通路由，以及 bias std=8、B512/D2 的强偏斜路由；每形状两个不同输入，共 240 个 rank/step。output、dX、dGate、dW1、dW2 全部通过。
- 最大 relative L2：smoke 0.0006562，普通长序列 0.0001163，强偏斜 0.0002750。
- 120 条验证 kernel 轨迹均没有 `training_count_notify_kernel`；每次正式分项采样也断言了独立 notify/main 各一次、mini 零次。
- 本地 main 168 registers、无 spill；多机 main 168 registers、336 B stack、76/68 B spill store/load，与保留版本相同。独立 notify：本地 84、跨机 111 registers，均无 spill，使用 64 KiB shared memory。
- 500 ms 连续频率日志中，正式性能与 CUPTI 采样的整段时间窗口内均为 1620 MHz。此前正确性验证阶段出现三个低频采样点（1575/1575/1485 MHz），全部位于该窗口之前，保留在原始审计中。测试结束两节点共 16 卡显存均恢复为 0 MiB。

## 代码与复现

实现 commit：`4a97c57`；频率审计脚本：`de4ff00`。分支 `experiment/pf-split-notify-20261012`，工作树：

`/storage/openpsi/codes/wanghaitao.wht/project/ops/WeftMoE-experiments/split-notify-20261012`

该工作树中的 `benchmarks/build_split_notify.py` 构建 local/network 两个配置；`benchmarks/run_split_notify.py` 提供 smoke、verify、skew、performance、trace；`benchmarks/summarize_split_notify.py` 汇总并校验 trace 一致性。新 artifact 在 `build/local`、`build/network`；原始日志、参数及动态库哈希在 `build/results`，连续时钟在 `build/continuous_clocks`。

对照 artifact：单机 `warp-budget-20261011/build/local_warp_budget/pull2`；双机 `borrow-training-20261011/build/cached`。launcher manifest 中的 `source_commit` 是 launcher 工作树版本，不是新 CUDA 版本；以本节实现 commit 和冻结 `.so` 哈希辨认 native。

完整机器可读记录：[pf_split_notify_20261012.json](../benchmarks/pf_split_notify_20261012.json)。此前的分项测量：[pf_cached_fwd_breakdown_20261012.md](pf_cached_fwd_breakdown_20261012.md)。
