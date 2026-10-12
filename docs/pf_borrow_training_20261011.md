# 借用 warp 多机 PF 接入训练缓存（2026-10-11）

2026-10-12 补充分段计时：mini-notify 完整入口 0.429/0.537 ms、四次缓存
申请 0.052/0.040 ms、main 11.749/24.934 ms，见[分段测量](pf_cached_fwd_breakdown_20261012.md)。

已将保存 X、gate/up 接到当前保留的 2-pull + 1-network 主核。EP16 的
8K/20K 缓存 FWD 为 **12.625/25.919 ms**，相对旧 PF 训练入口降低
**24.84%/24.00%**；相对 Megatron `moe_act`，8K 接近、略慢 **0.53%**，
20K 快 **8.44%**。这不是完整 FWD+BWD 或整模型训练 step 的性能。

## 实现

代码在 `experiment/pf-borrow-training-20261011`，提交 `d69dbab`，基于保留的
`8c47a46`。工作目录是 `../WeftMoE-experiments/borrow-training-20261011`；
本轮没有将整条网络实验分支合入主分支，也没有修改 BWD 或 GEMM 超参。

只改三个 native 接点，复用既有训练设施：

1. Host 允许 384-thread 借用 warp 入口接收训练缓存；WG3 的训练入口仍
   保持旧行为，因为既有训练行扫描按 384 threads 实现，本次未扩展 WG3。
2. 同一 `network::kernel` 按 `retain_backward_state` 选择训练版 ring
   planner。mini-notify 仍只负责计数/确定缓存大小；main 内生成 canonical
   saved-row、gate/header 和 BWD packet/fanout。source-row 目录随原有
   跨机元数据发布，payload pull 不增加一次行号搜索或额外缓存拷贝。
3. 原本只对 inference 生成的 return publication plan 同样覆盖训练，
   确保快速 return worker 使用的 local-row/remote-source 信息完整。

X 写出复用 FC1 从 X ring 经 TMA 读入 shared 的 A tile；由每个 M tile
指定的一对 N tile 各保存半个 M tile，避免重复写。vector 保存与已发出的
WGMMA 同时推进，完成 shared 读取后才归还 stage。gate/up 复用 FC1 的
累加器直接写到 canonical tape；不再发独立缓存写出 kernel。

仍然是 **mini-notify → 按实际收到行数分配 tape → 一个 FWD main**。
inference 不申请 tape、传空缓存指针，使用同一个 main 的无缓存分支。
跨机 per-token Get dispatch、机内 bulk pull、FC2 contribution/Put return、
RX credit 和本地 NIC completion 均不变。

## 公平对照

双机 16 卡 L20X，SM 固定 1620 MHz，BF16 H4096/I2048/E256/top-k6。
每卡 tokens 在 8192/20480 附近正态波动 10%、截断 ±3σ，routes/gates
每步变化，共享 expert bias std=0.5，seed123；是合成热点负载，不是实录。
新旧 PF 和 Megatron 使用相同 trace，并与上一轮记录逐 rank hash 对齐。

prime32/warm8/sample24，执行新 PF→旧 PF→Megatron→Megatron→旧 PF→新 PF。
每步取最慢 rank，再取 p50；主表是两轮独立 p50 的平均，单位 ms。

| 每卡 tokens | 新借用 warp PF | 旧 PF 训练入口 | Megatron + DeepEP | 新 PF vs 旧 PF | 新 PF vs Megatron |
|---|---:|---:|---:|---:|---:|
| 8K | 12.625 | 16.797 | 12.559 | −24.84% | +0.53% |
| 20K | 25.919 | 34.103 | 28.309 | −24.00% | −8.44% |

逐轮：新 PF 12.619/12.630、25.884/25.954；旧 PF 16.695/16.898、
34.108/34.099；Megatron 12.548/12.569、28.007/28.610。
上一轮旧 PF 为 16.722/34.152，Megatron 为 12.705/28.398 ms；复测趋势一致。

主指标从 FWD 调用开始到 device synchronize、缓存仍存活且已就绪为止。
包含 mini-notify、动态缓存申请、input staging、路由转换、计算、通信、
缓存写出和**每次 W1 重排**。不恢复“权重不变”缓存优化。排除固定池/IPC/
GIN 初始化、首次编译及随机输入生成；正式计时不运行 BWD、CUPTI profiler
或并行编译。Megatron 仍为 TEGroupedMLP + Flex/DeepEP、48 SM、TP1/ETP1、
selective `moe_act`、permutation/weighted-SwiGLU fusion，计时段无新图编译。

额外包含 tape 释放及再次同步时，新 PF 为 12.659/25.956 ms，旧 PF
16.852/34.153 ms，Megatron 12.596/28.399 ms，结论不变。

Useful GEMM TFLOPS/card：新 PF **192.5/237.2**，旧 PF 147.3/183.9，
Megatron 197.5/219.5。逐步按 `6*sum(actual_tokens)*top_k*H*I/EP/seconds/1e12`
计算，再合并两轮取 p50；不是 nominal tokens 除以主表平均时间，故吞吐比
不必精确等于主表延迟比。padding/elementwise/通信开销计入时间但不计 FLOPs。

新路径 B4096/D2、132 lanes、D32/R128、pending4、credit windows2，启用
publication plan/return-ready 去重；这些是保留的网络配置，不是本次新优化。
旧库的训练入口不识别这些网络主核选项，保持原协议/容量。新旧区别不只是
“缓存写出快了”，而是训练终于使用已经优化的通信/GEMM 主核。

## 正确性、资源与复现

无缓存 inference 回归按旧→新→新→旧执行，沿用相同随机负载和预准备
权重口径，不将其时间直接当成上面“每次重排的缓存 FWD”的可减基数。

| 每卡 tokens | 旧 kernel ms | 新 kernel ms | 旧 eager ms | 新 eager ms |
|---|---:|---:|---:|---:|
| 8K | 10.957 | 10.825 | 11.280 | 11.112 |
| 20K | 23.514 | 23.202 | 23.765 | 23.469 |

kernel 两档分别 −1.21%/−1.33%，没有观察到接入缓存后的 inference 回退。
不将小幅变化归因于本次未做的调优，也不外推到其它 shape。

256-token 两步 smoke、8K/20K 各两步正常路由，以及 B512/bias8 的 8K/20K
各两步强偏斜复用检查，共 **160 rank/sample**，全部通过 Torch 参考的
output、dX、dGate、dW1、dW2 校验（rtol0.02/atol1e-4、relative L2<1%）。
三组最大 relative L2 分别 0.04842%、0.01098%、0.02750%。实际 trace 均为
`weftmoe_pf::network::kernel`，不是旧 `weftmoe_pf::kernel`。BWD 只作正确性
验证，本轮不报告其性能；两节点以外未实测。

仍为 384 threads/132 CTA、Math208/Control88、222KiB shared；entry REG168、
stack336B，Get/Put spill 未增加。入口静态 spill store/load 从 56/48B 到
76/68B；不能将它解释成动态流量。SASS 保持每 K64 四条 N256 WGMMA 后
一次 wait，没有新增 WGMMA 串行化。没有因训练另分一套 GEMM 主核。

每卡固定 compute+GIN+notify workspace 相比旧路径，8K 减少约 0.545 GB，
20K 增加约 0.389 GB（十进制）。差异包含缩小通信 slot 容量和改为完整
contribution 空间，并非单纯新增 tape。固定池仍在初始化分配；实际 X/GU
tape 按每次路由计数独立申请，以上 workspace 不含动态 tape 或 NCCL 内部内存。

复现脚本位于代码 worktree 的 `benchmarks/run_borrow_training.py`：
`smoke / verify / skew / performance / inference` 五种阶段。复用主仓库
`c3e89e4` 的 `tools/benchmark_training.py`，没有新增细粒度单测。
库在 `build/cached/`，日志/时钟/launch manifest 在 `build/results/`。
新旧库分别冻结，全部编译结束后才串行测试。

正式缓存 FWD 和 inference 回归分别有 3488/2688 条 launcher GPU 频率
观测，全部 1620MHz；另有连续 500ms 采样，两个节点分别 11480/11752 条，
同样全部 1620MHz，无查询错误。未计时的强偏斜正确性测试有一条 node0/
GPU2 的 1515MHz 观测，原样保留，不用于性能表；轮询不保证逐周期恒频。
全部 13 次双机作业均 exit0；停止本轮监控后 16 张 GPU 均为 0MiB。

完整计时、输入 hash、库 hash、入口 trace、正确性和编译资源见
[机器记录](../benchmarks/pf_borrow_training_20261011.json)。
