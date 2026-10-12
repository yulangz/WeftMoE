# 单机 PF inference：独立 notify 的性能消融

后续已在 `19ab646` 正式接入，见[正式实现与复测](pf_notify_production_20261012.md)；下文保留此前实验口径。

结论：Flash 规模、EP8、8K/20K 下，拆出 notify 后完整推理性能基本持平，没有测到明显回退。这里是 **inference，不保存训练中间结果、不执行 mini-notify、不申请训练缓存**；与此前的[训练缓存实验](pf_split_notify_20261012.md)分开看。

## 完整推理结果

单位 ms。GPU 总时间从原 planner 开始到最终 combine 完成；拆分版包含 notify、两次 kernel 之间的间隙及 main，绝非只计缩短后的 main。

| 每 rank 名义 tokens | 当前保留版 GPU | 拆分版 GPU | GPU 耗时变化 | 当前保留版 wall | 拆分版 wall |
|---|---:|---:|---:|---:|---:|
| 8K | 6.648 | 6.675 | +0.40% | 6.876 | 6.847 |
| 20K | 16.709 | 16.750 | +0.24% | 16.913 | 16.967 |

为了分离重新编译和改变 kernel 边界的影响，同一个新二进制也关闭开关复测：

| 每 rank 名义 tokens | 新二进制融合 GPU | 新二进制拆分 GPU | 耗时变化 |
|---|---:|---:|---:|
| 8K | 6.703 | 6.675 | −0.42% |
| 20K | 16.750 | 16.750 | −0.003% |

三组均运行两轮，顺序为保留版→新融合→新拆分→新拆分→新融合→保留版；每轮每步取最慢 rank，再取 p50，表中取两轮 p50 的算术平均。拆分版逐轮 GPU 为 6.706/6.643、16.695/16.805 ms；这些小幅差异不足以证明稳定加速或回退。默认实现暂不切换。

另一次 CUDA-only CUPTI 诊断：

| 每 rank 名义 tokens | 独立 notify | 拆分后的 main | 融合版整个 main |
|---|---:|---:|---:|
| 8K | 0.150 | 6.530 | 6.603 |
| 20K | 0.286 | 16.403 | 16.746 |

这是独立 profiler 轮的分项 p50，不是正式完整耗时表的分解；不同项的最慢 rank/样本可能不同，不能直接相加。notify 的 0.150/0.286 ms 是从原 main 中搬出的工作，**不是新增的等量开销**。

## 改动及为何影响较小

原路径：一个 main，先准备路由和 X，再运行 pull/FC1/FC2/return/reduce。

实验路径：`inference_notify_kernel` → `main(plan_ready=true)`，同一 CUDA stream 连续发射。

- 将原本的初始化、source 路由统计/发布、X staging、combine 顺序和 expert/M-tile geometry 提取为共用 `local::prepare`。融合 main 和独立 notify 调用同一实现，不额外计算一遍训练的精确计数或 canonical saved-row 映射。
- notify 完成 ring/status 初始化和 epoch 推进；main 不重复清空计数或规划。所有固定 GM 缓冲仍复用原分配，三个变体的 workspace bytes 完全相同。
- 不在两个 kernel 之间读取行数到 CPU，不分配 saved X/GU，不执行训练分配确认或额外主机同步。普通 stream 顺序保证计划可见。
- 保留 notify 写入的起始时钟，main 完成时更新终止时钟，确保 native 计时覆盖完整设备区间。另测完整调用至同步完成的 wall 时间。
- pull warp、GEMM 参数和计算、FC epilogue、return/reduce、ring credit 均不变。原 main 本来就在 planner 完成并同步后才开始后半段流水线，拆分没有割断后半段的通信/计算重叠。

main 仍为 384 threads、132 CTA、cluster=2、2 pull warps、222 KiB shared、168 registers、stack/spill 均为 0。独立 notify 使用同样的 CTA/thread 数，64 KiB shared、86 registers、stack/spill 均为 0。没有在本轮搜索 notify 的 CTA 数或 shared memory 配额。

## 口径与验证

- 单机 8 GPU，BF16，H4096/I2048/E256/top-6，B2048/D2，配置 `pf_inference_flash_20261008`。
- tokens 在 8192/20480 附近正态波动，标准差 10%、截断 ±3σ；每步刷新 routes/gates，共享 expert bias std=0.5、seed=123。为模拟实际变化的合成负载，并非模型采集 trace。
- 三组同一 workload hash、token 序列及容量；完整 trace prime32、warm8、sample24。固定池/IPC 初始化和 immutable weight packing 已预热并摊销；路由规划与 X staging 每次执行且计时。wall 使用相同 native profiling 调用并等待 device 完成。
- 编译完成后串行测试，正式性能测试不启用 CUPTI；分项另测。连续 500 ms 时钟采样中，性能及 CUPTI 测量整段区间的 2824 个 GPU 观测均为 1620 MHz，整个本轮监控也未发现低频点。
- 拆分版验证普通随机路由及 bias8/B512/D2 强偏斜路由，均覆盖 8K/20K、每形状两个变化输入，共 64 个 rank/step；与 PyTorch reference 的最大 relative L2 为 9.353e-5，全部通过。
- 分项轨迹逐次确认：融合版一次 main；拆分版一次 inference notify 加一次 main；两版均无 training mini-notify。全部实验结束后 8 卡显存均恢复到 0 MiB。

本轮结论仅适用于上述单机 inference 规模，不据此推断 decode、训练或多机的拆分成本。

## 保存与复现

native 实现及运行/汇总脚本：`f63d7c8`，分支 `experiment/pf-inference-split-notify-20261012`，工作树：

`/storage/openpsi/codes/wanghaitao.wht/project/ops/WeftMoE-experiments/inference-split-notify-20261012`

在此前训练拆分实验基础上新增独立的 inference 开关 `WEFTMOE_PF_SPLIT_INFERENCE_NOTIFY=1`；没有打开训练开关 `WEFTMOE_PF_SPLIT_NOTIFY`。主分支只记录 benchmark 与结果，不更改默认 CUDA 实现。

- 构建：`/opt/.venv/bin/python benchmarks/build_split_notify.py local`。
- 运行：`benchmarks/run_inference_split_notify.py {verify,skew,performance,trace} --port PORT`；四个 phase 分别运行，使用不冲突端口。
- 新 artifact：`build/local`；native SHA256 `257ccf0da192231feb684a4c0a6ae34cc398a9a7a54414fb90c21a66052eb407`。
- 保留版 artifact：`../warp-budget-20261011/build/local_warp_budget/pull2`。launcher manifest 中的 source commit 属于 launcher 工作树，不是新 native 源码；以本节 commit/hash 为准。
- 原始日志/manifest：`build/results`；连续频率：`build/continuous_clocks`；汇总脚本：`benchmarks/summarize_inference_split_notify.py`。

机器可读的逐轮样本、路由 hash、正确性、分项、频率审计及 artifact manifest：[pf_inference_split_notify_20261012.json](../benchmarks/pf_inference_split_notify_20261012.json)。
