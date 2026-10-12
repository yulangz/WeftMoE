# PF 独立 notify 正式接入、权重重排与多机训练分析

结论：正式保留 **完整 notify → 可选缓存申请 → 共用 main**。相比此前保留版，带缓存 FWD 的 EP8 8K/20K 耗时下降约 0.9%/0.7%，EP16 下降约 1.8%/1.9%。W1 重排每卡 EP8 约 0.534 ms、EP16 约 0.268 ms。双机窗口 B2048/4096/8192 消融仍以 **B4096** 最好；不能只通过扩大窗口、降低 padding 获得收益。完整 FWD+BWD 的 20K 探索发生降频，不能用它宣称严格同频加速。

## 已完成的改动

`19ab646` 将已保留的借用 warp 数据面及 notify 拆分接回主分支。当前 streaming PF 默认执行 **完整 notify → 可选训练缓存申请 → main**，不再需要 `WEFTMOE_PF_SPLIT_NOTIFY` 或 `WEFTMOE_PF_SPLIT_INFERENCE_NOTIFY` 实验开关。

- inference 不申请 saved X/GU/gate/header，传空缓存指针；notify 与 main 在同一 stream 连续发射。
- 训练保存开关沿用 Python `TensorRuntime(training=...)` / C++ `retain_backward_state`，与所选 metadata 模式配套。需要缓存时，完整 notify 得到准确行数，再通过 tensor factory 申请 tape；无需预留最坏大小的激活缓存。
- 训练和推理复用同一个 local/network main。删除 main 内的 planner 及 `plan_ready` 分支；ring 初始化、epoch 推进、路由发布和物理布局统一由 notify 完成。训练仍生成稳定 canonical 行号及 BWD metadata，不复用上一调用的路由计划。
- Python 根据已编译 backend 的 `separates_forward_notify()` 能力配置 notify pool。DT 和旧/不支持的 backend 保持原入口；低层 C++ 调用方通过 `defer_training_notify` 接入延后计数，原先已物化的 metadata 仍可使用。
- 缓存指针就绪后，FC1 保存 X/GU 的方式不变；网络 gate/header 由对应 FC1 checkpoint owner 写出。没有修改 GEMM、dispatch/return、credit 或 BWD 数学流程。
- 起始设备时钟保留在 notify，终止时钟在 main；inference 设备总时间包含两个 kernel 及间隙。训练这个设备跨度还覆盖中间的 host 分配等待，因此分项使用独立 CUPTI，完整性能使用 wall。

main entry 资源：单机 168 registers、无 stack/spill；多机 168 registers，stack 从 336 B 到 320 B，静态 spill store/load 从 76/68 B 到 8/8 B。独立 notify 本地 90、网络 118 registers，均无 spill，使用 64 KiB shared。静态 spill 数不是动态访存字节数，也不将全部性能差异归因于它。

## 完整缓存 FWD

单位 ms。旧版是当前保留的 PF：单机 pull2、双机 borrowed-warp + mini-notify，不是 V3。每档两轮，旧→新→新→旧；每步取最慢 rank，再取 p50，最后平均两轮 p50。

| 拓扑 | 每 rank 名义 tokens | 旧版 | 正式拆分版 | 耗时变化 |
|---|---:|---:|---:|---:|
| EP8 | 8K | 8.297 | 8.220 | −0.93% |
| EP8 | 20K | 18.721 | 18.582 | −0.74% |
| EP16 | 8K | 12.660 | 12.426 | −1.85% |
| EP16 | 20K | 26.253 | 25.743 | −1.94% |

包含逐调用 W1 重排、notify、实际缓存申请及确认、main、同步，tape 仍存活；不含 BWD 或 tape 释放。单机 8K 的新结果 8.317/8.123 ms 存在轮间波动，小幅收益不宜过度解读。

无缓存 inference、权重已准备的完整 GPU 区间为 EP8 **6.634/16.700 ms**、EP16 **10.771/23.049 ms**。推理不包含逐调用权重重排，因此不能把上述训练 FWD 与它直接相减，称为纯缓存写出开销。

## 权重重排与 GPU 分项

另一次 CUDA-only CUPTI，单位 ms；每项为逐步最慢 rank 的 p50。分项之间不保证是同一个 rank/样本，不能直接相加解释完整 wall。

| 拓扑 / tokens | W1 重排 | 完整 notify | main |
|---|---:|---:|---:|
| EP8 / 8K | 0.534 | 0.376 | 6.913 |
| EP8 / 20K | 0.535 | 0.590 | 17.020 |
| EP16 / 8K | 0.268 | 1.306 | 10.402 |
| EP16 / 20K | 0.268 | 1.542 | 23.505 |

W1 从每专家独立的 gate/up 大片段，变为每 8 行 gate、8 行 up 交错；每行 hidden 方向仍连续。重排 kernel 通过 `uint4` 进行 128-bit vector 拷贝，**W2 不重排**。它与 tokens 数无关，只与权重 shape 有关：本例每卡 EP8 的 W1 为 1 GiB，EP16 为 512 MiB，完整读写各一次对应约 4.0 TB/s 的逻辑带宽。这不是整层算子的有效带宽。

训练计时仍每次重排，不按 tensor 地址假定权重未更新。其 GPU 时间约为双机完整 FWD 的 2.2%/1.0%，不是长序列的主要瓶颈；inference 的 immutable layer weight cache 则可在预热时完成重排，稳态不重复执行。

## 多机训练：窗口消融与瓶颈判断

### 窗口不是越大越好

同一新二进制、相同变化负载和两轮顺序 B2048→B8192→B4096→B8192→B2048→B4096。这里只改变 source window tokens，D2、132 lanes 和 credit windows2 不变。单位 ms；每卡 workspace 为 native pool 报告的 compute + GIN + notify bytes，单位 GiB，不包含权重、输入、输出、按实际路由申请的训练 tape 或 PyTorch allocator 的空闲缓存。

| B | 8K FWD | 20K FWD | 8K / 20K M128 padding | 8K / 20K 每卡 workspace |
|---:|---:|---:|---:|---:|
| 2048 | 13.204 | 28.853 | 23.65% / 21.92% | 4.885 / 6.447 |
| **4096（保留）** | **12.450** | **25.855** | 14.44% / 10.64% | 7.136 / 8.798 |
| 8192 | 13.013 | 26.245 | 9.85% / 5.00% | 12.225 / 13.887 |

padding 是由本轮实际 routes 按 `(source node group, source window, expert)` 汇总后逐项向上取整到 M128 得到的逻辑行数增量，分母为真实路由行数；统计在计时外完成。它不是 SM 利用率，也不是 profiler 实测的总指令数。相同负载单机全局按专家聚合的 padding 仅为 4.17%/1.65%。

- B2048 增加碎片与窗口轮转，完整耗时比 B4096 多 6.1%/11.6%。
- B8192 虽然进一步减少 padding，仍慢 4.5%/1.5%，每卡多占约 **5.09 GiB**。增量不只是 GIN slot；compute ring 的容量也随最大 M-block 数增长。
- 因而 **M-padding 是问题的一部分，不是唯一瓶颈**。窗口还影响任务可用时间、通信/计算重叠、工作集与权重重用。这里证明了“仅增大 B”无效，尚未用 timeline 分离这些因素，不能把 B8192 的回退单独归因于其中某一个。

### 后续值得做的优化，及边界

1. **先细分 notify 内部同步。** 双机完整 notify 为 1.306/1.542 ms。当前 `count_training_routes` 先做准确计数和跨 rank 交换，再构造 stable saved-row/BWD metadata 及窗口/packet 计划；训练需要的不只是一个 histogram。计数暂存复用了首个 `expert_prefix` 和 `SourceHeader`，因此存在“所有读者读完，才允许 prepare 覆盖”的第二次全节点同步。可消融一个很小的独立计数/错误区，尝试消除这次别名导致的同步；不能直接删 barrier。该优化尚未实施，收益未知。
2. **main 优先看等待与有效 M 聚合，而非继续放大缓冲。** 双机 main 为 10.402/23.505 ms，是 FWD 大头。可以先用 timeline 区分 dispatch-ready、compute-ring credit 和 return 等待；再考虑让通信 packet/window 大小与计算 M 聚合解耦，减少跨窗口的专家尾块浪费。跨窗口合并必须保留精确 saved-row 映射和释放条件，不能通过延长旧 slot 生命周期而覆盖仍被 FC1/BWD 使用的数据。本轮没有改这部分协议。
3. **权重重排是次要项。** 训练权重可能每步更新，不应重新引入按地址缓存旧重排结果的假设。以后可独立消融是否与 notify 重叠，但现在没有实现双 stream 并发，也不把 0.268 ms 当作已经省下的时间。
4. **如果目标是整步训练，BWD 要单独优化。** 本轮仅正式拆分 FWD，没有修改 BWD 计算。8K 完整调用的 BWD 及其等待/收尾约占 63%，继续只优化 FWD 会受这一比例限制；这不是按 CUPTI 分离出的纯 BWD GPU 时间。

以上是分析及下一步候选，不是已经兑现的性能收益。保留当前 B4096 数据面、训练/推理共用 main，不引入另一个训练专属 main。

### 完整 FWD+BWD 回归检查

旧→新→新→旧两轮，包含 tape/gradient 释放；单位 ms。固定 pool 已预热，逐步重新生成输入和路由。

| 每 rank tokens | 旧版 | 新版 | 频率审计与结论 |
|---|---:|---:|---|
| 8K | 33.724 | 33.502 | 测量跨度内采样均为 1620 MHz，小幅下降 0.66% |
| 20K | 74.686 | 73.803 | 四轮均采到低于 1620 MHz；仅保留原始观察，不能宣称同频收益 |

20K 最低观测为 1380 MHz，频率目标仍是 1620 MHz，功耗上限为 700 W。没有足够的 throttle-reason 采样确定降频原因。本轮不通过过滤低频样本重算更好看的数字，也不把该行用于正式 FWD 收益结论。

## 口径、验证与复现

BF16 H4096/I2048/E256/top-6，EP8 单机、EP16 双机，SM 目标频率 1620 MHz。每 rank tokens 围绕 8192/20480 正态波动 10%、截断 ±3σ，逐步刷新 routes/gates，共享 expert bias std=0.5，seed123。为合成变化负载，不是模型实录 trace。prime32、warm8、sample24；两版 workload hash 和 token 序列逐项核对。单机 B2048/D2，多机 B4096/D2、132 lanes、D32/R128、credit windows2、publication plan/return-ready 去重。

普通及 bias8/B512/D2 强偏斜路由覆盖单/双机 8K/20K，两步变化输入，共 192 个训练 rank/step，output、dX、dGate、dW1、dW2 均通过，最大 relative L2 为 2.7500e-4。另验证两种拓扑的 eager 和双次 replay CUDA Graph inference，共 192 个 rank/step，最大 relative L2 为 9.3530e-5。公共头文件独立编译通过；没有新增逐函数名的单测。

正式库在 `build/pf_notify_production/{local,network}`，对应 `configs/pf_inference_flash_20261008` / `configs/pf_network_l132_20261009`。网络编译额外使用 `WEFTMOE_PF_NETWORK_WORKERS=1`、`WEFTMOE_PF_BORROW_ALL_CTA=1`、`WEFTMOE_PF_BORROW_CONTROL_REGISTERS=88`；诊断通信模式关闭。构建完成后正式性能作业串行运行，正式计时不启用 CUPTI 或并行编译。

连续 500 ms 频率监控逐 case 与 wall 时间戳对齐：带缓存 FWD 对照、CUPTI 分项、inference、窗口消融的测量跨度内，已有 GPU 观测均为 1620 MHz。完整 cycle 的 20K 存在上述降频；8K 没有。11 个很短的 `(round, tokens, node)` 区间没有恰好落入频率采样点，JSON 保留空 frequencies，不当作已观测通过；500 ms 采样也不能排除采样间的短暂变化。实验结束后，全部 16 卡显存恢复到 0 MiB。

运行脚本 `benchmarks/run_pf_notify_production.py`，提供 verify/skew/verify-inference/verify-graph/performance/trace/inference/cycle/sweep；汇总脚本 `benchmarks/summarize_pf_notify_production.py`。日志与 manifest 在 `build/pf_notify_production/results`，连续频率日志在 `build/pf_notify_production/continuous_clocks`。`fc8cbb6` 修正了计时前的 padding 诊断 CPU/GPU 比对问题；失败的首次 EP16 对照没有产生性能数据，原日志保留，重跑带 retry 标签。

源代码 commit `19ab646`；后续 benchmark 改动不改变已冻结 native：

- local SHA256 `a2d3f1333fd85bc55ab7fc78f9eec6fc2fa9ee81ee2ef86af480657fb28bbfd2`。
- network SHA256 `69fc88bec088cd0831ee92b493299f8a08fa8c0874b318044e8cee32a2867461`。

机器可读的逐步记录、workload hash、两轮汇总、padding、workspace、正确性和逐 case 频率审计见 [pf_notify_production_20261012.json](../benchmarks/pf_notify_production_20261012.json)。本轮没有重测 Megatron；不要将此前 Megatron 的历史数字当作本轮同步配对结果。
