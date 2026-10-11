# PF 保存训练缓存的 FWD：与 Megatron 同口径配对（2026-10-11）

结论：包含 mini-notify、实际缓存申请、写出和逐调用权重重排后，单机 EP8
的 8K/20K FWD 比 Megatron + DeepEP 快 **13.46%/9.34%**；双机 EP16
慢 **31.62%/20.26%**。两边均保留 routed X、FC1 gate/up，缓存可用于 BWD。
这里不含 BWD 性能，也不是整模型训练 step。

**重要实现边界：最近优化的借用 warp 网络主核尚不支持训练缓存。**
`network_prefill_streaming()` 明确要求 `!config.enable_backward_state`。
本轮打开缓存后，单机仍走新 local main，双机则回到旧 PF 训练主核。
因此双机结果不能解释成“10.97/23.49 ms 的新推理主核仅增加缓存写出的开销”。
本轮只补 benchmark 和记录，没有修改 CUDA 或放开这个入口限制。

## 配对范围与主结果

- Flash BF16，H4096/I2048/E256/top-k6；单机八卡 EP8、双机十六卡 EP16，
  L20X，全部 GPU 设置 SM 1620 MHz。Megatron TP1、expert-TP1。
- 每 rank tokens 围绕 8192/20480 正态波动，标准差 10%、截断 ±3σ；
  每步重新生成 routes/gates，共享 expert bias 标准差 0.5，seed123。
  这是合成热点负载，不是实际模型采集的 trace。
- 三个版本同一完整 32 步 trace，prime32、warmup8、sample24。
  顺序为 PF → PF prepared → Megatron，再完全反序；两种拓扑串行执行。
  每步取最慢 rank，再取 p50；表格取两次独立 p50 的算术平均。
- 主指标是 **FWD 调用至 device synchronize 完成的 wall 时间，此时 tape
  仍然存活、缓存已就绪**。包括每次 notify、输入 staging/路由转换、动态
  缓存分配请求、计算和通信；PF 主表还包括每次 W1 重排。
  排除输入随机生成、首次编译、固定池分配、IPC/GIN 建连和模型初始化。
- Megatron 使用真实 TEGroupedMLP + Flex/DeepEP，selective `moe_act`，
  permutation/weighted-SwiGLU fusion 开启；DeepEP EP8=80 SM、EP16=48 SM，
  沿用历史较快配置，未额外穷举。没有关闭其默认快路径。

单位 ms；负数表示 PF 耗时更低。

| 拓扑 / T每卡 | PF：逐调用权重重排 | Megatron + DeepEP | PF 耗时变化 |
|---|---:|---:|---:|
| EP8 / 8K | 8.222 | 9.502 | **−13.46%** |
| EP8 / 20K | 18.786 | 20.722 | **−9.34%** |
| EP16 / 8K | 16.722 | 12.705 | **+31.62%** |
| EP16 / 20K | 34.152 | 28.398 | **+20.26%** |

逐轮 p50：PF 为 8.248/8.197、18.778/18.793、16.852/16.593、
34.141/34.163 ms；Megatron 为 9.460/9.543、20.786/20.658、
12.848/12.562、28.129/28.666 ms。四档的胜负在两轮中一致。

## 权重准备和释放边界消融

PF prepared 使用已有 layer 权重准备接口，仍执行相同输入检查并保存 X/GU，
只把固定权重的重排摊到 priming。它是本轮性能诊断，**不是恢复训练中跨
权重更新复用旧 packed weights 的优化**；主表仍使用逐调用重排结果。

| 拓扑 / T每卡 | PF 每次重排 | PF 权重已准备 | 两轮均值差 |
|---|---:|---:|---:|
| EP8 / 8K | 8.222 | 7.799 | 0.423 |
| EP8 / 20K | 18.786 | 18.326 | 0.459 |
| EP16 / 8K | 16.722 | 16.446 | 0.277 |
| EP16 / 20K | 34.152 | 33.772 | 0.380 |

这个差值包含重排、临时 packed buffer 管理及调用波动，不冒充孤立 copy
kernel 时间。预准备后双机仍明显慢于 Megatron，不能只靠省重排解决。

额外记录缓存就绪后释放 tape、再次同步的完整生命周期，便于与此前包含
释放的 eager 口径对应；不把下面的时间与主表混算。

| 拓扑 / T每卡 | PF 完整调用 ms | Megatron 完整调用 ms |
|---|---:|---:|
| EP8 / 8K | 8.256 | 9.547 |
| EP8 / 20K | 18.837 | 20.767 |
| EP16 / 8K | 16.840 | 12.877 |
| EP16 / 20K | 34.204 | 28.476 |

## 实际入口、正确性和频率

使用上一轮的冻结库，没有重编译：

- EP8：`62605ca`，`warp-budget-20261011/build/local_warp_budget/pull2`。
  B2048/D2，实际入口 `weftmoe_pf::local::kernel`，含训练缓存写出。
- EP16：`8c47a46`，`dispatch-optimized-20261010/build/optimized/borrow`。
  B4096/D2，实际缓存入口是 `weftmoe_pf::kernel<false, true, false, true>`，
  **不是** `weftmoe_pf::network::kernel`。同一库的 inference 与 training
  选择了不同主核；也没有将实验网络优化合入主分支生产默认。

两后端均完成 EP8/EP16、8K/20K、每档两步的共同 Torch 参考校验，共
192 个 rank/sample，各检查 output、dX、dGate、dW1、dW2。全部通过
rtol=0.02、atol=1e-4、relative L2<1%；最大 relative L2 为 PF 0.01163%、
Megatron 0.29531%。BWD 仅在该验证中执行，证明缓存可消费，不计入主表。
入口 trace 同样只在验证中采集；正式计时没有 profiler 或并发编译。

24 条性能记录的 tokens、route/gate hash、实际收到的行数及准备容量完全
匹配；route trace 也与 10 月 6 日的 V3/Megatron 对照一致。Megatron
八条性能记录的所有 rank 在计时段新增编译图均为零。
软件为 Megatron-Core 0.17.0、TE 2.14.1+27de67e1、DeepEP 1.2.1+567632d、
Torch 2.9.1+cu129，双方运行时 NCCL 2.31.2。

串行 launcher 在参与计算的节点共取得 4184 条 GPU 频率观测，均为
1620 MHz；双机期间另有连续 500 ms 采样，两个节点分别 6672/6880 条，
也均为 1620 MHz。首轮 Megatron EP8 的轮询线程在两档计时结束后的退出
阶段发生一次 nvidia-smi 查询超时；计时前后记录正常，未丢弃其性能样本。
轮询不能证明逐周期恒频，短于采样周期的区间也不保证有独立采样点。
全部作业正常退出，停止本轮监控后两节点 16 张 GPU 均为 0 MiB。

## 等效计算吞吐

主表 cache-ready wall 对应的 useful GEMM TFLOPS/card：

| 拓扑 / T每卡 | PF | Megatron |
|---|---:|---:|
| EP8 / 8K | 297.8 | 258.7 |
| EP8 / 20K | 329.7 | 299.3 |
| EP16 / 8K | 147.4 | 195.1 |
| EP16 / 20K | 184.0 | 219.5 |

逐 step 按 `6 * sum(actual_tokens) * top_k * H * I / EP / seconds / 1e12`
计算，合并两轮逐步吞吐再取 p50；不是 nominal tokens 除以平均 p50。
不计 padding/elementwise/拷贝 FLOPs，但这些工作和通信开销在时间内。

## 复现与后续边界

`tools/benchmark_training.py` 新增 `--forward-only`；复用原训练 backend、
路由生成器和数值参考，没有新建一套算子或细粒度单测。关键参数：

```text
--backend pf|megatron --forward-only --prime-trace --warmups 8 --samples 24
--hidden 4096 --intermediate 2048 --experts 256 --top-k 6 --tokens 8192 20480
```

PF 用上述冻结 extension、EP8 B2048/D2、EP16 B4096/D2；消融加
`--prepared-weights`。Megatron 加 `--dispatcher flex --deepep-num-sms 80|48`。
验证用 `--verify --trace-kernels --warmups 0 --samples 2`，不进行性能评分。

完整样本、逐 rank 时间、输入 hash、库 hash、launch manifest、正确性、
入口与时钟记录见 [机器记录](../benchmarks/pf_cached_fwd_20261011.json)。
原始日志和串行复现/汇总脚本在 `build/pf_cached_fwd_20261011/`；launcher
复用 `warp-budget-20261011` 的 `061ecf6`。库源码版本与 launcher 版本
分别记录，不能把后者当成 native 源码版本。

下一步若要评价**新多机通信主核**在训练 FWD 下的性能，需要接入 mini-notify
计数、canonical saved-row 映射及 X/GU 写出，再重新验证 BWD。当前网络 main
仍调用 `ring::prepare<true, false, ...>`，网络 pull 未接入单机保存行映射；
不能只删 host eligibility 条件就宣布支持训练。本次没有实现或预测其收益。
