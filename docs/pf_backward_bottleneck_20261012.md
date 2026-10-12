# PF BWD：通信、GEMM 与数据流拆解（2026-10-12）

基线：`feature/pf-backward-20261012@8e11d73`，原生 artifact 为
`build/pf_main_timeline/bwd_spill_r192`（与该提交的原生实现一致）。
实验原始日志、配置、时钟位于 `build/pf_bottleneck_20261012/`。

目标按吞吐加速比定义：完整训练周期单机至少 Megatron 的 1.25 倍，
双机至少 1.20 倍，即延迟分别不高于基线的 80% / 83.33%。
下面首先记录测量基线；不将纯算术或通信替身的收益视为目标达成。

## 口径与复测基线

Flash H4096/I2048/E256/top6，EP8 为单机 8 卡，EP16 为双机各 8 卡。
每 rank 名义 8192 / 20480 tokens，长度正态波动 10%、截断于 3 sigma，
每步重新路由，共享 logit bias std=0.5，seed123。完整 prime trace 后
4 warmups / 16 samples；同拓扑使用相同路由与长度。
两机均请求 SM 1620 MHz，保留实际频率日志，不按结果筛选样本。

完整周期包含 FWD、BWD、每调用权重重排、缓存申请/释放和最终同步；
排除输入生成及首次持久池准备。Megatron 使用生产 Flex/DeepEP dispatcher、
TE grouped GEMM、`recompute_modules=["moe_act"]`，与 PF 一样保留 X/GU。
各步先取最慢 rank，再取 p50（ms）：

| 实现 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| Megatron | 24.656 | 57.463 | 33.423 | 81.522 |
| PF 当前基线 | 25.048 | 62.929 | 32.838 | 77.460 |
| 目标延迟上限 | 19.725 | 45.970 | 27.853 | 67.935 |

首次双机启动端口 33422 被占用，未进入计时；保留失败日志，换已检查的
空闲端口重跑。单机标签 `base_ep8_*`，双机 `base_retry_ep16_*`。

## 设备阶段，不使用 Python host 边界冒充 kernel 时间

使用现有 `WEFTMOE_BACKWARD_PROGRESS=1`，由
`benchmarks/summarize_pf_backward_phases.py` 解析本设备的 globaltimer。
仅取 priming/warmup 后对应的 trace 位置；不同设备时间戳不互减。
各列是独立的 rank-max p50，可能来自不同 rank，且部分阶段重叠，不能相加。

| 阶段（ms） | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| main 入口到首次 dU | 0.021 | 0.022 | 0.941 | 0.968 |
| dU/SwiGLU/dX envelope | 8.313 | 25.893 | 10.700 | 27.332 |
| dW1/dW2 envelope | 7.880 | 17.483 | 8.603 | 20.133 |
| reduce | 0.877 | 2.147 | 1.158 | 2.914 |
| main 内工作 envelope | 16.231 | 43.382 | 19.392 | 47.571 |

main envelope 到最后一次 dW/reduce 完成为止，不包含随后终止 barrier 的尾部。
该组启用日志诊断，与无插桩完整周期分开报告。`return_tail` 还包含等待
其他 rank 的时间，不能称为 RDMA 传输时间。

按每步真正的最慢阶段 rank 的收到行数计算，有效算力如下（TFLOPS）：

| 阶段 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| dU/SwiGLU/dX | 388.3 | 320.6 | 368.1 | 384.3 |
| dW1/dW2 | 417.7 | 473.7 | 462.2 | 508.0 |

两个阶段的有效 GEMM FLOPs 都是 `6 * received_rows * H * I`，不计
padding、SwiGLU 和路由标量运算。这里包含数据依赖等待，不是 tensor core
指令本身的利用率。

## 纯 GEMM

复用 `pure_backward_gemm.cu`，分离 math/producer 模板角色，使用 208/88
寄存器配额，避免诊断自身出现 `setmaxnreg ignored`。dU/dX 使用 wait0；
dW 使用生产算术循环。M128/N256/K64、三级流水、132 CTA、cluster2。
预分配输入，CUDA Graph replay，30 个交替样本，对照 BF16/FP32/BF16
cuBLAS，禁用降精度归约；完整输出校验和重复运行逐位一致检查通过。

**它测的是均衡尺寸的纯算术，不是上述随机热点路由的重放**。dU/dX
采用公共算术 helper 的 TMA epilogue，不包含新 stream 的 peer vector
写回、checkpoint、SwiGLU、ring 依赖。EP16 在这里仅改变每卡专家数量与
每专家行数，仍在单 GPU 上测算术，不含 RDMA。

| EP / tokens | PF 四个 GEMM 累计 ms | cuBLAS 累计 ms |
|---|---:|---:|
| EP8 / 8K | 7.451 | 7.062 |
| EP8 / 20K | 17.659 | 16.870 |
| EP16 / 8K | 7.219 | 6.933 |
| EP16 / 20K | 17.491 | 16.793 |

各 GEMM 约 644–722 TFLOPS；算术差距约 4%–6%，显著小于完整数据流
的有效算力落差。但均衡与热点、普通输出与通信 epilogue 不同，不能把差值
直接当作某个同步环节的精确耗时。

### 补测真实热点专家行数

`export_training_gemm_rows.py` 从完整基线日志重建 CUDA 路由随机流，
逐 rank 验证 workload SHA256，并检查全部 timed-step 的收到行数。
按“最热 rank 收到行数”的下中位数选择一个代表步骤，不按性能挑样本。
每个专家的真实行数通过 `.rows` 文件传入 `pure_backward_gemm`；
DU/DX 按 M128 补齐，dW 按 K64 补齐，padding 输入为零，有效 FLOPs
只计真实行数。cuBLAS 对照是 CUDA Graph 内的逐专家 GEMM，不是 TE
grouped GEMM。两个实现都预分配，完整输出和逐位重复检查通过。

| EP / tokens | 代表最热 rank 行数 | 四 GEMM ms：PF / cuBLAS | PF dU / dX / dW1 / dW2 TFLOPS |
|---|---:|---:|---|
| EP8 / 8K | 64637 | 10.538 / 10.032 | 656.0 / 662.3 / 662.7 / 463.9 |
| EP8 / 20K | 164439 | 25.162 / 23.518 | 685.3 / 688.0 / 689.9 / 539.0 |
| EP16 / 8K | 78801 | 12.107 / 11.389 | 676.8 / 682.9 / 707.1 / 520.0 |
| EP16 / 20K | 198734 | 28.821 / 27.417 | 690.9 / 691.4 / 728.7 / 641.4 |

这里 dW 仍使用旧的 4-pair chunk。热点负载暴露出均衡测试看不到的
dW2 效率损失，不能笼统说“四个 GEMM 都没有问题”。下一步用同一行数
重放 1-pair chunk；与生产的静态排程消融相互印证。该组全部 30 次计时
前后 NVML 采样为 1620 MHz，原始数据在 `gemm_ragged/results`。

## 纯通信替身

`WEFTMOE_PF_BWD_IDENTITY_DIAGNOSTIC` 只供诊断。保留生产 Dispatch、
机内转发、TX credit、Return、Reduce，将专家 Jacobian 替换成 identity。
没有 WGMMA、SwiGLU 或 dW；从 ready dY 读取后写入原 contribution/TX。
这个额外 GM 读也计时，因此不是“免费生成输出”的理想网络带宽。
FWD 仍正常运行以产生真实 tape；下表只取 BWD main 的设备工作区间。

| 拓扑 | 8K ms | 20K ms |
|---|---:|---:|
| EP8 | 3.770 | 11.202 |
| EP16 | 9.362 | 24.209 |

Dispatch 与 Return 仍重叠，不将同一段时间称为两个独立单向带宽。
`--report-communication` 在计时外按实际路由统计 node 去重 Dispatch rows、
远端 contribution rows 和 off-rank expert rows，用于计算实际 payload。

按每个计时样本计算有效 payload / 全阶段设备时间，再取 p50：

| 口径 | 8K GB/s | 20K GB/s |
|---|---:|---:|
| EP8，off-rank Dispatch + Return，每 GPU 平均 | 185.3 | 159.7 |
| EP16，RDMA Dispatch + Return 累计发送，每 NIC 平均 | 28.46 | 27.57 |

第一行把 off-rank expert rows 的输入和输出各计一次；第二行 Dispatch
按目的节点去重，Return 按实际专家 contribution 计数，包含每行 4B dGate。
这是全流程均摊的有效载荷速率，不是网卡瞬时峰值，也不是分别隔离出的
Dispatch / Return 带宽。不能由它直接断言链路硬件只有这些带宽。

单机/双机均通过 129/8K/20K、每规模两次变化路由的 identity 检查：
`dX == BF16(top_k * FP32(dY))` 逐位相等，dGate 全零。
此模式的 dW 没有计算，不能拿它报告训练性能或梯度正确性。

## 待确认候选

- dW wait1：纯 GEMM 无稳定收益，完整周期回退，淘汰。
- activation K128 / 两级流水：单机 20K 初筛有所改善，多机没有收益；
  仍需结合数据流候选确认，不替换默认值。
- dW 静态 chunk 轮转：对比原按 FLOPs 切连续区间的分配方式，检查
  同时活跃专家数量与输入复用的影响，不引入共享领用队列。
- 单机 activation expert wave / 小 wave：不改 N tile 顺序，仅缩短
  dU→SwiGLU→dX 的数据存活范围；全局 dX 完成后才执行 dW 的约束不变。

## 第一轮数据流优化（完整周期，ms）

| 候选 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| 原基线 | 25.048 | 62.929 | 32.838 | 77.460 |
| dW 每 cluster 静态轮转 1 个 N-pair chunk | 23.081 | 58.387 | 30.912 | 73.234 |
| dW 每 cluster 轮转 4 个 chunk | 23.532 | 60.589 | 31.142 | 74.574 |
| 仅单机 activation wave 限为 128 M-block | 24.767 | 57.987 | — | — |
| 轮转 1 + 单机 wave128 | 23.071 | 53.871 | 同轮转 1 | 同轮转 1 |
| 上行再加 dU epilogue 融合 SwiGLU | 22.797 | 53.933 | 30.249 | 72.592 |
| 轮转 1 + wave128 + K128/两级流水 | 22.980 | 54.092 | — | — |

轮转分配仍是静态所有权，无共享任务队列；不改变任何 GEMM 的 K 归约
顺序。小 wave 缩短中间数据存活范围，并非缩小专家的完整 dW reduction。
按专家边界截 wave 的候选为 28.955 / 61.231 ms，8K 明显回退，淘汰。
轮转 1/4 均通过单/双机 129/8K/20K 的完整梯度校验；wave128 通过单机
同样校验。上述收益不简单相加，例如融合 SwiGLU 与 wave128 在 20K 上
重叠，组合后没有继续改善。

融合版保留原 BF16 dU 精度边界，再从寄存器内的 BF16 dU 计算 SwiGLU。
首版网络 math 出现 spill，压紧 dU 的寄存器存活期后主 kernel 恢复零 spill。
修正版 `packed_verify_ep8/16` 均通过 129/8K/20K、每规模两次变动路由的
完整梯度校验。当前本地 dY 与 dAB ring 仍复用地址，因此必须等全部 dU
N-tile 消费完 dY 后，再把 canonical dAB 写入 ring，不能提前覆盖。

以上是单轮候选，未作为最终达标结果；仍需反序复测和默认实现确认。

## CTA 服务时间诊断

`WEFTMOE_PF_BWD_TASK_TIMING` 只用于诊断：累计每个 CTA 的任务服务时间，
除以 132 个 CTA，再逐步取 rank-max、最后取 p50。它包括 TMA 等待及
epilogue，不是纯 WGMMA 时间，也不等于关键路径；各列不可相加。
`task_probe_reset_ep8/16` 在每次 BWD prepare 明确清零计数，避免 FWD
复用状态字段带来的残留。

| 平均 CTA 服务/等待（ms） | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| dU GEMM + epilogue | 2.404 | 6.181 | 3.354 | 7.122 |
| dX GEMM + epilogue | 3.861 | 13.157 | 4.789 | 13.319 |
| SwiGLU + dAB gather | 1.787 | 4.499 | 1.896 | 5.083 |
| producer 等待 dU 输入 ready | 0.930 | 2.489 | 1.569 | 2.007 |
| producer 等待 dX 输入 ready | 0.235 | 0.284 | 1.242 | 3.286 |

下一轮针对静态 dW 仍经过任务邮箱/cluster join，以及 SwiGLU/gather 的
向量化与重复 GM 读取做消融。缓存局部性目前是由排程消融支持的解释，
还没有采集 L2 miss 硬件计数器，不将其写成已测量的因果结论。
