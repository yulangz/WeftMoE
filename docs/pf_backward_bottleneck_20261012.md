# PF BWD：通信、GEMM 与数据流拆解（2026-10-12）

## 保留结果速览

保留实现为 `041391e`，FWD/DT 算法及参数不变，没有增加持久显存。
本轮优化的是 BWD，完整训练周期延迟相对本轮起点下降：单机 8K/20K
**15.5% / 19.4%**，双机 **13.3% / 12.1%**。

- dW 改为静态小 chunk 轮转，dW1/dW2 分别采用 4/1；改善热点专家下
  dW2 任务不足的问题。纯热点 dW2 在单机 8K 从 464 提高到 614 TFLOPS。
- activation 使用单机 wave128、网络 wave64；限制中间数据存活范围，
  同时保留足够的可调度任务。更小的 wave32/16 反而回退。
- SwiGLU 使用 128-bit 访存，同一寄存器值直接写 canonical dAB 和本地
  dAB ring，省去一次 GM 重读。单机组合初测从 23.071/53.871 降至
  21.906/51.496 ms；收益与 wave、分块有重叠，不能逐项相加。
- 本地逐 tile system fence 合并到 activation 阶段边界；网络仍保留
  逐 tile 发布与 NIC-read credit。组合初测 21.458/51.777 →
  21.131/50.634 ms。严格保持先完成全部本地 dX、再执行 dW。

正反序各一轮、每轮 16 个样本的完整 FWD+BWD 对照，表中为两轮 p50 的
算术均值（ms）；“领先”是吞吐比 `Megatron / PF - 1`：

| 拓扑 / 名义 tokens | PF | Megatron | PF 吞吐领先 | 目标 |
|---|---:|---:|---:|---:|
| 单机 EP8 / 8K | 21.157 | 24.826 | 17.3% | 25%，未达 |
| 单机 EP8 / 20K | 50.708 | 57.839 | 14.1% | 25%，未达 |
| 双机 EP16 / 8K | 28.462 | 33.735 | 18.5% | 20%，未达 |
| 双机 EP16 / 20K | 68.056 | 81.932 | 20.4% | 达到 |

**没有达成全部目标。** 以本次 Megatron 复测为准，前三项还需分别减少
约 1.296 / 4.437 / 0.350 ms；双机 20K 余量只有 0.221 ms，不能视为
对所有负载稳定领先 20% 的保证。下文保留所有重要消融，包括回退项。

主要证据指向热点任务粒度与 activation 数据流衔接，而不是简单的
“RDMA 带宽不够”或“所有纯 GEMM 都慢”。均衡纯 GEMM 距 cuBLAS 约
4%–6%，真实热点下旧 dW2 排程明显较差；本轮大部分完整周期收益来自
数据流调整，通信替身耗时基本不变。未采集 L2 miss/SM 利用率硬件计数器，
不把缓存命中率或 SM 空转比例写成已测量的结论。

基线：`feature/pf-backward-20261012@8e11d73`，原生 artifact 为
`build/pf_main_timeline/bwd_spill_r192`（与该提交的原生实现一致）。
实验原始日志、配置、时钟位于 `build/pf_bottleneck_20261012/`。

目标按吞吐加速比定义：完整训练周期单机至少 Megatron 的 1.25 倍，
双机至少 1.20 倍，即延迟分别不高于基线的 80% / 83.33%。
下面依次保留基线、诊断和优化过程；不将纯算术或通信替身的收益视为目标达成。

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
dW2 效率损失，不能笼统说“四个 GEMM 都没有问题”。该组全部 30 次计时
前后 NVML 采样为 1620 MHz，原始数据在 `gemm_ragged/results`。

同一真实行数重放 1-pair chunk，反转 native/cuBLAS 交替顺序：

| EP / tokens | dW1 TFLOPS：PF / cuBLAS | dW2 TFLOPS：PF / cuBLAS |
|---|---:|---:|
| EP8 / 8K | 616.2 / 642.2 | 614.3 / 634.2 |
| EP8 / 20K | 658.7 / 706.6 | 669.4 / 703.4 |
| EP16 / 8K | 675.9 / 701.7 | 679.5 / 694.4 |
| EP16 / 20K | 688.2 / 736.7 | 711.2 / 733.3 |

1-pair 明显改善 dW2，但 dW1 比 4-pair 下降。Flash 每专家 dW2 只有
128 个 N-pair，4-pair 分块只产生 32 个任务；dW1 有 256 个 N-pair，
同样分块产生 64 个任务，更接近 66 个 cluster 的全网格。
这支持分别选择 dW1/dW2 分块，而非将均衡 GEMM 的最优值直接套入热点
生产数据流。结果在 `gemm_ragged_c1/results`，完整输出及逐位重复检查通过。

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

## 第一轮前的候选（实验过程记录）

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

## 第二轮：向量化、stage 衔接和生命周期

下表仍为完整周期、4 warmups / 16 samples；“基座”是 dW 轮转 1 +
单机 wave128，网络 wave128。各候选保持相同 workload hash。

| 相对基座的变化 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| dAB gather 每线程先读 4 个 uint4 再写 | 23.061 | 53.643 | — | — |
| 小 K 专家改为 4-pair chunk（阈值 1024） | 22.654 | 54.199 | — | — |
| dW 独立 producer/math 静态遍历，移除逐任务邮箱 | 22.499 | 54.592 | 30.307 | 73.707 |
| 本地 dX system fence 合并到阶段末尾 | 22.842 | 54.064 | 不变 | 不变 |
| SwiGLU 128-bit 读写 | 22.327 | 51.768 | — | — |
| 上行再直接从寄存器写本地 dAB ring | 21.906 | 51.496 | 29.371 | 71.571 |
| 上行再将 local/network wave 都改成 64 | 21.873 | 51.719 | 28.751 | 68.279 |
| 仅 wave64，不加向量化 | 22.836 | 53.493 | 29.656 | 70.573 |
| 仅 activation wait1 | 23.414 | 54.667 | 30.288 | 73.424 |

向量化每条指令每线程读/写 16B，SwiGLU 的逐元素公式和 BF16 精度边界
不变。dGate 的 lane 内归约排列改变，但同一行的归约次序固定。
直接写 ring 在全部 dU N-tile 完成后进行，因此不会覆盖仍被读取的 dY；
canonical dAB 缓存仍保留给 dW1。原本的 GM 写→读→写变成同一寄存器值
分别写两处，没有新增缓冲。

wave128 向量化组合的设备插桩测量，activation envelope 为
7.211 / 18.163 ms（原基线 8.313 / 25.893），dW 为
5.916 / 13.573 ms（原基线 7.880 / 17.483）。这些是诊断运行而不是
表中的无插桩完整周期。dW 邮箱删除、wait1、额外 gather batching 暂未
确认跨规模稳定收益，不直接启用。

正确性：向量化 + 直接写 ring、activation wait1 都通过 EP8/EP16 的
129/8K/20K 两轮完整梯度检查。向量化组合还通过现有
`torchrun_tensor_runtime.py` 的多 tape 移动热点、权重原地更新、训练
workspace 释放及 inference graph 销毁/offload 检查，无新增细粒度单测。

另试“直接读旧执行 slot 的 row directory”的本地 pull 简化，已撤销：
即使紧邻的 FWD→BWD 微测通过，后续层也会覆盖共享执行目录，不能作为
真实训练优化。新的 canonical-local 候选只读当前 tape 独立保存的
headers，通过了完整梯度和上述多 tape 生命周期检查；其初筛为
21.859 / 51.633 ms，wave64 为 21.948 / 51.430 ms，尚无明显收益。
继续消融其源端拉取顺序及与融合 SwiGLU 的组合，而不是仅凭结构简洁保留。

## 第三轮：窗口和两个 dW 分别分块

同样的随机 trace、4 warmups / 16 samples，完整周期 ms：

| 候选 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| canonical-local，交错源端 pull | 21.882 | 51.001 | — | — |
| 上行加融合 SwiGLU | 22.088 | 52.032 | — | — |
| 向量化基座，wave32 | 22.776 | 53.412 | 29.062 | 69.180 |
| 向量化基座，wave16 | 24.601 | 58.367 | 31.512 | 74.649 |
| dW1/dW2 claim4/2，local128/network64 | 21.038 | 51.697 | 28.315 | 68.111 |
| claim4/1，其余相同 | 21.458 | 51.777 | 28.318 | 67.868 |
| claim2/2，其余相同 | 21.369 | 51.328 | 28.387 | 68.137 |

缩小 wave 到 32/16 不再有益：缩短数据存活范围的同时，也缩小了可调度的
计算集合，不能无限缩窗。canonical-local 收益很小，融合后反而回退；
两个版本都通过单机 129/8K/20K 完整梯度检查，但不据此保留额外数据流。
分别配置两个 dW 改善 8K；20K 的分块差异很小，须反序确认。

此时最好初测仍未覆盖全部目标。接着消融 dW 的寄存器向量写出：它省下
TMA store 的 shared tile，允许四级 operand 流水。这是资源置换实验，
不预设 vector store 一定比 TMA store 快，也不将编译成功当作性能收益。

## 第四轮：dW shared memory 与本地写回同步

基座为 claim4/1 + SwiGLU 128-bit + 寄存器直写 dAB ring，local128 /
network64。完整周期 ms：

| 改动 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| 基座，dW TMA store / 三级 | 21.458 | 51.777 | 28.318 | 67.868 |
| dW 寄存器 128-bit store / 三级 | 21.645 | 51.242 | 28.230 | 67.856 |
| dW 寄存器 128-bit store / 四级 | 21.443 | 51.185 | 28.627 | 68.759 |
| 本地 system fence 移到 activation 阶段边界 | 21.131 | 50.634 | 不变 | 不变 |

三个 dW store/stages 候选主 kernel、pull/return helper 都没有 spill。
四级再加 wait1 出现 C7511/C7517（主 kernel），没有作为保留候选。
两种 vector-store 候选都通过单/双机 129/8K/20K 两轮完整梯度检查。

纯 GEMM 的四级 vector store（同样 C1、相同热点 rows）确有小幅提高：

| EP / tokens | dW1 TFLOPS | dW2 TFLOPS |
|---|---:|---:|
| EP8 / 8K | 631.5 | 628.6 |
| EP8 / 20K | 690.5 | 692.9 |
| EP16 / 8K | 692.9 | 689.0 |
| EP16 / 20K | 715.2 | 731.1 |

但完整周期没有稳定收益，尤其双机回退，最终仍保留 TMA store / 三级。
这也说明纯 GEMM 改善不能直接外推到含通信、activation 的持续主 kernel。
原始数据在 `gemm_dwvec_s4`，完整输出和逐位重复校验通过。

本地 fence 合并不改变 RX/TX credit：本地 contribution 不在逐 tile
发布后立刻 reduce；所有写者先执行一次 system fence，再由 grid/rank
barrier 发布阶段完成。SwiGLU→dX 的本地缓存依赖仍有 device fence。
网络仍逐 tile 完成 system fence，保护 NIC 读取与有界 TX 复用。

实验源码保存在 `experiment/pf-bottleneck-20261012@cd84d9e`。
保留版删除 canonical-local、融合 SwiGLU、wait1、额外 gather batching、
独立 dW mailbox/pipeline 和 dW vector-store 分支；不新增持久 buffer。
dW 分块复用已有 `dw1_claim_tiles` / `dw2_claim_tiles` 调参项，并加入
2-pair 候选；移除已无作用的 `dw_partition_rounds` 调参项。

## 保留版正反序复测与独立 FWD

artifact 为 `build/pf_main_timeline/bottleneck_final`，不启用诊断宏。
`confirm_ep{8,16}_{pf,megatron}_r{0,1}`：r0 先 Megatron 后 PF，r1
顺序相反。每轮都重新 prime 完整 trace，下面记录各轮 p50（ms），不取
两轮最好值。全部 workload SHA256 逐 rank 相同，Megatron 测量期新编译
graph 数均为零。

| 实现 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| PF r0 | 21.019136 | 51.079370 | 28.607525 | 67.745886 |
| PF r1 | 21.294906 | 50.335710 | 28.317163 | 68.365746 |
| Megatron r0 | 24.913160 | 57.718422 | 33.925416 | 81.703469 |
| Megatron r1 | 24.739819 | 57.958649 | 33.544561 | 82.160888 |

中途端口 36402 冲突，尚未进入该次计时；已完成的 EP8 r0 保留，其余
改用检查过的 23000 段端口续跑，没有按快慢删除样本。

独立带缓存 FWD 使用 `--forward-only`，取
`forward_ready_slowest_rank_p50_us`，包括完成同步，tape 尚未释放。
这不是 host API 返回时刻，也不能用它从另一轮完整周期 p50 相减得到
独立 BWD 延迟。标签 `forward_ready_plain_ep{8,16}_{pf,megatron}`：

| FWD ready（ms） | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| PF | 8.127 | 18.596 | 12.376 | 28.543 |
| Megatron | 10.178 | 22.866 | 14.470 | 33.223 |
| PF 吞吐领先 | 25.2% | 23.0% | 16.9% | 16.4% |

### 实际时钟边界

所有测试均请求两机 1620 MHz；原始 1 秒间隔 NVML 日志完整保留。
不能声称整段计时都严格保持 1620：确认轮计时区间内，EP8 PF 20K 的
r0/r1 分别观察到 2/8、1/8 个 GPU 采样点低于 1620，最低 1515/1410；
EP16 PF 20K r1 为 1/8、最低 1455，Megatron 20K r1 为 2/16、最低
1545 MHz。其他有覆盖的区间未观测到降低，但部分短区间没有采到点。
这些样本没有过滤，结果是相同锁频请求下的实际运行表现，而非严格等频
硬件测量。纯 GEMM 的逐样本前后读数均为 1620，同样不代表连续频率轨迹。

## 保留版设备阶段与通信复查

`final_probe_ep8/16` 仅用于诊断，不替代无插桩确认结果。阶段定义同前文，
尤其不能相加、不能把 return_tail 全部当成网络传输时间：

| 阶段（ms） | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| main 到首次 dU | 0.021 | 0.022 | 0.936 | 0.967 |
| Dispatch envelope | 3.035 | 8.592 | 3.823 | 10.062 |
| dU/SwiGLU/dX envelope | 6.908 | 17.519 | 8.706 | 21.514 |
| dW1/dW2 envelope | 5.436 | 13.254 | 6.097 | 15.976 |
| Reduce | 0.600 | 1.682 | 0.867 | 2.440 |
| Return tail | 3.046 | 8.154 | 5.916 | 13.600 |
| main 工作 envelope | 12.389 | 30.727 | 14.817 | 37.570 |

平均 CTA 服务/等待的同口径复查（ms，累计值除以 132 CTA 后取 rank-max
p50，不是关键路径、不可相加）：

| 服务 / 等待 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| dU GEMM + epilogue | 1.907 | 4.387 | 2.924 | 6.128 |
| dX GEMM + epilogue | 3.681 | 9.822 | 4.533 | 11.859 |
| SwiGLU | 0.908 | 2.321 | 1.041 | 2.672 |
| producer 等待 dU ready | 0.286 | 0.335 | 1.447 | 2.229 |
| producer 等待 dX ready | 0.506 | 1.201 | 1.063 | 2.691 |

`final_identity_ep8/16` 通信替身复查，仍使用实际路由 payload，不重复
计作两个单向带宽：

| 口径 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| 通信 main envelope（ms） | 3.764 | 10.954 | 9.351 | 24.259 |
| 有效 GB/s，每 GPU / NIC | 185.80 | 160.85 | 28.34 | 27.64 |

和初始替身基本一致，本轮不是靠提高裸通信带宽取得完整周期收益。
网络 producer 的输入等待以及约 0.94–0.97 ms 的首次 GEMM 启动间隔仍在。
现有 source window 按节点分组、从目的端本节点开始，不应误判成“所有
目的端都先等待节点 0”，简单交换 local/remote pull 顺序没有依据。
完整 canonical M-block 的到达条件与真实计算波次如何耦合仍值得继续查，
目前没有足够证据把剩余间隔精确归因于某一个 barrier。

## 正确性与资源检查

- 保留版 EP8/EP16：129/8K/20K，每规模两轮新路由，完整校验输出、dX、
  dGate、dW1、dW2。所有项最大 relative L2 分别为 0.000839/0.000734。
- 极端偏置 `route_bias_std=8`：129/8K、两轮，最大 relative L2 分别
  为 0.000148/0.000513；没有依靠丢 token 或专家容量截断通过测试。
- 两机拓扑均通过现有 multi-tape 移动热点、权重原地更新、workspace
  释放、多层 graph 与 graph 销毁/offload 检查。
- 最终 identity 替身两机拓扑均通过 129/8K/20K 两轮精确 dX/dGate 校验。
- 现有 Python 测试 117 passed / 470 subtests，host C++ 17/17。
- main 和 pull/return/receive helper 均 0 spill，没有 C75 警告。
  原有 training mini-notify 仍有 28B store / 28B load spill，不能称为
  “整个库所有 kernel 都零 spill”。
- 持久池容量没有随本轮优化增加。该 profile 每卡 EP8 compute/notify
  为 4,054,878,104 / 33,386,952 B；EP16 compute/GIN/notify 为
  3,842,583,384 / 2,213,011,712 / 34,435,744 B。

确认结果、逐 rank workload hash、二进制 SHA256 和计时区间频率采样摘要
另保存于 [机器可读结果](../benchmarks/results/pf_backward_bottleneck_20261012.json)。

验证日志标签：`final_verify_*`、`final_skew_*`、`final_multitape_*`、
`final_identity_verify_*`。这些是具体覆盖范围，不代表已验证任意模型与
任意节点数；本轮可用硬件仍只有两机。

## 最后一组：TMA operand 的 L2 eviction hint（不保留）

只改变 TMA load 的 cache hint，不改变 tile 顺序、归约、同步和缓冲。
完整周期初筛如下（ms），每项仍为 4 warmups / 16 samples、相同随机
trace；它不是 L2 tile-order swizzle，也没有采集硬件 cache hit 指标：

| 变化 | EP8 8K | EP8 20K | EP16 8K | EP16 20K |
|---|---:|---:|---:|---:|
| 仅 BWD 权重 EVICT_LAST | 21.326 | 50.856 | 28.363 | 67.914 |
| 仅 FWD 权重 EVICT_LAST | 21.294 | 50.626 | 28.385 | 68.628 |
| FWD/BWD 权重都 EVICT_LAST | 21.243 | 50.546 | 28.306 | 67.633 |
| BWD 权重 LAST、dY/dAB 输入 FIRST | 21.230 | 50.652 | 28.821 | 68.090 |

差异不足以确认稳定收益：最好双机组合相对保留版两轮均值改善约 0.6%，
小于保留版自身两轮波动；单机 8K 均未改善。没有为这一初筛结果增加
生产配置，也没有把未单独完成梯度验证的候选当成保留版。
源码归档于 `experiment/pf-cache-hints-20261012@92ff50e`，主分支删除
这些额外 hint 宏，native 源码恢复为已完成全部正确性检查的 `041391e`。
标签为 `cache_{bwd,fwd,both,bwd_stream}_ep{8,16}`；所有候选的主 kernel
与通信 helper 仍无 spill，只有原有 mini-notify 的 28/28 B。
