# PF Backward 对齐现版 Forward（2026-10-12）

基线是 `main@1cfb423`；实现分支 `feature/pf-backward-20261012`。
旧静态 BWD 保留为迁移期间的正确性/性能对照，不将新 FWD 的收益外推到 BWD。

## M6：原生库仅保留 GDAKI 后端

按部署硬件约束，在项目统一的 `communication/nccl_device.cuh` 入口关闭
device 编译中的 Proxy/GPI/EFA GDA，覆盖 FWD、BWD、notify 和 NCCL 内部
barrier 调用；所有 GIN device communicator 的初始化显式请求
`NCCL_GIN_TYPE_GDAKI`，不再自动回退到 Proxy。单机 notify 仍不申请 GIN。
不修改系统 NCCL；独立通信参考 benchmark 不属于原生库的裁剪范围。

宏只作用于 NVCC device pass。host pass 保留 SDK 默认定义，因为该阶段
NCCL 默认关闭 GDAKI，若再关闭其余后端，非模板接口的 `auto` 返回类型
无法推导。host 头文件解析不等于生成或使用 GPU Proxy 分支。
保留原有 assertion 策略、GEMM 参数、缓冲容量、静态任务和同步协议。

`bwd_gdaki_only_final` 中 return helper 从 8/8 B spill 降为 0/0 B；
静态 SASS 指令数从 4243 降至 2383。裁剪也改变编译器寄存器分配：
多机 BWD 主 kernel 从 0/0 B 变为 20/44 B spill store/load，位置是
调度索引及阶段收尾的标量状态，不是 GEMM accumulator。单机主 kernel
仍为 0/0 B；没有 C7507/C7512/C7518。不能将这次修改称为整个算子零 spill，
也不能从删掉未执行分支直接推断性能收益。

双机 EP16 Flash、B2048/D2，沿用 M5 的变化长度/热点路由和
4 warmups / 16 samples、完整 prime trace。旧/新/新/旧各跑两轮，
逐 rank workload hash 一致；两轮 p50 均值如下（ms）：

| 口径 / 版本 | 8K | 20K |
|---|---:|---:|
| 完整 FWD+BWD，旧版 | 33.082 | 77.605 |
| 完整 FWD+BWD，GDAKI-only | 33.170 | 77.811 |
| BWD host API 边界，旧版 | 20.576 | 49.097 |
| BWD host API 边界，GDAKI-only | 20.585 | 49.123 |

完整周期差异约 +0.27%，BWD 约 +0.04%/+0.05%；视为基本持平，不宣称
加速，也没有由这些数据确认明显性能回退。BWD host 边界仍不是独立 GPU
kernel event 时间。两机均设置 SM 1620 MHz，计时内旧/新各观测到 3 个
低于目标的 GPU 频率点（合计范围 1545–1590 MHz），保留所有样本。
计时内采到的 HBM 均为 3201 MHz；部分短区间没有采样点。
一次新版本启动因端口占用失败，未进入 GPU 计时，保留日志并换端口重跑。

单机 EP8 / 双机 EP16 均通过 129/8K/20K 各两轮变化路由的完整数值验证，
最大 relative L2 分别为 0.000838 / 0.000734；双机还通过 8K/20K
无缓存推理、kernel-entry 检查及两次调用的 CUDA Graph 回放验证。
前后 native 工作区容量一致；现有 Python 回归 117 tests / 470 subtests
通过，没有增加单元测试。

编译、SASS 归因及回归原始结果位于 `build/pf_backward_gdaki_20261012/`
和 `build/pf_main_timeline/bwd_gdaki_only_final/`。

## M5：当前 GEMM 参数的 autotune

原生源码保持 `e4db718`，固定当前 FWD、通信协议、M128 canonical 行、
184/136 寄存器配额和 wave128。新增
`configs/pf_backward_stream_autotune.json`，复用训练 autotuner，搜索
dU/dX 各自的 N、共用的 activation stages，以及 dW1/dW2 共用的
M/N/stages。不混入会切回旧 kernel 的 activation K128，也不搜索
已经不作用于新 stream 的旧 return knobs。

测试为双机 EP16、Flash H4096/I2048/E256/top6、8K/20K，B2048/D2，
正态长度波动 10%、路由偏置 std=0.5、seed=123；请求时钟 1620 MHz。
每个候选完整 prime trace，再 4 warmups / 16 samples；比 M3/M4 的
8 samples 更多，因此只能与本轮同 trace 基线比较。完整周期包含
每调用重排、缓存分配/释放和 device synchronize，不计首次固定池准备。

初筛完整 FWD+BWD wall p50，单位 ms：

| 候选（仅列相对基线的变化） | 8K | 20K |
|---|---:|---:|
| 基线：dU/dX/dW M128/N256/K64，三级流水 | 33.101 | 77.767 |
| dW M256/N128 | 33.504 | 78.258 |
| dW M128/N128 | 37.465 | 88.181 |
| dW 两级流水 | 34.735 | 80.887 |
| dU N128 | 33.760 | 79.416 |
| dX N128 | 34.490 | 81.231 |
| dU/dX 都用 N128 | 34.974 | 82.926 |
| dU/dX 两级流水 | 34.405 | 79.250 |
| dU/dX 四级流水 | 33.587 | 78.819 |

另一个候选 dW M128/N256 四级流水因 shared memory 超限，在编译前过滤。
没有 C7507/C7512/C7518；dW M256/N128 和 dW 两级流水分别产生主 kernel
40/68 B、40/228 B 的静态 spill store/load，其他候选主 kernel 均为零。
不能据此给 spill 单独归因，因为它们同时改变了 tile 或流水深度。

小 tile 没有提高当前 384-thread/168 初始寄存器配额下的驻留 CTA 数。
仅计单个 tile 的输入、不计输出或跨 tile 复用时，BF16 GEMM 的算术强度为
`M*N/(M+N)` FLOPs/byte；M128/N256 为 85.3，M128/N128 为 64。
这给出了小 N 降低数据复用的一个解释，但不是实际 HBM 流量测量或
完整的瓶颈证明。M256/N128 理论强度相同，仍需用实测决定。

正反顺序各复测一次，两个 rank-max p50 的均值：

| 入围配置 | 8K / ms | 20K / ms |
|---|---:|---:|
| 原参数 | 33.096 | 77.722 |
| dW M256/N128 | 33.244 | 77.635 |

原参数的等权几何平均分仍最好，选择结果为 `incumbent_fastest`；
不是把不足 1% 的差异包装成收益。最终保留所有原生 GEMM 默认值，
没有修改 FWD/DT 的参数或算法。两个入围配置均通过 129-token 两轮
数值验证，保留版还通过 8K/20K 各两轮验证（output/dX/dr/dW1/dW2）。
这是本轮有限搜索空间的负结果，不表示四个 GEMM 已经全局最优；
dW1/dW2 仍共享几何/stages，本轮没有独立调参，也未改变单机参数。
单机性能没有在本轮独立搜索，不能将这组双机结论外推为 EP8 最优。

保留版最后一次完整周期为 **32.934 / 77.736 ms**，BWD host API 边界
为 **20.682 / 49.019 ms**，后者不是独立的 GPU kernel event 时间。
最终 8K/20K 验证最大 relative L2 为 0.000110；本轮现有 Python 回归
117 tests / 470 subtests 通过，没有增加单元测试。

时钟记录覆盖两节点，但不是连续逐 kernel 采样。确认轮的 20K 计时内，
原参数观测到 5 个 1455–1545 MHz 点，挑战者观测到 1 个 1560 MHz 点；
最后一次 20K sample 1 观测到 3 个 1530–1590 MHz 点。没有删除这些样本，
不能称全程严格恒频，也不能认定两候选约 0.17% 的综合差异有统计意义。
计时内采到的 HBM 频率均为 3201 MHz；部分短测量区间没有时钟采样点。
因此结论是“没有确认收益，保留原参数”，不是“已证明原参数全局最优”。

原始计划、编译、逐 rank 计时/验证日志和自动选择结果在
`build/pf_backward_tune_20261012/network/`；各候选共享同一 trace hash。
`table.json` 已导出保留参数的两个容量表项，`clocks.json` 保留全部时钟点。
紧凑结果见 `benchmarks/pf_backward_20261012.json` 的 `gemm_autotune_m5`。
通用命令和当前参数边界见 [autotune 说明](autotune.md)。

## Return helper 的剩余 spill（后续核查）

`return_canonical` 是 BWD main 内的 device helper，不是额外的 kernel。
它由 48 个静态 return owner 的借用通信 warp 的 lane0 执行：
等一个 M-block 的全部 dX N tiles 完成，把相邻 source/contribution 行
合并为 fragment；远端行通过两次 GIN Put 分别发送 BF16 dX 和 FP32
router gradient，并增加两个远端完成计数。每个 fragment 的本地 NIC
flush/wait 完成后，整个 block 才能发布 TX returned generation。
本机行已由 dX epilogue 直接写回，不走这里的跨机 Put。
这些 owner 与 dW math 并行；慢 return 仍可能通过 TX credit 反压计算，
所以不能笼统地说“通信路径 spill 不影响性能”。

对 `bwd_ready_final` 的进一步 SASS 检查，报告的 8/8 B 对应
`STL.64 [R1+0x90]` / `LDL.LU.64 [R1+0x90]`，位于
NCCL `gin/proxy/gin_proxy.h::postGfd` 的通用地址 atomic 分支。
`gin_backend_current` 双机运行日志再次确认使用 `GIN_IB_GDAKI`，
该次执行不进入 proxy backend，因而不执行这对 spill 指令。
这比“只有 8 B，影响应当很小”更准确；不能把静态报告当成当前热路径的
动态 local-memory 流量。换 backend 或编译超参后仍需重新检查。
原始日志在 `build/pf_backward_tune_20261012/backend/`，SASS 在
`build/pf_backward_20261012/ready_final_spills/`。

## M4：按预计到达顺序排静态任务

M3 消除了逐窗口 padding，但 expert-major 的第一个 canonical M-block
可能跨多个 RX 窗口，导致 math 等待尚未到达的行。M4 在 main 前按
`(该块最后需要的物理窗口, canonical block ID)` 排列任务；仍是静态分配，
没有运行时领任务或抢占队列。main 及 return 使用相同排列的 position
决定 TX slot/generation，cache 地址和 dW 仍使用原 canonical block ID。
因此不改变 saved-row 顺序、GEMM 的 K 顺序和 dW 归约顺序。

排序用 GPU merge pass，复用不再使用的旧 phase/DU-counter 数组，
不增加持久空间或临时分配；多次小 kernel 的开销包含在完整周期中。
这是静态的到达顺序估计，并不保证每块实际最早 ready。

双机 Flash、相同随机 trace、1620 MHz 请求、完整 FWD+BWD wall p50：

| 候选 | 8K / ms | 20K / ms |
|---|---:|---:|
| 本轮修改前 `b151d93` | 34.563 | 86.551 |
| M3 `b5d1cc4` | 33.238 | 81.522 |
| M4 首测 `ready_order_perf` | 32.836 | 80.201 |
| M4 复测 `ready_order_repeat` | 33.224 | 80.450 |
| M4 最终构建 `ready_final_perf` | 32.855 | 80.808 |

M4 两次 BWD host API 边界 p50 分别为 20.394/50.456、20.439/50.733 ms；
M3 为 21.113/51.841 ms。BWD 有一致改善，但完整周期的 8K 增益较小，
不能只引用最快一轮。首测有一次 1485 MHz 观测落在 20K sample 2；
复测四次 1500–1590 MHz 观测均不在计时区间。所有样本保留，
两轮 workload hash 与 M3 完全一致。

最终构建 BWD host p50 为 20.503/51.030 ms；三次 1560–1605 MHz
观测均在计时区间之外。相对本轮前，完整周期下降 4.9%/6.6%，
BWD host 边界下降 8.6%/10.0%。相对旧 PF 的本轮复测 33.502/80.921 ms，
最终完整周期仅快 1.9%/0.1%；20K 仍应称基本持平，不能宣称大幅领先。

单机最终构建回归也保留全部结果，而不是只引用 M3 较快的一轮：

| EP8 完整周期 | 8K / ms | 20K / ms |
|---|---:|---:|
| 本轮前 `b151d93` | 25.236 | 64.490 |
| M3 较早 `canonical_local_perf` | 25.213 | 62.569 |
| M4 `ready_final_local` | 25.443 | 63.426 |
| M4 `ready_final_local_repeat` | 25.191 | 67.370 |
| M3 同场对照 `ready_local_control` | 24.963 | 64.660 |

最后两轮 M4 的 BWD host p50 为 17.127/44.014、17.138/48.078 ms；
M3 同场为 16.968/45.113 ms。三轮 trace hash 一致，M4 首轮一次
1590 MHz 观测在计时外，第二轮 SM 频率观测全为 1620 MHz；
M3 对照一次 1560 MHz 观测落在 20K sample 0。计时内未观测到 HBM
频率下降，测试间隙也未发现其他 GPU 进程，但采样不能排除瞬时干扰。
单机 BWD 的 SASS 指令流在 M3/M4 **逐条相同**（含被调用函数 9368 条，
归一化指令 SHA256 `c4af9d4119c1e67638ffd838777f20eef754e446ff8e1cf328e9ecc693def397`）。
M4 排序仅用于多机；这些结果不足以把单机的差异归因于排序，也不足以
确认本轮单机 20K 相对 `b151d93` 的稳定收益。波动来源尚未完全定位。

`wave128_phases` 与 `ready_order_phases` 的独立诊断（warmup=1、sample=2）
显示，最终 dW 完成最晚 rank 的首次 dQ 开始时间，相对 main 入口：

| 规模 / sample | 原静态顺序 / ms | 到达顺序 / ms |
|---|---:|---:|
| 8K / 0 | 0.798 | 0.022 |
| 8K / 1 | 1.101 | 0.038 |
| 20K / 0 | 0.827 | 0.040 |
| 20K / 1 | 1.278 | 0.159 |

这些是 GPU 阶段时间，不含 main 前的排序；诊断 trace 不等同于性能表
的八个样本，不能把各 rank 的阶段最大值相加。return/reduce 仍与 dW
折叠，没有放宽所有本地 dX 必须先于任何 dW 的约束。

M4 编译器报告：单机/多机 BWD 主 kernel 均为 0/0 B spill store/load，
pull/Get/receive 也为零；return helper 为 8/8 B。这不代表整个程序
没有 local memory 访问，也不代表整个 FWD/BWD 都零 spill。
没有 C7507/C7512/C7518，math 分支没有 accumulator store-to-local。

保留构建为 `bwd_ready_final`（相对 `bwd_canonical_ready_order` 仅格式整理）。
数值检查覆盖双机 129/8K/20K、每规模两轮变化路由、512-token 窗口；
另测 8K/20K 的 bias=8 热点，普通/热点两组最大 relative L2
分别为 0.000734 / 0.000275。
最终构建 `ready_final_lifetime` 通过三份 live tape、变化长度/热点、
权重更新、workspace 释放重建、两层 graph/offload。
现有 Python 回归 117 tests / 470 subtests 通过，未增加细粒度单元测试。

## M3：消除窗口 padding 和修复寄存器分工

本轮改动的简要说明（详细消融和最终测量见本节末尾）：

- 多机 dQ/dX 按完整专家行切 M128，不再按每个通信窗口分别补齐。
  同一随机 trace 的额外计算行数 / 有效行数，8K 从 23.19% 降到 2.09%，
  20K 从 22.17% 降到 0.81%。剩余仅为各专家最后一个 M tile 的对齐。
- Dispatch 直接填充 dW2 本来就需要的 compact dY cache；dX 直接读完整
  dAB cache。RX 的 fanout credit 与 GEMM 解耦，不增加一份完整激活缓冲。
- dX 按 canonical M-block 写已有注册 TX，目的为源 rank 的完整 contribution。
  不再绕到同 rail GPU 的旧物理窗口 TX；连续 source/contribution 行合并 Put。
  本地 NIC completion 保护 TX 复用，源 contribution 在调用结束前不复用。
- 内联跨 math/producer/comm 的阶段收尾 helper，让 `setmaxnreg` 真正生效。
  单机静态 spill store/load 从 72/92 B 降至 0/0 B；完整周期
  25.236/64.490 ms → 25.071/61.155 ms（8K/20K）。
- 多机重新分配 math/control 寄存器，并将 compute wave 从整个 ring
  缩小到 128 个 M-block；提前开始 dX/return，不改变 RX/TX 容量。
- BWD 直接使用当前 tape 保存的 route ordinals 和 packet/fanout 表，省去
  重复生成；仍重建共享工作区的物理计划和当前通信 epoch。

### 布局与同步

FWD 的行缓存仍为 expert/source/route 顺序。BWD 的物理 RX 目录也改用
稳定 source 顺序，但只用于读取/退休通信 packet，不再决定 GEMM M-padding。
每次 BWD 从当前 tape 重建 canonical M-block 目录和 source/contribution
反向映射，分别复用旧 scheduler descriptor 和闲置 route-header 区域。
不能复用前次 FWD 的物理目录；后续层可能已覆盖它。

本地 pull 把 dY 写入 compact cache，release 增加对应 canonical M-block
的 arrived rows；producer acquire 达到 valid_rows 才发出该块 TMA。
dQ 的 N tiles 完成后，两份 SwiGLU 任务生成 Q、dAB、dr，再发布 dX ready。
return owner 等所有 dX N tiles 完成后发送；对本块全部 fragment 完成
本地 NIC flush/wait 才发布 TX returned generation。下代 SwiGLU/dr 和
dX 都等待这一 credit，因此不会在 NIC 仍读取时覆盖 TX。

return 不需要远端 RX-slot credit：它写的是源 rank 已有的整调用 contribution，
不是短 RX ring。每个源 rank 按远端节点累计期望 route 行数；各 expert rank
用两个独立 GIN weak signal 累加 dX/dr 完成行数，源端 acquire 两个计数后
才能 reduce。最终仍有全 rank 终止 barrier，防止下一次调用覆盖 contribution。
RX Get/fanout ring、TX ring、compute slot 仍是有界容量；没有 full dX 缓冲。
两个 signal 和一个累计行数字段填入原结构的 padding，结构大小不变。

### spill 的原因与不能混淆的指标

原来的跨角色 `__noinline__` 阶段 helper 令 ptxas 报 C7507：
`setmaxnreg` 被忽略，math 和 control 实际都受初始 168 registers 限制。
仅修复内联又会暴露另一个问题：单机合适的 208/88 配额不够多机 GIN
控制流使用，控制路径出现大量 spill。因此不能直接沿用单机配额。

384-thread CTA 初始池是 `384 * 168 = 64512`，不是 65536。
192/128 超过实际池，试验中在 setmaxnreg 等待而无法前进，已撤回；
192/120 和 184/136 才满足池约束。184/136 下同步 return 版本的
pull/Get/return/receive 编译器 spill 均为零，主 kernel 仍有少量 spill。
SASS 中大量 LDL/STL 来自 GIN/SDK 的调用栈和 local-address atomic 分支，
不能把它们全部算作 WGMMA accumulator spill；主 kernel 残留项包含线程编号
和控制状态的 reload，仍要检查是否落在 K-loop，并以实际耗时决定取舍。

### 逐步消融

以下均为 EP16、Flash H4096/I2048/E256/top6、2048-token window、相同
变化长度/变化路由 trace，4 warmups / 8 samples；单位 ms，完整 FWD+BWD。

| 候选 | 8K | 20K | 说明 |
|---|---:|---:|---|
| 本轮修改前 `b151d93` | 34.563 | 86.551 | 旧物理窗口切 GEMM |
| 只修复内联 | 35.659 | 86.799 | 208/88 引入 GIN spill，不单独保留 |
| canonical M-block + 192/120 | 33.575 | 83.742 | 消除重复 padding |
| 184/136，整 ring compute wave | 35.641 | 82.933 | spill 更少不等于更快 |
| 184/136，wave=128 | 33.385 | 81.893 | 提前启动 dX/return |
| 再去掉重复 tape 表生成 | 33.016 | 81.375 | 多份 live tape 回归通过 |
| return 每 owner 4 个在途 fragment | 33.973 | 81.782 | 无收益，撤回；增加了 spill |
| dQ/dX 保留一个在途 WGMMA group | 33.540 | 81.856 | 数值通过，无收益，撤回 |

没有筛除慢样本。前三个 canonical 候选和 tape 候选各存在少量低于请求
1620 MHz 的观测，其中部分落在 20K 计时区间，不能解读百分之一的差异。
return 流水候选频率采样全部 1620 MHz；其结果仍不支持保留此改动。
将整个通信收尾拆成 noinline 函数还触发 ptxas C7512/C7600
（184 寄存器下 WGMMA 分配失败），也未保留。
在途 WGMMA group 方案将 empty-stage release 延后一轮，最终 drain
剩下的一组；没有 C7512/C7518，EP16 的 129/8K/20K 数值检查通过，
但单机也测得 25.651/63.857 ms，不优于原 K-loop。因此保留逐轮 wait<0>。

构建日志、SASS 和逐 rank 原始测量分别在
`build/pf_main_timeline/bwd_canonical_*`、
`build/pf_backward_20261012/*_spills/` 和
`build/pf_backward_20261012/results/`。

### M3 阶段复测（M4 之前）

最终 artifact 为 `bwd_canonical_final`：184/136、wave=128、同步 fragment
return、原 wait<0> K-loop。相同 trace 的完整 FWD+BWD wall p50：

| 拓扑 / tokens | 旧 PF `main@1cfb423` | 本轮前 `b151d93` | 本轮后 | 对本轮前延迟下降 |
|---|---:|---:|---:|---:|
| 单机 EP8 / 8K | 26.483 | 25.236 | 25.213 | 0.1% |
| 单机 EP8 / 20K | 68.624 | 64.490 | 62.569 | 3.0% |
| 双机 EP16 / 8K | 33.502 | 34.563 | 33.238 | 3.8% |
| 双机 EP16 / 20K | 80.921 | 86.551 | 81.522 | 5.8% |

旧 PF 单机行沿用本轮任务早期基线；双机旧 PF 为此次重新运行的
`m3_old_repeat`。旧 PF 不是 V3。所有 workload hash 与对应旧 trace 一致。
最终单机频率观测全为 1620 MHz；最终双机仅在计时外观测到一次 1590 MHz。
旧 PF 双机复测有一次 1500 MHz 落在 20K sample 6，所有样本原样保留。
双机相较旧 PF 8K 快 0.8%、20K 慢 0.7%，只称基本持平，不宣称全面领先。

按 host API 边界拆分的 BWD p50（不是纯 kernel 时间）：

| 拓扑 | 本轮前 8K / 20K | 本轮后 8K / 20K |
|---|---:|---:|
| EP8 | 17.261 / 44.717 | 17.103 / 42.971 |
| EP16 | 22.429 / 56.678 | 21.113 / 51.841 |

padding 降幅不能直接当成完整周期收益：此次改变的是 dQ/dX 的 M 行数，
dW1/dW2 原本已经按完整专家行计算，且 FWD 未变。以有效 GEMM FLOPs 计，
dQ/dX 只占 BWD 的一半、FWD+BWD 的三分之一，此外还有通信和准备开销。

保留版编译：单机 BWD 0/0 B spill；多机主 kernel 12/224 B spill
store/load，pull/Get/return/receive 均为 0/0 B。没有 C7507、C7512、C7518。
这些字节是编译器静态统计，不是每调用或每 M-tile 的实际流量。
反汇编 math 分支无 accumulator store-to-local；少量标量 reload 尚未消除。

验证覆盖：EP8 的 1/129/8K/20K、EP16 的 129/8K/20K，每规模两轮变化
路由；EP16 8K/20K 额外 bias=8、512-token 窗口回绕，包含空专家。
output/dX/dr/dW1/dW2 全部通过原容差；热点最大 relative L2 为 0.000275。
保留 tape 优化通过三份同时存活 tape、权重更新、workspace 释放重建、
两层 inference graph/offload 检查。PF 无缓存推理的 EP16 8K/20K 数值和
kernel-entry 回归通过。现有 Python 回归 117 tests / 470 subtests 通过，
没有新增单元测试或更改误差阈值。DT 代码未改，此轮未重跑其性能。

后面的 M1/M2 章节保留当时的实现和测量过程；当前多机协议以本节及
前述 M4 静态任务排列为准。

## 设计与里程碑

- [x] M1 单机：复用 PF destination-pull、输入 ring 和完整 source
  contribution；producer/math 独立重放静态 dQ/SwiGLU/dX 任务流。
  dX epilogue 从 accumulator 直接 128-bit 写回。所有本地 dX 完成后
  math 开始静态 dW，通信 warp 进行 source reduce。
- [x] M2 多机协议：复用当前借用 warp 的按节点去重、逐 token Get、机内 pull、
  直接 TX contribution/Put return。路由和计数来自 FWD tape，不再重新
  counts notify。RX fanout、TX NIC completion 和远端 credit 分开退休。
- [x] M3 消除通信窗口引入的 dQ/dX padding，修复寄存器分工，缩小 compute
  wave，消除重复 tape 表生成；覆盖随机 tokens/routes、热点/空专家和
  slot 回绕。FWD 和 DT 不改变数学流程。dW 参数的进一步搜索不在本次结论内。

数学记号：A/B 为 FC1 gate/up，r 为 router 权重，Q=r*SiLU(A)*B。
dQ=dY*W2^T；SwiGLU backward 同时生成 dAB、dr 和供 dW2 使用的 Q。
dX=dAB*W1^T；dW2=Q^T*dY；dW1=X^T*dAB。权重方向以数学表示为准，
公共 API 保持原有 `[expert, out, in]`。dr 返回选中 route 的梯度，上层
框架继续处理 router softmax/归一化及 router 线性层。

## 单机第一阶段

输入 pull 和 reduce 复用 `local_kernel.cuh`，通过编译期参数选择读退休
条件与线程域，Forward 默认实例不变。独立 dQ/dX N-tile 任务不经过
旧 BWD 的共享 task mailbox；两个相邻 CTA 沿 N 共享 multicast A。
SwiGLU 由一对 CTA 分别处理半个 M-block，保留旧实现的 BF16 舍入和
行内 dr 归约，不引入浮点 atomic。

保存的 X/A/B 和路由属于 FWD tape；物理 ring 行经已有 saved-row 映射
写入专家连续的 dW cache。首版先复用现有 Q 工作区：dQ 写入后由
SwiGLU 原地覆盖成 Q，dAB 同时写完整 cache 和供 dX 使用的有界 ring。
不增加单独的全尺寸 dQ 或 dX 临时张量。

当 `2I <= H` 时，dQ 消费完的 dY ring 原地作为 dAB ring，物理行 stride
仍为 H。下次 pull 必须等本代所有 dX 读者完成，不能沿用 Forward 的
FC1-done credit。`2I > H` 才申请独立 dAB ring。已有 Forward contribution
直接用于 dX；不在 BWD peer workspace 再申请一份无用 contribution。

持久 peer workspace 目前仍保存 dY cache 和 IPC 可见 dr 标量；容量增长
与地址交换沿用现有策略。单机初始化使用 FWD tape 的专家目录，只生成
dW 所需 padded-row 映射、清空本调用计数器和空专家梯度，不再初始化
旧 activation scheduler 或重新拷贝 route headers。

## 测量口径与初始数据

固定 1620 MHz，Flash H4096/I2048/E256/top-6，EP8 为单机；EP16 为双机。
每 rank tokens 正态波动 10%，routes/gates 每步变化，共享专家 bias=0.5。
固定 seed=123，先完整 prime trace，再 4 warmups / 8 samples。
表中为同一步最慢 rank 的完整 FWD+BWD wall latency 的 p50，包含每调用
权重重排、缓存申请及回收，排除输入构造与首次通信池创建。

| 对照 | 8K (ms) | 20K (ms) |
|---|---:|---:|
| 旧 PF BWD，单机 EP8 | 26.483 | 68.624 |
| Megatron 缓存模式，单机 EP8 | 24.792 | 59.548 |
| 新 PF BWD，单机 EP8（最终候选） | 25.209 | 62.164 |
| 新 PF BWD，单机 EP8（复测） | 25.193 | 62.178 |
| 旧 PF BWD，双机 EP16 | 32.668 | 78.531 |
| Megatron 缓存模式，双机 EP16 | 33.288 | 87.202 |

双机旧 PF 基线检测到测量区间内 1455/1470 MHz 短暂降频，因此这一行
仅作为原始观测，正式同频比较需要重跑，不筛掉慢样本重算。
最终单机候选也检测到一次 1590 MHz（20K sample 4）；复测期间有一次
1485 MHz。设置为 1620 MHz 不代表实际运行始终维持该频率，以上均
保留原始样本，不能将百分之一量级的差别解释为稳定收益。

原始 manifest、workload hash、逐 rank 时间与频率采样保存在
`build/pf_backward_20261012/results/`。以上是本轮短对照，不替代历史长测。
脚本的 `backward_slowest_rank_p50_us` 是 host API 边界差值，不应称为
纯 BWD kernel 时间；单独设备阶段时间应另行测量。

初始单机原型通过 EP8、129/8K/20K 的 output/dX/dr/dW1/dW2 检查。
初版编译出现 setmaxnreg 忽略和 96/76 B 静态 spill store/load，正在
分离寄存器域和移除 dW-only 模板中的不可达 activation 分支。正式收益
以最后保留代码的无插桩测量为准，不将编译候选当作已完成版本。

初始任务流每个 expert 分别排 dQ/SwiGLU/dX。完整周期 8K/20K 为
29.112/63.671 ms：8K 比旧版慢，20K 较快。把 dY/dAB ring 复用并移除
冗余 contribution 后为 29.069/63.955 ms，没有稳定速度收益；该改动
主要减少容量。后者在未计时阶段观察到 1530 MHz，已保留频率记录。
反汇编表明已见的 spill 主要位于调度与阶段切换，不在 GEMM K 热循环；
增大 control register quota 仍不能恢复 setmaxnreg，不据此声称优化成功。

已保留的主要性能优化是将相邻专家合并为一个有界 compute wave：dQ 的输出列数
只有 Forward FC1 的一半，逐专家切阶段可能让短 expert 的 dQ 任务不够
填满 CTA。合并不改变 M padding、归约顺序或路由，只扩大阶段内可用
的静态 N-tile 任务集合，仍由 ring depth 限定窗口。
该项单独测量为 25.116/60.884 ms，频率采样均为 1620 MHz。
随后简化初始化并增加阶段时钟，四级 operand pipeline 测得
24.956/64.017 ms；三级最终候选为表中的 25.209/62.164 ms，保留三级。
最终版相对旧 PF 完整周期约快 4.8%/9.4%，但没有追平本轮 Megatron。
尚有 ptxas setmaxnreg 忽略与阶段切换 spill，未宣称已消除。

最终候选通过 EP8、1/129/8K/20K、每规模两次变化路由的 Torch 检查，
包括 output/dX/dr/dW1/dW2；全部相对 L2 误差最大约 0.00142。
四级前一候选另外通过 bias=8 的热点/空专家及 512-token window 回绕。
Python 回归为 117 tests / 470 subtests。没有新增碎片化单元测试。

## 多机实施约束

FWD tape 的 immutable routes/counts 可复用，但物理计划属于共享工作区，
可能已被后续层覆盖。必须重新规划并按当前 channel base 绑定 RX/TX，
不能复用 FWD 已退休的 packet generation。
dr 使用独立小型 FP32 TX/contribution 平面与 GIN 完成信号；dX 和 dr
分别 acquire 成功后才返还 packet credit，TX 仍保留本地 NIC completion。
通信共享内存必须与 dW arena 分离，否则为了复用 arena 而等待所有
Return 结束，会让 dW 与 Return 串行。

M2 已实现。`backward_kernel.cuh` 共享单机/多机 dQ/SwiGLU/dX/dW 数学流程；
`backward_network.cuh` 只负责窗口任务流、slot 计数单位、dr 地址和终止协议。
源 contribution、compute block 与 NIC packet generation 均单调前进。
每次 BWD 重新绑定 RX/TX/本地 contribution 地址，但不重新交换路由计数、
不重新分配通信池。结束时将共享 compute counters 归一到 FWD 的 N-tile
单位，下一次 FWD 可继续使用同一套 slot。

dr 为每 route 一个 FP32，新增 registered TX/contribution 小平面，不改
原有 BF16 dX/FWD payload 的行 stride。每包 dX、dr 各自 Put+weak signal；
接收方 acquire 两个信号后才 acknowledge。独立 flush request 保护本地
NIC 对两份 TX 的读取。FWD 不发 dr Put，也不推进 dr signal。

通信 staging 与 dW arena 分开；384-thread CTA 中仍为 256 math、32 TMA
producer、96 communication。保留 64 Get / 48 Put / 19 progress / 1 admit
的借用 warp 分配。compute-only grid completion 保证全部本地 dX 完成
后才执行 dW；通信继续 return/reduce，与 dW 并行，不加入该 compute join。

双机验证：129/8K/20K 各三轮变化路由通过完整 Torch 校验；另外，首个
协议候选通过 8K/20K bias=8 的热点/空专家、512-token window 回绕。
流程按节点数参数化，目前硬件只验证两节点、每节点 8 GPU。

性能仍待优化：相同 2048-token window，首版完整周期为
33.712/85.982 ms，慢于旧 PF。仅将 window 放大到 4096，得到
32.271/80.742 ms；此时实际路由的 M128 padding 仍为约 14.2%/10.7%。
放大窗口同时增大持久 ring 容量，因此不将它悄悄改为默认配置。

## 后续消融与诊断

同一组随机 workload、相同 2048-token window：

| 实验 | 8K (ms) | 20K (ms) | 决策 |
|---|---:|---:|---|
| M2 单调 generation 原版 | 33.712 | 85.982 | 保留 |
| 尾块跳过空的第二个 math WG | 34.815 | 87.720 | 撤回 |
| 尾块两个 WG 各算一半 N，均计算有效的前 64 行 | 34.034 | 86.287 | 撤回 |

第一个尾块方案出现 ptxas C7518（WGMMA 序列化）。第二个通过数值验证，
也没有 C7518，但网络 kernel 的静态 spill store/load 从 76/132 B
增至 124/216 B，没有测到速度收益。不能将降低 padding FLOPs 等同于
降低耗时。尝试提前执行 control reg-dealloc 也没有消除 C7507
（setmaxnreg 被忽略），同样撤回，不留运行时开关或死分支。

8192-token window 测得 31.396/77.158 ms；M128 padding 从 2048 window
的 23.19%/22.17% 降至 9.99%/5.43%。但 compute+GIN 池从
5650/8517 MiB 增至 13269/16135 MiB，约增加 **7.4 GiB/卡**。
4096 window 也增加约 2.35 GiB/卡。这些是容量/性能消融，不更改默认值；
窗口同时影响 FWD，不能把完整周期收益全部归给 BWD。

`m3_phase_profile` 使用 GPU 时间戳：各规模三次调用、16 ranks 中，
main 入口到首次 dQ 的最大观测约 46/47 us，dQ 开始到该 rank 全部
dX 完成最长约 15.84/34.80 ms，dW 阶段最长约 10.90/20.72 ms，
reduce 最长约 1.13/2.94 ms。不同 GPU 时钟不互相相减，阶段最大值
也不能相加当作总耗时（可能来自不同 rank/step，且阶段存在重叠）。
该组带诊断、包含初次分配，不能替代上面的无插桩性能结果。
目前瓶颈不是 main 到首次 GEMM 的启动空隙，优先看窗口内激活计算、
padding 与 dW。快 rank 的 dX 到 reduce 间隔还包括等待其它 rank，
不能解释成纯 RDMA 耗时。

同口径 cached FWD 单独回归：旧版 12.180/29.735 ms，新版
12.328/29.680 ms，没有发现明显退化；单机完整周期回归
25.165/62.174 ms。DT EP2/4/8、E256、1/64 tokens 的数值与 graph replay
检查通过；没有修改 DT 数学流程。

### 多 tape 生命周期修正

上述初始性能数据来自 FWD 紧接 BWD 的单 tape 周期，不能视为完整训练
生命周期验收。`m3_lifetime_network` 在三份不同热点、不同长度的 FWD
tape 同时存活时发现错误：Notify handle 保留了路由、计数和检查点，
但物理 ring plan 是共享执行工作区，不会随每份 tape 保存。
新 BWD 直接使用最后一次 FWD 的物理计划，因此早期 tape 的反向错误。

修正为每次 BWD 复用现有 PF planner，从该 tape 的 routes/counts 重建
物理计划；`training_counts_ready=true` 跳过 counts notify，不申请另一份
通信池或逐层保存大计划。dY staging 合入 planner，移除原来的独立
D2D 拷贝；网络 BWD 不再自己重复实现 RX/TX 地址重绑定。准备工作只
留下 canonical dW 行目录、padding 清零与 BWD slot 计数单位转换。
**修正后的性能须重新测量，不能沿用上表当最终结果。**

`m3_tape_network` / `m3_tape_local_accum` 已通过 EP16/EP8 的三份同时
存活 tape、热点迁移、短批次、权重原地更新、通信池释放和两层推理 graph
切换检查（H4096/I2048/E32/top-6，129 tokens）。沿用现有集成测试。
单机权重更新测试还暴露参考实现的累加口径差异：一项接近零的 dX
误差为 2.098e-5，略超 2e-5 绝对阈值。参考原先在 BF16 内累加 output
及不同 expert 的 dX，现改为 FP32 累加后转回 BF16；各 GEMM 输入、
中间结果仍为 BF16，**没有放宽容差**。与性能测试已有 FP32 combine
参考的口径一致。只改 output 的中间候选不足以修复 dX 的累加差异，
也保留了该失败日志。

修复版 `bwd_tape_plan` 通过 EP16、129/8K/20K、每规模两次变化路由，
512-token window 的完整数值检查，全部输出/梯度相对 L2 最大约
0.000734；8K/20K 最大约 0.000111/0.000105。Python 回归仍为
117 tests / 470 subtests。

收尾同步：return/reduce 可以在 dW 期间执行，但保护下一次调用复用
metadata 的最后一道 rank/network barrier 移到完整 CTA/cluster 的 dW
结束之后。不增加 barrier 数量，也不把 return/reduce 与 dW 串行化。

## 生命周期修复后的最终测量

代码 `27ba2b5`，artifact `build/pf_main_timeline/bwd_tape_plan`。
以下均包含每次 BWD 的物理计划重建，仍采用开头的完整周期口径、
2048-token window、4 warmups / 8 samples；新旧 workload hash 相同。

| 实现 | 单机 EP8 8K / 20K (ms) | 双机 EP16 8K / 20K (ms) |
|---|---:|---:|
| 旧 PF BWD | 26.483 / 68.624 | 33.514 / 81.035（复测） |
| 新 PF BWD | 25.236 / 64.490 | 34.563 / 86.551 |
| Megatron moe_act 缓存口径 | 24.792 / 59.548 | 33.288 / 87.202 |

单机比旧 PF 快 4.7%/6.0%；相同 2048 窗口的双机仍慢 3.1%/6.8%。Megatron 行是
本轮前段测得的相同 workload，不是新做一轮框架训练，也不是 V3。
此处旧 PF 指本分支起点 `main@1cfb423` 的 BWD，**未重新测 V3**。
不能宣称多机性能优化目标已经完成。

修复版另外测了 4096-token window：**33.733/81.761 ms**，比新版
2048 window 快 2.4%/5.5%。它接近旧 PF 的 2048 window 数字，但没有
复测旧 PF 的 4096 window，不能宣称同配置已追平。该版本比本轮
Megatron 8K 慢约 1.3%、20K 快约 6.2%；这些短测不代表框架端到端。
相对新版 2048 window 仍多占约 2.35 GiB/卡，持久池总量仍低于旧 PF
2048 window。此轮没有修改缺省容量：TensorRuntime 本来就是单机
2048、双机 4096；表内 2048 是 benchmark 显式选择的对照值，而非
双机 API 缺省值。所有这些测量均显式设 `WEFTMOE_PF_MIN_RING_DEPTH=2`
和 `WEFTMOE_PF_CREDIT_WINDOWS=2`。

设置时钟为 1620 MHz；新双机在 20K timed sample 0 观察到一次 1530 MHz，
旧双机复测有一次 1515 MHz、但不落在测量区间内。单机新版本有一次
1590 MHz，同样不在测量区间内。保留全部样本，不按频率删除样本。
频率只是间隔采样，不等于证明间隔内无瞬时降频。
4096 window 在 20K timed sample 5 观察到一次 1605 MHz，另有一次
1530 MHz 位于测量区间外；同样未删样本。

相同 2048 window，双机 rank 0 的持久 compute+GIN 容量：旧版为
8518.7/12133.2 MiB，新版为 5650.1/8516.6 MiB，约减少 2.8/3.5 GiB。
这不是总峰值显存（不包括当次的 Torch tape/dW/scratch）。主要来源是
复用 FWD contribution/dY-dAB ring，以及不再保留完整 dX 临时数据。

最终回归：EP8 的 1/129/8K/20K 与 EP16 的 129/8K/20K，各两次变化
路由，完整 output/dX/dr/dW1/dW2 检查通过；EP8/EP16 的 8K/20K
纯推理通过数值及 PF kernel 路径检查。多 tape、训练到推理共卡切换、
DT EP2/4/8 与 Python 回归见上文。未测三节点以上，也未在完整 AReaL
或 Megatron 模型中跑训练 step，不将算子级结果等同于端到端训练收益。

仍待完成的性能工作：在不增加全尺寸激活暂存的前提下，减少多机
window/expert 的 M padding 与阶段尾部空转；进一步消融 dQ/dX 的
N tile 与 operand pipeline；解决 setmaxnreg 被忽略及阶段切换 spill。
这些只是后续方向，尚未证明有效。没有保留失败的尾块分支，也没有
改动 DT 或推理的缺省调参。
