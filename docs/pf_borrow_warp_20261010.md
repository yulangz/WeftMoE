# PF：每 CTA 借用一个 pull warp（2026-10-10）

结论：借用版显著减少静态 spill，但完整 GEMM 与全 CTA WG3 基本持平：
8K 12.191 vs 12.015ms（慢 1.5%），20K 24.837 vs 24.975ms（快 0.6%）。
纯通信 8K 略退、20K 略好。实现可行，但不能把它报告成明显的端到端加速，
也没有证据说明 spill 是当前的主瓶颈。保留候选，暂不替换默认配置。

本轮把[单机少一个 pull warp 的结论](pf_pull_slots_resources_20261009.md)
落实为双机候选：384 线程、2 个机内 pull warp、1 个网络 warp，保留 math208。
对照是全 CTA WG3 的 512 线程、192/64/64 配额版本。
实验保存在 `experiment/pf-borrow-warp-20261010`，生产默认配置未修改。

## 实现与比较边界

| 每 CTA | 借用方案 | 全 CTA WG3 |
|---|---:|---:|
| math 线程 / 寄存器配额 | 256 / 208 | 256 / 192 |
| TMA producer warp | 1 | 1 |
| 机内 pull warp | 2 | 3 |
| 额外网络 warpgroup | 无 | 1 |
| 全 grid Get / Put / progress / admit warp | 64 / 48 / 19 / 1 | 132 / 132 / 132 / 1 |
| GIN contexts | 131 | 396 |
| control / producer 寄存器配额 | 80 或 88 | 64 |
| 每 math WG、K64 的 WGMMA | 4×N256 | 8×N128 |
| shared memory，含 1 KiB static | 222 KiB | 227 KiB |

132 个 CTA 各借用原 pull warp0；Get、Put、progress、admit 使用静态 CTA
编号区间，不引入动态任务队列。余下两个 pull warp 使用新的静态 residue/stride，
总计 264 个机内 pull worker。网络 pending requests 复用被借走的 warp 的
4 KiB shared staging，不与两个活跃 pull warp 共用数据区。

网络协议保持 token Get dispatch、packet Put return；不额外 pack。
同样使用 132 lanes、D32/R128、pending4、B4096、credit windows2、4 slots/channel、
planner publication 和 Return-ready 去重。本轮把 publication metadata 的准备
从仅 WG3 推广到两种 worker 布局，避免借用版缺少对应元数据。
保留远端 RX credit 和本地 NIC completion 的双重复用保护。

88 而非 96 的原因：384-thread entry 的初始寄存器配额是 168，CTA 总池为
`384×168=64512`，不是可任意使用全部 65536。
`256×208+128×88=64512` 已用完；不能在不降低 math 配额的情况下给 control96。

80/88 是同一角色划分内的寄存器配额消融。借用版与 WG3 的比较则同时包含
网络并发度、pull 并发度、指令形态和寄存器配额变化，不能把全部性能差异归因于 spill。

## 编译资源

CUDA12.9，`-lineinfo -Xptxas=-v,--warn-on-spills`。
以下为 ptxas **静态 spill store/load 字节统计**，不是每 packet 的实际访问字节。

| 模式 | kernel 入口 | Get | Put | 累计 stack B |
|---|---:|---:|---:|---:|
| WG3 copy | 68 / 48 | 420 / 844 | 308 / 600 | 424 |
| 借用 copy80 | 12 / 12 | 256 / 484 | 144 / 328 | 344 |
| 借用 copy88 | 12 / 12 | 204 / 316 | 116 / 236 | 320 |
| WG3 real GEMM | 144 / 140 | 560 / 1016 | 476 / 804 | 456 |
| 借用 real GEMM88 | 56 / 48 | 252 / 384 | 136 / 276 | 336 |

借用版没有消除所有 spill，但 Get/Put 控制路径明显减少。真实 GEMM 的热 K 循环
没有 LDL/STL，四条 N256 WGMMA 后一次 wait_group0；不能说 accumulator 被 spill，
也不能说每条 WGMMA 被等待串行化。完整控制路径仍有 stack 访问。
GIN 同时编译的其他 backend、泛型 atomic 的 local-address fallback 不等于当前
运行的 spill 流量；不能用所有静态 LDL/STL 指令数估算性能损失。

所有四个新 library 的单机 local kernel normalized SASS hash 完全一致，
未修改单机计算路径。WG3 real-GEMM normalized SASS hash 与上一轮编译审计版一致。

## 测量口径

双机 EP16、每节点 8×L20X、SM1620MHz，BF16 H4096/I2048/E256/topK6。
每步 tokens 在 8192/20480 附近正态波动 10%，路由每步重新生成，bias0.5、seed123。
权重预处理、固定 workspace 分配、GIN contexts 初始化/释放不计入 kernel 时间。
真实 GEMM 包含 FC1/SwiGLU/FC2/通信/最终 combine，不保存训练中间结果。

纯通信使用 copy-substitution 诊断，必须显式传 `--communication-copy`；benchmark
同时核对 native audit 的模式，防止把 copy kernel 误报为真实 MoE 性能。
带宽按每步实际 Get/Put payload 与最慢 rank 的通信阶段耗时计算，再取 p50，
报告每卡、单方向有效 GB/s，上限 50GB/s。真实 GEMM 不按此阶段反推纯通信带宽。

正式运行采用整轮 prime32、warm8、measure24；每步取最慢 rank，再取 p50。
全部编译结束后才启动计时，GPU 测试串行。

## 纯通信结果

以下为本轮同环境各一次正式运行的 p50，单轮探索，不把约 1% 的差异视为稳定收益。

| 布局 | 8K GB/s | 20K GB/s | 8K 诊断 kernel ms | 20K 诊断 kernel ms |
|---|---:|---:|---:|---:|
| WG3 192/64/64 | 34.384 | 38.512 | 9.062 | 19.407 |
| 借用 208/80 | 32.747 | 40.301 | 9.666 | 18.665 |
| 借用 208/88 | 32.692 | 40.220 | 9.600 | 18.776 |

借用 88 相对 WG3：8K 带宽下降 4.9%，20K 上升 4.4%；完整诊断 kernel
分别慢 5.9%、快 3.3%。不是所有规模都优于 WG3。
同一借用布局的 80→88 显著减少静态 spill，但带宽只变化 −0.17%/−0.20%，
本轮未观察到明确性能收益。不能仅凭 spill 字节减少推断通信加速。

## 真实 GEMM 结果

顺序为借用、WG3、WG3、借用。下表为两轮独立 p50 的均值，不含 copy-substitution。

| 布局 | 8K kernel ms | 20K kernel ms | 8K eager ms | 20K eager ms |
|---|---:|---:|---:|---:|
| WG3 192/64/64 | 12.015 | 24.975 | 12.374 | 25.258 |
| 借用 208/88 | 12.191 | 24.837 | 12.529 | 25.085 |
| 借用版 kernel 耗时变化 | +1.47% | −0.55% | — | — |

逐轮 kernel p50：借用版 12.207/12.176、24.829/24.846ms；
WG3 12.053/11.976、24.992/24.959ms。8K 两轮都略慢，20K 两轮都略快，
但总体差异很小，不包装成性能突破。

相同参数下 Get/Put 的静态 spill-load 统计下降约 62%/66%，并没有获得相应幅度
的端到端收益。原因不能只从静态编译统计推导：本轮还减少了网络/pull 并发度，
且部分访问可能被缓存或计算覆盖，部分静态路径在当前 GDAKI backend 不执行。
没有采集动态 stall/访存指标，因此不对这些因素分配具体的耗时占比。
可以确定的是：热 K 循环没有 accumulator spill，80→88 的单变量配额测试
也未见明显收益，所以不应把后续优化全部押在继续增加寄存器上。

借用方案的优点是 384 而非 512 线程、保留 math208/N256、少 5KiB shared，
131 而非 396 个 GIN contexts；代价是网络 worker 从 397 减到 132。
后续更有针对性的实验是 Get/Put/progress 配比、每个 worker 的 pending 窗口：
当前两版都用 pending4，但借用 worker 需要轮询更多 channel，聚合 in-flight
请求上限并不相同。这些尚未消融，不能声称当前 64/48/19 已是最佳划分。

七轮正式运行的完整 32 步 token/route/gate workload 逐 rank 完全一致，
24 个计时样本的实际 Get/Return 字节均与路由预期匹配。
918 次正式节点时钟观测全部 1620MHz。连同五轮正确性验证，12 次启动均两机 exit0；
结束后两机全部 GPU 0MiB，无遗留测试进程。

## 正确性

copy80、copy88、真实 GEMM88、真实 GEMM WG3 各验证 8K/20K×2 步×16 rank；
GEMM88 另测 B128、bias8 的强偏斜路由，各 3 步×16 rank，共 352 个 rank/sample。
两种 copy 配额的输出与参考逐位一致；两种真实 GEMM 正常路由输出也逐位相同。
真实 GEMM 最大 relative-L2 9.22e-5、max-abs 6.11e-5，所有检查通过。
验证期间 526 次节点时钟观测全部 1620MHz。

## 复现

实验目录 `../WeftMoE-experiments/borrow-warp-20261010`，代码提交 `9123f30`
（之前的两个提交为 `0c90117`、`ef2bc0a`）。配置目录
`configs/pf_network_l132_20261009`，四个 artifact 在 `build/borrow/` 下：
`copy80`、`copy88`、`gemm88`、`wg3gemm`。
旧 WG3 copy 对照直接使用上一轮 `wg3-slots-20261009/build/slots/diag` 的
匹配 library/adapter，不混用 native ABI。

编译增加 `WEFTMOE_PF_BORROW_ALL_CTA=1`、`WEFTMOE_PF_NETWORK_WORKERS=1`，
用 `WEFTMOE_PF_BORROW_CONTROL_REGISTERS=80/88` 切换配额；
`WEFTMOE_PF_COMMUNICATION_COPY=1/0` 区分诊断/真实 GEMM。
WG3 对照使用 `NETWORK_WORKERS=2`、`WG3_ALL_CTA=1`、192/64/64。
每次 launch 的 manifest 保存完整命令、环境、source commit、library/adapter hash；
clocks 文件记录两节点时钟，原始 node0/node1 日志保留在实验目录。
manifest 的 source commit 是启动侧源码；冻结结果的 `artifact_source_commit`
另行注明旧 WG3 copy 对照来自 `85cfc54`，其余来自 `9123f30`。

- [性能、workload、正确性、launch/hash/clock 记录](../benchmarks/pf_borrow_warp_20261010.json)。
- [编译 flags、ptxas/SASS/热 K 循环审计](../benchmarks/pf_borrow_warp_resources_20261010.json)。

测试前曾修正两处实验接线：native audit 的赋值必须在 handle 创建之后；
publication metadata 不能继续只为 512-thread planner 准备。前者在编译阶段发现，
后者在 GPU 测试前的代码检查发现。上述修正已包含在所有新 artifact 中，没有将
失败编译或旧诊断模式的数据混入真实 GEMM 表格。
