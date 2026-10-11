# PF：从两个 pull warp 减到一个（2026-10-11）

单机 EP8 交叉复测：2→1 个 pull warp 后，8K 基本持平（+0.21%），
20K 慢约 **1.83%**，不是完全无损。因此没有修改单机默认配置。
下一步单独检查双机新增 RDMA 并发能否抵消 pull 减少的损失。

## 单机：只改变实际 pull 工作者数量

基于确认保留的 `8c47a46`，复用之前 `30c6eda` 的 compile-time pull warp
参数；本轮源码 `62605ca`，分支 `experiment/pf-warp-budget-20261011`。
保持 384 threads、132 CTA、Math208/Control80、222KiB shared、三个
4KiB staging、GEMM 参数和同步协议不变。只修改 pull 的静态行 residue、
stride 和参加 pull 的 warp 边界；闲置 warp 仍参加 kernel 的收尾同步。
每卡活跃 pull worker 从 264 降到 132，没有同时收缩 shared 或改变 occupancy。

单节点 8 GPU，H4096/I2048/E256/topK6/BF16，SM1620MHz，B2048；
每步 tokens 在 8192/20480 附近正态波动 10%，route bias std=0.5，seed123。
真实 inference，不保存训练缓存。固定工作区/IPC、权重准备不计入调用耗时，
每次路由规划仍执行。配置 `pf_inference_flash_20261008`；两份库全部编译
完成后再串行测量，prime32/warm8/sample24，先每步取最慢 rank，再取 p50。

顺序为 2→1→1→2，表中是两轮独立 p50 的均值：

| pull warp/CTA | 8K kernel ms | 20K kernel ms | 8K eager ms | 20K eager ms |
|---|---:|---:|---:|---:|
| 2 | 6.682968 | 16.703792 | 6.895324 | 16.911742 |
| 1 | 6.697280 | 17.009424 | 6.928636 | 17.234368 |
| kernel 耗时变化 | +0.21% | **+1.83%** | — | — |

逐轮 kernel：2-warp 为 6.699968/16.642624、6.665968/16.764960ms；
1-warp 为 6.669664/17.036672、6.724896/16.982176ms。
8K 的差异小于轮间漂移；20K 的两个 1-warp 样本均慢于两个 2-warp 样本，
不将它写成无损。历史 3→2 的结果不构成 2→1 同样无损的保证。

四轮 workload hash 与 workspace bytes 相同。两版各正常路由 8K/20K×3 步，
1-warp 另做 B128、bias8 强偏斜路由 8K/20K×3 步，共 144 个 rank/sample
校验通过，最大 relative L2 9.353e-5；两版正常验证输出逐位相同。
196 次节点时钟观测均为 1620MHz（监控同时采集空闲远端，计算仅在 node0）。

两版单机 entry 都是 REG168、stack0，ptxas spill store/load0；每 K64
四条 N256 WGMMA 后一次 wait，shared 和寄存器配额相同。SASS 中除 pull
边界/stride 外仍有编译器排程与地址偏移差异，不宣称只有三条指令不同。
资源审计工具增加了 local-only 选择，以免要求单机库包含未编译的 network entry。

数据：[逐轮性能、正确性、workload、artifact hash、时钟和 SASS](../benchmarks/pf_warp_budget_local_20261011.json)。
冻结产物：`../WeftMoE-experiments/warp-budget-20261011/build/local_warp_budget/pull{1,2}`。

## 双机对照设计

在相同 384-thread、Math208/Control88 的借用路径上比较：

| 候选 | pull warp/CTA | Get / Put / progress / admit worker |
|---|---:|---:|
| 冻结 `8c47a46` | 2 | 64 / 48 / 19 / 1 |
| 少一个 pull，额外 warp 空闲 | 1 | 64 / 48 / 19 / 1 |
| 少一个 pull，增加网络 worker | 1 | 128 / 96 / 39 / 1 |

先分离“少 pull”与“多 RDMA”两种效应，再检查是否值得保留。第二个网络
warp 使用其自己原有的 4KiB shared 区保存 pending requests，不能与同 CTA
的第一个网络 warp 共用在途请求数组。网络角色仍按 CTA/warp 静态归属，
不新增动态任务队列；host 的 CTA 数校验相应按每 CTA 的网络 warp 数计算。

不改变 132 lanes、D32/R128、pending4、B4096、credit windows2、slot 下限，
不增加 payload ring 容量，不改 Get/Put、RX credit、本地 NIC completion、
publication 或 Return-ready 去重。GIN contexts 从 131 增至 263，初始化
不计入稳态时间；它不是无额外资源成本的修改。完整性能结果待补。
