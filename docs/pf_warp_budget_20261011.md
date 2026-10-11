# PF：从两个 pull warp 减到一个（2026-10-11）

单机 EP8 交叉复测：2→1 个 pull warp 后，8K 基本持平（+0.21%），
20K 慢约 **1.83%**，不是完全无损。因此没有修改单机默认配置。
双机 EP16 再把空出的 warp 分给 RDMA，完整 8K/20K 反而慢约
**4.93%/1.45%**；不采用，继续保留 `8c47a46` 的 2-pull + 1-network 候选。
本轮不能证明原 64/48/19 配比已全局最优，但不支持简单减少 pull、翻倍 RDMA。

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

原有设备计时中，从路由准备完成到最后 pull 完成的区间，2-warp 为
2.944/7.884ms，1-warp 为 5.121/14.122ms（8K/20K，两轮 p50 均值）。
这是融合执行中的跨度，包含 ring 等待，并非独立通信时间，不能用它反推
纯拷贝带宽。它说明并发减半后输入生产余量显著缩小；完整 kernel 只慢
1.8% 不等于被删掉的 warp 没有工作。区间原始样本一并保存在结果归档。

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
不计入稳态时间；它不是无额外资源成本的修改。

## 双机完整 kernel 结果

新 native 源码 `118846c`，产物在同 worktree 的
`build/network_warp_budget/network_{idle,double}`。基线直接使用此前冻结的
`dispatch-optimized-20261010/build/optimized/borrow`（native `8c47a46`），
不是新编译后标为旧版本。launcher 的 commit 是启动脚本版本，不能混作
冻结 native 的源码版本；library/extension hash 和 build flags 另有记录。

EP16、H4096/I2048/E256/topK6/BF16、SM1620MHz，随机 tokens/routes 与单机
同方法，B4096。全部编译完成后串行测试，prime32/warm8/sample24。
顺序为基线 A → 1-pull idle → 1-pull double → 基线 B。基线取两次 p50
均值，两个候选各测一轮；这是筛选消融，不把小幅变化当成稳定加速。

| 布局 | 8K kernel ms | 20K kernel ms | 8K eager ms | 20K eager ms |
|---|---:|---:|---:|---:|
| 2 pull + 1 network，冻结基线 | **10.971088** | 23.487368 | **11.181332** | 23.960064 |
| 1 pull + 1 network，额外 warp 空闲 | 11.225952 | 23.446736 | 11.452465 | 23.700620 |
| 1 pull + 2 network，128/96/39/1 | 11.511808 | 23.827296 | 11.835432 | 24.100068 |

基线逐轮 kernel 为 10.986848/23.503120、10.955328/23.471616ms。
减少一个 pull、不给 RDMA 时，8K 慢 2.32%，20K -0.17% 视为持平；再启用
额外 RDMA warp，相对 idle 候选又慢 2.55%/1.62%。相对冻结基线，完整
8K/20K 慢 4.93%/1.45%。没有把单机的约 1.8% 损失换成多机收益。

结果支持的决策是保留两个 pull，而不是断言任何增加网络 worker 的配比
都无效。双机的等待/重叠结构不同，不能把单机变化按百分比直接套上去。
本轮也没有动态 stall 证据将回退唯一归因于寄存器、缓存或某一种轮询。
两个新布局的编译资源相同：entry REG168、累计 stack344B，spill store/load
56/36B；Get 252/388B、Put 140/280B，pull 0/0B。冻结基线对应为
stack336B、entry56/48B、Get252/384B、Put136/276B。静态字节数不是实际
运行流量；没有观察到因第二个 RDMA warp 导致明显 spill 恶化。
三版均保持 Math208/Control88、222KiB shared、每 K64 四条 N256 WGMMA
后一次 wait，没有新增 WG3 或改变 GEMM tile。

四轮正式 workload hash 和算子报告的 workspace bytes 相同；NCCL 内部
contexts 资源不在这些 workspace 统计内，不能据此宣称增加 context 零显存。
新两版各做正常路由 8K/20K×3 步，共 192 个 rank/sample 检查通过，输出
逐位一致，最大 relative L2 9.327e-5。614 次节点时钟观测均为 1620MHz。
六次启动均双节点 exit0，结束后全部 16 张 GPU 0MiB。

没有将无完整收益的候选继续扩成纯通信带宽 sweep，也没有用 full kernel
中重叠的 FC/Return 跨度反推带宽。新候选未替换主分支默认或 `8c47a46`；
代码保留在实验分支，供复现。这次单机为 inference-only 检查，未外推为
其它模型 shape、节点数或训练路径的性能保证。

数据：[双机完整计时、正确性、workload、时钟与 SASS](../benchmarks/pf_warp_budget_network_20261011.json)。
汇总工具 `061ecf6` 同时保留重叠阶段的 rank-max 逐样本区间。
