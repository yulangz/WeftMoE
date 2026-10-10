# PF Dispatch：ready 顺序与接收 rail fanout（2026-10-10）

最新进展：将本机 X 与远端 RX 转发拆成两条独立静态流，独立 Dispatch
达到 **36.46/35.36 GB/s**，超过本轮复测 V3 的 **31.85/33.94 GB/s**。
完整 GEMM 的 20K 仍回退，**不替换生产路径**。前几轮单纯 rail push 的
完整性能回退没有被掩盖，失败的同步、staging、配额消融均记录在下面。

目标口径：Flash 随机路由、8K/20K、EP16、1620MHz；历史 V3 Dispatch
32.08/33.60 GB/s。本文候选仍是实验代码，尚未替换主分支。

## 重新定位

冻结的 `dispatch-runs-20261010/build/dispatch_runs_phases/borrow`，开启探针，
同一 binary 比较逐 token Get 与连续 Get。时间单位 ms：

| 配置 | Dispatch 8K / 20K | Get 最晚完成 | pull 平均 worker 等 ready |
|---|---:|---:|---:|
| 原目录、逐 token Get | 2.647 / 6.049 | 1.704 / 4.242 | 0.552 / 1.475 |
| 连续 Get | 2.576 / 5.814 | 1.653 / 4.110 | 0.499 / 1.348 |

连续 Get 的 post 平均 worker 时间从 0.555/1.497 降至 0.050/0.144，
但整个 Dispatch 的下降远小于发射开销下降。worker 计时互相重叠，不能相加，
也不能把 Get 最晚完成与 Dispatch 的差值全部归因于某一个步骤。

原始记录：`benchmarks/dispatch_ready_probe_20261010.json`，在 ready 实验分支。
探针用于定位，不替代下面关闭探针的正式消融。

## 候选与同步

`0ac410b` 在既有静态 pull 归属内测试：

- ready-first：一个 warp 并行检查自己接下来至多 32 个 packet，处理已就绪项。
  没有跨 worker 领任务；未消费项一直保留在 bitmask 中，每项恰好处理一次。
  执行 acquire 的 lane 与 lane 0 之间先 syncwarp，再读 payload。
- packet fence：本地不可变输入按 window 做 async-proxy fence，远端按每次
  新 RX generation 的 acquire 做 fence；不再对同一 packet 的每个 fanout 重复。
- 二者组合。RX credit 仍在本 worker 的 packet fanout 全部完成后归还。

`c72ec3e` 进一步增加接收 rail push（跨机仍是 Get）：

1. 每个 GPU 只负责本 rail 的 source；本机 source 从不可变 X 读，远端 source
   从该 rail 的 RX ring 读。各 token/packet 子块按 warp residue 静态分配。
2. 一个 token 的 4KiB segment 只做一次本地 G2S，route lanes 将同一份 shared
   数据 S2G 到各目的 GPU 的 X 行。shared 复用等所有 fanout 的 read completion。
3. 生产路径先 system-acquire 每个实际目的 compute slot 的 generation；不能
   用本 GPU 的 admission 推断所有 peer 均可覆盖。GEMM 的 Network X-ready
   acquire 相应使用 system scope，单机 specialization 不变。
4. 各 lane 等待自己所有 global store 完成、写 gate，再 system-release 到目的
   X-ready。warp join 后，lane 0 以精确 fanout 数归还 RX credit。没有提前释放。

没有新增 payload buffer；复用每 warp 的 4KiB staging 与原 compute/RX ring。
静态 packet 子块为 4 tokens，尾块按实际行数处理；topK 大于 32 时分组处理。
跨机 Get/Put owner、Return、GEMM 任务与行布局均不变。

独立通信诊断仍保留完整 canonical X 输出，其生产 compute-ring 复用由现有
phase 模拟替代；这不是为生产扩容。生产路径的 slot-generation 等待需要在
真实 GEMM 和小 ring 测试中另行验证。

## 同 binary 独立 Dispatch 消融

源码 `c72ec3e`，`build/dispatch_rail_phases/borrow`，连续 Get 开启，探针关闭。
每项 8 warmup + 24 samples，随机 trace 与历史 V3 相同：

| 配置 | Dispatch ms，8K / 20K | 平均 GB/s，8K / 20K |
|---|---:|---:|
| 原 pull | 2.565 / 5.824 | 25.42 / 28.63 |
| ready-first | 2.575 / 5.708 | 25.46 / 28.80 |
| packet fence | 2.575 / 5.826 | 25.38 / 28.67 |
| ready-first + packet fence | 2.570 / 5.700 | 25.52 / 28.93 |
| 接收 rail push | 2.178 / 5.335 | 30.65 / 31.33 |
| 历史 V3，两轮 p50 均值 | 2.096 / 4.931 | 32.08 / 33.60 |

ready-first/fence 没有消除差距；20K 的小收益不足以优先推进这些复杂度。
rail push 延迟比同 binary 原 pull 降低 15.1%/8.4%，平均带宽提高
20.6%/9.4%，但**尚未追上 V3**。最忙 rail 为 36.61/36.73 GB/s；Return
保持 36.03/38.23 GB/s 平均、44.53/45.86 GB/s 最忙 rail。
延迟与带宽各自逐样本计算再取中位数，百分比不是相互精确倒数。

480 个 rank/shape/step 校验逐位通过，各项输出 hash、workload 相同；434 次
双节点时钟观测均为 1620MHz。实际
跨机 payload 与 route 推导相符。原始记录
`benchmarks/dispatch_ready_fanout_phases_20261010.json`。

## 完整 GEMM 与跨机 Put

相同 `c72ec3e`，真实 GEMM artifact `build/dispatch_rail_fused/borrow`，
Get + 相同随机 workload，完整 kernel 包含 planner、GEMM、通信与 combine：

| 配置 | 8K ms | 20K ms |
|---|---:|---:|
| 原 pull | 10.719152 | 23.096816 |
| 接收 rail push | 10.912464 | 23.973024 |

输出校验通过，但 push 回退 1.8%/3.8%。独立 Dispatch 的改善不能外推为融合
收益。该版每条 fanout 都读取 peer block metadata，并在生产路径额外轮询目的
slot generation；独立通信诊断使用完整 canonical 输出，没有这项生产等待。
另一待验证因素是 source-token fanout 顺序与原 local expert-row pull 顺序的
差别：最早一批 FC1 tile 的 ready 时间可能变晚。这是代码分析假设，尚未用
后续消融确定各项占比。数据：`benchmarks/dispatch_ready_fanout_fused_20261010.json`。

同一 phase binary，保留 rail push，仅将跨机协议换为已有 pack-Put：

| 跨机协议 | Dispatch ms，8K / 20K | 平均 GB/s，8K / 20K |
|---|---:|---:|
| 连续 Get | 2.178 / 5.335 | 30.65 / 31.33 |
| pack-Put | 2.810 / 6.505 | 23.76 / 25.65 |

pack-Put 使用原 phase 专用 pack worker，pack 成本计入 Dispatch；它不是
重做过 pack/Put 重叠的生产实现，因此结论只针对这个实现，不是 Put 的上限。
该结果不支持直接把当前跨机 Get 换成已有 Put 路径。数据：
`benchmarks/dispatch_rail_put_20261010.json`。

## 验证状态

正式性能测试时不编译，各 GPU 测试串行。
编译产物保留于各实验 worktree 的 `build/dispatch_{ready,rail}_{phases,fused}`。
同 binary 消融关闭探针，使用相同 workload、固定 buffer、24 samples。
主要比较平均卡单方向有效 payload GB/s，不用
最忙 rail 或仅 NIC 子区间替代整个 Dispatch。

下一实验分支 `experiment/pf-dispatch-fanout-plan-20261010` 单独测试：

- 把 X-ready 增量写入 RouteLocation 原有 padding，尺寸仍为 24B，免去 peer
  block 的额外读取；无额外 kernel/buffer。
- 每个 warp 在一个 window 内按需缓存成功 acquire 的 peer admission，下一
  window 清空。未写入的行会阻止 FC1 retirement，故相应 slot 不能被覆盖。
  admission head 改为 system-release，与 peer system-acquire 配对。
- 保留本机数据的 expert-row 顺序 pull，仅对远端 RX 采用 rail push。

### 后续同 binary 消融：metadata / admission / 本机顺序

源码 `66dde52`，`dispatch-fanout-plan-20261010` 的
`build/dispatch_fanout_plan_{phases,fused}/borrow`。同一 workload、1620MHz，
每项各一次 24-sample 测量，不把跨 binary 的基线变化算作优化收益。

| 实现 | 完整 GEMM 8K ms | 完整 GEMM 20K ms |
|---|---:|---:|
| 原 pull，本 binary 对照 | 10.861808 | 23.490672 |
| rail push，仍读取 peer block | 11.039856 | 24.216464 |
| 预计算 ready 增量 | 11.079840 | 24.442544 |
| 再缓存 peer admission | 11.152432 | 24.432656 |
| 再保留本机 expert-row pull，仅远端 RX push | 10.932800 | 23.905120 |

metadata 与重复 admission 读取不是这组回退的主要解法，没有实测收益。
保留本机 expert 顺序收回了一部分损失，但仍比原 pull 慢 0.7%/1.8%。
不能采用这些候选替换原 pull，也不能据此宣称已证明全部回退来自就绪顺序。

独立 Dispatch：

| 实现 | ms，8K / 20K | 平均 GB/s，8K / 20K |
|---|---:|---:|
| rail push，仍读取 peer block | 2.170 / 5.367 | 30.85 / 31.08 |
| 预计算 ready 增量 | 2.174 / 5.343 | 30.83 / 31.17 |
| 本机 pull + 远端 RX push | 2.549 / 5.747 | 25.64 / 28.42 |

这说明本机 source 的搬运方式对独立 Dispatch 收益影响很大，不能把前面的
提升全部归因于跨机 Get 或远端 RX 转发。以上结果仍未达到 V3 目标。
原始数据：`benchmarks/dispatch_fanout_plan_{phases,fused}_20261010.json`。

### 批量完成与本机 expert 顺序 source push

源码 `ba44a02`，`dispatch-fanout-batch-20261010` 的
`build/dispatch_fanout_batch_{phases,fused}/borrow`。每 warp 连续发射 4 个 token，staging 仍按 shared-read
completion 复用；批末才等待 global completion、发布各行 ready、归还 RX
credit。用重新读取少量不可变描述符代替寄存器保留整批 descriptor，不加显存。
另测本机按 destination/expert 顺序的静态 source push，区别于源 token fanout
以及 destination pull。本机 route push 按 destination 静态划分 warp 子组，
每个子组遍历该目的 rank 的 expert-ordered source routes；没有动态领任务。

| 配置 | 独立 Dispatch 平均 GB/s，8K / 20K | 完整 GEMM ms，8K / 20K |
|---|---:|---:|
| 原 pull，本 binary 对照 | 未重复测 | 10.737152 / 23.095840 |
| rail push，预计算增量、逐 token 完成 | 30.83 / 30.97 | 11.030160 / 24.054160 |
| rail push，4-token 批量完成 | 29.70 / 31.02 | 10.906720 / 23.843072 |
| 本机 expert 顺序 source push，远端批量 fanout | 24.58 / 28.26 | 10.683616 / 23.339776 |
| 本机 pull，远端批量 fanout | 25.28 / 27.87 | 未重复测 |

批量完成没有独立通信收益；融合下比逐 token 的 rail push 略好，但仍落后
原 pull。本机 expert 顺序 source push 的完整 GEMM，8K 快约 0.5%、20K 慢
约 1.1%，这一轮不足以声称优于原实现，也不能用较好的 8K 掩盖 20K 回退。
因此仍不替换生产/保留基线。

数据：`benchmarks/dispatch_fanout_batch_{phases,fused}_20261010.json`。
所有性能测试串行、编译结束后开始；buffer 容量和随机 trace 保持相同。

下一组进一步对齐 V3 的发布协议：V3 的 `Receiver::publish_dispatch` 使用
按 tile/source rail 区分的到达计数，且更新使用 release；当前 fanout 则通过
通用 `ring::arrive` 对同一 X-ready 做 acq_rel 更新。实验分支
`experiment/pf-dispatch-ready-stripes-20261010` 将分开验证 per-rail 计数与
release-only reduction。前者需要少量随 pool 持久分配的计数器，后者不需要
新 buffer；两者均保留接收端 acquire，未取得结果前不宣称其收益。

### X-ready 的 release-only 与 per-rail 计数

源码 `a2720db`，`dispatch-ready-stripes-20261010` 的
`build/dispatch_ready_stripes_{phases,fused}/borrow`。不改 payload ring、任务
划分、Get/Return worker 或 GEMM tile；同 binary 单独切换两个因素：

- release-only：数据写完后只发布，不获取旧值。SASS 确认对应更新从
  `ATOM.E.ADD.64.STRONG.SYS` 变为 `REDG.E.ADD.64.STRONG.SYS`，消费者
  仍 acquire。RX credit 更新不改，不能将这项结论推广到所有原子操作。
- per-rail：每个 compute slot 为各生产 rail 分配独立 32B counter sector，
  消费者 acquire 并求和，到齐后缓存本地 X-ready。计数跨 generation 累积，
  随固定 pool 一次分配/释放。此 shape 的 pool 增加 794,624B，约 0.76MiB。

| 实现 | 独立 Dispatch ms，8K / 20K | 平均 GB/s，8K / 20K |
|---|---:|---:|
| rail push，同 binary 对照 | 2.086 / 5.344 | 31.63 / 31.07 |
| release-only | 2.083 / 5.332 | 31.88 / 30.94 |
| per-rail | 2.094 / 5.362 | 31.53 / 31.02 |
| per-rail + release-only | 2.083 / 5.353 | 31.67 / 31.29 |

纯通信没有明显收益。8K 与前轮绝对值略有变化，不将跨 binary 的差异计为
收益；上述四项共用 workload、产物与 timing 口径。384 个验证项逐位正确，
336 次双机时钟观测均为 1620MHz。Return 保持约 35.9/38.2 GB/s。
数据：`benchmarks/dispatch_ready_stripes_phases_20261010.json`。

完整 GEMM 同 binary 结果：

| 实现 | 8K ms | 20K ms |
|---|---:|---:|
| 原 pull | 10.730944 | 23.084032 |
| rail push，对照 | 10.783040 | 24.119760 |
| rail push + release-only | 10.713920 | 24.154096 |
| rail push + per-rail + release-only | 10.777600 | 24.189072 |
| 本机 expert 顺序 source push + release-only | 10.576624 | 23.253616 |

两个 ready 协议调整都没有追回 20K 的回退。最后一项保留 expert 顺序，8K
快约 1.4%，但 20K 仍慢约 0.7%；不足以替换原 pull。新增 counter 不保留到
生产路径。数据：`benchmarks/dispatch_ready_stripes_fused_20261010.json`。

下一候选将两段 4KiB G2S 同时发出、
使用同一 completion barrier，转发后统一保护 shared 复用；只增加 5KiB
shared、不增加 GM payload。另独立消融按 immutable route count 加权的静态
warp 配额，检查 expert-ordered source push 的目的 rank 负载不均问题。

### 两段并行加载与按 route count 分配 warp

源码 `c26eb40`，`dispatch-fanout-wide-20261010` 的
`build/dispatch_fanout_wide_{phases,fused}/borrow`。

- wide：复用闲置的 `pull[0]`，将 network pending 搬到额外 shared 区，两个
  转发 warp 各有 2×4KiB staging。两次 G2S 同时发出，合用一个 completion
  barrier；两次 S2G 合成一个 store group。保持 shared-read 和 global-write
  两个完成边界。总 shared 增加 5KiB，含 static 为 227KiB。
- weighted：仅对本机 expert-ordered source push，按各目的 rank 的 route
  count 静态分配 warp。每 rank 至少一个 warp，余量按累计计数分配，舍入后
  总数仍精确等于 worker 数；不增加共享任务队列。

同 binary 独立 Dispatch：

| 配置 | ms，8K / 20K | 平均 GB/s，8K / 20K |
|---|---:|---:|
| rail push + release-only，4KiB | 2.066 / 5.324 | 31.84 / 31.17 |
| 同上，8KiB | 2.064 / 5.269 | 32.00 / 31.31 |
| 本机 expert 顺序 + weighted + 8KiB | 2.304 / 5.318 | 28.95 / 31.43 |

宽 staging 的纯通信收益很小：20K 延迟约降低 1%，8K 基本持平。不能据此
声称两段串行加载是主要瓶颈。最后一行同时更改本机顺序和配额，不把它与
前一行的差异归因于 weighted 单一因素。

完整 kernel：

| 配置 | 8K ms | 20K ms |
|---|---:|---:|
| 原 pull，同 binary 对照 | 10.742048 | 23.024048 |
| rail push + 8KiB | 10.776992 | 24.315696 |
| 本机 expert 顺序 source push + 8KiB，均分 warp | 10.584848 | 23.400752 |
| 上一项 + weighted | 10.595728 | 23.327936 |

weighted 仅对该同库 expert-order 候选的 20K 提高约 0.3%，8K 基本持平。
两种 expert-order 候选仍在 20K 落后原 pull，均不采用。math208/control88、
N256 WGMMA 未改变；wide/weighted helper 均无 spill store，ptxas 报告
8/12B spill load，不宣称整个 kernel 无 spill。

数据：`benchmarks/dispatch_fanout_wide_{phases,fused}_20261010.json`。
下一轮单独验证：本机 source 与远端 RX 各一个专用转发 warp，使相邻的
source-node windows 可以重叠；另将本机 expert 顺序流中，同一 warp 对
同一目的 M-tile 的连续 ready 更新合并。合并只发生在本窗口内，窗口结束
前必须 flush，不能拿未发布的旧 ready 等待未来窗口 admission。

### 本机 X / 远端 RX 独立静态流：纯通信超过 V3

源码 `ef4db75`（分流 `9ba828d`，合并通知独立开关 `ef4db75`），
`dispatch-local-remote-20261010` 的 `build/dispatch_local_remote_phases/borrow`。
两种来源各使用原来的一个转发 warp：worker 的偶数/奇数 residue 分别只遍历
本机 source / 远端 source，各自推进窗口，不再先完成本机窗口才处理远端窗口。
源 rank/目的节点按实际拓扑判断，没有把协议写死成两节点。

仍为跨机连续 Get、接收 rail 的节点内 fanout push。RX 逐 slot 的可见 generation、
精确 fanout retirement、目的 compute slot 的 generation 检查全部保留。
不新增 warp、GM payload 或动态领任务原子；使用上一轮已有的 8KiB staging。

| 实现 | Dispatch ms，8K / 20K | 平均 GB/s，8K / 20K |
|---|---:|---:|
| rail fanout，同库未分流 | 2.066 / 5.308 | 31.95 / 31.23 |
| rail fanout，本机/远端分流 | **1.850 / 4.697** | **36.46 / 35.36** |
| 本机 expert 顺序 + 加权分工，再分流 | 2.017 / 4.988 | 32.85 / 32.87 |
| 上一项再合并同 tile ready | 2.087 / 5.199 | 32.37 / 31.30 |
| V3，当轮重新测量 | 2.123 / 4.960 | 31.85 / 33.94 |

分流相对同库 rail fanout 将延迟降低 10.5%/11.5%；相对新测 V3 降低
12.9%/5.3%。带宽各样本独立计算再取 p50，所以带宽与延迟百分比不互为精确
倒数。分流版最忙 dispatch rail 为 43.34/41.79 GB/s；Return 未修改，平均
35.95/38.18 GB/s，最忙 rail 44.51/45.85 GB/s。

V3 使用此前 `bbf2bd4` 的冻结 artifact，library hash 与历史一致；未重编译。
所有配置的实际 payload、workload、输出 hash 一致，480 个 rank/shape/step
校验项逐位通过，434 次节点时钟观测均为 1620MHz。数据：
`benchmarks/dispatch_local_remote_phases_20261010.json`。

这个结果确认本机与远端转发之间存在可利用的重叠空间；不能据此认为 GEMM
融合收益已得到验证。按 tile 合并 ready 的候选此轮反而更慢，不将它计为收益。
完整 kernel 同 binary 对照如下，仍包含主 kernel 内的 planner 和真实 GEMM：

| 实现 | 8K ms | 20K ms |
|---|---:|---:|
| 原 pull | 10.765856 | 23.101600 |
| 本机 expert 顺序 + weighted，未分流 | 10.572880 | 23.364368 |
| rail fanout 分流，纯通信最快者 | 10.846912 | 24.333664 |
| 本机 expert 顺序 + weighted，再分流 | 10.706512 | 23.313536 |
| 上一项再合并同 tile ready | 10.591072 | 23.326960 |

纯通信最快者的完整 20K 回退约 5.3%，不采用。保持 expert 顺序后回退缩小，
但仍未胜过原 pull；合并通知没有稳定解决 20K。不能将“独立 Dispatch 超过
V3”写成“生产 MoE 已有同等收益”。原始记录：
`benchmarks/dispatch_local_remote_fused_20261010.json`。

下一候选保留本机 destination-pull，只让另一个 warp 处理远端 RX fanout
push；分别测试本机 4KiB 与 2×4KiB staging。它复用已验证有效的分流机制，
但不要求本机输入也改成 source fanout。跨机仍为 Get，Return 不变。

### 本机 destination-pull / 远端 rail fanout 混合：仍未解决融合回退

源码 `29fa6ef`，`dispatch-hybrid-overlap-20261010` 的
`build/dispatch_hybrid_overlap_{phases,fused}/borrow`。一个转发 warp 按本机
目的 expert 行顺序 pull，另一个独立转发远端 RX；本机 staging 分别使用
4KiB、2×4KiB。不改变 compute/RX ring 的 generation、credit 和完成条件。

| 配置 | 独立 Dispatch ms，8K / 20K | 平均 GB/s，8K / 20K | 完整 kernel ms，8K / 20K |
|---|---:|---:|---:|
| 原 pull，同 binary | 2.587 / 5.827 | 25.20 / 28.42 | 10.733920 / 23.145696 |
| 混合，4KiB 本机 staging | 2.757 / 6.888 | 23.43 / 23.70 | 10.707632 / 23.296288 |
| 混合，8KiB 本机 staging | 2.474 / 6.152 | 26.12 / 26.56 | 10.661392 / 23.239184 |

混合版用 8KiB 比 4KiB 好，但完整 20K 仍比原 pull 慢约 0.4%，8K 仅快约
0.7%；不采用。相比全 rail fanout，它缓解了完整 GEMM 的回退，却也失去了
纯通信的大部分收益，不能认为问题已经解决。本机只剩一半 pull worker
是此方案同时引入的变化，不能把差异全部归因于 push/pull 方式本身。

phase 与 fused 各 288 个 rank/shape/step 校验项通过；同负载输出 hash
一致，fused 最大 relative L2 为 9.231e-5。两组分别 260/262 次时钟观测
均为 1620MHz。math208/control88、N256 WGMMA 不变；wide 本机 helper
仍有 ptxas 报告的 4B spill store、8B spill load，不宣称整个实现无 spill。
数据：`benchmarks/dispatch_hybrid_overlap_{phases,fused}_20261010.json`。

下一步对原 pull 与纯通信最快的 rail 分流版开启已有 Get 探针：区分请求
发射、NIC completion 等待与 RX reuse credit 停顿。full kernel 的
`dispatch_us` 未独立定义，不能把零值当成通信时间；各 worker 计时也不能
相加作为关键路径。新 fanout 暂未实现旧 pull 的行拷贝探针，其零值不表示
转发没有开销。
