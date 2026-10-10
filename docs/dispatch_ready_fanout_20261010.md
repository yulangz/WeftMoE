# PF Dispatch：ready 顺序与接收 rail fanout（2026-10-10）

结论：接收 rail push 将独立 Dispatch 提高到 30.65/31.33 GB/s，但首版
完整 GEMM 回退 1.8%/3.8%，**不采用这版替换原 pull**。ready-first/fence
收益很小；沿用旧 pack worker 的跨机 Put 也更慢。后续继续分离 metadata、
admission 与 FC1 就绪顺序的影响，尚未达成追上 V3 的目标。

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

正在准备的下一步：每 warp 连续发射 4 个 token，staging 仍按 shared-read
completion 复用；批末才等待 global completion、发布各行 ready、归还 RX
credit。用重新读取少量不可变描述符代替寄存器保留整批 descriptor，不加显存。
另测本机按 destination/expert 顺序的静态 source push，区别于源 token fanout
以及 destination pull。上述新候选尚未实测，不预设其收益。
