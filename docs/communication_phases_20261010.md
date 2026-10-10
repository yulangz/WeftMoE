# Dispatch / Return 分阶段通信诊断（2026-10-10）

本轮目标是分别测量通信阶段，不再用 Dispatch、Return 同时运行的区间
分别计算两个“独立阶段”的带宽。生产默认配置不变。

**后续最终保留的精简候选**：仅将借用 warp 的路由筛选并行化。
真实 GEMM 两轮复测，8K/20K kernel 从 **12.211/24.828ms →
10.919/23.505ms**，延迟降低 **10.6%/5.3%**。代码 `8c47a46`，
在 `experiment/pf-dispatch-optimized-20261010`；本主分支只同步记录，
尚未切换生产默认。没有合入双 staging、探针、payload-skip 等消融代码。

最新 [ready 顺序与接收 rail push](dispatch_ready_fanout_20261010.md) 消融：
将本机 X / 远端 RX 转发分成两条独立静态流，独立 Dispatch 达到
**36.46/35.36 GB/s**，超过当轮重新测量 V3 的 **31.85/33.94 GB/s**。
但纯通信最快者的完整 20K 仍回退约 5.3%，不替换生产路径；保留本机
destination-pull 的混合方案也未解决 20K 回退。完整 kernel 探针进一步看到
20K Get 更早结束（5.95 → 4.94ms），kernel 却更慢（23.14 → 24.36ms）；
接下来需要定位 tile 就绪顺序与转发/计算的竞争。所有无收益实验均保留记录。

最新两组后续消融：[本地拷贝描述表](dispatch_descriptors_20261010.md) 无端到端
收益，已放弃；[有序目录与连续 Get](dispatch_get_runs_20261010.md) 将请求数减少
约 95.5%，但干净版完整 MoE 仅 8K 快约 0.54%、20K 持平。`6e5d730` 保留为
候选，未替换上述基线；不把独立 Dispatch 的小收益当作显著融合收益。

后续借用 warp 消融已定位主要原因：单 lane 串行筛选转发路由。
保留 Get + bulk copy，仅改为 warp 并行筛选，独立 Dispatch 约
6.29/14.37 → 2.63/5.94 ms（8K/20K）。pack+Put 没有明确额外收益。
详见文末。后续 staging/ready 消融将独立 Dispatch 进一步降至约
2.5/5.8 ms；两级 2KiB staging 没有胜过单级 4KiB。

## 结论与实测结果

**当前 PF 的主要短板是 Dispatch，而不是 Return 的单 rail RDMA 带宽。**
V3 的 Dispatch 明显领先；Return 最忙 rail 三者都在约 44–46 GB/s，
但 PF 的 rail 流量更均衡，因此同负载下 Return 整体耗时反而低于 V3。

下表为两轮独立测试 p50 的均值。GB/s 两个数字依次为
**平均卡有效带宽 / 最忙 rail 带宽**，均使用全阶段时长，不是瞬时峰值。

| 每 rank tokens | 实现 | Dispatch ms | Dispatch GB/s | Return ms | Return GB/s |
|---|---|---:|---:|---:|---:|
| 8K | V3 | 2.096 | 32.08 / 38.19 | 7.251 | 27.77 / 45.03 |
| 8K | PF 借用 warp，208/88 | 5.991 | 11.15 / 13.29 | 5.659 | 35.91 / 44.50 |
| 8K | PF 全 CTA WG3 | 4.752 | 13.90 / 16.44 | 5.743 | 35.21 / 43.67 |
| 20K | V3 | 4.931 | 33.60 / 39.36 | 17.739 | 28.66 / 45.59 |
| 20K | PF 借用 warp，208/88 | 13.752 | 12.02 / 14.03 | 13.260 | 38.15 / 45.79 |
| 20K | PF 全 CTA WG3 | 10.591 | 15.77 / 18.62 | 13.523 | 37.42 / 45.01 |

结论：

- PF Dispatch 尚未打满：最忙 rail 也只有约 13–19 GB/s；
  V3 为约 38–39 GB/s。WG3 比借用版的平均 Dispatch 吞吐高 25% / 31%，
  仍未解决与 V3 的明显差距。
- Return 最忙 rail 达到单方向理论 50 GB/s 的约 87%–92%，三者接近。
  不能说整个节点所有 NIC 都打满了；流量不均及阶段尾部仍降低平均吞吐。
- PF 借用版 Return 比 V3 耗时低约 22% / 25%，不是其单条链路更快，
  rail 分布更均衡是一个重要原因。不能把全部收益精确归因于单个变化。
- 时间和带宽各自先按样本计算，再取 p50；不要用两个独立的 p50 相除重算带宽。

顺序为 V3 → 借用 → WG3 → WG3 → 借用 → V3，所有 GPU 测试串行，
正式计时期间无编译。两轮数值一致性良好，没有择取最优一轮。
每种实现首轮验证 3 步 × 2 规模 × 16 rank，共 288 个 rank/sample；
与 identity 参考逐位相等，各实现输出 hash 也完全相同。
六轮的 32 步 token/route/gate trace 逐 rank 相同，所有计时样本的
实际 Dispatch / Return payload 与路由预期相等。
738 次节点时钟观测全部 1620MHz；六次启动均两机 exit0；
结束后两机全部 GPU 0MiB。

[逐 rank 原始阶段时间、payload、workload、hash 与时钟摘要](../benchmarks/communication_phases_20261010.json)。
实验 native 源码提交为 `bbf2bd4`，artifact 在实验 worktree 的
`build/communication_phases/{borrow,wg3,v3}`；后续汇总脚本的修改不改变
这三个 native artifact，两轮分别使用完全相同的 library/adapter hash。

## 结果解释与历史口径

V3 的历史高带宽确实存在：[2026-09-16 记录](communication_spsc_20260916.md)
使用固定 Qwen fixture（H2048/I512/E256/K8），8K Return 为 42.4 GB/s，
64K 为 45.7 GB/s；更早对照为 42.3 / 46.2 GB/s。
它不是本轮 Flash H4096/K6、随机 token 数和带 bias 路由的同负载对照。

本轮需要区分“平均卡有效带宽”和“最忙 rail 带宽”。V3 的 Return 从
expert GPU 直接向 source rank 发送，单卡发送量随专家负载变化。
本次最忙卡/平均卡的 Return payload 比为 1.615（8K）、1.596（20K）。
PF 则先通过 NVLink 将 contribution 写到 source 对应的本地 rail，再跨机
Put；相同比值降为 1.236 / 1.196。因此全局 payload 完全相同，不代表
逐卡 NIC payload 相同。最忙卡接近链路上限时，全局平均仍可能明显较低。
不能将较低的 V3 平均值解读为其 NIC 只能传二十几 GB/s。

PF 当前 Dispatch 保留 per-token Get + 目的端逐行 pull：
每个 pull worker 的 4KiB G2S 完成后执行 S2G，等待完成再复用同一个 staging。
V3 则使用 pack-Put，接收端从本地 RX 做机内 push，并使用多 buffer 的
row-prefetch pipeline。两者还存在 worker 分配与 credit 等待方式的差别。
本实验能确定完整 Dispatch 链路有差距，但不能单独证明“Get 不如 Put”
或“静态分配不如动态队列”；这些变化不是单变量消融。

下一步应优先隔离 Dispatch 的 NIC 提交、RX-consumer credit 和机内转发，
比较多 staging 流水、源端 pack-Put、目的机内 push。
独立 wire-only 测试已显示 Get/Put 可以达到约 45–48 GB/s，
见 [GIN 通信测量](pf_communication_bandwidth_20261009.md)；
它不包含真实机内 fanout/credit，不能替代本轮端到端通信阶段的数字。

这里刻意串行 Dispatch、Return、Reduce，所以两段相加不能作为原融合
kernel 的延迟预测。生产版本允许三者与 GEMM 重叠；上一轮两方向同时运行
的 copy 诊断和本轮阶段串行的 identity 诊断，也不是同一个性能指标。

## 口径

- 双机 EP16，每节点 8×L20X，SM 固定 1620MHz。
- BF16，H4096 / I2048 / E256 / topK6，名义每 rank 8192、20480 tokens。
- 每步重新生成路由，bias 标准差 0.5；tokens 正态波动 10%，seed123。
- 相同逐 rank token/route/gate trace；先 prime32，再 warm8、measure24。
- 计时前校验前三个 priming 调用的输出；其余 priming 不计时。
- GEMM 替换为 identity expert；最终输出应为 X × 有效路由数，不乘 gate。
- 分阶段使用设备 globaltimer；不计固定 buffer、GIN context 创建/释放、
  权重预处理。metadata/planner、阶段边界 rendezvous、最终 reduce 不计入
  Dispatch / Return 的 payload 带宽。
- 每个样本：全局实际跨机 payload / 16 / 最慢 rank 阶段时长，再取 p50。
  单位 GB/s，单卡单方向；不把 TX+RX 加起来后比较单方向链路上限。
- 同时报告最忙 rail 的 payload / 同一个最慢 rank 时长。它使用整个阶段的
  时间，不是瞬时峰值，也不是取一个更短的有利发送区间。
  随机负载下最忙卡不等于平均卡，判断 NIC 是否接近上限必须区分二者。
- 每步核对实际 payload 与路由推导值。Dispatch 按目的节点去重；
  Return 按远端 expert contribution 数，不去重。
- 只有两个节点，未据此声称任意多节点的扩展效率。

## 诊断与融合 kernel 的区别

阶段顺序是 planner → Dispatch → 全 rank rendezvous → Return → 全 rank
rendezvous → Reduce。保留生产通信 worker、warp 划分、寄存器配额、
shared memory 预留、RX credit 和本地 NIC completion。

为让 Dispatch 在 Return 完全不启动时结束，PF 将完成的接收行保存在
完整 invocation-sized canonical X 输出中，而不是循环覆盖计算 X ring。
计算槽控制计数仍按原 admission 窗口推进；Dispatch 结束后恢复其计算、
Return 计数基线，只保留本次已经推进的 X-ready 计数。
网络 RX/TX ring 的大小、fanout 消费和复用协议均不变。
完整接收输出是独立通信 API 的必要输出，不是生产 PF 的新增长生命周期缓存。
本轮 8K/20K 的完整输出均能容纳在原先预留的 canonical arena 中：
借用版 compute / GIN / notify 分配字节数与上一轮 copy88 完全相同。
这里改变的是输出地址映射与存活期，不是通过扩大网络 ring 容量获取带宽。

Return 的输入全部 ready，PF 用静态 FC2 N-tile 顺序执行 uint4 拷贝，
写入原有 TX packet / 本地 contribution 地址，然后运行原有 Put 和
progress worker。这个读取已完成输出、写入 TX 的搬运计入 Return；
真实 PF 中对应写入由 FC2 epilogue 完成，所以这里的 Return 时间
不能直接当成融合 kernel 中不可折叠的通信开销。

V3 复用已有 transport-only 诊断：完整 canonical 接收行直接充当 FC2 输出，
保留生产 pack-Put、机内转发和 Return pack。补齐过时模板接口、
原生产寄存器配额及全 rank 阶段边界；Reduce 延后到 Return 全部完成。
边界只由 CTA0 和本地控制计数、GIN rail barrier 完成，不引入 V3 的
全 grid 静态工作等待。

V3 采用 recovery/v3-shared-queue 的 EP16 Flash 调优配置。
该分支与本实验基底的 persistent_v3、unified transport/kernel、
initialize_pipeline 的源码比较，只有 FANOUT_LANES=1 的缺省宏差异；
恢复配置已经显式定义该值，不改变 V3 算法。

## 候选

| 候选 | CTA 线程 | Math / Control 配额 | Get / Put / Progress / Admit |
|---|---:|---|---|
| PF 借用 warp | 384 | 208 / 88 | 64 / 48 / 19 / 1 |
| PF 全 CTA WG3 | 512 | 192 / Local64 / Network64 | 132 / 132 / 132 / 1 |
| V3 | 384 | 216 / 64 | 原 16 sender CTA + 116 receiver/local CTA |

PF 使用 token Get Dispatch、packet Put Return，132 lanes、D32/R128、
pending4、B4096、credit windows2、4 slots/channel、publication plan、
Return-ready 去重。Dispatch 不做 route pack；Return 不额外运行独立 pack kernel。
本分阶段诊断仅覆盖这个 Get 配置，不覆盖实验性的 Dispatch push 配置。

## 复现

实验分支：experiment/communication-phases-20261010。

```bash
/opt/.venv/bin/python benchmarks/build_communication_phases.py
/opt/.venv/bin/python benchmarks/run_pf_network.py \
  --artifact build/communication_phases/borrow \
  --label borrow_r1 --output build/communication_phases/runs \
  --master-port 32010 \
  --env WEFTMOE_PF_MIN_RING_DEPTH=2 \
  --env WEFTMOE_PF_CREDIT_WINDOWS=2 \
  --env WEFTMOE_PF_WG3_PUBLICATION_PLAN=1 \
  --env WEFTMOE_PF_DEDUPLICATE_RETURN_READY=1 -- \
  --backend pf --hidden 4096 --intermediate 2048 --experts 256 --top-k 6 \
  --tokens 8192 20480 --ring-tokens 4096 --prepared-weights \
  --profile-device --communication-phases --verify-steps 3 --verify-kernel
```

WG3 换 artifact 和 label；V3 换 artifact、label、backend=v3，并省略 PF 环境变量。
summary 脚本读取 node0 的逐 rank 原始结果并核对每步 payload。
每次启动的 manifest 记录 library/adapter hash、源码版本、环境与命令，
clocks 文件记录两节点的实际频率。GPU 测试串行，正式计时期间没有编译任务。

## 借用 warp：Dispatch 瓶颈定位

继续使用上面的 EP16、1620MHz、Flash、随机负载与 ring 几何，不增加 warp。
实验分支 `experiment/pf-dispatch-diagnosis-20261010`。

### 第一轮：不是单纯的 Get 吞吐不足，也不只是 payload 拷贝慢

为 Get owner 和 pull worker 添加可关闭的设备计时。计时只做 worker 本地
累加，退出时汇总；不在每次轮询中做全局 atomic。下表含探针开销，
不能与前面无探针数字直接作为优化前后对比。

| 消融 | 8K Dispatch ms | 20K Dispatch ms | 输出有效 |
|---|---:|---:|---|
| 原 bulk 转发 | 6.265 | 14.351 | 是，逐位一致 |
| 跳过所有转发 payload 拷贝 | 5.221 | 11.837 | 否，仅定位 |
| 只跳过本地来源拷贝 | 5.789 | 13.177 | 否，仅定位 |
| 只跳过远端 RX 拷贝 | 5.750 | 12.890 | 否，仅定位 |
| 改为全 warp uint4 load/store 转发 | 11.668 | 27.968 | 是，逐位一致 |

跳过 payload 仍保留 ready、路由扫描、X-ready 发布、fanout 消费、admission
与网络 RX 复用协议；绝不能当成可用算子或实际 Dispatch 带宽优化。
这里只解释 Dispatch；这些不完整输出的 Return 时间没有性能结论价值。

原 bulk 下，最晚 Get owner 完成全部任务的时间约 **1.689 / 5.212 ms**，
最晚 pull worker 为 **6.263 / 14.349 ms**。8K 的所有 Get owner 没有
RX credit 失败轮询；20K 平均每 packet 约 2.57 次。即使承认 owner 起始时间
有很小偏差，也无法用 NIC 未完成解释如此长的后续转发尾部。

拷贝探针：本地来源每行约 3.06 / 2.93 µs，远端 RX 每行约 2.65 / 2.73 µs。
直接 uint4 版本每行约 17–18 µs：循环每轮只搬一个 warp 的 512B，
仍受跨 GPU 读取依赖与逐行 system fence 约束，不代表优化过的所有 vector
实现都如此，也不能据此推广为一般的 TMA/vector 大小关系。

代码层面的关键嫌疑是转发控制路径：每个目的 GPU 的 worker 使用 lane0
逐一访问 packet 中全部 token×topK，再筛选本 rank 的行；多数源目录在
其它本地 rail 上。这既重复扫描非本 rank 路由，也串行化 metadata 依赖。
“跳过全部 payload 后仍剩 82%–83% 时间”定位的是这条控制路径及同步，
不是宣称已将剩余时间全部精确归因于某一个 metadata load。

原始数据：
[探针、逐 rank 结果、相同 workload 与 artifact hash](../benchmarks/dispatch_probe_20261010.json)。
native 源码 `67b146f`；artifact `build/communication_phases/borrow`。
五次有效启动均两机 exit0；中间一次跳过拷贝的 runner 被 CLI 参数检查拒绝
（不能在不做数值验证时要求 kernel profiler），修正 runner 后重跑；
失败启动没有进入 GPU 测量，也没有计入统计。

### 第二轮：并行化路由筛选，而不是更换 RDMA 方向

同一 native artifact `8666ebb`，细粒度探针关闭，首轮结果：

| 跨机协议 | RX 后的机内转发 | 8K ms | 20K ms | 8K / 20K 平均卡 GB/s |
|---|---|---:|---:|---:|
| token Get | 原串行筛选 + bulk | 6.290 | 14.367 | 10.68 / 11.53 |
| token Get | warp 并行筛选 + bulk | 2.630 | 5.939 | 24.61 / 27.79 |
| pack + packet Put | 原串行筛选 + bulk | 6.282 | 14.575 | 10.64 / 11.38 |
| pack + packet Put | warp 并行筛选 + bulk | 2.793 | 5.977 | 23.42 / 28.13 |

所有组都是 384 threads、Math208 / Control88、64/48/19/1 的借用布局，
264 个机内转发 worker。相同 buffer/ring 几何，相同 32 步随机路由。
这是纯 Dispatch 的有效 payload 带宽，**不是融合 GEMM kernel 的加速比**。

并行筛选的改动很小：

1. packet 仍由原静态 worker 负责，不引入动态任务队列。
2. 每轮 32 个 lane 各检查一条 route；使用 ballot 得到命中本 rank 的集合。
3. 命中 lane 预取 location/gate，经 shuffle 广播；仍由 lane0 使用原来的
   4KiB G2S→S2G bulk 中转完成该行 payload。
4. 原 X-ready、padding 到齐计数、packet consumed_total 与 generation/credit
   协议不变。没有新增 shared staging 或常驻通信 payload buffer。

同 artifact 中仅改变这一模式，Dispatch 延迟降低约 58%，即约 2.4× 加速。
同协议、同静态 ownership、同 payload copy 的反事实对照，把主要差距定位到
**逐 route 串行 metadata 访问/筛选**，而非静态任务划分必然低效。

需要说明诊断 artifact 的边界：为在同一二进制内切换模式，保留了探针、
vector 与 bulk 分支，ptxas 报告 pull helper 有 28B spill stores / 32B spill loads。
原算法在该 artifact 中约 6.29/14.37 ms，比上文旧 artifact 的
5.99/13.75 ms 稍慢。不要把这约 5% 的差别算作正常实现回退或优化收益；
即使对照旧 artifact，2.63/5.94 ms 仍是约 2.3× 的明确改善。
生产迁移时应删掉诊断分支、检查寄存器，并重新测真实 GEMM 融合入口。

### Put 对照到底覆盖什么

跨机 token Get 替换为：本地 pack → 单 packet Put → 远端 signal import，
**机内仍是 destination-pull**。不是把 RX rail→expert GPU 一跳改成 push。
pack 的数据搬运和等待包含在 Dispatch 计时内，不是预先免费准备连续输入。

这里只为独立 Dispatch 复用 48 个暂时空闲的 Return owner warp 做 pack，
原 19 个 progress owner 导入 RX ready 并回传 credit；没有新增线程。
TX 预留同时等待 remote credit 和 local NIC completion，继续保留全部
RX fanout 消费和精确 generation。它是有真实 pack、credit 的协议消融，
但**不是可直接照搬到融合生产 kernel 的调度**：生产 Return 同时工作，
不能把这 48 个 warp 阻塞在整个 dispatch pack 循环里。

在这种相对有利的独立阶段条件下，Put 也没有显示明确优势：未修筛选时基本
无改善，修好后 8K 略慢、20K 近似持平。不能据此断言所有配置中 Get 优于 Put；
能得出的工程决策是：**现在不优先切跨机 Put，先保留 Get 修转发控制路径。**

后续顺序：清理并迁移 cooperative route filtering；测真实 GEMM/Return 重叠；
再考虑紧凑的目的 rank 专属 route descriptor、bulk 多 staging/跨行流水，
以及保留 Get 时单独测试 RX rail 的机内 push/fanout reuse。后者与跨机
Get/Put 是两个独立决策，不能由本轮 Put 结果替代。

### 反向复测与最终数字

顺序：Get原版 → Get并行筛选 → Put原版 → Put并行筛选 →
Put并行筛选 → Get并行筛选 → Get原版。
除 Put原版一次，其余取两轮 p50 的均值：

| 方案 | 8K ms | 20K ms | 平均卡 GB/s（8K / 20K） |
|---|---:|---:|---:|
| Get，原筛选 | 6.299 | 14.411 | 10.63 / 11.47 |
| Get，并行筛选 | 2.626 | 5.918 | 24.69 / 27.86 |
| pack+Put，原筛选 | 6.282 | 14.575 | 10.64 / 11.38 |
| pack+Put，并行筛选 | 2.797 | 5.985 | 23.40 / 28.08 |
| 前轮 V3，同 workload | 2.096 | 4.931 | 32.08 / 33.60 |

Get 并行筛选相对本轮原筛选约快 2.40× / 2.44×，但仍比 V3 的独立 Dispatch
慢约 25% / 20%，不宣称已追平 V3。Put 相对改进后的 Get，8K 延迟高约 6.5%，
20K 约 1.1%，后者应视为基本持平。独立计算的带宽 p50 在 20K 顺序略有不同；
这点小差别不构成选择 Put 的证据。

七次 formal 启动均两机 exit0，使用同一 library hash。
672 个 rank/sample 输出逐位一致，各方案校验输出 hash 完全相同；
所有 token/route/gate trace 与前轮 V3 对照一致。
每个计时样本的跨机 payload 与路由预期一致。616 次节点时钟观测全为
1620MHz；第一轮探针另有 436 次观测，也全为 1620MHz。
测试结束后两机 16 张 GPU 均 0MiB。

[第二轮逐 rank 原始结果与复测归档](../benchmarks/dispatch_forwarding_20261010.json)。
源码与 runner 保存在上述实验分支，不合并实验开关到生产主路径。
复现第二轮（在实验 worktree 中，label 需唯一）：

```bash
/opt/.venv/bin/python benchmarks/build_communication_phases.py borrow \
  --output build/communication_phases_v2
/opt/.venv/bin/python benchmarks/run_dispatch_diagnosis.py bulk coop_bulk \
  --artifact build/communication_phases_v2/borrow --tag get_new --master-port 32130
/opt/.venv/bin/python benchmarks/run_dispatch_diagnosis.py bulk coop_bulk --push \
  --artifact build/communication_phases_v2/borrow --tag put_new --master-port 32132
```

第一轮在 `67b146f` native artifact 上使用 runner 的 `--profile`，
模式为 `bulk skip vector skip_local skip_remote`；跳过拷贝三组不做数值验证。
后续 native 修改使代码布局不同，不应把重编译的新库冒充第一轮旧库。

## 第三轮：pull staging 流水与 ready 发布

实验分支 `experiment/pf-dispatch-pipeline-20261010`，native `15689c8`，
runner `1adadcb`。跨机仍为 Get，384 线程、264 个机内 pull worker、
208/88 寄存器配额、packet/ring 容量均不变。

新 pull helper 把本地行的地址准备也分摊给整个 warp，远端命中 lane
提前准备目标地址、gate 和 x_full 指针。lane0 负责 bulk 指令，命中 lane
负责最终 gate/ready 发布。**这不只是增加 copy buffer 的单变量修改。**

G2S→S2G 后，只需等 shared read 完成便可复用 staging；但发布 ready
仍必须等 global write 完成。批量方案在一批有效行发出后统一等全部写完，
再通过 warp join 和 release atomic 发布 ready。整个 packet 的所有写出和
发布结束后才增加 RX consumed_total，不能提前归还 credit。

第一组保持每 warp 4KiB payload staging 不变。两级时拆成 2×2KiB；
增加的 barrier 落在原 Shared 的对齐 padding 中。五个候选使用同一个库，
下面是各一次完整 prime32/warm8/sample24 的 p50，不择取最快样本：

| 模式 | staging | ready 发布 | 8K Dispatch ms | 20K Dispatch ms |
|---|---|---|---:|---:|
| `coop_bulk`，前轮对照 | 1×4KiB | 逐行 | 2.672 | 6.011 |
| `pipe1_row` | 1×4KiB | 逐行 | 2.513 | 5.782 |
| `pipe2_row` | 2×2KiB | 逐行 | 2.536 | 5.875 |
| `pipe1_batch` | 1×4KiB | 批量 | 2.467 | 5.788 |
| `pipe2_batch` | 2×2KiB | 批量 | 2.508 | 5.832 |

单级新 helper 相对前轮对照减少约 5.9%/3.8% 延迟。两级小 segment 没有
额外收益；批量发布在 8K 略好、20K 持平，不能声称已找到决定性流水瓶颈。
Return 维持约 5.65/13.27ms。此时仍落后 V3 的 2.096/4.931ms。

480 个 rank/sample 的输出逐位正确，所有输出 hash、32 步 workload 与
前轮 V3 相同。440 次节点时钟观测全为 1620MHz。五次正式启动均两机 exit0。
第一次启动因端口 32200 被占用而未进入计算，清理该次启动的远端 rendezvous
进程后改用 32420–32424；该失败启动不计入结果。

四个新 pull helper 的 ptxas stack/spill 均为零。整个诊断入口和 GIN
Get/Put 仍有静态 spill，不能据此宣称整个 kernel 无 spill。
[原始结果及验证归档](../benchmarks/dispatch_pipeline_20261010.json)。

### 完整 4KiB 双 stage

进一步复用被网络 warp 借走的 `pull[0]`：它原来仅保存 pending requests。
把 requests 移到 ExtraShared，两个活跃 pull warp 各获得 2×4KiB staging。
只增加 5KiB shared，包含 static 后为 227KiB；不增加 GM ring 容量。
math208/control88、线程数和 packet geometry 不变。源码 `2642394`。

| 同一 wide-staging 库内对照 | 8K Dispatch ms | 20K Dispatch ms |
|---|---:|---:|
| 单 4KiB，逐行 ready | 2.490 | 5.848 |
| 双 4KiB，逐行 ready | 2.314 | 5.589 |
| 双 4KiB，批量 ready | 2.338 | 5.523 |

完整 segment 的双级预取有收益，单纯把 4KiB 拆成两半不能替代它。
逐行版本相对同库单级减少约 7.1%/4.4% 延迟；两种 ready 粒度差约 1%，
当前单轮不选择性宣称哪种更优。相对 V3，逐行版本仍慢约 10.4%/13.3%。
三次启动均两机 exit0，正式测量间不运行编译。
[wide-staging 原始结果](../benchmarks/dispatch_wide_20261010.json)。

### 真实 GEMM：通信单测收益不能直接外推

随后在真实 FC1/SwiGLU/FC2/通信/combine 入口做同库对照；未开启训练缓存。
不是 copy-substitution，也没有阶段串行化。每项仍 prime32/warm8/sample24，
取每步最慢 rank 再取 p50，排除固定 workspace 与权重预处理。

| 模式 | 8K kernel ms | 20K kernel ms |
|---|---:|---:|
| 原始串行筛选 | 12.211 | 24.621 |
| warp 并行筛选 | **10.770** | **23.170** |
| 新单 4KiB helper，逐行 ready | 10.812 | 23.261 |
| 双 4KiB，逐行 ready | 10.773 | 23.270 |
| 双 4KiB，批量 ready | 10.938 | 23.566 |

并行筛选让完整 kernel 延迟下降约 **11.8%/5.9%**。但本轮继续优化独立
Dispatch，并没有获得明确额外的融合收益；批量 ready 还略有回退。
这说明阶段单测不是当前融合入口的线性性能模型，不能把 Dispatch 缩短的
时间直接从完整 MoE 时间里扣掉。可能涉及计算/Return 已覆盖这段时间，
以及行发布时序变化；未采动态 stall 数据，不给这些因素分配耗时百分比。

因此选择将 **并行筛选单独提取到干净借用分支**，不引入额外 staging、
探针、payload-skip、vector 或运行时消融开关。其它候选保留在实验分支供复现。
不能用实验双 buffer 的 2.31/5.59ms 作为这个精简候选的 Dispatch 数字。

480 个 rank/sample 全部正确，各模式输出 hash 逐位相同；workload 也完全一致。
最大 relative-L2 为 9.24e-5 以下。444 次节点时钟观测全为 1620MHz。
五次正式启动均两机 exit0。最初 32440 端口冲突的启动未进入计算，
清理该次 rendezvous 后改用预检空闲的 25020–25024；失败启动单独归档。
SASS 保持每 K64 四条 N256 WGMMA 后统一等待，新 helper 均无 ptxas spill。
[融合入口原始数据](../benchmarks/dispatch_fused_20261010.json)。

## 精简候选：冻结旧库与新库复测

代码从借用版 `9123f30` 单独提取，提交 `8c47a46`，只有
`network_workers.cuh` 的 58 行增加、16 行删除。通过原来的
`WEFTMOE_PF_BORROW_ALL_CTA` 编译条件启用，无新增运行时开关。
跨机 Get、Return Put、packet/ring 容量、warp 数、208/88 配额均不变。
ready 在行 bulk write 完成之后发布；packet 全部 fanout 结束后才归还 RX credit。
没有引入动态任务领取。

新库在 `../WeftMoE-experiments/dispatch-optimized-20261010/build/optimized/borrow`；
对照直接使用借用实验中冻结的 `borrow-warp-20261010/build/borrow/gemm88`，
不是重新编译后标成旧版本。原库源码 `9123f30`，新库源码 `8c47a46`；
launch manifest 的 source_commit 属于启动脚本 worktree，不能混作这两个
native artifact 的源码版本，library/adapter hash 另外记录。

按旧→新→新→旧串行运行，两轮 p50 取均值。正式测量期间无编译。
EP16、Flash H4096/I2048/E256/K6、SM1620MHz、真实随机 token/route、
prime32/warm8/sample24，口径与前文一致。均为 inference，不缓存训练中间结果。

| 实现 | 8K kernel ms | 20K kernel ms | 8K eager ms | 20K eager ms |
|---|---:|---:|---:|---:|
| 冻结旧借用版 | 12.211 | 24.828 | 12.423 | 25.120 |
| 精简并行筛选 | **10.919** | **23.505** | **11.192** | **23.737** |
| kernel 延迟降低 | **10.6%** | **5.3%** | — | — |

逐轮 kernel：旧 12.195/24.914、12.228/24.742ms；
新 10.924/23.521、10.914/23.488ms。精简库比上面的多模式实验库略慢，
不追逐这一小差别带来的编译布局影响；以这个实际保留版本的数据作为最终结果。

四轮的完整 token/route/gate trace 相同，也与阶段消融的 trace 相同；
384 个 rank/sample 正确，旧新输出 hash 逐位相同，workspace 字节数完全一致。
364 次节点时钟观测全为 1620MHz。另以 B128、bias8 做 8K/20K 各 3 步
强偏斜验证，共 96 个 rank/sample；最大 relative-L2 7.83e-5 以下，
84 次节点时钟观测全为 1620MHz。五次启动均两机 exit0，结束后全部 GPU 0MiB。
压力测试只检查正确性/slot 复用，不把其 3 个样本作为正式性能结论。

资源审计：新 pull 函数 stack/spill 为零；入口 spill store/load 56/48B、
Get 252/384B、Put 136/276B，累计 stack336B，与旧借用版这些指标相同。
shared memory 仍 222KiB，未增加显存。每 K64 仍四条 N256 WGMMA 后统一 wait。
单机 PF kernel 的资源统计及 normalized SASS hash 与旧库完全相同
（`bf19584d6850f36af1e0506f91d5e4741c6b858f89ed0a39a346c9291f7204ce`）。
本轮未把双机结果外推为更多节点或其它模型 shape 的性能保证。

- [冻结旧库/精简新库逐轮数据](../benchmarks/dispatch_clean_20261010.json)
- [B128/bias8 复用压力验证](../benchmarks/dispatch_clean_stress_20261010.json)

### 精简候选阶段的边界

尚未追平 V3 的独立 Dispatch。实验双 4KiB stage 把差距缩至约 10%/13%，
但完整 MoE 没有随之加速，因此未为这项单测收益增加正式实现复杂度。
精简候选没有单独加入分阶段诊断入口，不能用双 stage 的带宽数字代替它。
也没有本轮 V3 真实 GEMM 的同口径结果，不据此声称完整 PF 已击败 V3。
下一步更应分析融合入口的 FC1/FC2 等待、Return 与 admission 时序，
而不是继续仅凭独立 Dispatch 数字切换 Get/Put 或增加 staging。
后续独立 Dispatch 已超过新测 V3，但完整 kernel 仍未获益，见本文开头的
[最新 fanout 消融与融合探针](dispatch_ready_fanout_20261010.md)。
