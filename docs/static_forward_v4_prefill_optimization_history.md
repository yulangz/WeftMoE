# V4 长文和多机优化历程

更新日期：2026-09-30。本文整理 V4 从早期静态 full-arena 到有界 ring、再到
GA+FZ 的有效优化过程，解释每一步解决了什么、收益有多少，以及哪些看似合理
的尝试没有保留。它是工程复盘，不替代逐次实验日志，也不把所有历史候选描述
为当前代码。

最终 inference 在主测形状下已超过冻结 V3；训练 Forward wall 尚未超过 V3。
单机短文在通用 ring 上的后续优化、专属 decode 和 SGLang 接入另见
[decode 优化历程](static_forward_v4_decode_optimization_history.md)。

## 测量口径和证据边界

- 主矩阵：H2048、I512、E64、top-k4、BF16，单机 EP8 为一台八卡，双机 EP16
  为两台各八卡。本文 T 均为每 rank token 数，8K=8192，20K=20480。
- 2026-09-23 及之后正式数据使用 L20X、SM 1620 MHz；显存时钟观测而非锁定。
  通常为 10 warmup、20 samples、每 sample 20 次 graph replay，rank-max p50。
  正式计时节点没有并发编译或其他 GPU workload。
- A：同条件正反序或交错复测；B：同条件相邻或历史冻结构建对照；C：组合
  里程碑，不能拆分归因；Q：少样本快筛或存在构建干扰，只用于解释选型过程。
  后续版本通过完整验收，不会使它的早期 Q 数据自动升级为正式单项收益。
- 不同表项的百分比不能相加。B 是 source-window 容量，D 是 contribution
  窗口深度；改变 B、D、packet Q、lane 数须同时报告，不能冒充纯代码收益。
- graph、eager wall、主 kernel、阶段 span、纯 GEMM 和链路 benchmark 是
  不同口径。阶段 span 包含等待且可以重叠，不能相加还原总耗时。

原始证据按时间保存在 [单机记录](static_forward_v4_optimize_record.md)、
[早期多机记录](static_forward_v4_multinode_record.md)、
[prefill GEMM 记录](static_forward_v4_prefill_record.md) 和
[ring 实验记录](static_forward_v4_ring_record.md)。下文保留实验标签，便于在
原记录定位配置、原始日志、构建 hash、正确性及 sanitizer 范围。

## 最终结果和最初的问题

GA+FZ 的完整正式验收为 `33d99a5`，B512/D8；以下为同轮冻结 V3 对照，单位 µs：

| T | EP8 V3 | EP8 GA+FZ | 延迟降低 | EP16 V3 | EP16 GA+FZ | 延迟降低 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300.277 | 100.910 | 66.4% | 682.034 | 236.034 | 65.4% |
| 64 | 370.696 | 135.406 | 63.5% | 824.120 | 286.896 | 65.2% |
| 8192 | 1489.090 | 1246.600 | 16.3% | 3355.780 | 2893.170 | 13.8% |
| 20480 | 3295.760 | 2937.720 | 10.9% | 7102.290 | 6863.170 | 3.4% |

这不是静态调度首版就取得的结果。1620 MHz 下，早期 full-arena 单机 V4 的
8K/20K 是 1600.65/4767.00 µs，比 V3 慢 7.6%/44.9%。首个单机 ring 更慢，
8K/20K 达到 5265.20/13042.8 µs；首个双机 ring 的两样本快筛达到
72605.2/175901.0 µs，且后来还发现原退休原子的作用域缺口，不能把它算作
正确实现的正式基线。

第一条经验是：静态分工只解决领任务成本。元数据访问、分工碰撞、ready 查询、
中间复制、窗口供给和 GEMM 粒度没有处理好时，静态 ring 完全可能更慢。

## 静态 full arena 的早期修正

2026-09-22 复审先处理三个问题：FC2 原来每 lane 写不同 token 行，产生分散
peer store；整个 invocation 先 FC1 后 FC2，工作集过大；M128 几何已经生成，
却仍沿用较小 Notify tile 的线性 readiness 扫描。

| 改动 | EP8 8K | EP8 20K |
|---|---:|---:|
| 复审前 | 1973.02 | 7161.11 |
| FC2 相邻八个 lane 合并写连续 128 bytes | 1706.21 | 6412.08 |
| 增加 32-Mblock 静态 FC1/FC2 wave | 1564.15 | 4869.70 |
| M128 readiness 直接索引 | 1482.39 | 4580.19 |

该表是当日历史相邻消融，未纳入后续 1620 MHz 锁频合同，只能在表内理解
24.9%/36.0% 的累计改善，不能跨表与锁频 V3 相减。提交为 `b4538e0`。

早期多机 full-arena 把通信收敛到固定 rail owner，FC2 直接写紧凑 Return TX，
并融合 route 规划及计算。随后删除 Combine 后不必要的跨机 barrier、同时裁剪
GIN context 分配，EP16/T64 历史值 191.47→173.31 µs，降低 9.5%。这是同批
组合，且属于锁频前记录；不能分别归因给 barrier 或 context。

这条路线的问题也很明确：完整调用容量预留，Dispatch/Return 仍有阶段边界，
payload 通信与 FC 没有充分流水。它作为正确性和性能参照保留，最终容量方案
重新回到逐 slot ring，不能把当时暂用 full-arena 解释成 ring 不必要。

## 先改善 GEMM 粒度和 epilogue

早期 V4 packed FC1 的物理 N64 只产生 32 列 S，I512 需 16 个 N-tile；
FC2 N128 也需 16 个。prefill 改用物理 FC1 N128、FC2 N256，变成 8/8 个
N-tile，并为完整块做合并写回。保留窄入口处理短序列，避免全量使用宽 tile。

1620 MHz，旧 V4→宽 prefill 与 FC1 合并写回的组合结果，单位 µs：

| 拓扑 | 8K | 20K |
|---|---:|---:|
| EP8 | 1601.93→1504.50 | 4740.10→4607.09 |
| EP16 | 3860.00→3725.31 | 9535.05→9146.10 |

同轮的安全 B-staging 修正自身只有 −0.22%～+0.05% 变化，不算加速；加宽和
合并写回一起启用，不能拆分它们的收益。详见 prefill 记录的“分步收益与证据边界”。

之后 AF/AS 独立比较 packed gate8/up8 与 separated gate/up，固定有效 S 宽度、
K、stages、cluster 和舍入合同。在有效 S-N64、K128、stages2、world8 均衡
几何下，8K/20K 的 FC1 为 packed 329.152/782.112 µs，separated
386.400/925.792 µs；FC1+FC2 chain 为 483.920/1152.640 对
539.504/1294.720 µs。该实验是单 GPU 重放专家几何，不含 EP 通信，也不是
原封不动的 V3 kernel。它支持保留 packed 布局，不支持把全部 V3/V4 差距归于布局。

纯 GEMM 更宽的 N256 在融合 ring 中并未产生稳定收益，AC/AL/EM 等尝试撤回。
纯算术更快不等于工作供给、同步和写回共同作用下更快。

## 单机 ring 从供给不足恢复吞吐

首版已经有精确计划和逐 slot 生命周期，但热循环仍反复读取 peer metadata、
计算地址和检查同一 reservation。下面列出有效演进，不跨配置累计百分比。

| 实验 | 保留的改动及原因 | 当时的收益和证据 |
|---|---|---|
| A | 窗口间旋转静态 residue，避免每窗都从同一 cluster 开始 | B256，8K 5269.15→5217.41 µs，约 1%；仅 B 级，不称显著收益 |
| B | 在本地 RingRow 冻结 source、route、gate，移走热路径逐行 peer 元数据读取 | B256，8K/20K 5217.41/12929.4→3421.16/8330.07 µs，降低 34.4%/35.6%；T1 反而变慢 |
| C | 按 window、expert 静态并行构建行目录 | T1/T64 123.939/242.914→114.920/200.846 µs；20K 基本不变 |
| E | control lanes 并行 snapshot source epoch，后续只查本地 | 相对 C，四档改善约 1.9%–2.7%；相邻 B 级 |
| G | 用单 issuer 双 staging 保持预取，替代未通过 racecheck 的双 issuer | 这是安全流水的落地；D2 初测有编译干扰，不能把它列为正式独立加速 |
| K | 每 source/window 发布 reservation head，在 destination 本地查询 | B256/D4，8K 3088.12→3056.75 µs，约 1%；尚无独立反序证据 |
| L | 预绑定物理 slot、generation、窗口源地址，移出运行时 64-bit 除余；Reduce 汇总窗口 credit | B256/D4，8K/20K 相对 J 为 3088.12/7550.21→2437.51/5931.10 µs；8K 相对直接前驱 K 降低 20.3% |
| N | 完整 TMA store wait 后直接 release 发布，删掉重复 SC fence | 同节点 B256/D4，8K 2426.69→2199.95 µs，降低 9.3%；保留完整 wait 和可见性链 |
| O | 唯一 admission owner 改为 warp 并行检查和发布 | T1/T64 125.853/180.406→110.208/162.150 µs；8K 仅改善 1.9%，20K 无收益 |
| AD | 本地 FC2 直接写 source contribution，删除 C→contribution 第二次搬运 | 正式 EP8，8K/20K 2156.34/5303.55→1682.80/4051.05 µs，降低 22.0%/23.6% |
| AP | top-k4 缓存四个有序 contribution 地址并展开，其他 K 保留通用循环 | EP8 8K/20K 1682.80/4051.05→1659.53/4005.92 µs，约 1%–1.5%；这是 K4 特化 |
| AT | 同 Mblock 内八行 Pull 合并 full wait 和 X-ready 发布，仍为单 issuer 双 staging | 同 B512/D4，8K/20K 1631.58/3968.52→1454.03/3585.28 µs，降低 10.9%/9.7% |
| AV | 每窗口缓存已成功 acquire 的 source reservation，批内只查一次 admission | B512/D4，8K/20K 1454.03/3585.28→1293.96/3121.93 µs，降低 11.0%/12.9% |

来源为 ring 记录对应 A–AV 节。J 删除静态 Reduce 不消费的动态 completion
计数，8K 3091.32→3088.12 µs，属于清理而非性能收益。

这里收益最大的不是减掉正确同步，而是避免重复做已经确认过的工作。
缓存 acquire 的前提是：当前 owner 尚未发布/消费完自己的行，当前窗口就不
可能退休。不能把这个证明扩大到下一窗口，更不能缓存失败的确认。

窗口也不是越大越好。E 阶段 B128/256/512/1024 的 8K 分别为
5601.93/3312.25/2886.87/3273.16 µs；AT 阶段 D8 对 20K 有约 2.1% 改善，
8K 没有改善，D16 反而更慢。后续采用 B512/D8 是具体配置的取舍，不是无成本算法收益。

## 多机先解决静态映射和元数据访问

首个网络 ring 已经实现 overlap：profile 中 FC1 在最后一批 Dispatch Put
之前很久就启动，Return 也与 FC2 同时推进。瓶颈不能继续笼统归因于“全节点
Dispatch barrier”；当时主要待处理的是 packet 控制和搬运供给。

| 实验 | 改动 | 收益和限制 |
|---|---|---|
| Q | 按连续 C slot 合并 ready 查询和退休，并修正多机退休原子为 system scope | EP16/8K 快筛约 72.6→53.7 ms；旧对照有作用域缺口，Q 级诊断，不作为交付版本加速比 |
| S | 按窗口内 packet ordinal、lane、source 交错静态 ownership，减少 worker 碰撞 | 同快筛 53.6→34.4 ms，约 35.9%；不是正式矩阵 |
| T/U | warp 协作读 route 目录，并把网络 lanes 从 2 增为 8 | T 约省 1.7%；U 快筛 33.8→10.7 ms，同时增加约 25.2 MB/rank 注册内存；正式 U 的 8K/20K 为 10.726/26.206 ms |
| Z | 将各 worker 真正拥有的 PullPacket 指针和 generation 预绑定到本地表，不再扫描全部远端 channel | 同 L32/B512/D4 快筛 11790.3→4766.67 µs；L32 只有消除扫描成本后才有效。后续 AB 正式 8K/20K 为 4716.07/11166.5 µs |
| AK/AM | 减小 Q 提高 Dispatch lane 利用率；payload 就绪后才查 NIC credit | Q32→Q8 的 8K 快筛 4757.62→4394.67 µs；AM 再到 4366.35，不能单独认定不足 1% 的变化。与 AP 合并后的正式 EP16 为 4405.93/10226.1 µs |
| AZ | 唯一 GIN issuer 使用 THREAD 独占资源模式，省去 GPU 共享队列锁 | 同 Q8/L32/B512/D4，8K/20K 4329.17/10196.3→4057.57/9462.10 µs；8K ABBA 重现约 6.3% |
| BA/BB | route 元数据单遍读取；Pull 每窗口只 acquire 一次已发布的整窗 admission | 相对 AZ，正式 8K/20K 到 3992.34/9406.51 µs，组合降低 1.6%/0.6%；T64 降低 4.6% |
| BD | FC2 ready 发布到发送 rail 的本地镜像，避免 TX 反复跨卡轮询 expert GPU | 正式 8K/20K 3992.34/9406.51→3798.20/8836.80 µs，降低 4.9%/6.1%；T64 略退化 |

AB 按包数安全上界裁剪目录，8K 每 rank 注册内存减少 5,211,648 bytes，
约 1.1% 快筛时间变化不算确定加速。R 把纯本 GPU ready 发布收紧为 device
scope，AU 使用满足单 Put 合同的 weak signal，两者都没有可辨认的端到端收益。
这些容量和合同简化值得保留，但不应包装成性能突破。

上述标签见 ring 记录 Q–BD。网络“独占”优化依据实际 devComm 为 GDAKI，
不是凭日志中的 RMA proxy 名称猜后端；该名称误读曾产生错误归因，后来已撤回。

## 分包和拓扑必须分别调

Dispatch 希望少量去重 token 能激活足够 lanes；Return 希望摊薄提交成本。
两个方向共用 Q 并不合适。

| 实验 | 对比和结果 | 结论 |
|---|---|---|
| BF | BD 后 Q8/Q8→Q16/Q16，8K/20K 3798.20/8836.80→3533.15/8181.34 µs | 长文降低 7.0%/7.4%，T64 却慢约 14.3% |
| BH | 再改 Dispatch Q8、Return Q16，8K/20K 到 3420.79/7937.91 µs | 相对 BF 再降 3.2%/3.0%，8K 注册区从 71.95 MB 降至 55.23 MB |
| BO/BR | source group 按节点对齐，减少同 Mblock 混合本地和远端供给；采用 G8/B1024/D4 | 正式 8K/20K 为 3280.02/7627.35 µs；相对 BH 约降 4.1%/3.9%，但 G 和 B 一起改变，不能全归给分组 |
| BW | 删除无读者的 route 反向目录和对应交换量 | 8K/20K 每 rank pool 各减 786432/1966080 bytes；时间只小幅变化，不称显著加速 |

BR 与 V3 的交错对照中，8K 已快约 2.4%，20K 仍慢约 7.4%。这一阶段尚未
完成目标。其后转为 B512/D8 的容量收益必须与代码收益分开，不能直接拿
G8/B1024/D4 的历史时间作同配置控制组。

BO 首轮 host 分配使用 G8、launch 却覆盖回 G16，尽管 hashes 通过，几何合同
仍不成立。该轮所有时间和内存结论作废，修正后重新验收。类似地，早期 CT 的
自适应 Q 被有效 B 裁剪后的早退绕过，候选实际上未生效；CW 修复并明确输出
T1/B1/Q1、T64/B64/Q8 后才重新计时。

## 继续削减规划和 Return 中间工作

| 实验 | 保留的改动 | 正式收益和证据边界 |
|---|---|---|
| CA/CE/CF/CH | 保留编译期 tile 常量；已 drain 的 ring 仅用一个 context 终端会合；规划仅 fence 实际 Put 的 context；warp 并行计算精确 consumer counts | 同 B512/D8，8K BW 3146.08→CH 3127.42 µs；20K 7312.64→7309.75，近乎持平。不能把容量切换收益归给此组合 |
| CL | RX 从 immutable plan 推导每代精确 retire target，删掉线上 header | EP16 8K/20K 3127.42/7309.75→3072.62/7149.84 µs，降低 1.75%/2.19%；每包少 128 bytes，注册区少 131072 bytes/rank |
| CN/CO | 先取得 TX 双 credit 并预留，再让 FC2 直接写 Return TX；删除无读者 C payload | 与 Q32 组合，8K/20K 到 3061.54/7099.11 µs，只降 0.36%/0.71%；T1/T64 退化。相对同 Q32 的 CN，CO 精确省 132 MiB/rank |
| CP/CQ | staging 完成后让全部 CTA 线程参与计划；复用 warp packet-prefix scan | 8K/20K 3061.54/7099.11→3007.87/6989.36 µs，组合降低 1.75%/1.55%；不能把 planning span 的缩短全算为关键路径收益 |
| CV/CW | warp 并行检查多个 source rail 的 TX reservation；短窗口按实际负载选 Return Q | T1/T64 290.987/414.488→276.125/333.984 µs，降低 5.11%/19.42%；长文持平 |
| DC | 通用 ring 按拓扑选择短/长 GEMM profile 阈值，单机 512、多机 2048 | EP16 T2048 约省 0.7%，T4096 约省 1.7%，反序复现；主矩阵 1/64/8K/20K 的入口本来未变，不能给它们重复记收益 |
| DS | Return 直接写 source 的 expert-major contribution 窗口，删 Return RX payload 和 scatter | 20K 真正相邻 7008.14→6986.62 µs，仅 0.31%；总 pool 减 33,226,752 bytes/rank，主要价值是减少复制和内存 |

来源为 ring 记录 CA–DS。这里的“直接写”不是无条件覆盖：FC2 必须等 Return TX
reservation；跨机 contribution 必须属于 source 已预留的窗口。

DS 特别重要的区分是：packet 控制 slot 在发布 route-ready 后可以退休，但
contribution payload 要等全部 token 的 Reduce 消费完才能复用。让 Reduce
直接占有 RX payload 的 AQ/CM/DB 等尝试没有稳定收益，也会延长网络 slot 的
占用，最终没有保留。

## 最后一段是小收益和严格生命周期修正

| 实验 | 改动 | 可归因的收益 |
|---|---|---|
| ES | 用 TX owner 的 completed head 和必要时的 GIN flush/wait 替代逐包本地完成 counter 通知 | 清理版 EP16/20K 正反序约省 0.64%/0.74%；8K 持平。仍检查本地 NIC 完成，未删除保守 credit |
| ET | 在 DS/ES 新协议上重新消融 Return Q32→Q16 | EP16/8K 2940.80→2905.63 µs，省 1.20%；20K 正反序省 0.42%/0.50%；少约 16.7 MB/rank |
| FJ2 | prefill FC1 用三级 TMA，FC2 两级；复用 192 KiB arena，补齐跨 GEMM multicast 退休及 scratch 边界 | 修正后 EP8/8K 1264.64→1254.23 µs，省 0.82%；EP8/20K 持平；EP16 新增收益很小 |
| FT | 完整 M128 的 FC1 S 用 TMA 写回，复用已退休 B staging | EP8/20K 正反序约省 0.46%/0.33%；8K 首轮约 1%，反序近乎持平，不能报稳定 1% |
| GA | RX 合并连续已退休 slot 的 ACK，TX 查询 consumed prefix | EP16/20K 单独反序 6888.60→6860.29 µs，约 0.41%；8K 仅 0.14%–0.24% |
| GA+FZ 集成 | GA 网络 credit 加 FZ 训练 checkpoint 分工，完整集成验收 | 相对 FT，网络 T1/T64 正反序稳定省约 12.2%/9.6%；长文新增收益很小。FZ 不改变 inference SASS，不能把 inference 收益归给 checkpoint |

对应提交：ES `14b62fd`，ET `abd1ebd`，FJ2 `e56ad94`，FT `b31a2c4`，
GA+FZ `33d99a5`。数值来源见 ring 记录各同名节及组合验收表。

GA 保留了用户要求的保守方案：**远端 consumed prefix 和本地 NIC completed
head 同时满足才复用 TX**。即使因果上远端消费意味着数据已经送达，也不绕过
GIN 本地源缓冲区复用合同。这里只批量化确认成本，没有把本地 credit 删除。

## 踩过的坑和没有保留的方向

### 生命周期错误不能用多数测试通过抵销

- F 的双 issuer TMA 在最小复现中也有 WAW；没有按“工具误报”忽略，而是
  撤回，改成单 issuer 双 staging。
- Q 暴露本地 device-scope 和远端 system-scope 原子混用；全部相关参与者的
  scope 都必须覆盖对方，仅修改一端不够。
- FJ 的每阶段 SMEM 总量都不超 192 KiB，仍不代表跨 CTA 安全。较快 CTA 的
  FC1 第三 A stage 会覆盖较慢 CTA 尚在用的 FC2 B/scratch。racecheck 虽有
  rank 报 0 hazards，应用数值已经出错，就不能通过。FJ2 等旧 empty barrier
  完成并移动 scratch 后重新验收，不能沿用有问题的 FJ 那个约 2% 的收益。
- TX 不得在 NIC 读完前覆盖；RX 控制 slot 退休不等于下游 contribution 消费完。
  payload、目录和控制状态应各自证明生命周期，不能笼统说“这个包已经 done”。

### 增加并行度或减小同步不一定更快

- 没有 Z 的本地 PullPacket 表时，L8→L32 增加全 channel 扫描成本，反而更慢。
  后来的 L64、拆分发包 owner、共享 contexts、通信专用 CTA 也未证明稳定收益。
- AT 的八行批量发布有效，早期 Y 的八行批次却退化；batch 的边界、ready
  发布位置和有效 owner 分布不同，不能只根据“都是八行”推断结果。
- I 把 Reduce warps 借给搬运，EP8/8K 变成 5613.32 µs，比当时对照慢约 82%。
  Reduce 本身有 payload 和地址处理成本，并非免费尾部操作。
- H 的 relaxed poll、成功后 acquire 与 nanosleep 组合退化约 12%–13%；
  这不是删除必要 acquire 的理由，也不能把组合退化全归给其中一项。
- P 的 warp 向量 Pull 达 6636.13 µs，超过当时 TMA 路径三倍；V/W 的网络
  向量 packet copy 同样明显退化。局部 instruction 更少不等于搬运吞吐更好。
- D 早期整窗 admission 组合退化约 10%，同时引入串行 peer epoch 查询；
  后来本地 snapshot、并行准入、成功 acquire 缓存有效。不能据 D 否定所有
  整窗协议，也不能隐去 D 中同时变化的其他因素。

### 独立微基准成功不等于融合算子成功

GM 无损 Return codec 独立测得约 58/63 GB/s，预计减少约 18.55% Return
字节，但正式融合结果 8K 2885.12→5075.19 µs、20K
6872.75→12456.00 µs，退化 75.9%/81.2%，已撤回。编码、解码、失去直接
写 contribution 和资源竞争均是待分离因素，不能把退化武断归于单一项。

类似地，更宽 FC1、更深 stage、FC2 权重 multicast、直接跨机 Return、多种
owner/context 数量搜索没有稳定净收益时均不保留。冻结 patch 留在 build
归档用于复现，不把候选宏和分支积累进生产路径。

### 测量错误也要记录

- EP8 曾误测为两机各四卡，已降为额外实验；本文 EP8 始终指单机八卡。
- V3 binary 本身包含 V4 entry，必须显式关闭 V4 才是 V3 对照。
- 错误的 token 环境变量曾让名为 20K 的日志实际跑 8K；以结果中的 shape
  为准，失效数据保留但不纳入结论。
- early AF 的均衡 route 公式与主夹具不同，AS 修正后保留原证据范围。
- RMA proxy 日志不等于 GIN 后端；核对实际 devComm 才确认 GDAKI。
- 开始时数值通过、hash 一致、sanitizer 零报告、完整生命周期通过和性能获益
  是不同门禁，任何一个都不能代替其他门禁。

## 训练相关收益单独记账

阶段一也接通了 opt-in ring training Forward，但它不是当前 RL rollout 替换
的 actor 实现。以下只作训练分支索引，详细配置、梯度和主核/wall 区别见
[训练记录](static_forward_v4_training_record.md)。

| 里程碑 | 有效改动 | 已记录的收益和边界 |
|---|---|---|
| M6g | Backward 使用独立 sender 配额，恢复 unified 配置 | 20K Backward 单机/双机约 5707/14571→4917/6874 µs，属于配置耦合修复 |
| M6h | 在已退休 FC1 B staging 合并 G/U checkpoint 写回 | 20K training main 单机约省 1.5%，双机约 0.7% |
| FM | SavedX 从 Pull 的第二次 S2G 移到 FC1 已就绪 A stage | 20K training main 单机约省 2.8%，双机约 3.1%–3.6% |
| FO | SavedX 目的地址移出 K 循环 | 单机 20K main 正反序约省 2.8%–3.0%；双机较小且有波动 |
| FR3 | 仅单机训练使用稳定 source 行序和 M32 SavedX TMA | 单机 8K/20K main 相对 FO 省 4.72%/6.06%；网络组合退化，未推广 |
| FS2 | 删除不参与 tape 分配的 ring-plan D2H 和 host 同步 | 单机长文 wall 正反序约省 1%–2%，双机持平；设备指令未变 |
| FZ | 轮换 checkpoint 保存 pair，并由两个 CTA 分担两半 X | 单机 8K main 约省 1.1%；双机指定容量下有约 0.8%–1.3% 收益，并非所有长度/容量都改善 |

最终 GA+FZ 训练 Forward wall 仍比 V3 慢：EP8 长文约 5.3%，EP16 即便
B1024/D4 也约慢 9.6%/14.4%。完整 Backward racecheck 门禁仍有已记录的
工具/原子报告限制，不能用 inference 的领先替代训练完成声明。

## 当前数据面和后续复用原则

单机由 destination 按 route pull source X，FC1 产生 S，FC2 直接写 source
contribution，source 稳定 Reduce。多机在这条节点内流程外增加按命中节点
去重的 Dispatch Put 和固定 rail 的 Return Put；它不是跨 NIC RDMA read。

通用 ring 的中间通信和计算 payload 有界，slot 用 generation、ready 和
consumed 保护在线换代；输入及 immutable plan 仍可能随 T 增长，所以不能说
整个进程显存与 token 数无关。fanout 和 Return 行数来自实际路由计划，RX
按代累计 retire target，没有运行时动态 expected-count 领用。

这些机制没有写死两节点，但端到端 GIN 数据面实测仅两节点；更多节点的
布局/规划检查不是更多节点的性能验收。H2048/I512/E64/K4 的最佳 Q、B/D、
GEMM tile 和 profile 阈值也不是其他模型的通用最优值。

复现实验时先核对实际路径、形状、容量和 clocks，再按原记录里的 frozen
binary/config 重放；未通过数值或生命周期门禁的快候选不进入性能比较。
