# WeftMoE Static Forward V4：静态任务与有界流水设计

更新：2026-09-23。本版整理已确认的设计，作为后续实现依据；**第 0 节描述主路径，
第 11 节跟踪实施进度，其余章节是目标合同，不表示 ring、通信/计算流水或 training
V4 已经落地。**
旧设计中的 source-push、完整 arena 作为最终容量方案、按 fragment 数维护通用
引用计数等选择，不再作为实现要求。历史实现和性能证据保留在：

- [单机优化记录](static_forward_v4_optimize_record.md)
- [多机优化记录](static_forward_v4_multinode_record.md)
- [prefill GEMM/epilogue 优化记录](static_forward_v4_prefill_record.md)
- [ring 实施与优化记录](static_forward_v4_ring_record.md)

## 0. 当前代码基线：已实现什么

默认 V4 仍保留单机/多机 inference-only 的静态 full-arena 实现，training 走 V3。
opt-in ring（`WEFTMOE_STATIC_RING_TOKENS=B`）已接通单机及多机
Dispatch→pull→X→FC1→S→FC2→C→Return→contribution→Reduce。多机按命中节点
去重发送 X，逐 packet ready/credit 推进，可与计算重叠；第 6.6 节描述当前
数据面。完整 EP8/EP16 数值测试已通过，但长序列性能仍未达标，不能当作最终
优化版。GEMM 独立消融和训练对接未完成，进展与测量见 ring 记录 M2/M3/S–U。

| 项目 | 默认 full-arena 路径（非 ring） |
|---|---|
| 单机 Dispatch | source staging 后，expert GPU 的固定通信 warp 通过 NVLink destination-pull |
| 多机 Dispatch | source 将完整 X/routes/gates 发给每个远端节点的同 rail GPU，包括未命中的节点；完成阶段 barrier 后再机内 pull |
| GEMM | FC1/FC2 静态 residue 分配，无动态 task claim；节点内 pull 与计算可重叠 |
| cluster TMA | 已使用 cluster size 2 的 multicast；V4 配对同 Mblock 的相邻 N tiles，共享 A |
| FC2/Return | FC2 push contribution；多机等节点内全部 FC2 完成后发送 Return，再等待网络阶段结束后 Reduce |
| 缓冲区 | canonical X/S 和通信区按完整 invocation 容量分配，没有应用层逐 slot 通信/计算 ring |
| inference launch | route planning 与主流程融合；正式 graph 基线每 rank 一 kernel |
| training | 保持 V3；新 tape/Backward 对接尚未实现 |

当前通常使用 132 个 384-thread CTA：256 math、32 TMA producer、96 pull/control
线程。此配置是基线，不是新流水必须保持的角色配额。

prefill 配置 `configs/static_inference_prefill_20260923/` 已有两个 GEMM entry：

- 短序列：M128，FC1 N64 / FC2 N128，K128，stages 3/3。
- 长序列：M128，FC1 N128 / FC2 N256，K128，stages 2/2。
- 该配置以 8192 tokens/rank 为入口阈值；其他配置未统一启用，不把阈值当成最终结论。
- FC1 宽入口已迁移合并写回；窄 N32 SwiGLU 输出保留直接写。
- epilogue 复用已退休的 CTA 私有 B staging，不复用可能被另一 CTA 提前 multicast
  覆盖的 A staging。

full-arena 历史基线的 graph rank-max p50（us），核心 1620 MHz：

| 拓扑 | tokens/rank | 冻结 V3 | prefill 配置 V4 |
|---|---:|---:|---:|
| 单机 EP8（1×8） | 1 | 299.78 | 64.50 |
| 单机 EP8（1×8） | 64 | 368.44 | 73.80 |
| 单机 EP8（1×8） | 8192 | 1487.58 | 1504.50 |
| 单机 EP8（1×8） | 20480 | 3289.72 | 4607.09 |
| 双机 EP16（2×8） | 1 | 680.49 | 169.99 |
| 双机 EP16（2×8） | 64 | 824.51 | 188.90 |
| 双机 EP16（2×8） | 8192 | 3355.23 | 3725.31 |
| 双机 EP16（2×8） | 20480 | 7105.66 | 9146.10 |

V3 是同日冻结对照，不是最后一轮 prefill 改动的交错重测。长序列差距仍在；
现有数据同时混合了布局、tile、调度和通信差异，不能证明 gate/up 的某种布局更优。
此表不是新 ring 的性能结果；详细配置、误差和证据等级以各优化记录为准。

## 1. 本轮确定的目标与边界

1. 保留静态 task ownership，删除动态队列领用、全局 claim cursor、task CAS。
   静态指本次 route plan 冻结后的映射，不是编译期固定 token 数。
2. 使用逐 slot、带 generation 的 ring；借鉴 MegaMoE 的 full/empty 计数协议，
   而不是移植其完整 scheduler。不能仅把完整 arena 地址取模就称为 ring。
3. 单机仍使用 destination-pull；多机跨节点用 GIN push 到 rail RX，目标 expert GPU
   再做节点内 pull。按实际 route 做节点级 X 去重，不命中某节点就不向它发送 X。
4. 跨机 Dispatch、节点内转发、FC1/FC2 要流水重叠；Return 也随已完成 Mblock
   推进，不等整个节点 FC2。数据面不使用每包、每窗口或每阶段的全节点 barrier。
5. X/S/C、网络 TX/RX、source contribution 分别管理生命周期。释放某一缓冲区
   不等待整条流水结束；后续需要的 metadata 必须仍有独立有效存储。
6. 支持长上下文 prefill 与后续 training；不以 capacity factor 截断合法 route。
   有界 transient buffer 满时背压，不丢 token。输入、输出、plan、training tape
   仍需实际规模的内存；真正 OOM 明确失败，ring 不等于无限总显存。
7. 单机/多机、decode/prefill 使用 specialization/profile，共享协议和数值模块，
   不复制四套 scheduler。gate/up 布局通过第 7 节独立 GEMM 消融决定，先不选边。
8. Dispatch/Return TX 复用先保留远端消费 credit 与本地 Put completion 双重检查；
   简化检查列为后续优化，不阻塞首版。
9. 协议参数化节点数，不硬编码两节点。当前实机主矩阵只有单机 EP8（1×8）和
   双机 EP16（2×8），均测 1/64/8192/20480 tokens/rank、核心 1620 MHz。
10. 本轮先完成 inference 流水；training 的保存、精确分配与 Backward 对接作为
    独立 milestone，不能把 inference 正确当成训练支持完成。

## 2. Route plan、窗口与静态任务

### 2.1 不可变的路由身份

每条有效 route 用 `(source_rank, token, topk_slot)` 唯一标识，另有 expert、
router weight 和有效位。重复 expert 的不同 top-k slot 仍是两条 route；只有
X 的网络传输可去重，不能去重掉 contribution。

plan 给出 source ordinal、destination logical Mblock/row、source contribution
位置及固定 combine order。canonical 行可沿用 source-rank round-robin；training
必须固定该顺序并保存，不在 Backward 重做原子分配来猜行号。

最终 FP32 combine 保持现有 `(expert, topk_slot)` 固定顺序，不改成到达顺序，
也不能擅自换成单纯 top-k slot 顺序。router weight 在 SwiGLU 阶段使用。

### 2.2 有界 source-token window

同一 token 的全部 top-k route 进入同一个窗口。窗口最多包含 G 个 source rank，
每 rank 最多 B 个 token；大 world 按固定 source-rank group 继续枚举窗口，
不让 G 随节点数无限增长。起始候选是 G×B 不超过 2048 个 source token：
EP8 可取 G=8/B=256，EP16 可取 G=16/B=128；这些是调优起点，不是性能结论。

所有 rank 使用同一窗口编号与顺序；不等长/空 rank 由实际 token counts 决定空区间。
窗口里的 route 按 expert 形成逻辑 Mblocks，不能把一个 Mblock 的未到齐行留给
未准入的后续窗口。一个 packet 也不跨窗口。

需要的几何是：

```text
count[w, source, expert]           // 当前窗口内的实际有效 route 数
R[w, expert] = sum_source count[w, source, expert]
mblocks[w, expert] = ceil(R[w, expert] / M)
```

这些计数从实际 route 遍历/分桶产生；整个 invocation 的 expert 总数不能代替
窗口计数。由 planning 阶段交换必要的窗口计数和紧凑目录，零项可以省略。
避免强制物化规模为所有窗口 × 所有 rank × 所有 expert 的稠密表。

窗口划分、route 顺序与 packet 边界在 payload 阶段前确定。descriptor 可按闭式
规则枚举，不要求全部物化；不能用运行时“当前 ready 多少就发多少”改变首版
分包边界。每个 channel 的准确 packet 数由实际非空分组及分包规则得到，而不是
仅用全局 route 总数除以 Q。

plan/紧凑目录还需让目标 GPU 枚举每个 (channel, sequence) 的实际 fanout 任务范围。
不命中的 worker 不扫描可复用 RX header 来临时发现任务；否则这些未被 fanout
计数覆盖的 metadata 读者也会影响 slot 释放。ACK owner 所需包头字段在返还 credit
前读完，之后不再引用该代 payload。

### 2.3 准入与无环背压

source owner 先预留自己参与窗口的 contribution 容量；destination owner 一次检查
并预留本 rank 该窗口全部 Mblocks 的 X/S/C，然后各自发布准入 credit。禁止在
destination 内占住部分 X/S/C 后等待本窗口剩余 plane。M3b 采用分级准入：发送方
等待自己的 contribution reservation 与 TX 双重 credit；有界 RX 可以预取尚未
获准计算的窗口，真正 pull 到 X 前才等待本 destination 的整窗口准入。因此无需
额外跨 NIC 交换每个 destination 的窗口 credit，也不做全节点窗口 barrier。

容量推导（M=128，K=top-k，E_local 为本 rank expert 数）：

```text
R_window_total <= G * B * K
mblocks_on_one_rank = sum_local_expert ceil(R[w,e] / M)
                    <= ceil(G * B * K / M) + E_local
D_window = ceil(G * B * K / M) + E_local
```

每个 X/S/C ring 至少能容纳一个最坏窗口；同时完整准入两个最坏窗口的保守容量
为各 2×D_window。实际准入使用真实 mblock 数。不同 plane 可独立退休，但不能
因 X 提前空出就越过 S/C 的容量限制。source contribution slot 必须在发出该窗口
Dispatch 前预留，否则 Return 可能反向等待仍未分配的目的地。

固定 owner 的 alloc/reclaim 序号只负责容量账本，不是动态 task queue。全局 head/tail
不是 readiness 依据；若账本用连续退休前缀，只能越过所有相关 slot 均已退休的区间，
不能把某个乱序完成 slot 当成之前所有 slot 都已完成。

同一窗口内可保留各 cluster 先 FC1、再 FC2 的静态枚举，但必须先保证整窗口
S 容量；不能在过大的 wave 里让 FC1 等 S 空位、FC2 又等该 wave 的 FC1 枚举结束。
通信/计算只按已准入窗口的有序任务推进，后续窗口不能占住前一窗口释放所需的资源。
具体角色配额、分级准入和等待图在实现 milestone 中验证，不能仅凭 ring
容量公式宣称已证明整条网络流水无死锁。

## 3. Buffer 布局与地址

### 3.1 数据区

以下是目标布局，不是当前 NetworkLayout 的现状；M=128，BF16 元素为 2 bytes。

| 存储 | 逻辑布局 | 分配与释放 |
|---|---|---|
| 输入 X/routes/weights | 调用方实际输入及必要的 peer 可达 staging | 本次调用输入生命周期，不复制成每远端节点一份完整 X |
| X ring | `[D_x, M, H]`，另有 gate/本代元数据 | 本 GPU，所有 FC1 读者退休后复用 |
| S ring | `[D_s, M, I]` | 本 GPU，所有 FC2 读者退休后复用 |
| C ring | `[D_c, M, H]` | 所有 Return 读者搬走完整行后复用 |
| Dispatch TX / RX | 分离的 `[remote_node, lane, D_net, slot_bytes]` | 本 rail 的注册区，逐 slot credit |
| Return TX / RX | 同样分离的 peer/channel ring | 不与 Dispatch 或对向 TX 别名 |
| contribution | source 上 `[D_contrib, B, K, H]` 及 route-ready | 同一 token 的 Reduce 完成后释放相应窗口 |
| immutable route plan | 实际路由/窗口几何及返回映射 | invocation；训练延长到 Backward |
| training tape | 实际大小的 X/G/U 等 checkpoint | 不放在会覆盖的 transient ring 里 |

每种 plane 有独立地址空间。逻辑 block b 映射到 `slot=b%D_plane`；读取/写入前
同时核对本代描述和 full/empty 条件。单凭取模地址不授予访问权。

通信 chunk 起始候选 Q=32/64 个 token 或完整 contribution 行，D_net=4，
每 peer 2/4 lanes；不在实现前把这些值定成最优。Dispatch slot 最坏容纳
Q 份 X 和 Q×K 个 fanout record，不随全调用 token 数扩大。

### 3.2 Packet 与返回地址

Dispatch packet 包含本代 header、去重 token records、fanout records 和 X payload：

```text
header: sequence, window, token_count, fanout_count, retire_target
token:  source token / X offset / fanout range
fanout: destination rank / logical Mblock / row / topk slot / router weight
```

source rank 可由 channel 得到，不必在每条 record 重复。fanout 与 X 属于同一次
Put 的有序 payload；ready signal 是独立的 GIN 控制字段。control 与大 payload
分离并按实际原子/NIC/TMA 要求对齐，padding 不承载额外协议。

路径和实际地址是：

```text
source X
  -> source rail Dispatch TX[peer,lane,slot]
  -> target rail Dispatch RX[source_node,lane,slot]
  -> target expert X[x_slot,row,:]
  -> S[s_slot,row,:] -> C[c_slot,row,:]
  -> compute node rail Return TX[source_node,lane,slot]
  -> source rail Return RX[compute_node,lane,slot]
  -> source contribution[window_slot,token_in_window,topk_slot,:]
  -> source output[token,:]
```

same-node Dispatch 直接从源 peer 可达 X pull 到 expert X ring；same-node Return
直接写 source contribution。返回标识必须能恢复 source window/token/top-k slot。

不能把 Return 唯一的反向映射放在已释放的 X slot：优先从 immutable plan 查询；
若把必要字段复制到 S/C/packet，它们必须在前一存储释放前完成复制。

### 3.3 内存界限

transient 数据容量由 D_x/D_s/D_c、G/B、channel 数、Q 和 ring depth 决定，
不随总 token 数 T 增长；通信资源仍随实际 peer 数增长，不声称与节点数无关。
plan、输入、输出以及训练 checkpoint 的总内存随实际 T/route 数增长。

热点 expert 超出一窗口时分多个窗口完成，而不是要求整次调用的热点 route
全驻留，也不以 overflow 后丢 token 换取固定容量。资源不足只能准入前报错/OOM
或等待已有窗口退出，不在主流水里依赖 host 临时扩容。

## 4. 逐 slot 协议与计数

### 4.1 通用规则

每个 slot 的 full/readiness 与 empty/consumed 是不同方向的状态，不能共用一个
含糊的 done。ready 表示本代数据可读；empty 表示本代所有读者已退休。
每个逻辑任务固定属于一个 worker，generation 标识代际，完成计数只负责同步，
不负责分配任务。

变长消费采用以下概念字段（不是已冻结的 C++ ABI）：

```text
slot persistent state:
    done_total                    // uint64，消费者累计完成数
    ready / empty generation      // 按 transport 选择发布机制

generation descriptor:
    expected_count                // 本代只读工作量，不做运行时原子增长
    retire_target                 // 该物理 slot 上一代 target + expected_count
```

唯一发布 owner 维护上一代 target，新 target 与本代描述在 ready 之前发布。
消费者先 acquire 本代 ready，完成全部异步读取，再 release-add 完成的单位数。
唯一回收 owner acquire 看到目标达到才发布该代 empty/credit。不同 generation
不能共享尚未结束的读取；即使 expected=0，也不能跳过本代 ready 判断。

同一个 slot 连续两代 expected 为 7、3 时，done_total 从 0 到 7 再到 10，
targets 为 7、10，不在换代时清零。跨调用可保持累积；重置只能发生在所有使用者
及通信请求 drain 后的初始化/重建边界，不能在 graph replay 中与旧操作竞态。
固定消费数 F 的情形可简化为第 g 代 target=(g+1)×F；不同 profile/F 混用时用累积
target，不能继续盲用 g×F。

一个 worker 可完成同一 (slot,generation) 的 n 项后只做一次 done+=n；
同一批涉及多个 slot 时分别记账。只有一个消费 owner 的 slot 可直接发布 empty，
不为了统一接口强行增加共享计数。跨 GPU 使用合适的 system-scope，同 GPU 用
device-scope；TMA/网络操作先真正完成，再走对应的 release/acquire 可见性链。

### 4.2 expected_count 的生成步骤

不新增“fragment 引用图”，也不让接收方猜 expected。来源如下：

| 被保护的数据 | expected 的单位与来源 | 谁生成 |
|---|---|---|
| Dispatch RX chunk | 本 chunk 所有 token 在目标节点上的有效 fanout 总数 | source 打包 owner |
| C Mblock | 当前窗口内本 block 的有效 route 行数 | destination geometry owner |
| Return RX chunk | packet 内完整 contribution 行数 | Return 打包 owner |

Dispatch 在生成 fanout record 的同一遍循环中计数：

```cpp
expected = 0;
for (token : packet_tokens) {
    for (k = 0; k < top_k; ++k) {
        if (valid(token, k) && node(expert(token, k)) == destination_node) {
            emit_fanout(token, k);
            ++expected;
        }
    }
}
```

并行 packing 时各 warp 求局部数量，再由块内归约交给唯一 owner 写包头。
已有 offsets 时，expected=Σ(offset[p+1]-offset[p])，p 仅取本 chunk 的 token
records。三个 token 分别有 2/1/3 个 fanout 时，expected=6，不是 token_count=3。

C block 的 expected 为 `min(M, R[w,e] - block_index*M)`，R 从第 2.2 节实际窗口
route 列表长度或 prefix 差得到，包括本地和远端 Return 的行。Return packet
每放入一条完整 contribution 就增加一，最终直接使用包头 rows。

Return fragment 只是一次搬运的一组完整 H 维行，不是计数单位。128 行分成
40/56/32 三次搬运，就上报 40/56/32；改变分包不改变 expected=128。
首版不把同一行拆给多个独立列片段任务；若以后这么做，必须重新定义计数单位。

packet fanout 与 fragment 个数在 route/分包规则固定后也能预计算，但 buffer 释放
不依赖 fragment 个数。channel terminal packet 数仍必须精确计算；它与这里按行的
retirement 计数是两个概念。所有批量记账都要求静态分区无重叠且完整覆盖。

### 4.3 独立释放点

- Dispatch RX：全部 fanout 的数据和后续所需 metadata 搬走后释放，不等 FC1。
- X：所有 FC1 N tiles 及训练需要的 X checkpoint 读取结束后释放。
- S：所有 FC2 N tiles 读取结束后释放。
- C：所有有效行连同返回标识搬到独立 TX/源 contribution 后释放，不等网络发送。
- Return RX：全部行 scatter 到 source contribution 并发布 readiness 后释放，不等 Reduce。
- contribution：对应窗口所有 token 的固定顺序 Reduce 完成后释放。
- TX：仍按第 6.4 节双重检查，不能把 C 的退休等同于 TX 可复用。

任何阶段的完成都必须包括其异步 TMA/peer 读者；“提交读取”“领取任务”均不是消费结束。

## 5. 单机 ring：pull 与 FC1/FC2 如何接力

参考实现位于本地 DeepGEMM-bk 的
`deep_gemm/include/deep_gemm/impls/sm90_bf16_mega_moe.cuh`：
pull 等 l1_empty，TMA store 完成后发布 l1_full，FC1 epilogue 发布 l2_full 并归还
l1_empty。这里借鉴的是逐物理 slot 的代际协议，不把当前 WeftMoE full arena 误称为 ring。

令一个 X 物理 slot 当前代为 g，完整 Mblock 为 M 行，本 profile 每个 Mblock
有 F1 个 FC1 逻辑 N tiles、F2 个 FC2 N tiles：

1. 窗口准入已预留 X/S/C。pull 先 acquire X slot 上一代 empty，才写本代行与 gate。
2. 每个 pull worker 等待 TMA store 完成后 release-add 已落地行数到 X full。
   尾块有效行 r<M，由唯一指定的尾行 publisher 多加 M-r，使本代仍贡献 M 个 credit。
   padding credit 不代表执行了虚构 route；计算/epilogue 使用 valid_m。
3. FC1 TMA producer acquire 等 X full 达 (g+1)×M，才发起该 block 的 GEMM load。
4. 一个 FC1 N tile 的所有 math warpgroups 完成 S 写回后，发布 S 的该 N tile ready，
   并向 X empty 累加一个完成单位。一个 task 覆盖多个有效 N tiles 时加实际数量。
   两个 math warpgroups 合作处理一个 M128 tile，不是各自计一个完整 tile。
5. X 的全部 F1 项完成后才能覆盖 X。S full 达到本代 F1 项后，FC2 可以读取 S。
6. FC2 每个有效 N tile 完成后发布 C full、累计 S empty。F2 项完成后允许 Return
   读取完整 H 维 C 行，同时允许复用 S。
7. Return 按第 4 节有效行数退休 C；源端按窗口完成 Reduce 并归还 contribution 容量。

这里的 full/empty 是 GM ring 状态，不能与 CTA 内 SMEM TMA stage 的 mbarrier
混为一谈。cluster multicast 的 shared stage 仍要等两个 CTA 的相应读者退休。

training 模式优先让 pull 在发布 X full 前同时完成 saved-X checkpoint；
FC1 在相应 S slice ready 前完成 saved-G/U 写回。若改成独立 checkpoint reader，
必须把该读者纳入释放条件，不能仍仅凭 F1/F2 计数覆盖数据。

各 cluster 在准入窗口内静态枚举自己的 FC1/FC2 residue，不引入全局 FC1→FC2
barrier；某 block 的全部 S 就绪即可开始 FC2。窗口容量/进度约束见第 2.3 节。
Return 和 Reduce 由固定角色推进，不等整个 invocation 的全部 GEMM。

## 6. 多机流程

### 6.0 复审：进度、容量与首版实现边界

历史第一版因 ring 进度证明不足而采用完整 invocation arena，属于第 0 节的基线，
不是这次后续实现的目标。必须修正的三个问题仍是：

- TX 等 credit 时不能阻塞负责 RX 消费/返回 credit 的角色。
- 全局 expert 总 count 不能推出每个窗口、channel、packet 的具体几何。
- NIC 本地完成、远端 payload ready、所有 RX 读者完成要明确区分发布机制与生命周期。

新方案用有界窗口和逐 slot full/empty 协议替代阶段级通信 barrier。以下各节是
待实现方案；旧的完整 X 广播、BeforeCombine 网络阶段不自动继承到新路径。

### 6.1 Route-aware Dispatch：每命中节点一份 X

source 从 top-k routes 得到每个 token 的实际目标节点集合；同一目标节点只放一份 X，
保留全部 fanout records。未命中的节点既不发 X，也不发无意义的空 data packet。
节点集合可用不超过 K 项的去重列表，不用写死两节点或 64 节点的 mask。

packet 按 (window, destination node, lane) 的固定 token 顺序切分，每包最多 Q 个
去重 token。一个 packet 可以 fanout 到同节点多个 GPU、expert 和 Mblock；
不再强制一个 packet 只能覆盖一个 destination Mblock，否则破坏节点级 X 去重。
这些目标必须属于同一个窗口；RX 可提前暂存，写 X 则必须已准入。

每个 rail/lane 的固定发送 owner：

1. 等 source contribution reservation；目的整窗口准入由 RX→X 的 pull 检查。
2. 等物理 TX slot 上一代的远端消费 credit 与本地 Put completion。
3. 按 immutable packet 目录打包 X/header，使用计划中的 fanout_count 累加该 RX
   slot 的 retire_target。M3b 已交换 route/token/consumer 目录，wire 不再重复携带
   TokenRecord/FanoutRecord；数据布局为 128-byte 对齐 header 加实际完整 X 行。
4. 等所有 packer 实际写入完成并导入 system visibility，再发一次绑定 ready signal 的 Put。
5. 按静态 sequence 前进；不使用 channel/task claim CAS。

首版一个 packet 对应一次 Put；若拆成多个 Put，必须先定义覆盖全部 pieces 的
ready 聚合，不能用最后一个 weak signal 推断其他操作完成。不同 GIN context
之间没有隐含的顺序。slot ready 使用绑定该 payload 的语义；不靠全网 flush
建立每包可见性。

跨机走 source rank 对应的同 rail GPU，进入目标节点本地注册 RX。
节点数循环、offset 和 terminal accounting 参数化；当前 expert 均匀连续分片、
每节点 rank 数等拓扑约束由公共接口明确校验，不静默假设 world=16。

### 6.2 目标节点 destination-pull

目标 rail RX owner acquire GIN ready 后，向机内读者发布本代 packet 可见性。
expert GPU 的固定 pull workers 按静态 fanout 分区处理自己拥有的 route：

```text
目标 rail RX 的 X/token metadata
  -> expert GPU X ring 的指定 logical Mblock/row
  -> X full 行计数
```

同一 RX slot 可以由多个 GPU 消费，done_total 放在 rail 控制区，使用满足 peer
访问语义的同步。每个 worker 在实际复制及需要的 metadata 读取结束后按完成
fanout 数批量累计；ACK owner 等第 4.2 节的 retire_target 后返还该代 credit。

不以“最后一个发起 pull 的 warp”代替全部读者完成；ready 的 NIC→rail→peer 可见性
链必须在实现时明确导入，不能在其他 GPU 上随意读一个普通标志替代 GIN acquire。

某个 Mblock 的全部有效行完成即可启动 FC1，不等当前 packet 的其他 fanout、
本窗口其他 Mblocks，更不等所有远端 Dispatch。后续包可同时被 NIC 接收。
RX packet 的尾部慢 reader 只背压对应物理 slot，不形成全局已完成前缀。

多机 pull 的任务遍历以 channel 的 packet sequence 为外层，再静态划分该包的
fanout；不能机械沿用 expert-major canonical row 的阻塞遍历。否则当前 RX slots
可能全装着后面 expert 的数据，而所有 pull warps 都在等尚未发进来的前面 expert，
即使 X/S/C 容量足够也会死锁。共享多个 channel 的 worker 必须继续服务能够释放
前序 packet 的任务；详见第 6.5 节的独立进度约束。

### 6.3 Return 与 FC2 流水

Return 不恢复 V3 的运行时 append/claim queue。固定 owner 从 immutable plan 的
route/mblock 列表枚举，按 (window, source rank, lane) 分包；一个 packet 可跨多个
已准入的 C Mblocks，但不跨窗口。它等待自己实际覆盖的完整 C 行就绪，不等节点
内所有 FC2。首版每行等待其 Mblock 的全部 FC2 N tiles 完成，不做列片段 Return。

- same-node：静态 worker 将完整 C 行写入 source contribution 并发布 route-ready。
- remote：M3b 的 rail TX owner 先非阻塞检查 packet 的全部 C 行，再从本节点各
  expert GPU pull 到独立 Return TX；不引入额外 packer→TX ready 计数。全部复制
  完成后按完整行退休 C，再 GIN Put；math 继续处理其他 Mblocks。返回标识来自
  immutable Return 目录，wire 同样只有 header 和 payload。
- source RX owner 收到 ready，按窗口/token/top-k slot scatter 到 contribution，
  完成复制和 readiness publication 后返还 RX credit。
- source Reduce 等该 token 的所有有效 contribution，按现有固定顺序求和。
  contribution 窗口在全部 token Reduce 完成后才能复用。

Return RX 不需要等 Reduce，Return TX 不需要持有 C 原址到网络完成。
不能为了沿用“全 Return 到齐后 Reduce”而持有一个无限长调用的全部 contribution；
在有界方案里至多按已预留的有限窗口批量 Reduce。Return/Reduce 的更细重叠可独立
消融，但不能破坏窗口回收进度。

### 6.4 逐 slot 复用决策（2026-09-23，待实现）

Dispatch/Return TX 覆盖前均保留：

```text
上一代远端消费 credit 已返回
AND
本 slot 上一代 Put 的本地 completion 已达到
```

“本地 credit”指 GIN 本地 completion counter，不额外增加一轮跨机 ACK。
completion 目标绑定本 slot 的实际 Put，不拿另一 slot 或另一个 context 的完成替代。

对于一份 TX、一次 Put、一份 RX，消费后返回 credit 的因果链已经包含该 payload
的传输；但不代表本地 completion counter 此刻必然更新。先采用保守双检查，
不把它描述为两个在数据依赖上必然独立的必要条件。

预计本地检查开销较低，尚未测量。后续核对具体 GIN 后端的源缓冲区生命周期契约，
再独立消融是否值得删除热路径检查；共享 TX 的多目标发送、提前 TX 回收和通信
资源 drain 需要另外处理。本轮不把这个优化作为 ring 实现的前置条件。

### 6.5 静态角色与 barrier 边界

Dispatch TX、Dispatch RX/pull、Return packing/TX、Return RX/credit、Reduce、
math/TMA 都有固定职责。需要哪些独立 warp/CTA 由 occupancy 和实测决定；
不能在一个阻塞的串行函数中先做完全部 TX 才服务 RX/Return。

各个有依赖的消费者必须能常驻并推进。同一个 owner 管多个 channel 时，只能使用
非阻塞 progress 或证明无环的固定处理顺序，不能等待一个后续 channel 而停止
服务仍被前序窗口依赖的 channel。角色间不插入会把消费者一起挂住的全 CTA barrier。

保留的同步边界：

| 边界 | 粒度与触发 |
|---|---|
| PlanReady | 路由/窗口目录完成后的控制面 exchange；main 前发布 |
| 窗口准入 | 相关 source/destination 的有限容量 credit，不是全节点 collective |
| data ready / consumed | 每物理 slot、每代；Mblock full 或 packet credit |
| SMEM stage | CTA/cluster 内 TMA/WGMMA 消费者的 barrier |
| invocation 结束 | 所有静态 task/route 完成，GIN 请求及 credit drain，必要的终态一致性 |

禁止沿用“所有 Dispatch Put 后一次全节点 barrier，再开始 FC1”和“全部 FC2 后
统一 Return”的主流程。终端 drain 不得提前放到每个 packet/window 的 hot path。

### 6.6 M3b 首版具体布局与进度约束（数值通过，吞吐仍待优化）

- 每 rail 使用 `(nodes-1)×L` 个 channel，Q=32 完整行，网络深度为 8；L 在
  `kNetworkPackets` 集中配置，已分别实测 L=2/8/16/32，收益与内存增量见 ring
  记录 U/X/Z，当前调优候选为 L32/B512；尚未改成默认发布配置。
  Dispatch
  TX/RX、Return TX/RX 四区不别名。每 channel 有只在终端更新的 `tx_base/rx_base`
  和各 owner 私有 cursor；每物理 slot 的信号、可见 generation、累计消费数、
  TX target/local completion、RX 已 ACK generation 均跨 invocation 保留。
- `W=min(channel_count,32)`。通信 warp 0 的前 W 个 CTA 做 Dispatch TX，其他做
  local/remote pull；warp 1 前 W 个做 Return TX、接下来 W 个做 Return RX scatter，
  其他做 local Return；warp 2 的 CTA 0 做准入、接下来 W 个做 NIC import/ACK，
  其他做 Reduce。math/producer 仍在全部 CTA 上静态分工。三类 GIN owner 分别
  使用独立的 W 个 context。TX 对 credit/C-ready 做非阻塞轮询，不阻塞其他 channel。
- receiver 的 immutable metadata replica 另有 `RouteLocation[route]`，由实际
  destination 写入 `{generation,logical_block,physical_row}`，经一次节点内
  PlanReady barrier 发布。Return rail 从该表找到 C；Dispatch pull 找到 X。
  此表属于 invocation metadata，不随 RX slot 复用；不跨 NIC 交换此本地地址表。
- 最老未完成窗口已有 source contribution 容量；destination 可完整预留该窗口，
  不等待 NIC credit。每 channel 的 TX 只按递增 window/packet 前进；pull 同样先
  window，再 source/lane/packet，静态 residue 不重叠。后窗 packet 即使提前占用
  部分 RX，也不能越过同 channel 的前窗 packet，且独立 RX/ACK 能继续回收前窗。
  Return TX 只复制全包已 ready 的 C 行，scatter 的目的 contribution 已预留，
  Reduce 释放 source 窗口，随后准入后窗。这个等待关系仍需热点/小深度实测验证，
  不能把推导替代多机验收。
- pull 的固定 worker 由窗口内 packet ordinal 与 source 交错映射，避免简单
  `packet+source` 导致小区间集中到少数 CTA。warp 协同读取远端目录，唯一
  TMA issuer 搬运；全局 copy 完成经 warp 同步传递后，各 lane 发布 X-ready，
  再归还该包的本 rank fanout credit。没有动态领用或运行时计算 expected count。
- source PlanReady 后，在已有窗口规划 pass 中预绑定普通本 GPU 的 `PullPacket`
  表：每窗 `G×ceil(B/Q)` 个 entry，固定索引延续上述 ownership。记录 RX
  payload/control 地址、generation、source 目录指针及已有本 rank fanout 数。
  空 entry 只设 rows=0；runtime 仅走自己的 residue，跳过空项不读 RX。此表
  不跨 NIC 交换，也不新增 barrier；保存的是当前调用的物理绑定，下一调用重建。
- NIC `readSignal(acquire)` → system-release `visible_generation` → peer acquire
  → 全部 TMA 完成 → system 累计 consumed → ACK，逐包建立可见性及复用链。
  多机 C.returned 同时接收本地 Return 与 peer rail 的更新，因此本地/远端
  RMW 及准入方的 acquire 全部使用 system scope；仅远端使用 system 不充分。
  同一 packet 内连续属于相同 C slot 的行可合并检查和退休，计数单位仍为行。
  所有 packet/reader/local completion drain 后才进行 invocation 结束 GIN barrier，
  防止快节点在慢节点仍读目录时开始覆盖下一次调用的 metadata。payload 中间无
  全节点 collective；跨调用仍保留这一边界。

## 7. Gate/up：独立 GEMM 消融，不预选 decode/prefill 布局

### 7.1 两种 GM 布局和计算路径

每个 expert 原始 W1 为 BF16 `[2I,H]`，H 连续；逻辑计算是 X 乘 W1 的转置：

```text
G_i = BF16(dot(X, W_gate[i,:]))
U_i = BF16(dot(X, W_up[i,:]))
S_i = BF16(SiLU(float(G_i)) * float(U_i) * router_weight)
```

V3-style separated：GM 行顺序为全部 gate I 行，然后全部 up I 行。
每个 FC1 任务覆盖 N_s 个 S 输出通道，共享 X，分别加载 gate/up B，
用两组 accumulator；现有实现 gate WGMMA 可先于 up TMA 全部完成开始。

V4-style interleaved：仍保持每个 H 维权重行连续，但行顺序为
`G0..7,U0..7,G8..15,U8..15,...`。p=0 表示 gate，p=1 表示 up：

```text
separated_row = p*I + i
packed_row    = (i/8)*16 + p*8 + i%8
address       = W1_base + (expert*2I + row)*H + h
```

上述映射描述当前按 8 通道分组的支持形状；扩展不整除尾部时必须显式处理 padding。
interleaved 一组 packed B/accumulator 计算两投影，物理 GEMM N 只产生 N/2 列 S。
它不减少 FLOPs。separated 的 N128 与 packed 的 N128 不是同样的有效输出宽度：
I512 时分别需要 4 与 8 个 FC1 N tiles；当前 packed N64 则需 16 个。

S 统一存成标准 `[M,I]`，FC2 不感知 W1 布局。ring 的 F1/ready 目标从所选
profile 的逻辑 N-tile 数推导，不写死。两种布局都可用于任一长度，不能先断言
“decode 用 packed、prefill 用 separated”。

### 7.2 必做的独立实验

把布局视为计算路径的独立变量，先隔离 GEMM，再验证端到端；不和 ring 改造捆绑
归因。独立 benchmark 使用已在本 GPU 的 canonical X、同一组权重、router weight，
相同 Mblock/valid_m 分布，无 Dispatch/GIN/Return：

1. 分别重放 1/64 和 8192/20480 tokens/rank 所产生的 expert 行数/尾块分布，
   包含 balanced 与热点样本；不能只跑大矩阵后推断 decode。
2. 比较 separated 双投影和 interleaved 单 packed 投影，至少报告
   FC1+SwiGLU，以及固定 FC2 下的 FC1+SwiGLU+FC2。
3. 先做受控对照：相同 M/K、精度、K 累加次序、有效 S 输出宽度、cluster 策略，
   尽可能匹配 stage/epilogue；记录布局必需的数据流、资源或指令形状差异。
   不能仅把两个实现的 N 宏设成一样就称为等粒度比较。
4. 再分别调 N/K/stages/有效 task 粒度，报告两种布局各自的最佳可行配置；
   “布局本身的差异”和“布局加各自调优后的差异”分开记录。
5. 核心固定 1620 MHz，检查实际时钟；记录延迟、register/spill、SMEM/occupancy、
   TMA/WGMMA stall 和有效吞吐，不用操作数大小推测胜负。
6. 权重 packing/descriptor 建立在计时前完成；另外报告转换耗时、额外权重副本内存、
   生命周期和更新/失效成本，不把一次性 packing 混进 steady-state GEMM。
7. 根据结果分别选择 decode/prefill profile。若选不同布局，明确同时持有两份
   权重的内存代价，训练权重更新后不能使用旧 packed 副本。
8. 在相同通信协议下回测 EP8/EP16 全矩阵，验证 profile 选择及阈值；真实 workload
   边界样本用于校准，不盲用当前 8192 阈值。无新实测前不填优化收益。

输入 token 数只是选 profile 的代理，expert 实际行数与 skew 也影响 GEMM。
若使用设备 plan 的计数选 entry，选择必须在静态任务/计数目标发布前固定，
不能让同一 block 的消费者对 F1/布局有不同理解，也不能增加 host 读回同步。

### 7.3 数值和 TMA 约束

- FP32 accumulator；G/U 各自先 BF16，再以 FP32 做 SiLU、乘 up 和 router weight，
  然后 S round BF16；FC2 输出 BF16，source 固定顺序 FP32 combine 后 BF16。
- router weight 不是 W_gate；不能把它移到 FC2 之后。
- tile、stage、布局是待调参数，不再沿用“TileN/TileK 必须永远不变”的旧约束。
  保持相同运算次序的受控比较检查 bitwise；若候选改变归约次序，明确标注并走
  独立 PyTorch 误差门禁，不能把数值差异藏在性能结果里。
- 当前 cluster multicast 分享 A，保持两 CTA 的 stage 读者生命周期。将来若消融
  share-B pairing，单独证明 GM/SMEM ring 与 epilogue staging 的安全性。
- 已迁移的 FC1/FC2 合并写回、CTA 私有 B staging 要在两种布局下复用/验证；
  不恢复不安全的 multicast A arena epilogue 复用。

## 8. Prefill/training 的保存与 no-drop 合同

prefill inference 与 training Forward 共用有界流水，但训练保存实际规模的 tape，
不是在 ring 中留住整次调用的激活。当前 training 仍走 V3；迁移计划如下：

```text
route_plan_kernel
  -> stable route ordinal / 窗口计数 / count exchange / 精确 R_actual、P_actual
  -> 冻结 main 与 Backward 共用的 plan；此时不发 activation payload

host allocation boundary
  -> 取得本 rank 实际 tape 大小及 collective status
  -> 分配 saved tensors / plan lease；失败则在 main 前一致报错

main_from_plan
  -> 复用原 plan，不重复 route counting/exchange
  -> bounded Dispatch / X / S / C / Return / contribution 流水
  -> 保存独立的 X/G/U、gate 与返回映射

Backward
  -> 消费该 Forward 的精确 plan/tape
  -> 固定逻辑行顺序和梯度归约合同
```

saved tensors 根据实际 R_actual 或所需的 padded P_actual 分配；transient X/S/C
仍是固定深度 ring，不重新申请完整 training canonical arena。稀疏/极端路由
不需要每个 rank 都预留 world×max_tokens×K 行激活。

tape 的永久逻辑行号与物理 ring slot 分开。若 tape 沿用全局 expert/source 顺序，
而执行按 window 顺序，必须保存明确映射；不能把 window 内 padded row 当成现有
Backward 的 canonical row。G/U 可保存标准逻辑布局，使 Backward 不依赖 FC1
选择 separated 还是 packed。dW 跨 M 归约的确定性需要单独验证，Forward 数值
一致并不自动保证梯度归约一致。

每个 live tape 持有独立 allocation/plan lease，直到 Backward 完成。资源不足可
等待或显式失败，不能覆盖尚未消费的 tape。输入/输出/plan/tape 真正 OOM 时允许
整个调用失败；“不丢 token”不是承诺任意总规模都可在有限显存内训练。

需要 host 分配的 plan/allocate/main 不是单 kernel，也不承诺整体 CUDA Graph。
PDL 不能替代动态地址的分配依赖，本轮不加入 pointer-mailbox 或运行中 host 扩容。

## 9. 模块划分与复用

基于现有 `src/cuda/static_v4/` 演进，不新建一份 V3 GEMM。拟划分的职责如下，
具体文件名以实施时最小改动为准；此表不是声称这些模块均已存在：

| 职责 | 复用/修改方向 |
|---|---|
| route/窗口几何 | 扩展 route_plan；共用 source identity、logical row、combine order |
| slot 协议 | 独立的小型 generation/full/empty helper，与 GIN 传输 API 分离 |
| 节点内搬运 | destination_pull 与返回完整行复制共用 TMA/peer copy primitive |
| 网络布局/访问 | network_layout/access：TX/RX 四区、控制字段、peer/lane offset |
| 网络进度 | 独立的 Dispatch TX/RX、Return TX/RX 固定 owner，不夹带 math task claim |
| 数值计算 | 现有 WGMMA/TMA/epilogue 参数化布局/profile，共享 BF16 合同 |
| 生命周期 | initialize/finalize 管理 invocation 与 drain，不在逐行路径加入全局状态机 |
| training | StableTape、实际分配及 Backward adapter，与 inference 快路径分开 |

单机 specialization 不编入无用 GIN 路径；decode/prefill 分离配置和必要的编译期
entry，但共享路由、ring 和 Return 协议。不能让 layout/profile 分支复制通信栈。

静态 worker 编号由 CTA/cluster/warp 及 round 推导；旋转 residue、owner 私有
小窗口 lookahead 可后续独立消融，不恢复跨 worker steal。GIN context 有唯一
提交 owner；按实际角色需要分配 context，不把整个 CTA 数无条件乘到所有资源上。

## 10. Invocation、Graph 与复用边界

“调用”指一次 MoE Forward invocation，例如模型中的一层或一次 graph replay 中
的一次该层执行；不是一个 packet、window、CUDA CTA 或服务端 request。
一次调用内部含许多 ring generations。

- 同一 EP 组按相同次序调用，runtime workspace 依靠 stream/graph dependency 串行复用。
  多 workspace 不自动意味着多 stream 的 GIN 并发安全；本轮不扩展该语义。
- ring 的 g 是该物理 slot 的累积代号；逻辑分配序号跨调用保留或加正确的 base。
  不能在每次调用把 block 序号重置为零，同时继续使用未归零的 full/empty targets。
- invocation 完成前确认输入读者、所有 Return、Reduce 和 GIN 请求/credit 已 drain。
  下一次调用不能覆盖仍被前次异步操作使用的 payload、metadata 或 control。
- Graph 使用稳定地址，routes/gates 可在输入和 plan workspace 合同内改变；
  transient ring 容量不要求容纳 graph 的全部 route。plan/control 本身仍需要
  与 graph 输入规模匹配的容量，不能以 graph capture 为理由忽略元数据容量。
- Eager prepare 或训练 plan 后可以分配实际输入/plan/tape；steady-state inference
  main 内不 cudaMalloc，不 D2H 读 count，不让 host 参与逐窗口推进。
- 只在完全 quiescent 的初始化/重建中 reset 累计状态；Graph、不同模型层共用
  workspace、profile 切换必须沿用同一生命周期合同。

## 11. 实现待办与 commit 里程碑

每个 milestone 完成并验证后提交一次 commit，同时更新对应优化记录。只记录
实际完成的代码和测量，未实现项保持未勾选。以下顺序不是要求 GEMM 消融等待网络：
M5 与协议实现正交，可在冻结基线后提前做，但不把两者收益混为一项。

### M0：设计与基线整理（本次文档工作）

- [x] 区分已实现 full-arena 基线与待实现 ring，统一节点内 pull、节点去重及流水目标。
- [x] 明确窗口准入、各 buffer 的独立生命周期、expected_count 来源及双 credit 决策。
- [x] 列出 gate/up 独立消融、实现 milestones、1620 MHz EP8/EP16 测量要求。
- [x] 保留历史实验记录，不把文档修改记为性能收益。

### M1：冻结可执行的 plan、布局与进度合同

M1a 已实现 host/device 共用窗口/packet/slot 合同与 GPU source-window planner；
M1b 增加独立于 RX 生命周期的 packet→consumer 目录及 destination-window 几何。
M3a 补齐 Dispatch route→packet/token 反向索引和 Return 精确逐 channel 目录，
并将 source epoch/gate/route/packet 元数据集中为不含 X 的独立区域。
详见 ring 实施记录。规划器已接入单机 Forward，包括窗口几何、source contribution
映射和整窗口准入；下面涉及网络目录交换、跨机准入的完整事项仍保持未完成。

- [x] 窗口编号、完整 token 分包、任意节点数的 peer/lane 地址和逐 slot 累积退休合同。
- [x] GPU source-window expert/node counts、去重 token 列表、source ordinal 与 channel
      packet prefix；显式区分 masked route 和非法 expert id。
- [x] 紧凑 channel packet 目录、各目标 GPU 的完整行消费数；destination-window
      counts→Mblocks/C expected 行数及 source round-robin ordinal 的逆映射。
- [ ] 定义窗口编号、实际窗口/expert counts、逻辑 Mblock、source contribution 映射，
      以及 packet-sequence 优先的接收侧静态任务目录。
- [ ] 冻结 packet/token/fanout 及 Return 行枚举，生成精确 channel packet 数和
      expected/retire targets；明确哪些目录存储、哪些闭式计算。
- [ ] 定义各 plane 的 GM 地址、slot 控制字段、GIN VA signal/local counter 分工和初始化。
- [ ] 实现准入/退休账本；画出最小等待依赖图，确认最老窗口的所有消费者可驻留推进。
- [ ] 用少量 host/device 共用枚举检查证明覆盖、无重复、no-drop 与 tail 计数，
      不为每个字段建立大量无意义单测。
- [ ] 冻结 V3、当前 V4 构建/config/hash 和重测脚本，作为后续逐步对照。

### M2：单机 X/S/C 与 contribution ring

- [x] 接入 destination-pull → X full、FC1 → S full/X empty、FC2 → C full/S empty。
- [x] 接入按完整行的 Return/C retirement 和窗口内 Reduce/contribution 释放。
- [x] 保持 reverse metadata 不依赖已释放 X；验证 multicast/epilogue stage 生命周期。
- [x] 用小 ring 深度、多代 wrap、热点 expert 和 tail 暴露错误的复用与容量依赖。
- [x] 单机 EP8 四规模 eager/graph 正确性与 1620 MHz 性能，记录 full arena→ring 的
      独立变化及实际峰值显存；不以更多常驻显存伪装有界流水。

M2 首版 X/S/C 有独立 completion/target，但三个 plane 使用相同物理 slot，
保守地等待三者全退休才共同复用；尚未利用 X/S 早退休的容量。source contribution
是独立的 ring，默认两窗口，可通过 pool 的 `static_forward_contribution_windows`
配置并独立消融。首版性能落后，不默认启用；已测固定 pool 的分配峰值，
尚未测应用整体显存峰值，性能与冗余 metadata 优化继续跟进。

### M3：多机 route-aware Dispatch 与节点内 pull/FC 重叠

- [x] 实现按目标节点去重的 packet，未命中节点不发送 X，支持参数化 peer 数。
- [x] 接入分离 TX/RX ring、固定 GIN owner、ready 可见性链、fanout 完成及双重 credit。
- [x] 实现分级窗口准入（source reservation、RX 暂存、destination 整窗 pull）；
      RX/pull/credit 的进度不受阻塞 TX owner 影响。
- [x] 移除 data-plane Dispatch 全节点 barrier，验证 NIC 接收、节点内 pull、FC 时间线重叠。
- [x] EP16 四规模和边界路由通过；若过渡阶段仍保留完整 Return arena，明确标为
      临时路径，不能在此宣称全部多机有界流水完成。

### M4：多机 Return/C 回收与端到端流水

- [x] 从静态 C/route 目录打包，按完整行退休 C；不引入 fragment 引用图或 append queue。
- [x] Return TX/RX 与 Dispatch 独立；source RX scatter 后归还 credit，Reduce 后释放窗口。
- [x] 移除节点级 FC2→Return barrier，确认 Return 与其他 Mblocks 的 FC1/FC2 重叠。
- [x] 完成所有 channel 的 terminal/drain、跨调用/Graph 状态复用。
- [ ] EP8/EP16 全矩阵、热点与小深度 wrap 通过；记录 Dispatch 和 Return 各步收益及
      资源变化。协议支持任意有效节点拓扑，实机仍只声明两机验证。

### M5：长短序列 gate/up 与 GEMM 独立消融

- [ ] 建立第 7.2 节纯 GEMM 基准，固定输入/权重/route 分布、时钟和数值参考。
- [ ] 做等有效 S 宽度的 separated/interleaved 受控对照，核对逻辑 N tiles 及计数目标。
- [ ] 分别调优两种布局的 N/K/stages/epilogue，保留失败候选和资源/误差原因。
- [ ] 分别选择 decode/prefill layout/profile，明确切换阈值及权重副本更新合同。
- [ ] 在同一通信版本回测 EP8/EP16 四规模；把纯 GEMM 收益和端到端收益分别落表。

### M6：训练 Forward tape 与 Backward 对接

- [ ] StableTape plan、按实际规模分配、main_from_plan 复用同一 plan。
- [ ] checkpoint 写出纳入 slot 生命周期，Backward 不依赖已覆盖的 transient 数据。
- [ ] 对齐 tape 行号、G/U 布局、固定 dW reduction 顺序；实现必要的 Backward adapter。
- [ ] Forward/Backward 参考校验、热点长序列、多个 live tapes 生命周期及 OOM 一致失败。
- [ ] 完成后再标注 training V4 支持；此前继续保留已验证的 V3 训练路径。

### 后续独立优化，不作为首版阻塞项

- [ ] 在后端契约确认后消融去掉 TX 热路径本地 completion 检查。
- [ ] 非一一绑定 TX/RX 的提前 TX 回收，仅在收益值得额外状态时引入。
- [ ] ring depth、Q、G/B、lane 数、comm warp 配额、owner 私有 lookahead。
- [ ] Return/Reduce 更细粒度 overlap；不改变已有数值顺序或打破有界窗口回收。

## 12. 验收与实验记录要求

### 12.1 正确性与进度

复用已有测试入口和 PyTorch 数值参考，补充能暴露本次协议风险的少量场景：

- 主矩阵：EP8=单机 8 GPU，EP16=双机各 8 GPU；T=1/64/8192/20480 per rank。
  2×4 EP8 只能作额外实验，不能替代单机 EP8。
- balanced、全部合法 route 指向同一 expert/rank、重复 expert、masked route、
  不等长/空 rank、tail Mblock、无远端命中、一个 token 在某节点命中多个 expert。
  非法 expert id 与合法 masked route 区分，不能把错误输入静默当成 drop。
- 小深度多次 ring wrap、变长 fanout、Return packet 跨 C slots、故意延迟某个
  RX reader/Return owner，验证 slot 退休不误用全局完成前缀。
- Graph routes/gates/inputs 变化、混合 profile、双层共享 workspace、连续 eager/graph
  调用；当前空输入 graph 的 pool 限制需明确处理或报告，不能冒充已通过。
- 每条有效 route 恰好执行/Return/combine 一次；独立于 packet 分片策略和 worker 数。
  同步正确性不只看最终 hash，还检查 no-drop、各代 target 和 drain。
- 任意节点数的枚举/offset 做小型模拟校验；三机以上性能和通信进度待实机验证。
- 不将外部 NIC 故障等价为普通 backpressure；错误终止复用 runtime 的协同 abort
  合同，不在每行 hot path 添加无依据的 system-scope 防御性轮询。

### 12.2 性能与归因

所有 GPU 核心固定 1620 MHz 并采样核实实际 SM/graphics 时钟，不使用无法维持的
1980 MHz。固定/记录其他环境、显存时钟与工作负载，正式计时不并发运行编译或其他
GPU 测试。沿用 graph CUDA-event rank-max p50，报告 warmup、sample 数和 replay batch。

每个性能 milestone 至少记录：

- commit、binary/config hash、拓扑、H/I/E/K、实际输入与路由分布、时钟；
- 修改前后交错 A/B（反转顺序），以及同条件 V3 对照；历史 V3 非本轮重测必须标明；
- kernel 端到端、可解释的 stage/timeline、峰值显存与协议计数开销；
- 已通过的数值门禁、失败候选、未测部分及是否为多项组合收益。

gate/up、GEMM tile/epilogue、网络节点去重、Dispatch overlap、Return overlap、
ring depth/窗口大小分别消融。多个改动一起启用只能报告组合收益，不凭直觉分摊。
重叠由 timeline 证明，不把并行 stage span 相加当成总延迟，也不拿旧的未完整赋值
device-clock 字段当通信计时。

实际实施、验证和测量以第 11 节及 ring 实施记录为准，未勾选项不能视为已实现。
