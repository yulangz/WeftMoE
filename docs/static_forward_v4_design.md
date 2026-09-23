# WeftMoE Static Forward V4：单机与多机重构设计

单机 inference-only 路径的逐项性能消融与失败实验见
[`static_forward_v4_optimize_record.md`](static_forward_v4_optimize_record.md)。

## 0. 当前实现状态（2026-09-22）

单机 inference-only V4 已经完成一次按 MegaMOE 数据面重写。本节是当前实现
的准绳；后文保留了最初的设计推演，其中 source-push 等已被实测否定的选择
不再代表当前单机实现。多机 inference-only 已实现第 6.0 节的完整 invocation
arena 协议；训练仍保持 V3。两机 EP8/EP16 的 1、64、8192、20480 token/rank
均已通过 eager/graph PyTorch 数值校验；空 rank、不等长输入、重复 expert、热点
路由、动态路由图回放及双层共享 workspace 也纳入同一实机门禁。

- route exchange、X/gate staging、destination pull、FC1/SwiGLU、FC2、Return/Combine 和
  终止清理已物理融合在一个 kernel 中。inference steady state 每层只有一个
  fused kernel，没有独立 notify kernel，也没有 host `handle.wait()`；
- kernel 使用 132 个 384-thread homogeneous CTA。所有 CTA 具有相同代码与资源
  合同：`0..255` 是两个 math warpgroup，`256..287` 是 TMA producer warp，
  `288..383` 是三个 pull/control warp。不存在专用 transport CTA；
- 每个 source 把 route id 发布到 destination 上按 `(expert, source, slot)` 索引的
  symmetric route table，并把 X/gates 一次 staging 到 source-local symmetric storage。各
  destination 的 control warps 用确定性 source-round-robin row 顺序 pull payload。它没有
  work queue、claim CAS 或特殊 CTA；
- route exchange 结束后，FC1 pair 和 FC2 unit 通过
  `cluster + round * cluster_count` 静态枚举。有限静态工作不再在每个 row/task/barrier
  中轮询 system-scope cancellation word；V3/训练路径仍保留可取消逻辑；
- 数值主体复用 BF16/WGMMA micro-kernel、FP32 accumulator、SwiGLU rounding 点和
  top-k FP32 固定 combine order。当前 autotune 结果为 M128、FC1 N64、FC2 N128、
  K128、FC1/FC2 各3 stages、cluster size 2。MegaMOE 形状的 N128/K64 候选在该
  workload 上反而约为 81 us，因此 tile 由 autotune 选择，不为形式对齐强制写死；
- CUDA Graph 计划直接绑定稳定的 PyTorch X/routes/gates 地址和固定 output，
  replay 没有 input `copy_` 或 output clone。外层 graph 中每层是一个 child graph node，
  child graph 内只有上述 fused kernel；
- pool 按公开的最大 route 容量预留 route table/canonical row 空间。host 在 launch
  前检查容量，合法 route 不使用 capacity factor，不截断、不取模覆盖、不丢 token。

严格 EP8 decode case（H=2048、I=512、64 experts、top-k=4、64 token/rank）的
裸 fused CUDA Graph rank-max p50 为 **72.50 us**，已包含 X staging 且直接读取
routes/gates。真实 tensor runtime 的双层 child graph 测得 **74.24 us/层**。
MegaMOE 同 shape 的 graph kernel 是 62.6 us，但图外三份 input copy 是 11.5 us；
公平口径为 74.1 us。因此裸 fused graph 快约 1.6 us（2.2%），而包含 child-graph
调度的应用口径与之相差 0.14 us（0.2%）。WeftMoE 内部 clock
与 MegaMOE CUDA-event kernel-only 不是同一计时口径，不将二者直接比较。

已通过平衡路由和 64 token/rank 极端 skew 路由的 8-rank tensor parity、双层
CUDA Graph replay、graph 销毁/offload 与 workspace 生命周期回归。平衡微基准在
多轮 graph replay 后的 8 个 rank output hash 也与固定期望值完全一致。

状态：单机与多机 inference-only V4 已实现；多机性能记录见
[`static_forward_v4_multinode_record.md`](static_forward_v4_multinode_record.md)。
2026-09-23 的两机 1620 MHz 锁频对照表明：1/64 tokens 相对 V3 加速
3.86–4.35×，但 8K/20K 延迟增加 15.1–34.2%；多机 prefill 性能目标尚未达成。
这组结果使用各自公开参数，不是 scheduler 单变量消融，不能把差距全部归因于同步。
training 重构尚未实现，不属于本次 inference-only 改造。

日期：2026-09-22

范围：Hopper BF16 Forward；覆盖 inference，以及需要保存 Forward 中间激活的
training Forward。Backward kernel 本身不在本轮重写范围内，但新 Forward tape
必须能被现有/后续 Backward 消费。

## 1. 结论先行

本次重构采用以下明确方案：

1. **单机和多机使用两个独立 kernel specialization**，共享路由布局、静态任务
   枚举器、GEMM micro-kernel 和同步协议；不把 GIN/NCCL device 代码编入单机
   kernel。拓扑差异值得专门化，inference/training 则只做模板级差异，不复制一套
   scheduler。
2. **Inference 的 Notify 与 main kernel 合并成一次 cooperative persistent
   launch**。路由计数、count exchange、几何生成、Dispatch、FC1、SwiGLU、FC2、
   Return 和 Reduce 均在一次 launch 内完成；hot path 不再生成一个需要 host
   `wait()` 后才能启动 main 的 `DeviceDispatchHandle`。
3. **取消运行时 work queue、全局 claim cursor 和 ready-task CAS**。路由完成后，
   task 集合被冻结；CTA/cluster/warp 通过 `worker + round * workers` 的闭式编号
   枚举自己的任务。原子计数仅用于 payload readiness、跨 GPU release/acquire、
   GIN ring credit 和错误发布，不再用于“谁来做这项工作”的分配。
4. **单机使用 destination-pull，多机保留 push/relay 方向**。单机 source 在
   fused kernel 内把 X/gates staging 到 source-local symmetric storage，并向 expert owner
   发布紧凑 route table；destination 的同构 control warp 静态 pull canonical row。
   这条实测路径比 source-push 更接近 MegaMOE，也避免了 push 方构造最终 row
   位置和频繁 remote readiness 原子操作。多机 remote route 仍必须使用 GIN push，
   再由目标 rail GPU 静态 relay，不使用跨机 pull。
5. **多机只使用跨机 push**。每个 GIN channel 有固定 owner，按静态 sequence
   顺序打包和发送；目标节点的 receive/relay channel 也有固定 owner，再通过
   NVLink/IPC relay 到 expert GPU。跨机 pull 不进入方案。
6. **不尝试让 route exchange 与 GEMM 重叠**。先完成 route counting、count
   exchange 和 task geometry 冻结，再开始 payload Dispatch 与计算流水。Payload
   Dispatch、FC1、FC2、Return、Reduce 仍然互相重叠。
7. **单机 inference canonical row 采用确定性 source-rank round-robin 布局**：

   ```text
   row(e, source, slot) = expert_base[e] + round_robin(source, slot)
   ```

   route table 仍按 `(expert, source, slot)` 存放，pull 时逐层轮转所有仍有行的
   source。这使相邻 warp 分散到不同 NVLink peer，同时保持路由给定时 row
   顺序唯一。training 未改用该 inference-only 布局；其 tape/dW 顺序仍由 V3
   合同保证。
8. **保留现有 GEMM 数值合同**：FC1/FC2 的 N tile、K tile、K 累加顺序、FP32
   accumulator、BF16 rounding 点、SwiGLU/gate 精度和最终 top-k reduction 顺序均
   不变。本轮只改任务如何枚举和数据何时 ready，不改一个输出元素内部的 reduction。
9. **需要保存中间激活时固定采用 `plan kernel -> host allocate -> main_from_plan`
   的 memory-tight 路径**。各 rank 的 token 数和 routing skew 不设静态最坏上界；
   host 根据 plan 给出的本 rank 精确 `R/P` 分配 canonical/saved tensors。planning
   结果由 main 直接复用，绝不重新 Notify。合法 route 不设 capacity factor、不
   截断、不覆盖取模。PDL 不能解决“main launch 时输出指针尚不存在”这一依赖，
   因此不用于这条动态分配路径。

## 2. 当前实现中需要被替换的部分

当前路径的数值 kernel 可以复用，但控制面过重：

- `notify/local_notify.cuh` 为稳定 route ordinal 做 chunk radix/count/scan；
  `notify/materialize.cuh` 随后物化 destination row、packet/fanout、wave 和多个排序
  permutation。
- `persistent_v3/forward/dispatch_pipeline.cuh` 由 publisher 等待 `x_arrived_rows`，
  再把 FC1 task 写入共享 ready queue。
- `persistent_v3/forward/compute_kernel.cuh` 用全局 cursor/CAS 领取 FC1 pair，FC1
  epilogue 再发布 FC2 ready queue，compute scheduler 在 FC1/FC2/Reduce 之间动态
  probe。
- 单机 Relay、GIN Dispatch/Return 和 Reduce 还各有自己的 cursor、claim 或 ready
  queue。即使某个 `StaticLocal` 分支跳过部分 queue read，也没有形成从 route 到
  compute/return 的完整静态协议。
- Dynamic training 在 Notify 后才知道实际 incoming rows，host 需要取得该值后再
  准备保存 X/A/B/header 的 tensor。这是 allocation dependency，不是 GPU kernel
  readiness dependency。

V4 不在这些队列上继续打补丁，而是让 route plan 直接生成所有阶段共享的不可变
几何，随后每个阶段从相同几何独立枚举任务。

## 3. 共同的数据模型

### 3.1 最小 route plan

每个 source rank 对本地扁平 route `r = token * K + topk_slot` 只保存：

```cpp
struct SourceRoute {
  uint32_t expert;       // invalid/masked route 使用单独 valid bit
  uint32_t source_slot;  // 本 source、该 expert 内的 ordinal
};
```

`token` 与 `topk_slot` 由 `r` 反解，无需重复保存。全 rank 交换的控制信息只包括：

```text
count[source_rank, global_expert]
config/error digest
```

每个 destination rank 从 count matrix 本地得到：

```text
expert_rows[local_expert]
expert_base[local_expert + 1]
source_prefix[local_expert, world + 1]
mblock_count[local_expert]
mblock_prefix[local_expert + 1]
```

因此 source 在 count exchange 之后可以直接计算任意 route 的 destination row；
destination 不再需要 Notify 预先生成 `route_dst_row`、`route_dst_rank`、
`packet_fanout`、`combine_order` 和多轮 wave sort。

Dispatch payload 在 canonical row 同时写入一个紧凑 header：

```cpp
struct CanonicalRouteHeader {
  uint32_t source_rank;
  uint32_t source_token;
  uint16_t topk_slot;
  uint16_t flags;
  float gate;
};
```

FC2/Return 用该 header 将结果写回 source 的固定 contribution slot
`[source_token, topk_slot]`。最终 Reduce 始终按 `topk_slot = 0..K-1` 累加，因此
不需要 Notify 构造额外 combine permutation。

### 3.2 no-drop 容量合同

“不丢 token”在实现中应精确定义为：每条未 masked 的合法 route 都必须执行 FC1、
FC2、Return 并参与 source token 的固定顺序 Reduce。masked route 是调用方显式语义，
不是容量丢弃。

对一次实际 invocation，本 rank 需要保存的 route 行数由 plan 精确给出：

```text
R_actual[rank]
  = sum_source sum_{expert owned by rank} count[source, expert]
```

若计算 arena 对每个 local expert 做 M-tile padding，plan 同时计算精确的：

```text
P_actual[rank]
  = sum_{expert owned by rank} round_up(expert_rows[expert], TileM)
```

training 不再要求调用方给出所有 rank 的 `T_cap`，也不按
`world*T_cap*K` 预分配。plan kernel 在 device 上完成 count exchange 和 prefix 后，
只把本 rank 的 `R_actual/P_actual` 及 collective error 状态带回 host；host 按这两个
精确值分配 main arena 和 tape，再启动 `main_from_plan`。极端 skew 只会让实际接收
较多的 rank 分配更多显存，不会迫使所有 rank 为同一最坏值常驻预留。

若 allocator 无法满足精确申请，整个 invocation 明确 OOM 并在任何 activation
payload push 前终止；不能只保留前 N 行继续执行。由于这条路径包含 host allocation，
training Forward 不承诺 CUDA Graph capture。Inference 不保存 tape，继续使用单次
融合 kernel 和它自己的 capacity/graph 合同。

### 3.3 source slot 的两种生成方式

两种模式产生相同的下游布局和 scheduler：

- `FastInference`：每条 route 对本地 expert counter 做 atomic reservation。
  `source_slot -> route` 不承诺跨运行稳定，但每个 Forward row 的 K reduction 和
  最终 top-k reduction 不受影响。
- `StableTape`：用于 training/需要可复现 route plan 的路径。每个固定 route chunk
  先写 `chunk_count[chunk, expert]`，按 chunk id 做 deterministic prefix；第二遍按
  route 原始顺序用 warp `match/ballot` 和固定 warp-prefix 生成 chunk 内 ordinal。
  它不需要对全部 route 做 radix sort，也不依赖 global atomic 的到达顺序。

`StableTape` 的 canonical M 顺序固定为：

```text
local expert -> source rank -> source 原始 (token, topk_slot) 顺序
```

Forward tape 保存 count/prefix/header，Backward 必须消费这一次 Forward 的精确 plan，
不能重新运行 atomic route reservation 来猜测行顺序。

### 3.4 不变的数值合同

重构不得改变以下内容：

- `TileM/TileN/TileK` 及 cluster pairing；尤其 FC1/FC2 的 N、K tile 不变；
- 每个输出元素沿 K 的 WGMMA stage 次序；
- FC1 FP32 accumulate 后 A/B 各自 round 到 BF16；
- SwiGLU 从这些 BF16 A/B 恢复到 FP32，乘 FP32 gate，再 round 一次到 BF16；
- FC2 FP32 accumulate 与 BF16 输出；
- source output 按固定 top-k slot/expert 顺序做 FP32 sum，最后 round BF16。

M tile 之间的执行先后可以变化，因为 Forward 不跨 M row reduction。training 的
dW 对 M 做 reduction，所以 `StableTape` 和 Backward 的固定 M traversal 仍然是
training 合同的一部分。

## 4. 静态 task 系统

这里的“静态”不是 task 数在编译期固定，而是：count exchange 完成后，task 数和
task 到数据的映射在该 invocation 内不可变，worker 不再竞争领取任务。

### 4.1 Dispatch task 生成

每个 source 根据最终 row/mblock，将本地 route 放入 bucket：

```text
bucket key = (destination rank, destination logical mblock, transport class)
transport class = local-NVLink | remote-GIN-channel
```

一次 count/prefix/scatter 得到 `dispatch_route_order[T*K]`。bucket 内顺序不影响
canonical row；training 稳定性来自 `source_slot`，所以这里可使用较便宜的 atomic
scatter。每个非空 bucket 按 `DispatchChunkRows` 切成不可变 descriptor：

```cpp
struct DispatchTask {
  uint32_t route_begin;
  uint16_t route_count;
  uint16_t channel;
  uint32_t destination_rank;
  uint32_t destination_mblock;
};
```

这不是 ready queue。descriptor 在 main phase 开始前一次性完成，之后永不追加，
也没有 head/tail/claim cursor。

### 4.2 各阶段的固定 ownership

| 阶段 | 静态 ownership |
|---|---|
| route chunks | `chunk = cta + round * num_ctas` |
| local destination-pull Dispatch | `row = global_warp + round * global_warps` |
| GIN send | `channel -> fixed GIN owner CTA`; owner 按 sequence 递增发送该 channel 的 task |
| GIN receive | `channel -> fixed receive owner`; owner按预期 sequence 消费，不做全局 claim |
| node-local relay | 一个 receive chunk 内 `row = relay_warp + round * relay_warps` |
| FC1 | cluster 拥有固定的 FC1 tile-pair residue class |
| FC2 | cluster 拥有固定的 FC2 tile-pair residue class |
| local/remote Return | 从 canonical mblock/header 静态枚举；固定 channel owner |
| token Reduce | `token = reduce_worker + round * num_reduce_workers` |

为了避免总是让同一 cluster 承担 phase 尾部，FC1、FC2、不同 wave 可使用确定性的
rotation offset；rotation 只改变 owner，不改变 tile 内数值顺序。

### 4.3 FC1/FC2 pair 枚举

保留当前 cluster-2 multicast 友好的 pair 规则：

- FC1 优先让两个 CTA 处理同一 expert/N tile 的相邻 M tile以共享 weight B；若该
  expert 最后只有一个 M tile，则配对相邻 N tile以共享 activation A。
- FC2 配对同一 M tile 的相邻 N tile，共享 SwiGLU activation A。

每个 wave 先枚举该 wave 的所有 FC1 pair，再枚举同一 wave 的所有 FC2 pair：

```text
for wave:
  for pair = rotated_cluster_id; pair < fc1_pairs(wave); pair += clusters:
    wait x_ready(mblock)
    run FC1 pair

  for pair = rotated_cluster_id; pair < fc2_pairs(wave); pair += clusters:
    wait s_ready(mblock)
    run FC2 pair
```

没有全局 FC1->FC2 barrier。一个 cluster 做完自己的 FC1 residue 后可以进入 FC2；
它只等待目标 mblock 的 `s_ready`。尚未结束 FC1 的 cluster 不依赖 FC2，因此不会
形成环形等待。若静态 assignment 暴露明显 head-of-line stall，可在**同一个 worker
自己的 residue list** 内加入小窗口 lowest-ready lookahead；仍不允许跨 worker steal
或恢复全局 claim cursor。

### 4.4 readiness 仍需原子同步

删除 task queue 不等于删除同步：

```text
Dispatch payload/header store
  -> system/device release
  -> x_arrived[mblock, source_or_channel] += rows

FC1/SwiGLU slice store
  -> device release
  -> s_ready[mblock] += 1

FC2 tile store
  -> device/system release
  -> y_ready[mblock] += 1

Return contribution store
  -> system release
  -> route_ready[token, topk_slot] = epoch
```

消费者在 successful readiness test 后做 acquire，再读取 payload。GIN slot 的
ready/consumed/credit generation 也保留。V4 删除的是 queue reservation、task CAS
和动态 selector，不是 release/acquire 或 bounded-ring backpressure。

## 5. 单机流程

单机使用一个不含 NCCL device/GIN 代码的 cooperative persistent kernel。建议
保持一个 CTA/SM、cluster size 2；CTA 内继续使用两个 WGMMA math warpgroups、一个
TMA producer 区域和若干 communication warps，具体线程布局由 autotune 决定。

完整阶段如下：

```text
1. invocation epoch/setup
2. math lanes: X staging + combine order
   control lanes: local route histogram + (expert, source, slot) table
3. publish source counts/table, node-wide CountsReady barrier
4. destination prefix/mblock geometry
5. homogeneous control warps destination-pull canonical rows ─┐
6. statically owned FC1 -> SwiGLU -> FC2                 ├─ readiness overlap
7. Return contribution publication                         ┤
8. node-wide BeforeCombine barrier + fixed-order Reduce    ┘
9. clear reusable counters + node-wide AfterClean barrier
```

实测后选择 destination-pull 的原因：

- destination 独占 canonical row 几何和 readiness publication，source 只发布紧凑
  route id/count，不需要计算 remote row 或对 remote counter 做逐 route 原子更新；
- row 按 source-rank round-robin 确定排列，相邻 pull warp 自然分散到不同 peer；
- X 的 source-local symmetric staging 在同一 kernel 开头由原本空闲的 math lanes 完成，
  替代 MegaMOE 在 graph 外的 input copy，routes/gates 直接从调用方稳定地址读取；
- control warp 以 `global_warp + round * global_warps` 闭式枚举 row，不需要
  DispatchTask materialization、queue、claim cursor 或专用 CTA；
- pull payload 落在本 GPU 后只需 device-scope release/acquire，不再需每行的
  system-scope payload publication。

## 6. 多机流程

### 6.0 复审：进度、容量与首版实现边界

2026-09-22 复审发现，原先 6.1–6.3 的 channel ring 方案还缺少三项证明：

- 固定 TX owner 阻塞等待 credit 时，必须有独立、始终 resident 的 RX owner
  继续返回 credit；同一 owner 顺序处理 TX/RX/Return 会形成跨 rank 等待环。
- `(destination, mblock)` bucket 数不能仅由每个 expert 的总 count 推出。
  若沿用这种分包，必须交换各 channel 的精确 descriptor 数；不能把 route count
  除以 chunk size 当作 terminal sequence。
- NIC 完成读取 TX staging、远端 payload 可见、消费者结束读取 RX staging
  是三种不同事件。weak signal 自身不能替代 flush/acquire，远端 credit 也不能
  代替本地 NIC completion 后才可复用 TX storage 的条件。

首版多机 inference 使用完整 invocation arena，取消 activation ring 的复用
依赖。跨机数据按 source rail 推送一次到目标节点的注册 staging，目标节点再用
现有静态 destination-pull 填充 canonical rows。这里的 pull 只发生在节点内部。
每个 source 的 route list 按 expert 紧凑排列，附带 expert prefix/count；不发送
V3 的 packet/fanout/wave 描述，也不构造动态 work queue。

Return 同样使用独立的 TX/RX arena。FC2 向 source 所在 rail 写紧凑 contribution，
按目标节点的 expert 区间排布；GIN 只发送该区间的实际 route 数。source 用本次
route plan 中的 compact index 直接读取返回值，保持现有 `(expert, top-k slot)`
固定 FP32 combine 顺序。TX 与 RX 不得别名：双向传输可能在对端读完 TX 前覆盖
相同地址。

所有 rank 经 GPU collective 确认阶段完成后才复用 arena，因此不需要 ring credit、
terminal packet 或 task claim。最终版用下一轮 Dispatch barrier 保护 Return TX/RX
复用，不再额外增加 Combine 后的跨机 barrier；详细复用证明见多机记录。
首先验证阶段间有明确边界的单 kernel 协议；跨机
Dispatch/Return 与计算的重叠作为后续独立优化测量，不能宣称首版已经具备这种重叠。
容量随节点数和公开的 token capacity 推导，循环与 offset 不假设恰好两机。
不同 source 可有不同实际 token 数，传输长度必须取 source 发布的实际计数。
pool 的容量、H/I、expert shard 和每节点 rank 数是整个 EP 组的对称合同；
expert 必须均匀连续分片。实际接收行数由设备计数，超出调用者预留容量时
集体报错，不要求调用者总是分配 `world * max_tokens * top_k` 行。
目前硬件只能验证两机；三机以上需要后续实机门禁。

实现模块：`static_v4/route_plan.cuh` 共享 combine order 和 M128 geometry，
`destination_pull.cuh` 共享节点内 pull，`initialize.cuh`/`collective.cuh` 共享
grid/节点内同步；`network_{layout,access,route_exchange,collective}` 只负责
注册区地址、跨机发布与 GIN 阶段边界。FC1/FC2 静态 wave 和数值 micro-kernel
仍是同一份实现。每个 GIN context 由唯一 CTA owner 使用，并分别执行 Put fence，
之后节点内 rank barrier 才允许其他 GPU 消费该 rail 的数据。

以下 6.1–6.3 保留为后续有界流式传输的设计候选，不作为首版 ring 协议已经完备
的声明。

多机 kernel 与单机共享 route count/count exchange、canonical geometry 和 compute
scheduler，但不执行单机 symmetric-input copy/route-table publication；remote route
由 source-side push task 直接携带。transport specialization 如下：

```text
source GPU
  local route plan
    ├─ same-node task -> NVLink direct push -> expert GPU canonical row
    └─ remote task    -> fixed GIN channel pack/push
                              |
                              v
target rail GPU fixed RX owner -> static intra-node relay -> expert GPU row
                                                           |
                                                           v
                                                     FC1 / FC2
                                                           |
source contribution <- fixed GIN return channel <- static Return pack/relay
        |
        v
fixed top-k Reduce
```

### 6.1 跨机 Dispatch

count exchange 后，每个 channel 已知精确的 task 数和 terminal sequence。固定 owner
按 sequence 顺序：

1. 等待 TX ring slot credit；
2. 将 descriptor、canonical row、source token/top-k、gate 和 X payload 打包；
3. `gin.put` 到目标 rail GPU 的同一 channel/sequence；
4. strong/weak signal 按 NCCL device contract 发布 ready；
5. 前进到该 owner 的下一个静态 task。

不存在“哪个 CTA 抢到这个 channel/task”的 CAS，也不需要 End task；terminal
sequence 已由 route plan 给出。

### 6.2 目标节点 relay

每个 RX channel 有固定 receive owner。它按 sequence 等 ready，chunk 内的 relay
warps 按固定 row stripe 工作。由于一个 DispatchTask 只覆盖一个 destination rank
和一个 destination mblock，relay 完成 chunk 后可用一次 release add 发布该 chunk
的行数，而不是每行做一次全局 RMW。消费完毕后返回该物理 ring slot 的 credit。

same-node Dispatch 和 cross-node relay 都写同一 canonical row arena，并累加同一组
`x_arrived[mblock, source/channel]`；compute 不需要知道一行来自 NVLink 还是 GIN。

### 6.3 Return

Return 不再由 FC2 epilogue向 append-only queue 发布 descriptor。固定 Return owner
从 canonical header/mblock 枚举其任务，等待相应 `y_ready[mblock]`：

- same-node contribution 直接 NVLink push 到 source 的
  `[token, topk_slot]`；
- remote contribution 由固定 return channel owner 按 sequence 打包并 GIN push；
- source RX 写 contribution 后 release-publish route epoch；
- Reduce worker 等本 token 的有效 route epochs 后，按 top-k slot 固定顺序求和。

## 7. Notify 与 main 的融合边界

### 7.1 Inference

只有一次 kernel launch。kernel 内部的 `grid.sync()` 和 EP/LSA/Gin barrier 表达：

```text
RouteCountComplete
  -> CountsReady（所有 rank 的 count matrix 可见）
  -> PlanFrozen（本 rank prefix/task descriptors 完成）
  -> Main phase
```

`CountsReady` 是 invocation 内 GPU-side barrier，不是 CPU `handle.wait()`。main phase
的 TMA/math warp 在此之前只可初始化 shared barrier/TMA descriptor 等不依赖 route
的数据，不得读取未完成的 count/route plan。

hot path 不做 D2H status copy、不读取 incoming row 数、不申请显存。错误写入 device
status，并以 collective-safe 的方式让所有 rank 进入相同 barrier/terminal 路径；
同步 API 或 debug 模式再把错误带回 host。

需要保存中间激活的 training Forward 不走这条单-launch 路径，而是使用第 8 节的
plan/allocate/main 边界。

### 7.2 PDL 的准确定位

PDL 可以让 fused MoE kernel 在前驱 kernel 尚未完全结束时被调度，并先初始化与
输入无关的 shared/barrier 状态；第一次读取 `routes/x/gates` 前仍必须执行
`cudaGridDependencySynchronize()`。这只是 launch-overhead/前驱尾部优化，不能替代
本 kernel 内的 CountsReady barrier。

Inference 中 Notify 已经融合，不再存在“用 PDL 让 main 提前越过 Notify”的需求。
Training 的 main 又依赖 host 新分配的指针，因此也不能提前 launch。

## 8. 保存中间激活时的内存策略

### 8.1 唯一路径：memory-tight plan/allocate/main

```text
route_plan_kernel
  -> stable source_slot + count exchange
  -> canonical prefix / exact R_actual / exact P_actual
  -> immutable DispatchTask/compute geometry
  -> 不发送 activation payload

host allocation boundary
  -> 只读取本 rank R_actual/P_actual 和 collective status
  -> framework allocator 分配 canonical arena 与 saved tensors

main_from_plan_kernel
  -> consume exact same plan; 不重复 route counting/exchange
  -> source-push / FC1 / FC2 / Return / Reduce
  -> 保存 X/A/B/gate/header/plan
```

这条路径保留 kernel 间 host allocation boundary，不能宣称为低延迟 fused path。
它是所有 `retain_backward_state=true` 调用的固定语义，不是 overflow fallback，也不
用于 SGLang inference。

每个 live Forward tape 持有自己实际大小的 tensor allocation 和 immutable plan
lease，直到 Backward 完成才释放。runtime 可以限制同时存活的 tape 数，但达到限制
时只能等待或返回 `ResourceExhausted`，不能覆盖旧 tape。只有显式 allocator OOM
可以使整个 invocation 失败；不存在“继续运行但少算一部分 route”的状态。

plan scratch 只依赖本 rank 调用时已经知道的 `T_local*K`、固定 expert/world count
matrix 和少量 prefix/task 上界；它不需要预知其它 rank 的最大 token 数。各 rank
实际 token 数随 count exchange 一并成为本 invocation plan 的一部分。

### 8.2 为什么这条路径不使用 PDL

main kernel 的 launch arguments 必须在 launch 时已有有效的 saved-buffer pointer。
若先 launch 一个常驻 main kernel，再让它等待 host 分配并写入 device pointer slot：

- host 仍需等待 planning 结果；
- 等待中的 cooperative kernel 会占住 SM，容易阻塞负责发布依赖的工作；
- launch 参数/graph node 不能在运行中安全改写；
- allocator、stream ordering、失败传播和 graph capture 都更复杂。

除非未来引入预留稳定 VA、预先映射的 memory pool——本质上又回到最坏容量预分配——
否则 PDL 对这个 allocation dependency 没有帮助。因此 V4 不实现半融合的
pointer-mailbox 方案。

## 9. Buffer 与生命周期

V4 将长期资源分为：

| 资源 | 生命周期 | 是否对称/注册 |
|---|---|---|
| count exchange、epoch、readiness | runtime capacity 生命周期 | 单机 peer-mapped；多机控制区注册 |
| SourceRoute、DispatchTask、prefix plan | Forward/Backward plan lease | 本地；按本次 local routes/geometry |
| inference canonical X/S/output arena | inference invocation slot | peer 可达；遵守 inference capacity/graph 合同 |
| training canonical X/S/output arena | Forward tape 生命周期 | plan 后按本 rank `P_actual` 精确分配并建立 peer mapping |
| GIN TX/RX ring 与 credit | runtime capacity 生命周期 | NCCL registered |
| contribution `[T_local,K,H]` 与 route epochs | invocation/tape | source peer/GIN 可达；local shape 已知 |
| saved X/A/B/gate/header | Backward 完成前 | training-only；按 `R_actual/P_actual` |

第一版静态 scheduler **继续使用完整 canonical X/S arena**：inference 使用其既有
capacity slot，training 在 plan 后按实际几何申请。
这样能先把 queue/claim/Notify 变量与 ring-reuse 变量分开，避免一次重构同时引入
source-push 的 ring deadlock。待静态 full-arena 路径达到性能与正确性目标后，再做
独立消融：

- 单机 inference 把 X/S 换成 generation-tagged Mblock ring；
- transport task 必须按 logical mblock generation 单调推进，并显式等待 empty credit；
- training saved_x/saved_ab 仍是完整 tape，不随 transient ring 覆盖；
- 多机只有在有完整的 channel/window backpressure 证明后才启用 canonical ring。

因此“借鉴 MegaMOE 的 persistent buffer”在 V4.0 中体现为 inference 固定地址、
epoch 复用和简单 full/ready counter；training 的 saved tensors 则明确保持动态
实际尺寸。物理 Mblock ring 是后续正交优化，不与本次 scheduler 重构绑死。所有
arena 的 correctness capacity 遵守第 3.2 节，不能通过 drop 换取固定显存。

## 10. Kernel specialization 与代码组织

当前 V4 控制面独立组织如下，数值主体复用既有模板：

```text
src/cuda/static_v4/
  contract.cuh                    # route/header 合同
  memory_order.cuh                # 单向 release/acquire PTX primitives
  initialize.cuh                  # 共享 invocation/pool 初始化
  route_plan.cuh                  # 共享 combine order/M128 geometry
  collective.cuh                  # grid/节点内 barrier 与 cleanup
  single_node_route_exchange.cuh  # fused count/table/X staging
  destination_pull.cuh            # 两种拓扑共享的节点内 pull
  network_layout.hpp              # 注册区布局与静态 context 上限
  network_access.cuh              # source rail/紧凑 Return 地址
  network_route_exchange.cuh      # expert 排序与跨机 Dispatch
  network_collective.cuh          # context-owned GIN put/fence/Return
  fc2_epilogue.cuh                # FC2 contribution publication
  single_node_finalize.cuh        # reduce/finalize
```

GEMM 与静态 FC1/FC2 task 枚举仍复用 `persistent_v3/forward` 中的数值主体，
通过 `StaticSchedule` 编译期分支移除 V3 queue/cancellation hot path，不复制一份 GEMM。

Host 侧复用 capacity-based pool 和 `TensorRuntime`。`DeviceDispatchHandle` 在
eager/graph preparation 期间提供容量和地址合同；graph replay 不构造 handle，
也不执行 CPU Notify 或 `handle.wait()`。V3 保留为训练和显式 fallback。

单机/多机在同一聚合 translation unit 中实例化为不同 kernel，使用
`StaticLocal`/`StaticSchedule` 在编译期消除无关 transport 分支；不是运行时
切换同一 kernel 的 GIN 状态。`RetainBackwardState` 保持训练数值主体的复用。

## 11. 与 CUDA Graph、slot 和并发 invocation 的关系

- 同一 framework stream 上模型逐层执行时，一个 invocation slot 可依靠 stream
  serialization 安全复用，不需要 CPU 等上一层 `handle.wait()`。
- Inference Graph replay 使用固定 slot 地址和递增/graph-local epoch；routes 和实际
  counts 可在 inference capacity 内变化。memory-tight training 不支持把
  plan/host-allocation/main 整体捕获进 CUDA Graph。
- 当前 V4 要求整个 EP 组以相同次序串行调用；两个 slot 不代表多 stream 并发
  已经安全。未来并发不仅要隔离 SourceRoute/canonical/readiness，还必须隔离
  GIN context/barrier 状态并证明跨 rank 的 launch 进度；不在本轮支持范围内。
- inference slot 是 runtime 预分配的 invocation workspace；training tape 是 plan 后
  按实际尺寸分配的独立 lease。二者都不是 FC1/FC2 ring slot 或 SGLang request slot。
- slot 复用通过 CUDA event/stream dependency 或 graph 固定依赖表达；host 仅在所有
  slot 都被真正并发占用时才需要等待，不在每层无条件同步。

## 12. 实施顺序

### Phase A：冻结合同和可观测性

1. 给 V3 增加统一 stage timing/counter，保留当前结果作基线。
2. 写 host/device 共用的 row/task 枚举器单元测试。
3. 固定数值合同测试：N/K tile、rounding、top-k reduction、masked/duplicate route。

### Phase B：单机 inference V4

1. 独立验证简化 route plan 与 canonical row。
2. 用完整 arena 实现 fused route/count/静态 source-push，不接 GEMM，验证每行/header。
3. 接入静态 FC1/FC2 pair scheduler，先关闭 overlap，再逐个打开 Dispatch/FC1/FC2/
   Return/Reduce overlap。
4. 删除 V4 路径中的 FC1/FC2/Reduce queue 与 claim cursor。
5. 对 DispatchChunkRows、comm warp 数、mblock readiness 批量粒度做 autotune。

### Phase C：多机 inference V4

1. 静态 GIN send channel ownership；
2. 静态 RX/relay ownership 与精确 terminal sequence；
3. 静态 Return；
4. 加入 skew、慢 channel、ring wrap 和故障注入测试；
5. 确认无跨机 pull fallback。

### Phase D：training tape

1. `StableTape` source slot；
2. 实现 `route_plan_kernel` 和精确 `R_actual/P_actual` host allocation；
3. 实现 `main_from_plan`，证明不会重复 route counting/exchange；
4. Backward 持有并消费精确 Forward plan/tensors。

### Phase E：独立优化

1. static-owner 小窗口 ready lookahead；
2. single-node Mblock ring；
3. specialized epilogue；
4. task/channel/wave autotune。

## 13. 测试与验收门禁

### 正确性

- EP1/2/4/8 单机，至少 EP16 两机；
- token `1/2/64/128/2048`，balanced、单 expert 极端 skew、空 expert；
- duplicate expert、masked route、tail Mblock、不同 gate；
- 所有 route 压到一个 rank/一个 expert 时不得 drop；容量不足只能触发 collective
  grow fallback 或显式 OOM；
- 与当前 WeftMoE reference 对齐；同一数值合同下检查 bitwise FWD，training 检查
  tape/Backward row plan 和梯度；
- generation wrap、slot reuse、GIN ring wrap、任意 rank error 均不得 deadlock；
- compute/relay/GIN owner 数变化后每个 task 恰好执行一次。

### 热路径约束

- inference 一次 forward 只 launch 一个主 kernel；允许框架自身的输入/output event，
  不允许 Notify->main 的 host wait；
- steady state 无 `cudaMalloc/cudaFree`、无 D2H count/status、无 CPU spin；
- V4 kernel 中不存在 FC1/FC2/Return/Reduce work-claim atomic；
- readiness atomic 数量按 mblock/chunk 计，而不是无条件按 row 计；
- CUDA Graph 内 routes 可变但不越 capacity。

### 性能消融

必须分别报告 kernel 内 stage 和端到端，不只看总时间：

1. V3 dynamic queue vs V4 static owner；
2. split Notify/main vs fused；
3. stable route vs atomic inference route；
4. source-push 的 route/task materialization、copy、readiness 分项；
5. full arena vs后续 Mblock ring；
6. baseline epilogue vs specialized epilogue；
7. 单机 EP8 decode/prefill，以及两机 GIN 的 balanced/skew；
8. WeftMoE V3、WeftMoE V4、MegaMOE、SGLang baseline 同条件三/四方表。

只有在 Nsight/timeline 证明 static ownership 没有把 queue 开销换成明显 head-of-line
空转后，V4 才成为默认路径。

## 14. 已知风险与对应约束

1. **静态 owner 的尾部不均衡**：用全局 flatten 后的 tile pair 做 residue 分配，必要
   时只在 owner 私有列表内 lookahead；不恢复 steal/CAS。
2. **融合 kernel register/SMEM 膨胀**：单机、多机拆 TU/entry；communication 与
   math warp 做寄存器重分配，逐 specialization 检查 SASS、spill 和 occupancy。
3. **cooperative grid deadlock**：launch 前验证所有 CTA/cluster 可同时 resident；
   任意错误路径必须加入同样的 EP/grid barrier，不能提前 return。
4. **多机 channel 顺序与 credit**：一个 channel 只有一个 submit owner；packer 可并行，
   但提交严格按 sequence。terminal 数来自冻结 plan。
5. **source-push 与未来 ring 的环形等待**：V4.0 使用 full arena；ring 只有在 transport
   task generation 单调性和 empty-credit 证明完成后启用。
6. **训练 allocation boundary**：明确保留 plan/main 两 kernel 和 host allocation，
   不用 PDL 掩盖这一事实；allocator OOM 必须在 payload phase 前 collective fail。
7. **极端 routing skew**：使用 plan 得到的逐 rank `R_actual/P_actual`，不按平均负载
   分配；绝不让部分 rank 继续执行一个被截断的 plan。

## 15. 最终目标状态

重构完成后，Forward 的核心状态机应缩减为：

```text
Count routes
  -> Exchange counts / freeze geometry
  -> Static Dispatch owners publish mblock readiness
  -> Static FC1 owners publish S readiness
  -> Static FC2 owners publish output readiness
  -> Static Return owners publish route readiness
  -> Static Reduce owners produce output
```

每条边只有 generation-tagged release/acquire；每个 task 的 owner 可由编号直接计算。
单机与多机只在 `Dispatch/Return` 的 transport primitive 上不同：单机为 NVLink/
IPC direct source-push，多机为 GIN push 加目标节点 relay。GEMM、row layout、task
geometry、precision、no-drop 和 tape contract 完全共享。这是本次重构要达到的
“像 MegaMOE 一样简单”的实质，而不是机械复制它的 pull 协议。
