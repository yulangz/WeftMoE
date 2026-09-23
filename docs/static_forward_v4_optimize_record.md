# Static Forward V4 优化记录

日期：2026-09-22；2026-09-23 补测单机 EP8 的 1620 MHz V3/V4 对照。

状态：单机与多机 inference-only 路径已完成主要重构；training 仍使用 V3。
多机实测见 [`static_forward_v4_multinode_record.md`](static_forward_v4_multinode_record.md)。
本文记录单机 Static Forward V4 优化，不替代
`forward_optimization_history.md` 中大 token/多机历史。

拓扑约定：用户要求的主矩阵为 **单机 EP8（1×8）和双机 EP16（2×8）**。
此前多机记录中的 EP8（2×4）是误解需求后增加的实验，不是单机 EP8，不能替代本节补测。

## 2026-09-23：单机 EP8，1620 MHz 锁频对照

本次确实为一台机器上的 8 GPU，launcher 使用 `--nodes 1`，所有 rank 的
backend 均为 `local-cuda-p2p`，没有跨机 GIN。H=2048、I=512、64 experts、
top-k=4、BF16；token 数均为每 rank，8K=8192、20K=20480。

使用与多机对照相同的两个冻结 binary，但执行其单机特化：V3 由
`configs/static_inference_v3_20260923/` 构建，设置
`WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`；V4 使用
`configs/static_inference_20260922/`，unset disable 变量。
V3 保留 N128/N256/K64、stages 4/2、claim=2；V4 保留 N64/N128/K128、
stages 3/3、wave=32。V3 的 GIN sender 配置在单机不生效，没有为 RDMA 留出 CTA。
这次没有针对单机各长度重新 autotune；结果比较的是这些公开配置，不是各自最优
配置的穷举结果，也不是 scheduler 单变量消融。

每个规模运行 V3 → V4 → V4 → V3 四个独立 session，每个 session 预热 10 个
sample、测量 20 个 sample，每 sample 20 次 CUDA Graph replay；开启
`WEFTMOE_BENCH_SYNC_RANKS=1`，rank 同步在 CUDA event 计时区间外。
每个 sample 先取 rank-max，再合并同版本两个 session 的 40 个 sample，取排序
后的第 20 个作为 p50。每个 case 的 pool 按自身 token capacity 独立创建。

两者都测完整 graph：V3 为 Notify + Forward，共 11 nodes（2 kernel、3 memcpy、
6 event-record、0 event-wait）；V4 为 1 kernel、0 memcpy/event。
不把 V3 的 kernel 内计时或历史 eager 的 270-us 数字与 V4 graph 时间混比。

| tokens/rank | V3 graph p50（us） | V4 graph p50（us） | V3/V4 加速比 | V4 延迟变化 |
|---|---:|---:|---:|---:|
| 1 | 299.78 | 64.76 | 4.63× | -78.4% |
| 64 | 368.44 | 74.07 | 4.97× | -79.9% |
| 8192 | 1487.58 | 1600.65 | 0.93× | +7.6% |
| 20480 | 3289.72 | 4767.00 | 0.69× | +44.9% |

结论：**单机 decode 加速 4.63–4.97×，但 8K/20K 分别慢 7.6%/44.9%。**
前一轮单机 prefill 的 24.9%/36.0% 收益是相对旧 V4，不是相对 V3。
单机大 token 也未达到优于 V3 的目标。单机没有跨机阶段，因此其回退不能归因于
多机 Dispatch/Return barrier；应单独检查 GEMM tile、static wave、节点内搬运
和 FC2/Combine 路径，尚未做消融，不能声称已定位根因。本次未修改生产调度。

16 个 session 全部完成，128 次 rank/output hash 检查全部与已通过 PyTorch 参考
的单机 V4 输出一致。04:15:13–04:18:11 UTC，200 ms 间隔采集的 7,096 条时钟
记录全部为 SM=graphics=1620 MHz；每张 GPU 均有负载样本，负载下显存为 3201 MHz。
测试期间没有并发编译或其他 GPU 测试，测试后保持锁频。

完整 rank/sample、两次 session p50、输出 hash、图节点和 binary/config SHA256 见
[`../benchmarks/static_inference_single_node_1620_20260923.json`](../benchmarks/static_inference_single_node_1620_20260923.json)。
原始日志为 `/tmp/weft-single-1620-abba-20260923-t{1,64,8192,20480}-{trial}-{v3,v4}.log`，
时钟日志为 `/tmp/weft-single-1620-clocks-20260923.csv`。
双机 EP16（2×8）的锁频数据沿用多机记录；2×4 仅保留为额外实验。

复现时，两种版本均设置 `LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64`、
`WEFTMOE_BENCH_EXPERTS=64`、`WEFTMOE_P2P_BENCH_TOKENS=<对应长度>`、
`WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY=1`、`WEFTMOE_FORWARD_GRAPH_BATCH=20` 和
`WEFTMOE_BENCH_SYNC_RANKS=1`。本机环境为 `NCCL_IB_DISABLE=1`、
`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`。V3 示例（V4 替换 binary 并 unset disable）：

```bash
WEFTMOE_DISABLE_STATIC_FORWARD_V4=1 python3 tests/run_local_activation_p2p.py \
  --binary "$PWD/build/static_inference_v3_20260923/test_cuda_notify" \
  --world 8 --top-k 4 --forward-benchmark --rounds 30 --timeout 180
```

以下章节保留 2026-09-22 的历史记录，不与本节 1620 MHz 对照跨时钟计算收益。

## 2026-09-22 复审：decode 与 prefill

基线提交 `8f3fa57` 保存复审前工作区里的完整实现。以下使用同一 H=2048、
I=512、64 experts、top-k=4、EP8 fixture，CUDA Graph 每个 sample 回放 20 次，
10 个 warmup sample + 20 个测量 sample；每个 sample 取 rank-max。
8K/20K 分别明确为 **8192/20480 token per rank**。这是相邻构建消融（B），
不是交错 A/B，微秒级差异仍需考虑测量噪声。

| 改动 | 1 token | 64 tokens | 8192 tokens | 20480 tokens |
|---|---:|---:|---:|---:|
| 复审基线 | 59.79 | 72.60 | 1973.02 | 7161.11 |
| FC2 相邻 8 lanes 合并写一个 contribution row | 58.64 | 68.92 | 1706.21 | 6412.08 |
| 加入 32-Mblock 静态 FC1/FC2 wave | 59.08 | 69.67 | 1564.15 | 4869.70 |
| M128 readiness 直接索引 | 58.45 | 69.33 | 1482.39 | 4580.19 |

单位为 us，均为 graph rank-max p50。最终相对基线改善约
2.2%/4.5%/24.9%/36.0%。四个规模的所有 rank output hash 均与基线一致。

三个改动的依据：

- FC2 原 epilogue 每 lane 写不同 token 行，H-strided 地址把 warp 的 16-byte
  store 分散为大量 NVLink 事务；现在 8 lanes 写同一行相邻的 128 bytes。
- 原静态 scheduler 对整个 invocation 先枚举 FC1，再枚举 FC2。现在每个
  32-Mblock wave 使用同样的 cluster residue 枚举，完成本 worker 的 FC1 后
  进入本 wave 的 FC2，再推进下一 wave。没有全局 phase barrier，也没有
  queue/claim。较早 wave 的 FC2 不依赖较晚 wave，等待关系保持无环。
- V4 已经直接生成 M128 几何，readiness 无需沿用 V3 小 Notify tile 的线性
  扫描。由 `expert_tile_base + local_row / 128` 直接定位，避免扫描长度随
  expert rows 增长。

重现使用 `tests/run_local_activation_p2p.py --world 8 --top-k 4
--forward-benchmark --rounds 30`，环境变量为 `WEFTMOE_BENCH_EXPERTS=64`、
`WEFTMOE_P2P_BENCH_TOKENS=<上述规模>`、`WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY=1`、
`WEFTMOE_FORWARD_GRAPH_BATCH=20`。保持原 N64/N128/K128、FC1/FC2 各 3 stages
配置。原始本机日志为 `/tmp/weft-v4-audit-{baseline,coalesced,waves,direct-ready}-*.log`。

多机接入后，共享了 stable insertion combine order、M128 geometry 和节点内
pull 模块。最终构建的单机 EP8 回归为 **58.12/68.16/1480.64/4583.34 us**，
四个规模所有 hash 仍与基线相同，没有单机性能回退。冻结构建参数与完整测量
数据见 `configs/static_inference_20260922/` 和
[`../benchmarks/static_inference_20260922.json`](../benchmarks/static_inference_20260922.json)。

## 1. 结论

主测试为 EP8、H=2048、I=512、64 experts、top-k=4、BF16、64 token/rank。
从第一个可重复的 homogeneous static graph 到最终版，rank-max graph p50 从
106.189 us 降到 72.500 us，总改善 **33.689 us / 31.7%**。这个总量不能按后文
每行简单相加，因为其中既有严格 A/B，也有多项同时变化的重构里程碑。

| 里程碑 | 指标 | 结果 |
|---|---|---:|
| V3 fallback（历史交错 A/B） | eager Forward rank-max p50 | 270.320 us |
| 早期过渡 V4 | eager total，历史口径 | 267.120 us |
| 首个可重复 homogeneous source-push | graph p50 | 106.189 us |
| notify/route 融合后的 source-push | graph p50 | 85.317 us |
| destination-pull + graph 节点清理 | graph p50 | 77.131 us |
| release/acquire、cancellation 和 barrier poll 优化后 | graph p50 | **72.500 us** |
| 真实 tensor runtime，双层 child graph | 每层 | **74.237 us** |
| MegaMOE graph kernel | graph p50 | 62.6 us |
| MegaMOE 图外 3 份 input copy | 单层 | 11.5 us |
| MegaMOE 公平总口径 | graph + copy | **74.1 us** |

最终 WeftMoE 裸 graph 已比 MegaMOE 的 graph+copy 口径快约 1.6 us；包含
PyTorch tensor runtime 和 child-graph 调度后差 0.14 us，约 0.2%，在当前测量噪声内。
MegaMOE 的 62.6-us kernel-only 不能直接与 WeftMoE 的应用结果对比。

## 2. 计时口径与证据等级

本轮日志同时存在四种指标：

- `forward_slowest_rank_us`：kernel 内 device clock 记录的 rank-max span；用于解释阶段，
  不与 MegaMOE CUDA event kernel-only 混用。
- `slowest_rank_us`：批量 CUDA Graph replay 的每次耗时，每个 sample 取 8 rank 最慢值；
  这是核心性能指标。
- tensor runtime 的 `TWO_LAYER_GRAPH_PER_LAYER_US`：PyTorch 外层 graph 调度两层
  WeftMoE child graph 的每层耗时；最接近 SGLang 接入形态。
- `wall_slowest_rank_us`：包含 CPU launch/调度干扰；本轮小 kernel 上抖动大，不用于
  最终归因。

后文用三个证据等级：

- **A：严格交错 A/B**。同 shape、同路由和权重，A/B/B/A 反转顺序，分别 warmup。
- **B：相邻构建**。测试条件一致，但没有做交错重复；适合判断 us 级以上的差距。
- **C：里程碑 bundle**。期间有多项源码同时变化，只能说明整个方向的收益，
  不能分配到其中某一行代码。

所有 delta 都定义为 `after - before`；负数表示变快。

### 2.1 V3 基线与两个不同的 106 us

历史最可信的 V3 基线是同一进程中与过渡 V4 逐 iteration
交错、并每轮反转顺序的 **270.320 us**。该数字是 `run_forward`
的 rank-max median，不包含外部 notify 的 host wall time。当时的过渡 V4
为 267.120 us，只快 3.200 us；因此“改成静态 task”的第一版
本身并没有带来从 270 us 到 106 us 的改善。

2026-09-22 使用当前 binary，设置
`WEFTMOE_DISABLE_STATIC_FORWARD_V4=1` 复测同一 shape，得到 V3
`forward` rank-max p50 **276.256 us**。测试受未同步的 host 调度干扰，
因此本文仍以上述 270.320-us 交错 A/B 为正式基线；新测只用于
确认 V3 仍在 270-us 量级。在当前 serialized graph replay harness 中
禁用 V4 后没有产出 graph benchmark marker，因此暂时没有可靠的
同 harness V3 graph 数字，也不用 eager 数字外推一个 graph 结果。
这是当时的测量限制；2026-09-23 已用现有冻结 V3 参数补齐 graph 对照，见本文首节。

日志里实际上存在两个容易混淆的“106 us”：

- `weft-v5-prepublish-custom-grid-n128-t64.log` 是 kernel 内 device-clock
  `forward_slowest_rank_us.p50 = 106.240 us`；
- `weft-static-v5-jit39-graph-bench64.log` 是更晚版本的 20 次批量
  CUDA Graph replay，`slowest_rank_us.p50 = 106.189 us`。该版本内部
  Forward 已经调到 85.632 us 量级，两者的数值相近只是巧合。

#### V3/old-V4 到首个 V5 的结构重写

这一段是 C 级 bundle，没有留下逐项交错 A/B。过渡 V4 的
267.120 us 到首个可反复测量的 V5 `finalize` 版 148.736 us，
改善 118.384 us。可以从 device timeline 确认的结构变化是：

- FC2 epilogue 直接把 contribution 写回 source rank，取消 V3/old-V4
  独立的 Return task/queue/copy 阶段。old-V4 的 Return span 是
  106.672 us，首个 V5 的 `return_us` 为 0；这是最大的结构性差异，
  但 106.672 us 与总收益不能直接相加，因为 Return 原本与 FC2/Reduce 有重叠；
- FC1/FC2 使用有限静态枚举与更短的 ready 协议，去掉 V3 的
  queue/cursor/CAS 任务领取和 Return scheduler。同时期阶段 span 从
  Dispatch/FC1/FC2 = 48.384/37.824/34.864 us 下降到
  30.464/30.336/21.632 us；
- Reduce 直接消费固定 contribution slot，不再等待/遍历 Return queue。

阶段 span 互相重叠，上述数字只能证明瓶颈移除到了哪里，不能
当成独立 delta 求和。

#### 首个 V5 从 148.736 us 到 device-clock 106.240 us

这一段保留了连续构建日志；下表为相邻日志的 p50，delta 仍是
`after - before`：

| 版本/主要变化 | forward p50 | delta |
|---|---:|---:|
| V5 finalize 基线 | 148.736 us | — |
| generation 协议收紧 | 146.336 us | -2.400 us |
| runtime initialization 精简 | 133.568 us | -12.768 us |
| minimal initialization | 132.096 us | -1.472 us |
| custom generation 路径 | 128.064 us | -4.032 us |
| independent scheduler | 127.904 us | -0.160 us |
| FC2 每次 claim 1 tile | 119.968 us | -7.936 us |
| fast finalization | 116.704 us | -3.264 us |
| 拆分 FC1 分支，缩短 accumulator 寄存生命期 | 100.864 us | -15.840 us |
| barrier/N256 实验 | 109.088 us | +8.224 us |
| FC2 N128 | 107.296 us | -1.792 us |
| merged initialization | 105.216 us | -2.080 us |
| packed-FC1 实验 | 106.112 us | +0.896 us |
| versioned-arrival 实验 | 108.096 us | +1.984 us |
| prepublish/custom-grid 收尾 | 106.240 us | -1.856 us |

最大的两个可见相邻收益是 runtime initialization 精简的
12.768 us，以及把 FC1 accumulator 限制在 FC1 task 分支生命期内的
15.840 us。FC2 claim 粒度改为 1 tile 又带来 7.936 us。这些是
相邻构建证据（B），不是交错 A/B。

之后 jit29 到 N/K winner 又把 kernel 内 Forward 从 105.888 us
降到 85.632 us（见第 4、5 节），再换用 CUDA Graph replay 的端到端
计时得到 106.189 us。因此不能把 `270.320 - 106.189` 全部
解释成 kernel 优化；其中包含了一次全面结构重写、后续 kernel tuning，
以及 device-clock/eager 与 graph replay 的口径切换。

## 3. 可独立归因的最终优化

这些是整个过程中证据最强的四组微优化。

| 优化 | before | after | delta | 相对变化 | 证据 |
|---|---:|---:|---:|---:|---|
| 保留 `programmatic_launch::wait()` 和 PDL launch 合同 | 76.925 | 删除后 77.632 | **+0.708 us 回退** | +0.92% | A |
| return-value atomic 改为单向 PTX release reduction/acquire load | 77.070 | 76.188 | **-0.883 us** | -1.15% | A |
| 删除 Static V4 有限任务热路上的 system-scope cancellation poll | 76.116 | 72.968 | **-3.148 us** | -4.14% | A |
| rank barrier 的 256-cycle nanosleep 改为 busy-poll | 72.944 | 72.500 | **-0.444 us** | -0.61% | A |

### 3.1 为什么保留 PDL wait

实验同时删除 StaticSchedule 中的 `programmatic_launch::wait()` 和对应 launch attribute。
理论上当前 graph 没有前置 kernel，wait 似乎无用；但交错 A/B 的两次结果为：

| 版本 | repeat 1 | repeat 2 | 均值 |
|---|---:|---:|---:|
| 保留 wait（build27） | 76.907 | 76.943 | 76.925 us |
| 删除 wait（build28） | 77.542 | 77.723 | 77.632 us |

删除后稳定慢 0.708 us。推测 wait 对 cluster CTA 进入和后续 barrier 到达时间有整形
作用，但这不是已被单独证明的硬件机理。最终决策是保留。

### 3.2 单向 release/acquire primitive

旧实现用 CUDA `atomic_ref::fetch_add` 发布 row-ready 和 rank signal。反汇编显示它产生
了返回旧值的强 `ATOM`，但 producer 完全不消费返回值。Static V4 改为：

- `red.release.gpu.global.add.u64` 发布本 GPU canonical row；
- `red.release.sys.global.add.s32` 发布跨 GPU rank-barrier signal；
- 对应的 device/system acquire load 轮询。

交错 A/B 为 77.070→76.188 us，收益 0.883 us。V3/training 不使用这个特化。

### 3.3 去掉 inference-only cancellation poll

旧 Static V4 在以下热点反复执行对同一 cancel word 的 system-scope atomic load：

- 每条 destination-pull row；
- 每次 static FC1/FC2 task readiness probe；
- static reduce token 等待；
- rank barrier 等待。

Static V4 的 task set 在 route exchange 后是有限且不再增长的，graph invocation 可以直接运行
到完成。因此这些 poll 被从 inference specialization 编译掉，V3/training 仍然可取消。

| 版本 | repeat 1 | repeat 2 | 均值 |
|---|---:|---:|---:|
| 保留 poll（build29） | 76.122 | 76.110 | 76.116 us |
| 删除 poll（build30） | 72.963 | 72.973 | 72.968 us |

收益 3.148 us，是最后阶段最大的单项改动。代价是这条单机 inference-only
kernel 不再支持 invocation 中途抢占；输入检查、capacity 门禁和 kernel 结束后错误处理
仍保留。

### 3.4 rank barrier busy-poll

去掉 rank signal acquire loop 中的 256-cycle nanosleep 后，两组反序 A/B 为：

| 版本 | repeat 1 | repeat 2 | 均值 |
|---|---:|---:|---:|
| nanosleep（build30） | 73.013 | 72.874 | 72.944 us |
| busy-poll（build31） | 72.449 | 72.550 | 72.500 us |

收益 0.444 us。该 barrier 只由 CTA 0 的一个线程轮询，因此没有把 busy-poll 扩散到
所有 CTA/所有 warp。

## 4. 同构 CTA 与静态任务分配

这一阶段把早期“24 个特殊 transport CTA + compute CTA”改为 132 个同构 CTA。每个 CTA
都有 384 threads：256 math lanes、32-thread TMA producer 与 96 条 transport/control lanes。
FC1 pair 和 FC2 unit 使用 `cluster + round * cluster_count` 闭式枚举，不再从全局
queue/CAS 领取。transport 同样使用 global warp stripe，没有专用 CTA。

这是大重构，没有保留“只替换 CTA ownership、其他代码不变”的严格 A/B。现存同口径
日志记录了同构 source-push 实现内部的演进：

| 里程碑 | forward p50 | 相对上一行 |
|---|---:|---:|
| 首个稳定 homogeneous 版（jit29） | 105.888 us | — |
| static task/ready 第一轮精简（jit30） | 99.392 us | -6.496 us |
| Static V4 独立 stage 合同（jit31） | 97.600 us | -1.792 us |
| K128/3-stage 候选（jit32） | 96.352 us | -1.248 us |
| 进一步缩短 task/finalize 控制路径（jit34） | 93.216 us | -3.136 us |
| 最终静态 source-push 控制结构（jit37） | 89.056 us | -4.160 us |
| N/K sweep winner | 85.632 us | -3.424 us |

jit29→N/K winner 整体改善 20.256 us / 19.1%。中间构建没有留下可应用的源码
snapshot，上表只按当时的实验目的和日志命名记录 bundle，不把 6.496 us 等数字
伪造成某一行代码的独立收益。

注册器优化的主要手段不是继续降低 `setmaxnreg`，而是把 FC1/FC2 accumulator 声明限制
在各自 task 分支的词法域内，离开 epilogue 后才进入 transport/finalize 控制逻辑。最终
Static V4 为 168 registers/thread、32-byte stack/thread、0 local bytes，每 SM 一个 CTA。这个资源
收缩与调度重构是耦合的，没有可信的独立耗时数字。

## 5. GEMM tile 与 stage autotune

同一 source-push checkpoint 上的 N/K sweep 如下。M 固定为 128，FC2 N 固定为 128，
Static FC1/FC2 均为 3 stage。

| FC1 N | K | forward p50 | 与 winner 差值 |
|---:|---:|---:|---:|
| 64 | 128 | **85.632 us** | 0 |
| 64 | 64 | 87.456 us | +1.824 us |
| 128 | 128 | 88.512 us | +2.880 us |
| 128 | 64 | 91.584 us | +5.952 us |

在 N64 下 K64→K128 收益 1.824 us；在 K128 下 N128→N64 收益 2.880 us。
因此最终保留 M128/FC1-N64/FC2-N128/K128/3-stage，而不是为了形式上与
MegaMOE 一致强制 N128/K64。

后续在更快的 fused-route checkpoint 上又做了一次 MegaMOE-aligned 候选：

- 当前 control，N64/K128/3-stage：76.992 us；
- N128/K64/5-stage：81.312 us；
- 强制对齐回退 **4.320 us / 5.6%**。

这证明“通信和计算结构参考 MegaMOE”不等于“tile 常数必须相同”。

## 6. notify/route exchange 与 main kernel 融合

融合前，route notify/plan 需要独立进度和 host-visible handle，main 只能在 plan 完成后启动。
最终路径把 route histogram、count/table publication、X/gate staging、Dispatch、FC1、FC2
epilogue/contribution 和 Reduce 放进一个 kernel，不再有每层 CPU `handle.wait()`。这里要区分两个 build 里程碑：最早的
fusion build 仍是 source-push，也没有 X/gate staging；X/gate staging 和 destination-pull 是
build14 才加入的。换句话说，“notify/main 融合”和“push 改 pull”不是同一个 patch。

具体地说，融合前的 hot path 是：

```text
routes/gates
  -> notify kernel: count/prefix/sort/materialize
  -> DeviceDispatchHandle + completion event
  -> main kernel: Dispatch -> FC1 -> FC2 -> contribution -> Reduce
```

notify 会为每条 route 物化 destination row/rank、local route order、combine
order，并沿用多机路径需要的 packet/fanout/wave 描述。单机 inference
并不需要这些通用描述的大部分，但 main kernel 只能消费完整
`DeviceDispatchView`，所以原路径仍会构造它们。

“融合”这一步本身并没有把 source-push 改成 pull。host 改为把稳定
地址的 `X/routes/gates` 指针和持久化 pool 直接传入一个 132-CTA
resident kernel（早期带 cooperative-launch attribute，build9 删除该 attribute）；kernel 的前缀阶段完成：

1. 对 route 做 expert histogram，得到本 source 到每个 expert 的 count
   和 source-local slot；
2. 将 count 发布给 expert owner，用 kernel 内 resident-grid barrier + rank
   release/acquire barrier 取代 notify completion event/host wait；
3. owner 从 `count[source, local_expert]` 本地生成
   `expert_row_count/prefix` 和 `mtile_offset/expected_rows`；
4. source 用 `expert prefix + lower-source counts + source-local slot` 在线算出
   destination row，继续走 source-push Dispatch；
5. 同一 kernel 的 route 前缀阶段生成固定 top-k combine order，不再等外部 notify
   将它作为完整 plan 的一部分传进来；真正把 combine-order/X staging 与 control
   histogram 并行的是后面的 pull build24。

因此 fusion checkpoint 取消的是独立 notify launch、host-visible handle/event
依赖，以及通用 packet/fanout/wave plan；`local_route_order` 也不再需要。但是 fused
source-push 为了让 source 写 remote canonical row，**仍然会物化并消费**
`route_dst_row/route_dst_rank/local_offset`。build14 改成 pull 后不再需要
`route_dst_row`，build22 才进一步停止 hot path 对 `route_dst_rank/local_offset/expert_id`
的物化。pool 中部分 backing storage 仍为 V3/多机 ABI 保留，不能把“ABI 还在”误认为
fusion/pull hot path 仍在使用它。

第 7 节的 destination-pull 在此基础上又把路由控制面收紧为
`(expert, source, slot) -> route` 紧凑表，并增加 source-local X/gate staging。
这两个动作在最终代码的同一个 `static_v4::route_exchange()` 中，但在性能
演进上属于前后两个里程碑：第 6 节数字是 fused source-push，第 7 节数字才是 pull。

route exchange 与 GEMM 并没有重叠：所有 rank 完成上述 publication/barrier
后才冻结 M-tile geometry 并开始 FC1。融合的价值是取消 kernel/host/event
边界和无用物化，不是把 route exchange 藏在 GEMM 后面。

早期融合原型明显回退：graph p50 从 106.189 us 上升到 137.664 us。原因不是
“融合必然更慢”，而是当时基本只把旧 notify materialization/event 协议
搬到了 main kernel 里。后续 source-push 阶段先减少 count fanout 和 per-route
prefix 重算，再以 CTA-local histogram + 每 `(CTA, expert)` 一次 global reservation
取代逐 route 全局计数。math/control 的真正并行发生在 destination-pull build24，
不应倒算成 build5→build12 的 fusion 收益。

### 6.1 build 标记与证据口径

下表中的 build 名称就是当时 `/tmp/weft-fused-route-build*.json`、JIT cache 和 benchmark
log 的实验编号，不是 Git commit。当时所有版本都在同一 dirty worktree 上迭代，没有为每个
build 提交源码 snapshot；下面的“代码改动”是根据完整会话中的 `apply_patch` 与紧随其后的
build 命令逐项对回来的，耗时则直接取对应 log。一个 build 前有多个 patch 或同时换了 tile
时会显式标为“bundle/混杂”，不会把它伪装成单项 A/B。

最初的 bring-up build 是：

| build | 该 build 前实际加入的代码 | 结果/判定 |
|---|---|---|
| `fused-route-build`（无数字） | 在 `ForwardKernelArgs` 和持久化 Forward pool 中加入 fused route metadata；`DeviceDispatchHandle::enqueue_notify()` 对 fused inference 变成 no-op；唯一的 unified kernel 在开头调用 `static_v4::route_exchange()`，之后仍走 source-push Dispatch/FC1/FC2/Reduce | notify kernel 与 main kernel 从这里开始真正合成 1 个 kernel；只是 bring-up，没有保留可信的稳定 p50 |
| `build2` | 尝试由 CTA0“一 lane 一个 expert”顺序扫描全部 routes，想同时去掉 histogram atomic 并获得稳定 slot | 复杂度变成 `experts × routes`，很快放弃；没有进入性能结论 |
| `build3` | 撤回 build2，恢复所有 CTA 并行逐 route 计数/slot reservation；generation 初始化统一清零 local counts | 恢复可扩展的 fused source-push 基线 |
| `build4` | 修正 fused `DeviceDispatchRuntime` 的 generation reset/错误位初始化，并补齐 graph 输出校验 | 第一个稳定 graph checkpoint：**137.664 us，13 nodes / 1 kernel** |

从第一个有完整 device timeline 的 build5 开始，逐 build 变化如下：

| build | 相对前一 build 的实际代码变化 | route / forward p50 | graph p50 / nodes | 判定 |
|---|---|---:|---:|---|
| `build5` | 只加入 `route_exchange_start/complete_clock` 及 runner 导出；算法仍是 build4 的逐 route global atomic + source-push | 37.440 / **110.944 us** | — | measurement anchor，不把它算成优化 |
| `build6` | 删除 source 把 count 再复制进 destination `source_expert_count` 的独立 fanout；rank release 后 owner 直接读各 source 的 symmetric local count；合并一次 post-count grid join，并把 geometry/route-row 生成集中到 CTA0 | 23.168 / **97.568 us** | — | 相对 build5 **-13.376 us**；这才是第一次大幅收缩 route control plane |
| `build7` | 新增一次性 `expert_total/source_prefix/expert_row_base/expert_mtile_base`；先按 expert 计算，再让每条 route O(1) 查表，去掉每 route 重扫 prior experts 和 prior sources | 20.128 / **92.832 us** | 120.352 us / 13 | 相对 build6 **-4.736 us** |
| `build8` | kernel 算法未作对应的大改；host/capture 路径把 cancel H2D 改为仅状态变化时执行，并把 status D2H 移出 capture/完成路径，同时加入 graph node 审计 | 15.264 / **89.792 us** | **93.730 us / 11** | graph 相对 build7 **-26.622 us、少 2 nodes**；device 的 -3.040 us 没有单项 kernel A/B，视为观测值而非因果收益 |
| `build9` | fused Static-V4 launch 不再请求 cooperative-launch attribute，继续使用 kernel 自己的 software grid/rank barrier；补齐 graph node 分类 | 16.352 / **90.112 us** | 93.568 us / 11 | 基本中性，保留简化后的 launch 约束 |
| `build10` | route histogram 改为 CTA-local `uint32[experts]`：第一遍 route 做 shared atomic，每 `(CTA, expert)` 只做一次 global 64-bit reservation，第二遍 route 分配具体 slot；但同一 build 把 FC1 N64 改成 N128 | 16.480 / **95.072 us** | — | **混杂实验**；histogram 优化保留，N128 因 FC1 变慢被撤回，不能用 95.072 us 衡量 histogram 收益 |
| `build11` | 恢复 N64/K128；保留 build10 histogram；同一 execution/stream 上的 routes、gates、activation、weights 不再插 `cudaStreamWaitEvent`，Static-V4 也不再记录 start/output-ready event | 无独立 device sample | **88.288 us / 6** | 相对 graph9 **-5.280 us、少 5 nodes**；这是 kernel + graph-control bundle |
| `build12` | Static-V4 再删除 `forward_complete` event record 和 dispatch consumer submit，完成/时间线复用 workspace `slot.complete` | 16.992 / **75.456 us** | **85.317 us / 4** | graph 相对 build11 **-2.971 us、少 2 nodes**；device 相对最后一个同 tile build9 累计 **-14.656 us**，中间没有 build11 device sample，不能只归因给 build12 |
| `build13` | 消融：route exchange 只让后 128 个 control threads 执行，前 256 个 math threads 空等 | `0.096*` / **76.992 us** | — | 相对 build12 **+1.536 us，拒绝**；`0.096*` 是 clock writer 不再包住实际 control-thread 工作后的失效 marker，不能当 route 耗时 |
| `mega-align` | 在 build13 control 上强制 MegaMOE-like N128/K64/5-stage | — / **81.312 us** | — | 相对同 control 的 76.992 us **+4.320 us，拒绝** |

因此 build5→build12 的可复现实测是 forward 110.944→75.456 us，累计改善
**35.488 us / 32.0%**；route marker 是 37.440→16.992 us，改善
**20.448 us / 54.6%**。但只有 build5→6、build6→7 能较干净地归到具体 route
算法；build8→12 混有 launch、event、同 stream dependency 和 CTA histogram，不能继续按
某一行代码拆账。

以 graph 口径看，融合前的 106.189 us 最终降到 85.317 us，里程碑收益是
20.872 us / 19.7%。这是整个 fusion bundle 的结果，没有独立的“只删 CPU wait”对照组。

## 7. source-push 到 destination-pull

完成 notify fusion 后，source-push 仍需要 source 构造 remote canonical row，并向 remote readiness
counter 发布 system-scope 操作。最终单机路径改为：

1. source 把 X 一次 staging 到 source-local symmetric storage；
2. source 只发布 route table 和 count；
3. destination control warp 以 `global_warp + round * global_warps` 静态枚举 row；
4. source rank 用确定性 round-robin 顺序选择，分散相邻 NVLink pull；
5. payload 落到本 GPU 后只做 device-scope row-ready publication。

这里不只是把一条 memcpy 的方向反过来。两种协议的实际差异是：

| 环节 | fused source-push | destination-pull |
|---|---|---|
| route 发布 | source 需要已知的 destination row | source 只写 32-bit route id + count |
| canonical row owner | 多个 source GPU 向 destination GPU 写 | destination GPU 独占写自己的 row |
| X 地址 | source 直接从调用方 X 为每条 route push | source 先把 X 本地 staging 一次，destination 再按 route pull |
| row 排列 | notify/source 预先物化最终 row | destination 从 count 按 source-rank round-robin 在线反解 |
| payload fence | remote payload 后 `__threadfence_system()` | payload 落在本 GPU，只需 `__threadfence()` |
| FC1 readiness | source 向 remote counter 发 system-scope release | destination 向 local counter 发 device-scope release |
| task 分配 | source route batch/remote row publication | `global_warp + round * global_warps` 静态 row stripe |

destination-pull 中，每个 control warp 独占一个 shared-memory staging buffer
和 `cuda::barrier`。warp 的 32 条 lane 先从该 expert 的 8 个 source count 中用
warp reduce/ballot 反解出 `(source, source_slot)`；lane 0 随后：

1. 从 `(expert, source, source_slot)` route table 取回扁平 route id；
2. 由 `route / top_k` 得到 source token，从 source symmetric X/gate 发起 TMA pull；
3. 把 X、gate 和 `{route, source}` header 写入 destination-local canonical row；
4. TMA store drain 后对所属 M-tile 的 local arrival counter 做 release-add。

source-rank round-robin 是“逐层剥离”的：先取所有 count 非零 source 的
slot 0，再取仍非零 source 的 slot 1，以此类推。因此 count 不均匀时
也不需 padding，路由给定时 row 次序唯一，相邻 warp 又不会长时集中拉取
同一个 NVLink peer。

这次改动只替换 Dispatch 方向和 readiness 协议。融合 source-push 版已经
使用 FC2 epilogue 直接写 source contribution；pull 版保留这一点，因此
75.456→69.312 us 的对比没有再重复计入“取消 Return”的收益。

### 7.1 destination-pull 的逐 build 修改

第一个 correctness-first pull 版本是 108.448 us，比当时 source-push 还慢。下面按真实
build 标出从“能跑”到最终 69.312 us 之间具体改了什么；被拒绝或没有完成测量的 build 也保留，
避免编号跳跃造成“中间还有一个未知大优化”的错觉。

| build | 相对上一 build 的实际代码变化 | route / forward p50 | graph p50 / nodes | 判定 |
|---|---|---:|---:|---|
| `build14` | **push→pull 的结构切换点**：新增 destination-owned `(local_expert, source, source_slot) -> route` table；source 在同一 kernel 开头把 X/gate staging 到 source-local symmetric pool；destination 的 control warp 用 global-warp 静态 row stripe 拉 X、写 canonical row/header/gate，再发布 M-tile readiness。初版仍按 source-contiguous prefix 反解 source，每 warp 有私有 staging buffer/barrier | 44.096 / **108.448 us** | — | correctness-first；相对最终 source-push build12 慢 32.992 us |
| `build15` | 把 CTA0 中串行的 `(source, local_expert)` peer-count 读取分给整个 CTA；count 汇总和最终 row/M-tile prefix 此时仍由 thread0 完成（逐 local-expert 汇总到 build20 才并行） | 21.280 / **79.552 us** | — | 相对 build14 **-28.896 us**；其中 route span **-22.816 us**，是 pull bring-up 最大的一步 |
| `build16` | canonical row 已落在 destination 本 GPU，因此删掉旧 push 路径的 system-scope publish：TMA store 后用 `finish_copies<false>() + __threadfence()`；arrival 改 device-scope release；FC1 readiness 从轮询 8 个 system-scope rail 改为只读本 rank 的 device-scope rail | 22.176 / **75.008 us** | — | 相对 build15 **-4.544 us**；收益主要在 Dispatch/FC1，不在 route exchange |
| `build17` | 消融：像前面的 build13 一样，只让 128 control threads 做 route exchange，256 math threads在边界等待 | `0.096*` / **75.808 us** | — | 相对 build16 **+0.800 us，拒绝**；`0.096*` 同样是失效 marker，实际 route 工作被计进了后续 Dispatch span |
| `build18` | 先撤回 build17，恢复全 CTA route exchange；然后把 source-contiguous row 顺序改为 source-rank round-robin“逐层剥离”，32 lanes 用 reduce/ballot/FNS 共同反解 `(source, source_slot)`，lane0 只负责 TMA issuer | 21.024 / **73.600 us** | — | 相对可比的 build16 **-1.408 us**；row-selection hot path 不再消费 source-prefix table，其 allocation/args 到 build20 才删除 |
| `tune19` | **不是代码 build**。在 build18 上扫 FC1 N64/N128、K64/K128 和 stage。初筛均值 N64/K128/3-stage 为 73.776 us，最佳 K64 候选为 78.122 us；重复确认受 barrier 长尾影响变成 76.728 vs 76.954 us，autotuner 因 0.5% 保守阈值错误保留 K64 incumbent | — | — | 不接受自动选择，继续使用 `grid_0088 = M128 / FC1-N64 / FC2-N128 / K128 / 3-stage` |
| `build20` | 删除 round-robin 后已无消费者的 `source_prefix/expert_row_base/expert_mtile_base` pool/args；每 local expert 的 total count 改为 CTA 并行汇总，thread0 只扫 8 个 local experts 生成 prefix/M-tile geometry | 17.920 / **70.432 us** | 81.770 us / 4 | 相对 build18 **-3.168 us**，route span **-3.104 us** |
| `build21` | 识别 external producer 其实就在 execution stream 上时，routes/gates/activation/weights 的 handoff 不再插 event wait；kernel hot path 与 build20 相同 | 无独立 device sample | **81.616 us / 3** | 相对 graph20 **-0.154 us、少 1 wait node**；属于 graph control，不是 pull kernel 算法收益 |
| `build22` | route hot path 不再物化 `expert_id/local_offset/dst_rank`；第二遍直接重读 `selected_experts`。source 先在本地 `send_counts[expert]` reservation，再由 CTA0 把最终 count 直接 push 到 destination 的 source-major table；rank barrier 不再重复进入一次 grid barrier | 18.048 / **70.400 us** | — | 相对 build20 **-0.032 us，性能中性**；保留是因为去掉无用 per-route metadata 和 peer count gather，协议更接近最终形态 |
| `build23` | 初次尝试把 route 前缀分工为 256 math threads 负责 X staging/combine-order、128 control threads 负责 histogram/route table | — | — | 编译过程中发现缺少两个 role 的显式 CTA join，artifact 被主动终止并由 build24 取代；没有性能数据 |
| `build24` | 在 build23 分工上补齐 role join：route table 与 X staging 都完成后 control 才发布 remote counts/rank-ready，末尾再把全 CTA 合流后进入 Dispatch/FC1。注意这是**同一个 homogeneous CTA 内的 warp-role 分工**，不是特殊 transport CTA | 16.064 / **69.312 us** | **80.011 us / 3** | 相对 build22 **-1.088 us**；相对上一 accepted 性能表节点 build20 为 **-1.120 us** |
| `build25` | serialized graph replay 下不再每次记录 tensor/workspace completion event，release 时一次性补记；kernel 与 build24 相同 | 无独立 device sample | **77.131 us / 1** | 相对 graph24 **-2.880 us、少 2 event-record nodes**；最终 graph 只剩 1 kernel node |

因此不能把 build14→24 简写成一个含混的“per-row 控制优化”：其中可明确归因的最大项是
build15 的 count/geometry 并行化；build16 是 remote system-scope readiness→local
device-scope readiness；build18 是 row 的 source round-robin 反解；build20 是 prefix/base
元数据删除和 geometry 并行；build22 是 route destination materialization 删除；build24 才是
X/combine 与 histogram/table 的 CTA 内 warp-role 并行。

从阶段 timeline 看，原型到 build24 的变化是：

| device-clock stage | pull build14 | pull build24 | 变化 |
|---|---:|---:|---:|
| route exchange | 44.096 us | 16.064 us | -28.032 us |
| Dispatch/pull | 18.880 us | 14.880 us | -4.000 us |
| FC1 | 19.072 us | 17.376 us | -1.696 us |
| FC2 | 16.960 us | 13.632 us | -3.328 us |
| Reduce | 12.672 us | 8.480 us | -4.192 us |
| Forward | 108.448 us | 69.312 us | -39.136 us |

route-exchange 的 28.032-us 改善可以按相邻 build 进一步约束：build14→15 的
count/geometry 并行贡献观测值 22.816 us，build18→20 的废弃 prefix/base 删除与
geometry 并行贡献 3.104 us，build22→24 的 math/control role 并行贡献 1.984 us；
余下差异包含噪声和 build16/18/22 的协议变化。CTA-local histogram、warp-private
copy buffer/barrier 和 global-warp row stripe 在 build14 已经存在，**不能**再列为
build14→24 的收益。Dispatch/FC1 的主要结构变化是 build16 的 destination-local
readiness，source 反解的变化则是 build18 的 warp round-robin。

从原型到 build24 共改善 39.136 us / 36.1%。更有意义的比较是最终 fused source-push
75.456 us 到最终 pull 69.312 us：**-6.144 us / -8.1%**。这是整个 transport
specialization 的收益，不能继续拆成“打包”、“发送”或“任务领取”的独立数字。

跨机仍保留 source push + GIN + local relay 方向。上述 6.144-us 收益只能推广到单机
NVLink 拓扑，不能作为跨机 pull 的依据。

## 8. CUDA Graph 和 runtime 控制面

融合 kernel 本身已经只有一个 kernel node，但早期 graph 仍有多个 event record/wait node。
这些 node 在 70-us 量级的 kernel 上不能忽略。

| graph 版本 | total nodes | kernel | event record | event wait | graph p50 | delta |
|---|---:|---:|---:|---:|---:|---:|
| fused graph9 | 11 | 1 | 6 | 4 | 93.568 us | — |
| fused graph11 | 6 | 1 | 4 | 1 | 88.288 us | -5.280 us |
| fused graph12 | 4 | 1 | 2 | 1 | 85.317 us | -2.971 us |
| pull graph21 | 3 | 1 | 2 | 0 | 81.616 us | -0.154 us vs graph20 |
| pull graph24 | 3 | 1 | 2 | 0 | 80.011 us | — |
| pull graph25 | **1** | **1** | **0** | **0** | **77.131 us** | **-2.880 us** |

graph9→graph12 之间同时有 kernel 内改动，所以 -5.280/-2.971 us 是控制面 bundle；
graph24→graph25 是最清楚的相邻节点清理，收益 2.880 us / 3.6%。

PyTorch runtime 同时做了两个结构优化：

- graph plan 直接绑定稳定的 X/routes/gates 地址，不再回放前 `copy_` 到内部 bucket；
- 直接返回固定 output，不做 replay 后 clone。

这两项没有保留“相同新 kernel + 旧 copy runtime”的完整 A/B，因此不单独宣称收益。
最终审计结果为每 rank 1 graph node、1 kernel node、0 memcpy node、0 event node。

在同一 tensor runtime 口径上，build26 到 build31 从 78.932 降到 74.237 us/层，
改善 **4.695 us / 5.95%**。这一段主要由第 3 节的 release atomic、cancellation poll
和 busy-poll 优化组成。当前 child graph 相对裸 graph 仍有约 1.7 us/层的调度差。

## 9. 未保留的实验

| 实验 | 观测 | 决策 |
|---|---|---|
| 删除 PDL wait/launch attribute | 76.925→77.632 us，回退 0.708 us | 回退，保留 wait |
| 强制 MegaMOE N128/K64/5-stage | 76.992→81.312 us，回退 4.320 us | 不保留，tile 继续 autotune |
| 早期 fused-route 原型 | graph 106.189→137.664 us | 不回退方向，重写 route/event 协议 |
| correctness-first destination-pull | source-push 75.456 us，初版 pull 108.448 us | 不直接启用，继续静态化后降到69.312 us |
| 把 child graph kernel node 手工 flatten 到 parent graph | 两次实验都挂起 | 回退；cluster launch 属性不能只靠普通 `cudaGraphAddKernelNode` 复制 |
| 只复制 kernel-node attributes 后 flatten | 仍挂起 | 回退，保留 child graph |
| 专用 24 transport CTA | 与所需的 all-CTA work-conserving 设计冲突；无干净单项 A/B | 删除，所有 CTA 同构 |

挂起实验没有性能数字，不把 timeout 当作无穷大回退记入收益表。

## 10. 数值与稳定性门禁

最终版在平衡路由下经过数千次 graph replay，8 rank output hash 固定为：

```text
9845168797502030967
6447824382240215722
3497438304625584484
7015812181918136931
3692855610873768654
1469494279972495450
6269424337513434764
11050250162159447178
```

额外通过：

- 64 token/rank 平衡路由的 8-rank tensor parity；
- 64 token/rank 全部集中到 rank 0 的极端 skew，包括七个 zero-incoming rank；
- 双层 CUDA Graph replay；
- graph destroy + runtime offload/recreate；
- 输出与 eager PyTorch 参考一致，没有 capacity drop 或 row 覆盖。

性能优化没有改变 FC1/FC2 的 K 累加顺序、FP32 accumulator、BF16 rounding 点、
SwiGLU/gate 精度和 top-k FP32 combine order。N/K tile 由 autotune 改变时，每个输出元素
内的 K 遍历与精度合同仍保持不变。

## 11. 如何解读这些收益

1. 第 3 节的四项可用于单项回归判断，但即使它们也不应把不同 A/B session 的
   delta 精确相加。
2. 同构 CTA、notify fusion 和 destination-pull 是重构 bundle，只能比较里程碑。
3. route/dispatch/FC1/FC2/reduce span 互相重叠，不能加总得到 forward 耗时。
4. 当前还有约 4.6 us 的 WeftMoE kernel-only 与 MegaMOE kernel-only 差距，但 WeftMoE 用
   kernel 内 X staging 和直接 route/gate 绑定省掉了 MegaMOE 的 11.5-us 图外 copy。应用结论
   必须看 74.237 vs 74.1 us，而不是 62.x vs 57.9 us。
5. 本文数字只适用于列出的 EP8 decode shape。prefill、不同 token bucket、training 与多机
   需要独立 autotune 和回归，不应直接套用 72.500-us 结论。
