# PF pull warp、channel 容量与 WG3 资源审计（2026-10-09）

后续：[每 CTA 实际借用一个 pull warp 的双机验证](pf_borrow_warp_20261010.md)。

单机 3→2 pull warp：8K 无变化，20K 快约 1.4%，本轮未见性能回退。
全 CTA 从16降到4 slot，每卡省2.32GiB，原2.53GiB增量只剩约216MiB，
8K/20K带宽保持，不需要减少网络 owner。
WG3 有真实的 Get/Put 控制路径 spill；GEMM 热 K 循环没有 accumulator spill。
shared memory 恰好达到每 CTA 上限，不是还有充裕空间。

生产默认配置未修改。实验分别保存于 `experiment/pf-local-pull2-20261009`
和 `experiment/pf-wg3-slots-20261009`，不是把纯通信诊断当成完整 MoE 性能。

## 1. 单机：只减少一个 pull warp

单节点 8 GPU、EP8、SM1620MHz，BF16 H4096/I2048/E256/topK6；
tokens 在 8192/20480 附近正态波动 10%，每步新路由，bias0.5、seed123。
使用 `pf_inference_flash_20261008` 配置，B2048、真实 FC1/FC2/SwiGLU、
prepared weights，不保存训练中间结果。全部编译完成后再测。

只把每 CTA 的 pull 工作者从 3 改为 2，并同步修改静态 residue/stride。
384 线程、208/80 寄存器配额、GEMM 实现、三个 4 KiB pull staging buffer 均保持
原样；空出的 warp 不做 pull，但仍参加收尾同步。没有同时改变 shared memory
或 occupancy，故这是 pull 并发度的单变量实验。

prime32/warm8/measure24；顺序 3、2、2、3。每步取最慢 rank，再取 p50，
表中为两轮独立 p50 的均值。四轮完整 32 步的 token 数、route/gate hash 相同。

| pull warp/CTA | 8K kernel ms | 20K kernel ms | 8K eager ms | 20K eager ms |
|---|---:|---:|---:|---:|
| 3 | 6.637 | 16.960 | 6.880 | 17.199 |
| 2 | 6.636 | 16.725 | 6.814 | 16.956 |
| kernel 耗时变化 | −0.01% | −1.39% | — | — |

kernel 只计原生 fused kernel；eager 包括输入 staging、enqueue、同步，不包括
一次性权重准备、工作区分配及 IPC 建立。3-warp 两轮 kernel 为
6.663/6.610、16.952/16.967 ms；2-warp 为 6.663/6.609、16.706/16.744 ms。
8K 差异小于复测漂移，不声称有加速；20K 两轮均未回退。
SASS 确认工作者边界及 stride 确实从3变为2；编译器也有少量指令排程变化，
不把20K的微小收益完全归因于少一个 pull warp。
132 CTA 的总 pull 并发从396降到264，结果表明本负载下264个已足够支撑
当前 GEMM 流水，不代表所有 hidden/intermediate、路由形态都可以无损减少。

两版正常路由各 8K/20K×2 步，2-warp 另测 B128、bias8 的极不均衡路由。
共 96 个 rank/sample 检查，最大 relative-L2 9.36e-5、max-abs 6.11e-5；
正常路由下两版输出逐位相同。114 次节点时钟观测均为 1620MHz。
两版 ptxas 均为 REG168、stack0、spill stores/loads0；其调用中的泛型 atomic
local-address fallback 不等于寄存器 spill。

结论：本 shape 下可以把 **2 pull + 1 RDMA warp** 作为下一版 384-thread
多机候选，避免先增加完整 WG3；但这里空出的 warp 尚未执行 RDMA。
多机 pull 还涉及 RX ready、fanout 消费计数、credit 等，不能从单机结果直接
保证多机无回退。另外，132 个新增 warp 不等于当前 WG3 的
132 Get + 132 Put + 132 progress warp；角色仍需分配或设计非阻塞合并调度。

## 2. Channel 与 slot 容量

沿用[全 CTA 通信实验](pf_wg3_communication_20261009.md)：132 lanes、
132/132/132 Get/Put/progress、512 线程、192/64/64 配额，D32/R128、
B4096、pending4、credit2、planner publication、Return-ready 去重。
本节是以真实拷贝替代 GEMM 的双机 EP16 诊断，不是完整 MoE。

旧实现继承 `minimum_depth=8`，再乘 credit windows2，得到每 channel 16 slot。
这不是 132 lanes 所必需的容量。当前协议需要先预留一个完整最坏 Return
window，容量下限仍需保留：

```text
window_packets = ceil(ceil(B × topK / R) / lanes)
               = ceil(192 / 132) = 2
depth = max(minimum_depth, window_packets) × credit_windows
      = max(2, 2) × 2 = 4
```

该下限使用所有路由都发往同一远端的最坏情况，不依赖平均路由均衡。
本轮仅增加实验性的 host floor 参数 `WEFTMOE_PF_MIN_RING_DEPTH`，
未改变 kernel、credit、NIC completion 或 generation 的语义。缩 slot 后仍然
同时等待远端 credit 和本地 NIC completion，不能覆盖未消费的数据。
随 B、topK、R、lane 数变化必须重新计算下限，不是所有配置都硬编码 4。

每个 slot 的 payload 是两个 D32 dispatch 区（TX/RX）加一个 R128 Return
区，共 1.5MiB，再加少量控制字段。因此 132×(16−4) 个 slot 可以省约
2.3205GiB/卡。与原来的16 lanes×24 slots相比，新布局是132×4=528 slots，
而非2112 slots；增量约216MiB，保留全部网络 owner。

实际工作区容量（compute+GIN，GiB/卡）：

| 布局 | 8K | 20K |
|---|---:|---:|
| 原16 lanes×24 slots，历史对照 | 5.968 | 7.141 |
| 全CTA，132 lanes×16 slots | 8.500 | 9.672 |
| 全CTA，132 lanes×4 slots | **6.179** | **7.352** |

每卡确实节省2.3205GiB，保留的增量约216MiB。另有不变的 notify
16.4/40.8MiB。表不含输入、权重、PyTorch allocator、NCCL 内部 context
资源；16/4-slot 两组都使用相同的396 contexts，没有隐藏减少 owner。

设备代码与上一轮 all-CTA 诊断的 normalized SASS hash 完全一致：
`cc714b9a801cd590fabe70277784a3bccb4ad41bd1ed75b781f7569b4a52e1f3`。

正式测试顺序16、4、4、16；与单机相同的随机 token/route 方法，
prime32/warm8/measure24。下表为两轮独立p50的均值，带宽按每步实际 payload
除以该步最慢rank通信耗时计算，再取p50；不是两个p50相除。
报告双向同时传输时的**每卡、单方向有效payload GB/s**，标称上限50。

| slots/channel | 8K GB/s | 20K GB/s | 8K完整诊断kernel ms | 20K完整诊断kernel ms |
|---|---:|---:|---:|---:|
| 16 | 34.23 | 38.14 | 9.071 | 19.601 |
| 4 | **34.31** | **38.38** | **9.023** | **19.440** |

通信阶段为7.807/17.649→7.773/17.498ms；完整kernel包括prepare和收尾，
不含一次性分配、注册、396个GIN contexts的初始化/释放。
4-slot 两轮带宽分别34.309/34.313和38.320/38.433GB/s；16-slot为
34.313/34.146和37.891/38.396。差异很小，结论是**显著省显存、性能保持**，
不将它包装成明显的通信加速。当前没有必要继续增加8-slot候选。

四轮全部16 rank的完整32步 token/route/gate workload均与历史all-CTA对照
完全一致；全部逐样本Get/Return字节核对通过。764次正式节点时钟观测均1620MHz。
4-slot另做正常路由8K/20K各2次、B128+bias8各3次，共160次rank/sample
校验，max-abs全部0；额外378次验证时钟观测也全部1620MHz。
六轮两机进程均exit0，测试结束后两机全部GPU显存回到0MiB，无遗留测试进程。

## 3. 寄存器：配额放得下不代表没有 spill

使用 CUDA12.9，重新编译 `-lineinfo -Xptxas=-v,--warn-on-spills`，并检查
带内联源码定位的 SASS。当前纯通信 all-CTA binary 和恢复真实 GEMM 的
all-CTA **编译审计 binary** 分开记录，后者没有用于本轮性能/数值验证。

下面是 ptxas 的静态 spill store/load 字节统计，**不是每 token/packet 实际
传输字节，也不能相加作为运行流量或显存预留量**。helper 的 stack frame0
不代表没有 spill，它们共享入口调用栈；入口累计 stack 分别为424/456B。

| 路径 | 纯通信 store/load B | 含 GEMM 编译版 store/load B |
|---|---:|---:|
| kernel 入口/内联路径 | 68 / 48 | 144 / 140 |
| Dispatch Get | 420 / 844 | 560 / 1016 |
| Return Put | 308 / 600 | 476 / 804 |
| progress | 4 / 8 | 0 / 0 |
| prepare | 0 / 0 | 4 / 4 |
| admit、pack、local pull | 0 / 0 | 0 / 0 |

GIN headers 同时编译多种 backend；大量静态 LDL/STL 还属于
`cuda_local.h` 的 generic-pointer local-address fallback。当前 ring/GIN
计数器在 GM，不能把这些分支全部视作运行时 spill 流量。因此，上一轮
1398/729 条 LDL/STL 只是一份静态计数，不能用来量化性能损失。
Get/Put 中确实也存在以 R1 为基址的栈存取，不全是上述 fallback。
当前测试使用 token Get/Return Put，pack/dispatch-push 分支不执行；表中仍然
列出这些已编译的函数，不能解读为全部路径都在当前运行中执行。

含 GEMM 的 all-CTA 编译版中，调度器 `Schedule::index` 在任务边界有
LDL/STL，不能声称整个 math 路径零 spill。但热 K 循环及其 full-barrier
慢等待分支没有 LDL/STL：8 条 N128 HGMMA 后一次 wait_group0，
没有 accumulator spill，也没有每条 WGMMA 后插 wait 的串行化。
详细地址区间、artifact hash 和 ptxas 结果保存在资源 JSON。

寄存器预算为 `256×192 + 128×64 + 128×64 = 65536`，刚好用完。
尤其 RDMA/control 只有64配额，不是可以任意分配更多活跃状态；后续应优先
压缩 Get/Put 的状态生命周期、减少跨 GIN 调用存活的临时量，再单独测收益。
本轮没有用静态 spill 统计宣称它已解释全部带宽差距。

## 4. Shared memory：能放下，但已到上限

| 每 CTA 分配 | 字节 |
|---|---:|
| local::Shared：GEMM stages、epilogue、3 个 pull staging、barrier | 226304 |
| WG3 ExtraShared：额外 pull staging、barrier、Get/Put pending | 5120 |
| dynamic shared | 231424 |
| static shared | 1024 |
| 合计 / CUDA 查询的 CTA opt-in 上限 | **232448 / 232448** |

也就是227KiB/CTA；每 SM 233472B 的上限更大，但剩下的1KiB不突破每 CTA
上限，不能继续给该 CTA 使用。当前仍是1 CTA/SM、2 CTA/cluster。

全 CTA Get 模式已不再借 WG3 warp0 做 extra pull，因此 ExtraShared 的
4KiB pull staging 及其 barrier 可以在该特化中去掉；pending requests 仍需
保留。预计可回收4KiB，但本轮没有同时改它，避免干扰 slot 消融。
这是 shared memory；第2节缩 slot 节省的是 GM，两者不同。
即使回收这4KiB，也不足以直接增加一个完整 A/B GEMM stage（当前需48KiB），
因此不能在这个布局上简单把 stages3 调成4。

## 复现与产物

- 单机代码 `30c6eda`，单机 launcher `b2f6743`；实验目录
  `../WeftMoE-experiments/local-pull2-20261009/build/pull_warps/`。
- 缩 slot/含 GEMM 编译审计代码 `85cfc54`；实验目录
  `../WeftMoE-experiments/wg3-slots-20261009/build/slots/`。
- [单机测量、正确性、launch/hash/clock](../benchmarks/pf_pull_warps_20261009.json)。
- [双机缩slot测量、正确性、launch/hash/clock](../benchmarks/pf_wg3_slots_20261009.json)。
- [ptxas、SASS 与热 K 循环资源记录](../benchmarks/pf_wg3_resources_20261009.json)。
- [源码定位的 local-memory 审计脚本](../benchmarks/audit_pf_local_memory.py)，
  与 `pf_wg3_sass.py` 配合用于编译产物诊断。

编译均开启 generated tuning、`-lineinfo -Xptxas=-v,--warn-on-spills`。
单机配置目录 `configs/pf_inference_flash_20261008`，仅切换
`WEFTMOE_PF_LOCAL_PULL_WARPS=3/2`。双机配置目录
`configs/pf_network_l132_20261009`，`WEFTMOE_PF_NETWORK_WORKERS=2`、
`WEFTMOE_PF_WG3_ALL_CTA=1`，math/local/network registers分别192/64/64；
`WEFTMOE_PF_COMMUNICATION_COPY=1`用于测量，0仅用于含GEMM编译审计。
运行时 `WEFTMOE_PF_MIN_RING_DEPTH=8/2`，
`WEFTMOE_PF_CREDIT_WINDOWS=2`、`WEFTMOE_PF_WG3_PUBLICATION_PLAN=1`、
`WEFTMOE_PF_DEDUPLICATE_RETURN_READY=1`。完整launcher命令保存在各轮manifest。

每个正式运行的 launch manifest 保存命令参数、library/extension hash 和
source commit。单机首次计时启动曾误传 benchmark 不支持的 `--priming-steps`，
在参数解析阶段退出，没有执行 kernel；移除后使用 benchmark 已有的整轮 prime，
失败启动不计入任何结果。
