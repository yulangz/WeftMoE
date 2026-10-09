# WG3 全 CTA 通信消融：2026-10-09

结论：**原 WG3 与部分专用 warp 的通信带宽接近；启用所有 CTA 的网络 owner
有稳定收益，但仍未满带宽。** 在不增加线程数、寄存器配额的条件下，全 CTA
版将 8K/20K 带宽提高 **24.1% / 23.3%**，完整诊断 kernel 耗时降低
**15.7% / 16.9%**。这不是包含 GEMM 的 MoE 性能，尚未改变生产默认实现。

## 测量口径

沿用[上一轮去 GEMM 诊断](pf_communication_bandwidth_20261009.md)：
两机各8张 L20X，EP16，SM1620MHz，BF16 H4096/I2048/E256/topK6。
tokens 在8192/20480附近正态波动10%，每次新路由，bias0.5、seed123。
本轮所有16 rank的完整32步 token数量、route/gate hash均与上一轮部分专用
warp 测试完全一致。
prime32/warm8/measure24；三组各两次，顺序为 base、lanes、all、all、lanes、base。
所有 CPU 编译结束后才开始正式计时；六轮均 exit0，550次节点时钟观测全部1620MHz。

FC1/FC2替换成可校验的 X→S、S→真实 contribution/TX 拷贝。保留源节点去重、
Get、NVLink fanout、静态 M/N 任务、X/S/TX 有限 ring、admission、generation、
本地 NIC completion、远端 credit、publication、最终 reduce/rendezvous。
权重加载、SwiGLU、GEMM staging、WGMMA均不执行。没有空操作推进完成计数。

表中是两轮独立p50的均值。每步按最慢rank通信时间和该步实际字节数计算带宽，
再取p50，不是用两个p50相除。Get在接收端、Return Put在发送端计一次，
每步全局字节数均与路由推导的精确节点去重token数/远端route数一致。
报告同时双向传输时的**每卡、单方向、有效payload GB/s**，标称上限50 GB/s。
通信阶段包含机内转发及拷贝替身，不含prepare和最后reduce/rendezvous。
初始化、分配、注册、权重准备均在计时外；396-context版本的初始化/释放更慢。

## 实现与公平对照

所有组均为512线程、192/64/64寄存器配额、D32/R128、B4096、pending4、
credit2、planner publication、连续同slot/generation的Return-ready去重。

| 组 | channel lanes | Get / Put / progress owner | local pull warp | GIN contexts |
|---|---:|---:|---:|---:|
| 原优化 WG3（base） | 16 | 16 / 8 / 4 | 512 | 28 |
| 只增加channel（lanes） | 132 | 16 / 8 / 4 | 396 | 28 |
| 全CTA（all） | 132 | 132 / 132 / 132 | 396 | 396 |

all中每个CTA的WG3 warp0负责Get，warp1负责Put，warp2负责publication及
RX/credit progress；CTA0的warp3仍是唯一admission owner，其余warp3在Get模式
下空闲。不是让多个warp竞争同一计数器：channel按静态residue分配，每条channel
的各角色cursor和每个context仍只有一个owner。有限packet数量可能使部分owner暂时或全程无任务，
并不声称132个owner在每一时刻都发RDMA。

原extra-pull借用CTA16..131的WG3 warp0；all把它们交还Get，所以local pull
回到396。lanes也关闭extra-pull，**lanes和all的ring布局、local pull数相同**，
用于隔离网络owner扩展的效果；base与all不是纯单变量对照。

396 contexts超过132 CTA，metadata Put和其完成barrier改为CTA跨context
步进；每个context仍单owner，未增加数据流水中的全局barrier。另修正旧host
启动检查：WG3所需CTA数按每类owner的最大值计算，而非context总数。
初次all验证在这条旧检查处退出、未发射kernel；修正后的所有验证和计时均通过。

## 结果

| 方案 | 8K通信 ms | 20K通信 ms | 8K GB/s/卡/方向 | 20K GB/s/卡/方向 |
|---|---:|---:|---:|---:|
| 部分专用warp，上一轮token Get | 9.860 | 21.452 | 27.36 | 31.35 |
| 原优化WG3 | 9.671 | 21.829 | 27.56 | 30.88 |
| 132 channel，仍少量owner | 13.842 | 35.563 | 19.11 | 19.08 |
| **132 CTA全部参与网络工作** | **7.802** | **17.658** | **34.20** | **38.08** |

全CTA约为标称带宽的 **68.4% / 76.2%**，折合每节点、每方向
**273.57 / 304.62 GB/s**。与同容量、同local-pull的lanes对照相比，带宽提高
约 **79% / 100%**；不能把改善解释成单纯扩大buffer。

完整诊断kernel包含prepare、拷贝/通信、reduce和复用rendezvous：

| 方案 | 8K完整kernel ms | 20K完整kernel ms |
|---|---:|---:|
| 原优化WG3 | 10.735 | 23.546 |
| 132 channel，仍少量owner | 15.088 | 37.459 |
| **全CTA** | **9.045** | **19.576** |

全CTA两轮带宽分别为34.198/34.195和38.093/38.061 GB/s；
完整kernel为9.044/9.045和19.616/19.535 ms。原WG3两轮为
10.729/10.740和23.508/23.583 ms，收益明显大于复测漂移。
各阶段独立取rank-max/p50，阶段列不能精确相加。

## 寄存器与容量

启用已有WG3中的空闲warp**不增加每CTA的线程/寄存器预算**：
`256×192 + 128×64 + 128×64 = 65536` 个32-bit寄存器。
预算能容纳这组工作者，但并非还有额外未分配寄存器。入口REG128是静态值，
不等于math的动态192配额。SASS确认两个control WG dealloc64、math alloc192。

三版诊断kernel均为REG128、stack424B、static shared1024B，
dynamic shared231424B；实际max-active-clusters均66，即132 CTA。
HGMMA/WGMMA均0，三版静态LDL/STL计数均1398/729，全CTA未增加这些计数。
这是整个控制路径的静态指令统计，不能据此声称没有spill或量化其运行开销。
此前含GEMM的192/64/64 WG3热K循环无LDL/STL，见
[原WG3审计](pf_network_20261009.md)；本轮没有重测全CTA含GEMM版本。

132 lanes触发继承的最小depth8，credit2后每channel有16 slot；原16 lanes
为24 slot。**容量没有随lane数按比例缩小**，所以本次探索的GM代价真实存在：

| compute+GIN工作区，GiB/卡 | 8K | 20K |
|---|---:|---:|
| 原WG3 | 5.968 | 7.141 |
| lanes与all | 8.500 | 9.672 |
| 增量 | 2.532 | 2.532 |

另有相同的notify16.4/40.8 MiB；表不含输入、权重、PyTorch allocator及NCCL
内部context资源。lanes与all的上述工作区逐字节相同，但context数不同。
尚未缩减132-lane版本的最小depth，也未证明同收益可在原容量内保持。

判断：原先只有少量网络owner是实现选择，不是WG3寄存器硬上限。owner扩展
确实能缓解完整通信路径的瓶颈，但剩余差距仍包含准入、ready检查、机内搬运、
拷贝替身和排空等，尚未逐项定量归因。不能从这轮诊断直接推出真实MoE的加速比。

## 验证与复现

三版正常8K/20K各2次；all另测B128、bias8、8K/20K各3次，共288次
rank/sample输出检查，max absolute error全部0。base校验后仅重建host启动
检查，device SASS hash保持不变。正式六轮所有逐样本payload核对通过，
完整workload均与上一轮一致；测试结束后两机显存均成功释放。

诊断分支 `experiment/pf-wg3-communication-20261009`，代码提交 `c685d4d`，
worktree为 `WeftMoE-experiments/wg3-comm-20261009`。该分支的
`docs/wg3_communication_20261009.md`给出构建/启动参数；
原始日志、launch manifests、时钟和SASS在 `build/wg3_comm/`。
它只能运行copy参考输出，**不能用作普通MoE库**。

冻结库SHA256：

- base：`de0e6525815376aa9f97480cb916a7c522c3ccfd446ec02805cc1372a3b14854`
- lanes：`48ab3142c9ee3372b4cc3286952abe59855d01ab5c376d77d6e8f72069ae3d49`
- all：`08c916b970911ac5d10ec93c21275e6cd8852081a002347e7c25192ea9d06bd7`

[冻结结果](../benchmarks/wg3_communication_bandwidth_20261009.json)包含六轮逐样本
时间、有效字节、双方向带宽、workload、manifest及clock。复算继续使用
[`summarize_communication.py`](../benchmarks/summarize_communication.py)。
