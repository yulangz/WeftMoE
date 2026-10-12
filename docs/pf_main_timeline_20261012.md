# PF 多机 main：timeline 拆解与衔接优化

正式保留 **本地窗口优先 + FC2 寄存器直接128-bit写回**。return→reduce 的提前启动完成了实现和消融，但没有证明稳定的整体收益，已撤出默认代码。不改 BWD 数学流程、不扩大通信 buffer、不引入动态领任务；训练与推理仍共用 main。性能结论以关闭 timeline 的独立对照为准。

## 测量口径

双机 EP16，BF16 H4096/I2048/E256/top-6，8K/20K 为每 rank 的名义 tokens。每步 tokens 正态波动 10%，路由重新随机生成，共享 expert bias std=0.5，seed123；是合成变化负载，不是模型实录。B4096/D2、132 lanes、credit windows2、publication plan/return-ready 去重均不变。

正式缓存 FWD 包含逐次 W1 重排、完整 notify、准确缓存申请、main 和同步，结束时 tape 仍存活；固定 pool/IPC 注册已预热。prime32、warm8、sample24；逐步最慢 rank 的 p50，再平均各轮 p50。配对 workload hash 一致。正式性能不启用 timeline/CUPTI、不并行编译。SM 目标 1620 MHz，另以 100 ms 连续采样核对测量区间。

timeline 为 warm2/sample2，下载最后一个样本；每 case 16 rank，均检查 overflow。它与正式计时是不同样本序列，只比较配对的 timeline。`%globaltimer` **只在同一 GPU 内相减**；没有拿两个节点的绝对时间戳相减。

## 逐步消融

首组六轮顺序为旧→local-first→FC2直接写→FC2直接写→local-first→旧；单位 ms，两轮 p50 的平均。这组六轮的测量区间内，16卡已有100 ms采样均为1620 MHz。

| 带缓存完整 FWD | 8K | 20K |
|---|---:|---:|
| 起点：独立 notify + borrowed warp | 12.442 | 25.843 |
| 本地窗口优先 | 11.816 | 25.279 |
| 再加 FC2 寄存器直接写回 | 11.906 | 24.901 |

local-first 降低耗时约5.0%/2.2%。FC2 直接写回在20K上进一步降低约1.5%；8K 的两轮为11.588/12.224 ms，p50波动明显，不能拿第一轮说它必然比 local-first 快。相同首组样本均值则从12.181降到12.020 ms。后续还做独立的最终版 ABBA 复核，原始轮次全部保留。

### 最终复核与 main 独立计时

最终旧→保留→保留→旧复核，8K 完整缓存 FWD 为 **12.482→11.587 ms，−7.17%**；两轮新版11.586/11.588 ms，已有频率观测均为1620 MHz。结合首组4.3%的结果，8K端到端收益有轮间波动，不能只保留最好一组。20K 复核观察为25.841→24.911 ms，但新版第二轮出现短暂1590 MHz，因此严格同频结论沿用首组 **25.843→24.901 ms，−3.65%**，不筛掉该轮的某个样本重算。

以下为另一次 CUDA-only CUPTI 分项，未开启 device timeline；两版测量跨度内的采样均为1620 MHz。单位 ms，每项取逐步最慢 rank 的 p50；分项的关键 rank/样本可能不同，不能直接相加解释 wall。

| tokens | W1重排，旧→新 | notify，旧→新 | main，旧→新 | main 耗时下降 |
|---|---:|---:|---:|---:|
| 8K | 0.268→0.268 | 1.310→1.308 | **10.437→9.635** | **7.69%** |
| 20K | 0.268→0.268 | 1.543→1.535 | **23.461→22.535** | **3.95%** |

收益集中在 main，没有把重排、notify 或缓存申请移出计时范围。

无缓存 inference，权重已准备、完整 GPU notify+main：8K **10.742→9.976 ms，−7.13%**，已有观测均为1620 MHz；20K观察值23.141→22.355 ms，但旧版有一个1605 MHz观测，单独标记，不用于严格同频收益结论。GPU时间不含逐调用训练重排，不能与训练FWD相减当作纯缓存开销。

单机 EP8 缓存 FWD 回归：8K **8.260→8.208 ms**，20K **18.550→18.542 ms**。每档一对24步；样本均值分别8.432→8.506、18.656→18.668 ms，变化均在约1%以内，不宣称单机加速，也未见明显回退。两组频率采样均为1620 MHz。

固定 pool 大小逐 rank 相同：8K 每卡 compute/GIN/notify 为5,415,405,144 / 2,212,397,824 / 34,435,744 bytes，20K 为6,203,848,024 / 3,157,356,288 / 85,610,368 bytes；新旧完全一致。这里不包含权重、输入输出、按实际路由申请的 tape 或 allocator 空闲缓存。

## main 如何拆解

保留逐 CTA/warp 的 main 进入/退出、Get 发射/完成、远端包本地 pull、输入 credit 等待、首次实际 WGMMA、每个 FC1/FC2 tile 的 math/epilogue、Return Put、RX signal 导入、reduce 和 terminal 事件。首次 WGMMA 的标记在首个 `s.full` 等待之后；原有 `fc1_start_clock` 在输入等待之前，不能当成真正开始 GEMM。

以下是旧版带插桩的两个完整样本，单位 ms。选取最后完成 FC2 的 rank，而不是把轻负载 rank 的长等待误认为 return 耗时。

| 关键 rank / 名义 tokens | 首次 GEMM | 最后 FC2 完成 | reduce 开始 | reduce 完成 | main 终止 |
|---|---:|---:|---:|---:|---:|
| rank2 / 8K | 0.055 | 10.956 | 11.163 | 11.324 | 11.535 |
| rank8 / 20K | 0.740 | 25.246 | 25.295 | 25.805 | 26.093 |

这些是重叠流水线的时间点，不能将 dispatch、GEMM、return 的各自跨度相加。插桩增加计时、记录及寄存器压力：旧生产 main 为 168 registers、stack320 B、静态 spill store/load 8/8 B；timeline 版为 168 registers、stack328 B、54/116 B。因此绝对 timeline 耗时不作为正式性能。

### dispatch → 首次 GEMM：消除了不必要的远端启动依赖

旧 scheduler 在所有节点都从 source group0 开始。节点0能立即计算本地 X，节点1却先等节点0的 RDMA Get→本地转发→X-ready，尽管自己的 X 已经可读。

| 首次真实 GEMM，rank 中位数 | 8K | 20K |
|---|---:|---:|
| 旧版节点0 | 41.8 μs | 44.7 μs |
| 旧版节点1 | 631.5 μs | 720.2 μs |
| local-first 后全部 rank | 41.4 μs | 42.7 μs |

改为每个 token window 内，从 destination 所在节点的 source group 开始循环，随后处理其它 group。所有 window 都采用这一顺序，不只是第一个 window；不是优先某台固定机器。Get worker 仍在并行搬远端 X，math 先消费本地已就绪输入。

`WindowShape::window/source_begin` 互为正反映射；`token_begin`、source packet sequence、ring generation 和稳定的 saved-row 编号不变。单机 `first_group=0`。不是重排数值归约顺序，也不改变接收的路由集合。

启动少等约 0.6–0.9 ms 不代表 main 必然少同样的时间。20K 样本中，rank8 最后 FC2 从25.246降到24.268 ms，但关键路径转移到 rank4；该单次 timeline 的整体 main 几乎不变。正式多步随机负载才用于判断是否保留。

### GEMM → return：已重叠，优化每 tile 的发布成本

旧版 Return Put 约在 main 开始后1.3–1.9 ms 就陆续发射，不是等所有 FC2 完成才发送。每个 M-block 的全部 FC2 N-tiles 发布后，progress 通知对应 rail，Put worker 发送；不同 M-block 的 FC1/FC2、Return 同时在途。

8K 关键 rank 上，FC2 math→tile 发布的 epilogue 中位数14.3 μs；最后一块 FC2 math→该块 return-ready 发布中位数24.1 μs、p95 43.6 μs。20K 关键 rank 的 FC2 epilogue 中位数12.8 μs。这里包含写出、同步及 system fence，不是纯 store 指令时间。

消融直接从 WGMMA accumulator 做 BF16 转换、四 lane transpose、`uint4` 写出，替换 accumulator→shared→`uint4` 的中转。复用 FC1 已有的 transpose helper；不扩大每线程 store，仍是128-bit。去掉一次 shared 写/读和对应 ready barrier，保留 S 读取退休、system fence 及贡献发布。只用于 network 特化，单机沿用已保留的 epilogue。

直接写回会改变 warp 内地址合并方式，不能凭少一次 shared 中转就断言一定更快；需要消融。生产 entry 的 168 registers、stack320 B、spill8/8 B 未增加。

在最终保留版的配对 timeline 中，8K/rank2 的 FC2 epilogue 中位数从 local-first 的13.70降到9.47 μs；20K/rank4 从14.37降到9.54 μs。每块最后 FC2 math→return-ready 发布中位数分别从25.07/26.27降到19.90/21.66 μs。最终 timeline 的静态 spill 为42/88 B，也不同于旧 timeline；这些数值只作为定位证据，正式收益另测。

最终保留版8K/rank2 的最后 FC2/reduce开始/main终止为10.613/10.825/11.196 ms；20K/rank4 为24.665/24.843/25.622 ms。仍保留完整 join，因此没有声称最终版本做到6–7 μs的衔接；那是下面未保留的提前 reduce 实验。

### return → reduce：区分真正依赖和无关回收（实验未保留）

旧版所有 CTA 的 math/producer/Get/Put/progress/pull 全部退出后，再做节点内 rank barrier，最后启动 reduce。这还等待了“给其它 source 发送”的 Put 退休和 credit 收尾，而这些不是本 source 读取 contribution 的必要条件。

候选方案只等待：

1. 节点内所有 math writer 完成，确保本机专家直接写来的 contribution 可见。
2. 本 rank 的 progress worker 已通过 GIN `readSignal` 导入全部入站 return，确保 RDMA payload 可见。

每个 progress worker 只在其全部 RX 到齐后执行一次 device-scope acq_rel 计数，组成可见性传递链；math acquire 到全部 worker 后开始 reduce。没有用普通 load 代替 GIN 的 payload acquire。此时其它控制 warp 仍可完成出站发送和 credit 工作。

计数的 expected 值就是编译期 `kProgressWorkers`（本配置19），不是 token 或 fragment 数。每个 worker 按 notify 已生成的各 channel `plan.count` 判断 `rx_cursor == rx_base + count`，全部满足时只发布一次；不增加 route 扫描或额外 kernel。

reduce 后用独立 CTA barrier 汇合 math/control，再进入原有完整 grid/global 结束同步；只有所有 NIC 请求、RX/X 读者和 reduce 都结束后，才推进 cursor/base 并允许下一调用复用。新增一个8-byte 调用状态计数，不新增逐包屏障或按 tokens 增长的持久容量。

配对 trace 对同一个关键 rank 的衔接观察如下；左列已经启用 local-first，右列叠加 FC2 直接写回及提前 reduce。这里的阶段差说明依赖被拆开，并非单独开关的正式性能收益。

| 最后 FC2→reduce 开始 | local-first | 新候选 |
|---|---:|---:|
| 8K / rank2 | 199.3 μs | 6.2 μs |
| 20K / rank4 | 193.5 μs | 7.3 μs |

8K/rank2 的 reduce 比最后一个 control worker 退出早232 μs；部分 rank 可提前约296 μs，20K 最多约312 μs。`reduce_minus_all_workers_done_us` 允许负值，用来表示这种重叠。各 rank 的终止等待可能变长，因为它们更早做完了 reduce；不能把这当成新的同步回退，也不能把所有 rank 的节省相加。

最终未保留这项改动：无插桩测试没有重复出稳定的整体收益。8K 的独立 B→C→C→B 对照中，B/C 的 p50 为11.717/11.764 ms、样本均值为11.997/12.033 ms；提前 reduce 没有更快。20K 观察值 B/C 为25.139/25.115 ms，但 C 两轮和 B 后一轮出现短暂1590/1605 MHz，**不能据此判断严格同频胜负**，也不筛掉个别慢样本重算。首组干净同频的 B 为24.901 ms。无法据此证明这种更复杂的同步有收益，因此恢复原完整 join；实验实现保存在 commit `a2a3a0a`，通过 `WEFTMOE_PF_EARLY_REDUCE=1` 可重现，正式版本已移除该开关和计数。

这也说明“局部 barrier 变短”不等于“整段 main 变快”：reduce 仍需等最后的必要 RX，收尾的全局复用同步仍在。提前 reduce 可能改变 HBM/NIC 竞争，但本轮没有足够证据将差异归因于某一种竞争。

### 仍然存在的成本

- 轻负载 rank 的“最后 FC2→reduce”可达数 ms，主要是在等其它重负载专家。旧样本所有 rank 中位数为8K 4.28 ms、20K 9.28 ms，但最后完成计算的 rank 只有207/49 μs。不能把前者全算成可消除的通信同步。
- 旧版 reduce 的各 rank 中位跨度约231/548 μs；终止 rendezvous 在本 rank reduce 结束后还约155/237 μs，中间包含等其它 rank reduce 的时间。
- FC1/FC2 的 TMA stage wait 仍值得研究，尤其 FC2。20K 关键 rank 上每 CTA 累积等待中位数约1.68/3.11 ms；其中也包含 producer 等 FC1 credit 后才发射输入的时间，不是单纯 HBM/TMA 传输延迟。
- network 的 FC2 不再需要 `Shared::output` 中转，但本轮保留公共 Shared 布局，不借此改变 stage 数。后续可以单独消融释放这64 KiB并将输入 pipeline 从3增至4 stages；需要与单机类型隔离，也需要区分真正的传输等待和 FC1-ready 依赖，不能预先把累计等待全当作可恢复收益。
- 一部分 M-padding 和跨 rank 专家不均衡仍在。本轮没有改变专家 placement，也不宣称静态队列的所有空闲都被解决。

## 复现

构建与运行使用 `benchmarks/build_pf_main_timeline.py`、`benchmarks/run_pf_main_timeline.py`。`--timeline` 构建诊断库；`trace` 模式将每 rank 快照保存到 build；`benchmarks/analyze_pf_main_timeline.py` 生成同设备的分段与配对统计。普通 `performance` 模式绝不下载 timeline。

旧生产库在 `build/pf_notify_production/network`。本轮 artifact、日志、manifest 和原始 timeline 在 `build/pf_main_timeline`。`benchmarks/summarize_pf_main_timeline.py` 汇总正式轮次、workload hash、数值验证、timeline 摘要及测量窗口内频率；原始大 trace 不提交仓库。

最终代码为 `269f63a`；插桩基础为 `fd447fe`，未保留的提前 reduce 实现为 `a2a3a0a`。生产网络 artifact 为 `build/pf_main_timeline/fc2_register`，先前冻结时显式指定 `WEFTMOE_PF_FC2_REGISTER_STORE=1`，其有效实现与最终默认配置一致；单机为 `build/pf_main_timeline/local_final`，最终诊断版为 `retained_trace`。两个优化开关默认开启：`WEFTMOE_PF_LOCAL_FIRST=1`、`WEFTMOE_PF_FC2_REGISTER_STORE=1`；本轮继续使用 B4096/D2 测试参数。提前 reduce 不在最终代码内。

最终保留路径通过160个训练 rank/step（双机普通及bias8/B512/D2热点，单机普通），检查 output/dX/dGate/dW1/dW2，最大 relative L2 为2.7500e−4；另160个 inference rank/step（双机eager与两次CUDA Graph replay、单机eager），最大 relative L2 为9.3530e−5。16个公共头独立编译和 Python 脚本语法检查通过；没有添加逐函数名的单测。应用退出后固定池释放，无残留 benchmark GPU 进程。

频率审计保留了所有观测，不删除不利轮次。除上述最终复核和 inference 的20K例外外，补充的 `compare_7_early_reduce`、`compare_8_early_reduce`、`compare_9_fc2_register` 20K轮次也出现1590/1605 MHz，均发生在调用内；不拿它们证明严格同频优势或回退。100 ms采样不能排除采样间的短暂变化。首组六轮、最终8K复核、CUPTI分项、8K inference及单机回归均有非空采样且观测为1620 MHz。

机器可读的全部轮次、hash、workspace、验证、频率偏离与timeline摘要见
[pf_main_timeline_20261012.json](../benchmarks/pf_main_timeline_20261012.json)。本轮的“旧版”均为优化前的PF，不是V3或Megatron；没有重测其它实现。

## main 代码清理回归（2026-10-12）

`main` 先快进到 `565037e`，以 tag `archive/pre-pf-dt-cleanup-20261012` 保存完整旧树，再提交清理版本 `956f21c`。净删除约 1.85 万行：移除 V3、历史 PF、WG3/旧通信实验分支、旧配置和对应测试；当前 PF BWD 所需的共享代码迁到 `backward/`、`communication/`，没有删除 BWD。保留当前单/多机 PF、训练缓存和稳定 DT，历史性能记录不改写。旧报告中的路径与开关在上述 tag 下复现。

这轮只清理，不调 GEMM 参数、不改变缓存/通信协议。正式计时无并行编译；SM 固定 1620 MHz，十次性能运行共有 3984 条 GPU 时钟观测，均为 1620，采样无失败。PF 使用相同 seed123、10% token 波动和随机路由；warm8/sample24，每步最慢 rank 的 p50。缓存 FWD 包含当前权重准备、notify、申请、main、同步，以 tape 仍存活的 cache-ready 时间为准。推理复用准备好的权重；固定工作区均已预热。

| PF（ms，清理前 → 后） | 8K | 20K |
|---|---:|---:|
| 单机 EP8 推理 | 6.816 → 6.899 | 16.654 → 16.622 |
| 单机 EP8 缓存 FWD | 8.211 → 8.152 | 18.179 → 18.193 |
| 双机 EP16 推理 | 10.207 → 10.232 | 22.839 → 22.512 |
| 双机 EP16 缓存 FWD | 11.597 → 11.670 | 24.955 → 25.085 |

DT 为单机 EP8、同一 Flash shape，固定 tokens/pool64，CUDA Graph 每次20次调用；表中是设备事件除以20后的每调用时间，不与上表 eager 时间混用。清理前 DT artifact 带有历史 T256 容量消融，但本轮全部 <=64，使用相同的原版 DT 算术与 epilogue。

| DT（µs，清理前 → 后） | 1 token | 8 tokens | 32 tokens | 64 tokens |
|---|---:|---:|---:|---:|
| EP8 | 125.056 → 124.578 | 297.476 → 298.006 | 485.126 → 481.298 | 688.818 → 687.866 |

全部配对 workload hash 一致。PF 差异 −1.43%～+1.22%，DT 差异 −0.79%～+0.18%，未观察到明显回退；这是单轮成对回归，不作为新的优化收益宣称，也没有重新测量 BWD 延迟。

验证：Python 117/117、Host 17/17、16个公共头、34项退役 API 编译拒绝检查通过；PF 单机/多机、DT、CUDA 测试程序均重新构建。EP8/EP16 的 8K/20K 各检查两组变化路由的 output/dX/dGate/dW1/dW2，最大 relative L2 分别为 1.16275e−4 / 1.09794e−4。DT EP4 检查 1/8/32/64，EP2/E32 检查 1/64 并确认 M16/M64 两种入口及 Graph replay；native Forward benchmark EP2/T512 也通过。没有增加逐函数符号名测试。

构建、运行命令、二进制 SHA256、逐样本时间、工作负载 hash 和校验汇总见 [pf_dt_cleanup_20261012.json](../benchmarks/pf_dt_cleanup_20261012.json)。原始运行日志与时钟采样保存在 `build/pf_cleanup/results`；清理前 PF artifact 分别是 `local_final` / `fc2_register`，清理后是 `cleanup_final_local` / `cleanup_final_network` / `cleanup_final_dt`。
