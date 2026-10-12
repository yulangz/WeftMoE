# PF Backward 对齐现版 Forward（2026-10-12）

基线是 `main@1cfb423`；实现分支 `feature/pf-backward-20261012`。
旧静态 BWD 保留为迁移期间的正确性/性能对照，不将新 FWD 的收益外推到 BWD。

## 设计与里程碑

- [x] M1 单机：复用 PF destination-pull、输入 ring 和完整 source
  contribution；producer/math 独立重放静态 dQ/SwiGLU/dX 任务流。
  dX epilogue 从 accumulator 直接 128-bit 写回。所有本地 dX 完成后
  math 开始静态 dW，通信 warp 进行 source reduce。
- [x] M2 多机协议：复用当前借用 warp 的按节点去重、逐 token Get、机内 pull、
  直接 TX contribution/Put return。路由和计数来自 FWD tape，不再重新
  counts notify。RX fanout、TX NIC completion 和远端 credit 分开退休。
- [ ] M3 优化与回归：分别测 dQ/dX/dW 参数、缓存写出和激活融合；覆盖随机
  tokens/routes、热点/空专家、多轮 slot 回绕。FWD 和 DT 不改变数学流程。

数学记号：A/B 为 FC1 gate/up，r 为 router 权重，Q=r*SiLU(A)*B。
dQ=dY*W2^T；SwiGLU backward 同时生成 dAB、dr 和供 dW2 使用的 Q。
dX=dAB*W1^T；dW2=Q^T*dY；dW1=X^T*dAB。权重方向以数学表示为准，
公共 API 保持原有 `[expert, out, in]`。dr 返回选中 route 的梯度，上层
框架继续处理 router softmax/归一化及 router 线性层。

## 单机第一阶段

输入 pull 和 reduce 复用 `local_kernel.cuh`，通过编译期参数选择读退休
条件与线程域，Forward 默认实例不变。独立 dQ/dX N-tile 任务不经过
旧 BWD 的共享 task mailbox；两个相邻 CTA 沿 N 共享 multicast A。
SwiGLU 由一对 CTA 分别处理半个 M-block，保留旧实现的 BF16 舍入和
行内 dr 归约，不引入浮点 atomic。

保存的 X/A/B 和路由属于 FWD tape；物理 ring 行经已有 saved-row 映射
写入专家连续的 dW cache。首版先复用现有 Q 工作区：dQ 写入后由
SwiGLU 原地覆盖成 Q，dAB 同时写完整 cache 和供 dX 使用的有界 ring。
不增加单独的全尺寸 dQ 或 dX 临时张量。

当 `2I <= H` 时，dQ 消费完的 dY ring 原地作为 dAB ring，物理行 stride
仍为 H。下次 pull 必须等本代所有 dX 读者完成，不能沿用 Forward 的
FC1-done credit。`2I > H` 才申请独立 dAB ring。已有 Forward contribution
直接用于 dX；不在 BWD peer workspace 再申请一份无用 contribution。

持久 peer workspace 目前仍保存 dY cache 和 IPC 可见 dr 标量；容量增长
与地址交换沿用现有策略。单机初始化使用 FWD tape 的专家目录，只生成
dW 所需 padded-row 映射、清空本调用计数器和空专家梯度，不再初始化
旧 activation scheduler 或重新拷贝 route headers。

## 测量口径与初始数据

固定 1620 MHz，Flash H4096/I2048/E256/top-6，EP8 为单机；EP16 为双机。
每 rank tokens 正态波动 10%，routes/gates 每步变化，共享专家 bias=0.5。
固定 seed=123，先完整 prime trace，再 4 warmups / 8 samples。
表中为同一步最慢 rank 的完整 FWD+BWD wall latency 的 p50，包含每调用
权重重排、缓存申请及回收，排除输入构造与首次通信池创建。

| 对照 | 8K (ms) | 20K (ms) |
|---|---:|---:|
| 旧 PF BWD，单机 EP8 | 26.483 | 68.624 |
| Megatron 缓存模式，单机 EP8 | 24.792 | 59.548 |
| 新 PF BWD，单机 EP8（最终候选） | 25.209 | 62.164 |
| 新 PF BWD，单机 EP8（复测） | 25.193 | 62.178 |
| 旧 PF BWD，双机 EP16 | 32.668 | 78.531 |
| Megatron 缓存模式，双机 EP16 | 33.288 | 87.202 |

双机旧 PF 基线检测到测量区间内 1455/1470 MHz 短暂降频，因此这一行
仅作为原始观测，正式同频比较需要重跑，不筛掉慢样本重算。
最终单机候选也检测到一次 1590 MHz（20K sample 4）；复测期间有一次
1485 MHz。设置为 1620 MHz 不代表实际运行始终维持该频率，以上均
保留原始样本，不能将百分之一量级的差别解释为稳定收益。

原始 manifest、workload hash、逐 rank 时间与频率采样保存在
`build/pf_backward_20261012/results/`。以上是本轮短对照，不替代历史长测。
脚本的 `backward_slowest_rank_p50_us` 是 host API 边界差值，不应称为
纯 BWD kernel 时间；单独设备阶段时间应另行测量。

初始单机原型通过 EP8、129/8K/20K 的 output/dX/dr/dW1/dW2 检查。
初版编译出现 setmaxnreg 忽略和 96/76 B 静态 spill store/load，正在
分离寄存器域和移除 dW-only 模板中的不可达 activation 分支。正式收益
以最后保留代码的无插桩测量为准，不将编译候选当作已完成版本。

初始任务流每个 expert 分别排 dQ/SwiGLU/dX。完整周期 8K/20K 为
29.112/63.671 ms：8K 比旧版慢，20K 较快。把 dY/dAB ring 复用并移除
冗余 contribution 后为 29.069/63.955 ms，没有稳定速度收益；该改动
主要减少容量。后者在未计时阶段观察到 1530 MHz，已保留频率记录。
反汇编表明已见的 spill 主要位于调度与阶段切换，不在 GEMM K 热循环；
增大 control register quota 仍不能恢复 setmaxnreg，不据此声称优化成功。

已保留的主要性能优化是将相邻专家合并为一个有界 compute wave：dQ 的输出列数
只有 Forward FC1 的一半，逐专家切阶段可能让短 expert 的 dQ 任务不够
填满 CTA。合并不改变 M padding、归约顺序或路由，只扩大阶段内可用
的静态 N-tile 任务集合，仍由 ring depth 限定窗口。
该项单独测量为 25.116/60.884 ms，频率采样均为 1620 MHz。
随后简化初始化并增加阶段时钟，四级 operand pipeline 测得
24.956/64.017 ms；三级最终候选为表中的 25.209/62.164 ms，保留三级。
最终版相对旧 PF 完整周期约快 4.8%/9.4%，但没有追平本轮 Megatron。
尚有 ptxas setmaxnreg 忽略与阶段切换 spill，未宣称已消除。

最终候选通过 EP8、1/129/8K/20K、每规模两次变化路由的 Torch 检查，
包括 output/dX/dr/dW1/dW2；全部相对 L2 误差最大约 0.00142。
四级前一候选另外通过 bias=8 的热点/空专家及 512-token window 回绕。
Python 回归为 117 tests / 470 subtests。没有新增碎片化单元测试。

## 多机实施约束

FWD tape 的 immutable routes/counts 可复用，但物理计划属于共享工作区，
可能已被后续层覆盖。必须重新规划并按当前 channel base 绑定 RX/TX，
不能复用 FWD 已退休的 packet generation。
dr 使用独立小型 FP32 TX/contribution 平面与 GIN 完成信号；dX 和 dr
分别 acquire 成功后才返还 packet credit，TX 仍保留本地 NIC completion。
通信共享内存必须与 dW arena 分离，否则为了复用 arena 而等待所有
Return 结束，会让 dW 与 Return 串行。

M2 已实现。`backward_kernel.cuh` 共享单机/多机 dQ/SwiGLU/dX/dW 数学流程；
`backward_network.cuh` 只负责窗口任务流、slot 计数单位、dr 地址和终止协议。
源 contribution、compute block 与 NIC packet generation 均单调前进。
每次 BWD 重新绑定 RX/TX/本地 contribution 地址，但不重新交换路由计数、
不重新分配通信池。结束时将共享 compute counters 归一到 FWD 的 N-tile
单位，下一次 FWD 可继续使用同一套 slot。

dr 为每 route 一个 FP32，新增 registered TX/contribution 小平面，不改
原有 BF16 dX/FWD payload 的行 stride。每包 dX、dr 各自 Put+weak signal；
接收方 acquire 两个信号后才 acknowledge。独立 flush request 保护本地
NIC 对两份 TX 的读取。FWD 不发 dr Put，也不推进 dr signal。

通信 staging 与 dW arena 分开；384-thread CTA 中仍为 256 math、32 TMA
producer、96 communication。保留 64 Get / 48 Put / 19 progress / 1 admit
的借用 warp 分配。compute-only grid completion 保证全部本地 dX 完成
后才执行 dW；通信继续 return/reduce，与 dW 并行，不加入该 compute join。

双机验证：129/8K/20K 各三轮变化路由通过完整 Torch 校验；另外，首个
协议候选通过 8K/20K bias=8 的热点/空专家、512-token window 回绕。
流程按节点数参数化，目前硬件只验证两节点、每节点 8 GPU。

性能仍待优化：相同 2048-token window，首版完整周期为
33.712/85.982 ms，慢于旧 PF。仅将 window 放大到 4096，得到
32.271/80.742 ms；此时实际路由的 M128 padding 仍为约 14.2%/10.7%。
放大窗口同时增大持久 ring 容量，因此不将它悄悄改为默认配置。

## 后续消融与诊断

同一组随机 workload、相同 2048-token window：

| 实验 | 8K (ms) | 20K (ms) | 决策 |
|---|---:|---:|---|
| M2 单调 generation 原版 | 33.712 | 85.982 | 保留 |
| 尾块跳过空的第二个 math WG | 34.815 | 87.720 | 撤回 |
| 尾块两个 WG 各算一半 N，均计算有效的前 64 行 | 34.034 | 86.287 | 撤回 |

第一个尾块方案出现 ptxas C7518（WGMMA 序列化）。第二个通过数值验证，
也没有 C7518，但网络 kernel 的静态 spill store/load 从 76/132 B
增至 124/216 B，没有测到速度收益。不能将降低 padding FLOPs 等同于
降低耗时。尝试提前执行 control reg-dealloc 也没有消除 C7507
（setmaxnreg 被忽略），同样撤回，不留运行时开关或死分支。

8192-token window 测得 31.396/77.158 ms；M128 padding 从 2048 window
的 23.19%/22.17% 降至 9.99%/5.43%。但 compute+GIN 池从
5650/8517 MiB 增至 13269/16135 MiB，约增加 **7.4 GiB/卡**。
4096 window 也增加约 2.35 GiB/卡。这些是容量/性能消融，不更改默认值；
窗口同时影响 FWD，不能把完整周期收益全部归给 BWD。

`m3_phase_profile` 使用 GPU 时间戳：各规模三次调用、16 ranks 中，
main 入口到首次 dQ 的最大观测约 46/47 us，dQ 开始到该 rank 全部
dX 完成最长约 15.84/34.80 ms，dW 阶段最长约 10.90/20.72 ms，
reduce 最长约 1.13/2.94 ms。不同 GPU 时钟不互相相减，阶段最大值
也不能相加当作总耗时（可能来自不同 rank/step，且阶段存在重叠）。
该组带诊断、包含初次分配，不能替代上面的无插桩性能结果。
目前瓶颈不是 main 到首次 GEMM 的启动空隙，优先看窗口内激活计算、
padding 与 dW。快 rank 的 dX 到 reduce 间隔还包括等待其它 rank，
不能解释成纯 RDMA 耗时。

同口径 cached FWD 单独回归：旧版 12.180/29.735 ms，新版
12.328/29.680 ms，没有发现明显退化；单机完整周期回归
25.165/62.174 ms。DT EP2/4/8、E256、1/64 tokens 的数值与 graph replay
检查通过；没有修改 DT 数学流程。

### 多 tape 生命周期修正

上述初始性能数据来自 FWD 紧接 BWD 的单 tape 周期，不能视为完整训练
生命周期验收。`m3_lifetime_network` 在三份不同热点、不同长度的 FWD
tape 同时存活时发现错误：Notify handle 保留了路由、计数和检查点，
但物理 ring plan 是共享执行工作区，不会随每份 tape 保存。
新 BWD 直接使用最后一次 FWD 的物理计划，因此早期 tape 的反向错误。

修正为每次 BWD 复用现有 PF planner，从该 tape 的 routes/counts 重建
物理计划；`training_counts_ready=true` 跳过 counts notify，不申请另一份
通信池或逐层保存大计划。dY staging 合入 planner，移除原来的独立
D2D 拷贝；网络 BWD 不再自己重复实现 RX/TX 地址重绑定。准备工作只
留下 canonical dW 行目录、padding 清零与 BWD slot 计数单位转换。
**修正后的性能须重新测量，不能沿用上表当最终结果。**

`m3_tape_network` / `m3_tape_local_accum` 已通过 EP16/EP8 的三份同时
存活 tape、热点迁移、短批次、权重原地更新、通信池释放和两层推理 graph
切换检查（H4096/I2048/E32/top-6，129 tokens）。沿用现有集成测试。
单机权重更新测试还暴露参考实现的累加口径差异：一项接近零的 dX
误差为 2.098e-5，略超 2e-5 绝对阈值。参考原先在 BF16 内累加 output
及不同 expert 的 dX，现改为 FP32 累加后转回 BF16；各 GEMM 输入、
中间结果仍为 BF16，**没有放宽容差**。与性能测试已有 FP32 combine
参考的口径一致。只改 output 的中间候选不足以修复 dX 的累加差异，
也保留了该失败日志。

修复版 `bwd_tape_plan` 通过 EP16、129/8K/20K、每规模两次变化路由，
512-token window 的完整数值检查，全部输出/梯度相对 L2 最大约
0.000734；8K/20K 最大约 0.000111/0.000105。Python 回归仍为
117 tests / 470 subtests。

收尾同步：return/reduce 可以在 dW 期间执行，但保护下一次调用复用
metadata 的最后一道 rank/network barrier 移到完整 CTA/cluster 的 dW
结束之后。不增加 barrier 数量，也不把 return/reduce 与 dW 串行化。

## 生命周期修复后的最终测量

代码 `27ba2b5`，artifact `build/pf_main_timeline/bwd_tape_plan`。
以下均包含每次 BWD 的物理计划重建，仍采用开头的完整周期口径、
2048-token window、4 warmups / 8 samples；新旧 workload hash 相同。

| 实现 | 单机 EP8 8K / 20K (ms) | 双机 EP16 8K / 20K (ms) |
|---|---:|---:|
| 旧 PF BWD | 26.483 / 68.624 | 33.514 / 81.035（复测） |
| 新 PF BWD | 25.236 / 64.490 | 34.563 / 86.551 |
| Megatron moe_act 缓存口径 | 24.792 / 59.548 | 33.288 / 87.202 |

单机比旧 PF 快 4.7%/6.0%；相同 2048 窗口的双机仍慢 3.1%/6.8%。Megatron 行是
本轮前段测得的相同 workload，不是新做一轮框架训练，也不是 V3。
此处旧 PF 指本分支起点 `main@1cfb423` 的 BWD，**未重新测 V3**。
不能宣称多机性能优化目标已经完成。

修复版另外测了 4096-token window：**33.733/81.761 ms**，比新版
2048 window 快 2.4%/5.5%。它接近旧 PF 的 2048 window 数字，但没有
复测旧 PF 的 4096 window，不能宣称同配置已追平。该版本比本轮
Megatron 8K 慢约 1.3%、20K 快约 6.2%；这些短测不代表框架端到端。
相对新版 2048 window 仍多占约 2.35 GiB/卡，持久池总量仍低于旧 PF
2048 window。此轮没有修改缺省容量：TensorRuntime 本来就是单机
2048、双机 4096；表内 2048 是 benchmark 显式选择的对照值，而非
双机 API 缺省值。所有这些测量均显式设 `WEFTMOE_PF_MIN_RING_DEPTH=2`
和 `WEFTMOE_PF_CREDIT_WINDOWS=2`。

设置时钟为 1620 MHz；新双机在 20K timed sample 0 观察到一次 1530 MHz，
旧双机复测有一次 1515 MHz、但不落在测量区间内。单机新版本有一次
1590 MHz，同样不在测量区间内。保留全部样本，不按频率删除样本。
频率只是间隔采样，不等于证明间隔内无瞬时降频。
4096 window 在 20K timed sample 5 观察到一次 1605 MHz，另有一次
1530 MHz 位于测量区间外；同样未删样本。

相同 2048 window，双机 rank 0 的持久 compute+GIN 容量：旧版为
8518.7/12133.2 MiB，新版为 5650.1/8516.6 MiB，约减少 2.8/3.5 GiB。
这不是总峰值显存（不包括当次的 Torch tape/dW/scratch）。主要来源是
复用 FWD contribution/dY-dAB ring，以及不再保留完整 dX 临时数据。

最终回归：EP8 的 1/129/8K/20K 与 EP16 的 129/8K/20K，各两次变化
路由，完整 output/dX/dr/dW1/dW2 检查通过；EP8/EP16 的 8K/20K
纯推理通过数值及 PF kernel 路径检查。多 tape、训练到推理共卡切换、
DT EP2/4/8 与 Python 回归见上文。未测三节点以上，也未在完整 AReaL
或 Megatron 模型中跑训练 step，不将算子级结果等同于端到端训练收益。

仍待完成的性能工作：在不增加全尺寸激活暂存的前提下，减少多机
window/expert 的 M padding 与阶段尾部空转；进一步消融 dQ/dX 的
N tile 与 operand pipeline；解决 setmaxnreg 被忽略及阶段切换 spill。
这些只是后续方向，尚未证明有效。没有保留失败的尾块分支，也没有
改动 DT 或推理的缺省调参。
