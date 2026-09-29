# 单机 decode 专属实现与验收记录

日期：2026-09-29。基线 HE 为 `bc71411`；HE 完整验证及其收益见
[ring 记录](static_forward_v4_ring_record.md)。本记录单独跟踪 HF，不把长文
GA+FZ 的既有收益算成 decode 优化。本轮先对齐 MegaMoE 的通信、计算和同步
工作流，没有进行 MNK 搜索。HF 已完成验收并采用：短文两轮对照均领先
MegaMoE，单机长文和多机回归持平。

## 1. 入口和适用范围

新增 `src/cuda/local_decode/`，而不是继续给通用 ring worker 叠加短文分支：

- `schedule.cuh`：容量选择、共享布局、静态任务编号。
- `compute.cuh`：固定 TMA producer、两个 math warpgroup、FC1/FC2 epilogue。
- `kernel.cuh`：路由发布、三条 pull warp、跨 rank 边界、source combine。

在已有 opt-in ring 下自动选择：单机 inference-only、pool 的
`max_tokens_per_rank <= 64`、最坏路由能装入一个容量窗口，且编译 profile 的
线程数、cluster 和 TMA box 与此实现匹配。验收配置为
`configs/static_inference_ring_20260924`，`B=512, D=8`。
非 ring、训练、多机、多窗口、小 ring 或不兼容 profile 仍走原入口。
不改变任何 workspace 分配，也不增加环境开关。

选择依据是各 rank 一致的 **pool 容量**，不是各 rank 当前 token 数。
因此不等长乃至空 source 不会进入不同协议。大容量 prefill pool 中只传入
1/64 个实际 token 时，仍走原 V4；要使用本专属入口，应创建独立短容量
inference runtime。这不是改动长文阈值，也不是把大容量 ring 改回 full arena。

## 2. 工作流、地址与同步

每个 CTA 固定为 256 个 math 线程、1 个 TMA warp、3 个 pull warp。
CTA 使用同一不可变任务序列：全部 FC1 task 后接 FC2 task，以
`blockIdx.x + n * gridDim.x` 划分；producer/math 独立重放，不领用动态任务。
相邻两个 CTA 共享 M block、承担相邻 N tile，X/S 继续使用 cluster multicast TMA。

1. math 将 X 暂存到本 rank 的 `source_input_offset`，构造稳定 combine 顺序，
   清零本次调用的 block ready 计数。CTA0 的 control 线程按实际 route 将
   `{route_id, gate}` 写入 destination 的 source/expert 目录，并发布精确 count。
2. **BeforePull**：已有 grid→rank→grid release/acquire 边界一次发布 X 和目录。
   各 CTA 从这些精确 count 推导本地 expert-major padded M-block 前缀。
3. 三个 pull warp 按固定行 residue，从实际命中的 source GPU 读取 X，
   通过 `peer GM → local SMEM → canonical_rows_offset` 搬运。
   `ring.rows[padded_row]` 同时保存 gate 和 source contribution 的最终地址。
   完成 bulk store 后 release 增加该 block 的 `x_ready`。
4. TMA producer acquire 等待 `x_ready == valid_rows`，装载 X/W1。
   FC1 维持原有 BF16 gate/up 舍入和 SwiGLU 合同，把 S 写入本地 S buffer；
   每个 N tile 的 TMA store 完成后 release 增加 `s_ready`。
   producer acquire 观察全部 FC1 N tile 完成后装载 S/W2。
   SMEM stage 的 full/empty mbarrier 负责 producer 与两个 math warpgroup 的
   循环复用，multicast empty 包含两个 CTA 的读者。
5. FC2 直接向 source 的 `contribution_offset + route_id * H * sizeof(BF16)`
   写回，不经过通用 return-fragment 目录或额外 C payload。
6. **BeforeCombine** 发布全部 FC2 peer store；source 按稳定 `(expert, slot)`
   顺序进行 FP32 累加、一次 BF16 输出舍入。保留 **AfterClean** 终端 rank 边界。

短容量的所有最坏情况 routed rows 都能在一个窗口内驻留，故本调用内没有
X/S payload slot 换代，不需要在线 admission、consumed credit 或 fragment 计数。
ready 计数只复用原 `RingBlock` 的调用私有目录，不改累计 ring slot credit。
跨调用的本地 kernel 完成和上述 rank 边界保护重用：上一调用 BeforeCombine
后不存在 X/S 读者；下一调用的 BeforePull 要等所有 rank 完成上一调用，
然后才能产生覆盖 contribution 的新 FC2 store。长文仍使用原逐 slot 有界协议。

初始 GEMM 为 M128/N128/K64、5-stage；复用已有宽 W1 TMA map、窄 W2 map及
数值 helper，不改变权重 packing。它是固定起点，不是本轮性能搜索结果。
首次照搬 208/80 动态寄存器配额时，编译器实际只分配 115 寄存器且去掉了
`setmaxnreg`，启动前的 pool 检查正确拒绝了它。删除不必要的动态配额，使用
普通编译器分配和 occupancy 检查；带完整统计的版本为 112 registers、
0 stack/local spill。原 V4 的动态配额检查保留。
HF收益是这次专属实现的整体收益；GEMM起点也随工作流切换，未做逐因素
消融，不能把全部加速单独归因于某一个同步点或某个MNK参数。

## 3. batch-invariance 合同与测试

固定 K16 WGMMA 累加次序，不使用 split-K 或浮点 atomic reduction。
其他 token 的数量/路由只改变当前 token 的物理行和 CTA，不改变其 GEMM
运算次序、gate/up 舍入或最终 top-k 累加顺序。复用 Weft 数值 helper，
不为追赶性能切换到另一套激活近似。

在现有 `tests/torchrun_static_inference.py` 增加 `--decode`，而非新增独立
单测体系。`--tokens 1 64 --decode --edge-cases` 共 248 个 rank/case 检查通过，
覆盖 eager/graph、空 source、不等长、重复热点、权重刷新、共享 workspace、
可变 graph 输入。额外严格 `torch.equal` 验证：

- batch 1/7/32/64 的相同 token；
- 行置换、各 source 不同 batch 大小；
- 全部 source 命中同一 expert，跨多个 M block 的 anchor token；
- 同样输入在独立容量 512 的通用 V4 runtime 中逐位一致。

rank0/7 memcheck 均 0 errors；decode kernel racecheck 均 0 hazards。
两轮 sanitizer 下各 248 个应用检查也通过。

## 4. 首轮收益

单机 EP8，8×L20X，SM **1620 MHz**，H2048/I512/E64/K4/BF16。
所有正式计时期间无并行编译或其他 GPU workload。
native 为 10 warmup/20 samples、graph batch20，最慢 rank p50：

|T/rank|HE（原有冻结对照）|HF 初版|HF 补齐状态统计|
|---:|---:|---:|---:|
|1|77.141|46.922|47.352|
|64|95.762|55.218|55.376|

单位 μs。两规模、8 rank 输出 hash 与 HE 完全一致，分配字节和 graph 拓扑不变。
native 与下表的输入不同，不能跨表相减宣称胜过 MegaMoE。

同输入交错 graph 对照采用现有 `benchmarks/torchrun_megamoe_decode.py`，
每轮正反序交错四个 case，10 warmup/20 samples、batch20，rank-max p50：

|T/rank|HF decode（已含 source staging）|Mega kernel-only|Mega 含3份输入拷贝|原非ring|
|---:|---:|---:|---:|---:|
|1|47.896|49.330|53.187|60.694|
|64|55.314|57.594|63.357|75.760|

首轮分别比 Mega kernel-only 低约 2.9%/4.0%，比含拷贝口径低约 9.9%/12.7%。
四 case 输出 SHA256 与 HE 记录一致。此轮是 T1→T64；独立进程反向复测见下节。
没有测试 MegaMoE 双机。

原始构建、日志和脚本：`build/local_decode_hf/`。原有 8 个 static-ring
主 kernel 的资源元组与 HE 一致。
最终额外逐 entry 比较了这8个原有主kernel的SASS指令行，全部与HE逐字节相同，
覆盖单机/多机、窄/宽入口、inference/retain-state组合。

## 5. 最终复测与回归

最终补充 host 的编译 profile 兼容性筛选；decode 的完整反汇编与已通过
sanitizer 的构建逐字节一致（SHA256
`8069ff540ad30e64392866b5e225ba461a4660e1d03758201127ebe9fb8fd480`）。
因此复用上述 memcheck/racecheck 报告，不声称对另一套设备代码重复验过。
最终二进制重新执行 native、248 项短容量应用检查及 MegaMoE 反向对照。

T64→T1、相同输入及交错 graph 口径，rank-max p50/μs：

|T/rank|HF decode|Mega kernel-only|Mega含输入拷贝|原非ring|
|---:|---:|---:|---:|---:|
|1|47.705|49.121|53.098|60.568|
|64|55.623|57.535|63.322|76.402|

两种进程顺序均领先：比 kernel-only 低约 **2.9%–4.0%**，比含拷贝低约
**9.9%–12.7%**。四 case、8 rank 的全部输出 SHA256 与 HE 完全相同，
两轮共 2560 组计时前后 SM 读数均为 1620 MHz。此处不把 kernel-only 与
含 staging 的代价混淆：HF 自己的 source staging 已在同一个 kernel 内计时。

最终 native p50/μs；短文 HE 使用首轮冻结结果，其余为本轮相邻进程对照：

|拓扑|T/rank|HE|HF|解释|
|---|---:|---:|---:|---|
|单机EP8|1|77.141|47.427|约快38.5%|
|单机EP8|64|95.762|55.270|约快42.3%|
|单机EP8|8192|1238.24|1242.18|+0.32%，持平|
|单机EP8|20480|2925.24|2924.61|−0.02%，持平|
|双机EP16|1|233.298|233.704|+0.17%，持平|
|双机EP16|64|286.632|287.315|+0.24%，持平|
|双机EP16|8192|2879.65|2893.70|+0.49%，持平|
|双机EP16|20480|6846.24|6854.99|+0.13%，持平|

单机长文应用层 240 项检查通过。上表全部输出 hash、分配字节、graph 拓扑
与 HE 相同，前后 SM 时钟全为 1620 MHz。训练保留状态的单机20K相邻回归
main 为 HE/HF `3200.61/3187.46 μs`，wall 为 `3768.24/3761.05 μs`；hash及
所有字节字段一致，记为持平，不当成训练优化收益。未修改 Backward，亦不将
本次 decode 验收表述为关闭此前完整训练性能门禁。

`build/local_decode_hf/verify_results.py` 完整通过，输出为 `verification.log`。
最终构建SHA256：native
`cd255240d8c73a52f8cb190e81375e2cd77a3cdf2272f35ab29c4cfd929a6e83`；
library `f52be6dc2c945685d417ab01b48d3309b84957e23d098da29bd128b3ef20d005`。
结论：HE 验证已补齐；HF 单机 decode 专属工作流、batch-invariance、达到
MegaMoE 性能以及不干扰长文路径的本次目标完成。没有额外 MNK 扫参或动态
工作队列，也未将该实现扩大到训练或多机。
