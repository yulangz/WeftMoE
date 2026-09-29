# 单机 decode 专属实现与验收记录

日期：2026-09-29。基线 HE 为 `bc71411`；HE 完整验证及其收益见
[ring 记录](static_forward_v4_ring_record.md)。本记录单独跟踪 HF，不把长文
GA+FZ 的既有收益算成 decode 优化。本轮先对齐 MegaMoE 的通信、计算和同步
工作流，没有进行 MNK 搜索。HF 已完成验收并采用：短文两轮对照均领先
MegaMoE，单机长文和多机回归持平。

2026-09-29 后续 HG：为 SGLang/Qwen 接入补齐单机 EP1/EP2/EP4/EP8，见第 6 节。

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

## 6. HG：单机 EP 通用化与 SGLang 双池接入

HF 将每卡专家数限制为 32：geometry 用一个 warp 的每个 lane 对应一个专家，
本地 rows/blocks 数组也是固定 32。Qwen E256 在 EP8 下满足条件，EP4/EP2/EP1
分别需要 64/128/256 个本地专家，原先只能回退到通用 V4。这不是通信或 GEMM
的限制，不能仅扩大数组而保留原来的一轮 warp scan。

HG 改为每轮 32 个专家，warp scan 的最后一项作为下一轮的 block 基址。
所有轮次累积实际 row/block 总数；任务编号、GEMM、gate/up packing、归约顺序
及通信协议不变。source counts 在 BeforePull 前已完成发布，因此它与 geometry
的 rows 共用一块 SMEM scratch。当前仍保留总专家数不超过 256、单节点和原
短容量/profile 条件，不再额外限制每卡专家数。该路径覆盖 EP1/EP2/EP4/EP8；
experts 必须能被 EP 均分。

编译资源保持 112 registers、0 spill；动态 SMEM 由 207 KiB 增至 208 KiB，
GM workspace 分配不变。8 个原有通用 V4 主 kernel 的资源元组和 SASS 指令
逐 entry 完全一致，未改动 prefill、训练或多机的计算实现。

SGLang 为图捕获时的 `local_tokens <= 64` 准备独立容量 64 的 runtime，
其余 bucket 和 eager/prefill 继续使用大池。capture bucket 在各 rank 等长
补齐，故选择一致；不能在不等长 eager DP 输入上按各自 token 数选择不同池。
大池容量也按实际布局计算：复制布局按 EP 分片，DP-local 布局不再除以 EP。
两套 runtime 使用同一执行 stream，offload 先销毁外层图，再释放两个池及
输入 staging；更新权重后重新捕获，避免保留旧 packed W1。

### 验证

- E256/K8，EP1/EP2/EP4/EP8：现有 static inference 测试共 **465** 个 rank/case
  通过，含严格逐位 batch 1/7/32/64、置换、不等长/空 source、重复热点、图输入
  变更、权重刷新，以及专属 decode 与通用 V4 的逐位对齐。
- 真实 SGLang FusedMoE：四种 EP 分别测试全局 `64*EP` 和 `64*EP+1` 个复制
  token，覆盖小池边界两侧；**30** 个 rank/case 通过数值、图回放、双池释放、
  权重变更后重新捕获。单层结果不等于整模型/RL 验证。
- EP1 的 rank0、EP2 的 rank0/1：decode memcheck 均 0 errors，racecheck 均
  0 hazards；两类工具下合计 **186** 个应用检查通过。
- 原 E64/K4 单机 EP8 长文回归：**240** 项通过，包括 8K/20K、热点不丢 token
  与复用；没有以短文测试代替长文验证。

### 性能（1620 MHz，单位 μs）

同一台 8×L20X，相邻独立进程 HF/HG，10 warmup、20 samples、graph batch20，
最慢 rank p50。正式计时无其他 GPU workload 或并行编译。

|Qwen E256/K8|T/rank|HF|HG|
|---|---:|---:|---:|
|EP1|1|105.893|69.390|
|EP1|64|971.586|716.998|
|EP2|1|—|61.349|
|EP2|64|—|341.038|
|EP4|1|105.470|53.968|
|EP4|64|336.587|177.136|
|EP8|1|52.718|52.768|
|EP8|64|108.061|108.370|

EP1/EP4 的 HF 数字是其原有通用 V4 回退；HG 是新启用的专属 decode。
EP4 延迟降低约 49%/47%，EP8 持平。冻结 HF 的 native benchmark 入口拒绝 EP2，
因此未填造 EP2 对照；本轮只补齐现有测试入口的 EP2 选项。以上不是与 SGLang
baseline 的性能对比，也不是相同全局 batch 下跨 EP 的扩展性比较。

原 E64/K4、EP8 的 HF/HG：T1 为 47.350/47.032，T64 为 55.280/55.558；
8K 为 1240.140/1236.390，20K 为 2918.800/2925.890，均记为持平。
全部可比较 case 的输出 hash、GM 分配字节和 graph 拓扑相同，计时前后全部卡
SM 时钟为 1620 MHz。

原始日志、复现脚本和完整检查位于 `build/local_decode_hg/`，
`verify_results.py` 已通过。library SHA256：
`d9f8a90e712b638bd9ac591ea0f801914f0ea49189190233dcd31cecbaa2860d`；
native SHA256：`88d10041e07f128a10e69f9a6b09de1dd2b970a3e4019dfbf4002b486601a572`。
完整 SGLang 服务及 inference-only RL 对照继续记录在
[AReaL 接入文档](areal_qwen35.md)，不把本节算子通过视为 RL 已完成。

## 7. HH：SGLang rollout 回退诊断与优化（进行中）

HG 的 RL 对照双方各完成 3 次 update 后，后两步 rollout 中位数慢 18.93%，
见接入文档。实际活跃 batch 主要为全局 32–64，而旧 native 表是每 rank 1/64；
E256/K8 的专家覆盖率也远高于最初对照 MegaMoE 的 E64/K4，不能沿用旧结论。

### HH1：融合图内 staging 和结果回填

保持 HG native 二进制不变，graph capture 路径把 X 分片/补齐、负 expert ID
清洗、route/gate 转换与稳定地址 staging 合并为一个 Triton kernel；Qwen 的
稀疏输出补零/回填合并为另一个。child graph 地址合同、外层 TP all-reduce、
eager/prefill 路径不变。新 helper 位于 `weftmoe/_sglang_kernels.py`。

`benchmarks/torchrun_sglang_decode.py` 测真实 SGLang EP4，E256/K8、随机 top-k，
同输入、同机交替 outer graph，5 warmup/20 samples、每次 50 replay，rank-max p50。
以下 T 是**全局复制 token 数**，单位 µs，非每 rank token 数：

|T|SGLang baseline|HG 原适配层|HG + HH1 staging|
|---:|---:|---:|---:|
|1|29.828|—|66.138|
|16|84.828|—|128.158|
|32|106.518|160.969|146.748|
|48|131.090|—|170.068|
|64|150.023|192.961|178.628|
|96|179.786|—|187.100|
|128|185.618|—|191.627|
|256|197.775|—|196.536|
|512|211.939|—|373.559|

baseline 列来自 HH1 同进程交替对照；HG 原适配层另一次同机运行的 baseline
T32/T64 为 105.789/150.416。HH1 省约 14 µs/层，尚未解决 native 部分回退。
T512 超过小池阈值，走通用 V4；当前 RL 的 HTTP 样本未达到此规模。
HH1 全部计时的 NVML 样本均为 1620 MHz，没有并行编译或其他 GPU workload。

上述 9 个 bucket 均通过 baseline 数值对照；另用现有 SGLang adapter 测试在
EP4/T17 覆盖尾部补齐、masked route 与 eager 清洗一致、变更输入的图回放、
权重更新后释放/重建，4 rank 通过。原始日志与 trace 在 `build/local_decode_hh/`。
