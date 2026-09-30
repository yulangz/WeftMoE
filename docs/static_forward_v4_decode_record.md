# 单机 decode 专属实现与验收记录

日期：2026-09-29。基线 HE 为 `bc71411`；HE 完整验证及其收益见
[ring 记录](static_forward_v4_ring_record.md)。本记录单独跟踪 HF，不把长文
GA+FZ 的既有收益算成 decode 优化。本轮先对齐 MegaMoE 的通信、计算和同步
工作流，没有进行 MNK 搜索。HF 已完成验收并采用：短文两轮对照均领先
MegaMoE，单机长文和多机回归持平。

2026-09-29 后续 HG：为 SGLang/Qwen 接入补齐单机 EP1/EP2/EP4/EP8，见第 6 节。
后续 HH 的 RL 回退定位与消融见第 7 节；第 1–5 节描述历史 HF，不能把其
M128/双 warpgroup 配置当作 HH 候选的当前实现。
2026-09-30 的 E256/K8 小 batch 三方独立对照见第 8 节；该节 T 为全局
batch，与前面的 E64/K4、每 rank T 的 MegaMoE 对照不是同一口径。

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

### HH2：跳过 M128 block 中为空的上半 warpgroup

保持 M128 物理布局、TMA box 和所有 SMEM credit 不变，仅在 `valid_rows <= 64`
时省略第二个 math warpgroup 的 WGMMA。它仍等待 full、归还 empty、参与 epilogue
同步；真实行的 K16 累加顺序和 BF16 舍入不变。热点块超过 64 行时正常计算两半。

在 HH1 上叠加此项，同机同配方、全部 SM 采样 1620 MHz：

|全局 T|baseline|HH1|HH2|
|---:|---:|---:|---:|
|1|30.024|66.138|63.980|
|16|84.870|128.158|123.373|
|32|106.437|146.748|141.588|
|48|130.948|170.068|161.845|
|64|149.973|178.628|172.087|
|96|179.378|187.100|179.694|
|128|185.586|191.627|185.087|
|256|197.956|196.536|190.254|
|512|211.983|373.559|373.049|

实际收益约 2–8 µs，而非按省掉一半 WGMMA 推断两倍提速。32/64 仍慢于
baseline，需要继续处理主循环等待和固定开销。decode 为 100 registers，
0 stack/local spill（HG 为 112）；8 个通用 V4 ring entry 的 SASS 逐指令不变。
EP1/2/4/8 共 465 个 native rank/case 通过，含逐位 batch-invariance、热点跨块、
通用 V4 对齐；同四种 EP 的 T17 与 `64*EP+1` 共 30 个 SGLang rank/case 通过。
本项尚未单独通过 sanitizer，也没有重跑 RL；最终候选再统一验收。

### 后续消融（候选，尚未完成最终验收）

- `local_decode_hh_pipe`：WGMMA 保留最新一组在飞，只在前一组退休后返还对应
  SMEM stage；末尾 drain 后归还最后一个。empty credit 从每 warp 改为每
  warpgroup 发布，期望计数从 16 降至 4，仍包含 cluster2 的全部读者。
- `local_decode_hh_sparse`：在上一项上增加 8-row X/S TMA descriptor。
  `valid_rows <= 8` 仅搬 8 行，其他块搬 128 行；物理 M128、K16 算术顺序、
  gate/up 舍入与最终 combine 顺序都不变。SMEM A 的未搬运行在调用开始置零。
  descriptor 随 decode pool 准备，热路径不修改 descriptor，不增加 GM payload。

同机 EP4 的交替图测量，全部 NVML 时钟样本 1620 MHz，单位 µs：

|全局 T|baseline（稀疏轮）|HH2|流水 + WG credit|再加稀疏 TMA|
|---:|---:|---:|---:|---:|
|1|30.103|63.980|63.942|63.154|
|16|84.849|123.373|121.026|115.742|
|32|106.629|141.588|139.844|131.951|
|48|131.141|161.845|160.551|152.216|
|64|150.067|172.087|169.774|161.104|
|96|179.545|179.694|177.834|168.662|
|128|185.381|185.087|182.490|172.935|
|256|198.149|190.254|187.591|182.134|
|512|211.962|373.049|373.843|373.980|

流水项只省约 2 µs，稀疏搬运再省约 8–10 µs；并未完全解决 32/64 的差距。
两轮的 9 个 bucket 均通过 baseline 数值对照，不代表完整 batch-invariance
和 sanitizer 验收。

另用 **EP1** 的 T32 做 NCU 完整 graph 诊断，见
`build/local_decode_hh/ncu_sparse_graph_r2_details.txt`。采样报告为约 38% DRAM、
15% compute 利用率，约 79% scheduler cycle 没有 eligible warp，barrier stall
占约 32%。这不是 EP4 的直接归因，也不是正式性能数据：NCU 报告的有效 SM
频率为 1.48 GHz，尽管目标及调用前后 NVML 为 1620 MHz，且多 pass 会扰动执行。
最初按单 kernel replay 的采样停滞，主动停止后确认 GPU 清空，改完整 graph
replay 才成功；不把这个 profiling 失败记成应用 kernel 失败。

`local_decode_hh_stats` 再把每 tile 的阶段计时原子操作缩减为每 CTA 每阶段的
首尾 extrema，统计含义不变。T1/16/32/48/64/96/128/256/512 依次为
62.769/115.542/130.742/151.056/159.502/166.642/171.863/180.692/373.585 µs，
9 个 bucket 的 baseline 数值对照通过，全部时钟采样 1620 MHz。此项只省约
1–2 µs，不支持把全局计时原子竞争当作主要原因。

稀疏版本已补齐 EP1/2/4/8 共 465 个 native 和 30 个 SGLang rank/case，全部
通过；包含逐位 batch-invariance、热点跨 M-block 和通用 V4 对齐。带首尾统计
版本的 EP1 memcheck 31 项应用检查通过、0 errors；racecheck 停在首个 case
后超过 9 分钟没有进展，已主动停止，不能记为通过。RL 尚未重跑。

### M64 紧凑 CTA 消融（尚未采用）

`local_decode_hh_compact` 保持 M128 的 GM 目录与物理行，只把 math 改为一个
warpgroup、M64 slice，N128/K64、3-stage、256 threads。超过 64 行的热点块
依次计算两片，不截断。FC2 等待该物理块的全部 FC1 slice，数值累加顺序不变。
动态 SMEM 从约 208 KiB 降至约 104 KiB，120 registers、0 spill；本地实际
token 数至少 4 时尝试双 CTA/SM，仍由现有 occupancy 门禁保证 grid 全驻留。

同机同配方 EP4，全部 NVML 样本 1620 MHz，单位 µs：

|全局 T|baseline|首尾统计版|M64/3-stage/双驻留|
|---:|---:|---:|---:|
|1|29.998|62.769|68.690|
|16|85.002|115.542|124.260|
|32|106.587|130.742|135.605|
|48|130.984|151.056|144.098|
|64|150.035|159.502|151.018|
|96|179.678|166.642|160.092|
|128|185.802|171.863|164.084|
|256|198.185|180.692|172.994|
|512|211.736|373.585|373.994|

紧凑版改善 48–256，但 1–32 退步，尚不足以解决 RL 常见 bucket 的差距。
这是 M、流水级数、线程数和驻留同时改变的组合消融，不能把结果只归因于
双驻留。EP1/2/4/8 的 465 个 native 和 30 个 SGLang rank/case 全部通过；
8 个通用 V4 ring entry 的 SASS 逐指令不变。EP1 memcheck/racecheck 各 31 个
应用检查通过，分别 0 errors / 0 hazards；EP2 两种 sanitizer 各 62 个应用
检查也已通过，两 rank 均为 0 errors / 0 hazards。
原始日志为 `build/local_decode_hh/compact_*`。下一项单独记录 N64/5-stage
和有效行直接 FC1 写回的结果，不覆盖本项数据。

`local_decode_hh_n64` 使用 M64/N64/K64、5-stage、双驻留和有效行直接 FC1
写回（保留同一 BF16 舍入 helper）。68 registers、0 spill。EP4 的 124 项
native 检查及 9 个 SGLang bucket 数值对照通过，全部 SM 样本 1620 MHz。
T1/16/32/48/64/96/128/256/512 的 p50 依次为
66.041/120.018/133.853/153.115/160.273/168.439/173.874/186.471/373.886 µs。
该组合未改善主流 bucket，暂不采用；尚未做全 EP 与 sanitizer 验收。

`local_decode_hh_n64x3` 将上一项改成 3-stage、三 CTA 驻留，T16–256 依次为
119.249/133.203/158.203/161.901/174.042/177.764/188.700 µs，未改善主要差距，
不采用。EP4 的 124 个 native 和 9 个 SGLang bucket 数值对照通过，全部
SM 样本 1620 MHz；不是完整验收。该轮已包含下面的补齐修复，T1 的改善
不能归因于三 CTA（T1 未触发增大 grid）。

### 补齐 token 不再引入额外专家

SGLang 复制输入按 rank 等长切片。T 小于 EP 时，空 source 原先以零 X/gate
搭配专家 0–7，仍会生成这些专家的 GEMM task。现在 graph staging 的补齐行
复用全局首个真实 token 的专家 ID，只改变无效行的 route，X/gate 仍为零。
所有真实 token 的 route、计算与输出不变；全局空输入仍由适配层原入口返回。
masked 真实 route 仍使用原清洗规则。

在同一个 M64/N128/3-stage 二进制上单独对照，EP4 全局 T1 从 68.690 降至
53.812 µs（baseline 29.858）；T16/32/48/64 分别为
124.281/135.577/144.101/150.932，与修复前持平，全部 SM 样本 1620 MHz。
另有 EP4/T1 的 4 个 SGLang rank/case 通过数值对照、masked route、图回放及
权重更新后重建。原始数据为 `compact_padded_sglang.log` 与
`padded_sglang_ep4_t1.log`。此项采用，但 T1 仍落后 baseline，不能据此宣称
端到端回退已经解决。

### 寄存器 A 的 WGMMA 消融（不采用）

`local_decode_hh_rs2` 为 M64/N64/K64、5-stage、双 CTA，权重 TMA 提前推进，
math acquire 行 ready 后直接读取有效 A 行到寄存器。寄存器 A 使用独立的
K16 fragments，先 fence，再发指令，等待该组完成后才重用寄存器；不能照搬
SS 版只退休旧 SMEM stage 的规则。参考
[PTX 的 WGMMA 同步合同](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)。

82 registers、0 spill；三 CTA 尝试被启动前 occupancy 检查拒绝，重新构建
双 CTA 后 EP4 124 个 native 和 9 个 SGLang bucket 数值对照通过，全部时钟
采样 1620 MHz。T1/16/32/48/64/96/128/256/512 依次为
59.315/148.292/166.925/199.323/206.488/219.724/229.348/254.091/373.955 µs。
显著退步，不采用；没有完整 sanitizer 验收。不以此推断所有 RS 设计都慢，
这里只测试了逐 K64 退休寄存器片段的实现。

后续 M16 候选先做最小累加对照：K512/K2048、输入幅度 0.03/1/20 的六组
FP32 GEMM 输出与原 WGMMA 逐位相等，见 `build/local_decode_hh/mma_probe.log`。
这只是数值可行性检查，不是完整 MoE 验收或性能结论。

### M16 warp-level MMA（继续消融）

`local_decode_hh_m16b3` 保留 M128 物理 block，按 M16/N128/K64 分片，四个
math warp 分别承担 N32。权重通过 3-stage TMA 装载，A 从有效行读取寄存器；
FC1 直接写有效 S 行，FC2 复用通用的 coalesced peer writeback。热点块依次
完成全部 M16 slice，ready target 包含每片，不截断 token。原三次 rank
collective、K16 累加顺序、激活和 combine 数值合同不变。

初版 109 registers；三 CTA 编译预算下为 80 registers、0 stack/local spill。
host 将请求 grid 限制在实际 occupancy 范围内。EP4 124 项 native（包含
逐位 batch-invariance、通用 V4 对齐）和 9 个 SGLang bucket 数值对照通过；
8 个通用 V4 ring entry 的 SASS 仍逐指令不变。

全部 SM 样本 1620 MHz，T1/16/32/48/64/96/128/256/512 的 p50 为
60.053/106.988/124.614/151.223/157.253/168.949/174.363/188.940/373.681 µs。
T32 比前面的 M64 紧凑版改善，但 48–256 没有整体胜出；尚不能作为最终版本。
原始日志 `build/local_decode_hh/m16b3_*`。下一项再测试 A 的 TMA 装载与
`ldmatrix`，不将两者的组合收益拆成未经独立测量的单项收益。

### M16 + A/B TMA + ldmatrix（端到端复测候选）

`local_decode_hh_m16tma` 保留上项 M16/N128/K64、3-stage、256 threads，
改为 X/S 和 W 均通过 TMA 装载，math 用 `ldmatrix` 读取 fragment。每个 stage
先发 W，首次 X/S 装载前 acquire 等待物理 block ready，随后 full barrier
发布 A/B；四个 math warp 读完后共同归还一个 CTA-local empty credit。
不再使用 decode 的 cluster multicast；通用 V4 的 multicast 不变。
80 registers、0 spill，实际驻留能力约束请求 grid。权重 map 与通用 N128
profile 相同，最终整理版复用已有 map，不保留实验中重复的 descriptor。

同机同输入 EP4，全部 NVML 样本 1620 MHz，单位 µs：

|全局 T|baseline|M16 + TMA|相对 baseline|
|---:|---:|---:|---:|
|1|29.884|49.579|+65.9%|
|16|85.085|107.595|+26.5%|
|32|106.546|125.066|+17.4%|
|48|130.747|144.377|+10.4%|
|64|150.110|151.068|+0.6%|
|96|179.667|159.549|−11.2%|
|128|185.608|164.698|−11.3%|
|256|197.817|175.034|−11.5%|
|512|211.630|374.116|+76.8%（通用路径未改）|

相对最初 HG 的真实适配层，T32 从 160.969 降至 125.066 µs（−22.3%），
T64 从 192.961 降至 151.068 µs（−21.7%）。这是多项优化的累计收益，
不是 M16 单项收益；小 batch 仍未全面追平 baseline。

EP1/2/4/8 共 465 个 native 和 30 个 SGLang rank/case 全部通过，包括热点
跨块、逐位 batch-invariance、通用 V4 对齐、masked route 与权重更新。
EP1/EP2 的 memcheck、racecheck 各 31/62 个应用检查通过，全部工具进程
为 0 errors / 0 hazards。日志 `build/local_decode_hh/m16tma_*`。
尚待最终整理版、原 E64/EP8 性能及同配方 RL 验收；不将微基准改善当作
端到端回退已解决。

E64/K4/EP8 复查使用相同输入与原 MegaMoE 对照脚本，T 为每 rank token 数。
M16 的 T1 为 39.563 µs，MegaMoE kernel 为 49.666 µs；T64 则为 67.774 µs，
落后 MegaMoE 的 58.962 µs。全部时钟样本 1620 MHz，所有数值检查通过。
说明按较稀疏 E256 调出的 M16 不能直接全量替代：T64 的平均每专家行数
达到 32，M16 重复遍历权重的代价增大。原始日志 `m16tma_mega_ep8_t*.log`。

第一次整理构建 `local_decode_hh_final` 曾错误地以 prefill FC2 N256 descriptor
检查 decode N128 兼容性，使 decode 入口始终不可选；对应 `final_mega_*` 测到
的是通用路径，已排除，不作为 M16 或最终性能。修正为原短文 W2 N128 map。

后续 `local_decode_hh_dual` 复用同一 geometry、pull、producer、epilogue 与
rank 同步，只模板化 M16 warp MMA / M64 WGMMA 的算术片和 SMEM 布局。
按 `local_tokens * world * top_k >= experts * 16` 选择 M64，其余使用 M16；
这是静态形状的密度估计，不是丢弃热点或动态任务领用。各 rank 可以使用
不同算术形状，但物理 M128 block 与跨 rank 协议完全相同，必须验证跨形状
逐位一致与不等长 source。此候选尚在构建验收，未运行 RL。

双形状首轮 E64/EP8 T1/T64 为 39.494/61.530 µs，MegaMoE kernel 为
49.500/59.082 µs，含输入 staging 为 53.094/64.948 µs；全部时钟 1620 MHz。
E256/EP4 的全局 T1/16/32/48/64/96/128/256/512 为
49.654/107.436/125.260/144.681/150.803/159.290/164.907/174.934/373.451 µs，
保持 M16 原收益，9 bucket 数值对照通过。8 个通用 V4 ring entry 的 SASS
仍逐指令不变，日志 `dual_*`。

随后 `local_decode_hh_release` 只调整静态 grid 上限：以每专家一个物理块的
tile 数估计可用任务量，将请求驻留倍数限制为
`clamp(owned_experts * (FC1_tiles + FC2_tiles) / SMs, 1, shape_limit)`。
M16/M64 的 shape limit 为 3/2，且仍受实际 occupancy 上限约束；实际热点
始终 grid-stride 遍历所有块。E64/EP8 因专家少改回一 CTA/SM，E256/EP4 不变。
这项尚待单独计时，不把资源估计当作已取得的收益。

最终 grid 轮 E64/EP8 T1/T64 为 **39.473/59.223 µs**，同轮 MegaMoE kernel
为 **49.418/59.917 µs**，含 staging 为 **52.954/65.697 µs**。T64 相对双驻留
轮省 2.307 µs；最终与 MegaMoE 的 kernel 接近，不能把约 1% 的领先夸大为
显著收益。E256/EP4 的 T1/16/32/48/64/96/128/256/512 依次为
49.628/107.740/125.245/145.242/150.888/159.698/164.889/175.076/373.556 µs。
正式计时没有并行编译或其它 GPU workload，全部 NVML 样本 1620 MHz。

双形状已通过 EP1/2/4/8 的 465 个 native 和 30 个 SGLang rank/case；另以
E64/K8 的 EP1/EP2 覆盖稠密 M64、跨形状切换，两种 sanitizer 各 31/62 项
应用检查通过，全部进程为 0 errors / 0 hazards。最终 grid 二进制的 M16、
M64 以及 8 个通用 V4 ring entry 均与该已验收双形状二进制逐指令一致；
改变的是 host grid 选择，不是设备端同步协议。最终二进制额外通过 E64/EP8
短文 248 项、1/64/8K/20K 长容量及热点复用 240 项，以及 E256/EP4 的
124 项 native 和 9 个 SGLang bucket 对照。日志为 `dual_dense_*` 和 `release_*`。

冻结 native library：`build/local_decode_hh_release/libweftmoe.so`，SHA256
`b7ea21bdf989b30b5cda1cbe7d813ad2de70bc3cae4ec087375511c3a7c5d7be`。
最终仍只在单机短容量 inference pool 内选 M16/M64；prefill、training 和
跨节点协议未改。小 batch 对 SGLang baseline 的剩余差距不能隐去，实际 RL
收益另以相同步数和接受的生成 token 吞吐复测。

### 单 pull warp 消融（不采用）

`local_decode_hh_onepull` 仅将 M16 的三个 pull warp 减为一个，CTA 从 256
降为 192 threads，释放 8 KiB SMEM；M64 不变。E256/EP4 的 124 个 native
rank/case 和 9 个 SGLang bucket 检查通过。为避免节点差异，同时在本机
重测冻结的 HHD；全部 SM 采样为 1620 MHz，无其它 GPU workload 或并行编译。

|全局 T|HHD 本机复测（µs）|单 pull warp（µs）|
|---:|---:|---:|
|1|49.334|49.078|
|16|107.242|107.198|
|32|125.215|125.009|
|48|144.577|145.595|
|64|150.227|150.833|
|96|159.561|160.030|
|128|164.915|165.444|
|256|174.983|173.366|

主流小 batch 没有稳定改善，不采用，代码已恢复。日志为
`build/local_decode_hh/{onepull_local_sglang,release_local_control}.log`；
没有将本项扩展到全 EP/sanitizer 验收，也没有替换正在运行的 HHD RL。

### HHE：单窗口 decode 省去末尾 rank barrier

`local_decode_hh_twobarrier` 保留原 M16/M64、三条 pull warp 和静态任务分配，
仅从专属 decode 移除 AfterClean。适用前提是同一 pool 的调用按 stream
串行、全部 payload 在一个容量窗口内，且末尾没有通用 ring 的计数器清理。

复用边界如下，不能把这一证明推广到需要在线复用的长文/网络 ring：

1. BeforeCombine 收齐所有 rank 的 FC2 写回，此时上一调用的 source X、route
   目录、canonical X/S 均已没有读者；下一调用可提前重写这些区域。
2. Reduce 只读取本 rank 的 contribution，不再读取 peer 的 source/目录。
   下一调用可以先规划，但任何新 FC2 写入都必须经过下一次 BeforePull。
3. 每个 rank 只有上一 kernel 完整退出后才能进入下一次 BeforePull，因此该
   barrier 同时确认所有旧 Reduce 已结束，新 FC2 不会覆盖尚在读取的贡献。
   保留双 phase add/sub rank 信号；快速 rank 不能越过下一次 BeforePull。

末尾只更新状态，`complete_clock` 同时覆盖活动 Reduce CTA 的完成时间；
不存在的 AfterClean 计时为零。不改变一般 V4 的 `after_combine` helper。

同机首轮全部 NVML 样本 1620 MHz，无并行编译/其它 GPU workload。全局
T1/16/32/48/64/96/128/256 的 p50 为
47.826/105.456/123.518/143.290/148.871/158.046/163.250/174.328 µs；
相邻 HHD 控制组为
49.334/107.242/125.215/144.577/150.227/159.561/164.915/174.983 µs。
T512 的通用路径为 372.822 µs，控制组 372.808 µs。主要 bucket 改善
约 1–2 µs，不夸大为大幅收益。

首轮 E256/EP4 的 124 个 native rank/case、9 个 SGLang bucket 通过；8 个
通用 V4 ring entry 的 SASS 与 HHD 逐指令一致。完整 EP1/2/4/8、sanitizer、
E64/EP8 与再次复测的结果如下；未替换正在运行的 HHD RL 二进制。

EP1/2/4/8 共 465 个 native、30 个 SGLang rank/case 通过。E64/K8 的
EP1/EP2 memcheck、racecheck 分别覆盖 M16/M64 及跨形状输入，两种工具
各 31/62 个应用检查通过，全部工具进程 0 errors / 0 hazards。
另在 EP4 的既有图复用检查中为末 rank 加入图内延迟，两个相依调用之间
其它 rank 可先进入下一轮；124 个 native rank/case 全部通过，未新增永久
测试入口。日志为 `twobarrier_skew_ep4.log`，诊断 shim 为 `skew_check.py`。

正式计时复测 T1/16/32/48/64/96/128/256 为
47.816/105.681/123.524/143.055/148.877/158.131/163.208/174.596 µs，
重现约 1–2 µs 的收益；T512 为 372.620 µs。全部 NVML 样本 1620 MHz。
E64/K4/EP8 的 T1/T64 为 36.956/59.360 µs，同轮 MegaMoE kernel 为
49.587/59.998 µs，含输入 staging 为 53.062/65.876 µs；全部数值检查和
1620 MHz 时钟检查通过。T64 仍只可称接近 MegaMoE kernel，不把约 1%
差距称为显著领先，也不把不同物理节点测量的细小变化归因于 barrier。

冻结 library 为 `build/local_decode_hh_twobarrier/libweftmoe.so`，SHA256
`962daf6fa6486f7afc930934ce636d2a7053e6227742e09356101ab6f3adf421`。

### 物理块优先的 FC1/FC2 顺序消融（不采用）

`local_decode_hh_blockorder` 将静态序列改为每个物理块先 8 个 FC1 tile、
再 16 个 FC2 tile，使较早完成的块可以提前进入 FC2，仍由相同的 grid-stride
owner 执行。相应地，phase 计时改为 CTA 私有 extrema，最终聚合一次；
以下是这个组合的结果，不把计时改动的影响单独归因于任务顺序。

EP4 的 124 个 native 和 9 个 SGLang bucket 数值检查通过，全部时钟采样
1620 MHz。T1/16/32/48/64/96/128/256 的 p50 为
48.584/116.645/140.277/166.622/169.528/176.422/182.185/192.582 µs，
相对 HHE 的 T32/T64 回退约 13.6%/13.9%。该顺序降低了初始可执行 FC1
的 CTA 比例；没有观察到提早安排 FC2 的净收益。编译资源为 M16 80 registers /
16-byte stack，M64 128 registers / 32-byte stack。未进一步做全 EP 或
sanitizer，代码已恢复 HHE，不替换 RL。原始日志为 `blockorder_*`。

## 8. E256/K8 小 batch：baseline / MegaMoE / WeftMoE 独立对照

2026-09-30，仅扩展既有 `benchmarks/torchrun_sglang_decode.py` 的可选
MegaMoE 对照，不修改算子、SGLang 生产适配层或 DeepGEMM。结论是：当前
WeftMoE HHE 在此次全部 EP/batch 点均领先冻结的 BF16 MegaMoE，但很小
batch 仍未追上 SGLang baseline，不能以领先 MegaMoE 代替 RL 收益验收。

### 配置与统计口径

- 单机 `10.10.16.219`，NVIDIA L20X，132 SM/GPU；分别使用 EP2/4/8，TP=EP、
  expert TP=1，无 DP attention。SM 锁定 1620 MHz，所有测量前后、所有 rank
  的 NVML 采样均为 1620；没有其它 GPU workload 或并行编译。
- Qwen3.5-35B-A3B 的 routed MoE 形状：E256/K8/H2048/I512，BF16、SwiGLU。
  **T 是此层的全局 batch**，每个 TP rank 持有同一份输入；WeftMoE/MegaMoE
  各自处理 `ceil(T/EP)` 个 source rows，例如 EP4/T32 是每 rank 8 行。
  baseline 的每个 rank 持有 E/EP 个完整专家，不是把 I512 切成 I128。
- baseline 为已安装 SGLang 0.5.10.post1 的 Triton FusedMoE，使用其默认
  配置选择；本机无此设备/形状的专用 tuning JSON，沿用 RL 的 baseline。
  WeftMoE 为 `07a586d` 的 HHE，而不是此前 RL 使用的 HHD。
- 同一 EP 内三者使用完全相同的随机权重、输入、top-k IDs 和 weights。
  权重 seed 为 `37 + rank`，输入/route seed 为 `100 + T`；不同 EP 的全局
  权重不完全相同，因此 EP2/4/8 表不当作同一权重的严格强扩展实验。
- 测量包含完整 routed MoE 和相同的外层 TP all-reduce；WeftMoE/MegaMoE
  还包含同一套 fused staging、输出 scatter，以及各自内部 dispatch、
  FC1/FC2、return/combine。补齐行 X/gate 为零，复用首个真实 token 的
  expert IDs，避免引入额外专家；MegaMoE 此处没有改用自身的 -1 padding。
- 不含 router/top-k、共享专家、attention、权重打包、JIT、分配和 graph
  capture。这是 warmed 单层 CUDA Graph 微基准，不是完整模型/RL，亦非
  40 层权重冷工作集测试；不能直接据此预测 rollout 吞吐。
- 每点先丢弃 5 轮，再测 30 轮，每轮每个 backend 连续 replay 50 次；
  每轮轮换三者测量顺序。CUDA event 计时，先取同一轮的 rank-max，再取
  30 轮 p50。EP4 额外按 T128→1 反序完整复测。
- 40 个 EP/batch 测量点、三者全部数值对照通过（`rtol=0.025, atol=2e-5`），
  相对 baseline 的最大绝对误差为 `7.62939453125e-6`。此项是容差检查，
  不宣称三种实现彼此 bitwise 相同，也不替代既有 Weft batch-invariance 验收。

### EP4 主表（与 RL 并行配置一致）

单位 µs，越低越好。相对变化为 `Weft / 对照 - 1`，负数表示 Weft 延迟更低。

|全局 T|baseline|MegaMoE|WeftMoE HHE|相对 baseline|相对 MegaMoE|
|---:|---:|---:|---:|---:|---:|
|1|29.825|63.712|47.269|+58.5%|-25.8%|
|2|32.727|64.244|48.327|+47.7%|-24.8%|
|4|51.094|81.329|59.738|+16.9%|-26.5%|
|8|55.649|84.710|65.537|+17.8%|-22.6%|
|16|83.596|143.279|104.855|+25.4%|-26.8%|
|32|106.241|162.474|123.420|+16.2%|-24.0%|
|48|131.983|181.508|143.462|+8.7%|-21.0%|
|64|150.177|190.054|149.761|-0.3%|-21.2%|
|96|179.786|200.523|159.576|-11.2%|-20.4%|
|128|186.565|202.729|163.707|-12.3%|-19.2%|

反序复测相对首轮的最大绝对 p50 变化：baseline 0.58%、Weft 1.51%、MegaMoE
0.31%。T64 复测为 baseline 149.853、Weft 149.855、MegaMoE 190.019 µs，
应称 Weft 与 baseline 持平，不将首轮 0.3% 差异解释为收益。
T1–48 的 baseline 优势、T96/128 的 Weft 优势均复现。

### EP2 / EP8 补充

|全局 T|EP2 baseline|EP2 MegaMoE|EP2 Weft|EP8 baseline|EP8 MegaMoE|EP8 Weft|
|---:|---:|---:|---:|---:|---:|---:|
|1|32.547|81.753|51.778|30.866|60.562|47.004|
|2|45.283|90.294|60.805|31.799|62.000|48.344|
|4|59.837|117.326|83.069|46.250|66.788|52.783|
|8|78.679|139.011|97.631|49.508|71.868|55.472|
|16|136.016|195.188|143.450|64.879|99.876|78.516|
|32|170.843|266.623|188.816|79.214|114.917|93.439|
|48|205.284|290.385|226.286|104.405|123.579|103.769|
|64|224.215|314.637|236.582|114.290|131.571|111.613|
|96|238.184|329.994|254.100|122.084|133.554|115.203|
|128|242.969|336.960|259.496|116.688|133.788|117.082|

EP2：Weft 比 MegaMoE 低 22.1%–36.7%，但所有所测 batch 均慢于 baseline。
EP8：Weft 比 MegaMoE 低 12.5%–22.8%；T1–32 慢于 baseline，T48/128 接近，
T64/96 此轮分别低 2.3%/5.6%。EP2/8 未做第二轮反序，不夸大小幅差异。

三者并非相同通信工作流：此处 replicated 输入已经在所有 rank 可用，
baseline 可直接筛选本地专家，最后 all-reduce；Weft/Mega 的兼容路径先
分 source 行，再执行自身 dispatch/return，最后还原并 all-reduce。
因此不能将 Weft 对 baseline 的全部差距归因于 GEMM，也不能把这一口径的
结果推广到 DP-sharded 输入。各部分占比需要另行 profile，本表没有拆分归因。

### 复现与冻结产物

环境：Torch 2.9.1+cu129、CUDA 12.9、NCCL 2.31.2、driver 570.148.08。
`build/decode_threeway_20260930/run.sh` 顺序运行 EP4、EP2、EP8、EP4 反序；
正式原始记录分别为同目录下的 `ep4.log`、`ep2.log`、`ep8.log`、
`ep4_reverse.log`。首轮 smoke 的包导入顺序错误已修复，两个 smoke 日志均
不参与上述统计。正式四次 torchrun 均正常退出，GPU 显存已释放。

EP4 命令（EP2/8 只改进程数；反序只改 `--tokens` 顺序）：

```bash
export PATH=/opt/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 NCCL_DEBUG=WARN
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0
export WEFTMOE_STATIC_RING_TOKENS=512 WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS=8
export WEFTMOE_TENSOR_EXTENSION=$PWD/build/local_decode_hh_twobarrier/_runtime.cpython-312-x86_64-linux-gnu.so
export DG_JIT_CACHE_DIR=$PWD/build/static_v4_megamoe_decode_20260929/jit
nvidia-smi -lgc 1620,1620
torchrun --standalone --nproc_per_node=4 benchmarks/torchrun_sglang_decode.py \
  --deepgemm-package build/deepgemm_sglang_torch291/lib.linux-x86_64-cpython-312 \
  --tokens 1 2 4 8 16 32 48 64 96 128 --samples 30
```

Weft library 路径/hash 同 HHE 节。MegaMoE 采用前轮已冻结的本地 BF16
DeepGEMM 包（支持 E256 的分段专家扫描），不是另取上游新版；`set_num_sms(132)`，
`fast_math=True`、无 activation clamp。请求的 source capacity 为 64/rank，
库实际对齐为 384/rank；实际工作行数仍由 output 的 `ceil(T/EP)` 决定。
冻结包的 SHA256：

- `deep_gemm/_C.cpython-312-x86_64-linux-gnu.so`：
  `100a06dd12331d178a8c8590da4703008598dcb9625bf7769fa69863029394ed`。
- `deep_gemm/include/deep_gemm/impls/sm90_bf16_mega_moe.cuh`：
  `fb636714d7e7703b9a480fa7a9b57dec29b79d833c9b55dd3140a7c859082a77`。
