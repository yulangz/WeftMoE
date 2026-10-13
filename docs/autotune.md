# Forward / Backward autotune

Qwen3.5-35B-A3B（H2048/I512/E256/top8）的 EP8/EP16 实测专属 profile、
查表文件和复现命令见
[`configs/pf_qwen35_20261013`](../configs/pf_qwen35_20261013/README.md)。
搜索配置为 `configs/pf_qwen35_training_autotune.json`，不是把 Flash 的
实测参数直接标成 Qwen 最优值。

M14 的数据流候选见 `configs/pf_qwen35_compact_20261013`，消融与最终
验证状态以 `docs/pf_backward_20261012.md` 为准。新增 V4 搜索轴包括
local pull warp 数/分块方式、紧凑 contribution、有序 dispatch/连续 Get、
window-ordered tape、BWD return pending/span、分散完成计数器
（`backward_striped_signals`）和 full-CTA reduce。
默认关闭新策略，不改 Flash profile。`network_credit_windows` 控制真正的
RDMA ring 深度倍率，进入编译 ID；`network_min_depth` 允许 1/2/4 等 floor，
但仍满足最坏窗口的容量下界。环境变量 `WEFTMOE_PF_CREDIT_WINDOWS` /
`WEFTMOE_PF_MIN_RING_DEPTH` 是显式覆盖；搜索这些轴时须从 `environment`
移除固定覆盖。`window_tokens` / `contribution_windows` 仍是运行时容量项；
network PF 使用全调用 contribution，不能把 D-only 候选当成有效 wire tune。
分散 BWD 完成计数器需要 `network_lanes >= 112 - network_get_workers`；
生成配置和 CUDA 编译同时检查此约束，避免较小 channel 配置越界。

## 当前 PF stream BWD 的 GEMM 搜索（2026-10-12）

`configs/pf_backward_stream_autotune.json` 用于新 BWD：固定 FWD 和通信协议，
只搜索实际生效的 GEMM 参数。沿用下面的 `tools/autotune_training.py`
命令，将 `--config` 换为本配置即可；shape 和节点仍由 CLI 传入。

- dU/dX 的 N tile 可分别取 128/256；二者共用 operand ring，stage 数
  联合比较 2/3/4。本轮 Flash 的 M128、K64 和 cluster N2 保持不变；
  stream 已支持 K128，但当前搜索配置未加入这项；此前 K128/两级流水
  没有确认跨规模收益，因此仍以 K64 为默认。
  更换 H/I 时还需确认 `I % (2*DuN) == 0`、`H % (2*DxN) == 0`，
  不满足的候选在资源检查时过滤；N256 尾部不足一个 cluster 时选择
  N128。当前只保留 cluster-N2 stream，不再回退到旧 BWD 实现。
- dW1/dW2 共用 M/N 几何和 stage 数，比较 M128/N256、M256/N128、
  M128/N128，以及两级/三级流水。四级流水也提出候选，但 M128/N256
  的 shared memory 超限会被资源检查过滤。
- 静态轮转的 chunk 粒度分别由 `backward.dw1_claim_tiles` 和
  `backward.dw2_claim_tiles` 控制，单位是 cluster tile-pair，默认 4/1。
  两者均支持 1/2/4/8/16。2026-10-13 修复了 C++ 仍拒绝 2、而 Python
  已允许 2 的不一致；搜索配置中的 2/2、4/2 现在可以正常编译。
  本配置比较有限的组合，若需要笛卡尔搜索，可使用已有 `axes` 字段；
  并非将 4/1 视为所有模型的最优解。
- 不搜索旧 `dx_claim_tiles` / `return_window_scale`，它们不控制当前
  canonical stream 的 dX/return；下面旧记录不能当作新实现的调参结论。
- `environment` 显式固定工作区环境参数，逐节点传入，避免改变 buffer
  容量后把通信收益误认为 GEMM 收益。配置仍不修改时钟。

2026-10-13 清理后，编译资源检查直接识别当前 PF stream 的单/双机入口，
不再检查已删除的 unified/split kernel。旧格式 BWD grid 保留生成头文件的
兼容字段，但不再搜索已失效的 `dispatch_batch_rows`；新实验优先使用
上面的 `pf_backward_stream_autotune.json`。

保留基线，与候选使用相同变化长度/热点路由 trace，初筛后正反顺序确认，
数值通过才导出训练表。候选编译日志包含 ptxas 寄存器/spill 报告；
不能只按静态 spill 字节数排序，需检查执行分支和完整周期性能。
该配置是有限预算的单参数/耦合组合筛选，不是全部超参的笛卡尔穷举，
也尚未把 dW1/dW2 的 tile/stages 拆成四套完全独立的 GEMM 配置。

### 独立 FWD / 缓存就绪后的 BWD 测量

`tools/benchmark_training.py --forward-only` 的
`forward_ready_slowest_rank_p50_us` 包含带 X/GU 缓存的 FWD 及完成同步，
计时结束时 tape 仍存活；默认包括每调用权重重排。

`--backward-only` 每次在计时外执行一次新 FWD、同步所有 rank，再执行
BWD。`backward_ready_slowest_rank_p50_us` 包含 BWD 前置、临时分配、计算、
通信与完成同步，以及 native BWD 内部释放 forward/dispatch handle 的
开销；计时结束时输出梯度仍存活。不是复用旧路由/tape，也不是
用完整周期减去 FWD host API 时间。两模式互斥，均另报包含外部释放的
`wall_slowest_rank_p50_us`。分开运行的两项 p50 不能直接拼成历史完整周期。

## PF 动态训练 FWD+BWD 搜索（2026-10-04）

本节保留当时的流程与消融记录。CLI 的完整周期计时口径仍适用；旧
bounded-return 实现及其 `dx_claim_tiles` / `return_window_scale` 优化已于
2026-10-13 删除，不是当前 PF stream 的有效搜索轴。

`tools/autotune_training.py` 使用 `configs/pf_training_autotune.json`，按传入
H/I/E/top-k、EP 拓扑和 token 规模自动生成组合头文件、编译、计时、校验、选优。
这条入口测 **完整 eager FWD+BWD+释放+device synchronize**，不能和下面
推理 CUDA Graph 的选优分数互换。输入生成、首次编译、固定池准备/IPC 建连
不计时；每个候选都先完整回放一次动态 trace，再 warmup 和采样。

```bash
python tools/autotune_training.py \
  --config configs/pf_training_autotune.json \
  --hidden 4096 --intermediate 2048 --experts 256 --top-k 6 \
  --tokens 8192 20480 --gpus-per-node 8 --jobs 4 \
  --output build/tune-flash-ep8

# 相同配置用于多机；远端要求相同 Python、共享源码和编译缓存路径。
python tools/autotune_training.py \
  --hidden 4096 --intermediate 2048 --experts 256 --top-k 6 \
  --tokens 8192 20480 --gpus-per-node 8 \
  --remotes root@HOST1 --master-addr LOCAL_IP --ssh-port 2222 \
  --output build/tune-flash-ep16
```

换模型只改 shape 参数；`--remotes` 可以列多台同构节点，不把 EP16 写进
搜索逻辑。GPU 必须空闲，并事先按实验要求固定目标频率；工具不擅自改变
时钟/功耗。编译依赖沿用 `build_native.py` 的 `WEFTMOE_ROOT`、CUDA、
NCCL 等环境设置，运行将 NCCL/UCX/CUDA 和动态库路径传给远端。
`--dry-run` 只展开配置并做资源检查，不编译或占用 GPU。

配置分为：

- `base_header`：当前冻结 profile，预处理展开 FWD/BWD 默认值后，显式注入
  新 H/I；不会把旧模型尺寸留在 BWD。`v4` 声明通信几何。
- `ring.local/network`：B/D 配对候选。它们只影响 prepare，**不增加编译数**。
- `forward/backward`：`fixed` 为分组公共覆盖；`variants` 为耦合的 tile/stage
  组合，`axes` 为可选笛卡尔搜索轴，例如 `"backward.dw_tile_m": [128, 256]`。
  使用现有参数名、资源约束和显存检查；不合法资源组合在编译前淘汰。
- `stages`：默认 ring → FWD → BWD → BWD Return，每步携带上一轮胜者，保留 incumbent。
  可再加 `ring` 做最后的 B/D 联合回查。这是有预算的分阶段搜索，报告的是
  **本次搜索空间内实测胜者**，不声称全局最优。

`backward.dx_claim_tiles` 控制 bounded Return 路径中，一次 dAB gather
复用给多少个相邻 dX N-cluster tile，可选 1/2/4/8/16，默认 2。
默认训练搜索比较2/4/8。它只改变静态任务的批量粒度，不切换 GEMM、
通信协议或累加次序；canonical dX 不受此参数影响。较大 batch 减少重复
gather，较小 batch 保留更多 N 方向并行度，不能假定总是选最大值。

`backward.return_window_scale` 在原有 BWD contribution 字节预算内，将
bounded dX 聚行窗口设为原来的 1 或 2 倍（默认 1），相应减少在途窗口数。
它不放大 FWD X/S/C，也不增加预算来避开 ring。较粗窗口减少专家尾块，
但可能加重 credit 等待；同时受 dX slot 容量约束，确保一个最坏窗口能够
完整驻留，避免跨 M-block packet 的循环等待。候选仍需数值、稳定性和
反序性能确认。默认训练搜索联合比较倍率1/2；Flash单机选出batch4/scale2，
双机选出batch8/scale2，20K的配对完整周期分别降低6.51%/7.50%，8K基本持平。
两者通过大规模数值、极小ring/热点路由及长回放；不把这个结果硬编码成
shape或拓扑分支。失败现场、验收范围及原始记录见
[训练记录](static_forward_v4_training_record.md)。

编译复用现有 native object cache（包含命令、源码/头文件依赖和编译器），
并在总 `--jobs` 预算内并行准备唯一编译配置。计时期间不编译、不并跑候选。
组合 binary 的 tuning ID 同时包含 FWD、BWD、通信参数，避免 BWD 换了而
查表仍误认为同一配置。生成的 profile 独立保存，不替换 DT/其他模型的库。

候选使用相同 seed 的正态 token 波动和带热点偏置的随机路由，校对各 rank
trace hash。分数为各 token 档 rank-max p50 的等权几何平均；原始分项全部
保留。前两名（含 incumbent）正反顺序复测，默认提升至少 0.5% 才切换。
入围者验证 output/dX/dGate/dW1/dW2，最终胜者再验证全部目标 token 档；
Torch reference 每次只保留一个本地专家的计算图，跨 rank FP32 合并结果，
不复制全局权重。共同门限为 rtol=0.02、atol=1e-4、relative L2 < 1%。
这是 BF16 数值校验，不是 batch-invariance 的逐位证明。

结果在 `result.json`，各编译、计时和校验日志保留。成功后才生成 `table.json`：
`training=true`，key 包含实际准备的 `max_tokens`（不是 nominal tokens）。
配合选中 `_runtime*.so` 和 `WEFTMOE_AUTOTUNE_TABLE` 使用；不同 trace 如果
改变 prepared capacity，需要相应的新表项，不能把一个样本的胜者推广到任意
容量。表只查参数，不在训练进程中自动跑搜索，也不在同一进程热切换 .so。
`workspace_limit_bytes` 只约束 native 持久池（compute+GIN+notify），不含
权重、按实际路由保存的训练激活或 Torch allocator reserved。

确认阶段任何一次失败都会取消该候选的资格，不把已经完成的单轮成绩当成
完整确认结果。Flash EP16 的 B2304/D2 曾在重复确认中出现 GPU 持续等待，
终止后两节点正常释放；根因尚未定位，默认搜索空间暂不包含这一组合。
此前成功的初筛/首轮确认不能替代稳定性验证，失败日志也保留在实验记录中。

## V4 inference 离线搜索（2026-09-30）

`tools/autotune_forward.py` 复用原有内容寻址 JIT、资源过滤、正反顺序复测和
最终验证，新增两份 V4 配置。下面的旧默认配置与历史参数说明不代表 V4 的搜索空间。

- `configs/forward_v4_decode_autotune.json`：单机 EP1/2/4/8，算术 M 为
  `0`（现有密度规则）、16、64；联合搜索 2/3 stages 和 CTA multiplier。
  N128/K64 不变，物理 block 仍为 M128。入口要求 pool 最大 tokens ≤64、
  总专家数 ≤256、一窗口覆盖全部最坏路由；不满足时拒绝该搜索，而非测到回退内核。
- `configs/forward_v4_ring_autotune.json`：单机/多机通用 ring，联合搜索
  FC1/FC2 N、K、stages、B 和 Dispatch Q。还可在 `static_v4.search_space`
  中增加 Return Q、network lanes/worker cap/min depth、source group、
  contribution windows、CTA count。单机去掉无效的网络维度。物理/算术 M
  暂为 128，网络 M16/M64 尚未实现，不能把 local decode 参数用于网络。

两份配置通过 `base_config: auto` 继承对应拓扑旧配置的第一项，再显式覆盖
V4 参数；不是展开旧版与 V4 的全部无关参数。第一组合法候选为 incumbent。
ring 搜索的起点使用当前长文计算几何 N128/N256、K128、3/2 stages，
B512/D8；网络为32 lanes、Dispatch Q8、Return Q16。B256 和 Dispatch Q16
作为候选，其他 N/K/stages 继续参与搜索。这是参数对齐，不表示生成的 binary
已与冻结 GA+FZ 二进制等价或性能相同，仍需端到端对照。
默认配置是待测候选，不是新测得的最优表。不同 GPU 修改配置中的 GPU 名称与
互联标识；默认频率 1620 MHz，每轮 suite 前后检查各节点 SM 时钟，工具不修改时钟。

当前ring搜索是**PF专用profile**：两项prefill阈值均为0，直接使用候选的通用
静态ring GEMM，关闭额外prefill模板。这也使`weftmoe_dt::eligible`返回false，
因此它的小token用例验证的是PF兼容性，不是DT性能。不能用选中的PF库直接替换
同时服务decode/prefill的生产库，也不能仅合并两张JSON表实现同进程跨编译
profile切换。混合部署需另行接通兼容的编译入口并回归DT；本轮保持原生产库不变。

示例（在空闲 GPU 上执行）：

```bash
# 单机 DP4+EP4，64 tokens/source
python tools/autotune_forward.py \
  --config configs/forward_v4_decode_autotune.json \
  --nodes 1 --gpus-per-node 4 \
  --hidden 2048 --intermediate 512 --experts 256 --top-k 8 \
  --tokens 64 --exact-tokens --jobs 2 \
  --output build/tune-decode.json --export-table build/v4-table.json

# 双机 EP16，8192 tokens/source
python tools/autotune_forward.py \
  --config configs/forward_v4_ring_autotune.json \
  --nodes 2 --gpus-per-node 8 --remote root@HOST --port 2222 \
  --hidden 2048 --intermediate 512 --experts 64 --top-k 4 \
  --tokens 8192 --exact-tokens --jobs 2 \
  --output build/tune-network.json --export-table build/v4-table.json
```

可先用 `--generate-only` 检查候选，用 `--build-only --candidate incumbent`
仅编译。单机 EP8 使用 `--nodes 1 --gpus-per-node 8`；四规模分别传
`--tokens 1/64/8192/20480 --exact-tokens`，不是一次把所有规模混成平均分。
当前性能 launcher 验证入口限单机或双机八卡；更多节点可生成/编译候选，
不宣称已在更多节点执行自动搜索。多机沿用既有 launcher 的 NCCL 路径和
远端 binary 可见性要求，按环境传 `--nccl-lib-dir` 等设置。

选择门禁：

1. 每个候选运行完整 CUDA Graph（Notify/Dispatch/FC1/FC2/Return/Reduce），
   平衡、热点、不等长三组输入分别计时。分数为各组 **rank-max 均值的等权平均**，
   原始分项和逐 rank 样本保留，不用纯 GEMM 时间代替。
2. 每个路由/seed 增加 tokens=1 的 `validate_only` 用例，比较相同首 token 的
   完整 BF16 输出 hash；不参与性能分数。每个完整输出、输入/权重 hash 还须
   与 incumbent 同 fixture 逐位一致。不等长当前为奇数 rank 半 batch（至少1），
   空 rank、完整跨桶 batch-invariance、权重更新和 sanitizer 仍需发布前独立回归。
3. workspace `device_bytes` 不超过显式 `memory_limit_bytes`；这是 pool 内存，
   不包含模型权重和框架其他显存。超限/失败候选不可入选；该检查发生在实际准备后，
   不替代 GPU 空闲内存检查。ring 最坏窗口容量下限不因搜索减小。
4. incumbent 必须成功；数值不一致的候选排除。保留 incumbent 和最快候选进行
   交错确认，默认收益至少 0.5% 才切换，最终复验成功才允许 `--export-table`。
   这些 checks 不是相对 PyTorch 的独立数值证明；新增算术模板仍需原有数值门禁。

每轮 route suite 在一次 rank launcher 中顺序执行，共用 CUDA context 与 NCCL
communicator 与不可变模型权重；不同候选仍独立启动。同容量路由连续测量并复用
pool，切换容量时释放旧池，始终按该用例的实际容量准备，不同时缓存多个 arena。
各用例重建输入/Graph，保留原有 warmup/sample 数，完成并释放 Graph/output 后
才复用；原始结果恢复到配置声明的顺序，不把准备时间计入 kernel 分数。
`WEFTMOE_TUNE_CASES` 仅是测试驱动的批量用例协议，不影响生产算子。
所有 case 的原始逐 rank 结果保存在同一个 `.tune.log` / `.confirmN.log` 中，
完整输出、batch-invariance、内存预算及选优规则不变；旧的逐 case 日志仍保留。

`window_tokens`/`contribution_windows` 为 prepare 参数，同一编译配置只编译一次；
改变它们仍单独测量、记录候选和内存。其他参数写入生成 header。新配置未指定时，
decode 的 M/CTA 密度规则、3 stages 及 ring 协议默认值保持不变。shared-memory
模型按实际 FC1/FC2 分时 arena 取最大值，另校验 epilogue 不覆盖 multicast A。

### 使用结果

成功后生成 `*-selected/weftmoe_forward_tuning_generated.hpp`，表按
GPU/互联、H/I/E/top-k、BF16、节点数/每节点卡数、DP 输入、inference、精确
`max_tokens` 匹配。它不是跨 workload 猜测最优值，也不是运行时编译器。

先用 selected header 构建 native 库（将路径换成实际结果目录）：

```bash
NVCCFLAGS="-I$PWD/build/tune-decode-selected -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1" \
CXXFLAGS="-I$PWD/build/tune-decode-selected -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1" \
python build_native.py lib --output build/tuned/libweftmoe.so --jobs 2
python tools/build_tensor_runtime.py --library build/tuned/libweftmoe.so
```

将 `WEFTMOE_TENSOR_EXTENSION` 指向上述新 adapter，并设置
`WEFTMOE_AUTOTUNE_TABLE=/absolute/path/v4-table.json`。`TensorRuntime` 在
native pool 准备前检查精确键、编译配置 ID 和 EP 全组一致性，再显式传入 B/D。
未设置表时保留原行为；显式设置表但缺项或 native ID 不符会报错，避免将不兼容
协议当成“fallback”。正常执行/graph replay 不读表、不试跑、不新增动态任务队列。

一个 native 库仍只编译一组 N/K/stages/通信参数。**本实现不在同一进程里自动热切换
不同编译 profile，也不为每个 graph 动态加载一个库。** 多个容量的表项要共用该库，
必须具有相同 compiled ID；不同 profile 需重建/重启并重新准备 pool/graph。
TP 输入/retain-state 搜索尚未接入此 V4 suite，不允许借用 DP inference 结果。

初次实现只完成搜索/导出/查表的 CPU 合同检查和代表性编译；GPU 当时由两组
RL 作业占用。2026-09-30 GPU 空闲后开始真实搜索，后续验证记录见下节。
未完成完整搜索和独立验收前，不将生成候选或编译通过称为最优配置及性能收益。

实现验证记录：

- `tests/test_static_forward_autotune.py` 的 5 项检查通过，包括完整的模拟
  选优/确认/导出流程及错误输出候选淘汰；模拟延迟不是 GPU 性能数据。
- native build 缓存测试 3 项、GIN 配置头测试 3 项通过；tensor adapter
  C++17 syntax check 通过。decode M 自动/3 stages、M64/2 stages，network
  N128/N256 K128 3/2 stages 的 Q32/Q64 代表性 CUDA 编译通过。
- H2048/I512 的默认 decode 为18候选；双机 ring 资源过滤后44候选、22份不同
  编译代码（B 改变复用 binary）。复现日志保存在 `build/v4_autotune/`。
- 旧 `test_forward_autotune.py` 存在6个失败和1个错误；在未修改的 HEAD 工具上
  也复现同组失败，主要是候选数、参数面和旧 SMEM 常量断言。不能将新增检查
  通过写成整个历史测试集已通过。旧 topology/relay 测试也有既存配置假设问题。

### 2026-09-30：GPU 验证中发现的问题

首轮 DT EP8/T64 的18组候选均编译成功，但 incumbent 的不等长输入用例
在 control agreement 被拒绝：奇数 rank 实际 T32 被误传为准备容量，与偶数
rank 的 T64 不一致。`DeviceNotifyGenerationRequest` 最后一个参数是全组
统一的 `max_tokens_per_rank`，不是本 rank 的实际长度；修复该参数，实际
长度仍由 route/activation lease 提供。保留 uneven 用例及原有检查。

PF 首轮22份编译中10份 K64 失败。非模板 host 函数的 `if constexpr`
不能阻止分支内函数模板实例化，禁用的宽 prefill companion 仍实例化其固定
3/2-stage 组合，触发 FC2 epilogue scratch 容量断言。在共享模板入口统一
将未启用的 `Prefill` 映射为 false；不放宽容量断言，不改有效模板的计算或同步。

另修正候选合法值遗漏的32 lanes和Dispatch Q8，并使 ring 搜索的通信/容量
起点与已验证长文配置对齐。以上只修复离线搜索和禁用模板实例化，不切换
生产默认 profile。复用现有 GPU suite/集成检查，不新增私有符号单测。
首轮失败原始记录保留在 `build/v4_autotune_gpu/dt_ep8_t64.json` 和
`pf_build.json`；后续复验使用带 `_verified` 的独立文件，不覆盖失败记录。

修复后的独立验收（H2048/I512、1620 MHz，未替换默认 profile）：

- PF 22份不同编译配置全部通过，含之前失败的10份 K64。
  `pf_build_verified.json` 保存各配置、资源统计和编译记录。
- PF N128/N256、K128 3/2 stages 与 K64 3/3 stages 在单机 EP8 执行完整
  Graph，T8192 的均衡/热点/不等长及同路由 T1 前缀全部通过；两种配置的
  完整输出逐位相同。使用上述二进制的本地路径，不是双机 GPU 验收。
  原始记录：`pf_local_templates/`。
- K64 PF 的 PyTorch 四档 T1/64/8192/20480 + edge cases 共240项通过，
  包含20K全热点 no-drop、图复用及权重刷新。B16/D1 小 ring、T1/64/512
  的 rank0/7 专项 memcheck 各0 errors、racecheck 各0 hazards，两轮各200项
  应用检查通过。日志：`pf_k64_ep8.log`、`pf_k64_{memcheck,racecheck}*`。
- DT 2-stage（M仍按原密度规则选择16/64）通过 E64/K4 EP8 的248项、
  E256/K8 EP4的124项和EP2的62项现有检查，包含严格 batch-invariance、
  空/不等长 source、热点和权重刷新。EP8 rank0/7 专项 memcheck 各0 errors、
  racecheck 各0 hazards，两轮各248项应用检查通过；日志为 `dt_stages2_*`。
- 现有 autotune/build Python 检查11项通过，未新增单测。DT EP8/T64 已
  重新开始18候选完整搜索，incumbent 的 uneven 和跨 batch 前缀门禁已通过；
  当时完整确认/导出、T1搜索、双机长文搜索及 prepare 查表 GPU 验收仍待完成。

全部路径相对 `build/v4_autotune_gpu/`。上面的 sanitizer 只覆盖这次候选
对应的单机 Forward 内核，不覆盖任意节点、多机或完整训练 Backward。
不把候选数值/资源验收写成性能收益；仍沿用已验证的生产 DT/PF 配置。

### DT EP8 完整搜索（2026-09-30）

H2048/I512/E64/K4、BF16、1620 MHz。T64 在 `10.10.16.219`，T1 在
`10.10.19.233`，两者均为**独立单机 EP8**，不是双机 EP16。每档18候选，
全部通过原有输入/完整输出及前缀合同；T64计时均衡/热点/不等长，另用三组
T1前缀验证。T1本身只有三组计时，uneven 的长度也为1，不声称它覆盖空 source。
此suite的balanced使用连续top-k expert；历史冻结矩阵使用stride53，节点/卡间
fanout不同。候选发布前还需回测历史主矩阵，不能把这里的绝对延迟直接与旧V3
表拼接，也不能将这三种合成路由当成服务中的真实路由分布。

每候选30 rounds（10 warmup + 20 samples），incumbent/最快候选正反序
各确认60 rounds（10 + 50），最终110 rounds（10 + 100）；graph batch10。
分数为三组路由的 rank-max **均值**等权平均，不是历史矩阵的 p50。
聚合分数按初筛20 + 两轮确认各50个样本加权，最终分数为独立复验。

| T/rank | incumbent聚合/μs | 选中配置 | 选中聚合/μs | 最终复验/μs |
|---:|---:|---|---:|---:|
| 1 | 37.442 | 原自动M、3 stages、自动CTA | 37.442 | 37.298 |
| 64 | 72.531 | M64、3 stages、CTA×1 | 71.305 | 71.192 |

T1 challenger 为自动M/3-stage/CTA×1，聚合37.283 μs，仅改善0.426%，
不足0.5%阈值，保留 incumbent。T64 工具按既定均值规则选择 `v4_0013`，
账面改善1.691%，但不将此称为稳定的内核加速：两轮确认中，均衡路由双方
p50均约58.8–58.9 μs，热点均约97 μs；incumbent 的170/189 μs长尾抬高了
均值。初筛均值也仅为71.273对71.191 μs。没有追究函数名/代码地址微差，
也没有据此替换生产默认。

固定M64还存在负载取舍：同一候选在T1的初筛均值42.082 μs，而该档
incumbent为37.609 μs，约慢12%。因此表只证明已记录 workload 的测量与
数值验收，不能把固定T64的选型推广为全部实际 batch 的默认值；服务中一个
容量池覆盖多种实际长度时，应按实际负载分布确认。两档都未选中2-stage。

结果/逐rank样本：`build/v4_autotune_gpu/dt_ep8_{t1,t64}_verified.json` 及
同名 `-logs/`。导出表为 `table_t1.json`（compiled ID `629d1864c1bd`）和
`table.json`（`d3c134849f63`）；两者不同，不能用同一个 native 库不经重建
切换。选中头文件保存在对应 `-selected/`。生产配置仍保持原样。

prepare 查表的真实 GPU 验证也已通过：分别用两个选中头构建 native 库，
在T1/T64各自的8 rank组读取对应表。故意将环境变量中的ring关闭，表仍正确
提供B/D；均衡与全热点Graph通过PyTorch参考，再以全热点输出与关闭表、
显式设置同一B/D的准备结果比较，输出逐位相同，pool compute/GIN/Notify字节及占用相同。
各档8个 `PREPARED_TABLE_PASS`，记录在 `table_{t1,t64}_validation.log`。
固定M64的T64选中库另通过现有248项短容量集成检查，包含空source、
严格batch-invariance、图/权重更新；见 `dt_t64_selected_torch.log`。
这验证了现有查表接入，不新增运行时选型或同进程多库热切换。

### PF EP8 长文搜索（2026-09-30）

沿用上面的H2048/I512/E64/K4、1620 MHz和三路由计分方法，T8192在本机，
T20480在另一台机器，各自独立单机EP8。11份编译代码对应22个含B容量的
候选，全部通过完整输出及T1前缀检查；编译在计时前完成，不与测量争用CPU。
初筛30 rounds，正反序确认各60 rounds，最终110 rounds，graph batch10。

| T/rank | incumbent聚合/μs | 选中候选 | 选中聚合/μs | 综合改善 | 最终复验/μs |
|---:|---:|---|---:|---:|---:|
| 8192 | 3298.863 | K64、3/3 stages、B256 | 3218.326 | 2.441% | 3219.177 |
| 20480 | 7819.968 | K64、3/3 stages、B512 | 7746.671 | 0.937% | 7748.546 |

FC1/FC2均保持N128/N256，D8；incumbent为K128、3/2 stages、B512。
两档共用同一编译ID `975be619c0f1`，只改变运行时窗口容量，未引入新的计算
模板或通信协议。资源检查未见local-memory spill。结果为
`build/v4_autotune_gpu/pf_ep8_t{8192,20480}.json`及对应`-logs/`，导出表为
`table_pf_ep8_t{8192,20480}.json`，编译检查见`pf_local_build.json`。

这里的综合改善不能直接作为通用默认的收益。8K初筛中，保持K128而仅将
B512改为B256，均衡均值从1235.817增至1312.419 μs，热点从7436.454降至
7164.690 μs；pool从345.90降至234.17 MiB/rank，约省112 MiB。选中K64/B256
的最终均衡均值为1308.549 μs，热点为7124.480 μs：总分的改善主要来自热点，
不是所有路由都更快。20K的B256均衡退化更明显，最终选择保留B512。
这些数值仍是连续top-k的合成路由，不与stride53历史矩阵直接拼接。

随后按历史stride53路由回测：每配置正反序各60 rounds（10 warmup + 50 samples），
graph batch20，仍为各自独立单机EP8、1620 MHz。当前发布版使用命名整理后的
`build/dt_pf_cleanup/after/test_cuda_notify`，V3使用冻结
`build/static_inference_v3_20260923/test_cuda_notify`，显式设置
`WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`。下表是两轮各自的rank-max p50，单位μs。

| T/rank | 冻结V3 | 当前发布版 | 搜索incumbent | 选中候选 |
|---:|---:|---:|---:|---:|
| 8192 | 1487.420 / 1487.400 | 1251.050 / 1238.090 | 1233.490 / 1231.350 | 1306.010 / 1306.690 |
| 20480 | 3298.450 / 3297.800 | 2908.740 / 2910.710 | 2909.200 / 2909.020 | 2919.530 / 2919.950 |

当前发布版相对V3延迟下降15.9%–16.8% / 11.7%–11.8%。选中候选相对当前版
分别慢4.4%–5.5% / 0.3%–0.4%，因此**不替换生产默认**。搜索incumbent与发布版
并非同一个二进制：前者直接特化宽tile并关闭额外prefill entry，不能只因MNK
相同就合并为同一基线；这里的小幅差异未进一步归因于命名或代码排列。
三种现代PF配置的输入和完整输出hash在两轮均一致；未要求V3不同归约路径逐位相等。

原始记录为`pf_ep8_t{8192,20480}_frozen_verified.json`及对应目录。首轮
`..._frozen.json`遗漏V3禁用静态路径的开关，实际进入冻结二进制内的早期静态
实现，不能用作V3性能证据；保留作诊断，以上表格只使用修正后的完整重测。

共用选中头文件构建的native库SHA256为
`bc795c784505e857c4bb6f7b5c4ad7c56c500e963ca2eb275e94ac5526be79da`，
保存于`build/v4_autotune_gpu/pf_selected/`。两张表分别完成真实8-rank准备：
均衡/全热点通过PyTorch参考；表驱动与显式相同B/D的全热点输出逐位相同，
compute/GIN/Notify分配字节及释放后占用相同，各8项`PREPARED_TABLE_PASS`。
日志为`table_pf_ep8_t{8192,20480}_validation.log`。
同一库在B256/D8及B512/D8各通过现有240项四档集成检查，覆盖T1/64/8192/20480、
20K全热点no-drop、空/不等长source、Graph/权重更新和热点后的workspace复用；
日志为`pf_selected_b{256,512}_torch.log`。没有新增单测或运行时选型功能。

这些是单机选中profile的数值、生命周期和查表证据；本轮未重复该库的sanitizer，
也不把前文另一K64编译profile的专项报告当作此库或完整训练的新sanitizer结果。
双机EP16长文搜索及独立发布门禁仍待完成。生产DT/PF默认和性能分流保持原样。

### 调参驱动：摊销进程、权重与通信池准备（2026-09-30）

EP16/8K原搜索在26/44个候选后中断，20K尚未开始。初筛最佳仅比incumbent
低0.752%，未做确认，未采用。旧实现每个候选独立启动六次16-rank作业，
实测每次约28–33秒；一次完整候选175.699秒。这主要是测试驱动的重复准备，
不是kernel运行三分钟，也不是编译耗时（该轮全部命中缓存）。
单次启动诊断总耗时29.933秒，其中bootstrap约0.041秒，结束清理约1.576秒；
不将剩余worker内耗时全部误归因于SSH或NCCL初始化。

只调整测试驱动，未改生产kernel、协议或默认参数：

1. 六个用例放进一次rank启动，共用CUDA context/NCCL communicator。
   EP16单候选降到118.483秒，所有完整输入/输出hash与旧流程一致。
2. 同一候选只准备一次不可变权重及其校验；同容量的三种路由连续运行并复用
   Notify/transport pool。切换长文/T1容量时释放旧池，各例Graph/output完全
   退休后再复用，使用不同generation；不同时保留两套注册arena。
   汇总恢复原case顺序，兼容原完整正确性fingerprint。

H2048/I512/E64/K4、1620 MHz、8K及T1的六例，仍各30 rounds（10 warmup +
20 samples）、Graph batch10。以下是一次候选的**benchmark墙钟**，不含编译，
不是算子加速比；单机和双机顺序执行，无其它GPU作业或CPU构建：

| 拓扑 | 六次独立启动/s | 仅会话复用/s | 加上权重与池复用/s | 调参加速 |
|---|---:|---:|---:|---:|
| 单机EP8 | 61.221 | 未测 | 12.756 | 4.80× |
| 双机EP16 | 175.699 | 118.483 | 54.255 | 3.24× |

两拓扑六例的完整输入/输出fingerprint均与原incumbent一致；每rank的compute、
GIN及总pool字节数逐例相同，batch-invariance/内存预算校验全部保留。
测量分数、交错确认和最终完整验证规则不变。EP16综合kernel均值从6727.772
变为6750.523 μs，不能把驱动墙钟收益当成GEMM/通信kernel性能收益。

复用既有5项static-autotune检查和13项launcher检查通过，未新增细粒度单测。
传统`test_forward_autotune.py`的6项失败、1项错误在HEAD版本复现相同结果，
不宣称整份历史套件通过。首次资源复用构建有测试结构体namespace错误，修正
后构建通过，失败日志保留。
证据在`build/v4_autotune_gpu/`：`startup_profile.log`、
`suite_ep16_validation.{json,log}`、`suite_reuse_ep{8,16}_validation.{json,log}`，
最终测试二进制为`suite_reuse/test_cuda_notify`。本次没有恢复44候选完整搜索，
也没有把中断的双机搜索标记为完成。
该二进制SHA256为`c4f526d44027b8ae964156eb4a9fb27befd4e8ee93988d38043eea9d17c1f740`。

## 传统默认配置（历史说明）

以下四份传统配置使用纯超参数文件。所有值是非空列表，第一项组成 incumbent，多值轴做笛卡尔积展开；模型维度、experts、top-k、tokens 和系统配置不在搜索空间中。

| 阶段 | 配置 | 组合数 | RDMA CTA |
|---|---|---:|---:|
| FWD 单机 | [forward_h200_single_node_autotune.json](../configs/forward_h200_single_node_autotune.json) | 10 | 0 |
| FWD 多机 | [forward_h200_multi_node_autotune.json](../configs/forward_h200_multi_node_autotune.json) | 10 | 8 |
| BWD 单机 | [backward_h200_single_node_autotune.json](../configs/backward_h200_single_node_autotune.json) | 10 | 0 |
| BWD 多机 | [backward_h200_multi_node_autotune.json](../configs/backward_h200_multi_node_autotune.json) | 40 | 8 |

并行配置决定调参配置：`EP = nodes × gpus_per_node`。默认 `nodes=1`、`gpus_per_node=8`；可以传 `--ep` 推导节点数，也可以传 `--nodes`，同时传入时必须一致。配置文件名、`rdma_ctas`、`--remote` 均不用于推导拓扑。未传 `--config` 时，按节点数选择对应阶段的单机/多机默认文件。

单机在候选展开前将 `rdma_ctas` 固定为 `[0]`，原文件保持不变；多机当前内核要求 `[8]`，这是每 GPU 的 GIN owner 数，与 EP 大小无关。其它超参数保留用户值并校验，显式复用另一拓扑的文件可能因其它参数不兼容而报错。候选数由实际文件的笛卡尔积计算。

拓扑表达和候选生成支持 EP32、EP64 等更大规模；**实际 GIN FWD/BWD 和 launcher 目前仍只支持两机各八卡**。内核包含固定远端节点的协议逻辑，因此三机以上的性能运行会提前报错，不会悄悄启动两机。这部分还需要独立改造，不能将计划生成成功视为多机执行已支持。模型由调用方解析，再通过 Python 或 CLI 传入；不猜测模型形状。候选列表全部在计时前确定，结果不反馈修改搜索空间。展开与编译检查不意味着所有组合均已实测。

## 候选轴

FWD 保留 Relay CTA `[8, 16, 24, 32, 40]` 和 local Return TMA warp 两个候选轴；单机 Return 候选为 `[0, 8]`，多机为 `[8, 0]`。FC1/FC2 tile N 固定为 128，local Dispatch TMA warp 单机为 8、多机为 12，local Return batch 为 4。

- `local_dispatch_tma_warps`：每个 Relay CTA 执行节点内 Dispatch 时的 TMA 复制流数量。
- `local_return_tma_warps`：每个 Relay CTA 执行节点内 Return 时的 TMA 复制流数量；0 表示使用向量复制。

配置入口仍接受旧名 `relay_tma_warps`，但同一组参数不能同时使用新旧名称。
为兼容正在运行的 autotune 和已有缓存，内部候选记录、digest 与生成的 CUDA 宏暂时沿用旧标识；新旧配置名称生成相同候选。

BWD 候选：

| 参数 | 单机 | 多机 |
|---|---|---|
| relay_ctas | 8, 16, 24, 32, 40 | 8, 16, 24, 32, 40 |
| dispatch_batch_rows | 16, 32 | 64, 32 |
| dab_tile_n | 256 | 256, 128 |
| dx_tile_n | 256 | 256, 128 |

`activation_owner_group` 固定为 4，`reduce_threads` 固定为 256。2026-09-04 的 BWD qualification 中，EP8 的 owner group 2/8 和 reduce threads 128 均未快于 incumbent；EP16 的 owner group 2/8 也没有收益。这是缩减默认搜索空间的依据，不代表其它模型下这些参数永远没有收益。保留 Dispatch batch 作为低成本的通信粒度候选，保留多机 GEMM tile 轴以覆盖不同模型形状。

全部超参数仍保留在文件中，只缩减候选列表；每份默认配置不超过 80 组。第一项组成的 incumbent 不变。阶段数保持固定，避免独立展开破坏 dAB/dX、dW1/dW2 的配对约束。单机 RDMA CTA 自动固定为 0，多机当前要求八个 GIN owner。

## Python

将 `tools` 加入 Python import path 后：

```python
from pathlib import Path
import autotune_backward as tune  # FWD 使用 autotune_forward，接口相同

plan = tune.load_plan(
    Path('configs/backward_h200_single_node_autotune.json'),
    model=dict(hidden=2560, intermediate=768, experts=512, top_k=8),
    benchmark=dict(tokens_per_rank=65536),
    system=dict(nodes=1, gpus_per_node=8),
)
candidates = tune.expand_candidates(plan)
```

`model` 的四个字段必须显式传入。H/I 用于编译期特化，experts/top-k 传入性能 launcher。BWD 小规模正确性 preflight 保留独立小 fixture，不等同于全模型规模性能测试。

## CLI

仅生成候选，不启动 GPU 测试：

```bash
python tools/autotune_backward.py --nodes 1 --gpus-per-node 8 \
  --config configs/backward_h200_single_node_autotune.json \
  --hidden 2560 --intermediate 768 --experts 512 --top-k 8 --tokens 65536 \
  --generate-only --output artifacts/bwd_candidates.json
```

替换脚本及配置即可生成 FWD 候选。去掉 `--generate-only` 进行 autotune；两机再传 `--nodes 2 --gpus-per-node 8 --remote root@HOST --port 2222`（也可以用 `--ep 16` 替代 `--nodes 2`）。`--build-only --candidate incumbent` 只构建指定候选，多机仅生成/编译时不要求 remote。

两个工具均支持 `--system-config /path/to/system.json`；系统配置由用户在仓库外提供，可省略。允许字段为 ep、nodes、gpus_per_node、remote、port、world（旧版单机别名，须与每节点 GPU 数一致）、socket_ifname、disable_ib（单机）、enable_ib（多机）、nccl_lib_dir、runtime_library_path 和 jit。CLI 同名字段覆盖系统配置；省略拓扑时使用单机八卡。网络字段按已确定的拓扑生成默认值，不强制网卡名；若存在仓库私有 NCCL 库则自动采用，并组合 CUDA compat、NCCL、Python CUDA runtime 路径。

`jit` 可覆盖 jobs、cache_root、architecture 及编译工具位置。`--jobs`、`--cache-root` 可直接覆盖；计时与确认轮数也可由 CLI 覆盖。默认筛选 10+20，incumbent 与最快 challenger 正反确认 10+50，改善不足 0.5% 保留 incumbent，最终 10+100。

Makefile 入口接受相同参数：

```bash
make backward-autotune \
  BWD_AUTOTUNE_CONFIG=configs/backward_h200_single_node_autotune.json \
  AUTOTUNE_ARGS='--nodes 1 --hidden 2560 --intermediate 768 --experts 512 --top-k 8 --generate-only'
```

中间产物写入忽略的 `artifacts/`；`benchmarks/results/` 也不再纳入 Git。Return 机制见 [设计说明](return_pipeline.md)。

验证覆盖四份配置的完整展开、全部超参数校验、CLI 生成、单机/多机 BWD 编译，以及零 RDMA BWD 的两卡小规模运行；未运行整组候选的 GPU autotune。
