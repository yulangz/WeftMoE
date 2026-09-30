# Forward / Backward autotune

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
  完整确认/导出、T1搜索、双机长文搜索及 prepare 查表 GPU 验收仍待完成。

全部路径相对 `build/v4_autotune_gpu/`。上面的 sanitizer 只覆盖这次候选
对应的单机 Forward 内核，不覆盖任意节点、多机或完整训练 Backward。
不把候选数值/资源验收写成性能收益；仍沿用已验证的生产 DT/PF 配置。

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
