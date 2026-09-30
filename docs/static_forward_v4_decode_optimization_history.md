# V4 单机 decode 优化历程和适用范围

更新日期：2026-09-30。本文整理通用 ring 的短文优化、HF 专属 decode、HG
通用 EP 支持、HH 系列及 SGLang DP-local 接入。重点是逐步收益、失败实验和
对其他模型的适用边界，不把实验流水账里的旧实现当成当前实现。

当前专属 kernel 为 HHE，提交 `07a586d`；后续接入和 benchmark 以
`a54556c` 为本次文档审计基线。阶段三并非所有项目都能无条件推广：消除冗余
staging 和重复同步的思路较通用，M16/M64、CTA 驻留和密度阈值则必须结合
模型形状与路由负载。当前没有其他 H/I 模型的完整性能验收。

长文和多机的演进另见 [prefill 优化历程](static_forward_v4_prefill_optimization_history.md)。
逐次参数、冻结 hash、原始日志和验收详情见
[decode 实验记录](static_forward_v4_decode_record.md)、
[ring 实验记录后半部分](static_forward_v4_ring_record.md) 和
[AReaL 接入记录](areal_qwen35.md)。

## 哪些优化通用 哪些需要重新测量

“通用”应拆为算法能复用、当前实现能接入、性能已经验证三件事。

| 优化 | 可以复用的原理 | 条件和已知限制 |
|---|---|---|
| 融合 staging 与输出回填 | 合并小 kernel、减少冗余张量和复制，通常降低固定开销 | 当前实现针对 SGLang BF16、稳定 child-graph 地址；DP-local 不需要 TP 布局的输出 scatter。更大 H 的 bandwidth/occupancy 仍要测 |
| padding 不激活无关专家 | 零 X/gate 的补齐行不应制造额外专家 GEMM 任务 | 仅有补齐且原 route 激活了无关专家时有收益；T≥EP 且整除时可能完全不触发 |
| 跳过空 warpgroup 和稀疏 TMA | 少做 padding 行的算术和搬运 | 只对稀疏专家有效；这些是历史中间实现，最终由 M16/M64 slice 取代，不是永久叠加的独立分支 |
| TMA 与 math 流水、合并 empty credit | 减少不必要等待和控制操作 | 最佳 stage 数随 H/I、tile、寄存器及 SMEM 变化；不是所有形状都应该固定三段 |
| M16 warp MMA 与 M64 WGMMA | 稀疏时避免无效 M 行，密集时摊薄权重加载 | 依赖实际每专家行数及硬件，M16 并非越小越快；已有 E64/T64 反例 |
| CTA 数量按专家任务量裁剪 | 避免大量空 CTA 参与 grid/rank collective | 最佳并行度受专家数、N-tile 数、SM 数和 occupancy 影响，不能复制某个固定 CTA 数 |
| 每 CTA 汇总阶段计时 | 避免逐 tile 的全局计时原子操作 | 原实现有这些统计才会省，独立收益仅约 1–2 µs，不是主因 |
| 省去 AfterClean | 下一次入口同步可以同时承担旧 contribution 退休确认 | 仅适用于同一 pool 串行、单容量窗口、无末尾通用计数器清理；不能推广到在线多窗口或多机 ring |
| DP-local 输入输出 | 让融合 MoE 自己完成路由和返回，取消重复外围 collective | 要与模型层布局、shared expert、权重更新协议一致；TP replicated baseline 本来不需要 dispatch |

已有性能证据主要覆盖两类 routed MoE：

| 覆盖项 | 已有证据 | 不能据此声称 |
|---|---|---|
| H2048/I512/E64/K4，EP8 | native、MegaMoE 单机 T1/T64 和长文回归 | 这等于测过所有小模型 |
| H2048/I512/E256/K8，EP1/2/4/8 | native 正确性；HG 部分性能；HHE EP2/4/8 的 TP 三方性能；DP4 三方性能 | EP 和专家数量覆盖等于其他 H/I 的性能覆盖 |
| E64/K8，EP1/2 | 密集 M64、跨形状切换、batch-invariance 及 sanitizer | 独立的全性能矩阵已完成 |
| Qwen3.5-35B-A3B，DP4+EP4 | 实际 SGLang/AReaL rollout 对照 | 其他模型、FP8/INT8、其他 GPU 或网络拓扑同样加速 |

当前专属路径还有明确边界：单机 inference-only，pool 每卡容量不超过 64，
最坏路由装得进一个窗口，配置总专家数上界不超过 256，编译 profile/TMA map
匹配。H/I 参与编译期任务数和 TensorMap 构造，不是把新模型权重直接送进当前
二进制就能获得相同专属实现。E>256、不同 H/I、量化或不同激活需重新检查支持
边界、构建和验证，不能将 fallback 或入口拒绝误当成同一 kernel 的测量。

## 计时口径

本文独立正式微基准均为 L20X、SM 1620 MHz，计时期间无并发编译或其他 GPU
workload。早期 E64 native/MegaMoE 为每 rank T；Qwen TP 表为全局复制 T；
DP 表为每卡 T、全局 EP×T。三者不能直接跨表相减。

native 通常 10 warmup、20 samples、每 sample 20 replay；HH SGLang 诊断
通常 5 warmup、20 samples、50 replay；最终三方对照为 5 warmup、30 samples、
50 replay，轮换后端顺序，先取同轮 rank-max 再取 p50。单位均为 µs。

SGLang 单层 benchmark 含 routed MoE 必需的 staging、输出适配及相应
collective，不含 router/top-k、shared expert、attention、JIT、权重打包、
分配和图捕获，也不是多层大权重冷工作集测试。kernel-only MegaMoE 和
含输入 staging 的 MegaMoE 分开列；不相加两次独立计时的 median。

## 先精简通用 ring 的短文开销

GA+FZ 的 E64/K4/EP8 同输入对照，ring T1/T64 为约 103.0/137.3 µs，
MegaMoE 含 staging 为约 53.1/63.5 µs，短文仍明显落后。下面先在不改
通用数据面合同的前提下精简规划和 Pull，所有数字均来自各自原记录的同条件
对照，不组成一条可直接相加的统一时间序列。

| 实验 | 有收益的改动 | native T1/T64 的独立结果 |
|---|---|---|
| GR | 单机单容量窗口规划由 CTA0 拥有，减少四次不必要 grid join | 101.288/135.104→97.035/130.512，降低 4.20%/3.40% |
| GS | source 前缀由串行循环改为分段 warp scan | 在 GR 上约再省 1.3%/0.7%，正反序重现 |
| GU | 少量工作窗口按 Mblock 交错有效行，改善静态 Pull owner 覆盖 | 95.710/129.486→91.218/114.931；反序也约省 4.4%/11.4% |
| GV | source 将 counts/routes 发布到 destination 本地目录 | T1/T64 约再省 1.7%/1.0%；X 仍是 destination-pull |
| GW | destination Mblock 前缀改为 warp scan | 约再省 1.0%/0.9%，多轮重现 |
| GY | 小窗口 reverse 规划并入 geometry，复用同一次发布 | 反序 88.910/112.578→85.506/108.054，降低 3.83%/4.02% |
| GZ | 单机 inference 不再生成没有读者的 node 目录 | 反序 85.619/107.741→78.498/98.848，降低约 8.3% |
| HC | 单容量窗口直接构建 destination 目录，删除中间 source 分组与转存 | T1 持平，T64 约再省 0.8%–0.9% |
| HE | route 与 gate 一起发布，消除逐 route 的 peer gate 读取 | T1 再省 1.6%–1.9%，T64 省 2.3%–2.4%；T1/T64 每 rank 额外目录各 1/64 KiB |

来源：ring 记录 GR–HE。GR/GV/GY/HC/HE 依赖公共 pool 容量条件，不能按
各 rank 当前 T 独立切换跨 rank 协议；GU 使用已发布的 destination 工作量，
GZ 覆盖单机 inference 的所有窗口。长文、多机和训练分别回归，不把未触发
分支上的微小计时变化算作新收益。

这期间也有失败：GT 只把 Batch8 换成 Batch1，T64 反而变慢，GU 修正分工后
才获益；GX 把 reverse metadata 延后到 Pull，规划 span 虽少 4–6 µs，
总延迟却慢约 0.3%–1.7%；HA/HB/HD 的提前 admission、集中 staging 和
geometry 发布候选均撤回。说明减少某一阶段工作不等于缩短关键路径。

HE 补齐完整门禁后的同输入 ring T1/T64 仍约 78.83/97.12 µs，未接近
MegaMoE。因此下一步不再继续堆通用 ring 短文分支，而是独立专属入口。

## HF 用单窗口工作流替代通用在线生命周期

HF 新增 `src/cuda/local_decode/`，复用已有数值 helper、packed 权重和 IPC
内存。静态 pull/TMA/math 角色直接执行单窗口任务，FC2 写 source contribution，
不需要通用 return-fragment 目录和调用内 slot 换代。独立短容量 runtime
保证最坏热点也装得下，不靠均匀路由假设或丢 token 换速度。

E64/K4/EP8 native 对照：

| 每卡 T | HE | HF | 延迟降低 |
|---:|---:|---:|---:|
| 1 | 77.141 | 47.427 | 38.5% |
| 64 | 95.762 | 55.270 | 42.3% |

独立同输入 MegaMoE 反向对照，HF 为 47.705/55.623，Mega kernel-only 为
49.121/57.535，Mega 含 staging 为 53.098/63.322。两轮分别领先 bare kernel
约 2.9%–4.0%，领先含 staging 约 9.9%–12.7%。单机 8K/20K 和多机四档
回归持平；八个通用 ring 主 kernel SASS 不变。

HF 初始为 M128、双 math warpgroup、cluster multicast 和三个 rank 边界。
这是工作流与 GEMM 起点共同变化的整体收益，没有单因素拆分。它不是当前 HHE
的线程配置，也不能把后来删除第三个 barrier 的收益重复计入 HF。

## HG 支持更多单机 EP 并处理图和权重生命周期

HF 每卡专家数最多 32，E256 在 EP4/2/1 下有 64/128/256 个本地专家，
会回退通用路径。HG 将专家前缀改为每轮 32 项的 warp scan，携带跨轮前缀，
保持任务、packing、归约次序和通信协议。不是仅把数组扩大。

| Qwen E256/K8 | 每卡 T | HF | HG | 解释 |
|---|---:|---:|---:|---|
| EP1 | 1 / 64 | 105.893 / 971.586 | 69.390 / 716.998 | HF 原为通用回退 |
| EP4 | 1 / 64 | 105.470 / 336.587 | 53.968 / 177.136 | 专属入口启用后约省 49%/47% |
| EP8 | 1 / 64 | 52.718 / 108.061 | 52.768 / 108.370 | 已支持的拓扑持平 |

EP2 的 HG 为 61.349/341.038 µs；冻结 HF benchmark 入口拒绝 EP2，没有
编造旧版对照。以上不是相同全局 batch 的 EP 强扩展比较。

SGLang 同时使用容量 64 的 decode 池与大容量 prefill 池；按层缓存权重 lease，
让多个 graph bucket 共享 packed W1；child graph 输入绑定稳定 staging 地址。
offload 先销毁外层图再释放池，权重更新后重建。早期重复打包导致的显存/OOM
问题由生命周期和共享处理，不将其冒充已量化的 kernel 加速。

HG 完成 EP1/2/4/8 的 465 个 native、30 个 SGLang rank/case 检查；原 E64
长文 240 项回归通过。完整应用成功和单层正确不代表 RL 已比 baseline 快。

## HH 针对真实 Qwen 小 batch 降低无效工作

实际 RL 常见全局 T32–64；E256/K8 的专家分布比最初 E64/K4 稀疏。下表是
TP4+EP4、全局复制 T、包含真实适配层的演进节点，不混入 per-rank benchmark。

| 阶段 | 改动 | T32 | T64 |
|---|---|---:|---:|
| HG | 初始适配层 | 160.969 | 192.961 |
| HH1 | 融合 staging 及输出补零/回填 | 146.748 | 178.628 |
| HH2 | M128 中有效行≤64 时跳过空的上半 WGMMA，仍归还全部必要 credit | 141.588 | 172.087 |
| 流水 | 保留最新 WGMMA group 在飞，按 warpgroup 合并 empty credit | 139.844 | 169.774 |
| 稀疏 TMA | ≤8 行只搬 8 行 A，而非 128 行 | 131.951 | 161.104 |
| 首尾统计 | 每 CTA 每阶段只汇总首尾，避免逐 tile 计时原子 | 130.742 | 159.502 |
| HHD | M16/M64 双形状、A/B TMA 和合理 grid，最终整理版 | 125.245 | 150.888 |
| HHE | 省去 AfterClean，复用下一调用 BeforePull 退休旧 contribution | 123.524 | 148.877 |

HG→HHE 累计降低 23.3%/22.8%，不是 M16 单项收益。中间候选后来被重新
组织为共享的一套 geometry、Pull、producer、epilogue 与同步；当前没有保留
旧的 M128 双 warpgroup 加稀疏 8-row descriptor 作为第三套生产实现。

### padding 路由修正的收益单独计算

TP 输入等长切片时，空 source 的零 X/gate 原来配专家 0–7，仍然触发这些
专家的 GEMM。改为复用首个真实 token 的专家 ID，避免 padding 引入无关
专家。真实 route 不变，也没有把真实零权重 route 擅自删除。

同一个 M64/N128/3-stage 二进制，EP4 全局 T1 68.690→53.812 µs，降低
21.7%；T16/32/48/64 持平。这项改善不能推广到没有补齐的 batch。

### 为什么不是全量使用 M16

M16 用四个 math warp 分别承担 N32，A/B 由 TMA 装载，`ldmatrix` 读
fragment，再执行固定 K16 次序的 `mma.sync`。单形状 M16 在 E256 稀疏
负载有益，但 E64/K4/EP8 每卡 T64 时达到 67.774 µs，反而慢于同轮
MegaMoE kernel 的 58.962 µs；平均每专家行数约 32，重复遍历权重的成本变大。

因此引入 M64 WGMMA，使用静态密度估计：

```text
local_tokens × world × top_k >= experts × 16  → M64
否则                                           → M16
```

它近似均匀路由下的每专家行数，不是逐专家实时负载判断，更不是所有模型的
最优分界。两种形状共享 M128 物理目录和跨 rank 协议；不等长 source 可以
选择不同 M，但必须保持逐位数值和协议兼容，热点遍历全部 slice，不丢 token。

随后 grid 依据本地专家的 tile 数限制请求驻留倍数，M16/M64 上限分别为
3/2，并受实际 occupancy 限制。E64/EP8 改回一 CTA/SM 后，T64
61.530→59.223 µs；E256/EP4 保留较高并行度。这个经验估计避免明显空 CTA，
不是已经完成所有 H/I 的 occupancy autotune。

### HHE 两个 rank 边界如何保护复用

HHE 只在同一 pool 的调用按 stream 串行、所有 payload 在一个窗口内，且
末尾没有通用 ring 计数器清理时删除 AfterClean。

1. BeforeCombine 收齐 FC2 peer stores，旧 source X、route 和 canonical X/S
   已经没有读者；之后只剩 source 本卡 Reduce 读取 contribution。
2. 下一调用可以提前准备 X/route，但新的 FC2 必须经过下一次 BeforePull。
3. 每个 rank 只有完成旧 kernel 的 Reduce 才能进入下一次 BeforePull，
   所以该同步也确认所有旧 contribution 读者结束。

保留双 phase 的 rank 信号，不能由快 rank 越过入口。EP4 另用图内延迟模拟
rank 偏斜，复用检查通过。主要 bucket 的独立收益为 1–2 µs；长文通用入口
没有改动，不能把这个证明用于多窗口在线复用或多个 stream 并发使用同一池。

## 没有保留的 decode 消融

| 候选 | 实际结果 | 教训 |
|---|---|---|
| M64 紧凑 CTA 全量替换 | T48–256 改善，T1–32 退步 | 减少 SMEM、增加驻留不保证稀疏计算效率 |
| N64 加更多 stages 或三 CTA | 主流 bucket 没有改善 | 更小 N 会增加任务/权重/同步开销，不能只看 occupancy |
| 寄存器 A 的 WGMMA | 测试过的实现明显退化，例如 T32 到 166.925 µs | 逐 K64 退休寄存器片段的成本较大，不能据此否定所有寄存器 A 设计 |
| 仅一条 Pull warp | T1/16/32 几乎不变，部分 bucket 变慢 | 省线程和 staging 没带来稳定净收益，保留三条 Pull warp |
| 每物理块 FC1 后紧跟 FC2 | T32/T64 140.277/169.528，对 HHE 慢约 13.6%/13.9% | 提前安排 FC2 降低初始可执行 FC1 的 CTA 比例，未获 overlap 净收益 |

另一次整理构建误用 prefill FC2 N256 descriptor 检查 decode N128，导致
专属入口始终不选中。对应 `final_mega_*` 数据已排除；测性能必须核对实际
kernel，不能只凭 binary 名称判断走了候选。

## 最终单层表现取决于输入布局

### E64 的单机 native 对照

HHE，EP8、每卡 T、同输入交错 graph：

| 每卡 T | HHE | Mega kernel-only | Mega 含 staging |
|---:|---:|---:|---:|
| 1 | 36.956 | 49.587 | 53.062 |
| 64 | 59.360 | 59.998 | 65.876 |

T64 与 bare kernel 仅约 1% 差异，结论是接近，不称显著领先。这个结果也不
证明 HHE 比历史 HF 的每个点都更快；历史 HF 和 HHE 不是此表的同轮单变量对照。

### Qwen 的 TP replicated 布局

EP4、全局 T，双方包含必要适配及共同的外层 all-reduce：

| 全局 T | SGLang baseline | MegaMoE | HHE |
|---:|---:|---:|---:|
| 1 | 29.825 | 63.712 | 47.269 |
| 32 | 106.241 | 162.474 | 123.420 |
| 64 | 150.177 | 190.054 | 149.761 |
| 128 | 186.565 | 202.729 | 163.707 |

T1–48 仍慢于 baseline，T64 反序确认持平，T96/128 更快；所有点均领先
所测 MegaMoE。EP2/EP8 补测同样领先 MegaMoE，但 EP2 全部点仍慢于 baseline。
因此“支持 EP2/4/8”“领先 MegaMoE”“领先 SGLang”是三个不同结论。

原因之一是工作流不同：baseline 的复制输入已在各卡，可直接计算本地专家后
all-reduce；Weft/Mega 兼容布局仍有自身 dispatch/return，再参加外层归约。
这不是一项 GEMM 单因素对照。完整表和反序稳定性见 decode 记录第 8 节。

### Qwen 的 DP local 布局

DP4+EP4 为四张卡，attention TP1、expert TP1，每卡输入不同。baseline
使用 SGLang 默认 Triton 和 `moe_a2a_backend=none`，不是 DeepEP。

```text
baseline：DP-local X → all-gather → 本地专家计算 → reduce-scatter
WeftMoE：DP-local staging → 融合 pull / FC1 / FC2 / return / combine
```

| 每卡 T | 全局 T | baseline | MegaMoE | HHE | 相对 baseline 延迟降低 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 67.588 | 74.278 | 53.032 | 21.5% |
| 2 | 8 | 73.116 | 77.590 | 58.398 | 20.1% |
| 4 | 16 | 102.288 | 135.062 | 97.009 | 5.2% |
| 8 | 32 | 122.100 | 154.525 | 115.328 | 5.5% |
| 12 | 48 | 148.347 | 172.280 | 135.388 | 8.7% |
| 16 | 64 | 166.205 | 179.893 | 140.693 | 15.3% |
| 24 | 96 | 194.852 | 188.935 | 148.291 | 23.9% |
| 32 | 128 | 204.268 | 188.752 | 149.610 | 26.8% |
| 48 | 192 | 210.642 | 189.807 | 152.589 | 27.6% |
| 64 | 256 | 218.958 | 192.745 | 158.202 | 27.7% |

反序全部点重现，p50 最大变化 baseline 0.45%、Weft 0.81%、Mega 0.61%。
Weft 相对 Mega 延迟降低 17.9%–28.6%。这里只测四卡均衡非空输入；没有
用此表代替不均衡/idle 调度的性能验收。trace 确认 baseline 有 gather 和
reduce-scatter，Weft/Mega 没有外围 NCCL collective；profile 时间不混入正式计时。

生产接入还需把 shared expert 建为 TP1 副本，并修正 AWEX 对 attention TP1
和 shared expert 的权重描述。AReaL reader 修复两组共用，Weft 的 shared
expert 规则仅在实际使用 DP-local 副本时安装。它们是正确性条件，没有独立
性能消融，不能再计一笔“权重同步加速”。详见 AReaL 接入记录。

## 实际 RL 的收益和证据边界

初版 HG 的 TP4 RL 共同第 2–3 步有效生成吞吐为 7496 token/s，对照约
9007；优化后的 HHD 在同两步约 8473。后续 HHD 与 baseline 共同第 2–16
步为 8446 对 8999 token/s，仍落后约 6.1%。不能把早期少步数的改善写成
最终已追平，也不能将 HHD 的 RL 结果标成 HHE。

2026-09-30 改为双方 DP4+EP4、Weft HHE 后，取已完成的共同第 2–11 步
快照，排除首步，模型/数据/seed 和 batch 配方相同，actor 都是原版 Megatron：

| 指标 | baseline | WeftMoE |
|---|---:|---:|
| 平均 rollout 时间 | 581.312 s | 533.343 s |
| 有效生成 token 总数除以 rollout 总时间 | 8360.07 token/s | 9105.43 token/s |
| 平均完整 update 间隔 | 791.999 s | 742.999 s |
| 平均序列长度 | 9650.77 | 9643.94 |

该快照 rollout 时间降低 8.25%，有效生成吞吐提高 8.92%。每组为独立单机
八卡、两个四卡 SGLang 实例，不是两机 EP8。两组运行在不同节点，目标 SM
1620 MHz，绝大多数采样达标但有少量瞬时降频；没有完成交换节点的交叉实验。
这不是完整 20-step 最终报告、训练收敛结论，也不是仅 decode kernel 的吞吐。
actor 未替换，完整 update 的 6.19% 改善不能全部归给 MoE。

原始数据为 `build/rl_dp4_ep4_20260930/rl_baseline.log` 和
`rl_weftmoe.log`；使用 `benchmarks/publish_areal_logs.py::read_updates`，筛选
`global_step=1..10`，吞吐为 `sum(ppo_actor/update/n_valid_tokens) /
sum(timeperf/rollout)`。值保留 console 打印精度；不以当前仍增长的最新步骤
覆盖此固定范围快照。后续完整 RL 结果应更新 AReaL 记录并明确新范围。

## 当前 HHE 实现

每 CTA 256 threads：128 math、一个 TMA warp、三个 Pull warp。物理目录
仍为 M128，计算为 M16 MMA 或 M64 WGMMA，N128/K64、三段 CTA-local TMA
流水。当前 decode 已不使用 HF 的 cluster multicast，通用 prefill 不变。

任务按 `blockIdx.x + n * gridDim.x` 静态分配，全部 FC1 task 后接 FC2 task；
各 CTA 独立推进，不新增跨 rank 的 FC1→FC2 barrier。Pull 完成有效行后
release 增加 `x_ready`，FC1 acquire 等待完整有效行。FC1 每个 slice/N-tile
完成 S 写回后发布 `s_ready`，FC2 等所有 FC1 slice 完成。stage 的 full/empty
barrier 管 A/B 复用，FC2 直接写 source 独占 contribution slot。

rank 边界只有 BeforePull 和 BeforeCombine；最终按稳定 expert/slot 顺序
FP32 累加，再舍入为 BF16。W1 仍为 gate8/up8 packed；G/U 投影先按原合同
舍入 BF16，再做 SwiGLU、router weight 和 S 的 BF16 舍入。没有 split-K 或
浮点 atomic reduction，不能为追性能移动 route weight 到 FC2 之后。

专属路径的固定 K16 累加次序、舍入位置及稳定 combine 顺序经过严格逐位
batch-invariance 验证，包括 batch 1/7/32/64、行置换、不等长/空 source、
热点跨块、形状切换、通用 V4 对齐、graph 输入和权重更新。三方 benchmark
使用容差检查，不声称 Weft、MegaMoE、SGLang 三者彼此逐位一致。

入口和实现：[schedule.cuh](../src/cuda/local_decode/schedule.cuh)、
[compute.cuh](../src/cuda/local_decode/compute.cuh)、
[kernel.cuh](../src/cuda/local_decode/kernel.cuh)、
[host 选择](../src/cuda/activation_transport.cu)、
[SGLang 适配](../weftmoe/sglang.py)、[staging](../weftmoe/_sglang_kernels.py)。

## 推广到其他模型时如何验收

先核对 H/I/E/K、dtype/激活和 compiled profile 是否支持，确认实际进入
`local_decode::kernel<16>` 或 `<64>`，不能仅检查 Python 层返回成功。
然后复用现有入口做三个层次的检查，不为每个候选增加一套单测：

1. 正确性与生命周期：普通/热点、空/不等长、容量边界、图复用、权重更新，
   以及 strict batch-invariance；改变算术或同步后做相关 sanitizer。
2. 同机同输入微基准：分别比较原形状、M16、M64、实际选择策略；覆盖每专家
   平均行数约 1/8/16/32/64 的区间和偏斜路由，记录 padding、SMEM、寄存器、
   真实 occupancy、容量和时钟。按实际 TP/DP 布局计入必要 staging/collective。
3. 多层/整模型：扩大权重工作集，再用目标服务/RL 的实际 batch 分布复测。
   单层 warm-cache 的优势不能直接乘层数推算 rollout 收益。

以上是推广验收建议，不是已经完成的其他模型性能实验。本次只整理证据和
文档，没有更改 kernel、重调阈值或启动新的 GPU 测试。
