# WeftMoE Static Forward V4：静态任务与有界流水设计

## 2026-10-02：Backward 私有 dR 与 Return ring 实施约定

本轮保持训练/推理的 Forward 主 kernel 和元数据容量布局不变，不做元数据瘦身。
Backward 保留完整 U/dAB，以及所有本地 dX 完成后才开始的完整 dW 归约。

- [ ] M7a：通信入口保持稳定地址，目标 GPU 自行将 dY 收入按实际 R 申请的私有
  dR；dR 与 U/dAB 同属本次 Backward workspace，不导出 IPC、不跨调用保留。
  节点内复用已导出的输入平面；跨机 RX 在所有实际 fanout 消费完后归还 credit。
- [ ] M7b：Return 改为与 Forward 相同的 source-window 准入、逐 slot generation
  与消费 credit；仅通信环常驻，移除全量 dX_rows/source contribution 临时量。
  dGate 随 route 发布并写回最终输出，不引入 dW split-reduce 或梯度原子加。
  窗口推进必须保证当前窗口的所有专家路由可完成，不能在未来窗口等待时饿死当前窗口。
- [ ] M7c：EP8 单机/EP16 双机、1620 MHz；随机不等长 8K/20K 正确性与相邻性能
  对照，补充空接收 rank、热点迁移、小 ring、live tapes/OOM 重试和 suspend 释放。
  分别记录实际峰值显存、常驻通信容量、初始化与稳态耗时；每个完成里程碑提交。

更新：2026-09-29。本版整理已确认的设计，作为后续实现依据；**第 0 节描述主路径，
第 11 节跟踪实施进度，其余章节是目标合同，不表示 ring、通信/计算流水或 training
V4 已经落地。**
旧设计中的 source-push、完整 arena 作为最终容量方案、按 fragment 数维护通用
引用计数等选择，不再作为实现要求。历史实现和性能证据保留在：

- [长文和多机优化历程](static_forward_v4_prefill_optimization_history.md)：已保留改动、分步收益、失败实验与证据边界。
- [单机 decode 优化历程和适用范围](static_forward_v4_decode_optimization_history.md)：短文分支、SGLang 接入及跨模型推广限制。
- [单机优化记录](static_forward_v4_optimize_record.md)
- [多机优化记录](static_forward_v4_multinode_record.md)
- [prefill GEMM/epilogue 优化记录](static_forward_v4_prefill_record.md)
- [ring 实施与优化记录](static_forward_v4_ring_record.md)
- [单机 decode 专属实现与验收](static_forward_v4_decode_record.md)

## 0. 当前代码基线：已实现什么

默认 V4 仍保留单机/多机 inference-only 的静态 full-arena 实现，默认 training 走 V3。
opt-in ring（`WEFTMOE_STATIC_RING_TOKENS=B`）已接通单机及多机
Dispatch→pull→X→FC1→S→FC2→C→Return→contribution→Reduce。多机按命中节点
去重发送 X，逐 packet ready/credit 推进，可与计算重叠；第 6.6 节描述当前
数据面。完整 EP8/EP16 数值测试已通过；2026-09-29用户已确认GA+FZ的8K/20K
inference领先V3符合预期。单机EP8/T1/T64接近或领先MegaMoE的收尾条件已由
HF专属decode实现满足（仅单机对照，见decode记录）。正式矩阵和后续消融见ring记录，
HG随后将专家前缀和改为分段warp扫描，验证Qwen E256/K8的单机EP1/EP2/EP4/EP8；
SGLang用独立短容量池承接decode图、保留prefill大池，并按层共享各bucket的packed W1。
算子、图/卸载与完整服务检查见decode记录及AReaL接入文档；RL端到端对照尚未完成。
下文按实施顺序保留各阶段当时的结果。纯GEMM布局/profile及入口选择已完成；训练对接
已接通，完整门禁仍未完成，见下方M6及训练记录。FC2 直接写预留的本地 contribution 或同节点 rail 的 Return
TX；已删除没有读者的 C payload，保留 FC2/returned 生命周期账本。远端 TX
须先取得双 credit 并发布 reservation，destination 才可准入相应整窗口。
DS进一步让远端Return直写source注册区中的expert-major contribution窗口，
删除Return RX payload及scatter。packet控制slot在ready发布后退休，payload仍
由全部token的Reduce credit保护；两种复用边界不能混同，详见6.6的DS说明。
M6d/M6e已接通opt-in **单机/多机训练ring Forward**：独立plan→实际tape分配→main，
canonical checkpoint接既有Backward；多机复用有界Backward GIN ring，保留独立
控制状态/counter区间。EP8/EP16长文热点、live tapes、OOM重试已验证，network
decode/prefill的rank0/15 memcheck及Forward racecheck均0。M6e的EP8/EP16四规模
inference正式回归通过且性能持平。M6f随后接通Notify精确P_actual及Backward
私有workspace分配，EP8/EP16主矩阵、热点迁移、OOM重试、memcheck及非ring
回退通过；完整训练sanitizer/性能门禁仍未完成，长序列inference显著领先目标
也尚未满足。

M6g补齐ring训练的unified Backward配置，并让Backward独立选择sender配额，
不再误读静态Forward的零legacy-sender配置。EP8/EP16数值、小ring热点、
memcheck及inference正式矩阵回归通过；20K Backward从原split的5707/14571us
恢复到4917/6874us，接近冻结V3。训练Forward长文性能门禁仍未关闭，详见训练记录。
M6h复用已退休的FC1 B staging合并G/U checkpoint写回，不增加共享内存或barrier；
20K训练main单机约改善1.5%、双机约0.7%。数值、热点复用、memcheck、Forward
racecheck及inference正式矩阵通过，不能据此关闭上述整体性能门禁。
ES将网络Forward逐包本地completion通知改为owner私有的已完成发送prefix，
仍保留远端逐slot credit和GIN显式本地完成。数值/复用/sanitizer门禁通过，
20K inference相邻对照再改善约0.6%–0.7%，训练Forward在反向对照中持平；
双机20K相对V3仍只快约2.5%，整体性能目标尚未关闭。具体合同见6.4。
ET随后将验收profile的Return Qmax32改为16，完整双机门禁通过，8K相邻对照
约改善1.2%、20K正反向约改善0.4%–0.5%，并少约16.7 MB/rank。训练Forward
持平；双机20K相对上述V3参考约快3.1%，仍未关闭整体长文性能目标。
FJ2将prefill FC1改为三级、FC2仍两级，复用192KiB arena；GEMM切换时用已有
empty barrier退休两CTA旧操作数，FC2 scratch避开两种A extent。完整EP8/EP16
数值、复用、Forward sanitizer门禁通过，单机8K相邻改善约0.8%，双机及20K
新增收益很小。训练20K main约回退0.3%–0.7%；单独恢复训练两级FC1未消除
回退，未保留额外分支。详见7.3及ring记录FJ/FJ2/FJ3，整体性能目标仍开放。
FM随后把SavedX复制从Pull第二次S2G移至FC1 Ntile0的ready A stage，沿用
已有empty/FC1完成链保护读者，不增加barrier、buffer或metadata。EP8/EP16
完整数值/复用及Forward专项sanitizer通过；20K训练main单机改善约2.8%、
双机约3.1%–3.6%，但单机相对V3仍慢约12%–13%。这是训练checkpoint优化，
不把inference的持平回归算作收益。当前SavedX生命周期见第5节及训练记录FM。
FO进一步将每task不变的SavedX目的指针移出K循环，沿用同一复制及释放链；
完整EP8/EP16门禁通过，单机20K训练main正反向约快2.8%–3.0%，双机收益更小
且有波动。Inference SASS不变；窗口容量独立消融见训练记录FP，未改默认配置。
FR3进一步仅对**单机训练**采用稳定source行序及M32 SavedX TMA；网络保留
FO的round-robin物理行序和向量写回。单机训练main相对FO改善4.72%/6.06%
（8K/20K），pool不变；全拓扑FQ+FR曾使双机B1024/8K退化3.79%，因此没有
保留网络分支。FR3单/双机集成门禁通过，decode/prefill全部入口及planner均与
对应已验证组件逐字节相同，专项sanitizer明确复用旧证据；双机最终训练及
inference性能回归持平。当前checkpoint合同见第5节，实验分项及撤回理由见
训练记录FQ–FR3；训练端到端与双机20K inference显著领先目标仍未完成。

FS2保留两种GPU planner，调整为Notify→实际tape分配/全rank失败确认→
物理ring plan→main；删除不参与分配的ring runtime D2H与host同步。GPU指令
与FR3逐字节相同；单/双机完整数值、OOM重试及Forward专项sanitizer通过，
单机长文wall正反向均有约1%–2%的收益，双机长文正反向差异约±1%，记为持平。第8节
按这一已验证的准备边界列出流程；不将被撤回的FS整合planner方案混入当前实现。

FT把ring完整M128的FC1 S写回改为TMA，复用原有output descriptor及CTA私有
B staging；decode/尾块/nonring保持原路径，训练G/U仍按canonical行写回。
所有issuer执行完整bulk wait0而非.read，再经原有math join/fc1_done发布，
不提前释放B。单/双机完整数值、复用及Forward专项sanitizer通过；单机20K
相邻正反向约改善0.3%–0.5%，双机及训练持平。详细证据与候选消融见ring记录FT–FV，
不据此关闭双机20K显著领先目标。

GA/FZ组合进一步将逐slot退休确认合并为channel consumed-prefix ACK，并将
SavedX交给轮转的FC1 pair、每CTA复制半个M。没有删除远端消费或本地NIC完成
条件，也不增加checkpoint读者/barrier。单/双机完整数值、复用及Forward专项
sanitizer通过，组合正式矩阵及短档反向复测完成并采用。EP16短档相对FT下降
约12.2%/9.6%，长文仅小幅改善；GA+FZ长文在09-29已获用户确认，短文MegaMoE
对照仍待达标。当前布局和同步合同见第5节及6.4，独立消融/组合证据见ring记录GA/FZ。
随后GR/GS减少单机单容量窗口的planner grid join，并将source expert/node前缀
改为warp scan；没有更改逐slot同步合同。详细适用范围、收益及回归见ring记录。
GU再按实际destination工作量选择单行交错pull，改善短文静态issuer分布；
单机1/64 token相邻正反向约快4%–5%/11%，但同输入ring仍明显慢于MegaMoE。
这不是新增动态任务领取或非ring自动回退，具体分工见第5节。
GV再复用已有destination表发布单容量窗口的source counts/route IDs，减少
目标端串行peer读取；单机短档相邻正反向再快约1%–1.7%。X仍destination-pull，
不改变有界payload及逐slot协议，也尚未满足MegaMoE短文收尾条件。
GW复用source的uint64 warp scan计算destination Mblock前缀，消除thread0逐expert
串行除法/累加；保留原CTA同步与运行时tile_m，短档再快约1%，详见ring记录。
GY再将少量工作、单容量窗口的reverse metadata并入CTA0几何规划，同一个
grid join发布blocks和rows，短档正反向约再快3.8%–4.0%。pull及逐slot合同不变；
把reverse metadata延后到pull的GX候选反而退化，已撤回，未混入当前实现。
GZ仅对单机inference省去无读者的node目录生成，保留expert计数/route列表和
原同步点；短档相邻正反向再快约8%。多机节点去重/fanout和training目录不变，
当前同输入ring仍慢于MegaMoE约50%/58%，详见ring记录。
后续HA提前单窗口admission没有实质净收益，HB集中source staging到CTA0虽省去
一次grid arrival却使T64慢约31%，两项均已撤回；不属于现行同步方案。
HC进一步在单容量窗口直接构建destination counts/routes，复用通用route
遍历/校验，省去中间source目录及转存；T64再快约0.8%，T1持平。全部门禁
通过后采用，最新同输入ring仍慢于MegaMoE约51%/57%，短文目标未关闭。
HE进一步把单容量窗口的route/gate合并发布到destination，省去reverse中的
peer gate读取；完整单/双机、训练与复用门禁通过后采用。native短档改善约
1.6%–2.4%，最新同输入ring仍慢于MegaMoE约49%/52%。用户要求下一里程碑独立
实现单机decode的通信/计算/同步工作流，参考MegaMoE并验证batch invariance；
先解决宏观流程，不以MNK参数扫描代替该重构，prefill和多机维持已验收路径。

HF已完成这一独立里程碑：新增`local_decode::kernel`，固定三条pull warp、
一个TMA producer及两个math WG；去掉短容量内不需要的通用窗口规划、
admission和return-fragment目录，FC2直接写source contribution。保留
BeforePull/BeforeCombine/AfterClean边界及SMEM full/empty流水，沿用Weft
数值合同；batch大小、行序及专家热点变化的逐位一致性通过。单机inference
短容量pool（max_tokens<=64、一个容量窗口）自动选择此入口；大容量pool、
多机、训练和非ring保持原路径。1620MHz同输入两轮1/64token均比Mega kernel-only
低约2.9%–4.0%；EP8/EP16四规模回归通过，长文及多机差异均小于0.5%。
完整布局、同步证明、收益及使用边界见[decode记录](static_forward_v4_decode_record.md)。

HH 根据真实 SGLang TP4/EP4 的回退进一步调整专属 decode：融合 graph 输入
staging/输出回填，补齐行复用真实 route，稀疏输入用 M16 warp MMA，密集输入
用 M64 WGMMA。两种算术片共用静态调度、三条 pull warp、一个 TMA warp 和
128 个 math threads；均使用 CTA-local A/B TMA full/empty credits，不做
decode cluster multicast。物理块仍为 M128，热点遍历全部 slice，不截断行。
采用的 HHD（`5d4c374`）已完成 EP1/2/4/8、batch invariance、sanitizer 及
单机 MegaMoE 对照；通用 V4 的 8 个 ring kernel 机器码不变。
RL 回退及相同步数复测另见 [AReaL 记录](areal_qwen35.md)，不把算子提速当作
端到端结论。HHD 保留上述三次 rank collective；随后 HHE 仅在单窗口 decode
省去无清理工作的 AfterClean：BeforeCombine 退休旧 source 读者，下一次
BeforePull 在新 FC2 写入前确认所有旧 Reduce 结束。EP1/2/4/8、sanitizer
及人为 rank 延迟复用检查通过；通用 ring 不变。独立消融及证明见 decode
记录，正在运行的 HHD RL 二进制不包含这项后续收益。

| 项目 | 默认 full-arena 路径（非 ring） |
|---|---|
| 单机 Dispatch | source staging 后，expert GPU 的固定通信 warp 通过 NVLink destination-pull |
| 多机 Dispatch | source 将完整 X/routes/gates 发给每个远端节点的同 rail GPU，包括未命中的节点；完成阶段 barrier 后再机内 pull |
| GEMM | FC1/FC2 静态 residue 分配，无动态 task claim；节点内 pull 与计算可重叠 |
| cluster TMA | 已使用 cluster size 2 的 multicast；V4 配对同 Mblock 的相邻 N tiles，共享 A |
| FC2/Return | FC2 push contribution；多机等节点内全部 FC2 完成后发送 Return，再等待网络阶段结束后 Reduce |
| 缓冲区 | canonical X/S 和通信区按完整 invocation 容量分配，没有应用层逐 slot 通信/计算 ring |
| inference launch | route planning 与主流程融合；正式 graph 基线每 rank 一 kernel |
| training | 非ring保持V3（须对应编译profile）；opt-in训练ring见M6d/M6e |

当前通常使用 132 个 384-thread CTA：256 math、32 TMA producer、96 pull/control
线程。此配置是基线，不是新流水必须保持的角色配额。

prefill 配置 `configs/static_inference_prefill_20260923/` 已有两个 GEMM entry：

- 短序列：M128，FC1 N64 / FC2 N128，K128，stages 3/3。
- 长序列：M128，FC1 N128 / FC2 N256，K128，stages 3/2（FJ2；历史测量为2/2）。
- 该配置以 8192 tokens/rank 为入口阈值；其他配置未统一启用，不把阈值当成最终结论。
- FC1 宽入口已迁移合并写回；窄 N32 SwiGLU 输出保留直接写。
- epilogue 复用已退休的 CTA 私有 B staging，不复用可能被另一 CTA 提前 multicast
  覆盖的 A staging。

ring 验收配置 `configs/static_inference_ring_20260924/` 沿用上述两组 GEMM，
都用 packed gate8/up8，依据当前通信版本的正反向对照将入口阈值调整为单机
512、多机2048 tokens/rank。单机/双机在六个规模及edges的完整数值检查通过，
四个指定规模已正式计时；阈值并非任意模型或三机以上的最优值。ring仍显式
启用，未改变默认非ring训练路径。见ring记录DC，不能将profile选择与通信改造的收益合并归因。

通用V4路径的decode/prefill是`unified_forward_kernel<..., Prefill, ...>`分别编译的
CUDA入口，不是同一个kernel内运行时修改GEMM参数，也不是两份复制的实现。
host在launch前选择入口及对应tile尺寸的TMA descriptor；两个入口共享同一份
packed gate8/up8权重和ring协议/调度模块。窄入口FC1输出N32，采用直接写回；
宽入口输出N64，完整M128块可复用已退休的B staging做合并/TMA写回，尾块保留
直接写。每次inference选择一个融合Forward main，不是先后启动decode和
prefill两个kernel。`RetainBackwardState`是另一个独立编译维度；训练ring
额外有planner/tape准备，Backward仍复用已验证的V3实现。
HF另有真正独立的单机短容量decode kernel/工作流，已不只是上述模板profile
的区分；它复用数值、地址和collective模块，不影响这里的通用入口。

full-arena 历史基线的 graph rank-max p50（us），核心 1620 MHz：

| 拓扑 | tokens/rank | 冻结 V3 | prefill 配置 V4 |
|---|---:|---:|---:|
| 单机 EP8（1×8） | 1 | 299.78 | 64.50 |
| 单机 EP8（1×8） | 64 | 368.44 | 73.80 |
| 单机 EP8（1×8） | 8192 | 1487.58 | 1504.50 |
| 单机 EP8（1×8） | 20480 | 3289.72 | 4607.09 |
| 双机 EP16（2×8） | 1 | 680.49 | 169.99 |
| 双机 EP16（2×8） | 64 | 824.51 | 188.90 |
| 双机 EP16（2×8） | 8192 | 3355.23 | 3725.31 |
| 双机 EP16（2×8） | 20480 | 7105.66 | 9146.10 |

V3 是同日冻结对照，不是最后一轮 prefill 改动的交错重测。长序列差距仍在；
现有数据同时混合了布局、tile、调度和通信差异，不能证明 gate/up 的某种布局更优。
此表不是新 ring 的性能结果；详细配置、误差和证据等级以各优化记录为准。

## 1. 本轮确定的目标与边界

1. 保留静态 task ownership，删除动态队列领用、全局 claim cursor、task CAS。
   静态指本次 route plan 冻结后的映射，不是编译期固定 token 数。
2. 使用逐 slot、带 generation 的 ring；借鉴 MegaMoE 的 full/empty 计数协议，
   而不是移植其完整 scheduler。不能仅把完整 arena 地址取模就称为 ring。
3. 单机仍使用 destination-pull；多机跨节点用 GIN push 到 rail RX，目标 expert GPU
   再做节点内 pull。按实际 route 做节点级 X 去重，不命中某节点就不向它发送 X。
4. 跨机 Dispatch、节点内转发、FC1/FC2 要流水重叠；Return 也随已完成 Mblock
   推进，不等整个节点 FC2。数据面不使用每包、每窗口或每阶段的全节点 barrier。
5. X/S/C、网络 TX/RX、source contribution 分别管理生命周期。释放某一缓冲区
   不等待整条流水结束；后续需要的 metadata 必须仍有独立有效存储。
6. 支持长上下文 prefill 与后续 training；不以 capacity factor 截断合法 route。
   有界 transient buffer 满时背压，不丢 token。输入、输出、plan、training tape
   仍需实际规模的内存；真正 OOM 明确失败，ring 不等于无限总显存。
7. 单机/多机、decode/prefill 使用 specialization/profile，共享协议和数值模块，
   不复制四套 scheduler。gate/up 布局通过第 7 节独立 GEMM 消融决定，先不选边。
   M/N/K、stages、CTA 配额和通信配置由统一 autotune 适配形状、负载及拓扑，
   不为每个模型手写一套实现；当前经验配置与后续自动选型的边界见第 7.4 节。
8. Dispatch/Return TX 复用先保留远端消费 credit 与本地 Put completion 双重检查；
   简化检查列为后续优化，不阻塞首版。
9. 协议参数化节点数，不硬编码两节点。当前实机主矩阵只有单机 EP8（1×8）和
   双机 EP16（2×8），均测 1/64/8192/20480 tokens/rank、核心 1620 MHz。
10. 本轮先完成 inference 流水；training 的保存、精确分配与 Backward 对接作为
    独立 milestone，不能把 inference 正确当成训练支持完成。

## 2. Route plan、窗口与静态任务

### 2.1 不可变的路由身份

每条有效 route 用 `(source_rank, token, topk_slot)` 唯一标识，另有 expert、
router weight 和有效位。重复 expert 的不同 top-k slot 仍是两条 route；只有
X 的网络传输可去重，不能去重掉 contribution。

plan 给出 source ordinal、destination physical Mblock/row、source contribution
位置及固定 combine order。物理行序与训练tape的canonical行号分开：inference
及network training保持source-rank round-robin；local training可使用稳定的
source-major物理行以改善checkpoint写回。tape仍沿用Notify冻结的expert/source
逻辑行号，不在Backward重做原子分配来猜行号。

最终 FP32 combine 保持现有 `(expert, topk_slot)` 固定顺序，不改成到达顺序，
也不能擅自换成单纯 top-k slot 顺序。router weight 在 SwiGLU 阶段使用。

### 2.2 有界 source-token window

同一 token 的全部 top-k route 进入同一个窗口。窗口最多包含 G 个 source rank，
每 rank 最多 B 个 token；大 world 按固定 source-rank group 继续枚举窗口，
不让 G 随节点数无限增长。起始候选是 G×B 不超过 2048 个 source token：
EP8 可取 G=8/B=256，EP16 可取 G=16/B=128；这些是调优起点，不是性能结论。

BO 后的实现将 G 选为 pool 的每节点 rank 容量中不超过 16 的最大因子，
实际 world 与容量一致时 source group 不跨节点；8-GPU 节点的 EP8/EP16 都是 G8。
launch 继承这个 G，不能重新按 world 计算。若 pool 容量大于实际拓扑、导致
group 跨实际节点边界，原 Pull 的逐行 local/remote 过滤仍然适用。网络路径的
本地 batch 保持两行，单机默认八行；GU为少量工作的一窗口inference采用单行
交错分工，见第5节。BP 的整窗过滤与网络八行 batch 无独立
收益，已撤回。完整 global completion 后发布 X credit 的合同不变。
B512/B1024 等参数继续按拓扑和规模实测选择，不改变有界最坏容量公式。

所有 rank 使用同一窗口编号与顺序；不等长/空 rank 由实际 token counts 决定空区间。
窗口里的 route 按 expert 形成逻辑 Mblocks，不能把一个 Mblock 的未到齐行留给
未准入的后续窗口。一个 packet 也不跨窗口。

需要的几何是：

```text
count[w, source, expert]           // 当前窗口内的实际有效 route 数
R[w, expert] = sum_source count[w, source, expert]
mblocks[w, expert] = ceil(R[w, expert] / M)
```

这些计数从实际 route 遍历/分桶产生；整个 invocation 的 expert 总数不能代替
窗口计数。由 planning 阶段交换必要的窗口计数和紧凑目录，零项可以省略。
避免强制物化规模为所有窗口 × 所有 rank × 所有 expert 的稠密表。

窗口划分、route 顺序与 packet 边界在 payload 阶段前确定。descriptor 可按闭式
规则枚举，不要求全部物化；不能用运行时“当前 ready 多少就发多少”改变首版
分包边界。每个 channel 的准确 packet 数由实际非空分组及分包规则得到，而不是
仅用全局 route 总数除以 Q。

plan/紧凑目录还需让目标 GPU 枚举每个 (channel, sequence) 的实际 fanout 任务范围。
不命中的 worker 不扫描可复用 RX header 来临时发现任务；否则这些未被 fanout
计数覆盖的 metadata 读者也会影响 slot 释放。ACK owner 所需包头字段在返还 credit
前读完，之后不再引用该代 payload。

### 2.3 准入与无环背压

source owner 先预留自己参与窗口的 contribution 容量；destination owner 一次检查
并预留本 rank 该窗口全部 Mblocks 的 X/S/C，然后各自发布准入 credit。禁止在
destination 内占住部分 X/S/C 后等待本窗口剩余 plane。M3b 采用分级准入：发送方
等待自己的 contribution reservation 与 TX 双重 credit；有界 RX 可以预取尚未
获准计算的窗口，真正 pull 到 X 前才等待本 destination 的整窗口准入。因此无需
额外跨 NIC 交换每个 destination 的窗口 credit，也不做全节点窗口 barrier。

容量推导（M=128，K=top-k，E_local 为本 rank expert 数）：

```text
R_window_total <= G * B * K
mblocks_on_one_rank = sum_local_expert ceil(R[w,e] / M)
                    <= ceil(G * B * K / M) + E_local
D_window = ceil(G * B * K / M) + E_local
```

每个 X/S/C ring 至少能容纳一个最坏窗口；同时完整准入两个最坏窗口的保守容量
为各 2×D_window。实际准入使用真实 mblock 数。不同 plane 可独立退休，但不能
因 X 提前空出就越过 S/C 的容量限制。source contribution slot 必须在发出该窗口
Dispatch 前预留，否则 Return 可能反向等待仍未分配的目的地。

固定 owner 的 alloc/reclaim 序号只负责容量账本，不是动态 task queue。全局 head/tail
不是 readiness 依据；若账本用连续退休前缀，只能越过所有相关 slot 均已退休的区间，
不能把某个乱序完成 slot 当成之前所有 slot 都已完成。

同一窗口内可保留各 cluster 先 FC1、再 FC2 的静态枚举，但必须先保证整窗口
S 容量；不能在过大的 wave 里让 FC1 等 S 空位、FC2 又等该 wave 的 FC1 枚举结束。
通信/计算只按已准入窗口的有序任务推进，后续窗口不能占住前一窗口释放所需的资源。
具体角色配额、分级准入和等待图在实现 milestone 中验证，不能仅凭 ring
容量公式宣称已证明整条网络流水无死锁。

## 3. Buffer 布局与地址

### 3.1 数据区

以下是目标布局，不是当前 NetworkLayout 的现状；M=128，BF16 元素为 2 bytes。

| 存储 | 逻辑布局 | 分配与释放 |
|---|---|---|
| 输入 X/routes/weights | 调用方实际输入及必要的 peer 可达 staging | 本次调用输入生命周期，不复制成每远端节点一份完整 X |
| X ring | `[D_x, M, H]`，另有 gate/本代元数据 | 本 GPU，所有 FC1 读者退休后复用 |
| S ring | `[D_s, M, I]` | 本 GPU，所有 FC2 读者退休后复用 |
| C / FC2 输出 | 首版 `[D_c, M, H]`；CN/CO 后不分配 payload | 保留逐 slot 的 FC2/returned 账本，输出直接写已预留的 contribution/TX |
| Dispatch TX / RX | 分离的 `[remote_node, lane, D_net, slot_bytes]` | 本 rail 的注册区，逐slot退休、channel consumed-prefix credit |
| Return TX / RX | 同样分离的 peer/channel ring | 不与 Dispatch 或对向 TX 别名 |
| contribution | source 上至多 `D_contrib×B×K×H` 及 route-ready；多机每窗expert-major，单机token-major | 窗口全部 token 的 Reduce 完成后释放 |
| immutable route plan | 实际路由/窗口几何及返回映射 | invocation；训练延长到 Backward |
| training tape | 实际大小的 X/G/U 等 checkpoint | 不放在会覆盖的 transient ring 里 |

每种 plane 有独立地址空间。逻辑 block b 映射到 `slot=b%D_plane`；读取/写入前
同时核对本代描述和 full/empty 条件。单凭取模地址不授予访问权。

通信 chunk 起始候选 Q=32/64 个 token 或完整 contribution 行，D_net=4，
每 peer 2/4 lanes；不在实现前把这些值定成最优。Dispatch slot 最坏容纳
Q份X；Q×K个fanout的身份和计数来自独立plan，不再重复放进数据包。

### 3.2 Packet 与返回地址

逻辑Dispatch packet的元数据如下。M3b以后这些身份、地址和消费数已冻结在
独立source/consumer目录，不要求在每次payload Put中重复传输：

```text
plan: window, token_count, fanout_count
token:  source token / X offset / fanout range
fanout: destination rank / logical Mblock / row / topk slot / router weight
slot control: ready / credit generation, consumed_total, retire_target
```

source rank可由channel得到。CH保留128-byte对齐header和完整X行；CL
进一步只发送完整X行，RX owner从已发布的incoming plan取expected，在导入
GIN ready后累加本地slot.retire_target，然后发布visible_generation。
generation由逐slot的GIN signal给出。CL已通过完整数值、wrap及sanitizer门禁，
见ring记录；control与大payload分离并按实际原子/NIC/TMA要求对齐。

路径和实际地址是：

```text
source X
  -> source rail Dispatch TX[peer,lane,slot]
  -> target rail Dispatch RX[source_node,lane,slot]
  -> target expert X[x_slot,row,:]
  -> S[s_slot,row,:]
  -> FC2 直接写 compute node rail Return TX[source_node,lane,slot]
  -> GIN 直接写 source registered contribution[window_slot,expert_ordered_row,:]
  -> source output[token,:]
```

same-node Dispatch 直接从源 peer 可达 X pull 到 expert X ring；same-node Return
直接写 source contribution。返回标识必须能恢复 source window/token/top-k slot。
以上地址为 DS 后的实现；首版曾经由 C 再搬到 TX，及 RX 再scatter。不可变 RingRow 中预绑定
FC2 output 地址，不把可复用 TX payload 当成路由目录。

不能把 Return 唯一的反向映射放在已释放的 X slot：优先从 immutable plan 查询；
若把必要字段复制到 S/C/packet，它们必须在前一存储释放前完成复制。

### 3.3 内存界限

transient 数据容量由 D_x/D_s/D_c、G/B、channel 数、Q 和 ring depth 决定，
不随总 token 数 T 增长；通信资源仍随实际 peer 数增长，不声称与节点数无关。
plan、输入、输出以及训练 checkpoint 的总内存随实际 T/route 数增长。

热点 expert 超出一窗口时分多个窗口完成，而不是要求整次调用的热点 route
全驻留，也不以 overflow 后丢 token 换取固定容量。资源不足只能准入前报错/OOM
或等待已有窗口退出，不在主流水里依赖 host 临时扩容。

## 4. 逐 slot 协议与计数

### 4.1 通用规则

每个 slot 的 full/readiness 与 empty/consumed 是不同方向的状态，不能共用一个
含糊的 done。ready 表示本代数据可读；empty 表示本代所有读者已退休。
每个逻辑任务固定属于一个 worker，generation 标识代际，完成计数只负责同步，
不负责分配任务。

变长消费采用以下概念字段（不是已冻结的 C++ ABI）：

```text
slot persistent state:
    done_total                    // uint64，消费者累计完成数
    ready / empty generation      // 按 transport 选择发布机制

generation descriptor:
    expected_count                // 本代只读工作量，不做运行时原子增长
    retire_target                 // 该物理 slot 上一代 target + expected_count
```

唯一发布 owner 维护上一代 target，新 target 与本代描述在 ready 之前发布。
消费者先 acquire 本代 ready，完成全部异步读取，再 release-add 完成的单位数。
唯一回收 owner acquire 看到目标达到才发布该代 empty/credit。不同 generation
不能共享尚未结束的读取；即使 expected=0，也不能跳过本代 ready 判断。

同一个 slot 连续两代 expected 为 7、3 时，done_total 从 0 到 7 再到 10，
targets 为 7、10，不在换代时清零。跨调用可保持累积；重置只能发生在所有使用者
及通信请求 drain 后的初始化/重建边界，不能在 graph replay 中与旧操作竞态。
固定消费数 F 的情形可简化为第 g 代 target=(g+1)×F；不同 profile/F 混用时用累积
target，不能继续盲用 g×F。

一个 worker 可完成同一 (slot,generation) 的 n 项后只做一次 done+=n；
同一批涉及多个 slot 时分别记账。只有一个消费 owner 的 slot 可直接发布 empty，
不为了统一接口强行增加共享计数。跨 GPU 使用合适的 system-scope，同 GPU 用
device-scope；TMA/网络操作先真正完成，再走对应的 release/acquire 可见性链。

### 4.2 expected_count 的生成步骤

不新增“fragment 引用图”，也不让接收方猜 expected。来源如下：

| 被保护的数据 | expected 的单位与来源 | 谁生成 |
|---|---|---|
| Dispatch RX chunk | 本 chunk 所有 token 在目标节点上的有效 fanout 总数 | source packet planner |
| C Mblock | 当前窗口内本 block 的有效 route 行数 | destination geometry owner |
| Return 接收控制slot | packet 内完整 contribution 行数；DS在这些行ready发布后退休 | source Return packet planner |

Dispatch在数据面开始前，从实际route及固定分包规则计算每packet计数：

```cpp
expected = 0;
for (token : packet_tokens) {
    for (k = 0; k < top_k; ++k) {
        if (valid(token, k) && node(expert(token, k)) == destination_node) {
            ++expected;
        }
    }
}
```

CH实现为每warp静态拥有一个packet，按目的GPU聚合consumer_rows并归约fanouts；
写入immutable DispatchPacketPlan，不在pack阶段重新推测或原子增长expected。
已有 offsets 时，expected=Σ(offset[p+1]-offset[p])，p 仅取本 chunk 的 token
records。三个 token 分别有 2/1/3 个 fanout 时，expected=6，不是 token_count=3。

C block 的 expected 为 `min(M, R[w,e] - block_index*M)`，R 从第 2.2 节实际窗口
route 列表长度或 prefix 差得到，包括本地和远端 Return 的行。Return packet
按节点expert route interval及固定Q/lane规则枚举，直接使用ReturnPacketPlan.rows。

Return fragment 只是一次搬运的一组完整 H 维行，不是计数单位。128 行分成
40/56/32 三次搬运，就上报 40/56/32；改变分包不改变 expected=128。
首版不把同一行拆给多个独立列片段任务；若以后这么做，必须重新定义计数单位。

packet fanout已按route/分包规则预计算；fragment个数也可预计算，但buffer释放
不依赖 fragment 个数。channel terminal packet 数仍必须精确计算；它与这里按行的
retirement 计数是两个概念。所有批量记账都要求静态分区无重叠且完整覆盖。

### 4.3 独立释放点

- Dispatch RX：全部 fanout 的数据和后续所需 metadata 搬走后释放，不等 FC1。
- X：所有 FC1 N tiles 及训练需要的 X checkpoint 读取结束后释放。
- S：所有 FC2 N tiles 读取结束后释放。
- C：所有有效行连同返回标识搬到独立 TX/源 contribution 后释放，不等网络发送。
- Return 接收控制slot：全部行发布 readiness 后释放，不等 Reduce。DS无独立RX payload。
- contribution：对应窗口所有 token 的固定顺序 Reduce 完成后释放。
- TX：仍按第 6.4 节双重检查，不能把 C 的退休等同于 TX 可复用。

任何阶段的完成都必须包括其异步 TMA/peer 读者；“提交读取”“领取任务”均不是消费结束。

## 5. 单机 ring：pull 与 FC1/FC2 如何接力

参考实现位于本地 DeepGEMM-bk 的
`deep_gemm/include/deep_gemm/impls/sm90_bf16_mega_moe.cuh`：
pull 等 l1_empty，TMA store 完成后发布 l1_full，FC1 epilogue 发布 l2_full 并归还
l1_empty。这里借鉴的是逐物理 slot 的代际协议，不把当前 WeftMoE full arena 误称为 ring。

令一个 X 物理 slot 当前代为 g，完整 Mblock 为 M 行，本 profile 每个 Mblock
有 F1 个 FC1 逻辑 N tiles、F2 个 FC2 N tiles：

1. 窗口准入已预留 X/S/C。pull 先 acquire X slot 上一代 empty，才写本代行与 gate。
2. 每个 pull worker 等待 TMA store 完成后 release-add 已落地行数到 X full。
   尾块有效行 r<M，由唯一指定的尾行 publisher 多加 M-r，使本代仍贡献 M 个 credit。
   padding credit 不代表执行了虚构 route；计算/epilogue 使用 valid_m。
3. FC1 TMA producer acquire 等 X full 达 (g+1)×M，才发起该 block 的 GEMM load。
4. 一个 FC1 N tile 的所有 math warpgroups 完成 S 写回后，发布 S 的该 N tile ready，
   并向 X empty 累加一个完成单位。一个 task 覆盖多个有效 N tiles 时加实际数量。
   两个 math warpgroups 合作处理一个 M128 tile，不是各自计一个完整 tile。
5. X 的全部 F1 项完成后才能覆盖 X。S full 达到本代 F1 项后，FC2 可以读取 S。
6. FC2 每个有效 N tile 完成后发布 C full、累计 S empty。F2 项完成后允许 Return
   读取完整 H 维 C 行，同时允许复用 S。
7. Return 按第 4 节有效行数退休 C；源端按窗口完成 Reduce 并归还 contribution 容量。

这里的 full/empty 是 GM ring 状态，不能与 CTA 内 SMEM TMA stage 的 mbarrier
混为一谈。cluster multicast 的 shared stage 仍要等两个 CTA 的相应读者退休。

GV仅对单机inference、公共容量至多一个窗口的规划，复用已有
`fused_local_expert_count`和`fused_source_routes`：source CTA0完成本地目录后，
把counts与route IDs写到destination上自己独占的source plane，再经过原有
source-ready rank barrier。每次覆盖全部expert计数，包括空source的零计数；
route列表只读取本代count内的部分，故不需要新增清理阶段。destination从本地
表取counts/route IDs，epoch、gate仍从source取，物理行序与几何计算不变。
每expert/source的route extent不超过已有`max_tokens_per_rank * max_top_k`步长。
没有调用full-arena的容量拒绝逻辑；单窗口热点超过max_rows仍由ring完成。
这是metadata发布方向优化，不是把X从pull改成push；大容量、多机及training
保留原目录访问方式。跨调用terminal仍保护这些invocation级目录。

GY对上述单容量窗口进一步限定：实际windows=1且本destination有效行数
不超过`2 * gridDim.x`。CTA0保留刚生成的source/expert计数表，四个lane静态
分担一个source/expert列表的ordinal，复用`WindowGeometryView::expert_row`
生成原round-robin物理行号。block descriptor与local Return地址计算与通用
planner复用；完整blocks/反向rows经原geometry grid join一起发布，省去单独
reverse pass和其grid join。没有新增分配、计数器、工作队列或额外scratch。
这不是GX的延后生成方案：**PlanReady仍表示所有反向rows已就绪**，pull不做
source ordinal解码，FC1/FC2/Return的读取合同不变。空/错误窗口、较多行、
大pool中的短调用、多窗口、training和network仍使用原分布式reverse pass。
不同destination可选择不同本地规划分支，但均在同一个source-ready rank
barrier之后，未删除或分叉跨rank同步。

GZ在`build_source_window`复用一个编译期`NodeDirectory`选择；ring caller传
`Network || Training`。分组规划器在单机inference只需expert_prefix/route_order，因此
不再生成node histogram、node_prefix和带fanout的node_tokens。已核查这些
node字段的生产端读者仅在network packet规划和transport中。此项也适用于
多窗口及大pool里的短调用，不受GV/GY单容量窗口条件限制；保留全部expert
计数、masked/非法route处理、route预留及CTA/grid/rank同步。没有减少有效route，
也没有取消跨机按节点去重；网络和训练均保留完整目录。pool布局与分配未改。

HC在单机inference、capacity windows=1时进一步替换这个分组步骤：CTA0
的source组清零共享expert计数，复用`for_each_source_route`遍历有效route，
一次shared atomic给每条route分配source-expert ordinal，直接写入既有
destination route表；组内join后发布全部count，包括零。无需生成或转存
source expert_prefix/route_order，因为后续cached source_count和两种reverse
分支均读destination目录。空source仍发布零计数，非法expert和masked route
复用同一个校验入口；训练、多窗口及network继续生成原分组目录。输入/gate/
combine准备仍分布在所有math CTAs，原grid/rank发布边界和逐slot合同均不变。
既有目录allocation保留供其它路径使用，没有增加新状态或运行时参数。

GU仅在单机inference的实际全局窗口数为1、该destination有效行数不超过
`2 * gridDim.x`时选择Batch1，判定不依赖本rank的输入T或pool最大容量。
令窗口有n个Mblock，固定issuer以`q = worker + iteration * workers`枚举，
映射为`block = q % n, row = q / n`，再用原有valid_m过滤padding。
这只是窗口内padded行的一一置换：每条有效route仍恰好复制一次，唯一尾行
仍补齐M-r个credit，完整bulk store完成后才发布X full。空窗口不执行映射。
多窗口/较大工作量、本地training和network保留原batch及分工；没有新队列、
计数器、buffer或协议旁路。区别于GR的容量条件，大pool中的实际短调用也可选GU。

当前training的Pull只完成物理X写入及canonical identity/gate保存后发布X full。
FM最初由每个Mblock唯一的FC1 Ntile0，从ready A stage保存canonical SavedX。
当前FZ改为唯一静态pair：`n_tile/2 == mblock % (fc1_ntiles/2)`，避免固定
Ntile0在66个cluster/偶数pair步长下只给半数cluster分配checkpoint工作。
pair的偶/奇CTA分别保存M的前/后64行；尾块只复制各自有效行，空半块不写。
两个CTA本来就是multicast A的读者，总SavedX字节数不变，不新增读者或计数。
Network沿用FO：进入K循环前缓存目的指针，math warp先完成uint4复制及WGMMA
wait0，再向两个CTA的stage empty到达；不会在仍有stage读者时覆盖它。

FR3的local training物理Mblock属于单个expert，且canonical行严格递增。
对齐的M32 chunk若完全有效，且末行等于首行+31，就能证明整个chunk连续。
使用现有`SavedX`描述符的`{64,32}` box写这些chunk，其余行仍由FO的uint4
路径写入；两种写入互斥。K128 stage中的两个N64 swizzle atoms按M128分隔，
第`chunk`段、第`atom`块的BF16元素偏移为`atom*128*64 + chunk*32*64`，
不能把它误当成普通row-major K128。无需额外tensor-map数组、GM目录或计数。
FZ的每CTA view先偏移`row_begin*64`，canonical映射也偏移row_begin；即使
view只有M64，K64 atom间距仍是`128*64`，不能缩成`64*64`。每CTA只保存
两个M32 chunk的状态及四个K128向量目的指针，原连续性判断不变。

TMA在A-ready acquire及WGMMA commit后发出，每stage提交一个bulk group。
WGMMA wait0退休矩阵读者；TMA issuer再等待**完整global completion**，随后
第一warp执行join，才让lane0/1分别向两个CTA的empty到达。当前A由CTA0
multicast；该显式join不依赖隐式warp收敛，也没有新增跨CTA barrier对象。
不采用仅`.read`等待或将完整完成推迟到task末尾的FR2候选。
SavedX完成仍先于各自tile的FC1发布，因此已有F1释放链涵盖两个checkpoint owner，
不需要独立读者计数。映射在整个task内不可变，valid-row及hidden-tail检查保留。

FC1在相应S slice ready前完成saved-G/U写回。SavedX/G/U属于独立tape，不随
ring slot回收；Backward在Forward完成后才读取。若以后引入独立checkpoint
reader或其它异步store，必须重新纳入相应stage/slot释放条件。

各 cluster 在准入窗口内静态枚举自己的 FC1/FC2 residue，不引入全局 FC1→FC2
barrier；某 block 的全部 S 就绪即可开始 FC2。窗口容量/进度约束见第 2.3 节。
Return 和 Reduce 由固定角色推进，不等整个 invocation 的全部 GEMM。

## 6. 多机流程

### 6.0 复审：进度、容量与首版实现边界

历史第一版因 ring 进度证明不足而采用完整 invocation arena，属于第 0 节的基线，
不是这次后续实现的目标。必须修正的三个问题仍是：

- TX 等 credit 时不能阻塞负责 RX 消费/返回 credit 的角色。
- 全局 expert 总 count 不能推出每个窗口、channel、packet 的具体几何。
- NIC 本地完成、远端 payload ready、所有 RX 读者完成要明确区分发布机制与生命周期。

新方案用有界窗口和逐 slot full/empty 协议替代阶段级通信 barrier。以下各节定义
协议，实际实现与验收状态见第 11 节；旧的完整 X 广播、BeforeCombine 网络阶段
不继承到 ring 路径。

### 6.1 Route-aware Dispatch：每命中节点一份 X

source 从 top-k routes 得到每个 token 的实际目标节点集合；同一目标节点只放一份 X，
保留全部 fanout records。未命中的节点既不发 X，也不发无意义的空 data packet。
节点集合可用不超过 K 项的去重列表，不用写死两节点或 64 节点的 mask。

packet 按 (window, destination node, lane) 的固定 token 顺序切分，每包最多 Q 个
去重 token。一个 packet 可以 fanout 到同节点多个 GPU、expert 和 Mblock；
不再强制一个 packet 只能覆盖一个 destination Mblock，否则破坏节点级 X 去重。
这些目标必须属于同一个窗口；RX 可提前暂存，写 X 则必须已准入。

每个 rail/lane 的固定发送 owner：

1. 等 source contribution reservation；目的整窗口准入由 RX→X 的 pull 检查。
2. 等物理 TX slot 上一代的远端消费 credit 与本地 Put completion。
3. 按immutable packet目录打包完整X行。M3b已交换route/token/consumer目录；
   CL进一步移除wire header。RX owner成功导入本代GIN ready后，从incoming
   DispatchPacketPlan.fanouts累加本地slot.retire_target，再发布visible_generation；
   不得取相反方向TX目录，也不在未成功import的轮询中重复累加。
4. 等所有 packer 实际写入完成并导入 system visibility，再发一次绑定 ready signal 的 Put。
5. 按静态 sequence 前进；不使用 channel/task claim CAS。

首版一个 packet 对应一次 Put；若拆成多个 Put，必须先定义覆盖全部 pieces 的
ready 聚合，不能用最后一个 weak signal 推断其他操作完成。不同 GIN context
之间没有隐含的顺序。slot ready 使用绑定该 payload 的语义；不靠全网 flush
建立每包可见性。AU 将单 Put 的 ready 改为 `WeakVASignalAdd`，只要求该 Put
完成，不承担同 context 其他 packet 的完成通知；接收仍通过 `readSignal` 导入
payload 可见性。远端 consumed credit 与本地 NIC completion 两项都保留。

跨机走 source rank 对应的同 rail GPU，进入目标节点本地注册 RX。
节点数循环、offset 和 terminal accounting 参数化；当前 expert 均匀连续分片、
每节点 rank 数等拓扑约束由公共接口明确校验，不静默假设 world=16。

### 6.2 目标节点 destination-pull

目标 rail RX owner acquire GIN ready 后，向机内读者发布本代 packet 可见性。
expert GPU 的固定 pull workers 按静态 fanout 分区处理自己拥有的 route：

```text
目标 rail RX 的 X/token metadata
  -> expert GPU X ring 的指定 logical Mblock/row
  -> X full 行计数
```

同一 RX slot 可以由多个 GPU 消费，done_total 放在 rail 控制区，使用满足 peer
访问语义的同步。每个 worker 在实际复制及需要的 metadata 读取结束后按完成
fanout 数批量累计；ACK owner 等第 4.2 节的 retire_target 后返还该代 credit。

不以“最后一个发起 pull 的 warp”代替全部读者完成；ready 的 NIC→rail→peer 可见性
链必须在实现时明确导入，不能在其他 GPU 上随意读一个普通标志替代 GIN acquire。

某个 Mblock 的全部有效行完成即可启动 FC1，不等当前 packet 的其他 fanout、
本窗口其他 Mblocks，更不等所有远端 Dispatch。后续包可同时被 NIC 接收。
RX packet的尾部慢reader背压其channel的静态发送序；GA只合并同channel中
已经逐slot确认退休的连续prefix，不形成跨channel、节点或窗口的全局前缀。

多机 pull 的任务遍历以 channel 的 packet sequence 为外层，再静态划分该包的
fanout；不能机械沿用 expert-major canonical row 的阻塞遍历。否则当前 RX slots
可能全装着后面 expert 的数据，而所有 pull warps 都在等尚未发进来的前面 expert，
即使 X/S/C 容量足够也会死锁。共享多个 channel 的 worker 必须继续服务能够释放
前序 packet 的任务；详见第 6.5 节的独立进度约束。

### 6.3 Return 与 FC2 流水

本节保留首版C→TX→RX→scatter的演进描述；当前CN/CO及DS分别消除了两次复制，
具体地址与生命周期以6.6补充说明为准，不能将旧RX credit当成contribution复用权。

Return 不恢复 V3 的运行时 append/claim queue。固定 owner 从 immutable plan 的
route/mblock 列表枚举，按 (window, source rank, lane) 分包；一个 packet 可跨多个
已准入的 C Mblocks，但不跨窗口。它等待自己实际覆盖的完整 C 行就绪，不等节点
内所有 FC2。首版每行等待其 Mblock 的全部 FC2 N tiles 完成，不做列片段 Return。

- same-node：静态 worker 将完整 C 行写入 source contribution 并发布 route-ready。
- remote：M3b 的 rail TX owner 先非阻塞检查 packet 的全部 C 行，再从本节点各
  expert GPU pull 到独立 Return TX；不引入额外 packer→TX ready 计数。全部复制
  完成后按完整行退休 C，再 GIN Put；math 继续处理其他 Mblocks。返回标识来自
  immutable Return目录，CL的wire同样只有完整行payload；RX从incoming
  ReturnPacketPlan.rows累加本代退休目标。
- source RX owner 收到 ready，按窗口/token/top-k slot scatter 到 contribution，
  完成复制和 readiness publication 后返还 RX credit。
- source Reduce 等该 token 的所有有效 contribution，按现有固定顺序求和。
  contribution 窗口在全部 token Reduce 完成后才能复用。

Return RX 不需要等 Reduce，Return TX 不需要持有 C 原址到网络完成。
不能为了沿用“全 Return 到齐后 Reduce”而持有一个无限长调用的全部 contribution；
在有界方案里至多按已预留的有限窗口批量 Reduce。Return/Reduce 的更细重叠可独立
消融，但不能破坏窗口回收进度。

### 6.4 逐 slot 复用决策（ES 本地完成 + GA 消费前缀，保留双检查）

Dispatch/Return TX 覆盖前均保留：

```text
上一代远端消费 credit 已返回
AND
本 slot 上一代 Put 的本地 completion 已达到
```

首版“本地 credit”使用GIN逐slot completion counter，不额外增加跨机ACK。
ES将它改为唯一TX owner维护的channel级`tx_completed_head`，没有删除
本地完成条件。GA再把原逐slot远端credit汇总为channel的`credit_signal`，
表示远端已消费的连续packet数。sequence `s`首次使用（`s<depth`）直接满足；
复用时须同时满足`credit_signal >= s-depth+1`和`tx_completed_head >= s-depth+1`。
两个head分别证明远端软件消费和显式本地完成，均不是reserve cursor，
也不拿另一context的完成替代。

对于一份 TX、一次 Put、一份 RX，消费后返回 credit 的因果链已经包含该 payload
的传输；但不代表GIN的本地完成通知此刻必然更新。仍采用保守双检查，
不把它描述为两个在数据依赖上必然独立的必要条件。

ES的具体推进方式：

- Put只携带自己的remote ready signal，不再申请逐包local counter通知。
- 若远端credit已到、上一代不在已完成prefix内，执行当前owner/context到该peer
  的`flushAsync`及`wait`，然后head推进至**已发出的**`tx_cursor`。
- Return reservation上界为`tx_cursor+depth`，因此待复用的上一代一定已发出。
  flush只等已发NIC请求，不等未来FC2/软件读者或其它channel的准入；没有持有
  共享软件锁，RX/ACK角色继续独立推进。多channel共享同一owner/context时，
  即使flush顺带完成同peer其它channel，也只保守推进当前channel的head。
- terminal确认剩余已发prefix完成，并等远端consumed head达到terminal；完全
  drain后才开始下一调用，以tx_base初始化completed_head、rx_base初始化
  acknowledged_head。GIN signal贯穿调用/graph replay累积，只在pool构造时reset。
- ES当时NetworkChannel仍64B；GA增加两个head/signal字段后为128B，见下文。
  已删除Forward无用途的counter分配/reset、counter_base和slot.local_target。
  Backward保留自己的逐slot counter。

GA仍逐slot确认退休，只合并已确认的ACK：

1. RX import取得本包ready后，将immutable incoming plan的fanouts或rows累加到
   此物理slot的retire_target，再发布visible_generation。各reader完成后按原合同
   release累加consumed_total，expected不做运行时竞争更新。
2. 唯一RX owner从acknowledged_head扫描到rx_cursor，逐slot system-acquire
   检查`consumed_total >= retire_target`；遇到首个未退休slot即停，不因其它
   slot ready而推断它已消费。扫描不会越过实际导入的包，最多覆盖ring depth。
3. 若head推进，用一次GIN signal-add向对端credit_signal累加推进量，然后更新
   owner私有acknowledged_head；不等待凑满batch。TX及reservation原本就按
   sequence顺序推进，晚slot提前返回credit也不能越过队首，所以没有新增可用
   发送顺序的限制。不同channel仍独立推进。

物理布局（均为uint64，GIN signal仅通过GIN访问）：

|对象|有效字段及byte offset|对齐/大小|
|---|---|---|
|NetworkSlot|ready_signal 0，visible_generation 8，consumed_total 16，retire_target 24|64B / 64B|
|NetworkChannel|原tx/rx base、cursor、reservation、tx_completed_head占0–63；acknowledged_head 64，credit_signal 72|64B / 128B|

Dispatch/Return分别拥有上述状态，控制slot仍逐物理slot保存。GA不增加payload
容量，2节点/32 lanes仅增加4096 bytes/rank，通常为`4096*(nodes-1)`。不能把
remote consumed-prefix当成local NIC completion，也不能把Return控制slot退休
当成source contribution窗口已被Reduce消费。

这遵循GIN显式本地完成合同，不是“收到ACK便假定API completion”。初版ES的
20K相邻/反向正式对照改善0.9%–1.1%，8K持平；清理版完整门禁已通过，20K
改善0.6%–0.7%，训练Forward反向对照未见稳定变化，详见ring及训练记录。
共享TX多目标发送或进一步合并完成范围仍须独立证明和消融。

### 6.5 静态角色与 barrier 边界

Dispatch TX、Dispatch RX/pull、Return packing/TX、Return RX/credit、Reduce、
math/TMA 都有固定职责。需要哪些独立 warp/CTA 由 occupancy 和实测决定；
不能在一个阻塞的串行函数中先做完全部 TX 才服务 RX/Return。

各个有依赖的消费者必须能常驻并推进。同一个 owner 管多个 channel 时，只能使用
非阻塞 progress 或证明无环的固定处理顺序，不能等待一个后续 channel 而停止
服务仍被前序窗口依赖的 channel。角色间不插入会把消费者一起挂住的全 CTA barrier。

保留的同步边界：

| 边界 | 粒度与触发 |
|---|---|
| PlanReady | 路由/窗口目录完成后的控制面 exchange；main 前发布 |
| 窗口准入 | 相关 source/destination 的有限容量 credit，不是全节点 collective |
| data ready / consumed | 每物理 slot、每代；Mblock full 或 packet credit |
| SMEM stage | CTA/cluster 内 TMA/WGMMA 消费者的 barrier |
| invocation 结束 | 所有静态 task/route 完成，GIN 请求及 credit drain，必要的终态一致性 |

禁止沿用“所有 Dispatch Put 后一次全节点 barrier，再开始 FC1”和“全部 FC2 后
统一 Return”的主流程。终端 drain 不得提前放到每个 packet/window 的 hot path。

### 6.6 M3b 首版具体布局与进度约束（数值通过，吞吐仍待优化）

- 每 rail 使用 `(nodes-1)×L` 个 channel，网络深度为 8；Q/L 在
  `kNetworkPackets` 集中配置，已分别实测 L=2/8/16/32、Q=8/16/32，收益与
  内存增量见 ring 记录 U/X/Z/AK/AU。当前调优为 Q8/L32/B512/D4；编译默认
  仍 Q32/L2，ring 仍 opt-in，尚未改成默认发布配置。
  Dispatch
  TX/RX、Return TX/RX 四区不别名。每 channel 有只在终端更新的 `tx_base/rx_base`
  和各 owner 私有 cursor；每物理 slot 的信号、可见 generation、累计消费数、
  RX已ACK generation均跨invocation保留。首版TX target/local completion是
  逐slot counter；ES改为上节的channel completed prefix，GA随后合并远端
  consumed-prefix ACK。两者均保留逐slot复用所需的两项证明。
- `W=min(channel_count,32)`。通信 warp 0 的前 W 个 CTA 做 Dispatch TX，其他做
  local/remote pull；warp 1 前 W 个做 Return TX、接下来 W 个做 Return RX scatter，
  其他做 local Return；warp 2 的 CTA 0 做准入、接下来 W 个做 NIC import/ACK，
  其他做 Reduce。math/producer 仍在全部 CTA 上静态分工。三类 GIN owner 分别
  使用独立的 W 个 context。TX 对 credit/C-ready 做非阻塞轮询，不阻塞其他 channel。
  AZ 将 ring Endpoint 设为 `NCCL_GIN_RESOURCE_SHARING_THREAD`：每个 context
  只有一个 lane-zero 提交者，host 已要求 exclusive contexts，控制面由原 barrier
  与 data plane 隔离。它省去共享队列争用协议，不改变 payload release scope 或
  双重 credit；若以后让多个线程共享提交，必须重新选择对应的 sharing mode。
- 节点内 FC2 epilogue 直接写 source 的 contribution slot，不再经 C 二次搬运；
  X-ready 之前已获取 source reservation。固定 local Return warp 按 Mblock
  residue acquire 所有 FC2 N tiles 的完成链，system fence 后协作发布逐 route
  ready，并按本地行数退休该 C slot。contribution 仍等 Reduce 后才能复用。
- BD 给每个 rail 增加普通 device pool 中的
  `return_ready[local_destination, physical_C_slot]` generation 镜像，跨调用保留。
  上述 Return warp 也处理 remote-only block：取得 FC2 完成后，从 immutable
  reverse routes 得到实际远端 source，按本地 rail 去重，warp 并行以 system
  release 发布 generation。TX 只 acquire 本 rail 的对应镜像，不再反复读取
  expert GPU 的 generation/fc2_done/target；C→TX 打包和逐行 C 退休不变。
  每个 block/rail 只发布一次，避免重复旧代 store 晚于 C 回收。任意一个未完成的
  TX reader 都持有其 C 行 credit，镜像不可能在该 reader 完成前进入下一代。
  remote-only block 的 TX 必须先取得本次发布，故不会先于发布者退休 C；发布者
  完成最后一次通知后不再访问可复用的 C 控制/载荷。source mask 仅覆盖有界 G，
  不是节点数 mask；多 source node 映射到同 rail 时仍只发布一次。
- receiver 的 immutable metadata replica 另有 `RouteLocation[route]`，由实际
  destination 写入 `{generation,logical_block,physical_row}`，经一次节点内
  PlanReady barrier 发布。Return rail 从该表找到 C；Dispatch pull 找到 X。
  此表属于 invocation metadata，不随 RX slot 复用；不跨 NIC 交换此本地地址表。
- 最老未完成窗口已有 source contribution 容量；destination 可完整预留该窗口，
  不等待 NIC credit。每 channel 的 TX 只按递增 window/packet 前进；pull 同样先
  window，再 source/lane/packet，静态 residue 不重叠。后窗 packet 即使提前占用
  部分 RX，也不能越过同 channel 的前窗 packet，且独立 RX/ACK 能继续回收前窗。
  Return TX 只复制全包已 ready 的 C 行，scatter 的目的 contribution 已预留，
  Reduce 释放 source 窗口，随后准入后窗。这个等待关系仍需热点/小深度实测验证，
  不能把推导替代多机验收。
- pull 的固定 worker 由窗口内 packet ordinal 与 source 交错映射，避免简单
  `packet+source` 导致小区间集中到少数 CTA。warp 协同读取远端目录，唯一
  TMA issuer 搬运；全局 copy 完成经 warp 同步传递后，各 lane 发布 X-ready，
  再归还该包的本 rank fanout credit。没有动态领用或运行时计算 expected count。
- source PlanReady 后，在已有窗口规划 pass 中预绑定普通本 GPU 的 `PullPacket`
  表：每窗 `G×ceil(B/Q)` 个 entry，固定索引延续上述 ownership。记录 RX
  payload/control 地址、generation、source 目录指针及已有本 rank fanout 数。
  空 entry 只设 rows=0；runtime 仅走自己的 residue，跳过空项不读 RX。此表
  不跨 NIC 交换，也不新增 barrier；保存的是当前调用的物理绑定，下一调用重建。
- NIC `readSignal(acquire)` → system-release `visible_generation` → peer acquire
  → 全部 TMA 完成 → system 累计 consumed → ACK，逐包建立可见性及复用链。
  多机 C.returned 同时接收本地 Return 与 peer rail 的更新，因此本地/远端
  RMW 及准入方的 acquire 全部使用 system scope；仅远端使用 system 不充分。
  同一 packet 内连续属于相同 C slot 的行可合并检查和退休，计数单位仍为行。
  所有 packet/reader/local completion drain 后才进行 invocation 结束 GIN barrier，
  防止快节点在慢节点仍读目录时开始覆盖下一次调用的 metadata。payload 中间无
  全节点 collective；跨调用仍保留这一边界。
- 后续 CF 将规划阶段的 GIN fence 限定为实际拥有 metadata chunk 的 context；
  仍由每个活跃 QP 分别发布，不能用 context0 的 signal 导入其它未完成 Put。
  终端所有 TX 已取得远端 consumed 与本地 NIC completion、全部本地角色 join
  后，只用一个 context 做 rail 会合，再经原节点内 barrier 保护 metadata。
  发出最后 ACK 的节点可以先到达，但收到该 ACK 的节点只有观察它才能结束 TX
  drain 并到达；故会合完成时全部 ACK 也已被观察。并未删去逐slot复用的双条件。
- CH 的 Dispatch packet 目录改由静态 warp 协作生成：按 route 展开 lane，合并
  相同目标 GPU 的计数后由唯一 lane 更新 packet 私有项，每批之间 warp 同步。
  expected_count 仍精确来自 route plan，不在数据面推测 fanout 或动态领用任务。
  这些规划优化的数值/资源及性能取舍见 ring 实施记录；20K 的最终性能验收未完成。
  CJ 的 uint32 source 目录压缩在8K有收益，但20K相邻轮次确认退化，已撤回，
  不能把该候选的目录宽度写成当前布局。
- CP 在输入 staging/source counting 的原 CTA join 后，让全部384线程参与
  后续目录、几何和地址规划，相关控制面 collective 使用同一完整线程组。
  source counting 仍由原128线程完成，与输入复制重叠；数据面角色不变。
  CQ 将逐channel的窗口packet prefix改为warp分批scan，Dispatch/Return
  共用同一实现，保留64-bit carry和精确terminal；无新增buffer或同步计数。
  两项合并通过完整门禁，正式结果见ring记录。减少CTA数量的CR没有获益，
  已撤回，不作为当前配置。

#### CN/CO：FC2 直写 Return TX 后的差异

6.3 和上面 M3b 的 C→TX 描述是初始实现。当前 Return owner 先用同一
`can_send` 双检查，按 FIFO 非阻塞预留最多一个 ring depth 的 TX slots，再以
system release 发布每 channel 的 `tx_reserved_head`；预留不依赖 FC2-ready。
destination 准入整窗时，除 X/S/returned 账本外，还检查该窗口实际涉及的所有
远端 Return packets 均已预留。source 的 node-expert 区间连续，每 lane 检查
该区间最后一个 packet 即可，不能跳过区间中的旧 slot。

网络深度为 `max(8, ceil(ceil(B*capacity_top_k/ReturnQmax)/lanes))`，保证最坏
一个 source window 可完整预留；此处使用独立的 Return Q，不是 Dispatch Q。
深度随固定 B/K/Q/L，而非总 tokens 增长。固定 TX owner 的预留进度不等待
本次 FC2，独立 RX/ACK 释放上一代，避免“计算等 TX、TX 等本次计算”的闭环。

FC2 按不可变 `RingRow.output` 直接写 contribution 或同节点 rail TX。
原 FC2 completion→system fence→rail-local `return_ready` 的发布链不变；
TX owner acquire 全包实际引用的 blocks 后直接 Put，不再复制 C→TX。
按行退休的 returned 仍防止 descriptor/ready 镜像提前换代；它不授予 TX
复用权。TX 从预留、FC2 写入、发送直至双 credit 返回始终归本代所有。
CO 删除已无读者的 C 数据区和 ring output-store TensorMap，非 ring 路径不变。
当前验收profile为 Dispatch Q8 / Return Qmax16 / L32 / B512 / contribution D8；
通用编译默认未切换；本profile的数值与性能验收见ring记录ET。

CV 将准入warp的检查按source rail分摊，再检查各source实际覆盖的packet lanes；
最终仍整warp一致决定整窗准入，不改变预留条件。CW按source epoch选择
`ReturnQ=min(Qmax, pow2ceil(ceil(min(source_tokens,B)*K/lanes)))`，最小1。
source planner及所有destination使用同一公式，不使用各自的token数；物理
TX/RX slot stride仍按Qmax。实际B被pool裁到max_tokens时也使用此公式。
主测T1/T64分别为Q1/Q8，长窗口Q16；两节点不等长和混合profile已验证。
Q小于Qmax时每window至多一个packet/lane，故原depth仍足够；目录每peer/window
容量上界改为`max(lanes, ceil(B*capacity_top_k/Qmax))`并受总routes上界限制。
CV/CW完整门禁与正式结果见ring记录；明确收益在短序列，长序列未明显改善。

#### DS：Return 直写 source contribution，删除 RX scatter

多机contribution迁至各source rail的注册window，物理容量仍为`D×B×K`行。
每个窗口内部沿用既有expert-major `route_order`，Return packet是其中连续区间，
因此GIN一次Put可直写最终contribution地址。节点内FC2同样预绑定该区的LSA地址；
source私有`uint32[route]`逆表把原route映射到窗口内行号。Reduce仍按原expert/slot
顺序逐项FP32相加再BF16舍入，ready数组仍采用原token-major索引。单机布局不改。

Return TX仍先取得远端packet credit和本地NIC completion，FC2-ready后才能Put。
source NIC import owner用GIN readSignal导入payload可见性，release发布packet
generation；固定接收角色acquire后发布各route ready，再累加既有`desc.rows`退休
packet控制slot。它不再复制payload，也不将可复用控制slot的指针交给Reduce。

payload有独立且更长的寿命：source全部token完成Reduce，才能给下一代窗口credit。
Dispatch发出该窗口X前已取得source contribution reservation；同节点Pull也有
同样依赖。因此下一代FC2/远端Return不可能先于该reservation写入contribution。
packet ACK只允许TX/control复用，不允许提前覆盖source payload；保守本地NIC
credit保留。两个ready/credit链共同成立，不能只凭远端ACK推出payload安全。

注册网络ring仅保留Dispatch TX/RX和Return TX三种payload plane。每channel/slot
Return控制仍在，既有expected rows、terminal drain及跨调用metadata会合不变。
逆行号在已有规划join后生成，无新barrier或运行时expected推测。
DS完整EP8/EP16及B16/D1/K8数值通过，rank1/9重复调用memcheck/racecheck均0；
20K相邻正式对照仅降低约0.3%，主要收益是少33,226,752bytes/rank缓冲区，不将其
称为显著提速或整体性能目标已完成。完整测量/冻结构建见ring记录DS。

### 6.7 Return RX 直接 Reduce 消融（AQ，未获益，已撤回）

基于逐 slot 协议消融掉 RX→contribution 的额外复制。每个 source route 的
不可变 `ReturnBinding` 保存 payload、ready word、generation、可选 consumed
counter：本地 route 指向 contribution；远端 route 指向本 GPU 的 Return RX。
Reduce acquire ready 后按同一固定顺序求和，所有 lanes 完成读取后，每个远端
route 归还一个 RX row credit。ACK 的 expected 仍是现有 packet desc.rows，
没有运行时推测 fragment/fanout 数量。

RX 从“scatter 后可释放”改为“引用它的全部 Reduce route 消费后可释放”，
因此须保证每 channel 可放下一个完整 source window 的最坏 Return packets：
`depth >= ceil(ceil(B*max_top_k/Q)/L)`，同时保留最小 depth8。否则最老 token
可能等尚未到达的 route，而 RX 被其他尚不能 Reduce 的行占满。整窗容量与
每 channel 的 window 单调顺序共同保证最老窗口可接收完整，随后 Reduce/ACK
释放容量。该容量上界随 B/K/Q/L，不随总 tokens 增长；较大 B 的内存代价要
单独报告。只在完整正确性、wrap、热点与性能验证后，才替代 6.6 的已验证 scatter
基线。AQ 的 8K 快筛没有收益，已撤回实现，未完成完整正确性验证；这里只
保留协议分析供后续参考。后续DS也消除了scatter，但Reduce持有的是独立source
窗口，不是AQ的可复用network RX slot，不能将二者视为同一协议。实测见ring记录。

### 6.8 计算 GPU 直接跨机 Return（BL/BM，无收益，已撤回）

6.6 是已验证基线。本项归档候选只改变 Return 拓扑：计算 GPU 从自己的 C 打包，
经 GIN world peer 直接发给源 GPU，不再通过节点内 rail 汇集 C payload。
Dispatch 仍按节点去重、同 rail 接收和 destination-pull；两者不必采用相同拓扑。
V3 已使用跨节点任意 GPU peer 的 Return，硬件支持不等于新的静态协议已验证。

- Return source plan 从 `[node,lane]` 改为 `[compute_rank,return_lane]`，每项
  覆盖本 source/window 中该 compute rank 的 expert 区间，仍按 Q 切完整行。
  前缀、packet 数与每包 expected rows 均在 source plan 阶段确定；同节点目录为空。
  sender 从对应 source 的 replica 读取目录，receiver 使用自己的同一目录，
  因此 wire 不增加逐行身份、不增加第二次网络目录交换。
- Dispatch 和 Return 的 channel 数分别为 `(nodes-1)*L_d` 与
  `(nodes-1)*local_world*L_r`。初始取 `L_r=ceil(L_d/local_world)`，单独分配
  两个 plane 的 channel/control/payload，避免把每个 GPU peer 无条件乘以 L_d。
  context 仍由静态 owner 独占；一个 owner 轮询其固定 channel residue。
- Return TX channel 编码目标 source rank，RX channel 编码发送 compute rank。
  Put 与 ACK 必须应用互逆的 channel 映射，不能继续假设两端 local rank 相同。
  sequence、expected/consumed 与本地 NIC counter 仍是各自逐 slot 累计状态。
- Return ready 只需本 compute GPU 的逐 C-slot generation；发布者 acquire
  FC2 completion 后 release ready，TX acquire 后复制，全部异步读取完成才按行
  退休 C。没有 peer C reader，C.returned 可统一使用 device scope。source
  contribution 的远端可见性与 local Return 的 system publication 不变。
- RX 仍复制到 source 已预留的 contribution，再发布 route ready、归还 RX
  credit；TX 仍同时等待远端 consumed 和本地 NIC completion。不将这次拓扑
  实验与去掉本地 completion、直接 RX Reduce 或新的 GEMM profile 混合。

门禁：复用现有 host 地址检查验证非同 rail 的双向映射与无重叠；GPU plan 检查
逐 compute rank 的覆盖、tail 和 exact expected，包含三节点几何；两机实测完整
四规模、热点/不等长、小深度 K8、多代 graph replay 及 rank1/9 sanitizer。
若无端到端收益，归档候选并保持 6.6；不以代码简化代替性能与正确性验收。

BL 的 8K 快筛退化至 4.60 ms；BM 在 PlanReady 后预绑定本地 packet/逻辑行，
降至 3.51 ms，但仍未好于原协议 BH 的 3.42 ms。Return Q32/Q64 也未获益，
因此两份实现均已撤回，完整训练/数值/sanitizer 门禁没有在失败候选上继续。
保留构建、patch 与资源记录，不把上面的协议分析标成现版本实现。

## 7. Gate/up：独立 GEMM 消融，不预选 decode/prefill 布局

### 7.1 两种 GM 布局和计算路径

每个 expert 原始 W1 为 BF16 `[2I,H]`，H 连续；逻辑计算是 X 乘 W1 的转置：

```text
G_i = BF16(dot(X, W_gate[i,:]))
U_i = BF16(dot(X, W_up[i,:]))
S_i = BF16(SiLU(float(G_i)) * float(U_i) * router_weight)
```

V3-style separated：GM 行顺序为全部 gate I 行，然后全部 up I 行。
每个 FC1 任务覆盖 N_s 个 S 输出通道，共享 X，分别加载 gate/up B，
用两组 accumulator；现有实现 gate WGMMA 可先于 up TMA 全部完成开始。

V4-style interleaved：仍保持每个 H 维权重行连续，但行顺序为
`G0..7,U0..7,G8..15,U8..15,...`。p=0 表示 gate，p=1 表示 up：

```text
separated_row = p*I + i
packed_row    = (i/8)*16 + p*8 + i%8
address       = W1_base + (expert*2I + row)*H + h
```

上述映射描述当前按 8 通道分组的支持形状；扩展不整除尾部时必须显式处理 padding。
interleaved 一组 packed B/accumulator 计算两投影，物理 GEMM N 只产生 N/2 列 S。
它不减少 FLOPs。separated 的 N128 与 packed 的 N128 不是同样的有效输出宽度：
I512 时分别需要 4 与 8 个 FC1 N tiles；当前 packed N64 则需 16 个。

S 统一存成标准 `[M,I]`，FC2 不感知 W1 布局。ring 的 F1/ready 目标从所选
profile 的逻辑 N-tile 数推导，不写死。两种布局都可用于任一长度，不能先断言
“decode 用 packed、prefill 用 separated”。

### 7.2 必做的独立实验

把布局视为计算路径的独立变量，先隔离 GEMM，再验证端到端；不和 ring 改造捆绑
归因。独立 benchmark 使用已在本 GPU 的 canonical X、同一组权重、router weight，
相同 Mblock/valid_m 分布，无 Dispatch/GIN/Return：

1. 分别重放 1/64 和 8192/20480 tokens/rank 所产生的 expert 行数/尾块分布，
   包含 balanced 与热点样本；不能只跑大矩阵后推断 decode。
2. 比较 separated 双投影和 interleaved 单 packed 投影，至少报告
   FC1+SwiGLU，以及固定 FC2 下的 FC1+SwiGLU+FC2。
3. 先做受控对照：相同 M/K、精度、K 累加次序、有效 S 输出宽度、cluster 策略，
   尽可能匹配 stage/epilogue；记录布局必需的数据流、资源或指令形状差异。
   不能仅把两个实现的 N 宏设成一样就称为等粒度比较。
4. 再分别调算术 M、N/K/stages、CTA 配额及有效 task 粒度，报告两种布局各自
   的最佳可行配置；
   “布局本身的差异”和“布局加各自调优后的差异”分开记录。
5. 核心固定 1620 MHz，检查实际时钟；记录延迟、register/spill、SMEM/occupancy、
   TMA/WGMMA stall 和有效吞吐，不用操作数大小推测胜负。
6. 权重 packing/descriptor 建立在计时前完成；另外报告转换耗时、额外权重副本内存、
   生命周期和更新/失效成本，不把一次性 packing 混进 steady-state GEMM。
7. 根据结果分别选择 decode/prefill profile。若选不同布局，明确同时持有两份
   权重的内存代价，训练权重更新后不能使用旧 packed 副本。
8. 在相同通信协议下回测 EP8/EP16 全矩阵，验证 profile 选择及阈值；真实 workload
   边界样本用于校准，不盲用当前 8192 阈值。无新实测前不填优化收益。

输入 token 数只是选 profile 的代理，expert 实际行数与 skew 也影响 GEMM。
若使用设备 plan 的计数选 entry，选择必须在静态任务/计数目标发布前固定，
不能让同一 block 的消费者对 F1/布局有不同理解，也不能增加 host 读回同步。

### 7.3 数值和 TMA 约束

- FP32 accumulator；G/U 各自先 BF16，再以 FP32 做 SiLU、乘 up 和 router weight，
  然后 S round BF16；FC2 输出 BF16，source 固定顺序 FP32 combine 后 BF16。
- router weight 不是 W_gate；不能把它移到 FC2 之后。
- tile、stage、布局是待调参数，不再沿用“TileN/TileK 必须永远不变”的旧约束。
  保持相同运算次序的受控比较检查 bitwise；若候选改变归约次序，明确标注并走
  独立 PyTorch 误差门禁，不能把数值差异藏在性能结果里。
- 当前 cluster multicast 分享 A，保持两 CTA 的 stage 读者生命周期。将来若消融
  share-B pairing，单独证明 GM/SMEM ring 与 epilogue staging 的安全性。
- 已迁移的 FC1/FC2 合并写回、CTA 私有 B staging 要在两种布局下复用/验证；
  不恢复不安全的 multicast A arena epilogue 复用。
- FC1/FC2使用不同的stage barrier，但复用同一SMEM。静态配对的两个CTA可有
  阶段偏斜，不能只按各GEMM独立的最大容量证明安全。切换前须确认旧阶段两个
  CTA的操作数读者都已退休；FJ2复用已有empty barrier完成这一边界，不恢复
  动态task claim或逐tile的pair任务握手。
- epilogue的“私有B区”必须位于**两种GEMM的最大A extent之后**。例如FC1
  stages3、FC2 stages2时，FC2 B从64KiB开始，但FC1 multicast A可写到96KiB；
  64KiB不能作为安全scratch起点。FJ2将scratch放在96KiB以后，允许peer在旧
  私有epilogue执行期间进入新GEMM。原FJ失败及修正验证见ring记录，最终采用
  仍以完整数值和端到端性能验收为准。

### 7.4 单机和多机的统一 autotune 原则

单机/多机、decode/prefill 都应使用统一工作流、已验证的参数化计算模板和
autotune 配置表。模型形状需要不同 M 或配额属于配置选择，不是机制不通用，
也不是新增模型专属分支的理由。当前 HHE 的 M16/M64 密度阈值和 GA+FZ 的
网络参数仍来自经验选择与独立消融，不能写成已经完成统一自动选型。

- 搜索键包含 GPU、H/I/E/top-k、dtype、EP、TP/DP 输入布局、batch bucket
  及 inference/retain-state 模式；多机还区分节点数、每节点 GPU 数、互联和
  通信后端。不同模式分别测量，不复用未经验证的训练或网络最优配置。
- 算术候选联合搜索 M/N/K、stages、CTA 配额及合法 cluster 策略；通信候选
  考虑 Dispatch/Return Q、lane/owner、source group、B 和各 ring 深度。
  可分层预筛，最终用同输入、完整通信计算路径的 rank-max 延迟，在给定内存
  预算下选优；保留多个路由样本、热点及不等长用例，不仅优化均匀随机种子。
- 区分算术 slice M 与物理 ring block M。当前单机 decode 的 M16/M64 已验证，
  通用 ring 仍使用自己的 M128 几何；网络算术变体必须先实现并验证。更改物理
  M 或通信几何时，同步生成 plan、容量、TMA maps、ready 目标及 ownership，
  不能只扫一个宏便假定整个协议适配。M32/M128 等新增 decode 算术候选同理。
- 配置离线搜索并缓存，运行时在 prepare/graph capture 前选择，不增加正常
  执行中的试跑、host 读回或动态 task claim。通信布局和容量由通信组一致
  选择；只有已证明共享协议兼容的算术变体可按 rank 的形状独立选型。更换
  不兼容配置须先 drain，并准备对应 pool/maps/graph，不能中途改变 slot 合同。
- 所有候选保留 no-drop、静态 ownership、远端消费与本地 NIC 完成双 credit，
  并满足原有生命周期证明。
  在要求 batch-invariance 的 inference 入口，跨 bucket/算术形状也必须严格
  逐位对齐；仅各自通过容差参考不足以允许自动切换。第 7.3 节允许归档的
  不同归约次序诊断候选，不能据此进入该自动选择集合。

2026-09-30 已复用扩展 `tools/autotune_forward.py`，接入现有 decode M16/M64
和通用 ring M128 的离线 Graph 搜索、确认、导出及 prepare 查表。物理
`tile_m=128` 不变。具体支持范围、命令和限制见 [autotune 文档](autotune.md)。
目前为 DP inference、精确 pool capacity、单一编译 profile；网络新增算术 M、
训练选型、同进程多编译 profile 自动装载及 GPU 完整门禁仍是后续工作，
不预先声明其他模型/节点数收益。

## 8. Prefill/training 的保存与 no-drop 合同

### mini Notify 与静态 Backward 重构（2026-10-01 已验收）

本轮目标取代下文 M6/FS2 的历史执行边界：训练只在 main 前做计数 Notify
和精确 tape 分配；训练与推理共用 main 内的物理规划、通信和计算流水。
推理的保存指针为空，不建立 tape。计数不赋予原子行号；main 必须生成稳定的
expert/source/route 逻辑顺序，保持 checkpoint 和 dW 的数值合同。

- [x] mini Notify：只统计、交换各 source/expert 行数和分配所需 R/P；不构造
  V3 的 packet/wave 排序，不启动 activation payload。
- [x] 将 ring planner 合入 main，保留全 rank 分配成功之后才能推进 ring 的
  边界；验证零行、热点、不等长、多个 live tape 与 allocation retry。
- [x] 静态 Backward：固定 CTA/cluster 的 dU、SwiGLU backward、dX ownership，
  不再动态领用计算任务；复用算术和通信模块。
- [x] dW1/dW2 严格位于本 rank 全部 dX GEMM 完成之后，而不是低优先级抢占；
  Return/Reduce 可以继续与 dW 重叠，权重梯度不在 combine 的依赖链上。
- [x] 在 SwiGLU backward 一并计算 selected-route dGate，随 dX 返回。算子不接收
  router 权重或 logits，router softmax/top-k 之外的参数反传仍由上游 autograd
  消费 dGate 完成。
- [x] 1620 MHz 下 EP8（单机）/EP16（双机）的正确性、Forward wall、Backward wall
  与 FWD+BWD 总耗时对照冻结 V3；分别记录 1/64/8192/20480，不把 main 计时范围
  变化误记为性能收益。每个验证完的 milestone 提交。
- [x] 按最新要求完成 8K/20K 性能门禁，不再测 1/64。最终同一候选
  `eb371c2` 的 EP8/EP16 × 两个长档 × FWD/BWD/实际循环共十二项，
  200 measured rank-max wall p50 均低于冻结 V3。双机 8K BWD 仅小幅
  领先 0.41%，不称显著提升；实际循环耗时降低 20.09%–38.68%，其中
  packing scratch 分配器复用贡献的是 host 开销优化，不是 GEMM 加速。
  具体口径、完整数值及限制见[最终统一验收](static_forward_v4_training_record.md#最终统一验收2026-10-01eb371c2)。

以下记录已落地实现及历史基线；性能以此次最终统一验收为准。

mini Notify 与 main 内规划已落地，EP8/EP16 native 梯度 hash 和 Torch reference
parity、热点/零入站、多个 live tape、allocation retry 已通过；结果和计时边界见
[训练记录](static_forward_v4_training_record.md#mini-notify--main-内规划2026-09-30)。
EP8/EP16 T20480 长序列 parity、EP8 不等长 source memcheck，以及 EP16
T64/32 在 B16/D1 与 B1024/D4 下的输入/权重/输出 hash 一致性也已通过；
最终静态 Backward 的性能门禁独立记录，不由 Forward 正确性推定。

main 的规划阶段同时生成永久 tape 的 route header/gate；它们不依赖 X 到达，
不能留给 Pull issuer 逐行串行生成。Pull 仍在 slot admission 之后写入物理
ring 的 gate/X，并按原 completion/credit 发布 ready；永久元数据与可复用
payload 的生命周期不可混同。训练 packet/fanout 不属于发布给 Forward 的
源目录，可与源目录 Put 重叠构建，但仍在原 context fence/collective 边界前
完成。推理不建立这些训练数据；两者复用相同 main 流水及物理 planner。

mini Notify 返回后实际 R/P 已知。应用入口把输入/权重准备和 tape 分配放在
同一次 CollectiveTensorPreparation 中，然后把已绑定 tensors 交给 main；
backend 不再对它们重复确认分配。未绑定 tensors 的原生入口仍由 backend
分配并一致确认。旧独立 planner 时期的延迟分配 callback 已删除。保留
单 rank 分配失败时全体报错、释放部分分配并重试的合同，不能省略这唯一
准备边界，也不在 payload 启动后补分配。

eager 权重发布的 interleaved W1 scratch 同样由应用缓存分配器提供，lease
持有到消费完成；准备失败则等 packing ready 再回收。每次仍重排当前权重，
不按地址缓存内容，因此训练原地更新不需要失效接口。原 native 分配 overload
保留；inference graph 的一次性权重准备及不可变合同不变。该优化消除每步
原生 cudaMalloc/cudaFree，属于完整训练循环开销，不计作 GEMM 加速。

当前 Backward 的静态合同：

- 有足够 M-block 时，cluster `c` 拥有 `c + n * clusters` 的完整
  `dU → SwiGLU backward/dGate → dX` 链。M128 epilogue 合为一个任务，dX
  连续遍历全部 N tile；整条链只交换一次任务描述符，保留阶段间数据完成同步。
  不足一个 cluster wave 时，改为固定 N/epilogue
  residue 以利用空闲 cluster；两种情况都没有动态 claim。
- Dispatch 的系统域 acquire 确认 dR 的有效行全部就绪后，由唯一 publisher
  release 发布该 M-block。计算 owner 的 dX store 完成后发布块级完成标记；
  固定的 Return CTA acquire 后发送 dX/dGate。没有计算/Return 队列领用或
  phase CAS。跨机复用原有 source-token 去重及 GIN ring，不宣称 Backward
  的全部中间缓存都已成为有界 ring。
- dW1/dW2 的静态 output-tile chunk 只在 acquire 到本 rank 的
  `completed_dx_tiles == total_dx_tiles` 后执行；不是“当前没有 ready dX”
  就允许执行。通信 warp 可在此期间继续 Return/Reduce，无需全节点 dX barrier。
  联合 dW2/dW1 tile-pair 流按各专家的实际 padded K extent 加权，cluster
  获取不相交的连续区间，区间内保留 TMA chunk 流水；空专家权重为零，
  不做 split-K，也不把一个热点专家的全部 dW 固定给单个 cluster。
- Tape 逻辑行仍是稳定 expert/source/route 顺序；main 由 counts 直接生成
  32-row expert 交错的 dY dispatch 索引，通信遍历顺序不改变 dW 归约顺序。
  算术、TMA maps、BF16 checkpoint 和逐行 dGate 的 FP32 shuffle tree 复用
  V3；非 ring 路径继续使用原动态 Backward，保留清晰的回归对照。
- 通信 warp 暂时无 Dispatch/Receive/Return 工作时，可以非阻塞尝试本 warp
  固定 token residue 的 Reduce；全部 top-k ready 才归约，否则立即继续通信。
  独立 pointer table 不覆盖活跃 TMA staging。传输退休后从原 cursor 继续，
  不重复归约、不等待缺失的 contribution 来阻塞通信。

Backward peer workspace 属于执行 slot，不是 live tape。2026-10-02 改为
**容量复用**：首次 Backward lazy 分配，实际 rows/routes 变化不再更换 allocation。
`backward_peer_capacity_rows` 是 dR 初始容量 hint；TensorRuntime 使用调用者的
`max_rows`，不会误用训练路由元数据的全局最坏行数。native hint 为 0 时取
两倍 `max_tokens_per_rank * max_top_k`，受 metadata 行数上限约束；source-side
contribution/dGate 按 `max_tokens_per_rank * max_top_k` 预留。

这部分不是网络 ring：dR 按整次调用的 compact row 索引，且在 dX 之后还用于
dW2。物理 buffer 的容量、各区偏移在驻留期内保持稳定；TMA 描述符仍使用
**实际 rows**，保留尾块越界补零语义。实际负载超过 hint 时按至少两倍旧容量
扩容（不超过 metadata 上限，且覆盖本次实际需要），之后不因缩短输入而收缩。
因此预留容量内无重分配/重映射，极端热点仍可安全扩容而非丢 token；没有声称
有限 heuristic 能覆盖任意大接收量。驻留结束统一释放，不做进程级永久缓存。

每份新的 Notify plan 仍执行分配确认、地址 AllGather 和 import 确认，未恢复
M 的 any-changed 协议。只有 peer 的 allocation/handle 真正变化才重建对应映射；
旧 imports 全部关闭并完成集体确认后才释放被替换的旧 export。相同不可变
plan 的重复 Backward 保留 generation 快路径，分配确认仍用 uint32 最低失败 rank。

共卡切换的 `TensorRuntime.close()` 要求无 live training tape，外层 CUDA Graph
先由应用销毁。所有 rank quiesce 后先 barrier，native 清理内部推理图并调用
`ActivationTransportResourcePool::retire_peer_mappings()`，统一关闭 FWD/BWD imports；
第二次 barrier 确认全部 imports 已退休，再销毁 runtime/export/GIN 资源。
`retire_peer_mappings()` 是终止边界，之后不能再次执行该 pool。
Megatron `suspend()` 同时清除共享 packed weights；SGLang 先清外层图，再关闭
PF/DT runtime，resume 在新驻留期重建。SavedX/GU 和梯度仍由各 tape/调用方
独立持有，不被通信 workspace 的容量缓存延长生命周期。

Backward 与 Forward 共用 backend 的 launch 准备缓存，按 entry、资源配额
与启动几何复用已通过的寄存器、shared-memory/stack 和 occupancy 检查。
首次检查仍完整执行，统一 BWD 不设置不会启动的 split kernel 属性；不缓存
路由或 tensor 地址，不改变跨调用数据/通信完成边界。

统一 BWD 的 GIN context 0 留给 metadata；仅当实际 context 数大于 worker
CTA 数时，data context `1+cta` 才采用 THREAD 独占模式（每 CTA 单一提交者）。
实际资源不足时仍用 GPU sharing，跨调用由 GinExecution 完成事件串行化。
remote credit、本地 NIC counter、ACK flush 与 payload 可见性合同均不变。

Router 梯度放在 SwiGLU epilogue，避免另读一遍 gate/up。记保存的两个投影为
`A/B`、选中专家的 routing weight 为 `p`、`S = SiLU(A) * B`：先做
`dU = dY * W2`，epilogue 同时得到 `dp = sum(dU * S)`、
`dA = p * dU * B * SiLU'(A)`、`dB = p * dU * SiLU(A)`，并重算供 dW2
使用的 `p*S`。随后 `dX = [dA,dB] * W1`。dX 块发布后可立即 Return dX/dp；
所有本地 dX 完成才开始 `dW2 = dYᵀ * (p*S)` 与
`dW1 = [dA,dB]ᵀ * SavedX`。上述是数学关系；实际 BF16 中间舍入及 FP32
逐行归约树沿用 V3，不以重新排列 GEMM 的机会改变 batch-invariance 合同。

### 以下为 M6/FS2 历史执行边界

prefill inference 与 training Forward 共用有界流水，但训练保存实际规模的 tape，
不是在 ring 中留住整次调用的激活。M6d/M6e已接通单机及多机版本，非ring训练
仍用V3及相应编译profile；Backward保留原GEMM/调度，多机继承8-GPU/node的
rail合同，节点数按runtime映射，当前仅有2机硬件验证。FS2的准备顺序如下：

```text
materialized Notify
  -> stable route ordinal / canonical count exchange / 精确 R_actual、P_actual
  -> 冻结 canonical plan；此时不发 activation payload

host allocation boundary
  -> 取得本 rank 实际 tape 大小及 collective status
  -> 分配 saved tensors / plan lease；失败则在物理规划前一致报错

static_ring_plan_kernel
  -> 窗口计数 / packet metadata exchange / physical row 与 canonical row 映射
  -> 冻结 main 的物理执行计划；同 stream 衔接 main，无中间 D2H 或 host 等待

main_from_plan
  -> 复用原 plan，不重复 route counting/exchange
  -> bounded Dispatch / X / S / C / Return / contribution 流水
  -> 保存独立的 X/G/U、gate 与返回映射

Backward
  -> 消费该 Forward 的精确 plan/tape
  -> 固定逻辑行顺序和梯度归约合同
```

saved tensors 根据实际 R_actual 或所需的 padded P_actual 分配；transient X/S/C
仍是固定深度 ring，不重新申请完整 training canonical arena。稀疏/极端路由
不需要每个 rank 都预留 world×max_tokens×K 行激活。

tape 的永久逻辑行号与物理 ring slot 分开。若 tape 沿用全局 expert/source 顺序，
而执行按 window 顺序，必须保存明确映射；不能把 window 内 padded row 当成现有
Backward 的 canonical row。G/U 可保存标准逻辑布局，使 Backward 不依赖 FC1
选择 separated 还是 packed。dW 跨 M 归约的确定性需要单独验证，Forward 数值
一致并不自动保证梯度归约一致。

M6对接优先复用materialized Notify已有的`route_dst_row`：它由expert base、
前序source计数与stable local_offset构成，可直接作为compact tape行号。
训练plan冻结并传递此映射，保存时映射到canonical row，Backward按原canonical
顺序归约。FR3的local training复用Notify的stable ordinal，在连续token window
内减去该expert最小ordinal形成连续source序列，再按source rank拼接物理行；
目的是使SavedX能成块写回，不是Backward确定性的必要条件。Network继续原有
物理枚举。Notify及ring窗口规划均属于main之前的plan阶段，
不能在main里再次执行。M6d/M6e已在单机及多机实现并验证该映射及两层plan边界，
不能将两层规划称为一遍planner。训练入口拒绝可变fused/replay-counts Notify。

FS2的tape分配只依赖Notify已经给出的实际R/P，不依赖物理planner回传的行数。
全rank准备成功之前，不执行物理planner、不推进设备ring epoch/base，也不准入
payload；分配失败确认后可释放lease并重试。这不保证host调用编号不增加，也
不承诺从CUDA/NCCL控制面失效中恢复。两层GPU规划仍各执行一次，main不重做
窗口规划；FS将物理规划合入main的实验在单机20K出现小幅回退，已撤回。

每个 live tape 持有独立 allocation/plan lease，直到 Backward 完成。资源不足可
等待或显式失败，不能覆盖尚未消费的 tape。输入/输出/plan/tape 真正 OOM 时允许
整个调用失败；“不丢 token”不是承诺任意总规模都可在有限显存内训练。

需要 host 分配的 plan/allocate/main 不是单 kernel，也不承诺整体 CUDA Graph。
PDL 不能替代动态地址的分配依赖，本轮不加入 pointer-mailbox 或运行中 host 扩容。

## 9. 模块划分与复用

当前代码统一使用 `weftmoe_dt` 命名单机 decode 专属路径，使用 `weftmoe_pf`
命名通用静态流水（含长文、跨机短文和训练 Forward）。原 `local_decode/`、
`static_v4/` 目录分别迁至同名目录；共享 GEMM/worker 实现在 `cuda/forward/`。
此命名整理不改变入口选择、配置宏/环境变量、缓冲区布局或 V3 回退。
性能优化重点为：`weftmoe_dt` 的单机短序列，以及 `weftmoe_pf` 的单机/多机
长序列（8K、20K，含训练）。双机短序列保留现有功能，不作为 PF 的重点性能目标。

基于现有 `src/cuda/weftmoe_pf/` 演进，不新建一份 V3 GEMM。拟划分的职责如下，
具体文件名以实施时最小改动为准；此表不是声称这些模块均已存在：

| 职责 | 复用/修改方向 |
|---|---|
| route/窗口几何 | 扩展 route_plan；共用 source identity、logical row、combine order |
| slot 协议 | 独立的小型 generation/full/empty helper，与 GIN 传输 API 分离 |
| 节点内搬运 | destination_pull 与返回完整行复制共用 TMA/peer copy primitive |
| 网络布局/访问 | network_layout/access：TX/RX 四区、控制字段、peer/lane offset |
| 网络进度 | 独立的 Dispatch TX/RX、Return TX/RX 固定 owner，不夹带 math task claim |
| 数值计算 | 现有 WGMMA/TMA/epilogue 参数化布局/profile，共享 BF16 合同 |
| 生命周期 | initialize/finalize 管理 invocation 与 drain，不在逐行路径加入全局状态机 |
| training | StableTape、实际分配及 Backward adapter，与 inference 快路径分开 |

单机 specialization 不编入无用 GIN 路径；decode/prefill 分离配置和必要的编译期
entry，但共享路由、ring 和 Return 协议。不能让 layout/profile 分支复制通信栈。

静态 worker 编号由 CTA/cluster/warp 及 round 推导；旋转 residue、owner 私有
小窗口 lookahead 可后续独立消融，不恢复跨 worker steal。GIN context 有唯一
提交 owner；按实际角色需要分配 context，不把整个 CTA 数无条件乘到所有资源上。

## 10. Invocation、Graph 与复用边界

“调用”指一次 MoE Forward invocation，例如模型中的一层或一次 graph replay 中
的一次该层执行；不是一个 packet、window、CUDA CTA 或服务端 request。
一次调用内部含许多 ring generations。

- 同一 EP 组按相同次序调用，runtime workspace 依靠 stream/graph dependency 串行复用。
  多 workspace 不自动意味着多 stream 的 GIN 并发安全；本轮不扩展该语义。
- ring 的 g 是该物理 slot 的累积代号；逻辑分配序号跨调用保留或加正确的 base。
  不能在每次调用把 block 序号重置为零，同时继续使用未归零的 full/empty targets。
- invocation 完成前确认输入读者、所有 Return、Reduce 和 GIN 请求/credit 已 drain。
  下一次调用不能覆盖仍被前次异步操作使用的 payload、metadata 或 control。
- Graph 使用稳定地址，routes/gates 可在输入和 plan workspace 合同内改变；
  transient ring 容量不要求容纳 graph 的全部 route。plan/control 本身仍需要
  与 graph 输入规模匹配的容量，不能以 graph capture 为理由忽略元数据容量。
- Graph plan的权重不可变，包括原地修改。W1 packed副本在prepare/publish时生成，
  指针相同不会自动刷新内容。更新一层权重前，各EP rank先丢弃引用该层的外层
  graphs，再在capture外调用`TensorRuntime.invalidate_inference_layer(layer)`；
  它等待已发出工作并释放该层全部token桶的plans。随后更新权重，重新warm及
  capture。共享权重的其它层也要失效；不在replay热路径做版本或内容hash检查。
- Eager prepare 或训练 plan 后可以分配实际输入/plan/tape；steady-state inference
  main 内不 cudaMalloc，不 D2H 读 count，不让 host 参与逐窗口推进。
- 只在完全 quiescent 的初始化/重建中 reset 累计状态；Graph、不同模型层共用
  workspace、profile 切换必须沿用同一生命周期合同。

## 11. 实现待办与 commit 里程碑

每个 milestone 完成并验证后提交一次 commit，同时更新对应优化记录。只记录
实际完成的代码和测量，未实现项保持未勾选。以下顺序不是要求 GEMM 消融等待网络：
M5 与协议实现正交，可在冻结基线后提前做，但不把两者收益混为一项。

### M0：设计与基线整理（本次文档工作）

- [x] 区分已实现 full-arena 基线与待实现 ring，统一节点内 pull、节点去重及流水目标。
- [x] 明确窗口准入、各 buffer 的独立生命周期、expected_count 来源及双 credit 决策。
- [x] 列出 gate/up 独立消融、实现 milestones、1620 MHz EP8/EP16 测量要求。
- [x] 保留历史实验记录，不把文档修改记为性能收益。

### M1：冻结可执行的 plan、布局与进度合同

M1a 已实现 host/device 共用窗口/packet/slot 合同与 GPU source-window planner；
M1b 增加独立于 RX 生命周期的 packet→consumer 目录及 destination-window 几何。
M3a 补齐 Dispatch route→packet/token 反向索引和 Return 精确逐 channel 目录，
并将 source epoch/gate/route/packet 元数据集中为不含 X 的独立区域。
后续 BW 移除了数据面不再读取的 source ordinal 逆表与 route→Dispatch packet
逆表；保留 route_order 顺向目录和 PullPacket 绑定，source ordinal 的定义不变。
详见 ring 实施记录。规划器已接入单机及多机 Forward，包括窗口几何、source
contribution 映射、网络目录交换和整窗口准入。这里的 source ordinal 是一次
invocation 内冻结的顺序，不是 M6 尚待实现的确定性 StableTape 行号。

- [x] 窗口编号、完整 token 分包、任意节点数的 peer/lane 地址和逐 slot 累积退休合同。
- [x] GPU source-window expert/node counts、去重 token 列表、source ordinal 与 channel
      packet prefix；显式区分 masked route 和非法 expert id。
- [x] 紧凑 channel packet 目录、各目标 GPU 的完整行消费数；destination-window
      counts→Mblocks/C expected 行数及 source round-robin ordinal 的逆映射。
- [x] 定义窗口编号、实际窗口/expert counts、逻辑 Mblock、source contribution 映射，
      以及 packet-sequence 优先的接收侧静态任务目录。
- [x] 冻结 packet/token/fanout 及 Return 行枚举，生成精确 channel packet 数和
      expected/retire targets；明确哪些目录存储、哪些闭式计算。
- [x] 定义各 plane 的 GM 地址、slot 控制字段、GIN VA signal/local counter 分工和初始化。
- [x] 实现准入/退休账本；画出最小等待依赖图，确认最老窗口的所有消费者可驻留推进。
- [x] 用少量 host/device 共用枚举检查证明覆盖、无重复、no-drop 与 tail 计数，
      不为每个字段建立大量无意义单测。
- [x] 冻结 V3、当前 V4 构建/config/hash 和重测脚本，作为后续逐步对照。

### M2：单机 X/S/C 与 contribution ring

- [x] 接入 destination-pull → X full、FC1 → S full/X empty、FC2 → C full/S empty。
- [x] 接入按完整行的 Return/C retirement 和窗口内 Reduce/contribution 释放。
- [x] 保持 reverse metadata 不依赖已释放 X；验证 multicast/epilogue stage 生命周期。
- [x] 用小 ring 深度、多代 wrap、热点 expert 和 tail 暴露错误的复用与容量依赖。
- [x] 单机 EP8 四规模 eager/graph 正确性与 1620 MHz 性能，记录 full arena→ring 的
      独立变化及实际峰值显存；不以更多常驻显存伪装有界流水。

M2 首版 X/S/C 有独立 completion/target，但三个 plane 使用相同物理 slot，
保守地等待三者全退休才共同复用；尚未利用 X/S 早退休的容量。source contribution
是独立的 ring，默认两窗口，可通过 pool 的 `static_forward_contribution_windows`
配置并独立消融。首版性能落后，不默认启用；已测固定 pool 的分配峰值，
尚未测应用整体显存峰值，性能与冗余 metadata 优化继续跟进。

### M3：多机 route-aware Dispatch 与节点内 pull/FC 重叠

- [x] 实现按目标节点去重的 packet，未命中节点不发送 X，支持参数化 peer 数。
- [x] 接入分离 TX/RX ring、固定 GIN owner、ready 可见性链、fanout 完成及双重 credit。
- [x] 实现分级窗口准入（source reservation、RX 暂存、destination 整窗 pull）；
      RX/pull/credit 的进度不受阻塞 TX owner 影响。
- [x] 移除 data-plane Dispatch 全节点 barrier，验证 NIC 接收、节点内 pull、FC 时间线重叠。
- [x] EP16 四规模和边界路由通过；若过渡阶段仍保留完整 Return arena，明确标为
      临时路径，不能在此宣称全部多机有界流水完成。

### M4：多机 Return/C 回收与端到端流水

- [x] 从静态 C/route 目录打包，按完整行退休 C；不引入 fragment 引用图或 append queue。
- [x] Return 与 Dispatch 独立；DS直写contribution，ready发布后归还packet credit，Reduce后释放payload窗口。
- [x] 移除节点级 FC2→Return barrier，确认 Return 与其他 Mblocks 的 FC1/FC2 重叠。
- [x] 完成所有 channel 的 terminal/drain、跨调用/Graph 状态复用。
- [x] EP8/EP16 全矩阵、热点与小深度 wrap 通过；记录 Dispatch 和 Return 各步收益及
      资源变化。协议支持任意有效节点拓扑，实机仍只声明两机验证。

M4 的功能与测量门禁通过不表示性能目标已达成；AV 时 EP16 长序列仍落后 V3，
后续优化和最终默认配置验收继续进行。AY 直写 Return TX 只是归档候选，未合入。

### M5：长短序列 gate/up 与 GEMM 独立消融

- [x] 建立第 7.2 节纯 GEMM 基准，固定输入/权重/route 分布、时钟和数值参考。
      N64/K128/stages2 的 EP8/EP16、四规模、均衡/热点、FC1/chain 两布局共 64
      用例通过完整输出/replay 校验；更宽 tile、独立调优与 profiler 分析仍在进行。
- [x] 做等有效 S 宽度的 separated/interleaved 受控对照，核对逻辑 N tiles 及计数目标。
  AF 已完成 N64/N128、balanced/hotspot、四规模、world8/16 重放；两边共享
  生产 SwiGLU/精度合同，完整输出与 replay 校验通过。AS 更正 balanced 的
  主矩阵路由公式并重测 T1；T64/8K/20K 逐专家 counts 相同，原结果有效。
- [x] 分别调优两种布局的 N/K/stages/epilogue，保留失败候选和资源/误差原因。
  已测 N32/64/128、K64/128、stages2/3/4/6；N32 仅 packed 支持。
  AF/AJ 记录资源、完整样本与失败的 group-overlap 候选。profile 选择与
  端到端验证仍是下列独立待办，不把诊断基准结果当成融合算子性能。
- [x] 分别选择 decode/prefill layout/profile，明确切换阈值及权重副本更新合同。
  权重更新合同及显式layer失效接口已实现，EP8/EP16的原地更新与全部token桶
  重建通过；DC冻结packed decode/prefill及单机512/多机2048阈值，旧配置默认
  继承原阈值。该配置仍opt-in，未宣称所有形状或所有节点数最优。
- [x] 在同一通信版本回测 EP8/EP16 四规模；把纯 GEMM 收益和端到端收益分别落表。
  AF/AS为纯GEMM，DC为当前CW数据面的阈值交错实验和最终四规模回归；六个
  token桶含阈值两侧的EP8/EP16数值检查共288/576项通过。双机长序列的最终
  性能目标仍未达成，不因M5勾选而结束后续优化。

### M5 后续 单机和多机统一 autotune

- [x] 区分物理 block M 与算术 slice M，接入现有 decode M16/M64 和 ring M128。
      decode 2-stage 已补 EP8/EP4/EP2 数值、batch-invariance 和 EP8 rank0/7
      专项 memcheck/racecheck；新网络算术模板仍待实现及独立验收。
- [x] 接入计算、通信及容量的联合候选搜索、资源过滤、内存预算和多路由完整
      Graph rank-max 选优；包含 incumbent 对照、确认与最终验证。
- [x] 导出精确形状/拓扑/容量表并接入 prepare；EP 组一致检查编译配置 ID
      和容量。显式表缺项时报错，未启用表保留原配置；不热切换不兼容 profile。
- [ ] 补充训练、TP 真实适配器路径和同进程多编译 profile 选择；网络新增
      算术 M 必须先实现，再完成数值和生命周期门禁。
- [ ] 验证自动选型前后的 batch-invariance、热点 no-drop、图/权重复用，
      在单机 EP8 和双机 EP16 的 1/64/8K/20K 回归，再扩展模型形状。
      DT EP8/T1/T64的18候选搜索、交错确认/导出和prepare查表GPU验证已完成，
      选中库通过既有逐位/复用检查。PF EP8/8K/20K的22候选搜索、交错确认、
      stride53历史矩阵复验、prepare查表及各240项集成检查也已完成；综合选中
      的K64/B256在8K历史均衡路由退化，未替换默认。双机长文及完整发布门禁
      仍在进行，详见autotune记录。

这些状态不改变上方 M5 的历史消融。离线工具初次实现时 GPU 被 RL 作业占用；
09-30 空闲后已开始真实搜索，首轮暴露了 uneven 夹具容量参数和禁用 prefill
模板实例化问题，修复及验收见 [autotune 记录](autotune.md)。完整 EP8/EP16
搜索、独立回归和发布门禁尚未完成，不将候选编译通过当作自动选优性能结论。

### M6：训练 Forward tape 与 Backward 对接

M6a已将rank-private Backward U/dAB/dX/dGate从固定pool移到调用级存储，按
实际R加有界expert padding分配；EP8/EP16均匀/热点数值及memcheck通过。
当时训练仍走V3，ring plan/checkpoint及peer arena迁移未完成。随后为Backward
增加完整cluster退出边界，逐任务同步粒度不变；重复调用的rank1/9完整报告中
DSM/payload hazards已消失，剩余共享原子报告由独立合法握手复现为工具误报。
训练Forward及其它rank仍未完成全量sanitizer门禁，不能声称训练sanitizer全部通过。
M6b补充了训练Runtime每调用tensor preparation的一致失败边界：预分配并预热
小控制区，Forward/Backward的本地分配回调结束后只在全部rank成功时进入执行。
EP8/EP16单rank OOM注入、partial allocation释放和Backward tape重试数值通过；
这不是首次pool创建/失效context恢复，也没有启用training ring或完成StableTape。
M6c进一步将Backward peer dR/contribution/dGate改为execution slot拥有的实际R、
本地T×K sized allocation，用CUDA IPC descriptor表访问同节点peer；换buffer时
先确认所有旧mapping已关闭，再释放旧export。三个live tapes及热点迁移的
EP8/EP16数值和显存回缩检查通过。M6c当时仍保留V3 Forward完整arena，新payload
与之并存，尚不是总训练显存节省，也没有启用training ring。
M6d随后接通单机training ring：冻结canonical Notify和物理window plan后，
用collective准备边界分配实际R的tape；Pull双写X/checkpoint后才发布ready，
FC1按canonical行保存BF16 G/U，Backward不持有物理ring。EP8的T1/64/512/8192/
20480数值、20K全热点R655360、三个live tapes及OOM重试通过；T512小ring的
rank0/7 memcheck0、Forward planner/main racecheck0。
M6e增加network checkpoint和有界Backward GIN window adapter；两个方向的
wire控制区、NIC counter编号独立持久化，仅在执行完成后复用context。两次S2G
都完成才发布X-ready及RX consumed，expected仍是逻辑route数。EP16 T1/64/8K/
20K及20K热点R1310720通过，T512/T2048分别覆盖decode/prefill的rank0/15
memcheck0和Forward racecheck0。没有恢复完整Forward X/S，也没有双机专用索引。
以上双S2G是M6d/M6e的历史实现；FM已将SavedX移至唯一FC1 Ntile0的ready A
stage，FZ再将此工作分到轮转pair的两个半M view；Pull只写物理X，当前生命周期
以第5节为准，expected仍未改变。
Python入口只为Notify/调度元数据预留最坏route容量，X/S保持有界，tape/dR按
实际R分配。M6f将Backward private padding从R+expert slack改为精确P_actual：
materialized Notify随已有status下载保存逐专家M128取整后的总行数，不增加host
同步或计数下载；空rank仅一行用于合法TMA描述符，实际计算任务仍为0。caller
复用tensor storage同时检查P容量，fused/replayed Notify不作为训练精确分配依据。
详见 [训练对接记录](static_forward_v4_training_record.md)。

- [x] 单机canonical tape、plan/实际分配/main复用、checkpoint纳入slot生命周期。
- [x] 单机Forward/Backward参考、长文热点、多个live tapes、OOM一致失败与重试。
- [x] 多机checkpoint及Backward有界GIN window adapter；通用节点索引/分区，EP16验证。
- [x] 精确P_actual分配（M6f）；EP8/EP16数值、零行rank、热点迁移、OOM重试、
  memcheck及非ring回退通过，inference四规模正式回归无稳定退化。
- [x] 训练profile解耦（M6g）：unified Backward及独立sender配额，旧计划0取
  Backward默认16；数值、热点/多tape、memcheck和正式性能回归通过。Backward
  恢复到冻结V3附近，未宣称训练Forward或完整训练性能门禁完成。
- [x] 合并训练G/U checkpoint写回（M6h）：复用退休B arena及已有barrier，
  单/双机数值与Forward专项sanitizer通过，长文main有小幅收益；修复纯GEMM
  诊断helper接口并通过packed/separated回归，不作为重新选择layout的证据。
- [x] SavedX复用FC1已就绪A stage（FM）：移除Pull第二次S2G，不改变canonical
  tape及Backward；单/双机数值、热点复用和Forward专项sanitizer通过，20K
  训练main改善约2.8%/3.1%–3.6%，整体训练性能目标仍未关闭。
- [x] SavedX地址外提（FO）及单机stable-source/M32 TMA（FR3）：保持网络原布局，
  FR3相对FO单机8K/20K训练main再改善4.72%/6.06%，wall改善3.61%/5.19%；
  单/双机集成门禁通过，专项sanitizer复用逐字节相同组件的已有证据。
- [x] 独立消融训练物理规划合入main（FS）：8K wall有小幅收益，但单机20K
  正反向均退化约0.3%–0.5%，已撤回；不以改变main计时范围掩盖结果。
- [x] FS2保留独立物理planner，仅移除冗余D2H/host同步：单/双机数值与
  Forward专项sanitizer已通过，单机长文wall正反向改善，双机长文持平。
- [ ] 完整训练sanitizer与性能门禁；不以新Forward通过覆盖旧Backward未决报告。
- [ ] 多机完成后才标注完整training V4支持；保留已验证的V3回退入口。

### 后续独立优化，不作为首版阻塞项

- [ ] 在后端契约确认后消融去掉 TX 热路径本地 completion 检查。
- [ ] 非一一绑定 TX/RX 的提前 TX 回收，仅在收益值得额外状态时引入。
- [ ] ring depth、Q、G/B、lane 数、comm warp 配额、owner 私有 lookahead。
- [ ] Return/Reduce 更细粒度 overlap；不改变已有数值顺序或打破有界窗口回收。

## 12. 验收与实验记录要求

### 12.1 正确性与进度

复用已有测试入口和 PyTorch 数值参考，补充能暴露本次协议风险的少量场景：

- 主矩阵：EP8=单机 8 GPU，EP16=双机各 8 GPU；T=1/64/8192/20480 per rank。
  2×4 EP8 只能作额外实验，不能替代单机 EP8。
- balanced、全部合法 route 指向同一 expert/rank、重复 expert、masked route、
  不等长/空 rank、tail Mblock、无远端命中、一个 token 在某节点命中多个 expert。
  非法 expert id 与合法 masked route 区分，不能把错误输入静默当成 drop。
- 小深度多次 ring wrap、变长 fanout、Return packet 跨 C slots、故意延迟某个
  RX reader/Return owner，验证 slot 退休不误用全局完成前缀。
- Graph routes/gates/inputs 变化、混合 profile、双层共享 workspace、连续 eager/graph
  调用；当前空输入 graph 的 pool 限制需明确处理或报告，不能冒充已通过。
- 每条有效 route 恰好执行/Return/combine 一次；独立于 packet 分片策略和 worker 数。
  同步正确性不只看最终 hash，还检查 no-drop、各代 target 和 drain。
- 任意节点数的枚举/offset 做小型模拟校验；三机以上性能和通信进度待实机验证。
- 不将外部 NIC 故障等价为普通 backpressure；错误终止复用 runtime 的协同 abort
  合同，不在每行 hot path 添加无依据的 system-scope 防御性轮询。

### 12.2 性能与归因

所有 GPU 核心固定 1620 MHz 并采样核实实际 SM/graphics 时钟，不使用无法维持的
1980 MHz。固定/记录其他环境、显存时钟与工作负载，正式计时不并发运行编译或其他
GPU 测试。沿用 graph CUDA-event rank-max p50，报告 warmup、sample 数和 replay batch。

每个性能 milestone 至少记录：

- commit、binary/config hash、拓扑、H/I/E/K、实际输入与路由分布、时钟；
- 修改前后交错 A/B（反转顺序），以及同条件 V3 对照；历史 V3 非本轮重测必须标明；
- kernel 端到端、可解释的 stage/timeline、峰值显存与协议计数开销；
- 已通过的数值门禁、失败候选、未测部分及是否为多项组合收益。

gate/up、GEMM tile/epilogue、网络节点去重、Dispatch overlap、Return overlap、
ring depth/窗口大小分别消融。多个改动一起启用只能报告组合收益，不凭直觉分摊。
重叠由 timeline 证明，不把并行 stage span 相加当成总延迟，也不拿旧的未完整赋值
device-clock 字段当通信计时。

实际实施、验证和测量以第 11 节及 ring 实施记录为准，未勾选项不能视为已实现。
