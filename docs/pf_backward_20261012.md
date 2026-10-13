# PF Backward 对齐现版 Forward（2026-10-12）

基线是 `main@1cfb423`；实现分支 `feature/pf-backward-20261012`。
旧静态 BWD 曾作为迁移期间的正确性/性能对照；M11 清理后只保留当前
PF stream BWD，旧代码可从 `7b558e8` 恢复。不将新 FWD 的收益外推到 BWD。

## M14：Qwen 数据流优化（进行中，2026-10-13）

目标：保持 Flash H4096/I2048/E256/top6 不回退，使 Qwen
H2048/I512/E256/top8 的 EP8/EP16、8K/20K、均衡/非均衡路由，
带缓存 FWD 和独立 BWD 全部达到 `Megatron / PF - 1 >= 15%`。
不将吞吐提升和延迟下降混用。整理后的 Qwen 配置在本轮 16 项复测中
全部超过目标，单机最小领先 40.2%、双机最小领先 15.9%。新的 Megatron
复测和最终 Flash 配对回归尚在进行，未将首轮达标当作最终回归完成。
Flash 默认路径较早的 16 项配对检查未发现系统性回退。

口径沿用 M13：1620 MHz、token 数正态随机 ±3σ（σ=10%）、seed 123；
4 warmups、16 samples，先完整回放 trace 预热。每步取最慢 rank，再取 p50。
FWD 包含当次权重重排、notify、缓存申请、main；BWD 使用当次新 FWD
tape，FWD 本身不计时。这里只是算子测试，非完整模型训练 step。
编译、数值检查和正式计时串行。

两个 shape 不能只按 TFLOPS 直接外推：每 token 的 FWD 专家 GEMM
为 `6 * H * I * topk`，Flash 是 Qwen 的 6 倍；contribution 字节数
为 `2 * H * topk`，Flash 只有 Qwen 的 1.5 倍。因此 Qwen 每字节
return 所对应的计算量仅为 Flash 的 1/4，而 top8 又比 top6 多 1/3
专家输入行及相关同步工作。较小 I 下，写回地址局部性、每 tile fence、
host 协商等开销更容易暴露，不能仅靠调 GEMM N/K/stages 假设解决。

### 单机：主要瓶颈是 contribution 地址布局，不是 GEMM tile

原布局 `C[topk_slot][max_tokens][H]` 让专家 GEMM 的相邻行写入离散
source-token 地址。训练 notify 已有每 source/expert 的稳定 ordinal；
将 C 地址改为 source 上的 expert-prefix + ordinal，reduce 再通过
route→C-row 表读回。只改变地址布局，不改变 GEMM 的行、BF16 舍入、
expert-sorted FP32 combine 顺序，不复用上一调用的路由计划。
Qwen 20K 测量容量下映射表约 0.79 MiB/卡，不增加 contribution payload。

| 单机随机路由候选 | FWD 8K / 20K (ms) | BWD 8K / 20K (ms) |
|---|---:|---:|
| M13（历史，本轮未改动二进制） | 4.598 / 14.398 | 5.918 / 16.212 |
| 2→3 pull warps | 4.197 / 12.774 | 5.507 / 15.204 |
| warp 领取 M-block，批量解码行，vector pull | 4.266 / 11.342 | 5.370 / 14.660 |
| 再加紧凑 contribution | **3.372 / 7.031** | **4.272 / 9.399** |

最后一项均衡路由为 FWD 3.248/5.916、BWD 3.431/7.468 ms。
同环境 Megatron 复测后，单机 8 个 case 首轮全部超过目标，最小吞吐
领先约 36.9%；最终生成配置仍须极端路由/多 tape 检查和确认轮。
artifact `build/pf_main_timeline/qwen_blocks_compact`；日志在
`build/qwen35_opt_20261013/results/compact_*`。
129/8K/20K 各两次变化路由，output/dX/dGate/dW1/dW2 校验已通过。

同环境 Megatron 复测和紧凑地址单独消融：

| 单机 | Megatron FWD / BWD (ms) | 紧凑 C + 原逐行 bulk pull，3 warps | 紧凑 C + M-block vector pull，3 warps |
|---|---:|---:|---:|
| 均衡 8K | 5.662 / 5.899 | 2.784 / 3.562 | 3.248 / 3.431 |
| 均衡 20K | 10.010 / 10.833 | 6.143 / 7.736 | 5.916 / 7.468 |
| 随机 8K | 6.225 / 6.093 | 3.258 / 4.406 | 3.372 / 4.272 |
| 随机 20K | 11.646 / 12.867 | 7.767 / 9.954 | 7.031 / 9.399 |

两种方案单机 8 项均达到吞吐领先 15% 的目标；大部分收益来自紧凑 C，
不能全部归功于 vector pull。原逐行 pull 在 8K FWD 更快，M-block
pull 在 20K 更快；尚未宣称任何一项在所有规模占优。
本次将 `local_pull_warps`、`local_pull_mode`（逐行 bulk / M-block bulk /
M-block vector）和 `local_compact_return` 纳入 autotune 参数及 compiled ID，
不修改 Flash 默认值。历史额外 `--define` 候选仅用于消融，其 artifact
SHA 和定义记录在 manifest 中，不冒充最终生成配置。

没有保留的消融：pull 4-buffer 异步流水、提前释放 shared epilogue、
单独 return warp、math warp bulk return、额外 source route+gate 打包。
它们没有稳定收益，部分反而变慢；没有据此断言某种拷贝指令普遍更快。
单机旧布局的插桩 timeline 显示 20K FC2 epilogue 明显拖长，纯 GEMM
诊断相对 cuBLAS 约差 12%–13%，不足以解释完整路径的大幅差距。
未成功取得可靠 TLB/warp-stall 计数器，不能将地址布局收益归因为已证实的 TLB miss。

### 双机：已测试的消融与下一步

在不提前释放 TX 的前提下，将 BWD return 的逐 fragment flush/wait
改为最多 4 个请求的 FIFO。仅最后一个 fragment 退役后释放 M-block
的 TX credit，跨 peer 的前序请求同样必须退役。均衡路由有小幅收益；
将完成信号分散到 channel 的消融没有进一步收益，已从源码撤销。

再把 BWD contribution 从 window-major 改为跨 window 的 source/expert
连续布局，可减少 return fragment，但收益远小于单机：

| 双机 BWD | 均衡 8K / 20K (ms) | 随机 8K / 20K (ms) |
|---|---:|---:|
| M13（历史） | 8.519 / 15.791 | 8.441 / 18.012 |
| 有界 pending=4 | 7.983 / 14.931 | 8.185 / 18.251 |
| 再加跨 window 紧凑 return | 7.830 / 14.692 | 8.073 / 18.189 |

以上双机候选均通过完整梯度检查，仍未达到目标。随后分别验证了：
直接利用 FWD canonical 行 tape 省去 BWD 的 FWD-window 目录重建、
机内转发 vector copy、按真实消费者缩小 system-fence 范围。
NIC 使用的 TX 数据仍保留发布与完成 credit，不用数值偶然通过代替同步证明。

canonical planner 第一版同样通过全部梯度，但未保留其“先转发完整本地 dY，
再消费远端 RX”的顺序：均衡 BWD 7.549/15.192、随机 8.310/19.705 ms。
这种顺序延后 RX credit，20K 回退；修订版恢复 window 内的本地/远端
交错，只从 source route-order 直接取 canonical saved-row，不重建 FWD
物理目录。第一版再叠加 vector forwarding 更慢：
均衡 BWD 7.613/15.541、随机 8.493/20.194 ms；FWD 6.320/12.893、
6.718/13.423 ms，与历史基线相近，不据此替换原 bulk copy。

canonical planner 恢复 window 内交错后的 BWD 为均衡 7.988/14.564、
随机 8.038/18.255 ms；再加 deferred local fence 和容量 fast path 为
7.905/14.534、7.920/18.435 ms，没有足够收益支持保留全部复杂度。
该组合 FWD 为均衡 6.325/12.911、随机 6.652/13.350 ms。
同环境 Megatron 双机 FWD 为均衡 7.475/14.288、随机 8.351/17.474；
BWD 为均衡 6.764/15.184、随机 7.936/18.932 ms。

通信隔离：用现有 identity diagnostic 替换专家 Jacobian，保留实际
dispatch、GM 读写、TX credit、return、reduce。随机 8K/20K 的设备
main envelope 仍为 6.055/15.166 ms，dispatch envelope 为 2.930/6.832，
return tail 为 3.828/8.193，reduce 为 0.305/0.760 ms。各项为独立
rank-max p50、可重叠，不可相加；identity 有额外 GM 读，且启用了
progress 输出，不能当作纯 RDMA 峰值或正式完整 BWD 性能。
日志 `identity_net_current_comm*`，129 token identity 数值检查通过。
这支持继续优先改变 return 数据组织，而非只调 GEMM。

节点内 FP32 partial combine 消融（未保留）：复用 dispatch 按节点去重的
token 目录和包顺序；中转仍采用 source/window/expert 紧凑 BF16 地址，
避免重新引入随机 token 写回。节点 partial 用 FP32 发送，最后转 BF16，
不额外引入 BF16 舍入；节点归约改变 FP32 加法结合顺序，必须重新做
原容差数值检查。先测整段 join 后发送，再测 source-window 完成后流水
发送。后者用每 GPU/window 的精确完成行计数，M-slot credit 不等待 NIC
读取全调用中转区，packet scratch 仍等待本地 NIC completion 后复用。
这些实验均已撤回，不是生产路径。

负结果：整段 join 后仅用一个 warp/CTA 合并发送，均衡 FWD
14.560/31.210、BWD 15.915/34.070 ms；窗口流水原型分别为
20.681/46.333、22.782/52.760 ms。减少 wire bytes 并没有补偿额外
合并/打包开销，不能保留为默认。该原型新增的 GEMM/通信 helper
没有寄存器 spill，不能将此次回退简单归咎于 spill。

将每条 token 的 contribution 地址解析和排序从 H 列循环中提到外面后，
8K 单次插桩 main 约 10.73 ms；各 rank 最后 FC2 完成于 main 后
1.93–3.53 ms，最后 pack 为 6.96–9.68 ms。单 CTA 累计 pack
5.22–7.92 ms，而累计 NIC wait 0.35–0.55 ms；不同 CTA 的区间
重叠，不能相加。此证据指向少量打包 warp 上的串行访存，而非 GEMM
或 NIC wait。再增加四组独立向量累加，随机 8K FWD/BWD 为
9.408/12.324 ms，仍不达标；这版 FWD helper 还出现少量 spill，未保留。
随后隔离消融直接批量发送 GEMM 已写好的连续节点区间，不做 FP32
partial combine：随机 8K FWD/BWD 为 6.925/9.950 ms；再加本地
ready 推送为 6.906/9.710 ms。逐包 ready 则为 9.356/12.023 ms。
均未保留。负结果原型保存在 ignored build 目录的补丁归档中。

后续有效改动是**先消除窗口之间的错误依赖，再改通信发射节奏**：

1. dispatch 节点去重目录按 token 递增生成；连续 token 合并为一个 Get。
   不增加 pack buffer，不多发送未命中节点的 token，不改变 RX credit。
2. 训练 tape 从 `[expert][source][source 的所有窗口]` 改为
   `[expert][window/group][source][ordinal]`，FWD/BWD 使用同一稳定行序。
   cache 仍按 expert 紧凑、不恢复窗口 M-padding、不增加激活容量。
   原来的 M-block 可能跨越一个 source 的末窗口和下一个 source 的首窗口，
   从而使首窗口部分行一直等到最后才完成。新顺序解除这类跨窗口依赖。
   dW 的 K 行遍历顺序改变，不能仅验证 dX；129/8K/20K、两次新 tape 的
   output/dX/dGate/dW1/dW2 均通过原容差，最大 dW relative-L2 约 4.72e-4。
3. 配合有界 pending=4 的 return FIFO，不在每个 fragment 后立即等 NIC。
   TX 仍必须等所有 fragment 本地完成才回收；B128、8K 的强制环复用验证通过。

| 双机候选 | 均衡 FWD 8K / 20K | 随机 FWD 8K / 20K | 均衡 BWD 8K / 20K | 随机 BWD 8K / 20K |
|---|---:|---:|---:|---:|
| 有序 dispatch + 连续 Get | 5.965 / 12.469 | 6.192 / 12.928 | 8.040 / 15.063 | 8.040 / 17.975 |
| 再加 window-ordered tape | 5.853 / 12.506 | 6.032 / 12.984 | 6.856 / 14.349 | 7.603 / 17.565 |
| 再加 pending=4（仅 BWD） | — | — | 6.503 / 13.663 | 7.395 / 17.463 |

单位 ms，仍是候选首轮、尚未全部达标。`window_tape_progress` 的均衡
8K 插桩中，main envelope 5.190 ms，activation envelope 2.384 ms，
dW 0.698 ms、return tail 2.564 ms、reduce 0.400 ms；这些区间可重叠，
不能相加或当作未插桩正式计时。

`dGate` 随 `dX` 合包的消融未保留：每行尾部多留 16B 保持向量对齐，
两条 Put/完成信号合为一条，129/8K/20K 完整梯度检查通过，但均衡
BWD 6.488/13.738、随机 7.521/17.522 ms，没有稳定收益；新增 padding
及专用分支已撤销。不能把当前尾部开销简单归因于 dGate 小消息。

window-ordered tape 每 expert 的 last-window key 单调，可用各 sorted
run 的二分 rank 在一个 kernel 内合并静态任务顺序，替代多轮 merge
和最后的复制。复用原 dU 计数 scratch 保存排列，不新增显存；所有
空 expert/rank 必须合法。129/8K 梯度检查通过；均衡 BWD 首轮为
6.449/13.711 ms，减少 kernel 数量，但不宣称这几十微秒差异是稳定收益。

继续消融（尚未选作默认）：静态 return 归属逐轮旋转 5 个 worker，
均衡 BWD 6.349/13.607 ms、随机 7.592/17.449 ms，随机 8K 回退，
不能仅据均衡 case 选型。把相同的 112 个数据通信 owner 从 Get64/Put48
改为 Get48/Put64，总 CTA/warp/context 数不变，结果如下；增加 return
worker 没有普遍收益，20K FWD 反而变慢。

| Get48/Put64 | 均衡 8K / 20K (ms) | 随机 8K / 20K (ms) |
|---|---:|---:|
| 缓存 FWD | 5.920 / 12.546 | 6.126 / 13.699 |
| BWD | 6.438 / 14.260 | 7.353 / 17.350 |

两者均通过 129/8K 两个新 tape 的梯度检查。基准包含随机 token 打乱，
实际 task key 还按 source group/window 排序，因此不能用理想均衡的
整除关系直接断言固定比例 worker 空闲。

预计算 BWD return span：利用 `CanonicalRow` 的 4B 对齐空隙记录
`min(当前 source/expert/window 剩余行数, 当前 canonical M-block 剩余行数)`。
只对 window-ordered tape 启用，连续区间在 planner 中已知，不增加容量，
Return 不再逐行扫描确认 source/address。129/8K/20K 两次新 tape 梯度通过；
均衡 BWD 6.352/13.606 ms、随机 7.431/17.306 ms，小幅改善，仍未达标。

当前 dU 每 M-block 两个 N-tile，设备 132 CTA，所以补测 66/132 块的
发射批次，检查整轮 CTA 对齐是否有帮助。两者 B128、129/8K 强制环复用
梯度均通过。66 块：均衡 BWD 6.389/13.638、随机 7.339/17.556 ms；
132 块：6.332/13.639、7.338/17.255 ms。差别很小，不能把 132 块认定
为稳定胜出，更不能用它宣称已解决主瓶颈。

增大 window 到 B4096（其余保持 span/64-wave 候选）也没有帮助：
均衡缓存 FWD 6.567/12.852 ms、随机 6.889/13.310 ms。减少 window 数量
或 M-padding 不自动等于提升完整路径，通信/计算衔接同样受影响。

将 network dR 改为当前 BWD workspace 内的私有精确尺寸缓冲，跳过
原 peer workspace 每调用的容量协商（network dR 不供 peer 直接写）。
复用现有 tensor allocation agreement，不取消分配失败的全 rank 一致处理。
129/8K/20K 两次新 tape 的完整梯度通过；均衡 BWD 6.408/13.551 ms、
随机 7.339/17.371 ms。相对 span 候选基本是波动，不能记为明确收益。

生命周期检查还复现了独立于该改动的旧问题：standalone notify 已完成后，
若缓存申请失败，异常分支会永久隔离 activation slot，导致重试失败。
前一版 `qwen_window_spans` 同样失败。Notify 只更新 rendezvous epoch、
规划元数据，尚未推进数据面的 ring generation；集体分配失败协议已经
同步完所有 rank 的 Notify，因此该分支可以释放槽位、重新规划。
`lean_vector_lifecycle` 已通过：EP16/B128，集中到一个 rank 的路由、空
rank、真实 CUDA OOM 重试、三份移动热点 tape、原地更新权重、关闭释放，
以及 EP16 训练→EP4 推理/graph/offload。保留原有集成测试，没有添加
函数名或编译符号匹配测试。

删除当前数据流不再读取的旧规划：32-row striped dY 顺序，以及额外的
training packet/fanout tape。当前 BWD 根据 retained route/counts 重建
PF ring 目录；不依赖这两张旧表。再独立消融 network warp-vector pull：
每 warp 按 uint4 协作拷贝，替代逐行 shared bulk-copy/wait。无新增 spill。

| B2048/D2 | 均衡 FWD 8K / 20K | 随机 FWD 8K / 20K | 均衡 BWD 8K / 20K | 随机 BWD 8K / 20K |
|---|---:|---:|---:|---:|
| 精简旧规划，bulk | 5.725 / 12.339 | 5.881 / 12.690 | 6.210 / 13.636 | 7.393 / 17.477 |
| 同上，warp vector | 5.760 / 12.218 | 5.883 / 12.678 | 6.279 / 13.520 | 7.393 / 17.114 |

单位 ms。FWD 首轮四项超过 15%；vector 没有普遍改善 BWD，不能据此
宣称 bulk copy 是剩余瓶颈。最终配置及 Flash 默认路径仍需确认轮。
前一版本 private-dR 的 B1024/D2 结果：均衡 FWD 5.712/12.426、BWD
6.142/13.296，随机 FWD 5.924/12.922、BWD 7.125/17.439 ms；不能用
较小窗口在一个 case 的收益外推到全部 case。

后续两项未保留（源码快照 `build/qwen35_opt_20261013/prototypes_08.patch`）：

- dX Put 暂缓 doorbell，由同 peer/context 紧接的 dGate Put 一并提交；
  使用生产者已有的 system release，省去重复 fence。仍保留完成信号、
  flush 和本地 TX credit。B128、129/8K 梯度通过；均衡 BWD
  6.322/13.571、随机 7.305/17.292 ms，没有明确收益。
- 在此基础上，BWD 每 token window 内先处理远端 source groups，期望
  更早启动 return。只旋转静态任务及 dY 转发顺序，不改变 tape/GEMM
  行内容。B128、129/8K/20K 梯度通过；均衡 BWD 6.513/13.729、随机
  7.208/17.301 ms。均衡 8K 回退，不能保留为统一默认；更早等待远端
  数据可能损失本地计算对 dispatch 的遮盖，尚不能把这个解释当作计数器证据。

进一步独立消融（基于精简规划的 bulk 候选，其余不变）：

| 双机 BWD 候选 | 均衡 8K / 20K (ms) | 随机 8K / 20K (ms) |
|---|---:|---:|
| activation K128、2 stages | 6.303 / 13.560 | 7.453 / 17.492 |
| dispatch packet 32→64 rows | 6.327 / 13.579 | 7.292 / 17.340 |
| wire credit windows 2→3 | 6.308 / 13.581 | 7.276 / 16.959 |
| wire credit windows 2→4 | 6.243 / 13.552 | 7.261 / 16.453 |

前两项通过 129/8K 的完整梯度检查、无新增 spill，但没有稳定全面收益。
credit 增大对随机 20K 有帮助，仍未达标。这里改变的是
`WEFTMOE_PF_CREDIT_WINDOWS`，不是 benchmark 的 contribution `--ring-depth`；
两者不能混称 D。双机 132 channels、最小 channel depth=2 时，credit=2
有 528 个 TX M-slot，credit=3 有 792 个；Qwen 每增加一个 credit window
约增加 198 MiB/卡 registered ring 容量。credit=4 随机 20K 首轮刚好
超过 15%（Megatron 18.932 ms），其它三项仍未达标，须保留确认裕量。

全 CTA Reduce 消融：原来每 CTA 96 个通信线程在 return 后 reduce，
与 288 个计算线程的 dW 重叠；新候选先 join dW/return，再用 384 个
线程 reduce。减少尾部访存等待，但失去原有重叠，不能假定总是更快。
单机与默认 Flash 路径不启用这一开关。

| 双机 BWD | 均衡 8K / 20K (ms) | 随机 8K / 20K (ms) |
|---|---:|---:|
| full-CTA reduce，B2048/credit2 | 6.009 / 12.952 | 7.065 / 17.856 |
| full-CTA reduce，B1024/credit4 | 5.888 / 12.664 | 6.717 / 16.008 |

B128、129/8K/20K 两次新 tape 的完整梯度检查通过，最大 relative-L2：
output 1.40e-4、dX 1.27e-4、dGate 1.46e-7、dW1 4.24e-4、dW2 4.71e-4。
第一行随机 20K 反而回退，因此不将 balanced 的收益外推。第二行首轮
三项超过 15%，均衡 8K 仅 14.9%，还不能宣布全部完成。
`backward_full_reduce` 和真正控制 wire 容量的 `network_credit_windows`
均加入生成配置；默认分别为 0、1，保留原环境变量显式覆盖。
允许 `network_min_depth` 搜索 1/2/4；总深度仍不低于完整最坏窗口所需容量。
无稳定收益的 return worker rotation 分支已撤销。

最后精简 BWD planner：canonical return 不使用 FWD 的 Return packet
prefix/directory、row-output 指针和 publication mask，跳过其构建；跨机
只发布 dispatch 所需的 metadata 前缀。保留 dispatch packet、route location、
checkpoint row 映射、credit 初始化及所有必要的 barrier。FWD 的 gates/headers
已经属于 tape，BWD 不再重复写入。不是复用前一次调用的路由计划。

生成候选 `0b87af4c8ddf`，B1024/credit4、full-CTA Reduce，直接使用 header
中的 ring 配置、不加环境覆盖：BWD 均衡 8K/20K 为 **5.795/12.505 ms**，
随机为 **6.732/15.984 ms**；对本轮 Megatron 分别领先 **16.7%/21.4%、
17.9%/18.4%**。首轮四项全部超过目标，仍须最终复测、FWD 与 Flash 回归。
129/8K/20K、B128 强制环复用、两次新 tape 的完整梯度检查通过。
日志 `lean_bwd_plan_screen_*`、`lean_bwd_plan_verify_*`；本次没有添加碎片单测。

容量组合并非单纯增大显存：B2048/credit2→B1024/credit4 后，8K 的
compute pool 从 1807.83 降到 1163.64 MiB，GIN 从 1169.14 增到 1533.14 MiB；
20K 分别从 2549.09/1693.20 变为 1910.05/2089.75 MiB。固定 pool 总量
反而减少约 280/242 MiB。这里不包含每 tape 的 X/GU、输出/梯度及 BWD
workspace，不把固定 pool 统计冒充完整训练峰值显存。

### 整理后 Qwen 配置的首轮正式复测

`qwen_release_candidate`，compiled ID `0b87af4c8ddf`；EP8 B2048/D2，
EP16 B1024、channel floor=2、credit4，使用生成配置，没有 ring 环境覆盖。
下表比较同一 workload SHA 的 `mt_refresh_*`；单位 ms，括号为吞吐提升。
不是将完整 FWD+BWD p50 拆成两个时间，也没有排除每调用的前置与缓存申请。

| 拓扑 / 路由 / tokens | PF 缓存 FWD | Megatron FWD | PF BWD | Megatron BWD |
|---|---:|---:|---:|---:|
| EP8 均衡 8K | 3.036 (+86.5%) | 5.662 | 3.299 (+78.8%) | 5.899 |
| EP8 均衡 20K | 5.759 (+73.8%) | 10.010 | 7.098 (+52.6%) | 10.833 |
| EP8 随机 8K | 3.121 (+99.5%) | 6.225 | 4.105 (+48.4%) | 6.093 |
| EP8 随机 20K | 6.754 (+72.4%) | 11.646 | 9.180 (+40.2%) | 12.867 |
| EP16 均衡 8K | 5.656 (+32.2%) | 7.475 | 5.781 (+17.0%) | 6.764 |
| EP16 均衡 20K | 12.331 (+15.9%) | 14.288 | 12.559 (+20.9%) | 15.184 |
| EP16 随机 8K | 5.742 (+45.4%) | 8.351 | 6.709 (+18.3%) | 7.936 |
| EP16 随机 20K | 12.629 (+38.4%) | 17.474 | 15.797 (+19.8%) | 18.932 |

日志 `release_local_*` / `release_network_*`。整理后的单机 129/8K/20K
完整梯度通过；双机的 B128 小环验证同样通过。`release_lifecycle` 覆盖
集中热点、空 incoming rank、真实 CUDA OOM 重试、三份独立 tape、原地
更新权重、工作区关闭释放，以及 EP16 训练→EP4 推理 graph/offload。
没有新增细粒度单测；现有四个 autotune 测试文件为 46 passed、15 subtests。
后续新的基线与 Flash 配对确认另列，不覆盖此前消融记录。

### Flash 默认路径首轮配对回归

所有 Qwen 专用新开关关闭。下表为当前/旧版 ms，旧版 artifact 为
`cleanup_release_20261013`，当前为 `flash_regression_default`；16 项中
最大慢约 0.73%，其余差异也很小，没有发现系统性退化。仅一轮，不能
将小幅波动认定为优化收益。最终清理后的二进制需再次复测。

| Flash case | 带缓存 FWD 当前 / 旧版 | BWD 当前 / 旧版 |
|---|---:|---:|
| EP8 均衡 8K | 6.744 / 6.706 | 10.693 / 10.636 |
| EP8 均衡 20K | 14.865 / 14.912 | 25.287 / 25.788 |
| EP8 随机 8K | 7.875 / 8.006 | 12.967 / 13.048 |
| EP8 随机 20K | 18.067 / 18.064 | 31.563 / 31.708 |
| EP16 均衡 8K | 9.975 / 10.047 | 13.269 / 13.359 |
| EP16 均衡 20K | 21.173 / 21.177 | 27.052 / 26.857 |
| EP16 随机 8K | 12.058 / 12.155 | 16.033 / 16.281 |
| EP16 随机 20K | 27.842 / 27.859 | 38.864 / 38.838 |

另一个同步消融用两类消费者区分可见性：NIC TX 仍逐 tile system fence；
仅在整次调用结束时消费的本节点 contribution 延迟到计算结束统一 drain。
BWD reduce 额外等待计算完成计数，防止通信线程先进入 reduce；FWD
已有全 CTA 的末尾 rank barrier。暖态容量优化则复用 FWD 计数得到的
全局最大 rows，以及上次 IPC 交换得到的所有 rank 最小容量，只有二者
证明全体都无需扩容才跳过协商；极端热点仍走原先的集体扩容/错误路径。
这些同步/容量改动没有足够收益，已撤回；保留实验记录，不保留生产代码。

后续交付待办：选出最小有效改动；将保留的参数纳入 autotune/compiled ID；
复测 Megatron 与 Flash；更新正式表格与冻结 profile；整理源码并提交 milestone。

## M13：Qwen shape 专属 autotune（2026-10-13）

在 M12 的同一份 `main@bb444c2` CUDA 代码上离线搜索，没有修改生产 kernel、
Flash/DT 默认配置。交付文件：

- [搜索配置](../configs/pf_qwen35_training_autotune.json)。
- [冻结 profile、编译/加载/复现说明](../configs/pf_qwen35_20261013/README.md)。
- [EP8/EP16 tune table](../configs/pf_qwen35_20261013/table.json)。
- [候选、确认轮和独立阶段结果 JSON](../benchmarks/results/qwen35_pf_autotune_20261013.json)。

### 搜索与选型

固定 H2048/I512/E256/top8、BF16、1620 MHz，沿用 M12 的随机 token 数和
合成非均衡路由。优化目标是 8K/20K **完整 FWD+BWD 周期**的 rank-max p50
几何平均；不是独立 BWD 时间，也不是全部路由分布的统一最优。

分 FWD、activation、dW、ring 四轮，分别比较：
FWD N128/256 与 2–5 stages；dU/dX 独立 N128/256、2/3/4 stages、
K128/两级流水；dW M128/N256、M256/N128、M128/N128、两级流水、
chunk 1/1、2/2、4/2；最后比较 ring 容量。
每轮保留 incumbent，前两名正反顺序确认两次，提升不足 1% 不切换。
实际各拓扑 25 次初筛（包含重复 incumbent），共复用 18 种编译配置，
编译与 GPU 计时串行，不在计时过程中编译。

初始网络 ring 网格还包含 B2048/D4 重复对照；网络 contribution 容量
由整次调用推导，单改 D 不改变实际数据路径。该项未被选中，交付配置已
去掉；原始成绩仍保留。真正入围确认的是 B1024 与 B2048。
B512 双机完整周期约 54.9/122.0 ms，明显回退，不能将缩窗口当作纯省显存。

**单/双机最终都只改变 `backward.dw1_claim_tiles: 4 → 1`**：
dW1/dW2 chunk=1/1，FWD/dU/dX/dW 继续使用原 M128/N256/K64、三级流水，
ring 保持 B2048/D2，network lanes132、D32/R128，min depth/credit windows=2/2。
compiled ID=`eab65a9995ff`，config ID=`481cffe7fbc7`。

dW 确认轮的完整周期：EP8 8K/20K 为 10.958→10.639 / 32.123→30.609 ms；
EP16 为 15.007→14.881 / 33.317→31.539 ms。综合延迟降低 **3.81% / 3.12%**。
最后一次完整周期复测分别为 EP8 10.728/30.720、EP16 14.879/31.521 ms，
不可将它们与独立阶段 p50 相加/相减混用。

此 shape 每专家 dW1 有 `(1024/128)*(2048/256)/2=32` 个 cluster tile-pair。
chunk4 将其合为 8 个静态任务，chunk1 则保留 32 个，能将热点专家分给
更多 cluster，而不改变 GEMM 的 M/N/K 或 reduction 顺序。更细任务有利于
减轻非均衡尾部，但也增加任务切换；以下均衡结果说明收益有 workload 依赖，
不能据此把 1/1 推广为所有模型的全局默认。本轮未额外做 timeline 归因。

### 独立 FWD/BWD 验收

同 M12 的计时边界：FWD 带 X/GU 缓存，BWD 每步在计时外生成新 FWD tape；
包含每调用权重准备/缓存与临时分配，排除首次固定池/IPC 初始化。
原配置和 tuned 各跑一轮相同 trace，20 prime、4 warmups、16 samples；
单机均衡 BWD 发现小幅回退后，额外做反序一轮，该两行取两轮 p50 均值。
本轮原配置也重新实测，不用历史 PF 数字代替对照。每项保存原始 p50。

Megatron 列引用 **M12 已测的两轮均值，本轮没有重测 Megatron**，
仍为 Flex/DeepEP20SM + TE grouped GEMM、`moe_act` 缓存口径。
表中单位 ms；末列为 `Megatron / tuned PF - 1` 的吞吐变化，不是延迟降幅。
所有原/tuned/FWD/BWD 与 M12 的逐 rank trace hash、token 数和专家行数一致。

FWD：

| 拓扑 | 路由 | tokens/rank | 原 PF | Tuned PF | Megatron（M12） | Tuned 吞吐相对 MT |
|---|---|---:|---:|---:|---:|---:|
| EP8 | 均衡 | 8K | 3.621 | 3.673 | 5.949 | +61.9% |
| EP8 | 均衡 | 20K | 10.845 | 10.904 | 9.845 | -9.7% |
| EP8 | 非均衡 | 8K | 4.631 | 4.598 | 6.244 | +35.8% |
| EP8 | 非均衡 | 20K | 14.397 | 14.398 | 11.428 | -20.6% |
| EP16 | 均衡 | 8K | 6.313 | 6.331 | 7.205 | +13.8% |
| EP16 | 均衡 | 20K | 12.913 | 12.955 | 14.288 | +10.3% |
| EP16 | 非均衡 | 8K | 6.537 | 6.567 | 8.038 | +22.4% |
| EP16 | 非均衡 | 20K | 13.289 | 13.366 | 17.176 | +28.5% |

BWD：

| 拓扑 | 路由 | tokens/rank | 原 PF | Tuned PF | Megatron（M12） | Tuned 吞吐相对 MT |
|---|---|---:|---:|---:|---:|---:|
| EP8 | 均衡 | 8K | 4.621 | 4.747 | 5.663 | +19.3% |
| EP8 | 均衡 | 20K | 13.266 | 13.118 | 10.773 | -17.9% |
| EP8 | 非均衡 | 8K | 6.228 | 5.918 | 5.821 | -1.6% |
| EP8 | 非均衡 | 20K | 17.522 | 16.212 | 12.767 | -21.2% |
| EP16 | 均衡 | 8K | 8.575 | 8.519 | 6.677 | -21.6% |
| EP16 | 均衡 | 20K | 15.553 | 15.791 | 15.175 | -3.9% |
| EP16 | 非均衡 | 8K | 8.394 | 8.441 | 7.889 | -6.5% |
| EP16 | 非均衡 | 20K | 19.507 | 18.012 | 19.022 | +5.6% |

结论：**FWD 没有确认收益**。非均衡 BWD 的 EP8 8K/20K 延迟降低
5.0%/7.5%，EP16 8K 持平、20K 降低 7.7%。
均衡不是普遍受益：EP8 均衡 8K 两轮均确认约 **2.7% 回退**，
20K 约改善 1.1%；EP16 均衡 20K 单轮约慢 1.5%。
该 profile 按非均衡联合目标选出，保留这些取舍，不宣称全面无回退。
单机 20K 仍落后 M12 Megatron；双机非均衡 20K BWD 从接近/略落后变为
领先约 5.6%。这些超参搜索只追回部分差距，没有消除所有剩余瓶颈。

### 验证与产物

单/双机随机 129-token 入围检查与最终 8K/20K 检查通过；
另补均衡 129/8K/20K 各两组 output/dX/dGate/dW1/dW2 检查，全通过。
最终随机/均衡大规模检查最大 relative L2 为 **0.000464 以下**。
选中配置单/双机 BWD main 均为 168 个初始寄存器、零 stack/spill；
未改变既有 mini-notify 的 spill。

全部 9 次 tuned 独立阶段启动实际使用交付的 `table.json` 加载，通过 EP8/EP16
和实际 prepared capacity 校验。同一头文件可用于两种拓扑；table 的
`max_tokens` 是 EP8 10055/25820、EP16 10277/25820，不是名义 8192/20480。
这是精确容量离线查表，不会自动在首调用重新搜索，也不代表任意容量已验证。

两机事先锁定 SM 1620 MHz；独立测量的粗粒度时钟采样覆盖 5/36 个计时
记录，共 40 个 GPU 点，均为 SM1620/HBM3201；没有筛除性能样本，
不将未采样区间声称为严格恒频。完整搜索和验收日志在
`build/qwen35_tune_20261013/`，原始配置/候选保留在各 `ep*/result.json`。
选中 native artifact 在 `build/qwen35_perf_20261013/artifacts/eab65a9995ff/`。

## M12：Qwen3.5-35B-A3B shape 对比 Megatron（2026-10-13）

测试代码为已合入 main 的 `bb444c2`，没有改生产代码或做新的 autotune。
使用现有 combined profile，只将 FWD/BWD 的 H/I 改为 2048/512 重新编译；
E256/top8、BF16，参数来自
[官方模型配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)。
这是合成权重/激活和路由的单层 routed MoE benchmark，不是整模型端到端，
不含 attention、shared expert、router projection/top-k。

### 口径

- EP8 为单机八卡，EP16 为两机各八卡；GPU 报告 NVIDIA L20X。
  8K/20K 指每 rank 的名义 8192/20480 tokens，每步长度为 10% 标准差的
  截断正态变化（±3σ），seed=123。
- 均衡路由保留随机完整均衡周期；非均衡为每步共享 expert 偏置 std=0.5
  的合成热点路由，不是模型采集的 route trace。最大 expert 行数/平均值
  的中位数，均衡约 1.0004–1.0015，非均衡 8K/20K 约 7.1/8.6。
- Megatron Core 0.17.0、TE 2.14.1+27de67e1、DeepEP 1.2.1+567632d；
  Flex/DeepEP 20 SM + grouped GEMM，`recompute_modules=["moe_act"]`。
  两者均缓存 X、gate/up，BWD 重算 SwiGLU。
- PF 固定 B2048/D2、min ring depth=2、credit windows=2；
  GEMM 参数沿用当前 profile，dW1/dW2 chunk=4/1，不按 case 换参数。
- 每轮 prime 全部 20 步，再 4 warmups、16 timed samples；第一轮
  Megatron→PF，第二轮 PF→Megatron。每轮取逐步最慢 rank 时间的 p50，
  表中取两轮 p50 的算术均值。32 次正式启动、64 条规模记录均成功。
- FWD 测到 output 和 X/GU 缓存就绪，包含每调用 notify、权重重排、
  缓存分配及完成同步；排除初次固定池/IPC 初始化。
  BWD 每步在计时外完成一次新 FWD 并同步，再独立测到梯度就绪，
  包含 BWD 临时分配、前置流程、通信/计算、native handle 释放及完成同步。
  两项独立 p50 不可相加当成完整周期。

表中单位 ms；吞吐相对变化为 `Megatron_ms / PF_ms - 1`，正值表示 PF
领先，负值表示 PF 落后，不是延迟变化。

### FWD：带训练缓存

| 拓扑 | 路由 | 每 rank tokens | PF（ms） | Megatron（ms） | PF 吞吐相对变化 |
|---|---|---:|---:|---:|---:|
| 单机 EP8 | 均衡 | 8K | 3.582 | 5.949 | +66.1% |
| 单机 EP8 | 均衡 | 20K | 10.813 | 9.845 | -9.0% |
| 单机 EP8 | 非均衡 | 8K | 4.739 | 6.244 | +31.8% |
| 单机 EP8 | 非均衡 | 20K | 14.420 | 11.428 | -20.7% |
| 双机 EP16 | 均衡 | 8K | 6.355 | 7.205 | +13.4% |
| 双机 EP16 | 均衡 | 20K | 12.979 | 14.288 | +10.1% |
| 双机 EP16 | 非均衡 | 8K | 6.536 | 8.038 | +23.0% |
| 双机 EP16 | 非均衡 | 20K | 13.379 | 17.176 | +28.4% |

### BWD：新 FWD 缓存就绪后独立测量

| 拓扑 | 路由 | 每 rank tokens | PF（ms） | Megatron（ms） | PF 吞吐相对变化 |
|---|---|---:|---:|---:|---:|
| 单机 EP8 | 均衡 | 8K | 4.632 | 5.663 | +22.3% |
| 单机 EP8 | 均衡 | 20K | 13.240 | 10.773 | -18.6% |
| 单机 EP8 | 非均衡 | 8K | 6.273 | 5.821 | -7.2% |
| 单机 EP8 | 非均衡 | 20K | 17.614 | 12.767 | -27.5% |
| 双机 EP16 | 均衡 | 8K | 8.588 | 6.677 | -22.2% |
| 双机 EP16 | 均衡 | 20K | 15.551 | 15.175 | -2.4% |
| 双机 EP16 | 非均衡 | 8K | 8.484 | 7.889 | -7.0% |
| 双机 EP16 | 非均衡 | 20K | 19.331 | 19.022 | -1.6% |

当前配置下，FWD 在单机 8K 和双机均领先，但单机 20K 落后；
BWD 只有单机均衡 8K 明显领先，双机 20K 接近，其余落后。
Flash 的性能收益不能直接外推到 Qwen shape。本轮没有做性能归因或调参，
不能据此断言是哪一项优化、或静态分配本身导致差距，也不把这组结果称为
Qwen shape 的最优性能。

### 验证与记录

PF 在 EP8/EP16、均衡/非均衡、129/8K/20K 各两组新路由下，全部通过
output/dX/dGate/dW1/dW2 数值检查；最大 relative L2 为 0.000465。
Megatron 在两种拓扑的 129-token 随机路由各两组检查通过，最大 relative
L2 为 0.003109。64 条正式记录的逐 rank workload hash、token 数、
收到行数、expert 行数统计在后端/轮次/FWD/BWD 间完全一致。
Megatron 所有 rank 的计时阶段新增编译图数量均为 0。

两机显式设置 SM 1620 MHz。粗粒度时钟采样覆盖 7/64 个计时区间，共
56 个 GPU 采样点，均为 SM 1620 / HBM 3201 MHz；不能据此宣称所有
未采样区间严格恒频。没有筛除性能样本。Qwen 单/双机 BWD 主 kernel
均无 stack/local spill；既有 mini-notify 的 28/28 B spill 仍存在。

[汇总 JSON](../benchmarks/results/qwen35_pf_fwd_bwd_20261013.json) 保存双轮
原始 p50、逐 rank workload hash、版本、时钟摘要与数值检查结果。
原始日志、manifest、时钟：`build/qwen35_perf_20261013/results/`；
构建 artifact：`build/qwen35_perf_20261013/artifacts/da93ec093a09/`。
批量复现脚本为 `build/qwen35_perf_20261013/run_matrix.py`，调用现有
`benchmarks/run_pf_network.py` 和 `tools/benchmark_training.py`，
未改变 benchmark 的生成/计时流程。

## M11：仅保留当前 PF/DT（2026-10-13）

本轮只整理代码，不改变已接受的计算、通信协议、调度参数或 buffer
容量，不引入新的优化方案。

- 删除旧 unified/split BWD、旧 GIN transport、旧 bounded-return helper
  和不可达的 fallback；当前 BWD 统一走 PF stream。
- 将仍在使用的 dW 调度、cluster 同步放入
  `src/cuda/weftmoe_pf/backward_weights.cuh`，保留当前 dW chunk 与累加顺序。
- 去除旧 dense dX/return-gather 分配、peer payload 字段和未使用的
  compute stream；当前 dR/dGate IPC 复用、容量增长和释放协议不变。
- 删除基于旧 GEMM 实现的 standalone diagnostics；保留当前 kernel 的
  identity/task timing 工具。旧测量报告保留，不重写历史性能记录。
- 删除 Python 模拟旧协议、固定默认值、源码文本和旧 kernel 分支的
  细碎测试；保留实际资源约束、编译、runtime、梯度与释放检查。
- autotune 的资源检查改为当前 PF stream 入口，过滤不满足 cluster-N2
  的候选；兼容 grid 不再变化已失效的 dispatch batch 参数。

C++ 不再提供脱离 retained FWD tape 的 `prepare_backward_tensors` 自由
函数。使用 `BackwardMegaKernel::prepare_tensors(retained_forward, ...)`；
Python TensorRuntime/Megatron 原本就走这条路径，无需修改调用方式。

验证：单/双机、均衡/非均衡、129/8K/20K 各两组新路由，完整
output/dX/dGate/dW1/dW2 数值检查通过，最大 relative L2 与清理前一致。
EP8/EP16 的 IPC 复用、权重刷新、工作区释放、多层图和 offload 检查通过。
Python 测试从 117/470 tests/subtests 精简为 101/170，全部通过；未增加
测试函数。host 17/17、16 个 public header 独立编译、34 个旧 API 编译拒绝
检查通过。
PF/DT 的 8 个非 BWD 设备函数，清理前后 SASS 指令序列完全相同；当前
BWD 单/双机主 kernel 均无 stack/local spill。既有 mini-notify spill 未改变。

### 清理前后回归

沿用 M10 的 Flash shape、B2048/D2、随机长度/非均衡路由和独立 ready
计时口径，FWD 带训练缓存。每轮 prime 20 步、4 warmups、16 samples。
FWD 旧→新一轮；BWD 旧→新、新→旧两轮，取两轮 p50 均值。下表单位
ms，变化为延迟变化，正值表示略慢，不是吞吐收益。

| 阶段 | 拓扑 / tokens | 清理前 | 清理后 | 变化 |
|---|---|---:|---:|---:|
| FWD | EP8 / 8K | 7.915 | 7.956 | +0.52% |
| FWD | EP8 / 20K | 18.215 | 18.188 | -0.15% |
| FWD | EP16 / 8K | 12.044 | 11.984 | -0.49% |
| FWD | EP16 / 20K | 27.816 | 27.870 | +0.19% |
| BWD | EP8 / 8K | 13.022 | 13.122 | +0.77% |
| BWD | EP8 / 20K | 32.047 | 32.032 | -0.05% |
| BWD | EP16 / 8K | 15.933 | 15.872 | -0.38% |
| BWD | EP16 / 20K | 38.964 | 38.690 | -0.70% |

12 次启动、24 条规模记录的逐 rank token/route hash、收到行数、expert
行数统计一致。平均变化均在 ±0.8% 内，未观察到明显性能回退；不把清理
宣称为性能优化。GIN/notify 容量相同，compute bytes 因删掉旧 peer 描述
字段，每 rank 减少 256–1024 B，而非增大数据缓冲换性能。

两机沿用 SM 1620 MHz 设置。粗粒度时钟采样覆盖 10/24 个计时区间，
88 个 GPU 采样点中 10 个低于 1620，最低 1365；HBM 均为 3201 MHz。
未过滤性能样本，也不声称整段严格恒频。

原始回归日志：`build/pf_cleanup_20261013/results/`；清理前 artifact 为
`build/pf_main_timeline/fwd_cache_vector`，清理后为
`build/pf_main_timeline/cleanup_release_20261013`。

## M10：autotune 修补与 FWD 训练缓存向量化（2026-10-13）

- 修复 dW chunk 的 Python/C++ 约束不一致：1/2/4/8/16 都是合法值，
  dW1/dW2 仍独立配置，生产默认保持 4/1。没有重新引入旧调度分支。
- FWD 的 saved gate/up 从 BF16×2 store 改为 128-bit store，复用已有
  四 lane 寄存器转置。直接保存 WGMMA 的 BF16 舍入结果，不新增 shared
  staging、GM 重读、持久缓冲，也不改变 saved-row 映射。
- FC1→FC2、saved-X 已有向量化/复用 GEMM 加载，单机 FC2 已有延后发布，
  不重复修改。多机逐 tile 的 NIC 可见性 fence 保留。
- benchmark 新增 `--backward-only`，先在计时外完成新 FWD 和跨 rank
  同步，再测 BWD 到梯度就绪；与 `--forward-only` 分开运行。

### 独立消融：FWD saved gate/up 向量化

相同热点 trace，旧→新、新→旧各测一次；取两轮 p50 的算术均值。
以下单位为 ms，下降比例是延迟下降，不是后面 PF/Megatron 的吞吐提升。

| 拓扑 / tokens | 改动前 | saved gate/up 向量化 | 延迟下降 |
|---|---:|---:|---:|
| EP8 / 8K | 8.179 | 7.859 | 3.9% |
| EP8 / 20K | 18.694 | 18.141 | 3.0% |
| EP16 / 8K | 12.388 | 12.272 | 0.9% |
| EP16 / 20K | 28.575 | 27.869 | 2.5% |

双机 8K 的反序一轮略慢 0.14%，两轮平均不到 1%，视为接近测量波动，
不宣称稳定收益。其余三项确认收益，保留向量化；新旧 native 工作区容量
逐 rank 相同。初筛与最终矩阵使用不同的首轮测量，因此这张表的 PF 均值
不应直接替换下面矩阵的 PF 均值。

### 独立 FWD/BWD 对比 Megatron

模型为 H4096/I2048/E256/top6、BF16。EP8 是单机八卡，EP16 是两机各
八卡，GPU 报告 NVIDIA L20X。8K/20K 指每 rank 的名义 8192/20480 tokens，
每步、每 rank 的长度按 10% 标准差正态变化，截断在 ±3σ；seed=123。

- 均衡路由使用随机化的完整均衡周期，尾部允许不完整。计时步骤中
  最大 expert 行数/平均值的中位数约 1.001–1.002。
- 非均衡路由每步重新生成，共享 expert 偏置 std=0.5；8K/20K 的上述
  比值约 7.8/9.6，不是只加入少量 token 抖动的近似均衡路由。
- PF 与 Megatron、FWD 与 BWD、正反两轮的逐 rank token/route hash、
  收到的行数与 expert 行数统计完全一致。
- Megatron 使用 Flex/DeepEP（20 SM）+ TE grouped GEMM，
  `recompute_modules=["moe_act"]`；与 PF 一样保留 X、gate/up，BWD 重算
  SwiGLU。测量已选择的路由和权重对应的 MoE，不包含 router projection/top-k。
- 固定 B2048/D2、min ring depth=2、credit windows=2；全矩阵均用
  dW1/dW2 chunk=4/1，没有逐 case 换参数。autotune 的 2/2 是另行验证的候选，
  本轮没有重跑全空间搜索，也不把修复编译约束说成搜索取得的性能收益。

各 case 完整 prime 20 步，4 warmups、16 timed samples。第一轮
Megatron→PF，第二轮 PF→Megatron；每轮取逐步最慢 rank 时间的 p50，
表中再取两轮 p50 的算术均值。32 次独立启动、64 个规模测量全部成功；
Megatron 所有 rank 的计时阶段新增编译图数量均为 0。

**FWD：带训练缓存，测到 output 和 routed X/gate-up 缓存完成。** 包括
每调用 notify、权重重排、缓存分配、main 与完成同步；计时结束 tape 存活。
排除首次固定池/IPC 初始化、输入和路由生成，未启用一次性预打包权重。

| 拓扑 | 路由 | 每 rank tokens | PF（ms） | Megatron（ms） | PF 吞吐领先 |
|---|---|---:|---:|---:|---:|
| 单机 EP8 | 均衡 | 8K | 6.738 | 8.670 | 28.7% |
| 单机 EP8 | 均衡 | 20K | 14.877 | 18.752 | 26.0% |
| 单机 EP8 | 非均衡 | 8K | 7.883 | 10.350 | 31.3% |
| 单机 EP8 | 非均衡 | 20K | 18.096 | 22.937 | 26.7% |
| 双机 EP16 | 均衡 | 8K | 10.112 | 11.253 | 11.3% |
| 双机 EP16 | 均衡 | 20K | 21.431 | 25.142 | 17.3% |
| 双机 EP16 | 非均衡 | 8K | 12.198 | 14.146 | 16.0% |
| 双机 EP16 | 非均衡 | 20K | 27.886 | 33.397 | 19.8% |

**BWD：每步先在计时外运行新 FWD，完成设备与跨 rank 同步，再独立测量。**
包括 BWD 的临时/梯度分配、前置流程、计算、通信和完成同步。native BWD
返回前本身会释放 forward/dispatch handle，这部分没有从计时中扣除；
计时结束返回的梯度仍存活，外部 gradient 引用删除不计入 ready 时间。

| 拓扑 | 路由 | 每 rank tokens | PF（ms） | Megatron（ms） | PF 吞吐领先 |
|---|---|---:|---:|---:|---:|
| 单机 EP8 | 均衡 | 8K | 10.670 | 11.976 | 12.2% |
| 单机 EP8 | 均衡 | 20K | 25.568 | 27.627 | 8.1% |
| 单机 EP8 | 非均衡 | 8K | 12.992 | 14.666 | 12.9% |
| 单机 EP8 | 非均衡 | 20K | 31.753 | 34.697 | 9.3% |
| 双机 EP16 | 均衡 | 8K | 13.262 | 14.393 | 8.5% |
| 双机 EP16 | 均衡 | 20K | 26.998 | 34.142 | 26.5% |
| 双机 EP16 | 非均衡 | 8K | 16.068 | 19.541 | 21.6% |
| 双机 EP16 | 非均衡 | 20K | 39.216 | 48.538 | 23.8% |

两张表的吞吐领先均按 `Megatron_ms / PF_ms - 1` 计算，不是延迟下降。
独立两项 p50 不可相加当成历史完整周期：同步边界、缓存释放位置、每步
最慢 rank 都可能不同。当前全部分项快于 Megatron，但单机 BWD 只有
8%–13% 左右收益，不能据此宣称完整周期的单机 25% / 双机 20% 目标全部达到。

两机均重新请求锁定 SM 1620 MHz。粗粒度时钟记录与计时区间相交的 104 个
GPU 采样点中，4 个低于 1620：PF 双机均衡 20K BWD 有 3 个点、最低
1515 MHz；Megatron 双机非均衡 20K BWD 有 1 个点、1455 MHz。其他采到的
点均为 1620，HBM 均为 3201。部分短区间未采到时钟点，不能声称全程严格
恒频；未过滤掉任何性能样本。

### 验证与复现

- 默认版通过 EP8/EP16、均衡/非均衡、129/8K/20K 各两次变化路由的
  output/dX/dGate/dW1/dW2 数值检查。最大 relative L2：单机 0.001006、
  双机 0.000928；最大值来自均衡 dX，非均衡分别为 0.000838/0.000734。
- autotune 单/双机候选计划均正常生成；真实生成的 2/2 profile 完整编译，
  并通过两种拓扑的 129/8K 各两次完整梯度检查。第一次通过命令行覆盖宏
  的尝试被 frozen profile 重定义回 4/1，已排除，不能作为 2/2 验证证据。
  有效日志使用 `claim2_generated_verify_*` 标签。
- EP8/EP16 的 IPC payload 复用、训练权重刷新、工作区释放、多层图、
  graph destroy/offload 均通过。已有 Python 117 tests / 470 subtests、
  host 17/17 通过；未新增细碎测试函数。
- FWD main 未新增 spill，未出现 C7507/C7512/C7518；不将既有 mini-notify
  的 28/28 B spill 隐去，也不将本轮称为整库零 spill。

实现提交 `dea1d57`（autotune/独立 BWD 计时）、`07d8b32`（FWD 缓存）。
原始逐 rank 日志、时钟和 artifact manifest：
`build/pf_fwd_bwd_20261013/results/`；最终矩阵标签为
`matrix_{forward,backward}_{balanced,random}_ep{8,16}_{pf,megatron}_r{0,1}`。
消融首轮标签 `screen_fwd_{old,vector}_ep*`，反序用
`matrix_forward_random_ep*_pf_r1` 与紧随其后的 `reverse_fwd_old_ep*`。
[汇总 JSON](../benchmarks/results/pf_fwd_bwd_20261013.json) 保存双轮原始 p50、
逐 rank workload hash、时钟摘要和验证结果。

复现一个 case：

```bash
/opt/.venv/bin/python benchmarks/run_pf_network.py \
  --artifact build/pf_main_timeline/fwd_cache_vector \
  --label reproduce_fwd_random_ep8_pf --nodes 1 \
  --output build/pf_fwd_bwd_20261013/reproduce --master-port 24500 \
  --benchmark tools/benchmark_training.py \
  --env WEFTMOE_PF_MIN_RING_DEPTH=2 --env WEFTMOE_PF_CREDIT_WINDOWS=2 \
  -- --backend pf --hidden 4096 --intermediate 2048 --experts 256 --top-k 6 \
  --ring-depth 2 --ring-tokens 2048 --tokens 8192 20480 \
  --route-distribution random --forward-only --prime-trace --warmups 4 --samples 16
```

换 `--nodes 2` 测双机；分别替换 backend、route-distribution、
`--forward-only`/`--backward-only`，使用不同 label 和空闲端口。
不要与编译、其他 GPU job 或 host 测试并行计时。

上一轮瓶颈拆解见 [M9：通信/GEMM 拆解与数据流优化](pf_backward_bottleneck_20261012.md)。
M7 的主 kernel 零 spill 保持；M9 改善的是热点专家任务排布和中间数据流，
不是把 M8 的无收益 L2 swizzle 重新启用。

## M9：静态任务轮转与 activation 数据流

- dW 从 FLOPs 加权连续区间改为静态 chunk 轮转，dW1/dW2 分别调参。
- 单机 activation wave128、多机 wave64，缩短中间数据存活范围。
- SwiGLU 使用 128-bit 读写，并从寄存器同时写 dW1 缓存和本地 dAB ring。
- 本地 contribution 的 system fence 合并到 activation 阶段边界；网络
  仍逐 tile 发布，不削弱有界 TX 的 NIC-read credit。

不改变先完成全部本地 dX、再执行 dW 的优先级；不新增持久显存。
FWD/DT 的算法与参数保持不变。各步收益、纯通信/真实热点 GEMM、最终
PF/Megatron 正反序对照，以及未达到目标的部分均记录在上述 M9 文档。
保留版 `041391e` 的完整周期相对 Megatron 吞吐领先：单机 8K/20K
17.3%/14.1%，双机 18.5%/20.4%；只有双机 20K 达到本轮目标，不能
将单独 FWD 或纯 GEMM 的收益替代完整周期结果。

## M8：L2 tile-order swizzle 消融（不启用）

**保留结论：只采用 M7 的零 spill 配额，不改变原来的 tile 顺序。**
本轮 N-panel swizzle 没有确认端到端收益；没有把无收益的开关和分支
留在保留版。消融实现保存在 `experiment/pf-l2-swizzle-20261012`：
`116c357` 是全 wave 分组，`f97bba7` 补充单机 expert 内分组。
这不是“所有 L2 swizzle 都无效”的结论，仅覆盖下面的 shape 和排列。

这里改的是 CTA/tile 的访问顺序，不是 shared-memory swizzle 或张量布局。
基线使用 M7 的 192/120；FWD 原本已有 N-group=16，在 Flash 的 FC1/FC2
各 16 个 N tiles 时等价于扫完整个 N。分别改变三个位置，其余不动：

- FWD：FC1/FC2 共用的 N panel 从 16 缩到 4 或 2，仍在同一个 expert 内。
- dU/dX：在 wave 内按 N-panel=4/2 扫 M-block；单机另外试了仅在当前
  expert 与 wave 的交集内分组，避免反复扫过多个 expert 的权重。
- dW1/dW2：在 cluster-pair 网格上使用 N-panel=4/1；重排整个 pair，
  不拆开 multicast 的两个 CTA，奇数维度的 singleton tail 留在原位。

没有改变 K 维累加顺序、dX 优先于 dW 的约束、network canonical 到达
顺序表、ring slot/generation 或完成计数。所有候选的 local/network
BWD main 和 pull/return/receive helper 均为 0/0 B spill，没有
C7507/C7512/C7518，因而下面的回退不能归因于新出现的 main spill。

统一使用 H4096/I2048/E256/top6、每 rank 8K/20K 附近正态波动 10%、
route bias std=0.5、seed=123，B2048/D2，完整 prime trace 后
4 warmups / 16 samples。同拓扑候选的逐 rank workload hash 和 native
工作区容量一致。表格是**完整训练 FWD+BWD+释放+同步**的 rank-max
wall p50（ms），包括每调用权重重排和缓存申请，不包括初次持久池准备；
不是独立 GEMM 或 main-kernel event 时间。

双机 EP16：

| 单项变化 | 8K | 20K |
|---|---:|---:|
| 原 tile 顺序，192/120 | 33.054 | 77.433 |
| dW N-panel=4 | 32.882 | 77.516 |
| dW N-panel=1 | 32.956 | 77.981 |
| dU/dX N-panel=4 | 32.946 | 77.724 |
| dU/dX N-panel=2 | 33.361 | 78.190 |
| FWD N-panel=4 | 34.213 | 80.628 |
| FWD N-panel=2 | 34.249 | 81.332 |

BWD 的小幅差异没有构成稳定收益；FWD N-panel=4/2 分别回退约
3.5%/4.1% 和 3.6%/5.0%（8K/20K），不启用。

单机 EP8（不是两机 EP8）：

| 单项变化 | 8K | 20K |
|---|---:|---:|
| 原 tile 顺序 | 24.922 | 61.494 |
| dW N-panel=4 | 25.134 | 62.181 |
| dU/dX 全 wave N-panel=4 | 31.090 | 79.690 |
| dU/dX expert 内 N-panel=4 | 25.421 | 70.132 |
| FWD N-panel=4 | 26.097 | 70.143 |

全 wave 的 dU/dX 分组打散了原来的 expert-major 访问顺序，产生约
25%/30% 的回退。限制在 expert 内后明显恢复，但仍比基线慢约
2%/14%，所以也不保留。这个对照支持“不能跨 expert 盲目缩窄 panel”，
并没有证明 expert 内分组已经是最优实现。
N-panel 还会改变一个 M-block 完成全部 N tiles 的时序，不能只按
理论缓存复用判断 fused ring pipeline 的收益。本轮没有采集 L2 hit-rate
或 HBM 流量计数器，不把回退单独归因于 cache miss 或某个通信阶段。

正确性：保留配额在 EP8/EP16 均通过 129/8K/20K 每规模两次变化路由的
output/dX/dr/dW1/dW2 检查，最大 relative L2 为 0.000838/0.000734。
N-panel=4 的三个独立消融也均通过 EP8/EP16 的 129-token 两次检查；
expert 内版本额外通过 EP8 的 8K/20K 两次检查。更窄分组仅作性能初筛，
不作为已验证的生产配置。已有 host 测试 17/17、Python 117 tests /
470 subtests 通过；只扩展已有 cluster-pair 覆盖检查，没有新增测试函数。

两机均设置 SM 1620 MHz；这些 swizzle 测量中，多机少量计时内采样
出现 1425–1605 MHz，单机采到的计时点均为 1620 MHz。部分区间没有
采样点，不能称全程严格恒频；HBM 采样均为 3201 MHz，未丢弃任何样本。
亚 1% 的差异不包装成性能收益。

复现源码见上述实验提交，编译入口为
`benchmarks/build_pf_main_timeline.py LABEL --define NAME=VALUE`，独立宏为
`WEFTMOE_PF_FORWARD_N_GROUP`、`WEFTMOE_PF_BWD_ACTIVATION_N_GROUP`、
`WEFTMOE_PF_BWD_DW_N_GROUP`，实验默认值分别为 16/0/0。
编译产物在 `build/pf_main_timeline/bwd_l2_*`，原始 trace/计时/时钟/
manifest 在 `build/pf_spill_l2_20261012/results`；`local_*` 标签为单机。

## M7：消除 GDAKI-only 主 kernel 的标量 spill

结论：多机 math/control 寄存器配额从 **184/136 改为 192/120**。
主 kernel 的 ptxas spill store/load 从 **20/44 B 降至 0/0 B**，
stack frame 从 8 B 降至 0 B；pull/return/receive helper 均为 0/0 B。
没有 C7507/C7512/C7518。单机仍使用 208/88，GEMM 几何、流水级数、
CTA 数、shared memory 和通信协议不变。

原因不是 WGMMA accumulator 整块溢出。SASS 对应的是跨 GEMM 存活的
静态调度游标，以及阶段收尾使用的 CTA 编号；GDAKI-only 裁剪改变了
整段代码的寄存器分配，使原来的 184 配额出现少量标量栈保存。
每个 math thread 多给 8 个寄存器，每个 control thread 少给 16 个，
总预算仍为 `256*192 + 128*120 = 384*168 = 64512`，不改变驻留数。
192 配额的 SASS 中原来的 R1-relative spill 消失；仍可看到通用地址
atomic fallback 的 LDL/STL，不能将这些静态指令全部算成实际 spill。

两个未保留的尝试：让 row geometry 返回 shared-memory 引用，仍为
20/44 B；给调度游标增加 warp-uniform shuffle，反而变成 32/60 B。
最终只保留配额调整，不添加 shuffle 或额外 shared-memory 状态。

沿用 M6 的随机长度/热点路由、完整 prime trace、4 warmups / 16 samples，
双机 EP16 Flash、B2048/D2。首轮完整 FWD+BWD wall p50（ms）：

| math/control | main spill store/load | 8K | 20K |
|---|---:|---:|---:|
| 184/136（本轮基线） | 20/44 B | 32.947 | 77.894 |
| 192/120 | 0/0 B | 33.049 | 77.845 |
| 200/104 | 0/0 B | 33.113 | 77.993 |

三个版本 workload hash 和 native 工作区容量一致。首轮基本持平，不能
把消除少量标量 spill 宣称成端到端加速，也没有理由继续挤压通信配额到
104。BWD host API 边界分别为 20.377/49.070、20.484/49.096、
20.517/48.785 ms；不是独立 CUDA-event kernel 时间。
时钟均请求 1620 MHz；192/120 的 20K 计时内观测到 1440/1590 MHz，
200/104 观测到 1605 MHz，基线采到的计时点均为 1620 MHz。
不剔除任何样本；采到的 HBM 均为 3201 MHz，部分区间无采样点。

192/120 通过双机 129/8K/20K、每个规模两次变化路由的完整数值验证，
output/dX/dr/dW1/dW2 最大 relative L2 为 0.000734。

完成 L2 消融后，再按“保留版 → 原版”反向复测，与最初的
“原版 → 192/120”组成两组对照。以下是两轮 rank-max p50 的均值，
不是把各轮原始样本合在一起重新取 p50（ms）：

| 口径 / 配额 | 8K | 20K |
|---|---:|---:|
| 完整周期，184/136 | 33.016 | 77.780 |
| 完整周期，192/120 | 32.927 | 78.014 |
| BWD host 边界，184/136 | 20.434 | 48.975 |
| BWD host 边界，192/120 | 20.453 | 49.070 |

完整周期变化 -0.27%/+0.30%，BWD host 边界变化 +0.09%/+0.19%，
结论仍是基本持平，而不是零 spill 带来了可观加速。
保留版最终单机 EP8 完整周期为 **24.833 / 61.585 ms**，
BWD host 边界为 16.810 / 43.115 ms；单机算术路径和寄存器配额未改。
两组双机对照中，旧/新分别观测到 4/2 个低于 1620 MHz 的计时点，
范围分别为 1455–1590 / 1440–1590 MHz。最终单机观测到
1575/1560 MHz 两个点；均保留。其他采到的计时点为 1620 MHz，
HBM 为 3201 MHz，仍有未采到时钟点的短区间。

编译/SASS 位于 `build/pf_main_timeline/bwd_spill_*` 和
`build/pf_spill_l2_20261012/audit_r192`，本轮逐 rank 日志、时钟与 manifest
位于 `build/pf_spill_l2_20261012/results`。

## M6：原生库仅保留 GDAKI 后端

按部署硬件约束，在项目统一的 `communication/nccl_device.cuh` 入口关闭
device 编译中的 Proxy/GPI/EFA GDA，覆盖 FWD、BWD、notify 和 NCCL 内部
barrier 调用；所有 GIN device communicator 的初始化显式请求
`NCCL_GIN_TYPE_GDAKI`，不再自动回退到 Proxy。单机 notify 仍不申请 GIN。
不修改系统 NCCL；独立通信参考 benchmark 不属于原生库的裁剪范围。

宏只作用于 NVCC device pass。host pass 保留 SDK 默认定义，因为该阶段
NCCL 默认关闭 GDAKI，若再关闭其余后端，非模板接口的 `auto` 返回类型
无法推导。host 头文件解析不等于生成或使用 GPU Proxy 分支。
保留原有 assertion 策略、GEMM 参数、缓冲容量、静态任务和同步协议。

`bwd_gdaki_only_final` 中 return helper 从 8/8 B spill 降为 0/0 B；
静态 SASS 指令数从 4243 降至 2383。裁剪也改变编译器寄存器分配：
多机 BWD 主 kernel 从 0/0 B 变为 20/44 B spill store/load，位置是
调度索引及阶段收尾的标量状态，不是 GEMM accumulator。单机主 kernel
仍为 0/0 B；没有 C7507/C7512/C7518。不能将这次修改称为整个算子零 spill，
也不能从删掉未执行分支直接推断性能收益。

双机 EP16 Flash、B2048/D2，沿用 M5 的变化长度/热点路由和
4 warmups / 16 samples、完整 prime trace。旧/新/新/旧各跑两轮，
逐 rank workload hash 一致；两轮 p50 均值如下（ms）：

| 口径 / 版本 | 8K | 20K |
|---|---:|---:|
| 完整 FWD+BWD，旧版 | 33.082 | 77.605 |
| 完整 FWD+BWD，GDAKI-only | 33.170 | 77.811 |
| BWD host API 边界，旧版 | 20.576 | 49.097 |
| BWD host API 边界，GDAKI-only | 20.585 | 49.123 |

完整周期差异约 +0.27%，BWD 约 +0.04%/+0.05%；视为基本持平，不宣称
加速，也没有由这些数据确认明显性能回退。BWD host 边界仍不是独立 GPU
kernel event 时间。两机均设置 SM 1620 MHz，计时内旧/新各观测到 3 个
低于目标的 GPU 频率点（合计范围 1545–1590 MHz），保留所有样本。
计时内采到的 HBM 均为 3201 MHz；部分短区间没有采样点。
一次新版本启动因端口占用失败，未进入 GPU 计时，保留日志并换端口重跑。

单机 EP8 / 双机 EP16 均通过 129/8K/20K 各两轮变化路由的完整数值验证，
最大 relative L2 分别为 0.000838 / 0.000734；双机还通过 8K/20K
无缓存推理、kernel-entry 检查及两次调用的 CUDA Graph 回放验证。
前后 native 工作区容量一致；现有 Python 回归 117 tests / 470 subtests
通过，没有增加单元测试。

编译、SASS 归因及回归原始结果位于 `build/pf_backward_gdaki_20261012/`
和 `build/pf_main_timeline/bwd_gdaki_only_final/`。

## M5：当前 GEMM 参数的 autotune

原生源码保持 `e4db718`，固定当前 FWD、通信协议、M128 canonical 行、
184/136 寄存器配额和 wave128。新增
`configs/pf_backward_stream_autotune.json`，复用训练 autotuner，搜索
dU/dX 各自的 N、共用的 activation stages，以及 dW1/dW2 共用的
M/N/stages。不混入会切回旧 kernel 的 activation K128，也不搜索
已经不作用于新 stream 的旧 return knobs。

测试为双机 EP16、Flash H4096/I2048/E256/top6、8K/20K，B2048/D2，
正态长度波动 10%、路由偏置 std=0.5、seed=123；请求时钟 1620 MHz。
每个候选完整 prime trace，再 4 warmups / 16 samples；比 M3/M4 的
8 samples 更多，因此只能与本轮同 trace 基线比较。完整周期包含
每调用重排、缓存分配/释放和 device synchronize，不计首次固定池准备。

初筛完整 FWD+BWD wall p50，单位 ms：

| 候选（仅列相对基线的变化） | 8K | 20K |
|---|---:|---:|
| 基线：dU/dX/dW M128/N256/K64，三级流水 | 33.101 | 77.767 |
| dW M256/N128 | 33.504 | 78.258 |
| dW M128/N128 | 37.465 | 88.181 |
| dW 两级流水 | 34.735 | 80.887 |
| dU N128 | 33.760 | 79.416 |
| dX N128 | 34.490 | 81.231 |
| dU/dX 都用 N128 | 34.974 | 82.926 |
| dU/dX 两级流水 | 34.405 | 79.250 |
| dU/dX 四级流水 | 33.587 | 78.819 |

另一个候选 dW M128/N256 四级流水因 shared memory 超限，在编译前过滤。
没有 C7507/C7512/C7518；dW M256/N128 和 dW 两级流水分别产生主 kernel
40/68 B、40/228 B 的静态 spill store/load，其他候选主 kernel 均为零。
不能据此给 spill 单独归因，因为它们同时改变了 tile 或流水深度。

小 tile 没有提高当前 384-thread/168 初始寄存器配额下的驻留 CTA 数。
仅计单个 tile 的输入、不计输出或跨 tile 复用时，BF16 GEMM 的算术强度为
`M*N/(M+N)` FLOPs/byte；M128/N256 为 85.3，M128/N128 为 64。
这给出了小 N 降低数据复用的一个解释，但不是实际 HBM 流量测量或
完整的瓶颈证明。M256/N128 理论强度相同，仍需用实测决定。

正反顺序各复测一次，两个 rank-max p50 的均值：

| 入围配置 | 8K / ms | 20K / ms |
|---|---:|---:|
| 原参数 | 33.096 | 77.722 |
| dW M256/N128 | 33.244 | 77.635 |

原参数的等权几何平均分仍最好，选择结果为 `incumbent_fastest`；
不是把不足 1% 的差异包装成收益。最终保留所有原生 GEMM 默认值，
没有修改 FWD/DT 的参数或算法。两个入围配置均通过 129-token 两轮
数值验证，保留版还通过 8K/20K 各两轮验证（output/dX/dr/dW1/dW2）。
这是本轮有限搜索空间的负结果，不表示四个 GEMM 已经全局最优；
dW1/dW2 仍共享几何/stages，本轮没有独立调参，也未改变单机参数。
单机性能没有在本轮独立搜索，不能将这组双机结论外推为 EP8 最优。

保留版最后一次完整周期为 **32.934 / 77.736 ms**，BWD host API 边界
为 **20.682 / 49.019 ms**，后者不是独立的 GPU kernel event 时间。
最终 8K/20K 验证最大 relative L2 为 0.000110；本轮现有 Python 回归
117 tests / 470 subtests 通过，没有增加单元测试。

时钟记录覆盖两节点，但不是连续逐 kernel 采样。确认轮的 20K 计时内，
原参数观测到 5 个 1455–1545 MHz 点，挑战者观测到 1 个 1560 MHz 点；
最后一次 20K sample 1 观测到 3 个 1530–1590 MHz 点。没有删除这些样本，
不能称全程严格恒频，也不能认定两候选约 0.17% 的综合差异有统计意义。
计时内采到的 HBM 频率均为 3201 MHz；部分短测量区间没有时钟采样点。
因此结论是“没有确认收益，保留原参数”，不是“已证明原参数全局最优”。

原始计划、编译、逐 rank 计时/验证日志和自动选择结果在
`build/pf_backward_tune_20261012/network/`；各候选共享同一 trace hash。
`table.json` 已导出保留参数的两个容量表项，`clocks.json` 保留全部时钟点。
紧凑结果见 `benchmarks/pf_backward_20261012.json` 的 `gemm_autotune_m5`。
通用命令和当前参数边界见 [autotune 说明](autotune.md)。

## Return helper 的剩余 spill（后续核查）

`return_canonical` 是 BWD main 内的 device helper，不是额外的 kernel。
它由 48 个静态 return owner 的借用通信 warp 的 lane0 执行：
等一个 M-block 的全部 dX N tiles 完成，把相邻 source/contribution 行
合并为 fragment；远端行通过两次 GIN Put 分别发送 BF16 dX 和 FP32
router gradient，并增加两个远端完成计数。每个 fragment 的本地 NIC
flush/wait 完成后，整个 block 才能发布 TX returned generation。
本机行已由 dX epilogue 直接写回，不走这里的跨机 Put。
这些 owner 与 dW math 并行；慢 return 仍可能通过 TX credit 反压计算，
所以不能笼统地说“通信路径 spill 不影响性能”。

对 `bwd_ready_final` 的进一步 SASS 检查，报告的 8/8 B 对应
`STL.64 [R1+0x90]` / `LDL.LU.64 [R1+0x90]`，位于
NCCL `gin/proxy/gin_proxy.h::postGfd` 的通用地址 atomic 分支。
`gin_backend_current` 双机运行日志再次确认使用 `GIN_IB_GDAKI`，
该次执行不进入 proxy backend，因而不执行这对 spill 指令。
这比“只有 8 B，影响应当很小”更准确；不能把静态报告当成当前热路径的
动态 local-memory 流量。换 backend 或编译超参后仍需重新检查。
原始日志在 `build/pf_backward_tune_20261012/backend/`，SASS 在
`build/pf_backward_20261012/ready_final_spills/`。

## M4：按预计到达顺序排静态任务

M3 消除了逐窗口 padding，但 expert-major 的第一个 canonical M-block
可能跨多个 RX 窗口，导致 math 等待尚未到达的行。M4 在 main 前按
`(该块最后需要的物理窗口, canonical block ID)` 排列任务；仍是静态分配，
没有运行时领任务或抢占队列。main 及 return 使用相同排列的 position
决定 TX slot/generation，cache 地址和 dW 仍使用原 canonical block ID。
因此不改变 saved-row 顺序、GEMM 的 K 顺序和 dW 归约顺序。

排序用 GPU merge pass，复用不再使用的旧 phase/DU-counter 数组，
不增加持久空间或临时分配；多次小 kernel 的开销包含在完整周期中。
这是静态的到达顺序估计，并不保证每块实际最早 ready。

双机 Flash、相同随机 trace、1620 MHz 请求、完整 FWD+BWD wall p50：

| 候选 | 8K / ms | 20K / ms |
|---|---:|---:|
| 本轮修改前 `b151d93` | 34.563 | 86.551 |
| M3 `b5d1cc4` | 33.238 | 81.522 |
| M4 首测 `ready_order_perf` | 32.836 | 80.201 |
| M4 复测 `ready_order_repeat` | 33.224 | 80.450 |
| M4 最终构建 `ready_final_perf` | 32.855 | 80.808 |

M4 两次 BWD host API 边界 p50 分别为 20.394/50.456、20.439/50.733 ms；
M3 为 21.113/51.841 ms。BWD 有一致改善，但完整周期的 8K 增益较小，
不能只引用最快一轮。首测有一次 1485 MHz 观测落在 20K sample 2；
复测四次 1500–1590 MHz 观测均不在计时区间。所有样本保留，
两轮 workload hash 与 M3 完全一致。

最终构建 BWD host p50 为 20.503/51.030 ms；三次 1560–1605 MHz
观测均在计时区间之外。相对本轮前，完整周期下降 4.9%/6.6%，
BWD host 边界下降 8.6%/10.0%。相对旧 PF 的本轮复测 33.502/80.921 ms，
最终完整周期仅快 1.9%/0.1%；20K 仍应称基本持平，不能宣称大幅领先。

单机最终构建回归也保留全部结果，而不是只引用 M3 较快的一轮：

| EP8 完整周期 | 8K / ms | 20K / ms |
|---|---:|---:|
| 本轮前 `b151d93` | 25.236 | 64.490 |
| M3 较早 `canonical_local_perf` | 25.213 | 62.569 |
| M4 `ready_final_local` | 25.443 | 63.426 |
| M4 `ready_final_local_repeat` | 25.191 | 67.370 |
| M3 同场对照 `ready_local_control` | 24.963 | 64.660 |

最后两轮 M4 的 BWD host p50 为 17.127/44.014、17.138/48.078 ms；
M3 同场为 16.968/45.113 ms。三轮 trace hash 一致，M4 首轮一次
1590 MHz 观测在计时外，第二轮 SM 频率观测全为 1620 MHz；
M3 对照一次 1560 MHz 观测落在 20K sample 0。计时内未观测到 HBM
频率下降，测试间隙也未发现其他 GPU 进程，但采样不能排除瞬时干扰。
单机 BWD 的 SASS 指令流在 M3/M4 **逐条相同**（含被调用函数 9368 条，
归一化指令 SHA256 `c4af9d4119c1e67638ffd838777f20eef754e446ff8e1cf328e9ecc693def397`）。
M4 排序仅用于多机；这些结果不足以把单机的差异归因于排序，也不足以
确认本轮单机 20K 相对 `b151d93` 的稳定收益。波动来源尚未完全定位。

`wave128_phases` 与 `ready_order_phases` 的独立诊断（warmup=1、sample=2）
显示，最终 dW 完成最晚 rank 的首次 dQ 开始时间，相对 main 入口：

| 规模 / sample | 原静态顺序 / ms | 到达顺序 / ms |
|---|---:|---:|
| 8K / 0 | 0.798 | 0.022 |
| 8K / 1 | 1.101 | 0.038 |
| 20K / 0 | 0.827 | 0.040 |
| 20K / 1 | 1.278 | 0.159 |

这些是 GPU 阶段时间，不含 main 前的排序；诊断 trace 不等同于性能表
的八个样本，不能把各 rank 的阶段最大值相加。return/reduce 仍与 dW
折叠，没有放宽所有本地 dX 必须先于任何 dW 的约束。

M4 编译器报告：单机/多机 BWD 主 kernel 均为 0/0 B spill store/load，
pull/Get/receive 也为零；return helper 为 8/8 B。这不代表整个程序
没有 local memory 访问，也不代表整个 FWD/BWD 都零 spill。
没有 C7507/C7512/C7518，math 分支没有 accumulator store-to-local。

保留构建为 `bwd_ready_final`（相对 `bwd_canonical_ready_order` 仅格式整理）。
数值检查覆盖双机 129/8K/20K、每规模两轮变化路由、512-token 窗口；
另测 8K/20K 的 bias=8 热点，普通/热点两组最大 relative L2
分别为 0.000734 / 0.000275。
最终构建 `ready_final_lifetime` 通过三份 live tape、变化长度/热点、
权重更新、workspace 释放重建、两层 graph/offload。
现有 Python 回归 117 tests / 470 subtests 通过，未增加细粒度单元测试。

## M3：消除窗口 padding 和修复寄存器分工

本轮改动的简要说明（详细消融和最终测量见本节末尾）：

- 多机 dQ/dX 按完整专家行切 M128，不再按每个通信窗口分别补齐。
  同一随机 trace 的额外计算行数 / 有效行数，8K 从 23.19% 降到 2.09%，
  20K 从 22.17% 降到 0.81%。剩余仅为各专家最后一个 M tile 的对齐。
- Dispatch 直接填充 dW2 本来就需要的 compact dY cache；dX 直接读完整
  dAB cache。RX 的 fanout credit 与 GEMM 解耦，不增加一份完整激活缓冲。
- dX 按 canonical M-block 写已有注册 TX，目的为源 rank 的完整 contribution。
  不再绕到同 rail GPU 的旧物理窗口 TX；连续 source/contribution 行合并 Put。
  本地 NIC completion 保护 TX 复用，源 contribution 在调用结束前不复用。
- 内联跨 math/producer/comm 的阶段收尾 helper，让 `setmaxnreg` 真正生效。
  单机静态 spill store/load 从 72/92 B 降至 0/0 B；完整周期
  25.236/64.490 ms → 25.071/61.155 ms（8K/20K）。
- 多机重新分配 math/control 寄存器，并将 compute wave 从整个 ring
  缩小到 128 个 M-block；提前开始 dX/return，不改变 RX/TX 容量。
- BWD 直接使用当前 tape 保存的 route ordinals 和 packet/fanout 表，省去
  重复生成；仍重建共享工作区的物理计划和当前通信 epoch。

### 布局与同步

FWD 的行缓存仍为 expert/source/route 顺序。BWD 的物理 RX 目录也改用
稳定 source 顺序，但只用于读取/退休通信 packet，不再决定 GEMM M-padding。
每次 BWD 从当前 tape 重建 canonical M-block 目录和 source/contribution
反向映射，分别复用旧 scheduler descriptor 和闲置 route-header 区域。
不能复用前次 FWD 的物理目录；后续层可能已覆盖它。

本地 pull 把 dY 写入 compact cache，release 增加对应 canonical M-block
的 arrived rows；producer acquire 达到 valid_rows 才发出该块 TMA。
dQ 的 N tiles 完成后，两份 SwiGLU 任务生成 Q、dAB、dr，再发布 dX ready。
return owner 等所有 dX N tiles 完成后发送；对本块全部 fragment 完成
本地 NIC flush/wait 才发布 TX returned generation。下代 SwiGLU/dr 和
dX 都等待这一 credit，因此不会在 NIC 仍读取时覆盖 TX。

return 不需要远端 RX-slot credit：它写的是源 rank 已有的整调用 contribution，
不是短 RX ring。每个源 rank 按远端节点累计期望 route 行数；各 expert rank
用两个独立 GIN weak signal 累加 dX/dr 完成行数，源端 acquire 两个计数后
才能 reduce。最终仍有全 rank 终止 barrier，防止下一次调用覆盖 contribution。
RX Get/fanout ring、TX ring、compute slot 仍是有界容量；没有 full dX 缓冲。
两个 signal 和一个累计行数字段填入原结构的 padding，结构大小不变。

### spill 的原因与不能混淆的指标

原来的跨角色 `__noinline__` 阶段 helper 令 ptxas 报 C7507：
`setmaxnreg` 被忽略，math 和 control 实际都受初始 168 registers 限制。
仅修复内联又会暴露另一个问题：单机合适的 208/88 配额不够多机 GIN
控制流使用，控制路径出现大量 spill。因此不能直接沿用单机配额。

384-thread CTA 初始池是 `384 * 168 = 64512`，不是 65536。
192/128 超过实际池，试验中在 setmaxnreg 等待而无法前进，已撤回；
192/120 和 184/136 才满足池约束。184/136 下同步 return 版本的
pull/Get/return/receive 编译器 spill 均为零，主 kernel 仍有少量 spill。
SASS 中大量 LDL/STL 来自 GIN/SDK 的调用栈和 local-address atomic 分支，
不能把它们全部算作 WGMMA accumulator spill；主 kernel 残留项包含线程编号
和控制状态的 reload，仍要检查是否落在 K-loop，并以实际耗时决定取舍。

### 逐步消融

以下均为 EP16、Flash H4096/I2048/E256/top6、2048-token window、相同
变化长度/变化路由 trace，4 warmups / 8 samples；单位 ms，完整 FWD+BWD。

| 候选 | 8K | 20K | 说明 |
|---|---:|---:|---|
| 本轮修改前 `b151d93` | 34.563 | 86.551 | 旧物理窗口切 GEMM |
| 只修复内联 | 35.659 | 86.799 | 208/88 引入 GIN spill，不单独保留 |
| canonical M-block + 192/120 | 33.575 | 83.742 | 消除重复 padding |
| 184/136，整 ring compute wave | 35.641 | 82.933 | spill 更少不等于更快 |
| 184/136，wave=128 | 33.385 | 81.893 | 提前启动 dX/return |
| 再去掉重复 tape 表生成 | 33.016 | 81.375 | 多份 live tape 回归通过 |
| return 每 owner 4 个在途 fragment | 33.973 | 81.782 | 无收益，撤回；增加了 spill |
| dQ/dX 保留一个在途 WGMMA group | 33.540 | 81.856 | 数值通过，无收益，撤回 |

没有筛除慢样本。前三个 canonical 候选和 tape 候选各存在少量低于请求
1620 MHz 的观测，其中部分落在 20K 计时区间，不能解读百分之一的差异。
return 流水候选频率采样全部 1620 MHz；其结果仍不支持保留此改动。
将整个通信收尾拆成 noinline 函数还触发 ptxas C7512/C7600
（184 寄存器下 WGMMA 分配失败），也未保留。
在途 WGMMA group 方案将 empty-stage release 延后一轮，最终 drain
剩下的一组；没有 C7512/C7518，EP16 的 129/8K/20K 数值检查通过，
但单机也测得 25.651/63.857 ms，不优于原 K-loop。因此保留逐轮 wait<0>。

构建日志、SASS 和逐 rank 原始测量分别在
`build/pf_main_timeline/bwd_canonical_*`、
`build/pf_backward_20261012/*_spills/` 和
`build/pf_backward_20261012/results/`。

### M3 阶段复测（M4 之前）

最终 artifact 为 `bwd_canonical_final`：184/136、wave=128、同步 fragment
return、原 wait<0> K-loop。相同 trace 的完整 FWD+BWD wall p50：

| 拓扑 / tokens | 旧 PF `main@1cfb423` | 本轮前 `b151d93` | 本轮后 | 对本轮前延迟下降 |
|---|---:|---:|---:|---:|
| 单机 EP8 / 8K | 26.483 | 25.236 | 25.213 | 0.1% |
| 单机 EP8 / 20K | 68.624 | 64.490 | 62.569 | 3.0% |
| 双机 EP16 / 8K | 33.502 | 34.563 | 33.238 | 3.8% |
| 双机 EP16 / 20K | 80.921 | 86.551 | 81.522 | 5.8% |

旧 PF 单机行沿用本轮任务早期基线；双机旧 PF 为此次重新运行的
`m3_old_repeat`。旧 PF 不是 V3。所有 workload hash 与对应旧 trace 一致。
最终单机频率观测全为 1620 MHz；最终双机仅在计时外观测到一次 1590 MHz。
旧 PF 双机复测有一次 1500 MHz 落在 20K sample 6，所有样本原样保留。
双机相较旧 PF 8K 快 0.8%、20K 慢 0.7%，只称基本持平，不宣称全面领先。

按 host API 边界拆分的 BWD p50（不是纯 kernel 时间）：

| 拓扑 | 本轮前 8K / 20K | 本轮后 8K / 20K |
|---|---:|---:|
| EP8 | 17.261 / 44.717 | 17.103 / 42.971 |
| EP16 | 22.429 / 56.678 | 21.113 / 51.841 |

padding 降幅不能直接当成完整周期收益：此次改变的是 dQ/dX 的 M 行数，
dW1/dW2 原本已经按完整专家行计算，且 FWD 未变。以有效 GEMM FLOPs 计，
dQ/dX 只占 BWD 的一半、FWD+BWD 的三分之一，此外还有通信和准备开销。

保留版编译：单机 BWD 0/0 B spill；多机主 kernel 12/224 B spill
store/load，pull/Get/return/receive 均为 0/0 B。没有 C7507、C7512、C7518。
这些字节是编译器静态统计，不是每调用或每 M-tile 的实际流量。
反汇编 math 分支无 accumulator store-to-local；少量标量 reload 尚未消除。

验证覆盖：EP8 的 1/129/8K/20K、EP16 的 129/8K/20K，每规模两轮变化
路由；EP16 8K/20K 额外 bias=8、512-token 窗口回绕，包含空专家。
output/dX/dr/dW1/dW2 全部通过原容差；热点最大 relative L2 为 0.000275。
保留 tape 优化通过三份同时存活 tape、权重更新、workspace 释放重建、
两层 inference graph/offload 检查。PF 无缓存推理的 EP16 8K/20K 数值和
kernel-entry 回归通过。现有 Python 回归 117 tests / 470 subtests 通过，
没有新增单元测试或更改误差阈值。DT 代码未改，此轮未重跑其性能。

后面的 M1/M2 章节保留当时的实现和测量过程；当前多机协议以本节及
前述 M4 静态任务排列为准。

## 设计与里程碑

- [x] M1 单机：复用 PF destination-pull、输入 ring 和完整 source
  contribution；producer/math 独立重放静态 dQ/SwiGLU/dX 任务流。
  dX epilogue 从 accumulator 直接 128-bit 写回。所有本地 dX 完成后
  math 开始静态 dW，通信 warp 进行 source reduce。
- [x] M2 多机协议：复用当前借用 warp 的按节点去重、逐 token Get、机内 pull、
  直接 TX contribution/Put return。路由和计数来自 FWD tape，不再重新
  counts notify。RX fanout、TX NIC completion 和远端 credit 分开退休。
- [x] M3 消除通信窗口引入的 dQ/dX padding，修复寄存器分工，缩小 compute
  wave，消除重复 tape 表生成；覆盖随机 tokens/routes、热点/空专家和
  slot 回绕。FWD 和 DT 不改变数学流程。dW 参数的进一步搜索不在本次结论内。

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
