# V4 ring 训练对接记录

## 2026-10-08：PF 单机 inference 对齐 MegaMoE（进行中）

已切回 `refactor/static-forward-v4`；V3 恢复分支及容量消融分支保留。
本轮目标为 EP8、H4096/I2048/E256/top-k6、8K/20K，无训练激活缓存。
新入口先显式 opt-in，不以尚未追平的候选替换已验收训练或跨机 profile。

本轮步骤及当前结论：

- M1：完整 source contribution、expert-global M128、三个 Pull warp、
  producer/math 独立静态任务流、独立 epilogue scratch、统一连续 TMA stages。
  大 expert 按 activation ring 深度切 wave，保留有界 X/S 复用。功能通过，
  但初版 20K 比旧 PF 慢，不能把架构迁移本身登记为性能收益。
- M2：Hoist WGMMA descriptor，移除 private scratch 不需要的 FC1 前置 join，
  完整 TMA tile 不再执行全 math 的冗余 threadfence/join；math/control
  register 配额改为 208/80，并使用匹配的 host quota audit。
  相邻 screening 的 kernel 耗时下降约 3%。
- M3：最终 Reduce 改为 warp/token、每 lane 16B 向量传输，top-k order 放在
  warp registers；FP32 求和次序不变。20K rank0 Reduce 中位数从
  2.253 ms 降至 1.012 ms，rank-max kernel 从 25.864 ms 降至 24.001 ms。
- M4：contribution 改为 MegaMoE 同样的 `[top_k,Tmax,H]`；单独消融未获得
  稳定收益，不能将“热点 expert 地址跨度更小”当作已证实的主要性能原因。
  同时 eager 可显式按 layer 复用不可变权重，沿用 graph 的 invalidate/close
  机制，不缓存 X、route、gate/up 或其它计算结果。训练仍每次发布新权重。
- M5：WGMMA descriptor 低位先 warp-uniform，再按 stage/atom 做简单加法，
  stage 游标使用条件回绕；沿用现有 `wgmma_wait_group` 轴比较 0/1。
  0 在途额外组的 8K/20K kernel 为 8.214/24.059 ms，1 为
  8.314/24.144 ms，未发现多一组在途的收益，保留 0。wait=1 也通过
  普通及强热点小 ring 的输出对照。
- M6：FC2 的 `fc2_done` 仅作为 S 读退休计数，改用 relaxed atomic；
  contribution 仍由最终 system release/acquire barrier 发布。SASS 确认
  移除了该计数前的 GPU fence，8K/20K kernel 为 8.053/23.804 ms，
  相对 M5 小幅改善 1.96%/1.06%，不是剩余差距的主要解释。
- M7：只对新 inference 入口的完整 N tile，展开 FC2 向量写回并移除
  不需要的列边界判断，N128 时采用 16 lanes/row。8K/20K kernel 为
  8.055/23.698 ms，未显示显著额外收益。M6、M7 均通过普通及
  B128/强热点 ring-wrap 输出对照。
- M8：将新入口的 N 接回已有 GEMM tuning 参数，比较 N128/5 stages 与
  N256/3 stages（均 K64）。N256 增加 FC1 两个 N64 TMA issuer 的 completion
  join，未改变 BF16 舍入与 FP32 算术。8K/20K kernel 降至
  7.254/18.355 ms，相对 M7 改善 **9.95%/22.55%**；完整调用
  7.413/18.529 ms。普通及强热点小 ring 校验通过。N256 的旧 fallback
  stages 配为 2/2 以满足原入口资源限制，新入口实际为 3/3。
- M9：增加显式 `local_prefill_fast_math` inference 消融，使用与 MegaMoE
  fast-math 相同的 SFU exponential/reciprocal；仍保留 gate/up 和 FC2 input
  的 BF16 舍入，不调整训练/其它入口的算术。N256 kernel 为
  6.982/17.636 ms，相对严格算术再改善 3.74%/3.92%；N128 则为
  7.723/23.225 ms，说明 fast-math 不能代替 tile 粒度优化。两种 N 都
  通过普通及强热点小 ring 原容差校验。默认关闭，只由显式 inference
  profile 选择；其语义不是与严格除法逐位一致。
- M10：最终 Reduce 每 lane 同时处理四个独立 16B 向量，提高 GM load 的
  在途并行度；每个元素仍按相同专家顺序做 FP32 逐项相加。N256/wait0
  的 8K/20K kernel 为 6.748/17.095 ms；wait1 为 6.784/17.103 ms，
  没有显示额外在途 WGMMA group 的收益。两者都通过原容差普通及热点
  小 ring 校验，继续选 wait0。
- M11：producer/math scheduler 将本次调用内不变的当前 expert 行数和
  block 边界留在 warp-uniform registers，避免逐 tile GM 重读；没有
  跨调用保留 route/count。FC2 的固定展开也单独隔离，原训练/跨机
  epilogue 保留原来的动态行循环策略。8K/20K kernel 为
  6.624/16.943 ms，完整调用为 6.796/17.179 ms；普通及热点小 ring
  输出对照通过。随后 Flash T256/8192/20480 的训练
  output/dX/dGate/dW1/dW2，以及独立 inference 生命周期校验均通过。
- M12（未采用）：Pull G2S 改为 MegaMoE 的 `EVICT_FIRST`，普通及
  热点小 ring 校验通过，但 kernel 为 6.668/17.007 ms，未优于 M11。
  已去掉该改动，不将缓存优先级对齐当作性能收益。
- M13（未采用）：N256 的 N-group 从 16 降为 8，使每组覆盖的权重
  列宽与 MegaMoE N128/group16 一致。kernel 回退至 7.331/20.587 ms，
  已撤回；不能据“相同列宽”推断相同复用或更好的流水，保留 M11。

以下是定位用单轮 screening，不是最终验收表。32 步相同正态变长/热点随机
trace（token std=10%、route bias std=0.5），完整 prime，8 warmup + 24 measured，
1620 MHz；表为每步 rank-max 的 p50，单位 ms。不是模型采集 trace。
M1–M3 kernel 数字不包含每次 W1 repack；M4 为预准备权重后的 kernel 数字。

| 版本 | 8K kernel | 20K kernel | 备注 |
|---|---:|---:|---|
| 旧 PF（冻结已验收库） | 9.201 | 23.197 | 原 source-window 工作流 |
| M1 | 9.230 | 26.762 | 功能迁移，20K 回退 |
| M2 | 8.939 | 25.864 | 同步/descriptor/register 精简 |
| M3 | 8.443 | 24.001 | 向量化 Reduce |
| M3，预准备权重复测 | 8.448 | 24.286 | 和 M4 相同 host 权重策略 |
| M4 | 8.420 | 24.514 | K-major contribution，无稳定收益 |
| MegaMoE | 6.894 | 16.557 | CUDA event 含外部输入 staging 及 enqueue gaps |

PF 的 `profile_inference.total_us` 为 native fused-kernel clock，含 kernel 内
输入 staging/route plan，但不含 host prepare；MegaMoE 这里为 CUDA event 区间，
含输入 staging 和多个 enqueue 间隙。最终报告还需给出两者都预准备权重的
完整调用耗时，不能将此表直接称为完全相同计时边界。

M1–M4 都通过 T256/8192/20480 每档两步 Torch 对照，以及 B128、route bias
std=8 的 8K/20K ring-wrap 对照。M4 另通过 Flash 三档训练
output/dX/dGate/dW1/dW2 对照。单独的 Flash inference 生命周期校验覆盖
变长/变路由、prepared 与 fresh 逐位对照、async、原址改权重前 invalidate、
graph 与 eager 交替、输入 weakref 释放及 close 后显存释放；8 ranks 全通过。
临时 CTA0 时钟探针显示 20K 热点 rank 的 FC2
epilogue 增长显著，producer ready 等待不是最大的部分；探针有扰动，仅用于
定位，未用于上表，也不保留在正式 kernel 中。数值门禁未放宽。

原始 runner、日志、编译产物保存在 `build/pf_megamoe_20261008/`；
`cache{,_m2,_m3,_m4}/a7a38001e7a7/` 分别对应各步骤。宏/profile 相同而
源码不同，因此目录区分源码版本，不能仅用 tuning digest 识别实验。
旧库使用 `return_frozen/batch4_window2`，MegaMoE 使用前节冻结 Torch 2.9.1 包。

原 `torchrun_tensor_runtime.py` 的 2048/512 小形状训练 fixture 直接扩到
4096/2048 时，dX 对照出现少量绝对误差 2.29e-5/3.05e-5 超过原 2e-5 阈值；
没有放宽阈值或将该 run 算作通过。Flash 的规范训练 reference 对照与上述
独立 inference 生命周期分别验证。冻结旧 PF 原样复现了 top-k4 case 的相同
rank、元素位置和 2.29e-5 最大绝对误差（`old_lifecycle_fixture_ep8.log`），
因此不是本轮 CUDA 改动引入的回退；没有改动这个 fixture 的容差。

CPU 的 `test_static_forward_autotune.py`、`test_autotune_defaults.py`、
`test_forward_autotune.py` 合计 33 passed、7 failed；在改动前的 `7aaaded`
独立 worktree 上同样是这 7 项失败（旧宏枚举、候选数量、shared-memory
固定值断言），不把它们列为本轮通过，也未增加或放宽微观断言。

待办：完整调用正反序确认；
导出验收 profile 后再关闭性能门禁。

## 2026-10-06：MegaMoE 缩容与 PF 工作流对齐诊断

本轮在独立分支 `experiment/pf-megamoe-alignment` 上进行，保留 V3 恢复分支。
首先只改变 MegaMoE 的对称区容量，不改变其 kernel 或调度。结论是：
**本次负载下，MegaMoE 的大额默认分配不是领先的主要解释**。
8K 压到 PF 同级预算后耗时变化 +0.71%，再降到最小合法容量为 −0.21%；
20K 降到最小合法容量变化 −0.11%。不能将这些小差异登记为稳定收益，
也不能据此排除两种算法的 credit 粒度、分块与流水组织差异。
另外固定 PF 工作流，把 GEMM 改成 MegaMoE 同样的 N128/K64，3→5
stages 确实降低耗时 8.85%/7.77%，但仍比现有 PF N256/K128、2 stages
慢约 9.4%。这两个候选均未采用，不能把局部 stages 收益算成现有版本的收益。

### 对称区容量消融

单机 EP8，H4096/I2048/E256/top-k6，BF16，设置 SM clock 1620 MHz。
沿用上一节的冻结 MegaMoE 包和推理计时范围：不可变权重在计时外转换，
包含输入 staging、eager 调用与 device synchronize，不保存训练激活。
每个配置完整预热同一条 32 步 trace，之后 8 warmup + 24 measured；
配置顺序及反序各跑一次，报告两轮 rank-max p50 的算术平均。
所有性能配置的全部 rank 的 token 长度和 route/gate hash 相同。
仍是正态变长、每步重采样的热点随机路由，**不是模型采集的真实 trace**。

PF 预算用已验收 `return_frozen/batch4_window2` 的单机推理 native
持久区 `compute_bytes + gin_bytes + notify_bytes`：8K/20K 分别为
2.725/3.021 GiB。该次 PF eager 耗时含每次 W1 repack，因此没有把它
与 MegaMoE 的预重排权重耗时作为纯 kernel 对照。
“同级预算”指两者的 native workspace 总量，不表示内部区域划分相同，
也不包括模型权重、用户输入或训练 tape 的整卡峰值。

| 名义 tokens/rank | MegaMoE 配置 | 对称区 GiB/rank | L1/L2 ring 行数 | 第 1 / 2 轮 ms | 均值 ms | 相对默认耗时 |
|---|---|---:|---:|---:|---:|---:|
| 8192 | 默认 | 6.478 | 509952 | 6.889 / 6.880 | 6.885 | — |
| 8192 | PF 同级预算 | 2.723 | 182016 | 7.011 / 6.856 | 6.933 | +0.71% |
| 8192 | 最小合法 ring | 1.589 | 82944 | 6.878 / 6.862 | 6.870 | −0.21% |
| 20480 | 默认 | 10.614 | 786432 | 16.549 / 16.625 | 16.587 | — |
| 20480 | 最小合法 ring | 4.001 | 208896 | 16.565 / 16.573 | 16.569 | −0.11% |

20K **无法在现有 MegaMoE 合同下压到 PF 的 3.021 GiB**：它按完整
expert wave 调度，最小 ring 必须能容纳任一 expert 接收全部 source
tokens 的情况。该 trace 的对齐 `max_tokens` 是 26112，最小行数为
`26112 × 8 = 208896`。没有绕过 `_C.get_ring_limit_for_mega_moe` 或
修改其最坏路由检查；这里报告的是最接近 PF 的合法配置，而非等容量结果。
8K 对齐容量为 10368，最小 ring 为 82944 行。

最小 ring 并不等于每步都被容量卡住：按同 trace 的 expert-global M128
几何统计，8K 的 24 个 measured step 中只有 4 步存在 rank 总 padded
行数超过 82944，最大为 89728；20K measured steps 最大为 208256，
仍低于最小合法 ring。因此，本轮主要排除了“默认大分配本身”的解释，
没有证明极端拥塞下缩容也完全无代价。

缩容配置通过 Torch 输出对照；最小 ring 另外在 route bias std=0.5/8.0、
8K/20K、各 4 次变长连续调用上校验，最大 relative L2 为 9.36e-5，
未放宽 rtol=0.02、atol=1e-4。强偏置是额外压力校验，不混入性能样本。

复现入口为 `benchmarks/torchrun_megamoe_prefill.py` 的互斥选项
`--megamoe-buffer-mib` / `--megamoe-minimum-ring`。前者按总字节预算
搜索合法、对齐的 ring 行数，预算低于合法下界则报错；默认调用不变。
原始日志、编译产物及串行 runner 在
`build/pf_megamoe_alignment_20261006/`。

### PF pipeline 参数对照

复用已有 `tools/autotune_training.py` 的 profile 展开、资源校验和 native
编译器，独立生成三个库，不修改 CUDA 源码。均以
`return_frozen/batch4_window2` header 为起点，物理 M128、B2048/D2、
静态任务分配和 Pull/Return/Reduce 分工不变，关闭额外 decode/prefill
入口选择。只改变 FC1/FC2 的 N、K 和 stages；基础库与候选从同一源码
重新构建。三个库并行编译耗时约 235–278 秒，正式 GPU 计时期间没有编译。
候选的 legacy GEMM stages 均为 3，实际 PF static stages 如下表所列。

计时是 **PF eager inference + synchronize**，均包含每次 W1 repack；
因此这张表只做 PF 内部消融，不与上表作“预重排权重的纯 kernel”对照。
同一 32 步动态 trace、8 warmup + 24 measured，三个配置正序/反序各一轮，
全 rank trace hash 与容量实验一致。单位 ms，仍报告两轮 rank-max p50
的算术平均，不是 FWD+BWD 耗时。

| FC1 / FC2 N | K | PF FC1 / FC2 stages | 8K | 20K |
|---|---:|---|---:|---:|
| 256 / 256（当前参数） | 128 | 2 / 2 | 9.977 | 24.064 |
| 128 / 128 | 64 | 3 / 3 | 11.975 | 28.547 |
| 128 / 128（MegaMoE 几何） | 64 | 5 / 5 | 10.915 | 26.329 |

固定 N128/K64，增加 stages 的改善是 **8.85%/7.77%**；但 5-stage
组合仍比当前 PF 慢 **9.40%/9.42%**，保留现有配置。本次只是针对
MegaMoE 几何的有界消融，不是对其它 N/K/stages 组合的全局最优证明。

参数之外还存在明确的任务间 pipeline 区别：PF 的 producer 每个 GEMM
tile 经过 `kTaskDoneBarrier` 等待 math/epilogue，FC1↔FC2 切换还通过
`retire_multicast_operands` 等待共享 A stage 的 cluster 读者退休；
MegaMoE producer 沿同一静态任务流及统一 stage 环推进，没有同样的
逐 tile producer/math 交接。PF 的 epilogue 复用已退休的 B staging，
MegaMoE 则有独立 `smem_d`。因此不能只删等待或增加 stage 数：要使
下一 tile 的 TMA 与本 tile epilogue 重叠，还需解决输入/输出 scratch
别名和 cluster 生命周期。这里是代码可确认的机制差异，尚未分离测出
它占总差距的百分比，也没有把所有空转归因为静态任务划分。

三个配置均通过单机 T256/8192/20480、各两步、全 rank 的
output/dX/dGate/dW1/dW2 对照，最大 relative L2 为 6.57e-4（dX）。
这些校验只覆盖本次参数变体，不是尚未实现的 expert-global planner
或 deferred-combine 的验收。未新增细粒度单测。

编译/配置记录为 `pipeline_builds.json`，复现脚本为 `pipeline.py`，
profile artifact ID 分别为 `2b4d3c8b8f65`、`2e313548a2d1`、
`a1229db7a985`，日志为 `pipeline_{performance,verify}_*_ep8.log`。
500 ms 时钟日志覆盖容量、pipeline 实验及数值校验；GPU-util ≥90%
的 404 个样本中，400 个为 1620 MHz，最低 1545 MHz。
未按频率修正结果，不能声称全程完全没有降频。实验结束后停止本轮
监控进程，单机 8 张 GPU 均为 0 MiB，保留 1620 MHz 设置。

### 工作流对齐的边界（待实现，不是已取得的收益）

- **M128 对齐**：PF 现在按 `(source window, expert)` 分别向上取整，
  MegaMoE 按 expert 全部路由行取整。同一 EP8 trace，PF 比 expert-global
  多约 18.4%/20.0% 的 padded 行工作量（8K/20K）；不是已归因的耗时。
  应解耦逻辑 expert 行目录与物理 slot，把通信 packet/window 当 transport
  单元，不再天然当 GEMM 尾块边界。
- **BWD 稳定性**：保留 `build_training_rows` 的 expert/source/route
  canonical 行号和 `route_dst_row`，physical ring row 通过 checkpoint
  rows 映射到 saved X/G/U。只重排 FWD 物理任务不必重排 BWD 输入，也
  不必改变 dW 的归约顺序。仍需校验输出和四项梯度、slot 重用与热点迁移；
  “数学等价”不是并发正确性或逐位相同的证明。
- **先不折叠 Return/Reduce**：可先保留 FC2 epilogue 的远端 contribution
  写回，在所有写入完成并 system-scope 同步后，用 math warps 做 Reduce，
  同时释放原 Return/Reduce 控制 warp 给 Pull。若将全部 Return 也移到
  FC2 之后，则必须另存完整 FC2 输出，不能无成本照搬。
- **容量约束仍存在**：整次调用末尾才 Reduce，source contribution 区
  就必须覆盖整次调用；固定深度的 source ring 不能继续在中途复用。
  有界长文需要按批次 drain，或在足够容量时选择整次合并。尤其不能把
  “完整 expert 先 FC1 再 FC2”直接套在原 D=2 source-window credit 上：
  后续窗口可能等前面的 Reduce 退役，而该 expert 的 FC1 又在等后续窗口，
  会形成等待环。先解耦／重建准入合同，再改变遍历顺序。
- **Pull 与任务顺序**：当前 PF 单机为一 warp Pull、一 warp Return、
  一 warp Reduce（CTA0 用于 admission）；MegaMoE 三 warp Pull。
  PF 当前 window 内先耗尽自有 FC1，再做 FC2；下一 FC1 未 ready 时
  不尝试 FC2。可借鉴 MegaMoE 的 expert wave、独立 L1/L2 slot 退休和
  集中末尾 Combine，但它并不是一个现成的跨节点训练协议。

后续顺序：先验证不折叠 Combine 的通信分工，再解决 expert-global 分块
与有界准入，最后在新工作流上联合搜索 GEMM N/K/stages 与通信 prefetch。
本轮容量实验没有修改 PF/DT/BWD kernel，也未宣称已解决上述结构问题。



## 2026-10-05：Flash 回退定位与通用优化（进行中）

优化摘要：W1 gate/up 重排改为按行的 16B vector copy，避免逐 BF16 的
64-bit 除法/取模；bounded BWD 的相邻 dX N-cluster tile 批量大小从固定 2
变成 `backward.dx_claim_tiles`，一次 gather 可供多个 N tile 复用。
同一 Flash EP8 动态 trace 的两轮完整 FWD+BWD，前者降低约 2.4 ms；
后者在 20K 再降低约 4.5 ms。后续联合搜索 dX 粒度和独立窗口，
在已含 vector W1 的基线上，EP8/EP16 的20K又降低6.51%/7.50%，8K基本持平。
**尚未达到超过 Megatron 的目标**。
这些改动不缓存权重值，不改变 GEMM K 累加次序、ring credit 或 dX→dW
严格先后关系；算子默认 dX batch 仍为2、窗口倍率仍为1。完成联合验证后，
默认训练搜索已加入batch2/4/8、窗口倍率1/2，不按模型名选择实现。

同步正确性补充：定位到 BWD 两个 CTA 读到不同代的任务 mailbox，改为
显式 cluster 级 release/acquire，并补齐 Return generic store→TMA read
的 proxy fence。Flash 三轮双机长回放（每 rank 共3072次训练调用）、
两模型 EP8/EP16 的数值与小模型生命周期回归通过；没有新增 buffer。
这是正确性里程碑，两模型八档配对耗时变化为−0.90%～+1.63%，
不把它登记为性能收益。

### 已确认的规模相关问题

1. 每卡权重体积增大远快于每 token 的 GEMM 工作量。由
   H2048/I512/E64/K4 切到 H4096/I2048/E256/K6，EP8 的 W1 从
   32 MiB 增至 1024 MiB、W1+W2 从 48 MiB 增至 1536 MiB，均为 32 倍；
   useful GEMM FLOPs/token 只增至 12 倍。原逐元素重排不再是可忽略的
   开销。单步 rank0 Kineto 中旧重排为 2.835 ms；下面的整轮消融更能
   代表最终收益，不能把单步 profile 的耗时当作 rank-max p50。
2. BWD source Return 容量预算与 FWD B/D 耦合：
   `B * D * top_k * (2 * H + 4) + 448 MiB`。小尺寸 20K trace 的完整
   贡献区约 403.83 MiB，能放入 512.06 MiB 预算；Flash EP8 20K 约
   1210.90 MiB，超过 B2048/D2 的 640.09 MiB 预算，进入 bounded 分窗
   dX。后者额外重新聚行、增加尾块并受 source credit 约束。
   这是容量启发式不够通用，不是按模型名称写了专用 CUDA 分支。
3. bounded dX 原来每个静态任务只算两个 N-cluster tile，却每次 gather
   完整 M128×2I 的 dAB。Flash 的 H4096/N256/cluster-N2 一共有 8 个
   N-cluster tile，同一输入最多重复 gather 四次。批量设为 8 后只做一次；
   对窗口较小的场景仍保留更小 batch 的 N 方向并行度，由搜索决定。

容量放大消融也支持第 2 点：同一 32 步 EP8 trace 上，D2 的 20K 完整周期
80.270 ms，D8 为 69.192 ms，CUDA stream BWD 区间从 51.964 降到
40.996 ms；8K 两者约 31.5 ms、没有收益。但 D8 同时增加 FWD 缓冲，
native 持久区额外约 1211.62 MiB/rank，且使该 20K case 不再需要 bounded
dX，**没有作为 ring 路径已经优化好的证据，也没有据此修改默认容量**。

对实际同一条随机路由 trace 逐步精确计数，20K 的
`sum(padded M rows) / sum(real rows)` 的 median 如下。这里只算几何量，
不是重新执行 GEMM 得到的速度预测；canonical 列合并全部 source/window。

| 拓扑 | 当前 window B | 当前分窗 padding 倍数 | B×2 | B×4 | canonical |
|---|---:|---:|---:|---:|---:|
| EP8 | 2048 | 1.2205 | 1.1045 | 1.0500 | 1.0165 |
| EP16 | 3072 | 1.1446 | 1.0687 | 1.0338 | 1.0082 |

随机路由不是全部原因：改成均衡路由但保留变长 tokens，旧 PF 的 EP8
8K/20K 为 25.288/66.338 ms，Megatron 为 19.205/43.342 ms，差距仍在。
Megatron `deterministic_mode=True` 的随机结果 23.183/52.955 ms，与默认
23.208/52.858 ms 接近。没有通过关闭 Megatron 快路径来制造 PF 优势。
当前 BWD static 入口的 SASS 中 WGMMA/ARRIVE/DEPBAR 为 48/12/12，
未发现逐条 WGMMA 串行化；不能把 legacy 入口的编译警告当作当前入口
的回退证据。dW 已按专家实际 padded rows 加权划分，也不是简单均分专家。

另外核查了参数继承问题：当前 Flash BWD unified math/control
寄存器配额为 208/88，来自小尺寸上的 Y2；历史 Flash profile
`configs/communication_spsc_20260916/flash.hpp` 则为 224/56。本文原 Y2
记录已明确它不是所有模型的通用默认值。10 月 4 日训练搜索只比较了 B/D、
少数 N/K/stage 与 dW tile，未搜索这组配额；因此“已 autotune”不等于已
排除资源划分问题。随后在当前同一 PF 代码上完成正反两轮配对，四档变化
为 +0.05%、−0.18%、+0.67%、−0.12%，**未发现换回旧配额的显著收益**，
详见下方独立消融。不能把这项参数继承直接登记为已证实的回退原因。

### 第一组实际改动的消融

全部为 H4096/I2048/E256/K6、EP8 单机、B2048/D2。使用相同 32 步
变长热点 trace，完整 untimed replay 后 8 warmup + 24 measured，先测
旧库→vector-only→vector+batch8，再反序测一轮；下表分别列每轮 rank-max
wall p50，单位 ms，不把它与上一节 80 步 trace 直接混算。

| 配置 | 8K：第一轮 / 第二轮 | 20K：第一轮 / 第二轮 |
|---|---:|---:|
| 原调优库 | 31.345 / 31.886 | 79.794 / 79.667 |
| 仅 vector W1，dX batch2 | 29.054 / 29.063 | 77.241 / 77.457 |
| vector W1 + dX batch8 | 28.718 / 29.251 | 72.983 / 72.740 |

vector-only 的 CUDA stream FWD 区间由约 13.45/28.12 ms 降至
11.10/25.73 ms；batch8 的 20K BWD 区间由约 51.59 降至 46.78 ms。
stream 区间包含 host enqueue 间隙，不是裸 kernel 时间，两个 phase 的
median 也不能直接相加当作整轮 median。8K 已使用 canonical dX，batch
参数不作用于它，约 0.3 ms 的波动不登记为这项优化的收益。

另用原有设备时间戳诊断 20K：每步先取 rank-max 再取 median，dX 全部
完成前的区间从 33.731 降至 29.045 ms；其后 dW 完成区间为
17.133/17.121 ms，基本不变。前一阶段包括 dU、SwiGLU、dX、通信等待，
不能解释成纯 dX GEMM 时间；诊断模式的 host polling 不用于正式评分。

Flash EP8/EP16、256/8K/20K、每档两步随机路由的 output/dX/dGate/dW1/dW2
均通过共同 Torch 参考，两个候选分别校验。**但双机性能反序确认暴露持续
等待，batch8 尚不能验收**：第一轮旧/vector-only/batch8 的 8K 为
38.140/36.800/36.927 ms，20K 为 94.036/93.631/87.228 ms；第二轮 batch8
完成 8K（36.758 ms）后，20K 长时间无进展。两节点 GPU 100% 占用、约
110–125 W，CUDA-GDB 看到 rank8 活跃的是 static unified BWD，采样线程
停在 cluster barrier；这不是单凭一个线程就定位了根因。已终止该候选的
两个 torchrun 进程，两节点显存均回到 0 MiB，失败日志原样保留。
正开启逐 rank BWD status/scheduler 诊断复现，不将第一轮成功成绩导出为
最优 profile，也不先归因于新 batch 参数或此前 B2304/D2 的等待问题。
开启原生进度轮询后的三轮诊断均完成，但轮询改变 host 等待方式，不能
据此抹掉正常路径的失败。随后冻结库的三轮 address-only 诊断也完成：
用诊断 interposer 只保留 launch 地址日志、保留原 `cudaEventSynchronize`，
不使用进度轮询；日志和 interposer 位于实验目录，没有进入生产代码。
额外日志仍可能扰动时序，六轮成功不等于已修复此前的正常路径等待。

进一步将 trace 加长到 256 步，每档先 prime 256 次、再 8/248，共连续
512 次调用；只在建池后打印一次地址，保留原 host 等待。前两轮均完成，
第三轮再次在 20K 卡住，最终由 420 秒进程超时收尾。两节点显存均归零。
现场 CUDA-GDB 再次看到驻留的 static BWD，部分 CTA 在调度轮询、部分
在 cluster join；attach/detach 没有恢复进展。最初按普通指针导出的
status/scheduler 全为零，未验证 GPU memory-space，**不能作为计数器证据**。
后续三轮长回放完成（其中首轮被调试器中途暂停，只用于诊断），没有新增
有效的根因证据。这些长 trace 不与 32 步性能表直接比较，batch8 仍未验收。

小尺寸 H2048/I512/E64/K4 已通过 EP8/EP16、1/64/8K/20K、每档两步的
完整五项数值校验；B16/D1、实际 T256/prepared capacity 40K 的热点/空
rank、peer payload 复用及 CUDA Graph 销毁/offload 检查也通过。
小尺寸沿用原 GEMM/B/D profile，按旧→新、新→旧各测一次完整训练周期，
同样 32 步 trace、8/24、完整 prime。两轮 p50（ms）如下：

| 拓扑 / T | 原库：第一轮 / 第二轮 | vector W1：第一轮 / 第二轮 |
|---|---:|---:|
| EP8 / 8K | 5.606 / 5.598 | 5.586 / 5.578 |
| EP8 / 20K | 12.263 / 12.467 | 12.243 / 12.175 |
| EP16 / 8K | 9.717 / 9.737 | 9.666 / 9.690 |
| EP16 / 20K | 22.809 / 22.999 | 22.978 / 22.920 |

小尺寸整体按基本持平看待，双机 20K 两轮平均差约 +0.2%，不宣称每点都
获益，也没有用函数布局差异再做专用优化。随后补测的 Flash Megatron
配对见下一节；小尺寸本轮没有重测 Megatron。
**本轮第一个代码里程碑 `fb374b1` 只合入 vector W1 重排**；
dX batch 参数化暂留为未验收候选，未导出新最优表。
没有新增细粒度单测。已有 backward/static-forward/unified autotune 测试
29 项、13 subtests 通过；更大集合 57 项通过、7 项失败，失败都在旧
`test_forward_autotune.py` 的宏清单/网格/SMEM 精确断言，参见此前
[autotune 记录](autotune.md)的已知失败，不能宣称完整测试集通过。

### 最新同 trace 配对：已验收 vector-only vs Megatron

仅使用 vector W1、dX batch2、原窗口的冻结库，**不包含以下未验收候选**。
shape 为 H4096/I2048/E256/K6，两节点目标 1620 MHz；EP8 为单机八卡。
与 Megatron TEGroupedMLP + DeepEP + `moe_act` 比较，通信配额沿用此前
筛选的 EP8=80 SM、EP16=48 SM。两边均保留 X/G/U、BWD 重算 gated SwiGLU；
共同边界包含每步 FWD/BWD、动态分配请求和释放，不含固定池准备/IPC 建连。
32 步相同随机热点变长 trace、完整 prime、8 warmup + 24 measured；
先 PF→Megatron，再 Megatron→PF。四次运行的各 rank workload hash 均一致。

| 拓扑 / T | PF 两轮 rank-max wall p50 / ms | Megatron 两轮 p50 / ms | 两轮 p50 均值比：PF 慢 |
|---|---:|---:|---:|
| EP8 / 8K | 29.338 / 29.118 | 23.171 / 23.199 | 26.1% |
| EP8 / 20K | 77.899 / 77.443 | 53.001 / 52.789 | 46.8% |
| EP16 / 8K | 36.997 / 37.039 | 31.591 / 30.597 | 19.1% |
| EP16 / 20K | 93.197 / 94.477 | 71.023 / 71.505 | 31.7% |

20K 的 CUDA stream BWD 区间：EP8 PF 51.76–52.00 ms、Megatron
31.86–31.97 ms；EP16 PF 59.78–60.53 ms、Megatron 42.74–42.95 ms。
因此剩余差距主要在 BWD，不只是 FWD 权重重排。8K 已是 canonical BWD，
仍有差距，也说明不能把所有回退归给分窗。stream 区间包含 host 间隙，
不是裸 kernel 时间，不能直接相加或当作纯 GEMM 的耗时。

原始记录为 `candidates/matched{0,1}_{pf,megatron}_ep*.log`，正式计时期间
没有并发 GPU 作业、编译或调试器。PF 产物另存于
`candidates/frozen/vector_dx2_20261005/`。时钟日志从单机首轮 PF 完成后
开始，因此不覆盖整张表；采样仍有少量低于 1620 MHz 的点，不能声称
每个 kernel 全程严格恒频。本表不与此前 80 步 trace 直接计算改善比例。

### 寄存器配额配对消融：没有恢复 Flash 性能

只改变 BWD unified math/control 的 208/88 → 224/56；dX batch2、原窗口、
FWD/GEMM tile/stage、B/D 全部相同。两份新库均包含下节待验收的 Return
proxy fence，因此这是**同一候选源码上的寄存器消融**，不能与上一节
vector-only 冻结库混用、也不构成 fence 的独立验证。编译完成后分别复制到
`build/pf_flash_diagnosis/registers/frozen/reg208_88`、`reg224_56` 再运行。

两份库都通过 EP8/EP16、256/8K/20K、每档两步的 output/dX/dGate/dW1/dW2
共同 Torch 参考。正式计时用相同 32 步随机热点变长 trace，完整 prime 后
8/24，先 208/88→224/56，再反序；四次运行的逐 rank workload hash 全部
一致。以下为完整训练周期 rank-max wall p50，单位 ms。

| 拓扑 / T | 208/88：第一轮 / 第二轮 | 224/56：第一轮 / 第二轮 | 两轮均值耗时变化 |
|---|---:|---:|---:|
| EP8 / 8K | 29.366 / 29.026 | 29.203 / 29.221 | +0.05% |
| EP8 / 20K | 77.251 / 77.387 | 77.213 / 77.153 | −0.18% |
| EP16 / 8K | 36.824 / 37.102 | 37.112 / 37.307 | +0.67% |
| EP16 / 20K | 93.610 / 93.816 | 93.732 / 93.463 | −0.12% |

没有据此修改已验收 profile 或导出新赢家。静态 BWD local/network 入口的
STACK 从 0/72 B 变为 64/280 B，REG168/SHARED2048 不变；栈大小不是性能
判据，也不把这一结果推广成其他配额均无效。主要差距仍应继续从计算链、
bounded 聚行与 Return 等待中分解，不能认为寄存器消融已解释全部回退。

日志为 `registers/verify_*`、`matched{0,1}_reg*_ep*.log`；计时期间无并发
GPU 作业、编译或调试器，结束后两节点显存均归零。1620 MHz 目标下，
全程监控每节点 7008 个样本，最低 1440 MHz；以 GPU utilization >90%
且功耗 >300 W 粗筛，node0/1 分别 915/515 个样本，其中 42/19 个低于
1620 MHz。该筛选未与 kernel 精确对齐，不做频率归一化，也不宣称严格恒频。

### 未验收：独立 BWD 窗口与 Return proxy 顺序

尝试增加数值参数 `backward.return_window_scale=1/2`：只在原 contribution
字节预算内放大 bounded dX 聚行窗口，相应减少在途窗口数，不扩大 FWD
X/S/C。窗口还必须受 dX slot 容量约束，保证最坏窗口可以完整驻留；否则
一个跨远隔专家 M-block 的 packet 可能等待复用其首行槽位的后续 block。

初版候选 `85b62f12d83e`（scale2、dX batch2、尚无 proxy fence）通过 EP8 的 256/8K/20K
随机五项校验，以及 EP8/EP16 的 B16/D1、容量 40K、实际 17/513 tokens
集中到一个 rank 的回绕/空专家校验。**但 EP16 普通随机 20K dX 校验失败**：
rank1/5 relative L2 为 1.01%/1.73%，rank2 最大绝对误差 0.00562；这不是
允许的 BF16 尾差。候选未进入性能评分，没有调宽参考门限。失败后精确终止
该测试的两节点 torchrun，显存均归零；重新运行相同校验，以及额外 12 步
20K 校验均通过，说明不能仅按确定性 shape 算错解释，也不能把重试通过
当作修复。12 步重试日志于 06:01:48 前写完，新 fence 库于 06:02:21
才完成链接，确认这两次重试仍使用初版代码。

代码检查发现 Return packer 从 generic vector-store 写出的 dX 槽位发起
TMA 读取前缺少显式 async-proxy fence。已补一个最小候选，在现有 queue/
mailbox acquire 后、每个 TMA issuer 的首次读取前执行
`fence.proxy.async.global`，尚在消融验证，不提前声称解释或修复了上述
数值偏差、或此前 batch8 的等待。跨 proxy 顺序依据
[PTX fence.proxy 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)。
新增窗口参数、dX batch 参数尚未提交或导出为最优配置。该 fence 后续与
下节的 cluster 内存顺序修正一并完成正确性回归，独立于参数搜索提交；
仍不把早先一次数值失败全部归因于缺少这一条 fence。

fence 候选的双机 20K、12 步共同五项校验通过，但随后 256 步 trace
长回放完成 8K 后，在 20K 再次持续等待（`stress_window2_proxy_ep16`，
420 秒超时，两节点显存归零）。**dX batch2 也能复现，不能把等待仅归因于
batch8，更不能把 fence 当作修复**。本次 CUDA-GDB 以显式 global 地址空间
成功读取 scheduler、ring state/blocks/slots；rank8 有 712 个 dX 块，
660 个已完成并退休，剩余 52 个均在 window7（source group1、token
window3），它们 `input_rows == rows`、对应 slot credit 为零且都是首代
空槽。dU 的 651 个 canonical M-block 全部完成。按 `input_ready` 的三个
条件，等待缩小到本地 source-window reservation，尚需追溯缺失的
Return/Reduce credit。rank0 的 dX/dW 完成计数也已取得，不能把此现场
解释为 GEMM 计算时间长。status 指针读取失败仍不作为计数器证据。

### 等待继续定位：跨 CTA 任务 mailbox 分歧

后续只对冻结的 window2 + batch2 + proxy-fence 库重放，不改变 CUDA 实现。
`credit2_window2_proxy_ep16` 中，rank0/1/8/9 缺少的 contribution 全部指向
rank1；rank1 的 1438 个 dX 块中，1331 个退休、62 个计算完成待 Return、
45 个只完成 6/8 个 N-pair。首个未完成块 input rows 已齐、无本地 source
依赖，且仍为无需回收 credit 的首代槽位。`credit3` 类似，但计算积压在
rank15。因而先前看到的 source-window 等待是下游症状，不能登记为根因。
两次都由 420 秒超时收尾；`credit4` 完成，不能用这一成功抹掉前面的失败。

`credit5` 再次失败，按未完成块的静态 owner 自动抓取，定位到 rank4 的
cluster44（CTA88/89）：CTA88 mailbox 为 `NoTask`，CTA89 为
`Dx(owner=1232, tile=0, count=2)`。块1232 已经完成并退休；CTA89 却再次进入
该任务，math 等 TMA full，producer 在耗尽可用 stage 后等 empty，CTA88
则等下一次 mailbox join。间隔一次 detach/attach 的第二份快照仍完全相同。
其后9个 dX 块各缺少2个 N-pair；这与该 cluster 的静态 residue 一致。
这是 CTA 对任务不一致的直接证据，不再只是任意线程停在 barrier 的采样。

检查本地 CUTLASS 头文件发现，此处 `ClusterBarrier::arrive/wait` 发出的
PTX 没有显式 `.scope`，默认是 CTA，而非 cluster；`.shared::cluster` 只
说明 barrier 地址空间，不会自动把内存顺序扩大到 cluster。跨 CTA 的
DSM mailbox 需要 cluster 级发布/读取顺序，参见
[PTX mbarrier.arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)
及 [mbarrier wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。
据此增加显式 release.cluster / acquire.cluster 的最小候选，并保留
双 arrival、原 phase 规则、原计算线程域及 mailbox-copy acknowledgement。
候选已通过 EP16、256/8K/20K、各两步五项 Torch oracle，以及三轮
每档512次连续调用的长回放（每 rank 共3072次完整训练调用）；第三轮
没有并发编译，三轮均无调试器 attach。local/network 的 REG168/SHARED2048、STACK
0/72 B 不变；static network 入口 HGMMA/DEPBAR 仍为48/12，额外的内存
顺序指令已在 SASS 中确认。**没有用诊断期间的耗时更新性能表。**冻结库 SHA256 为
`2698c99e26e3c1b6b23710511d687c411175f19003e5d2b048a7596e59054e88`。

随后完成原窗口（scale1）与扩大窗口（scale2）的 Flash EP8/EP16、
256/8K/20K、各两步五项共同 Torch oracle；小尺寸 H2048/I512/E64/K4 的
EP8/EP16、1/64/8K/20K、各两步也通过。小尺寸 B16/D1、实际 T256/容量
40K 的热点/空 rank、peer payload 复用、CUDA Graph 销毁/offload 检查
均通过。未新增单元测试或生产诊断分支。
本次代码里程碑只包含 cluster 内存顺序和 Return proxy fence，不包含
实验中的 batch/window 参数化，也不修改默认调优表。性能配对见下一节；
原窗口对照库 `registers/frozen/reg208_88` 已有相同 Return fence，能独立
消融 cluster 修正；小尺寸旧 vector 库没有此 fence，需按两项合并修正
报告，不能冒充单因素消融。

证据为 `candidates/credit{2,3,5}_window2_proxy_ep16/` 下的 CB0、ReturnBlock、
ready/route dumps；最后一轮另有 `compute_gdb_rank4.log`、
`mailbox_gdb_rank4.log` 和两次 shared-memory 快照。所有调试运行均不计入
性能评分；新候选独立保存在 `build/pf_flash_diagnosis/mailbox/`。

### 同步修正的配对成本

提交 `895c756` 后，保持原 GEMM、dX batch2、原窗口，两个模型分别跑
旧→新、新→旧；同一32步变长热点 trace、完整 prime、8/24、rank-max
完整 FWD+BWD wall p50，单位 ms。逐 rank workload hash 全部一致。
Flash 旧库为同有 Return proxy fence 的 `reg208_88`，只消融 cluster
修正；小尺寸旧库为 vector W1 版本，比较的是两项内存顺序修正的合计。

| 模型 / EP / T | 旧：第一轮 / 第二轮 | 新：第一轮 / 第二轮 | 两轮均值耗时变化 |
|---|---:|---:|---:|
| Flash / EP8 / 8K | 29.112 / 28.675 | 29.159 / 29.570 | +1.63% |
| Flash / EP8 / 20K | 77.537 / 76.536 | 77.475 / 77.956 | +0.88% |
| Flash / EP16 / 8K | 36.711 / 36.587 | 36.926 / 37.150 | +1.06% |
| Flash / EP16 / 20K | 93.270 / 93.371 | 93.672 / 93.197 | +0.12% |
| 小尺寸 / EP8 / 8K | 5.574 / 5.618 | 5.587 / 5.504 | −0.90% |
| 小尺寸 / EP8 / 20K | 12.114 / 12.117 | 12.249 / 12.282 | +1.24% |
| 小尺寸 / EP16 / 8K | 9.734 / 9.689 | 9.696 / 9.765 | +0.20% |
| 小尺寸 / EP16 / 20K | 22.929 / 23.637 | 22.880 / 23.346 | −0.73% |

没有把这一正确性修正登记为性能收益，也不把小差异都解释为指令的净成本。
日志在 `mailbox/paired/`，计时期间没有编译、调试器或其它 GPU 作业。
全程记录频率：Flash EP8 node0 为1800个样本，最低1500 MHz；Flash
EP16 两节点为5184/5192个样本，最低1455/1485 MHz。按 utilization>90%
且功耗>300 W 粗筛，低于1620 MHz的样本分别为21/407、18/495、12/517；
未与 kernel 精确对齐，不做频率归一化。小尺寸所有采样均为1620 MHz。
随后用同一 autotune 入口联合搜索 dX batch2/4/8 与窗口倍率1/2，
不切换模型专属实现；实验配置为 `build/pf_flash_diagnosis/return_search.json`。

### dX 粒度与独立窗口联合搜索

六个组合全部保持同一 GEMM、同一 source contribution 字节预算和严格的
dX→dW 顺序。EP8/EP16 复用同一批六份编译产物，只有 prepare 时的 B/D
不同。先完成所有编译，再逐候选执行32步完整 prime、8/24；每阶段保留
incumbent，入围者正反顺序确认。以下是完整 FWD+BWD 的 rank-max wall
p50，单位 ms，不与最终80步 trace 的数值混算。

| 拓扑 / T | 原 batch2/scale1，两轮 | 候选，两轮 | 两轮均值耗时变化 |
|---|---:|---:|---:|
| EP8 / 8K | 29.692 / 29.304 | batch4/scale2：29.263 / 29.301 | −0.73% |
| EP8 / 20K | 78.032 / 78.062 | batch4/scale2：72.767 / 73.173 | −6.51% |
| EP16 / 8K | 36.989 / 36.922 | batch8/scale2：36.602 / 37.042 | −0.36% |
| EP16 / 20K | 94.561 / 93.707 | batch8/scale2：86.781 / 87.359 | −7.50% |

8K 使用 canonical 路径，不把不足1%的差异归于这两个 bounded 参数。
EP16 的自动选优按两档等权几何平均确认改善4.00%；原配置与胜者均通过
256及最终8K/20K、各两步的 output/dX/dGate/dW1/dW2共同参考。
EP8 候选的256和8K/20K也通过，但原配置256校验的第二条 JSON 被 NCCL
日志拼到同一行，导致工具记作 `incomplete result`，决策原因为
`incumbent_not_confirmed`。原日志不修改、不把它改名为完整通过；已将
NCCL日志按进程分离，随后独立补验原配置256/8K/20K，各两步全部通过。

稳定性：batch8/scale2完成三轮EP16长回放，每档prime256次再8/248，
每rank共3072次完整FWD+BWD；第三轮无并发编译，三轮均无调试器attach，
使用原有host等待方式。batch4/scale2的EP8同样完成一轮，每rank1024次。
两种拓扑的选中配置还通过B16/D1、prepared capacity40K、实际17/513 tokens
每档两步的热点/空rank五项共同oracle，显式覆盖slot回绕与容量约束。
日志在 `return_validation/`。没有新增细粒度单测，没有增加contribution
字节预算；默认训练配置现已开放这两个数值搜索轴。

最终80步trace、完整prime、20/60得到EP8 29.276/72.839 ms，EP16
36.732/84.542 ms（8K/20K）。这些数值不与上表32步结果计算收益，也没有
把更早的Megatron成绩拼成新的最终配对。两档/两拓扑仍未证明超过Megatron。

双机首次启动继承 shell 的 `NCCL_IB_DISABLE=1`，六个候选均在 GIN 初始化
时报 `NCCL_GIN_TYPE_NONE`，未进入性能计时。确认进程退出和两节点显存归零
后，按既有网络设置显式启用RDMA重跑；失败目录保留，不计入上表。
有效日志为 `return_search_ep8/`、`return_search_ib_ep16/`；六份库在修改
后续CUDA源代码前冻结到 `return_frozen/`，含SHA256、配置和当时源码diff。
时钟记录仍在 `return_search_ep16/clocks_node*.csv`，覆盖失败启动与有效重跑。
每节点20488/20504个样本，均为1425–1620 MHz；utilization>90%、功耗>300 W
的粗筛样本中，77/1533、87/1547低于目标，不作频率归一化或严格恒频声明。

### 下一组数值消融：GEMM K与dW分区轮数（未验收）

新增实验参数 `backward.activation_tile_k=64/128`，dW保持K64；两阶段
复用同一SMEM arena，但分别计算stage间距。K128激活A使用两个K64 TMA
slab，权重MN-major map的K box扩为128，保留每个输出原有的K16累加次序。
另外把现有按实际padded K加权的dW静态区间细分成若干轮，cluster按轮
循环处理其预定区间；`backward.dw_partition_rounds=1/8/32/128`控制粒度。
仍不使用共享动态claim，不做split-K、不提前dW，不按模型名切换实现。
其目的分别是减少长归约的流水握手、检验专家数据的L2复用；目前只是待测
假设，不据代码结构宣称收益。默认仍为K64、单轮，既有公开搜索尚未开启
这两项。实验配置为 `weight_search.json`、`gemm_search.json`。

现有Backward/static-forward/unified/defaults共30项CPU测试通过，没有新增
单元测试；额外topology测试的Forward子项仍请求不存在的`incumbent`
候选而失败，对应Forward工具及配置未修改。五份库已并行编译完成并冻结：
K64配1/8/32/128轮，以及K128配1轮。local/network的REG168、SHARED2048、
LOCAL0、STACK0/72均不变；GPU数值与性能尚待验证，不能据资源相同宣称无回退。
构建及源码快照位于 `build/pf_flash_diagnosis/gemm_candidates/`。

### V3 历史结果的口径

V3 确实在该 Flash 形状领先过。9 月 13 日固定均衡负载、1980 MHz 的
FWD/BWD 独立阶段 p50 相加：EP8/8K 为 18.864 vs Megatron 21.151 ms，
EP16/8K 为 21.741 vs 23.967 ms，见
[历史报告](../benchmarks/performance_20260913.md)。1620 MHz 的 9 月 14 日
记录也有优势：EP8/8K 的阶段 wall **均值**为 7.885+10.847 vs
9.230+14.698 ms，EP16 为 11.203+10.774 vs 10.316+15.974 ms，见
[全 CTA 报告](../benchmarks/full_cta_all_worker_20260914.md)。
但这些是固定均衡路由、权重预备和 cached-state 独立阶段，不能冒充本轮
随机变长、逐调用重排的完整 FWD+BWD；本轮尚未新测匹配该 trace 的原版
Flash V3。既不能否认历史领先，也不能用历史毫秒数证明当前差距已恢复。

原始证据位于 `build/pf_flash_diagnosis/`：`candidates/perf*_ep8.log`、
`verify_*`、`clocks_*`、`geometry*`；此前分阶段/均衡/D8/Megatron 消融及
单步 trace 也保留在该目录。性能表对应的候选 CUDA 库与头文件保存在
`candidates/cache/`，正式测量期间没有并发 GPU 作业或编译。注意 native
cache 按配置命名、源码变化会原位更新库，并非不可变归档：
`85b62f12d83e` 已被后续 fence 构建覆盖，不能再用该路径复现初版窗口
失败。新 fence 库另存于 `candidates/frozen/window2_proxy_20261005/`，
SHA256 为 `07ed73ee80e86bbd163ddc213fa7f7f6aa18b475070ea892e17947a651faf9d6`；
后续诊断改为独立复制产物后测试。
双机 Flash 部分进程级时钟监控记录 4800 个 GPU 样本，1425–1620 MHz，
13 个低于目标；含准备、诊断与等待期，不等于计时窗口内的降频比例。

## 2026-10-04：配置驱动的训练 autotune 与 Flash 复测

优化摘要：接通按 shape/拓扑生成、编译和选优的完整训练 autotune；放大
ring 的 B、调整 D，并将长文 FC1 改为 N256 / 2 stages。新一轮完整
FWD+BWD 中，EP8 的 8K/20K 耗时下降 **14.0% / 22.2%**，EP16 下降
**4.8% / 13.5%**。BWD 的 tile/stage 消融未获得稳定收益，保留原参数。
**调优后仍慢于 Megatron + DeepEP**；另有一个网络候选在确认中卡住，
已剔除，但根因未定位。本轮没有修改 CUDA 算法或修复该等待问题。

### 自动搜索入口与最终参数

原有工具已经有配置展开、编译缓存和选优逻辑，缺的是当前 PF 动态训练
FWD+BWD 的评分入口。新增 `tools/autotune_training.py` 和
`configs/pf_training_autotune.json`，复用既有参数校验、资源过滤、native
object cache、并行构建和确认选择逻辑，不再依靠手改各 shape 的头文件。
H/I/E/top-k、token 档、节点和每节点 GPU 数由 CLI 输入，配置描述搜索空间。
使用方法及参数含义见 [autotune 文档](autotune.md)。

| 项目 | EP8 单机 | EP16 双机 |
|---|---|---|
| B / D | 2048 / 2 | 3072 / 3 |
| FC1/FC2 N，K | 256 / 256，128 | 相同 |
| FC1/FC2 static stages | 2 / 2 | 相同 |
| BWD | dU/dX N256；dW M128/N256；K64；activation 4 stages、dW 3 stages | 相同 |
| prepared max_tokens：8K / 20K trace | 10175 / 25820 | 10390 / 25964 |

实际使用同一份 compiled profile：`e09cd7591438`。B/D 是准备时参数，不
进入 CUDA 编译 key；同一模型的两种拓扑不需要为这些容量参数重复编译。
初次基准构建约 301.5 秒；EP16 复用后，依赖检查、链接和 adapter 构建约
10.4 秒，没有重新编译 CUDA 源码。单机初筛每个候选包含两个 token 档，
约 29–30 秒；双机约 80 秒，包含计时区间外的进程/通信池准备和 trace 回放。

冻结头文件和四个 `training=true` 表项在
[`configs/pf_training_flash_20261004`](../configs/pf_training_flash_20261004/README.md)。
最终对照通过 `WEFTMOE_AUTOTUNE_TABLE` 实际加载了自动导出的表。
key 使用真实 prepared capacity，不把 nominal 8192/20480 当作容量。
这是 PF 专用库，关闭额外 prefill/DT 入口选择；未替换默认 DT、其它模型
或原冻结库。组合 ID 同时覆盖 FWD/BWD 参数，避免载入错误的 BWD 配置。

### 各步消融与收益

搜索用同一条 32 步动态 trace：完整 untimed replay，然后 8 warmup +
24 measured。分数为两档 rank-max p50 的等权几何平均。每阶段保留
incumbent，前两名正反顺序各测一次，至少改善 0.5% 才切换，并校验五项
输出/梯度。以下初筛 ms **不能与最终 80 步 trace 的 ms 直接混算收益**。

第一步，搜索 B/D；GEMM 保持原来的宽 prefill 入口不变：

| 拓扑 | B / D | 8K ms | 20K ms |
|---|---|---:|---:|
| EP8 | 512 / 8（原值） | 36.669 | 102.640 |
| EP8 | 1024 / 4 | 33.223 | 87.317 |
| EP8 | **2048 / 2** | **31.928** | **81.145** |
| EP8 | 4096 / 1 | 33.761 | 83.282 |
| EP8 | 2048 / 4 | 32.151 | 81.860 |
| EP16 | 768 / 6（原值） | 39.453 | 106.595 |
| EP16 | 1536 / 3 | 38.261 | 99.030 |
| EP16 | 4608 / 1 | 44.530 | 112.787 |
| EP16 | **3072 / 3** | **38.507** | **96.955** |

确认后，ring 阶段的综合耗时下降 EP8 **16.61%**、EP16 **5.91%**。
较大的 B 改变窗口内的任务密度、padding 和流水组织；本轮测量的是整体
效果，没有把这几项分别归因为已测出的 GPU 时间。

第二步，在选定 B/D 上搜索 FWD；括号中为 FC1/FC2 N、K、static stages：

| FWD 配置 | EP8：8K / 20K ms | EP16：8K / 20K ms |
|---|---|---|
| 原宽入口（128/256，128，3/2） | 31.785 / 81.456 | 38.724 / 96.553 |
| **256/256，128，2/2** | **31.162 / 79.688** | **37.766 / 95.258** |
| 128/256，64，3/3 | 32.457 / 82.696 | 39.272 / 97.462 |

确认后，FWD 调参又使完整周期综合耗时下降 EP8 **1.72%**、EP16 **3.20%**。
这里是完整周期收益，不是独立 FC1 GEMM 的加速比。原库宏里的窄入口
N64/N128 在这些长文 case 不活跃，实际对照是上表的宽入口。
最初另有 N256/K64/4-stage 组合，因其附带的 legacy FC1 3-stage staging
超过共享内存预算，编译前就被过滤；发布配置已移除这个必然无效的组合。

第三步，固定前两步参数，分别尝试 dW M256/N128、activation stages 3、
dW stages 2。初筛如下：

| BWD 配置 | EP8：8K / 20K ms | EP16：8K / 20K ms |
|---|---|---|
| 原值 | 31.681 / 79.434 | 37.610 / 94.788 |
| dW M256/N128 | 31.253 / 79.659 | 37.964 / 94.298 |
| activation stages 3 | 31.115 / 80.764 | 38.051 / 95.192 |
| dW stages 2 | 32.412 / 81.852 | 38.953 / 96.670 |

两种拓扑的确认结果均保留原 BWD，未将初筛的小幅波动登记为收益。
这是一轮有限的分阶段搜索，不是全部参数组合的全局最优证明。

### 最终同负载对照

形状和随机负载沿用下一节：H4096/I2048/E256/top-k6，每 rank 的 tokens
按 nominal 附近 σ=10% 的正态分布生成、截断到 ±3σ，路由有 σ=0.5 的
共享 expert bias。不是实际模型采集的 route trace。

本次另跑一轮 PF 未调优、PF 调优、Megatron + DeepEP，每档完整回放
80 步，再 20 warmup + 60 measured；下面均为 **60 个 rank-max 样本的 p50**，
不是与上一轮样本混合后的 p50。Megatron 仍使用单机 80 SM / 双机 48 SM、
真实 TEGroupedMLP、selective `moe_act` 和此前相同的融合配置。
两边均保留 routed X 与 FC1 gate/up，BWD 重算 gated SwiGLU、不重算 FC1。

| 拓扑 / nominal T每卡 | PF 未调优 ms | PF 调优 ms | 耗时下降 | Megatron DeepEP ms | 调优 PF 相对 Megatron |
|---|---:|---:|---:|---:|---:|
| EP8 / 8192 | 36.5411 | 31.4364 | **13.97%** | 22.9898 | +36.74% |
| EP8 / 20480 | 101.2710 | 78.7509 | **22.24%** | 52.7426 | +49.31% |
| EP16 / 8192 | 39.7691 | 37.8442 | **4.84%** | 30.9829 | +22.15% |
| EP16 / 20480 | 106.2185 | 91.9099 | **13.47%** | 70.5138 | +30.34% |

等效 useful GEMM TFLOPS/card：

| 拓扑 / T | PF 未调优 | PF 调优 | Megatron DeepEP |
|---|---:|---:|---:|
| EP8 / 8192 | 199.61 | 233.53 | 319.04 |
| EP8 / 20480 | 183.28 | 234.68 | 352.81 |
| EP16 / 8192 | 186.15 | 194.50 | 237.64 |
| EP16 / 20480 | 175.32 | 201.50 | 264.09 |

计算仍为逐 step 的 `18 × sum(actual T_rank) × K × H × I / EP / seconds`，
再取 median、除以 1e12。18=FWD 6+BWD 12；不计 padding、repack、SwiGLU
elementwise 等额外工作，不使用 nominal T 除以 p50 来代替逐 step 统计。
时间包含完整 eager FWD+BWD、释放与 device synchronize；排除输入生成、
首次编译、固定池分配/IPC 准备，保留每次权重 repack 和每步实际 buffer 工作。

12 个本轮性能 JSON 的逐 rank route/gate hash、tokens、received rows 和
prepared capacities 均匹配，且与上一轮两组 PF 历史 trace 一致。
Megatron 四个结果中所有 rank 在计时段新增编译图均为 0。
上一轮未调优 PF 为 36.7235/101.9872、39.8510/105.7693 ms；较快 DeepEP
为 22.9785/52.5710、30.7498/69.9667 ms。本轮基线复测与其接近，收益并非
来自更换负载或采用更慢的 Megatron 设置。

### 内存、数值与未解决问题

native 持久区 `compute_bytes + gin_bytes + notify_bytes`，单位 MiB/rank：

| 拓扑 / T | 未调优 | 调优 | 增量 |
|---|---:|---:|---:|
| EP8 / 8192 | 2608.34 | 4245.61 | +1637.27 |
| EP8 / 20480 | 4407.02 | 5856.51 | +1449.49 |
| EP16 / 8192 | 4072.06 | 6918.81 | +2846.75 |
| EP16 / 20480 | 6393.55 | 9180.52 | +2786.98 |

这次是明确的空间换时间，持久区增加约 **1.4–2.8 GiB/rank**，超过此前
“500 MB 以下不用顾忌”的量级。上表不是整进程峰值，也没有包含 Torch
保存激活、梯度及 allocator reserved；不能把它当作净总显存增量。
默认搜索预算为 native 持久区 12 GiB；需要更紧预算时修改配置再搜索。

最终 PF 与 Megatron 均通过 EP8/EP16、8K/20K、每档两步的共同数值校验，
覆盖 output/dX/dGate/dW1/dW2。门限仍为 rtol=0.02、atol=1e-4、relative
L2 < 1%。新通用参考按本地专家逐个计算，再以 FP32 合并跨专家/跨 rank
output/dX/dGate，不复制全局权重；与旧参考的 BF16 逐次累加不同，因此
误差数字不能直接当作相对历史的精度提升。最终最大 relative L2：PF
0.01163%，Megatron 0.29531%，两者全部通过逐元素检查。

**失败候选保留：** 初始 EP16 搜索的 B2304/D2 初筛为 38.597/96.742 ms，
首轮确认也成功，但第二轮在首个性能 JSON 前持续占满 GPU、数分钟无进展。
人工终止这两个已定位的测试进程组后，两节点显存正常释放，未发现本轮
新增 GPU Xid。该组合已从默认空间剔除，全部确认重跑；根因未定位，不能
宣称排除组合就修复了底层同步问题。工具现会拒绝任何确认轮失败的候选，
不使用其部分成绩导出。其余最终选中配置完成了本节所有重复运行和校验。

两节点目标均锁 1620 MHz、功耗上限 700 W。最终对照的进程级监控含
warmup/数值校验等，共 3920 个 GPU 样本，实测 1440–1620 MHz，43 个
样本低于目标、46 个样本带软件功耗限制标志。因此仍不是严格恒频实验，
不作频率归一化，也不把这些样本比例当作计时区间内的降频比例。

工具里程碑提交：`2b9ca94`（训练 config/编译/选择/导出）、`df1dbd2`
（确认失败处理、剔除异常候选）。相关 static-forward/unified autotune
测试 15 项通过；另跑旧 topology 测试时，未改动的 Forward 入口仍有
`--candidate incumbent` 名称失配失败，本轮未改该无关入口。

原始证据：`build/pf_flash_autotune/automatic_ep8/`、
`automatic_ep16/`（初次失败保留）、`automatic_ep16_retry/`、
`comparison/`、`summary.json` 和对应 `*_clocks.log`。
最终调优库 SHA256 为 `33565b1f42831f27f7a3cd7fc6772843401ea99efe7494e058d3c2a6b65d3fee`；
未调优库仍为 `8663aea76fff12915b96deae9753515e2bfa171066fc27236a2673080c2c5bf5`。
全部作业已退出，两节点均确认 0 MiB GPU 占用、空闲 SM 时钟 1620 MHz。

## 2026-10-04：Flash 尺寸、变长热点负载与 Megatron 对照

本轮使用仓库历史报告中的 DeepSeek-V4-Flash routed-expert 尺寸：
H4096/I2048/E256/top-k6、BF16 SwiGLU。它是单个 MoE 层的完整训练
FWD+BWD，不是整模型训练 step；不含 router projection/top-k、shared
expert、attention、optimizer，也不复现模型特有的量化或其它激活策略。

本轮未 autotune 的结论：**PF 在这四档均慢于 Megatron + DeepEP**，耗时高
29.6%–94.0%；只有双机两档快于 Megatron + NCCL。小尺寸上的优势没有
自动泛化到 Flash。本轮仅完成尺寸构建、验证和 benchmark，没有性能修复。

### 完整 FWD+BWD 结果

单位 ms，均为两轮合并后的 rank-max p50。最后一列大于零表示 PF 更慢。
DeepEP 80/48 指单机 80 SM、双机 48 SM，而不是在两个轮次间切换配置。

| 拓扑 / nominal T每卡 | PF | Megatron NCCL | DeepEP 20 SM | DeepEP 80/48 SM | PF 相对较快 DeepEP 耗时 |
|---|---:|---:|---:|---:|---:|
| EP8 / 8192 | 36.7235 | 25.4283 | 24.7668 | 22.9785 | +59.82% |
| EP8 / 20480 | 101.9872 | 58.2629 | 57.0120 | 52.5710 | +94.00% |
| EP16 / 8192 | 39.8510 | 56.1553 | 34.0550 | 30.7498 | +29.60% |
| EP16 / 20480 | 105.7693 | 134.9669 | 78.6683 | 69.9667 | +51.17% |

PF 对 NCCL 的耗时变化依次为 +44.42%、+75.05%、−29.03%、−21.63%。
DeepEP 从 20 SM 改为 80/48 SM 后，四档耗时降低
7.22% / 7.79% / 9.71% / 11.06%；主结论使用这组更快的基线。

等效 TFLOPS/卡，同样对逐步吞吐取 p50。每步按实际全局输入 token 数
计算 `18 * sum(T_rank) * K * H * I / world / wall_seconds / 1e12`；
FWD 为 6RHI，BWD 为 12RHI。不计 padding、elementwise 的额外 FLOPs，
但它们、通信和主机开销均计入时间；不是纯 GEMM 或 cached-BWD TFLOPS。

| 拓扑 / nominal T每卡 | PF | Megatron NCCL | DeepEP 20 SM | DeepEP 80/48 SM |
|---|---:|---:|---:|---:|
| EP8 / 8192 | 198.8 | 291.2 | 296.2 | 320.0 |
| EP8 / 20480 | 183.0 | 319.6 | 324.9 | 353.4 |
| EP16 / 8192 | 185.6 | 132.8 | 215.6 | 238.7 |
| EP16 / 20480 | 175.9 | 138.1 | 237.3 | 265.2 |

PF A/B 两轮 p50 分别为 36.8021/36.6921、102.3804/101.4949、
39.8510/39.8350、106.2691/105.1251 ms；较快 DeepEP 为
22.9737/22.9906、52.5988/52.5710、30.9970/30.7311、
69.9667/70.1412 ms。差距在反序复测中仍成立，不是选取某一慢轮的结果。
全部均值、p95、逐步/逐 rank 样本保留在 JSON。

### 输入、缓存与计时

- 沿用之前的 `variable_tokens_random_routes`，不是前一轮的历史固定均衡
  数据，也不是从实际模型采集的 route trace。每 rank 的 T 从 nominal
  8192/20480、标准差 10%、截断 ±3σ 的正态分布独立采样；每步重新生成
  X/dY、routes/gates，共享专家 logit bias 的标准差为 0.5，制造跨源热点。
- seed=123，各实现消费相同的逐 rank 输入序列和权重。专家权重在本轮内
  不更新，但 PF 每次调用的 repack 仍计时，没有复用已删除的 M 快路径。
- 单机 EP8 使用第一台节点的八卡；双机 EP16 使用两台各八卡。PF 保留
  B512/D8、B768/D6 和 network lanes/Qdispatch/Qreturn=32/8/16。
  只为 H4096/I2048 重建尺寸 specialization，未优化生产 kernel 或重新
  autotune GEMM/ring；此前冻结的 H2048/I512 库及结果没有被覆盖。
- Megatron 使用真实 dispatcher + TEGroupedMLP，TP1/expert-TP1、EP8/16，
  permutation/activation fusion 均开启，selective `moe_act`：双方保存
  routed X 和 FC1 gate/up，BWD 重算 gated SwiGLU、不重算 FC1。
  保留 Megatron 默认非 deterministic 模式，没有人为关闭快路径。
- NCCL all-to-all 与 DeepEP 都测。DeepEP 比较默认 20 SM 和历史 Flash
  配置的 EP8=80、EP16=48 SM；这只是两个通信资源配置的比较，不声称
  对 Megatron 做了穷尽调参。
- 每配置两轮：A 为 PF→NCCL→DeepEP20→DeepEP80/48，B 反序。每轮先
  untimed replay 全部 80 步，再 20 warmup + 60 measured。先逐步取
  rank-max，合并两轮全部 120 个样本取 p50，不挑快轮、不删长尾。
- 计时边界为 `forward → backward → 释放 tape/gradients → device sync`；
  固定 runtime/通信 buffer 初始化、IPC 建链、输入生成和首次编译不计时。
  每次调用的 notify、分配、权重重排、计算、通信和释放均在计时内。
  这是充分预热的动态 eager 循环，不是 cached-BWD 单独计时，也未使用
  CUDA Graph。脚本中的 host forward/backward 分界不是 GPU 阶段计时，
  本节不据此拆分 FWD/BWD kernel 性能。

实测负载如下；“rank 倾斜”是单步最忙 rank 的接收行数 / rank 均值，
“专家倾斜”是最热专家行数 / 全局专家均值。它们均不是人工平衡的负载。

| 拓扑 / nominal T每卡 | 实际 T每卡范围 | 实际 T每卡均值 | rank 倾斜 p50 / max | 专家倾斜 p50 / max |
|---|---:|---:|---:|---:|
| EP8 / 8192 | 5861–10175 | 8138.1 | 1.35 / 2.18 | 8.22 / 18.45 |
| EP8 / 20480 | 14931–25776 | 20619.3 | 1.32 / 1.62 | 8.14 / 15.95 |
| EP16 / 8192 | 5861–10390 | 8150.2 | 1.66 / 2.58 | 8.27 / 18.47 |
| EP16 / 20480 | 14931–25964 | 20608.0 | 1.57 / 1.90 | 8.13 / 15.93 |

32 组 JSON、共 1920 个 measured rank-max 样本已核对。相同 case 的
逐 rank T/routes/gates SHA256、实际接收行数及容量一致，接收行数满足
`sum(received_rows) = K * sum(T_rank)`；24 组 Megatron 记录的所有 rank
在测量期间新增编译 graph 数均为 0。软件版本：Torch 2.9.1+cu129、
Megatron-Core 0.17.0、TE 2.14.1+27de67e1、DeepEP 1.2.1+567632d，
双方实际加载 NCCL 2.31.2；Torch 编译时的 2.27.5 不作为运行库版本。

### 时钟边界

两节点 L20X 都显式执行 `nvidia-smi -lgc 1620,1620`，功耗上限同为
700 W（硬件允许的最大值），没有再用 1980 MHz。首次 EP8 预跑仅保留在
`pilot_*`，不进入正式统计；重锁后重做 A/B 两轮，并增加运行期间的频率、
功耗及 clock-event 采样。

**锁定目标不等于全程实际恒频。** 重锁后的 Flash 大负载仍采到低于
1620 MHz 的频率，范围为 1455–1620 MHz，且有软件功耗限制标志。
7480 个 GPU 采样中有 102 个低于目标、117 个报告 software power cap。
因此本节只能称为相同 1620 MHz 锁定目标、相同 700 W 功耗上限的对照，
不能称作严格恒频的纯 kernel 消融。
采样覆盖整个进程，包括初始化、输入生成和预热；个别 NVML 采样超时。
不据这些采样推算测量区间的平均频率，也不对时间/TFLOPS 作频率修正。

### 正确性与数值口径

PF、Megatron/NCCL、Megatron/DeepEP 分别在 EP8/EP16 的 nominal
T256/8192/20480 与 Torch reference 比较 output、dX、dGate、dW1、dW2；
216 个 rank-step、1080 个 tensor 检查通过。EP16/20K 仅对不计时的 Torch
reference 按专家 checkpoint，保持相同算术和归约顺序，避免保存整个全局
batch 的 autograd 中间值；被测算子的 `moe_act` 缓存策略没有变化。

旧小尺寸的 `atol=2e-5` 在 Megatron Flash 检查中也会触发：EP8/8K 的
dX 在已报错 rank 的约 3200万–3700 万元素中，2–9 个接近零的元素差约
2.4e-5–3.1e-5。保留原始失败日志 `strict_atol_alltoall_ep8.log`。本轮双方
统一使用 `rtol=0.02, atol=1e-4`，同时要求每个 tensor 的相对 L2 误差
低于 1%，并记录完整误差指标；没有只放宽 PF 的阈值。

实测最大相对 L2 误差，单位为百分比，合并所有验证规模/rank：

| 实现 | output | dX | dGate | dW1 | dW2 |
|---|---:|---:|---:|---:|---:|
| PF | 0.35756% | 0.36026% | 0.01265% | 0.02110% | 0.01159% |
| Megatron NCCL | 0.35758% | 0.35766% | 0.00002% | 0.01248% | 0.01057% |
| Megatron DeepEP | 0.37905% | 0.38206% | 0.00944% | 0.02019% | 0.01191% |

### 对尺寸泛化的含义

本轮未做性能修复，不能把只改 H/I 的 specialization 当成 Flash 最优配置。
相对之前 H2048/I512/E64/top-k4 的小尺寸基准，有两个可直接计算的变化：

- 每输入 token 的有效 GEMM 工作量增加 12 倍，但专家权重总量增加
  **32 倍**。EP8 每卡原始 BF16 专家权重从 48 MiB 增至 1536 MiB，
  EP16 从 24 MiB 增至 768 MiB；原有权重访问/repack 成本不能按小模型
  的占比外推。
- PF 前向每个完整 source 窗口仍包含八个源 rank。按全局专家平均，
  B512 窗口的有效行数从 `8*512*4/64=256` 降为
  `8*512*6/256=96`；B768 从 384 降为 144。M128 的分块、专家热点、
  padding 和权重复用条件都已改变，不能只看整次调用的 8K/20K。

这些是后续分阶段 profiling、B/D 与 GEMM tile 消融的候选依据，**不是**
本轮已经测出的瓶颈占比，也不能据此把所有差距归咎于通信或某一个 GEMM。
9/13 的 Flash 历史数据使用 1980 MHz、固定均衡负载和独立阶段 p50，
不可与本轮动态整轮 p50 直接计算优化收益。

### 构建与产物

生产源码仍是 M7e。新 profile 只覆盖 H/I，复用既有配置：
`configs/static_flash_training_20261004/weftmoe_forward_tuning_generated.hpp`。
构建参数与前轮相同，仅替换 generated-header 的目录：

```sh
export NCCL_INCLUDE_DIR=/usr/include
export NCCL_LIB_DIR=/usr/lib/x86_64-linux-gnu
export NVCCFLAGS="-lineinfo -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 \
  -I$PWD/configs/static_flash_training_20261004 \
  -DWEFTMOE_FWD_STATIC_V4_NETWORK_LANES=32 \
  -DWEFTMOE_FWD_STATIC_V4_PACKET_ROWS=8 \
  -DWEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS=16"
/opt/.venv/bin/python build_native.py lib --output build/pf_flash_workload/libweftmoe.so --jobs 4
/opt/.venv/bin/python tools/build_tensor_runtime.py --library build/pf_flash_workload/libweftmoe.so
```

Flash native library SHA256：
`8663aea76fff12915b96deae9753515e2bfa171066fc27236a2673080c2c5bf5`。
第一次直接加载旧 H2048/I512 库被尺寸校验拒绝，日志保留在
`old_shape_rejected_ep8.log`；不是驱动故障，未绕过形状检查。

原始产物在 `build/pf_flash_workload/`：`build.py`、`run.py`、`validate.py`、
`summarize.py`、`parity_*`、正式 `{a,b}_*` 节点日志、`*_clocks.log` 和
最终 `summary.json`。benchmark 复用 `tools/benchmark_training.py` 与
`benchmarks/megatron_training_backend.py`，没有新增生产功能或细粒度单测。
`run.py verify 8 16` 验证，`run.py a 8 16` / `run.py b 8 16` 测量；
正式计时没有并行编译或其它 GPU 测试。
全部作业正常退出，结束后确认两节点均 0 MiB GPU 占用，锁定目标仍为
1620 MHz；没有遗留测试或监控进程。

## 2026-10-04：历史固定负载原样复测

结论：10 月 1 日旧 PF 的四档整轮结果可复现，本轮与历史差异均小于 0.4%。
当前 PF 相对同机复测的旧 PF，单机 8K 慢 7.24%，单机 20K 约持平；双机
8K/20K 慢 4.14%/4.20%。整轮仍比本轮冻结 V3 低 18.9%–34.8%，但不能用
这一优势抵消相对旧 PF 的剩余回退。本轮仅复测，不修改生产算子或 benchmark。

### 对齐的对象与边界

- 原样使用 `eb371c2:tools/benchmark_training.py`，文件保存在
  `build/pf_variable_workload/benchmark_fixed.py`。两者 SHA256 完全一致：
  `d5f9f333737dc7523e58d41d4536e89ea59c39407a977af2aee710909b06ede5`。
- 旧 PF 为 `build/pf_backward_packed_storage/`；library/extension 的 SHA256
  与文末 10 月 1 日验收一致。当前 PF 使用 M7e 的 `build/pf_peer_bindings/`，
  V3 使用 `build/pf_training_v3_reference/`；均使用冻结二进制，没有重编译。
  旧 PF 保留当时的 M 快路径，未为测试关掉，也没有将 M 移回当前实现。
- Python TensorRuntime 相对 `eb371c2` 仅 close 多出两阶段 IPC 退休处理，
  forward/backward 热路径未改。旧 PF/V3 的 wrapper 只恢复原 close 合同；
  close 在计时外，不改变输入或计算路径。
- 两节点 L20X 固定 1620 MHz，H2048/I512/E64/top-k4、BF16。EP8 与历史
  一样使用第二台节点 `10.10.19.233` 的八卡，EP16 使用两节点。
  PF 的 ring 分别为 B512/D8、B768/D6；不重新调参。
- 每 rank 固定 T8192/20480。X/dY/权重只生成一次，seed=123+rank；路由为
  `(token*17 + rank*31 + slot*53) % 64`，均衡且每轮不变；gate 为原脚本
  的固定 uniform 随机值，不改成上一轮 balanced 模式的 softmax scores。
  容量保留原值 `max_tokens=T`、`max_rows=2*T*K+local_experts*128`。
- 固定输入不等于复用一次 FWD 的计算结果：每步仍重新
  `forward → backward → 释放 tape/gradients → synchronize`，权重 repack、
  tensor 准备和 collective 均保留在 wall 计时内。runtime 初始准备不计时。
- 每版本两轮，均为 50 warmup + 200 measured；A 轮为旧 PF→当前 PF→V3，
  B 轮反转为 V3→当前 PF→旧 PF。先逐步取 rank-max，再合并全部 400 个
  样本取 p50，不挑较快轮，不将两个独立阶段 p50 相加。

### 完整 FWD+BWD 对照

全部为 ms；“历史 PF”指 `eb371c2`，不是 V3。

| 拓扑 / T每卡 | 10/1 历史 PF | 旧 PF 本轮 | 当前 PF 本轮 | V3 本轮 | 当前相对旧 PF 耗时变化 |
|---|---:|---:|---:|---:|---:|
| EP8 / 8192 | 3.9251 | 3.9134 | 4.1967 | 6.4340 | +7.24% |
| EP8 / 20480 | 8.6734 | 8.6395 | 8.6938 | 11.0289 | +0.63%，约持平 |
| EP16 / 8192 | 6.4983 | 6.4930 | 6.7618 | 9.2053 | +4.14% |
| EP16 / 20480 | 14.1127 | 14.1148 | 14.7070 | 18.1269 | +4.20% |

当前 PF 对本轮 V3 的耗时降低依次为 34.77% / 21.17% / 26.54% / 18.87%。
10 月 1 日 V3 整轮为 6.4008 / 10.9488 / 9.1194 / 17.6598 ms；本轮 V3
双机 20K 波动较大，A/B 两轮分别 18.4377 / 17.8312 ms，均保留，不挑更慢
的一轮扩大 PF 优势。旧 PF 双机 20K 两轮为 14.1146 / 14.1159 ms。

等效 TFLOPS/卡：以 `18*T*K*H*I / wall_seconds / 1e12` 换算上述整轮
p50；只计有效 GEMM FLOPs，通信/准备/释放仍计入时间。工作量固定，不能
拿这些数值与变长 workload 的逐步吞吐分布直接作严格单变量消融。

| 拓扑 / T每卡 | 10/1 历史 PF | 旧 PF 本轮 | 当前 PF 本轮 | V3 本轮 |
|---|---:|---:|---:|---:|
| EP8 / 8192 | 157.6 | 158.0 | 147.4 | 96.1 |
| EP8 / 20480 | 178.3 | 179.0 | 177.8 | 140.2 |
| EP16 / 8192 | 95.2 | 95.3 | 91.5 | 67.2 |
| EP16 / 20480 | 109.6 | 109.5 | 105.1 | 85.3 |

### 历史独立阶段的回顾

10 月 1 日另有 native 阶段基准：固定形状 tensor 预热后复用，权重计时外
发布；BWD 从已保存 X/GU 开始。下面仅回顾旧数据，不是本轮阶段复测，
也不将它们相加冒充上面的真实循环。单位 ms：

| 拓扑 / T每卡 | 历史 PF FWD | 历史 PF cached BWD | 历史 V3 FWD | 历史 V3 cached BWD |
|---|---:|---:|---:|---:|
| EP8 / 8192 | 1.4973 | 1.7655 | 1.7239 | 2.1275 |
| EP8 / 20480 | 3.2100 | 4.6068 | 3.5702 | 5.0457 |
| EP16 / 8192 | 3.1679 | 2.8405 | 3.4346 | 2.8522 |
| EP16 / 20480 | 6.9941 | 6.6202 | 7.2390 | 7.0022 |

历史 native 与 M7e 的含临时 tensor 分配阶段基准也有边界差别，不能把所有
数字的差额都解释成 GPU kernel 回退。当前静态调度与 Return 的性能门禁仍
保持开放；本轮固定 workload 配对已经证明剩余差距不能全归因于动态路由。
未增加 kernel 消融，因此不将回退进一步精确分摊到某一个改动。

### 证据与复现

本轮产物在 `build/pf_fixed_workload/`：`run.py`、旧 teardown wrapper、
`{a,b}_{historical,current,v3}_ep8.log`、对应 `ep16_node{0,1}.log`、
`summarize.py`、含逐轮/逐 rank 原始样本的 `summary.json`。
24 组记录的形状、50/200 次数、rank-max 和 p50 均已核对，总计 4800 个
measured rank-max 样本。历史整轮值直接读取 `build/pf_training_timeline/`
的原始 JSON，而非仅转抄四舍五入后的表格。

复现入口：`/opt/.venv/bin/python build/pf_fixed_workload/run.py`，先确认两
节点均空闲且锁定 1620 MHz。本次未运行编译、autotune 或并行 GPU 作业。
结束后确认两节点均 0 MiB GPU 占用。没有新增生产代码、细粒度单测或恢复
已删除的优化，也没有把上一轮变长 Megatron 数字混进此固定负载对照。

## 2026-10-04：均衡路由 + 动态 tokens 对照

本轮只增加 benchmark 的 `--route-distribution balanced`，生产算子仍使用
M7e 的冻结产物 `build/pf_peer_bindings/`，未重新编译或修改 GEMM/通信。
相同长度序列下，均衡路由相对热点路由使 PF 整轮耗时降低 16%–24%；
这是 workload 的变化，不记作新的代码优化收益。

### 数据与计时口径

- 两节点 16 张 L20X 均锁定并采样确认 1620 MHz；EP8 是单机，EP16 是双机。
  H2048/I512/E64/top-k4、BF16；8K/20K 是每 rank 的 nominal 8192/20480。
- 每 rank 的 tokens 保留原来的 σ=10%、±3σ 截断正态，seed=123；不同 rank
  独立采样，因此全局总 tokens 的实测标准差只有均值的约 1.9%–3.7%。
  均衡和热点对照的每步长度、输入 RNG、权重、gate scores 一致。
- 均衡路由在每个 source 的完整 16-token 分组内，将 64 个专家随机排列后
  分给 16 个 top-k4 token；再打乱 token 顺序。每步重新生成排列，每个 token
  无重复专家，每个完整分组的专家计数完全相等。最后一个不完整分组使每个
  source 的专家计数最多差 1，全局最多差 EP；不靠容量截断或丢 token 做均衡。
  通用分组长度为 `E / gcd(E, top_k)`，不要求 top-k 整除 E。
- `route_bias_std=0.5` 仍用于生成相同的 gate scores；balanced 模式单独替换
  expert indices，因此不再有该 bias 导致的专家负载热点。默认 random 模式不变。
- 先完整 prime 80 步，再 20 warmup + 60 measured；每步取全 rank 最大
  `forward → backward → 释放 tape/gradients → synchronize` wall，报告 p50。
  固定通信缓冲、IPC、分配器/编译预热在计时外；本批 X/GU 缓存由计时内 FWD
  生成，BWD 使用它们，不重算 FC1。Notify、权重 repack、私有 tensor 准备仍计时。
- 各 workload 的 row-capacity 按自身 trace 最大接收量计算；均衡后容量自然
  更小。因此结果也包含生产容量 heuristic 对路由分布的响应，不称为固定布局
  下仅改变 GEMM 负载的微基准。所有均衡后端使用相同 trace 和相同容量参数。

均衡数据的实测分布（60 measured）：

| 拓扑 / nominal T | 全局 tokens 均值 | 全局 tokens 标准差 / 均值 | 最大 rank 接收量 / 平均值，p50 | 专家计数最大差值 |
|---|---:|---:|---:|---:|
| EP8 / 8192 | 65104.9 | 3.65% | 1.0001 | 7 |
| EP8 / 20480 | 164954.8 | 3.27% | 1.0001 | 6 |
| EP16 / 8192 | 130402.6 | 2.41% | 1.0002 | 11 |
| EP16 / 20480 | 329728.6 | 1.94% | 1.0001 | 10 |

四档的 rank 接收量最大不均衡均低于 0.04%。这只保证 destination 专家/owner
接收量均衡；source 的输入长度仍然随机，并未把所有 rank 的输入强制设为等长。

### PF 相邻 workload 对照

热点列为本轮重新测试，不拼接昨日较快结果；逐 rank SHA 与 M7e 原热点 trace
一致。表内 ms 是整轮 wall p50，TFLOPS 是每卡有效 GEMM 吞吐的逐步 p50。

| 拓扑 / nominal T | 热点 ms | 均衡 ms | 耗时降低 | 热点 TFLOPS/卡 | 均衡 TFLOPS/卡 |
|---|---:|---:|---:|---:|---:|
| EP8 / 8192 | 5.6221 | 4.7059 | 16.30% | 109.8 | 131.1 |
| EP8 / 20480 | 11.8036 | 9.7845 | 17.11% | 129.0 | 159.3 |
| EP16 / 8192 | 9.3997 | 7.5534 | 19.64% | 65.1 | 81.4 |
| EP16 / 20480 | 22.4231 | 17.0022 | 24.18% | 69.0 | 91.6 |

每步有效 FWD+BWD FLOPs 为 `18 * sum(tokens_per_rank) * top_k * H * I`，
一次 FMA 计 2 FLOPs；除以该步 rank-max 秒数、EP 和 1e12，再取 p50。
不计 padding、SwiGLU/dGate 标量操作的 FLOPs，但它们及通信/host 准备耗时
均保留。不是 GPU GEMM 峰值，也不额外乘专家总数。
若改为整个采样段的总 FLOPs / 总 wall，均衡 PF 四档为
130.1 / 158.7 / 81.2 / 91.2 TFLOPS/卡；不能混用这两种统计口径。

10 月 1 日相同 H/I/E/K 的固定均衡输入整轮为
3.925 / 8.673 / 6.498 / 14.113 ms；本轮均衡后仍未恢复这些数值。
旧输入/路由每轮不变，本轮长度、路由排列、输入每轮改变，且算子版本不同；
不能把剩余差值全部归因于某一种开销，也不能声称均衡已解决 Return 重构的
全部回退。固定缓存 BWD 相对 V3 的门禁仍以 M7e 阶段对照为准。

### 均衡 workload 的 Megatron 对照

真实 Megatron `moe_act`，融合 permutation/weighted SwiGLU 开启，DeepEP
使用上游默认 20 SM；配置、软件版本和算子边界与 M7e 完全一致。
所有正式采样的新增 compile graph 数均为零。每格为 ms / TFLOPS每卡：

| 拓扑 / nominal T | PF | Megatron NCCL | Megatron DeepEP | PF 对较快基线耗时降低 |
|---|---:|---:|---:|---:|
| EP8 / 8192 | 4.7059 / 131.1 | 5.3665 / 113.8 | 6.9056 / 89.2 | 12.31% |
| EP8 / 20480 | 9.7845 / 159.3 | 9.4153 / 165.5 | 12.8345 / 122.0 | **-3.92%** |
| EP16 / 8192 | 7.5534 / 81.4 | 11.9960 / 51.3 | 10.2397 / 59.8 | 26.23% |
| EP16 / 20480 | 17.0022 / 91.6 | 27.5443 / 56.6 | 20.6659 / 75.9 | 17.73% |

单机 20K 本轮 PF 比 NCCL 慢约 3.9%，不能宣称所有均衡训练点均领先。
热点 workload 下的 19%–28% 优势也不能直接推广到均衡 workload。
这仍是单个 MoE 算子的训练循环，不是完整模型/AReaL step。

冻结原版 V3 也完成相同均衡动态 trace，仍采用原 teardown wrapper；库与 M7e
记录的 SHA256 一致。原样应用调用的 ms / TFLOPS每卡依次为：

| 拓扑 / nominal T | 冻结 V3 原样动态调用 |
|---|---:|
| EP8 / 8192 | 81.0759 / 7.7 |
| EP8 / 20480 | 189.2607 / 8.3 |
| EP16 / 8192 | 109.0479 / 5.6 |
| EP16 / 20480 | 279.9448 / 5.6 |

该口径包含旧 V3 的动态 R 缓冲重分配/IPC 生命周期成本，均衡并不意味着
接收行数跨调用不变。这些数值不用于宣称 PF 的 GEMM/调度快了几十倍；
也不替代 M7e 的缓存就绪 native V3/PF 阶段对照。

### 验证与复现

复用现有 Torch reference，EP8/EP16 的 8K/20K 均衡变长数据各验证一步，
共 48 个 rank-case 的 output、dX、dGate、dW1、dW2 通过。CPU 检查分组
均衡、top-k 唯一性及 gate/长度不变；未新增细粒度单测或修改生产 kernel。
JSON 增加 route distribution 和逐步专家最小/最大计数，保留所有实际 T/R。

```bash
torchrun --standalone --nproc-per-node=8 tools/benchmark_training.py \
  --backend pf --tokens 8192 20480 --warmups 20 --samples 60 \
  --prime-trace --route-distribution balanced --token-std-fraction 0.1 \
  --ring-tokens 512 --ring-depth 8
```

通过 `WEFTMOE_TENSOR_EXTENSION` 指定 M7e 的 `_runtime*.so`；EP16 两端
启动，ring 改为 B768/D6，NCCL IB/interface/GID 参数与 M7e 一致。
证据在 `build/pf_balanced_workload/`：`balanced_parity_*`、`random_pf_*`、
`balanced_pf_*`、`balanced_megatron_{alltoall,flex}_*`，以及 `run.py`、
`summarize.py` 和含原始记录的 `summary.json`。
冻结 V3 为 `balanced_v3_*`。全部 20 组计时记录审计通过：均衡模式各后端
trace/容量一致，热点与均衡的长度序列一致，热点路由 hash 复现 M7e。
测量完成后确认两节点 GPU 均已释放，无残留作业。

## M7e 地址复用与公平训练对照：2026-10-03

简述：Backward peer bindings 不再随 FWD generation 重建；将“本 rank 是否换了
allocation”并入已有分配一致性检查，容量满足时复用地址和 IPC 映射。本轮相邻
消融中，1/64 token 整轮耗时降低 8.5%–9.6%，8K 降低 1.2%–2.5%；单机 20K
持平、双机 20K 降低约 1.4%。这是减少固定 host 准备开销，不是 GEMM 吞吐提升。
实现提交 `6626864`；随机 workload 的 Megatron `moe_act` 基线接入提交 `16b1e7e`。

### 改动与调用计数

`generation` 标识本次 retained FWD/对应 BWD 的数据，不是地址版本。实际 R 变化
仍更新本地 TMA 边界，但只在初次分配、扩容或布局变化时交换 peer exports。
任意 rank 更换 buffer，全体都进入慢路径；不能让本地容量足够的 rank 独自返回。
旧 imports 全部关闭并达成 collective agreement 后才释放旧 exports；OOM/retry、
close/suspend 的顺序不变。没有缓存 route 或计算结果，也不更改 GEMM、持久容量、
ring slot credit、NIC completion、dX→dW 先后约束和推理 kernel。

以五次不同 tokens/routes 的调用核对 CUDA/NCCL API 次数；EP8、EP16 的每个
rank 在第 2–5 次调用结果一致。以下是**整个 BWD API**的稳态计数：

| 版本 | AllGather | AllReduce | 显式 stream synchronize | IPC open/close |
|---|---:|---:|---:|---:|
| M7d | 1 | 3 | 4 | 0 / 0 |
| M7e | 0 | 2 | 2 | 0 / 0 |

两次剩余 AllReduce 分别是本次私有 tensor 准备、peer workspace 容量/分配检查。
若仅看 peer prepare，已从 2 AllReduce + 1 AllGather + 3 次同步降为
1 AllReduce + 1 次同步。初次调用仍导入本节点七个 peer。计数探针不用于测性能；
正式测量未加载探针。证据：`build/pf_peer_bindings/audit_fixed_*`。

### 相邻 M7d/M7e 消融

两机均为 8×L20X，1620 MHz；EP8 是**单机八卡**，EP16 为两机十六卡。
H2048/I512、E64/top-k4、BF16 activation/weight/gradient，gate FP32。
每 rank 的 token 数服从 nominal 附近 σ=10% 的截断正态，专家共享 bias σ=0.5；
所有版本逐 rank trace SHA、T/R 与容量完全匹配。20 warmup + 60 measured，
单位 ms，取每步全 rank 最大 `FWD+BWD+释放+device synchronize` wall 的 p50。
固定 runtime 初始化不计时，权重重排和每次私有 tensor 准备仍计时。

| 拓扑 | nominal T/rank | M7d | M7e | 耗时降低 |
|---|---:|---:|---:|---:|
| EP8 | 1 | 1.0962 | 1.0032 | 8.49% |
| EP8 | 64 | 1.4540 | 1.3235 | 8.98% |
| EP8 | 8192 | 5.7525 | 5.6089 | 2.50% |
| EP8 | 20480 | 11.9251 | 11.9474 | -0.19%，持平 |
| EP16 | 1 | 1.4970 | 1.3655 | 8.78% |
| EP16 | 64 | 2.0452 | 1.8482 | 9.63% |
| EP16 | 8192 | 9.6068 | 9.4949 | 1.16% |
| EP16 | 20480 | 22.8469 | 22.5344 | 1.37% |

M7d 对照来自 `build/pf_private_dr/network_dx/`；新库来自
`build/pf_peer_bindings/`，库 SHA256 为
`6579bd40dd8b230d23aaa1e7c08dbc886c5bb2fb823790486b62a38c7ce15285`。
正式原始消融为 `performance_{old,new}_ep*`。长文收益小，不据此宣称已经消除
M7d 相对更早 resident 的所有回退。

### 基线公平性约定

- Megatron 使用真实 TEGroupedMLP、selective `moe_act`：保存 routed X、FC1
  gate/up，在 BWD 重算 gated SwiGLU，不拿“不缓存 FC1、重算整个 MoE”的模式对比。
  输出及 dX/dGate/dW1/dW2 都包括；三者都从 selected routes/probabilities 开始，
  不计 router projection/top-k、optimizer 或整模型 DDP 通信。
- Megatron 开启 fused permutation、fused weighted SwiGLU，分别测试
  NCCL all-to-all 和 DeepEP Flex（上游默认 20 SM），不强制较慢的非融合/确定性模式。
  Megatron-Core 0.17.0、TE 2.14.1+27de67e1；两机 NCCL 日志确认 IB/GDRDMA。
- 普通 20 warmup 在小 batch 下仍可能漏掉 Megatron 的 0/1-row 编译分支：
  EP16 的 1-token 正式采样曾有部分 rank 新编译 3 个 graph；长文该计数为零。
  新增 `--prime-trace`：先完整运行同一 trace 一次，再进行原来的 20/60。
  恢复显式 input RNG 状态，保证正式 X/dY 与未 prime 的试验相同；记录正式
  采样期间的 compile graph 增量。它只预热编译/分配器，不复用 FWD/BWD 结果。
- 冻结 V3 使用 `build/pf_training_v3_reference/`，SHA256
  `6d349d29f7120e00ae1225ac9ea8e0fa1423e6f18423828b3650c3b3c49812e1`。
  不把当前 PF 的 non-ring 分支重命名为“原版 V3”。旧库只用原 teardown 合同的
  benchmark wrapper，算子和 profile 未修改。其动态 R 下的分配/IPC 成本必须
  如实计入应用调用，但不能把数量级 wall 差距解释为计算调度收益。
- 因而另测冻结原版 native V3 与新 PF：固定相同输入/路由/权重、预备通信地址、
  已缓存 X/GU 的 BWD；临时输出/scratch 用 warmed caching allocator，分配仍计时。
  Megatron cached-state 对照同样把前置 FWD 放在 BWD 计时外。独立阶段 p50
  不相加冒充 FWD+BWD 整轮耗时；异步 FWD 的 host API 边界也不冒充 GPU 阶段耗时。

### 完整 trace 预热后的实际 FWD+BWD 调用（合成随机 workload）

下表所有版本统一先 prime 80 步，再 20 warmup + 60 measured；输入及路由
trace 与上节完全一致。全部 Megatron 正式采样的新增 compile graph 数均为零。
单位 ms，仍是整轮 rank-max p50，不是独立 FWD/BWD p50 之和。

| 拓扑 | nominal T/rank | 新 PF | 冻结 V3 原样调用 | Megatron NCCL | Megatron DeepEP | PF 对较快 Megatron 耗时降低 |
|---|---:|---:|---:|---:|---:|---:|
| EP8 | 1 | 0.9931 | 11.7874 | 4.9589 | 5.4790 | 79.97% |
| EP8 | 64 | 1.3190 | 14.0204 | 5.6575 | 6.5020 | 76.69% |
| EP8 | 8192 | 5.6208 | 84.3958 | 6.9672 | 9.5305 | 19.33% |
| EP8 | 20480 | 11.7321 | 194.7498 | 15.3929 | 18.5574 | 23.78% |
| EP16 | 1 | 1.3556 | 11.8460 | 4.7571 | 5.7642 | 71.50% |
| EP16 | 64 | 1.8627 | 19.8008 | 5.7190 | 5.5488 | 66.43% |
| EP16 | 8192 | 9.3066 | 98.4278 | 18.5958 | 12.8639 | 27.65% |
| EP16 | 20480 | 22.5249 | 275.1253 | 44.7328 | 28.4435 | 20.81% |

不只采用更慢的通信后端作为基线：本轮单机 NCCL 更快，双机长文 DeepEP 更快。
完整预热明显影响基线：例如 EP8 8K NCCL 的首轮 17.859 ms 降至 6.967 ms；
不能继续用首轮数据宣称约三倍优势，也不将所有差额都归因于 JIT。
新 PF 同点为 5.609→5.621 ms。上表只说明这个模型形状、分布和 eager MoE
算子边界的收益，不是整个 Megatron/AReaL 训练 step 的加速比。

Torch 为 2.9.1+cu129，DeepEP 为 1.2.1+567632d。Torch 的 NCCL 编译版本接口
返回 2.27.5，但实际已加载库的 `ncclGetVersion` 和 NCCL 日志均为 **2.31.2**；
不能把编译版本当成运行库版本。benchmark 现分别记录这两个值及实际库路径。
原始日志：`steady_{new,v3,megatron_alltoall,megatron_flex}_ep*`；
`summarize.py` 检查逐 rank SHA、tokens、received rows、容量及 API 计数。

### 固定缓存的阶段对照：不能宣称全面超过 V3

相同 H/I/E/K、1620 MHz、固定 balanced routes/输入/权重，10 warmup + 100
measured。V3 native 为 `build/static_inference_v3_20260923/test_cuda_notify`，
SHA256 `cc0471af98d0fcbcfa52b9c9669ae2e20fb9b8077aa36e483a11a4be6331a699`。
PF/V3 开启 `WEFTMOE_BENCH_ALLOCATE_TENSORS=1`、`WEFTMOE_BENCH_SYNC_RANKS=1`；
Megatron 开启 `--allocate-tensors`。均使用 warmed allocator，每轮 dY clone、
输出/中间 tensor 分配计入 BWD；前置 FWD、通信缓冲初始化和上一轮梯度释放
在 BWD 计时外。固定输入消除了 V3 动态 R 下的 IPC 重建成本。
这是 PF/V3 native 与 Megatron framework 的阶段 wall，不冒充裸 GEMM 时间。

缓存 X/GU 的 BWD，单位 ms，rank-max p50：

| 拓扑 | T/rank | 原版 V3 | 新 PF | Megatron NCCL | Megatron DeepEP | PF 对 V3 耗时变化 |
|---|---:|---:|---:|---:|---:|---:|
| EP8 | 8192 | 2.1414 | 2.0335 | 3.7926 | 3.4347 | -5.04% |
| EP8 | 20480 | 5.0703 | 4.8108 | 4.7352 | 6.1891 | -5.12% |
| EP16 | 8192 | 2.8710 | 3.0928 | 5.6714 | 4.7674 | **+7.72%** |
| EP16 | 20480 | 7.0381 | 7.2016 | 12.9485 | 9.8988 | **+2.32%** |

单机 20K 的 BWD 也不是稳定领先 Megatron：本轮 PF 比 NCCL 基线慢约 1.6%，
按接近看待。双机 BWD 与原版 V3 的差距仍明确保留为待优化项。
缓存就绪的 native BWD 本来就绕过重新绑定；本次地址复用优化主要作用于
不同 generation 的实际训练调用，不能用应用层的大幅领先替代这项计算路径验收。

同一 native 产物的训练 FWD（含 input clone、tensor allocation、Notify、
retained-state main），单位 ms：

| 拓扑 | T/rank | 原版 V3 | 新 PF | 耗时降低 |
|---|---:|---:|---:|---:|
| EP8 | 8192 | 1.7261 | 1.5171 | 12.10% |
| EP8 | 20480 | 3.5878 | 3.3021 | 7.96% |
| EP16 | 8192 | 3.4485 | 3.2009 | 7.18% |
| EP16 | 20480 | 7.2243 | 7.0203 | 2.82% |

不把 FWD 的优势抵消后就宣布“BWD 已不低于 V3”。原始结果为
`native_{v3,new}_ep*_t{8192,20480}_{forward,backward}.{json,measure.log}`、
`cached_{alltoall,flex}_t*_ep*.log`。全部八组 Megatron cached-state 验证中，
`moe_act` 与不重算版本的四组梯度，以及重复 BWD 的四组梯度，均逐字节相同。

### 推理回归与验收

M7d→M7e 的 inference-only CUDA Graph，对每档先 50 warmup，再 200 samples，
每 sample 重放 10 次，每轮更换有效 routes/gates；初始化、capture 与输入准备
均不计时。单位 µs，rank-max p50：

| 拓扑 | T/rank | M7d | M7e |
|---|---:|---:|---:|
| EP8 | 1 | 35.301 | 35.267 |
| EP8 | 64 | 59.293 | 57.579 |
| EP8 | 8192 | 1528.099 | 1528.126 |
| EP8 | 20480 | 3611.398 | 3613.594 |
| EP16 | 1 | 234.878 | 234.786 |
| EP16 | 64 | 290.667 | 289.958 |
| EP16 | 8192 | 3055.397 | 3060.789 |
| EP16 | 20480 | 7016.504 | 7031.014 |

最大正向耗时变化为约 0.21%，未观察到实质推理回退。单机 64 的变化不归因于
本次 host 改动：被移除的 peer prepare 不在 inference graph 路径内。

本轮复用现有分布式集成测试，不增加内部函数名或指令布局的细粒度单测：

- EP8/EP16，1/64/8K/20K 随机不等长输入：output、dX、dGate、dW1、dW2
  对 Torch reference，全体通过；两种 Megatron dispatcher 同矩阵也通过。
- 初始 hint 仅 64 rows，热点迁移、零接收、三份 live tape、扩缩 actual R、
  单 rank tensor 分配失败后全体退出/重试、非法 route、权重原地刷新、释放/恢复。
- 真实 Megatron TP4+EP8/EP16 DDP，两次优化器更新及 suspend/resume；通信
  buffer 在共卡切换时可释放，恢复时重建映射。
- 另以 `max_tokens=40000`、B16/D1、T256 强制 bounded source-window 路径，
  EP8/EP16 检查小 ring 多次复用；补充 EP4 和 non-ring 回退。

证据均在 `build/pf_peer_bindings/`：`random_*`、`lifecycle_*`、
`megatron_adapter_*`、`parity_*`、`bounded_reuse_*`、`small_team_*`、
`dense_fallback_*`、`inference_{old,new}_*`。数据汇总及 launcher 也保留在该目录。

复现实际调用比较时，使用同一个 `tools/benchmark_training.py`：

```bash
torchrun --standalone --nproc-per-node=8 tools/benchmark_training.py \
  --backend pf --tokens 1 64 8192 20480 --warmups 20 --samples 60 \
  --prime-trace --ring-tokens 512 --ring-depth 8
```

PF 通过 `WEFTMOE_TENSOR_EXTENSION` 指定对应库目录下的 `_runtime*.so`；
Megatron 改为 `--backend megatron --dispatcher alltoall` 或 `flex`，不需要
WeftMoE extension。EP16 两端启动相同参数，将 ring 改为 B768/D6，并显式
统一 NCCL IB 开关/GID/interface。冻结 V3 使用原 teardown wrapper 加
`--backend v3`，不替换其 kernel/profile。

结论：地址生命周期修复和稳态应用性能验证完成，随机整轮领先 Megatron；
但**双机 cached-state BWD 仍未达到不低于 V3 的目标**，不能关闭这项性能门禁。

## M7d 容量优先：2026-10-03（已落地，仍有剩余回退）

简述：扩大 Backward source 容量，恢复可复用 dR 和跨机 RX 直接 fanout；
利用 canonical 行顺序恢复静态 dU→SwiGLU→dX 链，取消重复 gather 和逐行
位置预约；跨机在剩余预算内复用完整 dX，单机保留直接写回。最终相对 M7b，
EP8/EP16 的 8K/20K 整轮耗时下降 12.6%–18.3%，实测峰值增量为 84–428 MB。
相对 M7 前 resident，单机 20K 基本持平、双机 8K 差约 0.9%，但单机 8K/
双机 20K 仍差约 7%/5%，不按“全部恢复”关闭性能门禁。推理四档差异均小于
0.4%。阶段一提交 `f69fffb`；分步消融与最终完整矩阵如下。

性能优先于进一步压缩 workspace；用户允许先使用每卡约 500 MB 的增量。
本轮只改变 Backward 容量，Forward 的 B512/D8（EP8）和 B768/D6（EP16）、
推理 kernel、GEMM tile 和实际复用 slot 的 credit 释放条件不变；完整 source
范围不发生调用内复用，因此移除这一路径上多余的窗口 credit 操作。

先将 source contribution 窗口数加深，再单独尝试加宽窗口。加宽时必须同步
扩充 dX slot，保留一整个最坏窗口可以推进的容量；不能只扩 contribution，
让 Return packet 等待的后续行被旧 dX slot 卡住。实验复用原 FWD X payload，
只为额外 dX slot 分配补充区，未重新分配一整套 dX ring。

10 warmup/30 sample，同一随机 trace 的初筛（FWD+BWD 最慢 rank p50，ms）：

| source 窗口宽度 / 深度倍率 | EP8 8K / 20K | EP16 8K / 20K | 每卡新增主要 payload，EP8 / EP16 |
|---|---:|---:|---:|
| 1 / 1 | 6.923 / 14.998 | 11.868 / 27.091 | 0 / 0 MiB |
| 1 / 4 | 6.482 / 13.859 | 12.130 / 25.739 | 192 / 216 MiB |
| 2 / 1 | 6.441 / 13.877 | 12.251 / 25.631 | 192 / 264 MiB |
| 2 / 2 | 6.317 / 13.790 | 11.958 / 25.548 | 320 / 408 MiB |

上表产物为 `build/pf_private_dr/return_capacity_dx/`，所有配置的逐 rank
trace SHA、token/接收行数与容量匹配。另一个不含分段 dX 寻址的独立容量
候选 `return_capacity`，1/1→1/4 的单机 8K/20K 为 6.987/15.348→
6.434/13.691 ms；1/6 为 6.451/13.904 ms，没有进一步改善。
双机为 12.034/26.858→11.824/26.377 ms，1/6 为 11.787/25.908 ms。
这些是筛选数据，部分与 CPU 编译重叠，不替代无编译干扰的正式相邻对照。

当前判断：加深窗口能收回单机的一部分回退，双机改善有限；加宽窗口尚未
显示值得保留额外寻址逻辑的稳定优势。后续正式验证优先选择更简单的 1/4，
并独立消融 SwiGLU epilogue 额外写 window-major dAB，避免 dX N-pair 重复 gather。
此处尚不宣布已恢复 resident 性能。

### 第二步：dR 容量缓存与 RX 直接 fanout

在 source D×4 上恢复可 IPC 访问的 dR 容量缓存，替代而非叠加私有 dR。
NIC RX 接收者一次读取 payload，直接分发给节点内目标 dR，完成存储与通知
后释放 RX，不再由八张 GPU 分别轮询、pull、ACK。节点内 dY→dR 的 destination
pull 不变，Return 仍为有界 ring，所有本地 dX 之后才开始 dW 的约束不变。

`resident_dr_ring` 的 10/30 随机短测：EP8 为 6.304/14.044 ms，EP16 为
10.981/23.025 ms（8K/20K）。双机改善明显，但还未恢复改造前性能。
随机、热点/空接收/live tapes、OOM 重试、权重刷新、卸载切换和 EP4/非 ring
回退通过；EP8 rank0/7 与 EP16 rank0/15 的 memcheck 均为 0 errors。

相同两样本实际 T/R 的每卡最大 native 常驻区 + Torch 活跃同时峰值（MiB）：

| 配置 | EP8 8K / 20K | EP16 8K / 20K |
|---|---:|---:|
| M7b | 1471.33 / 2805.81 | 2316.89 / 4417.86 |
| 仅 source D×4 | 1663.51 / 2997.99 | 2533.10 / 4634.07 |
| D×4 + resident dR | 1667.26 / 3001.28 | 2536.03 / 4636.83 |

常驻 compute 区确实变大，但私有 dR 同时退出 Torch workspace，不能将
前者的增长误报成整轮峰值增长。相对 M7b 的 20K 同时峰值增加约
195.5/219.0 MiB（EP8/EP16）；并非驱动整进程峰值。实际容量仍由已有 hint
和按需增长决定，不能将这一随机样本的差值承诺为所有热点形状的固定上限。

未保留的额外 dAB 副本：SwiGLU epilogue 同时写 canonical 与 window-major
dAB，让 dX 直接 TMA 读取后者，planner 仅清零其 M 尾行。EP8 为 6.334/13.486 ms，
EP16 为 11.777/26.349 ms，没有形成足够稳定的收益；删除该副本和实验分支。
这也说明探针内的 gather span 不能直接等同于可从整轮减去的关键路径耗时。

### 第三步：恢复行的局部性，避免重复窗口聚行

在 resident dR 上，独立测试完整 M-block 的 dX N tiles、静态 owner 内
lookahead=4，均未形成稳定收益，双机 lookahead 还可能变慢；不作为最终优化。
warp 聚合行位置预约只能把单机 20K 的 13.848 ms 降至 13.526 ms，双机
23.065→23.122 ms，收益不足。

训练 tape 的 canonical rows 已按 expert/source/route 稳定排序。因此一个
(window, expert, source) 是连续区间，可以由首末行及前缀直接计算 dX 行位置
和 Return queue 位置，取消每行的 scatter 预约，而不是只合并 atomic。
在 `ordered_plan` 的同一 binary 内关闭/开启该 planner：B×1/D×4 的
EP8 为 6.455/13.727→6.186/13.672 ms，EP16 为
10.718/23.438→10.247/23.562 ms。加宽到 B×8/D×1 的 EP8 为
6.049/13.051 ms；B×4/D×1 的 EP16 为 10.449/23.081 ms。
这些仍为 10/30 初筛，不替代最终相邻对照。

下一步候选是在容量许可时让 contribution 覆盖 source 的完整 token 范围，
使 dX 与 canonical dU/dAB 使用完全相同的 M-block，恢复静态
dU→SwiGLU→dX 计算链并直接 TMA 读取 dAB，不再重复 gather。跨机 dX
payload 仍保持有界；Return packet 在 M-block 边界截断，避免旧 slot
等待一个包含未来 M-block 的 packet 而产生循环依赖。预算之外仍保留
source-window 有界路径，不以截断路由或预留最坏 dX payload 解决热点。

`canonical_network` 已通过 EP8/EP16 的随机 1/64/8K/20K 数值校验；10/30
初筛为 EP8 5.739/12.305 ms、EP16 10.273/22.849 ms。清理实验开关并
接入容量自动选择的 `capacity_final` 为 5.795/12.115、10.322/23.035 ms。
这是组合变化的收益，不将容量、链式调度、padding 和 gather 的收益简单相加。

重新消融 canonical dX 本地 TMA 落地再由通信 warp 转发：单机为
6.028/12.411 ms，未优于直接写回。后续还将完整 source 范围中已不发生
的窗口复用单独简化：route 直接寻址，不再取模；setup 已准入全部 source，
不再逐块轮询准入或逐 token 原子归还同一窗口 credit。跨调用的 join、
dX slot credit、NIC 双 credit 与 bounded source 路径保持不变。

### 20/60 无编译干扰的三候选相邻筛选

GPU 均为 1620 MHz，逐 rank trace SHA、T/R 和容量一致。单位 ms：

| 候选 | EP8 8K / 20K | EP16 8K / 20K |
|---|---:|---:|
| capacity_final：容量 + canonical 计算链 | 5.717 / 11.956 | 10.004 / 22.726 |
| full_source_credit：完整范围简化寻址/credit | 5.759 / 12.051 | 9.917 / 22.607 |
| batched_credit：再聚合 Return ready/retire 原子计数 | 5.738 / 11.827 | 9.969 / 22.890 |

前两者差异很小，不能把去除 source credit 单独算成显著收益；初筛中
24.384 ms 的双机 20K 回退也未在正式对照复现。批量原子计数没有稳定
优势，撤回该实验，暂以 `full_source_credit` 做完整验收。TMA 写回短测
双机为 10.272/23.430 ms，同样未形成稳定改善，已删除其额外本地搬运。

另一个独立容量实验 `dx_capacity` 将剩余预算中最多 64 MiB 补给跨机 dX
slot；source 与补充 dX 的合计增量仍不超过 448 MiB。它复用 FWD X 作为
第一段，只为尾段申请显存。EP16 随机 64/8K/20K 校验通过；20/60 为
EP8 5.753/11.937 ms、EP16 9.947/22.411 ms。双机 20K 仅改善约 0.9%，
8K 无改善，不保留额外分段寻址代码和 64 MiB 尾段。

### 完整验收进度

`full_source_credit` 的 EP8/EP16 随机 256/8K/20K、EP16 20K 热点/空接收/
迁移、多个 live tape、OOM 重试、权重刷新、EP4、非 ring 回退均通过。
另外用实际 T256、容量上限 40K、B16/D1 强制走超预算 bounded 路径，
单机/双机都通过。EP8 rank0/7、EP16 rank0/15 memcheck 均为 0 errors。

Megatron 集成测试暴露了测试初始化问题：原脚本仅设置 CUDA model-parallel
RNG，但 Router 在 CPU 初始化权重，导致每次运行及 replica 间的 router
初始参数不一致。旧脚本的一次 EP16 运行在 router/FC1 梯度阈值处失败，
不能计为通过。补上 CPU `torch.manual_seed(123)`，没有放宽误差阈值；
冻结 resident 和新版本在同一初始化下均通过真实 TP4+EP16 DDP 两次更新、
suspend/resume 与缓冲释放。原失败日志保留，后续日志使用 `*_seeded` 标签。

### 容量优先候选的正式结果

`full_source_credit`，1620 MHz，H2048/I512/E64/K4 BF16；20 warmup、60 个
每步变化的随机 token/route 样本。以下均为每步最慢 rank 的 FWD+BWD p50，
不是固定路由 CUDA Graph，也不把不同阶段的 p50 相加。每组相邻对照的
逐 rank trace SHA、T/R、max_tokens/max_rows capacity 全部一致。

| 规模 | M7b 回退版 | 容量优先版 | 耗时下降 |
|---|---:|---:|---:|
| 单机 EP8 8K | 6.858 ms | 5.785 ms | 15.6% |
| 单机 EP8 20K | 14.954 ms | 11.929 ms | 20.2% |
| 双机 EP16 8K | 11.706 ms | 9.912 ms | 15.3% |
| 双机 EP16 20K | 26.390 ms | 22.716 ms | 13.9% |

独立相邻 resident 对照如下。resident 是 M7a/M7b 前的 PF，不是 V3；本轮
不能将这个对照误称为 V3。仍有剩余回退，性能门禁不能按“全部恢复”关闭。

| 规模 | resident 整轮 | 容量优先版整轮 | 剩余差距 | resident / 新版 BWD |
|---|---:|---:|---:|---:|
| 单机 EP8 8K | 5.431 ms | 5.734 ms | +5.6% | 2.942 / 3.235 ms |
| 单机 EP8 20K | 11.758 ms | 11.942 ms | +1.6% | 6.586 / 6.706 ms |
| 双机 EP16 8K | 9.275 ms | 10.043 ms | +8.3% | 4.785 / 5.342 ms |
| 双机 EP16 20K | 21.488 ms | 22.778 ms | +6.0% | 11.140 / 12.384 ms |

相同两样本 T/R 的每卡最大 native 常驻区 + Torch 活跃同时峰值，MiB：

| 配置 | EP8 8K / 20K | EP16 8K / 20K |
|---|---:|---:|
| resident | 1785.45 / 3645.85 | 2725.03 / 5510.23 |
| M7b | 1471.33 / 2805.81 | 2316.89 / 4417.86 |
| 容量优先版 | 1551.59 / 3149.11 | 2396.99 / 4752.26 |

20K 相比 M7b 增加 343.30/334.40 MiB（均小于 500 MB），相比 resident
仍少 496.74/757.97 MiB。此口径不是驱动整进程峰值；allocator reserved
与 BWD 之后的驱动占用快照另列于原始 MEMORY_JSON，不能混用。

证据：`build/pf_private_dr/full_source_credit/` 下的 `paired_*`、
`performance_*`、`memory_*` 与 `source.patch`。首轮 inference 阶段中途
与另一个候选的 CPU 编译重叠，功能检查保留，但不把该次时间作为性能结论。

### 第四步：剩余预算内选择实际 P 的完整 dX（收敛为仅跨机）

`adaptive_dx` 只增加 host 侧容量选择，不增加设备 kernel：完整 source
contribution 加实际 `[P,H]` BF16 dX 能放进同一个“原容量 + 448 MiB”预算
时，复用现有静态主 kernel 的完整 dX/Return 分支，省掉三个 Return planner
launch 和输出 slot 复用。否则继续使用上一阶段的 canonical dX ring；source
范围本身也放不下时仍走 bounded source-window 路径。

P 直接来自 retained FWD metadata，不新增计数 kernel/同步；不是按平均路由
分配，也不申请最坏 P。完整 dX 属于本次私有 workspace，生命周期与 U/dAB
一致。持久通信布局只取决于池的 shape/capacity，**不随各 rank 的 dX 选择改变**，
所以混合模式或下一次路由变化不触发 IPC 重建。两种模式的 Return wire/route
标记相同，仍等所有本地 dX 完成后才开始 dW。

20/60 相邻对照，逐 rank trace/T/R/capacity 一致，ms：

| 配置 | EP8 8K / 20K | EP16 8K / 20K |
|---|---:|---:|
| full_source_credit | 5.705 / 12.054 | 10.317 / 22.844 |
| adaptive_dx | 5.754 / 11.967 | 9.599 / 22.720 |

单机和双机 20K 差异不到 1%，不宣称显著收益；双机 8K 改善约 7.0%，但
本轮对照值也比前几轮偏慢，继续用 resident/M7b 相邻对照核实最终效果。
已通过 EP8/EP16 随机 1/64/8K/20K、20K 热点迁移/空 rank、三份 live tape、
OOM 重试、权重刷新、EP4/非 ring 回退；actual T256/capacity 40K 的 bounded
路径通过。另用 T256/capacity 27K/B16/D1：热点 P 超过剩余 dX 预算而空 rank
仍选完整 dX，后续短 tape 又切回完整 dX，覆盖 rank 间混合及跨调用切换。
EP8 rank0/7、EP16 rank0/15 的 memcheck 全部 0 errors。固定 CPU/CUDA
种子的真实 Megatron TP4+EP8/EP16 DDP 两次更新及 suspend/resume 也通过。
进一步与 resident 相邻复测，8K/20K：单机 5.364/11.744→5.756/12.017 ms，
双机 9.426/21.319→9.778/23.173 ms。跨机 8K 的剩余差距收窄至约 3.7%，
但单机没有改善。单机 epilogue 已直接写 source，完整 dX 会额外落地和搬运，
因此最终 `network_dx` 仅在跨机考虑完整 dX，单机保持阶段一的直接写回。
这是 host 容量策略的收紧，不增加或修改设备 kernel。中止 prototype 后续
内存/推理流程，保留已完成的功能、framework、performance 原始记录；最终
版本重新构建并进行对应验收，不能将 prototype 的整套流程标记为通过。

### 最终选择版本：network_dx

相对阶段一只改变 host 的容量选择：仅跨机且完整 dX 放得下时使用这条路径。
最终产物再次通过单机/双机 1/64/8K/20K 数值检查，以及 capacity 27K 的
热点/空 rank/live tapes/模式切换和四个被检查 rank 的 0-error memcheck。
超预算 bounded、20K 极端热点和真实 Megatron 的完整验收使用上述
`full_source_credit`/`adaptive_dx`，最终没有修改这些设备计算/通信流程。

1620 MHz、同一随机 trace、20/60，与 resident 相邻对照，单位 ms：

| 规模 | resident 整轮 | 最终版整轮 | 耗时变化 | 最终版 FWD / BWD |
|---|---:|---:|---:|---:|
| 单机 EP8 8K | 5.393 | 5.768 | +7.0% | 2.474 / 3.267 |
| 单机 EP8 20K | 11.905 | 11.882 | -0.2% | 5.255 / 6.708 |
| 双机 EP16 8K | 9.460 | 9.549 | +0.9% | 4.628 / 5.041 |
| 双机 EP16 20K | 21.604 | 22.689 | +5.0% | 10.381 / 12.363 |

单机 20K 已基本持平，双机 8K 的差距明显收窄；单机 8K 和双机 20K 仍有
回退，不按“全部夺回”验收。源码快照基于 `f69fffb`，最终证据位于
`build/pf_private_dr/network_dx/`；前面的候选数据仅用于逐步解释，不拼接为
最终性能矩阵。

随后对冻结 M7b 做独立相邻复测，仍为同一 trace、1620 MHz、20/60，ms：

| 规模 | M7b 整轮 | 最终版整轮 | 耗时下降 |
|---|---:|---:|---:|
| 单机 EP8 8K | 6.806 | 5.948 | 12.6% |
| 单机 EP8 20K | 14.852 | 12.141 | 18.3% |
| 双机 EP16 8K | 11.572 | 9.731 | 15.9% |
| 双机 EP16 20K | 26.465 | 22.822 | 13.8% |

这一轮最终版比上一轮高约 0.6%–3.1%，两轮数据都保留，不挑选最低值。
`performance_*` 与 `paired_*` 的逐 rank trace SHA、T/R/capacity 已逐项核对。
这里的 resident/M7b 都是 PF 历史版本，本轮不是新的 V3 性能对照。

最终显存复测，逐 rank 实际 T/R 与 M7b 完全一致，仍使用同一“native 常驻区 +
Torch 活跃同时峰值”口径：

| 规模 | 最终峰值 MiB | 相对 M7b 增量 MiB | 增量 MB（十进制） |
|---|---:|---:|---:|
| 单机 EP8 8K | 1551.59 | 80.27 | 84.17 |
| 单机 EP8 20K | 3149.11 | 343.30 | 359.98 |
| 双机 EP16 8K | 2725.06 | 408.17 | 427.99 |
| 双机 EP16 20K | 4752.26 | 334.39 | 350.64 |

四个测量点均低于 500 MB 增量。双机 8K 的完整 dX 将峰值恢复到 resident
的水平；20K 的热点 rank 仍保留 ring，单机/双机 20K 仍比 resident 少约
497/758 MiB。这不是任意 shape/热点历史下的整进程占用上限；dR 缓存 slack
和 Torch allocator reserved 仍需单独考虑。

无编译干扰的 inference 相邻复测，50 warmup/200 sample，每次 graph 含 10 次
调用，每个 sample 改变路由；T 为固定 graph bucket，时间为最慢 rank p50，μs：

| T | 单机 EP8 resident → 最终版 | 双机 EP16 resident → 最终版 |
|---|---:|---:|
| 1 | 35.371 → 35.254 | 235.344 → 235.272 |
| 64 | 57.694 → 57.563 | 290.386 → 289.811 |
| 8192 | 1528.120 → 1527.634 | 3054.256 → 3049.850 |
| 20480 | 3611.304 → 3611.435 | 7041.037 → 7019.237 |

全部差异小于 0.4%，未见推理回退。训练 FWD 在上面的正式 resident 对照中
也基本一致；不将这些微小差异当作本轮 Backward 改造带来的 Forward 收益。
最终脚本 `build/pf_private_dr/network_dx_acceptance.log` 为 `FINAL_ALL_PASS`；
它表示所列验证执行通过，不表示“训练性能全部恢复”的目标已完成。

## M7c 回退定位：2026-10-03

本轮先固定 1620 MHz、编译 profile、随机 token/route trace 和 B/D，区分
私有 dR 与 Return ring 各自引入的成本，不以固定路由旧数据解释新基准。
诊断产物保留在 `build/pf_private_dr/phases/`、`task_probe/`；插桩数据只用于
定位，不作为正式吞吐数据，也不将不同阶段的 span 相加。

### 已确认的回退位置

现有阶段时间戳，3 warmup/9 samples，逐步取最慢 rank 后取 p50，单位 ms：

| 阶段/配置 | resident | 仅私有 dR | 私有 dR + Return ring |
|---|---:|---:|---:|
| EP8 20K，dU span | 4.874 | 5.234 | 4.638 |
| EP8 20K，dX span | 4.841 | 5.219 | 7.679 |
| EP8 20K，dW1 / dW2 span | 1.746 / 1.742 | 1.729 / 1.710 | 1.748 / 1.749 |
| EP16 20K，dU span | 6.716 | 8.171 | 9.601 |
| EP16 20K，dX span | 6.703 | 8.155 | 13.793 |
| EP16 20K，dW1 / dW2 span | 2.710 / 2.704 | 2.699 / 2.698 | 2.723 / 2.722 |

这些 span 包含依赖/credit 等待和穿插执行的工作，不是纯 GEMM 时间。
dW 基本未变；主要问题在私有 dR 的跨 GPU RX pull，以及窗口化 dX 的
重新聚行、更多 M 尾块和流水线等待。不能据此断言换一个 GEMM tile 即可恢复。
静态 BWD 的寄存器/共享内存仍为 REG168/SHARED2048，未发现 occupancy
档位变化能解释此次回退。

额外的临时任务探针把 cluster 的有任务执行和 NoTask 等待分开。在 rank0
的 EP8/EP16、8K/20K 随机样本中，gather 占 dX 任务时间约三分之一至四成；
部分 cluster 的 NoTask 累计达数毫秒。该等待既含依赖和 credit，也含最后
所有本地 dX 完成的等待，不能全部归因于静态队首阻塞。M128 重新按 window
切分确实增加了任务：例如 EP8 的一个 20K 样本从 455 个 canonical 块变成
683 个 dX 块（+50.1%）；同批其它样本增幅不同。这是实际路由几何，不是
统一的固定 padding 比例。

### 排除没有形成端到端收益的方向

以下为 10 warmup/30 samples 的短测，单位 ms；不与上面的诊断表或原
20/60 正式表直接相减。逐项源码 patch 和日志保存在同名 build 子目录。

| 候选 | EP8 8K / 20K | EP16 8K / 20K |
|---|---:|---:|
| reduce_k4：仅展开 K4 Reduce | 6.964 / 15.131 | 12.017 / 27.680 |
| reduce_cached：每 token 缓存广播后的指针 | 7.104 / 15.241 | 12.028 / 27.512 |
| rx_directory：接收端生成紧凑 fanout 目录，保留所有 ACK | 未测性能 | 12.155 / 26.126 |
| local_tma：dX 本地 TMA 落地后由通信 warp 转发 | 7.416 / 16.110 | 12.307 / 27.662 |
| static_lookahead：静态 owner 内前瞻 8 个候选 | 7.276 / 15.237 | 12.858 / 28.277 |
| gather_once：前瞻上改为一个 M-block 完整执行 N tiles | 7.113 / 15.139 | 12.525 / 26.776 |
| warp_gather：独立改为 warp-per-row 搬运和广播行索引 | 7.004 / 14.973 | 12.013 / 26.439 |
| return_geometry：仅 BWD 合并相邻窗口，B×2/D÷2，容量不变 | 7.039 / 15.330 | 12.202 / 28.175 |

Reduce 独立微基准明显变快，没有直接转化成融合流水线收益；RX 目录只有
20K 初测较好，8K 未改善；恢复本地 TMA 落地则更慢。因此不将这些候选混入
生产路径，也不把直接分散写回定为主要瓶颈。前瞻增加 ready/credit 探测，
双机更慢；摊薄 gather 能改善前瞻版本，却未稳定超过原实现。warp-per-row
只有小幅初测变化，仍不能据此宣称消除了 24%–30% 的回退。后续以无插桩
完整算子和相邻对照决定是否保留。

`return_geometry` 没有增加空间、也没有改 Forward：仅在偶数 D 时将 BWD
的逻辑窗口 B 翻倍、D 减半。它减少了尾块，却在双机 20K 更慢，说明单纯
减少 padding 并不够，较粗的消费 credit 也会加重流水线背压。该候选的
EP8/EP16 随机 8K/20K、EP16 小 ring 和 20K 热点/生命周期检查通过后撤回。

## M7：私有 dR 与有界 Return（功能完成，性能未验收）

保留提交为 `f17a39f`（私有实际 R dR）和 `69647e8`（有界 Return）。
20K 的每卡 native 常驻区 + Torch 活跃峰值，EP8 减少约 840 MiB，
EP16 减少约 1092 MiB；随机数值、极端路由、内存检查、真实 Megatron
DDP/卸载恢复通过。推理四档规模相差均不到 0.4%。但训练 FWD+BWD 相对
改造前已验收 resident 仍慢 24%–30%，**本轮不是性能已验收版本**；
正式相邻对照及测量口径见本节后半部分，不以早期动态 IPC 开销作性能基线。

目标：dR 随实际 R 进入调用者 allocator 的私有 workspace；固定通信入口不变，
不再为 dR 长期预留容量或导出 IPC。Return 进一步按 source window 循环复用
dX/contribution，保留完整 U/dAB 与全部本地 dX 之后的完整 dW。

早期 dR 原型通过 EP8 随机 8K/20K、EP16 随机 8K/20K，以及 B16/D1 的
热点迁移、空接收 rank、live tapes、inference graph/offload 验证。
这只是正确性阶段，**还不能替代已验收 resident 版本**：匹配编译配置的
20 warmup/60 sample 双机 cycle 从 9.34/21.69 ms 回退至 13.86/30.46 ms；
FWD 持平，回退来自 BWD。RX 多 slot、连续行静态划分和双缓冲预取的独立
实验暂未消除回退，不把它们记录为性能收益。完整证据在 `build/pf_private_dr/`。

随后将跨 GPU fanout 元数据筛选改为 warp 并行：10 warmup/30 sample 同一
trace 的 EP16 cycle 从 cooperative 原型的 14.03/31.90 ms 降到
11.18/24.64 ms（8K/20K，约改善 20.3%/22.8%），EP8 随机数值和 EP16
小 ring 生命周期测试通过；相邻 resident 对照为 9.75/21.67 ms，仍有回退。
M7a 提交是可验证的私有 dR 正确性里程碑，性能门禁仍留在 M7c，不视为最终验收。

最早一次构建遗漏了三项网络 profile 宏（32 lanes/Q8/return Q16），导致
FWD 也变慢；该组性能已排除。后续对照显式传入相同宏，不能将编译配置差异
归因于 dR 改造。最终保留实现、显存收益和性能结果见下文。

### M7b Return ring 开发消融（未关闭性能门禁）

相同 H2048/I512/E64/K4、1620 MHz、随机 workload、10 warmup/30 sample；
EP16 使用 B768/D6。下表为逐步最慢 rank cycle p50，单位 ms，不是纯 GEMM。

| 候选 | 改动 | 8K | 20K |
|---|---|---:|---:|
| resident 对照 | 改造前常驻 dR/完整 Return | 9.750 | 21.670 |
| M7a warp_fanout | 私有实际 R dR、warp 并行筛选 RX fanout | 11.182 | 24.643 |
| return_ring | window-major dX、source credit、窗口边界 packet | 16.104 | 29.800 |
| fair_reduce | 通信繁忙时也推进 Reduce，避免窗口 credit 饥饿 | 14.041 | 29.397 |
| input_credit | epilogue 发布行计数，代替逐行扫描 dAB ready | 13.962 | 28.783 |
| reuse_forward | 复用已退休 FWD X/S、gather 向量宽度编译期化 | 13.042 | 28.385 |
| window_order | dU 静态序列按最早消费窗口排序 | 12.638 | 27.395 |
| direct_return | 节点内 dX epilogue 直接写已准入的 source window | 12.369 | 27.222 |
| prefetch_return | Return TX/RX 搬运使用双缓冲预取 | 12.079 | 27.121 |
| dx_pairs | 有界窗口内静态拆分 dX 的 N pairs | 11.937 | 27.007 |

直接写回保留原 WGMMA 与 BF16 转换，只替换 epilogue 目的地址；本地不用先
写 dX ring 再搬运。远端行仍经 dX ring → TX → RX → source contribution，
所有权由原逐 slot credit 保护。节点内发布在整个 N 维和 cluster store 完成后，
dGate 与 ready 一同由 Return owner 处理。

单机也存在回退，不能只归因于 NIC。固定 window_order 候选，EP8 的
B512/D8 为 7.235/16.405 ms，B512/D32 为 6.867/14.265 ms，B1024/D8 为
6.849/14.776 ms；相邻 resident B512/D8 为 5.416/12.015 ms。增大窗口只缓解
部分等待并增加空间，暂不因此修改默认 B/D。不同 B/D 同时影响 FWD，不能
把整轮变化都算作 BWD 优化。

`direct_return` 已通过 EP8/EP16 随机 8K/20K 全部五项数值，EP16 B16/D1
小 ring，以及 20K 极端热点、空接收 rank、三份 live tape、OOM 重试、权重刷新
和图/卸载测试。这是该候选当时的结果，最终代码的独立验证见下文。
产物/逐步日志均位于 `build/pf_private_dr/<candidate>/`。

未保留的实验：

- 用 coordinator 并行导入其它 GPU 的 source head，替代计算侧逐个 peer
  读取。EP16 BWD 为 7.688/17.112 ms，对照直接写回的 7.691/17.069 ms，
  没有稳定收益；EP8 也没有稳定改善，删除该缓存与额外状态。
- 每个 source window 内再按 expert 分组 Return packet：12.205/29.190 ms，
  20K 相对 dx_pairs 回退约 8.1%，撤回额外 prefix/cursor 元数据与计数逻辑。

各保留候选的逐 rank trace SHA256、token/接收行数、容量、warmup/sample
已逐项对齐。`prefetch_return` 通过 EP8 rank0/7 与 EP16 rank0/15 的 memcheck，
四份报告均为 0 errors；不把它冒充后续 N-pair 版本的 sanitizer 结果。

### M7b 功能里程碑

最终代码（`build/pf_private_dr/final/`）已独立通过 EP8/EP16 随机不等长
256/8K/20K 的 FWD、dX、dGate、dW1、dW2；EP16 20K 全局热点迁移、空接收
rank、三份 live tape、OOM 重试、权重原地刷新，以及训练释放后切换四个 EP4
inference 组的图/卸载检查通过。补充 EP4 B16/D1 和 EP8 非 ring 回退通过。
最终版本 EP8 rank0/7、EP16 rank0/15 的 memcheck 全部为 0 errors。
既有 native CUDA 测试源完成编译，未增加细粒度内部实现测试。

这个提交（`69647e8`）完成私有实际尺寸 dR、有界 Return 的功能合同，
**不代表性能验收**。此前 resident 产物保持不动。

#### 相邻完整复测

相同编译 profile、1620 MHz、20 warmup/60 sample；每步随机 token 数及
路由，逐 rank trace SHA256/容量/接收行数匹配。整轮为逐步最慢 rank 的
FWD+BWD wall p50，单位 ms：

| 规模 | resident | M7b | 耗时变化 |
|---|---:|---:|---:|
| 单机 EP8，8K | 5.363 | 6.981 | +30.2% |
| 单机 EP8，20K | 11.686 | 14.790 | +26.6% |
| 双机 EP16，8K | 9.301 | 11.564 | +24.3% |
| 双机 EP16，20K | 21.424 | 26.529 | +23.8% |

FWD 持平，回退集中在 BWD。因此尚不能将有界化视作性能优化，也不以早期
包含逐调用 IPC 开关开销的旧产物充当 V3 对照，掩盖当前 resident 回退。
后续继续检查 canonical dU 跨 source 边界与窗口 Return 的依赖。

同一两样本随机 workload 的每卡最大观测空间（MiB）：

| 规模 | 常驻 compute：resident → M7b | 常驻区 + Torch 活跃峰值：resident → M7b |
|---|---:|---:|
| 单机 EP8，8K | 747.8 → 435.7 | 1785.5 → 1471.3 |
| 单机 EP8，20K | 1521.3 → 683.3 | 3645.9 → 2805.8 |
| 双机 EP16，8K | 997.9 → 591.0 | 2725.0 → 2316.9 |
| 双机 EP16，20K | 2137.2 → 1046.8 | 5510.2 → 4417.9 |

第二列只计 native compute；第三列计 native compute/GIN/notify 与 Torch
活跃分配的同时峰值，不含 CUDA context、allocator 空闲缓存，不是驱动整进程
峰值。dR 转为私有分配后的空间已包含，未只展示移出常驻区的表面收益。

推理采用 50 warmup/200 sample、graph 内 10 次重复，1/64/8K/20K 全部
EP8/EP16 相对 resident 的耗时差均小于 0.4%；未改推理 main kernel。
真实 Megatron TP4+EP8 与 TP4+EP16 的 DDP、两次 optimizer update、
suspend/resume 均通过，不只是 mock adapter。证据位于
`build/pf_private_dr/final/{performance,inference,memory,framework}*.log`。

#### M7b 后续调度消融：未保留

继续检查是否能通过提前计算、改变窗口内顺序消除回退。相同 10 warmup/
30 sample 随机 trace，下面仍为整轮最慢 rank p50，单位 ms；不能与上面
20/60 的正式表直接相减。EP8 为单机 B512/D8，EP16 为双机 B768/D6。

| 候选 | EP8 8K / 20K | EP16 8K / 20K |
|---|---:|---:|
| 已保留 dx_pairs | 7.020 / 15.488 | 11.937 / 27.007 |
| planned_output_checked：预计算每行 dX 写回指针 | 未测性能 | 12.239 / 28.596 |
| window_aligned：窗口内串联 dU→SwiGLU→dX，pull 同序 | 7.336 / 16.456 | 11.731 / 26.019 |
| window_ahead：窗口 dU 提前执行，dX 独立 N-pair 调度 | 7.206 / 15.802 | 12.424 / 24.982 |

- canonical dU 的 M128 可以跨 source 边界，早期窗口因而可能依赖上一个
  source 的尾部 dispatch。尝试将整个 activation chain 改为 window-major，
  dU/SwiGLU 后另存 canonical U/dAB 给完整 dW；同时按相同目录的 batch
  residue 做节点内 dR pull。双机略有收益，但单机回退约 4.5%/6.3%。
- 串联又使 dU 受到 source contribution credit 限制，故进一步让窗口 dU
  不受该 credit 限制，dX 独立等待 canonical dAB 后 gather。双机 20K 相对
  dx_pairs 改善约 7.5%，但双机 8K 回退约 4.1%，单机两档回退约 2.6%/2.0%。
  没有形成跨规模收益，撤回整组改动，未给生产路径增加按规模分支。
- 首次串联原型错误地用各自紧密行距重解释共享 staging，H 与 2I 不同时
  不同 cluster 的区域互相覆盖，数值测试失败。修正版统一物理行距为
  `max(H,2I)`，不是额外保留整批 dR/dAB 副本。失败原型没有性能结论。

`window_aligned` 通过 EP8/EP16 随机数值、双机极端热点/空接收/live tapes/
OOM 重试/权重刷新/卸载，以及四份 memcheck（0 errors）。`window_ahead`
通过相同数值和生命周期测试，但未做独立 memcheck，不借用前一候选结论。
全部调度实验和预计算写回指针均未保留；代码恢复到 `69647e8`。日志与产物
位于对应 `build/pf_private_dr/<candidate>/`，最后一个候选的补丁另存为
`window_ahead/source.patch`，不增加内部实现细节单测。

目前证据说明单纯调整 activation 任务顺序不足以同时恢复四档训练性能。
私有 dR 的 RX fanout/pull 和有界 Return 的 credit 等待仍需单独优化；
保持 M7c 性能门禁开放，不把显存收益或某一规模的改善等同于整体验收。

## 2026-10-02：通信 workspace 按容量常驻，阶段切换显式退休 IPC

本轮解决下节动态 workload 暴露的重分配瓶颈：BWD peer payload 从“实际
rows/routes 一变就重新分配”改为按容量 lazy 分配、驻留期复用；不修改 GEMM、
ring 调度或数值流程，也没有恢复已删除的 M。代码提交 `ceec6a0`。
随机 workload 的 8K/20K FWD+BWD，EP8 从 79.43/196.18 ms 降至
5.30/11.77 ms，EP16 从 111.15/253.51 ms 降至 9.40/21.21 ms；
同一测量段反复 IPC open/close 消失，共卡 suspend 仍能实际释放显存。

### 改动与边界

- BWD dR 初始容量使用调用者的 `max_rows` hint；source contribution/dGate
  按 `max_tokens * top_k` 预留。它们属于执行 slot，串行层/microbatch 共享，
  不是每层、每份 saved tape 各保留一份通信缓存。首次 Backward lazy init，
  后续实际 token/专家接收行数变化只更新逻辑尺寸与 TMA 描述符，不更换地址。
- dR 还要留到 dW2 使用，因此当前不是可立即覆盖的 ring slot；它与完整
  source contribution 是整次调用的 payload。TMA 仍按实际行数限定有效范围，
  不把预留容量误当成有效数据。超过 hint 的极端热点按几何增长安全扩容，
  不丢 token、不随短 batch 收缩。有限 heuristic 不保证任意热点绝不扩容。
- FWD 的 X/S/contribution ring、跨机 GIN 注册窗口本来已按 runtime 容量
  常驻，本轮不改成重复分配。训练保存的 X/GU、输出和梯度仍按各 tape/调用
  分配；不能因缓存通信 workspace 而延长 autograd tensor 生命周期。
- 新 plan 仍有分配确认、地址 AllGather 和 import 确认；只在 handle 真正
  改变时重建 IPC 映射。这不是依赖路由不变的跨调用快路径。
- 共卡 offload 改为两阶段退出：同步/检查无 live tape → EP barrier →
  清理内部推理图并关闭所有 FWD/BWD IPC imports → EP barrier → 销毁
  export、GIN/NCCL 与 runtime。外层图先由 SGLang 清理；Megatron suspend
  也清除 packed weights。恢复时新建驻留期，没有进程级永久 CUDA 缓存。

空间代价是缓存已预留/增长到的容量，而不是每步缩回实际尺寸。dR 和 source
contribution 的主要字节数为 `2 * H * (row_capacity + max_tokens * top_k)`，
另有 dGate、地址表等；不是把 `world * max_tokens * top_k` 的全局最坏
路由元数据上限直接当作每卡 dR 初始容量。Megatron 继续使用
`2 * max_tokens * top_k + local_experts * 128` 的 hint；基准则使用其整条
trace 的最大接收行数，二者都走相同容量复用机制。

### 复测条件与时间线

L20X，全部 1620 MHz，H2048/I512/E64/top-k4，BF16；50 warmup + 200
measured，默认 token σ=10%、共享专家 bias σ=0.5、seed=123。EP8 是第二台
单机，EP16 是双机，分别使用 B512/D8、B768/D6。时长口径仍为逐步全 rank
最大 FWD+BWD wall 的 p50；不含初始化、路由生成与 X/dY 准备，不含 optimizer。
下节固定 workload 数字不与本节混比。

产物为 `build/pf_resident_workspace/`，库 SHA256：
`65746dbf0be207da2405c753bbdbdc188d94fdd264753fee29848990774fd24b`。
编译参数与下节 matched_profile 完全一致，包含 32 network lanes、Q8/Q16
及 generated tuning；没有更改 GEMM profile。对照“此前 PF”为删 M 后的
`3f7b59a` matched_profile，V3 仍使用冻结 reference。旧 adapter 不含新的
retire API，基准临时入口仅恢复旧版 close 流程，未改其计时路径。

三份产物的逐 rank routes/gates trace SHA256、每步 token 数、接收行数、
warmup/sample 数和容量逐项完全一致。本轮配对结果（ms）：

| 拓扑 | nominal T/rank | 此前 PF | 常驻容量 PF | 冻结 V3 | 相对此前 PF 耗时降低 |
|---|---:|---:|---:|---:|---:|
| 单机 EP8 | 8192 | 79.433 | 5.299 | 83.975 | 93.33% |
| 单机 EP8 | 20480 | 196.177 | 11.772 | 190.076 | 94.00% |
| 双机 EP16 | 8192 | 111.153 | 9.396 | 120.045 | 91.55% |
| 双机 EP16 | 20480 | 253.508 | 21.213 | 282.990 | 91.63% |

相对本轮冻结 V3，耗时分别降低 93.69%、93.81%、92.17%、92.50%；V3
仍包含同类反复重分配成本，不将此表解读为纯 kernel 或调度效率对比。
新 PF 的 FWD API p50 为 2.447/5.250 ms（EP8）、4.648/10.368 ms（EP16），
此前 PF 对应 2.525/5.353、4.762/10.465 ms，未出现 FWD 回退。新 BWD
remainder p50 为 2.859/6.551、4.757/10.814 ms；它还含释放/末尾同步，
不是纯 kernel 时间，也不能将两个阶段的独立 p50 直接相加当 cycle。

正式日志：`{resident_pf,previous_pf,v3}_ep8.log` 和
`{resident_pf,previous_pf,v3}_ep16_node{0,1}.log`；复现入口为当前
`tools/benchmark_training.py`（旧产物用 build 下的 `benchmark_frozen.py`），
参数与下节相同。逐项 trace 对齐检查已通过。

Nsight 同样测单机 EP8、nominal 8K、5 warmup + 10 measured，合计 80 个
rank-step：此前 `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle` 各 560 次，
本轮二者以及 `cudaIpcGetMemHandle` 均为 **0 次**。BWD API 平均由
81.435 ms 降为 3.233 ms。测量段仍有 9 次其它 `cudaMalloc`，不能把
“通信 payload 无反复重分配/映射”说成整个训练循环零分配。

这个数量级的收益主要来自消除 host CUDA IPC 生命周期开销，**不是 GEMM
快了十几倍，也不是静态任务划分独有的收益**；V3 若同样改成容量复用，也能
消除这类开销。FWD/inference 的计算 kernel 本轮没有改动。

### Inference 前后对照

同一库/profile、1620 MHz；固定 T bucket，逐次测量前更新随机 routes/gates，
50 warmup + 200 measured。外层 CUDA Graph 串行包含 10 次算子，CUDA event
测量后除以 10，取全 rank 最大值的 p50；不含输入准备/捕获/退出开销，不能
与上面的动态训练 wall 或完整服务请求延迟混比。单机 T1/64 使用 DT 小容量池，
较长输入和全部双机输入走 PF。

| T/rank | EP8 此前 → 本轮（µs） | 变化 | EP16 此前 → 本轮（µs） | 变化 |
|---|---:|---:|---:|---:|
| 1 | 35.328 → 35.285 | -0.12% | 235.219 → 235.936 | +0.30% |
| 64 | 57.646 → 57.672 | +0.04% | 291.562 → 290.771 | -0.27% |
| 8192 | 1531.283 → 1531.485 | +0.01% | 3066.459 → 3057.989 | -0.28% |
| 20480 | 3623.270 → 3621.966 | -0.04% | 7031.128 → 7021.375 | -0.14% |

变化均不超过 0.31%，视为持平；各档 `compute_bytes/gin_bytes/notify_bytes`
也与此前完全一致，inference 不会分配新增的 BWD 常驻 payload。两阶段退休
只发生在关闭 runtime 时，没有给 graph replay 添加 collective。
脚本与日志：build 下的 `inference_benchmark.py`、`run_inference.py`、
`inference_perf_{resident_pf,previous_pf}_{ep8,ep16_node0,ep16_node1}.log`。

### 生命周期与框架兼容

EP8/EP16 已通过随机不等长 nominal T256/8192/20480 的 FWD、dX、dGate、dW1、
dW2 Torch reference，分别 48/96 个 rank-step；既有生命周期测试增加小容量
hint、热点迁移及短 batch，
验证溢出增长、增长后不缩容、多份 live tape、分配失败重试、原地更新权重。
释放测试在记录起点前先 `empty_cache()`，关闭后检查驱动可用显存实际增加
至少 pool 报告的 native 字节数，不只检查 Python 对象置空。

真实 Megatron TP4+EP8、TP4+EP16、DDP 梯度累积、两次 optimizer 更新与两轮
suspend/resume 通过，分别 16/32 个 rank-step。EP16 训练关闭后切换为四个
节点内 EP4 inference 组，也通过多层图回放及卸载。PF 的 EP8/EP16
inference 在 T1/64/8192/20480 通过 eager/graph、混合长短输入、空 source、
极端热点、动态路由与权重刷新，分别 240/480 项检查。

完整 Qwen3.5-35B-A3B 的 SGLang DP4+EP4 服务通过
三轮 KV/graph 释放、恢复和 greedy 32-token 输出一致性检查；模型权重在该
检查中保持驻留，因此这不是完整 RL step 或 AWEX 权重更新吞吐验收。

详细日志、配对运行脚本和 Nsight 产物保存在 `build/pf_resident_workspace/`。
分别为 `validation_ep8.log` / `validation_ep16_node{0,1}.log`、
`lifecycle_release_ep8.log` / `lifecycle_ep16_node{0,1}.log`、
`megatron_release_ep8.log` / `megatron_ep16_node{0,1}.log`、
`inference_ep8.log` / `inference_ep16_node{0,1}.log`、`dynamic_ipc_profile.*`，
整模型服务为 `sglang_server.log` / `sglang_offload.log`。
只扩充既有集成测试，没有新增检查符号名或私有布局的细碎单测。

## 2026-10-02：撤回 M，改用动态 workload 基准

以下是容量复用修复前的历史诊断；最新结果以上一节为准。

当时的复测结论：删除 M 的影响是固定负载下少量 host collective 开销；随机 workload
则暴露了原本就存在的 BWD 通信缓冲重分配/IPC 重映射瓶颈。旧固定负载的收益
可以复现，但不能据此声称动态训练循环已经稳定、显著领先 V3。详见本节配对表
和时间线诊断，后面的旧表仅作为历史记录。

M 的跨调用 peer-mapping 快路径及专用 changed-bit collective 已删除。保留 M
之前的本地同尺寸 allocation、未变 IPC handle 和同一不可变 plan 的复用；新
plan 均重新执行完整地址交换/确认。下面的优化表和 `eb371c2` 数据是历史记录，
不代表撤回后的性能。

旧 `benchmark_training.py` 在循环外构造固定 X、路由和 gates，循环内不更新
权重。因此历史“实际训练循环”准确含义是带分配/释放的固定路由 FWD+BWD
微基准，不是真实训练 step，不能直接外推到动态路由 workload。

新版基准每个 rank、每步独立采样 token 数：以 `--tokens` 为均值，默认标准差
为均值的 10%，在 ±3σ 内重采样并取正整数；每步重新生成随机 logits，加上
所有 source 共享的随机专家 bias，再取不重复的 top-k，选中 scores 做 softmax
得到 gates。默认 bias 标准差为 0.5，使专家/rank 负载可偏斜而非强制均匀。
`--seed`（默认 123）、`--token-std-fraction`、`--route-bias-std` 可调；
`--token-std-fraction 0` 仅固定 token 数，不固定路由。

路由 trace 预生成并保存在 CPU；计时外上传当前 routes/gates、生成新的 X/dY。
预先汇总整条 trace 的真实接收计数，PF/V3 使用相同的 max-token/max-row
容量，不再假设接收行数不超过均值的两倍，也不截断/丢弃热点路由。
FWD/BWD、权重 packing、tape/gradient 分配与释放及末尾同步仍在计时内。
固定专家权重、随机 logits 并不等于真实模型 router；基准没有 optimizer，
仍属于更接近动态负载的算子微基准，不称完整训练吞吐。

JSON 升为 schema 2：`nominal_tokens_per_rank` 取代旧固定形状字段；记录实际
逐 rank/逐步 token 数、接收行数、token 分布、容量、路由/gates trace SHA256
及按实际输入 token 总量计算的吞吐。PF/V3 对照必须使用相同分布、seed、
warmup/sample 数，并核对每 rank 的 trace SHA256；计时分布不能与旧固定
workload 表直接混比。恢复 GPU 后的数值与性能复测见下节。

使用修改后的 PF 必须重建 native 库及 TensorRuntime adapter，旧 `eb371c2`
产物仍含 M。单机 EP8 的基准参数示例（先设置对应的
`WEFTMOE_TENSOR_EXTENSION`，V3 使用其独立 profile 构建）：

```sh
torchrun --standalone --nproc-per-node=8 tools/benchmark_training.py \
  --backend pf --tokens 8192 20480 --warmups 50 --samples 200 \
  --seed 123 --token-std-fraction 0.1 --route-bias-std 0.5 \
  --ring-tokens 512 --ring-depth 8
```

CPU 侧已检查 EP8/EP16、8K/20K 的采样可复现性、top-k 唯一性、gates
归一化和接收行数守恒；另以模拟 runtime 检查两 backend 的 trace/容量一致
及 JSON 主流程。这些不是 GPU 数值或性能验证，未添加新的细粒度测试文件。
既有 16 个公共头文件独立编译和 4 项 runtime-settings 检查通过。
初次修改时 GPU 初始化失败，只完成编译/语法检查；GPU 恢复后已完成 native
库与 TensorRuntime adapter 的完整构建。代码提交 `3f7b59a`，与旧验收配置
对齐的新 PF 产物为 `build/pf_variable_workload/matched_profile/`，
`libweftmoe.so` SHA256 为
`3c9277873503e5f65e49fd9c878ed27956e0ea8858cf4367fb36577ed3e2917b`。

首次重建漏了旧验收的 network lanes/Q 参数，误用默认 2/Q32/Q32，导致双机
固定 8K/20K 循环达到 21.076/50.778 ms；这些 `new_pf_*_ep16_node*.log`
不是同配置对照，**排除出正式表**，中断了该候选的动态测试。随后核对旧构建
cache 中的完整命令，恢复 32/Q8/Q16 并重建、重测；不把此修正算作代码优化。
复现必须保留以下 NVCCFLAGS（仓库根目录执行；库与 adapter 均需重建）：

```sh
export NVCCFLAGS="-lineinfo -I$PWD/configs/static_inference_ring_20260924 \
  -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 \
  -DWEFTMOE_FWD_STATIC_V4_NETWORK_LANES=32 \
  -DWEFTMOE_FWD_STATIC_V4_PACKET_ROWS=8 \
  -DWEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS=16"
python build_native.py lib --output build/pf_variable_workload/matched_profile/libweftmoe.so --jobs 4
python tools/build_tensor_runtime.py --library build/pf_variable_workload/matched_profile/libweftmoe.so
```

### GPU 恢复后的正确性复测

L20X、1620 MHz，单机 EP8 / 双机 EP16；H2048/I512/E64/top-k4、BF16。
以同一随机路由生成器检查每 rank 不等长的 nominal T256/8192/20480，各两步，
输出、dX、dGate、dW1、dW2 全部通过 Torch reference（rtol 0.02、atol 0.00002）。
EP8 共 48、EP16 共 96 个 rank-step 检查通过；使用临时验证脚本，未新增
细粒度单测。既有 TensorRuntime 生命周期回归也在 EP8/EP16 通过，包含
非法路由、分配失败后重试、多份 live tape、热点迁移/零入站、payload 扩缩容、
权重原地更新及单机 EP8 子组的 inference/Graph。

日志位于 `build/pf_variable_workload/`，同配置复测为
`validation_matched_ep8.log`、`validation_matched_ep16_node{0,1}.log`、
`lifecycle_matched_ep8.log`、`lifecycle_matched_ep16_node{0,1}.log`。
此前不带 `matched` 的首轮数值检查也通过，正式性能仍须使用匹配配置。
双机 launcher 显式设置两端
`NCCL_IB_DISABLE=0`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`；
首次启动因继承的 IB 开关不一致而在 NCCL 初始化失败，保留于
`validation_ep16_node{0,1}_env_mismatch.log`，不计作算法或性能结果。

### 同配置的历史 / 固定 / 动态 workload 对照

所有正式循环测量均为 50 warmup + 200 measured，单位 ms，取逐步
**全 rank 最大 wall 的 p50**，包含 FWD/BWD、权重重排、准备/分配/释放和
末尾同步；路由生成与当前 X/dY 准备在计时外。EP8 在第二台单机测试，
EP16 独占两节点；PF 使用 B512/D8、B768/D6。新 PF 四档统一使用上述
`matched_profile` 库，不拼接未匹配构建的结果。

旧 PF 使用 `build/pf_backward_packed_storage/`（`eb371c2`），冻结 V3 使用
`build/pf_training_v3_reference/`，库 SHA256 分别为
`02aad150be56b69fbefe77531e21047f4064af875bec7eb55a56f770e90902f5`、
`6d349d29f7120e00ae1225ac9ea8e0fa1423e6f18423828b3650c3b3c49812e1`。
固定 workload 脚本取自 `git show eb371c2:tools/benchmark_training.py`，
保存在 build 目录的 `benchmark_fixed.py`；动态 workload 使用当前脚本。

先保持旧固定 workload 不变，只比较产物和删除 M 的影响：

| 拓扑 | T/rank | 历史旧 PF | 本轮旧 PF | 删 M 后 PF | 本轮冻结 V3 | 删 M 相对本轮旧 PF |
|---|---:|---:|---:|---:|---:|---:|
| EP8 | 8192 | 3.925 | 3.911 | 3.986 | 6.423 | +1.94% |
| EP8 | 20480 | 8.673 | 8.652 | 8.748 | 10.979 | +1.11% |
| EP16 | 8192 | 6.498 | 6.503 | 6.622 | 10.082 | +1.83% |
| EP16 | 20480 | 14.113 | 14.180 | 14.297 | 17.581 | +0.83% |

相同旧 PF 产物与历史记录相差均不足 0.5%。删除 M 后多出约 76/96 µs
（EP8）和 119/118 µs（EP16），没有出现 kernel 性能的数量级回退。
V3 wall 的波动更大：EP16 8K 本轮 p10/p50/p90 为
8.946/10.082/22.910 ms，不能把与历史 V3 的差值算成新 PF 的额外优化收益。

再换成默认动态 workload（token σ=10%、共享专家 bias σ=0.5）：

| 拓扑 | nominal T/rank | 旧 PF（含 M） | 删 M 后 PF | 冻结 V3 | 新 PF 相对 V3 耗时降低 |
|---|---:|---:|---:|---:|---:|
| EP8 | 8192 | 79.223 | 77.651 | 88.654 | 12.41% |
| EP8 | 20480 | 189.886 | 189.377 | 190.990 | 0.84% |
| EP16 | 8192 | 117.364 | 115.100 | 114.700 | -0.35% |
| EP16 | 20480 | 251.682 | 257.742 | 260.921 | 1.22% |

三份产物使用逐 rank SHA256 完全相同的 routes/gates trace，逐步 token 数、
接收行数、warmup 长度和容量也逐项一致。旧 PF 同样出现数量级变慢，故不能
归咎于删除 M。动态结果有显著长尾：新 PF 的 p10/p90 分别为
67.349/121.006、155.983/240.632、85.443/196.795、204.809/389.401 ms
（按表顺序）。本轮单机 8K 的 p50 有优势，其余三档接近持平；小百分比差值
不作为稳定收益，旧固定负载的约 20%–39% 领先不能直接外推到动态训练。

新 PF 的 FWD API / BWD remainder p50 分别为 2.512/75.234、5.305/183.938、
4.764/109.992、10.512/245.822 ms。增长主要在 BWD 准备阶段；这里的 FWD
是 host API 边界，BWD remainder 还含释放和末尾同步，不是纯 GPU kernel
时长，也不是两个独立 p50 相加得到 cycle。

正式日志：EP8 新 PF 为 `matched_new_pf_{fixed,dynamic}_ep8.log`，
旧 PF/V3 为 `{old_pf,v3}_{fixed,dynamic}_ep8.log`；EP16 三份产物均为
`matched_{new_pf,old_pf,v3}_{fixed,dynamic}_ep16_node{0,1}.log`。
首轮不带 `matched` 的新 PF 性能日志保留审计，不用于本节的最终表。
全部测试结束后，两节点 16 卡均为 1620 MHz、无残留 GPU 作业。

动态 trace 的分布如下；均值/范围和 skew 仅统计 200 measured，容量覆盖
warmup + measured。`skew` 是每步最大接收行数 / 全 rank 平均接收行数。

| 拓扑 | nominal T | 实际 T 均值 / 范围 | skew p50 / max | max_tokens / max_rows 容量 |
|---|---:|---:|---:|---:|
| EP8 | 8192 | 8171.85 / 5781–10578 | 1.523 / 2.645 | 10578 / 91874 |
| EP8 | 20480 | 20532.98 / 14370–26289 | 1.532 / 2.275 | 26289 / 190159 |
| EP16 | 8192 | 8179.44 / 5781–10578 | 2.001 / 3.595 | 10578 / 121167 |
| EP16 | 20480 | 20533.04 / 14370–26353 | 1.966 / 3.348 | 26353 / 278127 |

每档 199 对相邻 measured 调用中，全 rank 的实际 rows/routes 同时不变的
次数均为 **0**。旧 M 的跨调用跳过交换条件在这些相邻调用上没有命中。
这也说明仅固定 nominal T 不能代表实际专家接收形状稳定。

### 动态负载暴露的主要瓶颈：BWD IPC 映射生命周期

对旧 PF（含 M、与旧验收完全相同的产物）另做单机 EP8、nominal 8K 的
Nsight Systems 诊断：5 warmup + 10 measured，默认随机分布，NVTX 标记
FWD/BWD API，仅采集测量段。下表是 8 ranks × 10 步的**每 rank-step 平均值**，
不是正式 200 步的 rank-max p50；插桩结果不与正式性能表混用。

| BWD 内部开销 | 每 rank-step 平均（ms） |
|---|---:|
| BWD API 整段 | 81.435 |
| `cudaIpcOpenMemHandle`：7 次 | 56.687 |
| `cudaIpcCloseMemHandle`：7 次 | 13.141 |
| `cudaMalloc` | 2.093 |
| `cudaFree` | 1.003 |
| unified BWD 主 kernel（device 时间） | 2.454 |

IPC open/close 共 560/560 次，占 BWD API 整段约 85.7%。其余还有集体确认、
stream/event 等待和调用准备；GPU 时间与同步 API 耗时可能重叠，不将表中
项目简单相加作为独立分解。

原因：`BackwardPeerWorkspace` 只按**精确 rows/routes 相等**复用 allocation。
随机 tokens/路由使实际接收行数变化，每个新 plan 重新 cudaMalloc、交换
export，并在本机打开 7 份新 peer 映射、关闭旧映射，然后释放旧 payload。
固定 workload 即使删除 M，仍可复用原 allocation/handle，不会支付这笔
映射开销；M 只是少做地址交换/确认，删除它并没有引入该分配策略。
因此动态循环的数量级变慢不能解释为 GEMM 本身退化，也不能用恢复 M 解决。

证据保存在 `dynamic_ipc_profile.nsys-rep`、`dynamic_ipc_profile.sqlite` 和
`profile_dynamic.log`（同上 build 目录）。下一步应优先设计可支持逻辑 shape
变化的有界通信存储/映射复用，明确扩缩容与旧映射退休；不是重新引入全 rank
shape 相等快路径，也不直接预留所有 rank 的最大热点容量。本轮仅验证与定位，
未把这项新的存储优化混入删除 M 的对照。

## 优化总览（2026-10-01）

这里汇总本文件中的 PF 训练优化：做了什么、测到了什么收益、最终是否保留。
下文仍按时间保存完整实验，历史段落中的“待验证”以最后的验收结论为准。
更早的推理优化见[单机记录](static_forward_v4_optimize_record.md)、
[ring 记录](static_forward_v4_ring_record.md)；decode 独立见
[DT 记录](static_forward_v4_decode_record.md)，不混入本表的训练收益。

### 历史固定 workload 结果：相对冻结 V3

最终代码 `eb371c2`，L20X、1620 MHz、H2048/I512/E64/top-k4、BF16；
8K/20K 均为每 rank tokens。EP8 是单机，EP16 是双机。
下表百分比都是耗时降低 `(V3 - PF) / V3`，不是吞吐增幅。

| 拓扑 | tokens/rank | FWD 耗时降低 | BWD 耗时降低 | 实际训练循环：V3 → PF（ms） | 循环耗时降低 |
|---|---:|---:|---:|---:|---:|
| EP8 单机 | 8192 | 13.14% | 17.01% | 6.401 → 3.925 | 38.68% |
| EP8 单机 | 20480 | 10.09% | 8.70% | 10.949 → 8.673 | 20.78% |
| EP16 双机 | 8192 | 7.76% | 0.41% | 9.119 → 6.498 | 28.74% |
| EP16 双机 | 20480 | 3.38% | 5.46% | 17.660 → 14.113 | 20.09% |

FWD 包含 mini Notify 和 main；native 计时外准备权重，BWD 使用已保存的 X/GU。
实际循环另行实测，包含每步权重重排、分配/回收及同步，不是两个 native p50 相加。
EP16/8K BWD 应视为接近持平、本轮小幅领先。循环的大幅改善包含通用的
host 分配优化，不能全部归因于静态调度；这些 host 优化移植到 V3 也可能获益。
采样、基线及正确性边界见文末“最终统一验收”。

### 逐项优化与收益

除特别注明外，`a% / b%` 按 8K / 20K 排列；收益来自各自当时的相邻版本
消融，不能相加，也不代表其它模型的固定增幅。`main` 仅指 GPU 主 kernel，
FWD/BWD wall 与实际训练循环是不同口径；未独立量化的项目明确保留这一限制。

先解决训练缓存和 checkpoint 的额外开销：

| 优化 | 简单描述 | 实测收益 / 取舍 |
|---|---|---|
| M6a–f：实际容量与独立 tape | SavedX/GU 按实际 R 保存；BWD 私有 workspace 按精确 padded P 分配，解耦通信池与多份 live tape。 | 消除按最坏容量预留的浪费；T64 示例精确 P 比旧 padding 上界再省 903,672 bytes/rank。迁出通信池本身不等于等量净省显存。 |
| M6g：BWD 独立 sender 配额 | 修正 BWD 错用 FWD 配额，恢复 unified BWD 和 16 个 sender。 | 20K BWD 相对错误的 split 配置，EP8/EP16 降低约 13.8%/52.8%；这是恢复到 V3 附近，不是新算法领先 V3。 |
| M6h：G/U 合并写回 | 复用 FC1 epilogue 的 shared staging，以 16B vector 保存 G/U，替代分散的小 store。 | EP8 main 降低约 0.4%–1.2% / 1.5%–1.6%；EP16 约 0.8% / 0.7%。无新增 pool。 |
| ES/ET/FJ2：共享推理改造 | ES 合并本地 NIC 完成 prefix；ET 将 Return Q32 改为 Q16；FJ2 使用三级 FC1，并补齐跨 GEMM 的 shared-memory 退休同步。 | ES/ET 训练持平，收益在协议简化、缓冲减少及推理；FJ2 修复内存复用安全性，但 EP16 20K main 回退约 0.3%–0.7%，不计为训练提速。 |
| FM：计算侧保存 X | SavedX 从已 ready 的 FC1 A stage 写出，移除 Pull 中的第二次写回。 | EP8 20K FWD wall 约降低 2.7%，8K 持平；EP16 降低 1.98% / 3.31%。总复制字节数不变。 |
| FO：SavedX 地址计算外提 | 每个 task 只算一次目的地址，不在每次 K 迭代重复加载 row 和算地址。 | 首轮 EP8 main 降低 4.49% / 2.82%；EP16 1.51% / 1.40%。双机 20K 反向复测仅 main 0.47%、wall 持平，收益不固定。 |
| FR3：单机连续 X 的 TMA 保存 | 单机使用稳定物理行序和 M32 TMA；多机保留 vector scatter，避免强行统一布局。 | EP8 FWD wall 降低 3.61% / 5.19%；多机持平，撤掉通用 FR 在多机上的退化。 |
| FS2：去掉 planner 后的 host 回传 | 分配在 planner 前完成，planner → main 在同一 stream 连续执行。 | EP8 长文 FWD wall 正反向约降低 1.0%–1.1% / 1.5%–1.8%。历史独立 planner 后来由 mini Notify + main 内规划替代。 |
| FT/FZ/GA：写回与 credit | FT 用 TMA 写完整 S tile；FZ 将 SavedX 轮转分给 FC1 pair、两 CTA 各存半块；GA 合并连续 consumed-prefix ACK，保留远端 credit 和本地 NIC 完成检查。 | FT 对训练持平；GA+FZ 相对 FT，EP8 FWD wall 降低 0.74% / 0.94%，EP16 同容量长档降低 0.20%–0.86%。GA 仅加 4096 bytes/rank 元数据；当时仍未整体超过 V3。 |

再改前置规划、静态 BWD 和每次调用的准备开销：

| 优化 | 简单描述 | 实测收益 / 取舍 |
|---|---|---|
| Mini Notify + main 内 planner | 前置 kernel 只交换计数/分配信息；物理 ring、route 和 packet 计划在 main 内生成，训练/推理共用 main 实现。 | 当轮 EP8 Notify 从约 323/362 µs 降到 123/131 µs；累计 FWD wall 比同期 V3 低 7.03% / 1.23%，不是单项独立收益。 |
| 静态 BWD + counts 构造交错 dispatch | 长文由一个 cluster 完成整块 dU→SwiGLU/dGate→dX；欠填充时采用静态细粒度分配。按 32 行交错专家，避免发送集中打同一卡。 | 加入 stripe 后 EP8 BWD 从整块但不交错的 2290/6145 µs 降到 1762/4693 µs；相对当轮 V3 低 17.17% / 7.38%。消除了首版静态调度的长文回退。 |
| dW 加权静态区间 | 按实际 padded K 加权切分联合 dW2/dW1 流；本地全部 dX 完成后才启动 dW，并与 Return/Reduce 重叠。 | 改善专家/尾部负载不均；EP16 8K BWD 2962 → 2928 µs，EP8 20K 4731 → 4644 µs，但 EP8 8K 略退。无 split-K，不改归约顺序。 |
| A–D：planner 并行化与访存 | 少排冗余 ordinal 位；packet 构建与目录 Put 重叠；全网格分摊 RX 描述符；按 source-route 合并元数据写入。 | EP16 main 规划 394/556 → 336/500 µs，FWD wall 约降低 1.75% / 1.45%。A 单独只省约 2 µs，无独立 wall 收益。 |
| F：永久 header/gate 提前写入 | 与 payload 无关的 tape 元数据交给并行 planner，移出串行 Pull issuer。 | FWD wall：EP8 降低 5.31% / 6.21%，EP16 降低 6.19% / 9.68%。不提前发布未 ready 的 X。 |
| B/D 窗口消融 | EP8 保留 B512/D8；EP16 从早期 B512/D8 经 B1024/D4，最终选择 B768/D6。 | 最后一步相对 1024/4，EP16 FWD 降低 1.98% / 1.37%；pool 少约 74.8/74.7 MB/rank，但注册部分多约 8.5 MB，不包含独立 tape。 |
| I/J：流式 Reduce + 整链描述符 | 通信空闲时非阻塞归约已齐的 token；整块计算链只传递一次静态描述符，少两次 DSM mailbox 往返。 | 组合后 EP8 BWD 降低 2.51% / 1.27%。I 单独在 EP8 20K 退化，不能把组合收益拆成两项通用加速。 |
| K：合并分配确认 | 输入/权重/tape 在同一次 collective preparation 中准备，已绑定 tensors 不再重复确认。 | FWD wall：EP8 降低 2.83% / 2.40%，EP16 降低 2.39% / 1.42%；device GEMM 不变。 |
| M：复用 BWD peer mapping（已撤回） | 所有 rank 的实际 shape 均不变时跳过地址/映射交换确认。2026-10-02 删除该快路径及 changed-bit 协议。 | 历史固定 workload 下这部分 host collective 从 3 次减到 1 次；缺少真实动态路由命中率与稳定独立耗时收益，不再采用。 |
| Y2：BWD 寄存器配额 | 本模型 math/control 从 216/64 调到 208/88，给通信/producer 更多寄存器；FWD 不变。 | GIN 栈从 208B 降到 72B，EP16 BWD 降低 0.91% / 0.96%，单机无回退；不是所有模型的通用默认值。 |
| AC：复用启动准备 | 缓存同一启动配置的属性/occupancy 检查，不再为未使用的 split kernel 设置属性。 | EP16 BWD 2840.62/6615.60 → 2832.94/6616.67 µs，8K 小幅改善、20K 持平；属于 host 提交优化。 |
| AD：条件独占 GIN 提交 | 实际 context 足够且 CTA 无别名时使用 THREAD 模式，否则保留 GPU sharing。 | EP16 BWD 2832.94/6616.67 → 2824.17/6609.28 µs，仅约 0.31% / 0.11%；不删除任何 credit/完成检查。 |
| AF：复用权重重排 scratch | packed W1 改由应用缓存分配器提供，避免每步原生 cudaMalloc/cudaFree；每步仍重排当前权重。 | EP16 实际循环从上一轮 AD 的 9.668/19.258 ms 降到 6.498/14.113 ms（32.78% / 26.72%）。主要是分配/释放收益，不是 GEMM 加速；完整最终 V3 配对见首表。 |

### 未采用的消融，以及权重重排的成本

以下候选未证明稳定的整体收益，已撤回或未选用；详细数值见时间记录：

- 训练保存：Pull 双 lane 写 X、G/U TMA、通用网络连续行/TMA、仅轮转不分半、
  streaming store 等；单个规模获益不足以支持通用采用。首次 main 融合 FS
  退化，先选 FS2，后来才以 mini Notify 的新边界完成 main 内规划。
- 规划/调度：节点原子合并、按 source 局部性重排、就绪探测重排、receiver-only
  Reduce、owner 原地等待，以及 K128、M128 dispatch stripe、额外预取等，
  均未稳定缩短总耗时；某个阶段变短不等于端到端更快。
- 通信/资源：提前关闭/flush、SIMT 替代本地 TMA、接收优先、增加 Return ports、
  增减 sender、200/104 配额、chunk32/128 等未优于最终组合；保留 2 ports、
  16 senders、BWD chunk64、208/88。诊断插桩本身不计性能收益。

补充 2026-10-01 的独立权重布局诊断：当前 W1 由分离的 gate/up 改排为每 8 行
gate、8 行 up 交错，使 FC1 可用同一 packed GEMM 计算二者。1620 MHz 下，
每卡 8/4 个专家的 32/16 MiB W1，生产重排 kernel 冷缓存 p50 为
96.224/52.192 µs（不含分配）。单 GPU 回放 EP8/EP16 的均衡每卡工作量，
不重排的 separated FC1 比 packed FC1 慢约 22%–25%；但 EP8/8K 将每步
重排也计入后，粗算未必划算。该实验不含通信/checkpoint，独立 p50 相加
不是完整 PF 耗时；尚未实测完整“不重排 PF”，也未实现跨训练调用的权重值缓存。
原始数据：`build/pf_weight_layout_20261001/measurement_summary.json`。

## 记录说明与当前状态

对应设计文档 M6，按时间保留改造、失败候选和验证记录。
截至 2026-10-01，PF 训练已使用 count-only mini Notify、main 内 planner、
按实际 R/P 分配的独立 tape，以及静态 Backward；非 ring 路径保留 V3 对照。
当日验收实现为 `eb371c2`：包含静态调度、peer mapping/launch 准备复用、Y2 的
208/88 Backward 寄存器配额、AD 条件独占 GIN，以及 AF 的应用侧权重 packing
scratch。Forward 寄存器配额不变。M 后续撤回及基准口径修订见本文开头。

1620 MHz、EP8 单机/EP16 双机、8K/20K 的 FWD、BWD、实际训练循环共十二项
最终配对均低于冻结 V3，完整数据见文末「最终统一验收」。双机 8K BWD 仅
小幅领先 0.41%，不能称显著加速，也不保证换机器/模型后仍有同样优势。
验收依据是此次完整训练循环，不借用更早 GA+FZ 或单个 GPU kernel 的收益。
完整数值/生命周期检查和已有专项 memcheck 已通过；旧 Backward racecheck
的未决报告及工具版本限制仍保留，不能称完整训练 sanitizer clean。

## M6a：按实际接收量分配 rank-private Backward 中间区

先解开 Backward 对 transport pool 全容量激活区的依赖。U/dAB/dX/dGate
中间数据均只由本 GPU 读写，不需要 CUDA IPC 或 GIN registration；远端需要的
compact dR、source contribution、通知和 canonical directory 仍维持原地址。

本步将这四个 private plane 合并为一次独立分配，由 `BackwardTensorStorage`
及对应 handle 持有到完成。分配发生在 prepare 阶段，不在 persistent kernel
内进行；支持调用方 allocator。TMA maps 跟随这次分配，不留在可复用的 pool
slot 中。删除原 pool 的四个 payload 区及五个私有 TensorMap，避免双重预留。

行容量取 materialized Notify 已知的本 rank 实际 `R` 加
`owned_experts × (TileM−1)`。这是按实际 R 的安全 padding 上界，**不是精确
P_actual**；不下载路由或再运行一次 count exchange。每 rank 的余量只与本地
expert 数有关，不用 global max(R) 让所有 rank 申请同样大小。后续 training
plan 可进一步提供精确 P。本步未改变 canonical 行号、dW 归约顺序或 GEMM。

H2048/I512 下，四个 plane 每容量行共7172字节，另有最多三个对齐间隙。
`BackwardTensorStorage::bytes()` 计入此分配；pool usage 相应下降，但不能只
报告 pool 减少而忽略单次 Backward workspace。多个 live Forward tapes 不会
提前占用这些 Backward 临时区。

迁移前参考：DC inference 配置的 EP8 T64 通过现有 tensor-runtime 的
output/dX/dGate/dW1/dW2、multilayer graph、offload 检查（各8 ranks）；同配置
EP16 在 Forward launch 前拒绝，原因是其 `RDMA_CTAS=0` 编译关闭了旧多机
训练入口，不是梯度数值失败。后续训练候选使用冻结 V3 multi-node profile
`configs/static_inference_v3_20260923/` 建立参考，不把改变 profile 当作改造收益。
日志 `build/m6_baseline_tensor_ep8_t64.log`、
`build/m6_baseline_tensor_ep16_t64_{local,remote}.log`。

冻结 V3 native 在 EP16 T64/E32/K6 的 Backward 双次运行通过16 ranks
`repeat_equal=1`，日志 `build/m6_frozen_v3_backward_ep16_t64.log`。此回归与
编译同时运行，只作为数值参考，不取其中时间作为性能基线。

M6a候选native/library已构建，host18/18通过；native SHA256
`8760069662d5daeef278dfc7374e456ea0aba6fc579f7dc939654d7d570da6c7`，library
`02d2e4286867e5793f5799bf495746c453758f98ee3a239244c7694e26b6e730`。
冻结于 `build/static_v4_training_private_workspace/`。新旧unified Backward
均REG168/SHARED2048/LOCAL0/CONSTANT3200，GIN/local栈分别208/40bytes。
EP8 T64/E64/K4现有tensor-runtime回归通过：五种数值输出、multilayer graph、
offload各8 ranks，日志 `training_private_workspace_torch_ep8_t64_retry.log`。
首轮训练数值也通过，但错误地对整个Python测试设置了disable-static开关，导致
随后依赖static入口的inference graph准备拒绝；去掉该测试开关后全部通过。
训练本身始终为V3，并未因此启用ring training。

EP16 native T64/E32/K6通过，两次运行稳定；64个梯度hash（16ranks×四类）
与冻结V3逐项一致。pool/compute arena各减少4,575,744 bytes/rank，新单次
Backward tensors共17,422,328bytes，其中private workspace为4,575,736bytes。
此fixture的capacity等于实际R384，因此主要是所有权/分配时点迁移，**不是净省
4.58MB**。日志 `training_private_workspace_backward_ep16_t64.log`。

同一EP16 fixture只instrument rank1/9的memcheck：两份报告0errors、launcher0。
racecheck则未通过：候选报告19,706/17,876 errors；完全未改的冻结V3同方法报告
19,761/16,911 errors，均包括Forward共享内存及Backward DSM mailbox访问。
这证实这些类别不是仅在M6a出现，但不等于逐项确认误报或证明全部hazard相同。
Backward终止/DSM与Forward staging的独立分析仍待处理，不能声明sanitizer
全通过。日志 `training_private_workspace_backward_ep16_t64_{memcheck,racecheck}.log`、
`m6_frozen_v3_backward_ep16_t64_racecheck.log` 及对应带hostname/PID的工具报告。

现有PyTorch回归进一步通过以下完整case（Forward、四种梯度、graph、offload）：

- EP16 T64/E64/K4、max_rows4096：均匀及全部命中rank0两种路由，各16 ranks。
  热点rank实际R4096，其余15 ranks为0，覆盖不同rank的private plane容量。
- 单机EP8 T512/E64/K4、max_rows16384：全部命中rank0，8 ranks通过。
  热点R16384，覆盖更长的private workspace与多M-block梯度归约。

日志 `training_private_workspace_torch_ep16_t64_skew{0,1}_{0,1}.log`、
`training_private_workspace_torch_ep8_t512_hotspot.log`；所有launcher exit0。
只为现有脚本增加`TEST_MAX_ROWS`以显式容纳热点，没有新增测试套件。

EP16 native另以调用方allocator分配12轮、warmup10轮运行，通过缓存warmup后
无新allocation、12次逐输出重复校验，hash仍与冻结V3一致。新workspace与输出
一起计入`tensor_bytes`；日志 `training_private_workspace_backward_ep16_t64_allocator.log`。
`DeviceBackwardHandle::Impl`最后一个成员是completion，销毁先等待完成再释放
tensors，新增private allocation沿用这个生命周期，不要求Forward提前持有它。

M6a数值与memcheck已验证，racecheck存在上述冻结基线也能复现的未决项。
本步没有修改inference路径，不重新声明inference性能收益，也不标记M6完成。

仍待完成：ring training plan/main 分离、canonical checkpoint 映射、rank-local
peer workspace 描述符、compact dR 等通信平面的实际分配、Backward 有界 GIN
窗口、collective OOM，以及训练长文/热点/多个 live tapes 的完整门禁。
M6a 即使通过，也不满足上述全部目标。

## 训练 racecheck 定位（进行中）

现有工具为Compute Sanitizer2025.2.1.0/build35969825。NVIDIA后续
[发布说明](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
记录2025.4修复了多项mbarrier相关racecheck误报；这只是对照工具版本的理由，
不是将本项目未决hazards定性为误报的依据。

从官方CUDA13.1.0 redistrib清单下载独立sanitizer13.1.75归档，SHA256为
`98326d9dd647977956ce1f4898dcaca8db549fe9657e1d7a4cd4343990a17887`，解压到
`build/toolchains/sanitizer_13_1/`，工具版本2025.4.0.0/build36782660。没有
修改系统CUDA、NCCL或驱动。尝试对同一个M6a冻结native、同一个EP16 T64夹具
插桩rank1/9，两端在启动时均报unsupported driver，要求580+；当前570.148.08。
该轮未进入kernel，不能算racecheck通过/失败的算子证据。launcher已主动中断
并清理本轮进程，两节点GPU确认无残留任务，不升级系统驱动。

日志 `training_private_workspace_backward_ep16_t64_racecheck_2025_4.log`、
`training_private_racecheck_2025_4_{hostname}_{pid}.log`。继续用已安装的2025.2.1
与独立`-lineinfo`诊断构建定位来源，不关闭hazard/DSM检查或添加suppression。

### 聚焦 Backward 与独立同步复现

同M6a代码/冻结V3配置增加`-lineinfo`，native SHA256
`c3947d8ff9a7c60c2d4b0fcf57be2b72ac0c4ec7b5792d0be3cdd4f0499bc11e`。
仅过滤`unified_backward_kernel`且`--launch-count 1`时，rank1/9均0errors，
16 ranks重复输出及64个梯度hash仍与冻结V3一致。**这只插桩第一次Backward，
不能作为重复调用的门禁。** 去掉launch-count后，两次Backward都插桩，lineinfo
版报告21,732/19,060 errors，原M6a版为21,762/19,916 errors，均未通过。
日志`training_{lineinfo,original}_backward_all_ep16_t64_racecheck.log`及对应
hostname/PID报告。lineinfo版资源仍REG168/STACK208/SHARED2048/LOCAL0。

已显示的一类RAW/WAR落在同CTA共享内存`cuda::atomic_ref<unsigned>`的
`ld.acquire.cta.b32`与`st.release.cta.b32`本身。构造不含NCCL、GEMM、ring的
双warp最小握手：producer等phase0、写payload、release phase1；consumer
acquire phase1、核验payload、release phase0；32轮、两次kernel。工具2025.2.1
同样在原子phase上报告1,512 errors，payload核验0错误。这确认了**共享原子
读写这一特定误报类别**，不能据此排除生产kernel其它payload或DSM hazards。
临时源码/日志`build/racecheck_shared_atomic_repro{.cu,.log,_output.log}`，没有
为此增加永久测试套件或关闭检查。

另一个独立双CTA cluster复现只保留计算warp的两阶段mbarrier mailbox握手，
其它warp提前退出。32轮读写、两次launch：payload核验0错误，但工具报告64个
DSM early-exit errors。相同计算加入全CTA汇合及完整cluster退出同步后，工具
0errors/0warnings，payload仍0错误。源码`build/racecheck_cluster_mailbox_repro.cu`，
通过`-DFULL_CLUSTER_EXIT`选择对照；日志`racecheck_cluster_mailbox_{repro,full_exit}`
及`_output.log`。这定位了工具可识别的生命周期边界；数据值正确本身不证明
原生产kernel的所有DSM生命周期正确。后续候选只在Backward收尾增加完整cluster
退出边界，逐任务同步仍限于计算域，再验证资源、数值与sanitizer，未提前合入。

### Backward 完整 cluster 退出边界

计算与通信域各自结束persistent loop后，全CTA汇合，再做一次完整cluster同步。
两个CTA都不允许某个warp提前退出；每任务的mailbox/TMA握手仍不包含通信warp，
没有将通信数据依赖变成逐任务全CTA barrier。两个寄存器域仍分别执行完整循环。

冻结`build/static_v4_training_cluster_exit/`，native SHA256
`75825b4f45bbe7e18a7be7bb6331fc5c457dc6a6bc1cb253e0822e5eef0aae36`，library
`18464d8225081b41324b26ec22bb8c00b5706d078bbe8b43507b7001e897f178`。
GIN/local Backward资源分别REG168/STACK208、REG168/STACK40；两者SHARED2048、
LOCAL0、CONSTANT3200，与M6a相同。只改变训练Backward入口，不改变inference。

EP16 native T64/E32/K6双次运行通过，16 ranks的64个梯度hash全部等于冻结V3。
PyTorch完整数值/graph/offload回归通过EP8 T512/E64/K4/max_rows16384热点和
EP16 T64/E64/K4/max_rows4096热点，分别8/16 ranks三类PASS齐全。日志
`training_cluster_exit_backward_ep16_t64.log`、
`training_cluster_exit_torch_ep8_t512_hotspot.log`、
`training_cluster_exit_torch_ep16_t64_hotspot_{0,1}.log`。

只插桩两次unified Backward、rank1/9的首轮报告为1,852/1,904 errors。随后提高
print-limit至100000完整展开，分别1,864/1,948 hazards（显示数等于总数）；合计
3,812条的7,624个读写侧**全部**为libcu++共享32-bit atomic load/store，包含
acquire与relaxed读取、release写入。普通payload及DSM early-exit报告均为0。
这些共享原子报告符合上述独立复现的工具误报类别，但工具原始退出码仍为1；
不写成原始racecheck 0errors，也不扩展结论到未插桩的14 ranks或训练Forward。
日志`training_cluster_exit_backward_full_ep16_t64_racecheck.log`及
`training_cluster_exit_backward_full_racecheck_{hostname}_{pid}.log`。后续工具升级后
仍需重跑无过滤的完整训练路径，目前未添加suppression。

同EP16双次运行另做rank1/9 memcheck，两份原始报告均0errors、launcher exit0，
16 ranks重复校验通过；日志`training_cluster_exit_backward_ep16_t64_memcheck.log`。
保留此生命周期边界，作为M6训练衔接的同步基础修正提交，不宣称训练迁移完成或
inference性能改善。没有为工具误报替换正确的acquire/release，也没有新增单测套件。

## M6b：每调用 tensor preparation 的一致失败边界（已验证，训练ring未完成）

新增独立host侧`CollectiveTensorPreparation`，在训练Runtime初始化时预分配
两个uint32的device/host状态并预热NCCL控制collective（native device申请最小
128bytes，pinned申请8bytes）。每次Forward/Backward的分配回调先在本rank运行，
捕获异常后用同一个预分配状态做rank-min AllReduce。只有全部rank成功才进入
执行；失败时所有rank在main前抛错并报告首个失败rank，发生异常的rank保留
nested原始异常。部分完成的临时tensor沿既有RAII释放，不覆盖仍存活的tape。

Forward边界覆盖输入/权重lease及packed权重准备、输出、X/G/U及路由保存；
Backward覆盖四个输出和M6a私有中间workspace。调用者提供的tensor allocator
也位于回调内。模块不依赖ring/Notify内部布局，后续native训练plan/实际payload
分配可复用相同边界；现有低层`prepare_*_tensors`仍是local API，不能据此声称
所有native调用者自动获得一致失败。Runtime的inference路径不创建或使用此模块。

这是eager allocation boundary：需要host同步，不能用于CUDA Graph capture，
也不是数据面新barrier。未处理Runtime/通信资源本身首次创建失败、失效CUDA
context或损坏communicator；不能称为任意错误恢复。M6 StableTape/window plan、
rank-local peer arena以及training ring仍待实现。

在现有`torchrun_tensor_runtime.py`加入显式可选的
`TEST_TENSOR_PREPARATION_FAILURE=1`：仅最后rank在Forward第三次tensor分配、
Backward第五次分配（私有workspace）注入OOM异常；后者先由`cudaMalloc(2^63)`
返回真实`cudaErrorMemoryAllocation`，不实际占用整机显存。检查全部rank得到错误、
Forward失败不遗留Notify lease、Backward失败仍持有一份可重试tape；重试的
output/dX/dgate/dW1/dW2沿用原PyTorch参考校验。超出可寻址容量的申请被拒绝，
不等于把机器填满后的allocator碎片/压力实测；两者不混称。

冻结`build/static_v4_training_tensor_preparation/`，library SHA256
`31984df9d12257e0bf06deef968a507e85c36107bd1cd315dadf24a1cf8fdf0b`，extension
`910296a012e59ac88e8b1dfa4e72a19fbe6a38d4fef7e9f6bd2b822f6a6e0cdf`。
使用冻结V3生成profile、`-lineinfo`，未改训练GEMM或数据面同步。

- EP8 T64/E64/K4/max_rows4096均匀：16个失败通知、8个重试数值标记、原有
  parity/graph/offload各8项，launcher0。
- EP16相同shape、hotspot集中到rank0（含零接收rank），错误注入rank15：
  32个失败通知、16个重试数值标记、原有parity/graph/offload各16项，两端launcher0。
- 原生两rank独立诊断不经PyTorch：rank1直接cudaMalloc失败，确认其last-error
  仍为2再抛`std::bad_alloc`；两rank同报origin1，随后再次run成功，均exit0。
  未添加全局“清除任意CUDA错误”逻辑，现用NCCL路径能够完成这项恢复。

日志`training_tensor_preparation_torch_ep8_t64_failure.log`、
`training_tensor_preparation_torch_ep16_t64_hotspot_failure_{0,1}.log`、
`collective_tensor_preparation_repro.log`。原生诊断源码仅build归档，不建立额外
常驻测试套件。门禁证明此分配边界及当前Runtime可恢复，不代表M6训练ring完成，
也未覆盖首次pool创建OOM或每一种NCCL/driver错误。

## M6c：按实际行数分配 Backward peer payload

Backward的compact dR、source contribution、source dGate不再引用Forward完整
arena中的三个固定offset，改为execution slot拥有的独立peer workspace。
每rank的payload按本次实际接收R及本地T×K分配：dR为`max(R,1)×H×BF16`，
contribution为`T×K×H×BF16`，dGate为`T×K×FP32`，另有128-byte对齐及本地
peer地址表。零接收rank只为有效TMA descriptor保留一行dR。scheduler、ready
flags、compact→padded等小型metadata仍保持共同容量/offset，没有将其错误地
改成各rank不同的scheduler几何。M6a的私有padded planes保持不变。

IPC payload用独立native CUDA allocation，不导出Torch allocator子分配。每rank
交换handle及两个plane offset，只导入实际同节点peers；设备表保存本进程导入
地址，不能直接使用对方进程的原始CUDA指针。GIN仍经原有有界ChunkRing搬运，
不把完整dR/Return payload注册成额外跨机arena。

同一immutable Notify generation重复Backward直接复用已准备的workspace。
新generation即使本地shape相同也交换descriptor，因为peer可能换buffer；只有
实际R及T×K相同才复用本地allocation，不按历史最大R缓存。换buffer的顺序为：

1. M6b一致失败边界内分配candidate，所有rank成功后交换descriptor。
2. 关闭变更peer的旧mapping，再打开新mapping；对import结果做一致检查。
3. 所有rank成功意味着旧mapping已关闭，此时才释放旧exported buffer，发布
   本地设备地址表，然后同stream进入Backward。
4. import失败时全部rank关闭所有mapping，再汇合后释放candidate/旧buffer；
   不保留能被下一次调用误认为有效的generation缓存。

这是host preparation，不是每packet数据面barrier。发生resize时会短暂同时持有
旧buffer和candidate；不是只有新shape大小的峰值承诺。当前测试没有注入IPC
import故障，已有OOM注入仍针对Runtime tensor preparation；不声称任意IPC/NCCL
错误可恢复。训练CUDA Graph不是本次验收范围，记录中的graph门禁是inference。

冻结`build/static_v4_training_peer_payload_m6c/`，沿用冻结V3生成profile及
`-lineinfo`。native SHA256
`a39dd60d8fae0e6111c4c5adb8182fc4132783f326fe7ed4eca451186812d903`，library
`099fc7e504a670f894cfb6f6e4ed142ee85564dee5dc0366414305917738bd5d`，extension
`910296a012e59ac88e8b1dfa4e72a19fbe6a38d4fef7e9f6bd2b822f6a6e0cdf`。
最终只有删除一处未验收graph描述的comment改动。Backward GIN/local资源保持
REG168、STACK208/40、SHARED2048、LOCAL0、CONSTANT3200，没有改变GEMM流程。

- host原18/18通过；单机EP8 T64/E16/K6原生两次输出一致。
- EP16 T64/E32/K6两次重复输出通过，64个梯度hash与cluster-exit冻结基线一致
  （该基线已与冻结V3逐位核对，本次没有重跑V3）。调用方allocator另跑12轮、
  warmup10轮，12次重复校验及缓存warmup后无新tensor allocation检查通过。
- EP8、EP16 Torch T64/E64/K4/max_rows4096分别通过8/16 ranks的输出、dX、
  dgate、dW1、dW2参考校验、单rank OOM一致失败及tape重试、inference graph/offload。
- 原Torch入口增加可选`TEST_PEER_PAYLOAD_REUSE=1`，复用同一参考计算函数，
  不新建测试套件。保留三个tape后按顺序反传：T64热点首rank→T16热点末rank→
  T64热点首rank。全部五项数值、Notify lease逐个释放、第三次pool用量等于
  第一次通过；EP8/EP16分别24/48个reuse标记。
- 原生EP16全流程memcheck只插桩rank1/9，两份报告均0errors，launcher0。
- Torch热点迁移另对rank0/15的`unified_backward_kernel`做memcheck，两份报告
  均0errors、两端launcher0；48个reuse标记及后续inference graph/offload全部
  通过。此项检查实际覆盖resize后的大R热点，不只是原生fixture的固定shape。

附加Torch memcheck首轮在训练结束后停顿：native栈显示非插桩rank正在
`ncclCommFinalize`，插桩rank已进入下一个Runtime的PyTorch `all_gather_object`。
该轮被显式终止，没有最终sanitizer summary，不计通过。测试原先仅`del runtime`，
现在在训练→inference阶段边界使用已有collective `runtime.close()`，不依赖
各rank恰好同时触发Python/C++析构；未修改算子数据面。相同条件重跑正常结束，
报告local195972/remote231758。保留首轮日志、rank0/1 native stack及重跑的
`torch_ep16_t64_close_memcheck_{0,1}.log`，不将停顿归因为已证实的工具误报。

EP16热点迁移的`compute_bytes`实测如下（含固定Forward arena，不含GIN及tape）：

|rank|T64热点rank0|T16热点rank15|T64热点rank0|
|---|---:|---:|---:|
|0|47,556,152|29,995,832|47,556,152|
|15|30,783,032|34,186,040|30,783,032|

**这一步不是总显存优化完成。** V3 Forward仍使用并保留原完整X/S/output及
contribution。当前实际-sized Backward payload与之并存，只删除了原pool dGate
plane；原生EP16均匀fixture的总pool由360,114,880升至363,262,272 bytes/rank。
`pool.usage()`计入新增payload和descriptor exchange；CLI旧`workspace_bytes`仍
仅表示固定compute arena，不能单独用于比较总空间。下一步training ring接通后
才能删除不再需要的Forward完整payload。本次不报告inference加速或训练性能收益。

日志统一前缀`training_peer_payload_m6c_`：`backward_ep8_t64.log`、
`backward_ep16_t64{,_allocator,_memcheck}.log`、`torch_ep8_t64_reuse.log`、
`torch_ep16_t64_reuse_{0,1}.log`；冻结目录含source patch与sanitizer wrapper。
M6的StableTape/window plan、checkpoint生命周期及Backward GIN window adapter
仍未完成，训练继续走V3。

## M6d：单机训练 ring、canonical tape 与 plan/allocate/main 边界

本步接通单机 opt-in training Forward，不是多机训练完成。未设置
`WEFTMOE_STATIC_RING_TOKENS` 时保留 V3；设置后单机使用静态 ring，当前多机
training pool 显式拒绝，等待 Backward GIN window adapter，不静默退回 V3。

### 执行和生命周期

- 先 materialized Notify，保留其 canonical `route_dst_row`、原 combine order
  及 Backward metadata。独立 `static_ring_plan_kernel` 再冻结窗口/物理行映射；
  它只发布 routes/gates/行号，不搬 X。两层规划都在 main 前，**不是一遍 planner**。
  训练入口拒绝 fused inference / replay-counts Notify，避免复用的可变路由元数据
  破坏 live tape 的 canonical 行号合同。
- 训练专属 source `uint64[route]` 和 physical-plan `uint64[row]` 记录 tape 行号。
  不改变 inference 的24B RingRow，不把 atomic 物理枚举当作稳定逻辑顺序；
  ring 使用自己的 DeviceDispatchRuntime，不改 retained Notify 的 M64 geometry。
- host 读取实际行数，与 Notify 核对后分配独立 saved-X/G/U/gates/headers及输出。
  新的 rank-local `ForwardMegaKernelConfig::prepare_tensors` 回调在此执行，backend
  用预分配的 CollectiveTensorPreparation 一致失败。原生未绑定 tensors 的路径
  也在此分配；显式绑定的 caller-owned storage 仍可复用。没有 main 内扩容。
- 已完成协议的分配失败用 `TensorPreparationFailure` 区分于 CUDA/NCCL 控制面故障。
  前者可释放 slot 并重试：plan 已 join，尚未准入 payload，source/block bases
  未推进。没有“恢复失效 context/comm”或首次 pool OOM 恢复的承诺。
- main 只 stage X、发布 node-local ready，并消费冻结 plan，不重数或重交换。
  Pull 复用既有 FanoutRowPrefetchPipeline：一次 G2S，向物理 X 和 saved-X 各写一次；
  **两次 S2G 全部完成才发布 X full**。gate/header 同时按 canonical 行保存。
- FC1 packed gate/up accumulator按同一 BF16 rounding 写入标准分离的 G/U tape；
  S 仍在物理 ring。FC2/Return/Reduce 沿用静态 inference 数据面，Backward 使用
  原 canonical 顺序及固定 dW reduction，不再访问被覆盖的 X/S slot。
- live tape 持有独立 Notify lease 和 tensors；物理 plan/ring 在 Forward 完成后
  即可被后续调用重用。训练 plan/allocation 入口明确拒绝 capture；后续测试中的
  graph/offload 是**独立 inference 阶段**，不是训练/Backward Graph 支持。

### 容量

Python training ring 将 Notify/调度**元数据**容量至少扩到 `world*max_tokens*K`，
旧的较小 `max_rows` 不再截断合法热点。原生接口仍须给 materialized Notify 足够
的元数据容量。完整 Forward X/S 被固定深度物理 ring 替代；tape、compact dR
按本 rank 实际 R，source Backward contribution按本地 T*K 分配。
private Backward workspace仍为 `R+owned_experts*127`，不是精确 P_actual。

20K、EP8/E64/K4全命中rank0时，实际R655,360：完成Backward后的 pool
`compute_bytes` 为rank0 **3,586,689,528**、其余rank **902,339,064**；Notify
另为89,877,032bytes/rank，无GIN注册区。两者差2,684,350,464B正是实际dR和
空rank最小dR的差异。**这些数不含 caller-owned tape、Backward tensors和PyTorch
参考，不是总显存或峰值。** 该热点仅saved-X/G/U/gates/72B headers就需
4,076,339,200B，加output83,886,080B；长文训练的持久数据不会因ring消失。
本步未做匹配配置的训练E2E性能对照，不报告训练加速百分比。

### 验证与冻结

初版`build/static_v4_training_ring_local_m6d/` library SHA256
`946d93e2c81ee486d6e4aa4106fd5a7716bbd428ff446937515a6d9a889e66fc`：
host18/18；EP8 T64均匀、T512热点，B16/D1、完整分配失败/retry/三个live tapes/
后续inference graph/offload通过。随后补上no-drop元数据容量、capture拒绝和
main计时边界（不把host分配停顿计入device_kernel_us），冻结checked版：

`build/static_v4_training_ring_local_m6d_checked/`

- native `fe85a17acc79b7ed841834f48f9440827ad613202bd848a3db1e9e37093d2cdb`
- library `841af316674ff19df33fb3ee589d4ab0ecc14797d76febe0e5d074031230b4ee`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

planner REG80/STACK0/SHARED1024/LOCAL0；训练 main prefill/decode均REG168、
SHARED2048/LOCAL0，STACK72/96。EP16 inference REG168/STACK432/440未变。

checked版已有门禁：

| case（全部单机EP8） | 结果 |
|---|---|
| native T64/E16/K6、B16/D1 | 8 ranks两次Backward repeat_equal=1；32个梯度hash逐字与冻结M6c V3训练路径一致 |
| Torch T64/E64/K4热点、max_rows64、B16/D1 | 五类数值、16个failure、8个retry、24个live-tape复用、8个graph及8个offload标记全部通过 |
| Torch T512热点、max_rows64、B16/D1 | 实际R16384，完整上述检查；rank0/7全部weftmoe kernel的memcheck均0，launcher0 |
| 同一T512，单独Forward racecheck | rank0/7 planner及unified Forward均0 hazards，launcher0；不包含旧Backward racecheck未决项 |
| Torch T1、T8192均匀，max_rows64、B512/D8 | 各8 ranks五类数值及后续inference graph/offload通过 |
| Torch T20480全热点，max_rows64、B512/D8 | R655360，8 ranks五类数值及后续inference graph/offload通过 |

Torch数值沿用现有脚本rtol0.02/atol0.00002；没有新增测试套件。短/长训练正确性
可与CPU构建或另一台机器的独立EP8正确性任务并行，不取其耗时作性能证据。
memcheck报告PID224875/224884，racecheck226064/226073；前缀
`training_ring_local_m6d_*`。Backward全量racecheck的历史工具原子报告仍保留，
不因新Forward通过而称整个训练sanitizer全通过。

### Inference 回归与参数布局消融

固定1620MHz、B512/D8、10 warmup / 20 samples / graph batch20，checked版
正式回归如下（us，slowest-rank p50；每个shape输出hash与冻结DS一致）：

| T/rank | EP8 checked | EP16 checked |
|---:|---:|---:|
| 1 | 101.122 | 279.278 |
| 64 | 134.638 | 334.571 |
| 8192 | 1261.710 | 2949.040 |
| 20480 | 2954.080 | 7024.960 |

checked新增的训练指针最初插在 ComputeRingView / ForwardKernelArgs 中间。
EP16 T20480相邻对照为 checked **7024.960** → DS **6991.060** → checked
**7029.450**，约0.5%回退。将四个训练指针集中到参数尾部的 RingCheckpointView，
保持既有 inference 字段偏移后，isolated版为 **6988.290**，随后DS反测
**6995.330**；范围分别6967.370–7025.190、6958.740–7020.000，视为持平，
**不报告加速**。这是参数布局A/B，不改变训练或inference的数据面协议。
REG/STACK未变，尚不能据此把回退归因到某一条SASS指令。

isolated native位于`build/static_v4_training_ring_local_m6d_isolated/`，SHA256
`f0595ea54e6a4f5c2d2190ae45fb4c465e51897cbb38969b9f97438d0c80c0ad`。
两个相邻正式20K日志为`training_ring_local_m6d_isolated_inference_ep16_t20480_formal.log`
及`training_ring_local_m6d_isolated_inference_ds_ep16_t20480_reverse_formal.log`。
所有正式性能轮次均无CPU构建或其他GPU任务并行；早期EP8的`quick`文件实际为
10 warmup/100 samples/batch2（launcher默认rounds110），不混入上述正式表。

最终版在isolated基础上增加上述immutable Notify入口检查，冻结于
`build/static_v4_training_ring_local_m6d_final/`（目录内`change.patch`）：

- native `1001c146a6145e9718a45b1c78354ac5db0e751c65dd73be493888ac35a450f1`
- library `5e0f53f0b64b15816202c404be154411322bd8eeb0e319deef3d731183a77218`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

最终版host18/18；native T64两次Backward重复一致、32梯度hash逐字等于M6c。
重新完成EP8 Torch T1/64/8192/20480（64和20480热点、max_rows64）；T64与
T512/B16/D1再跑完整OOM/retry/三个live tapes。T512 rank0/7 memcheck零错误
（PID241301/241315）；同case独立Forward planner/main racecheck零hazards
（PID242632/242653）。全部launcher退出0，五类数值、graph/offload标记齐全。
最终native的cuobjdump资源报告与isolated逐字一致。日志前缀
`training_ring_local_m6d_final_`；此处单机测试可在两台机器上分别独立运行，
**不是双机EP8**。

最终版同样在1620MHz、B512/D8、10/20/batch20重新完成正式inference矩阵：

| T/rank | 单机EP8 / us | 双机EP16 / us |
|---:|---:|---:|
| 1 | 101.008 | 277.709 |
| 64 | 134.936 | 333.302 |
| 8192 | 1259.570 | 2953.690 |
| 20480 | 2959.400 | 6985.580 |

八个case全部退出0，hash与DS一致，compute/registered/total pool bytes不变。
日志`training_ring_local_m6d_final_inference_ep{8,16}_t{1,64,8192,20480}_formal.log`。
这是训练功能合入的inference回归，不是新的通信性能收益；EP16 20K相对历史
冻结V3仍只有小幅优势，远未满足长序列显著领先的优化目标。

M6d提交`4c22ddc`后，冻结final native又通过EP8 T64调用方allocator的12轮
Backward（warmup10）：8 ranks均checked_repetitions12，warmup后无cache miss，
32梯度hash仍与M6c逐字一致。日志`training_ring_local_m6d_final_backward_ep8_t64_allocator.log`。
该正确性检查与下一候选CPU构建并行，不作性能比较。

## M6e：多机训练 ring 的 checkpoint / Backward GIN adapter

基于`07856f7`，以下长文正确性及inference回归已完成，不标注完整M6完成。
复用M6d canonical Notify、独立planner、实际tape分配边界及FC1 epilogue；
新增network planner/main实例，main只对节点内发布已stage的X。远端Pull复用
抽出的`checkpoint_copy.cuh`，以RouteLocation的逻辑block/行映射saved-X与
gate/header；一次G2S、两次S2G全部完成，才发布X-full并计入RX consumed。
expected/consumed仍是逻辑route数，不随物理写入次数翻倍。

GIN注册区追加既有有界Backward ChunkRing，不恢复旧完整Forward payload。
首版两套payload不别名，控制状态和NIC counters必须分别持久化；执行slot在
Forward/Backward完成间交替复用，不由live tape占有。counter0/1继续保留：
若每slot静态ring占F个counter、Backward占B个、共有S个slot，则静态slot i用
`[2+i*F, 2+(i+1)*F)`，Backward用`[2+S*F+i*B, 2+S*F+(i+1)*B)`。
两套初始化均执行，之后不在每调用重置。logical GIN contexts仍8；当前训练
devComm申请133个context（实际以backend返回为准），覆盖Forward静态96及既有
Backward需求，操作完成后才复用。
所有channel/peer映射沿用runtime节点数，不加双机专用数据面分支；硬件验证仅2机。

冻结`build/static_v4_training_ring_network_m6e/`，内含change.patch：

- native `3727e90ed001f46a0224b796069a22d6284ac8c21c3592f5e7ecdf650595fb2e`
- library `9cbbe860142249d289b71f3d4cedac1f2fea5ae008be9eb5b393f6ae2bdc6b62`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

network训练planner REG107/STACK72/SHARED1024；训练prefill/decode main均
REG168/STACK448/SHARED2048/LOCAL0；inference仍REG168/STACK432/440。
已通过host18/18；EP16 native T64/E32/K6、B16/D1两次Backward重复一致，64个
梯度hash逐字与M6c训练参考相同。EP16 Torch T64/E64/K4、max_rows64、B16/D1：
五类数值、32个failure、16个retry、48个live-tape复用、16个graph及16个offload
标记齐全，两节点launcher均0。日志前缀`training_ring_network_m6e_`。

EP16 T512全热点、max_rows64、B16/D1也完成完整上述检查，rank0实际R32768。
rank0/15所有weftmoe kernel memcheck均0errors（PID257087/243508），两个
launcher均0；日志`training_ring_network_m6e_torch_ep16_t512_memcheck_{0,1}.log`。
同case的独立Forward racecheck也通过，rank0/15 planner及unified Forward均
0hazards（PID258162/243897），完整脚本及两launcher均0；不包含旧Backward
racecheck原子报告未决项。日志后缀`t512_racecheck_{0,1}.log`。

EP16 T1/8192均匀及T20480全热点、max_rows64、B512/D8均完成五类数值及后续
inference graph/offload（每case各16 ranks、两个launcher均0）。20K热点rank0
接收R1,310,720，完成Backward后pool compute_bytes为 **6,485,332,024**，其他
rank **1,116,627,000**；差额5,368,705,024B等于实际dR减空rank最小4KiB。
GIN为473,820,928B/rank、Notify为90,379,952B/rank。仍不含独立tape、Backward
tensors及PyTorch参考，不是总显存或峰值；该热点saved-X/G/U/gates/headers本身
即需8,152,678,400B。没有以减少pool统计冒充总显存收益。

T512网络入口是decode profile；另补T2048/B16/D1全热点的memcheck及Forward
racecheck，以覆盖prefill宽tile、完整M128的staged epilogue和热点迁移时profile
切换。两次完整脚本均通过：memcheck rank0/15零错误（PID261957/245557），
Forward racecheck零hazards（PID262632/245927）；各有16数值、32failure、
16retry、48live-tape、16graph、16offload标记，两节点launcher均0。

单机兼容另通过native EP8 T64两次Backward（32hash逐字等于M6c）及Torch EP8
T512全热点、B16/D1的完整OOM/retry/live-tape/graph/offload。并发时仅在另一台
机器跑独立EP8正确性，不将这两种单机case写成双机EP8。
正式inference回归也完成：1620MHz、B512/D8、10warmup/20samples/batch20，
测试期间无并发CPU构建/GPU任务；各case输出hash及pool bytes与M6d/DS一致。

| T/rank | 单机EP8 / us | 双机EP16 / us |
|---:|---:|---:|
| 1 | 100.973 | 278.344 |
| 64 | 134.939 | 333.205 |
| 8192 | 1253.330 | 2957.980 |
| 20480 | 2960.660 | 6994.480 |

性能与原inference数据面持平，不把训练功能接入宣称为inference加速。
EP16 20K相对冻结V3仍仅小幅领先，长序列显著提速目标未完成。日志
`training_ring_network_m6e_inference_ep{8,16}_t{1,64,8192,20480}_formal.log`。

## M6f：按精确P分配Backward私有workspace（已验证）

在materialized Notify已有的逐专家计数归约中，同时求
`P=sum_e ceil(rows_e/128)*128`，随原有status一次下载保存到不可变handle；
不下载expert count数组，不增加host同步或二次路由规划。复用现有M128
canonical block定义，Notify自身M64 tile目录不变。

Backward U/dAB/dX/dGate私有存储由`R+owned_experts*127`改为精确P；
P=0时只保留一行供合法TMA描述符使用，设备端实际mblocks仍为0，不生成假任务。
caller提供已准备tensor storage时同时检查padded容量，不能仅凭R相同接受
专家分布不同的较小workspace。fused或replayed Notify没有不可变host P，
在训练tensor准备边界显式拒绝，未新增training Graph支持。

仅四个源文件改动，无Backward GEMM、ring协议、公共API或新增测试套件。
冻结`build/static_v4_training_exact_padding_m6f/`，保存change.patch和resources：

- native `042c09481f82c23ce17ed1b7cb7a1ea15b7f35b102ad32961d14262eb16de41c`
- lib `b69a6f47910a51837b1ce9b586019f54179e87a65996b54826ab7514f2ab7b5e`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

host18/18通过。Native EP8/EP16 T64/E16或E32/K6/B16/D1各两次Backward通过，
共96个梯度hash逐字等于M6e，每rank repeat_equal=1。这个case每rank R384、
实际P512，旧容量638行；Backward tensor bytes由17422328降到16518656，
减少903672bytes/rank（workspace减少126×7172bytes）。输出/权重梯度等其它
tensor容量不变；这是私有workspace节省，不是总训练显存或性能加速。

日志`training_exact_padding_m6f_backward_ep{8,16}_t64.log`。caller allocator的
EP8 T64另做12次Backward/warmup10，8个rank checked_repetitions=12、repeat_equal=1。

Torch EP8/EP16 T1/64/8192 balanced和T20480全热点均通过5项数值参考及后续
inference graph/offload；每case分别8/16个完整pass标记。20K热点仍覆盖
EP8 R655360、EP16 R1310720，未因收紧private workspace丢token。
EP16 T2048/B16/D1、EP8 T512/B16/D1的完整热点迁移脚本也通过：每rank两次
tensor preparation失败注入、Backward tape重试、三个live tapes及后续inference
graph/offload。memcheck覆盖所有weftmoe kernel，EP16 rank0/15为0错误
（PID283822/250241），EP8 rank0/7为0错误（PID290949/290963）。
没有以这些memcheck结果覆盖此前完整Backward racecheck未决报告；本轮未重新
声称训练Graph可用，graph/offload仍是脚本中独立的inference阶段。

日志`training_exact_padding_m6f_torch_ep{8,16}_t*`、
`training_exact_padding_m6f_mem_ep{8_t512,16_t2048_*}.log`以及对应memcheck报告。
非ring回退另用冻结V3生成profile构建，同一源代码的native SHA256
`8a946910f335a4d6a5a39aa13cdde169662c428692548fd9404444042fe72ef3`，冻结于
`build/static_v4_training_exact_padding_m6f_v3/`。取消ring环境变量后EP8/EP16
T64各两次Backward通过，96个hash与M6f ring/M6e完全一致；不是将inference
profile的RDMA_CTAS=0当作原多机训练入口。

1620MHz下inference四规模正式回归（10warmup/20samples/batch20）如下；
没有并发CPU构建或其它GPU任务，各case hash及device/registered容量均与M6e一致。

|T/rank|单机EP8/us|双机EP16/us|
|---:|---:|---:|
|1|100.819|278.118|
|64|134.979|333.563|
|8192|1268.100|2953.070|
|20480|2960.200|6991.990|

EP8 8K比M6e前次1253.330us高约1%，进一步作同机相邻正式对照：
M6e基线1267.940us（1264.370–1273.980），M6f复测1261.500us
（1256.920–1267.270），16个hash及pool容量一致，未见稳定退化，也不宣称加速。
两artifact的本地prefill main SASS完整输出相同（各33916行，含匹配function与
指令，非空dump），SHA256均为
`27131fb300a98572742dc766b0ef1936b0c7c56abbab6f25bb1c035be6838478`。
这支持GPU入口没有改变，但不能单凭机器码相同排除运行环境/地址分配的影响。
日志`training_exact_padding_m6f_inference_ep{8,16}_t*_formal.log`。
8K相邻对照为`training_exact_padding_m6f_inference_ep8_t8192_{baseline,repeat}.log`。
本里程碑完成精确私有容量分配及回归，不等于完整训练性能/racecheck验收完成。
EP16 20K仍仅比历史冻结V3的7096.930us快约1.48%，显著领先的性能目标未达成。

## M6性能基线与Backward profile审计（进行中，2026-09-24）

使用既有native benchmark，无新测试入口。Forward设置
`WEFTMOE_FORWARD_BENCH_RETAIN_STATE=1`并取消graph replay，固定路由及已准备
tensor复用；wall范围为Notify→buffer preparation→Forward完成，包含训练ring的
独立plan、实际行数下载及准备阶段。`device_kernel_us`仅为main kernel，不包括
这些主机边界；V4的`forward/total_us`当前等于main，不能将其直接当成与V3
event区间相同的训练E2E。此处分别列wall/main，不相加或混用span。

1620MHz、H2048/I512/E64/K4、10warmup/20samples、同步rank后每调用单独计时。
每个形状的两版本在同一节点相邻运行：T1/64在第一节点，8K/20K在第二节点。
测量节点无CPU构建或其它GPU任务；部分第二节点轮次与第一节点CPU构建或独立
EP8任务并行，不把它们描述成双机EP16。ring为已提交M6f、B512/D8，参考为冻结
`static_inference_v3_20260923`，非当前源码重编译的“V3”。

|T/rank|M6f wall/us|冻结V3 wall/us|M6f main/us|冻结V3 main/us|
|---:|---:|---:|---:|---:|
|1|404.456|472.547|74.112|198.336|
|64|469.037|551.276|107.040|264.416|
|8192|2017.380|1726.450|1522.820|1335.870|
|20480|4289.020|3577.480|3683.360|3137.500|

32个输出hash两版本完全一致。短序列wall更快，但长序列训练Forward仍退化，
不能以inference的改善代替训练性能门禁。20K pool分别408121664与1820708544
bytes/rank；这不是总训练显存，另有独立tape、Backward tensors等，不能据此
声称总内存减少同样比例。日志`training_perf_{m6f,v3}_{remote_,}ep8_t*_forward.log`。

复用固定tape的Backward（相同形状、1620MHz、10/20、第二节点EP8）device total：

|T/rank|M6f ring profile/us|冻结V3 profile/us|
|---:|---:|---:|
|8192|2387.810|2064.700|
|20480|5706.530|4979.970|

审计预处理宏发现，inference专用ring配置没有显式选择Backward：生成Forward
配置下默认`WEFTMOE_BWD_UNIFIED_WORKERS=0`、`BWD_RELAY_CTAS=32`；冻结V3配置
则为1/16。两者核心GEMM tile/stages相同，但执行路径不同，不能把上述退化
归因于精确分配或checkpoint。M6f同源码的冻结V3-profile构建
`static_v4_training_exact_padding_m6f_v3`，不用ring的20K Backward为4920.540us
（4885.570–4957.470），已接近冻结V3；仍需只切换ring配置的独立验证。
Backward benchmark的`gemm_launches=2`为既有固定audit值，不能据此判定split/
unified入口。已有数值通过不证明两配置性能相同。

日志`training_perf_{m6f,v3}_remote_ep8_t{8192,20480}_backward.log`及
`training_perf_m6f_v3profile_remote_ep8_t20480_backward.log`。计划显式复用
V3 unified Backward并完成矩阵、热点/多tape和sanitizer回归；当前尚未改默认配置，
双机训练性能矩阵也未完成。不将这份基线记录标成M6性能验收完成。

### M6g第一轮：unified开关恢复，但发现sender配额串用了Forward（未采用）

仅在ring生成配置显式设置`BWD_UNIFIED_WORKERS=1`、`BWD_RELAY_CTAS=16`，
没有改内核或生命周期。冻结`build/static_v4_training_unified_backward_m6g/`：

- native `533ede29925c9f1592c6c0d171fbc33540f7f5e59ffdcb7e980e9912c3574ca5`
- lib `2fff031c5a8a8d8d0928533b302bcdd93b3961ab2de5f025f3d281a1c4a09c9e`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

1620MHz、既有cached-tape Backward benchmark、H2048/I512/E64/K4、10/20，
单机在第二节点，双机1×8+1×8；测量节点无CPU构建/其它GPU任务：

|T/rank|EP8 M6f split/us|EP8 M6g unified/us|EP8冻结V3/us|EP16 M6f split/us|EP16 M6g unified/us|
|---:|---:|---:|---:|---:|---:|
|1|344.640|236.256|233.056|1355.550|2455.740|
|64|408.064|368.736|370.752|1392.160|2599.040|
|8192|2387.810|2064.830|2064.700|6661.090|24830.600|
|20480|5706.530|4910.780|4979.970|14571.300|60507.400|

列为device total的rank-max p50，不是Forward+Backward合并训练时间。EP8恢复
V3附近，但不能仅凭单机结果采用：EP16明显退化。进一步查到host launch仍用
`forward_tuning::unified_rdma_ctas`设置Backward sender数；ring Forward配置的
`FWD_UNIFIED_RDMA_CTAS=0`使Backward回退为**1个sender**，而原V3 Forward配额
默认为16。这是第二处执行profile耦合，不是unified算法本身已被证明不适合多机。
下一轮将Backward sender配额独立出来，再与冻结V3和原split路径比较。

第一轮正确性已通过host18/18、native EP8/EP16 T64/E16或E32/K6/B16/D1各两次
Backward，共96个梯度hash逐字等于M6f。Torch EP8/EP16 T1/64/8192均匀、T20480
全热点全部五项数值及脚本中的独立inference graph/offload通过；T64各完整
OOM/retry/三个live-tape检查通过。EP8 T512及EP16 T2048全热点B16/D1也通过
完整脚本，all-weftmoe memcheck rank0/last均0errors：EP8 PID329079/329089，
EP16 PID330657/263385。仍不覆盖旧Backward racecheck共享原子未决报告。

EP8 inference正式10/20/batch20为100.795/135.640/1266.920/2955.740us，
32个输出hash和全部pool bytes等于M6f；计时节点独立，另一节点仅跑EP8正确性。
这些结果只是该候选的回归，不是inference优化。日志前缀
`training_unified_backward_m6g_`、`training_perf_m6g_`、`training_perf_m6f_ep16_`。
双机inference正式回归随后完成，为278.469/332.790/2954.220/6990.330us，
64个hash与pool容量均等于M6f，未见稳定改变。同期冻结V3的EP16 Backward
device total为339.008/701.024/2757.600/6862.430us；T1有一次1641.440us
离群值，p95为347.200us，表述中不隐藏这一波动。原unified候选因sender耦合
未采用，尚未提交配置变更。

### M6g第二轮：接通Backward独立sender配额（已验证）

直接复用已有但此前未使用的`BWD_UNIFIED_RDMA_CTAS`，host按Backward配额划分
sender/receiver，clamp后至少保留一个receiver；所有CTA仍保留GEMM角色。
旧生成计划里的0解释为Backward默认16，不再受Forward的0/mixed/静态模式影响。
ring配置明确选择unified及16个sender。sender completion数组原已按SM数分配，
无需加buffer、改变GIN counter区间或修改设备内核。

首个显式16配置构建`static_v4_training_backward_senders_m6g2/`：native
`20e7a637461320eb381680d301686512963a928ebe35ebcb16531d8060b9cfb1`，
lib`193441a907de044549e2d5ce4f736541c10cd33cbc2e0fee48c135669eabe954`。
EP16 20K诊断测量10/20为6886.300us（6836.060–6995.330），64个梯度hash
逐字等于同期冻结V3；恢复到V3附近，明显优于原split。该轮启动期间还运行了
host配置语法检查，不作为最终无并发构建的正式验收结果。

审计发现通用资源头早已将此兼容字段定义为0，所以不能仅在Backward头加
`#ifndef ... 16`。最终改为显式解释旧值0，并同步调优工具的phase-owned说明。
在现有资源测试中只增加一个针对Forward=0串用及Backward缺省/0/显式值的检查；
修改前确实失败，兼容处理后phase资源15/15、参数验证10/10通过。
较大的搜索空间测试有4 failures/2 errors；换回HEAD的原`unified_tuning.py`
仍产生相同失败，全部是Forward候选规模/旧staging容量断言，本轮未扩大范围
修改这些既有问题。日志`training_backward_senders_m6g2_{macro_before,
phase_resources,tuning_tests,search_space_head}.log`。

最终冻结`build/static_v4_training_backward_senders_m6g2_final/`：

- native `cc30bf44040fa48efed7bab0a9545bb14525474e538f47b4bb0b1e4f7e77b683`
- lib `ade15d7cdf2d1ee7a2ad9a9b5f83d5148f087678a9ca9f7595b5e752bf07c73d`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

两个unified Backward kernel的SASS与第一轮M6g完全相同，组合dump共82689行，
确认包含local/network两个function而不是空dump，SHA256均为
`a8a044ff6c43deaaacfacb23672ad23856c879c474659becafe256b5085eb51c`。
这支持收益来自host角色配额修正，不是GEMM或数值优化；不能代替运行时验收。
EP8/EP16 native T64/K6/B16/D1已通过，两次Backward稳定，96个梯度hash等于M6g/M6f。
EP8四规模inference正式回归为100.688/135.434/1266.750/2950.030us，32hash与
全部pool bytes等于M6f。

EP16 Torch T1/64/8192均匀及T20480全热点，五项数值及后续独立inference
graph/offload全部通过；T64有32 failure、16 retry、48 live-tape标记。
EP16 T2048、EP8 T512全热点B16/D1完整脚本通过，all-weftmoe memcheck的
rank0/last均0errors：EP16 PID350801/270961，EP8 PID352107/352117。
EP8主矩阵此前第一轮M6g已完整验证，本轮本地Backward设备代码不变，另用
native重复调用、小ring热点/迁移/重试及下面的正式矩阵回归。

最终1620MHz、cached-tape Backward、10warmup/20samples的device total如下；
EP8在第二节点独立测量，部分轮次与第一节点构建/EP8正确性并行；测量节点无
CPU构建或其它GPU任务。EP16计时期间两节点均无并行构建/其它GPU任务。

|T/rank|EP8修正后/us|EP8冻结V3/us|EP16修正后/us|EP16冻结V3/us|
|---:|---:|---:|---:|---:|
|1|234.656|233.056|343.648|339.008|
|64|369.184|370.752|696.640|701.024|
|8192|2063.870|2064.700|2771.040|2757.600|
|20480|4917.220|4979.970|6873.570|6862.430|

EP16四形状共256个梯度hash逐字等于同期冻结V3。20K相对M6f split的EP8
5706.530us、EP16 14571.300us分别减少约13.8%/52.8%；主要是恢复应有profile，
不能宣称新Backward算法显著领先V3，也不能将独立Forward/Backward中位数相加
冒充实测整步训练时间。日志`training_perf_m6g2_{remote_ep8,ep16}_t*_backward.log`。

最终inference正式10/20/batch20如下，96个hash及全部pool bytes等于M6f：

|T/rank|单机EP8/us|双机EP16/us|
|---:|---:|---:|
|1|100.688|277.781|
|64|135.434|333.541|
|8192|1266.750|2956.430|
|20480|2950.030|6996.060|

无稳定inference改变；日志`training_backward_senders_m6g2_inference_*_formal.log`。
本里程碑采用Backward profile/sender解耦修正，不关闭训练Forward长文退化、旧
Backward racecheck未决报告或EP16长序列inference显著领先V3的剩余目标。

## M6h：复用FC1 epilogue staging合并训练checkpoint写回（已验证）

宽入口的完整M128 tile原已在退休的CTA私有B staging中重排S。本轮同时暂存
BF16 G/U，复用相同的两次math CTA barrier，再以16B vector写回canonical
checkpoint行；替代原按accumulator fragment分散的4B stores。G/U仍是分离的
canonical平面，保留原BF16舍入，不改Backward布局、ring credit或saved X。
不复用cluster multicast A。只有三个输出平面能放入实际B arena时启用；
窄decode、尾块及scratch不足的配置继续direct checkpoint/epilogue。

共享helper接收调用方实际scratch容量，并推导Args类型；这也修复纯GEMM工具
在M6d接口演进后缺mblock实参、精简Args被强制要求训练字段的编译问题。
诊断工具仍调用生产epilogue，未复制一套实现或添加假的checkpoint字段。
N64/stages2/K128的packed及separated、T1/8192完整chain均精确等于cuBLAS且
repeat逐字相同；N32/stages3/K128 packed T1也通过。N32 separated是工具
既有不支持组合，尝试运行返回unsupported，未计作通过项，也不放宽其合同。
这些是工具接口/计算回归，不能作为本轮checkpoint收益或重新选布局的证据。

最终冻结`build/static_v4_training_staged_checkpoint_m6h_final/`：

- native `5b3e9264afc3cdff75678b45c6ab304b80033f06744421f53ae86091ecc3ee62`
- lib `16e81efa4a96757e77f7f55f697a2a3c0361f05a1ea6d6b96bced0decffc3073`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

训练prefill local/network仍REG168、STACK72/448、SHARED2048、LOCAL0；
dynamic SMEM及pool容量不增加。首版候选`static_v4_training_staged_checkpoint_m6h/`
仅供初筛，正式对照使用上述修复诊断接口后的最终构建。

1620MHz、EP8（第二节点独立测量）、E64/K4/H2048/I512、B512/D8、保留tape，
10warmup/20samples，与已提交M6g2按G/H/H/G顺序交错。第一节点同时做EP8
正确性检查/纯GEMM CPU构建，测量节点无其它GPU任务或CPU构建。

|T/rank|M6g2 main/us，两轮|M6h main/us，两轮|M6g2 wall/us，两轮|M6h wall/us，两轮|
|---:|---:|---:|---:|---:|
|8192|1518.820 / 1529.380|1511.010 / 1512.930|2021.520 / 2040.070|2017.200 / 2011.230|
|20480|3698.270 / 3693.220|3638.850 / 3638.020|4305.780 / 4276.040|4220.910 / 4217.040|

20K main约降低1.5%–1.6%，8K约0.4%–1.2%，小幅收益，不足以恢复长文相对V3
的训练Forward差距。wall包含Notify/host准备/Forward，不与V3的不同阶段span
混比；更不把独立Forward/Backward中位数相加当作整步训练实测。
日志`training_perf_{m6g2,m6h,m6h_repeat,m6g2_repeat}_adjacent_remote_ep8_t*_forward.log`。

最终native EP8 T512/E16/K6/B512/D8两次Backward稳定，32个原始梯度hash逐字
等于M6g2，覆盖完整M128 staging。Torch EP8 T8192均匀、T20480全热点五项
数值及独立inference graph/offload通过。EP8 T512全热点B16/D1完整脚本
通过（OOM/retry/三个live tapes），rank0/7 all-weftmoe memcheck 0errors，
PID362837/362854。

EP16 T1/64/8192均匀、T20480全热点，五项数值及独立inference graph/offload
均通过；T64完整检查含32failure/16retry/48live-tape标记。EP16 T2048全热点
B16/D1也通过完整脚本；rank0/15 all-weftmoe memcheck 0errors（PID367822/
277218），独立Forward planner/main racecheck 0hazards（PID369184/277605）。
全部launcher exit0，不将Forward专项结果外推到旧Backward racecheck。
日志前缀`training_staged_checkpoint_m6h_torch_`及`..._{memcheck,racecheck}_`。
EP16训练Forward同配置、10/20、无并发构建/其它GPU任务的相邻对照如下。
两规模共32个Forward输出hash从原始rank记录按字符串逐字核对，相等；不能
用JavaScript浮点解析后的64-bit hash做精确一致性证据。

|T/rank|M6g2 main/us|M6h main/us|M6g2 wall/us|M6h wall/us|
|---:|---:|---:|---:|---:|
|8192|3518.050|3488.960|4319.950|4296.780|
|20480|8373.980|8318.430|9327.650|9281.240|

双机main约降低0.8%/0.7%，不夸大为显著端到端收益。GIN launcher中的
`forward_slowest_rank_us`在本ring构建对应main，native记录的`device_kernel`
与`forward`逐样本相同；wall仍按Notify→buffer→Forward完整范围报告。
日志`training_perf_{m6g2,m6h}_ep16_t*_forward.log`。
EP8 T1/64也完成Torch完整数值/graph/offload，T64含16failure/8retry/24live-tape
标记。EP8 T512全热点B16/D1的Forward racecheck rank0/7均0hazards，
PID377953/377968，完整脚本及launcher exit0。

最终1620MHz、10/20/batch20的inference正式回归：

|T/rank|单机EP8/us|双机EP16/us|
|---:|---:|---:|
|1|100.725|277.926|
|64|135.403|333.410|
|8192|1271.100|2956.020|
|20480|2964.940|6985.870|

96个输出hash逐字等于M6g2。EP8在第二节点独立测量，第一节点有候选CPU构建/
EP8正确性工作；EP16测量无并发构建/GPU任务。日志
`training_staged_checkpoint_m6h_inference_{remote_ep8,ep16}_t*_formal.log`。
本项是小幅训练checkpoint优化，不将这些inference波动归为收益；20K相对V3
显著领先的目标、训练Forward长文差距及旧Backward racecheck未决项仍未关闭。

随后EP8反向复测M6g2：8K/20K为1262.080/2953.510us，16个hash与M6h一致，
仍有约0.7%/0.4%进程间差异。核对最终native中的local/network两个inference
prefill kernel，SASS逐字相同（56282行、确认包含两个function），共同SHA256
`af9efeb01afc1e754002d74ea42f7e57a4e653563541d16eb49e782630df4c70`；dump含
函数名及指令，不含不同路径/调试行号。故不归因为本次训练epilogue修改，也不
选择较快一轮替换正式矩阵。日志后缀`m6g2_inference_remote_ep8_t*_reverse_formal.log`。

## EO：训练X/SavedX双lane写回（退化，撤回）

在已提交M6h的N128及原调度上，复用既有
`FanoutRowPrefetchPipeline<GroupLanes=2>`，每个Pull warp仍只发一次G2S，
lane0写ring X、lane1写SavedX。共享两个staging slot及一个loader，复用现有
subgroup read-completion join；不增加staging内存或新的计数协议。

本地Pull的两个lane执行相同不可变行枚举，gate/identity及X credit只由lane0
写；两lane各自full completion后再join，才发布X-ready。远端Pull同样使用
两lane copy/drain，再经过原full-warp join发布X-ready和RX consumed。inference
仍编译为单lane。没有改checkpoint布局、route expected、slot generation或
双NIC credit。这是提交开销消融，尚无正确性/性能结论；不混入EM/EN的N256/
lookahead变化。冻结目录`build/static_v4_training_dual_lane_checkpoint_eo/`。

native SHA256 `1ce55d2019cf9a777104a5ed48ecfc54530b406e97c8d8cf3cc7b969e3ef09fa`。
训练prefill local/network为REG168、STACK80/456（均比M6h增加8bytes）、
SHARED2048、LOCAL0。EP8 T512/E16/K6的B512/D8和B16/D1各两次Backward稳定，
共64个原始梯度hash逐字等于M6h；小ring覆盖多次复用，不代替完整数值/sanitizer。

1620MHz、B512/D8、10warmup/20samples，保留tape的训练Forward如下：

|拓扑|T/rank|EO main/us|M6h main/us|EO wall/us|M6h wall/us|
|---|---:|---:|---:|---:|---:|
|单机EP8|8192|1559.650|1501.920|2055.920|2022.320|
|单机EP8|20480|3772.450|3629.250|4354.150|4205.590|
|双机EP16|8192|3671.780|3488.960|4489.430|4296.780|
|双机EP16|20480|8842.560|8318.430|9809.580|9281.240|

EP8为第二节点独立测量，第一节点运行native正确性；EP16两节点无并发构建/
其它GPU工作。M6h参照来自本日EM相邻EP8及M6h正式EP16，不冒充本轮交错复测。
两拓扑均退化，三个源文件全部撤回，不继续library/完整Torch/sanitizer。
日志`training_dual_lane_checkpoint_eo_{native_ep8,remote_ep8,ep16}_*`。
两拓扑的48个Forward输出hash也逐字与各自M6h参照一致；EP16从原始rank记录
提取字符串核对，没有使用JSON中已丢精度的64-bit数值。

## EP：FC1大尾块复用合并写回（仅单机K1小收益，暂不采用）

在M6h上将现有staging条件从完整M128放宽到至少64行，vector store只枚举
有效行；较小尾块及窄decode保持direct。不改变G/U布局、scratch大小、barrier
个数或数值合同。冻结目录`build/static_v4_fc1_large_tail_staging_ep/`。

主矩阵E64/K4/G8/B512的均匀路由按64周期整齐分布，每窗口每expert为256行，
因此8K/20K没有该类尾块，此项不能解释或恢复主矩阵长文差距。独立用既有K1
基准产生每expert每窗64行，保持T/B不变做尾块消融；不拿K1时延代替K4主矩阵。
最初K3诊断调用被既有launcher拒绝（exit2，未测量），未修改其支持合同。

native SHA256 `2cbe82b693136d650e9e587e8eb0897ed2c7b7c9a7f9c149bc0b6f300b82f5e7`。
prefill训练/推理的local/network资源与M6h相同。EP8 T512/E64/K1/B512/D8
恰好覆盖M64尾块，两次Backward稳定，32个原始梯度hash逐字等于相邻M6h。
第二节点独立EP8、1620MHz、10warmup/20samples的K1训练对照：

|T/rank|M6h main/us|EP main/us|M6h wall/us|EP wall/us|
|---:|---:|---:|---:|---:|
|8192|837.664|833.952|1391.920|1388.350|
|20480|2004.220|1990.780|2569.840|2550.300|
|20480（反向）|2005.380|1989.540|2566.560|2552.200|

四个候选/基线对照共32个输出hash一致。20K主kernel约改善0.7%–0.8%，是
K1尾块诊断的小收益；第一节点同时有候选CPU构建，计时节点无其它任务。
没有据此合入，源码已恢复M6h以隔离下一项配额实验；双机及完整门禁待执行。
日志`fc1_large_tail_staging_{m6h,ep}_{native_ep8,remote_ep8}_*`。

EP16同K1/B512/D8、1620MHz、10/20、无并发构建或其它GPU任务的相邻对照：

|T/rank|M6h main/us|EP main/us|M6h wall/us|EP wall/us|
|---:|---:|---:|---:|---:|
|8192|1658.400|1649.410|2552.850|2529.620|
|20480|3866.020|3863.420|4738.720|4737.630|

32个原始字符串hash逐字一致，20K基本持平。此项未改善K4主矩阵，仅有单机K1
约0.7%的小收益，不优先保留或新增拓扑分支。源改动已撤回，保留冻结候选和
原始测量，未继续library/完整Torch/sanitizer。日志后缀`ep16_k1_t*_training_forward.log`。

## ES：训练共享网络本地完成prefix（数值通过，性能持平）

Forward的逐slot远端credit不变，本地完成改为唯一TX owner执行GIN flush并推进
channel已完成prefix。删除Forward未再使用的counter编号/分配/reset，Backward
仍有独立逐slot counter，不因多个live tapes而共用控制状态。完整协议、artifact
SHA及inference收益见[ring记录ES](static_forward_v4_ring_record.md)。

EP8/EP16 Torch四规模（20K热点）、1/64 OOM与重试、三份live tape/payload复用
通过；native EP8/EP16共96个梯度hash等于M6h/M6g2。EP16热点B16/D1的rank0/15
memcheck及Forward-filtered racecheck均0。未修改或宣称解决前文Backward
shared-atomic racecheck工具限制；附带graph检查是inference，不是训练graph。

1620MHz、B512/D8、10warmup/20samples，EP16保留tape的Forward：

|T/rank/顺序|M6h main/us|ES main/us|M6h wall/us|ES wall/us|
|---|---:|---:|---:|---:|
|8192，相邻|3480.990|3467.460|4283.240|4263.740|
|20480，相邻|8251.330|8306.850|9221.820|9268.270|
|20480，反向|8318.400|8272.610|9280.920|9222.650|

20K差异方向反转，属于本次测量波动，不报告训练收益。main与wall保持独立口径；
六份原始rank记录共96个Forward hash一致。计时节点无并发构建/GPU任务。
日志`batched_local_completion_{m6h,es_final}_ep16_t*_training_forward*.log`。
采用ES的理由是保持双credit合同并简化实现、改善inference，不是恢复了训练
Forward相对V3的差距；该项性能目标仍未完成。

### ET：Return Q16对训练Forward的回归

在已验收ES上仅将profile Return Qmax32降至16，保持Dispatch Q8/L32/B512/D8。
EP16四档训练与20K热点通过，短序列包含分配失败/重试及三份live tapes复用；
rank0/15热点B16/D1的memcheck为0，限定Forward/planner的racecheck为0。
没有重测或声称完整Backward racecheck clean。native EP8/EP16两次Backward
共96个梯度hash等于ES；EP8计算指令完全不变，复用ES完整Torch矩阵。

1620MHz、无并发构建/GPU任务、10warmup/20samples的eager retained-tape对照：

|T/rank|ES main/us|ET main/us|ES wall/us|ET wall/us|
|---:|---:|---:|---:|---:|
|8192|3461.790|3445.340|4282.190|4250.800|
|20480|8299.040|8300.320|9271.490|9263.080|

64个原始output_hash字符串一致。训练持平，不把小幅差异报告为收益；采用Q16
是因inference稳定小幅改善及缓冲区减少。主计算、含Notify/buffer的wall和
inference graph口径不同。日志`return_followup_{es,et}_ep16_t*_training_forward.log`。

### FB：source-major物理Mblock行序（无收益，撤回）

在ET上重测expert内source连续排布；canonical Notify行号和保存接口不变，
无额外padding。除inference快筛外，因checkpoint写回局部性可能变化，单独
测了保留tape的训练Forward。1620MHz、无并发构建/GPU任务、10/20：

|T/rank|ET main/us|FB main/us|ET wall/us|FB wall/us|
|---:|---:|---:|---:|---:|
|8192|3465.760|3474.590|4258.360|4275.310|
|20480|8328.380|8442.340|9296.990|9414.720|

64个Forward原始hash相同；20K main约退化1.37%，不采用训练专用行序。
只完成GPU几何检查/racecheck和上述Forward检查，没有继续完整tape/Backward
门禁，不能据此声明该候选的梯度已经验收。改动与测试参考已撤回，复用ET布局。
artifact与inference消融见[ring记录FB/FC](static_forward_v4_ring_record.md)。

### FJ2：三级FC1及跨GEMM共享内存退休（已验证，训练无收益）

原FJ的空间复用漏掉cluster两CTA阶段偏斜，prefill graph在racecheck下出现
数值错误，未采用。FJ2把FC2 scratch移到两种A extent之后，并在GEMM切换时
用已有empty barrier等待两个CTA的旧操作数读完；不增加SMEM或计数器。
原失败用例已通过，完整验证及机制见[ring记录FJ/FJ2](static_forward_v4_ring_record.md)。

EP8和EP16 native重复Backward的96个梯度hash与ES/M6h一致；两种拓扑的
1/64/8K/20K Torch训练矩阵已通过，20K为hotspot，短档含OOM/retry/live tapes。
FJ2双机decode热点、prefill热点训练memcheck均0，Forward专项racecheck两轮
通过，非ring长文也已回归；双机性能仍在测量，不声称完整Backward racecheck clean。

1620MHz、10warmup/20samples、eager retained-tape，第二节点单机EP8，
第一节点同时做Torch数值检查，计时节点独占且无编译：

|T/rank|ET main/us|FJ2 main/us|ET wall/us|FJ2 wall/us|
|---:|---:|---:|---:|---:|
|8192|1508.160|1502.460|2006.450|1999.240|
|20480|3635.360|3641.820|4217.450|4237.720|

32个字符串Forward output hash一致。两档差异小且方向不一，只判断训练
持平，不把inference的收益算到训练。日志
`prefill_fc1_stage3_v2_final_{et,fj2}_remote_ep8_t*_training_forward.log`。

双机独占、同频、无并发构建，正式对照补齐：

|T/rank/顺序|ET main/us|FJ2 main/us|ET wall/us|FJ2 wall/us|
|---|---:|---:|---:|---:|
|8192，ET→FJ2|3451.040|3452.350|4259.480|4262.680|
|20480，ET→FJ2|8282.620|8341.660|9226.900|9315.530|
|20480，FJ2→ET|8316.960|8340.100|9263.080|9343.920|

三组各32个原始hash字符串一致。8K持平，20K main两轮均回退（0.71%/0.28%），
wall也均回退（0.96%/0.87%），不能把这次stage调整当作训练优化。
暂不整体采用FJ2；FJ3将训练prefill FC1恢复两级，保留跨GEMM同步修正，
与inference三级配置独立消融。日志
`prefill_fc1_stage3_v2_{final,reverse}_{et,fj2}_ep16_t*_training_forward.log`。

FJ3只恢复训练prefill两级FC1，8K main ET3454.110/FJ33453.410us，20K
ET8319.460/FJ38358.400us；wall分别4266.660/4252.770及9279.800/9316.950us。
仍为1620MHz、独占10/20，各组32原始hash相同。20K仍慢0.47%，不保留这个
未带来收益的额外训练配置；未对FJ3执行完整梯度/sanitizer验收。
最终采用完整验收的FJ2同步/scratch修正及统一prefill三级FC1。此里程碑补全
跨CTA内存复用安全性，不是训练性能里程碑；约0.3%–0.7%的20K main回退及
训练Forward相对V3的总体差距均保持开放，不能被inference改善抵销。

### FM：从FC1的A stage保存X checkpoint（已验证，采用）

FJ2为基线。Pull不再向SavedX做第二次S2G，只写物理X及便宜的canonical
identity/gates。每个Mblock的FC1 Ntile0，从已ready的A stage以uint4 vector
scatter写canonical SavedX；复用原checkpoint.rows，不改物理行序、Notify、
G/U保存、tape格式或Backward。写入位于WGMMA commit之后、wait0和empty
arrival之前，沿用V3尾块的读者退休顺序。SavedX不复用ring payload，Forward
完成才允许Backward消费。不增加分配、barrier或计数器，复制总字节数不变。

checkpoint_copy模块改为计算侧stage helper；Pull复用已有普通row pipeline，
删除训练fanout wrapper。这个候选是vector scatter，不是V3完整tile的TMA
store，也不预设它一定快于通信warp逐行复制。

冻结`build/static_v4_math_x_checkpoint_fm/`，native SHA256
`482f535e85ea00ad89cb859b47345f092f9582309d34bd74331fc2d1b8816120`。
EP8 T512/K6/B512/D8与EP16 T64/K6/B16/D1重复Backward通过，共96个梯度
hash与FJ2完全相同。network prefill training STACK416（FJ2408），其它
prefill入口的REG/STACK/SHARED/LOCAL未变。尚未做完整Torch/sanitizer门禁。

全部构建结束后，第二节点独占EP8，1620MHz、eager retained-tape、10warmup/
20samples。首轮相邻对照如下，双机与反向复测仍待完成：

|T/rank|FJ2 main/us|FM main/us|FJ2 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|1505.120|1504.190|2012.400|2014.050|
|20480|3650.270|3547.900|4234.980|4120.930|

8K持平，20K main首轮改善2.80%、wall2.69%，仍不能凭单轮关闭训练性能目标。
日志`math_x_checkpoint_{fj2,fm}_remote_ep8_t*_training_forward.log`。

首轮EP8共32个字符串Forward hash一致。随后双机独占、无CPU构建/GPU任务
并发、同样1620MHz和10/20，完成相邻对照：

|T/rank|FJ2 main/us|FM main/us|FJ2 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|3457.920|3361.410|4254.760|4170.490|
|20480|8351.970|8051.840|9340.540|9031.110|

main分别改善2.79%/3.59%，wall1.98%/3.31%，64个原始Forward hash字符串
相同。日志`math_x_checkpoint_{fj2,fm}_ep16_t*_training_forward.log`。
尚需反向复测及完整训练门禁。FN会仅对SavedX尝试streaming store缓存提示，
保持所有地址、所有权、复制字节数及同步不变，独立比较，不预先认定收益。

单机20K反向FM→FJ2：main3547.650→3652.830us，wall4130.860→4244.480us，
对应改善2.88%/2.68%，16个字符串Forward hash一致，与首轮方向一致。
第二节点独占计时，第一节点同时编译FN，没有计时节点CPU构建；日志
`math_x_checkpoint_reverse_{fm,fj2}_remote_ep8_t20480_training_forward.log`。

### FN：SavedX streaming store（额外收益不稳定，撤回）

在FM上仅将SavedX vector store换成`st.global.cs.v4.u32`，尝试减少checkpoint
写流对活跃权重的缓存干扰。它只是eviction hint，不改变可见性或退休协议。
参见[NVIDIA PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#cache-operators)。
G/U、S、metadata、权重TMA及行序均不改。冻结
`build/static_v4_streaming_x_checkpoint_fn/`，native SHA256
`caf9083a0f867e417aa84d3fa9765ffa9ff910563d6b03a52844ba25517164d8`。
EP8/EP16重复Backward共96梯度hash与FJ2相同，四个prefill入口资源与FM相同。

1620MHz、无并发构建/GPU任务、eager retained-tape 10/20，相邻FM→FN：

|拓扑/T/rank|FM main/us|FN main/us|FM wall/us|FN wall/us|
|---|---:|---:|---:|---:|
|EP8/8192|1501.920|1503.330|1996.440|2006.580|
|EP8/20480|3546.780|3542.980|4129.840|4124.360|
|EP16/8192|3383.010|3351.140|4188.540|4156.910|
|EP16/20480|8064.060|8080.000|9007.420|9058.950|

EP8共32个字符串hash、EP16共64个原始uint64 hash字符串相同。单机基本持平，
双机8K首轮约改善0.94%，20K却回退约0.20%，不保留缺乏稳定额外收益的
特殊PTX store。FN未进行library/Torch/sanitizer完整门禁。源码恢复普通store的
FM后，patch与冻结FM逐字节相同；FM进入完整门禁，尚未提前提交为验收优化。
日志`streaming_x_checkpoint_{fm,fn}_{remote_ep8,ep16}_t*_training_forward.log`。

双机20K在FM→FN之后又测FJ2（间隔一个FN批次，不称FM/FJ2严格相邻）：
FJ2 main8324.990us、wall9299.890us，对照FM8064.060/9007.420us，改善
3.13%/3.14%。32个原始hash相同，与前一轮FJ2→FM的3.59% main改善方向一致。
日志`streaming_x_checkpoint_fj2_ep16_t20480_training_forward.log`。

### FM最终验收与同期V3对照

FM library SHA256 `44820160cb808ed7aaff376b244214ea989f48ab91438444a7784b842450892a`，
extension仍为`725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
单机完整训练四档通过：1/64每档8 parity、16 preparation-failure、8 retry、
24 live-tape reuse、8 inference graph及8 offload；8K balanced/20K hotspot
每档8 parity/graph/offload。Inference四档+edges 240条、tiny B16/D1 K8 152条、
非ring8K/20K 32条检查通过。双机完整门禁仍在进行，未提前声称验收。

同轮补测冻结V3，1620MHz、remote单机EP8独占、eager retained-tape 10/20：

|T/rank|V3 main/us|FM main/us|V3 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|1334.270|1497.950|1732.580|1995.680|
|20480|3138.820|3537.540|3578.370|4118.170|

32个Forward hash一致。FM相对V3 main仍慢12.27%/12.70%，wall慢15.19%/15.09%；
checkpoint迁移只缩小差距，不代表恢复了训练性能。计时节点独占，另一节点
同时构建FM库；没有计时节点CPU构建。日志
`math_x_checkpoint_final_{v3,fm}_remote_ep8_t*_training_forward.log`。

双机完整门禁于20:13 UTC完成，launcher退出0。训练1/64每档16 parity、
32 preparation-failure、16 retry、48 live-tape reuse、16 inference graph及
16 offload；8K balanced/20K hotspot每档16 parity/graph/offload。
Inference四档+edges共480条，tiny B16/D1 K8共304条通过。复用现有测试，
没有为本次地址迁移增加测试脚本。

rank0/15小ring T64热点训练memcheck（PID87941/359880）、T2048 prefill热点
训练memcheck（92771/360845）均0 errors；对应Forward racecheck
（90281/360269及94753/361236）均0 hazards/0 warnings，应用parity/graph/
offload检查全部完成。racecheck只过滤planner/Forward，不宣称完整Backward
raceclean。日志`math_x_checkpoint_fm_torch_ep16_*`及同前缀sanitizer日志。
双机正式性能矩阵在完整门禁结束、两节点GPU空闲后启动，无并发构建。

双机同轮冻结V3对照完成，1620MHz、独占、eager retained-tape 10/20：

|T/rank|V3 main/us|FM main/us|V3 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|2912.740|3366.690|3423.030|4175.610|
|20480|6648.130|8063.940|7177.980|9033.290|

64个原始uint64 Forward hash字符串一致，实际shape及16个rank的retained
标志均核实。FM相对V3 main仍慢15.58%/21.30%，wall慢21.99%/25.85%，不因
相对FJ2的小幅改善关闭训练性能目标。日志
`math_x_checkpoint_final_{v3,fm}_ep16_t*_training_forward.log`。
20K FM这一轮main范围7999.840–43197.100us，p95为8148.160us，mean9821.516us；
有一个43ms长尾，未删除样本、未只报告较好轮次。另行重测FM确认稳定值。

随后的同条件FM独占重测，20K main p50=8023.330us，范围7967.300–8118.820us，
p95=8104.800us；wall p50=9006.470us，16个原始hash与首轮相同。没有再次出现
43ms长尾，但无法仅凭复测确定其原因，首轮及全部样本继续保留；不以这轮较小
数值替换上表。日志`math_x_checkpoint_repeat_fm_ep16_t20480_training_forward.log`。

最终采用通过完整门禁的普通store版FM，源码patch与冻结artifact逐字节一致。
优化的是训练SavedX复制的执行位置，没有inference新增收益；正式inference
四档回归及对应V3数据见ring记录FM。后续将地址计算移出K循环的FO是独立候选，
不计入本里程碑，完整训练性能及Backward专项工具限制仍保持开放。

### FO：FC1 SavedX目的地址移出K循环（已采用）

独立基于FM。检查FM的local training prefill SASS发现，FC1每个K迭代重复
读取8个checkpoint row及计算64位目的地址。将其改为Ntile0每task准备一次
线程私有目的指针，仍从相同ready A stage保存相同uint4；固定vector数并
显式约束math thread索引。保留valid-row/hidden-tail检查，所有同步、行序、
布局与复制字节数不变。Inference的未使用对象应由编译器消除，待检查资源及
生成指令。冻结`build/static_v4_checkpoint_address_fo/`，尚未报告收益。

native SHA256 `fe87231b00287792081fa38a457048fb9e99a4350bc2333cb564b620e1d52795`。
EP8 T512/K6/B512/D8及EP16 T64/K6/B16/D1重复Backward共96梯度hash与FM相同。
network prefill training STACK416→408，local training72、inference local64/
network392不变，REG168/SHARED2048/LOCAL0不变；未新增Forward WGMMA序列化
警告。SASS确认K循环内不再加载checkpoint rows及计算row-stride乘积，8个
目的指针在寄存器而非local数组中；仍保留原thread-index的单次stack reload。

全部构建结束后独占计时、1620MHz、B512/D8、retained-tape 10/20，FM→FO：

|拓扑/T/rank|FM main/us|FO main/us|FM wall/us|FO wall/us|
|---|---:|---:|---:|---:|
|EP8/8192|1499.360|1432.100|2008.280|1932.100|
|EP8/20480|3546.400|3446.400|4125.110|4022.110|
|EP16/8192|3370.500|3319.740|4180.620|4114.800|
|EP16/20480|8046.530|7934.210|9026.540|8888.460|

EP8 main改善4.49%/2.82%，EP16改善1.51%/1.40%；wall也改善。EP8的32个字符串
hash、EP16的64个原始uint64 hash字符串一致，pool不变。尚需反向复测及完整
Torch/sanitizer门禁，不提前采用。日志
`checkpoint_address_{fm,fo}_{remote_ep8,ep16}_t*_training_forward.log`。

后续正反向、完整Torch/Forward专项sanitizer及四档inference回归已通过，
采用这项不增加内存/协议状态的改动。细节和后续FP容量实验交叉对照见下文。
收益不是所有规模固定百分比：双机20K反向仅0.47%、wall持平，保留该结果。
未改变默认B/D或加入训练自动阈值；inference SASS与FM相同。

### FP：训练窗口容量独立调节（验证中，不改变inference配置）

用冻结FM，无源码修改。第二节点单机EP8独占，T20K、1620MHz、retained-tape
10/20；第一节点同时构建FO，没有计时节点CPU构建。按下表顺序测试：

|B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|512/8|3550.530|4152.580|408121664|
|1024/4|3381.500|3984.130|575356480|
|2048/2|3850.980|4495.860|910853696|
|512/8复测|3557.150|4153.520|408121664|

32个Forward hash字符串一致，actual shape及retained flags已核实。B×D相同
不代表总pool相同：更大的整窗口需要更多compute payload容量。B1024/D4
main首轮约改善4.76%，但多167234816 bytes/rank；B2048退化8.46%。仍需
补测8K、分离B/D影响及双机，不先保留训练专用配置，也不把容量调节收益算作
FO地址计算优化。日志`training_window_fp_remote_ep8_b*_d*_r*.log`。

同一remote独占/本地FO编译条件下，补测8K并分离20K的B/D影响，按表中顺序：

|T/rank，B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|8192，1024/4|1453.020|2002.630|481305408|
|8192，512/8|1502.980|1999.050|313659456|
|20480，1024/8|3392.900|3992.890|642596672|
|20480，512/4|3602.690|4190.660|374501440|
|20480，1024/4复测|3383.870|3981.300|575356480|

40个字符串hash与FM对应规模相同。8K main约快3.32%，但wall持平，不称端到端
改善；20K B1024/D4复测与首轮方向一致，增加D8无额外收益。B512/D4反而慢于
B512/D8，不能把B和D视作可互换的相同容量。尚未对B1024训练做完整Torch或
双机性能门禁，不改变验收profile；所有计时来自冻结FM，与FO代码实验独立。
日志`training_window_fp_followup_remote_ep8_t*_b*_d*.log`。

随后两节点独占、无构建/GPU并发，1620MHz同样10/20；每个规模先测FO B512/D8
作反向对照，再测FM B512/D8及FM B1024/D4：

|T/rank，版本/B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|8192，FO/512/8|3290.110|4116.140|702615744|
|8192，FM/512/8|3374.500|4189.520|702615744|
|8192，FM/1024/4|3034.530|3872.850|870256832|
|20480，FO/512/8|7966.910|8948.170|819971520|
|20480，FM/512/8|8004.640|8958.050|819971520|
|20480，FM/1024/4|7451.260|8437.640|987170496|

各规模48个原始hash相同。FO反向main改善2.50%/0.47%，20K比首轮1.40%的
改善小，wall几乎持平，保留波动而不只选最好轮次。FM的窗口调节main改善
10.07%/6.91%，wall约7.56%/5.81%，但每rank额外约168/167 MB；仍需完整
B1024训练正确性，不能视为免费优化。日志
`checkpoint_address_reverse_fo_ep16_t*_training_forward.log`及
`training_window_fp_ep16_t*_b*_d*.log`。

FO进入library构建时，在另一节点独占补测组合及反向（无计时节点CPU构建）：

|单机T/rank，B/D|FO main/us|FM main/us|FO wall/us|FM wall/us|
|---|---:|---:|---:|---:|
|8192，1024/4|1440.510|1451.620|1976.270|1985.590|
|20480，1024/4|3346.850|3378.300|3945.400|3975.050|
|20480，512/8|3439.900|3546.850|4017.810|4127.270|

48个字符串hash相同，pool与对应FM几何一致。相同B1024时FO仅额外改善
0.77%/0.93%；B512/20K反向仍改善3.02%。不能把FO与FP首轮收益相加。单机
8K的FO B512（上一组1432.100/1932.100us）已经优于较大窗口，20K则B1024
更快；不据两个离散规模引入未经测试的自动阈值。日志
`checkpoint_window_{fo,fm}_remote_ep8_t*_b*_d*.log`。

FO与FM的local/network inference prefill完整SASS dump逐字节相同，未使用的
checkpoint对象已消除。该检查在第一节点进行，第二节点独占计时；仍执行
inference数值与正式矩阵回归，不将FO训练收益误记为inference改善。

FO单机正式inference回归（1620MHz、graph 10 warmup/20 samples/batch20）：
T1/64/8192/20480分别100.970/135.322/1249.680/2962.120us；同批冻结V3
长文1487.020/3299.810us。32个FO hash与FM一致。第二节点独占计时，第一
节点构建FO library；无计时节点CPU构建。日志
`checkpoint_address_final_{v3,fo}_remote_ep8_t*_formal.log`。Inference代码
相同，数据仅作回归，不归因于FO。

FO的EP8完整Torch检查已通过：T1/64的OOM/retry及三个live tapes、8K均匀/
20K热点的训练数值、inference四规模与tiny-ring/非ring回归；另外B1024/D4
的8K均匀/20K热点训练也通过。随后EP16相同训练四档、inference四档及tiny
ring全部通过；B1024/D4的8K均匀/20K热点训练亦通过。rank0/15的T64小ring
memcheck（134575/373948）和T2048热点训练memcheck（136819/374903）均为
0 errors；Forward专项racecheck（135213/374328、138431/375278）均为
0 hazards/0 warnings。不称完整Backward racecheck clean。

Library SHA256 `b36ee66e81ab7515c3d13573ae4df9a7162596d9e728af6a0bd2978b4b2f0cdb`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
20:54负载下16张GPU均SM/graphics1620MHz、memory3201MHz（memory仅观测）。
正式双机性能回归及B1024组合复测随后单独进行。

FP再以FO固定源码扫20K的中间窗口，按下表顺序，第二节点单机独占，第一
节点同时执行FO Torch正确性（无CPU构建）；同为1620MHz、retained-tape10/20：

|B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|1024/4|3344.380|3967.880|575356480|
|768/5|3354.020|3941.740|487722560|
|1280/3|3383.260|3990.030|654976832|
|1536/3|3405.440|4030.660|752630848|
|1024/4复测|3348.290|3964.540|575356480|

40个hash一致。B768近似持平且少87.6MB，继续增大到B1280/1536无收益；
B×D并不全部相同，不称等容量对照。B768尚无完整Torch门禁，不据此自动选
配置。日志`training_window_fp_fo_sweep_remote_ep8_b*_d*_r*.log`。

FO完整双机门禁之后，两节点独占且无构建，1620MHz、retained-tape10/20，
每规模按V3→FM→FO补测组合。FM/FO均B1024/D4：

|T/rank|V3 main/us|FM main/us|FO main/us|V3 wall/us|FM wall/us|FO wall/us|
|---|---:|---:|---:|---:|---:|---:|
|8192|2916.670|3027.490|2980.130|3435.810|3850.650|3806.700|
|20480|6635.900|7459.300|7348.100|7155.280|8428.500|8313.000|

96个原始uint64 hash字符串相同，各版本shape/retained/10warmup/20samples
已核对。相同窗口下FO main额外改善1.56%/1.49%，FM/FO pool完全相同；该窗口
的容量代价仍是上文每rank约168/167MB，不能忽略。组合FO相对V3 main仍慢
2.18%/10.73%、wall慢10.79%/16.18%。不据此关闭训练性能目标，也不更改
inference默认B512/D8或引入训练自动阈值。日志
`checkpoint_window_final_{v3,fm,fo}_ep16_t*_training_forward.log`。

FO代码里程碑提交`dd7d817`；完整inference正式矩阵另见ring记录FO，性能与
FM持平。FP是独立、显式的运行时容量选择，不是新增代码优化。

### FQ：训练稳定物理行序（验证中，TMA前置消融）

仅训练planner复用Notify的canonical `route_dst_row`。每个source window
在共享scratch中对各expert取最小行号，以`canonical-min`代替atomic reservation
得到稳定序号；同expert的连续token窗口在Notify稳定序列中是连续子段，重复
expert slots也保持不同序号。生产训练不含masked routes，generic/inference
planner保持原有行为。Destination物理行按source-rank拼接，canonical tape
布局、Backward顺序和固定combine顺序不变。

不增加GM metadata、交换量或pool；E64训练plan仅多约512B共享scratch及对齐，
main共享内存不变。先冻结仍采用FO uint4复制的版本，隔离行序影响，再测试
成块TMA checkpoint；此前FB仅source-major但source内部仍atomic乱序，两者
不是同一实验。不预设收益。冻结目录
`build/static_v4_canonical_training_plan_fq/`，native构建开始21:03。

Native SHA256 `29f6138208e763156f9d1b06e6faaf0224db3145537ce8eaa31a3df6a98e7276`。
EP8 T512/K6/B512/D8、EP16 T64/K6/B16/D1双次Backward的96个梯度hash与FO
相同。local planner REG80→86，network planner REG107/STACK72不变；main
训练/推理资源不变。local/network inference prefill完整SASS逐字节等于FO。

两节点独占、无构建，1620MHz、B512/D8、retained-tape10/20，FO→FQ：

|T/rank|FO main/us|FQ main/us|FO wall/us|FQ wall/us|
|---|---:|---:|---:|---:|
|8192|3324.540|3337.020|4150.280|4156.780|
|20480|7995.620|7987.520|8987.140|9005.200|

64个原始hash字符串与pool一致。单独稳定行序没有明显双机收益，保留为下一项
TMA复制的隔离对照，不独立采用。日志
`canonical_training_plan_{fo,fq}_ep16_t*_training_forward.log`。

随后第一节点编译FR，第二节点独占补测EP8（计时节点无CPU构建），同样
B512/D8、1620MHz、retained-tape10/20、FO→FQ：

|T/rank|FO main/us|FQ main/us|FO wall/us|FQ wall/us|
|---|---:|---:|---:|---:|
|8192|1427.680|1376.060|1926.140|1873.990|
|20480|3449.790|3303.490|4036.300|3884.300|

32个字符串hash及pool相同。单机main改善3.62%/4.24%，wall也改善，说明
稳定物理行序本身有可测单机收益，不能把未来组合收益全算给TMA。仍未完成
FQ完整Torch/sanitizer，不提前采用；FR若不划算，再考虑独立验证FQ。
日志`canonical_training_plan_{fo,fq}_remote_ep8_t*_training_forward.log`。

### FR：M32连续SavedX的TMA写回（单机改善，全拓扑方案未采用）

在FQ上识别物理Mblock内完整、对齐的32行段：canonical行严格递增，因此首尾
差31即证明整段连续。复用现有`saved_x_map`为ring编码64列×32行的BF16 box，
非ring仍用原M128 map；不新增map数组、GM段表或payload。连续段由FC1 Ntile0
的thread0从ready A stage发TMA，每个N64 swizzle atom分别复制；其余行沿用
FO uint4 scatter，二者不重叠。`tma_mask`在task开始计算，不在K循环扫描行号。

每K迭代提交一个bulk group，WGMMA wait0后thread0等完整global completion，
再通过首warp的`__syncwarp`交接给lane0/1的两CTA empty到达。没有新增全局/
cluster barrier或计数器；不能只等WGMMA或TMA的SMEM read completion。当前
ShareA由CTA0发全部A multicast，并非两CTA各发一半；warp join是显式完成交接，
不是在原版中复现了race的结论。SavedX仍先于FC1完成发布，tape生命周期不变。

CUDA12.9的[Tensor Map约束](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
允许该box；[PTX swizzle布局](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#tensor-swizzling-modes)
的128B模式以1024B重复，M32/N64 atom偏移4096/16384B保持原有对齐。这是布局
推导，仍须数值及sanitizer实测。先只实现M32单map以隔离效果，不预设收益；
可能受额外寄存器、issuer开销和小TMA写回延迟影响。
冻结`build/static_v4_tma_x_checkpoint_fr/`，21:12开始native构建；第一节点
编译时，第二节点独占补测FQ单机性能，不在计时节点编译。

Native SHA256 `816f36b325e4224b2b56e37e62c7fa2b6104c0340dd7407d56d6d95aacdcba15`。
EP8 T512/K6/B512/D8、EP16 T64/K6/B16/D1重复Backward共96个梯度hash与FQ
相同。local prefill training STACK72→80，network仍408；REG168/SHARED2048/
LOCAL0不变，未新增Forward WGMMA序列化警告。两个inference prefill完整SASS
仍与FO/FQ逐字节相同。训练SASS确认SavedX K循环有8处UTMASTG.2D及完成等待，
随后开始独占性能对照；还没有完整Torch/sanitizer门禁，不提前采用。

两节点独占、无构建、1620MHz、B512/D8、retained-tape10/20，FQ→FR：

|T/rank|FQ main/us|FR main/us|FQ wall/us|FR wall/us|
|---|---:|---:|---:|---:|
|8192|3319.780|3321.280|4138.790|4121.440|
|20480|8106.270|8010.780|9058.420|8982.970|

64个原始hash字符串及pool相同。8K持平，20K本轮main快1.18%；但FQ20K的
前轮为7987.520us，本轮baseline有所波动，不能只看这一轮百分比就采用。
日志`tma_x_checkpoint_{fq,fr}_ep16_t*_training_forward.log`。

本机编译FR2时，第二节点独占补测单机EP8（无计时节点CPU构建），同样
1620MHz、B512/D8、retained-tape10/20、FQ→FR：

|T/rank|FQ main/us|FR main/us|FQ wall/us|FR wall/us|
|---|---:|---:|---:|---:|
|8192|1377.950|1360.740|1882.920|1857.840|
|20480|3299.680|3235.840|3891.510|3848.890|

32个字符串hash及pool相同。单机main额外改善1.25%/1.93%，不能将FQ已有的
3.6%–4.2%行序收益再记一次。日志
`tma_x_checkpoint_{fq,fr}_remote_ep8_t*_training_forward.log`。

### FR2：区分SMEM读者退休与tape写回完成（无稳定额外收益，撤回）

基于冻结FR，仅分开两个完成条件。每stage用`wait_group.read 0`确认TMA不再
读取SMEM，经同一warp join交给两CTA empty；task结束仍由issuer执行完整
`wait_group 0`，位置在FC1 epilogue之后、现有threadfence/math join及fc1_done
发布之前。SavedX与S/G/U独立，允许其写回与epilogue重叠；没有去掉最终全局
完成检查，也不依赖kernel退出替代它。

[PTX等待语义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
明确区分source读取完成和destination写入可见。FR2利用该区别缩短stage的
保守等待，仍须验证完整slot/tape生命周期，不以hash相同代替同步证明。
源码冻结于`build/static_v4_tma_x_checkpoint_fr2/`；21:19本机开始构建，另一
节点独占测试冻结FQ/FR单机性能。未增加buffer/map/metadata/counter。

Native SHA256 `5fa5395390eef54caf8e919b07fa8c4ccbf2a13a3811585d9d566199f0ab46df`。
EP8/EP16相同native夹具的96个梯度hash与FR/FQ一致。local training STACK80
不变，network408→416；REG168/SHARED2048/LOCAL0不变，两种inference prefill
SASS仍逐字节相同。SASS确认K循环的等待后不再每次CCTL.IVALL，task末尾仍有
完整等待及invalidation。是否抵消新增live-range/stack代价，由独占对照决定。

两节点独占、无构建，1620MHz、B512/D8、retained-tape10/20，FR→FR2：

|T/rank|FR main/us|FR2 main/us|FR wall/us|FR2 wall/us|
|---|---:|---:|---:|---:|
|8192|3326.370|3318.050|4139.030|4118.160|
|20480|8097.860|8048.770|9057.820|8995.540|

64个原始hash字符串与pool相同。额外main改善0.25%/0.61%，幅度较小，尚需
反向对照与完整门禁，不先采用。日志
`tma_x_checkpoint_fr2{,_baseline_fr}_ep16_t*_training_forward.log`。

单机EP8首轮（第二节点独占，本机并行构建FR2 library；无计时节点CPU构建），
同样1620MHz、B512/D8、retained-tape10/20，FR→FR2：

|T/rank|FR main/us|FR2 main/us|FR wall/us|FR2 wall/us|
|---|---:|---:|---:|---:|
|8192|1355.550|1364.700|1852.500|1861.310|
|20480|3245.760|3229.630|3834.900|3805.120|

8K main回退0.68%，20K改善0.50%，方向不一致，立即按FR2→FR反向复测。
还不能确认分离等待比FR更好，不先采用。日志
`tma_x_checkpoint_fr2{,_baseline_fr}_remote_ep8_t*_training_forward.log`。

同一remote独占/本机FR2 library构建条件下，反向FR2→FR：

|T/rank|FR2 main/us|FR main/us|FR2 wall/us|FR wall/us|
|---|---:|---:|---:|---:|
|8192|1354.180|1355.550|1852.290|1855.910|
|20480|3229.700|3232.380|3811.470|3810.140|

32个字符串hash及pool相同。反向8K/20K main仅差0.10%/0.08%，未证明分离
等待有稳定额外收益；结合双机微小差异，不保留额外生命周期分支。已撤回FR2，
恢复FR逐stage完整global wait及warp join，源码patch与冻结FR逐字节相同。
FR2仅完成native及性能快筛，library虽已构建但没有运行完整Torch/sanitizer，
不称其已通过完整门禁。FR2库SHA256
`4937560a97806ccd087aeb081bc40191c608714bcf708d76ff680b6612e10c57`；相关源码/
binary/log均保留。反向日志
`tma_x_checkpoint_fr2_reverse_{fr2,fr}_remote_ep8_t*_training_forward.log`。

### FR：完整验收与全拓扑取舍

后续选择FQ+FR组合进行完整验证，尚未采用；FR library于21:33重新构建，
另一节点独占补测FO/FR及V3训练、B512/D8与B1024/D4容量组合。

完整候选库构建期间，第二节点独占、无计时节点CPU构建，1620MHz、retained-
tape10/20；每规模按表序V3→FO512→FR512→FO1024→FR1024：

|T/rank|版本/B/D|main/us|wall/us|pool bytes/rank|
|---|---|---:|---:|---:|
|8192|V3|1334.690|1741.160|735672000|
|8192|FO/512/8|1435.840|1933.240|313659456|
|8192|FR/512/8|1357.820|1856.730|313659456|
|8192|FO/1024/4|1433.380|1957.240|481305408|
|8192|FR/1024/4|1415.300|1974.470|481305408|
|20480|V3|3131.620|3567.740|1820708544|
|20480|FO/512/8|3445.310|4023.140|408121664|
|20480|FR/512/8|3247.200|3828.970|408121664|
|20480|FO/1024/4|3349.500|3966.290|575356480|
|20480|FR/1024/4|3254.210|3856.210|575356480|

80个字符串hash一致，actual shapes及10/20已核对。FR包含FQ行序和TMA两项，
B512相对FO的main改善5.43%/5.75%，不能再叠加FQ/FR分项收益。FR B512仍比
V3 main慢1.73%/3.69%，wall仍落后；B1024在当前FR单机没有额外收益且更占
pool，维持默认B512/D8，不自动扩大窗口。这里pool不含独立tape，不将差额
等同于全部训练内存节省。日志
`tma_x_checkpoint_final_{v3,fo,fr}_remote_ep8_t*_b*_d*_training_forward.log`。

随后第一节点执行FR完整Torch，第二节点独占反向FR→FO，B512/D8、同样1620MHz
及10/20：8K main1357.340/1430.820、wall1854.080/1935.590us；20K main
3230.750/3441.150、wall3816.590/4019.520us。32个hash及pool相同，反向main
改善5.14%/6.11%，与正向方向一致。日志
`tma_x_checkpoint_reverse_{fr,fo}_remote_ep8_t*_training_forward.log`。

FR library SHA256 `936622e8b32bbbdba0f722c0947c4a35999a437525d611982e573629c1b536cc`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
源码与冻结native patch一致。EP8完整Torch九项已通过：四档训练、20K热点、
短档OOM/retry及三个live tapes，inference完整240/tiny152/非ring32条检查，
另外B1024/D4的8K均匀/20K热点训练各通过。双机完整检查随后启动，尚未采用。

正式EP8 inference graph 10warmup/20samples/batch20、1620MHz：FR四档为
101.000/135.282/1247.670/2953.140us，同期V3长文1488.050/3299.380us。
32个FR hash与FO相同，shape/10/20/batch20已核实；第二节点独占计时，第一
节点先构建库、再执行Torch正确性。Inference SASS相同，仅作回归，不记为
新增inference收益。日志`tma_x_checkpoint_final_{v3,fr}_remote_ep8_t*_formal.log`。

21:55双机完整十二项检查exit0，24份node日志的rank级PASS数量全部核对：
四档训练及20K热点、短档OOM/retry/三个live tapes、inference完整480/tiny304，
另有B1024/D4的8K均衡/20K热点训练。T64/B16/D1与T2048/B512/D8热点训练的
rank0/15 memcheck各0 errors；同两档限定planner/main的Forward racecheck
各0 hazards/0 errors/0 warnings，应用数值/graph/offload也均完成。这里graph
是既有inference graph复用检查，不代表训练Backward支持graph或racecheck全净。
内存日志PID181548/391066、185605/392009；race日志183553/391440、188122/392389。
21:52负载下16张卡SM/graphics均1620MHz、memory观测3201MHz，后者未锁定。
源码patch再次与冻结native逐字节核对一致。全部检查后才开始两机独占的正式
inference及V3/FO/FR训练对照，不并行CPU构建或其它GPU工作。

最终双机独占、无构建，1620MHz、retained-tape10/20；每规模按表序
V3→FR512→FO512→FR1024→FO1024，刻意反向candidate→baseline：

|T/rank|版本/B/D|main/us|wall/us|pool bytes/rank|
|---|---|---:|---:|---:|
|8192|V3|2901.150|3410.570|1079452608|
|8192|FR/512/8|3329.310|4132.630|702615744|
|8192|FO/512/8|3313.500|4108.800|702615744|
|8192|FR/1024/4|3075.300|3911.600|870256832|
|8192|FO/1024/4|2963.100|3795.660|870256832|
|20480|V3|6660.160|7192.790|2167634880|
|20480|FR/512/8|8059.070|9019.420|819971520|
|20480|FO/512/8|7982.430|8930.300|819971520|
|20480|FR/1024/4|7239.900|8201.960|987170496|
|20480|FO/1024/4|7351.300|8327.940|987170496|

160个原始hash字符串与同期V3逐rank相同，实际shape/world/10/20/retained
均核实，FO/FR相同B/D的pool相等。B512 main反向慢0.48%/0.96%，没有稳定
网络收益；B1024的8K慢3.79%，FR范围3028.640–3152.860us、FO范围
2931.620–2996.060us不重叠，wall同样变慢。20K B1024则快1.52%，不能用它
掩盖8K回退，也不据此增加长度阈值。**不采用全拓扑FQ+FR组合**，后续只在
有稳定收益的单机specialization保留canonical/TMA，网络恢复FO。
日志`tma_x_checkpoint_final_{v3,fo,fr}_ep16_t*_b*_d*_training_forward.log`。

同一独占阶段的inference graph正式10/20/batch20：FR四档
268.560/317.882/2899.910/6890.180us；同期V3长文3348.090/7107.340us。
64个FR hash与FO相同，pool不变；shape及计数核实。这是未采用全拓扑FR的
inference回归证据，不能重标为后续FR3实测。日志
`tma_x_checkpoint_final_{v3,fr}_ep16_t*_formal.log`。

### FR3：只保留单机canonical/TMA checkpoint（采用）

已有`StaticLocal/Network`编译期分支选择：单机训练保持FQ稳定source顺序、
source-major物理行及FR M32 TMA；网络恢复FO的atomic source顺序、round-robin
物理行和uint4 SavedX。checkpoint helper的非contiguous特化不做TMA探测、
空指针过滤或wait/warp join；网络也不创建无用的SavedX tensor map。没有新增
运行时阈值/调参开关或另一套通信协议，pool和canonical tape布局不变。
须验证单机计算指令等于FR、网络计算/规划等于FO，不能凭源码直觉复用结论。
冻结`build/static_v4_local_tma_x_checkpoint_fr3/`；22:07在双机计时全部结束后
开始native构建；后续验收结果如下。

独立的FS规划融合已写出草案，但在FR网络回退确认后先完整撤回，保留unbuilt
patch；没有FS构建、性能或验收结果。待FR3稳定后再单独实施，避免混合归因。

FR3 native SHA256 `0761d1333ee43167a526af99ce1e8ec7998140bb1488770c544f0552c45c437b`。
EP8 T512/E16/K6/B512/D8与EP16 T64/E32/K6/B16/D1各重复Backward两次，
96个梯度hash与FR/FO逐rank相同。完整symbol SASS比较覆盖decode/prefill
各四个local/network×inference/training entry，以及两种training planner：
local training/plan逐字节等于FR，network training/plan逐字节等于FO，四种
inference entry等于FO。比较文件均含实际函数指令，不以空dump作证据。
Prefill local training STACK80、network408，inference64/392，REG168/
SHARED2048/LOCAL0不变。

本轮复用这些**完全相同代码**在FR/FO已有的专项sanitizer证据，不将旧日志重标为
FR3新跑；另做FR3 artifact的native、EP8完整Torch及EP16四档训练、OOM/retry/
live tapes、inference full/tiny与B1024长文集成回归。library构建与第二节点
独占EP8正式计时并行，计时节点没有CPU构建；双机计时仍等所有构建/检查结束。

FR3 library SHA256 `f0c4253e547c8a8cf78d0d4c6d9c1ade4f0ab7a2fe6295eae65bc78a92b06a7a`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
22:19 EP8完整九项Torch通过，包含四档训练、20K热点、短档OOM/retry/三个
live tapes、inference full240/tiny152/nonring32，以及B1024的8K/20K训练。
随后启动双机八项组件集成回归；本轮不重复字节相同kernel的sanitizer。

第二节点独占计时，第一节点构建FR3 library；1620MHz、B512/D8、retained-
tape10/20，FO→FR3：

|T/rank|FO main/us|FR3 main/us|FO wall/us|FR3 wall/us|
|---|---:|---:|---:|---:|
|8192|1423.390|1356.190|1922.000|1852.570|
|20480|3442.880|3234.370|4023.450|3814.580|

32个字符串hash及pool完全相同；main降低4.72%/6.06%，wall降低3.61%/5.19%。
FO8K有一个1850.110us高值，完整20samples保留，未删除该点；上述均为p50。
pool仍313659456/408121664 bytes/rank，不含独立tape。日志
`local_tma_x_checkpoint_{fo,fr3}_remote_ep8_t*_training_forward.log`。
同阶段正式inference10/20/batch20：FR3四档101.075/135.166/1251.630/
2957.950us，同期V3长文1489.390/3300.590us；48个hash及实际shape/计数
已核对，inference持平，不算新增收益。日志
`local_tma_x_checkpoint_final_{v3,fr3}_remote_ep8_t*_formal.log`。

22:27 FR3双机八项组件集成检查exit0，16份node日志的rank级PASS数量核实：
四档训练、20K热点、短档OOM/retry/三个live tapes、inference full480/tiny304，
以及B1024的8K/20K训练均通过。未运行FR3新sanitizer，复用前述字节相同
local FR/network FO代码的检查；不宣称完整Backward racecheck clean。
全部库构建、Torch检查结束且两机GPU空闲后，才启动FO→FR3双机同容量性能
对照及FR3正式inference回归；无并发CPU构建或其它GPU任务。

双机最终同容量对照，1620MHz、retained-tape10/20，每规模依次FO512→FR3512→
FO1024→FR31024，单位us：

|T/rank|B/D|FO main|FR3 main|FO wall|FR3 wall|pool bytes/rank|
|---|---|---:|---:|---:|---:|---:|
|8192|512/8|3316.610|3327.460|4110.980|4122.670|702615744|
|8192|1024/4|2991.840|2988.420|3824.130|3816.030|870256832|
|20480|512/8|8009.280|7979.550|8972.530|8963.560|819971520|
|20480|1024/4|7348.030|7345.540|8315.100|8320.020|987170496|

128个原始hash字符串与本轮FR阶段的同期V3逐rank一致，实际shape/world/10/20/
retained及同容量pool均已核对。网络结果持平，不将不足0.4%的差异算作收益；
全拓扑FR在B1024/8K的3.79%回退已消除。pool不含独立tape。日志
`local_tma_x_checkpoint_{fo,fr3}_ep16_t*_b*_d*_training_forward.log`。

同阶段正式EP16 inference graph 10/20/batch20：FR3四档269.571/317.544/
2897.900/6878.520us，同期V3长文3353.190/7102.160us。64个FR3 hash与FR
逐rank相同，长文同时与本次V3相同；pool/注册bytes、实际shape及计数已核实。
日志`local_tma_x_checkpoint_final_{v3,fr3}_ep16_t*_formal.log`。
Inference仍持平，不记为本次新增收益；双机20K相对V3仅降低3.15%，不能关闭
明显领先目标。采用FR3，拒绝通用FR及FR2；训练端到端仍落后V3，下一步独立
检查规划launch、D2H和host同步，不把main计时范围变化误报为计算加速。

### FS：将训练物理窗口规划合入驻留main（不采用）

基线FR3 `06f6095`。materialized Notify已提供精确R/P，原先ring plan回传的R
只用于重复校验，不用于tape分配。候选保留Notify及collective tape准备边界，
在全部rank分配成功之后由main调用同一个ring planner，同时stage input；删除
独立planner entry、runtime D2H和一次host stream同步。发生分配失败时尚未
改变设备ring epoch/base，也未准入payload；保留全rank失败一致性及retry。
不改变独立tape所有权、canonical映射、GEMM、通信协议或training graph限制。

planner借用GEMM operand arena，校验scratch不会覆盖已初始化的barrier；没有
新GM allocation或双份planner。main开始时间现在覆盖物理规划，不能与FR3
不含规划的main数值直接比较，验收以含Notify、准备及Forward的wall为准。
冻结`build/static_v4_fused_training_plan_fs/`；22:42开始native构建，本节尚无
数值、性能或sanitizer结论，不将FR3检查重标为FS。

FS native SHA256 `c8e7ec2410fc5a39675b6ae86711fa1b7b864226ee922b72340793f0be00a52b`。
EP8 T512/E16/K6与EP16 T64/E32/K6两次Backward的96个梯度hash等于FR3；
四种inference完整SASS逐字节相同。训练decode/prefill的local/network STACK
均为64/392，REG168/SHARED2048/LOCAL0；栈减小本身不等于性能改善。

第二节点独占EP8，第一节点构建FS库；1620MHz、B512/D8、retained10/20，
FR3→FS，单位us。旧main不含物理规划，FS main包含，不能直接比较其百分比。

|T/rank|FR3 wall|FS wall|FR3 main|FS main|FR3 plan|FS plan|
|---|---:|---:|---:|---:|---:|---:|
|1|408.015|409.613|77.568|115.968|41.824|46.848|
|64|463.371|434.412|107.040|149.088|43.840|53.920|
|8192|1861.490|1830.280|1361.120|1433.570|89.728|116.032|
|20480|3809.090|3821.730|3233.410|3386.850|137.440|212.800|

64个hash及同shape pool相同。反向FS→FR3的8K wall1823.610/1864.500、
20K3834.430/3814.970us；main1430.270/1362.180、3396.100/3232.960us，
32个hash一致。8K wall两次改善1.68%/2.19%，20K两次反而慢0.33%/0.51%，
融合后的规划阶段也更长，不将其笼统称为训练加速。日志
`fused_training_plan_{fr3,fs}_remote_ep8_t*_training_forward.log`及
`fused_training_plan_reverse_{fs,fr3}_remote_ep8_t*_training_forward.log`。
待双机同容量对照后决定，不增加长度阈值掩盖20K结果。

最终双机独占且无构建，1620MHz、retained10/20、FR3→FS，同容量对照：

|T/rank|B/D|FR3 wall/us|FS wall/us|FR3 main/us|FS main/us|
|---|---|---:|---:|---:|---:|
|8192|512/8|4109.490|4059.780|3314.750|3482.560|
|8192|1024/4|3822.250|3807.110|2984.190|3247.230|
|20480|512/8|8961.390|8933.330|7991.360|8345.570|
|20480|1024/4|8295.470|8180.320|7341.220|7589.660|

实际shape/world/10/20/retained已核实，128个原始hash字符串及各容量pool与FR3
相同。网络wall下降0.31%–1.39%，但单机20K正反向均小幅变慢，因此不采用
全拓扑规划融合，也不为这组微小差异引入长度开关。所有FS代码已撤回，冻结
patch保留。FS library已构建（SHA256
`57b4a805acc5f3c3a0bb888a5afccb467804d3b8bf436a10c21ee967c5b263e4`），
但**没有执行FS Torch/sanitizer完整门禁**，准备了脚本不等于验收。没有FS
inference正式性能数据；只有SASS相同证据及native训练hash检查。日志
`fused_training_plan_{fr3,fs}_ep16_t*_b*_d*_training_forward.log`。

### FS2：保留独立GPU规划，只消除host结果回传（采用）

直接基于FR3，不含FS融合。先按Notify实际R/P完成全rank tape准备，再在同一
stream顺序launch原来的独立planner和main，删除中间runtime D2H及host同步。
保留原384线程输入staging及全部GPU算法；仅移动host准备边界，避免将规划
资源/访存变化混进host同步实验。失败时仍无设备epoch/base/payload推进。
main计时口径与FR3相同，但收益仍以wall为主。没有新的buffer、runtime knob
或测试套件。冻结`build/static_v4_async_training_plan_fs2/`，22:55在上述双机
计时全部结束后才开始native构建；尚无FS2正确性或性能结论。

FS2 native SHA256 `20d4c0b80df703b7c917a7a1e06d845995f0863d85f5c1fc014829653d547705`。
EP8 T512/E16/K6与EP16 T64/E32/K6各两次Backward，96个梯度hash与FR3逐rank
一致。8种decode/prefill×local/network×inference/training main及2个独立
planner的完整SASS均逐字节等于FR3，dump均非空。Prefill训练STACK80/408、
REG168/SHARED2048/LOCAL0，与FR3相同。不是引用源码相似性来推断算法不变。
随后构建集成库，并在第二节点独占测量四档EP8；计时节点无CPU构建。

EP8首轮FR3→FS2，B512/D8、retained10/20、1620MHz；单位us：

|T/rank|FR3 wall|FS2 wall|FR3 main|FS2 main|pool bytes/rank|
|---|---:|---:|---:|---:|---:|
|1|407.037|378.599|78.272|71.232|17022528|
|64|463.149|435.704|105.984|100.672|45442880|
|8192|1855.280|1836.860|1358.300|1354.050|313659456|
|20480|3864.220|3794.070|3235.390|3229.890|408121664|

64个hash及pool一致，实际shape/world/10/20/retained核实。四档wall下降
6.99%/5.93%/0.99%/1.82%；main口径与FR3一致，但不把微小变化称为GEMM
代码优化。pool不含独立tape。日志
`async_training_plan_{fr3,fs2}_remote_ep8_t*_training_forward.log`。
尚需反向及双机对照和集成门禁，本表不是最终采用结论。

同样1620MHz/B512D8/10/20的反向FS2→FR3：8K wall1838.540/1859.180us，
20K3791.440/3847.430us，分别降低1.11%/1.46%；main1350.460/1359.490、
3226.530/3232.320us。32个hash和pool相同，未出现FS的20K回退。第一节点
先构建库、后执行Torch，第二节点独占计时。日志
`async_training_plan_reverse_{fs2,fr3}_remote_ep8_t*_training_forward.log`。

FS2 library SHA256 `4c07a0cb83da6596698064b9a7eba376d902a8979cb161765c3914ae2e962e84`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
23:06 EP8完整九项检查exit0：四档训练、20K hotspot、短档OOM/retry/三个live
tapes，inference full240/tiny152/nonring32，另有B1024/D4的8K/20K训练。
各文件rank级PASS数量逐项核实。随后开始双机十二项完整检查，不能提前称为
双机或sanitizer通过；现有graph检查不代表训练Backward graph支持。

同阶段正式EP8 inference graph10/20/batch20：FS2四档
101.083/135.254/1256.700/2949.580us，同期V3长文1487.820/3298.570us。
48个hash与FR3/同期V3相同，实际shape/计数及FS2 pool不变；这是inference
回归，代码相同、性能持平，不能记为host训练优化新增收益。日志
`async_training_plan_final_{v3,fs2}_remote_ep8_t*_formal.log`。

23:19双机十二项完整检查exit0，24份node日志的rank级PASS数量全部核实：
四档训练、20K热点、短档OOM/retry/三个live tapes、inference full480/tiny304，
以及B1024/D4的8K/20K训练均通过。T64/B16/D1和T2048/B512/D8热点训练的
rank0/15 memcheck均0 errors；Forward planner/main过滤的racecheck两档均
0 hazards/0 errors/0 warnings。memcheck PID258203/415143、260883/416092，
racecheck259443/415524、261942/416467。不宣称完整Backward racecheck clean。
这轮是FS2 artifact的新检查，不是复用FR3日志。部分正确性检查与FT独立候选
的CPU预编译并行，不使用这些检查的时间值作性能数据。所有CPU构建及GPU
检查结束后，23:20才开始FS2两机独占的最终训练/正式inference对照。

双机独占、无CPU构建，1620MHz、retained10/20；每规模按表序，单位us：

|T/rank|版本/B/D|main|wall|pool bytes/rank|
|---|---|---:|---:|---:|
|1|FR3/512/8|172.608|630.524|390711488|
|1|FS2/512/8|168.416|626.920|390711488|
|64|FR3/512/8|226.752|708.788|419244736|
|64|FS2/512/8|215.360|689.455|419244736|
|8192|V3|2922.300|3428.790|1079452608|
|8192|FR3/512/8|3307.840|4109.420|702615744|
|8192|FS2/512/8|3316.770|4128.670|702615744|
|8192|FR3/1024/4|2994.370|3822.300|870256832|
|8192|FS2/1024/4|2979.070|3814.420|870256832|
|20480|V3|6658.910|7181.200|2167634880|
|20480|FR3/512/8|7989.570|8938.050|819971520|
|20480|FS2/512/8|7965.410|8926.920|819971520|
|20480|FR3/1024/4|7345.700|8328.090|987170496|
|20480|FS2/1024/4|7352.830|8299.000|987170496|

224个原始hash字符串逐rank一致，长文与本次V3相同；实际shape/world/10/20/
retained及同B/D pool均核实。短档wall约持平/改善2.73%，长文多数差异小于
0.5%；8K B512的FS2 wall反而慢0.47%。不能将单机收益直接推广到网络，
仍需默认窗口的反向对照；不丢弃不利样本。pool不含独立tape。日志
`async_training_plan_{v3,fr3,fs2}_ep16_t*_b*_d*_training_forward.log`。

同阶段正式EP16 inference graph10/20/batch20：FS2四档268.682/317.037/
2899.510/6871.030us，同期V3长文3359.200/7103.860us。96个hash与FR3及
同期V3相同，实际shape/计数及FS2 pool不变。与FR3持平，不记为新增收益；
20K相对V3仍只降低3.28%，不能关闭整体性能目标。日志
`async_training_plan_final_{v3,fs2}_ep16_t*_formal.log`。

默认B512/D8的双机反向FS2→FR3，仍为两节点独占、无CPU构建、1620MHz及
retained10/20：8K wall4093.790/4127.520us，main3302.180/3330.140us；
20K wall8990.520/8941.950us，main8047.300/7981.920us。64个原始hash与
本次V3逐rank一致，实际shape/计数及pool核实。8K反向改善0.82%、20K反向
退化0.54%，与首轮的小幅变化方向相反；差值处于相互重叠的样本范围内。
因此将网络长文结果记为持平，不认定为稳定收益，也不引入长度分支。日志
`async_training_plan_reverse_{fs2,fr3}_ep16_t*_training_forward.log`。

采用FS2：单机wall正反向均改善，移除了一次不参与分配的设备结果回传与host
等待，且所有GPU入口指令、容量和通信协议保持不变。双机收益不作保证。
native retained benchmark复用已经准备的tensors，不把它描述为每轮包含实际
malloc；Torch检查另外覆盖真实tape准备、失败一致性和retry。主机generation
仍可能增加，分配失败不推进的是设备ring epoch/base及payload，而非所有状态。
完整Forward专项sanitizer通过不改变旧Backward racecheck的未决限制。

## FT之后：SavedX静态pair与消费前缀（GA + FZ，已采用，2026-09-25）

FT将完整M128的ring S写回改为TMA，训练仍保存canonical G/U；完整门禁通过，
训练性能持平。后续G/U M32 TMA候选FX没有稳定收益，已撤回。详细数据及
资源/产物哈希见[ring记录](static_forward_v4_ring_record.md)的FT、FX章节。

FZ保持canonical tape、Forward/Backward数值顺序、窗口容量不变：SavedX的
唯一owner由固定Ntile0改为轮转FC1 pair，两个CTA各复制64行。复用已就绪的
multicast A，复制字节数不变；每个CTA的目的指针状态减半，stage full wait、
两个CTA的empty到达、FC1发布前的device fence全部保留。没有新增checkpoint
缓冲区、映射或barrier。仅轮转而不分半的FY没有稳定收益，不单独采用。

FZ独立正反向对照中，单机8K main约改善1.1%；20K效果较小且有波动。网络
20K/B512 main约改善0.8%，8K/B1024约改善0.8%–1.33%；其它网络组合未形成
一致优势。不能将某一档收益推广为全部训练规模。各轮完整表和不利样本保留
在ring记录FY/FZ，不挑选较快的一轮代替相邻基线。

GA单独优化网络consumed credit：逐slot确认全部实际读者退休后，把连续
consumed prefix通过一次ACK返回；TX仍同时检查远端消费及显式本地NIC完成。
channel元数据增加4096 bytes/rank（2节点/32 lanes），payload容量不变。
不改变training checkpoint或Backward GIN协议。组合采用GA+FZ，不含额外
RX导入batch的GA2、G/U TMA的FX、FC2权重multicast或尚未实测的fence候选。

组合三项native检查160个梯度hash与FT逐rank一致，repeat_equal均1；单机9项、
双机12项完整Torch检查通过，包括四档、热点、多个live tapes、OOM/retry、
非ring回退及B1024/D4长文。双机T64/B16/D1和T2048/B512/D8的rank0/15
memcheck为0 errors，Forward planner/main过滤的racecheck为0 hazards/errors/
warnings。双机prefill检查首次因端口占用在launcher启动阶段失败，换端口后
补齐，保留失败日志；不是算子运行错误。详见ring记录组合验收的PID及覆盖数。

组合单机正式对照（1620MHz，retained 10warmup/20samples，rank-max p50/us）：

|T/rank|V3 main / wall|FT main / wall|GA+FZ main / wall|
|---:|---|---|---|
|1|197.760 / 471.682|71.584 / 384.761|69.856 / 383.170|
|64|264.320 / 554.903|100.416 / 436.316|97.152 / 430.782|
|8192|1330.340 / 1733.020|1349.660 / 1839.310|1335.140 / 1825.620|
|20480|3132.420 / 3588.340|3215.710 / 3816.910|3208.860 / 3780.860|

长文wall较FT改善0.74%/0.94%，但相对V3仍慢5.34%/5.37%；整体训练性能目标
未关闭。短档仅本轮相邻证据，不称反向验证过的稳定收益。local main使用
device_kernel_slowest_rank_us，不能误用V3包含其它阶段的forward字段。
计时在第二节点独占执行，第一节点同期仅做Torch正确性，没有CPU构建。
native retained测量复用已准备tensors，不包含每轮实际malloc；真实tape生命周期
另由Torch检查覆盖。日志及160个输出hash/shape/pool核验见ring记录组合EP8表。
双机28组最终对照正在运行，尚无组合网络训练性能结论。

02:39双机正式对照全部完成，1620MHz、两节点独占、无CPU构建，retained
10/20；各规模顺序为V3→FT→GA+FZ，长文再测FT→GA+FZ的B1024/D4。
256个原始输出hash逐rank等于同期V3，shape/world/计数/retained及pool核实：

|T/rank|B/D|V3 main / wall|FT main / wall|GA+FZ main / wall|
|---:|---|---|---|---|
|1|512/8|442.208 / 734.747|168.704 / 626.280|133.920 / 594.082|
|64|512/8|600.640 / 901.641|216.640 / 696.511|186.400 / 662.490|
|8192|512/8|2927.170 / 3438.270|3289.280 / 4092.800|3267.970 / 4058.140|
|8192|1024/4|同上|2975.490 / 3792.770|2930.590 / 3767.730|
|20480|512/8|6658.500 / 7181.430|7978.240 / 8929.010|7934.240 / 8911.160|
|20480|1024/4|同上|7332.220 / 8287.570|7238.080 / 8216.610|

长文同容量main改善0.55%–1.51%，wall改善0.20%–0.86%，小于短序列。
B1024的8K main接近V3，wall仍慢9.58%；20K wall仍慢14.41%。每rank元数据
仅较FT增加4096 bytes，但B1024本身比B512多约167–168 MB，不能称等容量
优化或直接翻转默认值。训练吞吐目标仍开放，继续独立消融物理行局部性。
本节不替代独立FZ的反向证据，也不将组合收益机械拆给两个改动。完整pool、
日志名称、graph回归及短档反向检查见ring记录GA+FZ最终矩阵。

## GC/GD：网络块内source分组及M16 SavedX（未采用，2026-09-25）

GC只在完整regular Mblock内按source分组，并将网络source ordinal改为Notify
稳定顺序；GD再对逐行验证连续的M16区间做TMA。均不增payload/目录容量，不
改变完整wait及slot释放条件。两候选各160个native梯度hash与GA+FZ一致；
GC规划检查及racecheck通过，未构建library或执行完整Torch/主路径sanitizer。

1620MHz，双机EP16独占，T8192/20480、B512/D8及B1024/D4的12组正式相邻
对照有192个输出hash一致。GC的8K两档main慢2.08%/1.18%、20K/B512慢0.63%；
只有20K/B1024快1.27%，反向再快1.46%。GD没有在GC上带来一致增益，尤其
20K/B1024的独立main变化仅±0.15%、方向翻转。故保留大窗长文的实验发现，
不为单个形状硬编码阈值，不采用GC/GD，也没有留下候选测试或TMA map选项。
全部main/wall、相同pool、反向48个hash及launcher参数误用记录，见
[ring优化记录](static_forward_v4_ring_record.md)的GC/GD小节；训练端到端目标仍开放。

## Mini Notify + main 内规划（2026-09-30）

训练前置阶段改为一次 counts-only cooperative kernel：统计并交换各 source/expert
的行数，发布实际 R/P 和容量检查结果。只保留每个 1536-route chunk 的 expert
前缀，不生成 route 行号、packet/fanout 或 V3 wave 排序。所有 rank 完成 tape
分配之后，PF main 在自己的 shared operand arena 中恢复稳定的
expert/source/route 行号，再构建 ring 物理计划；双机 Backward 所需的不可变
packet/fanout 直接从 main 的 node-token 目录生成。原独立 ring planner launch
被删除。推理仍不创建 tape，saved pointers 为空；训练/推理共用 main 实现，
保存代码仍由编译期 specialization 裁掉，而不是给推理增加运行时分支。

SavedX 为 `[R,H]`，G/U 为分离的两个 `[R,I]` 平面。逻辑行序和 route 返回映射
归属于各自 live tape，不依赖下一次 Forward 会覆盖的 ring 计划。mini Notify
是分配边界，不启动 activation payload；分配失败的 collective retry 合同保留。

第一轮 EP8 正式对照：同一远端节点独占，8×L20X、1620 MHz，H2048/I512、
E64/K4、BF16、30 轮（10 warmup + 20 measured），每轮取最慢 rank 后取 p50。
PF 使用 B512/D8，Forward 保留训练状态。单位 µs，包含 Notify 的 host wall：

| tokens/rank | V3 wall | mini wall | wall 降低 | V3 Notify | mini Notify |
|---:|---:|---:|---:|---:|---:|
| 1 | 469.704 | 307.791 | 34.47% | 201.338 | 110.799 |
| 64 | 553.803 | 355.534 | 35.80% | 216.157 | 116.381 |
| 8192 | 1737.380 | 1615.320 | 7.03% | 322.900 | 122.682 |
| 20480 | 3586.140 | 3542.170 | 1.23% | 362.455 | 130.557 |

main 的计时边界现在包含 planner，不能拿它直接对照旧 main 并将边界变化算成
收益。20K 差距较小，不能外推其他模型或节点数。日志为
`build/pf_training_mini_notify/{v3,mini_remote}_ep8_t*_forward.log`；所有输入、
权重和输出 hash 一致。冻结 mini native 二进制 SHA256 为
`72a362e1b8fade9c77c6ef17b50650e87adfff0b8b9426204a59229fa993edb5`。
V3 使用 `build/static_inference_v3_20260923/test_cuda_notify` 并显式设置
`WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`、清除 ring 环境变量；仅凭文件名不能
确定执行的是 V3。

Native 正确性：EP8 T512/E16/K6/B512/D8、EP16 T64/E32/K6/B16/D1，mini
Forward 接原 Backward，两次重复均稳定，dX/dGate/dW1/dW2 逐 rank hash 与
既有 FS 验证结果完全一致。日志位于该目录的 `ep{8,16}_t*_validation.log`。
随后使用 `build/pf_static_backward/` 的 library/extension 跑既有 Torch 集成测试：
EP8/EP16、T64/E64/K4、B16/D1，输出及四组梯度均通过 reference parity；
单 rank 前向/反向分配失败、保留 tape 后重试、三个 live tape、热点目的地切换
及工作区扩缩容均通过。双机测试随后在两个 EP8 group 上验证 inference。
日志为 `torch_ep8_t64.log` 和 `torch_ep16_t64_node{0,1}.log`。该构建含下述
首版静态 Backward；完整长序列矩阵及最终 Backward 性能另记后续结果。

### 静态 Backward 首版：细粒度 task residue 未采用

dU、SwiGLU epilogue、dX 分别将线性 tile 按 cluster 数取模分配，dW 必须等待
本 rank 全部 dX 完成；Return 也改为固定 M-block owner。EP8/EP16 上述 native
用例四组梯度 hash 均与 mini + 原 Backward 一致。但 EP8 同机同频对照，
T1/64 wall 为 257.792/390.887 µs（V3 为 281.948/420.911），T8192/20480
反而为 2540.650/6612.420 µs（V3 为 2127.380/5066.530），慢 19.43%/30.51%。
激活梯度流水明显变慢，保留六个微小 epilogue task 和跨 cluster readiness
造成的等待值得优先消除，不能仅因为去掉 atomic claim 就认为会更快。
记录于 `build/pf_static_backward/ep8_t*_backward.log`；该首版不是最终性能结论。

进一步拆分归因：mini Forward 接**原动态 Backward**时，8K/20K wall 已经是
2538.760/6227.080 µs。主要遗漏之一是旧 Notify 的 local route permutation：
按整个 expert 遍历会让大量发送 CTA 同时写同一张卡；V3 会按 32 行交错 expert。
因此不能把首版相对冻结 V3 的全部回退归因于静态计算调度。

第二版将一个 M-block 的 dU、一次完整 M128 SwiGLU epilogue 和全部 dX N tiles
固定交给同一 cluster。其 8K/20K wall 为 2290.390/6144.980 µs，仍未达标；
EP8 T8192 Torch 的输出及四组梯度已通过。下一步直接由 counts 构造 32 行
交错的 Backward dispatch 索引，不恢复 route-sized radix sort。上述消融都保持
dW 严格后置，不改变 GEMM 算术或归约顺序。

### Counts 直接构造交错 dispatch

先在 `packet_order` 临时区写稳定的 source/expert 列表，再按 counts 计算各
32-row stripe 的目的位置并写 `local_route_order`；网络 packet plan 随后覆盖
临时区。无需恢复 V3 wave 构造或全局 radix sort。对照结果：

| tokens/rank | V3 BWD wall (µs) | M-block + stripe | 耗时降低 |
|---:|---:|---:|---:|
| 1 | 281.948 | 303.724 | -7.72% |
| 64 | 420.911 | 445.834 | -5.92% |
| 8192 | 2127.380 | 1762.040 | 17.17% |
| 20480 | 5066.530 | 4692.760 | 7.38% |

仍为 EP8、同一远端节点 1620 MHz 的 rank-max p50。20K 的 dY dispatch
从 block-only 的 4006.110 µs 降到 1504.350 µs；8K 从 1165.470 降到
628.384 µs。阶段区间存在重叠，不能相加解释 wall。此时训练 FWD wall
四档为 327.081/376.645/1644.690/3562.870 µs；索引构造增加约 20–30 µs，
换回的 BWD 收益明显更大。日志在 `build/pf_static_backward_stripe/`。

EP8 T512 的四组梯度 hash 与原版本完全一致；T64 的 Torch parity、allocation
retry 和三个 live tape 测试通过。修复既有 eager benchmark 在不等长 source
下将 activation lease 的实际 token 数误填为 capacity 的问题后，EP8
T64/32、B16/D1、E64/K4 的训练 FWD `compute-sanitizer --tool memcheck`
完成，error-exitcode 为零。修复前两次超时不算 sanitizer 通过或 kernel 故障。

小规模的回退来自 M-block 数不足以铺满 cluster；后续只在
`mblocks < clusters` 时恢复固定 N/epilogue residue，将其与长文整块调度
分开，不回退动态 claim。最终矩阵另列。

### 激活梯度按占用率选择静态粒度

`mblocks < clusters` 时恢复独立的 dU/N、epilogue-row、dX/N 静态 residue，
其余仍由一个 cluster 完成整块链。没有 token 数或 EP 数硬编码，不改算术。
mini Notify 也保留原有的 out-of-range/duplicate 路由拒绝合同；既有 Torch
集成测试覆盖单 rank 非法路由后的集体报错及下一次正常调用。

`build/pf_static_backward_adaptive/` 的完整 native 矩阵如下。H2048/I512、E64/K4，
1620 MHz，PF B512/D8，30 轮、10 warmup；均为最慢 rank 的 wall p50（µs）。
EP8 在同一远端节点测量；EP16 同形状相邻对照，20K/64 反转 V3/PF 顺序。

| EP | tokens/rank | V3 FWD | PF FWD | V3 BWD | PF BWD |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 469.704 | 351.147 | 281.948 | 258.045 |
| 8 | 64 | 553.803 | 373.505 | 420.911 | 390.108 |
| 8 | 8192 | 1737.380 | 1646.090 | 2127.380 | 1758.660 |
| 8 | 20480 | 3586.140 | 3566.400 | 5066.530 | 4731.360 |
| 16 | 1 | 733.956 | 543.033 | 404.445 | 407.909 |
| 16 | 64 | 903.119 | 617.716 | 766.490 | 756.681 |
| 16 | 8192 | 3434.760 | 3813.370 | 2848.860 | 2961.910 |
| 16 | 20480 | 7213.590 | 8651.610 | 6982.980 | 6768.470 |

单机四档 BWD 已领先，但双机 8K BWD 仍慢 3.97%，双机 FWD 长档仍慢
11.02%/19.93%，没有达到全面领先。静态 dW 的独立 domain/chunk 分配在每卡
只有 4 个专家时留下明显不均衡尾部，后续改为按实际 padded K extent 加权切分
联合 dW2/dW1 流，每个 cluster 持有不相交区间，区间内保留连续 tile chunk
流水。空专家权重为零，热点专家不会被整个分配给一个 cluster；仍不做动态
claim 或 split-K，也不放松 dX 全完成的门槛。

此版本 native SHA256：
`9e4f08dc7435b04947e570e8be717a23e955f2f332aae208c15e9176fc07a05f`。
library：`f2cf1cc5693fa34c541d93f4682446a47dd26b70999fc584c330ef3c93e2e83f`。
EP8 T64 的 Torch parity、非法路由、allocation retry、三个 live tape，以及
T20480 的输出/四组梯度、后续 inference/Graph 生命周期检查通过。

额外用 `tools/benchmark_training.py` 测量真实 eager `forward → backward` 循环，
包含 Python 接口、tensor 分配/回收和末尾 stream synchronize，计时外做 rank
barrier；不是两段 native p50 相加。EP8 同远端节点、10 warmup + 20 measured：

| tokens/rank | V3 cycle (µs) | PF cycle (µs) | 耗时降低 |
|---:|---:|---:|---:|
| 1 | 3710.478 | 3189.257 | 14.05% |
| 64 | 3523.050 | 3306.644 | 6.14% |
| 8192 | 6971.879 | 6261.005 | 10.20% |
| 20480 | 13510.205 | 11215.811 | 16.98% |

cycle 基线使用本轮 stripe library 中保留的 V3/non-ring specialization，
显式 `--backend v3`，不是把旧 native 可执行文件当成 TensorRuntime extension。
日志分别为 `build/pf_static_backward_stripe/v3_ep8_training_cycle.log` 与
`build/pf_static_backward_adaptive/pf_ep8_training_cycle.log`。两种基准的输入生成、
分配边界不同，不能用它们相减推断 Python overhead；双机 cycle 尚待测量。

### dW 加权静态区间

按各专家实际 padded K 行数给联合 dW2/dW1 output-pair 流加权，每个 cluster
拥有连续、不相交的区间。相邻区间边界使用同一向上取整规则；空专家权重为零。
区间内沿用连续 tile chunk，不改 K 归约或 BF16 舍入，也不提前启动 dW。

`build/pf_static_backward_weighted/` 的 EP8 BWD 四档 wall 为
257.309/388.025/1819.440/4643.750 µs，均低于冻结 V3；相对上一版，8K 略退、
20K 改善。EP16 8K 为 2928.310 µs，仍比 V3 的 2848.860 µs 慢 2.79%，
加权分配不是剩余 activation-gradient 差距的解决方案。

该轮 EP16 脚本一度将 benchmark tokens 误填入仅 validation 使用的环境变量；
已按 JSON 的实际 shape 纠正：另外三次都是 8K 重复，不是 1/64/20K，日志改名
`pf_ep16_t8192_repeat{1,2,3}_backward.log`。不纳入四规模矩阵。

保持 B×D=4096，训练 FWD 的 B1024/D4 相比 B512/D8，EP16 8K 从
3832.900 降到 3613.320 µs，20K 从 8663.880 降到 7906.850 µs。
仍比冻结 V3 慢 5.20%/9.61%，因此没有将扩大窗口误记为全面领先，也不静默
改写默认参数；完整循环另用显式 B1024/D4 测量。

该轮 native/library SHA256 分别为
`f880e6297f30c98a81816e315f06642ed16ce17caeb8901ba2550194f3277943`、
`5ff101167e7f77785fa77e833ffe5c3c95318365994e524bc2e7615abe6faead`。
EP8 T64/T20480 Torch reference parity 已通过。EP8 初次 cycle 的 64 token
结果波动至 3870.830 µs，不能引用上一版更好的 cycle 作为本版最终值，待配对复测。

PF profile 的 `kRdmaCtas=0` 不包含非 ring 的多机 V3 Forward 回退；尝试用
同一 extension 测双机 V3 cycle 会在发起 Forward 时明确拒绝，不能视为性能
结果。双机 cycle 对照需单独按 `configs/static_inference_v3_20260923/` 构建
library/extension。冻结 native V3 的独立 FWD/BWD 结果不受此配置问题影响。

### 未采用：按本节点 source 优先的 M-block 排列

尝试将各专家的 M-block 交错，并从该节点的 canonical source 区间开始，
避免固定计算 owner 先等远端数据。`pf_static_backward_locality` 的 EP16
8K B512/D8 四组梯度 hash 与旧 Backward 相同，dU 区间缩短至 1353.340 µs，
但 wall 为 2957.510 µs，反而差于 weighted 的 2928.310 µs。

进一步让静态 Return 使用相同排列（`pf_static_backward_ordered_return`），
EP16 8K/20K B1024/D4 wall 为 3054.770/7361.010 µs。Return 区间变短并未
转换为端到端收益；不能只看阶段 span 就认定优化成功。两项排列及缓存 owner
状态均从最终代码移除，保留较简洁的 canonical residue + weighted dW。

额外正确性门禁：weighted EP16 T64/T20480 Torch 全部输出及四组梯度通过，
T64 还覆盖非法 route、分配重试与三个 live tape；EP16 T64/32 不等长 source
在 B16/D1 与 B1024/D4 下六组输入/权重/输出 hash 全部一致。
locality 的 EP8 T64/E16/K6/B16/D1 Backward memcheck launcher 退出码为零；
该用例只经过不受网络排列影响的本地调度。不将 memcheck 扩大解释为 racecheck
或完整多机 sanitizer 认证。

按原 V3 profile 新建的 TensorRuntime 对照库为
`build/pf_training_v3_reference/libweftmoe.so`，SHA256
`6d349d29f7120e00ae1225ac9ea8e0fa1423e6f18423828b3650c3b3c49812e1`。
EP8 20 warmup + 50 measured，V3 cycle 四档为
3398.326/3537.165/6416.721/11410.903 µs；weighted PF 复测为
2982.490/3259.546/6150.959/10848.984 µs，耗时降低 12.24%/7.85%/4.14%/4.92%。
长文收益小于先前同库 V3 回退对照，最终不再沿用那个较大的百分比。

### 最终静态 Backward native 矩阵

最终保留 weighted 版；EP8 B512/D8，EP16 B1024/D4（Forward-only 窗口消融
选择，Backward 不改变 K 归约），其余形状/频率/轮数同上。单位 µs：

| EP | tokens/rank | 冻结 V3 BWD wall | PF BWD wall | PF 耗时降低 |
|---:|---:|---:|---:|---:|
| 8 | 1 | 281.948 | 257.309 | 8.74% |
| 8 | 64 | 420.911 | 388.025 | 7.81% |
| 8 | 8192 | 2127.380 | 1819.440 | 14.47% |
| 8 | 20480 | 5066.530 | 4643.750 | 8.35% |
| 16 | 1 | 404.445 | 408.590 | -1.02% |
| 16 | 64 | 766.490 | 762.142 | 0.57% |
| 16 | 8192 | 2848.860 | 2931.490 | -2.90% |
| 16 | 20480 | 6982.980 | 6732.710 | 3.58% |

小于约 1% 的差异按接近看待，不表述成确定领先。8K 双机仍回退约 2.9%，
因此只完成了功能门禁，没有达成“所有规模 BWD 不低于 V3”。

### 双机完整循环：必须独立于 cached-state native 报告

同一 weighted PF 与原 V3 profile 的新 TensorRuntime extension，EP16
B1024/D4，20 warmup + 50 measured 的首轮真实 `forward → backward`：

| tokens/rank | V3 cycle (µs) | PF cycle (µs) |
|---:|---:|---:|
| 1 | 7151.888 | 5551.253 |
| 64 | 7187.492 | 4760.193 |
| 8192 | 9088.657 | 10038.992 |
| 20480 | 18300.539 | 23454.933 |

这组 host wall 抖动明显：PF 20K 单次范围 17414–185439 µs，各连续十轮的
中位数约 20.4/23.6/21.4/39.4/26.8 ms；不能把首轮 28.2% 的差距全归因于
kernel。新增 benchmark 的 Forward/Backward 接口分段计时（不改算子），
反转运行顺序，10 warmup + 20 measured 的长档复测如下。各列单独取最慢
rank p50，不能相加重建 cycle：

| tokens/rank | V3 FWD API | PF FWD API | V3 BWD API | PF BWD API | V3 cycle | PF cycle |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 4794.760 | 5141.901 | 4316.087 | 4443.026 | 9064.414 | 10004.030 |
| 20480 | 8542.127 | 10054.226 | 8444.019 | 8221.840 | 17210.427 | 18317.683 |

复测仍有约 10.4%/6.4% 的真实循环回退，稳定的主要差距在训练 Forward，
不能用单机收益、cached-state BWD 或 Notify 加速替代这一结论。原 V3 profile
与 PF profile 均保持各自生产 GEMM/通信参数，不把切换 profile 当成静态调度的
单项收益。

尝试仅绑定各 local rank 的调用线程到 CPU 8/10/12/14/56/58/60/62 并没有消除
抖动（PF 8K/20K 为 14301.164/20219.641 µs，V3 为 9357.899/17710.015 µs）。
未修改系统或 NCCL 的 CPU 亲和性，也没有改线上 Runtime；这个无收益的临时
benchmark 选项已移除。原始样本分别保留在 weighted 目录的
`{v3,pf}_ep16_cycle_node*.log`、`*_phase_node*.log`、`*_pinned_node*.log`。

保持 B×D=4096 再测 B2048/D2，EP16 8K/20K 训练 FWD 为
3807.160/8111.470 µs，均差于 B1024/D4 的 3613.320/7906.850 µs，未采用。
没有通过扩大 ring 默认值掩盖未解决的双机训练 Forward 差距。

复现完整循环使用 `tools/benchmark_training.py --backend pf|v3`，通过
`WEFTMOE_TENSOR_EXTENSION` 指向各自 profile 的 extension；EP8 用单机
`torchrun --standalone --nproc-per-node=8`，EP16 用两台机器各八进程的
`torchrun --nnodes=2`。PF 的 `--ring-tokens 512 --ring-depth 8` 对应单机，
`--ring-tokens 1024 --ring-depth 4` 对应上述双机对照。该工具只测 MoE 的
TensorRuntime FWD+BWD，不是完整模型、RL step 或包含 optimizer 的训练 step。

最终补充极端热点门禁：weighted EP16、T8192/E64/K4、B1024/D4，所有 source
均路由到 rank 0 的四个专家，rank 0 实际接收 524288 行，其余 rank 零入站。
Forward 输出和 dX/dGate/dW1/dW2 全部通过 Torch reference，随后 EP8 子组的
inference/Graph 生命周期检查也通过。日志为
`torch_ep16_hotspot_t8192_node{0,1}.log`；这验证了实际 tape 扩容、静态 dW
在严重倾斜下的完整覆盖以及 no-drop，而不只是均匀 benchmark。

最终又对单机完整循环做一次相邻、反转顺序（PF→V3）的 20+50 轮确认，
仍独占远端节点且调用线程不绑核。四档 V3/PF wall 分别为
3337.445/3158.278、3587.115/3325.652、6577.049/6212.620、
11247.264/10866.331 µs，降低 5.37%/7.29%/5.54%/3.39%。
两轮单机均领先，但不将不同轮次中更有利的值拼成一个表。
日志为 `build/pf_static_backward_final/{pf,v3}_ep8_cycle.log`，实际加载的
extension 路径已写入每个 JSON（weighted PF 和原 V3 profile）。

撤掉无收益候选后重新完整构建 `build/pf_static_backward_final/`，native SHA256
为 `6d512a8377b16c510868622910aa30a7d64fc581df22568e6d6a038d897dad2b`，
library 为 `5e1d15a9fc7d12741966d39971d295d95b270217a50ca80a2243ee1b11f7bda2`。
编译完成后、无编译干扰的 EP16 B1024/D4 长档 BWD 复测为
2919.860/6667.710 µs：8K 仍比冻结 V3 慢 2.49%，20K 快 4.52%，
与上述“8K 小幅回退、20K 领先”的结论一致。没有将构建产物哈希不同误当成
功能变更，也不把不同复测中最有利的数字拼成新完整矩阵。

最终 extension 又通过 EP16 **T20480 全热点**的完整 Torch reference：rank 0
实际接收 1,310,720 行，其余 rank 零入站；输出和全部四组梯度正确，随后
inference/Graph 生命周期检查通过。使用 B1024/D4，未改变 ring 容量来容纳
整段上下文；永久 tape/Backward 实际行数工作区则按需扩容。日志为
`build/pf_static_backward_final/torch_ep16_hotspot_t20480_node{0,1}.log`。

## 长文差距复核与下一轮测量边界（2026-10-01）

后续只验收 T8192/20480：单机 EP8、双机 EP16 的训练 Forward、Backward 和
实际 FWD+BWD cycle 都要优于 V3，共十二个比较项。已有单机收益不能替代
双机验收；`tools/benchmark_training.py` 默认也改为这两个长档。

### 原始日志暴露出的规划成本

重新解析上述 adaptive 目录的 `v3_ep16_t*_forward.log` 与 weighted 目录的
`pf_ep16_t*_b1024_forward.log`，没有运行新的 GPU 实验。双机汇总脚本原先
遗漏了 native 已输出的 `device_kernel`、`route_exchange`、`before_combine`、
`after_combine`；现在保留这些字段，旧二进制没有对应字段时仍可汇总。

单位 µs，仍是每轮最慢 rank 再取 p50；各列独立统计，不能直接相加减：

| T/rank | 版本 | Notify | device kernel | main 内规划/交换 | 去除规划后的 device 时间 | main 末尾 rendezvous 区间 |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | V3 | 468.316 | 2852.860 | 0 | 2852.860 | 未单独计时 |
| 8192 | PF | 184.574 | 3334.660 | 397.856 | 2944.064 | 313.088 |
| 20480 | V3 | 508.014 | 6570.500 | 0 | 6570.500 | 未单独计时 |
| 20480 | PF | 187.504 | 7621.410 | 555.648 | 7072.834 | 338.048 |

“去除规划”先在**同一 rank、同一次调用**内计算
`device_kernel - route_exchange`，然后取该轮 rank-max 和跨轮 p50，不是
两列 p50 相减。V3 的路由规划在 Notify，不是没有规划成本。
另一个口径差异是旧表的 `forward`：V3 使用 CUDA event 的总时间，PF 使用
device status 的 kernel 时间。比较 kernel 本身应使用这里共同的
`device_kernel`，端到端验收继续用原来的 wall/cycle。

由此可以定位、但还不能完全归因：8K 的 device 差距主要集中在 main 内规划；
20K 去除规划后仍有约 502 µs 差距，不能只优化 planner 就宣布目标达成。
PF Dispatch/FC1/FC2/Return 的整段跨度包含等待且互相重叠，不代表实际 GEMM
用时翻倍。末尾 `after_combine` 从 CTA 0 的通信角色到达开始计时，还包括
等待其它 CTA 和其它 rank 的尾部，不能把 313/338 µs 全算成网络 barrier 的
纯指令开销，更不能直接删除保护下次调用元数据复用的 rendezvous。

### 源码证据与诊断补齐

V3 训练也会保存 X 和 G/U，不能将差距归因为 PF 独有的整份 checkpoint
流量。实际差异是写回几何和指令路径：V3 完整 M 块的 SavedX 连续 TMA
写回；双机 PF 的 ring 物理行与 canonical tape 行不同，SavedX 使用 math
线程的 uint4 scatter，G/U 同样按 canonical 行映射写回。它们与 WGMMA
共享 stage 的读生命周期，必须全部读完才能归还 empty。单机的连续行 TMA
路径不能无条件搬到双机。

现有 timeline 的 `RingPlanBegin` 在 `build_training_rows` 之后，无法直接
看见训练行号重建；`RingPlanSourceReady` 又落在训练 packet 重建之后。
schema 13 增加 `TrainingRowsBegin/End`、`TrainingPacketsBegin/End`，并将
SourceReady 放回 source 目录就绪的边界。分析工具分别报告 `training_rows`
和 `training_packets`；后者包含在新的 `ring_plan_packets` 区间中，不能重复
累计。旧 schema 的 source/packets 分段边界不同，不能直接作为单项对照。
所有事件均受 `WEFTMOE_DEVICE_TIMELINE` 编译宏控制，不增加生产同步或任务。

下一轮先拆清 canonical 行重建、训练 packet 重建、ring 目录交换、destination
几何这四项，再决定 planner 改动；随后比较 retained/non-retained 的 math 与
epilogue，并检查 Return credit 引起的计算等待。历史 GC/GD 的 source 分组/
M16 TMA、AJ 的 wait1、GK 的两 K-stage group 合并都没有稳定收益，不能把
这些旧实验重新包装成已证实的优化。Backward 8K 还需独立处理，保持全部
本地 dX 完成后才进入 dW 的约束。

本轮环境没有暴露 `/dev/nvidiactl`、`/dev/nvidia0`，`nvidia-smi` 无法访问
驱动；SSH 创建 socket 被拒绝，`.git` 为只读。上述结论来自已保存日志和
源码，不是新的性能结果；未修改生产计算/通信算法，也未提交新的性能收益。

离线验证：PF profile 的 timeline native 完整编译通过，产物为
`build/pf_training_timeline/test_cuda_notify`，SHA256
`adce0eb6a77047bf6b50ab260d274e3bcb94e9176d451814db3beccbd4cf3f28`。
四份 EP16 长文原始日志重新汇总后，全部旧 JSON 指标逐项不变；缺少新增字段
的旧格式兼容检查、新事件区间解析及 Python 语法检查通过。未新增细碎单测，
未运行 GPU 性能或数值验证；诊断版不能代替生产二进制做最终性能验收。

继续核查 planner 的数据依赖：`build_training_packets` 只读取本地 source
目录、selected experts 和 canonical route 行号，写入 Notify lease 的
Backward packet/fanout、packet order 及 runtime counts。Forward 的
`build_dispatch_packets`/`build_return_packets` 使用独立的 ring 目录；训练
packet 重建的输出不在 `make_source_metadata_layout` 的 published 区间中。因此下一项
可测候选是将训练 packet 重建移到 metadata Put 提交后、对应 context 的 Put
fence 之前，尝试隐藏这段重建时间。仍需保持全体 CTA 的一致 collective
顺序和 scratch 生命周期，不删除 fence，也不延迟到 main 返回之后。
这是源码依赖分析，尚未改动调用顺序或验证收益；GPU/SSH 访问仍受限。

### 环境恢复后的 planner timeline

2026-10-01 恢复 GPU、SSH 和 Git 写权限，两节点各八张 L20X 均空闲且
SM 时钟为 1620MHz。使用上述 schema 13 诊断版，EP16/B1024/D4，保留
训练状态，12 轮（10 warmup），两个长档全部完成。各 rank 最后一次调用
的区间时长取跨 rank 中位数，单位 µs；只用于定位，不作为生产性能：

| T/rank | 训练行号 | 训练 packet | source 规划 | packet 规划总段 | 目录交换 | destination 几何 | 行映射绑定 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 60.896 | 27.616 | 86.384 | 69.856 | 59.328 | 64.512 | 42.592 |
| 20480 | 75.824 | 31.600 | 115.360 | 78.944 | 103.648 | 87.488 | 84.976 |

训练 packet 包含在 packet 总段内，不能重复相加。32 份 trace 均无 overflow；
文件为 `build/pf_training_timeline/ep16_t{8192,20480}.rank*.forward.json`。
首次具体消融选择缩小训练行号 radix sort 的 bit 范围：输入按 route ordinal
递增，已安装 CUB 的 BlockRadixSort 保证稳定性，只排 expert 位即可保持完整
行序；随后再独立测训练 packet 与目录 Put 的重叠，不混淆两个改动的收益。

### planner 分项消融（2026-10-01，继续优化中）

恢复环境后重新测生产二进制，H2048/I512/E64/K4、EP16/B1024/D4、
1620MHz，30 轮中前 10 轮预热。下表均为每轮 rank-max 后的 p50，单位 µs。
`base reverse` 是最后一个 overlap 候选之后反转顺序复测的 c6be309；
相邻轮次的 Notify/计算区间仍有波动，不能把 wall 的全部变化归因于 planner。

| 累计候选 | 8K wall | 20K wall | 8K main 规划 | 20K main 规划 |
|---|---:|---:|---:|---:|
| 原 V3 profile | 3433.490 | 7161.700 | 0（规划在 Notify） | 0 |
| c6be309，初次恢复 | 3558.880 | 7912.890 | 392.512 | 555.008 |
| A：只对 expert 位做稳定 radix sort | 3575.720 | 7954.880 | 390.688 | 553.024 |
| B：训练 packet 与目录 Put 重叠 | 3565.720 | 7898.460 | 363.712 | 520.928 |
| c6be309，base reverse | 3596.000 | 7949.260 | 393.504 | 555.776 |
| C：全网格静态分摊 RX 描述符 | 3566.460 | 7893.650 | 341.920 | 503.872 |
| D：训练表按 source-route 顺序合并写入 | 3533.150 | 7833.880 | 336.128 | 499.552 |

A 利用 CUB BlockRadixSort 的稳定性，输入 ordinal 原本已经递增，因此无需
再排序低位 ordinal。规划只改善约 2 µs，没有独立的端到端收益证据。
B 的 packet/fanout 重建不属于发布给 Forward 的源目录，移到目录 Put 提交后、
context fence 之前，保持原有 fence 和 collective 顺序；另外单机训练跳过
Backward 不使用的 node-token 目录。C 把每窗口一个 CTA 内的描述符循环改成
整个 grid 的固定 residue，没有领取队列或新增同步。D 将稳定排序所得 ordinal
暂存回原始 route 下标，再由相邻 lane 写相邻 route 的元数据；复用原 shared
scratch，新增两次 CTA barrier，无新增分配。D 仍是候选，尚未完成所有门禁。

A/B/C/D 的两个长档均与原版保持 16 ranks 的输入、route、gate、实际权重及
输出 hash 一致。B 另通过 EP8 两个长档的重复 Backward 验证：所有 rank 的
dX/dGate/dW1/dW2 hash 与 c6be309 逐项一致。不能据 Forward benchmark 的输出
hash 推断 C/D 的梯度已经验证。日志统一保存在 `build/pf_training_timeline/`，
前缀分别为 `v3_restored`、`base_restored`、`stable_sort`、`overlap`、
`base_reverse`、`parallel_pull`、`coalesced_rows`。

本轮单机独占远端节点的恢复基线，EP8/B512/D8，FWD V3/PF 分别为
1724.780/1640.600（8K）、3573.070/3556.640（20K）；BWD 为
2132.490/1818.290、5067.280/4721.710。20 warmup + 50 measured 的实际 cycle
V3/PF 为 6521.193/6294.860、10977.895/10866.903。20K FWD/cycle 优势很小，
仍须最终配对复测，不能沿用历史轮次更大的收益。

D 补充 EP8 重复 Backward 验证已通过，两个长档四组梯度 hash 均逐 rank
等于 c6be309。独占远端节点的反向顺序 FWD 复测，原版→D 的 wall 为
1640.210→1629.740、3588.240→3554.950 µs，规划为
173.568→163.424、291.264→284.032 µs；日志 `base_confirm_ep8_*` 与
`coalesced_confirm_ep8_*`。这支持小幅收益，不表述成 GEMM 提速。

未采用 E（`pf_plan_node_reservations`）：source node 计数/预约以
`match_any` 合并同 warp 的原子操作；不改变每 token 每命中节点一个记录的
语义，但 EP16 规划为 347.520/500.864 µs、wall 3537.020/7908.970 µs，
未优于 D，源码已撤回。不存在“减少原子数必然更快”的采用结论。

同日重新测 EP16 cached-state BWD，V3/PF c6be309 的 wall 分别为
2851.220/2917.230（8K）、6996.810/6696.970（20K）。8K 仍回退 2.32%，
20K 降低 4.29%；不能用其它旧轮次掩盖 8K 未达标。日志前缀
`v3_bwd_restored`、`base_bwd_restored`。现有 Forward timeline 的一个 rank
还显示 20K 每 CTA 平均约 5.21ms 落在 select/wait 区间，FC1/FC2 tile
区间合计约 1.31ms；事件插桩会扰动绝对耗时，这只说明不宜将整个差距
归为纯 GEMM 算力不足，后续仍须用生产 wall 验证候选。

### F：在规划中写完永久 tape 的 route header/gate

之前每条训练 route 的 72-byte `LocalP2pRouteHeader` 与 gate，都在 X Pull 的
issuer lane 写入。它们只依赖源 rank、route、expert、canonical 行号和本次
epoch，不依赖 X 到达，也不属于可复用的 payload slot。因此把写入移到已有
destination 行映射构建循环，复用该处已经解出的值；原有规划 join 发布它们。
Pull 保留物理 ring gate/X 的写入和完整 completion/credit，不能把物理 slot
的 gate 也提前写入，否则会覆盖前代。FC1 的 SavedX/GU 保存方式没有改变。

这不是减少永久 tape 的字节数，而是将不可变元数据从单 lane 的串行数据面
移到并行控制面。删除原 `save_identity` helper 和网络 Pull 中为它广播的
block/index；Pull 不再因训练而生成另一份函数实例。无新增 kernel、分配、
全局同步或任务队列，inference 的空缓存合同不变。

EP16/B1024/D4、1620MHz、10 warmup + 20 measured；先 F 后 D 的反向对照，
单位 µs，rank-max p50：

| T/rank | D wall | F wall | 耗时降低 | D main 规划 | F main 规划 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 3530.850 | 3312.190 | 6.19% | 337.920 | 337.888 |
| 20480 | 7870.230 | 7108.690 | 9.68% | 500.032 | 504.544 |

F 的 device kernel 为 3035.710/6816.900 µs。相比同日恢复后的 V3 wall
3433.490/7161.700 µs，降低 3.53%/0.74%；20K 领先仍小，必须最终配对复测，
此处不宣布十二项全面验收。两版全部输入/route/gate/实际权重/输出 hash 一致。
F 和 c6be309 又各自完成 EP16 T8192/20480 重复 Backward 验证，全部 16 ranks
的 dX/dGate/dW1/dW2 hash 逐项一致，repeat_equal 均为 1。该数值检查与本地
编译并行，**不引用其 wall 作性能证据**。完整 Torch/实际 cycle 尚待新库验证。

冻结 native `build/pf_plan_checkpoint_headers/test_cuda_notify`，SHA256
`3fded55c7ee9857c2166a172adb7a950239ff966b9ce31042e03f17b721eadf4`。
性能日志前缀 `checkpoint_headers`、`coalesced_reverse`；梯度日志
`{pf_plan_checkpoint_headers,pf_static_backward_final}_ep16_t*_validation.log`。

F 的单机补充：独占远端节点、B512/D8 的 FWD wall 为 1543.200/3334.340 µs，
明显优于刚才 D 的 1629.740/3554.950 µs（降低 5.31%/6.21%）。规划为
164.544/288.224 µs，未减少规划工作；SavedX/GU 算法仍不变。两个长档的
EP8 重复 Backward 也全部通过，与 c6be309 的四组梯度 hash 逐 rank 相同。
本轮采用 A–D 与 F 作为 planner/immutable-header 里程碑，E 已撤回；新的
Backward ready-residue 实验单独记录，不包含在此项收益内。

### F 后的窗口消融与静态 Backward 就绪探测（待最终验收）

移除 Pull 的串行 header 工作后，重测生产二进制的窗口参数；仍 EP16、
1620MHz、30 轮/10 warmup，单位 µs。这里 B 是每源 token 窗口，D 是
source contribution 深度，不是 GEMM M 或网络 packet ring depth。

| B/D | 8K wall | 20K wall | 8K main 规划 | 20K main 规划 |
|---|---:|---:|---:|---:|
| 1024/4 | 3312.190 | 7108.690 | 337.888 | 504.544 |
| 512/8 | 3311.680 | 7374.730 | 309.408 | 525.312 |
| 768/6 | 3246.660 | 7011.360 | 325.408 | 500.288 |

三个配置的六组输入/权重/输出 hash 均一致。768/6 的两个长档 device pool 为
795434176/912494784 bytes/rank，1024/4 为 870260928/987174592；总 pool
少约 74.8/74.7 MB。其 registered 部分反而多约 8.45/8.47 MB，因为 source
contribution 容量稍大；compute ring 的较小最坏窗口节省了更多空间。
永久 tape 与调用级 Backward tensors 不包含在这个 pool 数字中。没有新增
运行时 shape 阈值或模型专用分支；最终还需反向计时和同参数实际 cycle。

G（`pf_backward_ready_residues`）只改变 activation 链的访问顺序：cluster
仍独占 `c + n*clusters`，但原下一个 block 未 ready 时，会在**自己的**
residue 中探测 phase=1 的 block；不领取其它 cluster 的任务。dW 仍严格等待
全部本地 dX，canonical tape/dW K 顺序不变。EP16 两个长档的重复梯度 hash
均与 c6be309 完全相同。

| T/rank | 原静态 BWD wall | G wall | 原 dU/epilogue 区间 | G dU/epilogue 区间 | 原 combine 区间 | G combine 区间 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 2917.230 | 2898.330 | 1700.580 | 1380.740 | 187.552 | 414.560 |
| 20480 | 6696.970 | 6878.380 | 4284.800 | 3863.010 | 216.288 | 940.224 |

各区间独立统计且互相重叠，不能相加恢复 wall。G 单独没有全面收益，不采用
为最终版本：计算就绪更早，并不等于固定顺序 Return 能及时消费它。
下一候选 H 同时让每个 Return owner 在自己的固定 residue 内探测完成块，
用既有 phase=4 表示本 owner 已访问，不增加 CAS/领取队列；实际 returned
计数仍在 copy/publish 后推进。此时 H 尚未完成构建、数值或性能验证。

768/6 补充的 EP16 T8192/20480 重复 Backward 已通过，全部四组梯度 hash 与
c6be309 的 B1024/D4 逐 rank 相同；日志 `headers_b768d6_ep16_t*_validation.log`。

H 已构建 native/library/extension，EP8 两个长档四组梯度 hash 与 c6be309 一致。
EP16/B1024/D4 的 wall 为 2899.490/6878.910 µs，combine 区间降到
123.136/118.624 µs，但总耗时基本等于 G，仍不如原静态调度的长档组合。
**G/H 均已撤回**，没有只因为阶段 span 变短就采用，也未为 H 执行完整 Torch
或实际 cycle。资源用量核对中，原版/G/H 的静态 unified BWD 均 REG168、
SHARED2048、LOCAL0，网络 STACK208/本地 STACK0；没有发现新增 spill 的证据。

H native 的 Forward 代码仍是 F。它与原 V3 的相邻长档对照，PF 用 768/6，
V3 保持原 profile：8K wall 3243.280 vs 3437.380，20K 为
7089.200 vs 7180.770 µs，降低 5.65%/1.28%。这是 Forward 的再次确认，
不能作为 H Backward 的采用依据。日志 `pf_final_t*_forward.log`、
`v3_final_t*_forward.log`；此处的 `final` 是当轮文件标签，不代表目标完成。

下一候选 I 基于 F 的原静态计算/Return 次序，尝试在通信暂时无可做工作时
非阻塞 Reduce：每个 warp 保持原 token residue，只在全部 top-k ready 后
执行原 FP32 累加及 BF16 输出，不等待缺失 contribution，不领取其它 warp
的 token。单独的 shared pointer table 避免覆盖仍在使用的 TMA staging；
通信结束后从各自 cursor 继续处理剩余 token。尚无收益结论。

I 的 EP16 两个长档已通过重复反向验证，16 ranks 的四组梯度 hash 均与
c6be309 相同。B1024/D4 的首轮 wall 为 2862.840/6676.730 µs，相比同日原静态
版本 2917.230/6696.970 降低 1.86%/0.30%；8K 仍略高于 V3 的 2851.220。
静态 unified kernel 的 REG168、SHARED2048、LOCAL0 和 STACK 用量未变化，
额外 pointer table 属于 dynamic shared memory。`reduce` 时间戳现在包含与
通信重叠的就绪等待区间，不能拿其变长推断归约算术变慢，也不能与其它阶段
相加。日志 `stream_reduce_*`；单机与反向顺序复测尚未结束，不作正式采用结论。

J（`pf_backward_activation_chain`，在 I 上消融）：当 M-block 数足够时，
静态 cluster 已独占整块 dU→SwiGLU/dGate→dX，三段不必重新交换任务描述符。
调度器一次返回整块任务，沿用原 GEMM、epilogue 和逐阶段数据完成同步，
只省去中间两次调度回环/DSM mailbox 往返。欠填充的 N/epilogue residue
路径不变；dW 仍严格等待所有本地 dX。此时仅开始构建，尚未验证或采用。

I 补充 EP8 两个长档的全部梯度 hash 与 c6be309 相同。独占远端节点 I→原版
的相邻 BWD 消融为 1789.120→1821.890（8K）、4733.480→4670.430（20K）；
I 并非单独全面有收益。J 在相同节点为 1776.220/4611.050。J 的 EP16
两个长档全部梯度 hash 也与原版一致，资源用量仍无新增 spill；其 B1024/D4
wall 为 2869.660/6654.510，相邻 V3 为 2839.690/6990.520。8K 仍差 1.06%，
不能用更早稍慢的 V3 轮次宣布达标。

J library 的首次 EP16 实际 cycle（PF B768/D6，20 warmup + 50 measured）为
14642.389/20479.200 µs，随后 V3 为 10288.255/19187.677 µs；FWD API 各为
5791.449/9046.129 vs 4808.787/8891.286，BWD API 为
7223.399/9150.568 vs 5067.943/9399.583。主机延迟抖动明显，保留完整样本，
不裁掉大值，也不把 native cached-state 的小幅收益当成真实循环已通过。
日志 `activation_chain_*`、`v3_chain_pair_*`。

K（`pf_training_single_agreement`）：mini Notify 已提供准确 R/P，Runtime 可
在现有输入/权重准备的同一次 CollectiveTensorPreparation 内分配 saved tensors。
删除旧独立 planner 时代留下的延迟分配 callback；传入已绑定 tensors 时，
backend 不重复进行分配确认。未绑定 tensors 的原生入口仍在 backend 内分配
并集体确认，任何路径都没有在 main 中分配。应用自己准备 buffers 时本来就
必须在进入 Forward 前一致处理分配失败，否则旧的第二次确认同样救不了
没有进入 Forward 的失败 rank。该候选需要重新验证分配失败、重试与 live tape。
不改变 device planner、GEMM、ring 及输入验证；尚无性能结论。

L（`pf_backward_dispatch32`）在 K 上单独将既有 Backward dispatch batch 参数
从 64 调成 32，让通信 warp 更频繁返回 GIN/Return 进展循环；所有权仍为静态
residue，没有新增动态调度。只改构建参数，不增加规模阈值；尚未验证或采用。

J 的 EP8 两个长档也完成四组梯度 hash 对照，全部等于 c6be309。其真实单机
cycle 为 6071.962/10648.123 µs，相邻 V3 为 6592.669/11656.652，降低
7.90%/8.65%。该收益属于 F+I+J 的累计结果，不全部归因于 J 的 mailbox 简化。
I+J 作为小的静态 Backward 里程碑保留；双机 8K 与真实 cycle 仍未达标。

K 的 EP16 T8192 已通过既有 Torch 集成测试：输出及四组梯度、非法路由拒绝、
单 rank 前后向分配失败与重试、三个 live tape/热点切换、后续 EP8 inference
graph/offload。T20480 正在验证。L 的 EP16 两个长档四组梯度 hash 已全部
等于原版；这里的正确性任务可与 CPU 构建重叠，其 wall 不作为性能数据。

M（`pf_backward_peer_reuse`，基于 K，仍用 dispatch batch 64）：Backward
通信 workspace 在所有 rank 的实际 rows/routes 均不变时，复用原地址和映射。
用分配确认的同一个 MIN 状态字编码 failure-rank 优先级与 changed 位；一旦
有 rank 分配失败，全体仍抛错，只有全体成功且无人变化才跳过地址 AllGather
及第二次 import 确认。任一 rank 改变尺寸仍走完整交换和旧映射退休协议，
不能凭本 rank 未变就自行跳过 collective。exact-shape 策略不变，不保留历史
热点的大 payload。稳态新调用的这部分主机 collective 从三次减到一次；
不是删除 main 的通信完成同步。尚未验证或测得收益。

K 的 EP16 T20480 完整 Torch 检查也通过；两个长档每项有 16 个 rank 的
parity/retry/rejection/graph 完成标记，以及 48 个 live-tape reuse 标记。
EP8 T8192 的 FWD+重复 BWD 通过 compute-sanitizer memcheck，梯度 hash 与
原版一致。首次启用 API 错误报告时，NCCL 初始化的 `cudaFuncGetAttributes`
和 `cudaGetLastError` 返回 `cudaErrorNoKernelImageForDevice` 被记为 41 次 API
错误；保留该失败日志。重跑仅关闭 API 错误报告（`--report-api-errors no`），
内存访问检查保留且退出码为 0；不宣称首次检查通过或所有 CUDA API 无错误。

K 独占远端节点的 native FWD 为 1499.550/3254.340 µs，前一版本 F 为
1543.200/3334.340；device GEMM 不变，收益来自避免已绑定 tensors 的重复确认。
实际 EP8 cycle 为 5942.059/12007.974 µs，20K 主机波动明显、没有全面领先，
继续保留全部样本并测 M，不根据单次快的循环作最终验收。

L 的 EP16 BWD 为 2856.090/6662.820 µs，仍未超过相邻 V3 的 8K。
其 EP8 为 1751.040/4755.690，20K 差于 J 的 4611.050，**不采用 batch32**。
K 的 EP16/B768/D6 native FWD 为 3165.920/6988.520 µs；相比相邻 F 的
3243.280/7089.200，主机重复确认的减少与 wall 改善方向一致。

真实 EP16 cycle 的 K→M 首轮为 11529.613→8639.629（8K）、
16436.454→21512.953（20K）µs。M 没有改变 Forward kernel，但 20K 的 FWD API
也从 8328.170 波动到 9964.848，不能把两个独立进程组间的全部差异归于
Backward 缓存策略，更不能挑较快的一个规模拼出“全面通过”。后续确认循环
统一增加到 50 warmup + 200 measured，PF/V3 使用相同配置、保留所有样本。

N（`pf_backward_receiver_reduce`，基于 M）：仅静态网络 BWD 的 Reduce token
固定分配给 receiver CTA 的 warp。它们能在通信存活时非阻塞归约；sender
warp 继续只做 pack/submit，不再承担必须等全部 TX drain 后才开始的 token。
本地与旧动态路径不变，token 内算术顺序不变。尚未验证或采用。

O（`pf_backward_owner_wait`，在 N 上消融）：静态 cluster owner 暂时没有可做
工作时，只有 owner lane 原地探测固定依赖，其他计算线程停在已有 compute
join；通信 warp 不参与该 join，仍能提供缺失数据。省去空任务的反复 DSM
mailbox 往返，不做跨 owner 领取，也不提前 dW。临时构建开关仅用于 N/O
独立消融，采用后须删除实验开关。尚未验证或测得收益。

M 的 EP16 T8192/20480 完整 Torch 集成检查均已通过，与 K 一样每规模
16 ranks 的四组梯度 parity、分配失败重试、非法路由恢复和 48 个 live-tape
reuse 检查均通过。热点切换包含本 rank 尺寸不变、其它 rank 尺寸改变的情形，
证明不能省略的全局 changed 确认没有被本地缓存命中替代。保留 M 作为已验证
的控制面复用里程碑；200 样本的最终性能验收仍在进行，不能从集体操作数的
减少推断所有规模已达标。

N/O 的 EP16 8K/20K 四组梯度 hash 均与 c6be309 一致，repeat_equal=1，
但相邻 BWD wall 消融（µs）为 M 2866.660/6679.420、N 2873.980/6715.170、
O 2868.700/6713.820；V3 为 2838.730/7006.340。Reduce 尾部缩短未转化成
总时间收益，两个候选均不保留，临时 owner-wait 开关已删除。

M 的 50 warmup + 200 measured 实际循环确认（不裁剪慢样本）：

| 拓扑 | tokens/rank | M wall µs | 相邻 V3 wall µs |
|---|---:|---:|---:|
| EP8 单机 | 8192 | 5914.704 | 6458.674 |
| EP8 单机 | 20480 | 10614.076 | 10986.694 |
| EP16 双机 | 8192 | 9630.475 | 9243.788 |
| EP16 双机 | 20480 | 16630.619 | 17310.825 |

单机两个长档与双机 20K 均领先，但双机 8K 尚未通过。双机 8K 的 PF/V3
FWD API p50 为 4613.630/4795.851，BWD API 为 4288.800/4327.790；各自
分位数不能相加代替 total 的分位数。完整循环中仍有主机尾部抖动，不用较快的
单项或旧轮次掩盖这次 total 退步。日志 `peer_reuse_ep8_cycle_200.log`、
`v3_peer_pair_ep8_cycle_200.log`、`peer_reuse_200_ep16_cycle_node*.log`、
`v3_peer_pair_200_ep16_cycle_node*.log`。

P（`pf_backward_activation_k128`）：激活梯度 dU/dX 改用 K128、两级 TMA，
dW 保持 K64/三级。拆开两阶段的共享内存 stride，复用同一 arena，不把 dW
也扩大成 K128；激活的 A 按两个 K64 swizzle slab 寻址，B 的 MN-major TMA
box 扩成 K128。每个输出仍按原 K16 WGMMA 顺序累加，BF16 checkpoint 不变。
该候选仅开始构建，须验证全部梯度 hash、资源用量及两种拓扑两个长档性能，
不能仅凭 WGMMA wait 次数减半判断有收益。

P 的 EP16 两个长档四组梯度 hash 与 c6be309 完全相同，repeat_equal=1；
静态 unified kernel 仍 REG168/SHARED2048/LOCAL0，GIN STACK208。
但 BWD wall 为 2869.920/6681.320 µs，基本等于 M 的 2866.660/6679.420；
EP8 为 1793.470/4637.070，也未稳定超过 J/M 的算术实现。EP8 实际循环
200 样本为 5852.751/10505.897 µs，不能将这一小幅主机差异归因于 GEMM。
**不采用 P**，描述符、共享内存拆分、调参项均撤回；完整 Torch/sanitizer
门禁未继续。检查既有 autotune 单测时发现两个默认宏检查漏列已有 GIN 参数，
与 K128 项无关，单独记录，不能声明该轮单测全通过。

Q（`pf_backward_dispatch_m128`，回到 M 后独立消融）：训练 main 生成的
Backward 本地 dY 发送置换从 expert 间 32-row stripe 改为 128-row stripe，
与激活计算消费块对齐。保持 expert 间交错，不一次排空某个 expert；只改变
传输遍历顺序，不改变 canonical 行号、tape、K 累加或静态 compute ownership。
候选开始构建，待验证和测量；不同时保留 P 的任何变更。

Q 的 EP16 两个长档四组梯度 hash 与 c6be309 一致、repeat_equal=1。
BWD wall 为 2860.570/6740.220 µs；8K 仅略好于 M，20K 退步，**不保留**。
生成的本地 dY 置换恢复 32-row stripe，没有将这次尝试变成新配置项。
上述 autotune 测试漏列已有 GIN 参数的两处正则已修正，既有 14 项全部通过，
未新增测试用例。

R（`pf_backward_owned_completion`，独立基于 M）：整块激活链只有一个
cluster owner，省去 dU/dAB 的中间全局完成计数及 phase2/3 发布；dX 由该
owner 直接 release 块级 ready，不再先 fetch_add。欠填充的分裂任务仍保留
原计数和 phase 协议，TMA store、阶段间 cluster join、全本地 dX 完成门槛
以及 Return ready 均不变。静态 dW 用自身区间终止，不再为已无消费者的
全局 dW completed counter 做原子操作；每 CTA 缓存 dW 首尾时间戳，退出前
汇总，保留整体计时但避免逐任务争用。候选开始构建，尚未验证或采用。

R 的 EP16 8K/20K 四组梯度 hash 与 c6be309 一致，repeat_equal=1，资源无新增
spill。BWD 为 2867.270/6678.090 µs，与 M 持平；实际 200 样本循环为
8677.758/16346.678 µs，优于相邻更早 V3 的 9243.788/17310.825，但 native
未改善、Forward 也未改，不能将全部循环差异归因于同步删除。**不保留 R**，
避免为没有实测 device 收益的计时汇总引入额外共享状态和控制分支。

S（`pf_backward_row_prefetch`，重新基于 M）：Backward 六处 payload 搬运
（本地 Dispatch/Return、GIN Dispatch/Return pack、两种 Receive）复用
Forward 的 RowPrefetchPipeline/FanoutRowPrefetchPipeline，使用两个 4KB
staging slot。仍在 payload 全部 global-complete 后发布 ready/释放 RX credit，
TX 的本地完成及远端 credit 均保留。Reduce 改为 warp shuffle 广播每 lane
持有的 canonical source pointer，省去 768B shared pointer table；既有 H
必须为 256 的倍数，保证每个 vector round 的 full-mask shuffle 都有完整
warp。释放的 shared memory 让双槽搬运仍有机会装入原单 CTA 资源上限。
正在构建；需要严格检查资源、数值/生命周期和性能，尚未采用。扩充既有
BACKWARD_PROGRESS 日志的设备首尾时间戳，只在诊断开关开启时输出，不增
device 插桩或正常路径通信。

S 的 EP16 两个长档四组梯度 hash 与 c6be309 一致、repeat_equal=1，但 BWD
wall 为 2974.690/7078.530 µs，EP8 为 1872.880/4958.270 µs，均退步。
EP16 8K 的 Dispatch/dU/dX 虽加快，绝对时间戳显示 dW 约 1.98 ms 已结束，
最终 Reduce 却到约 2.8 ms 才结束；单机也回退，不能直接归因于网络排空。
**不采用 S**，双槽复制与 shuffle Reduce 均撤回，没有改变默认参数。
下一步在 M 上增加 sender/transport retirement 时间点，区分通信收尾与
最终归约尾部；诊断 wall 含进度轮询，不当作性能结果。

M 的 retirement 诊断（`retirement_ep16_t8192_profile.log`）显示：dX 完成于
1.70–1.81 ms，dW 于 2.05–2.17 ms，GIN sender retirement 于 2.46–2.59 ms，
通信域退出于 2.64–2.67 ms，最终 Reduce 于 2.68–2.73 ms。各 rank 使用
自己的 start 作原点；不能把跨 rank 时间戳相减。主要尾部在 GIN，其后的
机内 End/credit 排空也有约 0.1 ms 串行成本，剩余 Reduce 算术并非主要原因。

T（`pf_backward_early_close`）：静态 Return owner 的最后一个 M-block
发布完成后，立即向本 CTA 的各本地 peer 发送 End，publisher warp 继续接收
通知；GIN/Dispatch 仍可运行。通信主循环退出后再 join 所有 incoming End、
consumed、outgoing credit，并执行原 GIN flush。把共享通知 helper 分为
close/wait，Forward 和旧动态路径仍通过原 drain wrapper 顺序执行两步。
无新参数、无新增 payload buffer，不改变 dW 门槛、梯度运算或 credit 语义。
正在构建，尚未采用；必须覆盖热点/零入站及连续复用，不能只测均匀路由。

T 的 EP16 两个长档四组梯度 hash 均与 c6be309 一致，repeat_equal=1，
但 BWD wall 为 2872.930/6694.030 µs，未超过 M。**不采用 T**，close/wait
拆分与提前 End 分支全部撤回，没有继续其完整 Torch 门禁。约 0.1 ms 的
transport retirement 区间不能全部归因于 End 通知；改变发 End 时机没有
带来总耗时收益。

U（`pf_backward_local_vector`，重新基于 M）：只改变同一 GPU 内的 GIN TX
Dispatch/Return 打包及 Return RX→contribution 放置。由整个 copy warp
执行 16B 向量 load/store，复用一个 row helper，不再为这些本地搬运逐行
等待单 lane TMA staging。跨 GPU 的 dY fanout、本地 Dispatch/Return 仍走
原 TMA；所有线程完成 system fence 和 copy-domain join 后才发布 packed/
ready、释放 RX credit。共享内存容量、通信包和计算顺序均不变。补充最后一个
Return terminal Put/Receive 的时刻，区分 payload 尾部和纯协议 retirement。
候选正在构建，尚未验证或采用。

U 的 EP16 两个长档四组梯度 hash 完全一致，但 BWD wall 为
4451.110/10616.500 µs，显著退步；资源仍为 REG168/SHARED2048/LOCAL0，
GIN STACK208，不能只凭寄存器报告判断搬运效率。**不采用 U**，三个本地
SIMT copy 改动撤回，保留用于定位的 terminal Put/Receive 时间戳。

V（`pf_backward_receive_priority`）：仅静态网络 BWD 在本轮已接收到 GIN
payload 时暂缓本地 Return，先持续释放 RX credit；没有收到包的下一轮再
处理原固定 owner 的 Return。传输仍有限、静态 ownership 不变，计算和
单机路径不变。用来验证 M128 本地 Return 是否让共享 copy warp 停收太久。

W（`pf_backward_return_ports4`）：在 V 上只将现有 GIN Return ports 从 2
调到 4；sender 配额仍 16。增加独立 Return receive owner/credit 流，不改变
每 slot 容量、K 归约或消息含义。使用 build 内临时生成配置，尚未修改生产
profile；V/W 都在构建，须分别验证和比较，不预记收益。

V/W 的 EP16 两个长档均通过 native 重复验证；V 四组 hash 已核对与 c6be309
完全一致。V 的 wall 为 2859.240/6663.720 µs，相比 M 仅小幅变化，8K 仍未
达标；20K 本地 Return 区间扩至 1057.280 µs。W 为 3116.870/7247.540 µs，
两个长档都退步。**V/W 均不采用**；恢复原先每轮推进接收和 Return 的次序，
生产 ports 仍为 2。扩大独立通道而不增加 sender，并不保证吞吐更高。

W 的两档四组 hash 随后也核对为与 c6be309 完全一致。上述失败候选没有残留
在生产数据路径；本轮只保留诊断时间点，并提交已有调参检查的宏清单修正。
现有调参/资源/launcher 检查共 34 项通过，没有新增细粒度测试。

X1/X2 使用原 M 调度，分别测试 `(Return ports, sender CTAs)=(4,32)` 与
`(2,32)`，对照 M 的 `(2,16)`。两个值均为既有调参项，使用 build 内临时
profile，不新增源码分支。与 W 的比较包含撤回 V，不能单独归因为 sender
数量变化；与 M 的比较才是这轮配置消融。尚未构建验证或采用。

X1/X2 均通过 EP16 两个长档 native 重复验证，BWD wall 分别为
3302.000/7997.390 和 3032.180/7416.470 µs，均退步。**不采用 X1/X2**，
生产仍使用 2 ports/16 senders。更多 sender 减少 receiver 数，并改变同
CTA 中通信/计算的资源竞争，不能把增加并行度直接等同于更高吞吐。

Y1/Y2 重新使用原 2 ports/16 senders，分别消融 math/control 寄存器配额
`216/72`、`208/88`，对照原 `216/64`。384-thread CTA 初始寄存器预算为
`168*384=64512`；原配置用 `216*256+64*128=63488`，Y1 只使用剩余的 1024
个寄存器，不降低 math 配额；Y2 则显式测试向通信/producer 再倾斜的取舍。
两者均使用既有参数和 build 内配置，不改 GEMM 的 MNK、归约顺序或协议。
尚未构建、验证或采用。

Y1/Y2 的 EP16 两个长档 native 重复验证通过。BWD wall 分别为
2846.080/6636.550、2840.620/6615.600 µs；原 M 为 2866.660/6679.420。
静态 GIN 栈由 208B 降为 152B/72B，静态单机仍 0B；不能将 LOCAL0 当成
没有任何栈访问的证明。Y2 的 8K device kernel 为 2713.950 µs，相邻更早 V3
为 2717.220，但 wall 为 2840.620 vs 2838.730，仍应称接近，不能宣告全面
领先。下一步核对 Y2 的单机性能，生产寄存器配置尚未修改。

Z（`pf_backward_early_rx_flush`，基于 Y2）：静态网络 BWD 的 receiver
在自己负责的 Dispatch/Return RX channels 全部收到 terminal 后，即执行
原有 ACK flush，不再统一拖到本 rank 的所有 sender 退出、机内通知排空
之后。该 CTA 不再有后续 ACK，仍可继续自己的本地 Return；其它 CTA 的
payload、计算和 Return 可并行推进。每个 receiver 仍只 flush 一次，末尾
保留尚未执行时的原 drain；动态路径、每槽 credit/TX local counter 均不变。
不删除任何完成检查，也不增加 packet buffer。候选待构建验证。

Y1/Y2 的 EP16 两档、Y2 的 EP8 两档，所有 rank 的四组梯度 hash 均已
核对为与 c6be309 完全一致，repeat_equal=1。Y2 的 EP8 BWD 为
1774.260/4592.870 µs，相邻 V3 为 2127.020/5050.010 µs，耗时降低
16.59%/9.05%；没有单机回退。

Z 的 EP16 两档 native 重复验证通过，BWD 为 2838.350/6662.720 µs。
8K 相比 Y2 持平，20K 退步约 0.7%，没有消除关键路径。**不采用 Z**，
提前 flush 分支和状态均撤回，仍在原 drain 点执行完整 ACK flush。

AA1/AA2 以 Y2 为基础，将已有 Backward sender CTA 配额由 16 分别改为
8/12，其余参数和代码保持不变。目的是检查较少 sender 能否改善它们与
计算、receiver 的竞争；不是继续增加通道。仅 build 内配置消融，尚未采用。

AB 将 Y2 的 math/control 配额进一步改为 200/104，CTA 总预算仍 64512，
sender 仍 16。用于寻找 Y2 收益的边界，不能由栈大小直接推断速度。

Y2 的 EP16 T8192/20480 完整 Torch 回归均通过：每档 16 ranks 的 output、
dX/dGate/dW1/dW2 parity，非法/重复 route 拒绝，分配失败 ordinal 3/5
一致失败后重试，三份 live tape 热点/扩缩容，以及 EP8 subgroup inference
graph/offload。每档 48 个 peer-reuse marker；已有 CPU 检查 34 项通过，
未新增细粒度测试。Z 的四组 hash 也已确认与 c6be309 完全一致。

AC（`pf_backward_launch_cache`，基于 Y2）复用已有 backend 的
`prepare_unified_launch`：同一 kernel、资源配额、启动几何只检查一次寄存器
预算、设置 shared-memory/stack 并查询 cluster occupancy。统一 Backward
不再为不会启动的 split kernels 设置属性；首次检查和驻留门槛仍保留。
不增加新缓存类型，不改设备工作/通信或数学顺序。该消融属于 host 提交开销，
CUDA event 区间也可能包含 host 未提交后续 kernel 的空隙，不能称纯 GEMM
加速。正在构建和验收，尚不预记收益。

Y2 经上述长文回归后采用到 checked-in H2048/I512 profile；不是修改所有
模型的通用默认值。其 EP8 实际训练循环（50 warmup + 200 measured）为
5947.5945/10442.4655 µs，较此前相邻 V3 的 6458.6735/10986.6935 更快。
仍需对最终组合进行完整配对，不能拼接不同候选的最优单项宣告目标完成。
AA1/AA2/AB 均已通过 EP16 两档 native 重复验证，全部 hash 与 c6be309
一致；两机 16 卡于 08:08 UTC 复查均为 1620 MHz，开始独占性能比较。

AA1/AA2 的 EP16 BWD 分别为 3941.040/9467.750 和
3458.970/8144.130 µs，均显著回退。**不采用 8/12 sender 配额**，保留 16；
较少 sender 会让每个 owner 承担更多 channel，不能只看增加 receiver 数量。

AB 的 EP16 BWD 为 2835.600/6615.320 µs，未证明比 Y2 更快；静态 GIN
栈仍 72B，本地仍 0B。**不采用 200/104**，保留已完整验证的 208/88。
AC 为 2832.940/6616.670 µs，相邻 V3 为 2839.110/7011.880；8K 差别仅
0.22%，仍不能宣称已经形成稳定优势。AC 的两档四组 hash 已核对一致。

AD（`pf_backward_private_gin`，基于 AC）只调整统一 BWD 的 GIN 提交资源
模式：实际 context 数大于全部 worker CTA 数时，`1+cta` 无别名，每 CTA
只有一个线程调用 Put 或 ACK/flush，因此使用 THREAD 独占模式。context 0
仍给 metadata；`GinExecution` 的 last-use event 串行化共享该 pool 的调用。
实际 context 不足或非 unified 路径仍保留 GPU sharing，不以请求数量冒充
实际分配。逐 slot ready、remote credit、本地 NIC counter、ACK flush 和
数据内存序完全不变；不是删除本地 credit 检查。Forward 未改。
代价是 Backward 提交路径多一个运行时模式分支，须测编译资源和端到端效果。
沿用已有 BACKWARD_PROGRESS 输出实际 contexts/workers，不新增设备状态。
尚未构建验证或采用。

AC 的 EP8 两档四组 hash 亦与 c6be309 完全一致。EP8 BWD 为
1769.390/4663.810 µs，20K 的约 1.5% 波动没有设备算法变化支撑，不能归为
launch cache 的计算回退或收益。相邻实际循环（50/200）PF 为
5889.5745/10398.4815 µs，V3 为 6354.0935/11048.4865 µs。
AC 的 EP16 T8192/20480 完整 Torch 数值、route rejection、ordinal 3/5
失败及重试、三份 live tape/peer reuse、EP8 subgroup graph/offload 均通过。
**采用 AC 的启动准备复用**，明确不将整个相对 V3 收益归于这项 host 改动；
双机实际循环仍待最终组合对照。没有新增测试框架或细粒度源码断言。

AC 补测 EP16 T20480 全热点（所有 top-k 都在 rank 0 的四个专家，实际
R=1,310,720；其它 rank 零入站），16 ranks parity 及 graph/offload 均通过。

AD 两档 native 四组 hash 与 c6be309 一致；诊断实际报告
`gin_contexts=133, gin_workers=132`，确认采用了独占分支。静态 GIN 资源仍
REG168/STACK72/SHARED2048/LOCAL0，本地 STACK0。正式 BWD 为
2824.170/6609.280 µs，相比 AC 仅小幅变化；尚未完成实际循环与生命周期门禁，
不把 Forward AZ 的 6%–7% 历史收益套用到 Backward。

AE1/AE2 在 AD 上只消融已有 GIN chunk 行数 32/128，对照 64。PF main
使用独立 Dispatch Q8/Return Q16，保持不变；该参数调整训练 BWD 的旧式
wire slot（以及非 PF 的 legacy 路径），不改变 M128 或 dW K 顺序。
用来区分小包的流水延迟与大包的提交摊销，ring 深度、sender/receiver 配额
和 credit 语义不变；相应 GIN payload 容量也会变化，必须计入内存比较。
仅 build 内 profile，正在构建，尚未采用。

AD 的 EP16 两个长档完整 Torch 回归、20K 全热点均通过，覆盖分配失败后
重试、多个 live tapes、零入站 rank 和 inference graph/offload。8K native
仅插桩 rank 1/9 的 WeftMoE kernels，memcheck 两份报告均 0 errors，所有
16 ranks 的四组 hash 与 c6be309 一致；不等于全 API/全 rank racecheck clean。
首次插桩 rank 0/8 未进入算子：sanitizer 子进程保留 bootstrap pipe 写端，
多机 launcher 等待 EOF 超时。改用 1/9 规避 launcher 限制，没有修改协议。

AD 的 EP16 实际循环（50/200，B768/D6）为 9667.800/19257.550 µs，
相邻 V3 为 9215.203/17214.686。**两档都未达标**，不能用 native 小幅
收益替代端到端验收。AC/AD 尚未做稳定的完整循环直接配对，不能据此将
20K 的波动全部归因于 GIN sharing 模式。

AE1/AE2 的四组梯度 hash 均与 c6be309 一致。EP16 native BWD 分别为
2837.990/6921.580、3093.880/6729.060 µs，均未优于 64 行；**不采用**。
生产 profile 保留 chunk 64，未将额外通信缓冲容量引入正式实现。

AF（`pf_backward_packed_storage`，基于 AD）：eager 训练每次发布权重时，
原来均原生 cudaMalloc/cudaFree 一份 interleaved W1；native 算子基准在
计时前准备权重，因此不会反映这一开销。新增调用方提供 packed-W1 scratch
的 overload，复用现有 CudaTensorBuffer 校验/所有权机制；Torch 入口使用
自身缓存分配器。**每次仍重新 interleave 当前 W1**，不是按地址缓存权重值。
原 overload/ABI 和 native 分配路径保留。lease 保持 scratch 存活；准备失败
时先等 packing ready 再释放，避免缓存分配器提前回收。仍在同一 collective
allocation agreement 内准备，不增加 collective。候选正在构建，收益待测。

AF 已构建；EP16 实际循环（50/200，B768/D6，1620 MHz）为
6498.2745/14112.6975 µs，相邻冻结 V3 为 9119.4395/17659.7615，
耗时降低 28.74%/20.09%。对照前一轮 AD 的 9667.800/19257.550，AF
主要消除了训练入口的原生分配/释放停顿，并非 GEMM 加速；native 基准在
计时外绑定权重，不能拿该结果推断 native GPU 阶段同幅改善。每次 publication
仍重排完整 W1，CPU/GPU 工作没有被移出实际循环计时。已有 34 项 CPU 检查
通过；完整 Torch 回归和最终同候选十二项配对仍在进行。

补记 AE 的内存代价：相同 B1024/D4、8K 夹具中，各 rank 的 pool_bytes
为 chunk32 966206048、chunk64 1138827872、chunk128 1484071520；
workspace_bytes 均为 419800280、tensor_bytes 均为 293863424。20K 的 pool
分别 1485969504/1658591328/2003834976；每档差值只来自 chunk 配额，不能
把更大 payload 预算当成免费调整。三种配置最终仍选择原 chunk64。

AD 在单机 EP8 的 200 measured native BWD 为 1771.860/4634.810 µs，
相邻 V3 为 2128.690/5055.480；本地 kernel 没有 GIN mode 分支变化。

AF 的 EP16 两个长档完整 Torch 回归均通过：每档 16 个 parity、16 个原地
权重刷新、32 个 allocation failure、16 个 retry、48 个 peer reuse，以及
16 个 multilayer graph / offload marker。静态 BWD 编译资源与 AD 一致：
REG168/SHARED2048/LOCAL0，GIN STACK72、本地 STACK0。
**采用 AD 的条件独占提交及 AF 的应用侧 packing scratch**；最终同候选的
EP8/EP16 各档 FWD/BWD/cycle 性能门禁仍待下面统一汇总，不提前标记完成。

## 最终统一验收：2026-10-01，eb371c2

这是当日固定路由/固定 tokens 的历史验收；2026-10-02 已撤回 M 并修改基准，
本节不作为新实现或动态 workload 已通过性能门禁的证据。

最终候选统一为 `build/pf_backward_packed_storage/`，不是拼接不同候选的
最佳单项。两节点均为 8×L20X、132 SM/GPU，实际 graphics clock 为
1620 MHz；结束时再次确认 16 卡时钟与无残留 GPU 作业。H2048/I512、
E64/top-k4、BF16，tokens 均为 **每 rank** 的数量。
EP8 是单机八卡，B512/D8；EP16 是两机十六卡，B768/D6。
本次不重新测试 1/64 的短序列性能。

计时口径：

- Native FWD/BWD：10 warmup + 200 measured，取每轮 rank-max wall 的 p50。
  FWD 包括 Notify 和 main，开启 retain-state；固定形状的 tensors 在 warmup
  后复用，权重在计时外发布。BWD 复用已保存 X/GU，前置 FWD 不计入 BWD。
- 实际循环：50 warmup + 200 measured，直接计时应用入口的
  `forward → backward → 释放 tape/gradients → synchronize`，包括每步权重
  repack、精确 tensor 分配和 collective 准备边界。不是两个独立 p50 相加。
- 冻结 V3 native 为 `build/static_inference_v3_20260923/test_cuda_notify`，
  应用扩展为 `build/pf_training_v3_reference/`；V3 使用其原 profile 和动态路径。
  AF 的缓存分配器优化是通用的 host 优化，V3 若另行移植也可能受益；不能将
  下表实际循环的全部差距归于静态任务划分或 GEMM。
- 构建结束后才测性能。EP16 独占两节点；最终 EP8 native 在第一台机器、
  EP8 cycle 在第二台同时运行，各自与同机 V3 配对，没有共享 GPU。

### 训练 Forward

单位 µs，越小越好。

| 拓扑 | tokens/rank | PF | V3 | 耗时降低 |
|---|---:|---:|---:|---:|
| EP8 单机 | 8192 | 1497.320 | 1723.900 | 13.14% |
| EP8 单机 | 20480 | 3209.960 | 3570.160 | 10.09% |
| EP16 双机 | 8192 | 3167.910 | 3434.550 | 7.76% |
| EP16 双机 | 20480 | 6994.140 | 7238.990 | 3.38% |

### 缓存 X/GU 的 Backward

| 拓扑 | tokens/rank | PF | V3 | 耗时降低 |
|---|---:|---:|---:|---:|
| EP8 单机 | 8192 | 1765.540 | 2127.500 | 17.01% |
| EP8 单机 | 20480 | 4606.750 | 5045.690 | 8.70% |
| EP16 双机 | 8192 | 2840.510 | 2852.200 | 0.41% |
| EP16 双机 | 20480 | 6620.240 | 7002.220 | 5.46% |

EP16 8K 的 p95 为 PF 2858.51、V3 2893.38 µs，但优势仍小，应理解为
接近持平、此轮小幅领先。200 样本内的 iid bootstrap 中位数差 95% 区间为
7.58–16.09 µs（2000 次重采样，seed917）；它只描述本轮样本，不覆盖不同
运行之间、设备之间的系统性波动，不据此声称普遍显著提升。

### 实际 FWD+BWD 循环

| 拓扑 | tokens/rank | PF | V3 | 耗时降低 |
|---|---:|---:|---:|---:|
| EP8 单机 | 8192 | 3925.061 | 6400.766 | 38.68% |
| EP8 单机 | 20480 | 8673.361 | 10948.840 | 20.78% |
| EP16 双机 | 8192 | 6498.275 | 9119.440 | 28.74% |
| EP16 双机 | 20480 | 14112.698 | 17659.762 | 20.09% |

### 正确性、生命周期与复现

最终候选 EP8/EP16 × 8192/20480 的 native 两次运行，全部 rank 的
dX/dGate/dW1/dW2 hash 与 c6be309 一致，repeat_equal=1。
同四个形状的完整 Torch 回归均通过：output 和四组梯度 reference parity、
非法/重复 route 拒绝、单 rank 分配失败后的全体一致失败及重试、三份 live
tape 的热点/扩缩容/零入站迁移、原地 W1 更新后的完整 FWD/BWD，以及
EP8 inference graph/offload。每个 EP8 case 有 8 个 weight-refresh marker，
每个 EP16 case 有 16 个；没有缓存旧权重值。34 项既有 CPU 检查通过。

AD/AC 的额外 20K 全热点与部分 rank memcheck 证据见前文；AF 没有改变
payload 协议。这里不把旧 racecheck 的未决项改写成 sanitizer 全通过。
Backward 的部分通信 payload 仍按实际 R 分配，并非全部改成有界 ring。
没有恢复动态 GEMM 领用，也没有放宽「全部本地 dX 完成后才执行 dW」的边界。

日志均在 `build/pf_training_timeline/`：

- EP8 native：`{packed_storage_final,v3_packed_final}_ep8_t{8192,20480}_{forward,backward}.log`。
- EP16 native：`{packed_storage_final,v3_packed_final}_t{8192,20480}_{forward,backward}.log`。
- 实际循环：`{packed_storage_final,v3_packed_final}_ep8_cycle.log`；
  `{packed_storage_200,v3_packed_pair_200}_ep16_cycle_node{0,1}.log`。
- Native 数值：`packed_storage_final_ep{8,16}_t{8192,20480}_validation.log`。
- Torch：`packed_storage_final_torch_ep8_t{8192,20480}.log`；
  `packed_storage_torch_ep16_t{8192,20480}_node{0,1}.log`。

最终候选 SHA256：

```text
native    61385a316d22af0553de1ece81e48488b7a77614318f27391c04fa1327e39091
library   02aad150be56b69fbefe77531e21047f4064af875bec7eb55a56f770e90902f5
extension 50cca0cc124fb884ceb3c2cdea3b27e0ce76588bdda3b024e6ea6263278d7320
V3 native cc0471af98d0fcbcfa52b9c9669ae2e20fb9b8077aa36e483a11a4be6331a699
V3 ext    39ab0b7f16fe69bbaaa4b0887da0e54258874ac1ea9bb55a97e0dab86b7c55ba
```
