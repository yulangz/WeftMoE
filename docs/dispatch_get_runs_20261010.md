# PF Dispatch：有序节点目录与连续 Get（2026-10-10）

结论：连续 Get 是小优化，不是这次 Dispatch 差距的主要解法。干净版交叉复测
8K 约快 0.54%，20K 基本持平；保留 `6e5d730` 为候选，**不替换上一版默认/基线**。

## 本轮发现

上一轮保留的借用 warp 路径仍是：节点去重 Get 到 RX ring，目的 GPU 经 NVLink
pull 至 X ring，逐行发布 ready；Return 不变。本轮不采用额外拷贝描述表，负结果
见 `dispatch_descriptors_20261010.md`。

`node_tokens` 原本通过各 lane 的 atomicAdd 追加，节点去重后并不按源地址排序。
因此即使许多源 token 相邻，也仍然逐 token 发 8KiB Get。

本轮候选做两件事：

1. 一个 warp 静态负责一个 node 的目录，扫描源 token，以 ballot + lane prefix
   写入有序列表。只重排 metadata，不 pack X；保留每 token 的精确 fanout。
2. 在一个 packet 内，把源 token 编号连续的段合并成一个 Get。地址出现缺口即
   断开，不读取未命中 token，不跨 packet。flush、ready、RX credit 完全不变。

节点数超过规划 warp 数时，以 warp 的静态 node residue 分轮处理，没有写死两机；
实际仅验证两机。列表和 ring 使用原有空间，不新增 buffer。

## 同一 binary 的逐步消融

源码 `fb8332e`，artifact `build/dispatch_runs_{phases,fused}/borrow`。
使用 `WEFTMOE_PF_DISPATCH_COPY=6`，关闭额外描述表；
`WEFTMOE_PF_DISPATCH_RUNS=0/1/2` 分别选择以下三行。

两机 EP16，H4096/I2048/E256/topK6/BF16，SM1620MHz；随机 token 数（std=10%）
与随机路由（bias std=0.5），seed=123，32 priming + 8 warmup + 24 samples。
每步最慢 rank 后取 p50，各候选一次完整测量。

| 设置 | Get 次数/packet 8K / 20K | Dispatch ms 8K / 20K | 平均 Dispatch GB/s 8K / 20K | 完整 MoE ms 8K / 20K |
|---|---:|---:|---:|---:|
| 原目录、逐 token Get | 31.85 / 31.86 | 2.678 / 5.976 | 24.38 / 27.74 | 10.817 / 23.138 |
| 仅有序目录 | 31.85 / 31.86 | 2.631 / 5.952 | 24.85 / 27.93 | 10.750 / 23.298 |
| 有序目录 + 连续 Get | 1.43 / 1.43 | 2.551 / 5.825 | 25.43 / 28.61 | 10.697 / 23.092 |

带宽是有效跨机 payload 的平均每卡单方向 GB/s；不含 NVLink 字节，不是 TX+RX。
连续 Get 后最忙 rail 约 30.43/33.10GB/s。Return 平均仍为 35.86/38.12GB/s，
最忙 rail 44.45/45.84GB/s。本轮没有改 Return。

请求数减少约 95.5%，但纯 Dispatch 只快 4.7%/2.5%，完整 MoE 只快 1.1%/0.2%。
95.5% 依赖本次两机 topK6 负载的节点命中密度，不能外推到更多机器或稀疏节点命中。
后者连续段会缩短；仍只传实际命中的 token，但一次 Get 能合并的行数会减少。
这说明继续减少 Get 发射次数不足以消除当前差距；不能把请求数下降当成等比例
带宽或端到端收益。剩余开销还在本地转发、ready 发布与调度等待中，尚未由这组
消融精确分解。仅有序目录的完整 MoE 正负波动也说明必须做交叉复测。

576 个 rank/shape/step 校验通过，各设置 workload、输出 hash 和 workspace 相同。
526 个双节点时钟观测均为 SM1620MHz。记录：
`benchmarks/dispatch_runs_phases_20261010.json`、
`benchmarks/dispatch_runs_fused_20261010.json`。

## 干净候选

`experiment/pf-dispatch-runs-clean-20261010` 的 `6e5d730` 基于上一轮干净实现
`8c47a46`，仅修改 3 个文件（39 行新增、6 行删除）。没有上述诊断开关、请求
计数或额外描述表；目录改动只开启于借用 warp 的多机 inference，训练 planner
和单机路径不变。干净 artifact 与旧版的交叉复测结果单独记录，不套用消融版数字。

### 干净 artifact 交叉复测

按 old_a → new_a → new_b → old_b 顺序运行，旧版为冻结的 `8c47a46` artifact，
新版为 `6e5d730` 的 `build/optimized/borrow`。测试前完成所有编译。

| 运行 | 完整 MoE 8K ms | 完整 MoE 20K ms |
|---|---:|---:|
| old_a | 10.890464 | 23.458848 |
| new_a | 10.835056 | 23.436448 |
| new_b | 10.831408 | 23.570112 |
| old_b | 10.892592 | 23.514832 |
| 旧版两轮 p50 的均值 | 10.891528 | 23.486840 |
| 新版两轮 p50 的均值 | 10.833232 | 23.503280 |

8K 延迟降低 0.54%；20K 增加 0.07%，在本轮波动范围内，不能声称有改善。
eager 两轮 p50 均值分别为旧版 11.168/23.714ms、新版 11.110/23.803ms，
同样没有显示 20K 收益。这里没有训练缓存，也不是缓存已准备好的局部 FC 测量。

384 个 rank/shape/step 校验通过，workload、输出 hash、workspace bytes 完全相同；
最大 relative L2 <9.24e-5。356 个双节点时钟观测均为 SM1620MHz。
记录见 `benchmarks/dispatch_runs_clean_20261010.json`。

资源检查：pull 的 stack/spill 均为 0；entry spill 仍为 56B stores / 48B loads。
Get 的静态 spill 从 252/384B 变成 220/420B，Put 从 136/276B 变成 164/344B；
因此不声称控制函数编译结果完全不变。shared memory 仍为 222KiB，math/control
寄存器预算仍为 208/88。GEMM 保持四条 N256 WGMMA / K64、一次 wait。
单机 PF 规范化 SASS 与旧版完全相同：
`bf19584d6850f36af1e0506f91d5e4741c6b858f89ed0a39a346c9291f7204ce`。

小 ring 校验：B128、route bias std=8，8K/20K 各 3 次 priming + 3 次测量，
共 96 个 rank/shape/step 校验通过，最大 relative L2 <7.83e-5；84 个双节点
时钟观测均为 SM1620MHz。此组仅检查正确性，不作为正式性能结论。
记录见 `benchmarks/dispatch_runs_clean_stress_20261010.json`，其中也保留了最初
传入不支持的 `--priming-steps` 导致的 argparse 失败；修正后使用样本数自动确定
priming，成功启动 `new_stress_b`。两机所有 GPU 均已释放（0MiB）。

源码保留在上述干净实验分支；主分支只同步文档与原始结果，没有替换默认源码。

### 剩余方向

独立 Dispatch 的 25.43/28.61GB/s 仍未达到历史同口径 V3 的 32.08/33.60GB/s。
更值得继续单独量化的是：机内转发逐行完成与 ready 发布、同一静态 worker 等待
未 ready packet 时的队首阻塞，以及这些等待怎样延迟 FC1。可以在不恢复全 CTA
动态任务队列的前提下，尝试在每个 worker 自己的静态 packet 集合内优先处理 ready
项。这是后续假设，本轮没有实现或宣称其收益。
