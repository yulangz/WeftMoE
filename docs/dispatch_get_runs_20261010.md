# PF Dispatch：有序节点目录与连续 Get（2026-10-10）

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
