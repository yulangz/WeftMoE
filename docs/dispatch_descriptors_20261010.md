# PF Dispatch：本地化拷贝描述表消融（2026-10-10）

结论：**不保留额外描述表。** 纯 Dispatch 只有小幅改善，完整 MoE 无收益；
8K/20K 还分别增加 18/42MiB 本地 metadata。继续使用上一轮的干净版本 `8c47a46`。

## 目的与边界

沿用借用一个 pull warp 的 EP16 路径，继续检查上一轮 warp 并行路由筛选后
残留的 metadata 开销。上一轮结论见 `communication_phases_20261010.md`。

本轮没有改变跨机 Get、节点去重、RX/TX slot credit、Return、GEMM 参数、
264 个本地 pull worker 或单 warp 4KiB bulk-copy staging。

## 候选

1. `coop_bulk`（mode 6）：控制组；在转发时读取 rail 上的 token、expert、
   location、gate，warp 并行筛选，lane 0 逐行 bulk copy。
2. `compact`（mode 13）：在 planner 的节点内 location 发布屏障之后，
   每 warp 处理一个 packet，仅保留本卡命中的行，生成本地连续的 16B
   `{logical_row, packet_token, gate}` 描述。转发只读取本地描述，不再逐 packet
   扫描远端的全部 token × topK 或追踪 peer location 指针。
3. `compact_local`（mode 14）：在 2 的基础上，warp 同时解析该 worker 静态归属
   的下一组 32 个本地候选行；不改变行的归属、拷贝顺序与逐行 ready 发布。

描述表为每 packet 预留 `dispatch_packet_tokens × topK` 个条目，支持全部 route
命中同一卡的极端情形；只写实际命中的条目，不触碰未 ready 的 RX payload。
数据由本次调用重新规划，不复用上一调用的路由。planner 结束前增加一次节点内
grid join。原来的 packet ready acquire、逐行 copy 完成、X-ready release 和
全 packet 读者 retirement 顺序保持不变。

## 测试口径

- 两机 EP16，H4096/I2048/E256/topK6/BF16，1620MHz。
- 8K、20K 是 nominal；每步 token 数按标准差 10% 的正态分布变化，
  路由 bias std=0.5，seed=123；每个候选使用相同 workload。
- 32 次 priming、8 次 warmup、24 个测量样本；先每步取最慢 rank，再取 p50。
- 同时测独立 Dispatch/Return 与完整 GEMM kernel；额外规划时间计入完整 kernel。
- GB/s 是跨机有效 payload 的平均每卡单方向带宽，不是 TX+RX，也不包括 NVLink 字节。
- 固定 workspace/GIN 初始化与权重准备不计入调用时间；描述表的每调用构造计入。

消融源码在 `experiment/pf-dispatch-descriptors-20261010`；它继承了已有诊断开关，
不是默认实现。使用独立 artifact，避免覆盖上一轮的冻结结果。

## 结果

同一份 native，模式切换；各模式一次完整的 24-sample 测量。源码 `e9e6d91`，
artifact 分别为 `build/dispatch_descriptors_v2_{phases,fused}/borrow`。

| 模式 | Dispatch 8K / 20K ms | Dispatch 平均 GB/s 8K / 20K | 准备阶段 8K / 20K ms | 完整 MoE 8K / 20K ms |
|---|---:|---:|---:|---:|
| coop_bulk | 2.616 / 5.934 | 24.82 / 27.78 | 1.069 / 1.255 | 10.787 / 23.207 |
| compact | 2.581 / 5.901 | 25.44 / 28.27 | 1.130 / 1.372 | 10.828 / 23.220 |
| compact_local | 2.535 / 5.883 | 25.81 / 28.46 | 1.137 / 1.374 | 10.814 / 23.245 |

“准备阶段”是独立通信 benchmark 中的 route_exchange 计时，包含 planner、输入
staging 和阶段边界同步，不是只测 build_pull_rows。完整 MoE 包含所有新增规划。
Return 平均仍约 35.95/38.15GB/s，没有改变 Return 协议。

解释：上一轮 warp 并行筛选已经消除了大部分串行 metadata 开销；进一步预计算
不能把纯 Dispatch 的小收益转成端到端收益。因此不应为这组结果增加常驻缓冲和
额外规划 join。这里不把 <1% 的完整 MoE 差异解释为确定的性能退化。

6 次成功启动共 576 个 rank/shape/step 校验，三组的 workload 和输出 hash 一致；
536 个双节点时钟观测均为 SM1620MHz。新增消融分支导致 pull 函数静态报告
52B spill stores / 80B spill loads；这些代码不并入干净实现。

原始记录：`benchmarks/dispatch_descriptors_phases_20261010.json`、
`benchmarks/dispatch_descriptors_fused_20261010.json`。早期 `phase_a_compact`
被 host 模式编号检查拒绝，未进入 GPU；已修正，不计入表格。`phase_a_coop_bulk`
是修正前控制组，2.629/5.978ms，仅作环境一致性检查，不混入本表。
