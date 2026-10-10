# PF Dispatch：本地化拷贝描述表消融（2026-10-10）

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
