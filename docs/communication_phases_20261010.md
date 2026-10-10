# Dispatch / Return 分阶段通信诊断（2026-10-10）

本轮目标是分别测量通信阶段，不再用 Dispatch、Return 同时运行的区间
分别计算两个“独立阶段”的带宽。生产默认配置不变。

## 结论与实测结果

**当前 PF 的主要短板是 Dispatch，而不是 Return 的单 rail RDMA 带宽。**
V3 的 Dispatch 明显领先；Return 最忙 rail 三者都在约 44–46 GB/s，
但 PF 的 rail 流量更均衡，因此同负载下 Return 整体耗时反而低于 V3。

下表为两轮独立测试 p50 的均值。GB/s 两个数字依次为
**平均卡有效带宽 / 最忙 rail 带宽**，均使用全阶段时长，不是瞬时峰值。

| 每 rank tokens | 实现 | Dispatch ms | Dispatch GB/s | Return ms | Return GB/s |
|---|---|---:|---:|---:|---:|
| 8K | V3 | 2.096 | 32.08 / 38.19 | 7.251 | 27.77 / 45.03 |
| 8K | PF 借用 warp，208/88 | 5.991 | 11.15 / 13.29 | 5.659 | 35.91 / 44.50 |
| 8K | PF 全 CTA WG3 | 4.752 | 13.90 / 16.44 | 5.743 | 35.21 / 43.67 |
| 20K | V3 | 4.931 | 33.60 / 39.36 | 17.739 | 28.66 / 45.59 |
| 20K | PF 借用 warp，208/88 | 13.752 | 12.02 / 14.03 | 13.260 | 38.15 / 45.79 |
| 20K | PF 全 CTA WG3 | 10.591 | 15.77 / 18.62 | 13.523 | 37.42 / 45.01 |

结论：

- PF Dispatch 尚未打满：最忙 rail 也只有约 13–19 GB/s；
  V3 为约 38–39 GB/s。WG3 比借用版的平均 Dispatch 吞吐高 25% / 31%，
  仍未解决与 V3 的明显差距。
- Return 最忙 rail 达到单方向理论 50 GB/s 的约 87%–92%，三者接近。
  不能说整个节点所有 NIC 都打满了；流量不均及阶段尾部仍降低平均吞吐。
- PF 借用版 Return 比 V3 耗时低约 22% / 25%，不是其单条链路更快，
  rail 分布更均衡是一个重要原因。不能把全部收益精确归因于单个变化。
- 时间和带宽各自先按样本计算，再取 p50；不要用两个独立的 p50 相除重算带宽。

顺序为 V3 → 借用 → WG3 → WG3 → 借用 → V3，所有 GPU 测试串行，
正式计时期间无编译。两轮数值一致性良好，没有择取最优一轮。
每种实现首轮验证 3 步 × 2 规模 × 16 rank，共 288 个 rank/sample；
与 identity 参考逐位相等，各实现输出 hash 也完全相同。
六轮的 32 步 token/route/gate trace 逐 rank 相同，所有计时样本的
实际 Dispatch / Return payload 与路由预期相等。
738 次节点时钟观测全部 1620MHz；六次启动均两机 exit0；
结束后两机全部 GPU 0MiB。

[逐 rank 原始阶段时间、payload、workload、hash 与时钟摘要](../benchmarks/communication_phases_20261010.json)。
实验 native 源码提交为 `bbf2bd4`，artifact 在实验 worktree 的
`build/communication_phases/{borrow,wg3,v3}`；后续汇总脚本的修改不改变
这三个 native artifact，两轮分别使用完全相同的 library/adapter hash。

## 结果解释与历史口径

V3 的历史高带宽确实存在：[2026-09-16 记录](communication_spsc_20260916.md)
使用固定 Qwen fixture（H2048/I512/E256/K8），8K Return 为 42.4 GB/s，
64K 为 45.7 GB/s；更早对照为 42.3 / 46.2 GB/s。
它不是本轮 Flash H4096/K6、随机 token 数和带 bias 路由的同负载对照。

本轮需要区分“平均卡有效带宽”和“最忙 rail 带宽”。V3 的 Return 从
expert GPU 直接向 source rank 发送，单卡发送量随专家负载变化。
本次最忙卡/平均卡的 Return payload 比为 1.615（8K）、1.596（20K）。
PF 则先通过 NVLink 将 contribution 写到 source 对应的本地 rail，再跨机
Put；相同比值降为 1.236 / 1.196。因此全局 payload 完全相同，不代表
逐卡 NIC payload 相同。最忙卡接近链路上限时，全局平均仍可能明显较低。
不能将较低的 V3 平均值解读为其 NIC 只能传二十几 GB/s。

PF 当前 Dispatch 保留 per-token Get + 目的端逐行 pull：
每个 pull worker 的 4KiB G2S 完成后执行 S2G，等待完成再复用同一个 staging。
V3 则使用 pack-Put，接收端从本地 RX 做机内 push，并使用多 buffer 的
row-prefetch pipeline。两者还存在 worker 分配与 credit 等待方式的差别。
本实验能确定完整 Dispatch 链路有差距，但不能单独证明“Get 不如 Put”
或“静态分配不如动态队列”；这些变化不是单变量消融。

下一步应优先隔离 Dispatch 的 NIC 提交、RX-consumer credit 和机内转发，
比较多 staging 流水、源端 pack-Put、目的机内 push。
独立 wire-only 测试已显示 Get/Put 可以达到约 45–48 GB/s，
见 [GIN 通信测量](pf_communication_bandwidth_20261009.md)；
它不包含真实机内 fanout/credit，不能替代本轮端到端通信阶段的数字。

这里刻意串行 Dispatch、Return、Reduce，所以两段相加不能作为原融合
kernel 的延迟预测。生产版本允许三者与 GEMM 重叠；上一轮两方向同时运行
的 copy 诊断和本轮阶段串行的 identity 诊断，也不是同一个性能指标。

## 口径

- 双机 EP16，每节点 8×L20X，SM 固定 1620MHz。
- BF16，H4096 / I2048 / E256 / topK6，名义每 rank 8192、20480 tokens。
- 每步重新生成路由，bias 标准差 0.5；tokens 正态波动 10%，seed123。
- 相同逐 rank token/route/gate trace；先 prime32，再 warm8、measure24。
- 计时前校验前三个 priming 调用的输出；其余 priming 不计时。
- GEMM 替换为 identity expert；最终输出应为 X × 有效路由数，不乘 gate。
- 分阶段使用设备 globaltimer；不计固定 buffer、GIN context 创建/释放、
  权重预处理。metadata/planner、阶段边界 rendezvous、最终 reduce 不计入
  Dispatch / Return 的 payload 带宽。
- 每个样本：全局实际跨机 payload / 16 / 最慢 rank 阶段时长，再取 p50。
  单位 GB/s，单卡单方向；不把 TX+RX 加起来后比较单方向链路上限。
- 同时报告最忙 rail 的 payload / 同一个最慢 rank 时长。它使用整个阶段的
  时间，不是瞬时峰值，也不是取一个更短的有利发送区间。
  随机负载下最忙卡不等于平均卡，判断 NIC 是否接近上限必须区分二者。
- 每步核对实际 payload 与路由推导值。Dispatch 按目的节点去重；
  Return 按远端 expert contribution 数，不去重。
- 只有两个节点，未据此声称任意多节点的扩展效率。

## 诊断与融合 kernel 的区别

阶段顺序是 planner → Dispatch → 全 rank rendezvous → Return → 全 rank
rendezvous → Reduce。保留生产通信 worker、warp 划分、寄存器配额、
shared memory 预留、RX credit 和本地 NIC completion。

为让 Dispatch 在 Return 完全不启动时结束，PF 将完成的接收行保存在
完整 invocation-sized canonical X 输出中，而不是循环覆盖计算 X ring。
计算槽控制计数仍按原 admission 窗口推进；Dispatch 结束后恢复其计算、
Return 计数基线，只保留本次已经推进的 X-ready 计数。
网络 RX/TX ring 的大小、fanout 消费和复用协议均不变。
完整接收输出是独立通信 API 的必要输出，不是生产 PF 的新增长生命周期缓存。
本轮 8K/20K 的完整输出均能容纳在原先预留的 canonical arena 中：
借用版 compute / GIN / notify 分配字节数与上一轮 copy88 完全相同。
这里改变的是输出地址映射与存活期，不是通过扩大网络 ring 容量获取带宽。

Return 的输入全部 ready，PF 用静态 FC2 N-tile 顺序执行 uint4 拷贝，
写入原有 TX packet / 本地 contribution 地址，然后运行原有 Put 和
progress worker。这个读取已完成输出、写入 TX 的搬运计入 Return；
真实 PF 中对应写入由 FC2 epilogue 完成，所以这里的 Return 时间
不能直接当成融合 kernel 中不可折叠的通信开销。

V3 复用已有 transport-only 诊断：完整 canonical 接收行直接充当 FC2 输出，
保留生产 pack-Put、机内转发和 Return pack。补齐过时模板接口、
原生产寄存器配额及全 rank 阶段边界；Reduce 延后到 Return 全部完成。
边界只由 CTA0 和本地控制计数、GIN rail barrier 完成，不引入 V3 的
全 grid 静态工作等待。

V3 采用 recovery/v3-shared-queue 的 EP16 Flash 调优配置。
该分支与本实验基底的 persistent_v3、unified transport/kernel、
initialize_pipeline 的源码比较，只有 FANOUT_LANES=1 的缺省宏差异；
恢复配置已经显式定义该值，不改变 V3 算法。

## 候选

| 候选 | CTA 线程 | Math / Control 配额 | Get / Put / Progress / Admit |
|---|---:|---|---|
| PF 借用 warp | 384 | 208 / 88 | 64 / 48 / 19 / 1 |
| PF 全 CTA WG3 | 512 | 192 / Local64 / Network64 | 132 / 132 / 132 / 1 |
| V3 | 384 | 216 / 64 | 原 16 sender CTA + 116 receiver/local CTA |

PF 使用 token Get Dispatch、packet Put Return，132 lanes、D32/R128、
pending4、B4096、credit windows2、4 slots/channel、publication plan、
Return-ready 去重。Dispatch 不做 route pack；Return 不额外运行独立 pack kernel。
本分阶段诊断仅覆盖这个 Get 配置，不覆盖实验性的 Dispatch push 配置。

## 复现

实验分支：experiment/communication-phases-20261010。

```bash
/opt/.venv/bin/python benchmarks/build_communication_phases.py
/opt/.venv/bin/python benchmarks/run_pf_network.py \
  --artifact build/communication_phases/borrow \
  --label borrow_r1 --output build/communication_phases/runs \
  --master-port 32010 \
  --env WEFTMOE_PF_MIN_RING_DEPTH=2 \
  --env WEFTMOE_PF_CREDIT_WINDOWS=2 \
  --env WEFTMOE_PF_WG3_PUBLICATION_PLAN=1 \
  --env WEFTMOE_PF_DEDUPLICATE_RETURN_READY=1 -- \
  --backend pf --hidden 4096 --intermediate 2048 --experts 256 --top-k 6 \
  --tokens 8192 20480 --ring-tokens 4096 --prepared-weights \
  --profile-device --communication-phases --verify-steps 3 --verify-kernel
```

WG3 换 artifact 和 label；V3 换 artifact、label、backend=v3，并省略 PF 环境变量。
summary 脚本读取 node0 的逐 rank 原始结果并核对每步 payload。
每次启动的 manifest 记录 library/adapter hash、源码版本、环境与命令，
clocks 文件记录两节点的实际频率。GPU 测试串行，正式计时期间没有编译任务。
