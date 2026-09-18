# 本地 dX 批量领取验证（2026-09-18）

仅本地 unified BWD 一次领取同一行块的最多两个相邻 dX tile，GIN 和 standalone 保持单 tile。每个 tile 保留原 GEMM、K 累加顺序与写回；相邻 tile 之间保留 cluster 同步，保护 TMA multicast 与共享写回区域。最后一组按实际领取数完成计数，不按 batch 大小分支。

## 具体实现

dX 是输入梯度；本实现把同一 M block（一组输入行）的 dX 按输出列分成多个 N tile，由常驻计算 cluster 从队列领取。旧实现每领取一个 tile，都要经过任务查找、原子领取、任务信息同步、GEMM 写回及完成计数。本轮把本地路径的一次领取扩大为最多两个相邻 tile，让两次原有计算共用一次外层调度过程。

### 1. 调度器一次原子领取连续的可用 tile

在 [scheduler.cuh](../src/cuda/persistent_v3/backward/scheduler.cuh) 的 `TileDomain` 中增加 `claim_granularity`，默认值为 1。`try_claim_owner_tile` 从当前 64 位 claim word 找到第一个未领取的有效 bit，然后向后收集最多 `claim_granularity` 个连续可用 bit，用一次 CAS 将整个 mask 标记为已领取。CAS 失败时根据新的占用状态重试。

领取范围受三条边界限制：必须属于同一个 owner、不能跨 64 位 claim word、不能跨过已被其他 worker 领取的 tile。最后一个 word 的有效 mask 还限制了真实 tile 数。返回任务携带起始 `tile` 和实际 `tile_count`，而不是假定每次一定领到两个。例如 owner 有三个 tile 时，无竞争下两次领取分别为 `[0,1]` 和 `[2]`；有竞争或接近 word 边界时，也允许只领取一个。

这避免两个 worker 获得重叠任务，同时允许尾组自然缩小。DAB 仍使用默认的单 tile 粒度，其分阶段 ready/完成计数协议保留。

### 2. 计算端执行原 GEMM，并保留组内共享内存同步

在 [unified_compute.cuh](../src/cuda/persistent_v3/backward/unified_compute.cuh) 中，`run_compute` 新增模板参数 `DxClaimTiles = 1U`，并将其写入 dX 队列的领取粒度。双 tile 版本读取任务的实际数量，按 `tile + offset` 依次调用原来的 `run_k_mn_tile_tma_store`；每个 tile 的 K 累加、输出列范围和 TMA store 方式不变。

两个 tile 之间仍执行 `cluster_sync()`。原因是 CTA 0 的 TMA multicast 会写入同伴 CTA 的共享区域，单个 CTA 完成自己的 store，并不能保证同伴也已结束；如果直接装入下一 tile，可能覆盖同伴仍在写回的数据。保留这道同步后，优化减少的是重复领取和外层任务交接，并未取消必要的数据依赖。

一组 tile 全部写回后，继续执行原有 fence 和 cluster 同步；随后 owner 完成计数及全局完成计数均加上实际 `dx_count`。只有 owner 的累计完成数达到 `kDxTiles`，才发布该行块的 dX ready 状态及适用路径上的 Return 任务。尾组加 1，不会错误地按 2 宣告完成。

### 3. 按通信路径编译期选择，不按 batch 大小选择

在 [unified_kernel.cuh](../src/cuda/persistent_v3/backward/unified_kernel.cuh) 的两处 unified 调用点传入：

```cpp
run_compute<DabTileN, DxTileN, DwTileN, UseGin ? 1U : 2U>(...);
```

因此本地 unified 路径启用最多两个 tile，跨节点 GIN 路径仍为一个；standalone 调用保留模板默认值 1。单 tile 版本的 `dx_count` 也写成编译期常量 1，使编译器能够消去双 tile 逻辑。这个选择取决于通信路径，不是对 8K/64K batch 或模型名称进行特判。

先前对所有路径统一启用双 tile 的实验，在 Qwen EP16 两个配置中出现约 0.44% 和 0.12% 回退，因此最终收窄到本地路径；详见[前序领取实验](bwd_dx_claim_20260918.md)。这种改动改变任务分配粒度，收益需要完整算子对照确认，不能仅因原子领取次数减少就假定所有通信路径都更快。

### 4. 为并发领取和尾组补充验证

[tests/cuda/backward_scheduler_claims.cu](../tests/cuda/backward_scheduler_claims.cu) 使用 66 个 CTA 并发消费任务，覆盖领取粒度 1/2/4/64、1/2/3/63/64/65/127/128/129/257 个 tile，以及 owner ready word 和 summary 边界，共 120 个案例。检查每个有效 tile 恰好领取一次、无越界领取，并确认 ready/summary 最终清空。较大的粒度用于测试调度器通用边界；生产 dX 模板只允许 1 或 2。

完整算子另外用 H1536/I1024 产生三个 dX cluster tile，覆盖实际 GEMM 的 2+1 尾组，并检查参考张量、Graph 和单 GPU memcheck。两种验证分别针对调度器并发计数与真实计算/写回路径，不能相互替代。

本轮还修正 [本地测试启动器](../tests/run_local_activation_p2p.py) 的 Graph 结果解析：读取实际输出的 `inference_graph_replays`，要求每个 rank 恰好出现一次、6 次 inference replay、`retained_capacity=1`、`skew=1` 和 `gradient_bits=all`。[对应测试](../tests/test_buffer_mode_validation.py) 检查有效记录以及缺失、重复 rank 和旧字段等失败情况，避免“命令结束”被误当成生命周期校验通过。

## 测量结果与验证

两模型的 GIN 反向主核及四个前向主核均与已采用倒数修正的生产基线逐 128 位机器码一致。本地反向机器码有所变化，故重新测量。EP16 完成正确性检查，本报告不声称新的跨节点性能收益。

数值与生命周期：两模型 × EP8/EP16 的 480 份输入、检查点及梯度参考文件逐字节一致；两模型 × EP8/EP16 × 8K/64K 的八组完整梯度各重复三次，dX/dgate/dW1/dW2 哈希与冻结基线一致。两模型的 EP8 和 EP16 均通过 Graph、偏斜/空专家、跨 stream、资源扩容和旧反向状态复用检查。另以 H1536/I1024 的三个 dX cluster tile 覆盖 2+1 尾组：EP8 的 80 份文件与同配置生产基线逐字节一致，EP8 Graph 检查通过；单 GPU、129 tokens、三次完整梯度重复的 Compute Sanitizer memcheck 为零设备内存错误。此 sanitizer 检查关闭 NCCL 架构探测产生的 API 错误报告，不是所有 API 或所有多 GPU 交错的无错证明。

计时仅覆盖改变的 EP8 本地内核。两模型 × 8K/64K × 基线/候选 × 双顺序，每次 10 次预热、60 次计时，共 960 次。计时包含 tensor 申请及输入克隆，准备 Forward 与 rank barrier 在计时外，正式计时关闭重复检查。Flash 64K 每次在计时外暂停 500 ms，因此是间隔调用延迟，不代表持续吞吐。正式计时期间没有运行构建、审计或正确性 GPU 作业。

| 模型 / tokens | wall P50 基线→候选 ms | wall 改善 | GPU 改善 | 两轮 wall 改善 |
|---|---:|---:|---:|---:|
| qwen / 8192 | 3.9869 → 3.8756 | +2.792% | +2.907% | +2.918% / +2.502% |
| qwen / 65536 | 30.4778 → 29.8827 | +1.953% | +1.979% | +1.857% / +2.020% |
| flash / 8192 | 10.7267 → 10.6595 | +0.626% | +0.639% | +0.606% / +0.652% |
| flash / 65536 | 80.5908 → 80.1080 | +0.599% | +0.587% | +0.523% / +0.650% |

主表保留所有样本。频率条件补充对照只使用每个 rank 的调用窗口都有 NVML 样本、全部读数为 1620 MHz 且无非 idle 限频标志的 iteration，资格条件与耗时无关。有限采样不证明每周期恒频。

| 模型 / tokens | 合格样本数 基线/候选 | 合格 wall 改善 | 合格 GPU 改善 |
|---|---:|---:|---:|
| qwen / 8192 | 114 / 120 | +2.792% | +2.907% |
| qwen / 65536 | 120 / 120 | +1.953% | +1.979% |
| flash / 8192 | 114 / 119 | +0.613% | +0.616% |
| flash / 65536 | 110 / 112 | +0.601% | +0.608% |

全样本尾部：

| 模型 / tokens | wall P95 前→后 ms | P99 前→后 ms | 最大值前→后 ms |
|---|---:|---:|---:|
| qwen / 8192 | 4.037 → 3.912 | 4.050 → 3.967 | 4.052 → 4.477 |
| qwen / 65536 | 30.807 → 30.165 | 35.277 → 30.315 | 36.771 → 30.327 |
| flash / 8192 | 10.795 → 10.716 | 10.819 → 10.762 | 10.864 → 11.347 |
| flash / 65536 | 80.887 → 80.498 | 81.129 → 80.724 | 81.330 → 80.911 |

SM 设置 1620 MHz，NVML 目标间隔约 2 ms；7680 个 rank 窗口中 104 个没有采样，已覆盖窗口中 16 个频率非 1620 MHz、3 个含非 idle 限频位，两者可能重叠。采集的 NIC 错误/拥塞和 CPU 限流非零增量 0 项。原始样本和完整计数在 JSON 中。

验证过程中修正两项测试问题并保留失败记录：初次 EP8 金标准对照误用了 E32，而旧参考是 E16；按元数据改为 E=2×EP，生产基线控制及候选对照均通过。本地 Graph 启动器误查旧字段 graph_replays，现检查 inference_graph_replays=6，并要求完整 rank、retained_capacity=1、skew=1、gradient_bits=all；新增三项验证测试通过，另有 18 项启动器测试通过。未把这两次测试配置/解析失败当成候选数值通过的证据。

状态：已采用经过验证的本地 dX 批量领取版本。按用户要求，本轮测试与报告结束后停止继续优化，不声称达到理论硬件上限。

[全部样本、构建、频率覆盖、验证与采用记录](bwd_dx_local_20260918.json)。原始记录位于 `results/bwd_dx_local_20260918/`，候选构建及源码位于 `results/bwd_dx_claim_20260918/local_only/` 与 `source_local_only/`。

收尾验证：生产 src 与测试候选快照逐字节一致；已将并发领取回归测试纳入 tests/cuda/backward_scheduler_claims.cu，并在生产源码上通过 120 个案例。两节点 GPU 测试作业均已结束，SM 锁频已重置。
