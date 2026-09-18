**完成等待与稳定排序优化，2026-09-17，1620 MHz**

关闭重复校验后的 960 个正式计时样本中，八种配置的 wall 中位数均改善，幅度约 0.28%–7.46%；batch/EP 逐字节一致性、graph、Forward/Backward 参考及故障传播检查全部通过。独立 200 样本对照中，Flash EP8/64K 从 mapped + sleep 改为持续轮询后 P99 为 81.406 → 56.301 ms。

## 具体实现

本轮包含两部分：GPU 上整理路由记录的稳定 radix 排序，以及 CPU 等待 Forward 完成的通知路径。前者减少排序中的重复查找，后者缩短 GPU 已完成但调用尚未返回的等待；两者都不修改浮点计算、GEMM 累加或 routed contribution 的归约顺序。

### 1. 用区间起点直接计算 histogram 和稳定写回位置

实现位于 [notify/radix.cuh](../src/cuda/notify/radix.cuh) 的 `NotifySortStorage` 和 `notify_radix_sort`。原算法把记录分成 1024 项的 chunk，每轮处理最多 8 位 radix digit；chunk 内通过 CUB 稳定排序，再按 bucket 和源 chunk 次序分配全局位置。这一排序框架保留，本轮替换其中的两处查找：

- 旧 histogram 路径由每个相同 digit 区间的首项向后扫描，计算区间长度。现在每项并行比较前一个 digit，由首项写入共享内存 `starts[digit]`；经过 block 同步后，末项直接写入 `within + 1 - starts[digit]`，得到该 chunk 的 bucket 计数。
- 旧 scatter 路径中，每项二分查找自己所在 digit 区间的起点。现在同样先并行标记起点，再直接读取 `starts[digit]`，计算区间内偏移。

写回地址可表示为：

```text
dst = bucket_base[digit]
    + 当前 bucket 在更早源 chunk 中的记录数
    + within - starts[digit]
```

例如 chunk 内 digit 为 `[1, 1, 3, 3, 3]`，只需记录 `starts[1]=0`、`starts[3]=2`；digit 3 的长度由最后一项计算为 `4+1-2=3`，三项的区间内偏移分别为 0、1、2。全局地址由固定源顺序决定，不依赖 CTA 谁先执行，因此相同 key 的相对次序保持稳定。尾 chunk 只统计和写回有效项；scatter 的无效项使用 `bins` 哨兵，不访问 `starts[bins]`。

共享临时区将原来的 1024 个 64 位 digit 改为 1024 个 `unsigned` digit 和最多 256 个区间起点，仍与 CUB 临时存储共用 union。单 chunk 快路、按实际 key 位宽减少 radix 轮次、bucket 前缀扫描是已有逻辑，不计作本轮新增优化；union 总大小还取决于 CUB 临时存储，不能单凭 digit 数组变小声称 occupancy 提升。

### 2. 用 stream 有序写入代替常规主机回调通知

实现位于 [activation_transport.cu](../src/cuda/activation_transport.cu) 的资源池创建和 `LocalCudaP2pForwardMegaKernel::enqueue` 中的完成处理；指针保存在 [forward/kernel_support.cuh](../src/cuda/forward/kernel_support.cuh) 的 `PoolSlot`。

每个 slot 的 pinned Forward status 分配额外增加一个 4 字节完成标记，并加上 `cudaHostAllocMapped`。`cudaHostGetDevicePointer` 获取 GPU 可写地址，CPU 与 GPU 因而能访问同一个完成标记，无须新增通知 kernel。slot lease 一直持有到完成处理结束，避免另一次调用提前复用这个标记。

完成处理按以下顺序执行：

1. GPU stream 中已经排入结果就绪事件、status D2H 复制和 completion events。
2. CPU 用 release store 将标记清零，再在同一个 stream 上提交 `cuStreamWriteValue32(..., 1, CU_STREAM_WRITE_VALUE_DEFAULT)`。
3. CPU 用 acquire load 等待标记变成 1，然后才进入后续 event query 和状态检查。stream 顺序与默认 fence 保证通知排在前面的状态复制及事件之后。

若 stream write 返回 `CUDA_ERROR_NOT_SUPPORTED`，才使用原有 `cudaLaunchHostFunc` 和 atomic bool 通知；其他错误直接报告 CUDA failure，不静默降级。该回退在当前平台未触发，不能视作已实测覆盖。

### 3. 同步等待采用 CPU pause/yield

旧循环每次 `sleep_for(50µs)`，实际唤醒可能因操作系统调度延后数十毫秒。现在 x86 使用 `pause`、AArch64 使用处理器 `yield` 指令，其余平台使用编译器 atomic fence；这里的 `yield` 不是主动调用操作系统线程让出接口。线程持续轮询，同时保留取消和 deadline 检查。

`enqueue()` 的异步接口保持原样；占用 CPU 的是调用者同步等待结果的阶段。每个同步等待占用一个 host 线程，这是减少唤醒延迟的代价，目标平台每节点有 192 个逻辑 CPU。resident GIN 活跃时仍不循环调用 CUDA event-query API，以保留既有驱动锁规避逻辑；跨 stream event 依赖和 terminal agreement 也保留。

本报告主矩阵衡量上述 radix、mapped 通知和持续轮询的合并收益；后面的两组独立对照分别衡量通知方式、轮询方式。不能把主表全部收益单独归给其中一项。

## 测量结果与验证

测量包含当次 tensor 申请、输入克隆、Notify 和保留反向状态的 Forward。每次先做不计时的 rank barrier，再对每次 wall 取各 rank 最大值。两节点各八张 GPU 锁定 1620 MHz，测前测后记录频率，结束解除锁频。编译、correctness 和 profiler 均与正式计时分开。CHECK_REPEAT 环境变量彻底移除，并断言每个 rank 的实际校验次数为零。P50 为中位数，P95/P99 为 nearest-rank 分位数；保留全部离群值。

正式矩阵比较提交 `857e736fcb5a6c807f662ac7a9da54a38b29e58d` 与当前候选（稳定 radix 区间优化 + mapped 完成通知 + CPU 轮询），两者使用相同 tuning headers。每 case 两轮，各 10 次预热、30 次测量，第二轮反转 case 和实现顺序，共 60 个正式样本/实现。

| 模型 / EP / 每 rank tokens | wall P50 前 → 后 (ms) | wall 均值前 → 后 (ms) | wall P95 前 → 后 (ms) | Notify P50 前 → 后 (ms) |
|---|---:|---:|---:|---:|
| flash / 8 / 8192 | 7.811 → 7.619 | 7.830 → 7.637 | 7.998 → 7.753 | 0.522 → 0.450 |
| flash / 8 / 65536 | 56.111 → 55.956 | 56.525 → 55.960 | 56.420 → 56.187 | 1.070 → 1.031 |
| flash / 16 / 8192 | 10.833 → 10.398 | 10.856 → 10.429 | 11.278 → 10.665 | 0.868 → 0.591 |
| flash / 16 / 65536 | 72.658 → 71.908 | 72.903 → 71.643 | 73.635 → 73.249 | 1.602 → 1.240 |
| qwen / 8 / 8192 | 3.262 → 3.100 | 3.277 → 3.115 | 3.438 → 3.133 | 0.499 → 0.465 |
| qwen / 8 / 65536 | 20.858 → 20.720 | 20.866 → 20.726 | 20.950 → 20.823 | 1.040 → 1.011 |
| qwen / 16 / 8192 | 5.853 → 5.416 | 6.497 → 5.422 | 6.028 → 5.547 | 0.894 → 0.577 |
| qwen / 16 / 65536 | 36.564 → 36.049 | 37.254 → 36.062 | 36.801 → 36.288 | 1.714 → 1.266 |

Flash EP16/64K 的 GPU P50 第一轮为 71.089 → 71.137 ms，第二轮为 69.978 → 69.529 ms；跨轮 GPU 波动明显，不能将该 case 的全部 wall 收益归因于主机等待。每轮 wall 的均值和中位数以及全部 GPU 样本均保存在机器记录中。

为区分稳定排序和主机等待的作用，还做了两组独立对照。两组均使用优化后的 radix，每 case 两轮各 100 个测量样本，合并 200 个样本/实现，反转执行顺序。

回调 → mapped 标记（保留 50µs sleep）：

| 模型 / EP / tokens | wall P50 前 → 后 (ms) | wall P95 前 → 后 (ms) | wall P99 前 → 后 (ms) | wall 最大值前 → 后 (ms) |
|---|---:|---:|---:|---:|
| flash / 8 / 65536 | 56.109 → 56.010 | 65.299 → 56.202 | 86.345 → 56.352 | 130.407 → 97.098 |
| qwen / 8 / 8192 | 3.218 → 3.170 | 3.290 → 3.223 | 3.431 → 3.244 | 3.716 → 3.448 |
| qwen / 16 / 8192 | 5.604 → 5.536 | 5.744 → 5.737 | 5.882 → 5.911 | 6.104 → 6.315 |

mapped 标记 + sleep → mapped 标记 + pause/yield：

| 模型 / EP / tokens | wall P50 前 → 后 (ms) | wall P95 前 → 后 (ms) | wall P99 前 → 后 (ms) | wall 最大值前 → 后 (ms) |
|---|---:|---:|---:|---:|
| flash / 8 / 65536 | 56.006 → 55.940 | 56.332 → 56.180 | 81.406 → 56.301 | 83.927 → 56.438 |
| qwen / 8 / 8192 | 3.170 → 3.085 | 3.232 → 3.145 | 3.301 → 3.250 | 3.823 → 3.650 |
| qwen / 16 / 8192 | 5.537 → 5.419 | 5.706 → 5.585 | 5.906 → 5.750 | 6.159 → 6.257 |

定位证据来自独立的全 rank Nsight Systems 采样，不能作为未插桩性能样本。回调版本曾出现 status D2H 结束到 host completion query 间 18.8/29.4/39.2 ms 的延迟。mapped + sleep 的干净 trace 每 rank 的 Forward 主线程记录 210 次 StreamWriteValue、零次 LaunchHostFunc；其中一个 90.881 ms wall 样本的 rank 5 主线程 nanosleep 实际耗时 35.216 ms，远大于请求的 50µs，status copy 到 host query 间隔 34.843 ms。这是去掉 sleep 的直接依据。不同进程的 trace 时间没有跨 rank 相减。

最终 pause/yield 候选也做了独立 200 次测量的全 rank trace。8 个 rank 的 Forward 主线程均记录 210 次 StreamWriteValue、零次 LaunchHostFunc；去掉 10 次预热后，1600 个 rank 样本的 status copy 结束至 host query 开始间隔中位数为 10.124 µs，最大 37.263 µs。该插桩运行 wall 最大值为 56.811 ms，没有 60 ms 以上样本；这些数据只用于验证路径，不计入正式矩阵。原始分析见 `results/completion_latency_20260917/trace_spin_flash_ep8_64k.analysis.json`。

完成标记使用 `CU_STREAM_WRITE_VALUE_DEFAULT` 的默认 fence，CPU acquire 读取 mapped word 后才调用 CUDA event query；不在 resident GIN 活跃期间轮询 CUDA query，保留已有驱动锁规避逻辑。[NVIDIA Stream Memory Operations 文档](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html)说明了 stream write 的内存约束。独立 probe 在当前设备切换后也完成 1000 个 generation，并验证观察标记时先前 D2H 数据及 event 均已完成。NOT_SUPPORTED 回退分支未在本平台触发。

验证与限制：

- EP1/EP2 元数据 oracle、资源池复用、非法输入恢复、取消/deadline 和 API allocation audit 通过。
- Qwen/Flash 的 EP8×258 与 EP16×129 重分片后输出及输入/route/gate 逐字节一致；共同 129-token 前缀在 EP8 的 129/258/8192/65536 batch 中一致，EP16 的 8K/64K 前缀一致。
- 两模型 × EP8/EP16 × 8K/64K 每 rank 20 次全输出重复校验通过；输入、输出、路由、gate、权重哈希与冻结基线一致。这些校验独立于正式性能计时。
- 两模型动态路由 graph 重放、跨 stream 消费通过；每模型 160 个 Forward/Backward 文件与既有参考逐字节一致，包括 dW。
- EP8/EP16 注入 rank 0 Forward status 错误均通过 production-terminal-agreement 报错，约 9.1/32.6 秒结束，没有测试超时。
- 稳定 radix 的 48 组 GPU 回归和 sanitizer 结果见[排序报告](notify_radix_20260917.md)。该报告的旧时间样本实际上开启了计时外的重复校验，已降级为诊断；此处正式矩阵是关闭校验后的重新测量。

这仍不是实时延迟保证，也没有达到硬件上限。持续轮询不能避免操作系统抢占，CPU 资源紧张时需要重新评估；样本数也不足以断言罕见长尾消失。大 batch 仍受 GPU 关键路径影响，其中保留训练 X 的独立 D2D 拷贝是下一项候选优化。

[完整机器记录](completion_latency_20260917.json)包含每轮原始样本、构建信息、二进制与源码 SHA256、两轮分布和频率快照。原始日志和复现实验脚本在 `results/completion_latency_20260917/`：`measure.py {focused,spin,formal}`、`correctness_spin.py`、`repeat_spin.py`、`fault_spin.py`、`trace_mapped.py` 和 `analyze_fast.py`。独立设备 probe 位于 [tools/stream_completion_probe.cu](../tools/stream_completion_probe.cu)。benchmark 入口现在校验新记录和缓存记录的实际 repeat counters，拒绝计时/校验模式混用。
