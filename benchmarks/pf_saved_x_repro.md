# PF Saved-X：vector / TMA 最小复现

这是独立 CUDA 诊断，不改变正式 PF。问题是：为什么在 FC1 已加载的 A-stage
上保存 X，vector 比 C7/C8 的 TMA 集成更快？不是测通用 memcpy 峰值。

## 运行

在仓库根目录，依赖 CUDA 和 CUTLASS headers，不依赖 Python、Weft runtime、
NCCL 或多卡 launcher。以下路径对应本次环境：

```bash
mkdir -p build/pf_saved_x_repro_20261008
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
  -Isrc -I/opt/cutlass/include benchmarks/pf_saved_x_repro.cu \
  -lcuda -o build/pf_saved_x_repro_20261008/repro

# 参数：copy|gemm routed_rows lane0|elect tma_rows fence samples device [mode]
build/pf_saved_x_repro_20261008/repro copy 49152 lane0 32 1 40 0
build/pf_saved_x_repro_20261008/repro gemm 49152 lane0 32 1 40 0
build/pf_saved_x_repro_20261008/repro gemm 122880 lane0 32 1 40 0

# 去掉发起处 peeling loop / 两次 M32 合并一次 M64 / 去掉 source fence
build/pf_saved_x_repro_20261008/repro gemm 49152 elect 32 1 40 0
build/pf_saved_x_repro_20261008/repro gemm 49152 elect 64 1 40 0
build/pf_saved_x_repro_20261008/repro gemm 49152 elect 64 0 40 0
```

使用支持 SM90a、cluster2、足够 shared memory 的 GPU。复测前自行确认
设备空闲、时钟固定为 1620 MHz；程序不会修改整机设置。

## 保留了什么、去掉了什么

- 保留 M128/N256/K64、3-stage、cluster2 A multicast、384 threads、
  256 math threads、226304 B dynamic shared。H4096，32 个本地专家，
  每个专家 FC1 输出 4096 列，即 16 个 N tile。
- 每个 M-block 只有一个相邻 N-tile pair 保存 X，owner 在 8 对之间轮转；
  偶数 CTA 保存前 M64，奇数保存后 M64。每个 owner 每个 K64 写 8 KiB。
- `gemm` 保留真正的 BF16 WGMMA，saved-X 复制位于 WGMMA commit 与 wait
  之间；`copy` 去掉 WGMMA/B load，只保留 TMA load、A-stage 复用和保存。
- 去掉 route planner、远端 Pull、FC2、GU/S epilogue、return/reduce；
  GEMM 结果直接 vector 写 C。所有行 canonical 连续、无尾块；没有 C5 的
  round-robin → canonical 映射。这消除了 C5/C7 行序不同这个混杂因素。
- 49152 / 122880 是**每卡 routed rows**，对应 top-k6、均衡 EP8 下
  nominal 8192 / 20480 source tokens/rank 的计算量。这里不模拟随机路由；
  原完整 PF 性能测试仍是随机变长/热点负载。不能把这里的 ms 当作完整 FWD。

每个 `copy|gemm` × `lane0|elect` 特化内部，各复制模式使用**同一个 kernel
binary**，由 uniform runtime 参数切换；寄存器预算、shared 和其他代码一致。
不同特化之间仍可能有代码生成差异，不能只比较跨特化的绝对耗时。

| mode | A-stage 释放条件 | 目标写出完成条件 |
|---|---|---|
| `none` | WGMMA 读完（copy 中无 GEMM） | 不保存 X |
| `vector` | 每个 math thread 两次 16 B LDS/STG，再等 WGMMA | 常规 global store；kernel 完成后读结果 |
| `tma_full` | 每 K `bulk.wait_group 0`、warp join，再等 WGMMA | 每 K 完整等待 |
| `tma_read` | 每 K `bulk.wait_group.read 0`、warp join，再等 WGMMA | M-task 末尾 full wait |
| `tma_pipelined` | 保留当前 stage，下一 K `read 1` 后释放前一个 | M-task 末尾 full wait，并释放最后一个 stage |

`tma_pipelined` 仅允许 `lane0`：跨 K 的 group 数等待必须使用稳定 issuer。
`elect` 的 `tma_read` 在任务末尾让 warp0 的所有线程 drain 各自 group，
不假设每次 election 都选同一 lane。默认测所有可用 mode，也可指定单 mode
供 profiler 使用。没有新增运行时 PF 开关。

分配、初始化和校验都在计时外；8 次 warmup，然后 CUDA events 单 kernel
计时，mode 顺序逐轮正反交替。两次测量间隔 2 ms，**不计入 CUDA events**，
降低连续纯 GEMM 压测触发功率上限的影响；JSON 的 `pause_us` 明确记录它。
输出 JSON 行含全部样本和 median。
每种模式检查 saved-X bitwise 等于输入、C bitwise 等于无缓存 GEMM；另外
用 CPU 独立计算 32 个 dot products 检查无缓存 GEMM 的数值与元素布局。

## 2026-10-08 实测

NVIDIA L20X，GPU0，SM 时钟设定 1620 MHz、memory 3201 MHz；两轮配置正反序，
每轮每模式 8 warmup + 40 samples，以下是两轮 median 的均值，单位 ms。
这不是端到端 PF，也不是随机 route benchmark。

| 工作量 | 无 saved-X | vector | TMA full | TMA read/full 拆分 | TMA full 相对 vector |
|---|---:|---:|---:|---:|---:|
| copy，49152 routed rows | 0.135216 | 0.245656 | 0.269320 | 0.269896 | +9.63% |
| copy，122880 routed rows | 0.318448 | 0.602504 | 0.661352 | 0.662104 | +9.77% |
| FC1 GEMM，49152 routed rows | 2.480360 | 2.510408 | 2.544712 | 2.547568 | +1.37% |
| FC1 GEMM，122880 routed rows | 6.090336 | 6.152200 | 6.281896 | 6.289552 | +2.11% |

复制和计算可以 overlap，加入 GEMM 后相对差距缩小；但不能把两种 workload
的总时间直接相减当成 saved-X 独立延迟，也不能把本表外推为整个 PF 的收益。

单因素/逐步消融没有推翻 vector 的优势。为避免跨特化代码生成/运行间
波动混淆，下表报告各配置内部 TMA full 相对其 vector 的耗时增量：

| 发起/粒度/fence | copy 8K 等效 | copy 20K 等效 | GEMM 8K 等效 | GEMM 20K 等效 |
|---|---:|---:|---:|---:|
| lane0 / M32 / 有 | +9.63% | +9.77% | +1.37% | +2.11% |
| elect / M32 / 有 | +10.02% | +10.05% | +1.27% | +1.85% |
| elect / M64 / 有 | +10.16% | +10.12% | +1.17% | +1.88% |
| elect / M64 / 无 | +10.04% | +10.02% | +1.24% | +1.97% |
| lane0 / M32 / 无 | +9.53% | +9.50% | +1.19% | +2.17% |

SASS 确认原 C7 和样例 lane0 路径均有 TMA 参数统一化/peeling loop；样例
elect 路径消除了 store 周围的该循环。但时延差距仍在，**它不是主要解释**。
两次 M32（各 4 KiB）合成一次 M64（8 KiB）和删除 source fence 也没有
稳定、足以改变结论的收益，不能将“4 KiB 太小”或 fence 单独定为根因。

额外的 `tma_pipelined` 使两个 bulk group 在途，延后一个 K 才归还前一
A-stage credit：copy 为 **0.350792 / 0.859272 ms**，GEMM 为
**2.651360 / 6.491216 ms**，比逐 K full wait 更慢。它同时延长 A-stage
占用、削减了 load producer 的可用超前空间；不是无代价地增加 store overlap。

### 硬件计数器

同一最终 binary，49152 rows，application replay、cache/clock control 均为
none，warmup 后定向采集。以下是独立 profiler 运行，**不混入上面的 event
计时表**。GMMA wait 是 `smsp__warps_issue_stalled_gmma.sum / warps_active.sum`，
不是 kernel 总时长的百分比；Tensor active 也是 profiler 的周期指标。

| GEMM mode | Tensor active | GMMA arrive wait | DRAM read GB | DRAM write GB |
|---|---:|---:|---:|---:|
| vector | 83.97% | 0.77% | 2.953 | 0.890 |
| TMA full | 82.46% | 1.24% | 3.056 | 0.890 |
| TMA read/full | 82.57% | 1.27% | 3.189 | 0.891 |
| TMA pipelined | 79.05% | 0.58% | 4.050 | 0.887 |

copy-only 的 vector/TMA full 都约读 0.403 GB、写 0.402 GB，字节数相同，
TMA 仍慢。这不是 vector 少保存了数据。GEMM 下 TMA 使跨 warp 的到达等待
增加、Tensor active 降低；固定输入下 DRAM 读流量也变了，表明调度/L2
复用行为受到影响。不能把全部差距精确分摊到一条 wait 或单一 cache 因素。
延后 A credit 的实验虽然降低 GMMA arrive wait，却同时大幅增加 DRAM
读流量、降低 Tensor active，再次说明不能只优化一个等待计数。

示例采集命令（其他 mode 换最后一个参数）：

```bash
/usr/local/cuda/bin/ncu --replay-mode application \
  --kernel-name regex:saved_x_repro --launch-skip 6 --launch-count 1 \
  --clock-control none --cache-control none \
  --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.sum,smsp__warps_issue_stalled_gmma.sum,dram__bytes_read.sum,dram__bytes_write.sum \
  build/pf_saved_x_repro_20261008/repro gemm 49152 lane0 32 1 1 0 vector
```

### 时钟与验证限制

初始连续 GEMM 压测出现接近 700 W 上限及 1500–1605 MHz 的 NVML 样本，
因此不采用无间隔的 `final_r*.log` 作最终表。改成计时外 2 ms 间隔后，
100 ms 采样未见功率限制触发，观测 SM 为 1620 MHz。但这**不证明 kernel
内部全程恒频**：NCU 的 copy 有效频率约 1614–1619 MHz，GEMM vector /
TMA full 为 1440.5 / 1443.5 MHz，其他 GEMM 方案约 1446–1454 MHz。
因此本表是 **1620 MHz 锁频设置下的实测**，不是严格恒定有效频率验收。
vector 的优势并不是它测量时频率更高造成的；也不按设定频率反推峰值 FLOPS。
NCU 以观测周期数/时间计算有效频率，NVML 瞬时采样可能漏掉变化，参考
[NVIDIA 对频率指标的解释](https://forums.developer.nvidia.com/t/why-sm-frequency-drops-even-when-power-and-temperature-is-not-high/338911/2)。

两轮 176 条 mode 记录全部通过数值校验。GEMM lane0/M32/fence-on 的所有
模式、elect/M64/fence-off 的所有可用模式，均通过定向 memcheck/racecheck；
copy lane0/M32/fence-off 也通过定向 racecheck。没有改动生产 kernel，
没有增加 pytest 用例。样例 GEMM 是 REG168、localSizeBytes=56、dynamic
shared=226304；这不是生产 C5 的 STACK0，不能跨 binary 用绝对时延作归因。
加入 host pacing 前后的设备 SASS 完全相同。

最终数据与产物位于 `build/pf_saved_x_repro_20261008/`：

- `paced_r{0,1}.log`：正式表使用的完整样本，176 条记录；
- `paced_ncu_{copy,gemm}_{vector,tma_full,tma_read,tma_pipelined}.{csv,log}`：计数器；
- `clocks_paced.csv`：100 ms SM/功率限制采样；
- `racecheck_*.log`、`memcheck_*.log`：定向检查；
- `paced.sass`、`production_c7.sass`：样例和原候选的反汇编；
- executable SHA256：`5fb289b2c6d9228b5f3368012179cd6e777f8547267f26d83d7a9261c9c9948e`。

## 同步语义与分析边界

vector 的 LDS 完成后，待写数据已在寄存器中；未完成的 global store 不再
引用 A shared memory。TMA 则还会异步读取 A，必须确认源读完才能返还
empty credit。即使将目标完成延后到任务末尾，逐 K 的源读取等待仍在。
issuer warp 变慢，还会影响下一次四个 warp 一起到达的 WGMMA。

这里的 A 由 TMA load 写入，并已等待 full barrier，没有普通 SMEM store。
因此可独立消融 generic → async proxy fence；不能把这个结论套到用普通
SMEM store 准备数据的 GEMM epilogue。`elect` 消除发起处 peeling loop、
bulk group 为 per-thread、read wait 与 full wait 的区别，均依据
[NVIDIA async copies 文档](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html)。
WGMMA 的跨 warp 等待解释参见
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

这份复现可以排查局部复制与流水线耦合，不能单独解释全部 PF 回退：此前
C7/C8 在 cache-off 下也有回退，涉及完整 kernel 代码生成/物理行序等因素。
实验发现 peeling loop 也不等于实际重复发送了 32 份数据；活跃 issuer
只有一个线程，不能据静态反汇编夸大事务数量。
