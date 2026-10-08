# PF Saved-X：vector / TMA 最小复现

这是独立 CUDA 诊断，不改变正式 PF。问题是：为什么在 FC1 已加载的 A-stage
上保存 X，vector 比 C7/C8 的 TMA 集成更快？不是测通用 memcpy 峰值。

**术语澄清：本文件原来的 `copy` 是带 producer/ring 同步的 GM→SM→GM
流程，不是裸 shared→global 吞吐。后文新增的独立 store 对照表明：去掉
ring/input refill 后，多数配置 vector/TMA 持平，部分配置 TMA 更快。不能
把原来的约 10% 差距解读成 vector 的裸拷贝能力更强。**

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

## 2026-10-08 补充：独立 shared→global store

更正：本节保留初版 `4ff76fa` 的历史数据。进一步检查循环边界发现，
vector 的 LDS 被 ptxas 提升到循环外，实际循环是 register→global store；
仅有 `asm volatile` 并不足以阻止这一优化。因此本节不能作为严格的
逐 tile shared→global 对比。修正后的样例及更多 CTA 数据见下一节。

新增 [tma_store_repro.cu](tma_store_repro.cu)，只依赖 CUDA/CUTLASS headers。
没有 GEMM、GM→SM load、ring、source 覆盖、跨 CTA credit 或 multicast。
每个 CTA 一次性准备不可变的 128 B-swizzled shared tile，随后写到互不重叠
的 global tile。三种写法在同一个 kernel binary 内切换：

- vector：256 threads 合作；初版源代码意图逐 tile 执行 LDS，但生成的
  SASS 实际在循环外读取 shared，随后反复从寄存器写出。
- TMA serial：一次 tensor store 对应一个 tile，每个 tile commit + full
  wait。一次 election 包围整个循环，避免 peeling，并保持 issuer 一致。
- TMA queued：每个 tile commit + `wait_group 7`，允许最多八组在途，
  最后 full wait。源始终不变，因此没有逐 tile 源复用的等待需求。

shape 编译期固定，避免运行时整数除法影响地址计算；无 spill。两者都包含
相同的一次 shared 初始化、proxy fence 和同步，分配/数值检查在计时外。
`none` 测出的准备及 launch 开销约 5–9 us，下表没有人为扣除它。
所有模式都计到 kernel 完成，不能拿 TMA 发起耗时与 vector 完成耗时比较。
以下构建命令使用工作区源码；历史表来自初版 `4ff76fa`，不能用修正后的
当前源码复现其 binary。旧 binary/反汇编保留在本节末尾的产物目录中。

```bash
mkdir -p build/tma_store_repro_20261008
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
  -I/opt/cutlass/include benchmarks/tma_store_repro.cu \
  -lcuda -o build/tma_store_repro_20261008/repro

# 参数：tile_rows global_width output_MiB grid_multiplier samples device [mode]
# 64 rows × K64 × BF16 = 每 tile 8 KiB；width64 连续，width4096 为 PF 跨行布局。
build/tma_store_repro_20261008/repro 64 64 512 1 40 0
build/tma_store_repro_20261008/repro 64 4096 512 1 40 0
build/tma_store_repro_20261008/repro 64 4096 512 4 40 0
```

L20X GPU0/132 SM，SM 设定 1620 MHz，memory 3201 MHz，写出 512 MiB。
8 warmup、40 measured、2 ms 计时外间隔；配置正反序两轮，各轮内部 mode
顺序交替。下表为两轮 median 的均值，单位 **us**：

| tile | global 布局 | grid CTAs | vector | TMA serial | TMA queued |
|---|---|---:|---:|---:|---:|
| 4 KiB | 连续 | 132 | 126.376 | 126.776 | 126.872 |
| 4 KiB | 连续 | 528 | 128.568 | 128.648 | 128.800 |
| 4 KiB | PF 跨行 | 132 | 132.608 | 130.776 | 132.344 |
| 4 KiB | PF 跨行 | 528 | 138.608 | 133.688 | 134.432 |
| 8 KiB | 连续 | 132 | 131.216 | 131.904 | 130.872 |
| 8 KiB | 连续 | 528 | 133.776 | 131.792 | 132.240 |
| 8 KiB | PF 跨行 | 132 | 133.408 | 133.232 | 133.344 |
| 8 KiB | PF 跨行 | 528 | 151.600 | 134.176 | 136.536 |
| 32 KiB | 连续 | 132 | 131.528 | 131.320 | 131.544 |
| 32 KiB | 连续 | 528 | 132.200 | 131.560 | 132.608 |
| 32 KiB | PF 跨行 | 132 | 136.280 | 136.400 | 136.104 |
| 32 KiB | PF 跨行 | 528 | 148.888 | 135.728 | 138.264 |

多数配置差异约 0–2%，视为基本持平，而不是 vector 或 TMA 的稳定优势。
PF 跨行/grid528 时，TMA serial 的 8/32 KiB 耗时比 vector 低
**11.49% / 8.84%**，两轮方向一致。8 KiB/grid132 的有效写出带宽两者
都约 **4.0 TB/s**。这是持续写出的有效吞吐，不是一次小块拷贝的指令延迟，
也不是 GM→GM memcpy 或 GM→SM load 的比较。更多 TMA group 在途不保证
更快；本例 grid 级并发已很高，queued 没有稳定胜过 serial。

grid multiplier 只是 launch CTAs / SM 数，不强制逐 SM 均匀驻留。
PF 本身还有约 226 KiB shared 预留；这里 grid528 的结果不能直接移植为
PF 的驻留配置。新旧样例也不是只改变一项的实验：去掉了 input refill、
ring credit 和大 shared 预留，所以不能把两张表的差值归到某一条 wait。

全部 96 条记录逐 word 校验通过；定向 memcheck/racecheck 零错误。
100 ms NVML 采样均为 1620 MHz、无 SW power cap。另行 NCU application
replay 下，8 KiB/PF 布局的 vector/TMA、grid132/528 有效频率约
1579–1582 MHz，差异不到 0.2%；DRAM 写出均约 534.6–534.7 MB。
依然应区分设定时钟和 kernel 窗口内的有效频率，profiler 时间不混入上表。

初版源码 SHA256 为 `eedb764e32ce664ee0b9259172b607b38ef59aa375f8e222cd7e0e61a534d1f5`。
原始样本、时钟、NCU、检查和反汇编均在 `build/tma_store_repro_20261008/`：
`matrix_r{0,1}.log`、`clocks.csv`、`ncu_*.{csv,log}`、`memcheck.log`、
`racecheck.log`、`repro.sass`。早期 `probe_*.log` 使用了运行时 shape，
不属于最终结果，不与静态地址计算的上述数据混用。

## 2026-10-08：修正 LDS 外提，补充连续地址与更多 CTA

将 vector 的 PTX 改为 `ld.volatile.shared.v4.b32`，只改诊断样例，不改
正式 PF。原来的 C++ `asm volatile` 没有阻止 ptxas 外提不可变源的 LDS。
修正后，8 KiB/连续布局的 SASS 回边为 `BRA 0x410`，两条 LDS 位于
`0x410/0x430`；PF 跨行布局回到 `0x420`，两条 LDS 位于 `0x420/0x460`。
两者均在循环内实际读取 shared。NCU 在连续布局/grid1056 下测到
**4,194,304 个 shared-load wavefront**，对应完整 512 MiB 的读取，
shared-load bank conflict 为零，不再是每个 CTA 只读一次 shared。

连续布局为 width64 BF16：每行 128 B、每 tile 64 行共 8 KiB，整个 tile
连续，warp 每条 vector store 的 512 B 也连续；tile 按 grid-stride 分配。
仍固定写出 **512 MiB**，256 threads/CTA，GPU0/L20X/132 SM，SM 时钟
设置 1620 MHz、memory 3201 MHz。分配/逐 word 校验不计时；每配置
8 warmup + 40 samples、mode 正反序交替、计时外间隔 2 ms；配置顺序
正反各测一轮，下表为两轮 median 的均值，单位 **us**。

| launch CTAs | grid / SM 数 | vector | TMA serial | TMA queued |
|---:|---:|---:|---:|---:|
| 132 | 1 | 131.208 | 132.184 | 130.768 |
| 264 | 2 | 131.288 | 131.432 | 131.112 |
| 528 | 4 | 134.456 | 134.192 | 133.496 |
| 1056 | 8 | 133.176 | 131.376 | 131.432 |
| 2112 | 16 | 131.864 | 129.744 | 130.944 |
| 4224 | 32 | 128.280 | 128.112 | 128.456 |
| 8448 | 64 | 125.728 | 124.792 | 125.112 |

结论：连续布局下基本持平，TMA serial 相对 vector 的耗时变化为
**+0.74% 至 −1.61%**，没有跨行布局约 12% 的优势。更多在途 group
也没有稳定改善。增加 launch CTA 不等于同比增加同时驻留 CTA，超出
硬件驻留容量后分多批执行；表中第二列不是实测 occupancy。更多 CTA
仍只处理同一份 512 MiB，不是按 CTA 数增加数据量。一次性的 shared
初始化随 CTA 数增加，已包含在表中；`none` 为 5.176–21.672 us，未扣除。

同轮补测修正后的 PF 跨行布局：

| launch CTAs | vector us | TMA serial us | TMA queued us |
|---:|---:|---:|---:|
| 132 | 134.888 | 135.192 | 134.504 |
| 528 | 152.344 | 134.032 | 136.640 |

grid528 下 TMA serial 耗时低 **12.02%**，说明原先的性能方向并非由
LDS 外提造成，但旧版“逐 tile shared→global”的描述仍须撤回。
不能将本例的独立 store 吞吐推广为 PF 中的 stage 释放/流水线性能。

共 72 条计时记录，其中 54 条复制模式记录逐 word 校验通过，另外
18 条为 `none`。定向 memcheck（连续/grid8448）、racecheck
（连续/grid1056）均零错误/零 hazard，检查数据量为 2 MiB。
连续/PF 布局分别 REG26/27、LOCAL0、shared8192，模式共享同一 binary。
100 ms NVML 采样均为 1620 MHz、无 SW power cap；独立 NCU 的
grid1056 vector/TMA serial 有效频率为 1548.5/1581.1 MHz，因此仍只称
1620 MHz 设置下的测量，不宣称 kernel 窗口严格恒频。NCU 时间不混入
上表；约 1–2% 的差异不解读为稳定优势。

复现连续地址扫描（再逆序执行一轮）：

```bash
mkdir -p build/tma_store_repro_20261008/contiguous_volatile
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
  -I/opt/cutlass/include benchmarks/tma_store_repro.cu \
  -lcuda -o build/tma_store_repro_20261008/contiguous_volatile/repro
for grid_multiplier in 1 2 4 8 16 32 64; do
  build/tma_store_repro_20261008/contiguous_volatile/repro \
    64 64 512 "$grid_multiplier" 40 0
done
```

新产物单独保存于 `build/tma_store_repro_20261008/contiguous_volatile/`，
包括 `matrix_r{0,1}.log`、`repro.sass`、`ncu_8_*.{csv,log}`、
`clocks.csv`、`memcheck.log`、`racecheck.log`，没有覆盖初版数据。
源码 SHA256：`ca08ce96cdb69a8c7ae9580c04a04a05530fa551a860363c59b5ff6f56f96af4`。
