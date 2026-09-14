**WeftMoE 性能优化与稳定性验证，2026-09-13**

本报告及配套 JSON 是历史测量，代码与复现工具保存在 `precombine` 分支的
[`7483933`](https://github.com/yulangz/WeftMoE/tree/7483933) 提交中。
`optimize` 分支随后移除了 pre-combine；复现本报告时请使用上述提交，
这些数据不代表移除后的版本已经重新完成性能测量。

本次修改保留确定性的归约顺序，优化 BWD GEMM 的有效工作量、就绪任务查找和本机 Return 调度，并为两个目标模型选择 FWD 通信配置。优化前基线为 `fa78961`；所有正式数据均来自冻结的优化前/后可执行文件。同一个最终可执行文件同时包含 FWD 和 BWD。

BWD 相对原版的 P50 加速比为 **1.02–1.86×**。FWD+BWD 阶段 P50 之和相对原版为 **1.01–1.55×**；在 6/8 个 case 中低于 Flex。FWD 单独有 1/8 个 case 快于 Flex，BWD 为 6/8。详细数据包括仍落后的 case，不能据此声称全面超过 Flex。

测试使用模型的 routed-expert 尺寸，普通 BF16 SwiGLU，包含 gate 乘法；不包含 router、shared expert、attention 或完整模型中的其他激活/量化策略。8K/64K 均指**每个 EP rank 的 token 数**，即 EP8 的全局 batch 为 64K/512K，EP16 为 128K/1M。

| 目标尺寸 | Hidden | Intermediate | 全局 experts | Top-k |
|---|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 2048 | 512 | 256 | 8 |
| DeepSeek-V4-Flash | 4096 | 2048 | 256 | 6 |

尺寸来源：[Qwen 官方配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)、[DeepSeek 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)。本次没有测试 DeepSeek-V4-Pro 或 Ling-3-Flash。

硬件为两台各 8 张 SM90 GPU 的节点；驱动报告型号 `NVIDIA L20X`，每卡显存 143771 MiB，采样时 SM/显存时钟为 1980/3201 MHz，功耗上限 700 W。节点内为 NVLink，节点间使用 8 条系统报告为 200 Gb/s 的 RoCE rail。EP16 的 Flex 运行保留了两端 HCA 计数器、NCCL 日志和 DeepEP/NVSHMEM 配置，确认所有 rail 有双向流量。HCA 计数器包含初始化、控制和 payload，不能用其整段吞吐推算单个通信 kernel 的带宽。

每台主机为两颗 Intel Xeon Platinum 8558，共 96 个物理核、192 个逻辑 CPU、2 个 NUMA node。运行容器的 CPU 配额均为每 100000 µs 周期可用 10576000 µs CPU 时间，即 105.76 个逻辑 CPU 的平均时间额度。双方使用默认进程调度，没有单独给某一实现绑核；`OMP_NUM_THREADS=1`。

构建使用 CUTLASS 4.2.0、GCC 13.3.0 和 Python 3.12.3。软件为 CUDA compiler 12.9、驱动 570.148.08、PyTorch 2.9.1+cu129、Megatron-Core 0.17.0、Transformer Engine 2.14.1+27de67e1、DeepEP 1.2.1+567632d。双方实际加载同一 NCCL 2.31.2；PyTorch 编译时报告的 2.27.5 不是此次实际运行的 NCCL。完整设备和软件记录位于 `results/performance_20260913/hardware.json`。

**计时口径**

主指标是一次调用从主机发起到输出可用的 wall time，以 P50 比较典型调用性能，同时列出均值和 P95。先对每个 iteration 取所有 rank 的最大值，再计算均值、P50、P95 和标准差。双方在每次迭代开始前执行不计时的 NCCL 全 rank 屏障，屏障完成后才开始分配 input 和记录 wall time。每项先预热 10 次，再测量 30 次；第二轮反转 case 顺序和实现顺序。表格合并两轮全部 60 个样本，不挑选更快的一轮，也不剔除慢样本。实测出现过 wall time 超过 30 ms、GPU 区间仍约 3 ms 的主机端长尾，因此不只用均值推断 kernel 收益。pre-combine 额外测量 30 个样本。正式计时期间没有并行编译或其他 GPU 测试。

| 项目 | 计入每次调用 | 在预热前准备并复用 |
|---|---|---|
| FWD | 新 input、selected experts、gate 的分配与 D2D 初始化；输出、保存的 X/A/B 与相关固定 tensor 分配；Notify/路由处理、dispatch、GEMM、combine；等待输出可用 | 权重、workspace、peer mappings、NCCL GIN/RDMA context、DeepEP buffer |
| BWD | 新 dY 的分配与 D2D 初始化；dX、dGate、dW1、dW2 的分配；dispatch、重算 U、全部梯度计算、combine；等待输出可用 | 前向已保存的 X/A/B、权重、workspace、通信 context |

双方均使用预热后的 caching allocator，每次调用仍请求新的 tensor 所有权；native cache 在预热后发生 miss 会使测试失败。native 使用测试专用的同步 owner-aware allocator，Flex 使用 PyTorch allocator。旧输出在计时前释放，新输出和前向 checkpoint 保留到调用完成。输入从相同的预生成 device fixture 复制，未把随机数据生成或从磁盘加载计入算子成本。

共同输入是 compact selected-expert/gate 数据。Flex 在计时内构造其接口需要的 dense routing/probability tensor，native 直接消费 compact 数据；这一表示转换成本包含在端到端结果中。

BWD 单独计时从已保存的 X/A/B 开始。Flex 在每次 BWD 前于计时外重建 autograd 图；native 复用值相同的保存状态。两者均测量预热后的正常缓存状态，没有做人工 L2 flush。

Flex 使用真实的 `MoEFlexTokenDispatcher + DeepEP + TEGroupedMLP`，启用 deterministic 配置与 selective `moe_act` checkpoint。双方保留 routed X 和 BF16 FC1 A/B，在 BWD 重算 SwiGLU、gate multiply 和 U。DeepEP 的 SM 数量经过独立筛选：EP8 比较 16/48/80，EP16 比较 16/24/48，以 FWD+BWD wall time 之和选取；8 个 case 均选择 EP8=80、EP16=48。没有把 Flex 限制为 Socket 或刻意减少通信资源。

这是 C++ native 调用栈与 Megatron/PyTorch 调用栈的端到端比较，包含各自的主机调度和 tensor 管理开销。CUDA event 数据另存于 JSON：native FWD event 不包含 Notify/固定 tensor 准备，native BWD event 也不包含调用前的固定 tensor 准备，因此不直接拿它们与 Flex 的完整 event 区间作性能比值。各内部阶段存在重叠，不能把阶段耗时相加作为总耗时。

**FWD，pre-combine 关闭，wall time / ms**

每个单元依次为 P50 / 均值 / P95，单位均为 ms。

| 模型 | EP | tokens/rank | 原版 P50/均值/P95 | 优化后 P50/均值/P95 | Flex P50/均值/P95 | 前/后 | Flex/后 |
|---|---|---|---|---|---|---|---|
| Qwen | 8 | 8K | 3.748 / 3.764 / 3.856 | 3.485 / 3.510 / 3.748 | 4.554 / 4.617 / 4.825 | 1.08× | 1.31× |
| Qwen | 8 | 64K | 27.357 / 27.669 / 27.576 | 23.744 / 24.478 / 23.887 | 23.197 / 23.210 / 23.324 | 1.15× | 0.98× |
| Qwen | 16 | 8K | 6.597 / 6.616 / 6.853 | 6.594 / 7.193 / 6.893 | 5.077 / 5.195 / 5.672 | 1.00× | 0.77× |
| Qwen | 16 | 64K | 35.852 / 36.235 / 37.054 | 35.272 / 35.702 / 35.452 | 28.875 / 29.068 / 29.779 | 1.02× | 0.82× |
| Flash | 8 | 8K | 8.756 / 8.771 / 8.906 | 8.352 / 8.759 / 8.563 | 8.017 / 8.157 / 8.263 | 1.05× | 0.96× |
| Flash | 8 | 64K | 58.278 / 58.579 / 58.717 | 57.729 / 57.950 / 57.904 | 55.975 / 56.031 / 56.591 | 1.01× | 0.97× |
| Flash | 16 | 8K | 11.824 / 12.985 / 12.943 | 10.896 / 11.032 / 12.271 | 9.392 / 9.522 / 9.735 | 1.09× | 0.86× |
| Flash | 16 | 64K | 80.079 / 82.948 / 82.578 | 72.488 / 72.769 / 73.618 | 65.078 / 65.315 / 66.379 | 1.10× | 0.90× |

**BWD，wall time / ms**

每个单元依次为 P50 / 均值 / P95，单位均为 ms。

| 模型 | EP | tokens/rank | 原版 P50/均值/P95 | 优化后 P50/均值/P95 | Flex P50/均值/P95 | 前/后 | Flex/后 |
|---|---|---|---|---|---|---|---|
| Qwen | 8 | 8K | 4.372 / 4.374 / 4.413 | 3.859 / 3.866 / 3.927 | 5.635 / 5.731 / 6.223 | 1.13× | 1.46× |
| Qwen | 8 | 64K | 55.509 / 55.509 / 55.591 | 29.847 / 29.824 / 30.334 | 35.081 / 35.112 / 35.320 | 1.86× | 1.18× |
| Qwen | 16 | 8K | 6.363 / 6.363 / 6.413 | 6.235 / 6.236 / 6.269 | 6.152 / 6.189 / 6.453 | 1.02× | 0.99× |
| Qwen | 16 | 64K | 54.486 / 54.390 / 54.744 | 46.382 / 46.394 / 46.520 | 40.729 / 40.831 / 41.278 | 1.17× | 0.88× |
| Flash | 8 | 8K | 11.988 / 11.992 / 12.059 | 10.513 / 10.513 / 10.636 | 13.135 / 13.215 / 13.369 | 1.14× | 1.25× |
| Flash | 8 | 64K | 123.549 / 123.687 / 125.226 | 88.220 / 88.147 / 89.649 | 101.416 / 101.329 / 102.128 | 1.40× | 1.15× |
| Flash | 16 | 8K | 12.356 / 12.355 / 12.424 | 10.845 / 10.855 / 10.925 | 14.574 / 14.616 / 14.776 | 1.14× | 1.34× |
| Flash | 16 | 64K | 115.849 / 115.849 / 117.182 | 83.900 / 84.075 / 85.686 | 112.275 / 112.351 / 113.247 | 1.38× | 1.34× |

“前/后”和“Flex/后”均为 P50 比，数值大于 1 表示优化后的 WeftMoE 更快。下面的合计只是两个独立测得的阶段 P50 之和，不是另一次完整训练 step 测量。

| 模型 | EP | tokens/rank | 原版 FWD+BWD P50 | 优化后 FWD+BWD P50 | Flex FWD+BWD P50 | 前/后 | Flex/后 |
|---|---|---|---|---|---|---|---|
| Qwen | 8 | 8K | 8.120 | 7.344 | 10.189 | 1.11× | 1.39× |
| Qwen | 8 | 64K | 82.867 | 53.591 | 58.278 | 1.55× | 1.09× |
| Qwen | 16 | 8K | 12.961 | 12.830 | 11.229 | 1.01× | 0.88× |
| Qwen | 16 | 64K | 90.338 | 81.654 | 69.603 | 1.11× | 0.85× |
| Flash | 8 | 8K | 20.744 | 18.864 | 21.151 | 1.10× | 1.12× |
| Flash | 8 | 64K | 181.827 | 145.950 | 157.392 | 1.25× | 1.08× |
| Flash | 16 | 8K | 24.180 | 21.741 | 23.967 | 1.11× | 1.10× |
| Flash | 16 | 64K | 195.928 | 156.388 | 177.354 | 1.25× | 1.13× |

**开启 pre-combine 的 FWD，wall time / ms**

| 模型 | EP | tokens/rank | 原版 PC1 P50/均值/P95 | 优化后 PC1 P50/均值/P95 | 优化后 PC0 第1轮 P50 | PC0/PC1 |
|---|---|---|---|---|---|---|
| Qwen | 8 | 8K | 3.823 / 5.734 / 30.814 | 3.476 / 3.478 / 3.546 | 3.477 | 1.00× |
| Qwen | 8 | 64K | 24.838 / 25.608 / 24.987 | 22.272 / 22.849 / 22.365 | 23.717 | 1.06× |
| Qwen | 16 | 8K | 6.660 / 6.692 / 6.972 | 6.578 / 6.682 / 7.136 | 6.595 | 1.00× |
| Qwen | 16 | 64K | 35.784 / 36.058 / 38.550 | 34.644 / 35.184 / 35.112 | 35.296 | 1.02× |
| Flash | 8 | 8K | 9.196 / 9.192 / 9.358 | 8.658 / 8.652 / 8.750 | 8.345 | 0.96× |
| Flash | 8 | 64K | 60.931 / 60.923 / 61.312 | 60.281 / 60.403 / 60.872 | 57.721 | 0.96× |
| Flash | 16 | 8K | 11.762 / 12.612 / 12.546 | 11.342 / 11.393 / 11.569 | 10.762 | 0.95× |
| Flash | 16 | 64K | 83.630 / 83.368 / 84.756 | 70.609 / 71.786 / 81.723 | 72.709 | 1.03× |

开启 pre-combine 会增加 owner 内的一次 BF16 归约舍入，契约仅要求相同 EP 下的 batch invariance。没有把开启和关闭时输出逐 bit 相同作为要求，也没有把 EP8 与 EP16 的 pre-combine 输出逐 bit 相同作为要求。上表复用 primary 配置；这不是对 pre-combine 模式重新独立调优的最优值承诺。

全部 60 样本的最大 CV 为 59.0%。JSON 保留每轮均值和每个原始样本；接近 1× 的差异应结合波动判断。 以下项目两轮均值相差超过 5%，应谨慎解释：

| case | phase | 实现 | 第1轮/ms | 第2轮/ms | 合并 CV |
|---|---|---|---|---|---|
| qwen_ep8_t8192 | backward | flex | 5.560 | 5.901 | 4.6% |
| qwen_ep16_t8192 | forward | after | 7.748 | 6.638 | 59.0% |
| flash_ep8_t8192 | forward | after | 9.099 | 8.418 | 32.8% |
| flash_ep16_t8192 | forward | before | 14.186 | 11.784 | 38.3% |
| flash_ep16_t65536 | forward | before | 86.007 | 79.889 | 14.9% |

pre-combine 的以下单轮结果 CV 超过 10%，均值受长尾影响，不能把均值比全部归因于 kernel 优化：

| case | 实现 | P50/均值/P95 | CV |
|---|---|---|---|
| qwen_ep8_t8192 | before | 3.823 / 5.734 / 30.814 | 121.6% |
| qwen_ep8_t65536 | before | 24.838 / 25.608 / 24.987 | 16.3% |
| qwen_ep8_t65536 | after | 22.272 / 22.849 / 22.365 | 13.7% |
| flash_ep16_t8192 | before | 11.762 / 12.612 / 12.546 | 35.5% |

**实现与取舍**

- BWD 的 dU/dX projection 在两个 N 维 tile 都能成对时，改为两张完整 M128 CTA 沿 N 配对，两个 math warpgroup 都处理有效行，同时 multicast A。N tile 数为奇数时保留 M64 分行与 multicast B；不补无用 N tile，也不等待第二个 M-block 到达。TMA map、共享内存计算、任务数量和完成计数同步更新。
- dU/dAB/dX 就绪 bitmap 增加一层摘要，一次 acquire 跳过最多 64 个空 word，同时保留按专家优先的访问顺序，便于复用权重。发布、清除和修复均保留 acquire/release 链。额外处理了“旧阶段退役、下一阶段重发、摘要被另一消费者清空”的嵌套交错，修复时同时恢复 owner 和摘要。
- 本机 dX 完成后把 M-block 发布到 append-only Return 队列，减少 Return worker 对未完成块的循环扫描。GIN 路径保留按专家扫描；成对测量发现把 FIFO 同时应用到 GIN 会有小幅回退，因此没有采用。
- FWD 对选定尺寸使用现有的向量化 Return 拷贝路径，并调整 relay CTA 数：Qwen 通常 24，EP8/64K 为 40；Flash 为 16。参数保存在独立 profile，核心实现没有根据模型名称分支。
- 修复可选 `wgmma_wait_group=1` 的 stage 生命周期：FC1 每个 K iteration 提交两组，等待与回收必须按两组计算；FC1/FC2 都只回收上一 stage，并在 epilogue 前 drain。该模式验证通过，但筛选没有稳定性能收益，最终 profile 保持 0。

dW 的 K 维累加和最终归约顺序保持不变；任务完成先后不决定浮点加法顺序。没有引入浮点 atomicAdd、非确定性 split-K 或跨调用复用旧梯度。BWD 两行 TMA 预取也做过筛选，未保留没有稳定收益的实现。单独的本机 Return 队列相对摘要 bitmap 版本在 Qwen 的成对测量中，BWD GPU 区间约缩短 2–4%；这一消融结果不能代替上面的最终整体结果。

**正确性证据与覆盖范围**

| 验证 | 覆盖 | 结果 |
|---|---|---|
| 完整输出重复 | 2 模型 × EP8/EP16 × 8K/64K × PC0/PC1 × FWD/BWD，共32项，各25次 | 所有 rank、全部输出逐 bit 相同，包括全部 dW |
| FWD batch invariance | 8K/64K、129/513；两个模型、EP8/EP16、PC0/PC1 | 全部公共前缀零 bit 差异，输入/routes/gates 亦相同 |
| FWD 跨 EP | PC0 的相同源 shard 与同一全局 batch 的 EP8→EP16 重划分 | 零 bit 差异 |
| FWD 独立数值参考 | 8个模型/EP/PC组合，各56个坐标 | 与规定阶段 cast 的 FP64 参考一致 |
| BWD 独立数值参考 | 完整 E256、两个模型、EP8/EP16、T129，所有梯度 | 通过；保留 X/A/B 与输入/投影一致 |
| BWD 跨进程与原版 | E16/EP8、E32/EP16 的 T129 尾块，各模型；PC0/PC1 | 四类梯度完整文件逐字节相同 |
| Flex FWD 数值与输入 | 两个模型 × EP8/EP16，全8K输出 | 输入/routes/gates 完全一致；输出在设定误差内 |
| Flex 自身 BWD 对照 | 8个规模，checkpoint 对非重算及2次重复 | 逐 bit 一致 |
| 其他尺寸与生命周期 | H1536/I768，M64 fallback、wait1、动态/static/Graph；Qwen M128 Graph | 全部通过 |

最终的 FWD batch 检查比较 8K 与 64K 的完整 8K 公共前缀，覆盖 EP8/EP16、两个模型和 pre-combine 两种模式。关闭 pre-combine 时，另比较 EP8/EP16 的相同源 shard，以及同一全局 64K batch 在 EP8×8K 与 EP16×4K 间的重新划分。输出、input、routes 和 gates 都按完整文件检查字节和 SHA256。64K 的跨 EP dump 比较覆盖前 8K token；64K 全输出另有逐次重复检查。另用 129/513 token 测试非对齐 batch，改变后续源 rank 的 token 在 GEMM tile 中的行位置，避免只检查整 tile 对齐的长度。

独立 FWD 参考读取真实 dump，重建 fixture 权重，以 NumPy FP64 dot product 配合规定的 BF16 阶段 cast 和规范化 FP32 combine 计算。每个目标模型/EP/pre-combine 组合检查 56 个坐标，全部与参考一致；这项是抽样数值参考，不是全输出 FP64 证明。与 Flex 的比较检查全 8K 输出，并先确认 input、routes、gates 完全相同。

完整 256 专家的 BWD 独立 autograd 参考检查每个梯度元素，最大相对 L2 为 1.178e-04；以下为全局归约后的相对 L2。参考在精确的 BF16 X/A/B 上求导，与 native 的逐 bit 重复检查是不同的验证。

| 模型/EP | dX | dGate | dW1 | dW2 |
|---|---|---|---|---|
| qwen_ep8 | 6.483e-05 | 1.382e-07 | 1.377e-05 | 1.436e-05 |
| qwen_ep16 | 6.947e-05 | 1.400e-07 | 2.048e-05 | 2.260e-05 |
| flash_ep8 | 1.156e-04 | 1.970e-07 | 1.807e-05 | 1.832e-05 |
| flash_ep16 | 1.178e-04 | 1.940e-07 | 1.940e-05 | 2.446e-05 |

WeftMoE PC0 与 Flex 的完整 FWD 输出比较如下；owner 内合并/舍入次序不同，因此比较数值误差，不要求双方输出逐 bit 相同。

| 模型/EP | 最大 rank 相对 L2 | 最大绝对误差 |
|---|---|---|
| qwen_ep8 | 4.422e-03 | 7.629e-06 |
| qwen_ep16 | 4.730e-03 | 7.629e-06 |
| flash_ep8 | 4.296e-03 | 3.052e-05 |
| flash_ep16 | 4.672e-03 | 6.104e-05 |

另外检查 H=1536、I=768、E=16、top-k=6：该尺寸触发 BWD 奇数 N tile 的 M64 fallback，FWD 使用 wait=1 且有奇数个 FC2 pipeline wave。129/513 token 的共同前缀、pre-combine 两种模式、独立数值参考均通过。本机 EP2、E16、top-k=2 的动态/static buffer 回归另外覆盖严重路由偏斜、workspace 扩容后旧状态保留，以及 6 次 CUDA Graph 重放，比较了 dX、dGate、dW1、dW2 全部字节。

协议 CUDA 回归覆盖 4226 个 owner、跨 4096-owner 摘要边界及 11 种发布/退役交错；权重队列和 FWD input 队列回归通过。CPU 检查为 77 个 Python 测试及 102 个 subtest、16/16 host 测试、16 个公共头独立编译和 34 个旧 API 编译拒绝检查。生产 `.so` 实际构建成功。Compute Sanitizer memcheck 的设备内存检查为 0 errors；NCCL 初始化中的已处理架构探测触发 API error 报告，保留原始日志后，使用 `--report-api-errors no` 完成设备内存检查，未把原始 40 个 API 报告写成“无报错”。

这些数据验证了本次运行的确定性与已覆盖的边界条件，不是任意调度和所有支持尺寸的形式化证明。生产尺寸/拓扑接口未缩窄；未对其他模型、不同网络拓扑、真实 router 负载分布或完整模型训练性能作未经测量的保证。

**复现与交付**

最终配置位于 [profiles.json](../configs/performance_20260913/profiles.json)，引用的 5 个 header 均同时配置 FWD/BWD；Qwen EP8 的 64K 有单独 profile，其余组合共享 8K/64K 配置。生产构建示例：

```sh
NVCCFLAGS="-include $PWD/configs/performance_20260913/qwen_ep8.hpp" \
  python3 build_native.py lib --output build/libweftmoe_qwen_ep8.so --jobs 16
```

原始日志、dump、源代码快照和冻结二进制在 `results/performance_20260913/`，大文件不进入 Git。优化前只移植相同 benchmark 计时逻辑和合并配置的 include 支持，保留原始 kernel/调度。正式 manifest 记录二进制 SHA256、执行顺序、统一屏障口径和 Flex 参数。早期 `formal_unsynchronized/` 的 native 缺少与 Flex 对齐的迭代前屏障，已经排除在主表之外，保留为诊断材料。可移植统计与原始主指标样本见 [performance_20260913.json](performance_20260913.json)。

在相同工作区重跑正式比较时使用新的 output 目录，避免恢复已经完成的样本：

```sh
python3 benchmarks/run_fair_comparison.py \
  --before-builds results/performance_20260913/formal_before_build_sync \
  --after-short results/performance_20260913/formal_after_short_sync \
  --after-long results/performance_20260913/formal_after_long_sync \
  --flex-selection results/performance_20260913/flex_selection.json \
  --output results/recheck --iterations 30 --passes 2
python3 benchmarks/summarize_fair_comparison.py \
  --directory results/recheck --output results/recheck/summary.json
```

主机参数可通过 `--remote`、`--port`、`--socket-ifname` 更换。`run_performance_matrix.py build/build-pair` 构建新 shape/profile，`weft/flex` 分别执行实现；`run_backward_validation.py --experts 256` 执行完整专家数的独立 BWD 参考。普通 profile 调优继续使用 `tools/run.py` 和原有配置接口。详细计时与校验开关见 [benchmarks/README.md](README.md)。
