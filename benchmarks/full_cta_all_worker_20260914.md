# 全 CTA worker：性能与稳定性记录

全 CTA 布局已在 `full_cta_all_worker` 分支实现。最终测试中，BWD 的八个场景相对冻结 main 的历史同频记录均缩短耗时，范围为 4.4%–31.2%；FWD 尚未达到全面领先的目标。Qwen EP8/64K、Qwen EP16/8K 和 Flash EP16/8K 的 FWD 仍明确回退，不能据 BWD 收益或 Tensor Core 覆盖全部 SM 宣称整体目标已完成。

本报告记录 2026-09-14 18:20:20–18:26:58 UTC 的最终矩阵，2026-09-15 完成汇总。只包含 BF16 routed-expert 层：Qwen3.5-35B-A3B（H2048、I512、256 experts、top-k 8）和 DeepSeek-V4-Flash（H4096、I2048、256 experts、top-k 6）。长度为每 rank 的 8192/65536 tokens；不含 router、shared expert 或完整模型的其他层。

## 完整算子对照

主指标为各 iteration 最慢 rank 的 host wall time，包含当次 input、output、routing/gate、保存状态或梯度等固定 tensor 的申请请求，以及输入 fixture 的 device-to-device clone；缓存分配器经过预热，通信 buffer、RDMA context、权重及长期 workspace 的初次准备不计入。释放上一轮结果和 rank barrier 位于计时外，当前输出保留到调用完成。每项预热 10 次、测量 10 次，保留全部样本及离群点。

**FWD 的 main 是本轮相邻新测；BWD 的 main 是本轮开始时的历史同频测量。** 冻结 main 为 `62382e27e8c11c5cbc331da860b56868d6ecdba4`。最终版本 FWD/BWD 共 16 项都重新测量，未用旧候选的数字填表。8K 按 main FWD → 当前 FWD → 当前 BWD 排序；64K 按当前 FWD → 当前 BWD → main FWD 排序。计时窗口没有并行 CUDA 编译或其他测试作业。

下表单位 ms，单元格为 **main → 当前（耗时变化）**；负值表示当前更快。

| 模型 | EP | tokens/rank | FWD wall，相邻 main | BWD wall，历史 main |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 8 | 8192 | 3.848 → 3.858（+0.2%） | 4.317 → 4.005（−7.2%） |
| Qwen | 8 | 65536 | 23.842 → 25.080（+5.2%） | 32.858 → 30.537（−7.1%） |
| Qwen | 16 | 8192 | 6.835 → 7.228（+5.7%） | 6.794 → 5.641（−17.0%） |
| Qwen | 16 | 65536 | 37.140 → 34.450（−7.2%） | 50.433 → 34.696（−31.2%） |
| Flash | 8 | 8192 | 7.897 → 7.885（−0.2%） | 12.058 → 10.847（−10.0%） |
| Flash | 8 | 65536 | 54.862 → 54.639（−0.4%） | 91.841 → 80.873（−11.9%） |
| Flash | 16 | 8192 | 10.711 → 11.203（+4.6%） | 12.272 → 10.774（−12.2%） |
| Flash | 16 | 65536 | 67.958 → 66.367（−2.3%） | 85.605 → 81.871（−4.4%） |

Qwen EP8/8K 双方都有离群样本：main/current wall 的 P50 为 3.670/3.723 ms，P95 为 5.321/5.079 ms。其 GPU 均值稍快、中位数稍慢，不能解释为稳定收益。Flash EP8 两项小于 0.5% 的差异也按基本持平看待。FWD 的明确收益出现在两个模型的 EP16/64K。

CUDA-event 时间单列用于诊断，仍为 main → 当前。native FWD event 不覆盖全部 Notify 和固定 tensor 准备；不能用它替代上面的 wall 时间。

| 模型 | EP | tokens/rank | FWD GPU ms | BWD GPU ms（历史 main） |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 8 | 8192 | 2.885 → 2.864 | 4.206 → 3.900 |
| Qwen | 8 | 65536 | 22.580 → 23.852 | 32.461 → 30.151 |
| Qwen | 16 | 8192 | 5.412 → 5.808 | 6.673 → 5.520 |
| Qwen | 16 | 65536 | 35.334 → 32.609 | 50.016 → 34.283 |
| Flash | 8 | 8192 | 7.018 → 7.018 | 11.943 → 10.712 |
| Flash | 8 | 65536 | 53.538 → 53.356 | 91.350 → 80.380 |
| Flash | 16 | 8192 | 9.249 → 9.796 | 12.141 → 10.630 |
| Flash | 16 | 65536 | 66.107 → 64.488 | 85.105 → 81.372 |

## Flex 历史参考

以下 Flex 数字来自此前同为 1620 MHz 上限、700 W 的正式记录，**本次没有重新配对测量 Flex**。采用真实 Megatron `MoEFlexTokenDispatcher + DeepEP + TEGroupedMLP`，deterministic 与 selective `moe_act` checkpoint；DeepEP 通信 SM 配置为 EP8=80、EP16=48。两者保持此前定义的固定 tensor 申请计时口径；Flex 所需的 compact→dense routing 转换保留在其 wall 内。没有在本轮重新穷举 Flex 参数。

| 模型 | EP | tokens/rank | 当前 FWD wall | 历史 Flex FWD wall | 当前 BWD wall | 历史 Flex BWD wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 8 | 8192 | 3.858 | 6.586 | 4.005 | 6.109 |
| Qwen | 8 | 65536 | 25.080 | 25.189 | 30.537 | 37.527 |
| Qwen | 16 | 8192 | 7.228 | 5.360 | 5.641 | 6.743 |
| Qwen | 16 | 65536 | 34.450 | 31.576 | 34.696 | 43.878 |
| Flash | 8 | 8192 | 7.885 | 9.230 | 10.847 | 14.698 |
| Flash | 8 | 65536 | 54.639 | 60.939 | 80.873 | 105.502 |
| Flash | 16 | 8192 | 11.203 | 10.316 | 10.774 | 15.974 |
| Flash | 16 | 65536 | 66.367 | 70.554 | 81.871 | 116.096 |

BWD 八项均低于历史 Flex wall；FWD 的 Qwen EP16 两项和 Flash EP16/8K 仍落后。Qwen EP8/8K 的历史 Flex wall 有长尾，均值 6.586 ms、P50 4.719 ms；不删除慢样本，也不把本表均值与旧报告的稳态中位数混用。native 与 Flex 的 CUDA-event 边界不同，机器记录保留原值，但不计算跨实现 GPU 加速比。原始日志、均值、P50/P95、min/max 和全部 samples 见[机器记录](full_cta_all_worker_20260914.json)。

## 保留的实现

每个持久 CTA 同时包含计算与通信角色；CTA 数根据设备 SM 数生成。在本次 132 SM 设备上为 132 CTA、66 个两 CTA cluster。最终布局为 384 threads：256 math threads、32 TMA producer threads、64 通信 copy threads、32 发布/通知 threads。GIN/RDMA 和本地搬运共享通信进度循环，没有永久保留一批不做 GEMM 的 SM。

NVLink payload 使用分段 TMA load/store，计算与通信各有 SMEM 和 barrier 域。FWD 保留双 buffer 跨行预取、Return 通知合并、Dispatch 结束后的 Return 批量调整、同 channel 有限 packet 预约，以及闲时 math 的一次有界 GIN 协助。预约在租约外打包，提交仍严格遵守 channel sequence、credit 和 terminal 顺序；不按完成顺序改变归并次序。

BWD 保持各数学输出的唯一 owner 和固定 K 累加顺序。通信 warp 在本 CTA 搬运与通知退役后复用 staging 做 dX Reduce，使其与 math warp 的 dW 重叠。FWD/BWD 寄存器、通信 buffer、tile/stage 分别配置；共享协议参数在组合编译时检查一致性。

Qwen 和 Flash 各使用一份配置覆盖全部 EP/长度，没有增加模型或 shape 的 CUDA `if/else`。Qwen 采用 channel limit 3、启用闲时 GIN 协助；Flash 采用 limit 1、关闭协助。这两个选择是通用编译调参轴，不是新的专用实现。FWD 为 math216/control64、双通信 buffer；BWD 为 Qwen216/64、Flash224/56、单 buffer。公开网格覆盖本地 FWD 8 项、多机 FWD 48 项、各拓扑 BWD 12 项，并过滤寄存器池、SMEM 和 Assist arena 组合不合法的候选。完整参数见[已选配置](../configs/full_cta_all_worker_20260914/README.md)，线程与同步协议见[架构说明](../docs/full_cta_all_worker.md)。

## 寄存器问题的结论

用户指出的寄存器溢出确实存在，但还发现了不伴随热 K 循环 local-memory 指令的 WGMMA 串行化；两者需要分别检查。

| 问题 | 证据 | 处理结果 |
| --- | --- | --- |
| 384-thread FWD GIN 中外部 device assert 调用干扰编译调度 | Qwen 的 HGMMA/ARRIVE/WAIT 静态计数由 40/40/40 恢复为 40/10/9 | 仅 NCCL device 头范围内把非 debug 的断言失败处理改为内联 trap，保留条件和 backend；debug 保留原诊断 |
| BWD stage 释放与编译器流水线分析冲突 | 从 48/48/48 恢复为 48/12/12；四个实际数学 K 内环未观察到 LDL/STL | 当前统一路径在 K64 指令组 wait0 后释放 stage，固定累加顺序不变 |
| 512-thread 配置存在真实 accumulator spill | 将完整 FC1 单独作为 device callee 后，本地和远端 FC1 热 K 回边仍各有 64 条 LDL.64 和 64 条 STL.64，8/8/8 的 WGMMA 分组；静态每次回边 1024 bytes/thread 的 local 读写 | 未进入 GPU 性能矩阵，保留 384-thread 配置；动态 math216 额度不能保证静态128上限下的编译分配无 spill |

384-thread 的 math216/control64 需求为 `256×216+128×64=63488` 个寄存器，名义初始池为 `384×168=64512`。host 还依据实际 kernel entry 的 `numRegs` 检查池容量及 `setmaxnreg.inc/dec` 方向，成功结果缓存。单看不超过每 SM 65536 不足以证明不会等待寄存器池。640-thread 的一个候选还超过了共享内存上限，因此资源合法性与性能必须独立准入。

最终 Qwen FWD 恢复为 40/10/9，Flash 为 136/34/33；两模型 BWD 为 48/12/12。最终所选二进制各六个 unified 入口已核对 SASS 指令和机器编码；这证明对应入口的代码来源，不能把历史运行时间直接改名为最终结果。[入口审计](../results/full_cta_all_worker/final_selection_audit/report.md)与 [512 FC1 callee 审计](../results/full_cta_all_worker/builds/core_qwen_512_fc1outline_v1/fc1_physical_callee_audit.json)保留具体位置。

## FWD 尚未解决的瓶颈

Qwen EP16/8K 的 core_v4 诊断 trace 显示，FC2 最后 tile 在约 2.8–3.0 ms 完成，最后有效 Return payload 到约 4.4–5.3 ms 才到齐；每 rank 此后到 Reduce 完成仅约 25–28 μs。Return packing 每 128 行平均约 87 μs，pack 完成到提交约 32 μs，超过一半的 Return packing 工作仍落在 FC2 完成之后。这个场景继续只提高 GEMM 吞吐，不能消除 Return 依赖。

PackStart→End 覆盖本地 metadata、TMA、fence/barrier；PackEnd→Put 还包含 pending 交接、FIFO、context 服务与 GIN put 内部开销。该 trace 没有失败 lease、credit 等待或未就绪计数，也没有区分 Assist 与固定通信 warp 的线程 ID，因此不能把全部间隔归为 NIC 带宽不足。诊断依据见 [Return 分析](../results/full_cta_all_worker/return_bottleneck_reassessment/report.md)。

Qwen EP8/64K 的独立 trace 则显示通信在约 21.55 ms 完成，Reduce 到约 22.33–22.78 ms 才结束；65536 tokens 中约 61769–62451 个仍由尾部 Reduce 处理。尝试每批发布完成计数、增加领取批量和额外 Reduce warp 均未带来可靠收益。该 trace 是单次带 instrumentation 的运行，不能拿其阶段时间减去正式矩阵均值解释精确回退量。[关键时点](../results/full_cta_all_worker/diagnostics/core_v4/ep8_t65536/critical_points.json)。

多轮试验还排除了增加 packing pending 槽、重复 submit 扫描、空 Assist 快路、尾部 GIN handoff、Return metadata 协作加载，以及 512 小 N 布局等未见收益的方案；这些实验代码没有合入生产路径。此前 slots4/slots6 改的是每个 GIN context 的 packing/packed 待提交条目，物理 ring 深度仍为每 channel 4 槽。本轮没有物理 ring 深度 4→8/16 的性能对照，不能用这些负例证明 4 个物理槽足够。当前的主要限制已从错误的寄存器分配/流水线退化，转向通信服务效率、输入就绪和尾部工作分配。现有证据不足以承诺再增加通信 warpgroup 就能解决。

## 稳定性与一般性验证

没有 pre-combine。FWD 每个 token 的归并保持固定 `combine_order`，不按到达顺序求和；BWD 的 dW tile 由唯一 owner 按固定 K 次序计算，不使用浮点 atomic split-K。

最终所选源码和二进制完成下列非计时验证；[完整摘要](../results/full_cta_all_worker/validation/final_v6/summary.json)也内嵌于机器记录。

| 验证 | 覆盖与结果 |
| --- | --- |
| Qwen、Flash FWD | EP8/16，T129/513，重复输出一致；与 main 的 129-token 输出/input/routes/gates 前缀一致，跨 batch、跨 EP 一致 |
| Qwen、Flash BWD | EP8/16，T129，dX/dGate/dW1/dW2 等全部保存产物重复一致；两模型共 480 个保存文件与 main 完全相同。此 fixture 用模型 H/I/K，专家数为 `max(16, 2×EP)`，不是256专家性能 fixture |
| 正式模型矩阵重复检查 | E256、EP8/16、8K/64K，当前全部16项、每rank共20轮；FWD完整输出及BWD全部梯度在GPU逐32-bit word检查一致，检查在计时外 |
| 通用尺寸 | H2560/I1280、EP8/16、top-k 1/6/8、T1/129/513、空专家、inference 不保留状态，共12项；1152份保存文件与 generic_v3 参考一致，包含 batch/跨EP前缀；本轮是回归比较，没有重复建立独立数值 oracle |
| CUDA Graph | 两模型 EP16、T129、top-k2、每rank两个专家；三种路由、buffer扩容/复用、六次 Notify+FWD+BWD 重放通过。FWD为索引hash及alias检查；BWD为dX/dGate/dW1/dW2全量memcmp |
| Host 配置与资源检查 | unified、FWD/BWD autotune 85项通过；最终独立审阅检查98份src/include指纹和两份header一致，定向资源/配置检查通过 |

这些验证不覆盖取消注入、Graph recapture 或每个合法调参组合。较大 batch 的重复一致不能代替跨 batch/跨 EP 不变性证明；各类覆盖范围按表分别记录。

## 频率与可复现记录

硬件为两节点、每节点8张132-SM L20X（SM90a），CUDA12.9、NCCL2.31.2。两节点设置相同1620 MHz上限和700 W限制。每200 ms记录SM频率、功耗、温度；统计只取最终矩阵窗口，不包含测试后的长时间空闲记录。

功耗≥300 W样本的SM频率中位数两节点均为1620 MHz；本地最小1485 MHz、P05=1605 MHz，远端最小1500 MHz、P05=1560 MHz。该功耗筛选只是负载近似，采样未与各kernel精确对齐，不能宣称所有kernel恒定1600+ MHz，也不能据此做频率归一化修正。原始时间一律保留。[频率统计与限制](../results/full_cta_all_worker/final_v6/clock_analysis.json)。汇总结束后两节点已执行 `nvidia-smi -rgc` 恢复默认频率管理。

最终源码快照为 `results/full_cta_all_worker/sources/final_v6`；manifest SHA256 为 `db00778ab46bd1e93120e018a85781e39dcd4291a7277b973cf3fd11335cba7a`。这是未提交工作区的冻结快照，不是新的 Git commit。性能与稳定性使用同一快照和所选组合头：

| 模型 | 构建记录 | 完整组合头 SHA256 |
| --- | --- | --- |
| Qwen | `final_qwen_limit3` | `758cb40f324a5bdc772b52f6329344d5d42e0aaae6aab7d49692c58ab6b29f84` |
| Flash | `final_flash_limit1` | `a81762e04c7b471ce02ae24a11b33fb8dd0bee0593401d79dfc3de7765f778cd` |

[导出的组合头和解析参数](../configs/full_cta_all_worker_20260914/profiles.json)与实际JIT头字节一致；旧生成注释摘要不包括前置覆盖值，应以完整header SHA和解析参数为准。[机器记录](full_cta_all_worker_20260914.json)内嵌逐项统计、原始samples、所选binary/header/source证据、验证摘要和频率统计；其中路径按仓库根目录解析。更完整的本地过程资料位于 [final_v6](../results/full_cta_all_worker/final_v6/comparison.md)，包括最终驱动、原始日志和所有参考测量链接。
