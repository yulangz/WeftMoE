# PF 跨机 pull / direct Return：2026-10-09

更新：[WG3 全 CTA 去 GEMM 通信消融](pf_wg3_communication_20261009.md)显示
扩大owner有稳定收益；尚未测其含GEMM版本，不替代下文真实MoE的架构对比。

补充：[去 GEMM 后的通信带宽实测](pf_communication_bandwidth_20261009.md)。
区分纯 GIN 数据面上限与保留实际 ring/credit、机内转发的完整通信路径；
测量代码位于独立 diagnostic 分支，不改变本分支的生产算子。

## 更新：两条独立探索与同协议交叉验证

先后使用两个全新上下文的 agent，从 `7c57005` 分出独立分支，串行独占双机。
两者分别冻结、完成验证后，主线程才交换通用优化，另外建立交叉验证分支。
**目前仍优先部分专用 warp，但不能用下面旧的起点差距否定 WG3 潜力。**

共同测量口径仍为本页的 Flash、EP16、SM1620、随机 tokens/routes、无训练缓存，
prime32/warm8/sample24；表中为 kernel p50，单位 ms。独立方案取两轮 p50 均值。

| 方向 | 起点复测 8K / 20K | 独立优化后 8K / 20K | 主要配置 |
|---|---:|---:|---|
| 部分专用 warp，384 threads | 18.069 / 43.046 | **12.719 / 27.888** | B4096，D32/R128，credit×2，ready 去重，提前 pack Get |
| WG3，512 threads | 20.103 / 47.982 | **14.909 / 33.580** | 192/64/64，B4096，D32/R64，预计算发布元数据，额外 pull |

两行的优化覆盖不同，**不是公平的 warp 架构消融**。因此给 WG3 补齐 Return
Q128、credit×2、连续同 `(rank, slot, generation)` ready acquire 去重；双方都用
不 pack 的 token Get、B4096、pending4。保留各自的 warp 划分/GEMM 和 WG3 的
publication/extra-pull 优化，并在两个 WG3 正式重复之间穿插复测部分专用 warp。

| 同协议交叉验证 | 8K kernel / eager | 20K kernel / eager |
|---|---:|---:|
| WG3，第 1 轮 | 13.272 / 13.436 | 29.732 / 30.046 |
| 部分专用 warp，穿插复测 | 13.170 / 13.418 | 28.155 / 28.741 |
| WG3，第 2 轮 | 13.386 / 13.678 | 29.519 / 29.904 |

WG3 kernel 均值 **13.329 / 29.626 ms**，比穿插复测慢约 **1.2% / 5.2%**。
两者 rank0 compute、GIN、notify 容量逐字节一致：compute+GIN **5.968 / 7.141
GiB**，另有 notify **16.4 / 40.8 MiB**。全部正式 workload SHA 与独立实验完全
一致。这个结果比独立最优表更接近实际方向差距，但仍不是纯线程数消融：WG3
使用 N128 atom 和不同寄存器配额，并保留额外 pull / planner publication 优化。

有收益的机制与边界：

- **Return 粒度和控制开销是主要收益来源。** Q16→Q64/Q128 减少反复 ready
  检查、issue/flush 和退休频率；credit×2 让后续窗口更少被 TX reservation
  阻塞。去重只在一次 packet 检查中复用相邻同 slot/generation 的成功 acquire，
  不跨轮询；仍保留 NIC completion、远端 credit 和 FC2 可见性发布。
- **M-padding 是真实额外 GEMM。** 每个 source-window/expert 分别向 M128 取整，
  因而窗口过碎会制造许多尾块。B2048→B4096 将 `extra / valid` 从约
  **23.4% / 21.8%** 降到 **14.3% / 10.8%**。这是减少，不是完全消除。
  更大 B 或跨节点 G16 虽继续减少 padding，却增加容量、推迟数据就绪/Return，
  两方向都测到负收益。下一步应考虑细粒度 tail 合并及独立 source credit，
  不能简单合并整个跨节点 window，也不能让局部计算等待整批远端数据。
- **WG3 的控制寄存器和空闲 warp 有利用空间。** 208/48/48→192/64/64，
  将发布目的地在 planner 中预计算，以及补上 116 条 local/RX pull warp 有收益。
  最终 shared 已到每 CTA opt-in 上限；不是增加 warp 就能无成本加深 pipeline。
- **Push 和 pack 都实际实现并消融。** 部分专用 warp 覆盖 token Get、提前 pack
  Get、ring pack Push、提前 pack Push；WG3 实现独立 pack/NIC owner 的有限环
  Push 流水。Push 本轮均未胜出。部分专用 warp 的提前 pack Get 只再省约
  3% / 1%，却多预留 **576 / 1344 MiB**；token Get 是合理的低显存选择。
- **没有证据把最终差距归于热 K spill。** 部分专用 warp 保持 4×N256 / 1 wait，
  WG3 保持 8×N128 / 1 wait，热 K 都无 LDL/STL。控制/GIN 路径仍有 local-memory
  访问；不能声称整个 kernel 无寄存器压力。wait-group1 的独立 WG3 消融为负。

两条独立分支均通过常规、热点连续回绕、单机及训练/BWD 回归；交叉库另通过
64/8K/20K、B128 热点连续回绕、EP8 和 EP16 训练/BWD 回归。只实测了两台机器。
这些是 inference 数据，没有新跑 Megatron；不能与历史 cached-training FWD
混成公平 A/B，也没有将候选直接合入生产默认配置。

保留的里程碑与完整证据：

- `experiment/pf-network-partial-20261009`，`82e502f`：
  `docs/partial_warps_20261009.md`、`benchmarks/partial_warps_20261009.json`。
- `experiment/pf-network-wg3-20261009`，`4726b66`：
  `docs/pf_wg3_20261009.md`、`benchmarks/pf_wg3_20261009.json`。
- `experiment/pf-network-aligned-20261009`：在 WG3 冻结结果上移植通用控制优化，
  源码提交 `8eacade`；交叉测试原始日志/manifest 在该 worktree 的 `build/aligned/`。
  该分支文末说明及 `benchmarks/pf_network_aligned_20261009.json` 保存交叉证据。

## 结论与测量口径

以下为最初两阶段的历史结果，不代替上面的独立优化与交叉验证。双机 EP16
的 8K/20K kernel 比本轮重测的旧网络 PF 快约 **2.04× / 2.15×**。
WG3 math208/local48/network48 能运行，但分别比 384 线程版慢
**10.8% / 11.5%**；这是未经独立优化的起点差距。不是 V3 或单机 MegaMoE 对比。

Flash BF16 H4096/I2048/E256/top-k6，两机各 8 张 L20X，SM 1620 MHz，
memory 3201 MHz。每 rank tokens 围绕 8192/20480 正态波动（std=10%），
每次路由/gate 重新随机，route bias std=0.5，seed=123。三个版本必须使用
同一 token/route hash；权重和持久 workspace 预备一次，没有训练缓存。
先完整 prime 32 步，再 warmup 8 / measured 24；各步取最慢 rank，再取 p50。
kernel 为主核设备时钟，eager 包含输入准备、调用与 synchronize。
下表列各版本首轮；新版本另外各复测一轮，均值与范围见 WG3 一节。

| 版本 | 8K kernel / eager（ms） | 20K kernel / eager（ms） |
|---|---:|---:|
| 冻结旧 PF 网络入口 | 37.017 / 37.248 | 92.137 / 92.475 |
| 新协议，384 线程部分专用 warp | 18.211 / 18.382 | 42.841 / 43.154 |
| 同协议，WG3 208/48/48（N256 拆为两条 N128） | 20.101 / 20.374 | 47.900 / 48.221 |

这里只能将收益归于**整套重构**，不能将 2× 归因于“借用 29 个 warp”一项。
旧版使用 L2/Q32，新版使用 L16/Q16；调度、数据面与 GEMM pipeline 也有变化。

## 第一阶段实现

- Dispatch 按命中节点去重，接收 rail 对实际 token 发单行 RDMA Get；
  没有源端 gather-pack，也不发零命中节点。每 owner 保留四个 pending
  completion，Get 完成后 release 发布 RX slot。三个步骤可以流水重叠：
  跨机 Get → 节点内 bulk pull → FC1/FC2。
- 控制面继续复用 source-window planner，精确计算 fanout/Return 行区间；
  网络 packet 不切碎 GEMM。计算沿用单机 M128/N256/K64、3 stages、cluster2
  multicast、连续 A/B stage、FC1 寄存器 16B epilogue、FC2 shared→16B。
- FC2 直接向对应 rail 的连续 Return TX 写结果，Q16 Put 直达源 GPU 的
  contribution，无 Return pack/scatter。S 读退休和 NIC 可读发布分开计数；
  TX 重用仍同时要求远端 slot ACK 与本地 NIC completion。
- 每 GPU 16 Get / 8 Put / 4 progress / 1 admission owner，占前 29 CTA 的
  一个 pull warp；132 CTA 剩余 367 个本地 pull warp，全部 math WG 保留。
  Progress 合并非阻塞 FC2 publication 与 Return RX/ACK，不能阻塞等 GEMM。
- Contribution 为整次调用预留，Return 排空后做 Reduce。RX/X/S/Return TX
  仍是有限 ring；末尾全局 rendezvous 保护下一次调用的 input/metadata 复用。
  本轮保留原训练/BWD 入口，不将 inference 正确性外推到训练。

一个实际修复：Return RX 完成只保证远端返回，不能保证同节点其它 GPU 的
FC2 已写完本 GPU。Reduce 前增加节点内写者 release/acquire 汇合；它不是
Dispatch 阶段的全节点完成屏障，不阻止前面的计算/通信重叠。

资源：新 network kernel REG168 / STACK256 / static shared1024，dynamic
shared226304 bytes。配额 math208/control80；热 K loop 无 LDL/STL，四条
N256 WGMMA 一次 wait。STACK256 涉及控制/GIN 调用，不能说整个 kernel 无
local-memory 访问。ComputeSlot 新增独立发布计数，从 64B 对齐扩到 128B。

rank0 持久空间（GiB，compute / GIN）：8K **2.507 / 1.117**；20K
**2.797 / 1.902**。旧版 GIN 为 0.476 / 0.486 GiB。增量主要是完整
contribution、注册 input、更多 lane；当前布局还保留未使用的 Dispatch TX
区，不声称空间已最小化。

## 验证

- EP16：1/64/256/8K/20K，各两份变长随机路由、全部 rank 对 PyTorch 通过。
  最大 relative L2：小规模 1.595e-3，8K/20K 9.217e-5。
- B128、bias std=8、8K/20K：有限 ring 回绕与热点路由通过，最大
  relative L2 7.904e-5。重复调用使用同一分配、递增 generation，无缓存旧路由。
- 正式 EP16 测量的旧/新 workload 完全一致；两机时钟采样分别 42/40 组，
  所有 GPU 的 SM 都为 1620 MHz。
- 单机 EP8 8K/20K 对照通过，trace 仍是 local PF kernel；性能回归另列。
- 资源/GIN Python 测试：17 passed，1 个已有失败：bare-build 测试仍期待
  BWD copy buffers=1，而本轮开始前 HEAD 已默认 2。未改无关测试。
- 仅有两机硬件；peer/lane 索引没有写死节点数，但 >2 节点尚未实测。

EP8 kernel p50：旧库 **6.669 / 16.556 ms**，新库 **6.612 / 16.396 ms**；
eager 为 6.883 / 16.756 → 6.765 / 16.644 ms，未见回退。这是回归检查，
不把约 1% 的变化归因于网络重构。逐样本 rank-max 时间和路由 SHA256 保存在
[`pf_network_20261009.json`](../benchmarks/pf_network_20261009.json)。

## 第二阶段：WG3 208/48/48

512 线程中，WG0/1 为 math208，WG2 为 producer + 原三条 pull（48），
WG3 为 network（48）。每 GPU 仍然只有 16 Get / 8 Put / 4 progress /
1 admission owner，与第一阶段相同；没有让 132 个 CTA 的全部新增 warp
同时发 RDMA。未分配 owner 的 WG3 warp 直接进入 CTA 汇合，不轮询网络队列。
本地 pull 从 367 恢复为 396 个。Get/Put 各有四个独立 completion handle，
放在新增的 1 KiB shared arena，不能和仍在使用的三个 pull staging 区别名。

原样 N256 编译失败（`e762a24`、CUDA 12.9），没有该版本的性能数字：

```text
C7602: Insufficient registers (128) ... Try to compile with register target of 154 or higher.
```

配额预算 `256×208 + 128×48 + 128×48 = 65536` 合法，但不能解除 ptxas
对 512 线程入口静态 REG128 的 N256 指令分配限制。只给网络 WG3 的 Mma
类型选择 N128 atom，把逻辑 N256 拆成两条指令，保持 M/N/K、accumulator
布局、每个输出的 K 累加次序和通信协议。这个差异意味着本轮**不是纯粹的
warp 数消融**；不能把全部回退定量归因于额外 WG、配额或指令拆分中的一项。

| 资源 / 机器码 | 384 线程 | WG3 |
|---|---:|---:|
| 入口 REG / STACK（bytes） | 168 / 256 | 128 / 464 |
| 动态 shared（bytes） | 226304 | 227328 |
| static shared（bytes） | 1024 | 1024 |
| 动态配额 | math208 / control80 | math208 / local48 / network48 |
| 每 K64 的 WGMMA / wait | 4×N256 / 1 | 8×N128 / 1 |
| 热 K loop 的 LDL/STL | 0 / 0 | 0 / 0 |

WG3 热循环没有 accumulator spill，也没有逐 WGMMA 注入 wait 的串行化。
STACK 更大、控制配额更低和额外指令是真实差异，但尚未独立归因；不能用
“GEMM spill 导致慢”解释这次结果。单机 local kernel 的 21072 个 64-bit
指令字在 shared/WG3 两个库中完全相同，WG3 的指令拆分没有扩散至单机。

两轮结果（单位 ms；每轮都是 24 步 rank-max 的 p50）：

| 版本 | 8K kernel 两轮 | 20K kernel 两轮 | 8K/20K kernel 均值 | 8K/20K eager 均值 |
|---|---|---|---|---|
| 部分专用 warp | 18.211 / 18.102 | 42.841 / 42.996 | **18.156 / 42.919** | **18.376 / 43.199** |
| WG3 208/48/48 | 20.101 / 20.136 | 47.900 / 47.846 | 20.118 / 47.873 | 20.343 / 48.174 |

WG3 对 PyTorch 的 1/64/256/8K/20K、每档两次验证通过；8K/20K 最大
relative L2 9.470e-5。B128/bias8 的热点验证通过，64 组 rank/sample 输出
SHA256 与部分专用 warp 版全部一致。两版正式测量的 workload hash、GM
workspace 相同；四轮新版本的全部时钟采样均为 SM1620 MHz。

只推荐 mode1；mode2 保留作为显式候选，不修改默认构建、其它 profile 或
训练/BWD 的入口选择。两机以上没有硬件验证，不作扩展性能结论。

## 复现

编译后使用打印出的 tensor extension；单机 profile 和默认构建不自动启用
此入口。`WEFTMOE_PF_NETWORK_WORKERS=1` 为推荐的 384 线程，`=2` 为 WG3 候选。

```bash
export NVCCFLAGS="-lineinfo -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I${PWD}/configs/pf_network_flash_20261009 -DWEFTMOE_PF_NETWORK_WORKERS=1"
python build_native.py lib --output build/pf_network_20261009/shared/libweftmoe.so --jobs 4
python tools/build_tensor_runtime.py --library build/pf_network_20261009/shared/libweftmoe.so
python benchmarks/run_pf_network.py --artifact build/pf_network_20261009/shared --label shared_performance -- \
  --backend pf --hidden 4096 --intermediate 2048 --experts 256 --top-k 6 \
  --tokens 8192 20480 --ring-tokens 2048 --prepared-weights \
  --profile-device --warmups 8 --samples 24
```

正确性：将最后一行换成 `--verify --warmups 0 --samples 2`；压力测试另改
`--ring-tokens 128 --route-bias-std 8`。换成模式 2 构建 WG3，保持其它参数不变。
环境的 CUDA/NCCL/CUTLASS 依赖与原 build_native 相同。

原始日志在 `build/pf_network_20261009/`，正式前缀
`legacy_performance_b2048_bias0.5`、`shared_performance_b2048_bias0.5_t8192-20480`，
对应 `.clocks.json` 保存时钟采样。冻结旧库为
`build/pf_megamoe_20261008/cache_rquad/f47b1119685a/libweftmoe.so`，SHA256
`eacc6335a965d0a815ca7304021f1ac838d3603cfd1e36ca39cb0cab6836811b`；新 shared 库为
`build/pf_network_20261009/shared/libweftmoe.so`，SHA256
`c9c6bed29287c4f873cafd7b628b4275cdeb7abd22f8ed821baa2c9948d63b7a`。
WG3 产物为 `build/pf_network_20261009/wg3/libweftmoe.so`，SHA256
`47126e5921d083c3cddcd37a34ac061e8beccff17733c69c92b0b33102a80bb2`。
复测前缀 `shared_repeat_performance` / `wg3_repeat_performance`，WG3 首轮
为 `wg3_performance_b2048_bias0.5_t8192-20480`。原样 N256 编译错误日志是
`wg3_native_n256.build.log`。第一阶段提交 `e762a24`；后续提交保存 WG3
兼容指令与完整两版结果。
