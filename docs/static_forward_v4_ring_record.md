# Static Forward V4 ring 实施与优化记录

日期：2026-09-23。设计依据：[有界流水设计](static_forward_v4_design.md)。
本记录区分代码落地、独立协议验证和端到端启用，不把准备工作记为性能收益。

## 冻结基线

设计基线 commit：`f2acb2c`。以下文件保留，不覆盖：

| 文件 | SHA256 |
|---|---|
| `build/static_v4_prefill_safe/test_cuda_notify` | `28a0e56e8a306f1fe7e7dc2e06b45ac1d6804f0e8cf3b85076b83eb65e0fd419` |
| `build/static_inference_v3_20260923/test_cuda_notify` | `cc0471af98d0fcbcfa52b9c9669ae2e20fb9b8077aa36e483a11a4be6331a699` |
| `configs/static_inference_prefill_20260923/weftmoe_forward_tuning_generated.hpp` | `1bfcc74eac080b907111e313e5c62af178062726ba5c8725c147a761897046ed` |
| `configs/static_inference_v3_20260923/weftmoe_forward_tuning_generated.hpp` | `3718e402c247a4c921de1e907d38ea2873729dd12fd417bb1a1c2e0cc88eb424` |

对照性能沿用设计第 0 节和 prefill 记录，不视为本轮交错重测。主矩阵固定
单机 EP8（1×8）、双机 EP16（2×8），T/rank=1/64/8192/20480，H=2048、I=512、
E=64、K=4。两机检查 SM/graphics 时钟均为 1620 MHz，显存 3201 MHz。

## M1a：窗口、逐 slot 合同与 source GPU plan

Commit：`a6a8b3f`。

### 实现

- `static_v4/plan.hpp`：host/device 共用窗口编号、source group、容量上界、
  完整 token 分包及 lane 划分、wire record。重复 expert 的不同 top-k slot 不去重。
- `static_v4/ring_layout.hpp`：Dispatch/Return 的四个独立 TX/RX 区域、peer/lane/slot
  地址、generation、累积 retirement target 和容量 cursor。TX reuse 同时检查远端
  credit 与本地 56-bit completion；没有任务领取 CAS。
- `static_v4/source_plan.cuh`：固定 CTA 构建 source-local token window，生成实际
  expert counts/prefix、route 顺序及 inverse ordinal、每个命中节点的去重 token
  列表及 fanout 数。另一个固定 worker pass 生成各 node/lane 的精确 packet prefix。
  同节点不生成网络包，未命中节点的窗口不增加 packet sequence。
- 原主路径几何计算复用 `ceil_div`/`block_rows`。新规划器暂未接入 Forward；
  **canonical X/S、contribution 和网络区仍是 full arena，未启用 ring。**

source 计数/散射阶段的 shared atomic 只构建不可变 plan，不用于数据面动态领任务。
当前生成的是 inference ordinal；StableTape 仍需稳定排序。非法 expert 报错与显式
masked route 分开，后续主路径必须在发布 payload 前集体处理 plan 错误。

### 验证

- host 回归：18 项通过（新增 3 项聚合检查），包含 1/2/3 节点窗口枚举、
  变长/空 source、热点、重复与 masked route、packet 覆盖、乱序 slot 完成、
  不同代 expected_count、双 credit、四区不重叠。地址检查还覆盖 70 节点。
- GPU source planner：31 个组合通过；nodes=1/2/3、T=0/1/64/8192/20480，
  balanced/hotspot 与非法 id。独立 host reference 验证 counts、route inverse、
  node 去重、fanout 和精确 terminal sequence。
- 同一 GPU 测试 `compute-sanitizer --tool racecheck`：0 errors / 0 warnings。
- 重建主路径单机 EP8，T=1/64/8192/20480 的 32 个 rank/shape 输出 hash 全部与
  冻结 V4 一致；每 rank 的 graph 仍为一个 kernel。此轮每组仅 warmup=10、
  measured=2、batch=2，用于回归，不用于报告性能。主路径 binary SHA256：
  `edabded74de5a44270a34b9f9909235c69c9a0db5e9cf8613dd64bbed91936fa`。
- 三节点测试仅验证单 GPU 上的三节点 **规划**，不代表三机 RDMA/进度验证。

构建/独立检查命令：

```bash
python3 build_native.py host-test --jobs 8 --output build/static_v4_ring/test_notify_dispatch
build/static_v4_ring/test_notify_dispatch
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -gencode arch=compute_90a,code=sm_90a \
  -Isrc tests/cuda/static_forward_plan.cu -o build/static_v4_ring/test_source_plan
env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 \
  /usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  build/static_v4_ring/test_source_plan
env NVCCFLAGS="-I$PWD/configs/static_inference_prefill_20260923 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1" \
  python3 build_native.py cuda-test --jobs 8 --output build/static_v4_ring/test_cuda_notify
```

本阶段没有端到端优化收益；不填新的 V3/V4 性能对比。后续需要完成 destination
窗口几何、不可变 packet→consumer 目录和分布式 plan 交换/准入，再启用数据面 ring。

## M1b：不可变消费者目录与 destination-window 几何

### 实现

- `static_v4/packet_plan.cuh`：将各 channel 的实际 packet 数压成紧凑目录；每个
  packet 持有全局 window id、source token 列表区间、token/fanout 数，以及
  每个目标 GPU 的有效 fanout 行数。源节点 channel 为空。
- packet 与 consumer 目录存在 immutable plan 中，**不放在 RX ring slot 内**。
  非消费者可以跳过包而不读 RX；实际消费者必须先等本代 ready，完成指定的行
  及所需 metadata 搬运后记账。接收目录的分发和这些搬运动作仍待接入。
- packet 序号优先映射到 channel/window，用 upper-bound 查找跳过空窗口，
  不按 expert 顺序等待 RX。目录只为实际包占空间；包数上界是有效 route 数，
  没有为每个 remote node 再复制一份最坏 route 容量。
- `static_v4/window_geometry.cuh`：对已交换的窗口 source counts 生成 expert 行数、
  Mblock prefix、有效行数和 source round-robin 逆映射。逻辑行与物理 ring 地址
  分离；C expected 就是该 Mblock 的有效行数，不取 fragment 个数。

### 验证与边界

复用同一个 GPU 集成检查，不为每个字段新增测试入口：31 个 source/packet 组合
加 6 个 destination 组合通过。覆盖空 channel/window、无 mask 的全热点路由、
显式 mask、重复 expert、tail packet/Mblock、三节点最后一个不足 G 的 source group。
consumer counts 和 round-robin 映射由独立 host 枚举检查。
最终 GPU 检查 binary SHA256：
`70aac81dc883b098cca0b4d3b3dec5408fa5f83af9e63db34abb53ce50210ec6`。
`compute-sanitizer --tool memcheck`：0 errors；`--tool racecheck`：
0 errors / 0 warnings。

主路径追加双机 EP16 的 T=1/64/8192/20480 回归，64 个 rank/shape 输出 hash
均与冻结 V4 一致；与 M1a 的单机检查一样使用 warmup=10、measured=2、batch=2，
不作性能归因。两机 binary SHA256 相同，仍为 M1a 记录的 `edabded7…`。
验证前后两机的 16 张 GPU 均观察到 SM/graphics 1620 MHz。收尾采样本机显存
3201 MHz，远端空闲时回落到 2201 MHz；本轮没有锁定显存时钟，也没有把这组
correctness-only 运行用作性能数据。

回归使用下列环境，对冻结 V4 与本阶段 `test_cuda_notify` 各跑一次四规模，
比较 JSON 中全部 `per_rank_output_hash`，不比较短采样延迟：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
export NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0
export WEFTMOE_BENCH_EXPERTS=64 WEFTMOE_P2P_TOP_K=4
export WEFTMOE_BENCH_HIDDEN=2048 WEFTMOE_BENCH_INTERMEDIATE=512
export WEFTMOE_FORWARD_BENCH_ROUNDS=12 WEFTMOE_FORWARD_GRAPH_BATCH=2
export WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY=1 WEFTMOE_BENCH_SYNC_RANKS=1
unset WEFTMOE_DISABLE_STATIC_FORWARD_V4
# WEFTMOE_P2P_BENCH_TOKENS 分别设为 1、64、8192、20480。
# ring_binary 分别指向两个冻结/候选 binary 的绝对路径。
python3 tests/run_local_activation_p2p.py --binary "$ring_binary" \
  --world 8 --top-k 4 --forward-benchmark --rounds 12 --timeout 240
python3 tests/run_multinode_nccl.py --binary "$ring_binary" \
  --remote-binary "$ring_binary" --remote root@10.10.19.233 --port 2222 \
  --gpus-per-node 8 --scenario gin_forward_bench --socket-ifname eth0 \
  --enable-ib --timeout 240
```

此阶段仍未改 Forward 的数据流。M1 尚缺跨 rank count/目录交换、source 与
destination 全局映射、真正的整窗口准入及进度闭环；M2–M6 保持未完成。
没有新增性能数据，不能将协议模块的独立通过等同于 ring Forward 已验证。

## M2：单机有界数据面首版（正确性里程碑，非性能结论）

Commit：`cb14266`。

### 实现及生命周期

- `compute_ring.hpp` / `ring_access.cuh`：X/S/C slot 的 generation、累积
  X-full/FC1-done/FC2-done/returned 与本代 targets；source contribution 独立两窗口。
  不在 graph replay 中清零；结束后推进 block/source sequence base，profile 切换时
  FC1/FC2 expected 取本次配置，不用不适用的固定 generation×F 公式。
- `ring_plan.cuh`：固定 CTA 构建 source window；PlanReady 后读取实际 peer counts，
  生成窗口内 expert/Mblock 几何。热点不使用旧 `max_destination_rows` 截断。
  逻辑 reverse route metadata 独立于会覆盖的 X，输入仍有 invocation staging。
- `ring_admission.cuh`：CTA 0 的独立控制 warp 非阻塞推进 source 与 destination
  两本容量账。整窗口检查全部所需 X/S/C 后发布准入，无动态任务领取或 CAS。
- `ring_transport.cuh`：每 CTA 的三个通信 warp 分别 pull、Return、Reduce；
  CTA 0 的第三 warp 改为准入。pull 复制完成后累加 X-full；FC1/FC2 复用现有
  multicast TMA/WGMMA/epilogue，按窗口静态 residue 枚举。Return 随完整 C 行
  就绪而复制并发布 route-ready；Reduce 仍按 `(expert,slot)` 固定 FP32 顺序。
- 只保留控制面 PlanReady 和 invocation 末尾 rank barrier；不在每个窗口加 barrier。
  C 在完整行已复制后退休，contribution 在该窗口全部 token 已 Reduce 后退休。
- 首版三个计算 plane 等深、同 slot 映射，容量复用保守地等待 X/S/C 三者全退休。
  它们的计数独立，但尚未实现早释放 plane 的独立分配收益。source contribution
  与它们独立，准入角色不被 Return/pull 的阻塞等待占住。

开关为 `ActivationTransportPoolConfig::static_forward_window_tokens`，原生与 Python
实验读取 `WEFTMOE_STATIC_RING_TOKENS`。0 保持原路径；当前 ring 明确只支持单机
inference，不默默在多机/训练上退回旧协议。完整 contribution snapshot 的诊断
接口不适用于 transient ring，明确拒绝；最终输出下载不受影响。

### 正确性与检查

- 主矩阵 EP8 T=1/64/8192/20480，32 个 rank/shape hash 与冻结 V4 全部一致；
  每 rank graph 仍为一个 kernel，无 host 逐窗口操作。
- 现有 `torchrun_static_inference.py --tokens 1 64 8192 20480 --edge-cases` 通过：
  eager/graph、混合 64/8192 profile、不等长/空 source、重复 route、双层共享
  workspace、改变输入/routes/gates 的 graph replay、交错 eager/graph，以及
  maxT 全路由到同一 expert 的 eager/graph 和其后 workspace 复用。
  ring 模式将原来预期 overflow 的测试改成必须输出全部 token，并与 PyTorch 对照。
- 现有 host 检查 18/18；GPU plan 检查 31 source/packet + 6 destination 通过。
- B=16、T=64、EP8，11 轮（10 warmup + 1 measured）、batch=1 的生产 graph
  分别通过 memcheck 和 racecheck。NCCL init 的架构探测会产生
  `cudaErrorNoKernelImageForDevice` API 报告，故明确使用
  `--sanitizer-report-api-errors no`，仍开启设备内存/共享内存竞争检查。
  未关闭 API 报告的首轮 exit=1，不记为检查通过；检查不证明跨 GPU 协议自动正确。

首版性能/正确性 binary SHA256：
`5be1fa5c02e4bfb7c08b97ce5e11265a6a83751277caa09ea0e35cb8f1994c6f`，
位于 `build/static_v4_ring_m2/test_cuda_notify`；配置仍为 prefill 配置，B=256
（实际 pool maxT 小于 B 时取 maxT），X/S/C 各 `2*D_window`，contribution 深度 2。
PyTorch 对照的 library SHA256：
`f3f61914e78e21dde3c162e946f12c9b644788abf9ded9323e9a2ee76916a1af`。

### 首轮性能：退化需要继续优化

全部核心实测 1620 MHz、显存 3201 MHz；warmup=10、measured=20、batch=20，
graph CUDA-event rank-max p50，单位 us。计时不并发编译。

| 单机 EP8 tokens/rank | 重测 V3 | 重测冻结 V4 | ring 首版 |
|---:|---:|---:|---:|
| 1 | 300.384 | 63.9728 | 116.155 |
| 64 | 367.747 | 73.9040 | 267.486 |
| 8192 | 1488.19 | 1503.53 | 5265.20 |
| 20480 | 3288.22 | 4575.22 | 13042.8 |

V3 必须显式设置 `WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`；冻结 V3 **配置 binary**
本身也包含 V4 entry，只换 binary 而不设开关的测量不是 V3。本轮先识别并排除了
这类测量，重新单独跑动态 V3，graph 为 2 kernel/rank。旧 V4 与 ring 按每个 T
依次测量，尚未反向复测；此表是首版诊断，不是最终统计验收。

原始输出：`build/static_v4_ring_m2/ep8_v3_dynamic_t*.log`、
`ep8_static_v4_prefill_safe_t*.log`、`ep8_static_v4_ring_m2_t*.log`。
V3 T=1 的 setup 与独立 planner 检查短暂相邻，后续反向复测应独占 GPU 重验。

相对旧 V4，首版多了完整 C 的本地落地/读回、细粒度 Return/ready，减少了可同时
执行 pull 的 warp；另有按窗口重复的 peer metadata 读取和固定从 cluster 0 开始
的 residue。它们是待消融因素，不能凭此表分摊原因。目前不能宣称 ring 带来性能
收益或全面优于 V3，保持 opt-in。下一步分别测 immutable metadata 本地化、
静态 residue 旋转、等待策略与窗口大小，再接入多机和独立 GEMM 消融。

### M2 收尾复测与 workspace

加入 pool 字节数/所选路径及有效阶段 span 的 benchmark 输出，删除重复的 Return
system fence，并禁止用 transient ring 调用完整 arena snapshot/独立 transport
诊断接口。收尾 binary：`build/static_v4_ring_m2_checked/test_cuda_notify`，SHA256
`00175769123d06a790d5a116970e80b6a1eacd50dddf4e3f77db70cf5f099050`。
以 T=20480→8192→64→1、每个 T 先 ring 后 arena 的反向顺序复测，同样 10/20/20。
四规模全部输出 hash 再次与冻结 V4 一致。

| tokens/rank | 同 binary arena p50/us | ring p50/us | arena pool/MiB | ring pool/MiB |
|---:|---:|---:|---:|---:|
| 1 | 64.0896 | 115.403 | 5.00 | 25.12 |
| 64 | 73.8992 | 264.506 | 17.64 | 61.72 |
| 8192 | 1506.20 | 5269.15 | 1649.42 | 287.95 |
| 20480 | 4604.35 | 13061.1 | 4116.36 | 457.53 |

内存列为实际 `transport_pool.usage().compute_device_bytes`，每 rank 一 workspace
slot；该 pool 不在 main 内分配，因此也是该固定 pool 的分配峰值。它**不是整个
进程/模型的峰值**，不包含权重、外部输入输出、Notify 与 PyTorch reference。
H2048/I512/K4、B256 下 X/S/C+contribution 数据区合计固定 170 MiB；总 pool 仍
随输入、immutable plan 和尚未删除的旧 metadata 增长。短序列 pool 的两最坏窗口
及独立 C 区预留反而较大；不能声称所有长度都减少显存。

新增 stage span 仅用于定位：20K ring 的 plan 约 184 us，FC1/FC2 span 均约
12.75 ms；8K 分别约 89 us 和 5.08 ms。它们重叠且包含等待，不能相加，也不是
纯 GEMM 时间。长文主要退化不能归因于百微秒级 plan 本身，继续消融数据面。

## M2 优化实验（进行中）

### A：窗口间旋转静态 residue

保持全局 task 编号对 cluster 数取余，不再每个窗口都从 cluster 0 开始。
FC1/FC2 分别按各自的实际 tile/unit prefix 计算首个 residue；没有共享 cursor、
steal 或 CAS。除此之外与 M2 收尾代码相同。

候选 binary `build/static_v4_ring_rotate/test_cuda_notify`，SHA256
`683a39308aa1dc1af38d75b97579087de755178ff89b31317c6f035922f62b22`。
EP8、B256、1620 MHz、10/20/20，四规模 32 个 hash 与 M2 一致。

| tokens/rank | M2 收尾/us | 旋转 residue/us | 延迟变化 |
|---:|---:|---:|---:|
| 1 | 115.403 | 115.906 | +0.44% |
| 64 | 264.506 | 266.864 | +0.89% |
| 8192 | 5269.15 | 5217.41 | -0.98% |
| 20480 | 13061.1 | 12929.4 | -1.01% |

目前只是约 1% 的小变化，未反向交错确认，不作为显著优化收益。
均衡窗口中的 compute 覆盖虽更均匀，但数据面吞吐仍待定位。

### B：本地 immutable row 描述替代逐行远端 metadata 查询

在 A 上，将 source round-robin 映射、route 和 gate 一次性冻结到本地 `RingRow`，
pull 只读本地行描述并搬运 X。gate 使用原有结构 padding，描述仍为 16 bytes，
不增加 row directory 容量。Return 也只读该 immutable metadata。

候选 `build/static_v4_ring_planrows/test_cuda_notify`，SHA256
`06bd28f82a7bb78b5a026837b2947faf0f62ce8322254aa8101933b1e6e98f00`。
同样 EP8/B256/1620 MHz/10/20/20，四规模 hash 全部保持一致。

| tokens/rank | A/us | B/us | 延迟变化 | B plan span/us |
|---:|---:|---:|---:|---:|
| 1 | 115.906 | 123.939 | +6.93% | 48.32 |
| 64 | 266.864 | 242.914 | -8.97% | 92.58 |
| 8192 | 5217.41 | 3421.16 | -34.43% | 154.30 |
| 20480 | 12929.4 | 8330.07 | -35.57% | 252.80 |

这组改动减少数据面的串行 peer 读取，但提高了 planning 开销，短序列 1 token
反而更慢。B 的 metadata pass 仍每窗口只用一个 CTA，下一步单独改为固定
`(window,expert)` CTA 分区，避免短序列串行遍历全部 expert。本表尚未反向复测，
不作为最终性能验收；距 V3 长序列仍有明显差距。

### C：并行构建 `(window,expert)` 行描述

在 B 上只修改 metadata pass 的静态 CTA 分区，不改变 payload 遍历或 ring 同步。
候选 `build/static_v4_ring_planparallel/test_cuda_notify`，SHA256
`e8211207d261b1439a9edcdbc360ec9b91ac979a3e9f8b7c52f4c7b9d849fb72`。
同条件四规模 hash 全部一致。

| tokens/rank | B/us | C/us | 延迟变化 | C plan span/us |
|---:|---:|---:|---:|---:|
| 1 | 123.939 | 114.920 | -7.28% | 39.10 |
| 64 | 242.914 | 200.846 | -17.32% | 49.34 |
| 8192 | 3421.16 | 3377.29 | -1.28% | 109.54 |
| 20480 | 8330.07 | 8315.80 | -0.17% | 239.94 |

短序列改善来自 metadata pass 并行度提高；20K 总耗时变化极小，继续处理数据面。
接下来的候选将 source epoch 缓存在本地 plan，并把 source contribution reservation
的检查从每行 pull 提升到 destination 的整窗口准入；仅检查该窗口实际命中的 sources。
准入 owner 仍非阻塞地同时推进 source 与 destination，不变成 rank collective。

### D：整窗口导入 contribution credit（失败，已撤回）

候选同时缓存 source token/base epoch，并由 admission owner 检查本窗口实际命中
sources 的 contribution generation，随后发布 destination admission；pull 不再
逐行检查 contribution reservation。slot 计数协议未删除，但准入耦合更强。

`build/static_v4_ring_admit/test_cuda_notify`，SHA256
`37f5fe96653ae452139413a576e32ff7f0591b3eee6513e16cb223b39fca2a56`。
四规模 hash 一致，但同条件 p50 分别为 139.362/217.690/3729.83/9195.75 us，
相对 C 慢 21.27%/8.39%/10.44%/10.58%。plan span 也增加约 10–13 us：
source base 的额外 peer 读取被放进了 CTA 0 的串行扫描。

这一组合不保留。恢复逐行 reservation 检查；后续只保留 source epoch 本地
snapshot，并让 control lanes 并行读取 peer state，单独测量。不能把 D 的退化
全归因于 credit 粒度，因为它同时改变了 epoch 获取方式。

### E：并行 snapshot source epoch（保留逐行 credit）

回到 C 的逐行 reservation 等待。PlanReady 后由 control lanes 并行 snapshot
source tokens/base，错误码做 warp/block OR；CTA 0 只扫描本地 snapshot 计算窗口数。
几何规划、pull/Return 使用本地 epoch，不再逐行读取 peer 的 invocation state。

候选 `build/static_v4_ring_epochs/test_cuda_notify`，SHA256
`10726595c71ba44db943351893afba98434732c273a7c1a8ef18582634a308fb`。
四规模 hash 一致，同条件 p50=112.614/195.434/3312.25/8139.85 us；相对 C 改善
2.01%/2.69%/1.93%/2.12%。这只是小幅收益，待组合版本反向复测。

另外用已有 `WEFTMOE_FORWARD_DIAGNOSTIC_ROUTE_SCOPE=self` 将 route 改成只命中
source 自己的 GPU，保留每个 expert 的行数和 GEMM 形状，8K/20K 为
2817.51/6916.74 us。**该诊断改变路由语义，不是等输出性能对照，也不是用户主矩阵。**
它仍保留约 85% 的延迟，说明不能把剩余差距全部归因于 NVLink；后续用 compute
timeline 区分 GEMM 活跃区间和等待供给的空隙，并继续消融 copy issue 粒度。

E 的 library 已通过 PyTorch 四规模及全部现有 edge cases（含 max-token 热点不丢
token、graph 输入变化和 post-hotspot 复用），日志
`build/static_v4_ring_epochs/torch_edges.log`。B16/T64/EP8 的 memcheck 与 racecheck
均 exit 0；沿用 `--sanitizer-report-api-errors no` 排除 NCCL 初始化架构探测错误，
设备内存及同步检查仍开启。

额外编译 timeline 版本，以 eager 8K/EP8/B256 运行（不混入正式 graph 成绩）。
rank 0 的 23973 个事件无溢出：按 132 个 CTA 平均，compute selection 等待总量
2493.14 us；FC1 math/epilogue 分别 267.86/106.45 us，FC2 分别
114.38/55.18 us。这些是 CTA 活跃时间的平均，不是可串行相加的关键路径分解；
但足以确认主要空隙在任务供给，而非 GEMM 指令执行本身。该版本 Return 起点
telemetry 尚未接入，日志中的 Return span/FC2-Return overlap 无效，明确排除。

### F：两个 copy issuer 与容量消融（双 issuer 已撤回）

先在 E binary 上单独改变 B，8K/EP8/1620 MHz/10/20/20：
B128/256/512/1024 分别 5601.93/3312.25/2886.87/3273.16 us。
更大窗口不是单调改善；B128 还会提高 M128 padding 比例。

随后保持 B256 和两窗口 contribution，每个 pull/Return warp 改为两条 issuer
lane，各自占有一个 TMA staging/barrier，共四份 staging，放入已有 communication
SMEM，不增加 kernel 的共享内存预算。两 lane 按窗口内行号静态划分，并在窗口末
`syncwarp(3)`，避免一条 lane 等未来窗口容量而阻止另一条释放前序窗口。
同时补齐 Return 起点和实际 peer-byte telemetry（每 worker 收尾汇总一次）。

`build/static_v4_ring_copy2/test_cuda_notify`，SHA256
`b319dd38fbc8d444330a4665a90cf6f77ae91352b01104713059f920723a4247`。
四规模 hash 与 E 一致，同条件 p50=116.181/170.034/3115.03/7702.32 us，
相对 E 为 +3.17%/-13.00%/-5.95%/-5.38%。长文改善有限，继续独立测试
contribution 窗口容量，不能把额外 buffer 容量带来的效果归给 copy issuer。

F 的 memcheck exit 0，但 racecheck 在 pull/Return 的 TMA G2S 指令处报告 WAW
hazards；**尚未通过验证，不接受为最终优化**。正在用细粒度 hazard 报告及 build
目录内的最小双 issuer 复制复现区分 staging 复用问题与发射方式问题，测试代码
不加入项目单测套件。正式性能表不能采用未通过门禁的 F 成绩作为最终结果。

### G：单 issuer 双 staging

双 lane 最小复制程序可复现同一 WAW，逐行 syncwarp 和显式 lane 发射分支均未
消除；不据此宣称工具误报，F 实现撤回。改由每 warp 的单 lane 使用现有
`RowPrefetchPipeline` 同时预取两行。批次不跨窗口，所有 store 完成后才发布这
两行的 ready/retirement，奇数尾部显式 drain。pull/Return 合并为一个编译期方向
specialization，避免维护两份相同搬运循环。

新增 source contribution 窗口容量参数，与 destination X/S/C 容量独立。
`WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS` 仅供 runtime/benchmark 配置这个容量，
默认仍为 2；pool 兼容检查纳入它。不是动态任务队列或不受约束的 full arena。
G 的 B16/T64/EP8 多次 graph 复用，memcheck/racecheck 均 exit 0（同前述 API
报告选项）。后续对不同容量逐一记录性能和 pool bytes。

G binary SHA256：`a79113bbf3d39a6604aa8cf012520af635885da95b2560cf99d53115ca665a9a`。
8K、B256、EP8、1620 MHz、10/20/20 的 contribution 深度消融如下；首个 D2
运行与 tensor extension 编译有短暂重叠，标为暂测，正式验收必须复测。

| contribution 窗口数 | p50/us | pool MiB |
|---:|---:|---:|
| 1 | 5761.24 | 283.94 |
| 2（暂测） | 3457.63 | 287.95 |
| 4 | 3091.32 | 295.96 |
| 8 | 3059.86 | 311.99 |
| 16 | 3067.23 | 344.06 |

所有 hash 与 E 一致。G/D4 library 的 PyTorch 四规模及全部 edge cases exit 0，
host 18/18 通过。D4→D8 的收益约 1%，尚不值得据此固定为更大容量。

### H：relaxed poll + 成功后 acquire + nanosleep（失败，撤回）

仅改变 transport 的 generation/ready 等待：失败时 relaxed load + 64 ns
nanosleep，命中目标后 acquire fence；compute/admission 不变。8K/B256 的
D4/D2 变为 3502.52/3880.89 us，相对 G 均退化约 12%–13%。组合效果不保留，
不能将退化单独归于 relaxed load 或 sleep。后续回到 G 的等待方式。

### I：将大部分 Reduce warps 分给搬运（失败，撤回）

回到 G 后，把 Reduce 固定为 32 个 warp，剩余第三个 comm warp 静态分配给
pull/Return。132 CTA 下分别有 182/181 个复制 issuer；6 个 staging 比原方案
多约 8 KiB/CTA，host launch 按 ring specialization 正确增加动态 SMEM。

同样 EP8/B256/D4/1620 MHz/10/20/20，四规模全部 hash 一致，但 p50 为
129.478/198.486/5613.32/14801.3 us，8K 比 G/D4 慢 81.6%。撤回角色和额外
SMEM 改动。结果说明不能假定现有 Reduce 用少数 warp 就足够；继续检查其
实际 payload/记账开销，而不是继续减少 Reduce workers。

### J：静态 Reduce 不再逐 token 维护动态完成计数

恢复 G 的全部 131 个 Reduce warp。数值 helper 增加编译期 `TrackCompletion`，
原调用者默认保持不变；ring 不消费动态 token completion counter，省去每 token
的该计数及全局时间戳 min/max，仅由每个固定 worker 在首/尾记录时间戳。
contribution 的 consumed 仍按 token 发布，不改变容量协议和 FP32 固定归约顺序。

binary `build/static_v4_ring_reduce/test_cuda_notify` SHA256
`73bd0a2ba0dba21d00b21bbf2451ce181b36c469c2b21f899786248696a95af9`。
EP8/B256/D4/1620 MHz/10/20/20，四规模 hash 一致，p50 为
128.869/205.992/3088.12/7550.21 us。8K 与 G/D4 的 3091.32 us 基本相同，
**不能声称性能改善**。保留它作为静态路径记账清理，继续用细化 timeline 分析
搬运批次与 Reduce payload，不再从 span 直接猜测耗时来源。

J 的 B16/T64/EP8 memcheck/racecheck 均 exit 0。细化的 eager timeline（8K/D4，
非正式性能分数）rank 0 共 105893 个事件，无溢出：每两行 pull 批次 p50 9.15 us，
Return 批次 9.12 us，单 token Reduce payload p50 14.69 us。按 worker 平均的
对应活跃时间为 1203.90/1150.17/910.10 us；compute selection 等待仍约
2422.48 us/CTA。这些区间并发，不能相加当成关键路径。Reduce 数值搬运本身
并不便宜，后续考虑地址复用/加载并行度，并继续检查每行远端 reservation 读取。

### K：source reservation credit 导入到 destination 本地（进行中）

source owner 预留 contribution 窗口后，将累计的 exclusive reservation head
发布到各节点内 destination 的独立本地 word。pull 只 acquire 本 GPU 的缓存，
不再为每行读取远端 contribution generation。每 source/每窗口一次 credit
publication 替代每 route 的 peer control load；不增加窗口 collective，不把
source 与 destination 准入耦合成 D 的整窗口检查。

head 跨调用累计，源 contribution slot 的 generation/retire target 仍是容量真值。
缓存可以比 reader 的窗口更靠前：若该 reader 的 route 尚未 pull/Return，所属
窗口不可能已经完成 Reduce 并退休，所以 `head >= sequence+1` 不会允许覆盖
仍在用的 contribution。这是现阶段单机 credit 导入，不能用普通 peer store
代替未来跨 NIC 的 GIN 可见性/准入协议。

候选 `build/static_v4_ring_credit/test_cuda_notify`，SHA256
`513bd080ea3921cf06f9bb38b1ebd7fe928bc95d7189682add61fb0f98d12686`。
8K/EP8/B256/D4/1620 MHz/10/20/20 为 3056.75 us，8 个 hash 与 J 一致。
相对 J 的 3088.12 us 仅约 1.02% 改善；尚未反向复测，不能视为显著收益。
J library 的四规模及全部 PyTorch edge cases 已通过（`torch_edges_d4.log`）。

### L：冻结物理 slot 与窗口源地址（验证中）

J 的逐窗口 timeline 中，稳定期约每 100 us 推进一个 B256 窗口；每 worker
实际 copy 批次的时间不足以解释全部间隔。审查热路径发现每行反复做 runtime
64-bit 除法/取余来计算 block、slot、generation 和 source contribution 地址。

在 K 上，plan 为每个 Mblock 冻结物理 slot/generation，为每个 `(window,source)`
冻结 contribution sequence/generation/row base/slot；`RingRow` 在相同 16 bytes
中保存 uint32 route 和预计算 token（已有 pool 边界要求 route count <= UINT32）。
搬运热循环使用编译期 M128，避免 runtime 行号除法。Reduce 同样复用窗口地址，
按窗口内 token residue 静态划分 worker，并在每 worker 的窗口末汇总 consumed。
窗口只有全部 token 完成才可释放，汇总不会让未消费 token 被覆盖，也不改变固定
数值归约顺序。地址表是调用内 immutable metadata，不增加 payload ring 容量。

候选 `build/static_v4_ring_address/test_cuda_notify`，SHA256
`d1bdebae28d29bcf9e77c4e0b818a6824d4bd1ec366bc634314ecd725c789400`。
EP8/B256/D4/1620 MHz/10/20/20，正式四规模测试时无并行编译/GPU 任务，
32 个 hash 全部与 J 相同。另一个与 tensor extension 编译短暂重叠的 8K 初测
2436.81 us 不纳入下表；独立复测为 2437.51 us。

| tokens/rank | J/us | L/us | 相对 J 延迟变化 | L pool MiB |
|---:|---:|---:|---:|---:|
| 1 | 128.869 | 125.904 | -2.30% | 25.15 |
| 64 | 205.992 | 186.222 | -9.60% | 63.73 |
| 8192 | 3088.12 | 2437.51 | -21.07% | 296.01 |
| 20480 | 7550.21 | 5931.10 | -21.44% | 465.65 |

L 相对 K 的 8K 3056.75 us 改善 20.26%；相对冻结 V3 的 1488.19 us 仍慢
63.79%，20K 相对 V3 3288.22 us 仍慢 80.37%。**尚未达到最终长文性能目标。**
L library 的四规模及全部 PyTorch edge cases exit 0（`torch_edges_d4.log`），
包括 20K 热点不丢 token、mixed decode/prefill、空 source、改变图输入和跨层共享
workspace。B16/D1 下 T1/64 及现有 edge cases 同样 exit 0
（`torch_edges_b16_d1.log`），覆盖小容量频繁 wrap。
第二节点独立运行单机 EP8 的 B16/D1/T64 memcheck/racecheck 均 exit 0，沿用
`--sanitizer-report-api-errors no`；它们是同步验证，不是 EP16 性能测试。
host suite 18/18 通过。当前仍保留 contribution 默认深度 2，不由一次深度消融
擅自改变默认容量；性能表中的 D4 已明确标注。

第二节点单机 EP8 的独立容量消融（其 CPU/GPU 无其他任务；第一节点编译下一候选）：
8K/D4/1620 MHz/10/20/20，B256/512/1024 分别为 2427.71/2368.64/2491.15 us。
B512 仅较 B256 改善 2.43%，不将两节点间差异计作优化收益；暂无理由大幅增容。

### M：top-k=4 dense Reduce 指针缓存（无收益，撤回）

在 L 上为 dense top-k=4 增加编译期 specialization，跨列向量缓存四条 contribution
指针，展开四项加载/相同顺序的 FP32 累加。其他 top-k 保持通用路径。
候选 `build/static_v4_ring_reduce4/test_cuda_notify`，SHA256
`a574ee7330509e55ab98f9171d65e749c049b0f7ebc75b98fea9816902ebf577`。
第二节点上同条件、相邻执行 L/M，8K/B256/D4 为 2426.69/2428.19 us，hash 一致。
没有端到端收益，撤回 specialization 和分支，不保留额外模板来增加维护成本。

### N：完整 TMA wait 后直接 release publication（验证中）

回到 L。保留 `cp.async.bulk.wait_group 0`（没有 `.read`），它等待读和写全部完成，
将 async 写入导入 issuer 的 generic proxy；之后 X full 的 device release 和
contribution ready 的 system release 分别向正确范围的读者发布数据。
因此移除 ring 热路径中的额外 device/system SC fence，未缩短 TMA 完成条件，
也未删除 ready/consumed 同步。同时为原有 PTX acquire/release helper 补齐
compiler `memory` clobber，防止 C++ 内存访问越过这些同步点。

依据：[PTX bulk wait 语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)、
[async proxy 完成与隐式 fence](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#async-proxy)。
这是本地 TMA/peer 路径的消融，不删除未来 NIC 本地 completion credit。

N binary SHA256：`a04b7f1eadbfe22d03768a6503c01bb7d05f289eb3c5680a9d36bb24b8bc0f09`。
第二节点单机 EP8/B256/D4/1620 MHz/10/20/20，四规模 p50 为
125.853/180.406/2199.95/5335.52 us，32 个 hash 与 L 一致。
8K 相对同节点相邻 L 对照 2426.69 us 改善 9.34%；其他规模的 L 对照位于
第一节点，暂不据此声称精确的逐步收益。N library 的四规模及全部 PyTorch
edge cases exit 0（`torch_edges_d4.log`）；B16/D1/T64 memcheck/racecheck 均 exit 0。

### O：单 owner warp 并行准入（验证中）

N 上，将 CTA 0 的 admission 从单 lane 改为单个 owner warp。不同 lane 静态
检查/发布不同 slot，用 warp vote 判断整个目的窗口是否可准入；source contribution
的 target 仍由 lane 0 独占更新，随后经 warp 内同步并行发布各 destination 的
reservation head。source 与 destination 仍是两个独立的非阻塞进度，不增加
节点 collective、任务领取或第二个 cursor owner。

O binary SHA256：`8886699454a0ed1412f7a0998393fdff8dd0efa0d0bb4b64ae71bdee90aaecdf`。
同一第二节点、同样 EP8/B256/D4/1620 MHz/10/20/20：

| tokens/rank | N/us | O/us | 延迟变化 |
|---:|---:|---:|---:|
| 1 | 125.853 | 110.208 | -12.43% |
| 64 | 180.406 | 162.150 | -10.12% |
| 8192 | 2199.95 | 2158.59 | -1.88% |
| 20480 | 5335.52 | 5344.68 | +0.17% |

32 个 hash 一致。收益主要在短序列，20K 无改善，不声称其解决长文吞吐。
O library 的完整四规模/edge cases 和 B16/D1 的 T1/64/edge cases 均 exit 0。
B16/D1/T64 的 memcheck/racecheck 也均 exit 0（API 报告选项同前）。

### P：warp 合并向量搬运（失败，撤回）

O 上将每方向的 single-issuer TMA 改为一个 warp 共同搬一整行，lane 处理连续
uint4，四次循环展开。lane 0 acquire readiness，warp 同步导入该可见性；所有
payload 访问完成并再次 warp 同步后，lane 0 才发布 ready/retirement。pull 的
generic stores 额外经过 async-proxy fence，再发布 X full 供 FC1 TMA 读取。

第一轮保持相同 shared-memory launch 预算、window 和 contribution 深度；暂时
保留未使用的 staging 分配以隔离 copy 方式的变化。若有效再删除该实验冗余。
不改 FP32 Reduce 或 FC1/FC2 tile。

第一节点 EP8/8K/B256/D4/1620 MHz/10/20/20 为 6636.13 us，8 个 hash 一致，
但延迟超过现有 TMA 方案三倍。撤回整个候选，不保留向量搬运分支或未使用
staging 的实验代码；不继续为明显失败的候选追加完整测试。TMA 仍是主路径。

## M3a：双向 packet 目录与独立 source metadata（已验证）

- Dispatch 在冻结 consumer counts 的同一遍遍历中建立 remote route→packet/token
  row 反向索引。它只覆盖实际 remote valid routes，不将重复 expert slot 合并。
- Return 使用 source 的窗口/node/expert 连续 route 区间生成 channel prefix 和
  packet 目录；`rows` 就是完整 contribution 的退休数量，空 channel 不生成包。
  不依赖物理 C slot，也没有运行时 fragment append/claim。
- 两个目录复用 channel prefix 压缩和 packet 定位 helper，record/view 定义移到
  host/device 共用的 `plan.hpp`，供注册区布局直接使用。
- Source epoch/error、gate、window prefixes、route/node-token 数据集中到独立
  metadata region。单机已使用该区域；网络专用目录不占单机容量。X 不在 region
  中，后续 rail metadata 复制不会隐式广播完整 X。原 fused gate allocation 复用
  region 中的 gates，不保留第二份 gate buffer。
- pool 明确验证当前 M128、256 math / 32 producer / 96 communication 的布局
  前提，避免使用其他编译期 worker 划分时执行固定布局。

复用原 GPU planner 检查的 31 个 source/packet 和 6 个 destination cases，增加
Dispatch 反向位置、Return prefix/尾包/唯一覆盖检查，未新增测试入口。
1/2/3 节点规划、空 source、masked/duplicate route、20K 热点和非法 id 均通过；
同一测试 memcheck 为 0 errors，racecheck 为 0 errors / 0 warnings。
**此处的多节点检查仍是单 GPU 规划，不是多机 GIN 数据面验收。**

GPU planner binary SHA256：
`3a6da25db100f9cad0a67a170ced3ad0ee99eeafc1842fe60ba76ec9ab4f9b58`。
主路径 `build/static_v4_ring_metadata/test_cuda_notify` SHA256：
`e893d52544b19b3b3a20224ed4d4839c1419714fa69d62ac82af5166d10a9983`。
第二节点 EP8/B256/D4/1620 MHz/10/20/20 四规模为
110.080/162.856/2128.24/5250.85 us，32 个 hash 与 O 一致；约 1% 的长文变化
不归因为本次布局整理的优化收益。完整 PyTorch 四规模及 edge cases exit 0，
host suite 18/18 通过。B16/D1 的 T1/64 及全部现有 edge cases exit 0；
第二节点单机 B16/D1/T64 memcheck/racecheck 均 exit 0（同前述 API 报告选项）。
多机 ring 仍未启用。

## M3b/M4：双向网络 ring 首版（数值通过，性能未达标）

已接线的候选使用分离四区 TX/RX、固定 channel owner、逐 slot NIC ready/消费
credit 与本地 completion；Return rail 根据 immutable RouteLocation pull C，
source RX scatter 后即可归还网络 credit，Reduce 后释放 contribution。网络 payload
不再广播整个 X，只发送实际命中节点的去重 token；计划元数据仍在调用前交换。

相对原提案的具体收敛见设计第 6.6 节：RX 可有界预取，X 写入前做 destination
整窗准入，省掉跨 NIC 窗口准入消息；已有 immutable 目录，不再在 packet 重复
携带 token/fanout/Return records。Return rail 自己 pull 已 ready C，避免额外
packer publication。末尾保留调用级 GIN barrier 保护 metadata 复用，不将其搬到
逐 packet/window 热路径。此处只记录实现边界，不提前声明正确性或性能收益。

验证结果：host suite 18/18；单机 EP8/T64 graph 冒烟通过；双机 EP16 的现有
PyTorch 四规模/edge cases 全部通过，另跑 B16/D1 的 T1/64 与全部小规模 edge
cases 通过，两节点进程均 exit 0。包含混合 decode/prefill、空 source、重复
expert、共享 workspace 多层 graph、变动输入/路由、20K 全热点不丢 token及
热点后复用。没有新增测试入口。双机 sanitizer 首次尝试在 rank 0 bootstrap
前超过 launcher 的 60 秒限制，没有进入数据面，**不算 memcheck 通过或内核失败**。

冻结 binary `build/static_v4_ring_network/test_cuda_notify` SHA256：
`ded7fb2a29f2b10f01564e80e9e2802f048b00ca166d263e32a52301e7b23492`。
1620 MHz 快速 graph 筛选（10 warmup、2 samples、batch=2，非正式 10/20/20）：

| EP16 tokens/rank | B / contribution depth | rank-max p50/us |
|---:|---:|---:|
| 1 | 256 / 4 | 289.888 |
| 64 | 16 / 1，复用压力测试 | 2562.85 |
| 8192 | 256 / 4 | 72605.2 |
| 20480 | 256 / 4 | 175901.0 |

这不是性能完成版本；8K/20K 远慢于 V3。T64 小窗口结果不与常规 B256 性能比较。
T64/8K/20K 前八个 rank hash 与单机同 fixture 一致；独立 PyTorch 参考覆盖全部
16 ranks。日志见 `build/ring_network_{ep16_*,torch_ep16_*}.log`。

首个 profile，EP16/8K/B256/D4，rank 0 的每 packet p50：Dispatch pack 21.2 us，
pack end→Put 16.8 us；Return pack 51.0 us，pack end→C 退休及 Put 102.8 us；
source Return scatter（含逐行发布）71.6 us。两条 Return lane 各有 256 包，
每 lane 累积 pack 12.5 ms、退休/提交 25.8 ms，scatter owner 累积约 18.3 ms。
这些区间并行，不可直接相加为端到端延迟。trace overflow=0。
首个 FC1 在约 2.85 ms，最后 Dispatch Put 约 64.85 ms；Return 从约 3.57 ms 开始，
FC2 持续至约 68.21 ms。流水已重叠，问题主要是通信控制/搬运吞吐，不是仍有
Dispatch→FC 的全节点 barrier。下一步单独消融重复 C-ready 检查、逐行 system
退休与 scatter 发布，再考虑通信 worker 数，不能把这些代价归为 GEMM 效率。

补验：单机 EP8 同一 library 完整四规模/edge cases exit 0。双机 B16/D1/T64
graph 的 memcheck 改为同时 instrument rank 1、9，其余 rank 正常参与；两份
报告均 `ERROR SUMMARY: 0 errors`，launcher exit 0。这覆盖两节点各一个 rank，
不是全部 16 rank 的 sanitizer 验收；最初 rank 0 bootstrap 超时记录仍保留。

### Q：Return 按连续 C slot 合并 readiness/退休（验证中）

保持 packet、Q、lane、GIN completion 和 payload copy 不变。一个 packet 只含
一个窗口，该窗口内每个逻辑 C block 的物理 slot 唯一；对连续属于相同 C slot
的行，只检查一次 generation/FC2 done，复制完成后按该段完整行数更新一次
returned。并未省掉不同 slot 的检查，也未把“发起读取”当作消费完成。预期减少
重复 system 读取及 RMW，收益等待独立测量。

初次仅合并连续 C slot 的 8K 快速筛选为 53685.1 us，16 hash 与首版一致，约
下降 26.1%。它仍不是正式性能验收。同步复核随后发现首版 C.returned 的本地
RMW/准入 load 为 device scope，而 rail 的 remote RMW 为 system scope；虽然
现有数值测试通过，按 [CUDA 12.9.1 PTX 原子作用域合同](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#memory-consistency-model-morally-strong)
两边都必须覆盖对方线程。现统一多机 C.returned 的所有 RMW 和准入 load 为
system scope；退休只需要 release-add，不消费旧值，因此使用单向 PTX reduction。
单机保持 device scope。首版和初次 Q 测量带此合同缺口，不能作为最终交付的
正确实现；修正版需重新做完整数值、复用及性能验证。

修正版 binary `build/static_v4_ring_net_scope/test_cuda_notify` SHA256：
`f8d3853a4e86dd87b40d6f215df5dcde8db226dab510b7b6c3e5cb6320aad17c`。
同样 8K 快速筛选 53670.4 us，16 hash 一致；完整 EP16 PyTorch 四规模/edge
cases 和 B16/D1 的 T1/64/edge cases 均 exit 0，包含 20K 热点及跨调用复用。
同一修正版的双机 B16/D1/T64 racecheck（instrument rank 1、9）exit 0，两份
报告均为 0 errors、0 warnings。正式 10/20/20 四规模与冻结 V3 的相邻对照另记，
不将上述两样本快速筛选伪装成正式矩阵。

本轮另在第二台机器独立重测单机 EP8 的冻结 V3（1620 MHz、10/20/20；
`WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`，每 rank 图内 2 kernels）：1/64/8K/20K
分别为 300.374/370.218/1489.09/3294.90 us。与前一节点的正式基线相符，
日志为 `build/ring_v3_remote_ep8_t*.log`。该组是单机 EP8，不是两机 EP8。

正式 EP16 对照（两节点无其他 GPU 任务/编译负载，1620 MHz，10/20/20，逐规模
先冻结 V3、后修正 Q；相邻对照，不是交错反转顺序 A/B）：

| tokens/rank | 冻结 V3/us | Q + scope 修正/us |
|---:|---:|---:|
| 1 | 682.170 | 289.083 |
| 64 | 824.659 | 1504.43 |
| 8192 | 3353.94 | 54062.1 |
| 20480 | 7099.53 | 92113.8 |

全部 64 个 rank/hash 与 V3 一致。除 T1 外性能仍未达标，不作为最终版本。
日志：`build/ring_v3_ep16_t*.log`、`build/ring_net_scope_ep16_t*.log`。

### R：Return RX 的本 GPU ready 发布（验证中）

只将 Return RX scatter 后的逐 route ready store 从 system 改为 device release；
相应 Reduce 读者都在本 GPU。NIC ready 的 acquire、跨 GPU 的 C retirement、
跨 GPU local Return 的 contribution ready、TX 双重 credit 均不改变。
不同代写者受原有 contribution 生命周期约束，不能并发覆盖同一 route slot。
同时删除未使用的旧 TokenRecord/FanoutRecord/ReturnRecord 和旧 wire-layout
构造函数，保留实际 header+payload 布局，复用原地址覆盖测试；不把清理记作收益。

8K 快速筛选 53630.8 us，相对 Q 的 53670.4 us 仅约 0.07%，没有可辨认的
端到端提升。16 ranks hash 一致；EP16 PyTorch 完整四规模及 edge cases、
B16/D1 的 T1/64/edge cases 均通过，host suite 18/18。保留更准确的发布作用域
与无用类型清理，但不将 R 计作性能收益。日志 `build/ring_net_rxgpu_*.log`。

### S：按窗口内 packet ordinal 交错 source 的静态 pull 映射

旧 `(packet + source) % workers` 对 channel-major 的小 packet 区间产生大量
worker 碰撞；改为 `((packet_in_lane_window * lanes + lane) * source_group_size
+ source_in_group) % workers`。任务归属仍唯一且静态，source/lane/packet 的
遍历顺序和逐 slot credit 均不改。没有增加 task claim 原子或内存。

1620 MHz、EP16/8K/B256/D4、10 warmup/2 samples/batch2：53630.8 → 34369.7 us，
下降 35.9%；全部 16 rank hash 一致。这是快速筛选，尚非正式矩阵/完整正确性验收。
binary `build/static_v4_ring_net_spread/test_cuda_notify`，日志
`build/ring_net_spread_ep16_t8192_smoke.log`。

### T / U：warp 协同读取远端目录；独立增加网络 lane（验证中）

T 保持 2 lanes：一个 warp 并行读取 packet 的 route/Location/gate，匹配目标
rank 的行用 shuffle 交给唯一 TMA issuer；完成整包 copy 并 warp 同步后，参与
lane 发布本地 X-ready，issuer 归还累计 RX credit。两级 staging 仍只有一个
issuer，没有恢复失败实验 F 的多 issuer 复用。source/lane/packet 顺序不变，
不新增全局目录或激活内存。

U 在 T 上仅将 lane 从 2 增到 8，Q32/depth8 不变，GIN contexts 从 6 增至 24，
Return TX/RX scatter owners 各从 2 增至 8。协议、GEMM profile 不变，但网络
slot/payload 内存随 lane 增长；必须同时报告代价。配置集中到 `kNetworkPackets`，
通过构建宏 `WEFTMOE_FWD_STATIC_V4_NETWORK_LANES` 消融，尚不改变默认值。

EP16/8K/B256/D4 快速 graph 筛选（1620 MHz、10/2/2）：S 34369.7 us，T
33783.8 us，U 10714.7 us。T 相比 S 下降约 1.7%，U 相比 T 下降约 68.3%，
16 ranks hash 均一致。T/U 筛选时本节点同时有 CPU 编译，不作为正式性能验收；
最终矩阵必须在无其他任务/编译负载下重测。U 每 rank 的 GIN 注册内存由
18141184 增至 43350784 bytes，增量 25209600 bytes；该增量不随 tokens 增长。

U binary `build/static_v4_ring_net_lanes8/test_cuda_notify` SHA256：
`6ded3b4f111b3dc7b96af1bb2296830534e249f7c85ec31bd62dc3ac20ec44f1`。
完整 EP16 PyTorch 四规模和 edge cases 两节点 exit 0，包含 20K 热点后复用。

U 的 8K eager timeline rank0 overflow=0：每包 p50 Dispatch pack 23.184 us、
提交 16.032 us；Return pack 49.392 us、退休/提交 41.952 us，RX scatter
40.256 us。512 个 Return 包分给 8 个 owner；这些区间相互重叠，不能直接
相加解释端到端延迟。元数据串行处理和少量 copy issuer 仍值得优化。

U 的 B16/D1/T1/64 与小规模 edge cases 两节点 exit 0；B16/D1/T64 graph
racecheck 同时 instrument rank 1、9，两份报告均 0 errors、0 warnings，
launcher exit 0。这里仍只覆盖两节点各一个 rank，不是全部 rank 的 sanitizer。

U 的正式 EP16 矩阵（两节点无其他 GPU/编译任务，1620 MHz，10/20/20）：

| tokens/rank | 冻结 V3/us（Q 时重测） | U/us |
|---:|---:|---:|
| 1 | 682.170 | 336.082 |
| 64 | 824.659 | 641.600 |
| 8192 | 3353.94 | 10726.3 |
| 20480 | 7099.53 | 26205.8 |

全部 64 rank/hash 与 V3 一致。短序列优于 V3，长序列仍未达标；L8 比 L2
增加调用固定开销，T1 退化也如实保留。测试中两节点所有 SM 均为 1620 MHz，
内存 3201 MHz。日志 `build/ring_net_lanes8_ep16_t*.log`。

同一 U binary 的**单机 EP8** 正式回归在第二节点独立运行（1620 MHz、10/20/20，
该节点没有其他 GPU/编译任务；第一节点的 CPU 构建不参与此测量）：
T1/64/8K/20K 为 108.451/161.398/2142.31/5283.96 us，32 个 hash 与同节点
冻结 V3 一致。与前一单机 ring 候选基本相当，长序列仍落后 V3；多机 lane
改动没有解决单机瓶颈。日志 `build/ring_net_lanes8_remote_ep8_t*.log`。

### V / W：warp 协作 TX 与 RX，以及 packet 搬运方式（验证中）

V 在 U 上将每包 TX 描述符读取、C-ready 检查分给 warp，保留每 channel 唯一
GIN 调用者；复制后 `match_any` 合并同 C slot 的退休，不再串行重读 route/Location。
本地 generation/credit 的控制分支由 lane0 读取并广播，防止不同 lane 观察
不同状态而分叉等待。分别构建 TMA 单 issuer（warptx）与 warp `uint4` 搬运
（vectx），RX scatter 暂不改，独立测量搬运方式。

W 再将 RX 的地址计算/ready 发布也分给 warp，与 TX 复用 `copy_packet_rows`。
同样保留 TMA/向量两个构建（warprx/vecrx）对照。TMA 用 full wait；向量用 warp
同步传递全部普通 GM 写，GIN 仍使用 givenRelease=block / requiredRelease=system
完成 NIC 可见性发布，没有移除所需的 system release。双重 TX credit 不变。

V 的 EP16/8K 快速筛选（同 10/2/2，CPU 同时编译 W）：warptx 10637.2 us，
vectx 21880.6 us，全部 16 ranks hash 与 U 一致。TMA 协作版与 U 10714.7 us
接近，向量搬运明显退化，不采用向量路径；W 尚待测量，不提前保留无收益的复杂度。

W 快速筛选 warprx 10630.2 us、vecrx 21905.9 us，16 hash 均一致。W 的
eager timeline overflow=0，Return pack/退休提交/scatter p50 为
44.064/19.280/28.960 us；虽局部区间缩短，端到端未获明确收益。V/W 的
协作 TX/RX、共享 packet copier 和向量宏已全部撤回，避免积累无收益的复杂度；
冻结 binary/日志仍在 build 下作为失败实验记录。

### X / Y：网络 owner 数与单机 copy batch（验证中）

X 把 worker 上限从 8 放到 32，分别构建 L16/L32；Q32、depth8、GEMM 不变。
8-lane 的既有路径不受上限变化影响。Y 单独将单机 copy batch 从 2 行改成 8 行，
仍只用一个 issuer、两个 staging slots，在整批 full completion 后才发布；批次
不跨整窗准入边界。多机 local copy 仍为 2 行，不混入 X 的性能归因。

X 的 8K 快筛：L16/B256 14365.4 us，L16/B512 9878.74 us，L32/B256
19224.0 us，L32/B512 11790.3 us（D4、1620 MHz、10/2/2）。更多 lane 没有
自动带来收益：原 pull 热路径每个 worker 每窗扫描全部 source×lane，即便没有
自己的包，仍读 peer prefix/offset/state。这个代价随 lane 增加，下一项将独立移出。

Y 的远端单机 EP8 正式 10/20/20 为 112.552/188.981/2879.54/7085.94 us；
32 hash 均与 V3 一致，但四规模均退化，8K/20K 约慢 34%。已删除 Y 的宏与
实现变化，恢复 batch2，不保留此无收益调参分支。X 的四组 8K/16 rank hash
也均一致；X 的 owner 上限需与下一项共同验证，而非直接采用 L32。

### Z：预绑定本 GPU 的静态 PullPacket 表（验证中）

每窗口预留 `source_group_size * ceil(B/Q)` 个固定 entry，索引为
`packet_ordinal * source_group_size + source_in_group`，延续 S 的 ownership。
source-local、无命中、尾部空 entry 只标记 rows=0，运行时不读取 RX。
计划阶段复用窗口 CTA，将 RX payload/control 指针、目标 generation、source
目录指针和已有 consumer count 绑定到普通本 GPU metadata；不增加 GIN 元数据
交换，不新增全局 barrier，不重算 expected count。runtime 仅访问
`worker, worker+workers, ...` 的本地 entry，不再全量扫描远端 channel。

每 entry 80 bytes；EP16/B256/20K 上界为 819200 bytes/rank，仅 metadata，
TX/RX/X/S/C payload 容量不变。完整 plan 和跨调用物理地址仍逐调用冻结，不能
把上一调用的 RX generation/address 表直接沿用。源码替换删掉了 hot path
中的 source/lane/prefix 遍历；仍保留单 issuer TMA 和逐 packet fanout retirement。

Z 的 EP16/8K 快筛（1620 MHz、10/2/2、D4）：

| lanes / B | 原遍历/us | 预绑定/us |
|---|---:|---:|
| 8 / 256 | 10714.7（U） | 9684.38 |
| 8 / 512 | 未独立测量 | 9883.71 |
| 32 / 256 | 19224.0（X） | 7325.20 |
| 32 / 512 | 11790.3（X） | 4766.67 |

64 个 hash 全部一致。L32/B512 相同参数下下降 59.6%，但仍慢于 V3，不能
声称完成性能目标。Z32 binary SHA256：
`7050c1e95e0ec194b87d304ac0b0de6fee128ceeb11d75eff6d410a6428038ff`。
完整 EP16 PyTorch 四规模/edge cases（B512/D4）与 tiny ring（B16/D1、T1/64/
edge cases）两节点均 exit 0；host suite 18/18。两节点各 instrument rank 1/9
的 B16/D1/T64 graph memcheck 均 0 errors，launcher exit 0。

### AA：控制/数学 warpgroup 寄存器配额（验证中）

仅通过已有构建宏比较 math/control 216/64、216/72、208/88，均满足 384-thread
CTA 的实际初始 register pool（168×384=64512），不把 SM 的 65536 全额当作
该 CTA 可动态领用的配额。通信协议、packet/窗口、GEMM tile 均保持不变。
