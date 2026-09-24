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

保持 L32 的窗口/depth 快筛：B1024/D4 4748.72 us、B2048/D4 4690.93 us、
B512/D8 4781.50 us、B512/D16 4773.94 us、B1024/D8 4765.62 us。
全部 80 rank/hash 一致，但相对 B512/D4 的 4766.67 us 没有实质改善；
B1024/D4 与 B2048/D4 的总 pool 分别增至 1658820800、2934562240 bytes/rank。
保留 B512/D4，不用大幅增容换取快筛中的微小差异。

### AA：控制/数学 warpgroup 寄存器配额（验证中）

仅通过已有构建宏比较 math/control 216/64、216/72、208/88，均满足 384-thread
CTA 的实际初始 register pool（168×384=64512），不把 SM 的 65536 全额当作
该 CTA 可动态领用的配额。通信协议、packet/窗口、GEMM tile 均保持不变。

8K 快筛：216/72 为 4711.65 us、208/88 为 4736.96 us，32 rank/hash 均一致。
相比 216/64 的 4766.67 us 仅约 1%，暂不修改默认配额；这些不是独立 GEMM
测量，不能据此判断 math/control 的最终最优配比。

### AB：packet 目录容量使用包数上界（验证中）

原 Dispatch/Return packet 目录各按最大 route 数分配，consumer count 还乘了
local_world；PlanReady 交换连同这些大段空容量一起发送。现取两个合法上界的
较小值：route 总数，或 `local_windows × (nodes−1) × 每节点每窗最大包数`。
Dispatch 每节点每窗最多 `ceil(B/Q)` 包，Return 最多 `ceil(B×K/Q)` 包。
覆盖 partial window、热点、重复 expert 和多节点稀疏分散，未假设均匀路由。
源目录整体布局/Put 协议不变，仅缩小 allocation 与交换跨度；payload ring 不变。

EP16/8K、L32/B512/D4 快筛 4713.58 us，16 rank/hash 一致，相对 Z 的
4766.67 us 仅快约 1.1%，不算明确端到端收益。GIN 注册内存从 144189184
降至 138977536 bytes/rank，减少 5211648 bytes。保留此容量清理。
完整 EP16 PyTorch 四规模与 edge cases、B16/D1 的 T1/64 与 edge cases
均两节点 exit 0。小 ring graph memcheck 同时覆盖 rank 1/9，均 0 errors，
launcher exit 0；没有声称全部 16 ranks 都经过 sanitizer。
日志 `build/ring_packet_capacity_*`，冻结 binary/lib/extension 位于
`build/static_v4_ring_packet_capacity/`。

AB 正式矩阵（1620 MHz、10/20/20；EP8 第二节点独立测试，EP16 两节点均无
编译/GPU 竞争任务）：

| tokens/rank | V3 EP8/us | AB EP8/us | V3 EP16/us | AB EP16/us |
|---:|---:|---:|---:|---:|
| 1 | 300.374 | 108.637 | 682.170 | 303.306 |
| 64 | 370.218 | 160.864 | 824.659 | 590.722 |
| 8192 | 1489.09 | 2156.34 | 3353.94 | 4716.07 |
| 20480 | 3294.90 | 5303.55 | 7099.53 | 11166.5 |

全部 96 rank/hash 与相应 V3 一致。EP16 长序列相对 U 降低 56.0%/57.4%，
但仍慢于 V3；EP8 未随网络目录优化改善。日志 `build/ring_packet_capacity_ep16_t*.log`
及 `build/ring_packet_capacity_remote_ep8_t*.log`。EP8 时第一节点可有 CPU 编译，
实际测试的第二节点无其他任务，不能把此拓扑称作双机 EP8。

### AC：prefill FC1 N128→N256（无收益，撤回）

仅放大 packed FC1 的 N，逻辑 SwiGLU 列宽从 64 变为 128；FC2、stages、
权重排列、通信协议不变，共享内存上界仍由 FC2 决定。8K 快筛单机 EP8
B256/D4 为 2178.82 us，双机 EP16 L32/B512/D4 为 4742.54 us，24 hashes
与冻结 V3 一致。没有明确端到端收益，恢复 N128，不保留临时调参宏。
这是融合路径的 tile 实验，不是纯 GEMM 或 gate/up 布局的独立消融。
候选冻结在 `build/static_v4_ring_fc1n256/`。

### AD：节点内 FC2 直接写 contribution ring（验证中）

FC2 的节点内行直接写 source 已预留的 contribution slot，跳过 C→contribution
的第二次 payload 搬运；跨节点行继续写 C，交给现有网络 Return。FC1 的
X-ready 已包含 source reservation，不提前写未预留的 contribution。
固定 Return warp 按 Mblock residue 等所有 FC2 N tiles；acquire 完成链、system
fence、warp join 后协作发布逐 route ready，再按实际本地行数退休 C。零本地行
的 block 只检查 immutable plan，不读取可能已被网络 owner 释放的 C slot。
contribution 仍由 Reduce 退休；网络 packet/双重 credit 与 expected count 不变。

8K 快筛 EP8 B256/D4 为 1687.55 us，相对 AB 2156.34 us 降低约 21.7%；
EP16 L32/B512/D4 为 4757.62 us，与 AB 4716.07 us 相当。24 hashes 一致，
没有把单机收益归给多机。完整 EP16 PyTorch 四规模/edge cases，以及 B16/D1
T1/64/edge cases 均两节点 exit 0。两节点 rank 1/9 的小 ring graph
racecheck 均 0 errors、0 warnings，launcher exit 0。单机 EP8 的四规模 PyTorch
及全部 edge cases 也已 exit 0，包含 20K 热点后复用。正式性能矩阵待后续回测。

随后第二节点单机 EP8 正式 1620 MHz、10/20/20 为
105.128/154.624/1682.80/4051.05 us，32 hashes 与同节点 V3 一致。
8K/20K 相对 AB 降低 22.0%/23.6%，仍慢于 V3；日志
`build/ring_direct_return_remote_ep8_t*.log`。该节点没有其他 GPU/编译任务。

### AE：在 Z 之后重测 warp 协作 packet 处理（验证中）

恢复 V/W 中有局部收益的 TMA 版本，不恢复向量 copy 和试验宏。每包最多 32
行，warp 协作读取描述符、检查 C-ready，lane0 保持唯一 TMA/GIN issuer；
full completion 后 warp join，再按同 C slot 合并退休。RX scatter 复用同一个
packet copier，并行发布逐 route ready，join 后才释放 RX。generation、credit
等可能瞬时变化的分支由 lane0 判断并广播，避免 lane 分歧等待。

EP16/8K L32/B512/D4 快筛 4682.67 us，16 hashes 与 V3 一致；对照 AD
4757.62 us 仅降低 1.6%。仍未显示足以保留这份控制复杂度的收益，再次撤回
协作 TX/RX 实现。冻结候选 `build/static_v4_ring_packet_warp/` 与日志保留。

### AF：纯 GEMM gate/up 基准（验证中）

扩展已有 `run_gemm_diagnostics.py --phase static-forward`，直接从生产源码提取
packed/separated/FC2 的数学循环，复用生产 packed SwiGLU epilogue；separated
fragment 以编译期寄存器排列接入相同 epilogue，从而匹配有效 S 列宽与写回。
该 controlled 比较不是原封不动的 V3 全流程。两边 cluster2 共享 A、相同
M128/K/stages/输出宽度与精度，FC2 固定 N256/K128/stages2；无通信、route
planning 或队列。chain 用 FC1+SwiGLU、FC2 两个顺序 kernel，不模拟融合重叠。

输入按 route 计数重放 expert 几何、尾块；balanced 测 rank0，
hotspot 测最后一个 rank，包含重复 expert 的全部 route。比较完整输出与
cuBLAS+同精度 SwiGLU，检查 replay 字节一致；packing/分配/descriptor 均在
计时外，单独报告 packing 时间与额外副本大小。吞吐只按有效行的 GEMM FLOPs
统计，不把 M128 padding 或 SwiGLU scalar FLOPs 计入有效 tensor throughput。

**夹具更正（AS）**：本节首次测试的 balanced 公式为
`(token*4 + slot + source*7)%64`，并非主性能测试的
`(token*17 + source*31 + slot*53)%64`。逐专家计数核对表明 world8/16 的
T64/8192/20480 完全相同，热点也不受影响；原 T1 数据仅作为旧合成夹具的
受控比较保留，不再称为主矩阵尾块。AS 修正实现并列出主矩阵 T1 重测。

第一版将 producer/math 放在同一角色函数，separated 被 ptxas 报告 WGMMA
serialization、stack264，不能用该版本判断布局。已按生产模式分离角色代码；
N64/stages2/K128 的三个入口均 REG168、STACK0、LOCAL0，相关编译警告消失。
初步 T64/EP8 balanced 的 packed FC1、separated chain 完整输出校验通过，
正式矩阵尚未完成，不填布局胜负。

AF 的第一组受控结果如下。**单 GPU 重放 route world=8/16 的 expert 几何，不是
EP 通信性能**；balanced 取 rank0，hotspot 取最后一个 rank。两边每任务均输出
S 的 N64 列：packed 物理 N128，对照 separated 两个 N64 投影。K128、stages2、
cluster2、math/control 216/64；chain 的 FC2 固定 N256/K128/stages2。以下是
CUDA Graph 的 30 个 native event samples 的 p50；2 s 联合预热、5 个预热
samples、交替 cuBLAS 对照，cooldown=0（所有 3840 对时钟采样均为 1620:1620）。

| route world / distribution | tokens/rank | packed FC1/us | separated FC1/us | packed chain/us | separated chain/us |
|---|---:|---:|---:|---:|---:|
| 8 / balanced | 1 | 22.912 | 27.584 | 34.736 | 37.824 |
| 8 / balanced | 64 | 26.832 | 29.776 | 41.424 | 43.776 |
| 8 / balanced | 8192 | 329.152 | 386.400 | 483.920 | 539.504 |
| 8 / balanced | 20480 | 782.112 | 925.792 | 1152.640 | 1294.720 |
| 8 / hotspot | 1 | 23.776 | 28.576 | 33.856 | 38.592 |
| 8 / hotspot | 64 | 25.408 | 29.872 | 37.536 | 42.032 |
| 8 / hotspot | 8192 | 2440.144 | 2919.552 | 3590.624 | 4070.912 |
| 8 / hotspot | 20480 | 6069.264 | 7256.784 | 8893.744 | 10114.064 |
| 16 / balanced | 1 | 22.896 | 27.552 | 32.832 | 37.712 |
| 16 / balanced | 64 | 24.544 | 28.832 | 35.472 | 39.552 |
| 16 / balanced | 8192 | 326.336 | 383.456 | 486.208 | 539.504 |
| 16 / balanced | 20480 | 785.792 | 922.128 | 1155.680 | 1298.368 |
| 16 / hotspot | 1 | 23.936 | 28.576 | 34.448 | 38.784 |
| 16 / hotspot | 64 | 44.672 | 53.680 | 70.944 | 77.632 |
| 16 / hotspot | 8192 | 4858.896 | 5812.128 | 7127.664 | 8097.376 |
| 16 / hotspot | 20480 | 12137.104 | 14534.496 | 17767.792 | 20240.416 |

64 个用例全部通过完整输出、字节重复检查。两个布局的 T1024/world8/balanced
chain 另在第二节点单 GPU 做 racecheck，均 0 errors、0 warnings；这是独立
基准的同步验证，不是多机 sanitizer。正式样本在第一节点 GPU0，测量期间该
节点无 CPU 编译或其他 GPU 任务。原始记录与 provenance 位于
`build/static_gemm_control_n64_s2_k128/`；构建参数与 binary SHA 见其
`provenance.json`。N128 输出宽度与各自调优仍待完成，不能据此直接定最终 profile。

初次 Nsight Compute（packed FC1，8K/world8/balanced，clock-control=none、
cache-control=none）显示 168 registers/thread、动态 SMEM 131120 bytes、
driver SMEM 1024 bytes、理论/实测 occupancy 18.75%/18.78%，SM throughput
53.68%、DRAM throughput 13.39%。CTA barrier stall 包含空闲 producer lanes
等待整个 CTA 的时间，不等同于 math warp 的 TMA stall；不能把它直接用于解释
端到端流水瓶颈。其 profiler 运行的 event 时间不混入上表。

AF 第二组：有效 S 宽度 N128、K128/stages2，packed 物理 N256，
separated 两个 N128。其余合同相同；第二节点 GPU0，节点无其他 CPU/GPU
任务，64 个完整输出/replay 检查通过，3840 对时钟均为 1620:1620。
所有入口仍 REG168、STACK0、LOCAL0；结果位于
`build/static_gemm_control_n128_s2_k128/`。

| route world / distribution | tokens/rank | packed FC1/us | separated FC1/us | packed chain/us | separated chain/us |
|---|---:|---:|---:|---:|---:|
| 8 / balanced | 1 | 35.200 | 36.928 | 45.664 | 47.472 |
| 8 / balanced | 64 | 39.104 | 40.368 | 52.464 | 53.824 |
| 8 / balanced | 8192 | 271.184 | 285.312 | 429.952 | 437.952 |
| 8 / balanced | 20480 | 664.416 | 702.432 | 1036.192 | 1073.584 |
| 8 / hotspot | 1 | 37.728 | 39.424 | 47.584 | 49.264 |
| 8 / hotspot | 64 | 38.448 | 40.128 | 50.400 | 52.160 |
| 8 / hotspot | 8192 | 2046.416 | 2188.336 | 3236.496 | 3346.128 |
| 8 / hotspot | 20480 | 5053.664 | 5404.368 | 7911.840 | 8182.784 |
| 16 / balanced | 1 | 35.024 | 37.120 | 45.440 | 47.296 |
| 16 / balanced | 64 | 37.936 | 39.680 | 48.928 | 50.784 |
| 16 / balanced | 8192 | 268.208 | 285.328 | 426.208 | 439.472 |
| 16 / balanced | 20480 | 662.512 | 699.088 | 1035.776 | 1072.592 |
| 16 / hotspot | 1 | 37.680 | 39.232 | 47.728 | 49.664 |
| 16 / hotspot | 64 | 38.944 | 41.024 | 62.288 | 64.144 |
| 16 / hotspot | 8192 | 4106.800 | 4273.424 | 6345.856 | 6634.448 |
| 16 / hotspot | 20480 | 10067.936 | 10656.800 | 15720.192 | 16305.345 |

AF 参数消融：world8/balanced，列中依次为 T=1/64/8192/20480 的 p50/us。
N 是有效 S 宽度。N32/stages3 只测 packed（其物理 N64；separated N32
不支持当前 WGMMA atom）；N64/stages3 在第一节点，其余 K64/stages4 在
第二节点。各节点测量期间无其他 CPU/GPU 任务；56 个用例全通过，
3360 对时钟均为 1620:1620。原始目录为 `build/static_gemm_control_n*_s*_k*/`。

| N / K / stages | layout | FC1/us（四规模） | chain/us（四规模） |
|---|---|---|---|
| 32 / 128 / 3 | packed | 17.952 / 19.952 / 408.896 / 989.344 | 28.496 / 34.256 / 563.808 / 1360.256 |
| 64 / 128 / 3 | packed | 22.816 / 25.280 / 309.536 / 736.240 | 33.552 / 38.528 / 466.416 / 1106.336 |
| 64 / 128 / 3 | separated | 27.232 / 29.632 / 383.312 / 917.856 | 37.936 / 43.040 / 538.880 / 1285.440 |
| 64 / 64 / 4 | packed | 23.840 / 26.688 / 322.640 / 780.592 | 34.864 / 39.600 / 482.560 / 1152.176 |
| 64 / 64 / 4 | separated | 30.048 / 32.288 / 423.312 / 1018.336 | 40.640 / 45.488 / 580.576 / 1391.008 |
| 128 / 64 / 4 | packed | 35.312 / 39.184 / 271.328 / 665.136 | 45.840 / 52.304 / 429.328 / 1036.448 |
| 128 / 64 / 4 | separated | 39.200 / 42.848 / 304.560 / 746.560 | 49.776 / 56.112 / 462.128 / 1108.736 |
| 64 / 64 / 6 | packed | 23.920 / 26.448 / 322.592 / 774.224 | 34.704 / 39.760 / 482.624 / 1147.088 |
| 64 / 64 / 6 | separated | 29.680 / 31.776 / 417.888 / 1004.768 | 40.416 / 45.360 / 575.696 / 1378.576 |
| 32 / 128 / 4 | packed | 18.368 / 20.080 / 419.088 / 1008.864 | 29.232 / 33.120 / 576.128 / 1382.432 |

后补 N64/K64/stages6 与 N32/K128/stages4 也在第二节点 GPU0 测量；
24 个用例全通过，1440 对时钟均为 1620:1620。增加 stage 没有稳定改善。

当前候选中，小 N32/stages3 的 packed 适合短序列；长序列 packed N128/
K128/stages2 最好。增大 S 宽度后，两种布局差距明显缩小，但 separated
没有反超；K64/stages4 也未带来收益。不同 N 的部分测量来自不同物理节点，
最终选择仍需同卡复测及端到端确认。AC 已说明只增加 N 并不能消除通信等待，
不把纯 GEMM 的改善直接填成 EP8/EP16 收益。

### AG：四级 TMA row prefetch（无收益，撤回）

只扩展已有 `RowPrefetchPipeline` 的 staging 深度，保留唯一 issuer；
四个独立 slot/barrier 先发 G2S，再逐个退休 S2G，复用前等待相应 store group
的 shared read 完成；payload 发布仍使用 full completion。短 batch drain
后也必须完成所有 store，才再次使用该 pipeline。通过
`WEFTMOE_FWD_STATIC_V4_RING_COPY_STAGES=4` 试验，默认仍为 2。
ring 的 shared allocation 按实际 staging 大小计算，不把 4 份 buffer
放进原先 2 份容量；非 ring 路径预算不变，host 继续验证真实 entry 的 SMEM。

单机 8K/B256/D4 快筛 1669.82 us、8 hashes 一致，与 AD 约 1.68 ms 基本相同；
多机待测。独立 AD 窗口快筛 B256/512/1024/2048 为
1672.98/1654.40/1770.38/2014.24 us（CPU 同时构建 AG），32 hashes 一致，
不把 B512 约 1% 的差异当作明确收益；保留 B256。

AG 的 EP16/8K L32/B512/D4 快筛 4832.83 us，16 hashes 一致，但没有改善。
网络 ring entry 的 stack 从 480 增至 584 bytes，不能据此单独断言退化原因。
已撤回四级预取、临时宏和 shared-budget 扩展，保持两级实现。

### AH：expert 内改为 source-major 行序（无收益，撤回）

原 source round-robin 让每个 Mblock 混合更多 source，Return 同一 source 的
连续 routes 也映射为有间隔的 C 行。新候选采用 expert 内 source prefix +
source ordinal，相同 source 连续；减少一个 block 的 X-ready 依赖方，并让
Return 的源行按相邻地址排列。没有额外 padding、buffer 或动态任务分配，
packet/fanout/retirement 计数不变。同步修改现有几何校验的独立 host 枚举，
不增加新的单测入口。source 内 ordinal 暂仍是 inference 的一次性原子排布，
不能把 source-major 本身称为 StableTape；训练仍需稳定化此 ordinal。

EP8/8K B256/D4 快筛 1673.58 us，EP16/8K L32/B512/D4 4764.24 us；
24 个输出 hash 与冻结 V3 一致。现有 GPU plan 检查 31 source/packet +
6 destination cases（含 3 nodes）通过。没有端到端收益，已撤回布局改动；
冻结 `build/static_v4_ring_source_major/` 保留，不作为生产 profile。

### AI：节点内 Pull 的 warp 合并读写（退化，撤回）

只替换 node-local X pull：整个 warp 以连续 uint4 读取源行、写入目标 X，
warp join 后由 lane0 发布本行 X-ready。源 contribution reservation 与
compute slot admission 不变；远端 RX pull、TX/RX packet copier 仍用原 TMA。
不扩展 shared memory，不增加任务领取或运行时分支；用于判断本地两行一等的
TMA staging 是否拖慢流水，尚无收益结论。

EP8/8K B256/D4 快筛 3994.93 us，8 个 hash 一致，但相对 AD 1682.80 us
退化约 2.37 倍。未继续双机或完整矩阵，已撤回向量搬运。这与此前 V/W 的
网络向量复制实验一致：当前资源分配下，整 warp 的同步 vector loop 不适合
替代通信角色的 TMA pipeline。冻结 `build/static_v4_ring_local_warp/`。

### AJ：packed FC1 的 WGMMA group overlap（纯 GEMM 退化）

packed FC1 补齐对现有 `kWgmmaWaitGroup` 的支持：wait1 时保留当前 stage，
只回收已完成的前一个 stage；循环结束 wait0 后回收最后一个 stage。没有
提前释放仍被 WGMMA 读取的 A/B。默认 wait0 不变，先在独立基准对照再决定
是否改变生产 profile；FC2 与 separated 使用已有同类流水配置。

第二节点单 GPU，world8/balanced、四规模、FC1/chain、两种布局共 32
用例全部通过完整输出/replay 检查；1920 对时钟均为 1620:1620。N128/K128/
stages2 的 packed FC1 从 35.200/39.104/271.184/664.416 us 退化至
52.320/58.336/412.320/992.976 us，chain 从
45.664/52.464/429.952/1036.192 退化至
65.568/78.928/611.568/1463.824 us。
N64/K128/stages3 的 packed FC1 为 23.280/27.328/333.120/792.912 us，
chain 为 36.880/45.008/527.088/1259.584 us，同样无改善。
separated 没有反超（完整样本位于 `build/static_gemm_wait1_n*/`）。

资源仍 REG168、STACK0、LOCAL0，但 ptxas 新增 C7510：跨函数边界的
WGMMA pipeline 序列化。因此这不是成功隐藏数学延迟的实现，不据此宣布
异步 group overlap 本身无效；不进入端到端 profile，也不保留未获益的
生产 FC1 分支。

### AK：缩小 packet Q 提高 Dispatch lane 利用率（验证中）

窗口内从 lane0 开始 stripe；实际去重 token 数少于 Q×lanes 时，部分
Dispatch owner 始终无包。对照 Q16/Q8 与原 Q32，lanes32、B512/D4 不变。
保持逐 slot generation、双 credit、精确 fanout 与 Return rows 合同；
只改变静态 packet 几何和相应有界分配，检查并行收益与提交开销的权衡。

EP16/8K L32/B512/D4 快筛：Q16 4619.52 us，Q8 4394.67 us（对照 Q32
AD 4757.62 us）；32 hashes 均与冻结 V3 一致。registered bytes/rank 分别
71946496、38547712，对照 Q32 138977536。更小 Q 让目录变大，但 payload
slot 缩小，净减少注册内存。候选目录 `build/static_v4_ring_q16/`、`..._q8/`。

Q8 的 B256/1024/2048、D4 快筛分别 5066.99/4662.85/5321.04 us，均不如
B512，暂不扩大窗口。这些是筛选而非正式矩阵，期间第一节点有 CPU 编译。

### AL：在本地 direct Return 基线上重测宽 FC1（验证中）

AD 已消除 local C 的二次搬运，因此重新检查 AC 的物理 FC1 N256 候选；
除 FC1 N 外，通信、FC2、stages、数值合同均与 AD 相同。独立 AF 的
packed S-N128 结果只作为候选依据，仍以融合路径实测决定取舍。

第二节点 EP8/8K 正式 10/20/20，B256/512/1024/2048、D4 的 p50 为
1828.55/1672.48/1775.38/1964.17 us；32 hashes 一致，该节点没有并行
CPU/GPU 工作。加宽 N 在小窗口降低任务并行度，增大 B 也未优于 AD 的
1682.80 us，暂不保留 N256。原始样本 `build/ring_direct_n256_remote_ep8_*.log`，
候选 `build/static_v4_ring_direct_n256/`；生产 prefill 仍 N128。

### AM：数据就绪后再查询 NIC credit（验证中）

TX owner 先检查不可变 descriptor、source reservation 或 C-ready，只有
payload 能立即打包时才执行 `can_send`。仍同时要求 remote consumed credit
和本地 NIC completion，不提前访问或覆盖 TX payload；减少等待 FC 时的
GIN counter/signal 查询。以 Q8/B512/D4 和原 Q8 单独比较。

8K 快筛 4366.35 us，16 hashes 与 V3 相同；相对 Q8 原 4394.67 us 的
差异不足 1%，暂不宣称独立性能收益。此变更不添加状态或分支，继续放入
Q8 候选走正确性和后续正式 A/B；并未放松双 credit 条件。

AK/AM 组合候选已通过 EP16 B512/D4 的 PyTorch 四规模及全部 edge cases，
两节点共 416 pass records，包括 20K hotspot-no-drop eager/graph、mixed
decode/prefill、unequal/empty、route-changing graph、共享 workspace 和
post-hotspot 复用；B16/D1 的 T1/64 + edge cases 另有 272 records 通过。
日志 `build/ring_q8_{full,tiny}_torch_rank{0,1}.log`。
现有 GPU plan 测试改用同一个 `kNetworkPackets` 配置，以 Q8/L32 验证
31 source/packet + 6 destination cases（含 1/2/3 nodes），不增加入口。
EP16 B16/D1/T64 的 rank1、rank9 memcheck 均 0 errors；其他 rank 未插桩，
不能称为全部 GPU 的 sanitizer 覆盖。正式性能矩阵待补。

### AN：Reduce 每 token 缓存有序 contribution 地址（验证中）

AD 的新增单机 eager timeline（非正式 graph 成绩）有 73637 events、无
overflow。rank0 的两行 local Pull p50 6.848 us、Reduce/token p50
16.896 us；按 132 CTA 平均的累计活跃时间分别 865.95、1044.84 us，
FC1 math/epilogue 为 266.60/96.01 us，FC2 为 110.25/87.32 us。
这些并发区间不能串行相加，但表明 Reduce payload 本身也值得优化。

复用 `reduce_token_payload`，每 warp lane 预先计算一个有序 contribution
地址，hidden 循环通过 shuffle 取地址，不再为每个向量重读 combine_order
和 compact/dense 行映射。top_k<=32 已由现有 launch 边界保证；保持 FP32
逐 route 累加顺序、原 BF16 conversion、ready acquire 和退休条件不变。
不新增 Reduce 实现或单测入口；候选以原完整参考检查验证。

首版 warp 分摊地址/shuffle 候选 EP8/8K 快筛 1792.37 us，8 hashes
一致，慢于 AD 1682.80 us，已撤回。把查表换成每个向量的两条 32-bit
shuffle 并没有改善该工作负载；不保留这份无收益的通用改动。

### AO：窗口内固定 32-Mblock compute wave（验证中）

在 AN 首版冻结基线上，仅把大窗口分成至多 32 Mblocks 的静态 compute
wave；每个 wave 仍先本 owner 的 FC1，再 FC2，保持全局 residue，不能
跨 admission window。避免大窗口中 FC2 等整个窗口的本 owner FC1。
小窗口执行顺序不变，需在 B2048 等较大窗口单独对照 AN，不能直接和
含不同 Reduce 的 AD 比较。候选 `build/static_v4_ring_compute_waves/`。

第二节点 EP8/8K/D4 快筛，B1024/2048/4096 的 AN 对照为
1899.47/2130.88/2837.10 us，AO 为 1889.74/2058.98/2559.28 us。
较大窗口有改善，但仍不如 AD B256；不保留 compute-wave 改动。该结果
不否认它对其他更大 GEMM shape 的潜在用途，仅说明当前主矩阵不采用。

### AP：top-k=4 Reduce 地址缓存与循环展开（验证中）

同一个 `reduce_token_payload` 的模板内部，在 top-k=4 时一次缓存四个
ordered contribution 指针，并展开四次 load/add；不再每个 hidden 向量
广播指针。其余 top-k 保留原通用循环，不分配大指针数组。仍是同一份
FP32 顺序累加与 BF16 conversion，未改变 payload/slot 协议。
生产调度恢复到 AD 的整窗版本，单独比较 AP 的端到端收益。

AP 正式矩阵：1620 MHz、10 warmup/20 samples、graph batch20，EP8 在
第二节点单机、B256/D4；EP16 双机 B512/D4、Q8/L32。测量节点无 CPU
编译或其他 GPU 任务，全部输出 hash 与冻结 V3 一致。这里 EP16 包含 AK/AM
的网络改进，不能把其全部收益归因于 AP；AP 的隔离单机长文收益约 1%–1.5%。

| T/rank | V3 EP8/us | AP EP8/us | V3 EP16/us | AP EP16/us |
|---:|---:|---:|---:|---:|
| 1 | 300.374 | 101.043 | 682.170 | 306.398 |
| 64 | 370.218 | 149.442 | 824.659 | 397.850 |
| 8192 | 1489.09 | 1659.53 | 3353.94 | 4405.93 |
| 20480 | 3294.90 | 4005.92 | 7099.53 | 10226.1 |

短序列领先，长序列仍未达标。原始样本 `build/ring_reduce_topk_{remote_ep8,ep16}_t*.log`；
候选 `build/static_v4_ring_reduce_topk/`。EP16 四规模 + 全部 edge cases
共 416 PyTorch pass records；EP8 B256/D4 四规模 + edges 共 208 records；
EP8 B16/D1 的 top-k=8、T1/64 + edges 共 136 records，验证通用 fallback。
三组均 exit0，无参考检查失败。未增加单测入口。

### AQ：Reduce 直接读取 Return RX（快筛无收益，撤回）

删除 RX→contribution scatter，以 source-local immutable `ReturnBinding`
把每个 route 绑定到 contribution 或本地 Return RX 行、ready word、generation
和可选的 RX consumed counter。Reduce acquire 对应 ready，保持原有 FP32
combine 顺序，全部 payload loads 完成、warp join 后，每个 remote route
退休一个 RX row；ACK 仍等待该 packet 的精确 desc.rows。payload 生命周期
延长至 Reduce，而不再只到 scatter。原 scatter warps 改为额外静态 Reduce
residues，source contribution 的窗口退休合同不变。

为消除 token 等待尚未收到的其他 route 所造成的 RX 容量环，网络深度至少为
`max(8, ceil(ceil(B*max_top_k/Q)/L))`，使每个 channel 都能放下一个完整
source window 的最坏 Return packets；这只依赖固定 B/K/Q/L，不随 T 增长。
同 channel 按 window 单调发送，后窗包不能占用尚未到达的前窗包位置；最老
窗口全部 route 最终可到达，Reduce 释放它，进而释放 source contribution。
Q8/L32/B512/K4 仍 depth8；大 B 或小 L 的注册内存会增大，须如实记录。
plan/allocate 地址不跨调用保存，所有 reader/ACK/local completion drain 后
仍做 invocation metadata barrier。TX 仍保留 remote credit AND local completion。

Q8/L32/B512/D4 快筛 T64=407.296 us、T8192=4490.78 us，32 hashes 与
冻结 V3 一致；8K 不如 AP 正式 4405.93 us。移除 scatter 没有带来收益，
同时增加 immutable binding 表和 RX 生命周期约束，已撤回全部生产改动，
未投入完整矩阵/热点/sanitizer 验证，不将候选推导称为完整正确性证明。
恢复原 RX scatter、channel consume_cursor 和 depth8。冻结 native/lib/extension
与补丁位于 `build/static_v4_ring_direct_rx/`（`change.patch`），可复现实验。

### AR：增加静态 Reduce warp 配额（无收益，撤回）

AD 后本地 Return 只发布 ready，因此尝试将其限制为 32 个 warp，其余原
Return warp 静态划给 Reduce。单机 Reduce workers 从 131 增至 231；
Q8/L32 双机从 99 增至 135。所有任务仍按固定 residue 覆盖，无动态领用，
不增加 buffer、counter 或 barrier。8K 快筛 EP8/B256/D4 为 1657.23 us，
EP16/B512/D4 为 4425.20 us，未优于 AP 的 1659.53/4405.93 us。
故撤回生产改动，不为无收益候选增加完整验证负担。
候选 native/lib/extension：`build/static_v4_ring_reduce_roles/`。

### AS：纯 GEMM route 夹具与通信字节审计更正

修正独立 GEMM 的 balanced 公式，与 native 默认 stride53 的主矩阵一致。
T1 的 owned expert counts：world8 从 `[1,1,1,1,0,0,0,1]` 改为
`[1,0,0,1,0,1,0,1]`，world16 从 `[2,2,2,1]` 改为 `[1,1,0,1]`。
T64/8192/20480 逐专家 counts 完全一致，因此此前这些规模的测量不作废。
修正后的 T1 balanced p50/us（N 为有效 S 宽度，K128/stages2）：

| N | world | packed FC1 | separated FC1 | packed chain | separated chain |
|---|---:|---:|---:|---:|---:|
| 64 | 8 | 22.720 | 27.424 | 32.784 | 37.408 |
| 64 | 16 | 22.640 | 27.264 | 32.608 | 37.760 |
| 128 | 8 | 35.248 | 36.928 | 45.456 | 47.360 |
| 128 | 16 | 35.040 | 36.832 | 45.312 | 47.056 |

N64 在第一节点 GPU0，N128 在第二节点 GPU0；各节点无其他 GPU workload。
16 用例完整输出/replay 校验通过，960 对采样时钟全部 1620:1620。
日志、构建命令与 hash 位于 `build/static_gemm_native_route_n{64,128}_s2_k128/`。
该更正不改变当前 packed 优于同宽 separated 的结论；未声称融合路径获得收益。

另在第二节点 GPU0 补测 decode 候选 packed N32/K128/stages3 的主矩阵 T1：
world8 FC1/chain 为 18.048/28.544 us，world16 为 17.888/27.968 us。
四用例完整输出及 replay 校验通过，240 对时钟均 1620:1620，仍支持窄 N
作为 decode 候选。记录于 `build/static_gemm_native_route_n32_s3_k128/`。

另外以 eager audit 核对 EP16/T8192 的实际 payload：V3 与 AP 都是每 rank
Dispatch 32 MiB、Return 64 MiB；V3 Put 数为 128/256，AP Q8 为 1024/2048。
因此 V4 并非通过少传数据获得短序列收益，长序列也不能仅凭 sysfs 的单设备
200 Gb/s 标称速率断言已经线速：V3 传相同字节却更快，仍须分析有效通信吞吐。
V4 的 raw_remote_route_bytes、gin_metadata_bytes 目前未接 telemetry，日志中的
零不表示没有这类流量。审计日志 `ring_{v3,reduce_topk}_ep16_t8192_eager_audit.log`；
eager 含单次样本，仅用作计数审计，不替代 AP 的正式 Graph 性能表。

### AT：本地 Pull 批量退休（验证中）

保留单 issuer、两个 TMA staging slot，不增加共享内存；节点内 Pull 从每两行
full store wait 改为同一 Mblock 内每八行 wait。中间通过现有双 staging 流水
继续发射，完成整个 batch 后合并一次 X-full release credit。M128 能整除
batch，不跨 Mblock 或窗口；有效尾行仍补齐 padding credit，网络路径暂保留
两行 batch，也合并本 batch 的本地行 credit。删除发布阶段的重复元数据遍历。
不提前复用 staging，不以 read-completion 代替全局 store-completion。

第二节点单机 EP8、1620 MHz、10 warmup/20 samples/batch20、D4：

| 版本 / B | T1/us | T64/us | T8192/us | T20480/us |
|---|---:|---:|---:|---:|
| 冻结 V3 | 300.374 | 370.218 | 1489.09 | 3294.90 |
| AP / 256（原正式基线） | 101.043 | 149.442 | 1659.53 | 4005.92 |
| AP / 512（同窗对照） | — | — | 1631.58 | 3968.52 |
| 八行 / 512 | 101.122 | 139.627 | 1454.03 | 3585.28 |

同 B512 的 8K/20K 分别降低 10.9%/9.7%，不能全部归因于放大窗口。
8K 已略快于 V3，但 20K 仍慢 8.8%，**未达到全矩阵明显优于 V3 的目标**。
四规模 32 hashes 与同节点冻结 V3 一致，测量节点无其他 workload/编译。
八行的 B256/512/1024 8K p50 为 1640.07/1454.07/1571.82 us；因此后续
单机候选采用 B512。所有值保留实际窗口配置，不把大 buffer 当零成本收益。

继续尝试十六行没有改善：B512 的 8K/20K 为 1544.12/3697.64 us，
B1024 为 1643.56/3628.85 us，故恢复八行。AP/八行/十六行的单机 prefill
entry 均 REG168、STACK64、LOCAL0，未观察到资源计数变化；不能武断归因于 spill。
冻结八行 `build/static_v4_ring_pull_batch/`，十六行 `build/static_v4_ring_pull_batch16/`。
日志 `ring_{reduce_topk,pull_batch}_paired_remote_ep8_b512_t*.log`、
`ring_pull_batch_remote_ep8_b*_t*.log`、`ring_pull_batch16_remote_ep8_b*_t*.log`。

八行 B256/D4 和 B512/D4 的四规模+edges 各 208 PyTorch pass records；
B16/D1 的 T1/64+edges 136 records，均 exit0。rank1 的 B16/D1/T64 Graph
racecheck 为 0 errors/0 warnings，检查了更长 batch 内的双 staging 复用。
网络两行 batch 的聚合 credit 与 AU 一同通过完整 EP16 四规模及 tiny ring
验证（见 AU）；本地八行已分别通过 B256/B512 的完整单机验证。

八行/B512 的 D8、D16 正式容量消融：8K 为 1456.71/1495.20 us，20K 为
3509.29/3538.17 us。D8 对 20K 约降低 2.1%，8K 没有改善；D16 更慢。
8K 的 device pool 分别为 511605440/578845888 bytes/rank，20K 为
689079232/756319680 bytes/rank。暂保留 D4，不把这组额外容量收益混入 AT。

### AU：每 packet 的 ready 语义与 Q8/L16 消融

核对安装的 NCCL 2.31.2 `nccl_device/gin.h`：weak signal 保证绑定的 Put
完成；strong 额外保证同 context/peer 先前所有 Put 完成。当前一个 packet
就是一个 Put，header/payload 一起传输，每个物理 slot 有独立 ready/generation，
后包 ready 不承担前包完成通知，因此只需 `WeakVASignalAdd`。保留接收端
`readSignal` 可见性导入、ACK、远端 consumed credit AND 本地 NIC completion、
terminal drain 与 metadata barrier。若未来拆包，必须重新定义跨 pieces 聚合，
不可直接沿用最后一次 weak signal。

同 Q8/L32/B512/D4 的 EP16/8K 快筛 strong/weak 为 4342.40/4342.82 us，
**没有本机性能收益**；此项只收紧到实际所需的 packet 完成合同，不宣称加速。
两个冻结候选分别为 `static_v4_ring_pull_batch16`、`static_v4_ring_packet_signal`；
其网络 specialization 都是本地两行 batch，单机十六行候选不参与此性能结论。
weak 的 EP16 四规模+edges 416 PyTorch records 均通过，包含热点后复用。
B16/D1 的 T1/64+edges 另有 272 records 全通过，覆盖多代 wrap、变更输入的
Graph replay 与共享 workspace。未引入新测试入口。AT/AU 合并后的源码保留
单机八行、网络两行 batch；上述 weak library 的单机十六行是已排除的性能
候选，不将它的 EP8 性能冒充为合并后的基线。

将 Q8 的 lane 从 32 减至 16，固定 owner/context 数随之减少，注册内存从
38547712 降至 21668096 bytes/rank，但 8K 退化至 7325.07 us，故不保留。
该参数的已有 GPU plan 检查通过 31 source/packet + 6 destination cases，
包含三节点几何；不把三节点 plan 通过声称为三机实测。
冻结 `build/static_v4_ring_packet_signal_l16/`，生产调优继续 Q8/L32。

辅助 GIN link benchmark 使用现有单 context 入口，4 MiB×32 的 rank-max
平均有效吞吐 23.843 GB/s/rail，8 rails 总计 190.742 GB/s；小包发送明显更低。
该测试既非完整算子，也非多 context 的带宽上限，不据此解释 V3/V4 的全部差距。
原始日志 `build/ring_gin_link_bandwidth_audit.log`，其中历史字段名 `gbps`
按 bytes/ns 计算，单位实际为 GB/s 而非 Gb/s。

### AV：窗口内复用已确认的 source reservation

每个 Pull owner 用一个 32-bit mask 记录本窗口已 acquire 的 source reservation；
source group 至多 16 个。只缓存成功的确认，不跨窗口保存。该窗口不能在本
owner 的行完成 Return/Reduce 前退休，所以后续行可以复用该 acquire 链。
同一个 batch 内的 Mblock admission 也只检查一次；整批的 X credit 尚未发布，
其 X/S/C slot 不可能被复用。没有省掉第一次 acquire 或改变最终 release。

第二节点 EP8/B512/D4、1620 MHz、10/20/20：8K/20K 为
1293.96/3121.93 us，相对 AT 的 1454.03/3585.28 us 降低 11.0%/12.9%，
相对冻结 V3 降低 13.1%/5.2%。16 hashes 一致，完整四规模+edges 208
PyTorch records 与 B16/D1 的 T1/64+edges 136 records 通过。
窗口复测 B256 为 1667.91/4052.52 us，B1024 为 1373.18/3029.93 us。
8K 仍选择 B512；20K 的 B1024 比 V3 快约 8.0%，但必须另计其容量代价。
冻结 `build/static_v4_ring_pull_reservation/`，日志
`ring_pull_reservation_remote_ep8_b*_t*.log`。

该候选曾同时把网络路径的本地 batch 也放大至八行，EP16/8K 快筛
4454.77 us，无收益。两节点的 source round-robin 顺序为八个本地 source
与八个远端 source；八行 residue、100 个 Pull owners 会让一半 owners 的
本地部分持续为空，削弱并行度。因此网络恢复两行，保留 reservation 确认缓存。
恢复后的网络候选在 AW/AX 继续验证，不把单机收益直接填为多机收益。
最终清理构建 `build/static_v4_ring_reserved_heads/` 仅保留 reservation 缓存，
网络 batch=2、本地 batch=8、原控制面 barrier 和原 W1 hint。完整 EP16
B512/D4 四规模+edges 已通过 416 PyTorch records；正式四规模结果如下。
native SHA256 `2620209cda0918a5068bc711e5317d3f15d521933f4498520ad913e52d450382`，
library SHA256 `25bc578ae4615ae88272227736a41a9208d7678c781489bc3d2a150b6d85f5e8`。

随后 EP16/Q8/L32/B512/D4 的正式 1620 MHz、10/20/20 四规模为
306.454/398.174/4329.17/10196.3 us，64 hashes 与冻结 V3 一致。
测量时两个节点均无其他 CPU/GPU 工作。多机长序列仍未达目标；
日志 `ring_reserved_heads_ep16_t*.log`，commit `3f07d96`。
最终构建的第二节点单机 EP8/T1、T64 为 101.029/137.227 us，16 hashes
与冻结 V3 一致；日志 `ring_reserved_heads_remote_ep8_t{1,64}.log`。
上述 EP8 长序列使用 `pull_reservation` 构建，两者的单机 specialization 相同；
它们的网络 batch 不同，不混用其 EP16 测量。

### AW：FC1 权重 EVICT_LAST（无收益，撤回）

仅将静态 FC1 W1 TMA 的 L2 hint 从 EVICT_NORMAL 改为 EVICT_LAST；FC2
本来就用 EVICT_LAST。M/N/K、寄存器、权重布局、通信与精度均不变。
相同第二节点 EP8/B512/D4、10/20/20，8K/20K 为 1297.84/3116.38 us，
相对 AV 没有稳定收益，恢复原 hint，不保留调参开关。
冻结 `build/static_v4_ring_weight_cache/`。另外恢复网络两行 batch 的构建
`static_v4_ring_weight_cache_net2`，EP16/8K 快筛 4328.70 us；此数包含网络
batch 恢复，不作为 cache hint 的单独收益。

已有单 rank NCU 临时 wrapper 成功以单次采集测到 AT EP8/8K 约
1162174464 read bytes、375717376 write bytes；AW 为
1190856704/377015040 bytes，未见预期的大幅流量下降。两者还含 AV 的
reservation 改动，且 profiling 延迟使 peer 的同步时间失真；只保留计数诊断，
不把 1.790/1.570 ms 的 profiler kernel duration 放入正式性能表。
文件 `ring_{pull_batch,weight_cache}_ep8_ncu.csv`；没有新增项目单测。

### AX：仅对活跃 GIN contexts 做控制面 fence（无明确收益，撤回）

网络两行 batch、AV reservation 缓存，W1 hint 已恢复。保持所有逐 slot 双重
credit、ready、ACK 和精确 terminal drain 不变。metadata Put 按 256-KiB
chunks 分给 contexts；其后只 fence 实际分到 chunk 的 context，其他 context
在本阶段没有 Put。invocation 收尾已经完成全部 payload/readers/local completion
及 terminal credit，因此只需一个 rail rendezvous 再接节点内 join，保护下一次
metadata 覆盖，不再对 96 个已 drain 的 data contexts 重复 Put fence。
full-arena 路径仍 fence 原来的所有 contexts，未顺带改变其协议。
候选 `build/static_v4_ring_context_fence/`。EP16/B512/D4 快筛 T64 为
386.096 us，8K 为 4328.80 us；8K 与网络两行对照的 4328.70 us 无差别。
没有足够收益支撑新增调用参数与 fence 分支，已完整撤回三个生产文件的改动。
特别注意 `after_combine` 起点在 entry 的 CTA0、grid join 之前，span 含其他
CTA 尚未结束的工作，不能把整个 span 都归因于 GIN barrier 本身。

### AY：FC2 直写预留 Return TX（候选归档，暂不合入）

将 AD 的节点内直接写 contribution 扩展到远端：FC2 直接写本节点 rail 的
Return TX，发送 owner acquire 原有 C/FC2 completion 后直接 Put，省去 C→TX
打包。没有引入 fragment 引用图或 atomic expected count。每个 channel 新增
owner 私有 reserve cursor 与 release.sys reservation head，仍在原 64-byte
`NetworkChannel` 内；不可复用 slot 前仍检查远端 consumed AND 本地 NIC completion。
整窗口 admission 用 source expert interval 的最后一个 lane packet 检查 reservation；
TX owner 独立于 C-ready 提前预留最多 depth 个 packet，以免形成准入循环依赖。

RingRow 预绑定本地 contribution/远端 TX 指针，16→24 bytes。网络 depth 为
`max(8, ceil(ceil(B*capacity_top_k/Q)/L))`，确保最坏 source Return 窗口能整体
预留；Q8/B512/K4 仍为 8，B1024 或 K8 会增加。source/peer/node 枚举沿用
参数化拓扑，不限两节点。候选仍分配已不再使用的 C payload，尚未清理其 host
descriptor；下面内存数字如实包含它，不能声称已完成资源优化。

1620 MHz、Q8/L32/B512/D4 快筛：EP8/8K 1286.13 us，EP16/8K 4637.31 us，
输出 hashes 与 AV 一致。后者比 AV 的 4329.17 us 慢，不能将少一次拷贝视为
性能收益。EP16 全四规模+edges 416 records、小 B16/D1 的 T1/64+edges
272 records、K8/B512/D4 的 T64/8K+edges 352 records 均通过，覆盖多代 wrap、
变更输入 Graph、共享 workspace、热点 no-drop。尚未做此候选的 sanitizer。

重新消融 packet/window：EP16/8K、B512/D4 的 Q16/Q32 快筛为
4230.27/4312.26 us，注册区为 71946496/138977536 bytes/rank；Q8 为
38547712 bytes/rank。Q16 的 B256/B1024 快筛为 4680.10/4163.68 us。
Q16/B1024/D4 正式 10/20/20 的 8K/20K 为 4191.25/9425.64 us，32 hashes
与冻结 V3 一致。相对 AV 最佳 Q8/B512 的组合收益为 3.2%/7.6%，仍落后于
V3 的 3353.94/7099.53 us。8K compute pool 为 1591179456 bytes/rank，
registered 为 71921408 bytes/rank；不把协议、Q、B 的组合收益分摊为单项收益。

先恢复 AV 再独立检查 GIN owner 的资源共享模式，避免把后续更简单的优化混入
本候选。代码归档 `build/static_v4_ring_direct_tx/change.patch`、`return_tx.cuh`，
详细地址/进度推导 `build/direct_return_tx_plan.md`；生产源码当前不含此协议。
冻结 native SHA256：Q8 `a23bab9c83e5f37d550bde55a91ca6757a35cf8573fbfe10df79f73fe4f67790`，
Q16 `fffe9b62e0895424e96d823fb3c8b558ae2854d97c72541689d2afedfaeeb627`，
Q32 `e34f2be383d4c1076e99e5f3e10a7a69205ad7754896b700ab5e21579679e679`。
日志 `ring_direct_tx*_{quick,formal}.log` 与 `ring_direct_tx_*torch_rank*.log`。

### AZ：静态 GIN owner 使用线程独占资源模式

只将 ring `Endpoint` 的 GIN resource sharing 从默认 GPU 改为 THREAD。
Dispatch TX、Return TX、RX/ACK 各 context 在 data plane 均只有唯一 lane-zero
提交者；host 已要求 exclusive contexts，metadata 阶段与 data plane 由原有
barrier 分隔。未修改 full-arena/V3、metadata barrier、ready/ACK、Put release
scope、remote consumed AND local NIC completion 条件，也没有新增状态或容量。

安装的 NCCL 2.31.2 GDAKI header 将 THREAD 映射到 DOCA EXCLUSIVE，允许
owner 私有读写，省掉 GPU 共享模式的队列锁/原子更新。现有 `gin_link_bench`
开启 verbose 后实际报告 `gin_type=3, railed_gin_type=3`，即 GDAKI；不再仅凭
平台猜后端。日志 `build/ring_gin_backend_audit.log`。这不是改变 Put 内存序，
也不是用户留待以后消融的本地 NIC credit 检查。

Q8/L32/B512/D4 的 EP16/8K 快筛 4063.90 us，相对 AV 4329.17 us 约改善
6.1%，输出 hashes 一致。完整四规模+edges 416 records、
B16/D1 的 T1/64+edges 272 records 均通过。rank1/9 的 B16/D1/T64 eager
memcheck 与 racecheck（11 rounds）均 launcher exit 0、0 errors/0 hazards。
首次误设 rounds=2，在目标算子前被入口拒绝；其零错误日志不计入验证。
日志 `ring_exclusive_gin_{memcheck,racecheck}_ep16_t64_r11.log` 与对应带 PID
的 sanitizer 日志；不声称全部 16 ranks 的 sanitizer 覆盖。

冻结 `build/static_v4_ring_exclusive_gin/`，native SHA256
`3dbaa607c5c7d55ee20185c47fba47bc6ff93fac9f44f99b7e1ab93f64751c10`，
library SHA256 `f0f576769af90267603b4b9a88cf6627decdd8954a72571ea05dd52c3dab5ebb`。

正式 EP16、1620 MHz、10/20/20，Q8/L32/B512/D4 不变：

| tokens/rank | AV/us | AZ/us | 相对 AV 延迟降低 | 冻结 V3/us |
|---:|---:|---:|---:|---:|
| 1 | 306.454 | 301.467 | 1.6% | 682.170 |
| 64 | 398.174 | 393.101 | 1.3% | 824.659 |
| 8192 | 4329.17 | 4057.57 | 6.3% | 3353.94 |
| 20480 | 10196.3 | 9462.10 | 7.2% | 7099.53 |

64 rank hashes 与冻结 V3 一致；单机 specialization 没有变化。两个节点测量时
无 CPU 编译或其他 GPU 测试，SM 1620、活跃显存时钟 3201 MHz（远端空闲时
会降到 2201）。AV/V3 是同日历史测量，AZ 日志
`build/ring_exclusive_gin_ep16_t*_formal.log`。
随后同条件 8K、10/20/20、无其他 CPU/GPU 工作的 ABBA 为
AV 4332.55 → AZ 4062.22 → AZ 4059.43 → AV 4332.47 us，约 6.3% 收益稳定；
日志 `ring_exclusive_gin_abba_{av1,az1,az2,av2}.log`。
长 EP16 仍分别比 V3 慢 21.0%/33.3%，未达到最终性能目标。

### BA：远端 Pull 单遍读取路由元数据

在 AZ 基线上，把一次 packet 的两遍遍历改为每 32 个 route 一批：每个 lane
保留自己的 RouteLocation，lane0 搬完本批、完整 global wait 后，warp 直接
使用保留的地址发布 X-ready。整包所有批次完成后才累计 RX consumed，NIC
ready acquire、slot generation 检查与原有 TMA staging 生命周期不变。
Q8/K4 恰好一批，较大 K/Q 不再重复读取 metadata，但可能多一次批间 drain，
不能仅凭少了读取就声称所有配置更快。

Q8/L32/B512/D4、EP16/8K 快筛 4015.09 us，对照 AZ 4063.90 us，仅约 1.2%。
完整四规模+edges 416 PyTorch records 与 B16/D1/K8 的 T1/64+edges 272
records 通过。网络 prefill 入口两者均 REG168/STACK440/LOCAL0；这是静态
resource usage，不把非零 stack 一概称为 spill。组合正式性能及 sanitizer 见 BB，
不把 BB 的全部收益归给 BA。
冻结 `build/static_v4_ring_pull_onepass/`：native SHA256
`50f8ba8ab2267554a0a752e4b15ceeb68385f8dafc973ada72b6a359e818d2a5`，library
`49a2305ff1b66d4e71305b9a00c9ec86caa59573d8195c43b62d3befd1c7e303`。
日志 `ring_pull_onepass_ep16_t8192_quick.log`、
`ring_pull_onepass_{full,tiny_k8}_torch_rank*.log`。

### BB：Pull 复用整窗口准入

在 BA 上增加 invocation-local `RingState.admitted_windows`。准入 warp 发布
本窗口全部 ComputeSlot generation、经原有 warp join 后，由 lane0 release.device
发布 window+1。Pull 每窗口 acquire 一次，节点内 batch 和远端逐 route 不再
重复检查 generation；网络 warp 的其余 lanes 通过原有 syncwarp 导入完成链。
该窗口在本 owner 的 X 行发布完成前不能复用所需物理 slot，因此可缓存首次
确认。source reservation、数学任务的 generation/X/S 等待、Return 完成链和
所有 NIC credit 条件不变。head 只在下一次 quiescent prepare 中清零。

新增 8-byte 标量未改变实际 pool 大小：EP16/8K Q8/L32/B512/D4 的 compute
仍 916410560、registered 38547712 bytes/rank。快筛 3996.43 us，对照 BA
4015.09 us，只有约 0.5%；16 hashes 一致，不将其称为大幅网络加速。
第二节点单机 EP8、B512/D4、1620 MHz、10/20/20 四规模为
101.314/136.600/1286.800/3081.260 us；测量节点无其他 CPU/GPU 工作，另一
节点同时做独立 EP8 数值测试，不是两节点 EP8。

完整 EP8/EP16 四规模+edges 分别 208/416 records，通过热点、mixed profile、
unequal/empty、变更输入 Graph 与共享 workspace；EP16 B16/D1/K8 的
T1/64+edges 272 records 也通过。rank1/9 同时插桩的 B16/D1/T64/K8 eager
memcheck、racecheck（11 rounds）均 launcher exit 0、0 errors/0 hazards。
这组 sanitizer 验证包含 BA 的分批 X-ready 发布，不是只检查新增 head。

冻结 `build/static_v4_ring_admission_head/`，native SHA256
`dd9e65f9de563ba9703dd9162112046be1e296fd1786ca3f96405b3a4f6e630a`，library
`a74b8fa368c04235707c29c9d8bce7c0cd6bda737eeacc0c3fb3a0ae8d55df12`。
日志 `ring_admission_head_*`。EP16 同条件、1620 MHz、10/20/20 正式四规模为
297.082/374.875/3992.340/9406.510 us，测量时无 CPU 编译或其他 GPU 工作。
相对 AZ 的 301.467/393.101/4057.570/9462.100 us，BA+BB 的组合延迟降低
1.5%/4.6%/1.6%/0.6%。EP8 与 EP16 合计 96 rank hashes 与冻结 V3 一致；
长 EP16 仍比 V3 慢约 19.0%/32.5%，不宣称已达最终目标。

随后在同一第二节点、1620 MHz、10/20/20 独立复测 B1024/D4：单机 EP8
8K/20K 为 1353.54/2984.41 us，16 hashes 与上述 B512 及冻结 V3 一致。
对应 device pool 为 813600192/990859456 bytes/rank，不能把扩大窗口的收益
算成协议优化。8K 仍以 B512 较好；20K 的 B1024 比 V3 降低约 9.4%。
测量节点无其他 CPU/GPU 工作，另一节点仅编译后续候选。日志
`ring_admission_head_remote_ep8_b1024_t{8192,20480}_formal.log`。

### BC：将部分 local Return warp 分给 Pull（无收益，撤回）

在 BB 基线上，单机把一半 warp1 改为 Pull，固定 workers 从 132→198、
Return 132→66；双机保持 32 Dispatch TX/32 Return TX/32 RX scatter/
32 NIC progress，剩余 local Return 一半转 Pull，Pull 100→134、Return
68→34。两类 Pull 使用原有各 warp 的私有双 staging，无新增 SMEM、动态领用
或新同步协议；保留独立驻留的 Return owner。

Q8/L32/B512/D4、8K 快筛 EP8 1504.16 us（本机，BB 约 1287 us），
EP16 4028.05 us（BB 3996.43 us），均无收益。仅声明性能候选，不声称完成
这组角色划分的完整数值/sanitizer 验收。prefill REG168、local/network
STACK64/440、LOCAL0 均未变，不能凭退化推断 spill。已恢复 BB 的角色配额。
冻结 `build/static_v4_ring_pull_share/`，native SHA256
`a382221b3ba7fe950af27a651ee18cae0439b732854fb898f31aa5cc47be018c`；日志
`ring_pull_share_ep{8,16}_t8192_quick.log`。

### BD：Return ready 发布到发送 rail

原 TX 为确定 C 是否 ready，持续跨 NVLink 读取 expert GPU 的 generation、
fc2_done、fc2_target。现在复用固定 local Return warp 的 FC2 acquire/system
fence，在普通 device pool 的 rail-local `[local_destination, C_slot]` 镜像
发布 generation，TX 改为读取本 GPU 的镜像。逆向 route 表给出实际 interested
rails，warp 去重后每 rail 发布一次；不广播未命中的 rail，不增加 expected count。
镜像只在 pool 初始化清零，随原 C generation 跨 invocation 复用。

remote-only block 不再跳过 local Return owner：网络 TX 必须先取得它的发布，
所以不能在发布前退休 C。每个 rail 仅一次 store，最后一个 C reader 的行 credit
保护当前代；publisher 发布完成后不再访问可复用 C。原 C→TX pack、full global
wait、system C.returned、远端 consumed AND 本地 NIC completion 均不变。
新增元数据为 local_world×depth×8；EP16/B512/K4 为 33280 bytes/rank，registered
pool 仍为 38547712 bytes/rank。prefill local/network REG168、STACK64/440、
LOCAL0 均未变。

串行/warp 发布 8K 快筛分别 3806.82/3804.26 us，差别不足以单独宣称收益。
串行候选完整 EP16 四规模+edges 416、B16/D1/K8 的 T1/64+edges 272 records
通过。保留的 warp 版也通过这两组数值检查；rank1/9 同时插桩，B16/D1/T64/K8
eager 11 rounds 的 memcheck/racecheck 均 launcher exit 0、0 errors/0 hazards。
不声称 sanitizer 检查了所有 16 ranks。

正式 EP16、1620 MHz、Q8/L32/B512/D4、10 warmup / 20 samples / batch20，
两个节点无 CPU 编译或其他 GPU 测试：

| tokens/rank | BB/us | BD/us | 相对 BB 延迟降低 | 冻结 V3/us |
|---:|---:|---:|---:|---:|
| 1 | 297.082 | 293.669 | 1.1% | 682.170 |
| 64 | 374.875 | 378.358 | -0.9% | 824.659 |
| 8192 | 3992.340 | 3798.200 | 4.9% | 3353.940 |
| 20480 | 9406.510 | 8836.800 | 6.1% | 7099.530 |

64 rank hashes 与 BB/冻结 V3 一致。BB/V3 是同日历史对照，尚未将 ABBA
交错重测纳入此表。长 EP16 仍比 V3 慢 13.2%/24.5%，未达最终目标。
冻结 `build/static_v4_ring_return_ready_warp/`，native SHA256
`397119ee194a3cb57ad1bc97406832d1ae0a714a3a5438f76fb2d59c5622809e`，library
`979daa1ad5c0587d94215c80b45b0acd41cb6b64a1ef54453faa07e944b64812`；
日志 `ring_return_ready_warp_*`。串行候选归档 `static_v4_ring_return_ready/`，
native `f4182c4dd19a3ae0124e0f44ceb66ac7ddcc2d011b3dccc4b8c1db3d8449a305`。

单机 specialization 无新增通知逻辑。仍完成 EP8 四规模+edges 208 records 回归；
第二节点单机 EP8 的 10/20/20、1620 MHz、B512/D4 正式结果为
101.835/137.496/1286.880/3067.780 us，32 hashes 与 BB/V3 一致。第二节点
测量时无其他 CPU/GPU 工作，第一节点独立跑 EP8 数值测试及编译下一候选。
相对 BB 的差异很小，不给本次网络修改归因单机收益。

顺带核实 rank1/9 实际 activation devComm 为 **1 connection / 96 contexts**，
日志 `ring_return_ready_gin_gpulingjun*`。connection 与 context 不是同一层资源；
本项未改变它们。NCCL 2.31.2 的
[connection 构建代码](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/gin/gin_host.cc)
默认按本地 GIN device 数选 connection，再将请求的 contexts 分配到 connections。
不把单 context link benchmark 的带宽直接当成整个融合 kernel 的硬上限。

### BE：GIN connection 数及 Return packet 起点消融（无收益）

在 BD/Q8/L32/B512/D4 上，仅用启动 wrapper 设置
`NCCL_GIN_NCONNECTIONS=2/4`。8K 快筛 3819.62/3810.27 us，与默认 1
connection 的约 3804 us 相当；32 hashes 一致。rank1 的 NCCL 日志确认实际
activation devComm 分别为 2×48、4×24 contexts，非未生效的参数。没有更改
生产默认值，也不把此项称为增加物理 NIC 带宽。日志
`ring_return_ready_connections{2,4}_*`。

另在 packet plan 中，将一个 source/window/remote node 的 Return packet
ordinal 循环移动 `floor(packet_count * source_local_rank / local_world)`。
整个 packet 仍连续、仍在同一完整准入窗口内；各 channel 的 packet 数、credit
单位与静态 ownership 不变，目标是错开不同 rail 的首批 C 读取 GPU。
现有 GPU plan 31 source/packet + 6 destination cases（含三节点）通过，
EP16/8K 快筛 3811.12 us，16 hashes 一致，未显示收益，已撤回。没有额外
开展这份无收益候选的完整数值/sanitizer 验收。
归档 `build/static_v4_ring_return_rotate/change.patch` 和 native，SHA256
`9d8e10076d88aff7e06dd6058ad416fab72ec3ab3307bbe5afa383959bbc240b`；日志
`ring_return_rotate_ep16_t8192_quick.log`。上述快筛期间本机有 CPU 编译，因此
只用于筛选，不代替正式干净环境测量。

### BF：BD 后重新选择 packet Q

保持 BD 协议、L32/B512/D4，只更改编译期 Q（Dispatch 和 Return 同时改变）。
8K 快筛 Q16/Q32 分别 3526.05/3726.38 us；Q16 的 B1024 为 3569.17 us，
未好于 B512。48 hashes 一致。Q16/Q32 的 registered pool 分别
71946496/138977536 bytes/rank，对照 Q8 的 38547712；不是免费减少消息数。

Q16 完整 EP16 四规模+edges 416 records、B16/D1/K8 的 T1/64+edges 272
records 通过。rank1/9 的 B16/D1/T64/K8 eager 11 rounds memcheck/racecheck
均 launcher exit 0、0 errors/0 hazards。Q32 只完成性能快筛及 native hashes，
没有完成同样的全面验证。冻结 Q16 `build/static_v4_ring_return_ready_q16/`，
native SHA256 `8c4674cb8097f0a9d6db06f9306e76da8ce9b010dc8e0c431eb335d032807587`，
library `cea02ff33bb189f1e6fa768481630cfcf487931e82851b94444a09a03063e4ff`。
Q32 native SHA256 `e64464861eb2391e789e0990117c42178f050ef644a2c5baeb86060aa4b1b86c`。

Q16 在两个节点无其他 CPU/GPU 工作、1620 MHz、10/20/20 的正式 EP16
四规模为 **293.570/432.442/3533.150/8181.340 us**，64 rank hashes 与 Q8/冻结
V3 一致。8K/20K 比 BD/Q8 降低 7.0%/7.4%，仍比 V3 慢 5.3%/15.2%。
T64 反而比 Q8 慢约 14.3%，尚不作为全长度统一配置；也不把 Q16 的网络
收益归给 GEMM profile。日志 `ring_return_ready_q16_*`。

### BG：ACK 仅扫描 pending RX slots（无收益，撤回）

在 BF/Q16 上给唯一 NIC progress owner 添加 32-bit pending mask（当前网络
深度 8，字段放入 NetworkChannel 原有 padding）。import 成功后置位，逐 slot
消费达到包头 target、ACK 提交后清位；不改变独立乱序退休或终端 drain。
EP16/8K/B512/D4、1620 MHz 干净环境快筛 3573.22 us，16 hashes 与 BF 一致，
未优于 BF 的约 3533 us。未开展完整数值/sanitizer 验收，已撤回新增状态与逻辑。
归档 `build/static_v4_ring_pending_ack/change.patch`，native SHA256
`676a3661e9f2e251352afcf854d975c8cdea579564543f0014bc1a289639d44c`；
日志 `ring_pending_ack_ep16_t8192_quick.log`。

### BH：Dispatch / Return 独立 Q 消融（保留 D8/R16）

只拆分 Return 的 packet partition、目录容量和 TX/RX stride；固定 L32/B512/D4、
BD 协议。EP16/8K、1620 MHz 快筛如下，均与 BF 的 16 rank hashes 一致：

| Dispatch Q / Return Q | us | registered bytes/rank |
|---:|---:|---:|
| 8 / 32 | 3794.80 | 88731904 |
| 16 / 32 | 3730.62 | 105451776 |
| 16 / 8 | 3788.34 | 55267584 |
| 32 / 8 | 3808.77 | 88793344 |
| 8 / 16 | 3411.25 | 55226624 |

前四项均未优于 BF 的 16/16，D8/R16 的 8K 有正向信号；20K 快筛 7900.77 us，
hashes 一致（20K 快筛同时有本机 CPU 编译，非正式成绩）。现有 GPU plan 的
D8/R32 配置通过 31 source/
packet + 6 destination cases，含三节点；没有新增测试用例，未对这些较慢候选
开展完整数值/sanitizer 验收。D8/R16 已完成下列门禁，保留拆分接口；默认
Return Q 仍随 Dispatch Q，测试配置显式选择 8/16，不把它标为最终统一配置。
归档 `build/static_v4_ring_d8_r32/change.patch`，日志 `ring_d*_r*_ep16_t8192_quick.log`。
D16/R8、D32/R8 构建时曾临时放宽 owner 上限至 44，但两节点 L32 仍选择 32；
不据此声明这些二进制与当前三节点配置一致。源码上限已恢复 32。

D8/R16 精确配置另跑现有 GPU plan，通过 31+6 cases；EP16 四规模+edges
416 records、B16/D1/K8 的 T1/64+edges 272 records、第二节点单机 EP8 四规模
+edges 208 records 均通过。rank1/9 的 B16/D1/T64/K8 eager 11 rounds
memcheck/racecheck 均 launcher exit 0、0 errors/0 hazards，日志
`ring_d8_r16_*check_gpulingjun*`。没有用较慢候选或缓存 admission 的 library
替代此配置的验收。

正式 EP16、1620 MHz、B512/D4、10 warmup / 20 samples / batch20：

| tokens/rank | BF D16/R16 / us | BH D8/R16 / us | 冻结 V3 / us |
|---:|---:|---:|---:|
| 1 | 293.570 | 293.478 | 682.170 |
| 64 | 432.442 | 409.821 | 824.659 |
| 8192 | 3533.150 | 3420.790 | 3353.940 |
| 20480 | 8181.340 | 7937.910 | 7099.530 |

64 rank hashes 与 BF/V3 一致；8K/20K 比 BF 低 3.2%/3.0%，但比 V3 高
2.0%/11.8%，未达最终目标。两节点测量期间无其他 CPU/GPU 工作；BF/V3
仍是同日历史对照，不声称已完成 ABBA。registered bytes/rank 在 8K 从
71946496 降至 55226624，20K 为 62306048。冻结
`build/static_v4_ring_d8_r16/`，native SHA256
`455ef8e06c3d53c2e89b216a6c77e6d9a7da1f45c6cfd859096483ed76debacb`，library
`572bbb1640731a8ed94ae83a25c9b5165fd9fc36652a5fb5572334ffdf7272f9`；日志
`ring_d8_r16_ep16_t*_formal.log`。

同配置第二节点单机 EP8 的正式 10/20/20 为
101.688/137.246/1292.810/3091.230 us，32 hashes 与 BD/V3 一致。该节点没有
其他 CPU/GPU 工作；本机同时编译下一候选、运行独立的 plan 检查。单机代码
不使用网络 Return Q，本轮只作回归，不将微小波动归因为网络配置收益。
日志 `ring_d8_r16_remote_ep8_*`。

### BI：40 个网络 owners（无收益，撤回）

Q16/L40/B512/D4，通信 context 增至 120，Pull/local Return/Reduce 配额相应
从 100/68/99 变为 92/52/91。EP16/8K 快筛 3591.02 us、16 hashes 一致，
未优于 BF/L32，registered pool 增至 88774912 bytes/rank。恢复 32-owner 上限，
未开展完整数值/sanitizer 验收。冻结 `build/static_v4_ring_lanes40/`，native
SHA256 `b700ee49addde06bc496deb39fa3ac40bee461e825c84c3ede60709872e31e3e`；
日志 `ring_lanes40_ep16_t8192_quick.log`。

### BJ：窗口内交错 local Pull 与 remote packet（无收益，撤回）

保持各自静态 residue，让每个 Pull owner 完成一个本地两行 batch 后消费一个
远端 packet，再进入下一轮；每次仍完成全部异步复制后归还原 credit，不跨轮
持有 RX。Q16/L32/B512/D4 的 EP16/8K 快筛 3754.37 us，16 hashes 一致，
比 BF 更慢。该候选增加了 remote helper 的调用频率，没有保留。未开展完整
数值/sanitizer 验收。归档 `build/static_v4_ring_pull_interleaved/change.patch`，
native SHA256 `8b093a92c90e3f13ddd0ebdc3fe19b47244b427120153bc89ff733a52ca0dcce`；
日志 `ring_pull_interleaved_ep16_t8192_quick.log`。

### BK：计算 owner 缓存整窗准入（无收益，撤回）

在 BH/D8/R16 上，math task selector 缓存成功 acquire 的 admitted_windows，
省去逐任务的 slot generation 查询；X full/FC1 done 的 acquire 保留。尚未完成
的 owned FC1/FC2 tile 持有该窗口 slot，缓存不跨 invocation。ComputePairState
由 48 增至 64 bytes；网络 prefill 仍 REG168/STACK440/LOCAL0。
EP16/8K/B512/D4 快筛 3418.96 us，与未缓存的 3411.25 us 相当，未保留额外
状态，也未开展完整数值/sanitizer 验收。归档
`build/static_v4_ring_math_admission/change.patch`，native SHA256
`07fe35ace906d1c89dd470cd0d6725282fa01fcdff59a85edcdb0970844d1d09`；
日志 `ring_math_admission_ep16_t8192_quick.log`。

### BL：计算 GPU 直接跨机 Return（无收益，已撤回）

候选协议见设计 6.8：Dispatch 不变；Return 按 source/compute-rank/lane 预规划，
compute GPU 从自己的 C 打包并直接发送给源 GPU，移除 payload 的节点内 rail
汇集。Return ready 缩为本地 C-slot generation，C 的 reader/retirement 全在本
GPU，因此统一 device scope。Put/ACK 使用互逆的跨 local-rank channel 映射；
RX scatter、逐 slot 累计 target 和远端 consumed AND 本地 completion 保留。
每 remote node 的 Dispatch/Return channels 分别 32 与 8×4，不乘出 8×32。

现有 host 检查 18/18（原 layout case 扩展非同 rail 的双向映射及不等 plane
大小），GPU plan 31+6（逐 compute rank 的 Return、含三节点）通过，没有新增
成批单测。D8/R16/L32/B512/D4 的 EP16/8K 首轮快筛 **4600.08 us**，16 hashes
与 BH 一致；比 BH 更慢，尚未完成 PyTorch/sanitizer 门禁，不合入该版本。
pool 933099456 bytes/rank，其中 registered 55232256；网络 prefill
REG168/STACK456/LOCAL0。冻结 `build/static_v4_ring_direct_return/` 与
`change.patch`，native SHA256
`576db8866da630ffaeeb19949acfbc4fa277c0e03d8b7a289b50811699a9ee27`，library
`c5fd3212b9a64256b4919337aa391fcbff155f5fdd6496c41649f5f9e1d63cbd`；
日志 `ring_direct_return_ep16_t8192_quick.log`。

代码审计发现这次拓扑调整使 TX 热循环的 source directory/RouteLocation 从
owner-local 变为 peer 读取。它是后续预绑定的假设，不把单次计时当成因果证明。

### BM：直接 Return 的本地 packet/逻辑行绑定（无收益，已撤回）

在 BL 的既有 destination PlanReady 后，固定 CTA/warp 复制实际归本 GPU 的
channel interval、packet descriptor 和逻辑 C 行号到普通 device pool；source
目录只在这个 pass 读取。现有节点内 barrier 已发布 RouteLocation，新增一个
grid join 让发送 owner 看见全部绑定，无第二次网络 metadata exchange。非法
route 的 collective error 不读取尚未建立的 locations，仍由既有入口一致失败。
TX 热循环只访问本地不可变绑定、block geometry、ready 和 C payload；没有新增
runtime claim 或 expected count。绑定属于 invocation plan，不是 transient
payload，内存随配置 route 上界增长，需单独计入资源成本。

首轮 D8/R16/L32/B512/D4 的 EP16/8K 快筛 3506.54 us，16 hashes 与 BH 一致；
比未绑定的 BL 低 23.8%，但仍比 BH 慢约 2.5%。pool 935726528 bytes/rank，
比 BL 增加 2627072；registered 仍 55232256。网络 prefill
REG168/STACK456/LOCAL0，kernel 参数区 3328 bytes。绑定支持“热循环 peer
metadata 显著拖慢 BL”的解释，但未使新拓扑胜出，尚不合入；继续单独消融
Return Q。冻结 `build/static_v4_ring_direct_return_bound/` 与 `change.patch`，
native SHA256 `91a7cbb5ff03fefc42739e6e3197f28bc85275e6eb816b8c927fa1c8781bb3a4`，
library `542019df39ce5eae64f6e82438b04c48b9e517c189d88587e34fb54f284a60b0`；
日志 `ring_direct_return_bound_ep16_t8192_quick.log`。

20K 快筛 8164.53 us（本机同时编译 Q32/Q64，非正式成绩），仍未好于 BH。
保持 BM 其余逻辑，仅将 Return Q 改为 32/64；两个节点无其他编译/GPU 工作
的 8K 快筛分别 **3650.88/3817.76 us**，32 hashes 与 BH 一致。registered
bytes/rank 分别 88737536/155821824；冻结 `static_v4_ring_direct_return_r32/`
及 `static_v4_ring_direct_return_r64/`，日志 `ring_direct_return_r*_ep16_t8192_quick.log`。
不为这些失败候选继续跑完整 PyTorch/sanitizer。BL/BM 及其拓扑/绑定/地址表
改动全部撤回，恢复 BH，避免保留比基线更慢的额外路径。

### BN：64 lanes 与 CTA-residue 控制阶段（无收益，已撤回）

独立回到 BH/D8/R16，不包含 BL/BM。B512 的均衡窗口有 64 个 Dispatch
packet/source 和 64 个 Return packet/source/remote-node，L40 切分不均匀；
L64 可以每 lane 一个。原 owner cap32 及“一 context 一 CTA”的控制阶段限制
了这项实验。候选把 owner cap 提到 64，metadata Put 与 GIN fence 采用
`context=blockIdx; context<contexts; context+=gridDim` 的固定 residue，仍逐
context fence，不省略活跃或空 context。每个 CTA 的 context 顺序递增，
不存在跨 CTA 的动态领用。

132 CTA 下 W64 分配为 Pull68/local Return4/Reduce67，另有独立的 Dispatch
TX64、Return TX64、Return scatter64、NIC progress64 与 admission1。
192 contexts 不要求 192 个 CTA 同时执行控制阶段；host 仍要求至少 `2W+1`
CTA，保留所有数据消费者的常驻进度。

L64/D8/R16 的既有 GPU plan 31+6 cases 通过。EP16/8K/B512/D4 快筛
**4011.28 us**，16 hashes 与 BH 一致，比 BH 更慢。pool 983659200
bytes/rank、registered 105763072，几乎加倍的网络 ring 没有带来收益；网络
prefill 仍 REG168/STACK440/LOCAL0。没有继续完整 PyTorch/sanitizer 门禁，
已撤回 owner cap、host 条件和控制阶段 residue 三处改动，恢复 BH。
归档 `build/static_v4_ring_lanes64/change.patch`，native SHA256
`066504fbe55b61b39c220fb1ed524c46cc406a2d1ccf98d0ea3caa672ac63240`，library
`20541fdf9f57c90040b00a88c2811b2f78641e40c40a15fa4ed160a5894dcea6`；日志
`ring_lanes64_ep16_t8192_quick.log`。

### BO：按节点划分 source group（首轮配置无效，修复后重测）

独立回到 BH/D8/R16/L32。将 G 从 `min(world,16)` 改为
`min(local_world,16)`；8-GPU 节点不再把本地和远端 source 混进同一 Mblock。
先比较 G8/B1024 与 G16/B512：均衡时每目的窗口仍 16 Mblocks，最坏 X/S/C
窗口容量相同，source contribution 的 B 增大则另计内存。也会测 G8/B512，
不把 B 的影响混归因给 G。窗口编号、source 序号、packet 目录和 target 仍由
现有 WindowShape 推导，没有增加通信 barrier 或动态任务分配。实验未完成。

现有 GPU plan 调整 source-window 检查为 G8；destination 检查在两节点用 G8、
三节点保留 partial G16，31+6 cases 通过，未增加用例数。但随后发现 launch
仍将 `p.shape.sources` 覆盖为 `min(world,16)`，host 分配与运行时几何不一致。
因此首轮 **不能作为 G8 的性能、正确性或内存收益证据**。首轮 EP16/8K：
B1024 为 3548.18 us，pool 966713536、registered 55201536 bytes/rank；
B512 为 3435.17 us，pool 631103424、registered 55226624。20K/B512
为 7949.49 us，pool 976616640、registered 62306048。虽然这三轮共 48
hashes 与 BH 一致，但不能排除跨窗口或其它拓扑的越界；没有继续完整数值门禁。
这些值仅归档失效实验，不计入优化收益。修复 launch 继承 layout 的 sources
上界后，以独立 build 和日志重新测量。
冻结 `build/static_v4_ring_node_groups/` 与 `change.patch`，native SHA256
`7bff3edf78129bbc8d13038358b2421beae5c676d386b94699c668bd885ec2ea`，library
`c67f1bef65d3a62225b80ea8c455e214215ad7b90acb9a0baae4da46e3459c71`；日志
`ring_node_groups_b*_ep16_t*_quick.log`。

修复后的 native 单独冻结到 `build/static_v4_ring_node_groups_fixed/`，
SHA256 `2557ec4da14e8158b3718451d1b31f966340e82aafce92c1469cab5abe3e9722`。
首次实际 G8/B512 的 EP16/8K 快筛为 **3357.44 us**，16 hashes 与 BH
一致；pool 631103424、registered 55226624 bytes/rank。该轮同时编译 BP，
且仅两次 measured replay，因此仅作快筛，不作为正式收益结论。

### BP：节点内窗口的八行 Pull batch（修复后无独立收益，已撤回）

BO 正确配置时每个窗口仅包含一个节点的 source，因此 Pull 可按整窗选择 local 或 remote，
不再扫描全窗后逐行跳过另一类身份；本地 batch 从 2 恢复为 8，remote packet
helper 不变。G 进一步定义为 `local_world` 的不超过 16 的最大因子，使所有
支持的拓扑都保持 group 不跨节点，而不是依赖测试机器恰好有 8 GPUs。
单机原本就是八行 batch；所有复制仍在完整 global completion 后才发布聚合
X credit，source reservation、whole-window admission 及逐 slot 生命周期不变。
`build/static_v4_ring_node_pull_batch/` 已编译，但含 BO 的 launch 几何错误，
**从未运行**，不作为实验结果。修复后仍须兼容大于实际 world 的 pool 容量：
若 group 跨越实际节点边界，则保留逐行过滤并执行 local/remote 两类 Pull。
实验尚未完成，不将 BO 的容量变化与本项潜在流水收益混为一项。

修复后的 BO 快筛：B512 的 8K/20K 为 3357.44/7865.26 us；B1024 为
3279.57/7635.92 us。BP 在相同几何下为 B512/8K 3360.43 us，B1024
8K/20K 3300.90/7541.58 us，尚不能证明八行 batch 的稳定独立收益。
继续扩大 BP 至 B2048 反而退化至 3694.85/8047.39 us，8K/20K pool
1638179520/1983324352 bytes/rank，不继续扩大。均为两次样本快筛，部分
轮次并发 CPU 编译；正式计时与完整数值验收待完成。

BP 的 B1024/EP16/8K eager timeline，rank0 有 55205 events、overflow=0：
Dispatch pack、Return pack、RX scatter 单包 p50 分别 7.648/26.176/25.792 us；
FC1 math/epilogue 16.384/6.368 us，FC2 math/epilogue 6.880/6.528 us。
这些并行跨度不可相加作为 kernel 时间。归档 `ring_node_pull_batch_fixed_*`；
修复后的 native SHA256 `a77b7117fc881a0675378d8479a41eb6619fc6f31b503fbc73f506c59b8014d2`，
library `a2e8f057ef5e5ec03278546374cfd4de685f1acb3e0e4041f91319fcfb980360`。

### BQ：两窗内本节点优先的静态顺序（无收益，已撤回）

在 BP 上，只重排 Pull 与 math 的窗口访问顺序，不改变任何任务的 owner residue、
逻辑行号或 packet 顺序。将同一 source-token window 内相邻两个 source groups
配为一对；若第二个 group 完全属于本节点，则本节点先计算它。其它 group
顺序保持原样。这样可先消费本地已就绪 X，而不是先等待另一节点的 Dispatch。

重排跨度限于相邻两窗，compute ring 仍能容纳这两窗的最坏容量；不得跨
source-token window 边界，否则 contribution depth=1 时可能提前等待下一代，
而上一代尚需当前未处理 group 的结果。节点数为奇数时，每轮末尾 group 不交换。
admission 仍按原逻辑顺序发布，逐 slot 的 ready/FC1/FC2/returned 不变。
GPU plan 31+6 cases 通过，EP16/B1024/8K、20K 快筛为 3292.14/7565.95 us，
32 hashes 与 BP 一致，未改善 BP 的 3300.90/7541.58 us。没有继续完整数值与
sanitizer 门禁，已撤回窗口重排及仅服务它的 plan 断言。归档
`build/static_v4_ring_local_first/change.patch`，native SHA256
`839805529d486af8c1fd9de874d8be50de14c2a0bd7e98e06e43a3c10f7680de`。

### BR：节点对齐分组验收与八行 batch 取舍（已验证，保留 BO）

BO 的两行网络 Pull 与 BP 的八行/整窗过滤正式对照，均 EP16/G8/B1024/D4、
1620 MHz、10 warmup/20 samples/batch20，无并发 CPU 编译或其它 GPU 工作：

| tokens/rank | BO/us | BP/us |
|---:|---:|---:|
| 8192 | 3282.20 | 3285.27 |
| 20480 | 7630.13 | 7613.45 |

差异不超过约 0.22%，不认定八行 batch 有独立收益。撤回 BP 的 batch 和
整窗分类逻辑，只保留 BO 的 host/launch 分组一致性。最终清理构建还增加
pool-query 的 `max_world >= max_nodes` 必要检查，避免无效容量让分组除零。

BP 清理构建 `static_v4_ring_node_aligned/` 的正式四规模为
291.635/400.416/3285.270/7613.450 us；native SHA256
`599df7635cdf96578f52076a435196c52beefc1f58e6d58d43b332e69e1e3a23`，library
`04ef18ffe3965b0ae5f9eae0663e9e78f8cbe56d5880d2dce47fa9bedac351f1`。
此构建的 EP16 完整 416、小深度 B16/D1/K8 的 272 records 均通过；rank1/9
eager 11 rounds 的 memcheck/racecheck 均 launcher exit0、0 errors/0 hazards。
此前同 device 逻辑的 BP 构建另通过 EP8 的 208 records。host 18/18、GPU plan
31+6 cases 通过，网络 prefill REG168/STACK440/LOCAL0。上述 BP 门禁作为已测
但撤回候选归档，保留 BO 的最终构建及复测单独记录，不混用二者结果。

最终冻结 `build/static_v4_ring_node_groups_final/`：native SHA256
`0cd7ff8538d236c1a80d063760b7fca0958a91ab66e0ccc1eebc4a011016b6c2`，library
`6a053869674fb390c8b892c6eda475f89e5e991e31e63b71437242a23468bfec`。此最终
构建重新通过 EP16 四规模+edges 416、B16/D1/K8 的 272、单机 EP8 四规模
208 records；rank1/9 的 B16/D1/T64/K8 eager 11 rounds memcheck/racecheck
均 launcher exit0、0 errors/0 hazards。不声称 sanitizer 覆盖全部 ranks。
GPU plan 仍为既有 31+6 cases，host 18/18；网络 prefill REG168/STACK440/LOCAL0。

正式计时均核心 1620 MHz、10 warmup/20 samples/batch20、E64/K4/H2048/I512。
单机 EP8 在第二节点，B512/D4；EP16 双机各八卡，B1024/D4、D8/R16/L32。
计时节点无 CPU 编译或其它 GPU 工作；EP8 计时时第一节点独立做数值回归。

| tokens/rank | EP8 V3/us（本轮） | EP8 G8/us | EP16 V3/us | EP16 G8/us |
|---:|---:|---:|---:|---:|
| 1 | 300.557 | 101.878 | 682.170（历史） | 291.851 |
| 64 | 370.549 | 137.499 | 824.659（历史） | 401.750 |
| 8192 | 1486.870 | 1287.060 | 3359.260（本轮 B1） | 3280.020 |
| 20480 | 3299.890 | 3102.440 | 7099.960（本轮 B1） | 7627.350 |

EP8 的 20K 另测 B1024/D4 为 **3007.09 us**，相对本轮 V3 降低约 8.9%；
B512 的 8K/20K 分别降低约 13.4%/6.0%。B1024 增加 ring 容量，不把额外容量
归因为无成本代码收益。EP16 两个长规模相对 BH 的历史 3420.79/7937.91 us
降低约 4.1%/3.9%；仍未达到两项都显著优于 V3 的目标。

EP16 的长序列逐规模抽取交错顺序 A1/B1/B2/A2（A=本版，B=冻结 V3；两规模
交叉运行，每轮独立 warmup）如下，不使用历史 V3 推断这项收益：

| tokens/rank | A1/us | B1/us | B2/us | A2/us |
|---:|---:|---:|---:|---:|
| 8192 | 3280.02 | 3359.26 | 3358.35 | 3278.73 |
| 20480 | 7627.35 | 7099.96 | 7098.86 | 7625.32 |

两次 p50 的均值相比，8K 快约 **2.4%**，20K 仍慢约 **7.4%**。EP8 的 32、
EP16 首轮的 64 rank hashes 与各自 V3/BH 一致，长序列交错轮次的 hashes
也全部一致。EP16/B1024 的 8K/20K pool 为 966713536/1311981248 bytes/rank，
registered 为 55201536/62243072。B512 的已修复 BO 8K pool 为 631103424，
比 BH/B512 的 933122752 少约 32.4%，但与 B1024 的性能/容量权衡须分开看。
日志 `ring_node_groups_final_*`、`ring_node_groups_v3_*`；原 Q、GEMM profile、
double-credit 与逐 slot 生命周期不变。最终提交不包含 BP/BQ 的数据面改动。

### BS：规则跨行 Return TMA gather（无端到端收益，已撤回）

在 BR 上保持 packet、slot credit、数值和求和顺序不变，仅替换符合规则步长
的 C→TX 复制：同一 destination、完整 16 行片段、每行 physical C 地址增量
恰为 G。其它包仍用旧的逐行复制；不读取未参与当前包的 C 行。资格判断暂时
合并在已有的 C-ready 扫描中，未来是否预计算由测量决定。

每个 pool slot 在 ordinary device pool 保存 local_world 个 C TensorMap 和
一个 TX TensorMap；host 副本由 slot 持有，异步上传生命周期与 pool 一致。
C 是 `[H*G, ceil(C_rows/G)]` 的 2D 视图，TX 是带 packet stride 的 3D 视图。
无 swizzle，box 为 256×16 BF16（H 不整除 256 时使用 128 列）。返回 owner
每 CTA 一次 system-scope tensor-map proxy acquire，不能以 stream ordering
代替；参考 [CUDA 12.9.1 的 global TensorMap 用法](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html)。

网络 Return 私有 SMEM 从两个 4-KiB row buffers 增至两个 8-KiB tiles，
Dispatch staging 和单机路径容量不变。两个 TMA load 先发出，再等待/存储；
仍完整等待 global store 后才发 C reader credit 和 GIN Put，不使用 GEMM 的
multicast A。EP16 pool 增加 1280 bytes/rank，registered bytes 不变。

首轮 EP16/B1024/8K 快筛 **3333.82 us**，16 hashes 与 BR 一致，暂未好于
BR 的 3280.02 us。此轮同时编译 Python extension，后续完整数值/sanitizer
尚未开始；不能视为已获益优化。native SHA256
`c7b7628d66fbad7eed1b35fb8f26f06bf6b70761b50a444e03ee858cc8ca2ead`，library
`b0c05d1fc85ba737c3aa9c224c25817994314f0060ff2ad6edeeea8d25163677`，归档
`build/static_v4_ring_return_gather/change.patch`（含新 helper）。网络 prefill
REG168/STACK456/LOCAL0，较 BR 多 16 bytes stack。若保留，sanitizer 除 B16
fallback 外还必须覆盖 B256/D1、足以形成完整 affine packet 的 workload。

BS 的 20K 快筛为 **7668.61 us**，16 hashes 与 BR 一致；此轮并行编译
timeline 版本，仍不是正式性能验收。8K eager timeline 的 rank0 有 69541
events、overflow0，1024 个 Return 包全部进入 gather：Return pack p50
13.888 us，其中 gather 12.160 us；Dispatch pack 7.776 us，RX scatter
26.656 us。此前 BP 同配置 Return pack/RX scatter 为 26.176/25.792 us。
这说明局部 pack span 确实缩短约一半，但不能把并发 span 相加，也不能据此
认定端到端已获益。日志 `ring_return_gather_b1024_ep16_t*_quick.log` 与
`ring_return_gather_b1024_ep16_t8192_timeline*`。

### BT：Return 首末位置推导与逐 slot 检查（无明确收益，已撤回）

Return 包在 source expert order 内连续。同一专家的相邻 source ordinal
在目的布局中最多前进 G 行；首末 generation 相同排除 physical ring wrap，
若首末距离恰为 `(rows-1)*G`，即可证明中间每一步都是 G，无需逐行重复加载
expert/location 元数据。不同专家或非规则片段仍走旧扫描。

规则 span 仍 acquire 每个实际涉及的 C slot 的 exact generation；复制全部
完成后，按该 span 在每个 slot 的实际行数归还 reader credit。G 不大于16、
M128，故首末之间不存在完全未参与包的 slot。它独立于 TMA 的16行整除要求，
因此单独消融“span+TMA”和“span+旧逐行复制”，不把收益混在一起。

span+TMA 的 EP16/B1024 8K/20K 快筛为 **3274.51/7615.58 us**，全部32
hashes 与 BR 一致；相对 BR 仅约0.2%，不足以支持保留额外的描述符、SMEM
和分支。“span+旧逐行复制”的8K/20K为 **3301.44/7705.42 us**，32 hashes
一致，亦未改善。上述快筛
均存在并发 CPU 编译，不替代正式验收；未继续 PyTorch/sanitizer 门禁。
已撤回 BS/BT 全部源码，归档保留两种候选，不给生产路径增加调参开关。

`static_v4_ring_return_span/` native SHA256
`9883ae35acfe83a3cf31b48cccf64de2ef320319b790517dc7fe98c70368522e`，library
`43ccd348fda90c14b9d262402aaecedd93ce7134a93dc383a83b4fd888a0589a`；资源
REG168/STACK456/LOCAL0。“span+旧复制”归档 `static_v4_ring_return_span_rows/`，
native `47aa31a8f6491e570ee461064ddadfbf78ebd85f40d8891eb2892fad7b9ee9d9`。
两个 change.patch 都包含当时的新增 helper，不依赖已撤回的工作区文件。

### BU：Return RX warp 协作 scatter（退化，已撤回）

从 BR 单独替换 RX→contribution 的本地复制，32 lanes 按 H 的连续16B向量
协作。lane0 acquire RX generation 后经 warp memory barrier 传递可见性；
所有向量写完，经 warp join 再发布逐行 contribution-ready；第二次 join 后
lane0 才归还 RX consumed credit。网络 import/ACK 仍由原固定 owner 独占。
没有更改 Return pack、packet 目录、GEMM、求和顺序、ring 大小或双重 TX credit。

最初基于 BS/BT 的 CPU 编译中途取消，未执行 GPU；检查发现其还缺少 acquire
之后的 warp memory barrier。该目录 `static_v4_ring_return_warp_scatter/`
已标 INVALID；独立修正版本为 `static_v4_ring_return_warp_scatter_base/`，
不混用二者测试结果。

独立修正版本 EP16/B1024 的8K/20K快筛为 **3530.72/7953.65 us**，32 hashes
与 BR 一致；8K同时编译 extension，20K同时准备下一候选的 CPU 编译。
没有继续完整数值/sanitizer，全部数据面改动已撤回。native SHA256
`e657e2b135ddd580e153303f2e56ec3ff20d1fda75b8ea25caf18289d268ffa7`，library
`e8ba44723a87a223b523fd712f4134dccbada2498fb082d30f4d6aeea6bce80f`；资源
REG168/STACK440/LOCAL0。`static_v4_ring_return_warp_scatter_base/change.patch`
保留独立候选。普通 warp vector copy 不适合据此替代这里的 bulk async copy。

### BV：Return RX 连续多行加载、分行 scatter（无明确收益，已撤回）

仍从 BR 起步。RX payload 本身连续，不依赖 route_order 的 token 连续性。
最初两个16-KiB staging 的4行方案超过当前 GEMM profile 的 SMEM预算，被
编译期227-KiB检查拒绝，未运行GPU，保留 `build_static_v4_ring_return_batch_scatter*`
编译日志。改为两个8-KiB私有 staging，主测 H2048 下一次加载2行；发出两批 G2S 后分别
等待，将每批的行写到原 contribution 目的地并作为一个 store group commit。
覆盖 staging 前等该 slot 的 shared-read completion；发布 ready 和 RX credit
之前仍完整等待 global stores。最后不完整批按真实字节数处理；单行比 staging
大的其它 H 走原分段 copier，不越界也不静默截断。

只增网络通信 SMEM，Dispatch仍8 KiB、Return由8增至16 KiB，总计24 KiB；
单机与非 ring 的分配不变。没有新增 TensorMap、GM pool、registered bytes
或网络 metadata。独立 helper 为 `return_scatter.cuh`，不修改通用/V3 copier。

2行版8K快筛 **3295.65 us**，16 hashes 与 BR 一致，暂未改善；同轮编译
extension，后段另启动下一候选编译。native SHA256
`e4d92d5143625dd6b3a4a8017c772222d2db8086caf6bf5bf55e161cd61857e2`，library
`94d5eed8e2675db391a4f3ececeb408dc44523fc3895a6286068b2055c3167e6`；资源
REG168/STACK456/LOCAL0。`static_v4_ring_return_batch_scatter2/change.patch`
包含新 helper；完整数值/sanitizer 未开始。

20K快筛为 **7605.26 us**，16 hashes 与 BR 一致，同时进行下一候选 CPU
编译。不到0.3%的长规模差异、8K略退化，不足以保留新增 helper/分支及SMEM；
已撤回全部数据面改动。未声称 sanitizer 或非主测 H 的 fallback 已验证。

### BW：移除无读者的 route 反向目录（已验收，保留）

代码检索确认现有 ring 数据面只从 route_order 顺向枚举，从 immutable
PullPacket 定位 RX；不读取 `source_slot[route]` 与 `route_packets[route]`。
删除这两份早期方案的反向索引、生成时的全局写入、pool 与 GIN replica 分配。
不改变 source ordinal 的定义/枚举，只不再冗余存储其逆映射；将来 StableTape
按其真实需要设计保存映射，不让 inference 提前支付未使用目录的开销。

每个 source metadata 减少约 `4*window_capacity + 8*max_routes` bytes；
单机减少前一项，多机也减少每份副本及跨机发布量。既有 GPU plan cases 保留
route 覆盖/唯一性、mask/非法 expert、节点去重、consumer rows 和退休计数，
仅删除针对已移除数组的断言与分配。数据面、SMEM、GEMM 与 packet 顺序不改。

host 18/18、GPU plan 原31+6 cases 通过。EP16/B1024/8K 初筛 **3267.38 us**，
16 hashes 与 BR 一致，pool/registered 分别965927104/54415104 bytes/rank，
两者均比 BR 少786432 bytes。此轮同时编译 extension，尚非正式验收。
native SHA256 `5e9b839861f13a71232a01b18c9b2676daba18d7b0f2310c4a1146775ba8933d`，
library `9117db0d7e95c5caefbc1c9599203010f685a3bdb923726b1bc5340053c93075`；
REG168/STACK440/LOCAL0，归档 `static_v4_ring_compact_plan/change.patch`。

20K初筛7627.10 us，与BR持平，16 hashes一致；pool/registered降为
1310015168/60276992 bytes/rank，减少1966080 bytes。EP16完整四规模+edges
416、小深度B16/D1/K8的272条PyTorch检查均通过；rank1/9 eager 11 rounds
memcheck launcher exit0、0 errors，日志 `ring_compact_plan_*`。

rank1/9 的同配置 racecheck 也 launcher exit0、0 hazards；不声称覆盖所有 ranks。
memcheck进程local380594/remote123609，racecheck local381660/remote123885。
单机EP8完整四规模+edges的208条PyTorch检查通过。第二节点的单机正式计时
（1620MHz、10 warmup/20 samples/batch20，无该节点CPU编译或其它GPU任务；
第一节点独立数值检查）B512的四规模为101.198/137.301/1274.200/3080.790 us，
20K另测B1024为2974.260 us。不能将不同B的容量取舍隐藏在单一性能数字中。

双机EP16正式计时为B1024/D4、D8/R16/L32，核心1620MHz、10 warmup/
20 samples/batch20，两个节点无并发CPU编译或其它GPU任务：

| tokens/rank | BW/us | ordinary+registered bytes/rank | registered bytes/rank |
|---:|---:|---:|---:|
|1|292.093|67473856|50514688|
|64|399.363|108771776|50541312|
|8192|3255.970|965927104|54415104|
|20480|7605.830|1310015168|60276992|

EP16的64、EP8/B512的32个rank hashes与BR一致。相比BR历史同配置，长规模
降低约0.73%/0.28%，不声称是显著突破；本次收益确定的部分是移除无读者数组、
写入和传输，以及上述容量下降。对照BR轮次冻结V3的3359.26/7099.96 us，
BW的8K快约3.1%，20K仍慢约7.1%，端到端目标尚未实现。完整日志
`ring_compact_plan_*`，不混用后续BX/BY/BZ候选。

### BX：按 Mblock 合并两份静态任务序列（无明确收益，未保留）

在 BW 上，保留 FC1/FC2 的全局 residue 与 owner，只调整同一 owner 内的次序：
取两份静态序列较小的 Mblock，同 Mblock 先 FC1 后 FC2。不是“谁 ready 先做谁”；
cluster 内两个 CTA 的选择始终一致，ready 只控制等待，否则独立观察可能选择
不同算子、破坏 multicast 配对。依赖按 Mblock 单调推进，任何等待 FC2(b) 的
owner 都先完成自己的 FC1(<=b)，不会挡住更早的 FC1 依赖。

主测 B1024 均衡窗口中多数 owner 只有一个 FC1/FC2 pair，预期差别不大；主要
验证较大窗口是否能提前产生 Return，避免先耗尽整窗 FC1 才做 FC2。独立 BR
B1536/D4/20K 的初筛7747.70 us、pool1649117376 bytes/rank（CPU编译并行），
本身不如 B1024。不把较大容量的潜在收益误标为调度优化。

冻结 `static_v4_ring_mblock_merge/`，native SHA256
`e0f36b77ec7e1423c573247df57e8d5c5b7f5a7ac86fdce77b579fef10814f69`，library
`ce54a1f606ee083289ea8e211075c6882db28213a76362960e178d44b6dcac40`。仅编译
完成，尚未GPU验收；当前源码先恢复BW，以独立测试另一通信组合。

随后EP16/B2048/D4/20K快筛为8115.50 us，同容量BW对照8129.89 us，差异
不足0.2%；二者均明显不如BW/B1024的7605.83 us。没有继续完整数值或sanitizer，
工作区已撤回，不把 larger B 的容量与调度本身混为一种收益。
两份构建的16 hashes均与BW/B1024一致；pool均1981358272 bytes/rank。

### BY：大 Return packet + 规则 TMA gather（未获益，未保留）

在BW上恢复BS/BT的组合，仅编译期Return Q改为32/64，Dispatch Q8、L32不变。
此前Q16时局部pack缩短不转化为端到端收益；本项检验更快的pack是否使更大
packet及更少GIN提交获益。新增 maps、SMEM、endpoint span 与旧扫描fallback
均按BS/BT合同，不混入BV scatter或BX调度。仍只是组合消融，不默认合入。

R32 native SHA256 `cf56d1d49df6c938f2b25254c6b010852dcf210049c4c02122712e2fd4cebc84`，
R64 `2b9182d2f99e7b4898d92c099267cd61af630822c8ef5dabe7b6ee4b96aa8129`。
归档 `static_v4_ring_return_tma_r{32,64}/change.patch` 是相对BW的增量，
包含两个helper；本轮尚未GPU验收，也未构建library。工作区已恢复BW再开展独立候选。

随后EP16/B1024/8K快筛R32/R64分别3342.13/3332.82 us，均未改善BW的
3255.97 us；本轮有并发CPU编译。不继续完整数值/sanitizer，也不保留描述符、
SMEM或endpoint span分支。Q与copy方式共同消融仍未提供端到端正向证据。
R32/R64的32 hashes与BW一致，pool分别999433664/1066517952 bytes/rank。

### BZ：多机 prefill 通信专用 CTA（无明确收益，已撤回）

在BW上，不混入BX/BY。多机prefill可以预留前32/64个CTA，只运行原静态通信
角色；这些CTA的math/producer跳过主计算，但仍参加初始化、route plan和终端
collective。其余完整双CTA cluster承担全部FC1/FC2，owner residue按剩余
cluster数计算。两CTA同进退，multicast不会跨到无计算的CTA；F1/F2和C读者
计数不变。Host既有task-count容量检查使用剩余cluster数，并拒绝没有计算
cluster的配置。不是另起并发kernel，不引入“通信kernel必须先调度”的驻留风险。

本项用于检验共享SM上的计算/通信竞争，代价是更少math CTA；可能吞吐反而
降低。只在网络prefill启用，短序列与单机begin=0。编译开关暂默认0，实验
若不获益则移除，不为每个失败候选长期保留分支。

已冻结 `static_v4_ring_comm_only{32,64}/` native，32版SHA256
`4849fc5965852111e964033353048781ea278d4d4535451bb221cb5138150308`，64版
`151412f9a7c5b1613acf2d4b1200291eb43ab7d1e3f5b56222bba5de3b6d1347`；64版
REG168/STACK440/LOCAL0。当前工作区恢复BW后继续独立的索引候选。

EP16/B1024/D4的8K快筛，预留32/64 CTA分别3244.02/3319.31 us；32 CTA
的20K为7659.84 us，另测B768为7674.58 us（pool1142634432 bytes/rank）。
32 CTA的8K小幅改善没有延续至20K，64 CTA计算吞吐损失更大；未继续完整
数值/sanitizer，已撤回宏、host参数、role分支与residue变化。32/64初筛hash
正确，但这不替代完整门禁。日志 `ring_comm_only*`；部分轮次并行CPU编译。

### CA：任务索引保留编译期 GEMM 常量（候选测试中）

`select_static_owned_task` 原先通过noinline参数接收FC1/FC2 N tile数，丢失了
主形状H/I与profile tile宽度的常量信息。将已有的Fc1TileN/Fc2TileN作为模板
实参传递，在函数内据编译形状计算ntiles，消除相关运行时整数除法；删除未使用
的fc1_tasks参数。保持noinline边界、静态residue、FC1/FC2次序和全部ready检查。
只有已建立对应主形状TMA maps的专用入口调用此函数，不把任意运行时H/I偷偷
替换成固定数。此候选不含BX/BY/BZ，待单机与双机验证。

冻结 `static_v4_ring_const_task/`，native SHA256
`30706e0bc7687cee06c015b4bee96fd96a2783b5753aac208cbcad6e3dc15faa`，library
`d7a91243ac5da58635cfdf1c0b511bdf5aa5800a7af8acefd2292c455a04ff40`；
REG168/STACK440/LOCAL0，与BW相同。

CA 的 EP16/B1024/D4 快筛8K/20K为3259.74/7590.24 us，32 hashes与BW一致。
端到端差异在噪声范围；尚无完整数值/sanitizer，不将常量传播称为显著收益。

### CB：Return 顺序 ACK head（无明确收益，已撤回）

在CA上为Return的每个channel增加credit_cursor，按严格RX消费序列检查head，
避免反复扫8个slot；Dispatch仍允许乱序退休并保留逐slot检查。每个ACK仍核对
当前generation及收到的retire_target；TX的双credit不变。cursor复用原64B
channel padding，未增加pool。REG168/STACK424/LOCAL0。

冻结 `static_v4_ring_return_ack_head/`，native SHA256
`ed57165b57d9f2a06f05b31a2224d403396632cf0f641ae0efccd3921a0399da`，library
`fc1477d5c554d08ee42b34bbe1aa4dc18b0a7d227b96e06144cee65791c46781`。
B512/D8/20K快筛7286.46 us，和BW同配置7290.27 us持平；B1024/D4的
8K/20K为3244.66/7644.94 us，20K反而退化。三轮48 hashes均与BW一致，
未继续完整数值/sanitizer，已撤回，不保留额外cursor及方向分支。

### CC：窗口与 contribution 深度联合消融（参数候选）

只使用已验收BW二进制，网络Dispatch Q8/Return Q16/L32不变。下面D指source
contribution窗口深度，不是网络packet ring深度；所有初筛hash均与BW一致。

| B | contribution D | 8K quick/us | 20K quick/us | 20K pool bytes/rank |
|---:|---:|---:|---:|---:|
|1024|8|3280.05|7338.58|1377255360|
|1024|16|未测|7317.36|1511735744|
|512|8|3152.21|7290.27|1008270784|
|512|16|未测|7302.78|1075511232|
|256|8|未测|7696.34|824391872|
|256|16|未测|7714.70|858012352|

选择B512/D8继续验收：D16没有足够收益抵消额外容量，B256则窗口开销过大。
1620MHz、10 warmup/20 samples/batch20、无并发CPU编译或GPU任务的正式结果
为8K **3146.080 us**、20K **7312.640 us**。相对BW/B1024/D4降低3.38%/3.85%，
pool由965927104/1310015168降为663937216/1008270784 bytes/rank。
对照BR轮次冻结V3，8K快6.35%，20K仍慢3.00%，不宣称达到最终性能目标。
日志 `ring_compact_plan_b*_d*_ep16_*`；B512/D8完整PyTorch门禁正在进行。

随后B512/D8双机完整四规模+edges共416条PyTorch检查通过（208/node，两个
launcher均exit0）。T1/64的同20样本计时为291.490/401.341 us，但本节点同时
编译CE，不能标为无干扰正式复测；长规模上述正式两轮没有这个并发。

### CD：预绑定不可变 channel plan（无明确收益，已撤回）

在CA上把TX/RX source/begin/count预计算至每个NetworkChannel的独立只读
cacheline，规划完成后发布；不改任何payload或credit。channel由64B变128B，
EP16增加4096 bytes/rank。原运行时地址算法移至ring_access复用。

冻结 `static_v4_ring_channel_bind/`，native SHA256
`6335e8f483bd2583731c058f47df9a6d4511aa32691abafcf7c6dace85dcc7a8`，library
`1ac1d8b644f733c7e37f0952576765375faab2236625a6be0d92fb868d472fde`；
REG168/STACK456/LOCAL0。B512/D8快筛8K/20K为3156.62/7288.24 us，
32 hashes与BW一致，与未预绑定版本无明确差异。没有继续完整数值/sanitizer，
已撤回四处源文件改动；不因局部省了寻址就保留额外布局和初始化。

### CE：已 drain 的 ring 使用单 context 终端会合（候选）

在CA上仅改变ring的结束会合。所有TX owner退出前已核对每个slot的远端
consumed credit及本地NIC completion；RX owner已消费并发出ACK，所有本地
消费者和计算角色已join。此时单context rail barrier + 原node-local barrier
足以确保所有节点都停止使用本次metadata，随后才能开始下一调用。
没有用context0去替代仍有未完成Put的其它context fence：规划阶段仍原样
逐context fence，数据面双credit及terminal检查也完全保留。

ACK的发送者可能先进入会合，但其接收者只有观察到ACK才能完成自己的TX drain
并进入会合，因此会合完成时全部ACK也已被观察。任意有效节点数仍采用原rail
team与node-local同步，未引入双机专用路径。冻结构建与性能门禁进行中。

冻结 `static_v4_ring_drained_barrier/`，native SHA256
`28e2832122c3954dfb915e72a96ff829bb76b038e056063a958ccf7072f2e84e`，library
`08b09f6d94cccdbca7e9866476ef41250fef0c237e9994879952433c06edc529`；
REG168/STACK440/LOCAL0。B512/D8四规模快筛288.864/395.056/3158.640/
7334.590 us，长序列无明确改善。结束阶段status包含等待其它rank进入会合的
偏斜，不能把整个0.3 ms都当作GIN barrier自身开销。完整数值门禁进行中。

### CF：规划阶段仅 fence 实际 Put 的 context（候选）

在CE上，network_put的分片及其owner不变；有效owner数为
`min(contexts, ceil(source_bytes/256KiB))`。其它context在本阶段没有Put，
不必参与GIN fence；保留原grid/node-local同步，所有实际Put仍由自己的QP
signal导入可见性，不用单QP signal替代其它QP的未完成写入。非ring旧入口
仍显式使用全context，ring结束继续用CE的单context会合。没有改payload
阶段的32个Dispatch/32个Return/32个RX owner，也没有减少双credit。
本候选编译及快筛中；只有端到端受益且门禁通过才保留接口变化。

CF native SHA256 `a1e31d56913818c2e7dfa676b5112e2ab4cfff98ed040f8af958e14ad3258c00`，
library `bead827f71f1341f32ba8fd6568d6acf828c7228a83ccd1d95cb81efe7ccd9cb`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_control_contexts/`。B512/D8四规模
快筛281.936/391.840/3148.820/7304.820 us；64 hashes与BW一致，双机完整
四规模+edges的416条PyTorch检查通过。CE的同套416条检查也通过。
短规模初筛改善约3%，长规模持平；sanitizer和无并发正式复测尚未完成。

CF上增加仅profiling构建启用的六个规划时间点（schema11）：source、packet
目录、metadata交换、窗口几何、地址绑定，各阶段只在rank本地CTA0取时间差。
`static_v4_ring_plan_timeline/test_cuda_notify` SHA256
`675118760f66a495f294e65f69b9885205decea8bbd972651bba80fe0e07df9c`。
20K/B512/D8 trace每rank166955 events、全部0 overflow；各rank阶段耗时的
中位数111.216/134.736/91.376/113.568/80.832 us。不是只有跨机交换昂贵，
也不能把这五段解读成互不重叠的纯算术/纯网络时间：段内包括相关grid等待。
日志 `ring_plan_b512_d8_ep16_t20480_timeline*`，profiler时延不替代正式性能。

### CG：完整远端 FC2 tile 写 C 复用已有 TMA map（候选）

在CF上，针对整个source group都位于当前节点之外的窗口，所有FC2行均写本地
连续C。当M/N完整时，使用既有C output_store_map和已退休的CTA私有B arena
进行TMA store；不新增map、SMEM或buffer。本地分散contribution、混合节点
窗口以及tail仍走原向量写回。判定检查source区间与本node区间不相交，不假设
任意运行时拓扑必定是双机或G8。两math WG stage后proxy fence和join，完整
`cp.async.bulk.wait_group 0`及join之后才允许既有FC2完成链发布，未提前释放C。
候选编译中，不把使用TMA本身视为性能收益。

CG冻结 `static_v4_ring_remote_fc2_tma/`，native SHA256
`9f24de8e77fea7ae71285339eacd4b1df03158d2b3d6ef16746b87221f1786a9`，library
`c8c14883b71f8d5e99d230c74877f21d304355bdb79702fd3e580db0ad571154`；
REG168/STACK440/LOCAL0。B512/D8快筛8K/20K为3151.44/7274.58 us，
32 hashes与BW一致，8K持平、20K仅约0.4%改善，不足以确认端到端收益。
没有完整数值/sanitizer门禁，两个源文件的CG改动已从工作区撤回；CF和CA保留
待判定。不要把本次C写回TMA与BS/BY的Return打包TMA混为一项。

### CH：warp 协作生成 Dispatch consumer counts（保留）

基于CF，不包含CG。原来每thread逐token/route串行统计一个packet；现在每warp
静态拥有一个packet，route按lane展开，同目的GPU的lane使用match掩码聚合，
每组唯一lane更新该packet私有consumer计数。每批之间warp join，防止下一批
由不同lane更新同一项时缺少可见性；masked/非法/本节点以外route都不计数。
tail的无效lane仍参加match/reduce/join。packet布局、所有owner目录和精确
expected_count定义不变，不新增原子或分片引用图。调用者均为完整warp组。

复用GPU plan的31 source/packet+6 destination cases通过，包含mask、重复
expert、空节点、tail和三节点几何。编译及端到端门禁进行中，日志
`test_warp_packet_plan.log`、`ring_warp_packet_plan_*`。

CH native SHA256 `0903de0816a528faacd892d6c6b5f24523b9f253c11526d49da0926f9361fe37`，
library `8b5369a854a0d1b1d052fac951f10a8f41420d2766b82028c65800652ac7eab5`，
REG168/STACK440/LOCAL0。B512/D8快筛8K/20K为3124.190/7245.500 us，
32 hashes与BW一致；route planning status最大值287.328/522.752 us（CF
约310/537 us）。双机完整四规模+edges共416条PyTorch检查通过，两个launcher
均exit0。部分快筛有CPU编译；正式及sanitizer门禁仍待完成。

### CI：并行规划窗口几何与 RX 绑定（候选）

在CH上，网络窗口的control warp0计算source地址及expert/Mblock counts，
其余三个warp构建PullPacket。前者只用counts scratch，后者只读不可变source
packet目录，互不依赖；不同窗口的输出也不重叠。最终沿用原grid join发布两者。
warp0内部用warp同步，不能复用其它warp正在等候的128-thread named barrier。
共用source_plan_sync对Threads=32使用__syncwarp，其它线程数组仍原bar.sync。
单机仍128-thread原规划。没有新counter/barrier或额外metadata，编译/验证中。

CI native SHA256 `3afb28d1a5622fd879ec1f325a99cad79e5ce07f4be283963d7b500aeab6930a`，
library `beae1a2c543db500041d676672c176367c7b0dce2d5ea808aa822a53d3fa18ae`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_parallel_geometry/`。8K/20K
快筛3154.770/7286.980 us，32 hashes与BW一致；规划status反升至约299/538 us。
将RX绑定降为三个warp的代价超过重叠收益，未继续完整数值或sanitizer，已撤回
两处源文件改动，包括Threads32同步分支；不保留失败候选的复杂度。

### CJ：source token/route 目录统一现有 32-bit 索引合同（20K 退化，已撤回）

在CH上，不含CI或CG。Forward pool已有`max_tokens*top_k <= UINT32_MAX`
的边界检查，RingRow也保存uint32 route/token，但source目录仍用64-bit token
和route。将route_order改uint32，NodeToken从16B改8B；分配、绑定、生成和
读者类型统一。prefix、sequence、generation及跨窗口逻辑计数仍64-bit，不新增
更严格容量上限或运行时防御分支。已有GPU plan测试只修改对应数组元素类型。

每份source metadata节省`12*window_capacity*window_count` bytes，EP16两份
replica在20K/B512时合计节省1966080 bytes/rank，并同步减少source目录发布
量及读取带宽。此为容量推导，性能和完整门禁尚在进行；归档
`static_v4_ring_compact_ids/`。

CJ native SHA256 `ad8c9507bcdbbba3125aa0d92b582d1acf247d51432f4bd391634db626abd77a`，
library `36385b6075a5a5958f4dd1003e164d70e2ef69bfd83a32986d679b54d55b855a`，
REG168/STACK440/LOCAL0。host18/18、GPU plan31+6、EP16四规模+edges416、
B16/D1/K8小深度272均通过。1620MHz、10 warmup/20 samples/batch20、
两个节点无CPU编译/其它GPU任务的首轮正式四规模为
278.227/372.406/3074.870/7376.110 us，64 hashes与BW一致。
8K收益明显，但20K比BW/B512/D8的7312.640 us慢约0.87%，并未延续快筛趋势。
20K规划status约498us、低于BW约542us，主流水却变慢；下一步相邻轮次对照
和slot复用sanitizer，不能仅凭目录变小认定长序列端到端获益。

rank1/9 eager B16/D1/K8/T64、11 rounds的memcheck和racecheck两个launcher
均exit0，日志0 errors/0 hazards。memcheck进程local430043/remote143691，
racecheck local430684/remote143955；不声称覆盖所有16个rank。

EP8完整四规模+edges的208条PyTorch检查通过。同时间第二节点独立EP8正式
计时（无该节点CPU编译/其它GPU任务，1620MHz、10 warmup/20 samples/batch20），
统一B512/D8四规模101.472/137.682/1252.320/2957.200 us；注意此前BW/EP8
常用的是D4，不能把本轮差异全部归于目录压缩。日志 `ring_compact_ids_*`。

随后无并发编译/GPU任务的20K交替正式对照为CH/CJ/CH/CJ：
7315.140/7400.990/7309.750/7385.050 us。确认压缩目录虽然降低规划耗时，
端到端仍慢约1.0–1.2%，不是单次快筛波动。其节省仅约0.2%总pool，当前不以
这个容量收益交换长序列退化；六处源文件/测试类型改动已撤回，保留完整构建、
patch和正确性记录。8K与EP8的上述CJ数字不能作为撤回后的CH最终性能。
原子预约决定的source token顺序可能影响X-ready/主流水，但目前只是待验证
的假设，不能当作已确定原因；后续稳定枚举同时服务训练tape的确定性目标。

### Profile 切换区间初筛（同CF通信版本，尚未修改阈值）

强制prefill的独立native SHA256
`e8f0d0b73d7dcdb6430723e1d255a666e2706796fb9229a189ec343c7b8a2472`，归档
`static_v4_profile_prefill/`，临时config仅将原MIN_TOKENS覆盖为1。对照CF自动
入口在这些规模均为decode；两种profile都用packed gate/up，不重新打包权重。
第二节点独立EP8、1620MHz、B512/D8，10 warmup/2 samples/batch2：

| T/rank | decode/us | prefill/us |
|---:|---:|---:|
|128|150.064|161.408|
|256|182.224|193.152|
|512|252.720|245.872|
|1024|336.464|324.720|
|2048|496.464|461.392|
|4096|817.408|721.136|

所有96 hashes两profile一致。本节点在CPU编译，不占用第二节点CPU/GPU；
以上仍为两样本快筛而非正式阈值验收。EP16/B512/D8的T512快筛decode/prefill
644.048/649.904 us，尚不支持照搬单机T512阈值。日志 `ring_profile_*`。

补齐同版本EP16快筛：T1024为809.856/810.048 us，T2048为
1140.750/1135.220 us，T4096为1796.320/1770.290 us；四规模全部64
hashes两profile一致。仍需正式阈值验收，当前未改自动切换点。

### CH 里程碑验收（2026-09-24，CA/CE/CF/CH 合并提交）

源文件与冻结CH构建一致，不包含CB/CD/CG/CI/CJ。host18/18、GPU plan31+6、
EP16四规模+edges416、小深度B16/D1/K8的272、EP8四规模+edges208均通过。
rank1/9 eager B16/D1/K8/T64、11 rounds的memcheck与racecheck launcher
均exit0，分别0 errors/0 hazards；memcheck进程local434976/remote146341，
racecheck local435482/remote146618，不声称其余14 ranks受sanitizer检查。

固定1620MHz，B512/D8，10 warmup/20 samples/batch20，正式计时节点无
CPU编译或其它GPU任务。下列96 hashes全部与BW对应矩阵一致：

| T/rank | 单机EP8/us | 双机EP16/us |
|---:|---:|---:|
|1|101.763|278.987|
|64|138.230|376.650|
|8192|1253.880|3127.420|
|20480|2952.100|7309.750|

日志 `ring_warp_packet_plan_{remote_ep8,ep16}_t*_formal.log`；20K EP16取
前述相邻轮次CH2，另一次CH1为7315.140 us。EP8较冻结V3
300.557/370.549/1486.870/3299.890 us分别快66.1%/62.7%/15.7%/10.5%。
EP16 8K较V3 3359.26 us快约6.9%，20K仍比7099.96 us慢约3.0%，
因此只提交已验证的规划/collective范围优化，不标记最终性能目标完成。
与同B512/D8的BW正式3146.080/7312.640 us相比，长序列只是小幅/持平；
不能把之前B1024/D4切换带来的收益全归于这次代码修改。Ring仍保持opt-in。

下一候选CK只修改source窗口内的稳定枚举：warp拥有连续route/token区间，
用固定32桶×warp数的计数生成有序前缀，取消scatter的原子预约顺序。
保留现有64-bit目录及全部slot/credit合同。已有GPU plan测试增加按expert/node
的ID单调性检查，31+6通过；端到端尚待验证，不等同M6 StableTape已实现。

### CK：固定大小 scratch 的稳定 source 枚举（无收益，已撤回）

基于CH，仅替换source_window planner；不包含已撤回的CJ目录压缩。每warp
拥有连续的route/token区间，每次处理32个expert/node桶，先统计warp私有计数，
再按bucket/warp顺序前缀，最后按lane顺序scatter。expert散布阶段不再原子
预约位置；node列表仍按实际route去重并保留精确fanouts。scratch固定为
Threads个unsigned（当前128 threads即512B），不随T/E/node增长；expert/node
超过32以分批处理，不引入专家数或节点数的新容量上限。最终每expert内route
和每node内token严格递增；这只是确定性局部ordinal，不是完整训练tape。

GPU plan31+6及其racecheck通过，0 hazards。native SHA256
`debc0b3514dc8a7d5a402fb4cdc2baf681f35bfc58f585e3a7be74295e4cd9f9`，library
`26e609438081611df6f679e942ea88ddc5566e7a1bac0debf3a7aabc2885f81c`，
REG168/STACK440/LOCAL0，与CH相同。归档 `static_v4_ring_stable_source/`，
完整端到端与性能待验证。

Profile补充：同CF版本第二节点独立EP8、1620MHz、B512/D8的正式
10 warmup/20 samples/batch20，T256/512/1024的decode→prefill分别为
179.189→191.246、252.442→244.376、337.126→323.002 us，48 hashes一致。
当前证据支持单机在512附近切换；仍不据此设置多机相同阈值。

CK的EP16/B512/D8快筛8K/20K为3164.880/7443.540 us，32 hashes与BW一致；
规划status最大值约292/527 us，并未减少整体规划成本。双机四规模+edges的
416条PyTorch检查通过，两个launcher均exit0。但端到端比CH慢约1.2%/1.8%，
没有继续完整sanitizer或正式性能门禁，三处source/test改动已撤回。保留归档供
后续训练确定性规划参考，不给inference引入无收益的排序代价。

### CL：RX 从 immutable plan 维护逐 slot 退休目标（保留）

基于CH，不包含CK/CJ。Dispatch/Return的身份与精确expected本已存在双方
immutable目录，wire header唯一数据面读者只是ACK的retire_target。改由唯一
RX import owner在GIN readSignal确认本代ready后，从incoming packet目录取
fanouts/rows，累加该物理slot.retire_target，再发布visible_generation。
只有成功import才前进rx_cursor，故每代恰累加一次；ACK检查同一RX owner维护的
target，不读相反方向TX的描述。计数跨调用累积，仍无reset或运行时原子expected。

删除32-byte PacketHeader及其128-byte padding，移除冗余PacketLayout结构，
wire仅完整BF16行；GIN仍一次Put绑定逐slot ready，TX复用仍要求远端consumed
与本地NIC completion。source/Return计划、分包、GEMM及reduce顺序均不变。
减少不必要GM读写及协议字段，不预先声称性能收益。完整编译/门禁进行中。

CL冻结 `static_v4_ring_rx_targets/`，native SHA256
`d1bb89a1d31b18f75880b7f438c7c67dd98a76c78d06adb0e0de1c0eafe1be21`，library
`db5cec042b8f1c392dfbfd22266b8eee4c8f0e78622a2f4b5c1d14d8856281f7`；
REG168/STACK432/LOCAL0，stack较CH少8B。B512/D8快筛8K/20K为
3081.540/7125.570 us，32 hashes与BW一致；相对CH正式约快1.5%/2.5%，
尚需正式对照，不把快筛作为最终收益。
host18/18、GPU plan31+6、EP16完整416、小深度B16/D1/K8的272均通过。
rank1/9 eager B16/D1/K8/T64、11 rounds的memcheck/racecheck launcher
均exit0，0 errors/0 hazards；memcheck local449718/remote151295，
racecheck local450651/remote151554。日志 `ring_rx_targets_*`。

### CM：在当前通信配置下重测 Return RX 直接 Reduce（无收益，已撤回）

基于CL，重做AQ的独立消融；不是恢复AQ当时Q8/共享GIN资源的旧版本。
source-local ReturnBinding保存payload/ready/consumed/generation，plan阶段
只从immutable目录生成。Reduce保持原FP32顺序，读本地contribution或Return RX；
全部lane完成payload读取后，每条remote route归还一个RX row credit，K>32也
按lane stride覆盖。删除receive_return scatter及其consume_cursor，原scatter
warp转为静态Reduce residue，GIN仍由原固定RX owner独占。

RX容量必须至少容纳一个source window的最坏Return包数，深度为
`max(8, ReturnPartition.lane_packets(B*capacity_top_k, 0))`，使用Return Q，
不是Dispatch Q；同channel保持window单调顺序。无新原子expected或动态领用。
B512/K4/R16/L32的depth仍8；binding增加32*capacity_routes bytes/rank，
不声称省掉scatter就必然获益。需要重新验证长序列、热点和wrap。

CM冻结 `static_v4_ring_direct_rx_retry/`（含相对CL的单独patch），native SHA256
`110d9193c2b602794634600c33c1c5b27a33815c73a1aa0b02f5b2d61bec4e4c`，library
`7d0370194ad494883df7465a4d0da8fe4129d7239b8cdeb9d99e8c429195a94c`，
REG168/STACK448/LOCAL0。host18/18通过，端到端筛选进行中。

CM的8K/20K快筛为3125.920/7209.940 us，32 hashes与BW一致；均比CL的
3081.540/7125.570 us略慢，不保留额外绑定表与RX生命周期约束。没有继续完整
PyTorch/sanitizer门禁，八处CUDA源文件改动已撤回，恢复CL的RX scatter。

### CL 里程碑验收（2026-09-24）

除上面的host/GPU plan/EP16/sanitizer外，EP8四规模+edges208条检查也通过。
1620MHz、B512/D8、10 warmup/20 samples/batch20，正式计时节点无CPU编译
或其它GPU任务；EP8在第二节点计时，本节点独立跑数值检查及编译下一候选。

| T/rank | 单机EP8/us | 双机EP16/us |
|---:|---:|---:|
|1|101.757|280.597|
|64|138.259|377.998|
|8192|1254.380|3072.620|
|20480|2951.290|7149.840|

96 hashes均与BW一致。日志 `ring_rx_targets_{remote_ep8,b512_d8_ep16}_t*_formal.log`。
较CH同配置，EP8持平，EP16长序列分别快1.75%/2.19%，短序列约0.4–0.6%
的反向波动。EP16较冻结V3的3359.26/7099.96 us，8K快约8.53%，20K仍慢
约0.70%，所以最终“明显优于V3”目标尚未达成。每rank注册区减少131072 bytes
（两方向TX/RX共1024个slot各去掉128B），数据面实际wire也少每包128B；
没有新增分配或改变ring深度。Ring仍为opt-in，不把后续候选当成当前实现。

### CN：当前协议下重测 FC2 直写预留 Return TX（候选）

基于CL，不含CM。复用AY的地址/准入思想，但已采用当前THREAD独占GIN资源、
node-aligned group、rail-local C-ready以及独立Dispatch Q8/Return Q16协议。
Return TX owner在等待FC2前非阻塞地预留整个可用depth，can_send仍检查远端
consumed AND本地NIC completion；每channel发布单调reserved_head。
destination整窗准入同时检查所有实际涉及Return packets已预留，FC2才可直写TX。
source expert interval给出该window/rank的首末packet，每lane只检查最后一个。
Return包ready仍通过原FC2 completion→system fence→rail-local ready链，不
新增逐fragment完成图。发送owner不再复制C→TX，但保留C bookkeeping退休。

RingRow预绑定output指针，由16B增至24B，本地仍写source contribution；
remote写同节点rail的Return TX。网络depth至少容纳一个最坏Return source窗口，
计算使用`returning.partition`，不能误用Dispatch的Q8。D8/R16/B512/K4仍depth8。
暂保留C payload分配及原map，不将尚未实施的内存清理算作收益。

冻结 `static_v4_ring_direct_tx_retry/`（含相对CL的patch及return_tx helper），
native SHA256 `0af2dfcac3295f13dce304a6dc0d85ca698c7cee58b62e61d0241f66b68d1014`，
library `b06a14e9302baf990b2b4fc4a79df3d4a1f6786e2a610967905bd176f3e43f22`，
REG168/STACK440/LOCAL0。另编译Return Q32独立候选；尚未验收。

### 权重更新合同补齐（M5 子项，已验证）

Graph的W1 packed副本在prepare时生成，原先仅检查源权重地址；同地址原地更新
不会刷新它。新增显式`invalidate_inference_layer(layer)`：要求丢弃引用该层的
外层graphs，在capture外等待已发出工作完成，再释放该层全部token-bucket plans。
之后更新权重并重新warm/capture，重新publish权重产生新packed副本。
不在replay热路径查询版本或hash，也不声称pointer相同代表内容没变。
原有复用测试增加一次原地修改W1/W2后的全部token桶参考检查，没有新建测试套件。
为隔离通信候选，独立extension链接冻结CL library，目录`static_v4_weight_refresh/`。

该extension SHA256 `277fef7af8f36798877bd41256bdb63ae5a0768c666b25a1aa0486c5710aa488`，
底层library仍为CL的`db5cec...`，没有修改CUDA kernel。EP16完整四规模+edges
共480条检查、EP8共240条检查通过，所有launcher均exit0；分别含64/32条
`weight-refresh-graph`结果。W1/W2源地址保持不变，原地乘以0.75/0.875后，
全部四个token桶重新准备并与PyTorch参考比对。日志 `ring_weight_refresh_*`。
此项修正生命周期/API缺口，不记录为性能收益，也不标记整个M5已完成。

CN补充：Q16快筛8K/20K为3170.700/7221.650 us，Q32为3049.200/7089.870 us，
全部64 hashes与BW一致。Q16未优于CL；Q32略快，但20K仅贴近历史V3，仍需
正式对照、窗口联合消融和完整正确性门禁。Q32 native SHA256
`01a3b2d4395e25070022c7a3c660a077dfc3a05fcbb1873bd9766f98dee37d80`；
首轮Q16快筛存在本节点并行CPU编译，Q32快筛无并行CPU/GPU任务。

CN/Q32的20K窗口联合快筛，B256/1024/1536、contribution D8分别为
7630.660/7447.780/7404.620 us，仍不如B512。此轮有CPU构建，不作为正式
性能结论；三组48 hashes另核对。Q32 library SHA256
`228bfbe1aff7638e6bbc9b667c7ab2e8a3376cc39739348d244d6aae7abe594f`。
第二节点无其它CPU/GPU任务的EP8正式10/20/20、B512/D8为
101.034/137.973/1256.340/2964.990 us，单机与CL基本持平，不认定为收益。

### CO：删除直写 TX 路径已无读者的 C payload（与 CN/Q32 一同保留）

基于CN/Q32，FC2已经只写预绑定的source contribution或Return TX地址，
所以删除ComputeRingLayout.output及相应C数据分配；ring不再生成未使用的
output-store TensorMap。非ring的V3/full-arena与training map不变。
X/S slot中的FC2/returned计数仍作为计算与Return发布的生命周期账本，不能
因为数据C消失就删掉这些同步。容量节省与端到端收益分开记录，编译/验证中。

### CN/CO 里程碑验收（2026-09-24）

CN/Q32完整EP16四规模+edges480、小深度B16/D1/K8的304条检查通过。
CO进一步完整验证host18/18、GPU plan31+6、EP16四规模+edges480、EP8的240、
B16/D1/K8的304，以及B2048/D1/K8的304条检查；后者触发网络depth16，验证
整窗TX预留不依赖固定depth8。所有launcher exit0。沿用现有测试入口，无新增套件。
rank1/9 eager B16/D1/K8/T64、11 rounds的memcheck和racecheck均exit0，
0 errors/0 hazards；memcheck local470300/remote158747，racecheck
local470926/remote159026。不是对全部16个rank同时插桩。

CO冻结 `static_v4_ring_direct_tx_no_c/`，native SHA256
`c8b6c34dc92960509ebcd68295875ac17f2ab32dc15996494d5ae974b4e75a83`，library
`c414ff6636905b70f765e540f620e30e99ffb4ba72832c015cfacb39626c8e70`；
REG168/STACK440/LOCAL0，无spill。相对CL的完整source patch含新增return_tx helper。

1620MHz、B512/D8、10 warmup/20 samples/batch20，正式测量节点无CPU编译或
其它GPU任务，单机EP8在第二节点运行。下表不是后续规划线程候选的结果：

| T/rank | 单机EP8/us | 双机EP16/us |
|---:|---:|---:|
|1|100.995|291.643|
|64|138.315|420.726|
|8192|1262.370|3061.540|
|20480|2957.200|7099.110|

EP8与CL基本持平；EP16长序列较CL约快0.36%/0.71%，短规模反而慢3.94%/11.30%。
Q32与直写TX作为组合保留，不能将全部变化归因于删C，后续仍须优化短序列分包。
相邻冻结V3的8K/20K为3363.990/7106.450 us；CO的20K再测7102.090 us。
所以8K快约8.99%，20K只是持平，最终“明显优于V3”的目标仍未达成。
EP8/EP16四规模96 hashes与CL一致，另32 hashes与相邻V3一致。
日志 `ring_direct_tx_no_c_*`，formal与quick严格区分。

以20K/rank为例，EP8 device pool从CL的688751552降至551715776 bytes；
EP16从1008139712降至913972672 bytes。EP16 compute/registered分别为
820332216/93640448 bytes，CL为947930808/60208896 bytes：直写地址表有增量，
Q32也比CL的Q16多占注册内存；不能只报告删除的C而隐藏这些成本。
相对同Q32的CN，CO精确删除138412032 bytes（132MiB）C payload/rank。
ring仍为opt-in，未将本项标成M5最终profile或训练支持完成。

### CP：输入 staging 后复用全部 CTA 线程规划（候选）

基于CO。输入X/gates staging与128-thread source counting保持重叠；原有CTA
join之后，原来等待到PlanReady的256个math线程也参与packet prefix、目录、
窗口几何、PullPacket和route地址绑定。所有后续grid/rank/GIN控制面barrier的
参与者统一改为整个384-thread CTA，最后不再需要唤醒闲置math的第二次join。
source ordinal、分包和静态GEMM任务不变，没有增加plan/通信buffer或计数器。
数学warpgroup的动态寄存器配额调整仍在prepare之后，规划时没有live GEMM。
本项编译/性能筛选中；尚未替代CO，不能将其归入上述验收结果。

CP native SHA256 `042816f5a7b2c341cb48fcf2cd6e3bd9e10e944d681efa0e01bd84ff2f74fa6a`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_plan_all_threads/`。无并发构建的
四规模快筛296.208/412.416/3036.750/7036.980 us，64 hashes与CO一致。
8K/20K planning status最大值261.504/481.184 us，CO快筛为292.640/528.224 us。
局部规划约省31/47 us，端到端只小幅改善；尚未完成正式与完整数值门禁。

### CQ：共用 warp packet-prefix scan（候选）

基于CP。原先每thread串行遍历一个channel的全部windows，现为每warp静态
拥有一个channel，按32个windows一批做inclusive scan并保留64-bit carry。
Dispatch的node-token prefix与Return的node-expert prefix使用同一helper；
不改变packet顺序、目录布局或expected，不增加scratch/原子/barrier。
无效tail lane以0参与scan，空channel和0-window仍写prefix[0]。
现有GPU plan矩阵加4097 tokens，覆盖33-window的最后一批仅一有效lane，
其它验收仍沿用原入口。候选编译中，不预先声称收益。

CQ native SHA256 `04df01211b31a1e25078e62b3e0e5a68a69cad8a81bcdeffbf0780a6f8e1375b`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_warp_prefix/`。GPU plan37+6通过，
isolated plan racecheck为0 hazards。无并发编译的四规模快筛为
295.184/410.496/2990.780/7004.610 us，64 hashes与CP/CO一致；planning
status最大值8K/20K为242.624/419.200 us。相对CP省约19/62 us规划，
端到端省约46/32 us；正式及完整融合kernel门禁进行中，不据此标记目标完成。

CQ library SHA256 `326eaeb91348c612b2bb33646b7c8937fd8f01489105c4583fa208a84ff0f4d7`。
完整EP16四规模+edges480检查通过。第二节点无并发CPU/GPU任务的EP8正式
10/20/20为100.659/135.357/1270.060/2956.290 us，32 hashes与CO一致；
本节点独立编译library。单机基本持平，不将20K的不足0.1%变化视为收益。

### CR：静态 ring CTA 数量消融（无收益，已撤回）

基于CQ，临时host配置`WEFTMOE_STATIC_RING_CTAS`限制实际launch CTA数，
不修改kernel task ownership或角色公式。值仍受已有双CTA cluster与网络进度
所需context数量校验，非ring不受影响。launcher显式向两节点转发该值。
用于比较132/128等CTA数与窗口大小的组合，检验静态residue相位及角色配额；
没有运行时领用或work stealing。此配置尚未决定保留，编译/实测中。

CR native SHA256 `c465d15a6edde8e763bc1dc2147c2f883c664fa5416e8b3281085d0801b8b279`，
冻结 `static_v4_ring_cta_cap/`。EP16/B512/D8/20K、无并发编译的快筛，
CTA128/120/112/132分别为7064.270/7147.500/7950.720/7015.200 us，64 hashes
均与CQ一致。减少CTA没有收益，原132 CTA胜出；没有继续完整数值/sanitizer。
已撤回host配置入口及launcher转发，不保留无收益的调参分支。

### CP/CQ 里程碑验收（2026-09-24）

完整EP8四规模+edges240、EP16的480、B16/D1/K8的304条检查通过；GPU plan
37+6及isolated plan racecheck均通过。rank1/9 eager B16/D1/K8/T64、
11 rounds融合kernel的memcheck/racecheck均exit0，0 errors/0 hazards；
memcheck local487563/remote165082，racecheck local488186/remote165350。
原host布局/容量代码未改变，沿用CO已验证的18/18。所有执行launcher exit0。

正式1620MHz、B512/D8、10 warmup/20 samples/batch20，计时节点无并发构建
或其它GPU工作，EP8在第二节点运行：

| T/rank | 单机EP8/us | 双机EP16/us |
|---:|---:|---:|
|1|100.659|290.987|
|64|135.357|414.488|
|8192|1270.060|3007.870|
|20480|2956.290|6989.360|

96 hashes与CO一致。EP8基本持平，EP16长序列较CO正式快约1.75%/1.55%。
相对前一CO对照轮次V3的3363.990/7106.450 us，8K/20K快约10.59%/1.65%；
20K仍未达到最终“明显优于”的要求。不能把规划局部省下的全部时间当成总时延收益。
内存布局、stage资源和数值顺序不变，REG168/STACK440/LOCAL0，仍一kernel graph。
日志 `ring_warp_prefix_*`；不含CR或后续Pull合并完成候选。

### CS：两个 remote Pull 批次共用 full completion（无收益，已撤回）

基于CQ，不含CR。保留原packet/worker residue和双buffer row prefetch，只让
最多两个32-route warp批次共用一次prefetch drain/global store completion。
每lane暂存前一批的X-full地址与credit，不增加GM/SMEM；完整packet的RX lease
最多延迟到这次flush。所有TMA完成后才发布两批X-ready，再归还已完整消费的RX。

进入下一个packet时先非阻塞检查ready；若它未ready，必须先flush/归还旧包，
再等待新包。窗口结束也强制flush，不能持有旧RX credit等待下一窗口准入。
较大Q/K仍按32-route批次前进，只在整个packet的最后一批完成后归还该包credit。
Expected仍是immutable desc.rows，返回计数不改变；只改remote Pull，不混入
本地Pull batch或Return warp重写。候选编译/验证中。

CS native SHA256 `79713f5627c5dbae87aceeeecbeeb63cc397ad448c90d78c0019e1577e70b306`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_pull_batch_pair/`。无并发构建的
四规模快筛293.360/422.960/3002.000/7224.260 us，64 hashes与CQ一致；
20K较CQ明显退化，未继续完整数值或sanitizer。已撤回network_transport改动，
恢复逐批完成后立即发布X-ready/归还RX。不能因减少drain次数就认定更快。

### CT：短 source 按实际长度选择 Return Q（短规模夹具未生效，已撤回待修正）

基于CQ，不含CR/CS。source实际tokens<B时，令
`Q=min(configured_Q, pow2ceil(ceil(tokens*K/lanes)))`，至少1；完整窗口仍用
configured_Q。source planner和所有destination从同一source epoch推导Q，
不使用destination自己的token数，因此不等长rank可以使用不同Q而正确互通。
main形状下T1/T64分别为Q1/Q8，8K/20K仍Q32。TX/RX物理slot stride仍按最大Q
分配，wire仅desc.rows个完整行，不增加payload容量或新协议字段。

若Q小于配置上限，整source至多lanes个packet/目标node；否则沿用原最坏
窗口包数。因此每peer/window的目录容量上界为`max(lanes, ceil(B*K/Qmax))`，
再受总routes上界限制；网络depth原有`max(8, ceil(ceil(B*K/Qmax)/lanes))`
仍足够整窗预留。只调整小B的必要目录上界，不按最大T增加data-plane buffer。
现有GPU plan参考按所选partition枚举，端到端必须覆盖不等长、空source和复用。
候选编译/门禁中，不把预期短序列收益写成已测结果。

CT native SHA256 `a8145faea3621ade135539ae61727dc292515af33e3f57019b5bff0291b1e6aa`，
library `297d08ab0fa8846c478ba6450282eba32892a7a2729d074a94d8e311736f8177`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_short_return_packets/`。
GPU plan37+6与isolated racecheck通过；rank1/9 eager B512/D1/K8/T64、
11 rounds的memcheck/racecheck exit0，0 errors/0 hazards；当时误判实际Return Q16，
但后续确认此夹具仍Q32，不能作为自适应Q16的门禁；
memcheck local499364/remote169072，racecheck local499774/remote169342。
无并发构建的四规模快筛294.800/410.592/3002.780/6976.700 us，64 hashes与CQ
一致。此短序列性能结论无效，原因见下面更正；长序列差异不足以认定收益。
未继续完整PyTorch门禁，已撤回全部五处source/test改动，保持统一Q32及原目录容量公式。

**夹具更正**：`static_ring_shape`令实际B为`min(configured_B,max_tokens)`。
native benchmark每个T单独建pool，所以T1/T64时tokens恰等于实际B，CT的
`tokens>=B`分支直接返回Q32。上述无收益不能否定短包策略，sanitizer也没有走
所宣称的Q16。应按`min(tokens,B)*K`选Q，不以“是否完整窗口”跳过；这样实际
工作量相同而pool容量不同也得到相同Q。修正版须重做性能与融合kernel门禁。

### CU：warp 协作读取 Return 控制元数据（代码归档，尚未编译）

候选令整warp读取packet的route/location，按ComputeSlot地址match合并ready
检查与逐行retirement；GIN reserve/readSignal/readCounter/Put仍只有lane0提交。
可变sequence、reserved和drained由lane0广播，所有lane选择相同channel/packet。
保存于 `static_v4_ring_return_warp_control/from_short_packets.patch`；没有
编译、数值或性能结果。该增量只改network_transport，已从工作区撤回，先独立
验证下面更小的准入检查改动。不能将未构建候选称为优化收益。

### CV：整窗 TX reservation 并行检查多个 source rail（候选）

基于CQ，不含CT/CU。原warp按source串行循环，每次仅该destination实际涉及
的几个packet lanes工作；改为`source_index=worker%G`，每source获得自己的
worker residue，直接枚举首末packet区间实际覆盖的lane，检查其最后packet。
每source的worker数为`floor((31-source_index)/G)+1`，不假设G整除32。
G至多16、准入固定一个warp，故所有source都有worker；本地、空source直接满足。
条件仍是原预留head与target，最终仍__all_sync整窗准入；没有新buffer、字段、
counter或动态任务领用。临时枚举验证不同G、L与区间的精确覆盖/无重复，
源代码编译与端到端筛选中。

CV native SHA256 `5b23945548cdef2beac2d69a74d5f9952e52050f9b3ea9d848548a240b49ff5c`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_parallel_tx_reservation/`。
6048组G/L/区间枚举与原条件精确一致、无重复。无并发构建的EP16四规模
快筛281.472/397.504/3004.690/7007.380 us；64 hashes与CQ一致。
短序列有初步改善，长序列未见明确收益；尚未做完整数值与正式门禁。

### CW：修正 pool 裁剪后的短窗口 Return Q（候选）

基于CV，不含CU。修正CT的完整窗口早退，始终按
`Q=min(configured_Q, pow2ceil(ceil(min(tokens,B)*K/lanes)))`选择，最小1。
同样实际窗口负载不再因pool容量不同而得到不同Q；sender规划与destination
直写/准入均从source epoch推导，physical slot stride仍按最大Q。
每peer/window目录容量取`max(lanes, ceil(B*K/Qmax))`并受总routes上界限制；
若Q变小，每window至多一个packet/lane，原网络depth约束仍足够。
GPU plan原矩阵的短形状改为与native一致的B裁剪，37+6通过，明确输出
T1/B1/Q1、T64/B64/Q8。融合算子编译/门禁中；不能沿用CT未激活Q的测试结论。

CW native SHA256 `24e7a59a495ec60778d0f37e8f7a6ad3f926955b3b430d468b57a7d440075ebf`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_adaptive_return_packets/`。
host18/18、GPU plan37+6、isolated plan racecheck0 hazards通过。
无并发构建的EP16四规模快筛280.224/335.456/3003.180/6978.110 us，64 hashes
与CV/CQ一致。Q8的T64较CV快约15.6%，其余规模未见明显独立收益；正式与
完整PyTorch门禁待完成。T64目录容量增加1024 bytes/rank，其余主规模容量不变。

rank1/9 eager B16/D1/K8/T64、11 rounds（实际Q4）的融合kernel memcheck与
racecheck均launcher exit0、0 errors/0 hazards。memcheck local512405/
remote173153，racecheck local512932/remote173422。不声称所有16 ranks均被
instrument。与CT不同，这次确实覆盖了自适应Q及小窗口反复wrap。

CW library SHA256 `e0a6944f1b500a75e2ef749a2ff74f01d2909cf0244435ed0716cdff803abc76`。
现有host测试补上相同实际T在B裁剪前后Q不变的断言，18/18仍通过；不增加
独立测试入口。完整EP8四规模+edges240检查通过。
第二节点1620MHz、B512/D8、10/20/20正式EP8为
100.739/135.402/1263.650/2958.490 us，32 hashes与CQ一致；计时节点无并发
CPU构建/GPU工作，第一节点独立构建library。单机基本持平，收益待EP16正式确认。

完整EP16四规模+edges480、B16/D1/K8的304检查也通过，两端launcher均exit0。
前者包含混合64/8192、空/不等长source、热点重放和权重失效更新；小窗口
测试确实使用自适应Q，不沿用旧统一Q32门禁。EP16正式复测待完成。

### CV/CW 里程碑验收（2026-09-24）

上述host、GPU plan、EP8/EP16完整与tiny数值、rank1/9融合sanitizer均通过。
1620MHz、B512/D8、10 warmup/20 samples/batch20正式矩阵如下；计时节点
无并发CPU构建/GPU工作，EP8在第二节点执行：

| T/rank | 单机EP8/us | 双机EP16/us |
|---:|---:|---:|
|1|100.739|276.125|
|64|135.402|333.984|
|8192|1263.650|3008.180|
|20480|2958.490|7018.410|

96 hashes与CQ一致，仍为一kernel graph。EP16的1/64较CQ正式分别下降
5.11%/19.42%；8K持平，20K反向波动约0.42%，不声称长序列收益。
REG168/STACK440/LOCAL0；仅T64的Return目录增加1024 bytes/rank，主测其它
规模内存不变。日志 `ring_adaptive_return_*`；不包含后续warp Return候选。
保留CV/CW以恢复并改善短序列，ring仍opt-in。参考前一CO对照轮次V3的20K
7106.450 us，本轮只快约1.24%，不是新的相邻V3测量，也不是最终性能达标。

### CX：Return warp 控制元数据协作读取（无收益，已撤回）

在CW上实际构建CU归档方案：warp并行读取route/location，以ComputeSlot地址
match合并ready检查和returned计数；GIN reserve/counter/signal/Put仍仅lane0。
sequence/reserved/drained由lane0广播，ready acquire之后warp join再提交Put，
retirement结束join后才推进cursor。没有新buffer、目录或expected计数。
冻结 `static_v4_ring_return_warp_control_retry/`，native SHA256
`4efafa95f260f63e0a9115ec31e5d4b41d3ed74208d0863578bb4c01425833a6`，
REG168/STACK440/LOCAL0。无并发构建的四规模快筛为
279.056/328.032/3054.350/7136.160 us，64 hashes与CW一致。
T64只有小幅初步改善，长序列反而退化，不保留整warp控制分支和helper；
network_transport改动已撤回，增量patch/二进制归档。没有继续library、完整
PyTorch或sanitizer门禁，不把并行元数据读取本身当成已证实的收益。

### CY：当前 Return 控制阶段的时间线诊断

CW原schema11构建 `static_v4_ring_adaptive_return_timeline/`，native SHA256
`6c342453112b92dfc3ef85e2ad15cb95db34fd3146aae67f2bc82f22c3dbc4d9`。
EP16/B512/D8/20K eager11 rounds，每rank156715 events，全部overflow0。
只在同GPU、该rail本身拥有的四个expert上，以本地Return发布结束对应packet
Put事件，2560个匹配样本间隔p50为57.280 us；不能跨rank比较globaltimer。
RX观察到ready→scatter开始p50为2.496 us，scatter本身51.680 us。

进一步增加只在profiling构建生效的ReturnPacketReady、ReturnPutStart两个
时间点（schema12），区分退休元数据循环与GIN调用本身；普通构建无新事件/同步。
`static_v4_ring_return_control_timeline/` native SHA256
`6501776856b730ec0e3bdf74f40534b3729d4218f490ae39bf407d7e77e87fad`。
同配置每rank159275 events、全部overflow0，两轮32 hashes与CW的20K结果一致。
20480个packet样本中，ready后退休阶段p50/p95为26.336/31.136 us，
Put提交13.184/16.704 us；scatter为51.712/61.632 us。上述本GPU匹配样本
发布结束→全包ready观察为19.840/29.632 us，发布结束→Put为59.264/71.104 us。
不同样本集合的分位数不可相加解释端到端；插桩时延也不替代CW正式7018.410 us。

Put返回后的时间点不能全部归因于C-ready轮询。此处曾依据RMA_IB_PROXY日志
解释为GIN proxy描述符/队列开销，但这是错误的后端归因，见下方实际devComm
核验；时间线数值不变。后续分别考察重复route/location查询与RX scatter，
不改变双credit、release范围或数值顺序。
原始日志/JSON为 `ring_{adaptive_return,return_control}_b512_d8_ep16_t20480_timeline*`，
包级同GPU匹配脚本和结果另归档于build目录；通用分析器已识别新增事件。

### CZ：RouteLocation 预绑定 destination（无收益，已撤回）

在CY/CW上，利用RouteLocation原有4-byte padding保存destination rank，
结构仍24 bytes。原有destination planner写入，Return ready/retirement
直接读取，不再额外gather selected_experts并除以owned_experts。
不改变Pull的过滤策略、分包/分组、原子次数、静态owner或发布范围；没有
新增buffer/barrier，也不混入已撤回的CX warp方案。依据CY的控制面诊断进行
小改动消融，正在编译；收益与完整融合kernel门禁尚未确认。

CZ native SHA256 `3007cb027b4063ce219f3735157e0328ea8032cecffa03a0062acfede270bb2f`，
REG168/STACK432/LOCAL0，冻结 `static_v4_ring_return_location_destination/`。
无并发构建四规模快筛278.576/334.528/3009.100/7000.930 us，64 hashes和
所有pool bytes与CW一致。初测基本持平，不能把少一次gather或stack少8 bytes
当成已证实的加速；尚未继续library/完整门禁。

CZb仅再将ready地址中的`destination%local_world`换成
`destination-node*local_world`。Return packet已限定本compute node，目的GPU
必在本节点区间，故不需逐行除法取模；没有新增前提、检查或协议字段。
独立构建 `static_v4_ring_return_location_relative/`，待性能筛选。

CZb native SHA256 `11008fc1c6504eee830148d5e25b31e94a137541201a215e48f5492c75231034`，
REG168/STACK432/LOCAL0。无并发构建四规模快筛为
280.128/336.048/3001.680/7028.080 us，64 hashes与全部pool bytes与CW一致。
CZ/CZb均未见稳定收益，三个源文件的候选改动已撤回，保留冻结patch/二进制；
没有继续library、完整PyTorch或sanitizer门禁。

补充CW的单机profile边界正式基线（第二节点、无并发任务、1620MHz、10/20/20）：
T256/512/1024的decode为176.498/245.642/328.502 us，日志
`ring_adaptive_return_profile_decode_remote_ep8_*`。同通信版本的强制prefill
对照尚待完成；不直接用早先CF数据宣布最终阈值。

同一第二节点再测冻结V3四规模正式为300.432/370.392/1488.160/3299.860 us，
32 hashes与CW一致。相对本节CW的EP8正式，V4分别降低66.47%/63.44%/15.09%/
10.34%。日志 `ring_adaptive_return_adjacent_v3_remote_ep8_*`；此轮只更新单机
对照，不冒充双机V3新测量。计时节点无并发CPU构建或其它GPU任务。

### DA：Return ready 扫描缓存退休组（无收益，已撤回）

基于CY/CW，不含CZ/CZb或CX。仍由单lane owner扫描全包ready；成功acquire一个
新的连续C slot组时，将slot指针与行数缓存在其闲置的Return TX row-copy SMEM。
只有整个packet均ready后才按缓存退休，未ready重试不产生任何退休；不再第二次
gather routes/selected_experts/locations。组数至多Q，复用原两个staging segment，
不新增SMEM/GM分配或同步；静态owner、双credit、GIN提交及release范围均不变。
正在构建验证，尚无数值或性能结论。

DA native SHA256 `342055207d9415277ea37d716f8e2b4f409ef5e2f35f30e1984d9b7b848c2405`，
REG168/STACK432/LOCAL0，冻结 `static_v4_ring_cached_return_retirement/`。
无并发构建的四规模快筛278.304/335.392/3021.520/7037.980 us，未见端到端收益，
已撤回源代码，仅保留patch/二进制记录。未继续library、完整PyTorch或sanitizer。

### 当前通信版本的 profile 边界对照（M5，待双机及交错复测）

冻结CW/CY数据面，仅把prefill阈值强制为1；未混入DB候选。native SHA256
`476a8788e3ed058a2257a2ad5dee520d4348f9bac19f3b4625c5025d112c8334`，
config SHA256 `9ad387de8bb8c96a8f20f3faebd0cc3fdd13679793768d2d39bfc644c3371f70`，
目录 `static_v4_profile_current_prefill/`。第二节点单机EP8，1620MHz、B512/D8、
10 warmup/20 samples/batch20；计时节点无其它GPU工作或CPU构建。

| T/rank | decode/us | 强制prefill/us |
|---:|---:|---:|
|256|176.498|188.832|
|512|245.642|241.261|
|1024|328.502|322.075|
|2048|490.650|456.186|
|4096|806.250|723.029|

40 hashes全同。单机候选阈值512，尚待反向交错复测；双机独立阈值还未确认，
未修改默认。日志 `ring_{adaptive_return_profile_decode,current_profile_prefill}_remote_ep8_*`。

### DB：直接 RX Reduce + packet 聚合回收（无收益，已撤回）

基于CW/CY，不含CZ/DA。移除RX→contribution scatter，Reduce从预绑定的本地
contribution或Return RX读取，复用相同数值循环/顺序，仅以模板替换行地址选择。
每token的所有warp lanes读取完成后release发布有界`token_done[D_contrib,B]`。
原scatter owner保留静态channel/consume_cursor，warp检查packet涉及的全部token，
lane0一次归还desc.rows；GIN import/ACK仍归原owner，双TX credit不变。
token_done用`>=generation`，允许contribution先复用、旧packet后回收。

与CM不同，不新增逐route大表：用24-byte RouteAddress union复用原RouteLocation
区域。远端source副本仅写compute成员给Pull/TX；source自身副本仅写returning
成员给Reduce。节点内Pull只读RingRow，删除其没有读者的location写入，避免
不同成员并发写。local routes与remote packet绑定集合互斥，沿用现有plan末尾
barrier，不额外加一轮grid同步。元数据区不属于published_bytes，不增跨机流量。
仅网络模式新增D×B×8 bytes的token_done，主配置32KiB/rank；编译/正确性/性能待验证。

DB native SHA256 `775a28c0e2bb5c29a14403dce25a148e57f7d5a8c73fb269dc052805ccb1a389`，
REG168/STACK456/LOCAL0，冻结 `static_v4_ring_direct_rx_packet_ack/`。
host18/18通过；无并发构建的四规模快筛281.920/348.992/3082.020/7087.680 us，
64 hashes与CW一致。pool增量为256/4096/32768/32768 bytes/rank，registered不变。
长短均未获益，已撤回全部七处源文件改动，不保留union、额外状态和地址accessor；
没有继续library、完整PyTorch或sanitizer门禁。原始patch及二进制留档。

单机profile反向交错补测（prefill→decode）：T512为241.466/245.411 us，
T1024为322.469/328.950 us，与首轮选择一致；日志
`ring_current_profile_reverse_*_remote_ep8_*`。仍未启用最终默认配置。

### GIN 实际后端核验与 CY 归因纠正

旧日志的`RMA/Plugin: Assigned plugin RMA_IB_PROXY`属于RMA，不能用作GIN的
后端证据；AZ的link benchmark实际报告GDAKI。现用临时host-only hook读取
冻结CW二进制每次成功ncclDevCommCreate返回的ginNetDeviceTypes，不改配置、
同步或payload。16个rank的activation对象均为contexts96/connections1/types3，
即GDAKI；Notify对象为contexts1/connections1/types3。
日志 `ring_current_actual_gin_rank*.log`，launcher
`ring_current_actual_gin_backend_retry.log` 正常退出，T1的16 hashes与CW一致。
第一次hook写stderr被成功路径launcher过滤，没有留下类型证据；第二次独立
保存逐rank stderr。hook脚本/源码仅build归档，不加入生产代码，其计时不用于性能。

因此CY中关于CPU proxy队列/GFD的归因撤回。安装的GDAKI Put调用DOCA GPU
verbs，而且当前该路径不使用传入的descriptor SMEM；不会基于proxy实现继续
做共享内存描述符实验。已有THREAD sharing和双credit保留，未改release范围。

### DC：按拓扑选择 decode/prefill 切换阈值（M5，门禁中）

同CW通信、1620MHz、B512/D8、10/20/20正式双机对照；括号为反向交错复测。
计时期间两节点无CPU构建或其它GPU任务，临时后端hook未用于以下测量。

| T/rank | decode/us | 强制prefill/us |
|---:|---:|---:|
|512|606.181|610.050|
|1024|756.120|753.054|
|2048|1057.820（1057.490）|1050.750（1049.290）|
|4096|1712.500（1712.910）|1682.990（1683.160）|

首轮64及反向64 hashes相同。2048虽只降低约0.7%，但两轮均可重复，故将最初
候选4096提前为2048；1024约0.4%的单轮差异尚不足以提前。结合上方EP8两轮，
新配置 `configs/static_inference_ring_20260924/` 选择单机512/多机2048。
两者仍packed gate8/up8；decode物理FC1 N64/FC2 N128/K128/stages3/3，
prefill物理FC1 N128/FC2 N256/K128/stages2/2。没有复制通信/调度实现或权重副本。

新增网络阈值宏默认继承原阈值，旧生成配置行为不变；任一入口启用prefill才
构建相应TMA maps与kernel分支。这里只冻结已测H2048/I512配置，不将跨机2K
阈值称为所有模型/节点数的最优值。完整数值、四规模性能及最终验收待完成。

DC冻结 `build/static_v4_profiles_20260924/`，config SHA256
`10ac19f097a5c37c4315629cfe44971a235ded1b7df15fa6a4e75908ecf0131b`，native
`4e7d10ba7f9d66726b5f2907fe668b280204836ad45bb5d45f323a80e6d8d1a5`，library
`5b464a5afa574534344ef0f9518053d8a9b53926a97ab3ad07ea97e3b8d488a6`。
网络prefill REG168/STACK440/SHARED2048/LOCAL0/CONSTANT3264。
host18/18；完整Torch的T1/64/512/2048/8192/20480与edges：EP8共288、EP16共576项
通过，三端launcher均exit0。未改数据面协议，不把CW的小深度sanitizer冒充DC重跑。

第二节点单机EP8正式四规模为100.717/135.424/1258.980/2949.830 us；
阈值两侧T512/2048为241.928/456.106 us。同轮后续冻结V3为
300.467/369.942/1488.860/3297.280 us。双机正式数据仍在测量。
日志 `ring_profiles_final_{remote_ep8,adjacent_v3_remote_ep8}_t*_formal.log`、
`ring_profiles_final_torch_ep{8,16*}.log`。

DC最终双机正式计时及相邻冻结V3如下，均1620MHz、10 warmup/20 samples/batch20，
测量时两节点无其它GPU任务或CPU构建。05:06 UTC复核全部16卡实际及应用时钟
1620/3201MHz。DC与CW的64 hashes、全部pool/registered bytes一致，graph仍一kernel；
与本轮V3的64 hashes也一致。单机与CW/V3的各32 hashes一致。

| T/rank | 单机EP8 V3/us | 单机EP8 DC/us | 延迟降低 | 双机EP16 V3/us | 双机EP16 DC/us | 延迟降低 |
|---:|---:|---:|---:|---:|---:|---:|
|1|300.467|100.717|66.48%|685.624|275.669|59.79%|
|64|369.942|135.424|63.39%|823.624|333.766|59.48%|
|8192|1488.860|1258.980|15.44%|3357.020|3006.200|10.45%|
|20480|3297.280|2949.830|10.54%|7096.930|7011.600|1.20%|

EP16/20K total/compute/registered为913972672/820332216/93640448 bytes/rank，
未因profile选择增加内存。两轮四规模launcher均exit0；日志
`ring_profiles_final_{ep16,adjacent_v3_ep16}_t*_formal.log`。这些端到端数字同时
包含已合入的ring优化，不能全部归因于DC的阈值；四个指定规模的入口本来就未变。
M5的布局/profile选择及当前通信版本验收完成，但双机20K仅降低1.20%，整体目标
仍未达成，ring仍opt-in、M6仍未实现。继续通信优化，不把本表作为最终交付。

### DD：静态 owner 的 GDAKI 后端特化（编译失败，已撤回）

尝试在每个Dispatch/Return/RX owner进入时一次判断实际GIN backend集合，
全部GDAKI时实例化 `ncclGin_BackendOne<NCCL_NET_DEVICE_GIN_GDAKI>`，其它后端
保留通用实现。没有修改packet、scope、counter或双TX credit。当前安装的
NCCL2.31.2中BackendMask的put/signal调用了仅接受通用`ncclGin const&`的
signal descriptor/counter helper，Mask8无法转换，故native编译失败。

不修改系统NCCL头文件，也不为尚无性能证据的实验引入项目兼容层；已撤回三个
源文件修改。归档 `build/static_v4_ring_gdaki_owner/from_profiles.patch` 与
`build/build_ring_gdaki_owner_native.log`。没有可运行二进制、数值或性能结论；
这不是GDAKI本身性能差的证据，生产路径本来已使用GDAKI。

### DE：直写 TX 后的 Return Q64（无收益，不采用）

只将DC构建的Return Qmax从32调整为64，Dispatch Q8/L32/B512/D8及数据面不变。
短source仍自适应Q1/Q8，主形状长序列用Q64；每source/window的远端packet
由32变16，但活跃lanes也减半，单packet等待范围及物理TX/RX注册区变大。
此前Q64候选包含C→TX拷贝，故在当前FC2直写版本重新独立筛选，未修改默认配置。
冻结 `build/static_v4_ring_direct_return_q64/`，完整数值/性能结论待测试。

native SHA256 `999e3af42b6f3277a815ef3df8121b2101e97aa34f0a7275b313eda49c6295e1`，
REG168/STACK440/LOCAL0；无并发构建的四规模快筛为
280.224/333.904/3094.640/7022.930 us，64 hashes与DC一致。
20K total/registered为981020096/160687872 bytes/rank，相对DC分别增加
67047424/67047424 bytes；短序列即使自适应小Q，物理slot也多占67108864 bytes。
未获得明确端到端收益，不采用Q64，不继续library、完整Torch或sanitizer。
源码没有增加Q64分支，默认仍Q32；归档provenance及四规模日志。

归档更正（后续DU实验）：本目录名被误复用，DE原native及四份快筛日志被覆盖，
原始产物尚未恢复。原provenance、source commit a0fd05f、构建日志与上表汇总仍在；
DU新产物已移至带编号的独立目录，不能拿它重算或验证DE。此处明确保留证据缺口。

### DF：Return 复用已完成的 system release（正式无收益，已撤回）

独立基于DC/Q32，不混入DE/Q64。FC2普通peer stores→device fc2_done release
链→local Return acquire全部N tiles→既有system fence→warp join→rail-ready
system release→TX owner逐packet acquire全部ready，已把该包payload发布到
system范围。TX owner随后只退休C账本，不再写TX payload。因此只对Return
将GIN givenRelease设为system，requiredRelease仍system；Dispatch保留block→
system，ACK/双TX credit/终端drain均不变。不是删除生产者的system fence。

本机头文件及同版本[NCCL GDAKI实现](https://raw.githubusercontent.com/NVIDIA/nccl/v2.31.2-1/src/include/nccl_device/gin/gdaki/gin_gdaki.h)
在required=system而given范围不足时补fence；V3 send_packed在packer发布链后
也传system/system。上述候选依赖本项目已存在的完整发布链，这是源代码推导，
不是通过数值测试证明memory model。尚无性能结论，获益后才继续完整门禁。

DF native SHA256 `a8ce13fd3505602c4e5d512f5733a6c871d3c57eabce4483eeba10c6a8445d8f`，
REG168/STACK440/LOCAL0，冻结 `static_v4_ring_return_given_release/` 与change.patch。
无并发构建的四规模快筛280.192/335.216/2996.460/6966.320 us；8K/20K正式
交错复测中，尚未完成完整Torch/sanitizer。源码已暂恢复保守fence，供DG独立实验。

正式DF→DC交错复测（两节点无其它CPU/GPU任务，10/20/20）：8K为
3006.590/3008.890 us，20K为7011.260/7015.670 us；小于0.1%的差异不作为
收益，初筛约0.6%的改善未复现。32 hashes与全部pool bytes相同，四规模初筛
64 hashes也与DC一致。没有继续完整Torch/sanitizer，保留原block→system
调用，不引入候选分支。日志 `ring_return_given_release_{df,dc}_ep16_t*_formal.log`。

### DG：plan 阶段预合并 Return 的 C-slot 段（无收益，已撤回）

基于DC/Q32，**不含DF**。现有RouteLocation跨GPU发布join后，warp按静态packet
residue读取route/Location；以ComputeSlot指针match_any，将每个warp batch内
属于同slot的行合并成32B的`{ready指针, slot指针, generation, rows}`。Q超过32时可
保留跨batch重复段，returned仍按完整rows相加，不按段个数计数。packet不跨
整窗，整窗准入保证同包中的不同live blocks不会物理slot别名。

每包复用自身route_begin及rows区间作为段目录容量，实际段紧凑写在区间前部，
TX累加segment.rows到desc.rows为止；没有新增packet prefix/count表，也没有
动态fragment append、expected原子或引用图。发送阶段的ready与退休两轮都
只读段目录，删除逐行expert/Location重解析。两类credit和所有scope不变。

目录只在本rail普通device pool，按其它节点数×source padded route capacity
预留，最坏每route一段；不注册、不跨机传输。增加一次plan末尾grid join发布
段目录，不增加数据面barrier。需同时衡量此规划成本及额外metadata，不能只
报告退休跨度缩短。正在检查/构建，尚无正确性或性能结论。

native SHA256 `92e265dc23578b5b558582bd5db7f761e61e8c1cdd595330762fd97fd6903515`，
REG168/STACK432/LOCAL0，host18/18通过。无并发构建的四规模快筛为
282.656/332.160/3008.380/7068.320 us，64 hashes与DC相同；全部registered
bytes不变，device pool分别增加256/8192/1048576/2621440 bytes/rank。
20K没有改善，故已撤回六个既有源文件改动，并删除候选helper；完整patch和
native保留在 `build/static_v4_ring_return_segments/`，可恢复该实验。
没有继续library、完整Torch或sanitizer。移除两轮逐行解析未转化为整体收益，
不能仅从CY的局部跨度判定它是关键路径，下一步需补多context有效链路吞吐诊断。

### DH：多 context 链路吞吐诊断（已验证，`1729a2f`）

扩展既有`gin_link_bench`，新增`WEFTMOE_GIN_BENCH_CONTEXTS`（默认1），不新增
生产分支或独立测试套件。一个CTA/线程独占一个GIN context，使用THREAD sharing、
weak bundled signal和独立counter，所有context并行发送；cooperative grid在
每个sample开始/结束join，记录包含最慢context完成的GPU时间。多context结果
包括这两次grid同步成本，并非纯NIC线速，也不是完整MoE性能。

immutable TX可共同读取，RX按context分隔，每个context的最大payload都做完整
数据比较；reset各context的独立signal/counter后，沿用已有跨rankallreduce再
开始测量，避免reset覆盖早到的signal。新增32KiB/128KiB，对应主形状Dispatch
Q8/ReturnQ32；其余旧payload保留。汇总按payload×每context puts×contexts
计算有效字节，并将原有误导性的`gbps`字段改为`GBps`，单位明确为GB/s。

新协议标记`completion=weak-thread-grid`，不能将其与旧AZ单context/GPU sharing/
legacy strong信号的差异全归因于context数量；必须用此同一二进制的1/32/64等
对照。仍有NIC本地completion，但没有生产者计算、RX消费credit或混合包流。
当前尚无实测结论，后续据此决定通信优化方向，而不是宣称长序列已接近线速。

DH native SHA256 `cc19c4ea9addd2aa25279b9925ac1c0210c8273cdd229364990f198aed19074b`，
冻结 `build/static_v4_multicontext_link/` 与change.patch；launcher现有13项安全
回归及Python语法检查通过。基准queue depth仍64（生产为1024），不增加调参
分支；本轮先测Full connections、shift0、1/8/32/64 contexts的同版本对照。

首轮全部4×16ranks×16cases通过，每个rank/context的最大payload完整比较成功，
launcher exit0。无并发CPU构建/其它GPU任务；Full/shift0、10 warmup/20 samples、
每context32 puts。表为rank-max样本mean对应的有效GB/s/rail（不是p50延迟比）：

| contexts | 32KiB | 128KiB | 4MiB |
|---:|---:|---:|---:|
|1|7.693|20.338|23.863|
|8|41.449|45.950|47.615|
|32|35.929|36.341|37.815|
|64|31.829|30.689|30.243|

日志 `multicontext_link_full_c{1,8,32,64}.log`，保存每rank样本及hash校验。
生产网络prefill仍REG168/STACK440/SHARED2048/LOCAL0/CONSTANT3264。
8→32→64并发下吞吐下降，提示owner/context数量可能是当前版本的调优方向；
这是诊断假设，不据此直接宣布算子瓶颈或收益。反向32→8复测后，将保持逻辑
L32、Q与容量不变，独立消融更少owner复用固定channel residue；旧减lane实验
同时改变分包/容量，不能替代该实验。

反向32→8复测也全部通过：32contexts的32KiB/128KiB/4MiB为
35.745/36.263/37.717 GB/s，8contexts为42.007/45.910/47.635 GB/s。
日志 `multicontext_link_reverse_full_c{32,8}.log`；两轮无并发构建/其它GPU任务，
全部核心1620MHz，活跃显存3201MHz（远端启动前idle2201MHz不当作负载时钟）。
后续owner配额还会改变空出的Pull/local Return/Reduce warp分工及metadata
使用的context数量，需分别报告端到端和plan时间，不能只归因为NIC/QP数量。

共用reset helper的旧`gin_put_probe`也在EP16通过16个rank的Put/signal/local
completion及payload校验，launcher exit0，日志`multicontext_link_put_probe.log`。
DH只增加诊断能力，不将其作为生产算子性能优化提交。

### DI/DJ：L32 不变，静态 network owners 改为16/8（无收益，已撤回）

基于DC数据面加DH诊断后的代码，不含DE/DF/DG。增加与packet lanes独立的编译期
owner配额，默认仍32；本轮分别构建16和8，actual workers取quota与channel总数
的较小值，GIN contexts仍为三类owner总数（48/24）。每owner继续复用现有固定
channel residue，无动态领用，无新增counter或特殊双机映射。

L32、DispatchQ8、ReturnQ32及双credit、每slot容量不变。原kernel按worker数
分配的Pull/local Return/Reduce warp配额随之变化，metadata可用contexts也
改变，因此端到端改善若存在，不能只归因为NIC。先做冻结native快筛，只有
稳定获益后才更新配置并进行完整数值、wrap与sanitizer门禁。

两候选native分别冻结在 `build/static_v4_ring_owners16/` 和
`build/static_v4_ring_owners8/`，均有源patch；SHA256分别为
`22b67e28a0f68c27cccce8254858fcf6fcc433164f9d02322bcbf73c5a951dd2`、
`33bcd2a73afecda6e5ec9f3a7cf8c53d6fe7d3273d76634b6756f43656d69278`。
两者prefill均REG168/STACK440/SHARED2048/LOCAL0/CONSTANT3264。
1620MHz、无并发构建，10 warmup/2 measured、batch2快筛如下（us）：

|owners|T1|T64|T8192|T20480|
|---:|---:|---:|---:|---:|
|16|359.968|423.568|3130.000|7180.930|
|8|545.744|619.488|5323.070|12486.200|

两套共128个rank输出hash与DC相同，所有device/compute/registered bytes不变。
16-owner的route_exchange rank-max为100.384/109.056/242.624/421.632us，
8-owner为100.704/109.568/243.904/417.664us；DC正式数据为
102.240/116.768/242.912/420.416us。回退并非来自plan大幅变慢，但尚未用
timeline区分channel轮询、TX scalar解析及其它静态角色分工的贡献。
不能据DH推导减少owner必然改善算子：更少线程也要推进相同数量的packet。
已撤回编译期owner配额，恢复固定上限32；未继续完整Torch和sanitizer。
日志 `ring_owners{16,8}_ep16_t*_quick.log`，相邻DC复核另记。

相邻DC快筛T1/20K为279.632/7001.580us，确认不是基线整体漂移导致上述回退。
日志 `ring_owners_adjacent_dc_ep16_t{1,20480}_quick.log`；32-owner上限保持不变。

### DK/DL：保持32 owners，仅消融 GIN queue depth64/256（无收益，已撤回）

DH用queue64而生产用1024，是独立于owner数量的另一差异。只在静态inference
activation devComm创建时选择编译期queue depth，默认1024不变；Notify及V3/
training的queue不变。不改L32/Q8/Q32、slot depth或双credit，不能将NIC队列
深度与软件ring容量混为一谈。先快筛，不预设减少NIC资源一定产生收益。

native冻结 `build/static_v4_ring_gin_qdepth{64,256}/`，含change.patch。
SHA256分别为`6606cded714e99a416873e4482395785786a07b5898be231abde777fb0d4ab64`、
`c4f6b5194359ac582277a0405ec465f2a2ba20493efbf7b3ef2712215134b930`。
同DI/DJ的1620MHz快筛口径、无并发构建：

|queue depth|T1|T64|T8192|T20480|
|---:|---:|---:|---:|---:|
|64|282.256|338.000|3013.660|6987.410|
|256|281.488|334.608|3019.520|6989.280|

128个rank输出hash及所有device/compute/registered bytes与DC一致；此处不包含
NCCL内部QP资源，不能声称降低queue depth后NIC资源用量不变。
进一步64→DC的20K正式对照（10warmup/20measured/batch20）为
7008.940/7006.840us，没有收益。日志
`ring_gin_qdepth{64,256}_ep16_t*_quick.log`、
`ring_gin_qdepth64_{q64,dc}_ep16_t20480_formal.log`。
两个候选及编译期knob均撤回，默认queue1024保持；没有继续完整Torch或sanitizer。
DH的并发context现象成立，但减少生产owner或queue depth都未改善当前算子。

### DM/DN：32 owners 不变，每类角色共享8/16 contexts（小幅长序列收益，暂不采用）

与DI/DJ不同，本候选保留32个静态TX/RX owner以及原Pull/local Return/Reduce
warp分工，只让同类角色的owner按固定余数映射到8或16个GIN contexts，共
24或48个。共享context必须使用NCCL的GPU sharing，不能沿用线程独占模式。
默认quota32仍一owner一context、THREAD sharing，编译期消掉共享映射。

软件packet队列、counter编号、slot容量和remote consumed AND local completion
检查不变；只有GIN提交使用库的共享机制，不增加动态任务领用。metadata阶段
仍一context一CTA，和数据面之间沿用原join。queue depth1024不变。本实验需
同时计入共享锁成本，AZ曾证明在独占context下GPU sharing更慢，因此不预设收益。

两候选native已构建，冻结在 `build/static_v4_ring_shared_contexts{8,16}/`，
各有change.patch；先快筛，未通过完整门禁前不调整默认配置。

native SHA256：8contexts/role为
`0a7eaa67149cbb0ea8d4c0b014fa47b0729471ac45536aa399d601c42dbec67c`，16为
`9db727216ca848ab76f077fae0854c5a2d52609db33acff5560b9a36fc0ac475`。
两者网络prefill均REG168/STACK416/SHARED2048/LOCAL0/CONSTANT3264。
无并发构建、1620MHz，10warmup/2measured/batch2快筛（us）：

|contexts/role|T1|T64|T8192|T20480|
|---:|---:|---:|---:|---:|
|8|283.008|340.592|3029.520|6983.740|
|16|282.736|354.112|3020.780|6969.060|

首轮16→DC的20K正式10/20/20为6958.730/7049.640us（约1.3%）；正在做
反向对照及8K正式复核，尚不能认定稳定收益。短序列有代价，不能只选20K报告。
日志 `ring_shared_contexts{8,16}_ep16_t*_quick.log`、
`ring_shared_contexts16_{c16,dc}_ep16_t20480_formal.log`。

反向DC→16复核：8K为3007.210→3004.440us（基本持平），20K为
7009.880→6952.720us（改善0.815%）。20K两轮16-context内部样本范围约
6945–6970us，而基线跨进程的中位数漂移约40us，故只据反向也成立的小幅收益
继续门禁，不报告为确定的1.3%普适收益。全部quick的128个rank输出hash、
device/compute/registered bytes与DC一致。

短序列DC→16正式对照：T1为275.779→279.648us，T64为333.810→337.800us，
分别回退1.40%和1.20%。日志
`ring_shared_contexts16_short_{dc,c16}_ep16_t{1,64}_formal.log`。
20K小幅收益不足以解决相对V3优势偏小的问题，且需要维护共享映射及同步策略；
本轮不采用，不继续完整Torch/sanitizer，源patch和native保留供后续独立复核。
已撤回实验knob与共享代码，默认仍96个线程独占contexts。并非断言共享context
没有优化价值，只是目前端到端收益不足以优先保留这项复杂性。

### DO：计算 slot 准入与 Return TX 准入分离（无收益，已撤回）

独立基于DC，不含DM/DN。原`admit`在整窗X/S/C已空之后仍要等待所有Return TX
预约，因而延迟Pull/FC1；实际上只有FC2会写TX。保留计算slot三类旧代完成检查，
提前发布原`generation/admitted_windows`，由同一固定admission warp独立检查
Return TX并发布新`output_reserved_windows`。只有网络FC2增加该head的acquire；
单机任务选择不增加检查。每次plan在既有grid join之前重置新head。

这是准入条件拆分，不是提前复用TX。TX owner仍逐slot等待remote consumed AND
local NIC completion；TX head单调FIFO，已预约packet在本窗FC2完成前不能发送，
也不能再次预约。X/S/C仍须上一代FC1、FC2及Return全部退休后才能改目标计数。
后窗Pull/FC1不占用前窗Return/RX owner，source contribution准入独立推进；
没有新增动态领用或阻塞式跨owner调用。计算cluster仍按原window/residue顺序，
不能把本改动描述为任意深度跨窗计算lookahead。

新增一个uint64窗口head，不新增payload或packet目录；需实测确认资源、收益，
收益成立后才跑完整数值、tiny wrap和sanitizer。host现有18/18通过。

冻结 `build/static_v4_ring_split_admission/`，含change.patch，native SHA256
`2e328d184347a6e8e98b0929c6198b935daa79a537af0f9b4ec6e2c56f9479bf`。
REG168/STACK440/SHARED2048/LOCAL0/CONSTANT3264，与DC相同。1620MHz、无并行
CPU构建，10 warmup/2measured/batch2：B512的T1/T64/T20K为
277.728/339.744/7012.880us；四轮8K为3015.970/2995.070/3005.630/2996.510us。
初次四规模命令误用未被fixture读取的token变量，实际均为默认8192，已经按
JSON真实shape重命名为`t8192_probe{0,1,2,3}`；不作为四规模结果。

由于准入变更可能改变大窗口的取舍，进一步复测20K，仍D8：

|B|DO/us|相邻DC/us|device bytes/rank（两者相同）|
|---:|---:|---:|---:|
|1024|7442.860|7366.670|1148575680|
|1536|7423.250|7288.960|1385550272|

共176个rank输出hash与DC对应shape一致，B512全部pool/registered bytes不变。
没有足够收益，不继续完整Torch/sanitizer；已撤回新head与准入拆分，保留冻结
候选供复核。日志 `ring_split_admission_*`。该实验只说明这项拆分未改善当前
实现，不证明Return等待与所有计算完全折叠，也不排除其它静态计算流水改进。

### DP：相邻两 source-group 窗口合为一个静态计算 wave（无收益，已撤回）

独立基于DC，不含DO准入拆分。只改变网络ring的计算wave范围：同一个source
token window内最多合并相邻两个source groups，奇数组的最后一组独立处理。
每cluster先做两个窗口中归自己的FC1，再做FC2；全局task residue和原有slot
ready条件不变，Pull/Return/Reduce和packet目录不改。两CTA选择顺序依然完全
确定，不引入根据ready抢选stage的分支或额外DSM任务发布。单机保持原wave。

计算ring已分配两窗最坏容量，因此扩大wave不增加buffer。不能合并跨token
window的两个group，否则D1时下一代source reservation可能等待当前代FC2，
形成循环等待。本候选严格在epoch边界截断；两组包含不同source，同一个epoch
不会争用同一source的Return TX packet。更大wave可能改善GEMM利用率，也可能
延迟本来已ready的FC2；先实测，不预设正收益。冻结候选目录
`build/static_v4_ring_paired_waves/`。

DP native SHA256 `c39c2e31838a02fc6d9992f45f28156c1b1f925620149754d4a60f07a0855882`，
REG168/STACK440/SHARED2048/LOCAL0/CONSTANT3264，patch已归档。相同1620MHz快筛
口径四规模为280.112/338.656/3013.970/6996.060us；64 hashes和全部pool bytes
与DC一致。短序列没有改善，长序列也没有明显收益；再做20K正式相邻对照。
日志 `ring_paired_waves_ep16_t*_quick.log`。

20K正式10/20/20相邻DP→DC为7009.270/6996.910us；DP样本6988.120–7045.590，
DC为6985.950–7028.110us，未见收益。日志
`ring_paired_waves_{dp,dc}_ep16_t20480_formal.log`。已撤回计算wave合并；没有
继续library、完整Torch和sanitizer，不把快筛中的6996.060us认定为优化收益。

### DQ/DR：保留32路准备，分离8/16个独占发包owner（无收益，已撤回）

DC上将Dispatch pack/Return ready与退休检查保留为32路准备角色；另设每方向
8或16个固定GIN owner，循环处理固定channel余数，不做动态领取或context锁竞争。
准备角色只在取得`tx_reserved_head`后访问TX；复制/FC2完成后release发布
`tx_prepared_head`，GIN owner acquire后Put。预留仍检查远端credit和本地NIC
completion。新head填入NetworkChannel的原padding，结构仍64B，不增加payload；
借用第三warp的部分Reduce角色，32路RX import及scatter保持不变。

DQ为8个发包owner/方向，共48 contexts；DR为16个/方向，共64 contexts。
准备warp数不减，区别于DI/DJ减少全部owner，以及DM/DN共享context。这是检查
独立链路少context收益能否覆盖新增交接/轮询成本，不预设收益。

1620MHz、B512/D8、10 warmup/2 samples/batch2，测量无并发CPU构建或GPU任务：

| T/rank | DQ：8 owners/us | DR：16 owners/us | 当前DC正式参考/us |
|---:|---:|---:|---:|
|1|412.880|317.040|275.669|
|64|471.920|363.040|333.766|
|8192|3494.260|3057.440|3006.200|
|20480|8044.240|7015.340|7011.600|

128个输出hash及全部pool bytes与DC一致。DQ host18/18通过；两构建原始日志
`ring_posting{8,16}_ep16_t*_quick.log`。DQ/DR native SHA256分别
`f73437fa3548c467ee66272686036ebd68bcf1713cb983e1bb76e74f542d2c95`、
`427a084bade3a6f3a473af6cc0768c1235c32f80f335133df2524d79271a4565`，
冻结于`static_v4_ring_posting{8,16}/`并保存独立patch。DQ多机ring kernel资源
REG168/STACK424/SHARED2048/LOCAL0/CONSTANT3264；栈减少也没有转化为加速。
DR长序列只接近DC、短序列退化，未继续正式计时、library、Torch或sanitizer。
全部拆分代码已撤回；后续实验不含prepared head/额外发包角色。

### DS：Return 直写 source contribution 窗口（保留，性能目标未完成）

回到DC数据面，不含DQ/DR。将多机source contribution按每个source window内
既有expert-major route_order存放，搬到本rail注册window。source新增每route
4-byte窗口内行号，Reduce仍按原`combine_order`做相同FP32加法/BF16舍入。
本机FC2预绑定该注册区的peer地址；远端Return packet恰是expert-major连续
区间，GIN直接写入同一contribution窗口，不再先写RX payload再逐行scatter。
接收角色只在GIN导入可见性后发布原有逐route ready，随后释放packet控制slot。

这里有两种不同的复用边界，不能合并：

- Return TX仍由远端packet credit及本地NIC completion双检查保护。
- contribution payload由source窗口的全部token Reduce credit保护。
  下一代窗口只有在旧代所有token完成读取后才能准入。远端FC2须先收到该窗口X，
  而Dispatch发X之前必须看到source contribution reservation；本节点Pull同样
  先取得source reservation。因此下一代本地FC2/跨机Return都不能提前覆盖旧代。

这不同于CM/AQ让Reduce持续占有network RX slot：Reduce不访问可回收的packet
控制状态，packet在发布route ready后即可退休，payload一直保留在独立的source
窗口中。没有按token最坏接收R分配激活，也没有全节点数据面barrier或丢token。
行号逆映射在已有source-plan grid join之后生成，不新增规划barrier。

初版候选删除了原compute arena的contribution plane，但暂保留未使用的Return
RX payload预留以分离地址迁移验证；若有收益，验收前应删除该冗余预留。不能把
当前候选显存记成已经省掉RX，不能将实现本身视为收益。冻结目录
`build/static_v4_ring_return_contribution/`，编译与数值快筛进行中。

DS初版native SHA256 `82e8b7d12f270d0722a88a14b187301b21537a22def05d729da250981f7e66da`，
REG168/SHARED2048/LOCAL0/CONSTANT3264，多机prefill/decode栈为432/440bytes。
host18/18通过，无并发构建/其它GPU工作、1620MHz的T64/20K快筛分别
333.952/6944.340us；32 hashes与DC一致。只是初步结果，未做相邻正式对照。
20K pool为914,300,352、compute753,551,032、registered160,749,312bytes/rank，
相对DC净增加327,680bytes的行号表；保留的RX预留尚未删除。T64初版还错误地
按配置B512而非有效B64申请contribution，虽不影响结果但浪费容量，下一构建修正。

DS整理版删除Return RX payload预留，NetworkRings只保留Dispatch TX/RX和
Return TX三种地址；用具名dispatch_payload/return_payload消除不存在的Return RX
地址组合，同步更新已有布局范围/不重叠检查，没有新建测试套件。contribution
分配改用与compute ring一致的有效B。整理版冻结目录
`build/static_v4_ring_return_contribution_trimmed/`，待数值/复用/性能门禁。

整理版native SHA256 `523d35b301e8edd46bc8802074d5f1c36ba3daef14ba0da2f31c5b59d61e83f5`，
host18/18，REG168/SHARED2048/LOCAL0/CONSTANT3264，多机prefill/decode栈432/440。
1620MHz、B512/D8、无并发构建/GPU任务的10warmup/2samples/batch2快筛如下，
64个输出hash与DC一致；这是快筛，不作为正式性能结论。

| T/rank | latency/us | total bytes/rank | compute bytes/rank | registered bytes/rank |
|---:|---:|---:|---:|---:|
|1|277.632|62,176,192|11,661,496|50,514,688|
|64|335.424|91,914,432|33,115,064|58,799,360|
|8192|2953.970|529,801,408|408,432,568|121,368,832|
|20480|6948.720|880,745,920|753,551,032|127,194,880|

20K总pool相对DC的913,972,672减少33,226,752bytes/rank：删除33,554,432bytes
Return RX，增加327,680bytes逆行号表。registered变大来自contribution迁移，
不代表总pool增加；短序列也按有效B分配。日志`ring_return_contribution_trimmed_ep16_t*_quick.log`。

20K正式相邻DS→DC（1620MHz、10warmup/20samples/batch20、无并发构建/GPU工作）
为6986.620/7008.140us，p50降低0.307%；样本范围分别6967.520–7034.950和
6989.860–7030.720us，16hashes一致。收益很小，不能称为显著加速；候选的主要
价值是消除scatter及减少有界缓冲区。日志
`ring_return_contribution_trimmed_{ds,dc}_ep16_t20480_formal.log`。

整理版rank1/9 eager B16/D1/K8/T64、11rounds的memcheck与racecheck均launcher
exit0：两份memcheck各0errors，两份racecheck各0hazards。memcheck local160013/
remote224506，racecheck local161229/remote224786；只覆盖这两个受检查rank，
不声称所有rank完成sanitizer。完整Torch及其余正式计时尚待完成。

完整Torch现已完成：EP8的T1/64/512/2048/8192/20480+edges共288项、EP16共576项，
另EP16 B16/D1/K8的T1/64+edges共304项，全部launcher exit0。包括不等长/空source、
热点重复expert、长热点后短调用、共享workspace的graph链及输入/route变化。
library SHA256 `1d5f5905975cb0a45928f500321419cde5b19e4cf08ef4c6932035033fd551ec`。
日志`ring_return_contribution_trimmed_{tiny_,}torch_ep*`；未新增数值测试套件。

DS最终正式四规模（1620MHz、10warmup/20samples/batch20、无并发编译/GPU任务）：

| T/rank | 单机EP8/us | 双机EP16/us | 相对DC正式参考：EP8 | 相对DC正式参考：EP16 |
|---:|---:|---:|---:|---:|
|1|100.547|278.366|基本不变|+0.98%|
|64|135.494|333.082|基本不变|−0.20%|
|8192|1257.680|2955.570|基本不变|−1.69%|
|20480|2954.830|6986.620|+0.17%|−0.36%|

表中DC参考是此前DC正式矩阵，不是四规模逐一相邻重测；20K另有上述真正相邻
对照（−0.307%）。单机在第二节点独立8卡上测量，共享Reduce helper没有观察到
明显退化；全部32个hash与此前DC一致。相对同日冻结V3正式参考，双机20K仍只
快约1.55%，远未达到“明显优于”的目标，不能以省内存替代性能验收。
DS按完整数值/复用/内存门禁保留，主要记内存和复制简化收益；不恢复已失败的
直接RX Reduce、context共享或发包owner拆分。最终仅做缩进/换行和注释整理，
冻结二进制对应的功能patch已留档。日志`ring_return_contribution_trimmed_*_formal.log`。

### DT：DS直写Return后重测Dispatch Q16（无收益，不采用）

仅将Dispatch Q8编译为Q16，Return Qmax32/L32/B512/D8和DS协议不变。
这是新数据面下的独立复核，不复用早期仍有C→TX/RX→contribution复制版本的结论。
冻结`build/static_v4_ring_direct_return_dispatch16/test_cuda_notify`，SHA256
`8fc7bb796da47648cd0223ebba9a549d7554f6ee84bfb45efa3d5bd79d25ac4e`；源代码
为fd23867，仅构建参数`WEFTMOE_FWD_STATIC_V4_PACKET_ROWS=16`不同。

1620MHz、10warmup/2samples/batch2、无并发构建/GPU任务，T1/64/8192/20480
快筛277.712/344.736/3155.860/7667.140us，64hashes与DS一致。四规模pool分别
78,953,408/108,686,528/545,865,920/895,741,376bytes/rank。较大Dispatch包
没有提高当前端到端吞吐，且增加有界payload；长序列明显回退，不继续正式计时
或完整数值门禁，不改变当前Q8。日志`ring_direct_return_dispatch16_ep16_t*_quick.log`。

### DU：DS直写contribution后Return Q64（无稳定收益，不采用）

仅将DS的Return Qmax32编译为64，Dispatch Q8/L32/B512/D8不变；与DE区别是
本版本已无Return RX payload和scatter。训练host准备模块同时在源码中，但native
inference benchmark不创建/调用它，不改变本实验GPU数据面。

冻结`build/static_v4_ring_ds_return_q64_du/`，native SHA256
`8c803f715409d879e19dd0d0728cdda59d7962dba2fcb76ad4afac7fd674d988`。
1620MHz、10warmup/2samples/batch2、无并发构建/GPU任务，四规模如下：

| T/rank | p50/us | measured min–max/us | total bytes/rank |
|---:|---:|---:|---:|
|1|279.040|279.040–289.840|95,730,624|
|64|338.336|338.336–586.416|125,468,864|
|8192|3015.620|3015.620–3018.990|563,331,264|
|20480|6966.300|6966.300–7033.020|914,238,912|

全部64hashes与DS一致。T64有明显离群，20K两样本相差67us，不能以较小样本
的约20us优势判定收益；8K回退约2%，20K注册区增加33,492,992bytes/rank。
不继续正式门禁，不改变Q32。日志`ring_ds_return_q64_du_ep16_t*_quick.log`。
本轮发现并纠正了上述DE目录重名，不将已被覆盖的历史日志声称为仍可复核。

### DV：DS Return ready 由整 warp 并行发布（无明确收益，已撤回）

基于`652b9f5`，inference数据面仍为DS。lane0 acquire packet的visible generation
后，通过warp join将可见性传给其余lane；按lane residue发布各route ready，再
warp join，由lane0提交全部rows的consumed credit并推进唯一cursor。没有改变
Put、双credit、source contribution窗口的Reduce回收或求和顺序，内存容量不变。
这是DS RX端的纯ready发布，不是CX的TX端ready/retirement元数据协作读取。

冻结`build/static_v4_ring_ds_return_ready_warp_dv/`，native SHA256
`f01def1c3ed4166e2c23384668f5bdaee59b432db1b55219c04ef3092355c1f5`，保存patch及
provenance。network prefill/decode仍REG168、STACK432/440、SHARED2048、LOCAL0、
CONSTANT3264。1620MHz、Q8/Q32/L32/B512/D8，无并发CPU构建或其它GPU任务：

| T/rank | quick p50/us | quick min–max/us |
|---:|---:|---:|
|1|277.824|277.824–281.552|
|64|332.816|332.816–336.848|
|8192|2971.420|2971.420–2989.760|
|20480|6954.140|6954.140–6966.180|

快筛均为10warmup/2samples/batch2，64个hash与DS一致。随后20K相邻DS→DV正式
对照使用10warmup/20samples/batch20：DS p50 **7009.690us**（6982.130–7042.930），
DV **6982.520us**（6956.530–7023.530），仅降低0.39%，区间明显重叠。8K快筛也
不优于DS历史正式2955.570us，不能宣称局部并行化带来明确端到端改善。本轮没有
反序正式复测，也未继续library/Torch/sanitizer门禁；选择保留简单的原lane0发布，
所有CUDA改动撤回。日志`ring_ds_return_ready_warp_dv_ep16_t*_quick.log`和
`ring_ds_return_ready_warp_dv_{ds,dv}_ep16_t20480_formal.log`。

### DW：网络 slot 隔离 NIC signal 与 GPU 状态（无收益，已撤回）

基于`e051282`的DS数据面，不含DV。仅将NetworkSlot中的NIC ready/credit
signal与GPU visible/consumed/target状态分到不同128-byte区间，slot由64B变
256B；所有访问仍用同一布局及offsetof。没有改packet、scope、同步或计数。
EP16增加98,304bytes/rank控制区，检验原布局的共享缓存行是否影响性能。

冻结`build/static_v4_ring_split_slot_state_dw/`，native SHA256
`8c2e4ab7ae51c28396054149258678814939ffe1af3afab48e2b1537dbadd54e`及原始patch。
host18/18通过；network prefill/decode REG168、STACK440/448、SHARED2048、
LOCAL0、CONSTANT3264（相较DS，STACK均增加8B）。1620MHz、Q8/Q32/L32/B512/D8，
无并发构建/GPU任务，10warmup/2samples/batch2：

| T/rank | p50/us | min–max/us | total bytes/rank |
|---:|---:|---:|---:|
|8192|2961.070|2961.070–2965.780|529,899,712|
|20480|7019.920|7019.920–7041.060|880,844,224|

32个hash与DS一致。长序列未显示改善，不继续短规模、正式计时及完整数值/
sanitizer门禁，slot恢复64B。此结果不能证明硬件上不存在缓存行竞争，只说明
这项隔离没有得到可用的端到端收益。日志`ring_split_slot_state_dw_ep16_t*_quick.log`。

### DX：连续 token 的 Dispatch 直接读取 source X（退化，已撤回）

基于`487209b`的DS。对整个source window都命中的节点，以token次序直接写
NodeToken表，省去该节点的压缩计数atomic；稀疏节点仍用原压缩路径。packet plan
逐项确认token连续后记录起始token，否则保留原gather。将已有source X从compute
区迁到GIN注册区，让连续packet直接从X发Put，省去X→Dispatch TX复制；不额外
分配完整X。原有逐slot RX释放、本地NIC完成双credit及调用终止drain均保留。
未修改route_order、Return布局或求和顺序；不把此枚举称为StableTape。

冻结`build/static_v4_ring_direct_input_dx/`（32 owners/role、96 contexts）及
`build/static_v4_ring_direct_input_dx8/`（8 owners/role、24 contexts），各存patch
及provenance。native SHA256分别为
`0a2537b7015eb52b72b6afc18e37807770c9d6d9aee80eb75c2808a681f35e75`、
`ffb26d632cc4f0de9718762eb00b4e7fdf36ba204e80c7397d15b4ed395e8b6d`。
两者network prefill/decode均REG168、STACK432、SHARED2048、LOCAL0、CONSTANT3264。
DX32 host18/18、现有GPU plan 37个source/packet及6个geometry case通过，plan
memcheck/racecheck均0；不等同于端到端通信sanitizer或完整数值门禁。

1620MHz、EP16、E64/K4/H2048/I512、T20480、Q8/Q32/L32/B512/D8，10warmup/
2samples/batch2，无并发CPU构建/GPU任务：

| 版本 | p50/us | min–max/us | total bytes/rank |
|---|---:|---:|---:|
| DX32 | 7344.900 | 7344.900–7369.650 | 880,786,880 |
| DX8 | 12481.000 | 12481.000–12521.700 | 880,786,880 |

各16个hash均与DS一致。compute669,664,952、registered211,121,920bytes/rank；
总量仅因packet descriptor多40,960B，但注册区增加83,927,040B。主benchmark的
确定性路由使每个token都命中两节点，适于此连续快路径，不能推广为任意稀疏
路由的命中率。即便如此，DX32较DS约7.0ms的参考明显更慢；DX8还同时改变
Pull/local Return/Reduce的owner分配，不能将退化全部归因于NIC并发。

不再继续四规模/正式计时/library/Torch/端到端sanitizer门禁，全部源码改动撤回。
这里只否定这个组合，不单独归因于source地址、排序或复制消除；将来若重试须
独立消融。日志`ring_direct_input_dx{,8}_ep16_t20480_quick.log`。

### DY：仅减少 Dispatch 静态 owner（无收益，已撤回）

基于M6d `4c22ddc`，inference仍是DS数据面。Return发送、RX进口及Return发布
保持32 owners；只将Dispatch由32降至8/16，总独占contexts由96降至72/80。
空出的warp0参加Pull，其owner从100升至124/116；其他warp分工不变。因此也
不是纯NIC context数量消融。固定pool生命周期内的channel/context归属，保留
逐slot远端credit和本地NIC完成检查，无动态任务领用。Q8/Q32/L32/B512/D8不变。

冻结`build/static_v4_ring_asymmetric_dispatch_dy{,16}/`，各有patch/provenance；
native SHA256分别为
`40616cf7e6b4374177396ebfb4ca4a8daaf856fece832c1c73b156435a8065db`、
`c79f0bb05738449ad9b9ffc1aefdf2823bc1ae1de5ee8fa37f7eccbd65012b34`。
两者network prefill/decode均REG168、STACK432/440、SHARED2048、LOCAL0、
CONSTANT3264，total/compute/registered pool bytes与DS不变。

1620MHz、EP16 T20480/E64/K4/H2048/I512，10warmup/2samples/batch2；无并发
构建或其他GPU任务：

| Dispatch owners | contexts | p50/us | min–max/us | route exchange rank-max/us |
|---:|---:|---:|---:|---:|
| 8 | 72 | 8605.680 | 8605.680–8621.600 | 422.688 |
| 16 | 80 | 7300.270 | 7300.270–7532.820 | 422.400 |

32个输出hash与冻结DS/M6d final一致；其相邻正式20K参考为6985.580us。
两候选都没有收益，全部源码改动撤回，不继续四规模、正式计时或完整数值/
sanitizer门禁，也不把hash相同称为完整正确性证明。纯链路少context优势不能
直接推广到带copy/readiness/credit的混合算子。日志
`ring_asymmetric_dispatch_dy{,16}_ep16_t20480_quick.log`。

### DZ/EA：Return依赖预计算及独立发送配额（无收益，已撤回）

基于M6e `a9364f6`，inference数据面仍为DS。重测DG类假设在Return已直写
source contribution后的收益：目的GPU发布RouteLocation并完成node join后，
每warp规划一个Return包，将连续同C slot的行压成32B描述项
`{ready地址, returned地址, generation, rows}`，运行时两轮仅遍历描述项。
按remote source node分配目录，每包一个count，最坏Q项；Q>32按warp batch
处理，跨batch可重复同slot，rows仍精确相加。新增一次plan末尾grid join；
不增加数据面barrier，不改变双credit、payload、Reduce顺序或expected定义。
与DG相比是当前DS数据面、独立packet count/stride目录及连续分段实现，不是
此前从未测试过的优化方向。

DZ冻结`build/static_v4_return_dependencies_dz/`，native SHA256
`3cc1d570d07fa678137e96d390dd4b574ad419598b8720e4be3f8ab41e34616c`。
EA在DZ基础上只把Return发送owner改为8，Dispatch/RX仍32，contexts96→72；
Return本地发布warp随之增加24，未增加独立prepared队列或动态领取。
EA冻结`build/static_v4_return_dependencies_ea8/`，native SHA256
`ded0efce21f248b0f809273fa4e63152494729b36ccf40ee3962bea9c0f76779`。
两者保存完整source patch及资源报告，network inference均为
REG168/STACK416/SHARED2048/LOCAL0/CONSTANT3264。

两机1620MHz，EP16 T20480，B512/D8，计时无并发CPU构建或其它GPU工作：

|候选|采样（warmup/samples/batch）|p50/us|min–max/us|plan rank-max/us|
|---|---|---:|---:|---:|
|DZ|10/2/2|7016.590|7016.590–7096.020|430.368|
|DZ|10/20/20|7150.980|7112.710–7195.290|433.056|
|相邻M6e基线|10/20/20|6991.460|6961.910–7034.520|421.952|
|EA8|10/2/2|7134.980|7134.980–7154.660|432.768|

四份结果各16个输出hash均与DS/M6e一致，不等于完整数值/sanitizer通过。
DZ正式比相邻基线慢2.28%；候选pool从880745920增加到883377600bytes/rank，
其中新增2631680bytes全为compute metadata，注册区仍127194880bytes/rank。
EA没有逆转退化，不继续16-owner组合或四规模完整门禁。全部八个既有文件
改动及新helper已撤回，可由冻结patch恢复；没有把失败候选留成生产分支。
日志`return_dependencies_{dz,ea8}_ep16_t20480_*`及
`return_dependencies_dz_baseline_ep16_t20480_formal.log`。

### EB：B1024下Return Q128（局部改善，但不如当前配置，不采用）

DS/M6e数据面不变，仅编译Return Qmax128，L32/Dispatch Q8不变。注意
`return_partition`在B512/K4时会将其裁到Q64，不能称为实际Q128实验；本轮
选B1024/D4，在同窗口下成对测冻结M6e Q32与Q128，并对照已选B512/D8/Q32。
没有修改自适应分包策略。冻结`build/static_v4_ds_return_q128_eb/`，native
SHA256 `9be39627bfc8c40862300a5eae71d7eed3234db38ad6b23b14e06ccab8b480f2`，
源码`c9d0040`，prefill/decode REG168/STACK432/440/SHARED2048/LOCAL0。

1620MHz、EP16 T20480、10warmup/2samples/batch2、无并发CPU构建或GPU任务：

|B/D/Qmax|实际Q|p50/us|min–max/us|pool bytes/rank|
|---|---:|---:|---:|---:|
|1024/4/32|32|7820.260|7820.260–7921.440|1048108736|
|1024/4/128|128|7363.330|7363.330–7628.530|1148679872|

32个hash均与M6e一致。较大包在B1024下有局部收益，但两者仍明显慢于相邻
B512/D8/Q32正式6991.460us，且增大有界buffer；不采用，也不将2样本局部
差异当成正式性能提升。没有继续四规模、library或完整数值/sanitizer门禁。
日志`ds_return_q{32,128}_eb_b1024_ep16_t20480_quick.log`。

### EC–EE：B512实际Q128、紧凑依赖与warp发布（未证明收益，撤回）

基于M6f `35d7201`，保持Dispatch Q8/L32/B512/D8/32 owners。EC将长source
（实际tokens≥2048）的Return分包改为直接采用Qmax128，短source仍用原自适应
规则；因此这次B512下实际为Q128，不是EB中被裁成Q64的配置。各peer仍从同一
source epoch计算分包，不按接收端自己的token数决定。没有改变双credit。

ED在EC上复用DG的紧凑依赖思想：plan末尾按Return原始route区间存放32B
`{ready地址, C slot地址, generation, rows}`，warp内按slot分组；运行时累计rows
到packet.rows确定结束，不新增packet count目录。跨warp batch可重复同slot，
但逻辑行数仍精确相加。view附加于ForwardKernelArgs尾部，不改变既有字段偏移；
增加一次plan末尾grid join，数据面无新barrier。EE再将Return RX的ready发布
分给整warp，lane0 acquire后warp join，全部lane发布后再join并退休控制slot。

1620MHz、EP16 T20480、10warmup/2samples/batch2，无并发构建或GPU任务：

|候选|p50/us|min–max/us|pool bytes/rank|
|---|---:|---:|---:|
|EC：实际Q128|9657.420|9657.420–9889.500|981347776|
|ED：增加紧凑依赖|6942.350|6942.350–6945.140|983969216|
|EE：再增加warp ready发布|6952.530|6952.530–6965.200|983969216|

三轮各16个输出hash均与M6f一致。ED/EE恢复了EC的严重退化，但相对M6f正式
6991.990us的不足1%快筛差异，不能认定稳定收益。相对M6f pool 880745920，
候选注册区增加100601856 bytes/rank，ED/EE另增加2621440 bytes普通metadata。
为这样的小幅且未证实差异保留额外协议目录/内存不值得，全部源码撤回。
EC host 18/18、ED GPU plan 37个source/packet及6个destination case通过；
没有继续四规模正式计时、完整数值或sanitizer，不把hash一致称为完整验收。

冻结目录分别为`build/static_v4_prefill_return_q128_ec/`、
`build/static_v4_prefill_return_segments_ed/`、`build/static_v4_prefill_return_warp_ee/`，
各含完整patch及资源报告。native SHA256依次为：

- `d682ebba908573978f66c0eecaa6a3b266c6fa006e19276c63ad3e7a9e18c5b8`
- `1ab1d366195ef53830367dddfeb2a06122bdad7ccef37926a76e6892ec514279`
- `bdd878fd77085cd094610c5c5fffe194f9e35bb72c4a659456d05e8972d8219f`

日志`prefill_return_{q128_ec,segments_ed,warp_ee}_ep16_t20480_quick.log`；
ED planner日志`prefill_return_segments_ed_plan.log`。

### EF：本地/远端Pull独立静态owner集合（退化，撤回）

回到M6f/DS，不含EC–EE。将100个Pull warp固定分成50个本地、50个远端owner，
各自重编号residue并独立推进窗口；本地不等待远端packet，远端不扫描本地行。
仍沿用whole-window admission、逐行身份过滤（兼容group跨实际节点的容量配置）
和原有copy/credit边界。不增buffer、共享计数、动态claim或额外通信barrier。
与BQ的两窗顺序交换不同，本项改变两类owner的分工与并行推进。

1620MHz、EP16/B512/D8、10warmup/2samples/batch2、无并发构建或GPU任务：
8K p50 **3330.880us**（3330.880–3343.730），20K **7825.730us**
（7825.730–9080.910）。32个hash与M6f一致、pool不变，均不如当前正式
2953.070/6991.990us。该划分削减远端Pull并行度，没有提供端到端收益；
不能仅凭这轮实验排除固定owner内部packet等待顺序的其它优化。
已撤回唯一源码改动，未继续正式计时、完整数值或sanitizer。
冻结`build/static_v4_pull_partition_ef/`（含patch/resource report），native
SHA256 `31b84f195fc0a5cc2c7e60699ce0dfaa0d99accbca16301dd99a380530e14dca`；
日志`pull_partition_ef_ep16_t{8192,20480}_quick.log`，以JSON shape为准。

### EG：Pull owner私有ready-packet lookahead（无稳定收益，撤回）

独立回到M6f/DS，保留全部100个Pull warp及其residue。每owner将自己固定负责的
packet分成最多32项的批次，用寄存器位图记录尚未完成项；轮询跳过未ready包，
已复制的包不再访问。空项不读RX，lane0的system acquire经warp同步后才读payload，
仍在完整复制、X-ready发布后按desc.rows退休。窗口间顺序不变，immutable plan不改，
没有共享claim、动态expected、跨窗持有RX或新增buffer。现有逐slot ACK允许乱序退休。

1620MHz、EP16/B512/D8、10warmup/2samples/batch2，无并发构建或其它GPU任务：
8K p50 **2966.860us**（2966.860–2995.220），20K **6961.680us**
（6961.680–7088.080）。32 hashes及pool bytes与M6f相同；8K未改善，20K
波动跨过基线，不能认定稳定收益。网络prefill/decode REG168、STACK448/456、
SHARED2048、LOCAL0，相比基线各增加16B stack。撤回唯一源码改动，不继续完整
数值/sanitizer或正式计时。冻结`build/static_v4_pull_lookahead_eg/`与patch/resources，
native SHA256 `51623c0c4eeeed78274a9b1b0a6755ee79331ecaa92055dc3455cd0ed3bac142`；
日志`pull_lookahead_eg_ep16_t{8192,20480}_quick.log`。

### EH：仅缩减RX/ACK owner至8（显著退化，撤回）

M6f/DS上保持Dispatch/Return各32个发送owner、100个Pull warp及32个Return
ready发布owner；仅RX import/ACK从32减至8个独占owner，每人处理固定channel
余数，释放的24个warp转给Reduce。contexts96→72，metadata阶段随之采用72个
可用context。逐slot ready/consumed/ACK、本地NIC completion及所有payload
容量不变；未引入共享context锁，也不同于DI/DJ同时减少所有角色的实验。

1620MHz、EP16/B512/D8/T20480、10warmup/2samples/batch2，无并发构建或GPU
任务，p50 **11608.500us**（11608.500–11615.000），plan rank-max428.896us。
16 hashes和全部pool bytes与M6f一致，但端到端严重退化；该角色配额不采用，
未继续四规模/正式计时/完整数值/sanitizer。全部四文件改动已撤回。
网络prefill/decode REG168/STACK440/448/SHARED2048/LOCAL0。
冻结`build/static_v4_receive_owners8_eh/`，native SHA256
`e176a0dd746efb105760d0134f3b8e2cb09ae26f6b8d95aee476c53cfc628210`；
日志`receive_owners8_eh_ep16_t20480_quick.log`。

### EI：warp并行检查ACK消费状态（未获益，撤回）

M6f/DS上保持全部32路TX/RX owner和96 contexts。RX import仍由lane0执行，
warp同步后各lane读取一个物理slot的generation/consumed/target，以ballot得到
可退休集合；再经warp join传递acquire链，lane0逐项发送原GIN credit。GIN context
仍线程独占，expected/packet顺序及逐slot双credit不变，深度超过32分批处理。
没有修改payload、buffer容量或其他通信角色。

1620MHz、EP16/B512/D8/T20480、10warmup/2samples/batch2，无并发构建/GPU任务：
p50 **7053.760us**（7053.760–7093.550），plan rank-max424.160us，16 hashes和
pool bytes与M6f一致，没有改善当前6991.990us正式参考。网络prefill/decode
REG168/STACK448/464/SHARED2048/LOCAL0，额外warp协作未提供净收益。
两文件改动已撤回，不继续完整数值/sanitizer或正式矩阵。冻结
`build/static_v4_ack_warp_ei/`，native SHA256
`b1688739bb5d982ed62b8020249ce734e454ec58c0621d1736537393b489fa3d`；
日志`ack_warp_ei_ep16_t20480_quick.log`。

### EJ：计算owner的两窗口ready lookahead（无收益，撤回）

保持FC1/FC2全局residue及静态ownership，只在网络prefill上更改owner内部顺序：
优先探测当前窗口，自己的FC1未ready时允许执行已经ready的自有FC2；当前窗口
无可执行任务才探测下一窗口。两个私有progress保存各自FC1/FC2进度，旧窗口
耗尽后滑动；不抢他人任务，不增加全局claim/CAS，也不等待未准入的未来窗口。
不同于DP合并wave后固定先做两窗FC1，这里未来窗口只是非阻塞探测。

原静态路径的两个CTA独立选择相同固定序列，不能各自按ready跳过任务，否则
会破坏multicast配对。EJ由cluster leader统一决策，复用已有DSM ready/joined
barrier发布给另一CTA。新增64-bit进度使ComputePairState由48变64 bytes；
network prefill main为REG168/STACK440/SHARED2048/LOCAL0，相比基线stack增加8。
buffer容量、packet分区、全部ready/credit及数值顺序不变。

基于M6f、带独立M6g Backward配置的冻结native
`build/static_v4_compute_lookahead_ej/test_cuda_notify`，SHA256
`bedc5ef20399b8e1a4b874901b1a719cebdc249f7058a34f642b7a9de558b6d8`。
1620MHz、B512/D8、10warmup/2samples/batch2、无并发CPU构建/GPU任务：

|T/rank|EP16 rank-max p50/us|样本范围/us|M6f正式参考/us|
|---:|---:|---:|---:|
|8192|2961.620|2961.620–2963.180|2953.070|
|20480|7008.940|7008.940–7064.660|6991.990|

32个输出hash及全部pool bytes与M6f相同；未见性能收益，已撤回全部计算
lookahead源代码。没有继续library、完整Torch或sanitizer，不将快筛当作完整
正确性验收。冻结目录保留patch/resource报告，日志
`compute_lookahead_ej_ep16_t{8192,20480}_quick.log`。这只否定此两窗探测方案，
不能单凭一次端到端实验断言所有计算等待都无优化空间。

### EK：有界在途packet下跳过底层NIC队列检查（无稳定收益，撤回）

基于M6h冻结构建，仅在depth<=8、每owner/peer最多一个channel、单Put<=1MiB
时给GIN Put传`ncclGinOptFlagsMaySkipCreditCheck`；其它情况及ACK保持原样。
上层remote consumed、本地NIC completion、ready/acquire、payload发布均未删除。
这是尝试减少底层SQ/CQ重复检查，不是把远端credit当成本地completion。
安装的NCCL2.31.2 GDAKI头将该flag映射为DOCA availability-check跳过；相同映射
可见[NVIDIA源码](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/include/nccl_device/gin/gdaki/gin_gdaki.h)。
跨backend的队列容量单位、CQ软件进度仍需独立证明，浅payload ring本身不是
通用后端的完整安全证明；本轮没有把这个条件分支采用到生产。

native `856f50c16a5b8060a4d1847916305d5b780989b4571c0b64b19b26f1ebacd63b`，
冻结`build/static_v4_bounded_nic_credit_ek/`，含独立patch及M6h基线patch。
1620MHz、EP16、B512/D8、10warmup/2samples/batch2、无并发构建/GPU任务，
按EK→M6h→EK顺序相邻测量：

|T/rank|EK首轮/us|M6h对照/us|EK复测/us|
|---:|---:|---:|---:|
|8192|2930.060|2945.380|2947.950|
|20480|6958.860|6923.840|6978.220|

96个输出hash均等于M6h正式矩阵。20K未获益，8K差异不稳定；全部源码已撤回，
不增加后端特例或额外控制状态。没有继续library、完整Torch/sanitizer。
日志`ring_bounded_nic_credit_{ek,m6h,ek_repeat}_ep16_t*_quick.log`。

### EL：Return静态owner在CTA内共享context（退化，撤回）

保持32个Return channel owner与原packet/slot/counter分区，每4个owner放到同一
CTA的相邻lane，使用NCCL CTA sharing，共8个Return contexts。Dispatch/RX各32
个线程独占contexts不变，总计72；空出的24个Return warp暂不改派角色。
Return已由FC2直写TX，没有pack scratch供多个lane争用；预留、ready、returned
和Put仍由各自唯一owner推进。不把旧DM/DN的跨CTA GPU sharing或DQ/DR的
准备/发送交接开销混入这次消融。没有改上层双credit。

native `6d90db1d11b16bea587652a5141943da9c8106f762fd2d18b2b5a6a7cdeaa39e`，
冻结`build/static_v4_return_cta_contexts_el/`。网络inference prefill仍REG168/
SHARED2048/LOCAL0，STACK从432增为472。1620MHz、EP16、B512/D8、
10warmup/2samples/batch2、无并发构建或GPU任务：

|T/rank|p50/us|样本范围/us|M6h正式参考/us|
|---:|---:|---:|---:|
|1|294.000|294.000–659.296|277.926|
|64|364.768|364.768–369.232|333.410|
|8192|4087.330|4087.330–4098.610|2956.020|
|20480|10382.200|10382.200–10422.600|6985.870|

64个输出hash与M6h一致；长序列明显退化，T1还有一次大离群值。全部五个源文件
改动已撤回，不保留共享映射、额外分支或新常量；未继续library/完整数值/
sanitizer门禁。日志`ring_return_cta_contexts_el_ep16_t*_quick.log`。

### EM/EN：当前数据面下的宽FC1与静态ready lookahead（未证明收益，撤回）

EM只把prefill FC1物理N128加宽到N256，S逻辑列宽64→128；FC2、K128、2/2
stages、packed gate8/up8、窗口/通信及M6h checkpoint写回均不变。不是新的
gate/up布局对照。相比早期AC/AL，这次基线已包含DS直写Return及训练适配。
native `cb214ce085a0e0802a1ed1afc7033dc6a7fdc1b6b11f6dfaae7f1218f0a58c33`，
冻结`build/static_v4_prefill_fc1n256_em/`；prefill inference local/network仍
REG168、STACK64/432、SHARED2048、LOCAL0；训练为STACK72/448，arena不扩大。

1620MHz，第二节点独立EP8、B512/D8、10warmup/20samples；第一节点有CPU
构建/native正确性工作，测量节点无其它任务。inference为batch20 graph，
training为保留tape的eager Notify→buffer→Forward，不能混为同一计时范围。

|T/rank|同期M6h训练main/us|EM训练main/us|同期M6h训练wall/us|EM训练wall/us|EM inference/us|
|---:|---:|---:|---:|---:|---:|
|8192|1501.920|1563.840|2022.320|2089.400|1545.390|
|20480|3629.250|3736.060|4205.590|4308.880|3486.730|

同期冻结V3训练main为1337.980/3129.790us、wall为1748.060/3579.720us。
EM训练及inference各16个输出hash与M6h相同；native EP8 T512/E16/K6/B512/D8
两次Backward稳定，32个原始梯度hash逐字相同。仅加宽tile使训练和inference
均退化，不采用。日志`prefill_fc1n256_{v3,m6h,em}_remote_ep8_*`及
`prefill_fc1n256_em_native_ep8_t512.log`。尚无EM双机性能/完整数值验收。

EN在EM上复用已归档EJ的两窗口非阻塞ready探测，但用于N256的单机/网络prefill。
宽N会减少每窗口FC1 cluster数，此处检验相邻已ready窗口能否补足并行度。
leader统一选择，仍固定全局residue，不抢他人任务、不等待未来窗口credit；
对比EM可区分tile和调度收益。native
`14aa253ffefb4130eb86fdf5aa46d18aed37e40f7bf91724f97b5d032a2885f6`，
冻结`build/static_v4_prefill_n256_lookahead_en/`。同EP8计时合同下：

|T/rank|EN训练main/us|EN训练wall/us|EN inference/us|
|---:|---:|---:|---:|
|8192|1540.770|2037.010|1323.760|
|20480|3672.540|4248.380|3018.820|

相比EM恢复了一部分并行度，但仍未优于M6h，不能仅据相对EM的改善采用。
两种模式各16个输出hash与M6h一致。EP8 T512/E16/K6的B512/D8及B16/D1
两次Backward稳定，各32个原始梯度hash与M6h逐字一致；后者实际进入N256
prefill，验证lookahead不会等待D1尚不可准入的未来窗口。该native稳定检查
不代替完整Torch/sanitizer。两候选源码均已撤回，冻结binary的双机快筛待执行。
日志`prefill_n256_lookahead_en_{remote_ep8,native_ep8}_*`。

随后两候选完成EP16 inference快筛：1620MHz、B512/D8、10warmup/2samples、
batch2，无并发CPU构建或其它GPU工作。EM的8K/20K为2948.850/6946.830us，
EN为2981.140/6928.830us；全部64个hash与M6h相同，pool容量不变。
参照先前M6h正式2956.020/6985.870us，这些不足1%的差异不构成相邻正式收益，
且EP8均未优于基线。未继续完整Torch/sanitizer或采用网络专属调度分支；
两个源文件的全部候选改动已撤回。日志`{em,en}_ep16_t*_inference_quick.log`。

### EQ：当前数据面的math/control寄存器配额（未获益，不采用）

在M6h基线上仅通过既有编译宏将216/64改为192/120，不含EM/EN/EO/EP。
总配额`192×256+120×128=64512`不超过实际初始`168×384`，不增加SM资源、
改变tile/owner数或同步协议。早期AA只测到208/88且数据面尚未直写Return；
本项检验当前GIN角色增加寄存器是否有端到端收益，不能先把stack大小等同于
动态spill流量。冻结`build/static_v4_control_registers_eq/`及provenance，
未改变生产默认配置，正确性/性能尚待测量。

native SHA256 `fc548fb12e4be92eeae4f42021aa5fb9e94cfb464f7600131c7db47831ca83ef`。
inference prefill local/network的STACK64/432降为16/160，REG168/SHARED2048/
LOCAL0不变。1620MHz、B512/D8、10warmup/2samples/batch2的EP16四规模快筛：

|T/rank|p50/us|min–max/us|
|---:|---:|---:|
|1|272.960|272.960–277.920|
|64|333.008|333.008–333.744|
|8192|2946.820|2946.820–3009.540|
|20480|7051.440|7051.440–7142.990|

64个输出hash与M6h逐字一致，pool不变。计时无并发构建/其它GPU工作。
短规模接近M6h，20K未超过其6985.870us正式参考，stack下降不是端到端收益。
不采用192/120配额，未继续library/完整数值/sanitizer；生产仍216/64。
日志`control_registers_eq_ep16_t*_inference_quick.log`。

### ER：网络消费者等待中的轮询间隔（无收益，撤回）

独立于EQ/EP，基于M6h原216/64配额。在network Pull等准入/source reservation/
RX ready、local Return等FC2、Reduce等route-ready的未满足循环中，复用既有
64ns transport nanosleep。只有等待失败才延后下一次acquire；成功路径、每代
目标、ready发布和credit作用域均不变。不修改GIN TX/RX/ACK owner循环，也不
影响单机specialization。用来检验阻塞消费者的紧密轮询是否拖慢共享L2/控制流，
不预先认定这是瓶颈。等待现有双机测量完成才构建，避免CPU竞争。

native SHA256 `9b75ed5e9d85575d458ee2f76144670241663173c8fef34cc7d6ddcb795bdb80`，
冻结`build/static_v4_network_consumer_backoff_er/`。1620MHz、B512/D8、
10warmup/2samples/batch2、无并发构建/其它GPU任务：四规模分别为
280.448/336.080/2966.160/7082.000us，64个hash与M6h一致，pool不变。
没有端到端收益，两个源文件的候选改动已撤回；未继续library/完整数值/
sanitizer。日志`network_consumer_backoff_er_ep16_t*_inference_quick.log`。

### ES：以批量flush完成序号代替逐包本地counter通知（已验收，采用）

基于M6h原配额/原调度，不含ER轮询改动。保留逐slot远端consumed credit，
本地credit改为唯一TX owner维护的channel级`tx_completed_head`，复用
NetworkChannel原64-byte结构的末尾padding，不增payload/control容量。

Put仍携带自己的weak ready signal，但localAction改为None。首次将要复用
某TX slot时，若其上一代sequence尚不小于completed_head，则通过GIN
`flushAsync(peer)`、`wait(request)`确认该context对该peer的已发Put本地完成，
再将head推进到当前tx_cursor。成功之后才发放TX reservation/覆盖Dispatch TX。
Return reservation从不超过`tx_cursor+depth`，所以被复用的上一代一定已经
发出；flush不等待未发出的未来FC2或对方软件消费。其它TX/RX/ACK owner独立
推进，不在等待中持有共享软件锁。terminal同样flush全部已发包，并等逐slot
远端credit；下一调用只有完全drain后才用tx_base初始化completed_head。

一个context在多节点或更多channel时可以覆盖多条channel的Put，但本候选只
推进被检查channel的head，保守地允许额外flush，不错误推断别的owner已完成。
这保留了远端与本地两项条件，并非仅凭远端ACK删除本地completion；也不使用
skip-credit-check、aggregate-doorbell等后端可选优化。

依据已安装头文件及[NCCL 2.31.2 GDAKI实现](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/include/nccl_device/gin/gdaki/gin_gdaki.h)：
逐包counter涉及companion QP，而无Get的flush记录已发SQ位置、wait确认相应完成。
源缓冲区复用仍依赖显式本地完成，见[GIN合同](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/api/device_gin.html)。
这只是成本假设，flush等待也可能降低并发度。首轮为隔离协议比较，保留尚未使用
的counter编号预留/reset及slot local_target字段；只有性能与协议门禁获益后才
清理这些冗余，不把候选当最终代码。正确性、性能、资源及小ring反复复用待验证。

native SHA256 `3d04a541f052c686bb90763049e9ba5153a3ac9183950dfc86039eb10a872e99`，
冻结`build/static_v4_batched_local_completion_es/`。网络prefill inference/
training STACK400/424（M6h432/448），REG168/SHARED2048/LOCAL0；单机资源不变。
1620MHz、B512/D8、10warmup/2samples/batch2、无并发构建/其它GPU任务的
EP16四规模快筛为274.560/325.840/2931.330/6910.100us，64个hash与M6h一致，
全部pool容量不变。20K范围6910.100–6959.220us，初步差异仍小，正式相邻/
反向对照进行中，不预先宣布收益。

另EP16 T64/E64/K4/B16/D1 graph快筛通过（743.072us，不与B512计时混比），
16个hash相同；EP16 T64/E32/K6/B16/D1两次Backward稳定，64个原始梯度hash
与M6g2/M6h合同逐字一致。尚无完整Torch/sanitizer结论。
日志`batched_local_completion_es_{ep16,native_ep16}_*`。

随后EP16正式10warmup/20samples/batch20，按M6h→ES→ES→M6h顺序，
每轮先8K再20K，1620MHz、B512/D8，无并发CPU构建或其它GPU任务：

|T/rank|M6h相邻/us|ES相邻/us|ES反向/us|M6h反向/us|
|---:|---:|---:|---:|---:|
|8192|2956.620|2944.660|2953.630|2952.450|
|20480|6985.980|6922.910|6934.970|7012.400|

20K两组改善0.90%/1.10%，8K改善0.40%/退化0.04%，只能认为20K有小幅一致
收益，不能称为显著超越V3。128个输出hash与M6h相同，pool容量未变。日志
`batched_local_completion_{m6h,es}_ep16_t*_{adjacent,reverse}_formal.log`。
据此清理无用途的Forward counter编号/分配/reset及NetworkSlot.local_target；
保留Backward独立counter，将已有slot生命周期测试改为本地完成prefix加逐slot
远端credit，不另增单测框架。清理版冻结于
`build/static_v4_batched_local_completion_es_final/`，完整数值/sanitizer及
清理后的正式性能仍待验收，不以第一版结果替代最终artifact门禁。

清理版native SHA256
`3207471e9e80d6d293f109242864305bcb803c098120417d83428ae264117bf2`，
library `d9992af1336b35dbd8fc298f7a1951f01c8bbcd3976566d194a9d53cc3365098`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
网络prefill inference/training STACK392/408，REG168/SHARED2048/LOCAL0。
host原18项通过，包含逐slot生命周期和1/2/3/70节点布局；未新增单测框架。
native EP8 T512/E16/K6/B512/D8及EP16 T64/E32/K6/B16/D1均两次Backward稳定，
32+64个原始梯度hash与M6h/M6g2逐字一致。

清理版单机EP8在第二节点隔离测量，1620MHz、10/20/batch20：
T1/64/8192/20480分别100.883/135.429/1253.830/2948.240us，32个输出hash与
M6h一致，pool不变。第一节点同期有CPU构建/native正确性任务，不在测量节点。
单机prefill Function+instruction共16921行与M6h逐字相同，SHA256
`bd47119262cbb9a142a0ed189f0b7ee38bce236c4d48206cfaba2da45727feb1`，不能将
上述计时浮动归因于本次网络改动。

EP8/EP16 Torch训练矩阵1/64/8192均衡、20480热点全部通过；1/64均包含每rank
两次分配失败一致性、一次重试、三次peer payload/tape复用。附带graph与offload
是独立inference检查，不声称Backward支持graph。EP16专用inference
`--tokens 1 64 8192 20480 --edge-cases`有480条PASS；B16/D1/K8的1/64+edges
有304条PASS，覆盖unequal/empty、mixed profile、重复expert热点、跨代修改route、
多layer共享workspace和weight refresh。sanitizer与清理版EP16正式计时继续执行。
日志`batched_local_completion_es_final_{native,torch,remote_ep8}_*`。

EP16 T64热点/B16/D1、rank0/15的全weftmoe memcheck均0 errors；限定
`static_ring_plan_kernel`与`unified_forward_kernel`的Forward racecheck均
0 hazards/0 errors/0 warnings。两种运行均完成16 rank的数值、32次分配失败、
16次重试、48次live-tape/payload复用及inference graph/offload检查。
没有压制旧Backward shared-atomic工具报告，也不据Forward过滤声称整个
Backward racecheck clean；工具范围及已知限制沿用训练记录。
sanitizer原始日志前缀`batched_local_completion_es_final_{memcheck,racecheck}_`。
15:53两节点全部GPU graphics/SM均1620MHz，正式计时开始；没有并发CPU构建或
其它GPU任务，待结果确认后才提交该milestone。

清理版EP16 inference正式10warmup/20samples/batch20，B512/D8：

|T/rank|ES首轮/us|相邻M6h/us|ES反向/us|本轮冻结V3/us|
|---:|---:|---:|---:|---:|
|1|269.920|277.926（此前正式）|—|685.624（此前正式）|
|64|320.746|333.410（此前正式）|—|823.624（此前正式）|
|8192|2951.610|2956.270|2937.560|3360.000|
|20480|6931.390|6976.040|6924.360|7108.080|

20K比M6h降低0.64%/0.74%，延续首版ES的小幅一致收益；8K仍只能判断持平。
相比本轮V3，8K快12.15%–12.57%、20K快2.49%–2.58%，不能称为全部长文目标
达标。V3明确关闭V4且不启用ring，使用冻结V3 binary，不误测其自带V4入口。
ES两轮、相邻M6h及V3长文输出hash逐字相同；ES pool与M6h不变。计时节点无
并发CPU构建/GPU任务；新EU源码未参与这些冻结binary的计时。
日志`batched_local_completion_es_final_ep16_t*_{formal,reverse_formal}.log`及
`batched_local_completion_{m6h,v3}_ep16_t*_cleaned_adjacent_formal.log`。

训练Forward正式10warmup/20samples（eager，保留tape）另做相邻/反向：

|T/rank/顺序|M6h main/us|ES main/us|M6h wall/us|ES wall/us|
|---|---:|---:|---:|---:|
|8192，相邻|3480.990|3467.460|4283.240|4263.740|
|20480，相邻|8251.330|8306.850|9221.820|9268.270|
|20480，反向|8318.400|8272.610|9280.920|9222.650|

20K先慢0.67%、反序又快0.55%，两轮差异方向反转，不报告训练收益或稳定回退。
六份rank结果共96个输出hash与各自基线逐字一致，使用原始output_hash字符串，
不比较丢精度的JSON数值。wall含Notify→buffer→Forward，不能与inference graph
或main kernel混比。日志`batched_local_completion_{m6h,es_final}_ep16_t*_training_forward*.log`。

采用清理后的ES：保留全部保守复用条件，删除无用途的counter代码，网络inference
小幅改善，训练反向复测持平。未新增运行时开关/另一套训练Forward协议。
完整门禁只针对冻结ES artifact；EU的Return区间候选未编译、不包含在本次提交。
此milestone不是整体任务完成：EP16长文显著领先和训练Forward性能仍待优化。

### ET：ES完成机制下重测较小Return包（已验收，采用profile Q16）

只将Return Qmax32降为16，基线为清理后的ES，Dispatch Q8/L32/B512/D8、
216/64寄存器与静态owner不变。此前CN的Q16早于DS直写contribution和ES去除
逐包completion通知；本项检验较小包提前ready是否能抵消较多WQE，而非预先
假定小包更快。冻结`build/static_v4_return_q16_et/`，native SHA256
`511b129cb107c04406546a14cc773c571ded5bfcb630e4461472c4c366451e5d`。
仅使用既有编译参数，不改变生产默认Q32；测量会等ES门禁及所有构建结束。

1620MHz、B512/D8、无并发构建/其它GPU任务。10warmup/2samples/batch2的
EP16快筛：1/64/8192/20480为267.584/320.272/2923.460/6911.570us，全部64
输出hash与ES相同。随后ET→ES正式10/20/batch20：

|T/rank|Q16 ET p50/us|min–max/us|Q32 ES p50/us|min–max/us|
|---:|---:|---:|---:|---:|
|8192|2905.630|2898.260–2915.430|2940.800|2931.750–2944.730|
|20480|6899.340|6887.840–6914.270|6928.370|6913.100–6947.730|

8K约改善1.20%、20K约0.42%，仍是小收益，不取代最终V3性能目标。
四规模ET pool为45,398,976/75,137,216/513,073,344/864,091,584 bytes/rank，
比ES少约16.7MB；增加Return目录容量的代价已包含，不能只计TX减少的部分。
prefill inference/training资源与ES相同（REG168/STACK392或408/SHARED2048/LOCAL0）。
ET未包含EU的运行时Return区间搜索，library/完整数值/sanitizer及训练对照待完成。
日志`batched_local_completion_return_q16_et_ep16_t*_quick.log`及
`return_q16_compare_{return_q16_et,es_final}_ep16_t*_{quick,formal}.log`。

ET单机EP8隔离正式10/20/batch20为100.658/135.410/1276.780/2960.300us，
全部32个输出hash及pool与ES一致。第一节点同期有CPU library构建/native数值
检查，测量节点无其它任务。单机prefill的16921行Function+instruction与ES
完全相同（SHA256 `bd47119262cbb9a142a0ed189f0b7ee38bce236c4d48206cfaba2da45727feb1`），
不把长序列的小幅计时变化归给Q16。native EP8 T512/E16/K6/B512/D8及
EP16 T64/E32/K6/B16/D1两次Backward稳定，96个原始梯度hash等于ES/M6h。
日志`return_q16_et_{remote_ep8,native_ep8,native_ep16}_*`；完整Torch/sanitizer尚待完成。

ET完整EP16 Torch训练1/64/8192均衡、20480热点均通过；1/64各含32次分配失败、
16次重试、48次live-tape/payload复用。专用inference完整矩阵480条PASS，
B16/D1/K8短序列与edges304条PASS。T64热点/B16/D1的rank0/15 memcheck
各0 errors，限定planner及Forward的racecheck各0 hazards/0 errors/0 warnings，
两种sanitizer运行的应用检查均完成。Backward racecheck仍沿用已记录的工具限制，
不将Forward过滤结果称为全Backward clean。
library SHA256 `5f41c286aa0db5a179ec163f905ac060d528f450b126f4955d7bcdee64cc5345`；
extension SHA256 `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
ET library在ES干净源码上构建，未包含EU；EP8计算指令与ES相同，未重复整个
EP8 Torch矩阵，复用ES该部分验收，并新增上述ET单机native及四档正式检查。
日志`return_q16_et_torch_ep16_*`；memcheck日志PID445858/308702，
Forward racecheck日志PID446543/309076。16:21两节点全部SM/graphics仍为1620MHz，
开始EU独立快筛及ET短序列、20K反向和训练Forward对照；正式计时不并行构建。

ET短序列正式1/64为268.322/317.466us。20K反向ES→ET正式为
6924.510→6890.090us，ET范围6880.240–6906.940us、ES范围6914.470–6937.960us，
再改善0.50%，与首轮方向一致；全部64个输出hash仍等于ES。基于ES本轮冻结V3
7108.080us，ET20K约快3.07%，不是新的同轮V3复测。日志
`return_followup_{es,et}_ep16_t*_{formal,training_forward}.log`；训练对照继续执行。

训练Forward对照完成，10warmup/20samples、eager保留tape，B512/D8：

|T/rank|ES main/us|ET main/us|ES wall/us|ET wall/us|
|---:|---:|---:|---:|---:|
|8192|3461.790|3445.340|4282.190|4250.800|
|20480|8299.040|8300.320|9271.490|9263.080|

四份原始rank日志共64个输出hash一致；20K仅0.015%差异，8K样本范围重叠，
按训练持平处理，不声称训练性能恢复。wall含Notify→buffer→Forward，与main
及inference graph分别报告。全部计时结束前未启动EV构建。
采用ET作为本目录的推荐编译配置，只改既有Return Q参数，通用默认宏仍不变；
不新增Q16专用运行时路径。当前EP8正式四档100.658/135.410/1276.780/2960.300us，
EP16四档268.322/317.466/2905.630/6890.090us；8K使用首轮ET正式，其余为
后续正式轮次，不伪称同一轮交错矩阵。完整数值范围及单机复用证据见上文。

再做单机EP8长文同节点相邻V3→ET正式对照，1620MHz、10/20/batch20：
8K为1489.380→1268.040us（降低14.86%），20K为3295.890→2967.860us
（降低9.95%）。32个hash与已验收ET一致；V3明确关闭V4/ring。第一节点同期
构建EV，第二测量节点没有其它CPU构建/GPU任务；不是双机EP8。日志
`return_final_{v3,et}_remote_ep8_t*_formal.log`。不把这次同源EP8计时变化
归给仅改变网络包的ET。

### EU：运行时跳过Return连续块内部的路由读取（未获一致收益，撤回）

独立基于ES、Return Q32，不含ET。Return packet是单窗口expert-major路由
数组的连续区间，同一expert的source ordinal映射到严格递增逻辑行。因此
`(expert, logical Mblock)`形成连续run：先探测末行，必要时二分首个不同项。
ready检查和退休分别走run，returned仍按精确行数累加，不增加plan pass、
buffer、barrier或运行时依赖记录。expert必须参与比较，不能仅比较会复用的
物理slot。原线性实现已合并ready/acquire及retirement RMW，本候选只减少
run内部的不可变route/location读取，不将其误记为再次减少RMW。

冻结`build/static_v4_return_run_search_eu/`，native SHA256
`7a8e492f54e38fdebc795dfa0cab35174cae64606209fbadafae012724ceca63`，
网络prefill资源与ES相同（REG168/STACK392或408/SHARED2048/LOCAL0）。
1620MHz、无并发构建/GPU任务，10warmup/2samples/batch2：
8K为2914.990us（2914.990–2932.900），20K为6980.060us
（6980.060–7028.270）；32个hash及pool与ES一致。8K有改善迹象，但20K
未优于ES约6925us的正式对照，不能从减少读取推导端到端收益。未继续完整
数值/sanitizer，已撤回唯一源文件改动，不保留额外搜索逻辑。日志
`return_followup_eu_ep16_t*_quick.log`。

### EV：将网络packet ring最小深度从8降到4（长文退化，不采用固定4）

基于ES源码及ET的Q16配置，只修改host布局的最小depth。仍取
`max(4, ceil(B*K/(ReturnQ*lanes)))`，不减少最坏整窗口Return reservation，
所有逐slot远端credit及显式本地完成条件不变。source contribution深度D仍8，
底层NIC queue仍1024；三种深度不可混同。此改动同时减少Dispatch缓冲容量，
可能影响重叠，因此先独立测量，不预先宣称收益。冻结草稿位于
`build/static_v4_network_depth4_ev/`，不包含EU。

native SHA256 `86fa71fcadcdd1996e3ceb3a6b45b39cdb14f30b6d0ca5a676ef589c6d5258c9`；
网络prefill资源仍REG168/STACK392或408/SHARED2048/LOCAL0。1620MHz、
无并发构建/其它GPU任务、10warmup/2samples/batch2四档快筛：

|T/rank|EV p50/us|min–max/us|pool bytes/rank|
|---:|---:|---:|---:|
|1|252.848|252.848–253.360|28,605,376|
|64|300.768|300.768–301.504|58,343,616|
|8192|3437.500|3437.500–3472.450|496,279,744|
|20480|8076.500|8076.500–8104.670|847,297,984|

64个输出hash等于ET/ES。虽然少16,793,600 bytes/rank且短序列有改善迹象，
8K/20K显著退化，不采用固定4；未继续完整数值/sanitizer，原patch独立归档。
容量足以避免死锁，不意味着足以隐藏通信往返。这里同时减少Dispatch/Return
在途容量，不能直接把回退归因于其中一项。日志`network_depth4_ev_ep16_t*_quick.log`。

### EW：将网络packet ring最小深度从8增至16（无收益，撤回）

同ES+ET/Q16，反向验证EV暴露的在途容量敏感性。较小Return包在同一物理
深度下能容纳的窗口更少；测试多一些slot是否提升流水重叠，不预设结论。
仍只改变host最小depth，所有复用、整窗预留、contribution D8和NIC queue1024
不变；Dispatch容量也增加，必须同时报告总pool代价。冻结
`build/static_v4_network_depth16_ew/`。不包含EU或固定4，不新增运行时开关。

native SHA256 `f34e1602405c5325883c401a5008dd125b1a4e1f4d6ba2f42d7fca5dfc3209a6`；
网络prefill资源与ET相同。1620MHz、无并发构建/GPU任务、10/2/batch2快筛：

|T/rank|EW p50/us|min–max/us|pool bytes/rank|
|---:|---:|---:|---:|
|1|294.224|294.224–338.480|78,986,176|
|64|352.624|352.624–354.096|108,724,416|
|8192|2948.860|2948.860–2967.360|546,660,544|
|20480|6903.010|6903.010–6969.940|897,678,784|

64个hash等于ET/ES。长文没有明确收益，短序列退化，增加33,587,200 bytes/rank
也不划算；不采用、不继续完整门禁，源码已恢复最小depth8。日志
`network_depth16_ew_ep16_t*_quick.log`。EV/EW只能支持当前Q16配置下8优于
固定4/16，不能证明任意H/I/K/节点数的最优网络深度恒为8。

### EX：跳过本调用无包channel方向的slot扫描（无明确收益，撤回）

每个调用结束时，TX已确认本地完成及远端consumed credit，RX已发完ACK，
所有读者join后才会合并推进base。若新不可变plan中该方向count=0，双方均无
新payload/credit，不需要重新扫描上一代slot。TX与receive_progress在取可变
state前跳过count=0；非空方向沿用完整保守drain，反方向有独立plan与slot。
不保存额外done位，不新增状态、barrier或格式。active→empty→active的graph
复用须继续用既有路由修改/热点迁移门禁验证。

独立基于已验收ET及网络depth8，不包含EU/EV/EW；源改动仅两处提前continue，
冻结`build/static_v4_empty_channel_ex/`。尚未完成构建或测量，不报告收益。

native SHA256 `ac8200959287737967b9d0665e2b2938ce4872841ed86aa8fefc5d667ccc4457`。
1620MHz、无并发构建/GPU任务，10/2/batch2的1/64/8K/20K快筛为
263.008/318.288/2933.760/6921.630us，范围分别263.008–271.328、
318.288–496.336、2933.760–2949.090、6921.630–6930.370us。64 hashes及pool
等于ET；只有T1有小幅改善迹象，T64波动较大，长文未优于ET正式参考。
两处提前continue已撤回，未继续完整数值/sanitizer；不因逻辑上少做扫描就
保留未证明端到端收益的分支。日志`empty_channel_ex_ep16_t*_quick.log`。
16:43负载下两机全部GPU SM/graphics1620MHz、显存3201MHz。

### EY：RX先导入有界ready前缀，再扫描ACK（未获稳定收益，撤回）

独立基于ET，网络depth8，不含EX。原receive_progress每方向每轮最多导入
一个包，随后扫描全部slot的ACK；现每轮最多导入8个立即ready的连续包，
遇首个未ready/terminal即停止，然后执行原ACK扫描。每包仍由GIN读取ready、
累加自身exact expected、发布自身generation，不合并不同slot的生命周期。
无凑批等待，无额外字段/内存/barrier/开关，也不改变GIN doorbell或signal强度。
固定批量上限保证其它方向/peer获得推进机会；回卷包在旧ACK未发出时不可能
被TX重用并到达，因此不会凭空导入下一代。减少重复扫描只是待验证假设；
额外ready探测和延迟ACK也可能抵消收益。冻结`build/static_v4_rx_import_batch_ey/`。

补充源码审查：当前NCCL2.31.2的GDAKI Put/standalone signal实现不读取
`signal.isStrong`来区分发送路径；不对仅替换ACK Strong/Weak名称做无依据的
性能实验。保留原strong ACK及全部system release/acquire。

EY native SHA256 `7bd4f0fcc6c1b371483ee51ec23fa330c7139df9c87a39df4dd0af8c7491431b`；
网络prefill inference STACK392不变，training STACK408→416，REG168/SHARED2048/
LOCAL0不变。1620MHz、无并发构建/GPU任务，10/2/batch2快筛：
1/64/8K/20K为267.408/317.376/2898.000/6904.620us，范围分别
267.408–275.408、317.376–322.800、2898.000–2906.610、6904.620–6925.700us。
64个hash及pool等于ET。8K略快但仍落在ET正式样本范围内，20K未改善，
没有稳定端到端收益证据；未继续完整数值/sanitizer，源改动撤回。日志
`rx_import_batch_ey_ep16_t*_quick.log`。

### EZ/FA：在ES/ET上重测每角色共享16/8个GIN contexts（无稳定收益，撤回）

复用DM/DN的固定映射，保留每角色32个静态owner，使用GPU sharing提交；
任务不动态领取，packet/slot/双credit及contribution D8不变。新基线已删除
每Put的本地completion通知和companion-QP提交，与旧DM/DN成本不同；但批量
flush可能等待其它owner已预约的发送，也可能抵消收益。每channel完成prefix
仍只由自己的owner推进，不读取或推进其它owner的cursor。

EZ/FA分别使用48/24个物理contexts，普通编译默认32/role仍走THREAD独占。
初始化显式按`channel % physical_contexts`分配VA reset，避免24contexts少于
32workers时漏掉部分channel；Backward独立counter/slot协议不变。
transport pool大小不变，不据此断言NCCL内部context资源没有变化。
已有DH的64KiB链路对照（8→32及反向）是44.426/44.166 versus34.875/34.878GB/s；
这是不含计算/消费credit、仍有逐Put counter的微基准，仅作为重测动机。
两个候选源代码相同、编译quota不同，均不包含EY/EX/EV/EW；冻结
`build/static_v4_shared_contexts16_ez/`和`build/static_v4_shared_contexts8_fa/`。
两构建同时进行，结束后才串行计时，不把编译干扰带入性能数字。

native SHA256：EZ `24ec00c52e2cff1d9595eb0d30a515d73a271751079b5fc009395f3f40be410a`，
FA `8632abbcd8efcda7aee9ebe1ace6e69cee906f33931e5f3ddd65ad67d6d9a27c`。
两者网络prefill inference/training STACK376/392（ET392/408），其余
REG168/SHARED2048/LOCAL0不变。1620MHz、无并发构建/GPU任务，10/2/batch2：

|T/rank|16ctx/role EZ p50/us|min–max/us|8ctx/role FA p50/us|min–max/us|
|---:|---:|---:|---:|---:|
|1|270.160|270.160–271.488|265.712|265.712–278.064|
|64|324.288|324.288–324.880|321.936|321.936–324.128|
|8192|2917.140|2917.140–2917.310|2900.860|2900.860–2929.020|
|20480|6923.740|6923.740–6940.050|6926.980|6926.980–6951.860|

128个输出hash与ET/ES一致，transport pool不变。仍没有优于ET的稳定端到端
收益；微基准的context优势不能直接外推。三文件改动及临时quota宏全部撤回，
保持96个THREAD独占contexts，不继续完整数值/sanitizer。两个artifact保留
source patch及初始化修正，供以后有新证据时复核。日志`shared_contexts_{ez,fa}_ep16_t*_quick.log`。

### FB/FC：直写Return之后重测source-major Mblock行序（无收益，撤回）

早期AH在不同数据面上没有获益。当前FC2直接写Return TX后，同source的连续
Return routes仍因source round-robin跨多个Mblock，包ready需同时等它们。
重测`expert_row = prefix(previous sources) + source ordinal`，让同source行
连续，期望减少每包producer依赖及每Mblock的X来源；也可能造成新的就绪偏斜，
不预先假定改善。无额外padding、payload、plan pass、barrier或owner。

只改window geometry及既有GPU几何校验的独立参考枚举。当前planner统一由
该inverse绑定RingRow、gate、output、RouteLocation，运行时消费者不重建旧
round-robin行号。训练checkpoint仍依据Notify canonical row，未改变Backward
dW归约顺序，也不把不稳定的inference source ordinal称为StableTape。
FB编译Q16（对照ET），FC编译Q32（对照ES），其余L32/B512/D8、网络depth8和
96个独占contexts相同；同一source变更并行构建。冻结
`build/static_v4_source_major_q16_fb/`、`build/static_v4_source_major_q32_fc/`。

FB native SHA256 `17fa1fb2ef476551406e33e952f444c0931381a8fdf7c1db31fe95ee32d99e28`；
FC native SHA256 `07e7b062c60b54709a7301190a5b0f8dec5cadae19b00bb2a80f247d192d76ab`。
已有GPU plan检查37个source/packet+6个destination cases通过，包含不等长/空
source和三节点partial group；同程序racecheck0 hazards/0 errors/0 warnings。
网络prefill inference/training资源与ET/ES相同。两构建全部结束后，1620MHz、
无并发构建/GPU任务，10/2/batch2的EP16快筛：

|T/rank|FB Q16 p50/us|min–max/us|FC Q32 p50/us|min–max/us|
|---:|---:|---:|---:|---:|
|1|276.384|276.384–567.040|267.456|267.456–273.344|
|64|320.560|320.560–321.568|325.072|325.072–326.912|
|8192|2971.500|2971.500–2981.380|2988.900|2988.900–3034.960|
|20480|7011.700|7011.700–7041.310|7010.180|7010.180–7077.060|

128个输出hash与ET/ES相同；各自pool等于对应Q的round-robin基线。两个Q均未
改善inference，不采用该inference行序。完整训练checkpoint写回的局部性可能
不同，另做FB vs ET的8K/20K eager retained-tape对照后再决定是否保留训练用途。
源geometry和参考枚举已恢复，测量只用冻结FB/FC binary；后续FD草稿不包含它们。
日志`source_major_{fb,fc}_ep16_t*_quick.log`，规划器日志`source_major_fb_plan*.log`。

训练Forward正式10warmup/20samples、eager保留tape、B512/D8、1620MHz，
无并发构建/GPU任务的相邻ET→FB对照：

|T/rank|ET main/us|source-major FB main/us|ET wall/us|FB wall/us|
|---:|---:|---:|---:|---:|
|8192|3465.760|3474.590|4258.360|4275.310|
|20480|8328.380|8442.340|9296.990|9414.720|

64个原始Forward output_hash字符串一致。8K无收益，20K main慢约1.37%，
不为训练保留额外行序；没有声称训练tape/Backward已完成该候选的完整验证。
geometry与测试参考均撤回，继续共享原round-robin布局。日志
`source_major_{et,fb}_ep16_t*_training_forward.log`。FD构建在本组计时结束后才开始。

### FD：Return整包同Mblock时跳过内部重复读取（无稳定收益，撤回）

EU在Q32使用通用run二分搜索，未改善20K；本项独立基于已验收ET/Q16及原
round-robin，只处理整包同一block的情况，不含FB/FC或此前候选。先acquire
首行的FC2完成，再比较包首尾expert及逻辑block；expert-major区间内同expert
的逻辑block单调，首尾相同即证明所有行属于该block。未ready的首行不额外
探测末行。若证明成立，两处原扫描的步长改为rows，退休计数也加rows；否则
仍是原coalesced线性扫描。没有依赖目录、二分搜索、额外生命周期状态或barrier。
比较expert+logical block，不仅比较可能回卷的物理slot。冻结
`build/static_v4_return_single_block_fd/`；尚无性能/完整数值结论。

native SHA256 `fc7d5fbf7e4d22fc4d931a7a0e9de2023088889cc498de72cec67335b81dad54`。
网络prefill inference STACK384（ET392），training仍408，其余资源相同。
1620MHz、无并发构建/GPU任务，10/2/batch2的EP16快筛：
1/64/8K/20K为266.896/318.224/2927.360/6892.420us，范围分别
266.896–656.272、318.224–321.872、2927.360–2928.000、6892.420–6921.490us。
64个hash等于ET/ES。相邻ET的20K两样本6859.420–6940.370us，均值6899.895；
FD均值6906.955，没有稳定改善。两样本p50实际取较小值，不当作正式收益。
未继续完整数值/sanitizer，两处捷径已撤回。日志
`return_single_block_{et,fd}_ep16_t*_quick.log`。

### FE/FF：仅减少Return TX的静态owner（明显退化，撤回）

独立基于ET，保持Dispatch、RX/ACK、Return RX、local Return、Pull、Reduce的
owner数量及residue不变，仅将Return TX从32减到16/8；空出的warp不重分配。
每个context仍THREAD独占，不使用EZ/FA的GPU sharing或DQ/DR的额外准备队列。
contexts分别80/72，原96；每channel的双credit、完成prefix及整窗预留不变。
同一owner的flush可能覆盖其多个channel，但只保守推进当前channel的已完成头。
源代码五文件同一份、编译配额不同；不含FD捷径，初始化仍覆盖全部channel。
冻结`build/static_v4_return_owners16_fe/`与`build/static_v4_return_owners8_ff/`。
native SHA256分别`8790e852d2081237492702481c0bbe6d7de25c61413d871846ee2e3b95ae398d`、
`0237be8bc21955b0e74b91a5a0ad5799a37e097857d4a45bb678dd556571e9f0`。
构建全部结束后串行计时。长文明显退化，源改动已撤回；短规模仍用冻结binary补测。

1620MHz、无并发构建/GPU任务、10/2/batch2的完整四档快筛：

|T/rank|FE16 owners p50/us|min–max/us|FF8 owners p50/us|min–max/us|
|---:|---:|---:|---:|---:|
|1|282.560|282.560–298.560|325.488|325.488–333.056|
|64|334.704|334.704–339.472|389.440|389.440–391.936|
|8192|3927.540|3927.540–3951.570|7080.110|7080.110–7085.200|
|20480|9775.220|9775.220–9791.810|18477.800|18477.800–18498.200|

128个hash与ET/ES一致、pool不变；不因减少GIN资源而保留明显慢的配置。
保留原32Return owners及96个独占contexts，没有进行候选完整数值/sanitizer。
日志`return_owners_{fe,ff}_ep16_t*_quick.log`。这组对照不证明API提交、元数据
扫描和reservation各自的成本比例，仅证明当前实现不能无代价地减少Return owner。

### FG：单独将prefill FC2 N256改为N128（退化，撤回）

独立基于ET，FC1 N128、K128、stages2/2、math/control配额、通信布局和全部
静态owner不变；仅FC2列tile变窄。相比此前扩大FC1的EM，本项测试更早完成
单个FC2 tile与增加任务/同步次数之间的权衡。共享GEMM模板、描述符和epilogue，
不新增协议或运行时开关；decode不变。计时仍等待FE/FF串行快筛结束再构建。

冻结`build/static_v4_prefill_fc2_n128_fg/`，native SHA256
`864e9a5515716a5a76bccd4a5717c618fa700ba609346ec6aad29ca8f409ce03`。
1620MHz、B512/D8、10/2/batch2长文快筛：

|拓扑|T/rank|p50/us|min–max/us|
|---|---:|---:|---:|
|单机EP8|8192|1466.910|1466.910–1473.600|
|单机EP8|20480|3451.180|3451.180–3466.880|
|双机EP16|8192|2985.390|2985.390–3005.570|
|双机EP16|20480|7100.080|7100.080–7117.380|

48个hash与ET一致。双机计时无并发编译/GPU任务；单机在第二节点独占运行，
第一节点在其20K结束后开始FH CPU构建，不使用第二节点CPU/GPU。两种拓扑
均比ET慢，不继续短规模/完整数值/sanitizer，源码已恢复prefill FC2 N256。
该对照不能证明其它H/I或训练形状也应选择N256。日志
`prefill_fc2_n128_fg_{ep16,remote_ep8}_t*_quick.log`。

### FH：FC2发布完成后提前退休C控制，不再等Return扫描（正式无收益，撤回）

独立基于ET，非BL/BM的直接跨机拓扑；FC2 N256、全部peer/owner、包划分和
buffer容量不变。CN/CO之后FC2已直写预留TX，Return TX及NIC不再读C数据，
但仍通过逐包returned RMW保护rail-local完成镜像的等值读取。本项把镜像定义
为累计完成generation，TX以`published >= requested`确认此前FC2已完成。
同一物理slot的generation随窗口/调用单调增加；未命中某rail的代不写它，
不影响旧代完成证明。packet不跨窗口，原相邻同slot检查合并仍不会混合两代。

local Return owner acquire所有FC2 N-tile完成、执行原system fence、发布本地
route ready和每个相关rail恰好一次完成镜像，**全部lane join之后**归还该block
完整rows的C credit。准入只能在此release之后换代，旧镜像store不能晚于新代。
首轮保持原system-scope C退休/准入，不同时消融scope。删除TX的第二遍逐行
C退休扫描及已无调用者的c_slot helper，净减代码，无新增计数器/缓冲区/barrier数。
单机return的join位置在编译后逻辑不变。

C提前回收不授予TX复用：整窗TX预留仍是FC2前提，TX仍等远端consumed和
显式本地GIN completion。RouteLocation/RingRow是整次调用不可变目录，不随
C物理slot回卷；source contribution仍等该窗口全部token Reduce后才复用。
证明归档`build/static_v4_early_c_retire_fh/provenance.md`；首轮只构建native，
有收益后才继续完整数值/小ring wrap/训练/sanitizer，尚未声称候选正确性已验收。

native SHA256 `83d5896a27ef29225caee5de8474edb49212a1dca9ef0ac7bf7a0e4946cd58b3`；
网络prefill inference/training REG168、STACK392/408、SHARED2048、LOCAL0，与ET
相同。单机prefill16921条Function/指令行SHA256仍为
`bd47119262cbb9a142a0ed189f0b7ee38bce236c4d48206cfaba2da45727feb1`。
1620MHz、无并发构建/GPU任务，快筛8K/20K为2911.420/6944.210us，32hashes
及pool等于ET。随后正式10warmup/20samples/batch20按每档ET→FH→FH→ET：

|T/rank|首轮ET/us|首轮FH/us|反向FH/us|反向ET/us|
|---:|---:|---:|---:|---:|
|8192|2907.550|2911.740|2922.430|2908.390|
|20480|6955.740|6897.260|6901.690|6902.070|

128个正式hash一致。20K首轮约0.84%的差距未在反向复现，8K反而略慢，不能
把更早C回收或更少RMW直接宣称为端到端优化。源码三文件已撤回，不继续完整
数值/sanitizer；保留这一简化协议的证明与patch，若未来重用仍需完整验证。
日志`early_c_retire_fh_ep16_t*_quick.log`及
`early_c_retire_formal_{0..7}_{et,fh}_ep16_t*.log`。

### FI：只共享RX/ACK contexts，保留全部静态owner（无收益，撤回）

独立基于ET；FH及此前候选均撤回。每4个RX/ACK owner固定共享一个GPU-sharing
context，Dispatch/Return的32+32数据contexts仍THREAD独占，全部角色的数量
和channel residue不变。W32时总contexts96→72（32+32+8），pool布局/容量不变。
区别于EZ/FA的所有角色共享、EH的减少RX owner：本项不让data Put/flush共享QP，
也不减少RX轮询和ACK生成的并行owner，只检验ACK QP资源与提交串行化的取舍。

每个slot的exact expected、consumed、ready、ACK generation仍只有其原owner
推进；共享的是GIN内部提交资源，不是channel状态或动态任务。GPU sharing提供
跨CTA的WQE预约/发布顺序；ACK仍逐slot独立发送，没有合并credit。终端会合
需要对端TX观察全部ACK才能到达，未省略原drain。初始化仍按channel%W：
物理contexts始终至少2W，覆盖所有reset owner。metadata按实际contexts分片，
Backward的独立counter和contexts合同不变。适用于原固定channel residue遍历
多个远端peer的方式，不硬编码双机。

实现四文件、无新运行时开关/测试/状态字段，冻结
`build/static_v4_shared_ack_contexts_fi/`。先构建native、长文相邻对照；有明确收益
后再做四规模完整数值/训练/sanitizer。尚无测量或完整正确性结论。

native SHA256 `2ca7c6e8a50a4eaced7c0b56c06cc6bb2ba9d02f478825ee9f893f2157138ecf`；
网络prefill inference/training STACK400/416（ET392/408），其它资源不变。
1620MHz、无并发构建/GPU任务、10warmup/20samples/batch20，EP16 8K/20K
正式p50为2906.740/6919.990us，范围2900.620–2910.730/6907.370–6942.350us。
32个hash与ET一致。对照紧邻FH末轮ET的2908.390/6902.070us，8K持平、20K
略慢，未证明单独共享ACK QP有益。四文件已撤回，未继续短规模/完整数值/
sanitizer；保持所有contexts独占。日志`shared_ack_contexts_fi_ep16_t*_formal.log`。

### FJ：在原共享内存预算内增加prefill FC1 TMA stage（候选）

独立基于ET，FI/FH均撤回。FC1物理N128、FC2 N256、K128、cluster2及通信
全部不变，只把prefill stages2/2改为3/2；decode仍3/3，不新增运行时参数。
AF曾独立测过有效S-N64/K128/stages3的纯GEMM，但那不代表本次融合实现
或当前数据面的端到端收益。本项在当前融合路径独立验证。

旧compute helper将max(A stages)与max(B extents)相加；但FC2 B实际已使用
自身A extent作为偏移。3/2下旧公式是A96+Bmax128=224KiB，而实际FC1
A96+B96、FC2 A64+B128的上界都是192KiB。仅StaticSchedule按FC1自身B
extent计算该相加项，再与原FC2 arena取max；V3动态分支表达式保持原值。
packed FC1 epilogue接收实际可用scratch=arena-A，而不是FC1操作数B的大小，
保留decode的96KiB、原2/2prefill的128KiB可用scratch；3/2时为96KiB，
仍容纳M6h所需的48KiB S/G/U三平面。没有为了3-stage增加物理SMEM预算。

原host allocation覆盖192KiB operand和12个barrier；3/2需要192KiB和10个。
FC1/FC2切换仍经过原producer/math及cluster join，不在FC1第三A stage尚有
读者时将其交给FC2 B。K累加次序、舍入、TensorMap形状和completion targets
不变。候选只改两文件，冻结`build/static_v4_prefill_fc1_stage3_fj/`；先构建
native并比较单/双机长文。decode机器码、训练checkpoint及完整数值/sanitizer
仍需验证，不因空间算式成立就宣称stage复用已经通过验收。

native构建完成，SHA256 `fd6102c393ad4ffbc7116b303b31816ad47ba3d0d1095a4cd6ba7b8eabd130f0`。
四个prefill inference/training入口的REG/STACK/SHARED/LOCAL与ET相同。
decode的Function/指令行逐条一致：单机14273行SHA256
`c89c32287e283744b15580bb7f702db5eac307fcb2c92fc7f76f999ecccf27b4`，
多机41097行`b8f39bea98b1d9877e208ad21f48cc5ee77fe1ae233a9c4836d8d119ee30dbb9`。
构建结束、无并发GPU任务/编译，第二节点单机EP8，1620MHz、B512/D8、正式
10warmup/20samples/batch20，相邻ET→FJ的8K为1273.660→1247.510us，
20K为2959.860→2954.060us。32hashes与ET一致；8K约2.05%改善、20K基本持平，
仍待反向和EP16对照，不提前采用。日志
`prefill_fc1_stage3_{et,fj}_remote_ep8_t*_formal.log`。

随后EP16同规格ET→FJ：8K 2907.700→2897.600us，20K
6895.990→6873.880us，64个hash与已验收版本一致。反向单机FJ→ET：
8K 1248.530→1269.210us，20K 2950.260→2968.430us，32个hash一致。
单机8K两轮改善约1.6%–2.1%，20K接近持平至小幅改善；双机仍待反向确认。
反向单机测量期间只在第一节点编译库/做native数值检查，第二节点独占计时。

native EP8 T512/E16/K6/B512/D8覆盖新prefill路径：8个rank的重复Backward
均一致，32个dX/dGate/dW1/dW2 hash与ES/M6h完全一致。库SHA256
`ee1c9dbaeca45164505d474eacdbdfdfa183aec9d7bba56859378abb4ad2d498`，
extension为`725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
完整Torch及prefill sanitizer仍待验收；尚未采用。

EP16反向8K为FJ2904.730→ET2912.170us。第二组反向脚本曾误写未识别的
`WEFTMOE_BENCH_TOKENS`；核对结果JSON的shape发现文件名中的20K实际仍是
默认8K。这两份`reverse_{fj,et}_ep16_t20480_formal.log`不作为20K证据；
保留原日志，用正确的`WEFTMOE_P2P_BENCH_TOKENS`另跑`reverse_verified_*`。
首轮EP16及两轮EP8的JSON shape已复核正确。

修正后EP16反向20K为FJ6868.650→ET6903.970us，范围
6859.730–6875.390/6890.140–6914.090us，32个hash一致；两轮分别改善
0.32%/0.51%。测量结束时两节点全部GPU的SM/graphics为1620MHz、memory为
3201MHz。正式测量无并发构建/其它GPU任务。进入完整验收，未提前关闭目标。

**验收失败，暂不采用：** EP16 T2048/B512/D8 prefill memcheck的rank0/15
均0 errors，全部304条数值检查通过；但相同用例在Forward racecheck下，
balanced-eager通过后balanced-graph出现64/256个元素的局部错误，最大绝对
误差约0.10。rank15工具虽报告0 hazards，不能当作数值通过；rank0因其它
rank异常退出而等不到通信，已终止本次测试进程。1-token训练及native两组
重复Backward通过，不足以替代此失败。当前不发布FJ为可用优化。

代码审查发现遗漏：静态调度没有逐task的cluster rendezvous，较快CTA可能
进入下一FC1，multicast A第三stage写入64–96KiB；较慢CTA的FC2 B和epilogue
仍从64KiB开始。此前“两阶段各自占192KiB”的空间论证没有覆盖跨CTA的
阶段偏斜，是不充分的。正在以ET同用例复测，并检查跨阶段operand/scratch
边界；仅改epilogue偏移尚不足以证明尚存的FC2 B读者安全。收益暂不计入验收结果。

ET使用原冻结库复跑完全相同的T2048/B512/D8 Forward racecheck：304条数值
检查通过，rank0/15均0 hazards。日志`prefill_stage_audit_et_*`。FJ的T64及
8K训练随后也完成parity/graph/offload，但不能抵销上述失败；后续性能和完整
验收任务已暂停。没有修改容差、忽略局部错误或将0 hazards替代数值检查。

### FJ2：跨GEMM的multicast operand退休与scratch边界（修正候选）

保留三级FC1和192KiB预算，FC2非pipelined epilogue改从max(A extents)开始，
即96KiB，64KiB输出tile至160KiB结束，静态断言其落在arena内。不再使用
FC2操作数B的64KiB起点作为epilogue scratch。

producer从FC1切到FC2、或反向切换时，复用旧阶段每个empty barrier，按下一次
发射的phase^1等待全部退休。原empty arrival包含两个CTA各8个math warp，
且在WGMMA wait0之后发布；所以新的multicast不会覆盖peer仍在读取的旧A/B。
没有增加计数器、cluster barrier或数据面全局同步。math仍可独立完成私有
epilogue，较快peer可在其后继续multicast，但两个epilogue的scratch都避开
两种A extent。当前CTA自己的producer仍受原task barrier约束，不能提前
覆写自己的B/scratch。这补全了原FJ仅按阶段算空间所遗漏的跨CTA生命周期。

该修正适用于静态decode/prefill及ring/nonring，V3动态分支不变，因此FJ的
decode指令不变结论不再适用于FJ2。冻结
`build/static_v4_prefill_fc1_stage3_fj2/`，先复验失败用例，再做完整数值与
四规模性能；未经复测不沿用FJ原有收益。

FJ2库SHA256 `b9ae71fa6a7ef4d3ef873550011e0ddbddd7f4881eb0b6eab4b49ba83f4c92a1`，
extension仍为`725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`；
prefill四入口的REG/STACK/SHARED/LOCAL与ET相同。原失败的EP16 T2048/B512/D8
Forward racecheck复验已通过304条数值检查，rank0/15均0 hazards，覆盖原先
balanced-graph失败点及后续热点、复用。日志`prefill_fc1_stage3_fj2_prefill_audit_retry_*`。
首次启动因临时脚本库路径拼接为`fj22`而在import阶段失败，没有执行算子；
保留无retry日志，不将其计入数值测试。复验时本机同时进行native CPU编译，
仅作正确性检查，不用于性能结论。完整矩阵和性能仍待完成。

FJ2 native SHA256 `abd3ca7108283733de56b9e1d094b63d9db30dca874d8b1341ab5707e09fdb8e`。
单机EP8 Torch训练四档通过：短档每档8 parity、16 preparation-failure、8 retry、
24 live-tape reuse、8 graph、8 offload；8K balanced及20K hotspot每档8 parity/
graph/offload。Inference四规模+edges 240条、tiny B16/D1 K8 152条、非ring
8K/20K 32条检查通过。全部复用既有测试入口，没有增加单测。

1620MHz、正式10warmup/20samples/batch20，第二节点独占测EP8；第一节点
同时做Torch数值检查，无并发CPU构建。输出hash与已验收版本一致：

|T/rank|FJ2/us|本轮相邻V3/us|相对V3降低|
|---:|---:|---:|---:|
|1|101.050|未重测|—|
|64|135.368|未重测|—|
|8192|1250.880|1487.410|15.90%|
|20480|2953.440|3293.330|10.32%|

日志`prefill_fc1_stage3_v2_final_{v3,fj2}_remote_ep8_t*_formal.log`。
保留tape的EP8 eager Forward 10warmup/20samples相邻ET→FJ2：8K main
1508.160→1502.460us、wall2006.450→1999.240us；20K main3635.360→3641.820us、
wall4217.450→4237.720us。32个字符串output hash一致，差异小且方向不一，
只判断训练持平，不声称训练加速。双机完整矩阵、正式性能及ET inference
相邻对照仍待完成。

随后EP8反向FJ2→ET相邻正式对照：8K 1254.230→1264.640us，改善0.82%；
20K 2952.550→2951.360us，持平。32hashes一致。不能用修正前FJ的1.6%–2.1%
替代修正后收益；目前FJ2单机8K有较小改善，20K无新增收益。
日志`prefill_fc1_stage3_v2_adjacent_{fj2,et}_remote_ep8_t*_formal.log`。
FJ2 native EP8 T512/K6/B512/D8两次Backward也通过，32梯度hash等于ES/M6h。

双机native T64/K6/B16/D1两次Backward通过，64梯度hash等于ES，合计96个。
EP16训练四档完整parity/graph/offload通过，短档每档32次预期preparation失败、
16 retry及48 live-tape reuse；20K为热点。Inference四档+edges 480条、
tiny B16/D1 K8 304条通过。额外T2048 prefill memcheck/racecheck分别304条
数值检查通过，原失败用例共完成两轮Forward racecheck复验。

rank0/15的sanitizer摘要：decode热点训练memcheck（PID28050/342037）、
prefill inference memcheck（30999/343002）、prefill热点训练memcheck
（32580/343744）均0 errors；decode热点Forward racecheck（28992/342425）、
prefill inference第二轮racecheck（31551/343275）均0 hazards。
racecheck仅过滤planner/Forward，不将既有Backward原子报告的未关闭事项
说成clean。日志`prefill_fc1_stage3_fj2_torch_ep16_*`及同前缀sanitizer日志。

EP16非ring 8K/20K的64条数值检查通过，整套launcher退出0。四档EP8 pool
bytes与ET逐rank完全相同，新增同步没有增加payload或控制状态分配。
FJ2双机正式四档及相邻ET/训练对照正在进行，未以修正前结果替代。

双机正式矩阵完成，仍为1620MHz、10warmup/20samples/batch20；两节点独占，
无并发构建或其它GPU任务。长文V3为同轮相邻重测：

|T/rank|FJ2/us|本轮相邻V3/us|相对V3降低|
|---:|---:|---:|---:|
|1|269.517|未重测|—|
|64|317.286|未重测|—|
|8192|2905.500|3356.450|13.43%|
|20480|6875.650|7098.190|3.14%|

日志`prefill_fc1_stage3_v2_final_{v3,fj2}_ep16_t*_formal.log`。
相邻ET→FJ2 inference对照：8K 2911.820→2898.280us，20K
6893.770→6882.280us，分别改善0.47%/0.17%，不是显著新增收益。
这四份日志的32个output hash逐rank一致，pool及注册bytes完全不变，JSON
shape已核实。日志`prefill_fc1_stage3_v2_adjacent_{et,fj2}_ep16_t*_formal.log`。
V3 graph包含两个kernel，ring主路径一个，以上比较是端到端，不是纯GEMM。

训练8K main3451.040→3452.350us、wall4259.480→4262.680us，持平。
20K顺序ET→FJ2 main8282.620→8341.660us、wall9226.900→9315.530us；
反向FJ2→ET main8340.100→8316.960us、wall9343.920→9263.080us。
两轮main均回退约0.71%/0.28%，wall回退约0.96%/0.87%。三组共96个原始
uint64 hash按字符串比较一致；不以JSON浮点解析比较64位整数。
保留所有结果，不用反向复测掩盖首轮。日志
`prefill_fc1_stage3_v2_{final,reverse}_{et,fj2}_ep16_t*_training_forward.log`。

FJ2完整数值门禁通过，但暂不整体采用：下一步FJ3保留同步/scratch修正，
训练prefill的FC1恢复两级，inference保留三级，独立确认上述训练回退来自
stage参数还是同步修正。tile/weight布局、内存分配及decode不变。

### FJ3：训练FC1单独恢复两级（无收益，撤回；采用FJ2）

冻结`build/static_v4_prefill_training_stage2_fj3/`，native SHA256
`b512a33513a63fd0423227ac8b62a70bff8a5ac0788bdf24178b9602a94f8614`。
仅prefill retained-tape使用两级FC1，保留FJ2同步及scratch边界，inference
不变。1620MHz、EP16独占、无构建/其它GPU任务、10warmup/20samples：

|T/rank|ET main/us|FJ3 main/us|ET wall/us|FJ3 wall/us|
|---:|---:|---:|---:|---:|
|8192|3454.110|3453.410|4266.660|4252.770|
|20480|8319.460|8358.400|9279.800|9316.950|

各组32个原始hash字符串一致。20K仍回退0.47%，没有恢复此前差距，不保留
独立训练stage分支。FJ3未做library/full-gradient/sanitizer门禁，不称其已验收。
日志`prefill_training_stage2_{et,fj3}_ep16_t*_training_forward.log`。

最终恢复并采用已通过完整门禁的FJ2，恢复后源码patch与冻结FJ2逐字节一致。
采用理由是补全跨CTA multicast与私有scratch的生命周期证明，并取得单机8K
的小幅改善；不声称训练优化。训练20K观测到约0.3%–0.7%的main代价，继续作为
未解决性能项记录。没有以性能理由撤掉必要同步，也没有增加无收益的训练专用
调参分支。核心目标仍未完成：EP16 20K相对V3仅快约3.1%，训练Forward仍落后。

### FK/FL：保留全部Return owners，分离CTA排布与context共享（无收益，撤回）

在FJ2上对DH观察到的NIC context数量敏感性继续作独立验证。此前FE/FF减少
Return owners、EZ/FA跨CTA GPU共享、EL同一warp内多lane发送均未获益；这里
保持P个Return owners，用三个**独立warp**集中在ceil(P/3)个CTA上。
FK只改排布，仍保留3P个THREAD-exclusive contexts；FL才把Return改为
ceil(P/3)个CTA-sharing contexts，Dispatch/ACK各P个不变。未实测前不称收益。

固定C个CTA、P个network workers（当前132/32）：Return TX有P个warp；
剩余CTA的物理warp0/1承载C个copy owners，前P个Dispatch、其余C-P个pull。
只有这些角色读写私有双slot staging。所有剩余warp（含Return尾CTA的空位）
按稠密ordinal依次分给admission1、ACK P、Return RX P、local Return C-2P、
Reduce C-P-1。各角色数量和channel/block/token residues与FJ2相同，无动态领用。
copy pipeline改用实际warp0/1索引，删除Return路径无用途的prefetch对象。

FL共享仅发生在同一物理CTA，context数量由96降至75（32+11+32）。检查了
已安装NCCL的CTA SQ reservation、ready-prefix及doorbell实现。每个发送者
在进入GIN前完成软件ready/credit判断，GIN内不等待FC或ring软件依赖；共享
flush可以覆盖其它warp已预留工作，但只推进本channel的tx_completed_head
到自己的tx_cursor，仍同时要求远端consumed credit。metadata初始化/复制
仍在原阶段barrier之前，不与数据面共享context的调用并发。

冻结候选为`build/static_v4_return_cta_packing_fk/`和
`build/static_v4_return_cta_shared_fl/`；先快筛相邻FJ2/FK/FL，再决定是否
值得完整门禁。没有增加长期调参开关或专用单测。

native SHA256：FK
`74eae4c152d612986a72529752184693c7cb1cabd53f87e65631fdea69139400`，FL
`002aac16623fcccf001235b179ae4334e72aecf1091835e8728698e14009031f`。
network prefill inference STACK为FK400/FL408（FJ2392），training均416
（FJ2408）；REG168/SHARED2048/LOCAL0未变。两版EP16 T64/K6/B16/D1各两次
Backward通过，各64个梯度hash与FJ2完全一致。FK数值测试与FL CPU编译并发，
仅作正确性检查；以下计时在全部构建结束后，两节点独占、无其它GPU任务。

1620MHz，10warmup/2samples/batch2快筛，单位us，括号为min–max：

|T/rank|FJ2|FK：仅排布|FL：再共享Return contexts|
|---:|---:|---:|---:|
|8192|2889.010（2889.010–2897.460）|2897.570（2897.570–2908.500）|2914.640（2914.640–2977.330）|
|20480|6885.470（6885.470–6914.450）|6878.430（6878.430–6919.060）|6921.150（6921.150–6940.500）|

96个Forward hash逐rank相同，JSON实际shape已核实。FK20K仅0.1%的差异且范围
重叠，不认为改善；FL没有收益。未继续正式四档或library/Torch/sanitizer完整
门禁，不把native hash一致称为完整验收。撤回全部五个source文件改动，回到
FJ2；保留artifact和`return_cta_grouping_{fj2,fk,fl}_ep16_t*_quick.log`。
这次结果不支持仅改变Return CTA/context排布；未测的三类角色全CTA共享也
不能据此称为优化。后续先转向训练checkpoint复制，而不是继续叠加排布复杂度。

### FM：SavedX改由FC1保存（训练改善，inference持平）

基于已验收FJ2，Pull只向物理X写一次，每个Mblock唯一的FC1 Ntile0复用
ready A stage向canonical SavedX做uint4 scatter。已有WGMMA wait0及两个
CTA的empty arrivals保护共享stage，既有FC1完成链涵盖checkpoint owner。
删除Pull训练fanout wrapper，不增加buffer、barrier、metadata或调参分支；
G/U、Backward及数值顺序不变。不是将SavedX改成ring，也不是TMA checkpoint。

EP8/EP16完整训练四档、20K hotspot、短档OOM/retry/live tapes、inference
四档及edges/tiny ring均通过；96个native梯度hash等于FJ2。rank0/15的
T64小ring及T2048 prefill热点训练memcheck均0 errors，Forward专项racecheck
均0 hazards/0 warnings。详见[训练记录FM](static_forward_v4_training_record.md)。
不将Forward过滤的racecheck称为完整Backward clean。

冻结`build/static_v4_math_x_checkpoint_fm/`，native SHA256
`482f535e85ea00ad89cb859b47345f092f9582309d34bd74331fc2d1b8816120`，library
`44820160cb808ed7aaff376b244214ea989f48ab91438444a7784b842450892a`。
1620MHz、B512/D8、Qdispatch8/Qreturn16/L32、10warmup/20samples/batch20，
正式inference graph rank-max p50（us）：

|拓扑|T/rank|FM|同期冻结V3|相对V3降低|
|---|---:|---:|---:|---:|
|单机EP8|1|100.962|未重测|—|
|单机EP8|64|135.310|未重测|—|
|单机EP8|8192|1253.600|1488.020|15.75%|
|单机EP8|20480|2945.430|3300.250|10.75%|
|双机EP16|1|268.531|未重测|—|
|双机EP16|64|317.726|未重测|—|
|双机EP16|8192|2899.290|3360.410|13.72%|
|双机EP16|20480|6882.180|7098.420|3.05%|

EP8在第二节点独占计时，第一节点同时构建FM库/运行Torch正确性；计时节点
没有CPU构建。EP16在全部门禁结束后两节点独占计时，无并发构建/GPU任务。
20:17负载下两节点全部GPU SM/graphics1620MHz、memory3201MHz；memory只作
观测，不声称锁定。八档JSON实际shape均核实，96个Forward hash逐rank等于
FJ2，pool/注册bytes完全不变。V3 graph两个kernel、ring一个，此表是端到端，
不是纯GEMM。日志`math_x_checkpoint_final_{v3,fm}_{remote_ep8,ep16}_t*_formal.log`。

Inference相对FJ2持平，不把上述相对V3的既有收益归因于checkpoint迁移。
20K训练main相邻/反向对照：EP8改善2.80%/2.88%，EP16改善3.59%/3.13%；
后者反向中间有一个FN批次。FN仅将SavedX store换成.cs，没有稳定额外收益，
已撤回；所有证据及训练相对V3仍落后的结果见训练记录。EP16 20K inference
仍只领先V3约3%，不关闭“明显领先”及整体训练性能目标。
