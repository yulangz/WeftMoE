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
