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
