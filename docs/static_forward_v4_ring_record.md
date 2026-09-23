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
