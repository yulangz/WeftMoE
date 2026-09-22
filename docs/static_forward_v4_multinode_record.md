# Static Forward V4 多机 inference 记录

日期：2026-09-22。协议与首版边界见
[`static_forward_v4_design.md`](static_forward_v4_design.md#60-复审进度容量与首版实现边界)。

## 实现门禁

- 两台 Hopper 节点；EP8 = 2×4 GPU，EP16 = 2×8 GPU。
- H=2048，I=512，64 experts，top-k=4，BF16。
- 1、64、8K、20K 均为每 rank 的 token 数；8K=8192，20K=20480。
- 两种 EP 的四个规模均通过 PyTorch 独立参考的 eager 和 graph 校验。
  参考保留 FC1 BF16、SwiGLU/gate FP32 后 BF16、FC2 BF16、按 `(expert,slot)`
  FP32 combine 的数值合同。最大绝对误差 7.63e-6，阈值 rtol=0.02/atol=2e-5。
- 两种 EP 均检查不等长/空输入 rank、所有 route 重复指向最后一个 expert、四轮
  X/routes/gates 变化、eager 与 graph 交错、双层 graph 共用通信 workspace，
  以及容量不足时集体拒绝、随后复用同一 workspace 恢复计算。
- EP8 20K 的所有输出哈希与单机完全相同；EP16 1/64 的前 8 个 rank 也与
  对应单机输出完全相同。fixture 的输入和 expert 权重不依赖 EP 划分。
- 原型图中的 GIN reuse event 已移出 serialized graph hot path；每 rank
  一个 kernel node，零 memcpy/event node。eager 保留跨 stream 的 event 保护。

测试入口为 `tests/torchrun_static_inference.py --tokens 1 64 8192 20480
--edge-cases`，由两节点 torchrun 启动，各节点 `--nproc-per-node=4/8`。
环境需设置 `PYTHONPATH` 为 repo 根目录，`WEFTMOE_TENSOR_EXTENSION` 指向同一
native library 构建的 `_runtime*.so`，本机 RDMA 配置使用
`NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0`。

初版验证日志：`/tmp/weft-v4-parity-ep16-v3-{local,remote}.log`（四个规模通过，
随后空 gate 的旧接口检查报错）及 `/tmp/weft-v4-parity-ep8-v4-{local,remote}.log`
（零长度 gate 修正后全部通过）。零长度 gate 现在与零长度 route 一样是合法输入。

最终日志为 `/tmp/weft-v4-parity-ep{8,16}-final-{local,remote}.log`，两种 EP
分别完成 176/352 次 rank/case 校验，所有进程正常退出；容量拒绝后的恢复也已通过。
额外运行原有 `tests/torchrun_tensor_runtime.py`（单机 EP8、8 tokens、64 experts），
8 rank 的 Forward/Backward parity、双层 graph、销毁/offload 均通过；训练仍走 V3。
日志为 `/tmp/weft-v4-training-regression.log`。

## 测量合同

冻结的构建参数位于 `configs/static_inference_20260922/`，不依赖临时 JIT cache。
在 repo 根目录构建（使用 CUDA 12.9、CUTLASS、NCCL 2.31.2）：

```bash
export NVCCFLAGS="-I$PWD/configs/static_inference_20260922 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1"
python3 build_native.py cuda-test --jobs 8 --output build/static_v4_network/test_cuda_notify
python3 build_native.py lib --jobs 8 --output build/static_v4_network/libweftmoe.so
python3 tools/build_tensor_runtime.py --library build/static_v4_network/libweftmoe.so
```

默认产品构建/其他 H/I 应使用自己的 tuning，不把这个测量点冒充所有模型的最优配置。

裸 CUDA Graph 使用 CUDA event，10 个 warmup sample + 20 个 measured sample，
每 sample 连续 replay 20 次；每轮先取所有 rank 的最大耗时，再取 p50。
包括 route exchange、输入 staging、GIN Dispatch、节点内 pull、计算、Return、
combine 与清理。不包括进程/NCCL 初始化和 graph capture。

`tests/run_multinode_nccl.py --scenario gin_forward_bench --gpus-per-node=4/8`
使用以下环境：

```text
WEFTMOE_BENCH_EXPERTS=64
WEFTMOE_P2P_TOP_K=4
WEFTMOE_P2P_BENCH_TOKENS=<1|64|8192|20480>
WEFTMOE_FORWARD_BENCH_ROUNDS=30
WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY=1
WEFTMOE_FORWARD_GRAPH_BATCH=20
```

初版 EP16/64 的 p50=191.47 us；初测 EP8/20K=9206.34 us 只用了两个样本，
不能作为正式矩阵值。

最终构建的完整矩阵如下，单位 us，均为上述 graph rank-max p50：

| 拓扑 | 1 token | 64 tokens | 8192 tokens | 20480 tokens |
|---|---:|---:|---:|---:|
| 单机 EP8（最终回归） | 58.12 | 68.16 | 1480.64 | 4583.34 |
| 两机 EP8（2×4） | 145.61 | 186.95 | 3752.12 | 9201.61 |
| 两机 EP16（2×8） | 154.48 | 173.31 | 3703.26 | 9191.01 |

8 个 case 的前 8 rank 哈希全部与对应单机基准完全一致，所有 rank 的图都只有
1 kernel、0 memcpy、0 event。原始日志为
`/tmp/weft-v4-network-final-gpus{4,8}-t{1,64,8192,20480}.log`。
这些数值不是与旧 V3 多机路径的交错 A/B，不宣称未经测量的 V3 相对加速。
12 个 case 的完整 rank/sample、图节点数、输出 hash 及 native binary SHA256
已保存到 [`../benchmarks/static_inference_20260922.json`](../benchmarks/static_inference_20260922.json)。

## 首版后的同步精简

初版每轮有三次跨机 barrier：Dispatch 发布、Return 发布、Combine 后复用。
最终版只保留前两次，末尾只做本 GPU grid join。其复用证明依赖不相交的 arena，
不依赖两个节点或 rank 的相对速度：

| arena | 上一轮最后的使用 | 下一轮首次覆盖前的边界 |
|---|---|---|
| source replicas | 节点内 pull 与 Return extent 读取 | 上一轮 Return barrier 已保证所有 peer 完成 Dispatch 消费并进入 Return |
| Return TX | 本节点 NIC 读取 | 下一轮 Dispatch barrier 保证所有接收 rank 已结束上一轮 Return/Combine |
| Return RX | 本 rank Combine | 下一轮 Dispatch barrier 必须等本 rank 结束 Combine 并进入下一轮，之后才可能有新 Return |

因此快 rank 可以先准备下一轮的 source replicas，但不能越过 Dispatch barrier
提前覆盖仍被消费的 Return TX/RX。每个 context 仍独立使用带 Put fence 的 rail
barrier，节点内 barrier 汇合所有 rail 的可见性；没有用 context 0 signal 代替
其他 context 的 fence。

GIN context 数按对称 capacity 推导：`min(8, ceil(max_route_return_bytes / 256KiB))`。
本矩阵的 1/64/8192/20480 分别为 1/4/8/8 个固定 CTA owner。不再为仅需这些
owner 的 inference pool 分配 V3 的 133 个 context。训练和显式 V3 fallback
仍沿用自己的 context 配置。这里使用 capacity 而非本 rank 的实际 token 数，
避免不等长输入让不同 rank 进入不同数量的 barrier。

EP16/64 从初版 191.47 us 降至 173.31 us，约改善 9.5%；这是末尾 barrier 与
context 精简的相邻构建 bundle，不将收益进一步拆成未经独立 A/B 的单项。

## 尚未覆盖与已知限制

- runtime 循环和 offset 支持任意节点数，没有两机特判；仅两机有实测证据。
- 跨机 Dispatch/Return 与 GEMM 之间有阶段边界；节点内 pull 与 GEMM 重叠。
- source X 当前向每个远端节点发送一次，即使该节点没有被路由到。后续可引入
  node-deduplicated compact staging；不能把该优化尚未实现写成已有收益。
- 完整 invocation arena 按公开容量预留；没有 ring credit、动态任务领取或丢 token。
- 扩展回归中 59 个 Python 测试和 86 个 subtest 通过，另有 8 项旧 autotune
  断言失败（候选数量、共享内存常数、宏枚举、incumbent 名称）。相关工具、配置和
  测试与基线 `8f3fa57` 没有差异；本轮不通过改写这些旧断言掩盖问题。
  去掉这两个已知过期的测试文件后，针对 unified resource、runtime settings、
  buffer validation、launcher security 的复测为 30 passed / 62 subtests passed。
