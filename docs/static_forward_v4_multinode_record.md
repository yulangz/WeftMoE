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
- EP8 已通过不等长/空输入 rank、所有 route 重复指向最后一个 expert、四轮
  X/routes/gates 变化、eager 与 graph 交错、双层 graph 共用通信 workspace。
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

## 测量合同

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
不能作为正式矩阵值。完整矩阵将在同一最终构建下重新采样。

## 尚未覆盖与已知限制

- runtime 循环和 offset 支持任意节点数，没有两机特判；仅两机有实测证据。
- 跨机 Dispatch/Return 与 GEMM 之间有阶段边界；节点内 pull 与 GEMM 重叠。
- source X 当前向每个远端节点发送一次，即使该节点没有被路由到。后续可引入
  node-deduplicated compact staging；不能把该优化尚未实现写成已有收益。
- 完整 invocation arena 按公开容量预留；没有 ring credit、动态任务领取或丢 token。
- 扩展回归中 59 个 Python 测试和 86 个 subtest 通过，另有 8 项旧 autotune
  断言失败（候选数量、共享内存常数、宏枚举、incumbent 名称）。相关工具、配置和
  测试与基线 `8f3fa57` 没有差异；本轮不通过改写这些旧断言掩盖问题。
