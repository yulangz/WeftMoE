# Static Forward V4 多机 inference 记录

日期：2026-09-22；2026-09-23 补充优化分步记录与 V3 对照。协议与首版边界见
[`static_forward_v4_design.md`](static_forward_v4_design.md#60-复审进度容量与首版实现边界)。

后续的 prefill 双入口、epilogue 消融及跨机 Return/GEMM 重叠分析，见
[`static_forward_v4_prefill_record.md`](static_forward_v4_prefill_record.md)。

拓扑更正：用户要求的主矩阵是 **单机 EP8（1×8）和双机 EP16（2×8）**。
此前误将 EP8 测成两机 2×4；本篇保留该组数据作为额外实验，不代表单机 EP8。
单机 EP8 的 1620 MHz 锁频补测见
[`static_forward_v4_optimize_record.md`](static_forward_v4_optimize_record.md)。

本篇冻结版本结论（两节点全部 GPU 核心锁频 1620 MHz）：对照 V3 inference fallback
及已有多机调优参数，V4 在 1/64 tokens 加速 **3.86–4.35×**，但 8K/20K 延迟
增加 **15.1–34.2%**。功能与数值门禁已通过，**多机 prefill 性能目标尚未达成**，
不能称为全面优于 V3。完整锁频矩阵和比较边界见下文。

## 优化路径与逐步收益

沿用单机记录的证据等级：A = 同条件反转顺序的交错 A/B，B = 相邻构建消融，
C = 多项同时变化的里程碑 bundle。所有 token 数均为每 rank；延迟单位 us。
未独立测量的改动不分摊收益，也不把单机收益搬到多机。

| 步骤 / 提交 | 改动及解决的问题 | 多机收益与证据边界 |
|---|---|---|
| 0 / `b4538e0` | 复用单机的 FC2 连续 128-byte 写、32-Mblock 静态 wave、M128 readiness 直接索引 | 已包含在第一版多机中。单机 8K/20K 相对复审基线改善 24.9%/36.0%（B）；没有多机独立消融 |
| 1 / `d69f6b6` | 放弃 bounded ring 协议，改用完整 invocation arena；区分 NIC 本地读完、远端可见、consumer 用完 | 设计修正，无独立性能数值。避免 TX 等 credit 时阻塞同 owner 的 RX 进度，以及错误复用 TX buffer |
| 2 / `1154748` | 每个 rail 固定 CTA owner；原始 X 每远端节点一份，expert-sorted route/prefix 驱动节点内 pull；移除 V3 动态 claim/packet/fanout/credit 控制路径 | EP16/64 首个正式 graph 基线 **191.47**（C）。不是相对 V3 的加速比 |
| 3 / `1154748` | FC2 直接写源 rail 的独立 Return TX；按目的节点 expert 区间紧凑回传，combine 直接读 RX；共享 route geometry、pull、stable combine order | 减少额外 scatter 与控制元数据，已包含在 191.47 中；没有单项计时 |
| 4 / `1154748` | serialized graph 去掉 GIN reuse event，保留 eager 跨 stream 保护 | 原型 1 kernel + 1 event-wait 变为 **1 kernel、0 event/copy**。原型 EP16/1=168.90 仅 2 samples、batch=1，不与正式 batch=20 值计算收益 |
| 5 / `c3610bb` | Combine 后跨机 barrier 改为本 GPU grid join；GIN contexts 由统一 capacity 确定，并缩小实际分配 | EP16/64 **191.47 → 173.31**，−18.16 / **−9.5%**（C，相邻构建）。两项同批变化，不分别归因 |

步骤 2–4 属于一次多机大重构，没有保存每步独立可比较的 timing。
容量检查改为真实用户 buffer 大小、允许空 rank 的空 gate，以及 collective 拒绝后
恢复复用属于正确性修复，不计作性能优化。各规模最终数据见后文完整矩阵。

与旧路径的关键区别不是简单减少几个 CTA：静态 residue 分配替代动态 GEMM 任务领取，
full-invocation staging 替代依赖 credit 的逐包进度，返回数据直接落到 combine 可消费
的位置。V3 SPSC 已经有固定 TX/RX channel owner，不能把“去掉 GIN context 锁”重复
记成 V4 的新贡献；V4 进一步把通信收敛为 1–8 个 rail owner 的连续 Put 和阶段 fence。
代价是更大容量预留，以及跨机 Dispatch/Return 仍为显式阶段，未与 GEMM 流水重叠。
不能据此推断所有模型/路由都更快。

## 实现门禁

- 本篇的两机实验为额外 EP8（2×4 GPU）和主矩阵 EP16（2×8 GPU），不是单机 EP8。
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

## 2026-09-22 历史测量合同与矩阵

本节保留当日原始记录，不属于后文的 1620 MHz 锁频对照；不要跨表计算加速比。

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

当日最终构建的历史完整矩阵如下，单位 us，均为上述 graph rank-max p50：

| 拓扑 | 1 token | 64 tokens | 8192 tokens | 20480 tokens |
|---|---:|---:|---:|---:|
| 单机 EP8（最终回归） | 58.12 | 68.16 | 1480.64 | 4583.34 |
| 两机 EP8（2×4） | 145.61 | 186.95 | 3752.12 | 9201.61 |
| 两机 EP16（2×8） | 154.48 | 173.31 | 3703.26 | 9191.01 |

8 个 case 的前 8 rank 哈希全部与对应单机基准完全一致，所有 rank 的图都只有
1 kernel、0 memcpy、0 event。原始日志为
`/tmp/weft-v4-network-final-gpus{4,8}-t{1,64,8192,20480}.log`。
这些 2026-09-22 数值不是与旧 V3 多机路径的交错 A/B。
12 个 case 的完整 rank/sample、图节点数、输出 hash 及 native binary SHA256
已保存到 [`../benchmarks/static_inference_20260922.json`](../benchmarks/static_inference_20260922.json)。

## 2026-09-23：V3 / V4 同条件对照（核心锁频 1620 MHz）

对照的是当前提交 `c3610bb` 的 V3 inference fallback，不是训练模式，也不是用
此前其他 E/top-k 的历史时间作分母。V3 设置 `WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`，
使用仓库已有的 `communication_spsc_20260916/qwen.hpp` 多机参数；构建入口冻结在
`configs/static_inference_v3_20260923/`。V4 沿用上述最终构建和参数。
V3 参数曾在 H2048/I512 的多机路径验证，但这次没有针对 E64/K4 和四个长度重新
autotune；因此结论适用于这组公开配置，不宣称战胜所有可能的 V3 调参结果。

V3 的 FC1/FC2 N tile 为 128/256、K64、stages 4/2、FC2 claim=2，通信域为
16 sender + 116 receiver/local；V4 为 N64/N128、K128、stages 3/3，静态 wave=32。
二者均为 132 CTA、384 threads、cluster=2。比较保留各自参数，是实现与配置的
整体对照，不是仅替换 scheduler 的单变量实验。

每个 case 按 **V3 → V4 → V4 → V3** 启动四个独立 session，各自 10 个 warmup
和 20 个 measured sample，每 sample 20 次 graph replay。打开
`WEFTMOE_BENCH_SYNC_RANKS=1`，在每个 sample 的计时区间外同步所有 rank；
先逐 sample 取 rank-max，再合并同一版本两个 session 的 40 个 sample 取 p50
（排序后第 20 个）。这是独立进程的反序复测，不冒充同一进程逐 iteration 切换。
正式计时不并发编译或其他 GPU 测试。按要求，两节点全部 16 GPU 使用
`nvidia-smi -lgc 1620,1620` 锁定核心时钟，并设置 `-ac 3201,1620`；显存 application
clock 保持 3201 MHz，未另行锁显存频率，driver=570.148.08。以 200 ms 间隔记录
两节点各 GPU 的实际 SM/graphics 时钟及利用率，检查负载期间是否维持 1620 MHz，
而非只检查 application-clock 配置。
此前 application-clock=1980 MHz、未锁频的预跑不纳入本节结果，也不与本轮样本合并。
前文 2026-09-22 的里程碑数字保留为历史记录，不拿它们与本轮锁频结果计算优化收益。

V3 graph 包含 Notify + Forward，共 12 nodes：2 kernel、3 memcpy、6 event-record、
1 event-wait。V4 将 route/notify 融入 Forward，为 1 kernel、0 memcpy/event。
双方都测完整 graph，不人为从 V3 耗时中减去辅助节点；也不把 graph p50 与
最后一次 replay 的 device-kernel span 混用。固定输入/路由/权重在所有 session
中复用，输出以 2026-09-22 已通过 PyTorch 参考的 V4 全 rank 哈希为门禁。
每个 case 独立建 pool，token capacity 按该 case 配置；不是让同一个 20K capacity
pool 跑所有 decode 长度的测试。数值边界用例已验证，但本表不代表热点路由的性能。

### 锁频结果

单位 us；加速比 = V3/V4，延迟变化 = `(V4/V3 - 1) × 100%`，正数表示回退。

| 拓扑 | tokens/rank | V3 p50 | V4 p50 | 加速比 | 延迟变化 |
|---|---:|---:|---:|---:|---:|
| EP8（2×4） | 1 | 621.90 | 160.96 | 3.86× | -74.1% |
| EP8（2×4） | 64 | 793.43 | 199.48 | 3.98× | -74.9% |
| EP8（2×4） | 8192 | 3359.00 | 3939.05 | 0.85× | +17.3% |
| EP8（2×4） | 20480 | 7260.68 | 9614.13 | 0.76× | +32.4% |
| EP16（2×8） | 1 | 680.49 | 169.23 | 4.02× | -75.1% |
| EP16（2×8） | 64 | 824.51 | 189.52 | 4.35× | -77.0% |
| EP16（2×8） | 8192 | 3355.23 | 3860.82 | 0.87× | +15.1% |
| EP16（2×8） | 20480 | 7105.66 | 9537.37 | 0.75× | +34.2% |

32 个 session 全部完成，384 次 rank/output hash 检查全部与已验证参考一致。
03:56:26–04:08:06 UTC 的核心时钟采样共 55,928 条，全部为 SM=graphics=1620 MHz；
每张 GPU 均有负载样本，负载样本的显存频率均为 3201 MHz。测试后两节点仍保持
1620 MHz 锁频，没有恢复到 1980 MHz。

完整的各 session p50、rank/sample、图节点数、hash、构建 SHA256，以及逐 GPU
时钟核验汇总保存于
[`../benchmarks/static_inference_compare_1620_20260923.json`](../benchmarks/static_inference_compare_1620_20260923.json)。
原始日志为 `/tmp/weft-1620-abba-20260923-ep{8,16}-t{1,64,8192,20480}-{trial}-{v3,v4}.log`；
时钟日志为 `/tmp/weft-1620-clocks-{local,remote}-20260923.csv`。
该 JSON 另存了 2026-09-22 初版 EP16/64 的原始样本，明确放在不属于锁频对照的
historical 字段下，以保留 191.47 → 173.31 的里程碑证据。

### 结论与下一步

- decode 的静态任务、fused Notify、graph 节点精简确实带来整体收益，但这不是
  将整个 3.86–4.35× 分摊到单项优化的依据。
- prefill 回退在反序复测和 1620 MHz 锁频下仍存在。代码可确认 V4 的跨机 Dispatch、
  GEMM、Return 有阶段边界，而 V3 保留流式通信；同时两者 N tile、stages 和任务
  粒度不同。**没有单变量消融，不能将回退完全归因于同步或缺少通信重叠。**
- 后续应先比较 prefill 适用的 GEMM tile/wave 参数，再隔离测量跨机通信与 GEMM
  重叠的收益；保持静态任务与 full-arena 的进度保证，不重新引入已经否决的
  TX/RX 同 owner 等 credit 环。node-deduplicated X staging 仍是待实现项。
- 本次只补测和记录，生产调度未改，也未增加按长度自动切换 V3/V4 的策略。
  上述 prefill 回退仍未修复。

### 复现

V3 构建命令如下，V4 使用前文的构建命令：

```bash
NVCCFLAGS="-I$PWD/configs/static_inference_v3_20260923 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1" \
  python3 build_native.py cuda-test --jobs 8 \
  --output build/static_inference_v3_20260923/test_cuda_notify
```

运行时使用前文六项 benchmark 环境变量，额外设置 `WEFTMOE_BENCH_SYNC_RANKS=1`。
V3 设置 `WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`；V4 **unset** 此变量。
launcher 示例为 V3 / EP16；测试 V4 时替换 binary，EP8 则设置 `--gpus-per-node 4`：

```bash
test_binary="$PWD/build/static_inference_v3_20260923/test_cuda_notify"
python3 tests/run_multinode_nccl.py \
  --binary "$test_binary" --remote-binary "$test_binary" \
  --remote root@10.10.19.233 --port 2222 --gpus-per-node 8 \
  --scenario gin_forward_bench --socket-ifname eth0 --enable-ib --timeout 240
```

本机另需 `LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64` 和
`NCCL_IB_GID_INDEX=3`。原 V4 冻结参数的 `WEFTMOE_FWD_RDMA_CTAS=0` 关闭了旧多机
入口，不能只在那个 binary 上设置 disable 开关就当作有效的 V3 多机对照。

## 首版后的同步精简（2026-09-22 历史证据）

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
