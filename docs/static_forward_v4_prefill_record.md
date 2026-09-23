# V4 prefill：任务粒度、epilogue 与 Return 重叠

2026-09-23。本文接续单机优化记录与多机记录；EP8 指 **1×8 GPU**，EP16
指 **2×8 GPU**。性能测量统一核心 1620 MHz，H2048 / I512 / E64 / top-k4，
token 数均为每 rank。候选通过数值校验前不计优化收益。

## 实现上的几个区别

### FC1 gate/up 不是 router gate

对一条送往 expert e 的路由，计算合同为：

```text
G = BF16(X × W_gate[e])
U = BF16(X × W_up[e])
S = BF16(SiLU(G.float()) × U.float() × route_weight)
C = BF16(S × W2[e])
Y = BF16(按 (expert, slot) 固定顺序做 FP32 contribution 求和)
```

G/U 都是 I 维向量，route_weight 才是 router 为这条 top-k 路由给的标量。
不能把 route_weight 移到 FC2 之后，否则 BF16 舍入合同发生变化。

V3 分别持有 gate/up 两组 accumulator；V4 将 W1 列打包成
`[gate0..7, up0..7, gate8..15, up8..15, ...]`，一次 GEMM 产生两者。
这不是省掉一半投影计算。基准 V4 的 FC1 N64 只产出 32 列 S，I512 因此
需要 16 个 N-tile；V3 的 N128 分别计算 gate/up，产出 128 列 S，只需 4 个。
FC2 的基准配置则是 V4 N128/claim1，对照 V3 N256/claim2。
这同时改变了 GEMM 形状、任务选择频率、barrier 频率与 epilogue 粒度，
不能把两者的性能差全部归结为“静态 vs 动态调度”。

### 节点内只有 Dispatch 改成了 destination-pull

原始 V4 路径（`2e1899d`）：

以下通信方向在本轮优化后保持不变。

```text
源 GPU A 的 symmetric X[token]
  └─ expert GPU B 的通信 warp 主动拉取 → B 的 canonical[expert,row]
       └─ FC1 → SwiGLU → FC2
            └─ B 主动写 → A 的 contribution[token,top-k slot,:]
                 └─ A 固定顺序 Reduce
```

每个 homogeneous CTA 有 256 个 math 线程、32 个 TMA producer 线程、96 个
通信线程。132 CTA × 3 个通信 warp 静态跨步分配 canonical 行；每个 warp
只有 lane 0 发 TMA。完整输入行落到 destination 本 GPU 后，发布 device-scope
arrival；M128 的全部有效行就绪后计算可以启动，不需要动态任务队列/CAS。

“分散 contribution 写回”指 **不同行** 可能去不同 GPU、不同 token/slot，
不代表行内未合并：既有 FC2 epilogue 已按 8 lanes × 16 bytes 合并为连续
128-byte 写。每条路由写独占 slot，不是对最终 Y 做浮点原子累加。

多机不是跨机 RDMA pull：先由源 GPU 用同 rail 的 GIN Put，把 X、route 和
gate 复制到各远端节点的 rail GPU，再由该节点的 expert GPU 做 NVLink pull。
例如 node0/GPU3 的 token 在 node1/GPU1 上计算：

```text
node0/GPU3 X → GIN Put → node1/GPU3 输入副本
  → GPU1 NVLink pull → FC1/FC2
  → GPU3 Return TX → GIN Put → node0/GPU3 Return RX → Reduce
```

TX/RX 是不同 arena，防止 NIC 读尚未完成的 TX 被接收数据覆盖。跨机 Return
按目的节点的 expert 区间紧凑排列，不是传回整个容量上限。

### 跨机 Return 目前没有与 GEMM 流水重叠

基准实现先等本节点所有 FC2/TX 写入完成，再由固定 context owner 的 math
线程 0 发 256 KiB 分块 GIN Put。每个 context 分别做 Put fence/rail barrier，
节点内 barrier 导入远端可见性，随后才 Reduce。因此 **epilogue 正在写 TX**
不等于 **NIC 正在发送 TX**。

可以保持静态任务划分，实现以下流水，而不先改 Return/Reduce：

1. 通信/control 角色中固定每个 rail/context 的 sender，独立于 math 进度。
2. 完整 chunk 的所有写入 GPU、所有 FC2 N-tile 都完成并发布 system-release
   后，owner acquire readiness 即可发送该 chunk，math 继续后续 wave。
3. 保留终端 network drain/BeforeCombine；仍在所有 Return 到齐后统一 Reduce。

一个适合现有布局的具体切法是：**Return chunk 不跨 expert 边界**，单块不超过
现有 256 KiB。这样一个 chunk 的生产者只涉及一个 expert GPU；仍须等待该 GPU
上覆盖它的全部 Mblock/N-tile。TX 中某个 source 的连续行，在 canonical 的
source-rank round-robin 顺序中不连续，不能拿一个 Mblock ready 代表整个 chunk。
可利用 source counts 计算首末 source slot 的 canonical 行号，检查对应的 Mblock
ready 区间；完成计数经 GPU-local acquire 链汇合后，再一次 system-release 发布。
这不要求搬动 TX/RX 数据或修改 Reduce 的地址映射。

sender 可静态保留少量通信 warp，Dispatch 的行条带排除这些 warp，由其余通信
warp 完整覆盖。这样 sender 等 chunk ready 时不持有尚未履行的输入搬运义务，
所有 math CTA 又保持驻留，进度证明比在一个阻塞循环里混合 Dispatch/Return 简单。
还需实测 GIN sender 的寄存器需求及分块粒度；这里是后续方案，不是已实现功能。

不能用“某个 CTA 算完”替代 chunk ready，也不能用 context 0 的 fence 覆盖
别的 QP。需验证 sender 等待不会阻塞它尚未完成的 Dispatch 任务，并沿用
完整 invocation arena 的 epoch 复用边界。实际收益还受 chunk 就绪顺序、
NIC/显存带宽争用制约，不能把整段 Return 时间都宣称可消除。

已有 EP16/20K eager 观测：V4 BeforeCombine 约 3.737 ms、Reduce 约 0.406 ms。
这支持先做 Return/FC1/FC2 重叠。V3 对应 Return/FC2 交叠约 0.353 ms，并非
完全隐藏；其 Reduce span 约 2.97 ms 含等 Return 的时间，不是纯求和成本。
各 span 存在交叠，不能相加当作总延迟。
这些是本轮改动前冻结 binary 的 eager 数据，已收入结果 JSON 的
`prior_eager_diagnostics`，不是新 kernel 的分段计时。

## 候选与取舍

首轮筛选不是最终交错 A/B，以下只记录实验事实。

| 候选 | 单机 EP8 8K / 20K | 结论 |
|---|---|---|
| FC1 STSM 重排后 uint4 写 S | 1603.71→1594.51 / 4751.86→4727.45 us | 首轮 hash 通过；收益仅约 0.5–0.6%，但后续 N32 消融失败，不采用 N32 staging |
| 上述 FC1 + N128/N256/K128 全局替换 | 1511.25 / 4627.50 us | 64-token 达 89.83 us（基准 73.93），不能全局替换 decode 配置 |
| 每个 FC2 行片段发一次 bulk TMA | 相对 FC1 候选 1593.04→1583.51 / 4743.69→4855.25 us | hash 通过，但 20K 回退 2.35%；撤销，不保留 |
| 本地二维 TMA + 静态整行 Return relay | hash 门禁失败，不计性能收益 | 改用 B staging 后 8K hash 及 PyTorch 8K/20K 检查通过，但 20K native hash 仍失败（非微小舍入差）；候选撤销，未迁入生产路径 |

### 为什么不能直接照搬 V3 的 shared-memory epilogue

V3 的 pair join 阻止 cluster peer 在另一 CTA 的 epilogue 结束前进入下一任务。
V4 的静态 peer 独立推进：WGMMA 的 empty barrier 只保护输入读取，不保护随后
把 A arena 当作输出 staging 的写入。较快的 peer 可能提前 multicast 下一任务
的 A，覆盖仍被 epilogue 使用的 staging。

V4 的静态任务始终共享 A、不共享 B。因此单 tile epilogue 可复用已退休的 B
arena，由本 CTA 的 task barrier 约束 producer；多 tile claim 继续使用独立的
output arena。这避免恢复每任务 pair join，也不额外增加单 tile 的 shared memory。

首轮 FC1 候选的 hash 通过不证明复用 A 安全；最终实现改用 B。另一次保持窄 N
的独立构建，在 20K 严格 hash 门禁失败，因而撤掉 N32 SwiGLU staging，窄入口
恢复原 BF16 转换与直接写法；完整 N64 atom 的宽 prefill 路径单独验证。
这里定位的是
独立可证的 staging 生命周期风险，并未证明它是 relay 候选剩余错误的唯一原因。

## 保留的实现

- FC1：宽 prefill entry 的完整 M128、N64 SwiGLU 输出用 STSM 重排，再用 uint4
  连续写 S；窄入口的 N32 输出和 compact tail 保留直接写，不支付 staging barrier。
  保持投影与输出的 BF16 舍入。
- FC1/FC2 单 tile epilogue 复用 CTA 私有 B arena；不恢复动态队列或每任务 pair join。
- 新配置 `configs/static_inference_prefill_20260923/` 在同一个 binary 内保留两种
  GEMM entry。少于 8192 tokens/rank 使用原 N64/N128、K128、3/3 stages；
  从 8192 起使用 N128/N256、K128、2/2 stages。两者均 M128、claim1、32-Mblock wave。
  FC1/FC2 每 Mblock 的 N-tile 分别从 16/16 变成 8/8；V3 对照是 4/8 且 claim2，
  所以即使新 prefill entry 也不与 V3 的任务粒度完全相同。
- 两个 entry 共用 W1/W2 数据，只增加 weight TMA descriptor。计算操作数共享内存
  都为 192 KiB，宽 entry 的 barrier 更少；沿用原分配上限，不增加驻留 CTA 数。
- 选择仅依赖 invocation 的 source token 数，不读回 GPU route counts，不增加图节点。
  各 rank 可独立选择不同 entry。其他已有 tuning 默认 `static_v4_prefill_min_tokens=0`，
  不强行覆盖其它模型的配置；历史冻结 header 不修改。
- FC2 仍直接 push contribution/Return TX，跨机协议与 Reduce 时序不变。

### 验证口径

`tests/torchrun_static_inference.py --tokens 1 64 8192 20480 --edge-cases` 检查
eager/graph 的 PyTorch 参考、64/8192 混合 rank（不同 entry）、空输入 eager、
重复/热点 expert、四轮输入/routes/gates 修改、双层图共享 workspace、容量拒绝后恢复。
空输入 **graph** 尚受现有零容量 pool 限制，本轮未扩展这个 API。

容量错误测试修正了既有拓扑假设：GIN 汇总所有 rank 的终态错误；单机 P2P
只在发生溢出的 destination 报错。测试先归约拒绝数量，再让所有 rank 一起复用
workspace，而不是让非溢出 rank 提前抛 Python 断言、把其余 rank 留在清理中。
这不是改变生产错误协议，也不记作性能收益。

性能使用 graph CUDA event 的 rank-max p50，而不是把 device-clock span 相加。
单机 V4 control 的 `complete_clock` 可能先于 math 的 Reduce 尾部记录；V4 的
旧 Return span 字段也未完整赋值。它们不宜当作独立通信时间或完整 kernel 时间。

## 最终测量（1620 MHz）

代码里程碑：

- `036281d`：GEMM 形状参数化、prefill 双入口、FC1 epilogue、CTA 私有 B staging，
  以及混合 entry 的实机检查。
- `77aa384`：根据严格 hash 消融撤回 N32 staging，恢复窄入口直接写。

原始 rank-max samples、所有 rank 输出 hash、binary SHA256 和时钟汇总存于
[`static_inference_prefill_1620_20260923.json`](../benchmarks/static_inference_prefill_1620_20260923.json)。
最终共 **36 sessions、432 个 rank hash**，全部匹配冻结参考；**34248 个 GPU
时钟采样**的 SM/graphics 均为 1620 MHz。每 rank 的 graph 均为 1 kernel，
0 memcpy、0 event-record/wait。

主矩阵每个 case 按旧 V4 → 新 V4 → 新 V4 → 旧 V4 交错。每 session
10 warmup + 20 measured samples，每 sample 连续 replay 20 次；对每个 sample
先取 rank 最大值，每版本合并 40 个样本后取 nearest-rank p50。单位 us。
V3 列来自同日已提交的 1620 MHz 冻结矩阵，**不是本轮重新交错的 V3**：
单机来源为 `static_inference_single_node_1620_20260923.json`，多机来源为
`static_inference_compare_1620_20260923.json` 的 2×8 case。V3 使用独立的
`configs/static_inference_v3_20260923/`，不是在新 binary 中简单 disable V4。
各路径均 inference-only，未拿训练路径当 V3 对照。

| 拓扑 | tokens/rank | 旧 V4 | 新 V4 | 延迟变化 | 同日冻结 V3 | 新 V4 相对 V3 |
|---|---:|---:|---:|---:|---:|---:|
| 单机 EP8 | 1 | 64.18 | 64.50 | +0.49% | 299.78 | -78.48% |
| 单机 EP8 | 64 | 74.02 | 73.80 | -0.30% | 368.44 | -79.97% |
| 单机 EP8 | 8192 | 1601.93 | 1504.50 | -6.08% | 1487.58 | +1.14% |
| 单机 EP8 | 20480 | 4740.10 | 4607.09 | -2.81% | 3289.72 | +40.05% |
| 双机 EP16 | 1 | 170.51 | 169.99 | -0.30% | 680.49 | -75.02% |
| 双机 EP16 | 64 | 189.46 | 188.90 | -0.29% | 824.51 | -77.09% |
| 双机 EP16 | 8192 | 3860.00 | 3725.31 | -3.49% | 3355.23 | +11.03% |
| 双机 EP16 | 20480 | 9535.05 | 9146.10 | -4.08% | 7105.66 | +28.72% |

结论：1/64 tokens 基本持平。8K/20K 延迟分别下降 **6.08%/2.81%（单机 EP8）**、
**3.49%/4.08%（双机 EP16）**。单机 8K 接近冻结 V3，但单机 20K 仍慢 40.05%；
双机 8K/20K 仍慢 11.03%/28.72%。这两项改动没有消除长序列差距。
本轮有限参数与 epilogue 实验也不能证明“任何 GEMM 调参都无法继续改善”。

### 分步收益与证据边界

A = 交错反转顺序的 A/B；B = 相邻构建消融；C = 多项组合、不可逐项分摊。

| 拓扑 / tokens | 旧 V4 | 仅安全 B staging、窄 N 直接写 | 再加宽 prefill + FC1 合并写 |
|---|---:|---:|---:|
| 单机 EP8 / 8192 | 1601.93 | 1598.34 | 1504.50 |
| 单机 EP8 / 20480 | 4740.10 | 4740.19 | 4607.09 |
| 双机 EP16 / 8192 | 3860.00 | 3855.33 | 3725.31 |
| 双机 EP16 / 20480 | 9535.05 | 9539.38 | 9146.10 |

中间列每 case 仅一次独立 session / 20 samples，属于 **B 级消融**；它相对旧
V4 的变化约 −0.22% 至 +0.05%，不声称有性能收益。B staging 是生命周期安全性
修正，不把它计成加速项。

最后一列相对旧 V4 的组合收益有上述 **A 级**证据；但“加宽 GEMM”和“完整
N64 atom 的 FC1 合并写回”在这个入口中一起启用，没有同形状的独立关闭实验，
因此不能把组合收益任意分给两者（单项归因属于 **C 级边界**）。
旧的 N32 FC1 候选、逐行 FC2 TMA 和整行 relay 的数值不并入最终表。
尤其 relay 候选含被撤回的 N32 epilogue，不能据此断言 relay 协议本身不可行。

### 最终门禁

最终 library 在单机 EP8 / 双机 EP16 完成 **576 次 rank/case** PyTorch
数值与 workspace 复用检查，最大绝对误差 7.6294e-6。日志为：

- `/tmp/weft-v4-prefill-safe-parity-ep8.log`
- `/tmp/weft-v4-prefill-safe-parity-ep16-{local,remote}.log`

V3 training 的 8-rank Forward/Backward、graph、offload 回归也已通过
（`/tmp/weft-v4-prefill-training.log`，使用窄入口收尾修正前的 library；
该修正仅作用于静态 inference FC1，不涉及 V3 分支）。

Python 调参测试为 37 passed、18 subtests passed，另有 **7 个已有失败**。
在干净 `2e1899d` 快照上为 36 passed、相同 18 subtests passed、相同 7 失败，
主要涉及旧宏表、候选数和 shared-memory 预期；本轮新增的 companion 配置校验通过。
没有把旧失败默认为通过，也未扩展本轮范围去重写无关调参测试。

### 复现构建与运行

```bash
export NVCCFLAGS="-I$PWD/configs/static_inference_prefill_20260923 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1"
python3 build_native.py cuda-test --jobs 8 --output build/static_v4_prefill_safe/test_cuda_notify
python3 build_native.py lib --jobs 8 --output build/static_v4_prefill_safe/libweftmoe.so
python3 tools/build_tensor_runtime.py --library build/static_v4_prefill_safe/libweftmoe.so
```

单机性能入口是 `tests/run_local_activation_p2p.py --world 8 --top-k 4
--forward-benchmark --rounds 30`；双机是 `tests/run_multinode_nccl.py
--gpus-per-node 8 --scenario gin_forward_bench --enable-ib`。两者均显式设置：

```text
WEFTMOE_BENCH_EXPERTS=64
WEFTMOE_P2P_TOP_K=4
WEFTMOE_BENCH_HIDDEN=2048
WEFTMOE_BENCH_INTERMEDIATE=512
WEFTMOE_P2P_BENCH_TOKENS=<1|64|8192|20480>
WEFTMOE_FORWARD_BENCH_ROUNDS=30
WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY=1
WEFTMOE_FORWARD_GRAPH_BATCH=20
WEFTMOE_BENCH_SYNC_RANKS=1
```

都 unset `WEFTMOE_DISABLE_STATIC_FORWARD_V4`，两机各 8 张卡保持核心锁频
1620 MHz；本机网络环境使用 `NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0`。
数值门禁使用上述 torchrun 测试，extension 指向同一配置构建的 library。
协议仍不写死节点数，但本轮多机实测只有两节点，不宣称验证过更多机器。
