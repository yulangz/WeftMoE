# Benchmark

## 规模与口径

预设位于 [models.py](../benchmarks/models.py)，各工具均可用 `--model` 选择，或覆盖四个维度。
测试的是单层 routed MoE，不加载完整模型。

| `--model` | 模型规模 | H | I | E | top-k |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen35b` | Qwen3.5-35B-A3B | 2048 | 512 | 256 | 8 |
| `qwen122b` | Qwen3.5-122B-A10B | 3072 | 1024 | 256 | 8 |
| `flash` | DeepSeek V4 Flash | 4096 | 2048 | 256 | 6 |

训练 benchmark 的 8K/20K 是**每 rank** 名义 8192/20480 tokens，不是整机总量。
默认长度按 10% 标准差正态扰动、截断 ±3σ；每次生成新路由，共享 expert-logit bias std=0.5。
这是合成热点负载，不等于真实模型采集的路由。`--route-distribution balanced` 改为均衡周期，
`--route-trace 'path/rank{rank}.routes.pt'` 可回放真实路由及相邻的 `.gates.pt`。

计时包含通信、GEMM、当次权重重排及临时/训练缓存申请；排除首次 pool/IPC/GIN 初始化、
输入生成和编译。FWD 保存 X/GU，BWD 使用计时外准备的新鲜 FWD tape，分别测到缓存/梯度就绪。
比较使用逐样本最慢 rank 的 p50；独立 FWD/BWD 的 p50 不能相加冒充实测完整 cycle。
不包含 router projection/top-k、shared expert、attention、optimizer 或 DDP 梯度归约。

## 运行

先按 [使用文档](usage.md) 构建对应 shape，并设置扩展路径。以下在仓库根目录运行，
不要让编译与正式 GPU 计时重叠；比较双方使用相同 workload 参数和 seed。

```bash
unset WEFTMOE_AUTOTUNE_TABLE  # 使用显式 ring 参数；若用表，容量必须精确匹配。
export WEFTMOE_PF_MIN_RING_DEPTH=2
export WEFTMOE_PF_CREDIT_WINDOWS=2

# 带训练缓存的 FWD。
torchrun --standalone --nproc-per-node=8 benchmarks/torchrun_training.py \
  --backend pf --model qwen35b --tokens 8192 20480 --forward-only \
  --stream-ordered --ring-tokens 2048 --ring-depth 2 \
  --prime-trace --warmups 4 --samples 20 --sample-pause-ms 50
```

将 `--forward-only` 换为 `--backward-only` 测独立 BWD；去掉两者测完整 cycle。
换为 `--backend megatron --dispatcher flex --deepep-num-sms 20` 测基线：
TE grouped experts + DeepEP，开启 permute/activation fusion，使用 `moe_act` 重算缓存口径。
该基线需额外安装 Megatron-Core、Transformer Engine、DeepEP。

日志中的 `TRAINING_CYCLE_JSON`：FWD 读取 `forward_ready_slowest_rank_p50_us`，
BWD 读取 `backward_ready_slowest_rank_p50_us`，完整 cycle 读取 `wall_slowest_rank_p50_us`。
比较时检查 `workload_sha256_per_rank` 相同，且 Megatron 计时期间没有新编译图。

```bash
# 无缓存 prefill；固定权重已准备，不把重排计入每次推理。
torchrun --standalone --nproc-per-node=8 benchmarks/torchrun_inference.py \
  --backend pf --model qwen35b --tokens 8192 20480 --prepared-weights

# 小 batch，与 SGLang 的 DP=EP 路径同口径比较；改 dt 为 pf 可测 PF。
torchrun --standalone --nproc-per-node=8 benchmarks/torchrun_sglang_decode.py \
  --model qwen35b --weft-backend dt --dp-attention \
  --tokens 1 8 64 128 512 --fresh-routes --route-bias-std 0.5 --samples 20
```

SGLang 测试含输入 staging、dispatch、GEMM、return/reduce，不含 router、shared expert、
权重重排和 graph 准备。默认不加 `--dp-attention` 时是 TP 复制输入，tokens 口径不同，不能混比。
Flash 等较大 shape 的 BF16 检查可显式使用 `--atol 1e-4`，仍保留相对误差检查。
MegaMoE 单机对照用 `torchrun_inference.py --backend megamoe --deepgemm-package /path/to/DeepGEMM`；
其它 shape、trace 和采样参数需与 PF 一致。

双机 PF/Megatron 在**两台机器同时**启动同一命令，将 torchrun 前缀替换为：

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29531 \
  benchmarks/torchrun_training.py --backend pf --model qwen35b \
  --tokens 8192 20480 --forward-only --stream-ordered \
  --ring-tokens 2048 --ring-depth 2 --prime-trace --warmups 4 --samples 20
```

两侧分别设置 `NODE_RANK=0/1`，`MASTER_ADDR` 指向节点 0；库、环境和参数相同。
设置适合机器的 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 等网络参数，不照搬其他集群的网卡名。
EP8 指单机八卡，EP16 指双机十六卡；固定每 rank tokens 时，EP16 的全局工作量翻倍。

## FP8 结果汇总与归档

原始 JSON、候选补丁、SASS 和一次性实验脚本保存在
[归档提交 f40a921](https://github.com/yulangz/WeftMoE/tree/f40a921ea75cbd0ac6b99a2fb3c0d1b0c05015e1)
（分支 `archive/fp8-benchmarks-f40a921`），不随当前源码分发。
归档提交用于定位原始证据；各次实验的实际源码／二进制 hash 以报告中的记录为准。
不要将历史候选补丁应用到当前源码。

| 对照 | 数据范围 | 结果 | 实现提交 |
| --- | --- | --- | --- |
| 122B AReaL 权重更新 | 单机 DP8/EP8，32×16，16K，第 2–10 step | baseline 22.10 s；BF16 WeftMoE 25.54 s；FP8 WeftMoE 25.48 s | f40a921 |
| BF16 直接 TMA 权重读取 | 122B/EP8，每 rank | 去掉 18 GiB FC1 重排副本，PF/DT 直接读取原始布局 | f40a921 |
| 固定 M64 PF/DT 一致性 | 三个预设 shape，BF16/FP8 | 含训练缓存的 PF 前向与 DT 严格逐位回归通过 | 6a2f213 |

权重更新不含 decode graph 重捕获，数据量缩减不等于跨卡带宽同比加速。
RL 的采样长度不同，以上不是完整训练吞吐对照，也不承诺所有 shape 的 FP8 都快于 BF16。
当前可复用入口为 `benchmarks/torchrun_training.py`、`benchmarks/torchrun_sglang_decode.py`
和 `tools/autotune_dt_fp8.py`。

PF 训练历史结果（完整 FWD+BWD cycle，单位 ms；名义 tokens 为每 rank，权重量化在计时外）：

| EP | 规模 | tokens | BF16 | FP8 | FP8 延迟变化 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | qwen35b | 8192 | 6.575 | 6.653 | +1.2% |
| 8 | qwen35b | 20480 | 15.516 | 15.559 | +0.3% |
| 8 | qwen122b | 8192 | 12.266 | 12.067 | -1.6% |
| 8 | qwen122b | 20480 | 36.954 | 28.463 | -23.0% |
| 8 | flash | 8192 | 19.037 | 17.581 | -7.6% |
| 8 | flash | 20480 | 45.816 | 42.394 | -7.5% |
| 16 | qwen35b | 8192 | 13.416 | 18.161 | +35.4% |
| 16 | qwen35b | 20480 | 31.426 | 42.928 | +36.6% |
| 16 | qwen122b | 8192 | 24.316 | 25.314 | +4.1% |
| 16 | qwen122b | 20480 | 56.896 | 60.130 | +5.7% |
| 16 | flash | 8192 | 29.297 | 30.439 | +3.9% |
| 16 | flash | 20480 | 72.383 | 71.090 | -1.8% |

**训练结论：EP8 的 Q35 未超过 BF16；EP16 六组中五组更慢，只有 Flash 20K 约快 1.8%。**
这些历史数据不支持“所有规模 FP8 训练均优于 BF16”的结论，也不能用后面的 DT decode 收益替代。
EP8 来自 [2026-10-31 final-cycle 报告](https://github.com/yulangz/WeftMoE/blob/f40a921ea75cbd0ac6b99a2fb3c0d1b0c05015e1/docs/benchmarks/fp8_profile_opt_20261031.json)，
EP16 来自 [2026-10-29 源端 FP8 通信报告](https://github.com/yulangz/WeftMoE/blob/f40a921ea75cbd0ac6b99a2fb3c0d1b0c05015e1/docs/benchmarks/fp8_network_source_20261029.json)。
两表均为逐样本最慢 rank 的墙钟时间中位数；EP16 已使用 FP8 X/dY 通信和 FP8 X/AB 缓存。
报告归档在 `f40a921`，实际实验源码与二进制身份见各报告；这些不是本次修复后的重新测量。

固定 M64 的 DT 历史回放结果（EP8、256 experts、静态路由、单位 µs；实现提交 `6a2f213`）：

| 规模 | 每 rank tokens | BF16 | FP8 在线量化 | 预量化 FP8 输入 |
| --- | ---: | ---: | ---: | ---: |
| qwen35b | 64 | 104.66 | 82.83 | 80.37 |
| qwen35b | 512 | 215.80 | 218.29 | 216.93 |
| qwen122b | 64 | 210.24 | 141.66 | 137.06 |
| qwen122b | 512 | 423.40 | 326.64 | 324.18 |
| flash | 64 | 442.56 | 276.83 | 270.33 |
| flash | 512 | 790.67 | 536.30 | 534.19 |

此表只统计已准备权重和图的 decode 回放；预量化输入列不含上游量化成本，不能直接代表端到端收益。
