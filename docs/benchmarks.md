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

计时包含通信、GEMM 及临时/训练缓存申请和释放；排除首次 pool/IPC/GIN 初始化、
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

## 框架基线

训练使用 Megatron-Core 基线，decode 使用 SGLang 基线。两者均测试单层 routed MoE，
不是完整模型或 RL step；下文表格中的 **WeftMoE BF16** 是精度对照，不是这两个框架的基线。

| 场景 | 基线实现 | 精度和通信 | 计时边界 |
| --- | --- | --- | --- |
| PF 训练 | Megatron-Core `MoEFlexTokenDispatcher` + Transformer Engine grouped experts | BF16，DeepEP normal dispatch/combine，20 SM；TP=ETP=1，EP=world size | 路由格式转换、dispatch、FC1/SwiGLU/FC2、combine；BWD 包含 dX、dprob、dW |
| DT decode | SGLang `FusedMoE`，`moe_runner_backend=triton`、`moe_a2a_backend=none` | BF16，DP=EP=world size；输入 all-gather、输出 reduce-scatter | DP 通信、两次 GEMM 和激活；WeftMoE 对照包含输入 staging 和结果返回 |

SGLang 这里测的是 Triton FusedMoE 加 DP 通信路径，不是 SGLang 的 DeepEP 或原生 FP8 路径。
两种框架基线均排除 router projection/top-k、shared expert、权重初始化和编译。
Megatron 开启 permute/activation fusion，并使用 `moe_act` selective recompute；
默认 dW 为 BF16，不包含 optimizer 和 DDP 梯度归约。若测 FP32 `main_grad` 累加，
双方均需追加 `--accumulate-main-grad`，单独报告。

### Megatron 复现与结果读取

与下方 PF 表一致的单机 EP8 workload，先测完整 cycle：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  benchmarks/torchrun_training.py --backend megatron --model qwen35b \
  --dispatcher flex --deepep-num-sms 20 \
  --moe-permute-fusion --bias-activation-fusion \
  --tokens 8192 20480 --warmups 10 --samples 30 --prime-trace \
  --seed 123 --token-std-fraction 0.1 --route-bias-std 0.5
```

分别追加 `--forward-only`、`--backward-only` 测独立 FWD/BWD；将 `--model` 换为
`qwen122b`、`flash` 覆盖另两种规模。PF 对照使用相同 workload、seed、warmup/sample 数，
BF16 使用 `--backend pf --stream-ordered --ring-tokens 512 --ring-depth 8`，
FP8 再追加 `--fp8 --weight-grad-dtype bf16`，并选择对应规模的扩展。

读取 `TRAINING_CYCLE_JSON` 中的 `wall_slowest_rank_p50_us`（cycle）、
`forward_ready_slowest_rank_p50_us`（FWD）、`backward_ready_slowest_rank_p50_us`（BWD）。
每组必须核对 `workload_sha256_per_rank` 一致、`compiled_graphs_during_measurement` 全为 0；
保存 `implementation` 中的 Megatron-Core、Transformer Engine、DeepEP 和 NCCL 版本。
加速比按同轮次 `Megatron BF16 / WeftMoE` 计算；合并多轮时先合并逐样本耗时，再取 p50。

### SGLang 复现与结果读取

以下命令在同一进程中交错测量 SGLang BF16 和 WeftMoE DT BF16，输入、权重、路由共享：

```bash
WEFTMOE_FP8=0 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc-per-node=8 benchmarks/torchrun_sglang_decode.py \
  --model qwen35b --weft-backend dt --dp-attention \
  --tokens 64 512 --fresh-routes --route-bias-std 0.5 \
  --seed 1300 --samples 20 --replays 50
```

将 `--model` 换为 `qwen122b`、`flash` 并切换对应扩展；Flash BF16 校验追加 `--atol 1e-4`。
再用 `--seed 1700` 重复，合并两组样本。
读取 `SGLANG_DECODE_COMPARE.rank_max_p50_us`：`baseline` 为 SGLang，`dt` 为 WeftMoE，
加速比为 `baseline / dt`。结果应记录 `tokens_scope=dp_local`、`dp_size=8`、
`baseline_communication=all_gather+reduce_scatter`，以及 `routing_sha256`、软件版本和扩展 hash。
每种实现均先捕获 CUDA graph，计时交错 replay；取逐样本最慢 rank 后的 p50。

此脚本的显式 DT 对照路径目前为 BF16；设置 `WEFTMOE_FP8=1` 不会把它变成 FP8 三方比较。
它与下方 `benchmark_dt_fp8.py` 的静态均匀路由、输入 staging 和采样方式不同，
不能把这里单独测得的 SGLang 耗时拼入下方 FP8 表计算加速比。
FP8 对 SGLang 的数值结果需在相同输入、路由和计时边界下联合测量。

## EP8 实测结果（2026-11-05）

单机 8× NVIDIA L20X（SM90，132 SM，约 140 GiB/卡）；三个规模均在空闲节点串行测量。
驱动 570.148.08，CUDA 12.9，实际加载 NCCL 2.31.2；每卡功耗上限 700 W。
源码基线：`12977054368b0fe8978b4515e7e74ecd95bc27b4`；实际源文件清单 SHA256：`ade0890051bfdb1b1d87eeb158fca9716637b2b7edb0ac66a9007a62bc605939`。
测量使用该 commit 的干净源码快照；每个规模单独构建，BF16/FP8 共用该规模的二进制。
构建启用 `WEFTMOE_FORWARD_USE_GENERATED_TUNING=1`，选择对应 `configs/profiles/<model>/`；不启用 autotune。
性能概览见 [README](../README.md#性能)，以下保留完整计时口径与结果。
完整逐 rank 样本、workload hash 和源码文件 hash 保留在本地测量包中，原始大日志不入库。
未锁定时钟；500 ms GPU 遥测的 SM 时钟为 1410–1620 MHz，温度 29–53°C，峰值功耗约 692 W。
遥测包含启动和退出阶段，不能精确对应单个计时样本；结果包含默认功耗管理的影响。

依赖版本：torch `2.9.1+cu129`，sglang `0.5.10.post1`，flashinfer-python `0.6.18`，megatron-core `0.19.0`，transformer-engine `2.14.1+27de67e1`，deep-ep `1.2.1+567632d`，triton `3.5.1`。

### PF 训练：Megatron BF16 与 WeftMoE

E=256；Qwen top-k=8，Flash top-k=6；每 rank 名义 8K/20K tokens，长度 std=10%，路由 bias std=0.5，seed=123。
每个规模、模式使用 Megatron→WeftMoE BF16→WeftMoE FP8→FP8→BF16→Megatron 的顺序，
每轮完整预热路由后做 10 次 warmup、30 次采样。所有实现的逐 rank workload hash 相同，Megatron 计时期间没有新编译图。
PF ring tokens=512、depth=8，使用异步训练 API；FP8 源端量化计时，权重预量化不计时，dW 为 BF16。
数据是两轮合计 60 个逐样本最慢 rank 耗时的 p50；加速比为 Megatron BF16 / WeftMoE，低于 1× 表示较慢。
FWD、BWD、cycle 分别测量，不把独立 p50 相加作为 cycle。BF16/Megatron 数值预检使用每 rank 名义 16 tokens；FP8 使用现有正确性测试（名义 65 tokens、top-k=2）。这些预检不是对大规模计时输入的逐项验收。

完整 FWD+BWD cycle（ms）：

| 规模 | tokens/rank | Megatron BF16 | WeftMoE BF16 | WeftMoE FP8 | BF16 加速比 | FP8 加速比 | FP8 相对 WeftMoE BF16 延迟变化 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35b | 8192 | 11.367 | 6.525 | 6.703 | 1.74× | 1.70× | +2.7% |
| qwen35b | 20480 | 23.209 | 15.559 | 15.532 | 1.49× | 1.49× | -0.2% |
| qwen122b | 8192 | 16.332 | 12.206 | 12.141 | 1.34× | 1.35× | -0.5% |
| qwen122b | 20480 | 37.084 | 36.907 | 28.364 | 1.00× | 1.31× | -23.1% |
| flash | 8192 | 23.576 | 19.122 | 17.664 | 1.23× | 1.33× | -7.6% |
| flash | 20480 | 55.199 | 46.066 | 42.175 | 1.20× | 1.31× | -8.4% |

相对 WeftMoE BF16 的延迟变化为正表示 FP8 更慢；FP8 并非在所有训练规模上都有收益。

独立前向/反向（ms）：

| 规模 | tokens/rank | 阶段 | Megatron BF16 | WeftMoE BF16 | WeftMoE FP8 |
| --- | --- | --- | --- | --- | --- |
| qwen35b | 8192 | FWD | 5.614 | 2.815 | 2.857 |
| qwen35b | 8192 | BWD | 5.555 | 3.780 | 3.976 |
| qwen35b | 20480 | FWD | 10.472 | 6.971 | 6.497 |
| qwen35b | 20480 | BWD | 12.277 | 8.684 | 9.236 |
| qwen122b | 8192 | FWD | 7.109 | 4.677 | 4.355 |
| qwen122b | 8192 | BWD | 9.147 | 7.667 | 7.801 |
| qwen122b | 20480 | FWD | 15.817 | 14.945 | 9.689 |
| qwen122b | 20480 | BWD | 21.441 | 21.992 | 18.799 |
| flash | 8192 | FWD | 9.429 | 6.686 | 5.450 |
| flash | 8192 | BWD | 14.338 | 12.439 | 12.249 |
| flash | 20480 | FWD | 21.599 | 15.505 | 12.789 |
| flash | 20480 | BWD | 33.625 | 30.227 | 29.449 |

### DT decode：SGLang BF16 与 WeftMoE BF16

使用上文 SGLang 对照脚本，DP=EP=8，64/512 tokens/rank，固定 M64；路由 bias std=0.5，每次样本更新输入和路由。
两个 seed=1300/1700，各 5 次预热、20 次样本、每样本 50 次 graph replay，同一进程交错执行两种实现。
下表合并 40 个逐样本最慢 rank 耗时取 p50（µs），括号为两个 seed 各自 p50 的范围；所有样本均通过脚本的 BF16 数值检查。

| 规模 | tokens/rank | SGLang BF16 | WeftMoE DT BF16 | 加速比 |
| --- | --- | --- | --- | --- |
| qwen35b | 64 | 175.42（172.71–177.65） | 120.16（109.69–123.45） | 1.46× |
| qwen35b | 512 | 486.10（484.25–489.66） | 234.93（223.50–237.79） | 2.07× |
| qwen122b | 64 | 364.12（364.00–364.27） | 214.55（214.25–214.93） | 1.70× |
| qwen122b | 512 | 894.50（891.30–895.75） | 482.43（481.04–484.00） | 1.85× |
| flash | 64 | 748.34（739.79–749.84） | 440.07（435.79–441.00） | 1.70× |
| flash | 512 | 1599.08（1581.23–1613.36） | 972.72（930.65–1004.64） | 1.64× |

### DT decode：WeftMoE BF16/FP8 精度对照

单独运行 `benchmark_dt_fp8.py`；两个 seed=1300/1700，静态均匀路由，每个 graph 捕获 100 次调用，
11 轮交替 replay 并丢弃首轮，合并 20 个逐样本最慢 rank 耗时取 p50（µs）。
FP8/BF16 全局 diff < 5e-3（各 rank 最大值 0.002847），在线量化与预量化输入输出逐位一致。

| 规模 | tokens/rank | WeftMoE BF16 | WeftMoE FP8 在线量化 | WeftMoE 预量化 FP8 输入 | 在线 FP8 延迟变化 |
| --- | --- | --- | --- | --- | --- |
| qwen35b | 64 | 103.32 | 82.92 | 80.32 | -19.7% |
| qwen35b | 512 | 213.42 | 218.15 | 216.86 | +2.2% |
| qwen122b | 64 | 210.32 | 142.63 | 137.69 | -32.2% |
| qwen122b | 512 | 422.13 | 320.83 | 318.87 | -24.0% |
| flash | 64 | 444.66 | 278.92 | 272.29 | -37.3% |
| flash | 512 | 792.87 | 529.93 | 527.15 | -33.2% |

预量化输入排除上游量化成本。这一表与 SGLang 表的路由、staging 和采样方法不同，不能交叉计算加速比。
DT decode 收益不代表 PF 训练或完整 RL 的收益。

复现 Qwen122B 的内部 DT 精度对照（设置对应扩展路径）：

```bash
TEST_HIDDEN=3072 TEST_INTERMEDIATE=1024 BENCH_EXPERTS=256 BENCH_TOPK=8 \
BENCH_SIZES="64 512" BENCH_REPEATS=100 BENCH_SWEEPS=11 BENCH_SEED=1300 \
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/benchmark_dt_fp8.py
```

再用 `BENCH_SEED=1700` 重复；其它规模使用表中 H/I/top-k 及对应构建。
