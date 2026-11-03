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

## EP8 实测结果（2026-11-03）

本轮仅比较当前 WeftMoE 的 BF16 与 FP8，不代表 Megatron/DeepEP 或完整 RL 吞吐。
硬件：单机 8× NVIDIA L20X（SM90，132 SM，约 140 GiB/卡），CUDA 12.9，PyTorch 2.9.1+cu129，驱动 570.148.08。
算子源码：`24a20996be2c38191cef0d6cd41945f694c2d413`，使用三个对应的 `configs/profiles/<model>/`；
同一规模的 BF16/FP8 使用同一二进制，不开 autotune，不混用不同构建。
原始数据、二进制 hash、GPU 采样和完整复现脚本见 [本轮测量归档](https://github.com/yulangz/WeftMoE/tree/8d6301202b9fe0f3c927bf4d6570a4339db8aecc/measurements/ep8-20261103)。

### PF 训练

- E=256；Qwen top-k=8，Flash top-k=6；每 rank 名义 8K/20K tokens，长度 std=10%，路由 bias std=0.5，seed=123。
- ring tokens=512，depth=8；使用异步训练 API。FP8 源端在线量化计入耗时，W/Wᵀ 预量化在计时外，FP32 累加，dW 为 BF16。
- 每种模式按 BF16→FP8→FP8→BF16 顺序测量。每轮完整预热路由，再做 10 次 warmup、30 次采样；各 rank 的路由 hash 逐轮一致。
- 表中值为两轮合计 60 个“每样本最慢 rank”耗时的 p50；括号为两轮各自 p50 的范围。负的延迟变化表示 FP8 更快。
- FWD、BWD、完整 cycle 分别测量。FWD 到缓存可用，BWD 到梯度可用；不能相加独立 p50 来代替 cycle。

完整 FWD+BWD cycle（ms）：

| 规模 | tokens/rank | BF16（轮次范围） | FP8（轮次范围） | FP8 延迟变化 |
| --- | ---: | ---: | ---: | ---: |
| qwen35b | 8192 | 6.565（6.528–6.647） | 6.791（6.787–6.791） | +3.4% |
| qwen35b | 20480 | 15.596（15.560–15.628） | 15.559（15.558–15.593） | -0.2% |
| qwen122b | 8192 | 12.282（12.255–12.340） | 12.217（12.215–12.257） | -0.5% |
| qwen122b | 20480 | 37.043（37.005–37.064） | 28.494（28.464–28.542） | -23.1% |
| flash | 8192 | 19.130（18.926–19.130） | 17.608（17.580–17.608） | -8.0% |
| flash | 20480 | 45.904（45.863–45.992） | 42.027（41.988–42.070） | -8.4% |

单独前向/反向（ms）：

| 规模 | tokens/rank | BF16 FWD | FP8 FWD | BF16 BWD | FP8 BWD |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen35b | 8192 | 2.853 | 2.864 | 3.788 | 3.982 |
| qwen35b | 20480 | 6.960 | 6.515 | 8.661 | 9.261 |
| qwen122b | 8192 | 4.649 | 4.385 | 7.742 | 7.895 |
| qwen122b | 20480 | 15.049 | 9.634 | 22.059 | 18.936 |
| flash | 8192 | 6.724 | 5.511 | 12.419 | 12.342 |
| flash | 20480 | 15.576 | 12.853 | 30.105 | 29.199 |

Qwen35B 8K 的 FP8 完整 cycle 慢 3.4%，Qwen35B 20K 和 Qwen122B 8K 基本持平；
Qwen122B 20K 降低延迟 23.1%，Flash 两组降低约 8%。FP8 尚未实现所有规模的训练收益。

### DT decode

EP8、E=256，64/512 tokens/rank，固定 M64。每组分别使用路由 seed=1300/1700；每个 seed 捕获 100 次调用，
测量 11 轮并丢弃第一轮，轮次内交替 BF16/FP8/预量化 FP8 的执行顺序。下表为合并两个 seed 后的 p50（µs），
括号为两个 seed 各自 p50 的范围；路由静态回放，权重和图均已准备。

| 规模 | tokens/rank | BF16（轮次范围） | FP8 在线量化（轮次范围） | 预量化 FP8 输入（轮次范围） | 在线 FP8 延迟变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen35b | 64 | 103.57（103.47–103.71） | 83.31（83.16–83.51） | 80.67（80.43–80.93） | -19.6% |
| qwen35b | 512 | 214.62（214.35–214.89） | 218.84（218.04–219.59） | 217.75（217.01–218.49） | +2.0% |
| qwen122b | 64 | 210.40（210.34–210.42） | 141.77（141.42–142.26） | 136.99（136.62–137.48） | -32.6% |
| qwen122b | 512 | 422.98（422.82–423.32） | 321.94（320.21–324.21） | 320.65（318.57–322.89） | -23.9% |
| flash | 64 | 441.06（441.02–441.11） | 278.82（278.79–279.06） | 272.28（272.14–272.43） | -36.8% |
| flash | 512 | 756.51（755.27–757.88） | 531.02（530.65–531.48） | 528.33（527.95–528.98） | -29.8% |

预量化输入列排除了上游量化成本；它只衡量输入已经是 FP8 的场景。DT 收益不代表 PF 训练收益。
PF 计时前通过三个规模的 EP8 数值回归（E=256、top-k=2、名义 65 tokens/rank）；这不是 8K/20K 计时输入的逐项验收。
DT 两个 seed 均通过各 rank 的 FP8/BF16 全局 diff < 5e-3 检查（最大 0.002847），
在线量化和预量化输入的输出逐位一致。

测量期间每卡功耗上限为 700 W，未锁定时钟。500 ms GPU 采样记录的温度为 28–55°C、SM 时钟为 1350–1620 MHz，
峰值功耗约 694 W；Flash BF16 完整 cycle 和独立 BWD 出现短暂降频，Qwen 完整 cycle 采样均为 1620 MHz。
因此 Flash 对照包含默认功耗管理下的频率差异，不能将全部收益归因为相同频率下的算术吞吐。
遥测包含进程启动和退出，不能精确归因到单个计时样本。

复现 PF 的一个轮次：

```bash
export WEFTMOE_TENSOR_EXTENSION=/path/to/model-profile/_runtime.so
export OMP_NUM_THREADS=1
unset WEFTMOE_AUTOTUNE_TABLE
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/torchrun_training.py --backend pf --model qwen122b \
  --tokens 8192 20480 --warmups 10 --samples 30 --stream-ordered \
  --prime-trace --ring-tokens 512 --ring-depth 8 --seed 123
# FP8 轮次追加 --fp8 --weight-grad-dtype bf16；单独前向/反向追加对应的 --forward-only/--backward-only。
```

复现 Qwen122B DT 的一个 seed（扩展路径同上）：

```bash
TEST_HIDDEN=3072 TEST_INTERMEDIATE=1024 BENCH_EXPERTS=256 BENCH_TOPK=8 \
BENCH_SIZES="64 512" BENCH_REPEATS=100 BENCH_SWEEPS=11 BENCH_SEED=1300 \
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/benchmark_dt_fp8.py
```

另以 `BENCH_SEED=1700` 重复；其它规模使用表中 H/I/top-k 及对应构建。
