# SM100 / B300 Benchmark

使用与 [SM90](benchmarks.md) 共享的三个单层 routed MoE 预设。H/I 分别为 hidden / intermediate，
E 为专家数。

| `--model` | 模型规模 | H | I | E | top-k |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen35b` | Qwen3.5-35B-A3B | 2048 | 512 | 256 | 8 |
| `qwen122b` | Qwen3.5-122B-A10B | 3072 | 1024 | 256 | 8 |
| `flash` | DeepSeek V4 Flash | 4096 | 2048 | 256 | 6 |

`sm100` 是 Blackwell benchmark 目录。B300 的 CUDA compute capability 是 10.3，
脚本按 CUDA runtime 返回的 SM 版本检查设备，支持设备名显示为 L20D 的环境。
该目录运行框架基线和 Blackwell BF16、NVFP4 融合实现。Hopper PF/DT 使用独立的 `sm_90a` kernel。
[NVIDIA B300 设备规格](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html)

## 运行

在仓库根目录执行。torch 和两个框架使用支持 SM 10.x 的 CUDA 构建，SGLang 的 FlashInfer
首次运行时可能编译 kernel 和下载 cubin。各节点使用相同版本、相同的每节点 GPU 数和 workload 参数。

```bash
# 单机 EP8，依次运行三个规模、两个精度及全部计时模式。
benchmarks/sm100/run_matrix.sh benchmarks/results/sm100/ep8

# 单项。训练默认测 FWD+BWD，单独测量可用 --forward-only 或 --backward-only。
benchmarks/sm100/run.sh megatron --model qwen35b --precision fp4
benchmarks/sm100/run.sh sglang --model flash --precision bf16
```

双机 EP16 在两台机器上使用同一条命令。各自设置 `NODE_RANK=0` 或 `1`，
`MASTER_ADDR` 填节点 0 的地址。网络参数沿用集群的 NCCL 配置。
矩阵从 `MASTER_PORT` 开始为每项使用独立端口，完整矩阵使用 24 个连续端口。
更多节点可调整 `NNODES`，每个节点的 rank 从 0 开始连续编号，总 EP 必须能整除 E。

```bash
NNODES=2 NODE_RANK="$NODE_RANK" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=29531 \
  benchmarks/sm100/run_matrix.sh benchmarks/results/sm100/ep16
```

`NPROC_PER_NODE` 默认为 8。矩阵命令可追加 `megatron` 或 `sglang` 只运行一个框架。追加 `weft` 会运行 WeftMoE 的 BF16 和 NVFP4 推理、训练前向、反向与完整 cycle，三种模型共用 30 个连续端口。
每个节点写独立日志，节点 0 的日志包含结果 JSON 和逐样本数据。
单项运行的 token buckets、采样参数和 seed 均可通过 `--help` 查看。

### Megatron flex / DeepEP

Megatron-Core 0.19.0 可使用带 `sm_103` 的 DeepEP v1.2.1。以下构建使用 CUDA 13.0、
当前 Python 环境中的 PyTorch 2.13.0+cu130 和 `nvidia-nvshmem-cu13`。
在 DeepEP v1.2.1 源码目录中执行。
[DeepEP v1.2.1 构建脚本](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/setup.py)

```bash
env -u NVSHMEM_DIR -u DISABLE_SM90_FEATURES \
  CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=10.3 \
  DISABLE_AGGRESSIVE_PTX_INSTRS=1 \
  uv build --wheel --no-build-isolation --python "$(command -v python3)" --out-dir dist
```

将 wheel 装入干净环境，或用 `uv pip install --no-deps --target <目录> <wheel>` 单独安装，
再将该目录放在 `PYTHONPATH` 最前面。当前镜像的旧扩展只有 SM90/SM90a，且存在
`deep_ep.utils` 导入冲突。

```bash
benchmarks/sm100/run.sh megatron --dispatcher flex --model qwen35b --precision bf16
MEGATRON_DISPATCHER=flex benchmarks/sm100/run_matrix.sh benchmarks/results/sm100-deepep/ep8 megatron
```

flex 基线使用 DeepEP normal dispatch/combine，默认 20 个 SM。BF16 和 NVFP4
使用相同的 BF16 通信与计时口径。双机在两端设置相同的 `MEGATRON_DISPATCHER=flex`，
并配置集群对应的 NVSHMEM bootstrap 网卡和 RoCE GID。

本环境 EP8 使用原版 v1.2.1 和 NVSHMEM 3.4.5。EP16 的 NVSHMEM 3.4.5 在
IBGDA 初始化时出现 `Unable to create ah`。EP16 数值校验使用 NVSHMEM 3.5.19，
并把 DeepEP 的设备符号声明和 RC 队列索引修复回移到 v1.2.1。
[NVSHMEM 地址句柄修复](https://github.com/NVIDIA/nvshmem/issues/21) ·
[DeepEP PR 564](https://github.com/deepseek-ai/DeepEP/pull/564)

双机用 `qwen35b`、每 rank 名义 128 tokens 跑了 BF16 前向和反向数值校验。
相对 PyTorch 参考，输出与 dX 的逐 rank 最大相对 L2 误差均低于 0.0031。

复现 EP16 时，在 DeepEP v1.2.1 源码目录应用
[回移补丁](../benchmarks/sm100/deepep-v1.2.1-nvshmem35.patch)，然后沿用上面的构建命令。
若将 NVSHMEM 3.5.19 用 `uv pip install --no-deps --target "$NVSHMEM_PREFIX" nvidia-nvshmem-cu13==3.5.19`
装入独立目录，构建和运行时都要把 `$NVSHMEM_PREFIX` 放在系统 Python 包路径之前，
使 PyTorch 预加载对应版本的 NVSHMEM。运行时将新 DeepEP wheel 的目录也放在 `PYTHONPATH` 前面，
并将 `$NVSHMEM_PREFIX/nvidia/nvshmem/lib` 加到 `LD_LIBRARY_PATH` 前面。
本次双机使用 `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0`、`NVSHMEM_IB_GID_INDEX=3`，
`NVSHMEM_HCA_LIST` 选择 `mlx5_bond_0:1` 至 `mlx5_bond_7:1`。

## 计时口径

- **Megatron 训练**使用 Transformer Engine grouped experts、NCCL all-to-all、`moe_act` 重算，
  TP=ETP=1，EP=world size。FP4 使用 TE 的 NVFP4BlockScaling，BF16 主权重和 BF16 dispatch/combine，
  每次 FWD 都量化专家权重，专家 FWD/BWD 各自所需的量化和计算计入对应阶段。
  每 rank 名义 8192/20480 tokens，长度 std=10%，截断于 ±3σ，
  路由 bias std=0.5，seed=123。矩阵先完整预热同一 trace，再做 10 次 warmup 和 30 次采样。
  `TRAINING_CYCLE_JSON` 中分别读取 `wall_slowest_rank_p50_us`、`forward_ready_slowest_rank_p50_us`
  和 `backward_ready_slowest_rank_p50_us`，分别来自独立的 cycle、FWD 和 BWD 运行。
  [Megatron NVFP4 训练文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/low_precision_training.html)
- **SGLang decode**使用 FlashInfer TRT-LLM routed MoE，DP=EP=world size，BF16 all-gather 和
  reduce-scatter。FP4 在计时外把合成 BF16 权重加载为 NVFP4，激活按 token 在线量化。
  每 rank 固定 64/512 tokens，每样本更新输入及路由，bias std=0.5，seed=1300。
  5 次 warmup、20 次采样，每样本 50 次 CUDA Graph replay。读取 `SGLANG_DECODE_JSON` 的 `rank_max_p50_us`。
  [SGLang 量化文档](https://docs.sglang.io/docs/advanced_features/quantization)

两套测试均取逐样本最慢 rank 的 p50，包含通信和专家计算。router projection/top-k、
shared experts、attention、optimizer、权重初始化和首次编译在计时外。
固定每 rank tokens 时，EP16 的全局 token 数是 EP8 的两倍。

## B300 实测（2026-11-19）

每节点 8 张 B300，CUDA runtime 报告 SM 10.3，GPU 间为 NVLink，单卡功耗上限 1100 W。
驱动 580.105.08，torch 2.13.0+cu130，运行时 NCCL 2.29.7。
使用 Megatron-Core 0.19.0、Transformer Engine 2.20.2、SGLang 0.5.20 和 FlashInfer 0.6.18。
SGLang 实测加载同版本的 FlashInfer `0.6.18+cu130` 预编译模块。
[FlashInfer CUDA 13.0 JIT cache](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/)
EP8 的两个框架各用一台同规格节点，EP16 使用同一对节点，通过 NCCL IB 和 RoCE bond 网卡通信。
原始日志与逐样本数据保存在
`benchmarks/results/sm100-b300/ep8/` 和 `benchmarks/results/sm100-b300/ep16/`。

### Megatron NCCL all-to-all 训练（ms）

| EP | 规模 | tokens/rank | BF16 cycle | FP4 cycle | BF16 FWD | FP4 FWD | BF16 BWD | FP4 BWD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | qwen35b | 8192 | 7.951 | 15.064 | 3.944 | 8.132 | 4.326 | 7.359 |
| 8 | qwen35b | 20480 | 12.174 | 16.146 | 6.145 | 9.466 | 6.293 | 8.443 |
| 8 | qwen122b | 8192 | 10.147 | 15.357 | 4.782 | 9.340 | 5.565 | 7.734 |
| 8 | qwen122b | 20480 | 21.706 | 23.280 | 10.051 | 11.835 | 11.944 | 11.670 |
| 8 | flash | 8192 | 13.094 | 15.526 | 5.685 | 8.824 | 7.690 | 7.715 |
| 8 | flash | 20480 | 29.539 | 26.748 | 12.347 | 13.084 | 17.305 | 13.908 |
| 16 | qwen35b | 8192 | 17.740 | 19.058 | 9.018 | 10.189 | 8.995 | 9.472 |
| 16 | qwen35b | 20480 | 42.715 | 45.188 | 21.252 | 22.980 | 21.675 | 22.661 |
| 16 | qwen122b | 8192 | 27.444 | 27.855 | 13.129 | 14.325 | 14.207 | 14.302 |
| 16 | qwen122b | 20480 | 67.289 | 67.214 | 32.418 | 33.726 | 35.317 | 34.271 |
| 16 | flash | 8192 | 30.687 | 30.003 | 14.427 | 15.242 | 16.908 | 15.272 |
| 16 | flash | 20480 | 76.957 | 72.673 | 35.380 | 34.803 | 41.913 | 38.087 |

### SGLang decode（µs）

| EP / DP | 规模 | tokens/rank | BF16 | FP4 |
| ---: | --- | ---: | ---: | ---: |
| 8 | qwen35b | 64 | 98.35 | 91.50 |
| 8 | qwen35b | 512 | 193.77 | 208.48 |
| 8 | qwen122b | 64 | 164.58 | 136.21 |
| 8 | qwen122b | 512 | 318.60 | 279.36 |
| 8 | flash | 64 | 308.03 | 230.33 |
| 8 | flash | 512 | 586.56 | 431.05 |
| 16 | qwen35b | 64 | 306.78 | 309.17 |
| 16 | qwen35b | 512 | 539.68 | 580.93 |
| 16 | qwen122b | 64 | 345.36 | 334.31 |
| 16 | qwen122b | 512 | 717.82 | 720.42 |
| 16 | flash | 64 | 430.80 | 372.13 |
| 16 | flash | 512 | 1015.53 | 919.15 |

同一拓扑、规模和 token bucket 的 BF16 与 FP4 路由摘要一致。Megatron 的三个独立计时模式也使用相同路由，所有计时段新增编译图数均为 0。表中 p50 由原始逐 rank 样本重算。CSV 汇总位于 `benchmarks/results/sm100-b300/baseline.csv`。

### Megatron flex / DeepEP 训练（ms）

EP8 使用 DeepEP `1.2.1+9af0e0d` 和 NVSHMEM 3.4.5。EP16 使用回移补丁构建的 DeepEP `1.2.1+9c9b89a` 和 NVSHMEM 3.5.19。两组均使用 Megatron-Core 0.19.0 和 20 SM 的 DeepEP normal dispatch/combine。

| EP | 规模 | tokens/rank | BF16 cycle | FP4 cycle | BF16 FWD | FP4 FWD | BF16 BWD | FP4 BWD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | qwen35b | 8192 | 12.931 | 19.044 | 7.343 | 11.462 | 6.116 | 9.071 |
| 8 | qwen35b | 20480 | 26.706 | 31.412 | 13.330 | 17.837 | 13.269 | 13.559 |
| 8 | qwen122b | 8192 | 15.209 | 20.630 | 7.568 | 11.871 | 7.913 | 9.477 |
| 8 | qwen122b | 20480 | 34.144 | 36.790 | 15.720 | 19.541 | 18.659 | 17.722 |
| 8 | flash | 8192 | 18.731 | 22.124 | 8.664 | 13.161 | 10.236 | 9.755 |
| 8 | flash | 20480 | 43.325 | 41.112 | 19.399 | 20.936 | 24.342 | 20.310 |
| 16 | qwen35b | 8192 | 12.952 | 16.071 | 7.325 | 9.770 | 6.170 | 7.446 |
| 16 | qwen35b | 20480 | 28.212 | 31.615 | 14.178 | 17.094 | 14.267 | 14.488 |
| 16 | qwen122b | 8192 | 15.991 | 18.037 | 7.660 | 10.483 | 8.230 | 7.889 |
| 16 | qwen122b | 20480 | 36.655 | 36.302 | 17.068 | 18.353 | 19.778 | 18.278 |
| 16 | flash | 8192 | 18.776 | 18.372 | 8.486 | 9.895 | 10.612 | 8.683 |
| 16 | flash | 20480 | 44.572 | 38.418 | 18.844 | 17.933 | 25.765 | 20.391 |

两种 dispatcher 的同规模、同精度及同 token bucket workload 哈希相同，所有计时段新增编译图数均为 0。按完整 cycle p50 比较，EP8 的 12 组均由 NCCL all-to-all 取得更低延迟，EP16 的 12 组均由 DeepEP flex 取得更低延迟。

CSV 汇总与逐样本日志位于 `benchmarks/results/sm100-b300/deepep/`。

## Blackwell 单机 BF16

下文 BF16 和 NVFP4 的历史性能矩阵均使用动态 M 和可缩小的专家尾块。这也是当前默认路径。历史固定 M 的性能对照见[动态与固定 M 实测](#动态与固定-m-实测)。

配套的 DeepGEMM 提交为 `cd3f06dfe368d17004279727f99e30ba18dad27a`（分支 `feat/blackwell-moe`，基于 `5081f851`）。该版本增加了原始 FC1 权重的 5 维 TMA 描述符和可选训练 checkpoint。单机 `TensorRuntime` 按 CUDA compute capability 选择实现。权重保持 `[E_local, 2I, H]` 和 `[E_local, H, I]`。输入按 (expert, slot) 稳定排序，NVLink dispatch、两层 TCGEN05 GEMM、SwiGLU 和 combine 在一个 persistent kernel 中执行。训练时固定每个源 rank 的 expert 行序号，在同一 kernel 内写出 BF16 X 和 gate/up。BWD 通过对称内存读取 dY、返回 dX 和 dGate，按固定行序用 grouped GEMM 计算 dW。

本机已在仓库的 `build/sm100-env` 建立独立 Python 环境。使用下面的命令复现。

```bash
export PYTHON="$PWD/build/sm100-env/bin/python"
benchmarks/sm100/run.sh weft --mode decode --model qwen35b
benchmarks/sm100/run.sh weft --mode prefill --model flash
benchmarks/sm100/run.sh weft-training --model qwen35b --precision bf16 --samples 30 --backward-only
PYTHONPATH=. "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 tests/torchrun_sm100_invariant.py
```

在另一套已有 PyTorch、Triton、CUDA 13、GCC 13 和 elfutils 开发头文件的环境中，可从本工作区旁边的 DeepGEMM 源码单独安装。两个仓库的修改需要配套使用。

```bash
git -C ../DeepGEMM submodule update --init --recursive
CXX=g++-13 CXXFLAGS=-g0 DG_FORCE_BUILD=1 uv pip install --no-build-isolation --no-deps ../DeepGEMM
```

### EP8 前向延迟

下表单位均为 µs，取每个样本最慢 rank 的 p50。decode 使用 SGLang 的 5 次预热、20 个样本、每样本 50 次 graph replay。prefill 使用 Megatron 的变长 workload、完整 trace 预热、10 次 warmup 和 30 次采样。12 个点的路由哈希均与对应框架基线一致。

| 模型 | 场景 | tokens/rank | WeftMoE | MegaMoE | 框架基线 |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen35b | decode | 64 | 69.5 | 67.3 | 98.3 |
| qwen35b | decode | 512 | 109.6 | 109.6 | 193.8 |
| qwen35b | prefill | 8192 | 1306.4 | 1205.9 | 3944.5 |
| qwen35b | prefill | 20480 | 2983.3 | 2860.6 | 6144.7 |
| qwen122b | decode | 64 | 130.4 | 127.8 | 164.6 |
| qwen122b | decode | 512 | 171.0 | 168.2 | 318.6 |
| qwen122b | prefill | 8192 | 1865.9 | 1756.0 | 4781.7 |
| qwen122b | prefill | 20480 | 4388.3 | 4268.0 | 10050.6 |
| flash | decode | 64 | 282.0 | 279.5 | 308.0 |
| flash | decode | 512 | 361.8 | 358.7 | 586.6 |
| flash | prefill | 8192 | 2289.5 | 2184.6 | 5684.7 |
| flash | prefill | 20480 | 5557.1 | 5429.3 | 12346.5 |

WeftMoE 包含输入 staging、路由排序，并直接读取规范布局的权重。MegaMoE 使用提前准备的对称输入和交错权重。两者每个样本的 BF16 输出逐元素相等。框架列的 decode 来自 SGLang，prefill 来自 Megatron training FWD。此表的 WeftMoE/MegaMoE prefill 都是 inference forward，训练 checkpoint 的写入另计入训练 benchmark。

### EP8 BF16 训练延迟

单位为 ms。每个模式独立预热完整路由序列，再执行 10 次 warmup 和 30 次采样。每个样本取最慢 rank，表中给出样本 p50。两种 Megatron dispatcher 与 WeftMoE 的 18 组逐 rank workload SHA-256 全部一致。

| 模型 | tokens/rank | Weft FWD | Megatron 最快 FWD | Weft BWD | Megatron 最快 BWD | Weft cycle | Megatron 最快 cycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen35b | 8192 | 1.953 | 3.986 | 2.119 | 4.397 | 3.934 | 7.951 |
| qwen35b | 20480 | 3.864 | 6.191 | 5.091 | 6.355 | 8.803 | 12.174 |
| qwen122b | 8192 | 2.463 | 4.821 | 3.975 | 5.628 | 6.551 | 10.147 |
| qwen122b | 20480 | 5.350 | 10.089 | 10.030 | 12.010 | 15.392 | 21.706 |
| flash | 8192 | 3.100 | 5.724 | 6.263 | 7.754 | 9.281 | 13.094 |
| flash | 20480 | 6.858 | 12.386 | 16.122 | 17.366 | 22.934 | 29.539 |

EP1 与 EP8 的 32×128×64 tile 检查逐位一致。EP8 在包含空源 rank、空 expert、重复路由和两份交错 tape 的用例上，前向及 dX、dGate、dW1、dW2 重跑逐位一致。三个模型还使用 1024 nominal tokens/rank 的变长路由，与 PyTorch autograd 逐项核验。原始日志在 `benchmarks/results/sm100-b300/weft/`。

## 双机 B300 BF16

EP16 使用两台各八张 B300 的主机。通信和计算均计入延迟。decode 使用 CUDA Graph，以 SGLang 为基线。prefill 为推理前向，框架列按 Megatron 两种 dispatcher 中较快的训练前向输出就绪时间计算。训练 FWD、BWD 和 cycle 按完整调用的 wall time 分别测量。每个点逐 rank 校验输入路由 SHA-256，取每个样本的最慢 rank，再取 p50。三个模型共 30 个点，原始 p50、所选 dispatcher 和加速比见 [EP8 / EP16 完整 CSV](../benchmarks/sm100/bf16-results.csv)。

| 模型 | tokens/rank | Decode Weft / SGLang（µs） | tokens/rank | Prefill Weft / Megatron（ms） |
| --- | ---: | ---: | ---: | ---: |
| qwen35b | 64 | 172.1 / 306.8 | 8192 | 3.851 / 7.325 |
| qwen35b | 512 | 331.2 / 539.7 | 20480 | 9.193 / 14.178 |
| qwen122b | 64 | 207.0 / 345.4 | 8192 | 5.570 / 7.660 |
| qwen122b | 512 | 433.5 / 717.8 | 20480 | 13.183 / 17.068 |
| flash | 64 | 286.5 / 430.8 | 8192 | 6.106 / 8.486 |
| flash | 512 | 530.8 / 1015.5 | 20480 | 14.444 / 18.844 |

| 模型 | tokens/rank | 训练 FWD Weft / Megatron（ms） | BWD Weft / Megatron（ms） | Cycle Weft / Megatron（ms） |
| --- | ---: | ---: | ---: | ---: |
| qwen35b | 8192 | 3.815 / 7.353 | 3.638 / 6.241 | 7.400 / 12.952 |
| qwen35b | 20480 | 9.579 / 14.208 | 8.495 / 14.343 | 17.951 / 28.212 |
| qwen122b | 8192 | 5.612 / 7.687 | 6.199 / 8.297 | 11.554 / 15.991 |
| qwen122b | 20480 | 13.956 / 17.098 | 15.139 / 19.850 | 29.016 / 36.655 |
| flash | 8192 | 6.441 / 8.518 | 9.181 / 10.678 | 15.554 / 18.776 |
| flash | 20480 | 15.632 / 18.876 | 22.369 / 25.837 | 37.635 / 44.572 |

### 执行与数值校验

每个源 rank 通过 NCCL device GIN 将一份输入发往每个远端节点的同 lane GPU。节点内使用 NVLink 融合 MegaMoE，跨节点只回传实际命中该节点的 BF16 专家输出。源端固定按全局 (expert, slot) 顺序以 FP32 累加。训练保留目的端 checkpoint，反向返回 dX 和 dGate，独立 CUDA stream 上的 grouped GEMM 同时计算 dW。

容量达到 4096 tokens/rank 时，两个 token 窗口各自拥有对称内存和 GIN 注册窗口。B300 上的前向计算使用 148 个 SM 中的 144 个，给并发通信留出执行资源。反向在计算 dW 前按 expert 合并两份 checkpoint，每个权重梯度只做一次最终 BF16 舍入。Qwen35B、20480 tokens/rank 的训练时间线上，rank 0 单对 GIN 与 GEMM kernel 的最长重叠为前向 1.173 ms、反向 0.215 ms。

EP16 上，固定 tile 为 16×128×64 和 240×128×64 的两组输入均与 EP1 逐位一致，大负载覆盖全局 131072 tokens。4608 tokens/rank 容量下，两份交错存活的训练 tape 在空源 rank、空 expert、重复及无效路由下重跑，输出及四个梯度逐位相同。三个模型在 6144 nominal tokens/rank 下与 PyTorch autograd 比较，输出和全部梯度的最大相对 L2 误差为 2.43e-4。EP32 在四台机器上通过 16×128×64 tile 的 EP1 前向逐位对照，并通过同 EP 下输出与四个梯度的逐位复现。性能矩阵覆盖 EP8 和 EP16。EP8、EP16 也验证了相同 CUDA Graph 上修改输入和路由后的回放。这组历史 decode 测量使用相同的 BF16 前向，按负载选择 M tile。

### 每卡双路 400G RoCE

每个 `mlx5_bond_0..7` 对应一张 GPU 的两个 400G 物理接口。NCCL 2.29.7 原版 GDAKI 在建连时会重复以当前秒数初始化 RoCE 源端口随机数，使多个 QP 可能选到同一个 bond 成员。仓库提供针对这一版本的补丁，为不同 GIN connection 分配不同的 UDP 源端口。以下命令从 WeftMoE 根目录构建独立的 NCCL 前缀。

```bash
git clone --depth 1 --branch v2.29.7-1 https://github.com/NVIDIA/nccl.git build/nccl-2.29.7
git -C build/nccl-2.29.7 apply "$PWD/benchmarks/sm100/nccl-2.29.7-gdaki-roce-entropy.patch"
CUDA_HOME=/usr/local/cuda-13.0 make -C build/nccl-2.29.7 -j32 src.build \
    BUILDDIR="$PWD/build/sm100-b300/nccl-gdaki" NVCC_GENCODE='-gencode=arch=compute_103,code=sm_103'
```

双机 EP16 表格来自这个独立前缀的 GDAKI。启动脚本在创建 communicator 前选择对应 NCCL，设置 4 个 GIN connections、每连接 4 个 IB QP、关闭 NIC 合并，并指定八个 HCA。device kernel 使用 16 个 GIN contexts。可用 `WEFTMOE_GDAKI_NCCL_PREFIX` 指向同版本的其他构建位置。缺少私有库时脚本默认选用 NCCL GIN IB Proxy，它也已在本机型验证过双端口传输。

在两节点间做每卡 64 MiB × 50 次同时双向传输，最慢 GPU 的单向发送有效载荷带宽为 **88.99 GB/s（712 Gb/s）**。在表中的 12 个实际 MoE 前向负载上，两节点全部 16 个 bond 的 `tx_bytes_phy` 和 `rx_bytes_phy` 增量均已采样。各 bond 第一成员的发送与接收占比均在 **49.909%–50.090%**，剩余流量走第二成员。

两台机器分别设 `NODE_RANK=0` 和 `1`，设相同的 `NNODES=2`、`MASTER_ADDR` 和 `MASTER_PORT`，然后执行相同命令。

```bash
export PYTHON="$PWD/build/sm100-env/bin/python"
benchmarks/sm100/run.sh weft-gin --mib 64 --iterations 50
benchmarks/sm100/run.sh weft --mode prefill --model flash --port-counters
benchmarks/sm100/run_matrix.sh benchmarks/results/sm100-b300/replay weft
```

原始逐样本数据及网口计数位于 `benchmarks/results/sm100-b300/weft/release-{inference,training}.node*.log`，带宽和数值回归见同目录的 `gdaki-entropy-bw`、`release-verification` 和 `pipeline-validation` 日志。

## WeftMoE NVFP4（2026-11-20）

单机前向在同一个持久 kernel 内执行 NVLink dispatch、两次 block16 NVFP4 Tensor Core GEMM、SwiGLU、中间量化及按 slot 归并。推理从 BF16 输入开始计时，包含在线量化 X 和稳定路由排序，固定专家权重预先量化。训练从 BF16 参数开始，每次前向量化 FC1、FC2 权重，每次反向沿相应的 K 维量化转置权重。两次激活梯度和两次权重梯度 GEMM 均使用 E2M1 操作数。

跨节点通信使用 NCCL GIN。推理在源 rank 量化一次，每 16 个特征传输 8 字节 E2M1 数据和 1 字节 E4M3 尺度。训练前向向专家发送 BF16 X，在反向为权重梯度独立执行转置量化。反向传输 BF16 dY，返回 FP32 节点 dX 部分和及 gate 梯度。双窗口流水按确定的专家行序合并两个窗口的权重梯度计算。

三个模型每阶段各测两个 token 桶，每个样本先取各 rank 的最大延迟，再取样本 p50 计算加速比。Decode 为 64、512 tokens/rank，prefill 和训练为名义 8192、20480 tokens/rank。60 个点均与对应基线逐 rank 核对路由哈希。[逐点延迟 CSV](../benchmarks/sm100/nvfp4-results.csv)

| 阶段 | 基线 | EP8 | EP16 |
| --- | --- | ---: | ---: |
| Decode | SGLang NVFP4 | 1.83–2.88× | 1.69–2.13× |
| Prefill | Megatron NVFP4 训练前向 | 3.51–6.71× | 1.58–2.71× |
| 训练前向 | Megatron NVFP4 | 1.73–3.57× | 1.12–2.24× |
| 训练反向 | Megatron NVFP4 | 1.25–3.28× | 1.17–2.02× |
| 完整 FWD+BWD | Megatron NVFP4 | 1.43–3.25× | 1.15–2.01× |

### 推理前向延迟

单位为 µs。Decode 使用 SGLang 的固定长度路由，预热 5 次，采集 20 个样本，每样本重放 CUDA Graph 50 次。Prefill 使用 Megatron 的变长路由，完整 trace 预热后再预热 10 次、采样 30 次。框架列的 decode 取 SGLang，prefill 取 Megatron 训练前向的输出就绪时间，后者包含训练状态保存。

| 模型 | 场景 | tokens/rank | EP8 Weft | EP8 MegaMoE | EP8 框架 | EP16 Weft | EP16 框架 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen35b | decode | 64 | 50.0 | 47.7 | 91.5 | 158.6 | 309.2 |
| qwen35b | decode | 512 | 99.4 | 99.2 | 208.5 | 307.2 | 580.9 |
| qwen35b | prefill | 8192 | 1211.5 | 1090.5 | 8131.7 | 3601.7 | 9769.6 |
| qwen35b | prefill | 20480 | 2693.1 | 2574.9 | 9465.7 | 8452.5 | 17093.6 |
| qwen122b | decode | 64 | 74.0 | 69.3 | 136.2 | 175.1 | 334.3 |
| qwen122b | decode | 512 | 127.5 | 115.8 | 279.4 | 387.7 | 720.4 |
| qwen122b | prefill | 8192 | 1471.3 | 1324.0 | 9340.4 | 4778.9 | 10482.6 |
| qwen122b | prefill | 20480 | 3337.5 | 3119.5 | 11835.3 | 11620.4 | 18353.0 |
| flash | decode | 64 | 113.5 | 107.1 | 230.3 | 219.6 | 372.1 |
| flash | decode | 512 | 149.9 | 151.2 | 431.0 | 431.4 | 919.2 |
| flash | prefill | 8192 | 1367.1 | 1268.6 | 8823.5 | 4819.0 | 9894.7 |
| flash | prefill | 20480 | 2996.9 | 3001.5 | 13083.8 | 11273.4 | 17932.8 |

单机原版 MegaMoE 使用预先准备好的 MXFP8 激活、MXFP4 权重及路由。WeftMoE 的 W4A4 前向从 BF16 激活开始计时，12 个 EP8 点的 WeftMoE／MegaMoE 延迟比为 0.992–1.111，中位数为 1.064。

### 训练延迟

单位为 ms，每格为 WeftMoE / Megatron。FWD、BWD 和完整 FWD+BWD 分开运行，各自预热 10 次、采样 30 次。FWD 计到输出就绪，BWD 计到梯度就绪，完整 FWD+BWD 为一次前传加反传的 wall time。每个负载都从 alltoall 和 flex 中取较快的 Megatron 基线，本次 EP8 均为 alltoall，EP16 均为 flex。

| EP | 模型 | 名义 tokens/rank | 训练 FWD | BWD | 完整 FWD+BWD |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | qwen35b | 8192 | 2.280 / 8.132 | 2.246 / 7.359 | 4.629 / 15.064 |
| 8 | qwen35b | 20480 | 4.779 / 9.466 | 4.955 / 8.443 | 9.704 / 16.146 |
| 8 | qwen122b | 8192 | 3.023 / 9.340 | 3.665 / 7.734 | 6.604 / 15.357 |
| 8 | qwen122b | 20480 | 6.375 / 11.835 | 8.325 / 11.670 | 14.560 / 23.280 |
| 8 | flash | 8192 | 3.868 / 8.824 | 5.117 / 7.715 | 8.997 / 15.526 |
| 8 | flash | 20480 | 7.568 / 13.084 | 11.149 / 13.908 | 18.715 / 26.748 |
| 16 | qwen35b | 8192 | 4.358 / 9.770 | 3.680 / 7.446 | 7.987 / 16.071 |
| 16 | qwen35b | 20480 | 10.566 / 17.094 | 8.405 / 14.488 | 19.059 / 31.615 |
| 16 | qwen122b | 8192 | 5.656 / 10.483 | 5.520 / 7.889 | 11.339 / 18.037 |
| 16 | qwen122b | 20480 | 14.955 / 18.353 | 13.332 / 18.278 | 28.110 / 36.302 |
| 16 | flash | 8192 | 6.882 / 9.895 | 7.257 / 8.683 | 14.029 / 18.372 |
| 16 | flash | 20480 | 16.042 / 17.933 | 17.429 / 20.391 | 33.457 / 38.418 |

与本项目的 BF16 相比，24 个推理点使用 NVFP4 均有更低延迟。Qwen35B 的四个完整训练点则由 BF16 更快，NVFP4 延迟高出 6.2%–17.7%。例如 EP8、8192 tokens/rank 时，NVFP4 的完整 FWD+BWD 为 4.629 ms，WeftMoE BF16 为 3.934 ms，同负载的 Megatron NVFP4 为 15.064 ms。[BF16 逐点延迟](../benchmarks/sm100/bf16-results.csv)

### 双机网口流量

在 EP16 的两台主机上，按 GPU 配对读取两个 400G 物理端口的 TX、RX 字节增量，两个方向分别计算较轻端口占该对端口的流量比例。32 个端口均有收发。六个 decode 点的最低占比为 43.72%，六个 prefill 点均超过 49.99%。单独测量 2052 tokens/rank 的双窗口反向时，较轻端口占比为 49.91%–50.00%。

### 数值与确定性

独立参考显式计算 E2M1 最近偶数舍入、E4M3 块尺度及 FP32 GEMM。EP1 和 EP8 覆盖 K128、K256。65,536 个 token、64 个专家、top-4 的压力测试中，EP1 和 EP16 都使用 240×128×256 tile，前向输出逐位相等，相对独立参考的 L2 误差为 1.89e-4。该用例检查反转 token 顺序、CUDA Graph 改变输入后的重放、单个 rank 空输入和工作区输入字节。EP1 的对齐后专家行数为 270,960，工作区按最大接收路由数和每个本地专家的尾块合计预留 284,160 行。

训练用例包含重复路由、无效路由、空专家和零 gate。每个 rank 分别计算各张量对独立 NVFP4 参考的相对 L2，表中取各 rank 的最大值。每个用例同时保留两份训练 tape 并倒序反传，五组用例的输出和四个梯度均逐位复现。

| EP | tokens/rank | 标定与负载 | 输出 | dX | dGate | dW1 | dW2 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 36 | 默认 | 0 | 0 | 1.43e-7 | 0 | 0 |
| 8 | 36 | 自定义六个层级因子 | 3.75e-5 | 1.04e-4 | 5.39e-5 | 3.68e-5 | 4.96e-4 |
| 16 | 36 | 默认 | 0 | 0 | 1.50e-7 | 0 | 0 |
| 16 | 2052 | 默认，双窗口 | 6.31e-4 | 0 | 1.25e-7 | 0 | 0 |
| 32 | 36 | 默认 | 0 | 0 | 1.41e-7 | 0 | 0 |

自定义标定用例的最大误差出现在 dW2。针对 dY 因子 0.025，还单独构造了落在 E4M3 舍入中点的块，逐字节检查行向、转置方向的块尺度及 E2M1 编码。双窗口用例使用每 rank 4096 token 的容量和 2052 个实际 token，两个窗口都参与梯度计算。融合前向的 SwiGLU 使用快速指数与倒数，反向使用显式 FP32 运算顺序。误差表验证 WeftMoE 的确定性 NVFP4 配方。性能对照中的 Megatron 使用 TE 的 NVFP4 训练配方。

四机 EP32 使用相同的 16×128×256 tile 与 EP1 比较，前向输出逐位一致。EP32 用于数值验证，性能矩阵覆盖 EP8 和 EP16。

### 复现

```bash
benchmarks/sm100/run.sh weft --model qwen35b --precision fp4 --mode prefill --mega-mixed
benchmarks/sm100/run.sh weft-training --model qwen35b --precision fp4 --backward-only --samples 30
torchrun --standalone --nproc-per-node=8 tests/torchrun_sm100_nvfp4.py
torchrun --standalone --nproc-per-node=8 tests/torchrun_sm100_nvfp4_backward.py
torchrun --standalone --nproc-per-node=8 tests/torchrun_sm100_nvfp4_backward.py --scales 0.021 0.014 0.0023 0.018 0.025 0.0017
```

大负载前向测试使用双机 torchrun 和 `--total-tokens 65536`。双窗口反向测试使用双机 torchrun 和 `--capacity 4096 --tokens 2052 --hidden 512 --port-counters`。逐样本记录和网口计数保存在 `benchmarks/results/sm100-b300/weft/nvfp4-calibrated-*.node*.log`。

## 动态 M 的数值回归

[逐位回归脚本](../tests/torchrun_sm100_training_invariance.py)使用 128 个固定 token 作为 EP1 参考。分布式运行打乱这些 token 的位置并加入其他 token，以 EP1 训练前向对照分布式推理前向，再以 EP1 推理前向对照分布式训练前向。反向测试在相同 EP 下插入另一批次的前反向，然后重跑原批次，逐位比较 dX、dGate、dW1、dW2。

默认开启 `batch_invariant` 后，BF16、NVFP4 共跑了 95 组精度、形状、EP 和长度组合（EP8 30 组，EP16 35 组，EP32 30 组）。包含 Qwen35B 专家形状、NVFP4 的 K128/K256 及 EP16 跨窗口批次。实际选到的 M 有 16、32、64、128、192、240，包含 EP1 用 M16、分布式用 M240 的对照。两个方向的训练/推理前向与四个同 EP 梯度均为零逐位差异。

单机 EP8 的 NVFP4 回归可用下面的命令复现。多机使用相同脚本和每节点 8 个进程，并传入各节点的 torchrun rendezvous 参数。

```bash
torchrun --standalone --nproc-per-node=8 tests/torchrun_sm100_training_invariance.py --precision nvfp4
```

## 动态与固定 M 实测

本节保留历史消融数据。复现本节的固定 M、全宽尾块及其命令需使用 WeftMoE `18e8f63` 和 DeepGEMM `faa68bc`。当前实现统一使用动态 M。

在 B300 上对三个模型、BF16/NVFP4、EP8/EP16 分别做配对前向测试。每个 runtime 统一预留 20,480 tokens/rank，依次提交 8192、20480、1024、64、512，再循环。每轮各 rank 重新生成输入和路由，同一轮内所有策略复用相同输入、专家权重、对称工作区和 GIN 窗口。测的是含 staging、通信、两次专家 GEMM、combine 的推理前向。NVFP4 权重在计时前准备。

三种策略为动态选 M、固定外层 M 并保留专家尾块缩窄、固定 M 并让尾块也执行全宽 MMA。三种策略都使用相同的逻辑 N/K。本组模型 BF16 为 N128/K64，NVFP4 为 N128/K256。六个候选 M 均实测。固定策略轮换执行顺序，首次编译和 CUDA Graph 捕获在计时外。4 轮预热、20 轮采样，每个样本先取各 rank 最大值再取 p50。64 和 512 使用 CUDA Graph，EP8 每样本回放 20 次，EP16 回放 8 次。其余点为单次 eager 调用到设备同步的 wall time。EP8 全宽 M16/M192 用第二次同参数配对运行测量，使用该次运行的动态基线。

### 相对动态 M 的延迟

百分比等于固定策略 p50 除以同次动态策略 p50 再减一。正值表示固定策略更慢。每格对三个模型、两种精度的六个比值等权取几何平均。

| tokens/rank | EP8 固定 M128，缩窄尾块 | EP8 严格 M128 | EP16 固定 M128，缩窄尾块 | EP16 严格 M128 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | +3.2% | +4.7% | -1.9% | -2.0% |
| 512 | -0.1% | +0.2% | -0.4% | -0.4% |
| 1024 | -1.9% | -1.4% | -0.2% | -0.4% |
| 8192 | +0.2% | +0.5% | +0.6% | +0.4% |
| 20480 | +1.0% | +0.8% | +2.7% | +2.6% |

同样固定一个 M 时，各候选在每个 EP 的 30 个工作负载上表现如下。每组模型、精度和长度权重相同。

| M | EP8 缩窄尾块 | EP8 全宽 | EP16 缩窄尾块 | EP16 全宽 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | +92.5% | +89.6% | +37.8% | +37.9% |
| 32 | +31.1% | +31.0% | +12.9% | +13.0% |
| 64 | +7.0% | +7.4% | +2.6% | +2.8% |
| 128 | +0.5% | +0.9% | +0.1% | 0.0% |
| 192 | +1.0% | +2.6% | -0.3% | -0.2% |
| 240 | +2.7% | +4.5% | +0.5% | +0.8% |

当时的严格配置取 M128。全宽 M128 在 60 个等权负载上相对动态的几何平均开销为 0.45%，M192 为 1.17%；双机单独计算时两者相差约 0.2%。单机 M128 的逐点范围为 −7.0% 至 +10.4%，后者来自 Qwen35B NVFP4 的 64 tokens/rank（49.97 → 55.16 µs）。仅固定外层 M128 时，EP8/EP16 的平均开销分别为 0.48%/0.14%。这些百分比是样本统计量。

EP8 的动态策略在 64 tokens/rank 时三个模型都选择 M32。负载依次为 8K、20K、1K 时，两个 Qwen 模型选择 240、240、192，Flash 选择 240、240、240。双机的节点内 GEMM 使用预留通信窗口的行数，本实验五个长度全部选择 M240。为 decode 单独准备小容量 runtime 会改变双机启发式的输入。

### 8K → 20K → 1K 混合负载

每格为三个前向 p50 之和，单位 ms，顺序为动态 M / 严格 M128。这个数便于按一次长短混合负载估算总时间，未包含批次之间的调度间隔。

| 模型 | EP8 BF16 | EP8 NVFP4 | EP16 BF16 | EP16 NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| qwen35b | 4.622 / 4.592 | 4.215 / 4.151 | 13.191 / 13.227 | 11.641 / 11.932 |
| qwen122b | 6.744 / 6.717 | 5.132 / 5.259 | 18.110 / 18.108 | 15.211 / 15.680 |
| flash | 8.761 / 9.509 | 4.742 / 4.521 | 19.437 / 20.467 | 16.473 / 16.275 |

原始样本、各策略的 p50、实际选用的 M 和逐位差异计数见[逐点 CSV](../benchmarks/sm100/tile-m-results.csv)。在上述历史版本的仓库根目录运行下面的命令可复现主测试；双机在两台主机设置相同的 `NNODES=2`、`MASTER_ADDR`、`MASTER_PORT` 及各自的 `NODE_RANK`。两种精度分别使用 `bf16` 和 `nvfp4`。

```bash
benchmarks/sm100/run.sh weft-tile-m --model qwen35b --precision nvfp4 --full-m 32 64 128 240
# EP8 补测的两个全宽候选，脚本自动同时测动态基线。
benchmarks/sm100/run.sh weft-tile-m --model qwen35b --precision nvfp4 --fixed-m --full-m 16 192
# EP16
benchmarks/sm100/run.sh weft-tile-m --model qwen35b --precision nvfp4 --full-m 16 32 64 128 192 240 --replays 8
```

### 双机 Qwen122B NVFP4 的重复采样

主矩阵中 EP16、20K 的动态策略和显式 M240 运行同一个内核，20 次采样的 p50 却相差 17.4%。在同一对主机保留五种长度的交替顺序，缩小到四种策略，8 次预热后各取 160 个样本。20K 时动态和显式 M240 的差异收敛为 −0.06%。严格 M128 的 p50 高 7.45%，四种策略的分布都有慢样本。

| 20K 策略 | p25（ms） | p50（ms） | p75（ms） | p50 相对动态 |
| --- | ---: | ---: | ---: | ---: |
| 动态，选中 M240 | 9.002 | 9.076 | 9.760 | 基准 |
| 固定 M240，缩窄尾块 | 9.001 | 9.070 | 9.749 | −0.06% |
| 固定 M128，缩窄尾块 | 9.055 | 9.158 | 9.808 | +0.91% |
| 严格 M128 | 9.063 | 9.752 | 9.811 | +7.45% |

同次重复测量中，严格 M128 在 64、512、1024、8192 tokens/rank 上的 p50 分别比动态变动 −3.47%、−1.27%、−0.18%、+0.81%。上方主矩阵保留各点相同的 20 个样本。复验命令如下。

```bash
benchmarks/sm100/run.sh weft-tile-m --model qwen122b --precision nvfp4 --samples 160 --warmups 8 --replays 8 --fixed-m 128 240 --full-m 128
```

### 严格模式数值检查

BF16 和 NVFP4 均将同一份输入分别交给 EP1 与 EP8、EP16、EP32 计算，按原始行号对齐后比较输出位模式。用例覆盖裁短 batch 后保留的前缀、token 重排、无效或重复路由。NVFP4 还覆盖修改已捕获图的输入后回放及空源 rank。EP16 在 NVFP4 K128、K256 都通过逐位检查。K256 的 65,536-token 用例与独立量化及 FP32 GEMM 参考的相对 L2 为 1.89×10⁻⁴。

三个 EP 规模的训练用例均检查相同输入的输出、dX、dGate、dW1、dW2 重跑逐位一致，以及训练前向和推理前向逐位一致。EP16 另覆盖 BF16 容量 4608、NVFP4 容量 4096 的双窗口训练。在配对性能矩阵的 810 条策略记录中，每一轮的输出与同轮动态 M 的 BF16 输出位模式相比均为零差异。放宽模式的逐位相同是本次输入上的观测，API 的跨 batch 保证由严格开关提供。

同一环境下运行仓库 pytest 得到 112 passed、72 skipped、6 failed，与改动前的 B300 日志一致。五项依赖 Hopper 专用 FP8，一项依赖本机 SGLang 版本中已移动的 `cuda_graph_runner` 模块。框架基线使用 benchmark 的独立入口。

### 严格模式与框架基线

用原性能矩阵的路由和按桶准备的工作区重新测量严格 M128。24 个 decode 点逐一核对 SGLang 路由哈希，24 个 prefill 点逐 rank 核对 Megatron 的路由哈希，框架延迟复用原报告实测记录。decode 使用每样本 50 次图回放，prefill 使用变长输入。加速比取框架 p50 / WeftMoE 严格模式 p50。Megatron 取 all-to-all 和 flex 中更快的训练前向输出就绪时间。

| EP | 精度 | decode 相对 SGLang | prefill 相对 Megatron |
| ---: | --- | ---: | ---: |
| 8 | BF16 | 1.10–1.79× | 2.02–2.99× |
| 8 | NVFP4 | 1.74–2.86× | 3.49–7.01× |
| 16 | BF16 | 1.43–1.79× | 1.24–1.84× |
| 16 | NVFP4 | 1.69–2.26× | 1.55–2.79× |

48 个推理点全部快于对应框架基线。训练反向在这次变更中做了数值复验，历史训练耗时仍按原配置标注。各点耗时与基线见[严格前向 CSV](../benchmarks/sm100/strict-forward-results.csv)。
