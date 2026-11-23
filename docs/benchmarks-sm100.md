# B300 / SM103 性能测试

[单机 EP8 训练与 decode 汇总](../README.md#blackwell--sm103)

## 测试环境

| 项目 | 配置 |
| --- | --- |
| 硬件 | 每节点 8 张 B300（SM103），机内 NVLink，跨机每卡双 400 Gb/s RoCE |
| WeftMoE | 驱动 580.105.08，CUDA 13.0，PyTorch 2.13.0+cu130，Triton，CUTLASS/CuTe，NCCL 2.29.7 |
| SGLang | SGLang 0.5.20，FlashInfer 0.6.18+cu130，TRT-LLM routed MoE |
| Megatron | Megatron-Core 0.19.0，Transformer Engine 2.20.2，NCCL all-to-all 和 flex / DeepEP 1.2.1 |

EP16 的 DeepEP 使用 NVSHMEM 3.5.19 和[双机 RoCE 补丁](../benchmarks/sm100/deepep-v1.2.1-nvshmem35.patch)。

## 各模型实测

表头中的 token 数均为每 rank 输入量，H 和 I 分别是隐藏维度和专家中间维度。三模型全量基准使用 DeepGEMM 后端。EP8 和 EP16 均已用 CUTLASS 后端复测三个模型两档长度的 BF16 反向，以及 Qwen35B 两种精度的 64-token decode、8192-token prefill 和反向。表中这些配置取复测值。

推理表的框架基线为 SGLang decode 和 Megatron 训练前向（含缓存写入，两种 dispatcher 取较快者）。单机 MegaMoE 从已准备好的输入、路由和权重开始计时。训练表的 FWD、BWD 分别计至输出、梯度就绪。FWD+BWD 为一次完整调用的实测耗时。

### Qwen3.5-35B-A3B

H=2048，I=512，256 experts，top-k=8。

**推理**

| EP | 精度 | 实现 | Decode 64（µs） | Decode 512（µs） | Prefill 8192（ms） | Prefill 20480（ms） |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 8 | BF16 | WeftMoE | 69.52 | 109.64 | 1.306 | 2.983 |
| 8 | BF16 | SGLang / Megatron | 98.37 | 193.77 | 3.944 | 6.145 |
| 8 | BF16 | MegaMoE | 67.90 | 109.63 | 1.221 | 2.861 |
| 8 | NVFP4 | WeftMoE | 49.55 | 99.40 | 1.161 | 2.693 |
| 8 | NVFP4 | SGLang / Megatron | 91.48 | 208.48 | 8.132 | 9.466 |
| 8 | NVFP4 | MegaMoE | 47.23 | 93.94 | 1.075 | 2.517 |
| 8 | MXFP8×MXFP4 | MegaMoE | 47.72 | 99.21 | 1.091 | 2.575 |
| 16 | BF16 | WeftMoE | 173.83 | 331.20 | 3.951 | 9.193 |
| 16 | BF16 | SGLang / Megatron | 306.78 | 539.68 | 7.325 | 14.178 |
| 16 | NVFP4 | WeftMoE | 157.67 | 307.18 | 3.719 | 8.452 |
| 16 | NVFP4 | SGLang / Megatron | 309.17 | 580.93 | 9.770 | 17.094 |

**训练（ms）**

| EP | 精度 | 实现 | FWD 8192 | BWD 8192 | FWD+BWD 8192 | FWD 20480 | BWD 20480 | FWD+BWD 20480 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | BF16 | WeftMoE | 1.934 | 2.045 | 3.934 | 3.815 | 4.858 | 8.803 |
| 8 | BF16 | Megatron all-to-all | 3.944 | 4.326 | 7.951 | 6.145 | 6.293 | 12.174 |
| 8 | BF16 | Megatron flex | 7.343 | 6.116 | 12.931 | 13.330 | 13.269 | 26.706 |
| 8 | NVFP4 | WeftMoE | 2.280 | 2.248 | 4.629 | 4.779 | 4.955 | 9.704 |
| 8 | NVFP4 | Megatron all-to-all | 8.132 | 7.359 | 15.064 | 9.466 | 8.443 | 16.146 |
| 8 | NVFP4 | Megatron flex | 11.462 | 9.071 | 19.044 | 17.837 | 13.559 | 31.412 |
| 16 | BF16 | WeftMoE | 3.786 | 3.515 | 7.400 | 9.546 | 8.161 | 17.951 |
| 16 | BF16 | Megatron all-to-all | 9.018 | 8.995 | 17.740 | 21.252 | 21.675 | 42.715 |
| 16 | BF16 | Megatron flex | 7.325 | 6.170 | 12.952 | 14.178 | 14.267 | 28.212 |
| 16 | NVFP4 | WeftMoE | 4.358 | 3.653 | 7.987 | 10.566 | 8.405 | 19.059 |
| 16 | NVFP4 | Megatron all-to-all | 10.189 | 9.472 | 19.058 | 22.980 | 22.661 | 45.188 |
| 16 | NVFP4 | Megatron flex | 9.770 | 7.446 | 16.071 | 17.094 | 14.488 | 31.615 |

### Qwen3.5-122B-A10B

H=3072，I=1024，256 experts，top-k=8。

**推理**

| EP | 精度 | 实现 | Decode 64（µs） | Decode 512（µs） | Prefill 8192（ms） | Prefill 20480（ms） |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 8 | BF16 | WeftMoE | 130.43 | 171.02 | 1.866 | 4.388 |
| 8 | BF16 | SGLang / Megatron | 164.58 | 318.60 | 4.782 | 10.051 |
| 8 | BF16 | MegaMoE | 127.82 | 168.15 | 1.756 | 4.268 |
| 8 | NVFP4 | WeftMoE | 74.02 | 127.47 | 1.471 | 3.337 |
| 8 | NVFP4 | SGLang / Megatron | 136.21 | 279.36 | 9.340 | 11.835 |
| 8 | NVFP4 | MegaMoE | 70.82 | 121.54 | 1.344 | 3.236 |
| 8 | MXFP8×MXFP4 | MegaMoE | 69.25 | 115.75 | 1.324 | 3.119 |
| 16 | BF16 | WeftMoE | 206.95 | 433.49 | 5.570 | 13.183 |
| 16 | BF16 | SGLang / Megatron | 345.36 | 717.82 | 7.660 | 17.068 |
| 16 | NVFP4 | WeftMoE | 175.09 | 387.69 | 4.779 | 11.620 |
| 16 | NVFP4 | SGLang / Megatron | 334.31 | 720.42 | 10.483 | 18.353 |

**训练（ms）**

| EP | 精度 | 实现 | FWD 8192 | BWD 8192 | FWD+BWD 8192 | FWD 20480 | BWD 20480 | FWD+BWD 20480 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | BF16 | WeftMoE | 2.444 | 3.871 | 6.551 | 5.332 | 9.600 | 15.392 |
| 8 | BF16 | Megatron all-to-all | 4.782 | 5.565 | 10.147 | 10.051 | 11.944 | 21.706 |
| 8 | BF16 | Megatron flex | 7.568 | 7.913 | 15.209 | 15.720 | 18.659 | 34.144 |
| 8 | NVFP4 | WeftMoE | 3.023 | 3.665 | 6.604 | 6.375 | 8.325 | 14.560 |
| 8 | NVFP4 | Megatron all-to-all | 9.340 | 7.734 | 15.357 | 11.835 | 11.670 | 23.280 |
| 8 | NVFP4 | Megatron flex | 11.871 | 9.477 | 20.630 | 19.541 | 17.722 | 36.790 |
| 16 | BF16 | WeftMoE | 5.578 | 5.655 | 11.554 | 13.915 | 13.501 | 29.016 |
| 16 | BF16 | Megatron all-to-all | 13.129 | 14.207 | 27.444 | 32.418 | 35.317 | 67.289 |
| 16 | BF16 | Megatron flex | 7.660 | 8.230 | 15.991 | 17.068 | 19.778 | 36.655 |
| 16 | NVFP4 | WeftMoE | 5.656 | 5.520 | 11.339 | 14.955 | 13.332 | 28.110 |
| 16 | NVFP4 | Megatron all-to-all | 14.325 | 14.302 | 27.855 | 33.726 | 34.271 | 67.214 |
| 16 | NVFP4 | Megatron flex | 10.483 | 7.889 | 18.037 | 18.353 | 18.278 | 36.302 |

### DeepSeek V4 Flash

H=4096，I=2048，256 experts，top-k=6。

**推理**

| EP | 精度 | 实现 | Decode 64（µs） | Decode 512（µs） | Prefill 8192（ms） | Prefill 20480（ms） |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 8 | BF16 | WeftMoE | 281.97 | 361.76 | 2.289 | 5.557 |
| 8 | BF16 | SGLang / Megatron | 308.03 | 586.56 | 5.685 | 12.347 |
| 8 | BF16 | MegaMoE | 279.47 | 358.66 | 2.185 | 5.429 |
| 8 | NVFP4 | WeftMoE | 113.55 | 149.93 | 1.367 | 2.997 |
| 8 | NVFP4 | SGLang / Megatron | 230.33 | 431.05 | 8.824 | 13.084 |
| 8 | NVFP4 | MegaMoE | 109.86 | 144.76 | 1.159 | 2.715 |
| 8 | MXFP8×MXFP4 | MegaMoE | 107.15 | 151.18 | 1.269 | 3.002 |
| 16 | BF16 | WeftMoE | 286.55 | 530.76 | 6.106 | 14.444 |
| 16 | BF16 | SGLang / Megatron | 430.80 | 1015.53 | 8.486 | 18.844 |
| 16 | NVFP4 | WeftMoE | 219.60 | 431.44 | 4.819 | 11.273 |
| 16 | NVFP4 | SGLang / Megatron | 372.13 | 919.15 | 9.895 | 17.933 |

**训练（ms）**

| EP | 精度 | 实现 | FWD 8192 | BWD 8192 | FWD+BWD 8192 | FWD 20480 | BWD 20480 | FWD+BWD 20480 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | BF16 | WeftMoE | 3.084 | 6.052 | 9.281 | 6.838 | 15.211 | 22.934 |
| 8 | BF16 | Megatron all-to-all | 5.685 | 7.690 | 13.094 | 12.347 | 17.305 | 29.539 |
| 8 | BF16 | Megatron flex | 8.664 | 10.236 | 18.731 | 19.399 | 24.342 | 43.325 |
| 8 | NVFP4 | WeftMoE | 3.868 | 5.117 | 8.997 | 7.568 | 11.149 | 18.715 |
| 8 | NVFP4 | Megatron all-to-all | 8.824 | 7.715 | 15.526 | 13.084 | 13.908 | 26.748 |
| 8 | NVFP4 | Megatron flex | 13.161 | 9.755 | 22.124 | 20.936 | 20.310 | 41.112 |
| 16 | BF16 | WeftMoE | 6.408 | 8.314 | 15.554 | 15.599 | 20.495 | 37.635 |
| 16 | BF16 | Megatron all-to-all | 14.427 | 16.908 | 30.687 | 35.380 | 41.913 | 76.957 |
| 16 | BF16 | Megatron flex | 8.486 | 10.612 | 18.776 | 18.844 | 25.765 | 44.572 |
| 16 | NVFP4 | WeftMoE | 6.882 | 7.257 | 14.029 | 16.042 | 17.429 | 33.457 |
| 16 | NVFP4 | Megatron all-to-all | 15.242 | 15.272 | 30.003 | 34.803 | 38.087 | 72.673 |
| 16 | NVFP4 | Megatron flex | 9.895 | 8.683 | 18.372 | 17.933 | 20.391 | 38.418 |

## 数值验证

BF16 和 NVFP4 均通过 EP1 对 EP8、EP16 的训推前向逐位检查。固定 EP 和整批输入后，四个梯度重跑逐位相等。Qwen35B、2048 tokens/rank 的 BF16 输出及梯度对照 PyTorch，最大相对 L2 误差为 EP8 的 9.49×10⁻⁵ 和 EP16 的 1.15×10⁻⁴。[数值测试](../tests/torchrun_sm100_training_invariance.py)

## 计时与复现

每个样本包含专家计算和通信，取最慢 rank 的延迟，再对样本取 p50。两端使用相同路由。Decode 固定 64 或 512 tokens/rank，预热 5 次、采样 20 次，每个样本回放 50 次 CUDA Graph。Prefill 和训练的输入长度围绕 8192 或 20480 tokens/rank 波动，预热 10 次、采样 30 次。BWD 计至梯度就绪。NVFP4 推理预先打包权重。

在仓库根目录执行以下命令。精度参数取 `bf16` 或 `fp4`（NVFP4）。

```bash
precision=bf16
benchmarks/sm100/run.sh weft --model qwen35b --precision "$precision" --mode decode --tokens 64
benchmarks/sm100/run.sh weft --model qwen35b --precision "$precision" --mode prefill --tokens 8192
benchmarks/sm100/run.sh weft-training --model qwen35b --precision "$precision" --backward-only --tokens 8192 --warmups 10 --samples 30
```

三种模型均可通过 `--model qwen35b|qwen122b|flash` 选择。运行全部模型及 token 档位时使用下列命令。双机在两台机器上分别执行，网络配置见下节。

```bash
benchmarks/sm100/run_matrix.sh benchmarks/results/b300/weft weft
benchmarks/sm100/run_matrix.sh benchmarks/results/b300/sglang sglang
MEGATRON_DISPATCHER=alltoall benchmarks/sm100/run_matrix.sh benchmarks/results/b300/alltoall megatron
MEGATRON_DISPATCHER=flex benchmarks/sm100/run_matrix.sh benchmarks/results/b300/flex megatron
```

### 每卡双路 400G RoCE

EP16 使用 NCCL GIN/GDAKI。实测的 `mlx5_bond_0..7` 各绑定一张 GPU 的两个端口。以下命令构建带有 RoCE 源端口分流补丁的 NCCL，启动脚本会从该目录加载它。

```bash
git clone --depth 1 --branch v2.29.7-1 https://github.com/NVIDIA/nccl.git build/nccl-2.29.7
git -C build/nccl-2.29.7 apply "$PWD/benchmarks/sm100/nccl-2.29.7-gdaki-roce-entropy.patch"
CUDA_HOME=/usr/local/cuda-13.0 make -C build/nccl-2.29.7 -j32 src.build \
    BUILDDIR="$PWD/build/sm100-b300/nccl-gdaki" NVCC_GENCODE='-gencode=arch=compute_103,code=sm_103'
```

双机分别设置 `NODE_RANK=0` 和 `1`，并导出相同的 `NNODES=2`、`MASTER_ADDR`（节点 0 的地址）和 `MASTER_PORT`，再运行上述 benchmark。可用 `benchmarks/sm100/run.sh weft-gin --mib 64 --iterations 50` 检查双端口传输。
