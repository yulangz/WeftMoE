# B300 / SM103 性能测试

[Qwen3.5-35B-A3B 单层 MoE 延迟表](../README.md#blackwell--sm103)

## 测试环境

| 项目 | 配置 |
| --- | --- |
| 硬件 | 每节点 8 张 B300（SM103），机内 NVLink，跨机每卡双 400 Gb/s RoCE |
| WeftMoE | 驱动 580.105.08，CUDA 13.0，PyTorch 2.13.0+cu130，Triton，CUTLASS/CuTe，NCCL 2.29.7 |
| SGLang | SGLang 0.5.20，FlashInfer 0.6.18+cu130，TRT-LLM routed MoE |
| Megatron | Megatron-Core 0.19.0，Transformer Engine 2.20.2，取 NCCL all-to-all 和 flex / DeepEP 1.2.1 中较快的一项 |

EP8 的 Megatron 基线使用 NCCL all-to-all，EP16 使用 flex。EP16 的 DeepEP 使用 NVSHMEM 3.5.19 和[双机 RoCE 补丁](../benchmarks/sm100/deepep-v1.2.1-nvshmem35.patch)。

## 数值验证

BF16 和 NVFP4 均通过 EP1 对 EP8、EP16 的训推前向逐位检查。固定 EP 和整批输入后，四个梯度重跑逐位相等。Qwen35B、2048 tokens/rank 的 BF16 输出及梯度对照 PyTorch，最大相对 L2 误差为 EP8 的 9.49×10⁻⁵ 和 EP16 的 1.15×10⁻⁴。[数值测试](../tests/torchrun_sm100_training_invariance.py)

## 计时与复现

每个样本包含专家计算和通信，取最慢 rank 的延迟，再对样本取 p50。两端使用相同路由。Decode 固定 64 tokens/rank，预热 5 次、采样 20 次，每个样本回放 50 次 CUDA Graph。Prefill 和 BWD 的输入长度围绕 8192 tokens/rank 波动，预热 10 次、采样 30 次。BWD 计至梯度就绪。NVFP4 推理预先打包权重。

在仓库根目录执行以下命令。精度参数取 `bf16` 或 `fp4`（NVFP4）。

```bash
precision=bf16
benchmarks/sm100/run.sh weft --model qwen35b --precision "$precision" --mode decode --tokens 64
benchmarks/sm100/run.sh weft --model qwen35b --precision "$precision" --mode prefill --tokens 8192
benchmarks/sm100/run.sh weft-training --model qwen35b --precision "$precision" --backward-only --tokens 8192 --warmups 10 --samples 30
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
