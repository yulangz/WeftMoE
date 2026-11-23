# B300 融合 MoE

WeftMoE 在 B300（SM103）上支持 BF16 和 NVFP4 的 routed experts，前向、反向和推理沿用同一套 Tensor API。单机使用 NVLink，跨机使用 NCCL GIN。运行时按 CUDA compute capability 选择内核，驱动显示的设备名称可以是 L20D。

## 安装

准备 Linux、支持 SM103 的 PyTorch 和 Triton、CUDA 13 的 `nvcc` 及支持 C++20 的宿主编译器。WeftMoE 使用仓库固定版本的 CUTLASS/CuTe 头文件。首次遇到新的前向特化会调用 NVCC，后续从 `$WEFTMOE_ROOT/blackwell` 复用 cubin，默认缓存根目录为 `~/.cache/weftmoe`。

```bash
git submodule update --init third_party/cutlass
export CUDA_HOME=/usr/local/cuda-13.0
python -m pip install --no-build-isolation .
```

多机要求各节点使用相同数量的 GPU，EP rank 按节点连续排列。在初始化 NCCL 进程组前配置 HCA 和 GIN。仓库的启动脚本支持本次实测的每 GPU 双 400G RoCE 拓扑。其他集群按实际 HCA 名称和 GIN 后端配置。[网络配置与端口验证](benchmarks-sm100.md#每卡双路-400g-roce)

## 调用

假设每个进程已选好本地 GPU，创建了 NCCL 专家并行组 `ep_group`。每个 rank 提供本地 BF16 输入 `x[T,H]`、全局专家编号 `routes[T,K]`（int32）、权重 `gates[T,K]`（FP32），以及本地 BF16 专家参数 `w1[E/EP,2I,H]`、`w2[E/EP,H,I]`。下面以 Qwen35B 的专家形状为例，容量按每个 rank 的最大 token 数设置。

```python
from weftmoe.runtime import TensorRuntime, moe

config = dict(group=ep_group, experts=256, top_k=8, hidden=2048, intermediate=512,
              max_tokens=8192, max_rows=8192 * ep_group.size() * 8, batch_invariant=True)
train = TensorRuntime(**config, training=True, precision="bf16")
y = moe(train, x, routes, gates, w1, w2)
y.float().square().mean().backward()  # 需梯度的输入和参数须已设置 requires_grad
train.close()  # 各 rank 在最后一次反向后共同关闭

infer = TensorRuntime(**config, training=False, precision="bf16")
y = infer.inference(0, x, routes, gates, w1, w2)
infer.close()
```

NVFP4 训练只需改为 `precision="nvfp4"`。两次前向与四次反向 GEMM 均使用 E2M1 操作数，每 16 个值共享一个 E4M3 块尺度。对外的激活和专家参数梯度为 BF16，gate 梯度为 FP32。从 `weftmoe.nvfp4` 导入 `NVFP4Scales` 后，可用 `nvfp4_scales=NVFP4Scales(x=..., w1=..., intermediate=..., w2=..., dy=..., dgate_up=...)` 配置，所有 EP rank 使用相同的有限正数。NVFP4 推理可在加载层时预打包权重，跨 batch 复用，权重更新后重新打包。

```python
from weftmoe.nvfp4 import prepare_nvfp4_weights

infer = TensorRuntime(**config, training=False, precision="nvfp4")
weights = prepare_nvfp4_weights(w1, w2, weight_scales=(1/64, 1/64))
y = infer.inference_nvfp4(0, x, routes, gates, weights, x_scale=1/32, intermediate_scale=1/512)
infer.close()
```

默认层级因子来自 benchmark。创建 NVFP4 训练 runtime 时，若省略 `nvfp4_scales`，EP 组的 rank 0 会记录一条警告。训练时应按实际模型、loss 归一化和梯度缩放方式标定，包括反向的 dY 和 dGateUp。对非零的块最大绝对值 `amax` 和层级因子 `s`，当 `amax ≤ 6s × 2⁻¹⁰` 时块尺度舍入为零，当 `amax > 6s × 448` 时幅值超出可表示范围。靠近下界的非零尺度可能是 E4M3 次正规数。

| 操作数 | 默认 s | 整块清零阈值 | 幅值上限 |
| --- | ---: | ---: | ---: |
| x、dY | 1/32 | 0.0001831 | 84 |
| w1、w2 | 1/64 | 0.00009155 | 42 |
| intermediate、dGateUp | 1/512 | 0.00001144 | 5.25 |

可用 `from weftmoe.nvfp4 import nvfp4_block_stats` 离线检查有限值样本。按对应 GEMM 的 K 维放在最后一维，调用 `stats = nvfp4_block_stats(dy, scales.dy)`。`nonzero_to_zero`、`subnormal_scale` 和 `above_range` 为块数，除以 `blocks` 得到各自比例。该检查按需运行。

训练 tape 按本 rank 的实际接收行数和专家对齐填充后分配。Blackwell 的 `forward_async` 在确定 tape 大小时会读取一次 GPU 计数，CPU 可能等待之前排队的工作。NVFP4 训练每次前向重新打包权重，反向按转置方向重新量化，前向打包计入训练前向，转置量化计入反向。推理部署可用上面的显式预打包接口控制权重生命周期。

### Batch invariance 与 autotune

`batch_invariant=True` 默认开启。在相同硬件、内核版本、精度、专家权重和量化因子下，保证两点。

- **跨 EP 前向**：同一 token 的输入、路由和 gate 相同，训练与推理的输出逐位一致。token 可以换位置，也可以和不同的 token 组成 batch。
- **同 EP 反向**：固定整批输入、路由、gate 和上游梯度，重跑得到的 dX、dGate、dW1、dW2 逐位一致。

开关的两种设置均按本次调用选择 M。单机根据本地 token 数估算每个专家的负载，多机按提交给节点内核的窗口行数估算。两次前向 GEMM 的 N 均为 128，BF16 的 K 为 64，NVFP4 在 I 可被 256 整除时用 K256，其余支持的形状用 K128。B300 无需预报负载。`tools/autotune.py` 识别 B300 后跳过 Hopper 的离线搜索。

## 性能

[单机 EP8、双机 EP16 延迟](../README.md#blackwell--sm103) · [测试环境与复现命令](benchmarks-sm100.md)
