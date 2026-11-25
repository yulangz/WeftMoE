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

### 单机 decode

在单机推理的 `TensorRuntime` 上设置 `training=False, decode_only=True` 可启用 DT。BF16 使用 `inference`，预打包的 NVFP4 使用 `inference_nvfp4`。按照每卡服务的最大 decode token 数设置 `max_tokens`。同一个 runtime 可以为多种 token 数捕获 CUDA Graph，每个 bucket 在捕获前先预热一次。

DT 为一次调用预留足以容纳全部路由的本地行空间，包括所有 rank 的重复槽都发往同一专家的情况。小批次的 TMA、MMA 和 epilogue warp 根据专家计数独立遍历同一组 FC1、FC2 tiles。较大批次可使用动态任务分配。两种调度均使用 DT 的路由分发、跨卡就绪信号和分段 top-k 归约。一般推理及训练使用 PF。

每个源 rank 把自己的计数向量写到各 owner 的独立位置，以 release/acquire 信号发布。各 CTA 在共享内存中汇总本地专家的接收行数，供 pull、调度和清理复用。NVFP4 激活量化与路由排序共用一次 Triton launch。两次 GEMM 的 K 累加顺序、量化尺度和按专家及原始槽号排序的 top-k 归约保持相同。

DT 的离线搜索有两个参数，CTA 的 M 取 16、32、64、128、192、240，任务调度取静态或动态。表项按 SM 数、EP、专家数、H、I、top-k、精度及每卡 token 数匹配。未测过的形状或长度使用负载估计。下面的命令在 8 卡 B300 上搜索一个形状，可用 `--hidden`、`--intermediate`、`--experts` 和 `--top-k` 覆盖模型尺寸。

```bash
torchrun --standalone --nproc-per-node=8 tools/autotune_sm103_decode.py \
  --model qwen35b --precision nvfp4 --capacity 512 --tokens 1 16 256 512 \
  --output build/decode-profile.json
export WEFTMOE_SM103_DECODE_PROFILE="$PWD/build/decode-profile.json"
```

搜索时每个候选都与保存反向状态的训练前向逐位比较，同输入也检查 PF 推理。候选在相同路由上交错计时，再用留出的路由复测原配置和排名靠前的候选。默认要求确认阶段的 p50 延迟至少降低 1.5% 才替换原配置。环境变量指定的文件覆盖内置表中的对应行。也可在创建 DT runtime 时传入 `decode_plan={16: (32, True)}` 指定本次 runtime 的表，布尔值 `True` 表示静态调度。EP 组在分配工作区前核对配置。

PF 和 DT 各自用工作区中的调用序号标记跨卡 FC2 完成信号。DT 的源 rank 收到所有 owner 对本次调用的确认后，可在下一次调用中覆写输入和路由目录。这样也覆盖了空 rank、某个 owner 没有收到该源的 token，以及反复回放同一个 CUDA Graph 的情况。

单机 SM103 的 BF16 和 NVFP4 推理可在每层加载时准备一次分块权重。分别建立 prefill runtime（`training=False`）和 decode runtime（另加 `decode_only=True`），然后把同一份权重传给二者。

```python
from weftmoe.blackwell import prepare_bf16_weights

w1_tiles, w2_tiles = prepare_bf16_weights(w1, w2)
y_prefill = bf16_pf.inference(0, x_prefill, ids_prefill, gates_prefill, w1_tiles, w2_tiles)
y_decode = bf16_dt.inference(0, x_decode, ids_decode, gates_decode, w1_tiles, w2_tiles)
```

```python
from weftmoe.nvfp4 import prepare_nvfp4_weights

weights = prepare_nvfp4_weights(w1, w2, tiled=True)
y_prefill = nvfp4_pf.inference_nvfp4(0, x_prefill, ids_prefill, gates_prefill, weights)
y_decode = nvfp4_dt.inference_nvfp4(0, x_decode, ids_decode, gates_decode, weights)
```

BF16 分块副本的字节数和原参数相同。如果同时保留训练参数，两份都占显存。NVFP4 可以在量化时直接写出分块格式，两种打包方式的权重与尺度占用相同。更新专家参数后需要重新准备快照。传入原始 BF16 权重可直接使用当前参数。基准的 `--compare-weight-layouts` 配合 `--mode prefill` 或 `--mode decode` 可在相同路由上比较两种布局。部署分块权重时，可在上述 autotune 命令中加 `--tiled-weights` 搜索对应的 M 和调度，并用单独的 `--output` 保存该服务的配置。[权重准备耗时和各档延迟](benchmarks-sm100.md#单机-ep8-decode-专用-kernel)

## 性能

[EP8 decode 的 DT / PF 对照](benchmarks-sm100.md#单机-ep8-decode-专用-kernel) · [各模型 EP8 / EP16 延迟](benchmarks-sm100.md#各模型实测) · [测试环境与复现命令](benchmarks-sm100.md)
