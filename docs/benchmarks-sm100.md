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

## 单机 EP8 decode 专用 kernel

### 跨卡完成信号与权重布局

每格依次为 **上一版 DT / 当前默认 DT / 当前分块权重 DT / 当前 PF** 的端到端 p50，单位 µs。上一版复测使用变更前的 cubin。单机 8 张 B300，固定服务容量 512 tokens/rank，列为每卡实际输入长度。三个 shape 的 `(H, I, top-k)` 分别为 `(2048, 512, 8)`、`(3072, 1024, 8)` 和 `(4096, 2048, 6)`，均为 256 个专家。

| 模型 | 精度 | 1 | 16 | 256 | 512 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | BF16 | 36.43 / 32.99 / 33.15 / 44.31 | 57.39 / 54.75 / 54.36 / 65.83 | 79.08 / 78.93 / 78.15 / 84.82 | 102.79 / 105.31 / 105.26 / 110.90 |
| Qwen3.5-35B-A3B | NVFP4 | 33.16 / 30.90 / 30.93 / 42.27 | 35.18 / 32.95 / 33.07 / 45.25 | 62.31 / 62.41 / 62.60 / 68.81 | 88.70 / 90.45 / 90.56 / 98.97 |
| Qwen3.5-122B-A10B | BF16 | 62.22 / 59.66 / 59.19 / 70.89 | 108.77 / 106.36 / 104.28 / 117.69 | 142.04 / 144.09 / 140.28 / 148.79 | 169.10 / 172.77 / 169.92 / 174.40 |
| Qwen3.5-122B-A10B | NVFP4 | 36.74 / 34.59 / 34.76 / 45.11 | 53.71 / 52.31 / 51.83 / 64.63 | 83.07 / 84.52 / 84.90 / 90.07 | 114.56 / 117.58 / 117.68 / 123.72 |
| DeepSeek V4 Flash | BF16 | 94.46 / 91.49 / 89.47 / 103.80 | 219.42 / 217.03 / 212.47 / 228.72 | 313.28 / 315.62 / 306.15 / 329.87 | 354.55 / 360.38 / 348.48 / 369.81 |
| DeepSeek V4 Flash | NVFP4 | 48.48 / 45.66 / 44.32 / 56.42 | 86.98 / 84.04 / 83.39 / 96.98 | 119.05 / 120.24 / 118.91 / 131.68 | 142.41 / 144.99 / 144.85 / 148.70 |

每组共用同一批合成专家权重、输入和带专家偏置的 top-k 路由。每个样本更新输入与路由，并轮换四条路径的计时顺序。预热 5 个样本，采样 20 次，每个样本连续回放 CUDA Graph 50 次。CUDA event 计入输入准备、通信、量化激活、两层 GEMM 和 combine。先取 8 个 rank 中最大的回放均摊延迟，再对 20 个样本取中位数。权重准备单独计时。主表路由种子 1300，Flash 另用 2808 验证。驱动 580.105.08，CUDA 13.0，PyTorch 2.13.0+cu130，NCCL 2.31.2。[逐样本延迟、路由摘要和两次 profile](../benchmarks/sm100/decode-pipeline-results.csv)

默认 DT 在 1 和 16 tokens 的 12 档上比上一版快 1.1%–9.4%，512 tokens 的六档慢 1.6%–2.6%。当前默认 DT 在全部 24 档都快于当前 PF。Flash BF16 的分块权重在 256 / 512 tokens 时，相对当前默认布局再降 3.0% / 3.3%；在留出路由上为 3.9% / 2.1%。该布局是模型无关的可选项。权重较小或使用 NVFP4 时，两种布局的收益随形状和路由变化。

| 每层每卡准备耗时，ms | Qwen35B | Qwen122B | Flash |
| --- | ---: | ---: | ---: |
| BF16 分块 | 0.331 | 0.925 | 2.383 |
| NVFP4 常规打包 | 0.162 | 0.422 | 1.075 |
| NVFP4 直接打包为分块格式 | 0.173 | 0.429 | 1.111 |

准备耗时为预热后 9 次测量的最慢 rank 中位数。以 Flash BF16 为例，每卡每层的分块副本为 1536 MiB。按主表同版两种布局之差估算，一次准备成本相当于约 252 次 256-token 调用或 200 次 512-token 调用。

下表单独测量 BF16 权重已经在显存时，一层两个专家矩阵的重排时间（ms）。单机 B300 EP8，每卡 32 个专家。每个 shape 用新进程测一次首调，已初始化 CUDA 和通信。每次取最慢 rank，预热后取 21 次中位数，SGLang 的两行取 13 次中位数。

| 操作 | Qwen35B | Qwen122B | Flash |
| --- | ---: | ---: | ---: |
| `prepare_bf16_weights(w1, w2)` 首调，同步墙钟 | 35.11 | 29.28 | 34.22 |
| 预热后，CUDA event | 0.353 | 0.919 | 2.403 |
| 预热后，同步墙钟 | 0.378 | 0.936 | 2.427 |
| SGLang 常规转分块，写回原地址，同步墙钟 | 0.895 | 1.577 | 3.055 |
| SGLang 分块还原为常规布局，同步墙钟 | 0.869 | 1.439 | 2.923 |

SGLang 加载时支付一次正向重排。在线更新已分块的专家时，先还原布局，更新后再重排。常规 prefill 与 decode 请求直接复用同一份分块权重。[接入脚本和更新流程](integration.md)

下表是单机 EP8 按模型主干的实际专家层数连续转换的同步墙钟时间。每层单独分配 BF16 专家权重，输入已在显存，首次在新进程测量，预热后取 11 次中位数。热更新一列只计布局转换。

| 模型 | 专家层数 | SGLang 首次分块 | 预热后分块 | 全量更新时还原并重新分块 |
| --- | ---: | ---: | ---: | ---: |
| Qwen35B | 40 | 62.9 ms | 31.7 ms | 63.2 ms |
| Qwen122B | 48 | 97.7 ms | 69.2 ms | 132.2 ms |
| Flash | 43 | 155.5 ms | 127.1 ms | 250.2 ms |

以 256 tokens/rank 为例，用全层更新的布局耗时除以同版 DT 更换布局后每步预计节省的时间（单层 p50 差值 × 专家层数），上述模型分别约需 2022、723、615 个 decode step 收回一次更新的转换成本。实际步数受路由和 batch 大小影响。Qwen35B 在 512 tokens/rank 的 20 组配对样本中有 9 组分块更快，实测不足以给出可靠的回本步数。

**等待发生在哪里。** 在 Flash BF16 的固定路由上，以设备 `%globaltimer` 记录流水线中各 warp 的等待和任务时长。每个单元依次为上一版默认布局 / 当前分块布局，分母为对应 warp 在该层处理任务的时间，对 8 卡全部 CTA 求和。MMA warp 等待输入就绪，权重 TMA warp 等待可写的流水级。不同角色并发执行。

| tokens/rank | FC1 MMA 等输入 | FC2 MMA 等输入 | FC1 权重等空位 | FC2 权重等空位 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 63.2% / 60.9% | 55.6% / 53.5% | 66.0% / 72.3% | 61.3% / 67.5% |
| 512 | 76.1% / 74.8% | 69.4% / 68.0% | 70.0% / 78.1% | 63.0% / 72.4% |

搜索还覆盖了权重预取距离、BF16 的 K64/K128 搬运、专家和输出块的遍历顺序、小 batch 的 SM 数以及大 batch 的 epilogue 行数。另测了让每个 CTA 统一等待全部 owner 再归并，24 档中有 17 档更慢，保留按 token 等待实际 owner 的实现。分块布局让单次 TMA 读取的权重块在显存中连续，FC1、FC2 均沿用原有的 K 累加顺序。

分块权重也可以作为 autotune 的实际输入。用种子 943 搜索后，在固定容量 512 的工作区上重新捕获图。两组路由各取 25 个样本，每个样本 50 次回放。Flash BF16、256 tokens 从静态 M128 改为动态 M128，分块 DT 延迟分别为 304.12 → 301.13 µs（种子 1300）、303.62 → 297.94 µs（2808）。Qwen35B NVFP4、1 token 从静态 M16 改为动态 M16，分别为 31.06 → 30.86 µs、31.03 → 30.64 µs。两项都只有部分路由达到 1.5% 的确认门槛，主表继续使用通用配置。使用 `tools/autotune_sm103_decode.py --tiled-weights` 可为部署路由单独生成配置。

**训练逐位验证。** 每个模型、精度和长度都使用训练前向作为参考，在四组输入上比较 PF 和分块 DT，覆盖随机路由、全部发往同一专家、重复槽、无效槽及零 gate。另测 BF16 / NVFP4 在 EP1 与 EP8 间更换 batch 和 token 位置，两个方向的训推前向逐位相等。同一 EP 重跑的 dX、dGate、dW1、dW2 也逐位相等。在独立 EP8 测试中，PF 和 DT 分别对同一张 CUDA Graph 连续回放 128 次，每次更换输入、路由、gate 并轮流延迟一个 rank；16 和 512 tokens 的每次输出均与训练逐位相等。

上一版使用的 `%gridid` 会在同一图节点的连续回放中重复。另用 Qwen35B BF16、64 tokens/rank，在两个专家 owner 间逐次切换路由，每次与训练前向比较。上一版 PF 和 DT 各在 64 次中错了 63 次；新版两条路径各为 0/64。上一版的计时验证在相同输入连续回放 50 次后才比对输出，漏掉了这个错误。当前实现给工作区的每次集体调用分配序号，源 rank 等所有 owner 完成 FC2 后再复用输入。上一版的延迟仅供历史比较。

从仓库根目录复现当前三条路径：

```bash
for model in qwen35b qwen122b flash; do
  for precision in bf16 fp4; do
    benchmarks/sm100/run.sh weft --model "$model" --precision "$precision" \
      --mode decode --tokens 1 16 256 512 --decode-capacity 512 \
      --compare-pf --compare-weight-layouts
  done
done
```

### 上一版 DT 与 PF 的历史测量

以下为 B300（SM103）同机配对实测。每格为 **DT / PF** 的 MoE 前向 p50 延迟，单位 µs。列为每卡输入 token 数。两端都通过公开推理接口执行输入准备、路由分发、专家计算和结果归并。固定最大容量为 512 tokens/rank，以覆盖同一服务容量下的四种 batch 大小。

| 模型 | 精度 | 1 | 16 | 256 | 512 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | BF16 | 35.27 / 43.85 | 56.37 / 65.14 | 78.85 / 84.36 | 104.53 / 111.69 |
| Qwen3.5-35B-A3B | NVFP4 | 32.93 / 41.72 | 35.26 / 45.29 | 62.19 / 69.93 | 87.83 / 98.12 |
| Qwen3.5-122B-A10B | BF16 | 62.19 / 71.26 | 109.32 / 118.42 | 143.19 / 149.05 | 168.85 / 173.49 |
| Qwen3.5-122B-A10B | NVFP4 | 36.99 / 45.16 | 55.78 / 64.55 | 82.08 / 88.86 | 115.36 / 123.62 |
| DeepSeek V4 Flash | BF16 | 96.28 / 105.36 | 219.19 / 228.07 | 314.63 / 328.88 | 353.74 / 367.64 |
| DeepSeek V4 Flash | NVFP4 | 48.42 / 56.64 | 87.01 / 96.63 | 118.59 / 130.87 | 142.49 / 148.18 |

每组使用相同输入、专家权重和带专家偏置的 top-k 路由。每个样本更新输入和路由，轮换新版 DT、首版 DT 和 PF 的计时顺序。预热 5 个样本，采样 20 次，每个样本连续回放 CUDA Graph 50 次。CUDA event 测量每次回放的均摊耗时，先取最慢 rank，再对 20 个样本取中位数。每个样本回放后比较 BF16 输出的所有位。NVFP4 权重预先打包，在线激活量化计入耗时。设备驱动 580.105.08，CUDA 13.0，PyTorch 2.13.0+cu130，NCCL 2.31.2。[本轮三方逐样本数据](../benchmarks/sm100/decode-autotune-results.csv) · [首版 DT 的历史测量](../benchmarks/sm100/decode-dt-results.csv)

上一版测量使用的命令如下。

```bash
for model in qwen35b qwen122b flash; do
  for precision in bf16 fp4; do
    benchmarks/sm100/run.sh weft --model "$model" --precision "$precision" \
      --mode decode --tokens 1 16 256 512 --decode-capacity 512 --compare-pf
  done
done
```

### 上一版 profile 与离线搜索

同轮复测的 **新版 DT 延迟（较首版 DT 降低的百分比）** 如下，单位 µs。调参和最终复测使用不同的种子。24 个组合中有 23 个的 p50 降低，Flash BF16 的 256-token 档比首版慢 0.6%。

| 模型 | 精度 | 1 | 16 | 256 | 512 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | BF16 | 35.27（3.6%） | 56.37（3.1%） | 78.85（4.6%） | 104.53（2.2%） |
| Qwen3.5-35B-A3B | NVFP4 | 32.93（6.2%） | 35.26（5.6%） | 62.19（7.3%） | 87.83（10.4%） |
| Qwen3.5-122B-A10B | BF16 | 62.19（2.6%） | 109.32（5.2%） | 143.19（3.4%） | 168.85（2.4%） |
| Qwen3.5-122B-A10B | NVFP4 | 36.99（3.2%） | 55.78（4.5%） | 82.08（6.5%） | 115.36（5.4%） |
| DeepSeek V4 Flash | BF16 | 96.28（0.9%） | 219.19（4.2%） | 314.63（-0.6%） | 353.74（0.8%） |
| DeepSeek V4 Flash | NVFP4 | 48.42（5.7%） | 87.01（3.4%） | 118.59（1.6%） | 142.49（3.7%） |

Nsight Systems 在 rank 0 记录 CUDA Graph 的节点，对固定路由的 10 次回放取中位数。Qwen35B NVFP4 的 1-token 输入准备从两个 kernel 合计 2.94 µs 降到一个 kernel 的 1.50 µs，后续 routed kernel 两次分别为 28.32 和 28.34 µs。Flash BF16 的 512-token routed kernel 耗时 337.35 µs，输入准备 3.26 µs。Nsight 的单 rank 固定路由数据与上表的多路由 EP8 计时分别采集。

另一次插桩回放用设备 `%globaltimer` 记录各 CTA 的启动与路由目录就绪时刻。按 rank 减去最早 CTA 的启动时间，取该 rank 的 CTA 中位数，再取八个 rank 的最大值。Qwen35B NVFP4 的 1-token 目录就绪时间为 8.64 → 7.55 µs。CSV 给出了全部 24 组的两版 profile 与三方端到端采样。

Flash BF16 的 512-token 档从静态 M192 切到动态 M240，512 容量的对称工作区每卡从 411.6 增为 434.1 MiB。另用种子 2808，在实际分配的工作区上交错复测这两套新版 DT 配置，30 个样本、每个样本 50 次回放。512-token 的 p50 为 375.49 / 401.04 µs（M240 / M192），256-token 为 314.00 / 314.45 µs（两者均执行 M128）。

每个形状、精度、长度搜索 12 种 tile M 和调度组合。在 5 个路由样本上交错测量，再用 9 个留出样本确认原配置及两个候选，每个样本回放 40 次。所有候选与保存反向状态的训练前向逐位相等，PF 也使用同一参考输出。搜索另测了同专家重复路由、无效槽与零 gate。选择方法与命令见[单机 decode](blackwell.md#单机-decode)。

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
