# 使用与 autotune

## 编译

需要 Linux、Python ≥3.10、C++17、支持 SM90a 的 CUDA、CUTLASS/CUTE、NCCL，
以及运行 Tensor API 所需的 PyTorch。多机还需要 NCCL device API 与 GIN/GDAKI；
不支持以 Proxy 后端代替。已验证环境包括 CUDA 12.9、NCCL 2.31.2、PyTorch 2.9.1。
PyTorch 和框架依赖由使用者安装，不由 WeftMoE 自动选择版本。

以下在仓库根目录执行，路径按实际安装位置填写：

```bash
export CUDA_HOME=/usr/local/cuda
export CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
export NCCL_INCLUDE_DIR=/path/to/nccl/include
export NCCL_LIB_DIR=/path/to/nccl/lib
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 换模型时改为 qwen122b 或 flash，并重新构建。
export NVCCFLAGS="-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I$PWD/configs/profiles/qwen35b"
python -m pip install -e . --no-build-isolation
python tools/build_tensor_runtime.py --library "$PWD/weftmoe/lib/libweftmoe.so"
export WEFTMOE_TENSOR_EXTENSION="$PWD/weftmoe/lib/_runtime$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
```

包安装构建 `libweftmoe.so`；Tensor 绑定需要上面的第二步。独立产物可用
`python build_native.py lib --output build/native/libweftmoe.so --jobs 4`，再为它构建绑定。
不要混用不同 profile 的绑定与 native 库，也不要覆盖正在运行的库。

## Tensor API

每进程绑定一张 GPU，并初始化显式的 EP process group。各 rank 必须按相同顺序调用算子；
全局专家按连续编号均匀分片。张量均需 CUDA、contiguous，布局如下：

| 张量 | Shape | dtype / 含义 |
| --- | --- | --- |
| `x` | `[T, H]` | BF16，本 rank 的输入 |
| `routes` | `[T, K]` | int32，全局专家 ID；每个 token 的 K 个专家不重复 |
| `gates` | `[T, K]` | FP32，已选路由的权重 |
| `w1` | `[E/EP, 2I, H]` | BF16，前 I 行 gate、后 I 行 up |
| `w2` | `[E/EP, H, I]` | BF16 |

训练片段，假设上述张量和 `ep_group` 已准备好，需梯度的张量已设置 `requires_grad`：

```python
from weftmoe.runtime import TensorRuntime, moe

runtime = TensorRuntime(
    group=ep_group, experts=256, top_k=8, hidden=2048, intermediate=512,
    max_tokens=8192,
    max_rows=8192 * 8 * 2 + (256 // ep_group.size()) * 128,
    training=True, max_saved=64,
)
y = moe(runtime, x, routes, gates, w1, w2)
y.float().square().mean().backward()
# 返回 dX、dGate、dW1、dW2；离散 routes 不求导。
runtime.close()  # 所有 rank 在全部 backward 完成后共同关闭。
```

`max_tokens` 是每 rank 的输入容量；`max_rows` 是接收/peer 缓冲初始容量提示，
不是专家 token 丢弃阈值。训练 metadata 可覆盖最坏路由，saved X/GU 按实际接收行数分配；
热点超过初始 peer 容量时可能扩容。`max_saved` 限制同时存活的训练 tape 数。
固定 workspace 应跨层、跨调用复用，不要每层创建/销毁 runtime。

`moe()` 是同步便利接口。性能接入使用 `forward_async()` / `backward_async()`，
或直接使用 [Megatron adapter](integration.md)；训练提交必须使用 runtime 的准备流。
异步执行不等于取消依赖：stream/event 负责 workspace 复用，调用持有者保留输入与缓存。

推理创建 `training=False` 的 runtime，默认使用 PF；设置 `decode_only=True` 显式选择 DT。
DT 要求单机、纯推理、编译形状匹配、EP world 不超过 32，整个 pool 容量可由单个 source
window 容纳。Python 接口会将窗口设为 `max_tokens`，没有固定的 64-token 上限；各 EP rank
必须使用一致配置。BF16 和 FP8 DT 的所有 token shape 固定使用 M64，
不再提供 M16 路径、切换阈值或 M 维度调参。使用 `inference_eager(..., layer=layer_id)` 执行 eager 推理。
CUDA Graph 接口为 `inference(layer_id, ...)`，
先在 capture 外预热，并保持图使用的地址不变。设置 `layer` 会缓存权重 descriptor 并保留原张量：
BF16 原地更新完成后可直接继续推理，无需重排或刷新副本；更换权重地址时先丢弃外部图并
`invalidate_inference_layer()`，再预热、重新 capture。这些操作需所有 EP rank 一致执行。

## Block-wise FP8

SM90 PF 路径支持 E4M3FN 前向、输入梯度和权重梯度；单机 DT 支持 FP8 推理前向。
权重采用每个专家独立的
`128×128` block scale；激活和反向梯度在线采用 `1×128` scale。每个 K128
片段的 FP8 WGMMA 结果乘对应 scale 后累加到 FP32。单机及多机 GIN 的 N256
路径都在源端独立量化 X/dY，Dispatch 搬运 FP8 数据及 FP32 scale；Return、输出
和 dX 仍为 BF16，路由梯度为 FP32。训练保存目的端 row FP8 X 和 FP8 AB，
不分配或写回 BF16 X/AB checkpoint。Wgrad 的 X/dR 从已有行向
FP8 和 scale 重数量化为列向 FP8，Z/dAB 则在 SwiGLU 反向产生时直接从 BF16
列量化。row→col 会重新选取 scale 并再次舍入，不能视为无损转置，也不等价于
从原始 BF16 直接列量化。量化本身不求导，反向使用对应的 FP8 GEMM 近似梯度。

FP8 相对原始 BF16 的验收采用全局指标
`diff = sum((actual-reference)^2) / (sum(actual^2)+sum(reference^2))`，
要求 `diff < 5e-3`，每个 rank 单独检查，dW 再逐 expert 检查；dprob 仅要求有限。
不使用逐元素相对容差。布局、图回放、预量化输入等数学等价路径仍要求精确一致。
训练 PF 与固定 M64 DT 的前向另设严格逐位回归：三个模型、普通及紧凑 FP8 workspace，
3600 个 rank-case 全部一致，每份训练 tape 均实际执行反向。
完整 DT 回归的 816 项中保留此前 Qwen35B 单 token 的已知误差豁免；该豁免不适用于
训推逐位一致性测试。测试范围和成对性能数据见
[固定 M64 验证报告](benchmarks/dt_m64_20261101.json)。

构建方式与上面的 BF16 Tensor API 相同，无需安装 DeepGEMM。`H/I` 必须是 128
的倍数，并匹配构建 profile。先在 forward/backward 外准备权重，再显式传给 `moe`：

```python
from weftmoe.fp8 import prepare_fp8_weights
from weftmoe.runtime import moe

# runtime 使用 training=True, fp8_workspace=True；w1/w2 是 BF16 或 FP32 可训练参数。
prepared = prepare_fp8_weights(w1, w2, training=True, version=step)
y = moe(runtime, x, routes, gates, w1, w2, fp8_weights=prepared)
loss = y.float().square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
# 参数更新后重新 prepare，不能重复使用旧 prepared。
```

准备过程产生不可变的前向及 Dgrad 转置副本；已有 tape 保留对应旧版本，新的权重版本
可以并存。不要原地修改其中的数据或 scale；常规 PyTorch 原地写入和过期参数缓存会被
拒绝。autograd 在主参数均为 BF16 时直接输出 BF16 dW，避免 FP32 中间输出和额外转换；
累加仍为 FP32。底层接口默认输出 FP32 dW，适合接入 FP32 `main_grad`：

```python
tape = runtime.forward_fp8(x, routes, gates, prepared, synchronize=False)
dx, dgates, dw1, dw2 = runtime.backward_async(tape, dy)
# dw1/dw2 为 FP32，调用者可在同一 stream 累加到 main_grad。
runtime.drain()  # 仅在需要等待或释放 runtime 时调用。
# 若只需 BF16 dW，可在 forward_fp8 中指定 weight_grad_dtype=torch.bfloat16。
```

`prepare_fp8_weights` 也接受外部预量化的 `FP8Tensor`，只复制其 FP8 值和 scale，
不重新量化。布局为 contiguous `data[E,M,K]`、`scales[E,M/128,K/128]`，dtype
分别为 `torch.float8_e4m3fn` 和 `torch.float32`；scale 为正、有限的反量化乘数，
不要求是 2 的幂。例如 `FP8Tensor(qw1, scale1, tuple(qw1.shape), block_rows=128)`。
W1 仍使用 `[gate, up]` 规范行序，禁止先交错 gate/up 再划量化块。外部 checkpoint
与训练主参数的对应关系由调用者保证。自行量化时使用当前 block amax，scale 向上取到
2 的幂；全零块 scale 为 1，尾部补零且不跨专家。

预量化激活使用 rowwise `FP8Tensor`：`data[tokens,H]` 为 E4M3FN，
`scales[tokens,H/128]` 为 FP32 正有限 2 的幂，`block_rows=1`。
PF/DT 的 `forward_fp8` 均接受此格式；FP8 tape 的 `backward` 和
`backward_async` 也接受相同格式的 dY。源端直接复制数据和 scale 到通信 packet，
跳过 amax 和重新量化；量化器及输入准备耗时由调用者决定是否计入端到端测量。

```python
from weftmoe.fp8 import quantize_fp8

qx, qdy = quantize_fp8(x), quantize_fp8(dy)
tape = runtime.forward_fp8(qx, routes, gates, prepared,
                           weight_grad_dtype=torch.bfloat16)
dx, dgates, dw1, dw2 = runtime.backward(tape, qdy)
```

纯推理使用 `training=False` 的 runtime 和
`prepare_fp8_weights(..., training=False)`。Eager 调用
`forward_fp8(..., retain_backward_state=False)`；CUDA Graph 调用
`inference_fp8(layer_id, x_or_fp8_tensor, routes, gates, prepared)`。
每个 layer/token bucket 必须在外部 capture 前预热，保持 data、scale、routes、gates
地址不变。更换权重或输入地址时，先丢弃外部图并 `invalidate_inference_layer()`，
然后重新预热和 capture。Megatron/SGLang adapter 尚不自动选择 FP8。
PF FP8 使用 N256 和源端量化配置；原来的多机 BF16 通信 + 目的端 FP8 量化路径已删除。
DT 在同一个 kernel 中让 math warps 量化或复制源输入，control warps 执行 mini notify，
现有跨 rank barrier 等待两者完成后发布数据。FP8 M64 使用两个 math warpgroup，
各自计算 N128；每个 K128 块的四次 WGMMA 后做 scale 累加。

PF N256 的 X 由独立 `prepare_source_x` kernel 按源 token 量化，结果跨 top-k
路由和 N tile 复用。每行 packet 包含 `FP8[H] + FP32[H/128]`，pitch 向上对齐
到 16 字节。`TensorRuntime(..., fp8_workspace=True)` 按此大小分配源输入区；
默认 runtime 保留 BF16 容量以支持混用。Notify 不再拷贝 BF16 X。
单机 PF 的源端量化/复制与 planner 分别在两个 stream 执行；本地 event 汇合后，
`source_ready_kernel` 通过跨 rank barrier 发布所有源 packet，再允许主 kernel Pull。
多机 GIN 保持同一 transport/preparation stream 上的有序执行。
Pull 只搬运数据和 scale，并写入现有 ring 布局。源 packet 的复用受已有 Forward
跨 rank 完成协议保护；目的端 row FP8 X 随各自 tape 独立保留。原始 BF16 X 无需保存。
反向对 dY 使用同样的量化或直接复制方案：单机源准备与路由计划恢复并行，汇合并发布后
执行 BWD；多机 GIN 顺序恢复计划及准备源数据。反向 Pull 只搬运 FP8 + scale。
Z/dAB 的量化仍在 SwiGLU epilogue 中。
PF N256 路径在每个 M128 ring slot 内紧凑存放 `FP8[M128,C]` 与 `FP32 scale[C/128,M128]`。
启用 `fp8_workspace=True` 时，slot 跨度为 `128 * align16(C + 4*C/128)` 字节，
否则沿用 BF16 容量。每个 K128 的 128 个行 scale 连续，可以和操作数一起用 TMA 异步搬运。
X/dR 与 dAB 的通道数即使不同，写入也只覆盖当前 slot；不能把整个 ring 全局压紧，
否则复用 dR 存储时可能覆盖其他尚未退休的 slot。GIN 的源输入区及 Dispatch
收包区同样支持紧凑 FP8，Return 区仍为 BF16；总 workspace 并非整体减半。
前向及 Dgrad 复用多级 producer/math 流水线，双 CTA cluster multicast
共享激活；W1 的 5D TMA descriptor 直接
描述规范 gate/up 权重，无需额外预打包权重副本。math 使用 N128 WGMMA 子块与
N256 最终累加器，以控制寄存器用量；math/control 配额固定为 224/56，不按模型或 shape 调整。
单机 N256 训练路径由指定 FC1 N tile 保存目的端 row FP8 X；GIN 由 Pull 在
搬运时按专家的 canonical 行号保存，连同 gate 和反向索引一起写入独立 tape。BWD 中指定 dU
N tile 异步读取对应 X 块，在 dU 计算后完成一次 row→column 重量化，写入
tape 的同一块全局存储；所有 dW1 CTA 复用这一份结果。K128 块的所有权分散到不同 N tile，
不在每个 dW GEMM 内重复量化。dR 列缓存仍由指定 dU tile 生成。
首次 BWD 将 X checkpoint 原地改为列布局，后续原生 BWD 复用列缓存；
布局状态在成功 enqueue 后记录，原有执行流依赖确保读者等待转换完成。
Python 的 backward 仍按原接口消费 tape。
AB 采用两个独立的 row 1×128 FP8 平面和 scale；FWD 的 SwiGLU 也消费这份 FP8
反量化值，BWD 用相同值重算 Z 和导数，避免前反向的激活不一致。dU 保留 BF16，
但只分配 `ring_depth * 128 * I` 元素。dGate 的特征部分和也使用同一 ring 生命周期。
SwiGLU 反向按 M128×I128 分给 CTA，直接
写出 Z/dAB 列缓存与 dAB 行缓存；最后完成的特征 tile 合并 dGate 后发布 dAB。
新路径也省去不再被消费的 BF16 dR 写回；GIN 保存 canonical row FP8 dR，
让 dU 和 dW2 共用同一次源端量化。单机 FWD 调度 wave 按约 256 个 FC1 tile
限制长度；BWD 超过 128 个 M block 时，按约 128 个 dU tile 限制长度。
三个 profile 对应 64/32/16 个 M block；小工作量 BWD 保留 128 block。
反向切换 dU、SwiGLU、dX 阶段时复用当前 wave 的首个专家游标。
列向 amax 使用 BF16 成对整数归约，输出直接采用 Wgrad 的 K-major swizzle。
因此无需在 Dgrad 后额外读取四个 BF16 张量、集中构建列缓存和进行一次全 grid 同步。
SwiGLU 的每个 warp 处理 16 行、每 lane 处理 4 个连续通道，再通过共享内存归约列
scale，以改善 BF16 读取和 FP8 行写出的连续性。Wgrad 让原有 producer warp
继续工作，双缓冲 TMA 同时搬运数据与连续的列 scale；各 stage 独立使用 full/empty
barrier，math warpgroup 在每个 K128 完成后归还 stage，不再逐 K128 汇合整个 math CTA。
输出 tile 不与 operand stage 重叠，继续使用独立 CTA 调度和 BF16 TMA 梯度输出。
`WEFTMOE_FP8_FWD_WAVE_BLOCKS` / `WEFTMOE_FP8_BWD_WAVE_BLOCKS` 默认为 0（按上述
列数选择），可在编译时覆盖；`WEFTMOE_FP8_WGRAD_PIPELINE=0` 可回到旧 Wgrad 流水线。
Wgrad 默认一次领取 4 个 tile；qwen35b profile 将 `WEFTMOE_FP8_WGRAD_CLAIM_TILES`
设为 2，以减少较大 token 数时的任务尾部等待。GEMM 复用 descriptor 和部分累加器，TMA 输入不再执行普通
共享内存写入才需要的 proxy fence，短尾块跳过无有效行的 math warpgroup。

这些 FP8 缓存统一按 compute ring 的 token 顺序排列：同一专家内轮流取各 source
rank 的下一条路由，已耗尽的 source 跳过。AB checkpoint 仍按 source rank
连续排列，通过 `checkpoint.rows` 映射读取。dW1 的 dAB/X、dW2 的 dR/Z 必须使用
同一顺序；每 128 token 的分组随之确定，不能只重排其中一侧或只转置 scale。
gate/up 的特征交错是另一维度：Wgrad 缓存仍使用规范 `[gate, up]` 列序。
多机 GIN 和其他 N tile 配置继续使用原有 BF16→列向 FP8 缓存构建路径。
编译时设置 `WEFTMOE_FP8_SOURCE_X=0` 可对照原 Pull 量化实现；关闭
`WEFTMOE_FP8_STREAMED_WGRAD` 也会恢复 BF16 X checkpoint。
`WEFTMOE_FP8_SOURCE_DY=0` 可单独恢复反向 Pull 量化；反向源端量化要求
单机 N256、packed ring 和列向 FP8 缓存。`fp8_workspace=True` 只支持这条路径，
拒绝 BF16 调用；默认 `False` 仍支持 BF16/FP8 共用 runtime。紧凑路径缩减源输入和
X/S ring，反向不再分配未使用的全量 BF16 dR/dAB。BF16 输出、dX、dW 和贡献归约区
仍保留，因此整个 workspace 不会简单减半。
自定义 C++ `tensor_factory` 需遵循后端传入的 `ForwardMegaKernelConfig`：
`fp8_x_checkpoint=true` 时不得绑定 BF16 `fc1_input`；`fp8_ab_checkpoint=true` 时
不得绑定 BF16 `fc1_output`。BF16 checkpoint 验证下载对这两块分别返回空数组。
单机路径的 BWD 临时 Wgrad 缓存约为 `R_padded * (H + 3I) * (1 + 4/128)` 字节；tape 保留
`R_padded * H * (1 + 4/128)` 字节的 X（FWD row、首次 BWD 后 column）和 `2*R*I*(1 + 4/128)` 字节的 AB，
另加路由、输出和对齐开销。`R_padded` 是每个本地专家 token 数分别向上对齐到 128 后的总和。
权重仍预存 W1/W2 及转置副本。训练时从 BF16/FP32 准备权重，每个 W 使用一个
kernel 同时输出 W、Wᵀ 及两份 scale；每个 128×128 block 只读取一次高精度输入、
计算一次 amax，并复用同一份 FP8 量化字节。已量化 checkpoint 仍按原值复制和转置，
支持非二次幂的外部 scale。`training=False` 仅准备正向布局。
FP8 benchmark 的权重量化与 W/Wᵀ 输出全部在计时外，源端激活在线量化仍计时；
JSON 中以 `weight_preparation_in_timing=false` 和
`fp8_weight_preparation="fused_quantize_transpose"` 标明。
独立准备耗时诊断可运行 `benchmarks/torchrun_fp8_weight_preparation.py --output <json>`，
沿用 torchrun 和扩展环境变量。tile 转置加载候选的历史测试记录见下文。

实现参考 [DeepGEMM SM90 的 producer/math 流水线与 FP32 promotion](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh)。
[FlashMLA Hopper FP8 文档](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md)
中的 cluster 分担数据处理思路可参考，但该文的 FP8 KV 会解量化为 BF16 计算，
不能直接视为本实现 FP8 GEMM 的性能依据。

验证命令（扩展路径先按上文配置）：

```bash
pytest -q tests/test_fp8.py
NCCL_DEBUG=WARN TEST_TOKENS=257 torchrun --standalone --nproc-per-node=2 tests/torchrun_fp8.py
# 可选：空输入 rank；TEST_HIDDEN/TEST_INTERMEDIATE 可指定其他构建 profile 的形状。
TEST_TOKENS=1 TEST_ZERO_RANK=1 torchrun --standalone --nproc-per-node=2 tests/torchrun_fp8.py
# 权重准备不计入计时；测试固定权重的前向+反向，不含 optimizer.step。
torchrun --standalone --nproc-per-node=2 benchmarks/torchrun_training.py \
  --backend pf --fp8 --weight-grad-dtype bf16 --model qwen35b \
  --tokens 256 512 2048 8192 --stream-ordered --warmups 5 --samples 20
```

`tests/cuda/fp8_column.cu` 可按文件头命令单独编译，验证行向 FP8→列向 FP8 的
字节值、scale、swizzle、带污染数据的尾部、BF16 极小值和 NaN 传播。额外压力测试可设置
`TEST_EXPERTS=32 TEST_ROW_SCALES=1 TEST_TOKENS=257`。

基础测试对照独立反量化/FP64 参考；分布式测试用六次独立 FP8 GEMM 对照完整前反向，
覆盖不等长 rank、空专家、热点路由、K128 尾部、外部 scale、异步多版本、优化器更新
和纯推理。

以下是 2026-10-27 列缓存融合前的历史结果。在双 NVIDIA L20X（SM90）、CUDA 12.9、PyTorch 2.9.1 上的最终对照：
256 专家、top-8，5 次预热、20 次样本，取每个样本较慢 rank 的前反向总时延中位数。
两个版本使用相同 seed、路由和 token 序列，并均输出 BF16 dW；所有 benchmark
在其他 GPU 测试结束后顺序执行。token 是每 rank 的标称值，实际长度有 10% 标准差。
Qwen35B 的 H/I 为 2048/512，Qwen122B 为 3072/1024。

| Profile | token/rank | FP8 (ms) | BF16 (ms) | FP8 耗时变化 |
|---|---:|---:|---:|---:|
| qwen35b | 256 | 1.768 | 2.039 | -13.3% |
| qwen35b | 512 | 1.977 | 2.211 | -10.6% |
| qwen35b | 2048 | 2.976 | 2.931 | +1.5% |
| qwen35b | 8192 | 7.296 | 5.644 | +29.3% |
| qwen122b | 2048 | 5.864 | 6.378 | -8.1% |

**尚未达到所有场景不慢于 BF16。** 小 token 场景和 Qwen122B/2048 已获得加速，
Qwen35B/2048 接近持平，Qwen35B/8192 仍慢约 29%。同一 Qwen35B/2048 workload
使用底层默认 FP32 dW 时为 **3.198 ms**；不要将不同梯度输出类型的数字混为一谈。

这组历史测量中，FP8 排除一次性权重准备，BF16 包含当时的默认权重准备路径；两者均不包含 optimizer
及参数更新后的 FP8 重新量化。因此这不是完整训练 step 的性能结论。
当时的剩余瓶颈包括大 token 下的 Wgrad 列向量化缓存构建，以及 Dgrad 的 FP32 scale
累加/同步开销。N64 子块流水线和更高 math 寄存器配额在本机未改善总时延，未默认启用。
原始记录、命令、workload hash、源码 hash 和验证摘要见
[性能结果 JSON](benchmarks/fp8_blockwise_20261027.json)。

### 2026-10-28：Wgrad 列缓存融合

以下为列缓存融合完成时的历史结果，尚未应用下一节的 scale 布局和流水线优化。
硬件、E256/top-8、BF16 dW 和 5 次预热／20 次采样口径同上；逐项检查了旧 FP8、
新 FP8、BF16 的 workload hash、token 数和收到的路由数一致。该版本采用
row→col 与 SwiGLU 列缓存路径。

| Profile | token/rank | 旧 FP8 (ms) | 新 FP8 (ms) | BF16 (ms) | 新 FP8 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| qwen35b | 256 | 1.774 | 1.807 | 2.092 | -13.6% |
| qwen35b | 512 | 2.022 | 2.075 | 2.235 | -7.2% |
| qwen35b | 2048 | 2.995 | 3.020 | 2.957 | +2.1% |
| qwen35b | 8192 | 7.377 | 7.021 | 5.645 | +24.4% |
| qwen122b | 2048 | 5.876 | 5.873 | 6.399 | -8.2% |

Qwen35B／8192 相对旧 FP8 降低约 **4.8%**；小中 batch 相对旧 FP8 有约
**0.8–2.6%** 的开销，Qwen122B／2048 基本持平。因此这次修改实现了缓存复用和
大 batch 的一定改善，**仍未达到所有场景不慢于 BF16**：Qwen35B／2048 慢约
2.1%，8192 慢约 24.4%。

最终版本通过 EP2／EP4、不等长 source rank、32 专家及空专家、空输入 rank、
257／8193 token、行 scale 差异与下溢、多版本异步 tape、BF16/FP32 dW 输出、
外部权重 scale、优化器版本检查和推理测试；Qwen35B 与 Qwen122B 均通过，
原 BF16 Tensor API 回归通过。独立转换测试 36 组、FP8 Python 单测 37 项通过。
另修复了 Wgrad 任务 mailbox 在空专家快速跳过时缺少同步的问题。

可用二进制分别位于 `build/fp8-column-final/` 和 `build/fp8-column-qwen122b/`。
原始逐样本数据、完整命令、workload hash、源码／二进制 hash 和验证记录见
[列缓存性能记录](benchmarks/fp8_column_cache_20261028.json)。多机路径仍需要实际运行验证。

### 2026-10-28：scale 搬运与 GEMM/SwiGLU 优化

本轮采用上文的 slot 内紧凑布局、scale TMA、SwiGLU 行布局和 GEMM 内循环优化，
不增加全局缓存容量。前向、Dgrad、Wgrad 均执行 FP8 WGMMA，保持每 K128 的
FP32 累加及 BF16 dW。以下重新顺序测量了上一节的列缓存版本、当前版本和 BF16；
两张 L20X、E256/top-8、5 次预热、20 次采样，workload hash、token 和路由数量逐项一致。
最终构建不包含诊断计时插桩。

| Profile | token/rank | 列缓存版 FP8 (ms) | 当前 FP8 (ms) | BF16 (ms) | 当前 FP8 耗时相对 BF16 |
|---|---:|---:|---:|---:|---:|
| qwen35b | 256 | 1.842 | 1.652 | 2.049 | -19.4% |
| qwen35b | 512 | 2.035 | 1.832 | 2.204 | -16.9% |
| qwen35b | 2048 | 2.984 | 2.744 | 2.915 | -5.9% |
| qwen35b | 8192 | 7.062 | 6.446 | 5.677 | +13.6% |
| qwen122b | 2048 | 5.886 | 5.397 | 6.410 | -15.8% |

所有测量形状相对上一版 FP8 的时延降低约 **8.0–10.3%**。
Qwen35B/2048 从略慢于 BF16 变为时延降低约 **5.9%**；Qwen122B/2048 降低约
**15.8%**。但 **Qwen35B/8192 仍比 BF16 慢 13.6%**，尚未达到所有形状不慢于
BF16 的目标，这个形状仍应选择 BF16。同一当前构建的 BF16/8192 为 5.651 ms，
与表中的旧构建 BF16 5.677 ms 接近。
这些数字仍不包含 optimizer 和参数更新后的权重量化，不能作为完整训练 step 的加速比例。

当前两个 profile 的数值测试、EP2/EP4、257/8193 token 尾部、空 rank、32 专家、
外部 scale、异步多版本 tape、BF16/FP32 dW、推理及 BF16 API 回归均通过。
关闭列缓存融合后的备用路径也通过；FP8 Python 单测 37 项、独立转换测试 36 组通过。
GIN 多机路径仍仅编译，未在真实多机环境运行。

没有保留无收益的更小 Dgrad/Wgrad tile、N64 交替片段、dU/SwiGLU 融合和跨 K
双累加器方案；跨 K 方案在本次 CUDA 12.9 构建中出现 ptxas C7514/C7517 等待与
串行化提示，实测变慢。转换移到 GEMM 后的试验未通过 dW2 数值校验，也已撤回。

当前二进制为 `build/fp8-perf-final/` 与 `build/fp8-perf-final-qwen122b/`。
完整命令、逐样本数据、workload/源码/二进制 hash、验证和试验记录见
[本轮性能记录](benchmarks/fp8_performance_20261028.json)。

### 2026-10-28：源端 X 量化，移除 BF16 X checkpoint

单机 N256 路径现在在 Notify 前运行独立源端量化 kernel，每个 token 只量化一次。
源端输出 `FP8[H] + FP32[H/128]` packet，跨 top-k 路由复用；Pull 只做搬运和
ring 布局转换。BF16 X checkpoint 不分配、不传输、不写回，dW1 继续消费 tape
中的列向 FP8 X，dW 默认输出 BF16。AB checkpoint 仍为 BF16。
本次 H=2048/3072 时，X packet 字节数是 BF16 X 的 51.5625%；这不代表整个
MoE 通信量减半，因为 Return 等路径仍使用 BF16。源 packet 复用原源输入池，
不新增该缓冲容量；每个 tape 另省去 `received_row_capacity * H * 2` 字节的 BF16 X。

以下只比较 **训练 FWD**，包括源端在线量化、训练缓存构建、tape 释放和设备同步，
不含 BWD、optimizer 和预先进行的权重量化。两张 L20X、E256/top-8、随机变长
token/路由；20 次预热、50 次采样，反序再测一次。表中是合并 100 个样本的
最慢 rank 时延中位数，所有版本的 workload hash、token 和接收行数一致。

| Profile | nominal token/rank | 旧 Pull 量化 FP8 (ms) | 源端量化 FP8 (ms) | BF16 (ms) | 相对 BF16 加速比 |
|---|---:|---:|---:|---:|---:|
| qwen35b | 256 | 0.647 | 0.625 | 0.803 | 1.28× |
| qwen35b | 2048 | 1.170 | 1.039 | 1.253 | 1.21× |
| qwen35b | 8192 | 2.842 | 2.384 | 2.497 | 1.05× |
| qwen122b | 2048 | 1.862 | 1.488 | 2.197 | 1.48× |

四个已测形状的 FWD 均快于 BF16。此前较慢的 qwen35b/8192 相对旧 FP8
时延降低 16.1%，相对 BF16 降低 4.6%。BF16 预准备权重对照在 256/8192
分别为 0.801/2.513 ms，未改变结论。不能由这些 FWD 数字推断完整训练 step
的加速，也不保证其他形状、GPU 或多机环境有同样收益。

试过让源端量化与 Notify 分 stream 执行，但小负载测到约 0.20 ms 的额外主机
提交开销，且 profiler 未显示有效的量化/Notify 重叠，因此撤回。最终实现按
同一 stream 顺序执行，不声称隐藏了量化时延；收益来自量化复用、减少 X 的
搬运字节和省去 BF16 X 写回。GEMM 及原有行转列缓存算法保持不变。

最终构建通过 EP2/EP4、8193 token 尾部、空输入 rank、两个 H/I profile 的
完整前反向参考校验、异步多版本 tape、BF16/FP32 dW、外部 scale 和推理验证，
并断言 `Tape.saved_x_bytes == 0`。BF16 tensor runtime 回归、37 项 Python
单测和 51 组独立源 packet 字节/scale/尾部测试通过。GIN 多机仍未实机验证。

构建位于 `build/fp8-source-final/` 和 `build/fp8-source-final-qwen122b/`，按 profile
设置 `WEFTMOE_TENSOR_EXTENSION` 指向其中的 `_runtime.cpython-312-x86_64-linux-gnu.so`。
编译选项 `WEFTMOE_FP8_SOURCE_X=0` 可恢复旧路径。
原始样本、命令、验证输出、源码/二进制 hash 和被撤回的异步试验见
[源端量化性能记录](benchmarks/fp8_source_quantization_20261028.json)。

### 2026-10-28：反向 dY 源端量化

单机 N256、packed ring、列向 FP8 缓存路径默认启用 `WEFTMOE_FP8_SOURCE_DY`。
反向在等待 dY producer 和调用方 stream 后，直接向对称源输入区量化 dY。
量化、路由计划恢复和 BWD 主 kernel 使用原有 transport stream 顺序执行，
没有新增 stream 或跨 stream 事件，也没有在 GEMM 内量化。dR 的列向 FP8
缓存及 Wgrad 算法保持不变，默认输出 BF16 dW。编译时设置
`WEFTMOE_FP8_SOURCE_DY=0` 可单独关闭这项优化。

下表测量完整 **BWD 调用 + 释放 + 设备同步**，包括新量化 kernel、路由计划和
输出处理。每次先执行并等待新的 FWD 完成，再开始 BWD 计时；固定权重的
预量化及 optimizer 不在计时范围。两张 L20X、E256/top-8，20 次预热、
50 次采样，反序再测一轮；合并 100 个样本，取每次最慢 rank 时延的中位数。
三种配置的 workload hash、token 数和接收行数完全一致，dW 均为 BF16。
这里的“旧 FP8”已经使用源端 X 量化，仅 dY 仍在 Pull 中量化。

| Profile | nominal token/rank | 旧 FP8 BWD (ms) | 源端 dY FP8 BWD (ms) | BF16 BWD (ms) | 相对旧 FP8 时延变化 |
|---|---:|---:|---:|---:|---:|
| qwen35b | 256 | 1.206 | 1.213 | 1.319 | +0.5% |
| qwen35b | 2048 | 1.762 | 1.735 | 1.814 | -1.6% |
| qwen35b | 8192 | 3.821 | 3.560 | 3.337 | -6.8% |
| qwen122b | 2048 | 3.704 | 3.555 | 4.338 | -4.0% |

小负载基本持平；三个测点快于 BF16，但 qwen35b/8192 的 BWD 仍比 BF16
慢 6.7%，不能声称所有形状都达到 BF16 性能。dY packet 在本次 H=2048/3072
下占原 BF16 字节数的 51.5625%，对称输入区的实际分配容量未缩小，因为
同一个 runtime 仍支持 BF16/FP8 混用和备用路径。

另对 qwen35b/8192 测了同样两轮的 FWD+BWD 总时延：旧 FP8 5.952 ms、
新 FP8 5.714 ms、BF16 5.733 ms。新 FP8 相对旧版降低 4.0%，与 BF16
基本持平（差约 0.3%，不足以声称有明确优势）。总调用允许正常的异步提交，
不能由独立 FWD/BWD 计时直接相加得到；仍不包含 optimizer 和权重预量化。

两个 profile 的完整数值测试、EP2/EP4、8193 token 尾部、空 rank、行 scale
压力、外部权重 scale、异步多版本 tape 和 BF16/FP32 dW 均通过。
BF16 tensor runtime 回归及 37 项 Python 单测、51 组源 packet 测试通过。
同时修复 benchmark 在 Backward 消费 tape 后读取 checkpoint 属性的问题，
该属性现在在 Backward 前读取，仅用于第一次预热的结果标签。

当前构建位于 `build/fp8-source-dy/` 和 `build/fp8-source-dy-qwen122b/`。
原始数据、命令、验证和源码/二进制 hash 见
[反向源端量化记录](benchmarks/fp8_source_dy_20261028.json)。

### 2026-10-28：三个规模的 8K/20K 调度与 Wgrad 优化

本轮保留 Wgrad producer/math 双缓冲流水线、按 GEMM 列数限制调度 wave、
BWD wave 专家游标复用，以及 qwen35b 的每次 2-tile Wgrad 领取粒度。
源端 X/dY 在线量化、列向缓存算法、K128 的 FP32 promotion 和 BF16 dW 保持原有精度约定。
没有增加 GEMM 内量化，也没有增加原有动态共享内存或全局缓存的分配容量。

**目标尚未全部达标：五个负载的 FWD/BWD 均重复测得收益，qwen35b/20K 仍基本持平。**
两张 NVIDIA L20X（SM90，132 SM）、CUDA 12.9、EP2；三个 H/I 分别为
2048/512、3072/1024、4096/2048，E256，两个 Qwen profile 为 top-8，Flash 为 top-6。
8K/20K 指每 rank 的 nominal token 数，实际 token 和路由按同一确定性变长 trace 变化。
每项 20 次预热、50 次采样，第二轮反转 FP8/BF16 顺序；测量时没有并行编译或其他测试。
下面取合并 100 个样本中每次最慢 rank 的时延中位数。逐项确认 workload hash、
token、接收行数、精度和形状一致；括号内为 BF16 / FP8 的加速比，大于 1 表示 FP8 更快。

| Profile | nominal token/rank | FWD：BF16 / FP8 ms | BWD：BF16 / FP8 ms | FWD+BWD：BF16 / FP8 ms |
|---|---:|---:|---:|---:|
| qwen35b | 8192 | 2.495 / 2.224（1.122×） | 3.340 / 3.257（1.025×） | 5.713 / 5.262（1.086×） |
| qwen35b | 20480 | 4.738 / 4.769（0.994×） | 6.619 / 6.628（0.999×） | 11.179 / 11.170（1.001×） |
| qwen122b | 8192 | 4.528 / 3.413（1.327×） | 8.085 / 6.666（1.213×） | 12.535 / 9.893（1.267×） |
| qwen122b | 20480 | 8.807 / 7.536（1.169×） | 15.496 / 14.783（1.048×） | 24.206 / 21.979（1.101×） |
| flash | 8192 | 7.195 / 5.095（1.412×） | 14.113 / 11.746（1.201×） | 21.278 / 16.635（1.279×） |
| flash | 20480 | 14.402 / 11.399（1.263×） | 29.489 / 24.567（1.200×） | 44.161 / 35.594（1.241×） |

qwen35b/20K 的 FWD 慢约 0.65%、BWD 慢约 0.14%，总调用快约 0.08%，这些差异不足以
构成使用 FP8 的性能理由。其他五个负载的总调用时延降低约 7.9%～21.8%；
两轮各自的 FWD/BWD 加速方向均一致，qwen35b/8K 的 BWD 收益仍较小（约 2.5%）。

FWD 包含在线源端量化、训练缓存写出、释放和同步；BWD 在新建并同步的 tape 上单独计时，
包含 dY 量化、路由计划和输出处理。总调用单独测量，不能将两个独立阶段的中位数相加。
FP8 权重预量化在计时外；BF16 沿用常规训练 API 的权重准备。另补测六个负载的同步
预准备 BF16 权重 FWD，对照结果接近，未改变结论。这里不包含 optimizer 及参数更新后的
重新量化，不能当作完整训练 step 的收益。

未保留的试验包括：成对 CTA 拆分列缓存、整数解码、向量直接 Pull、3-stage Wgrad、
qwen35b 的 32/128/66 block wave、单 tile Wgrad 领取，以及进入 Wgrad 后重新分配
224/56 math/control 寄存器并采用 FWD wave 33。最后一项通过数值校验，但没有改善
目标 20K 负载。这些试验配置均未采用，额外的转换、Pull 与寄存器重分配代码已撤回。
原始筛选结果保留在下方记录中。

最终三个 profile 均通过 8193-token 完整前反向参考校验和行 scale 压力测试；
EP4/257 token、空 rank、异步多版本 tape、BF16/FP32 dW、外部 scale、optimizer 与
推理验证均通过。BF16 TensorRuntime 回归、37 项 Python 单测、51 组源端 packet
测试、36 组列转换测试及 `git diff --check` 通过。GIN 多机路径仅编译，未实机验证。

最终库和绑定位于 `build/fp8-opt3-final-{qwen35b,qwen122b,flash}/`。
按前文编译命令选择对应 profile 即可复现；`WEFTMOE_TENSOR_EXTENSION` 指向对应目录的
`_runtime.cpython-312-x86_64-linux-gnu.so`。完整命令、逐样本数据、两轮结果、构建/源码
hash、数值验证与撤回试验见 [六个负载的性能记录](benchmarks/fp8_six_shapes_20261028.json)。

## Autotune

一个 [配置](../configs/autotune.json) 描述 workload、拓扑、搜索空间和验收流程。
默认读取 `configs/profiles/<model>/` 作为起点，自动编译、筛选、反序确认、校验数值并导出结果。
这是分阶段离线搜索，不是运行时调参，也不承诺全空间最优。

```bash
# 只检查计划，不启动 GPU 测试。
python tools/autotune.py --config configs/autotune.json --model qwen35b --dry-run

python tools/autotune.py --config configs/autotune.json --model qwen35b --mode training
python tools/autotune.py --config configs/autotune.json --model qwen122b --mode inference
python tools/autotune.py --config configs/autotune.json --model qwen35b --mode decode

# 双机 PF：共享仓库/缓存绝对路径、相同 Python 环境、免密 SSH。
python tools/autotune.py --config configs/autotune.json --model flash \
  --mode training --gpus-per-node 8 --remotes user@node1 \
  --master-addr node0-address --ssh-port 22
```

主要配置项：

| 配置 | 内容 |
| --- | --- |
| `workload` | 模型、tokens；可覆盖 `hidden/intermediate/experts/top_k` |
| `system` | 每节点 GPU 数、远端节点、rendezvous/SSH 地址 |
| `forward/activation/weight/decode` | `axes` 的笛卡尔积与 `variants` 的显式组合，参数名带 `forward.` / `backward.` / `v4.` 前缀 |
| `ring.local/network` | 窗口 tokens、contribution 深度等运行参数 |
| `screen/final`、`confirmation_*` | 预热、采样、确认次数与最小收益门槛 |
| `workspace_limit_bytes` | native workspace 限额，不包含所有权重、训练缓存和框架显存 |
| `environment`、`run` | 编译/网络环境、并发编译数、输出和缓存路径 |

CLI 覆盖配置；`decode` 仅单机。训练默认联合优化 8K/20K 完整 FWD+BWD，
inference 优化无缓存 PF，decode 优化固定 token CUDA Graph。换 shape 无需修改调优脚本。

输出目录按 shape、模式、EP 分开：`result.json` 保存候选与测量，`runtime.json` 保存所选
扩展/环境路径，`table.json` 保存容量与 ring 配置，编译缓存保留生成的 header 和二进制。
部署时使用同次结果的库与表，按 `runtime.json` 设置环境。

`WEFTMOE_AUTOTUNE_TABLE` **精确匹配** GPU、H/I/E/K、节点拓扑、训练/推理、输入布局及
`max_tokens`，并核对 compiled ID；它不自动选择最近的容量。应用容量不同应重新验证/生成匹配项。
不使用表时将该变量留空，可显式设置 `WEFTMOE_STATIC_RING_TOKENS` 和
`WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS`；示例 benchmark 使用 B2048/D2，不代表所有负载最优。

## 快速验证

```bash
python -m pytest -q
torchrun --standalone --nproc-per-node=8 benchmarks/torchrun_training.py \
  --backend pf --model qwen35b --tokens 129 --warmups 0 --samples 2 --verify
```

GPU 命令依赖前面构建的匹配扩展。完整计时与多机启动方式见 [Benchmark](benchmarks.md)。

## 2026-10-29：FP8 checkpoint 与紧凑 workspace

这版按一次转换、全局复用实现 X 的 BWD 列量化：指定 dU CTA 异步读取完整 row FP8 块，随后原地写回 column FP8；dW1 复用这份结果。首次转换状态随 tape 保留，重复原生 BWD 不再转换。AB 保存 row FP8，FWD/BWD 都消费对应反量化值。dU（此处指 `dR @ W2`，尚未乘路由 gate）保留 BF16，但仅占 ring 容量。

使用 `TensorRuntime(..., fp8_workspace=True)` 启用紧凑源包和 X/S ring。FP8 路径不再分配全量 BF16 dR/dAB；同一个 runtime 混用 BF16/FP8 时保持默认 `False`。

测试为 L20X / SM90a、EP2、三种 profile，名义 tokens 为 8192/20480，沿用随机 token 数和路由负载。每项 10 次 warmup、30 次采样；以下是每次取慢 rank 后的 p50，单位 ms。权重预处理不计时，源端在线量化计时。

| Profile / tokens | FWD BF16 / FP8 | BWD BF16 / FP8 | 合计 BF16 / FP8 | 合计加速比 |
|---|---:|---:|---:|---:|
| qwen35b / 8192 | 2.510 / 2.187 | 3.390 / 3.690 | 5.772 / 5.675 | 1.02× |
| qwen35b / 20480 | 4.710 / 4.723 | 6.638 / 7.398 | 11.041 / 11.996 | 0.92× |
| qwen122b / 8192 | 4.579 / 3.415 | 8.010 / 7.478 | 12.538 / 10.572 | 1.19× |
| qwen122b / 20480 | 8.720 / 7.389 | 15.391 / 16.250 | 24.118 / 23.428 | 1.03× |
| flash / 8192 | 7.149 / 5.187 | 13.905 / 12.660 | 21.117 / 17.649 | 1.20× |
| flash / 20480 | 13.869 / 11.947 | 29.066 / 26.389 | 44.289 / 38.051 | 1.16× |

没有达到六组 FWD/BWD 都优于 BF16 的目标。Qwen35B 20K 仍明显落后；Qwen35B 8K 和其他很小的差距不能视为稳定提速。相比旧 FP8，本轮节省显存，但 AB 量化/反量化和后移的 X 转换增加了计算成本。没有把所有时延差额都归因于单个转换步骤。

下表单位 MiB，使用最后一次测量的实际分配尺寸、取较大 rank。活动张量合计包含 compute/communication workspace、notify、tape、dX/dGate/dW 和 BWD 临时区。最后一列再加 FP8 的 W1/W2 及转置副本与 scale，或 BF16 的 packed W1。共有 BF16 主权重、输入 X/dY、优化器状态、CUDA/NCCL 上下文和 allocator 预留不计入，因此不是 GPU 实测峰值。

| Profile / tokens | compute workspace BF16 / FP8 | 活动张量 BF16 / FP8 | 含额外权重缓存 BF16 / FP8 |
|---|---:|---:|---:|
| qwen35b / 8192 | 1107 / 581 | 2544 / 1949 | 3056 / 2717 |
| qwen35b / 20480 | 2308 / 1275 | 5116 / 3707 | 5628 / 4475 |
| qwen122b / 8192 | 1622 / 823 | 5071 / 4112 | 6607 / 6417 |
| qwen122b / 20480 | 3332 / 1771 | 9132 / 6791 | 10668 / 9095 |
| flash / 8192 | 1799 / 882 | 9478 / 8328 | 13574 / 14473 |
| flash / 20480 | 3501 / 1807 | 14254 / 11371 | 18350 / 17516 |

通信及计算 workspace 减少约 45%–51%。整个算子不能简单声称显存减半；Flash 8K 加上权重缓存后仍高于 BF16。TMA 读取原权重再在 shared memory 转置的候选精度通过，但 Qwen35B 8K BWD 从约 3.61 ms 增至 5.31 ms，因此没有合入默认路径，仍保留 W1ᵀ/W2ᵀ。

验证覆盖三个 profile 的 EP2、8193-token 尾块、变化 scale、偏斜路由、EP4 空源 rank、多版本异步、BF16/FP32 dW、checkpoint、optimizer、推理，以及原生 tape 的重复异步 BWD。BF16/FP8 共用 pool 的格式切换及 BF16 autograd/graph/offload 回归通过；FP8 单元测试 37 项、host 测试 5 项通过。Compute Sanitizer 在新旧库均遇到 `cuKernelGetFunction` 加载错误，这项未完成。

最终构建：`build/fp8-memory-20261029-{qwen35b,qwen122b,flash}/`。原始采样、逐项内存、构建 hash、验证日志和未采用候选见 [本轮记录](benchmarks/fp8_memory_20261029.json)。

## 2026-10-29：FP8 权重转置替代路径实测

已实现并测试 swap A/B + WGMMA RS（权重寄存器、梯度 shared memory），每 K128 应用原来的 block scale，dW 保持 BF16。第一版按字节加载；第二版通过 `ldmatrix.x4.trans.b16` 加 shuffle/byte-permute 装配 FP8 寄存器 fragment。输出在 math 专属的 X 转换 scratch 中分块重排，不覆盖可能被 producer 提前填充的 operand stage。另测试了 N128 小 tile，以及 warp 成对处理 16×16 子块的 shared-memory 转置。后者参考 CUTLASS 的小块组织思路，属于本仓库的自定义候选，并非直接移植其完整主循环。

结论：均未替换默认预转置路径。RS 最终候选真正省去 W1ᵀ/W2ᵀ 分配，正确性通过，但六组 BWD 全部慢于预转置 FP8 和 BF16。当前生产源码及公开 Python API 不变；实验代码以隔离补丁保存。

以下为 L20X / SM90a、EP2；10 warmups、30 samples，每个样本取慢 rank，报告 p50，单位 ms。8K/20K 是每 rank 的名义 token 数，实际数量和路由按固定种子变化。BWD 包含 backward、release、device synchronize；新 FWD 在计时外完成。权重准备不计时；FWD 的源端在线量化计时。

| Profile / tokens | BWD BF16 | BWD 预转置 FP8 | BWD RS | RS 相对预转置回退 | FWD 预转置 / RS |
|---|---:|---:|---:|---:|---:|
| qwen35b / 8192 | 3.370 | 3.633 | 4.317 | 18.8% | 2.253 / 2.197 |
| qwen35b / 20480 | 6.501 | 7.492 | 9.305 | 24.2% | 4.693 / 4.704 |
| qwen122b / 8192 | 8.007 | 7.415 | 9.285 | 25.2% | 3.390 / 3.416 |
| qwen122b / 20480 | 15.352 | 16.309 | 20.431 | 25.3% | 7.387 / 7.386 |
| flash / 8192 | 13.825 | 12.742 | 16.063 | 26.1% | 5.192 / 5.167 |
| flash / 20480 | 29.010 | 26.278 | 34.596 | 31.7% | 11.852 / 11.883 |

Qwen35B 8K 筛选实验：逐字节 RS 4.503 ms，ldmatrix RS N256 4.316 ms，ldmatrix RS N128 4.510 ms，warp 分块 shared 转置 4.715 ms。同轮预转置基线约 3.66 ms。这些候选改动了加载、寄存器 fragment 和/或 epilogue，因此不能把全部差额归因于某一条指令或某一种资源。未完成阶段级 profiler 归因。

只统计每 rank 额外准备的 FP8 权重及 FP32 scales，单位 MiB：

| Profile | 预转置四份矩阵 | RS 两份矩阵 | 减少 |
|---|---:|---:|---:|
| qwen35b | 768.19 | 384.09 | 50% |
| qwen122b | 2304.56 | 1152.28 | 50% |
| flash | 6145.50 | 3072.75 | 50% |

RS 数字来自准备对象实际 tensor 字节数；预转置数字按同形状双份数据及 scale 计算。不是 GPU allocator 峰值，也不包含共有 BF16 主权重、输入、workspace 或梯度。FWD 改变量约 −2.5% 到 +0.8%，不视为稳定提速。

验证：三个 profile 的 EP2 / T257 / E32 / top-k 2，变化行及权重 scale、均匀/偏斜路由、多版本异步、BF16/FP32 dW、checkpoint、optimizer、推理；另通过 Qwen35B EP2 T8193 尾块和 Flash EP4 T257 空源 rank。每个本地实际训练权重对象都断言没有转置数据；独立数值参考仍使用完整权重。测试 shim 仅绕过 Python 的旧“训练必须准备转置”检查，保留 tensor 版本和 CUDA event 等检查；native 候选允许单节点 original-only 权重，多节点仍要求转置权重。

实验入口及补丁位于 `benchmarks/experiments/fp8_weight_layout/`。`build.py` 校验当前源码 hash，再复制源码和应用补丁，不修改生产文件；输出目录必须未包含已有库。

```bash
python benchmarks/experiments/fp8_weight_layout/build.py \
  --variant rs --model qwen35b --tile-n 256 \
  --output build/fp8-weight-rs-repro

export WEFTMOE_TENSOR_EXTENSION="$PWD/build/fp8-weight-rs-repro/_runtime.cpython-312-x86_64-linux-gnu.so"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  benchmarks/experiments/fp8_weight_layout/run_original_weights_only.py \
  benchmarks/torchrun_training.py --backend pf --model qwen35b \
  --tokens 8192 --warmups 10 --samples 30 --stream-ordered \
  --fp8 --weight-grad-dtype bf16 --backward-only
```

`rs_scalar` 和 `warp_transpose` 可作为 `--variant`；warp_transpose 仍按普通方式准备权重，运行原 benchmark，不使用 original-only shim。N128 通过 `--tile-n 128` 选择。完整测量、库 SHA256、测试输出与命令见 [`fp8_weight_layout_20261029.json`](benchmarks/fp8_weight_layout_20261029.json)。

## 2026-10-29：权重量化同时输出 W 与 Wᵀ

保留预转置权重，BF16/FP32 训练参数改为每个权重一个 kernel 输出两种布局及 scale。
128×128 block 只读取一次输入，四元素向量加载和打包写出，通过 swizzled shared
memory 写出转置副本。已有 FP8 checkpoint 继续逐位复制和转置。两种布局的显存容量不变。

以下仅诊断 W1、W2 的 GPU 准备耗时，单位 ms。L20X、EP2、每 rank 128 个专家，
BF16 输入、E4M3 输出、FP32 block scales；预分配输出、CUDA Graph 和 CUDA events，
10 次预热、30 次采样，取较慢 rank 的 p50。旧路径为量化再独立转置。

| Profile | 旧路径 | 双输出路径 | 加速比 |
|---|---:|---:|---:|
| qwen35b | 2.971 | 1.277 | 2.33× |
| qwen122b | 8.794 | 3.772 | 2.33× |
| flash | 23.408 | 9.986 | 2.34× |

权重准备不计入 FWD/BWD benchmark 总耗时，源端激活在线量化仍计时。
三个 profile 的 EP2 正确性回归通过，包含变化 scale、偏斜路由、多版本异步、
BF16/FP32 dW、checkpoint、optimizer 和推理；56 项 FP8 单测通过，覆盖双布局逐位参考、
尾块、空张量、非对齐输入、非有限值及 CUDA Graph。三个 profile 的 8192/20480-token
FWD+BWD benchmark 均完成，保持权重准备在计时外；这次修改不改变 FWD/BWD 计算 kernel。

构建为 `build/fp8-dual-weight-{qwen35b,qwen122b,flash}/`。
原始采样见 [准备耗时](benchmarks/fp8_dual_weight_preparation_20261029.json)，
完整命令、训练 benchmark 与验证输出见 [验证记录](benchmarks/fp8_dual_weight_validation_20261029.json)。

## 2026-10-29：正式 FWD/BWD kernel profile

生产二进制仍为 `build/fp8-dual-weight-{qwen35b,qwen122b,flash}/`，本次只增加
benchmark profiling 入口，没有修改生产计算 kernel。GPU 为 NVIDIA L20X（SM90），
EP2、256 experts；qwen35b 的 H/I/top-k 为 2048/512/8，qwen122b 为
3072/1024/8，flash 为 4096/2048/6。FP8 为 E4M3 block-wise，双方 dW 都输出 BF16。

以下单位 ms，是 CUPTI 记录的 **主 kernel GPU duration**，10 次预热、30 次采样，
每个 sample 取两个 rank 的较大值，再取中位数。tokens 是每 rank 的名义值，实际
tokens 使用标准差 10%、截断 ±3σ 的变化工作负载；BF16/FP8 的路由 hash 逐组一致。
权重准备在计时外；源端激活量化、notify、prepare/setup 是其他 kernel，单列在原始
JSON 中，不属于此表。表中的加速比为 BF16 / FP8；它不是完整训练步的加速比。

| Profile | Tokens | BF16 FWD | FP8 FWD | 加速比 | BF16 BWD | FP8 BWD | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen35b | 8192 | 2.039 | 1.732 | 1.18× | 3.022 | 3.246 | 0.93× |
| qwen35b | 20480 | 4.050 | 4.126 | 0.98× | 6.109 | 7.028 | 0.87× |
| qwen122b | 8192 | 4.058 | 2.920 | 1.39× | 7.658 | 7.018 | 1.09× |
| qwen122b | 20480 | 8.094 | 6.856 | 1.18× | 14.901 | 15.705 | 0.95× |
| flash | 8192 | 6.681 | 4.684 | 1.43× | 13.487 | 12.190 | 1.11× |
| flash | 20480 | 13.944 | 11.236 | 1.24× | 27.922 | 25.880 | 1.08× |

源端量化的 GPU duration：FWD 35.8–142.7 µs，BWD 32.9–139.6 µs。
Qwen35B 20K 分别为 74.2/72.8 µs。即使将它们完全隐藏，也不能解释或消除该场景
BWD 主 kernel 的约 0.919 ms 回退。异步调用的 CPU 边界不是 GPU 阶段边界：分析器
通过 CUDA launch correlation 归属 FWD/BWD；不同阶段的 GPU span 可能重叠，不能
将 span、各 kernel 的中位数或累计 CTA 时间相加当作端到端耗时。

生产采样、完整命令、库 SHA256、launch resources 和逐 rank/sample 数据见
[正式 profile](benchmarks/fp8_kernel_profile_20261029.json)，source_id 为
`weftmoe-fp8-kernel-profile-20261029`。原始 CUPTI trace 保存在
`build/fp8-kernel-profile-traces/`。复现单组：

```bash
export WEFTMOE_TENSOR_EXTENSION="$PWD/build/fp8-dual-weight-qwen35b/_runtime.cpython-312-x86_64-linux-gnu.so"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  benchmarks/torchrun_training.py --backend pf --model qwen35b \
  --tokens 20480 --warmups 10 --samples 30 --stream-ordered \
  --fp8 --weight-grad-dtype bf16 --cuda-trace-dir /tmp/weftmoe-trace
python benchmarks/analyze_training_trace.py /tmp/weftmoe-trace/t20480.rank0.json \
  --output /tmp/weftmoe-trace/rank0-summary.json
```

### 阶段定位与优化优先级

按 KernelWiki 的 [Pipeline Stalls](../.agents/skills/KernelWiki/wiki/patterns/pipeline-stalls.md)
（`pattern-pipeline-stalls`）方法，先测等待角色及其依赖，再选择优化。
额外构建仅启用 `WEFTMOE_DEVICE_TIMELINE=1` 和 `WEFTMOE_PF_BWD_TASK_TIMING=1`，
用 Qwen35B、EP2、固定 20480 tokens 诊断。以下数字来自插桩构建，不替代上面的正式耗时；
FWD 只有一个采样，两个 rank 均无 timeline overflow；BWD 为 3 次预热、5 次采样。

1. **优先优化 FC1 epilogue。** 每 tile 中位数：FP8 20.352 µs，BF16 5.856 µs。
   FC1 math 区间为 22.624/25.888 µs，FC2 math 为 6.304/12.544 µs（FP8/BF16）。
   FP8 的 FC2 已有明显收益，FC1 epilogue 吃掉了相当一部分收益。
   `local_compute.cuh::store_fp8_swiglu` 要计算 A/B scale、量化并保存 AB、解码量化后的
   AB 做 SwiGLU，再量化 S。可实验打包/vectorized stores、缩短临时值生存期、减少重复
   地址及转换指令。必须保持当前量化 AB 与反向重算一致的数值语义。
   epilogue 区间还包含上一代 FC2 的 ring 退役等待和发布同步，尚不能把全部差值归因于
   转换算术；下一步应把这些边界单独插桩后做 A/B 实验。
2. **优化 BWD 的 SwiGLU 双方向输出，以及 dU 中的转换排布。** 每 rank/sample 的
   累计 CTA SwiGLU 服务时间中位数为 FP8 222.6 ms、BF16 101.5 ms；dU 服务时间
   221.6/159.4 ms，dX 服务时间 129.6/163.9 ms。这里服务时间是所有 CTA 任务 elapsed
   time 的累加，dU/dX 包含 operand wait，dU 还包含 X/dR row→column 转换，并非纯 GEMM。
   producer 等待 dX 输入就绪的累计时间为 211.6/40.4 ms，说明激活发布链值得检查。
   `fp8_swiglu.cuh` 同时输出 dAB row、dAB column、重算激活 column；应优化跨 warp
   reduction、打包转换和寄存器生存期。X/dR 仍保持“一次转换、全局复用”，可以实验
   独立转换缓冲或重排 stage 退役时机，减少转换阻塞下一次 TMA；必须确保所有读者完成。
3. **dW 不是这组回退的首要嫌疑。** 诊断中 dW2 extent 为 FP8/BF16 0.604/0.744 ms，
   dW1 为 1.105/1.404 ms；从 BWD start 到 dW2 start 为 5.041/3.815 ms。
   dW1/dW2 和 reduce 可重叠，不可直接相加；这些是跨 CTA 阶段边界，不是独立 kernel。
   FP8 的旧 du_start/dx_start 字段未覆盖该分支，已排除 sentinel 值。
4. **最后再调 MMA 粒度和 stage 数。** `fp8/tile.cuh::multiply_tile` 的 K128 block-scale
   promotion 和 N128 partial accumulator 会引入 CUDA Core 工作与 WGMMA wait。
   可测试更合适的 tile/role 分配，但不能直接删掉依赖等待或盲目增加 partial accumulator。
   生产 Qwen35B FWD/BWD 均为 384 threads、168 registers/thread，shared memory 分别为
   230912/231008 bytes，当前只有 1 CTA/SM 的驻留预算。压缩 FP8 stage 缓冲可释放共享
   内存，但寄存器约束仍在，不代表自动得到两个 CTA/SM 或更高性能。

### 硬件计数器及其适用范围

Nsight Compute 2025.2.1 在 EP2 的 application replay 和 app-range 均未正常结束，
超时退出；没有采用受阻 profiling 的时间。改用 EP1、128 experts，匹配 EP2 每卡
权重规模，固定 20K tokens。该实验改变路由、通信与尾块，**仅作为计算侧诊断代理**。
使用 application replay，未锁时钟/清缓存；FP8 FWD 完整 sections 为 17 passes，其他
三组为精简计数器。原始 `.ncu-rep` 和 CSV 在 `build/fp8-kernel-profile-diagnostics/`。

| 代理实验 | Tensor pipe active，% elapsed | DRAM active/throughput，% peak | Issue active，% active |
|---|---:|---:|---:|
| BF16 FWD | 35.74 | 29.84 | 16.06 |
| FP8 FWD | 20.28 | 22.77 | 24.89 |
| BF16 BWD | 49.88 | 42.24 | 17.20 |
| FP8 BWD | 19.85 | 22.27 | 29.32 |

FP8 FWD 的 DRAM 列为 `dram__cycles_active`，其他行为 `dram__throughput`；完整 metric
名称、单位及值见 JSON。这些指标显示 FP8 的 Tensor Core 活跃时间占比低、普通指令
发射占比上升，支持继续检查转换/epilogue 与依赖链；不证明单个 CUDA Core 单元饱和。
FWD 的 long-scoreboard 指标较高，但 persistent kernel 内也有通信轮询，不能直接认定
GEMM 全局访存是根因。未见整段 DRAM 饱和，仍不能排除局部访存延迟或通信带宽限制。
NCU 的并发/replay 限制参见 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

KernelWiki 的 [Pipeline Stages](../.agents/skills/KernelWiki/wiki/techniques/pipeline-stages.md)
（`technique-pipeline-stages`，source-reported）要求结合 stage 字节数、等待时间及资源
预算选深度。该页引用 FlashInfer PR 1039 的 SM100 示例：

```cpp
pipeline_q.consumer_wait(pipeline_q_consumer_state);
++pipeline_q_consumer_state;
pipeline_s0.producer_acquire(pipeline_s0_producer_state);
gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);
pipeline_s0.producer_commit(pipeline_s0_producer_state);
++pipeline_s0_producer_state;
```

这里只借鉴明确 stage 所有权及 producer/consumer 重叠的方法，SM100 指令和其性能数字
不适用于当前 SM90。寄存器调参依据
[Register Pressure](../.agents/skills/KernelWiki/wiki/patterns/register-pressure.md)
（`pattern-register-pressure`）；降低寄存器上限必须检查 spills 与实际延迟。
以上四项是待验证优化方向，尚未声称取得优化收益。
阶段原始 status、汇总及计数器见 [诊断数据](benchmarks/fp8_kernel_diagnosis_20261029.json)。

## 2026-10-29：EP8 实测与多进程 NCU 采集修正

上节 NCU 超时是当时采集方式的限制，不能解读成 EP2 无法 profile。
本机 NCU 2025.2.1 不提供新版的 lockstep/communicator 选项。用 NCU 包住整个
torchrun 进程树时，kernel profiling 的跨进程串行化/重放与必须并发推进的 peer
kernel 存在冲突：本仓库的 `collective.cuh::rank_barrier` 会向各 rank 写入信号并
等待全部 peer 到达，FWD before-combine 和 BWD 的 rank barrier 都有该依赖。
若 profiler 只允许其中一个 kernel 前进，该 kernel 就可能一直等待尚未被放行的 peer。
这符合 [NVIDIA 对 Mandatory Concurrent Kernels 的说明](https://docs.nvidia.com/nsight-compute/NsightComputeCli/#mandatory-concurrent-kernels)。
旧日志没有定位到具体 GPU 指令，因此不将某一次超时断言为某个确定的 barrier 死锁。

本次改为 **torchrun 启动所有 rank，仅一个 rank exec NCU，其他 rank 完全不注入；
仅采集经实测为一个 pass 的硬件计数器**。EP8 rank 0、rank 2 的 FP8/BF16 FWD/BWD
共八组全部成功，EP2 rank 0 FP8 FWD 复查也成功，共九组日志均确认 `1 pass`。
这支持旧采集方式干扰进程协作的判断。不要把此方法扩展为 multi-pass kernel replay：
peer 的通信状态不会跟着单个 rank 一起重放。未来使用新版 communicator/lockstep 前
还要检查驱动要求；本次没有修改系统驱动或 NCU 安装。

EP1 没有跨 GPU 专家交换，只有本地搬运与同步；其旧计数器只诊断计算侧，不能代表
EP2/EP8 的 NVLink 通信与等待。当前 EP8 是单机八卡 NVLink P2P，不是跨机 RDMA。

### EP8 正式主 kernel 时间

GPU、模型 H/I/top-k、FP8/BF16 dW 口径与上节相同；256 experts，EP8 每 rank 32
个专家。使用正式生产二进制，CUPTI 采样保留并发，不注入 NCU。每 rank 名义
8192/20480 tokens，变化 tokens 与偏斜随机路由，10 次预热、30 次采样，逐 sample
取八个 rank 的最大主 kernel duration，再取中位数。BF16/FP8 路由 hash 逐组一致。
单位 ms；加速比为 BF16/FP8。权重准备排除，独立源端量化/notify/setup 不在主 kernel
表内。source_id：`weftmoe-fp8-kernel-profile-ep8-20261029`。

| Profile | Tokens | BF16 FWD | FP8 FWD | 加速比 | BF16 BWD | FP8 BWD | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen35b | 8192 | 2.426 | 2.464 | 0.98× | 3.464 | 3.706 | 0.93× |
| qwen35b | 20480 | 6.414 | 6.069 | 1.06× | 8.246 | 9.047 | 0.91× |
| qwen122b | 8192 | 4.224 | 3.798 | 1.11× | 7.335 | 7.785 | 0.94× |
| qwen122b | 20480 | 14.534 | 9.066 | 1.60× | 21.375 | 19.142 | 1.12× |
| flash | 8192 | 6.147 | 5.342 | 1.15× | 11.996 | 12.172 | 0.99× |
| flash | 20480 | 15.158 | 13.487 | 1.12× | 29.381 | 29.771 | 0.99× |

FP8 的 FWD 五组更快，BWD 只有 Qwen122B 20K 明显更快；Flash BWD 约 1% 的差距
应视为接近持平，不承诺稳定优劣。此表不能与 EP2 直接做强缩放比较：这里保持每卡
tokens 和全局专家数，EP8 的总 tokens、每个专家的行数及远程比例均改变。
完整逐样本数据、各辅助 kernel、二进制 SHA256 和命令见
[EP8 正式采样](benchmarks/fp8_kernel_profile_ep8_20261029.json)。

### EP8 的等待与激活处理

使用上一节的隔离诊断构建，Qwen35B、EP8、固定 20480 tokens/rank，FWD 为 3 次预热
和 1 次采样，BWD 为 3 次预热和 5 次采样。16 份 FWD rank trace 都没有溢出，80 条
BWD rank/sample status 完整。阶段结果仍是插桩诊断，不能替换正式 kernel 时间。

| FWD 每 tile 区间中位数，µs | BF16 | FP8 |
|---|---:|---:|
| FC1 math（包含 operand wait） | 50.880 | 52.832 |
| FC1 full-stage wait 累计/每 tile | 30.240 | 31.744 |
| FC1 epilogue（含 ring 退役和发布） | 6.208 | 23.072 |
| FC2 math（包含 operand wait） | 6.912 | 6.304 |

EP8 FC1 的 operand 等待已很明显，不能仅沿用 EP1 的算术侧判断。应拆开定位
跨 GPU pull、输入就绪、producer TMA 提交和 consumer full-stage 等待，检查这些环节
的衔接；当前计数无法将等待全部归为 NVLink 带宽限制。FP8 FC1 epilogue 的差距仍在，
优化 AB/S 量化、打包写出和同步仍有依据。

BWD 每 rank/sample 的累计 CTA SwiGLU 服务时间中位数为 FP8/BF16
206.678/94.838 ms；producer 等待 dX 输入就绪为 185.032/40.344 ms。这些是多个 CTA
并行区间的累加，不能相加成墙钟延迟。dW2 阶段 extent 为 0.621/0.587 ms，dW1 为
1.053/1.087 ms：EP8 下 dW 接近持平，不能直接沿用 EP2 的“dW 更快”结果。

该 FWD 样本共 1,310,720 次专家分配，其中 1,146,317 次跨 rank，占 87.457%。
这些是路由计数，不是 NVLink 字节计数。rank 2 的 FC2 tile 数为 13632，rank 0 为
8688；FP8 各 rank 从本地 main-enter 到最后 FC2 tile 结束约 3.78–5.77 ms，显示
需要考虑专家负载不均。这里只比较各 GPU 自己的相对耗时，不对齐不同 GPU 的 globaltimer。
原始阶段数据及命令见 [EP8 阶段诊断](benchmarks/fp8_kernel_ep8_diagnosis_20261029.json)。

### EP8 内的 NCU 单 pass 计数器

相同 Qwen35B 固定 20K 工作负载，在实际 EP8 执行中分别采 rank 0 和该样本较忙的
rank 2；每种设置三次预热、一次采样，两个 rank 来自不同运行。下面给出两者的范围，
不是八卡平均。source_id：`weftmoe-fp8-ep8-ncu-20261029`。

| Kernel | Tensor pipe active，% elapsed | DRAM throughput，% peak | Issue active，% active |
|---|---:|---:|---:|
| BF16 FWD | 30.20–30.94 | 17.02–18.47 | 13.68–13.83 |
| FP8 FWD | 13.83–14.75 | 9.51–10.46 | 17.37–18.04 |
| BF16 BWD | 44.78–46.64 | 27.15–27.52 | 14.87–15.00 |
| FP8 BWD | 18.75–18.87 | 16.54–16.75 | 26.98–27.12 |

FP8 的 Tensor Core 活跃占比低、普通指令发射占比高，但这些整 kernel 指标不能单独
证明 CUDA Core 单元饱和。DRAM 指标也不是 NVLink 利用率，不能据此排除通信瓶颈。
单 rank 注入仍会改变 peer 启动时序，因此 NCU 下的 GPU duration 不用于正式性能表。
计数器和日志见 [NCU 记录](benchmarks/fp8_kernel_ep8_ncu_20261029.json)，原始报告位于
`build/fp8-kernel-profile-ep8-ncu/`。

可复用入口为 `benchmarks/experiments/profile_single_rank.py`。例如采集 rank 2 FWD：

```bash
export WEFTMOE_TENSOR_EXTENSION="$PWD/build/fp8-dual-weight-qwen35b/_runtime.cpython-312-x86_64-linux-gnu.so"
export WEFTMOE_NCU_RANK=2
export WEFTMOE_NCU_KERNEL='regex:.*weftmoe_pf::local::kernel.*'
export WEFTMOE_NCU_OUTPUT=/tmp/weftmoe-ep8-rank2-fwd
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 \
  benchmarks/experiments/profile_single_rank.py --backend pf --model qwen35b \
  --tokens 20480 --token-std-fraction 0 --warmups 3 --samples 1 \
  --stream-ordered --ncu-range --forward-only --fp8 --weight-grad-dtype bf16
```

本次没有修改正式计算 kernel。新增入口的 rank 2 四组实际运行均成功且为一个 pass。


## 2026-10-29：根据 profile 优化 FP8 转换和调度

本轮保留四项改动：FC1 epilogue 的 AB/S 改为 8 字节打包写出；S 的中间值保持
BF16 packed pairs；行 amax 使用整数归约；SwiGLU 以及 X/dR 列量化使用 PRMT 打包
替代逐字节移位累积。FWD 对小专家缩短 wave，三个 profile 的 M-block wave 从
64/32/16 改为 **32/32/16**。BWD wave、源端量化位置、X/AB 缓存布局、一次转换全局
复用及 BF16 dW 语义保持一致；没有新增转换 kernel 或 workspace。

遵循 KernelWiki 的 [Vectorized Loads](../.agents/skills/KernelWiki/wiki/techniques/vectorized-loads.md)
（`technique-vectorized-loads`，source-reported）与
[Register Budgeting](../.agents/skills/KernelWiki/wiki/techniques/register-budgeting.md)
（`technique-register-budgeting`，source-reported）的验证方法：先核对对齐、尾块和
生成指令，再用完整 kernel A/B 决定是否保留。这里的性能数字均来自本机实测，不引用
wiki 中其他 GPU/算子的加速比。

### 主 kernel 的正式对照

L20X SM90，单机 NVLink，256 experts；Qwen35B H/I/top-k=2048/512/8，
Qwen122B=3072/1024/8，Flash=4096/2048/6。FP8 E4M3 计算、BF16 dW；BF16 对照也
输出 BF16 dW。每 rank 名义 8192/20480 tokens，10% token 变化和偏斜随机路由。
10 次预热、30 次采样，CUPTI 保留 peer 并发，逐 sample 取最大 rank 主 kernel
duration 后取中位数。旧 FP8、优化 FP8、BF16 的 workload hash 逐组一致。
权重预量化不计时；此表不包含独立源端量化、notify/setup。单位 ms。

| EP | Profile | Tokens | 旧 FP8 FWD | 新 FP8 FWD | BF16 FWD | 旧 FP8 BWD | 新 FP8 BWD | BF16 BWD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | qwen35b | 8192 | 2.468 | 2.302 | 2.437 | 3.706 | 3.645 | 3.462 |
| 8 | qwen35b | 20480 | 6.075 | 5.671 | 6.426 | 9.043 | 8.852 | 8.239 |
| 8 | qwen122b | 8192 | 3.818 | 3.725 | 4.205 | 7.808 | 7.610 | 7.316 |
| 8 | qwen122b | 20480 | 9.096 | 8.888 | 14.437 | 19.154 | 18.601 | 21.323 |
| 8 | flash | 8192 | 5.356 | 5.219 | 6.178 | 12.297 | 11.952 | 11.925 |
| 8 | flash | 20480 | 13.479 | 12.755 | 15.110 | 29.709 | 28.979 | 29.237 |
| 2 | qwen35b | 8192 | 1.732 | 1.687 | 2.043 | 3.257 | 3.178 | 3.035 |
| 2 | qwen35b | 20480 | 4.109 | 4.020 | 4.053 | 6.989 | 6.843 | 6.175 |
| 2 | qwen122b | 8192 | 2.926 | 2.837 | 4.064 | 6.904 | 6.796 | 7.712 |
| 2 | qwen122b | 20480 | 6.781 | 6.514 | 8.093 | 15.645 | 15.260 | 14.928 |
| 2 | flash | 8192 | 4.667 | 4.595 | 6.669 | 12.133 | 11.987 | 13.412 |
| 2 | flash | 20480 | 11.226 | 10.535 | 13.889 | 25.774 | 25.275 | 28.611 |

### 无 profiler 的整轮 FWD+BWD

另起进程、不启用 CUPTI/NCU，保持相同 seed、10 次预热和 30 次采样。计时覆盖源端
量化、通信、notify/setup、FWD/BWD、API 开销以及最后的 CUDA 同步，权重准备与
输入/路由生成在计时外。逐 sample 最大 rank wall time 的中位数；单位 ms。
这不是前表两列相加，也不是异步 FWD/BWD CPU 调用区间。

| EP | Profile | Tokens | 优化 FP8 整轮 | BF16 整轮 | BF16/FP8 |
|---:|---|---:|---:|---:|---:|
| 8 | qwen35b | 8192 | 6.651 | 6.517 | 0.980× |
| 8 | qwen35b | 20480 | 15.344 | 15.517 | 1.011× |
| 8 | qwen122b | 8192 | 12.158 | 12.283 | 1.010× |
| 8 | qwen122b | 20480 | 28.358 | 37.052 | 1.307× |
| 8 | flash | 8192 | 17.896 | 18.980 | 1.061× |
| 8 | flash | 20480 | 42.805 | 46.255 | 1.081× |

本轮没有达到所有模型、所有 token 规模的 FWD/BWD 都稳定快于 BF16 的目标。
Qwen35B 及 Qwen122B 部分场景的 BWD 仍落后；接近 1% 的差异不宣称为稳定收益。主 kernel 的小幅收益
也不等于整轮同等比例提升，整轮还包含通信、源端转换和调用开销。

### 排除的方案和资源检查

- 统一缩小 FWD wave 会使 Flash EP8 的 FWD 回退约 5–8%，因此恢复大专家的宽 wave。
- 将 Qwen35B 的 BWD wave 缩到 32，EP8 pilot 从约 9.11 ms 变成 11.75 ms；
  math register budget 从 216 增到 232 也变慢，没有保留。
- FP8 解码指数调整路径虽然通过约 1743 万组逐位检查，但 BWD 约慢 17%，已撤回。
- X 独立转换：Qwen35B 20K 的 EP2 主 BWD 约 6.19 ms，另加约 0.56 ms 转换；
  按逐 rank/sample 合计后约 6.74 ms。EP8 合计约 9.10 ms，没有超过融合转换的
  指令优化版本。不能只引用变短的主 BWD 时间，最终仍在 dU owner tile 内转换。
- 提前释放 stage、将 X 转换或 dR casting 放到释放之后，Qwen35B pilot 没有
  稳定额外收益，已撤回。生产路径维持原有 barrier 和 stage 生命周期。

pilot 为 5 次预热、20 次采样，其具体 token/route 序列不同于正式 10/30 测试，
不能直接拿两套中位数算优化比例。原始命令、库 SHA256 和各候选结果均保存在 JSON。

三个 profile 的主 kernel 仍为 384 threads、168 static registers，math/control
动态预算仍为 216/72，shared memory 分配不变，仍是 1 CTA/SM。FWD stack frame
变为 48 bytes（旧值 Qwen35B/Flash 为 24、Qwen122B 为 0）；BWD stack 不变。
`LOCAL:0` 不代表没有 LDL/STL，SASS 仍有 local 指令，因此没有把该修改描述为
“消除了 spilling”。Qwen35B FWD 的静态 `STG.E.U16` 从 96 条变为 0，
`STG.E.64` 从 35 条变为 73，伴随更多 warp 重排；这只是编译指令数，不能当作
运行时 transaction 数。是否获益以正式 kernel 耗时为准。

### 验证和复现

- 640 组 FWD epilogue 与旧实现逐字节一致，包含 FastMath 两种模式、M 尾块、
  NaN/Inf、极大/极小值、packed/unpacked ring、AB 量化开关及无 checkpoint。
- 36 组列量化检查通过，覆盖 poisoned tails、BF16 subnormals 和 NaN。
- 三个 profile 的 EP2/EP8 完整 FWD/dX/dGate/dW 校验通过；Qwen35B EP8 使用
  8193 tokens 并包含空 rank，覆盖多 wave、尾块、不同 row scales。检查还包括
  多 tape 异步复用、checkpoint、optimizer、inference 及 BF16/FP8 混用。
- `tests/test_fp8.py`：56 passed。

生产库位于 `build/fp8-activation-opt-production-{qwen35b,qwen122b,flash}/`。
复现全矩阵（默认旧 FP8、新 FP8、BF16 三组）：

```bash
python benchmarks/experiments/profile_fp8_activation_optimization.py \
  --candidate 'build/fp8-activation-opt-production-{model}' \
  --worlds 8 2 --output /tmp/fp8-activation-comparison
```

[完整结果与验证记录](benchmarks/fp8_activation_optimization_20261029.json)

## 2026-10-29 双机 EP16 FP8 旧路径基线

两台各八卡 NVIDIA L20X（SM90），NCCL 2.31.2，节点内 NVLink、节点间
GIN/RoCE GDRDMA。使用上述 production 库，BF16 与 FP8 加载相同模型 specialization。
本节记录接入源端 FP8 之前的历史基线：**GEMM 为 FP8，但通信、workspace 和保存的
X/AB 仍为 BF16**，必须关闭 compact workspace。以下不是单机源端 FP8 路径的扩展性测试。

### 不开 profiler 的整轮耗时

单位 ms；10 次预热、30 次采样，每次取 16 个 rank 中最慢耗时，再取中位数。
包含 FWD、BWD、释放 tape 和 device synchronize；输入生成及 FP8 权重预量化
不计时。两种精度均输出 BF16 dW。每卡名义 token 为 8192/20480，实际长度按
标准差 10% 波动；E256，Qwen 两种规模 top-k=8，Flash top-k=6。
BF16/FP8 的逐 rank 工作负载 SHA256 完全一致；ring tokens=512，depth=8。

| Profile | 每卡名义 tokens | BF16 整轮 | FP8 整轮 | FP8/BF16 耗时 |
|---|---:|---:|---:|---:|
| qwen35b | 8192 | 13.352 | 25.803 | 1.933× |
| qwen35b | 20480 | 31.719 | 63.262 | 1.994× |
| qwen122b | 8192 | 24.394 | 58.234 | 2.387× |
| qwen122b | 20480 | 57.922 | 138.100 | 2.384× |
| flash | 8192 | 29.097 | 92.910 | 3.193× |
| flash | 20480 | 72.143 | 228.206 | 3.163× |

旧路径的六组多机 FP8 均无性能收益。这里是保持每卡 token 规模的 EP16 对比，
全局 token 数是 EP8 的两倍，不能直接据此计算强扩展效率。

### FWD/BWD 主 kernel 耗时

单位 ms；另跑 CUPTI，10 次预热、5 次采样，每次取跨 rank 最大值再取中位数。
通过 CUDA launch correlation 归属 FWD/BWD，统计 `network::kernel` 和
`backward_stream::kernel`；主 kernel 包含通信和等待，不是裸 GEMM 耗时。
两种精度的诊断工作负载一致，但采样数与上表不同，不将两个阶段的中位数相加
作为整轮耗时，也不使用 profiler 下的 wall time 代替上表。

| Profile | Tokens | BF16 FWD | FP8 FWD | BF16 BWD | FP8 BWD |
|---|---:|---:|---:|---:|---:|
| qwen35b | 8192 | 5.006 | 14.385 | 6.094 | 9.086 |
| qwen35b | 20480 | 11.760 | 38.251 | 15.936 | 23.298 |
| qwen122b | 8192 | 9.169 | 33.077 | 11.501 | 23.150 |
| qwen122b | 20480 | 22.717 | 83.782 | 31.403 | 58.462 |
| flash | 8192 | 11.832 | 51.015 | 15.495 | 41.305 |
| flash | 20480 | 30.111 | 129.641 | 41.049 | 109.622 |

FP8 FWD 主 kernel 耗时为 BF16 的 2.87–4.31 倍，BWD 为 1.46–2.67 倍。
FWD 的退化更大，应先处理多机 FWD 路径。

### 旧多机路径与单机优化的差异

- FWD：收到 BF16 X 后，由一个 N-tile 保存 BF16 checkpoint 并原地量化，
  其它 N-tile 等待量化完成；FC2 读取 epilogue 发布的 FP8。
  Network FP8 的 producer 直接返回，计算使用 `tile_cached`，没有走单机
  N256 的 TMA producer/consumer 流水线。
- BWD：Network dU/dX 仍用 `tile_1d2d`，在每个计算 tile 内装载并量化 BF16。
  dW 前用 `prepare_fp8_cache` 从 BF16 缓存生成列量化操作数，随后使用
  `run_fp8_weights`，没有启用单机 `run_fp8_weights_pipeline`。
- 通信、compute workspace、前向 checkpoint 没有缩小；相同工作负载的
  runtime usage 也确认它们与 BF16 一样大。FP8 反向另有列量化缓存，
  `last_backward_tensor_bytes` 反而更大。此字段不是进程总显存。

以上是代码和运行统计确认的差异，尚未做分项消融，不能把全部退化归因于某一项。
这些差异已由后面的源端 FP8 网络路径修正；本节保留原始基线，便于比较迁移收益。

### 正确性与复现

BF16 EP16 的 Torch FWD/dX/dGate/dW1/dW2 对照通过。三个模型的 FP8 EP16
独立六 GEMM 对照均通过：测试 E32、top-k=2、每卡最多 257 tokens，
长度不等且最后一个 rank 为空，覆盖均衡/偏斜路由、空专家、BF16/FP8 混用、
多 tape 异步反向、BF16/FP32 dW、checkpoint、optimizer 和 inference。
上述数值检查是小规模测试；8K/20K 表格是性能测试。

GIN 的参考计算使用 canonical source 顺序及 BF16 X/dR 列量化输入，
不使用单机 row FP8 checkpoint 的反量化值。测试脚本也允许空 source 且无路由
专家的 rank 拥有零字节 checkpoint。没有修改 native kernel。

历史命令和库 SHA256 见本节 JSON。当前 benchmark 脚本默认使用新网络路径；
旧路径的库及日志保留在 `build/fp8-activation-opt-production-*` 和
`build/fp8-multinode/`，对应的旧测试参数不能与新路径混用。

脚本要求两台机器共享相同仓库路径和 native 库，默认使用这两台机器的
bond0 / mlx5_bond_0..7 配置。通用双机 launcher 为
`benchmarks/experiments/run_multinode_training.py`，支持 `--env KEY=VALUE` 覆盖网络环境。

[完整双机结果、命令、库哈希及验证记录](benchmarks/fp8_multinode_20261029.json)
的 source_id 为 `weftmoe-fp8-activation-optimization-20261029`。原始 trace、
测试日志、编译资源在 `build/fp8-activation-opt-production-*`；旧 FP8/BF16
同轮对照 trace 在 `build/fp8-activation-opt-v5-formal/`。


## 2026-10-29 源端 FP8 多机路径

单机与 GIN 的 FP8 PF N256 共用预量化操作数的 FWD/Dgrad 流水线和双缓冲
Wgrad。独立 source kernel 将 X/dY 与 scale 直接写入注册的对称输入区，notify
发布现有协议；Dispatch 不再携带 BF16 X/dY。Return、输出和 dX 保持 BF16。
权重一次产生 W/Wᵀ，权重量化不计入性能测试。

网络 Pull 使用 TMA bulk copy 搬运 FP8 数据，并复用 shared memory 中的数据
保存 canonical row FP8 X，减少一次 peer payload 读取。当前三个 H≤4096 的
profile 都可复用完整行；更宽的行仍使用分段 bulk copy，checkpoint 单独读取。
训练 gate 和反向索引随 payload 保存，不能只依赖 notify：动态 tape 在 notify
之后才分配。AB 保存为 FP8，前反向消费同一份反量化值。

BWD 由 dU owner tile 完成一次 X/dR 列向重量化；SwiGLU 同时生成 Z/dAB 列缓存
和 dAB 行缓存，所有 dW tile 复用，dU 只保留 BF16 ring。网络 FP8 的 canonical
dR 一律独立分配，不能继承 BF16 profile 的 peer dR 开关，因为 streamed 路径已
取消 peer BF16 dR 分配。Qwen122B 的 8K 数值回归覆盖这一分配边界。

已删除 `tile_cached`、`tile_1d2d`、`prepare_fp8_cache`、非流水线
`run_fp8_weights` 及其备用分支。FP8 不满足 PF N256/源端量化条件时直接报错。
FP8 BWD shared memory 按自身流水线大小分配，网络通信 scratch 位于其外，
避免沿用 BF16 dW 大缓冲区导致启动超限。

### 无 profiler 的整轮耗时

两节点、每节点 8×L20X（SM90），EP16；E256，Qwen top-k=8、Flash top-k=6。
每卡名义 tokens 为 8192/20480，实际长度按 10% 标准差变化。10 次预热、30 次
采样，每步取 16 rank 中最大 wall time，再取中位数。两种精度均输出 BF16 dW。
输入、路由和权重预量化在计时外；在线量化、notify/setup、FWD/BWD、释放 tape
和最终 CUDA synchronize 在计时内。旧/新 FP8 与 BF16 的工作负载 SHA256 一致。
旧 FP8 一列为已存档的历史运行，其余为最终生产库的本轮配对运行。单位 ms。

| Profile | Tokens/卡 | BF16 | 旧 FP8 | 新 FP8 | BF16/新 FP8 |
|---|---:|---:|---:|---:|---:|
| qwen35b | 8192 | 13.416 | 25.803 | 18.161 | 0.739× |
| qwen35b | 20480 | 31.426 | 63.262 | 42.928 | 0.732× |
| qwen122b | 8192 | 24.316 | 58.234 | 25.314 | 0.961× |
| qwen122b | 20480 | 56.896 | 138.100 | 60.130 | 0.946× |
| flash | 8192 | 29.297 | 92.910 | 30.439 | 0.962× |
| flash | 20480 | 72.383 | 228.206 | 71.090 | 1.018× |

本轮 1/6 组整轮中位数快于 BF16；小幅收益仍需多轮复测确认稳定性。
通信量缩小不能代替实际性能结论。

### FWD/BWD 主 kernel 与源端量化

另起 CUPTI 运行：10 次预热、5 次采样，按 CUDA launch correlation 归属阶段，
每步取最大 rank kernel duration，再取中位数。此处的主 kernel 包含通信和等待，
不是裸 GEMM，也不包含独立 source quantization 与 notify/setup；不将两个阶段
中位数相加代替上表的整轮测量。单位 ms。

| Profile | Tokens | BF16 FWD | FP8 FWD | BF16 BWD | FP8 BWD |
|---|---:|---:|---:|---:|---:|
| qwen35b | 8192 | 5.143 | 9.273 | 6.092 | 6.808 |
| qwen35b | 20480 | 11.763 | 22.279 | 16.053 | 16.709 |
| qwen122b | 8192 | 12.778 | 11.291 | 11.526 | 11.851 |
| qwen122b | 20480 | 23.204 | 27.291 | 30.205 | 28.699 |
| flash | 8192 | 11.841 | 12.621 | 15.678 | 16.691 |
| flash | 20480 | 29.922 | 26.987 | 41.364 | 44.462 |

独立源端量化为每阶段 0.035–0.155 ms，远小于主 kernel 的毫秒级差距。
逐阶段数据见 JSON；这轮 kernel 级计时不能单独判断各 warp 的等待或寄存器开销。

TMA Pull 的针对性对比使用相同工作负载：Qwen35B 的向量 Pull 版整轮为
22.905/55.285 ms，TMA 候选为 17.976/42.935 ms（8K/20K）。这项对比同时改变
payload 搬运方式与 checkpoint 的 shared-memory 复用，不能把收益分别归给其中一项。
最终三模型矩阵使用上表的正式生产库重新测量。

### 对称通信及计算 workspace

包含 scale 的源数据为 BF16 的 51.56%。Return 与元数据不缩小，所以整个
workspace 不会减半。X checkpoint 按专家 M128 补齐，AB 按 compact rows 保存。
下表为 `compute_bytes + gin_bytes` 的跨 rank 最大值，单位 GiB；不包括独立
forward/backward tensor storage、应用的 BF16 参数、FP8 W/Wᵀ 和框架缓存，
不能视为总显存峰值。Qwen122B 将 dR 从 peer workspace 移到独立 FP8 backward
storage，该行降幅包含分配归属变化。各 rank 的完整 usage 见 JSON。

| Profile | Tokens/卡 | BF16 workspace | FP8 workspace |
|---|---:|---:|---:|
| qwen35b | 8192 | 2.196 | 1.935 |
| qwen35b | 20480 | 3.501 | 3.182 |
| qwen122b | 8192 | 3.951 | 2.689 |
| qwen122b | 20480 | 6.622 | 4.231 |
| flash | 8192 | 3.595 | 3.099 |
| flash | 20480 | 5.141 | 4.530 |

### 验证与复现

最终库的三模型 EP16 完整数值检查通过：E32、top-k=2，Qwen35B/Flash 最多
257 tokens，Qwen122B 最多 8193 tokens；长度不等、最后一个 rank 为空，通信
窗口分别为 128/512。覆盖均衡/偏斜路由、空专家、row scales、多 tape 异步反向、
BF16/FP32 dW、外部 checkpoint、optimizer 和 inference；Qwen35B 混用池另测通过。
参考为独立六次 FP8 GEMM，X/dR 列输入来自保存的 row FP8 反量化值。
最终库 `tests/test_fp8.py`：56 passed。本轮另有 EP8 三模型回归、BF16 Torch
数值对照及 graph/workspace 生命周期回归；对应库 hash 和日志均记录在 JSON。
8K/20K E256 表格是性能测量，不等同于上述 E32 的数值检查。

生产库：`build/fp8-network-production-{qwen35b,qwen122b,flash}/`。
正式日志：`build/fp8-network-v4/{wall,trace,correctness-*}/`。

```bash
python benchmarks/experiments/benchmark_multinode_fp8.py \
  --remote root@10.10.19.233 --ssh-port 2222 \
  --master-addr 10.10.16.219 --output /tmp/fp8-network-wall

python benchmarks/experiments/benchmark_multinode_fp8.py \
  --remote root@10.10.19.233 --ssh-port 2222 \
  --master-addr 10.10.16.219 --trace --samples 5 \
  --output /tmp/fp8-network-trace
```

[完整结果、命令、库 SHA256 与验证记录](benchmarks/fp8_network_source_20261029.json)
