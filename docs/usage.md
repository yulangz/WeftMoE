# 使用与 autotune

## 安装与自动编译

需要 Linux、Python ≥3.10、C++17、支持 SM90a 的 CUDA、CUTLASS/CUTE、NCCL，
以及运行 Tensor API 所需的 PyTorch。多机还需要 NCCL device API 与 GIN/GDAKI；
不支持以 Proxy 后端代替。已验证环境包括 CUDA 12.9、NCCL 2.31.2、PyTorch 2.9.1。
PyTorch 和框架依赖由使用者安装，不由 WeftMoE 自动选择版本。

安装方式与模型无关，wheel 包含编译所需的 native 源码。安装阶段不编译 CUDA；首次准备算子时需要本机工具链和上述开发头文件。路径按实际环境填写：

```bash
export CUDA_HOME=/usr/local/cuda
export CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
export NCCL_INCLUDE_DIR=/path/to/nccl/include
export NCCL_LIB_DIR=/path/to/nccl/lib
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -e . --no-build-isolation
# 可选；默认 ~/.cache/weftmoe。多进程可共享同一目录。
export WEFTMOE_ROOT=/path/to/weftmoe-cache
```

无需设置 `NVCCFLAGS` 或 `WEFTMOE_TENSOR_EXTENSION`。`TensorRuntime` 创建/adapter 首次准备 runtime 时：

1. 查询本次请求的已编译缓存，命中后直接加载。
2. 未命中则查询 `WEFTMOE_AUTOTUNE_TABLE`（默认 `$WEFTMOE_ROOT/tuning.json`）；精确 shape 和长度桶命中时，按已验证的配置编译。
3. 没有匹配配置则使用已验证 shape 的默认配置编译：H/I 为 2048/512、3072/1024 或 4096/2048。其他 shape 需要先通过手动 autotune 的数值检查，或显式使用 `TensorRuntime(..., allow_unvalidated_shape=True)` 进行实验性编译；不能把维度整除当作正确性证明。所有自动生成的 profile 仍要求 H/I 不小于 512 且为 256 的倍数。

该流程只编译和加载，不运行搜索、计时或 benchmark。缓存包含源码、配置、Python/PyTorch ABI、CUDA 架构、工具链及依赖标识；不同 shape/配置的库使用独立名称，可共存于同一进程。多进程用文件锁避免重复构建；失败构建不会发布完整缓存，错误信息指向 `build.log`。

长度选择按 runtime 的 **`max_tokens` 容量**，不是每次调用的实际 token 数。一个 runtime 生命周期内固定实现，训练 tape 和 CUDA Graph 不会被中途换库；需要不同桶时在 capture 外准备对应容量的 runtime。先预热再捕获 CUDA Graph。

先检查 tune 表的文件版本（仅 stat，不解析或选配置），再查有效的已编译请求缓存。手动 tune 发布新表后，旧请求选择自动失效，下次准备 runtime 会采用新配置；未改动的表仍优先命中编译缓存。版本检查不修改已加载的 runtime、tape 或图；新选择仍会复用匹配的二进制产物。构建 header 保留在确定性缓存目录中，失败重试可复用已经编好的对象文件。

需要固定手工构建时，仍可使用 `build_native.py lib --output ...` 和 `tools/build_tensor_runtime.py --library ...`，然后设置 `WEFTMOE_TENSOR_EXTENSION`。该变量显式覆盖自动选择，使用者负责 shape 和一致性；旧 schema-1 表仅供这种固定库路径使用，不与新的自动 tune 表混用。

`get_include()` 返回公开头文件目录。`get_library_path()` 仅用于旧式预构建安装；lazy wheel 没有统一的 `libweftmoe.so`，因此该调用会明确报错。C++ 用户使用 `build_native.py lib --output ...` 显式构建并链接该路径；Tensor 用户使用 runtime 或下面的预热入口。`tests/check_install.py --wheel ...` / `--editable` 验证源码载荷和头文件，另可传入 `--native-library ...` 验证显式 native 库的 C++ 链接与运行。

### 多机启动前预热

首次完整 CUDA 编译可能需要数分钟，取决于 CPU、并发度和缓存状态。不同节点缓存不一致时，已准备好的 rank 会在 collective 等待其他节点，可能耗尽调用方配置的 process-group timeout。不要依赖某个固定的编译耗时上限。

在**每个节点、初始化 process group 之前**运行相同的预热命令；它不建立 NCCL process group，也不运行 benchmark：

```bash
python -m weftmoe.jit prepare --hidden 2048 --intermediate 512 \
  --experts 256 --top-k 8 --max-tokens 8192 \
  --nodes 2 --gpus-per-node 8 --mode training --dtype fp8
```

预热与实际 worker 必须使用相同 Python、依赖、`WEFTMOE_ROOT`、tune 表以及 shape、拓扑、精度、模式和容量；混合 GPU 型号时分别指定 `--device`。FP8 预热同时准备量化 helper。自定义未验证 shape 可显式使用 `--allow-unvalidated-shape`。未预热时需在框架创建 process group 处配置能覆盖首次编译的 timeout；WeftMoE 不修改调用方的 timeout。

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

主通信池每个 slot 和独立 BWD peer payload 的申请长度向上对齐到 64 KiB，避免 CUDA IPC 导入时因长度
未对齐而可能将整段 peer 映射限制为 4 KiB 小页。内部张量偏移、容量与 MNK 不变；
每块分配增加不足 64 KiB，`usage()` 的显存统计包含该尾部。对齐解除长度对
大页的限制，实际页大小仍受映射基址、偏移、物理分配和驱动限制。
GIN 注册窗口使用 `ncclMemAlloc` 的 VMM 分配与 NCCL window 映射，
由 NCCL 处理其分配粒度，不套用此 CUDA IPC 长度取整。

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
GEMM 对反量化后精确参考的算术检查单独使用 `diff < 1e-6`，不包含 FP8 量化误差。
不使用逐元素相对容差。布局、图回放、预量化输入等数学等价路径仍要求精确一致。
训练 PF 与固定 M64 DT 的前向另设严格逐位回归，入口为 `tests/torchrun_pf_dt.py`，
每份训练 tape 均实际执行反向。`tests/torchrun_dt_fp8.py` 的
`TEST_DT_ALLOW_KNOWN_ONE_TOKEN=1` 只允许 Qwen35B 单 token 的已知误差豁免；
该豁免不适用于训推逐位一致性测试。本轮性能测量使用 64/512 tokens，没有使用该豁免。
当前测量范围、计时口径与结果见 [Benchmark](benchmarks.md)。

构建方式与上面的 BF16 Tensor API 相同，无需安装 DeepGEMM。自动编译要求 `H/I` 为不小于 512 的 256
倍数。先在 forward/backward 外准备权重，再显式传给 `moe`：

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
`scales[tokens,H/128]` 为 FP32 正有限 2 的幂，`block_rows=1`，data 指针须 16 字节对齐。
布局和对齐会在入口检查；scale 内容是外部生产者的契约，热路径不做需要设备同步的
逐值检查。违反该契约可能产生非有限输出或反向列转换的额外舍入误差。
推荐使用 `quantize_fp8` 生成正常有限输入的表示；非有限输入通过 NaN scale 传播。
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
然后重新预热和 capture。Megatron/SGLang adapter 可通过 `WEFTMOE_FP8=1` 启用，
生命周期要求见 [框架接入](integration.md)。
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
多机 GIN 也使用源端 FP8。`fp8_workspace=True` 缩减源输入与计算 ring，
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
沿用 torchrun 和扩展环境变量。历史测试与原始数据见 [Benchmark](benchmarks.md)。

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

## 手动 Autotune

[统一脚本](../tools/autotune.py)由用户显式运行，配置见 [autotune.json](../configs/autotune.json)。以 shape 默认配置为起点，离线编译候选、筛选、反序确认、检查正确性并导出配置；不承诺全空间最优。

```bash
# 只检查计划，不启动 GPU 测试。
python tools/autotune.py --model qwen35b --dry-run
# 两个桶分别选择实现，默认为 BF16；FP8 加 --fp8。
python tools/autotune.py --model qwen35b --mode training --tokens 8192 20480
python tools/autotune.py --model qwen35b --mode training --fp8 --tokens 8192 20480
python tools/autotune.py --model qwen122b --mode inference
python tools/autotune.py --model qwen35b --mode decode --fp8
# 也可覆盖 --hidden/--intermediate/--experts/--top-k；不按模型名称模糊匹配。

# 双机 PF：共享仓库/缓存绝对路径、相同 Python 环境、免密 SSH。
python tools/autotune.py --model flash --mode training --gpus-per-node 8 \
  --remotes user@node1 --master-addr node0-address --ssh-port 22
```

主要配置项：

| 配置 | 内容 |
| --- | --- |
| `workload` | 模型、tokens；可覆盖 `hidden/intermediate/experts/top_k` |
| `batch_invariant` | 默认 `true`；固定算术路径，剪除改变 M/N/K、MMA 分解、量化等策略的候选 |
| `token_buckets` | 默认 `[8192, 20480, 65536]`，对应 `[0,8192]`、`[8193,20480]`、`[20481,65536]`；更大容量按上界翻倍扩展 |
| `system` | 每节点 GPU 数、远端节点、rendezvous/SSH 地址 |
| `forward/activation/weight/decode` | `axes` 笛卡尔积与 `variants` 显式组合；默认一致性模式会自动剪枝 |
| `ring.local/network` | 窗口 tokens、contribution 深度 |
| `screen/final`、`confirmation_*` | 预热、采样、确认次数与最小收益门槛 |
| `workspace_limit_bytes` | native workspace 限额，不含全部权重、训练缓存和框架显存 |
| `environment`、`run` | 编译/网络环境、并发编译数、输出与缓存路径 |

CLI 覆盖配置。`--token-buckets 8192 20480 65536` 设置长度上界；`--tokens` 提供各桶用于计时的代表负载。同桶多个代表负载共同选择一个实现，不同桶独立搜索。DT 仅支持单机。

默认 `--batch-invariant` 不仅固定算术参数，还对最终候选执行严格 PF 训练（保留真实反向缓存）/PF 推理/DT、不同 batch 长度与 token 位置、重复前向的逐位比较，并检查重复反向的有限值与 `diff ≤ 1e-6`。覆盖小 M 边界和桶端点；候选与参考各自使用对应的 ring 参数，PF 推理与 DT 通过 graph 接口验证，重复调用保留输入地址。多机不运行 DT。所有桶对照同一 shape 的默认算术路径，避免各桶独立通过却互不一致。普通 FP8 对 BF16 的误差检查另行保留。

显式 `--no-batch-invariant` 可扩大搜索，但结果标为非一致性配置。部署时还需显式传入 `TensorRuntime(..., batch_invariant=False)` 才会采用它；默认加载器忽略该配置。直接设置自定义二进制不受此保护。

输出按 shape、模式、EP、长度桶分目录。`result.json` 保留候选/测量/验证，`table.json` 保存该桶的生成 header、源码标识和 ring 配置，`runtime.json` 给出部署环境。脚本还将结果原子合并到 `--table` 指定的共享表，默认 `$WEFTMOE_ROOT/tuning.json`。

查表 **精确匹配 GPU、dtype、H/I/E/K、节点拓扑、模式、输入布局**；只有 tokens 使用区间匹配，拒绝重叠区间。源码过期、无精确 shape、未覆盖容量或一致性不兼容时，仅已验证 shape 回退默认配置，不寻找相近模型。未知 shape 的查表结果还必须带有调优脚本在通过数值校验后写出的 `validated` 标记；不匹配时默认报错。BF16/FP8、训练/PF 推理/DT 分别记录性能选择，但默认共享前向算术约束。编译时 ABI、工具链及依赖变化还会使二进制缓存失效。

不使用 tune 表时也可显式设置 `WEFTMOE_STATIC_RING_TOKENS` 和 `WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS`。新调优建议使用统一脚本；`autotune_dt_fp8.py` 是独立的诊断 sweep，不发布自动加载表。

## 快速验证

```bash
python -m pytest -q
torchrun --standalone --nproc-per-node=8 benchmarks/torchrun_training.py \
  --backend pf --model qwen35b --tokens 129 --warmups 0 --samples 2 --verify
```

GPU 命令在首次准备时自动构建匹配扩展。完整计时与多机启动方式见 [Benchmark](benchmarks.md)。
