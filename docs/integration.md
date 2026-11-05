# 框架接入

先完成 [native 库和 Tensor 绑定构建](usage.md)，确保每个 worker 都能 import `weftmoe`，
并收到同一个 shape 对应的 `WEFTMOE_TENSOR_EXTENSION`。adapter 归 WeftMoE；
RL 启动、调度、权重同步及共卡生命周期 hook 归 AReaL，不需要把 CUDA 实现复制进框架。

## Megatron

```python
from weftmoe.megatron import MegatronAdapter

# models：本 rank 的 Megatron model chunks，已初始化 EP group。
adapter = MegatronAdapter(models, max_tokens=max_local_moe_tokens)
adapter.prepare()
# 之后使用原有 model forward/backward、DDP 和 optimizer。
# 共卡 offload 前：adapter.suspend()
# 整个引擎退出时：adapter.close()
```

替换范围是 `MoELayer` 的 routed experts。router 和 shared experts 仍由 Megatron 执行，
返回普通 parameter gradients，沿用 DDP hooks。多层共享固定的 expert 参数拼接缓冲区和通信 runtime。
`max_tokens` 应是 TP/CP 切分后本 rank 实际进入 MoE 的最大行数，不是全局序列总长。

要求 ETP=1、BF16、无 bias 的 SwiGLU；关闭 `overlap_param_gather` 和 shared-expert overlap。
当前 adapter 不支持 latent MoE、input-probability placement、激活 clamp/offset 或部分 MoE 重算。
支持整层 checkpoint：首次 `no_grad` FWD 不保存 X/GU，BWD 中重算 FWD 时才保存。

设置 `WEFTMOE_FP8=1`（或 `MegatronAdapter(..., fp8=True)`）启用 routed experts 的
blockwise FP8 FWD/BWD，dW 返回 BF16；模型参数和优化器精度不变。
未绑定 master 时，默认每次 forward 从参数重新量化，避免 `.data` 更新绕过 Tensor version
后静默使用旧权重。未绑定时原生缓存只保留一个槽位，活跃 tape 单独保留所需版本。BF16/FP32 参数现量化不做主机有限性检查，不在每层或重算时读取 GPU
标量；非有限块通过 NaN scale 传播。导入的 FP8 权重仍检查字节和 scale，并同步一次。
当前优化器接入范围是 AdamW 和下述 AReaL GPU-staged Muon。
`adapter.bind_optimizer(optimizer)` 只接受具有 step post hook 的 `torch.optim.AdamW`，
由 hook 自动失效缓存，从而在 micro-batch 和重算之间复用；checkpoint load 等 step 之外的
直接写入仍需调用 `adapter.weights_updated()`。
Megatron 的 `MegatronOptimizer` / `ChainedOptimizer` 包装器不支持此绑定，会明确抛出
`TypeError`；不会递归绑定其内部 optimizer，以免在完整 step 结束前发布部分权重。
使用这些包装器的 AdamW 用户应不调用 `bind_optimizer()`，保持每次 forward 异步现量化。
router/shared experts 保持原实现。

AReaL 的 GPU-staged Muon 在初始化后调用 `adapter.bind_fp32_masters(optimizer)`。
当前支持 expert DP=1、ETP=1：routed expert 的 BF16 GPU storage 释放，不保留 CPU BF16 副本；
Parameter 对象、DDP hooks 和 FP32 梯度 buffer 保留。每次更新直接在 Muon compute stream
将 FP32 master 量化为 W/Wᵀ，跳过 BF16 写回，所有 micro-batch 和重算复用。
第一次 master 写入立即使旧版本和旧 tape 失效，未发布期间禁止新计算；原生写入也显式更新
Tensor version，直接持有旧 `PreparedFP8Weights` 的调用者同样会报错。更新必须发生在全部
反向完成后，不支持与旧版本计算并发覆写。引擎在完整 optimizer step 后调用 `adapter.weights_updated()`；发布会等待所有量化 writer
stream，再记录消费者等待的 ready event。`close()` 从 master 恢复普通 BF16 参数后解除绑定，
因此退出 adapter 后仍可读取模型 checkpoint。
`suspend()` 释放 FP8 缓存，不保留 CPU FP8 W/Wᵀ 备份；`release_bf16_weights()` 从
已有 CPU FP32 master 重建 GPU FP8 缓存。重建仅在恢复计算/传权阶段发生，micro-batch 间复用。
checkpoint 按需从 FP32 master 生成临时 BF16 参数。FP32 直接量化不承诺与经过 BF16
中间舍入的版本逐位相同；训练与 rollout 通过 AWEX 复用同一份 FP8 字节和 scale。
量化、搬运、传权与 runtime/graph 重建均计入端到端耗时。

BF16 Muon 共卡 offload 也会实际释放没有 DDP `param_data` 的参数。完整且独占 storage 的
本地专家直接释放 BF16 storage，以 CPU FP32 master 为权威来源，不做搬运和逐位比较，
也不保留 CPU BF16 备份。onload 保留原 Parameter/Storage identity，并从 master 恢复。
共享存储或没有完整本地 master 的参数使用原字节备份。释放前等待 optimizer 完成，
恢复前检查 master 身份和版本；checkpoint lease 先恢复权重，再允许修改 master。

正常热路径不逐层 CPU drain，也不 collective 确认每次分配成功。
notify 返回实际行数后申请训练缓存的同步边界仍然保留，并非全流程没有 CPU 等待。
`suspend()` 在共卡边界等待完成、检查不存在 live tape，并释放训练 workspace 和 expert 参数拼接缓冲区，
只保留 NCCL session；下一次训练按需重建 workspace。重建开销计入端到端运行时间。
分配异常向上传播；不能捕获后继续复用已中断的通信状态。

## SGLang

在每个 scheduler worker 创建模型前调用：

```python
from weftmoe.sglang import install
install()
```

只在启动父进程调用不足以覆盖 spawn 出来的 worker。当前接入的是 BF16 权重的 `FusedMoE`，
要求 `ep_size == tp_size`、expert TP=1，不直接接收 SGLang 的量化 checkpoint。
这里的 `tp_size` 是 SGLang 的进程组大小；开启 DP attention 后可形成 DP=EP 的布局。

BF16 的 PF/DT 通过 5D TMA 直接读取原始 FC1 权重，在共享内存形成 gate/up 排列，
不保存额外的显存重排副本。训练侧同样直接读取原始布局；Megatron 的跨 expert 参数拼接
缓冲区仍按形状复用。SGLang 在计算 KV/Mamba 容量之前预热 PF/DT workspace，
使其计入实际可用显存。否则按原始模型权重估算的 KV pool 会过大，
后续长 prefill 的 attention 临时张量可能 OOM。`mem_fraction_static` 无需因此手动调低；
可容纳的 KV token 数会按持久内存相应调整。FP8 同样在 KV 预算前完成量化、释放 BF16
专家 storage，并预热 PF/DT workspace，避免预算仍按 BF16 权重或遗漏 workspace 计算。

FP8 rollout 在初次加载后量化并释放该层 BF16 GPU storage。后续 AWEX 更新直接接收训练侧
FP8 权重字节（uint8 view）和 FP32 block scale，不恢复 BF16，不重新量化。
AWEX 确定下一次完整覆盖权重时调用 `suspend(..., discard_fp8=True)`，只保留张量元数据，
不拷贝旧 FP8 到 CPU；接收时分配 GPU 空间，`commit_fp8_weights()` 后才允许推理。
普通 suspend/resume 需要继续使用旧权重，默认仍保存 CPU 副本，恢复时保留原版本号。
启用 TMS 时，初始专家 BF16 分配使用临时普通 CUDA pool，避免其释放后仍由 TMS 持久池
保留并在 onload 时重新占用显存。此兼容路径使用 PyTorch 2.9 / 当前 TMS 的内部接口，
只在触发该 pool 兼容路径时检查 PyTorch 2.9.x 和所需接口；其它版本会明确报错。
升级依赖时需重跑包含 TMS、权重更新和图重捕获的 8 卡生命周期测试。
重新捕获前重置 decode 的 dummy request、KV 和 Mamba 输入，避免使用上一批请求的索引。

`WEFTMOE_FP8=1` 可从 BF16 模型权重准备 FP8 专家缓存，prefill PF 和 decode DT 都使用
FP8。suspend 会释放权重和 workspace 地址，因此即使 KV-only 恢复也必须重新捕获图；
完整更新走上面的 discard 路径，普通恢复保留原权重。日志单独输出
`WeftMoE decode graph recapture wall time`，不把它混入权重传输耗时。
AWEX 包和协议不变，兼容代码位于 AReaL 的 adapter 层。

- Prefill 使用 PF，decode/CUDA Graph 使用 DT；按集体执行阶段选择，不按各 rank 的局部 token 数独立切换。
- TP attention 输入先按行分片，结果按模型原有 TP 布局合并。
- DP attention 保持 DP-local 输入/输出，避免额外的 MoE all-gather/reduce-scatter；
  Qwen3.5 shared expert 在推理侧复制，AWEX 的 sharding 元数据必须与之匹配。

DT 是机内协议，因此 rollout EP group 应留在单节点；训练 EP16 不要求 rollout 也使用跨机 EP16。
不能只设置 `AREAL_USE_WEFTMOE=1` 就让原版 `python -m sglang.launch_server` 自动安装 hook；
可使用下面已接入的 AReaL worker，或在自己的 worker 入口显式调用 `install()`。

BF16 共卡切换由 `suspend(model_runner)` / `resume(model_runner)` 管理：默认保留 DT 图，
释放 prefill runtime；原地址的 BF16 权重更新完成后可直接回放，无需刷新权重副本或重新 capture。
显存紧张可设 `WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS=0`，代价是释放后需要重新准备图。
模型权重地址改变时不能继续回放旧图。最终退出使用 `close(model_runner)`。

## AReaL

配套 AReaL 提交为 `88572016`（`feat/weftmoe-fp8-residency`），包含 FP8 AWEX 和权重
生命周期改动；不是任意上游版本均已内置接入。
关键位置：

| AReaL 位置 | 职责 |
| --- | --- |
| `areal/engine/megatron_engine.py` | 创建 adapter，按 TP×CP 估算容量，offload/退出时调用生命周期 hook |
| `areal/engine/awex/` | 权重交换、shared-expert 布局和共卡切换 |
| `areal/v2/inference_service/sglang/` | SGLang server/scheduler worker 安装 hook |
| `examples/cpu_staged_offload/` | DAPO RL、Muon 与 optimizer offload 配置 |
| `benchmark/weftmoe/` | 真实训练回放、权重回放、profile 与分析工具 |

在 **actor 和 rollout 各自的** `scheduling_spec[*].env_vars` 中传入：

```yaml
AREAL_USE_WEFTMOE: "1"
WEFTMOE_TENSOR_EXTENSION: "/absolute/path/to/_runtime.cpython-312-x86_64-linux-gnu.so"
WEFTMOE_AUTOTUNE_TABLE: ""
WEFTMOE_STATIC_RING_TOKENS: "2048"
WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS: "2"
WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS: "1"
```

扩展文件名以本机 Python 构建结果为准；同时传递所需的 `PYTHONPATH`、`LD_LIBRARY_PATH` 和网络环境。
开启 autotune 表时分别核对 actor/rollout 的 training 标记、容量与图 bucket；
不能把一张仅含训练项的表直接用于两侧。运行 DeepEP baseline 时两侧都去掉 WeftMoE 开关，
Megatron 显式选择 Flex/DeepEP；两组保持模型、数据、并行策略和缓存口径一致。

已通过的快速 RL 配置：35B Base、DAPO-Math-17k、8K、actor TP2×DP4/EP8、
rollout TP8/EP8、Muon CPU-staged offload、整层重算、8 prompts ×4 samples、3 steps。
对应例程为 `examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py`；
8K Muon 配置为同目录 `dapo-math_grpo_qwen3_5_122b_muon.yaml`，测试 35B 时覆盖
`actor.path` 并使用 35B native profile，不能仅按 YAML 文件名判断实际模型。

SGLang 服务端 context 需要为中断后续生成留出输入余量。SGLang 0.5.10.post1 的
`max_req_input_len = min(context_len - 1, max_token_pool_size - 1) - 5`，
且拒绝长度大于或等于该值的输入，即使未启用 speculative decoding 也适用。
8K 配置可保留 `gconfig.max_tokens=8191` 和 proxy `engine_max_tokens=8191`，
将 `sglang.context_length` 设为 `8208`（同时确保 KV pool 足够），训练 microbatch token 上限独立设置。
不要用自动截断绕过续生成长度检查，以免已有 token、logprob 和训练轨迹不一致。

短测关闭 saver/recover 的 steps、epochs、secs 频率，`recover.mode=disabled`；
需要 W&B 时显式配置 online 和凭据，不把密钥写进仓库。RL 回归结果见 [Benchmark](benchmarks.md)。
