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
返回普通 parameter gradients，沿用 DDP hooks。多层共享固定权重重排 workspace 和通信 runtime。
`max_tokens` 应是 TP/CP 切分后本 rank 实际进入 MoE 的最大行数，不是全局序列总长。

要求 ETP=1、BF16、无 bias 的 SwiGLU；关闭 `overlap_param_gather` 和 shared-expert overlap。
当前 adapter 不支持 latent MoE、input-probability placement、激活 clamp/offset 或部分 MoE 重算。
支持整层 checkpoint：首次 `no_grad` FWD 不保存 X/GU，BWD 中重算 FWD 时才保存。

正常热路径不逐层 CPU drain，也不 collective 确认每次分配成功。
notify 返回实际行数后申请训练缓存的同步边界仍然保留，并非全流程没有 CPU 等待。
`suspend()` 在共卡边界等待完成、检查不存在 live tape，但保留通信池；`close()` 才释放。
分配异常向上传播；不能捕获后继续复用已中断的通信状态。

## SGLang

在每个 scheduler worker 创建模型前调用：

```python
from weftmoe.sglang import install
install()
```

只在启动父进程调用不足以覆盖 spawn 出来的 worker。当前接入的是 BF16 `FusedMoE`，
要求 `ep_size == tp_size`、expert TP=1，不支持量化专家。
这里的 `tp_size` 是 SGLang 的进程组大小；开启 DP attention 后可形成 DP=EP 的布局。

- Prefill 使用 PF，decode/CUDA Graph 使用 DT；按集体执行阶段选择，不按各 rank 的局部 token 数独立切换。
- TP attention 输入先按行分片，结果按模型原有 TP 布局合并。
- DP attention 保持 DP-local 输入/输出，避免额外的 MoE all-gather/reduce-scatter；
  Qwen3.5 shared expert 在推理侧复制，AWEX 的 sharding 元数据必须与之匹配。

DT 是机内协议，因此 rollout EP group 应留在单节点；训练 EP16 不要求 rollout 也使用跨机 EP16。
不能只设置 `AREAL_USE_WEFTMOE=1` 就让原版 `python -m sglang.launch_server` 自动安装 hook；
可使用下面已接入的 AReaL worker，或在自己的 worker 入口显式调用 `install()`。

共卡切换由 `suspend(model_runner)` / `resume(model_runner)` 管理：默认保留 DT 图与重排权重，
释放 prefill runtime；恢复权重后刷新 packed weights，避免每轮重新 capture。
显存紧张可设 `WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS=0`，代价是释放后需要重新准备图。
模型权重地址改变时不能继续回放旧图。最终退出使用 `close(model_runner)`。

## AReaL

本轮验证对应 AReaL `dev@d7023ba9`；不是任意上游版本均已内置接入。
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

短测关闭 saver/recover 的 steps、epochs、secs 频率，`recover.mode=disabled`；
需要 W&B 时显式配置 online 和凭据，不把密钥写进仓库。RL 回归结果见 [Benchmark](benchmarks.md)。
