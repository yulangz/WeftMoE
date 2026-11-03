# WeftMoE

面向 Hopper/SM90a 的 BF16 / blockwise FP8 routed MoE 算子，将专家计算与机内、跨机通信流水化。
只替换 routed experts；router、shared expert、attention 和 optimizer 仍由上层框架负责。

| 实现 | 用途 | 通信范围 | 训练缓存与反向 |
| --- | --- | --- | --- |
| `weftmoe_pf` | Prefill、训练、长序列 | 单机 / 多机 EP | 支持 |
| `weftmoe_dt` | 小 batch decode | 单机 EP | 不支持 |

PF 使用静态任务划分、有界计算/通信 ring 和逐槽代际同步。FWD 将 notify 与 main 分开；
训练按实际路由申请并保存 X、gate/up，推理跳过训练缓存。BWD 复用通信基础设施，
计算输入、路由权重与专家权重梯度。训练保存的激活不是固定容量 ring。

单机采用 CUDA IPC/NVLink；跨机采用 NCCL GIN/GDAKI。多机协议按同构节点组织，
已验证单机与双机，更多节点尚待硬件验证。

## 文档

- [使用与 autotune](docs/usage.md)：依赖、编译、Tensor API、配置搜索。
- [框架接入](docs/integration.md)：Megatron、SGLang、AReaL 与共卡切换。
- [Benchmark](docs/benchmarks.md)：运行命令与计时口径。
- [CUDA 实现结构](docs/cuda_structure.md)：资源、执行和验证模块的边界。

## 代码结构

| 目录 / 文件 | 职责 |
| --- | --- |
| `weftmoe/runtime.py` | Python Tensor API、EP group、资源生命周期 |
| `weftmoe/megatron.py`、`sglang.py`、`awex.py` | 框架适配、推理输入布局、权重交换元数据 |
| `src/python/runtime.cpp` | Tensor 绑定、训练 tape、异步调用和 CUDA Graph |
| `src/cuda/activation_transport.cu`、`activation_transport/` | 资源与后端准备、FWD/BWD 执行、传输验证 |
| `src/cuda/device_notify.cu`、`notify/dispatch.cuh` | 输入与 workspace、notify 准备、dispatch 与 consumer 生命周期 |
| `src/cuda/weftmoe_pf/` | PF 单机/多机 FWD、BWD、路由规划与 ring |
| `src/cuda/weftmoe_dt/` | 单机 decode 的调度与计算 |
| `src/cuda/{forward,backward,communication,notify}/` | 共享 GEMM、搬运、同步与 notify 基础组件 |
| `include/weftmoe/` | C++ API、配置、metadata 与资源接口 |
| `configs/`、`tools/autotune.py` | 三种 shape 的初始 profile、统一搜索空间与离线调优 |
| `benchmarks/`、`tests/` | 同口径性能比较、数值与生命周期回归 |
| `build_native.py`、`tools/build_tensor_runtime.py` | native 库与 Python 绑定构建 |

## 支持范围

支持 BF16 或 E4M3 blockwise FP8 计算（FP32 累加）；FP8 权重预量化，激活支持在线或预量化输入。
输出与输入梯度为 BF16，框架接入默认 BF16 dW。专家为无 bias 的 SwiGLU，专家均匀分片且 `E % EP == 0`。
native 库按 hidden/intermediate 编译；其他模型可配置 shape 后重新编译或 autotune，
不是修改 Python 参数就能复用任意二进制。单机 EP2/4/8 等拓扑可用，具体容量取决于 profile 与显存。
多机需要相同的每节点 GPU 数、连续排列的机内 EP rank，以及可用的 GDAKI 网络。

当前维护三种 benchmark shape：Qwen3.5-35B-A3B、Qwen3.5-122B-A10B、DeepSeek V4 Flash。
这些预设描述单层专家尺寸，不表示 benchmark 会加载完整模型。

许可证：[Apache-2.0](LICENSE)。
