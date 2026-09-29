# Qwen3.5 接入 AReaL

## 当前 V4 接入检查（2026-09-29）

本轮基线为 HF（`7569ae0`），后续 HG 的 EP 通用化验收见
[decode 记录第 6 节](static_forward_v4_decode_record.md)。下面的旧 RL 性能数字来自历史实现，不能用作
当前 V4 的端到端结论；当前版本尚未完成新的 RL baseline 对照。

本轮仅替换 SGLang rollout，Megatron actor 保持 baseline。使用
`ops/AReaL/examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py` 和同目录的
`dapo-math_grpo_qwen3_5_cpu_staged.yaml`。已有比较脚本的 `BACKEND=weftmoe`
会同时启用 actor，因此必须保持 actor 的 `AREAL_USE_WEFTMOE=0`，并用独立的
`rollout.scheduling_spec` 设置 rollout 的 `AREAL_USE_WEFTMOE=1`、
`NCCL_CUMEM_ENABLE=1`、tensor extension 和 `WEFTMOE_STATIC_RING_TOKENS=512`。
不能在原本继承 actor 的 scheduling spec 下只改嵌套环境变量。

旧配方的 rollout 是两个独立 TP4/EP4 引擎，默认没有启用 DP attention。
Qwen 的 256 个专家在 EP4 下每卡 64 个：HF 的单轮 warp scan 仅支持每卡 32 个，
HG 已改为分段 scan，验证了 EP1/EP2/EP4/EP8。旧适配层统一使用 prefill 大池，
也选不中 decode 专属入口；现在为等长补齐的 decode capture bucket 准备容量
64 token/rank 的独立小池，较大 bucket 与 eager/prefill 使用通用 V4。

`benchmarks/run_areal_sglang.py --areal-root <AReaL> --backend baseline|weftmoe`
复用原比较脚本，仅用已有配置覆盖隔离 rollout 开关，不修改 AReaL 的调度逻辑。
`--dry-run` 可查看覆盖参数，模型、数据、AWEX 和实验目录仍由原环境变量指定；
两组默认只记录本地指标（W&B disabled），保持原 20 updates/完整训练 horizon。
配置解析已验证 actor 始终为 baseline，只有实验组 rollout 开启 WeftMoE。

本轮最小测试发现旧适配层 warmup 与 outer capture 的中间 tensor 地址不同，
触发当前 native runtime 的稳定地址检查。适配层现为每个 token bucket 保存
一份 X/routes/gates staging，所有串行层共享；拷贝进入 outer graph，子图绑定
固定地址。eager prefill 不增加此拷贝，suspend 销毁图和 native runtime 后清理
staging。原先“不复制输入”的描述只适用于调用者能提供稳定地址的 native API，
不再适用于这一 SGLang 适配路径。EP4 的 E16/K4/T17 与 E256/K8/T17、T257
真实 SGLang 单层测试已通过 eager 数值对齐、变更输入/路由后的图回放和 runtime 释放/重建；
这不是整模型或 RL 验证。

整模型首次捕获发现每个 bucket 都重复分配全部 40 层的 packed W1：每增加一个
bucket 消耗约 10 GiB/rank，最终在第 8 个 bucket OOM。native graph preparation
现按 layer 缓存 session-scoped 的只读 weight lease，同一个 runtime 的全部
bucket 共用；`invalidate_inference_layer()` 同时销毁该层所有 plan 和 weight
lease。两套 runtime 各保留一份，eager 和训练的权重发布方式不变。

修复后完整 Qwen3.5、TP4/EP4、context 32768、mem fraction 0.4、graph max bs 512
成功捕获全部 46 个实际 bucket（SGLang 按 request 容量将最大值裁至 408）。
连续 3 轮图/KV 卸载、恢复、重新捕获的 greedy token 完全一致；恢复态约
85,600–86,300 MiB/card，不再按 bucket 累积 packed W1。该检查未卸载模型权重，
不能替代 RL 的 AWEX 权重更新验证。缓存改动后另有 EP4/E256/K8 的 124 项 native
检查通过，包含跨 bucket 复用、权重失效/刷新和 batch-invariance。
日志为 `build/local_decode_hg/sglang_full_server_shared_weights.log` 和
`sglang_offload_cycles.log`；最终 tensor extension SHA256 为
`f894dae763a4b95df6c9a662278c862a980502b9a8a551db449e20a03e6e33db`。

应用侧 tensor 绑定支持 BF16 的 Qwen3.5-35B-A3B routed experts：H=2048、I=512、
无 bias 的 SwiGLU、ETP=1。它需要 PyTorch、已安装的 Megatron Core/Bridge 与 SGLang
环境，以及 AReaL 侧的接入改动。这是一个显式构建的开发适配层，只执行 `pip install .`
不会构建 CPython tensor 扩展。

## 构建

在 WeftMoE 目录下、激活训练用 Python 环境后：

```bash
NVCCFLAGS="-include $PWD/configs/communication_spsc_20260916/qwen.hpp" \
  python build_native.py lib --output build/libweftmoe_qwen.so --jobs 12
python tools/build_tensor_runtime.py --library build/libweftmoe_qwen.so
export WEFTMOE_SOURCE_ROOT="$PWD"
export WEFTMOE_TENSOR_EXTENSION="$(realpath build/_runtime*.so)"
```

两个节点上 native 库和 tensor 扩展的路径必须一致。两边都要使用带 GIN 支持的
NCCL 2.31.2 及以上版本。

## 运行时归属

- Megatron 保留自己的 router、shared experts、参数、优化器和 DDP 梯度钩子。
  routed 路径调用融合的前向/反向算子并缓存 X 与 AB。专家参数复制进所有层共享的
  一块固定 packed-weight 工作区；反向执行前恢复该层权重。每个训练驻留阶段内，
  所有层共享一份传输工作区和 EP 组的一份通信 context。
- SGLang 使用 TP=EP=4，把复制的 attention 输入按 EP rank 分片，并把路由输出还原
  成 Qwen 外层 TP 归约所需的形式。它不保留 X/AB，也没有反向状态。
- decode 为每一层、每个 token bucket 捕获一个子 CUDA graph，共享一份串行化的传输
  工作区。graph plan 绑定固定的 X/routes/gates staging 地址和 output；SGLang
  的外层图先复制当前层输入到 staging，不 clone output。
  外层 graph 中每层有一个 child graph node，其内只有一个融合 route/communication/
  compute 的 inference kernel。动态 prefill 复用同一套已准备好的 runtime，不为每个长度
  单独缓存。该共享工作区不支持不同 stream 并发使用。
- AReaL 的 offload 钩子会先销毁推理图，再释放 native CUDA、IPC 和 GIN 分配；
  resume 每个驻留阶段重建一次 context 并重新捕获图。训练侧 offload 拒绝存在存活
  反向 tape 的情况。context 不会按层或按 token 重建。
  WeftMoE 需要开启 NCCL cuMem，launcher 会把该设置显式传给两个角色。AWEX 插件和
  v2 SGLang 入口都已接入。CUDA Graph 保持开启，但
  `SGLANG_MEMORY_SAVER_CUDA_GRAPH=0` 关闭了 TMS 的图内存重映射：这些图是被显式
  销毁再重建的，而 TMS 期望图分配一直存活。权重与 KV cache 的卸载仍走 TMS。
  适配层在驻留切换前后保持自己的执行 stream 与 SGLang 的捕获 stream。SGLang 的
  custom all-reduce 复用其已有的 staging buffer 路径，避免把已销毁图上的分配
  永久注册为 IPC。这些选择可以避免按 stream 的工作区和旧图分配在多次 update 中
  不断累积。
- 外部 producer 的 execution token 通过 producer handle 持有后端，而不是在后端里
  自引用。这个问题同时影响训练 tensor 和推理驻留：此前的所有权环会让输入 tensor
  和 NCCL session 在 runtime 销毁后仍然存活。
- 不同进程看到的 GPU 可见性可能不同。本地 IPC bootstrap 用 UUID 识别物理 GPU；
  GIN 校验 NCCL team rank 而不是 CUDA device ordinal。这样 AReaL actor 即使各自
  只能看到被分配的一张卡也能正常工作。Megatron 在初始化阶段就准备好 native
  context，使拓扑错误在 rollout 之前暴露；随后的首次 offload 会正常释放它。

## 实验启动器

在打过补丁的 AReaL 仓库中使用
`examples/cpu_staged_offload/run_qwen35_comparison.sh`，并通过环境变量提供模型与
数据集路径、共享的 `AREAL_FILEROOT` 和 `WANDB_API_KEY`。`WANDB_BASE_URL` 默认是
`http://8.150.1.98:8080`。环境中的 AWEX 必须注册
`Qwen3_5MoeForConditionalGeneration`。该环境安装的 0.7.0 wheel 缺少对应 converter
（AReaL 固定依赖 0.8.1），因此需要把 `AWEX_SOURCE_ROOT` 指向包含
`awex/models/qwen3_5.py` 的共享 AWEX 仓库。launcher 会把该路径传给每个 worker，
并在分配 GPU 之前检查注册情况。实验使用的是 revision
`4e19c60fb2edf6df46aa4f2cecab9fd29a7e1b8d` 的 checkout。

`NODES=1` 使用 actor d2t4e8；`NODES=2` 使用 actor d4t4e16 加 Ray 调度器。rollout
分别使用 2 个或 4 个 TP4/EP4 引擎。设置 `BACKEND=baseline` 或
`BACKEND=weftmoe`，后者还需要上面两个 WeftMoE 路径。rollout 的调度规格会继承
actor 的接入变量。

两机实验前先启动 Ray head 和 worker：把 `RAY_ADDRESS` 设为 head 地址，并为 RDMA
网络设置 `NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`，必要时设置 `NCCL_SOCKET_IFNAME`。
baseline 与 WeftMoE 的两机实验使用相同 RDMA 设置。当前 native 多机反向要求每个
节点 8 张卡、节点内 EP rank 连续，并且需要 8 条 rail。

launcher 保持 `total_train_steps=null` 和完整的数据集/epoch horizon，由
`AREAL_STOP_AFTER_STEPS=20` 独立于优化器初始化与 LR 日程限制训练循环。完成指
20 次 update（global step 0 到 19），而不是把训练 horizon 设为 20。rollout 队列
设为 128，因为原始示例的队列 32 在消费 batch 也是 32 时会停滞。数据集过滤使用
16 个 CPU 进程，并保持原始顺序与内容。

baseline 的选择性重算为 `[core_attn,moe_act,layernorm,shared_experts,mlp]`。
WeftMoE 使用相同的周边模型重算策略，其 routed expert 反向保留 X/AB 并按仓库
benchmark 配置重算激活。该环境安装的 MCore 没有为 Gated DeltaNet 实现
`core_attn` 重算，因此 AReaL 对两个后端都在这类分支上施加 attention-only
checkpoint，把 routed expert 的 X/AB 放在 checkpoint 之外。没有这个处理时，
65,536 token 的 microbatch 会超出训练可用显存。CPU-staged 优化器步骤会在反向
之后、梯度范数集合通信之前同步并释放 PyTorch 未使用的 allocator cache；否则
NCCL 的惰性 communicator 分配会失败，即使被释放的激活显存仍然留在 PyTorch
cache 中。

## 验证

设置 `WEFTMOE_TENSOR_EXTENSION` 后用 `torchrun` 运行 GPU 测试：

- `tests/torchrun_tensor_runtime.py`：前向、输入/router/权重梯度、多层多 bucket
  CUDA Graph，以及与 eager prefill 交替执行。
- `tests/torchrun_megatron_adapter.py`：真实 Megatron MoE 模块、router 梯度与
  适配层 offload。设置 `TEST_DDP=1 TEST_TP_SIZE=4` 可检查两个 microbatch 的 DDP
  `main_grad` 累积、优化器更新，以及跨两个 step 的 runtime 重建。
- `tests/torchrun_sglang_adapter.py`：真实 SGLang MoE 数值一致性、图回放与 runtime
  释放/重建。
- `tests/torchrun_gdn_recompute.py`：开启 attention checkpoint 后真实 TP Gated
  DeltaNet 的输出与梯度一致性。
- `tests/sglang_weft_offload_server.py` 与 `tests/check_sglang_offload.py`：整模型
  AWEX 图/KV 驻留循环与 greedy token 一致性。用常规 SGLang 参数启动 server，在同
  一主机上用 server URL 与 `--cycles 20` 运行检查脚本即可记录多次切换的显存。
  在完整 Qwen3.5 模型、默认 52 个捕获 bucket 下，20 轮循环保持了 greedy token id
  一致。预热后 GPU 0 恢复态稳定在 65,910 MiB，释放图/KV 后为 23,800 MiB；该测试
  保留模型权重。

Qwen 规模的 routing 需在 tensor 测试上设置 `TEST_EXPERTS=256 TEST_TOP_K=8`。
若要复现 actor 的卡隔离，把 `CUDA_VISIBLE_DEVICES` 设为物理 GPU 列表，并在
`torchrun` 命令中把 `tests/run_with_isolated_gpu.py` 插在测试路径之前，使每个子
进程只看到一张卡且 ordinal 为 0。`TEST_TOKENS=16384` 还会覆盖长训练/eager-prefill
路径；decode 图检查最多使用 128 个本地 row，对应 EP4 server 512 row 的捕获
bucket。该长 token 测试在 EP8 上通过了前向与全部梯度，随后是两个 EP4 推理组的
图回放与 offload。多机 tensor 测试在每个节点上运行 8 个进程。完整 20 step 的
实验结果另行记录；算子测试本身不能证明端到端训练收敛性或性能。

对 AReaL 两机拓扑需设置 `TEST_INFERENCE_EP_SIZE=4`：训练使用 EP16，而推理图使用
4 个独立的 EP4 组。跨节点 EP16 推理图复用目前会以
`GIN reuse wait: invalid argument` 失败，本适配层已测试的推理配置不支持这种用法。

## 性能说明

接入后 SGLang 解码路径明显慢于 baseline，原因是设备端 kernel 的固定开销，与
tensor 申请、buffer/GIN context 重复创建无关。单层 MoE、EP4、CUDA Graph 回放
下，baseline 为 0.035-0.216 ms，WeftMoE 为 0.458-0.693 ms，且几乎不随 token 数
变化；其中 `single_notify_kernel` 约占 0.42 ms，`unified_forward_kernel` 约占
0.35 ms。40 层叠加即每个 decode step 多出约 18 ms。详细测量、已排除项与优化
方向见 `results/areal_qwen35_20260918/sglang_moe_perf_findings.md`。
