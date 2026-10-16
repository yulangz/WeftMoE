# Qwen3.5 接入 AReaL

## 2026-10-16：当前 PF/DT 重新接入，端到端对照进行中

沿用 9 月 30 日的 Qwen3.5-35B-A3B-Base / DAPO-Math-17k 配方：每组
独立单机 8×L20X、1620 MHz，rollout 为两个 DP4/EP4、attention TP1
实例；actor 为 attention TP4/DP2、EP8。seed1、prompt batch32、每题
16 samples、生成上限20480、context32768、training microbatch32768，
完整 epoch/LR horizon 不变，20 次 update 后停止。训练重算保持
`[core_attn,moe_act,layernorm,shared_experts,mlp]`。本轮 baseline 在
10.10.19.233，WeftMoE 在 10.10.16.219，两组都是单机任务，不是 EP16。

与旧对照不同，本轮实验组**同时替换 rollout 和 actor**：SGLang decode
使用 DT，prefill 使用当前 PF；Megatron routed FWD/BWD 使用当前 PF，
保留框架的 router、shared expert、优化器与 DDP。通过
`benchmarks/run_areal_sglang.py --backend weftmoe --actor-weftmoe --dp-attention`
启用；不加 `--actor-weftmoe` 仍兼容之前只替换 rollout 的行为。
没有修改 AReaL 调度器或其已有 dirty worktree。

接入修正：

- 同一 native 库保留 tuned PF 与稳定 DT。`TensorRuntime(decode_only=True)`
  显式选择独立的单节点 decode 池，窗口覆盖服务端 decode 容量；DT 的
  N128/K64 权重 TMA descriptor 独立于 PF N256 配置。不改 DT/PF GEMM
  数学或通信实现，也不再用 PF 编译 profile 间接决定能否进入 DT。
- SGLang 按 collective forward phase / decode graph 选择 DT，不按本 rank
  tokens 判断；取消 adapter 的 64-token 分界。PF 大池 B2048/D2，DT
  单窗口按 decode 容量分配，二者均跨层复用，卸载时一起释放。
- eager prefill 传入 layer ID，复用驻留期内 immutable packed weights；
  AWEX 更新前 suspend 销毁图/缓存，恢复后重新准备。跨执行池边界排空
  上一 eager generation，避免将待完成状态归给错误的池。

构建产物：`build/pf_main_timeline/qwen_areal_pf_dt_20261016`，基于
`configs/pf_qwen35_compact_20261013`；完整 build manifest 与编译日志留档。
正式启动脚本、日志、5 秒时钟/显存采样在 `build/rl_pf_dt_20261016`，
配置和训练产物在 `results/areal_pf_dt_20261016`，不覆盖旧记录。

启动前复用现有 integration tests，未增加细碎单测：

- SGLang EP4、E256/top8、T257：PF eager 对齐、强制确认 DT kernel、
  变化输入/路由的图回放、权重变化后的两轮 residency 均通过。
  原测试对 eager PF 与 graph DT 的逐位相等假设不再成立（最大差
  4.76837e-7），保留原跨实现 BF16 容差，DT 自身重复 replay 仍逐位一致。
  首次失败日志 `validation/sglang_ep4_ep4_node0.log` 保留，修正后为
  `validation/sglang_ep4_r2_ep4_node0.log`。
- Megatron TP4/EP8、E256/top8、T128、两层：FWD、dX、router/expert
  梯度、两次 optimizer 更新、每步两次 DDP microbatch 累积、两轮
  offload/rebuild 全部通过；释放约 0.74–1.04 GB/rank，实际驱动释放量
  不小于 native pool 报告值。日志 `validation/megatron_ep8_ep8_node0.log`。

以上是接入验收，不是端到端加速结论。两组 RL 已启动，性能待完整 update
后分别报告 rollout 时间/有效生成 token 吞吐、训练时间及 update 间隔。
W&B 与历史启动器一致暂为 disabled，仅本地日志，不声称已同步线上。

首次整模型尝试在首批生成中停住，未完成任何 update，不纳入性能：
DP 组同时出现 decode 与 prefill，当前 SGLang 的 `global_forward_mode`
只在部分调度模式赋值，此时为 None；退回本 rank 的模式导致通信池
不一致。采样显示 DT graph 回放分别停在约 3.7K/5.0K，py-spy 确认同组
一张卡在 `forward_decode`、另一张在 `forward_extend`，都等在 native
入口。现改用 SGLang 已有全组 MAX 汇总的 `is_extend_in_batch`：混合
批次全部走 PF，纯 decode（含 idle rank）统一 DT，无新增 collective。
原始日志、时钟与 `mixed_phase_rank{0,1}_stack.json` 保留，修正后的
第一次修正后重试 `-r1` 在服务启动期遇到 SGLang 派生握手端口占用，
同样没有完成 update，不纳入性能。未修改 AReaL 端口分配器，也未终止
未知端口占用者；清理本次失败任务、确认八张卡显存释放和 CUDA 检查
通过后，以新自动端口启动 `-r2`。baseline 原配置继续运行。

### 首轮端到端验收（中间结果，不作为稳态结论）

`-r2` 完成首个 update、AWEX 权重同步和下一轮生成，无 OOM 或 kernel
错误。首轮 LR=0，仍需后续非零 LR update 验证更新后的持续运行。

|首个完整 update|baseline|当前 DT/PF|
|---|---:|---:|
|rollout，s|725.41|673.20|
|train_step，s|239.11|205.98|
|update_weights，s|16.657|27.415|
|有效训练 token|5,313,900|5,603,300|
|有效 token / rollout 秒|7,325|8,323|
|有效 token / train_step 秒|22,224|27,203|

有效 token 来自 `ppo_actor/update/n_valid_tokens`（loss mask），不是服务端
所有生成 token；训练吞吐包含整个 actor step，不是单层 PF FWD/BWD。
该轮实验组 token 多约 5.4%，rollout/训练归一化吞吐高约 13.6%/22.4%；
但权重更新及恢复多约 10.76 s，不能省略此开销。首轮含编译和首次分配，
需单列，后续按相同 update 编号排除首轮配对统计。

## 常驻通信缓存与共卡释放复测（2026-10-02）

`ceec6a0` 将 PF Backward 的 peer payload 改为容量复用：一个 runtime
驻留期内按 `max_tokens` 与接收行数 hint 首次 lazy 分配，串行层和变长
microbatch 共用；实际行数变化不再反复分配/IPC open/close。超出 hint 的
热点仍会安全增长，不丢 token。FWD ring、GIN 注册窗口原本已常驻；训练
saved X/GU 仍跟随各 autograd tape，不放进长期通信缓存。

兼容现有 AReaL 生命周期，不新增进程级 cache：

- Megatron 的 weight-residency offload 先调用 adapter `suspend()`；无 live
  tape 后关闭 runtime，并清 packed weights、清理 Torch cache。恢复训练时
  `prepare()` 重建 runtime，Backward peer payload 在首次需要时创建。
- SGLang/AWEX release hook 先销毁外层 CUDA Graph，再关闭 DT/PF runtime
  及 staging。恢复全部 offload tags 后重新准备和捕获图。
- `TensorRuntime.close()` 现在是明确的两阶段 IPC 退休：同步并在 EP 组
  barrier；清内部图、关闭所有本地 IPC imports；第二次 EP barrier 后才释放
  exporters/GIN/NCCL。第二个 barrier 只在退出驻留期时发生，不进入每步算子。
  native 库与 Python adapter 需要同时重建。

本轮真实框架检查：Megatron TP4+EP8、TP4+EP16 均通过两层 MoE、DDP
microbatch 梯度累积、两次 optimizer 更新及两轮 suspend/resume；驱动显存
检查在测量前先清除旧 Torch cache，要求关闭后实际释放至少 native pool
报告的字节数。另有 EP16 训练 runtime 释放后切换为四个节点内 EP4 inference
组的 graph/offload 回归通过。

完整 Qwen3.5-35B-A3B-Base 的 SGLang DP4+EP4 单机服务，context 32768、
mem-fraction-static 0.4、cuda-graph-max-bs 512、1620 MHz，完成三轮
`release_memory_occupation` / `resume_memory_occupation`，每轮重新生成的
greedy 32-token IDs 与卸载前完全一致。仅 release KV/graph，权重保持驻留；
卸载后各 rank 的驱动可用显存约 86.9–87.9 GB，恢复后约 63.5–64.2 GB，
包含 native workspace、图及框架资源，不能把全部差额算作 BWD payload。
这不是新一轮完整 RL 性能对照，也不代替 AWEX 权重更新验收。

产物与日志：`build/pf_resident_workspace/` 的
`megatron_release_ep8.log`、`megatron_ep16_node{0,1}.log`、
`lifecycle_ep16_node{0,1}.log`、`sglang_server.log`、`sglang_offload.log`。
动态算子性能、内存取舍及配对条件见
[训练记录开头](static_forward_v4_training_record.md)。下面保留历史 RL 对照记录。

## DP4+EP4 对照（2026-09-30，运行中）

在 [decode 记录第 9 节](static_forward_v4_decode_record.md) 的单层 DP-local
对照之后，将实际 RL 的两组 rollout 都改为 DP4+EP4。仍是每组独立单机
8 GPU、两个 4 卡 SGLang 实例；不是单个 DP8/EP8，也不是双机 EP8。

|项目|baseline|WeftMoE|
|---|---|---|
|物理节点|10.10.16.219|10.10.19.233|
|每个实例|DP4 / EP4 / attention TP1|DP4 / EP4 / attention TP1|
|routed MoE|SGLang 默认 Triton + gather/reduce-scatter|HHE 专属 decode + 通用 V4 prefill|
|actor|原版 Megatron|原版 Megatron|
|目标 SM 时钟|1620 MHz|1620 MHz|

新增启动参数 `benchmarks/run_areal_sglang.py --dp-attention`，只覆盖
`sglang.enable_dp_attention=true`、`sglang.dp_size=4`；原配方的
`sglang.ep_size=4` 和 `rollout.backend=sglang:d2t4p1` 保持不变。
AReaL 的外层 `d2` 表示两个实例，不是实例内的 attention DP 数。
解析后的两组 YAML 已逐项比较：除 trial/W&B 名称和 rollout 的 WeftMoE
环境外一致，actor 的 `AREAL_USE_WEFTMOE` 均为 0。沿用相同模型、数据、
seed=1、全局 prompt batch 32、每题 16 samples、20K 生成上限、32K context、
32,768-token training microbatch，训练 horizon 不变，在第 20 个 update 后停止。

### 启动前的权重布局修复

不能只打开 DP 开关：检查发现两处 TP 语义混淆，已作最小修复。

1. AReaL `AwexColocateReader` 原来把实例总 TP=4 写入
   `infer_conf.infer_atten_tp_size`。改用已有 model context 的实际 attention
   TP=1，与 AWEX 原生 reader 一致；否则 writer 会错误重排 gated QKV 和
   GDN 权重。此修复两组共用，不改变实例数、EP、训练侧分片或通信协议。
2. Weft 的 DP-local 路径将 shared expert 建为 TP1 完整副本，但 AWEX 的
   Qwen3.5 shared-expert 规则仍按 TP4 描述。新增 `weftmoe/awex.py`，仅在
   AWEX 场景且实际重建 shared MLP 后安装元数据适配：SGLang + DP 使用
   `NO_SHARDING`；Megatron、非 DP 和 baseline 保留原规则。

最小 CPU 检查覆盖 shared gate/up/down、重复安装、非 DP/Megatron 不变，
以及 reader 的 TP4→attention TP4、DP4→attention TP1 两种 context，全部通过。
原生 CUDA library 未改，仍为 `local_decode_hh_twobarrier/libweftmoe.so`，
SHA256 `962daf6fa6486f7afc930934ce636d2a7053e6227742e09356101ab6f3adf421`。
这不等于完成首次训练后的 AWEX 权重更新验证，后者需观察实际 RL。

AReaL 修复提交为 `28220009`（`fix/awex-dp-attention-metadata`）。为保留
该仓库原有未提交改动，在独立 worktree 提交；运行目录已应用同一补丁，
未切换其运行分支。全量 pre-commit 在隔离目录通过，hook 生成的 lockfile
改动未带入运行目录或本次提交。WeftMoE 的本轮改动不包含训练/算子优化。

### 实验与日志

当前 trial：

- baseline：`qwen35-sglang-baseline-dp4ep4-hhe-mb32-8gpu-20260930`。
- WeftMoE：`qwen35-sglang-weftmoe-dp4ep4-hhe-mb32-8gpu-20260930-r1`。

两组数据位于 `results/areal_sglang_dp4_ep4_20260930`；启动脚本、入口日志、
5 秒 GPU 时钟/显存采样及 HTTP 指标位于 `build/rl_dp4_ep4_20260930/`。
HTTP 采样保留全部 DP rank 的 labels，不能沿用 TP 基准只取 `tp_rank=0`
的过滤方式。W&B 沿用 console-table 旁路同步，仅发布完整 update：

- [baseline run](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/qwen35-sglang-dp4ep4-hhe-mb32-20260930-baseline)
- [WeftMoE run](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/qwen35-sglang-dp4ep4-hhe-mb32-20260930-weftmoe)

首个配置解析尝试缺少 Hydra 的 `++`，已修正，日志留在
`config_parse_attempt/`。Weft 首次正式启动的附加 metrics 端口冲突，尚未
生成训练数据；失败日志/时钟记录留在 `port_collision_attempt/`，原 trial
结果目录保留。仅清理该作业残留进程，确认 8 卡显存归零、时钟 1620 后，
于 04:36 UTC 以 `-r1` trial 原配置重启；没有重置 GPU/驱动或中断 baseline。
W&B 的 Weft run 在失败尝试中没有 update 数据，继续用于同配置的有效重跑。

04:42 UTC 启动检查：两组的四个 SGLang 实例 `/health` 均返回 200，在线
`tp_size/dp_size/ep_size=4/4/4`、`enable_dp_attention=true`、
`moe_a2a_backend=none`。每个实例都有 DP rank 0–3 的活跃请求，生成 token
计数已增长；两组 MetaServer 中的 `infer_atten_tp_size=1`、
`infer_world_size=8`、`num_infer_engines=2` 均读回核实。当前有效运行日志
没有 traceback；baseline/Weft 的 1256/560 个 GPU 时钟样本均为 1620 MHz。
尚无完整 update，尤其未验证首次训练后的 AWEX 更新及恢复生成，故不报告
端到端性能优劣。后台作业、GPU/HTTP 采样和 W&B 同步继续运行。

## 当前 V4 接入检查（2026-09-29）

本轮基线为 HF（`7569ae0`），后续 HG 的 EP 通用化验收见
[decode 记录第 6 节](static_forward_v4_decode_record.md)。下面的旧 RL 性能数字来自历史实现，不能用作
当前 V4 的端到端结论。HG 的新 RL 对照确认 rollout 回退；HH 优化后已重跑
3 次 update，回退明显缩小但未追平 baseline，详见下方相同步数对照。
本轮尚未完成 20 updates，不能据此给出训练收敛结论。

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

### 本轮 RL 实测进度

2026-09-29 首轮沿用训练 microbatch 65,536 tokens。两组均为单机 8 GPU、
两个 TP4/EP4 rollout 引擎，分别运行在同型号节点；不是一次双机 EP8 实验。
模型、数据、全局 batch 32、每题 16 samples、20K 生成上限和 actor 实现相同。
baseline 完成了第 1 次 update（rollout 687.62 s、train 162.50 s、权重更新
16.981 s），但 WeftMoE 在第 1 次训练的 `finalize_model_grads` →
`reduce_scatter_tensor_coalesced` 处失败：NCCL 懒分配 536,870,912 bytes 时 OOM。
这是梯度同步阶段，早于现有 `optimizer_step()` 的清缓存，并非已定位为推理
kernel 错误。卸载推理、恢复 actor 后总显存为 baseline 21.89 GiB、WeftMoE
21.38 GiB，现有记录不支持推理双池未释放的判断。

这轮 **不构成完成的 RL 性能对照**。首轮 SGLang 实际 decode 计数均约 625 万，
WeftMoE 从 proxy ready 到首次暂停采样约 821 s，慢于 baseline 的约 688 s；
不能用单算子提速宣称 RL 已加速，也不能用未完成的更新推断稳态或训练收敛。
旧作业已停止，日志、5 秒 GPU 采样及 HTTP 指标保存在
`build/local_decode_hg/rl_{baseline,weftmoe}*`。时钟目标为 1620 MHz，但存在少量
降频样本：baseline 2/2376 GPU-samples，WeftMoE 6/2344，原始记录未过滤。

后续重跑仅将两组 `actor.mb_spec.max_tokens_per_mb` 统一降至 32,768；
不修改 Megatron 实现、全局 batch、采样长度或优化器，不能与 64K 结果混算。
两节点交换实验角色，重新锁定 1620 MHz，仍各自为单机 8 GPU、20 updates。
原始数据目录为 `results/areal_sglang_v4_20260929_mb32`，入口日志为
`build/local_decode_hg/mb32/rl_{baseline,weftmoe}.log`。OOM 后本机进程清理曾触发短暂 NVRM 超时，随后
自行恢复；确认显存归零、全部 8 卡 CUDA 分配/同步通过后才重新启动，未重置驱动。

32K 两组均已完成第 1 次 update，并恢复下一轮生成；WeftMoE 的 AWEX v1
权重传输、推理恢复及图回放通过。首步学习率为 0，这不代替非零学习率的后续
更新或稳态验证。以下仅是首步记录，不是 20 updates 的最终对照：

|指标|baseline|WeftMoE|
|---|---:|---:|
|rollout（s）|675.38|806.99|
|train（s）|224.02|209.52|
|权重更新（s）|19.487|21.186|
|平均序列长度|10,130|10,275|
|task reward|0.85938|0.84180|

首轮 rollout 慢约 19.5%，当前没有端到端加速的证据。actor 实现相同，表中的
训练耗时也不能解读为 WeftMoE 训练收益。

截至 12:11 UTC，双方各完成 3 次 update；去掉首步后的两步中位数如下：

|指标|baseline|HG WeftMoE|
|---|---:|---:|
|rollout（s）|527.72|627.605（+18.93%）|
|train（s）|182.935|169.745|
|权重更新（s）|14.754|12.7355|
|平均序列长度|9432.35|9338.25|
|update 完成间隔（s）|732.20|834.4575（+13.96%）|

这足以触发性能诊断，但不是最终 20-step 结果或收敛结论。WeftMoE 的序列略短，
不能以生成长度解释慢约 19%。已验证非零学习率下的 AWEX 更新和图重建。
HTTP 采样中的活跃 batch 中位数为 baseline 40、WeftMoE 42，没有超过 256 的样本，
不支持“大 batch 进入通用 V4”是主要原因。该时点的 5 秒采样中，baseline
5064 个 GPU 样本均为 1620 MHz；WeftMoE 5136 个样本有 2 个低于目标频率，未过滤。
12:12 UTC 停止 WeftMoE 组（保留 3 个完整 update，丢弃第 4 个未完成 update），
释放该节点做同机诊断；baseline 继续。原始日志、JSON 和 W&B 历史不覆盖。

最小复现入口为 `benchmarks/torchrun_sglang_decode.py`：真实 SGLang BF16
E256/K8/H2048/I512、复制输入 TP4/EP4、包含模型外层 TP all-reduce、随机 top-k，
同机交替 CUDA Graph 回放取 rank-max。HG 在全局 T32/T64 的单层延迟分别是
160.969/192.961 µs，baseline 为 105.789/150.416 µs。trace 确认存在适配层
多节点输入整理/回填开销，且 native decode 本身也偏慢。优化必须分别消融，
不能把单层结果直接当作端到端收益。

用户随后提供 W&B 凭据；为避免重启正在运行的对照，用
`benchmarks/publish_areal_logs.py` 将完整 console metric table 旁路同步至
[W&B 项目](http://8.150.1.98:8080/sct-test/areal-qwen35)，group 为
`qwen35-sglang-hg-mb32-20260929`。baseline/WeftMoE 的首个点已从服务端读回
核验，后续每个完整 update 自动同步，结束于第 20 次更新。训练进程本身仍为
`wandb.mode=disabled`；上传指标保留日志打印精度，并在 run config 中标明
`metric_source=completed_console_tables`，不是原进程的全精度原生 W&B history。
凭据通过无回显输入只提供给上传进程，没有写入源码或实验配置。
HG WeftMoE 停止后用 `--finished weftmoe=<reason>` 上传完 3 个完整 update 并
结束该 W&B run，summary 保留停止原因；不是训练异常，也不上传未完成 step。
baseline 的旁路同步继续，优化版使用新 run，不覆盖 HG 历史。

### HH 优化版复测（2026-09-29）

HH 的逐项消融和负结果见 [decode 记录第 7 节](static_forward_v4_decode_record.md)。
采用 commit `5d4c374`：融合 graph staging/输出回填、避免 padding 激活无关
专家，稀疏 decode 用 M16 MMA/TMA，较密输入用共享调度的 M64 WGMMA，并
按专家任务量限制静态 grid。prefill、training 和多机 kernel 没有改动。
E256/EP4 的全局 T32/T64 单层图从 HG 的 160.969/192.961 降至
125.245/150.888 µs，但 T1–48 仍慢于 SGLang baseline，不把这些算子收益
直接写成端到端收益。

14:32 UTC 在 `10.10.19.233` 启动新 trial
`qwen35-sglang-weftmoe-hhd-mb32-8gpu-20260929`。冻结 library 为
`build/local_decode_hh_release/libweftmoe.so`，SHA256
`b7ea21bdf989b30b5cda1cbe7d813ad2de70bc3cae4ec087375511c3a7c5d7be`。
解析后的完整 YAML 与原 baseline 对照，差异仅为 trial/目录/W&B 名称以及
rollout 的 WeftMoE 环境；actor 后端、采样、数据、seed 和 32K microbatch 一致。
实际后端字段仍为 actor `megatron:(attn:d2p1t4c1|ffn:d1e8)`、rollout
`sglang:d2t4p1`，两个 TP4/EP4 引擎。YAML 中未使用的旧 `allocation_mode`
字符串在两组都保留 TP2，不能据此误判实际并行配置。

日志为 `build/local_decode_hh/hhd-mb32/rl_weftmoe.log`，独立
[W&B run](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/qwen35-sglang-hhd-mb32-20260929-weftmoe)
已在线，仍由同一个 console-table 旁路上传器记录完整 update。
此轮至少观察 3 个完整 update，以同样的前 3 步对照、剔除第 1 步 warm-up，
并同时报告接受的生成 token 吞吐。
按日志的 `ppo_actor/update/n_valid_tokens / timeperf/rollout` 聚合，第 2–3 步
baseline 为 9006.6 tokens/s，旧 HG 为 7496.0 tokens/s（−16.8%）。这是接受的
训练样本的生成 token 吞吐，不是包括预取/丢弃请求的服务端总吞吐；序列长度
和实际采样结果并非严格固定，不能只比较 rollout 秒数。

15:04 UTC 已完成 2 个 update。首步 rollout 739.40 s、接受的生成 token
5,367,800，相对 baseline 首步 675.38 s / 5,106,800，归一化吞吐仍慢约 4%。
第 2 步为 496.66 s / 4,556,900，baseline 同步为 498.25 s / 4,658,900：
rollout 时间已接近，接受的生成 token 吞吐仍慢 1.9%，尚不作为稳态最终结论。
第 2 次非零学习率训练、AWEX 权重更新及恢复生成均完成，没有训练异常。

14:55 UTC baseline 已主动停止，保留 16 个完整 update（丢弃未完成的
第 17 步），W&B run 正常结束并记录停止原因；释放本机用于独立 decode
消融，没有干扰另一节点的 HHD RL。停止后短暂 NVRM 回收超时已自行恢复，
确认 8 卡空闲、全部 1620 MHz 后才运行测试，没有重置 GPU 或驱动。

15:17:45 UTC，HHD 已完成第 3 次 update；第 3 步 rollout 为 595.67 s，
接受的生成 token 为 4,698,300。使用双方**同样的第 2–3 步**（非 baseline
全部 16 步）比较，原始汇总为 `build/local_decode_hh/hhd-mb32/rl_first3_comparison.json`：

|指标|baseline|原 HG|HHD（`5d4c374`）|
|---|---:|---:|---:|
|rollout 中位数（s）|527.720|627.605|546.165|
|接受的生成 token/s|9006.6|7496.0|8472.9|
|update 完成间隔中位数（s）|732.200|834.458|740.386|

HHD 相对 HG 的 rollout 时间减少 13.0%，接受的生成吞吐提高 13.0%；
相对 baseline 仍慢 3.5%（时间）/5.9%（吞吐），不能写成已经追平。
两步采样长度、预取和尾部请求都不同，这不是固定输出 token 集的纯 decode
计时；单层对照中 T1–48 的剩余差距也仍存在。三轮均完成训练/权重更新，
第 4 步继续，W&B 保留完整记录。时钟固定目标仍为 1620 MHz；截至 15:20
的 4,600 个 GPU 样本中有 4 个低于目标（1605×1、1590×2、1575×1），
原始采样未过滤。独立算子正式计时的全部采样均为 1620 MHz。

HHE 的末尾 barrier 消融不包含在上表 HHD 的 RL 收益中，见 decode 记录。

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
