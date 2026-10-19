# Qwen3.5 接入 AReaL

## 2026-10-19：仅 Megatron 开启 DeepEP 的 8K baseline

按用户要求，暂不修补 SGLang BF16 + DeepEP 计算路径，改为仅训练端
启用 `moe_token_dispatcher_type=flex / moe_enable_deepep=true`。
推理端显式 `moe_a2a_backend=none`，恢复原生通信/GEMM/CUDA Graph；
删除前次 rollout 的 DeepEP BF16、LL capacity、NVSHMEM QP-depth 环境设置。
AWEX 的 DeepEP shared-expert 元数据分支因此不启用。

CPU 配置比对确认：SGLang 参数及 rollout 环境与原 8K baseline 一致
（仅新增显式 backend=none 和未设置的 mode）；Megatron 配置只改变上述
两个开关。模型、数据、采样、seed、optimizer、重算及实际并行拓扑不变。
没有生产源码改动，复用 AReaL commit `d5395ffa` 的 dispatcher 透传。

在 `10.10.19.233` 的八卡空闲后启动；trainer 时间 **16:56:03 UTC**，
supervisor `316696`、trainer `316766`。使用新的 run 名称和输出目录，
保留前两次双侧 DeepEP 失败记录；从官方 Base 权重及新 optimizer 开始，
目标 **100 updates**，不续跑旧 baseline 的第 79 步。
仍为 8K 生成预算、32 prompts × 16 samples、BF16、1620 MHz，
actor attention TP4/DP2 + EP8/ETP1、两组 rollout DP4/EP4。
W&B 配置原生 online，checkpoint 关闭、recover disabled；本机 WeftMoE
实验未停止或重启。

启动核验：八个 actor rank 各 40 层均实际构建
`MoEFlexTokenDispatcher / _DeepepManager`。两组 SGLang 于 17:00 UTC
就绪，17:00:36 proxy 初始化完成；服务接口核验 `moe_a2a_backend=none`、
`moe_runner_backend=auto`、BF16、CUDA Graph 开启、DP4/EP4。
17:01 UTC 两组均有实际生成请求，已进入首轮 rollout；W&B 服务端确认
run 为 running。尚未完成首个 training update，也尚未验证更新后的 AWEX
权重同步；没有据此报告 reward 或性能改善。

配置、监督脚本和 manifest：`build/rl_baseline_mcore_deepep_8k_20261019/`；
训练输出：`results/areal_baseline_mcore_deepep_8k_20261019/`。
[仅训练端 DeepEP 的 W&B](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-baseline-mcore-deepep-dp4ep4-mb32-8k-100-20261019_train)。

## 2026-10-19：双侧 DeepEP 接入与 8K baseline 重启

**当前状态：未成功跑起。** 两次均在推理初始化时失败，完成 0 updates，
没有可用的 reward/性能结果，也尚未创建原生 W&B run。三处 AReaL 适配
已完成，但当前 SGLang 的 BF16 + DeepEP GEMM 路径尚不完整。

经用户确认，完成三处 AReaL 适配（commit `d5395ffa`），不修改 WeftMoE
kernel 或训练配方：

- Megatron-Bridge 透传 `moe_token_dispatcher_type=flex` 和
  `moe_enable_deepep=true`，不再强制 alltoall；保持原 permutation fusion。
  首次启动八个 actor rank 均实际构建 40 层
  `MoEFlexTokenDispatcher / _DeepepManager`，非仅配置开关。
- SGLangConfig 新增可选 `moe_a2a_backend` / `deepep_mode`，默认不添加 CLI
  参数；本组使用 `deepep / auto`，prefill normal、decode low-latency。
- AWEX 接收端对原生 SGLang DeepEP 的 Qwen3.5 shared expert 使用
  `NO_SHARDING` 元数据：训练端仍是 TP4 源分片，推理每 rank 接收完整
  shared MLP。routed experts、attention、shared-expert gate 不改变。

新增 CPU 测试覆盖配置透传与两种 shared-expert 布局，3 cases 通过；
连同已有 Qwen TP4 权重布局测试共 4 cases 通过。全量 pre-commit 在隔离
工作树通过，自动生成两种语言的 CLI 文档；依赖锁文件的环境差异不带入
生产工作树，没有安装或升级依赖。

首次启动在 CUDA Graph 捕获 batch=263 时超过 DeepEP 默认 LL 容量 128，
于 rollout 前退出，未产生 update。改为
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512`、
`NVSHMEM_QP_DEPTH=2048`，满足 `QP_DEPTH >= 2*(capacity+1)`；
保留 CUDA Graph、batch、并发配置不变。此容量限制针对 LL decode，
不限制 normal prefill 的完整 token 数。旧 attempt 日志和状态完整保留。

远端 `10.10.19.233` 第二次 trainer 启动于 **16:39:42 UTC**，
supervisor `312387`、trainer `312457`。目标 **100 updates**，
从官方 Base 权重和新 optimizer 开始。8K 生成上限、32 prompts × 16 samples、
seed=1、actor attention TP4/DP2 + EP8/ETP1、两组 rollout DP4/EP4
不变；八卡固定 1620 MHz，双方 `AREAL_USE_WEFTMOE=0`。
显式启用 `SGLANG_DEEPEP_BF16_DISPATCH=1`，避免 LL 默认 FP8 通信
改变比较口径。W&B 配置为原生 online，recover disabled，不保存 checkpoint。
本机 WeftMoE 8K 组未停止或重启。

配置、source 指纹、监督脚本及 attempt1 记录：
`build/rl_baseline_deepep_8k_20261019/`；
输出：`results/areal_baseline_deepep_8k_20261019/`。

第二次已越过通信容量检查，于 **16:43:50 UTC** 在
`DeepEPMoE.run_moe_core` 报 `forward_deepgemm_masked is deprecated`，
supervisor 于 **16:44:16 UTC** 记录 failed/0 updates。当前安装的
SGLang **0.5.10.post1** 中，未量化 BF16 未切入新的 DeepGemm runner；
旧 LL 分支是上述硬断言，normal 分支同样是
`forward_deepgemm_contiguous is deprecated`。简单切成 normal 不足以修复，
且该模式自动关闭 CUDA Graph。普通 BF16 Triton runner 仅有 standard
dispatch 布局接线，DeepGemm runner 当前实际调用 FP8 GEMM，不能靠改开关
得到同口径的 BF16 baseline。

未切换 FP8、未关闭 CUDA Graph、未修改第三方 GEMM 或升级依赖；下一步
需要另行确定 BF16 + DeepEP 的计算接线方案。这超出原定三项配置/元数据
适配。失败进程的残留 RPC worker 已定向停止，日志保留；本机 WeftMoE
任务不变。AWEX shared-expert 元数据仅完成 CPU 验证，尚未完成端到端验证。

## 2026-10-19：停止旧 8K baseline，准备双侧 DeepEP 对照

按用户要求停止 `10.10.19.233` 上未启用 DeepEP 的 8K baseline，保留日志
和 W&B。共完成 **79 updates**，最后一轮 reward=0.76172；第 80 轮期间
收到停止信号。监督状态已标记 `stopped_by_user`，原 session 的活跃子进程
已退出，八卡无 compute process。未删除模型、日志或历史实验数据。
本机 WeftMoE 8K 组不改动。

新 DeepEP 组尚未启动。接入检查发现：

- Megatron-Bridge 无条件覆盖 dispatcher 为 alltoall，需透传已有配置中的
  flex + moe_enable_deepep，且不可顺手改变 provider 的 permutation fusion。
- AReaL SGLangConfig 缺少 moe_a2a_backend/deepep_mode 配置入口。
- 原生 SGLang DeepEP 将 Qwen shared expert 改为 TP1，AWEX reader 的
  shared-expert 元数据也需对应 NO_SHARDING；不能通过开启 WeftMoE 间接适配。
- 同 BF16 比较需明确设置 SGLANG_DEEPEP_BF16_DISPATCH=1，避免默认 low-latency
  FP8 通信改变数值口径。

远端已确认 DeepEP 1.2.1+567632d、MCore 0.17.0、Bridge 0.4.0 可导入，无需
安装新依赖。依仓库规则，已请求用户确认配置结构的小适配，再启动新组。

## 2026-10-19：启动 8K WeftMoE 对照组

在本机 `10.10.16.219` 启动与另一台 8K baseline 对齐的 WeftMoE 组，
目标同为 **100 updates**，从原始 Base 模型与新 optimizer 开始。
trainer 启动于 **07:24:16 UTC**：supervisor `519421`、trainer `519538`。
**07:29:22 UTC** 两组 SGLang ready，**07:29:44 UTC** rollout proxy 初始化
完成，进入首轮 rollout；W&B 已开始 online 同步。尚未完成第一个 update，
另一台 baseline 不改动。

- 模型 `Qwen/Qwen3.5-35B-A3B-Base`，DAPO-Math-17k；生成上限 8192，
  总 context 32767，32 prompts × 16 samples，seed=1，采样、训练和评分配方
  与 8K baseline 相同；三个关键 AReaL 源文件指纹也与 baseline 启动时一致。
- 单机八卡固定 1620 MHz；actor attention TP4/DP2、EP8/ETP1；rollout
  两组 DP4/EP4、attention TP1。Megatron 使用 PF FWD/BWD，SGLang 使用
  PF prefill / DT decode，均固定为已验证的 `qwen_areal_barrier_fix_20261018`。
- 原生 online W&B，同 baseline 的 group，不启动独立上传进程；认证已核验。
  saver/recover 频率全部 null，recover disabled，无 checkpoint。
- 保留修复版 `check_for_nan_in_grad=true`，baseline 为 false；此外只变更
  WeftMoE 开关及必要 runtime 配置、run 标识和输出目录。不加逐层诊断同步。

配置、manifest、监督脚本和状态：`build/rl_weftmoe_8k_20261019/`；
训练输出：`results/areal_weftmoe_8k_20261019/`。
[WeftMoE 8K W&B](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-weftmoe-pfdt-dp4ep4-mb32-8k-100-20261019_train)。

## 2026-10-19：barrier 故障已独立复现

后续测试找到有效触发条件：同卡两个独立 CUDA context。最小程序仅做
barrier init/arrive/wait、覆盖为普通 shared 数据并读取，不含 GEMM/TMA。
旧协议三轮均捕获损坏，invalidate 对照三轮均未捕获；两个进程不共享内存。
普通数据被改成 `0xfffff800001ffffe`，按 BF16 解读含 NaN 和约 `-1.04e34`。

保留 PF 流水线的四组消融也确认：仅 drain 仍损坏，仅 invalidate 未损坏。
原生未修复 PF EP4 在第 218 轮、layer 11 的 dW1 捕获 36 个 NaN 和 12 个
超大有限值，均位于 M128 tile 第 96 行的旧 barrier 区域，发生在 optimizer
之前。修复版完成 500 轮、每 rank 8000 次 BWD，同轮共卡压力测试未见坏值。

代码、运行方法和原始结果见
[独立复现说明](../../Amazing-Hopper/mbarrier_arena_reuse/README.md)。
未 invalidate 的协议错误及其数据损坏已确认；context 切换时硬件内部的
保存/恢复细节仍属推断，不混同为已观测机制。本轮不修改生产 kernel，
保留修复 `9b38230`。测试已结束，另一台机器的 8K baseline 未受影响。

## 2026-10-19：停止 WeftMoE 长跑，提取 barrier 独立诊断

按用户要求停止 `10.10.16.219` 上的修复版 WeftMoE 长跑。最后完整完成
**101 updates**；06:20:14 UTC 已更新状态为 `stopped_by_user`，训练/推理
子进程退出、八卡释放。保留日志和 W&B，不生成 checkpoint。
`10.10.19.233` 上的 8K baseline 没有停止或修改。

独立 CUDA 程序、运行脚本、现场说明和本轮原始结果放在
`../Amazing-Hopper/mbarrier_arena_reuse/`。保留旧 barrier 与 dW 输出的
重叠布局、TMA/WGMMA、cluster multicast、阶段交接、STSM/TMA 写回，
drain/invalidate 可分别开关；剥离模型、AReaL、optimizer 和 RDMA。

**以下为早期负结果，已被上方的成功复现补充。** 单 CTA、全 SM cluster1/2、copy/GEMM、
不同 stage 尾部的 28 组测试各 10000 次，均未捕获坏值；双 CTA 的
racecheck/synccheck 也未报错。关闭修复仍通过，不能据此证明旧协议正确，
也不能把“旧 barrier 晚到写回”当作已验证机制。后续需要进一步保留原生
PF 的真实任务/时序条件，不能以人为添加的另一种竞态替代现场问题。

## 2026-10-19：8K baseline 长度消融

按用户要求，在 `10.10.19.233` 启动原版 Megatron + SGLang baseline，
从已核验的官方 `Qwen/Qwen3.5-35B-A3B-Base` 权重和新 optimizer 开始。
目标先跑 **100 updates**，用于观察 8K 预算下的起始 reward 和早期趋势，
不把外部实验的 0.6 → 0.8 曲线当作本配置必然收敛的证据。

- trainer 于 **04:29:54 UTC** 启动：supervisor `147934`，trainer `148011`。
  **04:35:01 UTC** 两组推理服务均 ready，04:35:04 proxy 初始化完成；
  W&B 服务端已查到新 run 为 running。此时尚无完整 update/reward。
- 基于旧 20K baseline 配置，除 trial/输出目录/W&B 名称外，只将
  `gconfig/eval_gconfig/actor.max_new_tokens` 从 20480 改为 **8192**。
  总 context 32767、32 prompts × 16 samples、temperature/top-p=1、seed=1、
  数据、学习率、batching 和重算配置不变。没有添加独立 step-0/验证集评估。
- 仍为单机 8 GPU，actor attention TP4/DP2、EP8/ETP1；rollout 为两组
  DP4/EP4、attention TP1。两端 `AREAL_USE_WEFTMOE=0`，时钟固定 1620 MHz。
- 使用现有共享 AReaL 代码，因此包含 10 月 18 日的专家 optimizer TP-group
  norm 修正；旧 20K baseline 没有这项修正。不是旧二进制的严格单变量复跑。
- 保留原有 `math_reward_fn` 评分逻辑，避免本轮同时更换长度和评分口径。
  `verify(ans, gold)` 与库的 `verify(gold, target)` 参数约定相反，
  对实际 reward 的影响尚未量化；不能据此断言旧 0.8 得分虚高。
- 原生 online W&B，私有 NETRC 认证，不开独立上传进程。
  saver/recover 的所有频率仍为 null，recover disabled，不保存 checkpoint。
- 旧 baseline 已自然退出、状态为 failed/284 updates，启动前八卡均空闲；
  本轮没有杀进程，也没有清理历史 NCCL shared-memory 文件。
  `10.10.16.219` 上的 WeftMoE 任务不变。

配置、监督脚本和 source 指纹：`build/rl_baseline_8k_20261019/`；
输出：`results/areal_baseline_8k_20261019/`。
[8K baseline W&B](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-baseline-dp4ep4-mb32-8k-100-20261019_train)。

## 2026-10-19：barrier 问题留档与 baseline 进度

### 待做的最小复现

继续保留 10 月 18 日的 BWD barrier 生命周期问题，后续独立复现，不把
12-update 诊断通过等同于已经捕获了首次损坏。修复 `9b38230` 同时加入
producer-tail drain 和旧 mbarrier invalidate，需要分开消融两者的作用。

补充的二进制证据：旧单机 BWD 没有 `SYNCS.CCTL.IV`；修复版在
`0x27000` 至 `0x27040`、间隔 8 B 的 9 个旧 barrier 地址上实际发出
该指令。dW 新 barrier 初始化在 `0x34000`，不会使旧对象自动失效。
这证明修复不是空的 C++ 标记，但仍不能证明具体的硬件内部写回机制。

后续最小复现按以下顺序进行：

- 去掉 AReaL、SGLang、Adam 和 RDMA，保留 SM90a 上 barrier arena
  到 dW 输出 tile 的复用，以及 swizzled STSM → TMA-store 数据流。
  首先测试单 CTA；如需保留跨 CTA arrivals，再用双 CTA cluster。
- 做 drain 开/关 × invalidate 开/关的四组对照。区分“异步事务未结束”
  与“事务结束后未 invalidate”两种错误，不用人为制造的另一种竞态
  冒充现场问题的复现。
- 检查整个输出 tile 的位模式及损坏位置，重点核对第 96 行的 36 个
  BF16；不能只检查 NaN，因为现场还有超大有限值进入 Adam 后溢出。
- 当前没有可重复触发原故障的独立脚本。等待用户确认释放 baseline
  节点后再实现和测试，本轮没有停止训练或启动 GPU 复现实验。

### baseline：284 / 1500，已经失败，不是正常长跑

03:30 UTC 检查 `10.10.19.233`：最后完成第 **284** 次 update（18.9%），
完成时间为 **00:14:05 UTC**。随后第 285 次更新在 **00:21:29 UTC**
报告 `ppo_update` 失败：`validate_transport_padding` 的 CPU/Gloo
all-reduce 出现 `Connection closed by peer`。02:21 起又出现 NCCL
collective timeout，属于更晚的故障，不能据此定位最初原因。

supervisor 的状态文件仍是启动时留下的 `running`，supervisor/trainer
进程也未退出；实际已经超过 3 小时没有完成更新。只读 GPU 查询仍有
5 个 actor 进程各占约 49–52 GiB，不能把这台机器当作已经释放。
同节点内核日志另有 memory-cgroup OOM-kill（`dmesg -T` 显示
00:22:48，host PID 71649）；尚未核对 host/container PID 和时钟对应，
因此仅记录为排查线索，不把它或 WeftMoE barrier bug 当作已确认原因。
baseline 未使用 WeftMoE native。

Reward 使用已完成 update 的 `ppo_actor/task_reward/avg`，即原始任务
得分，不使用归一化后接近零的 `final_reward/avg`。本配方得分为 0/1，
每步 512 个样本；下表为各区间的平均值，以百分比表示：

|完成的 update|平均原始 reward|
|---|---:|
|1–50|81.63%|
|51–100|81.85%|
|101–150|81.93%|
|151–200|84.19%|
|201–250|82.06%|
|251–284（34 步）|80.10%|

等长窗口对比：最初 50 步 **81.63%**，最近 50 步（235–284）
**80.47%**，下降 **1.16 个百分点**；最后单步为 81.25%。当前没有
持续上涨的证据。训练 batch 在变化，且本次没有固定验证集评估，不能
据此认定泛化能力下降。以上由 `benchmarks/publish_areal_logs.py`
的 `read_updates()` 只读解析完整日志表格获得，没有启动上传进程。

原始日志保留在远端 `build/rl_pf_dt_1500_20261016/baseline.log`；
[原生 W&B run](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-baseline-pfdt-dp4ep4-mb32-1500-20261016_train)。
本轮未发送停止信号、未清理残留进程、未重启或修改 baseline。

## 2026-10-18：修复版重新启动 1500-update 长跑

12-update 诊断已通过：07:33:29 UTC 完成第 12 次 update，07:36:28
supervisor 确认 exit code 0；12 次 native 梯度和更新后权重检查均通过，
无非有限 logprob/梯度异常，W&B 已同步，未产生 checkpoint。

按用户要求，仅在本机 10.10.16.219 重新启动 WeftMoE 这一组长跑。
另一节点的 baseline 不停止、不重启，也不启动 EP16。诊断未保存
checkpoint，因此本次重新加载初始模型和新 optimizer，不接续第 12 步。

- trainer 于 **10:18:10 UTC** 启动；supervisor 428883、trainer 429020。
  此处记录的是启动时状态，不表示已完成第一个 update。
- 目标 1500 updates，沿用现有入口的 `AREAL_STOP_AFTER_STEPS=1500`；
  epoch=1、total_train_steps=null，原 LR horizon、seed、batch、长度、
  重算及 CPU-staged optimizer 配方不变。
- 仍为单机 8 GPU：actor attention TP4/DP2、EP8/ETP1；rollout 两组
  DP4/EP4、attention TP1。实际拓扑由各自 `backend` 字段决定；旧的
  顶层 `allocation_mode` 字符串不参与该入口的拓扑解析。
- native 固定为 `qwen_areal_barrier_fix_20261018`，库 SHA256
  `9889c7606071872a169cf67569c3130629928047ceef47cc6b4cefb63cae8d99`，
  trainer、actor、rollout 三处路径一致，仍锁频 1620 MHz。
- actor 恢复普通 `areal.infra.rpc.rpc_server` 入口，移除 `debug_actor`
  的逐层抓取/同步检查；保留专家 optimizer 的 ETP norm 修正与 DDP
  `check_for_nan_in_grad=true`。旧 baseline 未热修改这两项，对比时需注明。
- 原生 online W&B，使用权限 0600 的 NETRC，不启动独立上传进程；
  saver/recover 三种频率均 null，recover disabled，不保存 checkpoint。

新 [W&B run](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-weftmoe-pfdt-dp4ep4-mb32-1500-fixed-20261018_train)。
配置、启动脚本、source/native 指纹、状态及日志位于
`build/rl_pf_dt_1500_fix_20261018/`，训练输出位于
`results/areal_pf_dt_1500_fix_20261018/`；旧长跑和诊断记录均保留。

## 2026-10-18：长跑 NaN 排查与 BWD barrier 生命周期修复

截至 04:40 UTC，baseline 已完成 194 次 update，仍在运行；WeftMoE
只完成 7 次，最后一次为 10 月 16 日 11:19:57。不能把旧 supervisor 的
`running` 状态当作训练仍有进展。未启动双机 EP16。

已确认的故障链：

- v7 权重发布之后，两组独立 rollout 服务均出现非有限 logprob，首次为
  11:20:47；后续 workflow 失败/丢弃，固定大小 batch 不断补样，训练等待
  下一批数据。后来的 RPC 连接错误不是首次故障。
- 使用现有 weights checker 的 snapshot/compare 做只读定位，两组推理
  副本都只有 `model.layers.13.mlp.experts.w13_weight` 的同一 EP 分片
  自比较不等。不是简单 KV cache/CUDA graph 问题。
- 检查训练 actor 保留的 CPU 权重和 Adam 状态：rank3、layer13（零基）、
  local expert3/global expert99 的 FC1，BF16 权重、FP32 master 和 m1
  各 72 个非有限值，m2 有 96 个。异常集中于行 608/736/864/992、列
  1280–1315；另 24 个位置 m1 约 `-5.19e32`、m2 为 Inf。
  **损坏已存在于训练权重和优化器状态，不能归因于 AWEX 发布或推理端。**

已修正一个独立的检测漏洞：当前 TE 普通 grouped expert 权重的
`tensor_model_parallel=False`，但 MCore 给专家 optimizer leaf 使用了
attention TP4 组，导致 TP 非零 rank 的独立 EP 专家梯度漏计 norm。
AReaL `MegatronEngine.initialize()` 在 optimizer 创建后，改为给专家
leaf 设置 expert TP 组（当前 ETP1）；保留参数元数据，不改变 AWEX
权重布局。此修改属于公共 Megatron 接入，不是 WeftMoE 特例。
旧 baseline 未热修改；修正后的 norm 不能直接与历史 norm 对比。
这项修正只补上检测漏洞，不负责修复 native 梯度损坏。

Native BWD 中发现了与现场损坏位置精确对应的协议错误：激活阶段结束后，
`Shared.full/empty/pull_full` 共 9 个 mbarrier 未失效，其存储就被作为
dW 的 TMA-store 输出暂存复用。Qwen 配置下的布局如下（字节偏移）：

|对象|shared-memory 偏移|与坏值的对应关系|
|---|---:|---|
|dW 输出 tile 起点|147456|每个 N64 atom 每行 128 B|
|旧 9 个 mbarrier|159744–159815|tile 第 96 行、连续 36 个 BF16|
|损坏的 FC1 行|608/736/864/992|均为 `M-tile * 128 + 96`|
|损坏的 FC1 列|1280–1315|同一个 N256 tile 的前 36 列|

每个 8 B barrier 的第 1/4 个 BF16 对应 NaN；额外 24 个超大梯度位置，
又恰好对应 `full/pull_full` transaction barrier 的第 3 个 BF16，而
`empty` 中没有这类额外值。这是很强的定位证据，不是普通的 GEMM 误差。
[PTX 明确要求](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval)
复用 mbarrier 的存储前先 invalidate；普通 STSM 覆盖不能代替这个要求。
未定义行为的具体硬件内部机制尚不能确定。

修复只发生在 BWD 激活阶段到 dW 的交接：

- producer 尾部 drain 全部 operand empty stages，包含远端 multicast
  consumer 的最后 arrivals。
- 现有 compute join 后，由 producer 线程 invalidate 旧 9 个 barrier，
  再初始化 dW 状态，经现有同步后复用 arena。
- 多机仍在工作的 `NetworkShared` 通信 barrier 不动；没有增大缓冲，
  没有改 GEMM tile、路由、FWD/DT 或稳态流水线。
- 新 artifact 为 `build/pf_main_timeline/qwen_areal_barrier_fix_20261018/`；
  与原长跑使用相同 Qwen tuning，单/多机 BWD 均为 168 个静态寄存器、
  0 spill。多机本轮仅编译验证，另一节点继续运行 baseline。

修复前的排查测试（不能作为修复后验证或长跑稳定的证据）：

- EP8/E256/top8/H2048/I512：原生 FWD/BWD parity、workspace 清理和多层
  graph 回归通过；新分配浮点张量先填 NaN，也通过。
- 实际 Megatron TP4/EP8、两层 MoE、DDP、两 microbatch、两次 optimizer/
  offload 周期通过。补上显式 finite 断言，避免 NaN 使误差比值比较静默通过。
- 1000 次变化 token 数（8191/20003/32767/17）随机路由 FWD/BWD 通过；
  另 500 次强偏斜路由、NaN 初始化分配测试也通过。
- AReaL 专家 TP 组修正及重复参数回归：2 passed；相关 Ruff/diff 检查通过。
  尚未复现完整 RL 中首次坏梯度，不据此宣称长跑稳定。

修复后二进制验证：

- EP8/E256/top8/H2048/I512、T257 的数值对照、workspace 释放、多层
  graph/offload 测试通过；实际 Megatron TP4/EP8、DDP、两层/两
  microbatch、两次 optimizer/offload 测试通过。
- 500 次强偏斜随机路由，token 数循环 8191/20003/32767/17；浮点新
  分配先填 NaN，byte scratch 先填 0xff，所有输出/梯度保持有限。
- 另测 EP8/T17、全部路由集中 rank0：数值与 graph/offload 回归通过，
  覆盖其余 rank 没有激活任务时的 producer-tail/barrier 退出。
- 1620 MHz、EP8，使用原 `benchmark_training.py --backward-only`，
  随机 tokens 的标准差 10%、共享路由偏置 0.5、6 warmup/24 samples，
  先 prime trace。两版逐 rank workload hash 一致；缓存就绪后测 BWD，
  含调用开销、权重重排、临时分配，不含 FWD 和输入准备：

  |每 rank nominal tokens|修复前 BWD p50|修复后 BWD p50|变化|
  |---|---:|---:|---:|
  |8192|4.163 ms|4.161 ms|-0.04%|
  |20480|9.541 ms|9.699 ms|+1.66%|

  这是单轮性能回归检查，不把 20K 的差异直接解释为稳定回退或测量噪声。
  原始结果保留在 `build/rl_nan_20261018/bwd_{before,after}.log`。
  后续完整 RL 已通过 12 次 update（见本页开头）；这不替代 1500-step
  长跑验证。

本机损坏 run 已停止，保留日志和小型异常摘要，没有生成 checkpoint。
诊断重跑配置保留原 DP4/EP4 rollout、TP4/EP8 actor、1620 MHz、原生
online W&B，关闭 checkpoint/recover；启用 DDP NaN 检查，在 native BWD
返回（DDP 累加前）和 optimizer 更新后检查异常。首次 native 异常时只
保存对应单层的跨 rank 输入/梯度以便独立复现，不保存整个模型。
诊断最多 12 次 update，额外检查的耗时不能作为性能结果。

重跑前曾阻塞于旧 actor 的驱动清理：残留进程为 D 状态，内核栈在
`nvidia: portSyncRwLockAcquireWrite / os_acquire_rwlock_write`，SIGKILL
不能使其退出，`nvidia-smi` 查询超时。未重置 GPU、卸载驱动或重启节点，
也未影响另一台健康 baseline。04:42 UTC 八卡 CUDA 冒烟检查恢复通过，
另外发现旧 rollout 的 14 个孤儿子进程仍继承着 CUDA 句柄；按精确 trial
和 PID 清理。诊断资料与启动配置位于 `build/rl_nan_20261018/`，后续
保留诊断记录，不能直接恢复损坏的 v7 权重。旧 native 的诊断重启尚未进入
首次 update 就停止，日志保留为 `old_native_*`；修复版使用新 trial
`qwen35-weftmoe-barrier-fix-20261018`，仍从初始模型开始。
修复版 trainer 于 04:58:01 UTC 启动（supervisor 330634、trainer 330744），
该诊断已完成 12 次 update 并退出。AReaL 公共 norm
修正位于其现有 working tree；未将其他已有集成改动混入本次 WeftMoE commit。

## 2026-10-16：两机分别长跑 1500 updates，原生 online W&B

每台机器仍运行一组独立的单机 8 GPU 实验，不改成双机 EP16：

|节点|长跑内容|W&B|
|---|---|---|
|10.10.19.233|原版 SGLang + 原版 Megatron|[baseline](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-baseline-pfdt-dp4ep4-mb32-1500-20261016_train)|
|10.10.16.219|SGLang DT decode / PF prefill + Megatron PF FWD/BWD|[DT/PF](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/dapo-math-grpo-qwen3.5-35b-a3b-base-cpu-staged_qwen35-weftmoe-pfdt-dp4ep4-mb32-1500-20261016_train)|

最初排队等待 20-update 短测，09:40 UTC 按用户要求改为立即启动，并
切换到训练进程原生 W&B。停止旧排队进程、两组旁路 publisher 和短测；
短测最终完成 baseline 10 次、WeftMoE 8 次 update，原始记录保留。
先 SIGINT 正常清理，再按 PID＋创建时间仅清理本次遗留进程，确认 GPU
释放后启动。baseline trainer 于 09:44:51 UTC 启动，WeftMoE 于
09:47:00 UTC 启动；此时为初始化阶段，不表示已经完成第一个 update。

两组长跑从相同初始模型和新 optimizer 开始，不续接短测。使用已验证
的接入代码和 native 库（启动记录 commit `758f651`）；逐项对比旧、新
resolved config，训练参数不变，仅更新 trial、输出/name-resolve 目录、
认证及 W&B 配置。仍设 1620 MHz，保留 5 秒时钟/显存采样。
`total_train_epochs=1`、`total_train_steps=null` 和原 55,990-update LR
horizon 不变，直接通过已有 Python 训练入口的
`AREAL_STOP_AFTER_STEPS=1500` 停止；未修改 AReaL 的 launcher/scheduler。

`stats_logger.wandb.mode=online`（不是无效值 `enabled`），由 AReaL
`StatsLogger` 原生登录和上传；**不再运行独立上传进程**。新 group 为
`qwen35-pfdt-dp4ep4-1500-native-20261016`，不要看之前空的旁路长跑 run。
两节点使用各自 `/tmp` 下权限 0600 的 NETRC，已验证服务器认证；配置的
`wandb_api_key` 留空、环境不传 `WANDB_API_KEY`，避免 AWEX 环境日志泄露
密钥。仅关闭 W&B 自动 console 上传，原生数值指标和系统监控正常启用。
后续 benchmark 入口同样支持通过 `NETRC` 开启原生 online W&B。

checkpoint 和 recover 全部关闭：saver 三种频率均 null，recover 为
disabled 且三种频率均 null。完成后只检查对应 trial 的 checkpoint
目录，若意外产生则删除，不动原模型和日志。因此失败时没有 checkpoint
可恢复。按当前约 12 分钟/update 粗估，单组约需 12 天，实际取决于后续
生成长度和系统稳定性；这不是完成时间保证。

配置、native SHA256、前驱 PID/创建时间、启动脚本、状态、console 日志及
旁路历史记录保存在 `build/rl_pf_dt_1500_20261016/`；训练输出和原生 W&B
本地记录使用 `results/areal_pf_dt_1500_20261016/`。两份配置已通过 AReaL 的结构化配置
解析，后台进程与 SSH 会话脱离；短测记录和下文历史结果不覆盖。

## 2026-10-16：当前 PF/DT 重新接入，阶段性短测记录

下文为切换长跑前的过程和快照；短测现已按上文停止，不再继续到 20 次。

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

以上为最小接入验收；下面另列完整 RL update 的阶段性结果，20-update
对照仍在运行，尚不能据小样本作最终稳态结论。
启动时 W&B 为 disabled；随后按用户要求启用旁路补传和持续同步，见下文。

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
错误。首轮 LR=0；后续已完成 LR=1e-6 的两次 update 及再次生成。

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
但 `update_weights` 多约 10.76 s，不能省略此开销。该指标仅覆盖其
框架计时区间，**不包含随后全部 Graph/KV 恢复**：AReaL 的
`_export_stats_then_restore_awex_rollout()` 先打印统计，再恢复推理资源。
因此同时统计相邻 update 日志的墙钟间隔，覆盖切换/恢复开销；不要简单
相加 rollout、train_step、update_weights 后就称为完整端到端时间。
首轮含编译和首次分配，需单列，后续按相同 update 编号排除首轮配对统计。

### 前三轮配对结果（2026-10-16，阶段性快照）

截至该快照 baseline 完成 5 次、WeftMoE 完成 3 次 update；只配对两组
前 3 次，排除第 1 次，以下为 **updates 2–3**。两组仍继续跑到 20 次。
这是在线 RL 采样，不是固定张量 replay；相同 seed/配方不代表相同的
生成轨迹、长度或 reward。两轮有效 token 总量分别为 9,288,900 和
8,975,300，不能直接把耗时下降全部当作吞吐收益。

|指标|本轮 baseline|当前 DT/PF|变化|
|---|---:|---:|---:|
|rollout 平均耗时，s|552.76|516.28|-6.60%|
|actor train_step 平均耗时，s|163.20|141.53|-13.28%|
|update_weights 平均耗时，s|14.761|16.408|+11.15%|
|相邻 update 墙钟间隔均值，s|736.893|701.443|-4.81%|
|有效 token / rollout 秒|8,402|8,692|+3.45%|
|有效 token / train_step 秒|28,459|31,709|+11.42%|
|有效 token / 完整墙钟秒|6,303|6,398|+1.51%|

吞吐统一采用 `sum(valid_tokens) / sum(seconds)`，不是平均单步加速比。
墙钟区间为第 1 次统计到第 3 次统计，包含区间内的恢复、共卡切换等
框架开销；不能与单独的 routed-MoE kernel 加速比混为一谈。当前观察是
rollout 和训练都有收益，但完整 RL 的收益较小。第二轮框架未单列时间
约 23.90 s，baseline 约 3.10 s；现有计时还不能将差额全归因于某个
restore/IPC 操作，未做额外 profiler 消融，也未为改善数字修改框架调度。

历史同一 **updates 2–3** 口径：

|2026-09-30 版本|rollout 有效 token/s|训练有效 token/s|墙钟有效 token/s|
|---|---:|---:|---:|
|历史 baseline|8,265|25,563|6,057|
|历史 rollout-only WeftMoE|9,053|27,767|6,468|

旧 WeftMoE 的 actor 也是原版 Megatron。历史与本轮基线自身已有变化，
不能拿历史较慢的训练基线计算本次 PF 接入收益；上述加速比均使用本轮
新测 baseline。两轮样本不足以判定新旧 rollout 或完整 RL 的稳定排名。

两组均设置 `-lgc 1620,1620`。截至快照，5 秒采样中 baseline 有 3 条、
WeftMoE 有 1 条短暂低于 1620 MHz，其余为 1620；原始记录不删样。
驱动显存峰值约 baseline 102,729 MiB、WeftMoE 107,823 MiB，包含整个
框架及动态缓存，并非 native pool 单独占用。

接入代码为 `159dbb8`，DP 混合批次修复为 `23f0a89`；按 AReaL
`debug-distributed` 的跨 rank 堆栈检查定位后，只修正执行池选择，未改
算子数学、训练配方或调度器。此阶段无源码热更新、无新的 GEMM autotune。
原始日志、时钟、服务端 metrics，以及自动汇总的 `comparison.json`
保存在 `build/rl_pf_dt_20261016/`；该三轮快照另存
`comparison_first3_snapshot.json`，后续完整结果不要与此小样本表混用。

### W&B 同步与 checkpoint 策略

2026-10-16 按用户要求启动 `benchmarks/publish_areal_logs.py`，无需重启
两组训练：补传已完成的 update，随后跟随日志持续同步到第 20 次。
服务端 API 已核实收到 baseline 1–5、WeftMoE 1–3 的完整统计表，后续
记录继续追加。项目 `sct-test/areal-qwen35`，group
`qwen35-pfdt-dp4ep4-20261016`：

- [baseline](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/qwen35-pfdt-dp4ep4-20261016-baseline)
- [DT/PF](http://8.150.1.98:8080/sct-test/areal-qwen35/runs/qwen35-pfdt-dp4ep4-20261016-weftmoe)

只上传完整 console 指标和不含秘密的配方元数据，精度保持 console 输出；
不上传 checkpoint、数据集或源码，密钥通过无回显输入提供，未写入代码。
原始日志和首轮失败尝试仍保存在本地，不将失败尝试混入上述两条曲线。

本轮实际配置已经是 `saver.freq_{epochs,steps,secs}=null`、
`recover.mode=disabled`，检查时没有生成 checkpoint；`saver.mode=auto`
只是同步/异步保存方式，不代表开启保存。后续 benchmark 入口在配置覆盖
末尾显式固定上述策略，同时清空 recover 的三种频率，防止配方变更重新
开启保存。若环境提供 `WANDB_API_KEY`，入口默认启用在线 W&B；地址从
`WANDB_BASE_URL` 读取。

本轮另有 `build/rl_pf_dt_20261016/cleanup_checkpoints_after_run.py` 等待
20 次 update 完成及对应 trainer 退出，再复查/删除这两个 trial 各自的
checkpoint 目录；不触碰日志、原始模型或其他 trial。检查/清理结果写入
同目录的 `checkpoint_cleanup.log`。当前尚未删除任何文件。

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
