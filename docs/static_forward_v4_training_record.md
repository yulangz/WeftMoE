# V4 ring 训练对接记录

对应设计文档 M6，按时间保留改造、失败候选和验证记录。
2026-10-01 当前 PF 训练已使用 count-only mini Notify、main 内 planner、
按实际 R/P 分配的独立 tape，以及静态 Backward；非 ring 路径保留 V3 对照。
当前已采用 `cf909bf`（Backward peer mapping 复用）及本文末尾 Y2 的
208/88 Backward 寄存器配额，Forward 配额不变。

1620 MHz、仅验收 EP8 单机/EP16 双机的 8K/20K：FWD 已领先，BWD 的双机
8K 已接近 V3；实际 50 warmup + 200 measured 循环尚未全部领先。
不把更早 GA+FZ 的性能或单个 GPU kernel 收益当作当前完整训练已达标。
完整数值/生命周期检查和已有专项 memcheck 已通过；旧 Backward racecheck
的未决报告及工具版本限制仍保留，不能称完整训练 sanitizer clean。

## M6a：按实际接收量分配 rank-private Backward 中间区

先解开 Backward 对 transport pool 全容量激活区的依赖。U/dAB/dX/dGate
中间数据均只由本 GPU 读写，不需要 CUDA IPC 或 GIN registration；远端需要的
compact dR、source contribution、通知和 canonical directory 仍维持原地址。

本步将这四个 private plane 合并为一次独立分配，由 `BackwardTensorStorage`
及对应 handle 持有到完成。分配发生在 prepare 阶段，不在 persistent kernel
内进行；支持调用方 allocator。TMA maps 跟随这次分配，不留在可复用的 pool
slot 中。删除原 pool 的四个 payload 区及五个私有 TensorMap，避免双重预留。

行容量取 materialized Notify 已知的本 rank 实际 `R` 加
`owned_experts × (TileM−1)`。这是按实际 R 的安全 padding 上界，**不是精确
P_actual**；不下载路由或再运行一次 count exchange。每 rank 的余量只与本地
expert 数有关，不用 global max(R) 让所有 rank 申请同样大小。后续 training
plan 可进一步提供精确 P。本步未改变 canonical 行号、dW 归约顺序或 GEMM。

H2048/I512 下，四个 plane 每容量行共7172字节，另有最多三个对齐间隙。
`BackwardTensorStorage::bytes()` 计入此分配；pool usage 相应下降，但不能只
报告 pool 减少而忽略单次 Backward workspace。多个 live Forward tapes 不会
提前占用这些 Backward 临时区。

迁移前参考：DC inference 配置的 EP8 T64 通过现有 tensor-runtime 的
output/dX/dGate/dW1/dW2、multilayer graph、offload 检查（各8 ranks）；同配置
EP16 在 Forward launch 前拒绝，原因是其 `RDMA_CTAS=0` 编译关闭了旧多机
训练入口，不是梯度数值失败。后续训练候选使用冻结 V3 multi-node profile
`configs/static_inference_v3_20260923/` 建立参考，不把改变 profile 当作改造收益。
日志 `build/m6_baseline_tensor_ep8_t64.log`、
`build/m6_baseline_tensor_ep16_t64_{local,remote}.log`。

冻结 V3 native 在 EP16 T64/E32/K6 的 Backward 双次运行通过16 ranks
`repeat_equal=1`，日志 `build/m6_frozen_v3_backward_ep16_t64.log`。此回归与
编译同时运行，只作为数值参考，不取其中时间作为性能基线。

M6a候选native/library已构建，host18/18通过；native SHA256
`8760069662d5daeef278dfc7374e456ea0aba6fc579f7dc939654d7d570da6c7`，library
`02d2e4286867e5793f5799bf495746c453758f98ee3a239244c7694e26b6e730`。
冻结于 `build/static_v4_training_private_workspace/`。新旧unified Backward
均REG168/SHARED2048/LOCAL0/CONSTANT3200，GIN/local栈分别208/40bytes。
EP8 T64/E64/K4现有tensor-runtime回归通过：五种数值输出、multilayer graph、
offload各8 ranks，日志 `training_private_workspace_torch_ep8_t64_retry.log`。
首轮训练数值也通过，但错误地对整个Python测试设置了disable-static开关，导致
随后依赖static入口的inference graph准备拒绝；去掉该测试开关后全部通过。
训练本身始终为V3，并未因此启用ring training。

EP16 native T64/E32/K6通过，两次运行稳定；64个梯度hash（16ranks×四类）
与冻结V3逐项一致。pool/compute arena各减少4,575,744 bytes/rank，新单次
Backward tensors共17,422,328bytes，其中private workspace为4,575,736bytes。
此fixture的capacity等于实际R384，因此主要是所有权/分配时点迁移，**不是净省
4.58MB**。日志 `training_private_workspace_backward_ep16_t64.log`。

同一EP16 fixture只instrument rank1/9的memcheck：两份报告0errors、launcher0。
racecheck则未通过：候选报告19,706/17,876 errors；完全未改的冻结V3同方法报告
19,761/16,911 errors，均包括Forward共享内存及Backward DSM mailbox访问。
这证实这些类别不是仅在M6a出现，但不等于逐项确认误报或证明全部hazard相同。
Backward终止/DSM与Forward staging的独立分析仍待处理，不能声明sanitizer
全通过。日志 `training_private_workspace_backward_ep16_t64_{memcheck,racecheck}.log`、
`m6_frozen_v3_backward_ep16_t64_racecheck.log` 及对应带hostname/PID的工具报告。

现有PyTorch回归进一步通过以下完整case（Forward、四种梯度、graph、offload）：

- EP16 T64/E64/K4、max_rows4096：均匀及全部命中rank0两种路由，各16 ranks。
  热点rank实际R4096，其余15 ranks为0，覆盖不同rank的private plane容量。
- 单机EP8 T512/E64/K4、max_rows16384：全部命中rank0，8 ranks通过。
  热点R16384，覆盖更长的private workspace与多M-block梯度归约。

日志 `training_private_workspace_torch_ep16_t64_skew{0,1}_{0,1}.log`、
`training_private_workspace_torch_ep8_t512_hotspot.log`；所有launcher exit0。
只为现有脚本增加`TEST_MAX_ROWS`以显式容纳热点，没有新增测试套件。

EP16 native另以调用方allocator分配12轮、warmup10轮运行，通过缓存warmup后
无新allocation、12次逐输出重复校验，hash仍与冻结V3一致。新workspace与输出
一起计入`tensor_bytes`；日志 `training_private_workspace_backward_ep16_t64_allocator.log`。
`DeviceBackwardHandle::Impl`最后一个成员是completion，销毁先等待完成再释放
tensors，新增private allocation沿用这个生命周期，不要求Forward提前持有它。

M6a数值与memcheck已验证，racecheck存在上述冻结基线也能复现的未决项。
本步没有修改inference路径，不重新声明inference性能收益，也不标记M6完成。

仍待完成：ring training plan/main 分离、canonical checkpoint 映射、rank-local
peer workspace 描述符、compact dR 等通信平面的实际分配、Backward 有界 GIN
窗口、collective OOM，以及训练长文/热点/多个 live tapes 的完整门禁。
M6a 即使通过，也不满足上述全部目标。

## 训练 racecheck 定位（进行中）

现有工具为Compute Sanitizer2025.2.1.0/build35969825。NVIDIA后续
[发布说明](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
记录2025.4修复了多项mbarrier相关racecheck误报；这只是对照工具版本的理由，
不是将本项目未决hazards定性为误报的依据。

从官方CUDA13.1.0 redistrib清单下载独立sanitizer13.1.75归档，SHA256为
`98326d9dd647977956ce1f4898dcaca8db549fe9657e1d7a4cd4343990a17887`，解压到
`build/toolchains/sanitizer_13_1/`，工具版本2025.4.0.0/build36782660。没有
修改系统CUDA、NCCL或驱动。尝试对同一个M6a冻结native、同一个EP16 T64夹具
插桩rank1/9，两端在启动时均报unsupported driver，要求580+；当前570.148.08。
该轮未进入kernel，不能算racecheck通过/失败的算子证据。launcher已主动中断
并清理本轮进程，两节点GPU确认无残留任务，不升级系统驱动。

日志 `training_private_workspace_backward_ep16_t64_racecheck_2025_4.log`、
`training_private_racecheck_2025_4_{hostname}_{pid}.log`。继续用已安装的2025.2.1
与独立`-lineinfo`诊断构建定位来源，不关闭hazard/DSM检查或添加suppression。

### 聚焦 Backward 与独立同步复现

同M6a代码/冻结V3配置增加`-lineinfo`，native SHA256
`c3947d8ff9a7c60c2d4b0fcf57be2b72ac0c4ec7b5792d0be3cdd4f0499bc11e`。
仅过滤`unified_backward_kernel`且`--launch-count 1`时，rank1/9均0errors，
16 ranks重复输出及64个梯度hash仍与冻结V3一致。**这只插桩第一次Backward，
不能作为重复调用的门禁。** 去掉launch-count后，两次Backward都插桩，lineinfo
版报告21,732/19,060 errors，原M6a版为21,762/19,916 errors，均未通过。
日志`training_{lineinfo,original}_backward_all_ep16_t64_racecheck.log`及对应
hostname/PID报告。lineinfo版资源仍REG168/STACK208/SHARED2048/LOCAL0。

已显示的一类RAW/WAR落在同CTA共享内存`cuda::atomic_ref<unsigned>`的
`ld.acquire.cta.b32`与`st.release.cta.b32`本身。构造不含NCCL、GEMM、ring的
双warp最小握手：producer等phase0、写payload、release phase1；consumer
acquire phase1、核验payload、release phase0；32轮、两次kernel。工具2025.2.1
同样在原子phase上报告1,512 errors，payload核验0错误。这确认了**共享原子
读写这一特定误报类别**，不能据此排除生产kernel其它payload或DSM hazards。
临时源码/日志`build/racecheck_shared_atomic_repro{.cu,.log,_output.log}`，没有
为此增加永久测试套件或关闭检查。

另一个独立双CTA cluster复现只保留计算warp的两阶段mbarrier mailbox握手，
其它warp提前退出。32轮读写、两次launch：payload核验0错误，但工具报告64个
DSM early-exit errors。相同计算加入全CTA汇合及完整cluster退出同步后，工具
0errors/0warnings，payload仍0错误。源码`build/racecheck_cluster_mailbox_repro.cu`，
通过`-DFULL_CLUSTER_EXIT`选择对照；日志`racecheck_cluster_mailbox_{repro,full_exit}`
及`_output.log`。这定位了工具可识别的生命周期边界；数据值正确本身不证明
原生产kernel的所有DSM生命周期正确。后续候选只在Backward收尾增加完整cluster
退出边界，逐任务同步仍限于计算域，再验证资源、数值与sanitizer，未提前合入。

### Backward 完整 cluster 退出边界

计算与通信域各自结束persistent loop后，全CTA汇合，再做一次完整cluster同步。
两个CTA都不允许某个warp提前退出；每任务的mailbox/TMA握手仍不包含通信warp，
没有将通信数据依赖变成逐任务全CTA barrier。两个寄存器域仍分别执行完整循环。

冻结`build/static_v4_training_cluster_exit/`，native SHA256
`75825b4f45bbe7e18a7be7bb6331fc5c457dc6a6bc1cb253e0822e5eef0aae36`，library
`18464d8225081b41324b26ec22bb8c00b5706d078bbe8b43507b7001e897f178`。
GIN/local Backward资源分别REG168/STACK208、REG168/STACK40；两者SHARED2048、
LOCAL0、CONSTANT3200，与M6a相同。只改变训练Backward入口，不改变inference。

EP16 native T64/E32/K6双次运行通过，16 ranks的64个梯度hash全部等于冻结V3。
PyTorch完整数值/graph/offload回归通过EP8 T512/E64/K4/max_rows16384热点和
EP16 T64/E64/K4/max_rows4096热点，分别8/16 ranks三类PASS齐全。日志
`training_cluster_exit_backward_ep16_t64.log`、
`training_cluster_exit_torch_ep8_t512_hotspot.log`、
`training_cluster_exit_torch_ep16_t64_hotspot_{0,1}.log`。

只插桩两次unified Backward、rank1/9的首轮报告为1,852/1,904 errors。随后提高
print-limit至100000完整展开，分别1,864/1,948 hazards（显示数等于总数）；合计
3,812条的7,624个读写侧**全部**为libcu++共享32-bit atomic load/store，包含
acquire与relaxed读取、release写入。普通payload及DSM early-exit报告均为0。
这些共享原子报告符合上述独立复现的工具误报类别，但工具原始退出码仍为1；
不写成原始racecheck 0errors，也不扩展结论到未插桩的14 ranks或训练Forward。
日志`training_cluster_exit_backward_full_ep16_t64_racecheck.log`及
`training_cluster_exit_backward_full_racecheck_{hostname}_{pid}.log`。后续工具升级后
仍需重跑无过滤的完整训练路径，目前未添加suppression。

同EP16双次运行另做rank1/9 memcheck，两份原始报告均0errors、launcher exit0，
16 ranks重复校验通过；日志`training_cluster_exit_backward_ep16_t64_memcheck.log`。
保留此生命周期边界，作为M6训练衔接的同步基础修正提交，不宣称训练迁移完成或
inference性能改善。没有为工具误报替换正确的acquire/release，也没有新增单测套件。

## M6b：每调用 tensor preparation 的一致失败边界（已验证，训练ring未完成）

新增独立host侧`CollectiveTensorPreparation`，在训练Runtime初始化时预分配
两个uint32的device/host状态并预热NCCL控制collective（native device申请最小
128bytes，pinned申请8bytes）。每次Forward/Backward的分配回调先在本rank运行，
捕获异常后用同一个预分配状态做rank-min AllReduce。只有全部rank成功才进入
执行；失败时所有rank在main前抛错并报告首个失败rank，发生异常的rank保留
nested原始异常。部分完成的临时tensor沿既有RAII释放，不覆盖仍存活的tape。

Forward边界覆盖输入/权重lease及packed权重准备、输出、X/G/U及路由保存；
Backward覆盖四个输出和M6a私有中间workspace。调用者提供的tensor allocator
也位于回调内。模块不依赖ring/Notify内部布局，后续native训练plan/实际payload
分配可复用相同边界；现有低层`prepare_*_tensors`仍是local API，不能据此声称
所有native调用者自动获得一致失败。Runtime的inference路径不创建或使用此模块。

这是eager allocation boundary：需要host同步，不能用于CUDA Graph capture，
也不是数据面新barrier。未处理Runtime/通信资源本身首次创建失败、失效CUDA
context或损坏communicator；不能称为任意错误恢复。M6 StableTape/window plan、
rank-local peer arena以及training ring仍待实现。

在现有`torchrun_tensor_runtime.py`加入显式可选的
`TEST_TENSOR_PREPARATION_FAILURE=1`：仅最后rank在Forward第三次tensor分配、
Backward第五次分配（私有workspace）注入OOM异常；后者先由`cudaMalloc(2^63)`
返回真实`cudaErrorMemoryAllocation`，不实际占用整机显存。检查全部rank得到错误、
Forward失败不遗留Notify lease、Backward失败仍持有一份可重试tape；重试的
output/dX/dgate/dW1/dW2沿用原PyTorch参考校验。超出可寻址容量的申请被拒绝，
不等于把机器填满后的allocator碎片/压力实测；两者不混称。

冻结`build/static_v4_training_tensor_preparation/`，library SHA256
`31984df9d12257e0bf06deef968a507e85c36107bd1cd315dadf24a1cf8fdf0b`，extension
`910296a012e59ac88e8b1dfa4e72a19fbe6a38d4fef7e9f6bd2b822f6a6e0cdf`。
使用冻结V3生成profile、`-lineinfo`，未改训练GEMM或数据面同步。

- EP8 T64/E64/K4/max_rows4096均匀：16个失败通知、8个重试数值标记、原有
  parity/graph/offload各8项，launcher0。
- EP16相同shape、hotspot集中到rank0（含零接收rank），错误注入rank15：
  32个失败通知、16个重试数值标记、原有parity/graph/offload各16项，两端launcher0。
- 原生两rank独立诊断不经PyTorch：rank1直接cudaMalloc失败，确认其last-error
  仍为2再抛`std::bad_alloc`；两rank同报origin1，随后再次run成功，均exit0。
  未添加全局“清除任意CUDA错误”逻辑，现用NCCL路径能够完成这项恢复。

日志`training_tensor_preparation_torch_ep8_t64_failure.log`、
`training_tensor_preparation_torch_ep16_t64_hotspot_failure_{0,1}.log`、
`collective_tensor_preparation_repro.log`。原生诊断源码仅build归档，不建立额外
常驻测试套件。门禁证明此分配边界及当前Runtime可恢复，不代表M6训练ring完成，
也未覆盖首次pool创建OOM或每一种NCCL/driver错误。

## M6c：按实际行数分配 Backward peer payload

Backward的compact dR、source contribution、source dGate不再引用Forward完整
arena中的三个固定offset，改为execution slot拥有的独立peer workspace。
每rank的payload按本次实际接收R及本地T×K分配：dR为`max(R,1)×H×BF16`，
contribution为`T×K×H×BF16`，dGate为`T×K×FP32`，另有128-byte对齐及本地
peer地址表。零接收rank只为有效TMA descriptor保留一行dR。scheduler、ready
flags、compact→padded等小型metadata仍保持共同容量/offset，没有将其错误地
改成各rank不同的scheduler几何。M6a的私有padded planes保持不变。

IPC payload用独立native CUDA allocation，不导出Torch allocator子分配。每rank
交换handle及两个plane offset，只导入实际同节点peers；设备表保存本进程导入
地址，不能直接使用对方进程的原始CUDA指针。GIN仍经原有有界ChunkRing搬运，
不把完整dR/Return payload注册成额外跨机arena。

同一immutable Notify generation重复Backward直接复用已准备的workspace。
新generation即使本地shape相同也交换descriptor，因为peer可能换buffer；只有
实际R及T×K相同才复用本地allocation，不按历史最大R缓存。换buffer的顺序为：

1. M6b一致失败边界内分配candidate，所有rank成功后交换descriptor。
2. 关闭变更peer的旧mapping，再打开新mapping；对import结果做一致检查。
3. 所有rank成功意味着旧mapping已关闭，此时才释放旧exported buffer，发布
   本地设备地址表，然后同stream进入Backward。
4. import失败时全部rank关闭所有mapping，再汇合后释放candidate/旧buffer；
   不保留能被下一次调用误认为有效的generation缓存。

这是host preparation，不是每packet数据面barrier。发生resize时会短暂同时持有
旧buffer和candidate；不是只有新shape大小的峰值承诺。当前测试没有注入IPC
import故障，已有OOM注入仍针对Runtime tensor preparation；不声称任意IPC/NCCL
错误可恢复。训练CUDA Graph不是本次验收范围，记录中的graph门禁是inference。

冻结`build/static_v4_training_peer_payload_m6c/`，沿用冻结V3生成profile及
`-lineinfo`。native SHA256
`a39dd60d8fae0e6111c4c5adb8182fc4132783f326fe7ed4eca451186812d903`，library
`099fc7e504a670f894cfb6f6e4ed142ee85564dee5dc0366414305917738bd5d`，extension
`910296a012e59ac88e8b1dfa4e72a19fbe6a38d4fef7e9f6bd2b822f6a6e0cdf`。
最终只有删除一处未验收graph描述的comment改动。Backward GIN/local资源保持
REG168、STACK208/40、SHARED2048、LOCAL0、CONSTANT3200，没有改变GEMM流程。

- host原18/18通过；单机EP8 T64/E16/K6原生两次输出一致。
- EP16 T64/E32/K6两次重复输出通过，64个梯度hash与cluster-exit冻结基线一致
  （该基线已与冻结V3逐位核对，本次没有重跑V3）。调用方allocator另跑12轮、
  warmup10轮，12次重复校验及缓存warmup后无新tensor allocation检查通过。
- EP8、EP16 Torch T64/E64/K4/max_rows4096分别通过8/16 ranks的输出、dX、
  dgate、dW1、dW2参考校验、单rank OOM一致失败及tape重试、inference graph/offload。
- 原Torch入口增加可选`TEST_PEER_PAYLOAD_REUSE=1`，复用同一参考计算函数，
  不新建测试套件。保留三个tape后按顺序反传：T64热点首rank→T16热点末rank→
  T64热点首rank。全部五项数值、Notify lease逐个释放、第三次pool用量等于
  第一次通过；EP8/EP16分别24/48个reuse标记。
- 原生EP16全流程memcheck只插桩rank1/9，两份报告均0errors，launcher0。
- Torch热点迁移另对rank0/15的`unified_backward_kernel`做memcheck，两份报告
  均0errors、两端launcher0；48个reuse标记及后续inference graph/offload全部
  通过。此项检查实际覆盖resize后的大R热点，不只是原生fixture的固定shape。

附加Torch memcheck首轮在训练结束后停顿：native栈显示非插桩rank正在
`ncclCommFinalize`，插桩rank已进入下一个Runtime的PyTorch `all_gather_object`。
该轮被显式终止，没有最终sanitizer summary，不计通过。测试原先仅`del runtime`，
现在在训练→inference阶段边界使用已有collective `runtime.close()`，不依赖
各rank恰好同时触发Python/C++析构；未修改算子数据面。相同条件重跑正常结束，
报告local195972/remote231758。保留首轮日志、rank0/1 native stack及重跑的
`torch_ep16_t64_close_memcheck_{0,1}.log`，不将停顿归因为已证实的工具误报。

EP16热点迁移的`compute_bytes`实测如下（含固定Forward arena，不含GIN及tape）：

|rank|T64热点rank0|T16热点rank15|T64热点rank0|
|---|---:|---:|---:|
|0|47,556,152|29,995,832|47,556,152|
|15|30,783,032|34,186,040|30,783,032|

**这一步不是总显存优化完成。** V3 Forward仍使用并保留原完整X/S/output及
contribution。当前实际-sized Backward payload与之并存，只删除了原pool dGate
plane；原生EP16均匀fixture的总pool由360,114,880升至363,262,272 bytes/rank。
`pool.usage()`计入新增payload和descriptor exchange；CLI旧`workspace_bytes`仍
仅表示固定compute arena，不能单独用于比较总空间。下一步training ring接通后
才能删除不再需要的Forward完整payload。本次不报告inference加速或训练性能收益。

日志统一前缀`training_peer_payload_m6c_`：`backward_ep8_t64.log`、
`backward_ep16_t64{,_allocator,_memcheck}.log`、`torch_ep8_t64_reuse.log`、
`torch_ep16_t64_reuse_{0,1}.log`；冻结目录含source patch与sanitizer wrapper。
M6的StableTape/window plan、checkpoint生命周期及Backward GIN window adapter
仍未完成，训练继续走V3。

## M6d：单机训练 ring、canonical tape 与 plan/allocate/main 边界

本步接通单机 opt-in training Forward，不是多机训练完成。未设置
`WEFTMOE_STATIC_RING_TOKENS` 时保留 V3；设置后单机使用静态 ring，当前多机
training pool 显式拒绝，等待 Backward GIN window adapter，不静默退回 V3。

### 执行和生命周期

- 先 materialized Notify，保留其 canonical `route_dst_row`、原 combine order
  及 Backward metadata。独立 `static_ring_plan_kernel` 再冻结窗口/物理行映射；
  它只发布 routes/gates/行号，不搬 X。两层规划都在 main 前，**不是一遍 planner**。
  训练入口拒绝 fused inference / replay-counts Notify，避免复用的可变路由元数据
  破坏 live tape 的 canonical 行号合同。
- 训练专属 source `uint64[route]` 和 physical-plan `uint64[row]` 记录 tape 行号。
  不改变 inference 的24B RingRow，不把 atomic 物理枚举当作稳定逻辑顺序；
  ring 使用自己的 DeviceDispatchRuntime，不改 retained Notify 的 M64 geometry。
- host 读取实际行数，与 Notify 核对后分配独立 saved-X/G/U/gates/headers及输出。
  新的 rank-local `ForwardMegaKernelConfig::prepare_tensors` 回调在此执行，backend
  用预分配的 CollectiveTensorPreparation 一致失败。原生未绑定 tensors 的路径
  也在此分配；显式绑定的 caller-owned storage 仍可复用。没有 main 内扩容。
- 已完成协议的分配失败用 `TensorPreparationFailure` 区分于 CUDA/NCCL 控制面故障。
  前者可释放 slot 并重试：plan 已 join，尚未准入 payload，source/block bases
  未推进。没有“恢复失效 context/comm”或首次 pool OOM 恢复的承诺。
- main 只 stage X、发布 node-local ready，并消费冻结 plan，不重数或重交换。
  Pull 复用既有 FanoutRowPrefetchPipeline：一次 G2S，向物理 X 和 saved-X 各写一次；
  **两次 S2G 全部完成才发布 X full**。gate/header 同时按 canonical 行保存。
- FC1 packed gate/up accumulator按同一 BF16 rounding 写入标准分离的 G/U tape；
  S 仍在物理 ring。FC2/Return/Reduce 沿用静态 inference 数据面，Backward 使用
  原 canonical 顺序及固定 dW reduction，不再访问被覆盖的 X/S slot。
- live tape 持有独立 Notify lease 和 tensors；物理 plan/ring 在 Forward 完成后
  即可被后续调用重用。训练 plan/allocation 入口明确拒绝 capture；后续测试中的
  graph/offload 是**独立 inference 阶段**，不是训练/Backward Graph 支持。

### 容量

Python training ring 将 Notify/调度**元数据**容量至少扩到 `world*max_tokens*K`，
旧的较小 `max_rows` 不再截断合法热点。原生接口仍须给 materialized Notify 足够
的元数据容量。完整 Forward X/S 被固定深度物理 ring 替代；tape、compact dR
按本 rank 实际 R，source Backward contribution按本地 T*K 分配。
private Backward workspace仍为 `R+owned_experts*127`，不是精确 P_actual。

20K、EP8/E64/K4全命中rank0时，实际R655,360：完成Backward后的 pool
`compute_bytes` 为rank0 **3,586,689,528**、其余rank **902,339,064**；Notify
另为89,877,032bytes/rank，无GIN注册区。两者差2,684,350,464B正是实际dR和
空rank最小dR的差异。**这些数不含 caller-owned tape、Backward tensors和PyTorch
参考，不是总显存或峰值。** 该热点仅saved-X/G/U/gates/72B headers就需
4,076,339,200B，加output83,886,080B；长文训练的持久数据不会因ring消失。
本步未做匹配配置的训练E2E性能对照，不报告训练加速百分比。

### 验证与冻结

初版`build/static_v4_training_ring_local_m6d/` library SHA256
`946d93e2c81ee486d6e4aa4106fd5a7716bbd428ff446937515a6d9a889e66fc`：
host18/18；EP8 T64均匀、T512热点，B16/D1、完整分配失败/retry/三个live tapes/
后续inference graph/offload通过。随后补上no-drop元数据容量、capture拒绝和
main计时边界（不把host分配停顿计入device_kernel_us），冻结checked版：

`build/static_v4_training_ring_local_m6d_checked/`

- native `fe85a17acc79b7ed841834f48f9440827ad613202bd848a3db1e9e37093d2cdb`
- library `841af316674ff19df33fb3ee589d4ab0ecc14797d76febe0e5d074031230b4ee`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

planner REG80/STACK0/SHARED1024/LOCAL0；训练 main prefill/decode均REG168、
SHARED2048/LOCAL0，STACK72/96。EP16 inference REG168/STACK432/440未变。

checked版已有门禁：

| case（全部单机EP8） | 结果 |
|---|---|
| native T64/E16/K6、B16/D1 | 8 ranks两次Backward repeat_equal=1；32个梯度hash逐字与冻结M6c V3训练路径一致 |
| Torch T64/E64/K4热点、max_rows64、B16/D1 | 五类数值、16个failure、8个retry、24个live-tape复用、8个graph及8个offload标记全部通过 |
| Torch T512热点、max_rows64、B16/D1 | 实际R16384，完整上述检查；rank0/7全部weftmoe kernel的memcheck均0，launcher0 |
| 同一T512，单独Forward racecheck | rank0/7 planner及unified Forward均0 hazards，launcher0；不包含旧Backward racecheck未决项 |
| Torch T1、T8192均匀，max_rows64、B512/D8 | 各8 ranks五类数值及后续inference graph/offload通过 |
| Torch T20480全热点，max_rows64、B512/D8 | R655360，8 ranks五类数值及后续inference graph/offload通过 |

Torch数值沿用现有脚本rtol0.02/atol0.00002；没有新增测试套件。短/长训练正确性
可与CPU构建或另一台机器的独立EP8正确性任务并行，不取其耗时作性能证据。
memcheck报告PID224875/224884，racecheck226064/226073；前缀
`training_ring_local_m6d_*`。Backward全量racecheck的历史工具原子报告仍保留，
不因新Forward通过而称整个训练sanitizer全通过。

### Inference 回归与参数布局消融

固定1620MHz、B512/D8、10 warmup / 20 samples / graph batch20，checked版
正式回归如下（us，slowest-rank p50；每个shape输出hash与冻结DS一致）：

| T/rank | EP8 checked | EP16 checked |
|---:|---:|---:|
| 1 | 101.122 | 279.278 |
| 64 | 134.638 | 334.571 |
| 8192 | 1261.710 | 2949.040 |
| 20480 | 2954.080 | 7024.960 |

checked新增的训练指针最初插在 ComputeRingView / ForwardKernelArgs 中间。
EP16 T20480相邻对照为 checked **7024.960** → DS **6991.060** → checked
**7029.450**，约0.5%回退。将四个训练指针集中到参数尾部的 RingCheckpointView，
保持既有 inference 字段偏移后，isolated版为 **6988.290**，随后DS反测
**6995.330**；范围分别6967.370–7025.190、6958.740–7020.000，视为持平，
**不报告加速**。这是参数布局A/B，不改变训练或inference的数据面协议。
REG/STACK未变，尚不能据此把回退归因到某一条SASS指令。

isolated native位于`build/static_v4_training_ring_local_m6d_isolated/`，SHA256
`f0595ea54e6a4f5c2d2190ae45fb4c465e51897cbb38969b9f97438d0c80c0ad`。
两个相邻正式20K日志为`training_ring_local_m6d_isolated_inference_ep16_t20480_formal.log`
及`training_ring_local_m6d_isolated_inference_ds_ep16_t20480_reverse_formal.log`。
所有正式性能轮次均无CPU构建或其他GPU任务并行；早期EP8的`quick`文件实际为
10 warmup/100 samples/batch2（launcher默认rounds110），不混入上述正式表。

最终版在isolated基础上增加上述immutable Notify入口检查，冻结于
`build/static_v4_training_ring_local_m6d_final/`（目录内`change.patch`）：

- native `1001c146a6145e9718a45b1c78354ac5db0e751c65dd73be493888ac35a450f1`
- library `5e0f53f0b64b15816202c404be154411322bd8eeb0e319deef3d731183a77218`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

最终版host18/18；native T64两次Backward重复一致、32梯度hash逐字等于M6c。
重新完成EP8 Torch T1/64/8192/20480（64和20480热点、max_rows64）；T64与
T512/B16/D1再跑完整OOM/retry/三个live tapes。T512 rank0/7 memcheck零错误
（PID241301/241315）；同case独立Forward planner/main racecheck零hazards
（PID242632/242653）。全部launcher退出0，五类数值、graph/offload标记齐全。
最终native的cuobjdump资源报告与isolated逐字一致。日志前缀
`training_ring_local_m6d_final_`；此处单机测试可在两台机器上分别独立运行，
**不是双机EP8**。

最终版同样在1620MHz、B512/D8、10/20/batch20重新完成正式inference矩阵：

| T/rank | 单机EP8 / us | 双机EP16 / us |
|---:|---:|---:|
| 1 | 101.008 | 277.709 |
| 64 | 134.936 | 333.302 |
| 8192 | 1259.570 | 2953.690 |
| 20480 | 2959.400 | 6985.580 |

八个case全部退出0，hash与DS一致，compute/registered/total pool bytes不变。
日志`training_ring_local_m6d_final_inference_ep{8,16}_t{1,64,8192,20480}_formal.log`。
这是训练功能合入的inference回归，不是新的通信性能收益；EP16 20K相对历史
冻结V3仍只有小幅优势，远未满足长序列显著领先的优化目标。

M6d提交`4c22ddc`后，冻结final native又通过EP8 T64调用方allocator的12轮
Backward（warmup10）：8 ranks均checked_repetitions12，warmup后无cache miss，
32梯度hash仍与M6c逐字一致。日志`training_ring_local_m6d_final_backward_ep8_t64_allocator.log`。
该正确性检查与下一候选CPU构建并行，不作性能比较。

## M6e：多机训练 ring 的 checkpoint / Backward GIN adapter

基于`07856f7`，以下长文正确性及inference回归已完成，不标注完整M6完成。
复用M6d canonical Notify、独立planner、实际tape分配边界及FC1 epilogue；
新增network planner/main实例，main只对节点内发布已stage的X。远端Pull复用
抽出的`checkpoint_copy.cuh`，以RouteLocation的逻辑block/行映射saved-X与
gate/header；一次G2S、两次S2G全部完成，才发布X-full并计入RX consumed。
expected/consumed仍是逻辑route数，不随物理写入次数翻倍。

GIN注册区追加既有有界Backward ChunkRing，不恢复旧完整Forward payload。
首版两套payload不别名，控制状态和NIC counters必须分别持久化；执行slot在
Forward/Backward完成间交替复用，不由live tape占有。counter0/1继续保留：
若每slot静态ring占F个counter、Backward占B个、共有S个slot，则静态slot i用
`[2+i*F, 2+(i+1)*F)`，Backward用`[2+S*F+i*B, 2+S*F+(i+1)*B)`。
两套初始化均执行，之后不在每调用重置。logical GIN contexts仍8；当前训练
devComm申请133个context（实际以backend返回为准），覆盖Forward静态96及既有
Backward需求，操作完成后才复用。
所有channel/peer映射沿用runtime节点数，不加双机专用数据面分支；硬件验证仅2机。

冻结`build/static_v4_training_ring_network_m6e/`，内含change.patch：

- native `3727e90ed001f46a0224b796069a22d6284ac8c21c3592f5e7ecdf650595fb2e`
- library `9cbbe860142249d289b71f3d4cedac1f2fea5ae008be9eb5b393f6ae2bdc6b62`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

network训练planner REG107/STACK72/SHARED1024；训练prefill/decode main均
REG168/STACK448/SHARED2048/LOCAL0；inference仍REG168/STACK432/440。
已通过host18/18；EP16 native T64/E32/K6、B16/D1两次Backward重复一致，64个
梯度hash逐字与M6c训练参考相同。EP16 Torch T64/E64/K4、max_rows64、B16/D1：
五类数值、32个failure、16个retry、48个live-tape复用、16个graph及16个offload
标记齐全，两节点launcher均0。日志前缀`training_ring_network_m6e_`。

EP16 T512全热点、max_rows64、B16/D1也完成完整上述检查，rank0实际R32768。
rank0/15所有weftmoe kernel memcheck均0errors（PID257087/243508），两个
launcher均0；日志`training_ring_network_m6e_torch_ep16_t512_memcheck_{0,1}.log`。
同case的独立Forward racecheck也通过，rank0/15 planner及unified Forward均
0hazards（PID258162/243897），完整脚本及两launcher均0；不包含旧Backward
racecheck原子报告未决项。日志后缀`t512_racecheck_{0,1}.log`。

EP16 T1/8192均匀及T20480全热点、max_rows64、B512/D8均完成五类数值及后续
inference graph/offload（每case各16 ranks、两个launcher均0）。20K热点rank0
接收R1,310,720，完成Backward后pool compute_bytes为 **6,485,332,024**，其他
rank **1,116,627,000**；差额5,368,705,024B等于实际dR减空rank最小4KiB。
GIN为473,820,928B/rank、Notify为90,379,952B/rank。仍不含独立tape、Backward
tensors及PyTorch参考，不是总显存或峰值；该热点saved-X/G/U/gates/headers本身
即需8,152,678,400B。没有以减少pool统计冒充总显存收益。

T512网络入口是decode profile；另补T2048/B16/D1全热点的memcheck及Forward
racecheck，以覆盖prefill宽tile、完整M128的staged epilogue和热点迁移时profile
切换。两次完整脚本均通过：memcheck rank0/15零错误（PID261957/245557），
Forward racecheck零hazards（PID262632/245927）；各有16数值、32failure、
16retry、48live-tape、16graph、16offload标记，两节点launcher均0。

单机兼容另通过native EP8 T64两次Backward（32hash逐字等于M6c）及Torch EP8
T512全热点、B16/D1的完整OOM/retry/live-tape/graph/offload。并发时仅在另一台
机器跑独立EP8正确性，不将这两种单机case写成双机EP8。
正式inference回归也完成：1620MHz、B512/D8、10warmup/20samples/batch20，
测试期间无并发CPU构建/GPU任务；各case输出hash及pool bytes与M6d/DS一致。

| T/rank | 单机EP8 / us | 双机EP16 / us |
|---:|---:|---:|
| 1 | 100.973 | 278.344 |
| 64 | 134.939 | 333.205 |
| 8192 | 1253.330 | 2957.980 |
| 20480 | 2960.660 | 6994.480 |

性能与原inference数据面持平，不把训练功能接入宣称为inference加速。
EP16 20K相对冻结V3仍仅小幅领先，长序列显著提速目标未完成。日志
`training_ring_network_m6e_inference_ep{8,16}_t{1,64,8192,20480}_formal.log`。

## M6f：按精确P分配Backward私有workspace（已验证）

在materialized Notify已有的逐专家计数归约中，同时求
`P=sum_e ceil(rows_e/128)*128`，随原有status一次下载保存到不可变handle；
不下载expert count数组，不增加host同步或二次路由规划。复用现有M128
canonical block定义，Notify自身M64 tile目录不变。

Backward U/dAB/dX/dGate私有存储由`R+owned_experts*127`改为精确P；
P=0时只保留一行供合法TMA描述符使用，设备端实际mblocks仍为0，不生成假任务。
caller提供已准备tensor storage时同时检查padded容量，不能仅凭R相同接受
专家分布不同的较小workspace。fused或replayed Notify没有不可变host P，
在训练tensor准备边界显式拒绝，未新增training Graph支持。

仅四个源文件改动，无Backward GEMM、ring协议、公共API或新增测试套件。
冻结`build/static_v4_training_exact_padding_m6f/`，保存change.patch和resources：

- native `042c09481f82c23ce17ed1b7cb7a1ea15b7f35b102ad32961d14262eb16de41c`
- lib `b69a6f47910a51837b1ce9b586019f54179e87a65996b54826ab7514f2ab7b5e`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

host18/18通过。Native EP8/EP16 T64/E16或E32/K6/B16/D1各两次Backward通过，
共96个梯度hash逐字等于M6e，每rank repeat_equal=1。这个case每rank R384、
实际P512，旧容量638行；Backward tensor bytes由17422328降到16518656，
减少903672bytes/rank（workspace减少126×7172bytes）。输出/权重梯度等其它
tensor容量不变；这是私有workspace节省，不是总训练显存或性能加速。

日志`training_exact_padding_m6f_backward_ep{8,16}_t64.log`。caller allocator的
EP8 T64另做12次Backward/warmup10，8个rank checked_repetitions=12、repeat_equal=1。

Torch EP8/EP16 T1/64/8192 balanced和T20480全热点均通过5项数值参考及后续
inference graph/offload；每case分别8/16个完整pass标记。20K热点仍覆盖
EP8 R655360、EP16 R1310720，未因收紧private workspace丢token。
EP16 T2048/B16/D1、EP8 T512/B16/D1的完整热点迁移脚本也通过：每rank两次
tensor preparation失败注入、Backward tape重试、三个live tapes及后续inference
graph/offload。memcheck覆盖所有weftmoe kernel，EP16 rank0/15为0错误
（PID283822/250241），EP8 rank0/7为0错误（PID290949/290963）。
没有以这些memcheck结果覆盖此前完整Backward racecheck未决报告；本轮未重新
声称训练Graph可用，graph/offload仍是脚本中独立的inference阶段。

日志`training_exact_padding_m6f_torch_ep{8,16}_t*`、
`training_exact_padding_m6f_mem_ep{8_t512,16_t2048_*}.log`以及对应memcheck报告。
非ring回退另用冻结V3生成profile构建，同一源代码的native SHA256
`8a946910f335a4d6a5a39aa13cdde169662c428692548fd9404444042fe72ef3`，冻结于
`build/static_v4_training_exact_padding_m6f_v3/`。取消ring环境变量后EP8/EP16
T64各两次Backward通过，96个hash与M6f ring/M6e完全一致；不是将inference
profile的RDMA_CTAS=0当作原多机训练入口。

1620MHz下inference四规模正式回归（10warmup/20samples/batch20）如下；
没有并发CPU构建或其它GPU任务，各case hash及device/registered容量均与M6e一致。

|T/rank|单机EP8/us|双机EP16/us|
|---:|---:|---:|
|1|100.819|278.118|
|64|134.979|333.563|
|8192|1268.100|2953.070|
|20480|2960.200|6991.990|

EP8 8K比M6e前次1253.330us高约1%，进一步作同机相邻正式对照：
M6e基线1267.940us（1264.370–1273.980），M6f复测1261.500us
（1256.920–1267.270），16个hash及pool容量一致，未见稳定退化，也不宣称加速。
两artifact的本地prefill main SASS完整输出相同（各33916行，含匹配function与
指令，非空dump），SHA256均为
`27131fb300a98572742dc766b0ef1936b0c7c56abbab6f25bb1c035be6838478`。
这支持GPU入口没有改变，但不能单凭机器码相同排除运行环境/地址分配的影响。
日志`training_exact_padding_m6f_inference_ep{8,16}_t*_formal.log`。
8K相邻对照为`training_exact_padding_m6f_inference_ep8_t8192_{baseline,repeat}.log`。
本里程碑完成精确私有容量分配及回归，不等于完整训练性能/racecheck验收完成。
EP16 20K仍仅比历史冻结V3的7096.930us快约1.48%，显著领先的性能目标未达成。

## M6性能基线与Backward profile审计（进行中，2026-09-24）

使用既有native benchmark，无新测试入口。Forward设置
`WEFTMOE_FORWARD_BENCH_RETAIN_STATE=1`并取消graph replay，固定路由及已准备
tensor复用；wall范围为Notify→buffer preparation→Forward完成，包含训练ring的
独立plan、实际行数下载及准备阶段。`device_kernel_us`仅为main kernel，不包括
这些主机边界；V4的`forward/total_us`当前等于main，不能将其直接当成与V3
event区间相同的训练E2E。此处分别列wall/main，不相加或混用span。

1620MHz、H2048/I512/E64/K4、10warmup/20samples、同步rank后每调用单独计时。
每个形状的两版本在同一节点相邻运行：T1/64在第一节点，8K/20K在第二节点。
测量节点无CPU构建或其它GPU任务；部分第二节点轮次与第一节点CPU构建或独立
EP8任务并行，不把它们描述成双机EP16。ring为已提交M6f、B512/D8，参考为冻结
`static_inference_v3_20260923`，非当前源码重编译的“V3”。

|T/rank|M6f wall/us|冻结V3 wall/us|M6f main/us|冻结V3 main/us|
|---:|---:|---:|---:|---:|
|1|404.456|472.547|74.112|198.336|
|64|469.037|551.276|107.040|264.416|
|8192|2017.380|1726.450|1522.820|1335.870|
|20480|4289.020|3577.480|3683.360|3137.500|

32个输出hash两版本完全一致。短序列wall更快，但长序列训练Forward仍退化，
不能以inference的改善代替训练性能门禁。20K pool分别408121664与1820708544
bytes/rank；这不是总训练显存，另有独立tape、Backward tensors等，不能据此
声称总内存减少同样比例。日志`training_perf_{m6f,v3}_{remote_,}ep8_t*_forward.log`。

复用固定tape的Backward（相同形状、1620MHz、10/20、第二节点EP8）device total：

|T/rank|M6f ring profile/us|冻结V3 profile/us|
|---:|---:|---:|
|8192|2387.810|2064.700|
|20480|5706.530|4979.970|

审计预处理宏发现，inference专用ring配置没有显式选择Backward：生成Forward
配置下默认`WEFTMOE_BWD_UNIFIED_WORKERS=0`、`BWD_RELAY_CTAS=32`；冻结V3配置
则为1/16。两者核心GEMM tile/stages相同，但执行路径不同，不能把上述退化
归因于精确分配或checkpoint。M6f同源码的冻结V3-profile构建
`static_v4_training_exact_padding_m6f_v3`，不用ring的20K Backward为4920.540us
（4885.570–4957.470），已接近冻结V3；仍需只切换ring配置的独立验证。
Backward benchmark的`gemm_launches=2`为既有固定audit值，不能据此判定split/
unified入口。已有数值通过不证明两配置性能相同。

日志`training_perf_{m6f,v3}_remote_ep8_t{8192,20480}_backward.log`及
`training_perf_m6f_v3profile_remote_ep8_t20480_backward.log`。计划显式复用
V3 unified Backward并完成矩阵、热点/多tape和sanitizer回归；当前尚未改默认配置，
双机训练性能矩阵也未完成。不将这份基线记录标成M6性能验收完成。

### M6g第一轮：unified开关恢复，但发现sender配额串用了Forward（未采用）

仅在ring生成配置显式设置`BWD_UNIFIED_WORKERS=1`、`BWD_RELAY_CTAS=16`，
没有改内核或生命周期。冻结`build/static_v4_training_unified_backward_m6g/`：

- native `533ede29925c9f1592c6c0d171fbc33540f7f5e59ffdcb7e980e9912c3574ca5`
- lib `2fff031c5a8a8d8d0928533b302bcdd93b3961ab2de5f025f3d281a1c4a09c9e`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

1620MHz、既有cached-tape Backward benchmark、H2048/I512/E64/K4、10/20，
单机在第二节点，双机1×8+1×8；测量节点无CPU构建/其它GPU任务：

|T/rank|EP8 M6f split/us|EP8 M6g unified/us|EP8冻结V3/us|EP16 M6f split/us|EP16 M6g unified/us|
|---:|---:|---:|---:|---:|---:|
|1|344.640|236.256|233.056|1355.550|2455.740|
|64|408.064|368.736|370.752|1392.160|2599.040|
|8192|2387.810|2064.830|2064.700|6661.090|24830.600|
|20480|5706.530|4910.780|4979.970|14571.300|60507.400|

列为device total的rank-max p50，不是Forward+Backward合并训练时间。EP8恢复
V3附近，但不能仅凭单机结果采用：EP16明显退化。进一步查到host launch仍用
`forward_tuning::unified_rdma_ctas`设置Backward sender数；ring Forward配置的
`FWD_UNIFIED_RDMA_CTAS=0`使Backward回退为**1个sender**，而原V3 Forward配额
默认为16。这是第二处执行profile耦合，不是unified算法本身已被证明不适合多机。
下一轮将Backward sender配额独立出来，再与冻结V3和原split路径比较。

第一轮正确性已通过host18/18、native EP8/EP16 T64/E16或E32/K6/B16/D1各两次
Backward，共96个梯度hash逐字等于M6f。Torch EP8/EP16 T1/64/8192均匀、T20480
全热点全部五项数值及脚本中的独立inference graph/offload通过；T64各完整
OOM/retry/三个live-tape检查通过。EP8 T512及EP16 T2048全热点B16/D1也通过
完整脚本，all-weftmoe memcheck rank0/last均0errors：EP8 PID329079/329089，
EP16 PID330657/263385。仍不覆盖旧Backward racecheck共享原子未决报告。

EP8 inference正式10/20/batch20为100.795/135.640/1266.920/2955.740us，
32个输出hash和全部pool bytes等于M6f；计时节点独立，另一节点仅跑EP8正确性。
这些结果只是该候选的回归，不是inference优化。日志前缀
`training_unified_backward_m6g_`、`training_perf_m6g_`、`training_perf_m6f_ep16_`。
双机inference正式回归随后完成，为278.469/332.790/2954.220/6990.330us，
64个hash与pool容量均等于M6f，未见稳定改变。同期冻结V3的EP16 Backward
device total为339.008/701.024/2757.600/6862.430us；T1有一次1641.440us
离群值，p95为347.200us，表述中不隐藏这一波动。原unified候选因sender耦合
未采用，尚未提交配置变更。

### M6g第二轮：接通Backward独立sender配额（已验证）

直接复用已有但此前未使用的`BWD_UNIFIED_RDMA_CTAS`，host按Backward配额划分
sender/receiver，clamp后至少保留一个receiver；所有CTA仍保留GEMM角色。
旧生成计划里的0解释为Backward默认16，不再受Forward的0/mixed/静态模式影响。
ring配置明确选择unified及16个sender。sender completion数组原已按SM数分配，
无需加buffer、改变GIN counter区间或修改设备内核。

首个显式16配置构建`static_v4_training_backward_senders_m6g2/`：native
`20e7a637461320eb381680d301686512963a928ebe35ebcb16531d8060b9cfb1`，
lib`193441a907de044549e2d5ce4f736541c10cd33cbc2e0fee48c135669eabe954`。
EP16 20K诊断测量10/20为6886.300us（6836.060–6995.330），64个梯度hash
逐字等于同期冻结V3；恢复到V3附近，明显优于原split。该轮启动期间还运行了
host配置语法检查，不作为最终无并发构建的正式验收结果。

审计发现通用资源头早已将此兼容字段定义为0，所以不能仅在Backward头加
`#ifndef ... 16`。最终改为显式解释旧值0，并同步调优工具的phase-owned说明。
在现有资源测试中只增加一个针对Forward=0串用及Backward缺省/0/显式值的检查；
修改前确实失败，兼容处理后phase资源15/15、参数验证10/10通过。
较大的搜索空间测试有4 failures/2 errors；换回HEAD的原`unified_tuning.py`
仍产生相同失败，全部是Forward候选规模/旧staging容量断言，本轮未扩大范围
修改这些既有问题。日志`training_backward_senders_m6g2_{macro_before,
phase_resources,tuning_tests,search_space_head}.log`。

最终冻结`build/static_v4_training_backward_senders_m6g2_final/`：

- native `cc30bf44040fa48efed7bab0a9545bb14525474e538f47b4bb0b1e4f7e77b683`
- lib `ade15d7cdf2d1ee7a2ad9a9b5f83d5148f087678a9ca9f7595b5e752bf07c73d`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

两个unified Backward kernel的SASS与第一轮M6g完全相同，组合dump共82689行，
确认包含local/network两个function而不是空dump，SHA256均为
`a8a044ff6c43deaaacfacb23672ad23856c879c474659becafe256b5085eb51c`。
这支持收益来自host角色配额修正，不是GEMM或数值优化；不能代替运行时验收。
EP8/EP16 native T64/K6/B16/D1已通过，两次Backward稳定，96个梯度hash等于M6g/M6f。
EP8四规模inference正式回归为100.688/135.434/1266.750/2950.030us，32hash与
全部pool bytes等于M6f。

EP16 Torch T1/64/8192均匀及T20480全热点，五项数值及后续独立inference
graph/offload全部通过；T64有32 failure、16 retry、48 live-tape标记。
EP16 T2048、EP8 T512全热点B16/D1完整脚本通过，all-weftmoe memcheck的
rank0/last均0errors：EP16 PID350801/270961，EP8 PID352107/352117。
EP8主矩阵此前第一轮M6g已完整验证，本轮本地Backward设备代码不变，另用
native重复调用、小ring热点/迁移/重试及下面的正式矩阵回归。

最终1620MHz、cached-tape Backward、10warmup/20samples的device total如下；
EP8在第二节点独立测量，部分轮次与第一节点构建/EP8正确性并行；测量节点无
CPU构建或其它GPU任务。EP16计时期间两节点均无并行构建/其它GPU任务。

|T/rank|EP8修正后/us|EP8冻结V3/us|EP16修正后/us|EP16冻结V3/us|
|---:|---:|---:|---:|---:|
|1|234.656|233.056|343.648|339.008|
|64|369.184|370.752|696.640|701.024|
|8192|2063.870|2064.700|2771.040|2757.600|
|20480|4917.220|4979.970|6873.570|6862.430|

EP16四形状共256个梯度hash逐字等于同期冻结V3。20K相对M6f split的EP8
5706.530us、EP16 14571.300us分别减少约13.8%/52.8%；主要是恢复应有profile，
不能宣称新Backward算法显著领先V3，也不能将独立Forward/Backward中位数相加
冒充实测整步训练时间。日志`training_perf_m6g2_{remote_ep8,ep16}_t*_backward.log`。

最终inference正式10/20/batch20如下，96个hash及全部pool bytes等于M6f：

|T/rank|单机EP8/us|双机EP16/us|
|---:|---:|---:|
|1|100.688|277.781|
|64|135.434|333.541|
|8192|1266.750|2956.430|
|20480|2950.030|6996.060|

无稳定inference改变；日志`training_backward_senders_m6g2_inference_*_formal.log`。
本里程碑采用Backward profile/sender解耦修正，不关闭训练Forward长文退化、旧
Backward racecheck未决报告或EP16长序列inference显著领先V3的剩余目标。

## M6h：复用FC1 epilogue staging合并训练checkpoint写回（已验证）

宽入口的完整M128 tile原已在退休的CTA私有B staging中重排S。本轮同时暂存
BF16 G/U，复用相同的两次math CTA barrier，再以16B vector写回canonical
checkpoint行；替代原按accumulator fragment分散的4B stores。G/U仍是分离的
canonical平面，保留原BF16舍入，不改Backward布局、ring credit或saved X。
不复用cluster multicast A。只有三个输出平面能放入实际B arena时启用；
窄decode、尾块及scratch不足的配置继续direct checkpoint/epilogue。

共享helper接收调用方实际scratch容量，并推导Args类型；这也修复纯GEMM工具
在M6d接口演进后缺mblock实参、精简Args被强制要求训练字段的编译问题。
诊断工具仍调用生产epilogue，未复制一套实现或添加假的checkpoint字段。
N64/stages2/K128的packed及separated、T1/8192完整chain均精确等于cuBLAS且
repeat逐字相同；N32/stages3/K128 packed T1也通过。N32 separated是工具
既有不支持组合，尝试运行返回unsupported，未计作通过项，也不放宽其合同。
这些是工具接口/计算回归，不能作为本轮checkpoint收益或重新选布局的证据。

最终冻结`build/static_v4_training_staged_checkpoint_m6h_final/`：

- native `5b3e9264afc3cdff75678b45c6ab304b80033f06744421f53ae86091ecc3ee62`
- lib `16e81efa4a96757e77f7f55f697a2a3c0361f05a1ea6d6b96bced0decffc3073`
- extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`

训练prefill local/network仍REG168、STACK72/448、SHARED2048、LOCAL0；
dynamic SMEM及pool容量不增加。首版候选`static_v4_training_staged_checkpoint_m6h/`
仅供初筛，正式对照使用上述修复诊断接口后的最终构建。

1620MHz、EP8（第二节点独立测量）、E64/K4/H2048/I512、B512/D8、保留tape，
10warmup/20samples，与已提交M6g2按G/H/H/G顺序交错。第一节点同时做EP8
正确性检查/纯GEMM CPU构建，测量节点无其它GPU任务或CPU构建。

|T/rank|M6g2 main/us，两轮|M6h main/us，两轮|M6g2 wall/us，两轮|M6h wall/us，两轮|
|---:|---:|---:|---:|---:|
|8192|1518.820 / 1529.380|1511.010 / 1512.930|2021.520 / 2040.070|2017.200 / 2011.230|
|20480|3698.270 / 3693.220|3638.850 / 3638.020|4305.780 / 4276.040|4220.910 / 4217.040|

20K main约降低1.5%–1.6%，8K约0.4%–1.2%，小幅收益，不足以恢复长文相对V3
的训练Forward差距。wall包含Notify/host准备/Forward，不与V3的不同阶段span
混比；更不把独立Forward/Backward中位数相加当作整步训练实测。
日志`training_perf_{m6g2,m6h,m6h_repeat,m6g2_repeat}_adjacent_remote_ep8_t*_forward.log`。

最终native EP8 T512/E16/K6/B512/D8两次Backward稳定，32个原始梯度hash逐字
等于M6g2，覆盖完整M128 staging。Torch EP8 T8192均匀、T20480全热点五项
数值及独立inference graph/offload通过。EP8 T512全热点B16/D1完整脚本
通过（OOM/retry/三个live tapes），rank0/7 all-weftmoe memcheck 0errors，
PID362837/362854。

EP16 T1/64/8192均匀、T20480全热点，五项数值及独立inference graph/offload
均通过；T64完整检查含32failure/16retry/48live-tape标记。EP16 T2048全热点
B16/D1也通过完整脚本；rank0/15 all-weftmoe memcheck 0errors（PID367822/
277218），独立Forward planner/main racecheck 0hazards（PID369184/277605）。
全部launcher exit0，不将Forward专项结果外推到旧Backward racecheck。
日志前缀`training_staged_checkpoint_m6h_torch_`及`..._{memcheck,racecheck}_`。
EP16训练Forward同配置、10/20、无并发构建/其它GPU任务的相邻对照如下。
两规模共32个Forward输出hash从原始rank记录按字符串逐字核对，相等；不能
用JavaScript浮点解析后的64-bit hash做精确一致性证据。

|T/rank|M6g2 main/us|M6h main/us|M6g2 wall/us|M6h wall/us|
|---:|---:|---:|---:|---:|
|8192|3518.050|3488.960|4319.950|4296.780|
|20480|8373.980|8318.430|9327.650|9281.240|

双机main约降低0.8%/0.7%，不夸大为显著端到端收益。GIN launcher中的
`forward_slowest_rank_us`在本ring构建对应main，native记录的`device_kernel`
与`forward`逐样本相同；wall仍按Notify→buffer→Forward完整范围报告。
日志`training_perf_{m6g2,m6h}_ep16_t*_forward.log`。
EP8 T1/64也完成Torch完整数值/graph/offload，T64含16failure/8retry/24live-tape
标记。EP8 T512全热点B16/D1的Forward racecheck rank0/7均0hazards，
PID377953/377968，完整脚本及launcher exit0。

最终1620MHz、10/20/batch20的inference正式回归：

|T/rank|单机EP8/us|双机EP16/us|
|---:|---:|---:|
|1|100.725|277.926|
|64|135.403|333.410|
|8192|1271.100|2956.020|
|20480|2964.940|6985.870|

96个输出hash逐字等于M6g2。EP8在第二节点独立测量，第一节点有候选CPU构建/
EP8正确性工作；EP16测量无并发构建/GPU任务。日志
`training_staged_checkpoint_m6h_inference_{remote_ep8,ep16}_t*_formal.log`。
本项是小幅训练checkpoint优化，不将这些inference波动归为收益；20K相对V3
显著领先的目标、训练Forward长文差距及旧Backward racecheck未决项仍未关闭。

随后EP8反向复测M6g2：8K/20K为1262.080/2953.510us，16个hash与M6h一致，
仍有约0.7%/0.4%进程间差异。核对最终native中的local/network两个inference
prefill kernel，SASS逐字相同（56282行、确认包含两个function），共同SHA256
`af9efeb01afc1e754002d74ea42f7e57a4e653563541d16eb49e782630df4c70`；dump含
函数名及指令，不含不同路径/调试行号。故不归因为本次训练epilogue修改，也不
选择较快一轮替换正式矩阵。日志后缀`m6g2_inference_remote_ep8_t*_reverse_formal.log`。

## EO：训练X/SavedX双lane写回（退化，撤回）

在已提交M6h的N128及原调度上，复用既有
`FanoutRowPrefetchPipeline<GroupLanes=2>`，每个Pull warp仍只发一次G2S，
lane0写ring X、lane1写SavedX。共享两个staging slot及一个loader，复用现有
subgroup read-completion join；不增加staging内存或新的计数协议。

本地Pull的两个lane执行相同不可变行枚举，gate/identity及X credit只由lane0
写；两lane各自full completion后再join，才发布X-ready。远端Pull同样使用
两lane copy/drain，再经过原full-warp join发布X-ready和RX consumed。inference
仍编译为单lane。没有改checkpoint布局、route expected、slot generation或
双NIC credit。这是提交开销消融，尚无正确性/性能结论；不混入EM/EN的N256/
lookahead变化。冻结目录`build/static_v4_training_dual_lane_checkpoint_eo/`。

native SHA256 `1ce55d2019cf9a777104a5ed48ecfc54530b406e97c8d8cf3cc7b969e3ef09fa`。
训练prefill local/network为REG168、STACK80/456（均比M6h增加8bytes）、
SHARED2048、LOCAL0。EP8 T512/E16/K6的B512/D8和B16/D1各两次Backward稳定，
共64个原始梯度hash逐字等于M6h；小ring覆盖多次复用，不代替完整数值/sanitizer。

1620MHz、B512/D8、10warmup/20samples，保留tape的训练Forward如下：

|拓扑|T/rank|EO main/us|M6h main/us|EO wall/us|M6h wall/us|
|---|---:|---:|---:|---:|---:|
|单机EP8|8192|1559.650|1501.920|2055.920|2022.320|
|单机EP8|20480|3772.450|3629.250|4354.150|4205.590|
|双机EP16|8192|3671.780|3488.960|4489.430|4296.780|
|双机EP16|20480|8842.560|8318.430|9809.580|9281.240|

EP8为第二节点独立测量，第一节点运行native正确性；EP16两节点无并发构建/
其它GPU工作。M6h参照来自本日EM相邻EP8及M6h正式EP16，不冒充本轮交错复测。
两拓扑均退化，三个源文件全部撤回，不继续library/完整Torch/sanitizer。
日志`training_dual_lane_checkpoint_eo_{native_ep8,remote_ep8,ep16}_*`。
两拓扑的48个Forward输出hash也逐字与各自M6h参照一致；EP16从原始rank记录
提取字符串核对，没有使用JSON中已丢精度的64-bit数值。

## EP：FC1大尾块复用合并写回（仅单机K1小收益，暂不采用）

在M6h上将现有staging条件从完整M128放宽到至少64行，vector store只枚举
有效行；较小尾块及窄decode保持direct。不改变G/U布局、scratch大小、barrier
个数或数值合同。冻结目录`build/static_v4_fc1_large_tail_staging_ep/`。

主矩阵E64/K4/G8/B512的均匀路由按64周期整齐分布，每窗口每expert为256行，
因此8K/20K没有该类尾块，此项不能解释或恢复主矩阵长文差距。独立用既有K1
基准产生每expert每窗64行，保持T/B不变做尾块消融；不拿K1时延代替K4主矩阵。
最初K3诊断调用被既有launcher拒绝（exit2，未测量），未修改其支持合同。

native SHA256 `2cbe82b693136d650e9e587e8eb0897ed2c7b7c9a7f9c149bc0b6f300b82f5e7`。
prefill训练/推理的local/network资源与M6h相同。EP8 T512/E64/K1/B512/D8
恰好覆盖M64尾块，两次Backward稳定，32个原始梯度hash逐字等于相邻M6h。
第二节点独立EP8、1620MHz、10warmup/20samples的K1训练对照：

|T/rank|M6h main/us|EP main/us|M6h wall/us|EP wall/us|
|---:|---:|---:|---:|---:|
|8192|837.664|833.952|1391.920|1388.350|
|20480|2004.220|1990.780|2569.840|2550.300|
|20480（反向）|2005.380|1989.540|2566.560|2552.200|

四个候选/基线对照共32个输出hash一致。20K主kernel约改善0.7%–0.8%，是
K1尾块诊断的小收益；第一节点同时有候选CPU构建，计时节点无其它任务。
没有据此合入，源码已恢复M6h以隔离下一项配额实验；双机及完整门禁待执行。
日志`fc1_large_tail_staging_{m6h,ep}_{native_ep8,remote_ep8}_*`。

EP16同K1/B512/D8、1620MHz、10/20、无并发构建或其它GPU任务的相邻对照：

|T/rank|M6h main/us|EP main/us|M6h wall/us|EP wall/us|
|---:|---:|---:|---:|---:|
|8192|1658.400|1649.410|2552.850|2529.620|
|20480|3866.020|3863.420|4738.720|4737.630|

32个原始字符串hash逐字一致，20K基本持平。此项未改善K4主矩阵，仅有单机K1
约0.7%的小收益，不优先保留或新增拓扑分支。源改动已撤回，保留冻结候选和
原始测量，未继续library/完整Torch/sanitizer。日志后缀`ep16_k1_t*_training_forward.log`。

## ES：训练共享网络本地完成prefix（数值通过，性能持平）

Forward的逐slot远端credit不变，本地完成改为唯一TX owner执行GIN flush并推进
channel已完成prefix。删除Forward未再使用的counter编号/分配/reset，Backward
仍有独立逐slot counter，不因多个live tapes而共用控制状态。完整协议、artifact
SHA及inference收益见[ring记录ES](static_forward_v4_ring_record.md)。

EP8/EP16 Torch四规模（20K热点）、1/64 OOM与重试、三份live tape/payload复用
通过；native EP8/EP16共96个梯度hash等于M6h/M6g2。EP16热点B16/D1的rank0/15
memcheck及Forward-filtered racecheck均0。未修改或宣称解决前文Backward
shared-atomic racecheck工具限制；附带graph检查是inference，不是训练graph。

1620MHz、B512/D8、10warmup/20samples，EP16保留tape的Forward：

|T/rank/顺序|M6h main/us|ES main/us|M6h wall/us|ES wall/us|
|---|---:|---:|---:|---:|
|8192，相邻|3480.990|3467.460|4283.240|4263.740|
|20480，相邻|8251.330|8306.850|9221.820|9268.270|
|20480，反向|8318.400|8272.610|9280.920|9222.650|

20K差异方向反转，属于本次测量波动，不报告训练收益。main与wall保持独立口径；
六份原始rank记录共96个Forward hash一致。计时节点无并发构建/GPU任务。
日志`batched_local_completion_{m6h,es_final}_ep16_t*_training_forward*.log`。
采用ES的理由是保持双credit合同并简化实现、改善inference，不是恢复了训练
Forward相对V3的差距；该项性能目标仍未完成。

### ET：Return Q16对训练Forward的回归

在已验收ES上仅将profile Return Qmax32降至16，保持Dispatch Q8/L32/B512/D8。
EP16四档训练与20K热点通过，短序列包含分配失败/重试及三份live tapes复用；
rank0/15热点B16/D1的memcheck为0，限定Forward/planner的racecheck为0。
没有重测或声称完整Backward racecheck clean。native EP8/EP16两次Backward
共96个梯度hash等于ES；EP8计算指令完全不变，复用ES完整Torch矩阵。

1620MHz、无并发构建/GPU任务、10warmup/20samples的eager retained-tape对照：

|T/rank|ES main/us|ET main/us|ES wall/us|ET wall/us|
|---:|---:|---:|---:|---:|
|8192|3461.790|3445.340|4282.190|4250.800|
|20480|8299.040|8300.320|9271.490|9263.080|

64个原始output_hash字符串一致。训练持平，不把小幅差异报告为收益；采用Q16
是因inference稳定小幅改善及缓冲区减少。主计算、含Notify/buffer的wall和
inference graph口径不同。日志`return_followup_{es,et}_ep16_t*_training_forward.log`。

### FB：source-major物理Mblock行序（无收益，撤回）

在ET上重测expert内source连续排布；canonical Notify行号和保存接口不变，
无额外padding。除inference快筛外，因checkpoint写回局部性可能变化，单独
测了保留tape的训练Forward。1620MHz、无并发构建/GPU任务、10/20：

|T/rank|ET main/us|FB main/us|ET wall/us|FB wall/us|
|---:|---:|---:|---:|---:|
|8192|3465.760|3474.590|4258.360|4275.310|
|20480|8328.380|8442.340|9296.990|9414.720|

64个Forward原始hash相同；20K main约退化1.37%，不采用训练专用行序。
只完成GPU几何检查/racecheck和上述Forward检查，没有继续完整tape/Backward
门禁，不能据此声明该候选的梯度已经验收。改动与测试参考已撤回，复用ET布局。
artifact与inference消融见[ring记录FB/FC](static_forward_v4_ring_record.md)。

### FJ2：三级FC1及跨GEMM共享内存退休（已验证，训练无收益）

原FJ的空间复用漏掉cluster两CTA阶段偏斜，prefill graph在racecheck下出现
数值错误，未采用。FJ2把FC2 scratch移到两种A extent之后，并在GEMM切换时
用已有empty barrier等待两个CTA的旧操作数读完；不增加SMEM或计数器。
原失败用例已通过，完整验证及机制见[ring记录FJ/FJ2](static_forward_v4_ring_record.md)。

EP8和EP16 native重复Backward的96个梯度hash与ES/M6h一致；两种拓扑的
1/64/8K/20K Torch训练矩阵已通过，20K为hotspot，短档含OOM/retry/live tapes。
FJ2双机decode热点、prefill热点训练memcheck均0，Forward专项racecheck两轮
通过，非ring长文也已回归；双机性能仍在测量，不声称完整Backward racecheck clean。

1620MHz、10warmup/20samples、eager retained-tape，第二节点单机EP8，
第一节点同时做Torch数值检查，计时节点独占且无编译：

|T/rank|ET main/us|FJ2 main/us|ET wall/us|FJ2 wall/us|
|---:|---:|---:|---:|---:|
|8192|1508.160|1502.460|2006.450|1999.240|
|20480|3635.360|3641.820|4217.450|4237.720|

32个字符串Forward output hash一致。两档差异小且方向不一，只判断训练
持平，不把inference的收益算到训练。日志
`prefill_fc1_stage3_v2_final_{et,fj2}_remote_ep8_t*_training_forward.log`。

双机独占、同频、无并发构建，正式对照补齐：

|T/rank/顺序|ET main/us|FJ2 main/us|ET wall/us|FJ2 wall/us|
|---|---:|---:|---:|---:|
|8192，ET→FJ2|3451.040|3452.350|4259.480|4262.680|
|20480，ET→FJ2|8282.620|8341.660|9226.900|9315.530|
|20480，FJ2→ET|8316.960|8340.100|9263.080|9343.920|

三组各32个原始hash字符串一致。8K持平，20K main两轮均回退（0.71%/0.28%），
wall也均回退（0.96%/0.87%），不能把这次stage调整当作训练优化。
暂不整体采用FJ2；FJ3将训练prefill FC1恢复两级，保留跨GEMM同步修正，
与inference三级配置独立消融。日志
`prefill_fc1_stage3_v2_{final,reverse}_{et,fj2}_ep16_t*_training_forward.log`。

FJ3只恢复训练prefill两级FC1，8K main ET3454.110/FJ33453.410us，20K
ET8319.460/FJ38358.400us；wall分别4266.660/4252.770及9279.800/9316.950us。
仍为1620MHz、独占10/20，各组32原始hash相同。20K仍慢0.47%，不保留这个
未带来收益的额外训练配置；未对FJ3执行完整梯度/sanitizer验收。
最终采用完整验收的FJ2同步/scratch修正及统一prefill三级FC1。此里程碑补全
跨CTA内存复用安全性，不是训练性能里程碑；约0.3%–0.7%的20K main回退及
训练Forward相对V3的总体差距均保持开放，不能被inference改善抵销。

### FM：从FC1的A stage保存X checkpoint（已验证，采用）

FJ2为基线。Pull不再向SavedX做第二次S2G，只写物理X及便宜的canonical
identity/gates。每个Mblock的FC1 Ntile0，从已ready的A stage以uint4 vector
scatter写canonical SavedX；复用原checkpoint.rows，不改物理行序、Notify、
G/U保存、tape格式或Backward。写入位于WGMMA commit之后、wait0和empty
arrival之前，沿用V3尾块的读者退休顺序。SavedX不复用ring payload，Forward
完成才允许Backward消费。不增加分配、barrier或计数器，复制总字节数不变。

checkpoint_copy模块改为计算侧stage helper；Pull复用已有普通row pipeline，
删除训练fanout wrapper。这个候选是vector scatter，不是V3完整tile的TMA
store，也不预设它一定快于通信warp逐行复制。

冻结`build/static_v4_math_x_checkpoint_fm/`，native SHA256
`482f535e85ea00ad89cb859b47345f092f9582309d34bd74331fc2d1b8816120`。
EP8 T512/K6/B512/D8与EP16 T64/K6/B16/D1重复Backward通过，共96个梯度
hash与FJ2完全相同。network prefill training STACK416（FJ2408），其它
prefill入口的REG/STACK/SHARED/LOCAL未变。尚未做完整Torch/sanitizer门禁。

全部构建结束后，第二节点独占EP8，1620MHz、eager retained-tape、10warmup/
20samples。首轮相邻对照如下，双机与反向复测仍待完成：

|T/rank|FJ2 main/us|FM main/us|FJ2 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|1505.120|1504.190|2012.400|2014.050|
|20480|3650.270|3547.900|4234.980|4120.930|

8K持平，20K main首轮改善2.80%、wall2.69%，仍不能凭单轮关闭训练性能目标。
日志`math_x_checkpoint_{fj2,fm}_remote_ep8_t*_training_forward.log`。

首轮EP8共32个字符串Forward hash一致。随后双机独占、无CPU构建/GPU任务
并发、同样1620MHz和10/20，完成相邻对照：

|T/rank|FJ2 main/us|FM main/us|FJ2 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|3457.920|3361.410|4254.760|4170.490|
|20480|8351.970|8051.840|9340.540|9031.110|

main分别改善2.79%/3.59%，wall1.98%/3.31%，64个原始Forward hash字符串
相同。日志`math_x_checkpoint_{fj2,fm}_ep16_t*_training_forward.log`。
尚需反向复测及完整训练门禁。FN会仅对SavedX尝试streaming store缓存提示，
保持所有地址、所有权、复制字节数及同步不变，独立比较，不预先认定收益。

单机20K反向FM→FJ2：main3547.650→3652.830us，wall4130.860→4244.480us，
对应改善2.88%/2.68%，16个字符串Forward hash一致，与首轮方向一致。
第二节点独占计时，第一节点同时编译FN，没有计时节点CPU构建；日志
`math_x_checkpoint_reverse_{fm,fj2}_remote_ep8_t20480_training_forward.log`。

### FN：SavedX streaming store（额外收益不稳定，撤回）

在FM上仅将SavedX vector store换成`st.global.cs.v4.u32`，尝试减少checkpoint
写流对活跃权重的缓存干扰。它只是eviction hint，不改变可见性或退休协议。
参见[NVIDIA PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#cache-operators)。
G/U、S、metadata、权重TMA及行序均不改。冻结
`build/static_v4_streaming_x_checkpoint_fn/`，native SHA256
`caf9083a0f867e417aa84d3fa9765ffa9ff910563d6b03a52844ba25517164d8`。
EP8/EP16重复Backward共96梯度hash与FJ2相同，四个prefill入口资源与FM相同。

1620MHz、无并发构建/GPU任务、eager retained-tape 10/20，相邻FM→FN：

|拓扑/T/rank|FM main/us|FN main/us|FM wall/us|FN wall/us|
|---|---:|---:|---:|---:|
|EP8/8192|1501.920|1503.330|1996.440|2006.580|
|EP8/20480|3546.780|3542.980|4129.840|4124.360|
|EP16/8192|3383.010|3351.140|4188.540|4156.910|
|EP16/20480|8064.060|8080.000|9007.420|9058.950|

EP8共32个字符串hash、EP16共64个原始uint64 hash字符串相同。单机基本持平，
双机8K首轮约改善0.94%，20K却回退约0.20%，不保留缺乏稳定额外收益的
特殊PTX store。FN未进行library/Torch/sanitizer完整门禁。源码恢复普通store的
FM后，patch与冻结FM逐字节相同；FM进入完整门禁，尚未提前提交为验收优化。
日志`streaming_x_checkpoint_{fm,fn}_{remote_ep8,ep16}_t*_training_forward.log`。

双机20K在FM→FN之后又测FJ2（间隔一个FN批次，不称FM/FJ2严格相邻）：
FJ2 main8324.990us、wall9299.890us，对照FM8064.060/9007.420us，改善
3.13%/3.14%。32个原始hash相同，与前一轮FJ2→FM的3.59% main改善方向一致。
日志`streaming_x_checkpoint_fj2_ep16_t20480_training_forward.log`。

### FM最终验收与同期V3对照

FM library SHA256 `44820160cb808ed7aaff376b244214ea989f48ab91438444a7784b842450892a`，
extension仍为`725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
单机完整训练四档通过：1/64每档8 parity、16 preparation-failure、8 retry、
24 live-tape reuse、8 inference graph及8 offload；8K balanced/20K hotspot
每档8 parity/graph/offload。Inference四档+edges 240条、tiny B16/D1 K8 152条、
非ring8K/20K 32条检查通过。双机完整门禁仍在进行，未提前声称验收。

同轮补测冻结V3，1620MHz、remote单机EP8独占、eager retained-tape 10/20：

|T/rank|V3 main/us|FM main/us|V3 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|1334.270|1497.950|1732.580|1995.680|
|20480|3138.820|3537.540|3578.370|4118.170|

32个Forward hash一致。FM相对V3 main仍慢12.27%/12.70%，wall慢15.19%/15.09%；
checkpoint迁移只缩小差距，不代表恢复了训练性能。计时节点独占，另一节点
同时构建FM库；没有计时节点CPU构建。日志
`math_x_checkpoint_final_{v3,fm}_remote_ep8_t*_training_forward.log`。

双机完整门禁于20:13 UTC完成，launcher退出0。训练1/64每档16 parity、
32 preparation-failure、16 retry、48 live-tape reuse、16 inference graph及
16 offload；8K balanced/20K hotspot每档16 parity/graph/offload。
Inference四档+edges共480条，tiny B16/D1 K8共304条通过。复用现有测试，
没有为本次地址迁移增加测试脚本。

rank0/15小ring T64热点训练memcheck（PID87941/359880）、T2048 prefill热点
训练memcheck（92771/360845）均0 errors；对应Forward racecheck
（90281/360269及94753/361236）均0 hazards/0 warnings，应用parity/graph/
offload检查全部完成。racecheck只过滤planner/Forward，不宣称完整Backward
raceclean。日志`math_x_checkpoint_fm_torch_ep16_*`及同前缀sanitizer日志。
双机正式性能矩阵在完整门禁结束、两节点GPU空闲后启动，无并发构建。

双机同轮冻结V3对照完成，1620MHz、独占、eager retained-tape 10/20：

|T/rank|V3 main/us|FM main/us|V3 wall/us|FM wall/us|
|---:|---:|---:|---:|---:|
|8192|2912.740|3366.690|3423.030|4175.610|
|20480|6648.130|8063.940|7177.980|9033.290|

64个原始uint64 Forward hash字符串一致，实际shape及16个rank的retained
标志均核实。FM相对V3 main仍慢15.58%/21.30%，wall慢21.99%/25.85%，不因
相对FJ2的小幅改善关闭训练性能目标。日志
`math_x_checkpoint_final_{v3,fm}_ep16_t*_training_forward.log`。
20K FM这一轮main范围7999.840–43197.100us，p95为8148.160us，mean9821.516us；
有一个43ms长尾，未删除样本、未只报告较好轮次。另行重测FM确认稳定值。

随后的同条件FM独占重测，20K main p50=8023.330us，范围7967.300–8118.820us，
p95=8104.800us；wall p50=9006.470us，16个原始hash与首轮相同。没有再次出现
43ms长尾，但无法仅凭复测确定其原因，首轮及全部样本继续保留；不以这轮较小
数值替换上表。日志`math_x_checkpoint_repeat_fm_ep16_t20480_training_forward.log`。

最终采用通过完整门禁的普通store版FM，源码patch与冻结artifact逐字节一致。
优化的是训练SavedX复制的执行位置，没有inference新增收益；正式inference
四档回归及对应V3数据见ring记录FM。后续将地址计算移出K循环的FO是独立候选，
不计入本里程碑，完整训练性能及Backward专项工具限制仍保持开放。

### FO：FC1 SavedX目的地址移出K循环（已采用）

独立基于FM。检查FM的local training prefill SASS发现，FC1每个K迭代重复
读取8个checkpoint row及计算64位目的地址。将其改为Ntile0每task准备一次
线程私有目的指针，仍从相同ready A stage保存相同uint4；固定vector数并
显式约束math thread索引。保留valid-row/hidden-tail检查，所有同步、行序、
布局与复制字节数不变。Inference的未使用对象应由编译器消除，待检查资源及
生成指令。冻结`build/static_v4_checkpoint_address_fo/`，尚未报告收益。

native SHA256 `fe87231b00287792081fa38a457048fb9e99a4350bc2333cb564b620e1d52795`。
EP8 T512/K6/B512/D8及EP16 T64/K6/B16/D1重复Backward共96梯度hash与FM相同。
network prefill training STACK416→408，local training72、inference local64/
network392不变，REG168/SHARED2048/LOCAL0不变；未新增Forward WGMMA序列化
警告。SASS确认K循环内不再加载checkpoint rows及计算row-stride乘积，8个
目的指针在寄存器而非local数组中；仍保留原thread-index的单次stack reload。

全部构建结束后独占计时、1620MHz、B512/D8、retained-tape 10/20，FM→FO：

|拓扑/T/rank|FM main/us|FO main/us|FM wall/us|FO wall/us|
|---|---:|---:|---:|---:|
|EP8/8192|1499.360|1432.100|2008.280|1932.100|
|EP8/20480|3546.400|3446.400|4125.110|4022.110|
|EP16/8192|3370.500|3319.740|4180.620|4114.800|
|EP16/20480|8046.530|7934.210|9026.540|8888.460|

EP8 main改善4.49%/2.82%，EP16改善1.51%/1.40%；wall也改善。EP8的32个字符串
hash、EP16的64个原始uint64 hash字符串一致，pool不变。尚需反向复测及完整
Torch/sanitizer门禁，不提前采用。日志
`checkpoint_address_{fm,fo}_{remote_ep8,ep16}_t*_training_forward.log`。

后续正反向、完整Torch/Forward专项sanitizer及四档inference回归已通过，
采用这项不增加内存/协议状态的改动。细节和后续FP容量实验交叉对照见下文。
收益不是所有规模固定百分比：双机20K反向仅0.47%、wall持平，保留该结果。
未改变默认B/D或加入训练自动阈值；inference SASS与FM相同。

### FP：训练窗口容量独立调节（验证中，不改变inference配置）

用冻结FM，无源码修改。第二节点单机EP8独占，T20K、1620MHz、retained-tape
10/20；第一节点同时构建FO，没有计时节点CPU构建。按下表顺序测试：

|B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|512/8|3550.530|4152.580|408121664|
|1024/4|3381.500|3984.130|575356480|
|2048/2|3850.980|4495.860|910853696|
|512/8复测|3557.150|4153.520|408121664|

32个Forward hash字符串一致，actual shape及retained flags已核实。B×D相同
不代表总pool相同：更大的整窗口需要更多compute payload容量。B1024/D4
main首轮约改善4.76%，但多167234816 bytes/rank；B2048退化8.46%。仍需
补测8K、分离B/D影响及双机，不先保留训练专用配置，也不把容量调节收益算作
FO地址计算优化。日志`training_window_fp_remote_ep8_b*_d*_r*.log`。

同一remote独占/本地FO编译条件下，补测8K并分离20K的B/D影响，按表中顺序：

|T/rank，B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|8192，1024/4|1453.020|2002.630|481305408|
|8192，512/8|1502.980|1999.050|313659456|
|20480，1024/8|3392.900|3992.890|642596672|
|20480，512/4|3602.690|4190.660|374501440|
|20480，1024/4复测|3383.870|3981.300|575356480|

40个字符串hash与FM对应规模相同。8K main约快3.32%，但wall持平，不称端到端
改善；20K B1024/D4复测与首轮方向一致，增加D8无额外收益。B512/D4反而慢于
B512/D8，不能把B和D视作可互换的相同容量。尚未对B1024训练做完整Torch或
双机性能门禁，不改变验收profile；所有计时来自冻结FM，与FO代码实验独立。
日志`training_window_fp_followup_remote_ep8_t*_b*_d*.log`。

随后两节点独占、无构建/GPU并发，1620MHz同样10/20；每个规模先测FO B512/D8
作反向对照，再测FM B512/D8及FM B1024/D4：

|T/rank，版本/B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|8192，FO/512/8|3290.110|4116.140|702615744|
|8192，FM/512/8|3374.500|4189.520|702615744|
|8192，FM/1024/4|3034.530|3872.850|870256832|
|20480，FO/512/8|7966.910|8948.170|819971520|
|20480，FM/512/8|8004.640|8958.050|819971520|
|20480，FM/1024/4|7451.260|8437.640|987170496|

各规模48个原始hash相同。FO反向main改善2.50%/0.47%，20K比首轮1.40%的
改善小，wall几乎持平，保留波动而不只选最好轮次。FM的窗口调节main改善
10.07%/6.91%，wall约7.56%/5.81%，但每rank额外约168/167 MB；仍需完整
B1024训练正确性，不能视为免费优化。日志
`checkpoint_address_reverse_fo_ep16_t*_training_forward.log`及
`training_window_fp_ep16_t*_b*_d*.log`。

FO进入library构建时，在另一节点独占补测组合及反向（无计时节点CPU构建）：

|单机T/rank，B/D|FO main/us|FM main/us|FO wall/us|FM wall/us|
|---|---:|---:|---:|---:|
|8192，1024/4|1440.510|1451.620|1976.270|1985.590|
|20480，1024/4|3346.850|3378.300|3945.400|3975.050|
|20480，512/8|3439.900|3546.850|4017.810|4127.270|

48个字符串hash相同，pool与对应FM几何一致。相同B1024时FO仅额外改善
0.77%/0.93%；B512/20K反向仍改善3.02%。不能把FO与FP首轮收益相加。单机
8K的FO B512（上一组1432.100/1932.100us）已经优于较大窗口，20K则B1024
更快；不据两个离散规模引入未经测试的自动阈值。日志
`checkpoint_window_{fo,fm}_remote_ep8_t*_b*_d*.log`。

FO与FM的local/network inference prefill完整SASS dump逐字节相同，未使用的
checkpoint对象已消除。该检查在第一节点进行，第二节点独占计时；仍执行
inference数值与正式矩阵回归，不将FO训练收益误记为inference改善。

FO单机正式inference回归（1620MHz、graph 10 warmup/20 samples/batch20）：
T1/64/8192/20480分别100.970/135.322/1249.680/2962.120us；同批冻结V3
长文1487.020/3299.810us。32个FO hash与FM一致。第二节点独占计时，第一
节点构建FO library；无计时节点CPU构建。日志
`checkpoint_address_final_{v3,fo}_remote_ep8_t*_formal.log`。Inference代码
相同，数据仅作回归，不归因于FO。

FO的EP8完整Torch检查已通过：T1/64的OOM/retry及三个live tapes、8K均匀/
20K热点的训练数值、inference四规模与tiny-ring/非ring回归；另外B1024/D4
的8K均匀/20K热点训练也通过。随后EP16相同训练四档、inference四档及tiny
ring全部通过；B1024/D4的8K均匀/20K热点训练亦通过。rank0/15的T64小ring
memcheck（134575/373948）和T2048热点训练memcheck（136819/374903）均为
0 errors；Forward专项racecheck（135213/374328、138431/375278）均为
0 hazards/0 warnings。不称完整Backward racecheck clean。

Library SHA256 `b36ee66e81ab7515c3d13573ae4df9a7162596d9e728af6a0bd2978b4b2f0cdb`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
20:54负载下16张GPU均SM/graphics1620MHz、memory3201MHz（memory仅观测）。
正式双机性能回归及B1024组合复测随后单独进行。

FP再以FO固定源码扫20K的中间窗口，按下表顺序，第二节点单机独占，第一
节点同时执行FO Torch正确性（无CPU构建）；同为1620MHz、retained-tape10/20：

|B/D|main p50/us|wall p50/us|device pool bytes/rank|
|---|---:|---:|---:|
|1024/4|3344.380|3967.880|575356480|
|768/5|3354.020|3941.740|487722560|
|1280/3|3383.260|3990.030|654976832|
|1536/3|3405.440|4030.660|752630848|
|1024/4复测|3348.290|3964.540|575356480|

40个hash一致。B768近似持平且少87.6MB，继续增大到B1280/1536无收益；
B×D并不全部相同，不称等容量对照。B768尚无完整Torch门禁，不据此自动选
配置。日志`training_window_fp_fo_sweep_remote_ep8_b*_d*_r*.log`。

FO完整双机门禁之后，两节点独占且无构建，1620MHz、retained-tape10/20，
每规模按V3→FM→FO补测组合。FM/FO均B1024/D4：

|T/rank|V3 main/us|FM main/us|FO main/us|V3 wall/us|FM wall/us|FO wall/us|
|---|---:|---:|---:|---:|---:|---:|
|8192|2916.670|3027.490|2980.130|3435.810|3850.650|3806.700|
|20480|6635.900|7459.300|7348.100|7155.280|8428.500|8313.000|

96个原始uint64 hash字符串相同，各版本shape/retained/10warmup/20samples
已核对。相同窗口下FO main额外改善1.56%/1.49%，FM/FO pool完全相同；该窗口
的容量代价仍是上文每rank约168/167MB，不能忽略。组合FO相对V3 main仍慢
2.18%/10.73%、wall慢10.79%/16.18%。不据此关闭训练性能目标，也不更改
inference默认B512/D8或引入训练自动阈值。日志
`checkpoint_window_final_{v3,fm,fo}_ep16_t*_training_forward.log`。

FO代码里程碑提交`dd7d817`；完整inference正式矩阵另见ring记录FO，性能与
FM持平。FP是独立、显式的运行时容量选择，不是新增代码优化。

### FQ：训练稳定物理行序（验证中，TMA前置消融）

仅训练planner复用Notify的canonical `route_dst_row`。每个source window
在共享scratch中对各expert取最小行号，以`canonical-min`代替atomic reservation
得到稳定序号；同expert的连续token窗口在Notify稳定序列中是连续子段，重复
expert slots也保持不同序号。生产训练不含masked routes，generic/inference
planner保持原有行为。Destination物理行按source-rank拼接，canonical tape
布局、Backward顺序和固定combine顺序不变。

不增加GM metadata、交换量或pool；E64训练plan仅多约512B共享scratch及对齐，
main共享内存不变。先冻结仍采用FO uint4复制的版本，隔离行序影响，再测试
成块TMA checkpoint；此前FB仅source-major但source内部仍atomic乱序，两者
不是同一实验。不预设收益。冻结目录
`build/static_v4_canonical_training_plan_fq/`，native构建开始21:03。

Native SHA256 `29f6138208e763156f9d1b06e6faaf0224db3145537ce8eaa31a3df6a98e7276`。
EP8 T512/K6/B512/D8、EP16 T64/K6/B16/D1双次Backward的96个梯度hash与FO
相同。local planner REG80→86，network planner REG107/STACK72不变；main
训练/推理资源不变。local/network inference prefill完整SASS逐字节等于FO。

两节点独占、无构建，1620MHz、B512/D8、retained-tape10/20，FO→FQ：

|T/rank|FO main/us|FQ main/us|FO wall/us|FQ wall/us|
|---|---:|---:|---:|---:|
|8192|3324.540|3337.020|4150.280|4156.780|
|20480|7995.620|7987.520|8987.140|9005.200|

64个原始hash字符串与pool一致。单独稳定行序没有明显双机收益，保留为下一项
TMA复制的隔离对照，不独立采用。日志
`canonical_training_plan_{fo,fq}_ep16_t*_training_forward.log`。

随后第一节点编译FR，第二节点独占补测EP8（计时节点无CPU构建），同样
B512/D8、1620MHz、retained-tape10/20、FO→FQ：

|T/rank|FO main/us|FQ main/us|FO wall/us|FQ wall/us|
|---|---:|---:|---:|---:|
|8192|1427.680|1376.060|1926.140|1873.990|
|20480|3449.790|3303.490|4036.300|3884.300|

32个字符串hash及pool相同。单机main改善3.62%/4.24%，wall也改善，说明
稳定物理行序本身有可测单机收益，不能把未来组合收益全算给TMA。仍未完成
FQ完整Torch/sanitizer，不提前采用；FR若不划算，再考虑独立验证FQ。
日志`canonical_training_plan_{fo,fq}_remote_ep8_t*_training_forward.log`。

### FR：M32连续SavedX的TMA写回（单机改善，全拓扑方案未采用）

在FQ上识别物理Mblock内完整、对齐的32行段：canonical行严格递增，因此首尾
差31即证明整段连续。复用现有`saved_x_map`为ring编码64列×32行的BF16 box，
非ring仍用原M128 map；不新增map数组、GM段表或payload。连续段由FC1 Ntile0
的thread0从ready A stage发TMA，每个N64 swizzle atom分别复制；其余行沿用
FO uint4 scatter，二者不重叠。`tma_mask`在task开始计算，不在K循环扫描行号。

每K迭代提交一个bulk group，WGMMA wait0后thread0等完整global completion，
再通过首warp的`__syncwarp`交接给lane0/1的两CTA empty到达。没有新增全局/
cluster barrier或计数器；不能只等WGMMA或TMA的SMEM read completion。当前
ShareA由CTA0发全部A multicast，并非两CTA各发一半；warp join是显式完成交接，
不是在原版中复现了race的结论。SavedX仍先于FC1完成发布，tape生命周期不变。

CUDA12.9的[Tensor Map约束](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
允许该box；[PTX swizzle布局](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#tensor-swizzling-modes)
的128B模式以1024B重复，M32/N64 atom偏移4096/16384B保持原有对齐。这是布局
推导，仍须数值及sanitizer实测。先只实现M32单map以隔离效果，不预设收益；
可能受额外寄存器、issuer开销和小TMA写回延迟影响。
冻结`build/static_v4_tma_x_checkpoint_fr/`，21:12开始native构建；第一节点
编译时，第二节点独占补测FQ单机性能，不在计时节点编译。

Native SHA256 `816f36b325e4224b2b56e37e62c7fa2b6104c0340dd7407d56d6d95aacdcba15`。
EP8 T512/K6/B512/D8、EP16 T64/K6/B16/D1重复Backward共96个梯度hash与FQ
相同。local prefill training STACK72→80，network仍408；REG168/SHARED2048/
LOCAL0不变，未新增Forward WGMMA序列化警告。两个inference prefill完整SASS
仍与FO/FQ逐字节相同。训练SASS确认SavedX K循环有8处UTMASTG.2D及完成等待，
随后开始独占性能对照；还没有完整Torch/sanitizer门禁，不提前采用。

两节点独占、无构建、1620MHz、B512/D8、retained-tape10/20，FQ→FR：

|T/rank|FQ main/us|FR main/us|FQ wall/us|FR wall/us|
|---|---:|---:|---:|---:|
|8192|3319.780|3321.280|4138.790|4121.440|
|20480|8106.270|8010.780|9058.420|8982.970|

64个原始hash字符串及pool相同。8K持平，20K本轮main快1.18%；但FQ20K的
前轮为7987.520us，本轮baseline有所波动，不能只看这一轮百分比就采用。
日志`tma_x_checkpoint_{fq,fr}_ep16_t*_training_forward.log`。

本机编译FR2时，第二节点独占补测单机EP8（无计时节点CPU构建），同样
1620MHz、B512/D8、retained-tape10/20、FQ→FR：

|T/rank|FQ main/us|FR main/us|FQ wall/us|FR wall/us|
|---|---:|---:|---:|---:|
|8192|1377.950|1360.740|1882.920|1857.840|
|20480|3299.680|3235.840|3891.510|3848.890|

32个字符串hash及pool相同。单机main额外改善1.25%/1.93%，不能将FQ已有的
3.6%–4.2%行序收益再记一次。日志
`tma_x_checkpoint_{fq,fr}_remote_ep8_t*_training_forward.log`。

### FR2：区分SMEM读者退休与tape写回完成（无稳定额外收益，撤回）

基于冻结FR，仅分开两个完成条件。每stage用`wait_group.read 0`确认TMA不再
读取SMEM，经同一warp join交给两CTA empty；task结束仍由issuer执行完整
`wait_group 0`，位置在FC1 epilogue之后、现有threadfence/math join及fc1_done
发布之前。SavedX与S/G/U独立，允许其写回与epilogue重叠；没有去掉最终全局
完成检查，也不依赖kernel退出替代它。

[PTX等待语义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
明确区分source读取完成和destination写入可见。FR2利用该区别缩短stage的
保守等待，仍须验证完整slot/tape生命周期，不以hash相同代替同步证明。
源码冻结于`build/static_v4_tma_x_checkpoint_fr2/`；21:19本机开始构建，另一
节点独占测试冻结FQ/FR单机性能。未增加buffer/map/metadata/counter。

Native SHA256 `5fa5395390eef54caf8e919b07fa8c4ccbf2a13a3811585d9d566199f0ab46df`。
EP8/EP16相同native夹具的96个梯度hash与FR/FQ一致。local training STACK80
不变，network408→416；REG168/SHARED2048/LOCAL0不变，两种inference prefill
SASS仍逐字节相同。SASS确认K循环的等待后不再每次CCTL.IVALL，task末尾仍有
完整等待及invalidation。是否抵消新增live-range/stack代价，由独占对照决定。

两节点独占、无构建，1620MHz、B512/D8、retained-tape10/20，FR→FR2：

|T/rank|FR main/us|FR2 main/us|FR wall/us|FR2 wall/us|
|---|---:|---:|---:|---:|
|8192|3326.370|3318.050|4139.030|4118.160|
|20480|8097.860|8048.770|9057.820|8995.540|

64个原始hash字符串与pool相同。额外main改善0.25%/0.61%，幅度较小，尚需
反向对照与完整门禁，不先采用。日志
`tma_x_checkpoint_fr2{,_baseline_fr}_ep16_t*_training_forward.log`。

单机EP8首轮（第二节点独占，本机并行构建FR2 library；无计时节点CPU构建），
同样1620MHz、B512/D8、retained-tape10/20，FR→FR2：

|T/rank|FR main/us|FR2 main/us|FR wall/us|FR2 wall/us|
|---|---:|---:|---:|---:|
|8192|1355.550|1364.700|1852.500|1861.310|
|20480|3245.760|3229.630|3834.900|3805.120|

8K main回退0.68%，20K改善0.50%，方向不一致，立即按FR2→FR反向复测。
还不能确认分离等待比FR更好，不先采用。日志
`tma_x_checkpoint_fr2{,_baseline_fr}_remote_ep8_t*_training_forward.log`。

同一remote独占/本机FR2 library构建条件下，反向FR2→FR：

|T/rank|FR2 main/us|FR main/us|FR2 wall/us|FR wall/us|
|---|---:|---:|---:|---:|
|8192|1354.180|1355.550|1852.290|1855.910|
|20480|3229.700|3232.380|3811.470|3810.140|

32个字符串hash及pool相同。反向8K/20K main仅差0.10%/0.08%，未证明分离
等待有稳定额外收益；结合双机微小差异，不保留额外生命周期分支。已撤回FR2，
恢复FR逐stage完整global wait及warp join，源码patch与冻结FR逐字节相同。
FR2仅完成native及性能快筛，library虽已构建但没有运行完整Torch/sanitizer，
不称其已通过完整门禁。FR2库SHA256
`4937560a97806ccd087aeb081bc40191c608714bcf708d76ff680b6612e10c57`；相关源码/
binary/log均保留。反向日志
`tma_x_checkpoint_fr2_reverse_{fr2,fr}_remote_ep8_t*_training_forward.log`。

### FR：完整验收与全拓扑取舍

后续选择FQ+FR组合进行完整验证，尚未采用；FR library于21:33重新构建，
另一节点独占补测FO/FR及V3训练、B512/D8与B1024/D4容量组合。

完整候选库构建期间，第二节点独占、无计时节点CPU构建，1620MHz、retained-
tape10/20；每规模按表序V3→FO512→FR512→FO1024→FR1024：

|T/rank|版本/B/D|main/us|wall/us|pool bytes/rank|
|---|---|---:|---:|---:|
|8192|V3|1334.690|1741.160|735672000|
|8192|FO/512/8|1435.840|1933.240|313659456|
|8192|FR/512/8|1357.820|1856.730|313659456|
|8192|FO/1024/4|1433.380|1957.240|481305408|
|8192|FR/1024/4|1415.300|1974.470|481305408|
|20480|V3|3131.620|3567.740|1820708544|
|20480|FO/512/8|3445.310|4023.140|408121664|
|20480|FR/512/8|3247.200|3828.970|408121664|
|20480|FO/1024/4|3349.500|3966.290|575356480|
|20480|FR/1024/4|3254.210|3856.210|575356480|

80个字符串hash一致，actual shapes及10/20已核对。FR包含FQ行序和TMA两项，
B512相对FO的main改善5.43%/5.75%，不能再叠加FQ/FR分项收益。FR B512仍比
V3 main慢1.73%/3.69%，wall仍落后；B1024在当前FR单机没有额外收益且更占
pool，维持默认B512/D8，不自动扩大窗口。这里pool不含独立tape，不将差额
等同于全部训练内存节省。日志
`tma_x_checkpoint_final_{v3,fo,fr}_remote_ep8_t*_b*_d*_training_forward.log`。

随后第一节点执行FR完整Torch，第二节点独占反向FR→FO，B512/D8、同样1620MHz
及10/20：8K main1357.340/1430.820、wall1854.080/1935.590us；20K main
3230.750/3441.150、wall3816.590/4019.520us。32个hash及pool相同，反向main
改善5.14%/6.11%，与正向方向一致。日志
`tma_x_checkpoint_reverse_{fr,fo}_remote_ep8_t*_training_forward.log`。

FR library SHA256 `936622e8b32bbbdba0f722c0947c4a35999a437525d611982e573629c1b536cc`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
源码与冻结native patch一致。EP8完整Torch九项已通过：四档训练、20K热点、
短档OOM/retry及三个live tapes，inference完整240/tiny152/非ring32条检查，
另外B1024/D4的8K均匀/20K热点训练各通过。双机完整检查随后启动，尚未采用。

正式EP8 inference graph 10warmup/20samples/batch20、1620MHz：FR四档为
101.000/135.282/1247.670/2953.140us，同期V3长文1488.050/3299.380us。
32个FR hash与FO相同，shape/10/20/batch20已核实；第二节点独占计时，第一
节点先构建库、再执行Torch正确性。Inference SASS相同，仅作回归，不记为
新增inference收益。日志`tma_x_checkpoint_final_{v3,fr}_remote_ep8_t*_formal.log`。

21:55双机完整十二项检查exit0，24份node日志的rank级PASS数量全部核对：
四档训练及20K热点、短档OOM/retry/三个live tapes、inference完整480/tiny304，
另有B1024/D4的8K均衡/20K热点训练。T64/B16/D1与T2048/B512/D8热点训练的
rank0/15 memcheck各0 errors；同两档限定planner/main的Forward racecheck
各0 hazards/0 errors/0 warnings，应用数值/graph/offload也均完成。这里graph
是既有inference graph复用检查，不代表训练Backward支持graph或racecheck全净。
内存日志PID181548/391066、185605/392009；race日志183553/391440、188122/392389。
21:52负载下16张卡SM/graphics均1620MHz、memory观测3201MHz，后者未锁定。
源码patch再次与冻结native逐字节核对一致。全部检查后才开始两机独占的正式
inference及V3/FO/FR训练对照，不并行CPU构建或其它GPU工作。

最终双机独占、无构建，1620MHz、retained-tape10/20；每规模按表序
V3→FR512→FO512→FR1024→FO1024，刻意反向candidate→baseline：

|T/rank|版本/B/D|main/us|wall/us|pool bytes/rank|
|---|---|---:|---:|---:|
|8192|V3|2901.150|3410.570|1079452608|
|8192|FR/512/8|3329.310|4132.630|702615744|
|8192|FO/512/8|3313.500|4108.800|702615744|
|8192|FR/1024/4|3075.300|3911.600|870256832|
|8192|FO/1024/4|2963.100|3795.660|870256832|
|20480|V3|6660.160|7192.790|2167634880|
|20480|FR/512/8|8059.070|9019.420|819971520|
|20480|FO/512/8|7982.430|8930.300|819971520|
|20480|FR/1024/4|7239.900|8201.960|987170496|
|20480|FO/1024/4|7351.300|8327.940|987170496|

160个原始hash字符串与同期V3逐rank相同，实际shape/world/10/20/retained
均核实，FO/FR相同B/D的pool相等。B512 main反向慢0.48%/0.96%，没有稳定
网络收益；B1024的8K慢3.79%，FR范围3028.640–3152.860us、FO范围
2931.620–2996.060us不重叠，wall同样变慢。20K B1024则快1.52%，不能用它
掩盖8K回退，也不据此增加长度阈值。**不采用全拓扑FQ+FR组合**，后续只在
有稳定收益的单机specialization保留canonical/TMA，网络恢复FO。
日志`tma_x_checkpoint_final_{v3,fo,fr}_ep16_t*_b*_d*_training_forward.log`。

同一独占阶段的inference graph正式10/20/batch20：FR四档
268.560/317.882/2899.910/6890.180us；同期V3长文3348.090/7107.340us。
64个FR hash与FO相同，pool不变；shape及计数核实。这是未采用全拓扑FR的
inference回归证据，不能重标为后续FR3实测。日志
`tma_x_checkpoint_final_{v3,fr}_ep16_t*_formal.log`。

### FR3：只保留单机canonical/TMA checkpoint（采用）

已有`StaticLocal/Network`编译期分支选择：单机训练保持FQ稳定source顺序、
source-major物理行及FR M32 TMA；网络恢复FO的atomic source顺序、round-robin
物理行和uint4 SavedX。checkpoint helper的非contiguous特化不做TMA探测、
空指针过滤或wait/warp join；网络也不创建无用的SavedX tensor map。没有新增
运行时阈值/调参开关或另一套通信协议，pool和canonical tape布局不变。
须验证单机计算指令等于FR、网络计算/规划等于FO，不能凭源码直觉复用结论。
冻结`build/static_v4_local_tma_x_checkpoint_fr3/`；22:07在双机计时全部结束后
开始native构建；后续验收结果如下。

独立的FS规划融合已写出草案，但在FR网络回退确认后先完整撤回，保留unbuilt
patch；没有FS构建、性能或验收结果。待FR3稳定后再单独实施，避免混合归因。

FR3 native SHA256 `0761d1333ee43167a526af99ce1e8ec7998140bb1488770c544f0552c45c437b`。
EP8 T512/E16/K6/B512/D8与EP16 T64/E32/K6/B16/D1各重复Backward两次，
96个梯度hash与FR/FO逐rank相同。完整symbol SASS比较覆盖decode/prefill
各四个local/network×inference/training entry，以及两种training planner：
local training/plan逐字节等于FR，network training/plan逐字节等于FO，四种
inference entry等于FO。比较文件均含实际函数指令，不以空dump作证据。
Prefill local training STACK80、network408，inference64/392，REG168/
SHARED2048/LOCAL0不变。

本轮复用这些**完全相同代码**在FR/FO已有的专项sanitizer证据，不将旧日志重标为
FR3新跑；另做FR3 artifact的native、EP8完整Torch及EP16四档训练、OOM/retry/
live tapes、inference full/tiny与B1024长文集成回归。library构建与第二节点
独占EP8正式计时并行，计时节点没有CPU构建；双机计时仍等所有构建/检查结束。

FR3 library SHA256 `f0c4253e547c8a8cf78d0d4c6d9c1ade4f0ab7a2fe6295eae65bc78a92b06a7a`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
22:19 EP8完整九项Torch通过，包含四档训练、20K热点、短档OOM/retry/三个
live tapes、inference full240/tiny152/nonring32，以及B1024的8K/20K训练。
随后启动双机八项组件集成回归；本轮不重复字节相同kernel的sanitizer。

第二节点独占计时，第一节点构建FR3 library；1620MHz、B512/D8、retained-
tape10/20，FO→FR3：

|T/rank|FO main/us|FR3 main/us|FO wall/us|FR3 wall/us|
|---|---:|---:|---:|---:|
|8192|1423.390|1356.190|1922.000|1852.570|
|20480|3442.880|3234.370|4023.450|3814.580|

32个字符串hash及pool完全相同；main降低4.72%/6.06%，wall降低3.61%/5.19%。
FO8K有一个1850.110us高值，完整20samples保留，未删除该点；上述均为p50。
pool仍313659456/408121664 bytes/rank，不含独立tape。日志
`local_tma_x_checkpoint_{fo,fr3}_remote_ep8_t*_training_forward.log`。
同阶段正式inference10/20/batch20：FR3四档101.075/135.166/1251.630/
2957.950us，同期V3长文1489.390/3300.590us；48个hash及实际shape/计数
已核对，inference持平，不算新增收益。日志
`local_tma_x_checkpoint_final_{v3,fr3}_remote_ep8_t*_formal.log`。

22:27 FR3双机八项组件集成检查exit0，16份node日志的rank级PASS数量核实：
四档训练、20K热点、短档OOM/retry/三个live tapes、inference full480/tiny304，
以及B1024的8K/20K训练均通过。未运行FR3新sanitizer，复用前述字节相同
local FR/network FO代码的检查；不宣称完整Backward racecheck clean。
全部库构建、Torch检查结束且两机GPU空闲后，才启动FO→FR3双机同容量性能
对照及FR3正式inference回归；无并发CPU构建或其它GPU任务。

双机最终同容量对照，1620MHz、retained-tape10/20，每规模依次FO512→FR3512→
FO1024→FR31024，单位us：

|T/rank|B/D|FO main|FR3 main|FO wall|FR3 wall|pool bytes/rank|
|---|---|---:|---:|---:|---:|---:|
|8192|512/8|3316.610|3327.460|4110.980|4122.670|702615744|
|8192|1024/4|2991.840|2988.420|3824.130|3816.030|870256832|
|20480|512/8|8009.280|7979.550|8972.530|8963.560|819971520|
|20480|1024/4|7348.030|7345.540|8315.100|8320.020|987170496|

128个原始hash字符串与本轮FR阶段的同期V3逐rank一致，实际shape/world/10/20/
retained及同容量pool均已核对。网络结果持平，不将不足0.4%的差异算作收益；
全拓扑FR在B1024/8K的3.79%回退已消除。pool不含独立tape。日志
`local_tma_x_checkpoint_{fo,fr3}_ep16_t*_b*_d*_training_forward.log`。

同阶段正式EP16 inference graph 10/20/batch20：FR3四档269.571/317.544/
2897.900/6878.520us，同期V3长文3353.190/7102.160us。64个FR3 hash与FR
逐rank相同，长文同时与本次V3相同；pool/注册bytes、实际shape及计数已核实。
日志`local_tma_x_checkpoint_final_{v3,fr3}_ep16_t*_formal.log`。
Inference仍持平，不记为本次新增收益；双机20K相对V3仅降低3.15%，不能关闭
明显领先目标。采用FR3，拒绝通用FR及FR2；训练端到端仍落后V3，下一步独立
检查规划launch、D2H和host同步，不把main计时范围变化误报为计算加速。

### FS：将训练物理窗口规划合入驻留main（不采用）

基线FR3 `06f6095`。materialized Notify已提供精确R/P，原先ring plan回传的R
只用于重复校验，不用于tape分配。候选保留Notify及collective tape准备边界，
在全部rank分配成功之后由main调用同一个ring planner，同时stage input；删除
独立planner entry、runtime D2H和一次host stream同步。发生分配失败时尚未
改变设备ring epoch/base，也未准入payload；保留全rank失败一致性及retry。
不改变独立tape所有权、canonical映射、GEMM、通信协议或training graph限制。

planner借用GEMM operand arena，校验scratch不会覆盖已初始化的barrier；没有
新GM allocation或双份planner。main开始时间现在覆盖物理规划，不能与FR3
不含规划的main数值直接比较，验收以含Notify、准备及Forward的wall为准。
冻结`build/static_v4_fused_training_plan_fs/`；22:42开始native构建，本节尚无
数值、性能或sanitizer结论，不将FR3检查重标为FS。

FS native SHA256 `c8e7ec2410fc5a39675b6ae86711fa1b7b864226ee922b72340793f0be00a52b`。
EP8 T512/E16/K6与EP16 T64/E32/K6两次Backward的96个梯度hash等于FR3；
四种inference完整SASS逐字节相同。训练decode/prefill的local/network STACK
均为64/392，REG168/SHARED2048/LOCAL0；栈减小本身不等于性能改善。

第二节点独占EP8，第一节点构建FS库；1620MHz、B512/D8、retained10/20，
FR3→FS，单位us。旧main不含物理规划，FS main包含，不能直接比较其百分比。

|T/rank|FR3 wall|FS wall|FR3 main|FS main|FR3 plan|FS plan|
|---|---:|---:|---:|---:|---:|---:|
|1|408.015|409.613|77.568|115.968|41.824|46.848|
|64|463.371|434.412|107.040|149.088|43.840|53.920|
|8192|1861.490|1830.280|1361.120|1433.570|89.728|116.032|
|20480|3809.090|3821.730|3233.410|3386.850|137.440|212.800|

64个hash及同shape pool相同。反向FS→FR3的8K wall1823.610/1864.500、
20K3834.430/3814.970us；main1430.270/1362.180、3396.100/3232.960us，
32个hash一致。8K wall两次改善1.68%/2.19%，20K两次反而慢0.33%/0.51%，
融合后的规划阶段也更长，不将其笼统称为训练加速。日志
`fused_training_plan_{fr3,fs}_remote_ep8_t*_training_forward.log`及
`fused_training_plan_reverse_{fs,fr3}_remote_ep8_t*_training_forward.log`。
待双机同容量对照后决定，不增加长度阈值掩盖20K结果。

最终双机独占且无构建，1620MHz、retained10/20、FR3→FS，同容量对照：

|T/rank|B/D|FR3 wall/us|FS wall/us|FR3 main/us|FS main/us|
|---|---|---:|---:|---:|---:|
|8192|512/8|4109.490|4059.780|3314.750|3482.560|
|8192|1024/4|3822.250|3807.110|2984.190|3247.230|
|20480|512/8|8961.390|8933.330|7991.360|8345.570|
|20480|1024/4|8295.470|8180.320|7341.220|7589.660|

实际shape/world/10/20/retained已核实，128个原始hash字符串及各容量pool与FR3
相同。网络wall下降0.31%–1.39%，但单机20K正反向均小幅变慢，因此不采用
全拓扑规划融合，也不为这组微小差异引入长度开关。所有FS代码已撤回，冻结
patch保留。FS library已构建（SHA256
`57b4a805acc5f3c3a0bb888a5afccb467804d3b8bf436a10c21ee967c5b263e4`），
但**没有执行FS Torch/sanitizer完整门禁**，准备了脚本不等于验收。没有FS
inference正式性能数据；只有SASS相同证据及native训练hash检查。日志
`fused_training_plan_{fr3,fs}_ep16_t*_b*_d*_training_forward.log`。

### FS2：保留独立GPU规划，只消除host结果回传（采用）

直接基于FR3，不含FS融合。先按Notify实际R/P完成全rank tape准备，再在同一
stream顺序launch原来的独立planner和main，删除中间runtime D2H及host同步。
保留原384线程输入staging及全部GPU算法；仅移动host准备边界，避免将规划
资源/访存变化混进host同步实验。失败时仍无设备epoch/base/payload推进。
main计时口径与FR3相同，但收益仍以wall为主。没有新的buffer、runtime knob
或测试套件。冻结`build/static_v4_async_training_plan_fs2/`，22:55在上述双机
计时全部结束后才开始native构建；尚无FS2正确性或性能结论。

FS2 native SHA256 `20d4c0b80df703b7c917a7a1e06d845995f0863d85f5c1fc014829653d547705`。
EP8 T512/E16/K6与EP16 T64/E32/K6各两次Backward，96个梯度hash与FR3逐rank
一致。8种decode/prefill×local/network×inference/training main及2个独立
planner的完整SASS均逐字节等于FR3，dump均非空。Prefill训练STACK80/408、
REG168/SHARED2048/LOCAL0，与FR3相同。不是引用源码相似性来推断算法不变。
随后构建集成库，并在第二节点独占测量四档EP8；计时节点无CPU构建。

EP8首轮FR3→FS2，B512/D8、retained10/20、1620MHz；单位us：

|T/rank|FR3 wall|FS2 wall|FR3 main|FS2 main|pool bytes/rank|
|---|---:|---:|---:|---:|---:|
|1|407.037|378.599|78.272|71.232|17022528|
|64|463.149|435.704|105.984|100.672|45442880|
|8192|1855.280|1836.860|1358.300|1354.050|313659456|
|20480|3864.220|3794.070|3235.390|3229.890|408121664|

64个hash及pool一致，实际shape/world/10/20/retained核实。四档wall下降
6.99%/5.93%/0.99%/1.82%；main口径与FR3一致，但不把微小变化称为GEMM
代码优化。pool不含独立tape。日志
`async_training_plan_{fr3,fs2}_remote_ep8_t*_training_forward.log`。
尚需反向及双机对照和集成门禁，本表不是最终采用结论。

同样1620MHz/B512D8/10/20的反向FS2→FR3：8K wall1838.540/1859.180us，
20K3791.440/3847.430us，分别降低1.11%/1.46%；main1350.460/1359.490、
3226.530/3232.320us。32个hash和pool相同，未出现FS的20K回退。第一节点
先构建库、后执行Torch，第二节点独占计时。日志
`async_training_plan_reverse_{fs2,fr3}_remote_ep8_t*_training_forward.log`。

FS2 library SHA256 `4c07a0cb83da6596698064b9a7eba376d902a8979cb161765c3914ae2e962e84`，
extension `725bf863a6f3fc19a618f5afb739a957df2fa3ae2d7105e930869b5ee8b825c3`。
23:06 EP8完整九项检查exit0：四档训练、20K hotspot、短档OOM/retry/三个live
tapes，inference full240/tiny152/nonring32，另有B1024/D4的8K/20K训练。
各文件rank级PASS数量逐项核实。随后开始双机十二项完整检查，不能提前称为
双机或sanitizer通过；现有graph检查不代表训练Backward graph支持。

同阶段正式EP8 inference graph10/20/batch20：FS2四档
101.083/135.254/1256.700/2949.580us，同期V3长文1487.820/3298.570us。
48个hash与FR3/同期V3相同，实际shape/计数及FS2 pool不变；这是inference
回归，代码相同、性能持平，不能记为host训练优化新增收益。日志
`async_training_plan_final_{v3,fs2}_remote_ep8_t*_formal.log`。

23:19双机十二项完整检查exit0，24份node日志的rank级PASS数量全部核实：
四档训练、20K热点、短档OOM/retry/三个live tapes、inference full480/tiny304，
以及B1024/D4的8K/20K训练均通过。T64/B16/D1和T2048/B512/D8热点训练的
rank0/15 memcheck均0 errors；Forward planner/main过滤的racecheck两档均
0 hazards/0 errors/0 warnings。memcheck PID258203/415143、260883/416092，
racecheck259443/415524、261942/416467。不宣称完整Backward racecheck clean。
这轮是FS2 artifact的新检查，不是复用FR3日志。部分正确性检查与FT独立候选
的CPU预编译并行，不使用这些检查的时间值作性能数据。所有CPU构建及GPU
检查结束后，23:20才开始FS2两机独占的最终训练/正式inference对照。

双机独占、无CPU构建，1620MHz、retained10/20；每规模按表序，单位us：

|T/rank|版本/B/D|main|wall|pool bytes/rank|
|---|---|---:|---:|---:|
|1|FR3/512/8|172.608|630.524|390711488|
|1|FS2/512/8|168.416|626.920|390711488|
|64|FR3/512/8|226.752|708.788|419244736|
|64|FS2/512/8|215.360|689.455|419244736|
|8192|V3|2922.300|3428.790|1079452608|
|8192|FR3/512/8|3307.840|4109.420|702615744|
|8192|FS2/512/8|3316.770|4128.670|702615744|
|8192|FR3/1024/4|2994.370|3822.300|870256832|
|8192|FS2/1024/4|2979.070|3814.420|870256832|
|20480|V3|6658.910|7181.200|2167634880|
|20480|FR3/512/8|7989.570|8938.050|819971520|
|20480|FS2/512/8|7965.410|8926.920|819971520|
|20480|FR3/1024/4|7345.700|8328.090|987170496|
|20480|FS2/1024/4|7352.830|8299.000|987170496|

224个原始hash字符串逐rank一致，长文与本次V3相同；实际shape/world/10/20/
retained及同B/D pool均核实。短档wall约持平/改善2.73%，长文多数差异小于
0.5%；8K B512的FS2 wall反而慢0.47%。不能将单机收益直接推广到网络，
仍需默认窗口的反向对照；不丢弃不利样本。pool不含独立tape。日志
`async_training_plan_{v3,fr3,fs2}_ep16_t*_b*_d*_training_forward.log`。

同阶段正式EP16 inference graph10/20/batch20：FS2四档268.682/317.037/
2899.510/6871.030us，同期V3长文3359.200/7103.860us。96个hash与FR3及
同期V3相同，实际shape/计数及FS2 pool不变。与FR3持平，不记为新增收益；
20K相对V3仍只降低3.28%，不能关闭整体性能目标。日志
`async_training_plan_final_{v3,fs2}_ep16_t*_formal.log`。

默认B512/D8的双机反向FS2→FR3，仍为两节点独占、无CPU构建、1620MHz及
retained10/20：8K wall4093.790/4127.520us，main3302.180/3330.140us；
20K wall8990.520/8941.950us，main8047.300/7981.920us。64个原始hash与
本次V3逐rank一致，实际shape/计数及pool核实。8K反向改善0.82%、20K反向
退化0.54%，与首轮的小幅变化方向相反；差值处于相互重叠的样本范围内。
因此将网络长文结果记为持平，不认定为稳定收益，也不引入长度分支。日志
`async_training_plan_reverse_{fs2,fr3}_ep16_t*_training_forward.log`。

采用FS2：单机wall正反向均改善，移除了一次不参与分配的设备结果回传与host
等待，且所有GPU入口指令、容量和通信协议保持不变。双机收益不作保证。
native retained benchmark复用已经准备的tensors，不把它描述为每轮包含实际
malloc；Torch检查另外覆盖真实tape准备、失败一致性和retry。主机generation
仍可能增加，分配失败不推进的是设备ring epoch/base及payload，而非所有状态。
完整Forward专项sanitizer通过不改变旧Backward racecheck的未决限制。

## FT之后：SavedX静态pair与消费前缀（GA + FZ，已采用，2026-09-25）

FT将完整M128的ring S写回改为TMA，训练仍保存canonical G/U；完整门禁通过，
训练性能持平。后续G/U M32 TMA候选FX没有稳定收益，已撤回。详细数据及
资源/产物哈希见[ring记录](static_forward_v4_ring_record.md)的FT、FX章节。

FZ保持canonical tape、Forward/Backward数值顺序、窗口容量不变：SavedX的
唯一owner由固定Ntile0改为轮转FC1 pair，两个CTA各复制64行。复用已就绪的
multicast A，复制字节数不变；每个CTA的目的指针状态减半，stage full wait、
两个CTA的empty到达、FC1发布前的device fence全部保留。没有新增checkpoint
缓冲区、映射或barrier。仅轮转而不分半的FY没有稳定收益，不单独采用。

FZ独立正反向对照中，单机8K main约改善1.1%；20K效果较小且有波动。网络
20K/B512 main约改善0.8%，8K/B1024约改善0.8%–1.33%；其它网络组合未形成
一致优势。不能将某一档收益推广为全部训练规模。各轮完整表和不利样本保留
在ring记录FY/FZ，不挑选较快的一轮代替相邻基线。

GA单独优化网络consumed credit：逐slot确认全部实际读者退休后，把连续
consumed prefix通过一次ACK返回；TX仍同时检查远端消费及显式本地NIC完成。
channel元数据增加4096 bytes/rank（2节点/32 lanes），payload容量不变。
不改变training checkpoint或Backward GIN协议。组合采用GA+FZ，不含额外
RX导入batch的GA2、G/U TMA的FX、FC2权重multicast或尚未实测的fence候选。

组合三项native检查160个梯度hash与FT逐rank一致，repeat_equal均1；单机9项、
双机12项完整Torch检查通过，包括四档、热点、多个live tapes、OOM/retry、
非ring回退及B1024/D4长文。双机T64/B16/D1和T2048/B512/D8的rank0/15
memcheck为0 errors，Forward planner/main过滤的racecheck为0 hazards/errors/
warnings。双机prefill检查首次因端口占用在launcher启动阶段失败，换端口后
补齐，保留失败日志；不是算子运行错误。详见ring记录组合验收的PID及覆盖数。

组合单机正式对照（1620MHz，retained 10warmup/20samples，rank-max p50/us）：

|T/rank|V3 main / wall|FT main / wall|GA+FZ main / wall|
|---:|---|---|---|
|1|197.760 / 471.682|71.584 / 384.761|69.856 / 383.170|
|64|264.320 / 554.903|100.416 / 436.316|97.152 / 430.782|
|8192|1330.340 / 1733.020|1349.660 / 1839.310|1335.140 / 1825.620|
|20480|3132.420 / 3588.340|3215.710 / 3816.910|3208.860 / 3780.860|

长文wall较FT改善0.74%/0.94%，但相对V3仍慢5.34%/5.37%；整体训练性能目标
未关闭。短档仅本轮相邻证据，不称反向验证过的稳定收益。local main使用
device_kernel_slowest_rank_us，不能误用V3包含其它阶段的forward字段。
计时在第二节点独占执行，第一节点同期仅做Torch正确性，没有CPU构建。
native retained测量复用已准备tensors，不包含每轮实际malloc；真实tape生命周期
另由Torch检查覆盖。日志及160个输出hash/shape/pool核验见ring记录组合EP8表。
双机28组最终对照正在运行，尚无组合网络训练性能结论。

02:39双机正式对照全部完成，1620MHz、两节点独占、无CPU构建，retained
10/20；各规模顺序为V3→FT→GA+FZ，长文再测FT→GA+FZ的B1024/D4。
256个原始输出hash逐rank等于同期V3，shape/world/计数/retained及pool核实：

|T/rank|B/D|V3 main / wall|FT main / wall|GA+FZ main / wall|
|---:|---|---|---|---|
|1|512/8|442.208 / 734.747|168.704 / 626.280|133.920 / 594.082|
|64|512/8|600.640 / 901.641|216.640 / 696.511|186.400 / 662.490|
|8192|512/8|2927.170 / 3438.270|3289.280 / 4092.800|3267.970 / 4058.140|
|8192|1024/4|同上|2975.490 / 3792.770|2930.590 / 3767.730|
|20480|512/8|6658.500 / 7181.430|7978.240 / 8929.010|7934.240 / 8911.160|
|20480|1024/4|同上|7332.220 / 8287.570|7238.080 / 8216.610|

长文同容量main改善0.55%–1.51%，wall改善0.20%–0.86%，小于短序列。
B1024的8K main接近V3，wall仍慢9.58%；20K wall仍慢14.41%。每rank元数据
仅较FT增加4096 bytes，但B1024本身比B512多约167–168 MB，不能称等容量
优化或直接翻转默认值。训练吞吐目标仍开放，继续独立消融物理行局部性。
本节不替代独立FZ的反向证据，也不将组合收益机械拆给两个改动。完整pool、
日志名称、graph回归及短档反向检查见ring记录GA+FZ最终矩阵。

## GC/GD：网络块内source分组及M16 SavedX（未采用，2026-09-25）

GC只在完整regular Mblock内按source分组，并将网络source ordinal改为Notify
稳定顺序；GD再对逐行验证连续的M16区间做TMA。均不增payload/目录容量，不
改变完整wait及slot释放条件。两候选各160个native梯度hash与GA+FZ一致；
GC规划检查及racecheck通过，未构建library或执行完整Torch/主路径sanitizer。

1620MHz，双机EP16独占，T8192/20480、B512/D8及B1024/D4的12组正式相邻
对照有192个输出hash一致。GC的8K两档main慢2.08%/1.18%、20K/B512慢0.63%；
只有20K/B1024快1.27%，反向再快1.46%。GD没有在GC上带来一致增益，尤其
20K/B1024的独立main变化仅±0.15%、方向翻转。故保留大窗长文的实验发现，
不为单个形状硬编码阈值，不采用GC/GD，也没有留下候选测试或TMA map选项。
全部main/wall、相同pool、反向48个hash及launcher参数误用记录，见
[ring优化记录](static_forward_v4_ring_record.md)的GC/GD小节；训练端到端目标仍开放。

## Mini Notify + main 内规划（2026-09-30）

训练前置阶段改为一次 counts-only cooperative kernel：统计并交换各 source/expert
的行数，发布实际 R/P 和容量检查结果。只保留每个 1536-route chunk 的 expert
前缀，不生成 route 行号、packet/fanout 或 V3 wave 排序。所有 rank 完成 tape
分配之后，PF main 在自己的 shared operand arena 中恢复稳定的
expert/source/route 行号，再构建 ring 物理计划；双机 Backward 所需的不可变
packet/fanout 直接从 main 的 node-token 目录生成。原独立 ring planner launch
被删除。推理仍不创建 tape，saved pointers 为空；训练/推理共用 main 实现，
保存代码仍由编译期 specialization 裁掉，而不是给推理增加运行时分支。

SavedX 为 `[R,H]`，G/U 为分离的两个 `[R,I]` 平面。逻辑行序和 route 返回映射
归属于各自 live tape，不依赖下一次 Forward 会覆盖的 ring 计划。mini Notify
是分配边界，不启动 activation payload；分配失败的 collective retry 合同保留。

第一轮 EP8 正式对照：同一远端节点独占，8×L20X、1620 MHz，H2048/I512、
E64/K4、BF16、30 轮（10 warmup + 20 measured），每轮取最慢 rank 后取 p50。
PF 使用 B512/D8，Forward 保留训练状态。单位 µs，包含 Notify 的 host wall：

| tokens/rank | V3 wall | mini wall | wall 降低 | V3 Notify | mini Notify |
|---:|---:|---:|---:|---:|---:|
| 1 | 469.704 | 307.791 | 34.47% | 201.338 | 110.799 |
| 64 | 553.803 | 355.534 | 35.80% | 216.157 | 116.381 |
| 8192 | 1737.380 | 1615.320 | 7.03% | 322.900 | 122.682 |
| 20480 | 3586.140 | 3542.170 | 1.23% | 362.455 | 130.557 |

main 的计时边界现在包含 planner，不能拿它直接对照旧 main 并将边界变化算成
收益。20K 差距较小，不能外推其他模型或节点数。日志为
`build/pf_training_mini_notify/{v3,mini_remote}_ep8_t*_forward.log`；所有输入、
权重和输出 hash 一致。冻结 mini native 二进制 SHA256 为
`72a362e1b8fade9c77c6ef17b50650e87adfff0b8b9426204a59229fa993edb5`。
V3 使用 `build/static_inference_v3_20260923/test_cuda_notify` 并显式设置
`WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`、清除 ring 环境变量；仅凭文件名不能
确定执行的是 V3。

Native 正确性：EP8 T512/E16/K6/B512/D8、EP16 T64/E32/K6/B16/D1，mini
Forward 接原 Backward，两次重复均稳定，dX/dGate/dW1/dW2 逐 rank hash 与
既有 FS 验证结果完全一致。日志位于该目录的 `ep{8,16}_t*_validation.log`。
随后使用 `build/pf_static_backward/` 的 library/extension 跑既有 Torch 集成测试：
EP8/EP16、T64/E64/K4、B16/D1，输出及四组梯度均通过 reference parity；
单 rank 前向/反向分配失败、保留 tape 后重试、三个 live tape、热点目的地切换
及工作区扩缩容均通过。双机测试随后在两个 EP8 group 上验证 inference。
日志为 `torch_ep8_t64.log` 和 `torch_ep16_t64_node{0,1}.log`。该构建含下述
首版静态 Backward；完整长序列矩阵及最终 Backward 性能另记后续结果。

### 静态 Backward 首版：细粒度 task residue 未采用

dU、SwiGLU epilogue、dX 分别将线性 tile 按 cluster 数取模分配，dW 必须等待
本 rank 全部 dX 完成；Return 也改为固定 M-block owner。EP8/EP16 上述 native
用例四组梯度 hash 均与 mini + 原 Backward 一致。但 EP8 同机同频对照，
T1/64 wall 为 257.792/390.887 µs（V3 为 281.948/420.911），T8192/20480
反而为 2540.650/6612.420 µs（V3 为 2127.380/5066.530），慢 19.43%/30.51%。
激活梯度流水明显变慢，保留六个微小 epilogue task 和跨 cluster readiness
造成的等待值得优先消除，不能仅因为去掉 atomic claim 就认为会更快。
记录于 `build/pf_static_backward/ep8_t*_backward.log`；该首版不是最终性能结论。

进一步拆分归因：mini Forward 接**原动态 Backward**时，8K/20K wall 已经是
2538.760/6227.080 µs。主要遗漏之一是旧 Notify 的 local route permutation：
按整个 expert 遍历会让大量发送 CTA 同时写同一张卡；V3 会按 32 行交错 expert。
因此不能把首版相对冻结 V3 的全部回退归因于静态计算调度。

第二版将一个 M-block 的 dU、一次完整 M128 SwiGLU epilogue 和全部 dX N tiles
固定交给同一 cluster。其 8K/20K wall 为 2290.390/6144.980 µs，仍未达标；
EP8 T8192 Torch 的输出及四组梯度已通过。下一步直接由 counts 构造 32 行
交错的 Backward dispatch 索引，不恢复 route-sized radix sort。上述消融都保持
dW 严格后置，不改变 GEMM 算术或归约顺序。

### Counts 直接构造交错 dispatch

先在 `packet_order` 临时区写稳定的 source/expert 列表，再按 counts 计算各
32-row stripe 的目的位置并写 `local_route_order`；网络 packet plan 随后覆盖
临时区。无需恢复 V3 wave 构造或全局 radix sort。对照结果：

| tokens/rank | V3 BWD wall (µs) | M-block + stripe | 耗时降低 |
|---:|---:|---:|---:|
| 1 | 281.948 | 303.724 | -7.72% |
| 64 | 420.911 | 445.834 | -5.92% |
| 8192 | 2127.380 | 1762.040 | 17.17% |
| 20480 | 5066.530 | 4692.760 | 7.38% |

仍为 EP8、同一远端节点 1620 MHz 的 rank-max p50。20K 的 dY dispatch
从 block-only 的 4006.110 µs 降到 1504.350 µs；8K 从 1165.470 降到
628.384 µs。阶段区间存在重叠，不能相加解释 wall。此时训练 FWD wall
四档为 327.081/376.645/1644.690/3562.870 µs；索引构造增加约 20–30 µs，
换回的 BWD 收益明显更大。日志在 `build/pf_static_backward_stripe/`。

EP8 T512 的四组梯度 hash 与原版本完全一致；T64 的 Torch parity、allocation
retry 和三个 live tape 测试通过。修复既有 eager benchmark 在不等长 source
下将 activation lease 的实际 token 数误填为 capacity 的问题后，EP8
T64/32、B16/D1、E64/K4 的训练 FWD `compute-sanitizer --tool memcheck`
完成，error-exitcode 为零。修复前两次超时不算 sanitizer 通过或 kernel 故障。

小规模的回退来自 M-block 数不足以铺满 cluster；后续只在
`mblocks < clusters` 时恢复固定 N/epilogue residue，将其与长文整块调度
分开，不回退动态 claim。最终矩阵另列。

### 激活梯度按占用率选择静态粒度

`mblocks < clusters` 时恢复独立的 dU/N、epilogue-row、dX/N 静态 residue，
其余仍由一个 cluster 完成整块链。没有 token 数或 EP 数硬编码，不改算术。
mini Notify 也保留原有的 out-of-range/duplicate 路由拒绝合同；既有 Torch
集成测试覆盖单 rank 非法路由后的集体报错及下一次正常调用。

`build/pf_static_backward_adaptive/` 的完整 native 矩阵如下。H2048/I512、E64/K4，
1620 MHz，PF B512/D8，30 轮、10 warmup；均为最慢 rank 的 wall p50（µs）。
EP8 在同一远端节点测量；EP16 同形状相邻对照，20K/64 反转 V3/PF 顺序。

| EP | tokens/rank | V3 FWD | PF FWD | V3 BWD | PF BWD |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 469.704 | 351.147 | 281.948 | 258.045 |
| 8 | 64 | 553.803 | 373.505 | 420.911 | 390.108 |
| 8 | 8192 | 1737.380 | 1646.090 | 2127.380 | 1758.660 |
| 8 | 20480 | 3586.140 | 3566.400 | 5066.530 | 4731.360 |
| 16 | 1 | 733.956 | 543.033 | 404.445 | 407.909 |
| 16 | 64 | 903.119 | 617.716 | 766.490 | 756.681 |
| 16 | 8192 | 3434.760 | 3813.370 | 2848.860 | 2961.910 |
| 16 | 20480 | 7213.590 | 8651.610 | 6982.980 | 6768.470 |

单机四档 BWD 已领先，但双机 8K BWD 仍慢 3.97%，双机 FWD 长档仍慢
11.02%/19.93%，没有达到全面领先。静态 dW 的独立 domain/chunk 分配在每卡
只有 4 个专家时留下明显不均衡尾部，后续改为按实际 padded K extent 加权切分
联合 dW2/dW1 流，每个 cluster 持有不相交区间，区间内保留连续 tile chunk
流水。空专家权重为零，热点专家不会被整个分配给一个 cluster；仍不做动态
claim 或 split-K，也不放松 dX 全完成的门槛。

此版本 native SHA256：
`9e4f08dc7435b04947e570e8be717a23e955f2f332aae208c15e9176fc07a05f`。
library：`f2cf1cc5693fa34c541d93f4682446a47dd26b70999fc584c330ef3c93e2e83f`。
EP8 T64 的 Torch parity、非法路由、allocation retry、三个 live tape，以及
T20480 的输出/四组梯度、后续 inference/Graph 生命周期检查通过。

额外用 `tools/benchmark_training.py` 测量真实 eager `forward → backward` 循环，
包含 Python 接口、tensor 分配/回收和末尾 stream synchronize，计时外做 rank
barrier；不是两段 native p50 相加。EP8 同远端节点、10 warmup + 20 measured：

| tokens/rank | V3 cycle (µs) | PF cycle (µs) | 耗时降低 |
|---:|---:|---:|---:|
| 1 | 3710.478 | 3189.257 | 14.05% |
| 64 | 3523.050 | 3306.644 | 6.14% |
| 8192 | 6971.879 | 6261.005 | 10.20% |
| 20480 | 13510.205 | 11215.811 | 16.98% |

cycle 基线使用本轮 stripe library 中保留的 V3/non-ring specialization，
显式 `--backend v3`，不是把旧 native 可执行文件当成 TensorRuntime extension。
日志分别为 `build/pf_static_backward_stripe/v3_ep8_training_cycle.log` 与
`build/pf_static_backward_adaptive/pf_ep8_training_cycle.log`。两种基准的输入生成、
分配边界不同，不能用它们相减推断 Python overhead；双机 cycle 尚待测量。

### dW 加权静态区间

按各专家实际 padded K 行数给联合 dW2/dW1 output-pair 流加权，每个 cluster
拥有连续、不相交的区间。相邻区间边界使用同一向上取整规则；空专家权重为零。
区间内沿用连续 tile chunk，不改 K 归约或 BF16 舍入，也不提前启动 dW。

`build/pf_static_backward_weighted/` 的 EP8 BWD 四档 wall 为
257.309/388.025/1819.440/4643.750 µs，均低于冻结 V3；相对上一版，8K 略退、
20K 改善。EP16 8K 为 2928.310 µs，仍比 V3 的 2848.860 µs 慢 2.79%，
加权分配不是剩余 activation-gradient 差距的解决方案。

该轮 EP16 脚本一度将 benchmark tokens 误填入仅 validation 使用的环境变量；
已按 JSON 的实际 shape 纠正：另外三次都是 8K 重复，不是 1/64/20K，日志改名
`pf_ep16_t8192_repeat{1,2,3}_backward.log`。不纳入四规模矩阵。

保持 B×D=4096，训练 FWD 的 B1024/D4 相比 B512/D8，EP16 8K 从
3832.900 降到 3613.320 µs，20K 从 8663.880 降到 7906.850 µs。
仍比冻结 V3 慢 5.20%/9.61%，因此没有将扩大窗口误记为全面领先，也不静默
改写默认参数；完整循环另用显式 B1024/D4 测量。

该轮 native/library SHA256 分别为
`f880e6297f30c98a81816e315f06642ed16ce17caeb8901ba2550194f3277943`、
`5ff101167e7f77785fa77e833ffe5c3c95318365994e524bc2e7615abe6faead`。
EP8 T64/T20480 Torch reference parity 已通过。EP8 初次 cycle 的 64 token
结果波动至 3870.830 µs，不能引用上一版更好的 cycle 作为本版最终值，待配对复测。

PF profile 的 `kRdmaCtas=0` 不包含非 ring 的多机 V3 Forward 回退；尝试用
同一 extension 测双机 V3 cycle 会在发起 Forward 时明确拒绝，不能视为性能
结果。双机 cycle 对照需单独按 `configs/static_inference_v3_20260923/` 构建
library/extension。冻结 native V3 的独立 FWD/BWD 结果不受此配置问题影响。

### 未采用：按本节点 source 优先的 M-block 排列

尝试将各专家的 M-block 交错，并从该节点的 canonical source 区间开始，
避免固定计算 owner 先等远端数据。`pf_static_backward_locality` 的 EP16
8K B512/D8 四组梯度 hash 与旧 Backward 相同，dU 区间缩短至 1353.340 µs，
但 wall 为 2957.510 µs，反而差于 weighted 的 2928.310 µs。

进一步让静态 Return 使用相同排列（`pf_static_backward_ordered_return`），
EP16 8K/20K B1024/D4 wall 为 3054.770/7361.010 µs。Return 区间变短并未
转换为端到端收益；不能只看阶段 span 就认定优化成功。两项排列及缓存 owner
状态均从最终代码移除，保留较简洁的 canonical residue + weighted dW。

额外正确性门禁：weighted EP16 T64/T20480 Torch 全部输出及四组梯度通过，
T64 还覆盖非法 route、分配重试与三个 live tape；EP16 T64/32 不等长 source
在 B16/D1 与 B1024/D4 下六组输入/权重/输出 hash 全部一致。
locality 的 EP8 T64/E16/K6/B16/D1 Backward memcheck launcher 退出码为零；
该用例只经过不受网络排列影响的本地调度。不将 memcheck 扩大解释为 racecheck
或完整多机 sanitizer 认证。

按原 V3 profile 新建的 TensorRuntime 对照库为
`build/pf_training_v3_reference/libweftmoe.so`，SHA256
`6d349d29f7120e00ae1225ac9ea8e0fa1423e6f18423828b3650c3b3c49812e1`。
EP8 20 warmup + 50 measured，V3 cycle 四档为
3398.326/3537.165/6416.721/11410.903 µs；weighted PF 复测为
2982.490/3259.546/6150.959/10848.984 µs，耗时降低 12.24%/7.85%/4.14%/4.92%。
长文收益小于先前同库 V3 回退对照，最终不再沿用那个较大的百分比。

### 最终静态 Backward native 矩阵

最终保留 weighted 版；EP8 B512/D8，EP16 B1024/D4（Forward-only 窗口消融
选择，Backward 不改变 K 归约），其余形状/频率/轮数同上。单位 µs：

| EP | tokens/rank | 冻结 V3 BWD wall | PF BWD wall | PF 耗时降低 |
|---:|---:|---:|---:|---:|
| 8 | 1 | 281.948 | 257.309 | 8.74% |
| 8 | 64 | 420.911 | 388.025 | 7.81% |
| 8 | 8192 | 2127.380 | 1819.440 | 14.47% |
| 8 | 20480 | 5066.530 | 4643.750 | 8.35% |
| 16 | 1 | 404.445 | 408.590 | -1.02% |
| 16 | 64 | 766.490 | 762.142 | 0.57% |
| 16 | 8192 | 2848.860 | 2931.490 | -2.90% |
| 16 | 20480 | 6982.980 | 6732.710 | 3.58% |

小于约 1% 的差异按接近看待，不表述成确定领先。8K 双机仍回退约 2.9%，
因此只完成了功能门禁，没有达成“所有规模 BWD 不低于 V3”。

### 双机完整循环：必须独立于 cached-state native 报告

同一 weighted PF 与原 V3 profile 的新 TensorRuntime extension，EP16
B1024/D4，20 warmup + 50 measured 的首轮真实 `forward → backward`：

| tokens/rank | V3 cycle (µs) | PF cycle (µs) |
|---:|---:|---:|
| 1 | 7151.888 | 5551.253 |
| 64 | 7187.492 | 4760.193 |
| 8192 | 9088.657 | 10038.992 |
| 20480 | 18300.539 | 23454.933 |

这组 host wall 抖动明显：PF 20K 单次范围 17414–185439 µs，各连续十轮的
中位数约 20.4/23.6/21.4/39.4/26.8 ms；不能把首轮 28.2% 的差距全归因于
kernel。新增 benchmark 的 Forward/Backward 接口分段计时（不改算子），
反转运行顺序，10 warmup + 20 measured 的长档复测如下。各列单独取最慢
rank p50，不能相加重建 cycle：

| tokens/rank | V3 FWD API | PF FWD API | V3 BWD API | PF BWD API | V3 cycle | PF cycle |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 4794.760 | 5141.901 | 4316.087 | 4443.026 | 9064.414 | 10004.030 |
| 20480 | 8542.127 | 10054.226 | 8444.019 | 8221.840 | 17210.427 | 18317.683 |

复测仍有约 10.4%/6.4% 的真实循环回退，稳定的主要差距在训练 Forward，
不能用单机收益、cached-state BWD 或 Notify 加速替代这一结论。原 V3 profile
与 PF profile 均保持各自生产 GEMM/通信参数，不把切换 profile 当成静态调度的
单项收益。

尝试仅绑定各 local rank 的调用线程到 CPU 8/10/12/14/56/58/60/62 并没有消除
抖动（PF 8K/20K 为 14301.164/20219.641 µs，V3 为 9357.899/17710.015 µs）。
未修改系统或 NCCL 的 CPU 亲和性，也没有改线上 Runtime；这个无收益的临时
benchmark 选项已移除。原始样本分别保留在 weighted 目录的
`{v3,pf}_ep16_cycle_node*.log`、`*_phase_node*.log`、`*_pinned_node*.log`。

保持 B×D=4096 再测 B2048/D2，EP16 8K/20K 训练 FWD 为
3807.160/8111.470 µs，均差于 B1024/D4 的 3613.320/7906.850 µs，未采用。
没有通过扩大 ring 默认值掩盖未解决的双机训练 Forward 差距。

复现完整循环使用 `tools/benchmark_training.py --backend pf|v3`，通过
`WEFTMOE_TENSOR_EXTENSION` 指向各自 profile 的 extension；EP8 用单机
`torchrun --standalone --nproc-per-node=8`，EP16 用两台机器各八进程的
`torchrun --nnodes=2`。PF 的 `--ring-tokens 512 --ring-depth 8` 对应单机，
`--ring-tokens 1024 --ring-depth 4` 对应上述双机对照。该工具只测 MoE 的
TensorRuntime FWD+BWD，不是完整模型、RL step 或包含 optimizer 的训练 step。

最终补充极端热点门禁：weighted EP16、T8192/E64/K4、B1024/D4，所有 source
均路由到 rank 0 的四个专家，rank 0 实际接收 524288 行，其余 rank 零入站。
Forward 输出和 dX/dGate/dW1/dW2 全部通过 Torch reference，随后 EP8 子组的
inference/Graph 生命周期检查也通过。日志为
`torch_ep16_hotspot_t8192_node{0,1}.log`；这验证了实际 tape 扩容、静态 dW
在严重倾斜下的完整覆盖以及 no-drop，而不只是均匀 benchmark。

最终又对单机完整循环做一次相邻、反转顺序（PF→V3）的 20+50 轮确认，
仍独占远端节点且调用线程不绑核。四档 V3/PF wall 分别为
3337.445/3158.278、3587.115/3325.652、6577.049/6212.620、
11247.264/10866.331 µs，降低 5.37%/7.29%/5.54%/3.39%。
两轮单机均领先，但不将不同轮次中更有利的值拼成一个表。
日志为 `build/pf_static_backward_final/{pf,v3}_ep8_cycle.log`，实际加载的
extension 路径已写入每个 JSON（weighted PF 和原 V3 profile）。

撤掉无收益候选后重新完整构建 `build/pf_static_backward_final/`，native SHA256
为 `6d512a8377b16c510868622910aa30a7d64fc581df22568e6d6a038d897dad2b`，
library 为 `5e1d15a9fc7d12741966d39971d295d95b270217a50ca80a2243ee1b11f7bda2`。
编译完成后、无编译干扰的 EP16 B1024/D4 长档 BWD 复测为
2919.860/6667.710 µs：8K 仍比冻结 V3 慢 2.49%，20K 快 4.52%，
与上述“8K 小幅回退、20K 领先”的结论一致。没有将构建产物哈希不同误当成
功能变更，也不把不同复测中最有利的数字拼成新完整矩阵。

最终 extension 又通过 EP16 **T20480 全热点**的完整 Torch reference：rank 0
实际接收 1,310,720 行，其余 rank 零入站；输出和全部四组梯度正确，随后
inference/Graph 生命周期检查通过。使用 B1024/D4，未改变 ring 容量来容纳
整段上下文；永久 tape/Backward 实际行数工作区则按需扩容。日志为
`build/pf_static_backward_final/torch_ep16_hotspot_t20480_node{0,1}.log`。

## 长文差距复核与下一轮测量边界（2026-10-01）

后续只验收 T8192/20480：单机 EP8、双机 EP16 的训练 Forward、Backward 和
实际 FWD+BWD cycle 都要优于 V3，共十二个比较项。已有单机收益不能替代
双机验收；`tools/benchmark_training.py` 默认也改为这两个长档。

### 原始日志暴露出的规划成本

重新解析上述 adaptive 目录的 `v3_ep16_t*_forward.log` 与 weighted 目录的
`pf_ep16_t*_b1024_forward.log`，没有运行新的 GPU 实验。双机汇总脚本原先
遗漏了 native 已输出的 `device_kernel`、`route_exchange`、`before_combine`、
`after_combine`；现在保留这些字段，旧二进制没有对应字段时仍可汇总。

单位 µs，仍是每轮最慢 rank 再取 p50；各列独立统计，不能直接相加减：

| T/rank | 版本 | Notify | device kernel | main 内规划/交换 | 去除规划后的 device 时间 | main 末尾 rendezvous 区间 |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | V3 | 468.316 | 2852.860 | 0 | 2852.860 | 未单独计时 |
| 8192 | PF | 184.574 | 3334.660 | 397.856 | 2944.064 | 313.088 |
| 20480 | V3 | 508.014 | 6570.500 | 0 | 6570.500 | 未单独计时 |
| 20480 | PF | 187.504 | 7621.410 | 555.648 | 7072.834 | 338.048 |

“去除规划”先在**同一 rank、同一次调用**内计算
`device_kernel - route_exchange`，然后取该轮 rank-max 和跨轮 p50，不是
两列 p50 相减。V3 的路由规划在 Notify，不是没有规划成本。
另一个口径差异是旧表的 `forward`：V3 使用 CUDA event 的总时间，PF 使用
device status 的 kernel 时间。比较 kernel 本身应使用这里共同的
`device_kernel`，端到端验收继续用原来的 wall/cycle。

由此可以定位、但还不能完全归因：8K 的 device 差距主要集中在 main 内规划；
20K 去除规划后仍有约 502 µs 差距，不能只优化 planner 就宣布目标达成。
PF Dispatch/FC1/FC2/Return 的整段跨度包含等待且互相重叠，不代表实际 GEMM
用时翻倍。末尾 `after_combine` 从 CTA 0 的通信角色到达开始计时，还包括
等待其它 CTA 和其它 rank 的尾部，不能把 313/338 µs 全算成网络 barrier 的
纯指令开销，更不能直接删除保护下次调用元数据复用的 rendezvous。

### 源码证据与诊断补齐

V3 训练也会保存 X 和 G/U，不能将差距归因为 PF 独有的整份 checkpoint
流量。实际差异是写回几何和指令路径：V3 完整 M 块的 SavedX 连续 TMA
写回；双机 PF 的 ring 物理行与 canonical tape 行不同，SavedX 使用 math
线程的 uint4 scatter，G/U 同样按 canonical 行映射写回。它们与 WGMMA
共享 stage 的读生命周期，必须全部读完才能归还 empty。单机的连续行 TMA
路径不能无条件搬到双机。

现有 timeline 的 `RingPlanBegin` 在 `build_training_rows` 之后，无法直接
看见训练行号重建；`RingPlanSourceReady` 又落在训练 packet 重建之后。
schema 13 增加 `TrainingRowsBegin/End`、`TrainingPacketsBegin/End`，并将
SourceReady 放回 source 目录就绪的边界。分析工具分别报告 `training_rows`
和 `training_packets`；后者包含在新的 `ring_plan_packets` 区间中，不能重复
累计。旧 schema 的 source/packets 分段边界不同，不能直接作为单项对照。
所有事件均受 `WEFTMOE_DEVICE_TIMELINE` 编译宏控制，不增加生产同步或任务。

下一轮先拆清 canonical 行重建、训练 packet 重建、ring 目录交换、destination
几何这四项，再决定 planner 改动；随后比较 retained/non-retained 的 math 与
epilogue，并检查 Return credit 引起的计算等待。历史 GC/GD 的 source 分组/
M16 TMA、AJ 的 wait1、GK 的两 K-stage group 合并都没有稳定收益，不能把
这些旧实验重新包装成已证实的优化。Backward 8K 还需独立处理，保持全部
本地 dX 完成后才进入 dW 的约束。

本轮环境没有暴露 `/dev/nvidiactl`、`/dev/nvidia0`，`nvidia-smi` 无法访问
驱动；SSH 创建 socket 被拒绝，`.git` 为只读。上述结论来自已保存日志和
源码，不是新的性能结果；未修改生产计算/通信算法，也未提交新的性能收益。

离线验证：PF profile 的 timeline native 完整编译通过，产物为
`build/pf_training_timeline/test_cuda_notify`，SHA256
`adce0eb6a77047bf6b50ab260d274e3bcb94e9176d451814db3beccbd4cf3f28`。
四份 EP16 长文原始日志重新汇总后，全部旧 JSON 指标逐项不变；缺少新增字段
的旧格式兼容检查、新事件区间解析及 Python 语法检查通过。未新增细碎单测，
未运行 GPU 性能或数值验证；诊断版不能代替生产二进制做最终性能验收。

继续核查 planner 的数据依赖：`build_training_packets` 只读取本地 source
目录、selected experts 和 canonical route 行号，写入 Notify lease 的
Backward packet/fanout、packet order 及 runtime counts。Forward 的
`build_dispatch_packets`/`build_return_packets` 使用独立的 ring 目录；训练
packet 重建的输出不在 `make_source_metadata_layout` 的 published 区间中。因此下一项
可测候选是将训练 packet 重建移到 metadata Put 提交后、对应 context 的 Put
fence 之前，尝试隐藏这段重建时间。仍需保持全体 CTA 的一致 collective
顺序和 scratch 生命周期，不删除 fence，也不延迟到 main 返回之后。
这是源码依赖分析，尚未改动调用顺序或验证收益；GPU/SSH 访问仍受限。

### 环境恢复后的 planner timeline

2026-10-01 恢复 GPU、SSH 和 Git 写权限，两节点各八张 L20X 均空闲且
SM 时钟为 1620MHz。使用上述 schema 13 诊断版，EP16/B1024/D4，保留
训练状态，12 轮（10 warmup），两个长档全部完成。各 rank 最后一次调用
的区间时长取跨 rank 中位数，单位 µs；只用于定位，不作为生产性能：

| T/rank | 训练行号 | 训练 packet | source 规划 | packet 规划总段 | 目录交换 | destination 几何 | 行映射绑定 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 60.896 | 27.616 | 86.384 | 69.856 | 59.328 | 64.512 | 42.592 |
| 20480 | 75.824 | 31.600 | 115.360 | 78.944 | 103.648 | 87.488 | 84.976 |

训练 packet 包含在 packet 总段内，不能重复相加。32 份 trace 均无 overflow；
文件为 `build/pf_training_timeline/ep16_t{8192,20480}.rank*.forward.json`。
首次具体消融选择缩小训练行号 radix sort 的 bit 范围：输入按 route ordinal
递增，已安装 CUB 的 BlockRadixSort 保证稳定性，只排 expert 位即可保持完整
行序；随后再独立测训练 packet 与目录 Put 的重叠，不混淆两个改动的收益。

### planner 分项消融（2026-10-01，继续优化中）

恢复环境后重新测生产二进制，H2048/I512/E64/K4、EP16/B1024/D4、
1620MHz，30 轮中前 10 轮预热。下表均为每轮 rank-max 后的 p50，单位 µs。
`base reverse` 是最后一个 overlap 候选之后反转顺序复测的 c6be309；
相邻轮次的 Notify/计算区间仍有波动，不能把 wall 的全部变化归因于 planner。

| 累计候选 | 8K wall | 20K wall | 8K main 规划 | 20K main 规划 |
|---|---:|---:|---:|---:|
| 原 V3 profile | 3433.490 | 7161.700 | 0（规划在 Notify） | 0 |
| c6be309，初次恢复 | 3558.880 | 7912.890 | 392.512 | 555.008 |
| A：只对 expert 位做稳定 radix sort | 3575.720 | 7954.880 | 390.688 | 553.024 |
| B：训练 packet 与目录 Put 重叠 | 3565.720 | 7898.460 | 363.712 | 520.928 |
| c6be309，base reverse | 3596.000 | 7949.260 | 393.504 | 555.776 |
| C：全网格静态分摊 RX 描述符 | 3566.460 | 7893.650 | 341.920 | 503.872 |
| D：训练表按 source-route 顺序合并写入 | 3533.150 | 7833.880 | 336.128 | 499.552 |

A 利用 CUB BlockRadixSort 的稳定性，输入 ordinal 原本已经递增，因此无需
再排序低位 ordinal。规划只改善约 2 µs，没有独立的端到端收益证据。
B 的 packet/fanout 重建不属于发布给 Forward 的源目录，移到目录 Put 提交后、
context fence 之前，保持原有 fence 和 collective 顺序；另外单机训练跳过
Backward 不使用的 node-token 目录。C 把每窗口一个 CTA 内的描述符循环改成
整个 grid 的固定 residue，没有领取队列或新增同步。D 将稳定排序所得 ordinal
暂存回原始 route 下标，再由相邻 lane 写相邻 route 的元数据；复用原 shared
scratch，新增两次 CTA barrier，无新增分配。D 仍是候选，尚未完成所有门禁。

A/B/C/D 的两个长档均与原版保持 16 ranks 的输入、route、gate、实际权重及
输出 hash 一致。B 另通过 EP8 两个长档的重复 Backward 验证：所有 rank 的
dX/dGate/dW1/dW2 hash 与 c6be309 逐项一致。不能据 Forward benchmark 的输出
hash 推断 C/D 的梯度已经验证。日志统一保存在 `build/pf_training_timeline/`，
前缀分别为 `v3_restored`、`base_restored`、`stable_sort`、`overlap`、
`base_reverse`、`parallel_pull`、`coalesced_rows`。

本轮单机独占远端节点的恢复基线，EP8/B512/D8，FWD V3/PF 分别为
1724.780/1640.600（8K）、3573.070/3556.640（20K）；BWD 为
2132.490/1818.290、5067.280/4721.710。20 warmup + 50 measured 的实际 cycle
V3/PF 为 6521.193/6294.860、10977.895/10866.903。20K FWD/cycle 优势很小，
仍须最终配对复测，不能沿用历史轮次更大的收益。

D 补充 EP8 重复 Backward 验证已通过，两个长档四组梯度 hash 均逐 rank
等于 c6be309。独占远端节点的反向顺序 FWD 复测，原版→D 的 wall 为
1640.210→1629.740、3588.240→3554.950 µs，规划为
173.568→163.424、291.264→284.032 µs；日志 `base_confirm_ep8_*` 与
`coalesced_confirm_ep8_*`。这支持小幅收益，不表述成 GEMM 提速。

未采用 E（`pf_plan_node_reservations`）：source node 计数/预约以
`match_any` 合并同 warp 的原子操作；不改变每 token 每命中节点一个记录的
语义，但 EP16 规划为 347.520/500.864 µs、wall 3537.020/7908.970 µs，
未优于 D，源码已撤回。不存在“减少原子数必然更快”的采用结论。

同日重新测 EP16 cached-state BWD，V3/PF c6be309 的 wall 分别为
2851.220/2917.230（8K）、6996.810/6696.970（20K）。8K 仍回退 2.32%，
20K 降低 4.29%；不能用其它旧轮次掩盖 8K 未达标。日志前缀
`v3_bwd_restored`、`base_bwd_restored`。现有 Forward timeline 的一个 rank
还显示 20K 每 CTA 平均约 5.21ms 落在 select/wait 区间，FC1/FC2 tile
区间合计约 1.31ms；事件插桩会扰动绝对耗时，这只说明不宜将整个差距
归为纯 GEMM 算力不足，后续仍须用生产 wall 验证候选。

### F：在规划中写完永久 tape 的 route header/gate

之前每条训练 route 的 72-byte `LocalP2pRouteHeader` 与 gate，都在 X Pull 的
issuer lane 写入。它们只依赖源 rank、route、expert、canonical 行号和本次
epoch，不依赖 X 到达，也不属于可复用的 payload slot。因此把写入移到已有
destination 行映射构建循环，复用该处已经解出的值；原有规划 join 发布它们。
Pull 保留物理 ring gate/X 的写入和完整 completion/credit，不能把物理 slot
的 gate 也提前写入，否则会覆盖前代。FC1 的 SavedX/GU 保存方式没有改变。

这不是减少永久 tape 的字节数，而是将不可变元数据从单 lane 的串行数据面
移到并行控制面。删除原 `save_identity` helper 和网络 Pull 中为它广播的
block/index；Pull 不再因训练而生成另一份函数实例。无新增 kernel、分配、
全局同步或任务队列，inference 的空缓存合同不变。

EP16/B1024/D4、1620MHz、10 warmup + 20 measured；先 F 后 D 的反向对照，
单位 µs，rank-max p50：

| T/rank | D wall | F wall | 耗时降低 | D main 规划 | F main 规划 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 3530.850 | 3312.190 | 6.19% | 337.920 | 337.888 |
| 20480 | 7870.230 | 7108.690 | 9.68% | 500.032 | 504.544 |

F 的 device kernel 为 3035.710/6816.900 µs。相比同日恢复后的 V3 wall
3433.490/7161.700 µs，降低 3.53%/0.74%；20K 领先仍小，必须最终配对复测，
此处不宣布十二项全面验收。两版全部输入/route/gate/实际权重/输出 hash 一致。
F 和 c6be309 又各自完成 EP16 T8192/20480 重复 Backward 验证，全部 16 ranks
的 dX/dGate/dW1/dW2 hash 逐项一致，repeat_equal 均为 1。该数值检查与本地
编译并行，**不引用其 wall 作性能证据**。完整 Torch/实际 cycle 尚待新库验证。

冻结 native `build/pf_plan_checkpoint_headers/test_cuda_notify`，SHA256
`3fded55c7ee9857c2166a172adb7a950239ff966b9ce31042e03f17b721eadf4`。
性能日志前缀 `checkpoint_headers`、`coalesced_reverse`；梯度日志
`{pf_plan_checkpoint_headers,pf_static_backward_final}_ep16_t*_validation.log`。

F 的单机补充：独占远端节点、B512/D8 的 FWD wall 为 1543.200/3334.340 µs，
明显优于刚才 D 的 1629.740/3554.950 µs（降低 5.31%/6.21%）。规划为
164.544/288.224 µs，未减少规划工作；SavedX/GU 算法仍不变。两个长档的
EP8 重复 Backward 也全部通过，与 c6be309 的四组梯度 hash 逐 rank 相同。
本轮采用 A–D 与 F 作为 planner/immutable-header 里程碑，E 已撤回；新的
Backward ready-residue 实验单独记录，不包含在此项收益内。

### F 后的窗口消融与静态 Backward 就绪探测（待最终验收）

移除 Pull 的串行 header 工作后，重测生产二进制的窗口参数；仍 EP16、
1620MHz、30 轮/10 warmup，单位 µs。这里 B 是每源 token 窗口，D 是
source contribution 深度，不是 GEMM M 或网络 packet ring depth。

| B/D | 8K wall | 20K wall | 8K main 规划 | 20K main 规划 |
|---|---:|---:|---:|---:|
| 1024/4 | 3312.190 | 7108.690 | 337.888 | 504.544 |
| 512/8 | 3311.680 | 7374.730 | 309.408 | 525.312 |
| 768/6 | 3246.660 | 7011.360 | 325.408 | 500.288 |

三个配置的六组输入/权重/输出 hash 均一致。768/6 的两个长档 device pool 为
795434176/912494784 bytes/rank，1024/4 为 870260928/987174592；总 pool
少约 74.8/74.7 MB。其 registered 部分反而多约 8.45/8.47 MB，因为 source
contribution 容量稍大；compute ring 的较小最坏窗口节省了更多空间。
永久 tape 与调用级 Backward tensors 不包含在这个 pool 数字中。没有新增
运行时 shape 阈值或模型专用分支；最终还需反向计时和同参数实际 cycle。

G（`pf_backward_ready_residues`）只改变 activation 链的访问顺序：cluster
仍独占 `c + n*clusters`，但原下一个 block 未 ready 时，会在**自己的**
residue 中探测 phase=1 的 block；不领取其它 cluster 的任务。dW 仍严格等待
全部本地 dX，canonical tape/dW K 顺序不变。EP16 两个长档的重复梯度 hash
均与 c6be309 完全相同。

| T/rank | 原静态 BWD wall | G wall | 原 dU/epilogue 区间 | G dU/epilogue 区间 | 原 combine 区间 | G combine 区间 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 2917.230 | 2898.330 | 1700.580 | 1380.740 | 187.552 | 414.560 |
| 20480 | 6696.970 | 6878.380 | 4284.800 | 3863.010 | 216.288 | 940.224 |

各区间独立统计且互相重叠，不能相加恢复 wall。G 单独没有全面收益，不采用
为最终版本：计算就绪更早，并不等于固定顺序 Return 能及时消费它。
下一候选 H 同时让每个 Return owner 在自己的固定 residue 内探测完成块，
用既有 phase=4 表示本 owner 已访问，不增加 CAS/领取队列；实际 returned
计数仍在 copy/publish 后推进。此时 H 尚未完成构建、数值或性能验证。

768/6 补充的 EP16 T8192/20480 重复 Backward 已通过，全部四组梯度 hash 与
c6be309 的 B1024/D4 逐 rank 相同；日志 `headers_b768d6_ep16_t*_validation.log`。

H 已构建 native/library/extension，EP8 两个长档四组梯度 hash 与 c6be309 一致。
EP16/B1024/D4 的 wall 为 2899.490/6878.910 µs，combine 区间降到
123.136/118.624 µs，但总耗时基本等于 G，仍不如原静态调度的长档组合。
**G/H 均已撤回**，没有只因为阶段 span 变短就采用，也未为 H 执行完整 Torch
或实际 cycle。资源用量核对中，原版/G/H 的静态 unified BWD 均 REG168、
SHARED2048、LOCAL0，网络 STACK208/本地 STACK0；没有发现新增 spill 的证据。

H native 的 Forward 代码仍是 F。它与原 V3 的相邻长档对照，PF 用 768/6，
V3 保持原 profile：8K wall 3243.280 vs 3437.380，20K 为
7089.200 vs 7180.770 µs，降低 5.65%/1.28%。这是 Forward 的再次确认，
不能作为 H Backward 的采用依据。日志 `pf_final_t*_forward.log`、
`v3_final_t*_forward.log`；此处的 `final` 是当轮文件标签，不代表目标完成。

下一候选 I 基于 F 的原静态计算/Return 次序，尝试在通信暂时无可做工作时
非阻塞 Reduce：每个 warp 保持原 token residue，只在全部 top-k ready 后
执行原 FP32 累加及 BF16 输出，不等待缺失 contribution，不领取其它 warp
的 token。单独的 shared pointer table 避免覆盖仍在使用的 TMA staging；
通信结束后从各自 cursor 继续处理剩余 token。尚无收益结论。

I 的 EP16 两个长档已通过重复反向验证，16 ranks 的四组梯度 hash 均与
c6be309 相同。B1024/D4 的首轮 wall 为 2862.840/6676.730 µs，相比同日原静态
版本 2917.230/6696.970 降低 1.86%/0.30%；8K 仍略高于 V3 的 2851.220。
静态 unified kernel 的 REG168、SHARED2048、LOCAL0 和 STACK 用量未变化，
额外 pointer table 属于 dynamic shared memory。`reduce` 时间戳现在包含与
通信重叠的就绪等待区间，不能拿其变长推断归约算术变慢，也不能与其它阶段
相加。日志 `stream_reduce_*`；单机与反向顺序复测尚未结束，不作正式采用结论。

J（`pf_backward_activation_chain`，在 I 上消融）：当 M-block 数足够时，
静态 cluster 已独占整块 dU→SwiGLU/dGate→dX，三段不必重新交换任务描述符。
调度器一次返回整块任务，沿用原 GEMM、epilogue 和逐阶段数据完成同步，
只省去中间两次调度回环/DSM mailbox 往返。欠填充的 N/epilogue residue
路径不变；dW 仍严格等待所有本地 dX。此时仅开始构建，尚未验证或采用。

I 补充 EP8 两个长档的全部梯度 hash 与 c6be309 相同。独占远端节点 I→原版
的相邻 BWD 消融为 1789.120→1821.890（8K）、4733.480→4670.430（20K）；
I 并非单独全面有收益。J 在相同节点为 1776.220/4611.050。J 的 EP16
两个长档全部梯度 hash 也与原版一致，资源用量仍无新增 spill；其 B1024/D4
wall 为 2869.660/6654.510，相邻 V3 为 2839.690/6990.520。8K 仍差 1.06%，
不能用更早稍慢的 V3 轮次宣布达标。

J library 的首次 EP16 实际 cycle（PF B768/D6，20 warmup + 50 measured）为
14642.389/20479.200 µs，随后 V3 为 10288.255/19187.677 µs；FWD API 各为
5791.449/9046.129 vs 4808.787/8891.286，BWD API 为
7223.399/9150.568 vs 5067.943/9399.583。主机延迟抖动明显，保留完整样本，
不裁掉大值，也不把 native cached-state 的小幅收益当成真实循环已通过。
日志 `activation_chain_*`、`v3_chain_pair_*`。

K（`pf_training_single_agreement`）：mini Notify 已提供准确 R/P，Runtime 可
在现有输入/权重准备的同一次 CollectiveTensorPreparation 内分配 saved tensors。
删除旧独立 planner 时代留下的延迟分配 callback；传入已绑定 tensors 时，
backend 不重复进行分配确认。未绑定 tensors 的原生入口仍在 backend 内分配
并集体确认，任何路径都没有在 main 中分配。应用自己准备 buffers 时本来就
必须在进入 Forward 前一致处理分配失败，否则旧的第二次确认同样救不了
没有进入 Forward 的失败 rank。该候选需要重新验证分配失败、重试与 live tape。
不改变 device planner、GEMM、ring 及输入验证；尚无性能结论。

L（`pf_backward_dispatch32`）在 K 上单独将既有 Backward dispatch batch 参数
从 64 调成 32，让通信 warp 更频繁返回 GIN/Return 进展循环；所有权仍为静态
residue，没有新增动态调度。只改构建参数，不增加规模阈值；尚未验证或采用。

J 的 EP8 两个长档也完成四组梯度 hash 对照，全部等于 c6be309。其真实单机
cycle 为 6071.962/10648.123 µs，相邻 V3 为 6592.669/11656.652，降低
7.90%/8.65%。该收益属于 F+I+J 的累计结果，不全部归因于 J 的 mailbox 简化。
I+J 作为小的静态 Backward 里程碑保留；双机 8K 与真实 cycle 仍未达标。

K 的 EP16 T8192 已通过既有 Torch 集成测试：输出及四组梯度、非法路由拒绝、
单 rank 前后向分配失败与重试、三个 live tape/热点切换、后续 EP8 inference
graph/offload。T20480 正在验证。L 的 EP16 两个长档四组梯度 hash 已全部
等于原版；这里的正确性任务可与 CPU 构建重叠，其 wall 不作为性能数据。

M（`pf_backward_peer_reuse`，基于 K，仍用 dispatch batch 64）：Backward
通信 workspace 在所有 rank 的实际 rows/routes 均不变时，复用原地址和映射。
用分配确认的同一个 MIN 状态字编码 failure-rank 优先级与 changed 位；一旦
有 rank 分配失败，全体仍抛错，只有全体成功且无人变化才跳过地址 AllGather
及第二次 import 确认。任一 rank 改变尺寸仍走完整交换和旧映射退休协议，
不能凭本 rank 未变就自行跳过 collective。exact-shape 策略不变，不保留历史
热点的大 payload。稳态新调用的这部分主机 collective 从三次减到一次；
不是删除 main 的通信完成同步。尚未验证或测得收益。

K 的 EP16 T20480 完整 Torch 检查也通过；两个长档每项有 16 个 rank 的
parity/retry/rejection/graph 完成标记，以及 48 个 live-tape reuse 标记。
EP8 T8192 的 FWD+重复 BWD 通过 compute-sanitizer memcheck，梯度 hash 与
原版一致。首次启用 API 错误报告时，NCCL 初始化的 `cudaFuncGetAttributes`
和 `cudaGetLastError` 返回 `cudaErrorNoKernelImageForDevice` 被记为 41 次 API
错误；保留该失败日志。重跑仅关闭 API 错误报告（`--report-api-errors no`），
内存访问检查保留且退出码为 0；不宣称首次检查通过或所有 CUDA API 无错误。

K 独占远端节点的 native FWD 为 1499.550/3254.340 µs，前一版本 F 为
1543.200/3334.340；device GEMM 不变，收益来自避免已绑定 tensors 的重复确认。
实际 EP8 cycle 为 5942.059/12007.974 µs，20K 主机波动明显、没有全面领先，
继续保留全部样本并测 M，不根据单次快的循环作最终验收。

L 的 EP16 BWD 为 2856.090/6662.820 µs，仍未超过相邻 V3 的 8K。
其 EP8 为 1751.040/4755.690，20K 差于 J 的 4611.050，**不采用 batch32**。
K 的 EP16/B768/D6 native FWD 为 3165.920/6988.520 µs；相比相邻 F 的
3243.280/7089.200，主机重复确认的减少与 wall 改善方向一致。

真实 EP16 cycle 的 K→M 首轮为 11529.613→8639.629（8K）、
16436.454→21512.953（20K）µs。M 没有改变 Forward kernel，但 20K 的 FWD API
也从 8328.170 波动到 9964.848，不能把两个独立进程组间的全部差异归于
Backward 缓存策略，更不能挑较快的一个规模拼出“全面通过”。后续确认循环
统一增加到 50 warmup + 200 measured，PF/V3 使用相同配置、保留所有样本。

N（`pf_backward_receiver_reduce`，基于 M）：仅静态网络 BWD 的 Reduce token
固定分配给 receiver CTA 的 warp。它们能在通信存活时非阻塞归约；sender
warp 继续只做 pack/submit，不再承担必须等全部 TX drain 后才开始的 token。
本地与旧动态路径不变，token 内算术顺序不变。尚未验证或采用。

O（`pf_backward_owner_wait`，在 N 上消融）：静态 cluster owner 暂时没有可做
工作时，只有 owner lane 原地探测固定依赖，其他计算线程停在已有 compute
join；通信 warp 不参与该 join，仍能提供缺失数据。省去空任务的反复 DSM
mailbox 往返，不做跨 owner 领取，也不提前 dW。临时构建开关仅用于 N/O
独立消融，采用后须删除实验开关。尚未验证或测得收益。

M 的 EP16 T8192/20480 完整 Torch 集成检查均已通过，与 K 一样每规模
16 ranks 的四组梯度 parity、分配失败重试、非法路由恢复和 48 个 live-tape
reuse 检查均通过。热点切换包含本 rank 尺寸不变、其它 rank 尺寸改变的情形，
证明不能省略的全局 changed 确认没有被本地缓存命中替代。保留 M 作为已验证
的控制面复用里程碑；200 样本的最终性能验收仍在进行，不能从集体操作数的
减少推断所有规模已达标。

N/O 的 EP16 8K/20K 四组梯度 hash 均与 c6be309 一致，repeat_equal=1，
但相邻 BWD wall 消融（µs）为 M 2866.660/6679.420、N 2873.980/6715.170、
O 2868.700/6713.820；V3 为 2838.730/7006.340。Reduce 尾部缩短未转化成
总时间收益，两个候选均不保留，临时 owner-wait 开关已删除。

M 的 50 warmup + 200 measured 实际循环确认（不裁剪慢样本）：

| 拓扑 | tokens/rank | M wall µs | 相邻 V3 wall µs |
|---|---:|---:|---:|
| EP8 单机 | 8192 | 5914.704 | 6458.674 |
| EP8 单机 | 20480 | 10614.076 | 10986.694 |
| EP16 双机 | 8192 | 9630.475 | 9243.788 |
| EP16 双机 | 20480 | 16630.619 | 17310.825 |

单机两个长档与双机 20K 均领先，但双机 8K 尚未通过。双机 8K 的 PF/V3
FWD API p50 为 4613.630/4795.851，BWD API 为 4288.800/4327.790；各自
分位数不能相加代替 total 的分位数。完整循环中仍有主机尾部抖动，不用较快的
单项或旧轮次掩盖这次 total 退步。日志 `peer_reuse_ep8_cycle_200.log`、
`v3_peer_pair_ep8_cycle_200.log`、`peer_reuse_200_ep16_cycle_node*.log`、
`v3_peer_pair_200_ep16_cycle_node*.log`。

P（`pf_backward_activation_k128`）：激活梯度 dU/dX 改用 K128、两级 TMA，
dW 保持 K64/三级。拆开两阶段的共享内存 stride，复用同一 arena，不把 dW
也扩大成 K128；激活的 A 按两个 K64 swizzle slab 寻址，B 的 MN-major TMA
box 扩成 K128。每个输出仍按原 K16 WGMMA 顺序累加，BF16 checkpoint 不变。
该候选仅开始构建，须验证全部梯度 hash、资源用量及两种拓扑两个长档性能，
不能仅凭 WGMMA wait 次数减半判断有收益。

P 的 EP16 两个长档四组梯度 hash 与 c6be309 完全相同，repeat_equal=1；
静态 unified kernel 仍 REG168/SHARED2048/LOCAL0，GIN STACK208。
但 BWD wall 为 2869.920/6681.320 µs，基本等于 M 的 2866.660/6679.420；
EP8 为 1793.470/4637.070，也未稳定超过 J/M 的算术实现。EP8 实际循环
200 样本为 5852.751/10505.897 µs，不能将这一小幅主机差异归因于 GEMM。
**不采用 P**，描述符、共享内存拆分、调参项均撤回；完整 Torch/sanitizer
门禁未继续。检查既有 autotune 单测时发现两个默认宏检查漏列已有 GIN 参数，
与 K128 项无关，单独记录，不能声明该轮单测全通过。

Q（`pf_backward_dispatch_m128`，回到 M 后独立消融）：训练 main 生成的
Backward 本地 dY 发送置换从 expert 间 32-row stripe 改为 128-row stripe，
与激活计算消费块对齐。保持 expert 间交错，不一次排空某个 expert；只改变
传输遍历顺序，不改变 canonical 行号、tape、K 累加或静态 compute ownership。
候选开始构建，待验证和测量；不同时保留 P 的任何变更。

Q 的 EP16 两个长档四组梯度 hash 与 c6be309 一致、repeat_equal=1。
BWD wall 为 2860.570/6740.220 µs；8K 仅略好于 M，20K 退步，**不保留**。
生成的本地 dY 置换恢复 32-row stripe，没有将这次尝试变成新配置项。
上述 autotune 测试漏列已有 GIN 参数的两处正则已修正，既有 14 项全部通过，
未新增测试用例。

R（`pf_backward_owned_completion`，独立基于 M）：整块激活链只有一个
cluster owner，省去 dU/dAB 的中间全局完成计数及 phase2/3 发布；dX 由该
owner 直接 release 块级 ready，不再先 fetch_add。欠填充的分裂任务仍保留
原计数和 phase 协议，TMA store、阶段间 cluster join、全本地 dX 完成门槛
以及 Return ready 均不变。静态 dW 用自身区间终止，不再为已无消费者的
全局 dW completed counter 做原子操作；每 CTA 缓存 dW 首尾时间戳，退出前
汇总，保留整体计时但避免逐任务争用。候选开始构建，尚未验证或采用。

R 的 EP16 8K/20K 四组梯度 hash 与 c6be309 一致，repeat_equal=1，资源无新增
spill。BWD 为 2867.270/6678.090 µs，与 M 持平；实际 200 样本循环为
8677.758/16346.678 µs，优于相邻更早 V3 的 9243.788/17310.825，但 native
未改善、Forward 也未改，不能将全部循环差异归因于同步删除。**不保留 R**，
避免为没有实测 device 收益的计时汇总引入额外共享状态和控制分支。

S（`pf_backward_row_prefetch`，重新基于 M）：Backward 六处 payload 搬运
（本地 Dispatch/Return、GIN Dispatch/Return pack、两种 Receive）复用
Forward 的 RowPrefetchPipeline/FanoutRowPrefetchPipeline，使用两个 4KB
staging slot。仍在 payload 全部 global-complete 后发布 ready/释放 RX credit，
TX 的本地完成及远端 credit 均保留。Reduce 改为 warp shuffle 广播每 lane
持有的 canonical source pointer，省去 768B shared pointer table；既有 H
必须为 256 的倍数，保证每个 vector round 的 full-mask shuffle 都有完整
warp。释放的 shared memory 让双槽搬运仍有机会装入原单 CTA 资源上限。
正在构建；需要严格检查资源、数值/生命周期和性能，尚未采用。扩充既有
BACKWARD_PROGRESS 日志的设备首尾时间戳，只在诊断开关开启时输出，不增
device 插桩或正常路径通信。

S 的 EP16 两个长档四组梯度 hash 与 c6be309 一致、repeat_equal=1，但 BWD
wall 为 2974.690/7078.530 µs，EP8 为 1872.880/4958.270 µs，均退步。
EP16 8K 的 Dispatch/dU/dX 虽加快，绝对时间戳显示 dW 约 1.98 ms 已结束，
最终 Reduce 却到约 2.8 ms 才结束；单机也回退，不能直接归因于网络排空。
**不采用 S**，双槽复制与 shuffle Reduce 均撤回，没有改变默认参数。
下一步在 M 上增加 sender/transport retirement 时间点，区分通信收尾与
最终归约尾部；诊断 wall 含进度轮询，不当作性能结果。

M 的 retirement 诊断（`retirement_ep16_t8192_profile.log`）显示：dX 完成于
1.70–1.81 ms，dW 于 2.05–2.17 ms，GIN sender retirement 于 2.46–2.59 ms，
通信域退出于 2.64–2.67 ms，最终 Reduce 于 2.68–2.73 ms。各 rank 使用
自己的 start 作原点；不能把跨 rank 时间戳相减。主要尾部在 GIN，其后的
机内 End/credit 排空也有约 0.1 ms 串行成本，剩余 Reduce 算术并非主要原因。

T（`pf_backward_early_close`）：静态 Return owner 的最后一个 M-block
发布完成后，立即向本 CTA 的各本地 peer 发送 End，publisher warp 继续接收
通知；GIN/Dispatch 仍可运行。通信主循环退出后再 join 所有 incoming End、
consumed、outgoing credit，并执行原 GIN flush。把共享通知 helper 分为
close/wait，Forward 和旧动态路径仍通过原 drain wrapper 顺序执行两步。
无新参数、无新增 payload buffer，不改变 dW 门槛、梯度运算或 credit 语义。
正在构建，尚未采用；必须覆盖热点/零入站及连续复用，不能只测均匀路由。

T 的 EP16 两个长档四组梯度 hash 均与 c6be309 一致，repeat_equal=1，
但 BWD wall 为 2872.930/6694.030 µs，未超过 M。**不采用 T**，close/wait
拆分与提前 End 分支全部撤回，没有继续其完整 Torch 门禁。约 0.1 ms 的
transport retirement 区间不能全部归因于 End 通知；改变发 End 时机没有
带来总耗时收益。

U（`pf_backward_local_vector`，重新基于 M）：只改变同一 GPU 内的 GIN TX
Dispatch/Return 打包及 Return RX→contribution 放置。由整个 copy warp
执行 16B 向量 load/store，复用一个 row helper，不再为这些本地搬运逐行
等待单 lane TMA staging。跨 GPU 的 dY fanout、本地 Dispatch/Return 仍走
原 TMA；所有线程完成 system fence 和 copy-domain join 后才发布 packed/
ready、释放 RX credit。共享内存容量、通信包和计算顺序均不变。补充最后一个
Return terminal Put/Receive 的时刻，区分 payload 尾部和纯协议 retirement。
候选正在构建，尚未验证或采用。

U 的 EP16 两个长档四组梯度 hash 完全一致，但 BWD wall 为
4451.110/10616.500 µs，显著退步；资源仍为 REG168/SHARED2048/LOCAL0，
GIN STACK208，不能只凭寄存器报告判断搬运效率。**不采用 U**，三个本地
SIMT copy 改动撤回，保留用于定位的 terminal Put/Receive 时间戳。

V（`pf_backward_receive_priority`）：仅静态网络 BWD 在本轮已接收到 GIN
payload 时暂缓本地 Return，先持续释放 RX credit；没有收到包的下一轮再
处理原固定 owner 的 Return。传输仍有限、静态 ownership 不变，计算和
单机路径不变。用来验证 M128 本地 Return 是否让共享 copy warp 停收太久。

W（`pf_backward_return_ports4`）：在 V 上只将现有 GIN Return ports 从 2
调到 4；sender 配额仍 16。增加独立 Return receive owner/credit 流，不改变
每 slot 容量、K 归约或消息含义。使用 build 内临时生成配置，尚未修改生产
profile；V/W 都在构建，须分别验证和比较，不预记收益。

V/W 的 EP16 两个长档均通过 native 重复验证；V 四组 hash 已核对与 c6be309
完全一致。V 的 wall 为 2859.240/6663.720 µs，相比 M 仅小幅变化，8K 仍未
达标；20K 本地 Return 区间扩至 1057.280 µs。W 为 3116.870/7247.540 µs，
两个长档都退步。**V/W 均不采用**；恢复原先每轮推进接收和 Return 的次序，
生产 ports 仍为 2。扩大独立通道而不增加 sender，并不保证吞吐更高。

W 的两档四组 hash 随后也核对为与 c6be309 完全一致。上述失败候选没有残留
在生产数据路径；本轮只保留诊断时间点，并提交已有调参检查的宏清单修正。
现有调参/资源/launcher 检查共 34 项通过，没有新增细粒度测试。

X1/X2 使用原 M 调度，分别测试 `(Return ports, sender CTAs)=(4,32)` 与
`(2,32)`，对照 M 的 `(2,16)`。两个值均为既有调参项，使用 build 内临时
profile，不新增源码分支。与 W 的比较包含撤回 V，不能单独归因为 sender
数量变化；与 M 的比较才是这轮配置消融。尚未构建验证或采用。

X1/X2 均通过 EP16 两个长档 native 重复验证，BWD wall 分别为
3302.000/7997.390 和 3032.180/7416.470 µs，均退步。**不采用 X1/X2**，
生产仍使用 2 ports/16 senders。更多 sender 减少 receiver 数，并改变同
CTA 中通信/计算的资源竞争，不能把增加并行度直接等同于更高吞吐。

Y1/Y2 重新使用原 2 ports/16 senders，分别消融 math/control 寄存器配额
`216/72`、`208/88`，对照原 `216/64`。384-thread CTA 初始寄存器预算为
`168*384=64512`；原配置用 `216*256+64*128=63488`，Y1 只使用剩余的 1024
个寄存器，不降低 math 配额；Y2 则显式测试向通信/producer 再倾斜的取舍。
两者均使用既有参数和 build 内配置，不改 GEMM 的 MNK、归约顺序或协议。
尚未构建、验证或采用。

Y1/Y2 的 EP16 两个长档 native 重复验证通过。BWD wall 分别为
2846.080/6636.550、2840.620/6615.600 µs；原 M 为 2866.660/6679.420。
静态 GIN 栈由 208B 降为 152B/72B，静态单机仍 0B；不能将 LOCAL0 当成
没有任何栈访问的证明。Y2 的 8K device kernel 为 2713.950 µs，相邻更早 V3
为 2717.220，但 wall 为 2840.620 vs 2838.730，仍应称接近，不能宣告全面
领先。下一步核对 Y2 的单机性能，生产寄存器配置尚未修改。

Z（`pf_backward_early_rx_flush`，基于 Y2）：静态网络 BWD 的 receiver
在自己负责的 Dispatch/Return RX channels 全部收到 terminal 后，即执行
原有 ACK flush，不再统一拖到本 rank 的所有 sender 退出、机内通知排空
之后。该 CTA 不再有后续 ACK，仍可继续自己的本地 Return；其它 CTA 的
payload、计算和 Return 可并行推进。每个 receiver 仍只 flush 一次，末尾
保留尚未执行时的原 drain；动态路径、每槽 credit/TX local counter 均不变。
不删除任何完成检查，也不增加 packet buffer。候选待构建验证。

Y1/Y2 的 EP16 两档、Y2 的 EP8 两档，所有 rank 的四组梯度 hash 均已
核对为与 c6be309 完全一致，repeat_equal=1。Y2 的 EP8 BWD 为
1774.260/4592.870 µs，相邻 V3 为 2127.020/5050.010 µs，耗时降低
16.59%/9.05%；没有单机回退。

Z 的 EP16 两档 native 重复验证通过，BWD 为 2838.350/6662.720 µs。
8K 相比 Y2 持平，20K 退步约 0.7%，没有消除关键路径。**不采用 Z**，
提前 flush 分支和状态均撤回，仍在原 drain 点执行完整 ACK flush。

AA1/AA2 以 Y2 为基础，将已有 Backward sender CTA 配额由 16 分别改为
8/12，其余参数和代码保持不变。目的是检查较少 sender 能否改善它们与
计算、receiver 的竞争；不是继续增加通道。仅 build 内配置消融，尚未采用。

AB 将 Y2 的 math/control 配额进一步改为 200/104，CTA 总预算仍 64512，
sender 仍 16。用于寻找 Y2 收益的边界，不能由栈大小直接推断速度。

Y2 的 EP16 T8192/20480 完整 Torch 回归均通过：每档 16 ranks 的 output、
dX/dGate/dW1/dW2 parity，非法/重复 route 拒绝，分配失败 ordinal 3/5
一致失败后重试，三份 live tape 热点/扩缩容，以及 EP8 subgroup inference
graph/offload。每档 48 个 peer-reuse marker；已有 CPU 检查 34 项通过，
未新增细粒度测试。Z 的四组 hash 也已确认与 c6be309 完全一致。

AC（`pf_backward_launch_cache`，基于 Y2）复用已有 backend 的
`prepare_unified_launch`：同一 kernel、资源配额、启动几何只检查一次寄存器
预算、设置 shared-memory/stack 并查询 cluster occupancy。统一 Backward
不再为不会启动的 split kernels 设置属性；首次检查和驻留门槛仍保留。
不增加新缓存类型，不改设备工作/通信或数学顺序。该消融属于 host 提交开销，
CUDA event 区间也可能包含 host 未提交后续 kernel 的空隙，不能称纯 GEMM
加速。正在构建和验收，尚不预记收益。

Y2 经上述长文回归后采用到 checked-in H2048/I512 profile；不是修改所有
模型的通用默认值。其 EP8 实际训练循环（50 warmup + 200 measured）为
5947.5945/10442.4655 µs，较此前相邻 V3 的 6458.6735/10986.6935 更快。
仍需对最终组合进行完整配对，不能拼接不同候选的最优单项宣告目标完成。
AA1/AA2/AB 均已通过 EP16 两档 native 重复验证，全部 hash 与 c6be309
一致；两机 16 卡于 08:08 UTC 复查均为 1620 MHz，开始独占性能比较。

AA1/AA2 的 EP16 BWD 分别为 3941.040/9467.750 和
3458.970/8144.130 µs，均显著回退。**不采用 8/12 sender 配额**，保留 16；
较少 sender 会让每个 owner 承担更多 channel，不能只看增加 receiver 数量。

AB 的 EP16 BWD 为 2835.600/6615.320 µs，未证明比 Y2 更快；静态 GIN
栈仍 72B，本地仍 0B。**不采用 200/104**，保留已完整验证的 208/88。
AC 为 2832.940/6616.670 µs，相邻 V3 为 2839.110/7011.880；8K 差别仅
0.22%，仍不能宣称已经形成稳定优势。AC 的两档四组 hash 已核对一致。

AD（`pf_backward_private_gin`，基于 AC）只调整统一 BWD 的 GIN 提交资源
模式：实际 context 数大于全部 worker CTA 数时，`1+cta` 无别名，每 CTA
只有一个线程调用 Put 或 ACK/flush，因此使用 THREAD 独占模式。context 0
仍给 metadata；`GinExecution` 的 last-use event 串行化共享该 pool 的调用。
实际 context 不足或非 unified 路径仍保留 GPU sharing，不以请求数量冒充
实际分配。逐 slot ready、remote credit、本地 NIC counter、ACK flush 和
数据内存序完全不变；不是删除本地 credit 检查。Forward 未改。
代价是 Backward 提交路径多一个运行时模式分支，须测编译资源和端到端效果。
沿用已有 BACKWARD_PROGRESS 输出实际 contexts/workers，不新增设备状态。
尚未构建验证或采用。

AC 的 EP8 两档四组 hash 亦与 c6be309 完全一致。EP8 BWD 为
1769.390/4663.810 µs，20K 的约 1.5% 波动没有设备算法变化支撑，不能归为
launch cache 的计算回退或收益。相邻实际循环（50/200）PF 为
5889.5745/10398.4815 µs，V3 为 6354.0935/11048.4865 µs。
AC 的 EP16 T8192/20480 完整 Torch 数值、route rejection、ordinal 3/5
失败及重试、三份 live tape/peer reuse、EP8 subgroup graph/offload 均通过。
**采用 AC 的启动准备复用**，明确不将整个相对 V3 收益归于这项 host 改动；
双机实际循环仍待最终组合对照。没有新增测试框架或细粒度源码断言。
