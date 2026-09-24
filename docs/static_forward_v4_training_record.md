# V4 ring 训练对接记录

对应设计文档 M6。默认训练仍执行 V3；M6d 接通 opt-in 单机 ring Forward，
Backward 仍复用既有调度/GEMM；M6e进一步接通多机checkpoint和有界Backward
GIN adapter，M6f已完成精确P_actual分配。完整训练sanitizer/性能门禁尚未完成，不能将
inference或新Forward的门禁视为整个训练工程完成。

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
