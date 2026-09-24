# V4 ring 训练对接记录

对应设计文档 M6。当前训练仍执行 V3，不能将 inference 的正确性和性能门禁
视为训练对接完成。所有未验证改动保持候选状态。

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
