# PF Backward 对齐现版 Forward（2026-10-12）

基线是 `main@1cfb423`；实现分支 `feature/pf-backward-20261012`。
旧静态 BWD 保留为迁移期间的正确性/性能对照，不将新 FWD 的收益外推到 BWD。

## 设计与里程碑

- [x] M1 单机：复用 PF destination-pull、输入 ring 和完整 source
  contribution；producer/math 独立重放静态 dQ/SwiGLU/dX 任务流。
  dX epilogue 从 accumulator 直接 128-bit 写回。所有本地 dX 完成后
  math 开始静态 dW，通信 warp 进行 source reduce。
- [ ] M2 多机：复用当前借用 warp 的按节点去重、逐 token Get、机内 pull、
  直接 TX contribution/Put return。路由计划来自 FWD tape，不再重新
  counts notify。RX fanout、TX NIC completion 和远端 credit 分开退休。
- [ ] M3 优化与回归：分别测 dQ/dX/dW 参数、缓存写出和激活融合；覆盖随机
  tokens/routes、热点/空专家、多轮 slot 回绕。FWD 和 DT 不改变数学流程。

数学记号：A/B 为 FC1 gate/up，r 为 router 权重，Q=r*SiLU(A)*B。
dQ=dY*W2^T；SwiGLU backward 同时生成 dAB、dr 和供 dW2 使用的 Q。
dX=dAB*W1^T；dW2=Q^T*dY；dW1=X^T*dAB。权重方向以数学表示为准，
公共 API 保持原有 `[expert, out, in]`。dr 返回选中 route 的梯度，上层
框架继续处理 router softmax/归一化及 router 线性层。

## 单机第一阶段

输入 pull 和 reduce 复用 `local_kernel.cuh`，通过编译期参数选择读退休
条件与线程域，Forward 默认实例不变。独立 dQ/dX N-tile 任务不经过
旧 BWD 的共享 task mailbox；两个相邻 CTA 沿 N 共享 multicast A。
SwiGLU 由一对 CTA 分别处理半个 M-block，保留旧实现的 BF16 舍入和
行内 dr 归约，不引入浮点 atomic。

保存的 X/A/B 和路由属于 FWD tape；物理 ring 行经已有 saved-row 映射
写入专家连续的 dW cache。首版先复用现有 Q 工作区：dQ 写入后由
SwiGLU 原地覆盖成 Q，dAB 同时写完整 cache 和供 dX 使用的有界 ring。
不增加单独的全尺寸 dQ 或 dX 临时张量。

当 `2I <= H` 时，dQ 消费完的 dY ring 原地作为 dAB ring，物理行 stride
仍为 H。下次 pull 必须等本代所有 dX 读者完成，不能沿用 Forward 的
FC1-done credit。`2I > H` 才申请独立 dAB ring。已有 Forward contribution
直接用于 dX；不在 BWD peer workspace 再申请一份无用 contribution。

持久 peer workspace 目前仍保存 dY cache 和 IPC 可见 dr 标量；容量增长
与地址交换沿用现有策略。单机初始化直接复用 FWD 的专家目录，只生成
dW 所需 padded-row 映射、清空本调用计数器和空专家梯度，不再初始化
旧 activation scheduler 或重新拷贝 route headers。

## 测量口径与初始数据

固定 1620 MHz，Flash H4096/I2048/E256/top-6，EP8 为单机；EP16 为双机。
每 rank tokens 正态波动 10%，routes/gates 每步变化，共享专家 bias=0.5。
固定 seed=123，先完整 prime trace，再 4 warmups / 8 samples。
表中为同一步最慢 rank 的完整 FWD+BWD wall latency 的 p50，包含每调用
权重重排、缓存申请及回收，排除输入构造与首次通信池创建。

| 对照 | 8K (ms) | 20K (ms) |
|---|---:|---:|
| 旧 PF BWD，单机 EP8 | 26.483 | 68.624 |
| Megatron 缓存模式，单机 EP8 | 24.792 | 59.548 |
| 新 PF BWD，单机 EP8（最终候选） | 25.209 | 62.164 |
| 新 PF BWD，单机 EP8（复测） | 25.193 | 62.178 |
| 旧 PF BWD，双机 EP16 | 32.668 | 78.531 |
| Megatron 缓存模式，双机 EP16 | 33.288 | 87.202 |

双机旧 PF 基线检测到测量区间内 1455/1470 MHz 短暂降频，因此这一行
仅作为原始观测，正式同频比较需要重跑，不筛掉慢样本重算。
最终单机候选也检测到一次 1590 MHz（20K sample 4）；复测期间有一次
1485 MHz。设置为 1620 MHz 不代表实际运行始终维持该频率，以上均
保留原始样本，不能将百分之一量级的差别解释为稳定收益。

原始 manifest、workload hash、逐 rank 时间与频率采样保存在
`build/pf_backward_20261012/results/`。以上是本轮短对照，不替代历史长测。
脚本的 `backward_slowest_rank_p50_us` 是 host API 边界差值，不应称为
纯 BWD kernel 时间；单独设备阶段时间应另行测量。

初始单机原型通过 EP8、129/8K/20K 的 output/dX/dr/dW1/dW2 检查。
初版编译出现 setmaxnreg 忽略和 96/76 B 静态 spill store/load，正在
分离寄存器域和移除 dW-only 模板中的不可达 activation 分支。正式收益
以最后保留代码的无插桩测量为准，不将编译候选当作已完成版本。

初始任务流每个 expert 分别排 dQ/SwiGLU/dX。完整周期 8K/20K 为
29.112/63.671 ms：8K 比旧版慢，20K 较快。把 dY/dAB ring 复用并移除
冗余 contribution 后为 29.069/63.955 ms，没有稳定速度收益；该改动
主要减少容量。后者在未计时阶段观察到 1530 MHz，已保留频率记录。
反汇编表明已见的 spill 主要位于调度与阶段切换，不在 GEMM K 热循环；
增大 control register quota 仍不能恢复 setmaxnreg，不据此声称优化成功。

已保留的主要性能优化是将相邻专家合并为一个有界 compute wave：dQ 的输出列数
只有 Forward FC1 的一半，逐专家切阶段可能让短 expert 的 dQ 任务不够
填满 CTA。合并不改变 M padding、归约顺序或路由，只扩大阶段内可用
的静态 N-tile 任务集合，仍由 ring depth 限定窗口。
该项单独测量为 25.116/60.884 ms，频率采样均为 1620 MHz。
随后简化初始化并增加阶段时钟，四级 operand pipeline 测得
24.956/64.017 ms；三级最终候选为表中的 25.209/62.164 ms，保留三级。
最终版相对旧 PF 完整周期约快 4.8%/9.4%，但没有追平本轮 Megatron。
尚有 ptxas setmaxnreg 忽略与阶段切换 spill，未宣称已消除。

最终候选通过 EP8、1/129/8K/20K、每规模两次变化路由的 Torch 检查，
包括 output/dX/dr/dW1/dW2；全部相对 L2 误差最大约 0.00142。
四级前一候选另外通过 bias=8 的热点/空专家及 512-token window 回绕。
Python 回归为 117 tests / 470 subtests。没有新增碎片化单元测试。

## 多机实施约束

FWD 的 immutable route/packet plan 可复用，但物理 RX/TX 地址必须按
当前 channel base 重新绑定，不能复用 FWD 已退休的 packet generation。
dr 使用独立小型 FP32 TX/contribution 平面与 GIN 完成信号；dX 和 dr
分别 acquire 成功后才返还 packet credit，TX 仍保留本地 NIC completion。
通信共享内存必须与 dW arena 分离，否则为了复用 arena 而等待所有
Return 结束，会让 dW 与 Return 串行。上述多机实现尚未完成。
