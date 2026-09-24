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
