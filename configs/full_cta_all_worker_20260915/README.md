# 每 channel 8 槽的全 CTA worker 配置

`qwen.hpp`、`flash.hpp` 精确复制自本轮已编译、验证的
`comm_final_qwen_d8_v1`、`comm_final_flash_d8_v1` 组合头。相对
[9 月 14 日配置](../full_cta_all_worker_20260914/README.md)，唯一调参变化是
`WEFTMOE_GIN_RING_DEPTH=8`；生产源码的 ACK 同时统一改用完成位图，删除了
旧扫描分支和实验开关。原配置及其 4 槽历史结果保持原样。

每个模型在 EP8/EP16 使用同一组合头，FWD/BWD 必须采用相同物理深度，所有
rank 也必须使用一致的编译配置。该参数改变持久 TX/RX 显存窗口、counter 和
credit 槽数，不改变每个 context 的 3 个 packing 条目，也不增加 CTA shared
memory。Qwen/Flash 的原 GEMM、寄存器、预取和 channel packing
配置均保留。

`profiles.json` 保存实际 phase 参数、完整头 SHA，以及冻结源码 manifest 中
99 个 `src/`、`include/` 文件的 SHA。头内较早的生成注释摘要未包含前置覆盖，
应以完整 header SHA 和 phase 参数为准。来源是
[最终源码 manifest](../../results/full_cta_all_worker/ring_depth_20260915/final_source_manifest.json)，
不会把实验源码或历史计时重新标记为本轮结果。

最终版本已经完成的验证范围：

- 两模型 FWD：最终二进制 EP16、129/513 tokens/rank，与 main 及此前验证过的
  EP8 输出前缀逐字节一致。EP8 统一入口的机器码未改变；本轮未重跑最终二进制的
  EP8 正式性能矩阵。
- 两模型 BWD：最终二进制 EP16、129 tokens/rank，每模型完整 160 份张量
  （包括 dW）与 main 逐字节相同。此原生 fixture 保留模型 H/I/K，使用每 rank
  两个专家；正确性运行不作为性能计时。
- 两模型 EP16、E256、8192/65536 tokens/rank，FWD/BWD 共八个正式测量点，
  全部完整输出重复检查通过；FWD 另通过 main 前缀与全部 rank 注册容量检查。
- 完成位图候选的两模型 EP16 Graph，每模型六次 Notify/FWD/BWD 重放及
  buffer growth/reuse 通过。最终二进制十二个统一入口与该候选的机器码完全一致；
  没有把候选的 Graph 记录改标为最终二进制的新运行。Graph fixture 为 129 tokens、
  K2、每 rank 两个专家，不包含 cancellation 验证。

证据见 [最终 native 验证](../../results/full_cta_all_worker/ring_depth_20260915/final_native_validation.json)、
[最终运行检查](../../results/full_cta_all_worker/ring_depth_20260915/final_runtime_audit.json)、
[机器码等价检查](../../results/full_cta_all_worker/ring_depth_20260915/final_resource_audit.md)
和 [Graph 候选验证](../../results/full_cta_all_worker/ring_depth_20260915/validation/dirty_ack_graph/graph_validation.json)。
性能及全部离群样本见[测试报告](../../benchmarks/ring_depth_20260915.md)；本目录不宣称所有规模均加速。

publisher warp 的 GIN service 仍为隔离实验，未进入生产路径。位图 ACK 不增加
注册窗口容量，普通 ACK 不再遍历全部槽，最终 drained 仍逐槽核对代次。
TMA 完成、payload-before-ready 和 RX
credit 约束见[架构说明](../../docs/full_cta_all_worker.md)。
