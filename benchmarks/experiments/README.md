# FP8 与 DT 实验记录

当前实现使用固定 M64 的 BF16/FP8 DT。PF 保持原有 tile，FP8 math/control
寄存器配额统一为 224/56。实际 API 和数值约定见 [使用文档](../../docs/usage.md)。

| 目录 | 用途与状态 |
| --- | --- |
| [dt_m64](dt_m64/README.json) | 当前训练 PF / DT 逐位一致性、完整 DT 回归及清理前后成对性能验证 |
| [dt_fp8](dt_fp8/README.json) | FP8 DT 构建、验证、benchmark、profile 和固定 M64 的离线调参工具 |
| [fp8_prequant_input](fp8_prequant_input/README.json) | 预量化 X/dY 接入与跳过在线量化的测量记录 |
| [fp8_notify_overlap](fp8_notify_overlap/README.json) | 源端准备与 notify/planner 并行的候选及已采用实现记录 |
| [fp8_profile_opt](fp8_profile_opt/README.json) | PF epilogue、Wgrad 等优化候选；采纳情况以对应报告为准 |
| fp8_profile_review | 诊断用时间线、计数器及 SASS 分析 |
| fp8_rowcol_gemm | 未采用的 GEMM 内 row→col 重量化方案 |
| [pf_stage_release](pf_stage_release/README.json) | 未采用的 BWD 提前释放 operand stage 方案 |
| [dt_fast_math](dt_fast_math/README.json) | 历史 fast-math 对齐实验；当前结果以 dt_m64 为准 |
| [dt_m_shapes](dt_m_shapes/README.json)、[dt_fp8_pipeline](dt_fp8_pipeline/README.json) | 历史比较入口；旧 M16 构建器及候选补丁已移出活动源码 |

日期固定的实验脚本依赖对应 `build/` 下的不可变快照和日志，不应把旧补丁应用到当前
源码。历史报告保存在 [docs/benchmarks](../../docs/benchmarks)，其中的性能及数值结论
只适用于各报告记录的版本、设备和输入。构建产物、完整 profiler 文件和本地工具库不入库。
