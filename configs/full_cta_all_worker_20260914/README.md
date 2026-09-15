# 全 CTA worker 的已选配置

`qwen.hpp` 和 `flash.hpp` 是本轮实际编译、验证的组合头，字节内容与 JIT 产物一致。每个模型在 EP8/EP16、8192/65536 tokens/rank 使用同一配置；没有按长度或 EP 增加 CUDA 特判。两阶段应使用同一组合头。

`profiles.json` 保存全部实际生效的 FWD/BWD 参数、组合头校验值和对应源码校验值。头内原生成配置的注释摘要不包含前置覆盖值，应以本文件的完整 header SHA 和解析参数为准。公开 autotune 网格仍支持一般性参数搜索；本目录只保存本轮选择。

Qwen 使用每 channel 最多 3 个预约并启用闲时 GIN 协作；Flash 使用单预约并关闭协作。这些是同一 CUDA 实现的编译参数。FWD 为 math216/control64、双通信 buffer；BWD 分别为 Qwen216/64、Flash224/56，使用单 buffer。

实现与资源约束见 [架构说明](../../docs/full_cta_all_worker.md)。性能结果与限制见 [测试报告](../../benchmarks/full_cta_all_worker_20260914.md)。
