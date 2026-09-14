本轮统一采用 1620 MHz 测量。纯 GEMM 使用 BF16 输入/输出、FP32 累加，统计单 GPU 的有效 FLOPS（一次 FMA 计 2 FLOPs），开放全部 132 个 CTA。EP8/EP16 在纯 GEMM 中决定每卡专家数和每专家 M；通信性能由完整算子单独测量。

两边均通过 CUDA Graph 重放，交替顺序取样。每次取样前做一次相同实现的预热，并间隔 50 ms，避免持续满载触发 700 W 功耗上限而降频；这些间隔不计入 GEMM 时间。因此这些是同频 GEMM 数据，不是持续训练吞吐。每个配置有两轮、每轮双方各 30 个样本，以下使用合并样本的中位数。cuBLAS 使用 `cublasGemmStridedBatchedEx`，关闭低精度归约。

全部 5760 个计时窗口的前后频率均为 1620 MHz；额外约 3 ms 间隔的监控在 3354 个窗口内采到 7691 个读数，均为 1620 MHz，显存为 3201 MHz。未按耗时筛选样本。

FC2 将一组已就绪 N tile 的输入预取和输出写回与后续 GEMM 重叠，使用独立输出暂存区，并给 FC2 权重设置 L2 保留提示。两种模型统一采用 3 个输入 stage、每 CTA 最多处理 4 个 N tile；分组不跨 M block。新增 `fc2_claim_tiles` 可调为 1/2/4/8，尾部分组与 stage 相位均按实际剩余工作处理。没有按模型名称分支。

| 模型 | EP / 每 rank 长度 | FC2 修改前 | FC2 修改后 | cuBLAS | 自研 / cuBLAS | FC2 加速 |
|---|---|---:|---:|---:|---:|---:|
| qwen | 8 / 8K | 440.5 | 504.9 | 543.4 | 92.9% | 14.6% |
| qwen | 8 / 64K | 458.7 | 564.7 | 603.6 | 93.6% | 23.1% |
| qwen | 16 / 8K | 437.4 | 511.2 | 546.0 | 93.6% | 16.9% |
| qwen | 16 / 64K | 458.3 | 564.6 | 603.3 | 93.6% | 23.2% |
| flash | 8 / 8K | 631.0 | 655.5 | 686.1 | 95.5% | 3.9% |
| flash | 8 / 64K | 657.4 | 684.5 | 712.1 | 96.1% | 4.1% |
| flash | 16 / 8K | 635.7 | 655.3 | 687.0 | 95.4% | 3.1% |
| flash | 16 / 64K | 658.1 | 683.6 | 711.7 | 96.0% | 3.9% |

以上及下表均为 TFLOPS。BWD 的固定 K 展开、额外缓存提示和等待策略探索没有形成足够稳定的收益，保留原有 BWD GEMM 与 dW 的确定性累加实现。

| 模型 | EP / 长度 | GEMM | 自研 | cuBLAS | 自研 / cuBLAS |
|---|---|---|---:|---:|---:|
| qwen | 8 / 8K | BWD FC2 → dU | 568.1 | 618.3 | 91.9% |
| qwen | 8 / 8K | BWD FC1 → dX | 539.5 | 636.2 | 84.8% |
| qwen | 8 / 8K | dW1 | 609.6 | 656.3 | 92.9% |
| qwen | 8 / 8K | dW2 | 561.0 | 613.9 | 91.4% |
| qwen | 8 / 64K | BWD FC2 → dU | 636.6 | 709.5 | 89.7% |
| qwen | 8 / 64K | BWD FC1 → dX | 570.8 | 683.9 | 83.5% |
| qwen | 8 / 64K | dW1 | 703.6 | 744.3 | 94.5% |
| qwen | 8 / 64K | dW2 | 681.8 | 730.8 | 93.3% |
| qwen | 16 / 8K | BWD FC2 → dU | 569.9 | 619.1 | 92.1% |
| qwen | 16 / 8K | BWD FC1 → dX | 539.2 | 635.3 | 84.9% |
| qwen | 16 / 8K | dW1 | 642.4 | 679.6 | 94.5% |
| qwen | 16 / 8K | dW2 | 594.7 | 637.3 | 93.3% |
| qwen | 16 / 64K | BWD FC2 → dU | 637.2 | 710.1 | 89.7% |
| qwen | 16 / 64K | BWD FC1 → dX | 570.4 | 683.5 | 83.5% |
| qwen | 16 / 64K | dW1 | 706.3 | 744.0 | 94.9% |
| qwen | 16 / 64K | dW2 | 691.8 | 733.5 | 94.3% |
| flash | 8 / 8K | BWD FC2 → dU | 672.6 | 716.6 | 93.9% |
| flash | 8 / 8K | BWD FC1 → dX | 697.5 | 718.1 | 97.1% |
| flash | 8 / 8K | dW1 | 631.7 | 687.8 | 91.8% |
| flash | 8 / 8K | dW2 | 623.7 | 680.0 | 91.7% |
| flash | 8 / 64K | BWD FC2 → dU | 710.7 | 755.3 | 94.1% |
| flash | 8 / 64K | BWD FC1 → dX | 713.4 | 737.4 | 96.7% |
| flash | 8 / 64K | dW1 | 700.8 | 752.0 | 93.2% |
| flash | 8 / 64K | dW2 | 703.0 | 767.7 | 91.6% |
| flash | 16 / 8K | BWD FC2 → dU | 675.0 | 715.7 | 94.3% |
| flash | 16 / 8K | BWD FC1 → dX | 696.0 | 719.0 | 96.8% |
| flash | 16 / 8K | dW1 | 670.4 | 711.7 | 94.2% |
| flash | 16 / 8K | dW2 | 660.8 | 706.3 | 93.6% |
| flash | 16 / 64K | BWD FC2 → dU | 711.4 | 754.7 | 94.3% |
| flash | 16 / 64K | BWD FC1 → dX | 714.5 | 738.1 | 96.8% |
| flash | 16 / 64K | dW1 | 705.3 | 752.2 | 93.8% |
| flash | 16 / 64K | dW2 | 707.5 | 761.5 | 92.9% |

完整 FWD 的 wall 时间包含 input、output 等固定 tensor 的申请；复用的 buffer、RDMA context、初次准备工作不计入。双方采用相同的 1620 MHz 频率上限和 700 W 功耗限制，连续运行中的实际频率保存在监控日志中。纯 GEMM 的间隔采样未用于此处。

| 模型 | EP / 长度 | 修改前 wall ms | 修改后 wall ms | Flex wall ms | wall 加速 |
|---|---|---:|---:|---:|---:|
| qwen | 8 / 8K | 3.678 | 3.657 | 4.720 | 0.6% |
| qwen | 8 / 64K | 23.956 | 23.822 | 25.174 | 0.6% |
| qwen | 16 / 8K | 6.840 | 6.858 | 5.294 | -0.3% |
| qwen | 16 / 64K | 37.239 | 37.152 | 31.319 | 0.2% |
| flash | 8 / 8K | 8.162 | 7.861 | 8.919 | 3.8% |
| flash | 8 / 64K | 55.761 | 54.909 | 60.828 | 1.6% |
| flash | 16 / 8K | 10.804 | 10.524 | 10.162 | 2.7% |
| flash | 16 / 64K | 70.122 | 67.962 | 70.363 | 3.2% |

Qwen 的完整 FWD 基本没有变化，Flash 的 wall 时间缩短约 1.5%–3.7%。FC2 纯 GEMM 的收益不能等同于融合算子的收益。Flex 的 DeepEP SM 数采用此前测试的配置（EP8 为 80，EP16 为 48），本轮未在 1620 MHz 下重新穷举调参。

稳定性覆盖两种模型的 EP8/EP16、8K/64K：完整输出与已有参考哈希一致，多次运行逐字节一致，包括 BWD 的 dW；FWD 额外比较 129/513 token 尾部、不同 batch、不同 EP 及相同全局 batch 重分片。也检查 CUDA Graph / buffer 复用，以及 H=2560、I=1280 的 partial stage cycle、N tile 尾部分组和两种 WGMMA wait 设置。调试版本检查任务计数与完成协议。

正式计时的源码快照为 `results/gemm_optimization_20260914/after_source`。之后只更新了诊断程序的非整组静态任务分配和非有限值检查，生产 kernel 未改变；更新后的诊断又通过全部 40 个配置的完整输出、cuBLAS 误差和逐字节重复检查。两种目标模型的正式数字仍对应原快照；下列命令构建当前诊断，要复现原二进制应附加 `--source results/gemm_optimization_20260914/after_source`。

复现时先设置 GPU 频率，再构建两个阶段的纯 GEMM 诊断：

```sh
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
nvidia-smi --lock-gpu-clocks=1620,1620
python benchmarks/run_gemm_diagnostics.py --phase forward --profile configs/performance_20260914/qwen_ep8.hpp --output-dir results/gemm/qwen_forward
python benchmarks/run_gemm_diagnostics.py --phase backward --profile configs/performance_20260914/qwen_ep8.hpp --output-dir results/gemm/qwen_backward
python benchmarks/run_gemm_matrix.py --binaries results/gemm --output results/gemm/qwen_results --models qwen --clock-mhz 1620
nvidia-smi --reset-gpu-clocks
```

本次 cuBLAS 版本为 12.9.1.4，采用 strided-batched 接口，未穷举 cuBLASLt 算法。需要 CUDA 12.9、CUTLASS 与 NVML 动态库；运行时使用与构建时一致的 CUDA/cuBLAS 库。完整数字、原始日志路径、源代码与配置指纹见同目录 JSON。旧的短热身/混合频率实验仅用于探索，本报告以本轮同频结果为准。

补充通用尺寸的 FC2 检查（更新后的诊断，单轮 30 样本，前后频率均为 1620 MHz）：

| H / I / 每专家 M / 专家数 | 自研 TFLOPS | cuBLAS TFLOPS |
|---|---:|---:|
| 2560 / 1280 / 3072 / 16 | 595.0 | 638.8 |
| 7168 / 3072 / 1536 / 32 | 655.3 | 720.3 |
