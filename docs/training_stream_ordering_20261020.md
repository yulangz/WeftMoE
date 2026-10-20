# 训练热路径：本地分配与 GPU 顺序复用

## 改动

- Python runtime 的 FWD/BWD tensor 分配直接调用 allocator，不再逐次
  AllReduce 确认所有 rank 分配成功。
- split notify 后的 saved X/AB 分配同样直接执行；本地分配异常原样向上传播。
- 单机 BWD 的 peer-buffer 协议只确认“是否有地址改变”，不再同时协调
  每次分配失败。初始化、IPC remap 的既有协议不变。
- Megatron 使用的异步调用入口不再执行阻塞式 `drain_training()`，只回收
  已完成调用。显式 drain 和 offload/销毁边界仍等待完成。

这次不修改外部监督、作业终止或重启策略；正常训练路径不承诺原进程内的
分配失败重试。删除了测试中对应的 collective OOM/retry 要求。

## 共享 workspace 如何安全复用

workspace 仍只有一份，不增加通信 ring，也不改变 kernel 内的 head/tail、
代际握手、数学计算或任务划分。

1. FWD 在执行流上提交 notify/main；BWD 在 transport stream 上执行，
   并通过现有 CUDA event join 回执行流。
2. 在同一执行流上，将本次状态拷到独立的 pinned completion record，
   记录完成事件，再发布 mapped-memory 完成标记。
3. workspace 的使用权在提交阶段交回。后续 reset/notify/main 通过
   stream 顺序和完成事件依赖排在旧调用之后，而不是等待 CPU 确认完成。
4. runtime 队列保留未完成调用的输入、临时 tensor 和 metadata 引用。
   只有 CPU 以 acquire 语义观察到完成标记，才回收对应调用。

completion record 和计时事件会缓存复用。不能继续使用共享 slot 中的
单份状态作为旧调用的完成记录，否则下一层可能覆盖尚未读取的状态。
回收时不轮询 CUDA event，避免在 persistent GIN kernel 运行期间引入
之前观察到的驱动进度锁问题。

## 没有删除的同步

- FWD notify 得到实际接收行数后，CPU 为 saved X/AB 分配精确空间的边界。
- BWD transport stream 到调用流的依赖。
- 单机 BWD 判断 peer 地址是否变化的控制通信，以及必要的 IPC remap。
- 初始化、offload、销毁边界的资源同步。

因此这不是“全流程没有 CPU 同步”，而是删除每层分配成功确认，以及
为了回收执行 workspace 而主动等待上一层完成的 CPU 阻塞。

## 验证

- Qwen H2048/I512 CUDA 库、Python 扩展编译及扩展加载通过。
  PF main 没有新增 spill；旧 `training_count_notify_kernel` 的 28 B
  spill stores/loads 与改动前一致。
- CPU 控制层：17/17 通过。
- runtime 配置：4/4 通过；修改的 Python 文件通过语法检查。
- 分布式测试复用 `tests/torchrun_tensor_runtime.py`，设置
  `TEST_STREAM_ORDERED_TRAINING=1`，比较同步/异步的两层 FWD、反序 BWD、
  共享权重 packing、输出与梯度以及最终释放。
- 候选产物独立放在 `build/rl_ep16_stream_ordered_20261020/`，
  不覆盖现有 RL 长跑使用的库。

### GPU 正确性与释放

2026-10-20，两台机器的 16 张 L20X 均固定 1620 MHz。旧 RL 组在完成
23 次 update 后主动停止，未覆盖旧库或日志。以下测试使用 `73bfc5f`
对应的独立候选产物：

| 检查 | 单机 EP8 | 双机 EP16 |
| --- | --- | --- |
| T64 的 FWD、dX、dGate、dW1、dW2 对照 PyTorch | 8/8 rank 通过 | 16/16 rank 通过 |
| 两层共享 packing、异步 FWD、反序 BWD | 输出和所有梯度逐位一致 | 输出和所有梯度逐位一致 |
| 三组移动热点及变化 token 数、权重原地更新 | 通过 | 通过 |
| 训练 workspace 释放、切换 EP4 推理 CUDA Graph、图销毁释放 | 通过 | 通过 |
| T2048 随机路由，5 次 runtime 重建/释放 | 通过 | 通过 |

T2048 使用两套不同层权重、两份同时存活的 tape，同步/异步各执行一次，
中间插入输出 clone 及无关 GPU 计算。5 个周期内数值均逐位一致。
EP16 的 16 个 rank，5 次关闭后的 native 显存占用与 PyTorch 存活 tensor
字节数均完全不变。

释放测试最初只比较 `cudaMemGetInfo`，曾因 PyTorch allocator 的分段碎片
误报：某 rank 空闲显存少了 98 MiB，但 `torch.cuda.memory_reserved()`
恰好多了 98 MiB，`memory_allocated()` 不变。保留原始失败日志；
最终分别检查 `free + torch_reserved`（隔离 native 占用）和
`torch_allocated`，没有增大允许泄漏的阈值。采样后增加的 barrier、
GC 和同步仅存在于诊断脚本，不进入生产热路径。

日志在候选产物目录：

- `tensor-ep8.node0.log`、`tensor-ep16.node{0,1}.log`；
- `async-ep8-release.node0.log`、`async-ep16-accounted.node{0,1}.log`。

### 整模型性能回放

使用之前保存的真实 RL 训练 batch，Qwen3.5-35B-A3B-Base、
TP4/DP4/EP16，每次 update 34 个 microbatch。新版本、修改前版本、
Megatron + DeepEP 串行各执行 6 次，排除 step 0、1，对后 4 次逐步取
16 个 rank 的最大耗时后求均值。包含 CPU-staged onload/offload，
单独列出 `train_batch` 和包含切换的总耗时；不包含 rollout，
不能解释为完整 RL 端到端收益。

三组均完成 6 次 update，全部 `update_successful=1`，两节点进程
正常退出：

| 版本 | 训练均值 / s | 训练范围 / s | 含 onload/offload 均值 / s |
| --- | ---: | ---: | ---: |
| Megatron + DeepEP | 48.127 | 47.471–48.606 | 50.874 |
| 修改前，`rl_ep16_residency_fix_20261020` | 47.271 | 46.275–48.245 | 50.525 |
| 新版，`73bfc5f` | 45.526 | 45.361–45.848 | 48.717 |

新版训练耗时下降 **3.69%**；包含切换的耗时下降 **3.58%**。这次没有
修改 GEMM 或 GPU 通信 kernel，收益来自宿主端同步和执行 workspace
的复用方式，不应解释成算子计算效率提高或完整 RL 提速同样比例。
相对于本轮重测的 Megatron + DeepEP，新版训练耗时低 **5.40%**，
包含切换低 **4.24%**。

原始结果在 `build/rl_ep16_session_reuse_20261020/stream_ordered_{new6,old6,baseline6}.node{0,1}.log`，
命令、环境和启动 PID 保存在对应 `.manifest.json`；汇总在候选目录
`replay_summary.json`。其中旧库 SHA256 为
`d5b5c0493e3c30ab32d466817440a908954b4d53a75de59a5aa6eed9f580aa11`，
新库为 `b9b4916b5d5d8eb4d8276c30c4c43446edd0d967e66f550ebe3c58240b920182`。

### 固定权重整模型梯度复核

同一份真实 batch，学习率始终为 0；依次运行同步参考、异步、再次同步，
恢复 CUDA RNG 状态，比较优化器更新前的完整梯度 buffer。16 个 rank
均完成三轮，均 `update_successful=1`。这组保留参考梯度的诊断不计入
性能表。

相对 L2 按各 rank 对应 buffer 拼接后的
`sqrt(sum(||g-g_ref||²) / sum(||g_ref||²))` 计算：

| 参数组 | 异步 vs 同步参考 | 再次同步 vs 同步参考 |
| --- | ---: | ---: |
| 普通参数 `buffers` | 0.5556% | 0.5438% |
| 专家参数 `expert_parallel_buffers` | 0.6134% | 0.6058% |

异步误差略高，但与重复同步自身的数值波动同量级，本轮未观察到明显的
额外异步误差。这里不声称整模型逐位一致，也不以三轮/六轮短回放证明
长期 reward 或收敛一致性。

原始日志为同目录的 `stream_ordered_gradient_parity3.node{0,1}.log`，
复用 `check_training_order.py` 和 `run_replay.py --order-parity --steps 3`。
本次 GPU 验证没有要求继续修改 `73bfc5f` 的生产实现；额外调整仅限
显存诊断脚本的采样和统计口径。
