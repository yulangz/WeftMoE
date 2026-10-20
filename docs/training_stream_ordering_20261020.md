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
  共享权重 packing、输出与梯度以及最终释放。GPU 验证尚未执行。
- 候选产物独立放在 `build/rl_ep16_stream_ordered_20261020/`，
  不覆盖现有 RL 长跑使用的库。

尚无此次改动的 GPU 正确性结论或性能收益数字。性能比较应排除冷启动，
使用稳定后的连续多步数据，而非首步或单个热态 step。
