# CUDA 实现结构

主机侧的 transport/notify 按完整职责组织。共享状态由所属 `.cu` 管理，相关的启动、
等待和释放逻辑放在一起；PF/DT 的设备计算仍位于现有 kernel 目录。

| 文件 | 职责 |
| --- | --- |
| `src/cuda/activation_transport.cu` | 共享状态、权重与输入 lease、资源池、P2P/GIN 后端初始化、生产环境能力探测 |
| `src/cuda/activation_transport/execution.cuh` | FWD/BWD 参数准备、kernel 启动、完成处理、计算 handle 的等待和释放 |
| `src/cuda/activation_transport/validation.cuh` | 独立传输验证路径、结果下载、GIN/P2P 正确性探测和链路 benchmark |
| `src/cuda/device_notify.cu` | route/gate 输入、执行上下文、workspace 布局与资源池、notify kernel 和 preflight/operation |
| `src/cuda/notify/dispatch.cuh` | dispatch 元数据 materialization、consumer 所有权与完成状态、transport bridge 和 dispatch 验证 |

这三个 `.cuh` 是私有实现片段，各自只由对应的 `.cu` 包含一次，位于
`weftmoe::cuda` namespace 内。它们依赖所属 `.cu` 前面定义的状态和 kernel，
不作为独立头文件或独立编译单元使用。共享状态保持原有可见性，公开 C++ API 不变。

`activation_transport.cu` 在完成资源与后端定义后包含 execution 和 validation；
`device_notify.cu` 在定义 workspace 和 notify operation 后包含 dispatch。
保留同一 CUDA 编译单元，可以沿用现有匿名 namespace、模板实例化和设备内联方式。
此次划分用于改善代码导航和维护，增量编译粒度仍是原来的两个 `.cu`。

修改执行模块时，应同时检查 kernel 参数、stream/event 依赖及完成后的 slot 回收；
修改 dispatch 时，应同时检查 materialization、consumer 获取与释放、错误后的回收。
修改验证路径时，保持生产环境能力探测可用，并运行 native CUDA 测试。
结构调整的验证包括 native 构建、公开接口检查、BF16/FP8 前后向与图生命周期回归，
以及 kernel 指令/资源用量和成对性能对照。
