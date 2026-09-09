# WeftMoE autotune 配置

这四份 JSON 只描述超参数候选。模型由 Python/CLI 传入，拓扑由调用方显式指定，系统环境可使用独立的 `--system-config`。不要把 hidden、intermediate、experts、top-k、tokens 或网络接口写进搜索空间。

## 默认搜索空间

每个参数是非空数组：第一项共同组成 incumbent（基准候选），多值参数按笛卡尔积展开。单值表示保留当前选择，不表示实现只支持这个值。候选在测试前一次性确定，不根据中途结果追加候选。

| 文件 | 组合数 | 多候选参数 |
|---|---:|---|
| [forward_h200_single_node_autotune.json](forward_h200_single_node_autotune.json) | 10 | Relay 8/16/24/32/40；local Return TMA warp 0/8 |
| [forward_h200_multi_node_autotune.json](forward_h200_multi_node_autotune.json) | 10 | Relay 8/16/24/32/40；local Return TMA warp 8/0 |
| [backward_h200_single_node_autotune.json](backward_h200_single_node_autotune.json) | 10 | Relay 8/16/24/32/40；Dispatch batch 16/32 |
| [backward_h200_multi_node_autotune.json](backward_h200_multi_node_autotune.json) | 40 | Relay 8/16/24/32/40；Dispatch batch 64/32；dAB、dX tile N 各 256/128 |

每份默认文件不超过 80 个组合。默认收窄历史收益不明显的 owner group、reduce threads、流水线阶段数等轴，保留参数供定向实验使用。修改多个轴时先计算组合数；阶段数等有配对约束，不能任意独立展开。

## 运行方式

从仓库根目录运行 Ling 单机 EP8（tokens 是每 rank 的输入 token 数）：

```bash
OUT="artifacts/ling_autotune_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
for T in 8192 65536; do
  for phase in forward backward; do
    python3 "tools/autotune_${phase}.py" \
      --nodes 1 --gpus-per-node 8 \
      --config "configs/${phase}_h200_single_node_autotune.json" \
      --hidden 2560 --intermediate 768 --experts 512 --top-k 8 \
      --tokens "$T" --jobs 8 \
      --output "$OUT/${phase}_ep8_t${T}.json"
  done
done
```

加 `--generate-only` 只展开并校验候选，不运行 GPU；`--build-only --candidate incumbent` 只编译指定候选。不要同时启动多组 GPU 性能测试。

多机改用 `--nodes 2 --gpus-per-node 8 --remote root@HOST --port 2222` 和对应 multi_node 文件；也可用 `--ep 16` 代替 `--nodes 2`。不传 `--config` 时按阶段和节点数选默认文件。

`EP = nodes × gpus_per_node`，缺省为单机八卡。配置文件名、`rdma_ctas` 和 remote 地址均不决定 EP。单机展开候选前自动将 `rdma_ctas` 设为 0，不改原文件。当前多机要求八个 GIN owner，`rdma_ctas=8` 与节点数不是同一个概念。计划可以表达更多节点，但当前 GIN 协议和 launcher 实际只支持两机各八卡；更大拓扑仅可生成/编译计划，性能执行会报错。

默认筛选为 10 轮预热 + 20 轮计时；incumbent 和最快 challenger 按正反顺序确认，每次 10 + 50 轮；改善不足 0.5% 时保留 incumbent，最终验证为 10 + 100 轮。以输出 JSON 的实际候选和计时为准。编译通过不代表该配置已完成性能或正确性验证。

## GEMM 参数（1_gemm）

一个 CTA 当前由两个 math warpgroup 和一个 producer warpgroup 组成；一个 warpgroup 有 128 个线程。producer 搬运下一批输入，math 执行 WGMMA。tile 和阶段数共同影响寄存器、共享内存、任务粒度与数据复用。

| 参数 | 阶段 | 作用 |
|---|---|---|
| `tile_m`、`tile_k` | FWD/BWD | 激活 GEMM 的 M tile 和 K 流水线步长，默认 128、64。 |
| `fc1_tile_n`、`fc2_tile_n` | FWD | 两层 GEMM 的 N tile。更宽不必然更快，要考虑模型维度、寄存器与复用。 |
| `dab_tile_n`、`dx_tile_n` | BWD | dAB 前置 GEMM 与 dX GEMM 的 N tile。单机默认均为 256。 |
| `dw_tile_m`、`dw_tile_n` | BWD | 权重梯度 GEMM 的输出 tile，默认 128×256。 |
| `fc1_stages`、`fc2_stages` | FWD | 各 GEMM 的输入流水线缓冲级数。 |
| `dab_stages`、`dx_stages`、`dw1_stages`、`dw2_stages` | BWD | 各 GEMM 的缓冲级数；当前要求 dAB=dX、dW1=dW2。增加阶段会增加共享内存。 |
| `compute_threads`、`math_warpgroups`、`producer_warpgroups` | FWD/BWD | CTA 线程数与角色划分；默认 384、2、1，是当前实现约束，不宜独立修改。 |
| `math_registers`、`producer_registers` | FWD/BWD | 各角色的 `setmaxnreg` 寄存器预算请求。不是实际寄存器使用量，编译器可能忽略请求；须查看编译产物。 |
| `wgmma_wait_group` | FWD | WGMMA wait 时允许尚未完成的提交组数；影响流水线重叠与依赖。 |
| `fc2_store_atom_n` | FWD | FC2 输出 TMA store 的 N 方向分块宽度。 |
| `cluster_size` | FWD | CTA cluster 大小，影响协作调度和资源约束。 |
| `fc2_tma_store` | FWD | 使用共享内存暂存并由 TMA 写出 FC2 结果；当前默认 true。 |

## 通信参数（2_transport）

单机也需要通信 CTA：路由后的激活/梯度仍要在八张卡之间 Dispatch 和 Return。Relay 越多，通信可并行处理的任务越多，同时可用计算资源越少。

| 参数 | 阶段 | 作用 |
|---|---|---|
| `rdma_ctas` | FWD/BWD | 跨节点 GIN 通信角色数量。单机为 0，多机当前为 8。 |
| `relay_ctas` | FWD/BWD | 节点内搬运及相关进度推进的 CTA 数。重点调优轴；仍受对齐、最低数量和 SM 驻留限制，不是任意整数。 |
| `rdma_control_warps` | FWD | RDMA CTA 中负责控制/推进的 warp 数，其余 warp 负责数据打包等工作。 |
| `local_dispatch_tma_warps` | FWD | 每个 Relay CTA 用于节点内 Dispatch 的 TMA 复制 warp 数；单机还须为 ready 发布保留 warp。 |
| `local_return_tma_warps` | FWD | 节点内 Return 的 TMA 复制 warp 数；0 使用向量复制。入口兼容旧名 `relay_tma_warps`，同一配置不能同时给两个名字；内部缓存键暂保留旧名。 |
| `relay_scatter_warps` | FWD | Relay scatter 搬运使用的 warp 数。 |
| `local_return_batch_mblocks` | FWD | 节点内 Return 每批处理的 M-block 数。增大可摊薄领取开销，但会延后其它任务。 |
| `dispatch_publish_batch_rows` | FWD | Dispatch 到达状态分批发布的行数，平衡原子操作开销与计算启动延迟。 |
| `return_chunk_rows` | FWD | 跨节点 Return 的 chunk 行数，影响发送/完成跟踪粒度。 |
| `return_max_inflight` | FWD | Return 的在途窗口上限，平衡发送并行度与资源占用；0 关闭该窗口限制。 |
| `registered_output` | FWD | 将 FC2 输出放在 GIN 注册窗口中，便于直接发送；GIN 的 FC2 TMA store 要求 true。 |
| `relay_local_return` | FWD | 由 Relay 处理节点内 Return；当前支持路径要求 true，不作为性能搜索轴。 |
| `dispatch_batch_rows` | BWD | Relay 每次领取的路由行数；摊薄领取/发布开销，但大 batch 会延后 Return 等任务的服务。 |
| `gin_max_inflight` | BWD | GIN 传输的在途请求上限，仅影响跨节点路径。 |
| `setup_threads` | BWD | 初始化/准备 kernel 的线程数。 |
| `setup_idle_nanosleep` | BWD | 准备阶段等待时的退避参数。 |

## 调度参数（3_scheduler）

| 参数 | 阶段 | 作用 |
|---|---|---|
| `local_route_batch_rows` | FWD | 节点内 Dispatch 路由工作分配的批大小。 |
| `startup_mblocks` | FWD | 每个 expert 最前面直接启动的 M-block 数，降低首批计算等待。 |
| `fc1_locality_group` | FWD | 后续相邻 M-block 的 FC1 发布分组大小，让相同 N tile 接近执行以复用 W1/L2；更大也可能造成 FC2/Return 突发。 |
| `fc1_locality_group4_begin`、`fc1_locality_group4_end` | FWD | 当 locality_group=2 时，在 expert 内 M-block 索引的 `[begin,end)` 中间区间改用四块分组。默认 `[4,12)`；不是 expert ID 或 GPU 分组。首批仍由 startup_mblocks 控制。 |
| `fc2_priority_burst` | FWD | 优先连续领取 FC2 的次数限制；0 不限制该连续优先策略。 |
| `task_claim_chunk` | FWD | 每次领取的计算任务数，权衡原子领取开销与负载均衡。 |
| `task_idle_nanosleep`、`transport_idle_nanosleep`、`setup_nanosleep` | FWD | 计算、通信、准备阶段空闲轮询时的退避参数。 |
| `queue_idle_nanosleep` | BWD | 计算 ready 队列无任务时的退避。 |
| `ready_word_stagger` | BWD | 非零时让不同 CTA 从不同 bitmap word 开始扫描；分散竞争，但可能降低相邻 expert 任务的 L2 复用。 |
| `activation_owner_group` | BWD | dAB/dX 在相邻 ready owner 的小组内分散领取；owner 对应 M-block，不是通信 owner。默认 4，兼顾竞争和权重复用。 |
| `epilogue_rows_per_task` | BWD | 每个激活 epilogue 任务处理的行数，影响非 GEMM 计算粒度。 |
| `dw2_claim_tiles`、`dw1_claim_tiles` | BWD | dW 领取/执行的分块粒度；更大摊薄调度开销，但会影响其它任务的及时推进。 |

所有 nanosleep 参数都是 GPU 退避提示，不是精确延迟保证。调小可能改善响应，也可能增加轮询竞争；一般最后再调。

## 完成与规约参数（4_completion_reduce）

FWD 的 Return 将每个 top-k route 的结果放入源 token 的固定位置。数据可见后发布到达状态；token 的候选全部就绪后进入 reduce ready queue。空闲 CTA 可领取 token，按固定 top-k 顺序累加，保持规约顺序。跨节点接收端还要先检查 chunk 的 fragment 完成标记，再发布对应 route 的到达。完成检查和规约是不同的工作。

| 参数 | 阶段 | 作用 |
|---|---|---|
| `completion_chunks_per_pass` | FWD | 每轮进度推进尝试处理的 Return completion chunk 数；过大可能挤占其它通信任务。 |
| `completion_marker_warps` | FWD | 并行检查 fragment 完成标记的 warp 数。 |
| `reduce_scheduler` | FWD | 规约就绪任务的发现/调度方式；默认 `queue` 使用 ready queue。 |
| `reduce_warps` | FWD | CTA 执行规约时参与的 warp 数。 |
| `reduce_tokens_per_claim` | FWD | 每次领取的 token 数，不超过 reduce_warps；增大可减少领取次数。 |
| `reduce_registers` | FWD | CTA 转入规约角色时的寄存器预算请求。 |
| `return_idle_nanosleep`、`reduce_queue_nanosleep`、`finalize_nanosleep` | FWD | Return 等待、reduce 队列等待、最终完成等待时的退避。 |
| `reduce_threads` | BWD | dX reduce kernel 的 CTA 线程数，当前每个 warp 处理一个 token。 |
| `reduce_idle_nanosleep` | BWD | 等待 dX route 到达时的退避。 |

BWD 当前将 dX reduce 排在 persistent transport 之后，利用释放的 SM，并与尚未完成的 dW 重叠。它与 FWD 的空闲 CTA ready queue 机制不同。增大 reduce_threads 不等于增加主计算资源。

## 系统配置不属于搜索空间

可选系统 JSON 支持 `ep`、`nodes`、`gpus_per_node`、`remote`、`port`、`socket_ifname`、`disable_ib`（单机）、`enable_ib`（多机）、`nccl_lib_dir`、`runtime_library_path` 和 `jit` 等字段。CLI 同名值覆盖系统文件。

- `socket_ifname`：NCCL socket/bootstrap 使用的网卡接口选择，不是 IB HCA/端口分流设置。
- `disable_ib` / `enable_ib`：控制 launcher 的 NCCL IB 开关；通常采用按拓扑生成的默认值。
- `nccl_lib_dir`：指定 NCCL 库目录；缺省会检测仓库私有库，不要求手填。
- `runtime_library_path`：运行时动态库搜索路径覆盖，用于让 CUDA compat、NCCL、Python CUDA runtime 等加载正确版本；不是超参数。
- `jit`：编译工具、架构、并行度和缓存路径；`--jobs`、`--cache-root` 可直接指定。

## 已测经验：Ling 单机需要覆盖更大的 Relay

2026-09-09，单机八张 H200（驱动显示 L20X），EP8，H=2560、I=768、experts=512、top-k=8。使用 math-only accumulator 修复后的 BWD，固定 dAB/dX N=256、dW M/N=128/256、dispatch batch=16。每组正反顺序各一次，每次 10 轮预热 + 50 轮计时；表格为最慢 rank 耗时的均值。

| Relay CTAs | 8K tokens/rank | 64K tokens/rank |
|---:|---:|---:|
| 8 | 12.220 ms | 98.130 ms |
| 16 | 6.759 ms | 56.719 ms |
| 24 | 5.323 ms | **54.574 ms** |
| 32 | 5.028 ms | 56.844 ms |
| 40 | **5.008 ms** | 59.923 ms |

8K 可从 Relay=40 开始（32 仅慢约 0.4%）；64K 从 24 开始。不要把它们视为其它模型、tokens 或 EP16 的已验证最优值。本次只扫描 Relay，未重新搜索 Dispatch batch。

8K 的 Dispatch 跨度随 Relay 从 8 到 40，由 11.30 ms 缩短到 3.46 ms，数据供给明显改善。64K 中 Relay 超过 24 后，Dispatch 继续缩短但总时间增加，说明必须平衡通信与计算资源。单机不等于无需通信；默认候选应覆盖 32、40 等较大的 Relay。

旧 BWD 的 128/128 激活 tile 搭配 dW N256 曾发生严重寄存器溢出，math-only 作用域修复消除了本次所测编译产物的 LDL/STL。修复前的计算瓶颈掩盖了 Relay 调优收益，不能继续沿用当时选出的 Relay=8。当前单机先保留 256/256，避免同时展开过多低收益轴；这不表示 128 永远更差。

比较时看端到端时间。`compute_us` 含等待及尾部 worker 的时间；dAB/dX/dW 是首末任务跨度，彼此重叠，不能相加或当成纯 GEMM 时间。更换内核、编译器或模型后应重新 autotune。结果和日志保存在被 Git 忽略的 `artifacts/` 或 `benchmarks/results/`，不提交中间产物。
