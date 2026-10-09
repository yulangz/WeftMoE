# PF 去 GEMM 通信带宽：2026-10-09

结论：**裸 GIN 数据面接近线速，但当前完整 ring 数据流没有打满。**
保留真实协议、用可校验拷贝替代 FC1/FC2 后，token Get 的 8K/20K 为
**27.36 / 31.35 GB/s/卡/方向**；pack Get 为 **28.78 / 32.21**，约为标称
带宽的 **58% / 64%**。不能把完整 PF 的剩余差距全部归于 GEMM。

本轮只测量，不修改生产 PF。诊断实现保存在独立分支
`experiment/pf-comm-only-20261009`，提交 `43dcd97`；从已验证的部分专用 warp
分支 `82e502f` 派生。原始日志位于该 worktree 的 `build/comm_only/`。

## 硬件与口径

- 两机各 8 张 L20X，全部 SM 固定 1620 MHz。四轮实际协议正式运行的 172 次
  节点时钟观测全部成功，所有 GPU 的 SM 都是 1620；未锁定显存时钟。
- 每张 GPU 对应一组 `mlx5_bond_*`，`/proc/net/bonding/bond*` 确认每组为两条
  active、全双工 200000 Mbps 链路。**不能把 ibstat 的 Rate:200 当作整组上限。**
  每 GPU/rail 每方向标称 50 GB/s，每节点每方向 400 GB/s；双向相加才是
  100 GB/s/卡、800 GB/s/节点。下表从不将 TX+RX 相加后除以单向上限。
- 同时双向传输，GB 为十进制；报告有效 payload 带宽，不是包含包头、请求、
  ACK 的 wire 字节速率。主机初始化、分配、注册、权重准备均在计时外。
- 完整协议为 BF16 H4096/I2048/E256/topK6、EP16；每 rank tokens 在 8192/20480
  附近正态波动 10%，每步随机路由，bias std0.5，seed123。所有四轮使用完整
  相同 workload。每轮 prime32/warmup8/measure24，token/pack 交错跑两轮。

## 裸 GIN 数据面上限

复用既有 `gin_pack.cu`，增加只测指定 packet 的 bandwidth suite。所有输入
随机置换，每次 token 数正态波动 10%；每项 warmup5/measure20，正反顺序重复
两遍。以下数字为两遍 p50 的均值，全部正式计时开始前已结束 CPU 编译。

| 对应操作 | owner 数 | 每次 RDMA 行数 | 8K GB/s/卡/方向 | 20K GB/s/卡/方向 |
|---|---:|---:|---:|---:|
| token Get | 16 | 1 | 46.69 | 47.54 |
| 预 pack 后 packet Get | 16 | 32 | 47.44 | 48.15 |
| 连续 Return Put | 8 | 128 | 45.51 | 46.70 |

Get 每 32 行记录一个完成句柄，Return 每 128 行记录一个，均保留四个 pending
请求组，不是逐 token 等待完成。对应标称带宽利用率为约 91%–96%；例如
20K pack Get 达 **385.2 GB/s/节点/方向**，Return Put 达 **373.6**。

这些是 `network_us`：不计提前 pack；源/目的使用整批不可覆盖区域，仅约束
请求窗口，不包含真实 PF 的消费 credit、机内 fanout 或最终 reduce。Return
对应 FC2 已直接写成连续 TX 的状态，不应额外加上微基准人为 gather 的时间。
每个样本在计时外全量校验 payload；三组共 **9600 次 rank/sample 校验通过**。

## 真实 ring 路径：去 GEMM，不短路同步

保持 D32/R128、B4096、credit2、pending4、16 Get / 8 Put / 4 progress /
1 admission owner，以及原始 384-thread / REG168 / shared-memory 预留。
保留 planner、节点去重、跨机 Get、机内 pull/fanout、X/S 生命周期、Return TX
写入、本地 NIC completion、远端 credit 和 publication。FC1 变为 X→S 的
前 I 列拷贝，FC2 变为 S→H 的重复拷贝；无权重加载、SwiGLU 或 WGMMA。
SASS 中该 network entry 的 HGMMA/HMMA/WGMMA 均为零。

因此它包含拷贝替身及静态 tile 调度成本，**不是仅 RDMA 的带宽**。没有用
空操作直接推进完成计数，也没有把有限 ring 改为无限缓冲。

表中取两次独立运行 p50 的均值。每个样本先取最慢 rank 的通信阶段时间，
再按实际 bytes/time 计算带宽，最后取 p50；不是 p50(bytes)/p50(time)。

| 方案 | tokens/rank | 通信阶段 ms | GB/s/卡/方向 | GB/s/节点/方向 | 标称利用率 |
|---|---:|---:|---:|---:|---:|
| token Get + Q128 Return | 8K | 9.860 | 27.36 | 218.88 | 54.7% |
| token Get + Q128 Return | 20K | 21.452 | 31.35 | 250.80 | 62.7% |
| pack Get + Q128 Return | 8K | 9.406 | 28.78 | 230.26 | 57.6% |
| pack Get + Q128 Return | 20K | 20.924 | 32.21 | 257.71 | 64.4% |

通信计时从 planner 完成到所有本地/网络写者汇合完成，包含 ring 等待、机内
搬运和拷贝替身；不包含前面的 pack/notify，也不包含后面的最终 reduce。
不能把这张表的 pack Get 带宽称为“包含 pack 的端到端带宽”。完整计时如下：

| 方案 | tokens/rank | prepare ms | 通信阶段 ms | 完整 kernel ms |
|---|---:|---:|---:|---:|
| token Get | 8K | 0.960 | 9.860 | 11.094 |
| token Get | 20K | 1.123 | 21.452 | 23.209 |
| pack Get | 8K | 1.051 | 9.406 | 10.741 |
| pack Get | 20K | 1.347 | 20.924 | 22.947 |

完整 kernel 包含 prepare、通信、最终 reduce 和跨调用复用 rendezvous。
各列独立取 rank-max/p50，不能直接相加。包括 pack 的完整耗时下降约
3.2% / 1.1%，不是通信阶段带宽增幅那么大。

计数：Get 在接收 rank 计数，Return Put 在发送 rank 计数；全 rank 相加后，
每次真实跨机 payload 恰好计一次。平均每节点单向带宽为：

```text
(sum_all_ranks(dispatch_bytes + return_bytes) / 2) / max_rank(communication_time)
```

再除以每节点 8 卡得到每卡平均；JSON 另存 A→B、B→A 独立统计，避免路由
不均衡被平均值掩盖。统计的是通信数据量，不是 GEMM 的 expanded/padded rows。
每个正式样本的全局 Get bytes、Return bytes 均逐项等于随机路由计算的精确
节点去重 token 数、远端 route 数乘 8192 bytes。

## 判断与边界

1. 该硬件和 GIN API 能接近线速；当前方案没有充分持续利用它。
2. 去掉 GEMM 后仍有明显差距，剩余包含窗口准入、任务/ready 检查、credit、
   机内 fanout、实际 payload 读写及拷贝替身。**本测量不能单独证明 credit
   是瓶颈，也不能把整个差距归于静态调度。**
3. pack 能减少 Get 请求数，但当前完整路径收益较小；仅增加 pack 不是补齐
   36%–45% 未利用带宽的充分条件。
4. 不应把裸 GIN 测到的带宽当作完整 PF 已达到的带宽，也不把此诊断替身
   的时间报告为真实 inference 性能。

## 校验与复现

- token Get：64/8K/20K，各 2 次变长随机调用，16 rank 全部通过。
- pack Get：B128、route bias8，8K/20K 各 3 次，验证有限 ring 多次回绕与
  跨调用复用。上述共 192 次 rank/sample 输出检查，max absolute error=0。
- 完整协议四次正式运行 exit0；源代码、构建与启动命令见诊断分支的
  `docs/communication_only_20261009.md`。生产 PF、DT、训练/BWD 未改动。
- 诊断库 SHA256：`15b3557467085863d815f5b943796fe1659aae922160cb865d8e4d69b409aef6`。
- 裸 GIN binary SHA256：`16a46cd66899f6d9f00a8f19a1b1a933af08aad007963953321c866695ce0627`。
- [冻结结果](../benchmarks/communication_bandwidth_20261009.json) 保存逐样本
  bytes/time、workload、manifest、clock 和日志 hash；
  [汇总脚本](../benchmarks/summarize_communication.py) 检查数据量并复算带宽。

裸 GIN 构建沿用 `docs/static_forward_v4_multinode_record.md` 的命令，binary
命名为 `build/gin_bandwidth` 并同步到第二节点。下面测 token Get/Put；将
packet 改为 32 即测 pack32，将 contexts/batch/packet 改为 8/512/128 即测
Return 对应粒度。原始正式日志是 `build/gin_bandwidth_final_{token,pack32,return128}.log`。

```bash
env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 \
  NCCL_IB_GID_INDEX=3 WEFTMOE_GIN_BENCH_MODE=bandwidth \
  WEFTMOE_GIN_BENCH_CONTEXTS=16 WEFTMOE_GIN_BENCH_BATCH=128 \
  WEFTMOE_GIN_BENCH_PACKET=1 \
  /opt/.venv/bin/python benchmarks/run_gin_pack.py \
  --binary "$PWD/build/gin_bandwidth" --remote-binary "$PWD/build/gin_bandwidth" \
  --remote root@10.10.19.233 --port 2222 --gpus-per-node 8 \
  --socket-ifname eth0 --enable-ib \
  --nccl-lib-dir /usr/local/cuda/compat:/usr/local/cuda/lib64 --timeout 240
```
